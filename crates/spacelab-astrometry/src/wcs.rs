//! The world-coordinate system: a gnomonic (TAN) projection with a polynomial distortion, and the
//! least-squares fit that determines it from matched stars.
//!
//! # Why the fit is linear
//!
//! A plate solution maps pixels to directions, and that map is not linear — a gnomonic projection is
//! a ratio of trigonometric functions. But it factors: **sky → tangent plane** is fixed geometry once
//! a tangent point is chosen, and **tangent plane → pixel** is where the telescope's unknowns live.
//! So the fit is done in the intermediate plane, where
//!
//! ```text
//! ξ = Σ a_pq · uᵖ vᵠ,     η = Σ b_pq · uᵖ vᵠ,     u = x − x₀,  v = y − y₀
//! ```
//!
//! is *linear in its coefficients* however high the order goes. Two independent normal-equation
//! solves, no iteration, no starting guess, and a covariance that means something. An optimiser over
//! `(α₀, δ₀, scale, rotation, distortion)` would be solving a problem that has a closed form.
//!
//! The price is that the tangent point is not fitted. It does not need to be: moving it is absorbed
//! by the linear coefficients to first order, and what is left is a second-order term far below the
//! distortion the polynomial is already carrying. What the tangent point does have to be is *close*
//! — the projection diverges 90° away — and [`PlateSolution::fit`] is given one, normally the
//! catalogue direction nearest the frame centre from the blind solve in [`crate::quads`].
//!
//! # Conventions
//!
//! Angles are radians throughout, including the tangent-plane coordinates `ξ` and `η`, which are the
//! FITS *intermediate world coordinates* in radians rather than the degrees a FITS header stores.
//! Pixel coordinates are the detector's own, in pixels, with whatever origin the caller's frame uses
//! — the fit only ever sees differences from `crpix`. `ξ` increases with right ascension and `η`
//! with declination, the standard orientation; a detector that flips is described by the sign of the
//! fitted coefficients and needs no flag.
//!
//! FITS WCS is Greisen & Calabretta (2002), A&A **395**, 1061 and Calabretta & Greisen (2002), A&A
//! **395**, 1077; the distortion convention is SIP, Shupe et al. (2005), ASP Conf. Ser. **347**.

use alloc::vec;
use alloc::vec::Vec;

use spacelab_core::Vec3;

/// A measured position on the detector, pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pixel {
    /// Column.
    pub x: f64,
    /// Row.
    pub y: f64,
}

impl Pixel {
    /// Construct.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Number of monomials `uᵖvᵠ` with `p + q ≤ order`.
#[must_use]
pub const fn term_count(order: usize) -> usize {
    (order + 1) * (order + 2) / 2
}

/// The monomials of `(u, v)` up to `order`, graded by total degree and with `u` first inside each
/// degree: `1, u, v, u², uv, v², u³, u²v, …`. The first three are the plate constants a pure TAN
/// solution needs; everything after them is distortion. Indices `1` and `2` are the linear terms,
/// which is what [`TanSip::sky_to_pixel`] and [`TanSip::plate_scale`] read as the Jacobian.
fn monomials(u: f64, v: f64, order: usize, out: &mut [f64]) {
    let mut index = 0;
    for degree in 0..=order {
        for p in (0..=degree).rev() {
            out[index] = power(u, p) * power(v, degree - p);
            index += 1;
        }
    }
    debug_assert_eq!(index, term_count(order));
}

/// `base^exponent` by repeated multiplication — exact for the small integer powers a plate uses,
/// where `libm::pow` would round.
fn power(base: f64, exponent: usize) -> f64 {
    let mut result = 1.0;
    for _ in 0..exponent {
        result *= base;
    }
    result
}

/// Deproject a tangent-plane offset onto the sky: the gnomonic inverse about `(ra0, dec0)`.
///
/// `δ = asin((sin δ₀ + η cos δ₀)/√(1+ξ²+η²))`, `α = α₀ + atan2(ξ, cos δ₀ − η sin δ₀)`.
#[must_use]
pub fn deproject(xi: f64, eta: f64, ra0: f64, dec0: f64) -> (f64, f64) {
    let (sin_dec0, cos_dec0) = (libm::sin(dec0), libm::cos(dec0));
    let denominator = libm::sqrt(1.0 + xi * xi + eta * eta);
    let dec = libm::asin(eta.mul_add(cos_dec0, sin_dec0) / denominator);
    let ra = ra0 + libm::atan2(xi, eta.mul_add(-sin_dec0, cos_dec0));
    (ra, dec)
}

/// Project a direction onto the tangent plane at `(ra0, dec0)`, returning `(ξ, η)` in radians.
///
/// `None` on the far hemisphere, where the gnomonic projection has no image: the denominator is the
/// cosine of the angle from the tangent point, and a star 90° away projects to infinity. Rejecting
/// it is the difference between a solution that fails and one that fits a hyperbola.
#[must_use]
pub fn project(ra: f64, dec: f64, ra0: f64, dec0: f64) -> Option<(f64, f64)> {
    let (sin_dec, cos_dec) = (libm::sin(dec), libm::cos(dec));
    let (sin_dec0, cos_dec0) = (libm::sin(dec0), libm::cos(dec0));
    let delta_ra = ra - ra0;
    let (sin_dra, cos_dra) = (libm::sin(delta_ra), libm::cos(delta_ra));
    let denominator = sin_dec * sin_dec0 + cos_dec * cos_dec0 * cos_dra;
    if denominator <= 1e-9 {
        return None;
    }
    Some((
        cos_dec * sin_dra / denominator,
        (sin_dec * cos_dec0 - cos_dec * sin_dec0 * cos_dra) / denominator,
    ))
}

/// A fitted world-coordinate system: a tangent point, a reference pixel, and the two polynomials
/// that carry the plate from pixels to the tangent plane.
#[derive(Debug, Clone, PartialEq)]
pub struct TanSip {
    /// Tangent point right ascension, radians.
    pub ra0: f64,
    /// Tangent point declination, radians.
    pub dec0: f64,
    /// Reference pixel — the polynomials are in `(x − crpix.x, y − crpix.y)`.
    pub crpix: Pixel,
    /// Total polynomial order; `1` is a pure TAN plate with six constants and no distortion.
    pub order: usize,
    /// `ξ` coefficients, in `monomials` order.
    pub xi: Vec<f64>,
    /// `η` coefficients, in `monomials` order.
    pub eta: Vec<f64>,
}

impl TanSip {
    /// Tangent-plane coordinates of a pixel, radians.
    #[must_use]
    pub fn intermediate(&self, pixel: Pixel) -> (f64, f64) {
        let mut terms = vec![0.0; term_count(self.order)];
        monomials(
            pixel.x - self.crpix.x,
            pixel.y - self.crpix.y,
            self.order,
            &mut terms,
        );
        let combine = |coefficients: &[f64]| -> f64 {
            coefficients
                .iter()
                .zip(&terms)
                .map(|(c, t)| c * t)
                .sum::<f64>()
        };
        (combine(&self.xi), combine(&self.eta))
    }

    /// Sky position of a pixel: `(right ascension, declination)` in radians, right ascension wrapped
    /// into `[0, 2π)`.
    #[must_use]
    pub fn pixel_to_sky(&self, pixel: Pixel) -> (f64, f64) {
        let (xi, eta) = self.intermediate(pixel);
        let (ra, dec) = deproject(xi, eta, self.ra0, self.dec0);
        (wrap_two_pi(ra), dec)
    }

    /// Unit direction of a pixel in the catalogue frame.
    #[must_use]
    pub fn pixel_to_direction(&self, pixel: Pixel) -> Vec3 {
        let (ra, dec) = self.pixel_to_sky(pixel);
        let cos_dec = libm::cos(dec);
        Vec3::new(
            cos_dec * libm::cos(ra),
            cos_dec * libm::sin(ra),
            libm::sin(dec),
        )
    }

    /// The pixel a sky position lands on, by Newton iteration on the forward polynomial.
    ///
    /// `None` when the direction is on the far hemisphere, when the linear part is singular — a
    /// degenerate plate, which a fit from collinear stars produces — or when the iteration has not
    /// converged to a hundredth of a pixel in eight steps. Inverting numerically rather than fitting
    /// a second polynomial (SIP's `AP`/`BP`) keeps one description of the plate instead of two that
    /// can disagree, at the cost of a few evaluations per call.
    #[must_use]
    pub fn sky_to_pixel(&self, ra: f64, dec: f64) -> Option<Pixel> {
        let (xi, eta) = project(ra, dec, self.ra0, self.dec0)?;
        // Linear part: [ξ; η] ≈ [a0; b0] + J·[u; v].
        let (a1, a2) = (self.xi[1], self.xi[2]);
        let (b1, b2) = (self.eta[1], self.eta[2]);
        let determinant = a1.mul_add(b2, -(a2 * b1));
        if determinant == 0.0 || !determinant.is_finite() {
            return None;
        }
        let solve = |dxi: f64, deta: f64| {
            (
                b2.mul_add(dxi, -(a2 * deta)) / determinant,
                a1.mul_add(deta, -(b1 * dxi)) / determinant,
            )
        };
        let (mut u, mut v) = solve(xi - self.xi[0], eta - self.eta[0]);
        for _ in 0..8 {
            let guess = Pixel::new(u + self.crpix.x, v + self.crpix.y);
            let (got_xi, got_eta) = self.intermediate(guess);
            let (du, dv) = solve(xi - got_xi, eta - got_eta);
            u += du;
            v += dv;
            if libm::fabs(du) < 1e-2 && libm::fabs(dv) < 1e-2 {
                return Some(Pixel::new(u + self.crpix.x, v + self.crpix.y));
            }
        }
        None
    }

    /// Plate scale at the reference pixel, radians per pixel — the square root of the Jacobian's
    /// determinant, which is the scale of the area element and so is invariant to how the detector
    /// is rotated or flipped.
    #[must_use]
    pub fn plate_scale(&self) -> f64 {
        let (a1, a2) = (self.xi[1], self.xi[2]);
        let (b1, b2) = (self.eta[1], self.eta[2]);
        libm::sqrt(libm::fabs(a1.mul_add(b2, -(a2 * b1))))
    }
}

fn wrap_two_pi(angle: f64) -> f64 {
    let turns = core::f64::consts::TAU;
    let wrapped = libm::fmod(angle, turns);
    if wrapped < 0.0 {
        wrapped + turns
    } else {
        wrapped
    }
}

/// One matched pair: where the source was measured, and where the catalogue says it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correspondence {
    /// Measured detector position.
    pub pixel: Pixel,
    /// Catalogue right ascension at the observation epoch, radians.
    pub right_ascension: f64,
    /// Catalogue declination at the observation epoch, radians.
    pub declination: f64,
}

/// A fitted plate: the world-coordinate system, what each star contributed to it, and how well the
/// coefficients are known.
#[derive(Debug, Clone, PartialEq)]
pub struct PlateSolution {
    /// The fitted mapping.
    pub wcs: TanSip,
    /// Per-star tangent-plane residual `(ξ_measured − ξ_fitted, η_…)`, radians, in the order the
    /// correspondences were given. Publishing these rather than a single number is the point: a
    /// solve with a good RMS and one 5σ star is a mismatched star, not a good solve.
    pub residuals: Vec<(f64, f64)>,
    /// Root-mean-square residual over both coordinates, radians.
    pub rms: f64,
    /// `(AᵀA)⁻¹` for the shared design matrix, row-major `m × m` with `m = term_count(order)`.
    /// Multiply by [`Self::variance_xi`] or [`Self::variance_eta`] for the covariance of the
    /// corresponding coefficient vector.
    pub normal_inverse: Vec<f64>,
    /// Residual variance of `ξ`, rad², with the `n − m` degrees of freedom the fit spent.
    pub variance_xi: f64,
    /// Residual variance of `η`, rad².
    pub variance_eta: f64,
}

impl PlateSolution {
    /// Fit a plate of total order `order` to `correspondences`, about the tangent point
    /// `(ra0, dec0)` and the reference pixel `crpix`.
    ///
    /// `None` when there are fewer stars than coefficients, when a star falls on the far hemisphere,
    /// or when the normal matrix is not positive definite — which is what a degenerate geometry
    /// looks like arithmetically: stars along a line cannot determine a rotation, and no amount of
    /// regularisation would make the answer meaningful.
    ///
    /// Conditioning: the monomials are formed in raw pixel offsets, so an order-3 fit over a
    /// 4096-pixel frame spans `u³ ≈ 10¹⁰` against `1`, and the normal matrix's condition number goes
    /// as the square of that. It is why the fit is done in `f64` with a Cholesky that *reports*
    /// failure rather than a solver that always returns something.
    #[must_use]
    pub fn fit(
        correspondences: &[Correspondence],
        ra0: f64,
        dec0: f64,
        crpix: Pixel,
        order: usize,
    ) -> Option<Self> {
        let terms = term_count(order);
        if correspondences.len() < terms {
            return None;
        }

        let mut normal = vec![0.0; terms * terms];
        let mut rhs_xi = vec![0.0; terms];
        let mut rhs_eta = vec![0.0; terms];
        let mut targets = Vec::with_capacity(correspondences.len());
        let mut row = vec![0.0; terms];
        let mut design = Vec::with_capacity(correspondences.len() * terms);

        for correspondence in correspondences {
            let (xi, eta) = project(
                correspondence.right_ascension,
                correspondence.declination,
                ra0,
                dec0,
            )?;
            monomials(
                correspondence.pixel.x - crpix.x,
                correspondence.pixel.y - crpix.y,
                order,
                &mut row,
            );
            for i in 0..terms {
                rhs_xi[i] += row[i] * xi;
                rhs_eta[i] += row[i] * eta;
                for j in 0..terms {
                    normal[i * terms + j] += row[i] * row[j];
                }
            }
            design.extend_from_slice(&row);
            targets.push((xi, eta));
        }

        let factor = cholesky(&mut normal, terms)?;
        let xi_coefficients = cholesky_solve(&factor, terms, &rhs_xi);
        let eta_coefficients = cholesky_solve(&factor, terms, &rhs_eta);

        let mut residuals = Vec::with_capacity(targets.len());
        let mut sum_xi = 0.0;
        let mut sum_eta = 0.0;
        for (index, (xi, eta)) in targets.iter().enumerate() {
            let row = &design[index * terms..(index + 1) * terms];
            let dot = |c: &[f64]| c.iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
            let (dxi, deta) = (xi - dot(&xi_coefficients), eta - dot(&eta_coefficients));
            sum_xi += dxi * dxi;
            sum_eta += deta * deta;
            residuals.push((dxi, deta));
        }

        #[allow(clippy::cast_precision_loss)]
        let freedom = (targets.len().saturating_sub(terms)).max(1) as f64;
        #[allow(clippy::cast_precision_loss)]
        let count = targets.len() as f64;
        Some(Self {
            wcs: TanSip {
                ra0,
                dec0,
                crpix,
                order,
                xi: xi_coefficients,
                eta: eta_coefficients,
            },
            residuals,
            rms: libm::sqrt((sum_xi + sum_eta) / (2.0 * count)),
            normal_inverse: cholesky_inverse(&factor, terms),
            variance_xi: sum_xi / freedom,
            variance_eta: sum_eta / freedom,
        })
    }

    /// Covariance of the tangent-plane position at `pixel`, rad², as
    /// `[[σ²ξ, 0], [0, σ²η]]` — the propagated coefficient uncertainty `mᵀ(AᵀA)⁻¹m·σ²`.
    ///
    /// The off-diagonal is zero because the two coordinates are fitted from the same design matrix
    /// against *independent* residuals: their coefficient estimates are uncorrelated even though
    /// their variances share a shape. A correlated pair of coordinates would need the residuals
    /// themselves to be correlated, which happens when the centroiding is asymmetric — a trailed
    /// image — and is the caller's to add.
    #[must_use]
    pub fn position_covariance(&self, pixel: Pixel) -> [[f64; 2]; 2] {
        let terms = term_count(self.wcs.order);
        let mut row = vec![0.0; terms];
        monomials(
            pixel.x - self.wcs.crpix.x,
            pixel.y - self.wcs.crpix.y,
            self.wcs.order,
            &mut row,
        );
        let mut quadratic = 0.0;
        for i in 0..terms {
            for j in 0..terms {
                quadratic += row[i] * self.normal_inverse[i * terms + j] * row[j];
            }
        }
        [
            [quadratic * self.variance_xi, 0.0],
            [0.0, quadratic * self.variance_eta],
        ]
    }
}

/// Cholesky factor `L` of a symmetric matrix, lower triangle, row-major. `None` if it is not
/// positive definite.
fn cholesky(matrix: &mut [f64], n: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if sum <= 0.0 || sum.is_nan() {
                    return None;
                }
                lower[i * n + i] = libm::sqrt(sum);
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Some(lower)
}

/// Solve `LLᵀx = b` for one right-hand side.
fn cholesky_solve(lower: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut x = rhs.to_vec();
    for i in 0..n {
        for k in 0..i {
            x[i] -= lower[i * n + k] * x[k];
        }
        x[i] /= lower[i * n + i];
    }
    for i in (0..n).rev() {
        for k in (i + 1)..n {
            x[i] -= lower[k * n + i] * x[k];
        }
        x[i] /= lower[i * n + i];
    }
    x
}

/// `(LLᵀ)⁻¹`, row-major, by solving against each unit vector.
fn cholesky_inverse(lower: &[f64], n: usize) -> Vec<f64> {
    let mut inverse = vec![0.0; n * n];
    let mut unit = vec![0.0; n];
    for column in 0..n {
        unit.fill(0.0);
        unit[column] = 1.0;
        let solved = cholesky_solve(lower, n, &unit);
        for (row, value) in solved.iter().enumerate() {
            inverse[row * n + column] = *value;
        }
    }
    inverse
}
