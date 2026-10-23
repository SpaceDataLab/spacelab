//! Conjunction assessment: the probability of collision `Pc` for a short-term encounter, from a
//! relative state and a combined covariance at the time of closest approach.
//!
//! # The 2-D short-encounter model
//!
//! Two objects pass close at TCA with a high relative speed. Over the seconds the encounter lasts the
//! relative motion is effectively rectilinear, the covariance effectively frozen, and the objects
//! spheres of combined radius `R = R₁ + R₂`. Under those three assumptions (Foster & Estes 1992) the
//! 3-D collision integral collapses to a 2-D one: project the relative position and the *combined*
//! relative-position covariance `C = C₁ ⊕ C₂` onto the **encounter B-plane** — the plane perpendicular
//! to the relative velocity — and integrate the resulting 2-D Gaussian over the hard-body disk of
//! radius `R`. Because the hard body is a *circle*, the answer is invariant to how the two in-plane
//! axes are oriented, so [`ConjunctionGeometry::project`] is free to pick any orthonormal basis of the
//! plane.
//!
//! A [`ConjunctionGeometry`] carries the two objects separately (two states are one relative state,
//! two covariances, two radii) and reduces to an [`EncounterPlane`] — miss vector, 2×2 covariance,
//! combined radius. The [`EncounterPlane`] is validated positive-definite *at construction*, so the
//! four probability methods below cannot meet a singular covariance and are therefore infallible.
//!
//! # Three ways to do the 2-D integral
//!
//! The integral `Pc = ∬_{|x|≤R} N(x; m, C) dx` has no elementary closed form; each method is a
//! different quadrature, and they trade accuracy against speed differently as the covariance ellipse
//! becomes elongated (`σ_max/σ_min` large — the usual case, since along-track uncertainty dwarfs
//! radial):
//!
//! * [`probability_foster`](EncounterPlane::probability_foster) — the baseline. In polar coordinates
//!   about the disk centre the radial integral of the Gaussian is *analytic* (an `erf` pair, since the
//!   exponent is quadratic in the radius); only the angular integral is numerical, and it is a smooth
//!   periodic trapezoid. This is the NASA-JSC Foster & Estes 1992 method, and the analytic radial step
//!   keeps it accurate even for very thin ellipses where a raw 2-D grid would miss the ridge.
//! * [`probability_patera`](EncounterPlane::probability_patera) — reduce the *area* integral to a 1-D
//!   *contour* integral around the hard-body circle by Green's theorem (Patera 2005). Whitening the
//!   covariance to the identity turns the circle into an ellipse and the isotropic Gaussian's radial
//!   primitive `(1 − e^{−r²/2})/r²` into the contour weight; that weight is smooth through `r = 0`, so
//!   — unlike the textbook Patera form — no separate case split for the disk centre is needed. Fast and
//!   accurate for high-aspect covariances.
//! * [`probability_chan`](EncounterPlane::probability_chan) — Chan's analytic series (Chan 1997/2008).
//!   Approximate the ellipse by an equal-area circle, reducing `Pc` to the CDF of a non-central
//!   chi-square, evaluated as a convergent series. Closed-form-fast, and accurate for near-circular to
//!   moderate aspect ratios; the equal-area step degrades as the ellipse becomes extreme (documented,
//!   and demonstrated against Patera in the tests).
//!
//! [`probability_max`](EncounterPlane::probability_max) adds the dilution-of-precision safety metric:
//! the maximum `Pc` over an isotropic scaling of the covariance (Alfriend/Frisbee-style), the
//! worst-case probability when the covariance magnitude is itself uncertain.

// Matrix code indexes 2- and 3-vectors by one i/j/k, and the domain symbols (C, R, Pc, σ, u, v) are one
// letter by convention; the quadrature locals (aa, bb, mu) read as the method derivations' notation.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use core::f64::consts::TAU;

/// A close-approach geometry at TCA: two objects, each a relative-frame contribution to the miss, the
/// combined covariance, and the combined hard body. The two covariances must be expressed in one
/// common inertial frame (the caller rotates each object's covariance into it before building this);
/// the relative position and velocity are `object₂ − object₁` in the same frame.
///
/// Holding the two objects separately — rather than a pre-summed covariance and a pre-differenced
/// state — keeps the collision-independence assumption (`C = C₁ + C₂`) explicit and the combined
/// radius (`R = R₁ + R₂`) impossible to forget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConjunctionGeometry {
    /// Relative position `r₂ − r₁` at TCA, metres, in the common inertial frame.
    pub relative_position: [f64; 3],
    /// Relative velocity `v₂ − v₁` at TCA, m/s, in the same frame. Defines the B-plane normal.
    pub relative_velocity: [f64; 3],
    /// Object 1 position covariance, m², in the common frame.
    pub primary_covariance: [[f64; 3]; 3],
    /// Object 2 position covariance, m², in the common frame.
    pub secondary_covariance: [[f64; 3]; 3],
    /// Object 1 hard-body (equivalent-sphere) radius, metres.
    pub primary_radius: f64,
    /// Object 2 hard-body (equivalent-sphere) radius, metres.
    pub secondary_radius: f64,
}

impl ConjunctionGeometry {
    /// Assemble a conjunction from the two objects' relative state, covariances, and radii.
    #[must_use]
    pub const fn new(
        relative_position: [f64; 3],
        relative_velocity: [f64; 3],
        primary_covariance: [[f64; 3]; 3],
        secondary_covariance: [[f64; 3]; 3],
        primary_radius: f64,
        secondary_radius: f64,
    ) -> Self {
        Self {
            relative_position,
            relative_velocity,
            primary_covariance,
            secondary_covariance,
            primary_radius,
            secondary_radius,
        }
    }

    /// Reduce to the 2-D encounter problem: build the B-plane perpendicular to the relative velocity,
    /// project the miss vector and the combined covariance `C₁ + C₂` onto it, and combine the radii.
    ///
    /// # Errors
    ///
    /// * [`ConjunctionError::DegenerateEncounterVelocity`] if the relative velocity is zero or
    ///   non-finite — there is then no encounter plane to project onto.
    /// * [`ConjunctionError::NonPositiveDefiniteCovariance`] if the *projected* covariance is not
    ///   positive-definite (a combined covariance with no spread in some B-plane direction), or
    ///   [`ConjunctionError::InvalidHardBodyRadius`] if a radius is negative or non-finite — both
    ///   surfaced from [`EncounterPlane::new`].
    pub fn project(&self) -> Result<EncounterPlane, ConjunctionError> {
        let v = self.relative_velocity;
        let vmag = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if vmag <= 0.0 || !vmag.is_finite() {
            return Err(ConjunctionError::DegenerateEncounterVelocity);
        }
        let uv = [v[0] / vmag, v[1] / vmag, v[2] / vmag];

        // Any two orthonormal in-plane axes suffice (the disk is rotationally symmetric): cross the
        // velocity with whichever cardinal axis it is least aligned with, so the cross never degenerates.
        let helper = if libm::fabs(uv[0]) < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let e1 = normalize3(cross3(&uv, &helper));
        let e2 = cross3(&uv, &e1); // unit and ⊥ both, since uv ⟂ e1 are orthonormal

        let mut c = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = self.primary_covariance[i][j] + self.secondary_covariance[i][j];
            }
        }

        let r = self.relative_position;
        let miss = [dot3(&e1, &r), dot3(&e2, &r)];
        // Cₚ = Bᵀ C B with B = [e1 e2]; the four scalars eᵢᵀ C eⱼ.
        let cov = [
            [bilinear3(&c, &e1, &e1), bilinear3(&c, &e1, &e2)],
            [bilinear3(&c, &e2, &e1), bilinear3(&c, &e2, &e2)],
        ];
        let radius = self.primary_radius + self.secondary_radius;
        EncounterPlane::new(miss, cov, radius)
    }
}

/// The projected 2-D encounter: the miss vector, the combined positional covariance, and the combined
/// hard-body radius, all in the B-plane. The covariance is guaranteed symmetric positive-definite and
/// the radius non-negative *by construction* — the only ways to build one ([`new`](Self::new),
/// [`from_diagonal`](Self::from_diagonal), [`ConjunctionGeometry::project`]) validate — so the
/// probability methods never meet an illegal state and return a bare `f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EncounterPlane {
    /// Projected miss vector `(x_m, y_m)`, metres.
    miss: [f64; 2],
    /// Projected combined covariance, m², symmetric positive-definite.
    covariance: [[f64; 2]; 2],
    /// Combined hard-body radius `R = R₁ + R₂`, metres, non-negative.
    radius: f64,
}

impl EncounterPlane {
    /// Build from a miss vector, a 2×2 covariance, and the combined radius. The covariance is
    /// symmetrised (its off-diagonal averaged — any asymmetry is rounding, and a covariance is
    /// symmetric by definition) before the positive-definite check.
    ///
    /// # Errors
    ///
    /// [`ConjunctionError::NonPositiveDefiniteCovariance`] if the symmetrised covariance is not
    /// positive-definite, or [`ConjunctionError::InvalidHardBodyRadius`] if `radius` is negative or
    /// non-finite.
    pub fn new(
        miss: [f64; 2],
        covariance: [[f64; 2]; 2],
        radius: f64,
    ) -> Result<Self, ConjunctionError> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(ConjunctionError::InvalidHardBodyRadius);
        }
        let off = f64::midpoint(covariance[0][1], covariance[1][0]);
        let cov = [[covariance[0][0], off], [off, covariance[1][1]]];
        let (lambda_min, _) = eigenvalues_2x2(&cov);
        if lambda_min <= 0.0
            || lambda_min.is_nan()
            || !cov[0][0].is_finite()
            || !cov[1][1].is_finite()
        {
            return Err(ConjunctionError::NonPositiveDefiniteCovariance);
        }
        Ok(Self {
            miss,
            covariance: cov,
            radius,
        })
    }

    /// Build from a miss vector and the covariance in its own principal axes: standard deviations
    /// `σ_x, σ_y` (so the covariance is `diag(σ_x², σ_y²)`). This is the reduced form the classical
    /// test suites tabulate.
    ///
    /// # Errors
    ///
    /// [`ConjunctionError::NonPositiveDefiniteCovariance`] if either sigma is not strictly positive, or
    /// [`ConjunctionError::InvalidHardBodyRadius`] if `radius` is negative or non-finite.
    pub fn from_diagonal(
        miss_x: f64,
        miss_y: f64,
        sigma_x: f64,
        sigma_y: f64,
        radius: f64,
    ) -> Result<Self, ConjunctionError> {
        if sigma_x <= 0.0 || sigma_y <= 0.0 {
            return Err(ConjunctionError::NonPositiveDefiniteCovariance);
        }
        Self::new(
            [miss_x, miss_y],
            [[sigma_x * sigma_x, 0.0], [0.0, sigma_y * sigma_y]],
            radius,
        )
    }

    /// The projected miss vector `(x_m, y_m)`, metres.
    #[must_use]
    pub const fn miss(&self) -> [f64; 2] {
        self.miss
    }

    /// The projected combined covariance, m² (symmetric positive-definite).
    #[must_use]
    pub const fn covariance(&self) -> [[f64; 2]; 2] {
        self.covariance
    }

    /// The combined hard-body radius `R = R₁ + R₂`, metres.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// Collision probability by the Foster & Estes 1992 method: the polar-quadrature baseline, with the
    /// radial integral taken analytically (an `erf` pair per angle) and only the smooth periodic
    /// angular integral done numerically. Accurate across aspect ratios; the reference other methods
    /// are checked against.
    #[must_use]
    pub fn probability_foster(&self) -> f64 {
        if self.radius <= 0.0 {
            return 0.0;
        }
        let (det, inv) = self.inverse();
        let inv_sqrt_det = 1.0 / libm::sqrt(det);
        let m = self.miss;
        let rr = self.radius;
        // cc = mᵀ C⁻¹ m, the whole-plane Mahalanobis² of the miss.
        let cc = bilinear2(&inv, &m, &m);

        let n = self.angular_samples();
        let dphi = TAU / usize_as_f64(n);
        let mut acc = 0.0;
        for k in 0..n {
            let phi = (usize_as_f64(k) + 0.5) * dphi;
            let u = [libm::cos(phi), libm::sin(phi)];
            let aa = bilinear2(&inv, &u, &u); // uᵀC⁻¹u > 0 (C⁻¹ positive-definite)
            let bb = bilinear2(&inv, &u, &m); // uᵀC⁻¹m
            let mu = bb / aa;
            let s = libm::sqrt(aa * 0.5);
            // ∫₀^R ρ exp(−½ aa (ρ−μ)²) dρ in closed form.
            let radial = (libm::exp(-0.5 * aa * mu * mu)
                - libm::exp(-0.5 * aa * (rr - mu) * (rr - mu)))
                / aa
                + mu * 0.5 * libm::sqrt(TAU / aa) * (libm::erf(s * (rr - mu)) + libm::erf(s * mu));
            acc += libm::exp(-0.5 * (cc - bb * bb / aa)) * radial;
        }
        clamp_unit(acc * dphi * inv_sqrt_det / TAU)
    }

    /// Collision probability by Patera 2005: Green's theorem turns the area integral into a contour
    /// integral around the hard-body circle, evaluated in whitened coordinates where the Gaussian is
    /// isotropic. The `(1 − e^{−r²/2})/r²` weight is smooth through the origin, so a single integral
    /// covers the disk-centre-inside and -outside cases alike. Fast for elongated covariances.
    #[must_use]
    pub fn probability_patera(&self) -> f64 {
        if self.radius <= 0.0 {
            return 0.0;
        }
        // Cholesky C = L Lᵀ (L exists: C is positive-definite by construction), whiten by W = L⁻¹.
        let l00 = libm::sqrt(self.covariance[0][0]);
        let l10 = self.covariance[1][0] / l00;
        let l11 = libm::sqrt(self.covariance[1][1] - l10 * l10);
        let w00 = 1.0 / l00;
        let w11 = 1.0 / l11;
        let w10 = -l10 / (l00 * l11);
        // Whitened hard-body-circle centre a = −W m (the Gaussian sits at the origin after whitening).
        let a0 = -(w00 * self.miss[0]);
        let a1 = -(w10 * self.miss[0] + w11 * self.miss[1]);
        let rr = self.radius;

        let n = self.angular_samples();
        let dt = TAU / usize_as_f64(n);
        let mut acc = 0.0;
        for k in 0..n {
            let t = (usize_as_f64(k) + 0.5) * dt;
            let (ct, st) = (libm::cos(t), libm::sin(t));
            let s1 = a0 + rr * w00 * ct;
            let s2 = a1 + rr * (w10 * ct + w11 * st);
            let s1p = -rr * w00 * st;
            let s2p = rr * (-w10 * st + w11 * ct);
            let r2 = s1 * s1 + s2 * s2;
            // h(r) = (1 − e^{−r²/2})/r², with its removable r→0 limit ½ − r²/8 + r⁴/48.
            let h = if r2 < 1e-8 {
                0.5 - r2 / 8.0 + r2 * r2 / 48.0
            } else {
                (1.0 - libm::exp(-0.5 * r2)) / r2
            };
            acc += h * (s1 * s2p - s2 * s1p);
        }
        clamp_unit(acc * dt / TAU)
    }

    /// Collision probability by Chan's analytic series (Chan 1997/2008): the equal-area circle
    /// approximation reduces `Pc` to a non-central chi-square CDF. Evaluated in the cancellation-free
    /// form `Pc = Σ_{i≥1} q_i · P(J ≤ i−1)` with `q ∼ Poisson(u/2)`, `J ∼ Poisson(v/2)`,
    /// `u = R²/√det C`, `v = mᵀC⁻¹m` — all terms are positive, so it stays accurate down to the
    /// `~10⁻²⁸` tails where the textbook `1 − Σ` form loses every digit.
    ///
    /// Fast and accurate for near-circular to moderately elongated covariances; the equal-area step
    /// makes it degrade for extreme aspect ratios, where [`probability_patera`](Self::probability_patera)
    /// or [`probability_foster`](Self::probability_foster) should be preferred.
    #[must_use]
    pub fn probability_chan(&self) -> f64 {
        if self.radius <= 0.0 {
            return 0.0;
        }
        let (det, inv) = self.inverse();
        let u = self.radius * self.radius / libm::sqrt(det);
        let v = bilinear2(&inv, &self.miss, &self.miss);
        let (a, b) = (0.5 * v, 0.5 * u);

        // Pc = P(I > J), I ∼ Poisson(b), J ∼ Poisson(a): Σ_i q_i · P(J ≤ i−1), all terms ≥ 0.
        //
        // Both pmfs are walked in the log domain. The recurrence has to start at i = 0, where the pmf
        // is at its *smallest* — `e^{−b}`, against a mode of order `1/√b` — so the linear form needs a
        // dynamic range of `e^{b}` to compute an answer that never leaves [0, 1]. That range runs out
        // at b ≈ 745: `e^{−746}` is exactly zero, `q` stays zero, and `Pc` comes back 0 instead of ½;
        // in the subnormal band just below, `q₀` carries one or two bits and the error is tens of per
        // cent. `b = R²/(2√det C)` reaches those values for a large hard body against a tight
        // covariance, which is a real conjunction, not a pathological one.
        let ln_b = libm::log(b);
        let ln_a = libm::log(a); // −∞ for a centred conjunction, which is the correct limit
        let mut ln_q = -b; // ln Poisson(b) pmf at 0
        let mut ln_pj = -a; // ln P(J = i−1), advanced each step
        let mut cdf_j = 0.0; // P(J ≤ i−1)
        let mut pc = 0.0;
        let imax = poisson_terms(b);
        for i in 1..=imax {
            let fi = usize_as_f64(i);
            let ln_fi = libm::log(fi);
            ln_q += ln_b - ln_fi; // now ln_q = ln Poisson(b) pmf at i
            cdf_j += libm::exp(ln_pj); // P(J ≤ i−1)
            let term = libm::exp(ln_q) * cdf_j;
            pc += term;
            ln_pj += ln_a - ln_fi; // advance to P(J = i)
            // Past the mode the Poisson tail decays super-exponentially and `cdf_j` is bounded by 1,
            // so a term that cannot move the running sum at f64 resolution bounds every later one.
            if fi > b && term < 1e-18 * pc {
                break;
            }
        }
        clamp_unit(pc)
    }

    /// The maximum collision probability over an isotropic scaling `k·C` of the combined covariance,
    /// `k > 0` — the dilution-of-precision worst case when the covariance *magnitude* is itself
    /// uncertain (Alfriend et al. 1999; Frisbee). A shrinking covariance concentrates the Gaussian off
    /// the hard body and a growing one dilutes it, so for a miss outside the hard body there is an
    /// interior maximum; this returns that peak, found by bracketing then golden-section search on the
    /// accurate Foster probability.
    #[must_use]
    pub fn probability_max(&self) -> f64 {
        let eval = |log_k: f64| -> f64 {
            let k = libm::exp(log_k);
            let scaled = Self {
                miss: self.miss,
                covariance: [
                    [self.covariance[0][0] * k, self.covariance[0][1] * k],
                    [self.covariance[1][0] * k, self.covariance[1][1] * k],
                ],
                radius: self.radius,
            };
            scaled.probability_foster()
        };

        // Coarse scan to bracket the peak over 22 decades of covariance scale.
        let (lo, hi) = (-25.0, 25.0);
        let steps = 48usize;
        let mut best_x = lo;
        let mut best_f = eval(lo);
        for i in 1..=steps {
            let x = lo + (hi - lo) * usize_as_f64(i) / usize_as_f64(steps);
            let f = eval(x);
            if f > best_f {
                best_f = f;
                best_x = x;
            }
        }
        // Golden-section refine on the bracket around the best coarse point.
        let width = (hi - lo) / usize_as_f64(steps);
        let (mut a, mut b) = (best_x - width, best_x + width);
        let gr = 0.5 * (libm::sqrt(5.0) - 1.0);
        let (mut c, mut d) = (b - gr * (b - a), a + gr * (b - a));
        let (mut fc, mut fd) = (eval(c), eval(d));
        for _ in 0..64 {
            if fc > fd {
                b = d;
                d = c;
                fd = fc;
                c = b - gr * (b - a);
                fc = eval(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + gr * (b - a);
                fd = eval(d);
            }
            if libm::fabs(b - a) < 1e-10 {
                break;
            }
        }
        best_f.max(fc).max(fd)
    }

    /// Determinant and inverse of the (positive-definite) 2×2 covariance.
    fn inverse(&self) -> (f64, [[f64; 2]; 2]) {
        let c = &self.covariance;
        let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
        let inv = [
            [c[1][1] / det, -c[0][1] / det],
            [-c[1][0] / det, c[0][0] / det],
        ];
        (det, inv)
    }

    /// Angular sample count for the Foster and Patera quadratures, scaled with the covariance aspect
    /// ratio `σ_max/σ_min = √(λ_max/λ_min)`. A thin ellipse concentrates the integrand into a narrow
    /// angular ridge that needs proportionally more samples; the `16·aspect` law (floored at 1024,
    /// capped at 2¹⁸) holds the relative error at or below `~10⁻¹²` from near-circular to `~5000:1`
    /// (calibrated against a converged reference in the tests).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn angular_samples(&self) -> usize {
        let (lambda_min, lambda_max) = eigenvalues_2x2(&self.covariance);
        let aspect = libm::sqrt(lambda_max / lambda_min);
        (16.0 * aspect).clamp(1024.0, 262_144.0) as usize
    }
}

/// Why a conjunction geometry could not be assessed. Each variant names a specific ill-posed input
/// rather than returning a fabricated probability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjunctionError {
    /// The relative velocity is zero or non-finite, so the encounter B-plane (its normal) is undefined.
    DegenerateEncounterVelocity,
    /// The projected combined covariance is not positive-definite: the uncertainty collapses in some
    /// B-plane direction, and the 2-D Gaussian is degenerate.
    NonPositiveDefiniteCovariance,
    /// The combined hard-body radius is negative or non-finite.
    InvalidHardBodyRadius,
}

impl core::fmt::Display for ConjunctionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DegenerateEncounterVelocity => {
                write!(
                    f,
                    "relative velocity is zero or non-finite; no encounter plane is defined"
                )
            }
            Self::NonPositiveDefiniteCovariance => {
                write!(
                    f,
                    "projected covariance is not positive-definite (degenerate uncertainty)"
                )
            }
            Self::InvalidHardBodyRadius => {
                write!(f, "combined hard-body radius is negative or non-finite")
            }
        }
    }
}

impl core::error::Error for ConjunctionError {}

/// Eigenvalues `(λ_min, λ_max)` of a symmetric 2×2 matrix, closed form. Used for the positive-definite
/// test (`λ_min > 0`) and the aspect ratio; never needs the eigenvectors.
fn eigenvalues_2x2(c: &[[f64; 2]; 2]) -> (f64, f64) {
    let (a, d) = (c[0][0], c[1][1]);
    let b = f64::midpoint(c[0][1], c[1][0]);
    let half_trace = f64::midpoint(a, d);
    let radius = libm::sqrt(0.25 * (a - d) * (a - d) + b * b);
    (half_trace - radius, half_trace + radius)
}

/// Upper bound on the number of `Poisson(mean)` terms carrying non-negligible mass — the mean plus a
/// wide margin in standard deviations `√mean`, capped so a pathological input cannot loop unbounded.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn poisson_terms(mean: f64) -> usize {
    let n = mean + 50.0 * libm::sqrt(mean + 1.0) + 64.0;
    (n as usize).min(1 << 20)
}

/// `uᵀ M v` for a 2×2 `M` and 2-vectors.
fn bilinear2(m: &[[f64; 2]; 2], u: &[f64; 2], v: &[f64; 2]) -> f64 {
    u[0] * (m[0][0] * v[0] + m[0][1] * v[1]) + u[1] * (m[1][0] * v[0] + m[1][1] * v[1])
}

/// `uᵀ M v` for a 3×3 `M` and 3-vectors.
fn bilinear3(m: &[[f64; 3]; 3], u: &[f64; 3], v: &[f64; 3]) -> f64 {
    let mut acc = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            acc += u[i] * m[i][j] * v[j];
        }
    }
    acc
}

/// 3-vector dot product.
fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// 3-vector cross product.
fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Normalise a 3-vector to unit length.
fn normalize3(a: [f64; 3]) -> [f64; 3] {
    let n = libm::sqrt(dot3(&a, &a));
    [a[0] / n, a[1] / n, a[2] / n]
}

/// A probability is in `[0, 1]`; clamp away the rounding dust a finite quadrature can leave just
/// outside. The magnitude is already correct — this only trims a last-bit overshoot.
fn clamp_unit(p: f64) -> f64 {
    p.clamp(0.0, 1.0)
}

/// `usize → f64` for loop indices and counts; the values here are small quadrature counts, exactly
/// representable.
#[allow(clippy::cast_precision_loss)]
fn usize_as_f64(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        // Reference covariances and probabilities are transcribed verbatim from the source CDM files
        // and papers; keep every digit for provenance even though f64 cannot resolve the last few, and
        // do not sprinkle digit separators through primary-source data.
        clippy::excessive_precision,
        clippy::unreadable_literal
    )]
    use super::*;

    /// One NASA CARA reference case: relative position [m], relative velocity [m/s], the two objects'
    /// RTN position covariances [m²], combined hard-body radius [m], reference `Pc`, and a label.
    type CaraCase = (
        [f64; 3],
        [f64; 3],
        [[f64; 3]; 3],
        [[f64; 3]; 3],
        f64,
        f64,
        &'static str,
    );

    /// Independent oracle for Pc (b): a fine Cartesian midpoint sum of the same B-plane Gaussian over
    /// the hard-body disk. Deliberately a *different* quadrature family from Foster (polar) and Patera
    /// (contour) — a Riemann sum on a uniform grid — so agreement is not a shared-bug artefact.
    fn grid_probability(plane: &EncounterPlane, n: usize) -> f64 {
        let (det, inv) = plane.inverse();
        let norm = 1.0 / (TAU * libm::sqrt(det));
        let r = plane.radius();
        let step = 2.0 * r / n as f64;
        let m = plane.miss();
        let mut acc = 0.0;
        for iy in 0..n {
            let y = -r + (iy as f64 + 0.5) * step;
            for ix in 0..n {
                let x = -r + (ix as f64 + 0.5) * step;
                if x * x + y * y <= r * r {
                    let d = [x - m[0], y - m[1]];
                    acc += norm * libm::exp(-0.5 * bilinear2(&inv, &d, &d));
                }
            }
        }
        acc * step * step
    }

    /// Oracle (b) — Foster, Patera, and the independent Cartesian grid must agree across geometries:
    /// circular, moderately elongated, grazing (miss ≈ R), and off-centre. The two analytic
    /// quadratures agree to near machine precision; the grid, resolution-limited, agrees to a few
    /// parts in 10⁵. Any method disagreeing would be a real bug, not a tolerance choice.
    #[test]
    fn foster_patera_and_grid_agree_across_geometries() {
        // (miss_x, miss_y, sigma_x, sigma_y, radius, grid_n, label)
        let cases = [
            (0.0, 0.0, 50.0, 50.0, 20.0, 4000, "circular, centred"),
            (30.0, 40.0, 50.0, 50.0, 20.0, 4000, "circular, off-centre"),
            (10.0, 25.0, 30.0, 90.0, 15.0, 4000, "3:1 elongated"),
            (14.9, 0.0, 20.0, 60.0, 15.0, 5000, "grazing miss ≈ R"),
            (100.0, 0.0, 40.0, 120.0, 10.0, 4000, "far off-centre"),
        ];
        let mut worst_fp = 0.0_f64;
        let mut worst_fg = 0.0_f64;
        for (mx, my, sx, sy, r, n, label) in cases {
            let plane = EncounterPlane::from_diagonal(mx, my, sx, sy, r).unwrap();
            let foster = plane.probability_foster();
            let patera = plane.probability_patera();
            let grid = grid_probability(&plane, n);
            let fp = (foster - patera).abs() / foster;
            let fg = (foster - grid).abs() / foster;
            worst_fp = worst_fp.max(fp);
            worst_fg = worst_fg.max(fg);
            println!(
                "{label:22}: Pc={foster:.6e}  |Foster−Patera|/Pc={fp:.2e}  |Foster−grid|/Pc={fg:.2e}"
            );
            assert!(fp < 1e-9, "{label}: Foster vs Patera {fp:.2e}");
            assert!(fg < 5e-4, "{label}: Foster vs grid {fg:.2e}");
        }
        println!("worst: Foster–Patera {worst_fp:.2e}, Foster–grid {worst_fg:.2e}");
    }

    /// Oracle (a), 2-D — Chan's method reproduces its published reference values. The twelve canonical
    /// Chan cases (Chan 1997/2008, transcribed via Orekit's `Chan1997Test`) plus two Alfano-paper and
    /// two operational (CSM/CDM) cases, spanning `Pc` from `10⁻²` to `10⁻²⁸`. Matching to the four
    /// significant figures the references are quoted at exercises the cancellation-free series exactly
    /// on the deep tails (cases 8, 10, CDM-2) where the naive form fails.
    #[test]
    fn chan_reproduces_published_reference_values() {
        // (miss_x, miss_y, sigma_x, sigma_y, radius, reference Pc, label)
        let cases = [
            (0.0, 10.0, 25.0, 50.0, 5.0, 9.754e-3, "Chan 01"),
            (10.0, 0.0, 25.0, 50.0, 5.0, 9.189e-3, "Chan 02"),
            (0.0, 10.0, 25.0, 75.0, 5.0, 6.586e-3, "Chan 03"),
            (10.0, 0.0, 25.0, 75.0, 5.0, 6.135e-3, "Chan 04"),
            (0.0, 1000.0, 1000.0, 3000.0, 10.0, 1.577e-5, "Chan 05"),
            (1000.0, 0.0, 1000.0, 3000.0, 10.0, 1.011e-5, "Chan 06"),
            (0.0, 10000.0, 1000.0, 3000.0, 10.0, 6.443e-8, "Chan 07"),
            (10000.0, 0.0, 1000.0, 3000.0, 10.0, 3.216e-27, "Chan 08"),
            (0.0, 10000.0, 1000.0, 10000.0, 10.0, 3.033e-6, "Chan 09"),
            (10000.0, 0.0, 1000.0, 10000.0, 10.0, 9.645e-28, "Chan 10"),
            (0.0, 5000.0, 1000.0, 3000.0, 50.0, 1.039e-4, "Chan 11"),
            (5000.0, 0.0, 1000.0, 3000.0, 50.0, 1.556e-9, "Chan 12"),
            (
                84.875546,
                60.583685,
                57.918666,
                152.8814468,
                10.3,
                1.8934e-3,
                "CSM 1",
            ),
            (
                -692.362272,
                4475.456261,
                193.454603,
                562.027293,
                13.2,
                2.2880e-20,
                "CDM 2",
            ),
            (
                -3.8872073,
                0.1591646,
                1.4101830,
                114.2585190,
                15.0,
                3.1264e-2,
                "Alfano 3 (Chan)",
            ),
        ];
        let mut worst = 0.0_f64;
        for (mx, my, sx, sy, r, reference, label) in cases {
            let plane = EncounterPlane::from_diagonal(mx, my, sx, sy, r).unwrap();
            let chan = plane.probability_chan();
            let rel = (chan - reference).abs() / reference;
            worst = worst.max(rel);
            println!("{label:16}: ref={reference:.4e} chan={chan:.4e} rel={rel:.2e}");
            assert!(
                rel < 1e-3,
                "{label}: Chan {chan:.4e} vs ref {reference:.4e} = {rel:.2e}"
            );
        }
        println!("worst Chan relative error vs published (4-sig-fig) references: {worst:.2e}");
    }

    /// Oracle (a), 3-D — the full projection pipeline reproduces the NASA CARA "Alfano" conjunction
    /// test cases. Each case is a real 3-D encounter (relative RTN state, both objects' RTN position
    /// covariances, combined hard-body radius) from the CARA CDM suite, with the published reference
    /// `Pc`. Feeding the raw geometry through [`ConjunctionGeometry::project`] and then Foster/Patera
    /// reproduces every reference to `< 3·10⁻⁴` relative — the residual is the small same-frame
    /// approximation of summing the two objects' RTN covariances (their frames differ by `~10⁻⁷` rad).
    ///
    /// Ten of the twelve canonical cases are covered: CARA case 10 duplicates case 9, and case 12 is
    /// not distributed in the CDM set used here.
    #[test]
    fn alfano_cara_cases_reproduce_published_reference() {
        // relative position [m], relative velocity [m/s], object-1 cov, object-2 cov (RTN, m²),
        // combined HBR [m], reference Pc, label. Covariances laid out [[rr,rt,rn],[rt,tt,tn],[rn,tn,nn]].
        let cases: [CaraCase; 10] = [
            (
                [-0.502165, -0.497806, 4.999966],
                [-0.009963, 0.010037, -0.000001],
                rtn(
                    1.988970273925819e1,
                    -3.524140813027809e2,
                    6.496749385722737e3,
                    0.0,
                    0.0,
                    1.205039522307600e0,
                ),
                rtn(
                    1.989074842319444e1,
                    -3.524310089952797e2,
                    6.496894829253188e3,
                    -9.767348673026151e-21,
                    6.463165376425425e-19,
                    1.204674647143796e0,
                ),
                15.0,
                0.146749549,
                "CARA 01",
            ),
            (
                [-0.502163, -0.497792, 4.999966],
                [-0.009963, 0.010037, -0.000001],
                rtn(
                    1.988970262612387e1,
                    -3.524140803158783e2,
                    6.496749387784603e3,
                    0.0,
                    0.0,
                    1.205039521089100e0,
                ),
                rtn(
                    1.989074830997710e1,
                    -3.524310080076631e2,
                    6.496894831316016e3,
                    -8.716498422838784e-20,
                    9.405936919789482e-19,
                    1.204674645925095e0,
                ),
                4.0,
                0.006222267,
                "CARA 02",
            ),
            (
                [-2.751685, -0.487898, 2.752074],
                [-11.365912, 0.051787, -11.356027],
                rtn(
                    1.988980036134080e1,
                    -3.524149328959712e2,
                    6.496747606851101e3,
                    0.0,
                    0.0,
                    1.205040573210700e0,
                ),
                rtn(
                    1.746930568576392e1,
                    -3.305057350225742e2,
                    6.542324010830698e3,
                    -1.279563505801461e-15,
                    -4.449721840993348e-13,
                    1.177810899317289e0,
                ),
                15.0,
                0.100351176,
                "CARA 03",
            ),
            (
                [-37.720379, -128.952321, -3.757988],
                [0.018266, -0.005344, 0.000250],
                rtn(
                    4.080844572465061e-1,
                    4.674493755532058e1,
                    6.024200059574752e3,
                    0.0,
                    0.0,
                    4.593629039291300e-2,
                ),
                rtn(
                    4.078489163619697e-1,
                    4.672853966111639e1,
                    6.024412056804277e3,
                    8.207065180527589e-18,
                    3.523840694673876e-18,
                    4.596631194107787e-2,
                ),
                15.0,
                0.049323406,
                "CARA 04",
            ),
            (
                [1.008140, -2.117599, 0.706652],
                [-0.300002, -0.001116, 0.424270],
                rtn(
                    4.777224467280041e-1,
                    -8.519384381202406e1,
                    1.580759742365653e4,
                    0.0,
                    0.0,
                    7.845896810140191e-2,
                ),
                rtn(
                    4.710007645702890e-1,
                    -8.457040115987763e1,
                    1.580846029392646e4,
                    -1.748541090564049e-13,
                    -2.121117864684052e-11,
                    7.846753355604119e-2,
                ),
                10.0,
                0.044487386,
                "CARA 05",
            ),
            (
                [-1.023573, -2.110003, -0.706847],
                [-0.100006, 0.001133, 0.141439],
                rtn(
                    4.777224465446061e1,
                    -8.519384379508079e3,
                    1.580759742368094e6,
                    0.0,
                    0.0,
                    3.551023949974670e0,
                ),
                rtn(
                    4.754421287113225e1,
                    -8.498287373777353e3,
                    1.580789400229213e6,
                    5.141187961466009e-10,
                    -6.830025434112486e-9,
                    3.551139141711098e0,
                ),
                10.0,
                0.004335455,
                "CARA 06",
            ),
            (
                [0.440517, -3.134352, 0.340075],
                [0.136067, 0.003780, -0.141425],
                rtn(
                    3.451363822699375e3,
                    -6.727558503785931e5,
                    1.316025642426085e8,
                    0.0,
                    0.0,
                    3.550903989987834e2,
                ),
                rtn(
                    3.475004257748021e3,
                    -6.750594898692167e5,
                    1.316000293638659e8,
                    5.564041494210414e-8,
                    8.331824119522935e-9,
                    3.551481937268812e2,
                ),
                10.0,
                0.000158147,
                "CARA 07",
            ),
            (
                [-1.381103, -2.008757, -1.666274],
                [-0.000426, -0.000311, 0.000727],
                rtn(
                    2.730011483545464e0,
                    -7.798527087944655e1,
                    2.665962873957034e3,
                    -6.338020386298382e-13,
                    -7.062822549031012e-12,
                    4.153963500401933e-1,
                ),
                rtn(
                    2.729944919062438e0,
                    -7.798412102927176e1,
                    2.665964447851581e3,
                    -6.731837309814637e-12,
                    1.225114454328491e-11,
                    4.153969295672138e-1,
                ),
                4.0,
                0.036948008,
                "CARA 08",
            ),
            (
                [-5.035858, 7.313429, -0.000420],
                [-0.001712, -0.001179, -0.000000],
                rtn(
                    1.020887925177703e1,
                    -2.755254218552751e1,
                    7.470382860526989e1,
                    2.171526847227767e-14,
                    1.976582959806184e-13,
                    4.300813243978246e-2,
                ),
                rtn(
                    1.020890130430932e1,
                    -2.755270298941628e1,
                    7.470453231276690e1,
                    -1.325051179890124e-13,
                    3.318135696761537e-13,
                    4.300744509946160e-2,
                ),
                6.0,
                0.290146291,
                "CARA 09",
            ),
            (
                [0.000421, -76.126083, 0.000000],
                [0.084255, 0.000000, 0.000000],
                rtn(
                    2.741843786640948e1,
                    -4.036901902356651e3,
                    6.965222505462344e5,
                    0.0,
                    0.0,
                    1.542470821442500e-1,
                ),
                rtn(
                    2.741843786543052e1,
                    -4.036901902333630e3,
                    6.965222505447816e5,
                    0.0,
                    0.0,
                    1.542470821440900e-1,
                ),
                4.0,
                0.002672026,
                "CARA 11",
            ),
        ];
        let mut worst = 0.0_f64;
        for (rpos, rvel, c1, c2, hbr, reference, label) in cases {
            // Combined radius split evenly across the two objects; the model uses only the sum.
            let geom = ConjunctionGeometry::new(rpos, rvel, c1, c2, 0.5 * hbr, 0.5 * hbr);
            let plane = geom.project().unwrap();
            let foster = plane.probability_foster();
            let patera = plane.probability_patera();
            let rel = (foster - reference).abs() / reference;
            worst = worst.max(rel);
            assert!(
                (foster - patera).abs() / foster < 1e-8,
                "{label}: Foster vs Patera"
            );
            println!(
                "{label}: ref={reference:.9e} foster={foster:.9e} patera={patera:.9e} rel={rel:.2e}"
            );
            assert!(
                rel < 3e-4,
                "{label}: {foster:.6e} vs ref {reference:.6e} = {rel:.2e}"
            );
        }
        println!("worst CARA relative error (Foster vs published reference): {worst:.2e}");
    }

    /// The documented Chan limitation, made concrete against an exact reference. Alfano paper cases 3
    /// (`σ_y/σ_x ≈ 81`) and 5 (`≈ 4764`) are extreme ellipses; Patera reproduces the exact reference
    /// `Pc` (Alfano 2007, via Orekit's `Alfano2005Test`), while Chan's equal-area approximation is off
    /// by `3×` on case 3 and collapses to a meaningless underflow on case 5. This is why elongated
    /// covariances call for the contour method, not the series.
    #[test]
    fn chan_degrades_where_patera_stays_exact_for_extreme_ellipses() {
        // Alfano 3: exact reference 0.10038.
        let a3 = EncounterPlane::from_diagonal(-3.8872073, 0.1591646, 1.4101830, 114.2585190, 15.0)
            .unwrap();
        let (p3, c3) = (a3.probability_patera(), a3.probability_chan());
        println!("Alfano 3 (81:1): exact≈0.10038  patera={p3:.5e}  chan={c3:.5e}");
        assert!(
            (p3 - 0.10038).abs() < 1e-4,
            "Patera should match the exact reference: {p3:.5e}"
        );
        assert!(
            c3 < 0.5 * p3,
            "Chan must visibly under-predict here: {c3:.5e} vs {p3:.5e}"
        );

        // Alfano 5: exact reference 0.044510; Chan underflows to ≈0.
        let a5 = EncounterPlane::from_diagonal(-1.2217895, 2.1230067, 0.0373279, 177.8109003, 10.0)
            .unwrap();
        let (p5, c5) = (a5.probability_patera(), a5.probability_chan());
        let f5 = a5.probability_foster();
        println!(
            "Alfano 5 (4764:1): exact≈0.044510  patera={p5:.5e}  foster={f5:.5e}  chan={c5:.5e}"
        );
        assert!(
            (p5 - 0.044510).abs() < 1e-4,
            "Patera should match the exact reference: {p5:.5e}"
        );
        assert!(
            (f5 - 0.044510).abs() < 1e-4,
            "Foster's analytic radial should too: {f5:.5e}"
        );
        assert!(c5 < 1e-10, "Chan collapses for this aspect ratio: {c5:.5e}");
    }

    /// The near-zero-miss trap: a perfectly centred conjunction (`m = 0`) is the peak of the Gaussian,
    /// not a singularity. Foster, Patera, and the grid must all agree there.
    #[test]
    fn centred_conjunction_is_handled() {
        let plane = EncounterPlane::from_diagonal(0.0, 0.0, 10.0, 30.0, 5.0).unwrap();
        let foster = plane.probability_foster();
        let patera = plane.probability_patera();
        let grid = grid_probability(&plane, 4000);
        println!("centred: foster={foster:.6e} patera={patera:.6e} grid={grid:.6e}");
        assert!((foster - patera).abs() / foster < 1e-9);
        assert!((foster - grid).abs() / foster < 5e-4);
    }

    /// The removable singularity in Patera's contour weight. `h(x) = (1 − e^{−x/2})/x` loses every
    /// digit to cancellation as `x = r² → 0`, so the code switches to the series `½ − x/8 + x²/48`
    /// below `x = 10⁻⁸`. A hard body far inside one sigma of a *centred* conjunction puts every contour
    /// point at `x = (R/σ)²`, so the whole quadrature runs in that branch and `Pc` collapses to
    /// `x·h(x)` — checkable against the exact circular closed form `1 − e^{−R²/2σ²}`, which the series
    /// reproduces to `x³/192` relative, below `10⁻²⁶` across this range. The direct form would be
    /// wrong by `2u/x`: four parts in `10⁵` at the smallest radius here.
    #[test]
    fn patera_takes_the_series_where_its_contour_weight_would_cancel() {
        for r in [1e-6, 1e-5, 5e-5, 9.9e-5] {
            let plane = EncounterPlane::from_diagonal(0.0, 0.0, 1.0, 1.0, r).unwrap();
            let want = -libm::expm1(-0.5 * r * r);
            let got = plane.probability_patera();
            let rel = (got - want).abs() / want;
            println!("R/sigma={r:e}: patera={got:.17e} exact={want:.17e} rel={rel:.2e}");
            assert!(
                rel < 1e-13,
                "R/sigma={r:e}: {got:.17e} vs exact {want:.17e} (rel {rel:.2e})"
            );
        }
    }

    /// Illegal inputs are rejected, not papered over: a non-positive-definite covariance, a degenerate
    /// (zero) relative velocity, and a negative radius each surface their specific error.
    #[test]
    fn ill_posed_inputs_are_rejected() {
        // Indefinite 2×2 (negative eigenvalue).
        assert_eq!(
            EncounterPlane::new([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], 5.0),
            Err(ConjunctionError::NonPositiveDefiniteCovariance)
        );
        // Zero variance along one axis.
        assert_eq!(
            EncounterPlane::from_diagonal(0.0, 0.0, 0.0, 10.0, 5.0),
            Err(ConjunctionError::NonPositiveDefiniteCovariance)
        );
        // Negative radius.
        assert_eq!(
            EncounterPlane::from_diagonal(0.0, 0.0, 10.0, 10.0, -1.0),
            Err(ConjunctionError::InvalidHardBodyRadius)
        );
        // Zero relative velocity: no encounter plane.
        let geom = ConjunctionGeometry::new(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            1.0,
            1.0,
        );
        assert_eq!(
            geom.project(),
            Err(ConjunctionError::DegenerateEncounterVelocity)
        );
        // A *negative* sigma squares to a positive variance, so the covariance it builds is perfectly
        // positive-definite and only `from_diagonal`'s own guard can reject it. Zero on one axis (above)
        // is caught downstream either way and so says nothing about that guard.
        assert_eq!(
            EncounterPlane::from_diagonal(0.0, 0.0, -3.0, 10.0, 5.0),
            Err(ConjunctionError::NonPositiveDefiniteCovariance)
        );
        assert_eq!(
            EncounterPlane::from_diagonal(0.0, 0.0, 10.0, -3.0, 5.0),
            Err(ConjunctionError::NonPositiveDefiniteCovariance)
        );

        // Each error says something, and says something different: an empty or shared message is how a
        // caller loses the diagnosis at the point it is needed.
        let rendered = [
            ConjunctionError::DegenerateEncounterVelocity.to_string(),
            ConjunctionError::NonPositiveDefiniteCovariance.to_string(),
            ConjunctionError::InvalidHardBodyRadius.to_string(),
        ];
        for (i, text) in rendered.iter().enumerate() {
            assert!(!text.is_empty(), "error {i} renders as the empty string");
            for other in &rendered[i + 1..] {
                assert_ne!(text, other, "two errors render identically: {text:?}");
            }
        }
    }

    /// The B-plane basis stays well conditioned whichever way the relative velocity points. The helper
    /// axis is chosen *against* the velocity's x-alignment, so a velocity along +y — where the guard's
    /// two branches disagree — is the case that tells a working rule from a broken one: pick the wrong
    /// helper and the cross product is zero, the basis is NaN, and the projection reports a covariance
    /// failure it did not have.
    #[test]
    fn the_encounter_basis_survives_a_velocity_along_each_axis() {
        for v in [
            [7000.0_f64, 0.0, 0.0],
            [0.0, 7000.0, 0.0],
            [0.0, 0.0, 7000.0],
            [-7000.0, 0.0, 0.0],
            [0.0, -7000.0, 0.0],
            [4000.0, 4000.0, 4000.0],
        ] {
            let geom = ConjunctionGeometry::new(
                [300.0, -200.0, 150.0],
                v,
                identity3(9.0),
                identity3(16.0),
                2.0,
                3.0,
            );
            let plane = geom
                .project()
                .unwrap_or_else(|e| panic!("velocity {v:?} should project cleanly, got {e}"));
            // The combined covariance is isotropic (9 + 16 = 25 on the diagonal), so *any* orthonormal
            // in-plane basis must reproduce it exactly, and the miss must keep the norm of the component
            // of the relative position perpendicular to v.
            let cov = plane.covariance();
            assert!(
                (cov[0][0] - 25.0).abs() < 1e-12
                    && (cov[1][1] - 25.0).abs() < 1e-12
                    && cov[0][1].abs() < 1e-12,
                "velocity {v:?}: projected covariance {cov:?} is not 25·I"
            );
            let vmag = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            let r = [300.0, -200.0, 150.0];
            let along = (r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) / vmag;
            let want = libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] - along * along);
            let got = libm::sqrt(plane.miss()[0].powi(2) + plane.miss()[1].powi(2));
            assert!(
                (got - want).abs() < 1e-9,
                "velocity {v:?}: |miss| {got} vs perpendicular component {want}"
            );
        }
    }

    /// The projection is frame-consistent: a miss purely along the relative velocity projects to zero
    /// (the along-track component is integrated out), and a miss purely perpendicular is preserved in
    /// full. Built so the B-plane geometry is checkable by hand.
    #[test]
    fn projection_drops_along_velocity_and_keeps_perpendicular() {
        // Velocity along x; miss purely along x ⇒ zero projected miss.
        let along = ConjunctionGeometry::new(
            [7.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            identity3(4.0),
            identity3(4.0),
            2.0,
            3.0,
        );
        let plane = along.project().unwrap();
        let miss_mag = libm::sqrt(plane.miss()[0].powi(2) + plane.miss()[1].powi(2));
        println!(
            "along-velocity miss magnitude = {miss_mag:.3e} (expect ≈0), R={}",
            plane.radius()
        );
        assert!(
            miss_mag < 1e-12,
            "along-track miss should project out: {miss_mag:.3e}"
        );
        assert!((plane.radius() - 5.0).abs() < 1e-12, "combined radius 2+3");
        // Combined covariance 4·I ⊕ 4·I = 8·I, projected 2×2 is 8·I.
        assert!((plane.covariance()[0][0] - 8.0).abs() < 1e-12);

        // Velocity along x; miss purely along z ⇒ preserved (magnitude 7 in-plane).
        let perp = ConjunctionGeometry::new(
            [0.0, 0.0, 7.0],
            [100.0, 0.0, 0.0],
            identity3(4.0),
            identity3(4.0),
            2.0,
            3.0,
        );
        let plane = perp.project().unwrap();
        let miss_mag = libm::sqrt(plane.miss()[0].powi(2) + plane.miss()[1].powi(2));
        assert!(
            (miss_mag - 7.0).abs() < 1e-12,
            "perpendicular miss preserved: {miss_mag:.3e}"
        );
    }

    /// The maximum probability is a genuine maximum over covariance scaling: it is at least the nominal
    /// `Pc` and at least the `Pc` at any hand-picked rescaling, and it is stationary (rescaling around
    /// the optimum does not beat it). For a miss well outside the hard body the peak is interior and
    /// strictly above the nominal.
    #[test]
    fn max_probability_dominates_every_scaling() {
        let plane = EncounterPlane::from_diagonal(200.0, 150.0, 300.0, 900.0, 20.0).unwrap();
        let nominal = plane.probability_foster();
        let pmax = plane.probability_max();
        println!("nominal Pc={nominal:.6e}  max Pc={pmax:.6e}");
        assert!(pmax >= nominal * (1.0 - 1e-9), "max below nominal");
        // Dominates a sweep of rescalings.
        for k in [0.1_f64, 0.3, 0.5, 2.0, 5.0, 10.0] {
            let scaled = EncounterPlane::new(
                plane.miss(),
                [
                    [plane.covariance()[0][0] * k, plane.covariance()[0][1] * k],
                    [plane.covariance()[1][0] * k, plane.covariance()[1][1] * k],
                ],
                plane.radius(),
            )
            .unwrap();
            let scaled_pc = scaled.probability_foster();
            assert!(
                pmax >= scaled_pc * (1.0 - 1e-6),
                "max below scaling k={k}: {pmax:.6e} < {scaled_pc:.6e}"
            );
        }
        // For this far miss the peak is strictly above the nominal.
        assert!(pmax > nominal, "expected an interior maximum above nominal");
    }

    /// The dilution maximum has a closed form for a circular covariance, and it is the one the
    /// literature quotes: with the hard body far outside the covariance the probability is the
    /// Gaussian density at the miss times the disc area,
    ///
    /// ```text
    /// Pc(σ) ≈ (R²/2σ²)·exp(−d²/2σ²)   ⇒   dPc/dσ² = 0 at σ² = d²/2   ⇒   Pc_max = R²/(e·d²)
    /// ```
    ///
    /// (Alfriend et al. 1999; Frisbee). Every assertion in `max_probability_dominates_every_scaling`
    /// is an inequality against a probability the same search produced, so any rearrangement of the
    /// bracket, of the golden-section update or of the 22-decade scan still returns something above
    /// the nominal and survives. This pins the value itself.
    #[test]
    fn the_dilution_maximum_is_the_closed_form_r_squared_over_e_d_squared() {
        for (d, radius) in [
            (1000.0_f64, 1.0_f64),
            (1000.0, 5.0),
            (250.0, 0.5),
            (8000.0, 10.0),
        ] {
            // The starting covariance is immaterial — the search scales it — so use a circular one far
            // from the optimum to make the bracket work for its living.
            let plane = EncounterPlane::from_diagonal(d, 0.0, 3.0, 3.0, radius).unwrap();
            let want = radius * radius / (core::f64::consts::E * d * d);
            let got = plane.probability_max();
            // The closed form drops terms of order (R/σ*)² = 2(R/d)², which is 2e-6 at the loosest of
            // these geometries, so a part in 1e4 is a bound the approximation itself can meet.
            assert!(
                (got - want).abs() < 1e-4 * want,
                "d = {d}, R = {radius}: max Pc {got:e} vs R²/(e·d²) = {want:e}"
            );
        }

        // The maximum depends on the geometry only through R/d, so scaling both leaves it fixed —
        // a property no rescaling of the search's own output can imitate.
        let a = EncounterPlane::from_diagonal(1000.0, 0.0, 3.0, 3.0, 2.0).unwrap();
        let b = EncounterPlane::from_diagonal(7000.0, 0.0, 21.0, 21.0, 14.0).unwrap();
        let (pa, pb) = (a.probability_max(), b.probability_max());
        assert!(
            (pa - pb).abs() < 1e-6 * pa,
            "the maximum is not scale-free: {pa:e} vs {pb:e}"
        );

        // The same closed form for a *correlated* covariance, where R²/(e·d²) no longer applies:
        // maximising R²/(2k√det C)·exp(−v/2k) over k gives k* = v/2 and Pc_max = R²/(e·v·√det C),
        // with v = mᵀC⁻¹m. A diagonal fixture cannot see the off-diagonal scaling at all — 0·k and
        // 0/k are the same number — and the three geometries put log k* at 7.9, −0.7 and 21.8, so a
        // scan that loses its range or a bracket that loses the peak has nowhere to hide.
        for (miss, cov, radius) in [
            (
                [1200.0_f64, -500.0],
                [[900.0_f64, 350.0], [350.0, 400.0]],
                1.0,
            ),
            ([1000.0, 0.0], [[1.0e6, 0.0], [0.0, 1.0e6]], 1.0),
            ([100_000.0, 0.0], [[4.0, 1.5], [1.5, 1.0]], 1.0),
        ] {
            let plane = EncounterPlane::new(miss, cov, radius).unwrap();
            let det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
            // v = mᵀC⁻¹m, with C⁻¹ written out rather than taken from the code under test.
            let inv_m = [
                (cov[1][1] * miss[0] - cov[0][1] * miss[1]) / det,
                (-cov[1][0] * miss[0] + cov[0][0] * miss[1]) / det,
            ];
            let v = miss[0] * inv_m[0] + miss[1] * inv_m[1];
            let want = radius * radius / (core::f64::consts::E * v * libm::sqrt(det));
            let got = plane.probability_max();
            println!(
                "log k* = {:.3}, max Pc = {got:e} vs R²/(e·v·√det) = {want:e}, rel {:.2e}",
                libm::log(0.5 * v),
                (got - want).abs() / want
            );
            assert!(
                (got - want).abs() < 1e-5 * want,
                "miss {miss:?}, C {cov:?}: max Pc {got:e} vs R²/(e·v·√det C) = {want:e}"
            );
        }
    }

    /// The quadrature resolution follows the documented `16·aspect` law between its two clamps. Every
    /// caller of `angular_samples` swallows its value into a converged integral, so nothing downstream
    /// can tell 1600 samples from 12800 — only the count itself pins the law, and the eigenvalues of a
    /// diagonal covariance are its diagonal, so no eigen-solver is needed to state the answer.
    #[test]
    fn the_angular_sample_count_follows_the_documented_aspect_law() {
        for (sigma_x, sigma_y, want) in [
            (400.0_f64, 2.0_f64, 3200_usize), // 16·200, between the clamps
            (30.0, 1.0, 1024),                // 16·30 = 480 → floored
            (1.0, 1.0, 1024),                 // isotropic → floored
            (2.0, 250.0, 2000),               // aspect is a ratio, not an order: 16·125
            (20_000.0, 1.0, 262_144),         // 16·20000 = 320000 → capped
        ] {
            let plane = EncounterPlane::from_diagonal(0.0, 0.0, sigma_x, sigma_y, 1.0).unwrap();
            let got = plane.angular_samples();
            assert_eq!(
                got, want,
                "σ = ({sigma_x}, {sigma_y}): {got} samples, expected {want}"
            );
        }

        // Confirm the premise rather than assume it: the extreme values of uᵀCu over the unit circle
        // are the eigenvalues, and for a diagonal covariance they are σ_x², σ_y².
        let cov = [[400.0 * 400.0, 0.0], [0.0, 4.0]];
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..3600 {
            let t = TAU * f64::from(k) / 3600.0;
            let u = [libm::cos(t), libm::sin(t)];
            let q = bilinear2(&cov, &u, &u);
            lo = lo.min(q);
            hi = hi.max(q);
        }
        assert!(
            (lo - 4.0).abs() < 1e-6 && (hi - 160_000.0).abs() < 1e-3,
            "{lo} {hi}"
        );
    }

    /// Chan's series must still be summed to convergence when its Poisson mean is in the hundreds.
    /// `poisson_terms` sets that cut, and every published check in this file has a mean below 1, where
    /// the truncation is 64 terms past a mean of ~0 and any term count at all suffices. Here the mean
    /// is 600 and the cut has to reach ~37√600 ≈ 900 terms past it, or the series stops inside the
    /// distribution instead of past its tail.
    #[test]
    fn chan_is_summed_to_convergence_at_a_poisson_mean_of_six_hundred() {
        // Circular covariance: Chan's equal-area step is exact, so Foster is a like-for-like reference.
        // R² = 1200 with σ = 1 puts the series mean b = R²/(2√det C) at 600, and |m| = R puts the
        // Gaussian centre on the hard-body edge, so Pc ≈ ½ and a truncated series is visible at once.
        let r = libm::sqrt(1200.0);
        let plane = EncounterPlane::new([r, 0.0], [[1.0, 0.0], [0.0, 1.0]], r).unwrap();
        let (chan, foster) = (plane.probability_chan(), plane.probability_foster());
        println!(
            "b = 600: chan = {chan:.12e}, foster = {foster:.12e}, rel = {:.2e}",
            (chan - foster).abs() / foster
        );
        assert!(
            (chan - 0.5).abs() < 0.05,
            "the fixture should sit near Pc = ½, got {chan:e}"
        );
        assert!(
            (chan - foster).abs() < 1e-12 * foster,
            "chan {chan:.12e} vs foster {foster:.12e}: the series was cut inside its own distribution"
        );

        // The series used to be walked in the linear domain from `q₀ = e^{−b}`, which stops being
        // representable at b ≈ 745 while nothing the answer needs does. Measured on that code:
        // b = 744 → 0.8208676 (66% high, from a two-bit subnormal q₀), b = 745 → 1.0, b ≥ 746 → 0.0,
        // against a true ½. Sweep straight through that band against an external oracle: for a circular
        // covariance the equal-area step is exact and Pc is the non-central χ² CDF outright, so these
        // are `scipy.stats.ncx2.cdf(R²/σ², df=2, nc=|m|²/σ²)` — cross-checked at four of the nine
        // points against an independent mpmath Poisson-mixture sum at 40 digits, agreeing to 1e-16.
        for (b, exact) in [
            (600.0_f64, 0.494_241_164_077_836_69_f64),
            (700.0, 0.494_668_433_352_817_46),
            (743.0, 0.494_825_037_465_882_97),
            (744.0, 0.494_828_517_017_779_75),
            (745.0, 0.494_831_989_560_337_14),
            (746.0, 0.494_835_455_117_055_96),
            (800.0, 0.495_012_831_765_856_03),
            (2000.0, 0.496_845_985_773_746_28),
            (50_000.0, 0.499_369_216_081_007_09),
        ] {
            let r = libm::sqrt(2.0 * b);
            let plane = EncounterPlane::new([r, 0.0], [[1.0, 0.0], [0.0, 1.0]], r).unwrap();
            let chan = plane.probability_chan();
            println!(
                "b = {b}: chan = {chan:.15e}, ncx2 = {exact:.15e}, rel = {:.2e}",
                (chan - exact).abs() / exact
            );
            assert!(
                (chan - exact).abs() < 1e-9 * exact,
                "b = {b}: chan {chan:.15e} vs ncx2 {exact:.15e}"
            );
        }
    }

    #[test]
    fn the_two_by_two_eigenvalues_carry_the_trace_and_the_determinant() {
        // `eigenvalues_2x2` is reached only through the positive-definiteness gate and the aspect
        // ratio, and both are inequalities — `λ_min > 0` and a sample count that is then clamped — so
        // the closed form itself was unconstrained. Its two defining invariants are exact: the
        // eigenvalues of a symmetric matrix sum to the trace and multiply to the determinant, taken
        // on the symmetrised matrix, which is the one the function actually decomposes.
        for c in [
            [[4.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[900.0, 250.0], [250.0, 400.0]],
            [[1.0e-6, 3.0e-7], [3.0e-7, 5.0e-6]],
            [[2.5, -1.75], [-1.75, 2.5]], // equal diagonal — the (a − d) term vanishes
            [[7.0, 2.0], [1.0, 3.0]],     // deliberately unsymmetric: b = (c01 + c10)/2
        ] {
            let (lo, hi) = eigenvalues_2x2(&c);
            let b = f64::midpoint(c[0][1], c[1][0]);
            let (trace, det) = (c[0][0] + c[1][1], c[0][0] * c[1][1] - b * b);
            let scale = trace.abs().max(1.0);
            assert!(lo <= hi, "{c:?}: eigenvalues out of order, {lo} > {hi}");
            assert!(
                (lo + hi - trace).abs() < 1e-12 * scale,
                "{c:?}: λ_min + λ_max = {} vs trace {trace}",
                lo + hi
            );
            assert!(
                (lo * hi - det).abs() < 1e-12 * scale * scale,
                "{c:?}: λ_min·λ_max = {} vs det {det}",
                lo * hi
            );
            // And each is a genuine root of the characteristic polynomial.
            for lambda in [lo, hi] {
                let residual = lambda * lambda - trace * lambda + det;
                assert!(
                    residual.abs() < 1e-11 * scale * scale,
                    "{c:?}: λ = {lambda} leaves χ(λ) = {residual}"
                );
            }
        }
    }

    /// A symmetric 3×3 covariance with equal diagonal `d` and zero off-diagonal.
    fn identity3(d: f64) -> [[f64; 3]; 3] {
        [[d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, d]]
    }

    /// Assemble a symmetric 3×3 RTN position covariance from its lower triangle.
    fn rtn(rr: f64, rt: f64, tt: f64, rn: f64, tn: f64, nn: f64) -> [[f64; 3]; 3] {
        [[rr, rt, rn], [rt, tt, tn], [rn, tn, nn]]
    }
}
