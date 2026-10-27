//! Rational Polynomial Camera (RPC00B) — the replacement sensor model optical satellites ship instead
//! of a physical pushbroom model.
//!
//! A very-high-resolution optical satellite (WorldView, Pléiades, Sentinel-2 L1, SPOT, …) does not
//! publish its interior/exterior orientation; it publishes an *RPC*: two ratios of cubic polynomials
//! that map ground `(latitude, longitude, height)` to image `(row, column)`. This is the RPC00B model
//! of the NITF / DigitalGlobe `.RPB` convention. With normalised inputs
//! `P = (lat − lat_off)/lat_scale`, `L = (lon − lon_off)/lon_scale`, `H = (h − h_off)/h_scale`,
//!
//! ```text
//! row = LINE_SCALE · NumL(P,L,H)/DenL(P,L,H) + LINE_OFF,
//! col = SAMP_SCALE · NumS(P,L,H)/DenS(P,L,H) + SAMP_OFF,
//! ```
//!
//! each `Num`/`Den` a 20-term cubic in `(P, L, H)` in the standard RPC00B term order.
//! [`RationalPolynomialCamera::forward`] evaluates that map; [`RationalPolynomialCamera::inverse`]
//! solves it for `(lat, lon)` at a known height by Newton iteration on the analytic 2×2 Jacobian
//! (fast and exact, no lookup grid). [`RationalPolynomialCamera::parse_rpb`] reads a `.RPB` file.
//!
//! Units are the RPC convention: latitude/longitude in **degrees**, height and offsets in **metres**,
//! image coordinates in **pixels** (`row` = line, `col` = sample). `no_std`, no allocation.
//!
//! Gate: `derivations/rpc00b.mac` shows that the twenty terms are the full cubic basis, that the
//! hand-written gradient is the polynomial's derivative and that the Newton step is `J⁻¹F`;
//! `derivations/rpc00b_inverse.gp` solves the RPC equations at 60 digits on the Pléiades fixture.
//! The module test `the_inverse_is_the_root_of_the_rpc_equations` (G2) holds the shipped inverse to
//! those roots.

// Single-letter P/L/H are the standard RPC normalised-coordinate names; the numerator/denominator and
// their derivative bindings (ln_n/ld_n/…) mirror the rational-function algebra; the module doc quotes
// RPC field names (LINE_OFF, NumL, …).
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown
)]

/// The 20 RPC00B cubic monomials in `(P, L, H)` (latitude, longitude, height normalised), in the
/// canonical term order shared by NITF RPC00B, GDAL and the `.RPB` format.
#[inline]
fn terms(p: f64, l: f64, h: f64) -> [f64; 20] {
    [
        1.0,
        l,
        p,
        h,
        l * p,
        l * h,
        p * h,
        l * l,
        p * p,
        h * h,
        p * l * h,
        l * l * l,
        l * p * p,
        l * h * h,
        l * l * p,
        p * p * p,
        p * h * h,
        l * l * h,
        p * p * h,
        h * h * h,
    ]
}

/// Evaluate a 20-coefficient RPC polynomial at normalised `(P, L, H)`.
#[inline]
fn poly(c: &[f64; 20], p: f64, l: f64, h: f64) -> f64 {
    let t = terms(p, l, h);
    let mut s = 0.0;
    for i in 0..20 {
        s += c[i] * t[i];
    }
    s
}

/// The partial derivatives `(∂/∂P, ∂/∂L)` of a 20-coefficient RPC polynomial at `(P, L, H)` (height
/// held fixed) — the inverse map's Jacobian needs them.
#[inline]
fn poly_grad_pl(c: &[f64; 20], p: f64, l: f64, h: f64) -> (f64, f64) {
    // ∂/∂L
    let dl = c[1]
        + c[4] * p
        + c[5] * h
        + 2.0 * c[7] * l
        + c[10] * p * h
        + 3.0 * c[11] * l * l
        + c[12] * p * p
        + c[13] * h * h
        + 2.0 * c[14] * l * p
        + 2.0 * c[17] * l * h;
    // ∂/∂P
    let dp = c[2]
        + c[4] * l
        + c[6] * h
        + 2.0 * c[8] * p
        + c[10] * l * h
        + 2.0 * c[12] * l * p
        + c[14] * l * l
        + 3.0 * c[15] * p * p
        + c[16] * h * h
        + 2.0 * c[18] * p * h;
    (dp, dl)
}

/// An RPC00B rational-polynomial camera: the offsets/scales that normalise ground and image
/// coordinates, and the four 20-term cubic coefficient vectors.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalPolynomialCamera {
    /// Image line (row) offset and scale (pixels).
    pub line_offset: f64,
    /// Image line (row) scale (pixels).
    pub line_scale: f64,
    /// Image sample (column) offset (pixels).
    pub samp_offset: f64,
    /// Image sample (column) scale (pixels).
    pub samp_scale: f64,
    /// Latitude offset (degrees).
    pub lat_offset: f64,
    /// Latitude scale (degrees).
    pub lat_scale: f64,
    /// Longitude offset (degrees).
    pub lon_offset: f64,
    /// Longitude scale (degrees).
    pub lon_scale: f64,
    /// Height offset (metres).
    pub height_offset: f64,
    /// Height scale (metres).
    pub height_scale: f64,
    /// Line (row) numerator coefficients.
    pub line_num: [f64; 20],
    /// Line (row) denominator coefficients.
    pub line_den: [f64; 20],
    /// Sample (column) numerator coefficients.
    pub samp_num: [f64; 20],
    /// Sample (column) denominator coefficients.
    pub samp_den: [f64; 20],
}

/// Why an image → ground solve produced no point.
///
/// A separate type rather than a flag on [`GroundPoint`] because the two are mutually exclusive and
/// the flag form was measured to be unreadable: with the failure carried only by `iterations`, the
/// sole thing distinguishing an answer from a non-answer was a count equal to the private iteration
/// cap, and the singular case did not even set that.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InverseError {
    /// Newton spent its whole iteration budget with the correction still above tolerance — the pixel
    /// has no ground point at this height, or one too far outside the RPC's normalisation box to be
    /// reached from its centre.
    NotConverged {
        /// Iterations spent, i.e. the cap.
        iterations: u32,
    },
    /// The Newton step left the reals: the 2×2 Jacobian was singular, or a denominator polynomial
    /// vanished and took the residual to infinity with it.
    Singular {
        /// Iterations completed before the step became unusable.
        iterations: u32,
    },
}

impl core::fmt::Display for InverseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotConverged { iterations } => {
                write!(f, "RPC inverse did not converge in {iterations} iterations")
            }
            Self::Singular { iterations } => {
                write!(
                    f,
                    "RPC inverse hit a singular Jacobian at iteration {iterations}"
                )
            }
        }
    }
}

impl core::error::Error for InverseError {}

/// The result of an inverse (image → ground) solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundPoint {
    /// Geodetic latitude (degrees).
    pub latitude: f64,
    /// Geodetic longitude (degrees).
    pub longitude: f64,
    /// Height above the ellipsoid (metres) — the fixed height the inverse was solved at.
    pub height: f64,
    /// Newton iterations taken (for diagnostics).
    pub iterations: u32,
}

impl RationalPolynomialCamera {
    /// Project a ground point `(latitude, longitude)` in degrees at `height` metres to image
    /// `(row, column)` in pixels — the RPC forward map.
    #[must_use]
    pub fn forward(&self, latitude: f64, longitude: f64, height: f64) -> (f64, f64) {
        let p = (latitude - self.lat_offset) / self.lat_scale;
        let l = (longitude - self.lon_offset) / self.lon_scale;
        let h = (height - self.height_offset) / self.height_scale;
        let row = poly(&self.line_num, p, l, h) / poly(&self.line_den, p, l, h);
        let col = poly(&self.samp_num, p, l, h) / poly(&self.samp_den, p, l, h);
        (
            row * self.line_scale + self.line_offset,
            col * self.samp_scale + self.samp_offset,
        )
    }

    /// Solve the RPC for the ground `(latitude, longitude)` that images to `(row, column)` pixels at a
    /// known `height` (metres) — the localisation / inverse map. Newton iteration on the analytic 2×2
    /// Jacobian from the offset/scale centre; converges in a handful of steps for a well-formed RPC.
    ///
    /// # Errors
    ///
    /// [`InverseError`] when Newton does not reach a root: a pixel outside the image the RPC was
    /// fitted for has none, and a rational camera evaluated far outside its normalisation box can
    /// lose its Jacobian. A returned [`GroundPoint`] is always a converged solution.
    pub fn inverse(&self, row: f64, col: f64, height: f64) -> Result<GroundPoint, InverseError> {
        const MAX_ITERS: u32 = 30;
        const TOL: f64 = 1e-12; // in normalised units ⇒ ~µdeg on the ground
        let rn = (row - self.line_offset) / self.line_scale;
        let cn = (col - self.samp_offset) / self.samp_scale;
        let h = (height - self.height_offset) / self.height_scale;

        let (mut p, mut l) = (0.0_f64, 0.0_f64); // start at the normalisation centre
        let mut iterations = 0;
        while iterations < MAX_ITERS {
            let (ln_n, ld_n) = (poly(&self.line_num, p, l, h), poly(&self.line_den, p, l, h));
            let (sn_n, sd_n) = (poly(&self.samp_num, p, l, h), poly(&self.samp_den, p, l, h));
            let r = ln_n / ld_n;
            let c = sn_n / sd_n;
            let (fr, fc) = (r - rn, c - cn); // residual

            let (ln_np, ln_nl) = poly_grad_pl(&self.line_num, p, l, h);
            let (ld_np, ld_nl) = poly_grad_pl(&self.line_den, p, l, h);
            let (sn_np, sn_nl) = poly_grad_pl(&self.samp_num, p, l, h);
            let (sd_np, sd_nl) = poly_grad_pl(&self.samp_den, p, l, h);
            // Quotient rule: d(N/D) = (N' D − N D') / D².
            let dr_dp = (ln_np * ld_n - ln_n * ld_np) / (ld_n * ld_n);
            let dr_dl = (ln_nl * ld_n - ln_n * ld_nl) / (ld_n * ld_n);
            let dc_dp = (sn_np * sd_n - sn_n * sd_np) / (sd_n * sd_n);
            let dc_dl = (sn_nl * sd_n - sn_n * sd_nl) / (sd_n * sd_n);

            let det = dr_dp * dc_dl - dr_dl * dc_dp;
            // Newton step: (Δp, Δl) = J⁻¹ · (fr, fc).
            let dp = (dc_dl * fr - dr_dl * fc) / det;
            let dl = (dr_dp * fc - dc_dp * fr) / det;
            // The step's L1 norm answers both questions below, so it is formed once. Asked in the
            // affirmative, so a NaN fails it: this subsumes the `det == 0.0` test it replaced and
            // reaches three states that one could not — a denominator polynomial that vanished makes
            // the residual infinite and the step NaN, a determinant that underflowed makes the step
            // infinite, and two finite corrections whose magnitudes sum past `f64::MAX` destroy the
            // iterate just as thoroughly. The old test broke out of the loop on none of them and
            // returned the iterate as though it were a solution.
            let step = libm::fabs(dp) + libm::fabs(dl);
            if !step.is_finite() {
                return Err(InverseError::Singular { iterations });
            }
            p -= dp;
            l -= dl;
            iterations += 1;
            if step < TOL {
                return Ok(GroundPoint {
                    latitude: p * self.lat_scale + self.lat_offset,
                    longitude: l * self.lon_scale + self.lon_offset,
                    height,
                    iterations,
                });
            }
        }
        Err(InverseError::NotConverged { iterations })
    }

    /// Parse an RPC from `.RPB` file text (the DigitalGlobe / NITF RPC00B ASCII form: `key = value;`
    /// scalars and `key = ( v1, v2, … );` coefficient lists). Returns `None` if any required field is
    /// missing. Fields other than the RPC (`satId`, `errBias`, …) are ignored.
    #[must_use]
    pub fn parse_rpb(text: &str) -> Option<Self> {
        let mut cam = Self {
            line_offset: 0.0,
            line_scale: 0.0,
            samp_offset: 0.0,
            samp_scale: 0.0,
            lat_offset: 0.0,
            lat_scale: 0.0,
            lon_offset: 0.0,
            lon_scale: 0.0,
            height_offset: 0.0,
            height_scale: 0.0,
            line_num: [0.0; 20],
            line_den: [0.0; 20],
            samp_num: [0.0; 20],
            samp_den: [0.0; 20],
        };
        cam.line_offset = scalar(text, "lineOffset")?;
        cam.samp_offset = scalar(text, "sampOffset")?;
        cam.lat_offset = scalar(text, "latOffset")?;
        cam.lon_offset = scalar(text, "longOffset")?;
        cam.height_offset = scalar(text, "heightOffset")?;
        cam.line_scale = scalar(text, "lineScale")?;
        cam.samp_scale = scalar(text, "sampScale")?;
        cam.lat_scale = scalar(text, "latScale")?;
        cam.lon_scale = scalar(text, "longScale")?;
        cam.height_scale = scalar(text, "heightScale")?;
        vector(text, "lineNumCoef", &mut cam.line_num)?;
        vector(text, "lineDenCoef", &mut cam.line_den)?;
        vector(text, "sampNumCoef", &mut cam.samp_num)?;
        vector(text, "sampDenCoef", &mut cam.samp_den)?;
        Some(cam)
    }
}

/// Is `ch` part of a floating-point token (`[0-9.eE+-]`)?
#[inline]
fn is_num_char(ch: char) -> bool {
    ch.is_ascii_digit() || matches!(ch, '.' | 'e' | 'E' | '+' | '-')
}

/// The first floating-point number appearing after `key` in `text`.
fn scalar(text: &str, key: &str) -> Option<f64> {
    let start = text.find(key)? + key.len();
    text[start..]
        .split(|ch: char| !is_num_char(ch))
        .find_map(|tok| tok.parse::<f64>().ok())
}

/// Fill `out` with the first `out.len()` floating-point numbers appearing after `key`; `None` unless
/// all were found.
fn vector(text: &str, key: &str, out: &mut [f64]) -> Option<()> {
    let start = text.find(key)? + key.len();
    let mut n = 0;
    for tok in text[start..].split(|ch: char| !is_num_char(ch)) {
        if n == out.len() {
            break;
        }
        if let Ok(v) = tok.parse::<f64>() {
            out[n] = v;
            n += 1;
        }
    }
    (n == out.len()).then_some(())
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::unwrap_used,
        clippy::excessive_precision
    )]
    use super::*;

    // A real RPC (RPC00B), Pléiades over Réunion Island — coefficients extracted from the public s2p
    // test image centreborelli/s2p tests/data/input_pair/img_01.tif, written in .RPB form. The oracle
    // values below are from `rpcm` (an independent RPC00B implementation) on the same real RPC.
    const RPB: &str = include_str!("../tests/fixtures/pleiades_reunion.rpb");

    #[test]
    fn parse_rpb_reads_the_real_offsets_scales_and_coefficients() {
        let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        assert_eq!(cam.line_offset, 19403.5);
        assert_eq!(cam.samp_offset, 19999.5);
        assert_eq!(cam.lat_offset, -21.2316081288);
        assert_eq!(cam.lon_offset, 55.7119698801);
        assert_eq!(cam.height_offset, 1295.0);
        assert_eq!(cam.line_scale, 512.0);
        assert_eq!(cam.samp_scale, 512.0);
        assert_eq!(cam.lat_scale, 0.0911805852907);
        assert_eq!(cam.lon_scale, 0.0985353286675);
        assert_eq!(cam.height_scale, 1315.0);
        assert_eq!(cam.line_num[0], -37.284870906);
        assert_eq!(cam.line_num[19], 9.58883770134e-05);
        assert_eq!(cam.samp_den[10], -1.26431542483e-07);
    }

    #[test]
    fn forward_matches_the_rpcm_oracle() {
        let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        // (lat, lon, height, row, col) from rpcm.projection on the same real RPC.
        let cases = [
            (
                -21.2316081288,
                55.7119698801,
                1295.0,
                313.64609612799904,
                13058.5944177152,
            ),
            (
                -21.19513589468372,
                55.68240928149975,
                1495.0,
                -7563.522457597814,
                7015.398676830246,
            ),
            (
                -21.277198421445352,
                55.76123754443375,
                895.0,
                10074.528498659847,
                23098.192074390816,
            ),
            (
                -21.21337201174186,
                55.72182341296675,
                2095.0,
                -3460.516398825199,
                15153.561849683078,
            ),
        ];
        for (lat, lon, h, row, col) in cases {
            let (r, c) = cam.forward(lat, lon, h);
            assert!(
                (r - row).abs() < 1e-4 && (c - col).abs() < 1e-4,
                "forward({lat},{lon},{h}) = ({r},{c}) vs rpcm ({row},{col})"
            );
        }
    }

    /// The analytic gradient against a difference of the polynomial it is the gradient of.
    ///
    /// `poly_grad_pl` is twenty hand-written terms and it is only ever read by the Newton loop in
    /// `inverse`, which re-linearises every iteration and converges from any descent direction. That
    /// is why 51 mutants live in these two expressions with the `rpcm` oracle green.
    ///
    /// The oracle is `poly` itself, differenced. Richardson-extrapolating two central differences,
    /// `(4·D(e) − D(2e))/3`, is *exact* for a cubic: the `e²/6` term that one central difference
    /// carries cancels between the two, and a cubic has no fourth derivative to leave a remainder.
    /// So this is an equality check to rounding, not a tolerance on a truncation.
    #[test]
    fn the_analytic_gradient_is_the_gradient_of_the_polynomial() {
        let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        let e = 0.125; // a power of two, so `p ± e` and `p ± 2e` are exact
        for (name, c) in [
            ("line_num", &cam.line_num),
            ("line_den", &cam.line_den),
            ("samp_num", &cam.samp_num),
            ("samp_den", &cam.samp_den),
        ] {
            let scale = c.iter().fold(0.0_f64, |m, x| {
                if m >= libm::fabs(*x) {
                    m
                } else {
                    libm::fabs(*x)
                }
            });
            for &(p, l, h) in &[
                (0.0, 0.0, 0.0),
                (0.5, -0.25, 0.75),
                (-0.875, 0.625, -0.5),
                (1.0, 1.0, 1.0),
                (-1.0, 0.375, 0.0),
            ] {
                let central =
                    |f: &dyn Fn(f64) -> f64, step: f64| (f(step) - f(-step)) / (2.0 * step);
                let along_p = |d: f64| poly(c, p + d, l, h);
                let along_l = |d: f64| poly(c, p, l + d, h);
                let want_dp = (4.0 * central(&along_p, e) - central(&along_p, 2.0 * e)) / 3.0;
                let want_dl = (4.0 * central(&along_l, e) - central(&along_l, 2.0 * e)) / 3.0;
                let (dp, dl) = poly_grad_pl(c, p, l, h);
                assert!(
                    libm::fabs(dp - want_dp) < 1e-12 * scale,
                    "{name} d/dP at ({p},{l},{h}): {dp} vs {want_dp}"
                );
                assert!(
                    libm::fabs(dl - want_dl) < 1e-12 * scale,
                    "{name} d/dL at ({p},{l},{h}): {dl} vs {want_dl}"
                );
            }
        }
    }

    /// A camera whose image axes are not aligned with latitude and longitude.
    ///
    /// The Pléiades RPC above is very nearly north-up: row is a function of `P` and column of `L`, so
    /// the off-diagonal entries of the Newton Jacobian, `dr/dL` and `dc/dP`, are two orders below the
    /// diagonal ones. Corrupting them barely turns the step and the solve still lands in four
    /// iterations — which is why eleven mutants of exactly those two lines outlive both the `rpcm`
    /// oracle and the exact iteration count.
    ///
    /// This one is synthetic and says so: `row ≈ P + L` and `col ≈ P − L` put the mixing at 45°, and
    /// the `L·P` cross terms give Newton something to iterate on. The ground truth is not a second
    /// implementation but the forward map itself — a point solved from a pixel must re-image to that
    /// pixel — plus the cost, which is where a wrongly-turned step shows up first.
    #[test]
    fn inverse_solves_a_camera_whose_axes_mix_latitude_and_longitude() {
        let mut c_line = [0.0f64; 20];
        c_line[2] = 1.0; // P
        c_line[1] = 1.0; // L
        c_line[4] = 0.30; // L·P
        c_line[8] = -0.15; // P²
        let mut c_samp = [0.0f64; 20];
        c_samp[2] = 1.0; // P
        c_samp[1] = -1.0; // L
        c_samp[4] = -0.25; // L·P
        c_samp[7] = 0.20; // L²
        // Denominators that actually vary. A real RPC normalises them to about 1 over its own
        // domain, and `ld_n ≈ 1` makes `ld_n · ld_n` and `ld_n / ld_n` the same number to a part in
        // a thousand -- so the quotient rule's divisor is untested by any near-unity camera. These
        // stay positive over |P|, |L| <= 1: the smaller bottoms out at 1 − 0.4 − 0.25 = 0.35.
        let mut line_den = [0.0f64; 20];
        line_den[0] = 1.0;
        line_den[1] = 0.40; // L
        line_den[2] = 0.25; // P
        let mut samp_den = [0.0f64; 20];
        samp_den[0] = 1.0;
        samp_den[1] = -0.30; // L
        samp_den[2] = 0.35; // P
        let cam = RationalPolynomialCamera {
            line_offset: 0.0,
            line_scale: 1000.0,
            samp_offset: 0.0,
            samp_scale: 1000.0,
            lat_offset: 0.0,
            lat_scale: 1.0,
            lon_offset: 0.0,
            lon_scale: 1.0,
            height_offset: 0.0,
            height_scale: 1000.0,
            line_num: c_line,
            line_den,
            samp_num: c_samp,
            samp_den,
        };
        // The origin costs one pass because the normalisation centre is already the solution; the
        // two off-centre pixels take the quadratic four on top of it.
        let expected = [(1u32, 0.0, 0.0), (5, 400.0, -300.0), (6, -700.0, 250.0)];
        for (iters, row, col) in expected {
            let g = cam.inverse(row, col, 0.0).unwrap();
            let (r, c) = cam.forward(g.latitude, g.longitude, 0.0);
            println!(
                "rpc mixed-axis inverse({row},{col}): ({}, {}) in {} iterations",
                g.latitude, g.longitude, g.iterations
            );
            assert!(
                (r - row).abs() < 1e-9 && (c - col).abs() < 1e-9,
                "round-trip off by ({}, {})",
                r - row,
                c - col
            );
            assert_eq!(g.iterations, iters, "({row},{col})");
        }
        // A pixel the camera cannot see: Newton has no root to find and runs to the cap. Pinned
        // because the cap is otherwise unreachable, and because this is the failure the error path
        // exists for -- before it, the same call returned a `GroundPoint` whose only mark of not
        // being a solution was an iteration count equal to a private constant.
        let lost = cam.inverse(1.0e9, -4.0e8, 0.0);
        println!("rpc unreachable pixel: {lost:?}");
        assert_eq!(lost, Err(InverseError::NotConverged { iterations: 30 }));
    }

    /// A camera whose Jacobian is singular everywhere is refused, not answered.
    ///
    /// `line_num = samp_num = P` makes both image coordinates the same function of the ground, so
    /// `dc/dP · dr/dL − dr/dP · dc/dL` is exactly zero at every iterate and Newton has no direction
    /// to step in. This is the branch the old `det == 0.0` test broke out of, returning the
    /// normalisation centre with `iterations == 0` — indistinguishable from an instant convergence.
    #[test]
    fn a_camera_with_no_invertible_direction_is_refused() {
        let mut num = [0.0f64; 20];
        num[2] = 1.0; // P
        let mut den = [0.0f64; 20];
        den[0] = 1.0;
        let cam = RationalPolynomialCamera {
            line_offset: 0.0,
            line_scale: 1000.0,
            samp_offset: 0.0,
            samp_scale: 1000.0,
            lat_offset: 0.0,
            lat_scale: 1.0,
            lon_offset: 0.0,
            lon_scale: 1.0,
            height_offset: 0.0,
            height_scale: 1000.0,
            line_num: num,
            line_den: den,
            samp_num: num,
            samp_den: den,
        };
        let got = cam.inverse(400.0, -300.0, 0.0);
        println!("rpc singular camera: {got:?}");
        assert_eq!(got, Err(InverseError::Singular { iterations: 0 }));
    }

    /// The two refusals must read differently and must name the iteration they carry, or the caller
    /// logging the error learns nothing the `Debug` form would not have told them.
    #[test]
    fn the_two_refusals_say_which_one_happened_and_where() {
        assert_eq!(
            InverseError::NotConverged { iterations: 30 }.to_string(),
            "RPC inverse did not converge in 30 iterations"
        );
        assert_eq!(
            InverseError::Singular { iterations: 4 }.to_string(),
            "RPC inverse hit a singular Jacobian at iteration 4"
        );
    }

    /// `(row, col, height, latitude, longitude)` from `derivations/rpc00b_inverse.gp`: the root of
    /// the RPC00B equations for the fixture's camera, solved at 60 digits with PARI's own polynomial
    /// derivatives after rounding every fixture value to the double it parses to.
    const RPC_ROOTS: [(f64, f64, f64, f64, f64); 5] = [
        (
            19403.5,
            19999.5,
            1295.0,
            -2.131916072552630990937423630761e1,
            5.574577016548363170061062134390e1,
        ),
        (
            5000.0,
            8000.0,
            1595.0,
            -2.125239828214131955235100892956e1,
            5.568707736118836504842441152456e1,
        ),
        (
            30000.0,
            32000.0,
            795.0,
            -2.136884259468818979614969844290e1,
            5.580482793891648726624951334009e1,
        ),
        (
            19403.5,
            19999.5,
            2295.0,
            -2.131780866575942820779899334821e1,
            5.574519026048746351658582153936e1,
        ),
        (
            19150.25,
            20311.75,
            1043.5,
            -2.131835903983073324720912747178e1,
            5.574744692592110660888244346599e1,
        ),
    ];

    /// (G2) `inverse` returns the root of the RPC00B equations `NumL/DenL = r̂`, `NumS/DenS = ĉ` —
    /// the Newton step `derivations/rpc00b.mac` proves to be `J⁻¹F` with the exact gradient, solved
    /// independently at 60 digits. The bound is the floor of evaluating the cubics in `f64`: an ulp
    /// of the sum of their term magnitudes, carried to the ground by the scale, plus an ulp of the
    /// de-normalised coordinate itself.
    #[test]
    fn the_inverse_is_the_root_of_the_rpc_equations() {
        let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        let magnitude = |c: &[f64; 20], p: f64, l: f64, h: f64| {
            terms(p, l, h)
                .iter()
                .zip(c)
                .map(|(t, ci)| (t * ci).abs())
                .sum::<f64>()
        };
        let mut worst = 0.0_f64;
        for (row, col, height, lat, lon) in RPC_ROOTS {
            let g = cam.inverse(row, col, height).unwrap();
            let p = (lat - cam.lat_offset) / cam.lat_scale;
            let l = (lon - cam.lon_offset) / cam.lon_scale;
            let h = (height - cam.height_offset) / cam.height_scale;
            let floor = magnitude(&cam.line_num, p, l, h)
                + magnitude(&cam.line_den, p, l, h)
                + magnitude(&cam.samp_num, p, l, h)
                + magnitude(&cam.samp_den, p, l, h);
            let tol_lat = 16.0 * f64::EPSILON * (lat.abs() + cam.lat_scale * floor);
            let tol_lon = 16.0 * f64::EPSILON * (lon.abs() + cam.lon_scale * floor);
            let (dlat, dlon) = ((g.latitude - lat).abs(), (g.longitude - lon).abs());
            assert!(
                dlat <= tol_lat && dlon <= tol_lon,
                "({row}, {col}, {height}): off by ({dlat:.3e}, {dlon:.3e})° (allowed ({tol_lat:.3e}, {tol_lon:.3e}))"
            );
            worst = worst.max(dlat / tol_lat).max(dlon / tol_lon);
        }
        println!("rpc inverse vs 60 digits: worst {worst:.3} of the evaluation floor");
    }

    #[test]
    fn inverse_matches_the_rpcm_oracle_and_round_trips() {
        let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        // (row, col, height, lat, lon) from rpcm.localization on the same real RPC.
        let cases = [
            (
                19403.5,
                19999.5,
                1295.0,
                -21.31916072552632,
                55.74577016548356,
            ),
            (
                5000.0,
                8000.0,
                1595.0,
                -21.252398282141336,
                55.68707736118835,
            ),
            (
                30000.0,
                32000.0,
                795.0,
                -21.368842594688147,
                55.80482793891624,
            ),
            (
                19403.5,
                19999.5,
                2295.0,
                -21.317808665759436,
                55.74519026048741,
            ),
        ];
        for (row, col, h, lat, lon) in cases {
            let g = cam.inverse(row, col, h).unwrap();
            assert!(
                (g.latitude - lat).abs() < 1e-6 && (g.longitude - lon).abs() < 1e-6,
                "inverse({row},{col},{h}) = ({}, {}) vs rpcm ({lat}, {lon})",
                g.latitude,
                g.longitude
            );
            // rpcm-independent self-check: the solved ground point re-images to the pixel.
            let (r, c) = cam.forward(g.latitude, g.longitude, h);
            // The Newton tolerance is 1e-12 in normalised units and the scales are 512 pixels, so a
            // converged solve re-images to well under a nanopixel; 1e-6 left three orders of slack.
            assert!(
                (r - row).abs() < 1e-8 && (c - col).abs() < 1e-8,
                "round-trip off by ({}, {}) pixels",
                r - row,
                c - col
            );
            println!("rpc inverse({row},{col},{h}): {} iterations", g.iterations);
            // Exact, not a bound. Newton re-linearises every pass, so it reaches the same ground
            // point from any descent direction and a corrupted Jacobian shows up only in the price:
            // 37 mutants of the quotient rule and the 2x2 solve below survived `iterations < 30`.
            // Four is what the true Jacobian earns on every case here -- one step to leave the
            // normalisation centre and three to converge quadratically -- and the count is a
            // deterministic function of f64 arithmetic on a fixed RPC, so a band buys only blindness.
            assert_eq!(
                g.iterations, 4,
                "({row},{col},{h}) took {} iterations",
                g.iterations
            );
        }
    }
}
