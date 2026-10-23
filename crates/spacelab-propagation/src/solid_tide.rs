//! IERS 2010 solid Earth tide (IERS Conventions 2010, Chapter 6).
//!
//! The tide-raising potential of the Sun and Moon deforms the Earth, changing its low-degree
//! geopotential coefficients by ΔC̄_nm, ΔS̄_nm. Step 1 (frequency-independent) is the degree-2 and
//! degree-3 changes (Eq. 6.6) plus the degree-4 back-coupling from the degree-2 tide through the k⁺
//! Love numbers (Eq. 6.7), with the nominal anelastic Love numbers of Table 6.3. The
//! frequency-dependent **Step 2** degree-2 corrections (Eq. 6.8, Tables 6.5a/b/c) are supplied by the
//! caller through the [`SolidEarthTide::step2`] closure ([`step2_corrections`]) and added to the
//! degree-2 band each step; pass `|_| [0.0; 5]` for Step 1 only. It is a peer [`Acceleration`],
//! superposed on the static [`GeoPotential`] — not injected into it, since the changes track the
//! Sun/Moon and so vary every step.
//!
//! The tidal potential's gradient is evaluated by *reusing* the tested [`GeoPotential`] recursion: a
//! degree-4 field carrying only the ΔC̄/ΔS̄ (plus the unavoidable monopole) is built each step and the
//! monopole acceleration subtracted, leaving the tidal acceleration — no second harmonic evaluator to
//! validate. The Sun, Moon and satellite are all rotated into the ITRF here (a fused
//! [`crate::forces::BodyOriented`]), because the tidal potential is body-fixed and the ephemeris is
//! inertial.
//!
//! # Provenance
//!
//! Love numbers: IERS Conventions 2010, Table 6.3 (anelastic Earth). The ΔC̄/ΔS̄ formula and the
//! resulting acceleration are cross-checked in `derivations/solid_tide.py`, which validates the
//! fully-normalized Legendre against closed forms and the analytic gradient against a finite
//! difference of the potential. The embedded geopotential is zero-tide, matching the IERS convention.

// Dense IERS/geodesy notation (ΔC̄_nm, P̄_nm, k₂ₘ⁺) and a Love number that happens to sit near log10(2).
#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::approx_constant
)]

use spacelab_autodiff::Scalar;

use crate::GeoPotential;
use crate::ephemeris::Ephemeris;
use crate::forces::Acceleration;
use crate::tide_step2_tables::{DIURNAL, LONG_PERIOD, SEMIDIURNAL};

// IERS 2010 Table 6.3, nominal (anelastic) solid Earth tide Love numbers, indexed by order m.
/// Real part of the degree-2 Love numbers k₂ₘ.
const K2_RE: [f64; 3] = [0.301_90, 0.298_30, 0.301_02];
/// Imaginary part of the degree-2 Love numbers k₂ₘ (anelastic out-of-phase response).
const K2_IM: [f64; 3] = [0.0, -0.001_44, -0.001_30];
/// Degree-4 coupling Love numbers k₂ₘ⁺ (the degree-2 tide's effect on the degree-4 field).
const K2_PLUS: [f64; 3] = [-0.000_89, -0.000_80, -0.000_57];
/// Degree-3 Love numbers k₃ₘ (real; the imaginary parts are zero at degree 3).
const K3_RE: [f64; 4] = [0.093, 0.093, 0.093, 0.094];
/// Permanent (zero-frequency) part of ΔC̄₂₀ to remove for a zero-tide background field:
/// `A₀·H₀·k₂₀` = 4.4228e-8 · (−0.31460) · 0.30190 (IERS Conventions 2010, Eq. 6.13). Without this the
/// permanent tide is double-counted against the zero-tide C̄₂₀ already in the embedded geopotential.
const PERMANENT_TIDE_C20: f64 = 4.4228e-8 * -0.314_60 * 0.301_90;

/// IERS 2010 Step 2 (frequency-dependent) corrections to the degree-2 tidal coefficients,
/// `[ΔC̄₂₀, ΔC̄₂₁, ΔS̄₂₁, ΔC̄₂₂, ΔS̄₂₂]`, from the mean sidereal time `gmst` and the five Delaunay
/// arguments `[l, l′, F, D, Ω]` (all radians). Each constituent's argument is
/// `θ = τ·(GMST + π) − Σ nᵢ·argᵢ`, and the tabulated in-/out-of-phase amplitudes (IERS Tables
/// 6.5a/b/c, units 10⁻¹²) combine per band (Eq. 6.8a/6.8b). Adds the ~mm-level frequency dependence
/// that Step 1 (the frequency-independent Love numbers) omits.
#[must_use]
pub fn step2_corrections(gmst: f64, delaunay: [f64; 5]) -> [f64; 5] {
    let [l, lp, f, d, om] = delaunay;
    let gamma = gmst + core::f64::consts::PI;
    let theta = |tau: i32, m: [i32; 5]| {
        f64::from(tau) * gamma
            - (f64::from(m[0]) * l
                + f64::from(m[1]) * lp
                + f64::from(m[2]) * f
                + f64::from(m[3]) * d
                + f64::from(m[4]) * om)
    };
    let (mut c20, mut c21, mut s21, mut c22, mut s22) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(tau, m, ip, op) in &LONG_PERIOD {
        let (s, c) = libm::sincos(theta(tau, m));
        c20 += ip * c - op * s;
    }
    for &(tau, m, ip, op) in &DIURNAL {
        let (s, c) = libm::sincos(theta(tau, m));
        c21 += ip * s + op * c;
        s21 += ip * c - op * s;
    }
    for &(tau, m, ip, op) in &SEMIDIURNAL {
        let (s, c) = libm::sincos(theta(tau, m));
        c22 += ip * c - op * s;
        s22 += -ip * s - op * c;
    }
    [
        c20 * 1e-12,
        c21 * 1e-12,
        s21 * 1e-12,
        c22 * 1e-12,
        s22 * 1e-12,
    ]
}

/// arcsec → rad (IERS/Hipparchus `ARC_SECONDS_TO_RADIANS`), for the pole-tide mean-pole model.
const ARC_SEC_TO_RAD: f64 = 4.848_136_811_095_36e-6;

/// Which conventional pole the wobble variables `(m1, m2)` of the pole tides are measured from.
///
/// The IERS replaced one model with the other, and they diverge fast: the `x` coordinates separate
/// at 5.94 mas/yr and the `y` coordinates at 4.09 mas/yr, so by 2024 the two poles are 111 mas and
/// −60 mas apart — larger than the polar motion excursion the pole tide is meant to respond to. A
/// current computation therefore has to pick, and picking wrong is not a rounding difference: at a
/// representative `(xp, yp)` the resulting `ΔC̄21` comes out with the opposite sign.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MeanPole {
    /// Chapter 7 as amended 2018/02/01, Eq. (21): the **secular pole**, one straight line fitted to
    /// the 1900–2017 polar motion, `xs = 55.0 + 1.677 (t − 2000)`, `ys = 320.5 + 3.460 (t − 2000)`
    /// mas. This is what §6.4 and §6.5 mean by `(m1, m2)` today.
    #[default]
    Secular2018,
    /// Table 7.7 of the 2010 printing: a cubic before 2010-01-01 and a linear extension after.
    /// Retired by the 2018 update, and kept only because the Orekit fixtures were taken against it —
    /// its linear branch is an extrapolation of a fit that ended in 2010 and drifts without bound.
    Iers2010,
}

/// Pole-tide wobble variables `(m1, m2)` in radians — `m1 = xp − x̄p`, `m2 = ȳp − yp` (IERS
/// Conventions §7.1.4 Eq. 25) against the conventional pole `pole` names, `t` seconds since J2000
/// (2000-01-01 12:00 TT). Horner in seconds with the mas→radian factor pre-divided once per power;
/// the [`MeanPole::Iers2010`] branch is exactly Orekit's `computePoleWobble`. Shared by the solid
/// and ocean pole tides, which is why the choice belongs here and not in either of them.
fn pole_wobble(xp: f64, yp: f64, t: f64, pole: MeanPole) -> (f64, f64) {
    const JULIAN_YEAR: f64 = 31_557_600.0;
    const F0: f64 = ARC_SEC_TO_RAD / 1000.0; // mas → rad
    const F1: f64 = F0 / JULIAN_YEAR;
    const F2: f64 = F1 / JULIAN_YEAR;
    const F3: f64 = F2 / JULIAN_YEAR;
    const T_CHANGE: f64 = 3652.5 * 86400.0; // 2010-01-01 00:00 TT since J2000
    let (mean_x, mean_y) = match pole {
        MeanPole::Secular2018 => (1.677 * F1 * t + 55.0 * F0, 3.460 * F1 * t + 320.5 * F0),
        MeanPole::Iers2010 if t <= T_CHANGE => (
            ((0.007_024 * F3 * t + 0.18413 * F2) * t + 1.8243 * F1) * t + 55.974 * F0,
            ((-0.000_908 * F3 * t - 0.10729 * F2) * t + 1.7896 * F1) * t + 346.346 * F0,
        ),
        MeanPole::Iers2010 => (
            7.6141 * F1 * t + 23.513 * F0,
            -0.6287 * F1 * t + 358.891 * F0,
        ),
    };
    (xp - mean_x, mean_y - yp)
}

/// IERS solid Earth **pole tide** (§6.4, Eq. 6.23): the degree-2 order-1 corrections `[ΔC̄21, ΔS̄21]`
/// driven by the polar motion `(xp, yp)` (radians) about the conventional pole `pole` names, `t`
/// seconds since J2000. These add to the same C̄21/S̄21 band as [`step2_corrections`], so a caller
/// applying the full IERS tide sums them into the `SolidEarthTide::step2` closure's ΔC21/ΔS21 slots —
/// no force change. Orekit includes this by default (`SolidTides` `poleTide=true`) and measures it
/// from [`MeanPole::Iers2010`].
#[must_use]
pub fn pole_tide_corrections(xp: f64, yp: f64, t: f64, pole: MeanPole) -> [f64; 2] {
    const GLOBAL: f64 = -1.333e-9 / ARC_SEC_TO_RAD;
    const RATIO: f64 = 0.0115;
    let (m1, m2) = pole_wobble(xp, yp, t, pole);
    [GLOBAL * (m1 + RATIO * m2), GLOBAL * (m2 - RATIO * m1)]
}

/// IERS **ocean** pole tide (§6.5, Eq. 6.24, degree-2 form): the degree-2 order-1 corrections
/// `[ΔC̄21, ΔS̄21]` from the polar motion, about the same conventional pole as the solid pole tide —
/// §6.5 and §6.4 both defer to §7.1.4 for `(m1, m2)`, so the two must be given the same `pole` or the
/// superposition is inconsistent. Matches Orekit's `getOceanPoleTide` at [`MeanPole::Iers2010`]. The
/// full pole-tide contribution is `pole_tide_corrections + ocean_pole_tide_corrections`.
#[must_use]
pub fn ocean_pole_tide_corrections(xp: f64, yp: f64, t: f64, pole: MeanPole) -> [f64; 2] {
    let (m1, m2) = pole_wobble(xp, yp, t, pole);
    [
        -2.1778e-10 * (m1 - 0.01724 * m2) / ARC_SEC_TO_RAD,
        -1.7232e-10 * (m2 - 0.03365 * m1) / ARC_SEC_TO_RAD,
    ]
}

/// Fully-normalized associated Legendre `P̄_nm(x)` (geodesy 4π convention, no Condon–Shortley phase),
/// for the low degrees the tide needs (`n ≤ 3`).
fn pbar(n: usize, m: usize, x: f64) -> f64 {
    let t = libm::sqrt((1.0 - x * x).max(0.0)); // cos(latitude)
    let mut pmm = 1.0;
    for k in 1..=m {
        pmm *= libm::sqrt((2.0 * k as f64 + 1.0) / (2.0 * k as f64)) * t;
    }
    if m >= 1 {
        pmm *= core::f64::consts::SQRT_2; // the (2 − δ_0m) factor of the geodesy 4π normalization
    }
    if n == m {
        return pmm;
    }
    let mf = m as f64;
    let mut p_prev = pmm; // P̄_{m,m}
    let mut p_cur = libm::sqrt(2.0 * mf + 3.0) * x * pmm; // P̄_{m+1,m}
    for nn in (m + 2)..=n {
        let nf = nn as f64;
        let a = libm::sqrt((2.0 * nf - 1.0) * (2.0 * nf + 1.0) / ((nf - mf) * (nf + mf)));
        let b = libm::sqrt(
            (2.0 * nf + 1.0) * (nf + mf - 1.0) * (nf - mf - 1.0)
                / ((nf - mf) * (nf + mf) * (2.0 * nf - 3.0)),
        );
        let p_next = a * x * p_cur - b * p_prev;
        p_prev = p_cur;
        p_cur = p_next;
    }
    p_cur
}

/// IERS 2010 solid Earth tide (Step 1) as a superposed acceleration.
///
/// Embeds a Sun and Moon [`Ephemeris`] (inertial, like [`crate::forces::ThirdBody`]) and a GCRF→ITRF
/// rotation closure (like [`crate::forces::BodyOriented`]). `mu_earth`/`re_earth` must be the same
/// constants as the static [`GeoPotential`] the tide is added to.
pub struct SolidEarthTide<ES, EM, R, F2> {
    /// Earth gravitational parameter, m³/s² (the geopotential's `μ`).
    pub mu_earth: f64,
    /// Earth reference radius, m (the geopotential's `Rₑ`).
    pub re_earth: f64,
    /// Sun gravitational parameter, m³/s².
    pub mu_sun: f64,
    /// Moon gravitational parameter, m³/s².
    pub mu_moon: f64,
    /// Sun ephemeris (geocentric inertial position).
    pub sun: ES,
    /// Moon ephemeris (geocentric inertial position).
    pub moon: EM,
    /// GCRF→ITRF rotation (row-major) at time `t`, seconds from the propagation epoch.
    pub rotate: R,
    /// IERS 2010 Step-2 frequency-dependent degree-2 corrections `[ΔC̄₂₀, ΔC̄₂₁, ΔS̄₂₁, ΔC̄₂₂, ΔS̄₂₂]`
    /// at time `t` (seconds from epoch). The caller owns the time scales: compute GMST and the five
    /// Delaunay arguments at `t` and return [`step2_corrections`]. Pass `|_| [0.0; 5]` for the
    /// frequency-independent Step-1 tide only.
    pub step2: F2,
}

impl<ES, EM, R, F2> SolidEarthTide<ES, EM, R, F2> {
    /// Step-1 ΔC̄_nm, ΔS̄_nm for `n = 2, 3, 4` from the Sun and Moon positions in the ITRF.
    fn delta_cs(&self, sun_itrf: [f64; 3], moon_itrf: [f64; 3]) -> [(usize, usize, f64, f64); 10] {
        // sc[n][m], ss[n][m] = Σ_j (GM_j/GM_E)(Rₑ/r_j)^{n+1} P̄_nm(sinφ_j) · {cos, sin}(mλ_j), n = 2, 3.
        let mut sc = [[0.0_f64; 4]; 4];
        let mut ss = [[0.0_f64; 4]; 4];
        for (pos, gm) in [(moon_itrf, self.mu_moon), (sun_itrf, self.mu_sun)] {
            // A body with no position raises no tide. The permanent-tide subtraction below stays: it
            // is bookkeeping against the zero-tide background field, not a term either body raises.
            if crate::ephemeris::is_absent(pos) {
                continue;
            }
            let r = libm::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
            let x = pos[2] / r; // sin(latitude)
            let lon = libm::atan2(pos[1], pos[0]);
            let gm_ratio = gm / self.mu_earth;
            for n in 2..=3 {
                let radial = gm_ratio * libm::pow(self.re_earth / r, (n + 1) as f64);
                for m in 0..=n {
                    let fac = radial * pbar(n, m, x);
                    sc[n][m] += fac * libm::cos(m as f64 * lon);
                    ss[n][m] += fac * libm::sin(m as f64 * lon);
                }
            }
        }

        // ΔC̄ = (kR·Sc + kI·Ss)/(2n+1), ΔS̄ = (kR·Ss − kI·Sc)/(2n+1); degree 4 uses the real k⁺ and the
        // degree-2 sums (Eq. 6.7).
        let two = |m: usize| {
            (
                (K2_RE[m] * sc[2][m] + K2_IM[m] * ss[2][m]) / 5.0,
                (K2_RE[m] * ss[2][m] - K2_IM[m] * sc[2][m]) / 5.0,
            )
        };
        let three = |m: usize| (K3_RE[m] * sc[3][m] / 7.0, K3_RE[m] * ss[3][m] / 7.0);
        let four = |m: usize| (K2_PLUS[m] * sc[2][m] / 5.0, K2_PLUS[m] * ss[2][m] / 5.0);
        let (c20_raw, s20) = two(0);
        let c20 = c20_raw - PERMANENT_TIDE_C20; // zero-tide background (IERS Eq. 6.13)
        let (c21, s21) = two(1);
        let (c22, s22) = two(2);
        let (c30, s30) = three(0);
        let (c31, s31) = three(1);
        let (c32, s32) = three(2);
        let (c33, s33) = three(3);
        let (c40, s40) = four(0);
        let (c41, s41) = four(1);
        let (c42, s42) = four(2);
        [
            (2, 0, c20, s20),
            (2, 1, c21, s21),
            (2, 2, c22, s22),
            (3, 0, c30, s30),
            (3, 1, c31, s31),
            (3, 2, c32, s32),
            (3, 3, c33, s33),
            (4, 0, c40, s40),
            (4, 1, c41, s41),
            (4, 2, c42, s42),
        ]
    }
}

impl<ES: Ephemeris, EM: Ephemeris, R: Fn(f64) -> [[f64; 3]; 3], F2: Fn(f64) -> [f64; 5]>
    Acceleration for SolidEarthTide<ES, EM, R, F2>
{
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let m = (self.rotate)(t.value());
        let e = |i: usize, j: usize| S::from_f64(m[i][j]);
        // r_itrf = M · r_gcrf
        let rf = [
            e(0, 0) * r[0] + e(0, 1) * r[1] + e(0, 2) * r[2],
            e(1, 0) * r[0] + e(1, 1) * r[1] + e(1, 2) * r[2],
            e(2, 0) * r[0] + e(2, 1) * r[1] + e(2, 2) * r[2],
        ];
        // Sun/Moon are inertial (f64); rotate into the ITRF too.
        let rot = |v: [f64; 3]| {
            [
                m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
                m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
                m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
            ]
        };
        let mut coeffs = self.delta_cs(
            rot(self.sun.position(t.value())),
            rot(self.moon.position(t.value())),
        );
        // Add the IERS 2010 Step-2 frequency-dependent corrections to the degree-2 band. Layout:
        // coeffs[0] = (2,0,·), coeffs[1] = (2,1,·), coeffs[2] = (2,2,·); s2 = [ΔC20,ΔC21,ΔS21,ΔC22,ΔS22].
        let s2 = (self.step2)(t.value());
        coeffs[0].2 += s2[0];
        coeffs[1].2 += s2[1];
        coeffs[1].3 += s2[2];
        coeffs[2].2 += s2[3];
        coeffs[2].3 += s2[4];
        let zero = [S::from_f64(0.0); 3];

        // The tidal field carries a forced monopole; the tidal acceleration is the degree-4 field's
        // acceleration minus the monopole −μ r/r³, i.e. plus μ r/r³.
        let a_itrf = match GeoPotential::from_normalized(self.mu_earth, self.re_earth, 4, &coeffs) {
            Some(geo) => {
                let full = geo.acceleration::<S>(S::from_f64(0.0), &rf, &zero);
                let r2 = rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2];
                let mu_inv_r3 = S::from_f64(self.mu_earth) * (r2 * r2.sqrt()).recip();
                [
                    full[0] + rf[0] * mu_inv_r3,
                    full[1] + rf[1] * mu_inv_r3,
                    full[2] + rf[2] * mu_inv_r3,
                ]
            }
            None => zero,
        };
        // a_gcrf = Mᵀ · a_itrf
        [
            e(0, 0) * a_itrf[0] + e(1, 0) * a_itrf[1] + e(2, 0) * a_itrf[2],
            e(0, 1) * a_itrf[0] + e(1, 1) * a_itrf[1] + e(2, 1) * a_itrf[2],
            e(0, 2) * a_itrf[0] + e(1, 2) * a_itrf[1] + e(2, 2) * a_itrf[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::excessive_precision // verbatim solid_tide.py reference
    )]
    use spacelab_autodiff::DoubleDouble;

    use super::*;

    /// The IERS 2010 Step-2 corrections `[ΔC20, ΔC21, ΔS21, ΔC22, ΔS22]` match Orekit's
    /// `getTideFrequencyDependenceFunction` at 2024-01-01T00:00 UTC (`orekit_tide_step2.csv`), given
    /// the same GMST and Delaunay arguments (from pyerfa). The small residual is the GMST/UT1 source.
    #[test]
    fn step2_matches_orekit() {
        let gmst = 1.747_993_533_381_043_7;
        let delaunay = [
            3.074_748_156_506_035,
            6.227_494_630_588_011,
            2.358_847_296_999_664_3,
            4.116_729_537_850_365,
            -5.918_788_376_630_828,
        ];
        let got = step2_corrections(gmst, delaunay);
        let orekit = [
            2.597_677_943_876_342_7e-11,
            -5.498_848_696_902_133e-10,
            1.662_436_036_735_818_3e-11,
            3.524_413_863_644_123e-13,
            -8.290_916_822_465_907e-13,
        ];
        for i in 0..5 {
            let rel = (got[i] - orekit[i]).abs() / orekit[i].abs();
            assert!(
                rel < 2e-3,
                "component {i}: got {}, want {}, rel {rel:.2e}",
                got[i],
                orekit[i]
            );
        }
    }

    /// At zero tide argument the sums collapse to the amplitude columns of IERS Tables 6.5a/b/c.
    ///
    /// With every Delaunay argument zero and `GMST = −π`, `γ = GMST + π` is exactly `0.0`, so
    /// `θ_f = 0` for every row of all three bands: `sin θ = 0`, `cos θ = 1`, and each constituent
    /// contributes its own tabulated amplitude, unrotated. In units of 10⁻¹²,
    ///
    /// * ΔC̄20 = Σ_lp A^ip = 10.1, ΔC̄21 = Σ_di A^op = −30.5, ΔS̄21 = Σ_di A^ip = 449.2,
    /// * ΔC̄22 = Σ_sd A^ip = −1.5, ΔS̄22 = −Σ_sd A^op = 0,
    ///
    /// the sums taken from `tide_step2_tables.rs` by an exact-decimal Python parse. That pins which
    /// trig factor each band's in- and out-of-phase column carries and with which sign — including
    /// the diurnal band's swap, where ΔC̄21 takes the *out*-of-phase column and ΔS̄21 the in-phase one.
    /// The last line is exact rather than small, and is also why the semidiurnal loop's out-of-phase
    /// arithmetic is unreachable: Table 6.5c has no out-of-phase amplitude at all.
    #[test]
    fn at_zero_tide_argument_step2_is_the_amplitude_columns() {
        let got = step2_corrections(-core::f64::consts::PI, [0.0; 5]);
        for (i, want) in [10.1e-12, -30.5e-12, 449.2e-12, -1.5e-12]
            .into_iter()
            .enumerate()
        {
            let rel = (got[i] - want).abs() / want.abs();
            assert!(
                rel < 1e-13,
                "component {i}: got {}, want {want}, rel {rel:.2e}",
                got[i]
            );
        }
        assert!(
            got[4].abs() < f64::MIN_POSITIVE,
            "ΔS̄22 vanishes identically at zero argument, got {}",
            got[4]
        );
    }

    /// The solid Earth pole tide (§6.4) `[ΔC̄21, ΔS̄21]` matches Orekit's `getSolidPoleTide` at the
    /// oracle epoch (2024, post-2010 linear mean pole), given the same polar motion and
    /// seconds-since-J2000 (`orekit_solid_tide.csv`).
    ///
    /// Orekit 13 measures the wobble from the 2010 printing's Table 7.7, so this fixture pins
    /// [`MeanPole::Iers2010`] and nothing else: it is agreement with Orekit's reading of a
    /// superseded convention. What the workspace *ships* is the secular pole, pinned against the
    /// Conventions' own coefficients in [`the_secular_pole_is_the_2018_chapter_7_line`].
    #[test]
    fn pole_tide_matches_orekit() {
        let got = pole_tide_corrections(
            6.636_808_406_180_881e-07,
            9.802_205_411_513_153e-07,
            7.573_392_691_84e08,
            MeanPole::Iers2010,
        );
        let orekit = [9.026_860_837_280_10e-11, -1.898_399_277_381_072_2e-10];
        for i in 0..2 {
            let rel = (got[i] - orekit[i]).abs() / orekit[i].abs();
            assert!(
                rel < 1e-9,
                "component {i}: got {}, want {}, rel {rel:.2e}",
                got[i],
                orekit[i]
            );
        }
    }

    /// The OCEAN pole tide (§6.5) `[ΔC̄21, ΔS̄21]` matches Orekit's `getOceanPoleTide` at the oracle
    /// epoch, given the same polar motion (`orekit_solid_tide.csv`). Same caveat as
    /// [`pole_tide_matches_orekit`]: it pins [`MeanPole::Iers2010`], because that is the pole Orekit
    /// measures from.
    #[test]
    fn ocean_pole_tide_matches_orekit() {
        let got = ocean_pole_tide_corrections(
            6.636_808_406_180_881e-07,
            9.802_205_411_513_153e-07,
            7.573_392_691_84e08,
            MeanPole::Iers2010,
        );
        let orekit = [1.563_409_711_250_010_3e-11, -2.480_573_723_029_982e-11];
        for i in 0..2 {
            let rel = (got[i] - orekit[i]).abs() / orekit[i].abs();
            assert!(
                rel < 1e-9,
                "component {i}: got {}, want {}, rel {rel:.2e}",
                got[i],
                orekit[i]
            );
        }
    }

    /// `pole_wobble` against `derivations/pole_wobble.py`: IERS 2010 Table 7.7 at 50 digits in the
    /// form the Conventions print it — milliarcseconds, a polynomial in *years*, direct powers —
    /// versus the code's Horner in *seconds* with the mas→radian factor pre-divided once per power.
    ///
    /// Both Orekit pole-tide oracles above sit at one epoch, 2024, which takes the linear branch, so
    /// the whole pre-2010 cubic — both Horner nests, the `F2`/`F3` scale factors and the changeover
    /// comparison — was never executed. Two of the epochs here sit one second apart across that
    /// changeover, where the branches meet in `x̄p` exactly (99.654 mas at t = 10 yr) but step by
    /// 0.001 mas in `ȳp`; that 3e-6 discontinuity is the only observable of `t <= T_CHANGE`.
    #[test]
    fn the_mean_pole_is_iers_table_7_7_on_both_branches() {
        // Same polar motion as the Orekit fixture. At `xp = yp = 0` the wobble is the mean pole
        // alone (`m1 = −x̄p`, `m2 = ȳp`) and `mean_y − yp` cannot be told from `mean_y + yp`; the
        // offset column is what pins the two subtractions and their opposite signs.
        const XP: f64 = 6.636_808_406_180_881e-7;
        const YP: f64 = 9.802_205_411_513_153e-7;
        // t, then (m1, m2) at zero polar motion, then (m1, m2) at (XP, YP).
        const GOLDEN: [(f64, f64, f64, f64, f64); 7] = [
            // 1995.25: t < 0, so the odd powers of the cubic change sign.
            (
                -1.5e8,
                -2.458_415_752_418_707_340_060_364e-7,
                1.626_613_665_781_185_004_592_947e-6,
                4.178_392_653_762_173_659_939_636e-7,
                6.463_931_246_298_697_045_929_471e-7,
            ),
            // 2000.0: the constant terms alone.
            (
                0.0,
                -2.713_696_098_642_516_806_4e-7,
                1.679_132_791_975_633_554_56e-6,
                3.923_112_307_538_364_193_6e-7,
                6.989_122_508_243_182_545_6e-7,
            ),
            (
                1.0e8,
                -3.094_433_320_680_529_097_174_069e-7,
                1.701_262_946_787_141_202_976_331e-6,
                3.542_375_085_500_351_902_825_931e-7,
                7.210_424_056_358_259_029_763_31e-7,
            ),
            (
                3.0e8,
                -4.653_786_041_310_494_545_982_395e-7,
                1.710_823_103_953_267_479_077_48e-6,
                1.983_022_364_870_386_454_017_605e-7,
                7.306_025_628_019_521_790_774_798e-7,
            ),
            // 2010.0 exactly: the last epoch on the cubic branch.
            (
                315_576_000.0,
                -4.831_362_257_728_970_054_4e-7,
                1.709_477_280_276_279_412_8e-6,
                1.805_446_148_451_910_945_6e-7,
                7.292_567_391_249_641_128e-7,
            ),
            // One second later: the linear extension, and `ȳp` a thousandth of a mas away.
            (
                315_576_001.0,
                -4.831_362_269_426_373_688_821_243e-7,
                1.709_472_432_042_882_277_527_724e-6,
                1.805_446_136_754_507_311_178_757e-7,
                7.292_518_908_915_669_775_277_236e-7,
            ),
            // 2024.0, the epoch the two Orekit oracles use.
            (
                7.573_392_691_84e8,
                -9.998_845_528_235_701_381_158_537e-7,
                1.666_804_267_390_284_802_840_335e-6,
                -3.362_037_122_054_820_381_158_537e-7,
                6.865_837_262_389_695_028_403_35e-7,
            ),
        ];

        let mut worst = 0.0_f64;
        for &(t, m1_zero, m2_zero, m1_offset, m2_offset) in &GOLDEN {
            for (label, xp, yp, want) in [
                ("zero", 0.0, 0.0, [m1_zero, m2_zero]),
                ("fixture", XP, YP, [m1_offset, m2_offset]),
            ] {
                let (m1, m2) = pole_wobble(xp, yp, t, MeanPole::Iers2010);
                for (i, (got, want)) in [m1, m2].into_iter().zip(want).enumerate() {
                    let rel = (got - want).abs() / want.abs();
                    worst = spacelab_testkit::nan_max(worst, rel);
                    assert!(
                        rel < 1e-15,
                        "m{} at t = {t} ({label}): got {got}, want {want}, rel {rel:.3e}",
                        i + 1
                    );
                }
            }
        }
        println!("pole_wobble vs IERS Table 7.7 at 50 dps: {worst:.3e} worst relative");
    }

    /// [`MeanPole::Secular2018`] against `derivations/pole_wobble.py`: Eq. (21) of the 2018/02/01
    /// Chapter 7 update at 50 digits, in milliarcseconds and years, versus the code's Horner in
    /// seconds. Same seven epochs as the Table 7.7 golden, so the two models are read off the same
    /// grid — and unlike that one there is no changeover here, which is the point: a single line
    /// fitted to 1900–2017 rather than a cubic and an unbounded extrapolation of it.
    ///
    /// The default is the model this test pins; `MeanPole::default()` is checked here too, because a
    /// default that silently reverted to the retired pole would leave every other test green.
    #[test]
    fn the_secular_pole_is_the_2018_chapter_7_line() {
        const XP: f64 = 6.636_808_406_180_881e-7;
        const YP: f64 = 9.802_205_411_513_153e-7;
        // t, then (m1, m2) at zero polar motion, then (m1, m2) at (XP, YP).
        const GOLDEN: [(f64, f64, f64, f64, f64); 7] = [
            (
                -1.5e8,
                -2.280_023_546_723_839_421_400_867e-7,
                1.474_094_819_938_770_949_054_681e-6,
                4.356_784_859_457_041_578_599_133e-7,
                4.938_742_787_874_556_490_546_81e-7,
            ),
            (
                0.0,
                -2.666_475_246_102_448e-7,
                1.553_827_847_956_062_88e-6,
                3.970_333_160_078_433e-7,
                5.736_073_068_047_475_8e-7,
            ),
            (
                1.0e8,
                -2.924_109_712_354_853_719_066_089e-7,
                1.606_983_199_967_590_833_963_546e-6,
                3.712_698_693_826_027_280_933_911e-7,
                6.267_626_588_162_755_339_635_46e-7,
            ),
            (
                3.0e8,
                -3.439_378_644_859_665_157_198_266e-7,
                1.713_293_903_990_646_741_890_638e-6,
                3.197_429_761_321_215_842_801_734e-7,
                7.330_733_628_393_314_418_906_381e-7,
            ),
            // 2010.0 and one second later: no changeover, so these differ only in the last digits.
            (
                315_576_000.0,
                -3.479_507_789_323_139_872e-7,
                1.721_573_381_619_962_336e-6,
                3.157_300_616_857_741_128e-7,
                7.413_528_404_686_470_36e-7,
            ),
            (
                315_576_001.0,
                -3.479_507_791_899_484_534_524_057e-7,
                1.721_573_382_151_515_856_115_28e-6,
                3.157_300_614_281_396_465_475_943e-7,
                7.413_528_410_002_005_561_152_795e-7,
            ),
            (
                7.573_392_691_84e8,
                -4.617_642_229_984_516_585_593_736e-7,
                1.956_394_202_412_351_329_979_387e-6,
                2.019_166_176_196_364_414_406_264e-7,
                9.761_736_612_610_360_299_793_874e-7,
            ),
        ];

        let mut worst = 0.0_f64;
        for &(t, m1_zero, m2_zero, m1_offset, m2_offset) in &GOLDEN {
            for (label, xp, yp, want) in [
                ("zero", 0.0, 0.0, [m1_zero, m2_zero]),
                ("fixture", XP, YP, [m1_offset, m2_offset]),
            ] {
                let (m1, m2) = pole_wobble(xp, yp, t, MeanPole::Secular2018);
                assert_eq!(
                    (m1, m2),
                    pole_wobble(xp, yp, t, MeanPole::default()),
                    "MeanPole::default() must be the secular pole"
                );
                for (i, (got, want)) in [m1, m2].into_iter().zip(want).enumerate() {
                    let rel = (got - want).abs() / want.abs();
                    worst = spacelab_testkit::nan_max(worst, rel);
                    assert!(
                        rel < 1e-15,
                        "m{} at t = {t} ({label}): got {got}, want {want}, rel {rel:.3e}",
                        i + 1
                    );
                }
            }
        }
        println!("pole_wobble vs IERS Eq. (21) at 50 dps: {worst:.3e} worst relative");
    }

    /// The two models are not interchangeable, and the pole tide is where it shows.
    ///
    /// They separate at 5.94 mas/yr in `x` and 4.09 mas/yr in `y`, so the epoch decides how bad it
    /// is: 28 mas apart at 2010.0, 111 mas at 2024.0. At the Orekit fixture's polar motion — an
    /// ordinary one, 0.24 arcsec from the pole — that is enough for `ΔC̄21` to come out with the
    /// **opposite sign**, and for `ΔS̄21` to move by 41% of its own magnitude. A test asserting only
    /// "small" or "same order" would pass either way; this one asserts the disagreement, so wiring
    /// the wrong model into a caller cannot go unnoticed.
    #[test]
    fn the_two_mean_poles_disagree_on_the_sign_of_the_pole_tide() {
        const XP: f64 = 6.636_808_406_180_881e-7;
        const YP: f64 = 9.802_205_411_513_153e-7;
        const T2024: f64 = 7.573_392_691_84e8;
        let legacy = pole_tide_corrections(XP, YP, T2024, MeanPole::Iers2010);
        let secular = pole_tide_corrections(XP, YP, T2024, MeanPole::Secular2018);
        assert!(
            legacy[0] > 0.0 && secular[0] < 0.0,
            "ΔC̄21 must flip sign between the models: legacy {}, secular {}",
            legacy[0],
            secular[0]
        );
        let s_shift = (secular[1] - legacy[1]).abs() / legacy[1].abs();
        assert!(
            s_shift > 0.35,
            "ΔS̄21 must move by a large fraction of itself: legacy {}, secular {}, shift {s_shift:.3}",
            legacy[1],
            secular[1]
        );
        // The ocean pole tide is driven by the same `(m1, m2)`, so it moves with it.
        let ocean_legacy = ocean_pole_tide_corrections(XP, YP, T2024, MeanPole::Iers2010);
        let ocean_secular = ocean_pole_tide_corrections(XP, YP, T2024, MeanPole::Secular2018);
        assert!(
            ocean_legacy[0] > 0.0 && ocean_secular[0] < 0.0,
            "the ocean ΔC̄21 must flip with the solid one: legacy {}, secular {}",
            ocean_legacy[0],
            ocean_secular[0]
        );
    }

    /// The fully-normalized Legendre must match its closed forms (same check as the Python oracle).
    #[test]
    fn legendre_matches_closed_forms() {
        assert!((pbar(2, 0, 0.0) - (-libm::sqrt(5.0) / 2.0)).abs() < 1e-13);
        assert!((pbar(2, 2, 0.0) - (libm::sqrt(15.0) / 2.0)).abs() < 1e-13);
        assert!((pbar(1, 1, 0.0) - libm::sqrt(3.0)).abs() < 1e-13);
        let x = 0.4;
        assert!(
            (pbar(3, 0, x) - libm::sqrt(7.0) / 2.0 * (5.0 * x * x * x - 3.0 * x)).abs() < 1e-12
        );
    }

    struct Fixed(pub [f64; 3]);
    impl Ephemeris for Fixed {
        fn position(&self, _t: f64) -> [f64; 3] {
            self.0
        }
    }

    /// A body at the origin is [`crate::ephemeris::is_absent`]'s sentinel, and raises no tide.
    ///
    /// `delta_cs` divides by `‖r_j‖^{n+1}`, so an absent body left in the sum contributes `inf` and
    /// the whole degree-4 field, and the acceleration through it, comes back `NaN`. Skipping it
    /// instead is what makes the ephemeris contract true here: the Moon's term disappears and the
    /// Sun's stays, exactly as if the tide had been built with one body.
    ///
    /// What does *not* disappear is the permanent-tide subtraction. `ΔC̄₂₀` carries
    /// `−PERMANENT_TIDE_C20` whatever the bodies are doing, because it is bookkeeping against the
    /// zero-tide background of the static field it is added to, not a term either body raises — so
    /// "no bodies" is a small constant acceleration, not zero, and that is the right answer.
    #[test]
    fn a_body_at_the_origin_raises_no_tide() {
        let build = |sun: [f64; 3], moon: [f64; 3]| SolidEarthTide {
            mu_earth: 3.986_004_415e14,
            re_earth: 6_378_136.3,
            mu_sun: 1.327_124_400_18e20,
            mu_moon: 4.902_800_066e12,
            sun: Fixed(sun),
            moon: Fixed(moon),
            rotate: |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            step2: |_t: f64| [0.0; 5],
        };
        let sat = [6.5e6, 2.0e6, 3.0e6];
        let (sun, moon) = ([1.4e11, 0.5e11, 0.3e11], [3.5e8, 1.5e8, 1.0e8]);

        let both = build(sun, moon).acceleration::<f64>(0.0, &sat, &[0.0; 3]);
        let no_moon = build(sun, [0.0; 3]).acceleration::<f64>(0.0, &sat, &[0.0; 3]);
        let neither = build([0.0; 3], [0.0; 3]).acceleration::<f64>(0.0, &sat, &[0.0; 3]);

        for (label, a) in [("no moon", no_moon), ("neither", neither)] {
            assert!(
                a.iter().all(|c| c.is_finite()),
                "{label}: {a:?} is not finite"
            );
        }
        // The Moon raises about two thirds of the total tide, so dropping it has to move the answer
        // by far more than round-off — otherwise the skip could be dropping nothing.
        let moved = (0..3)
            .map(|i| (both[i] - no_moon[i]) * (both[i] - no_moon[i]))
            .sum::<f64>()
            .sqrt();
        assert!(
            moved > 1e-7,
            "dropping the Moon moved the tide by {moved:e}"
        );
    }

    /// The tidal acceleration matches the `derivations/solid_tide.py` reference (which computes it by
    /// a finite difference of the potential — an independent gradient), at the same test geometry.
    #[test]
    fn acceleration_matches_python_reference() {
        let tide = SolidEarthTide {
            mu_earth: 3.986_004_415e14,
            re_earth: 6_378_136.3,
            mu_sun: 1.327_124_400_18e20,
            mu_moon: 4.902_800_066e12,
            sun: Fixed([1.4e11, 0.5e11, 0.3e11]),
            moon: Fixed([3.5e8, 1.5e8, 1.0e8]),
            rotate: |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            step2: |_t: f64| [0.0; 5], // Step-1 only: the Python reference has no frequency dependence
        };
        let sat = [6.5e6, 2.0e6, 3.0e6];
        let a = tide.acceleration::<f64>(0.0, &sat, &[0.0; 3]);
        let reference = [
            -2.768_972_471_867_137_13e-07,
            -5.234_395_306_086_270_24e-08,
            -1.341_338_418_114_901_29e-07,
        ];
        for i in 0..3 {
            let rel = (a[i] - reference[i]).abs() / reference[i].abs();
            assert!(
                rel < 1e-4,
                "axis {i}: got {}, want {}, rel {rel:.2e}",
                a[i],
                reference[i]
            );
        }
    }

    /// The tidal acceleration matches Orekit's `SolidTides` (IERS 2010) at a fixed geometry Orekit
    /// emitted (`tools/orekit-oracle`, `orekit_solid_tide.csv`) — the independent third-party check.
    /// With the Step-2 frequency-dependent corrections and the solid Earth pole tide (§6.4) wired in
    /// (same epoch as `step2_matches_orekit`: the emitted Sun in ITRF equals the third-body Sun in
    /// GCRF rotated by GMST = 1.74799…), spacelab applies the full IERS 2010 degree-2 tide
    /// (Step 1 + Step 2 + pole tide) and matches Orekit's default config to round-off (~4e-9),
    /// closing the former ~5.7 % pole-tide gap.
    #[test]
    fn acceleration_matches_orekit() {
        // Step-2 corrections at the oracle epoch (GMST + Delaunay from `step2_matches_orekit`).
        let gmst = 1.747_993_533_381_043_7;
        let delaunay = [
            3.074_748_156_506_035,
            6.227_494_630_588_011,
            2.358_847_296_999_664_3,
            4.116_729_537_850_365,
            -5.918_788_376_630_828,
        ];
        let tide = SolidEarthTide {
            mu_earth: 3.986_004_415e14,
            re_earth: 6.378_136_46e6,
            mu_sun: 1.327_124_400_412_794_6e20,
            mu_moon: 4.902_800_118_457_551e12,
            sun: Fixed([
                -135_337_748_175.940_03,
                -1_832_917_823.254_058_8,
                -57_614_487_601.187_485,
            ]),
            moon: Fixed([
                203_571_377.964_981_02,
                338_362_165.014_877_2,
                88_464_793.006_448_67,
            ]),
            rotate: |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            // Full IERS 2010 degree-2 tide: Step 2 + the solid Earth pole tide (§6.4), the latter
            // added to the same ΔC̄21/ΔS̄21 slots (xp, yp, t from `orekit_solid_tide.csv`), and
            // measured from the pole Orekit measures from rather than the one this crate defaults to.
            step2: move |_t: f64| {
                let pole = pole_tide_corrections(
                    6.636_808_406_180_881e-07,
                    9.802_205_411_513_153e-07,
                    7.573_392_691_84e08,
                    MeanPole::Iers2010,
                );
                let mut s = step2_corrections(gmst, delaunay);
                s[1] += pole[0]; // ΔC̄21
                s[2] += pole[1]; // ΔS̄21
                s
            },
        };
        let sat = [
            860_234.487_436_545,
            -6_739_356.482_322_041,
            3_015_140.267_186_367,
        ];
        let a = tide.acceleration::<f64>(0.0, &sat, &[0.0; 3]);
        // Orekit's `SolidTides` WITH the pole tide (its default `poleTide=true`) — spacelab now applies
        // the full IERS 2010 degree-2 tide: Step 1 + Step 2 + the solid Earth pole tide, closing the
        // former ~5.7% pole-tide gap.
        let orekit = [
            -5.007_455_277_686_382_4e-08,
            -7.576_258_161_787_424e-08,
            2.564_281_013_490_208_3e-08,
        ];
        let mag = libm::sqrt(orekit.iter().map(|v| v * v).sum());
        let diff = libm::sqrt((0..3).map(|i| (a[i] - orekit[i]).powi(2)).sum());
        let rel = diff / mag;
        std::eprintln!("solid tide vs Orekit (full IERS 2010, with pole tide): rel diff {rel:.3e}");
        assert!(
            rel < 0.01,
            "spacelab full IERS tide should match Orekit's with-pole config to < 1%, got {rel:.2e}"
        );
    }

    const IDENTITY: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const MU_EARTH: f64 = 3.986_004_415e14;
    const RE_EARTH: f64 = 6_378_136.3;
    const SUN: [f64; 3] = [1.4e11, 0.5e11, 0.3e11];
    const MOON: [f64; 3] = [3.5e8, 1.5e8, 1.0e8];
    const SAT: [f64; 3] = [6.5e6, 2.0e6, 3.0e6];

    fn norm(v: [f64; 3]) -> f64 {
        libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }

    /// A ZXZ rotation whose every entry is well away from 0 and ±1, so each of the nine products and
    /// six sums in a frame transform has an observable effect of its own.
    fn general_rotation() -> [[f64; 3]; 3] {
        let (alpha, beta, gamma) = (0.7_f64, 0.4_f64, 1.1_f64);
        let (ca, sa) = (libm::cos(alpha), libm::sin(alpha));
        let (cb, sb) = (libm::cos(beta), libm::sin(beta));
        let (cg, sg) = (libm::cos(gamma), libm::sin(gamma));
        [
            [cg * ca - sg * cb * sa, cg * sa + sg * cb * ca, sg * sb],
            [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
            [sb * sa, -sb * ca, cb],
        ]
    }

    /// `M · v`, written as the definition rather than as the code's unrolled expression.
    fn apply(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|i| (0..3).map(|j| m[i][j] * v[j]).sum())
    }

    /// `Mᵀ · v`.
    fn apply_transpose(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        [0, 1, 2].map(|i| (0..3).map(|j| m[j][i] * v[j]).sum())
    }

    /// The Earth orientation must be transparent: rotating the inputs into the ITRF and rotating the
    /// answer back out is the same computation, whichever side of the force does the rotating.
    ///
    /// Both oracles above pass the identity, where every off-diagonal `e(i, j)` is an exact zero and
    /// every diagonal one an exact one — and `x · 1 ≡ x / 1`, `x + 0 ≡ x − 0`. That leaves all three
    /// frame transforms (the satellite in, the Sun and Moon in, the acceleration out) unassertable:
    /// 42 of this function's mutants live in those nine lines.
    #[test]
    fn a_general_earth_orientation_is_transparent() {
        let m = general_rotation();
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[k][i] * m[k][j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-15, "MᵀM at ({i},{j}) is {dot}");
                assert!(m[i][j].abs() > 0.1, "entry ({i},{j}) hides its operator");
            }
        }
        // A Step-2 vector that is nowhere near zero, so the degree-2 band it feeds rotates too.
        let s2 = [3.1e-11, -1.7e-11, 8.3e-12, 4.5e-12, -9.2e-12];

        let inertial = SolidEarthTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            mu_sun: 1.327_124_400_18e20,
            mu_moon: 4.902_800_066e12,
            sun: Fixed(SUN),
            moon: Fixed(MOON),
            rotate: move |_t: f64| m,
            step2: move |_t: f64| s2,
        };
        let got = inertial.acceleration::<f64>(0.0, &SAT, &[0.0; 3]);

        // The same physics with the rotation done outside: everything already in the ITRF.
        let in_frame = SolidEarthTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            mu_sun: 1.327_124_400_18e20,
            mu_moon: 4.902_800_066e12,
            sun: Fixed(apply(m, SUN)),
            moon: Fixed(apply(m, MOON)),
            rotate: |_t: f64| IDENTITY,
            step2: move |_t: f64| s2,
        };
        let want = apply_transpose(
            m,
            in_frame.acceleration::<f64>(0.0, &apply(m, SAT), &[0.0; 3]),
        );

        let rel = norm([got[0] - want[0], got[1] - want[1], got[2] - want[2]]) / norm(want);
        println!("solid tide, rotated vs pre-rotated: {rel:.3e} relative");
        // Loose by the standard of this file: subtracting the monopole cancels eight digits, so a
        // reassociation of the same arithmetic already costs ~1e-9. Every mutation here moves the
        // answer by a fraction of itself, not by an ulp.
        assert!(
            rel < 1e-9,
            "the Earth orientation is not transparent: {rel:.3e}"
        );
    }

    /// Each Step-2 slot must reach the coefficient the layout comment names, with a `+`.
    ///
    /// The Python oracle passes `[0.0; 5]`, where `+=` and `-=` are the same operation; the Orekit
    /// oracle's 1 % bar is set by the pole tide in ΔC̄21/ΔS̄21, and the other three Step-2 terms —
    /// 2.6e-11, 3.5e-13, −8.3e-13 — sit four to six orders below it, so their sign is invisible. Here
    /// each slot is driven on its own, and the expected move is the acceleration of a degree-2 field
    /// carrying that one coefficient: an independent construction that pins the slot, the degree and
    /// order it must land on, and the sign it must arrive with. Double-double throughout, because the
    /// monopole subtraction would otherwise leave only eight digits of the answer — eight orders of
    /// margin that the `+=` under test then spends none of.
    #[test]
    fn each_step2_slot_reaches_its_own_coefficient() {
        const DELTA: f64 = 1e-8; // ~10³ the real Step-2 terms: a move comparable to the tide itself
        let sat = SAT.map(DoubleDouble::from_f64);
        let zero = [DoubleDouble::from_f64(0.0); 3];
        let tide = |s2: [f64; 5]| SolidEarthTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            mu_sun: 1.327_124_400_18e20,
            mu_moon: 4.902_800_066e12,
            sun: Fixed(SUN),
            moon: Fixed(MOON),
            rotate: |_t: f64| IDENTITY,
            step2: move |_t: f64| s2,
        };
        let base =
            tide([0.0; 5]).acceleration::<DoubleDouble>(DoubleDouble::from_f64(0.0), &sat, &zero);

        // `from_normalized` carries an implicit C̄₀₀ = 1, so the reference field is read as a
        // difference against the bare point mass. Subtracting `−μ r/r³` in closed form instead would
        // leave the normalization ratios' own f64 round-off — 6e-16 of a 7 m/s² monopole, which is
        // 1e-8 of the answer, eight orders above what the rest of this comparison resolves.
        let monopole = GeoPotential::from_normalized(MU_EARTH, RE_EARTH, 2, &[])
            .expect("degree 2 is within MAX_DEGREE")
            .acceleration::<DoubleDouble>(DoubleDouble::from_f64(0.0), &sat, &zero);

        // slot k of `[ΔC̄20, ΔC̄21, ΔS̄21, ΔC̄22, ΔS̄22]` → the (n, m) and the half it belongs to.
        let slots = [
            (2_usize, 0_usize, true),
            (2, 1, true),
            (2, 1, false),
            (2, 2, true),
            (2, 2, false),
        ];
        let mut worst = 0.0_f64;
        for (k, &(n, m, is_cosine)) in slots.iter().enumerate() {
            let mut s2 = [0.0; 5];
            s2[k] = DELTA;
            let driven =
                tide(s2).acceleration::<DoubleDouble>(DoubleDouble::from_f64(0.0), &sat, &zero);

            let (cbar, sbar) = if is_cosine {
                (DELTA, 0.0)
            } else {
                (0.0, DELTA)
            };
            let field = GeoPotential::from_normalized(MU_EARTH, RE_EARTH, 2, &[(n, m, cbar, sbar)])
                .expect("degree 2 is within MAX_DEGREE");
            let full = field.acceleration::<DoubleDouble>(DoubleDouble::from_f64(0.0), &sat, &zero);

            // Every difference is taken in double-double and only the residual is rounded to f64;
            // rounding `moved` and `want` first would put the comparison back on an f64 floor.
            let moved = [0, 1, 2].map(|i| driven[i] - base[i]);
            let want = [0, 1, 2].map(|i| full[i] - monopole[i]);
            let magnitude = norm(want.map(Scalar::value));
            assert!(
                magnitude > 0.1 * norm(base.map(Scalar::value)),
                "slot {k} barely moves the answer: {magnitude:.3e} against {:.3e}",
                norm(base.map(Scalar::value))
            );
            let rel = norm([0, 1, 2].map(|i| (moved[i] - want[i]).value())) / magnitude;
            worst = spacelab_testkit::nan_max(worst, rel);
            // The floor is the `+=` under test: it lands on an f64 coefficient, so the increment the
            // field actually sees is `fl(C̄ + δ) − C̄`, which is δ only to half an ulp of their sum.
            assert!(
                rel < 1e-14,
                "slot {k} does not land on C̄{n}{m}/S̄{n}{m}: {rel:.3e} relative"
            );
        }
        println!("Step-2 slots vs their own degree-2 fields: {worst:.3e} worst relative");
    }

    /// The IERS Table 6.5 Step-2 constituents, sealed.
    ///
    /// Step-2 contributes at the `1e-12` level, so a sign deleted from one of the 71 constituents
    /// moves the acceleration far below what any tide oracle here asserts — `cargo mutants` finds
    /// 24 such deletions the suite cannot see. The constant comes from
    /// `derivations/table_seals.py`, which reads the generated file with Python's float parser — a
    /// different parse of the same text than rustc's.
    #[test]
    fn the_step2_constituents_are_the_ones_the_generator_wrote() {
        let flat: Vec<f64> = [&DIURNAL[..], &LONG_PERIOD[..], &SEMIDIURNAL[..]]
            .into_iter()
            .flatten()
            .flat_map(|&(tau, args, ip, op)| {
                [f64::from(tau)]
                    .into_iter()
                    .chain(args.map(f64::from))
                    .chain([ip, op])
            })
            .collect();
        assert_eq!(flat.len(), (48 + 21 + 2) * 8, "constituent count moved");
        assert_eq!(
            spacelab_testkit::fnv1a_f64(flat),
            0xc427_0698_da3a_cc62,
            "IERS 2010 Table 6.5 amplitudes moved"
        );
    }
}
