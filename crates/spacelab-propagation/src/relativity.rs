//! IERS 2010 relativistic corrections to the satellite equation of motion (Conventions Ch. 10,
//! Eq. 10.12).
//!
//! Three post-Newtonian accelerations, each a peer [`Acceleration`] evaluated in the geocentric
//! celestial frame (GCRF) — the frame the propagation state already lives in, so unlike the
//! geopotential and the solid tide these need no body-fixed rotation:
//!
//! * [`Schwarzschild`] — the dominant static-field 1/c² correction (~10⁻⁸ m/s² at LEO), the
//!   relativistic analogue of the perihelion precession;
//! * [`LenseThirring`] — frame dragging by the Earth's rotation (~10⁻¹⁰ m/s²);
//! * [`DeSitter`] — geodesic (de Sitter) precession from the Earth's orbit about the Sun (~10⁻¹¹ m/s²).
//!
//! All three use the general-relativity PPN values β = γ = 1. Compose them (and with the other forces)
//! through the tuple [`Acceleration`] impls; there is deliberately no combined god-struct, matching
//! Orekit's split into `Relativity`, `LenseThirringRelativity`, `DeSitterRelativity`, against which
//! each term is validated (`tools/orekit-oracle`, `orekit_relativity.csv`).
//!
//! # Provenance
//!
//! Formula: IERS Conventions 2010, Eq. 10.12. Earth angular momentum per unit mass J = 9.8×10⁸ m²/s
//! (IERS 2010, matching Orekit's `LenseThirringRelativity`). Speed of light is the exact SI value.
//! Cross-checked term-by-term in `derivations/relativity.py`.

// Post-Newtonian vector algebra: single-letter physics names (r, v, J, u) and GM_S/GM_E doc notation.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown
)]

use spacelab_autodiff::Scalar;

use crate::forces::Acceleration;

/// Speed of light in vacuum, m/s (exact, SI definition).
const C_LIGHT: f64 = 299_792_458.0;

/// Earth's angular momentum per unit mass, m²/s, directed along the celestial pole (GCRF +Z).
/// IERS Conventions 2010, Ch. 10 (also Orekit's `LenseThirringRelativity`).
pub const EARTH_ANGULAR_MOMENTUM_PER_MASS: f64 = 9.8e8;

/// Vector cross product, over any [`Scalar`].
fn cross<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Schwarzschild relativistic correction (IERS 2010 Eq. 10.12, first line, β = γ = 1):
/// `a = GM/(c² r³) · { [4 GM/r − v·v] r + 4 (r·v) v }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Schwarzschild {
    /// Earth gravitational parameter, m³/s².
    pub mu: f64,
}

impl Acceleration for Schwarzschild {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let rn = r2.sqrt();
        let inv_r3 = (r2 * rn).recip();
        let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
        let mu = S::from_f64(self.mu);
        let inv_c2 = S::from_f64((C_LIGHT * C_LIGHT).recip());
        let pre = mu * inv_r3 * inv_c2; // GM/(c² r³)
        let k = S::from_f64(4.0) * mu * rn.recip() - vv; // 4 GM/r − v·v
        let four = S::from_f64(4.0);
        [
            pre * (k * r[0] + four * rv * v[0]),
            pre * (k * r[1] + four * rv * v[1]),
            pre * (k * r[2] + four * rv * v[2]),
        ]
    }
}

/// Lense–Thirring frame-dragging correction (IERS 2010 Eq. 10.12, second line, γ = 1):
/// `a = 2 GM/(c² r³) · { (3/r²)(r×v)(r·J) + (v×J) }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LenseThirring {
    /// Earth gravitational parameter, m³/s².
    pub mu: f64,
    /// Earth angular momentum per unit mass, m²/s, in the GCRF (≈ `[0, 0, 9.8e8]`; see
    /// [`EARTH_ANGULAR_MOMENTUM_PER_MASS`]).
    pub j: [f64; 3],
}

impl Acceleration for LenseThirring {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let rn = r2.sqrt();
        let inv_r3 = (r2 * rn).recip();
        let inv_r2 = r2.recip();
        let j = [
            S::from_f64(self.j[0]),
            S::from_f64(self.j[1]),
            S::from_f64(self.j[2]),
        ];
        let rxv = cross(r, v);
        let rj = r[0] * j[0] + r[1] * j[1] + r[2] * j[2];
        let vxj = cross(v, &j);
        let mu = S::from_f64(self.mu);
        let inv_c2 = S::from_f64((C_LIGHT * C_LIGHT).recip());
        let pre = S::from_f64(2.0) * mu * inv_r3 * inv_c2; // 2 GM/(c² r³)
        let three = S::from_f64(3.0);
        [
            pre * (three * inv_r2 * rxv[0] * rj + vxj[0]),
            pre * (three * inv_r2 * rxv[1] * rj + vxj[1]),
            pre * (three * inv_r2 * rxv[2] * rj + vxj[2]),
        ]
    }
}

/// de Sitter (geodesic) precession (IERS 2010 Eq. 10.12, third line, γ = 1):
/// `a = 3 · { [v_E × (−GM_S R/(c² R³))] × v }`, where `R` is the Earth's position relative to the Sun
/// and `v_E = Ṙ` its velocity. Supplied through a closure giving the *geocentric* Sun position and
/// velocity `(r_sun, v_sun)`; then `R = −r_sun`, `v_E = −v_sun`, so `−GM_S R/(c² R³) = +GM_S
/// r_sun/(c² |r_sun|³)`.
pub struct DeSitter<F> {
    /// Sun gravitational parameter, m³/s².
    pub mu_sun: f64,
    /// Geocentric Sun position (m) and velocity (m/s) at time `t`, seconds from the propagation epoch.
    pub sun_pv: F,
}

impl<F: Fn(f64) -> ([f64; 3], [f64; 3])> Acceleration for DeSitter<F> {
    fn acceleration<S: Scalar>(&self, t: S, _r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let (r_sun, v_sun) = (self.sun_pv)(t.value());
        let rsn = libm::sqrt(r_sun[0] * r_sun[0] + r_sun[1] * r_sun[1] + r_sun[2] * r_sun[2]);
        let u_scale = self.mu_sun / (C_LIGHT * C_LIGHT * rsn * rsn * rsn);
        let u = [
            S::from_f64(u_scale * r_sun[0]),
            S::from_f64(u_scale * r_sun[1]),
            S::from_f64(u_scale * r_sun[2]),
        ];
        let v_e = [
            S::from_f64(-v_sun[0]),
            S::from_f64(-v_sun[1]),
            S::from_f64(-v_sun[2]),
        ];
        let a = cross(&cross(&v_e, &u), v);
        let three = S::from_f64(3.0);
        [three * a[0], three * a[1], three * a[2]]
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)] // verbatim derivations/relativity.py golden vectors
    use super::*;

    const R: [f64; 3] = [7_000e3, 1_000e3, 2_000e3];
    const V: [f64; 3] = [-1_000.0, 6_500.0, 3_500.0];
    const R_SUN: [f64; 3] = [1.35e11, -5.0e10, -2.2e10];
    const V_SUN: [f64; 3] = [11_000.0, 25_000.0, 10_800.0];
    const MU_EARTH: f64 = 3.986_004_415e14;
    const MU_SUN: f64 = 1.327_124_400_18e20;

    fn mag(a: [f64; 3]) -> f64 {
        libm::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    }

    fn assert_close(a: [f64; 3], want: [f64; 3], tol: f64) {
        for i in 0..3 {
            let rel = (a[i] - want[i]).abs() / want[i].abs();
            assert!(
                rel < tol,
                "axis {i}: got {}, want {}, rel {rel:.2e}",
                a[i],
                want[i]
            );
        }
    }

    /// The three terms sit in their expected LEO magnitude bands (Schwarzschild dominant).
    #[test]
    fn magnitudes_are_physical() {
        let s = Schwarzschild { mu: MU_EARTH }.acceleration::<f64>(0.0, &R, &V);
        let lt = LenseThirring {
            mu: MU_EARTH,
            j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
        }
        .acceleration::<f64>(0.0, &R, &V);
        let ds = DeSitter {
            mu_sun: MU_SUN,
            sun_pv: |_t| (R_SUN, V_SUN),
        }
        .acceleration::<f64>(0.0, &R, &V);
        assert!(
            (3e-9..3e-8).contains(&mag(s)),
            "Schwarzschild {:.3e}",
            mag(s)
        );
        assert!(
            (5e-11..5e-10).contains(&mag(lt)),
            "Lense-Thirring {:.3e}",
            mag(lt)
        );
        assert!(
            (1e-11..1e-10).contains(&mag(ds)),
            "de Sitter {:.3e}",
            mag(ds)
        );
    }

    /// Each term matches its `derivations/relativity.py` closed-form golden vector to round-off.
    #[test]
    fn matches_python_reference() {
        let s = Schwarzschild { mu: MU_EARTH }.acceleration::<f64>(0.0, &R, &V);
        let lt = LenseThirring {
            mu: MU_EARTH,
            j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
        }
        .acceleration::<f64>(0.0, &R, &V);
        let ds = DeSitter {
            mu_sun: MU_SUN,
            sun_pv: |_t| (R_SUN, V_SUN),
        }
        .acceleration::<f64>(0.0, &R, &V);
        assert_close(
            s,
            [
                1.234_214_137_045_129_14e-08,
                3.693_503_855_429_151_67e-09,
                4.626_412_799_745_371_90e-09,
            ],
            1e-12,
        );
        assert_close(
            lt,
            [
                1.192_656_478_917_165_39e-10,
                -4.259_487_424_704_163_97e-11,
                1.131_806_658_564_249_04e-10,
            ],
            1e-12,
        );
        assert_close(
            ds,
            [
                -4.512_367_092_218_367_00e-11,
                -5.679_451_310_348_743_49e-12,
                -2.344_924_972_833_383_46e-12,
            ],
            1e-12,
        );
    }

    /// The same Lense-Thirring closed form on a spin axis that is *not* the GCRF z axis. The axial
    /// `J = [0, 0, 9.8e8]` every case above uses makes `r·J` a sum of two exact zeros and the third
    /// term, and makes `(v × J)_z = v_x J_y − v_y J_x` an exact zero, so three of the sum's
    /// operations carry no information at all under it — yet `j` is a free vector of the API.
    #[test]
    fn lense_thirring_matches_python_reference_off_axis() {
        let lt = LenseThirring {
            mu: MU_EARTH,
            j: [2.7e8, -4.1e8, 8.9e8],
        }
        .acceleration::<f64>(0.0, &R, &V);
        assert_close(
            lt,
            [
                1.229_290_554_435_757_97e-10,
                -6.626_421_253_125_777_02e-11,
                1.581_846_962_562_146_58e-10,
            ],
            1e-12,
        );
    }

    /// Third-party validation against Orekit 13's `Relativity` / `LenseThirringRelativity` /
    /// `DeSitterRelativity` at a fixed GCRF geometry (`tools/orekit-oracle`,
    /// `orekit_relativity.csv`). Schwarzschild and Lense-Thirring match Orekit to round-off (same
    /// formula, same J = [0,0,9.8e8] since Orekit is given the GCRF as the body frame).
    ///
    /// For de Sitter the force is driven by the exact Earth-relative-to-Sun PV that Orekit's model
    /// consumes; the two implementations then agree to round-off. Note Orekit reads that PV in the
    /// Sun's *body-inertial* frame (pole at Dec ≈ +63.9°), whereas the production force uses the
    /// GCRS-aligned geocentric Sun per IERS 2010 — the two differ only by the Sun's pole tilt, a
    /// negligible rotation of a ~4×10⁻¹¹ m/s² term.
    #[test]
    fn matches_orekit() {
        let mu = 3.986_004_415e14;
        let mu_sun = 1.327_124_400_412_794_6e20;
        let r = [7_000_000.0, 1_200_000.0, 2_000_000.0];
        let v = [-1_500.0, 6_500.0, 3_200.0];

        let s = Schwarzschild { mu }.acceleration::<f64>(0.0, &r, &v);
        assert_close(
            s,
            [
                1.222_560_616_623_010_5e-08,
                3.200_008_711_459_487_6e-09,
                4.085_978_303_461_707_0e-09,
            ],
            1e-12,
        );

        let lt = LenseThirring {
            mu,
            j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
        }
        .acceleration::<f64>(0.0, &r, &v);
        assert_close(
            lt,
            [
                1.188_182_261_915_046_0e-10,
                -2.812_044_603_809_647_3e-11,
                1.128_156_995_421_512_4e-10,
            ],
            1e-12,
        );

        // Earth relative to Sun, exactly as Orekit's de Sitter reads it. The force takes the
        // geocentric Sun (r_sun, v_sun) and internally sets R = −r_sun, v_E = −v_sun, so pass the
        // negation to make it use Orekit's R, v_E verbatim.
        let r_es = [
            13_122_691_835.549_543,
            146_321_465_681.630_86,
            -7_544_434_398.738_96,
        ];
        let v_es = [-29_973.536_593_835, 2_488.441_675_955, -3_490.902_679_888];
        let ds = DeSitter {
            mu_sun,
            sun_pv: move |_t| {
                (
                    [-r_es[0], -r_es[1], -r_es[2]],
                    [-v_es[0], -v_es[1], -v_es[2]],
                )
            },
        }
        .acceleration::<f64>(0.0, &r, &v);
        assert_close(
            ds,
            [
                -3.875_708_605_574_370_0e-11,
                -7.032_298_000_337_574_0e-12,
                -3.883_028_775_444_161_0e-12,
            ],
            1e-10,
        );
    }
}
