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
//! Orekit's split into `Relativity`, `LenseThirringRelativity`, `DeSitterRelativity`, which are a
//! comparator (`docs/comparisons.md`).
//!
//! # Provenance
//!
//! Formula: IERS Conventions 2010, Eq. 10.12. Earth angular momentum per unit mass J = 9.8×10⁸ m²/s
//! (IERS 2010, matching Orekit's `LenseThirringRelativity`). Speed of light is the exact SI value.
//! `derivations/relativity_iers.mac` holds each term to its line of Eq. 10.12, derives Einstein's
//! perigee advance, the Lense-Thirring node and perigee rates and the geodesic precession from them,
//! recovers the chapter's 0.8, 180 and 19 mas/y, and evaluates the test goldens at 40 digits;
//! `derivations/relativity.py`'s vectors are held to it there.

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
    // Verbatim derivations/relativity.py golden vectors and relativity_iers.mac's 40-digit output.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]
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

    /// The geometry of the Orekit oracle (`tools/orekit-oracle`, `orekit_relativity.csv`): a GCRF
    /// satellite, and the Earth's position and velocity about the Sun as Orekit's de Sitter reads
    /// them, which eq. (10.12) calls `R` and `Ṙ`.
    const ORACLE_MU_SUN: f64 = 1.327_124_400_412_794_6e20;
    const ORACLE_R: [f64; 3] = [7_000_000.0, 1_200_000.0, 2_000_000.0];
    const ORACLE_V: [f64; 3] = [-1_500.0, 6_500.0, 3_200.0];
    const EARTH_FROM_SUN_R: [f64; 3] = [
        13_122_691_835.549_543,
        146_321_465_681.630_86,
        -7_544_434_398.738_96,
    ];
    const EARTH_FROM_SUN_V: [f64; 3] = [-29_973.536_593_835, 2_488.441_675_955, -3_490.902_679_888];

    /// Orekit 13's `Relativity`, `LenseThirringRelativity` and `DeSitterRelativity` at the oracle
    /// geometry.
    const OREKIT_TERMS: [[f64; 3]; 3] = [
        [
            1.222_560_616_623_010_5e-08,
            3.200_008_711_459_487_6e-09,
            4.085_978_303_461_707_0e-09,
        ],
        [
            1.188_182_261_915_046_0e-10,
            -2.812_044_603_809_647_3e-11,
            1.128_156_995_421_512_4e-10,
        ],
        [
            -3.875_708_605_574_370_0e-11,
            -7.032_298_000_337_574_0e-12,
            -3.883_028_775_444_161_0e-12,
        ],
    ];

    /// Eq. (10.12)'s three lines at the oracle geometry, at 40 digits
    /// (`derivations/relativity_iers.mac` (6)).
    const SCHWARZSCHILD_MAXIMA: [f64; 3] = [
        1.222560616623010378e-8,
        3.2000087114594870656e-9,
        4.0859783034617065981e-9,
    ];
    const LENSE_THIRRING_MAXIMA: [f64; 3] = [
        1.1881822619150458855e-10,
        -2.8120446038096469033e-11,
        1.1281569954215122861e-10,
    ];
    const DE_SITTER_MAXIMA: [f64; 3] = [
        -3.8757086055743790924e-11,
        -7.0322980003376513426e-12,
        -3.8830287754440477062e-12,
    ];

    /// Schwarzschild, Lense-Thirring and de Sitter at the oracle geometry. The force takes the
    /// geocentric Sun and sets `R = −r_sun`, `Ṙ = −v_sun`, so it is handed the negation of the Earth's
    /// heliocentric state.
    fn oracle_terms() -> [[f64; 3]; 3] {
        let s = Schwarzschild { mu: MU_EARTH }.acceleration::<f64>(0.0, &ORACLE_R, &ORACLE_V);
        let lt = LenseThirring {
            mu: MU_EARTH,
            j: [0.0, 0.0, EARTH_ANGULAR_MOMENTUM_PER_MASS],
        }
        .acceleration::<f64>(0.0, &ORACLE_R, &ORACLE_V);
        let ds = DeSitter {
            mu_sun: ORACLE_MU_SUN,
            sun_pv: |_t| (EARTH_FROM_SUN_R.map(|x| -x), EARTH_FROM_SUN_V.map(|x| -x)),
        }
        .acceleration::<f64>(0.0, &ORACLE_R, &ORACLE_V);
        [s, lt, ds]
    }

    fn relative_distance(got: [f64; 3], want: [f64; 3]) -> f64 {
        let diff = libm::sqrt((0..3).map(|i| (got[i] - want[i]).powi(2)).sum());
        diff / mag(want)
    }

    const TERMS: [&str; 3] = [
        "IERS 2010 eq 10.12 Schwarzschild acceleration",
        "IERS 2010 eq 10.12 Lense-Thirring acceleration",
        "IERS 2010 eq 10.12 de Sitter acceleration",
    ];

    /// The distance to Orekit 13 at the oracle geometry, term by term. Orekit reads the de Sitter
    /// state in the Sun's body-inertial frame; the oracle hands that state over verbatim.
    /// [`each_term_is_its_line_of_eq_10_12`] decides.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn distance_to_orekit() {
        for ((name, got), want) in TERMS.iter().zip(oracle_terms()).zip(OREKIT_TERMS) {
            crate::dsst::comparison_row(
                &std::format!("{name}, norm of the difference"),
                "relative",
                &[relative_distance(got, want)],
            );
        }
    }

    /// Each term at the oracle geometry is its line of eq. (10.12) at β = γ = 1, evaluated at 40
    /// digits by `derivations/relativity_iers.mac` (6) from the Earth's heliocentric `R`, `Ṙ` as the
    /// chapter writes them. The worst term is 2.6e-16 relative, a few roundings of the f64 sum. (G3)
    #[test]
    fn each_term_is_its_line_of_eq_10_12() {
        let want = [
            SCHWARZSCHILD_MAXIMA,
            LENSE_THIRRING_MAXIMA,
            DE_SITTER_MAXIMA,
        ];
        for ((name, got), want) in TERMS.iter().zip(oracle_terms()).zip(want) {
            let rel = relative_distance(got, want);
            println!("{name}, against 40 digits: {rel:.3e} relative");
            assert!(
                rel < 1e-15,
                "{name}: got {got:?}, want {want:?}, rel {rel:.3e}"
            );
        }
    }
}
