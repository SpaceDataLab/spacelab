//! Physical free-molecular-flow drag coefficient (Sentman's gas–surface interaction), replacing the
//! fixed `C_d ≈ 2.2` assumption of a cannonball or a constant-coefficient box-wing.
//!
//! At thermospheric altitudes the flow is free-molecular (mean free path ≫ spacecraft): each incoming
//! atmospheric particle strikes the surface and re-emits without inter-molecular collisions. The
//! resulting per-panel force depends on the **speed ratio** `s` (bulk speed over thermal speed), the
//! **incidence angle**, the **energy accommodation coefficient** `α` (how thermally equilibrated the
//! re-emitted gas is with the wall), and the **wall temperature** — none of which a single fixed `C_d`
//! captures. A physical `C_d` swings from ~2.1 (windward, hyperthermal) down through the diffuse tail,
//! and its variation is a leading source of orbit-prediction error in LEO.
//!
//! This is Sentman's (1961) flat-panel model in the Doornbos (2011, §3.4.2, Eqs 3.55–3.59)
//! parameterisation, cross-checked against the ADBSat `coeff_sentman.m` implementation
//! (Mostaza-Prieto et al.); both agree character-for-character. The re-emission uses Koppenwallner's
//! (2009) `½` factor for `v_re/v_inc`. Feed [`sentman_panel`] per facet from the box-wing
//! ([`crate::spacecraft::BoxWing::sentman_drag_acceleration`]).
//!
//! `derivations/sentman_panel.mac` integrates the drifting Maxwellian over the molecules that reach
//! the panel and gets the incident pressure and shear in closed form, on both sides of the panel.
//! It also derives the diffuse re-emission term from the half-range Maxwellian, and `v_re/v_inc`
//! from the definition of `α`, in which `T_∞` cancels. Quadrature reproduces the 45° benchmark row.
//! The model re-emits diffusely at every `α`: at `α = 0` the re-emitted gas keeps the incident
//! energy, and normal incidence tends to `c_p = 2 + √(π/2)`, not specular reflection's 4.
//!
//! Beyond Sentman: [`cll_panel`] re-emits by the Cercignani–Lampis–Lord kernels (Lord 1991), with a
//! normal-energy and a tangential-momentum accommodation, from diffuse at `α_n = σ_t = 1` to
//! specular at 0. `derivations/cll_panel.mac` derives its moments from Lord's kernels: the mean
//! reflected normal speed in closed form, a confluent hypergeometric function, and the limits
//! `α_n = 1` (Sentman at `α = 1`), `α_n → 0` (specular) and `s → ∞`; one integral over the incident
//! Maxwellian remains, and `derivations/cll_panel.gp` holds it at 40 digits. [`SurfaceModel`]
//! weights a clean and an adsorbate-covered re-emission by the coverage, which
//! [`langmuir_coverage`] gives from the atomic-oxygen partial pressure (Pilinski et al. 2013).

// The gas-dynamics symbols (s, γ, ℓ, cp, cτ, cd, cl) are one/two letters by universal convention; the
// reference/tool names (ADBSat, Doornbos, Koppenwallner) are proper nouns, not code identifiers.
#![allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::doc_markdown
)]

use libm::{erf, exp, sqrt};

use core::f64::consts::PI;

/// Universal gas constant `R = N_A k` (J·mol⁻¹·K⁻¹), exact in the 2019 SI.
const GAS_CONSTANT: f64 = 8.314_462_618_153_24;

/// Sentman free-molecular-flow coefficients for one flat panel, all per unit **panel** area (not
/// projected). `pressure` acts along `−n̂` (into the surface), `shear` along the tangential flow
/// direction; `drag`/`lift` are the same force resolved along the flow (`û_D`) and perpendicular
/// (`û_L`): `C_d = c_p cos δ + c_τ sin δ`, `C_l = c_p sin δ − c_τ cos δ`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SentmanCoefficients {
    /// Pressure (normal) coefficient `c_p`, force along `−n̂`.
    pub pressure: f64,
    /// Shear (tangential) coefficient `c_τ`, force along the in-plane flow direction.
    pub shear: f64,
    /// Drag coefficient `C_d`, force along the incoming-flow direction `û_D`.
    pub drag: f64,
    /// Lift coefficient `C_l`, force perpendicular to the flow in the `(û_D, n̂)` plane.
    pub lift: f64,
}

/// The free-molecular-flow gas and surface conditions a physical drag coefficient needs: the ambient
/// gas (from the atmosphere model) and the spacecraft surface. Bundled so the box-wing physical-drag
/// entry point stays a single parameter object rather than a long argument list.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeMolecularConditions {
    /// Ambient (exospheric) translational temperature `T_∞` (K).
    pub atmosphere_temperature: f64,
    /// Ambient mean molar mass `M` (kg/mol) — e.g. ~0.016 where atomic oxygen dominates.
    pub mean_molar_mass: f64,
    /// Energy accommodation coefficient `α ∈ [0, 1]` of the surface.
    pub accommodation: f64,
    /// Spacecraft surface (wall) temperature `T_w` (K).
    pub wall_temperature: f64,
}

/// The gas and surface conditions of [`crate::spacecraft::BoxWing::gas_surface_drag_acceleration`]:
/// [`FreeMolecularConditions`] with any [`SurfaceModel`] in place of one diffuse accommodation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasSurfaceConditions {
    /// Ambient (exospheric) translational temperature `T_∞` (K).
    pub atmosphere_temperature: f64,
    /// Ambient mean molar mass `M` (kg/mol).
    pub mean_molar_mass: f64,
    /// Spacecraft surface (wall) temperature `T_w` (K).
    pub wall_temperature: f64,
    /// How the surface re-emits.
    pub surface: SurfaceModel,
}

impl From<FreeMolecularConditions> for GasSurfaceConditions {
    /// A clean surface re-emitting diffusely at the conditions' accommodation: Sentman's model.
    fn from(conditions: FreeMolecularConditions) -> Self {
        Self {
            atmosphere_temperature: conditions.atmosphere_temperature,
            mean_molar_mass: conditions.mean_molar_mass,
            wall_temperature: conditions.wall_temperature,
            surface: SurfaceModel {
                clean: Reemission::Diffuse {
                    accommodation: conditions.accommodation,
                },
                adsorbate: Reemission::Diffuse { accommodation: 1.0 },
                coverage: 0.0,
            },
        }
    }
}

/// The molecular speed ratio `s = v_rel / √(2 R T / M)`: the bulk relative speed `relative_speed`
/// (m/s) over the most-probable thermal speed of the ambient gas at translational `temperature` (K)
/// and mean molar mass `mean_molar_mass` (kg/mol, e.g. ~0.016 for atomic-oxygen-dominated upper
/// thermosphere). Large `s` (hyperthermal, ~7–9 in LEO) means the drag is nearly a directed beam.
#[must_use]
pub fn speed_ratio(relative_speed: f64, temperature: f64, mean_molar_mass: f64) -> f64 {
    let thermal_speed = sqrt(2.0 * GAS_CONSTANT * temperature / mean_molar_mass);
    relative_speed / thermal_speed
}

/// Sentman's free-molecular coefficients for a panel whose outward normal makes `cos_incidence`
/// `= cos δ = −(v̂_inc · n̂)` with the flow (windward ⟺ `cos_incidence ≥ 0`), at speed ratio
/// `speed_ratio`, energy accommodation `accommodation` (`α ∈ [0, 1]`; 1 = diffuse re-emission at the
/// wall temperature, 0 = diffuse re-emission at the incident energy, never specular), wall
/// temperature `wall_temperature` (K) and ambient `atmosphere_temperature` (K).
///
/// Doornbos Eqs 3.55–3.59 / ADBSat `coeff_sentman.m`, with `v_re/v_inc = √(½(1 + α(2 T_w/(T_∞ s²) −
/// 1)))` (Koppenwallner). Valid for `cos_incidence < 0` too (leeward thermal back-flux); the caller
/// decides whether to include shadowed panels.
#[must_use]
pub fn sentman_panel(
    speed_ratio: f64,
    cos_incidence: f64,
    accommodation: f64,
    wall_temperature: f64,
    atmosphere_temperature: f64,
) -> SentmanCoefficients {
    let s = speed_ratio;
    let g = cos_incidence;
    let incident = Incident::new(s, g);
    let (z, e) = (incident.z, incident.e);

    // Re-emitted / incident speed ratio (Koppenwallner ½ factor); the wall-temperature dependence.
    let v_ratio = sqrt(
        0.5 * (1.0
            + accommodation * (2.0 * wall_temperature / (atmosphere_temperature * s * s) - 1.0)),
    );

    let pressure = incident.pressure + 0.5 * v_ratio * (sqrt(PI) * g * z + e / s);
    incident.resolve(pressure, incident.shear)
}

/// The incident molecules' pressure and shear on a panel, the part every re-emission model shares
/// (`derivations/sentman_panel.mac` (2)), with the `erf` and `exp` terms the re-emission reuses.
struct Incident {
    g: f64,
    l: f64,
    z: f64,
    e: f64,
    pressure: f64,
    shear: f64,
}

impl Incident {
    fn new(s: f64, g: f64) -> Self {
        let l = sqrt((1.0 - g * g).max(0.0)); // sin δ ≥ 0
        let z = 1.0 + erf(s * g);
        let e = exp(-s * s * g * g);
        let sqrt_pi = sqrt(PI);
        Self {
            g,
            l,
            z,
            e,
            pressure: (g * g + 1.0 / (2.0 * s * s)) * z + g / (sqrt_pi * s) * e,
            shear: l * g * z + l / (s * sqrt_pi) * e,
        }
    }

    fn resolve(&self, pressure: f64, shear: f64) -> SentmanCoefficients {
        SentmanCoefficients {
            pressure,
            shear,
            drag: pressure * self.g + shear * self.l,
            lift: pressure * self.l - shear * self.g,
        }
    }
}

/// Cercignani–Lampis–Lord coefficients for a flat panel: the incident terms of [`sentman_panel`],
/// with the molecules re-emitted by Lord's (1991) kernels instead of diffusely.
/// `normal_accommodation` `α_n ∈ [0, 1]` is the normal-energy accommodation and
/// `tangential_accommodation` `σ_t ∈ [0, 1]` the tangential-momentum accommodation; the other
/// arguments are [`sentman_panel`]'s. `α_n = σ_t = 1` is [`sentman_panel`] at `α = 1`, and
/// `α_n = σ_t = 0` is specular reflection, which tends to `c_p = 4` at normal incidence.
///
/// The shear is `σ_t` times the incident shear: the tangential kernel returns `1 − σ_t` of each
/// molecule's tangential velocity on average. The re-emitted pressure is exact up to one integral:
/// `c_re = 2w ∫₀^∞ t (s/√π) e^{−s²(t−g)²} m(t/w) dt`, `w = √(T_w/T_∞)/s`, where `m(x) =
/// (√(π α_n)/2)·₁F₁(−½; 1; −(1 − α_n)x²/α_n)` is the mean reflected normal speed of a molecule
/// arriving at `x` wall thermal speeds (`derivations/cll_panel.mac`). The integral is taken by
/// adaptive Gauss–Kronrod G7–K15 to a relative 1e-13; `derivations/cll_panel.gp` gives the rows
/// at 40 digits. Walker, Mehta & Koller (2014) fit the normal kernel into Schaaf and Chambre's
/// closed forms instead. Hyperthermal limit at normal incidence: `c_p → 2(1 + √(1 − α_n))`.
#[must_use]
pub fn cll_panel(
    speed_ratio: f64,
    cos_incidence: f64,
    normal_accommodation: f64,
    tangential_accommodation: f64,
    wall_temperature: f64,
    atmosphere_temperature: f64,
) -> SentmanCoefficients {
    let s = speed_ratio;
    let g = cos_incidence;
    let a = normal_accommodation;
    let incident = Incident::new(s, g);
    let w = sqrt(wall_temperature / atmosphere_temperature) / s;

    // t = g + y/s: the drifting Maxwellian becomes e^{−y²}, cut where it is below e^{−81}.
    let lower = (-s * g).max(-CUT);
    let upper = lower.max(0.0) + CUT;
    let integrand = |y: f64| {
        let t = g + y / s;
        t * exp(-y * y) * mean_reflected_normal_speed(t / w, a)
    };
    let reemitted = 2.0 * w / sqrt(PI) * kronrod(&integrand, lower, upper, 0);

    incident.resolve(
        incident.pressure + reemitted,
        tangential_accommodation * incident.shear,
    )
}

/// Half-width, in thermal speeds, of the incident distribution [`cll_panel`] integrates over:
/// `e^{−81}` ≈ 6.6e-36 of the peak, far below the integral's tolerance.
const CUT: f64 = 9.0;

/// [`cll_panel`]'s `m(x)`: the mean normal speed, in wall thermal speeds, of the molecules Lord's
/// normal kernel re-emits from one that arrives at `x`. `α_n = 0` is specular, `m = x`.
fn mean_reflected_normal_speed(x: f64, normal_accommodation: f64) -> f64 {
    let a = normal_accommodation;
    if a == 0.0 {
        return x;
    }
    0.5 * sqrt(PI * a) * kummer_h((1.0 - a) * x * x / a)
}

/// `h(q) = ₁F₁(−½; 1; −q)`, by `e^{−q}·₁F₁(3/2; 1; q)`, whose terms are all positive, up to
/// `q = 60`, and above it by the asymptotic series `(q^{½}/Γ(3/2))·Σ (−½)_j²/j!·q^{−j}` cut at its
/// smallest term (DLMF 13.7.1), whose error and the dropped `e^{−q}` part are near 1e-26
/// (`derivations/cll_panel.mac` (4)).
fn kummer_h(q: f64) -> f64 {
    // A NaN argument would never meet the asymptotic loop's stopping test.
    if q.is_nan() {
        return f64::NAN;
    }
    if q <= 60.0 {
        let (mut term, mut sum, mut k) = (1.0, 1.0, 0.0);
        while k <= q || term > f64::EPSILON * 1.0e-2 * sum {
            term *= (k + 1.5) * q / ((k + 1.0) * (k + 1.0));
            sum += term;
            k += 1.0;
        }
        sum * exp(-q)
    } else {
        let (mut term, mut sum, mut j) = (1.0_f64, 1.0, 0.0);
        loop {
            let next = term * (j - 0.5) * (j - 0.5) / ((j + 1.0) * q);
            if next.abs() >= term.abs() || next.abs() <= f64::EPSILON * 1.0e-2 * sum {
                break;
            }
            term = next;
            sum += term;
            j += 1.0;
        }
        sum * sqrt(q) * 2.0 / sqrt(PI)
    }
}

/// Recursion limit of [`kronrod`]: an interval 2⁻²⁰ of the integration range, 1.7e-5 thermal
/// speeds wide, is below any width the smooth integrand needs, and the bound caps the work at 2²⁰
/// intervals where the tolerance cannot be met.
const MAX_RECURSION: u32 = 20;

/// Adaptive Gauss–Kronrod G7–K15 quadrature of `f` over `[a, b]`, to a relative 1e-13 between the
/// K15 and the embedded G7 estimate. Equal estimates (zero on both, where the incident
/// distribution underflows) are accepted, and so is a non-finite one, which halving cannot improve.
fn kronrod(f: &impl Fn(f64) -> f64, a: f64, b: f64, level: u32) -> f64 {
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let mut k15 = K15_WEIGHTS[0] * f(mid);
    let mut g7 = G7_WEIGHTS[0] * f(mid);
    for i in 1..K15_NODES.len() {
        let pair = f(mid - half * K15_NODES[i]) + f(mid + half * K15_NODES[i]);
        k15 += K15_WEIGHTS[i] * pair;
        if i % 2 == 0 {
            g7 += G7_WEIGHTS[i / 2] * pair;
        }
    }
    k15 *= half;
    g7 *= half;
    if (k15 - g7).abs() <= 1.0e-13 * k15.abs() || !k15.is_finite() || level == MAX_RECURSION {
        k15
    } else {
        kronrod(f, a, mid, level + 1) + kronrod(f, mid, b, level + 1)
    }
}

/// Kronrod K15 abscissae on `[0, 1]`, the rule being symmetric; the even-indexed ones are the G7
/// nodes. The 40-digit values `derivations/kronrod_g7k15.mac` derives from their definitions.
#[expect(
    clippy::excessive_precision,
    reason = "the derivation's 40-digit values, kept whole"
)]
const K15_NODES: [f64; 8] = [
    0.0,
    0.207_784_955_007_898_467_600_689_403_773_244_913_479_8,
    0.405_845_151_377_397_166_906_606_412_076_961_463_347_4,
    0.586_087_235_467_691_130_294_144_838_258_729_598_436_8,
    0.741_531_185_599_394_439_863_864_773_280_788_407_074_1,
    0.864_864_423_359_769_072_789_712_788_640_926_201_211,
    0.949_107_912_342_758_524_526_189_684_047_851_262_400_8,
    0.991_455_371_120_812_639_206_854_697_526_328_516_642,
];

/// Kronrod K15 weights for [`K15_NODES`] (`derivations/kronrod_g7k15.mac`).
#[expect(
    clippy::excessive_precision,
    reason = "the derivation's 40-digit values, kept whole"
)]
const K15_WEIGHTS: [f64; 8] = [
    0.209_482_141_084_727_828_012_999_174_891_714_263_697_8,
    0.204_432_940_075_298_892_414_161_999_234_649_084_716_5,
    0.190_350_578_064_785_409_913_256_402_421_013_682_826_1,
    0.169_004_726_639_267_902_826_583_426_598_550_284_106_2,
    0.140_653_259_715_525_918_745_189_590_510_237_920_399_9,
    0.104_790_010_322_250_183_839_876_322_541_518_017_443_8,
    0.063_092_092_629_978_553_290_700_663_189_204_286_665_07,
    0.022_935_322_010_529_224_963_732_008_058_969_591_993_56,
];

/// Gauss G7 weights for the even-indexed [`K15_NODES`] (`derivations/kronrod_g7k15.mac`).
#[expect(
    clippy::excessive_precision,
    reason = "the derivation's 40-digit values, kept whole"
)]
const G7_WEIGHTS: [f64; 4] = [
    0.417_959_183_673_469_387_755_102_040_816_326_530_612_2,
    0.381_830_050_505_118_944_950_369_775_488_975_133_878_4,
    0.279_705_391_489_276_667_901_467_771_423_779_582_486_9,
    0.129_484_966_168_869_693_270_611_432_679_082_018_328_6,
];

/// The fraction of a surface covered by adsorbed atomic oxygen, by the Langmuir isotherm
/// `θ = K·P_O/(1 + K·P_O)` with the oxygen partial pressure `P_O = n_O·k_B·T` (Pilinski et al.
/// 2013): `langmuir_constant` `K` (Pa⁻¹), `oxygen_number_density` `n_O` (m⁻³) and the ambient
/// `temperature` (K). `K` is a fit to satellite drag, with no default here: Walker et al. (2014)
/// quote 1.44e6 Pa⁻¹ for DRIA on one data set and 3e6 Pa⁻¹ on another (as cited by Paul et al.
/// 2022 and Antona et al. 2024; the primary papers were not readable).
#[must_use]
pub fn langmuir_coverage(
    langmuir_constant: f64,
    oxygen_number_density: f64,
    temperature: f64,
) -> f64 {
    let kp = langmuir_constant * oxygen_number_density * BOLTZMANN * temperature;
    kp / (1.0 + kp)
}

/// Boltzmann constant `k_B` (J/K), exact in the 2019 SI.
const BOLTZMANN: f64 = 1.380_649e-23;

/// How a panel re-emits the molecules that strike it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reemission {
    /// Diffuse re-emission at energy accommodation `α` ([`sentman_panel`]; DRIA).
    Diffuse {
        /// Energy accommodation `α ∈ [0, 1]`.
        accommodation: f64,
    },
    /// Lord's kernels ([`cll_panel`]).
    Cll {
        /// Normal-energy accommodation `α_n ∈ [0, 1]`.
        normal_accommodation: f64,
        /// Tangential-momentum accommodation `σ_t ∈ [0, 1]`.
        tangential_accommodation: f64,
    },
}

impl Reemission {
    /// This re-emission's panel coefficients ([`sentman_panel`] or [`cll_panel`]).
    #[must_use]
    pub fn panel(
        &self,
        speed_ratio: f64,
        cos_incidence: f64,
        wall_temperature: f64,
        atmosphere_temperature: f64,
    ) -> SentmanCoefficients {
        match *self {
            Self::Diffuse { accommodation } => sentman_panel(
                speed_ratio,
                cos_incidence,
                accommodation,
                wall_temperature,
                atmosphere_temperature,
            ),
            Self::Cll {
                normal_accommodation,
                tangential_accommodation,
            } => cll_panel(
                speed_ratio,
                cos_incidence,
                normal_accommodation,
                tangential_accommodation,
                wall_temperature,
                atmosphere_temperature,
            ),
        }
    }
}

/// A surface whose fraction `coverage` holds adsorbed atomic oxygen: the panel coefficients are
/// `θ·c_adsorbate + (1 − θ)·c_clean` (Pilinski et al. 2013; Walker et al. 2014), each part by its
/// own [`Reemission`]. `θ` comes from [`langmuir_coverage`] at the sample's oxygen density, or is
/// 0 for a clean surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceModel {
    /// Re-emission from the clean surface.
    pub clean: Reemission,
    /// Re-emission from the adsorbate, usually diffuse at full accommodation.
    pub adsorbate: Reemission,
    /// Adsorbate coverage `θ ∈ [0, 1]`.
    pub coverage: f64,
}

impl SurfaceModel {
    /// The coverage-weighted panel coefficients; the clean part alone at `θ = 0`, the adsorbate's
    /// at `θ = 1`, each evaluated only when its weight is not zero.
    #[must_use]
    pub fn panel(
        &self,
        speed_ratio: f64,
        cos_incidence: f64,
        wall_temperature: f64,
        atmosphere_temperature: f64,
    ) -> SentmanCoefficients {
        let theta = self.coverage;
        let part = |r: &Reemission, weight: f64| {
            if weight == 0.0 {
                SentmanCoefficients {
                    pressure: 0.0,
                    shear: 0.0,
                    drag: 0.0,
                    lift: 0.0,
                }
            } else {
                let c = r.panel(
                    speed_ratio,
                    cos_incidence,
                    wall_temperature,
                    atmosphere_temperature,
                );
                SentmanCoefficients {
                    pressure: weight * c.pressure,
                    shear: weight * c.shear,
                    drag: weight * c.drag,
                    lift: weight * c.lift,
                }
            }
        };
        let (c, a) = (part(&self.clean, 1.0 - theta), part(&self.adsorbate, theta));
        SentmanCoefficients {
            pressure: c.pressure + a.pressure,
            shear: c.shear + a.shear,
            drag: c.drag + a.drag,
            lift: c.lift + a.lift,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_ratio_matches_the_atomic_oxygen_leo_reference() {
        // Atomic O (M = 0.016 kg/mol), T = 1000 K, v_rel = 7600 m/s → s = 7.4549 (Doornbos Fig 3.8).
        let s = speed_ratio(7600.0, 1000.0, 0.016);
        assert!((s - 7.4549).abs() < 1e-3, "s = {s}");
    }

    #[test]
    fn sentman_panel_matches_the_cross_verified_benchmark_table() {
        // s = 7.4549, α = 1, T_w = 300 K, T_∞ = 1000 K — computed independently and confirmed against
        // ADBSat coeff_sentman.m + Doornbos §3.4.2 (both agree). Columns: c_p, c_τ, c_d, c_l.
        let cases = [
            (0.0_f64, 2.14822, 0.00000, 2.14822, 0.00000),
            (15.0, 2.00981, 0.50000, 2.07073, 0.03721),
            (45.0, 1.11008, 1.00000, 1.49205, 0.07784),
            (75.0, 0.18567, 0.50018, 0.53119, 0.04989),
            (90.0, 0.01392, 0.07568, 0.07568, 0.01392),
        ];
        for (deg, cp, ct, cd, cl) in cases {
            let g = libm::cos(deg.to_radians());
            let c = sentman_panel(7.4549, g, 1.0, 300.0, 1000.0);
            assert!(
                (c.pressure - cp).abs() < 1e-4,
                "δ={deg} cp {} vs {cp}",
                c.pressure
            );
            assert!(
                (c.shear - ct).abs() < 1e-4,
                "δ={deg} cτ {} vs {ct}",
                c.shear
            );
            assert!((c.drag - cd).abs() < 1e-4, "δ={deg} cd {} vs {cd}", c.drag);
            assert!((c.lift - cl).abs() < 1e-4, "δ={deg} cl {} vs {cl}", c.lift);
        }
    }

    #[test]
    fn normal_incidence_is_pure_drag_in_the_expected_2_to_3_band() {
        // Windward, normal to the flow: no shear, and C_d in the physical ~2–2.3 band (not the fixed 2.2).
        let c = sentman_panel(7.4549, 1.0, 1.0, 300.0, 1000.0);
        assert!(c.shear.abs() < 1e-12 && c.lift.abs() < 1e-12);
        assert!((2.0..2.3).contains(&c.drag), "cd {}", c.drag);
    }

    /// The rows `derivations/cll_panel.gp` integrates at 40 digits, the mean reflected normal speed
    /// in the Bessel form: s, δ or cos δ, α_n, σ_t, T_w, T_∞, then c_p, c_τ, C_d, C_l.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the reference's 21 digits, kept whole"
    )]
    #[expect(
        clippy::too_many_lines,
        reason = "the ten reference rows of derivations/cll_panel.gp"
    )]
    fn cll_panel_matches_the_40_digit_quadrature() {
        let deg = |d: f64| libm::cos(d.to_radians());
        let rows: [([f64; 6], [f64; 4]); 10] = [
            (
                [7.4549, 1.0, 0.5, 1.0, 300.0, 1000.0],
                [
                    3.446_840_270_251_643_754_70,
                    0.0,
                    3.446_840_270_251_643_754_70,
                    0.0,
                ],
            ),
            (
                [7.4549, deg(15.0), 0.9, 0.7, 300.0, 1000.0],
                [
                    2.487_533_523_126_846_526_97,
                    3.500_000_000_000_000_000_00e-1,
                    2.493_359_539_533_938_654_92,
                    3.057_470_119_150_257_647_68e-1,
                ],
            ),
            (
                [7.4549, deg(45.0), 0.5, 1.0, 300.0, 1000.0],
                [
                    1.739_734_914_156_624_998_11,
                    1.000_000_000_000_000_782_34,
                    1.937_285_136_453_693_751_18,
                    5.230_715_740_805_975_959_92e-1,
                ],
            ),
            (
                [7.4549, deg(75.0), 0.05, 0.3, 300.0, 1000.0],
                [
                    3.002_070_708_611_623_075_47e-1,
                    1.500_530_388_286_255_791_13e-1,
                    2.226_394_129_310_366_236_01e-1,
                    2.511_411_787_550_327_338_99e-1,
                ],
            ),
            (
                [7.4549, deg(89.0), 0.75, 1.0, 300.0, 1000.0],
                [
                    1.994_975_716_351_094_455_94e-2,
                    9.439_587_807_085_488_818_39e-2,
                    9.472_967_239_175_069_625_53e-2,
                    1.829_928_348_876_930_583_45e-2,
                ],
            ),
            (
                [7.4549, 0.0, 0.5, 1.0, 300.0, 1000.0],
                [
                    1.625_018_820_786_185_519_22e-2,
                    7.568_036_909_251_046_787_32e-2,
                    7.568_036_909_251_046_787_32e-2,
                    1.625_018_820_786_185_519_22e-2,
                ],
            ),
            (
                [7.4549, -0.2, 0.5, 1.0, 300.0, 1000.0],
                [
                    1.525_985_943_351_264_917_44e-4,
                    1.174_125_068_767_597_084_28e-3,
                    1.119_883_206_209_303_772_99e-3,
                    3.843_404_903_883_288_283_35e-4,
                ],
            ),
            (
                [3.0, 0.6, 0.2, 0.5, 400.0, 700.0],
                [
                    1.581_392_418_923_755_179_00,
                    4.803_278_352_813_761_088_20e-1,
                    1.333_097_719_579_353_994_45,
                    9.769_172_339_701_784_779_05e-1,
                ],
            ),
            (
                [12.0, 1.0, 0.999, 1.0, 300.0, 1500.0],
                [
                    2.095_035_356_772_827_702_02,
                    0.0,
                    2.095_035_356_772_827_702_02,
                    0.0,
                ],
            ),
            (
                [9.0, 1.0, 1.0e-4, 0.0, 300.0, 900.0],
                [
                    4.024_590_943_996_798_689_26,
                    0.0,
                    4.024_590_943_996_798_689_26,
                    0.0,
                ],
            ),
        ];
        for ([s, g, a, sigma, tw, ti], want) in rows {
            let c = cll_panel(s, g, a, sigma, tw, ti);
            let got = [c.pressure, c.shear, c.drag, c.lift];
            for (k, (x, y)) in got.iter().zip(want).enumerate() {
                assert!(
                    (x - y).abs() <= 1.0e-13 * y.abs().max(1.0e-3),
                    "s {s} g {g} α_n {a} σ_t {sigma}: coefficient {k} {x} vs {y}"
                );
            }
        }
    }

    #[test]
    fn cll_at_full_accommodation_is_sentman_at_full_accommodation() {
        for g in [1.0, 0.8, 0.3, 0.0, -0.1] {
            let cll = cll_panel(7.4549, g, 1.0, 1.0, 300.0, 1000.0);
            let sentman = sentman_panel(7.4549, g, 1.0, 300.0, 1000.0);
            for (x, y) in [
                (cll.pressure, sentman.pressure),
                (cll.shear, sentman.shear),
                (cll.drag, sentman.drag),
                (cll.lift, sentman.lift),
            ] {
                assert!((x - y).abs() <= 1.0e-13 * y.abs(), "g {g}: {x} vs {y}");
            }
        }
    }

    #[test]
    fn cll_without_accommodation_reflects_specularly() {
        // Specular: the re-emitted normal momentum equals the incident, no shear survives.
        let (s, g) = (7.4549, libm::cos(30.0_f64.to_radians()));
        let incident = Incident::new(s, g);
        let c = cll_panel(s, g, 0.0, 0.0, 300.0, 1000.0);
        assert!((c.pressure - 2.0 * incident.pressure).abs() <= 1.0e-13 * c.pressure);
        assert_eq!(c.shear, 0.0);
        // Normal incidence tends to 4, not Sentman's 2 + √(π/2).
        let normal = cll_panel(1.0e4, 1.0, 0.0, 0.0, 300.0, 1000.0).pressure;
        assert!((normal - 4.0).abs() < 1.0e-7, "{normal}");
    }

    #[test]
    fn cll_returns_nan_promptly_for_a_nan_input() {
        // Neither the asymptotic series nor the adaptive quadrature may spin on a NaN.
        for c in [
            cll_panel(7.4549, 1.0, 0.5, 1.0, f64::NAN, 1000.0),
            cll_panel(7.4549, 0.7, f64::NAN, 1.0, 300.0, 1000.0),
        ] {
            assert!(c.pressure.is_nan() && c.drag.is_nan(), "{c:?}");
        }
        assert!(kummer_h(f64::NAN).is_nan());
    }

    #[test]
    fn kronrod_bounds_its_work_where_the_tolerance_cannot_be_met() {
        // 1/√x is scale-invariant at 0: the interval that holds it never meets the relative
        // tolerance, so only the depth limit stops the halving there.
        // The singularity at either end: each half of the split carries its own depth.
        for at_right in [false, true] {
            let calls = core::cell::Cell::new(0_u32);
            let f = |x: f64| {
                calls.set(calls.get() + 1);
                1.0 / libm::sqrt(if at_right { 1.0 - x } else { x })
            };
            let got = kronrod(&f, 0.0, 1.0, 0);
            // Measured at the left end: 1264 evaluations and 4.5e-5 short of 2, the integral over
            // [0, 2⁻²⁰] being 2e-3. Without the limit the halving runs on to the smallest
            // subnormal, about 60 000.
            assert!(calls.get() <= 2000, "{} evaluations", calls.get());
            assert!((got - 2.0).abs() < 1.0e-4, "{got}");
        }
    }

    #[test]
    fn cll_tends_to_its_hyperthermal_limit() {
        // c_p → 2(1 + √(1 − α_n)) at normal incidence; s = 2000 is 3.41421401566629… at 40 digits.
        let c = cll_panel(2000.0, 1.0, 0.5, 1.0, 300.0, 1000.0).pressure;
        assert!((c - 3.414_214_015_666_295).abs() < 1.0e-12, "{c}");
        assert!((c - 2.0 * (1.0 + libm::sqrt(0.5))).abs() < 1.0e-6, "{c}");
    }

    /// `h(q)` against `derivations/cll_panel.gp`'s Bessel form at 40 digits, on both sides of the
    /// branch seam at 60.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the reference's 21 digits, kept whole"
    )]
    fn kummer_h_matches_the_bessel_form_on_both_branches() {
        let rows = [
            (1.0e-3, 1.000_499_937_510_415_039_29),
            (0.1, 1.049_385_256_156_729_251_16),
            (1.0, 1.446_491_344_083_171_833_36),
            (17.0, 4.721_372_211_595_386_279_23),
            (60.0, 8.776_882_571_335_328_212_03),
            (61.0, 8.849_116_508_789_284_973_91),
            (300.0, 1.956_039_402_971_401_563_95e1),
            (1.0e4, 1.128_407_376_927_334_901_95e2),
            (1.0e6, 1.128_379_449_190_339_609_65e3),
        ];
        for (q, want) in rows {
            let got = kummer_h(q);
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want,
                "h({q}) {got} vs {want}"
            );
        }
        assert_eq!(kummer_h(0.0), 1.0);
    }

    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the reference's 21 digits, kept whole"
    )]
    fn langmuir_coverage_matches_the_reference_and_saturates() {
        // K = 1.44e6 /Pa, n_O = 1e14 /m³, T = 1000 K (`derivations/cll_panel.gp`).
        let theta = langmuir_coverage(1.44e6, 1.0e14, 1000.0);
        assert!(
            (theta - 6.653_430_493_438_019_738_98e-1).abs() < 1.0e-15,
            "{theta}"
        );
        assert_eq!(langmuir_coverage(1.44e6, 0.0, 1000.0), 0.0);
        assert!(langmuir_coverage(1.44e6, 1.0e23, 1000.0) > 1.0 - 1.0e-9);
    }

    #[test]
    fn a_surface_model_weights_its_two_re_emissions_by_the_coverage() {
        let clean = Reemission::Cll {
            normal_accommodation: 0.6,
            tangential_accommodation: 0.8,
        };
        let adsorbate = Reemission::Diffuse { accommodation: 1.0 };
        let (s, g) = (7.4549, 0.7);
        let c = clean.panel(s, g, 300.0, 1000.0);
        let a = adsorbate.panel(s, g, 300.0, 1000.0);
        assert_eq!(c, cll_panel(s, g, 0.6, 0.8, 300.0, 1000.0));
        assert_eq!(a, sentman_panel(s, g, 1.0, 300.0, 1000.0));
        let surface = |coverage| SurfaceModel {
            clean,
            adsorbate,
            coverage,
        };
        assert_eq!(surface(0.0).panel(s, g, 300.0, 1000.0), c);
        assert_eq!(surface(1.0).panel(s, g, 300.0, 1000.0), a);
        let mixed = surface(0.3).panel(s, g, 300.0, 1000.0);
        for (m, x, y) in [
            (mixed.pressure, c.pressure, a.pressure),
            (mixed.shear, c.shear, a.shear),
            (mixed.drag, c.drag, a.drag),
            (mixed.lift, c.lift, a.lift),
        ] {
            assert!((m - (0.7 * x + 0.3 * y)).abs() <= 1.0e-15 * m.abs(), "{m}");
        }
    }

    #[test]
    fn colder_walls_and_lower_accommodation_change_the_coefficient() {
        // Accommodation and wall temperature genuinely move C_d — the whole point over a fixed constant.
        let full = sentman_panel(7.4549, 1.0, 1.0, 300.0, 1000.0).drag;
        let half = sentman_panel(7.4549, 1.0, 0.5, 300.0, 1000.0).drag;
        assert!(
            (full - half).abs() > 1e-3,
            "α should matter: {full} vs {half}"
        );
    }
}
