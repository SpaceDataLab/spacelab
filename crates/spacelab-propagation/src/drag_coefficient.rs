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
/// `speed_ratio`, energy accommodation `accommodation` (`α ∈ [0, 1]`; 1 = full diffuse re-emission at
/// the wall temperature, 0 = specular), wall temperature `wall_temperature` (K) and ambient
/// `atmosphere_temperature` (K).
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
    let l = sqrt((1.0 - g * g).max(0.0)); // sin δ ≥ 0

    let z = 1.0 + erf(s * g);
    let e = exp(-s * s * g * g);
    let sqrt_pi = sqrt(PI);

    // Re-emitted / incident speed ratio (Koppenwallner ½ factor); the wall-temperature dependence.
    let v_ratio = sqrt(
        0.5 * (1.0
            + accommodation * (2.0 * wall_temperature / (atmosphere_temperature * s * s) - 1.0)),
    );

    let pressure = (g * g + 1.0 / (2.0 * s * s)) * z
        + g / (sqrt_pi * s) * e
        + 0.5 * v_ratio * (sqrt_pi * g * z + e / s);
    let shear = l * g * z + l / (s * sqrt_pi) * e;

    SentmanCoefficients {
        pressure,
        shear,
        drag: pressure * g + shear * l,
        lift: pressure * l - shear * g,
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

    #[test]
    fn colder_walls_and_lower_accommodation_change_the_coefficient() {
        // Accommodation and wall temperature genuinely move C_d — the whole point over a fixed constant.
        let full = sentman_panel(7.4549, 1.0, 1.0, 300.0, 1000.0).drag;
        let specularish = sentman_panel(7.4549, 1.0, 0.5, 300.0, 1000.0).drag;
        assert!(
            (full - specularish).abs() > 1e-3,
            "α should matter: {full} vs {specularish}"
        );
    }
}
