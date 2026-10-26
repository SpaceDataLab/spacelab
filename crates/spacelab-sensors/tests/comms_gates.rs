//! `CommsBudget` against its defining equations, at the values `derivations/comms_link_budget.gp`
//! prints at 120 digits: the slant range measured from the triangle, and `C/N0` with the exact SI
//! Boltzmann term. `derivations/comms_link_budget.mac` derives the identities.

#![allow(
    clippy::expect_used,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]

use core::f64::consts::{FRAC_PI_2, FRAC_PI_6};

use spacelab_sensors::{CommsAntenna, CommsBudget};

/// The budget of `derivations/comms_link_budget.gp`.
fn budget(required_cn0_dbhz: f64) -> CommsBudget {
    CommsBudget {
        eirp_dbw: 20.0,
        g_over_t_db: 10.0,
        frequency_hz: 8.2e9,
        required_cn0_dbhz,
        zenith_atmo_db: 0.5,
        reference_altitude_m: 700e3,
        floor_elevation_rad: 5.0_f64.to_radians(),
    }
}

const SLANT_ZENITH_M: f64 = 7.000000000000000000000000e5;
const SLANT_HORIZON_M: f64 = 3.069102767911169181154006e6;
const SLANT_30_M: f64 = 1.236939825533544452236805e6;
const CN0_5_DBHZ: f64 = 7.396277609105089784970028e1;
const CN0_30_DBHZ: f64 = 8.502813545067965695619697e1;
const CN0_90_DBHZ: f64 = 9.047314610337482107371259e1;
/// `dM/de` at 30 deg, dB/rad.
const MARGIN_SLOPE_30: f64 = 1.257198275247519688855737e1;

/// (G2) The slant range is the distance in the centre-station-satellite triangle: `h` at the
/// zenith, `sqrt(r_s^2 - R^2)` at the horizon tangency, and the 120-digit triangle at 30 deg,
/// each within `32 eps_mach r_s / d` relative (the closed form subtracts terms of order `r_s`).
#[test]
fn the_slant_range_is_the_triangles_side() {
    let b = budget(0.0);
    let rs = 6_378_137.0 + b.reference_altitude_m;
    for (eps, want) in [
        (FRAC_PI_2, SLANT_ZENITH_M),
        (0.0, SLANT_HORIZON_M),
        (FRAC_PI_6, SLANT_30_M),
    ] {
        let got = b.slant_range(eps);
        let rel = (got - want).abs() / want;
        assert!(
            rel <= 32.0 * f64::EPSILON * rs / want,
            "at {eps} rad: {got:.17e} m against {want:.17e} m"
        );
    }
}

/// (G2) `C/N0 = EIRP + G/T - 20 log10(4 pi d f/c) - L_atm/sin(e) - 10 log10(k)` with the exact SI
/// `k`, against the 120-digit budget at 5, 30 and 90 deg, within `64 eps_mach x 512 dB` -- the
/// rounding of a sum of terms none larger than 512 dB. The customary 228.6 dB Boltzmann term would
/// miss by 8.3e-4 dB.
#[test]
fn cn0_is_the_link_equation_with_the_exact_boltzmann_term() {
    let b = budget(0.0);
    let tol = 64.0 * f64::EPSILON * 512.0;
    for (eps, want) in [
        (5.0_f64.to_radians(), CN0_5_DBHZ),
        (FRAC_PI_6, CN0_30_DBHZ),
        (FRAC_PI_2, CN0_90_DBHZ),
    ] {
        let got = b.cn0_dbhz(eps);
        assert!(
            (got - want).abs() <= tol,
            "at {eps} rad: C/N0 {got:.17e} dBHz against {want:.17e}"
        );
    }
}

/// (G2) A budget that needs exactly the 30 deg `C/N0` breaks even at exactly `pi/6`: the
/// bisected root sits within `(64 eps_mach x 512 dB)/(dM/de) + 2^-60 (pi/2)` rad of it -- the
/// margin's rounding over its 120-digit slope, plus the bisection's last interval.
#[test]
fn the_break_even_elevation_is_the_root_of_the_margin() {
    let antenna = CommsAntenna::from_budget(&budget(CN0_30_DBHZ)).expect("the link closes");
    let tol = 64.0 * f64::EPSILON * 512.0 / MARGIN_SLOPE_30 + FRAC_PI_2 / 2.0_f64.powi(60);
    let got = antenna.break_even_elevation();
    assert!(
        (got - FRAC_PI_6).abs() <= tol,
        "break-even {got:.17e} rad against pi/6 = {FRAC_PI_6:.17e} (tol {tol:.1e})"
    );
}
