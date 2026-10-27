//! The full model: B08 Tc, B06 corrections, semiannual and seasonal-latitudinal terms, the
//! combination into T_inf, input checks, and the size of the 200 / 700 km steps.

#![allow(
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::unwrap_used,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::float_cmp
)] // printed values digit for digit, the papers' notation, test-only unwraps

use spacelab_propagation::jacchia_bowman::calendar::year_and_day_of_year;
use spacelab_propagation::jacchia_bowman::constants::*;
use spacelab_propagation::jacchia_bowman::variations::{
    delta_tc_b06, fs_bar, high_altitude_factor, semiannual_log10, tc,
};
use spacelab_propagation::jacchia_bowman::{
    Error, Input, Profile, SolarIndices, evaluate, static_state,
};
use std::f64::consts::TAU;

fn indices(f10: f64, f10_bar: f64) -> SolarIndices {
    SolarIndices {
        f10,
        f10_bar,
        s10: f10 * 0.9,
        s10_bar: f10_bar * 0.9,
        m10: f10 * 0.95,
        m10_bar: f10_bar * 0.95,
        y10: f10 * 1.05,
        y10_bar: f10_bar * 1.05,
    }
}

fn input(z: f64, lat_deg: f64, hour_angle_deg: f64) -> Input {
    Input {
        mjd: 54_800.25,
        sun_ra: 1.0,
        sun_dec: (-20f64).to_radians(),
        sat_ra: 1.0 + hour_angle_deg.to_radians(),
        sat_lat: lat_deg.to_radians(),
        height_km: z,
        indices: indices(150.0, 140.0),
        dtc_storm: 25.0,
    }
}

/// (G2) B08 eqs. (1)-(2) evaluated by hand for one set of indices.
#[test]
fn tc_equation() {
    let ix = SolarIndices {
        f10: 160.0,
        f10_bar: 150.0,
        s10: 140.0,
        s10_bar: 145.0,
        m10: 155.0,
        m10_bar: 148.0,
        y10: 130.0,
        y10_bar: 135.0,
    };
    let w = (150.0f64 / 240.0).powf(0.25);
    let fs = 150.0 * w + 145.0 * (1.0 - w);
    let expect = 392.4 + 3.227 * fs + 0.298 * 10.0 + 2.259 * (-5.0) + 0.312 * 7.0 + 0.178 * (-5.0);
    assert!((fs_bar(&ix) - fs).abs() < 1e-12);
    assert!((tc(&ix) - expect).abs() < 1e-10);
}

/// (G2) B08 eq. (1) as printed has no clamp: at F10bar = 300 the weight is
/// (300/240)^(1/4) = 1.0574 > 1 and the S10bar weight is negative.
#[test]
fn fs_weight_not_clamped() {
    let ix = SolarIndices {
        f10: 300.0,
        f10_bar: 300.0,
        s10: 250.0,
        s10_bar: 250.0,
        m10: 0.0,
        m10_bar: 0.0,
        y10: 0.0,
        y10_bar: 0.0,
    };
    let w = 1.25f64.powf(0.25);
    assert!((w - 1.0574).abs() < 5e-5);
    assert!((fs_bar(&ix) - (300.0 * w + 250.0 * (1.0 - w))).abs() < 1e-12);
    assert!(fs_bar(&ix) > 300.0);
}

/// (G2) The semiannual term is F(z) G(t) with no floor or cut-off of F(z): for indices that
/// make F(z) negative the term is still the product.
#[test]
fn semiannual_no_floor() {
    let ix = SolarIndices {
        f10: 200.0,
        f10_bar: 200.0,
        s10: 0.0,
        s10_bar: 0.0,
        m10: 0.0,
        m10_bar: 0.0,
        y10: 0.0,
        y10_bar: 0.0,
    };
    let z = 90.0;
    let fsmj = 200.0;
    let zz = z / 1000.0;
    let b = SEMIANNUAL_B;
    let fz =
        b[0] + b[1] * fsmj + b[2] * zz * fsmj + b[3] * zz * zz * fsmj + b[4] * zz * fsmj * fsmj;
    assert!(fz < 0.0, "{fz}");
    let doy = 100.0;
    let w = TAU * (doy - 1.0) / 365.0;
    let c = SEMIANNUAL_C;
    let fsm = 200.0;
    let gt = c[0]
        + c[1] * w.sin()
        + c[2] * w.cos()
        + c[3] * (2.0 * w).sin()
        + c[4] * (2.0 * w).cos()
        + fsm
            * (c[5]
                + c[6] * w.sin()
                + c[7] * w.cos()
                + c[8] * (2.0 * w).sin()
                + c[9] * (2.0 * w).cos());
    assert!((semiannual_log10(&ix, z, doy) - fz * gt).abs() < 1e-14);
}

/// (G2) Day of year from MJD: MJD 36204 = 1958 Jan 1 (SR332 p. 44) gives t = 1.0; 1969 Jan
/// 20.8 (SR313 Sec. 15) gives 20.8; 2000 Dec 31 (leap year) gives 366; MJD 0 is 1858 Nov 17.
#[test]
fn calendar() {
    assert_eq!(year_and_day_of_year(36204.0), (1958, 1.0));
    let (y, t) = year_and_day_of_year(36204.0 + 11.0 * 365.0 + 3.0 + 19.8);
    assert_eq!(y, 1969);
    assert!((t - 20.8).abs() < 1e-9);
    // 2000 Jan 1 = 1958 Jan 1 + 42 years with 11 leap days (1960..2000 step 4, 2000 excluded)
    let jan1_2000 = 36204.0 + 42.0 * 365.0 + 10.0;
    assert_eq!(year_and_day_of_year(jan1_2000), (2000, 1.0));
    assert_eq!(year_and_day_of_year(jan1_2000 + 365.0), (2000, 366.0));
    assert_eq!(year_and_day_of_year(jan1_2000 + 366.0), (2001, 1.0));
    // 1858 Nov 17 is day 321 of a common year
    assert_eq!(year_and_day_of_year(0.0), (1858, 321.0));
}

/// (G2) T_inf = (Tc + Delta Tc_B06) x (T_l/Tc) + dTc_storm, and the density is the static
/// density at T_inf times 10^(semiannual + seasonal-latitudinal) times the high-altitude factor.
#[test]
fn combination_into_t_inf_and_density() {
    let inp = input(400.0, 30.0, 40.0);
    let out = evaluate(&inp).unwrap();
    let t = out.terms;
    assert!((out.t_inf - ((t.tc + t.delta_tc_b06) * t.diurnal_ratio + 25.0)).abs() < 1e-9);
    let s = static_state(&Profile::new(out.t_inf).unwrap(), 400.0);
    assert_eq!(s.density, t.static_density);
    let expect = s.density
        * 10f64.powf(t.semiannual_log10 + t.seasonal_latitudinal_log10)
        * t.high_altitude_factor;
    assert!((out.density / expect - 1.0).abs() < 1e-14);
    assert!((t.local_solar_time - (12.0 + 40.0 / 15.0)).abs() < 1e-9);
    assert!(t.delta_tc_b06 != 0.0);
}

/// (G2) B06 Delta Tc applies only inside 200 <= z < 700 km and is exactly 0 outside.
#[test]
fn delta_tc_band_limits() {
    for z in [150.0, 199.999, 700.0, 701.0, 1200.0] {
        assert_eq!(delta_tc_b06(170.0, 9.0, 0.5, z), 0.0, "z {z}");
    }
    for z in [200.0, 249.0, 250.0, 699.9] {
        assert!(delta_tc_b06(170.0, 9.0, 0.5, z) != 0.0, "z {z}");
    }
}

/// (G2) B06 eqs. (7) and (8) agree in value at 250 km (the census found residual <= 3e-6,
/// print rounding) and not in slope (B06 p. 784 claims both): measured over a grid.
#[test]
fn b06_value_continuity_at_250km() {
    let mut worst_value: f64 = 0.0;
    let mut worst_slope: f64 = 0.0;
    let h = 1e-6;
    for f10 in [70.0, 150.0, 250.0] {
        for lst in [0.0, 5.0, 11.0, 17.0, 23.0] {
            for lat in [0.0f64, 40.0, 80.0] {
                let lat = lat.to_radians();
                let below = delta_tc_b06(f10, lst, lat, 250.0 - h);
                let at = delta_tc_b06(f10, lst, lat, 250.0);
                worst_value = worst_value.max((at - below).abs());
                let s7 = (delta_tc_b06(f10, lst, lat, 251.0) - at) / 1.0;
                let s8 = (at - delta_tc_b06(f10, lst, lat, 249.0)) / 1.0;
                worst_slope = worst_slope.max((s7 - s8).abs());
            }
        }
    }
    println!(
        "B06 at 250 km: worst value jump {worst_value:.2e} K, worst slope jump {worst_slope:.3} K/km"
    );
    assert!(worst_value < 1e-4);
    assert!(worst_slope > 0.05);
}

/// Step of T_inf and of log10 rho across a band edge z_e: values at z_e - eps and z_e.
fn step_at(z_edge: f64, lat_deg: f64, ha_deg: f64, f10: f64) -> (f64, f64) {
    let mut a = input(z_edge - 1e-9, lat_deg, ha_deg);
    a.indices = indices(f10, f10);
    let mut b = a;
    b.height_km = z_edge;
    let oa = evaluate(&a).unwrap();
    let ob = evaluate(&b).unwrap();
    (ob.t_inf - oa.t_inf, (ob.density / oa.density).log10())
}

/// (G2) The steps at 200 and 700 km that the band limits of B06 eqs. (7)-(8) produce, measured
/// over latitude, local time and solar flux. Reported in REPORT.md.
#[test]
fn band_edge_steps() {
    for edge in [200.0, 700.0] {
        let (mut max_t, mut max_r) = (0.0f64, 0.0f64);
        for f10 in [70.0, 150.0, 250.0] {
            for lat in [-80.0, -45.0, 0.0, 45.0, 80.0] {
                for ha in (0..24).map(|h| -180.0 + 15.0 * h as f64) {
                    let (dt, dr) = step_at(edge, lat, ha, f10);
                    max_t = max_t.max(dt.abs());
                    max_r = max_r.max(dr.abs());
                }
            }
        }
        println!(
            "step at {edge} km: max |dT_inf| = {max_t:.2} K, max |d log10 rho| = {max_r:.4} ({:.1} %)",
            100.0 * (10f64.powf(max_r) - 1.0)
        );
        assert!(max_t > 1.0);
    }
}

/// (G2) Input checks: below 90 km and non-finite inputs are rejected; a T_inf below the
/// range where eqs. (11) and (13) define a profile is rejected.
#[test]
fn input_errors() {
    assert!(matches!(
        evaluate(&input(89.9, 0.0, 0.0)),
        Err(Error::BelowLowerBoundary { .. })
    ));
    let mut bad = input(400.0, 0.0, 0.0);
    bad.indices.f10 = f64::NAN;
    assert_eq!(evaluate(&bad), Err(Error::NonFiniteInput));
    let mut cold = input(400.0, 0.0, 0.0);
    cold.dtc_storm = -2000.0;
    assert!(matches!(
        evaluate(&cold),
        Err(Error::NoTemperatureProfile { .. })
    ));
    assert!(evaluate(&input(90.0, 0.0, 0.0)).is_ok());
}

/// (G5, graphical) B06 Fig. 15 (p. 789): density factor of eq. (10) read from the plot, about
/// 1.58 at 1500 km and F10bar = 70, 0.79 at 1500 km and 220, 2.6 at 2500 km and 70, 3.6 at
/// 3500 km and 70 (read by eye, +-0.05).
#[test]
fn b06_fig15_high_altitude_factor() {
    for (fb, z, read) in [
        (70.0, 1500.0, 1.58),
        (220.0, 1500.0, 0.79),
        (70.0, 2500.0, 2.6),
        (70.0, 3500.0, 3.6),
    ] {
        let f = high_altitude_factor(fb, z);
        assert!((f - read).abs() < 0.05, "F10bar {fb} z {z}: {f}");
    }
}

/// (G2) The evaluation is finite and positive over a grid of heights (90-3000 km), latitudes,
/// local times and solar activity, and the density falls with height up to 2000 km. Above,
/// see `density_rises_above_2000km_at_low_flux`.
#[test]
fn smoke_grid() {
    for f10 in [65.0, 150.0, 300.0] {
        for lat in [-89.0, -30.0, 0.0, 30.0, 89.0] {
            for ha in [-179.0, -90.0, 0.0, 90.0, 179.0] {
                let mut prev = f64::MAX;
                for z in [90.0, 120.0, 200.0, 400.0, 800.0, 1200.0, 2000.0, 3000.0] {
                    let mut inp = input(z, lat, ha);
                    inp.indices = indices(f10, f10);
                    let o = evaluate(&inp).unwrap();
                    assert!(
                        o.density.is_finite()
                            && o.density > 0.0
                            && (z > 2000.0 || o.density < prev),
                        "f10 {f10} lat {lat} ha {ha} z {z}: {} after {prev}; T_inf {} factor {}",
                        o.density,
                        o.t_inf,
                        o.terms.high_altitude_factor
                    );
                    prev = o.density;
                }
            }
        }
    }
}

/// (G2) Finding: at low solar flux and night-time T_inf (about 700 K) the B06 eq. (10)
/// factor grows with height faster than the hydrogen-dominated static density falls, so the
/// corrected density increases with height above about 2000 km. No upper limit is printed for
/// eq. (10); the J70 tables stop at 2500 km. Measured, not changed.
#[test]
fn density_rises_above_2000km_at_low_flux() {
    let mut inp = input(2000.0, -60.0, 180.0);
    inp.indices = indices(65.0, 65.0);
    inp.dtc_storm = 0.0;
    let mut rho = Vec::new();
    for z in [1500.0, 2000.0, 2500.0, 3000.0, 3500.0] {
        inp.height_km = z;
        let o = evaluate(&inp).unwrap();
        println!(
            "{z} km: T_inf {:.1} rho {:.4e} factor {:.3} static {:.4e}",
            o.t_inf, o.density, o.terms.high_altitude_factor, o.terms.static_density
        );
        rho.push(o.density);
    }
    assert!(rho[1] < rho[0]);
    assert!(rho[4] > rho[2], "density at 3500 km above that at 2500 km");
}

/// (G1) One full evaluation against an independent Maxima evaluation of the same printed
/// equations (derivations/jacchia_bowman_full_model_terms.mac / .out): Tc, B06 Delta Tc, diurnal ratio,
/// T_inf, semiannual and seasonal-latitudinal terms, static and final log10 rho.
#[test]
fn full_evaluation_matches_maxima() {
    let inp = Input {
        mjd: 54800.25,
        sun_ra: 1.0,
        sun_dec: (-20f64).to_radians(),
        sat_ra: 1.0 + 40f64.to_radians(),
        sat_lat: 30f64.to_radians(),
        height_km: 400.0,
        indices: SolarIndices {
            f10: 150.0,
            f10_bar: 140.0,
            s10: 135.0,
            s10_bar: 126.0,
            m10: 142.5,
            m10_bar: 133.0,
            y10: 157.5,
            y10_bar: 147.0,
        },
        dtc_storm: 25.0,
    };
    let o = evaluate(&inp).unwrap();
    let t = o.terms;
    let close =
        |a: f64, b: f64, tol: f64| assert!((a - b).abs() <= tol * b.abs().max(1.0), "{a} vs {b}");
    assert_eq!(t.day_of_year, 335.25);
    close(t.tc, 866.6286414162965, 1e-13);
    close(t.delta_tc_b06, 4.897405769257427, 1e-11);
    close(t.diurnal_ratio, 1.2402037635168295, 1e-13);
    close(o.t_inf, 1105.86988372247, 1e-13);
    close(t.semiannual_log10, -0.03219820136816526, 1e-12);
    close(t.seasonal_latitudinal_log10, 1.2553383227638623e-6, 1e-9);
    close(t.static_density.log10(), -11.321079941912453, 1e-11);
    close(o.density.log10(), -11.353276887942297, 1e-11);
}
