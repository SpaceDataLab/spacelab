//! Sun-synchronous orbit design, closed against the physics it is derived from.
//!
//! * The inclination comes out at the **known ≈ 98°** for a low orbit.
//! * **Solve then evaluate is the identity**: feeding the solved inclination back into the rate
//!   formula returns the target rate.
//! * **Closure**: propagating the designed orbit under real `J₂` gravity, the node actually tracks
//!   the mean Sun — the design formula and the force model agree.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]
#![allow(clippy::cast_precision_loss)]

use spacelab_core::Angle;
use spacelab_elements::{
    KeplerianElements, SUN_SYNCHRONOUS_NODE_RATE, critical_inclination, repeat_ground_track_sma,
    sun_synchronous_inclination,
};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const OMEGA_EARTH: f64 = 7.292_115_9e-5;

/// Propagate an eccentric orbit at inclination `inc` under point mass + J₂ for `orbits` revolutions
/// and return the secular drift of the argument of perigee, rad/s.
fn measured_apsidal_rate(inc: Angle, orbits: f64) -> f64 {
    let els0 = KeplerianElements {
        semi_major_axis: RE + 900e3,
        eccentricity: 0.1,
        inclination: inc,
        raan: Angle::from_radians(0.3),
        arg_periapsis: Angle::from_radians(0.5),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    let (r0, v0) = els0.to_state();
    let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];
    let period = els0.period();

    let field = Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2_COEFF,
        },
    ));
    let traj = integrate(
        &Dop853,
        &Plain(field),
        0.0,
        &y0,
        orbits * period,
        60.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-3),
    )
    .expect("propagated");
    let yf = traj.final_state();
    let els_f = KeplerianElements::from_state(
        spacelab_core::Vec3::new(yf[0], yf[1], yf[2]),
        spacelab_core::Vec3::new(yf[3], yf[4], yf[5]),
        MU,
    );
    let mut d = els_f.arg_periapsis.radians() - els0.arg_periapsis.radians();
    while d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    while d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d / (orbits * period)
}

fn circular_sso(altitude: f64) -> KeplerianElements {
    let a = RE + altitude;
    let inc = sun_synchronous_inclination(a, 0.0, MU, RE, J2_COEFF, SUN_SYNCHRONOUS_NODE_RATE)
        .expect("a low orbit has a Sun-synchronous inclination");
    KeplerianElements {
        semi_major_axis: a,
        eccentricity: 0.0,
        inclination: inc,
        raan: Angle::from_radians(0.0),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    }
}

#[test]
fn a_low_sun_synchronous_orbit_is_inclined_about_98_degrees() {
    let inc = circular_sso(700e3).inclination.degrees();
    assert!(
        (98.0..98.4).contains(&inc),
        "700 km SSO inclination {inc:.3}° is not the textbook ≈98.2°"
    );
}

#[test]
fn solving_the_inclination_then_evaluating_the_rate_is_the_identity() {
    // For a spread of altitudes, the solved inclination must reproduce exactly the target rate.
    for altitude in [400e3, 700e3, 1200e3] {
        let els = circular_sso(altitude);
        let rate = els.nodal_regression_rate(RE, J2_COEFF);
        let rel = (rate - SUN_SYNCHRONOUS_NODE_RATE).abs() / SUN_SYNCHRONOUS_NODE_RATE;
        assert!(
            rel < 1e-14,
            "at {altitude} m the designed rate {rate:.6e} ≠ target ({rel:.2e})"
        );
    }
}

#[test]
fn the_designed_orbit_tracks_the_mean_sun_under_j2() {
    // Propagate the designed orbit under point mass + J₂ and measure the node's actual drift; it must
    // match the Sun's rate — the whole point of the design.
    let els = circular_sso(700e3);
    let (r0, v0) = els.to_state();
    let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];
    let period = els.period();
    let orbits = 60.0;

    let field = Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2_COEFF,
        },
    ));
    let traj = integrate(
        &Dop853,
        &Plain(field),
        0.0,
        &y0,
        orbits * period,
        60.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-3),
    )
    .expect("propagated");

    // RAAN from the orbit normal L = r × v: L ∝ (sin i sinΩ, −sin i cosΩ, cos i).
    let raan = |y: &[f64; 6]| {
        let lx = y[1] * y[5] - y[2] * y[4];
        let ly = y[2] * y[3] - y[0] * y[5];
        lx.atan2(-ly)
    };
    let mut drift = raan(&traj.final_state()) - raan(&y0);
    while drift > std::f64::consts::PI {
        drift -= std::f64::consts::TAU;
    }
    while drift < -std::f64::consts::PI {
        drift += std::f64::consts::TAU;
    }
    let measured = drift / (orbits * period);

    let rel = (measured - SUN_SYNCHRONOUS_NODE_RATE).abs() / SUN_SYNCHRONOUS_NODE_RATE;
    println!(
        "SSO node rate: measured {measured:.6e}, Sun {SUN_SYNCHRONOUS_NODE_RATE:.6e} rad/s ({:.2}%, {:.4}°/day)",
        100.0 * rel,
        measured * 86400.0 * 180.0 / std::f64::consts::PI
    );
    assert!(measured > 0.0, "a Sun-synchronous node advances eastward");
    assert!(
        rel < 0.02,
        "node rate {measured:.6e} vs Sun {SUN_SYNCHRONOUS_NODE_RATE:.6e} ({:.1}%)",
        100.0 * rel
    );
}

#[test]
fn the_nodal_rate_matches_the_known_700km_51deg_value() {
    // Independent check of the rate formula: a 700 km, 51.6° orbit regresses at ≈ −4.31°/day.
    let els = KeplerianElements {
        semi_major_axis: RE + 700e3,
        eccentricity: 0.0,
        inclination: Angle::from_degrees(51.6),
        raan: Angle::from_radians(0.0),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    let deg_per_day =
        els.nodal_regression_rate(RE, J2_COEFF) * 86400.0 * 180.0 / std::f64::consts::PI;
    assert!(
        (-4.4..-4.2).contains(&deg_per_day),
        "51.6° node rate {deg_per_day:.4}°/day off the known −4.31"
    );
}

#[test]
fn too_high_an_orbit_has_no_sun_synchronous_inclination() {
    // At geosynchronous altitude J₂ cannot precess the node a full turn per year: no solution.
    assert!(
        sun_synchronous_inclination(42_164e3, 0.0, MU, RE, J2_COEFF, SUN_SYNCHRONOUS_NODE_RATE)
            .is_none(),
        "a GEO-altitude orbit cannot be Sun-synchronous"
    );
}

#[test]
fn the_apsidal_rate_matches_the_measured_perigee_precession() {
    // The ω̇ formula, unlike Ω̇, was not yet checked against the dynamics. At 45° the perigee should
    // advance; the measured secular rate must match the first-order formula.
    let inc = Angle::from_degrees(45.0);
    let els = KeplerianElements {
        semi_major_axis: RE + 900e3,
        eccentricity: 0.1,
        inclination: inc,
        raan: Angle::from_radians(0.0),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    let predicted = els.apsidal_precession_rate(RE, J2_COEFF);
    let measured = measured_apsidal_rate(inc, 40.0);
    let rel = (measured - predicted).abs() / predicted.abs();
    println!(
        "apsidal rate: measured {measured:.6e}, formula {predicted:.6e} rad/s ({:.2}%)",
        100.0 * rel
    );
    assert!(
        predicted > 0.0 && measured > 0.0,
        "at 45° the perigee advances"
    );
    assert!(
        rel < 0.03,
        "apsidal rate {measured:.6e} vs formula {predicted:.6e} ({:.1}%)",
        100.0 * rel
    );
}

#[test]
fn the_perigee_is_frozen_at_the_critical_inclination() {
    // ω̇ ∝ 5cos²i − 1 vanishes at i = acos(1/√5); the formula gives zero and the propagated perigee
    // barely moves compared with an off-critical orbit — the Molniya condition.
    let crit = critical_inclination();
    let els = KeplerianElements {
        semi_major_axis: RE + 900e3,
        eccentricity: 0.1,
        inclination: crit,
        raan: Angle::from_radians(0.0),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    assert!(
        els.apsidal_precession_rate(RE, J2_COEFF).abs() < 1e-16,
        "formula ω̇ not zero at critical i"
    );

    let at_critical = measured_apsidal_rate(crit, 40.0).abs();
    let off_critical = measured_apsidal_rate(Angle::from_degrees(45.0), 40.0).abs();
    println!("perigee drift: critical {at_critical:.3e}, off-critical {off_critical:.3e} rad/s");
    assert!(
        at_critical < 0.05 * off_critical,
        "perigee not frozen at critical i: {at_critical:.3e} vs {off_critical:.3e}"
    );
}

#[test]
fn the_repeat_ground_track_recovers_the_landsat_orbit() {
    // Landsat-8: 233 revolutions in 16 days, Sun-synchronous (i ≈ 98.2°). Its semi-major axis is the
    // published WRS-2 value 7077.7 km (mean altitude ≈ 699.6 km) — a non-circular check of the
    // composite J₂ rate model.
    let a = repeat_ground_track_sma(
        233,
        16,
        0.0,
        Angle::from_degrees(98.2),
        MU,
        RE,
        J2_COEFF,
        OMEGA_EARTH,
    );
    println!(
        "Landsat repeat sma: {:.1} km (alt {:.1} km)",
        a / 1e3,
        (a - RE) / 1e3
    );
    assert!(
        (7_074e3..7_081e3).contains(&a),
        "233/16 semi-major axis {a:.0} m off the 7077.7 km Landsat value"
    );
}

#[test]
fn the_repeat_solution_is_self_consistent() {
    // Solving for a and then evaluating the repeat parameter Q = (Ṁ + ω̇)/(ω⊕ − Ω̇) must return the
    // requested revs/days — the solver inverts exactly the resonance it is built on.
    let (revs, days) = (233u32, 16u32);
    let e = 0.001;
    let inc = Angle::from_degrees(98.2);
    let a = repeat_ground_track_sma(revs, days, e, inc, MU, RE, J2_COEFF, OMEGA_EARTH);

    let els = KeplerianElements {
        semi_major_axis: a,
        eccentricity: e,
        inclination: inc,
        raan: Angle::from_radians(0.0),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    let q = els.draconitic_mean_motion(RE, J2_COEFF)
        / (OMEGA_EARTH - els.nodal_regression_rate(RE, J2_COEFF));
    let target = f64::from(revs) / f64::from(days);
    let rel = (q - target).abs() / target;
    assert!(
        rel < 1e-9,
        "repeat parameter {q:.6} ≠ {target:.6} ({rel:.2e})"
    );
}

#[test]
fn the_draconitic_period_matches_the_node_crossing_interval() {
    // Direct check of the anomalistic/nodal mean motion (the Ṁ correction the repeat solver leans on
    // and that Ω̇, ω̇ do not cover): the mean interval between ascending-node crossings of a real J₂
    // trajectory must equal 2π / (Ṁ + ω̇).
    let els = KeplerianElements {
        semi_major_axis: RE + 900e3,
        eccentricity: 0.05,
        inclination: Angle::from_degrees(60.0),
        raan: Angle::from_radians(0.0),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    let (r0, v0) = els.to_state();
    let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];
    let period = els.period();
    let span = 12.0 * period;

    let field = Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2_COEFF,
        },
    ));
    let traj = integrate(
        &Dop853,
        &Plain(field),
        0.0,
        &y0,
        span,
        60.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-3),
    )
    .expect("propagated");

    // scan z for ascending crossings (− → +), bisect each to the second
    let dt = 2.0;
    let mut crossings = Vec::new();
    let mut t = 0.0;
    let mut prev = traj.eval(0.0)[2];
    while t < span {
        t += dt;
        let z = traj.eval(t)[2];
        if prev < 0.0 && z >= 0.0 {
            let (mut lo, mut hi) = (t - dt, t);
            for _ in 0..40 {
                let mid = f64::midpoint(lo, hi);
                if traj.eval(mid)[2] < 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            crossings.push(f64::midpoint(lo, hi));
        }
        prev = z;
    }

    assert!(
        crossings.len() >= 10,
        "too few node crossings found: {}",
        crossings.len()
    );
    let measured = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;
    let predicted = std::f64::consts::TAU / els.draconitic_mean_motion(RE, J2_COEFF);
    let rel = (measured - predicted).abs() / predicted;
    println!(
        "nodal period: measured {measured:.4} s, formula {predicted:.4} s ({:.3}%)",
        100.0 * rel
    );
    // The first-order secular formula can only match the *osculating* node crossings to O(J₂): the
    // crossings carry a short-period wobble in z that no secular mean removes. That ~0.1–0.2 % floor,
    // not a formula error, is what this bounds — the secular rates themselves (Ω̇, ω̇) check to 0.37 %.
    assert!(
        rel < 5e-3,
        "nodal period {measured:.4} vs formula {predicted:.4} ({:.2}%)",
        100.0 * rel
    );
}
