//! Keplerian elements, checked by the identities they must satisfy — not a reference table.
//!
//! * **Round-trip** `state → elements → state` reproduces the state, including the circular,
//!   equatorial and circular-equatorial singular cases where a naïve conversion returns `NaN`.
//! * **Invariance**: the osculating elements of a two-body orbit are constant except the anomaly.
//! * **Cross-check**: advancing the anomaly in closed form (`propagate`) agrees with numerically
//!   integrating the two-body ODE — two independent implementations of the same flow.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]

use spacelab_core::{Angle, Vec3};
use spacelab_elements::KeplerianElements;
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;

fn deg(d: f64) -> Angle {
    Angle::from_degrees(d)
}

fn elements(a: f64, e: f64, i: f64, raan: f64, argp: f64, nu: f64) -> KeplerianElements {
    KeplerianElements {
        semi_major_axis: a,
        eccentricity: e,
        inclination: deg(i),
        raan: deg(raan),
        arg_periapsis: deg(argp),
        true_anomaly: deg(nu),
        mu: MU,
    }
}

/// Relative difference of two vectors, scaled by the reference length.
fn rel(a: Vec3, b: Vec3) -> f64 {
    (a - b).norm() / b.norm()
}

/// A spread of orbit types: near-circular LEO, SSO, Molniya, near-GEO, retrograde, polar.
fn cases() -> [KeplerianElements; 6] {
    [
        elements(7_000e3, 0.001, 51.6, 40.0, 30.0, 10.0),
        elements(7_200e3, 0.1, 98.0, 120.0, 200.0, 45.0),
        elements(26_600e3, 0.74, 63.4, 10.0, 270.0, 100.0),
        elements(42_164e3, 0.0005, 0.05, 80.0, 60.0, 300.0),
        elements(8_000e3, 0.3, 145.0, 200.0, 15.0, 250.0),
        elements(7_500e3, 0.02, 89.9, 5.0, 350.0, 123.0),
    ]
}

#[test]
fn state_to_elements_and_back_reproduces_the_state() {
    for els in cases() {
        let (r, v) = els.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        let (r2, v2) = back.to_state();
        assert!(
            rel(r2, r) < 1e-14,
            "position round-trip {:.2e} for {els:?}",
            rel(r2, r)
        );
        assert!(
            rel(v2, v) < 1e-14,
            "velocity round-trip {:.2e} for {els:?}",
            rel(v2, v)
        );

        // for the non-singular cases the elements themselves round-trip too
        if els.eccentricity > 1e-3 && els.inclination.degrees() > 1.0 {
            assert!((back.semi_major_axis - els.semi_major_axis).abs() < 1e-3);
            assert!((back.eccentricity - els.eccentricity).abs() < 1e-12);
            assert!((back.inclination.radians() - els.inclination.radians()).abs() < 1e-12);
            assert!((back.raan.radians() - els.raan.radians()).abs() < 1e-12);
            assert!((back.arg_periapsis.radians() - els.arg_periapsis.radians()).abs() < 1e-11);
            assert!((back.true_anomaly.radians() - els.true_anomaly.radians()).abs() < 1e-11);
        }
    }
}

#[test]
fn the_singular_cases_round_trip_without_nan() {
    // ω undefined (circular inclined), Ω undefined (equatorial elliptical), both (circular
    // equatorial): the geometry is carried in the anomaly, and the state must still reconstruct.
    //
    // Each branch resolves its quadrant from the sign of one component — r_z, e_y, r·v, r_y — and a
    // single sample per branch exercises only one side of that test, leaving the correction that
    // handles the other side free to be anything. The anomaly and the periapsis longitude are swept
    // over the full turn so every sign is visited.
    let mut singular = Vec::new();
    for j in 0..16 {
        let nu = 360.0 * f64::from(j) / 16.0;
        singular.push(elements(7_000e3, 0.0, 45.0, 30.0, 0.0, nu)); // circular inclined
        singular.push(elements(7_000e3, 0.0, 0.0, 0.0, 0.0, nu)); // circular equatorial
        // Retrograde equatorial. `i = 180°` takes the same no-node branch as `i = 0`, but the
        // substitute longitude is measured the other way round; sweeping only the prograde side
        // left that reversal free to be wrong, and it was — the state came back reflected in y.
        singular.push(elements(7_000e3, 0.0, 180.0, 0.0, 0.0, nu));
        for k in 0..8 {
            // elliptical equatorial: ω is the longitude of periapsis, and its sign matters too
            singular.push(elements(7_000e3, 0.2, 0.0, 0.0, 45.0 * f64::from(k), nu));
            singular.push(elements(7_000e3, 0.2, 180.0, 0.0, 45.0 * f64::from(k), nu));
        }
    }
    for els in singular {
        let (r, v) = els.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        assert!(back.semi_major_axis.is_finite() && back.eccentricity.is_finite());
        let (r2, v2) = back.to_state();
        assert!(
            rel(r2, r) < 1e-14,
            "singular position round-trip {:.2e} for {els:?}",
            rel(r2, r)
        );
        assert!(
            rel(v2, v) < 1e-14,
            "singular velocity round-trip {:.2e} for {els:?}",
            rel(v2, v)
        );
    }
}

#[test]
fn mean_and_true_anomaly_are_mutual_inverses() {
    // ν → M → ν over a range of eccentricities and anomalies (Kepler's equation solved back).
    for &e in &[0.0, 0.01, 0.3, 0.74, 0.95] {
        for nu_deg in [1.0, 47.0, 123.0, 200.0, 340.0] {
            let els = elements(1e7, e, 30.0, 0.0, 0.0, nu_deg);
            let m = els.mean_anomaly();
            let recovered = els.at_mean_anomaly(m).true_anomaly.radians();
            let want = els.true_anomaly.radians();
            let d = (recovered - want).rem_euclid(Angle::TAU);
            let d = d.min(Angle::TAU - d);
            assert!(d < 1e-12, "ν→M→ν off by {d:.2e} at e={e}, ν={nu_deg}°");
        }
    }
}

#[test]
fn period_matches_the_closed_form_for_a_known_orbit() {
    // 700 km circular: T = 2π√(a³/μ) ≈ 5926 s.
    let els = elements(RE + 700e3, 0.0, 51.6, 0.0, 0.0, 0.0);
    let a = RE + 700e3;
    let want = Angle::TAU * (a * a * a / MU).sqrt();
    assert!(
        (els.period() - want).abs() < 1e-6,
        "period {} vs {want}",
        els.period()
    );
    assert!(
        (5900.0..5950.0).contains(&els.period()),
        "period {} s out of range",
        els.period()
    );
}

#[test]
fn osculating_elements_are_invariant_along_a_two_body_orbit() {
    // Under pure two-body motion a, e, i, Ω, ω are constants of the motion; only ν advances.
    let els0 = elements(9_000e3, 0.15, 55.0, 70.0, 130.0, 20.0);
    let (r0, v0) = els0.to_state();
    let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];

    let traj = integrate(
        &Dop853,
        &Plain(Newtonian(PointMass { mu: MU })),
        0.0,
        &y0,
        3.0 * els0.period(),
        60.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-3),
    )
    .expect("propagated");

    let yf = traj.final_state();
    let els = KeplerianElements::from_state(
        Vec3::new(yf[0], yf[1], yf[2]),
        Vec3::new(yf[3], yf[4], yf[5]),
        MU,
    );
    assert!(
        (els.semi_major_axis - els0.semi_major_axis).abs() < 1e-3,
        "a drifted"
    );
    assert!(
        (els.eccentricity - els0.eccentricity).abs() < 1e-10,
        "e drifted"
    );
    assert!(
        (els.inclination.radians() - els0.inclination.radians()).abs() < 1e-10,
        "i drifted"
    );
    assert!(
        (els.raan.radians() - els0.raan.radians()).abs() < 1e-10,
        "Ω drifted"
    );
    assert!(
        (els.arg_periapsis.radians() - els0.arg_periapsis.radians()).abs() < 1e-9,
        "ω drifted"
    );
}

#[test]
fn closed_form_propagation_matches_the_numerical_two_body_flow() {
    // The analytic Kepler flow (advance M, solve back to ν) and numerically integrating ẍ = −μr/r³
    // are two implementations of the same trajectory; they must agree to the integrator tolerance.
    for els0 in cases() {
        let (r0, v0) = els0.to_state();
        let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];
        let dt = 0.37 * els0.period(); // an arbitrary fraction of an orbit

        let (r_a, v_a) = els0.propagate(dt).to_state();

        let traj = integrate(
            &Dop853,
            &Plain(Newtonian(PointMass { mu: MU })),
            0.0,
            &y0,
            dt,
            60.0,
            Controls::new(Tolerance::new(1e-13, 1e-13), 1e-3),
        )
        .expect("propagated");
        let yf = traj.final_state();
        let r_n = Vec3::new(yf[0], yf[1], yf[2]);
        let v_n = Vec3::new(yf[3], yf[4], yf[5]);

        assert!(
            rel(r_a, r_n) < 1e-9,
            "position mismatch {:.2e} for {els0:?}",
            rel(r_a, r_n)
        );
        assert!(
            rel(v_a, v_n) < 1e-9,
            "velocity mismatch {:.2e} for {els0:?}",
            rel(v_a, v_n)
        );
    }
}

// Ω is compared with `==`: below the threshold the convention is that it is set to the literal
// zero, and "approximately folded away" is not a thing the convention can mean.
#[allow(clippy::float_cmp)]
#[test]
fn the_equatorial_classification_is_scale_free() {
    // The equatorial test is `|n| ≤ ε·|h|`, not `|n| ≤ ε`. Since `|n| = |h| sin i`, dividing by `|h|`
    // is exactly what makes the classification a statement about the inclination instead of about
    // the orbit's size. No round-trip can see the difference: Ω and the argument of latitude are
    // both measured from the same node vector, so any node direction rebuilds the same state. The
    // reported Ω can — below the threshold the convention is Ω = 0, and it has to hold at every
    // orbit radius, from LEO to geostationary.
    for &a in &[7_000e3, 42_164e3] {
        // sin i = 1e-13 is two decades under the 1e-11 threshold and eight above the node vector's
        // own round-off (|n| ≈ 5e-3 m²/s against ~1e-5 of cancellation error inside h).
        let flat = elements(a, 0.2, 1e-13_f64.to_degrees(), 30.0, 40.0, 70.0);
        let (r, v) = flat.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        assert_eq!(
            back.raan.radians(),
            0.0,
            "a={a}: Ω was not folded into the longitude of periapsis"
        );
        assert!(
            (back.arg_periapsis.degrees() - 70.0).abs() < 1e-6,
            "a={a}: ϖ = {} should be Ω+ω",
            back.arg_periapsis.degrees()
        );

        // Two decades above the threshold the node is real and must be reported.
        let tilted = elements(a, 0.2, 1e-9_f64.to_degrees(), 30.0, 40.0, 70.0);
        let (r, v) = tilted.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        assert!(
            (back.raan.degrees() - 30.0).abs() < 1e-6,
            "a={a}: Ω = {} should be 30°",
            back.raan.degrees()
        );
    }
}
