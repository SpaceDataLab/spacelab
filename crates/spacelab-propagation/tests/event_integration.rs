//! The end-to-end proof: a numerically propagated trajectory feeds the event detector through its
//! dense output, and the detected events land where the analytic orbit says they should.
//!
//! This is the composition `notes/15-integrators.md` argues is the whole point of dense output —
//! `find_events` evaluates `g(t)` at arbitrary `t` inside a step, and the propagator's interpolant
//! serves it without re-integration. Without dense output this would be the ring-buffer spline hack
//! GMAT is stuck with.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_coverage::{EventKind, find_events};
use spacelab_propagation::driver::{Controls, integrate_dopri5};
use spacelab_propagation::{DynamicalSystem, Tolerance};

/// Planar two-body, mu = 1.
struct Kepler;
impl DynamicalSystem<4> for Kepler {
    fn derivative(&self, _t: f64, y: &[f64; 4], dydt: &mut [f64; 4]) {
        let r3 = (y[0] * y[0] + y[1] * y[1]).powf(1.5);
        dydt[0] = y[2];
        dydt[1] = y[3];
        dydt[2] = -y[0] / r3;
        dydt[3] = -y[1] / r3;
    }
}

#[test]
fn events_on_a_propagated_orbit_land_where_the_analytic_orbit_says() {
    // Perigee at (1, 0), velocity (0, 1.3), counter-clockwise. The y-coordinate is 0 at perigee
    // (t=0), rises, returns to 0 at apogee (t = T/2), falls, and returns at t = T. So g(t) = y has
    // a falling crossing at T/2 and a rising crossing at T.
    let y0 = [1.0, 0.0, 0.0, 1.3];
    let energy: f64 = 0.5 * 1.3 * 1.3 - 1.0;
    let a = -1.0 / (2.0 * energy);
    let period = std::f64::consts::TAU * (a * a * a).sqrt();

    let traj = integrate_dopri5(
        &Kepler,
        0.0,
        &y0,
        period,
        0.01,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
    )
    .expect("integrated");

    // g(t) = y-coordinate, served entirely from the propagator's dense output.
    let g = |t: f64| traj.eval(t)[1];

    // coarse step well below the half-period so both crossings are bracketed
    let mut events = Vec::new();
    find_events(g, 1e-6, period - 1e-6, period / 40.0, 1e-9, |e| {
        events.push(e);
    });

    assert_eq!(
        events.len(),
        1,
        "expected the apogee crossing on the open interval: {events:?}"
    );
    let apogee = &events[0];
    assert_eq!(apogee.kind, EventKind::Falling, "y goes + → − at apogee");
    assert!(
        (apogee.at.estimate() - period / 2.0).abs() < 1e-6,
        "apogee crossing at {} vs analytic T/2 = {}",
        apogee.at.estimate(),
        period / 2.0
    );

    // and at that instant the orbit really is at apogee: x < 0, |r| = a(1+e)
    let at_apo = traj.eval(apogee.at.estimate());
    assert!(
        at_apo[0] < 0.0,
        "apogee is on the −x side, got x = {}",
        at_apo[0]
    );
    let r_apo = (at_apo[0] * at_apo[0] + at_apo[1] * at_apo[1]).sqrt();
    // e from angular momentum and energy: L=1.3, and r_apo = a(1+e)
    let l: f64 = 1.3;
    let e = (1.0 + 2.0 * energy * l * l).sqrt();
    assert!(
        (r_apo - a * (1.0 + e)).abs() < 1e-6,
        "apogee radius {r_apo} vs a(1+e) = {}",
        a * (1.0 + e)
    );
}

#[test]
fn a_ground_track_style_sign_function_is_detected_through_dense_output() {
    // A minimal analogue of the real pipeline: g(t) crosses zero as a propagated coordinate changes
    // sign, and the detector refines the crossing to the tolerance using only the interpolant.
    let y0 = [1.0, 0.0, 0.0, 1.1];
    let traj = integrate_dopri5(
        &Kepler,
        0.0,
        &y0,
        12.0,
        0.01,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-9),
    )
    .expect("integrated");

    // vx = radial-ish velocity x-component; it changes sign as the satellite rounds the orbit
    let g = |t: f64| traj.eval(t)[2];
    let mut n = 0;
    let mut last = f64::NEG_INFINITY;
    find_events(g, 0.0, 12.0, 0.05, 1e-10, |e| {
        // events strictly increasing in time, each bracket genuinely straddling zero
        assert!(e.at.lo > last, "events must be ordered");
        assert!(e.at.f_lo.signum() != e.at.f_hi.signum() || e.at.f_lo == 0.0);
        last = e.at.hi;
        n += 1;
    });
    assert!(
        n >= 2,
        "a 12 s arc of this orbit crosses vx=0 at least twice, found {n}"
    );
}
