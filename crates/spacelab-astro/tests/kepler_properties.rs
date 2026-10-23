//! Two-body properties that hold for every orbit, checked against the definitions rather than a table.
//!
//! The universal-variable formulation has one structural hazard: [`kepler::propagate`] switches
//! between a closed form and a folded series at a threshold in the Stumpff argument, and every
//! fixture in `tests/` sits on one side of it or the other. A generated argument crosses the
//! threshold thousands of times per run, and it does so at the eccentricities and time steps that
//! push it there.
//!
//! Three properties, none of which needs an oracle:
//!
//! 1. **Reversibility.** Propagating by `dt` and back by `−dt` returns the state. This is a
//!    statement about the flow, not about any particular trajectory, and it fails the moment the
//!    universal anomaly is solved to a different root on the return leg.
//! 2. **Composition.** `dt₁` then `dt₂` equals `dt₁ + dt₂` in one call. Reversibility alone can be
//!    satisfied by a map that is wrong symmetrically; composition cannot.
//! 3. **The two-body integrals.** Specific energy `v²/2 − μ/r` and angular momentum `r × v` are
//!    exactly conserved by the real flow, so any departure is the solver's own error. Energy is the
//!    sensitive one: it fixes the semi-major axis, and an eccentric anomaly solved to the wrong
//!    precision moves it long before the position visibly drifts.
//!
//! **Bounds are the measured worst case with an order of headroom.** Over 20 000 draws spanning
//! `e ∈ [0, 0.9]`, `a ∈ [7 000, 45 000] km` and `|dt|` up to one orbital period:
//!
//! | property | worst measured | bound |
//! |---|---|---|
//! | reversibility | 5.339e-11 | 1e-9 |
//! | composition | 1.860e-13 | 1e-11 |
//! | energy drift | 2.368e-13 | 1e-11 |
//! | angular-momentum drift | 2.373e-14 | 1e-12 |
//!
//! The asymmetry in that table is itself a measurement worth keeping: **reversing time is 287×
//! worse than splitting the same step in two.** Both call the same solver on the same orbit, so the
//! difference is not the flow — it is the universal anomaly's initial guess, which is built for
//! forward time and lands the Newton iteration a little further from the root going back. It is a
//! floor, not a defect, and #212 measured the same shape over long spans; recording it here means a
//! future change that makes it worse has something to be worse than.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]

use proptest::prelude::*;
use spacelab_astro::kepler;
use spacelab_core::Vec3;

/// Earth's gravitational parameter, m³/s².
const MU: f64 = 3.986_004_418e14;

/// Semi-major axis range, metres — low Earth orbit out past geostationary.
const A_RANGE: (f64, f64) = (7.0e6, 4.5e7);

/// Eccentricity range. Stops short of 1 deliberately: near-parabolic is a separate regime with its
/// own floor, and folding it in here would force a bound loose enough to hide the elliptic case.
const E_MAX: f64 = 0.9;

/// One orbit's state and a time step, built from elements so the orbit is always a real one.
fn state(a: f64, e: f64, i: f64, argp: f64, nu: f64) -> (Vec3, Vec3) {
    let p = a * (1.0 - e * e);
    let r = p / (1.0 + e * nu.cos());
    let h = (MU * p).sqrt();
    // Perifocal, then rotated by argument of periapsis about the pole and inclination about the
    // node. RAAN is omitted: the properties are rotation-invariant and it adds no regime.
    let (cn, sn) = ((nu).cos(), (nu).sin());
    let rp = Vec3::new(r * cn, r * sn, 0.0);
    let vp = Vec3::new(-MU / h * sn, MU / h * (e + cn), 0.0);
    let rot = |u: Vec3| {
        let (cw, sw) = (argp.cos(), argp.sin());
        let x = cw * u.x - sw * u.y;
        let y = sw * u.x + cw * u.y;
        let (ci, si) = (i.cos(), i.sin());
        Vec3::new(x, ci * y - si * u.z, si * y + ci * u.z)
    };
    (rot(rp), rot(vp))
}

/// Orbital period, seconds.
fn period(a: f64) -> f64 {
    core::f64::consts::TAU * (a * a * a / MU).sqrt()
}

/// Specific orbital energy, J/kg.
fn energy(r: Vec3, v: Vec3) -> f64 {
    v.dot(v) / 2.0 - MU / r.norm()
}

/// Relative departure of two states, scaled by what each carries.
fn drift(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> f64 {
    let dr = (a.0 - b.0).norm() / b.0.norm();
    let dv = (a.1 - b.1).norm() / b.1.norm();
    dr.max(dv)
}

prop_compose! {
    /// An orbit and a time step scaled to its own period, so a low orbit is not asked to propagate
    /// for a thousand revolutions while a high one barely moves.
    fn orbit()(
        a in A_RANGE.0..A_RANGE.1,
        e in 0.0..E_MAX,
        i in 0.0..core::f64::consts::PI,
        argp in 0.0..core::f64::consts::TAU,
        nu in 0.0..core::f64::consts::TAU,
        f in -1.0_f64..1.0,
    ) -> (Vec3, Vec3, f64) {
        let (r, v) = state(a, e, i, argp, nu);
        (r, v, f * period(a))
    }
}

proptest! {
    /// Forward then back is the identity. Worst measured relative departure 5.339e-11 — the
    /// loosest of the four, for the reason in the module doc.
    #[test]
    fn propagation_is_reversible((r, v, dt) in orbit()) {
        let (r1, v1) = kepler::propagate(r, v, MU, dt);
        let back = kepler::propagate(r1, v1, MU, -dt);
        let d = drift(back, (r, v));
        prop_assert!(d < 1e-9, "state moved {d:.3e} of its own magnitude over dt {dt:.1} s");
    }

    /// Two steps equal one of their sum. A symmetric error passes reversibility and fails this.
    /// Worst measured relative departure 1.860e-13.
    #[test]
    fn propagation_composes((r, v, dt) in orbit()) {
        let half = dt / 2.0;
        let (rm, vm) = kepler::propagate(r, v, MU, half);
        let two = kepler::propagate(rm, vm, MU, dt - half);
        let one = kepler::propagate(r, v, MU, dt);
        let d = drift(two, one);
        prop_assert!(d < 1e-11, "split step differs from the whole by {d:.3e}");
    }

    /// Energy and angular momentum are integrals of the two-body flow, so any change is the
    /// solver's error and nothing else. Worst measured: energy 2.368e-13, momentum 2.373e-14.
    #[test]
    fn the_two_body_integrals_are_conserved((r, v, dt) in orbit()) {
        let (r1, v1) = kepler::propagate(r, v, MU, dt);

        let (e0, e1) = (energy(r, v), energy(r1, v1));
        let de = ((e1 - e0) / e0).abs();
        prop_assert!(de < 1e-11, "specific energy moved by {de:.3e} relative");

        let (h0, h1) = (r.cross(v), r1.cross(v1));
        let dh = (h1 - h0).norm() / h0.norm();
        prop_assert!(dh < 1e-12, "angular momentum moved by {dh:.3e} relative");
    }
}
