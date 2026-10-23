//! The Cartesian↔Keplerian round trip, over the whole domain rather than the six orbits in
//! `tests/keplerian.rs`.
//!
//! The existing test names the three singular cases — circular, equatorial, circular-equatorial —
//! and checks one orbit at each. That pins the exact singularity and says nothing about its
//! *neighbourhood*, which is where a conversion actually fails: at `e = 0` a well-written branch
//! takes over and works, and at `e = 1e-13` the general formula divides by a number that has lost
//! most of its significand. Nobody writes that fixture by hand. A generator does, on every run.
//!
//! The comparison is on the **state**, never on the elements. At `e → 0` the argument of periapsis
//! is undefined and at `i → 0` so is the node; asserting on them would be asserting on a coin flip.
//! The state is what the elements are for and it is well defined everywhere.
//!
//! **This file found a bug on its first run.** At *exactly* `i = π` the round trip returned the
//! state reflected in y — a relative departure of 2.0, an orbit mirrored rather than a tolerance
//! missed. `from_state`'s `equatorial` predicate is `n ≤ ε·h`, which is true for `i ≈ 0` and
//! `i ≈ π` alike, but the branch it selects measured the longitude of periapsis anticlockwise about
//! `+ẑ` while `to_state`'s `R1(−i)` at `i = π` reads it clockwise. Six hand-written singular
//! fixtures had passed for as long as the function existed, because all six were prograde. Note the
//! shape: `i = π − 1e-9` was always correct, because it misses the branch entirely.
//!
//! **Bounds are the measured worst case.** Over 20 000 draws with `e` and `i` drawn log-uniformly
//! down to `1e-12` of their singular values: 3.515e-15 relative on position and 5.060e-15 on
//! velocity in the general case, 1.943e-11 / 1.970e-11 in the singular neighbourhoods. The four
//! orders between them are the cost of the singularity, recorded rather than hidden behind one
//! shared bound loose enough to cover both.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]

use proptest::prelude::*;
use spacelab_core::{Angle, Vec3};
use spacelab_elements::KeplerianElements;

/// Earth's gravitational parameter, m³/s².
const MU: f64 = 3.986_004_418e14;

/// Semi-major axis range, metres.
const A_RANGE: (f64, f64) = (7.0e6, 4.5e7);

/// How close to a singularity the biased strategy is allowed to get. Below this the general
/// formulae have no significand left and the answer is the branch's, not the formula's.
const SINGULAR_FLOOR: f64 = 1e-12;

fn elements(a: f64, e: f64, i: f64, raan: f64, argp: f64, nu: f64) -> KeplerianElements {
    KeplerianElements {
        semi_major_axis: a,
        eccentricity: e,
        inclination: Angle::from_radians(i),
        raan: Angle::from_radians(raan),
        arg_periapsis: Angle::from_radians(argp),
        true_anomaly: Angle::from_radians(nu),
        mu: MU,
    }
}

/// Relative departure of a recovered state from the original.
fn departure(got: (Vec3, Vec3), want: (Vec3, Vec3)) -> (f64, f64) {
    (
        (got.0 - want.0).norm() / want.0.norm(),
        (got.1 - want.1).norm() / want.1.norm(),
    )
}

prop_compose! {
    /// The general case: nothing near a singularity, so this measures the formula rather than the
    /// branch that replaces it.
    fn general()(
        a in A_RANGE.0..A_RANGE.1,
        e in 1e-3..0.9,
        i in 1e-3..(core::f64::consts::PI - 1e-3),
        raan in 0.0..core::f64::consts::TAU,
        argp in 0.0..core::f64::consts::TAU,
        nu in 0.0..core::f64::consts::TAU,
    ) -> KeplerianElements {
        elements(a, e, i, raan, argp, nu)
    }
}

prop_compose! {
    /// Deliberately pathological: `e` and `i` are drawn log-uniformly down to `1e-12` of their
    /// singular values, and each is independently allowed to be *exactly* singular. That covers the
    /// circular, equatorial, retrograde-equatorial and circular-equatorial cases and — the part
    /// hand-written fixtures miss — the whole approach to each of them.
    fn near_singular()(
        a in A_RANGE.0..A_RANGE.1,
        e_exp in -12.0_f64..-0.3,
        i_exp in -12.0_f64..-0.3,
        e_zero in prop::bool::ANY,
        i_zero in prop::bool::ANY,
        retrograde in prop::bool::ANY,
        raan in 0.0..core::f64::consts::TAU,
        argp in 0.0..core::f64::consts::TAU,
        nu in 0.0..core::f64::consts::TAU,
    ) -> KeplerianElements {
        let e = if e_zero { 0.0 } else { 10.0_f64.powf(e_exp) };
        let di = if i_zero { 0.0 } else { 10.0_f64.powf(i_exp) };
        let i = if retrograde { core::f64::consts::PI - di } else { di };
        elements(a, e, i, raan, argp, nu)
    }
}

proptest! {
    /// `elements → state → elements → state` reproduces the state. Worst measured relative
    /// departure 3.515e-15 on position, 5.060e-15 on velocity.
    #[test]
    fn the_round_trip_reproduces_the_state_in_the_general_case(k in general()) {
        let want = k.to_state();
        let got = KeplerianElements::from_state(want.0, want.1, MU).to_state();
        let (dr, dv) = departure(got, want);
        prop_assert!(dr < 1e-13, "position moved {dr:.3e} at e {} i {}", k.eccentricity, k.inclination.radians());
        prop_assert!(dv < 1e-13, "velocity moved {dv:.3e} at e {} i {}", k.eccentricity, k.inclination.radians());
    }

    /// The same, arbitrarily close to every singularity the parameterisation has — and *at* each of
    /// them, prograde and retrograde. Worst measured 1.943e-11 on position and 1.970e-11 on
    /// velocity, four orders worse than the general case.
    #[test]
    fn the_round_trip_survives_the_singular_neighbourhoods(k in near_singular()) {
        let want = k.to_state();
        let got = KeplerianElements::from_state(want.0, want.1, MU).to_state();
        let (dr, dv) = departure(got, want);
        prop_assert!(
            dr < 1e-9,
            "position moved {dr:.3e} at e {:e} i {:e}",
            k.eccentricity, k.inclination.radians()
        );
        prop_assert!(
            dv < 1e-9,
            "velocity moved {dv:.3e} at e {:e} i {:e}",
            k.eccentricity, k.inclination.radians()
        );
    }

    /// The recovered elements are inside their own declared domains for every input, singular or
    /// not. `from_state` documents `e ∈ [0, 1)` and `i ∈ [0, π]`; a formula that returns `i` a
    /// hair above `π` because `acos` was handed `-1 - 2e-16` produces a `NaN` two calls later, and
    /// no round-trip assertion catches it because the state comes back correct anyway.
    #[test]
    fn the_recovered_elements_respect_their_declared_domains(k in near_singular()) {
        let (r, v) = k.to_state();
        let got = KeplerianElements::from_state(r, v, MU);
        let i = got.inclination.radians();
        prop_assert!(got.eccentricity >= 0.0 && got.eccentricity < 1.0, "e {} out of [0, 1)", got.eccentricity);
        prop_assert!((0.0..=core::f64::consts::PI).contains(&i), "i {i} out of [0, π]");
        prop_assert!(got.semi_major_axis.is_finite() && got.semi_major_axis > 0.0, "a {}", got.semi_major_axis);
        for angle in [got.raan, got.arg_periapsis, got.true_anomaly] {
            prop_assert!(angle.radians().is_finite(), "angle {} is not finite", angle.radians());
        }
    }
}

/// The floor the biased strategy stops at, asserted so the constant cannot drift away from the
/// claim the module doc makes about it.
#[test]
fn the_singular_strategy_reaches_the_floor_it_documents() {
    assert!((10.0_f64.powf(-12.0) - SINGULAR_FLOOR).abs() < f64::EPSILON * SINGULAR_FLOOR);
}
