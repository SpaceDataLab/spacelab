//! Which epoch a measurement's geometry belongs to.
//!
//! Everything in [`measurement`](crate::measurement) takes one state and one station and forms the
//! observable from the straight line between them. That is the geometry at a single instant, and it
//! is not what an instrument reads: light left the satellite before it arrived, so the direction and
//! the distance an observer records belong to where the satellite *was*, one transit time ago.
//!
//! # What the instantaneous form costs
//!
//! For a range the error is `(v_sat · ρ̂) τ` — the satellite's *own* radial speed, because light
//! time moves the satellite and leaves the observer where the observation happened. It vanishes at
//! closest approach and peaks at the edges of a pass. Reaching for the relative range rate instead,
//! which is what [`measurement::range_rate`](crate::measurement::range_rate) computes, adds the
//! station's own motion and is a different quantity.
//!
//! For an angle the range cancels: the satellite moves `v τ` while the station is `c τ` away, so
//! the lag is `v_⊥ / c`, bounded by the orbital speed and independent of how high the pass is.
//!
//! `spacelab/tests/light_time_in_measurements.rs` measures both on the ISS day the access oracle
//! validates against Orekit: **51.8 m** at the horizon against 0.68 m at the day's closest
//! approach, and **5.30″** of angular lag that barely moves across the sky.
//!
//! # Aberration is not here, and that is a decision
//!
//! The other half of the classical "planetary aberration" is the observer's own motion, and it is
//! deliberately absent from this crate. Two reasons, in order of weight:
//!
//! 1. **It is a property of the observer, not of the orbit.** An observed direction can be
//!    de-aberrated once, on the way in, and then the model never has to reproduce it — which also
//!    means it never has to be differentiated. `spacelab_astrometry::Aberration::remove` is that
//!    map, already written against SOFA's `iauAb`, and a second copy here is how two
//!    implementations of one formula drift apart.
//! 2. **In a geocentric frame it is small, and not for the reason one might reach for.** Aberration
//!    is frame-dependent: what enters is the observer's velocity *with respect to the frame the
//!    directions are expressed in*. Station and satellite share the Earth's barycentric velocity,
//!    so the 20.5″ annual term cancels in GCRF and what is left is the diurnal term — Earth
//!    rotation, at most 465 m/s at the equator, **0.32″**. The analogy with #306's Sun, which is a
//!    distant body and does carry the annual term, does not hold for a satellite seen from the
//!    ground.
//!
//! Light time, by contrast, cannot be removed from the observation: undoing it needs the orbit,
//! which is what the fit is trying to find. It has to be in the model, and so it is here.

/// Speed of light in vacuum, m/s — exact, by the SI definition of the metre.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// The solved geometry of one signal leg: when it left, when it arrived, and how long it took.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transit {
    /// Flight time, seconds.
    pub delay: f64,
    /// Epoch the signal left the emitter, in the caller's own timescale.
    pub emit: f64,
    /// Epoch the signal reached the receiver, in the caller's own timescale.
    pub receive: f64,
}

impl Transit {
    /// The distance the signal covered, metres: `c τ`.
    ///
    /// This is what a one-way ranging system measures, and it is *not*
    /// [`measurement::range`](crate::measurement::range) evaluated at either endpoint's epoch — the
    /// two endpoints are at different instants and the straight line between simultaneous positions
    /// is a third quantity.
    #[must_use]
    pub fn path(&self) -> f64 {
        SPEED_OF_LIGHT * self.delay
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

/// Solve `τ = |r_emit(t_r − τ) − r_recv| / c` for a signal observed at `receive`.
///
/// The receiver is where the observation happens, so its position is a constant of the problem;
/// the emitter is the moving one, and `emitter` returns its position at an epoch or `None` where
/// the ephemeris does not reach — a sighting near the end of an orbit file has to fail rather than
/// extrapolate.
///
/// Iteration stops when the estimate moves by less than `tolerance` seconds and gives up after
/// `max_iterations` regardless. The map is a contraction with ratio `v/c ≈ 2.5e-5`, so two
/// iterations reach femtoseconds from a cold start; the cap exists so that a caller who hands in a
/// discontinuous ephemeris gets `None` rather than a hang, which is the same non-termination
/// discipline the integrators carry.
///
/// # The other legs
///
/// This is the *down* leg — something arrives at the observer, which is every passive observation
/// and the receiving half of any ranging system. The two-way case is
/// `spacelab_slr::two_way_light_time`, and it is a separate implementation on purpose: its up leg
/// holds the *station* fixed and runs the satellite forward, its down leg starts from the up leg's
/// answer rather than from zero, and both of those iterates are pinned to the ulp by tests written
/// against ILRS normal points. Sharing an iteration between the two would change which iterate is
/// returned, which is a verified number, to save nine lines.
pub fn down_leg<E>(
    receive: f64,
    receiver: [f64; 3],
    emitter: E,
    tolerance: f64,
    max_iterations: usize,
) -> Option<Transit>
where
    E: Fn(f64) -> Option<[f64; 3]>,
{
    let mut delay = 0.0;
    let mut source = emitter(receive)?;
    for _ in 0..max_iterations {
        let next = distance(source, receiver) / SPEED_OF_LIGHT;
        source = emitter(receive - next)?;
        let step = libm::fabs(next - delay);
        delay = next;
        if step < tolerance {
            return Some(Transit {
                delay,
                emit: receive - delay,
                receive,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    /// A geocentric distance and a LEO one, so the numbers below are of a real size.
    const EARTH_RADIUS: f64 = 6_378_137.0;
    const LEO: f64 = EARTH_RADIUS + 500_000.0;

    /// The separation is the Euclidean norm, in all three components and both signs.
    ///
    /// Every other test here puts both endpoints on the x-axis, where `y` and `z` are identically
    /// zero — and a component that is zero cannot tell a difference from a sum, nor a square from a
    /// doubling. The offsets are chosen so that no coordinate is zero, no pair is symmetric about
    /// the origin, and the answer is an exact small integer.
    #[test]
    fn the_separation_is_the_euclidean_norm_in_all_three_components() {
        assert_within_ulps(
            distance([10.0, 20.0, 30.0], [11.0, 22.0, 32.0]),
            3.0,
            0,
            "the (1, 2, 2) quadruple",
        );
        assert_within_ulps(
            distance([-4.0, 13.0, -8.0], [-8.0, 1.0, -5.0]),
            13.0,
            0,
            "the (4, 12, 3) quadruple through the origin",
        );
        let (p, q) = ([1.0, -5.0, 7.0], [-2.0, 3.0, -4.0]);
        assert_within_ulps(distance(p, q), distance(q, p), 0, "the swapped pair");
    }

    /// A frozen emitter gives the geometric distance and nothing else: the delay is `d/c`, the
    /// emission epoch is that much before reception, and the reception epoch is untouched.
    #[test]
    fn a_frozen_emitter_returns_the_geometric_distance() {
        let t = 1234.5;
        let receiver = [EARTH_RADIUS, 0.0, 0.0];
        let transit = down_leg(t, receiver, |_| Some([LEO, 0.0, 0.0]), 1e-15, 20)
            .expect("a frozen geometry converges");
        assert_within_ulps(transit.delay, 500_000.0 / SPEED_OF_LIGHT, 1, "the delay");
        assert_within_ulps(transit.path(), 500_000.0, 2, "the path length");
        assert_within_ulps(transit.receive, t, 0, "the reception epoch");
        assert_within_ulps(transit.emit, t - transit.delay, 0, "the emission epoch");
    }

    /// The emitter is taken at `receive − τ`, not at `receive`, and not at `receive + τ`.
    ///
    /// The whole point of the module is that sign. A source receding along the line of sight is
    /// *closer* at the emission epoch than at reception, so the solved path must come out shorter
    /// than the instantaneous separation — and by `v τ` to first order, which is checked rather
    /// than merely signed, because a sign test alone passes for a factor of two.
    #[test]
    fn the_emitter_is_taken_before_reception_and_not_after() {
        let speed = 7_600.0;
        let receiver = [EARTH_RADIUS, 0.0, 0.0];
        let receding = |t: f64| Some([LEO + speed * t, 0.0, 0.0]);

        let transit = down_leg(0.0, receiver, receding, 1e-18, 20).expect("converges");
        let instantaneous = LEO - EARTH_RADIUS;
        assert!(
            transit.path() < instantaneous,
            "a receding source gave a path {} m longer than the instantaneous separation",
            transit.path() - instantaneous
        );
        let expected = speed * transit.delay;
        let observed = instantaneous - transit.path();
        assert!(
            libm::fabs(observed - expected) < 1e-6,
            "{observed} m of shortening against the {expected} m the recession implies"
        );
    }

    /// The answer satisfies the equation it claims to solve, to the tolerance asked for.
    ///
    /// The tests above check the delay against a closed form available only because the geometry is
    /// on one axis. This one is the general statement, on a source moving across the line of sight
    /// as well as along it: substitute the solution back and the residual must vanish.
    #[test]
    fn the_solution_satisfies_the_light_time_equation() {
        let receiver = [EARTH_RADIUS * 0.6, EARTH_RADIUS * 0.5, EARTH_RADIUS * 0.6];
        let orbiting = |t: f64| Some([LEO - 1_000.0 * t, 7_500.0 * t, 2_000.0 * t]);
        let transit = down_leg(0.0, receiver, orbiting, 1e-18, 30).expect("converges");
        let residual = transit.delay
            - distance(orbiting(transit.emit).expect("in range"), receiver) / SPEED_OF_LIGHT;
        assert!(
            libm::fabs(residual) < 1e-18,
            "{residual} s of residual in the light-time equation"
        );
    }

    /// Convergence is `step < tolerance`, and the boundary is where `<` and `<=` part company.
    ///
    /// Every other test here hands in a tolerance orders below anything the iteration produces, so
    /// the comparison is decided long before its boundary and either operator serves. On the
    /// boundary they differ by a whole iteration, and with the budget exhausted, by the difference
    /// between an answer and `None`. A frozen geometry moves by its whole flight time on the first
    /// pass, so setting `tolerance` to that step and allowing one iteration sits exactly there.
    #[test]
    fn a_step_that_only_equals_the_tolerance_is_not_convergence() {
        let receiver = [EARTH_RADIUS, 0.0, 0.0];
        let source = |_: f64| Some([LEO, 0.0, 0.0]);
        let first_step = (LEO - EARTH_RADIUS) / SPEED_OF_LIGHT;
        assert!(
            down_leg(0.0, receiver, source, first_step, 1).is_none(),
            "a step that only equals the tolerance was taken as convergence"
        );
        down_leg(0.0, receiver, source, first_step, 2).expect("converges on the second pass");
    }

    /// An ephemeris that does not cover the emission epoch produces `None`, not a range built from
    /// an extrapolation the caller never authorised. The gap is *behind* the reception epoch, which
    /// is the direction this solver steps in and the one a forward-only guard would miss.
    #[test]
    fn a_gap_behind_the_reception_epoch_is_reported_rather_than_extrapolated() {
        let receiver = [EARTH_RADIUS, 0.0, 0.0];
        let starts_at_zero = |t: f64| (t >= 0.0).then_some([LEO, 0.0, 0.0]);
        assert!(down_leg(0.0, receiver, starts_at_zero, 1e-15, 20).is_none());
        // One transit later the same ephemeris covers the emission epoch and the solve succeeds.
        let after = 2.0 * (LEO - EARTH_RADIUS) / SPEED_OF_LIGHT;
        down_leg(after, receiver, starts_at_zero, 1e-15, 20).expect("the gap is behind us");
    }

    /// An emitter whose position jumps every call cannot converge, and the cap has to turn that
    /// into `None` rather than an unbounded loop, with the evaluations bounded too.
    #[test]
    fn a_non_convergent_ephemeris_terminates() {
        let receiver = [EARTH_RADIUS, 0.0, 0.0];
        let calls = core::cell::Cell::new(0_u32);
        let jumpy = |_: f64| {
            calls.set(calls.get() + 1);
            let jump = if calls.get().is_multiple_of(2) {
                0.0
            } else {
                1e7
            };
            Some([LEO + jump, 0.0, 0.0])
        };
        assert!(down_leg(0.0, receiver, jumpy, 1e-18, 8).is_none());
        assert!(
            calls.get() <= 9,
            "{} evaluations is past the cap",
            calls.get()
        );
    }

    /// Zero iterations is a budget, not a special case: it must return `None` rather than the
    /// zero-delay starting guess dressed up as a solution.
    #[test]
    fn an_exhausted_budget_never_returns_the_starting_guess() {
        let receiver = [EARTH_RADIUS, 0.0, 0.0];
        assert!(down_leg(0.0, receiver, |_| Some([LEO, 0.0, 0.0]), 1e-15, 0).is_none());
    }
}
