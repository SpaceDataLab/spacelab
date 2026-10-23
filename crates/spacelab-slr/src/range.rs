//! The two-way range observation model — what a station should have measured.
//!
//! A laser ranging measurement is a *flight time*, and the two ends of the flight are at three
//! different instants: the pulse leaves the telescope at `t`, reflects at `t + τ_up`, and returns
//! at `t + τ_up + τ_down`. Over the 0.08 s round trip to LAGEOS the ground station moves about
//! 37 m through inertial space and the satellite about 300 km, so treating the geometry as
//! instantaneous is not a small-angle approximation — it is a kilometre-scale error. Both legs are
//! solved separately here.
//!
//! Everything in this module works in a **geocentric inertial** frame. That is not a preference: an
//! Earth-fixed frame is rotating, and the light-time iteration in it would have to carry the
//! rotation of the frame during the flight as a fictitious term. Rotating the station and the
//! satellite into GCRF first, at their own epochs, is the same computation done once and openly.

/// Speed of light in vacuum, m/s — the defining constant of the SI metre, so exact.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Solved geometry of one two-way measurement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoWayRange {
    /// Transmit-to-reflection flight time, seconds.
    pub up_leg: f64,
    /// Reflection-to-receive flight time, seconds.
    pub down_leg: f64,
    /// The one-way range the round trip corresponds to, metres: `c(τ_up + τ_down)/2`. This is the
    /// quantity a normal point is compared against, because that is how the publisher formed it.
    pub range: f64,
}

impl TwoWayRange {
    /// Total round-trip flight time, seconds.
    #[must_use]
    pub const fn time_of_flight(&self) -> f64 {
        self.up_leg + self.down_leg
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

/// Solve the two legs of a two-way range by light-time iteration.
///
/// `transmit` is the epoch the pulse leaves the station, in whatever seconds-since-something the
/// two callbacks agree on. `station` returns the station's inertial position at an epoch and
/// `satellite` returns the satellite's, or `None` where the ephemeris does not reach — a pass that
/// straddles the end of an orbit file has to fail rather than extrapolate.
///
/// Iteration stops when a leg moves by less than `tolerance` seconds, and gives up after
/// `max_iterations` regardless. Both legs converge geometrically with ratio `v/c ≈ 2e-5`, so three
/// iterations reach femtoseconds and the cap exists only so that a caller who hands in a
/// discontinuous ephemeris gets `None` rather than a hang.
pub fn two_way_light_time<S, T>(
    transmit: f64,
    station: S,
    satellite: T,
    tolerance: f64,
    max_iterations: usize,
) -> Option<TwoWayRange>
where
    S: Fn(f64) -> [f64; 3],
    T: Fn(f64) -> Option<[f64; 3]>,
{
    let transmitter = station(transmit);

    let mut up = 0.0;
    let mut bounce = satellite(transmit)?;
    let mut converged = false;
    for _ in 0..max_iterations {
        let next = distance(bounce, transmitter) / SPEED_OF_LIGHT;
        bounce = satellite(transmit + next)?;
        let step = libm::fabs(next - up);
        up = next;
        if step < tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }

    let mut down = up;
    converged = false;
    for _ in 0..max_iterations {
        let next = distance(bounce, station(transmit + up + down)) / SPEED_OF_LIGHT;
        let step = libm::fabs(next - down);
        down = next;
        if step < tolerance {
            converged = true;
            break;
        }
    }
    if !converged {
        return None;
    }

    Some(TwoWayRange {
        up_leg: up,
        down_leg: down,
        range: 0.5 * SPEED_OF_LIGHT * (up + down),
    })
}

/// Shapiro gravitational delay of one leg, metres — IERS Conventions (2010) §11.
///
/// Light slows in the Earth's potential well, so the flight takes longer than the coordinate
/// distance divided by `c`. `r_emitter` and `r_receiver` are the geocentric distances of the two
/// endpoints and `path` the straight-line distance between them, all in metres; `gm` is the
/// Earth's gravitational parameter, m³/s².
///
/// It is about 7 mm one-way to LAGEOS and 14 mm on the round trip — small, but ten times the
/// precision of a normal point, so it is not optional at this level.
#[must_use]
pub fn shapiro_delay(r_emitter: f64, r_receiver: f64, path: f64, gm: f64) -> f64 {
    let sum = r_emitter + r_receiver;
    2.0 * gm / (SPEED_OF_LIGHT * SPEED_OF_LIGHT) * libm::log((sum + path) / (sum - path))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    const EARTH_GM: f64 = 3.986_004_418e14;

    /// The separation is the Euclidean norm, in all three components and both signs.
    ///
    /// Every other test in this file puts the station and the satellite on the x-axis, where `y`
    /// and `z` are identically zero — and a component that is zero cannot tell a difference from a
    /// sum, nor a square from a doubling. The offsets below are chosen so that no coordinate is
    /// zero, no pair is symmetric about the origin, and the answer is an exact small integer.
    #[test]
    fn the_separation_is_the_euclidean_norm_in_all_three_components() {
        // (1, 2, 2) about a point far from the origin: reading `a + b` for `a - b` gives 61.4, not 3.
        assert_within_ulps(
            distance([10.0, 20.0, 30.0], [11.0, 22.0, 32.0]),
            3.0,
            0,
            "the (1, 2, 2) quadruple",
        );
        // Negative components on both sides, so a dropped sign shows up rather than cancelling.
        assert_within_ulps(
            distance([-4.0, 13.0, -8.0], [-8.0, 1.0, -5.0]),
            13.0,
            0,
            "the (4, 12, 3) quadruple through the origin",
        );
        // Symmetric in its arguments: the difference enters only squared.
        let (p, q) = ([1.0, -5.0, 7.0], [-2.0, 3.0, -4.0]);
        assert_within_ulps(distance(p, q), distance(q, p), 0, "the swapped pair");
    }

    /// Convergence is `step < tolerance`, and the boundary is where `<` and `<=` part company.
    ///
    /// Every other test here hands in a tolerance orders below anything the iteration produces, so
    /// the comparison is decided long before its boundary and either operator serves. On the
    /// boundary they differ by a whole iteration — and with the iteration budget exhausted, by the
    /// difference between an answer and `None`. A caller who sizes `tolerance` to the precision it
    /// actually needs sits exactly there.
    ///
    /// The up leg: a frozen geometry moves by its whole flight time on the first pass, so setting
    /// `tolerance` to that step and allowing one iteration puts the test on the boundary. `<`
    /// declines and runs out of budget; `<=` accepts.
    #[test]
    fn the_up_leg_does_not_converge_when_the_step_only_equals_the_tolerance() {
        let sta = [6_378_137.0, 0.0, 0.0];
        let sat = [6_378_137.0 + 12_270_000.0, 0.0, 0.0];
        let first_step = distance(sat, sta) / SPEED_OF_LIGHT;
        assert!(
            two_way_light_time(0.0, |_| sta, |_| Some(sat), first_step, 1).is_none(),
            "a step that only equals the tolerance was taken as convergence"
        );
        // One iteration more and it converges on the second pass, where the step is zero.
        two_way_light_time(0.0, |_| sta, |_| Some(sat), first_step, 2).expect("converges at two");
    }

    /// The same boundary on the down leg, which has its own comparison and its own budget.
    ///
    /// A frozen geometry cannot reach it — the down leg starts from the up leg's answer, so its
    /// first step is zero and the boundary would need a zero tolerance, which stops the up leg
    /// getting there at all. A receding station gives the down leg a first step of its own. Both
    /// operators converge here; what differs is *which* iterate is returned, and the two are a
    /// station-motion correction apart.
    #[test]
    fn the_down_leg_takes_the_iterate_after_the_one_that_only_equals_the_tolerance() {
        let speed = 400.0;
        let sat = [6_378_137.0 + 12_270_000.0, 0.0, 0.0];
        let receding = |t: f64| [6_378_137.0 - speed * t, 0.0, 0.0];

        // The iteration reproduced here exactly, so `tolerance` lands on the boundary to the bit.
        let up = distance(sat, receding(0.0)) / SPEED_OF_LIGHT;
        let first = distance(sat, receding(up + up)) / SPEED_OF_LIGHT;
        let second = distance(sat, receding(up + first)) / SPEED_OF_LIGHT;
        let tolerance = libm::fabs(first - up);

        let r = two_way_light_time(0.0, receding, |_| Some(sat), tolerance, 2).expect("converges");
        assert_within_ulps(r.up_leg, up, 0, "the up leg");
        assert_within_ulps(
            r.down_leg,
            second,
            0,
            "the down leg stopped one iterate early",
        );
    }

    /// A stationary station and a stationary satellite have to give both legs equal and the range
    /// equal to the geometric distance. Anything else means the iteration is not solving the
    /// equation it claims to.
    #[test]
    fn a_frozen_geometry_returns_the_geometric_distance() {
        let station = |_: f64| [6_378_137.0, 0.0, 0.0];
        let satellite = |_: f64| Some([6_378_137.0 + 12_270_000.0, 0.0, 0.0]);
        let r = two_way_light_time(0.0, station, satellite, 1e-15, 20).expect("frozen converges");
        assert_within_ulps(r.up_leg, r.down_leg, 0, "r.up_leg");
        assert_within_ulps(r.range, 12_270_000.0, 2, "r.range");
        assert_within_ulps(
            r.time_of_flight(),
            2.0 * 12_270_000.0 / SPEED_OF_LIGHT,
            2,
            "r.time_of_flight()",
        );
    }

    /// It is the *station's* motion that splits the two legs, and only that. A satellite moving
    /// along the line of sight does not: the up leg ends where the down leg begins, so both legs
    /// span the same distance whatever the satellite did to get there. A station receding at `v`,
    /// on the other hand, is `v·(τ_up + τ_down)` further away when the pulse returns.
    ///
    /// This is the property that breaks if the down leg is taken as a copy of the up leg — the
    /// tempting shortcut, and the one that costs 37 m of station motion on a LAGEOS pass.
    #[test]
    fn only_the_stations_motion_splits_the_two_legs() {
        let speed = 400.0;
        let still = |_: f64| [6_378_137.0, 0.0, 0.0];
        let receding = |t: f64| [6_378_137.0 - speed * t, 0.0, 0.0];
        let orbiting = |t: f64| Some([6_378_137.0 + 12_270_000.0 + 3000.0 * t, 0.0, 0.0]);
        let fixed = |_: f64| Some([6_378_137.0 + 12_270_000.0, 0.0, 0.0]);

        let a = two_way_light_time(0.0, still, orbiting, 1e-15, 20).expect("converges");
        assert_within_ulps(a.up_leg, a.down_leg, 0, "legs under a still station");

        let b = two_way_light_time(0.0, receding, fixed, 1e-15, 20).expect("converges");
        assert!(b.down_leg > b.up_leg, "{b:?}");
        let expected = speed * b.time_of_flight() / SPEED_OF_LIGHT;
        let observed = b.down_leg - b.up_leg;
        assert!(
            (observed - expected).abs() < 1e-15,
            "{observed} s of split against the {expected} s the station's recession implies"
        );
    }

    /// An ephemeris that does not cover the reflection epoch must produce `None`, not a range built
    /// from an extrapolation the caller never authorised.
    #[test]
    fn a_gap_in_the_ephemeris_is_reported_rather_than_extrapolated() {
        let station = |_: f64| [6_378_137.0, 0.0, 0.0];
        let satellite = |t: f64| (t < 0.01).then_some([18_648_137.0, 0.0, 0.0]);
        assert!(two_way_light_time(0.0, station, satellite, 1e-15, 20).is_none());
    }

    /// A satellite whose position jumps every call cannot converge, and the iteration cap has to
    /// turn that into `None` rather than an unbounded loop. This is the same non-termination
    /// discipline the integrators carry.
    #[test]
    fn a_non_convergent_ephemeris_terminates() {
        let station = |_: f64| [6_378_137.0, 0.0, 0.0];
        let calls = core::cell::Cell::new(0_u32);
        let satellite = |_: f64| {
            calls.set(calls.get() + 1);
            let jump = if calls.get().is_multiple_of(2) {
                0.0
            } else {
                1e7
            };
            Some([18_648_137.0 + jump, 0.0, 0.0])
        };
        assert!(two_way_light_time(0.0, station, satellite, 1e-18, 8).is_none());
        assert!(
            calls.get() <= 9,
            "{} evaluations is past the cap",
            calls.get()
        );
    }

    /// The Shapiro delay for a LAGEOS-geometry leg is a few millimetres, positive, and grows with
    /// path length. The magnitude is what §11 states for a low-Earth-to-ground link.
    #[test]
    fn the_shapiro_delay_is_millimetres_and_grows_with_the_path() {
        let ground = 6_378_137.0;
        let lageos = 12_270_000.0 + ground;
        let near = shapiro_delay(ground, lageos, lageos - ground, EARTH_GM);
        let far = shapiro_delay(ground, lageos, 14_000_000.0, EARTH_GM);
        assert!(
            (0.004..0.015).contains(&near),
            "{near} m is out of family for a LAGEOS leg"
        );
        assert!(far > near, "{far} is not above {near}");
        // Degenerate coincident endpoints: no path, no delay.
        assert_within_ulps(
            shapiro_delay(ground, ground, 0.0, EARTH_GM),
            0.0,
            0,
            "shapiro_delay(ground",
        );
    }
}
