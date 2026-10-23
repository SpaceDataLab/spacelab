//! Which epoch an observable's geometry is evaluated at, and what that does to its partials.
//!
//! [`measurement`](crate::measurement) forms every observable from one state, one station and one
//! instant, and [`modifier`](crate::modifier) composes corrections onto it. Neither chooses *when*.
//! An instrument does not read the geometry at the reception epoch: the light it records left the
//! satellite a transit time ago, so the distance and the direction belong to where the satellite
//! *was*. [`light_time::down_leg`](crate::light_time::down_leg) has solved for that epoch since it
//! was written, and until this module nothing in the model path called it — the solver existed and
//! the convention it fixes was still inherited rather than stated.
//!
//! Measured on the ISS day `spacelab/tests/light_time_in_measurements.rs` runs: **51.8 m** of range
//! at the horizon and **5.30″** of line of sight at every elevation.
//!
//! # The range is not a second formula
//!
//! `c τ = |r_sat(t_r − τ) − r_station|` is the equation [`down_leg`]
//! solves, so at convergence `Transit::path()` and the [`Range`](crate::modifier::Range) modifier
//! evaluated at the emission state are *the same number*, not two models of one thing. That is the
//! whole content of "light time is a choice of epoch": the formulae in `measurement` are already
//! right, and what was missing was the state to hand them.
//!
//! # The partials are not a choice of epoch
//!
//! This is where the analogy stops, and it is the reason this module exists rather than a line in
//! someone's calling code. `τ` depends on the state, so an orbit determination's
//! `∂observable/∂x₀` is not the automatic-differentiation row evaluated at the emission state.
//! `derivations/light_time_partials.mac` derives what it is instead, and checks each step against
//! an independent route:
//!
//! ```text
//! ∂τ/∂x₀ = ρ̂·Φ_r / (c + ρ̇)                       ρ̇ = ρ̂·v_e, the satellite's *own* radial speed
//! ∂f/∂x₀ = [w − s·(ρ̂, 0)] Φ                      s = (w·ẋ_e)/(c + ρ̇),  ẋ_e = (v_e, a_e)
//! ```
//!
//! The second line is the load-bearing one: the correction is a **rank-one update to the row**, and
//! the state transition matrix the caller chains is untouched. So the interface does not change —
//! [`Partials::state`](crate::modifier::Partials::state) is still a row to multiply by `Φ` — and a
//! caller who already had a light-time-free pipeline changes which row they get, not what they do
//! with it.
//!
//! **The state transition matrix has to reach the emission epoch.** `Φ` above is `Φ(t_e, t₀)`, not
//! `Φ(t_r, t₀)`: the row belongs to the state at emission, so the matrix chained to it must reach
//! that epoch. This is easy to get wrong precisely because the interface did not change, and it is
//! worth **`τ/t_r`** in the partial — 2.3e-4 over a 100 s arc in the unit test below, which is an
//! order of magnitude *more* than the light-time correction the row itself carries. A caller who
//! adopts this module and keeps propagating the STM to the reception epoch is worse off than
//! before, so the number is stated here rather than left to be discovered.
//!
//! For a pure range it collapses to a scalar, `1/(1 + ρ̇/c)`: **2.56e-5** relative at LEO's
//! 7.66 km/s, falling to 1.02e-5 at a geostationary transfer's radial rate and to exactly zero at
//! closest approach, where the radial speed is zero. That is a scaling of the sensitivity rather
//! than a bias in the value, so it moves a converged solution very little and moves the covariance
//! it reports by its own size.
//!
//! # What it costs
//!
//! Two or three ephemeris evaluations per observation instead of one: the map is a contraction with
//! ratio `v/c ≈ 2.5e-5`, so it converges in two from a cold start, and `reduce` then evaluates once
//! more at the epoch it found. Against a numerical propagation that would be expensive; against the
//! dense output a batch fit already carries it is three interpolations, which is nothing beside the
//! force evaluations that produced the trajectory. A caller for whom it *is* expensive has
//! [`Signal::Instantaneous`] and a measured number for what it costs them.
//!
//! # Why the acceleration is an argument
//!
//! `ẋ_e = (v_e, a_e)` needs the satellite's acceleration at the emission epoch, and this crate has
//! no force model — `spacelab-propagation` depends on it, not the other way round. The caller who
//! produced the emission state has the acceleration in hand from the step that produced it, so it
//! is an argument rather than a dependency edge. Passing zero is a *documented* choice with a
//! measured cost, not a silent one: the `a_e` term enters only through the velocity components of
//! `w`, so a range or an angle (whose `w` has zero velocity rows) is unaffected by it exactly, and
//! only a range-rate is not.
//!
//! # What is not here
//!
//! **The two-way solution.** `spacelab_slr::two_way_light_time` is it, its up leg holds the station
//! fixed and its down leg starts from the up leg's answer, and both iterates are pinned to the ULP
//! by tests written against ILRS normal points. A second implementation here would be a third copy
//! of one iteration, and the one that is already validated against real normal points would not be
//! the one an orbit determination called.
//!
//! **Aberration.** The observer's own motion belongs on the observation, not on the model, where it
//! never has to be differentiated. See [`light_time`](crate::light_time) and
//! `spacelab_astrometry::reduce::Aberration::diurnal`.

use crate::light_time::{SPEED_OF_LIGHT, Transit, down_leg};
use crate::measurement::Station;
use crate::modifier::{Observable, Partials, partials};

/// The epoch convention an observation's geometry is evaluated at.
///
/// A two-variant enum rather than a `bool` or an `Option`, and required rather than defaulted,
/// because the whole defect this module closes is a convention that was inherited instead of
/// stated. A caller who wants the instantaneous geometry now says so.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Signal {
    /// One state, one instant, a straight line between them — what `measurement` implements
    /// directly. Correct for a geometry that is *defined* instantaneously (a simulation's truth, a
    /// covariance study) and wrong by 51.8 m at the horizon for anything an instrument read.
    Instantaneous,
    /// The satellite is where it was when the light left it: `c τ = |r_sat(t_r − τ) − r_station|`.
    ///
    /// `tolerance` is in seconds and `max_iterations` bounds a caller who hands in a discontinuous
    /// ephemeris. The map is a contraction with ratio `v/c ≈ 2.5e-5`, so two iterations reach
    /// femtoseconds from a cold start; twenty is the budget the rest of the workspace uses.
    DownLeg {
        /// Stop when the flight time moves by less than this, seconds.
        tolerance: f64,
        /// Give up after this many iterations rather than not terminating.
        max_iterations: usize,
    },
}

impl Signal {
    /// The down-leg convention at the tolerance and budget the workspace's other light-time
    /// callers use: femtoseconds, twenty iterations.
    #[must_use]
    pub const fn down_leg() -> Self {
        Self::DownLeg {
            tolerance: 1e-15,
            max_iterations: 20,
        }
    }
}

/// An observation reduced to the state and the epoch its geometry belongs to.
///
/// Constructed by [`reduce`] and consumed by [`partials_at_emission`]. It carries the line of sight
/// and the radial speed as well as the state because the partial correction needs them and
/// recomputing them at the call site is how two call sites come to disagree about which `ρ̂` they
/// meant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observed {
    state: [f64; 6],
    transit: Transit,
    line_of_sight: [f64; 3],
    range_rate: f64,
    corrected: bool,
}

impl Observed {
    /// The satellite state the observable is to be evaluated at.
    #[must_use]
    pub const fn state(&self) -> &[f64; 6] {
        &self.state
    }

    /// When the signal left, when it arrived, and how long it took. Zero delay, and `emit ==
    /// receive`, under [`Signal::Instantaneous`].
    #[must_use]
    pub const fn transit(&self) -> Transit {
        self.transit
    }

    /// The unit line of sight `ρ̂`, from the station to the satellite at the emission epoch.
    #[must_use]
    pub const fn line_of_sight(&self) -> [f64; 3] {
        self.line_of_sight
    }

    /// `ρ̇ = ρ̂·v_sat`, the satellite's **own** radial speed — not the relative range rate, which
    /// adds the station's motion and is a different quantity (2.6 m of range on the ISS day).
    #[must_use]
    pub const fn range_rate(&self) -> f64 {
        self.range_rate
    }
}

/// Solve for the epoch an observation's geometry belongs to.
///
/// `receive` is the reception epoch in the caller's own timescale, `station` is where the receiver
/// was at that epoch — a constant of the problem, because the observation happened there — and
/// `ephemeris` returns the satellite's six-element state at an epoch, or `None` where it does not
/// reach. It must be the same station the observable carries; nothing here can check that, which is
/// the one way to misuse this function.
///
/// Returns `None` if the ephemeris does not reach an epoch the iteration asks for, or if the
/// iteration does not converge inside its budget — the same non-termination discipline the
/// integrators carry.
///
/// # Frames
///
/// Work in an inertial frame. A light-time iteration in a rotating frame has to carry the frame's
/// rotation during the transit as a fictitious term — the Sagnac correction, about 1.5 m at the
/// horizon for a LEO pass, which is the same size as the effect being solved for. Rotating the
/// station into an inertial frame at its own epoch is the same computation done once and openly.
pub fn reduce<E>(signal: Signal, receive: f64, station: &Station, ephemeris: E) -> Option<Observed>
where
    E: Fn(f64) -> Option<[f64; 6]>,
{
    let (state, transit, corrected) = match signal {
        Signal::Instantaneous => {
            let state = ephemeris(receive)?;
            let transit = Transit {
                delay: 0.0,
                emit: receive,
                receive,
            };
            (state, transit, false)
        }
        Signal::DownLeg {
            tolerance,
            max_iterations,
        } => {
            let position = |t: f64| ephemeris(t).map(|s| [s[0], s[1], s[2]]);
            let transit = down_leg(
                receive,
                station.position,
                position,
                tolerance,
                max_iterations,
            )?;
            (ephemeris(transit.emit)?, transit, true)
        }
    };

    let dr = [
        state[0] - station.position[0],
        state[1] - station.position[1],
        state[2] - station.position[2],
    ];
    let range = libm::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]);
    // A station standing exactly where the satellite is has no line of sight, and neither does a
    // non-finite state -- both would divide by zero or by NaN and hand back a direction that looks
    // like one. Refused here rather than downstream, where the NaN would already be in a partial.
    if !range.is_finite() || range <= 0.0 {
        return None;
    }
    let line_of_sight = [dr[0] / range, dr[1] / range, dr[2] / range];
    let range_rate =
        line_of_sight[0] * state[3] + line_of_sight[1] * state[4] + line_of_sight[2] * state[5];

    Some(Observed {
        state,
        transit,
        line_of_sight,
        range_rate,
        corrected,
    })
}

/// Evaluate an observable and its partials at the epoch its geometry belongs to.
///
/// The value is [`partials`] at the emission state and nothing more —
/// the formulae in `measurement` are already right there. The state row is that row corrected for
/// the state's own effect on the flight time:
///
/// ```text
/// ∂f/∂x₀ = [w − s·(ρ̂, 0)] Φ        s = (w·ẋ_e)/(c + ρ̇)
/// ```
///
/// so what comes back is still a row to chain through the state transition matrix, and a caller's
/// pipeline does not change shape. Under [`Signal::Instantaneous`] the correction is not applied,
/// which is what makes that variant a statement rather than a default.
///
/// `acceleration` is the satellite's acceleration at the emission epoch, in the working frame. It
/// enters only through the velocity components of `w`, so a range or an angle is exactly unaffected
/// by it and only a range-rate is not — see the module documentation for why it is an argument.
#[must_use]
pub fn partials_at_emission<const K: usize, O: Observable<K>>(
    observable: &O,
    observed: &Observed,
    acceleration: [f64; 3],
    parameters: &[f64; K],
) -> Partials<K> {
    let mut computed = partials(observable, observed.state(), parameters);
    if !observed.corrected {
        return computed;
    }

    let state_rate = [
        observed.state[3],
        observed.state[4],
        observed.state[5],
        acceleration[0],
        acceleration[1],
        acceleration[2],
    ];
    let mut numerator = 0.0;
    for (w, x) in computed.state.iter().zip(&state_rate) {
        numerator += w * x;
    }
    let scale = numerator / (SPEED_OF_LIGHT + observed.range_rate);
    for (slot, u) in computed.state.iter_mut().zip(&observed.line_of_sight) {
        *slot -= scale * u;
    }
    computed
}

#[cfg(test)]
mod tests {
    // `float_cmp` is allowed deliberately and narrowly: several assertions here are *about* bit
    // identity -- that the instantaneous convention leaves a row exactly alone, and that the
    // acceleration reaches a range-rate partial and no other. A tolerance there would pass on a
    // correction that was applied and happened to be small, which is the defect being excluded.
    #![allow(clippy::expect_used, clippy::float_cmp, clippy::unwrap_used)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;
    use crate::modifier::{Declination, Range, RangeRate, RightAscension};

    /// A straight-line satellite, which is the one trajectory whose light-time solution is a closed
    /// form: `c τ = |r₀ + v(t_r − τ)|` is linear when the motion is radial, giving
    /// `τ = (x₀ + v t_r)/(c + v)`. Every number below is checked against that rather than against a
    /// second run of the iteration.
    fn straight(r0: [f64; 3], v: [f64; 3]) -> impl Fn(f64) -> Option<[f64; 6]> {
        move |t: f64| {
            Some([
                r0[0] + v[0] * t,
                r0[1] + v[1] * t,
                r0[2] + v[2] * t,
                v[0],
                v[1],
                v[2],
            ])
        }
    }

    /// The state transition matrix of that trajectory, exactly: position picks up `t` times the
    /// velocity and the velocity is constant.
    fn straight_stm(t: f64) -> [[f64; 6]; 6] {
        let mut phi = [[0.0; 6]; 6];
        for (i, row) in phi.iter_mut().enumerate() {
            row[i] = 1.0;
            if i < 3 {
                row[i + 3] = t;
            }
        }
        phi
    }

    /// A station **on** the line of motion but not at the origin. The offset matters: a station at
    /// the origin makes `r_sat − r_station` equal to `r_sat`, so the subtraction that forms the line
    /// of sight cannot be wrong, and a mutation run showed exactly that — three sign mutants in that
    /// subtraction survived a test suite that put every station at zero. Being *on* the line keeps
    /// the closed form `τ = (x₀ − p + v t_r)/(c + v)` exact.
    const STATION: Station = Station::fixed([1_500_000.0, 0.0, 0.0]);

    /// And a station off it entirely, for the geometry that has no closed form: the same run showed
    /// that with axis-aligned motion two of the three terms of `ρ̂·v` are identically zero, so the
    /// operators joining them are unmeasured.
    const OFF_AXIS: Station = Station {
        position: [1_500_000.0, -900_000.0, 2_100_000.0],
        velocity: [40.0, 110.0, -7.0],
    };

    const RECEIVE: f64 = 100.0;

    /// Radial motion **relative to `STATION`**, so the closed form applies component by component.
    fn radial() -> ([f64; 3], [f64; 3]) {
        ([7_000_000.0, 0.0, 0.0], [7660.0, 0.0, 0.0])
    }

    /// A trajectory with no component identically zero, for the tests that are about the geometry
    /// rather than about a closed form.
    fn skewed() -> ([f64; 3], [f64; 3]) {
        (
            [7_000_000.0, 1_000_000.0, 2_000_000.0],
            [1200.0, -6900.0, 3100.0],
        )
    }

    #[test]
    fn the_flight_time_is_the_closed_form_of_the_straight_line() {
        let (r0, v) = radial();
        let observed = reduce(Signal::down_leg(), RECEIVE, &STATION, straight(r0, v))
            .expect("a straight line reaches every epoch");
        let expected = (r0[0] - STATION.position[0] + v[0] * RECEIVE) / (SPEED_OF_LIGHT + v[0]);
        assert_within_ulps(observed.transit().delay, expected, 2, "delay");
        assert_within_ulps(observed.transit().emit, RECEIVE - expected, 2, "emit");
        assert_within_ulps(observed.range_rate(), v[0], 0, "range rate");
    }

    /// `c τ` and the range observable at the emission state are one number, not two models of it.
    /// That is the whole content of "light time is a choice of epoch".
    #[test]
    fn the_path_length_and_the_range_at_emission_are_the_same_number() {
        let (r0, v) = radial();
        let observed = reduce(Signal::down_leg(), RECEIVE, &STATION, straight(r0, v)).unwrap();
        let range = partials_at_emission(&Range { station: STATION }, &observed, [0.0; 3], &[]);
        assert_within_ulps(
            range.value,
            observed.transit().path(),
            1,
            "path against range",
        );
    }

    /// The instantaneous convention is what `measurement` computes directly, and it is left exactly
    /// alone — no delay, no correction, the same row `partials` would have produced.
    #[test]
    fn the_instantaneous_convention_changes_nothing_at_all() {
        let (r0, v) = radial();
        let observed = reduce(Signal::Instantaneous, RECEIVE, &STATION, straight(r0, v)).unwrap();
        assert_eq!(observed.transit().delay, 0.0);
        assert_eq!(observed.transit().emit, RECEIVE);
        let at = straight(r0, v)(RECEIVE).unwrap();
        assert_eq!(observed.state(), &at);

        let corrected = partials_at_emission(&Range { station: STATION }, &observed, [0.0; 3], &[]);
        let plain = crate::modifier::partials(&Range { station: STATION }, &at, &[]);
        assert_eq!(corrected.state, plain.state, "the row was corrected anyway");
        assert_eq!(corrected.value, plain.value);
    }

    /// The range partial collapses to `1/(1 + ρ̇/c)`, derived in
    /// `derivations/light_time_partials.mac` and checked here against the closed form's own
    /// derivative `c/(c + v)` rather than against the formula it came from.
    #[test]
    fn a_range_partial_is_the_instantaneous_one_over_one_plus_the_radial_speed_over_c() {
        let (r0, v) = radial();
        let observed = reduce(Signal::down_leg(), RECEIVE, &STATION, straight(r0, v)).unwrap();
        let corrected = partials_at_emission(&Range { station: STATION }, &observed, [0.0; 3], &[]);
        let closed = SPEED_OF_LIGHT / (SPEED_OF_LIGHT + v[0]);
        assert_within_ulps(corrected.state[0], closed, 2, "d(c tau)/d(x0)");
        // 2.56e-5 relative at LEO -- the number the module documentation quotes.
        let relative = (corrected.state[0] - 1.0).abs();
        assert!(
            (relative - 2.555e-5).abs() < 1e-8,
            "the correction is {relative}, not 2.56e-5"
        );
    }

    /// At the instant of closest approach the radial speed is zero and the range partial is
    /// unscaled. The instant that matters is the **emission** one, and writing the test the obvious
    /// way -- closest approach at the reception epoch -- puts the satellite a whole transit time
    /// past it and leaves `ρ̇ = -0.196` m/s. That is the same class of error the module exists to
    /// close, arrived at from the other side.
    #[test]
    fn the_radial_speed_is_zero_at_the_emission_epoch_not_the_reception_one() {
        let (r0, v) = skewed();
        let p = OFF_AXIS.position;
        // Closest approach to *this* station, not to the origin: for a straight line it is where
        // `(r(t) − p)·v = 0`. Solved here rather than arranged by putting the station at zero,
        // which is what let three sign mutants live in the subtraction that forms `ρ̂`.
        let along: f64 = (0..3).map(|k| (p[k] - r0[k]) * v[k]).sum();
        let speed_squared: f64 = v.iter().map(|c| c * c).sum();
        let closest = along / speed_squared;
        let at_closest = straight(r0, v)(closest).unwrap();
        let miss: f64 = (0..3)
            .map(|k| (at_closest[k] - p[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        // The reception epoch sits one transit time after the emission we want.
        let receive = closest + miss / SPEED_OF_LIGHT;
        let observed = reduce(Signal::down_leg(), receive, &OFF_AXIS, straight(r0, v)).unwrap();
        assert!(
            (observed.transit().emit - closest).abs() < 1e-12,
            "emission at {}, not at closest approach {closest}",
            observed.transit().emit
        );
        assert!(
            observed.range_rate().abs() < 1e-9,
            "radial speed {} at closest approach",
            observed.range_rate()
        );
        // At zero radial speed `s = (w·ẋ_e)/(c + ρ̇)` has `w·ẋ_e = ρ̂·v = ρ̇ = 0` for a range, so the
        // correction vanishes identically and the corrected row *is* the uncorrected one. Stated as
        // a comparison of the two rows rather than as a component value, which is what makes it a
        // statement about the correction and not about where this particular station happens to be.
        let corrected =
            partials_at_emission(&Range { station: OFF_AXIS }, &observed, [0.0; 3], &[]);
        let plain = crate::modifier::partials(&Range { station: OFF_AXIS }, observed.state(), &[]);
        for (c, u) in corrected.state.iter().zip(&plain.state) {
            assert!(
                (c - u).abs() <= 1e-15 * u.abs().max(1.0),
                "the correction did not vanish at closest approach: {c} against {u}"
            );
        }

        // And taking the reception epoch instead is the error being guarded against: at the epoch
        // the satellite is already `v τ` past closest approach, so the radial speed is no longer
        // zero. Asserted against the geometry recomputed at that epoch rather than against a
        // remembered constant, so it stays true if the fixture orbit ever moves.
        let late = reduce(Signal::down_leg(), receive, &OFF_AXIS, straight(r0, v)).unwrap();
        let at_reception = straight(r0, v)(receive).unwrap();
        let d: Vec<f64> = (0..3).map(|k| at_reception[k] - p[k]).collect();
        let norm: f64 = d.iter().map(|c| c * c).sum::<f64>().sqrt();
        let radial_at_reception: f64 = (0..3).map(|k| d[k] / norm * at_reception[k + 3]).sum();
        assert!(
            radial_at_reception.abs() > 100.0 * observed.range_rate().abs().max(1e-9),
            "the two epochs read the same radial speed, so the test proves nothing: {} against {}",
            radial_at_reception,
            observed.range_rate()
        );
        assert!(
            (late.range_rate() - observed.range_rate()).abs() < 1e-9,
            "the same reception epoch gave two answers"
        );
    }

    /// The rank-one row form against a finite difference of the *whole* pipeline: perturb the epoch
    /// state, re-solve the light time, re-evaluate the observable, and compare against
    /// `[w − s(ρ̂,0)] Φ`. This is what makes the derivation a measurement rather than an assertion,
    /// and it is run on three observables whose `w` has a different shape.
    #[test]
    fn the_rank_one_row_reproduces_a_finite_difference_through_the_whole_solution() {
        let (r0, v) = radial();
        // Off-axis, so right ascension and declination are both non-degenerate.
        let r0 = [r0[0], 1_000_000.0, 2_000_000.0];
        let v = [v[0], 900.0, -300.0];

        let value = |x0: [f64; 6], which: usize| -> f64 {
            let eph = straight([x0[0], x0[1], x0[2]], [x0[3], x0[4], x0[5]]);
            let o = reduce(Signal::down_leg(), RECEIVE, &STATION, eph).unwrap();
            match which {
                0 => partials_at_emission(&Range { station: STATION }, &o, [0.0; 3], &[]).value,
                1 => {
                    partials_at_emission(&RightAscension { station: STATION }, &o, [0.0; 3], &[])
                        .value
                }
                _ => {
                    partials_at_emission(&Declination { station: STATION }, &o, [0.0; 3], &[]).value
                }
            }
        };

        let x0 = [r0[0], r0[1], r0[2], v[0], v[1], v[2]];
        let observed = reduce(Signal::down_leg(), RECEIVE, &STATION, straight(r0, v)).unwrap();
        // `Φ(t_e, t₀)`, not `Φ(t_r, t₀)`. The row belongs to the emission state, so the transition
        // matrix chained to it has to reach that epoch and not the reception one. Getting this
        // wrong is worth `τ/t_r = 2.3e-4` here -- an order of magnitude *more* than the 2.56e-5
        // light-time correction itself, which is why it is written out rather than inlined.
        let phi = straight_stm(observed.transit().emit);

        for which in 0..3 {
            let row = match which {
                0 => partials_at_emission(&Range { station: STATION }, &observed, [0.0; 3], &[]),
                1 => partials_at_emission(
                    &RightAscension { station: STATION },
                    &observed,
                    [0.0; 3],
                    &[],
                ),
                _ => partials_at_emission(
                    &Declination { station: STATION },
                    &observed,
                    [0.0; 3],
                    &[],
                ),
            }
            .state;

            for j in 0..6 {
                // Chain the analytic row through the exact STM, which is what a caller does.
                let mut analytic = 0.0;
                for i in 0..6 {
                    analytic += row[i] * phi[i][j];
                }
                // Central difference on the epoch state, stepped at the scale of the component.
                let h = if j < 3 { 1.0 } else { 1e-3 };
                let mut plus = x0;
                let mut minus = x0;
                plus[j] += h;
                minus[j] -= h;
                let numeric = (value(plus, which) - value(minus, which)) / (2.0 * h);
                // A central difference cannot resolve better than the cancellation in its own
                // subtraction: two values of size `|f|` differ in their last `eps*|f|` bits, and
                // dividing by `2h` turns that into a floor on the derivative. Right ascension
                // against the z component sits exactly there -- it depends on z *only* through the
                // flight time, so the true derivative is 1.8e-14 while the observable is 0.14 rad,
                // and a relative gate alone would be asserting f64 round-off.
                let nominal = value(x0, which).abs();
                let floor = 8.0 * f64::EPSILON * nominal / (2.0 * h);
                let scale = analytic.abs().max(numeric.abs()).max(1e-12);
                assert!(
                    (analytic - numeric).abs() <= (5e-8 * scale).max(floor),
                    "observable {which}, component {j}: {analytic} against {numeric}, floor {floor}"
                );
            }
        }
    }

    /// The acceleration reaches only the velocity rows of `w`, so a range and an angle are exactly
    /// unaffected by it and a range-rate is not. Asserted as an equality on the first two, because
    /// "small" would not distinguish an exact zero from a term that happens to be tiny here.
    #[test]
    fn the_acceleration_reaches_a_range_rate_and_nothing_else() {
        let (r0, v) = radial();
        let r0 = [r0[0], 1_000_000.0, 0.0];
        let v = [v[0], 900.0, 0.0];
        let observed = reduce(Signal::down_leg(), RECEIVE, &STATION, straight(r0, v)).unwrap();
        let gravity = [-8.0, -1.1, 0.0];

        for zero_and_gravity in [
            (
                partials_at_emission(&Range { station: STATION }, &observed, [0.0; 3], &[]),
                partials_at_emission(&Range { station: STATION }, &observed, gravity, &[]),
            ),
            (
                partials_at_emission(&Declination { station: STATION }, &observed, [0.0; 3], &[]),
                partials_at_emission(&Declination { station: STATION }, &observed, gravity, &[]),
            ),
        ] {
            assert_eq!(
                zero_and_gravity.0.state, zero_and_gravity.1.state,
                "the acceleration moved a partial whose velocity rows are zero"
            );
        }

        let still = partials_at_emission(&RangeRate { station: STATION }, &observed, [0.0; 3], &[]);
        let falling =
            partials_at_emission(&RangeRate { station: STATION }, &observed, gravity, &[]);
        assert!(
            still.state != falling.state,
            "a range-rate partial ignored the acceleration"
        );
    }

    /// The two accessors, against the geometry recomputed from the emission epoch by hand.
    ///
    /// They exist because the partial correction needs them and recomputing them at the call site is
    /// how two call sites come to disagree; a mutation run showed that nothing asserted either of
    /// them, so `line_of_sight` could have returned a constant. The geometry here is fully
    /// three-dimensional on both sides — the same run showed that with an axis-aligned trajectory
    /// two of the three terms of `ρ̂·v` are identically zero, so the operators joining them are
    /// unmeasured whatever the tolerance.
    #[test]
    fn the_line_of_sight_and_the_radial_speed_are_the_emission_geometry() {
        let (r0, v) = skewed();
        let observed = reduce(Signal::down_leg(), RECEIVE, &OFF_AXIS, straight(r0, v)).unwrap();

        let at_emission = straight(r0, v)(observed.transit().emit).unwrap();
        let d: Vec<f64> = (0..3)
            .map(|k| at_emission[k] - OFF_AXIS.position[k])
            .collect();
        let norm: f64 = d.iter().map(|c| c * c).sum::<f64>().sqrt();

        // No component of either vector is zero, so every operator that builds them is exercised.
        for (k, u) in observed.line_of_sight().iter().enumerate() {
            // The smallest of the three is 0.036 here. The gate is not a tolerance on the answer,
            // it is the guard that keeps this test from degenerating into the axis-aligned geometry
            // whose zeros hid three mutants: a component that drifts to zero makes the operators
            // around it unmeasured again, silently.
            assert!(u.abs() > 0.01, "component {k} of the line of sight is {u}");
            assert!(
                (u - d[k] / norm).abs() < 1e-15,
                "line of sight component {k}: {u} against {}",
                d[k] / norm
            );
        }

        let radial: f64 = (0..3).map(|k| d[k] / norm * at_emission[k + 3]).sum();
        assert!(
            radial.abs() > 100.0,
            "the geometry is nearly transverse: {radial} m/s"
        );
        assert!(
            (observed.range_rate() - radial).abs() < 1e-9,
            "radial speed {} against {radial}",
            observed.range_rate()
        );

        // And it is the *satellite's own* radial speed, not the relative one: the station moves too,
        // and mistaking the two is the 2.6 m error the module documentation names.
        let relative: f64 = (0..3)
            .map(|k| d[k] / norm * (at_emission[k + 3] - OFF_AXIS.velocity[k]))
            .sum();
        assert!(
            (observed.range_rate() - relative).abs() > 1.0,
            "the two range rates are indistinguishable here, so the test proves nothing"
        );
    }

    /// The refusals: an ephemeris that does not reach, and a station standing where the satellite
    /// is. Both return `None` rather than a NaN that would already be inside a partial.
    #[test]
    fn a_gap_and_a_zero_baseline_are_refused_rather_than_returned() {
        let gap = |_: f64| None;
        assert!(reduce(Signal::down_leg(), RECEIVE, &STATION, gap).is_none());
        assert!(reduce(Signal::Instantaneous, RECEIVE, &STATION, gap).is_none());

        let p = STATION.position;
        let at_the_station = move |_: f64| Some([p[0], p[1], p[2], 0.0, 0.0, 0.0]);
        assert!(reduce(Signal::Instantaneous, RECEIVE, &STATION, at_the_station).is_none());

        // A budget of one iteration cannot converge from a cold start, and says so.
        let (r0, v) = radial();
        let starved = Signal::DownLeg {
            tolerance: 1e-15,
            max_iterations: 1,
        };
        assert!(reduce(starved, RECEIVE, &STATION, straight(r0, v)).is_none());
    }
}
