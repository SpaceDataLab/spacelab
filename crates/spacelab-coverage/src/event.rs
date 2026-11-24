//! Coarse sign-change scan, then bracket-preserving refinement.
//!
//! [`find_events`] walks a continuous `g(t)` across `[t0, t1]` in steps no larger than `step`,
//! brackets every interval where `g` changes sign, and refines each with [`crate::brent`]. It emits
//! events through a callback, so it allocates nothing and stays `no_std`.
//!
//! # What `step` guarantees, and what it does not
//!
//! With step `h`, the scan finds every root that is *isolated within a step* — every sign change of
//! `g` between consecutive samples. It cannot see a feature entirely inside one step: an access that
//! opens and closes within `h` (two roots, `g` negative at both samples) leaves no sign change and
//! is missed. That is the Nyquist limit, and it is the caller's job to choose `h` below the shortest
//! access worth reporting.
//!
//! The right `h` is *derived*, not guessed: `h = g_swing / max_rate`, where `max_rate` bounds
//! `|dg/dt|` (`spacelab_sensors::Constraint::max_rate`, falsification-tested) and `g_swing` is the
//! smallest peak `g` a reportable access must reach. Gpredict fits this step (`find_aos`), PreviSat
//! asserts it, GMAT asks the user; here it comes from a bound, as in Orekit's
//! `ElevationDetectionAdaptableIntervalFactory`.
//!
//! Where that bound is known, [`find_events_rate_bounded`] takes it in place of a step and walks with
//! the step rule of [`crate::windows`]: from a sample `g` the next one is `|g| / max_rate` away, so a
//! step holds at most one crossing unless it is the floor, a billionth of the span. No pair of
//! crossings at least the floor apart is missed, under the hypothesis that `max_rate` bounds
//! `|dg/dt|` of the `g` the scan evaluates.
//!
//! `tests/event.rs` holds the scanner to known roots and crossing directions, and to the limit the
//! step bound states. It plants a pair of crossings inside one fixed step: [`find_events`] misses
//! it, and [`find_events_rate_bounded`] finds both at their exact times.

#![allow(clippy::doc_markdown)] // program names and math notation, not identifiers

use crate::root::{Bracket, advancing_step, brent, rate_bound_floor, rate_bounded_step};

/// Whether `g` crosses zero upward (into coverage) or downward (out of coverage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// `g` goes from negative to positive: an access *opens*.
    Rising,
    /// `g` goes from positive to negative: an access *closes*.
    Falling,
}

/// A located crossing of `g(t)`: its direction and the bracket around its time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    /// Rising (access opens) or falling (access closes).
    pub kind: EventKind,
    /// The refined root, as a bracket in `t`.
    pub at: Bracket,
}

/// Why [`find_events_rate_bounded`] refused its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanError {
    /// `max_rate` was not a positive, finite number.
    BadRate,
    /// `t0` or `t1` was not finite, `t1 <= t0`, or the step floor underflowed to zero.
    BadInterval,
    /// `tol` was not a positive, finite number.
    BadTolerance,
}

/// Scan `g` on `[t0, t1]` and report every sign change, refined to abscissa tolerance `tol`.
///
/// `step` is the coarse sampling interval; it must be positive, and is raised to four ulps of the
/// scan bounds if it is finer than the abscissa can represent there — a step below one ulp of `t`
/// leaves `t + step == t`, so the scan would never reach `t1`. `tol` is the Brent tolerance, in
/// the units of `t` (seconds). Each event is passed to `emit` in increasing time order. Exact-zero
/// samples are treated as the sign of the preceding interval so a grazing sample does not double
/// count.
///
/// Returns the number of events emitted.
///
/// # What it misses
///
/// Two crossings inside one step leave both samples on the same side of zero, so neither is
/// reported: a pass, an eclipse or a contact shorter than `step` is lost without a sign. An odd
/// number of crossings inside one step is reported once. The scan cannot detect the miss; the step
/// is the caller's choice. When `|dg/dt|` has a known bound, [`find_events_rate_bounded`] misses no
/// pair of crossings at least its floor apart.
///
/// # Design
///
/// The sign is tracked with `signum`, and an interval qualifies when consecutive signs differ. A
/// zero sample is not itself an event — a continuous `g` that touches zero without crossing (a
/// tangency, `g = g' = 0`) is not an access boundary, and treating a touch as a crossing would
/// invent an event. Only a genuine sign change between the bracket ends is refined.
pub fn find_events<F, E>(g: F, t0: f64, t1: f64, step: f64, tol: f64, emit: E) -> u32
where
    F: FnMut(f64) -> f64,
    E: FnMut(Event),
{
    // Positive form so a NaN argument fails the guard (NaN > 0.0 is false) rather than slipping
    // through a `<=` test.
    let arguments_ok = step > 0.0 && t1 > t0 && tol.is_finite();
    if !arguments_ok {
        return 0;
    }

    let step = advancing_step(t0, t1, step);
    scan(g, t0, t1, tol, emit, |_| step)
}

/// Scan `g` on `[t0, t1]` with the step rule of [`crate::windows`], and report every sign change,
/// refined to abscissa tolerance `tol`.
///
/// `max_rate` bounds `|dg/dt|` on `[t0, t1]`, in units of `g` per unit of `t`. From a sample `g` the
/// next one is `|g| / max_rate` away, an interval over which `g` cannot reach zero before its end.
/// The step never falls below the floor `h_min`: a billionth of `t1 − t0`, raised to four ulps of
/// the bounds so the walk always moves. Events go to `emit` in increasing time order, with the
/// zero-sample rule of [`find_events`]. Returns the number of events emitted.
///
/// # What it guarantees
///
/// A step longer than `h_min` holds at most one crossing, at its end. Two crossings can share a step
/// only if they are less than `h_min` apart, so **no pair of crossings `h_min` or more apart is
/// missed**. The guarantee holds only if `max_rate` bounds the rate of the `g` that is evaluated: a
/// bound that is too small lets a step jump a pass, and `tests/event.rs` shows that too.
///
/// The cost: near a crossing where `|dg/dt| = r·max_rate`, each step shrinks the distance to it by
/// the factor `1 − r`. A bound `1/r` times too loose spends about `ln(d/h_min)/r` steps to close a
/// distance `d` to the crossing.
///
/// # Errors
///
/// `g` is not evaluated when the arguments are refused.
///
/// * [`ScanError::BadRate`] if `max_rate` is not positive and finite.
/// * [`ScanError::BadInterval`] if `t0` or `t1` is not finite, if `t1 <= t0`, or if `h_min`
///   underflows to zero, where the walk could stall: a span below about `2.5e-315` with both bounds
///   below about `2.8e-309`.
/// * [`ScanError::BadTolerance`] if `tol` is not positive and finite.
pub fn find_events_rate_bounded<F, E>(
    g: F,
    max_rate: f64,
    t0: f64,
    t1: f64,
    tol: f64,
    emit: E,
) -> Result<u32, ScanError>
where
    F: FnMut(f64) -> f64,
    E: FnMut(Event),
{
    if !(max_rate.is_finite() && max_rate > 0.0) {
        return Err(ScanError::BadRate);
    }
    let floor = rate_bound_floor(t0, t1);
    if !(t0.is_finite() && t1.is_finite() && t1 > t0 && floor > 0.0) {
        return Err(ScanError::BadInterval);
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(ScanError::BadTolerance);
    }
    Ok(scan(g, t0, t1, tol, emit, |g_now| {
        rate_bounded_step(g_now, max_rate, floor)
    }))
}

/// The walk both scans share. `next_step` maps the latest sample of `g` to the length of the next
/// step; everything else — the bracket, the zero-sample rule, the event order — is common, so the
/// two scans differ in their step and in nothing else.
fn scan<F, E, S>(mut g: F, t0: f64, t1: f64, tol: f64, mut emit: E, mut next_step: S) -> u32
where
    F: FnMut(f64) -> f64,
    E: FnMut(Event),
    S: FnMut(f64) -> f64,
{
    let mut count = 0u32;
    let mut t_prev = t0;
    let mut g_prev = g(t0);
    let mut sign_prev = sign_of(g_prev);

    // Number of whole steps, plus a final partial step landing exactly on t1.
    loop {
        let t_next = (t_prev + next_step(g_prev)).min(t1);
        let g_next = g(t_next);
        let sign_next = sign_of(g_next);

        // A sign change strictly across the interval is a root to refine. `sign_of` maps 0 to the
        // neutral 0; a transition through 0 (e.g. -1 -> 0 -> +1 across two intervals) is caught on
        // the interval whose ends are genuinely opposite in sign.
        if sign_prev != 0 && sign_next != 0 && sign_prev != sign_next {
            // Bracket exists by construction; brent cannot return NotBracketed here.
            if let Ok(at) = brent(&mut g, t_prev, t_next, tol) {
                let kind = if sign_next > 0 {
                    EventKind::Rising
                } else {
                    EventKind::Falling
                };
                emit(Event { kind, at });
                count += 1;
            }
        }

        if t_next >= t1 {
            break;
        }
        t_prev = t_next;
        g_prev = g_next;
        // Carry a nonzero sign forward across an exact zero so the next real sign sets the direction.
        if sign_next != 0 {
            sign_prev = sign_next;
        }
    }

    count
}

/// `-1`, `0`, or `+1`. Zero maps to `0` (neutral), so a touch is not mistaken for a crossing.
#[inline]
fn sign_of(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}
