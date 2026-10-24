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

#![allow(clippy::doc_markdown)] // program names and math notation, not identifiers

use crate::root::{Bracket, advancing_step, brent};

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
/// # Design
///
/// The sign is tracked with `signum`, and an interval qualifies when consecutive signs differ. A
/// zero sample is not itself an event — a continuous `g` that touches zero without crossing (a
/// tangency, `g = g' = 0`) is not an access boundary, and treating a touch as a crossing would
/// invent an event. Only a genuine sign change between the bracket ends is refined.
pub fn find_events<F, E>(mut g: F, t0: f64, t1: f64, step: f64, tol: f64, mut emit: E) -> u32
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
    let mut count = 0u32;
    let mut t_prev = t0;
    let mut sign_prev = sign_of(g(t0));

    // Number of whole steps, plus a final partial step landing exactly on t1.
    loop {
        let t_next = (t_prev + step).min(t1);
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
