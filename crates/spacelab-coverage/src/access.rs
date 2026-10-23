//! Access windows — the seam that turns a satellite trajectory, a ground cell and a sensor into the
//! time intervals the cell is covered.
//!
//! The building blocks upstream ([`crate::find_events`], [`crate::brent`],
//! [`spacelab_sensors::Constraint`]) are pieces; this is where they compose. The scan is the *correct
//! primitive* from the design: rather than sampling `g(t)` on a fixed grid and hoping a short pass
//! falls on a sample, it steps by `|g| / max_rate` — an interval the switching function provably
//! cannot cross, because `max_rate` bounds `|dg/dt|` — so **no pass, however short, is stepped over**.
//! Each sign change is then bracketed and refined to `tol` seconds by Brent. This is the consumer the
//! sensors' [`max_rate`](spacelab_sensors::Constraint::max_rate) bound was built for.

// The scan driver's arguments are each a distinct, named quantity (the sensor, the orbit bound, the
// frame rotation, the cell geometry, the trajectory, the scan interval, the sink); grouping them
// behind a struct would hide the seam rather than clarify it.
#![allow(clippy::too_many_arguments)]

use spacelab_core::Vec3;
use spacelab_sensors::{AccessGeometry, Constraint, OrbitBound};

use crate::root::{advancing_step, brent};

/// One interval of coverage, in the caller's time base (seconds from its reference epoch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// When coverage opens (a rising zero of the switching function, or the scan start if already
    /// covered).
    pub rise: f64,
    /// When coverage closes (a falling zero, or the scan end if still covered).
    pub set: f64,
}

impl Window {
    /// The window's length in seconds.
    #[must_use]
    pub fn duration(&self) -> f64 {
        self.set - self.rise
    }

    /// The window's midpoint, the epoch of closest approach for a symmetric pass.
    #[must_use]
    pub fn midpoint(&self) -> f64 {
        f64::midpoint(self.rise, self.set)
    }

    /// Whether `t` lies within the window.
    #[must_use]
    pub fn contains(&self, t: f64) -> bool {
        t >= self.rise && t <= self.set
    }
}

/// Emit the intervals on which `g(t) >= 0`, over `[t0, t1]`, given an upper bound `max_rate` on
/// `|dg/dt|`. The general scanner beneath [`access_windows`]: works with any switching function —
/// a sensor constraint, a link margin, a raw coordinate — for which a rate bound is known.
///
/// The step is adaptive and rigorous: from a value `g` the next probe is `|g| / max_rate` away, an
/// interval `g` cannot traverse, floored by a tiny fraction of the span so a zero it grazes cannot
/// stall the walk, and by four ulps of the endpoints, without which that first floor stalls anyway
/// once `t0` is an absolute epoch large enough that a nanosecond falls below its ulp. Windows
/// open/close at
/// Brent-refined crossings; a window already open at `t0` or
/// still open at `t1` is clamped to the scan bound. `no_std`, no allocation — each window via `emit`.
pub fn windows<G, E>(mut g: G, max_rate: f64, t0: f64, t1: f64, tol: f64, mut emit: E)
where
    G: FnMut(f64) -> f64,
    E: FnMut(Window),
{
    let h_min = advancing_step(t0, t1, (t1 - t0) * 1e-9);
    let mut t = t0;
    let mut g_prev = g(t0);
    let mut open = if g_prev >= 0.0 { Some(t0) } else { None };

    while t < t1 {
        let step = (libm::fabs(g_prev) / max_rate).max(h_min);
        let t_next = (t + step).min(t1);
        let g_next = g(t_next);

        if (g_prev < 0.0) != (g_next < 0.0)
            && let Ok(bracket) = brent(&mut g, t, t_next, tol)
        {
            let crossing = bracket.estimate();
            if g_next >= 0.0 {
                open = Some(crossing); // rising: coverage begins
            } else if let Some(rise) = open.take() {
                emit(Window {
                    rise,
                    set: crossing,
                }); // falling: coverage ends
            }
        }
        t = t_next;
        g_prev = g_next;
    }

    if let Some(rise) = open {
        emit(Window { rise, set: t1 });
    }
}

/// Emit the access windows of `sensor` over the ground `cell` (outward normal `zenith`) as the
/// satellite follows `satellite_at`, on `[t0, t1]`, edges refined to `tol` seconds.
///
/// `cell` and the satellite must share a frame; `frame_rate` is that frame's rotation rate — `0` for
/// an inertial cell, `ω⊕` for an Earth-fixed one — which enters the [`max_rate`] bound. This is the
/// convenience over [`windows`]: it forms `g(t) = sensor.g(geometry(t))` and the rate bound for you.
///
/// [`max_rate`]: spacelab_sensors::Constraint::max_rate
pub fn access_windows<C, P, E>(
    sensor: &C,
    orbit: &OrbitBound,
    frame_rate: f64,
    cell: Vec3,
    zenith: Vec3,
    satellite_at: P,
    t0: f64,
    t1: f64,
    tol: f64,
    emit: E,
) where
    C: Constraint,
    P: Fn(f64) -> Vec3,
    E: FnMut(Window),
{
    let g = |t: f64| {
        sensor.g(&AccessGeometry {
            satellite: satellite_at(t),
            cell,
            zenith,
        })
    };
    windows(g, sensor.max_rate(orbit, frame_rate), t0, t1, tol, emit);
}
