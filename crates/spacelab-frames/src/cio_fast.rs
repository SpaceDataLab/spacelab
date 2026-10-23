//! CIO-frame interpolation — the fast GCRF→ITRF provider for high-fidelity Earth-fixed propagation.
//!
//! Almost all of the exact [`crate::gcrf_to_itrf`] is the IAU 2006/2000A CIP series
//! ([`spacelab_fa::cip_xy`], ~1600 amplitudes over 1309 distinct frequencies). Those coordinates are
//! driven by TT and so vary on **day** scales, while the Earth Rotation Angle (from UT1) is what
//! moves fast within a step. [`CioInterpolator`] precomputes the four slow scalars `(X, Y, s, s′)` on
//! a coarse grid over the propagation span and degree-`(STENCIL−1)` Lagrange-interpolates them per
//! query, applying the **exact** ERA every time. The per-stage cost drops from a ~1600-term series to
//! a handful of multiplies — two to three orders of magnitude on the machines measured so far; run
//! `cargo run --release --features std --example cio_bench` for the ratio on yours. It matches the exact
//! transform to the f64 floor (sub-nanometre at LEO) — see
//! `cio_interpolation_matches_the_exact_transform`.
//!
//! **A query outside the grid is never extrapolated.** A degree-7 polynomial leaves its stencil
//! badly, and an interpolator that answers plausibly for an epoch it does not cover is worse than one
//! that is slow: it is wrong without saying so. Outside the grid the four scalars are re-evaluated
//! from the series, so every query is correct and only the uncovered ones are slow.
//!
//! Polar motion `(xp, yp)`, the CIP offsets `(dx, dy)` and the length-of-day are held fixed over the
//! span — that is the approximation this type does make, and it is the one that decides the answer
//! (up to decimetres over a six-hour arc), not the interpolation. For a time-varying-EOP or
//! double-double path, use the exact per-stage [`crate::gcrf_to_itrf`].
//!
//! # How small the interpolation error is, as a bound rather than a sample
//!
//! `derivations/cio_interpolation.mac` does the Lagrange remainder — the node product is exactly
//! `11025/256` for a centred query and 14.87× that when the stencil is clamped at a grid end — and
//! `derivations/cio_interpolation.py` bounds `sup|d⁸X/dt⁸|` from the shipped
//! [`spacelab_fa`] table itself: **9.84e-48 rad/s⁸**, set by a 306 µas term of 5.6-day period, with
//! the fastest argument combination in the series at 3.495 days. At `GRID_DT` the truncation error
//! is then **1.3e-21 m at LEO** centred and 1.9e-20 m clamped.
//!
//! That is seven orders *below* the other half of the error, which is what actually sets the floor:
//! the nodes are `f64`, and Lagrange interpolation amplifies whatever they carry by the stencil's
//! Lebesgue constant — **1.488** centred, **6.93** clamped — so one ulp of a CIP coordinate
//! (4.34e-19 rad, 2.9 pm at LEO) comes out as 4.4 pm or 20 pm. The measured sub-nanometre agreement
//! is arithmetic, not method, and `GRID_DT` is not what sets it: truncation would not reach one
//! nanometre at LEO until the grid was **22× coarser**. `s′` is linear in `t`, so its eighth
//! derivative is zero and degree-7 interpolation reproduces it exactly.

// Grid indices and nanosecond stamps are small, in-range integer-valued quantities; the single-letter
// names (X, Y, s, s′ and the Lagrange indices) are the notation of the CIO chain and the interpolant.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names
)]

use spacelab_core::{Angle, Epoch, Tt, Ut1, Vec3};
use spacelab_fa::{cio_locator_s, cip_xy, era};

use crate::cio::{cirs_to_tirs, gcrf_to_cirs, tirs_to_itrf};
use crate::polar_motion::tio_locator;
use crate::rotation::{Gcrf, Itrf, Rotation, Transform};
use crate::teme::EARTH_ROTATION_RATE;

/// Lagrange stencil width; the interpolating polynomial has degree `STENCIL − 1`.
const STENCIL: usize = 8;
/// Grid spacing in seconds — the CIP is smooth on day scales, so a degree-7 stencil at 10-minute
/// spacing resolves it far below the f64 floor.
const GRID_DT: f64 = 600.0;

/// The four slow, TT-driven scalars of the CIO chain at one grid node.
#[derive(Clone, Copy)]
struct Node {
    x: f64,
    y: f64,
    s: f64,
    sp: f64,
}

/// A GCRF→ITRF transform provider that interpolates the slow precession-nutation geometry from a grid
/// precomputed over the propagation span, applying the exact Earth Rotation Angle per query.
pub struct CioInterpolator {
    nodes: Vec<Node>,
    /// Propagation time (s) of node 0; `≤ 0` so backward arcs are covered.
    t_start: f64,
    epoch_ut1: Epoch<Ut1>,
    /// The arc's start on TT, kept so an uncovered query can re-evaluate the series.
    epoch_tt: Epoch<Tt>,
    xp: Angle,
    yp: Angle,
    dx: Angle,
    dy: Angle,
    lod_seconds: f64,
}

impl CioInterpolator {
    /// Build the interpolator for propagation over `[min(0, span), max(0, span)]` seconds from the
    /// epoch. `epoch_ut1`/`epoch_tt` are the arc's start on the two time scales; `xp`/`yp` polar
    /// motion, `dx`/`dy` the celestial-pole offsets, `lod_seconds` the length-of-day excess — all held
    /// fixed over the span.
    #[must_use]
    #[allow(clippy::too_many_arguments)] // two epochs, the span, and the five fixed EOP scalars
    pub fn new(
        epoch_ut1: Epoch<Ut1>,
        epoch_tt: Epoch<Tt>,
        span: f64,
        xp: Angle,
        yp: Angle,
        dx: Angle,
        dy: Angle,
        lod_seconds: f64,
    ) -> Self {
        // Cover the arc plus a full stencil of margin at each end, so no query ever extrapolates.
        let margin = STENCIL as f64 * GRID_DT;
        let t_start = span.min(0.0) - margin;
        let reach = span.abs() + 2.0 * margin;
        let count = (reach / GRID_DT).ceil() as usize + 1;
        let mut nodes = Vec::with_capacity(count);
        let epoch_tt_ns = epoch_tt.ns_since_j2000();
        for i in 0..count {
            let ti = t_start + i as f64 * GRID_DT;
            let tt = Epoch::<Tt>::from_ns_since_j2000(epoch_tt_ns + (ti * 1e9).round() as i128);
            let (x, y) = cip_xy(tt);
            let s = cio_locator_s(tt, x, y);
            let sp = tio_locator(tt).radians();
            nodes.push(Node { x, y, s, sp });
        }
        Self {
            nodes,
            t_start,
            epoch_ut1,
            epoch_tt,
            xp,
            yp,
            dx,
            dy,
            lod_seconds,
        }
    }

    /// The GCRF→ITRF rotation at propagation time `t` seconds.
    #[must_use]
    pub fn rotation(&self, t: f64) -> Rotation<Gcrf, Itrf> {
        let node = self.interpolate(t);
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
            self.epoch_ut1.ns_since_j2000() + (t * 1e9).round() as i128,
        );
        let q = gcrf_to_cirs(
            node.x + self.dx.radians(),
            node.y + self.dy.radians(),
            node.s,
        );
        tirs_to_itrf(self.xp, self.yp, Angle::from_radians(node.sp)) * cirs_to_tirs(era(ut1)) * q
    }

    /// The GCRF→ITRF rotation as row-major matrix rows at propagation time `t` seconds — the form a
    /// force model's `BodyOriented` closure consumes, without the caller reaching through
    /// [`Rotation::matrix`].
    #[must_use]
    pub fn rotation_rows(&self, t: f64) -> [[f64; 3]; 3] {
        self.rotation(t).matrix().rows
    }

    /// The full GCRF→ITRF transform (rotation and angular velocity) at propagation time `t` seconds.
    #[must_use]
    pub fn transform(&self, t: f64) -> Transform<Gcrf, Itrf> {
        let node = self.interpolate(t);
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
            self.epoch_ut1.ns_since_j2000() + (t * 1e9).round() as i128,
        );
        let q = gcrf_to_cirs(
            node.x + self.dx.radians(),
            node.y + self.dy.radians(),
            node.s,
        );
        let rotation = tirs_to_itrf(self.xp, self.yp, Angle::from_radians(node.sp))
            * cirs_to_tirs(era(ut1))
            * q;
        let rate = EARTH_ROTATION_RATE * (1.0 - self.lod_seconds / 86_400.0);
        let omega = q.inverse().matrix() * Vec3::new(0.0, 0.0, rate);
        Transform::new(rotation, omega)
    }

    /// Whether propagation time `t` falls inside the precomputed grid.
    ///
    /// The grid covers the construction span plus a full stencil of margin at each end. A query
    /// outside it is still answered — from the series rather than the polynomial — so this reports
    /// which of the two paths a query would take, not whether it is allowed.
    #[must_use]
    pub fn covers(&self, t: f64) -> bool {
        self.node_index(t).is_some()
    }

    /// The fractional grid index of propagation time `t`, or `None` if the grid does not reach it.
    fn node_index(&self, t: f64) -> Option<f64> {
        let u = (t - self.t_start) / GRID_DT;
        (0.0..=(self.nodes.len() - 1) as f64)
            .contains(&u)
            .then_some(u)
    }

    /// The four scalars evaluated from the series, for a query the grid does not reach.
    fn exact_node(&self, t: f64) -> Node {
        let tt = Epoch::<Tt>::from_ns_since_j2000(
            self.epoch_tt.ns_since_j2000() + (t * 1e9).round() as i128,
        );
        let (x, y) = cip_xy(tt);
        Node {
            x,
            y,
            s: cio_locator_s(tt, x, y),
            sp: tio_locator(tt).radians(),
        }
    }

    /// Degree-`(STENCIL−1)` Lagrange interpolation of the four scalars at propagation time `t`.
    ///
    /// Outside the grid the stencil would have to be clamped and the polynomial evaluated beyond its
    /// nodes; a degree-7 extrapolation leaves the truth quickly and quietly. Those queries take the
    /// series instead, so this function interpolates or it is exact — it never extrapolates.
    fn interpolate(&self, t: f64) -> Node {
        let Some(u) = self.node_index(t) else {
            return self.exact_node(t);
        };
        let last = self.nodes.len() - STENCIL;
        let i0 = (u.floor() as isize - STENCIL as isize / 2 + 1).clamp(0, last as isize) as usize;
        // The promise above, checked rather than trusted. No accuracy test can make it: the CIP is
        // smooth enough over 600 s that sliding this stencil by four nodes in either direction —
        // past the point where the query leaves it — moves the position by 1.86 nm to 2.31 nm at
        // LEO. The property is which nodes were chosen, so it is asserted as an index.
        debug_assert!(
            i0 as f64 <= u && u <= (i0 + STENCIL - 1) as f64,
            "stencil [{i0}, {}] does not contain the query at u = {u}: this is extrapolation",
            i0 + STENCIL - 1
        );
        let xq = u - i0 as f64;
        let (mut x, mut y, mut s, mut sp) = (0.0, 0.0, 0.0, 0.0);
        for j in 0..STENCIL {
            let mut l = 1.0;
            for k in 0..STENCIL {
                if k != j {
                    l *= (xq - k as f64) / (j as f64 - k as f64);
                }
            }
            let n = self.nodes[i0 + j];
            x += l * n.x;
            y += l * n.y;
            s += l * n.s;
            sp += l * n.sp;
        }
        Node { x, y, s, sp }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gcrf_to_itrf;
    use spacelab_testkit::nan_max;

    #[test]
    fn cio_interpolation_matches_the_exact_transform() {
        // A representative recent epoch (TT ns since J2000) and a small UT1−TT offset.
        let epoch_tt = Epoch::<Tt>::from_ns_since_j2000(788_400_000_000_000_000);
        let epoch_ut1 = Epoch::<Ut1>::from_ns_since_j2000(788_400_000_000_000_000 - 69_000_000_000);
        let xp = Angle::from_arcseconds(0.16);
        let yp = Angle::from_arcseconds(0.35);
        let zero = Angle::from_radians(0.0);
        let span = 6.0 * 3600.0; // 6-hour arc

        let cio = CioInterpolator::new(epoch_ut1, epoch_tt, span, xp, yp, zero, zero, 0.0);
        // A LEO position to turn a rotation-matrix difference into metres.
        let r = Vec3::new(6_800e3, 1_200e3, 900e3);

        let mut worst = 0.0_f64;
        for k in 0..=240 {
            let t = span * f64::from(k) / 240.0;
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
                epoch_ut1.ns_since_j2000() + (t * 1e9).round() as i128,
            );
            let tt = Epoch::<Tt>::from_ns_since_j2000(
                epoch_tt.ns_since_j2000() + (t * 1e9).round() as i128,
            );
            let exact = gcrf_to_itrf(ut1, tt, xp, yp, zero, zero, 0.0)
                .rotation
                .matrix()
                * r;
            let interp = cio.rotation(t).matrix() * r;
            worst = nan_max(worst, (exact - interp).norm());
        }
        // Interpolation error is far below a micron at LEO — the CIP is polynomial-smooth on this arc.
        assert!(
            worst < 1e-6,
            "CIO interpolation error {worst:.3e} m exceeds 1 µm"
        );
    }

    /// A query the grid does not reach is answered from the series, not extrapolated.
    ///
    /// This is the test the type exists to survive. Clamping the stencil and evaluating the
    /// degree-7 polynomial a year past its last node is not a small error — it is metres to
    /// kilometres, delivered with no diagnostic. Here the uncovered queries must agree with the
    /// exact chain to the floating-point floor, i.e. be *the same computation*, while an in-span
    /// query must still take the fast path.
    #[test]
    fn a_query_outside_the_grid_is_exact_rather_than_extrapolated() {
        let epoch_tt = Epoch::<Tt>::from_ns_since_j2000(788_400_000_000_000_000);
        let epoch_ut1 = Epoch::<Ut1>::from_ns_since_j2000(788_400_000_000_000_000 - 69_000_000_000);
        let xp = Angle::from_arcseconds(0.16);
        let yp = Angle::from_arcseconds(0.35);
        let zero = Angle::from_radians(0.0);
        let span = 6.0 * 3600.0;

        let cio = CioInterpolator::new(epoch_ut1, epoch_tt, span, xp, yp, zero, zero, 0.0);
        let r = Vec3::new(6_800e3, 1_200e3, 900e3);

        assert!(cio.covers(0.0) && cio.covers(span), "the arc is covered");

        // One day, one month and one year outside the arc, in both directions.
        let mut worst = 0.0_f64;
        for days in [-365.0, -30.0, -1.0, 1.0, 30.0, 365.0] {
            let t = span + days * 86_400.0;
            assert!(!cio.covers(t), "t = {t} should be outside the grid");
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
                epoch_ut1.ns_since_j2000() + (t * 1e9).round() as i128,
            );
            let tt = Epoch::<Tt>::from_ns_since_j2000(
                epoch_tt.ns_since_j2000() + (t * 1e9).round() as i128,
            );
            let exact = gcrf_to_itrf(ut1, tt, xp, yp, zero, zero, 0.0)
                .rotation
                .matrix()
                * r;
            worst = nan_max(worst, (exact - cio.rotation(t).matrix() * r).norm());
        }
        // Not "close to" the exact chain: the same arithmetic, so nanometres at LEO at worst.
        assert!(
            worst < 1e-9,
            "an uncovered query differs from the exact chain by {worst:.3e} m — it extrapolated"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // rotation_rows is a bit-for-bit copy of rotation().matrix().rows
    fn rows_and_transform_match_rotation_and_carry_the_earth_rate() {
        let epoch_tt = Epoch::<Tt>::from_ns_since_j2000(788_400_000_000_000_000);
        let epoch_ut1 = Epoch::<Ut1>::from_ns_since_j2000(788_400_000_000_000_000 - 69_000_000_000);
        let xp = Angle::from_arcseconds(0.16);
        let yp = Angle::from_arcseconds(0.35);
        let zero = Angle::from_radians(0.0);
        let span = 6.0 * 3600.0;
        let cio = CioInterpolator::new(epoch_ut1, epoch_tt, span, xp, yp, zero, zero, 0.0);

        let t = span * 0.37;
        // rotation_rows is exactly rotation().matrix().rows.
        let rows = cio.rotation_rows(t);
        assert_eq!(rows, cio.rotation(t).matrix().rows);

        // transform() carries the same rotation, and |omega| is the Earth rotation rate: a pure
        // rotation of (0, 0, rate), with lod = 0 here.
        let tf = cio.transform(t);
        assert_eq!(tf.rotation.matrix().rows, rows);
        assert!((tf.omega.norm() - EARTH_ROTATION_RATE).abs() <= 1e-12 * EARTH_ROTATION_RATE);

        // The full transform's rotation matches the exact chain to below a micron at LEO.
        let r = Vec3::new(6_800e3, 1_200e3, 900e3);
        let ut1 = Epoch::<Ut1>::from_ns_since_j2000(
            epoch_ut1.ns_since_j2000() + (t * 1e9).round() as i128,
        );
        let tt =
            Epoch::<Tt>::from_ns_since_j2000(epoch_tt.ns_since_j2000() + (t * 1e9).round() as i128);
        let exact = gcrf_to_itrf(ut1, tt, xp, yp, zero, zero, 0.0)
            .rotation
            .matrix()
            * r;
        let got = tf.rotation.matrix() * r;
        assert!((exact - got).norm() < 1e-6);
    }

    #[test]
    fn the_celestial_pole_offsets_reach_the_interpolated_stencil() {
        // The interpolator writes the `X + dX`, `Y + dY` correction out again for its own nodes.
        // Every test above passes zeros, where dropping the offsets, flipping their sign and
        // swapping them are all the same interpolator. The scalar chain is held against pyerfa at
        // these offsets in `tests/cio_oracle.rs`; here the fast path must reproduce it.
        let epoch_tt = Epoch::<Tt>::from_ns_since_j2000(788_400_000_000_000_000);
        let epoch_ut1 = Epoch::<Ut1>::from_ns_since_j2000(788_400_000_000_000_000 - 69_000_000_000);
        let xp = Angle::from_arcseconds(0.16);
        let yp = Angle::from_arcseconds(0.35);
        let zero = Angle::from_radians(0.0);
        // Asymmetric in size and opposite in sign, so a swapped pair is a different pole.
        let dx = Angle::from_arcseconds(12e-3);
        let dy = Angle::from_arcseconds(-7e-3);
        let span = 6.0 * 3600.0;
        let r = Vec3::new(6_800e3, 1_200e3, 900e3);

        let cio = CioInterpolator::new(epoch_ut1, epoch_tt, span, xp, yp, dx, dy, 0.0);
        let mut worst = 0.0_f64;
        let mut offsets_are_worth = 0.0_f64;
        for k in 0..=48 {
            let t = span * f64::from(k) / 48.0;
            let ns = (t * 1e9).round() as i128;
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1.ns_since_j2000() + ns);
            let tt = Epoch::<Tt>::from_ns_since_j2000(epoch_tt.ns_since_j2000() + ns);
            let exact = gcrf_to_itrf(ut1, tt, xp, yp, dx, dy, 0.0).rotation.matrix() * r;
            let zeroed = gcrf_to_itrf(ut1, tt, xp, yp, zero, zero, 0.0)
                .rotation
                .matrix()
                * r;
            // `rotation` and `transform` each write the `X + dX`, `Y + dY` correction out for
            // themselves, so checking one leaves the other free to drift. Both, every sample.
            worst = nan_max(worst, (exact - cio.rotation(t).matrix() * r).norm());
            worst = nan_max(
                worst,
                (exact - cio.transform(t).rotation.matrix() * r).norm(),
            );
            offsets_are_worth = nan_max(offsets_are_worth, (exact - zeroed).norm());
        }
        assert!(
            worst < 1e-6,
            "interpolated rotation differs from the exact chain by {worst:.3e} m with offsets"
        );
        // and that agreement is evidence only because the offsets move the position at all:
        // centimetres, four orders above the micron just asserted.
        assert!(
            offsets_are_worth > 0.01,
            "the offsets moved the position by {offsets_are_worth:.3e} m — too little for the \
             agreement above to show they were applied"
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // exact by construction: Lagrange on its own node
    fn the_grid_boundary_is_exact_and_every_node_reproduces_the_series() {
        // `covers` is the whole safety property: inside it a degree-7 polynomial is evaluated, and
        // one node past the last one that polynomial is an extrapolation. A boundary off by a
        // single node is invisible to an accuracy test — 600 s of extrapolation on a curve this
        // smooth is still sub-micron — so the boundary is pinned as an index, not as an error.
        let epoch_tt = Epoch::<Tt>::from_ns_since_j2000(788_400_000_000_000_000);
        let epoch_ut1 = Epoch::<Ut1>::from_ns_since_j2000(788_400_000_000_000_000 - 69_000_000_000);
        let zero = Angle::from_radians(0.0);
        let span = 6.0 * 3600.0;
        let cio = CioInterpolator::new(epoch_ut1, epoch_tt, span, zero, zero, zero, zero, 0.0);

        // The grid runs from `t_start` to the last node, both of which are a whole number of
        // `GRID_DT` steps from the construction span; nothing outside is covered.
        let t_first = cio.t_start;
        let t_last = cio.t_start + (cio.nodes.len() - 1) as f64 * GRID_DT;
        assert!(cio.covers(t_first), "the first node must be covered");
        assert!(cio.covers(t_last), "the last node must be covered");
        assert!(
            !cio.covers(t_last + GRID_DT * 1e-6),
            "a query past the last node must not be covered — that is extrapolation"
        );
        assert!(
            !cio.covers(t_first - GRID_DT * 1e-6),
            "a query before the first node must not be covered"
        );
        assert!(cio.covers(0.0) && cio.covers(span), "the arc itself");

        // On its own nodes the interpolator is not an approximation. A Lagrange basis evaluated at
        // node `j` is exactly 1 there and exactly 0 at every other node — the weights are products
        // of `(xq − k)/(j − k)` at integer `xq`, so one factor is exactly zero or every factor is
        // exactly one — and the node holds the series value. So an exact-node query must reproduce
        // the exact chain bit for bit, and does: measured, zero metres at all 53 nodes.
        //
        // That is also the only query that resolves *which* nodes the stencil picked. Away from a
        // node the CIP is smooth enough over 600 s that shifting the stencil by one node changes
        // the position by less than an ulp — measured, both give 2.0825e-9 m, identical. On a node
        // a stencil that has slid off it no longer contains it, and interpolation becomes
        // extrapolation. Sweeping every node also reaches the two at the ends, where the stencil
        // clamps and an over-long clamp bound indexes past the grid.
        let r = Vec3::new(6_800e3, 1_200e3, 900e3);
        for i in 0..cio.nodes.len() {
            let t = cio.t_start + i as f64 * GRID_DT;
            let ns = (t * 1e9).round() as i128;
            let ut1 = Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1.ns_since_j2000() + ns);
            let tt = Epoch::<Tt>::from_ns_since_j2000(epoch_tt.ns_since_j2000() + ns);
            let exact = gcrf_to_itrf(ut1, tt, zero, zero, zero, zero, 0.0)
                .rotation
                .matrix()
                * r;
            let gap = (exact - cio.rotation(t).matrix() * r).norm();
            assert!(
                gap == 0.0,
                "node {i} at t = {t} is off the exact chain by {gap:.3e} m — the stencil that \
                 interpolated it does not contain it"
            );
        }
    }

    #[test]
    fn the_interpolated_rate_carries_lod_exactly_as_the_scalar_chain_does() {
        // `transform` writes `AVE * (1 - LOD/86400)` out for itself. `tests/cio_oracle.rs` pins
        // that expression against Orekit 13.1.7's bytecode for the two public entry points; this
        // requires the third to be bit-for-bit the same, which is what makes it inherit the oracle.
        let epoch_tt = Epoch::<Tt>::from_ns_since_j2000(788_400_000_000_000_000);
        let epoch_ut1 = Epoch::<Ut1>::from_ns_since_j2000(788_400_000_000_000_000 - 69_000_000_000);
        let zero = Angle::from_radians(0.0);
        let span = 6.0 * 3600.0;

        for lod in [0.0, 0.0018, -0.0005, 0.0035] {
            let cio = CioInterpolator::new(epoch_ut1, epoch_tt, span, zero, zero, zero, zero, lod);
            let scalar = gcrf_to_itrf(epoch_ut1, epoch_tt, zero, zero, zero, zero, lod)
                .omega
                .norm();
            let interpolated = cio.transform(span * 0.37).omega.norm();
            assert!(
                (interpolated - scalar).abs() < 1e-20,
                "LOD {lod} s: interpolator |ω| = {interpolated:.20e}, scalar = {scalar:.20e}"
            );
            // and LOD is not being ignored: 1.8 ms is 1.5e-12 rad/s below the nominal rate.
            if lod > 0.0 {
                assert!(
                    EARTH_ROTATION_RATE - interpolated > 0.5e-12,
                    "a positive LOD must slow the interpolated rate"
                );
            }
        }
    }
}
