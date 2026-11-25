//! Repeat ground tracks: the exact gaps between equator crossings after any number of revolutions,
//! the sub-cycles, and whether a swath closes the equator (plan 18 Ф27.D).
//!
//! **The crossings.** A circular orbit whose ground track repeats after `Np` nodal revolutions in
//! `Nd` nodal days has the rate ratio `κ = (ω⊕ − Ω̇)/u̇ = Nd/Np`: `Np` periods of the argument of
//! latitude `u` last as long as `Nd` turns of the Earth under the node. A *slot* is `2π/Np` of
//! longitude and a *half-slot* `π/Np`. Over one revolution the Earth turns `2πNd/Np` under the node,
//! so the ascending node of revolution `k` lies `−k·Nd` slots east of the first. Over half a
//! revolution the orbit carries the crossing `π` east and the Earth turns `πNd/Np` under it, so the
//! descending node lies `Np − Nd` half-slots east of its ascending node, and the `j`-th equator
//! crossing of either kind, in time order, lies `j·(Np − Nd)` half-slots east of the first, modulo
//! `2Np` (`derivations/repeat_track_gaps.mac`). With `gcd(Np, Nd) = g > 1` the track is that of
//! `(Np/g)/(Nd/g)`, whose cycle visits `Np/g` of the `Np` slots: 143/11 visits 13 of its 143, so such a
//! pair is refused ([`RepeatTrackError::NotCoprime`]).
//!
//! **Three gaps.** Each set of crossings is the first `N` points `r_j = j·a mod q` of a rotation on a
//! circle of `q` cells with `gcd(a, q) = 1`: the ascending nodes after `n` revolutions have
//! `a = −Nd mod Np`, `q = Np`, `N = min(n, Np)`; both kinds have `a = (Np − Nd)/c` and `q = 2Np/c`,
//! cells of `c = gcd(Np − Nd, 2Np)` half-slots, and `N = min(2n, q)`. `c` is 2 when `Np` and `Nd` are
//! both odd, and the descending nodes then fall on ascending ones. The gaps follow from two indices
//! (the three-distance theorem, re-derived here; no source is cited). For `N ≥ 2` let `u` be the
//! `j ∈ [1, N − 1]` with the smallest `r_j` and `v` the one with the largest, and `A = r_u`,
//! `B = q − r_v`.
//!
//! 1. `u + v ≥ N`. Otherwise `j = u + v ≤ N − 1` has `r_j ≡ A − B (mod q)`: below `A` if `A > B`,
//!    against the choice of `u`; above `r_v` if `A < B`, against the choice of `v`; and `A = B` would
//!    put `q` in `u + v < q`.
//! 2. Going up the circle from `r_k`, the next point is `r_{k+u}`, `A` away, when `k < N − u`;
//!    `r_{k−v}`, `B` away, when `k ≥ v`; and `r_{k+u−v}`, `A + B` away, when `N − u ≤ k < v`. By (1)
//!    the cases exclude each other. A nearer point `r_{k±d}` would give, with `u` or `v`, an index of
//!    `[1, N − 1]` whose residue beats `A` or `B`.
//!
//! So there are `N − u` gaps of `A`, `N − v` of `B` and `u + v − N` of `A + B`: at most three lengths,
//! and with three the largest is the sum of the other two. `A = B` exactly when `u + v = q`, and the
//! two shorter lengths merge. `derivations/repeat_track_gaps.gp` holds the formula equal to the sorted
//! crossings for every `Np ≤ 120`, every `Nd < 2Np` prime to it, every `N` and both kinds of node;
//! `every_gap_set_is_the_sorted_crossings` does the same in `tests/repeat_track.rs` for `Np ≤ 60`.
//!
//! **Sub-cycles.** A sub-cycle is a revolution `k` whose ascending node comes nearer the first one
//! than the node of every earlier revolution: `|o_k| < |o_j|` for `j < k`, where `o_k ∈ (−Np/2, Np/2]`
//! is the slot offset `−k·Nd` reduced mod `Np`. The last before `Np` is one slot away, the classic
//! sub-cycle (233/16: revolution 102, 7.004 days); the cycle itself, `k = Np`, ends the list with
//! offset 0. [`RepeatTrack::sub_cycles`] scans the definition. The records are the distinct
//! denominators of the convergents of `Nd/Np`, which the `.gp` checks for every `Np ≤ 300`.
//!
//! **The swath rule.** Near a crossing the track is taken as its tangent great circle, which meets
//! the equator at the apparent inclination `i′`. In the Earth-fixed frame the sub-satellite point
//! moves north at `u̇ sin i` and east at `u̇ cos i − (ω⊕ − Ω̇)` at both nodes, so
//! `sin i′ = sin i / √(sin²i + (cos i − κ)²)`. The point of the equator at arc `x` from the crossing is
//! at `asin(sin x · sin i′)` from that circle, so a swath of ground width `W` on a sphere of radius `R`
//! covers `|x| ≤ asin(sin(W/2R)/sin i′)`, whose limit as `R → ∞` is `W/(2 sin i′)`. The equator is
//! covered when its largest gap `G(n)·2πR/Np` (`G` in slots) is at most `2R·asin(sin(W/2R)/sin i′)`,
//! with the planar limit `W/sin i′`. The rule holds where the tangent circle stands for the track, so
//! `sin(W/2R) < sin i′` is required ([`RepeatTrackError::SwathBeyondTrack`]).
//!
//! The decision compares `sin(x_g)·sin i` with `sin(W/2R)·√(sin²i + (cos i − κ)²)`, `x_g` half the
//! largest gap, so no inverse function is in it. It accepts only with a first-order allowance for
//! rounding, under H-libm and not outward-rounded ([`RepeatTrack::LABEL`]): a gap within that
//! allowance of the swath is reported open. So [`RepeatTrack::equator_covered`] errs towards "not
//! covered", and [`RepeatTrack::revolutions_to_cover_equator`] towards more revolutions.
//!
//! The gap sets and the sub-cycles are integer arithmetic and exact. The cost is `O(N)` per gap set
//! and `O(Np)` for the sub-cycles and for the first covering revolution. No allocation: `no_std`.

const PI: f64 = core::f64::consts::PI;
/// The unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;

/// A repeat ground track of `Np` nodal revolutions in `Nd` nodal days, `gcd(Np, Nd) = 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepeatTrack {
    revolutions: u32,
    days: u32,
}

/// Which equator crossings a gap set counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nodes {
    /// The ascending nodes only: one crossing per revolution.
    Ascending,
    /// The ascending and the descending nodes: two crossings per revolution.
    Both,
}

/// Gaps of one length between consecutive equator crossings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gap {
    /// The length, half-slots: one half-slot is `π/Np` of longitude.
    pub length: u64,
    /// How many gaps have that length.
    pub count: u64,
}

/// The gaps between consecutive equator crossings: one, two or three distinct lengths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GapSet {
    gaps: [Gap; 3],
    distinct: usize,
}

impl GapSet {
    /// The distinct lengths in increasing order, each with its count.
    #[must_use]
    pub fn gaps(&self) -> &[Gap] {
        &self.gaps[..self.distinct]
    }

    /// The largest gap, half-slots.
    #[must_use]
    pub fn largest(&self) -> u64 {
        self.gaps[self.distinct - 1].length
    }

    /// The number of distinct crossings, which is the number of gaps.
    #[must_use]
    pub fn crossings(&self) -> u64 {
        self.gaps().iter().map(|g| g.count).sum()
    }
}

/// A revolution whose ascending node comes nearer the first than every earlier one (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubCycle {
    /// The revolution `k`; it ends after `k·Nd/Np` nodal days.
    pub revolutions: u32,
    /// The node's offset from the first, slots east, in `(−Np/2, Np/2]`.
    pub offset: i64,
}

/// The sub-cycles of a track in increasing order, ending with the whole cycle (offset 0).
#[derive(Clone, Debug)]
pub struct SubCycles {
    cells: u64,
    step: u64,
    k: u64,
    residue: u64,
    best: u64,
}

impl Iterator for SubCycles {
    type Item = SubCycle;

    fn next(&mut self) -> Option<SubCycle> {
        while self.k < self.cells {
            self.k += 1;
            self.residue = add_mod(self.residue, self.step, self.cells);
            let distance = self.residue.min(self.cells - self.residue);
            if distance < self.best {
                self.best = distance;
                let offset = if 2 * self.residue <= self.cells {
                    to_i64(self.residue)
                } else {
                    to_i64(self.residue) - to_i64(self.cells)
                };
                return Some(SubCycle {
                    revolutions: u32::try_from(self.k).unwrap_or(u32::MAX),
                    offset,
                });
            }
        }
        None
    }
}

/// A swath centred on the track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Swath {
    /// The swath's ground width `W`, metres, measured across the track on the sphere.
    pub width: f64,
    /// The sphere's radius `R`, metres.
    pub radius: f64,
    /// The orbit's inclination `i`, radians, in `(0, π)`.
    pub inclination: f64,
}

/// Why a repeat track or its gaps were not computed: input outside the domain is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepeatTrackError {
    /// `Np` or `Nd` is zero.
    ZeroCount,
    /// `gcd(Np, Nd) > 1`: the track is that of the reduced pair and visits `Np/gcd` of its slots.
    NotCoprime {
        /// The common factor.
        gcd: u32,
    },
    /// Zero revolutions were asked for: no crossing, no gap.
    NoRevolutions,
    /// The inclination is not finite or not in `(0, π)`; at 0 and `π` the track runs along the
    /// equator and `sin i′ = 0`.
    BadInclination,
    /// The swath width or the radius is not finite and positive.
    BadSwath,
    /// `sin(W/2R) ≥ sin i′`: the swath reaches the tangent circle's farthest point from the equator,
    /// where the rule no longer stands for the track.
    SwathBeyondTrack,
}

impl core::fmt::Display for RepeatTrackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroCount => write!(f, "a repeat track needs Np >= 1 and Nd >= 1"),
            Self::NotCoprime { gcd } => {
                write!(f, "Np and Nd share the factor {gcd}: reduce the pair")
            }
            Self::NoRevolutions => write!(f, "zero revolutions have no crossings"),
            Self::BadInclination => write!(f, "the inclination is not finite and in (0, pi)"),
            Self::BadSwath => write!(
                f,
                "the swath width or the radius is not finite and positive"
            ),
            Self::SwathBeyondTrack => {
                write!(
                    f,
                    "sin(W/2R) >= sin i': the swath reaches past the track's tangent circle"
                )
            }
        }
    }
}

impl core::error::Error for RepeatTrackError {}

impl RepeatTrack {
    /// What the swath decision rests on: the `libm` elementary functions within their measured error
    /// (H-libm), and round-to-nearest arithmetic with a first-order allowance rather than outward
    /// rounding. The gap sets and the sub-cycles are exact.
    pub const LABEL: &'static str = "under H-libm, not outward-rounded";

    /// The track of `revolutions` (`Np`) nodal revolutions in `days` (`Nd`) nodal days.
    ///
    /// # Errors
    /// [`RepeatTrackError::ZeroCount`] when either is zero, [`RepeatTrackError::NotCoprime`] when they
    /// share a factor.
    pub fn new(revolutions: u32, days: u32) -> Result<Self, RepeatTrackError> {
        if revolutions == 0 || days == 0 {
            return Err(RepeatTrackError::ZeroCount);
        }
        let g = gcd(u64::from(revolutions), u64::from(days));
        if g != 1 {
            return Err(RepeatTrackError::NotCoprime {
                gcd: u32::try_from(g).unwrap_or(u32::MAX),
            });
        }
        Ok(Self { revolutions, days })
    }

    /// `Np`, the revolutions of one cycle.
    #[must_use]
    pub const fn revolutions(self) -> u32 {
        self.revolutions
    }

    /// `Nd`, the nodal days of one cycle.
    #[must_use]
    pub const fn days(self) -> u32 {
        self.days
    }

    /// The rate ratio `κ = (ω⊕ − Ω̇)/u̇ = Nd/Np`, rounded once.
    #[must_use]
    pub fn rate_ratio(self) -> f64 {
        f64::from(self.days) / f64::from(self.revolutions)
    }

    /// The gaps between consecutive equator crossings after `revolutions` revolutions (module docs).
    /// From `Np` revolutions on, the set is the whole cycle's.
    ///
    /// # Errors
    /// [`RepeatTrackError::NoRevolutions`] for zero revolutions.
    pub fn gaps(self, revolutions: u32, nodes: Nodes) -> Result<GapSet, RepeatTrackError> {
        if revolutions == 0 {
            return Err(RepeatTrackError::NoRevolutions);
        }
        let mut rotation = self.rotation(nodes);
        let points = rotation.points_after(revolutions);
        while rotation.points < points {
            rotation.advance();
        }
        Ok(rotation.gap_set())
    }

    /// The sub-cycles: each revolution whose ascending node comes nearer the first than every earlier
    /// one, ending with the whole cycle (module docs).
    #[must_use]
    pub fn sub_cycles(self) -> SubCycles {
        let cells = u64::from(self.revolutions);
        SubCycles {
            cells,
            step: (cells - u64::from(self.days) % cells) % cells,
            k: 0,
            residue: 0,
            best: cells,
        }
    }

    /// The equator a swath covers on each side of a crossing, both sides together:
    /// `2R·asin(sin(W/2R)/sin i′)`, metres. For reporting; the decision does not use it.
    ///
    /// # Errors
    /// [`RepeatTrackError::BadInclination`], [`RepeatTrackError::BadSwath`] and
    /// [`RepeatTrackError::SwathBeyondTrack`].
    pub fn covered_equator(self, swath: &Swath) -> Result<f64, RepeatTrackError> {
        let rule = SwathRule::new(self, swath)?;
        Ok(2.0 * swath.radius * libm::asin(rule.sin_beta * rule.hypot / rule.sin_i))
    }

    /// Whether the swaths of the first `revolutions` revolutions cover the whole equator, on the safe
    /// side: `false` also when the largest gap is within the rounding allowance of the swath.
    ///
    /// # Errors
    /// [`RepeatTrackError::NoRevolutions`], and the swath's errors as for
    /// [`RepeatTrack::covered_equator`].
    pub fn equator_covered(
        self,
        revolutions: u32,
        nodes: Nodes,
        swath: &Swath,
    ) -> Result<bool, RepeatTrackError> {
        let rule = SwathRule::new(self, swath)?;
        let gaps = self.gaps(revolutions, nodes)?;
        Ok(rule.covers(gaps.largest()))
    }

    /// The first number of revolutions whose swaths cover the whole equator, on the safe side (never
    /// fewer than the exact answer), or `None` when a whole cycle leaves a gap.
    ///
    /// # Errors
    /// The swath's errors, as for [`RepeatTrack::covered_equator`].
    pub fn revolutions_to_cover_equator(
        self,
        nodes: Nodes,
        swath: &Swath,
    ) -> Result<Option<u32>, RepeatTrackError> {
        let rule = SwathRule::new(self, swath)?;
        let mut rotation = self.rotation(nodes);
        for n in 1..=self.revolutions {
            let points = rotation.points_after(n);
            while rotation.points < points {
                rotation.advance();
            }
            if rule.covers(rotation.largest()) {
                return Ok(Some(n));
            }
        }
        Ok(None)
    }

    /// The rotation whose first points are the crossings of `nodes`.
    fn rotation(self, nodes: Nodes) -> Rotation {
        let np = u64::from(self.revolutions);
        let nd = u64::from(self.days);
        match nodes {
            Nodes::Ascending => Rotation::new((np - nd % np) % np, np, 2, 1),
            Nodes::Both => {
                let circle = 2 * np;
                let step = (np + circle - nd % circle) % circle;
                let c = gcd(step, circle);
                Rotation::new(step / c, circle / c, c, 2)
            }
        }
    }
}

/// The first points `r_j = j·step mod cells` of a rotation, with the indices of the smallest and the
/// largest `r_j` past the first (module docs).
struct Rotation {
    step: u64,
    cells: u64,
    /// Half-slots per cell.
    scale: u64,
    /// Crossings per revolution.
    per_revolution: u64,
    points: u64,
    residue: u64,
    low: (u64, u64),
    high: (u64, u64),
}

impl Rotation {
    const fn new(step: u64, cells: u64, scale: u64, per_revolution: u64) -> Self {
        Self {
            step,
            cells,
            scale,
            per_revolution,
            points: 1,
            residue: 0,
            low: (0, 0),
            high: (0, 0),
        }
    }

    /// The distinct points after `n` revolutions.
    fn points_after(&self, n: u32) -> u64 {
        (u64::from(n) * self.per_revolution).min(self.cells)
    }

    /// Add the next point; `low` and `high` keep `(j, r_j)` of the smallest and the largest residue.
    fn advance(&mut self) {
        let j = self.points;
        self.residue = add_mod(self.residue, self.step, self.cells);
        if j == 1 || self.residue < self.low.1 {
            self.low = (j, self.residue);
        }
        if j == 1 || self.residue > self.high.1 {
            self.high = (j, self.residue);
        }
        self.points += 1;
    }

    /// The largest gap, half-slots.
    fn largest(&self) -> u64 {
        if self.points == 1 {
            return self.cells * self.scale;
        }
        let (u, a) = self.low;
        let (v, high) = self.high;
        let b = self.cells - high;
        let cells = if u + v > self.points { a + b } else { a.max(b) };
        cells * self.scale
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "n, u, v, a and b are the names of the three-distance theorem in the module docs"
    )]
    fn gap_set(&self) -> GapSet {
        let empty = Gap {
            length: 0,
            count: 0,
        };
        let mut gaps = [empty; 3];
        if self.points == 1 {
            gaps[0] = Gap {
                length: self.cells * self.scale,
                count: 1,
            };
            return GapSet { gaps, distinct: 1 };
        }
        let n = self.points;
        let (u, a) = self.low;
        let (v, high) = self.high;
        let b = self.cells - high;
        let (short, long) = if a <= b {
            ((a, n - u), (b, n - v))
        } else {
            ((b, n - v), (a, n - u))
        };
        let mut distinct = 0;
        let mut push = |length: u64, count: u64| {
            if count == 0 {
                return;
            }
            let length = length * self.scale;
            if distinct > 0 && gaps[distinct - 1].length == length {
                gaps[distinct - 1].count += count;
            } else {
                gaps[distinct] = Gap { length, count };
                distinct += 1;
            }
        };
        push(short.0, short.1);
        push(long.0, long.1);
        push(a + b, u + v - n);
        GapSet { gaps, distinct }
    }
}

/// The swath rule's three factors, and the allowance the decision keeps from the boundary.
struct SwathRule {
    np: u64,
    sin_i: f64,
    hypot: f64,
    sin_beta: f64,
    allowance: f64,
}

impl SwathRule {
    fn new(track: RepeatTrack, swath: &Swath) -> Result<Self, RepeatTrackError> {
        let i = swath.inclination;
        if !(i.is_finite() && i > 0.0 && i < PI) {
            return Err(RepeatTrackError::BadInclination);
        }
        let (w, r) = (swath.width, swath.radius);
        if !(w.is_finite() && w > 0.0 && r.is_finite() && r > 0.0) {
            return Err(RepeatTrackError::BadSwath);
        }
        let beta = w / (2.0 * r);
        let (sin_i, cos_i) = libm::sincos(i);
        let kappa = track.rate_ratio();
        let hypot = libm::hypot(sin_i, cos_i - kappa);
        let sin_beta = libm::sin(beta.min(PI / 2.0));
        // sin(W/2R) < sin i′ = sin i / hypot, without the division.
        if !(beta < PI / 2.0 && sin_beta * hypot < sin_i) {
            return Err(RepeatTrackError::SwathBeyondTrack);
        }
        // First order: a few roundings on each side, and cos i − κ's absolute error passed through
        // the hypotenuse, at most (|cos i| + κ)·u relative to it per rounding.
        let allowance = 32.0 * U + 8.0 * U * (cos_i.abs() + kappa) / hypot;
        Ok(Self {
            np: u64::from(track.revolutions),
            sin_i,
            hypot,
            sin_beta,
            allowance,
        })
    }

    /// Whether a gap of `largest` half-slots lies inside the swaths on its two sides, on the safe
    /// side of the allowance.
    fn covers(&self, largest: u64) -> bool {
        // Half the gap, x_g = largest·π/(2Np), is above π/2 when the gap is above half the equator,
        // and the tangent circle covers at most π/2 on each side.
        if largest > self.np {
            return false;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "a gap and Np are at most 2^33, exact in f64"
        )]
        let half_gap = largest as f64 * PI / (2.0 * self.np as f64);
        let lhs = libm::sin(half_gap) * self.sin_i;
        let rhs = self.sin_beta * self.hypot;
        lhs <= rhs * (1.0 - self.allowance)
    }
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// `(r + step) mod cells` for `r, step < cells ≤ 2^33`, so the sum does not overflow.
const fn add_mod(r: u64, step: u64, cells: u64) -> u64 {
    let s = r + step;
    if s >= cells { s - cells } else { s }
}

fn to_i64(x: u64) -> i64 {
    i64::try_from(x).unwrap_or(i64::MAX)
}
