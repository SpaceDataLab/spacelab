//! Lambert's problem: the transfer arc between two positions in a given time.
//!
//! Given `r1`, `r2` and a time of flight, find the two-body velocities that connect them. This is the
//! core transfer primitive — the arc of a Mars or lunar injection, the leg of a porkchop plot. The
//! universal-variable formulation (Bate–Mueller–White; Curtis, *Orbital Mechanics for Engineering
//! Students*, Algorithm 5.2) reuses the Stumpff functions of [`crate::kepler`] and covers the single
//! revolution across ellipse and hyperbola through one iteration on the universal variable `z`.
//! [`lambert_multirev`] solves for `N ≥ 0` complete revolutions in Izzo's variable `x` (Izzo 2015,
//! the Lancaster–Blanchard form of Lagrange's equation), on either branch of an `N`-revolution
//! family, and refuses a time of flight below the family's minimum with
//! [`LambertError::BelowMinimumTime`].
//!
//! Gate: `derivations/lambert_universal.mac` shows that the coded `y(z)` is Curtis's quotient,
//! that `dfdz` is `dF/dz` with its `z → 0` branch the limit, and that the arc `(r1, v1)` built from
//! the root has `α χ² = z` and satisfies the universal Kepler equation from `r1` to `r2` — the
//! solution is the two-body boundary-value problem's — with `f ġ − ḟ g = 1`. It also builds an
//! ellipse and a hyperbola in closed form, and derives the forms of `A` and `y` that do not cancel
//! near a half turn and on short arcs. `derivations/lambert_universal.gp` solves the test sweep
//! at 100 digits by Lagrange's theorem instead and closes every arc by propagation. The module tests
//! `the_arc_is_the_one_lagrange_s_theorem_gives_at_100_digits` (G2) and
//! `the_arc_is_the_exact_conic_it_was_built_from` (G3) hold both solvers to them.
//!
//! `derivations/lambert_izzo.mac` derives the multi-revolution solver's `λ` without `1 − c/s`, its
//! time of flight in Stumpff form with the parabolic limit `⅔(1 − λ³)` at `x = 1`, and Izzo's
//! `dT/dx`. `derivations/lambert_izzo.gp` solves Lagrange's equation at 100 digits on each branch,
//! finds each family's minimum time two ways, and closes every arc by propagation. The module tests
//! `every_multi_revolution_arc_is_the_one_lagrange_s_equation_gives_at_100_digits` (G2),
//! `the_time_of_flight_at_x_equal_one_is_the_parabolic_time` (G3) and
//! `a_time_below_the_family_s_minimum_is_refused_and_one_above_is_solved` (S2) hold it to them.

// Single-letter names are the standard notation of the universal-variable formulation.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]

use spacelab_core::Vec3;

use crate::kepler::{propagate, stumpff_c, stumpff_s};

/// `z = 4π²` closes one revolution: the open upper end of the domain this formulation solves.
const REVOLUTION: f64 = 4.0 * core::f64::consts::PI * core::f64::consts::PI;

/// Rungs of the hyperbolic search, which doubles from `z = −1`: the deepest probe is `−2²⁶ ≈
/// −6.7e7`, past which the arc is faster than the formulation carries and [`Arc::bracket`] gives up.
///
/// The three ladders below are counted rather than bounded by the value they walk toward, and that
/// is deliberate. Written as `while lo > NEAR_ZERO { lo *= 0.5 }` the loop terminates because of
/// what `*= 0.5` does, not because of anything the loop itself guarantees, and the mutation seal
/// reports exactly that: `*=` → `+=` and `*=` → `/=` at these three sites, and the same shape in
/// [`crate::iod::gooding`], produced six 180-second timeouts and left orphaned test binaries at
/// 100 % of a core (#292). A counted loop cannot be mutated into a non-terminating one.
const HYPERBOLIC_RUNGS: usize = 27;

/// Rungs of the halving search toward zero, from `z = 0.5`: the smallest probe is `2⁻⁵⁹ ≈ 1.7e-18`,
/// and `below(0)` is true, so zero itself closes the bracket when none of them does.
const HALVING_RUNGS: usize = 59;

/// Rungs of the doubling ladder from `z = 2` to the ceiling: `4π² < 2⁶`, so six probes reach it.
const DOUBLING_RUNGS: usize = 8;

/// How far the returned arc may miss `r2`, as a fraction of `|r2|`, before [`lambert`] refuses it.
///
/// A converged root is not the same thing as a transfer. As the transfer angle approaches a whole
/// revolution the root migrates to `z = 4π²`, and there `z` stops being a usable coordinate: the
/// doubles near `4π²` are 7.1e-15 apart, so `4π² − z` — which is what the arc actually depends on —
/// is resolved to about seven significant figures however exactly the terms are evaluated. The
/// answer degrades continuously rather than failing, which is why the residual on `z` cannot detect
/// it and a closure test can.
///
/// One part in 1e6 of the radius is a metre on a low Earth orbit, and the gate is load-bearing at
/// that level: over the 25 264-case sweep of `examples/lambert_longway_sweep.rs` the solver without
/// it answers 8 near-whole-revolution arcs with a miss above the tolerance, and with it refuses
/// those 8 and returns nothing worse than 6.0e-7. It was 43 of 25 270 and 9.3e-7 while the chord
/// parameter was `sin Δν·√(r1 r2/(1 − cos Δν))`, which near a whole revolution carried `cos Δν`'s
/// rounding up by `1/(1 − cos Δν)`; the six cases the sweep no longer considers are those whose
/// [`lambert_multirev`] reference itself closes only at the example's 1e-9 threshold. Callers who
/// need the refused arcs should use [`lambert_multirev`], whose Lancaster–Blanchard variable does
/// not crowd at one revolution.
const CLOSURE_TOLERANCE: f64 = 1e-6;

/// Solve Lambert's problem for the departure and arrival velocities `(v1, v2)`.
///
/// `prograde` selects the transfer sense (the short/long way is then fixed by the geometry). Units
/// are the caller's, consistent. `tof` must be finite and positive. Returns `None` for any other
/// time of flight, when the transfer angle is `0` or `π` (the transfer plane is undefined — an
/// invariant of the single-revolution universal-variable method), if the iteration does not
/// converge, or if the arc it converged on does not reach `r2` — see `CLOSURE_TOLERANCE`.
#[must_use]
pub fn lambert(r1: Vec3, r2: Vec3, tof: f64, mu: f64, prograde: bool) -> Option<(Vec3, Vec3)> {
    // A non-positive `tof` has no root to bracket: the residual is then `√µ·t(z)` plus a
    // non-negative constant, so it never turns, and the search runs down to the `y = 0` edge and
    // bisects onto a degenerate near-rectilinear arc the caller cannot tell from a transfer. Measured before this guard over 2000 heliocentric geometries, 1284 of the 4000
    // (geometry, sense) pairs came back with a finite solution at `tof = 0`, |v1| near 1e12 m/s.
    // `lambert_multirev` refuses these already.
    if !(tof.is_finite() && tof > 0.0) {
        return None;
    }
    let arc = Arc::new(r1, r2, tof, mu, prograde)?;
    let (r1n, r2n) = (arc.r1n, arc.r2n);
    let (z, _) = arc.solve()?;

    let yz = arc.y(z);
    if !yz.is_finite() || yz <= 0.0 {
        return None;
    }
    let lf = 1.0 - yz / r1n;
    let lg = arc.a * libm::sqrt(yz / mu);
    let lgd = 1.0 - yz / r2n;
    let v1 = Vec3::new(
        (r2.x - lf * r1.x) / lg,
        (r2.y - lf * r1.y) / lg,
        (r2.z - lf * r1.z) / lg,
    );
    let v2 = Vec3::new(
        (lgd * r2.x - r1.x) / lg,
        (lgd * r2.y - r1.y) / lg,
        (lgd * r2.z - r1.z) / lg,
    );

    // Lambert's problem is Kepler's read backwards, so the answer is checked by reading it forwards:
    // depart `r1` with `v1` and require the arc to arrive at `r2`. This is not the residual on `z`
    // again — that one is a statement about the transfer *time*, and near a whole revolution the
    // position is far more sensitive to `z` than the time is, so a root can satisfy it to a
    // nanosecond and still be 87 km wide. `propagate` shares the Stumpff pair but not the iteration:
    // it solves for `χ` at fixed `dt`, where this solved for `z` at fixed endpoints.
    let (arrival, _) = propagate(r1, v1, mu, tof).ok()?;
    let miss = (arrival - r2).norm();
    if !(miss.is_finite() && miss <= CLOSURE_TOLERANCE * r2n) {
        return None;
    }
    Some((v1, v2))
}

/// `|A| = √(r1 r2 + r1·r2) = √(2 r1 r2)·|cos(Δν/2)|`, the magnitude of Lambert's chord parameter,
/// from the radii, `r1·r2` and `|r1 × r2|`.
///
/// Curtis writes `A = sin Δν · √(r1 r2/(1 − cos Δν))`, and with `cos Δν` from a dot product that is
/// `r1 r2 + r1·r2` formed as a difference of near-equal numbers within `ε` of a half turn: it lost
/// `u/ε²`, 1.2e-6 relative at `ε = 1e-5`, and [`CLOSURE_TOLERANCE`] then refused every arc within
/// about 1e-5 rad of 180°. Past a quarter turn the same quantity is `|r1 × r2|²/(r1 r2 − r1·r2)`
/// (Lagrange's identity), whose denominator adds like-signed terms, so the error left is the `u/ε`
/// with which the cross product of two near-antiparallel vectors carries their doubles.
/// `derivations/lambert_universal.mac` 31–35 and `derivations/lambert_izzo.mac` 1–2.
fn chord_parameter_magnitude(r1n: f64, r2n: f64, dot: f64, cross_n: f64) -> f64 {
    if dot < 0.0 {
        cross_n / libm::sqrt(r1n * r2n - dot)
    } else {
        libm::sqrt(r1n * r2n + dot)
    }
}

/// What the difference form of [`Arc::y`] needs on a short arc, `r1·r2 > 0` the short way round:
/// `(√r1 − √r2)²`, `√(r1 r2)` and the half transfer angle `h = Δν/2`.
#[derive(Clone, Copy)]
struct ShortArc {
    gap: f64,
    mean: f64,
    half: f64,
}

/// A single-revolution arc as the universal-variable iteration sees it: the two radii, the chord
/// parameter `A = √(2 r1 r2)·cos(Δν/2)`, the short-arc terms, `√µ`, and the time of flight to match.
///
/// The three functions below share all of them, which is why they are a type rather than three
/// signatures. They were closures inside [`lambert`] and could only be reached through it — and
/// [`Self::dfdz`] cannot be reached through it at all, since the iteration runs to a fixed point and
/// the derivative decides only how fast `z` arrives, never where.
#[derive(Clone, Copy)]
struct Arc {
    r1n: f64,
    r2n: f64,
    a: f64,
    short: Option<ShortArc>,
    sqmu: f64,
    tof: f64,
}

impl Arc {
    /// The arc [`lambert`] iterates on, or `None` where the transfer plane is undefined.
    ///
    /// `A` is [`chord_parameter_magnitude`] with the sign of `cos(Δν/2)`: positive the short way
    /// round, negative past `Δν = π`, which is the side `prograde` and the sense of `r1 × r2` put
    /// the transfer on. Nothing here takes `Δν` itself, so nothing here goes through `acos`, which
    /// returned exactly `0` or `π` within 1.49e-8 rad of either and put a band of that width around
    /// both degenerate angles. The refusal is now what it says: `|sin Δν| = |r1 × r2|/(r1 r2)` at
    /// most 1e-12, where `A` and the `g` it divides by are zero.
    fn new(r1: Vec3, r2: Vec3, tof: f64, mu: f64, prograde: bool) -> Option<Self> {
        let r1n = r1.norm();
        let r2n = r2.norm();
        let cross = r1.cross(r2);
        let cross_n = cross.norm();
        if cross_n.is_nan() || cross_n <= 1e-12 * r1n * r2n {
            return None;
        }
        let dot = r1.dot(r2);
        let long_way = if prograde {
            cross.z < 0.0
        } else {
            cross.z >= 0.0
        };
        let magnitude = chord_parameter_magnitude(r1n, r2n, dot, cross_n);
        // The radii's difference through the vectors carries the chord's relative error, where
        // `r1n − r2n` would carry `u (r1 + r2)/(r1 − r2)`.
        let short = (!long_way && dot > 0.0).then(|| {
            let diff = (r1 - r2).dot(r1 + r2) / (r1n + r2n);
            let gap = diff / (libm::sqrt(r1n) + libm::sqrt(r2n));
            ShortArc {
                gap: gap * gap,
                mean: libm::sqrt(r1n * r2n),
                half: 0.5 * libm::atan2(cross_n, dot),
            }
        });
        Some(Self {
            r1n,
            r2n,
            a: if long_way { -magnitude } else { magnitude },
            short,
            sqmu: libm::sqrt(mu),
            tof,
        })
    }

    /// `y(z)`, the chord-normal distance the arc reaches (Curtis Eq. 5.38).
    ///
    /// Curtis writes it as `r1 + r2 + A(zS(z) − 1)/√C(z)`, and both halves of that quotient vanish
    /// at `z = 4π²` — the whole-revolution end of the domain, which is exactly where a long-way
    /// transfer puts its root. Neither is a genuine singularity: for `z > 0`,
    ///
    /// ```text
    /// zS(z) − 1 = −sin√z/√z          C(z) = 2 sin²(√z/2)/z
    /// ```
    ///
    /// are both identities, and `√z/2 ∈ (0, π)` keeps `sin(√z/2) > 0`, so the quotient is
    /// `−√2 cos(√z/2)` with the half-angle `sin√z = 2 sin(√z/2)cos(√z/2)` cancelled analytically.
    /// The hyperbolic branch runs the same way through `sinh` and lands on `cosh`. What is left has
    /// no subtraction of near-equal quantities anywhere in it and is exact across the whole domain,
    /// including the `y = 0` edge that decides where the search may go.
    ///
    /// Evaluated through the Stumpff pair instead, the same expression is a `0/0` in which `C` is
    /// 1.25e-19 a part in 1e9 below `4π²`, and what the quotient loses there the search cannot get
    /// back: holding everything else fixed, `examples/lambert_longway_sweep.rs` refused 227 of its
    /// 25 270 arcs with the Curtis quotient and 111 with the form below; with the chord parameter
    /// in the form of `chord_parameter_magnitude` it refuses 76 of 25 264.
    ///
    /// On a short arc that form cancels in its turn: with `A = √(2 r1 r2)·cos h`, `h = Δν/2`, it is
    /// `r1 + r2 − 2√(r1 r2)·cos h·cos ψ`, `ψ = √z/2`, two terms that agree to `O(Δν²)` near the
    /// root, and `v1` lost 1.7e-6 at `Δν = 1e-5`. Written out,
    ///
    /// ```text
    /// y = (√r1 − √r2)² + 2√(r1 r2)·[sin²((h − ψ)/2) + sin²((h + ψ)/2)]
    /// ```
    ///
    /// is a sum of non-negative terms (`derivations/lambert_universal.mac` 36–38). It is used where
    /// the cancellation lives — the short way with `r1·r2 > 0`, so `h < π/4`, and `z ≥ 0`.
    fn y(self, z: f64) -> f64 {
        if let Some(short) = self.short
            && z >= 0.0
        {
            let psi = 0.5 * libm::sqrt(z);
            let below = libm::sin(0.5 * (short.half - psi));
            let above = libm::sin(0.5 * (short.half + psi));
            return short.gap + 2.0 * short.mean * (below * below + above * above);
        }
        let half = if z > 0.0 {
            libm::cos(0.5 * libm::sqrt(z))
        } else {
            libm::cosh(0.5 * libm::sqrt(-z))
        };
        self.r1n + self.r2n - core::f64::consts::SQRT_2 * self.a * half
    }

    /// `F(z) = √µ · (t(z) − tof)`, the residual the iteration drives to zero (Curtis Eq. 5.40).
    fn tof_err(self, z: f64) -> f64 {
        let c = stumpff_c(z);
        let s = stumpff_s(z);
        let yz = self.y(z);
        libm::pow(yz / c, 1.5) * s + self.a * libm::sqrt(yz) - self.sqmu * self.tof
    }

    /// `F′(z)` (Curtis Eq. 5.43), with the `z → 0` limit taken analytically: the general form divides
    /// by `z` and by `C(z) − 3S(z)/2C(z)`, both of which vanish there.
    fn dfdz(self, z: f64) -> f64 {
        let c = stumpff_c(z);
        let s = stumpff_s(z);
        let yz = self.y(z);
        let a = self.a;
        if z.abs() < 1e-9 {
            let y0 = self.y(0.0);
            core::f64::consts::SQRT_2 / 40.0 * libm::pow(y0, 1.5)
                + a / 8.0 * (libm::sqrt(y0) + a * libm::sqrt(1.0 / (2.0 * y0)))
        } else {
            libm::pow(yz / c, 1.5)
                * (1.0 / (2.0 * z) * (c - 3.0 * s / (2.0 * c)) + 3.0 * s * s / (4.0 * c))
                + a / 8.0 * (3.0 * s / c * libm::sqrt(yz) + a * libm::sqrt(c / yz))
        }
    }

    /// The residual at `z`, and whether `z` sits below the root — negative residual there, or the
    /// arc is under the `y = 0` edge where the residual is not defined at all.
    ///
    /// The two ends of the domain behave in opposite directions and the test has to keep them apart.
    /// Under the `y = 0` edge `y` goes negative and the residual is `NaN`, which is *below*. At
    /// `z = 4π²`, `y` stays finite — it tends to `r1 + r2 + √2·A`, which [`Self::y`] evaluates
    /// exactly — while `C(z) → 0` sends the residual to `+∞`, which is *above*. Testing `is_finite`
    /// first would call the second one "below" and walk a search straight off its own ceiling.
    fn probe(self, z: f64) -> (f64, bool) {
        let value = self.tof_err(z);
        let below = !(self.y(z) > 0.0 && (value >= 0.0 || value.is_nan()));
        (value, below)
    }

    /// A bracket `(lo, hi)` with [`Self::probe`] reporting below the root at `lo` and not at `hi`.
    ///
    /// The search is **relative**, which is the point of it. The previous version walked
    /// `z += 0.1` from zero, an absolute step in a dimensionless quantity whose root can be
    /// anywhere: on a 6-second arc between two points 44 km apart the root is at `z = 3.9e-5`, the
    /// first step overshot it by 2500×, and unguarded Newton — started outside any bracket, on a
    /// part of the curve where `y` is a difference of two near-equal numbers — did not come back.
    /// Every arc shorter than about 1 % of a period failed that way, which is precisely the regime
    /// initial orbit determination works in. The same walk only ever went *upward*, so a transfer
    /// whose root is hyperbolic was never bracketed either, and the `y ≤ 0` guard in [`lambert`]
    /// reported Newton's overshoot as "not representable" — measured after this change, the arc at
    /// `tof = 10 s` across a 20 000 km chord departs at 634 km/s and reaches `r2` to 5 nanometres.
    ///
    /// Both defects are the same one: the solver had no bracket, only a walk that happened to suit
    /// the elliptical case.
    fn bracket(self) -> Option<(f64, f64)> {
        if !self.probe(0.0).1 {
            // Hyperbolic: the domain runs to `z → −∞`, where the residual tends to `−√µ·tof` and is
            // monotone the whole way, so doubling cannot step over anything.
            let mut hi = 0.0;
            let mut lo = -1.0;
            for _ in 0..HYPERBOLIC_RUNGS {
                if self.probe(lo).1 {
                    return Some((lo, hi));
                }
                hi = lo;
                lo *= 2.0;
            }
            return None;
        }

        // Elliptical, and the root is somewhere in `(0, 4π²)`.
        if !self.probe(1.0).1 {
            // Below one, halving resolves the initial-orbit-determination root at 3.9e-5 in
            // fourteen probes, where an absolute step cannot resolve it at all.
            let mut hi = 1.0;
            let mut lo = 0.5;
            for _ in 0..HALVING_RUNGS {
                if self.probe(lo).1 {
                    return Some((lo, hi));
                }
                hi = lo;
                lo *= 0.5;
            }
            // `below(0)` is true, so zero closes the bracket.
            return Some((0.0, hi));
        }

        // Doubling to the ceiling. There is no fallback below this one and none is needed: `y` has
        // no zero on `[0, 4π²]` for any geometry this function is reachable with, so the domain
        // cannot end early and the ladder cannot be walking past an edge. With `A² = (1 + cos Δν)
        // r1 r2`, `y` is smallest at `cos(√z/2) = −1`, where it is `r1 + r2 − √2|A|`, so an edge
        // inside the domain needs `2A² > (r1 + r2)²`, i.e. `2(1 + cos Δν) r1 r2 > (r1 + r2)²`. The
        // left side is largest at `cos Δν = 1`, and `4 r1 r2 > (r1 + r2)²` is `0 > (r1 − r2)²` —
        // false for every geometry, and an equality only in the `Δν → 0` limit that [`lambert`]
        // refuses at `sin Δν`. Measured over the 25 270-case sweep of
        // `examples/lambert_longway_sweep.rs`, the uniform 80-rung sweep that used to sit here was
        // entered 0 times against 8 728 for the ceiling probe below.
        let ceiling = REVOLUTION * (1.0 - 1e-9);
        let mut previous = 1.0;
        let mut z = 2.0;
        for _ in 0..DOUBLING_RUNGS {
            if !self.probe(z).1 {
                return Some((previous, z));
            }
            if z >= ceiling {
                return None;
            }
            previous = z;
            z = (z * 2.0).min(ceiling);
        }
        None
    }

    /// The root `z` of [`Self::tof_err`], and the number of safeguarded Newton steps it cost.
    ///
    /// The step count is returned for the same reason [`crate::kepler::universal_anomaly`] returns
    /// one: it is the only observable a wrong `dfdz` moves, so it is the only thing a test can hold
    /// the derivative to.
    fn solve(self) -> Option<(f64, u32)> {
        let (mut lo, mut hi) = self.bracket()?;
        let mut z = f64::midpoint(lo, hi);

        // Newton where it helps, bisection where it does not. An unguarded Newton is what made the
        // old bracket's overshoot fatal rather than merely slow; with the bracket maintained as an
        // invariant, a step that leaves it is simply replaced by the midpoint and the iteration
        // cannot leave the root behind.
        //
        // It stops where the residual runs out of information, not where the step runs out of size.
        // Within a few ulps of the root `tof_err` is dominated by its own cancellation error; the
        // step becomes that noise divided by `F′`, and the iterate joins a small attractor instead
        // of settling. Measured over the 24 cases of
        // `newton_stops_when_the_residual_stops_carrying_information`, 12 reached the root in 3 to 5
        // steps and then cycled with period 6 or 7 across 23 ulps of `z` for all 95 remaining
        // iterations, because a bare `next == z` test can only see a cycle of period 1. The last
        // informative step before the attractor is 8e4 ulps wide, so a band of `32 ε|z|` — between
        // 32 and 64 ulps — separates signal from noise by four decades.
        let mut answer = z;
        let mut steps = 100;
        for k in 0..100 {
            let (residual, below) = self.probe(z);
            let newton = z - residual / self.dfdz(z);
            // Tested against the bracket that still has `z` strictly inside it. Shrinking the
            // bracket onto `z` first would put an endpoint *at* `z`, and then every step short
            // enough to be the converging one gets rejected as out of bounds: measured that way,
            // the sweep took 24 iterations to reach a root Newton alone reaches in four.
            let inside = newton.is_finite() && newton > lo && newton < hi;
            if below {
                lo = z;
            } else {
                hi = z;
            }
            // The bisection fallback uses the *shrunk* bracket, so `z` stays inside it either way.
            let next = if inside {
                newton
            } else {
                f64::midpoint(lo, hi)
            };
            answer = next;
            if libm::fabs(next - z) <= 32.0 * f64::EPSILON * libm::fabs(z) {
                steps = k;
                break;
            }
            z = next;
        }

        // A bracket is not a root. Where the transfer is faster than the formulation can represent
        // there is no crossing at all, and the sign change the search found is the `y = 0` edge of
        // the domain — at which both positive terms of the residual vanish and it equals `−√µ·tof`
        // exactly, a full scale away from zero. Without this the edge is returned as an answer and
        // the caller reads a degenerate near-rectilinear arc as a transfer.
        let residual = self.tof_err(answer);
        if !residual.is_finite() || libm::fabs(residual) > 1e-8 * self.sqmu * self.tof {
            return None;
        }
        Some((answer, steps))
    }
}

/// The nondimensional time of flight `T(x)` of the Izzo/Lancaster–Blanchard formulation, for `n_rev`
/// full revolutions, in the unit `T = t·√(2µ/s³)`, `s` the semi-perimeter.
///
/// Lagrange's equation in Izzo's variable is `a^{3/2}[(α − sin α) − (β − sin β) + 2πN]/2` with
/// `a = 1/(1 − x²)`, `α = 2 acos x`, `β = 2 asin(λ√(1 − x²))`, and its `sinh` form on the
/// hyperbola. Evaluated that way it is `∞·0` at the parabola `x = 1`, which came back as NaN, and
/// near it `α − sin α` keeps only the digits above `α³/6`. Here `w = √|1 − x²|` is formed as
/// `(1 − x)(1 + x)`, so it is exact to a rounding up to the parabola, and
///
/// ```text
/// T = [(F(α) − F(β))/2 + πN]/w³,    F(u) = u − sin u (ellipse),  sinh u − u (hyperbola),
/// ```
///
/// with `F(u) = u³ S(±u²)` below `|u| = 1`, `S` the Stumpff function, whose series [`stumpff_s`]
/// holds there (`derivations/lambert_izzo.mac` 4–5). The quotient then has no cancellation left but
/// the one `λ → 1` puts in the problem, up to `x = 1` itself, where it is `0/0` and takes its limit
/// `⅔(1 − λ³)`, Euler's parabolic time (G3, `lambert_izzo.mac` 6–15) — for `n_rev = 0`, the only
/// family that reaches the parabola. `β` carries the sign of `λ`, negative the long way round,
/// through `asin` and `asinh`, which are odd, as `F` is.
fn izzo_tof(x: f64, lambda: f64, n_rev: usize) -> f64 {
    let ellipse = x < 1.0;
    let w = libm::sqrt(if ellipse {
        (1.0 - x) * (1.0 + x)
    } else {
        (x - 1.0) * (x + 1.0)
    });
    if n_rev == 0 && w == 0.0 {
        return 2.0 / 3.0 * (1.0 - lambda * lambda * lambda);
    }
    let bracket = if ellipse {
        x_minus_sin(2.0 * libm::acos(x)) - x_minus_sin(2.0 * libm::asin(lambda * w))
    } else {
        sinh_minus_x(2.0 * libm::acosh(x)) - sinh_minus_x(2.0 * libm::asinh(lambda * w))
    };
    (0.5 * bracket + core::f64::consts::PI * n_rev as f64) / (w * w * w)
}

/// `u − sin u`, as `u³ S(u²)` where the difference cancels.
fn x_minus_sin(u: f64) -> f64 {
    if libm::fabs(u) <= 1.0 {
        u * u * u * stumpff_s(u * u)
    } else {
        u - libm::sin(u)
    }
}

/// `sinh u − u`, as `u³ S(−u²)` where the difference cancels.
fn sinh_minus_x(u: f64) -> f64 {
    if libm::fabs(u) <= 1.0 {
        u * u * u * stumpff_s(-u * u)
    } else {
        libm::sinh(u) - u
    }
}

/// The `x` at which the `n_rev`-revolution time of flight is least, `n_rev ≥ 1`: the root of
/// `dT/dx = (3Tx − 2 + 2λ³x/y)/(1 − x²)`, `y = √(1 − λ²(1 − x²))` (Izzo 2015, Eq. 22; derived in
/// `derivations/lambert_izzo.mac` 16–17).
///
/// `T` tends to `∞` at both ends of `(−1, 1)` and has one minimum between, so the numerator is
/// negative at `lo` and positive at `hi` and bisection to adjacent doubles finds the root. Where
/// on the flat bottom the doubles stop is immaterial: `T` moves by the square of that distance.
fn minimum_time_x(lambda: f64, n_rev: usize, lo: f64, hi: f64) -> f64 {
    let slope = |x: f64| {
        let y = libm::sqrt(1.0 - lambda * lambda * ((1.0 - x) * (1.0 + x)));
        3.0 * izzo_tof(x, lambda, n_rev) * x - 2.0 + 2.0 * lambda * lambda * lambda * x / y
    };
    let (mut a, mut b) = (lo, hi);
    let mut x = f64::midpoint(a, b);
    for _ in 0..100 {
        if slope(x) < 0.0 {
            a = x;
        } else {
            b = x;
        }
        let next = f64::midpoint(a, b);
        #[expect(
            clippy::float_cmp,
            reason = "the fixed point is exactly where the step stops moving `x`"
        )]
        let settled = next == x;
        if settled {
            break;
        }
        x = next;
    }
    x
}

/// The interval of Izzo's `x` that holds the root of `T(x) = t_target` on the branch asked for.
///
/// `n_rev = 0`: `x ∈ (−1, ∞)` and `T` falls monotonically from `∞` to `0`, so a coarse scan brackets
/// the one sign change. `n_rev ≥ 1`: `x ∈ (−1, 1)`, `T` has one minimum, and the two roots lie on
/// either side of it — bracketed exactly, where a scan misses a pair closer than its step, which near
/// the minimum time they always are. Below that minimum the refusal carries it, divided by `unit`
/// back into the caller's time.
fn izzo_bracket(
    lambda: f64,
    n_rev: usize,
    t_target: f64,
    unit: f64,
    long_branch: bool,
) -> Result<(f64, f64), LambertError> {
    let lo = -1.0 + 1e-6;
    let f = |x: f64| izzo_tof(x, lambda, n_rev) - t_target;
    if n_rev == 0 {
        let hi = 20.0;
        let steps = 2000;
        let mut xa = lo;
        let mut fa = f(xa);
        for k in 1..=steps {
            let xb = lo + (hi - lo) * f64::from(k) / f64::from(steps);
            let fb = f(xb);
            if fa * fb < 0.0 {
                return Ok((xa, xb));
            }
            xa = xb;
            fa = fb;
        }
        return Err(LambertError::OutOfRange);
    }
    let hi = 1.0 - 1e-6;
    let x_min = minimum_time_x(lambda, n_rev, lo, hi);
    let t_min = izzo_tof(x_min, lambda, n_rev);
    if t_min > t_target {
        return Err(LambertError::BelowMinimumTime {
            minimum: t_min / unit,
        });
    }
    let (bracket, end) = if long_branch {
        ((x_min, hi), hi)
    } else {
        ((lo, x_min), lo)
    };
    let at_end = f(end);
    if at_end.is_nan() || at_end <= 0.0 {
        return Err(LambertError::OutOfRange);
    }
    Ok(bracket)
}

/// Why [`lambert_multirev`] returned no transfer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LambertError {
    /// The time of flight is not finite and positive.
    InvalidTimeOfFlight,
    /// `r1` and `r2` coincide, or are parallel or antiparallel: the transfer plane is undefined.
    NoTransferPlane,
    /// No transfer with `n_rev` revolutions is that fast. The family's least time of flight is the
    /// root of `dT/dx = 0` in Izzo's variable; below it the time equation has no root.
    BelowMinimumTime {
        /// The least time of flight of the family, in the caller's unit.
        minimum: f64,
    },
    /// The transfer exists, but its root lies outside the interval the solver searches, `x` from
    /// `−1 + 1e-6` to `20` for zero revolutions and to `1 − 1e-6` for more: a time of flight
    /// beyond about a billion times `√(s³/2µ)`, or a hyperbola faster than `x = 20` carries.
    OutOfRange,
}

impl core::fmt::Display for LambertError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidTimeOfFlight => write!(f, "the time of flight is not finite and positive"),
            Self::NoTransferPlane => write!(
                f,
                "the positions are coincident, parallel or antiparallel; the transfer plane is undefined"
            ),
            Self::BelowMinimumTime { minimum } => write!(
                f,
                "the time of flight is below the {minimum:e} the revolution count needs at least"
            ),
            Self::OutOfRange => write!(
                f,
                "the transfer's root lies outside the interval of x the solver searches"
            ),
        }
    }
}

impl core::error::Error for LambertError {}

/// Solve Lambert's problem allowing `n_rev` complete revolutions (Izzo formulation).
///
/// For `n_rev = 0` this is a single-revolution transfer (matching [`lambert`]); for `n_rev ≥ 1` two
/// solutions exist above the family's minimum time and `long_branch` selects the higher-`x` (shorter,
/// higher-energy) branch when `true`. `prograde` and the geometry fix the transfer sense.
///
/// # Errors
///
/// [`LambertError::InvalidTimeOfFlight`] unless `tof` is finite and positive;
/// [`LambertError::NoTransferPlane`] when the chord or `|r1 × r2|` is below `1e-12`;
/// [`LambertError::BelowMinimumTime`] when `n_rev ≥ 1` revolutions cannot be flown in `tof`;
/// [`LambertError::OutOfRange`] when the root lies outside the interval the solver searches.
pub fn lambert_multirev(
    r1: Vec3,
    r2: Vec3,
    tof: f64,
    mu: f64,
    prograde: bool,
    n_rev: usize,
    long_branch: bool,
) -> Result<(Vec3, Vec3), LambertError> {
    if !(tof.is_finite() && tof > 0.0) {
        return Err(LambertError::InvalidTimeOfFlight);
    }
    let r1n = r1.norm();
    let r2n = r2.norm();
    let c = Vec3::new(r2.x - r1.x, r2.y - r1.y, r2.z - r1.z);
    let cn = c.norm();
    let s = (r1n + r2n + cn) / 2.0;
    // Only the chord needs a floor. The triangle inequality gives `r1n + r2n ≥ cn`, so
    // `s = (r1n + r2n + cn)/2 ≥ cn`, and a second test on `s` could never fire once this one has
    // passed — it read as a guard while guarding nothing.
    if cn < 1e-12 {
        return Err(LambertError::NoTransferPlane);
    }
    let i_r1 = r1.scale(1.0 / r1n);
    let i_r2 = r2.scale(1.0 / r2n);
    let cross = r1.cross(r2);
    let cross_n = cross.norm();
    if cross_n < 1e-12 {
        return Err(LambertError::NoTransferPlane); // 0° or 180° transfer
    }
    let i_h = cross.scale(1.0 / cross_n);
    // λ = √(r1 r2)·cos(Δν/2)/s = |A|/(√2 s), the sign below. As `√(1 − c/s)` it cancelled within
    // ε of a half turn, where λ ≈ ε/4: on 13 % of random geometries at ε ~ 1e-9 `1 − c/s` rounded
    // below zero and λ was NaN, and elsewhere it was off by up to 1.7e-8 absolute.
    let mut lambda =
        chord_parameter_magnitude(r1n, r2n, r1.dot(r2), cross_n) / (core::f64::consts::SQRT_2 * s);
    let (mut i_t1, mut i_t2);
    if i_h.z < 0.0 {
        lambda = -lambda;
        i_t1 = i_r1.cross(i_h);
        i_t2 = i_r2.cross(i_h);
    } else {
        i_t1 = i_h.cross(i_r1);
        i_t2 = i_h.cross(i_r2);
    }
    if !prograde {
        lambda = -lambda;
        i_t1 = i_t1.scale(-1.0);
        i_t2 = i_t2.scale(-1.0);
    }

    let unit = libm::sqrt(2.0 * mu / (s * s * s));
    let t_target = unit * tof;
    let f = |x: f64| izzo_tof(x, lambda, n_rev) - t_target;
    let (mut a, mut b) = izzo_bracket(lambda, n_rev, t_target, unit, long_branch)?;
    // Bisect until the bracket is one ulp wide. Stopping on an absolute residual instead leaves
    // whatever `x`-error that residual is worth, which is two orders of magnitude at `T ~ 1`.
    let mut x = f64::midpoint(a, b);
    for _ in 0..100 {
        if f(x) * f(a) < 0.0 {
            b = x;
        } else {
            a = x;
        }
        let next = f64::midpoint(a, b);
        #[expect(
            clippy::float_cmp,
            reason = "the fixed point is exactly where the step stops moving `x`"
        )]
        let settled = next == x;
        if settled {
            break;
        }
        x = next;
    }

    // Reconstruct the velocities (Izzo).
    let gamma = libm::sqrt(mu * s / 2.0);
    let rho = (r1n - r2n) / cn;
    let sigma = libm::sqrt(1.0 - rho * rho);
    let y = libm::sqrt(1.0 - lambda * lambda * (1.0 - x * x));
    let vr1 = gamma * ((lambda * y - x) - rho * (lambda * y + x)) / r1n;
    let vr2 = -gamma * ((lambda * y - x) + rho * (lambda * y + x)) / r2n;
    let vt1 = gamma * sigma * (y + lambda * x) / r1n;
    let vt2 = gamma * sigma * (y + lambda * x) / r2n;
    let v1 = Vec3::new(
        vr1 * i_r1.x + vt1 * i_t1.x,
        vr1 * i_r1.y + vt1 * i_t1.y,
        vr1 * i_r1.z + vt1 * i_t1.z,
    );
    let v2 = Vec3::new(
        vr2 * i_r2.x + vt2 * i_t2.x,
        vr2 * i_r2.y + vt2 * i_t2.y,
        vr2 * i_r2.z + vt2 * i_t2.z,
    );
    Ok((v1, v2))
}

#[cfg(test)]
mod tests {
    // `LAGRANGE_ARCS` is carried at the 25 digits it was printed to, so that regenerating it is a
    // copy rather than a rounding decision.
    #![allow(
        clippy::expect_used,
        clippy::excessive_precision,
        clippy::unreadable_literal
    )]
    use super::*;
    use crate::kepler::propagate;
    use spacelab_testkit::assert_within_ulps;

    const MU_EARTH_KM: f64 = 398_600.0;

    /// Two geometries whose out-of-plane products straddle one another: the first has
    /// `r1.x·r2.y > r1.y·r2.x > 0`, the second the reverse. Corrupting either factor of the cross
    /// product therefore reverses the transfer sense on at least one of them.
    const GEOMETRIES: [(Vec3, Vec3); 2] = [
        (
            Vec3 {
                x: 6000.0,
                y: 2000.0,
                z: 1500.0,
            },
            Vec3 {
                x: 1000.0,
                y: 5000.0,
                z: 4000.0,
            },
        ),
        (
            Vec3 {
                x: 2000.0,
                y: 6000.0,
                z: 1500.0,
            },
            Vec3 {
                x: 5000.0,
                y: 1000.0,
                z: 4000.0,
            },
        ),
    ];

    /// 400 s is hyperbolic on both geometries and both senses, the rest elliptic and progressively
    /// slower, so the sweep crosses the parabolic boundary the Stumpff branches hinge on.
    const TOFS: [f64; 6] = [400.0, 1000.0, 3000.0, 10_000.0, 40_000.0, 120_000.0];

    /// `T(x, λ, N)` of the Lancaster–Blanchard form, Lagrange's equation at 100 digits
    /// (`derivations/lambert_izzo.gp` §2, which also holds the literals to it, to 2e-25), on both
    /// conics and both signs of `λ`, with the condition number `κ` of the equation's one subtraction,
    /// `F(α) − F(β) + 2πN`: `(x, λ, N, T, κ)`. `κ` is the factor by which the rounding of the two
    /// terms reaches `T`, and it is the problem's: it grows as `λ → 1`, the chord to zero.
    #[rustfmt::skip]
    const IZZO_TOF: [(f64, f64, usize, f64, f64); 18] = [
        (0.0, 0.5, 0, 1.480210253088817069536076, 1.122),
        (-0.9, 0.3, 0, 37.20601928472000445313341, 1.001),
        (0.5, 0.9, 0, 0.3213882554999886628004576, 4.884),
        (0.75, 0.45, 0, 0.7208031367246040107950984, 1.173),
        (-0.25, 0.8, 0, 1.839046851196585578166719, 1.475),
        (0.3, 0.2, 1, 4.742432784142829841881213, 1.002),
        (-0.5, 0.7, 2, 13.30433815213550543478457, 1.039),
        (0.0, -0.5, 0, 1.661382400500976168926567, 1.000),
        (-0.7, -0.8, 0, 8.196593203105028973287010, 1.000),
        (0.4, -0.3, 1, 5.128706840676636927341485, 1.000),
        (0.6, -0.95, 3, 20.00282179450132532506031, 1.000),
        (1.05, 0.2, 0, 0.6418958560667003134648856, 1.017),
        (1.5, 0.4, 0, 0.4709958762024938184400980, 1.171),
        (3.0, 0.9, 0, 0.06295435879031136660460637, 8.438),
        (12.0, 0.6, 0, 0.05322346229764423099810463, 2.084),
        (1.5, -0.4, 0, 0.5516955127126932571137157, 1.000),
        (2.5, -0.85, 0, 0.5823615525963159046384228, 1.000),
        (1.02, -0.75, 0, 0.9381126730306875345381481, 1.000),
    ];

    /// Curtis, Example 5.2: a one-hour prograde transfer.
    #[test]
    fn matches_curtis_example_5_2() {
        let r1 = Vec3::new(5000.0, 10000.0, 2100.0);
        let r2 = Vec3::new(-14600.0, 2500.0, 7000.0);
        let (v1, v2) = lambert(r1, r2, 3600.0, MU_EARTH_KM, true).expect("solvable");
        let v1_exp = Vec3::new(-5.9925, 1.9254, 3.2456);
        let v2_exp = Vec3::new(-3.3125, -4.1966, -0.38529);
        let d1 =
            ((v1.x - v1_exp.x).powi(2) + (v1.y - v1_exp.y).powi(2) + (v1.z - v1_exp.z).powi(2))
                .sqrt();
        let d2 =
            ((v2.x - v2_exp.x).powi(2) + (v2.y - v2_exp.y).powi(2) + (v2.z - v2_exp.z).powi(2))
                .sqrt();
        assert!(d1 < 1e-3, "v1 off by {d1:.5} km/s: got {v1:?}");
        assert!(d2 < 1e-3, "v2 off by {d2:.5} km/s: got {v2:?}");
    }

    /// The transfer sense of a `LAGRANGE_ARCS` row, named so the `.gp` that prints the rows never
    /// prints a bare `false`.
    const PROGRADE: bool = true;
    const RETROGRADE: bool = false;

    /// The branch of a `MULTIREV_ARCS` row, `long_branch` of [`lambert_multirev`], named for the
    /// same reason.
    const HIGH_X: bool = true;
    const LOW_X: bool = false;

    /// One Lambert arc and its two velocities: `(r1, r2, tof, prograde, v1, v2)`.
    type LagrangeArc = ([f64; 3], [f64; 3], f64, bool, [f64; 3], [f64; 3]);

    /// The arcs of `GEOMETRIES` × `TOFS` × both senses, and Curtis Example 5.2, solved at 100 digits
    /// by `derivations/lambert_universal.gp` through Lagrange's theorem — the semi-major axis from
    /// Lagrange's time equation, the velocities from the semi-latus rectum — which shares nothing with
    /// the universal-variable iteration under test. Each arc there is propagated by the universal
    /// Kepler equation and lands on `r2` to 7e-111 relative.
    const LAGRANGE_ARCS: [LagrangeArc; 25] = [
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            400.0000000000000000000000,
            PROGRADE,
            [
                -10.95320049888329200865424,
                8.610173844145451005077400,
                7.114482562239796200663165,
            ],
            [
                -13.50209907819297837846145,
                6.056948671674398155465612,
                5.108299808991801703114535,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            1000.000000000000000000000,
            PROGRADE,
            [
                -1.755361912817030093791646,
                5.299870493201658680317338,
                4.290170251908779834072712,
            ],
            [
                -7.065932183491950678618628,
                -0.01971413261574112360582448,
                0.1103356467104214306492237,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            3000.000000000000000000000,
            PROGRADE,
            [
                3.891017162671157399031714,
                4.799907705654800400674104,
                3.787586242710622511273267,
            ],
            [
                -5.030916445058062378781991,
                -4.137170316703824287928761,
                -3.234674067975250546035933,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            10000.00000000000000000000,
            PROGRADE,
            [
                6.713135941743045433239608,
                5.004570527673860592450830,
                3.901652460778226450482996,
            ],
            [
                -4.582222462208746975318254,
                -6.309961028486662188365503,
                -4.988678996780197348234453,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            40000.00000000000000000000,
            PROGRADE,
            [
                8.013298973369985443975761,
                5.174610120855454341566545,
                4.015074222591526070116407,
            ],
            [
                -4.470235409060057517547722,
                -7.330114266907532426290860,
                -5.810444760746051815456089,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            120000.0000000000000000000,
            PROGRADE,
            [
                8.437147152796479593691889,
                5.238546621023793456304276,
                4.058883049879896891785724,
            ],
            [
                -4.444362873997504369460780,
                -7.664828949437720296862024,
                -6.079873925905355517666612,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            400.0000000000000000000000,
            RETROGRADE,
            [
                -25.86552987765022635015436,
                -9.616117581756226404263527,
                -7.265352880381785032887578,
            ],
            [
                5.567103231472794233660449,
                21.86987042212706544302981,
                17.47459046007580626254788,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            1000.000000000000000000000,
            RETROGRADE,
            [
                -6.569035891162300095474699,
                -4.988351819730561152682096,
                -3.891192785655588424557493,
            ],
            [
                4.597919600981561394153396,
                6.197558868849040245623805,
                4.898075526736165172480675,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            3000.000000000000000000000,
            RETROGRADE,
            [
                1.742353643927987583633439,
                -5.297157965720159360116190,
                -4.287758251809556406943964,
            ],
            [
                7.059292159340379386378477,
                0.02880571452496560392836333,
                -0.1029113393878022716000380,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            10000.00000000000000000000,
            RETROGRADE,
            [
                4.748179087824014103542721,
                -6.101304315753074432425572,
                -4.987622737441292206476669,
            ],
            [
                8.815401129305617491074794,
                -2.027178423638387346264907,
                -1.786400539161304429874921,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            40000.00000000000000000000,
            RETROGRADE,
            [
                5.986383596577170122074898,
                -6.524961361036919836647065,
                -5.350172229343545763771300,
            ],
            [
                9.654354137093716168291127,
                -2.850764673907278422576546,
                -2.463192222552165092575638,
            ],
        ),
        (
            [
                6000.000000000000000000000,
                2000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                1000.000000000000000000000,
                5000.000000000000000000000,
                4000.000000000000000000000,
            ],
            120000.0000000000000000000,
            RETROGRADE,
            [
                6.377710020375588185156056,
                -6.668408147188403940387592,
                -5.472429940068817241117816,
            ],
            [
                9.931445749151528286368807,
                -3.108640178123958580793632,
                -2.675361674370172578965754,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            400.0000000000000000000000,
            PROGRADE,
            [
                -9.616117581756226404263527,
                -25.86552987765022635015436,
                -7.265352880381785032887578,
            ],
            [
                21.86987042212706544302981,
                5.567103231472794233660449,
                17.47459046007580626254788,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            1000.000000000000000000000,
            PROGRADE,
            [
                -4.988351819730561152682096,
                -6.569035891162300095474699,
                -3.891192785655588424557493,
            ],
            [
                6.197558868849040245623805,
                4.597919600981561394153396,
                4.898075526736165172480675,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            3000.000000000000000000000,
            PROGRADE,
            [
                -5.297157965720159360116190,
                1.742353643927987583633439,
                -4.287758251809556406943964,
            ],
            [
                0.02880571452496560392836333,
                7.059292159340379386378477,
                -0.1029113393878022716000380,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            10000.00000000000000000000,
            PROGRADE,
            [
                -6.101304315753074432425572,
                4.748179087824014103542721,
                -4.987622737441292206476669,
            ],
            [
                -2.027178423638387346264907,
                8.815401129305617491074794,
                -1.786400539161304429874921,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            40000.00000000000000000000,
            PROGRADE,
            [
                -6.524961361036919836647065,
                5.986383596577170122074898,
                -5.350172229343545763771300,
            ],
            [
                -2.850764673907278422576546,
                9.654354137093716168291127,
                -2.463192222552165092575638,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            120000.0000000000000000000,
            PROGRADE,
            [
                -6.668408147188403940387592,
                6.377710020375588185156056,
                -5.472429940068817241117816,
            ],
            [
                -3.108640178123958580793632,
                9.931445749151528286368807,
                -2.675361674370172578965754,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            400.0000000000000000000000,
            RETROGRADE,
            [
                8.610173844145451005077400,
                -10.95320049888329200865424,
                7.114482562239796200663165,
            ],
            [
                6.056948671674398155465612,
                -13.50209907819297837846145,
                5.108299808991801703114535,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            1000.000000000000000000000,
            RETROGRADE,
            [
                5.299870493201658680317338,
                -1.755361912817030093791646,
                4.290170251908779834072712,
            ],
            [
                -0.01971413261574112360582448,
                -7.065932183491950678618628,
                0.1103356467104214306492237,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            3000.000000000000000000000,
            RETROGRADE,
            [
                4.799907705654800400674104,
                3.891017162671157399031714,
                3.787586242710622511273267,
            ],
            [
                -4.137170316703824287928761,
                -5.030916445058062378781991,
                -3.234674067975250546035933,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            10000.00000000000000000000,
            RETROGRADE,
            [
                5.004570527673860592450830,
                6.713135941743045433239608,
                3.901652460778226450482996,
            ],
            [
                -6.309961028486662188365503,
                -4.582222462208746975318254,
                -4.988678996780197348234453,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            40000.00000000000000000000,
            RETROGRADE,
            [
                5.174610120855454341566545,
                8.013298973369985443975761,
                4.015074222591526070116407,
            ],
            [
                -7.330114266907532426290860,
                -4.470235409060057517547722,
                -5.810444760746051815456089,
            ],
        ),
        (
            [
                2000.000000000000000000000,
                6000.000000000000000000000,
                1500.000000000000000000000,
            ],
            [
                5000.000000000000000000000,
                1000.000000000000000000000,
                4000.000000000000000000000,
            ],
            120000.0000000000000000000,
            RETROGRADE,
            [
                5.238546621023793456304276,
                8.437147152796479593691889,
                4.058883049879896891785724,
            ],
            [
                -7.664828949437720296862024,
                -4.444362873997504369460780,
                -6.079873925905355517666612,
            ],
        ),
        (
            [
                5000.000000000000000000000,
                10000.00000000000000000000,
                2100.000000000000000000000,
            ],
            [
                -14600.00000000000000000000,
                2500.000000000000000000000,
                7000.000000000000000000000,
            ],
            3600.000000000000000000000,
            PROGRADE,
            [
                -5.992494639666396316880508,
                1.925363415280892576055172,
                3.245636528490489388402495,
            ],
            [
                -3.312460310936791953033341,
                -4.196617307926468915513533,
                -0.3852876170681051737142574,
            ],
        ),
    ];

    fn vec3(a: [f64; 3]) -> Vec3 {
        Vec3::new(a[0], a[1], a[2])
    }

    /// (G2) The universal-variable solver returns the arc that solves the two-body boundary-value
    /// problem, against 100-digit arcs from Lagrange's theorem. `derivations/lambert_universal.mac`
    /// shows the coded `y`, `F` and `F′` are the universal Kepler equation of the returned arc read
    /// backwards, so the root of `F` is that arc and nothing else.
    ///
    /// The bound is 64 ulps of `|v1|`. The iteration stops inside a band of `32 ε|z|` and the
    /// velocity is `(r2 − f r1)/g` with `f` and `g` formed from `y(z)`, so a few tens of ulps is the
    /// solver's own floor; measured, the worst row is 15.3 ulps (the 400 s hyperbolic arc the long
    /// way round). A wrong branch, a dropped term or a sign leaves 1e-9 or worse.
    #[test]
    fn the_arc_is_the_one_lagrange_s_theorem_gives_at_100_digits() {
        let mut worst = 0.0_f64;
        for (r1, r2, tof, prograde, v1, v2) in LAGRANGE_ARCS {
            let (got1, got2) = lambert(vec3(r1), vec3(r2), tof, MU_EARTH_KM, prograde)
                .unwrap_or_else(|| panic!("no arc at tof {tof} s, prograde {prograde}"));
            let scale = vec3(v1).norm();
            let d = (got1 - vec3(v1)).norm().max((got2 - vec3(v2)).norm()) / scale;
            assert!(
                d <= 64.0 * f64::EPSILON,
                "r1 {r1:?} tof {tof} prograde {prograde}: velocity off by {d:.3e} of |v1|"
            );
            worst = worst.max(d);
        }
        println!(
            "lambert vs Lagrange's theorem: worst {:.1} ulps of |v1|",
            worst / f64::EPSILON
        );
    }

    /// (G3) Two conics built in closed form by `derivations/lambert_universal.mac`, `µ = 1`, from
    /// periapsis at `r1 = (1, 0, 0)` to a true anomaly of 90°. The ellipse `e = 1/2` reaches
    /// `(0, 3/2, 0)` after `2√2(π/3 − √3/4)` with `v1 = (0, √(3/2), 0)`, `v2 = √(2/3)(−1, 1/2, 0)`;
    /// the hyperbola `e = 2` reaches `(0, 3, 0)` after `2√3 − ln(2 + √3)` with `v1 = (0, √3, 0)`,
    /// `v2 = (−1, 2, 0)/√3`. Rounding the time of flight to a double shifts the exact answer by
    /// 8e-18 and 5e-17 (`derivations/lambert_universal.gp`), below an ulp of the speeds, so the
    /// bound is the solvers' own: 16 ulps of `|v1|`, against a measured 1.0 (universal) and 3.3
    /// (Izzo).
    #[test]
    fn the_arc_is_the_exact_conic_it_was_built_from() {
        let third = 3.0_f64.sqrt();
        let cases = [
            (
                Vec3::new(0.0, 1.5, 0.0),
                2.0 * core::f64::consts::SQRT_2 * (core::f64::consts::PI / 3.0 - third / 4.0),
                Vec3::new(0.0, 1.5_f64.sqrt(), 0.0),
                Vec3::new(-(2.0_f64 / 3.0).sqrt(), 0.5 * (2.0_f64 / 3.0).sqrt(), 0.0),
            ),
            (
                Vec3::new(0.0, 3.0, 0.0),
                2.0 * third - (2.0 + third).ln(),
                Vec3::new(0.0, third, 0.0),
                Vec3::new(-1.0 / third, 2.0 / third, 0.0),
            ),
        ];
        let r1 = Vec3::new(1.0, 0.0, 0.0);
        for (r2, tof, v1, v2) in cases {
            let tol = 16.0 * f64::EPSILON * v1.norm();
            let universal = lambert(r1, r2, tof, 1.0, true).expect("a 90° arc is solvable");
            let izzo = lambert_multirev(r1, r2, tof, 1.0, true, 0, false).expect("solvable");
            for (name, (got1, got2)) in [("universal", universal), ("izzo", izzo)] {
                let (d1, d2) = ((got1 - v1).norm(), (got2 - v2).norm());
                assert!(
                    d1 <= tol && d2 <= tol,
                    "{name} to {r2:?}: v1 off by {d1:.3e}, v2 by {d2:.3e}, bound {tol:.3e}"
                );
            }
        }
    }

    /// (G2) The retrograde porkchop cell (46, 62) of `spacelab-gpu`'s test grid: a ten-day
    /// heliocentric arc at 252 km/s, SI units. `lambert` solved its `z` to 2e-16 of the 100-digit
    /// root and then refused it, because the closure check's `propagate` ran out its step cap on this
    /// arc and missed `r2` by 13 719 km. The departure velocity is the one
    /// `derivations/gpu_porkchop.gp` solves at 100 digits from the same `f32` inputs; measured 1.4e-14
    /// relative from it.
    #[test]
    fn the_fast_heliocentric_porkchop_cell_is_solved() {
        let r1 = Vec3::new(-81_028_456_448.0, -125_753_376_768.0, 0.0);
        let r2 = Vec3::new(-225_017_004_032.0, 36_680_925_184.0, 0.0);
        let mu = 132_712_443_255_260_708_864.0; // the Sun's μ in m³/s², rounded to `f32` by the grid
        let want = Vec3::new(
            -168_605.020_757_491_745_635_087,
            186_730.361_114_462_195_696_811,
            0.0,
        );
        let (v1, _) = lambert(r1, r2, 864_000.0, mu, false).expect("the cell has a transfer");
        let rel = (v1 - want).norm() / want.norm();
        assert!(rel < 1e-13, "v1 {v1:?} is {rel:.3e} from the 100-digit arc");
    }

    /// The departure velocity, propagated forward by the time of flight, must arrive at `r2` — a
    /// self-consistency check that closes Lambert against the independent Kepler propagator.
    #[test]
    fn the_solution_arc_reaches_the_target() {
        let r1 = Vec3::new(5000.0, 10000.0, 2100.0);
        let r2 = Vec3::new(-14600.0, 2500.0, 7000.0);
        let tof = 3600.0;
        let (v1, v2) = lambert(r1, r2, tof, MU_EARTH_KM, true).expect("solvable");
        let (rf, vf) =
            propagate(r1, v1, MU_EARTH_KM, tof).expect("the universal Kepler solve settles");
        let dr = ((rf.x - r2.x).powi(2) + (rf.y - r2.y).powi(2) + (rf.z - r2.z).powi(2)).sqrt();
        let dv = ((vf.x - v2.x).powi(2) + (vf.y - v2.y).powi(2) + (vf.z - v2.z).powi(2)).sqrt();
        assert!(dr < 1e-3, "arc missed r2 by {dr:.3e} km");
        assert!(dv < 1e-6, "arrival velocity disagrees by {dv:.3e} km/s");
    }

    /// A 180° transfer (antiparallel positions) has an undefined plane and returns `None`.
    #[test]
    fn a_half_turn_transfer_is_refused() {
        let r1 = Vec3::new(7000.0, 0.0, 0.0);
        let r2 = Vec3::new(-9000.0, 0.0, 0.0);
        assert!(lambert(r1, r2, 3000.0, MU_EARTH_KM, true).is_none());
    }

    /// The multi-rev solver's single-revolution case matches the universal-variable Lambert and
    /// re-propagates to the target.
    #[test]
    fn izzo_single_rev_matches_and_reaches_target() {
        let r1 = Vec3::new(5000.0, 10000.0, 2100.0);
        let r2 = Vec3::new(-14600.0, 2500.0, 7000.0);
        let tof = 3600.0;
        let (v1, _) =
            lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false).expect("N=0 solution");
        let (v1u, _) = lambert(r1, r2, tof, MU_EARTH_KM, true).expect("universal-variable");
        let d = ((v1.x - v1u.x).powi(2) + (v1.y - v1u.y).powi(2) + (v1.z - v1u.z).powi(2)).sqrt();
        assert!(
            d < 1e-6,
            "Izzo N=0 disagrees with universal-variable by {d:.3e} km/s"
        );
        let (rf, _) =
            propagate(r1, v1, MU_EARTH_KM, tof).expect("the universal Kepler solve settles");
        let miss = ((rf.x - r2.x).powi(2) + (rf.y - r2.y).powi(2) + (rf.z - r2.z).powi(2)).sqrt();
        assert!(miss < 1e-3, "N=0 arc missed by {miss:.3e} km");
    }

    /// Every arc the universal-variable solver returns must start at `r1`, end at `r2` after the
    /// requested time, hand back the arrival velocity the propagator finds there, conserve angular
    /// momentum and energy between the endpoints, and circulate in the requested sense. The last is
    /// what pins the transfer angle: both the short and the long way connect the two positions in
    /// the same time, and they differ only in the sign of the angular momentum.
    #[test]
    fn every_transfer_arc_connects_the_endpoints_and_keeps_the_requested_sense() {
        for &(r1, r2) in &GEOMETRIES {
            for prograde in [true, false] {
                for &tof in &TOFS {
                    let (v1, v2) = lambert(r1, r2, tof, MU_EARTH_KM, prograde)
                        .unwrap_or_else(|| panic!("no arc at tof {tof} s, prograde {prograde}"));
                    let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof)
                        .expect("the universal Kepler solve settles");
                    let scale = r2.norm() + tof * v1.norm();
                    let miss = (rf - r2).norm();
                    assert!(
                        miss < 2e-12 * scale,
                        "arc missed r2 by {miss:.3e} km at tof {tof} s, prograde {prograde}"
                    );
                    let dv = (vf - v2).norm();
                    assert!(
                        dv < 1e-10 * v1.norm(),
                        "arrival velocity off by {dv:.3e} km/s at tof {tof} s"
                    );
                    let (h1, h2) = (r1.cross(v1), r2.cross(v2));
                    assert_eq!(
                        h1.z > 0.0,
                        prograde,
                        "sense reversed at tof {tof} s: h_z = {:.4e}",
                        h1.z
                    );
                    let dh = (h1 - h2).norm();
                    assert!(
                        dh < 2e-14 * h1.norm(),
                        "angular momentum drifted by {dh:.3e} at tof {tof} s"
                    );
                    let e1 = v1.dot(v1) / 2.0 - MU_EARTH_KM / r1.norm();
                    let e2 = v2.dot(v2) / 2.0 - MU_EARTH_KM / r2.norm();
                    let de = (e1 - e2).abs();
                    assert!(
                        de < 3e-14 * (MU_EARTH_KM / r1.norm()),
                        "energy drifted by {de:.3e} at tof {tof} s"
                    );
                }
            }
        }
    }

    /// An arc far faster than the parabolic one is an ordinary hyperbolic transfer, and the solver
    /// has to find it: its root is at `z < 0`, which a search that only ever walks upward never
    /// brackets.
    ///
    /// This test used to assert the opposite — that everything below 236.5 s on these geometries was
    /// refused, "the iteration drives `y(z)` through zero, and the solver must say so". That was a
    /// description of the search, not of the formulation. The walk had no downward branch, unguarded
    /// Newton overshot the `y = 0` edge, and [`lambert`]'s guard reported the overshoot as "not
    /// representable". The arcs are real: at 50 s the departure is 127 km/s and the propagated state
    /// lands on `r2` to 3.5e-10 of the arc's own scale, the arrival velocity matching the propagated
    /// one to 5.5e-10 of the departure speed.
    ///
    /// A floor does exist, and it is a band rather than a threshold — so the fixture asserts the two
    /// ends of it and says nothing about the middle. Sampled 400 times a decade from 1 ms to 1e4 s,
    /// the short way refuses every time below 0.0310 s and solves every time above 0.1129 s; the
    /// long way refuses below 5.2396 s and solves above 7.5369 s and 8.5079 s on the two geometries.
    /// In between the closure test passes case by case, which is what a degeneracy reached through a
    /// coordinate that is running out of resolution looks like from the outside.
    ///
    /// The two senses are two orders apart because they degenerate for different reasons. The short
    /// way runs out of representable speed — 1.05e5 km/s at 0.06 s, against 20 000 km in one second
    /// being 6.7 % of the speed of light — while `y` stays healthy and the arc still closes to
    /// 1.5e-11 at 2 s. The long way runs into the whole-revolution end of the domain, where the
    /// residual still converges and the arc no longer arrives: it closes to 9.9e-14 at 200 s and
    /// 2.6e-9 at 20 s, and at 2 s [`CLOSURE_TOLERANCE`] refuses it outright. That is the case the
    /// closure test exists for.
    #[test]
    fn a_transfer_faster_than_the_parabolic_one_is_hyperbolic_and_gets_solved() {
        for (geometry, &(r1, r2)) in GEOMETRIES.iter().enumerate() {
            for prograde in [true, false] {
                for tof in [50.0, 100.0, 200.0, 236.0] {
                    let (v1, v2) =
                        lambert(r1, r2, tof, MU_EARTH_KM, prograde).unwrap_or_else(|| {
                            panic!(
                                "no arc at tof {tof} s, geometry {geometry}, prograde {prograde}"
                            )
                        });
                    let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof)
                        .expect("the universal Kepler solve settles");
                    let miss = (rf - r2).norm() / (r2.norm() + tof * v1.norm());
                    assert!(
                        miss < 5e-10,
                        "arc missed r2 by {miss:.3e} of scale at tof {tof} s, prograde {prograde}"
                    );
                    let dv = (vf - v2).norm() / v1.norm();
                    assert!(
                        dv < 1e-9,
                        "arrival velocity off by {dv:.3e} of |v1| at tof {tof} s"
                    );
                }
                // Under the floor the solver says so, rather than handing back an arc that does not
                // arrive. The two senses degenerate three orders apart, so one fixed time of flight
                // cannot straddle both — the bisected floor is the fixture.
                let (refused, solved) = if prograde == (geometry == 0) {
                    (0.031, 0.113)
                } else {
                    [(5.239, 7.537), (5.239, 8.508)][geometry]
                };
                assert!(
                    lambert(r1, r2, refused, MU_EARTH_KM, prograde).is_none(),
                    "expected no arc at {refused} s, geometry {geometry}, prograde {prograde}"
                );
                assert!(
                    lambert(r1, r2, solved, MU_EARTH_KM, prograde).is_some(),
                    "expected an arc at {solved} s, geometry {geometry}, prograde {prograde}"
                );
            }
        }
    }

    /// The Lancaster–Blanchard time of flight against Lagrange's equation at 100 digits, on the
    /// ellipse and the hyperbola and for both signs of `λ`. The budget is 8 ulps per unit of `κ`:
    /// over 3 000 random points of `x ∈ (−1, 20)`, `λ ∈ (−0.99, 0.99)`, `N ≤ 3` the worst was 4.8
    /// `κ` ulps, where the Lagrange form it replaced reached 17.5 on the ellipse and was unbounded
    /// near `x = 1`.
    #[test]
    fn the_nondimensional_time_of_flight_matches_its_closed_form_on_both_conics() {
        for (x, lambda, n_rev, want, kappa) in IZZO_TOF {
            let got = izzo_tof(x, lambda, n_rev);
            assert!(
                ulps(got, want) <= 8.0 * kappa,
                "T(x = {x}, lambda = {lambda}, N = {n_rev}): {got:e} against {want:e}, {:.1} ulps",
                ulps(got, want)
            );
        }
    }

    /// The Izzo formulation must land on the same arc as the universal-variable one over the whole
    /// sweep — both conics, both senses, and both signs of the orbit normal.
    #[test]
    fn the_izzo_solver_reproduces_the_universal_variable_arc_over_the_sweep() {
        for &(r1, r2) in &GEOMETRIES {
            for prograde in [true, false] {
                for &tof in &TOFS {
                    let (v1, v2) = lambert_multirev(r1, r2, tof, MU_EARTH_KM, prograde, 0, false)
                        .unwrap_or_else(|e| {
                            panic!("no arc at tof {tof} s, prograde {prograde}: {e}")
                        });
                    let (v1u, _) = lambert(r1, r2, tof, MU_EARTH_KM, prograde).expect("uv arc");
                    let d = (v1 - v1u).norm();
                    assert!(
                        d < 1e-10 * v1.norm(),
                        "Izzo differs from universal-variable by {d:.3e} km/s at tof {tof} s"
                    );
                    let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof)
                        .expect("the universal Kepler solve settles");
                    let miss = (rf - r2).norm();
                    assert!(
                        miss < 5e-13 * (r2.norm() + tof * v1.norm()),
                        "Izzo arc missed r2 by {miss:.3e} km at tof {tof} s"
                    );
                    let dv = (vf - v2).norm();
                    assert!(
                        dv < 1e-10 * v1.norm(),
                        "Izzo arrival velocity off by {dv:.3e} km/s at tof {tof} s"
                    );
                    assert_eq!(
                        r1.cross(v1).z > 0.0,
                        prograde,
                        "Izzo sense reversed at tof {tof} s"
                    );
                }
            }
        }
    }

    /// A one-revolution transfer has two solutions. Both reach the target; `long_branch` picks the
    /// higher-energy one, and the two are separated by hundreds of metres per second.
    #[test]
    fn both_one_revolution_branches_reach_the_target_and_differ_in_energy() {
        let r1 = Vec3::new(8000.0, 0.0, 0.0);
        let r2 = Vec3::new(-2000.0, 8000.0, 3000.0);
        let period = core::f64::consts::TAU * (8000f64.powi(3) / MU_EARTH_KM).sqrt();
        for k in [1.5, 2.0, 2.5] {
            let tof = k * period;
            let (vs, _) =
                lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 1, false).expect("short branch");
            let (vl, _) =
                lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 1, true).expect("long branch");
            for (name, v) in [("short", vs), ("long", vl)] {
                let (rf, _) =
                    propagate(r1, v, MU_EARTH_KM, tof).expect("the universal Kepler solve settles");
                let miss = (rf - r2).norm();
                assert!(
                    miss < 5e-12 * (r2.norm() + tof * v.norm()),
                    "{name} branch missed r2 by {miss:.3e} km at {k} periods"
                );
            }
            let sep = vl.norm() - vs.norm();
            assert!(
                sep > 0.4,
                "the two branches coincide at {k} periods: separation {sep:.3e} km/s"
            );
        }
    }

    /// The long way round, where the root crowds against the whole-revolution end of the domain.
    ///
    /// `Δν = 355°` on a circle: `A = −9890.07`, and the root sits at `z = 39.10` against a ceiling
    /// of `4π² = 39.478`. All three of these came back `None` while `C(z)` was evaluated as
    /// `1 − cos√z`, which returns a bare `0.0` at the ceiling — `y` is then `NaN`, `probe` reads
    /// that as *below the root*, and the search reports no crossing on a curve whose residual at the
    /// ceiling is `+2.8e28`. The half-angle form in [`crate::kepler::stumpff`] is what makes the
    /// ceiling probe answer, and the closed form of [`Arc::y`] is what makes the arc it finds
    /// accurate; neither alone was enough, and the pair is measured over 25 271 geometries in the
    /// ticket rather than here.
    ///
    /// Both solvers are asked, because agreement between two formulations that crowd at different
    /// places is the evidence: Izzo's Lancaster–Blanchard variable has no singularity at one
    /// revolution at all. At 60 000 s the two departure velocities agree to 4.8e-15 relative and the
    /// arc closes to 1.1e-11 of the radius; at 600 000 s to 6.4e-14 and 7.5e-9; at 6e6 s — a
    /// 714 000 km semi-major axis, `e ≈ 0.99` — to 2.4e-14 and 1.3e-7. The last was refused, and
    /// the first two closed to 4.2e-9 and 2.2e-7, while `A` was `sin Δν·√(r1 r2/(1 − cos Δν))`:
    /// `1 − cos 355° = 0.0038` carried `cos Δν`'s rounding up 260 times, and near a whole revolution
    /// the arc is that sensitive to `A`. At 2e7 s the closure test refuses, the honest end of what
    /// `z` can carry.
    #[test]
    fn the_long_way_round_is_solved_where_the_root_crowds_the_whole_revolution() {
        let r1 = Vec3::new(7000.0, 0.0, 0.0);
        let dnu = 5f64.to_radians();
        let r2 = Vec3::new(7000.0 * dnu.cos(), 7000.0 * dnu.sin(), 0.0);
        for (tof, bound) in [(60_000.0, 5e-11), (600_000.0, 3e-8), (6.0e6, 5e-7)] {
            let (v1, _) = lambert(r1, r2, tof, MU_EARTH_KM, false)
                .unwrap_or_else(|| panic!("no long-way arc at tof {tof} s"));
            let miss = (propagate(r1, v1, MU_EARTH_KM, tof)
                .expect("the universal Kepler solve settles")
                .0
                - r2)
                .norm()
                / r2.norm();
            assert!(
                miss < bound,
                "long-way arc missed r2 by {miss:.3e} at {tof} s"
            );
            let (w1, _) = lambert_multirev(r1, r2, tof, MU_EARTH_KM, false, 0, false)
                .unwrap_or_else(|e| panic!("no Izzo arc at tof {tof} s: {e}"));
            let d = (w1 - v1).norm() / v1.norm();
            assert!(
                d < 1e-12,
                "the two solvers differ by {d:.3e} at tof {tof} s"
            );
        }

        // Past what `z` can resolve the arc is refused rather than answered. Izzo still solves it,
        // which is the whole reason the refusal is the right answer and not a gap.
        assert!(
            lambert(r1, r2, 2.0e7, MU_EARTH_KM, false).is_none(),
            "2e7 s is past the resolution of z and must be refused"
        );
        let (w1, _) = lambert_multirev(r1, r2, 2.0e7, MU_EARTH_KM, false, 0, false)
            .expect("Izzo solves what the universal variable cannot");
        let miss = (propagate(r1, w1, MU_EARTH_KM, 2.0e7)
            .expect("the universal Kepler solve settles")
            .0
            - r2)
            .norm()
            / r2.norm();
        assert!(miss < 1e-8, "the Izzo arc missed r2 by {miss:.3e} at 2e7 s");
    }

    /// The two senses are always two different arcs — one each way round. That stays true on the
    /// boundary of the sense test, where the transfer plane contains the polar axis: `r1 × r2` has
    /// no `z` component, neither sense is achievable, and a solver that collapsed the tie would
    /// hand back the same trajectory for both requests.
    #[test]
    fn the_two_senses_are_distinct_arcs_even_when_the_plane_contains_the_pole() {
        let polar = (Vec3::new(7000.0, 0.0, 0.0), Vec3::new(-3000.0, 0.0, 6000.0));
        let on_the_boundary =
            polar.0.x * polar.1.y - polar.0.y * polar.1.x == 0.0 && polar.0.cross(polar.1).z == 0.0;
        assert!(
            on_the_boundary,
            "the polar geometry no longer straddles the sense test"
        );
        for (r1, r2) in [polar, GEOMETRIES[0], GEOMETRIES[1]] {
            for tof in [3000.0, 9000.0] {
                let arcs = [true, false].map(|prograde| {
                    let (v1, _) = lambert(r1, r2, tof, MU_EARTH_KM, prograde)
                        .unwrap_or_else(|| panic!("no arc at tof {tof} s, prograde {prograde}"));
                    let (w1, _) = lambert_multirev(r1, r2, tof, MU_EARTH_KM, prograde, 0, false)
                        .unwrap_or_else(|e| panic!("no Izzo arc at tof {tof} s: {e}"));
                    let d = (w1 - v1).norm();
                    assert!(
                        d < 1e-10 * v1.norm(),
                        "the two solvers disagree by {d:.3e} km/s at tof {tof} s, \
                         prograde {prograde}"
                    );
                    v1
                });
                let sep = (arcs[0] - arcs[1]).norm();
                assert!(
                    sep > 1.0,
                    "the two senses returned the same arc at tof {tof} s: {sep:.3e} km/s apart"
                );
            }
        }
    }

    /// The scan's first cell is not decoration: a root can live inside it, and the sweep never puts
    /// one there.
    ///
    /// `x` is sampled uniformly on `(-1 + 1e-6, 20)`, but `T(x)` is not: over the first cell alone
    /// it falls from 1.1e9 to 1039.7 -- six decades -- while the remaining 1999 cells share
    /// `(0, 1039.7)`. Every other fixture in this file asks for a transfer of hours to days, whose
    /// `T` is single digits, so it is bracketed a thousand cells along and the first one could be
    /// anything. Move the lower limit off the domain where `T` is real and `f(lo)` is NaN, the first
    /// cell can never bracket, and this transfer disappears entirely.
    ///
    /// 4.0e6 s on this geometry is `T_target = 1951`, which sits between the two endpoints above and
    /// so brackets in the first cell and nowhere else -- a 0.9996-revolution ellipse with `e ≈ 0.99`
    /// just inside escape. The whole window `(2.1e6, 2.2e12) s` lands there, so the fixture is not
    /// knife-edge.
    #[test]
    fn a_root_in_the_scans_first_cell_is_found() {
        let r1 = Vec3::new(8000.0, 0.0, 0.0);
        let r2 = Vec3::new(-2000.0, 8000.0, 3000.0);
        let tof = 4.0e6;
        let (v1, _) = lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false)
            .expect("a near-parabolic transfer whose root is in the first cell of the scan");
        let (rf, _) =
            propagate(r1, v1, MU_EARTH_KM, tof).expect("the universal Kepler solve settles");
        let miss = (rf - r2).norm();
        assert!(
            miss < 1e-9 * (r2.norm() + tof * v1.norm()),
            "missed r2 by {miss:.3e} km"
        );
    }

    /// The Izzo solver refuses geometries with no transfer plane and revolution counts the time of
    /// flight cannot support, and says which. The near-coincident case is the one the chord guard
    /// exists for: the positions are 0.1 pm apart, which the cross product alone does not resolve.
    #[test]
    fn the_izzo_solver_refuses_degenerate_geometries_and_impossible_revolution_counts() {
        let r1 = Vec3::new(7000.0, 0.0, 0.0);
        let cases = [
            ("near-coincident", Vec3::new(7000.0, 1e-13, 0.0), 0),
            ("antiparallel", Vec3::new(-9000.0, 0.0, 0.0), 0),
            (
                "three revolutions in an hour",
                Vec3::new(0.0, 8000.0, 1000.0),
                3,
            ),
        ];
        for (name, r2, n_rev) in cases {
            let got = lambert_multirev(r1, r2, 3600.0, MU_EARTH_KM, true, n_rev, false);
            let typed = match got {
                Err(LambertError::NoTransferPlane) => n_rev == 0,
                Err(LambertError::BelowMinimumTime { minimum }) => n_rev == 3 && minimum > 3600.0,
                _ => false,
            };
            assert!(typed, "the {name} case returned {got:?}");
        }
    }

    /// A one-revolution transfer reaches the target after completing a full revolution.
    #[test]
    fn izzo_one_rev_reaches_target() {
        let r1 = Vec3::new(8000.0, 0.0, 0.0);
        let r2 = Vec3::new(0.0, 8000.0, 1000.0); // ~90° plus out of plane
        let period = core::f64::consts::TAU * (8000f64.powi(3) / MU_EARTH_KM).sqrt();
        let tof = 1.3 * period; // one full revolution plus the transfer
        let (v1, v2) =
            lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 1, false).expect("1-rev solution");
        let (rf, vf) =
            propagate(r1, v1, MU_EARTH_KM, tof).expect("the universal Kepler solve settles");
        let miss = ((rf.x - r2.x).powi(2) + (rf.y - r2.y).powi(2) + (rf.z - r2.z).powi(2)).sqrt();
        assert!(miss < 1e-2, "1-rev arc missed the target by {miss:.3e} km");
        let dv = ((vf.x - v2.x).powi(2) + (vf.y - v2.y).powi(2) + (vf.z - v2.z).powi(2)).sqrt();
        assert!(
            dv < 1e-4,
            "1-rev arrival velocity disagrees by {dv:.3e} km/s"
        );
    }

    /// A transfer takes time. The universal-variable bracket opens at `z = 0` and walks upward
    /// while the time error is negative, so at `tof ≤ 0` it never opens and Newton settles on a
    /// degenerate near-rectilinear arc. The Izzo path always refused these, and the two share a
    /// contract, so this pins both.
    #[test]
    fn no_transfer_is_solved_in_no_time() {
        for k in 0..1000 {
            let a = f64::from(k) * 0.0062;
            let r1 = Vec3::new(8000.0 * a.cos(), 8000.0 * a.sin(), 1500.0 * (a * 2.0).sin());
            let b = a * 1.7 + 0.4;
            let r2 = Vec3::new(
                12000.0 * b.cos(),
                12000.0 * b.sin(),
                2500.0 * (b * 3.0).cos(),
            );
            for tof in [0.0, -1.0, -3600.0, f64::NAN, f64::INFINITY] {
                assert!(
                    lambert(r1, r2, tof, MU_EARTH_KM, true).is_none(),
                    "single-rev solved tof = {tof}"
                );
                assert_eq!(
                    lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false),
                    Err(LambertError::InvalidTimeOfFlight),
                    "izzo solved tof = {tof}"
                );
            }
        }
    }

    /// The `Arc` the public solver builds for one of the sweep geometries, so the iteration and its
    /// derivative can be reached directly.
    fn arc_of(r1: Vec3, r2: Vec3, tof: f64, prograde: bool) -> Arc {
        Arc::new(r1, r2, tof, MU_EARTH_KM, prograde).expect("a geometry with a transfer plane")
    }

    /// Every case of the sweep, as the iteration sees it.
    fn arcs() -> impl Iterator<Item = (f64, bool, Arc)> {
        GEOMETRIES.into_iter().flat_map(|(r1, r2)| {
            TOFS.into_iter().flat_map(move |tof| {
                [true, false]
                    .into_iter()
                    .map(move |pro| (tof, pro, arc_of(r1, r2, tof, pro)))
            })
        })
    }

    /// The step count is not a performance detail — it is the only thing that separates an iteration
    /// that has converged from one that is circling the root inside its own rounding noise, and it is
    /// the only observable `dfdz` moves at all. Before the noise band was recognised, 12 of these 24
    /// cases ran the full 100 iterations having reached the root in 3 to 5.
    ///
    /// The bound is 8, measured: 3 steps in eight cases, 4 in eight, 5 in four, 7 in two, 8 in two.
    /// It was 6 while the iteration started at `z = 0`; it starts from the middle of a bracket now,
    /// which is further from the root in the cases that cost 7 and 8. That is not a regression — the
    /// walk that used to place `z = 0` so conveniently spent about 160 residual evaluations getting
    /// there on this sweep's largest root, against the five probes the doubling ladder spends.
    #[test]
    fn newton_stops_when_the_residual_stops_carrying_information() {
        for (tof, pro, arc) in arcs() {
            let (z, steps) = arc.solve().expect("the sweep is all solvable");
            assert!(
                steps <= 8,
                "tof {tof} prograde {pro}: {steps} steps to reach z = {z}"
            );
            // Converged means the residual is down at the level `tof_err` can still resolve: worst
            // over the sweep is 2.3e-14 of the opening residual, which is where its own cancellation
            // error puts the floor.
            let opening = libm::fabs(arc.tof_err(0.0));
            let landed = libm::fabs(arc.tof_err(z));
            assert!(
                landed <= 1e-13 * opening,
                "tof {tof} prograde {pro}: residual {landed:e} against opening {opening:e}"
            );
        }
    }

    /// The bracket must not reach one whole revolution, and the ceiling that keeps it out must be on
    /// the near side of `4π²` rather than the far side.
    ///
    /// `z = 4π²` is where `C` and `S` have their double zero and the parameterisation stops carrying
    /// information: doubles are 7.1e-15 apart there, so `4π² − z` has about seven significant figures
    /// however it is computed, and arcs that close to a whole revolution belong to
    /// [`lambert_multirev`]. The ladder therefore stands off by a *relative* 1e-9, and which side of
    /// the singularity that standoff lands on is invisible in every count the solver reports:
    /// measured with `examples/lambert_domain_probe.rs`, putting the ceiling 1e-9 *above* `4π²`
    /// instead of below leaves solved, refused and the worst miss identical over 13 620 calls and
    /// moves only the fingerprint of the accepted velocities. So the invariant is asserted where it
    /// lives, on the bracket itself.
    ///
    /// The final assertion is not decoration. Without it a geometry family that never reaches the
    /// ceiling would satisfy every line above it vacuously, and the test would pass while measuring
    /// nothing.
    #[test]
    fn the_bracket_never_reaches_one_whole_revolution() {
        let r1 = Vec3::new(7000.0, 0.0, 0.0);
        let mut at_the_ceiling = 0usize;
        let mut bracketed = 0usize;
        for dnu_deg in 1..=359 {
            let dnu = f64::from(dnu_deg).to_radians();
            for r2n in [7050.0_f64, 12000.0, 42164.0] {
                let r2 = Vec3::new(r2n * libm::cos(dnu), r2n * libm::sin(dnu), 0.0);
                for tof in [60.0_f64, 3600.0, 6e5, 6e6] {
                    for pro in [true, false] {
                        // 180° has no transfer plane, and `lambert` refuses it before any bracket.
                        let Some(arc) = Arc::new(r1, r2, tof, MU_EARTH_KM, pro) else {
                            continue;
                        };
                        let Some((lo, hi)) = arc.bracket() else {
                            continue;
                        };
                        bracketed += 1;
                        assert!(
                            lo < hi,
                            "dnu {dnu_deg} r2 {r2n} tof {tof} prograde {pro}: empty bracket ({lo}, {hi})"
                        );
                        assert!(
                            hi < REVOLUTION,
                            "dnu {dnu_deg} r2 {r2n} tof {tof} prograde {pro}: bracket ({lo}, {hi}) \
                             reaches z = 4pi^2 = {REVOLUTION}"
                        );
                        if hi > REVOLUTION * (1.0 - 1e-8) {
                            at_the_ceiling += 1;
                        }
                    }
                }
            }
        }
        assert!(bracketed > 1000, "only {bracketed} brackets to check");
        assert!(
            at_the_ceiling > 0,
            "no case reached the ceiling, so the domain assertion measured nothing"
        );
    }

    /// `dfdz` decides only how fast the iteration arrives, never where, so no assertion on a returned
    /// velocity can see it being wrong — including the whole `z → 0` closed form, which the general
    /// expression above it would otherwise agree with to seven digits. Hold it to the slope of the
    /// residual it claims to differentiate.
    #[test]
    fn the_derivative_is_the_slope_of_the_residual_it_differentiates() {
        for (tof, pro, arc) in arcs() {
            let (z, _) = arc.solve().expect("the sweep is all solvable");
            // Richardson-extrapolated central difference: the plain `O(h²)` quotient leaves 2.8e-7
            // relative at the sharpest case, which is a weaker claim than `dfdz` deserves.
            let h = 1e-3;
            for probe in [0.0, z, z * 0.5] {
                let d = |k: f64| (arc.tof_err(probe + k) - arc.tof_err(probe - k)) / (2.0 * k);
                let fd = (4.0 * d(h / 2.0) - d(h)) / 3.0;
                let got = arc.dfdz(probe);
                // Worst deviation over the 72 probes: 5.6e-10 relative.
                assert!(
                    libm::fabs(got - fd) <= 5e-9 * libm::fabs(fd),
                    "tof {tof} prograde {pro} at z = {probe}: F' = {got:e}, slope = {fd:e}"
                );
            }
        }
    }

    /// The `z → 0` closed form drops `z` entirely, so it returns one value across the whole window it
    /// owns — which is what makes the window's edge observable at all. Widening the test by one ulp
    /// hands `z = 1e-9` to a branch that is *more* accurate there, so only the bit pattern can tell
    /// the two apart.
    #[test]
    fn the_small_z_closed_form_owns_exactly_the_window_it_claims() {
        let arc = arc_of(GEOMETRIES[0].0, GEOMETRIES[0].1, 10_000.0, true);
        let at_zero = arc.dfdz(0.0).to_bits();
        for z in [1e-300, 1e-10, 9.99e-10, -9.99e-10] {
            assert_eq!(arc.dfdz(z).to_bits(), at_zero, "z = {z:e} left the window");
        }
        for z in [1e-9, -1e-9, 1.01e-9] {
            assert_ne!(
                arc.dfdz(z).to_bits(),
                at_zero,
                "z = {z:e} stayed in the window"
            );
        }
    }

    /// Both degeneracy floors in [`lambert_multirev`] are strict, and both boundaries are reachable
    /// exactly: a chord of `1e-12` between two radii that differ only in `y`, and a cross product of
    /// `1e-12` between `(1,0,0)` and `(2,1e-12,0)`. Widening either test by one value refuses a
    /// transfer the method can still solve.
    #[test]
    fn the_degeneracy_floors_exclude_their_own_boundary() {
        let chord = (Vec3::new(7000.0, 0.0, 0.0), Vec3::new(7000.0, 1e-12, 0.0));
        let plane = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 1e-12, 0.0));
        let c = Vec3::new(
            chord.1.x - chord.0.x,
            chord.1.y - chord.0.y,
            chord.1.z - chord.0.z,
        );
        assert_eq!(
            c.norm().to_bits(),
            1e-12_f64.to_bits(),
            "chord off the floor"
        );
        assert_eq!(
            plane.0.cross(plane.1).norm().to_bits(),
            1e-12_f64.to_bits(),
            "cross product off the floor"
        );
        for ((r1, r2), tof) in [(chord, 1000.0), (plane, 0.01)] {
            assert!(
                lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false).is_ok(),
                "a transfer sitting exactly on the floor was refused"
            );
        }
    }

    /// The bracket exists because Newton from `z = 0` on an elliptic transfer starts on the far side
    /// of the root. Anything that stops the walk advancing — a step that scales `z` instead of adding
    /// to it, or subtracts — leaves `z` at or below zero forever and the guard turns the transfer into
    /// a `None`.
    #[test]
    fn the_bracket_walks_upward_until_the_residual_turns() {
        let mut bracketed = 0;
        for (tof, pro, arc) in arcs() {
            let (z, _) = arc.solve().expect("the sweep is all solvable");
            if arc.tof_err(0.0) < 0.0 {
                bracketed += 1;
                assert!(
                    z > 0.0,
                    "tof {tof} prograde {pro}: elliptic root at z = {z}"
                );
            } else {
                assert!(
                    z < 0.0,
                    "tof {tof} prograde {pro}: hyperbolic root at z = {z}"
                );
            }
        }
        assert_eq!(
            bracketed, 20,
            "the sweep stopped covering both sides of z = 0"
        );
    }

    /// The chord parameter `A` at four geometries `ε = π − Δν` from a half turn, `ε` from 1e-3 to
    /// 1e-9, with `A` and `ε` of the rounded inputs at 100 digits: `derivations/lambert_universal.gp`
    /// (an ellipse `a = 8000`, `e = 1/10` in a tilted plane, from `ν = 1/5`).
    type ChordRow = ([f64; 3], [f64; 3], f64, f64);
    #[rustfmt::skip]
    const HALF_TURN_CHORD_PARAMETER: [ChordRow; 4] = [
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7278.142952334410438197665, -3891.782426811342247674474, -2997.358325967428299918538], 5.627438843804057132305699, 0.0009999999999999982989955449),
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7273.119570508919423446059, -3897.270545531217521784129, -3001.858124614212101732846], 0.05627377876677908560904413, 9.999999999992352284393373e-6),
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7273.069297876281780190766, -3897.325404136211091099540, -3001.903105147378937545000], 0.0005627377263706891664315383, 1.000000000092437755546228e-7),
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7273.068795146074990043417, -3897.325952720001623674762, -3001.903554950964462477714], 5.627377208753631083399044e-6, 9.999999914173621598894867e-10),
    ];

    /// Two arcs near a half turn (`ε` = 1e-5, 1e-7, same ellipse) and two short arcs (`Δν` = 1e-5,
    /// 1e-6 on `a = 7000`, `e = 1/100`, from `ν = 3/10`), solved at 100 digits by Lagrange's theorem
    /// and closed by propagation in `derivations/lambert_universal.gp`: `(r1, r2, tof, v1, v2)`.
    type ConditionedArc = ([f64; 3], [f64; 3], f64, [f64; 3], [f64; 3]);
    #[rustfmt::skip]
    const CONDITIONED_ARCS: [ConditionedArc; 4] = [
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7273.119570508919423446059, -3897.270545531217521784129, -3001.858124614212101732846], 3651.023950078398229379673, [-4.246083195582606909858931, 5.054541148717857784932724, 4.137507765878255070161449], [3.700670878039358920815984, -4.038300971060049907025663, -3.311146688871541222974586]),
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7273.069297876281780190766, -3897.325404136211091099540, -3001.903105147378937545000], 3651.037534706469614320667, [-4.246083195548371617057054, 5.054541147177273384719584, 4.137507767795426881067365], [3.700729053304269517066541, -4.038269796481149410782788, -3.311122679307051220863110]),
        ([5326.410355657255422556773, 3505.500152526995861990144, 2721.805294786741796997376], [5326.366140774000086821616, 3505.541391958417079877108, 2721.839223167049567564391], 0.009100279199550379005678735, [-4.858600322713788264227000, 4.531685691613534483699758, 3.728293865144129084044200], [-4.858658298791240977913491, 4.531647535111542407661273, 3.728264238956411392830121]),
        ([5326.410355657255422556773, 3505.500152526995861990144, 2721.805294786741796997376], [5326.405934192788663494866, 3505.504276485654827411054, 2721.808687636815648147603], 0.0009100278959799462620214316, [-4.858600322350830616464859, 4.531685691556924932192853, 3.728293865346473073206667], [-4.858606119980309159269656, 4.531681875926875011372758, 3.728290902744280789383657]),
    ];

    /// `T(x, λ, N)` at the doubles next to the parabola `x = 1`, from Lagrange's equation at 100
    /// digits (`derivations/lambert_izzo.gp` §3), with the condition number `κ` of the equation's
    /// one subtraction: `(x, λ, T, κ)`.
    #[rustfmt::skip]
    const NEAR_PARABOLA: [(f64, f64, f64, f64); 52] = [
        (0.99900000000000011, 0.6, 0.5230357949262925489338788, 1.551),
        (0.99999900000000008, 0.6, 0.5226670355628987786108234, 1.551),
        (0.99999999000000006, 0.6, 0.5226666703556266834469480, 1.551),
        (0.99999999989999999, 0.6, 0.5226666667035562857084587, 1.551),
        (0.99999999999990008, 0.6, 0.5226666666667035427688385, 1.551),
        (0.99999999999999989, 0.6, 0.5226666666666667236095615, 1.551),
        (1.0000000000000000, 0.6, 0.5226666666666666826538782, 1.551),
        (1.0000000000000002, 0.6, 0.5226666666666666007425116, 1.551),
        (1.0000000000000999, 0.6, 0.5226666666666298225389179, 1.551),
        (1.0000000001000000, 0.6, 0.5226666666297770796039403, 1.551),
        (1.0000000099999999, 0.6, 0.5226666629777067282860422, 1.551),
        (1.0000009999999999, 0.6, 0.5226662977708988390397139, 1.551),
        (1.0009999999999999, 0.6, 0.5222980026595272273517125, 1.551),
        (0.99900000000000011, -0.6, 0.8110979958040046957871804, 1.000),
        (0.99999900000000008, -0.6, 0.8106670977708916320202356, 1.000),
        (0.99999999000000006, -0.6, 0.8106666709777066469809030, 1.000),
        (0.99999999989999999, -0.6, 0.8106666667097770542486758, 1.000),
        (0.99999999999990008, -0.6, 0.8106666666667097266222678, 1.000),
        (0.99999999999999989, -0.6, 0.8106666666666666985416138, 1.000),
        (1.0000000000000000, -0.6, 0.8106666666666666506794551, 1.000),
        (1.0000000000000002, -0.6, 0.8106666666666665549551378, 1.000),
        (1.0000000000000999, -0.6, 0.8106666666666235747366424, 1.000),
        (1.0000000001000000, -0.6, 0.8106666666235562471147348, 1.000),
        (1.0000000099999999, -0.6, 0.8106666623556266993813438, 1.000),
        (1.0000009999999999, -0.6, 0.8106662355628917027100293, 1.000),
        (1.0009999999999999, -0.6, 0.8102357875628335364634137, 1.000),
        (0.99900000000000011, 0.95, 0.09517390507473709519295696, 13.018),
        (0.99999900000000008, 0.95, 0.09508342382104244632697441, 13.023),
        (0.99999999000000006, 0.95, 0.09508333423820966639609761, 13.023),
        (0.99999999989999999, 0.95, 0.09508333334238217674097409, 13.023),
        (0.99999999999990008, 0.95, 0.09508333333334245502146039, 13.023),
        (0.99999999999999989, 0.95, 0.09508333333333342353758018, 13.023),
        (1.0000000000000000, 0.95, 0.09508333333333341349143571, 13.023),
        (1.0000000000000002, 0.95, 0.09508333333333339339914677, 13.023),
        (1.0000000000000999, 0.95, 0.09508333333332437196141103, 13.023),
        (1.0000000001000000, 0.95, 0.09508333332428465024357814, 13.023),
        (1.0000000099999999, 0.95, 0.09508333242845717739481430, 13.023),
        (1.0000009999999999, 0.95, 0.09508324284579246106278914, 13.023),
        (1.0009999999999999, 0.95, 0.09499292967247104236920686, 13.028),
        (0.99900000000000011, -0.95, 1.238959885655560149528102, 1.000),
        (0.99999900000000008, -0.95, 1.238250709512747964304085, 1.000),
        (0.99999999000000006, -0.95, 1.238250007095123664031753, 1.000),
        (0.99999999989999999, -0.95, 1.238250000070951163216160, 1.000),
        (0.99999999999990008, -0.95, 1.238250000000070814369646, 1.000),
        (0.99999999999999989, -0.95, 1.238249999999999998613595, 1.000),
        (1.0000000000000000, -0.95, 1.238249999999999919841898, 1.000),
        (1.0000000000000002, -0.95, 1.238249999999999762298503, 1.000),
        (1.0000000000000999, -0.95, 1.238249999999929025314149, 1.000),
        (1.0000000001000000, -0.95, 1.238249999929048676475097, 1.000),
        (1.0000000099999999, -0.95, 1.238249992904876250272572, 1.000),
        (1.0000009999999999, -0.95, 1.238249290487998080686954, 1.000),
        (1.0009999999999999, -0.95, 1.237540860549889721445919, 1.000),
    ];

    /// The least time of flight of each `N`-revolution family, the minimum of Lagrange's equation at
    /// 100 digits, which `derivations/lambert_izzo.gp` §4 also finds as the root of Izzo's `dT/dx`
    /// and agrees with to 1e-115: `(geometry index into MULTIREV_GEOMETRIES, prograde, N, t_min)`.
    #[rustfmt::skip]
    const MINIMUM_TIMES: [(usize, bool, usize, f64); 12] = [
        (0, PROGRADE, 1, 9302.621563421778164293195),
        (0, PROGRADE, 2, 15866.25342541269979173278),
        (0, PROGRADE, 3, 22357.86550009264789483184),
        (0, RETROGRADE, 1, 9422.081966674056984523967),
        (0, RETROGRADE, 2, 15985.78052989836772947771),
        (0, RETROGRADE, 3, 22477.41007313714738512720),
        (1, PROGRADE, 1, 4697.472596821732873642248),
        (1, PROGRADE, 2, 8102.902678944104841198308),
        (1, PROGRADE, 3, 11471.18217741325526525274),
        (1, RETROGRADE, 1, 5018.369904929783449880600),
        (1, RETROGRADE, 2, 8424.431161844781279402131),
        (1, RETROGRADE, 3, 11792.87651659091571985435),
    ];

    /// The two geometries of `MINIMUM_TIMES`.
    const MULTIREV_GEOMETRIES: [(Vec3, Vec3); 2] = [
        (
            Vec3 {
                x: 8000.0,
                y: 0.0,
                z: 0.0,
            },
            Vec3 {
                x: -2000.0,
                y: 8000.0,
                z: 3000.0,
            },
        ),
        (
            Vec3 {
                x: 6000.0,
                y: 2000.0,
                z: 1500.0,
            },
            Vec3 {
                x: 1000.0,
                y: 5000.0,
                z: 4000.0,
            },
        ),
    ];

    /// One multi-revolution arc: `(r1, r2, tof, prograde, n_rev, long_branch, v1, v2)`.
    type MultirevArc = (
        [f64; 3],
        [f64; 3],
        f64,
        bool,
        usize,
        bool,
        [f64; 3],
        [f64; 3],
    );

    /// Every branch of `MULTIREV_GEOMETRIES` for `N` = 0, 1, 2, both senses, at times of flight on
    /// both sides of each family's minimum-energy time, and the near-half-turn geometry
    /// (`π − Δν = 1.5e-9`) on which `1 − c/s` rounded below zero: solved at 100 digits by Lagrange's
    /// equation in its own variable on the branch named, and closed by propagation, in
    /// `derivations/lambert_izzo.gp` §5.
    #[rustfmt::skip]
    const MULTIREV_ARCS: [MultirevArc; 29] = [
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 3600.000000000000000000000, PROGRADE, 0, LOW_X, [2.970662637551662682734492, 5.774448490799108682903637, 2.165418184049665756088864], [-4.895835063344011908239369, -3.514453709820387098657072, -1.317920141182645161996402]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 9500.000000000000000000000, PROGRADE, 1, LOW_X, [2.182190245682532646982994, 6.042998305090253720702135, 2.266124364408845145263301], [-5.334721638695905407278939, -2.833106665577393253692785, -1.062414999591522470134794]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 9500.000000000000000000000, PROGRADE, 1, HIGH_X, [0.6989346462145072013654281, 6.586689144233182849362754, 2.470008429087443568511033], [-6.197502210317909657679085, -1.556747735661092766734678, -0.5837804008729097875255043]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 14000.00000000000000000000, PROGRADE, 1, LOW_X, [4.803095689167012135764747, 5.203266099519307707256191, 1.951224787319740390221072], [-3.926937506907304739737380, -5.105314370448011870075245, -1.914492888918004451278217]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 14000.00000000000000000000, PROGRADE, 1, HIGH_X, [-1.380169724884940583296231, 7.434813347638514062267487, 2.788055005364442773350307], [-7.489897522051458917289080, 0.2203366976517794200863751, 0.08262626161941728253239068]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 20000.00000000000000000000, PROGRADE, 2, LOW_X, [4.330163417228450004918462, 5.343770733199087279854503, 2.003914024949657729945439], [-4.170329595056503945213813, -4.693764552570333338562760, -1.760161707213875001961035]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 20000.00000000000000000000, PROGRADE, 2, HIGH_X, [-0.4898578494807611139245032, 7.059309645683605677174880, 2.647241117131352128940580], [-6.924578531351753729526930, -0.5389244573274077905918024, -0.2020966714977779214719259]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 3600.000000000000000000000, RETROGRADE, 0, LOW_X, [-1.899362746420236758958556, -6.142763046661228127113595, -2.303536142497960547667598], [5.495466816719323923983111, 2.589184919767616812521936, 0.9709443449128563046957261]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 9500.000000000000000000000, RETROGRADE, 1, LOW_X, [-2.818584112155820826442972, -5.825156714401365351356647, -2.184433767900512006758742], [4.979435409706922926845377, 3.382885218777769698045078, 1.268581957041663636766904]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 9500.000000000000000000000, RETROGRADE, 1, HIGH_X, [-3.769152572130378883548384, -5.516614582061647909501189, -2.068730468273117966062946], [4.465007183114974537590420, 4.206429595786693487643074, 1.577411098420010057866153]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 14000.00000000000000000000, RETROGRADE, 1, LOW_X, [0.009409213407498504435232032, -6.864333650423221288354010, -2.574125118908707983132754], [6.626903072263124225598711, 0.9497223126403882510211971, 0.3561458672401455941329489]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 14000.00000000000000000000, RETROGRADE, 1, HIGH_X, [-6.182602511819540317588844, -4.819649712094341056065894, -1.807368642035377896024710], [3.242291098772907648612153, 6.309434453285733629814965, 2.366037919982150111180612]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 20000.00000000000000000000, RETROGRADE, 2, LOW_X, [-0.4459324949587460329650534, -6.684514368611448380983713, -2.506692888229293142868892], [6.349577735389910265697371, 1.339746532886152461145368, 0.5024049498323071729295130]),
        ([8000.000000000000000000000, 0.0, 0.0], [-2000.000000000000000000000, 8000.000000000000000000000, 3000.000000000000000000000], 20000.00000000000000000000, RETROGRADE, 2, HIGH_X, [-5.242385969646618076005016, -5.076930458228480785700527, -1.903848921835680294637698], [3.704887614372878908771935, 5.488171375422407507714367, 2.058064265783402815392888]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 3600.000000000000000000000, PROGRADE, 0, LOW_X, [4.487252819722471211542561, 4.821715139461940624407553, 3.794463008144015301549952], [-4.909318239772895027335977, -4.590806001537773813319688, -3.601374139811195117358035]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 5100.000000000000000000000, PROGRADE, 1, LOW_X, [1.532815685816215430035093, 4.845693027586030571527304, 3.866488759920627862298100], [-5.676971901952998506802159, -2.376332715881239964917159, -1.808178577012549998472678]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 5100.000000000000000000000, PROGRADE, 1, HIGH_X, [-1.935331673107803457681880, 5.338123536635158053846986, 4.324123050387319926442790], [-7.158703924584142893535872, 0.1058849431058407707663119, 0.2129201136490531710360698]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 9000.000000000000000000000, PROGRADE, 1, LOW_X, [4.738222135308154227007676, 4.834569594435638207129324, 3.800310885969563662389498], [-4.862730710863302177317153, -4.782680258318990097825174, -3.756390730598058075443137]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 9000.000000000000000000000, PROGRADE, 1, HIGH_X, [-4.290732570332036841730016, 5.957345137666162235310263, 4.863772567237666739833783], [-8.521159784923285591751190, 1.719737042044619136565662, 1.534095119230913334592964]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 10000.00000000000000000000, PROGRADE, 2, LOW_X, [3.075616823786005926925622, 4.791029992635771237446321, 3.795013086514708924252836], [-5.223405050246970250519671, -3.522078942992235681771673, -2.736966917578636346950101]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 10000.00000000000000000000, PROGRADE, 2, HIGH_X, [-2.458236883023714774243003, 5.456851856935183535513919, 4.428867329376910247720881], [-7.437737138138865618636332, 0.4688992169642026683878590, 0.5096107482415711731444640]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 3600.000000000000000000000, RETROGRADE, 0, LOW_X, [2.424532809346644448846352, -5.448863037179587354726529, -4.421845883614787060920360], [7.419331587092296993783611, -0.4455859063093280571338273, -0.4905481673395010636572471]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 5100.000000000000000000000, RETROGRADE, 1, LOW_X, [-1.161868489074682141386158, -4.873553536728364985827809, -3.895500726137531111086879], [5.804431011182223958608520, 2.104570813690294160848062, 1.587522421515732379992040]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 5100.000000000000000000000, RETROGRADE, 1, HIGH_X, [-2.793573992003126469408575, -4.793863291127917982594875, -3.802326323370592549060014], [5.297369081176583024330684, 3.310813643121660164901322, 2.565879372487466507075514]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 9000.000000000000000000000, RETROGRADE, 1, LOW_X, [2.820076936789090515601759, -5.545484871207516398468534, -4.506551716805845150833675], [7.638920243183514488563765, -0.7184618849057069791958949, -0.7137447332846487241496256]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 9000.000000000000000000000, RETROGRADE, 1, HIGH_X, [-6.091296696339854526794413, -4.938646281010617507388599, -3.859781891948891666173081], [4.655043984567266753235207, 5.825935629452337775433269, 4.598429631085498806093962]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 10000.00000000000000000000, RETROGRADE, 2, LOW_X, [1.123974988774327221766458, -5.176456636178528263724388, -4.179723636014430340881499], [6.753953675068045432781848, 0.4630785807204031380299940, 0.2515104010241088531887122]),
        ([6000.000000000000000000000, 2000.000000000000000000000, 1500.000000000000000000000], [1000.000000000000000000000, 5000.000000000000000000000, 4000.000000000000000000000], 10000.00000000000000000000, RETROGRADE, 2, HIGH_X, [-4.035748083857048107333788, -4.804045997438638101703554, -3.788327175015744044095110], [4.999954858543107061028173, 4.246994475797802909587117, 3.323478509863536140542716]),
        ([5974.699314991947176167741, 3201.585383727156113309320, 2466.011481298751732538221], [-7273.068797685116805951111, -3897.325949949376536096679, -3001.903552679229051136645], 3651.037669866628220916027, PROGRADE, 0, LOW_X, [-4.246083192866958655451662, 5.054541026513750495004759, 4.137507917954474353749989], [3.700729629914363094400195, -4.038269387197204949194431, -3.311122563759548711099668]),
    ];

    /// The ulps of `want` by which `got` misses it.
    fn ulps(got: f64, want: f64) -> f64 {
        (got - want).abs() / (want.abs() * f64::EPSILON)
    }

    /// (G1, on the numbers) `A` near a half turn keeps the digits its inputs carry. The cross product
    /// of two near-antiparallel doubles holds `|r1 × r2|` to `u/ε` relative and no better, so that is
    /// the floor of any form; Curtis's `sin Δν·√(r1 r2/(1 − cos Δν))` took `cos Δν` from a dot
    /// product and lost `u/ε²`. The bound is `4u/ε`; measured, the worst is 0.16 `u/ε` at `ε = 1e-3`.
    #[test]
    fn the_chord_parameter_near_a_half_turn_keeps_the_digits_its_inputs_carry() {
        for (r1, r2, want, eps) in HALF_TURN_CHORD_PARAMETER {
            let arc = Arc::new(vec3(r1), vec3(r2), 3600.0, MU_EARTH_KM, true)
                .expect("a half turn short of 1e-9 still has a plane");
            let rel = (arc.a - want).abs() / want;
            println!(
                "chord parameter at pi - {eps:.1e}: {:.2} u/eps",
                rel * eps / f64::EPSILON
            );
            assert!(
                rel <= 4.0 * f64::EPSILON / eps,
                "A at pi - {eps:e}: {} against {want}, {rel:.3e} relative",
                arc.a
            );
        }
    }

    /// (G2) Both solvers return the 100-digit arc near a half turn and on short arcs. Near a half
    /// turn the universal-variable solver refused every arc within about 1e-5 rad: `A` was off by
    /// 1.2e-6 there and the arc missed `r2` by more than `CLOSURE_TOLERANCE`. On a short arc
    /// `y(z) = r1 + r2 − √2·A·cos(√z/2)` cancelled and `v1` was off by 1.7e-6 at `Δν = 1e-5`.
    ///
    /// The bound is the conditioning of the rounded inputs, `u/δ` with `δ` the smaller of `Δν` and
    /// `π − Δν`: both solvers form `v1` from `r2 − f·r1` and a `g` proportional to `A`, which is
    /// `O(δ)` there. Measured, the worst is 1.41 `u/δ`; the bound is 16.
    #[test]
    fn the_arcs_near_a_half_turn_and_on_short_arcs_are_the_ones_lagrange_s_theorem_gives() {
        for (r1, r2, tof, v1, v2) in CONDITIONED_ARCS {
            let (r1, r2) = (vec3(r1), vec3(r2));
            let dnu = libm::atan2(r1.cross(r2).norm(), r1.dot(r2));
            let delta = dnu.min(core::f64::consts::PI - dnu);
            let scale = vec3(v1).norm();
            let universal = lambert(r1, r2, tof, MU_EARTH_KM, true)
                .unwrap_or_else(|| panic!("no arc at {delta:.1e} rad from a degenerate angle"));
            let izzo = lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false)
                .unwrap_or_else(|e| panic!("no Izzo arc at {delta:.1e} rad: {e}"));
            for (name, (got1, got2)) in [("universal", universal), ("izzo", izzo)] {
                let d = (got1 - vec3(v1)).norm().max((got2 - vec3(v2)).norm()) / scale;
                println!(
                    "{name} at {delta:.1e} rad: {:.3} u/delta",
                    d * delta / f64::EPSILON
                );
                assert!(
                    d <= 16.0 * f64::EPSILON / delta,
                    "{name} at {delta:.1e} rad: velocity off by {d:.3e} of |v1|"
                );
            }
        }
    }

    /// (G2) `T` at the doubles next to the parabola is Lagrange's equation at 100 digits. Evaluated
    /// as `a^{3/2}` times the bracket it was NaN at `x = 1` and lost `α³/6`'s worth of digits near
    /// it. The budget is 8 ulps per unit of the condition number `κ` of the equation's one
    /// subtraction (`derivations/lambert_izzo.gp`); measured, the worst is 2.47 `κ` ulps.
    #[test]
    fn the_time_of_flight_next_to_the_parabola_is_lagrange_s_at_100_digits() {
        let mut worst = 0.0_f64;
        for (x, lambda, want, kappa) in NEAR_PARABOLA {
            let got = izzo_tof(x, lambda, 0);
            let off = ulps(got, want);
            assert!(
                off <= 8.0 * kappa,
                "T(x = {x:e}, lambda = {lambda}): {got:e} against {want:e}, {off:.1} ulps"
            );
            worst = worst.max(off / kappa);
        }
        println!("T next to x = 1: worst {worst:.2} kappa ulps");
    }

    /// (G3) At `x = 1` the time of flight is its limit `⅔(1 − λ³)` (`derivations/lambert_izzo.mac`
    /// 6–8), for both signs of `λ`, and in seconds that is Euler's parabolic time. Two parabolas in
    /// closed form (`lambert_izzo.mac` 10–15, `µ = 1`, `p = 2`): the short way from periapsis
    /// `(1, 0, 0)` to `(0, 2, 0)` in `4√2/3` with `v1 = (0, √2, 0)`, `v2 = (−1, 1, 0)/√2`, and the
    /// long way, 233°, from `(0, −2, 0)` to `(−8, 6, 0)` in `40√2/3` with `v1 = (1, 1, 0)/√2`,
    /// `v2 = (−0.6, 0.2, 0)/√2`. Rounding the times to doubles moves the answers by 9.7e-18 and
    /// 2.6e-17 of `|v1|` (`derivations/lambert_izzo.gp` §6), so the bound is the solvers' own, 16
    /// ulps of `|v1|`; measured, 2.12.
    #[test]
    fn the_time_of_flight_at_x_equal_one_is_the_parabolic_time() {
        for lambda in [-0.999, -0.75, -0.5, -0.1, 0.0, 0.1, 0.5, 0.75, 0.999] {
            let want = 2.0 / 3.0 * (1.0 - lambda * lambda * lambda);
            assert_within_ulps(
                izzo_tof(1.0, lambda, 0),
                want,
                1,
                &format!("T(1, {lambda})"),
            );
        }
        let half = core::f64::consts::FRAC_1_SQRT_2;
        let cases = [
            (
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 2.0, 0.0),
                4.0 * core::f64::consts::SQRT_2 / 3.0,
                Vec3::new(0.0, core::f64::consts::SQRT_2, 0.0),
                Vec3::new(-half, half, 0.0),
            ),
            (
                Vec3::new(0.0, -2.0, 0.0),
                Vec3::new(-8.0, 6.0, 0.0),
                40.0 * core::f64::consts::SQRT_2 / 3.0,
                Vec3::new(half, half, 0.0),
                Vec3::new(-0.6 * half, 0.2 * half, 0.0),
            ),
        ];
        for (r1, r2, tof, v1, v2) in cases {
            let tol = 16.0 * f64::EPSILON * v1.norm();
            let izzo = lambert_multirev(r1, r2, tof, 1.0, true, 0, false)
                .unwrap_or_else(|e| panic!("no parabola to {r2:?}: {e}"));
            let universal = lambert(r1, r2, tof, 1.0, true).expect("a parabola is solvable");
            for (name, (got1, got2)) in [("izzo", izzo), ("universal", universal)] {
                let (d1, d2) = ((got1 - v1).norm(), (got2 - v2).norm());
                println!(
                    "{name} parabola to {r2:?}: {:.2} ulps of |v1|",
                    d1.max(d2) / (f64::EPSILON * v1.norm())
                );
                assert!(
                    d1 <= tol && d2 <= tol,
                    "{name} parabola to {r2:?}: v1 off by {d1:.3e}, v2 by {d2:.3e}, bound {tol:.3e}"
                );
            }
        }
    }

    /// (G2) Every multi-revolution arc is the root of Lagrange's equation at 100 digits on the
    /// branch the solver reports, and lands on `r2` (`derivations/lambert_izzo.gp` §5). Before
    /// this the solver had no derivation gate at all, and on the near-half-turn row it returned
    /// nothing, `1 − c/s` having rounded below zero.
    ///
    /// The bound: 64 ulps of `|v1|`, the universal solver's, on every row but the near-half-turn
    /// one, where the transfer plane is known to `u/ε` from the rounded inputs and the bound is
    /// `16u/ε`, `ε = 1.5e-9`. Measured, 18.8 ulps.
    #[test]
    fn every_multi_revolution_arc_is_the_one_lagrange_s_equation_gives_at_100_digits() {
        let mut worst = 0.0_f64;
        for (r1, r2, tof, prograde, n_rev, long_branch, v1, v2) in MULTIREV_ARCS {
            let (r1, r2) = (vec3(r1), vec3(r2));
            let (got1, got2) =
                lambert_multirev(r1, r2, tof, MU_EARTH_KM, prograde, n_rev, long_branch)
                    .unwrap_or_else(|e| {
                        panic!("no arc at tof {tof} s, N = {n_rev}, long {long_branch}: {e}")
                    });
            let d = (got1 - vec3(v1)).norm().max((got2 - vec3(v2)).norm()) / vec3(v1).norm();
            let sin = r1.cross(r2).norm() / (r1.norm() * r2.norm());
            let bound = if sin < 1e-6 {
                16.0 * f64::EPSILON / sin
            } else {
                worst = worst.max(d / f64::EPSILON);
                64.0 * f64::EPSILON
            };
            assert!(
                d <= bound,
                "tof {tof} s, prograde {prograde}, N = {n_rev}, long {long_branch}: velocity off \
                 by {d:.3e} of |v1|, bound {bound:.3e}"
            );
        }
        println!("multi-revolution arcs: worst {worst:.1} ulps of |v1|");
    }

    /// (G2) The least time of flight the solver reports is the minimum of Lagrange's equation at
    /// 100 digits, which the `.gp` also finds as the root of Izzo's `dT/dx` (`derivations/
    /// lambert_izzo.mac` 16–18 derive that derivative). `T` is flat at its minimum, so where the
    /// bisection on `dT/dx` stops is immaterial to it; the bound is 16 ulps; measured,
    /// 1.55 ulps.
    #[test]
    fn the_minimum_time_of_each_family_is_the_minimum_of_lagrange_s_equation() {
        for (geometry, prograde, n_rev, want) in MINIMUM_TIMES {
            let (r1, r2) = MULTIREV_GEOMETRIES[geometry];
            let got = lambert_multirev(r1, r2, 0.5 * want, MU_EARTH_KM, prograde, n_rev, false);
            let Err(LambertError::BelowMinimumTime { minimum }) = got else {
                panic!("half the minimum time solved: {got:?}");
            };
            println!(
                "t_min geometry {geometry} prograde {prograde} N {n_rev}: {:.2} ulps",
                ulps(minimum, want)
            );
            assert!(
                ulps(minimum, want) <= 16.0,
                "geometry {geometry}, prograde {prograde}, N = {n_rev}: minimum {minimum} s \
                 against {want} s"
            );
        }
    }

    /// (S2) A time of flight a part in 1e9 below a family's minimum is refused, and says so with the
    /// minimum; a part in 1e9 above it, both branches are solved and both arcs arrive. The scan this
    /// replaced sampled `x` every 1e-3, and the two roots a part in 1e9 above the minimum are about
    /// 1e-4 apart, so it found neither and returned nothing on both sides of the minimum alike.
    ///
    /// Above the minimum the two roots are `O(√δT)` from it, and where `T` is that flat `x` is fixed
    /// only to `√u`: each arc is then the exact arc of a time within a rounding of the one asked
    /// for, so it still arrives, while its `v1` can sit `√u` from the true root's. The bound on the
    /// arrival is the sweep's, 1e-12 of the arc's scale `|r2| + tof·|v1|`; measured, 1.6e-14.
    #[test]
    fn a_time_below_the_family_s_minimum_is_refused_and_one_above_is_solved() {
        let mut worst = 0.0_f64;
        for (geometry, prograde, n_rev, t_min) in MINIMUM_TIMES {
            let (r1, r2) = MULTIREV_GEOMETRIES[geometry];
            let below = t_min * (1.0 - 1e-9);
            let got = lambert_multirev(r1, r2, below, MU_EARTH_KM, prograde, n_rev, false);
            assert!(
                matches!(got, Err(LambertError::BelowMinimumTime { minimum }) if minimum > below),
                "geometry {geometry}, prograde {prograde}, N = {n_rev}: {got:?} below the minimum"
            );
            let above = t_min * (1.0 + 1e-9);
            let arcs = [false, true].map(|long| {
                let (v1, _) = lambert_multirev(r1, r2, above, MU_EARTH_KM, prograde, n_rev, long)
                    .unwrap_or_else(|e| {
                        panic!(
                            "geometry {geometry}, N = {n_rev}, long {long}: {e} above the minimum"
                        )
                    });
                let miss = (propagate(r1, v1, MU_EARTH_KM, above)
                    .expect("the universal Kepler solve settles")
                    .0
                    - r2)
                    .norm()
                    / (r2.norm() + above * v1.norm());
                worst = worst.max(miss);
                assert!(
                    miss <= 1e-12,
                    "geometry {geometry}, N = {n_rev}, long {long}: missed r2 by {miss:.3e}"
                );
                v1
            });
            assert!(
                (arcs[0] - arcs[1]).norm() > 0.0,
                "geometry {geometry}, N = {n_rev}: the two branches returned one arc"
            );
        }
        println!("a part in 1e9 above the minimum: worst miss {worst:.3e} of scale");
    }

    /// (G2, closure) Every arc the multi-revolution solver returns over a sweep of both geometries of
    /// `MULTIREV_GEOMETRIES` and `GEOMETRIES`, `N` from 0 to 3, both senses and both branches, at
    /// times of flight from 1.01 to 4 times each family's minimum, is carried onto `r2` by the gated
    /// universal propagator, `crate::kepler::propagate`. The bound is 1e-12 of the arc's scale
    /// `|r2| + tof·|v1|`; measured, 4.7e-15.
    #[test]
    fn every_multi_revolution_arc_closes_through_the_universal_propagator() {
        let mut worst = 0.0_f64;
        let mut flown = 0usize;
        for (r1, r2) in MULTIREV_GEOMETRIES.into_iter().chain(GEOMETRIES) {
            for prograde in [true, false] {
                for n_rev in 0..=3 {
                    // The family's minimum, or for N = 0 an hour, as the unit of the sweep.
                    let unit =
                        match lambert_multirev(r1, r2, 1.0, MU_EARTH_KM, prograde, n_rev, false) {
                            Err(LambertError::BelowMinimumTime { minimum }) => minimum,
                            _ => 3600.0,
                        };
                    for k in [1.01, 1.1, 1.5, 2.0, 4.0] {
                        let tof = k * unit;
                        for long in [false, true] {
                            let (v1, v2) =
                                lambert_multirev(r1, r2, tof, MU_EARTH_KM, prograde, n_rev, long)
                                    .unwrap_or_else(|e| {
                                        panic!("no arc at {k} t_min, N = {n_rev}, long {long}: {e}")
                                    });
                            let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof)
                                .expect("the universal Kepler solve settles");
                            let scale = r2.norm() + tof * v1.norm();
                            let miss = (rf - r2).norm() / scale;
                            let dv = (vf - v2).norm() / v1.norm();
                            worst = worst.max(miss);
                            flown += 1;
                            assert!(
                                miss <= 1e-12 && dv <= 1e-9,
                                "{k} t_min, prograde {prograde}, N = {n_rev}, long {long}: \
                                 missed r2 by {miss:.3e} of scale, v2 by {dv:.3e} of |v1|"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(
            flown,
            4 * 2 * 4 * 5 * 2,
            "the sweep stopped covering its cases"
        );
        println!("{flown} multi-revolution arcs: worst miss {worst:.3e} of scale");
    }
}
