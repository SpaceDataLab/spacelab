//! Lambert's problem: the transfer arc between two positions in a given time.
//!
//! Given `r1`, `r2` and a time of flight, find the two-body velocities that connect them. This is the
//! core transfer primitive — the arc of a Mars or lunar injection, the leg of a porkchop plot. The
//! universal-variable formulation (Bate–Mueller–White; Curtis, *Orbital Mechanics for Engineering
//! Students*, Algorithm 5.2) reuses the Stumpff functions of [`crate::kepler`] and covers the single
//! revolution across ellipse and hyperbola through one iteration on the universal variable `z`.
//! Multi-revolution transfers (Izzo's Householder method) are a later extension.

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
/// that level: over the 25 270-case sweep of `examples/lambert_longway_sweep.rs` the solver without
/// it answers 43 near-whole-revolution arcs with a miss above the tolerance, and with it refuses
/// those 43 and returns nothing worse than 9.3e-7. Callers who need them should use
/// [`lambert_multirev`], whose Lancaster–Blanchard variable does not crowd at one revolution.
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
    let r1n = r1.norm();
    let r2n = r2.norm();
    let cross_z = r1.x * r2.y - r1.y * r2.x;
    let cos_dnu = (r1.dot(r2) / (r1n * r2n)).clamp(-1.0, 1.0);
    let acos = libm::acos(cos_dnu);
    let tau = core::f64::consts::TAU;
    // Transfer angle, resolved to (0, 2π) by the sense of the out-of-plane component.
    let dnu = if prograde {
        if cross_z >= 0.0 { acos } else { tau - acos }
    } else if cross_z < 0.0 {
        acos
    } else {
        tau - acos
    };

    let sin_dnu = libm::sin(dnu);
    // A = 0 at a 0° or 180° transfer: the plane is undefined and the method is singular.
    if sin_dnu.abs() < 1e-12 {
        return None;
    }
    let arc = Arc {
        r1n,
        r2n,
        a: sin_dnu * libm::sqrt(r1n * r2n / (1.0 - cos_dnu)),
        sqmu: libm::sqrt(mu),
        tof,
    };
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
    let miss = (propagate(r1, v1, mu, tof).0 - r2).norm();
    if !(miss.is_finite() && miss <= CLOSURE_TOLERANCE * r2n) {
        return None;
    }
    Some((v1, v2))
}

/// A single-revolution arc as the universal-variable iteration sees it: the two radii, the chord
/// parameter `A = sin Δν · √(r1 r2 / (1 − cos Δν))`, `√µ`, and the time of flight to match.
///
/// The three functions below share all five values, which is why they are a type rather than three
/// signatures. They were closures inside [`lambert`] and could only be reached through it — and
/// [`Self::dfdz`] cannot be reached through it at all, since the iteration runs to a fixed point and
/// the derivative decides only how fast `z` arrives, never where.
#[derive(Clone, Copy)]
struct Arc {
    r1n: f64,
    r2n: f64,
    a: f64,
    sqmu: f64,
    tof: f64,
}

impl Arc {
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
    /// back: holding everything else fixed, `examples/lambert_longway_sweep.rs` refuses 227 of its
    /// 25 270 arcs with the Curtis quotient and 111 with the form below.
    fn y(self, z: f64) -> f64 {
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
/// full revolutions, using the Lagrange form (valid away from `x = 1`).
fn izzo_tof(x: f64, lambda: f64, n_rev: usize) -> f64 {
    let a = 1.0 / (1.0 - x * x);
    if a > 0.0 {
        // Ellipse.
        let alpha = 2.0 * libm::acos(x);
        let beta0 = 2.0 * libm::asin(libm::sqrt(lambda * lambda * (1.0 - x * x)));
        let beta = if lambda < 0.0 { -beta0 } else { beta0 };
        a * libm::sqrt(a)
            * ((alpha - libm::sin(alpha)) - (beta - libm::sin(beta))
                + core::f64::consts::TAU * n_rev as f64)
            / 2.0
    } else {
        // Hyperbola (n_rev = 0 only).
        let alpha = 2.0 * libm::acosh(x);
        let beta0 = 2.0 * libm::asinh(libm::sqrt(-lambda * lambda * (1.0 - x * x)));
        let beta = if lambda < 0.0 { -beta0 } else { beta0 };
        -a * libm::sqrt(-a) * ((beta - libm::sinh(beta)) - (alpha - libm::sinh(alpha))) / 2.0
    }
}

/// Solve Lambert's problem allowing `n_rev` complete revolutions (Izzo formulation).
///
/// For `n_rev = 0` this is a single-revolution transfer (matching [`lambert`]); for `n_rev ≥ 1` two
/// solutions exist and `long_branch` selects the higher-`x` (shorter, higher-energy) branch when
/// `true`. `prograde` and the geometry fix the transfer sense. Returns `None` on a degenerate
/// geometry, or when no `n_rev`-revolution transfer exists for the given time of flight.
#[must_use]
pub fn lambert_multirev(
    r1: Vec3,
    r2: Vec3,
    tof: f64,
    mu: f64,
    prograde: bool,
    n_rev: usize,
    long_branch: bool,
) -> Option<(Vec3, Vec3)> {
    let r1n = r1.norm();
    let r2n = r2.norm();
    let c = Vec3::new(r2.x - r1.x, r2.y - r1.y, r2.z - r1.z);
    let cn = c.norm();
    let s = (r1n + r2n + cn) / 2.0;
    // Only the chord needs a floor. The triangle inequality gives `r1n + r2n ≥ cn`, so
    // `s = (r1n + r2n + cn)/2 ≥ cn`, and a second test on `s` could never fire once this one has
    // passed — it read as a guard while guarding nothing.
    if cn < 1e-12 {
        return None;
    }
    let i_r1 = r1.scale(1.0 / r1n);
    let i_r2 = r2.scale(1.0 / r2n);
    let cross = r1.cross(r2);
    let cross_n = cross.norm();
    if cross_n < 1e-12 {
        return None; // 0° or 180° transfer
    }
    let i_h = cross.scale(1.0 / cross_n);
    let mut lambda = libm::sqrt(1.0 - cn / s);
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

    let t_target = libm::sqrt(2.0 * mu / (s * s * s)) * tof;

    // Root-find x. n_rev=0: x ∈ (−1, ∞); n_rev≥1: x ∈ (−1, 1) with two roots. A coarse scan brackets
    // the sign change of T(x) − T_target (T decreases in x), then bisection refines.
    let hi = if n_rev == 0 { 20.0 } else { 1.0 - 1e-6 };
    let lo = -1.0 + 1e-6;
    let steps = 2000;
    let f = |x: f64| izzo_tof(x, lambda, n_rev) - t_target;
    let mut brackets = [(0.0, 0.0); 2];
    let mut nb = 0;
    let mut xa = lo;
    let mut fa = f(xa);
    for k in 1..=steps {
        let xb = lo + (hi - lo) * f64::from(k) / f64::from(steps);
        let fb = f(xb);
        if fa * fb < 0.0 && nb < 2 {
            brackets[nb] = (xa, xb);
            nb += 1;
        }
        xa = xb;
        fa = fb;
    }
    if nb == 0 {
        return None; // no transfer for this tof / n_rev
    }
    let (mut a, mut b) = if n_rev >= 1 && long_branch && nb == 2 {
        brackets[1]
    } else {
        brackets[0]
    };
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
    Some((v1, v2))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::excessive_precision)]
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

    /// `T(x, λ, N)` of the Lancaster–Blanchard form, evaluated at 60 decimal digits with `mpmath`,
    /// with the ulp budget each entry is entitled to. The table spans both conics and both signs of
    /// `λ`. The budget is one ulp per digit the Lagrange form cancels away: at `x = 1.02` the
    /// hyperbolic arc is near-parabolic, `sinh α − α` loses `sinh α / (sinh α − α) ≈ 38` in the
    /// leading term alone, and no rearrangement of *this* form recovers it.
    const IZZO_TOF: [(f64, f64, usize, f64, i64); 18] = [
        (0.0, 0.5, 0, 1.480_210_253_088_817_069_536_076, 8),
        (-0.9, 0.3, 0, 37.206_019_284_720_004_453_133_41, 8),
        (0.5, 0.9, 0, 0.321_388_255_499_988_662_800_457_6, 8),
        (0.75, 0.45, 0, 0.720_803_136_724_604_010_795_098_4, 8),
        (-0.25, 0.8, 0, 1.839_046_851_196_585_578_166_719, 8),
        (0.3, 0.2, 1, 4.742_432_784_142_829_841_881_213, 8),
        (-0.5, 0.7, 2, 13.304_338_152_135_505_434_784_57, 8),
        (0.0, -0.5, 0, 1.661_382_400_500_976_168_926_567, 8),
        (-0.7, -0.8, 0, 8.196_593_203_105_028_973_287_010, 8),
        (0.4, -0.3, 1, 5.128_706_840_676_636_927_341_485, 8),
        (0.6, -0.95, 3, 20.002_821_794_501_325_325_060_31, 8),
        (1.05, 0.2, 0, 0.641_895_856_066_700_313_464_885_6, 8),
        (1.5, 0.4, 0, 0.470_995_876_202_493_818_440_098_0, 8),
        (3.0, 0.9, 0, 0.062_954_358_790_311_366_604_606_37, 8),
        (12.0, 0.6, 0, 0.053_223_462_297_644_230_998_104_63, 8),
        (1.5, -0.4, 0, 0.551_695_512_712_693_257_113_715_7, 8),
        (2.5, -0.85, 0, 0.582_361_552_596_315_904_638_422_8, 8),
        (1.02, -0.75, 0, 0.938_112_673_030_687_534_538_148_1, 64),
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

    /// The departure velocity, propagated forward by the time of flight, must arrive at `r2` — a
    /// self-consistency check that closes Lambert against the independent Kepler propagator.
    #[test]
    fn the_solution_arc_reaches_the_target() {
        let r1 = Vec3::new(5000.0, 10000.0, 2100.0);
        let r2 = Vec3::new(-14600.0, 2500.0, 7000.0);
        let tof = 3600.0;
        let (v1, v2) = lambert(r1, r2, tof, MU_EARTH_KM, true).expect("solvable");
        let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof);
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
        let (rf, _) = propagate(r1, v1, MU_EARTH_KM, tof);
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
                    let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof);
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
                    let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof);
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

    /// The Lancaster–Blanchard time of flight against a 60-digit evaluation of its closed form, on
    /// the ellipse and the hyperbola and for both signs of `λ`.
    #[test]
    fn the_nondimensional_time_of_flight_matches_its_closed_form_on_both_conics() {
        for &(x, lambda, n_rev, want, ulps) in &IZZO_TOF {
            assert_within_ulps(
                izzo_tof(x, lambda, n_rev),
                want,
                ulps,
                &format!("T(x = {x}, lambda = {lambda}, N = {n_rev})"),
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
                        .unwrap_or_else(|| panic!("no arc at tof {tof} s, prograde {prograde}"));
                    let (v1u, _) = lambert(r1, r2, tof, MU_EARTH_KM, prograde).expect("uv arc");
                    let d = (v1 - v1u).norm();
                    assert!(
                        d < 1e-10 * v1.norm(),
                        "Izzo differs from universal-variable by {d:.3e} km/s at tof {tof} s"
                    );
                    let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof);
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
                let (rf, _) = propagate(r1, v, MU_EARTH_KM, tof);
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
    /// revolution at all. At 60 000 s the two departure velocities agree to 1.9e-12 relative and the
    /// arc closes to 4.2e-9 of the radius; at 600 000 s they agree to 1.9e-12 and it closes to
    /// 2.2e-7. At 6e6 s — a 714 000 km semi-major axis, `e ≈ 0.99` — the closure test refuses,
    /// which is the third case above and the honest end of what `z` can carry.
    #[test]
    fn the_long_way_round_is_solved_where_the_root_crowds_the_whole_revolution() {
        let r1 = Vec3::new(7000.0, 0.0, 0.0);
        let dnu = 5f64.to_radians();
        let r2 = Vec3::new(7000.0 * dnu.cos(), 7000.0 * dnu.sin(), 0.0);
        for (tof, bound) in [(60_000.0, 1e-8), (600_000.0, 5e-7)] {
            let (v1, _) = lambert(r1, r2, tof, MU_EARTH_KM, false)
                .unwrap_or_else(|| panic!("no long-way arc at tof {tof} s"));
            let miss = (propagate(r1, v1, MU_EARTH_KM, tof).0 - r2).norm() / r2.norm();
            assert!(
                miss < bound,
                "long-way arc missed r2 by {miss:.3e} at {tof} s"
            );
            let (w1, _) = lambert_multirev(r1, r2, tof, MU_EARTH_KM, false, 0, false)
                .unwrap_or_else(|| panic!("no Izzo arc at tof {tof} s"));
            let d = (w1 - v1).norm() / v1.norm();
            assert!(
                d < 1e-11,
                "the two solvers differ by {d:.3e} at tof {tof} s"
            );
        }

        // The third of the ticket's cases is past what `z` can resolve, and is refused rather than
        // answered. Izzo still solves it, which is the whole reason the refusal is the right answer
        // and not a gap.
        assert!(
            lambert(r1, r2, 6.0e6, MU_EARTH_KM, false).is_none(),
            "6e6 s is past the resolution of z and must be refused"
        );
        let (w1, _) = lambert_multirev(r1, r2, 6.0e6, MU_EARTH_KM, false, 0, false)
            .expect("Izzo solves what the universal variable cannot");
        let miss = (propagate(r1, w1, MU_EARTH_KM, 6.0e6).0 - r2).norm() / r2.norm();
        assert!(miss < 1e-9, "the Izzo arc missed r2 by {miss:.3e} at 6e6 s");
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
                        .unwrap_or_else(|| panic!("no Izzo arc at tof {tof} s"));
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
        let (rf, _) = propagate(r1, v1, MU_EARTH_KM, tof);
        let miss = (rf - r2).norm();
        assert!(
            miss < 1e-9 * (r2.norm() + tof * v1.norm()),
            "missed r2 by {miss:.3e} km"
        );
    }

    /// The Izzo solver refuses geometries with no transfer plane and revolution counts the time of
    /// flight cannot support. The near-coincident case is the one the chord guard exists for: the
    /// positions are 0.1 pm apart, which the cross product alone does not resolve.
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
            assert!(
                lambert_multirev(r1, r2, 3600.0, MU_EARTH_KM, true, n_rev, false).is_none(),
                "expected no solution for the {name} case"
            );
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
        let (rf, vf) = propagate(r1, v1, MU_EARTH_KM, tof);
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
                assert!(
                    lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false).is_none(),
                    "izzo solved tof = {tof}"
                );
            }
        }
    }

    /// The `Arc` the public solver builds for one of the sweep geometries, so the iteration and its
    /// derivative can be reached directly. Transcribed from [`lambert`]'s prologue.
    fn arc_of(r1: Vec3, r2: Vec3, tof: f64, prograde: bool) -> Arc {
        let (r1n, r2n) = (r1.norm(), r2.norm());
        let cross_z = r1.x * r2.y - r1.y * r2.x;
        let cos_dnu = (r1.dot(r2) / (r1n * r2n)).clamp(-1.0, 1.0);
        let acos = libm::acos(cos_dnu);
        let tau = core::f64::consts::TAU;
        let dnu = if prograde {
            if cross_z >= 0.0 { acos } else { tau - acos }
        } else if cross_z < 0.0 {
            acos
        } else {
            tau - acos
        };
        Arc {
            r1n,
            r2n,
            a: libm::sin(dnu) * libm::sqrt(r1n * r2n / (1.0 - cos_dnu)),
            sqmu: libm::sqrt(MU_EARTH_KM),
            tof,
        }
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
                        let arc = arc_of(r1, r2, tof, pro);
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
                lambert_multirev(r1, r2, tof, MU_EARTH_KM, true, 0, false).is_some(),
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
}
