//! WHFast — the Wisdom–Holman symplectic map with an exact Kepler drift (Wisdom & Holman 1991; Rein
//! & Tamayo 2015).
//!
//! # The splitting idea
//!
//! A perturbed two-body Hamiltonian splits as `H = H_Kepler + H_pert`, where the Keplerian part
//! (the central point mass) dominates and the perturbation (`J₂`, third bodies, SRP, …) is small.
//! WHFast integrates the *dominant* part **exactly** — the Kepler drift advances the osculating orbit
//! by the analytic universal-variable solution, not a truncated series — and handles only the small
//! perturbation with a symplectic kick. The leapfrog composition drift(½h)–kick(h)–drift(½h) is
//! therefore second order in the step, but its error is proportional to the *perturbation* strength,
//! not the full acceleration: for a nearly-Keplerian orbit it is dramatically more accurate per step
//! than a general integrator, and being symplectic its energy error stays bounded over long times.
//!
//! # The Kepler drift
//!
//! The drift solves the universal Kepler equation `√μ·Δt = r₀ X c₁(z) + σ₀ X² c₂(z) + (1−αr₀) X³ c₃(z)`
//! for the universal anomaly `X` (Newton's method; the derivative is exactly the radius `r(X)`), with
//! `α = 1/a`, `σ₀ = r₀·v₀/√μ`, `z = αX²`. The Stumpff functions `c₀…c₃` are evaluated by argument
//! reduction (quarter until small, sum the series, double back up) for accuracy at small `z`. The
//! new state follows from the Lagrange `f`/`g` coefficients. `derivations/whfast_stumpff.mac` verifies
//! the Stumpff closed forms, the `c_k = 1/k! − z c_{k+2}` recurrence and the Danby doubling; the same
//! file's companion derivation confirms the `f`/`g` map is symplectic (`f ġ − ḟ g = 1`), which the
//! test re-checks numerically.
//!
//! The drift is valid on every conic, not only the bound ones: `z = αX²` simply goes negative for a
//! hyperbola and the Stumpff series carries the `cosh`/`sinh` branch without a separate code path.
//! Newton's *starting* point does need one, since `α√μΔt` points the wrong way when `α < 0`, and
//! near `e = 1` no closed form is any good at all — so `solve_universal_kepler` treats the guess as
//! a speed hint and falls back on a bracket, which the monotonicity of the equation makes proof
//! against a bad one. `kepler_drift_is_exact_on_every_conic` measures the result from `e = 0.1` to
//! `e = 3`, `kepler_drift_carries_a_long_hyperbolic_arc` out to two hundred `2π√(|a|³/μ)`,
//! `kepler_drift_solves_the_near_parabolic_band` across `e = 1 ± 1e-6`, and
//! `the_drift_runs_backwards_to_where_it_came_from` in the one direction nothing else exercises.

// Stumpff/Kepler math uses many single-letter symbols; the step count casts a small integer.
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::doc_markdown
)]

use std::vec::Vec;

use crate::driver::{HermiteDense, IntegrationError, Trajectory, step_count};
use crate::forces::Acceleration;

/// The Stumpff argument `z = α X²`, restricted to the values the argument reduction terminates on.
///
/// A newtype rather than a bare `f64` because the reduction loop scales its argument instead of
/// counting down: `while |z| > 0.1 { z *= 0.25 }` brings every *finite* `z` under the series' range
/// in at most `⌈log₄(f64::MAX/0.1)⌉ = 514` passes and an infinite one **never**, since `∞ · 0.25` is
/// `∞`. In release that is a live-lock; in debug the counter `n` overflows after `2³²` passes, about
/// twenty seconds later. Same shape as [`crate::earth_radiation::AngularResolution`] and
/// `coverage::root::advancing_step`: a loop whose exit test is a magnitude the body only rescales.
///
/// `z = α X²` is infinite exactly when the state has no osculating orbit to advance —
/// `α = 2/r₀ − v²/μ` is `+∞` for a satellite at the centre of attraction and `−∞` as `μ` vanishes —
/// and there is nothing to return there: `c₀(z) = cos √z` has no limit at infinity. `is_finite` also
/// rejects `NaN`, which the raw loop exits on its first test and answers with a silent `NaN`.
#[derive(Debug, Clone, Copy)]
struct StumpffArg(f64);

impl StumpffArg {
    fn new(z: f64) -> Option<Self> {
        z.is_finite().then_some(Self(z))
    }
}

/// Stumpff functions `c₀, c₁, c₂, c₃` at `z`, by argument reduction and the Danby doubling.
fn stumpff(arg: StumpffArg) -> (f64, f64, f64, f64) {
    let z = arg.0;
    // Reduce |z| below the series' comfortable range.
    let mut n = 0u32;
    let mut zr = z;
    while zr.abs() > 0.1 {
        zr *= 0.25;
        n += 1;
    }
    // Series c₂ = Σ (−zr)^k/(2k+2)!, c₃ = Σ (−zr)^k/(2k+3)!; eight terms saturate f64 for |zr| ≤ 0.1.
    let (mut c2, mut c3) = (0.5, 1.0 / 6.0);
    let (mut t2, mut t3) = (0.5, 1.0 / 6.0);
    for k in 1..=8 {
        let kk = f64::from(k);
        t2 *= -zr / ((2.0 * kk + 1.0) * (2.0 * kk + 2.0));
        t3 *= -zr / ((2.0 * kk + 2.0) * (2.0 * kk + 3.0));
        c2 += t2;
        c3 += t3;
    }
    // Double the argument back up n times: c₂(4u)=c₁(u)²/2, c₃(4u)=(c₂(u)+c₀(u)c₃(u))/4.
    for _ in 0..n {
        let c1 = 1.0 - zr * c3;
        let c0 = 1.0 - zr * c2;
        c3 = (c2 + c0 * c3) * 0.25;
        c2 = c1 * c1 * 0.5;
        zr *= 4.0;
    }
    let c1 = 1.0 - z * c3;
    let c0 = 1.0 - z * c2;
    (c0, c1, c2, c3)
}

/// Newton passes taken from the conic guess before the safeguarded solver is set up.
///
/// Chosen from a matrix of 532 `(e, start, span)` rows — `e` from 0 to 8 including eight steps
/// within a millionth of 1, four starting points per orbit, spans from a ten-thousandth to two
/// hundred of `2π√(|a|³/μ)` — where the median solve costs 4 evaluations of `F` and the 90th
/// percentile 6, so 16 leaves the ordinary orbit well clear of the safeguard while still handing
/// over early enough that a hard row does not spend its budget crawling.
/// [`the_solve_costs_what_the_budgets_were_set_from`] holds the in-repo figures: at most 9
/// evaluations over 84 ordinary rows.
const FAST_PASSES: usize = 16;

/// Cap on each of the safeguarded solver's two loops — the geometric bracket search, and the
/// bisection-backed Newton inside it.
///
/// The band that needs the safeguard costs 51 evaluations of `F` at worst and 24 at the median, so
/// 128 is not a tuned number but a structural one: a bisection halves the bracket every pass a
/// Newton step leaves it, and 128 of them exhaust an `f64` mantissa nearly two and a half times
/// over, while 128 doublings carry the bracket search across `2¹²⁸` in either direction.
const SAFE_PASSES: usize = 128;

/// Newton stops when its own correction falls this far below `X`. Held in one place because the fast
/// and the safeguarded pass must not converge to different answers.
const STEP_TOLERANCE: f64 = 1e-13;

/// A converged universal anomaly, and what it cost to reach it.
///
/// The count is not decoration. Everything the safeguarded solver does below — the closed-form
/// seed, the geometric bracket walk, the choice between a Newton step and a bisection — lands on the
/// *same* root whatever is done to it, because that is exactly what bracketing a monotone function
/// guarantees. Twenty-eight mutations of that code were injected one at a time, and every accuracy
/// assertion in this module passed under all twenty-eight. Only the cost separates them, so only
/// the cost can hold them, and [`the_solve_costs_what_the_budgets_were_set_from`] is where the
/// numbers quoted on [`FAST_PASSES`] and [`SAFE_PASSES`] stop being a comment and become a test —
/// it now refuses fourteen of the twenty-eight.
#[derive(Debug, Clone, Copy)]
struct Solved {
    x: f64,
    /// Only the tests read this, so outside them the lint has to be told; the type's doc is the
    /// argument for why it exists at all.
    #[cfg_attr(not(test), allow(dead_code))]
    evaluations: u32,
}

/// Solve the universal Kepler equation `F(X) = r₀Xc₁ + σ₀X²c₂ + X³c₃ − √μΔt = 0` for `X`, or return
/// `None` if both solvers are exhausted.
///
/// # Two facts about `F` do all the work
///
/// Its derivative is the orbital radius `r(X) = r₀c₀ + σ₀Xc₁ + X²c₂`, which is positive wherever the
/// orbit exists — so `F` is strictly increasing and has **exactly one** root. And `F(0) = −√μΔt`, so
/// that root carries the sign of `Δt`. Together they turn the initial guess from a correctness
/// concern into a performance one: a guess on the wrong side is provably wrong and is discarded, and
/// a sign change between two abscissae *brackets* the root, which makes bisection a fallback that
/// cannot fail rather than a heuristic that might.
///
/// That matters because no closed-form guess is good near `e = 1`. The circular form `α√μΔt` points
/// the wrong way on a hyperbola, where α is negative — Newton then has to crawl up the exponential
/// branch, 40 passes at `e = 3` over five of the orbit's own `2π√(|a|³/μ)` and more than 64 at
/// twenty of them, which this used to answer with a position 1.3e26 radii out. Vallado's hyperbolic
/// form (*Fundamentals of Astrodynamics and Applications*, Algorithm 8) fixes that but degenerates
/// at the parabola, `√(−a) → ∞` against `ln(·) → −∞`, and it can return the wrong sign outright when
/// the state is far from periapsis. Vallado's third branch, Barker's equation, was measured as a
/// guess here and is *worse* on this matrix, not better: the worst pass count rises from 136 to 198
/// and the 99th percentile from 17 to 164, because a parabolic guess is poor on everything that is
/// not a parabola and the near-parabolic band is not where the cost lives. The bracket needs no such
/// case analysis.
fn solve_universal_kepler(
    r0: f64,
    sigma0: f64,
    alpha: f64,
    sqrt_mu: f64,
    dt: f64,
) -> Option<Solved> {
    let evaluations = core::cell::Cell::new(0u32);
    // A non-finite value is not an error here: on a hyperbola the Stumpff series is cosh/sinh and
    // overflows for a large enough X, and "overflowed" and "past the root" are the same statement.
    let f_and_radius = |x: f64| {
        evaluations.set(evaluations.get() + 1);
        let (c0, c1, c2, c3) = stumpff(StumpffArg::new(alpha * x * x)?);
        let f = r0 * x * c1 + sigma0 * x * x * c2 + x * x * x * c3 - sqrt_mu * dt;
        let radius = r0 * c0 + sigma0 * x * c1 + x * x * c2;
        (f.is_finite() && radius.is_finite()).then_some((f, radius))
    };
    // `F` increases through zero, so beyond the root `F` and `Δt` share a sign; an overflow is only
    // reachable beyond it.
    let past_root = |probe: f64| match f_and_radius(probe) {
        None => true,
        Some((f, _)) => (f > 0.0) == (dt > 0.0),
    };

    let seed = if alpha < 0.0 {
        let sign = if dt >= 0.0 { 1.0 } else { -1.0 };
        let scale = (-1.0 / alpha).sqrt();
        let denominator = sigma0 + sign * scale * (1.0 - r0 * alpha);
        sign * scale * libm::log(-2.0 * sqrt_mu * alpha * dt / denominator)
    } else {
        alpha * sqrt_mu * dt
    };
    // `√μΔt/r₀` is Newton's own step from the origin: exact at `z = 0`, and on the right side of it
    // by construction, which is what makes it the repair for a guess that is not.
    let seed = if seed.is_finite() && seed != 0.0 && (seed > 0.0) == (dt > 0.0) {
        seed
    } else {
        sqrt_mu * dt / r0
    };

    let mut x = seed;
    for _ in 0..FAST_PASSES {
        let Some((f, radius)) = f_and_radius(x) else {
            break;
        };
        let dx = f / radius;
        x -= dx;
        if dx.abs() <= STEP_TOLERANCE * x.abs().max(1.0) {
            return Some(Solved {
                x,
                evaluations: evaluations.get(),
            });
        }
    }

    // Bracket the root by walking the seed geometrically until it crosses: `near` stays on the
    // origin's side of the root and `far` past it, so one halving or doubling of whichever the seed
    // turned out to be lands the other within a factor of two.
    let (mut near, mut far) = (seed, seed);
    let mut bracketed = false;
    if past_root(seed) {
        for _ in 0..SAFE_PASSES {
            far = near;
            near *= 0.5;
            if !past_root(near) {
                bracketed = true;
                break;
            }
        }
    } else {
        for _ in 0..SAFE_PASSES {
            near = far;
            far *= 2.0;
            if past_root(far) {
                bracketed = true;
                break;
            }
        }
    }
    if !bracketed {
        return None;
    }

    let (mut lo, mut hi) = if dt > 0.0 { (near, far) } else { (far, near) };
    let mut x = f64::midpoint(lo, hi);
    for _ in 0..SAFE_PASSES {
        let Some((f, radius)) = f_and_radius(x) else {
            // Only the far side can overflow, so this end of the bracket is the one to pull in.
            if dt > 0.0 {
                hi = x;
            } else {
                lo = x;
            }
            x = f64::midpoint(lo, hi);
            continue;
        };
        if f > 0.0 {
            hi = x;
        } else {
            lo = x;
        }
        let mut dx = f / radius;
        let mut next = x - dx;
        if !(next > lo && next < hi) {
            next = f64::midpoint(lo, hi);
            dx = x - next;
        }
        x = next;
        if dx.abs() <= STEP_TOLERANCE * x.abs().max(1.0) {
            return Some(Solved {
                x,
                evaluations: evaluations.get(),
            });
        }
    }
    None
}

/// Advance a two-body orbit about `mu` by `dt` in place, exactly, via the universal-variable solver.
/// Shared with [`crate::saba`] and [`crate::mercurius`], whose splittings use the same exact drift.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the state has no osculating orbit to advance — a position
/// at the centre of attraction, a non-positive `mu`, or a non-finite `r` or `v`. See [`StumpffArg`]
/// for why this is an error and not a value.
/// [`IntegrationError::StageSolveFailed`] at `t` if [`solve_universal_kepler`] exhausts both its
/// solvers; the state is left untouched rather than advanced along an unconverged `X`. No row of the
/// matrix behind [`FAST_PASSES`] reaches this, `e` within a millionth of 1 included.
pub(crate) fn kepler_drift(
    mu: f64,
    t: f64,
    r: &mut [f64; 3],
    v: &mut [f64; 3],
    dt: f64,
) -> Result<(), IntegrationError> {
    if dt == 0.0 {
        return Ok(());
    }
    let r0 = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let v2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let rv = r[0] * v[0] + r[1] * v[1] + r[2] * v[2];
    let sqrt_mu = mu.sqrt();
    let alpha = 2.0 / r0 - v2 / mu; // 1/a; > 0 for a bound orbit
    let sigma0 = rv / sqrt_mu;
    // Every degenerate state the drift refuses fails here, at the boundary, rather than as a NaN
    // wandering into the iteration: r0 = 0 is a position at the centre of attraction, sqrt_mu = 0 or
    // NaN a non-positive mu, and a non-finite r or v poisons r0 or v2 and so alpha.
    if !(r0 > 0.0 && sqrt_mu > 0.0 && alpha.is_finite() && sigma0.is_finite()) {
        return Err(IntegrationError::BadArguments);
    }

    let x = solve_universal_kepler(r0, sigma0, alpha, sqrt_mu, dt)
        .ok_or(IntegrationError::StageSolveFailed { t })?
        .x;

    let z = StumpffArg::new(alpha * x * x).ok_or(IntegrationError::BadArguments)?;
    let (c0, c1, c2, c3) = stumpff(z);
    let r_new = r0 * c0 + sigma0 * x * c1 + x * x * c2;
    let f = 1.0 - x * x / r0 * c2;
    let g = dt - x * x * x / sqrt_mu * c3;
    let fdot = -sqrt_mu / (r_new * r0) * x * c1;
    let gdot = 1.0 - x * x / r_new * c2;
    for i in 0..3 {
        let (ri, vi) = (r[i], v[i]);
        r[i] = f * ri + g * vi;
        v[i] = fdot * ri + gdot * vi;
    }
    Ok(())
}

/// Integrate a satellite about a central mass `mu` under a small perturbation `pert` with WHFast.
///
/// `pert` is the **non-central** acceleration (`J₂`, third bodies, SRP, …) — everything except the
/// central point mass, which the Kepler drift already carries exactly. The map is drift(½h)–kick(h)–
/// drift(½h) per step; the step is adjusted so an integer number spans `[t0, t1]`.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span, step or `mu` is not positive, or if the state
/// stops having an osculating orbit to drift along — see `kepler_drift`.
pub fn integrate_whfast<A: Acceleration>(
    mu: f64,
    pert: &A,
    r0: &[f64; 3],
    v0: &[f64; 3],
    t0: f64,
    t1: f64,
    dt: f64,
) -> Result<([f64; 3], [f64; 3]), IntegrationError> {
    if !(t1 > t0 && dt > 0.0 && mu > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, dt, 1)?;
    let step = (t1 - t0) / n_steps as f64;

    let mut r = *r0;
    let mut v = *v0;
    let mut t = t0;
    for _ in 0..n_steps {
        whfast_step(mu, pert, &mut r, &mut v, t, step)?;
        t += step;
    }
    Ok((r, v))
}

/// One WHFast drift–kick–drift step of size `step` in place.
fn whfast_step<A: Acceleration>(
    mu: f64,
    pert: &A,
    r: &mut [f64; 3],
    v: &mut [f64; 3],
    t: f64,
    step: f64,
) -> Result<(), IntegrationError> {
    kepler_drift(mu, t, r, v, 0.5 * step)?;
    let a = pert.acceleration::<f64>(t + 0.5 * step, r, v);
    for i in 0..3 {
        v[i] += a[i] * step;
    }
    kepler_drift(mu, t + 0.5 * step, r, v, 0.5 * step)
}

/// The full-system derivative `[v, a_central + a_pert]` at `(r, v)`, for the Hermite dense output.
fn whfast_deriv<A: Acceleration>(
    mu: f64,
    pert: &A,
    t: f64,
    r: &[f64; 3],
    v: &[f64; 3],
) -> [f64; 6] {
    let inv_r3 = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).powf(-1.5);
    let ap = pert.acceleration::<f64>(t, r, v);
    [
        v[0],
        v[1],
        v[2],
        -mu * r[0] * inv_r3 + ap[0],
        -mu * r[1] * inv_r3 + ap[1],
        -mu * r[2] * inv_r3 + ap[2],
    ]
}

/// Integrate as [`integrate_whfast`], returning a dense cubic-Hermite [`Trajectory`] over the `[r, v]`
/// state (6-D) for the event detector.
///
/// # Errors
///
/// As [`integrate_whfast`].
pub fn integrate_whfast_dense<A: Acceleration>(
    mu: f64,
    pert: &A,
    r0: &[f64; 3],
    v0: &[f64; 3],
    t0: f64,
    t1: f64,
    dt: f64,
) -> Result<Trajectory<6, HermiteDense<6>>, IntegrationError> {
    if !(t1 > t0 && dt > 0.0 && mu > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    let n_steps = step_count(t0, t1, dt, 1)?;
    let step = (t1 - t0) / n_steps as f64;

    let mut r = *r0;
    let mut v = *v0;
    let mut t = t0;
    let mut y0v = [r[0], r[1], r[2], v[0], v[1], v[2]];
    let mut f0 = whfast_deriv(mu, pert, t, &r, &v);
    let mut segments = Vec::with_capacity(n_steps);
    let mut ends = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        whfast_step(mu, pert, &mut r, &mut v, t, step)?;
        t += step;
        let y1v = [r[0], r[1], r[2], v[0], v[1], v[2]];
        let f1 = whfast_deriv(mu, pert, t, &r, &v);
        segments.push(HermiteDense::new(t - step, t, y0v, f0, y1v, f1));
        ends.push(t);
        y0v = y1v;
        f0 = f1;
    }
    Ok(Trajectory::from_segments(
        t0,
        t1,
        y0v,
        segments,
        ends,
        n_steps as u32,
        0,
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use crate::forces::{J2, PointMass};

    const MU_E: f64 = 3.986_004_418e14;

    /// Newton on a strictly increasing `f`, confined to `[lo, hi]` by bisection.
    ///
    /// Both Kepler equations increase strictly in their anomaly — `1 − e cos E > 0` on an ellipse,
    /// `e cosh H − 1 > 0` on a hyperbola — so each has exactly one root, and any `[lo, hi]` whose
    /// values straddle zero contains it. Plain Newton from `M` does not always reach it: at
    /// `e = 0.99999` and `M = 6.22` it leaves a relative residual of 5e23, and the oracle would then
    /// be grading the drift against a state that is not on the orbit at that time.
    fn monotone_root(f: impl Fn(f64) -> (f64, f64), mut lo: f64, mut hi: f64) -> f64 {
        let mut x = f64::midpoint(lo, hi);
        for _ in 0..200 {
            let (value, slope) = f(x);
            // Every pass runs; there is no tolerance to stop at, because an oracle should converge
            // as far as the arithmetic allows and not one step less. Once Newton reaches its fixed
            // point it stays inside the bracket and the remaining passes are idempotent — except at
            // an exact root, where `x` would become an endpoint and the interior test would send the
            // next pass off to the midpoint.
            if value == 0.0 {
                return x;
            }
            if value > 0.0 {
                hi = x;
            } else {
                lo = x;
            }
            let next = x - value / slope;
            x = if next > lo && next < hi {
                next
            } else {
                f64::midpoint(lo, hi)
            };
        }
        x
    }

    /// Closed-form two-body state at `dt` past periapsis, from the eccentric or hyperbolic anomaly.
    ///
    /// An oracle for [`kepler_drift`] that shares none of its arithmetic: it solves Kepler's
    /// equation in the classical anomaly rather than in the universal variable, and it never touches
    /// a Stumpff function. `a < 0` selects the hyperbolic branch, where `M = e sinh H − H`.
    ///
    /// Each branch brackets its root outright. `E = M + e sin E` and `|e sin E| ≤ e`, so `E` lies in
    /// `[M − e, M + e]`. On the hyperbola `M = e sinh H − H` sits between `e sinh H − sinh H` and
    /// `e sinh H`, because `sinh H ≥ H`, so `H` lies between `asinh(M/e)` and `asinh(M/(e−1))` —
    /// both logarithmic in `M`, where the algebraically simpler `M/(e−1)` overflows `sinh` at
    /// `e = 1 + 1e-6`. Measured worst relative residual over `e = 0…8` and spans to two hundred
    /// `2π√(|a|³/μ)`: 2.1e-15 elliptic, 7.8e-15 hyperbolic.
    fn conic(a: f64, e: f64, dt: f64) -> ([f64; 3], [f64; 3]) {
        let mean_motion = (MU_E / libm::fabs(a).powi(3)).sqrt();
        let m = mean_motion * dt;
        if a > 0.0 {
            let ea = monotone_root(|x| (x - e * x.sin() - m, 1.0 - e * x.cos()), m - e, m + e);
            let b = a * (1.0 - e * e).sqrt();
            let rate = mean_motion / (1.0 - e * ea.cos());
            (
                [a * (ea.cos() - e), b * ea.sin(), 0.0],
                [-a * ea.sin() * rate, b * ea.cos() * rate, 0.0],
            )
        } else {
            let ha = monotone_root(
                |x| (e * libm::sinh(x) - x - m, e * libm::cosh(x) - 1.0),
                libm::asinh(m / e),
                libm::asinh(m / (e - 1.0)),
            );
            let b = -a * (e * e - 1.0).sqrt();
            let rate = mean_motion / (e * libm::cosh(ha) - 1.0);
            (
                [a * (libm::cosh(ha) - e), b * libm::sinh(ha), 0.0],
                [a * libm::sinh(ha) * rate, b * libm::cosh(ha) * rate, 0.0],
            )
        }
    }

    /// The drift is exact across the conic families, not only on the one orbit this crate flies.
    ///
    /// Every integration test here uses `e = 0.1` with a step of a fortieth of a period, where the
    /// universal-Kepler Newton solve converges in three passes from almost any starting point. That
    /// is why every mutation of its initial guess `X₀ = α√μΔt` and of its derivative `r(X)`
    /// survived: Newton is self-correcting, so a wrong step size and a wrong derivative both reach
    /// the same root, and all they change is how many of the 64 available passes it takes — a cost,
    /// which nothing was measuring. Ten mutants lived in that slack.
    ///
    /// The rows below spend it. High eccentricity with a step of most of a period leaves `X₀` far
    /// from its root; the hyperbolic rows drive `z = αX²` past 1 in the negative direction, which is
    /// also the only place in this crate where the Danby doubling runs inside a real propagation.
    #[test]
    fn kepler_drift_is_exact_on_every_conic() {
        // (semi-major axis, eccentricity, step in units of the orbit's own 2π√(|a|³/μ))
        let cases = [
            (8_000e3, 0.1, 0.1),
            (24_400e3, 0.73, 0.45),
            (26_554e3, 0.9, 0.49),
            (42_164e3, 0.99, 0.5),
            (-20_000e3, 1.5, 2.0),
            (-12_000e3, 3.0, 5.0),
        ];
        let mut worst: f64 = 0.0;
        for (a, e, frac) in cases {
            let scale = (libm::fabs(a).powi(3) / MU_E).sqrt();
            let dt = frac * core::f64::consts::TAU * scale;
            // Start off-apsis. At periapsis `r · v` is exactly zero, so `σ₀ = r·v/√μ` is zero, and
            // the middle term of Newton's derivative `r(X) = r₀c₀ + σ₀Xc₁ + X²c₂` contributes
            // nothing whatever the code does to it -- one mutant of that term survived a first
            // version of this test which started at periapsis for every row.
            let t_start = 0.17 * core::f64::consts::TAU * scale;
            let (r_want, v_want) = conic(a, e, t_start + dt);
            let (mut r_got, mut v_got) = conic(a, e, t_start);
            let speed2 = v_got.iter().map(|c| c * c).sum::<f64>();
            let radius = r_got.iter().map(|c| c * c).sum::<f64>().sqrt();
            // The oracle supplies the input state as well as the answer, so vis-viva is checked on
            // it first: a wrong oracle would otherwise excuse a wrong drift.
            assert!(
                libm::fabs(speed2 - MU_E * (2.0 / radius - 1.0 / a)) < 1e-9 * speed2,
                "the oracle itself violates vis-viva at a = {a:e}, e = {e}"
            );
            kepler_drift(MU_E, 0.0, &mut r_got, &mut v_got, dt).expect("drift");
            let miss = (0..3)
                .map(|i| (r_got[i] - r_want[i]).powi(2))
                .sum::<f64>()
                .sqrt()
                / radius;
            let miss_v = (0..3)
                .map(|i| (v_got[i] - v_want[i]).powi(2))
                .sum::<f64>()
                .sqrt()
                / speed2.sqrt();
            println!("conic a = {a:.3e} e = {e}: dr/r = {miss:.3e}, dv/v = {miss_v:.3e}");
            worst = worst.max(miss).max(miss_v);
        }
        println!("kepler_drift worst relative miss across the conics: {worst:.3e}");
        // 8.327e-15 measured -- every row within a few ulps of the classical solution, ellipse and
        // hyperbola alike. Starting off-apsis rather than at periapsis cut this by four orders,
        // which locates the earlier 2.246e-11 in the *oracle*: at `e = 0.99` its Newton solve has
        // `1 − e cos E = 0.01` at perigee and loses two digits there, while `kepler_drift`, working
        // in the universal variable, has no such point.
        assert!(
            worst < 3e-14,
            "the drift is not exact: {worst:.3e} relative"
        );
    }

    /// A hyperbolic pass is a *long* arc. Twenty of the orbit's own `2π√(|a|³/μ)` is under three
    /// days at `a = −12 000 km`, and an interplanetary departure is longer still, so a drift that
    /// only works for a couple of those does not work.
    #[test]
    fn kepler_drift_carries_a_long_hyperbolic_arc() {
        let mut worst: f64 = 0.0;
        for (a, e) in [(-20_000e3, 1.5), (-12_000e3, 3.0), (-5_000e3, 8.0)] {
            let scale = (libm::fabs(a).powi(3) / MU_E).sqrt();
            for k in [20.0, 60.0, 200.0] {
                let dt = k * core::f64::consts::TAU * scale;
                let t_start = 0.17 * core::f64::consts::TAU * scale;
                let (r_want, _) = conic(a, e, t_start + dt);
                let (mut r_got, mut v_got) = conic(a, e, t_start);
                let radius = r_got.iter().map(|c| c * c).sum::<f64>().sqrt();
                kepler_drift(MU_E, 0.0, &mut r_got, &mut v_got, dt).expect("drift");
                let miss = (0..3)
                    .map(|i| (r_got[i] - r_want[i]).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    / radius;
                println!("hyperbola a = {a:.3e} e = {e} over {k} scales: dr/r = {miss:.3e}");
                worst = worst.max(miss);
            }
        }
        assert!(
            worst < 1e-12,
            "the long hyperbolic drift is off: {worst:.3e} relative"
        );
    }

    /// The near-parabolic band — the one regime where no closed-form guess is any good.
    ///
    /// The circular form `α√μΔt` points the wrong way on a hyperbola; Vallado's hyperbolic form has
    /// `√(−a) → ∞` against `ln(·) → −∞` at the parabola and returns the wrong *sign* outright when
    /// the state is far from periapsis; and Barker's parabolic form, measured as a guess here, is
    /// worse than either on everything that is not a parabola. These are the rows that reach
    /// [`solve_universal_kepler`]'s bracket — eight of the 532 in the matrix behind [`FAST_PASSES`],
    /// every one of them in this band — so they are what proves the bracket does its job.
    ///
    /// The oracle had to be repaired before it could grade this. Started at `M`, `conic`'s elliptic
    /// Newton leaves a relative residual of 5e23 at `e = 0.99999, M = 6.22`, and started at
    /// `asinh(M/e)` its hyperbolic Newton leaves 2e7 at `e = 1.00001` and 1e47 at `e = 1.000001` —
    /// the classical anomaly degenerates near the parabola for the same reason the universal
    /// variable does. Both branches are bracketed now; every row here still re-derives the anomaly
    /// from the state the oracle returned and holds it against `M = nΔt`, so a regression in the
    /// oracle fails this test rather than quietly grading the drift against a wrong answer.
    #[test]
    fn kepler_drift_solves_the_near_parabolic_band() {
        const PERIAPSIS: f64 = 7_000e3;
        // Recover the anomaly from the state and check it satisfies the equation that defines it.
        let oracle_residual = |a: f64, e: f64, dt: f64, r: &[f64; 3]| {
            let mean = (MU_E / libm::fabs(a).powi(3)).sqrt() * dt;
            if a > 0.0 {
                let b = a * (1.0 - e * e).sqrt();
                let ea = libm::atan2(r[1] / b, r[0] / a + e);
                // `E` comes back in (−π, π]; `M` is unbounded, and `E − e sin E` is odd and 2π-
                // periodic in that shift, so reduce `M` to the same branch before comparing.
                let mut m = mean.rem_euclid(core::f64::consts::TAU);
                if m > core::f64::consts::PI {
                    m -= core::f64::consts::TAU;
                }
                libm::fabs(ea - e * ea.sin() - m) / libm::fabs(m).max(1.0)
            } else {
                let b = -a * (e * e - 1.0).sqrt();
                let ha = libm::asinh(r[1] / b);
                libm::fabs(e * libm::sinh(ha) - ha - mean) / libm::fabs(mean).max(1.0)
            }
        };

        let mut worst: f64 = 0.0;
        let mut worst_oracle: f64 = 0.0;
        for e in [
            0.99,
            0.999,
            0.9999,
            0.99999,
            1.0 - 1e-6,
            1.0 + 1e-6,
            1.00001,
            1.0001,
            1.001,
            1.01,
        ] {
            let a = PERIAPSIS / (1.0 - e);
            let scale = (libm::fabs(a).powi(3) / MU_E).sqrt();
            let mut worst_here: f64 = 0.0;
            for frac in [0.001, 0.35, 0.49] {
                for span in [0.1, 0.5] {
                    let t_start = frac * core::f64::consts::TAU * scale;
                    let dt = span * core::f64::consts::TAU * scale;
                    let (r_want, _) = conic(a, e, t_start + dt);
                    let (mut r_got, mut v_got) = conic(a, e, t_start);
                    worst_oracle = worst_oracle
                        .max(oracle_residual(a, e, t_start, &r_got))
                        .max(oracle_residual(a, e, t_start + dt, &r_want));
                    kepler_drift(MU_E, 0.0, &mut r_got, &mut v_got, dt).expect("drift");
                    // Relative to where the drift lands, not to where it started: across half of
                    // `2π√(|a|³/μ)` a near-parabolic radius moves by orders of magnitude.
                    let target = r_want.iter().map(|c| c * c).sum::<f64>().sqrt();
                    let miss = (0..3)
                        .map(|i| (r_got[i] - r_want[i]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / target;
                    worst_here = worst_here.max(miss);
                }
            }
            println!("e = {e:<12} a = {a:+.4e}: worst dr/r = {worst_here:.3e}");
            worst = worst.max(worst_here);
        }
        assert!(
            worst_oracle < 1e-12,
            "the oracle does not satisfy Kepler's equation on these rows: {worst_oracle:.3e}"
        );
        // 6.2e-14 measured, at `e = 0.9999`, against the 8.3e-15 of
        // `kepler_drift_is_exact_on_every_conic`: the band costs a factor of seven, not a decade,
        // and the arithmetic says that is the oracle's cancellation rather than the drift's — it
        // forms `a(cos E − e)` with `a` up to 7e12 m and `cos E − e` down to 1e-6, spending six
        // decades of the f64 mantissa before the drift is asked anything. The bound stays at the
        // 1e-12 the sibling tests use: these rows are the ones most exposed to a different libm.
        assert!(
            worst < 1e-12,
            "the near-parabolic drift is off: {worst:.3e} relative"
        );
    }

    /// A drift is a group action: undoing it has to return the state it started from.
    ///
    /// Nothing in this crate runs the drift backwards — WHFast splits a positive step, SABA's
    /// coefficients are all positive, MERCURIUS halves a positive one — so `Δt < 0` reaches
    /// [`solve_universal_kepler`] only from here, and every orientation the bracket makes on the sign
    /// of `Δt` would otherwise go unmeasured. The round trip needs no oracle: it is exact for the
    /// exact solution and for nothing else.
    #[test]
    fn the_drift_runs_backwards_to_where_it_came_from() {
        let mut worst: f64 = 0.0;
        for (a, e) in [
            (8_000e3, 0.1),
            (42_164e3, 0.99),
            (7_000e3 / (1.0 - 0.999_99), 0.999_99),
            (-20_000e3, 1.5),
            (-5_000e3, 8.0),
        ] {
            let scale = (libm::fabs(a).powi(3) / MU_E).sqrt();
            for span in [0.1, 0.5, 20.0] {
                let dt = span * core::f64::consts::TAU * scale;
                let (r0, v0) = conic(a, e, 0.17 * core::f64::consts::TAU * scale);
                let (mut r, mut v) = (r0, v0);
                let radius = r0.iter().map(|c| c * c).sum::<f64>().sqrt();
                kepler_drift(MU_E, 0.0, &mut r, &mut v, dt).expect("forward");
                kepler_drift(MU_E, dt, &mut r, &mut v, -dt).expect("backward");
                let miss = (0..3).map(|i| (r[i] - r0[i]).powi(2)).sum::<f64>().sqrt() / radius;
                println!("a = {a:+.3e} e = {e} over {span} scales: round trip dr/r = {miss:.3e}");
                worst = worst.max(miss);
            }
        }
        assert!(
            worst < 1e-12,
            "the drift does not undo itself: {worst:.3e} relative"
        );
    }

    /// What the solve costs — the only thing the safeguarded solver's own code can be held to.
    ///
    /// Bracketing a monotone function pins the root: the seed, the geometric walk and the choice
    /// between a Newton step and a bisection all land on the same answer whatever is done to them.
    /// Seven representative mutations of that code were injected and every accuracy assertion in
    /// this module still passed. So the accuracy tests, however tight, cannot see any of it, and
    /// what does see it is the work — a broken seed costs bisections, a broken Newton step costs all
    /// of them. This holds the profile [`FAST_PASSES`] and [`SAFE_PASSES`] were chosen from.
    ///
    /// Measured: 9 evaluations at worst over the 84 ordinary rows, and over the 16-row
    /// near-parabolic band a median of 24 and a worst of 51. The bounds around those are one or two
    /// counts wide, and both band bounds are two-sided on purpose — a mutation that made the solve
    /// *cheaper* would otherwise pass, and three of them do exactly that, one by dropping the worst
    /// case from 51 to 26. The width is not slack for a different `libm` to move a row through: if
    /// a platform moves one, the answer is to measure it there and record both figures, not to
    /// widen the range until nothing can fail it.
    #[test]
    fn the_solve_costs_what_the_budgets_were_set_from() {
        let solve = |a: f64, e: f64, frac: f64, span: f64| {
            let scale = (libm::fabs(a).powi(3) / MU_E).sqrt();
            let (r, v) = conic(a, e, frac * core::f64::consts::TAU * scale);
            let r0 = r.iter().map(|c| c * c).sum::<f64>().sqrt();
            let v2 = v.iter().map(|c| c * c).sum::<f64>();
            let rv = (0..3).map(|i| r[i] * v[i]).sum::<f64>();
            let sqrt_mu = MU_E.sqrt();
            solve_universal_kepler(
                r0,
                rv / sqrt_mu,
                2.0 / r0 - v2 / MU_E,
                sqrt_mu,
                span * core::f64::consts::TAU * scale,
            )
            .expect("solve")
            .evaluations
        };

        // An ordinary orbit must not pay for the safeguard: it never reaches it. The long
        // hyperbolic spans are here because they are the only rows where Vallado's closed form is
        // both used and load-bearing — an ellipse takes the circular guess, and near the parabola
        // the sign test throws the closed form away. The negative spans are load-bearing too, and
        // for a sharper reason: with only forward rows every mutation of the closed form cost
        // exactly what the original did, because `sign` is `+1` on all of them and a wrong `sign`
        // is what the formula is most easily broken in. Reversing the drift moves seven of them
        // from 9 evaluations to between 40 and 56.
        let mut worst_ordinary = 0;
        for (a, e) in [
            (8_000e3, 0.0),
            (24_400e3, 0.73),
            (-20_000e3, 1.5),
            (-12_000e3, 3.0),
        ] {
            for frac in [0.001, 0.17, 0.35] {
                for span in [0.01, 0.1, 0.5, 20.0, 200.0, -0.1, -20.0] {
                    worst_ordinary = worst_ordinary.max(solve(a, e, frac, span));
                }
            }
        }
        println!("ordinary orbits: worst {worst_ordinary} evaluations of F");
        assert!(
            worst_ordinary <= 12,
            "the fast path costs {worst_ordinary} evaluations, not the 9 measured"
        );

        // The near-parabolic band, where the safeguard is what answers. Rows chosen from a scan of
        // 84 000 states for the ones that reach it; a solve that stops reaching it is as much a
        // regression here as one that costs too much, so the median is held from below as well.
        let band = [
            (0.99, 0.49, 0.5),
            (0.999, 0.6, 0.5),
            (0.9999, 0.49, 0.5),
            (0.9999, 0.6, 0.5),
            (0.95, 0.35, 0.5),
            (1.0001, 0.001, 0.1),
            (1.001, 0.001, 0.1),
            (1.01, 0.001, 0.1),
        ];
        let mut costs: Vec<u32> = band
            .iter()
            .flat_map(|&(e, frac, span)| {
                [span, -span].map(move |s| solve(7_000e3 / (1.0 - e), e, frac, s))
            })
            .collect();
        let worst_band = *costs.iter().max().expect("rows");
        costs.sort_unstable();
        let median = costs[costs.len() / 2];
        println!("near-parabolic band: median {median}, worst {worst_band} evaluations of F");
        assert!(
            (45..=60).contains(&worst_band),
            "the safeguard costs {worst_band} evaluations, not the 51 measured"
        );
        assert!(
            (24..=26).contains(&median),
            "the band's median is {median} evaluations, not the 24 measured"
        );
    }

    /// The closed-form guess can point the *wrong way*, and the answer must not depend on it.
    ///
    /// Vallado's hyperbolic form is derived about periapsis. Start well past it and its logarithm
    /// takes an argument below 1, so it returns a negative `X` while `Δt > 0` fixes the root to be
    /// positive — the guess is not merely poor, it is on the other side of the origin. Newton from
    /// there climbs the exponential branch away from the root, which is what
    /// [`solve_universal_kepler`]'s sign test and bracket exist to make irrelevant.
    ///
    /// Each row re-evaluates the guess and asserts it really is wrong-signed, so the test cannot
    /// silently stop exercising the path if the seed is ever changed.
    #[test]
    fn the_answer_does_not_depend_on_a_guess_that_points_the_wrong_way() {
        let mut worst: f64 = 0.0;
        let mut rows = 0;
        for (a, e) in [(-14_000e3, 1.5), (-7_000e3, 2.0), (-3_500e3, 3.0)] {
            let scale = (libm::fabs(a).powi(3) / MU_E).sqrt();
            for frac in [0.35, 0.49, 0.6] {
                for span in [0.02, 0.1] {
                    let t_start = frac * core::f64::consts::TAU * scale;
                    let dt = span * core::f64::consts::TAU * scale;
                    let (r_want, _) = conic(a, e, t_start + dt);
                    let (mut r, mut v) = conic(a, e, t_start);

                    // Vallado's guess, re-formed here from the state alone.
                    let r0 = r.iter().map(|c| c * c).sum::<f64>().sqrt();
                    let v2 = v.iter().map(|c| c * c).sum::<f64>();
                    let rv = (0..3).map(|i| r[i] * v[i]).sum::<f64>();
                    let alpha = 2.0 / r0 - v2 / MU_E;
                    let scale_x = (-1.0 / alpha).sqrt();
                    let sigma0 = rv / MU_E.sqrt();
                    let denominator = sigma0 + scale_x * (1.0 - r0 * alpha);
                    let guess = scale_x * libm::log(-2.0 * MU_E.sqrt() * alpha * dt / denominator);
                    assert!(
                        guess < 0.0 && dt > 0.0,
                        "row (a = {a:e}, e = {e}, frac = {frac}, span = {span}) no longer has a \
                         wrong-signed guess: X0 = {guess:e} for dt = {dt:e}"
                    );

                    kepler_drift(MU_E, 0.0, &mut r, &mut v, dt).expect("drift");
                    let target = r_want.iter().map(|c| c * c).sum::<f64>().sqrt();
                    let miss = (0..3)
                        .map(|i| (r[i] - r_want[i]).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        / target;
                    worst = worst.max(miss);
                    rows += 1;
                }
            }
        }
        println!("{rows} wrong-signed-guess rows: worst dr/r = {worst:.3e}");
        assert!(
            worst < 1e-12,
            "the drift followed its guess off the orbit: {worst:.3e} relative"
        );
    }

    /// The drift's own boundary guard, which nothing reaches through the integrators.
    ///
    /// `integrate_whfast` refuses `mu <= 0` itself, so its tests exercise *that* guard and never
    /// this one. But [`kepler_drift`] is `pub(crate)` and [`crate::saba`] and [`crate::mercurius`]
    /// call it directly, so its own refusal of a state with no osculating orbit is a contract in its
    /// own right — and it has to hold before anything is written, or a caller that ignores the error
    /// is left with a half-advanced state.
    #[test]
    fn the_drift_refuses_a_state_that_has_no_orbit() {
        let good_r = [7_000e3, 0.0, 0.0];
        let good_v = [0.0, 7_546.0, 0.0];
        let cases: [(&str, f64, [f64; 3], [f64; 3]); 6] = [
            ("at the centre of attraction", MU_E, [0.0; 3], good_v),
            ("mu = 0", 0.0, good_r, good_v),
            ("mu < 0", -MU_E, good_r, good_v),
            ("non-finite position", MU_E, [f64::NAN, 0.0, 0.0], good_v),
            ("infinite position", MU_E, [f64::INFINITY, 0.0, 0.0], good_v),
            ("non-finite velocity", MU_E, good_r, [0.0, f64::NAN, 0.0]),
        ];
        for (name, mu, r0, v0) in cases {
            let (mut r, mut v) = (r0, v0);
            assert_eq!(
                kepler_drift(mu, 0.0, &mut r, &mut v, 60.0),
                Err(IntegrationError::BadArguments),
                "{name}"
            );
            // Bit-compared, because NaN is one of the values under test and `NaN != NaN`.
            for i in 0..3 {
                assert_eq!(
                    r[i].to_bits(),
                    r0[i].to_bits(),
                    "{name}: r[{i}] was written"
                );
                assert_eq!(
                    v[i].to_bits(),
                    v0[i].to_bits(),
                    "{name}: v[{i}] was written"
                );
            }
        }
    }

    /// The kick samples the perturbation at the *midpoint* of the step, and only a time-dependent
    /// force can say so.
    ///
    /// Every other test in this module perturbs with `J2`, which does not read `t` — so `t + h/2`
    /// could be `t`, `t − h/2` or `t · h/2` and no trajectory would move. `tests/whfast.rs` does run
    /// a time-modulated `J2` against DOP853, and the mid-step mutants survived it anyway: a
    /// trajectory comparison has to allow for the splitting error it is measuring *through*, so its
    /// bound sits above the defect. The order does not — it is a ratio, and the reference cancels
    /// out of it. That is the second-order accuracy of the whole leapfrog: sampling the force
    /// off-centre by `δ` costs `δ·ȧ` per step, which is
    /// exactly what makes drift–kick–drift second order rather than first. The check is therefore
    /// the *order*, measured by halving the step: right midpoint, the error falls by four; wrong
    /// one, by two.
    #[test]
    fn the_kick_samples_the_perturbation_at_the_midpoint_of_the_step() {
        /// A perturbation that is nothing but a clock: `a = (κt, 0, 0)`.
        struct Ramp {
            rate: f64,
        }
        impl Acceleration for Ramp {
            fn acceleration<S: spacelab_autodiff::Scalar>(
                &self,
                t: S,
                _r: &[S; 3],
                _v: &[S; 3],
            ) -> [S; 3] {
                [
                    S::from_f64(self.rate) * t,
                    S::from_f64(0.0),
                    S::from_f64(0.0),
                ]
            }
        }

        let pert = Ramp { rate: 1e-6 };
        let r0 = [7_000e3, 0.0, 0.0];
        let v0 = [0.0, 7_546.0, 0.0];
        let (t0, t1) = (0.0, 600.0);
        // Reference at a step 32x finer than the coarsest compared below.
        let fine = integrate_whfast(MU_E, &pert, &r0, &v0, t0, t1, 0.312_5).expect("reference");
        let miss = |step: f64| {
            let got = integrate_whfast(MU_E, &pert, &r0, &v0, t0, t1, step).expect("coarse");
            (0..3)
                .map(|i| (got.0[i] - fine.0[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let (coarse, half) = (miss(10.0), miss(5.0));
        let order = libm::log2(coarse / half);
        println!("whfast on a ramp: |dr| {coarse:.3e} -> {half:.3e}, observed order {order:.3}");
        assert!(
            order > 1.8,
            "halving the step gained only 2^{order:.3}: the kick is not sampling the midpoint"
        );
    }

    /// The dense derivative is the one place WHFast writes the central acceleration out longhand —
    /// the step never forms it, the Kepler drift being analytic — so nothing else in the module
    /// cross-checks it, and it is what every Hermite slope is built from. The force model computes
    /// the same field by a wholly separate path, so it is the check.
    ///
    /// An end-to-end comparison cannot do this job. A sign error on `ap` flips it at *both* ends of
    /// a segment, and the mid-segment value of a cubic Hermite carries the slopes as `(h/8)(f₀−f₁)`,
    /// so the visible defect is `(h/4)·|ȧ_pert|·h` — second order in the step, and buried under the
    /// splitting error that any reference trajectory is being compared through.
    #[test]
    fn the_dense_derivative_is_the_field_the_forces_module_computes() {
        let j2 = J2 {
            mu: MU_E,
            re: 6_378_136.3,
            j2: 1.082_635_9e-3,
        };
        let field = (PointMass { mu: MU_E }, j2);
        // Off-axis, off-plane and off-diagonal: a sample on an axis or in a plane leaves whole rows
        // identically zero, and a symmetric one lets two rows stand in for each other.
        let r = [4_231_000.0, -5_114_500.0, 3_002_900.0];
        let v = [3_812.4, 4_907.1, -2_366.8];
        let t = 12_345.678;
        let got = whfast_deriv(MU_E, &j2, t, &r, &v);
        let want = field.acceleration::<f64>(t, &r, &v);
        let scale = (want[0] * want[0] + want[1] * want[1] + want[2] * want[2]).sqrt();
        let mut worst: f64 = 0.0;
        for i in 0..3 {
            assert_eq!(
                got[i].to_bits(),
                v[i].to_bits(),
                "row {i} of the derivative must be the velocity verbatim"
            );
            worst = worst.max(libm::fabs(got[i + 3] - want[i]) / scale);
        }
        println!("whfast_deriv vs the force model: {worst:.3e} relative");
        // 0.0 measured -- bit-identical at this sample, but the bound is a few ulps rather than
        // equality because the two do not evaluate the same expression: `whfast_deriv` forms `1/r³`
        // as `(r·r).powf(-1.5)`, `PointMass` as `(r²√r²).recip()`, which its own comment prefers for
        // ~1 ulp against powf's ~2-3. Whether they agree to the last bit is a property of this
        // libm and this sample; that they agree to a few ulps is a property of the mathematics.
        assert!(
            worst < 5e-15,
            "the dense derivative is not the field: {worst:.3e} relative"
        );
    }

    /// Only a finite Stumpff argument is constructible, and every finite one leaves the reduction
    /// loop in a bounded number of passes.
    ///
    /// The bound is measured, not assumed: the widest argument the type admits is `f64::MAX`, and
    /// quartering it down to `0.1` takes `⌈log₄(f64::MAX/0.1)⌉` passes. Counting them here is what
    /// makes "terminates" a fact about this code rather than about the arithmetic in the abstract —
    /// and it is the exact quantity that goes to infinity for the values `new` refuses.
    #[test]
    fn only_a_finite_stumpff_argument_is_constructible_and_all_of_them_terminate() {
        for z in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(StumpffArg::new(z).is_none(), "{z}");
        }
        let mut worst = 0u32;
        for z in [
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            1e-30,
            0.1,
            -0.1,
            1.0,
            -1.0,
            1e30,
            f64::MAX,
            f64::MIN,
        ] {
            assert!(StumpffArg::new(z).is_some(), "{z}");
            let (mut zr, mut n) = (z, 0u32);
            while zr.abs() > 0.1 {
                zr *= 0.25;
                n += 1;
            }
            worst = worst.max(n);
        }
        assert_eq!(worst, 514);
    }

    /// `stumpff` against the closed forms it is an implementation of.
    ///
    /// The series-plus-Danby-doubling in `stumpff` exists because the closed forms cancel badly as
    /// `z → 0`: `c₂ = (1 − cos √z)/z` loses every significant digit there. Away from zero the
    /// closed forms are the accurate ones, which makes them an oracle that shares no arithmetic
    /// with the code under test — and `z` is exactly where the integration tests have no reach,
    /// because they take steps of a fortieth to a five-hundredth of an orbit and leave `z = αX²`
    /// so small that `n` is 0 and the doubling loop never runs at all. The `|z| ≥ 1` rows below are
    /// the first thing in this crate to execute it.
    ///
    /// Residuals are relative and the tolerance is measured, not assumed: the worst row is
    /// `z = -5000`, where `c₀ = cosh √5000 ≈ 1.1e30` and 6 doublings have squared its way up from
    /// the series.
    #[test]
    fn stumpff_matches_its_closed_form_away_from_the_series_regime() {
        // c₀ = cos √z, c₁ = sin √z/√z, c₂ = (1 − c₀)/z, c₃ = (1 − c₁)/z, and the hyperbolic
        // counterparts for z < 0 -- the same four identities in both signs once √z is complex.
        let closed = |z: f64| -> (f64, f64, f64, f64) {
            let (c0, c1) = if z > 0.0 {
                let s = z.sqrt();
                (s.cos(), s.sin() / s)
            } else {
                let s = (-z).sqrt();
                (s.cosh(), s.sinh() / s)
            };
            (c0, c1, (1.0 - c0) / z, (1.0 - c1) / z)
        };

        // Down to |z| = 1e-3 and no further: the closed form's own cancellation error is
        // `eps/(1 − cos √z)`, which is 4.4e-15 at |z| = 0.1 and 4.4e-13 at 1e-3 -- still two orders
        // under the bound -- but 4.4e-10 by 1e-6, where the oracle would be the less accurate of
        // the two. Below 0.1 the reduction loop does not run, so those rows exercise the bare
        // eight-term series; from 0.5 up, `n` climbs and the doubling recurrence carries the value.
        let mut worst: f64 = 0.0;
        for &z in &[
            1e-3, -1e-3, 0.01, -0.01, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.5, -2.5, 9.0, -9.0, 40.0,
            -40.0, 300.0, -300.0, 5000.0, -5000.0,
        ] {
            let got = stumpff(StumpffArg::new(z).expect("finite"));
            let want = closed(z);
            for (g, w) in [
                (got.0, want.0),
                (got.1, want.1),
                (got.2, want.2),
                (got.3, want.3),
            ] {
                worst = worst.max((g - w).abs() / w.abs().max(1e-300));
            }
        }
        println!("stumpff worst relative residual vs closed form: {worst:.3e}");
        assert!(
            worst < 1e-11,
            "stumpff disagrees with cos/cosh by {worst:.3e} relative"
        );
    }

    /// `z = 0` exactly, where every closed form is 0/0 and only the series has an answer.
    #[test]
    fn stumpff_at_zero_is_the_series_limit() {
        for z in [0.0, -0.0] {
            let (c0, c1, c2, c3) = stumpff(StumpffArg::new(z).expect("finite"));
            assert_eq!((c0, c1), (1.0, 1.0), "c0, c1 at z = {z}");
            assert!((c2 - 0.5).abs() < 1e-16, "c2 at z = {z}: {c2}");
            assert!((c3 - 1.0 / 6.0).abs() < 1e-16, "c3 at z = {z}: {c3}");
        }
    }
}
