//! IAS15 — a 15th-order, adaptive Gauss–Radau integrator (Everhart 1985; Rein & Spiegel 2015).
//!
//! IAS15 is the gold standard for high-precision gravitational dynamics: an implicit Runge–Kutta on
//! eight Gauss–Radau substep nodes, order 15, with a predictor–corrector stage solve and a step
//! controller that holds the *fractional* error at a set tolerance. Over a billion orbits its energy
//! error grows only as the square root of time (Brouwer's law) — it does not drift — because every
//! accumulation is compensated and the method carries no systematic bias.
//!
//! # How it works, per step
//!
//! The acceleration across a step is fitted by a degree-7 polynomial in the substep parameter,
//! `a(h) ≈ a₀ + b₀h + … + b₆h⁷`. The seven `b` coefficients are found by iteration: predict the
//! position at each Gauss–Radau node from the current `b`s, evaluate the force there, update the `b`s
//! from the new accelerations through Newton divided differences (`g`) and the constant conversion
//! matrix `c`. Two or three sweeps converge to machine precision. The step is then advanced by the
//! Gauss–Radau quadrature — exact to order 15 — and the next step size is chosen from the last
//! coefficient `b₆`, which measures the unmodelled part of the acceleration.
//!
//! # Coefficients
//!
//! Everything is derived from the eight nodes in `derivations/ias15_nodes.mac`: the nodes are the
//! roots of `d⁷/dx⁷[x⁸(x−1)⁷]` (the Gauss–Radau-left polynomial), and the `c`/`d` conversion matrices
//! are the coefficients of the Newton-basis products expanded into the power basis (and its inverse).
//! The literals below are transcribed from that CAS output; they reproduce REBOUND's hardcoded arrays
//! to \~1e-33.
//!
//! # Constraint
//!
//! This implementation takes a **position-only** acceleration `a(r)` (gravity), like the
//! [symplectic](crate::symplectic) and [Gauss–Jackson](crate::gj) methods; a velocity-dependent
//! force would additionally need the velocity predicted at each substep (REBOUND does this; it is
//! omitted here for clarity). The predictor/corrector and step control otherwise follow REBOUND.

// The node/conversion literals are full precision; the step arithmetic casts small counts.
#![allow(
    clippy::needless_range_loop,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]

use std::vec::Vec;

use crate::DenseOutput;
use crate::driver::{IntegrationError, Trajectory};

/// Number of substep nodes (`h₀ = 0 … h₇`).
const STAGES: usize = 8;
/// Number of `b`/`g` coefficients (the degree-7 fit).
const NB: usize = 7;

/// Gauss–Radau-left nodes, roots of `d⁷/dx⁷[x⁸(x−1)⁷]` (see the module derivation).
const H: [f64; STAGES] = [
    0.0,
    0.05626256053692214646565219103231117577976,
    0.1802406917368923649875799428091817845381,
    0.3526247171131696373739077701712412028272,
    0.5471536263305553830014485576523488545953,
    0.7342101772154105315232106083066100026233,
    0.8853209468390957680903597629324853728798,
    0.9775206135612875018911745004291549400898,
];

/// `c` conversion matrix (Newton `g` → power `b`), REBOUND flat packing `c[k(k-1)/2 + j]`.
const CC: [f64; 21] = [
    -0.05626256053692214646565219103231117,
    0.01014080283006362998648180478770816,
    -0.2365032522738145114532321338414930,
    -0.003575897729251617594934458905322593,
    0.09353769525946206589574846128131372,
    -0.5891279693869841488271399040127335,
    0.001956565409947221076900567073878080,
    -0.05475538688906868644080842978970579,
    0.4158812000823068616886219133299269,
    -1.136281595717539531828588461665085,
    -0.001436530236370891542445955315192092,
    0.04215852772126870770729734745685035,
    -0.3600995965020568122897664641213853,
    1.250150711840691025850544082896871,
    -1.870491772932950063351799069971692,
    0.001271790309026867749294311616776803,
    -0.03876035791590677036990462486748009,
    0.3609622434528459832253398082517925,
    -1.466884208400426964370155258144916,
    2.906136259308429301423791305409338,
    -2.755812719772045831442158832904176,
];

/// `d` conversion matrix (power `b` → Newton `g`, for warm starting), flat packing `d[j(j-1)/2 + i]`.
const DD: [f64; 21] = [
    0.05626256053692214646565219103231117,
    0.003165475718170829249990480039457002,
    0.2365032522738145114532321338414930,
    0.0001780977692217433881125279219790365,
    0.04579298550602791889545387302348872,
    0.5891279693869841488271399040127335,
    0.00001002023652232912720956721522478201,
    0.008431857153525701544499974166334916,
    0.2535340690545692665214615976219055,
    1.136281595717539531828588461665085,
    0.0000005637641639318207610383850115697061,
    0.001529784002500465818949007959786274,
    0.09783423653244400536536483996035257,
    0.8752546646840910912297245940456613,
    1.870491772932950063351799069971692,
    0.00000003171881540176136647585481788136691,
    0.0002762930909826476593130226395742977,
    0.03602855398373645960038707429083216,
    0.5767330002770787313544596120844282,
    2.248588760769159793392689541369861,
    2.755812719772045831442158832904176,
];

/// Fractional-error tolerance controlling the step size (REBOUND default).
const EPSILON: f64 = 1e-9;
/// Step-shrink safety factor: a step that would shrink by more than this is rejected and retried.
const SAFETY: f64 = 0.25;
/// Predictor–corrector convergence threshold on the relative change of `b₆`.
const CONVERGENCE: f64 = 1e-16;
/// Cap on predictor–corrector sweeps before the step is accepted anyway.
const MAX_SWEEPS: usize = 12;

/// Divided-difference denominators `rr[l] = h_j − h_k`, computed from the nodes.
fn rr_table() -> [f64; 28] {
    let mut rr = [0.0; 28];
    let mut l = 0;
    for j in 1..STAGES {
        for k in 0..j {
            rr[l] = H[j] - H[k];
            l += 1;
        }
    }
    rr
}

/// Compensated (Kahan–Neumaier) addition `*y += inc`, carrying the low bits in `*c`.
#[inline]
fn add_cs(y: &mut f64, c: &mut f64, inc: f64) {
    let t = *y + inc;
    if y.abs() >= inc.abs() {
        *c += (*y - t) + inc;
    } else {
        *c += (inc - t) + *y;
    }
    *y = t;
}

/// Whether accepting `dt` at `t` would move the clock.
///
/// This stands where the sibling integrators put a step floor, because a floor cannot be expressed
/// here: the driver clamps the last step to exactly `t1 - t`, a remainder that can be arbitrarily
/// small, so any span-relative floor large enough to be a floor would reject the legitimate last
/// step. The exact condition is the one that matters anyway — a step that does not advance `t`
/// cannot terminate the loop, whatever its size.
///
/// Written as `t + dt > t` rather than `dt > 0.0` on purpose. It is the whole condition: it rejects
/// zero and negative steps, and equally a step too small to survive the addition, which is how an
/// approach to a singularity ends. And being asked in the affirmative it also rejects NaN, where the
/// `dt < min_step` form the siblings use would wave a NaN step through.
#[inline]
fn step_advances(t: f64, dt: f64) -> bool {
    t + dt > t
}

/// One IAS15 predictor–corrector solve at `(r, v, a0)` with step `dt`: iterate the b-coefficients to
/// convergence (they are updated in place through the c-matrix), and return the acceleration at the
/// last node and the step size the controller suggests next. Shared by the plain and dense drivers.
fn ias15_pc<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    r: &[f64; D],
    v: &[f64; D],
    a0: &[f64; D],
    dt: f64,
    b: &mut [[f64; D]; NB],
    rr: &[f64; 28],
) -> ([f64; D], f64) {
    // Initialise g from the warm-started b via the d-matrix (b→g); zero on the first step.
    let mut g = [[0.0; D]; NB];
    for i in 0..D {
        for m in 0..NB {
            let mut acc = b[m][i];
            for j in (m + 1)..NB {
                acc += DD[j * (j - 1) / 2 + m] * b[j][i];
            }
            g[m][i] = acc;
        }
    }

    let mut a_last = *a0; // acceleration at the final node h₇ (for the error estimate)
    let mut predictor_error = f64::MAX;
    let mut sweeps = 0;
    loop {
        let mut db6_max = 0.0_f64;
        let mut a7_max = 0.0_f64;
        for n in 1..STAGES {
            let s = H[n];
            // Predict position at substep n from the current b's (Horner, matches REBOUND).
            let mut rs = *r;
            for i in 0..D {
                let dx = ((((((((b[6][i] * 7.0 * s / 9.0 + b[5][i]) * 3.0 * s / 4.0
                    + b[4][i])
                    * 5.0
                    * s
                    / 7.0
                    + b[3][i])
                    * 2.0
                    * s
                    / 3.0
                    + b[2][i])
                    * 3.0
                    * s
                    / 5.0
                    + b[1][i])
                    * s
                    / 2.0
                    + b[0][i])
                    * s
                    / 3.0
                    + a0[i])
                    * dt
                    * s
                    / 2.0
                    + v[i])
                    * dt
                    * s;
                rs[i] += dx;
            }
            let at = accel(&rs);
            if n == STAGES - 1 {
                a_last = at;
            }

            // Newton divided difference for g[m], m = n-1, then propagate to the b's via c.
            let m = n - 1;
            let base = m * (m + 1) / 2;
            for i in 0..D {
                let mut gk = (at[i] - a0[i]) / rr[base];
                for p in 1..=m {
                    gk = (gk - g[p - 1][i]) / rr[base + p];
                }
                let delta = gk - g[m][i];
                g[m][i] = gk;
                b[m][i] += delta;
                // `m = 0` on the first stage: the propagation loop is empty and `m*(m-1)/2` would
                // underflow `usize` in debug (release wraps to 0). Guard rather than lean on wrap.
                if m > 0 {
                    let cbase = m * (m - 1) / 2;
                    for j in 0..m {
                        b[j][i] += delta * CC[cbase + j];
                    }
                }
                if n == STAGES - 1 {
                    db6_max = db6_max.max(delta.abs());
                    a7_max = a7_max.max(at[i].abs());
                }
            }
        }

        sweeps += 1;
        let err = if a7_max > 0.0 { db6_max / a7_max } else { 0.0 };
        if err < CONVERGENCE {
            break;
        }
        if sweeps > 2 && err >= predictor_error {
            break; // oscillating — no further gain
        }
        predictor_error = err;
        if sweeps >= MAX_SWEEPS {
            break;
        }
    }

    // Step size from the last coefficient b₆ vs the acceleration (fractional error, exponent 1/7).
    //
    // Written as `if m >= x { m } else { x }` rather than `m.max(x)` because `f64::max` returns the
    // *other* operand when one is NaN: an acceleration that has gone non-finite disappears from the
    // maximum, `a0_max` stays 0, and a diverged step reads as the force-free one below.
    let mut b6_max = 0.0_f64;
    let mut a0_max = 0.0_f64;
    for i in 0..D {
        let (bi, ai) = (b[6][i].abs(), a_last[i].abs());
        b6_max = if b6_max >= bi { b6_max } else { bi };
        a0_max = if a0_max >= ai { a0_max } else { ai };
    }
    // `== 0.0` and not `> 0.0`: `a0_max` is a maximum of absolute values, so the two partition the
    // finite range identically, and only this form lets a NaN through to the ratio instead of
    // reading it as an acceleration that cancelled.
    let integrator_error = if a0_max == 0.0 { 0.0 } else { b6_max / a0_max };
    let dt_new = if integrator_error > 0.0 && integrator_error.is_finite() {
        dt * (EPSILON / integrator_error).powf(1.0 / 7.0)
    } else if integrator_error == 0.0 {
        dt / SAFETY // forces cancel to rounding — grow freely
    } else {
        f64::NAN // diverged: a step the driver is required to refuse
    };
    (a_last, dt_new)
}

/// Integrate the second-order system `r'' = a(r)` from `t0` to `t1` with adaptive IAS15.
///
/// `accel` is the position-only acceleration; `dt0` is the initial step guess (the controller takes
/// over immediately). Returns the final `(r, v)`.
///
/// `max_steps` bounds the passes — accepted plus rejected — the controller may take. There is no
/// tolerance and no step floor to pass alongside it: IAS15 holds its own fractional error at
/// `EPSILON`, and a step is refused for not moving the clock rather than for being small, so the
/// ceiling is the one limit a caller can set. Pass
/// [`DEFAULT_MAX_STEPS`](crate::driver::DEFAULT_MAX_STEPS) unless the run is known to need more.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if the span, the initial step or `max_steps` is not
/// positive;
/// [`IntegrationError::StepUnderflow`] if the controller reaches a step that cannot advance `t`,
/// which is what an approach to a singularity looks like from inside the loop;
/// [`IntegrationError::StepLimit`] if the run needs more than `max_steps` passes.
pub fn integrate_ias15<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    r0: &[f64; D],
    v0: &[f64; D],
    t0: f64,
    t1: f64,
    dt0: f64,
    max_steps: u32,
) -> Result<([f64; D], [f64; D]), IntegrationError> {
    if !(t1 > t0 && dt0 > 0.0 && max_steps > 0) {
        return Err(IntegrationError::BadArguments);
    }
    let rr = rr_table();

    let mut r = *r0;
    let mut v = *v0;
    let mut csr = [0.0; D]; // Kahan compensation carried across steps
    let mut csv = [0.0; D];
    let mut b = [[0.0; D]; NB]; // acceleration-fit coefficients (warm-started between steps)
    let mut e = [[0.0; D]; NB]; // predicted b's (the "e" array), for the warm-start correction

    let mut t = t0;
    let mut dt = dt0.min(t1 - t0);
    let mut passes = 0u32;

    while t < t1 {
        if passes >= max_steps {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(passes),
            });
        }
        passes += 1;
        // Never overshoot the endpoint. The loop needs no tolerance around `t1` on top of this:
        // once `t` and `t1` are within a factor of two, Sterbenz makes `t1 - t` exact, so the
        // clamped final step lands on `t1` bit for bit. A tolerance would instead stop the run up
        // to `1e-15 · span` short of the endpoint it was asked for.
        dt = dt.min(t1 - t);
        let a0 = accel(&r);

        // Predictor–corrector: converge the b-coefficients and get the suggested next step.
        let (_a_last, dt_new) = ias15_pc(accel, &r, &v, &a0, dt, &mut b, &rr);

        // Reject and retry with the smaller step if it shrank too much — the clamped final step
        // included, because the dynamics can sharpen inside the remaining span and a controller
        // asking for a quarter of the step it was given is reporting error, not a shorter horizon.
        if dt_new / dt < SAFETY {
            let ratio = dt_new / dt;
            predict_next_step(ratio, &mut b, &mut e);
            dt = dt_new;
            continue;
        }

        // A non-finite suggestion is the corrector reporting that the state has left the reals; it
        // reaches here as a NaN rather than as a growth factor precisely so that the run ends at the
        // first step that diverged instead of marching to `t1` and returning `Ok` on a NaN state.
        if !(dt_new.is_finite() && step_advances(t, dt)) {
            return Err(IntegrationError::StepUnderflow { t });
        }

        // --- Accept: advance (r, v) over the full step with compensated summation ---------------
        for i in 0..D {
            add_cs(
                &mut r[i],
                &mut csr[i],
                v[i] * dt
                    + dt * dt
                        * (a0[i] / 2.0
                            + b[0][i] / 6.0
                            + b[1][i] / 12.0
                            + b[2][i] / 20.0
                            + b[3][i] / 30.0
                            + b[4][i] / 42.0
                            + b[5][i] / 56.0
                            + b[6][i] / 72.0),
            );
            add_cs(
                &mut v[i],
                &mut csv[i],
                dt * (a0[i]
                    + b[0][i] / 2.0
                    + b[1][i] / 3.0
                    + b[2][i] / 4.0
                    + b[3][i] / 5.0
                    + b[4][i] / 6.0
                    + b[5][i] / 7.0
                    + b[6][i] / 8.0),
            );
        }
        t += dt;

        // Cap growth to ×(1/SAFETY); warm-start the next step's b's from this step's converged ones.
        let mut dt_next = dt_new;
        if dt_next / dt > 1.0 / SAFETY {
            dt_next = dt / SAFETY;
        }
        predict_next_step(dt_next / dt, &mut b, &mut e);
        dt = dt_next;
    }

    Ok((r, v))
}

/// Rescale the `b` coefficients from the just-finished step's grid to the next step of ratio `q`
/// (Everhart's warm start): the polynomial fit is re-expanded for the new step size, carrying the
/// previous `(b − e)` correction.
///
/// Everhart discards the old fit outright above a step ratio of about twenty, where re-expanding it
/// is worthless. This driver cannot reach that ratio and so does not carry the branch: the accept
/// path caps growth at `1 / SAFETY` before calling, and the reject path calls with a ratio the
/// branch it sits in has just measured below `SAFETY`.
fn predict_next_step<const D: usize>(q: f64, b: &mut [[f64; D]; NB], e: &mut [[f64; D]; NB]) {
    debug_assert!(q <= 1.0 / SAFETY, "step ratio {q} escaped the growth cap");
    let (q2, q3, q4, q5, q6, q7) = (q * q, q * q * q, q.powi(4), q.powi(5), q.powi(6), q.powi(7));
    for i in 0..D {
        let (b0, b1, b2, b3, b4, b5, b6) = (
            b[0][i], b[1][i], b[2][i], b[3][i], b[4][i], b[5][i], b[6][i],
        );
        let be = [
            b0 - e[0][i],
            b1 - e[1][i],
            b2 - e[2][i],
            b3 - e[3][i],
            b4 - e[4][i],
            b5 - e[5][i],
            b6 - e[6][i],
        ];
        e[0][i] = q * (7.0 * b6 + 6.0 * b5 + 5.0 * b4 + 4.0 * b3 + 3.0 * b2 + 2.0 * b1 + b0);
        e[1][i] = q2 * (21.0 * b6 + 15.0 * b5 + 10.0 * b4 + 6.0 * b3 + 3.0 * b2 + b1);
        e[2][i] = q3 * (35.0 * b6 + 20.0 * b5 + 10.0 * b4 + 4.0 * b3 + b2);
        e[3][i] = q4 * (35.0 * b6 + 15.0 * b5 + 5.0 * b4 + b3);
        e[4][i] = q5 * (21.0 * b6 + 6.0 * b5 + b4);
        e[5][i] = q6 * (7.0 * b6 + b5);
        e[6][i] = q7 * b6;
        for m in 0..NB {
            b[m][i] = e[m][i] + be[m];
        }
    }
}

/// IAS15's native dense output over one step: the degree-7 acceleration fit integrated to position
/// (degree 9) and velocity (degree 8) — the same order-15 polynomial the step itself uses, evaluated
/// at any interior fraction. The state is `[r, v]`, so `N2 = 2D`; the caller supplies it.
#[derive(Debug, Clone, Copy)]
pub struct Ias15Dense<const D: usize> {
    t0: f64,
    dt: f64,
    r0: [f64; D],
    v0: [f64; D],
    a0: [f64; D],
    b: [[f64; D]; NB],
}

impl<const D: usize, const N2: usize> DenseOutput<N2> for Ias15Dense<D> {
    fn eval(&self, t: f64, out: &mut [f64; N2]) {
        assert!(N2 == 2 * D, "IAS15 dense state must be [r; v] with N2 = 2D");
        let s = if self.dt == 0.0 {
            0.0
        } else {
            (t - self.t0) / self.dt
        };
        let (dt, b, a0) = (self.dt, &self.b, &self.a0);
        for i in 0..D {
            let dx = ((((((((b[6][i] * 7.0 * s / 9.0 + b[5][i]) * 3.0 * s / 4.0 + b[4][i])
                * 5.0
                * s
                / 7.0
                + b[3][i])
                * 2.0
                * s
                / 3.0
                + b[2][i])
                * 3.0
                * s
                / 5.0
                + b[1][i])
                * s
                / 2.0
                + b[0][i])
                * s
                / 3.0
                + a0[i])
                * dt
                * s
                / 2.0
                + self.v0[i])
                * dt
                * s;
            out[i] = self.r0[i] + dx;
            let dv =
                (((((((b[6][i] * 7.0 * s / 8.0 + b[5][i]) * 6.0 * s / 7.0 + b[4][i]) * 5.0 * s
                    / 6.0
                    + b[3][i])
                    * 4.0
                    * s
                    / 5.0
                    + b[2][i])
                    * 3.0
                    * s
                    / 4.0
                    + b[1][i])
                    * 2.0
                    * s
                    / 3.0
                    + b[0][i])
                    * s
                    / 2.0
                    + a0[i])
                    * dt
                    * s;
            out[D + i] = self.v0[i] + dv;
        }
    }
}

/// Integrate as [`integrate_ias15`], returning a dense [`Trajectory`] over the `[r, v]` state
/// (`N2 = 2D`), each segment the step's order-15 Gauss–Radau polynomial — for the event detector.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] if `N2 ≠ 2D` or the span, initial step or `max_steps` is
/// not positive; the
/// same [`IntegrationError::StepUnderflow`] and [`IntegrationError::StepLimit`] as
/// [`integrate_ias15`], the ceiling here bounding the retained segments as well as the work.
pub fn integrate_ias15_dense<const D: usize, const N2: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    r0: &[f64; D],
    v0: &[f64; D],
    t0: f64,
    t1: f64,
    dt0: f64,
    max_steps: u32,
) -> Result<Trajectory<N2, Ias15Dense<D>>, IntegrationError> {
    if N2 != 2 * D || !(t1 > t0 && dt0 > 0.0 && max_steps > 0) {
        return Err(IntegrationError::BadArguments);
    }
    let rr = rr_table();
    let mut r = *r0;
    let mut v = *v0;
    let mut csr = [0.0; D];
    let mut csv = [0.0; D];
    let mut b = [[0.0; D]; NB];
    let mut e = [[0.0; D]; NB];
    let mut t = t0;
    let mut dt = dt0.min(t1 - t0);
    let mut segments = Vec::new();
    let mut ends = Vec::new();
    let mut rejected = 0u32;

    while t < t1 {
        if rejected + segments.len() as u32 >= max_steps {
            return Err(IntegrationError::StepLimit {
                steps: f64::from(rejected) + segments.len() as f64,
            });
        }
        // Exact for the same reason as the plain driver, and load-bearing here: the segments have
        // to reach `t1` or `sample` extrapolates off the end of the last one.
        dt = dt.min(t1 - t);
        let a0 = accel(&r);
        let (_a_last, dt_new) = ias15_pc(accel, &r, &v, &a0, dt, &mut b, &rr);
        if dt_new / dt < SAFETY {
            rejected += 1;
            predict_next_step(dt_new / dt, &mut b, &mut e);
            dt = dt_new;
            continue;
        }
        // A non-finite suggestion is the corrector reporting that the state has left the reals; it
        // reaches here as a NaN rather than as a growth factor precisely so that the run ends at the
        // first step that diverged instead of marching to `t1` and returning `Ok` on a NaN state.
        if !(dt_new.is_finite() && step_advances(t, dt)) {
            return Err(IntegrationError::StepUnderflow { t });
        }
        // Capture the step's dense segment before advancing (r, v are still at the step start).
        segments.push(Ias15Dense {
            t0: t,
            dt,
            r0: r,
            v0: v,
            a0,
            b,
        });
        for i in 0..D {
            add_cs(
                &mut r[i],
                &mut csr[i],
                v[i] * dt
                    + dt * dt
                        * (a0[i] / 2.0
                            + b[0][i] / 6.0
                            + b[1][i] / 12.0
                            + b[2][i] / 20.0
                            + b[3][i] / 30.0
                            + b[4][i] / 42.0
                            + b[5][i] / 56.0
                            + b[6][i] / 72.0),
            );
            add_cs(
                &mut v[i],
                &mut csv[i],
                dt * (a0[i]
                    + b[0][i] / 2.0
                    + b[1][i] / 3.0
                    + b[2][i] / 4.0
                    + b[3][i] / 5.0
                    + b[4][i] / 6.0
                    + b[5][i] / 7.0
                    + b[6][i] / 8.0),
            );
        }
        t += dt;
        ends.push(t);
        let mut dt_next = dt_new;
        if dt_next / dt > 1.0 / SAFETY {
            dt_next = dt / SAFETY;
        }
        predict_next_step(dt_next / dt, &mut b, &mut e);
        dt = dt_next;
    }

    let mut y_final = [0.0; N2];
    y_final[..D].copy_from_slice(&r);
    y_final[D..2 * D].copy_from_slice(&v);
    let steps = segments.len() as u32;
    Ok(Trajectory::from_segments(
        t0, t1, y_final, segments, ends, steps, rejected,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compensated_addition_survives_a_summand_larger_than_the_running_total() {
        // Neumaier's counterexample summed in order: 1, 1e100, 1, -1e100. The two huge terms cancel
        // exactly and the naive total loses both ones to rounding -- it comes back +0.0. The
        // compensated total is exactly 2.0, hand-derived from IEEE-754 rather than read off a run:
        // the ulp at 1e100 is ~1e84, so `1e100 + 1` rounds back to `1e100` and the whole `1` has to
        // land in `c`.
        //
        // The sequence takes *both* arms on purpose. `|y| >= |inc|` is false for the first two terms
        // and true for the last two, and the false arm is the one plain Kahan does not have: this is
        // Neumaier, and without it the second term's `1` is lost outright.
        let mut y = 0.0;
        let mut c = 0.0;
        let mut naive = 0.0;
        for inc in [1.0, 1e100, 1.0, -1e100] {
            add_cs(&mut y, &mut c, inc);
            naive += inc;
        }
        assert_eq!(
            naive.to_bits(),
            0.0_f64.to_bits(),
            "the naive sum kept {naive:.17e}"
        );
        assert_eq!(
            (y + c).to_bits(),
            2.0_f64.to_bits(),
            "compensation lost: y = {y:.17e}, c = {c:.17e}"
        );
    }

    #[test]
    fn only_a_step_that_moves_the_clock_counts_as_progress() {
        // A day into a J2000-second epoch: 45 ns is a legal step there and 4.5 ns is not, because
        // ulp(86400) is 1.455e-11 s. The pair is the point -- the threshold is the epoch's
        // resolution, not a constant, which is why no fixed floor can express it.
        let t = 86_400.0;
        assert!(step_advances(t, 4.5e-11));
        assert!(!step_advances(t, 4.5e-12));

        // Half an ulp rounds to even and lands back on `t`; the whole ulp is the first step that
        // survives. Both bound the same thing from either side.
        let ulp = f64::from_bits(t.to_bits() + 1) - t;
        assert!(step_advances(t, ulp));
        assert!(!step_advances(t, ulp / 2.0));

        // The three the affirmative form is written for. `dt < min_step` accepts every one of them:
        // NaN compares false against anything, and neither zero nor a negative step is small.
        assert!(!step_advances(t, f64::NAN));
        assert!(!step_advances(t, 0.0));
        assert!(!step_advances(t, -1.0));
    }
}
