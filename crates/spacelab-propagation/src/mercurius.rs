//! MERCURIUS — a hybrid symplectic/adaptive integrator: the Wisdom–Holman map far from a close
//! encounter, IAS15 near it, joined by splitting the interaction with a smooth changeover
//! (Rein et al. 2019).
//!
//! # Symplectic far, adaptive near
//!
//! A symplectic map like [WHFast](crate::whfast) is superb for a smooth, nearly-Keplerian orbit —
//! bounded energy error over billions of revolutions — but it takes a *fixed* step, so it cannot
//! resolve a sudden close encounter, where the interaction spikes and the trajectory turns sharply.
//! An adaptive integrator like [IAS15](crate::ias15) resolves the encounter to machine precision but
//! is not symplectic, so using it everywhere loses the long-term energy behaviour. MERCURIUS keeps
//! the best of both.
//!
//! # The splitting, not a switch
//!
//! The changeover `K(d)` is a `C²` partition of unity — the smootherstep polynomial `6y⁵−15y⁴+10y³`
//! on the transition band `[r_lo, r_hi]`, `K = 1` (fully far) beyond `r_hi` and `K = 0` (fully near)
//! inside `r_lo`, with the first two derivatives vanishing at both ends (checked in
//! `derivations/mercurius_changeover.mac`).
//!
//! What `K` splits is the **interaction potential**, not the choice of method:
//!
//! ```text
//! H  =  [ H_Kepler + (1 − K(d)) V_int ]  +  [ K(d) V_int ]
//!         \_______ the near part _______/     \__ the far part __/
//! ```
//!
//! The near part is integrated by IAS15 — it carries the whole close encounter, so it is the part
//! that needs an adaptive step — and the far part, a position-only potential, is a kick. The map is
//! then one composition, `drift(h/2) · kick(h) · drift(h/2)`, applied at *every* step, with `K`
//! evaluated at the current position inside both halves. Where `K = 1` the near part is exactly
//! Keplerian and the kick is the whole perturbation, which is the WHFast step; where `K = 0` the
//! kick vanishes and IAS15 carries everything. Nothing switches, so nothing is discontinuous.
//!
//! Splitting the *step* instead — WHFast when `K = 1`, IAS15 when `K < 1` — is the natural
//! simplification, and this module used to do it. It is wrong for a reason worth recording. A
//! symplectic map's energy error is bounded but *oscillating*, peaking where the field is strongest:
//! on a Molniya orbit under `J₂` at `T/256` WHFast's relative energy error is `1e-12` sampled once
//! per revolution at a fixed phase and `6e-5` sampled anywhere, with the maximum at periapsis.
//! Handing the state to a non-symplectic integrator freezes whatever the oscillation happens to be
//! at the handover, and a proximity-triggered handover is at periapsis by construction.
//!
//! Measured against this implementation on that orbit, 20 revolutions at `T/256`, sweeping the
//! outer edge of the changeover band in units of the periapsis radius (energy error, step-split vs
//! force-split): `1.2` — `8.2e-5` vs `1.8e-4`; `1.6` — `2.7e-5` vs `8.8e-7`; `2.0` — `6.2e-7` vs
//! `2.7e-7`; `3.0` — `3.7e-7` vs `1.8e-8`; `4.0` — `2.9e-8` vs `5.6e-9`; `6.0` — `1.6e-12` vs
//! `1.2e-9`. On the close encounter of `tests/mercurius.rs` the step-split reaches `46 m` from a
//! tight reference and the force-split `8.5 m`.
//!
//! Two things in those numbers. The force splitting wins the case the method exists for and four of
//! the six band widths. And where the step splitting wins — the widest band, where 62% of steps are
//! handed over — it wins by ceasing to be a hybrid: almost every step is then pure IAS15 on the full
//! force, which is accurate over 20 revolutions and gives up exactly the symplecticity the far part
//! is there to keep. A hybrid that improves by degenerating into one of its halves is not being
//! measured on the property it was built for.
//!
//! The composition below reduces *exactly* to WHFast's `drift–kick–drift` in the far limit, rather
//! than to some other second-order composition of the same two pieces. That matters: `kick–drift–kick`
//! and `drift–kick–drift` are both second order and both symplectic, but their modified Hamiltonians
//! differ at `O(h²)`, so alternating between them across a step boundary would reintroduce the same
//! frozen-offset error in miniature.
//!
//! The one place a switch survives is the fast path: when `K = 1` at the start of a step the near
//! Hamiltonian is exactly Keplerian and the analytic Kepler drift replaces the two IAS15 calls. The
//! two agree to IAS15's own per-step error rather than to the oscillation amplitude, because they
//! integrate the same splitting.
//!
//! The perturbation and the encounter measure must be functions of position (as for IAS15).

#![allow(
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use crate::driver::{DEFAULT_MAX_STEPS, IntegrationError, step_count};
use crate::ias15::integrate_ias15;
use crate::whfast::kepler_drift;

/// The smootherstep changeover `K(d) ∈ [0,1]`: `1` (fully far) for `d ≥ r_hi`, `0` (fully near) for
/// `d ≤ r_lo`, and the `C²` polynomial `6y⁵−15y⁴+10y³` between.
#[must_use]
pub fn changeover(d: f64, r_lo: f64, r_hi: f64) -> f64 {
    if d >= r_hi {
        1.0
    } else if d <= r_lo {
        0.0
    } else {
        let y = (d - r_lo) / (r_hi - r_lo);
        6.0 * y.powi(5) - 15.0 * y.powi(4) + 10.0 * y.powi(3)
    }
}

/// Integrate a satellite about central mass `mu` under a position-only perturbation `pert`,
/// splitting that perturbation by the [`changeover`] of the encounter measure `encounter` (a
/// distance that shrinks as an encounter approaches) over the transition band `[r_lo, r_hi]`.
///
/// Returns the final `(r, v)` and the number of steps in which any part of the perturbation was
/// carried by IAS15 rather than kicked — the steps for which the encounter was close enough to
/// matter.
///
/// # Errors
///
/// [`IntegrationError::BadArguments`] for a non-positive span/step, or a state with no osculating
/// orbit for the far-field Kepler drift; propagates an IAS15 failure.
pub fn integrate_mercurius<Pert, Enc>(
    mu: f64,
    pert: &Pert,
    encounter: &Enc,
    r_lo: f64,
    r_hi: f64,
    r0: &[f64; 3],
    v0: &[f64; 3],
    t0: f64,
    t1: f64,
    dt: f64,
) -> Result<([f64; 3], [f64; 3], u32), IntegrationError>
where
    Pert: Fn(&[f64; 3]) -> [f64; 3],
    Enc: Fn(&[f64; 3]) -> f64,
{
    if !(t1 > t0 && dt > 0.0 && mu > 0.0) {
        return Err(IntegrationError::BadArguments);
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss
    )]
    let n_steps = step_count(t0, t1, dt, 1)?;
    #[allow(clippy::cast_precision_loss)]
    let step = (t1 - t0) / n_steps as f64;

    // The far half of the interaction: the fraction `K` of the perturbation that the kick carries.
    let far = |rr: &[f64; 3]| {
        let k = changeover(encounter(rr), r_lo, r_hi);
        let p = pert(rr);
        [k * p[0], k * p[1], k * p[2]]
    };
    // The near half: the central term plus the remaining fraction `1 − K`, which is what IAS15
    // integrates. At `K = 1` this is exactly the Kepler problem.
    let near_force = |rr: &[f64; 3]| {
        let inv_r3 = (rr[0] * rr[0] + rr[1] * rr[1] + rr[2] * rr[2]).powf(-1.5);
        let k = changeover(encounter(rr), r_lo, r_hi);
        let p = pert(rr);
        [
            -mu * rr[0] * inv_r3 + (1.0 - k) * p[0],
            -mu * rr[1] * inv_r3 + (1.0 - k) * p[1],
            -mu * rr[2] * inv_r3 + (1.0 - k) * p[2],
        ]
    };

    let mut r = *r0;
    let mut v = *v0;
    let mut t = t0;
    let mut near = 0u32;
    let half = 0.5 * step;
    for _ in 0..n_steps {
        let mut adaptive = false;
        // Two half-drifts around one kick. The second is decided on its own position, not on the
        // first half's: the encounter may have opened or closed across the kick, and a stale
        // verdict would put the two halves of one step on different sides of the changeover.
        for k in 0..2 {
            let from = t + f64::from(k) * half;
            // The analytic drift is only the right answer if the changeover stays at one for the
            // whole half-step, and the *end* of the half-step is where it stops being. Rather than
            // predict that with a second solve, drift speculatively and test where it landed: the
            // drift is the one that would have been taken anyway, so a far half-step pays nothing
            // for the check and a near one throws away one Kepler solve.
            let mut trial_r = r;
            let mut trial_v = v;
            let stays_far = if changeover(encounter(&r), r_lo, r_hi) >= 1.0 {
                kepler_drift(mu, from, &mut trial_r, &mut trial_v, half)?;
                changeover(encounter(&trial_r), r_lo, r_hi) >= 1.0
            } else {
                false
            };
            if stays_far {
                r = trial_r;
                v = trial_v;
            } else {
                adaptive = true;
                let (rn, vn) = integrate_ias15(
                    &near_force,
                    &r,
                    &v,
                    from,
                    from + half,
                    half,
                    DEFAULT_MAX_STEPS,
                )?;
                r = rn;
                v = vn;
            }
            if k == 0 {
                let acc = far(&r);
                for i in 0..3 {
                    v[i] += acc[i] * step;
                }
            }
        }
        near += u32::from(adaptive);
        t += step;
    }
    Ok((r, v, near))
}
