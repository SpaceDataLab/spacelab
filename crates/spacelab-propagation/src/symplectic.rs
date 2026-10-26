//! Symplectic integrators for a separable Hamiltonian `H = T(v) + V(x)` — the structure a
//! gravitational problem has, where the force depends on position alone.
//!
//! # Why a different integrator at all
//!
//! An adaptive Runge–Kutta like [`crate::Dop853`] is more accurate per step, but its energy error
//! *drifts*: integrate a Kepler orbit for ten thousand periods and the semi-major axis walks away
//! linearly with time. A symplectic integrator preserves the geometric (symplectic) structure of the
//! flow exactly, so it conserves a *shadow* Hamiltonian close to the true one — its energy error
//! **oscillates within a fixed bound forever** and never drifts. For long-term dynamics — resonances,
//! stability studies, million-orbit ensembles — that bounded error is worth more than a smaller
//! per-step one. The price is a fixed step and a separable, velocity-independent force: these apply
//! to gravity (`PointMass`, `Zonal`, `ThirdBody`), **not** to drag, whose `v`-dependence breaks the
//! separability the method assumes.
//!
//! The force is supplied as a closure `x ↦ a(x)`; wrap a conservative [`crate::forces::Acceleration`]
//! by evaluating it at any velocity (it will ignore one).
//!
//! Gate: `derivations/symplectic_composition.mac` proves the orders in the free associative algebra
//! of drift and kick, exactly — leapfrog symmetric of order 2, `yoshida4` of order 4 with `2^{1/3}`
//! carried as an algebraic number, and the triple jump `S(z)S(1−2z)S(z)` leaving only the
//! `2z^q + (1−2z)^q` error that the closed-form `z` cancels for `q = 3, 5, 7`;
//! `derivations/symplectic_composition.gp` checks the fraction lists `triple_jump` builds at 100
//! digits, exact through words of length 2, 4, 6, 8 and off at the next. The module tests
//! `one_step_on_the_oscillator_is_the_composition_map` (G3) and
//! `the_oscillator_map_preserves_area_and_reverses` (G4) hold the shipped functions to that map.

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]

#[cfg(feature = "std")]
use crate::driver::{HermiteDense, Trajectory};

use spacelab_autodiff::Scalar;

/// Compensated (Kahan) addition `*y += inc`, carrying the lost low bits in `*c`.
///
/// A symplectic integrator's guarantee is a *bounded* energy error over arbitrarily long times. Naive
/// summation defeats it: each `x += h·v` rounds at `ε·|x|`, and over the `~10⁸` steps of a
/// million-orbit run that random-walks into a secular drift — reintroducing exactly the energy error
/// the method exists to eliminate. Carrying the compensation keeps the accumulation error at `~ε·|x|`
/// forever (Hairer–Lubich–Wanner, *Geometric Numerical Integration* §VIII.5).
#[inline]
fn kahan_add(y: &mut f64, c: &mut f64, inc: f64) {
    let inc_c = inc + *c;
    let y_new = *y + inc_c;
    *c = (*y - y_new) + inc_c;
    *y = y_new;
}

/// One velocity-Verlet (kick–drift–kick) step: `v ← v + ½h·a(x)`, `x ← x + h·v`, `v ← v + ½h·a(x)`,
/// each accumulation compensated through `cx`/`cv`. Symmetric in time, and symplectic — the building
/// block both integrators below are made of.
#[inline]
fn verlet_step<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x: &mut [f64; D],
    v: &mut [f64; D],
    cx: &mut [f64; D],
    cv: &mut [f64; D],
    dt: f64,
) {
    let a0 = accel(x);
    for i in 0..D {
        kahan_add(&mut v[i], &mut cv[i], 0.5 * dt * a0[i]);
        kahan_add(&mut x[i], &mut cx[i], dt * v[i]);
    }
    let a1 = accel(x);
    for i in 0..D {
        kahan_add(&mut v[i], &mut cv[i], 0.5 * dt * a1[i]);
    }
}

/// Velocity Verlet (leapfrog): second-order, symplectic, one amortised force evaluation per step.
///
/// Advances `(x, v)` by `steps` fixed steps of `dt`. The carried acceleration means the interior
/// half-kicks fuse into full kicks, so only one `accel` call happens per step after the first. The
/// accumulation is Kahan-compensated (see `kahan_add`); the compensation is per call, so a long arc
/// wants one call with a large `steps`, not many one-step calls.
pub fn leapfrog<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x: &mut [f64; D],
    v: &mut [f64; D],
    dt: f64,
    steps: usize,
) {
    let mut a = accel(x);
    let mut cx = [0.0; D];
    let mut cv = [0.0; D];
    for _ in 0..steps {
        for i in 0..D {
            kahan_add(&mut v[i], &mut cv[i], 0.5 * dt * a[i]);
            kahan_add(&mut x[i], &mut cx[i], dt * v[i]);
        }
        a = accel(x);
        for i in 0..D {
            kahan_add(&mut v[i], &mut cv[i], 0.5 * dt * a[i]);
        }
    }
    for i in 0..D {
        x[i] += cx[i];
        v[i] += cv[i];
    }
}

/// Yoshida's fourth-order symplectic integrator: the composition `S(w₁h) ∘ S(w₀h) ∘ S(w₁h)` of the
/// Verlet step `S`, with `w₁ = 1/(2 − 2^{1/3})` and `w₀ = −2^{1/3} w₁` — the unique coefficients that
/// cancel the third-order error of the symmetric second-order method (Yoshida 1990).
///
/// The coefficients are a closed form, computed here from `2^{1/3}` at machine precision, never a
/// truncated table.
pub fn yoshida4<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x: &mut [f64; D],
    v: &mut [f64; D],
    dt: f64,
    steps: usize,
) {
    let cbrt2 = libm::cbrt(2.0);
    let w1 = 1.0 / (2.0 - cbrt2);
    let w0 = -cbrt2 * w1;
    compose(accel, x, v, &[w1, w0, w1], dt, steps);
}

/// Apply the palindromic sequence of Verlet sub-step fractions `fracs` (summing to 1) `steps` times,
/// sharing one Kahan compensation across the whole run. Every symmetric composition below is one such
/// sequence; the composition is exact, at the cost of a redundant force evaluation at each internal
/// junction that a fused implementation would save.
fn compose<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x: &mut [f64; D],
    v: &mut [f64; D],
    fracs: &[f64],
    dt: f64,
    steps: usize,
) {
    let mut cx = [0.0; D];
    let mut cv = [0.0; D];
    for _ in 0..steps {
        for &frac in fracs {
            verlet_step(accel, x, v, &mut cx, &mut cv, frac * dt);
        }
    }
    for i in 0..D {
        x[i] += cx[i];
        v[i] += cv[i];
    }
}

/// Triple-jump one symmetric composition into the next even order: `S₂ₖ₊₂(h) = S₂ₖ(z h) S₂ₖ((1−2z) h)
/// S₂ₖ(z h)` with `z = 1/(2 − 2^{1/(2k+1)})`, the coefficient that cancels the leading `(2k+1)`th-order
/// error (Suzuki 1990, Yoshida 1990). Returns the fraction list of the higher-order method.
fn triple_jump<const M: usize>(base: &[f64], order: u32) -> [f64; M] {
    let z = 1.0 / (2.0 - libm::pow(2.0, 1.0 / f64::from(order + 1)));
    let outer = [z, 1.0 - 2.0 * z, z];
    let mut out = [0.0; M];
    let mut k = 0;
    for scale in outer {
        for &c in base {
            out[k] = scale * c;
            k += 1;
        }
    }
    out
}

/// Sixth-order symplectic integrator: the triple-jump of [`yoshida4`], nine Verlet sub-steps per step
/// with closed-form coefficients (`z = 1/(2 − 2^{1/5})`), verified to sixth order by the order test.
pub fn yoshida6<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x: &mut [f64; D],
    v: &mut [f64; D],
    dt: f64,
    steps: usize,
) {
    let cbrt2 = libm::cbrt(2.0);
    let w1 = 1.0 / (2.0 - cbrt2);
    let s4 = [w1, -cbrt2 * w1, w1];
    let s6: [f64; 9] = triple_jump(&s4, 4);
    compose(accel, x, v, &s6, dt, steps);
}

/// Eighth-order symplectic integrator: the triple-jump of [`yoshida6`], twenty-seven Verlet sub-steps
/// per step (`z = 1/(2 − 2^{1/7})`), verified to eighth order by the order test.
pub fn yoshida8<const D: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x: &mut [f64; D],
    v: &mut [f64; D],
    dt: f64,
    steps: usize,
) {
    let cbrt2 = libm::cbrt(2.0);
    let w1 = 1.0 / (2.0 - cbrt2);
    let s4 = [w1, -cbrt2 * w1, w1];
    let s6: [f64; 9] = triple_jump(&s4, 4);
    let s8: [f64; 27] = triple_jump(&s6, 6);
    compose(accel, x, v, &s8, dt, steps);
}

/// One velocity-Verlet step over any [`Scalar`] — the [`yoshida8_scalar`] building block.
///
/// Plain (uncompensated) summation: in double-double the ~106-bit mantissa already carries far more
/// than Kahan recovers in `f64`, so the [`kahan_add`] the `f64` path needs is dead weight here. The
/// time sub-steps are `f64` method constants lifted into `S`.
fn verlet_step_scalar<const D: usize, S: Scalar, F: Fn(&[S; D]) -> [S; D]>(
    accel: &F,
    x: &mut [S; D],
    v: &mut [S; D],
    dt: f64,
) {
    let half = S::from_f64(0.5 * dt);
    let full = S::from_f64(dt);
    let a0 = accel(x);
    for i in 0..D {
        v[i] = v[i] + half * a0[i];
        x[i] = x[i] + full * v[i];
    }
    let a1 = accel(x);
    for i in 0..D {
        v[i] = v[i] + half * a1[i];
    }
}

/// Eighth-order symplectic integrator over any [`Scalar`] — the precision-generic twin of
/// [`yoshida8`].
///
/// Running it with [`spacelab_autodiff::DoubleDouble`] as `S` lifts the whole propagation to ~32
/// digits, so the round-off floor that caps an `f64` orbit near a micrometre a day drops below a
/// nanometre and the accuracy becomes truncation-limited — this is the switchable-precision path. The
/// `f64` [`yoshida8`] keeps its Kahan compensation and stays the default; this trades that for `S`.
/// Same coefficients, computed from `2^{1/(2k+1)}` at machine precision.
pub fn yoshida8_scalar<const D: usize, S: Scalar, F: Fn(&[S; D]) -> [S; D]>(
    accel: &F,
    x: &mut [S; D],
    v: &mut [S; D],
    dt: f64,
    steps: usize,
) {
    let cbrt2 = libm::cbrt(2.0);
    let w1 = 1.0 / (2.0 - cbrt2);
    let s4 = [w1, -cbrt2 * w1, w1];
    let s6: [f64; 9] = triple_jump(&s4, 4);
    let s8: [f64; 27] = triple_jump(&s6, 6);
    for _ in 0..steps {
        for &frac in &s8 {
            verlet_step_scalar(accel, x, v, frac * dt);
        }
    }
}

/// Advance `steps` velocity-Verlet steps as [`leapfrog`], collecting a cubic-Hermite [`Trajectory`]
/// over the `[x, v]` state (`N2 = 2D`) for the event detector. `t0` is the start time; the state is
/// autonomous, so time only labels the segments.
///
/// Each step is a self-contained `leapfrog` call, so the cross-step Kahan compensation of a single
/// long call is not carried here — adequate for a bounded event-detection span, not for a
/// million-orbit energy study (use [`leapfrog`] directly for that).
///
/// # Panics
///
/// If `N2 != 2·D` or `steps == 0`.
#[cfg(feature = "std")]
pub fn leapfrog_dense<const D: usize, const N2: usize, F: Fn(&[f64; D]) -> [f64; D]>(
    accel: &F,
    x0: &[f64; D],
    v0: &[f64; D],
    t0: f64,
    dt: f64,
    steps: usize,
) -> Trajectory<N2, HermiteDense<N2>> {
    assert!(
        N2 == 2 * D && steps >= 1,
        "leapfrog_dense needs N2 = 2D and steps >= 1"
    );
    let combine = |x: &[f64; D], v: &[f64; D]| {
        let mut y = [0.0; N2];
        y[..D].copy_from_slice(x);
        y[D..2 * D].copy_from_slice(v);
        y
    };
    let deriv = |x: &[f64; D], v: &[f64; D]| {
        let a = accel(x);
        let mut f = [0.0; N2];
        f[..D].copy_from_slice(v);
        f[D..2 * D].copy_from_slice(&a);
        f
    };

    let mut x = *x0;
    let mut v = *v0;
    let mut t = t0;
    let mut y0v = combine(&x, &v);
    let mut f0 = deriv(&x, &v);
    let mut segments = std::vec::Vec::with_capacity(steps);
    let mut ends = std::vec::Vec::with_capacity(steps);
    for _ in 0..steps {
        leapfrog(accel, &mut x, &mut v, dt, 1);
        t += dt;
        let y1v = combine(&x, &v);
        let f1 = deriv(&x, &v);
        segments.push(HermiteDense::new(t - dt, t, y0v, f0, y1v, f1));
        ends.push(t);
        y0v = y1v;
        f0 = f1;
    }
    Trajectory::from_segments(
        t0,
        t0 + steps as f64 * dt,
        y0v,
        segments,
        ends,
        steps as u32,
        0,
    )
}

#[cfg(test)]
mod gate_tests {
    #![allow(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        // the force must match the integrators' `Fn(&[f64; D]) -> [f64; D]` bound, by reference
        // for every D including the 8-byte D = 1
        clippy::trivially_copy_pass_by_ref
    )]
    use super::{leapfrog, yoshida4, yoshida6, yoshida8, yoshida8_scalar};

    /// A method applied to the unit oscillator: `(x, v, h, steps)`.
    type Method = fn(&mut [f64; 1], &mut [f64; 1], f64, usize);

    const H: f64 = 0.5;

    /// Each method's one-step map on `x'' = −x` at `h = 1/2`, `[M₁₁, M₂₁, M₁₂, M₂₂]` with the state
    /// as the column `(x, v)`: the product of its sub-steps' kick `[1, 0; −c, 1]` and drift
    /// `[1, c; 0, 1]` matrices over the closed-form fractions, printed by
    /// `derivations/symplectic_composition.gp` at 100 digits. `yoshida8_scalar` at `f64` shares
    /// `yoshida8`'s fractions and so its map.
    const MAPS: [(&str, Method, [f64; 4]); 5] = [
        (
            "leapfrog",
            |x, v, h, n| leapfrog(&oscillator, x, v, h, n),
            [0.875, -0.46875, 0.5, 0.875],
        ),
        (
            "yoshida4",
            |x, v, h, n| yoshida4(&oscillator, x, v, h, n),
            [
                8.786159510339272549762827080942643433e-1,
                -4.788905408028607908718282675172684088e-1,
                4.761714654176466816225425435445030733e-1,
                8.786159510339272549762827080942643433e-1,
            ],
        ),
        (
            "yoshida6",
            |x, v, h, n| yoshida6(&oscillator, x, v, h, n),
            [
                8.774876305656666264944384095623673788e-1,
                -4.792408996882825510283214577621631432e-1,
                4.799579049990587584129801261416828851e-1,
                8.774876305656666264944384095623673788e-1,
            ],
        ),
        (
            "yoshida8",
            |x, v, h, n| yoshida8(&oscillator, x, v, h, n),
            [
                8.775925384322334959986094926583486593e-1,
                -4.794031425322119290067027665582868374e-1,
                4.794114099338137066703058211311134073e-1,
                8.775925384322334959986094926583486593e-1,
            ],
        ),
        (
            "yoshida8_scalar",
            |x, v, h, n| yoshida8_scalar(&oscillator, x, v, h, n),
            [
                8.775925384322334959986094926583486593e-1,
                -4.794031425322119290067027665582868374e-1,
                4.794114099338137066703058211311134073e-1,
                8.775925384322334959986094926583486593e-1,
            ],
        ),
    ];

    fn oscillator(x: &[f64; 1]) -> [f64; 1] {
        [-x[0]]
    }

    /// The map `method` applies over one step of `h`, read off its two unit columns.
    fn one_step(method: Method, h: f64) -> [f64; 4] {
        let mut out = [0.0; 4];
        for (k, (x0, v0)) in [(1.0, 0.0), (0.0, 1.0)].into_iter().enumerate() {
            let (mut x, mut v) = ([x0], [v0]);
            method(&mut x, &mut v, h, 1);
            out[2 * k] = x[0];
            out[2 * k + 1] = v[0];
        }
        out
    }

    /// (G3) One step of each shipped method on the unit oscillator is the exact one-step map of its
    /// composition: the fractions `[w₁, w₀, w₁]` and the two triple jumps, whose orders 2, 4, 6, 8
    /// `derivations/symplectic_composition.mac` proves in the free algebra and
    /// `derivations/symplectic_composition.gp` confirms on the very fraction lists `triple_jump`
    /// lays out (exact through words of length `p`, off at `p + 1`).
    ///
    /// The map depends on every fraction, so a wrong `z`, a sub-step out of place or a lost kick
    /// moves it by far more than the bar: at `h = 1/2` the four maps sit `2.1e-2`, `3.3e-3`,
    /// `5.3e-4` and `2.2e-5` from the rotation they approximate, and the shipped steps must land on
    /// the composition, not on the rotation, to `16 ε` — the rounding of at most 81 sub-step
    /// products of numbers below one.
    #[test]
    fn one_step_on_the_oscillator_is_the_composition_map() {
        for (name, method, want) in MAPS {
            let got = one_step(method, H);
            for k in 0..4 {
                let d = (got[k] - want[k]).abs();
                assert!(
                    d <= 16.0 * f64::EPSILON,
                    "{name} entry {k}: {:.17e} vs {:.17e}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    /// (G4) Each shipped method is symplectic and time-symmetric on the oscillator: its one-step map
    /// has determinant one (area preservation, the one-degree-of-freedom form of `MᵀJM = J`), and a
    /// step of `h` followed by a step of `−h` returns the start. Both are identities of the
    /// composition — every sub-step is a shear, and `Ψ(h)Ψ(−h) = 1`
    /// (`derivations/symplectic_composition.mac`) — so the bar is rounding alone.
    #[test]
    fn the_oscillator_map_preserves_area_and_reverses() {
        for (name, method, _) in MAPS {
            let m = one_step(method, H);
            let det = m[0] * m[3] - m[2] * m[1];
            assert!(
                (det - 1.0).abs() <= 16.0 * f64::EPSILON,
                "{name}: det {det:.17e}"
            );
            let (mut x, mut v) = ([0.3], [-0.8]);
            method(&mut x, &mut v, H, 1);
            method(&mut x, &mut v, -H, 1);
            assert!(
                (x[0] - 0.3).abs() <= 16.0 * f64::EPSILON
                    && (v[0] + 0.8).abs() <= 16.0 * f64::EPSILON,
                "{name}: returned to ({}, {})",
                x[0],
                v[0]
            );
        }
    }
}

#[cfg(test)]
mod dd_precision_tests {
    #![allow(clippy::float_cmp)]
    use super::{Scalar, yoshida8_scalar};
    use spacelab_autodiff::DoubleDouble as Dd;

    /// Propagating a circular orbit for an integer number of periods must return it to its start; the
    /// residual is the integrator's error. Running the *identical* generic method in `f64` and then in
    /// double-double isolates precision from method: the `f64` run random-walks the round-off of every
    /// `x += h·v` into a large accumulated error, while the dd run carries ~32 digits and that
    /// round-off vanishes. dd being the markedly more accurate of the two — the same code, one type
    /// parameter apart — is the switchable-precision claim, measured.
    ///
    /// The dd run does not reach dd's own floor here: the initial conditions and the timestep are `f64`
    /// (`vc`, `dt`, `period`), so the whole problem is only posed to ~2e-13 relative and dd bottoms out
    /// there, not at its ~1e-32 arithmetic floor (which `double_double.rs`'s unit tests exercise
    /// directly). What this isolates is the integration round-off, which dd removes.
    #[test]
    fn double_double_beats_the_f64_round_off_floor() {
        let mu = 3.986_004_418e14_f64;
        let r0 = 7.0e6_f64;
        let vc = (mu / r0).sqrt();
        let period = core::f64::consts::TAU * (r0 * r0 * r0 / mu).sqrt();
        let n = 1024_usize;
        let revs = 15_usize;
        let dt = period / n as f64;
        let steps = n * revs;

        // The same closure, once over f64 and once over double-double.
        let accel_f = |x: &[f64; 3]| {
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let k = -mu / (r2 * r2.sqrt());
            [k * x[0], k * x[1], k * x[2]]
        };
        let mut xf = [r0, 0.0, 0.0];
        let mut vf = [0.0, vc, 0.0];
        yoshida8_scalar(&accel_f, &mut xf, &mut vf, dt, steps);
        let err_f = ((xf[0] - r0).powi(2) + xf[1].powi(2) + xf[2].powi(2)).sqrt();

        let accel_d = |x: &[Dd; 3]| {
            let r2 = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
            let k = Dd::from_f64(-mu) / (r2 * r2.sqrt());
            [k * x[0], k * x[1], k * x[2]]
        };
        let mut xd = [Dd::from_f64(r0), Dd::from_f64(0.0), Dd::from_f64(0.0)];
        let mut vd = [Dd::from_f64(0.0), Dd::from_f64(vc), Dd::from_f64(0.0)];
        yoshida8_scalar(&accel_d, &mut xd, &mut vd, dt, steps);
        let dx = xd[0] - Dd::from_f64(r0);
        let err_d = (dx * dx + xd[1] * xd[1] + xd[2] * xd[2]).sqrt().value();

        std::eprintln!(
            "yoshida8 return error, {revs} revs: f64 = {err_f:.3e} m, dd = {err_d:.3e} m"
        );
        assert!(
            err_d * 10.0 < err_f,
            "dd ({err_d:.2e}) should clearly beat f64 ({err_f:.2e})"
        );
        assert!(
            err_d < 1e-5,
            "dd error {err_d:.2e} should stay well below f64 round-off"
        );
    }
}
