//! WHFast: the exact Kepler drift reproduces an unperturbed orbit to rounding, the map is
//! symplectic (its Kepler part preserves the Wronskian), and under a J₂ perturbation the energy
//! error stays bounded while the trajectory tracks a DOP853 reference.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use spacelab_autodiff::Scalar;
use spacelab_coverage::find_events;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::IntegrationError;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Acceleration, J2, Newtonian, Plain, PointMass};
use spacelab_propagation::mercurius::integrate_mercurius;
use spacelab_propagation::saba::{integrate_saba2, integrate_saba4};
use spacelab_propagation::whfast::{integrate_whfast, integrate_whfast_dense};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_testkit::nan_max;

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2C: f64 = 1.082_626_68e-3;
const SEMI_MAJOR: f64 = 8_000e3;
const ECC: f64 = 0.1;

fn state0() -> ([f64; 3], [f64; 3], f64) {
    let r_p = SEMI_MAJOR * (1.0 - ECC);
    let v_p = (MU * (1.0 + ECC) / (SEMI_MAJOR * (1.0 - ECC))).sqrt();
    let period = std::f64::consts::TAU * (SEMI_MAJOR.powi(3) / MU).sqrt();
    ([r_p, 0.0, 0.0], [0.0, v_p, 0.0], period)
}

/// Exact Kepler position *and* velocity at `span` after periapsis, from the eccentric anomaly.
///
/// `Ė = n/(1 − e cos E)` follows from differentiating Kepler's equation, so the velocity is as
/// closed-form as the position and shares no arithmetic with the propagator under test.
fn kepler_state(span: f64) -> ([f64; 3], [f64; 3]) {
    let n = (MU / SEMI_MAJOR.powi(3)).sqrt();
    let m = n * span;
    let mut ea = m;
    for _ in 0..100 {
        ea -= (ea - ECC * ea.sin() - m) / (1.0 - ECC * ea.cos());
    }
    let b = SEMI_MAJOR * (1.0 - ECC * ECC).sqrt();
    let ea_dot = n / (1.0 - ECC * ea.cos());
    (
        [SEMI_MAJOR * (ea.cos() - ECC), b * ea.sin(), 0.0],
        [-SEMI_MAJOR * ea.sin() * ea_dot, b * ea.cos() * ea_dot, 0.0],
    )
}

fn kepler(span: f64) -> [f64; 3] {
    kepler_state(span).0
}

fn energy(mu: f64, r: &[f64; 3], v: &[f64; 3]) -> f64 {
    let rm = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - mu / rm
}

#[test]
fn whfast_kepler_drift_is_exact_with_no_perturbation() {
    // With J₂ = 0 the kick vanishes and only the exact Kepler drift acts: WHFast reproduces the
    // analytic orbit to rounding, no matter the step, over 50 orbits.
    let (r0, v0, period) = state0();
    let no_pert = J2 {
        mu: MU,
        re: RE,
        j2: 0.0,
    };
    let span = 50.0 * period;
    let (r, _) =
        integrate_whfast(MU, &no_pert, &r0, &v0, 0.0, span, period / 40.0).expect("whfast");
    let exact = kepler(span);
    let err =
        ((r[0] - exact[0]).powi(2) + (r[1] - exact[1]).powi(2) + (r[2] - exact[2]).powi(2)).sqrt();
    println!("WHFast exact-drift over 50 orbits: {err:.3e} m");
    // 5.459e-6 m measured. The old bound was 1e-2 -- 1833x the achieved value, which is to say it
    // could not fail: every mutation of the Stumpff series and of the universal-Kepler Newton step
    // degrades this by one or two orders and still passes at 1e-2.
    assert!(
        err < 2e-5,
        "Kepler drift is not exact: {err:.3e} m vs 5.459e-6 measured"
    );
}

#[test]
fn whfast_tracks_dop853_under_j2() {
    // Under a real J₂ perturbation WHFast (drift exact, J₂ as the kick) should follow a tight DOP853
    // reference over several orbits — the second-order splitting error is set by J₂'s smallness.
    let (r0, v0, period) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: J2C,
    };
    let span = 5.0 * period;
    let (r, _) = integrate_whfast(MU, &pert, &r0, &v0, 0.0, span, period / 2000.0).expect("whfast");

    let field = Plain(Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2C,
        },
    )));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let dop = integrate(
        &Dop853,
        &field,
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-4),
    )
    .expect("reference")
    .final_state();
    let err = ((r[0] - dop[0]).powi(2) + (r[1] - dop[1]).powi(2) + (r[2] - dop[2]).powi(2)).sqrt();
    println!("WHFast vs DOP853 under J2 over 5 orbits: {err:.3e} m");
    // 6.303e-2 m measured; the old bound was 50 m, 793x looser.
    assert!(
        err < 0.2,
        "WHFast diverged from DOP853 under J2: {err:.3e} m vs 6.303e-2 measured"
    );
}

/// A J₂ whose strength is modulated in time, so that *when* the kick is evaluated becomes visible.
///
/// Every other fixture in this file is autonomous — `J2::acceleration` ignores `t` — and every one
/// of them starts at `t0 = 0`. Between them those two choices hide the entire time axis of the
/// method: the mid-step argument `t + h/2`, the clock update `t += step`, and the schedule
/// `(t1 − t0)` are all unobservable, the last one twice over because `t1 − 0 == t1 + 0`. Sixteen
/// mutants across `integrate_whfast`, `whfast_step` and `integrate_whfast_dense` survived on that.
struct ModulatedJ2 {
    j2: J2,
    omega: f64,
}

impl Acceleration for ModulatedJ2 {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let scale = S::from_f64(1.0) + S::from_f64(0.5) * (S::from_f64(self.omega) * t).sin();
        let base = self.j2.acceleration(t, r, v);
        [base[0] * scale, base[1] * scale, base[2] * scale]
    }
}

fn modulated(period: f64) -> ModulatedJ2 {
    ModulatedJ2 {
        j2: J2 {
            mu: MU,
            re: RE,
            j2: J2C,
        },
        omega: std::f64::consts::TAU / period,
    }
}

#[test]
fn whfast_evaluates_the_perturbation_on_the_right_clock() {
    let (r0, v0_planar, period) = state0();
    // Inclined by 45°, because every other fixture in this file lies in the x–y plane. A planar
    // orbit makes `r[2]` identically zero, and with it the `r[2] * r[2]` term of `whfast_deriv`'s
    // `|r|²` and the whole z row `−μ r[2]/|r|³ + a_z` — seven mutants of arithmetic that no planar
    // test can reach, however tight its bound.
    let inc = 45.0_f64.to_radians();
    let v0 = [
        v0_planar[0],
        v0_planar[1] * inc.cos(),
        v0_planar[1] * inc.sin(),
    ];
    // Far from zero and not a multiple of the step, so `t1 − t0` and `t1 + t0` are different runs
    // and the mid-step epoch is not the node epoch.
    let t0 = 12_345.678;
    let span = 5.0 * period;
    let dt = period / 2000.0;
    let (r, _) =
        integrate_whfast(MU, &modulated(period), &r0, &v0, t0, t0 + span, dt).expect("whfast");

    let field = Plain(Newtonian((PointMass { mu: MU }, modulated(period))));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let reference = integrate(
        &Dop853,
        &field,
        t0,
        &y0,
        t0 + span,
        30.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-4),
    )
    .expect("reference");
    let dop = reference.final_state();
    let err = ((r[0] - dop[0]).powi(2) + (r[1] - dop[1]).powi(2) + (r[2] - dop[2]).powi(2)).sqrt();
    println!("WHFast vs DOP853 under a time-modulated J2 over 5 orbits: {err:.3e} m");
    // 3.989e-1 m measured. An order above the 3.078e-2 m the same run gave while planar, which is
    // the physics: at 45° inclination J₂'s out-of-plane component is at its largest, so the
    // second-order splitting error it drives grows with it.
    assert!(
        err < 1.2,
        "WHFast is on the wrong clock: {err:.3e} m vs 3.989e-1 measured"
    );

    // The dense form carries its own copy of the schedule, so it needs its own comparison. Plain and
    // dense walk identical nodes with identical drifts and kicks and must agree to rounding.
    let traj =
        integrate_whfast_dense(MU, &modulated(period), &r0, &v0, t0, t0 + span, dt).expect("dense");
    let f = traj.final_state();
    let d = ((f[0] - r[0]).powi(2) + (f[1] - r[1]).powi(2) + (f[2] - r[2]).powi(2)).sqrt();
    println!("WHFast dense-vs-plain on a shifted epoch: {d:.3e} m");
    assert!(
        d < 1e-8,
        "dense and plain disagree off zero epoch: {d:.3e} m"
    );

    // The perturbation term of the dense derivative, `+ ap[i]`, is what the J₂ = 0 fixture in
    // `whfast_dense_matches_plain_and_finds_events` cannot see: with `ap` identically zero, `+` and
    // `−` and `*` before it are the same expression. Here `ap` is a real modulated J₂ and the
    // reference is DOP853's own dense output, so the comparison covers all six rows of the
    // derivative at points that are not nodes.
    let mut worst_r = 0.0_f64;
    let mut worst_v = 0.0_f64;
    let nodes = 5 * 2000_u32; // span / dt, by construction above
    for k in 0..nodes {
        let t = t0 + (f64::from(k) + 0.5) * dt;
        let got = traj.eval(t);
        let want = reference.eval(t);
        worst_r = nan_max(
            worst_r,
            ((got[0] - want[0]).powi(2) + (got[1] - want[1]).powi(2) + (got[2] - want[2]).powi(2))
                .sqrt(),
        );
        worst_v = nan_max(
            worst_v,
            ((got[3] - want[3]).powi(2) + (got[4] - want[4]).powi(2) + (got[5] - want[5]).powi(2))
                .sqrt(),
        );
    }
    println!("WHFast dense vs DOP853 dense, worst mid-segment: {worst_r:.3e} m, {worst_v:.3e} m/s");
    // 4.005e-1 m and 3.944e-4 m/s measured -- the splitting error again, since at a step of a
    // two-thousandth of a period the cubic Hermite's own interpolation error is three orders below
    // it. Both bounds carry the same 3x margin as the rest of the file.
    assert!(
        worst_r < 1.2,
        "dense position off: {worst_r:.3e} m vs 4.005e-1 measured"
    );
    assert!(
        worst_v < 1.2e-3,
        "dense velocity off: {worst_v:.3e} m/s vs 3.944e-4 measured"
    );
}

/// `saba_step` keeps its own clock, and the fixtures that reached it are all time-independent.
///
/// The palindrome `drift(c₀h) kick(d₀h) … drift(cₙh)` advances a local `tl` across the step and
/// samples the perturbation at it, and `integrate_saba` sizes its step as `(t1 − t0)/n` and
/// advances `t += step`. `J2` reads position only, so none of that is visible through it: `tl`
/// could run backwards, the `c` index could be off by one, or the whole run could start at the
/// wrong epoch and every existing assertion would still hold.
///
/// Same fixture, epoch and reference as `whfast_evaluates_the_perturbation_on_the_right_clock` —
/// SABA is the same splitting with more stages, so what makes WHFast's clock observable makes
/// theirs observable too, and the higher order shows in the bound: 5.018e-4 m for SABA₂ and
/// 1.381e-4 m for SABA₄ over five orbits, against WHFast's 3.989e-1 m on the same run. Both bounds
/// carry the 3x margin the rest of the file uses.
#[test]
fn the_saba_splittings_evaluate_the_perturbation_on_the_right_clock() {
    let (r0, v0_planar, period) = state0();
    let inc = 45.0_f64.to_radians();
    let v0 = [
        v0_planar[0],
        v0_planar[1] * inc.cos(),
        v0_planar[1] * inc.sin(),
    ];
    let t0 = 12_345.678;
    let span = 5.0 * period;
    let dt = period / 2000.0;

    let field = Plain(Newtonian((PointMass { mu: MU }, modulated(period))));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let dop = integrate(
        &Dop853,
        &field,
        t0,
        &y0,
        t0 + span,
        30.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-4),
    )
    .expect("reference")
    .final_state();

    let miss = |r: &[f64; 3]| {
        ((r[0] - dop[0]).powi(2) + (r[1] - dop[1]).powi(2) + (r[2] - dop[2]).powi(2)).sqrt()
    };
    let (r2, _) =
        integrate_saba2(MU, &modulated(period), &r0, &v0, t0, t0 + span, dt).expect("saba2");
    let (r4, _) =
        integrate_saba4(MU, &modulated(period), &r0, &v0, t0, t0 + span, dt).expect("saba4");
    println!(
        "SABA vs DOP853 under a time-modulated J2 over 5 orbits: SABA2 {:.3e} m, SABA4 {:.3e} m",
        miss(&r2),
        miss(&r4)
    );
    assert!(
        miss(&r2) < 1.5e-3,
        "SABA2 is on the wrong clock: {:.3e} m",
        miss(&r2)
    );
    assert!(
        miss(&r4) < 4.5e-4,
        "SABA4 is on the wrong clock: {:.3e} m",
        miss(&r4)
    );
}

/// Every splitting entry point refuses an empty span, a non-positive step and a massless centre.
///
/// `if !(t1 > t0 && dt > 0.0 && mu > 0.0)` is written out four times over, and nothing asserted any
/// of its boundaries. `t1 == t0` is the one that matters: `step_count` rounds the ratio to zero and
/// clamps it to its floor of one, so an empty span that gets past the guard comes back as
/// `Ok((r0, v0))` — the state handed back unchanged, with the caller told the propagation ran.
#[test]
fn the_splittings_refuse_the_arguments_they_document() {
    let (r0, v0, period) = state0();
    let pert = modulated(period);
    let bad = |e: Option<IntegrationError>| e == Some(IntegrationError::BadArguments);
    let zonal = |_r: &[f64; 3]| [0.0; 3];
    let far = |_r: &[f64; 3]| 1.0e9;
    let dt = period / 100.0;

    for &(t0, t1, step, mu) in &[
        (0.0, 0.0, dt, MU),     // empty span
        (100.0, 0.0, dt, MU),   // backwards span
        (0.0, period, -dt, MU), // negative step
        (0.0, period, dt, 0.0), // massless centre
        (0.0, period, dt, -MU), // negative mass
    ] {
        let what = format!("t0={t0} t1={t1} dt={step} mu={mu}");
        assert!(
            bad(integrate_whfast(mu, &pert, &r0, &v0, t0, t1, step).err()),
            "whfast accepted {what}"
        );
        assert!(
            bad(integrate_whfast_dense(mu, &pert, &r0, &v0, t0, t1, step).err()),
            "whfast_dense accepted {what}"
        );
        assert!(
            bad(integrate_saba2(mu, &pert, &r0, &v0, t0, t1, step).err()),
            "saba2 accepted {what}"
        );
        assert!(
            bad(integrate_saba4(mu, &pert, &r0, &v0, t0, t1, step).err()),
            "saba4 accepted {what}"
        );
        assert!(
            bad(integrate_mercurius(mu, &zonal, &far, RE, 2.0 * RE, &r0, &v0, t0, t1, step).err()),
            "mercurius accepted {what}"
        );
    }

    // MERCURIUS is the one splitting that does not always reach the Kepler drift, which is what
    // refuses a massless centre everywhere else: inside the changeover band it hands the step to
    // IAS15, whose force is the perturbation plus `−mu·r/|r|³` — at `mu = 0` simply the
    // perturbation, integrable without complaint. So the entry guard is the only thing that refuses
    // it, and only a state that starts inside the band can show that.
    let near = |_r: &[f64; 3]| 0.0;
    assert!(
        bad(integrate_mercurius(0.0, &zonal, &near, RE, 2.0 * RE, &r0, &v0, 0.0, period, dt).err()),
        "mercurius integrated a massless centre inside the changeover band"
    );

    // Not vacuous: the same arguments made valid return a state.
    assert!(integrate_whfast(MU, &pert, &r0, &v0, 0.0, period, dt).is_ok());
    assert!(integrate_saba2(MU, &pert, &r0, &v0, 0.0, period, dt).is_ok());
    assert!(integrate_saba4(MU, &pert, &r0, &v0, 0.0, period, dt).is_ok());
    assert!(integrate_mercurius(MU, &zonal, &far, RE, 2.0 * RE, &r0, &v0, 0.0, period, dt).is_ok());
}

#[test]
fn whfast_energy_error_stays_bounded_under_j2() {
    // The symplectic payoff: under J₂ the total energy error oscillates within a fixed envelope
    // rather than drifting. Compare the peak relative error in the first half of a long run to the
    // second; a non-symplectic method of the same order would grow it roughly linearly.
    let (r0, v0, period) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: J2C,
    };
    let total_energy = |r: &[f64; 3], v: &[f64; 3]| {
        // Kepler energy plus the J₂ potential U₂ = (μ J₂ Rₑ²/r³)(3z²/r² − 1)/2 (so H is conserved).
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let rm = r2.sqrt();
        let u2 = MU * J2C * RE * RE / (rm * rm * rm) * (3.0 * r[2] * r[2] / r2 - 1.0) / 2.0;
        energy(MU, r, v) + u2
    };
    let e0 = total_energy(&r0, &v0);

    let orbits = 400;
    let (mut r, mut v) = (r0, v0);
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    for orbit in 0..orbits {
        let out = integrate_whfast(MU, &pert, &r, &v, 0.0, period, period / 500.0).expect("whfast");
        r = out.0;
        v = out.1;
        let rel = (total_energy(&r, &v) - e0).abs() / e0.abs();
        if orbit < orbits / 2 {
            first = nan_max(first, rel);
        } else {
            second = nan_max(second, rel);
        }
    }
    println!(
        "WHFast J2 energy error over {orbits} orbits: first half {first:.3e}, second half {second:.3e}"
    );
    // 3.361e-9 measured; the old bound was 1e-6, 298x looser.
    assert!(
        second < 1e-8,
        "energy error too large: {second:.3e} vs 3.361e-9 measured"
    );
    assert!(
        second < 3.0 * first.max(1e-12),
        "energy drifted: {second:.3e} vs {first:.3e}"
    );
}

#[test]
fn whfast_dense_matches_plain_and_finds_events() {
    let (r0, v0, period) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: 0.0,
    }; // pure Kepler, so dense tracks the exact orbit
    let span = 2.0 * period;
    let dt = period / 400.0;
    let (rf, _) = integrate_whfast(MU, &pert, &r0, &v0, 0.0, span, dt).expect("whfast");
    let traj = integrate_whfast_dense(MU, &pert, &r0, &v0, 0.0, span, dt).expect("dense");

    let f = traj.final_state();
    let dfinal = ((f[0] - rf[0]).powi(2) + (f[1] - rf[1]).powi(2) + (f[2] - rf[2]).powi(2)).sqrt();
    println!("WHFast dense-vs-plain final state: {dfinal:.3e} m");
    // The two walk the same node schedule with the same drift and kick, so they agree bit for bit;
    // anything above the round-off floor of a 7000 km radius means they have stopped doing so.
    assert!(
        dfinal < 1e-8,
        "dense final state differs from plain: {dfinal:.3e} m"
    );

    // Sample the *interior* of every segment, not one arbitrary time. The single sample this
    // replaced was `0.37 * span`, and `0.37 * 2 * 400` is 296 exactly: a node, where the cubic
    // Hermite returns the node value and the slopes it was built from cancel out of the answer
    // entirely. Replacing the whole body of `whfast_deriv` with `[0.0; 6]` passed that assertion —
    // every derivative in the dense output could be zero and nothing here would notice. Halfway
    // between nodes is where the two slope terms carry their largest weight, `±h/8`.
    //
    // Both halves of the state, and the velocity half is not decoration: the position block of a
    // cubic Hermite is built from `y[0..3]` and `f[0..3]`, and `whfast_deriv`'s first three entries
    // are a verbatim copy of `v`. Its *last* three — the whole `−μ r/|r|³ + a_pert` expression — are
    // slopes of the velocity block alone, so a position-only comparison cannot see any mutation of
    // them. Thirty of the thirty-three mutants in `whfast_deriv` survived on exactly that.
    let mut worst = 0.0_f64;
    let mut worst_t = 0.0;
    let mut worst_v = 0.0_f64;
    let nodes = 2 * 400_u32; // span / dt, by construction above
    for k in 0..nodes {
        let t = (f64::from(k) + 0.5) * dt;
        let got = traj.eval(t);
        let (exact, exact_v) = kepler_state(t);
        let d = ((got[0] - exact[0]).powi(2)
            + (got[1] - exact[1]).powi(2)
            + (got[2] - exact[2]).powi(2))
        .sqrt();
        if d > worst {
            worst = d;
            worst_t = t;
        }
        worst_v = nan_max(
            worst_v,
            ((got[3] - exact_v[0]).powi(2)
                + (got[4] - exact_v[1]).powi(2)
                + (got[5] - exact_v[2]).powi(2))
            .sqrt(),
        );
    }
    println!(
        "WHFast Hermite-dense mid-segment error: worst {worst:.3e} m at t = {worst_t:.1} s, {worst_v:.3e} m/s"
    );
    // 2.792e-3 m measured over all 800 mid-segment samples -- the true interpolation error of a
    // cubic Hermite at this step, four orders above the 2.075e-7 m the node sample reported, and
    // the only number in this file that sees `whfast_deriv` at all.
    assert!(
        worst < 1e-2,
        "Hermite interior interpolation off: {worst:.3e} m vs 2.792e-3 measured"
    );
    // 4.423e-6 m/s measured, the same 3.6x margin the position bound above carries. The velocity
    // block is where `−μ r/|r|³ + a_pert` lands, so this is the only assertion in the crate that
    // reads the acceleration half of `whfast_deriv` at all.
    assert!(
        worst_v < 1.6e-5,
        "Hermite interior velocity off: {worst_v:.3e} m/s vs 4.423e-6 measured"
    );

    let n = find_events(|t| traj.eval(t)[0], 0.0, span, period / 20.0, 1e-3, |_| {});
    assert_eq!(n, 4, "expected 4 x-axis crossings, got {n}");
}

/// A convenience: J₂ used directly as an [`Acceleration`] confirms the trait object works with WHFast.
#[test]
fn whfast_rejects_bad_arguments() {
    let (r0, v0, _) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: 0.0,
    };
    assert!(integrate_whfast(MU, &pert, &r0, &v0, 0.0, -1.0, 10.0).is_err());
    assert!(integrate_whfast(MU, &pert, &r0, &v0, 0.0, 100.0, 0.0).is_err());
    // The dense form repeats the guard rather than delegating, so it has to be checked separately.
    // A `dt >= 0.0` there is not a slower path but a panic: `(t1 − t0)/0` is `inf`, `inf as usize`
    // saturates to `usize::MAX`, and `Vec::with_capacity` overflows on the byte count.
    assert!(integrate_whfast_dense(MU, &pert, &r0, &v0, 0.0, -1.0, 10.0).is_err());
    assert!(integrate_whfast_dense(MU, &pert, &r0, &v0, 0.0, 100.0, 0.0).is_err());

    // `t1 == t0` is a caller error, not a zero-length run, and both entry points must say so. With
    // the guard weakened to `>=` the schedule below reads `n_steps = max(0.0.round(), 1.0) = 1` and
    // `step = 0`: one no-op drift–kick–drift that hands the input state back as though it had been
    // propagated. That is the worst of the three outcomes -- not an error, not a propagation, but a
    // plausible-looking state at the wrong epoch.
    assert_eq!(
        integrate_whfast(MU, &pert, &r0, &v0, 500.0, 500.0, 10.0),
        Err(IntegrationError::BadArguments)
    );
    assert_eq!(
        integrate_whfast_dense(MU, &pert, &r0, &v0, 500.0, 500.0, 10.0).err(),
        Some(IntegrationError::BadArguments)
    );

    // A non-positive `mu` is refused, and it stays refused if this entry guard is weakened: the
    // drift has its own, and `√μ = 0` at `mu = 0` and `NaN` at `mu < 0` fail it before any state is
    // written. Pinned here because it is the whole content of the `mu > 0.0 -> >=` equivalence
    // recorded in `.cargo/mutants.toml`.
    for mu in [0.0, -MU] {
        assert_eq!(
            integrate_whfast(mu, &pert, &r0, &v0, 0.0, 100.0, 10.0),
            Err(IntegrationError::BadArguments),
            "mu = {mu:e}"
        );
        assert_eq!(
            integrate_whfast_dense(mu, &pert, &r0, &v0, 0.0, 100.0, 10.0).err(),
            Some(IntegrationError::BadArguments),
            "mu = {mu:e}"
        );
    }
}

/// A satellite *at* the centre of attraction has no orbit, and the drift says so instead of spinning.
///
/// `α = 2/r₀ − v²/μ` is `+∞` at `r₀ = 0`, so the universal anomaly and `z = αX²` are infinite too,
/// and the Stumpff argument reduction `while |z| > 0.1 { z *= 0.25 }` never brings an infinite `z`
/// down. Before the [`StumpffArg`](https://docs.rs/) invariant this was an unbounded loop in release
/// and a `u32` overflow panic in debug some twenty seconds in — reached from an ordinary caller
/// argument, since the entry check validated `mu`, the span and the step but never the state.
///
/// All three splittings that share the drift are checked: they must agree, because the defect is in
/// the drift, not in any one composition.
#[test]
fn a_state_with_no_orbit_is_reported_rather_than_drifted() {
    use spacelab_propagation::mercurius::integrate_mercurius;
    use spacelab_propagation::saba::integrate_saba2;

    let no_pert = J2 {
        mu: MU,
        re: RE,
        j2: 0.0,
    };
    let (_, v0, _) = state0();
    let origin = [0.0; 3];
    assert_eq!(
        integrate_whfast(MU, &no_pert, &origin, &v0, 0.0, 100.0, 10.0),
        Err(IntegrationError::BadArguments)
    );
    assert_eq!(
        integrate_saba2(MU, &no_pert, &origin, &v0, 0.0, 100.0, 10.0),
        Err(IntegrationError::BadArguments)
    );
    assert_eq!(
        integrate_whfast_dense(MU, &no_pert, &origin, &v0, 0.0, 100.0, 10.0).err(),
        Some(IntegrationError::BadArguments)
    );
    let zero_pert = |_: &[f64; 3]| [0.0; 3];
    let far_encounter = |_: &[f64; 3]| 1.0e9;
    assert_eq!(
        integrate_mercurius(
            MU,
            &zero_pert,
            &far_encounter,
            RE,
            2.0 * RE,
            &origin,
            &v0,
            0.0,
            100.0,
            10.0,
        )
        .err(),
        Some(IntegrationError::BadArguments)
    );

    // Not vacuous: the same call from a real orbit returns a state.
    let (r0, _, _) = state0();
    assert!(integrate_whfast(MU, &no_pert, &r0, &v0, 0.0, 100.0, 10.0).is_ok());
}

/// A step too small to walk the span is refused by the splitting integrators too.
///
/// The companion of `a_step_that_cannot_walk_the_span_is_refused_before_the_loop` in
/// `tests/integrators.rs`, for the three entry points that take an [`Acceleration`] rather than a
/// `DynamicalSystem`. These are the ones that pointed at the defect: `integrate_whfast`'s
/// `dt > 0.0 -> >=` and `t1 - t0 -> t1 / t0` mutants were caught by the mutation campaign only by
/// *not terminating*, both of them through `((t1 - t0) / dt).round() as usize` saturating to
/// `usize::MAX` and handing the loop 1.8e19 no-op steps.
#[test]
fn a_step_too_small_to_walk_the_span_is_refused_by_the_splittings() {
    let (r0, v0, _) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: J2C,
    };
    let dt = 1e-300;
    let is_limit =
        |e: Option<IntegrationError>| matches!(e, Some(IntegrationError::StepLimit { .. }));

    assert!(is_limit(
        integrate_whfast(MU, &pert, &r0, &v0, 0.0, 1.0, dt).err()
    ));
    assert!(is_limit(
        integrate_whfast_dense(MU, &pert, &r0, &v0, 0.0, 1.0, dt).err()
    ));
    assert!(is_limit(
        integrate_saba2(MU, &pert, &r0, &v0, 0.0, 1.0, dt).err()
    ));
    assert!(is_limit(
        integrate_saba4(MU, &pert, &r0, &v0, 0.0, 1.0, dt).err()
    ));
    let zonal = |_r: &[f64; 3]| [0.0; 3];
    let radius = |r: &[f64; 3]| r.iter().map(|c| c * c).sum::<f64>().sqrt();
    assert!(is_limit(
        integrate_mercurius(MU, &zonal, &radius, 1e6, 3e6, &r0, &v0, 0.0, 1.0, dt).err()
    ));
}
