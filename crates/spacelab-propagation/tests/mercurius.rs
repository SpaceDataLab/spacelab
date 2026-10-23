//! MERCURIUS: a satellite passes close to a fixed perturbing mass. MERCURIUS switches to IAS15 for
//! the encounter and matches a tight reference, where a fixed-step WHFast at the same step misses the
//! sharp kick; away from the encounter it runs the symplectic WHFast.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::doc_markdown
)]

use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Acceleration, J2};
use spacelab_propagation::mercurius::{changeover, integrate_mercurius};
use spacelab_propagation::whfast::integrate_whfast;
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};
use spacelab_testkit::nan_max;

const MU: f64 = 3.986_004_418e14;
const MU_P: f64 = MU * 1.0e-3; // a strong (but not catastrophic) close flyby
const RP: [f64; 3] = [0.0, 8_000e3, 2.0e5]; // 200 km out of the orbit plane at the quarter point

fn pert(r: &[f64; 3]) -> [f64; 3] {
    let d = [r[0] - RP[0], r[1] - RP[1], r[2] - RP[2]];
    let inv_d3 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).powf(-1.5);
    [
        -MU_P * d[0] * inv_d3,
        -MU_P * d[1] * inv_d3,
        -MU_P * d[2] * inv_d3,
    ]
}
fn encounter(r: &[f64; 3]) -> f64 {
    ((r[0] - RP[0]).powi(2) + (r[1] - RP[1]).powi(2) + (r[2] - RP[2]).powi(2)).sqrt()
}

/// Full force (central + perturber) as a plain first-order system, for the DOP853 reference.
struct Full;
impl DynamicalSystem<6> for Full {
    fn derivative(&self, _t: f64, y: &[f64; 6], dy: &mut [f64; 6]) {
        let r = [y[0], y[1], y[2]];
        let inv_r3 = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).powf(-1.5);
        let p = pert(&r);
        dy[0] = y[3];
        dy[1] = y[4];
        dy[2] = y[5];
        dy[3] = -MU * r[0] * inv_r3 + p[0];
        dy[4] = -MU * r[1] * inv_r3 + p[1];
        dy[5] = -MU * r[2] * inv_r3 + p[2];
    }
}

fn state0() -> ([f64; 3], [f64; 3], f64) {
    let r = 8_000e3;
    let v = (MU / r).sqrt();
    let period = std::f64::consts::TAU * (r * r * r / MU).sqrt();
    ([r, 0.0, 0.0], [0.0, v, 0.0], period)
}

#[test]
fn changeover_is_a_smooth_zero_to_one_ramp() {
    assert_eq!(changeover(2.0, 0.5, 1.0), 1.0); // far
    assert_eq!(changeover(0.2, 0.5, 1.0), 0.0); // near
    let mid = changeover(0.75, 0.5, 1.0);
    assert!(
        (mid - 0.5).abs() < 1e-12,
        "midpoint should be 1/2 by symmetry, got {mid}"
    );
    assert!(
        changeover(0.6, 0.5, 1.0) < changeover(0.9, 0.5, 1.0),
        "changeover must be monotone"
    );
}

#[test]
fn mercurius_resolves_a_close_encounter_that_whfast_misses() {
    let (r0, v0, period) = state0();
    let span = 0.6 * period; // covers the quarter-point encounter
    let dt = period / 300.0;
    let (r_lo, r_hi) = (7.0e5, 2.0e6); // switch to IAS15 well before the flyby

    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let dop = integrate(
        &Dop853,
        &Full,
        0.0,
        &y0,
        span,
        30.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-4),
    )
    .expect("reference")
    .final_state();
    let refr = [dop[0], dop[1], dop[2]];

    let (rm, _, near) =
        integrate_mercurius(MU, &pert, &encounter, r_lo, r_hi, &r0, &v0, 0.0, span, dt)
            .expect("merc");
    let merc_err =
        ((rm[0] - refr[0]).powi(2) + (rm[1] - refr[1]).powi(2) + (rm[2] - refr[2]).powi(2)).sqrt();

    // Plain WHFast at the same fixed step kicks with the full perturbation and under-resolves the encounter.
    let no_j2 = J2 {
        mu: MU,
        re: 6_378_137.0,
        j2: 0.0,
    };
    let _ = no_j2.acceleration::<f64>(0.0, &r0, &v0); // (trait in scope)
    let (rw, _) = integrate_whfast_pert(MU, &r0, &v0, span, dt);
    let wh_err =
        ((rw[0] - refr[0]).powi(2) + (rw[1] - refr[1]).powi(2) + (rw[2] - refr[2]).powi(2)).sqrt();

    println!(
        "MERCURIUS err {merc_err:.3e} m ({near} near/IAS15 steps), fixed-WHFast err {wh_err:.3e} m"
    );
    // A band, not a floor. The point of the changeover is that the *far* field costs nothing:
    // 24 of the 180 steps are adaptive here, the ones that cross the flyby. `near > 0` alone is
    // satisfied by a test that inverts the changeover and hands every step to IAS15 — an answer
    // that is still accurate, still passes both error bounds below, and has thrown the whole
    // splitting away. The upper bound is what says the hybrid is still a hybrid.
    assert!(
        (12..=48).contains(&near),
        "{near} of the 180 steps went to IAS15; 24 measured, and the far field should cost none"
    );
    // These bounds are the regression gate on the splitting itself, not a generous sanity check:
    // splitting the interaction by the changeover reaches ~8.5 m here, splitting the *step* --
    // WHFast far, IAS15 near -- reaches ~46 m, because the handover freezes the symplectic map's
    // periapsis-peaked energy oscillation. See the module documentation for the band sweep.
    assert!(
        merc_err < 20.0,
        "MERCURIUS did not resolve the encounter: {merc_err:.3e} m"
    );
    assert!(
        merc_err < wh_err / 500.0,
        "MERCURIUS ({merc_err:.3e}) not clearly better than fixed WHFast ({wh_err:.3e})"
    );
}

// --- the weakly perturbed regime, where a whole-step switch used to destroy the result -----------

const RE: f64 = 6_378_137.0;
const J2C: f64 = 1.082_626_68e-3;

/// A Molniya orbit at periapsis: eccentric enough that the changeover band is crossed twice per
/// revolution, and perturbed only by `J2`, so the splitting of [`integrate_mercurius`] never fails
/// and the hybrid has nothing to rescue. That is the regime the regression is about.
fn molniya() -> ([f64; 3], [f64; 3], f64) {
    let (a, e, inc) = (26_554e3, 0.74, 63.4_f64.to_radians());
    let rp = a * (1.0 - e);
    let vp = (MU * (1.0 + e) / rp).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    (
        [rp, 0.0, 0.0],
        [0.0, vp * inc.cos(), vp * inc.sin()],
        period,
    )
}

/// The `J2` perturbing potential per unit mass, excluding the central term.
fn zonal_potential(r: &[f64; 3]) -> f64 {
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let inv_r = r2.sqrt().recip();
    let inv_r5 = inv_r * inv_r * inv_r * inv_r * inv_r;
    0.5 * MU * J2C * RE * RE * (3.0 * r[2] * r[2] - r2) * inv_r5
}

/// Its negative gradient, written out. `zonal_agrees_with_its_potential` checks the pair.
fn zonal(r: &[f64; 3]) -> [f64; 3] {
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let inv_r = r2.sqrt().recip();
    let inv_r5 = inv_r * inv_r * inv_r * inv_r * inv_r;
    let k = -1.5 * J2C * MU * RE * RE * inv_r5;
    let z_term = 5.0 * r[2] * r[2] / r2;
    [
        k * r[0] * (1.0 - z_term),
        k * r[1] * (1.0 - z_term),
        k * r[2] * (3.0 - z_term),
    ]
}

fn total_energy(r: &[f64; 3], v: &[f64; 3]) -> f64 {
    let speed2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let radius = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    0.5 * speed2 - MU / radius + zonal_potential(r)
}

#[test]
fn zonal_agrees_with_its_potential() {
    // The energy below is only an invariant if the potential it is built from is the one the
    // integrator integrates, so the pair is checked rather than assumed.
    let (r, _, _) = molniya();
    let a = zonal(&r);
    for i in 0..3 {
        let h = 1.0;
        let (mut lo, mut hi) = (r, r);
        lo[i] -= h;
        hi[i] += h;
        let numeric = -(zonal_potential(&hi) - zonal_potential(&lo)) / (2.0 * h);
        assert!(
            (numeric - a[i]).abs() <= 1e-6 * a[i].abs().max(1e-12),
            "component {i}: -dU/dx = {numeric:e} but the force is {:e}",
            a[i]
        );
    }
}

#[test]
fn mercurius_does_not_wreck_a_weakly_perturbed_orbit() {
    let (r0, v0, period) = molniya();
    let rp = r0[0];
    let (r_lo, r_hi) = (1.05 * rp, 2.0 * rp); // crossed twice per revolution, at periapsis
    let dt = period / 256.0;
    let revolutions = 20;

    let radius = |r: &[f64; 3]| (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let (mut r, mut v) = (r0, v0);
    let e0 = total_energy(&r, &v);
    let (mut worst, mut near_total) = (0.0_f64, 0u32);
    for k in 0..revolutions {
        let (t0, t1) = (f64::from(k) * period, f64::from(k + 1) * period);
        let (rn, vn, near) =
            integrate_mercurius(MU, &zonal, &radius, r_lo, r_hi, &r, &v, t0, t1, dt)
                .expect("mercurius");
        r = rn;
        v = vn;
        near_total += near;
        worst = nan_max(worst, ((total_energy(&r, &v) - e0) / e0).abs());
    }

    println!("MERCURIUS on J2 only: worst |dE/E| = {worst:.3e} over {near_total} adaptive steps");
    // Bounded above for the same reason as in the encounter test: 526 of the 5120 steps cross the
    // band, and a changeover read the wrong way round would send all 5120 through IAS15.
    assert!(
        (260..=1050).contains(&near_total),
        "{near_total} of the 5120 steps went to IAS15; 526 measured, and only the band should"
    );
    // A bound, not a comparison: on a problem with no encounter the hybrid has nothing to rescue
    // and every step crossing the band costs something, so this asserts only that the cost stays
    // small. The regime boundary itself is the encounter test above, which is where the two
    // constructions actually separate.
    assert!(
        worst < 1e-5,
        "the changeover band is costing far more than it should: {worst:.3e}"
    );
}

/// Fixed-step WHFast with the same perturbation (no encounter handling), via a local closure adapter.
fn integrate_whfast_pert(
    mu: f64,
    r0: &[f64; 3],
    v0: &[f64; 3],
    span: f64,
    dt: f64,
) -> ([f64; 3], [f64; 3]) {
    // Reuse the crate's WHFast through a small Acceleration adapter.
    struct P;
    impl Acceleration for P {
        fn acceleration<S: spacelab_autodiff::Scalar>(
            &self,
            _t: S,
            r: &[S; 3],
            _v: &[S; 3],
        ) -> [S; 3] {
            let rp = [S::from_f64(RP[0]), S::from_f64(RP[1]), S::from_f64(RP[2])];
            let d = [r[0] - rp[0], r[1] - rp[1], r[2] - rp[2]];
            let inv_d3 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).powf(-1.5);
            [
                d[0] * inv_d3 * S::from_f64(-MU_P),
                d[1] * inv_d3 * S::from_f64(-MU_P),
                d[2] * inv_d3 * S::from_f64(-MU_P),
            ]
        }
    }
    let (r, v) = integrate_whfast(mu, &P, r0, v0, 0.0, span, dt).expect("whfast");
    (r, v)
}
