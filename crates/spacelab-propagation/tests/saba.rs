//! SABA₂/SABA₄: higher-order symplectic splitting on a J₂-perturbed orbit — more accurate per force
//! evaluation than WHFast at the same step, with a bounded (non-drifting) energy error.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::saba::{integrate_saba2, integrate_saba4};
use spacelab_propagation::whfast::integrate_whfast;
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

fn dop_reference(span: f64) -> [f64; 3] {
    let (r0, v0, _) = state0();
    let field = Plain(Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2C,
        },
    )));
    let y0 = [r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]];
    let yf = integrate(
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
    [yf[0], yf[1], yf[2]]
}

fn err(rf: &[f64; 3], span: f64) -> f64 {
    let d = dop_reference(span);
    ((rf[0] - d[0]).powi(2) + (rf[1] - d[1]).powi(2) + (rf[2] - d[2]).powi(2)).sqrt()
}

fn total_energy(r: &[f64; 3], v: &[f64; 3]) -> f64 {
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let rm = r2.sqrt();
    let kepler = 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) - MU / rm;
    let u2 = MU * J2C * RE * RE / (rm * rm * rm) * (3.0 * r[2] * r[2] / r2 - 1.0) / 2.0;
    kepler + u2
}

#[test]
fn saba_beats_whfast_at_the_same_step() {
    // At one fixed step, higher-order composition gives lower error: SABA₄ < SABA₂ < WHFast, all on
    // the identical Kepler-drift + J₂-kick splitting.
    let (r0, v0, period) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: J2C,
    };
    let span = 3.0 * period;
    let dt = period / 200.0;
    let e_wh = err(
        &integrate_whfast(MU, &pert, &r0, &v0, 0.0, span, dt)
            .expect("wh")
            .0,
        span,
    );
    let e_s2 = err(
        &integrate_saba2(MU, &pert, &r0, &v0, 0.0, span, dt)
            .expect("s2")
            .0,
        span,
    );
    let e_s4 = err(
        &integrate_saba4(MU, &pert, &r0, &v0, 0.0, span, dt)
            .expect("s4")
            .0,
        span,
    );
    println!(
        "at dt=period/200 over 3 orbits: WHFast {e_wh:.3e} m, SABA2 {e_s2:.3e} m, SABA4 {e_s4:.3e} m"
    );
    assert!(
        e_s2 < e_wh,
        "SABA2 not better than WHFast: {e_s2:.3e} vs {e_wh:.3e}"
    );
    assert!(
        e_s4 < e_s2,
        "SABA4 not better than SABA2: {e_s4:.3e} vs {e_s2:.3e}"
    );
    assert!(e_s4 < 1e-2, "SABA4 unexpectedly inaccurate: {e_s4:.3e} m");
}

#[test]
fn saba4_energy_error_stays_bounded() {
    // Symplectic: the total (Kepler + J₂) energy error oscillates within a fixed envelope over many
    // orbits rather than drifting. First-half peak vs second-half peak.
    let (r0, v0, period) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: J2C,
    };
    let e0 = total_energy(&r0, &v0);
    let orbits = 400;
    let (mut r, mut v) = (r0, v0);
    let (mut first, mut second) = (0.0_f64, 0.0_f64);
    for orbit in 0..orbits {
        let out = integrate_saba4(MU, &pert, &r, &v, 0.0, period, period / 200.0).expect("saba4");
        r = out.0;
        v = out.1;
        let rel = (total_energy(&r, &v) - e0).abs() / e0.abs();
        if orbit < orbits / 2 {
            first = nan_max(first, rel);
        } else {
            second = nan_max(second, rel);
        }
    }
    println!("SABA4 J2 energy error over {orbits} orbits: first {first:.3e}, second {second:.3e}");
    assert!(second < 1e-9, "energy error too large: {second:.3e}");
    assert!(
        second < 3.0 * first.max(1e-14),
        "energy drifted: {second:.3e} vs {first:.3e}"
    );
}

#[test]
fn saba_rejects_bad_arguments() {
    let (r0, v0, _) = state0();
    let pert = J2 {
        mu: MU,
        re: RE,
        j2: 0.0,
    };
    assert!(integrate_saba2(MU, &pert, &r0, &v0, 0.0, -1.0, 10.0).is_err());
    assert!(integrate_saba4(MU, &pert, &r0, &v0, 0.0, 100.0, 0.0).is_err());
}
