//! The J₂ force model, checked by the perturbation physics it must reproduce — not a reference
//! trajectory, but the conservation laws and the secular node regression the geopotential dictates.
//!
//! * **Energy** `E = ½v² − μ/r + (μJ₂Rₑ²)/(2r³)(3z²/r² − 1)` is conserved: the J₂ field is
//!   conservative in the inertial frame, so a wrong acceleration-to-potential pairing drifts.
//! * **`Lz = x·v_y − y·v_x`** is conserved (J₂ is axially symmetric about the pole); the full
//!   angular momentum is *not* — J₂ torques it, which is what regresses the node.
//! * **Node regression** `Ω̇ = −³⁄₂ n J₂ (Rₑ/p)² cos i` — the defining secular effect, measured from
//!   the drift of the orbit normal and matched to the first-order formula.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{J2, Newtonian, Plain, PointMass};
use spacelab_propagation::variational::{Variational, augment, split};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_testkit::nan_max;

// WGS-84-ish constants (SI).
const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;

fn field() -> Newtonian<(PointMass, J2)> {
    Newtonian((
        PointMass { mu: MU },
        J2 {
            mu: MU,
            re: RE,
            j2: J2_COEFF,
        },
    ))
}

/// A circular LEO orbit at inclination `inc`, ascending node at `x`.
fn leo(alt: f64, inc: f64) -> ([f64; 6], f64, f64) {
    let a = RE + alt;
    let v = (MU / a).sqrt();
    let y = [a, 0.0, 0.0, 0.0, v * inc.cos(), v * inc.sin()];
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    (y, period, a)
}

fn energy(y: &[f64; 6]) -> f64 {
    let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
    let r = r2.sqrt();
    let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    let u_j2 = (MU * J2_COEFF * RE * RE) / (2.0 * r2 * r) * (3.0 * y[2] * y[2] / r2 - 1.0);
    0.5 * v2 - MU / r + u_j2
}

fn lz(y: &[f64; 6]) -> f64 {
    y[0] * y[4] - y[1] * y[3]
}

#[test]
fn j2_conserves_energy_and_polar_angular_momentum() {
    let (y0, period, _) = leo(700e3, 51.6_f64.to_radians());
    let e0 = energy(&y0);
    let lz0 = lz(&y0);
    let traj = integrate(
        &Dop853,
        &Plain(field()),
        0.0,
        &y0,
        20.0 * period,
        60.0,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-3),
    )
    .expect("propagated");
    let yf = traj.final_state();
    assert!(
        (energy(&yf) - e0).abs() < 1e-7 * e0.abs(),
        "energy drifted by {:.3e} relative over 20 orbits",
        (energy(&yf) - e0).abs() / e0.abs()
    );
    assert!(
        (lz(&yf) - lz0).abs() < 1e-8 * lz0.abs(),
        "Lz drifted by {:.3e} relative — J₂ must preserve the polar component",
        (lz(&yf) - lz0).abs() / lz0.abs()
    );
}

#[test]
fn the_full_angular_momentum_is_not_conserved_but_lz_is() {
    // Sanity that the test above is not vacuous: J₂ does torque the orbit (|L| direction moves),
    // so the x/y components of L must change even as Lz holds.
    let (y0, period, _) = leo(700e3, 63.0_f64.to_radians());
    let lx = |y: &[f64; 6]| y[1] * y[5] - y[2] * y[4];
    let traj = integrate(
        &Dop853,
        &Plain(field()),
        0.0,
        &y0,
        10.0 * period,
        60.0,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-3),
    )
    .expect("propagated");
    let yf = traj.final_state();
    assert!(
        (lx(&yf) - lx(&y0)).abs() > 1e-3 * lz(&y0).abs(),
        "the node should have moved"
    );
}

#[test]
fn the_node_regresses_at_the_first_order_j2_rate() {
    let inc = 51.6_f64.to_radians();
    let (y0, period, a) = leo(700e3, inc);

    // analytic first-order secular node rate: Ω̇ = −3/2 n J₂ (Re/p)² cos i, p = a for a circular orbit
    let n = std::f64::consts::TAU / period;
    let omega_dot = -1.5 * n * J2_COEFF * (RE / a).powi(2) * inc.cos();

    // orbit normal L = r × v; RAAN from its components: L ∝ (sin i sinΩ, −sin i cosΩ, cos i).
    let raan = |y: &[f64; 6]| {
        let lx = y[1] * y[5] - y[2] * y[4];
        let ly = y[2] * y[3] - y[0] * y[5];
        lx.atan2(-ly)
    };

    let orbits = 15.0;
    let traj = integrate(
        &Dop853,
        &Plain(field()),
        0.0,
        &y0,
        orbits * period,
        60.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-3),
    )
    .expect("propagated");

    // measure the secular drift over an integer number of orbits (short-period terms ~cancel)
    let om0 = raan(&y0);
    let omf = raan(&traj.final_state());
    let mut d = omf - om0;
    while d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    while d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    let measured_rate = d / (orbits * period);

    // first-order theory vs the true nonlinear drift agree to O(J₂) ~ 1e-3; allow 3 %.
    let rel = (measured_rate - omega_dot).abs() / omega_dot.abs();
    println!(
        "node rate: measured {:.6e} rad/s, first-order {:.6e} rad/s, {:.2}% ({:.4}°/day measured)",
        measured_rate,
        omega_dot,
        100.0 * rel,
        measured_rate * 86400.0 * 180.0 / std::f64::consts::PI
    );
    assert!(
        rel < 0.03,
        "measured node rate {measured_rate:.6e} vs theory {omega_dot:.6e} ({:.1}%)",
        100.0 * rel
    );
    // and the sign is right: a prograde orbit (i < 90°) regresses its node westward (Ω̇ < 0)
    assert!(
        measured_rate < 0.0,
        "a prograde orbit must regress, not advance, its node"
    );
}

#[test]
fn the_j2_stm_is_symplectic() {
    // J₂ is Hamiltonian, so its STM stays symplectic — the AD Jacobian of the J₂ acceleration is
    // exercised here exactly as it will be in orbit determination with a real force model.
    let (y0, period, _) = leo(800e3, 45.0_f64.to_radians());
    let f = field();
    let var = Variational::<6, _>::new(&f);
    let aug0: [f64; 42] = augment::<6, 42>(&y0);
    let traj = integrate(
        &Dop853,
        &var,
        0.0,
        &aug0,
        2.0 * period,
        30.0,
        Controls::new(Tolerance::new(1e-11, 1e-11), 1e-3),
    )
    .expect("propagated with STM");
    let (_, phi) = split::<6, 42>(&traj.final_state());

    // Φᵀ J Φ = J
    let mut jj = [[0.0; 6]; 6];
    for i in 0..3 {
        jj[i][i + 3] = 1.0;
        jj[i + 3][i] = -1.0;
    }
    let mut worst = 0.0_f64;
    for i in 0..6 {
        for k in 0..6 {
            let mut acc = 0.0;
            for m in 0..6 {
                for p in 0..6 {
                    acc += phi[m][i] * jj[m][p] * phi[p][k];
                }
            }
            worst = nan_max(worst, (acc - jj[i][k]).abs());
        }
    }
    assert!(
        worst < 1e-6,
        "J₂ STM not symplectic: ΦᵀJΦ − J worst {worst:.3e}"
    );
}
