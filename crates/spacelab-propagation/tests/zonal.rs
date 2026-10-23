//! General zonal harmonics `J₂…J_{N+1}`, checked by the physics — no reference trajectory.
//!
//! * **Degree one is `J2`**: the recurrence at `N = 1` must equal the hand-derived closed form, or
//!   one of them is wrong.
//! * **Energy and `Lz`** are conserved: the zonal field is conservative and axially symmetric, so a
//!   wrong `−∇U` drifts against an *independently written* potential.
//! * **The STM is symplectic**: the field is Hamiltonian, and the Jacobian comes from AD.
//! * **`J₃` breaks the equatorial reflection symmetry** that the even harmonics preserve — the crisp,
//!   reference-free signature of an odd zonal, and proof the recurrence carries the right parity.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]

use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{Acceleration, J2, Newtonian, Plain, PointMass, Zonal};
use spacelab_propagation::variational::{Variational, augment, split};
use spacelab_propagation::{Dop853, Tolerance};
use spacelab_testkit::nan_max;

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_683e-3;
const J3_COEFF: f64 = -2.532_656_49e-6;
const J4_COEFF: f64 = -1.619_621_59e-6;

fn leo(alt: f64, inc: f64) -> ([f64; 6], f64) {
    let a = RE + alt;
    let v = (MU / a).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    ([a, 0.0, 0.0, 0.0, v * inc.cos(), v * inc.sin()], period)
}

fn prop<A: Acceleration>(force: A, y0: &[f64; 6], t: f64) -> [f64; 6] {
    integrate(
        &Dop853,
        &Plain(Newtonian(force)),
        0.0,
        y0,
        t,
        30.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-4),
    )
    .expect("propagated")
    .final_state()
}

/// Total specific energy `½v² + U`, with the zonal potential written out independently of the
/// force model: `U = −μ/r + (μ/r) Σ J_n (Rₑ/r)ⁿ P_n(z/r)`.
fn energy(y: &[f64; 6]) -> f64 {
    let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
    let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    let u = y[2] / r;
    let rho = RE / r;
    let p2 = (3.0 * u * u - 1.0) / 2.0;
    let p3 = (5.0 * u * u * u - 3.0 * u) / 2.0;
    let p4 = (35.0 * u.powi(4) - 30.0 * u * u + 3.0) / 8.0;
    let disturbing =
        J2_COEFF * rho.powi(2) * p2 + J3_COEFF * rho.powi(3) * p3 + J4_COEFF * rho.powi(4) * p4;
    0.5 * v2 - MU / r + (MU / r) * disturbing
}

fn lz(y: &[f64; 6]) -> f64 {
    y[0] * y[4] - y[1] * y[3]
}

fn field() -> (PointMass, Zonal<3>) {
    (
        PointMass { mu: MU },
        Zonal {
            mu: MU,
            re: RE,
            j: [J2_COEFF, J3_COEFF, J4_COEFF],
        },
    )
}

#[test]
fn zonal_degree_one_reproduces_the_standalone_j2() {
    // The recurrence (Zonal<1>) and the hand-derived closed form (J2) must agree bit-for-bit-ish.
    let one_term = Zonal::<1> {
        mu: MU,
        re: RE,
        j: [J2_COEFF],
    };
    let closed = J2 {
        mu: MU,
        re: RE,
        j2: J2_COEFF,
    };
    for &(r, v) in &[
        ([7.0e6, 1.0e6, 2.0e6], [1.0e3, 7.0e3, -0.5e3]),
        ([-4.0e6, 5.0e6, -3.0e6], [6.0e3, 2.0e3, 3.0e3]),
        ([2.0e6, -6.0e6, 4.5e6], [-3.0e3, 1.0e3, 5.0e3]),
    ] {
        let a = one_term.acceleration::<f64>(0.0, &r, &v);
        let b = closed.acceleration::<f64>(0.0, &r, &v);
        for k in 0..3 {
            let scale = b[k].abs().max(1e-30);
            assert!(
                (a[k] - b[k]).abs() < 1e-14 * scale,
                "Zonal<1> vs J2 differ at {k}: {} vs {}",
                a[k],
                b[k]
            );
        }
    }
}

#[test]
fn zonal_gravity_conserves_energy_and_lz() {
    let (y0, period) = leo(650e3, 51.6_f64.to_radians());
    let e0 = energy(&y0);
    let lz0 = lz(&y0);
    let yf = prop(field(), &y0, 20.0 * period);
    assert!(
        (energy(&yf) - e0).abs() < 1e-9 * e0.abs(),
        "energy drifted {:.3e} relative over 20 orbits",
        (energy(&yf) - e0).abs() / e0.abs()
    );
    assert!(
        (lz(&yf) - lz0).abs() < 1e-8 * lz0.abs(),
        "Lz drifted {:.3e} relative — the zonal field is axially symmetric",
        (lz(&yf) - lz0).abs() / lz0.abs()
    );
}

#[test]
fn the_zonal_stm_is_symplectic() {
    let (y0, period) = leo(800e3, 45.0_f64.to_radians());
    let force = Newtonian(field());
    let var = Variational::<6, _>::new(&force);
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
        "zonal STM not symplectic: ΦᵀJΦ − J worst {worst:.3e}"
    );
}

#[test]
fn j3_breaks_the_equatorial_reflection_symmetry_that_even_zonals_preserve() {
    // Reflect through the equator: (x,y,z, vx,vy,vz) → (x,y,−z, vx,vy,−vz). The point mass and the
    // even zonals commute with this map, so propagating the reflected state equals reflecting the
    // propagated one. The odd J₃ does not commute — that is its defining signature.
    let reflect = |y: &[f64; 6]| [y[0], y[1], -y[2], y[3], y[4], -y[5]];
    let max_diff =
        |a: &[f64; 6], b: &[f64; 6]| (0..6).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max);

    let (y0, period) = leo(700e3, 55.0_f64.to_radians());
    let span = 10.0 * period;

    // Even only: J₃ = 0. Symmetry must hold to integrator precision.
    let even = || {
        (
            PointMass { mu: MU },
            Zonal::<3> {
                mu: MU,
                re: RE,
                j: [J2_COEFF, 0.0, J4_COEFF],
            },
        )
    };
    let even_residual = max_diff(
        &prop(even(), &reflect(&y0), span),
        &reflect(&prop(even(), &y0, span)),
    );

    // With J₃: the same construction must now differ, by far more than the even residual.
    let odd = || {
        (
            PointMass { mu: MU },
            Zonal::<3> {
                mu: MU,
                re: RE,
                j: [J2_COEFF, J3_COEFF, J4_COEFF],
            },
        )
    };
    let odd_violation = max_diff(
        &prop(odd(), &reflect(&y0), span),
        &reflect(&prop(odd(), &y0, span)),
    );

    println!(
        "reflection symmetry: even residual {even_residual:.3e} m, J₃ violation {odd_violation:.3e} m"
    );
    assert!(
        even_residual < 1e-2,
        "even zonals must preserve equatorial reflection: {even_residual:.3e} m"
    );
    assert!(
        odd_violation > 1e3 * even_residual.max(1e-6),
        "J₃ must visibly break the symmetry: violation {odd_violation:.3e} vs even residual {even_residual:.3e}"
    );
}
