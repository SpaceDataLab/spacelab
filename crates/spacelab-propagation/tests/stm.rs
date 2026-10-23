//! The state transition matrix of a two-body orbit, checked by the invariants a Hamiltonian flow
//! must satisfy — not against a hand-computed reference, but against physics.
//!
//! Three checks, tightest first:
//! * **Symplectic:** `Φᵀ J Φ = J`. The two-body flow is Hamiltonian, so its STM is a symplectic
//!   matrix, to the integration tolerance. This is the strong one — a wrong Jacobian breaks it.
//! * **Liouville:** `det Φ = 1`, because `tr A = 0` for a Hamiltonian system.
//! * **Finite-difference cross-check:** column `j` of `Φ` is the sensitivity of the final state to a
//!   perturbation of initial component `j`; it must match a finite difference of the trajectory to
//!   the finite-difference floor (~1e-6). Coarse, but independent of the variational machinery.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::needless_range_loop)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};
use spacelab_testkit::nan_max;

/// Two-body in canonical units (`μ = 1`), so state and STM are all `O(1)` and one tolerance serves.
struct TwoBody;
impl ScalarSystem<6> for TwoBody {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 6]) -> [S; 6] {
        let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        let inv_r3 = r2.powf(-1.5);
        [
            y[3],
            y[4],
            y[5],
            -(y[0] * inv_r3),
            -(y[1] * inv_r3),
            -(y[2] * inv_r3),
        ]
    }
}

/// Plain 6-state view of the same force, for propagating perturbed trajectories in the FD check.
struct TwoBodyState;
impl DynamicalSystem<6> for TwoBodyState {
    fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        *dydt = TwoBody.eval::<f64>(t, y);
    }
}

/// 6×6 matrix multiply.
fn matmul(a: &[[f64; 6]; 6], b: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for k in 0..6 {
            let mut acc = 0.0;
            for j in 0..6 {
                acc += a[i][j] * b[j][k];
            }
            c[i][k] = acc;
        }
    }
    c
}

fn transpose(a: &[[f64; 6]; 6]) -> [[f64; 6]; 6] {
    let mut t = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            t[i][j] = a[j][i];
        }
    }
    t
}

/// The symplectic form `J = [[0, I], [-I, 0]]` (position/velocity split).
fn symplectic_j() -> [[f64; 6]; 6] {
    let mut j = [[0.0; 6]; 6];
    for i in 0..3 {
        j[i][i + 3] = 1.0;
        j[i + 3][i] = -1.0;
    }
    j
}

/// Determinant by Gaussian elimination with partial pivoting.
fn det6(m: &[[f64; 6]; 6]) -> f64 {
    let mut a = *m;
    let mut det = 1.0;
    for col in 0..6 {
        let mut piv = col;
        for r in (col + 1)..6 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if piv != col {
            a.swap(piv, col);
            det = -det;
        }
        let d = a[col][col];
        if d == 0.0 {
            return 0.0;
        }
        det *= d;
        for r in (col + 1)..6 {
            let factor = a[r][col] / d;
            for c in col..6 {
                a[r][c] -= factor * a[col][c];
            }
        }
    }
    det
}

fn propagate_stm(y0: &[f64; 6], t1: f64) -> ([f64; 6], [[f64; 6]; 6]) {
    let force = TwoBody;
    let var = Variational::<6, _>::new(&force);
    let aug0: [f64; 42] = augment::<6, 42>(y0);
    let traj = integrate(
        &Dop853,
        &var,
        0.0,
        &aug0,
        t1,
        0.01,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-9),
    )
    .expect("integrated augmented system");
    split::<6, 42>(&traj.final_state())
}

#[test]
fn the_two_body_stm_is_symplectic() {
    // r0 = (1,0,0), v0 = (0, 1.1, 0.1): a bound, mildly out-of-plane orbit.
    let y0 = [1.0, 0.0, 0.0, 0.0, 1.1, 0.1];
    let (_, phi) = propagate_stm(&y0, 3.0);

    let j = symplectic_j();
    let phit_j_phi = matmul(&matmul(&transpose(&phi), &j), &phi);
    let mut worst = 0.0_f64;
    for i in 0..6 {
        for k in 0..6 {
            worst = nan_max(worst, (phit_j_phi[i][k] - j[i][k]).abs());
        }
    }
    assert!(
        worst < 1e-8,
        "ΦᵀJΦ deviates from J by {worst:.3e} — the Jacobian or the STM ODE is wrong"
    );
}

#[test]
fn the_two_body_stm_has_unit_determinant() {
    // Liouville: a Hamiltonian flow preserves phase-space volume, so det Φ = 1.
    let y0 = [1.0, 0.0, 0.0, 0.0, 1.2, 0.0];
    let (_, phi) = propagate_stm(&y0, 4.0);
    let d = det6(&phi);
    assert!(
        (d - 1.0).abs() < 1e-8,
        "det Φ = {d}, not 1 (Liouville violated)"
    );
}

#[test]
fn the_stm_matches_a_finite_difference_of_the_trajectory() {
    // Column j of Φ is ∂y(t)/∂y₀[j]. Compare to a central finite difference of the propagated state.
    let y0 = [1.0, 0.0, 0.0, 0.0, 1.1, 0.05];
    let t1 = 2.0;
    let (_, phi) = propagate_stm(&y0, t1);

    let propagate_plain = |y: &[f64; 6]| -> [f64; 6] {
        integrate(
            &Dop853,
            &TwoBodyState,
            0.0,
            y,
            t1,
            0.01,
            Controls::new(Tolerance::new(1e-13, 1e-13), 1e-9),
        )
        .expect("plain")
        .final_state()
    };

    let eps = 1e-6;
    let mut worst = 0.0_f64;
    for j in 0..6 {
        let mut yp = y0;
        let mut ym = y0;
        yp[j] += eps;
        ym[j] -= eps;
        let fp = propagate_plain(&yp);
        let fm = propagate_plain(&ym);
        for i in 0..6 {
            let fd = (fp[i] - fm[i]) / (2.0 * eps);
            worst = nan_max(worst, (fd - phi[i][j]).abs());
        }
    }
    assert!(
        worst < 1e-5,
        "STM disagrees with the trajectory finite difference by {worst:.3e} (above the FD floor)"
    );
}

#[test]
fn the_stm_starts_at_the_identity() {
    let y0 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let aug0: [f64; 42] = augment::<6, 42>(&y0);
    let (_, phi0) = split::<6, 42>(&aug0);
    for i in 0..6 {
        for k in 0..6 {
            let want = f64::from(u8::from(i == k));
            assert!(
                (phi0[i][k] - want).abs() < 1e-15,
                "Φ(t₀) must be the identity"
            );
        }
    }
}
