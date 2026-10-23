//! Minimum-energy low-thrust optimal control by Pontryagin's Maximum Principle (indirect method).
//!
//! For the cost `J = ½∫|u|² dt` under `r̈ = −μr/|r|³ + u`, the PMP gives the optimal control as the
//! (negated) velocity costate, `u* = −λ_v` — the primer vector of Lawden's theory. The state and
//! costate obey the augmented Hamiltonian dynamics
//!
//! ```text
//! ṙ = v                        λ̇_r = −A(r)ᵀ λ_v
//! v̇ = −μr/|r|³ − λ_v            λ̇_v = −λ_r
//! ```
//!
//! with `A(r) = ∂(−μr/|r|³)/∂r` the gravity gradient. The fixed-endpoint two-point boundary-value
//! problem — reach a target state at a fixed time — is solved by single-shooting: guess the initial
//! costate, integrate, and Newton-correct on the terminal-state error with a finite-difference
//! Jacobian. Direct (collocation / pseudospectral) transcription is an alternative that needs an NLP
//! solver; the indirect method here is self-contained and satisfies the optimality conditions exactly.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// Augmented state `[r, v, λ_r, λ_v, J]` (13): the state, the costate, and the cost accrued so far.
type Aug = [f64; 13];

/// The PMP augmented dynamics for the minimum-energy problem (`u = −λ_v`), with the running cost
/// `J̇ = ½|u|²` carried as a state so that the same integrator that resolves the trajectory resolves
/// its cost. These are Hamilton's equations for `H = λ_r·v − μ(λ_v·r)/|r|³ − ½|λ_v|²`.
fn deriv(y: Aug, mu: f64) -> Aug {
    let r = [y[0], y[1], y[2]];
    let lv = [y[9], y[10], y[11]];
    let rn = libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    let rn3 = rn * rn * rn;
    let rn5 = rn3 * rn * rn;
    let rlv = r[0] * lv[0] + r[1] * lv[1] + r[2] * lv[2];
    let mut d = [0.0; 13];
    d[12] = 0.5 * (lv[0] * lv[0] + lv[1] * lv[1] + lv[2] * lv[2]);
    // ṙ = v
    d[0] = y[3];
    d[1] = y[4];
    d[2] = y[5];
    for i in 0..3 {
        // v̇ = −μr/|r|³ − λ_v
        d[3 + i] = -mu * r[i] / rn3 - lv[i];
        // λ̇_r = −Aλ_v,  (Aλ_v)_i = −μ/|r|³ λ_v,i + 3μ/|r|⁵ (r·λ_v) r_i
        d[6 + i] = mu / rn3 * lv[i] - 3.0 * mu / rn5 * rlv * r[i];
        // λ̇_v = −λ_r
        d[9 + i] = -y[6 + i];
    }
    d
}

fn rk4(mut y: Aug, mu: f64, t: f64, steps: usize) -> Aug {
    let h = t / steps as f64;
    let add = |a: Aug, b: Aug, s: f64| core::array::from_fn(|i| a[i] + b[i] * s);
    for _ in 0..steps {
        let k1 = deriv(y, mu);
        let k2 = deriv(add(y, k1, h / 2.0), mu);
        let k3 = deriv(add(y, k2, h / 2.0), mu);
        let k4 = deriv(add(y, k3, h), mu);
        y = core::array::from_fn(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    }
    y
}

/// Solve the 6×6 linear system `J x = b` by Gaussian elimination with partial pivoting.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let mut piv = col;
        for row in (col + 1)..6 {
            if libm::fabs(a[row][col]) > libm::fabs(a[piv][col]) {
                piv = row;
            }
        }
        if libm::fabs(a[piv][col]) < 1e-18 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in (col + 1)..6 {
            let f = a[row][col] / a[col][col];
            for c in col..6 {
                a[row][c] -= f * a[col][c];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        let mut s = b[i];
        for c in (i + 1)..6 {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// The result of a minimum-energy transfer: the optimal initial costate, whether the shooter
/// converged, the achieved terminal-state error, and the energy cost `½∫|u|² dt`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinEnergyTransfer {
    /// Optimal initial costate `[λ_r(0), λ_v(0)]`; the optimal control is `u(t) = −λ_v(t)`.
    pub initial_costate: [f64; 6],
    /// Terminal-state error norm `‖[r(T)−r_f, v(T)−v_f]‖` at convergence.
    pub terminal_error: f64,
    /// The energy cost `½∫₀ᵀ|u|² dt` of the converged trajectory.
    pub energy_cost: f64,
    /// Whether the shooter reached the target within tolerance.
    pub converged: bool,
}

/// Solve the fixed-time, fixed-endpoint minimum-energy low-thrust transfer from `(r0, v0)` to
/// `(rf, vf)` in time `tf` under gravitational parameter `mu`. Returns the optimal costate and cost.
///
/// Positions/velocities are the caller's consistent units; the optimal control at time `t` is
/// `u(t) = −λ_v(t)`, obtained by integrating [`optimal_control_at`] from `initial_costate`.
#[must_use]
pub fn min_energy_transfer(
    r0: [f64; 3],
    v0: [f64; 3],
    rf: [f64; 3],
    vf: [f64; 3],
    mu: f64,
    tf: f64,
) -> MinEnergyTransfer {
    let steps = 800;
    let augment = |p: [f64; 6]| -> Aug {
        [
            r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], p[0], p[1], p[2], p[3], p[4], p[5], 0.0,
        ]
    };
    let shoot = |p: [f64; 6]| -> [f64; 6] {
        let y = rk4(augment(p), mu, tf, steps);
        [
            y[0] - rf[0],
            y[1] - rf[1],
            y[2] - rf[2],
            y[3] - vf[0],
            y[4] - vf[1],
            y[5] - vf[2],
        ]
    };

    let mut p = [0.0; 6];
    let mut res = shoot(p);
    let mut err = norm6(res);
    for _ in 0..60 {
        if err < 1e-9 {
            break;
        }
        // Finite-difference the 6×6 shooting Jacobian ∂res/∂p.
        let mut jac = [[0.0; 6]; 6];
        for j in 0..6 {
            let eps = 1e-6 * (1.0 + libm::fabs(p[j]));
            let mut pp = p;
            pp[j] += eps;
            let rp = shoot(pp);
            for i in 0..6 {
                jac[i][j] = (rp[i] - res[i]) / eps;
            }
        }
        let Some(dp) = solve6(jac, res) else { break };
        for j in 0..6 {
            p[j] -= dp[j];
        }
        res = shoot(p);
        err = norm6(res);
    }

    MinEnergyTransfer {
        initial_costate: p,
        terminal_error: err,
        energy_cost: rk4(augment(p), mu, tf, steps)[12],
        converged: err < 1e-6,
    }
}

/// The optimal thrust acceleration `u = −λ_v` at time `t`, integrating the costate from `initial`.
#[must_use]
pub fn optimal_control_at(
    r0: [f64; 3],
    v0: [f64; 3],
    initial_costate: [f64; 6],
    mu: f64,
    t: f64,
) -> [f64; 3] {
    let y0: Aug = [
        r0[0],
        r0[1],
        r0[2],
        v0[0],
        v0[1],
        v0[2],
        initial_costate[0],
        initial_costate[1],
        initial_costate[2],
        initial_costate[3],
        initial_costate[4],
        initial_costate[5],
        0.0,
    ];
    let y = rk4(y0, mu, t, 400.max((t.abs() * 4.0) as usize));
    [-y[9], -y[10], -y[11]]
}

fn norm6(v: [f64; 6]) -> f64 {
    libm::sqrt(v.iter().map(|x| x * x).sum())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision, clippy::expect_used)]
    use super::*;

    const MU_EARTH: f64 = 3.986_004_418e14; // m³/s²

    /// A double-integrator transfer with no zero and no symmetry in any component, so that no axis
    /// of the shooter can be corrupted without moving the answer.
    const R0: [f64; 3] = [1200.0, -430.0, 260.0];
    const V0: [f64; 3] = [1.7, 2.3, -0.9];
    const RF: [f64; 3] = [1580.0, 65.0, -140.0];
    const VF: [f64; 3] = [-0.6, 1.1, 0.45];
    const TF: f64 = 137.0;

    /// The optimal initial costate `[λ_r(0), λ_v(0)]` of that transfer, at 60 decimal digits. The
    /// double integrator's controllability Gramian over `[0, T]` is `[[T³/3, T²/2], [T²/2, T]]`, so
    /// with `Δr = r_f − r₀ − v₀T` and `Δv = v_f − v₀` the minimum-energy control is
    /// `u(t) = (T−t)(12Δr/T³ − 6Δv/T²) + (−6Δr/T² + 4Δv/T)`, whence `λ_v = −u` and `λ_r = u̇`.
    /// Derived and checked in `derivations/min_energy_double_integrator.mac`.
    const COSTATE: [f64; 6] = [
        -0.001_421_741_783_411_301_365_467_907,
        -0.001_223_169_280_919_422_576_363_494,
        0.001_722_867_299_822_311_444_597_455,
        -0.080_600_990_995_790_931_855_719_54,
        -0.075_027_971_655_389_205_604_986_95,
        0.108_162_395_439_288_187_969_524_2,
    ];

    /// `½∫₀ᵀ|u|² dt = ½ δᵀ W⁻¹ δ` for the same transfer, at 60 decimal digits.
    const ENERGY_COST: f64 = 0.726_096_422_778_202_759_403_318_0;

    /// `(t, u(t))` at five points of the same optimal trajectory, at 60 decimal digits.
    const CONTROL: [(f64, [f64; 3]); 5] = [
        (
            0.0,
            [
                0.080_600_990_995_790_931_855_719_54,
                0.075_027_971_655_389_205_604_986_95,
                -0.108_162_395_439_288_187_969_524_2,
            ],
        ),
        (
            34.25,
            [
                0.031_906_334_913_953_860_088_443_71,
                0.033_134_423_783_898_982_364_537_27,
                -0.049_154_190_420_374_020_992_061_38,
            ],
        ),
        (
            68.5,
            [
                -0.016_788_321_167_883_211_678_832_12,
                -0.008_759_124_087_591_240_875_912_409,
                0.009_854_014_598_540_145_985_401_460,
            ],
        ),
        (
            102.75,
            [
                -0.065_482_977_249_720_283_446_107_94,
                -0.050_652_671_959_081_464_116_362_09,
                0.068_862_219_617_454_312_962_864_30,
            ],
        ),
        (
            137.0,
            [
                -0.114_177_633_331_557_355_213_383_8,
                -0.092_546_219_830_571_687_356_811_76,
                0.127_870_424_636_368_479_940_327_1,
            ],
        ),
    ];

    /// A state and costate with nothing at zero: a real low-Earth orbit position and velocity, and a
    /// costate of the magnitude the shooter converges on.
    const AUGMENTED: Aug = [
        7.1e6, -2.4e6, 1.9e6, 1.1e3, 6.9e3, -2.2e3, 3.1e-7, -1.4e-7, 8.2e-8, -2.7e-4, 5.3e-4,
        1.6e-4, 0.0,
    ];

    /// The PMP Hamiltonian `H = λ_r·v − μ(λ_v·r)/|r|³ − ½|λ_v|²`, written once and independently of
    /// the dynamics it generates.
    fn hamiltonian(y: Aug, mu: f64) -> f64 {
        let rn = libm::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
        (y[6] * y[3] + y[7] * y[4] + y[8] * y[5])
            - mu * (y[9] * y[0] + y[10] * y[1] + y[11] * y[2]) / (rn * rn * rn)
            - 0.5 * (y[9] * y[9] + y[10] * y[10] + y[11] * y[11])
    }

    /// The module rests on one claim: the augmented dynamics are Hamilton's equations for the PMP
    /// Hamiltonian. Written out that is `ṙ = ∂H/∂λ_r`, `v̇ = ∂H/∂λ_v`, `λ̇_r = −∂H/∂r` and
    /// `λ̇_v = −∂H/∂v`. Differencing `H` — one scalar formula that shares no arithmetic with the
    /// twelve components it generates — checks all of them at once, including the gravity gradient
    /// in the position costate, which no terminal-state assertion can see because the shooter
    /// converges on whatever adjoint equation it is given.
    #[test]
    fn the_dynamics_are_hamiltons_equations_for_the_pmp_hamiltonian() {
        let d = deriv(AUGMENTED, MU_EARTH);
        for i in 0..12 {
            let step = 1e-5 * AUGMENTED[i].abs();
            let (mut lo, mut hi) = (AUGMENTED, AUGMENTED);
            lo[i] -= step;
            hi[i] += step;
            let slope = (hamiltonian(hi, MU_EARTH) - hamiltonian(lo, MU_EARTH)) / (2.0 * step);
            // The first six are −∂H/∂(r, v) = λ̇; the last six are ∂H/∂(λ_r, λ_v) = (ṙ, v̇).
            let want = if i < 6 { -d[6 + i] } else { d[i - 6] };
            let err = (slope - want).abs();
            assert!(
                err <= 1e-7 * want.abs(),
                "component {i}: the dynamics give {want:.17e}, differencing H gives {slope:.17e}"
            );
        }
    }

    /// Four Runge-Kutta stages weighted `1, 2, 2, 1` are fourth order, so halving the step must cut
    /// the gap between consecutive solutions by sixteen. Nothing weaker pins the individual stage
    /// weights: any consistent one-step method reaches the same limit, only more slowly. The cost is
    /// carried as a state precisely so that it converges at the same order as the trajectory, which
    /// is why it gets a ladder of its own.
    #[test]
    fn the_augmented_integrator_converges_at_fourth_order() {
        let span = 600.0;
        let solutions: Vec<Aug> = [10usize, 20, 40, 80, 160]
            .iter()
            .map(|&n| rk4(AUGMENTED, MU_EARTH, span, n))
            .collect();
        for (name, component) in [("position", 0usize), ("cost", 12usize)] {
            for (k, w) in solutions.windows(3).enumerate() {
                let ratio = (w[0][component] - w[1][component]).abs()
                    / (w[1][component] - w[2][component]).abs();
                assert!(
                    (13.0..19.0).contains(&ratio),
                    "{name}: halving {k} converged at a ratio of {ratio:.2}, not 16"
                );
            }
        }
    }

    /// The shooting Jacobian is solved with partial pivoting, so a zero in the leading position is
    /// ordinary input, not a failure — and a singular system must be refused rather than answered
    /// with infinities. Newton hides both: a wrong step still walks to the same root, or fails to
    /// converge for a reason the caller cannot attribute.
    #[test]
    fn the_linear_solver_pivots_and_refuses_a_singular_system() {
        // b = A x for x = [1, −2, 3, −4, 5, −6], every entry exact in binary. Row 0 leads with a
        // zero, so the answer exists only if the solver pivots.
        let a = [
            [0.0, 2.0, -1.0, 4.0, 0.5, 3.0],
            [3.0, -1.0, 2.0, 0.25, -2.0, 1.0],
            [1.0, 4.0, -3.0, 2.0, 1.0, -0.5],
            [-2.0, 1.0, 5.0, -1.0, 0.5, 2.0],
            [4.0, 0.5, 1.0, 3.0, -1.0, -2.0],
            [1.0, -3.0, 0.25, 2.0, 4.0, 1.0],
        ];
        let want = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let b: [f64; 6] = core::array::from_fn(|i| (0..6).map(|j| a[i][j] * want[j]).sum::<f64>());
        let got = solve6(a, b).expect("a nonsingular system was refused");
        for i in 0..6 {
            assert!(
                (got[i] - want[i]).abs() <= 1e-13 * want[i].abs(),
                "component {i}: solved {:.17e}, want {:.17e}",
                got[i],
                want[i]
            );
        }
        let mut singular = a;
        singular[3] = core::array::from_fn(|j| 2.0 * singular[1][j]);
        assert!(
            solve6(singular, b).is_none(),
            "a row that is twice another was solved instead of refused"
        );
    }

    /// On the double integrator the minimum-energy problem has a closed form, so the shooter can be
    /// held to it rather than to its own residual: the optimal costate, the energy cost and the
    /// control at five points of the trajectory all follow from the controllability Gramian. The
    /// whole augmented system is polynomial of degree three in time there, so the integrator is
    /// exact and what is left is the shooter itself.
    #[test]
    fn the_double_integrator_transfer_is_its_closed_form() {
        let sol = min_energy_transfer(R0, V0, RF, VF, 0.0, TF);
        assert!(
            sol.converged && sol.terminal_error < 1e-9,
            "the shooter did not converge: {sol:?}"
        );
        let scale = COSTATE.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for i in 0..6 {
            let err = (sol.initial_costate[i] - COSTATE[i]).abs();
            assert!(
                err <= 1e-11 * scale,
                "costate {i}: got {:.17e}, want {:.17e}",
                sol.initial_costate[i],
                COSTATE[i]
            );
        }
        let cost_err = (sol.energy_cost - ENERGY_COST).abs();
        assert!(
            cost_err <= 1e-11 * ENERGY_COST,
            "energy cost {:.17e}, want {ENERGY_COST:.17e}",
            sol.energy_cost
        );
        let u_scale = CONTROL[0].1.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for &(t, want) in &CONTROL {
            let got = optimal_control_at(R0, V0, sol.initial_costate, 0.0, t);
            for i in 0..3 {
                assert!(
                    (got[i] - want[i]).abs() <= 1e-11 * u_scale,
                    "u({t}) axis {i}: got {:.17e}, want {:.17e}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    /// A minimum-energy transfer between two LEO states converges to the target, and its control is
    /// the primer vector `u = −λ_v`.
    #[test]
    fn min_energy_transfer_reaches_the_target() {
        let r0 = [7_000e3, 0.0, 0.0];
        let vc = (MU_EARTH / 7_000e3).sqrt();
        let v0 = [0.0, vc, 0.0];
        // Target: a slightly raised, phased state a quarter orbit later.
        let period = std::f64::consts::TAU * (7_000e3_f64.powi(3) / MU_EARTH).sqrt();
        let tf = period / 4.0;
        let rf = [0.0, 7_050e3, 0.0];
        let vf = [-vc * 0.99, 0.0, 0.0];

        let sol = min_energy_transfer(r0, v0, rf, vf, MU_EARTH, tf);
        println!(
            "min-energy: terminal error {:.3e}, cost {:.3e}",
            sol.terminal_error, sol.energy_cost
        );
        assert!(
            sol.converged,
            "shooter did not converge (err {:.3e})",
            sol.terminal_error
        );
        assert!(
            sol.terminal_error < 1.0,
            "terminal state off by {:.3e}",
            sol.terminal_error
        );
    }

    /// On the double-integrator (μ = 0) the minimum-energy control is affine in time — the analytic
    /// signature of the optimum. Verifies the shooter machinery against a known optimality structure.
    #[test]
    fn double_integrator_optimal_control_is_affine() {
        let r0 = [1000.0, -500.0, 200.0];
        let v0 = [1.0, 2.0, -0.5];
        let rf = [1500.0, 0.0, 100.0];
        let vf = [0.0, 0.0, 0.0];
        let tf = 100.0;
        let sol = min_energy_transfer(r0, v0, rf, vf, 0.0, tf);
        assert!(
            sol.converged,
            "double integrator did not converge ({:.3e})",
            sol.terminal_error
        );
        // u(t) = −λ_v(t) must be affine: u(T/2) is the midpoint of u(0) and u(T).
        let u0 = optimal_control_at(r0, v0, sol.initial_costate, 0.0, 0.0);
        let um = optimal_control_at(r0, v0, sol.initial_costate, 0.0, tf / 2.0);
        let ut = optimal_control_at(r0, v0, sol.initial_costate, 0.0, tf);
        for i in 0..3 {
            assert!(
                (um[i] - f64::midpoint(u0[i], ut[i])).abs() < 1e-6,
                "control not affine on axis {i}"
            );
        }
    }
}
