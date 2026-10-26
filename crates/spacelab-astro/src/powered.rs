//! Powered-descent guidance for a soft landing.
//!
//! Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV) guidance is the closed-form energy-optimal
//! feedback law for landing under constant gravity: it minimises `∫|a_T|²` and drives the state to a
//! target position at rest. With time-to-go `t_go`, current state `(r, v)`, target `(r_f, v_f)` and
//! gravity `g`,
//!
//! ```text
//! ZEM = r_f − (r + v·t_go + ½ g t_go²)      ZEV = v_f − (v + g t_go)
//! a_T = (6/t_go²) ZEM − (2/t_go) ZEV
//! ```
//!
//! is the commanded thrust acceleration (gravity is already folded into ZEM/ZEV). It is real-time and
//! optimal for the unconstrained-magnitude problem. The thrust-*constrained* fuel-optimal landing
//! (bounded `|T|`, minimise propellant) is solved to global optimum by **lossless convexification**
//! (Açıkmeşe–Ploen) as a second-order cone program — the production SOTA — which requires a conic
//! solver and is a further increment. SI units.
//!
//! Gate: `derivations/powered_zem_zev.mac` derives the affine minimum-energy command from
//! Pontryagin's principle, shows the ZEM/ZEV feedback reproduces it identically along its own
//! trajectory, and shows the held-command loop of [`powered_descent`] is deadbeat — exact landing in
//! position and velocity — from three steps on. `derivations/powered_zem_zev.gp` recomputes every
//! reference value of the tests in exact rationals. The module tests
//! `the_command_is_the_open_loop_optimum_at_every_point_of_its_own_trajectory` (G2) and
//! `the_closed_loop_reproduces_the_analytic_outcome_step_for_step` (G3) hold the code to them.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss
)]

use spacelab_core::Vec3;

/// The ZEM/ZEV optimal thrust-acceleration command for a soft landing.
///
/// `g` is the gravity acceleration vector (e.g. `[0, 0, −3.71]` on Mars). `t_go` is the time
/// remaining; it must be positive.
#[must_use]
pub fn zem_zev(r: Vec3, v: Vec3, r_target: Vec3, v_target: Vec3, g: Vec3, t_go: f64) -> Vec3 {
    let zem = Vec3::new(
        r_target.x - (r.x + v.x * t_go + 0.5 * g.x * t_go * t_go),
        r_target.y - (r.y + v.y * t_go + 0.5 * g.y * t_go * t_go),
        r_target.z - (r.z + v.z * t_go + 0.5 * g.z * t_go * t_go),
    );
    let zev = Vec3::new(
        v_target.x - (v.x + g.x * t_go),
        v_target.y - (v.y + g.y * t_go),
        v_target.z - (v.z + g.z * t_go),
    );
    Vec3::new(
        6.0 / (t_go * t_go) * zem.x - 2.0 / t_go * zev.x,
        6.0 / (t_go * t_go) * zem.y - 2.0 / t_go * zev.y,
        6.0 / (t_go * t_go) * zem.z - 2.0 / t_go * zev.z,
    )
}

/// The outcome of a closed-loop powered descent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landing {
    /// Position error at the terminal time, m.
    pub position_error: f64,
    /// Velocity error at the terminal time, m/s (0 is a soft landing).
    pub velocity_error: f64,
    /// Cumulative `∫|a_T| dt`, m/s — the guidance's velocity-change (fuel-proportional) budget.
    pub delta_v: f64,
}

/// A closed-loop ZEM/ZEV powered descent from `(r0, v0)` to rest at `r_target`, over `tof` seconds in
/// `steps` RK4 steps, under gravity `g`. Returns the terminal errors and the Δv expended.
#[must_use]
pub fn powered_descent(
    r0: Vec3,
    v0: Vec3,
    r_target: Vec3,
    g: Vec3,
    tof: f64,
    steps: usize,
) -> Landing {
    let v_target = Vec3::new(0.0, 0.0, 0.0);
    let h = tof / steps as f64;
    let (mut r, mut v) = (r0, v0);
    let mut dv = 0.0;
    for k in 0..steps {
        let t = k as f64 * h;
        let t_go = (tof - t).max(h); // floor at one step to avoid the 1/t_go² singularity at the end
        let a_t = zem_zev(r, v, r_target, v_target, g, t_go);
        dv += a_t.norm() * h;
        // Total acceleration = thrust command + gravity. The command is held across the step, so the
        // acceleration is constant there and the exact solution is the uniform-acceleration one; a
        // Runge-Kutta stage ladder would evaluate the same constant four times and reduce to this.
        let a = Vec3::new(a_t.x + g.x, a_t.y + g.y, a_t.z + g.z);
        r = Vec3::new(
            r.x + (v.x + 0.5 * a.x * h) * h,
            r.y + (v.y + 0.5 * a.y * h) * h,
            r.z + (v.z + 0.5 * a.z * h) * h,
        );
        v = Vec3::new(v.x + a.x * h, v.y + a.y * h, v.z + a.z * h);
    }
    let miss = Vec3::new(r.x - r_target.x, r.y - r_target.y, r.z - r_target.z);
    Landing {
        position_error: miss.norm(),
        velocity_error: v.norm(),
        delta_v: dv,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    use super::*;

    const MARS_G: f64 = 3.71; // m/s²

    /// A descent with no zero component and no symmetry anywhere — tilted gravity, an off-axis pad,
    /// a non-zero arrival velocity — so that no coordinate of the guidance law can be corrupted
    /// without moving the answer.
    const R_START: Vec3 = Vec3 {
        x: 1200.0,
        y: -450.0,
        z: 2600.0,
    };
    const V_START: Vec3 = Vec3 {
        x: 85.0,
        y: 37.0,
        z: -110.0,
    };
    const R_PAD: Vec3 = Vec3 {
        x: 640.0,
        y: 215.0,
        z: 12.0,
    };
    const V_ARRIVAL: Vec3 = Vec3 {
        x: 1.5,
        y: -0.8,
        z: -0.6,
    };
    const GRAVITY: Vec3 = Vec3 {
        x: 0.04,
        y: -0.03,
        z: -MARS_G,
    };
    const DESCENT_TIME: f64 = 42.0;

    /// One sample of the optimal trajectory: `(t_go, r, v, a)`.
    type OptimalSample = (f64, [f64; 3], [f64; 3], [f64; 3]);

    /// The minimum-energy solution of the landing boundary-value problem, evaluated at 60 decimal
    /// digits: `(t_go, r, v, a)` at five points of the optimal trajectory from `R_START`/`V_START`
    /// to `R_PAD`/`V_ARRIVAL`. The command is the affine `A + B t` the terminal conditions imply,
    /// derived independently of the feedback form the code implements; recomputed in exact
    /// rationals by `derivations/powered_zem_zev.gp`, which agrees to the 25 digits written.
    const OPTIMAL_TRAJECTORY: [OptimalSample; 5] = [
        (
            42.0,
            [1200.0, -450.0, 2600.0],
            [85.0, 37.0, -110.0],
            [
                -10.111_428_571_428_571_428_571_43,
                -1.193_809_523_809_523_809_523_810,
                5.412_040_816_326_530_612_244_898,
            ],
        ),
        (
            31.5,
            [1_611.578_125, -125.9875, 1547.11875],
            [0.46875, 25.0, -89.758_928_571_428_571_428_571_43],
            [
                -6.069_761_904_761_904_761_904_762,
                -1.031_904_761_904_761_904_761_905,
                5.863_401_360_544_217_687_074_830,
            ],
        ),
        (
            21.0,
            [1358.375, 80.95, 731.65],
            [-41.625, 14.7, -64.778_571_428_571_428_571_428_57],
            [
                -2.028_095_238_095_238_095_238_095,
                -0.87,
                6.314_761_904_761_904_761_904_762,
            ],
        ),
        (
            10.5,
            [885.984_375, 188.6625, 203.35625],
            [-41.28125, 6.1, -35.058_928_571_428_571_428_571_43],
            [
                2.013_571_428_571_428_571_428_571,
                -0.708_095_238_095_238_095_238_095_2,
                6.766_122_448_979_591_836_734_694,
            ],
        ),
        (
            2.1,
            [649.695_875, 215.3857, 20.92765],
            [-10.45125, 0.444, -7.870_928_571_428_571_428_571_429],
            [
                5.246_904_761_904_761_904_761_905,
                -0.578_571_428_571_428_571_428_571_4,
                7.127_210_884_353_741_496_598_639,
            ],
        ),
    ];

    /// `(steps, position error, velocity error, Δv)` of the closed loop, at 60 decimal digits. One
    /// and two steps are too few for the guidance to null the state; three already land exactly.
    /// Recomputed in exact rationals by `derivations/powered_zem_zev.gp`.
    const DESCENT_OUTCOME: [(usize, f64, f64, f64); 4] = [
        (
            1,
            4_728.151_012_816_743_779_516_674,
            337.725_072_344_053_127_108_333_8,
            481.263_798_797_138_970_121_599_8,
        ),
        (
            2,
            2_364.075_506_408_371_889_758_337,
            337.725_072_344_053_127_108_333_8,
            738.542_115_044_822_277_610_989_7,
        ),
        (3, 0.0, 0.0, 359.209_607_027_126_200_648_613_1),
        (5, 0.0, 0.0, 342.579_904_624_153_023_115_878_1),
    ];

    /// (G2) Under constant gravity the minimum-energy thrust is affine in time and its two coefficients
    /// are fixed by the terminal conditions alone. The feedback law must reproduce that open-loop
    /// optimum at every point of the trajectory it generates, not only at the start — which is what
    /// makes it optimal rather than merely convergent. `derivations/powered_zem_zev.mac` proves the
    /// identity symbolically, for every point of every optimal trajectory.
    #[test]
    fn the_command_is_the_open_loop_optimum_at_every_point_of_its_own_trajectory() {
        let pad = [R_PAD.x, R_PAD.y, R_PAD.z];
        let arrival = [V_ARRIVAL.x, V_ARRIVAL.y, V_ARRIVAL.z];
        let g = [GRAVITY.x, GRAVITY.y, GRAVITY.z];
        for &(t_go, r, v, want) in &OPTIMAL_TRAJECTORY {
            let got = zem_zev(
                Vec3::new(r[0], r[1], r[2]),
                Vec3::new(v[0], v[1], v[2]),
                R_PAD,
                V_ARRIVAL,
                GRAVITY,
                t_go,
            );
            for (axis, got) in [(0, got.x), (1, got.y), (2, got.z)] {
                // ZEM cancels the arrival position against the coasted one; what survives is the
                // rounding of the terms that cancelled, amplified by the 6/t_go² the law applies.
                let bound = 4.0
                    * f64::EPSILON
                    * (6.0 / (t_go * t_go)
                        * (pad[axis].abs()
                            + r[axis].abs()
                            + v[axis].abs() * t_go
                            + 0.5 * g[axis].abs() * t_go * t_go)
                        + 2.0 / t_go
                            * (arrival[axis].abs() + v[axis].abs() + g[axis].abs() * t_go));
                let err = (got - want[axis]).abs();
                assert!(
                    err <= bound,
                    "axis {axis} at t_go {t_go} s: off by {err:.3e} m/s^2, bound {bound:.3e}"
                );
            }
        }
    }

    /// (G3) The command is held across the step, so the acceleration is constant there and the step is the
    /// uniform-acceleration closed form. One and two steps leave the state far from the pad, which is
    /// where an error in the step is large enough to see; from three steps on the discrete controller
    /// is deadbeat and nulls position and velocity outright — the exact landing
    /// `derivations/powered_zem_zev.mac` proves for 3 to 6 steps in symbolic arithmetic.
    #[test]
    fn the_closed_loop_reproduces_the_analytic_outcome_step_for_step() {
        let scale = (R_START - R_PAD).norm() + DESCENT_TIME * V_START.norm();
        for &(steps, position_error, velocity_error, delta_v) in &DESCENT_OUTCOME {
            let got = powered_descent(R_START, V_START, R_PAD, GRAVITY, DESCENT_TIME, steps);
            let dp = (got.position_error - position_error).abs();
            assert!(
                dp <= 1e-12 * scale,
                "{steps} steps: position error {:.6e} m, expected {position_error:.6e}",
                got.position_error
            );
            let dv_err = (got.velocity_error - velocity_error).abs();
            assert!(
                dv_err <= 1e-12 * V_START.norm(),
                "{steps} steps: velocity error {:.6e} m/s, expected {velocity_error:.6e}",
                got.velocity_error
            );
            let dd = (got.delta_v - delta_v).abs();
            assert!(
                dd <= 1e-13 * delta_v,
                "{steps} steps: delta-v {:.9} m/s, expected {delta_v:.9}",
                got.delta_v
            );
        }
    }

    /// The only error left once the controller is deadbeat is the zero-order hold on the command,
    /// which is first order in the step. So halving the step must halve the gap between the
    /// propellant spent and the analytic `∫|a| dt` of the continuous optimum — and every run on
    /// the ladder must still land exactly.
    #[test]
    fn the_propellant_spent_converges_to_the_optimum_at_first_order() {
        const OPTIMAL_DELTA_V: f64 = 338.138_284_788_635_156_077_605_1;
        let mut gaps = Vec::new();
        for steps in [512usize, 1024, 2048, 4096, 8192] {
            let landing = powered_descent(R_START, V_START, R_PAD, GRAVITY, DESCENT_TIME, steps);
            assert!(
                landing.position_error < 1e-9 && landing.velocity_error < 1e-9,
                "{steps} steps did not land: {landing:?}"
            );
            gaps.push((OPTIMAL_DELTA_V - landing.delta_v).abs());
        }
        for (k, pair) in gaps.windows(2).enumerate() {
            let ratio = pair[0] / pair[1];
            assert!(
                (1.8..2.2).contains(&ratio),
                "halving the step {k} changed the propellant gap by {ratio:.3}, not 2: {gaps:?}"
            );
        }
        let finest = gaps[gaps.len() - 1];
        assert!(
            finest < 2e-6 * OPTIMAL_DELTA_V,
            "the finest run is still {finest:.3e} m/s off the optimum"
        );
    }

    /// A ZEM/ZEV powered descent lands softly: the terminal position hits the target and the velocity
    /// is driven to zero.
    #[test]
    fn zem_zev_descent_lands_softly() {
        let r0 = Vec3::new(0.0, 0.0, 2_000.0); // 2 km up
        let v0 = Vec3::new(120.0, 0.0, -60.0); // cross-range plus descent
        let target = Vec3::new(600.0, 0.0, 0.0); // 600 m downrange, on the surface
        let g = Vec3::new(0.0, 0.0, -MARS_G);
        let landing = powered_descent(r0, v0, target, g, 40.0, 20_000);
        println!(
            "ZEM/ZEV landing: position error {:.3} m, velocity error {:.4} m/s, Δv {:.1} m/s",
            landing.position_error, landing.velocity_error, landing.delta_v
        );
        assert!(
            landing.position_error < 1.0,
            "missed the pad by {:.2} m",
            landing.position_error
        );
        assert!(
            landing.velocity_error < 0.5,
            "not a soft landing: {:.3} m/s",
            landing.velocity_error
        );
    }

    /// The optimality signature: on a linear plant (constant gravity) the minimum-energy ZEM/ZEV
    /// command is affine in time, so along its own closed-loop trajectory the command at the midpoint
    /// equals the mean of the endpoint commands. This is the analytic fingerprint of the optimum.
    #[test]
    fn optimal_thrust_is_affine_in_time() {
        let g = Vec3::new(0.0, 0.0, -MARS_G);
        let target = Vec3::new(600.0, 0.0, 0.0);
        let (r0, v0) = (Vec3::new(0.0, 0.0, 2_000.0), Vec3::new(120.0, 0.0, -60.0));
        let tof = 40.0;
        let steps = 4000usize;
        let h = tof / steps as f64;
        let (mut r, mut v) = (r0, v0);
        let mut cmd = [Vec3::ZERO; 3]; // command at t = 0, T/2, and the last full step
        for k in 0..steps {
            let t = k as f64 * h;
            let t_go = (tof - t).max(h);
            let a_t = zem_zev(r, v, target, Vec3::ZERO, g, t_go);
            if k == 0 {
                cmd[0] = a_t;
            } else if k == steps / 2 {
                cmd[1] = a_t;
            } else if k == steps - 1 {
                cmd[2] = a_t;
            }
            let accel = Vec3::new(a_t.x + g.x, a_t.y + g.y, a_t.z + g.z);
            r = Vec3::new(
                r.x + v.x * h + 0.5 * accel.x * h * h,
                r.y + v.y * h + 0.5 * accel.y * h * h,
                r.z + v.z * h + 0.5 * accel.z * h * h,
            );
            v = Vec3::new(v.x + accel.x * h, v.y + accel.y * h, v.z + accel.z * h);
        }
        // Midpoint command is the mean of the endpoints on each axis (affine ⇒ linear ⇒ this holds).
        for (axis, (mid, (a, b))) in [
            (cmd[1].x, (cmd[0].x, cmd[2].x)),
            (cmd[1].z, (cmd[0].z, cmd[2].z)),
        ]
        .iter()
        .enumerate()
        {
            let mean = 0.5 * (a + b);
            assert!(
                (mid - mean).abs() < 0.02,
                "command not affine on axis {axis}: mid {mid:.4} vs mean {mean:.4}"
            );
        }
    }
}
