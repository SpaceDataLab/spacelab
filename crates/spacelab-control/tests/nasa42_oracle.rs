//! The attitude plant against a second, independent oracle: NASA's 42.
//!
//! Basilisk already agrees with this workspace to 1.2e-11 rad on the closed loop
//! (`basilisk_oracle.rs`). Two implementations agreeing is worth less than three, and the two that
//! agree are both descendants of the same textbook. 42 (`ericstoneking/42`, NASA Goddard, **NASA
//! Open Source Agreement v1.3**) is a separate code base with its own rigid-body formulation, and
//! it supplies the leg Basilisk's fixture covers only pointwise: **gravity-gradient torque driving
//! the attitude over a whole orbit arc**, rather than the torque evaluated at five geometries.
//!
//! 42 is run unmodified, from input decks only, and nothing of it is redistributed — so none of
//! NOSA's Modification obligations arise. `tools/gen_42_oracle.py` records the licence reading and
//! generates the fixture; 42 is an oracle and never a dependency.
//!
//! # What this leg costs, and what it therefore cannot claim
//!
//! 42 prints seven significant digits. That is a hard floor at ~1e-7 in a unit quaternion, so this
//! comparison cannot resolve the 1e-11 the Basilisk leg does and does not try to. What it can do is
//! separate a *model* error from a *discretisation* error, because those differ by four orders of
//! magnitude here: a transposed body rotation or a sign in `r × (I r)` moves the arc by ~1e-2 rad,
//! and the quantisation floor is 1e-7.
//!
//! # The finding this leg shares with the Basilisk one
//!
//! 42 recomputes the environment once per simulation step and holds it across the integrator's
//! stages. Its own step-refinement sweep shows it: halving the step cuts the attitude difference by
//! 2.3x and then 3.0x, an apparent order near 1.5 rather than 4, which is what a fourth-order method
//! looks like when part of its right-hand side is zero-order held. Re-evaluating the torque per
//! stage on this side would therefore compare a schedule against an integrator and read the
//! difference as a model defect.
//!
//! Held the same way, the workspace reproduces 42 at **every** step it was run at — 1.0e-6 rad
//! against a 5.290e-4 rad spread between 42's own coarsest and finest runs. That factor of 503 is
//! what turns "we agree in the converged limit" into "we agree with the oracle's actual schedule",
//! and it is the same lesson the Basilisk actuator ordering taught, reached from a different
//! direction: in both simulators the thing that decides the answer at the accuracy anyone cares
//! about is *when* the right-hand side is sampled, not how it is integrated.

#![allow(clippy::expect_used)]

use spacelab_control::attitude::AttitudeError;
use spacelab_propagation::attitude::{Inertia, exp_so3, integrate_attitude_torqued};
use spacelab_propagation::torque::gravity_gradient;

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

const FIXTURE: &str = include_str!("fixtures/nasa42_gravity_gradient.txt");

const MOMENTS: Vec3 = [10.0, 12.0, 8.0];
/// 42's own Earth `mu`, from the world table in its source. Basilisk's differs in the ninth digit,
/// which is above what this leg resolves, so each oracle is compared in its own constant.
const MU_42: f64 = 3.986_004_418e14;

fn fields(line: &str) -> Vec<f64> {
    line.split_whitespace()
        .map(|t| t.parse().expect("a number"))
        .collect()
}

fn rows(fixture: &str) -> impl Iterator<Item = Vec<f64>> {
    fixture
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(fields)
}

fn vec3(v: &[f64]) -> Vec3 {
    [v[0], v[1], v[2]]
}

fn norm(v: Vec3) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: Vec3, k: f64) -> Vec3 {
    [v[0] * k, v[1] * k, v[2] * k]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The rotation whose quaternion is `q = (q_v, q_0)`, built through the crate's verified `exp_so3`
/// rather than through a second transcription of the quaternion-to-matrix formula.
///
/// `q_v = n̂ sin(θ/2)`, `q_0 = cos(θ/2)`, so `θ = 2 atan2(‖q_v‖, q_0)` about `n̂`. Same discipline as
/// the MRP helper in the Basilisk leg, and for the same reason: a convention mismatch is the failure
/// this file exists to catch, so the conversion must not be a second place to make one.
fn rotation_of_quaternion(q: [f64; 4]) -> Mat3 {
    let v = [q[0], q[1], q[2]];
    let s = norm(v);
    if s == 0.0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    exp_so3(scale(v, 2.0 * libm::atan2(s, q[3]) / s))
}

/// The body-frame components of an inertial vector. [`rotation_of_quaternion`] returns the
/// inertial-from-body rotation, so this is the transpose — the same convention the Basilisk leg
/// measured for MRPs, and it holds for 42's quaternions too.
fn to_body(inertial_from_body: &Mat3, v: Vec3) -> Vec3 {
    let m = inertial_from_body;
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// The principal angle between two rotations, in radians.
fn angle_between(a: &Mat3, b: &Mat3) -> f64 {
    AttitudeError::new(a, b, [0.0; 3], [0.0; 3]).angle()
}

struct Sample {
    step: f64,
    time: f64,
    position: Vec3,
    rotation: Mat3,
    omega: Vec3,
}

fn samples() -> Vec<Sample> {
    rows(FIXTURE)
        .map(|f| Sample {
            step: f[0],
            time: f[1],
            position: vec3(&f[2..5]),
            rotation: rotation_of_quaternion([f[5], f[6], f[7], f[8]]),
            omega: vec3(&f[9..12]),
        })
        .collect()
}

/// The circular orbit 42 was started on, in closed form.
///
/// The fixture's initial state is exactly circular in 42's own `mu`, so the arc needs no solver on
/// this side — which is the point: it makes the position an *independent* input to the torque rather
/// than a second numerical result, and the agreement against 42's own `PosN` is itself a check.
fn position_at(r0: Vec3, v0: Vec3, t: f64) -> Vec3 {
    let radius = norm(r0);
    let speed = norm(v0);
    let mean_motion = speed / radius;
    let (c, s) = (libm::cos(mean_motion * t), libm::sin(mean_motion * t));
    let along = scale(v0, radius / speed);
    [
        r0[0] * c + along[0] * s,
        r0[1] * c + along[1] * s,
        r0[2] * c + along[2] * s,
    ]
}

/// The circular orbit's initial state, recovered from the fixture's own position samples.
///
/// Not from the two-point difference `(r(t) − r0 cos nt)/sin nt`: at the first sample that divides a
/// one-metre printing quantum by `sin(0.114) = 0.113` and lands nine metres of noise in the
/// velocity. The orbit *plane* is what the samples determine well — two 6.8e6 m vectors fix their
/// normal to 1.5e-7 rad — so the direction comes from `n̂ × r̂0` and the magnitude from `√(μ/R)`,
/// which the deck made exact.
fn initial_state(samples: &[Sample]) -> (Vec3, Vec3) {
    let r0 = samples.first().expect("a first sample").position;
    let later = samples
        .iter()
        .find(|s| s.time > 0.0)
        .expect("a second sample")
        .position;
    let radius = norm(r0);
    let normal = {
        let n = cross(r0, later);
        scale(n, 1.0 / norm(n))
    };
    let along = cross(normal, scale(r0, 1.0 / radius));
    (r0, scale(along, libm::sqrt(MU_42 / radius)))
}

/// The quaternion-to-rotation conversion is the crate's own exponential, checked against the axis
/// and angle that built the fixture's initial attitude — 40 degrees about `(1, 2, 3)/√14`. (G3)
#[test]
fn the_quaternion_convention_is_the_crates_own_exponential() {
    let axis = {
        let n = norm([1.0, 2.0, 3.0]);
        scale([1.0, 2.0, 3.0], 1.0 / n)
    };
    let angle = 40.0_f64.to_radians();
    let half = 0.5 * angle;
    let q = [
        axis[0] * libm::sin(half),
        axis[1] * libm::sin(half),
        axis[2] * libm::sin(half),
        libm::cos(half),
    ];
    let gap = angle_between(&rotation_of_quaternion(q), &exp_so3(scale(axis, angle)));
    println!("quaternion vs exp_so3 on the fixture's initial attitude: {gap:.3e} rad");
    assert!(gap < 1e-15, "quaternion convention: {gap} rad");

    let first = samples().into_iter().next().expect("a first sample");
    let stated = angle_between(&first.rotation, &exp_so3(scale(axis, angle)));
    println!("42's t = 0 attitude against the deck it was given: {stated:.3e} rad");
    // 42 prints seven digits, so the round trip through its own output file is the floor here.
    assert!(stated < 1e-6, "42's initial attitude: {stated} rad");
}

/// 42's orbit against the closed-form circle it was started on.
///
/// The gravity-gradient torque is driven by this position, so an orbit that has quietly drifted
/// would show up as an attitude discrepancy with no attitude cause. Measuring it separately keeps
/// the two apart.
#[test]
fn the_orbit_42_propagated_is_the_circle_it_was_given() {
    let samples = samples();
    let (r0, v0) = initial_state(&samples);
    let radius = norm(r0);

    let (mut worst, mut worst_radius) = (0.0_f64, 0.0_f64);
    for sample in &samples {
        worst = worst.max(norm(sub(position_at(r0, v0, sample.time), sample.position)));
        worst_radius = worst_radius.max(libm::fabs(norm(sample.position) - radius));
    }
    println!(
        "42's orbit vs the closed-form circle: worst {worst:.3e} m over {} samples on a \
         {radius:.1} m radius; worst radius excursion {worst_radius:.3e} m",
        samples.len()
    );
    // 42 prints seven digits of a 6.8e6 m position, so one metre is the printing resolution and
    // the assertion is that the orbit is a circle to it, not that it is a circle to machine epsilon.
    assert!(worst < 2.0, "orbit gap {worst} m");
    assert!(worst_radius < 2.0, "radius excursion {worst_radius} m");
}

/// The gravity-gradient torque driving a tumbling body over a 2000 s arc, against 42.
///
/// The torque is held across each step rather than re-evaluated per integrator stage, because that
/// is what 42 does — and the flat residual below is the evidence, not an appeal to its source. 42's
/// coarsest and finest runs differ from each other by 5.290e-4 rad; if it re-evaluated the
/// environment per stage, holding it here would show that difference. It shows 1.0e-6 rad at every
/// step instead, which is 42's seven printed digits. Held the same way, this workspace reproduces
/// 42 *at each of its step sizes*, not merely in the converged limit.
///
/// There is therefore no convergence order to read off, and asserting one would be asserting noise.
/// What is asserted is the two things the run does establish: the residual sits at the oracle's
/// printing floor, and it is 503x below the oracle's own discretisation spread.
#[test]
fn the_gravity_gradient_arc_agrees_with_42_at_every_step_it_was_run_at() {
    let inertia = Inertia::principal(MOMENTS).expect("positive definite");
    let samples = samples();
    let mut steps: Vec<f64> = samples.iter().map(|s| s.step).collect();
    steps.dedup();
    assert!(
        steps.len() >= 3,
        "need a sweep to compare 42 against itself"
    );

    let (r0, v0) = initial_state(&samples);

    #[allow(clippy::float_cmp)] // the same parsed fixture field on both sides
    let run_at =
        |step: f64| -> Vec<&Sample> { samples.iter().filter(|s| s.step == step).collect() };

    let mut worst_gap = 0.0_f64;
    for &step in &steps {
        let run = run_at(step);
        let start = run.first().expect("a first sample");

        let mut rotation = start.rotation;
        let mut omega = start.omega;
        let mut time = 0.0;
        let (mut worst_angle, mut worst_rate) = (0.0_f64, 0.0_f64);

        for sample in run.iter().skip(1) {
            while time < sample.time - 0.5 * step {
                let held = gravity_gradient(
                    &inertia,
                    to_body(&rotation, position_at(r0, v0, time)),
                    MU_42,
                );
                let (r, w) = integrate_attitude_torqued(
                    &rotation,
                    omega,
                    time,
                    &inertia,
                    step,
                    1,
                    |_, _, _| held,
                );
                rotation = r;
                omega = w;
                time += step;
            }
            worst_angle = worst_angle.max(angle_between(&rotation, &sample.rotation));
            worst_rate = worst_rate.max(norm(sub(omega, sample.omega)));
        }
        println!(
            "  step {step:6.4} s: attitude gap {worst_angle:.3e} rad, rate gap {worst_rate:.3e} rad/s"
        );
        worst_gap = worst_gap.max(worst_angle);
    }

    // 42 against itself: the same scenario at the coarsest and the finest step it was run at. This
    // is the oracle's own discretisation error, and it is what the residual above has to be small
    // compared with for "reproduces 42 at each step" to mean anything.
    let (coarse, fine) = (run_at(steps[0]), run_at(steps[steps.len() - 1]));
    let mut spread = 0.0_f64;
    for (a, b) in coarse.iter().zip(&fine) {
        assert!(
            libm::fabs(a.time - b.time) < 1e-9,
            "the two runs must be sampled at the same times"
        );
        spread = spread.max(angle_between(&a.rotation, &b.rotation));
    }

    println!(
        "42 vs spacelab over a 2000 s arc: {worst_gap:.3e} rad at every step, against 42's own \
         {:.4}-to-{:.4} s spread of {spread:.3e} rad -- a factor {:.0}",
        steps[0],
        steps[steps.len() - 1],
        spread / worst_gap
    );

    assert!(
        worst_gap < 3e-6,
        "42 and spacelab must agree at 42's printed precision: {worst_gap} rad"
    );
    assert!(
        spread / worst_gap > 50.0,
        "the residual must be far below the oracle's own discretisation spread: \
         {worst_gap} rad against {spread} rad"
    );
}
