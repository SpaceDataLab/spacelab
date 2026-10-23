//! The 2-D usage-violation detector against exact two-body kinematics, on the two regimes the source
//! paper separates.
//!
//! The unit tests in the module build their curvature by hand — a constant acceleration, an inflating
//! covariance — which pins the arithmetic but says nothing about whether the thresholds sort *real*
//! conjunctions the way the paper says they do. This file supplies that: two circular orbits, whose
//! positions and velocities are closed form and therefore exact, arranged into the two cases the
//! paper's statistics are about.
//!
//! * **A crossing conjunction** at ~10.7 km/s. The paper's 94.7 %: the encounter is over in a third of
//!   a second, the trajectory has no time to bend, and nothing should flag.
//! * **A co-orbiting pair** 300 m apart in altitude, closing at 16 cm/s. The paper's low-velocity
//!   failure case, and one where the exact answer is known in closed form: for two coplanar circular
//!   orbits the along-track drift rate is `(3/2)nδ` while the *inertial* speed difference — all the
//!   rectilinear model has — is `(1/2)nδ`. The straight line through TCA therefore separates exactly
//!   three times too slowly, the encounter it predicts is three times too long, and the 2-D estimate
//!   is three times too large. `F_c = 1/3`, to be checked and not asserted from a plausible sign.
//!
//! Both samplers hold the covariance frozen and vary only the mean trajectory. That is deliberately
//! *half* of what the paper's curvilinear model varies, and it makes the test conservative rather than
//! generous: every flag raised here is raised by geometry alone, with the covariance given the benefit
//! of the 2-D method's own assumption.
//!
//! What this file does not cover: the offset check, which needs TCA and the Mahalanobis minimum to
//! separate and is pinned on both sides of its threshold by construction in the module's own tests.

#![allow(clippy::unwrap_used)]

use std::f64::consts::FRAC_PI_2;

use spacelab_estimation::conjunction::ConjunctionGeometry;
use spacelab_estimation::conjunction_domain::{
    CurvilinearEncounter, EXTENDED_LIMIT, RectilinearEncounter, RelativeSample, UsageViolations,
    usage_violations,
};

/// Earth's gravitational parameter, m³/s² — EGM2008 / WGS-84, the value the rest of the workspace
/// uses. Only the orbital rates depend on it, and only through their ratio.
const MU: f64 = 3.986_004_418e14;

/// Isotropic position uncertainty per object, metres.
const SIGMA: f64 = 300.0;

/// State on a circular orbit of the given radius, inclined about the x axis, at argument of latitude
/// zero when `t = 0`. Closed form: the trajectory handed to the detector is the exact two-body one,
/// not an integrator's approximation of it.
fn circular(radius: f64, inclination: f64, t: f64) -> ([f64; 3], [f64; 3]) {
    let rate = (MU / (radius * radius * radius)).sqrt();
    let (sin_u, cos_u) = (rate * t).sin_cos();
    let (sin_i, cos_i) = inclination.sin_cos();
    let speed = radius * rate;
    (
        [
            radius * cos_u,
            radius * sin_u * cos_i,
            radius * sin_u * sin_i,
        ],
        [-speed * sin_u, speed * cos_u * cos_i, speed * cos_u * sin_i],
    )
}

fn isotropic(sigma: f64) -> [[f64; 3]; 3] {
    let variance = sigma * sigma;
    [
        [variance, 0.0, 0.0],
        [0.0, variance, 0.0],
        [0.0, 0.0, variance],
    ]
}

/// The relative state of the pair `t` seconds after the epoch at which both are at argument of
/// latitude zero — which is a closest approach, since the separation there is purely radial and the
/// relative velocity purely transverse.
fn pair(lower: f64, upper: f64, inclination: f64, t: f64) -> ([f64; 3], [f64; 3]) {
    let (primary_position, primary_velocity) = circular(lower, 0.0, t);
    let (secondary_position, secondary_velocity) = circular(upper, inclination, t);
    let mut position = [0.0_f64; 3];
    let mut velocity = [0.0_f64; 3];
    for axis in 0..3 {
        position[axis] = secondary_position[axis] - primary_position[axis];
        velocity[axis] = secondary_velocity[axis] - primary_velocity[axis];
    }
    (position, velocity)
}

fn geometry_at_tca(lower: f64, upper: f64, inclination: f64) -> ConjunctionGeometry {
    let (position, velocity) = pair(lower, upper, inclination, 0.0);
    ConjunctionGeometry::new(
        position,
        velocity,
        isotropic(SIGMA),
        isotropic(SIGMA),
        3.0,
        2.0,
    )
}

fn period(radius: f64) -> f64 {
    std::f64::consts::TAU * (radius * radius * radius / MU).sqrt()
}

/// A crossing conjunction: two circular orbits 300 m apart in radius, at right angles. Relative speed
/// ~10.7 km/s, encounter duration a third of a second, and the trajectory is a straight line over
/// every scale that matters — which is the regime the 2-D method was built for and the one it holds
/// on 94.7 % of real conjunctions.
#[test]
fn a_ten_kilometre_per_second_crossing_registers_no_usage_violation() {
    let lower = 7.0e6;
    let upper = lower + 300.0;
    let geometry = geometry_at_tca(lower, upper, FRAC_PI_2);
    let covariance = isotropic(SIGMA * std::f64::consts::SQRT_2);

    let speed = {
        let v = geometry.relative_velocity;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    };
    assert!(
        (speed - 10_672.0).abs() < 5.0,
        "the crossing closes at {speed} m/s"
    );

    let curve = RectilinearEncounter::from_geometry(&geometry).unwrap();
    assert!(
        curve.duration() < 0.5,
        "the encounter lasts {} s",
        curve.duration()
    );

    let violations = usage_violations(&geometry, period(lower), |t| {
        let (position, velocity) = pair(lower, upper, FRAC_PI_2, t);
        RelativeSample {
            position,
            velocity,
            covariance,
        }
    })
    .unwrap();

    assert!(
        !violations.any(),
        "a 10.7 km/s crossing was flagged: V_o = {:e}, V_e = {:e}, V_i = {:e}",
        violations.offset_indicator(),
        violations.extended_indicator(),
        violations.inaccuracy_indicator()
    );
    assert!(violations.extended_indicator() < 1.0e-4);
    assert!(violations.inaccuracy_indicator() < 1.0e-6);
    assert!((violations.correction_factor() - 1.0).abs() < 1.0e-6);
}

/// A co-orbiting pair: the same 300 m of radial separation, now in one plane. The two orbits differ
/// only in period, and the exact answer is known:
///
/// ```text
/// n(r) = √(μ/r³)   v(r) = √(μ/r)
/// along-track drift rate = √(r₁r₂)|n₁ − n₂|      inertial speed difference = |v₁ − v₂|
/// ratio = (a² + ab + b²)/(ab) = a/b + 1 + b/a    with a = r₁^{−1/2}, b = r₂^{−1/2}
/// ```
///
/// which is `3 + O((δ/r)²)` — for these orbits `3(1 + 1.5·10⁻¹⁰)`. Both widths are the same
/// covariance over one of those two rates, so the correction factor is the reciprocal of that ratio:
/// **`F_c = 1/3`**. The straight line through TCA separates three times too slowly, so it keeps the
/// pair inside the covariance three times too long, so the 2-D estimate is three times too large.
///
/// This is the file's real oracle: an exactly-known answer for a quantity the module obtains by
/// searching, on a trajectory that is a genuine two-body one. Measured: `F_c = 0.333333333334331`,
/// `3.0·10⁻¹²` relative from `1/3`, and `w` within `1.54·10⁻¹⁰` of `σ/(√(r₁r₂)|Δn|)` — which is
/// `Δθ²/24` at the width, i.e. the small-angle truncation of the comparison formula rather than
/// anything the search did. The two second-order terms have opposite signs and very nearly cancel in
/// the ratio, which is why `F_c` lands two decades closer than either width does.
#[test]
fn a_co_orbiting_pair_puts_the_correction_factor_at_exactly_one_third() {
    let lower = 7.0e6;
    let upper = lower + 300.0;
    let geometry = geometry_at_tca(lower, upper, 0.0);
    let covariance = isotropic(SIGMA * std::f64::consts::SQRT_2);
    let orbit = period(lower);

    let rate = |radius: f64| (MU / (radius * radius * radius)).sqrt();
    let speed = |radius: f64| (MU / radius).sqrt();
    let drift_rate = (lower * upper).sqrt() * (rate(lower) - rate(upper)).abs();
    let closing_rate = (speed(lower) - speed(upper)).abs();
    assert!(
        (closing_rate - 0.1617).abs() < 1.0e-3,
        "the pair closes at {closing_rate} m/s"
    );

    let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
    assert!(
        rectilinear.duration() > orbit,
        "the rectilinear encounter lasts {} s against an orbital period of {orbit} s",
        rectilinear.duration()
    );
    // w′ = σ / |Δv| for an isotropic covariance, and σ here is the joint one.
    let joint_sigma = SIGMA * std::f64::consts::SQRT_2;
    let expected_rectilinear_width = joint_sigma / closing_rate;
    assert!(
        (rectilinear.width() - expected_rectilinear_width).abs()
            < 1.0e-9 * expected_rectilinear_width,
        "w′ = {} against {expected_rectilinear_width}",
        rectilinear.width()
    );

    let curvilinear = CurvilinearEncounter::locate(&rectilinear, |t| {
        let (position, velocity) = pair(lower, upper, 0.0, t);
        RelativeSample {
            position,
            velocity,
            covariance,
        }
    })
    .unwrap();
    let expected_curvilinear_width = joint_sigma / drift_rate;
    let width_residual =
        (curvilinear.width() - expected_curvilinear_width).abs() / expected_curvilinear_width;
    assert!(
        width_residual < 1.0e-6,
        "w = {} against the closed form {expected_curvilinear_width}, residual {width_residual:e}",
        curvilinear.width()
    );

    let violations = UsageViolations::assess(&rectilinear, &curvilinear, orbit).unwrap();
    assert!(
        violations.extended(),
        "V_e = {} against a limit of {EXTENDED_LIMIT}",
        violations.extended_indicator()
    );
    let residual = (violations.correction_factor() - 1.0 / 3.0).abs() * 3.0;
    assert!(
        residual < 1.0e-8,
        "F_c = {} against the closed-form 1/3, relative residual {residual:e}",
        violations.correction_factor()
    );
    assert!(
        violations.inaccuracy(),
        "V_i = {}",
        violations.inaccuracy_indicator()
    );
    assert!(!violations.non_positive_definite());
}
