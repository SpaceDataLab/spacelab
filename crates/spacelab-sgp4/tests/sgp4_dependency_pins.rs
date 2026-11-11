//! What spacelab relies on inside its `sgp4` dependency, pinned so that a `cargo update` cannot move
//! it unseen (plans/17 Ф17.G).
//!
//! The Vallado vectors (`sgp4_verification.rs`, `sgp4_primary_source.rs`) and the resonance windows
//! (`satellite.rs`) are pinned elsewhere. Two details of the crate are not visible to them:
//!
//! 1. The deep-space Earth-rotation rate, `SIDEREAL_SPEED = 4.37526908801129966e-3` rad/min
//!    (`deep_space.rs:15`), Vallado's value. It enters only the resonance argument
//!    `λ = M + Ω + ω − θ`. Orekit's truncated `4.3752691e-3` moves λ by 0.52 µrad in 30 days, so no
//!    state comparison can see it. The constant is private, so the locked version is the pin: a bump
//!    fails [`the_locked_sgp4_is_the_audited_2_4_0`], and its message names what to re-read.
//! 2. The two near-Earth drag terms that exist only when `e₀ > 1e-4`, strictly
//!    (`near_earth.rs:128`). `derivations/sgp4_eccentricity_boundary.mac` derives their jump, and
//!    [`the_drag_terms_switch_on_strictly_above_e_1e_4`] measures it.
//!
//! The pin is a dependency decision, made here rather than as `sgp4 = "=2.4.0"` in `Cargo.toml`. An
//! exact requirement in a published library forces every downstream build onto 2.4.0 and conflicts
//! with any crate that needs a later 2.4.x. The lock file pins what this workspace builds and tests,
//! and a downstream build resolves the newest 2.4.x, as it does for every other dependency.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use core::f64::consts::PI;

#[test]
fn the_locked_sgp4_is_the_audited_2_4_0() {
    let lock = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../Cargo.lock"))
        .expect("the workspace's Cargo.lock");
    let versions: Vec<&str> = lock
        .split("[[package]]")
        .filter(|package| package.lines().any(|line| line.trim() == "name = \"sgp4\""))
        .filter_map(|package| {
            package
                .lines()
                .find_map(|line| line.trim().strip_prefix("version = \"")?.strip_suffix('"'))
        })
        .collect();
    assert_eq!(
        versions,
        ["2.4.0"],
        "sgp4 moved: re-read deep_space.rs (SIDEREAL_SPEED must stay Vallado's \
         4.37526908801129966e-3 rad/min) and near_earth.rs (the drag terms only for \
         eccentricity > 1.0e-4, strictly), rerun this file, then move the pin"
    );
}

/// TEME position and velocity, km and km/s, of one ISS-like element set with eccentricity `e`,
/// `t` minutes after its epoch, the elements of `derivations/sgp4_eccentricity_boundary.mac` (3).
fn state(e: f64, t: f64) -> ([f64; 3], [f64; 3]) {
    let deg = PI / 180.0;
    let orbit = sgp4::Orbit::from_kozai_elements(
        &sgp4::WGS84,
        51.6461 * deg,
        221.2784 * deg,
        e,
        89.1723 * deg,
        280.4612 * deg,
        15.495_078_96 * PI / 720.0,
    )
    .unwrap();
    // The epoch, in years from J2000, reaches only the deep-space branch.
    let constants = sgp4::Constants::new(
        sgp4::WGS84,
        sgp4::iau_epoch_to_sidereal_time,
        20.5,
        1.0e-4,
        orbit,
    )
    .unwrap();
    let prediction = constants.propagate(sgp4::MinutesSinceEpoch(t)).unwrap();
    (prediction.position, prediction.velocity)
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn unit(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

/// `a − 2b + c`: over three equally spaced states their second difference, and for a state `a`
/// after two others, the step to it less the linear trend of the two.
fn second_difference(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    [
        a[0] - 2.0 * b[0] + c[0],
        a[1] - 2.0 * b[1] + c[1],
        a[2] - 2.0 * b[2] + c[2],
    ]
}

/// Four element sets 1e-7 apart in eccentricity, three at or below the boundary and one above it.
///
/// Below it the state is smooth in `e`: the second difference of 0.998e-4, 0.999e-4 and 1.000e-4 is
/// rounding and `(Δe)² ∂²r/∂e²`, under a micrometre. Above it the two drag terms appear, and the step
/// from 1.000e-4 to 1.001e-4 less the linear trend below is the jump the derivation predicts: `a e δ
/// sin M` radially and `2 a e δ cos M` along the track, nothing across, with `δ` the terms' shift of
/// the mean anomaly. Its size is `a coef B*`, 15.7 cm here, whatever `e₀` is just above 1e-4.
///
/// The form is first order in the eccentricity vector SGP4 solves Kepler's equation with, whose
/// `J₃` part is 1e-3 here, and it leaves out the `J₂` short-period terms' response to the shift, also
/// 1e-3 of it. Measured, the jump is the derivation's within 1.4e-3 of its size at 720 min and 5.6e-4
/// at 1440 min, and the second difference below the boundary is 1e-10 to 3e-10 km; the bounds are
/// 5e-3 and 1e-9 km. If the boundary became `≥`, the jump would land between 0.999e-4 and 1.000e-4
/// and fail the smoothness check; if it moved off 1e-4 either way, the jump would vanish.
#[test]
fn the_drag_terms_switch_on_strictly_above_e_1e_4() {
    // (t min, radial km, along-track km), `derivations/sgp4_eccentricity_boundary.mac` (3) at 40 digits.
    let cases = [
        (720.0, -6.235_560_106_558_019e-5, -7.363_784_642_118_864e-4),
        (1440.0, 1.180_756_622_613_992e-4, -3.622_302_513_888_101e-5),
    ];
    for (t, radial, along) in cases {
        let position = |e: f64| state(e, t).0;
        let below = second_difference(position(1.0e-4), position(0.999e-4), position(0.998e-4));
        assert!(
            dot(below, below).sqrt() < 1e-9,
            "t = {t} min: the state is not smooth below e = 1e-4, second difference {:.3e} km",
            dot(below, below).sqrt()
        );

        let (r, v) = state(1.0e-4, t);
        let jump = second_difference(position(1.001e-4), r, position(0.999e-4));
        let radial_axis = unit(r);
        let normal = unit(cross(r, v));
        let along_axis = cross(normal, radial_axis);
        let got = [
            dot(jump, radial_axis),
            dot(jump, along_axis),
            dot(jump, normal),
        ];
        let miss = [got[0] - radial, got[1] - along, got[2]];
        let size = (radial * radial + along * along).sqrt();
        assert!(
            dot(miss, miss).sqrt() < 5e-3 * size,
            "t = {t} min: jump [{:.6e}, {:.6e}, {:.6e}] km against the derivation's \
             [{radial:.6e}, {along:.6e}, 0], {:.3e} of it",
            got[0],
            got[1],
            got[2],
            dot(miss, miss).sqrt() / size
        );
    }
}
