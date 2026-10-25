//! The lunar gravity field against a high-precision oracle, and against the two things that make it
//! a lunar field rather than a rescaled terrestrial one: the roughness of the near-surface potential,
//! and the frame the coefficients live in.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

use spacelab_propagation::forces::{Acceleration, BodyOriented};
use spacelab_propagation::lunar::{MOON_MU, Matrix3, embedded, me_to_pa, pa_to_me};
use spacelab_propagation::lunar_coeffs::MODEL_RE;

const ORACLE: &str = include_str!("fixtures/lunar_gravity_oracle.txt");
/// The degree the oracle was summed to. Comparing against a field truncated anywhere else is
/// comparing two different models, which is a much larger difference than the one being measured.
const ORACLE_DEGREE: usize = 20;

fn oracle_rows() -> Vec<([f64; 3], [f64; 3])> {
    ORACLE
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|token| token.parse().unwrap())
                .collect();
            ([v[0], v[1], v[2]], [v[3], v[4], v[5]])
        })
        .collect()
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn apply(m: &Matrix3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// The field against our own 60-digit Legendre summation of the same coefficients (S1).
#[test]
fn the_field_matches_a_sixty_digit_direct_summation_of_the_same_coefficients() {
    // The crate sums Gottlieb's solid harmonics and differentiates them in closed form; the oracle
    // sums the Legendre form at 60 decimal digits and differentiates numerically. Same coefficients,
    // different algebra — so the normalization ratios, the degree-mixing and the Cartesian gradient
    // are all being checked against something that shares none of them.
    let field = embedded(ORACLE_DEGREE).unwrap();
    let rows = oracle_rows();
    assert_eq!(rows.len(), 5, "the fixture lost rows");

    let mut worst: f64 = 0.0;
    for (position, want) in rows {
        let got = field.acceleration_f64(0.0, &position, &[0.0; 3]);
        let error = norm([got[0] - want[0], got[1] - want[1], got[2] - want[2]]);
        let relative = error / norm(want);
        println!(
            "r = {:.3e} m: |a| = {:.9e}, error {:.2e} m/s² ({relative:.2e} relative)",
            norm(position),
            norm(got),
            error
        );
        worst = worst.max(relative);
    }
    // The oracle's own floor is the central-difference step, about 1e-12 relative; anything the
    // recursion adds on top of that would have to be larger to be visible at all.
    println!("worst relative disagreement {worst:.3e}");
    assert!(worst < 1.0e-11, "worst relative disagreement {worst:.3e}");
}

#[test]
fn the_lunar_field_is_rough_where_the_terrestrial_one_is_smooth() {
    // The reason a lunar propagation needs degree and a terrestrial one at the same altitude does
    // not. Measured, not asserted from the literature: the non-spherical part of the acceleration
    // 100 km above the Moon is a few times 1e-3 of the total, where the Earth's J2-dominated field
    // at 100 km is about 1e-3 and falls away almost entirely by degree 8.
    let point_mass = embedded(0).unwrap();
    let full = embedded(20).unwrap();
    let surface = 1_837_400.0_f64; // 100 km above the mean radius

    let mut worst_fraction: f64 = 0.0;
    let mut worst_tail: f64 = 0.0;
    for (i, j) in [(0, 1), (1, 2), (2, 0), (0, 2)] {
        let mut position = [0.0; 3];
        position[i] = surface * 0.6;
        position[j] = surface * 0.8;
        let central = point_mass.acceleration_f64(0.0, &position, &[0.0; 3]);
        let complete = full.acceleration_f64(0.0, &position, &[0.0; 3]);
        let perturbation = norm([
            complete[0] - central[0],
            complete[1] - central[1],
            complete[2] - central[2],
        ]);
        worst_fraction = worst_fraction.max(perturbation / norm(central));

        // How much of that lives above degree 8 — the degree an Earth model of this altitude would
        // stop at. On the Moon it is not negligible, which is the whole point.
        let truncated = embedded(8)
            .unwrap()
            .acceleration_f64(0.0, &position, &[0.0; 3]);
        let tail = norm([
            complete[0] - truncated[0],
            complete[1] - truncated[1],
            complete[2] - truncated[2],
        ]);
        worst_tail = worst_tail.max(tail / norm(central));
    }
    println!(
        "100 km up: non-spherical part is {worst_fraction:.3e} of the central term, \
         of which {worst_tail:.3e} sits above degree 8"
    );
    assert!(
        worst_fraction > 5.0e-4,
        "the field looks smoother than the Moon is: {worst_fraction:.3e}"
    );
    // Measured: 6.09e-4 total, 4.07e-4 of it above degree 8 — **two thirds of the whole
    // non-spherical signal lives in degrees 9-20**. That is the number that separates the Moon from
    // the Earth, where the same split at the same altitude is dominated by J2 and the tail past
    // degree 8 is an order smaller than what is kept.
    assert!(
        worst_tail > 2.0e-4,
        "degrees 9-20 contribute {worst_tail:.3e}, which would make truncation at 8 free"
    );
}

#[test]
fn evaluating_the_field_in_the_wrong_lunar_frame_is_a_visible_error() {
    // The PA/ME distinction is a 0.03° rotation, and the temptation is to treat it as negligible.
    // It is not: 900 m of horizontal displacement across a field whose horizontal gradient is set by
    // the mascons moves the acceleration by far more than any integration tolerance, and the error
    // is a smooth bias rather than noise — the shape a wrong frame always takes.
    let field = embedded(20).unwrap();
    let surface = 1_837_400.0_f64;

    let mut worst: f64 = 0.0;
    for point in [
        [surface, 0.0, 0.0],
        [0.0, surface, 0.0],
        [surface * 0.5, surface * 0.5, surface * 0.707],
        [-surface * 0.6, surface * 0.3, -surface * 0.74],
    ] {
        let correct = field.acceleration_f64(0.0, &point, &[0.0; 3]);
        // What a caller does who hands in a selenographic (mean-Earth) position unconverted.
        let mistaken = field.acceleration_f64(0.0, &apply(&me_to_pa(), point), &[0.0; 3]);
        let error = norm([
            mistaken[0] - correct[0],
            mistaken[1] - correct[1],
            mistaken[2] - correct[2],
        ]);
        worst = worst.max(error);
    }
    println!("using the wrong lunar frame costs up to {worst:.3e} m/s²");
    // Measured at 7.3e-4 m/s², which is 5e-4 of the total acceleration — larger than the *entire*
    // non-spherical signal above degree 8 that the previous test measures at 4.1e-4. So a caller who
    // gets the frame wrong loses more than they would by dropping two thirds of the field, and loses
    // it as a smooth bias no integration tolerance reports.
    assert!(
        worst > 1.0e-5,
        "the frame error is invisible here, so this test proves nothing: {worst:.3e}"
    );
}

#[test]
fn the_body_fixed_wrapper_carries_a_librating_orientation_the_earth_form_cannot() {
    // The Moon's orientation is not `theta0 + omega*t` about a fixed axis: the physical libration in
    // longitude alone reaches 8°, and the pole wanders too. `BodyOriented` takes the whole matrix, so
    // a field evaluated through it follows the real orientation.
    //
    // The check is an invariant rather than a number: a rotation of the *body* must leave the
    // magnitude of the acceleration at the corresponding rotated point unchanged, whatever the
    // orientation. If the wrapper rotated in and out inconsistently, this would fail immediately —
    // and it is exactly the failure mode a symmetric test fixture hides.
    let field = embedded(20).unwrap();
    let position = [1.5e6, -6.0e5, 9.0e5];

    let libration = |t: f64| -> Matrix3 {
        // A stand-in for the DE440 libration: three angles at incommensurate rates, all of the size
        // the Moon's actually reaches, so no two axes ever line up.
        let (yaw, pitch, roll) = (
            0.14 * t.sin(),
            0.05 * (0.7 * t).cos(),
            0.02 * (1.3 * t).sin(),
        );
        // Z(yaw) then Y(pitch) then X(roll), as frame rotations.
        let about_z: Matrix3 = [
            [yaw.cos(), yaw.sin(), 0.0],
            [-yaw.sin(), yaw.cos(), 0.0],
            [0.0, 0.0, 1.0],
        ];
        let about_y: Matrix3 = [
            [pitch.cos(), 0.0, -pitch.sin()],
            [0.0, 1.0, 0.0],
            [pitch.sin(), 0.0, pitch.cos()],
        ];
        let about_x: Matrix3 = [
            [1.0, 0.0, 0.0],
            [0.0, roll.cos(), roll.sin()],
            [0.0, -roll.sin(), roll.cos()],
        ];
        let mut product = [[0.0; 3]; 3];
        for (i, row) in product.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3)
                    .map(|k| {
                        about_x[i][k] * (0..3).map(|l| about_y[k][l] * about_z[l][j]).sum::<f64>()
                    })
                    .sum();
            }
        }
        product
    };

    let oriented = BodyOriented {
        inner: embedded(20).unwrap(),
        rotate: libration,
    };

    let mut worst: f64 = 0.0;
    let mut spread: f64 = 0.0;
    let reference = norm(field.acceleration_f64(0.0, &position, &[0.0; 3]));
    for step in 0..12 {
        let t = f64::from(step) * 1.7;
        let m = libration(t);
        // Through the wrapper, at the inertial position.
        let via_wrapper = oriented.acceleration_f64(t, &position, &[0.0; 3]);
        // By hand: rotate in, evaluate, rotate back.
        let body = apply(&m, position);
        let inner = field.acceleration_f64(t, &body, &[0.0; 3]);
        let back = apply(&transpose(&m), inner);
        let disagreement = norm([
            via_wrapper[0] - back[0],
            via_wrapper[1] - back[1],
            via_wrapper[2] - back[2],
        ]);
        worst = worst.max(disagreement / norm(back));
        // And the orientation has to matter: the magnitude must move as the body turns beneath the
        // point, or the wrapper is silently discarding the matrix.
        spread = spread.max((norm(via_wrapper) - reference).abs() / reference);
    }
    println!("wrapper vs hand-rotation: {worst:.3e} relative; libration moves |a| by {spread:.3e}");
    assert!(worst < 1e-14, "the wrapper is not a rotate-in/rotate-out");
    assert!(
        spread > 1e-6,
        "the orientation changed nothing, so this proves nothing: {spread:.3e}"
    );
}

fn transpose(m: &Matrix3) -> Matrix3 {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

#[test]
fn the_moons_constants_are_the_models_own() {
    // The one way to get a lunar field subtly wrong that no gradient check would catch: right
    // coefficients, wrong GM or wrong reference radius. The published GM is 4902.800 km³/s² and the
    // expansion radius is 1738 km exactly.
    let field = embedded(4).unwrap();
    assert_eq!(field.mu(), MOON_MU);
    assert!((MOON_MU - 4.902_800_122_445_3e12).abs() < 1.0, "GM");
    assert_eq!(MODEL_RE, 1_738_000.0);

    // Far from the body the field has to become the point mass to the precision the higher degrees
    // can still contribute — (Re/r)² at 100 lunar radii is 1e-4, times J2's 2e-4, so 2e-8.
    let far = [1.738e8, 0.0, 0.0];
    let a = field.acceleration_f64(0.0, &far, &[0.0; 3]);
    let point = -MOON_MU / (far[0] * far[0]);
    println!("at 100 R: ax = {:.12e}, point mass {point:.12e}", a[0]);
    assert!(
        (a[0] / point - 1.0).abs() < 1.0e-7,
        "not point-mass at 100 R"
    );

    // pa_to_me and me_to_pa compose to the identity, which is what makes the "wrong frame" test
    // above a statement about the frames rather than about a broken matrix.
    let round = apply(&me_to_pa(), apply(&pa_to_me(), far));
    assert!((round[0] - far[0]).abs() < 1.0e-6);
}
