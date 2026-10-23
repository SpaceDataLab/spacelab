//! Frame algebra, polar motion, and the velocity term everybody forgets.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::{Angle, Epoch, Mat3, Tt, Ut1, Vec3};
use spacelab_frames::{
    EARTH_ROTATION_RATE, Itrf, Orientation, Position, StateVector, Teme, Transform, pef_to_itrf,
    teme_to_itrf, teme_to_itrf_into, teme_to_pef, tio_locator,
};
use spacelab_testkit::nan_max;

const FIXTURES: &str = include_str!("fixtures/pom00_pyerfa.csv");

/// The polar-motion matrix, element by element, against the oracle.
#[test]
fn polar_motion_matches_pyerfa_pom00() {
    let mut checked = 0;
    let mut worst = 0.0_f64;

    for line in FIXTURES.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = line
            .split(',')
            .map(|t| t.trim().parse().expect("a float"))
            .collect();
        assert_eq!(v.len(), 12, "xp, yp, sp and nine matrix elements");

        let (xp, yp, sp) = (
            Angle::from_radians(v[0]),
            Angle::from_radians(v[1]),
            Angle::from_radians(v[2]),
        );
        let got = pef_to_itrf(xp, yp, sp).matrix();

        for i in 0..3 {
            for j in 0..3 {
                let delta = (got.rows[i][j] - v[3 + i * 3 + j]).abs();
                worst = nan_max(worst, delta);
                assert!(delta < 1e-15, "element ({i},{j}) differs by {delta:.3e}");
            }
        }
        checked += 1;
    }

    assert!(checked >= 30);
    println!("pom00: {checked} matrices, worst element delta {worst:.3e}");
}

/// The three orderings [`pef_to_itrf`] rejects, and how far each of them misses.
///
/// The `pef_to_itrf` doc quotes these separations. Rebuilding all four candidates here
/// is what keeps that paragraph true: if the implementation drifted onto one of the
/// rejected forms, or if the rejected forms stopped being distinguishable, this fails.
/// The last one is the point of the test — right factors, right signs, product the
/// other way round — because rotations through microradians almost commute and the
/// error is second order, so no tolerance set by eye would ever catch it.
#[test]
fn polar_motion_rejects_the_other_three_orderings() {
    let (mut fwd, mut sign, mut rev, mut angle) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);

    for line in FIXTURES.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = line
            .split(',')
            .map(|t| t.trim().parse().expect("a float"))
            .collect();
        let (xp, yp, sp) = (v[0], v[1], v[2]);
        angle = nan_max(nan_max(angle, xp.abs()), yp.abs());

        let candidates = [
            // W itself, applied where its transpose belongs.
            Mat3::rot3(-sp) * Mat3::rot2(xp) * Mat3::rot1(yp),
            // The transposed factor order with the signs as the Conventions print them.
            Mat3::rot1(yp) * Mat3::rot2(xp) * Mat3::rot3(-sp),
            // The right factors and the right signs, multiplied the other way round.
            Mat3::rot3(sp) * Mat3::rot2(-xp) * Mat3::rot1(-yp),
        ];
        for (worst, m) in [&mut fwd, &mut sign, &mut rev].into_iter().zip(candidates) {
            for i in 0..3 {
                for j in 0..3 {
                    *worst = worst.max((m.rows[i][j] - v[3 + i * 3 + j]).abs());
                }
            }
        }
    }

    // First order in the angles: both land within a factor of two of `2 * angle`.
    for (name, miss) in [("transpose", fwd), ("signs", sign)] {
        assert!(
            miss > angle && miss < 4.0 * angle,
            "the {name} candidate misses by {miss:.2e}, not by first order in {angle:.2e}"
        );
    }
    // Second order: the commutator, which is why it is invisible and why it is here.
    assert!(
        rev > 0.0 && rev < angle * angle,
        "the reversed product misses by {rev:.2e}, which is not the commutator"
    );
    println!(
        "rejected orderings: transpose {fwd:.2e}, signs {sign:.2e}, reversed {rev:.2e}; \
         largest |xp| or |yp| {angle:.2e} rad"
    );
}

#[test]
fn the_tio_locator_is_minus_47_microarcseconds_per_century() {
    let one_century = Epoch::<Tt>::from_days_since_j2000(36_525.0);
    assert!((tio_locator(one_century).arcseconds() - -47e-6).abs() < 1e-18);
    assert!(tio_locator(Epoch::<Tt>::J2000).arcseconds().abs() < 1e-30);
}

/// `R = pef_to_itrf(xp, yp, sp) · teme_to_pef(ut1)` must stay orthonormal, and must stay a
/// *rotation* rather than a reflection. `det R = 1` is necessary and not sufficient: a matrix can
/// have unit determinant and not be orthogonal, so `R Rᵀ = I` is asserted too.
///
/// # The tolerance is an error model, not a number that made the test pass
///
/// Every entry of `R` has magnitude at most 1 and carries the rounding of a `sin` or `cos`
/// (≤ 1 ulp) plus the roundings of the two 3×3 products that compose it. Each product costs three
/// multiplies and two adds per entry, so an entry of `R` is in error by at most about `16·eps`.
/// Forming `(R Rᵀ)ᵢⱼ` is another three multiplies and two adds over entries of that size, giving
/// about `100·eps`. Round the budgets up to `128·eps` and `64·eps`.
///
/// Swept over a century of epochs and the full observed range of polar motion (`|xp|, |yp| < 0.6″`,
/// 33 969 rotations), the worst deviations are **4.44e-16** and **5.55e-16** — two ulp, and 1.6 %
/// and 3.9 % of the budgets. That is 2.8 nm at the equator. If a change pushes either number to a
/// large fraction of its budget, the composition has become numerically worse and the tolerance
/// must not be widened to accommodate it.
#[test]
fn every_frame_rotation_is_a_proper_rotation_over_a_century_and_all_polar_motion() {
    let budget_orth = 128.0 * f64::EPSILON;
    let budget_det = 64.0 * f64::EPSILON;
    let (mut worst_orth, mut worst_det) = (0.0_f64, 0.0_f64);

    for d in 0..=200 {
        // 1950 through 2050, every half year
        let day = -18_262.0 + f64::from(d) * 182.6;
        for k in 0..13 {
            for l in 0..13 {
                let xp = -0.6 + f64::from(k) * 0.1;
                let yp = -0.6 + f64::from(l) * 0.1;
                let t = teme_to_itrf(
                    Epoch::<Ut1>::from_days_since_j2000(day),
                    Epoch::<Tt>::from_days_since_j2000(day),
                    Angle::from_arcseconds(xp),
                    Angle::from_arcseconds(yp),
                    0.0,
                );
                let m = t.rotation.matrix();
                let gram = m * m.transpose();
                for i in 0..3 {
                    for j in 0..3 {
                        let want = f64::from(u8::from(i == j));
                        worst_orth = nan_max(worst_orth, (gram.rows[i][j] - want).abs());
                    }
                }
                worst_det = nan_max(worst_det, (m.determinant() - 1.0).abs());
            }
        }
    }

    assert!(
        worst_orth < budget_orth,
        "worst |R Rᵀ - I| = {worst_orth:.3e} exceeds the derived budget {budget_orth:.3e}"
    );
    assert!(
        worst_det < budget_det,
        "worst |det R - 1| = {worst_det:.3e} exceeds the derived budget {budget_det:.3e}"
    );
}

/// The test that cannot pass by accident.
///
/// A point bolted to the equator has zero velocity in the ITRF and 465 m/s in any
/// frame that does not rotate with the Earth. An implementation that rotates the
/// velocity vector without the `ω × r` term returns zero, and the resulting orbit
/// still looks entirely plausible.
#[test]
fn a_point_at_rest_on_the_equator_moves_at_465_metres_per_second() {
    const EQUATORIAL_RADIUS: f64 = 6_378_137.0;

    let ut1 = Epoch::<Ut1>::from_days_since_j2000(1_234.5);
    let tt = Epoch::<Tt>::from_days_since_j2000(1_234.5);
    let to_teme = teme_to_itrf(
        ut1,
        tt,
        Angle::from_arcseconds(0.05),
        Angle::from_arcseconds(0.4),
        0.0,
    )
    .inverse();

    let at_rest = StateVector::<Itrf>::new(Vec3::new(EQUATORIAL_RADIUS, 0.0, 0.0), Vec3::ZERO);

    let moving = to_teme.state(at_rest);
    let speed = moving.velocity.vector.norm();
    let expected = EARTH_ROTATION_RATE * EQUATORIAL_RADIUS;

    assert!(
        (speed - expected).abs() < 1e-3,
        "got {speed:.4} m/s, expected {expected:.4} m/s — the ω × r term is missing"
    );
    assert!((465.0 - speed).abs() < 0.2, "sanity: {speed:.2} m/s");
}

#[test]
fn the_pole_is_left_alone_by_sidereal_rotation() {
    // teme_to_pef turns about ẑ, so ẑ is its fixed axis. This is what makes the
    // angular velocity vector the same in both frames, and hence the velocity
    // transform correct at the PEF stage.
    let r = teme_to_pef(Epoch::<Ut1>::from_days_since_j2000(77.7));
    let pole = r * Position::<Teme>::new(Vec3::new(0.0, 0.0, 1.0));
    assert!((pole.vector.z - 1.0).abs() < 1e-15);
    assert!(pole.vector.x.abs() < 1e-15 && pole.vector.y.abs() < 1e-15);
}

#[test]
fn a_transform_and_its_inverse_return_the_state_untouched() {
    let ut1 = Epoch::<Ut1>::from_days_since_j2000(-812.3);
    let tt = Epoch::<Tt>::from_days_since_j2000(-812.3);
    let t = teme_to_itrf(
        ut1,
        tt,
        Angle::from_arcseconds(-0.2),
        Angle::from_arcseconds(0.11),
        0.0,
    );

    let original = StateVector::<Teme>::new(
        Vec3::new(4_100_000.0, -5_200_000.0, 1_900_000.0),
        Vec3::new(3_100.0, 5_600.0, -2_200.0),
    );
    let back = t.inverse().state(t.state(original));

    let dr = (back.position.vector - original.position.vector).norm();
    let dv = (back.velocity.vector - original.velocity.vector).norm();
    assert!(dr < 1e-8, "position drifted {dr:.3e} m");
    assert!(dv < 1e-11, "velocity drifted {dv:.3e} m/s");
}

#[test]
fn the_batch_path_agrees_with_the_scalar_path() {
    let epochs: Vec<Orientation> = (0..1_000)
        .map(|k| {
            let d = f64::from(k) * 0.017;
            Orientation {
                ut1: Epoch::<Ut1>::from_days_since_j2000(d),
                tt: Epoch::<Tt>::from_days_since_j2000(d),
                xp: Angle::from_arcseconds(0.1 + d * 1e-5),
                yp: Angle::from_arcseconds(0.3 - d * 1e-5),
                // A realistic ~1 ms LOD, varied, so both paths must apply it identically.
                lod_seconds: 0.001 + d * 1e-6,
            }
        })
        .collect();

    let identity: Transform<Teme, Itrf> = teme_to_itrf(
        Epoch::J2000,
        Epoch::J2000,
        Angle::from_radians(0.0),
        Angle::from_radians(0.0),
        0.0,
    );
    let mut out = vec![identity; epochs.len()];
    teme_to_itrf_into(&epochs, &mut out);

    for (o, batched) in epochs.iter().zip(&out) {
        let scalar = teme_to_itrf(o.ut1, o.tt, o.xp, o.yp, o.lod_seconds);
        assert_eq!(*batched, scalar, "batch and scalar diverged");
    }
}

#[test]
#[should_panic(expected = "output slice holds")]
fn a_short_output_slice_is_refused_not_truncated() {
    let epochs = [Orientation {
        ut1: Epoch::J2000,
        tt: Epoch::J2000,
        xp: Angle::from_radians(0.0),
        yp: Angle::from_radians(0.0),
        lod_seconds: 0.0,
    }];
    let mut out: [Transform<Teme, Itrf>; 0] = [];
    teme_to_itrf_into(&epochs, &mut out);
}
