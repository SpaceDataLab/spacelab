//! Frame algebra, polar motion, and the velocity term everybody forgets.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::{Angle, Epoch, Mat3, Tt, Ut1, Vec3};
use spacelab_frames::{
    EARTH_ROTATION_RATE, Itrf, Orientation, Position, StateVector, Teme, Transform, pef_to_itrf,
    teme_to_itrf, teme_to_itrf_into, teme_to_pef, tio_locator,
};
use spacelab_testkit::nan_max;

const FIXTURES: &str = include_str!("fixtures/pom00_pyerfa.csv");

/// `(x_p, y_p, s′)` in radians: no polar motion, a nanoradian, and six rows of the observed range,
/// both signs of each coordinate among them.
#[expect(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    reason = "the fixture's doubles at 17 digits"
)]
const POM_INPUTS: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0000000000000001e-09, -1.0000000000000001e-09, 0.0],
    [
        5.7825367913444852e-07,
        1.7064712906508029e-06,
        -2.1068072122758105e-10,
    ],
    [
        2.5983462753026055e-06,
        2.2200475214688046e-06,
        1.6561135782611204e-10,
    ],
    [
        -2.491417404151455e-06,
        -2.482602812688112e-06,
        -2.9024768657039105e-11,
    ],
    [
        2.7542273578020037e-06,
        -1.9778436416473006e-06,
        -1.8828789429033793e-10,
    ],
    [
        -2.9410341917590808e-06,
        -1.0797117133062152e-06,
        1.9418241213306112e-10,
    ],
    [
        -8.332197551115247e-07,
        -2.7560448962887512e-06,
        -1.7561494781879532e-10,
    ],
];

/// `Wᵀ = R₁(−y_p)·R₂(−x_p)·R₃(s′)` at [`POM_INPUTS`] in 40-digit arithmetic, to 25 digits
/// (`derivations/gcrf_itrf_chain.mac` (7) reads this table back).
#[expect(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    reason = "25-digit literals written by the derivation"
)]
const POM_EXACT: [[[f64; 3]; 3]; 8] = [
    [
        [1.000000000000000000000000e0, 0.0, 0.0],
        [0.0, 1.000000000000000000000000e0, 0.0],
        [0.0, 0.0, 1.000000000000000000000000e0],
    ],
    [
        [
            9.999999999999999995000000e-1,
            0.0,
            1.000000000000000062114925e-9,
        ],
        [
            -1.000000000000000124229850e-18,
            9.999999999999999995000000e-1,
            1.000000000000000061614925e-9,
        ],
        [
            -1.000000000000000061614925e-9,
            -1.000000000000000062114925e-9,
            9.999999999999999990000000e-1,
        ],
    ],
    [
        [
            9.999999999998328113190906e-1,
            -2.106807212275458273365991e-10,
            5.782536791344162927486881e-7,
        ],
        [
            2.116674945294298986542275e-10,
            9.999999999985439778446916e-1,
            -1.706471290649689341032435e-6,
        ],
        [
            -5.782536787740537403123023e-7,
            1.706471290771801545813148e-6,
            9.999999999983767892083766e-1,
        ],
    ],
    [
        [
            9.999999999966242983030989e-1,
            1.656113578255529815511207e-10,
            2.598346275299681747977200e-6,
        ],
        [
            -1.598429056173118988114437e-10,
            9.999999999975356944884529e-1,
            -2.220047521459486766720192e-6,
        ],
        [
            -2.598346275660943713396230e-6,
            2.220047521036665330090979e-6,
            9.999999999941599928180319e-1,
        ],
    ],
    [
        [
            9.999999999968964196587260e-1,
            -2.902476865694902451956251e-11,
            -2.491417404148877569519371e-6,
        ],
        [
            3.520996851206342533243865e-11,
            9.999999999969183416366174e-1,
            2.482602812677856847398308e-6,
        ],
        [
            2.491417404069142899934133e-6,
            -2.482602812757874618464178e-6,
            9.999999999938147612963749e-1,
        ],
    ],
    [
        [
            9.999999999962071158130437e-1,
            -1.882878942896237746782015e-10,
            2.754227357798521522764460e-6,
        ],
        [
            1.828404632227003513518182e-10,
            9.999999999980440672478978e-1,
            1.977843641638509384793629e-6,
        ],
        [
            -2.754227358165538453786842e-6,
            -1.977843641127423447000471e-6,
            9.999999999942511830953756e-1,
        ],
    ],
    [
        [
            9.999999999956751589226017e-1,
            1.941824121322213101226695e-10,
            -2.941034191754840944268523e-6,
        ],
        [
            -1.910069430668767693919151e-10,
            9.999999999994171112898380e-1,
            1.079711713301335795061571e-6,
        ],
        [
            2.941034191962787673538215e-6,
            -1.079711712734908263069882e-6,
            9.999999999950922702495324e-1,
        ],
    ],
    [
        [
            9.999999999996528724044257e-1,
            -1.756149478187343612600351e-10,
            -8.332197551114282915279488e-7,
        ],
        [
            1.779113388716872633830751e-10,
            9.999999999962021082489992e-1,
            2.756044896284305423433063e-6,
        ],
        [
            8.332197546242611324461828e-7,
            -2.756044896431587966401134e-6,
            9.999999999958549806846701e-1,
        ],
    ],
];

/// The worst miss of the shipped matrix measured is 1.110e-16, one ulp of an entry just below 1.
/// The bound is four times that, the same four ulps `src/cio.rs` holds the CIO product to.
const POM_BOUND: f64 = 4.5e-16;

/// The largest element-wise gap between a matrix and an exact table entry.
fn gap(m: &Mat3, exact: &[[f64; 3]; 3]) -> f64 {
    let mut worst = 0.0_f64;
    for (row, exact_row) in m.rows.iter().zip(exact) {
        for (got, want) in row.iter().zip(exact_row) {
            worst = nan_max(worst, (got - want).abs());
        }
    }
    worst
}

/// The polar-motion matrix is `Wᵀ`, and none of the three orderings [`pef_to_itrf`] rejects. (G3)
///
/// The `pef_to_itrf` doc quotes these separations. Rebuilding all four candidates against `Wᵀ` in
/// exact arithmetic is what keeps that paragraph true: if the implementation drifted onto one of
/// the rejected forms, or if the rejected forms stopped being distinguishable, this fails. The last
/// one is the point of the test — right factors, right signs, product the other way round — because
/// rotations through microradians almost commute and the error is second order, so no tolerance set
/// by eye would ever catch it.
#[test]
fn the_polar_motion_matrix_is_w_transposed_and_no_other_ordering() {
    let (mut shipped, mut fwd, mut sign, mut rev, mut angle) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for (&[xp, yp, sp], exact) in POM_INPUTS.iter().zip(&POM_EXACT) {
        angle = nan_max(nan_max(angle, xp.abs()), yp.abs());
        let got = pef_to_itrf(
            Angle::from_radians(xp),
            Angle::from_radians(yp),
            Angle::from_radians(sp),
        )
        .matrix();
        shipped = nan_max(shipped, gap(&got, exact));
        let candidates = [
            // W itself, applied where its transpose belongs.
            Mat3::rot3(-sp) * Mat3::rot2(xp) * Mat3::rot1(yp),
            // The transposed factor order with the signs as the Conventions print them.
            Mat3::rot1(yp) * Mat3::rot2(xp) * Mat3::rot3(-sp),
            // The right factors and the right signs, multiplied the other way round.
            Mat3::rot3(sp) * Mat3::rot2(-xp) * Mat3::rot1(-yp),
        ];
        for (worst, m) in [&mut fwd, &mut sign, &mut rev].into_iter().zip(candidates) {
            *worst = nan_max(*worst, gap(&m, exact));
        }
    }
    println!(
        "Wᵀ {shipped:.2e}; rejected orderings: transpose {fwd:.2e}, signs {sign:.2e}, reversed \
         {rev:.2e}; largest |xp| or |yp| {angle:.2e} rad"
    );
    assert!(shipped < POM_BOUND, "Wᵀ misses by {shipped:.3e}");
    // First order in the angles: both land within a factor of two of `2 * angle`.
    for (name, miss) in [("transpose", fwd), ("signs", sign)] {
        assert!(
            miss > angle && miss < 4.0 * angle,
            "the {name} candidate misses by {miss:.2e}, not by first order in {angle:.2e}"
        );
    }
    // Second order: the commutator, which is why it is invisible and why it is here.
    assert!(
        rev > 1000.0 * POM_BOUND && rev < angle * angle,
        "the reversed product misses by {rev:.2e}, which is not the commutator"
    );
}

/// The polar-motion matrix against `pyerfa`'s `pom00`, element by element.
#[test]
#[ignore = "comparator: the distance to pyerfa, not a verdict"]
fn polar_motion_distance_to_pyerfa_pom00() {
    let mut distances = Vec::new();
    for line in FIXTURES.lines().filter(|l| !l.trim().is_empty()) {
        let v: Vec<f64> = line
            .split(',')
            .map(|t| t.trim().parse().expect("a float"))
            .collect();
        assert_eq!(v.len(), 12, "xp, yp, sp and nine matrix elements");
        let got = pef_to_itrf(
            Angle::from_radians(v[0]),
            Angle::from_radians(v[1]),
            Angle::from_radians(v[2]),
        )
        .matrix();
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                worst = nan_max(worst, (got.rows[i][j] - v[3 + i * 3 + j]).abs());
            }
        }
        distances.push(worst);
    }
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of a few dozen matrices"
    )]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | pyerfa | not recorded | the polar-motion matrix, worst entry | 1 | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}
#[test]
fn the_tio_locator_is_minus_47_microarcseconds_per_century() {
    let one_century = Epoch::<Tt>::from_days_since_j2000(36_525.0);
    assert!((tio_locator(one_century).arcseconds() - -47e-6).abs() < 1e-18);
    assert!(tio_locator(Epoch::<Tt>::J2000).arcseconds().abs() < 1e-30);
}

/// `R = pef_to_itrf(xp, yp, sp) · teme_to_pef(ut1, tt)` must stay orthonormal, and must stay a
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
    let r = teme_to_pef(
        Epoch::<Ut1>::from_days_since_j2000(77.7),
        Epoch::<Tt>::from_days_since_j2000(77.7),
    );
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
