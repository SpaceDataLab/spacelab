//! The reduction from a measured plate position to a true direction, and the epoch bookkeeping that
//! dominates it for a moving target.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    // The exact comparisons below are exact by construction: pixel indices, epoch nanoseconds and
    // catalogue magnitudes that were written as literals in this file.
    clippy::float_cmp,
    // Loop counters and pixel indices into f64: the frames here are at most 128 square.
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

use spacelab_astrometry::reduce::{
    Aberration, Astrometric, SPEED_OF_LIGHT, Streak, apply_refraction, remove_refraction,
};
use spacelab_astrometry::wcs::{Correspondence, Pixel, PlateSolution, deproject};
use spacelab_core::{Epoch, Tt, Vec3};
use spacelab_coverage::observability::{ApparentAltitude, Refraction, TrueAltitude};

const ARCSEC: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// A direction at altitude `h` and azimuth `a`, in a frame whose `z` is the zenith.
fn look(h: f64, a: f64) -> Vec3 {
    Vec3::new(h.cos() * a.cos(), h.cos() * a.sin(), h.sin())
}

fn angle_between(u: Vec3, v: Vec3) -> f64 {
    u.cross(v).norm().atan2(u.dot(v))
}

// ---------------------------------------------------------------------------------------------
// Aberration
// ---------------------------------------------------------------------------------------------

#[test]
fn the_constant_of_aberration_comes_out_of_the_earths_orbital_speed() {
    // κ = 2πa / (c T √(1 − e²)), the IAU definition. Constants at their primary precision:
    // a is the IAU 2012 astronomical unit (exact), T the sidereal year, e the mean eccentricity.
    let au = 1.495_978_707e11_f64;
    let sidereal_year = 365.256_363_004_f64 * 86_400.0;
    let eccentricity = 0.016_708_6_f64;
    let speed =
        core::f64::consts::TAU * au / (sidereal_year * (1.0 - eccentricity * eccentricity).sqrt());

    let aberration = Aberration::new(Vec3::new(speed, 0.0, 0.0)).expect("a sub-luminal velocity");
    let arcsec = aberration.constant() / ARCSEC;
    // The IAU constant of aberration is 20.49552″ and this chain gives 20.495514″, agreeing to
    // 6.3e-6 arcsec. The gate is 1e-5 — the last digit the IAU value is published to, and therefore
    // the most that can be asserted without asserting someone else's rounding rather than this
    // crate's arithmetic.
    assert!(
        (arcsec - 20.495_52).abs() < 1.0e-5,
        "constant of aberration {arcsec} arcsec"
    );
    assert!((aberration.beta().x - speed / SPEED_OF_LIGHT).abs() < 1.0e-18);
}

#[test]
fn a_direction_square_to_the_apex_is_displaced_by_exactly_arctan_gamma_beta() {
    let speed = 29_785.9_f64;
    let velocity = Vec3::new(speed, 0.0, 0.0);
    let aberration = Aberration::new(velocity).expect("sub-luminal");
    let beta = speed / SPEED_OF_LIGHT;
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();

    // Perpendicular to the motion: the exact displacement is arctan(γβ), the first-order one is β.
    let natural = Vec3::new(0.0, 1.0, 0.0);
    let apparent = aberration
        .apply(natural)
        .expect("a unit direction aberrates");
    let displacement = angle_between(natural, apparent);
    assert!(
        (displacement - (gamma * beta).atan()).abs() < 1.0e-18,
        "displacement {displacement} against arctan(γβ) {}",
        (gamma * beta).atan()
    );
    // The relativistic correction to the classical β is real but well below any plate: 1.6e-7
    // arcsec here.
    let correction = (displacement - beta) / ARCSEC;
    assert!(
        correction > 0.0 && correction < 1.0e-6,
        "relativistic excess over the classical constant is {correction} arcsec"
    );

    // And the displacement is *toward* the apex, which is the sign every textbook figure shows and
    // the one an implementation gets backwards.
    let toward_apex = apparent.dot(Vec3::new(1.0, 0.0, 0.0));
    assert!(
        toward_apex > 0.0,
        "aberration must lean the direction into the velocity, got {toward_apex}"
    );
}

#[test]
fn a_direction_along_the_apex_is_not_displaced_at_all() {
    let aberration = Aberration::new(Vec3::new(0.0, 0.0, 29_785.9)).unwrap();
    for &along in &[Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0)] {
        let apparent = aberration.apply(along).unwrap();
        assert!(
            angle_between(along, apparent) < 1.0e-15,
            "a direction parallel to the motion has no transverse component to aberrate"
        );
    }
}

#[test]
fn removing_aberration_is_the_exact_inverse_of_applying_it() {
    // The map is the action of a Lorentz boost on the sphere, so the inverse is the boost by −β and
    // nothing is iterated. A round trip that only closed to the iteration tolerance would be the
    // signature of the other implementation.
    let aberration = Aberration::new(Vec3::new(12_000.0, -25_000.0, 3_000.0)).unwrap();
    for &direction in &[
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(0.3, -0.5, 0.81).scale(1.0 / Vec3::new(0.3, -0.5, 0.81).norm()),
        Vec3::new(-0.6, -0.6, -0.52915).scale(1.0 / Vec3::new(-0.6, -0.6, -0.52915).norm()),
    ] {
        let apparent = aberration.apply(direction).unwrap();
        let back = aberration.remove(apparent).unwrap();
        assert!(
            angle_between(direction, back) < 1.0e-15,
            "round trip closed to {} rad",
            angle_between(direction, back)
        );
    }
}

#[test]
fn a_luminal_or_non_finite_velocity_has_no_aberration() {
    assert!(Aberration::new(Vec3::new(SPEED_OF_LIGHT, 0.0, 0.0)).is_none());
    assert!(Aberration::new(Vec3::new(2.0 * SPEED_OF_LIGHT, 0.0, 0.0)).is_none());
    assert!(Aberration::new(Vec3::new(f64::NAN, 0.0, 0.0)).is_none());
    assert!(Aberration::new(Vec3::new(f64::INFINITY, 0.0, 0.0)).is_none());
    // A zero velocity is legitimate and is the identity.
    let rest = Aberration::new(Vec3::ZERO).expect("an observer at rest is an observer");
    let direction = Vec3::new(0.0, 1.0, 0.0);
    assert_eq!(rest.apply(direction).unwrap(), direction);
    // And a zero-length direction has no direction to aberrate.
    assert!(rest.apply(Vec3::ZERO).is_none());
}

// ---------------------------------------------------------------------------------------------
// Refraction
// ---------------------------------------------------------------------------------------------

#[test]
fn removing_refraction_lowers_the_direction_by_the_published_bend() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    let apparent_altitude = 45.0_f64.to_radians();
    let azimuth = 1.234_f64;
    let apparent = look(apparent_altitude, azimuth);

    let unbent = remove_refraction(apparent, up, Refraction::STANDARD).expect("above the horizon");
    let moved = angle_between(apparent, unbent) / ARCSEC;

    // Bennett (1982) at 45°: R = 1/tan(45° + 7.31/49.4°) arcmin = 59.69″.
    assert!(
        (moved - 59.691).abs() < 0.01,
        "removed {moved} arcsec of refraction at 45 degrees"
    );

    // Downward, not upward: the true object is lower than it looks.
    assert!(
        unbent.z < apparent.z,
        "unrefracted z {} against apparent {}",
        unbent.z,
        apparent.z
    );
    // And it moved in the vertical plane only — the azimuth is untouched.
    assert!(
        (unbent.y.atan2(unbent.x) - azimuth).abs() < 1.0e-15,
        "azimuth changed from {azimuth} to {}",
        unbent.y.atan2(unbent.x)
    );
    // Which is the same as saying the direction stayed unit length.
    assert!((unbent.norm() - 1.0).abs() < 1.0e-15);

    // The removed angle is exactly what the coverage crate's own fit says, evaluated independently.
    let via_fit = ApparentAltitude(apparent_altitude).0
        - Refraction::STANDARD
            .geometric(ApparentAltitude(apparent_altitude))
            .unwrap()
            .0;
    assert!((moved * ARCSEC - via_fit).abs() < 1.0e-15);
}

#[test]
fn refraction_grows_toward_the_horizon_and_vanishes_at_the_zenith() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    let bend = |degrees: f64| -> f64 {
        let direction = look(degrees.to_radians(), 0.5);
        let unbent = remove_refraction(direction, up, Refraction::STANDARD).unwrap();
        angle_between(direction, unbent) / ARCSEC
    };
    let mut previous = f64::INFINITY;
    for degrees in [1.0, 5.0, 10.0, 20.0, 45.0, 70.0, 89.0] {
        let current = bend(degrees);
        assert!(
            current < previous,
            "refraction at {degrees}° is {current} arcsec, not below the {previous} above it"
        );
        previous = current;
    }
    // At the horizon it is the classical half-degree-ish: Bennett gives 34.5′ at h = 0.
    assert!(
        (bend(0.0) / 60.0 - 34.5).abs() < 0.3,
        "horizon refraction {} arcmin",
        bend(0.0) / 60.0
    );

    // Exactly at the zenith the direction comes back untouched, bit for bit — see the function's
    // own documentation for why passing the fits' residual through would be worse.
    let zenith = Vec3::new(0.0, 0.0, 1.0);
    assert_eq!(
        remove_refraction(zenith, up, Refraction::STANDARD).unwrap(),
        zenith
    );
    assert_eq!(
        apply_refraction(zenith, up, Refraction::STANDARD).unwrap(),
        zenith
    );
}

#[test]
fn applying_and_removing_refraction_close_to_the_disagreement_between_the_two_fits() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    // Sæmundsson is fitted in the true altitude and Bennett in the apparent one; they are not each
    // other's inverse, and the residual of a round trip is exactly
    //
    //     R_Sæmundsson(h) − R_Bennett(h + R_Sæmundsson(h)),
    //
    // which this computes independently from the coverage crate's own fits. Asserting *that*
    // rather than a tolerance is what makes the test say something: it fails if the rotation is
    // applied in the wrong plane or by the wrong amount, and it does not fail merely because the
    // two published fits disagree, which they are entitled to do.
    for degrees in [80.0_f64, 45.0, 20.0, 10.0, 2.0] {
        let radians = degrees.to_radians();
        let direction = look(radians, -0.9);
        let round_trip = remove_refraction(
            apply_refraction(direction, up, Refraction::STANDARD).unwrap(),
            up,
            Refraction::STANDARD,
        )
        .unwrap();
        let measured = round_trip.z.asin() - radians;

        let bent = Refraction::STANDARD
            .apparent(TrueAltitude(radians))
            .unwrap();
        let predicted = Refraction::STANDARD
            .geometric(ApparentAltitude(bent.0))
            .unwrap()
            .0
            - radians;
        assert!(
            (measured - predicted).abs() < 1.0e-15,
            "at {degrees}° the round trip left {} arcsec against the fits' own {} arcsec",
            measured / ARCSEC,
            predicted / ARCSEC
        );
    }

    // And the size of that disagreement, signed, so the test records what it actually is. It is
    // **not monotone in altitude**, which is the natural guess and is wrong: measured, the residual
    // is +0.175″ at 80°, rises to a maximum of +3.69″ near 10°, and then changes sign twice on the
    // way down — −1.55″ at 2°, +3.42″ at the horizon. Both fits were tuned to the classical horizon
    // refraction, so they cross rather than diverge there. Anything that treated the low-altitude
    // residual as a bound derived from the high-altitude one would be wrong by a factor of 20.
    let residual = |degrees: f64| -> f64 {
        let radians = degrees.to_radians();
        let direction = look(radians, -0.9);
        let round_trip = remove_refraction(
            apply_refraction(direction, up, Refraction::STANDARD).unwrap(),
            up,
            Refraction::STANDARD,
        )
        .unwrap();
        (round_trip.z.asin() - radians) / ARCSEC
    };
    for (degrees, expected) in [(80.0, 0.175), (10.0, 3.687), (2.0, -1.549), (0.0, 3.417)] {
        assert!(
            (residual(degrees) - expected).abs() < 0.01,
            "residual at {degrees}° is {} arcsec, not {expected}",
            residual(degrees)
        );
    }
}

#[test]
fn refraction_scales_with_the_air_it_passes_through() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    let direction = look(30.0_f64.to_radians(), 0.0);
    let bend = |refraction: Refraction| {
        angle_between(
            direction,
            remove_refraction(direction, up, refraction).unwrap(),
        )
    };
    let dense = bend(Refraction::STANDARD);
    // Half the pressure, half the refraction; twice the temperature, half again.
    let thin = bend(Refraction {
        pressure_hpa: Refraction::STANDARD.pressure_hpa / 2.0,
        temperature_k: Refraction::STANDARD.temperature_k,
    });
    let hot = bend(Refraction {
        pressure_hpa: Refraction::STANDARD.pressure_hpa,
        temperature_k: 2.0 * Refraction::STANDARD.temperature_k,
    });
    assert!((thin / dense - 0.5).abs() < 1.0e-9, "pressure scaling");
    assert!((hot / dense - 0.5).abs() < 1.0e-9, "temperature scaling");
}

#[test]
fn a_direction_below_the_horizon_has_no_ray_to_unbend() {
    let up = Vec3::new(0.0, 0.0, 1.0);
    assert!(remove_refraction(look(-0.1, 0.0), up, Refraction::STANDARD).is_none());
    // The true-altitude fit reaches one degree below, where the geometric ray still clears the
    // limb; below that both are extrapolating.
    assert!(apply_refraction(look(-0.5_f64.to_radians(), 0.0), up, Refraction::STANDARD).is_some());
    assert!(apply_refraction(look(-0.1, 0.0), up, Refraction::STANDARD).is_none());
    // Degenerate inputs are rejected rather than producing a NaN direction.
    assert!(remove_refraction(Vec3::ZERO, up, Refraction::STANDARD).is_none());
    assert!(remove_refraction(look(0.5, 0.0), Vec3::ZERO, Refraction::STANDARD).is_none());
}

#[test]
fn the_two_altitude_conventions_do_not_mix() {
    // A compile-level point made at run time: `TrueAltitude` goes into `apparent` and
    // `ApparentAltitude` into `geometric`, and swapping them is a type error rather than a few
    // arcseconds of silent bias. What is left to check is that they disagree at all.
    let h = 20.0_f64.to_radians();
    let up_the_fit = Refraction::STANDARD.apparent(TrueAltitude(h)).unwrap().0 - h;
    let down_the_fit = h - Refraction::STANDARD
        .geometric(ApparentAltitude(h))
        .unwrap()
        .0;
    assert!(
        (up_the_fit - down_the_fit).abs() / ARCSEC > 0.1,
        "the two fits differ by {} arcsec at 20 degrees, which is the reason both exist",
        (up_the_fit - down_the_fit).abs() / ARCSEC
    );
}

// ---------------------------------------------------------------------------------------------
// Epochs and the reduced observation
// ---------------------------------------------------------------------------------------------

const NS_PER_S: i128 = 1_000_000_000;

#[test]
fn a_streak_keeps_each_endpoint_with_its_own_epoch() {
    let open = Epoch::<Tt>::from_ns_since_j2000(1_000 * NS_PER_S);
    let close = Epoch::<Tt>::from_ns_since_j2000(1_010 * NS_PER_S);
    let streak = Streak::new(
        Pixel::new(100.0, 200.0),
        open,
        Pixel::new(340.0, 260.0),
        close,
    )
    .expect("an exposure that runs forward");

    assert!((streak.duration() - 10.0).abs() < 1.0e-12);

    let [(start, start_epoch), (end, end_epoch)] = streak.endpoints();
    assert_eq!(start_epoch, open);
    assert_eq!(end_epoch, close);
    assert_eq!(start.x, 100.0);
    assert_eq!(end.x, 340.0);

    let (mid, mid_epoch) = streak.midpoint();
    assert_eq!(mid_epoch.ns_since_j2000(), 1_005 * NS_PER_S);
    assert!((mid.x - 220.0).abs() < 1.0e-12 && (mid.y - 230.0).abs() < 1.0e-12);

    // 240 px in 10 s along x, 60 px along y.
    let (rate_x, rate_y) = streak.rate();
    assert!((rate_x - 24.0).abs() < 1.0e-12 && (rate_y - 6.0).abs() < 1.0e-12);

    // Reporting the midpoint against the shutter-open epoch is the error this type exists to make
    // impossible, and here is its size: half an exposure of motion.
    let half_exposure_pixels = rate_x * streak.duration() / 2.0;
    assert!((half_exposure_pixels - 120.0).abs() < 1.0e-12);
}

#[test]
fn an_exposure_that_does_not_run_forward_is_not_a_streak() {
    let open = Epoch::<Tt>::from_ns_since_j2000(0);
    let same = Epoch::<Tt>::from_ns_since_j2000(0);
    let earlier = Epoch::<Tt>::from_ns_since_j2000(-NS_PER_S);
    let pixel = Pixel::new(0.0, 0.0);
    assert!(Streak::new(pixel, open, pixel, same).is_none());
    assert!(Streak::new(pixel, open, pixel, earlier).is_none());
    // One nanosecond is a legitimate exposure; the guard is on the ordering, not on a duration
    // somebody thought was too short.
    assert!(Streak::new(pixel, open, pixel, Epoch::<Tt>::from_ns_since_j2000(1)).is_some());
}

/// A trivial exact plate: one arcsecond per pixel, no rotation, no distortion.
fn unit_plate() -> PlateSolution {
    let (ra0, dec0) = (1.0_f64, 0.4_f64);
    let crpix = Pixel::new(0.0, 0.0);
    let scale = ARCSEC;
    let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let correspondences: Vec<Correspondence> = (0..40)
        .map(|i| {
            let radius = 500.0 * ((i as f64 + 0.5) / 40.0).sqrt();
            let theta = i as f64 * golden;
            let (u, v) = (radius * theta.cos(), radius * theta.sin());
            let (ra, dec) = deproject(scale * u, scale * v, ra0, dec0);
            Correspondence {
                pixel: Pixel::new(u, v),
                right_ascension: ra,
                declination: dec,
            }
        })
        .collect();
    PlateSolution::fit(&correspondences, ra0, dec0, crpix, 1).expect("a well-posed plate")
}

#[test]
fn the_reduced_position_carries_the_centroid_error_as_well_as_the_plates() {
    let solution = unit_plate();
    let epoch = Epoch::<Tt>::from_ns_since_j2000(42 * NS_PER_S);
    let pixel = Pixel::new(120.0, -80.0);

    let plate_only = Astrometric::from_plate(&solution, pixel, epoch, 0.0);
    let with_centroid = Astrometric::from_plate(&solution, pixel, epoch, 0.3);
    let twice = Astrometric::from_plate(&solution, pixel, epoch, 0.6);

    // The plate was fitted to noiseless correspondences, so its own contribution is rounding and
    // the whole of the reported error is the centroid — 0.3 px at 1 arcsec/px.
    assert!(
        (with_centroid.covariance[0][0].sqrt() / ARCSEC - 0.3).abs() < 1.0e-6,
        "reported sigma {} arcsec for a 0.3 px centroid",
        with_centroid.covariance[0][0].sqrt() / ARCSEC
    );
    // Twice the centroid error, four times the variance — and the plate term, being zero here, does
    // not dilute the ratio.
    let added_once = with_centroid.covariance[0][0] - plate_only.covariance[0][0];
    let added_twice = twice.covariance[0][0] - plate_only.covariance[0][0];
    assert!(
        (added_twice / added_once - 4.0).abs() < 1.0e-9,
        "variance ratio {}",
        added_twice / added_once
    );

    // Both coordinates carry it, and the trace is the error circle.
    assert!((with_centroid.covariance[1][1] - with_centroid.covariance[0][0]).abs() < 1.0e-30);
    assert!(
        (with_centroid.position_sigma() / ARCSEC - 0.3 * 2.0_f64.sqrt()).abs() < 1.0e-6,
        "position sigma {} arcsec",
        with_centroid.position_sigma() / ARCSEC
    );

    // The epoch is carried, not invented.
    assert_eq!(with_centroid.epoch, epoch);
    // And the direction is the plate's own answer for that pixel.
    let (ra, dec) = solution.wcs.pixel_to_sky(pixel);
    assert!((with_centroid.right_ascension - ra).abs() < 1.0e-15);
    assert!((with_centroid.declination - dec).abs() < 1.0e-15);
    assert!((with_centroid.direction().norm() - 1.0).abs() < 1.0e-15);
}

#[test]
fn the_full_reduction_moves_the_direction_by_refraction_plus_aberration_and_keeps_the_rest() {
    let solution = unit_plate();
    let epoch = Epoch::<Tt>::from_ns_since_j2000(7 * NS_PER_S);
    let measured = Astrometric::from_plate(&solution, Pixel::new(50.0, 50.0), epoch, 0.2);

    // A local vertical 40° from the measured direction, so the object sits at altitude 50°.
    let apparent = measured.direction();
    let sideways = apparent.cross(Vec3::new(0.0, 0.0, 1.0));
    let axis = sideways.scale(1.0 / sideways.norm());
    let tilt = 40.0_f64.to_radians();
    let up = apparent.scale(tilt.cos()) + axis.cross(apparent).scale(tilt.sin());

    let aberration = Aberration::new(Vec3::new(0.0, 29_785.9, 0.0)).unwrap();
    let reduced = measured
        .to_true_direction(up, Refraction::STANDARD, &aberration)
        .expect("a target at 50 degrees altitude reduces");

    let moved = angle_between(apparent, reduced.direction()) / ARCSEC;
    // Refraction at 50° is 50.1″ (Bennett) and the aberration is at most 20.5″, so the total is
    // bounded by their sum and is at least their difference — a band that a dropped correction
    // falls out of at either end.
    assert!(
        (28.0..70.0).contains(&moved),
        "the reduction moved the direction {moved} arcsec"
    );

    // Everything but the direction survives untouched — that is what `with_direction` promises.
    assert_eq!(reduced.epoch, measured.epoch);
    assert_eq!(reduced.covariance, measured.covariance);

    // Right ascension comes back wrapped into a usable range and the declination is a declination.
    assert!(reduced.declination.abs() <= core::f64::consts::FRAC_PI_2);

    // Undoing the two corrections in the opposite order returns the measured direction, to the
    // 0.93″ by which Sæmundsson and Bennett disagree at 50° altitude. The aberration leg is exact
    // and contributes nothing here; the whole residual is the two published fits, and the previous
    // test pins that identity to 1e-15 rather than to a tolerance.
    let back = apply_refraction(
        aberration.apply(reduced.direction()).unwrap(),
        up,
        Refraction::STANDARD,
    )
    .unwrap();
    assert!(
        angle_between(apparent, back) / ARCSEC < 1.0,
        "the inverse reduction closed to {} arcsec",
        angle_between(apparent, back) / ARCSEC
    );

    // Below the horizon there is nothing to reduce.
    assert!(
        measured
            .to_true_direction(-up, Refraction::STANDARD, &aberration)
            .is_none()
    );
}
