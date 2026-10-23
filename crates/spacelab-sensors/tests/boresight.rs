//! A cone about an explicit axis, and the geodetic nadir it exists to make expressible.
//!
//! `Cone` measured off-nadir from `−r/|r|` and nothing at the call site said which of the two
//! "downs" that was. The deflection from the geodetic normal a nadir-pointing attitude actually
//! holds is 0.192 424 3° at the surface and 0.177 125° at 550 km — 1.70 km on the ground straight
//! below — so it is a footprint error, not a rounding question.
//!
//! Three separate claims are tested here and they fail for different reasons:
//!
//! 1. the new code path reproduces the old one where they describe the same cone, so
//!    `Boresight::Fixed(−r/|r|)` is not merely close to `GeocentricNadir` but exactly it;
//! 2. the two paths' *scalar* and *batch* forms agree with each other, as `covered`/`covered_into`
//!    must, or an event edge and a coverage grid would disagree within an ulp and no test would say;
//! 3. the footprint edge moves by exactly the deflection — the number the whole ticket is about.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use spacelab_core::Vec3;
use spacelab_earth::{FLATTENING, Geodetic, geodetic_nadir, geodetic_to_itrf};
use spacelab_sensors::{
    AccessGeometry, Boresight, Cone, Constraint, SensorError,
    grid::CellGrid,
    kernel::{cone_covered_into, cone_covered_one},
};

fn geom(satellite: Vec3, cell: Vec3) -> AccessGeometry {
    AccessGeometry {
        satellite,
        cell,
        zenith: Vec3::ZERO, // the cone does not read it
    }
}

/// The geodetic point `(lat, lon, h)`, as the earth crate's constructors take it. The `Position<Itrf>`
/// they return is never named: `spacelab-sensors` depends on `spacelab-core` and `libm` and nothing
/// else, and a test that named `spacelab_frames` would be the first thing to widen that.
fn point(lat_deg: f64, lon_deg: f64, height_m: f64) -> Geodetic {
    Geodetic {
        latitude: spacelab_core::Angle::from_degrees(lat_deg),
        longitude: spacelab_core::Angle::from_degrees(lon_deg),
        height: spacelab_core::Length::from_metres(height_m),
    }
}

/// A satellite at `height` over that point.
fn over(lat_deg: f64, lon_deg: f64, height_m: f64) -> Vec3 {
    geodetic_to_itrf(point(lat_deg, lon_deg, height_m)).vector
}

/// The inward geodetic normal there — the axis a nadir-pointing attitude holds.
fn geodetic_axis(lat_deg: f64, lon_deg: f64, height_m: f64) -> Vec3 {
    geodetic_nadir(geodetic_to_itrf(point(lat_deg, lon_deg, height_m))).unwrap()
}

fn on_surface(lat_deg: f64, lon_deg: f64) -> Vec3 {
    over(lat_deg, lon_deg, 0.0)
}

/// The constant `cone.rs` writes from the flattening rather than transcribing, so the two crates
/// cannot drift. Bit-for-bit: it is the same arithmetic on the same literal, and anything less would
/// mean one of them rounded differently.
#[test]
fn the_rate_bounds_flattening_is_the_earth_crates_flattening() {
    let f = 1.0 / 298.257_223_563_f64;
    assert_eq!(
        f.to_bits(),
        FLATTENING.to_bits(),
        "the two flattenings are not the same bits"
    );
    let e2 = f * (2.0 - f);
    // 1/(1−e²) = 1.006 739 5, the factor the pointed cone's turn term is multiplied by.
    assert!(
        (1.0 / (1.0 - e2) - 1.006_739_496_742_276).abs() < 1e-15,
        "1/(1-e^2) is {}",
        1.0 / (1.0 - e2)
    );
}

/// `Fixed(−r/|r|)` is the same cone as `GeocentricNadir`, and the two code paths must produce the
/// same answer for it — not a close one.
///
/// `g` cannot be asserted bit-for-bit, and the reason is a property of the *old* form worth stating:
/// the geocentric arm builds `ũ = |r|² − c·r` from two quantities of order `|r|²` whose difference is
/// orders smaller, so it loses about `|r|²·ε/|r| = |r|·ε` metres to cancellation before dividing by
/// `|r|`. The pointed arm never forms that difference — a supplied boresight is already unit — so it
/// is better conditioned as well as one multiply cheaper per cell. The bound is that estimate,
/// `4·|r|·ε` (6.3e-9 m at 700 km), derived rather than fitted to the failure, and the measured worst
/// is printed so a regression shows as a number moving rather than as a pass.
///
/// `covered` *is* asserted exactly, because it is a boolean and there is no tolerance to hide a
/// disagreement in.
#[test]
fn a_fixed_axis_along_the_geocentric_nadir_is_the_geocentric_cone() {
    let mut worst = 0.0_f64;
    for (lat, lon, h) in [
        (0.0_f64, 0.0_f64, 700e3),
        (45.0, 33.0, 550e3),
        (-71.0, -122.0, 1200e3),
    ] {
        let sat = over(lat, lon, h);
        let bound = 4.0 * sat.norm() * f64::EPSILON;
        let nadir = -sat.scale(1.0 / sat.norm());
        for half_deg in [1.0_f64, 15.0, 45.0, 89.0] {
            let geocentric = Cone::new(half_deg.to_radians()).unwrap();
            let pointed = Cone::pointed(half_deg.to_radians(), Boresight::Fixed(nadir)).unwrap();
            for (clat, clon) in [
                (lat, lon),
                (lat + 3.0, lon),
                (lat, lon + 7.0),
                (-lat, lon + 180.0),
            ] {
                let g = geom(sat, on_surface(clat, clon));
                let (a, b) = (geocentric.g(&g), pointed.g(&g));
                worst = worst.max((a - b).abs());
                assert!(
                    (a - b).abs() <= bound,
                    "{lat},{lon},{h},{half_deg}°: g {a} against {b}, {} m apart against the \
                     {bound} m cancellation bound",
                    (a - b).abs()
                );
                assert_eq!(
                    geocentric.covered(&g),
                    pointed.covered(&g),
                    "{lat},{lon},{h},{half_deg}° at cell {clat},{clon}: the two paths disagree"
                );
            }
        }
    }
    println!("worst g disagreement between the two arms: {worst:.3e} m");
}

/// The batch kernel and the scalar predicate are two implementations of one condition, and the
/// pointed arm is a *new* one. They are compared over a grid that straddles the cone edge, so the
/// cells that matter — the ones within an ulp of `g = 0` — are actually in the sample.
#[test]
fn the_pointed_batch_kernel_agrees_with_the_pointed_scalar_predicate() {
    let sat = over(45.0962, 12.0, 550e3);
    let axis = geodetic_axis(45.0962, 12.0, 550e3);
    let sensor = Cone::pointed(20.0_f64.to_radians(), Boresight::Fixed(axis)).unwrap();

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut zs = Vec::new();
    for i in -60..=60 {
        for j in -60..=60 {
            // 20° at 550 km subtends about 1.8° of arc, so ±3° at 0.05° steps brackets the edge.
            let cell = on_surface(45.0962 + f64::from(i) * 0.05, 12.0 + f64::from(j) * 0.05);
            xs.push(cell.x);
            ys.push(cell.y);
            zs.push(cell.z);
        }
    }
    let zeros = vec![0.0; xs.len()];
    let grid = CellGrid::new(&xs, &ys, &zs, &zeros, &zeros, &zeros).unwrap();
    let mut out = vec![0_u8; xs.len()];
    cone_covered_into(sensor, sat, &grid, &mut out).unwrap();

    let mut inside = 0_u32;
    for k in 0..xs.len() {
        let cell = Vec3::new(xs[k], ys[k], zs[k]);
        let scalar = cone_covered_one(sensor, sat, cell);
        assert_eq!(
            out[k] == 1,
            scalar,
            "cell {k} at {cell:?}: kernel {} against scalar {scalar}",
            out[k]
        );
        assert_eq!(
            scalar,
            sensor.covered(&geom(sat, cell)),
            "cell {k}: the trait and the kernel helper disagree"
        );
        inside += u32::from(scalar);
    }
    // A grid that is entirely inside or entirely outside would pass the comparison vacuously.
    assert!(
        inside > 0 && inside < u32::try_from(xs.len()).unwrap(),
        "{inside} of {} cells covered — the grid does not straddle the cone edge",
        xs.len()
    );
}

/// The number the ticket is about: swapping the geocentric nadir for the geodetic one turns the cone
/// by exactly the deflection, so a cell sitting on one cone's edge is off the other's by that angle.
///
/// Asserted as an *angle between the two axes* against `derivations/nadir_deflection.mac`'s value at
/// height, and then as the consequence — a cone half-angle shrunk by the deflection contains no cell
/// the tilted cone does not.
#[test]
fn the_geodetic_axis_moves_the_footprint_by_the_published_deflection() {
    for (lat, h, want_deg) in [
        (45.096_2_f64, 550e3_f64, 0.177_125_f64),
        (45.096_2, 700e3, 0.173_366),
    ] {
        let sat = over(lat, 88.0, h);
        let geocentric = -sat.scale(1.0 / sat.norm());
        let geodetic = geodetic_axis(lat, 88.0, h);

        let between = geocentric
            .cross(geodetic)
            .norm()
            .atan2(geocentric.dot(geodetic));
        assert!(
            (between.to_degrees() - want_deg).abs() < 1e-6,
            "at {h} m the two axes are {}° apart, not {want_deg}°",
            between.to_degrees()
        );

        // 1.70 km of ground displacement straight below, at 550 km. `h * tan(delta)` is the
        // first-order figure the doc comment quotes; assert it to the 10 m it is quoted to.
        if (h - 550e3).abs() < 1.0 {
            let ground = h * between.tan();
            assert!(
                (ground - 1700.0).abs() < 10.0,
                "the ground displacement is {ground} m, not the documented 1.70 km"
            );
        }

        // The consequence: a cone about the geodetic axis, opened by the deflection, contains every
        // cell the geocentric cone of the nominal half-angle does. If the axes were the same, this
        // would hold for *zero* extra opening, so the second assertion is the one with teeth.
        let half = 15.0_f64.to_radians();
        let tight = Cone::pointed(half, Boresight::Fixed(geodetic)).unwrap();
        let opened = Cone::pointed(half + between, Boresight::Fixed(geodetic)).unwrap();
        let nominal = Cone::new(half).unwrap();

        let (mut escapes_tight, mut escapes_opened) = (0_u32, 0_u32);
        for i in -40..=40 {
            for j in -40..=40 {
                let cell = on_surface(lat + f64::from(i) * 0.1, 88.0 + f64::from(j) * 0.1);
                let g = geom(sat, cell);
                if nominal.covered(&g) {
                    escapes_tight += u32::from(!tight.covered(&g));
                    escapes_opened += u32::from(!opened.covered(&g));
                }
            }
        }
        assert_eq!(
            escapes_opened, 0,
            "opening by the deflection still leaves {escapes_opened} cells outside at {h} m"
        );
        assert!(
            escapes_tight > 0,
            "the two axes produced identical footprints at {h} m, so the deflection is not applied"
        );
    }
}

/// An axis with no length has no cone about it, and normalising it would give `NaN` and no error.
#[test]
fn a_degenerate_axis_is_refused_rather_than_normalised() {
    for bad in [
        Vec3::ZERO,
        Vec3::new(f64::NAN, 0.0, 0.0),
        Vec3::new(f64::INFINITY, 0.0, 0.0),
    ] {
        let err = Cone::pointed(0.3, Boresight::Fixed(bad)).unwrap_err();
        assert!(
            matches!(err, SensorError::DegenerateBoresight { .. }),
            "{bad:?} gave {err:?}"
        );
    }
    // And the half-angle is still checked first, so a bad angle with a bad axis names the angle.
    assert!(matches!(
        Cone::pointed(0.0, Boresight::Fixed(Vec3::ZERO)).unwrap_err(),
        SensorError::HalfAngleOutOfRange { .. }
    ));
}

/// The axis is normalised on construction, so a caller's unnormalised direction is not a silent
/// scale error in `g`.
#[test]
fn the_axis_is_normalised_on_construction() {
    let axis = Vec3::new(0.0, 0.0, -1.0);
    let unit = Cone::pointed(0.4, Boresight::Fixed(axis)).unwrap();
    for scale in [1e-6_f64, 3.0, 1e9] {
        let scaled = Cone::pointed(0.4, Boresight::Fixed(axis.scale(scale))).unwrap();
        assert_eq!(
            scaled, unit,
            "a boresight scaled by {scale} produced a different cone"
        );
    }
}

/// The rate bound is the thing the whole event search rests on, so a pointed cone must not report a
/// *smaller* one than the geocentric cone it generalises — and the excess is the derived 0.674 %,
/// not a fudge.
#[test]
fn a_pointed_cone_bounds_its_rate_above_the_geocentric_one() {
    let orbit = spacelab_sensors::OrbitBound {
        semi_major_axis: 6_928_137.0,
        eccentricity: 0.001,
        mu: 3.986_004_418e14,
    };
    let frame_rate = 7.292_115e-5;
    let half = 20.0_f64.to_radians();
    let geocentric = Cone::new(half).unwrap().max_rate(&orbit, frame_rate);
    let pointed = Cone::pointed(half, Boresight::Fixed(Vec3::new(0.0, 0.0, -1.0)))
        .unwrap()
        .max_rate(&orbit, frame_rate);
    assert!(
        pointed > geocentric,
        "the pointed bound {pointed} is not above the geocentric {geocentric}"
    );
    // The excess is the turn term scaled by 1/(1−e²) and nothing else, so the whole-bound ratio is
    // strictly between 1 and 1.006 74 — it would equal it only if the cone term vanished.
    let ratio = pointed / geocentric;
    assert!(
        ratio > 1.0 && ratio < 1.006_739_496_742_277,
        "the bound ratio is {ratio}, outside (1, 1/(1-e^2))"
    );
}
