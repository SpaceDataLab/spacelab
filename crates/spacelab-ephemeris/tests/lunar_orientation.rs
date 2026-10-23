//! Selenocentric propagation over DE440: the Moon's orientation, the origin shift, and the forces
//! that only make sense once both are right.
//!
//! Gated on the kernels — `tools/fetch-ephemeris.sh` for `de440s.bsp` and
//! `tools/fetch-lunar-kernels.sh` for the lunar orientation pair — and skipping cleanly without
//! them, so CI stays network-free.
//!
//! The oracle here is ANISE reading NAIF's own files. That makes these third-party checks of
//! `spacelab_propagation::lunar`, which derives the same PA→ME rotation from the three published
//! angles and evaluates the same gravity field through its own Gottlieb recursion.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;
use std::str::FromStr;

use spacelab_core::Vec3;
use spacelab_ephemeris::frames::{
    EARTH_J2000, MOON_J2000, MOON_ME_DE440_ME421_FRAME, MOON_PA_DE440_FRAME, SUN_J2000,
};
use spacelab_ephemeris::{BodyState, Ephemeris, Epoch, Matrix3};
use spacelab_propagation::ephemeris::Ephemeris as EphemerisTrait;
use spacelab_propagation::forces::{Acceleration, BodyOriented, PointMass, ThirdBody};
use spacelab_propagation::lunar;

const SPK: &str = "../../data/ephemeris/de440s.bsp";
const BPC: &str = "../../data/ephemeris/moon_pa_de440_200625.bpc";
const FK: &str = "../../data/ephemeris/moon_de440_250416.tf";

/// A working epoch inside every kernel's span, and far enough from J2000 that a rotation frozen at
/// J2000 would be visibly wrong.
fn epoch() -> Epoch {
    Epoch::from_str("2024-05-01T00:00:00 TDB").expect("literal epoch parses")
}

fn kernels_present() -> bool {
    for path in [SPK, BPC, FK] {
        if !Path::new(path).exists() {
            eprintln!(
                "skipping: {path} not present — run tools/fetch-ephemeris.sh and tools/fetch-lunar-kernels.sh"
            );
            return false;
        }
    }
    true
}

fn loaded() -> Ephemeris {
    Ephemeris::load(SPK)
        .expect("de440s.bsp loads")
        .with_kernel(BPC)
        .expect("lunar PA orientation loads")
        .with_frame_kernel(FK)
        .expect("DE440 frame kernel parses")
}

fn mul(a: &Matrix3, b: &Matrix3) -> Matrix3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

fn worst_elementwise(a: &Matrix3, b: &Matrix3) -> f64 {
    (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| (a[i][j] - b[i][j]).abs())
        .fold(0.0, f64::max)
}

/// The DCM out of the BPC is a rotation, and it is the *lunar* one.
///
/// Orthonormality alone is satisfied by the identity, so the second half asserts the thing that
/// distinguishes a real orientation from a placeholder: the PA x-axis points within a few degrees of
/// the Earth, because the Moon is tidally locked and the principal axis is the near-side axis.
#[test]
fn the_principal_axis_dcm_is_a_rotation_that_faces_the_earth() {
    if !kernels_present() {
        return;
    }
    let eph = loaded();
    let m = eph
        .rotation(MOON_J2000, MOON_PA_DE440_FRAME, epoch())
        .expect("the BPC covers 2024");

    let mut identity = [[0.0; 3]; 3];
    for (i, row) in identity.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    let transpose = [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ];
    let orthonormality = worst_elementwise(&mul(&m, &transpose), &identity);
    println!("PA DCM: worst |M·Mᵀ − I| = {orthonormality:.3e}");
    assert!(orthonormality < 1e-14, "PA DCM is not orthonormal");

    let determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    assert!(
        (determinant - 1.0).abs() < 1e-14,
        "PA DCM has determinant {determinant}, so it reflects as well as rotates"
    );

    // Earth as seen from the Moon, rotated into the principal axes.
    let earth = eph
        .state(EARTH_J2000, MOON_J2000, epoch())
        .expect("inside the de440s span")
        .position;
    let inertial = [earth.x, earth.y, earth.z];
    let body: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| m[i][j] * inertial[j]).sum())
        .collect();
    let norm = body.iter().map(|c| c * c).sum::<f64>().sqrt();
    let off_axis = (body[0] / norm).acos().to_degrees();
    println!("Earth is {off_axis:.2}° off the lunar PA x-axis");
    // The optical libration keeps this within ~8°; anything larger means the wrong frame.
    assert!(
        off_axis < 10.0,
        "Earth sits {off_axis:.2}° off the near-side axis — that is not the PA frame"
    );
}

/// `spacelab_propagation::lunar::pa_to_me` against ANISE reading NAIF's own frame kernel.
///
/// Our rotation is built from the kernel's three arcsecond angles about axes 3-2-1; ANISE parses the
/// same assignments out of `moon_de440_250416.tf`, converts them to Euler parameters and composes
/// them through its orientation tree. Two code paths, one number.
///
/// The tolerance pins the *direction*, which is the thing that goes wrong: transposing the matrix
/// moves every off-diagonal element by twice its value, 7.6e-4. It caught exactly that when this test
/// was first run — the kernel's angles run ME→PA, not PA→ME as their placement suggests.
///
/// **A residual of 1.3e-7 remains, and it is an ordering disagreement, not noise.** We compose
/// `R₁(a₃)·R₂(a₂)·R₃(a₁)`, which is SPICE's documented `eul2m(ANGLES(3), ANGLES(2), ANGLES(1),
/// AXES(3), AXES(2), AXES(1))` for a `TKFRAME` `ANGLES` spec; ANISE (0.10.4,
/// `naif::kpl::parser::convert_fk_items`) multiplies the axes in array order, giving
/// `R₃(a₁)·R₂(a₂)·R₁(a₃)`. The two agree to first order and differ by `a₁·a₂ = 1.255e-7` rad — 0.22 m
/// on the lunar surface. Nothing in the data here distinguishes them: the measurement that fixes the
/// direction (the mean Earth longitude) is 2.8e-4 rad of libration residual wide, three orders above
/// this. So the tolerance admits the difference and names it rather than hiding it.
#[test]
fn the_pa_to_me_rotation_matches_anise_reading_the_de440_frame_kernel() {
    if !kernels_present() {
        return;
    }
    let eph = loaded();
    let reference = eph
        .rotation(MOON_PA_DE440_FRAME, MOON_ME_DE440_ME421_FRAME, epoch())
        .expect("the frame kernel defines ME off PA");
    let ours = lunar::pa_to_me();

    let worst = worst_elementwise(&ours, &reference);
    println!(
        "pa_to_me vs ANISE/DE440 FK: worst element difference = {worst:.3e} \
         ({:.2} m on the surface)",
        worst * lunar::MOON_MEAN_RADIUS
    );
    assert!(
        worst < 2e-7,
        "our PA→ME differs from the frame kernel's by {worst:.3e}, beyond the known \
         composition-order term"
    );

    // And the rotation is not the identity: it moves a surface point by roughly 900 m, which is why
    // the distinction between the gravity frame and the map frame is worth carrying at all.
    let surface = [lunar::MOON_MEAN_RADIUS, 0.0, 0.0];
    let rotated: Vec<f64> = (0..3)
        .map(|i| (0..3).map(|j| reference[i][j] * surface[j]).sum())
        .collect();
    let displacement = (0..3)
        .map(|i| (rotated[i] - surface[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("PA→ME moves an equatorial surface point by {displacement:.1} m");
    assert!(
        (500.0..1500.0).contains(&displacement),
        "PA→ME surface displacement {displacement:.1} m is not the published ~900 m"
    );
}

/// The origin shift is exact both ways, and it is not a no-op.
#[test]
fn recentering_on_the_moon_round_trips_and_moves_the_state() {
    if !kernels_present() {
        return;
    }
    let eph = loaded();
    // A geocentric state near the Moon's distance, so the shift dominates it.
    let geocentric = BodyState {
        position: Vec3::new(-300_000.0, -250_000.0, -70_000.0),
        velocity: Vec3::new(0.6, -0.8, -0.3),
    };

    let seleno = eph
        .recenter(geocentric, EARTH_J2000, MOON_J2000, epoch())
        .expect("both centres are in the kernel");
    let back = eph
        .recenter(seleno, MOON_J2000, EARTH_J2000, epoch())
        .expect("and back again");

    let position_error = (back.position - geocentric.position).norm();
    let velocity_error = (back.velocity - geocentric.velocity).norm();
    println!("recenter round trip: {position_error:.3e} km, {velocity_error:.3e} km/s");
    assert!(position_error < 1e-9, "round trip lost {position_error} km");
    assert!(
        velocity_error < 1e-12,
        "round trip lost {velocity_error} km/s"
    );

    // A round trip cannot see the sign of the shift — it cancels either way, and so does the
    // magnitude below. The only check that can is an absolute one against a state ANISE produces
    // directly: recentring the Sun's geocentric state onto the Moon must give the selenocentric Sun.
    let sun_from_earth = eph
        .state(SUN_J2000, EARTH_J2000, epoch())
        .expect("the Sun is in DE440");
    let sun_recentred = eph
        .recenter(sun_from_earth, EARTH_J2000, MOON_J2000, epoch())
        .expect("both centres are in the kernel");
    let sun_direct = eph
        .state(SUN_J2000, MOON_J2000, epoch())
        .expect("ANISE chains the same two segments");
    let absolute_error = (sun_recentred.position - sun_direct.position).norm();
    println!("recentred Sun vs ANISE direct: {absolute_error:.3e} km");
    assert!(
        absolute_error < 1e-9,
        "recentring the Sun onto the Moon is off by {absolute_error} km — a wrong sign puts it \
         {:.0} km out",
        2.0 * (sun_from_earth.position - sun_direct.position).norm()
    );

    // And the same absolutely for the velocity, which is a separate line of the implementation and
    // is otherwise seen only through a round trip and a magnitude — both blind to its sign.
    let absolute_rate_error = (sun_recentred.velocity - sun_direct.velocity).norm();
    println!("recentred Sun rate vs ANISE direct: {absolute_rate_error:.3e} km/s");
    assert!(
        absolute_rate_error < 1e-12,
        "the recentred Sun's velocity is off by {absolute_rate_error} km/s — a wrong sign puts it \
         {:.3} km/s out, twice the Moon's own speed",
        2.0 * (sun_from_earth.velocity - sun_direct.velocity).norm()
    );

    // The velocity shift is the part an origin-shift implementation forgets: the Moon moves at about
    // 1 km/s, a quarter of a low-lunar orbital velocity, so dropping it is not a rounding error.
    let velocity_shift = (seleno.velocity - geocentric.velocity).norm();
    println!("the Moon's velocity shift is {velocity_shift:.3} km/s");
    assert!(
        (0.9..1.1).contains(&velocity_shift),
        "lunar velocity {velocity_shift:.3} km/s is not the ~1 km/s the ephemeris says"
    );

    // A recentre across differing axes is a rotation, and this function refuses it rather than
    // silently translating a state into a frame it does not belong to.
    assert!(
        eph.recenter(geocentric, EARTH_J2000, MOON_PA_DE440_FRAME, epoch())
            .is_err(),
        "recenter accepted two centres whose axes differ"
    );
}

/// About the Moon, the Earth is the dominant perturbation — the opposite of the geocentric case.
#[test]
fn about_the_moon_the_earth_outweighs_the_sun_as_a_third_body() {
    if !kernels_present() {
        return;
    }
    let eph = loaded();
    let et = epoch().to_et_seconds();
    // A 100 km circular low lunar orbit, in the selenocentric J2000 frame.
    let r = [lunar::MOON_MEAN_RADIUS + 100e3, 0.0, 0.0];
    let v = [0.0, 1_630.0, 0.0];

    let earth = ThirdBody {
        mu: 3.986_004_418e14,
        ephemeris: eph.perturbing_body(EARTH_J2000, MOON_J2000, et),
    };
    let sun = ThirdBody {
        mu: 1.327_124_400_41e20,
        ephemeris: eph.perturbing_body(SUN_J2000, MOON_J2000, et),
    };
    let magnitude = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let from_earth = magnitude(earth.acceleration::<f64>(0.0, &r, &v));
    let from_sun = magnitude(sun.acceleration::<f64>(0.0, &r, &v));
    println!("LLO third body: Earth {from_earth:.3e} m/s², Sun {from_sun:.3e} m/s²");

    // The tidal term is 2·GM⊕·r/d³ = 2·3.986e14·1.837e6/(3.84e8)³ ≈ 2.6e-5 m/s², so the band is set
    // around a computed number rather than a remembered order of magnitude.
    assert!(
        (5e-6..5e-5).contains(&from_earth),
        "Earth third-body {from_earth:.3e} m/s² is not the ~2.6e-5 the tidal term predicts"
    );
    assert!(
        from_earth > 100.0 * from_sun,
        "Earth/Sun third-body ratio is only {:.1}",
        from_earth / from_sun
    );

    // The centre is not decoration. The Sun seen from the Moon and the Sun seen from the Earth differ
    // by exactly the Earth-Moon vector, so a bridge that ignored `center` and stayed geocentric would
    // put these two at the same place.
    let from_moon = eph.perturbing_body(SUN_J2000, MOON_J2000, et).position(0.0);
    let from_earth_centre = eph
        .perturbing_body(SUN_J2000, EARTH_J2000, et)
        .position(0.0);
    let separation = (0..3)
        .map(|i| (from_moon[i] - from_earth_centre[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("the two Sun vectors differ by {separation:.3e} m");
    assert!(
        (3.5e8..4.1e8).contains(&separation),
        "the two centres are {separation:.3e} m apart, not the lunar distance"
    );
}

/// The gravity field, the BPC orientation and the origin shift, composed as a propagation would.
///
/// This is the end the ticket is about: an acceleration on a low lunar orbiter that carries the real
/// mascon field in the real principal-axis frame. The measurement is what the harmonics add over a
/// point mass, and that it changes as the Moon librates — a frozen or identity rotation would give a
/// constant.
#[test]
fn the_oriented_lunar_field_departs_from_a_point_mass_and_tracks_the_libration() {
    if !kernels_present() {
        return;
    }
    let eph = loaded();
    let et = epoch().to_et_seconds();
    let field = lunar::embedded(40).expect("degree 40 is inside the embedded model");
    let oriented = BodyOriented {
        inner: field,
        rotate: |t: f64| {
            eph.rotation(
                MOON_J2000,
                MOON_PA_DE440_FRAME,
                Epoch::from_et_seconds(et + t),
            )
            .expect("the BPC covers the arc")
        },
    };
    let point = PointMass { mu: lunar::MOON_MU };

    let r = [1.2e6, 9.0e5, 1.0e6];
    let v = [0.0, 1_630.0, 0.0];
    let full = oriented.acceleration::<f64>(0.0, &r, &v);
    let central = point.acceleration::<f64>(0.0, &r, &v);
    let harmonics = (0..3)
        .map(|i| (full[i] - central[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let total = (full[0] * full[0] + full[1] * full[1] + full[2] * full[2]).sqrt();
    println!("LLO: |a| = {total:.6} m/s², harmonics above the point mass = {harmonics:.3e} m/s²");
    // Measured 6.09e-4 m/s² at this point in `spacelab-propagation/tests/lunar_gravity.rs`; the band
    // is loose enough for a coefficient set change and tight enough that a dropped field is caught.
    assert!(
        (1e-4..1e-2).contains(&harmonics),
        "lunar harmonics {harmonics:.3e} m/s² are not the ~6e-4 the field carries"
    );

    // Half a month later the Moon has turned, so the same inertial point sees a different part of
    // the field. A rotation stuck at the construction epoch would give exactly zero here.
    let later = oriented.acceleration::<f64>(14.0 * 86_400.0, &r, &v);
    let turned = (0..3)
        .map(|i| (later[i] - full[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    println!("half a synodic month later the acceleration moved by {turned:.3e} m/s²");
    assert!(
        turned > 1e-4,
        "the field did not turn with the Moon: {turned:.3e} m/s²"
    );
}
