//! What the instantaneous measurement geometry costs, in metres and in arcseconds.
//!
//! `spacelab_estimation::measurement` forms every observable from one state, one station and one
//! instant. An instrument does not: the light it reads left the satellite a transit time ago, so the
//! distance and the direction it records belong to where the satellite *was*. This file measures the
//! difference on the ISS day `access_oracle.rs` validates against Orekit, so the geometry is one an
//! independent implementation has already agreed with.
//!
//! Everything runs in **TEME**, not ITRF. That is not a preference: a light-time iteration in a
//! rotating frame would have to carry the frame's rotation during the transit as a fictitious term,
//! which is the Sagnac correction and is worth about 1.5 m at the horizon here — the same size as
//! the effect being measured. Rotating the station into an inertial frame at its own epoch is the
//! same computation done once and openly, and it is what `spacelab_slr`'s two-way model does for
//! the same reason.
//!
//! **The range term is the satellite's own radial speed times the transit,** and the word "own" is
//! the whole of it: light time moves the *satellite* back to where it emitted, and leaves the
//! station where the observation happened. Using the relative range rate instead — the obvious
//! reach, since that is what [`spacelab_estimation::measurement::range_rate`] computes — adds the
//! station's 381 m/s and is wrong by **2.6 m** here, which is a hundred times an SLR normal point.
//! Measured: **up to 51.8 m**, at the horizon, where the transit is longest and the satellite is
//! closing fastest; it passes through zero at closest approach, where the radial speed does.
//!
//! **The angular term does not depend on the range at all.** The satellite moves `v τ` while the
//! light is in flight and the station is `ρ = c τ` away, so the angle subtended is `v_⊥ τ / ρ =
//! v_⊥ / c` — the range cancels. It is therefore bounded by `|v|/c` for the orbit and by nothing
//! about the pass: **5.30″** measured, which is this element set's 7.71 km/s over `c` to three
//! figures, and it is reached at the pass peak where the velocity is almost wholly transverse. What
//! varies beneath that ceiling is the transverse *fraction*, a property of the pass rather than of
//! the range: **1.79″..5.30″** over the day, a factor of 3, against a factor of 1461 in the range
//! error over the same samples. The direction is always *behind* the satellite along its velocity,
//! because that is where it was.
//!
//! **Diurnal aberration** is the other half of the classical correction and is not modelled either.
//! It is the observer's velocity with respect to TEME divided by `c` — 0.26″ at this station's
//! latitude, and the reason it is not the 20.5″ of the annual term is that the station and the
//! satellite share the Earth's barycentric velocity, which therefore cancels in a geocentric frame.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
#![allow(clippy::doc_markdown)] // SGP4, Orekit, ISS, TEME are names in prose

use std::sync::LazyLock;

use spacelab::{EopSource, teme_to_itrf_at};
use spacelab_core::{Angle, Length, UtcDateTime, Vec3};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_eop::EopSeries;
use spacelab_estimation::light_time::{SPEED_OF_LIGHT, down_leg};
use spacelab_frames::{Itrf, Position, StateVector, Teme, Transform};
use spacelab_sgp4::Satellite;

/// The `access_oracle.rs` element set and station, so the geometry is the Orekit-validated one.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// The same IERS excerpt `access_oracle.rs` reads, so the geometry is the validated one.
const EOP: &str = include_str!("fixtures/finals2000A_2008264.txt");

/// One day of scan, sampled every five seconds — fine enough that a pass gets a hundred samples.
const SPAN: f64 = 86_400.0;
const STEP: f64 = 5.0;

/// Light-time convergence: femtoseconds, and a budget the contraction reaches it in twice over.
const TOLERANCE: f64 = 1e-15;
const MAX_ITERATIONS: usize = 20;

const ARCSEC_PER_RADIAN: f64 = 180.0 * 3600.0 / core::f64::consts::PI;

/// Station geodetic position: 35°N, 33°E, on the ellipsoid.
fn station_geodetic() -> Geodetic {
    Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    }
}

/// The TEME→ITRF transform `t` seconds after 2008-09-20 12:00 UTC, on the measured Earth
/// orientation the oracle uses.
fn transform_at(start_ns: i128, t: f64) -> Transform<Teme, Itrf> {
    // Parsed once: this is called at every sample of a day, and re-parsing the excerpt each time
    // would make the file's own cost the thing it measures.
    static SERIES: LazyLock<EopSeries> =
        LazyLock::new(|| EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses"));
    let series = &*SERIES;
    let ns = start_ns + (t * 1e9) as i128;
    let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
    let (transform, _) = teme_to_itrf_at(label, EopSource::Series(series))
        .expect("the scan lies inside the excerpt");
    transform
}

/// Station position, velocity and zenith in TEME at `t`.
///
/// The station is at rest in ITRF, so its inertial velocity is entirely the `ω × r` term
/// [`Transform::state`] carries — 381 m/s at this latitude, and the whole of the diurnal aberration.
fn station_teme(start_ns: i128, t: f64) -> (Vec3, Vec3, Vec3) {
    let g = station_geodetic();
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    let zenith_itrf = Vec3::new(cla * clo, cla * slo, sla);

    let itrf_to_teme = transform_at(start_ns, t).inverse();
    let state = itrf_to_teme.state(StateVector::<Itrf>::new(
        geodetic_to_itrf(g).vector,
        Vec3::new(0.0, 0.0, 0.0),
    ));
    (
        state.position.vector,
        state.velocity.vector,
        itrf_to_teme
            .position(Position::<Itrf>::new(zenith_itrf))
            .vector,
    )
}

/// ISS position and velocity in TEME at `t`, straight out of SGP4 — no frame change at all.
fn iss_teme(satellite: &Satellite, start_ns: i128, t: f64) -> (Vec3, Vec3) {
    let ns = start_ns + (t * 1e9) as i128;
    let state = satellite
        .propagate_at(UtcDateTime::from_naive_ns_since_j2000(ns, false))
        .expect("the ISS has not decayed");
    (state.position.vector, state.velocity.vector)
}

/// `spacelab_estimation` does not depend on `spacelab_core`, so its geometry is plain arrays.
fn xyz(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

fn start_ns() -> i128 {
    UtcDateTime::new(2008, 9, 20, 12, 0, 0.0)
        .unwrap()
        .naive_ns_since_j2000()
}

/// Everything one sample of a pass has to say.
struct Sample {
    elevation_deg: f64,
    /// Instantaneous geometric range minus the light-time path, metres.
    range_error: f64,
    /// `(v_sat · ρ̂) τ`, the first-order prediction of the same quantity, metres.
    predicted_range: f64,
    /// The same first order with the *relative* range rate — the wrong reach, kept to size it.
    predicted_relative: f64,
    /// Angle between the instantaneous line of sight and the light-time-corrected one, arcseconds.
    angle_error: f64,
    /// `v_⊥/c` in arcseconds, the prediction the range cancels out of.
    predicted_angle: f64,
    /// `|v_sat|/c` in arcseconds — the ceiling `v_⊥/c` cannot cross at any range.
    angle_ceiling: f64,
    /// Whether the correction moved the satellite backwards along its own velocity.
    trails: bool,
}

/// Sample the day and keep everything above the horizon.
fn samples() -> Vec<Sample> {
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses");
    let start = start_ns();
    let mut out = Vec::new();

    let mut t = 0.0;
    while t <= SPAN {
        let (station, station_velocity, zenith) = station_teme(start, t);
        let (satellite_now, satellite_velocity) = iss_teme(&satellite, start, t);

        let instantaneous = satellite_now - station;
        let distance = instantaneous.norm();
        let elevation = (instantaneous.dot(zenith) / distance).asin();
        if elevation <= 0.0 {
            t += STEP;
            continue;
        }

        let transit = down_leg(
            t,
            xyz(station),
            |epoch| Some(xyz(iss_teme(&satellite, start, epoch).0)),
            TOLERANCE,
            MAX_ITERATIONS,
        )
        .expect("the light-time iteration converges on a LEO pass");
        let (satellite_then, _) = iss_teme(&satellite, start, transit.emit);
        let retarded = satellite_then - station;

        // The satellite's own radial speed, positive receding. Not the relative one: the station is
        // held at the reception epoch, so its motion is not part of this correction.
        let line_of_sight = instantaneous.scale(1.0 / distance);
        let radial = line_of_sight.dot(satellite_velocity);
        let transverse = (satellite_velocity - line_of_sight.scale(radial)).norm();
        let cosine = (instantaneous.dot(retarded) / (distance * retarded.norm())).clamp(-1.0, 1.0);

        out.push(Sample {
            elevation_deg: elevation.to_degrees(),
            range_error: distance - transit.path(),
            predicted_range: radial * transit.delay,
            predicted_relative: line_of_sight.dot(satellite_velocity - station_velocity)
                * transit.delay,
            angle_error: cosine.acos() * ARCSEC_PER_RADIAN,
            predicted_angle: transverse / SPEED_OF_LIGHT * ARCSEC_PER_RADIAN,
            angle_ceiling: satellite_velocity.norm() / SPEED_OF_LIGHT * ARCSEC_PER_RADIAN,
            trails: (satellite_then - satellite_now).dot(satellite_velocity) < 0.0,
        });
        t += STEP;
    }
    out
}

/// The range error against its first-order closed form `(v_sat·ρ̂)τ`; the term it drops,
/// `½ρ̈τ²`, is sub-millimetre at a LEO transit. (G3)
#[test]
fn the_range_error_is_the_satellites_own_radial_speed_times_the_transit() {
    let samples = samples();
    assert!(
        samples.len() > 200,
        "{} samples above the horizon is too few to say anything",
        samples.len()
    );

    let (mut worst_error, mut worst_at) = (0.0_f64, f64::NAN);
    let (mut worst_residual, mut worst_relative_residual) = (0.0_f64, 0.0_f64);
    let (mut highest, mut error_at_highest) = (0.0_f64, f64::NAN);
    for s in &samples {
        if s.range_error.abs() > worst_error.abs() {
            worst_error = s.range_error;
            worst_at = s.elevation_deg;
        }
        // First order in `τ`; what it drops is `½ ρ̈ τ²`, sub-millimetre at a LEO transit.
        worst_residual = worst_residual.max((s.range_error - s.predicted_range).abs());
        worst_relative_residual =
            worst_relative_residual.max((s.range_error - s.predicted_relative).abs());
        if s.elevation_deg > highest {
            highest = s.elevation_deg;
            error_at_highest = s.range_error;
        }
    }
    println!(
        "range: worst {worst_error:+.2} m at {worst_at:.1}° elevation; \
         {error_at_highest:+.2} m at the day's peak of {highest:.1}°"
    );
    println!(
        "range: residual against `(v_sat·ρ̂)τ` is {worst_residual:.2e} m, \
         against the relative range rate {worst_relative_residual:.2} m"
    );

    assert!(
        (20.0..80.0).contains(&worst_error.abs()),
        "worst range error {worst_error:.2} m is outside the measured band"
    );
    assert!(
        worst_at < 5.0,
        "the worst range error sat at {worst_at:.1}° elevation, not at the horizon where the \
         transit is longest and the satellite closes fastest"
    );
    // Sharper than a magnitude band: name the mechanism and let it fail if it is a different one.
    assert!(
        worst_residual < 0.01,
        "{worst_residual:.3} m between the measured error and `(v_sat·ρ̂)τ`: the effect is not the \
         one claimed"
    );
    // The station's own motion is *not* part of this correction, and reaching for the relative
    // range rate — which is what the range-rate observable computes — is wrong by this much.
    assert!(
        worst_relative_residual > 1.0,
        "using the relative range rate differs by only {worst_relative_residual:.3} m, so this \
         file is no longer distinguishing the two"
    );
    // The satellite's radial speed passes through zero at closest approach, so the error must too,
    // and the highest sample of the day is the closest one it has.
    assert!(
        error_at_highest.abs() < 10.0,
        "{error_at_highest:.2} m at the closest approach of the day, where the radial speed is \
         smallest and the error has to follow it down"
    );
}

/// The angular error against its closed form `v⊥/c`, in which the range cancels. (G3)
#[test]
fn light_time_leaves_every_line_of_sight_behind_the_satellite() {
    let samples = samples();

    let (mut worst_angle, mut worst_at, mut worst_residual) = (0.0_f64, f64::NAN, 0.0_f64);
    let (mut lowest, mut angle_at_lowest) = (f64::INFINITY, f64::NAN);
    let (mut highest, mut angle_at_highest) = (0.0_f64, f64::NAN);
    for s in &samples {
        assert!(
            s.trails,
            "at {:.1}° the light-time correction moved the satellite *forward* along its own \
             velocity, which is the wrong sign of time",
            s.elevation_deg
        );
        if s.angle_error > worst_angle {
            worst_angle = s.angle_error;
            worst_at = s.elevation_deg;
        }
        worst_residual = worst_residual.max((s.angle_error - s.predicted_angle).abs());
        if s.elevation_deg > highest {
            highest = s.elevation_deg;
            angle_at_highest = s.angle_error;
        }
        if s.elevation_deg < lowest {
            lowest = s.elevation_deg;
            angle_at_lowest = s.angle_error;
        }
    }
    println!(
        "angles: worst {worst_angle:.2}\" at {worst_at:.1}° elevation; \
         {angle_at_lowest:.2}\" at {lowest:.2}° and {angle_at_highest:.2}\" at the day's peak of \
         {highest:.1}°; residual against `v_⊥/c` is {worst_residual:.2e}\""
    );

    assert!(
        (4.0..7.0).contains(&worst_angle),
        "worst angular lag {worst_angle:.2}\" is outside the measured band"
    );
    // It is `v_⊥/c` — the range cancels between the displacement `v τ` and the distance `c τ`, so
    // the size of the pass has nothing to do with it. This is the assertion that says so.
    assert!(
        worst_residual < 0.01,
        "{worst_residual:.3}\" between the measured lag and `v_⊥/c`: the effect is not the one \
         claimed"
    );
    // The consequence, and the reason the range term cannot stand in for this one: over the whole
    // day the angular lag stays inside a narrow band while the range error it accompanies spans two
    // orders of magnitude. Stated over every sample rather than over the horizon and the peak,
    // because `v_⊥` is a property of the *pass* — a sample at 0° elevation on a pass that crosses
    // the station's meridian obliquely has a different transverse fraction from one on a pass that
    // rises straight at it, and comparing two single samples measures which passes the scan grid
    // happened to land on.
    let (angle_min, angle_max, ceiling) =
        samples
            .iter()
            .fold((f64::INFINITY, 0.0_f64, 0.0_f64), |(lo, hi, c), s| {
                (
                    lo.min(s.angle_error),
                    hi.max(s.angle_error),
                    c.max(s.angle_ceiling),
                )
            });
    let (range_min, range_max) = samples
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), s| {
            (lo.min(s.range_error.abs()), hi.max(s.range_error.abs()))
        });
    println!(
        "over the day: lag {angle_min:.2}\"..{angle_max:.2}\" against a `|v|/c` ceiling of \
         {ceiling:.2}\" (x{:.1} spread); range error {range_min:.2}..{range_max:.2} m (x{:.0})",
        angle_max / angle_min,
        range_max / range_min
    );

    // The ceiling is the range-independence, exactly stated: `v τ` and `c τ` carry the same τ, so it
    // cancels and only the direction of the velocity survives — the lag can never exceed `|v|/c`
    // however far away the satellite is. The spread beneath it is the transverse fraction, which is
    // a property of the *pass* and not of the range: a factor of 3 across the day, against a factor
    // of 1461 in the range error over the same samples.
    assert!(
        angle_max <= ceiling,
        "lag {angle_max:.4}\" exceeds the `|v|/c` ceiling of {ceiling:.4}\", which no transverse \
         component can do"
    );
    assert!(
        angle_max / angle_min < 5.0,
        "the lag spans x{:.1} over the day, too wide to be a transverse term under a fixed ceiling",
        angle_max / angle_min
    );
    assert!(
        range_max / range_min > 100.0,
        "the range error spans only x{:.0}, so this file is no longer contrasting the two",
        range_max / range_min
    );
}

#[test]
fn diurnal_aberration_is_the_stations_own_speed_and_nothing_annual() {
    let start = start_ns();
    let mut fastest = 0.0_f64;
    let mut t = 0.0;
    while t <= SPAN {
        let (_, velocity, _) = station_teme(start, t);
        fastest = fastest.max(velocity.norm());
        t += 60.0;
    }
    let arcsec = fastest / SPEED_OF_LIGHT * ARCSEC_PER_RADIAN;
    println!("aberration: station speed {fastest:.1} m/s in TEME, {arcsec:.3}\" of aberration");

    // 465 m/s at the equator, times cos(35°) — the station is at rest in ITRF, so this is entirely
    // the `ω × r` term, and a transform that dropped it would report zero here.
    let equatorial = 465.1;
    let expected = equatorial * (35.0_f64.to_radians()).cos();
    assert!(
        (fastest - expected).abs() < 5.0,
        "{fastest:.1} m/s against the {expected:.1} m/s Earth rotation implies at 35° latitude"
    );
    assert!(
        (0.2..0.3).contains(&arcsec),
        "{arcsec:.3}\" of diurnal aberration is outside the measured band"
    );
    // The annual term would be 20.5″, and its absence is the point: station and satellite share the
    // Earth's barycentric velocity, so it cancels in any geocentric frame.
    assert!(
        arcsec < 1.0,
        "{arcsec:.3}\" is annual-sized, which would mean the station velocity is barycentric"
    );
}
