//! What refraction costs an access window, and why the correction is a constant.
//!
//! Two elevation paths exist in this workspace and neither bends the ray: `spacelab_sensors`'
//! `ElevationMask`, which the coverage engine and both language bridges scan with, and
//! `spacelab_coverage::detectors::ElevationDetector`, which is the Orekit-parity `g`-function. The
//! atmosphere appears only in `spacelab_coverage::observability::Refraction`, on the optical
//! observability path, so a caller who asked for "visible above 10°" got the *geometric* 10° and
//! nothing said so. This file is the measurement that decides what to do about it.
//!
//! Both parts run on the ISS geometry `access_oracle.rs` validates against Orekit, so the pass set is
//! one an independent implementation has already agreed with.
//!
//! **① The cost.** Refraction lifts the satellite, so a refracted window opens early and closes late —
//! a *bias* in one direction on every edge of every pass, not a residual that averages out. Measured
//! over this day: **0.81–1.18 s** at a 10° mask, **2.0–8.6 s** at 5°, **9.3–20.3 s** at the horizon.
//!
//! The size is `R(mask)/(dε/dt)` at the crossing, so the *pass* sets it as much as the atmosphere
//! does: the 20.3 s belongs to a pass peaking at 2.06°, which crosses the horizon almost
//! tangentially, and the 9.3 s to one peaking at 47.8°, which climbs through it. That is why
//! `docs/limitations.md` had this filed under "inside the TLE floor" and why it is not — a second or
//! two is the *best* case at a 10° mask, and nothing about it averages out.
//!
//! **② The equivalence.** `h ↦ h + R(h)` is strictly increasing, so `apparent(h) ≥ mask` and
//! `h ≥ geometric(mask)` are the same condition. Refraction applied once to the *mask* therefore
//! gives the same edges as refraction applied to every sample, which is why the correction costs
//! nothing at runtime and the batch kernel never has to see it: it is `Refraction::geometric` on the
//! threshold, and `ElevationMask::new` on the result.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
#![allow(clippy::doc_markdown)] // SGP4, Orekit, ISS are names in prose

use spacelab::{EopSource, teme_to_itrf_at};
use spacelab_core::{Angle, Length, UtcDateTime, Vec3};
use spacelab_coverage::{
    ApparentAltitude, ElevationDetector, Refraction, TrueAltitude, Window, windows,
};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_eop::EopSeries;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sgp4::Satellite;

const MU: f64 = 3.986_004_418e14;
const EARTH_RADIUS: f64 = 6_378_137.0;

/// The `access_oracle.rs` element set and station, so the pass set is the Orekit-validated one.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// The same IERS excerpt `access_oracle.rs` reads, so the pass set really is the validated one.
const EOP: &str = include_str!("fixtures/finals2000A_2008264.txt");

/// Semi-major axis and eccentricity of that element set, for the rate bound below.
const SEMI_MAJOR_AXIS: f64 = 6_795e3;
const ECCENTRICITY: f64 = 0.0007;

/// Edge tolerance, seconds — `access_oracle.rs`'s, so the two files' numbers are comparable.
const TOL: f64 = 1e-4;

/// One sidereal day of scan.
const SPAN: f64 = 86_400.0;

/// Upper bound on `|dε/dt|`, radians per second, for the elevation angle of a satellite on this orbit
/// seen from a station on the rotating Earth.
///
/// The satellite's angular rate about the station is `v_⊥/d ≤ ‖v_rel‖/d_min`. Perigee speed bounds
/// the inertial term, the station's own `ω⊕R⊕` bounds the frame term, and the closest the satellite
/// can come to the station bounds the range. Overestimating is safe here — `windows` steps by
/// `|g|/max_rate`, so a loose bound costs probes, never a missed crossing — but it is derived rather
/// than picked so that a change of orbit cannot silently invalidate it.
fn max_elevation_rate() -> f64 {
    let perigee = SEMI_MAJOR_AXIS * (1.0 - ECCENTRICITY);
    let speed = (MU * (1.0 + ECCENTRICITY) / perigee).sqrt();
    (speed + EARTH_ROTATION_RATE * EARTH_RADIUS) / (perigee - EARTH_RADIUS)
}

/// The station of `access_oracle.rs`: 35°N, 33°E, on the ellipsoid.
fn station() -> (Vec3, Vec3) {
    let g = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    (
        geodetic_to_itrf(g).vector,
        Vec3::new(cla * clo, cla * slo, sla),
    )
}

/// ITRF position of the ISS `t` seconds after 2008-09-20 12:00 UTC, on the same measured Earth
/// orientation the oracle uses.
fn iss() -> impl Fn(f64) -> Vec3 {
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses");
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let start_ns = UtcDateTime::new(2008, 9, 20, 12, 0, 0.0)
        .unwrap()
        .naive_ns_since_j2000();
    move |t: f64| {
        let ns = start_ns + (t * 1e9) as i128;
        let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
        let teme = satellite
            .propagate_at(label)
            .expect("the ISS has not decayed");
        let (transform, _) = teme_to_itrf_at(label, EopSource::Series(&series))
            .expect("the scan lies inside the excerpt");
        transform.state(teme).position.vector
    }
}

/// Which elevation the mask is compared against.
#[derive(Clone, Copy)]
enum Ray {
    /// The geometric elevation against the mask as given — what the engine does today.
    Straight,
    /// The refracted elevation against the mask as given — the expensive, obviously-correct form.
    Bent,
    /// The geometric elevation against the mask `Refraction::geometric` moves it to — the cheap form.
    ShiftedMask,
}

/// The windows above `mask` radians over the scan, with the elevation treated according to `ray`.
fn passes(mask: f64, ray: Ray) -> Vec<Window> {
    let (cell, zenith) = station();
    let sat_at = iss();
    let refraction = Refraction::STANDARD;

    // `min_elevation = 0` makes `g` the elevation itself, in the well-conditioned `atan2` form.
    let threshold = match ray {
        Ray::ShiftedMask => {
            refraction
                .geometric(ApparentAltitude(mask))
                .expect("an apparent mask above the horizon has a geometric counterpart")
                .0
        }
        Ray::Straight | Ray::Bent => mask,
    };
    let detector = ElevationDetector::new(cell, zenith, 0.0);

    let g = |t: f64| {
        let h = detector.g(sat_at(t));
        match ray {
            Ray::Straight | Ray::ShiftedMask => h - threshold,
            // Below the fit's floor the satellite is a degree under the horizon and 40 air masses
            // deep; the gate is shut, and the unrefracted difference already says so.
            Ray::Bent => refraction
                .apparent(TrueAltitude(h))
                .map_or(h - threshold, |a| a.0 - threshold),
        }
    };

    let mut out = Vec::new();
    windows(g, max_elevation_rate(), 0.0, SPAN, TOL, |w| out.push(w));
    out
}

/// `dε/dt` at `t`, radians per second, by central difference over a millisecond — the divisor that
/// turns an angle at the mask into a time at the edge.
fn elevation_rate(t: f64) -> f64 {
    let (cell, zenith) = station();
    let sat_at = iss();
    let detector = ElevationDetector::new(cell, zenith, 0.0);
    let h = 1e-3;
    (detector.g(sat_at(t + h)) - detector.g(sat_at(t - h))) / (2.0 * h)
}

/// The highest elevation the satellite reaches inside `w`, degrees — the number that says whether a
/// pass is overhead or grazing, which is what the edge shift turns out to depend on.
fn peak_elevation(w: &Window) -> f64 {
    let (cell, zenith) = station();
    let sat_at = iss();
    let detector = ElevationDetector::new(cell, zenith, 0.0);
    (0..=400)
        .map(|i| {
            let t = w.rise + w.duration() * f64::from(i) / 400.0;
            detector.g(sat_at(t))
        })
        .fold(f64::NEG_INFINITY, f64::max)
        .to_degrees()
}

#[test]
fn refraction_opens_every_pass_early_and_closes_it_late() {
    for mask_deg in [0.0_f64, 5.0, 10.0] {
        let mask = mask_deg.to_radians();
        let straight = passes(mask, Ray::Straight);
        let bent = passes(mask, Ray::Bent);
        // Equal counts are a property of this pass set, not a theorem: a pass peaking within `R(mask)`
        // of the mask exists refracted and does not exist geometrically. None here does — the closest
        // is 2.06° against 34′ — so the edges pair up and can be differenced.
        assert_eq!(
            straight.len(),
            bent.len(),
            "refraction changed the pass count at {mask_deg}°: a pass peaks within a refraction of \
             the mask, so there is no geometric edge to difference against"
        );

        println!("\nmask {mask_deg:.0}°, {} passes", straight.len());
        println!("  peak elevation |  Δrise  |  Δset");
        let (mut worst_rise, mut worst_set, mut peak_of_worst) = (0.0_f64, 0.0_f64, f64::NAN);
        for (s, b) in straight.iter().zip(&bent) {
            let peak = peak_elevation(s);
            // A window clamped to the scan bound has no crossing to compare; skip that edge only.
            let d_rise = (s.rise > 0.0 && b.rise > 0.0).then_some(s.rise - b.rise);
            let d_set = (s.set < SPAN && b.set < SPAN).then_some(b.set - s.set);
            println!(
                "  {peak:13.2}° | {:7} | {:7}",
                d_rise.map_or_else(|| "  --  ".to_string(), |d| format!("{d:6.3}s")),
                d_set.map_or_else(|| "  --  ".to_string(), |d| format!("{d:6.3}s")),
            );

            // One-sided by construction: refraction lifts the satellite, so it can only bring a rise
            // forward and push a set back. A violation would mean the transform is not monotone.
            if let Some(d) = d_rise {
                assert!(
                    d > 0.0,
                    "refraction delayed a rise at {mask_deg}° by {d:.3} s"
                );
                if d > worst_rise {
                    worst_rise = d;
                    peak_of_worst = peak;
                }
            }
            if let Some(d) = d_set {
                assert!(
                    d > 0.0,
                    "refraction advanced a set at {mask_deg}° by {d:.3} s"
                );
                worst_set = worst_set.max(d);
            }
        }
        println!("  worst Δrise {worst_rise:.3} s, on the pass peaking at {peak_of_worst:.2}°");

        // The shift is `R(mask)/(dε/dt)` at the crossing, so it is set by how steeply the pass climbs
        // through the mask — not by the refraction alone. A grazing pass crosses almost tangentially
        // and pays the whole of it; an overhead pass crosses steeply and pays a fraction. That is why
        // the bands are wide: they bound a quantity whose worst case is a property of the pass set.
        let bound = if mask_deg == 0.0 { 30.0 } else { 12.0 };
        assert!(
            worst_rise > 0.2 && worst_rise < bound,
            "rise shift {worst_rise:.3} s at {mask_deg}° outside the measured band"
        );
        assert!(
            worst_set > 0.2 && worst_set < bound,
            "set shift {worst_set:.3} s at {mask_deg}° outside the measured band"
        );
    }
}

#[test]
fn a_refracted_mask_is_the_same_answer_as_a_refracted_elevation() {
    for mask_deg in [0.0_f64, 5.0, 10.0] {
        let mask = mask_deg.to_radians();
        let bent = passes(mask, Ray::Bent);
        let shifted = passes(mask, Ray::ShiftedMask);
        assert_eq!(
            bent.len(),
            shifted.len(),
            "pass count differs at {mask_deg}°: {} against {}",
            bent.len(),
            shifted.len()
        );

        let (mut worst, mut worst_arcsec) = (0.0_f64, 0.0_f64);
        for (b, s) in bent.iter().zip(&shifted) {
            for (lhs, rhs) in [(b.rise, s.rise), (b.set, s.set)] {
                let gap = (lhs - rhs).abs();
                worst = worst.max(gap);
                worst_arcsec =
                    worst_arcsec.max(gap * elevation_rate(lhs).abs().to_degrees() * 3600.0);
            }
        }
        println!(
            "mask {mask_deg:4.0}°: pointwise vs shifted-mask edges differ by {worst:.2e} s, \
             {worst_arcsec:.2}\" of elevation"
        );

        // The two conditions are the same condition, so the gap is not an approximation error to be
        // bounded — it is the *threshold* difference between the two published fits, which are not
        // exact inverses of each other. In seconds it is whatever the pass's elevation rate makes of
        // it, so the scale-free form is the one to assert: converted back through `dε/dt` it is a few
        // arcseconds at every mask, matching the round-trip residual `observability.rs` documents.
        assert!(
            worst_arcsec < 6.0,
            "the two routes disagree by {worst_arcsec:.2}\" at {mask_deg}°, which is more than the \
             Saemundsson/Bennett round trip: the mask shift is not reproducing the pointwise answer"
        );
    }
}
