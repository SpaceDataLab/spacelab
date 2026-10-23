//! Access-window edges against Orekit — the plan's C2 criterion, third-lineage validated.
//!
//! Orekit's `ElevationDetector` computed the ISS rise/set times over a ground station
//! (`tools/orekit-oracle/OrekitAccessOracle.java`, fixture below); this recomputes them with
//! `access_windows` and demands they agree. Orekit's SGP4 shares Vallado's lineage with the `sgp4`
//! crate — a transcription check — but its topocentric frame and event root-finding are its own, so
//! the pass edges are an independent check of the whole coverage chain.
//!
//! Orekit reads its bundled `finals2000A` (`FramesFactory.getITRF(IERS_2010, simpleEOP = true)`),
//! so the comparison is run **both ways**: once with the same IERS series interpolated by
//! `EopSeries::at_mjd`, and once with [`EopSource::NoEop`]. The gap between the two residuals is
//! what Earth orientation is worth on a pass edge, measured rather than quoted -- at this epoch
//! `ΔUT1` is −0.481 s, which turns the Earth by 7.2 arcsec, about 223 m at the equator, and a
//! ground track sweeping at ~6.9 km/s crosses that in tens of milliseconds.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
#![allow(clippy::doc_markdown)] // SGP4, Orekit, EOP, C2 are names in prose

use spacelab::{EopSource, teme_to_itrf_at};
use spacelab_core::{Angle, Length, UtcDateTime, Vec3};
use spacelab_coverage::{Window, access_windows};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_eop::EopSeries;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sensors::{ElevationMask, OrbitBound};
use spacelab_sgp4::Satellite;

const MU: f64 = 3.986_004_418e14;
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

const FIXTURE: &str = include_str!("fixtures/orekit_iss_passes.csv");

/// The IERS series across the scan, cut from `finals2000A.all` at MJD 54725..54735 — the same
/// product Orekit's bundled data is, so a residual is not two tables disagreeing.
const EOP: &str = include_str!("fixtures/finals2000A_2008264.txt");

/// Orekit's passes as `(rise, set)` seconds past the scan start, parsed from the CSV of alternating
/// rise/set events.
fn orekit_passes() -> Vec<(f64, f64)> {
    let mut events = Vec::new();
    for line in FIXTURE.lines().skip(1) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (kind, offset) = line.split_once(',').expect("kind,offset");
        events.push((kind.to_string(), offset.parse::<f64>().expect("float")));
    }
    let mut passes = Vec::new();
    let mut open = None;
    for (kind, at) in events {
        match kind.as_str() {
            "rise" => open = Some(at),
            "set" => passes.push((open.take().expect("a set follows a rise"), at)),
            _ => panic!("unexpected event kind"),
        }
    }
    passes
}

/// Pass edges over the fixture's station, with Earth orientation from `source`.
fn passes(source: EopSource<'_>) -> Vec<Window> {
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses");

    // Station at (35°N, 33°E, 0), the fixture's geometry.
    let g = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cell = geodetic_to_itrf(g).vector;
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    let zenith = Vec3::new(cla * clo, cla * slo, sla);

    let start_ns = UtcDateTime::new(2008, 9, 20, 12, 0, 0.0)
        .unwrap()
        .naive_ns_since_j2000();
    let sat_at = |t: f64| {
        let ns = start_ns + (t * 1e9) as i128;
        let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
        let teme = satellite
            .propagate_at(label)
            .expect("the ISS has not decayed");
        let (transform, _) =
            teme_to_itrf_at(label, source).expect("the scan lies inside the excerpt");
        transform.state(teme).position.vector
    };

    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    let orbit = OrbitBound {
        semi_major_axis: 6_795e3,
        eccentricity: 0.0007,
        mu: MU,
    };

    let mut mine: Vec<Window> = Vec::new();
    access_windows(
        &mask,
        &orbit,
        EARTH_ROTATION_RATE,
        cell,
        zenith,
        sat_at,
        0.0,
        86_400.0,
        1e-4,
        |w| mine.push(w),
    );
    mine
}

/// The worst rise/set disagreement against the Orekit fixture, and the pass count.
fn worst_edge(mine: &[Window]) -> (f64, usize) {
    let orekit = orekit_passes();
    assert_eq!(
        mine.len(),
        orekit.len(),
        "pass count differs: {} vs Orekit's {}",
        mine.len(),
        orekit.len()
    );
    let mut worst = 0.0_f64;
    for (w, &(rise, set)) in mine.iter().zip(&orekit) {
        worst = worst.max((w.rise - rise).abs()).max((w.set - set).abs());
    }
    (worst, mine.len())
}

#[test]
fn iss_pass_edges_agree_with_orekit_within_a_second() {
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let mine = passes(EopSource::Series(&series));
    let orekit = orekit_passes();
    assert_eq!(
        mine.len(),
        orekit.len(),
        "pass count differs: {} vs Orekit's {}",
        mine.len(),
        orekit.len()
    );

    let mut worst = 0.0_f64;
    for (w, &(rise, set)) in mine.iter().zip(&orekit) {
        let d_rise = (w.rise - rise).abs();
        let d_set = (w.set - set).abs();
        worst = worst.max(d_rise).max(d_set);
        assert!(
            d_rise < 1.0,
            "rise off by {d_rise:.3} s (mine {:.3}, Orekit {rise:.3})",
            w.rise
        );
        assert!(
            d_set < 1.0,
            "set off by {d_set:.3} s (mine {:.3}, Orekit {set:.3})",
            w.set
        );
    }
    println!(
        "access edges vs Orekit: {} passes, worst edge disagreement {worst:.6} s",
        mine.len()
    );

    // The plan's C2 criterion is one second and the per-edge assertions above enforce it. This is
    // the floor actually reached once both sides read the same IERS series, and it is here so that
    // a regression in the frame chain shows up as a failure rather than as a still-passing second.
    assert!(
        worst < 1e-3,
        "with matched Earth orientation the edges should agree to well under a millisecond; \
         saw {worst:.6} s"
    );
}

/// What Earth orientation is worth on a pass edge, as a measurement rather than an estimate.
///
/// Both runs are the same code over the same TLE against the same Orekit fixture; the only
/// difference is the [`EopSource`]. Whatever survives the tabulated run is *not* Earth orientation.
///
/// It is also not root-finder convergence: dropping the Brent abscissa tolerance from `1e-4` to
/// `1e-9` moves the residual from 82 µs to 107 µs rather than reducing it, so the floor is
/// elsewhere -- the TEME-to-ITRF path (a GMST rotation here against Orekit's full IERS chain) and
/// the two SGP4 transcriptions. Which of those two dominates is not measured here and is not
/// claimed.
#[test]
fn earth_orientation_is_worth_tens_of_milliseconds_on_a_pass_edge() {
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let (with_eop, passes_with) = worst_edge(&passes(EopSource::Series(&series)));
    let (without, passes_without) = worst_edge(&passes(EopSource::NoEop));

    println!(
        "worst pass-edge disagreement with Orekit: {with_eop:.6} s with IERS EOP, \
         {without:.6} s with none ({passes_with} passes either way)"
    );
    assert_eq!(
        passes_with, passes_without,
        "EOP must not change pass count"
    );
    assert!(
        with_eop * 100.0 < without,
        "matching Orekit's Earth orientation should buy two orders of magnitude: \
         {with_eop:.6} s against {without:.6} s"
    );
}
