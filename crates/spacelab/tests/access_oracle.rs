//! ISS access windows over a ground station: every edge a root of the elevation mask, and what
//! Earth orientation is worth on one.
//!
//! The definition decides (plans/17 Ф19.G). A pass edge is an instant where the elevation,
//! `asin(d·ẑ/|d|)` for the line of sight `d`, crosses the mask: at every edge the elevation equals
//! the mask to within the scan's abscissa tolerance, the window lies above it and the instants
//! either side below it, and a sample every five seconds of the day finds the satellite above the
//! mask only inside a window (G2). Earth orientation moves each edge by the elevation change it
//! makes there over the elevation's rate, the implicit-function theorem's first order (S1).
//!
//! Orekit's `ElevationDetector` decided the edges until then, the plan's C2 criterion
//! (`tools/orekit-oracle/OrekitAccessOracle.java`, fixture below); it is a comparator now, parked
//! below and published in `docs/comparisons.md`. Orekit's SGP4 shares Vallado's lineage with the
//! `sgp4` crate, but its topocentric frame and event root-finding are its own. It reads its bundled
//! `finals2000A` (`FramesFactory.getITRF(IERS_2010, simpleEOP = true)`), so the comparison is run
//! both ways: once with the same IERS series interpolated by `EopSeries::at_mjd`, and once with
//! [`EopSource::NoEop`]. At this epoch `ΔUT1` is −0.481 s, which turns the Earth by 7.2 arcsec,
//! about 223 m at the equator, and a ground track sweeping at ~6.9 km/s crosses that in tens of
//! milliseconds.

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

/// The elevation mask, degrees.
const MASK_DEG: f64 = 10.0;
/// The scan's abscissa tolerance, s.
const TOLERANCE: f64 = 1e-4;
/// The scan, seconds past 2008-09-20 12:00 UTC.
const SPAN: f64 = 86_400.0;

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

/// The ISS over the fixture's station, (35°N, 33°E, 0), with Earth orientation from `source`.
struct Scan<'a> {
    satellite: Satellite,
    source: EopSource<'a>,
    start_ns: i128,
    cell: Vec3,
    zenith: Vec3,
}

impl<'a> Scan<'a> {
    fn new(source: EopSource<'a>) -> Self {
        let g = Geodetic {
            latitude: Angle::from_degrees(35.0),
            longitude: Angle::from_degrees(33.0),
            height: Length::from_metres(0.0),
        };
        let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
        let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
        Self {
            satellite: Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses"),
            source,
            start_ns: UtcDateTime::new(2008, 9, 20, 12, 0, 0.0)
                .unwrap()
                .naive_ns_since_j2000(),
            cell: geodetic_to_itrf(g).vector,
            zenith: Vec3::new(cla * clo, cla * slo, sla),
        }
    }

    /// The satellite in ITRF, `t` seconds past the scan start.
    fn satellite_at(&self, t: f64) -> Vec3 {
        let ns = self.start_ns + (t * 1e9) as i128;
        let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
        let teme = self
            .satellite
            .propagate_at(label)
            .expect("the ISS has not decayed");
        let (transform, _) =
            teme_to_itrf_at(label, self.source).expect("the scan lies inside the excerpt");
        transform.state(teme).position.vector
    }

    /// The elevation from its definition, `asin(d·ẑ/|d|)`, rad.
    fn elevation(&self, t: f64) -> f64 {
        let d = self.satellite_at(t) - self.cell;
        (d.dot(self.zenith) / d.norm()).asin()
    }

    /// The elevation's rate by a central difference, rad/s.
    fn elevation_rate(&self, t: f64) -> f64 {
        let h = 0.05;
        (self.elevation(t + h) - self.elevation(t - h)) / (2.0 * h)
    }

    fn passes(&self) -> Vec<Window> {
        let mask = ElevationMask::new(MASK_DEG.to_radians()).unwrap();
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
            self.cell,
            self.zenith,
            |t| self.satellite_at(t),
            0.0,
            SPAN,
            TOLERANCE,
            |w| mine.push(w),
        );
        mine
    }
}

/// (G2) Every edge the scan reports is a root of `ε(t) − ε_min`: the elevation there equals the mask
/// to within the abscissa tolerance, rising at a rise and falling at a set; the window's midpoint is
/// above the mask and a second either side of it below; and no sample of the day lies above the
/// mask outside a window.
#[test]
fn every_pass_edge_is_a_root_of_the_elevation_mask() {
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let scan = Scan::new(EopSource::Series(&series));
    let windows = scan.passes();
    let mask = MASK_DEG.to_radians();
    assert!(!windows.is_empty(), "no pass in a day");

    let mut worst = 0.0_f64;
    for w in &windows {
        assert!(
            0.0 < w.rise && w.rise < w.set && w.set < SPAN,
            "a window clamped to the scan or empty: {w:?}"
        );
        let (rate_rise, rate_set) = (scan.elevation_rate(w.rise), scan.elevation_rate(w.set));
        assert!(
            rate_rise > 0.0 && rate_set < 0.0,
            "not a rise and a set: {w:?}"
        );
        for (t, rate) in [(w.rise, rate_rise), (w.set, rate_set)] {
            worst = worst.max(((scan.elevation(t) - mask) / rate).abs());
        }
        assert!(scan.elevation(0.5 * (w.rise + w.set)) > mask, "{w:?}");
        assert!(scan.elevation(w.rise - 1.0) < mask, "{w:?}");
        assert!(scan.elevation(w.set + 1.0) < mask, "{w:?}");
    }

    let mut missed = 0;
    for k in 0..=17_280_u32 {
        let t = 5.0 * f64::from(k);
        let inside = windows.iter().any(|w| w.rise <= t && t <= w.set);
        if !inside && scan.elevation(t) > mask {
            missed += 1;
        }
    }
    println!(
        "{} passes; the worst edge lies {worst:.3e} s from its root",
        windows.len()
    );
    assert_eq!(missed, 0, "samples above the mask outside every window");
    // Measured 2.5e-5 s; the bound is the abscissa tolerance the scan was given, four times that.
    assert!(worst < TOLERANCE, "an edge {worst} s from its root");
}

/// (S1) What Earth orientation is worth on a pass edge, as a measurement: the same scan with the
/// tabulated series and with none. Each edge moves by `−Δε/ε̇`, the elevation change the
/// orientation makes there over the elevation's rate, to the scan's tolerance; the worst move is
/// tens of milliseconds.
#[test]
fn earth_orientation_is_worth_tens_of_milliseconds_on_a_pass_edge() {
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let with = Scan::new(EopSource::Series(&series));
    let without = Scan::new(EopSource::NoEop);
    let (a, b) = (with.passes(), without.passes());
    assert_eq!(a.len(), b.len(), "EOP must not change the pass count");

    let (mut moved, mut worst) = (0.0_f64, 0.0_f64);
    for (wa, wb) in a.iter().zip(&b) {
        for (ta, tb) in [(wa.rise, wb.rise), (wa.set, wb.set)] {
            let predicted = -(without.elevation(ta) - with.elevation(ta)) / with.elevation_rate(ta);
            moved = moved.max((tb - ta).abs());
            worst = worst.max((tb - ta - predicted).abs());
        }
    }
    println!(
        "Earth orientation moves a pass edge by up to {moved:.6} s; the first-order prediction \
         misses by {worst:.3e} s"
    );
    assert!(
        (0.01..0.1).contains(&moved),
        "Earth orientation moved an edge by {moved} s"
    );
    // Measured 1.5e-5 s, the two scans' root errors (2.5e-5 s at worst, the G2 test above); the
    // second order, Δt² ε̈ / 2ε̇, is 2.2e-6 s at worst on these six edges.
    assert!(worst < 6.0e-5, "an edge moved {worst} s off its prediction");
}

fn comparison_row(quantity: &str, unit: &str, distances: &[f64]) {
    let max = distances.iter().copied().fold(0.0, f64::max);
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | Orekit | 13.1.7 | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// The distance to Orekit's `ElevationDetector` edges, with the IERS series both sides read and
/// with none on ours. What survives the tabulated run is not Earth orientation, and it is not
/// root-finder convergence either: dropping the abscissa tolerance from `1e-4` to `1e-9` moves it
/// from 82 µs to 107 µs rather than reducing it. The floor is the TEME-to-ITRF path (a GMST
/// rotation here against Orekit's full IERS chain) and the two SGP4 transcriptions; which of them
/// dominates is not measured here.
#[test]
#[ignore = "comparator: the distance to Orekit, not a verdict"]
fn iss_pass_edges_distance_to_orekit() {
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let orekit = orekit_passes();
    for (quantity, source) in [
        (
            "ISS pass edges, mask 10 deg, IERS EOP",
            EopSource::Series(&series),
        ),
        ("ISS pass edges, mask 10 deg, no EOP", EopSource::NoEop),
    ] {
        let mine = Scan::new(source).passes();
        if mine.len() != orekit.len() {
            println!("pass count: ours {}, Orekit's {}", mine.len(), orekit.len());
        }
        let distances: Vec<f64> = mine
            .iter()
            .zip(&orekit)
            .flat_map(|(w, &(rise, set))| [(w.rise - rise).abs(), (w.set - set).abs()])
            .collect();
        comparison_row(quantity, "s", &distances);
    }
}
