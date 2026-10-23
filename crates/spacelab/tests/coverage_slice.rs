//! The MVP vertical slice on real data: an ISS element set, propagated by SGP4 into the Earth-fixed
//! frame, its access windows over an H3 area of interest, and the revisit map written out as CSV and
//! GeoJSON — the whole pipeline the plan describes, end to end, with a real TLE.
//!
//! Earth orientation is the real IERS series over the scan, interpolated by `EopSeries::at_mjd`,
//! because the frame the windows are computed in is part of what "composes on real data" means:
//! with `xp = yp = 0` and `UT1 = UTC` this same pipeline puts a pass edge 38 ms out
//! (`access_oracle.rs`). What it still does not prove is a sub-second edge in an operational
//! sense — that needs precise ephemerides, not a TLE whose own along-track error is kilometres.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
#![allow(clippy::doc_markdown)] // SGP4, TLE, GeoJSON, H3, EOP, UT1/TT/UTC are domain terms in prose

use spacelab::{EopSource, teme_to_itrf_at};
use spacelab_core::{Angle, Length, UtcDateTime};
use spacelab_coverage::{Window, access_windows, metrics};
use spacelab_earth::Geodetic;
use spacelab_eop::EopSeries;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_grid::{
    CellCoverage, Resolution, cell_center_itrf, disk_cells, to_csv, to_geojson, to_svg,
};
use spacelab_sensors::{ElevationMask, OrbitBound};
use spacelab_sgp4::Satellite;

const MU: f64 = 3.986_004_418e14;

/// The IERS series across the scan, cut from `finals2000A.all` at MJD 54725..54735.
const EOP: &str = include_str!("fixtures/finals2000A_2008264.txt");

// ISS, epoch 2008-264 — the `sgp4` crate's own documented element set (Vallado's).
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

fn cell_zenith(g: Geodetic) -> spacelab_core::Vec3 {
    let (sla, cla) = (g.latitude.radians().sin(), g.latitude.radians().cos());
    let (slo, clo) = (g.longitude.radians().sin(), g.longitude.radians().cos());
    spacelab_core::Vec3::new(cla * clo, cla * slo, sla)
}

#[test]
#[allow(clippy::too_many_lines)] // one end-to-end integration scenario
fn the_iss_covers_a_mediterranean_aoi_over_a_day() {
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses");
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");

    // AOI near the ISS-epoch date, at a latitude the 51.6° orbit overflies.
    let aoi_center = Geodetic {
        latitude: Angle::from_degrees(35.0),
        longitude: Angle::from_degrees(33.0),
        height: Length::from_metres(0.0),
    };
    let cells = disk_cells(aoi_center, 2, Resolution::Four).unwrap(); // ~180 km AOI, 19 cells

    // Satellite position in ITRF at `t` seconds past 2008-09-20 12:00 UTC.
    let start_ns = UtcDateTime::new(2008, 9, 20, 12, 0, 0.0)
        .unwrap()
        .naive_ns_since_j2000();
    let sat_at = |t: f64| {
        let ns = start_ns + (t * 1e9) as i128;
        let label = UtcDateTime::from_naive_ns_since_j2000(ns, false);
        let teme = satellite
            .propagate_at(label)
            .expect("the ISS has not decayed");
        let (transform, _) = teme_to_itrf_at(label, EopSource::Series(&series))
            .expect("the scan lies inside the excerpt");
        transform.state(teme).position.vector
    };

    let mask = ElevationMask::new(10.0_f64.to_radians()).unwrap();
    let orbit = OrbitBound {
        semi_major_axis: 6_795e3,
        eccentricity: 0.0007,
        mu: MU,
    };
    let span = 86_400.0;

    let mut records = Vec::with_capacity(cells.len());
    let mut cells_seen = 0;
    for cell in &cells {
        let center = cell_center_itrf(*cell);
        let zenith = cell_zenith(spacelab_grid::cell_center(*cell));
        let mut windows: Vec<Window> = Vec::new();
        access_windows(
            &mask,
            &orbit,
            EARTH_ROTATION_RATE,
            center,
            zenith,
            sat_at,
            0.0,
            span,
            1e-2,
            |w| windows.push(w),
        );
        let m = metrics(&windows, 0.0, span);
        if m.passes > 0 {
            cells_seen += 1;
        }
        records.push(CellCoverage::new(
            *cell,
            m.passes,
            m.covered_seconds,
            m.max_gap,
        ));
    }

    let total: u32 = records.iter().map(|r| r.passes).sum();
    println!(
        "ISS over AOI: {cells_seen}/{} cells seen, {total} passes total",
        cells.len()
    );

    // The ISS overflies a 35°N band several times a day; every cell of a small AOI is seen.
    assert_eq!(
        cells_seen,
        cells.len(),
        "some AOI cell was never overflown by the ISS"
    );
    // Each pass is a real LEO pass — minutes, not seconds or hours.
    for r in &records {
        assert!(r.passes >= 1, "cell {} unseen", r.h3_index);
        let mean = r.coverage_seconds / f64::from(r.passes);
        assert!(
            (60.0..900.0).contains(&mean),
            "mean pass {mean:.0} s off scale for {}",
            r.h3_index
        );
    }

    // The output artefacts are well-formed.
    let csv = to_csv(&records);
    assert_eq!(
        csv.lines().count(),
        records.len() + 1,
        "CSV: header + a row per cell"
    );
    let json = to_geojson(&records);
    assert_eq!(
        json.matches("\"type\":\"Feature\"").count(),
        records.len(),
        "GeoJSON: a feature per cell"
    );
    assert!(json.starts_with("{\"type\":\"FeatureCollection\""));
    let svg = to_svg(&records);
    assert_eq!(
        svg.matches("<circle").count(),
        records.len(),
        "SVG heatmap: a disc per cell"
    );
}
