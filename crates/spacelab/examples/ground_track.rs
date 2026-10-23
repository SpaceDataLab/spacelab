//! The ISS ground track, with the Earth's orientation taken from measurement.
//!
//!     tools/fetch-eop.sh
//!     cargo run -p spacelab --example ground_track
//!
//! Every stage is checked against an independent implementation:
//!
//!     TLE ─SGP4─▶ TEME ─GMST82─▶ PEF ─polar motion─▶ ITRF ─Vermeille─▶ lat, lon, h
//!
//! Neglecting `xp`, `yp` and `ΔUT1` moves this track by 237 metres. The numbers
//! below are not a demonstration of a formula; they are where the station was.

#![allow(clippy::expect_used)] // an example that cannot run should say so loudly

use spacelab::{
    EopSource, earth::itrf_to_geodetic, eop::EopSeries, sgp4::Satellite, teme_to_itrf_at,
};
use spacelab_core::UtcDateTime;

fn main() {
    let snapshot = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/eop");
    let path = std::fs::read_dir(&snapshot)
        .unwrap_or_else(|e| panic!("run tools/fetch-eop.sh first ({}): {e}", snapshot.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "txt"))
        .expect("an Earth-orientation snapshot");

    let series = EopSeries::parse_finals2000a(&std::fs::read_to_string(&path).expect("readable"))
        .expect("the published series parses");
    let (first, last) = series.span();
    println!(
        "earth orientation: {} (MJD {first}..{last})\n",
        path.display()
    );

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let satellite = Satellite::from_tle(line1, line2).expect("the ISS element set parses");

    let start = UtcDateTime::new(2008, 9, 20, 12, 0, 0.0).expect("a real instant");

    println!(
        "{:^26} {:>11} {:>12} {:>12}  EOP",
        "UTC", "lat °", "lon °", "height km"
    );
    for step in 0..20 {
        let ns = start.naive_ns_since_j2000() + i128::from(step) * 5 * 60_000_000_000;
        let utc = UtcDateTime::from_naive_ns_since_j2000(ns, false);

        let teme = satellite
            .propagate_at(utc)
            .expect("the ISS has not decayed");
        let (transform, quality) =
            teme_to_itrf_at(utc, EopSource::Series(&series)).expect("2008 is tabulated");
        let geodetic = itrf_to_geodetic(transform.state(teme).position).expect("in orbit");

        println!(
            "{utc} {:>11.5} {:>12.5} {:>12.2}  {quality:?}",
            geodetic.latitude.degrees(),
            geodetic.longitude.degrees(),
            geodetic.height.metres() / 1000.0,
        );
    }
}
