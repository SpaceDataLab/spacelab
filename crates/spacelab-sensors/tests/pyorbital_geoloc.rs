//! Comparator: `spacelab-sensors`' instrument catalogue and direct sensor model, against pyorbital.
//! Benchmark material, not a gate. The catalogue's closed forms are decided by
//! `derivations/scan_catalogue.mac` (`tests/scan_gates.rs`), the line of sight's hit on the
//! ellipsoid by `derivations/ray_ellipsoid_geolocation.mac` (`src/geoloc.rs`, G1/G2).
//!
//! Both fixtures come from `tools/gen_pyorbital_oracle.py`; see its header for provenance and for
//! why there are two of them. The short version is that a port fails by transcription, not by
//! physics, and a single end-to-end residual lets a mistyped scan angle hide inside a geolocation
//! difference. So the catalogue is checked with no orbit in the loop, and the geometry is checked on
//! **pyorbital's own exported state vectors** — the residual cannot absorb a disagreement between
//! two SGP4s, because only one SGP4 ran.
//!
//! # Two places the two implementations are expected to differ, and by how much
//!
//! * **pyorbital quantises pixel times to the nanosecond.** `ScanGeometry.__init__` multiplies the
//!   seconds array by `np.timedelta64(1000000000, "ns")`, which truncates: AVHRR's `1/6` s scan
//!   period comes out of the fixture as `1.666 666 66e-1`, short by 6.7e-10 s. At 7.5 km/s that is
//!   7.5 µm of platform travel, so it is not worth reproducing — but it is worth *naming*, because
//!   otherwise the tolerance below looks arbitrary.
//! * **pyorbital uses two ellipsoids in one computation**, GRS80's semi-minor axis in `subpoint` and
//!   WGS-84's in the ray intersection; this port uses WGS-84 throughout. It also stops its
//!   geodetic-latitude iteration at `np.allclose`'s default `rtol = 1e-5`, where this one runs to
//!   the `f64` fixed point. Both land in the geolocation residual and both have been measured
//!   separately — 0.9 µm for the axis, 0.24 mm for the convergence — so the residual is essentially
//!   all convergence, and it is pyorbital's rather than this port's. The numbers sit beside the
//!   assertion that uses them.

#![allow(clippy::expect_used, clippy::unwrap_used)] // a malformed fixture should stop the run

use spacelab_core::Vec3;
use spacelab_sensors::geoloc::{Attitude, WGS84, geolocate};
use spacelab_sensors::scan::{Instrument, ScanSample};

fn instrument(name: &str) -> Instrument {
    match name {
        "avhrr" => Instrument::Avhrr,
        "avhrr_gac" => Instrument::AvhrrGac,
        "viirs" => Instrument::Viirs,
        "amsua" => Instrument::AmsuA,
        "mhs" => Instrument::Mhs,
        "hirs4" => Instrument::Hirs4,
        "atms" => Instrument::Atms,
        "mwhs2" => Instrument::Mwhs2,
        "olci" => Instrument::Olci,
        "ascat" => Instrument::Ascat,
        "slstr_nadir" => Instrument::SlstrNadir,
        other => panic!("{other} is not in the catalogue"),
    }
}

fn rows(path: &str) -> Vec<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("{path} — see tools/gen_pyorbital_oracle.py"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(str::to_owned).collect())
        .collect()
}

/// The distances to pyorbital's scan angles and pixel times, every instrument of the catalogue.
///
/// pyorbital truncates its time array to integer nanoseconds; nothing else in the scan geometry is
/// quantised, so the time distance is at most a nanosecond and the angle distance is arithmetic.
#[test]
#[ignore = "comparator: the distance to pyorbital, not a verdict"]
fn scan_catalogue_distance_to_pyorbital() {
    let rows = rows("tests/fixtures/pyorbital_scan.txt");
    assert!(rows.len() > 700, "fixture has only {} rows", rows.len());

    let mut angles = Vec::new();
    let mut times = Vec::new();
    let mut seen: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    for row in &rows {
        let name = row[0].as_str();
        let (scan, line, pixel) = (
            row[1].parse::<u32>().unwrap(),
            row[2].parse::<u32>().unwrap(),
            row[3].parse::<u32>().unwrap(),
        );
        let (roll, pitch, time) = (
            row[4].parse::<f64>().unwrap(),
            row[5].parse::<f64>().unwrap(),
            row[6].parse::<f64>().unwrap(),
        );
        seen.insert(name);
        let ours = instrument(name)
            .sample(scan, line, pixel)
            .unwrap_or_else(|| panic!("{name} refused ({scan}, {line}, {pixel})"));
        angles.push((ours.roll - roll).abs());
        angles.push((ours.pitch - pitch).abs());
        times.push((ours.time_offset - time).abs());
    }
    assert_eq!(seen.len(), 11, "the fixture covers {seen:?}, not all 11");
    comparison_row("scan roll and pitch, 11 instruments", "rad", &angles);
    comparison_row("pixel time offsets, 11 instruments", "s", &times);
}

/// The distances to pyorbital's ground points, on pyorbital's own state vectors.
///
/// Two known differences account for the 1.1 mm: swapping the semi-minor axis from GRS80's to
/// WGS-84's moves the boresight by 1.04e-12 rad, 0.9 µm at an 830 km nadir range; running the
/// geodetic-latitude iteration to the `f64` fixed point instead of stopping at `np.allclose`'s
/// `rtol = 1e-5` moves it by 2.86e-10 rad, 0.24 mm at that range and 1.8 mm within the subpoint
/// itself. VIIRS is the worst case: at a 56.28° look angle the slant range is more than twice the
/// nadir one.
#[test]
#[ignore = "comparator: the distance to pyorbital, not a verdict"]
fn direct_sensor_model_distance_to_pyorbital() {
    let rows = rows("tests/fixtures/pyorbital_geoloc.txt");
    assert!(rows.len() > 150, "fixture has only {} rows", rows.len());

    let distances: Vec<f64> = rows
        .iter()
        .map(|row| {
            let name = row[0].as_str();
            let v: Vec<f64> = row[1..].iter().map(|f| f.parse().unwrap()).collect();
            let sample = ScanSample {
                roll: v[0],
                pitch: v[1],
                time_offset: 0.0, // the state vector is already at this pixel's own instant
            };
            let position = Vec3::new(v[2], v[3], v[4]);
            let velocity = Vec3::new(v[5], v[6], v[7]);
            let want = Vec3::new(v[8], v[9], v[10]);
            let got = geolocate(position, velocity, sample, Attitude::default(), WGS84)
                .unwrap_or_else(|| panic!("{name}: our line of sight missed the ellipsoid"));
            // kilometres to metres
            (got - want).norm() * 1e3
        })
        .collect();
    comparison_row(
        "ground points of the direct sensor model, 11 instruments",
        "m",
        &distances,
    );
}

/// One `comparison | ...` row of `tools/compare.py`.
fn comparison_row(quantity: &str, unit: &str, distances: &[f64]) {
    let max = distances.iter().copied().fold(0.0, f64::max);
    #[expect(
        clippy::cast_precision_loss,
        reason = "a count of a few thousand distances"
    )]
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | pyorbital | not recorded | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}
