//! `spacelab-sensors`' instrument catalogue and direct sensor model, against pyorbital.
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

/// pyorbital truncates its time array to integer nanoseconds; nothing else in the scan geometry is
/// quantised, so the angles get the tolerance the arithmetic deserves and the times get 1 ns.
const NANOSECOND: f64 = 1e-9;

#[test]
fn the_catalogue_reproduces_pyorbitals_scan_angles_and_pixel_times() {
    let rows = rows("tests/fixtures/pyorbital_scan.txt");
    assert!(rows.len() > 700, "fixture has only {} rows", rows.len());

    let mut worst_angle = 0.0_f64;
    let mut worst_angle_at = String::new();
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

        for (label, got, want) in [("roll", ours.roll, roll), ("pitch", ours.pitch, pitch)] {
            let error = (got - want).abs();
            assert!(
                error < 1e-15,
                "{name} {label} at ({scan}, {line}, {pixel}): {got:.17e} against pyorbital's \
                 {want:.17e}, off by {error:.3e} rad. This is a transcription check, not an \
                 accuracy one — the two should be computing the same expression."
            );
            if error > worst_angle {
                worst_angle = error;
                worst_angle_at = format!("{name} {label} ({scan}, {line}, {pixel})");
            }
        }

        let time_error = (ours.time_offset - time).abs();
        assert!(
            time_error <= NANOSECOND,
            "{name} time at ({scan}, {line}, {pixel}): {:.17e} s against pyorbital's {time:.17e} s, \
             off by {time_error:.3e} s. Up to 1 ns is pyorbital's own truncation into \
             `timedelta64[ns]`; more than that is a different scan rate or a different sampling \
             interval.",
            ours.time_offset,
        );
    }

    assert_eq!(
        seen.len(),
        11,
        "the fixture covers {} instruments, not all 11: {seen:?}",
        seen.len()
    );
    println!("worst scan-angle difference {worst_angle:.3e} rad at {worst_angle_at}");
}

#[test]
fn the_direct_sensor_model_reproduces_pyorbitals_ground_points() {
    let rows = rows("tests/fixtures/pyorbital_geoloc.txt");
    assert!(rows.len() > 150, "fixture has only {} rows", rows.len());

    let mut worst = 0.0_f64;
    let mut worst_at = String::new();
    for row in &rows {
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

        let got = geolocate(position, velocity, sample, Attitude::default(), WGS84).unwrap_or_else(
            || {
                panic!(
                    "{name}: our line of sight missed the ellipsoid where \
                                      pyorbital's hit it at {want:?}"
                )
            },
        );
        let error = (got - want).norm();
        if error > worst {
            worst = error;
            worst_at = name.to_owned();
        }
    }

    // The gate is 5 mm because the 1.1 mm residual has been attributed rather than tolerated.
    // Measured by holding one difference and varying the other: swapping the semi-minor axis from
    // GRS80's to WGS-84's moves the boresight by 1.04e-12 rad — 0.9 **micro**metres at an 830 km
    // nadir range. Running the geodetic-latitude iteration to the `f64` fixed point instead of
    // stopping at `np.allclose`'s `rtol = 1e-5` moves it by 2.86e-10 rad, 0.24 mm at that range and
    // 1.8 mm within the subpoint itself. The second dominates by three hundred to one, and it is
    // pyorbital's convergence rather than this port's — here the iteration runs further. VIIRS is
    // the worst case for the obvious reason: at a 56.28° look angle the slant range is more than
    // twice the nadir one, and the ground incidence stretches what is left of it.
    assert!(
        worst < 5e-6,
        "worst ground-point difference {:.4} mm, on {worst_at}, against a 5 mm gate. The two known \
         differences account for 1.1 mm between them and neither can grow; a residual above this \
         is a third difference nobody has looked for.",
        worst * 1e6,
    );
    println!(
        "worst ground-point difference {:.4} mm, on {worst_at}",
        worst * 1e6
    );
}
