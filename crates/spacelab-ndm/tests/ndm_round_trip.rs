//! Write → read → compare, field by field, against every message the CCSDS annexes publish.
//!
//! This is the test that makes the writers worth having. A writer nobody reads back is a formatter;
//! a writer whose output re-parses to the value it was given is an interchange format.
//!
//! The comparison is **exact** — `assert_eq!` on the parsed structures, which compares `f64` by
//! value — everywhere except the Keplerian angles. That split is a property of the unit conversions
//! rather than of the code:
//!
//! * position, velocity, covariance and `GM` scale by a power of ten, and `(k · 1000) / 1000 = k`
//!   holds on every double that was itself produced by scaling a decimal, which is what a value read
//!   from a file always is;
//! * an angle scales by `π / 180`, which is not a power of ten, so degrees → radians → degrees is a
//!   double rounding and the last bit is not guaranteed.
//!
//! The angle bound is asserted as a number rather than waved at, so that a real regression in the
//! conversion is still a failure rather than "well, angles".

#![allow(clippy::unwrap_used, clippy::float_cmp)]

use spacelab_ndm::{Keplerian, Oem, Opm, parse_oem, parse_opm, write_oem, write_opm};

/// Relative tolerance on an angle that has been through degrees → radians → degrees.
///
/// Two roundings of a value whose exponent does not change, so a handful of ulp; `4 · f64::EPSILON`
/// is 8.9e-16 and leaves room without hiding anything a wrong conversion factor would do.
const ANGLE_TOLERANCE: f64 = 4.0 * f64::EPSILON;

const OPMS: [(&str, &str); 3] = [
    (
        "figure G-1, state only",
        include_str!("fixtures/ccsds_502_g1.opm"),
    ),
    (
        "figure G-2, elements and two manoeuvres",
        include_str!("fixtures/ccsds_502_g2.opm"),
    ),
    (
        "figure G-4, elements, covariance and a user-defined keyword",
        include_str!("fixtures/ccsds_502_g4.opm"),
    ),
];

const OEMS: [(&str, &str); 3] = [
    (
        "figure G-11, two segments",
        include_str!("fixtures/ccsds_502_g11.oem"),
    ),
    (
        "figure G-12, with accelerations",
        include_str!("fixtures/ccsds_502_g12.oem"),
    ),
    (
        "figure G-13, with two covariance matrices",
        include_str!("fixtures/ccsds_502_g13.oem"),
    ),
];

/// The same OPM with its Keplerian angles zeroed, so the rest can be compared exactly.
fn without_angles(opm: &Opm) -> Opm {
    let mut stripped = opm.clone();
    if let Some(keplerian) = stripped.keplerian.as_mut() {
        keplerian.inclination = spacelab_core::Angle::from_radians(0.0);
        keplerian.right_ascension = spacelab_core::Angle::from_radians(0.0);
        keplerian.argument_of_pericenter = spacelab_core::Angle::from_radians(0.0);
        keplerian.anomaly = match keplerian.anomaly {
            spacelab_ndm::Anomaly::True(_) => {
                spacelab_ndm::Anomaly::True(spacelab_core::Angle::from_radians(0.0))
            }
            spacelab_ndm::Anomaly::Mean(_) => {
                spacelab_ndm::Anomaly::Mean(spacelab_core::Angle::from_radians(0.0))
            }
        };
    }
    stripped
}

fn angles(keplerian: &Keplerian) -> [f64; 4] {
    let anomaly = match keplerian.anomaly {
        spacelab_ndm::Anomaly::True(angle) | spacelab_ndm::Anomaly::Mean(angle) => angle.radians(),
    };
    [
        keplerian.inclination.radians(),
        keplerian.right_ascension.radians(),
        keplerian.argument_of_pericenter.radians(),
        anomaly,
    ]
}

#[test]
fn every_published_opm_survives_a_write_and_a_read() {
    for (label, text) in OPMS {
        let original = parse_opm(text).unwrap_or_else(|error| panic!("{label}: {error}"));
        let reparsed = parse_opm(&write_opm(&original))
            .unwrap_or_else(|error| panic!("{label}, re-read: {error}"));

        assert_eq!(
            without_angles(&original),
            without_angles(&reparsed),
            "{label}: a field other than an angle changed"
        );

        if let (Some(before), Some(after)) = (original.keplerian, reparsed.keplerian) {
            for (before, after) in angles(&before).iter().zip(angles(&after)) {
                let error = (after - before).abs() / before.abs().max(f64::MIN_POSITIVE);
                assert!(
                    error <= ANGLE_TOLERANCE,
                    "{label}: {before} became {after}, a relative move of {error:e}"
                );
            }
        }
    }
}

#[test]
fn every_published_oem_survives_a_write_and_a_read_exactly() {
    for (label, text) in OEMS {
        let original: Oem = parse_oem(text).unwrap_or_else(|error| panic!("{label}: {error}"));
        let reparsed = parse_oem(&write_oem(&original))
            .unwrap_or_else(|error| panic!("{label}, re-read: {error}"));
        // No angles anywhere in an OEM, so this one is exact with nothing carved out.
        assert_eq!(original, reparsed, "{label}");
    }
}

/// A second cycle must not move anything a first one did not.
///
/// This is what separates a writer that round-trips from one that merely converges slowly: if the
/// angle conversion drifted by an ulp per cycle, an ephemeris re-exported ten times would be ten
/// ulp out and each individual round trip would still pass the tolerance above.
#[test]
fn a_second_write_reproduces_the_first_one_byte_for_byte() {
    for (label, text) in OPMS {
        let once = write_opm(&parse_opm(text).unwrap());
        let twice = write_opm(&parse_opm(&once).unwrap());
        assert_eq!(once, twice, "{label}");
    }
    for (label, text) in OEMS {
        let once = write_oem(&parse_oem(text).unwrap());
        let twice = write_oem(&parse_oem(&once).unwrap());
        assert_eq!(once, twice, "{label}");
    }
}

/// Every keyword the fixture states and this crate reads must come back out.
///
/// A writer that silently dropped a field would still pass a structural round trip if the reader
/// dropped it too — both sides would agree on the absence. This walks the *file* instead, and holds
/// the written text to the same keyword set.
#[test]
fn no_keyword_the_reader_understands_is_lost_on_the_way_out() {
    // Keywords the crate deliberately does not carry: they are metadata about how the producer
    // computed the orbit, not the orbit, and nothing in this workspace consumes them.
    const NOT_CARRIED: [&str; 4] = [
        "OBJECT_TYPE",
        "ORIGINATOR_POC",
        "TECH_ADDRESS",
        "CENTER_NAME_EPOCH",
    ];

    for (label, text) in OPMS {
        let written = write_opm(&parse_opm(text).unwrap());
        for line in text.lines() {
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if key.starts_with("COMMENT") || value.trim().is_empty() || NOT_CARRIED.contains(&key) {
                continue;
            }
            assert!(
                written
                    .lines()
                    .any(|out| out.starts_with(&format!("{key} "))),
                "{label}: {key} is in the file and not in what was written back"
            );
        }
    }
}
