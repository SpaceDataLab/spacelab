//! What the OMM encodings carry that the TLE cannot, measured on one Celestrak query.
//!
//! The three fixtures are the same request — Celestrak's `GROUP=stations` general-perturbations
//! set, retrieved 2026-08-14 — served as `CSV`, `KVN` and `3LE`. Same objects, same epochs, same
//! fit: the only thing that varies is how many digits the encoding can hold. That makes the triple
//! a controlled comparison rather than three unrelated files.
//!
//! The query turned out to contain a six-digit object: `SOYUZ-MS 29`, catalog **100057**. It appears
//! in the CSV and the KVN and is **absent from the 3LE**, which carries 21 of the 22 records and says
//! nothing about the one it dropped. What that omission is *not* is a limit of this workspace's
//! reader: `from_tle` decodes Alpha-5 through the vendor crate's `parse_norad_id`, so `A0057` reads
//! back as 100057 — pinned below, because it was neither tested nor documented here before.
//!
//! Why Celestrak served this particular query without the object, rather than in Alpha-5, is not
//! settled by these files, and re-querying to find out is bounded by Celestrak's own usage policy
//! (GP data no more often than every two hours). The measurement stands on its own either way: the
//! two encodings of one request do not carry the same catalogue.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::{Convention, Satellite, parse_omm_csv, parse_omm_kvn};

const CSV: &str = include_str!("fixtures/omm_stations.csv");
const KVN: &str = include_str!("fixtures/omm_stations.kvn");
const TLE: &str = include_str!("fixtures/omm_stations.tle");

/// The catalog number of the object the 3LE fixture does not carry.
const SIX_DIGIT: u64 = 100_057;

/// The 3LE fixture as `(line1, line2)` pairs, in file order.
fn tle_pairs() -> Vec<(&'static str, &'static str)> {
    let lines: Vec<&str> = TLE.lines().filter(|l| !l.trim().is_empty()).collect();
    lines.chunks(3).map(|c| (c[1], c[2])).collect()
}

/// The two OMM encodings are the same record served twice, so they must agree field for field.
///
/// This includes the free-text fields, which is what distinguishes a KVN reader that understands the
/// keywords from one that silently drops the ones it does not recognise.
#[test]
fn the_two_omm_encodings_of_one_query_are_identical() {
    let csv = parse_omm_csv(CSV).expect("CSV parses");
    let kvn = parse_omm_kvn(KVN).expect("KVN parses");
    assert_eq!(csv.len(), 22, "records in the fixture");
    assert_eq!(csv, kvn, "the two OMM encodings disagree");
}

/// The 3LE encoding of the *same query* is short by exactly the six-digit object.
///
/// A caller that ingests the 3LE gets 21 of 22 objects and no error: from inside the format the
/// omission is invisible.
#[test]
fn the_tle_encoding_of_the_same_query_silently_omits_the_six_digit_object() {
    let omm = parse_omm_csv(CSV).expect("CSV parses");
    let tle = tle_pairs();
    assert_eq!(tle.len(), omm.len() - 1, "3LE against OMM record count");

    let mut from_tle: Vec<u64> = tle
        .iter()
        .map(|(line1, line2)| {
            let satellite = Satellite::from_tle(line1, line2).expect("the 3LE parses");
            u64::from(satellite.norad_id().get())
        })
        .collect();
    let mut from_omm: Vec<u64> = omm.iter().map(|e| e.norad_id).collect();
    from_tle.sort_unstable();
    from_omm.sort_unstable();

    let missing: Vec<u64> = from_omm
        .iter()
        .copied()
        .filter(|id| !from_tle.contains(id))
        .collect();
    assert_eq!(missing, vec![SIX_DIGIT], "objects the 3LE does not carry");

    // And the object is genuinely propagatable — its absence is a limit of the encoding, not a
    // property of the orbit.
    let elements = omm
        .iter()
        .find(|e| e.norad_id == SIX_DIGIT)
        .expect("the six-digit object");
    assert_eq!(
        elements.international_designator.as_deref(),
        Some("2026-162A")
    );
    let satellite =
        Satellite::from_elements(elements, Convention::default()).expect("OMM elements usable");
    let r = satellite
        .propagate_minutes(0.0)
        .expect("propagates")
        .position
        .vector
        .norm();
    assert!((6.5e6..8.0e6).contains(&r), "radius {r:.0} m");
}

/// Alpha-5 is already readable, which is worth pinning because nothing said so.
///
/// The five columns are re-read with the leading character as a base-34 digit — `A` is 10, with `I`
/// and `O` skipped so they cannot be confused with 1 and 0 — which buys 240 000 slots past 99 999
/// without widening the line. `sgp4::parse_norad_id` implements it and [`Satellite::from_tle`] gets
/// it for free, so the ticket's premise that a six-digit object "simply cannot be read today" holds
/// only for the *legacy* encoding.
///
/// It is still not a substitute for OMM: Alpha-5 caps at 339 999, it cannot express the fitted
/// parameters at source precision, and — as the fixture above shows — a producer is free to serve
/// neither Alpha-5 nor the object.
#[test]
fn the_tle_reader_already_decodes_an_alpha_5_catalog_number() {
    // The same ISS element set, renumbered: 100057 = 10·10⁴ + 0057, and 10 is `A`.
    let line1 = "1 A0057U 26162A   26226.82569811  .00005167  00000+0  10032-3 0  9995";
    let line2 = "2 A0057  51.6328   9.8801 0007567  46.9314 313.2307 15.49444701580848";
    let satellite = Satellite::from_tle_unchecked(line1, line2).expect("Alpha-5 parses");
    assert_eq!(u64::from(satellite.norad_id().get()), SIX_DIGIT);
}

/// Why the *legacy* five-digit encoding has no room for the object at all.
///
/// Columns 3–7 of both TLE lines hold exactly five characters. Writing a sixth digit displaces every
/// field after it by one place, the checksum column included, and the result is a 70-column line
/// that is not a TLE at all. There is no wrong answer to compare against here — there is nothing to
/// hand the parser.
#[test]
fn a_sixth_digit_displaces_every_column_after_it() {
    let (line1, line2) = tle_pairs()[0];
    assert_eq!(line1.len(), 69, "the fixture is a well-formed TLE");
    let widened1 = line1.replace("25544", "270123");
    let widened2 = line2.replace("25544", "270123");
    assert_eq!(widened1.len(), 70, "the sixth digit displaces a column");
    assert!(
        Satellite::from_tle_unchecked(&widened1, &widened2).is_err(),
        "a 70-column line must not parse as a TLE"
    );
}

/// What the TLE's fixed-width fields cost on the objects it *can* carry, as a distance.
///
/// Two of the fitted quantities are visibly truncated on the ISS record alone: `BSTAR` is
/// `.10032188E-3` in OMM against `10032-3` in the TLE, and the eccentricity is `.00075677` against
/// `0007567`. Nothing about the resulting orbit looks wrong — the difference is a smooth, growing
/// offset, which is exactly why it needs a number rather than a description.
///
/// Measured over the 21 shared objects at 1-minute steps across ±24 hours of the epoch: **max
/// position difference 1.5230e0 m, max velocity difference 7.9401e-4 m/s**. The bounds below sit an
/// order of magnitude above that, because the quantity is a property of *these* elements and another
/// catalogue would land elsewhere; what the assertions pin is the sign of the claim — metres over a
/// day, and not zero either.
#[test]
fn the_tle_and_the_omm_disagree_by_what_the_fixed_width_fields_dropped() {
    let omm = parse_omm_csv(CSV).expect("CSV parses");

    let mut worst_position = 0.0_f64;
    let mut worst_velocity = 0.0_f64;
    let mut compared = 0_usize;
    for (line1, line2) in tle_pairs() {
        let from_tle = Satellite::from_tle(line1, line2).expect("the 3LE parses");
        // Pair by catalog number, not by file position: the two encodings do not carry the same
        // number of records, so index alignment would compare different satellites.
        let id = u64::from(from_tle.norad_id().get());
        let elements = omm
            .iter()
            .find(|e| e.norad_id == id)
            .unwrap_or_else(|| panic!("catalog {id} is in the 3LE but not the OMM"));
        let from_omm =
            Satellite::from_elements(elements, Convention::default()).expect("OMM elements usable");
        compared += 1;

        for step in -1440..=1440 {
            let minutes = f64::from(step);
            let (Ok(a), Ok(b)) = (
                from_omm.propagate_minutes(minutes),
                from_tle.propagate_minutes(minutes),
            ) else {
                continue;
            };
            worst_position = worst_position.max((a.position.vector - b.position.vector).norm());
            worst_velocity = worst_velocity.max((a.velocity.vector - b.velocity.vector).norm());
        }
    }
    assert_eq!(compared, 21, "objects present in both encodings");

    // Not zero: if the two encodings agreed exactly, the test would be comparing the TLE with
    // itself and the whole premise of preferring OMM would be unsupported.
    assert!(
        worst_position > 1.0e-3,
        "OMM and TLE agree to {worst_position:.4e} m — the fixtures are not independent"
    );
    assert!(
        worst_position < 1.5e1,
        "OMM against TLE: {worst_position:.4e} m"
    );
    assert!(
        worst_velocity < 1.0e-2,
        "OMM against TLE: {worst_velocity:.4e} m/s"
    );
}

/// Every OMM record in the fixture has to survive the whole path, not just the parser.
#[test]
fn every_record_reaches_a_propagatable_satellite() {
    for elements in parse_omm_kvn(KVN).expect("KVN parses") {
        let satellite = Satellite::from_elements(&elements, Convention::default())
            .unwrap_or_else(|e| panic!("catalog {}: {e}", elements.norad_id));
        let state = satellite
            .propagate_minutes(0.0)
            .unwrap_or_else(|e| panic!("catalog {}: {e}", elements.norad_id));
        let r = state.position.vector.norm();
        assert!(
            (6.5e6..8.0e6).contains(&r),
            "catalog {}: radius {r:.0} m is not a station-keeping LEO",
            elements.norad_id
        );
    }
}
