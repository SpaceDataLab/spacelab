//! The RINEX 4 reader against a real product, and against the RINEX 3 reader.
//!
//! Two questions, and they need two different oracles.
//!
//! **Does the walk stay in step?** RINEX 4's block length depends on the navigation *message*, and
//! the file interleaves eighteen of them. A reader one line out at any point reads every record
//! after it from the wrong columns, and every value it then reports is plausible and wrong. The
//! fixture is 219 verbatim lines of DLR's `BRD400DLR` product for 2026-07-19 -- one record of each
//! of the 38 (record type, constellation, message, subtype) shapes the whole 148 789-line file
//! contains -- so the walk is exercised on every transition the format actually produces, not on a
//! shape someone thought to write down.
//!
//! **Are the values right?** For the legacy Keplerian messages the answer is decided by
//! construction: `rinex4` hands those records to the same `parse_record` the RINEX 3 reader uses,
//! so the two cannot disagree. What that leaves to check is that the *framing* delivers the right
//! eight lines, and the check for it is physical -- each constellation's semi-major axis has to
//! land in its own band. For everything RINEX 3 could not read at all -- CNAV, the system time
//! offsets, Earth orientation, the ionosphere models -- the values are read off the fixture's own
//! columns by eye.

// Exact equality is the assertion: each expected value is the decimal literal the file carries in
// that column, and the question is whether the reader produced that double.
#![allow(clippy::doc_markdown, clippy::float_cmp, clippy::expect_used)]

use spacelab_gnss::GnssSystem;
use spacelab_gnss::rinex4::{IonosphereModel, KeplerianRecord, NavMessage, Subtype, parse};

const FILE: &str = include_str!("fixtures/rinex4_nav_brd400dlr.rnx");

/// Nothing in a real file is lost, and the record counts are the ones the file was built with.
#[test]
fn every_record_shape_the_product_contains_is_read() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");
    assert_eq!(nav.version, 4.02);
    assert_eq!(nav.leap_seconds, Some(18));
    assert!(
        nav.rejected.is_empty(),
        "records the reader could not decode: {:?}",
        nav.rejected
    );

    // 38 shapes: 15 EPH, 9 STO, 4 EOP, 10 ION. The EPH split is by decoded type rather than by
    // message, so it is a statement about the dispatch and not a restatement of the fixture.
    assert_eq!(nav.keplerian.len(), 7, "G/J/I LNAV, INAV, FNAV, D1, D2");
    assert_eq!(nav.cnav.len(), 6, "G/J CNAV, C CNV1, C/J CNV2, C CNV3");
    assert_eq!(nav.glonass.len(), 1);
    assert_eq!(nav.sbas.len(), 1);
    assert_eq!(nav.time_offsets.len(), 9);
    assert_eq!(nav.earth_orientation.len(), 4);
    assert_eq!(nav.ionosphere.len(), 10);
    let total = nav.keplerian.len()
        + nav.cnav.len()
        + nav.glonass.len()
        + nav.glonass_cdma.len()
        + nav.sbas.len()
        + nav.time_offsets.len()
        + nav.earth_orientation.len()
        + nav.ionosphere.len();
    assert_eq!(total, 38, "one record of every shape in the product");
}

/// Every ephemeris lands in its constellation's own orbit band.
///
/// The framing check with teeth. Each of these records is preceded in the fixture by records of a
/// different length, so a block table that is wrong anywhere puts one of these eight lines out of
/// alignment -- and an orbit read from the wrong columns is not a small error, it is kilometres to
/// megametres out of band.
#[test]
fn each_ephemeris_lands_in_its_constellations_orbit_band() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");

    for record in &nav.keplerian {
        let eph = &record.ephemeris;
        let a = eph.sqrt_a * eph.sqrt_a;
        let band = match eph.system {
            GnssSystem::Gps => 26_400e3..26_700e3,
            GnssSystem::Galileo => 29_500e3..29_700e3,
            // BeiDou spans GEO/IGSO (42 164 km) and MEO (27 906 km).
            GnssSystem::Beidou => 27_800e3..42_300e3,
            // QZSS is a highly eccentric IGSO; NavIC is GEO/IGSO.
            GnssSystem::Qzss | GnssSystem::Navic => 41_000e3..42_500e3,
        };
        assert!(
            band.contains(&a),
            "{:?} {} semi-major axis {a} m outside {band:?}",
            eph.system,
            eph.prn
        );
    }

    // The CNAV family broadcasts a reference axis with a rate, so the axis at the reference epoch
    // is `sqrt_a0^2` and `a_dot` is a separate, small number -- metres per second, not metres.
    for eph in &nav.cnav {
        let a = eph.sqrt_a0 * eph.sqrt_a0;
        assert!(
            (26_400e3..42_500e3).contains(&a),
            "{:?} {} {:?} axis {a} m",
            eph.system,
            eph.prn,
            eph.message
        );
        assert!(
            eph.a_dot.abs() < 1.0,
            "{:?} {} A-dot {} m/s is not a rate",
            eph.system,
            eph.prn,
            eph.a_dot
        );
        // QZSS flies a deliberately eccentric IGSO to sit high over Japan; everything else here
        // is near-circular by design. A blanket bound would have to be QZSS's and would then say
        // nothing at all about the other five.
        let max_e = if eph.system == GnssSystem::Qzss {
            0.1
        } else {
            0.02
        };
        assert!(eph.e < max_e, "{:?} {} e = {}", eph.system, eph.prn, eph.e);
    }

    // GLONASS and SBAS broadcast a state vector rather than elements.
    let glo = nav.glonass[0];
    let r = (glo.position[0].powi(2) + glo.position[1].powi(2) + glo.position[2].powi(2)).sqrt();
    assert!((24_500e3..26_500e3).contains(&r), "GLONASS radius {r} m");
    let sbas = nav.sbas[0];
    let r = (sbas.position[0].powi(2) + sbas.position[1].powi(2) + sbas.position[2].powi(2)).sqrt();
    assert!((41_000e3..43_000e3).contains(&r), "SBAS radius {r} m");
}

/// The SBAS record is a Cartesian state in km, km/s and km/s², and the reader owes metres.
///
/// The fixture's `S27` is the one SBAS record in the day's product with a non-zero hour and a
/// non-zero value in all twelve state columns. That matters: two thirds of the day's SBAS records
/// broadcast a zero velocity and a zero acceleration, and a scale error is invisible on a zero.
#[test]
fn the_sbas_record_is_a_geostationary_state_in_metres() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");
    let sbas = nav.sbas[0];
    assert_eq!(sbas.prn, 27);

    // Stamped 10:02:40 UTC; SBAS `Toc` is seconds of day, so 10·3600 + 2·60 + 40 = 36 160 s.
    assert_eq!(sbas.toc, 36_160.0);
    // The epoch line's third field is the message transmission time, not a clock term.
    assert_eq!(sbas.transmission_time, 3.613_700_000_000e+04);
    assert_eq!(sbas.af0, 7.543_712_854_385e-08);
    assert_eq!(sbas.af1, 5.456_968_210_638e-11);

    // Every component against the file's own number, because a norm is degenerate under a
    // single-component error: 42 166 km is still 42 166 km with one axis divided by a thousand.
    assert_eq!(
        sbas.position,
        [
            2.414_981_192_000e+04 * 1000.0,
            3.453_605_192_000e+04 * 1000.0,
            1.443_168_400_000e+03 * 1000.0
        ]
    );
    assert_eq!(
        sbas.velocity,
        [
            -2.685_625_000_000e-03 * 1000.0,
            -1.630_000_000_000e-03 * 1000.0,
            5.692_800_000_000e-02 * 1000.0
        ]
    );
    assert_eq!(
        sbas.acceleration,
        [
            -2.000_000_000_000e-07 * 1000.0,
            4.000_000_000_000e-07 * 1000.0,
            -7.687_500_000_000e-06 * 1000.0
        ]
    );

    let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    // A geostationary satellite sits at 42 164 km, holds well under 10 m/s of residual drift rate
    // in the broadcast state, and its acceleration column is a correction term, not gravity.
    let radius = norm(sbas.position);
    assert!((42_100e3..42_250e3).contains(&radius), "radius {radius} m");
    // 57 m/s in ECEF, which is an inclined geosynchronous orbit rather than a drift rate -- this
    // satellite's orbit plane is about a degree off the equator. The bound is what the geometry
    // allows (a few degrees of inclination), not what a station-kept satellite would show.
    let speed = norm(sbas.velocity);
    assert!((1.0..500.0).contains(&speed), "speed {speed} m/s");
    let accel = norm(sbas.acceleration);
    assert!((1e-6..1e-1).contains(&accel), "acceleration {accel} m/s^2");

    // Each state line's fourth column is a different flag, and they are not interchangeable.
    assert_eq!(sbas.health, 63);
    // 32767 is the accuracy column's "not available" sentinel and it has to survive the read --
    // it does not fit a byte, and neither does the 4096 the same column takes elsewhere.
    assert_eq!(sbas.accuracy_code, 32_767);
    assert_eq!(sbas.iodn, 72);
}

/// GPS's LNAV and CNAV records for the same satellite describe the same orbit.
///
/// The two messages are transmitted on different signals with different parameterisations, and the
/// fixture's `G05 LNAV` and `G05 CNAV` are five minutes apart. If the CNAV block table or its field
/// map were wrong, the two axes would not agree -- and they agree to well under a metre, which no
/// column error survives.
#[test]
fn the_legacy_and_modern_gps_messages_describe_the_same_orbit() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");
    let lnav = &nav
        .keplerian
        .iter()
        .find(|r| r.ephemeris.system == GnssSystem::Gps)
        .expect("a GPS LNAV record")
        .ephemeris;
    let cnav = nav
        .cnav
        .iter()
        .find(|e| e.system == GnssSystem::Gps && e.message == NavMessage::Cnav)
        .expect("a GPS CNAV record");
    assert_eq!(lnav.prn, cnav.prn, "the fixture's two GPS records agree");

    // What this can and cannot say, stated rather than implied. The two messages are fitted
    // separately, over different arcs and to different reference epochs, so their osculating
    // elements are *not* expected to be equal -- measured here, the axes differ by 10.8 m, which is
    // the J2 short-period oscillation of a GPS orbit and not an error in either. What the
    // comparison does bound is a framing error, and it bounds it by four orders of margin: reading
    // a broadcast-orbit line one column to the left divides its first field by ten and its axis by
    // a hundred, so 1e-5 relative is a wall a mis-framed record cannot get near.
    let legacy_axis = lnav.sqrt_a * lnav.sqrt_a;
    let modern_axis = cnav.sqrt_a0 * cnav.sqrt_a0;
    assert!(
        (legacy_axis - modern_axis).abs() / legacy_axis < 1e-5,
        "LNAV axis {legacy_axis} m against CNAV {modern_axis} m"
    );
    assert!((lnav.e - cnav.e).abs() < 1e-5, "{} vs {}", lnav.e, cnav.e);
    assert!(
        (lnav.i0 - cnav.i0).abs() < 1e-4,
        "{} vs {}",
        lnav.i0,
        cnav.i0
    );
    // The two weeks are *not* equal and should not be. LNAV's week goes with `toe`; CNAV's
    // `wn_op` goes with `t_op`, the prediction epoch the fit was referenced to, which precedes the
    // message. Here `toe` is second 0 of week 2428 and `t_op` is second 520 200 of week 2427 --
    // a Sunday-midnight record whose fit was anchored the previous evening. What has to hold is
    // the ordering, and it is the assertion worth making because a week read from the wrong column
    // would break it by thousands of weeks rather than by one.
    let wn_op = cnav.week.expect("CNAV carries wn_op");
    let t_op = cnav.t_op.expect("CNAV carries t_op");
    let op_seconds = f64::from(wn_op) * 604_800.0 + t_op;
    let toc_seconds = f64::from(lnav.week) * 604_800.0 + cnav.toc;
    assert!(
        op_seconds < toc_seconds && toc_seconds - op_seconds < 3.0 * 86_400.0,
        "prediction epoch {op_seconds} s is not shortly before the message epoch {toc_seconds} s"
    );
}

/// The system time offset records, read off the fixture's own columns.
///
/// `STO` puts three left-justified `A18` text fields on its epoch line at columns 24, 43 and 62.
/// Those are what name the offset -- `GPUT` is GPS minus UTC and `BDGA` is BeiDou minus Galileo --
/// and reading them one column out yields a truncated code that matches nothing.
#[test]
fn a_system_time_offset_names_the_two_scales_it_relates() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");

    let bdga = nav
        .time_offsets
        .iter()
        .find(|s| s.constellation == 'C' && s.message == NavMessage::Cnvx)
        .expect("BeiDou's CNVX time offset");
    assert_eq!(bdga.correction, "BDGA");
    assert_eq!(bdga.prn, Some(30));
    assert_eq!(bdga.epoch.year, 2026);
    assert_eq!(bdga.epoch.month, 7);
    assert_eq!(bdga.transmission_time, 6.048_300_000_000e+05);
    assert_eq!(bdga.a0, -1.504_668_034_613e-08);
    assert_eq!(bdga.a1, 3.730_349_362_741e-14);
    assert_eq!(bdga.a2, 2.134_523_027_081e-19);

    // Every correction code is four characters of two two-letter system codes, and the offsets are
    // physical: a system-to-system offset is nanoseconds to microseconds, never seconds.
    for sto in &nav.time_offsets {
        assert_eq!(sto.correction.len(), 4, "{sto:?}");
        assert!(sto.correction.is_ascii(), "{sto:?}");
        assert!(sto.a0.abs() < 1.0, "{} s is not a clock offset", sto.a0);
    }
    // A UTC offset names the laboratory realising it; a system-to-system one has nothing to name.
    let utc: Vec<&str> = nav
        .time_offsets
        .iter()
        .filter(|s| s.correction.ends_with("UT"))
        .map(|s| s.utc_id.as_str())
        .collect();
    assert!(
        utc.iter().any(|id| !id.is_empty()),
        "no UTC realisation named on any UT offset: {utc:?}"
    );
}

/// Broadcast Earth orientation, which RINEX 3 could not carry at all.
///
/// The check is against the IERS: the pole wanders inside a circle of about 0.3 arc-seconds and
/// `UT1 - UTC` is bounded by a leap second, so a value outside those is a column error rather than
/// a satellite's opinion. The specification warns that some constellations broadcast `UT1 - GPST`
/// instead, which is larger than one second by construction, so the bound here is the leap-second
/// count and not one second.
#[test]
fn the_broadcast_earth_orientation_is_within_the_ranges_the_iers_publishes() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");
    assert_eq!(nav.earth_orientation.len(), 4);

    for eop in &nav.earth_orientation {
        assert!(
            eop.x_pole[0].abs() < 0.5 && eop.y_pole[0].abs() < 0.7,
            "pole ({}, {}) arc-seconds is outside the Chandler circle",
            eop.x_pole[0],
            eop.y_pole[0]
        );
        // Rates are arc-seconds per day: the pole moves about 0.3 arc-seconds a year.
        assert!(
            eop.x_pole[1].abs() < 0.01 && eop.y_pole[1].abs() < 0.01,
            "{eop:?}"
        );
        assert!(
            eop.delta_ut1[0].abs() < 20.0,
            "delta UT1 {} s exceeds the leap-second count",
            eop.delta_ut1[0]
        );
    }

    // The four records come from four constellations and must agree about where the pole is --
    // they are four independent measurements of one Earth, and this is the strongest statement
    // available without an IERS file: a reader that mis-read one of them would disagree with the
    // other three.
    let xs: Vec<f64> = nav.earth_orientation.iter().map(|e| e.x_pole[0]).collect();
    let ys: Vec<f64> = nav.earth_orientation.iter().map(|e| e.y_pole[0]).collect();
    let spread = |v: &[f64]| {
        v.iter().copied().fold(f64::MIN, f64::max) - v.iter().copied().fold(f64::MAX, f64::min)
    };
    assert!(
        spread(&xs) < 0.02 && spread(&ys) < 0.02,
        "four constellations disagree about the pole: x {xs:?}, y {ys:?}"
    );
}

/// The four ionosphere models, each decoded as the model its message carries.
///
/// This is the one place the reader has to choose between shapes that occupy the same lines: three
/// lines of a Klobuchar record and three of a BDGIM record are indistinguishable by length, and
/// only the constellation and the message tell them apart. Getting it wrong reports nine BDGIM
/// coefficients as eight Klobuchar ones.
#[test]
fn each_ionosphere_record_decodes_as_the_model_its_message_carries() {
    let nav = parse(FILE).expect("a real RINEX 4.02 navigation file");

    let mut klobuchar = 0;
    let mut nequick = 0;
    let mut bdgim = 0;
    for ion in &nav.ionosphere {
        match ion.model {
            IonosphereModel::Klobuchar {
                alpha,
                beta,
                region,
            } => {
                klobuchar += 1;
                // Only NavIC L1 broadcasts a region, and no product carries that message yet, so
                // every Klobuchar record in a real file is a global one.
                assert_eq!(region, None, "{ion:?}");
                // The bounds are the message's own quantisation, not a guess. IS-GPS-200 sends
                // each coefficient as eight signed bits with a fixed scale, so the largest
                // representable alpha is 127*2^-24 = 7.6e-6 and the largest beta is 127*2^16 =
                // 8.3e6. A set decoded from the wrong columns lands outside that at once.
                assert!(
                    alpha.iter().all(|a| a.abs() < 1e-5),
                    "{ion:?} has an alpha outside the Klobuchar quantisation"
                );
                assert!(
                    beta.iter().all(|b| b.abs() < 1e7),
                    "{ion:?} has a beta outside the Klobuchar quantisation"
                );
            }
            IonosphereModel::NequickG {
                ai,
                disturbance_flags,
            } => {
                nequick += 1;
                // a0 is an effective ionisation level in solar flux units, order 10-200.
                assert!((1.0..500.0).contains(&ai[0]), "{ion:?}");
                assert!(disturbance_flags < 32, "a five-bit field");
            }
            IonosphereModel::Bdgim { alpha } => {
                bdgim += 1;
                // TECu, and a total electron content of hundreds would be extraordinary.
                assert!(alpha.iter().all(|a| a.abs() < 200.0), "{ion:?}");
            }
            // No satellite broadcasts either of these into a published product yet -- measured
            // across BRD400DLR days 2024/001, 2025/300, 2026/150 and 2026/200, none carries an
            // `LXOC` or an `L1NV` ionosphere record. They are read against the specification's own
            // example bytes in the unit tests instead.
            IonosphereModel::GlonassCdma { .. } | IonosphereModel::NequickN { .. } => {
                panic!("the product carries no LXOC or NavIC L1 record: {ion:?}")
            }
        }
    }
    // Eight Klobuchar, one NeQuick-G, one BDGIM. The split is the point: BeiDou appears twice in
    // this file and broadcasts a *different model* in each message -- Klobuchar in D1/D2 and BDGIM
    // in CNVX -- and the two records are the same three lines long, so nothing but the message type
    // distinguishes them. Reading the CNVX one as Klobuchar would report nine BDGIM coefficients as
    // eight Klobuchar ones and the count here would read (9, 1, 0).
    assert_eq!((klobuchar, nequick, bdgim), (8, 1, 1), "model counts");

    // QZSS broadcasts two regional Klobuchar sets and the subtype is the only thing telling them
    // apart -- a reader that dropped it would report one of them twice.
    let qzss: Vec<Option<Subtype>> = nav
        .ionosphere
        .iter()
        .filter(|i| i.constellation == 'J')
        .map(|i| i.subtype)
        .collect();
    assert_eq!(
        qzss.iter().filter(|s| **s == Some(Subtype::Wide)).count(),
        2
    );
    assert_eq!(
        qzss.iter().filter(|s| **s == Some(Subtype::Japn)).count(),
        2
    );
    let wide = nav
        .ionosphere
        .iter()
        .find(|i| i.subtype == Some(Subtype::Wide))
        .expect("a QZSS wide-area record");
    let japan = nav
        .ionosphere
        .iter()
        .find(|i| i.subtype == Some(Subtype::Japn))
        .expect("a QZSS Japan-area record");
    assert_ne!(wide.model, japan.model, "the two regions are the same set");
}

// ---------------------------------------------------------------------------------------------
// Whole-product tests. `#[ignore]`d because they need ~50 MB of navigation files that are not
// committed; `tools/fetch-gnss-nav.sh` puts them in the cache and `cargo test -- --ignored` runs
// them. The fixture above is one record of each shape; these are every record in file order, which
// is the only thing that exercises the transitions between shapes at the density a real day has.
// ---------------------------------------------------------------------------------------------

use std::path::PathBuf;

/// The four RINEX 4 products the fetch script retrieves, if they are in the cache.
fn cached(name: &str) -> Option<String> {
    let path: PathBuf = [
        &std::env::var("HOME").ok()?,
        ".cache",
        "spacelab",
        "gnss",
        name,
    ]
    .iter()
    .collect();
    std::fs::read_to_string(path).ok()
}

const PRODUCTS: [&str; 4] = [
    "brd400dlr_2024001.rnx",
    "brd400dlr_2025300.rnx",
    "brd400dlr_2026150.rnx",
    "brd400dlr_2026200.rnx",
];

/// Nothing is lost from a whole day of every constellation, on four days over two and a half years.
///
/// The number this test exists to produce is `rejected`, and it has to be zero on every file. A
/// block length wrong for one message type would show up here as a run of rejections starting at
/// the first record of that type and continuing to the end of the file, because the walk never
/// recovers -- which is why the count matters more than any individual value.
///
/// 2024/001 is a RINEX **4.00** file and the other three are 4.02. A reader that only works on the
/// newest revision of a format is a reader that will break.
#[test]
#[ignore = "needs tools/fetch-gnss-nav.sh"]
fn a_whole_day_of_broadcast_navigation_is_read_without_losing_a_record() {
    let mut ran = 0;
    for name in PRODUCTS {
        let Some(text) = cached(name) else { continue };
        ran += 1;
        let nav = parse(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(
            (4.0..5.0).contains(&nav.version),
            "{name}: version {}",
            nav.version
        );
        assert!(
            nav.rejected.is_empty(),
            "{name}: {} of {} records could not be decoded, first at lines {:?}",
            nav.rejected.len(),
            nav.rejected.len(),
            &nav.rejected[..nav.rejected.len().min(8)]
        );
        let total = nav.keplerian.len()
            + nav.cnav.len()
            + nav.glonass.len()
            + nav.glonass_cdma.len()
            + nav.sbas.len()
            + nav.time_offsets.len()
            + nav.earth_orientation.len()
            + nav.ionosphere.len();
        // A day of six constellations is tens of thousands of records; a walk that desynchronised
        // early would still report `rejected` empty if it landed on lines that happen to parse, so
        // the count is a second, independent statement.
        assert!(total > 15_000, "{name}: only {total} records");

        // Every ephemeris in the file, not just the first of each shape.
        for record in &nav.keplerian {
            let eph = &record.ephemeris;
            let a = eph.sqrt_a * eph.sqrt_a;
            assert!(
                (26_000e3..42_600e3).contains(&a),
                "{name}: {:?} {} {:?} axis {a} m",
                eph.system,
                eph.prn,
                record.message
            );
        }
        for eph in &nav.cnav {
            let a = eph.sqrt_a0 * eph.sqrt_a0;
            assert!(
                (26_000e3..42_600e3).contains(&a),
                "{name}: {:?} {} {:?} axis {a} m",
                eph.system,
                eph.prn,
                eph.message
            );
        }
    }
    assert!(
        ran > 0,
        "no products in the cache; run tools/fetch-gnss-nav.sh"
    );
}

/// Two products, two producers, two format versions, one day — and what actually agrees.
///
/// `BRDC00WRD` is BKG's RINEX 3 merge and `BRD400DLR` is DLR's RINEX 4 merge of the same day's
/// broadcast messages. The ticket this test was written for asked for bit-exact agreement between a
/// RINEX 4 file and "its RINEX 3 twin", and **the data refuses that premise**: the two products are
/// not twins, and no pair of independently produced RINEX files is. Measured on the **944** records
/// the two share unambiguously, only **269 are identical in every field**. Three unrelated causes,
/// none of them a reader:
///
/// * **Written precision.** `E19.12` holds thirteen significant digits and neither producer always
///   fills them. BKG writes G32's `√A` as `5.153623479840e+03` where DLR writes
///   `5.153623479843e+03` — the same number to twelve digits, two different doubles.
/// * **Flag fields.** QZSS's "codes on L2" and "L2 P data flag" are decoded differently by the two:
///   J04 and J07 carry 1 and 0 where the other carries 0 and 1.
/// * **The Galileo ambiguity**, below, which is why the comparison is restricted to unambiguous
///   keys at all.
///
/// So byte identity is a property of one producer's writer, not of the format, and asserting it
/// would be asserting something false about real data. What *is* true, and is what a user of this
/// library actually depends on, is that the two describe the same orbit: every element agrees to
/// the precision the weaker of the two writes, and the propagated ECEF positions agree to well
/// under a millimetre. That is what this asserts, and the numbers it measures are **4.98e-12**
/// worst relative element disagreement — exactly the twelve-digit write precision — and
/// **0.42 mm** worst position disagreement an hour past the reference epoch, on a 26 560 km orbit.
///
/// **Two attempts at the key failed first, and both failures are findings of their own.**
///
/// `(constellation, PRN, toe, week)` is not a key: BeiDou rebroadcasts one ephemeris across several
/// clock epochs, so C04 appears twice at `toe` 165 600 with `toc` an hour apart. Adding `toc`
/// removes every one of those, completely — with `toc` in the key not one non-Galileo key carries
/// two records.
///
/// `(constellation, PRN, toe, toc, week)` is still not a key, and cannot be made one from RINEX 3
/// alone. Galileo transmits the same ephemeris in F/NAV and I/NAV, carrying the E5a−E1 and E5b−E1
/// ionosphere-free clock calibrations. Measured here: **2 659 keys carry two records**, bit-identical
/// in every orbit element and differing in the frequency-specific fields. A caller looking up "E08
/// at `toe` 86 400" in a `Vec<BroadcastEphemeris>` gets whichever came last and is up to 0.58 m out
/// half the time, silently. RINEX 3 puts the frequency pair in bits 8 and 9 of the data-source
/// field, which `BroadcastEphemeris` does not keep, so it cannot be resolved there at all. RINEX 4
/// states it in the record header — which is what
/// [`spacelab_gnss::rinex4::KeplerianRecord`] exists to carry.
#[test]
#[ignore = "needs tools/fetch-gnss-nav.sh"]
fn the_two_products_for_one_day_describe_the_same_orbits() {
    let (Some(v3), Some(v4)) = (
        cached("brdc00wrd_2024001.rnx"),
        cached("brd400dlr_2024001.rnx"),
    ) else {
        panic!("both 2024/001 products are needed; run tools/fetch-gnss-nav.sh");
    };

    let three = spacelab_gnss::parse_nav(&v3);
    let four = parse(&v4).expect("the RINEX 4 product").keplerian;
    assert!(three.len() > 1000 && four.len() > 1000, "thin products");

    // `GnssSystem` is not `Hash`, and making it so to key a test map would be changing the public
    // API for a test's convenience. The discriminant is a total, stable function of it.
    let key = |e: &spacelab_gnss::BroadcastEphemeris| {
        (
            e.system as u8,
            e.prn,
            e.toe.to_bits(),
            e.toc.to_bits(),
            e.week,
        )
    };
    // Every parameter that says where the satellite is, and none that says anything about its clock
    // or about one signal's health. Written out rather than derived by zeroing fields, so a field
    // added to `BroadcastEphemeris` later fails to compile here instead of going silently unchecked.
    let orbit = |e: &spacelab_gnss::BroadcastEphemeris| {
        [
            e.sqrt_a,
            e.e,
            e.i0,
            e.omega0,
            e.omega,
            e.m0,
            e.delta_n,
            e.idot,
            e.omega_dot,
            e.cuc,
            e.cus,
            e.crc,
            e.crs,
            e.cic,
            e.cis,
        ]
    };

    let mut by_key: std::collections::HashMap<_, Vec<spacelab_gnss::BroadcastEphemeris>> =
        std::collections::HashMap::new();
    for record in &four {
        by_key
            .entry(key(&record.ephemeris))
            .or_default()
            .push(record.ephemeris);
    }

    // The cross-version comparison, on the keys that identify one message. The keys that do not
    // are Galileo's, and they are `the_galileo_ephemeris_arrives_in_two_messages_with_two_clocks`.
    let unique: std::collections::HashMap<_, _> = by_key
        .iter()
        .filter(|(_, g)| g.len() == 1)
        .map(|(k, g)| (*k, g[0]))
        .collect();
    let mut matched = 0;
    let mut identical = 0;
    let mut worst_element = 0.0_f64;
    let mut worst_position = 0.0_f64;
    for a in &three {
        let Some(b) = unique.get(&key(a)) else {
            continue;
        };
        matched += 1;
        if a == b {
            identical += 1;
        }
        for (x, y) in orbit(a).into_iter().zip(orbit(b)) {
            if x != y {
                worst_element = worst_element.max((x - y).abs() / x.abs().max(1e-30));
            }
        }
        // The statement that survives the formatting: the same satellite, in the same place. Both
        // ephemerides are propagated an hour past their own reference epoch, so an element read
        // differently has an hour of dynamics to grow in before the positions are compared.
        let (pa, _) = a.position_velocity_ecef(3600.0);
        let (pb, _) = b.position_velocity_ecef(3600.0);
        let d =
            ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
        worst_position = worst_position.max(d);
    }
    // The two merges do not carry the same satellites, so a partial overlap is expected; what is
    // not expected is a small one.
    assert!(
        matched > 900,
        "only {matched} messages in common, {identical} of them byte-identical"
    );
    // Byte identity is not claimed and does not hold: measured, 269 of 944.
    assert!(
        identical < matched,
        "every record is byte-identical, which contradicts the measurement this test records"
    );
    // Twelve significant digits is what the weaker writer carries, and the worst measured
    // disagreement is 4.98e-12 -- one unit in the twelfth digit. 1e-11 is the wall just above it.
    assert!(
        worst_element < 1e-11,
        "worst element disagreement {worst_element} exceeds the written precision"
    );
    // And that is 0.42 mm of orbit an hour later, on a 26 560 km semi-major axis: eleven orders
    // below the axis itself, which is the scale any column or framing error would move it by.
    assert!(
        worst_position < 1e-3,
        "the two products put a satellite {worst_position} m apart an hour after its epoch"
    );
}

/// Galileo transmits one ephemeris in two messages with two clocks, and RINEX 3 cannot say which.
///
/// The finding that made `KeplerianRecord` carry its message type. F/NAV carries the E5a−E1
/// ionosphere-free clock calibration and I/NAV the E5b−E1 one, so for one satellite at one `toe`
/// and one `toc` there are two records that agree bit for bit in every orbit element and differ in
/// exactly the fields the Galileo ICD makes frequency-specific. Measured on DLR's 2024-01-01
/// product, over 2 659 such keys:
///
/// * `af0` differs on **2 584**, worst 1.92e-9 s — 0.58 m of range.
/// * the health word differs on **214**: F/NAV reports 16 (the E5a bits) where I/NAV reports 130
///   (E1B and E5b) for the same satellite at the same epoch. Table A13's note 6* says a message
///   zeroes the bits it does not carry, so both are right.
/// * `BGD E5a/E1` — what `BroadcastEphemeris::tgd` holds — differs on **39**, worst 1.16e-9 s, five
///   steps of its own 2⁻³² s quantisation: one quantity transmitted in two message streams that
///   were updated at different times.
/// * `BGD E5b/E1` is exactly zero in F/NAV on all 2 659, which is the zeroing rule doing what it
///   says. `BroadcastEphemeris` keeps only the E5a one, so this test cannot see it and says so
///   rather than implying otherwise.
///
/// A `Vec<BroadcastEphemeris>` cannot express any of that. RINEX 3 puts the frequency pair in bits 8
/// and 9 of the data-source field, which `BroadcastEphemeris` does not keep, so a RINEX 3 reader
/// cannot resolve it at all — a lookup by satellite and epoch returns whichever record came last
/// and is up to 0.58 m out about half the time, with nothing to indicate it.
#[test]
#[ignore = "needs tools/fetch-gnss-nav.sh"]
fn the_galileo_ephemeris_arrives_in_two_messages_with_two_clocks() {
    let Some(v4) = cached("brd400dlr_2024001.rnx") else {
        panic!("the 2024/001 RINEX 4 product is needed; run tools/fetch-gnss-nav.sh");
    };
    let four = parse(&v4).expect("the RINEX 4 product").keplerian;

    let key = |e: &spacelab_gnss::BroadcastEphemeris| {
        (
            e.system as u8,
            e.prn,
            e.toe.to_bits(),
            e.toc.to_bits(),
            e.week,
        )
    };
    let orbit = |e: &spacelab_gnss::BroadcastEphemeris| {
        [
            e.sqrt_a,
            e.e,
            e.i0,
            e.omega0,
            e.omega,
            e.m0,
            e.delta_n,
            e.idot,
            e.omega_dot,
            e.cuc,
            e.cus,
            e.crc,
            e.crs,
            e.cic,
            e.cis,
        ]
        .map(f64::to_bits)
    };

    let mut by_key: std::collections::HashMap<_, Vec<&KeplerianRecord>> =
        std::collections::HashMap::new();
    for record in &four {
        by_key
            .entry(key(&record.ephemeris))
            .or_default()
            .push(record);
    }

    let mut ambiguous = 0;
    let mut worst_af0 = 0.0_f64;
    let mut differing_clock = 0;
    let mut differing_health = 0;
    let mut triples = 0;
    for group in by_key.values().filter(|g| g.len() > 1) {
        ambiguous += 1;
        // The messages in a group have to be the two the ICD says they are. A group of two I/NAV
        // records would be a duplicate rather than a calibration pair, and the clock difference
        // below would mean something else entirely.
        let messages: Vec<NavMessage> = group.iter().map(|r| r.message).collect();
        assert!(
            messages.contains(&NavMessage::Fnav) && messages.contains(&NavMessage::Inav),
            "a group that is not one F/NAV and one I/NAV: {messages:?}"
        );
        for record in group {
            let e = &record.ephemeris;
            let first = &group[0].ephemeris;
            assert_eq!(
                e.system,
                GnssSystem::Galileo,
                "{:?} {} carries one ephemeris in two messages at one toe and toc",
                e.system,
                e.prn
            );
            assert_eq!(
                orbit(e),
                orbit(first),
                "two messages, one epoch, two orbits"
            );
            worst_af0 = worst_af0.max((e.af0 - first.af0).abs());
        }
        // Compare the F/NAV record against an I/NAV one *by message*, not by position. 46 of the
        // groups carry three records -- one F/NAV and two I/NAV -- so `group[0]` against `group[1]`
        // would sometimes be I/NAV against I/NAV, and would count 2 541 differing clocks where the
        // question asked is about the two calibrations.
        let pick = |m: NavMessage| {
            group
                .iter()
                .find(|r| r.message == m)
                .map(|r| &r.ephemeris)
                .expect("the group holds both messages")
        };
        let (a, b) = (pick(NavMessage::Fnav), pick(NavMessage::Inav));
        differing_clock += usize::from(a.af0 != b.af0);
        differing_health += usize::from(a.health != b.health);
        triples += usize::from(group.len() > 2);
    }

    assert_eq!(ambiguous, 2659, "keys carrying more than one message");
    assert_eq!(differing_clock, 2584, "pairs whose clock bias differs");
    assert_eq!(differing_health, 214, "pairs whose health word differs");
    // 46 keys carry a third record -- a second I/NAV for the same epoch, which the merge did not
    // collapse. Counted rather than filtered out, because it is why the comparison above picks its
    // two records by message name instead of by position.
    assert_eq!(triples, 46, "keys carrying a third record");
    // The E5a-E1 against E5b-E1 clock calibration: nanoseconds, tens of centimetres of range.
    assert!(
        (1e-9..2e-9).contains(&worst_af0),
        "the two Galileo clocks differ by {worst_af0} s, which is not a group delay"
    );
}

/// The CNAV propagator against an IGS precise orbit, and against the legacy message on the same day.
///
/// This is the only oracle for `CnavEphemeris::position_velocity_ecef` that is not the algorithm
/// itself: `IGS0OPSRAP_20262240000_01D_15M_ORB.SP3` is a post-processed combination of eleven
/// analysis centres, accurate to centimetres, and it is in the local `data/validation/`. The broadcast
/// message is expected to sit one to two metres from it — that is the signal-in-space range error
/// the constellation is specified to, not a tolerance chosen to pass.
///
/// The comparison that makes it worth running is CNAV *against LNAV* on the same satellites, the
/// same epochs and the same precise orbit. The two messages are independent fits broadcast on
/// different signals; if the CNAV transcription of Table 30-II were wrong in a term the
/// rates-zeroed identity cannot see — `Ȧ` with the wrong sign, say — CNAV would be the worse of
/// the two by metres, and the unit test would still pass.
///
/// Measured 2026-09-05 on 2026-08-12: **LNAV 1.816 m RMS, worst 4.475 m** over 3 072 comparisons;
/// **CNAV 1.854 m RMS, worst 4.324 m** over 2 688 — fewer because four GPS satellites broadcast no
/// CNAV at all. Controls, so the numbers are known to be measuring the `Ȧ` term and not the orbit:
/// zeroing `Ȧ` takes CNAV to **15.13 m RMS** and flipping its sign to **30.08 m RMS**, eight and
/// sixteen times worse.
#[test]
#[ignore = "needs tools/fetch-gnss-nav.sh"]
fn the_cnav_propagator_lands_where_the_precise_orbit_says() {
    let Some(nav_text) = cached("brd400dlr_2026224.rnx") else {
        panic!("run tools/fetch-gnss-nav.sh");
    };
    // Read at run time, not with `include_str!`: `data/` is not committed, and a compile-time
    // read would fail the build of every test in this file on a checkout that lacks it.
    let sp3_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/validation/IGS0OPSRAP_20262240000_01D_15M_ORB.SP3");
    let Ok(sp3_text) = std::fs::read_to_string(&sp3_path) else {
        panic!("{} is not present", sp3_path.display());
    };
    let sp3_text = sp3_text.as_str();
    let nav = parse(&nav_text).expect("the day's broadcast navigation");
    let sp3 = spacelab_gnss::sp3::Sp3::parse(sp3_text);

    // The `##` header line gives the GPS week and the seconds of week of the first epoch, which is
    // what puts the SP3 grid and the broadcast `toe` on one clock.
    let week_line = sp3_text
        .lines()
        .find(|l| l.starts_with("##"))
        .expect("the GPS-week line");
    let mut fields = week_line[2..].split_whitespace();
    let week: f64 = fields.next().expect("week").parse().expect("week");
    let sow0: f64 = fields.next().expect("sow").parse().expect("sow");

    // Worst and root-mean-square 3D error, per message.
    let mut stats = [(0.0_f64, 0.0_f64, 0_usize); 2];
    for (slot, want) in [NavMessage::Lnav, NavMessage::Cnav].into_iter().enumerate() {
        for sat in sp3.satellites().filter(|s| s.system == 'G') {
            let precise = sp3.positions(sat).expect("a satellite the file lists");
            for (i, sample) in precise.iter().enumerate() {
                let Some(truth) = sample else { continue };
                let sow =
                    900.0f64.mul_add(f64::from(u32::try_from(i).expect("an epoch index")), sow0);
                // The record whose reference epoch is nearest this instant, within a fit interval.
                let best = match want {
                    NavMessage::Lnav => nav
                        .keplerian
                        .iter()
                        .filter(|r| {
                            r.message == NavMessage::Lnav
                                && r.ephemeris.system == GnssSystem::Gps
                                && r.ephemeris.prn == sat.prn
                                && f64::from(r.ephemeris.week) == week
                        })
                        .map(|r| {
                            (
                                sow - r.ephemeris.toe,
                                r.ephemeris.position_ecef(sow - r.ephemeris.toe),
                            )
                        })
                        .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs())),
                    _ => nav
                        .cnav
                        .iter()
                        .filter(|e| {
                            e.message == NavMessage::Cnav
                                && e.system == GnssSystem::Gps
                                && e.prn == sat.prn
                        })
                        .map(|e| (sow - e.toc, e.position_ecef(sow - e.toc)))
                        .min_by(|a, b| a.0.abs().total_cmp(&b.0.abs())),
                };
                let Some((tk, got)) = best else { continue };
                if tk.abs() > 7200.0 {
                    continue;
                }
                let d = (0..3)
                    .map(|k| (got[k] - truth[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                stats[slot].0 = stats[slot].0.max(d);
                stats[slot].1 += d * d;
                stats[slot].2 += 1;
            }
        }
    }

    let report = |(worst, sq, n): (f64, f64, usize), name: &str| {
        // 32 satellites x 96 epochs for LNAV; fewer for CNAV, because the older blocks do not
        // broadcast it at all.
        assert!(n >= 2_500, "{name}: only {n} comparisons");
        let rms = (sq / f64::from(u32::try_from(n).expect("a comparison count"))).sqrt();
        println!("{name}: {n} comparisons, RMS {rms:.3} m, worst {worst:.3} m");
        rms
    };
    let lnav_rms = report(stats[0], "LNAV");
    let cnav_rms = report(stats[1], "CNAV");

    // Both messages are broadcast ephemerides and both are specified to a few metres of URE.
    assert!(lnav_rms < 5.0, "LNAV RMS {lnav_rms} m");
    assert!(cnav_rms < 5.0, "CNAV RMS {cnav_rms} m");
    // And CNAV is not the worse of the two by a margin only a wrong term could produce.
    assert!(
        cnav_rms < 2.0 * lnav_rms,
        "CNAV {cnav_rms} m against LNAV {lnav_rms} m"
    );
}
