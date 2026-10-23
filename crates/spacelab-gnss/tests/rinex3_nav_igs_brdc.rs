//! The navigation reader against a file it did not write.
//!
//! Every other test of this reader builds its own RINEX text. That is worth having -- it pins the
//! reader as the writer's inverse -- but it cannot answer the question the format actually poses,
//! because a fixture written by the same understanding that wrote the parser agrees with the parser
//! by construction. It agreed for a long time: `orbit_fields` read the four `E19.12` columns of a
//! broadcast-orbit line at offsets 3, 22, 41 and 60 where Table A9 specifies `4X,4E19.12`, so
//! offsets 4, 23, 42 and 61. On a hand-built fixture with a three-space indent the two agree. On a
//! real file the first field comes back divided by ten, and the other three straddle their
//! neighbours and fail to parse -- which `parse_field`'s `unwrap_or(0.0)` then reported as zero.
//!
//! The fixture is 40 lines of `BRDC00WRD_R_20240010000_01D_MN.rnx` (IGS/BKG multi-GNSS broadcast
//! navigation, 2024-01-01), verbatim, plus attribution comments. The expected values below are read
//! off those columns by eye rather than computed, so they are independent of everything the crate
//! does with them.

// Exact equality is the assertion, not an oversight: each expected value is the decimal literal the
// file carries in that column, and the question is whether the reader produced *that* double or a
// different one. A tolerance would be a second convention to get wrong, and the failure this test
// exists for -- a field read from the wrong columns -- is never within a tolerance anyway.
#![allow(clippy::doc_markdown, clippy::float_cmp, clippy::expect_used)]

use spacelab_gnss::{GnssSystem, parse_glonass_nav, parse_nav};

const FILE: &str = include_str!("fixtures/rinex3_nav_igs_brdc.24n");

/// A field-by-field read of G01's record, against the file's own columns.
///
/// `sqrt_a`, `e` and `crs` are the three that name the bug: under the old offsets `crs` came back
/// as 6.95625 rather than 69.5625, and `e` and `sqrt_a` -- second and fourth on their line -- came
/// back as 0.
#[test]
fn every_field_of_a_real_gps_record_is_the_one_in_its_column() {
    let g01 = parse_nav(FILE)
        .into_iter()
        .find(|e| e.system == GnssSystem::Gps && e.prn == 1)
        .expect("the fixture carries a G01 record");

    // Epoch line: SV clock bias, drift, drift rate.
    assert_eq!(g01.af0, 1.649_968_326_092e-04);
    assert_eq!(g01.af1, 7.958_078_640_513e-13);
    assert_eq!(g01.af2, 0.0);
    // Orbit 1: IODE, Crs, delta n, M0. Field 1 is the one that read as 7.2 instead of 72.
    assert_eq!(g01.crs, 6.956_250_000_000e+01);
    assert_eq!(g01.delta_n, 3.974_094_108_400e-09);
    assert_eq!(g01.m0, 2.047_872_163_810e-01);
    // Orbit 2: Cuc, e, Cus, sqrt(A). Fields 2 and 4 read as 0 under the old offsets.
    assert_eq!(g01.cuc, 3.783_032_298_088e-06);
    assert_eq!(g01.e, 1.305_347_960_442e-02);
    assert_eq!(g01.cus, 6.282_702_088_356e-06);
    assert_eq!(g01.sqrt_a, 5.154_024_417_877e+03);
    // Orbit 3: Toe, Cic, OMEGA0, Cis.
    assert_eq!(g01.toe, 8.640_000_000_000e+04);
    assert_eq!(g01.cic, -8.009_374_141_693e-08);
    assert_eq!(g01.omega0, -1.609_681_262_806e+00);
    assert_eq!(g01.cis, 1.192_092_895_508e-07);
    // Orbit 4: i0, Crc, omega, OMEGA DOT.
    assert_eq!(g01.i0, 9.903_294_508_767e-01);
    assert_eq!(g01.crc, 2.726_875_000_000e+02);
    assert_eq!(g01.omega, 9.986_756_881_767e-01);
    assert_eq!(g01.omega_dot, -8.096_765_834_225e-09);
    // Orbit 5: IDOT, codes on L2, GPS week, L2 P flag.
    assert_eq!(g01.idot, 8.143_196_339_671e-11);
    assert_eq!(g01.week, 2295);
    // Orbit 6: SV accuracy, SV health, TGD, IODC.
    assert_eq!(g01.tgd, 5.122_274_160_385e-09);
    assert_eq!(g01.health, 63);
    // The epoch line's calendar, 2024-01-01 00:00:00, is a Monday: second 86 400 of the GPS week,
    // and the record's own Toe says the same thing independently.
    assert_eq!(g01.toc, 86_400.0);
    assert_eq!(g01.toc, g01.toe);
}

/// Every Keplerian record in the file, at the right constellation and in the right orbit band.
///
/// The semi-major axes are the check that survives a change of fixture: GPS and BeiDou MEO sit near
/// 26 560 km and 27 900 km, Galileo near 29 600 km, and a column error puts any of them nowhere
/// near -- reading `sqrt_a` as 0 makes the propagator's radius 0, and reading it one column left
/// divides it by ten and the axis by a hundred.
#[test]
fn each_constellation_lands_in_its_own_orbit_band() {
    let records = parse_nav(FILE);
    assert_eq!(
        records
            .iter()
            .map(|e| (e.system, e.prn))
            .collect::<Vec<_>>(),
        vec![
            (GnssSystem::Galileo, 8),
            (GnssSystem::Gps, 1),
            (GnssSystem::Gps, 2),
            (GnssSystem::Beidou, 25),
        ],
        "the file's four Keplerian records, in file order"
    );

    for eph in &records {
        let a = eph.sqrt_a * eph.sqrt_a;
        let band = match eph.system {
            GnssSystem::Gps => 26_500e3..26_620e3,
            GnssSystem::Galileo => 29_540e3..29_660e3,
            GnssSystem::Beidou => 27_850e3..27_960e3,
            other => panic!("unexpected constellation {other:?}"),
        };
        assert!(
            band.contains(&a),
            "{:?} {} semi-major axis {a} m is outside {band:?}",
            eph.system,
            eph.prn
        );
        // Every GNSS orbit is very nearly circular; a mis-read column would not be.
        assert!(eph.e < 0.02, "{:?} {} e = {}", eph.system, eph.prn, eph.e);
        // And the propagated radius has to agree with the axis it was built from.
        let (p, _) = eph.position_velocity_ecef(0.0);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!(
            (r - a).abs() < 0.03 * a,
            "{:?} {} radius {r} m against axis {a} m",
            eph.system,
            eph.prn
        );
    }
}

/// The GLONASS records, whose state vector is broadcast in kilometres in the PZ-90 frame.
///
/// This is the same column question asked of a different table: the FDMA record's `4X,4E19.12`
/// lines carry position, velocity, acceleration and a flag rather than Keplerian elements, so a
/// wrong offset shows up as a radius rather than as an axis. Both R11 and R10 are read because the
/// second one has a negative frequency-channel number in the fourth column of its second line --
/// the field furthest from the start, and the one a cumulative offset error corrupts worst.
#[test]
fn the_glonass_records_carry_a_pz90_state_in_kilometres() {
    let records = parse_glonass_nav(FILE);
    assert_eq!(
        records.iter().map(|e| e.slot).collect::<Vec<_>>(),
        vec![11, 10],
        "the file's two FDMA records, in file order"
    );

    // R11, line by line against the file. The file is in km and km/s; the reader scales to SI, so
    // the factor of 1000 in each of these is the only thing between the assertion and the columns.
    let r11 = &records[0];
    assert_eq!(
        r11.position,
        [
            1.163_025_927_734e+04 * 1e3,
            -2.265_464_013_672e+04 * 1e3,
            1.345_364_746_094e+03 * 1e3,
        ]
    );
    assert_eq!(
        r11.velocity,
        [
            -5.490_779_876_709e-02 * 1e3,
            -2.390_241_622_925e-01 * 1e3,
            -3.565_714_836_121e+00 * 1e3,
        ]
    );
    // Third column of each orbit line: the luni-solar acceleration. On this record only the x
    // component is non-zero, which is the assertion catching an axis permutation as well as a
    // column one -- the three lines are read as x, y, z and not the other way round.
    assert_eq!(r11.luni_solar, [-9.313_225_746_155e-10 * 1e3, 0.0, 0.0]);
    // Epoch line: -TauN, +GammaN, and the message frame time in the third field.
    assert_eq!(r11.tau, -2.246_890_217_066e-04);
    assert_eq!(r11.gamma, -2.728_484_105_319e-12);
    // 00:15:00 on the epoch line.
    assert_eq!(r11.toe, 900.0);

    // R10's frequency channel is -7: the fourth column of its *second* orbit line, the field
    // furthest from the start of a line and so the one a cumulative offset corrupts worst.
    assert_eq!(records[1].freq_channel, -7);

    for eph in &records {
        let [x, y, z] = eph.position;
        let r = (x * x + y * y + z * z).sqrt();
        assert!(
            (24_500e3..26_500e3).contains(&r),
            "R{} radius {r} m is outside the GLONASS band",
            eph.slot
        );
    }
}

/// A column that holds something which is not a number refuses the record, rather than reporting it
/// as zero.
///
/// This is the other half of the same defect. `parse_field` used to end in `.unwrap_or(0.0)`, so a
/// field it could not read was indistinguishable from a field the format left blank -- and blank
/// fields are ordinary, the BeiDou record in this very fixture has one. That is what made the
/// column error survivable: three of every four fields failed to parse and were reported as exactly
/// 0.0, and a record of zeros is a record, not an error. Blank still means zero; unreadable now
/// means the record is not returned at all.
#[test]
fn a_column_that_is_not_a_number_takes_its_record_out_rather_than_zeroing_a_field() {
    let baseline = parse_nav(FILE);
    assert_eq!(baseline.len(), 4);

    // The fixture's own blank column: BeiDou C25's sixth orbit line leaves field 2 empty, and that
    // record parses. Absence is a value in this format.
    let c25 = baseline
        .iter()
        .find(|e| e.system == GnssSystem::Beidou)
        .expect("the fixture carries a BeiDou record");
    assert_eq!(c25.week, 939, "the field after the blank one is still read");

    // Now corrupt one column of G01's sqrt(A) and nothing else. Under `unwrap_or(0.0)` this
    // returned a G01 with sqrt_a = 0 -- a satellite at the centre of the Earth, reported as fact.
    let broken = FILE.replace("5.154024417877e+03", "5.15402441X877e+03");
    assert_ne!(broken, FILE, "the corruption has to actually apply");
    let records = parse_nav(&broken);
    assert_eq!(
        records
            .iter()
            .map(|e| (e.system, e.prn))
            .collect::<Vec<_>>(),
        vec![
            (GnssSystem::Galileo, 8),
            (GnssSystem::Gps, 2),
            (GnssSystem::Beidou, 25),
        ],
        "G01 is dropped and the walk stays in step over the records after it"
    );
}
