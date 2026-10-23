//! Every field of a RINEX 3 observation file, read back off the fixture.
//!
//! The fixture is **hand-built to the RINEX 3.05 format specification**, not a vendored agency file:
//! the repository carries navigation, almanac and VMF products but no observation file, and an
//! observation reader cannot be tested against a file that does not exist. It is written at the
//! spec's fixed columns and carries, deliberately, the cases a synthetic file usually omits —
//! a record that stops before the header's last descriptor, a record with *interior* blanks, an
//! epoch flagged for a power failure, a receiver clock offset of each sign, and a loss-of-lock
//! indicator that is set on exactly one observation.
//!
//! What this test cannot do is catch a column convention that is wrong in both the fixture and the
//! reader. The mitigation is that the values were chosen so that a misread column is not a plausible
//! observation: pseudoranges are 2e7 m and phases 1e8 cycles, so a value read one field left or right
//! is off by an order of magnitude, and the trailing digits of every value are distinct.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::doc_markdown)] // RINEX, GLONASS, Doppler are names in prose

use spacelab_gnss::combinations::{FrequencyPair, carrier_frequency, geometry_free};
use spacelab_gnss::rinex_obs::{ObsCode, ObsKind, RinexObs};
use spacelab_gnss::{SatId, TimeSystem};

const FIXTURE: &str = include_str!("fixtures/rinex3_obs_handbuilt.24o");

/// 2024-01-09 12:00:00 is 8774 days after the J2000 epoch (2000-01-01 12:00): 24 years of 365 days,
/// six leap days (2000, 2004, 2008, 2012, 2016, 2020) and eight days into January.
const EPOCH_ONE_SECONDS: f64 = 8774.0 * 86_400.0;

fn code(kind: ObsKind, band: char, attribute: char) -> ObsCode {
    ObsCode {
        kind,
        band,
        attribute: Some(attribute),
    }
}

#[test]
fn every_header_field_reads_back() {
    let obs = RinexObs::parse(FIXTURE).unwrap();
    let h = &obs.header;

    assert!((h.version - 3.05).abs() < 1e-12, "version {}", h.version);
    assert_eq!(h.file_system, 'M');
    assert_eq!(h.marker_name, "SPDL00CYP");
    assert_eq!(h.time_system, TimeSystem::Gps);
    assert_eq!(h.interval, Some(30.0));

    let xyz = h.approx_position.expect("APPROX POSITION XYZ");
    // The eccentricity is non-zero in H only, so a reader that transposed the triple would show it.
    let hen = h.antenna_delta_hen.expect("ANTENNA: DELTA H/E/N");
    for (got, want, name) in [
        (xyz[0], 4_075_539.888, "X"),
        (xyz[1], 931_735.216, "Y"),
        (xyz[2], 4_801_629.145, "Z"),
        (hen[0], 1.534, "H"),
        (hen[1], 0.0, "E"),
        (hen[2], 0.0, "N"),
    ] {
        assert!((got - want).abs() < 1e-9, "{name}: {got} vs {want}");
    }

    assert_eq!(
        h.obs_types.keys().copied().collect::<Vec<_>>(),
        vec!['G', 'R']
    );
    assert_eq!(
        h.obs_types[&'G'],
        vec![
            code(ObsKind::Pseudorange, '1', 'C'),
            code(ObsKind::CarrierPhase, '1', 'C'),
            code(ObsKind::SignalStrength, '1', 'C'),
            code(ObsKind::Pseudorange, '2', 'W'),
            code(ObsKind::CarrierPhase, '2', 'W'),
            code(ObsKind::SignalStrength, '2', 'W'),
        ]
    );
    assert_eq!(
        h.obs_types[&'R'],
        vec![
            code(ObsKind::Pseudorange, '1', 'C'),
            code(ObsKind::CarrierPhase, '1', 'C'),
            code(ObsKind::Pseudorange, '2', 'P'),
            code(ObsKind::CarrierPhase, '2', 'P'),
        ]
    );
}

#[test]
fn every_observation_field_of_the_first_epoch_reads_back() {
    let obs = RinexObs::parse(FIXTURE).unwrap();
    assert_eq!(obs.epochs.len(), 2);

    let first = &obs.epochs[0];
    assert!(
        (first.seconds_j2000 - EPOCH_ONE_SECONDS).abs() < 1e-9,
        "epoch one {} vs {EPOCH_ONE_SECONDS}",
        first.seconds_j2000
    );
    assert!(!first.power_failure);
    let clock = first.receiver_clock_offset.expect("clock offset");
    assert!((clock - 1.234_56e-7).abs() < 1e-18, "clock {clock:.18}");
    assert_eq!(first.satellites.len(), 3);

    // G07: every one of the six declared observations present, with its own quality digits.
    let g07 = &first.satellites[0];
    assert_eq!(
        g07.sat,
        SatId {
            system: 'G',
            prn: 7
        }
    );
    let values: Vec<Option<f64>> = g07.values.iter().map(|v| v.map(|o| o.value)).collect();
    assert_eq!(
        values,
        vec![
            Some(23_629_347.915),
            Some(124_144_623.325),
            Some(45.500),
            Some(23_629_350.021),
            Some(96_745_123.114),
            Some(41.000),
        ]
    );
    let lli: Vec<Option<u8>> = g07.values.iter().map(|v| v.and_then(|o| o.lli)).collect();
    assert_eq!(lli, vec![None, Some(0), None, None, Some(1), None]);
    let snr: Vec<Option<u8>> = g07
        .values
        .iter()
        .map(|v| v.and_then(|o| o.signal_strength))
        .collect();
    assert_eq!(snr, vec![Some(6), Some(7), None, Some(5), Some(6), None]);
    // Exactly one observation on this record declares a loss of lock — the L2W phase.
    let lost: Vec<bool> = g07
        .values
        .iter()
        .map(|v| v.is_some_and(|o| o.lost_lock()))
        .collect();
    assert_eq!(lost, vec![false, false, false, false, true, false]);

    // G30: the record stops after three of the six descriptors, and the rest are absent, not zero.
    let g30 = &first.satellites[1];
    assert_eq!(
        g30.sat,
        SatId {
            system: 'G',
            prn: 30
        }
    );
    assert_eq!(
        g30.values.len(),
        6,
        "slots follow the header, not the record"
    );
    assert_eq!(
        g30.values
            .iter()
            .map(|v| v.map(|o| o.value))
            .collect::<Vec<_>>(),
        vec![
            Some(21_112_233.445),
            Some(110_987_654.321),
            Some(50.000),
            None,
            None,
            None,
        ]
    );

    // R05: a different constellation, and so a different descriptor list and a different slot count.
    let r05 = &first.satellites[2];
    assert_eq!(
        r05.sat,
        SatId {
            system: 'R',
            prn: 5
        }
    );
    assert_eq!(r05.values.len(), 4);
    assert_eq!(
        r05.values
            .iter()
            .map(|v| v.map(|o| o.value))
            .collect::<Vec<_>>(),
        vec![
            Some(19_876_543.210),
            Some(106_333_222.111),
            Some(19_876_545.678),
            Some(82_666_111.000),
        ]
    );
}

#[test]
fn the_second_epoch_reads_back() {
    let obs = RinexObs::parse(FIXTURE).unwrap();
    let second = &obs.epochs[1];
    assert!(
        (second.seconds_j2000 - EPOCH_ONE_SECONDS - 30.0).abs() < 1e-9,
        "epoch two {}",
        second.seconds_j2000
    );
    assert!(second.power_failure, "epoch flag 1 was dropped");
    let clock = second.receiver_clock_offset.expect("clock offset");
    assert!((clock + 6.543_21e-7).abs() < 1e-18, "clock {clock:.18}");

    // An interior blank: slots 0 and 2 absent, 1 and 3 present. A reader that packed the values it
    // found rather than indexing by column would shift these two into the wrong slots.
    let r05 = &second.satellites[1];
    assert_eq!(
        r05.values
            .iter()
            .map(|v| v.map(|o| o.value))
            .collect::<Vec<_>>(),
        vec![None, Some(106_333_199.900), None, Some(82_666_095.500)]
    );
}

/// The header's descriptor order is the record's value order, and [`RinexObs::slot_of`] is how a
/// caller reaches an observation by name instead of by an index it guessed.
#[test]
fn a_named_observation_is_reachable_through_its_slot() {
    let obs = RinexObs::parse(FIXTURE).unwrap();
    let l2w = obs
        .slot_of('G', code(ObsKind::CarrierPhase, '2', 'W'))
        .expect("L2W declared for GPS");
    assert_eq!(l2w, 4);
    let value = obs.epochs[0].satellites[0].values[l2w].unwrap().value;
    assert!((value - 96_745_123.114).abs() < 1e-6);

    // A descriptor the header does not declare has no slot, and neither does a constellation the
    // file never observed.
    assert!(obs.slot_of('G', code(ObsKind::Doppler, '1', 'C')).is_none());
    assert!(
        obs.slot_of('E', code(ObsKind::Pseudorange, '1', 'C'))
            .is_none()
    );
}

/// The parsed observables feed the linear combinations, which is the reason the reader exists —
/// and the check that the units came out of the file the way the format defines them.
///
/// RINEX reports phase in cycles, so it is scaled by the wavelength before the combination sees it.
/// Measured on the fixture's G07 record at the first epoch: the geometry-free code combination is
/// **−2.106000 m**, the ionosphere-free code combination **23 629 344.659697 m** and the
/// Melbourne–Wübbena combination **−13 215.445848 m**, the last dominated by the wide-lane ambiguity
/// these synthetic phases carry. All three were computed independently, in exact decimal arithmetic
/// from the fixture's printed values and the `n × 10.23 MHz` carriers.
#[test]
fn the_parsed_observables_feed_the_combinations() {
    let obs = RinexObs::parse(FIXTURE).unwrap();
    let slot = |kind, band, attribute| {
        obs.slot_of('G', code(kind, band, attribute))
            .expect("declared")
    };
    let g07 = &obs.epochs[0].satellites[0];
    let value = |s: usize| g07.values[s].expect("present").value;

    let pair = FrequencyPair::GPS_L1_L2;
    let p1 = value(slot(ObsKind::Pseudorange, '1', 'C'));
    let p2 = value(slot(ObsKind::Pseudorange, '2', 'W'));
    let l1 = value(slot(ObsKind::CarrierPhase, '1', 'C')) * pair.wavelength1();
    let l2 = value(slot(ObsKind::CarrierPhase, '2', 'W')) * pair.wavelength2();

    let gf = geometry_free(p1, p2);
    assert!((gf - -2.106_000).abs() < 1e-6, "GF {gf:.6}");
    let mw = pair.melbourne_wubbena(p1, p2, l1, l2);
    assert!((mw - -13_215.445_848).abs() < 1e-5, "MW {mw:.6}");
    let iff = pair.ionosphere_free(p1, p2);
    assert!((iff - 23_629_344.659_697).abs() < 1e-5, "IF {iff:.6}");

    // The band characters the header carries resolve to the GPS carriers the pair was built from.
    assert_eq!(carrier_frequency('G', '1'), Some(pair.f1()));
    assert_eq!(carrier_frequency('G', '2'), Some(pair.f2()));
}
