//! The seam: measured Earth orientation, wired into the frame chain.
//!
//! The fixtures are Skyfield's ISS ground track computed **with** the polar-motion
//! table from the same `finals2000A` snapshot we parse. Skyfield's frame code and
//! its geodetic solver are its own; only the input data is shared.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab::{
    EarthOrientationError, EarthOrientationQuality, EopSource, earth_orientation_at,
    teme_to_itrf_at, tt_minus_tai_ns,
};
use spacelab_core::{Angle, Epoch, Ut1, UtcDateTime};
use spacelab_earth::itrf_to_geodetic;
use spacelab_eop::{EopQuality, EopSeries};
use spacelab_fa::{LeapError, delta_at, utc_to_tai};
use spacelab_frames::teme_to_itrf;
use spacelab_sgp4::Satellite;
use spacelab_testkit::nan_max;

const EXCERPT: &str = include_str!("../../spacelab-eop/tests/fixtures/finals2000A_excerpt.txt");
const TRACK: &str = include_str!("fixtures/iss_groundtrack_eop_skyfield.csv");

const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// The full published series, when a snapshot has been fetched. The committed
/// excerpt does not span 2008, so the ground-track test needs the real file.
fn full_series() -> Option<EopSeries> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/eop");
    let path = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "txt"))?;
    EopSeries::parse_finals2000a(&std::fs::read_to_string(path).ok()?).ok()
}

/// The three coordinates, checked by hand against the definitions.
///
/// `TAI = naive + ΔAT` and `TT = TAI + 32.184 s` are exact. `UT1 = naive + ΔUT1` is
/// measured. Nothing else should have moved.
#[test]
fn the_time_scales_are_offsets_on_the_naive_coordinate() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("the excerpt parses");

    // A day the excerpt covers, well inside a contiguous block.
    let utc = UtcDateTime::new(2016, 12, 30, 6, 0, 0.0).expect("a real instant");
    let naive = utc.naive_ns_since_j2000();

    let e = earth_orientation_at(utc, EopSource::Series(&series)).expect("in range");

    let expected_tt =
        naive + i128::from(delta_at(utc).unwrap()) * 1_000_000_000 + tt_minus_tai_ns();
    assert_eq!(e.orientation.tt.ns_since_j2000(), expected_tt);

    let dut1_ns = e.orientation.ut1.ns_since_j2000() - naive;
    #[allow(clippy::cast_precision_loss)]
    let dut1_seconds = dut1_ns as f64 / 1e9;
    // Between the tabulated 2016-12-31 and 2016-12-30 values (`delta_ut1_carries_the_sign_…`
    // below), widened by the sub-daily libration, which stays under 10 µs.
    assert!(
        (-0.407_760_1 - 5e-5..=-0.406_918_0 + 5e-5).contains(&dut1_seconds),
        "ΔUT1 {dut1_seconds} s is not between the two tabulated days"
    );

    assert_eq!(
        e.quality.tabulated(),
        Some(EopQuality::Final),
        "2016 is long since measured"
    );
    let sigma = e
        .quality
        .dut1_sigma_seconds()
        .expect("a tabulated day reports one");
    assert!(sigma > 0.0 && sigma < 1e-3);
}

#[test]
fn a_predicted_day_is_reported_as_predicted() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("parses");
    // The excerpt straddles the measured/predicted boundary; find a predicted day.
    let predicted = series
        .samples()
        .iter()
        .find(|s| s.bulletin_a.quality == EopQuality::Predicted)
        .expect("the excerpt contains predictions");

    #[allow(clippy::cast_possible_truncation)]
    let mjd = predicted.mjd as i64;
    let days_from_civil = mjd - 40_587;
    let (y, m, d) = spacelab_core::calendar::civil_from_days(days_from_civil);
    let utc = UtcDateTime::from_date(y, m, d).expect("a real date");

    match earth_orientation_at(utc, EopSource::Series(&series)) {
        Ok(e) => assert_eq!(e.quality.tabulated(), Some(EopQuality::Predicted)),
        // Past the leap-second horizon is also a legitimate refusal.
        Err(e) => println!("refused, as it may: {e}"),
    }
}

/// Refusing to extrapolate is behaviour, not an accident.
#[test]
fn we_do_not_invent_an_orientation_outside_the_series() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("parses");
    let ancient = UtcDateTime::from_date(1960, 1, 1).expect("a real date");
    assert!(earth_orientation_at(ancient, EopSource::Series(&series)).is_err());
}

/// The seam, end to end, against Skyfield fed from the same measurements.
#[test]
#[ignore = "needs a snapshot from tools/fetch-eop.sh"]
fn the_iss_ground_track_with_real_earth_orientation() {
    let Some(series) = full_series() else {
        panic!("run tools/fetch-eop.sh first");
    };
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the ISS element set parses");

    let (mut worst_lat, mut worst_lon, mut worst_h) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut checked = 0;

    for line in TRACK
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
    {
        let f: Vec<&str> = line.split(',').collect();
        let utc = UtcDateTime::new(
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
            f[5].parse().unwrap(),
        )
        .expect("a real instant");

        let e = earth_orientation_at(utc, EopSource::Series(&series)).expect("2008 is tabulated");

        // Our ΔUT1 comes from the IERS file; Skyfield's comes from its ΔT tables.
        // They should agree closely, and a disagreement here would be a finding.
        let theirs = Epoch::<Ut1>::from_ns_since_j2000(f[6].parse::<i128>().unwrap());
        let dut1_gap_ns = e.orientation.ut1.ns_since_j2000() - theirs.ns_since_j2000();
        assert!(
            dut1_gap_ns.abs() < 2_000_000,
            "our ΔUT1 and Skyfield's differ by {dut1_gap_ns} ns"
        );

        let (transform, quality) =
            teme_to_itrf_at(utc, EopSource::Series(&series)).expect("in range");
        assert_eq!(quality.tabulated(), Some(EopQuality::Final));

        let teme = satellite
            .propagate_at(utc)
            .expect("the ISS has not decayed");
        let geo = itrf_to_geodetic(transform.state(teme).position).expect("in orbit");

        let (lat, lon, h): (f64, f64, f64) = (
            f[7].parse().unwrap(),
            f[8].parse().unwrap(),
            f[9].parse().unwrap(),
        );
        let dlat = (geo.latitude.degrees() - lat).abs();
        let dlon = {
            let d = (geo.longitude.degrees() - lon).rem_euclid(360.0);
            d.min(360.0 - d)
        };
        let dh = (geo.height.metres() - h).abs();

        worst_lat = nan_max(worst_lat, dlat);
        worst_lon = nan_max(worst_lon, dlon);
        worst_h = nan_max(worst_h, dh);
        checked += 1;
    }

    println!(
        "with EOP: {checked} points; worst dlat {worst_lat:.2e}°, dlon {worst_lon:.2e}°, dh {worst_h:.2e} m"
    );
    assert!(checked >= 40);
    // 1e-6° is 11 cm. Our ΔUT1 is interpolated linearly and Skyfield's comes from a
    // different table, so a few centimetres of disagreement is expected and honest.
    assert!(worst_lat < 1e-6, "latitude worst case {worst_lat:.3e}°");
    assert!(worst_lon < 1e-6, "longitude worst case {worst_lon:.3e}°");
    assert!(worst_h < 0.1, "height worst case {worst_h:.3e} m");
}

/// What ignoring the Earth's orientation actually costs, measured rather than quoted.
#[test]
#[ignore = "needs a snapshot from tools/fetch-eop.sh"]
fn neglecting_earth_orientation_moves_the_ground_track() {
    let Some(series) = full_series() else {
        panic!("run tools/fetch-eop.sh first");
    };
    let satellite = Satellite::from_tle(LINE1, LINE2).unwrap();
    let utc = UtcDateTime::new(2008, 9, 20, 12, 0, 0.0).unwrap();

    let teme = satellite.propagate_at(utc).unwrap();

    let (with_eop, _) = teme_to_itrf_at(utc, EopSource::Series(&series)).unwrap();
    let (without, quality) = teme_to_itrf_at(utc, EopSource::NoEop).unwrap();
    assert_eq!(quality, EarthOrientationQuality::Absent);

    let separation =
        (with_eop.state(teme).position.vector - without.state(teme).position.vector).norm();
    println!("ignoring EOP displaces the ISS by {separation:.1} m");
    assert!(
        separation > 10.0,
        "EOP should matter by tens of metres at least; saw {separation:.3} m"
    );
}

/// [`EopSource::NoEop`] is exactly the zeroed pole and `UT1 = UTC`, and reports that it is.
///
/// It is *not* zeroed TT: the leap-second count is known exactly and is not an Earth-orientation
/// measurement, so `NoEop` still carries the real TT into the TIO locator. That distinction is
/// worth about a tenth of a millimetre and is asserted here only so it cannot drift into meaning
/// something else.
#[test]
// The three zero comparisons below are *about* exactness: `NoEop` sets these fields to a literal
// zero, and a tolerance would let a small non-zero pole through as if it were no pole at all.
#[allow(clippy::float_cmp)]
fn no_eop_is_the_zeroed_transform_and_says_so() {
    let utc = UtcDateTime::new(2016, 12, 30, 6, 0, 0.0).expect("a real instant");
    let naive = utc.naive_ns_since_j2000();

    let e = earth_orientation_at(utc, EopSource::NoEop).expect("leap seconds cover 2016");
    assert_eq!(e.quality, EarthOrientationQuality::Absent);
    assert_eq!(e.quality.tabulated(), None);
    assert_eq!(e.quality.dut1_sigma_seconds(), None);
    assert_eq!(e.orientation.ut1.ns_since_j2000(), naive);
    assert_eq!(e.orientation.xp.radians(), 0.0);
    assert_eq!(e.orientation.yp.radians(), 0.0);
    assert_eq!(e.orientation.lod_seconds, 0.0);

    // TT is the real one, so `NoEop` and the tabulated arm differ only in what a table supplies.
    let tabulated = earth_orientation_at(
        utc,
        EopSource::Series(&EopSeries::parse_finals2000a(EXCERPT).expect("the excerpt parses")),
    )
    .expect("in range");
    assert_eq!(e.orientation.tt, tabulated.orientation.tt);

    let (transform, _) = teme_to_itrf_at(utc, EopSource::NoEop).expect("leap seconds cover 2016");
    assert_eq!(
        transform,
        teme_to_itrf(
            e.orientation.ut1,
            e.orientation.tt,
            Angle::from_radians(0.0),
            Angle::from_radians(0.0),
            0.0,
        )
    );
}

/// `teme_to_itrf_at` is exactly `teme_to_itrf` fed the orientation `earth_orientation_at` returns —
/// verified on the committed excerpt (so no fetched snapshot is needed, unlike the ground-track test).
#[test]
fn teme_to_itrf_at_is_the_transform_from_the_measured_orientation() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("the excerpt parses");
    let utc = UtcDateTime::new(2016, 12, 30, 6, 0, 0.0).expect("a real instant");

    let (transform, quality) = teme_to_itrf_at(utc, EopSource::Series(&series)).expect("in range");
    let e = earth_orientation_at(utc, EopSource::Series(&series)).expect("in range");
    let manual = teme_to_itrf(
        e.orientation.ut1,
        e.orientation.tt,
        e.orientation.xp,
        e.orientation.yp,
        e.orientation.lod_seconds,
    );
    assert_eq!(
        transform, manual,
        "the wrapper must not alter the transform"
    );
    assert_eq!(quality, e.quality);
    assert_eq!(quality.tabulated(), Some(EopQuality::Final));
}

/// `UT1 − TAI` at the half seconds either side of a leap day's `23:59:60`, and inside it.
fn ut1_minus_tai_ns(series: &EopSeries, year: i32, month: u8, day: u8) -> [i128; 3] {
    let next = UtcDateTime::from_date(year, month, day)
        .unwrap()
        .days_from_civil()
        + 1;
    let (ny, nm, nd) = spacelab_core::calendar::civil_from_days(next);
    [
        UtcDateTime::new(year, month, day, 23, 59, 59.5).unwrap(),
        UtcDateTime::new(year, month, day, 23, 59, 60.5).unwrap(),
        UtcDateTime::new(ny, nm, nd, 0, 0, 0.5).unwrap(),
    ]
    .map(|utc| {
        let e = earth_orientation_at(utc, EopSource::Series(series)).expect("tabulated");
        e.orientation.ut1.ns_since_j2000() - utc_to_tai(utc).unwrap().ns_since_j2000()
    })
}

/// UT1 runs through a leap second without a step. `UT1 − TAI` drifts at the rate the excess length
/// of day sets, `−LOD/86400`: the worst second of the 26 cases moves it −37 ns, at 1977-12-31,
/// where the excess was 3.3 ms. Inside the leap second the label's MJD falls in the next day's row,
/// whose `ΔUT1` carries one more second of `ΔAT`; read on the label's own naive coordinate, UT1
/// stood a whole second ahead for that second. The committed excerpt holds the 2016 leap second; a
/// fetched snapshot adds the 25 from 1973-12-31 on.
#[test]
fn ut1_runs_through_a_leap_second_without_a_step() {
    let excerpt = EopSeries::parse_finals2000a(EXCERPT).expect("the excerpt parses");
    let full = full_series();
    let mut cases = vec![(&excerpt, 2016, 12, 31)];
    if let Some(full) = &full {
        for year in 1973..=2016 {
            for (month, day, next) in [(6_u8, 30_u8, (year, 7_u8)), (12, 31, (year + 1, 1))] {
                let before = delta_at(UtcDateTime::from_date(year, month, day).unwrap()).unwrap();
                let after = delta_at(UtcDateTime::from_date(next.0, next.1, 1).unwrap()).unwrap();
                if after != before {
                    cases.push((full, year, month, day));
                }
            }
        }
        assert_eq!(
            cases.len(),
            26,
            "the snapshot spans 25 leap seconds from 1973-12-31"
        );
    }
    for &(series, year, month, day) in &cases {
        let [a, b, c] = ut1_minus_tai_ns(series, year, month, day);
        for step in [b - a, c - b] {
            assert!(
                step.abs() < 1_000,
                "{year}-{month:02}-{day:02}: UT1 − TAI moved {step} ns in one second"
            );
        }
    }
}

/// A `23:59:60` on a day without a leap second is refused at the leap stage on both arms, before
/// it can alias the next midnight in TAI and in UT1.
#[test]
fn a_spurious_leap_second_label_is_refused() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("parses");
    let utc = UtcDateTime::new(2018, 12, 31, 23, 59, 60.0).unwrap();
    for source in [EopSource::NoEop, EopSource::Series(&series)] {
        let err = earth_orientation_at(utc, source).unwrap_err();
        assert!(
            matches!(
                err,
                EarthOrientationError::Leap(LeapError::SpuriousLeapSecond)
            ),
            "{err}"
        );
    }
}

/// Each stage of the chain reports its own failure, and each prints a message. The leap-second table
/// refuses a date before it begins; the orientation series refuses a date it does not tabulate.
#[test]
fn each_stage_refuses_with_its_own_error_and_message() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("parses");

    // Before the leap-second table (1960): the leap stage refuses.
    let leap = earth_orientation_at(
        UtcDateTime::from_date(1960, 1, 1).unwrap(),
        EopSource::Series(&series),
    )
    .unwrap_err();
    assert!(matches!(leap, EarthOrientationError::Leap(_)));
    assert!(leap.to_string().contains("leap"));

    // In the leap era but outside the excerpt (2010): the orientation stage refuses.
    let eop = earth_orientation_at(
        UtcDateTime::from_date(2010, 1, 1).unwrap(),
        EopSource::Series(&series),
    )
    .unwrap_err();
    assert!(matches!(eop, EarthOrientationError::Eop(_)));
    assert!(eop.to_string().contains("orientation"));
}

/// A bound cannot see the sign or the size of what it bounds: a `ΔUT1` negated, or dropped to
/// zero, satisfies `|ΔUT1| < 0.9 s` just as well, and from 2027-05-20 no leap second keeps it. Both are worth 0.4 to 0.8 s of
/// Earth rotation, 190 to 380 m at the equator. Pin the value instead, against the Bulletin A
/// column of `finals2000A` itself rather than against our own parse of it.
///
/// The tabulated value is daily; `at_mjd` adds the IERS sub-daily libration on top, which is why
/// this is a band and not an equality. Measured at these two nodes the correction is 7.4 µs and
/// 6.9 µs, four orders of magnitude inside the band and five below either mutation.
#[test]
fn delta_ut1_carries_the_sign_and_size_the_iers_bulletin_gives_it() {
    let series = EopSeries::parse_finals2000a(EXCERPT).expect("the excerpt parses");
    // `161230 57752.00 I ... I-0.4069180 0.0000079` and the row after it, read off the fixture.
    for (day, want_seconds, want_sigma) in [
        (30_u8, -0.406_918_0_f64, 7.9e-6),
        (31, -0.407_760_1, 7.8e-6),
    ] {
        let utc = UtcDateTime::new(2016, 12, day, 0, 0, 0.0).expect("a real instant");
        let e = earth_orientation_at(utc, EopSource::Series(&series)).expect("in range");
        let got_ns = e.orientation.ut1.ns_since_j2000() - utc.naive_ns_since_j2000();
        #[allow(clippy::cast_possible_truncation)] // under a second, so the product is tiny
        let want_ns = (want_seconds * 1e9).round() as i128;
        assert!(
            (got_ns - want_ns).abs() < 50_000,
            "2016-12-{day}: ΔUT1 {got_ns} ns is not the tabulated {want_ns} ns"
        );
        let sigma = e
            .quality
            .dut1_sigma_seconds()
            .expect("a tabulated day reports one");
        assert!(
            (sigma - want_sigma).abs() < 1e-9,
            "2016-12-{day}: σ(ΔUT1) {sigma} is not the tabulated {want_sigma}"
        );
    }
}
