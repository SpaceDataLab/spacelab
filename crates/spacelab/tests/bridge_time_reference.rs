//! The bridges' reference: the Rust API's time scales and Earth orientation on every IERS case in
//! the tree, to the bit (Ф31.B).
//!
//! The C ABI (`crates/spacelab-capi/tests/time_scales.rs`) and the Python bridge
//! (`crates/spacelab-python/tests/bindings.py`) read `fixtures/bridge_time_reference.tsv`. Each must
//! return every value in it to the bit and refuse every case it refuses with its own typed error.
//! This test is what makes the file the Rust API's: it rebuilds the file from the IERS fixtures and
//! the API, and fails on any byte that differs.
//!
//! The cases are the IERS files themselves, every row of each (rule 25): each step of
//! `leap-seconds.list` from 23:59:59 the day before to 00:00:00.5 after, the leap second included;
//! each tabulated day of every `finals2000A` and C04 excerpt at its node and at a quarter and half
//! day past it, so the interpolation, the gaps and the series' ends are all crossed. A fixture added
//! later makes the file stale, and this test fails until it is regenerated:
//!
//! ```text
//! SPACELAB_WRITE_BRIDGE_REFERENCE=1 cargo test -p spacelab --test bridge_time_reference
//! ```

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use spacelab::{EarthOrientationError, utc_to_ut1};
use spacelab_core::{
    CalendarError, Epoch, Tai, UtcDateTime,
    calendar::{civil_from_days, days_from_civil},
};
use spacelab_eop::{EopError, EopQuality, EopSeries};
use spacelab_fa::{LeapError, leap_table_status, tai_to_utc, utc_to_tai, utc_to_tdb, utc_to_tt};

const REFERENCE: &str = "tests/fixtures/bridge_time_reference.tsv";
const WRITE: &str = "SPACELAB_WRITE_BRIDGE_REFERENCE";

/// Days from 1970-01-01 to MJD 0.
const UNIX_DAYS_TO_MJD_EPOCH: i64 = 40_587;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A label as the bridges take it: `(year, month, day, hour, minute, second)`.
type Label = (i32, u8, u8, u8, u8, f64);

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

fn label_text(l: Label) -> String {
    format!("{}\t{}\t{}\t{}\t{}\t{}", l.0, l.1, l.2, l.3, l.4, bits(l.5))
}

fn date_text(d: (i32, u8, u8)) -> String {
    format!("{}\t{}\t{}", d.0, d.1, d.2)
}

/// The refusal kinds, one name per typed error the bridges map.
fn leap_kind(e: LeapError) -> &'static str {
    match e {
        LeapError::BeforeIntegerSecondEra => "before-1972",
        LeapError::SpuriousLeapSecond => "spurious-leap-second",
        LeapError::OutsideCalendar => "outside-calendar",
    }
}

fn eop_kind(e: &EopError) -> &'static str {
    match e {
        EopError::OutOfRange { .. } => "out-of-range",
        EopError::Gap { .. } => "gap",
        other => panic!("a query cannot fail with {other:?}"),
    }
}

fn calendar(l: Label) -> Result<UtcDateTime, CalendarError> {
    UtcDateTime::new(l.0, l.1, l.2, l.3, l.4, l.5)
}

/// The IERS sources, in the order the file lists them: the two `spacelab-eop` excerpts, then every
/// `finals2000A` snapshot `crates/spacelab/tests/fixtures` holds, by name.
fn sources() -> Vec<(String, &'static str, String, EopSeries)> {
    let mut out = Vec::new();
    for (name, format, path) in [
        (
            "finals2000A_excerpt",
            "finals2000a",
            "crates/spacelab-eop/tests/fixtures/finals2000A_excerpt.txt",
        ),
        (
            "eopc04_excerpt",
            "c04",
            "crates/spacelab-eop/tests/fixtures/eopc04_excerpt.txt",
        ),
    ] {
        out.push((name.to_owned(), format, path.to_owned()));
    }
    let mut snapshots: Vec<String> =
        std::fs::read_dir(root().join("crates/spacelab/tests/fixtures"))
            .expect("the fixture directory")
            .map(|e| {
                e.expect("a directory entry")
                    .file_name()
                    .into_string()
                    .expect("UTF-8")
            })
            .filter_map(|n| {
                let stem = n.strip_suffix(".txt")?;
                stem.starts_with("finals2000A_").then(|| stem.to_owned())
            })
            .collect();
    snapshots.sort();
    for name in snapshots {
        let path = format!("crates/spacelab/tests/fixtures/{name}.txt");
        out.push((name, "finals2000a", path));
    }
    out.into_iter()
        .map(|(name, format, path)| {
            let text = std::fs::read_to_string(root().join(&path)).expect("an IERS fixture");
            let series = match format {
                "c04" => EopSeries::parse_c04(&text),
                _ => EopSeries::parse_finals2000a(&text),
            }
            .unwrap_or_else(|e| panic!("{path}: {e}"));
            (name, format, path, series)
        })
        .collect()
}

/// Each step of `leap-seconds.list`: the day it takes effect and `TAI − UTC` from then on.
fn leap_steps() -> Vec<((i32, u8, u8), i32)> {
    let list =
        std::fs::read_to_string(root().join("crates/spacelab-fa/tests/fixtures/leap-seconds.list"))
            .expect("the IERS list");
    list.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let mut f = l.split_whitespace();
            let ntp: i64 = f.next().unwrap().parse().unwrap();
            let value: i32 = f.next().unwrap().parse().unwrap();
            // The list's own definition: MJD = NTP / 86400 + 15020.
            let day = ntp.div_euclid(86_400) + 15_020 - UNIX_DAYS_TO_MJD_EPOCH;
            (civil_from_days(day).unwrap(), value)
        })
        .collect()
}

fn day_before(d: (i32, u8, u8)) -> (i32, u8, u8) {
    civil_from_days(days_from_civil(d.0, d.1, d.2) - 1).unwrap()
}

/// The UTC labels: around every step of the list, past the list's expiry and continuity, the
/// int64 horizon of the C ABI's epochs, three days the leap-second table refuses, and labels the
/// calendar refuses.
fn utc_labels(steps: &[((i32, u8, u8), i32)]) -> Vec<Label> {
    let mut labels = Vec::new();
    for &(d, _) in steps {
        let (y, m, dd) = day_before(d);
        for s in [59.0, 59.5, 60.0, 60.75] {
            labels.push((y, m, dd, 23, 59, s));
        }
        labels.push((d.0, d.1, d.2, 0, 0, 0.0));
        labels.push((d.0, d.1, d.2, 0, 0, 0.5));
    }
    labels.extend([
        (1960, 1, 1, 0, 0, 0.0),
        (2018, 12, 31, 23, 59, 60.0),
        (2026, 12, 31, 23, 59, 60.0),
        (2027, 6, 30, 23, 59, 60.0),
        (2027, 5, 20, 0, 0, 0.0),
        (2027, 6, 28, 0, 0, 0.0),
        (2027, 6, 29, 12, 0, 0.0),
        (2100, 1, 1, 0, 0, 0.0),
        // Inside and outside ±2⁶³ ns of J2000, the C ABI's `int64_t` epoch.
        (2200, 1, 1, 0, 0, 0.0),
        (2300, 1, 1, 0, 0, 0.0),
        (2017, 1, 1, 0, 0, -0.0),
        (2017, 2, 29, 0, 0, 0.0),
        (2017, 13, 1, 0, 0, 0.0),
        (2017, 1, 1, 24, 0, 0.0),
        (2017, 1, 1, 0, 60, 0.0),
        (2017, 1, 1, 0, 0, 61.0),
        (2017, 1, 1, 0, 0, -1.0),
        (2017, 1, 1, 0, 0, f64::NAN),
    ]);
    labels
}

/// The midnight label of an integral MJD.
#[allow(clippy::cast_possible_truncation)] // a tabulated MJD is a whole day near 5e4
fn midnight(mjd: f64) -> (i32, u8, u8) {
    civil_from_days(mjd as i64 - UNIX_DAYS_TO_MJD_EPOCH).unwrap()
}

fn quality(q: EopQuality) -> &'static str {
    match q {
        EopQuality::Final => "final",
        EopQuality::Predicted => "predicted",
    }
}

/// The whole file, from the IERS fixtures and the Rust API.
#[allow(clippy::too_many_lines)] // one pass that writes every row kind of the file
fn reference() -> String {
    let mut out = String::new();
    out.push_str(
        "# The Rust API's time scales and Earth orientation on every IERS case in the tree, to the \
         bit (Ф31.B).\n\
         # Written by crates/spacelab/tests/bridge_time_reference.rs from the IERS fixtures; the C \
         ABI and the Python bridge must reproduce it.\n\
         # A double is the 16 hex digits of its bits; an epoch is integer nanoseconds since J2000 \
         on its own scale. Rows, tab-separated:\n\
         #   source <name> <finals2000a|c04> <path from the repository root>\n\
         #   status <first> <last step> <TAI-UTC from it> <list updated> <list expires> \
         <continuous UTC from>, each date year month day\n\
         #   span   <source> <first MJD> <last MJD>\n\
         #   utc    <year month day hour minute second> <TAI> <TT> <TDB> | refused <kind>\n\
         #   tai    <TAI, inside int64> <year month day hour minute second> | refused <kind>\n\
         #   eop    <source> <MJD, UTC> <xp yp rad> <dUT1 LOD s> <dX dY rad> <final|predicted> \
         <sigma xp yp rad> <sigma dUT1 s> | refused <kind>\n\
         #   ut1    <source> <year month day hour minute second> <UT1> | refused <kind>\n",
    );

    let sources = sources();
    for (name, format, path, _) in &sources {
        writeln!(out, "source\t{name}\t{format}\t{path}").unwrap();
    }
    let s = leap_table_status();
    writeln!(
        out,
        "status\t{}\t{}\t{}\t{}\t{}\t{}",
        date_text(s.first),
        date_text(s.last_step),
        s.last_delta_at,
        date_text(s.list_updated),
        date_text(s.list_expires),
        date_text(s.continuous_from)
    )
    .unwrap();
    for (name, _, _, series) in &sources {
        let (first, last) = series.span();
        writeln!(out, "span\t{name}\t{}\t{}", bits(first), bits(last)).unwrap();
    }

    let steps = leap_steps();
    let mut tai_inputs: Vec<i128> = Vec::new();
    for l in utc_labels(&steps) {
        let result = calendar(l).map_err(|_| "calendar").and_then(|utc| {
            let tai = utc_to_tai(utc).map_err(leap_kind)?;
            let tt = utc_to_tt(utc).map_err(leap_kind)?;
            let tdb = utc_to_tdb(utc).map_err(leap_kind)?;
            Ok((
                tai.ns_since_j2000(),
                tt.ns_since_j2000(),
                tdb.ns_since_j2000(),
            ))
        });
        match result {
            Ok((tai, tt, tdb)) => {
                writeln!(out, "utc\t{}\t{tai}\t{tt}\t{tdb}", label_text(l)).unwrap();
                tai_inputs.push(tai);
            }
            Err(kind) => writeln!(out, "utc\t{}\trefused\t{kind}", label_text(l)).unwrap(),
        }
    }

    // The way back, from every TAI the labels above produced that an `int64_t` holds, and from the
    // nanosecond before the table's first instant.
    let first_tai = utc_to_tai(UtcDateTime::from_date(1972, 1, 1).unwrap())
        .unwrap()
        .ns_since_j2000();
    tai_inputs.push(first_tai - 1);
    tai_inputs.retain(|&ns| i64::try_from(ns).is_ok());
    for ns in tai_inputs {
        match tai_to_utc(Epoch::<Tai>::from_ns_since_j2000(ns)) {
            Ok(u) => {
                let l = (
                    u.year(),
                    u.month(),
                    u.day(),
                    u.hour(),
                    u.minute(),
                    u.second(),
                );
                writeln!(out, "tai\t{ns}\t{}", label_text(l)).unwrap();
            }
            Err(e) => writeln!(out, "tai\t{ns}\trefused\t{}", leap_kind(e)).unwrap(),
        }
    }

    for (name, _, _, series) in &sources {
        let (first, _) = series.span();
        let mut mjds = vec![first - 0.5, f64::NAN];
        for sample in series.samples() {
            mjds.extend([sample.mjd, sample.mjd + 0.25, sample.mjd + 0.5]);
        }
        for mjd in mjds {
            match series.at_mjd(mjd) {
                Ok(e) => writeln!(
                    out,
                    "eop\t{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    bits(mjd),
                    bits(e.xp.radians()),
                    bits(e.yp.radians()),
                    bits(e.dut1_seconds),
                    bits(e.lod_seconds),
                    bits(e.dx.radians()),
                    bits(e.dy.radians()),
                    quality(e.quality),
                    bits(e.xp_sigma.radians()),
                    bits(e.yp_sigma.radians()),
                    bits(e.dut1_sigma_seconds),
                )
                .unwrap(),
                Err(e) => {
                    writeln!(out, "eop\t{name}\t{}\trefused\t{}", bits(mjd), eop_kind(&e)).unwrap();
                }
            }
        }
    }

    for (name, _, _, series) in &sources {
        let (_, last) = series.span();
        let mut labels: Vec<Label> = vec![
            (1960, 1, 1, 0, 0, 0.0),
            (2016, 12, 31, 23, 59, 59.5),
            (2016, 12, 31, 23, 59, 60.0),
            (2018, 12, 31, 23, 59, 60.0),
        ];
        let after = midnight(last + 1.0);
        labels.push((after.0, after.1, after.2, 0, 0, 0.0));
        for sample in series.samples() {
            let d = midnight(sample.mjd);
            labels.push((d.0, d.1, d.2, 0, 0, 0.0));
            labels.push((d.0, d.1, d.2, 6, 0, 0.0));
        }
        for l in labels {
            let utc = calendar(l).expect("a calendar label");
            match utc_to_ut1(utc, series) {
                Ok(ut1) => {
                    writeln!(
                        out,
                        "ut1\t{name}\t{}\t{}",
                        label_text(l),
                        ut1.ns_since_j2000()
                    )
                    .unwrap();
                }
                Err(e) => {
                    let kind = match e {
                        EarthOrientationError::Leap(e) => leap_kind(e),
                        EarthOrientationError::Eop(e) => eop_kind(&e),
                    };
                    writeln!(out, "ut1\t{name}\t{}\trefused\t{kind}", label_text(l)).unwrap();
                }
            }
        }
    }
    out
}

/// The committed file is the Rust API's answer, byte for byte.
#[test]
fn the_reference_is_the_rust_api_on_the_iers_cases() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(REFERENCE);
    let want = reference();
    if std::env::var_os(WRITE).is_some() {
        std::fs::write(&path, &want).expect("write the reference");
    }
    let have = std::fs::read_to_string(&path).unwrap_or_default();
    if have != want {
        let line = have
            .lines()
            .zip(want.lines())
            .position(|(a, b)| a != b)
            .map_or(have.lines().count().min(want.lines().count()) + 1, |i| {
                i + 1
            });
        panic!(
            "{REFERENCE} is not the Rust API's answer from line {line}: rerun with {WRITE}=1 after \
             a deliberate change, and read the diff before committing it"
        );
    }
}

/// The file covers its whole set: every step of the list and every tabulated day of every source
/// appears, so a case cannot drop out without this test saying which (rule 25). And the step rows
/// are the list's own `TAI − UTC`, exactly (G5).
#[test]
fn the_reference_covers_every_iers_row() {
    let text = reference();
    let rows: Vec<Vec<&str>> = text
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').collect())
        .collect();

    for ((y, m, d), value) in leap_steps() {
        let midnight = rows
            .iter()
            .find(|r| {
                r[0] == "utc"
                    && r[1..6]
                        == [
                            y.to_string(),
                            m.to_string(),
                            d.to_string(),
                            "0".into(),
                            "0".into(),
                        ]
                    && r[6] == bits(0.0)
            })
            .unwrap_or_else(|| panic!("no row for the step of {y}-{m:02}-{d:02}"));
        let naive = UtcDateTime::from_date(y, m, d)
            .unwrap()
            .naive_ns_since_j2000();
        let tai: i128 = midnight[7].parse().unwrap();
        let tt: i128 = midnight[8].parse().unwrap();
        assert_eq!(
            tai - naive,
            i128::from(value) * 1_000_000_000,
            "{y}-{m:02}-{d:02}"
        );
        assert_eq!(tt - tai, 32_184_000_000, "{y}-{m:02}-{d:02}");
    }

    for (name, _, _, series) in sources() {
        for sample in series.samples() {
            let node = bits(sample.mjd);
            assert!(
                rows.iter()
                    .any(|r| r[0] == "eop" && r[1] == name && r[2] == node),
                "{name}: no row at MJD {}",
                sample.mjd
            );
            let (y, m, d) = midnight(sample.mjd);
            assert!(
                rows.iter().any(|r| r[0] == "ut1"
                    && r[1] == name
                    && r[2..5] == [y.to_string(), m.to_string(), d.to_string()]),
                "{name}: no UT1 row on {y}-{m:02}-{d:02}"
            );
        }
    }
}
