//! The bridges' reference for access windows: the Rust API's windows on a set of SGP4 scans, to
//! the bit (Ф31.C).
//!
//! The C ABI (`crates/spacelab-capi/tests/access.rs`) and the Python bridge
//! (`crates/spacelab-python/tests/bindings.py`) read `fixtures/bridge_access_reference.tsv`. Each
//! must return every window in it with the same bits, and refuse every case it refuses with its own
//! typed error. This file makes the reference the Rust API's: it rebuilds it from
//! `spacelab::access` and fails on any byte that differs.
//!
//! It also holds the API to its mathematics. Every edge is a root of the elevation constraint (G2);
//! a dense scan at a fixed step finds no window the API missed (S2); and SGP4's rate bound, which
//! is a hypothesis (H-rate, `spacelab::access`), is sampled on every case and must hold. A scan with
//! a rate bound a thousand times too small is shown to miss a window the dense scan finds, so the
//! S2 check is known to fail when it should (rule 16).
//!
//! ```text
//! SPACELAB_WRITE_BRIDGE_ACCESS=1 cargo test -p spacelab --test bridge_access_reference
//! ```

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use spacelab::access::{
    AccessError, Station, Window, elevation_mask, sgp4_itrf_position, sgp4_rate_bound, sgp4_windows,
};
use spacelab::{EarthOrientationError, EopSource};
use spacelab_core::UtcDateTime;
use spacelab_eop::{EopError, EopSeries};
use spacelab_fa::LeapError;
use spacelab_sgp4::{PropagationError, Satellite};

const REFERENCE: &str = "tests/fixtures/bridge_access_reference.tsv";
const WRITE: &str = "SPACELAB_WRITE_BRIDGE_ACCESS";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// A label as the bridges take it: `(year, month, day, hour, minute, second)`.
type Label = (i32, u8, u8, u8, u8, f64);

fn label_text(l: Label) -> String {
    format!("{}\t{}\t{}\t{}\t{}\t{}", l.0, l.1, l.2, l.3, l.4, bits(l.5))
}

/// The two lines of catalogue object `id` from a TLE fixture, either the `line1|line2|…` rows of
/// the SGP4 verification set or a plain three-line file.
fn tle(path: &str, id: &str) -> (String, String) {
    let text = std::fs::read_to_string(root().join(path)).expect("a TLE fixture");
    let key = format!("1 {id}");
    if let Some(row) = text
        .lines()
        .find(|l| l.starts_with(&key) && l.contains('|'))
    {
        let mut f = row.split('|');
        return (f.next().unwrap().to_owned(), f.next().unwrap().to_owned());
    }
    let lines: Vec<&str> = text.lines().collect();
    let i = lines
        .iter()
        .position(|l| l.starts_with(&key))
        .expect("the object is in the fixture");
    (
        lines[i].trim_end().to_owned(),
        lines[i + 1].trim_end().to_owned(),
    )
}

/// One scan of the reference.
struct Case {
    name: &'static str,
    tle: (String, String),
    /// The EOP fixture, as a path from the repository root, or `None` for `EopSource::NoEop`.
    eop: Option<&'static str>,
    start: Label,
    span: f64,
    mask_deg: f64,
    tol: f64,
    /// Geodetic latitude and longitude in degrees, height in metres.
    stations: Vec<(f64, f64, f64)>,
}

const SGP4_VER: &str = "crates/spacelab-sgp4/tests/fixtures/sgp4_ver_tle.txt";
const STATIONS_TLE: &str = "crates/spacelab-sgp4/tests/fixtures/omm_stations.tle";
const ISS_2008: (&str, &str) = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
);
const EOP_2008: &str = "crates/spacelab/tests/fixtures/finals2000A_2008264.txt";
const EOP_2025: &str = "crates/spacelab/tests/fixtures/finals2000A_2025001.txt";
const EOP_2026: &str = "crates/spacelab/tests/fixtures/finals2000A_2026224.txt";

/// The stations most cases share: the Orekit comparison's (35°N, 33°E), Svalbard, the Andes, the
/// Cape, the North Pole and a point below the ellipsoid in the Pacific.
fn stations() -> Vec<(f64, f64, f64)> {
    vec![
        (35.0, 33.0, 0.0),
        (78.23, 15.39, 500.0),
        (-0.2, -78.5, 2800.0),
        (-33.9, 18.4, 0.0),
        (90.0, 0.0, 0.0),
        (-45.0, 170.0, -30.0),
    ]
}

#[expect(clippy::too_many_lines, reason = "the case list, one literal per case")]
fn cases() -> Vec<Case> {
    let iss = (ISS_2008.0.to_owned(), ISS_2008.1.to_owned());
    let base = |name, tle: (String, String), eop, start, span, mask_deg| Case {
        name,
        tle,
        eop,
        start,
        span,
        mask_deg,
        tol: 1e-4,
        stations: stations(),
    };
    let mut out = vec![
        base(
            "iss-2008-eop",
            iss.clone(),
            Some(EOP_2008),
            (2008, 9, 20, 12, 0, 0.0),
            86_400.0,
            10.0,
        ),
        base(
            "iss-2008-no-eop",
            iss.clone(),
            None,
            (2008, 9, 20, 12, 0, 0.0),
            86_400.0,
            0.0,
        ),
        base(
            "iss-2026-eop",
            tle(STATIONS_TLE, "25544"),
            Some(EOP_2026),
            (2026, 8, 15, 0, 0, 0.0),
            86_400.0,
            5.0,
        ),
        base(
            "sun-synchronous-28057",
            tle(SGP4_VER, "28057"),
            None,
            (2006, 6, 27, 0, 0, 0.0),
            172_800.0,
            5.0,
        ),
        base(
            "molniya-08195",
            tle(SGP4_VER, "08195"),
            None,
            (2006, 6, 25, 12, 0, 0.0),
            172_800.0,
            10.0,
        ),
        Case {
            stations: vec![
                (0.0, 0.0, 0.0),
                (0.0, 90.0, 0.0),
                (0.0, 180.0, 0.0),
                (0.0, -90.0, 0.0),
                (60.0, 0.0, 0.0),
                (75.0, -60.0, 0.0),
            ],
            ..base(
                "geostationary-28626",
                tle(SGP4_VER, "28626"),
                None,
                (2006, 6, 26, 0, 0, 0.0),
                86_400.0,
                5.0,
            )
        },
        Case {
            stations: vec![(35.0, 33.0, 0.0)],
            ..base(
                "iss-2008-negative-mask",
                iss.clone(),
                None,
                (2008, 9, 20, 12, 0, 0.0),
                21_600.0,
                -1.0,
            )
        },
    ];
    // The refusals, one per kind the API can return.
    let one = vec![(35.0, 33.0, 0.0)];
    let refused = |name, tle: (String, String), eop, start, span, mask_deg, tol, stations| Case {
        name,
        tle,
        eop,
        start,
        span,
        mask_deg,
        tol,
        stations,
    };
    let start = (2008, 9, 20, 12, 0, 0.0);
    out.extend([
        refused(
            "refused-mask",
            iss.clone(),
            None,
            start,
            3_600.0,
            90.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-span-zero",
            iss.clone(),
            None,
            start,
            0.0,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-span-nan",
            iss.clone(),
            None,
            start,
            f64::NAN,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-tolerance",
            iss.clone(),
            None,
            start,
            3_600.0,
            10.0,
            0.0,
            one.clone(),
        ),
        refused(
            "refused-station",
            iss.clone(),
            None,
            start,
            3_600.0,
            10.0,
            1e-4,
            vec![(35.0, 33.0, 0.0), (91.0, 0.0, 0.0)],
        ),
        refused(
            "refused-eop-out-of-range",
            iss.clone(),
            Some(EOP_2025),
            start,
            3_600.0,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-eop-ends-inside",
            iss.clone(),
            Some(EOP_2008),
            start,
            864_000.0,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-before-1972",
            iss.clone(),
            None,
            (1960, 1, 1, 0, 0, 0.0),
            3_600.0,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-decayed-28872",
            tle(SGP4_VER, "28872"),
            None,
            (2005, 11, 29, 0, 0, 0.0),
            864_000.0,
            10.0,
            1e-4,
            one,
        ),
    ]);
    out
}

/// The refusal kinds, one name per typed error the bridges map.
fn kind(e: &AccessError) -> &'static str {
    match e {
        AccessError::Station => "station",
        AccessError::Mask => "mask",
        AccessError::Interval => "interval",
        AccessError::Tolerance => "tolerance",
        AccessError::Samples => panic!("an SGP4 scan has no table to refuse"),
        AccessError::BelowSurface { .. } => panic!("a station scan has no zone to refuse for"),
        AccessError::EarthOrientation { error, .. } => match error {
            EarthOrientationError::Leap(LeapError::BeforeIntegerSecondEra) => "before-1972",
            EarthOrientationError::Leap(LeapError::SpuriousLeapSecond) => "spurious-leap-second",
            EarthOrientationError::Leap(LeapError::OutsideCalendar) => "outside-calendar",
            EarthOrientationError::Eop(EopError::OutOfRange { .. }) => "out-of-range",
            EarthOrientationError::Eop(EopError::Gap { .. }) => "gap",
            EarthOrientationError::Eop(other) => panic!("a query cannot fail with {other:?}"),
        },
        AccessError::Propagation { error, .. } => match error {
            PropagationError::Decayed => "decayed",
            PropagationError::Diverged => "diverged",
            other => panic!("a built satellite cannot fail with {other:?}"),
        },
    }
}

fn series(path: &str) -> EopSeries {
    let text = std::fs::read_to_string(root().join(path)).expect("an IERS fixture");
    EopSeries::parse_finals2000a(&text).expect("the EOP fixture parses")
}

/// A case's scan of every station, in station order, as the bridges batch it: the first refusal
/// refuses the batch.
fn run(case: &Case, series: Option<&EopSeries>) -> Result<Vec<Vec<Window>>, AccessError> {
    let satellite = Satellite::from_tle(&case.tle.0, &case.tle.1).expect("the TLE parses");
    let source = series.map_or(EopSource::NoEop, EopSource::Series);
    let l = case.start;
    let start = UtcDateTime::new(l.0, l.1, l.2, l.3, l.4, l.5).expect("a calendar label");
    let stations = case
        .stations
        .iter()
        .map(|&(lat, lon, h)| Station::geodetic(lat.to_radians(), lon.to_radians(), h))
        .collect::<Result<Vec<_>, _>>()?;
    stations
        .iter()
        .map(|s| {
            sgp4_windows(
                &satellite,
                source,
                start,
                case.span,
                s,
                case.mask_deg.to_radians(),
                case.tol,
            )
        })
        .collect()
}

/// The whole file, from the cases and the Rust API.
fn reference() -> String {
    let mut out = String::new();
    out.push_str(
        "# The Rust API's access windows (spacelab::access::sgp4_windows) on its reference scans, \
         to the bit (Ф31.C).\n\
         # Written by crates/spacelab/tests/bridge_access_reference.rs; the C ABI and the Python \
         bridge must reproduce it. A double is the 16 hex digits of its bits.\n\
         # Angles are radians, as every bridge takes them; times are seconds past the start label \
         on the naive UTC coordinate. Rows, tab-separated:\n\
         #   scan    <case> <TLE line 1> <TLE line 2> <EOP path from the repository root | none> \
         <start year month day hour minute second> <span s> <mask rad> <tol s>\n\
         #   station <case> <index> <latitude rad> <longitude rad> <height m>\n\
         #   window  <case> <station index> <rise s> <set s>\n\
         #   refused <case> <kind>\n",
    );
    for case in cases() {
        let eop = case.eop.map(series);
        writeln!(
            out,
            "scan\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.name,
            case.tle.0,
            case.tle.1,
            case.eop.unwrap_or("none"),
            label_text(case.start),
            bits(case.span),
            bits(case.mask_deg.to_radians()),
            bits(case.tol),
        )
        .unwrap();
        for (i, &(lat, lon, h)) in case.stations.iter().enumerate() {
            writeln!(
                out,
                "station\t{}\t{i}\t{}\t{}\t{}",
                case.name,
                bits(lat.to_radians()),
                bits(lon.to_radians()),
                bits(h)
            )
            .unwrap();
        }
        match run(&case, eop.as_ref()) {
            Ok(per_station) => {
                for (i, ws) in per_station.iter().enumerate() {
                    for w in ws {
                        writeln!(
                            out,
                            "window\t{}\t{i}\t{}\t{}",
                            case.name,
                            bits(w.rise),
                            bits(w.set)
                        )
                        .unwrap();
                    }
                }
            }
            Err(e) => writeln!(out, "refused\t{}\t{}", case.name, kind(&e)).unwrap(),
        }
    }
    out
}

/// The committed file is the Rust API's answer, byte for byte.
#[test]
fn the_reference_is_the_rust_api_on_its_scans() {
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

/// Every refusal kind the API has is in the reference, so a bridge that mismaps one is caught.
#[test]
fn the_reference_refuses_every_kind_the_api_has() {
    let text = reference();
    let kinds: std::collections::BTreeSet<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("refused\t"))
        .map(|l| l.split('\t').nth(1).unwrap())
        .collect();
    for want in [
        "mask",
        "interval",
        "tolerance",
        "station",
        "out-of-range",
        "before-1972",
        "decayed",
    ] {
        assert!(kinds.contains(want), "no refused case of kind {want}");
    }
}

/// The satellite of a case, sampled every `h` seconds across its span, once for all its stations.
struct Dense {
    h: f64,
    t: Vec<f64>,
    r: Vec<spacelab_core::Vec3>,
}

fn dense(case: &Case, series: Option<&EopSeries>, step: f64) -> Dense {
    let satellite = Satellite::from_tle(&case.tle.0, &case.tle.1).unwrap();
    let source = series.map_or(EopSource::NoEop, EopSource::Series);
    let label = case.start;
    let start = UtcDateTime::new(label.0, label.1, label.2, label.3, label.4, label.5).unwrap();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive span over a positive step, far inside usize"
    )]
    let count = (case.span / step) as usize;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample index below 2^53 is exact in f64"
    )]
    let times: Vec<f64> = (0..=count).map(|k| k as f64 * step).collect();
    let positions = times
        .iter()
        .map(|&at| sgp4_itrf_position(&satellite, source, start, at).unwrap())
        .collect();
    Dense {
        h: step,
        t: times,
        r: positions,
    }
}

/// The samples of `d` above the mask that no window holds, to within `tol`.
fn missed(d: &Dense, station: &Station, mask_rad: f64, windows: &[Window], tol: f64) -> usize {
    let mask = elevation_mask(mask_rad).unwrap();
    d.t.iter()
        .zip(&d.r)
        .filter(|&(&t, &r)| {
            station.g(&mask, r) > 0.0
                && !windows
                    .iter()
                    .any(|w| w.rise - tol <= t && t <= w.set + tol)
        })
        .count()
}

/// The scans that succeed, with their windows.
fn answered() -> Vec<(Case, Option<EopSeries>, Vec<Vec<Window>>)> {
    cases()
        .into_iter()
        .filter_map(|c| {
            let eop = c.eop.map(series);
            let w = run(&c, eop.as_ref()).ok()?;
            Some((c, eop, w))
        })
        .collect()
}

/// The dense scan's step, s.
const DENSE_STEP: f64 = 0.5;

/// (S2) A fixed-step scan every half second finds the satellite above the mask only inside a
/// window the API reported, and every window longer than a step holds a sample above the mask. The
/// step is stated against the shortest window found, which the test prints.
#[test]
fn a_dense_scan_finds_no_window_the_api_missed() {
    let mut shortest = f64::INFINITY;
    let (mut samples, mut windows) = (0_usize, 0_usize);
    for (case, eop, per_station) in answered() {
        let d = dense(&case, eop.as_ref(), DENSE_STEP);
        let mask_rad = case.mask_deg.to_radians();
        let mask = elevation_mask(mask_rad).unwrap();
        for (&(lat, lon, h), ws) in case.stations.iter().zip(&per_station) {
            let station = Station::geodetic(lat.to_radians(), lon.to_radians(), h).unwrap();
            let lost = missed(&d, &station, mask_rad, ws, case.tol);
            assert_eq!(
                lost, 0,
                "{}: {lost} samples above the mask outside every window",
                case.name
            );
            for w in ws {
                shortest = shortest.min(w.duration());
                if w.duration() > 2.0 * d.h {
                    let inside =
                        d.t.iter()
                            .zip(&d.r)
                            .any(|(&t, &r)| w.rise < t && t < w.set && station.g(&mask, r) > 0.0);
                    assert!(inside, "{}: no sample above the mask in {w:?}", case.name);
                }
            }
            samples += d.t.len();
            windows += ws.len();
        }
    }
    println!(
        "{windows} windows, {samples} samples at a {DENSE_STEP} s step; the shortest window is \
         {shortest:.3} s"
    );
    assert!(
        shortest > 20.0 * DENSE_STEP,
        "the dense step is not far below every window"
    );
}

/// (G2) Every edge the API reports is a root of `g`: the constraint changes sign within twice the
/// tolerance of it, rising at a rise and falling at a set, unless the edge is the span's own end.
#[test]
fn every_edge_is_a_root_of_the_elevation_constraint() {
    let mut edges = 0;
    for (case, eop, per_station) in answered() {
        let satellite = Satellite::from_tle(&case.tle.0, &case.tle.1).unwrap();
        let source = eop.as_ref().map_or(EopSource::NoEop, EopSource::Series);
        let l = case.start;
        let start = UtcDateTime::new(l.0, l.1, l.2, l.3, l.4, l.5).unwrap();
        let mask = elevation_mask(case.mask_deg.to_radians()).unwrap();
        let delta = 2.0 * case.tol;
        for (&(lat, lon, h), ws) in case.stations.iter().zip(&per_station) {
            let station = Station::geodetic(lat.to_radians(), lon.to_radians(), h).unwrap();
            let g = |t: f64| {
                station.g(
                    &mask,
                    sgp4_itrf_position(&satellite, source, start, t).unwrap(),
                )
            };
            for w in ws {
                if w.rise > 0.0 {
                    assert!(
                        g(w.rise - delta) < 0.0 && g(w.rise + delta) > 0.0,
                        "{}: {w:?}",
                        case.name
                    );
                    edges += 1;
                }
                if w.set < case.span {
                    assert!(
                        g(w.set - delta) > 0.0 && g(w.set + delta) < 0.0,
                        "{}: {w:?}",
                        case.name
                    );
                    edges += 1;
                }
            }
        }
    }
    println!("{edges} edges, each a sign change of g within twice its tolerance");
    assert!(edges > 100, "too few edges to mean anything: {edges}");
}

/// (H-rate, tested) Every difference quotient of `g` on the dense samples is a value of `dg/dt`
/// somewhere between them, by the mean-value theorem, so one above the bound would falsify it.
/// None is; the test prints the largest share of the bound one reaches.
#[test]
fn the_sgp4_rate_bound_holds_on_every_reference_case() {
    let mut worst = 0.0_f64;
    for (case, eop, _) in answered() {
        let satellite = Satellite::from_tle(&case.tle.0, &case.tle.1).unwrap();
        let mask = elevation_mask(case.mask_deg.to_radians()).unwrap();
        let bound = sgp4_rate_bound(&satellite, &mask);
        let d = dense(&case, eop.as_ref(), DENSE_STEP);
        for &(lat, lon, h) in &case.stations {
            let station = Station::geodetic(lat.to_radians(), lon.to_radians(), h).unwrap();
            for pair in d.r.windows(2) {
                let rate = (station.g(&mask, pair[1]) - station.g(&mask, pair[0])).abs() / d.h;
                worst = worst.max(rate / bound);
            }
        }
    }
    println!("the largest difference quotient of g is {worst:.4} of H-rate's bound");
    assert!(
        worst < 1.0,
        "H-rate is falsified: a quotient reaches {worst} of the bound"
    );
}

/// (Rule 16) The S2 check fails when the scan misses a window: the same scan under a rate bound a
/// thousand times too small steps over passes, and the dense scan finds them.
#[test]
fn a_rate_bound_too_small_misses_windows_the_dense_scan_finds() {
    let case = cases()
        .into_iter()
        .find(|c| c.name == "iss-2008-no-eop")
        .unwrap();
    let satellite = Satellite::from_tle(&case.tle.0, &case.tle.1).unwrap();
    let l = case.start;
    let start = UtcDateTime::new(l.0, l.1, l.2, l.3, l.4, l.5).unwrap();
    let mask_rad = case.mask_deg.to_radians();
    let mask = elevation_mask(mask_rad).unwrap();
    let (lat, lon, h) = case.stations[0];
    let station = Station::geodetic(lat.to_radians(), lon.to_radians(), h).unwrap();
    let g = |t: f64| {
        station.g(
            &mask,
            sgp4_itrf_position(&satellite, EopSource::NoEop, start, t).unwrap(),
        )
    };
    let mut planted = Vec::new();
    spacelab_coverage::windows(
        g,
        sgp4_rate_bound(&satellite, &mask) / 1000.0,
        0.0,
        case.span,
        case.tol,
        |w| planted.push(w),
    );
    let d = dense(&case, None, DENSE_STEP);
    let lost = missed(&d, &station, mask_rad, &planted, case.tol);
    println!(
        "a bound 1000x too small: {} windows, {lost} samples above the mask outside them",
        planted.len()
    );
    assert!(
        lost > 0,
        "the dense scan did not catch a scan that steps over passes"
    );
}
