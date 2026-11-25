//! The bridges' reference for access windows on a tabulated ephemeris: the Rust API's windows on a
//! set of tables, to the bit (Ф31.C).
//!
//! The C ABI (`crates/spacelab-capi/tests/access.rs`) and the Python bridge
//! (`crates/spacelab-python/tests/bindings.py`) read
//! `fixtures/bridge_access_tabulated_reference.tsv`, which carries every sample, station and window
//! as the bits of its doubles. Each bridge must return every window in it with the same bits, and
//! refuse every case it refuses with its own typed error. This file makes the reference the Rust
//! API's: it rebuilds it from `spacelab::access` and fails on any byte that differs.
//!
//! The tables are an SP3 product (the ILRS LAGEOS-1 combination with its velocity records), a
//! two-body orbit sampled every two minutes in ITRF from its closed form, a straight pass, and a
//! fixed point. The file also holds the API to its mathematics: the interpolant takes every sample
//! exactly, every edge is a root of the elevation constraint (G2), a dense scan of the interpolant
//! at a fixed step finds no window the API missed (S2), and no difference quotient of `g` exceeds
//! the hull bound, which is proved (`derivations/access_hermite_speed.mac`). A scan under a bound a
//! thousand times too small is shown to miss windows the dense scan finds, so the S2 check fails
//! when it should (rule 16).
//!
//! ```text
//! SPACELAB_WRITE_BRIDGE_ACCESS=1 cargo test -p spacelab --test bridge_access_tabulated
//! ```

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use std::fmt::Write as _;
use std::path::Path;

use spacelab::access::{
    AccessError, Station, Tabulated, Window, elevation_mask, tabulated_rate_bound,
    tabulated_windows,
};
use spacelab_core::Vec3;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_gnss::sp3::{SatId, Sp3};

const REFERENCE: &str = "tests/fixtures/bridge_access_tabulated_reference.tsv";
const WRITE: &str = "SPACELAB_WRITE_BRIDGE_ACCESS";
const LAGEOS_SP3: &str = "tests/fixtures/ilrsa_lageos1_20250106.sp3";

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// One sample: time (s), ITRF position (m) and velocity (m/s).
type Sample = (f64, [f64; 3], [f64; 3]);

/// One table of the reference.
struct Case {
    name: &'static str,
    samples: Vec<Sample>,
    mask_deg: f64,
    tol: f64,
    /// Geodetic latitude and longitude in degrees, height in metres.
    stations: Vec<(f64, f64, f64)>,
}

/// The first `count` epochs of the ILRS LAGEOS-1 product: its `P` records through
/// [`Sp3::positions`], and its `V` records read here with the reader's own arithmetic (decimetres
/// per second times 0.1), since the reader keeps them for its interpolator only.
fn lageos(count: usize) -> Vec<Sample> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(LAGEOS_SP3);
    let text = std::fs::read_to_string(&path).expect("the LAGEOS SP3 fixture");
    let sp3 = Sp3::parse(&text);
    let positions = sp3
        .positions(SatId {
            system: 'L',
            prn: 51,
        })
        .expect("LAGEOS-1 is in the file");
    let velocities: Vec<[f64; 3]> = text
        .lines()
        .filter_map(|l| l.strip_prefix("VL51"))
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .take(3)
                .map(|x| x.parse().unwrap())
                .collect();
            [v[0] * 0.1, v[1] * 0.1, v[2] * 0.1]
        })
        .collect();
    assert_eq!(velocities.len(), sp3.epoch_count(), "a V record per epoch");
    sp3.epochs()
        .iter()
        .zip(positions)
        .zip(&velocities)
        .take(count)
        .map(|((&t, p), &v)| (t, p.expect("a P record at every epoch"), v))
        .collect()
}

/// A circular orbit of radius `radius`, inclination `incl` and node `node` (radians), sampled every
/// `step` seconds `count` times from `origin` in the frame turning with the Earth at
/// [`EARTH_ROTATION_RATE`]: `r_E = R₃(−θ) r_I` and `v_E = R₃(−θ) v_I − ω ẑ × r_E`, with
/// `θ = ω (t − origin)`.
fn two_body(
    radius: f64,
    incl: f64,
    node: f64,
    origin: f64,
    step: f64,
    count: usize,
) -> Vec<Sample> {
    const MU: f64 = 3.986_004_418e14;
    let motion = libm::sqrt(MU / (radius * radius * radius));
    let (si, ci) = libm::sincos(incl);
    let (sn, cn) = libm::sincos(node);
    let to_node = [cn, sn, 0.0];
    let past_node = [-ci * sn, ci * cn, si];
    (0..count)
        .map(|k| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "a sample index below 2^53 is exact in f64"
            )]
            let dt = k as f64 * step;
            let (su, cu) = libm::sincos(motion * dt);
            let (st, ct) = libm::sincos(EARTH_ROTATION_RATE * dt);
            let r: [f64; 3] =
                core::array::from_fn(|j| radius * (cu * to_node[j] + su * past_node[j]));
            let v: [f64; 3] =
                core::array::from_fn(|j| radius * motion * (-su * to_node[j] + cu * past_node[j]));
            let re = [ct * r[0] + st * r[1], -st * r[0] + ct * r[1], r[2]];
            let ve = [
                ct * v[0] + st * v[1] + EARTH_ROTATION_RATE * re[1],
                -st * v[0] + ct * v[1] - EARTH_ROTATION_RATE * re[0],
                v[2],
            ];
            (origin + dt, re, ve)
        })
        .collect()
}

#[expect(clippy::too_many_lines, reason = "the case list, one literal per case")]
fn cases() -> Vec<Case> {
    let slr = vec![
        (-29.0464, 115.3467, 244.0),
        (47.0671, 15.4933, 539.0),
        (50.8674, 0.3361, 75.0),
        (39.0206, -76.8277, 19.0),
        (-35.3161, 149.0099, 805.0),
        (43.7905, 125.4434, 274.0),
    ];
    // Seconds past J2000 near 2025, so that the scan runs where an ulp of the time is 0.12 µs.
    let sso = two_body(
        6_378_137.0 + 700e3,
        98.19_f64.to_radians(),
        30.0_f64.to_radians(),
        7.9e8,
        120.0,
        361,
    );
    let ground = vec![
        (78.23, 15.39, 500.0),
        (5.2515, -52.8047, 15.0),
        (90.0, 0.0, 0.0),
        (-77.85, 166.67, 10.0),
        (35.0, 33.0, 0.0),
    ];
    let above = Station::geodetic(35.0_f64.to_radians(), 33.0_f64.to_radians(), 20e3)
        .unwrap()
        .position();
    let fixed = vec![
        (0.0, [above.x, above.y, above.z], [0.0; 3]),
        (600.0, [above.x, above.y, above.z], [0.0; 3]),
    ];
    let head = sso[..3].to_vec();
    let one = vec![(35.0, 33.0, 0.0)];
    let refused = |name, samples, mask_deg, tol, stations| Case {
        name,
        samples,
        mask_deg,
        tol,
        stations,
    };
    let mut repeated = head.clone();
    repeated[2].0 = repeated[1].0;
    let mut decreasing = head.clone();
    decreasing[2].0 = decreasing[1].0 - 30.0;
    let mut nan_time = head.clone();
    nan_time[1].0 = f64::NAN;
    let mut nan_position = head.clone();
    nan_position[1].1[2] = f64::NAN;
    let mut infinite_velocity = head.clone();
    infinite_velocity[0].2[0] = f64::INFINITY;
    vec![
        Case {
            name: "lageos1-sp3-2025-01-05",
            samples: lageos(361),
            mask_deg: 10.0,
            tol: 1e-4,
            stations: slr,
        },
        Case {
            name: "two-body-sso-700km",
            samples: sso,
            mask_deg: 5.0,
            tol: 1e-4,
            stations: ground,
        },
        // Uniform motion 500 km over the equator at 0° E, which the interpolant holds exactly: the
        // satellite is above a 60° mask while |y| ≤ 500 km / tan 60°, from 18.76 s to 101.24 s.
        Case {
            name: "straight-pass-over-the-equator",
            samples: [0.0, 60.0, 120.0]
                .map(|t| (t, [6_878_137.0, -420e3 + 7e3 * t, 0.0], [0.0, 7e3, 0.0]))
                .to_vec(),
            mask_deg: 60.0,
            tol: 1e-4,
            stations: vec![(0.0, 0.0, 0.0)],
        },
        Case {
            name: "fixed-point-20km-up",
            samples: fixed,
            mask_deg: 10.0,
            tol: 1e-4,
            stations: vec![(35.0, 33.0, 0.0), (-35.0, -147.0, 0.0)],
        },
        refused(
            "refused-one-sample",
            head[..1].to_vec(),
            10.0,
            1e-4,
            one.clone(),
        ),
        refused("refused-repeated-time", repeated, 10.0, 1e-4, one.clone()),
        refused(
            "refused-decreasing-time",
            decreasing,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused("refused-nan-time", nan_time, 10.0, 1e-4, one.clone()),
        refused(
            "refused-nan-position",
            nan_position,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused(
            "refused-infinite-velocity",
            infinite_velocity,
            10.0,
            1e-4,
            one.clone(),
        ),
        refused("refused-mask", head.clone(), 90.0, 1e-4, one.clone()),
        refused("refused-tolerance", head.clone(), 10.0, 0.0, one.clone()),
        refused(
            "refused-station",
            head,
            10.0,
            1e-4,
            vec![(35.0, 33.0, 0.0), (91.0, 0.0, 0.0)],
        ),
    ]
}

/// The refusal kinds, one name per typed error a table scan can return.
fn kind(e: &AccessError) -> &'static str {
    match e {
        AccessError::Samples => "samples",
        AccessError::Station => "station",
        AccessError::Mask => "mask",
        AccessError::Tolerance => "tolerance",
        other => panic!("a table scan cannot fail with {other:?}"),
    }
}

fn table(samples: &[Sample]) -> Result<Tabulated, AccessError> {
    let v = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
    Tabulated::new(
        samples.iter().map(|s| s.0).collect(),
        samples.iter().map(|s| v(s.1)).collect(),
        samples.iter().map(|s| v(s.2)).collect(),
    )
}

fn station(&(lat, lon, h): &(f64, f64, f64)) -> Result<Station, AccessError> {
    Station::geodetic(lat.to_radians(), lon.to_radians(), h)
}

/// A case's scan of every station, in station order, as the bridges batch it: the table is built
/// first, then the stations, then each scan; the first refusal refuses the batch.
fn run(case: &Case) -> Result<Vec<Vec<Window>>, AccessError> {
    let eph = table(&case.samples)?;
    let stations = case
        .stations
        .iter()
        .map(station)
        .collect::<Result<Vec<_>, _>>()?;
    stations
        .iter()
        .map(|s| tabulated_windows(&eph, s, case.mask_deg.to_radians(), case.tol))
        .collect()
}

/// The whole file, from the cases and the Rust API.
fn reference() -> String {
    let mut out = String::new();
    out.push_str(
        "# The Rust API's access windows on a tabulated ephemeris \
         (spacelab::access::tabulated_windows), to the bit (Ф31.C).\n\
         # Written by crates/spacelab/tests/bridge_access_tabulated.rs; the C ABI and the Python \
         bridge must reproduce it. A double is the 16 hex digits of its bits.\n\
         # The LAGEOS-1 samples are the ILRS combination in \
         tests/fixtures/ilrsa_lageos1_20250106.sp3: ILRS data, cited as Pearlman et al. (2019), \
         doi:10.1007/s00190-019-01241-1.\n\
         # Angles are radians, as every bridge takes them; positions are ITRF metres, velocities \
         metres per second, and times the table's own seconds. Rows, tab-separated:\n\
         #   table   <case> <sample count> <mask rad> <tol s>\n\
         #   sample  <case> <index> <t s> <x> <y> <z> <vx> <vy> <vz>\n\
         #   station <case> <index> <latitude rad> <longitude rad> <height m>\n\
         #   window  <case> <station index> <rise s> <set s>\n\
         #   refused <case> <kind>\n",
    );
    for case in cases() {
        writeln!(
            out,
            "table\t{}\t{}\t{}\t{}",
            case.name,
            case.samples.len(),
            bits(case.mask_deg.to_radians()),
            bits(case.tol),
        )
        .unwrap();
        for (i, (t, r, v)) in case.samples.iter().enumerate() {
            let row: Vec<String> = [*t, r[0], r[1], r[2], v[0], v[1], v[2]]
                .iter()
                .map(|&x| bits(x))
                .collect();
            writeln!(out, "sample\t{}\t{i}\t{}", case.name, row.join("\t")).unwrap();
        }
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
        match run(&case) {
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
fn the_tabulated_reference_is_the_rust_api_on_its_tables() {
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

/// Every refusal kind a table scan has is in the reference, so a bridge that mismaps one is caught.
#[test]
fn the_tabulated_reference_refuses_every_kind_the_api_has() {
    let text = reference();
    let kinds: std::collections::BTreeSet<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("refused\t"))
        .map(|l| l.split('\t').nth(1).unwrap())
        .collect();
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        ["mask", "samples", "station", "tolerance"],
        "the reference does not refuse every kind"
    );
}

/// The checks a table makes that the reference's bridges cannot reach: arrays of unequal lengths
/// (the C ABI has one count for all three), and a position asked for outside the span.
#[test]
fn a_table_refuses_unequal_arrays_and_answers_none_outside_its_span() {
    let samples = &cases()[1].samples[..4];
    let v = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
    let times: Vec<f64> = samples.iter().map(|s| s.0).collect();
    let positions: Vec<Vec3> = samples.iter().map(|s| v(s.1)).collect();
    let velocities: Vec<Vec3> = samples.iter().map(|s| v(s.2)).collect();
    assert_eq!(
        Tabulated::new(times.clone(), positions[..3].to_vec(), velocities.clone()),
        Err(AccessError::Samples)
    );
    assert_eq!(
        Tabulated::new(times.clone(), positions.clone(), velocities[..3].to_vec()),
        Err(AccessError::Samples)
    );
    let eph = Tabulated::new(times.clone(), positions, velocities).unwrap();
    assert_eq!(eph.span(), (times[0], times[3]));
    assert!(eph.position(times[0] - 1e-3).is_none());
    assert!(eph.position(times[3] + 1e-3).is_none());
    assert!(eph.position(f64::NAN).is_none());
}

/// The speed bound is the derivation's hull, computed here apart from the API on two intervals: one
/// at rest at both ends, where the middle hull vector `3(r₁ − r₀)/h` is the largest and the bound
/// is twice the interpolant's peak speed `1.5|r₁ − r₀|/h` at mid-interval, and one in uniform
/// motion, where all three hull vectors are the velocity and the bound is reached.
#[test]
fn the_speed_bound_is_the_hull_of_the_derivation() {
    let rest = Tabulated::new(
        vec![100.0, 110.0],
        vec![Vec3::new(7e6, 0.0, 0.0), Vec3::new(7e6 + 300.0, 400.0, 0.0)],
        vec![Vec3::new(0.0, 0.0, 0.0); 2],
    )
    .unwrap();
    assert!(
        (rest.speed_bound() / 150.0 - 1.0).abs() < 1e-13,
        "{}",
        rest.speed_bound()
    );
    let mid = rest.position(105.0 + 1e-3).unwrap() - rest.position(105.0 - 1e-3).unwrap();
    assert!(
        (mid.norm() / 2e-3 / 75.0 - 1.0).abs() < 1e-6,
        "{}",
        mid.norm() / 2e-3
    );
    let v = Vec3::new(-1_200.0, 7_000.0, 2_500.0);
    let uniform = Tabulated::new(
        vec![0.0, 60.0],
        vec![
            Vec3::new(7e6, 0.0, 0.0),
            Vec3::new(7e6, 0.0, 0.0) + v.scale(60.0),
        ],
        vec![v, v],
    )
    .unwrap();
    assert!(
        (uniform.speed_bound() / v.norm() - 1.0).abs() < 1e-13,
        "{}",
        uniform.speed_bound()
    );
}

/// The interpolant takes every sample of every table exactly, so consecutive intervals meet with
/// no gap and a window edge at a sample is the sample's.
#[test]
fn the_interpolant_takes_every_sample_to_the_bit() {
    let mut count = 0;
    for case in cases() {
        let Ok(eph) = table(&case.samples) else {
            continue;
        };
        for &(t, r, _) in &case.samples {
            let p = eph.position(t).unwrap();
            assert_eq!(
                [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()],
                r.map(f64::to_bits),
                "{} at {t}",
                case.name
            );
            count += 1;
        }
    }
    assert!(count > 700, "too few samples: {count}");
}

/// The answered tables, with their windows.
fn answered() -> Vec<(Case, Tabulated, Vec<Vec<Window>>)> {
    cases()
        .into_iter()
        .filter_map(|c| {
            let w = run(&c).ok()?;
            let eph = table(&c.samples).unwrap();
            Some((c, eph, w))
        })
        .collect()
}

/// The dense scan's step, s.
const DENSE_STEP: f64 = 0.5;

/// The interpolant of `eph` every `step` seconds across its span, the last sample included.
fn dense(eph: &Tabulated, step: f64) -> (Vec<f64>, Vec<Vec3>) {
    let (first, last) = eph.span();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive span over a positive step, far inside usize"
    )]
    let count = ((last - first) / step) as usize;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample index below 2^53 is exact in f64"
    )]
    let mut times: Vec<f64> = (0..=count).map(|k| first + k as f64 * step).collect();
    if *times.last().unwrap() < last {
        times.push(last);
    }
    let positions = times.iter().map(|&t| eph.position(t).unwrap()).collect();
    (times, positions)
}

/// The dense samples above the mask that no window holds, to within `tol`.
fn missed(
    d: &(Vec<f64>, Vec<Vec3>),
    station: &Station,
    mask_rad: f64,
    windows: &[Window],
    tol: f64,
) -> usize {
    let mask = elevation_mask(mask_rad).unwrap();
    d.0.iter()
        .zip(&d.1)
        .filter(|&(&t, &r)| {
            station.g(&mask, r) > 0.0
                && !windows
                    .iter()
                    .any(|w| w.rise - tol <= t && t <= w.set + tol)
        })
        .count()
}

/// (S2) A fixed-step scan of the interpolant every half second finds it above the mask only inside
/// a window the API reported, and every window longer than two steps holds a sample above the
/// mask. The step is stated against the shortest window found, which the test prints.
#[test]
fn a_dense_scan_finds_no_window_the_api_missed() {
    let mut shortest = f64::INFINITY;
    let (mut samples, mut windows) = (0_usize, 0_usize);
    for (case, eph, per_station) in answered() {
        let d = dense(&eph, DENSE_STEP);
        let mask_rad = case.mask_deg.to_radians();
        let mask = elevation_mask(mask_rad).unwrap();
        for (spec, ws) in case.stations.iter().zip(&per_station) {
            let station = station(spec).unwrap();
            let lost = missed(&d, &station, mask_rad, ws, case.tol);
            assert_eq!(
                lost, 0,
                "{}: {lost} samples above the mask outside every window",
                case.name
            );
            for w in ws {
                shortest = shortest.min(w.duration());
                if w.duration() > 2.0 * DENSE_STEP {
                    let inside =
                        d.0.iter()
                            .zip(&d.1)
                            .any(|(&t, &r)| w.rise < t && t < w.set && station.g(&mask, r) > 0.0);
                    assert!(inside, "{}: no sample above the mask in {w:?}", case.name);
                }
            }
            samples += d.0.len();
            windows += ws.len();
        }
    }
    println!(
        "{windows} windows, {samples} samples at a {DENSE_STEP} s step; the shortest window is \
         {shortest:.3} s"
    );
    assert!(windows > 40, "the reference lost its windows: {windows}");
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
    for (case, eph, per_station) in answered() {
        let mask = elevation_mask(case.mask_deg.to_radians()).unwrap();
        let (first, last) = eph.span();
        let delta = 2.0 * case.tol;
        for (spec, ws) in case.stations.iter().zip(&per_station) {
            let station = station(spec).unwrap();
            let g = |t: f64| station.g(&mask, eph.position(t).unwrap());
            for w in ws {
                if w.rise > first {
                    assert!(
                        g(w.rise - delta) < 0.0 && g(w.rise + delta) > 0.0,
                        "{}: {w:?}",
                        case.name
                    );
                    edges += 1;
                }
                if w.set < last {
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
    assert!(edges > 70, "too few edges to mean anything: {edges}");
}

/// The hull bound is proved; this samples it. Every difference quotient of `g` on the dense
/// samples is a value of `dg/dt` between them, by the mean-value theorem, so one above the bound
/// would contradict the proof. None is; the test prints the largest share of the bound one reaches.
#[test]
fn no_difference_quotient_of_g_exceeds_the_hull_bound() {
    let mut worst = 0.0_f64;
    for (case, eph, _) in answered() {
        let mask = elevation_mask(case.mask_deg.to_radians()).unwrap();
        let bound = tabulated_rate_bound(&eph, &mask);
        let (t, r) = dense(&eph, DENSE_STEP);
        for spec in &case.stations {
            let station = station(spec).unwrap();
            for k in 1..t.len() {
                let step = t[k] - t[k - 1];
                let change = (station.g(&mask, r[k]) - station.g(&mask, r[k - 1])).abs();
                // The bound is the exact interpolant's. The rounding of the lerps and of g moves
                // g by nanometres, which is all a fixed point's g does.
                assert!(
                    change <= bound * step + ROUNDING_M,
                    "{}: g changes by {change} m in {step} s, over the bound {bound} m/s",
                    case.name
                );
                if bound > 0.0 {
                    worst = worst.max(change / step / bound);
                }
            }
        }
    }
    println!("the largest difference quotient of g is {worst:.4} of the hull bound");
}

/// A change of `g` the rounding of its evaluation can make, m: a thousand times the nanometres.
const ROUNDING_M: f64 = 1e-6;

/// (Rule 16) The S2 check fails when the scan misses a window: the same scan under a bound a
/// thousand times too small steps over passes, and the dense scan finds them.
#[test]
fn a_rate_bound_too_small_misses_windows_the_dense_scan_finds() {
    let case = cases()
        .into_iter()
        .find(|c| c.name == "two-body-sso-700km")
        .unwrap();
    let eph = table(&case.samples).unwrap();
    let mask_rad = case.mask_deg.to_radians();
    let mask = elevation_mask(mask_rad).unwrap();
    let station = station(&case.stations[0]).unwrap();
    let (first, last) = eph.span();
    let mut planted = Vec::new();
    spacelab_coverage::windows(
        |t| station.g(&mask, eph.position(t).unwrap()),
        tabulated_rate_bound(&eph, &mask) / 1000.0,
        first,
        last,
        case.tol,
        |w| planted.push(w),
    );
    let uncovered = missed(
        &dense(&eph, DENSE_STEP),
        &station,
        mask_rad,
        &planted,
        case.tol,
    );
    println!(
        "a bound 1000x too small: {} windows, {uncovered} samples above the mask outside them",
        planted.len()
    );
    assert!(
        uncovered > 0,
        "the dense scan did not catch a scan that steps over passes"
    );
}
