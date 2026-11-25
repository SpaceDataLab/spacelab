//! The bridges' reference for access windows over zones: the Rust API's windows on a set of SGP4
//! scans and tables, to the bit (Ф31.C).
//!
//! The C ABI (`crates/spacelab-capi/tests/zone.rs`) and the Python bridge
//! (`crates/spacelab-python/tests/bindings.py`) read `fixtures/bridge_access_zone_reference.tsv`,
//! which carries every scan, sample, zone vertex and window as the bits of its doubles. Each bridge
//! must return every window in it with the same bits, and refuse every case it refuses with its
//! own typed error. This file makes the reference the Rust API's: it rebuilds it from
//! `spacelab::access` and fails on any byte that differs.
//!
//! It also holds the API to its mathematics. Every edge is a sign change of the zone's `g` (G2). A
//! straight pass over a square at a 90° mask is in access exactly while its nadir is between the
//! square's meridian edges, in closed form (G2). The sign of `g` is the answer of a brute-force
//! search over the boundary, and the interior never rises above the boundary unless the nadir is
//! in the zone, at every sampled instant where that answer is clear (G3: the property
//! `derivations/access_zone_elevation.mac` proves). A dense scan at a fixed step finds no window
//! the API missed (S2). A scan under a rate bound a thousand times too small misses windows the
//! dense scan finds, so the S2 check fails when it should (rule 16).
//!
//! ```text
//! SPACELAB_WRITE_BRIDGE_ZONE_ACCESS=1 cargo test -p spacelab --test bridge_access_zone
//! ```

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use spacelab::EopSource;
use spacelab::access::{
    AccessError, Tabulated, Window, ZoneAccess, ZoneConstraint, sgp4_itrf_position,
    zone_sgp4_windows, zone_tabulated_windows,
};
use spacelab::zone::Zone;
use spacelab_core::{Angle, UtcDateTime, Vec3};
use spacelab_earth::{
    EQUATORIAL_RADIUS, FLATTENING, GeodesicLine, Geodetic, SurfacePoint, geodesic_inverse,
    geodetic_to_itrf,
};
use spacelab_eop::EopSeries;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sgp4::Satellite;

const REFERENCE: &str = "tests/fixtures/bridge_access_zone_reference.tsv";
const WRITE: &str = "SPACELAB_WRITE_BRIDGE_ZONE_ACCESS";

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// A label as the bridges take it: `(year, month, day, hour, minute, second)`.
type Label = (i32, u8, u8, u8, u8, f64);

/// One sample: time (s), ITRF position (m) and velocity (m/s).
type Sample = (f64, [f64; 3], [f64; 3]);

const ISS_2008: (&str, &str) = (
    "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537",
);
const EOP_2008: &str = "crates/spacelab/tests/fixtures/finals2000A_2008264.txt";

/// Where a scan's satellite comes from.
enum Source {
    /// An SGP4 element set, an EOP fixture path from the repository root or none, the start
    /// label and the span, s.
    Sgp4 {
        tle: (&'static str, &'static str),
        eop: Option<&'static str>,
        start: Label,
        span: f64,
    },
    /// A table of samples.
    Table(Vec<Sample>),
}

/// One scan of the reference: a source, the access kind, the mask and the zones, each a list of
/// vertices in degrees.
struct Case {
    name: &'static str,
    source: Source,
    access: ZoneAccess,
    mask_rad: f64,
    tol: f64,
    zones: Vec<Vec<(f64, f64)>>,
}

/// The square of half-side `half` degrees about `(lat, lon)`, counter-clockwise seen from outside.
fn square(lat: f64, lon: f64, half: f64) -> Vec<(f64, f64)> {
    vec![
        (lat - half, lon - half),
        (lat - half, lon + half),
        (lat + half, lon + half),
        (lat + half, lon - half),
    ]
}

/// The cap north of four geodesics through latitude `lat` (south of it for a negative `lat`),
/// which holds the pole.
fn cap(lat: f64) -> Vec<(f64, f64)> {
    let lons = if lat > 0.0 {
        [0.0, 90.0, 180.0, -90.0]
    } else {
        [0.0, -90.0, 180.0, 90.0]
    };
    lons.iter().map(|&lon| (lat, lon)).collect()
}

/// The complement of a zone: its vertices reversed.
fn reversed(mut zone: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    zone.reverse();
    zone
}

/// A circular orbit of radius `radius`, inclination `incl` and node `node` (radians), sampled every
/// `step` seconds `count` times in the frame turning with the Earth, as in
/// `bridge_access_tabulated.rs`.
fn two_body(radius: f64, incl: f64, node: f64, step: f64, count: usize) -> Vec<Sample> {
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
            (dt, re, ve)
        })
        .collect()
}

/// A point fixed `height` metres above `(lat, lon)` in degrees, sampled at the start and the end
/// of ten minutes.
fn fixed(lat: f64, lon: f64, height: f64) -> Vec<Sample> {
    let p = geodetic_to_itrf(Geodetic {
        latitude: Angle::from_degrees(lat),
        longitude: Angle::from_degrees(lon),
        height: spacelab_core::Length::from_metres(height),
    })
    .vector;
    let r = [p.x, p.y, p.z];
    vec![(0.0, r, [0.0; 3]), (600.0, r, [0.0; 3])]
}

/// The straight pass: uniform motion 500 km over the equator at 0° E, 7 km/s eastward, which the
/// interpolant holds exactly.
fn straight() -> Vec<Sample> {
    [0.0, 60.0, 120.0]
        .map(|t| (t, [6_878_137.0, -420e3 + 7e3 * t, 0.0], [0.0, 7e3, 0.0]))
        .to_vec()
}

const CYPRUS: (f64, f64) = (35.0, 33.0);

#[expect(clippy::too_many_lines, reason = "the case list, one literal per case")]
fn cases() -> Vec<Case> {
    let sso = two_body(
        6_378_137.0 + 700e3,
        98.19_f64.to_radians(),
        30.0_f64.to_radians(),
        120.0,
        181,
    );
    let iss = |eop, span| Source::Sgp4 {
        tle: ISS_2008,
        eop,
        start: (2008, 9, 20, 12, 0, 0.0),
        span,
    };
    let case = |name, source, access, mask_deg: f64, zones| Case {
        name,
        source,
        access,
        mask_rad: mask_deg.to_radians(),
        tol: 1e-4,
        zones,
    };
    let bow_tie = vec![(0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0)];
    let mut out = vec![
        case(
            "iss-2008-any-10",
            iss(None, 43_200.0),
            ZoneAccess::Any,
            10.0,
            vec![
                square(CYPRUS.0, CYPRUS.1, 3.0),
                square(-17.0, 180.0, 4.0),
                reversed(square(-30.0, -60.0, 10.0)),
            ],
        ),
        case(
            "iss-2008-eop-all-0",
            iss(Some(EOP_2008), 43_200.0),
            ZoneAccess::All,
            0.0,
            vec![square(CYPRUS.0, CYPRUS.1, 1.0), square(-17.0, 180.0, 2.0)],
        ),
        case(
            "two-body-sso-any-90",
            Source::Table(sso.clone()),
            ZoneAccess::Any,
            90.0,
            vec![
                cap(75.0),
                cap(-75.0),
                square(CYPRUS.0, CYPRUS.1, 3.0),
                square(-17.0, 180.0, 4.0),
            ],
        ),
        case(
            "two-body-sso-all-minus-10",
            Source::Table(sso.clone()),
            ZoneAccess::All,
            -10.0,
            vec![cap(75.0), reversed(square(-30.0, -60.0, 10.0))],
        ),
        case(
            "straight-pass-square-90",
            Source::Table(straight()),
            ZoneAccess::Any,
            90.0,
            vec![square(0.0, 0.0, 1.0)],
        ),
        case(
            "fixed-point-20km-up",
            Source::Table(fixed(CYPRUS.0, CYPRUS.1, 20e3)),
            ZoneAccess::Any,
            10.0,
            vec![square(CYPRUS.0, CYPRUS.1, 3.0), square(-35.0, -147.0, 3.0)],
        ),
        case(
            "refused-zone",
            Source::Table(sso[..3].to_vec()),
            ZoneAccess::Any,
            10.0,
            vec![square(CYPRUS.0, CYPRUS.1, 3.0), bow_tie],
        ),
        case(
            "refused-mask",
            Source::Table(sso[..3].to_vec()),
            ZoneAccess::Any,
            90.0,
            vec![square(CYPRUS.0, CYPRUS.1, 3.0)],
        ),
        case(
            "refused-samples",
            Source::Table(sso[..1].to_vec()),
            ZoneAccess::Any,
            10.0,
            vec![square(CYPRUS.0, CYPRUS.1, 3.0)],
        ),
        case(
            "refused-below-surface",
            Source::Table(fixed(CYPRUS.0, CYPRUS.1, -10e3)),
            ZoneAccess::All,
            10.0,
            vec![square(CYPRUS.0, CYPRUS.1, 3.0)],
        ),
        case(
            "refused-interval",
            iss(None, 0.0),
            ZoneAccess::Any,
            10.0,
            vec![square(CYPRUS.0, CYPRUS.1, 3.0)],
        ),
    ];
    // Just above 90°: the first mask a zone refuses.
    out[7].mask_rad = core::f64::consts::FRAC_PI_2.next_up();
    let mut tolerance = case(
        "refused-tolerance",
        iss(None, 3_600.0),
        ZoneAccess::All,
        10.0,
        vec![square(CYPRUS.0, CYPRUS.1, 3.0)],
    );
    tolerance.tol = 0.0;
    out.push(tolerance);
    out
}

/// The refusal kinds, one name per typed error a zone scan can return, the zone's own first.
fn kind(e: &Refusal) -> &'static str {
    match e {
        Refusal::Zone => "zone",
        Refusal::Access(AccessError::Mask) => "mask",
        Refusal::Access(AccessError::Interval) => "interval",
        Refusal::Access(AccessError::Tolerance) => "tolerance",
        Refusal::Access(AccessError::Samples) => "samples",
        Refusal::Access(AccessError::BelowSurface { .. }) => "below-surface",
        Refusal::Access(other) => panic!("no case refuses with {other:?}"),
    }
}

/// Why a case was refused: a zone, or the scan.
#[derive(Debug)]
enum Refusal {
    Zone,
    Access(AccessError),
}

fn zone(vertices: &[(f64, f64)]) -> Result<Zone, spacelab::zone::ZoneError> {
    let points: Vec<SurfacePoint> = vertices
        .iter()
        .map(|&(lat, lon)| SurfacePoint {
            latitude: Angle::from_radians(lat.to_radians()),
            longitude: Angle::from_radians(lon.to_radians()),
        })
        .collect();
    Zone::new(&points)
}

fn table(samples: &[Sample]) -> Result<Tabulated, AccessError> {
    let v = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
    Tabulated::new(
        samples.iter().map(|s| s.0).collect(),
        samples.iter().map(|s| v(s.1)).collect(),
        samples.iter().map(|s| v(s.2)).collect(),
    )
}

fn eop(path: Option<&str>) -> Option<EopSeries> {
    path.map(|p| {
        EopSeries::parse_finals2000a(&std::fs::read_to_string(root().join(p)).unwrap()).unwrap()
    })
}

fn start(l: Label) -> UtcDateTime {
    UtcDateTime::new(l.0, l.1, l.2, l.3, l.4, l.5).unwrap()
}

/// A case's zones, in order; the first a zone refuses refuses them all.
fn zones(case: &Case) -> Result<Vec<Zone>, Refusal> {
    case.zones
        .iter()
        .map(|v| zone(v))
        .collect::<Result<_, _>>()
        .map_err(|_| Refusal::Zone)
}

/// A case's scan of every zone, in zone order, as the bridges batch it: the table first, then
/// every zone, then for each zone its constraint and its scan; the first refusal refuses the
/// batch.
/// A case's windows, zone by zone, or what refused it.
type Answer = Result<Vec<Vec<Window>>, Refusal>;

fn run(case: &Case) -> Answer {
    let mut out = Vec::new();
    match &case.source {
        Source::Sgp4 {
            tle,
            eop: path,
            start: label,
            span,
        } => {
            let satellite = Satellite::from_tle(tle.0, tle.1).unwrap();
            let series = eop(*path);
            let source = series.as_ref().map_or(EopSource::NoEop, EopSource::Series);
            for zone in &zones(case)? {
                let constraint = ZoneConstraint::new(zone, case.access, case.mask_rad)
                    .map_err(Refusal::Access)?;
                out.push(
                    zone_sgp4_windows(
                        &satellite,
                        source,
                        start(*label),
                        *span,
                        &constraint,
                        case.tol,
                    )
                    .map_err(Refusal::Access)?,
                );
            }
        }
        Source::Table(samples) => {
            let eph = table(samples).map_err(Refusal::Access)?;
            for zone in &zones(case)? {
                let constraint = ZoneConstraint::new(zone, case.access, case.mask_rad)
                    .map_err(Refusal::Access)?;
                out.push(
                    zone_tabulated_windows(&eph, &constraint, case.tol).map_err(Refusal::Access)?,
                );
            }
        }
    }
    Ok(out)
}

fn access_name(access: ZoneAccess) -> &'static str {
    match access {
        ZoneAccess::Any => "any",
        ZoneAccess::All => "all",
    }
}

/// Every case's answer, computed once for all the tests of this file.
fn answers() -> &'static [(Case, Answer)] {
    static ANSWERS: OnceLock<Vec<(Case, Answer)>> = OnceLock::new();
    ANSWERS.get_or_init(|| {
        cases()
            .into_iter()
            .map(|c| {
                let r = run(&c);
                (c, r)
            })
            .collect()
    })
}

/// The whole file, from the cases and the Rust API.
fn reference() -> String {
    let mut out = String::new();
    out.push_str(
        "# The Rust API's access windows over zones (spacelab::access::zone_sgp4_windows and \
         zone_tabulated_windows), to the bit (Ф31.C).\n\
         # Written by crates/spacelab/tests/bridge_access_zone.rs; the C ABI and the Python bridge \
         must reproduce it. A double is the 16 hex digits of its bits.\n\
         # Angles are radians; positions are ITRF metres, velocities metres per second, and times \
         seconds past the start label (SGP4) or the table's own. A zone is its vertices, \
         counter-clockwise seen from outside. Rows, tab-separated:\n\
         #   sgp4    <case> <any|all> <mask rad> <tol s> <TLE line 1> <TLE line 2> \
         <EOP path from the repository root | none> <year> <month> <day> <hour> <minute> \
         <second> <span s>\n\
         #   table   <case> <any|all> <mask rad> <tol s> <sample count>\n\
         #   sample  <case> <index> <t s> <x> <y> <z> <vx> <vy> <vz>\n\
         #   zone    <case> <zone> <vertex count>\n\
         #   vertex  <case> <zone> <index> <latitude rad> <longitude rad>\n\
         #   window  <case> <zone> <rise s> <set s>\n\
         #   refused <case> <kind>\n",
    );
    for (case, answer) in answers() {
        let head = format!(
            "{}\t{}\t{}\t{}",
            case.name,
            access_name(case.access),
            bits(case.mask_rad),
            bits(case.tol)
        );
        match &case.source {
            Source::Sgp4 {
                tle,
                eop,
                start: l,
                span,
            } => writeln!(
                out,
                "sgp4\t{head}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                tle.0,
                tle.1,
                eop.unwrap_or("none"),
                l.0,
                l.1,
                l.2,
                l.3,
                l.4,
                bits(l.5),
                bits(*span)
            )
            .unwrap(),
            Source::Table(samples) => {
                writeln!(out, "table\t{head}\t{}", samples.len()).unwrap();
                for (i, (t, r, v)) in samples.iter().enumerate() {
                    let row: Vec<String> = [*t, r[0], r[1], r[2], v[0], v[1], v[2]]
                        .iter()
                        .map(|&x| bits(x))
                        .collect();
                    writeln!(out, "sample\t{}\t{i}\t{}", case.name, row.join("\t")).unwrap();
                }
            }
        }
        for (z, vertices) in case.zones.iter().enumerate() {
            writeln!(out, "zone\t{}\t{z}\t{}", case.name, vertices.len()).unwrap();
            for (i, &(lat, lon)) in vertices.iter().enumerate() {
                writeln!(
                    out,
                    "vertex\t{}\t{z}\t{i}\t{}\t{}",
                    case.name,
                    bits(lat.to_radians()),
                    bits(lon.to_radians())
                )
                .unwrap();
            }
        }
        match answer {
            Ok(per_zone) => {
                for (z, ws) in per_zone.iter().enumerate() {
                    for w in ws {
                        writeln!(
                            out,
                            "window\t{}\t{z}\t{}\t{}",
                            case.name,
                            bits(w.rise),
                            bits(w.set)
                        )
                        .unwrap();
                    }
                }
            }
            Err(e) => writeln!(out, "refused\t{}\t{}", case.name, kind(e)).unwrap(),
        }
    }
    out
}

/// The committed file is the Rust API's answer, byte for byte.
#[test]
fn the_zone_access_reference_is_the_rust_api_on_its_cases() {
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

/// Every refusal kind a zone scan has is in the reference, and both access kinds and both sources
/// are answered, so a bridge that mismaps one is caught.
#[test]
fn the_zone_access_reference_refuses_every_kind_and_answers_every_source() {
    let text = reference();
    let kinds: std::collections::BTreeSet<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("refused\t"))
        .map(|l| l.split('\t').nth(1).unwrap())
        .collect();
    assert_eq!(
        kinds.into_iter().collect::<Vec<_>>(),
        [
            "below-surface",
            "interval",
            "mask",
            "samples",
            "tolerance",
            "zone"
        ],
        "the reference does not refuse every kind"
    );
    let answered: std::collections::BTreeSet<(bool, &str)> = answers()
        .iter()
        .filter(|(_, a)| a.as_ref().is_ok_and(|w| w.iter().any(|z| !z.is_empty())))
        .map(|(c, _)| {
            (
                matches!(c.source, Source::Sgp4 { .. }),
                access_name(c.access),
            )
        })
        .collect();
    assert_eq!(
        answered.len(),
        4,
        "a source or an access kind has no window"
    );
}

/// A case's satellite at `t`, and its scan's span.
struct Track {
    eph: Option<Tabulated>,
    sgp4: Option<(Satellite, Option<EopSeries>, UtcDateTime)>,
    span: (f64, f64),
}

impl Track {
    fn of(case: &Case) -> Self {
        match &case.source {
            Source::Sgp4 {
                tle,
                eop: path,
                start: l,
                span,
            } => Self {
                eph: None,
                sgp4: Some((
                    Satellite::from_tle(tle.0, tle.1).unwrap(),
                    eop(*path),
                    start(*l),
                )),
                span: (0.0, *span),
            },
            Source::Table(samples) => {
                let eph = table(samples).unwrap();
                let span = eph.span();
                Self {
                    eph: Some(eph),
                    sgp4: None,
                    span,
                }
            }
        }
    }

    fn at(&self, t: f64) -> Vec3 {
        if let Some(eph) = &self.eph {
            return eph.position(t).unwrap();
        }
        let (sat, series, start) = self.sgp4.as_ref().unwrap();
        let source = series.as_ref().map_or(EopSource::NoEop, EopSource::Series);
        sgp4_itrf_position(sat, source, *start, t).unwrap()
    }
}

/// The answered cases, with their tracks.
fn answered() -> Vec<(&'static Case, Track, &'static Vec<Vec<Window>>)> {
    answers()
        .iter()
        .filter_map(|(c, a)| {
            let windows = a.as_ref().ok()?;
            Some((c, Track::of(c), windows))
        })
        .collect()
}

/// The dense scan's step, s.
const DENSE_STEP: f64 = 5.0;

/// The times of a dense scan across `span`, its end included.
fn dense_times((first, last): (f64, f64)) -> Vec<f64> {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a positive span over a positive step, far inside usize"
    )]
    let count = ((last - first) / DENSE_STEP) as usize;
    #[expect(
        clippy::cast_precision_loss,
        reason = "a sample index below 2^53 is exact in f64"
    )]
    let mut times: Vec<f64> = (0..=count).map(|k| first + k as f64 * DENSE_STEP).collect();
    if *times.last().unwrap() < last {
        times.push(last);
    }
    times
}

/// The dense samples in access that no window holds, to within `tol`.
fn missed(times: &[f64], inside: &[bool], windows: &[Window], tol: f64) -> usize {
    times
        .iter()
        .zip(inside)
        .filter(|&(&t, &held)| {
            held && !windows
                .iter()
                .any(|w| w.rise - tol <= t && t <= w.set + tol)
        })
        .count()
}

/// (S2) A fixed-step scan of every answered case finds the zone in access only inside a window the
/// API reported, and every window longer than two steps holds a sample in access.
#[test]
fn a_dense_scan_finds_no_window_the_api_missed() {
    let (mut samples, mut windows, mut shortest) = (0_usize, 0_usize, f64::INFINITY);
    for (case, track, per_zone) in answered() {
        let times = dense_times(track.span);
        let positions: Vec<Vec3> = times.iter().map(|&t| track.at(t)).collect();
        for (vertices, ws) in case.zones.iter().zip(per_zone) {
            let zone = zone(vertices).unwrap();
            let constraint = ZoneConstraint::new(&zone, case.access, case.mask_rad).unwrap();
            let inside: Vec<bool> = positions
                .iter()
                .map(|&r| constraint.g(r).unwrap() >= 0.0)
                .collect();
            let lost = missed(&times, &inside, ws, case.tol);
            assert_eq!(
                lost, 0,
                "{}: {lost} samples in access outside every window",
                case.name
            );
            for w in ws {
                shortest = shortest.min(w.duration());
                if w.duration() > 2.0 * DENSE_STEP {
                    assert!(
                        times
                            .iter()
                            .zip(&inside)
                            .any(|(&t, &held)| held && w.rise < t && t < w.set),
                        "{}: no sample in access in {w:?}",
                        case.name
                    );
                }
            }
            samples += times.len();
            windows += ws.len();
        }
    }
    println!(
        "{windows} windows, {samples} samples at a {DENSE_STEP} s step; the shortest window is \
         {shortest:.3} s"
    );
    assert!(windows > 20, "the reference lost its windows: {windows}");
}

/// (G2) Every edge the API reports is a sign change of the zone's `g` within twice the tolerance,
/// rising at a rise and falling at a set, unless the edge is the span's own end.
#[test]
fn every_edge_is_a_sign_change_of_the_zone_g() {
    let mut edges = 0;
    for (case, track, per_zone) in answered() {
        let delta = 2.0 * case.tol;
        for (vertices, ws) in case.zones.iter().zip(per_zone) {
            let zone = zone(vertices).unwrap();
            let constraint = ZoneConstraint::new(&zone, case.access, case.mask_rad).unwrap();
            let g = |t: f64| constraint.g(track.at(t)).unwrap();
            for w in ws {
                if w.rise > track.span.0 {
                    assert!(
                        g(w.rise - delta) < 0.0 && g(w.rise + delta) >= 0.0,
                        "{}: {w:?}",
                        case.name
                    );
                    edges += 1;
                }
                if w.set < track.span.1 {
                    assert!(
                        g(w.set - delta) >= 0.0 && g(w.set + delta) < 0.0,
                        "{}: {w:?}",
                        case.name
                    );
                    edges += 1;
                }
            }
        }
    }
    println!("{edges} edges, each a sign change of g within twice its tolerance");
    assert!(edges > 40, "too few edges to mean anything: {edges}");
}

/// (G2) The straight pass over the 2° square about (0°, 0°) at a 90° mask is in access exactly
/// while its nadir, on the equator at longitude `atan2(y, x)`, lies between the meridian edges at
/// ±1°: from `(420 km − x tan 1°)/7 km/s` to `(420 km + x tan 1°)/7 km/s`.
#[test]
fn a_straight_pass_is_over_the_square_where_its_nadir_crosses_the_meridian_edges() {
    let (_, answer) = answers()
        .iter()
        .find(|(c, _)| c.name == "straight-pass-square-90")
        .unwrap();
    let windows = &answer.as_ref().unwrap()[0];
    let reach = 6_878_137.0 * libm::tan(1.0_f64.to_radians());
    let (rise, set) = ((420e3 - reach) / 7e3, (420e3 + reach) / 7e3);
    assert_eq!(windows.len(), 1, "{windows:?}");
    let w = windows[0];
    println!(
        "rise {:.3e} s and set {:.3e} s from the closed form",
        w.rise - rise,
        w.set - set
    );
    assert!(
        (w.rise - rise).abs() <= 2e-4 && (w.set - set).abs() <= 2e-4,
        "{w:?} against [{rise}, {set}]"
    );
}

/// Points along each edge of a zone, about every `step` metres, its vertices included.
fn boundary(vertices: &[(f64, f64)], step: f64) -> Vec<(f64, f64)> {
    let point = |(lat, lon): (f64, f64)| SurfacePoint {
        latitude: Angle::from_degrees(lat),
        longitude: Angle::from_degrees(lon),
    };
    let mut out = Vec::new();
    for i in 0..vertices.len() {
        let (a, b) = (vertices[i], vertices[(i + 1) % vertices.len()]);
        let inverse = geodesic_inverse(point(a), point(b)).unwrap();
        let line = GeodesicLine::new(point(a), inverse.azimuth1).unwrap();
        let length = inverse.distance.metres();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "an edge over a step, far inside usize"
        )]
        let count = (length / step).ceil() as usize;
        for k in 0..count {
            #[expect(clippy::cast_precision_loss, reason = "a small index")]
            let s = length * k as f64 / count as f64;
            let p = line
                .position(spacelab_core::Length::from_metres(s))
                .unwrap();
            out.push((p.point.latitude.radians(), p.point.longitude.radians()));
        }
    }
    out
}

/// The elevation of `sat` seen from the ellipsoid's point `(lat, lon)`, radians.
fn elevation(sat: Vec3, lat: f64, lon: f64) -> f64 {
    let p = geodetic_to_itrf(Geodetic {
        latitude: Angle::from_radians(lat),
        longitude: Angle::from_radians(lon),
        height: spacelab_core::Length::from_metres(0.0),
    })
    .vector;
    let (s, c) = libm::sincos(lat);
    let (sl, cl) = libm::sincos(lon);
    spacelab_sensors::elevation::elevation(sat, p, Vec3::new(c * cl, c * sl, s))
}

/// (G3) At every sampled instant of every answered case, the sign of `g` is the brute-force answer
/// wherever it is clear: the boundary sampled every kilometre, the nadir and the far foot tested by
/// `Zone::contains`, and a margin of a hundredth of a degree. And a grid over the interior never
/// rises above the boundary unless the nadir is in the zone, nor sinks below it unless the far foot
/// is: the property the scan rests on.
#[test]
fn the_sign_of_g_is_the_brute_force_answer_and_the_interior_never_beats_the_boundary() {
    let margin = 0.01_f64.to_radians();
    let (mut compared, mut interior) = (0_usize, 0_usize);
    for (case, track, _) in answered() {
        for vertices in &case.zones {
            let zone = zone(vertices).unwrap();
            let constraint = ZoneConstraint::new(&zone, case.access, case.mask_rad).unwrap();
            let edge = boundary(vertices, 1e3);
            let grid: Vec<(f64, f64)> = (0..90)
                .flat_map(|i| (0..180).map(move |j| (i, j)))
                .map(|(i, j)| {
                    (
                        (-89.0 + 2.0 * f64::from(i)).to_radians(),
                        (-179.0 + 2.0 * f64::from(j)).to_radians(),
                    )
                })
                .filter(|&(lat, lon)| {
                    zone.contains(SurfacePoint {
                        latitude: Angle::from_radians(lat),
                        longitude: Angle::from_radians(lon),
                    })
                    .unwrap()
                })
                .collect();
            let (first, last) = track.span;
            let step = ((last - first) / 40.0).max(1.0);
            let mut t = first;
            while t <= last {
                let sat = track.at(t);
                let nadir =
                    spacelab_earth::itrf_to_geodetic(spacelab_frames::Position::new(sat)).unwrap();
                let contains = |lat: f64, lon: f64| {
                    zone.contains(SurfacePoint {
                        latitude: Angle::from_radians(lat),
                        longitude: Angle::from_radians(lon),
                    })
                    .unwrap()
                };
                let nadir_in = contains(nadir.latitude.radians(), nadir.longitude.radians());
                let far = far_foot(sat);
                let far_in = contains(far.0, far.1);
                let on_edge: Vec<f64> = edge.iter().map(|&(a, b)| elevation(sat, a, b)).collect();
                let (hi, lo) = on_edge
                    .iter()
                    .fold((f64::NEG_INFINITY, f64::INFINITY), |(h, l), &e| {
                        (h.max(e), l.min(e))
                    });
                for &(a, b) in &grid {
                    let e = elevation(sat, a, b);
                    assert!(nadir_in || e <= hi + 1e-9, "{}: {e} over {hi}", case.name);
                    assert!(far_in || e >= lo - 1e-9, "{}: {e} under {lo}", case.name);
                    interior += 1;
                }
                // A nadir in the zone is in access at any mask, a far foot in it out of access at
                // any; otherwise the boundary decides, where it clears the mask by the margin.
                let extreme = match case.access {
                    ZoneAccess::Any => hi,
                    ZoneAccess::All => lo,
                };
                let clear = match case.access {
                    ZoneAccess::Any if nadir_in => Some(true),
                    ZoneAccess::All if far_in => Some(false),
                    ZoneAccess::Any | ZoneAccess::All if extreme >= case.mask_rad + margin => {
                        Some(true)
                    }
                    ZoneAccess::Any | ZoneAccess::All if extreme <= case.mask_rad - margin => {
                        Some(false)
                    }
                    ZoneAccess::Any | ZoneAccess::All => None,
                };
                let g = constraint.g(sat).unwrap();
                if let Some(held) = clear {
                    assert_eq!(
                        g >= 0.0,
                        held,
                        "{} at {t}: g {g}, the boundary's extreme {extreme}, the nadir in: \
                         {nadir_in}, the far foot in: {far_in}",
                        case.name
                    );
                    compared += 1;
                }
                t += step;
            }
        }
    }
    println!("{compared} signs against the brute force, {interior} interior points");
    assert!(compared > 300, "too few comparisons: {compared}");
}

/// The far foot of the normal through `sat`, by bisection on the latitude of the opposite meridian
/// where `S − p` turns along the normal: claim 19 of the derivation, `F(φ) = r sin φ + z cos φ +
/// N e² sin φ cos φ = 0`, apart from the API's route.
fn far_foot(sat: Vec3) -> (f64, f64) {
    let e2 = FLATTENING * (2.0 - FLATTENING);
    let (axis, height) = (libm::hypot(sat.x, sat.y), sat.z);
    let turn = |phi: f64| {
        let (sin, cos) = libm::sincos(phi);
        let normal = EQUATORIAL_RADIUS / libm::sqrt(1.0 - e2 * sin * sin);
        axis * sin + height * cos + normal * e2 * sin * cos
    };
    let (mut south, mut north) = (-core::f64::consts::FRAC_PI_2, core::f64::consts::FRAC_PI_2);
    for _ in 0..200 {
        let mid = 0.5 * (south + north);
        if (turn(mid) < 0.0) == (turn(south) < 0.0) {
            south = mid;
        } else {
            north = mid;
        }
    }
    (
        0.5 * (south + north),
        libm::atan2(sat.y, sat.x) + core::f64::consts::PI,
    )
}

/// (Rule 16) The S2 check fails when the scan misses a window: the same scan under a rate bound a
/// thousand times too small steps over passes, and the dense scan finds them.
#[test]
fn a_rate_bound_too_small_misses_windows_the_dense_scan_finds() {
    let (case, track, _) = answered()
        .into_iter()
        .find(|(c, _, _)| c.name == "two-body-sso-any-90")
        .unwrap();
    let zone = zone(&case.zones[0]).unwrap();
    let constraint = ZoneConstraint::new(&zone, case.access, case.mask_rad).unwrap();
    let eph = track.eph.as_ref().unwrap();
    let bound = constraint.rate_factor()
        * spacelab::access::tabulated_rate_bound(eph, &spacelab_sensors::ElevationMask::HORIZON);
    let mut planted = Vec::new();
    spacelab_coverage::windows(
        |t| constraint.g(eph.position(t).unwrap()).unwrap(),
        bound / 1000.0,
        track.span.0,
        track.span.1,
        case.tol,
        |w| planted.push(w),
    );
    let times = dense_times(track.span);
    let inside: Vec<bool> = times
        .iter()
        .map(|&t| constraint.g(track.at(t)).unwrap() >= 0.0)
        .collect();
    let uncovered = missed(&times, &inside, &planted, case.tol);
    println!(
        "a bound 1000x too small: {} windows, {uncovered} samples in access outside them",
        planted.len()
    );
    assert!(
        uncovered > 0,
        "the dense scan did not catch a scan that steps over passes"
    );
}
