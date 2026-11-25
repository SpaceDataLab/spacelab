//! The C ABI's point-in-zone answers and zone access windows are the Rust API's (Ф31.C).
//!
//! `crates/spacelab/tests/fixtures/bridge_zone_contains_reference.tsv` is
//! `spacelab::zone::Zone::contains` on a set of zones and points, and `bridge_zone_contains.rs` in
//! `crates/spacelab/tests` holds it to the API byte for byte and the API to the cases its model
//! names. Here every case goes through the exported `sl_zone_contains`, all points in one batch and
//! then one point at a time, and every answer must come back the same, every refusal with
//! `SL_ERR_INVALID_ARGUMENT` and nothing written. Every row of the file is counted.
//!
//! `bridge_access_zone_reference.tsv` beside it is `spacelab::access::zone_sgp4_windows` and
//! `zone_tabulated_windows` on a set of scans, which `bridge_access_zone.rs` holds to the API the
//! same way. Every scan goes through `sl_access_zone_sgp4` or `sl_access_zone_tabulated`, all
//! zones in one batch and then one zone at a time, and every window must come back with the same
//! bits (S2), every refusal with the status the header names.

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use core::ffi::c_char;
use std::ffi::CString;
use std::path::Path;

use spacelab_capi::{
    SL_ERR_BUFFER_TOO_SMALL, SL_ERR_DOMAIN, SL_ERR_INVALID_ARGUMENT, SL_OK, SlEopSeries, SlSgp4,
    SlSurfacePoint, SlUtcDateTime, SlVec3, SlZoneWindow, sl_access_zone_sgp4,
    sl_access_zone_tabulated, sl_eop_series_close, sl_eop_series_parse_finals2000a, sl_sgp4_close,
    sl_sgp4_open, sl_zone_contains,
};

/// `SL_ZONE_ANY` and `SL_ZONE_ALL` as `include/spacelab.h` declares them; the C smoke test passes
/// the header's own names and must get the same windows.
const SL_ZONE_ANY: i32 = 0;
const SL_ZONE_ALL: i32 = 1;

fn f64_of(hex: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(hex, 16).expect("16 hex digits"))
}

/// One case of the reference: its vertices, its points, and the API's answers or refusal.
struct Case {
    name: String,
    vertices: Vec<SlSurfacePoint>,
    points: Vec<SlSurfacePoint>,
    answers: Vec<Option<u8>>,
    refused: Option<String>,
}

fn cases() -> (Vec<Case>, usize) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../spacelab/tests/fixtures/bridge_zone_contains_reference.tsv");
    let text = std::fs::read_to_string(path).expect("the zone reference");
    let (mut cases, mut rows): (Vec<Case>, usize) = (Vec::new(), 0);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        rows += 1;
        let f: Vec<&str> = line.split('\t').collect();
        let point = |i: usize| SlSurfacePoint {
            latitude_rad: f64_of(f[i]),
            longitude_rad: f64_of(f[i + 1]),
        };
        if f[0] == "zone" {
            cases.push(Case {
                name: f[1].to_owned(),
                vertices: Vec::new(),
                points: Vec::new(),
                answers: Vec::new(),
                refused: None,
            });
            continue;
        }
        let case = cases.last_mut().expect("a row after its zone");
        assert_eq!(f[1], case.name, "a row outside its case: {line}");
        match f[0] {
            "vertex" => case.vertices.push(point(3)),
            "point" => {
                case.points.push(point(3));
                case.answers.push(match f[5] {
                    "1" => Some(1),
                    "0" => Some(0),
                    "-" => None,
                    other => panic!("an unknown answer {other}"),
                });
            }
            "refused" => case.refused = Some(f[2].to_owned()),
            other => panic!("an unknown row kind {other}"),
        }
    }
    (cases, rows)
}

/// One call of `sl_zone_contains`: the status and the bytes, which start as `0xAA` so that a
/// refusal that writes anything shows.
fn contains(vertices: &[SlSurfacePoint], points: &[SlSurfacePoint]) -> (i32, Vec<u8>) {
    let mut inside = vec![0xAA_u8; points.len()];
    // SAFETY: each slice is readable for its length and `inside` writable for `points.len()`.
    let status = unsafe {
        sl_zone_contains(
            vertices.as_ptr(),
            vertices.len(),
            points.as_ptr(),
            points.len(),
            inside.as_mut_ptr(),
        )
    };
    (status, inside)
}

#[test]
fn the_c_abi_returns_the_zone_reference() {
    let (cases, rows) = cases();
    let (mut counted, mut answered, mut refusals) = (0, 0, 0);
    for case in &cases {
        counted +=
            1 + case.vertices.len() + case.points.len() + usize::from(case.refused.is_some());
        let (status, inside) = contains(&case.vertices, &case.points);
        if let Some(kind) = &case.refused {
            assert_eq!(status, SL_ERR_INVALID_ARGUMENT, "{}: {kind}", case.name);
            assert!(
                inside.iter().all(|&b| b == 0xAA),
                "{}: a refusal wrote",
                case.name
            );
            refusals += 1;
            continue;
        }
        assert_eq!(status, SL_OK, "{}", case.name);
        let want: Vec<u8> = case.answers.iter().map(|a| a.unwrap()).collect();
        assert_eq!(inside, want, "{}", case.name);
        for (i, p) in case.points.iter().enumerate() {
            let (status, alone) = contains(&case.vertices, core::slice::from_ref(p));
            assert_eq!(
                (status, alone[0]),
                (SL_OK, want[i]),
                "{}: point {i} alone",
                case.name
            );
        }
        answered += want.len();
    }
    assert_eq!(counted, rows, "a row of the reference was not read");
    assert!(
        answered > 1500 && refusals == 7,
        "the reference lost its cases"
    );
}

#[test]
fn the_zone_entry_point_refuses_null_pointers_and_no_points() {
    let triangle = [
        SlSurfacePoint {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
        },
        SlSurfacePoint {
            latitude_rad: 0.0,
            longitude_rad: 0.01,
        },
        SlSurfacePoint {
            latitude_rad: 0.01,
            longitude_rad: 0.01,
        },
    ];
    let point = [SlSurfacePoint {
        latitude_rad: 0.004,
        longitude_rad: 0.006,
    }];
    let mut inside = [0xAA_u8];
    // SAFETY: every non-null pointer is valid for the count passed with it.
    unsafe {
        assert_eq!(
            sl_zone_contains(core::ptr::null(), 3, point.as_ptr(), 1, inside.as_mut_ptr()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_zone_contains(
                triangle.as_ptr(),
                3,
                core::ptr::null(),
                1,
                inside.as_mut_ptr()
            ),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_zone_contains(
                triangle.as_ptr(),
                3,
                point.as_ptr(),
                1,
                core::ptr::null_mut()
            ),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_zone_contains(triangle.as_ptr(), 3, point.as_ptr(), 0, inside.as_mut_ptr()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(inside[0], 0xAA);
        assert_eq!(
            sl_zone_contains(triangle.as_ptr(), 3, point.as_ptr(), 1, inside.as_mut_ptr()),
            SL_OK
        );
    }
    assert_eq!(inside[0], 1);
}

/// One scan of the zone access reference, with the answer the Rust API gave.
struct Scan {
    name: String,
    access: i32,
    mask: f64,
    tol: f64,
    /// TLE lines, EOP path, start label and span, for an SGP4 scan.
    sgp4: Option<(String, String, Option<String>, SlUtcDateTime, f64)>,
    times: Vec<f64>,
    positions: Vec<SlVec3>,
    velocities: Vec<SlVec3>,
    zones: Vec<Vec<SlSurfacePoint>>,
    windows: Vec<(usize, u64, u64)>,
    refused: Option<String>,
}

fn scans() -> (Vec<Scan>, usize) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../spacelab/tests/fixtures/bridge_access_zone_reference.tsv");
    let text = std::fs::read_to_string(path).expect("the zone access reference");
    let (mut out, mut rows): (Vec<Scan>, usize) = (Vec::new(), 0);
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        rows += 1;
        let f: Vec<&str> = line.split('\t').collect();
        let vec3 = |i: usize| SlVec3 {
            x: f64_of(f[i]),
            y: f64_of(f[i + 1]),
            z: f64_of(f[i + 2]),
        };
        if f[0] == "sgp4" || f[0] == "table" {
            out.push(Scan {
                name: f[1].to_owned(),
                access: match f[2] {
                    "any" => SL_ZONE_ANY,
                    "all" => SL_ZONE_ALL,
                    other => panic!("an unknown access kind {other}"),
                },
                mask: f64_of(f[3]),
                tol: f64_of(f[4]),
                sgp4: (f[0] == "sgp4").then(|| {
                    (
                        f[5].to_owned(),
                        f[6].to_owned(),
                        (f[7] != "none").then(|| f[7].to_owned()),
                        SlUtcDateTime {
                            year: f[8].parse().unwrap(),
                            month: f[9].parse().unwrap(),
                            day: f[10].parse().unwrap(),
                            hour: f[11].parse().unwrap(),
                            minute: f[12].parse().unwrap(),
                            second: f64_of(f[13]),
                        },
                        f64_of(f[14]),
                    )
                }),
                times: Vec::new(),
                positions: Vec::new(),
                velocities: Vec::new(),
                zones: Vec::new(),
                windows: Vec::new(),
                refused: None,
            });
            continue;
        }
        let scan = out.last_mut().expect("a scan row opens each case");
        assert_eq!(f[1], scan.name, "a row outside its scan: {line}");
        match f[0] {
            "sample" => {
                scan.times.push(f64_of(f[3]));
                scan.positions.push(vec3(4));
                scan.velocities.push(vec3(7));
            }
            "zone" => scan.zones.push(Vec::new()),
            "vertex" => scan.zones[f[2].parse::<usize>().unwrap()].push(SlSurfacePoint {
                latitude_rad: f64_of(f[4]),
                longitude_rad: f64_of(f[5]),
            }),
            "window" => scan.windows.push((
                f[2].parse().unwrap(),
                u64::from_str_radix(f[3], 16).unwrap(),
                u64::from_str_radix(f[4], 16).unwrap(),
            )),
            "refused" => scan.refused = Some(f[2].to_owned()),
            other => panic!("an unknown row kind {other}"),
        }
    }
    (out, rows)
}

/// The status each refusal kind of the zone access reference is reported with.
fn status_of(kind: &str) -> i32 {
    match kind {
        "zone" | "mask" | "interval" | "tolerance" | "samples" => SL_ERR_INVALID_ARGUMENT,
        "below-surface" => SL_ERR_DOMAIN,
        other => panic!("an unknown refusal kind {other}"),
    }
}

/// The open handles of an SGP4 scan, closed on drop.
struct Handles {
    sat: *mut SlSgp4,
    eop: *mut SlEopSeries,
}

impl Drop for Handles {
    fn drop(&mut self) {
        // SAFETY: each is null or a handle opened below and closed only here.
        unsafe {
            sl_sgp4_close(self.sat);
            sl_eop_series_close(self.eop);
        }
    }
}

fn open(scan: &Scan) -> Handles {
    let mut h = Handles {
        sat: core::ptr::null_mut(),
        eop: core::ptr::null_mut(),
    };
    let Some((line1, line2, eop, _, _)) = &scan.sgp4 else {
        return h;
    };
    let (l1, l2) = (
        CString::new(line1.as_str()).unwrap(),
        CString::new(line2.as_str()).unwrap(),
    );
    // SAFETY: two NUL-terminated lines and a writable out pointer.
    let status = unsafe { sl_sgp4_open(l1.as_ptr(), l2.as_ptr(), &raw mut h.sat) };
    assert_eq!(status, SL_OK, "{}: the TLE opens", scan.name);
    if let Some(path) = eop {
        let text = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(path),
        )
        .expect("an IERS fixture");
        // SAFETY: `text` is readable for its length; the out pointer is writable.
        let status = unsafe {
            sl_eop_series_parse_finals2000a(
                text.as_ptr().cast::<c_char>(),
                text.len(),
                &raw mut h.eop,
            )
        };
        assert_eq!(status, SL_OK, "{}: the series parses", scan.name);
    }
    h
}

/// One call of the scan's entry point over `zones`, writing to `out` (null for none): the status.
fn call_zones(
    h: &Handles,
    scan: &Scan,
    zones: &[Vec<SlSurfacePoint>],
    access: i32,
    out: &mut [SlZoneWindow],
    len: &mut usize,
) -> i32 {
    let vertices: Vec<SlSurfacePoint> = zones.iter().flatten().copied().collect();
    let counts: Vec<usize> = zones.iter().map(Vec::len).collect();
    let (ptr, cap) = if out.is_empty() {
        (core::ptr::null_mut(), 0)
    } else {
        (out.as_mut_ptr(), out.len())
    };
    // SAFETY: live handles, one label, `vertices` readable for the sum of `counts`, `counts` for
    // its length, the samples for their count, and `out` null or writable for `cap` windows.
    unsafe {
        match &scan.sgp4 {
            Some((_, _, _, start, span)) => sl_access_zone_sgp4(
                h.sat,
                h.eop,
                start,
                *span,
                vertices.as_ptr(),
                counts.as_ptr(),
                counts.len(),
                access,
                scan.mask,
                scan.tol,
                ptr,
                cap,
                len,
            ),
            None => sl_access_zone_tabulated(
                scan.times.as_ptr(),
                scan.positions.as_ptr(),
                scan.velocities.as_ptr(),
                scan.times.len(),
                vertices.as_ptr(),
                counts.as_ptr(),
                counts.len(),
                access,
                scan.mask,
                scan.tol,
                ptr,
                cap,
                len,
            ),
        }
    }
}

/// The two-call protocol over `zones`: the status, and the windows as bits.
fn access(
    h: &Handles,
    scan: &Scan,
    zones: &[Vec<SlSurfacePoint>],
) -> (i32, Vec<(usize, u64, u64)>) {
    let mut len = usize::MAX;
    let status = call_zones(h, scan, zones, scan.access, &mut [], &mut len);
    if status != SL_OK {
        assert_eq!(
            len,
            usize::MAX,
            "{}: a refused scan wrote its length",
            scan.name
        );
        return (status, Vec::new());
    }
    let mut out = vec![
        SlZoneWindow {
            zone: usize::MAX,
            rise_s: f64::NAN,
            set_s: f64::NAN,
        };
        len
    ];
    let mut again = 0;
    let status = call_zones(h, scan, zones, scan.access, &mut out, &mut again);
    assert_eq!(
        (status, again),
        (SL_OK, len),
        "{}: the second call",
        scan.name
    );
    let windows = out
        .iter()
        .map(|w| (w.zone, w.rise_s.to_bits(), w.set_s.to_bits()))
        .collect();
    (status, windows)
}

/// (S2) Every scan of the zone access reference through the C ABI: the batch returns every window
/// with the same bits and in the same order, one zone at a time returns the batch's windows of
/// that zone, and every refusal is the status its kind names, nothing written. An access kind
/// that is not an `sl_zone_access` is refused. Every row is counted.
#[test]
fn the_c_abi_returns_the_zone_access_reference_to_the_bit() {
    let (scans, rows) = scans();
    let (mut counted, mut windows, mut refusals) = (0, 0, 0);
    for scan in &scans {
        counted += 1
            + scan.times.len()
            + scan.zones.len()
            + scan.zones.iter().map(Vec::len).sum::<usize>()
            + scan.windows.len()
            + usize::from(scan.refused.is_some());
        let h = open(scan);
        let (status, got) = access(&h, scan, &scan.zones);
        if let Some(kind) = &scan.refused {
            assert_eq!(status, status_of(kind), "{}: {kind}", scan.name);
            refusals += 1;
            continue;
        }
        assert_eq!(status, SL_OK, "{}", scan.name);
        assert_eq!(got, scan.windows, "{}", scan.name);
        windows += got.len();
        for (i, zone) in scan.zones.iter().enumerate() {
            let (status, alone) = access(&h, scan, core::slice::from_ref(zone));
            let want: Vec<(u64, u64)> = scan
                .windows
                .iter()
                .filter(|w| w.0 == i)
                .map(|w| (w.1, w.2))
                .collect();
            let alone: Vec<(u64, u64)> = alone.iter().map(|w| (w.1, w.2)).collect();
            assert_eq!(
                (status, alone),
                (SL_OK, want),
                "{}: zone {i} alone",
                scan.name
            );
        }
        let mut len = usize::MAX;
        assert_eq!(
            call_zones(&h, scan, &scan.zones, 2, &mut [], &mut len),
            SL_ERR_INVALID_ARGUMENT,
            "{}: an access kind that is not one",
            scan.name
        );
        assert_eq!(len, usize::MAX);
    }
    assert_eq!(counted, rows, "a row of the reference was not read");
    assert!(
        windows > 20 && refusals == 6,
        "the reference lost its cases"
    );
}

/// The zone access entry points refuse null pointers, no zones, counts that sum past `SIZE_MAX`
/// and a zone of two vertices, writing nothing, and a short buffer with `SL_ERR_BUFFER_TOO_SMALL`
/// and the length it needs.
#[test]
fn the_zone_access_entry_points_refuse_null_pointers_no_zones_and_overflowing_counts() {
    let square = [(0.0, 0.0), (0.0, 0.02), (0.02, 0.02), (0.02, 0.0)].map(
        |(latitude_rad, longitude_rad)| SlSurfacePoint {
            latitude_rad,
            longitude_rad,
        },
    );
    let vertices: Vec<SlSurfacePoint> = square.iter().chain(&square).copied().collect();
    let times = [0.0, 60.0, 120.0];
    let positions = times.map(|t| SlVec3 {
        x: 6_878_137.0,
        y: -420e3 + 7e3 * t,
        z: 0.0,
    });
    let velocities = [SlVec3 {
        x: 0.0,
        y: 7e3,
        z: 0.0,
    }; 3];
    let scan = |counts: &[usize],
                vertices: *const SlSurfacePoint,
                out: &mut [SlZoneWindow],
                len: &mut usize| {
        let ptr = if out.is_empty() {
            core::ptr::null_mut()
        } else {
            out.as_mut_ptr()
        };
        // SAFETY: every non-null pointer is valid for the count passed with it.
        unsafe {
            sl_access_zone_tabulated(
                times.as_ptr(),
                positions.as_ptr(),
                velocities.as_ptr(),
                3,
                vertices,
                counts.as_ptr(),
                counts.len(),
                SL_ZONE_ANY,
                0.5,
                1e-4,
                ptr,
                out.len(),
                len,
            )
        }
    };
    let mut len = usize::MAX;
    assert_eq!(
        scan(&[4, 4], core::ptr::null(), &mut [], &mut len),
        SL_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        scan(&[], vertices.as_ptr(), &mut [], &mut len),
        SL_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        scan(&[usize::MAX, 4], vertices.as_ptr(), &mut [], &mut len),
        SL_ERR_INVALID_ARGUMENT
    );
    assert_eq!(
        scan(&[4, 2], vertices.as_ptr(), &mut [], &mut len),
        SL_ERR_INVALID_ARGUMENT
    );
    assert_eq!(len, usize::MAX);
    assert_eq!(scan(&[4, 4], vertices.as_ptr(), &mut [], &mut len), SL_OK);
    assert_eq!(len, 2, "one window per copy of the square");
    let mut one = [SlZoneWindow {
        zone: usize::MAX,
        rise_s: 0.0,
        set_s: 0.0,
    }];
    let mut needed = 0;
    assert_eq!(
        scan(&[4, 4], vertices.as_ptr(), &mut one, &mut needed),
        SL_ERR_BUFFER_TOO_SMALL
    );
    assert_eq!(needed, 2);
    let mut two = [one[0]; 2];
    assert_eq!(
        scan(&[4, 4], vertices.as_ptr(), &mut two, &mut needed),
        SL_OK
    );
    assert_eq!((two[0].zone, two[1].zone), (0, 1));
    assert_eq!(two[0].rise_s.to_bits(), two[1].rise_s.to_bits());
}
