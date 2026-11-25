//! The C ABI's access windows return the Rust API's to the bit (Ф31.C).
//!
//! `crates/spacelab/tests/fixtures/bridge_access_reference.tsv` is `spacelab::access::sgp4_windows`
//! on a set of SGP4 scans, and `bridge_access_tabulated_reference.tsv` beside it is
//! `spacelab::access::tabulated_windows` on a set of tables; `bridge_access_reference.rs` and
//! `bridge_access_tabulated.rs` in `crates/spacelab/tests` hold each to the API byte for byte and
//! hold the API to its mathematics. Here every scan of both goes through the exported
//! `sl_access_sgp4` or `sl_access_tabulated`, all stations in one batch and then one station at a
//! time, and every window must come back with the same bits (S2), every refusal with the status the
//! header names. Each set is counted: no row of either file is skipped.

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use core::ffi::c_char;
use std::ffi::CString;
use std::path::{Path, PathBuf};

use spacelab_capi::{
    SL_ERR_BUFFER_TOO_SMALL, SL_ERR_COMPUTE, SL_ERR_DOMAIN, SL_ERR_INVALID_ARGUMENT, SL_OK,
    SlEopSeries, SlGeodetic, SlSgp4, SlStationWindow, SlUtcDateTime, SlVec3, sl_access_sgp4,
    sl_access_tabulated, sl_eop_series_close, sl_eop_series_parse_finals2000a, sl_sgp4_close,
    sl_sgp4_open,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn f64_of(hex: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(hex, 16).expect("16 hex digits"))
}

/// The status each refusal kind of the reference is reported with.
fn status_of(kind: &str) -> i32 {
    match kind {
        "mask" | "interval" | "tolerance" | "station" => SL_ERR_INVALID_ARGUMENT,
        "out-of-range" | "gap" | "before-1972" | "outside-calendar" | "spurious-leap-second" => {
            SL_ERR_DOMAIN
        }
        "decayed" | "diverged" => SL_ERR_COMPUTE,
        other => panic!("an unknown refusal kind {other}"),
    }
}

/// One scan of the reference, with the answer the Rust API gave.
struct Scan {
    name: String,
    line1: String,
    line2: String,
    eop: Option<String>,
    start: SlUtcDateTime,
    span: f64,
    mask: f64,
    tol: f64,
    stations: Vec<SlGeodetic>,
    windows: Vec<(usize, u64, u64)>,
    refused: Option<String>,
}

fn scans() -> (Vec<Scan>, usize) {
    let path = root().join("crates/spacelab/tests/fixtures/bridge_access_reference.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out: Vec<Scan> = Vec::new();
    let mut rows = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        rows += 1;
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] == "scan" {
            out.push(Scan {
                name: f[1].to_owned(),
                line1: f[2].to_owned(),
                line2: f[3].to_owned(),
                eop: (f[4] != "none").then(|| f[4].to_owned()),
                start: SlUtcDateTime {
                    year: f[5].parse().unwrap(),
                    month: f[6].parse().unwrap(),
                    day: f[7].parse().unwrap(),
                    hour: f[8].parse().unwrap(),
                    minute: f[9].parse().unwrap(),
                    second: f64_of(f[10]),
                },
                span: f64_of(f[11]),
                mask: f64_of(f[12]),
                tol: f64_of(f[13]),
                stations: Vec::new(),
                windows: Vec::new(),
                refused: None,
            });
            continue;
        }
        let scan = out.last_mut().expect("a scan row opens each case");
        assert_eq!(f[1], scan.name, "a row outside its scan");
        match f[0] {
            "station" => scan.stations.push(SlGeodetic {
                latitude_rad: f64_of(f[3]),
                longitude_rad: f64_of(f[4]),
                height_m: f64_of(f[5]),
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
    let (l1, l2) = (
        CString::new(scan.line1.as_str()).unwrap(),
        CString::new(scan.line2.as_str()).unwrap(),
    );
    let mut h = Handles {
        sat: core::ptr::null_mut(),
        eop: core::ptr::null_mut(),
    };
    // SAFETY: two NUL-terminated lines and a writable out pointer.
    let status = unsafe { sl_sgp4_open(l1.as_ptr(), l2.as_ptr(), &raw mut h.sat) };
    assert_eq!(status, SL_OK, "{}: the TLE opens", scan.name);
    if let Some(path) = &scan.eop {
        let text = std::fs::read_to_string(root().join(path)).expect("an IERS fixture");
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

/// One call of the two-call protocol over `stations`: the status, and the windows as bits.
fn access(h: &Handles, scan: &Scan, stations: &[SlGeodetic]) -> (i32, Vec<(usize, u64, u64)>) {
    let mut len = usize::MAX;
    // SAFETY: live handles, one label, `stations` readable for its length, a null `out`.
    let status = unsafe {
        sl_access_sgp4(
            h.sat,
            h.eop,
            &raw const scan.start,
            scan.span,
            stations.as_ptr(),
            stations.len(),
            scan.mask,
            scan.tol,
            core::ptr::null_mut(),
            0,
            &raw mut len,
        )
    };
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
        SlStationWindow {
            station: usize::MAX,
            rise_s: f64::NAN,
            set_s: f64::NAN,
        };
        len
    ];
    let mut again = 0;
    // SAFETY: as above, with `out` writable for `len` windows.
    let status = unsafe {
        sl_access_sgp4(
            h.sat,
            h.eop,
            &raw const scan.start,
            scan.span,
            stations.as_ptr(),
            stations.len(),
            scan.mask,
            scan.tol,
            out.as_mut_ptr(),
            out.len(),
            &raw mut again,
        )
    };
    assert_eq!(
        (status, again),
        (SL_OK, len),
        "{}: the second call",
        scan.name
    );
    let windows = out
        .iter()
        .map(|w| (w.station, w.rise_s.to_bits(), w.set_s.to_bits()))
        .collect();
    (status, windows)
}

/// (S2) Every scan of the reference through the C ABI: the batch returns every window with the
/// same bits and in the same order, one station at a time returns the batch's windows of that
/// station, and every refusal is the status its kind names. Every row is counted.
#[test]
fn the_c_abi_returns_the_access_reference_to_the_bit() {
    let (scans, rows) = scans();
    let (mut counted, mut windows, mut refusals) = (0, 0, 0);
    for scan in &scans {
        counted +=
            1 + scan.stations.len() + scan.windows.len() + usize::from(scan.refused.is_some());
        let h = open(scan);
        let (status, got) = access(&h, scan, &scan.stations);
        if let Some(kind) = &scan.refused {
            assert_eq!(status, status_of(kind), "{}: refused as {kind}", scan.name);
            refusals += 1;
            continue;
        }
        assert_eq!(status, SL_OK, "{}", scan.name);
        assert_eq!(got, scan.windows, "{}: the batch", scan.name);
        windows += got.len();
        for (i, station) in scan.stations.iter().enumerate() {
            let (status, single) = access(&h, scan, core::slice::from_ref(station));
            assert_eq!(status, SL_OK, "{}: station {i} alone", scan.name);
            let want: Vec<(u64, u64)> = scan
                .windows
                .iter()
                .filter(|w| w.0 == i)
                .map(|w| (w.1, w.2))
                .collect();
            let have: Vec<(u64, u64)> = single.iter().map(|w| (w.1, w.2)).collect();
            assert_eq!(have, want, "{}: station {i} alone", scan.name);
        }
    }
    assert_eq!(counted, rows, "a row of the reference was not read");
    println!(
        "{} scans, {windows} windows to the bit, {refusals} refusals",
        scans.len()
    );
    assert!(
        windows > 100 && refusals >= 9,
        "the reference lost its cases"
    );
}

/// The protocol's edges: a short buffer reports the length and writes nothing, and a null pointer
/// or an empty station array is refused before any scan.
#[test]
fn the_access_entry_point_refuses_a_short_buffer_and_null_pointers() {
    let (scans, _) = scans();
    let scan = scans
        .iter()
        .find(|s| s.refused.is_none() && s.windows.len() > 2)
        .expect("an answered scan");
    let h = open(scan);
    let mut out = [SlStationWindow {
        station: 7,
        rise_s: 1.5,
        set_s: 2.5,
    }];
    let mut len = 0;
    // SAFETY: live handles; `out` is writable for its one window.
    let status = unsafe {
        sl_access_sgp4(
            h.sat,
            h.eop,
            &raw const scan.start,
            scan.span,
            scan.stations.as_ptr(),
            scan.stations.len(),
            scan.mask,
            scan.tol,
            out.as_mut_ptr(),
            out.len(),
            &raw mut len,
        )
    };
    assert_eq!(status, SL_ERR_BUFFER_TOO_SMALL);
    assert_eq!(len, scan.windows.len());
    assert_eq!(out[0].station, 7, "a short buffer was written");

    let mut len = usize::MAX;
    for (sat, start, stations, n) in [
        (
            core::ptr::null(),
            &raw const scan.start,
            scan.stations.as_ptr(),
            1,
        ),
        (
            h.sat.cast_const(),
            core::ptr::null(),
            scan.stations.as_ptr(),
            1,
        ),
        (
            h.sat.cast_const(),
            &raw const scan.start,
            core::ptr::null(),
            1,
        ),
        (
            h.sat.cast_const(),
            &raw const scan.start,
            scan.stations.as_ptr(),
            0,
        ),
    ] {
        // SAFETY: every non-null pointer is live; the null ones are what is tested.
        let status = unsafe {
            sl_access_sgp4(
                sat,
                h.eop,
                start,
                scan.span,
                stations,
                n,
                scan.mask,
                scan.tol,
                core::ptr::null_mut(),
                0,
                &raw mut len,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert_eq!(len, usize::MAX, "a refused call wrote its length");
    }
    let bad = SlUtcDateTime {
        year: 2008,
        month: 2,
        day: 30,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    // SAFETY: live handles and readable inputs; the label is what is refused.
    let status = unsafe {
        sl_access_sgp4(
            h.sat,
            h.eop,
            &raw const bad,
            scan.span,
            scan.stations.as_ptr(),
            1,
            scan.mask,
            scan.tol,
            core::ptr::null_mut(),
            0,
            &raw mut len,
        )
    };
    assert_eq!(
        status, SL_ERR_INVALID_ARGUMENT,
        "a start the calendar refuses"
    );
}

/// One table of the tabulated reference, with the answer the Rust API gave.
struct Table {
    name: String,
    times: Vec<f64>,
    positions: Vec<SlVec3>,
    velocities: Vec<SlVec3>,
    mask: f64,
    tol: f64,
    stations: Vec<SlGeodetic>,
    windows: Vec<(usize, u64, u64)>,
    refused: Option<String>,
}

fn tables() -> (Vec<Table>, usize) {
    let path = root().join("crates/spacelab/tests/fixtures/bridge_access_tabulated_reference.tsv");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut out: Vec<Table> = Vec::new();
    let mut rows = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        rows += 1;
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] == "table" {
            out.push(Table {
                name: f[1].to_owned(),
                times: Vec::new(),
                positions: Vec::new(),
                velocities: Vec::new(),
                mask: f64_of(f[3]),
                tol: f64_of(f[4]),
                stations: Vec::new(),
                windows: Vec::new(),
                refused: None,
            });
            assert!(f[2].parse::<usize>().is_ok(), "a sample count");
            continue;
        }
        let table = out.last_mut().expect("a table row opens each case");
        assert_eq!(f[1], table.name, "a row outside its table");
        match f[0] {
            "sample" => {
                let x: Vec<f64> = f[3..10].iter().map(|h| f64_of(h)).collect();
                table.times.push(x[0]);
                table.positions.push(SlVec3 {
                    x: x[1],
                    y: x[2],
                    z: x[3],
                });
                table.velocities.push(SlVec3 {
                    x: x[4],
                    y: x[5],
                    z: x[6],
                });
            }
            "station" => table.stations.push(SlGeodetic {
                latitude_rad: f64_of(f[3]),
                longitude_rad: f64_of(f[4]),
                height_m: f64_of(f[5]),
            }),
            "window" => table.windows.push((
                f[2].parse().unwrap(),
                u64::from_str_radix(f[3], 16).unwrap(),
                u64::from_str_radix(f[4], 16).unwrap(),
            )),
            "refused" => table.refused = Some(f[2].to_owned()),
            other => panic!("an unknown row kind {other}"),
        }
    }
    (out, rows)
}

/// One call of the two-call protocol of `sl_access_tabulated` over `stations`: the status, and the
/// windows as bits.
fn access_tabulated(table: &Table, stations: &[SlGeodetic]) -> (i32, Vec<(usize, u64, u64)>) {
    let call = |out: *mut SlStationWindow, cap: usize, len: &mut usize| {
        // SAFETY: the three sample arrays are readable for their common length, `stations` for
        // its own; `out` is null or writable for `cap` windows.
        unsafe {
            sl_access_tabulated(
                table.times.as_ptr(),
                table.positions.as_ptr(),
                table.velocities.as_ptr(),
                table.times.len(),
                stations.as_ptr(),
                stations.len(),
                table.mask,
                table.tol,
                out,
                cap,
                len,
            )
        }
    };
    let mut len = usize::MAX;
    let status = call(core::ptr::null_mut(), 0, &mut len);
    if status != SL_OK {
        assert_eq!(
            len,
            usize::MAX,
            "{}: a refused scan wrote its length",
            table.name
        );
        return (status, Vec::new());
    }
    let mut out = vec![
        SlStationWindow {
            station: usize::MAX,
            rise_s: f64::NAN,
            set_s: f64::NAN,
        };
        len
    ];
    let mut again = 0;
    let status = call(out.as_mut_ptr(), out.len(), &mut again);
    assert_eq!(
        (status, again),
        (SL_OK, len),
        "{}: the second call",
        table.name
    );
    let windows = out
        .iter()
        .map(|w| (w.station, w.rise_s.to_bits(), w.set_s.to_bits()))
        .collect();
    (status, windows)
}

/// (S2) Every table of the tabulated reference through the C ABI: the batch returns every window
/// with the same bits and in the same order, one station at a time returns the batch's windows of
/// that station, and every refusal is `SL_ERR_INVALID_ARGUMENT`, the status the header names for
/// each kind a table scan has. Every row is counted.
#[test]
fn the_c_abi_returns_the_tabulated_reference_to_the_bit() {
    let (tables, rows) = tables();
    let (mut counted, mut windows, mut refusals) = (0, 0, 0);
    for table in &tables {
        counted += 1
            + table.times.len()
            + table.stations.len()
            + table.windows.len()
            + usize::from(table.refused.is_some());
        let (status, got) = access_tabulated(table, &table.stations);
        if let Some(kind) = &table.refused {
            assert!(
                matches!(kind.as_str(), "samples" | "station" | "mask" | "tolerance"),
                "{}: an unknown refusal kind {kind}",
                table.name
            );
            assert_eq!(
                status, SL_ERR_INVALID_ARGUMENT,
                "{}: refused as {kind}",
                table.name
            );
            refusals += 1;
            continue;
        }
        assert_eq!(status, SL_OK, "{}", table.name);
        assert_eq!(got, table.windows, "{}: the batch", table.name);
        windows += got.len();
        for (i, station) in table.stations.iter().enumerate() {
            let (status, single) = access_tabulated(table, core::slice::from_ref(station));
            assert_eq!(status, SL_OK, "{}: station {i} alone", table.name);
            let want: Vec<(u64, u64)> = table
                .windows
                .iter()
                .filter(|w| w.0 == i)
                .map(|w| (w.1, w.2))
                .collect();
            let have: Vec<(u64, u64)> = single.iter().map(|w| (w.1, w.2)).collect();
            assert_eq!(have, want, "{}: station {i} alone", table.name);
        }
    }
    assert_eq!(counted, rows, "a row of the reference was not read");
    println!(
        "{} tables, {windows} windows to the bit, {refusals} refusals",
        tables.len()
    );
    assert!(
        windows > 40 && refusals >= 9,
        "the reference lost its cases"
    );
}

/// The tabulated entry point's protocol edges: a short buffer reports the length and writes
/// nothing, and each null pointer or an empty station array is refused before any scan.
#[test]
fn the_tabulated_entry_point_refuses_a_short_buffer_and_null_pointers() {
    let (tables, _) = tables();
    let table = tables
        .iter()
        .find(|t| t.refused.is_none() && t.windows.len() > 2)
        .expect("an answered table");
    let mut out = [SlStationWindow {
        station: 7,
        rise_s: 1.5,
        set_s: 2.5,
    }];
    let mut len = 0;
    // SAFETY: the arrays are readable for their lengths; `out` is writable for its one window.
    let status = unsafe {
        sl_access_tabulated(
            table.times.as_ptr(),
            table.positions.as_ptr(),
            table.velocities.as_ptr(),
            table.times.len(),
            table.stations.as_ptr(),
            table.stations.len(),
            table.mask,
            table.tol,
            out.as_mut_ptr(),
            out.len(),
            &raw mut len,
        )
    };
    assert_eq!(status, SL_ERR_BUFFER_TOO_SMALL);
    assert_eq!(len, table.windows.len());
    assert_eq!(out[0].station, 7, "a short buffer was written");

    let (t, p, v, s) = (
        table.times.as_ptr(),
        table.positions.as_ptr(),
        table.velocities.as_ptr(),
        table.stations.as_ptr(),
    );
    let mut len = usize::MAX;
    for (t, p, v, s, n) in [
        (core::ptr::null(), p, v, s, 1),
        (t, core::ptr::null(), v, s, 1),
        (t, p, core::ptr::null(), s, 1),
        (t, p, v, core::ptr::null(), 1),
        (t, p, v, s, 0),
    ] {
        // SAFETY: every non-null pointer is readable for its count; the null ones are what is
        // tested.
        let status = unsafe {
            sl_access_tabulated(
                t,
                p,
                v,
                table.times.len(),
                s,
                n,
                table.mask,
                table.tol,
                core::ptr::null_mut(),
                0,
                &raw mut len,
            )
        };
        assert_eq!(status, SL_ERR_INVALID_ARGUMENT);
        assert_eq!(len, usize::MAX, "a refused call wrote its length");
    }
}
