//! The C ABI's time scales and Earth orientation return the Rust API's values to the bit (Ф31.B).
//!
//! `crates/spacelab/tests/fixtures/bridge_time_reference.tsv` is the Rust API on every IERS case in
//! the tree; `crates/spacelab/tests/bridge_time_reference.rs` holds it to the API byte for byte.
//! Here every row of it goes through the exported `extern "C"` functions, one label at a time and
//! then as one batch, and each value must come back with the same bits (S2), each refusal with the
//! status the header names. The cases are the IERS files themselves (G5), so the set is counted:
//! no row is skipped, and the one divergence from the Python bridge, an epoch past `int64_t`, is
//! counted by name.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use core::ffi::c_char;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use spacelab_capi::{
    SL_ERR_DOMAIN, SL_ERR_INVALID_ARGUMENT, SL_ERR_PARSE, SL_OK, SlEopSeries, SlEopValue,
    SlLeapStatus, SlUtcDateTime, sl_eop_series_at, sl_eop_series_close, sl_eop_series_parse_c04,
    sl_eop_series_parse_finals2000a, sl_eop_series_span, sl_leap_table_status, sl_tai_to_utc,
    sl_utc_to_tai, sl_utc_to_tdb, sl_utc_to_tt, sl_utc_to_ut1,
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn reference() -> Vec<Vec<String>> {
    let path = root().join("crates/spacelab/tests/fixtures/bridge_time_reference.tsv");
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| l.split('\t').map(str::to_owned).collect())
        .collect()
}

fn f64_of(hex: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(hex, 16).expect("16 hex digits"))
}

fn label(f: &[String]) -> SlUtcDateTime {
    SlUtcDateTime {
        year: f[0].parse().unwrap(),
        month: f[1].parse().unwrap(),
        day: f[2].parse().unwrap(),
        hour: f[3].parse().unwrap(),
        minute: f[4].parse().unwrap(),
        second: f64_of(&f[5]),
    }
}

fn same_label(got: &SlUtcDateTime, want: &SlUtcDateTime) -> bool {
    (
        got.year,
        got.month,
        got.day,
        got.hour,
        got.minute,
        got.second.to_bits(),
    ) == (
        want.year,
        want.month,
        want.day,
        want.hour,
        want.minute,
        want.second.to_bits(),
    )
}

/// The status a refusal kind is reported with. A label the calendar refuses is malformed input; the
/// rest are questions with no answer.
fn refusal(kind: &str) -> i32 {
    match kind {
        "calendar" => SL_ERR_INVALID_ARGUMENT,
        "before-1972" | "spurious-leap-second" | "outside-calendar" | "out-of-range" | "gap" => {
            SL_ERR_DOMAIN
        }
        other => panic!("an unknown refusal kind {other}"),
    }
}

type Convert = unsafe extern "C" fn(*const SlUtcDateTime, usize, *mut i64) -> i32;

/// One label through one conversion: the status, and the epoch when there is one.
fn one(convert: Convert, l: SlUtcDateTime) -> (i32, i64) {
    let mut out = i64::MIN;
    // SAFETY: both pointers are to locals, each one element long.
    let status = unsafe { convert(&raw const l, 1, &raw mut out) };
    (status, out)
}

struct Series(*mut SlEopSeries);

impl Drop for Series {
    fn drop(&mut self) {
        // SAFETY: the handle came from a parse and is closed once, here.
        unsafe { sl_eop_series_close(self.0) };
    }
}

fn open(format: &str, text: &str) -> (i32, Series) {
    let mut raw: *mut SlEopSeries = core::ptr::null_mut();
    let parse = match format {
        "c04" => sl_eop_series_parse_c04,
        "finals2000a" => sl_eop_series_parse_finals2000a,
        other => panic!("an unknown format {other}"),
    };
    // SAFETY: `text` is readable for its length; `raw` is a local pointer.
    let status = unsafe { parse(text.as_ptr().cast::<c_char>(), text.len(), &raw mut raw) };
    (status, Series(raw))
}

/// Every row of the reference, through the C ABI, to the bit.
#[test]
#[allow(clippy::too_many_lines)] // one pass over the file, one arm per row kind
fn the_c_abi_returns_the_reference_to_the_bit() {
    let rows = reference();
    let mut series: BTreeMap<String, Series> = BTreeMap::new();
    let mut counted: BTreeMap<&str, usize> = BTreeMap::new();
    let mut past_int64 = 0_usize;
    let (mut batch_in, mut batch_tai) = (Vec::new(), Vec::new());

    for r in &rows {
        *counted
            .entry(match r[0].as_str() {
                "source" => "source",
                "status" => "status",
                "span" => "span",
                "utc" => "utc",
                "tai" => "tai",
                "eop" => "eop",
                "ut1" => "ut1",
                other => panic!("an unknown row kind {other}"),
            })
            .or_default() += 1;
        match r[0].as_str() {
            "source" => {
                let text = std::fs::read_to_string(root().join(&r[3])).expect("an IERS fixture");
                let (status, handle) = open(&r[2], &text);
                assert_eq!(status, SL_OK, "{}", r[1]);
                series.insert(r[1].clone(), handle);
            }
            "status" => {
                let mut s = SlLeapStatus {
                    first: label(&["0", "1", "1", "0", "0", "0"].map(String::from)),
                    last_step: label(&["0", "1", "1", "0", "0", "0"].map(String::from)),
                    list_updated: label(&["0", "1", "1", "0", "0", "0"].map(String::from)),
                    list_expires: label(&["0", "1", "1", "0", "0", "0"].map(String::from)),
                    continuous_from: label(&["0", "1", "1", "0", "0", "0"].map(String::from)),
                    last_delta_at_seconds: 0,
                };
                // SAFETY: `s` is a local of the right type.
                assert_eq!(unsafe { sl_leap_table_status(&raw mut s) }, SL_OK);
                let date = |i: usize| {
                    let mut f: Vec<String> = r[i..i + 3].to_vec();
                    f.extend([
                        "0".into(),
                        "0".into(),
                        format!("{:016x}", 0.0_f64.to_bits()),
                    ]);
                    label(&f)
                };
                for (got, want, what) in [
                    (&s.first, date(1), "first"),
                    (&s.last_step, date(4), "last step"),
                    (&s.list_updated, date(8), "list updated"),
                    (&s.list_expires, date(11), "list expires"),
                    (&s.continuous_from, date(14), "continuous from"),
                ] {
                    assert!(same_label(got, &want), "{what}: {got:?} is not {want:?}");
                }
                assert_eq!(s.last_delta_at_seconds.to_string(), r[7]);
            }
            "span" => {
                let (mut first, mut last) = (f64::NAN, f64::NAN);
                // SAFETY: a live handle and two locals.
                let status =
                    unsafe { sl_eop_series_span(series[&r[1]].0, &raw mut first, &raw mut last) };
                assert_eq!(status, SL_OK);
                assert_eq!(first.to_bits(), f64_of(&r[2]).to_bits(), "{}", r[1]);
                assert_eq!(last.to_bits(), f64_of(&r[3]).to_bits(), "{}", r[1]);
            }
            "utc" => {
                let l = label(&r[1..7]);
                if r[7] == "refused" {
                    for convert in [sl_utc_to_tai as Convert, sl_utc_to_tt, sl_utc_to_tdb] {
                        assert_eq!(one(convert, l).0, refusal(&r[8]), "{l:?}");
                    }
                    continue;
                }
                let mut fits = true;
                for (convert, want) in [
                    (sl_utc_to_tai as Convert, &r[7]),
                    (sl_utc_to_tt, &r[8]),
                    (sl_utc_to_tdb, &r[9]),
                ] {
                    let want: i128 = want.parse().unwrap();
                    let (status, got) = one(convert, l);
                    if let Ok(want) = i64::try_from(want) {
                        assert_eq!((status, got), (SL_OK, want), "{l:?}");
                    } else {
                        assert_eq!(status, SL_ERR_DOMAIN, "{l:?} is past int64_t");
                        fits = false;
                    }
                }
                if fits {
                    batch_in.push(l);
                    batch_tai.push(r[7].parse::<i64>().unwrap());
                } else {
                    past_int64 += 1;
                }
            }
            "tai" => {
                let ns: i64 = r[1].parse().unwrap();
                let mut got = label(&["0", "1", "1", "0", "0", "0"].map(String::from));
                // SAFETY: both pointers are to locals, one element each.
                let status = unsafe { sl_tai_to_utc(&raw const ns, 1, &raw mut got) };
                if r[2] == "refused" {
                    assert_eq!(status, refusal(&r[3]), "{ns}");
                } else {
                    assert_eq!(status, SL_OK, "{ns}");
                    assert!(same_label(&got, &label(&r[2..8])), "{ns}: {got:?}");
                }
            }
            "eop" => {
                let mjd = f64_of(&r[2]);
                let mut got = SlEopValue {
                    xp_rad: f64::NAN,
                    yp_rad: f64::NAN,
                    dut1_seconds: f64::NAN,
                    lod_seconds: f64::NAN,
                    dx_rad: f64::NAN,
                    dy_rad: f64::NAN,
                    xp_sigma_rad: f64::NAN,
                    yp_sigma_rad: f64::NAN,
                    dut1_sigma_seconds: f64::NAN,
                    quality: -1,
                };
                // SAFETY: a live handle and two locals, one element each.
                let status =
                    unsafe { sl_eop_series_at(series[&r[1]].0, &raw const mjd, 1, &raw mut got) };
                if r[3] == "refused" {
                    assert_eq!(status, refusal(&r[4]), "{} at {mjd}", r[1]);
                    continue;
                }
                assert_eq!(status, SL_OK, "{} at {mjd}", r[1]);
                let values = [
                    got.xp_rad,
                    got.yp_rad,
                    got.dut1_seconds,
                    got.lod_seconds,
                    got.dx_rad,
                    got.dy_rad,
                ];
                for (k, v) in values.iter().enumerate() {
                    assert_eq!(
                        v.to_bits(),
                        f64_of(&r[3 + k]).to_bits(),
                        "{} at {mjd}, {k}",
                        r[1]
                    );
                }
                // `SL_EOP_FINAL` and `SL_EOP_PREDICTED`, whose values the header test pins.
                let quality = match r[9].as_str() {
                    "final" => 0,
                    "predicted" => 1,
                    other => panic!("an unknown quality {other}"),
                };
                assert_eq!(got.quality, quality, "{} at {mjd}", r[1]);
                let sigmas = [got.xp_sigma_rad, got.yp_sigma_rad, got.dut1_sigma_seconds];
                for (k, v) in sigmas.iter().enumerate() {
                    assert_eq!(
                        v.to_bits(),
                        f64_of(&r[10 + k]).to_bits(),
                        "{} at {mjd}",
                        r[1]
                    );
                }
            }
            "ut1" => {
                let l = label(&r[2..8]);
                let mut got = i64::MIN;
                // SAFETY: a live handle and two locals, one element each.
                let status =
                    unsafe { sl_utc_to_ut1(series[&r[1]].0, &raw const l, 1, &raw mut got) };
                if r[8] == "refused" {
                    assert_eq!(status, refusal(&r[9]), "{} at {l:?}", r[1]);
                } else {
                    assert_eq!(
                        (status, got),
                        (SL_OK, r[8].parse().unwrap()),
                        "{} {l:?}",
                        r[1]
                    );
                }
            }
            _ => unreachable!(),
        }
    }

    // The whole set was read, kind by kind, and the one named divergence is the label of 2300.
    assert_eq!(counted.values().sum::<usize>(), rows.len());
    for kind in ["source", "status", "span", "utc", "tai", "eop", "ut1"] {
        assert!(counted.get(kind).copied().unwrap_or(0) > 0, "no {kind} row");
    }
    assert_eq!(past_int64, 1, "the labels past int64_t");

    // Batch ≡ scalar: every representable label at once gives what one at a time gave.
    let mut all = vec![i64::MIN; batch_in.len()];
    // SAFETY: both buffers are `batch_in.len()` long.
    let status = unsafe { sl_utc_to_tai(batch_in.as_ptr(), batch_in.len(), all.as_mut_ptr()) };
    assert_eq!(status, SL_OK);
    assert_eq!(all, batch_tai);
}

/// A batch that meets one refused label writes nothing, wherever the label sits, and an empty batch
/// is an answer.
#[test]
fn a_refused_label_leaves_the_whole_batch_unwritten() {
    let good = SlUtcDateTime {
        year: 2017,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let spurious = SlUtcDateTime {
        year: 2018,
        month: 12,
        day: 31,
        hour: 23,
        minute: 59,
        second: 60.0,
    };
    let bad_month = SlUtcDateTime { month: 13, ..good };
    for (labels, status) in [
        ([good, spurious, good], SL_ERR_DOMAIN),
        ([good, good, bad_month], SL_ERR_INVALID_ARGUMENT),
    ] {
        for convert in [sl_utc_to_tai as Convert, sl_utc_to_tt, sl_utc_to_tdb] {
            let mut out = [7_i64; 3];
            // SAFETY: both buffers are three long.
            assert_eq!(
                unsafe { convert(labels.as_ptr(), 3, out.as_mut_ptr()) },
                status
            );
            assert_eq!(out, [7; 3], "a refused batch wrote");
        }
    }
    let mut out = [7_i64; 1];
    // SAFETY: an empty batch reads and writes nothing; the pointers are valid anyway.
    assert_eq!(
        unsafe { sl_utc_to_tai(&raw const good, 0, out.as_mut_ptr()) },
        SL_OK
    );
    assert_eq!(out, [7]);
}

/// Every pointer is checked, a closed `NULL` is a no-op, and text that does not parse is
/// `SL_ERR_PARSE` while text that is not UTF-8 is `SL_ERR_INVALID_ARGUMENT`.
#[test]
fn the_time_entry_points_refuse_null_pointers_and_bad_text() {
    let l = SlUtcDateTime {
        year: 2017,
        month: 1,
        day: 1,
        hour: 0,
        minute: 0,
        second: 0.0,
    };
    let finals = std::fs::read_to_string(
        root().join("crates/spacelab-eop/tests/fixtures/finals2000A_excerpt.txt"),
    )
    .unwrap();
    let (status, s) = open("finals2000a", &finals);
    assert_eq!(status, SL_OK);
    let (mut epoch, mut value) = (0_i64, core::mem::MaybeUninit::<SlEopValue>::uninit());
    let mut status_out = core::mem::MaybeUninit::<SlLeapStatus>::uninit();
    let mut utc_out = core::mem::MaybeUninit::<SlUtcDateTime>::uninit();
    let (mut first, mut last) = (0.0, 0.0);
    let mjd = 57_754.0;
    let null_l = core::ptr::null::<SlUtcDateTime>();
    let null_i = core::ptr::null_mut::<i64>();

    // SAFETY: every non-null pointer is to a live local; the null ones are the input under test.
    unsafe {
        for convert in [sl_utc_to_tai as Convert, sl_utc_to_tt, sl_utc_to_tdb] {
            assert_eq!(convert(null_l, 1, &raw mut epoch), SL_ERR_INVALID_ARGUMENT);
            assert_eq!(convert(&raw const l, 1, null_i), SL_ERR_INVALID_ARGUMENT);
        }
        assert_eq!(
            sl_utc_to_ut1(core::ptr::null(), &raw const l, 1, &raw mut epoch),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_utc_to_ut1(s.0, null_l, 1, &raw mut epoch),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_utc_to_ut1(s.0, &raw const l, 1, null_i),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_tai_to_utc(core::ptr::null(), 1, utc_out.as_mut_ptr()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_tai_to_utc(&raw const epoch, 1, core::ptr::null_mut()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_leap_table_status(core::ptr::null_mut()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(sl_leap_table_status(status_out.as_mut_ptr()), SL_OK);
        assert_eq!(
            sl_eop_series_span(core::ptr::null(), &raw mut first, &raw mut last),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_eop_series_span(s.0, core::ptr::null_mut(), &raw mut last),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_eop_series_span(s.0, &raw mut first, core::ptr::null_mut()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_eop_series_at(core::ptr::null(), &raw const mjd, 1, value.as_mut_ptr()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_eop_series_at(s.0, core::ptr::null(), 1, value.as_mut_ptr()),
            SL_ERR_INVALID_ARGUMENT
        );
        assert_eq!(
            sl_eop_series_at(s.0, &raw const mjd, 1, core::ptr::null_mut()),
            SL_ERR_INVALID_ARGUMENT
        );
        for parse in [sl_eop_series_parse_finals2000a, sl_eop_series_parse_c04] {
            let mut raw: *mut SlEopSeries = core::ptr::null_mut();
            assert_eq!(
                parse(finals.as_ptr().cast(), finals.len(), core::ptr::null_mut()),
                SL_ERR_INVALID_ARGUMENT
            );
            assert_eq!(
                parse(core::ptr::null(), 0, &raw mut raw),
                SL_ERR_INVALID_ARGUMENT
            );
            let not_utf8 = [0xff_u8, 0xfe, b'\n'];
            assert_eq!(
                parse(not_utf8.as_ptr().cast(), not_utf8.len(), &raw mut raw),
                SL_ERR_INVALID_ARGUMENT
            );
            assert!(raw.is_null(), "a refused parse handed out a handle");
        }
        // A finals2000A file is not a C04 one: it has no model line, and the reader says so.
        let mut raw: *mut SlEopSeries = core::ptr::null_mut();
        assert_eq!(
            sl_eop_series_parse_c04(finals.as_ptr().cast(), finals.len(), &raw mut raw),
            SL_ERR_PARSE
        );
        assert_eq!(
            sl_eop_series_parse_finals2000a(b"".as_ptr().cast(), 0, &raw mut raw),
            SL_ERR_PARSE
        );
        assert!(raw.is_null());
        sl_eop_series_close(core::ptr::null_mut());
    }
}
