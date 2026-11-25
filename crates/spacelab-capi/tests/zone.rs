//! The C ABI's point-in-zone answers are the Rust API's (Ф31.C).
//!
//! `crates/spacelab/tests/fixtures/bridge_zone_contains_reference.tsv` is
//! `spacelab::zone::Zone::contains` on a set of zones and points, and `bridge_zone_contains.rs` in
//! `crates/spacelab/tests` holds it to the API byte for byte and the API to the cases its model
//! names. Here every case goes through the exported `sl_zone_contains`, all points in one batch and
//! then one point at a time, and every answer must come back the same, every refusal with
//! `SL_ERR_INVALID_ARGUMENT` and nothing written. Every row of the file is counted.

#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "a test fails loudly on a fixture or a case that does not hold"
)]

use std::path::Path;

use spacelab_capi::{SL_ERR_INVALID_ARGUMENT, SL_OK, SlSurfacePoint, sl_zone_contains};

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
