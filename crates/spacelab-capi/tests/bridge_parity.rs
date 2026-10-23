//! The two bridges name the same control surface.
//!
//! #319 was filed because the C ABI and the Python bridge drift apart silently: a capability
//! reaches one and nobody notices the other never got it. #378 reintroduced the drift while
//! closing it — three of the control exports shipped under two different spellings. Nothing
//! catches that except a test that reads both surfaces and compares them, so this is it.
//!
//! It compares *names*, not signatures. The two bridges cannot have identical signatures: the C
//! side splits a variable-length Python return into a two-call length protocol and writes through
//! out-parameters. What has to match is which capabilities each one offers, and under what name.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::BTreeSet;
use std::path::Path;

/// The banner that opens the control section of the C header; the section runs to the end.
const HEADER_MARK: &str = "/* ---- Closed-loop control ---- */";
/// The banner that opens the control section of the Python bridge …
const BRIDGE_MARK: &str = "// ── Control: the loop,";
/// … and the module registration that closes it.
const BRIDGE_END: &str = "pub fn register(";

/// A marker that stopped matching would leave an empty set on one side, and two empty sets compare
/// equal — the vacuous pass this test exists to prevent. Both extractions must clear this floor.
const FLOOR: usize = 20;

fn read(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every `sl_*` declared in the header's control section, with the prefix stripped.
fn c_abi_names() -> BTreeSet<String> {
    let header = read("include/spacelab.h");
    let (_, control) = header
        .split_once(HEADER_MARK)
        .expect("the header still has a control banner");
    control
        .lines()
        .filter_map(|line| {
            let rest = line
                .strip_prefix("sl_status sl_")
                .or_else(|| line.strip_prefix("size_t sl_"))?;
            let (name, _) = rest.split_once('(')?;
            Some(name.to_owned())
        })
        .collect()
}

/// Every `#[pyfunction]` in the bridge's control section, named by the `fn` that follows it.
fn bridge_names() -> BTreeSet<String> {
    let bridge = read("../spacelab-python/src/lib.rs");
    let (_, rest) = bridge
        .split_once(BRIDGE_MARK)
        .expect("the bridge still has a control banner");
    let (control, _) = rest
        .split_once(BRIDGE_END)
        .expect("the bridge still registers its module after the control section");
    let mut names = BTreeSet::new();
    let mut pending = false;
    for line in control.lines() {
        if line.trim_start() == "#[pyfunction]" {
            pending = true;
        } else if let (true, Some(rest)) = (pending, line.strip_prefix("fn ")) {
            let (name, _) = rest
                .split_once('(')
                .expect("a `fn` line opens its arguments");
            names.insert(name.to_owned());
            pending = false;
        }
    }
    names
}

#[test]
fn the_two_bridges_expose_the_same_control_surface() {
    let c = c_abi_names();
    let py = bridge_names();
    assert!(
        c.len() >= FLOOR,
        "the header extraction found only {}",
        c.len()
    );
    assert!(
        py.len() >= FLOOR,
        "the bridge extraction found only {}",
        py.len()
    );

    let only_c: Vec<_> = c.difference(&py).collect();
    let only_py: Vec<_> = py.difference(&c).collect();
    assert!(
        only_c.is_empty() && only_py.is_empty(),
        "the bridges have drifted: the C ABI alone has {only_c:?}, the Python bridge alone has \
         {only_py:?}. A capability that reaches one bridge reaches both, under one name."
    );
}

/// The four `sdtm1_*` exports, name for name, on both bridges.
///
/// Not a section comparison like the test above: the atmospheres section of the C header is mostly
/// two-call length protocols and out-parameter writers that have no same-named Python counterpart, so
/// comparing the whole section would fail on drift that predates this model and is not drift at all.
/// What is worth pinning is the surface that shipped together — SDTM-1 reached both bridges in one
/// ticket, and the next capability added to one of them should not quietly skip the other.
#[test]
fn both_bridges_name_the_same_sdtm1_surface() {
    const SURFACE: [&str; 4] = [
        "sdtm1_density",
        "sdtm1_exospheric_temperature",
        "sdtm1_storm_response",
        "sdtm1_density_interval",
    ];
    let header = read("include/spacelab.h");
    let bridge = read("../spacelab-python/src/lib.rs");
    for name in SURFACE {
        assert!(
            header.contains(&format!("sl_{name}(")),
            "the C header declares no sl_{name}"
        );
        assert!(
            bridge.contains(&format!("fn {name}(")),
            "the Python bridge defines no {name}"
        );
        assert!(
            bridge.contains(&format!("wrap_pyfunction!({name},")),
            "the Python bridge never registers {name}"
        );
    }
}
