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

/// The five `sdtm1_*` exports of each of SDTM-1's two fields, name for name, on both bridges, and
/// each field's drag constructor under each bridge's own spelling of one.
///
/// Not a section comparison like the test above: the atmospheres section of the C header is mostly
/// two-call length protocols and out-parameter writers that have no same-named Python counterpart, so
/// comparing the whole section would fail on drift that predates this model and is not drift at all.
/// What is worth pinning is the surface that shipped together — each field reached both bridges in
/// one ticket, and the next capability added to one of them should not quietly skip the other. A dense
/// constructor is `sl_ephemeris_*` in C and `propagate_*_dense` in Python, as NRLMSISE-00's drag is.
#[test]
fn both_bridges_name_the_same_sdtm1_surface() {
    const SURFACE: [(&str, &str); 12] = [
        ("sdtm1_density", "sdtm1_density"),
        (
            "sdtm1_exospheric_temperature",
            "sdtm1_exospheric_temperature",
        ),
        ("sdtm1_storm_response", "sdtm1_storm_response"),
        ("sdtm1_density_interval", "sdtm1_density_interval"),
        ("sdtm1_envelope", "sdtm1_envelope"),
        ("ephemeris_drag_sdtm1", "propagate_drag_sdtm1_dense"),
        ("sdtm1_solar_wind_density", "sdtm1_solar_wind_density"),
        (
            "sdtm1_solar_wind_exospheric_temperature",
            "sdtm1_solar_wind_exospheric_temperature",
        ),
        (
            "sdtm1_solar_wind_storm_response",
            "sdtm1_solar_wind_storm_response",
        ),
        (
            "sdtm1_solar_wind_density_interval",
            "sdtm1_solar_wind_density_interval",
        ),
        ("sdtm1_solar_wind_envelope", "sdtm1_solar_wind_envelope"),
        (
            "ephemeris_drag_sdtm1_solar_wind",
            "propagate_drag_sdtm1_solar_wind_dense",
        ),
    ];
    let header = read("include/spacelab.h");
    let bridge = read("../spacelab-python/src/lib.rs");
    // rustfmt wraps a long registration over three lines, so registrations are matched with the
    // whitespace taken out.
    let registrations: String = bridge.split_whitespace().collect();
    for (c_name, py_name) in SURFACE {
        assert!(
            header.contains(&format!("sl_{c_name}(")),
            "the C header declares no sl_{c_name}"
        );
        assert!(
            bridge.contains(&format!("fn {py_name}(")),
            "the Python bridge defines no {py_name}"
        );
        assert!(
            registrations.contains(&format!("wrap_pyfunction!({py_name},")),
            "the Python bridge never registers {py_name}"
        );
    }
}

/// Both bridges tell a caller how many hours of history the solar-wind field's storm state needs,
/// in prose a refit does not regenerate: it has to say the number the generator wrote.
#[test]
fn both_bridges_quote_the_hourly_history_the_solar_wind_field_needs() {
    let hours = format!(
        "{} hours",
        spacelab_propagation::sdtm::solar_wind_coefficients::HISTORY
    );
    for (bridge, relative, at_least) in [
        ("the C header", "include/spacelab.h", 1),
        ("the Python bridge", "../spacelab-python/src/lib.rs", 2),
    ] {
        let quoted = read(relative).matches(&hours).count();
        assert!(
            quoted >= at_least,
            "{bridge} quotes `{hours}` {quoted} times, fewer than its {at_least}"
        );
    }
}

/// The time scales and the Earth-orientation series, capability for capability, on both bridges
/// (Ф31.B). Where the C ABI takes the series handle as its first argument the Python bridge has a
/// method of `EopSeries`; `sl_eop_series_close` alone has no counterpart, because Python's collector
/// closes the handle. Both sections are read whole, so a capability added to one bridge and not the
/// other fails here.
#[test]
fn both_bridges_name_the_same_time_surface() {
    const FUNCTIONS: [&str; 6] = [
        "utc_to_tai",
        "utc_to_tt",
        "utc_to_tdb",
        "utc_to_ut1",
        "tai_to_utc",
        "leap_table_status",
    ];
    const METHODS: [(&str, &str); 4] = [
        ("eop_series_parse_finals2000a", "parse_finals2000a"),
        ("eop_series_parse_c04", "parse_c04"),
        ("eop_series_span", "span"),
        ("eop_series_at", "at"),
    ];
    const C_ONLY: &str = "eop_series_close";

    let header = read("include/spacelab.h");
    let (_, section) = header
        .split_once("/* ---- Time scales and Earth orientation ---- */")
        .expect("the header still has a time banner");
    let (section, _) = section
        .split_once("/* ---- SGP4 ---- */")
        .expect("the SGP4 banner still follows it");
    let declared: BTreeSet<&str> = section
        .lines()
        .filter_map(|line| {
            let rest = line
                .strip_prefix("sl_status sl_")
                .or_else(|| line.strip_prefix("void sl_"))?;
            Some(rest.split_once('(')?.0)
        })
        .collect();
    let mut named: BTreeSet<&str> = FUNCTIONS.into_iter().collect();
    named.extend(METHODS.map(|(c, _)| c));
    named.insert(C_ONLY);
    assert_eq!(declared, named, "the C time section");

    let bridge = read("../spacelab-python/src/lib.rs");
    let (_, section) = bridge
        .split_once("// Time scales and Earth orientation (Ф31.B).")
        .expect("the bridge still has a time banner");
    let (section, _) = section
        .split_once("// ── Control: the loop,")
        .expect("the control banner still follows it");
    let mut functions = BTreeSet::new();
    let mut pending = false;
    for line in section.lines() {
        if line.trim_start() == "#[pyfunction]" {
            pending = true;
        } else if let (true, Some(rest)) = (pending, line.strip_prefix("fn ")) {
            functions.insert(rest.split_once('(').expect("a `fn` opens its arguments").0);
            pending = false;
        }
    }
    assert_eq!(
        functions,
        FUNCTIONS.into_iter().collect(),
        "the Python time section"
    );
    let (_, class) = section
        .split_once("impl PyEopSeries {")
        .expect("the bridge still has the EopSeries methods");
    for (_, method) in METHODS {
        assert!(
            class.contains(&format!("    fn {method}(")),
            "EopSeries has no {method}"
        );
    }

    let registrations: String = bridge.split_whitespace().collect();
    for name in FUNCTIONS {
        assert!(
            registrations.contains(&format!("wrap_pyfunction!({name},")),
            "the Python bridge never registers {name}"
        );
    }
    for registered in [
        "m.add_class::<PyEopSeries>()".to_owned(),
        "m.add(\"LeapSecondError\",m.py().get_type::<LeapSecondError>())".to_owned(),
        "m.add(\"EopParseError\",m.py().get_type::<EopParseError>())".to_owned(),
        "m.add(\"EopRangeError\",m.py().get_type::<EopRangeError>())".to_owned(),
    ] {
        assert!(
            registrations.contains(&registered),
            "the Python bridge never calls {registered}"
        );
    }
}

/// The access scans, capability for capability, on both bridges (Ф31.C). Both sections are read
/// whole, so a scan added to one bridge and not the other fails here.
#[test]
fn both_bridges_name_the_same_access_surface() {
    const FUNCTIONS: [&str; 2] = ["access_sgp4", "access_tabulated"];

    let header = read("include/spacelab.h");
    let (_, section) = header
        .split_once("/* ---- Access on any ephemeris (Ф31.C) ---- */")
        .expect("the header still has an access banner");
    let (section, _) = section
        .split_once("/* ---- Semi-analytical propagation (DSST) ---- */")
        .expect("the DSST banner still follows it");
    let declared: BTreeSet<&str> = section
        .lines()
        .filter_map(|line| Some(line.strip_prefix("sl_status sl_")?.split_once('(')?.0))
        .collect();
    assert_eq!(
        declared,
        FUNCTIONS.into_iter().collect(),
        "the C access section"
    );

    let bridge = read("../spacelab-python/src/lib.rs");
    let (_, section) = bridge
        .split_once("// Access on any ephemeris (Ф31.C).")
        .expect("the bridge still has an access banner");
    let (section, _) = section
        .split_once("// Time scales and Earth orientation (Ф31.B).")
        .expect("the time banner still follows it");
    let mut functions = BTreeSet::new();
    let mut pending = false;
    for line in section.lines() {
        if line.trim_start() == "#[pyfunction]" {
            pending = true;
        } else if let (true, Some(rest)) = (pending, line.strip_prefix("fn ")) {
            functions.insert(rest.split_once('(').expect("a `fn` opens its arguments").0);
            pending = false;
        }
    }
    assert_eq!(
        functions,
        FUNCTIONS.into_iter().collect(),
        "the Python access section"
    );
    let registrations: String = bridge.split_whitespace().collect();
    for registered in [
        "wrap_pyfunction!(access_sgp4,",
        "wrap_pyfunction!(access_tabulated,",
        "m.add(\"PropagationError\",m.py().get_type::<PropagationError>())",
    ] {
        assert!(
            registrations.contains(registered),
            "the Python bridge never calls {registered}"
        );
    }
}
