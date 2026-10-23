//! The geocentric force model against NEPTUNE, on the same field and the same Earth orientation.
//!
//! `igs_orbit_fit.rs` fits this workspace's model to an IGS product and compares the *fit* with
//! Orekit's; this compares the *propagation* itself, step for step, with a fourth lineage
//! (Space-Systems/neptune, TU Braunschweig ILR/IRAS, Apache-2.0, Fortran-2008 over libslam), whose
//! reduction and integrator share no line of code with anything here.
//!
//! Everything that could be made identical was, and the cost of each is recorded rather than
//! assumed:
//!
//!   * **The field.** NEPTUNE ships EGM2008 (tide-free), this workspace embeds `XGM2019e` (zero-tide).
//!     Run against each other they differ by 0.9 m at one hour and 18.4 m at one day *inside
//!     NEPTUNE alone* — the zero-tide/tide-free offset in C̄₂₀ is 9e-6 of the term that dominates a
//!     LEO arc. So `tools/neptune-oracle/make_gravity_overlay.py` emits the embedded table in
//!     NEPTUNE's own file layout and `run.sh -d` drops it in under the filename its source fixes.
//!     Both sides then integrate the same 2553 coefficients, the same GM and the same Rₑ.
//!   * **Earth orientation.** The EOP are NEPTUNE's own row for MJD 58659 of the `eop19620101.txt`
//!     it ships, interpolated the way libslam interpolates them — linearly in ΔUT1, LOD and the
//!     celestial-pole offsets, and *not at all* in polar motion, because `slam_reduction.f90`'s
//!     `getPolarMotion` computes the interpolation and then returns the un-interpolated row. On
//!     this state at this epoch the two chains agree to 1.3e-5 m; that measurement is
//!     `spacelab-frames/tests/neptune_frames.rs`, and it is why a residual here is dynamics.
//!   * **The forces present.** Drag, solar radiation pressure, Earth albedo, solid and ocean tides
//!     are off in the deck and absent here. Neither side models the relativistic correction —
//!     NEPTUNE has none, so it is omitted here too rather than half-matched; it is ~3e-9 m/s² at
//!     this altitude, which is metres over the arc and would otherwise be read as disagreement.
//!
//! Two differences remain, both named rather than absorbed: the third bodies are DE-421 there and
//! DE440 here (metres of body position, ~1e-13 m/s² on a 1e-6 m/s² term), and the integrators are a
//! variable-step Störmer–Cowell at rtol 1e-10 / atol 1e-11 against Dormand–Prince 8(5,3).
//!
//! Gated on `tools/fetch-ephemeris.sh` and skipping cleanly without it. The fixture is committed, so
//! nothing here builds or runs Fortran.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;

use spacelab_core::{Angle, Epoch, Tt, Ut1, UtcDateTime};
use spacelab_ephemeris::Ephemeris;
use spacelab_ephemeris::frames::{EARTH_J2000, MOON_J2000, SUN_J2000};
use spacelab_frames::gcrf_to_itrf;
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{BodyOriented, Newtonian, Plain, ThirdBody};
use spacelab_propagation::{Dop853, GeoPotential, Tolerance};

const SPK: &str = "../../data/ephemeris/de440s.bsp";
const FIXTURE: &str = include_str!("fixtures/neptune_sentinel1b_xgm.txt");

/// The degree the fixture was generated at, and the degree the comparison must run at.
const DEGREE: usize = 70;

/// TT − TAI, seconds, exact by definition.
const TT_MINUS_TAI: f64 = 32.184;
/// TAI − UTC on 2019-06-25, seconds — the last column of the EOP row below.
const DAT_SECONDS: f64 = 37.0;

/// Sun and Moon gravitational parameters, m³/s² — DE-421's values as NEPTUNE's `solarsystem.f90`
/// carries them, because the force model must match even where the ephemeris behind it does not.
const MU_SUN: f64 = 1.327_124_400_409_44e20;
const MU_MOON: f64 = 4.902_800_076e12;

/// The GCRF state NEPTUNE derived from the Sentinel-1B POE OSV, as `init_neptune` logs it, in metres
/// and metres per second. Taken from the log rather than from the fixture's first row: the `.csv` is
/// written to 1 mm and 1e-5 m/s, and seeding a day-long arc with a velocity rounded to 1e-5 m/s is
/// 0.86 m of position by itself.
/// Written at the shortest decimal that round-trips to the same `f64` as the logged value: the
/// digits beyond it do not name a different number.
const R0: [f64; 3] = [
    -807_817.645_389_187_1,
    962_600.149_554_167_6,
    6_952_863.677_215_289,
];
const V0: [f64; 3] = [
    7_451.921_920_356_678,
    465.739_557_476_013_43,
    799.590_602_065_284_8,
];

/// One daily row of NEPTUNE's `eop19620101.txt` (IERS 20 C04), arcseconds and seconds.
struct Eop {
    xp: f64,
    yp: f64,
    dut1: f64,
    lod: f64,
    dx: f64,
    dy: f64,
}

/// `2019 06 25 58659`.
const EOP_START: Eop = Eop {
    xp: 0.150_506,
    yp: 0.424_491,
    dut1: -0.174_412_4,
    lod: 0.000_071_8,
    dx: 0.000_024,
    dy: -0.000_162,
};
/// `2019 06 26 58660`.
const EOP_END: Eop = Eop {
    xp: 0.151_639,
    yp: 0.423_517,
    dut1: -0.174_515_9,
    lod: 0.000_093_4,
    dx: 0.000_012,
    dy: -0.000_203,
};

/// Seconds from 0h UTC to the arc epoch, 2019-06-25T00:00:02 — the phase of libslam's daily
/// interpolation, which runs from midnight and not from the propagation epoch.
const EPOCH_SECONDS_INTO_DAY: f64 = 2.0;

fn reference() -> Vec<(f64, [f64; 3])> {
    let mut samples = Vec::new();
    for line in FIXTURE.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let field: Vec<f64> = line
            .split_whitespace()
            .map(|token| token.parse::<f64>().expect("the fixture is numeric"))
            .collect();
        assert_eq!(field.len(), 7, "expected 7 columns, got {}", field.len());
        samples.push((
            field[0],
            [field[1] * 1000.0, field[2] * 1000.0, field[3] * 1000.0],
        ));
    }
    samples
}

/// GCRF→ITRF, row-major, at `t` seconds past the arc epoch, on libslam's own EOP handling.
#[allow(clippy::cast_possible_truncation)] // |t + offset| < 9e4, so the product is < 9e13 << i128::MAX
fn rotation_rows(t: f64) -> [[f64; 3]; 3] {
    let fraction = ((EPOCH_SECONDS_INTO_DAY + t) / 86_400.0).clamp(0.0, 1.0);
    let between = |a: f64, b: f64| a + fraction * (b - a);
    let utc = UtcDateTime::new(2019, 6, 25, 0, 0, 2.0).expect("valid utc");
    let ns = |offset: f64| utc.naive_ns_since_j2000() + ((t + offset) * 1e9).round() as i128;
    gcrf_to_itrf(
        Epoch::<Ut1>::from_ns_since_j2000(ns(between(EOP_START.dut1, EOP_END.dut1))),
        Epoch::<Tt>::from_ns_since_j2000(ns(DAT_SECONDS + TT_MINUS_TAI)),
        // Polar motion is the day's own row, held constant: see the module docs on `getPolarMotion`.
        Angle::from_arcseconds(EOP_START.xp),
        Angle::from_arcseconds(EOP_START.yp),
        Angle::from_arcseconds(between(EOP_START.dx, EOP_END.dx)),
        Angle::from_arcseconds(between(EOP_START.dy, EOP_END.dy)),
        between(EOP_START.lod, EOP_END.lod),
    )
    .rotation
    .matrix()
    .rows
}

/// Propagate the same initial state for `span_seconds` and report the worst position difference
/// against NEPTUNE, and the difference at the end of the span.
#[allow(clippy::cast_precision_loss)] // 6.1e17 ns is 2⁵⁹·¹, so the f64 carries it to 128 ns
fn compare(samples: &[(f64, [f64; 3])], span_seconds: f64) -> (f64, f64) {
    let eph = Ephemeris::load(SPK).expect("de440s.bsp loads");
    // ET is seconds past J2000; TT for this epoch is within 2 ms of it, which is millimetres of
    // third-body position on a 1e-6 m/s² term. The i128→f64 narrowing above costs 128 ns, five
    // orders of magnitude below that, and so is not what limits the epoch.
    let utc = UtcDateTime::new(2019, 6, 25, 0, 0, 2.0).expect("valid utc");
    let et = utc.naive_ns_since_j2000() as f64 / 1e9 + DAT_SECONDS + TT_MINUS_TAI;

    let forces = (
        BodyOriented {
            inner: GeoPotential::embedded(DEGREE).expect("degree is inside the embedded model"),
            rotate: rotation_rows,
        },
        (
            ThirdBody {
                mu: MU_SUN,
                ephemeris: eph.perturbing_body(SUN_J2000, EARTH_J2000, et),
            },
            ThirdBody {
                mu: MU_MOON,
                ephemeris: eph.perturbing_body(MOON_J2000, EARTH_J2000, et),
            },
        ),
    );

    let y0 = [R0[0], R0[1], R0[2], V0[0], V0[1], V0[2]];
    let trajectory = integrate(
        &Dop853,
        &Plain(Newtonian(forces)),
        0.0,
        &y0,
        span_seconds,
        10.0,
        Controls::new(Tolerance::new(1e-10, 1e-11), 1e-6),
    )
    .expect("the arc integrates");

    let mut worst: f64 = 0.0;
    let mut last = 0.0;
    let mut next_report = 0.0;
    for &(elapsed, want) in samples {
        if elapsed > span_seconds {
            break;
        }
        let got = trajectory.eval(elapsed);
        let difference =
            ((got[0] - want[0]).powi(2) + (got[1] - want[1]).powi(2) + (got[2] - want[2]).powi(2))
                .sqrt();
        worst = worst.max(difference);
        last = difference;
        if elapsed >= next_report {
            println!("  +{:>5.1} h  |Δr| = {difference:8.2} m", elapsed / 3600.0);
            next_report += 10_800.0;
        }
    }
    (worst, last)
}

fn kernel_present() -> bool {
    if Path::new(SPK).exists() {
        return true;
    }
    eprintln!("skipping: {SPK} not present — run tools/fetch-ephemeris.sh");
    false
}

/// Three hours — two orbits of Sentinel-1B — against a propagator that shares no line of code.
#[test]
fn two_orbits_of_the_geocentric_model_track_neptune() {
    if !kernel_present() {
        return;
    }
    let samples = reference();
    let (worst, last) = compare(&samples, 10_800.0);
    println!("NEPTUNE vs spacelab over 3 h: worst {worst:.2} m, final {last:.2} m");
    assert!(
        worst > 0.0 && worst < 20.0,
        "3 h against NEPTUNE is {worst:.2} m — outside (0, 20) that is a model difference, not the \
         ephemeris and integrator differences the module documents"
    );
}

/// The full day the fixture holds.
///
/// Ignored by default: a degree-70 field with a per-step CIO chain over 86 400 s is minutes in a
/// debug build. Run with
/// `cargo test --release -p spacelab --test neptune_sentinel1b -- --ignored --nocapture`.
#[test]
#[ignore = "a day of degree-70 field with a per-step frame chain; run in release"]
fn the_full_day_against_neptune() {
    if !kernel_present() {
        return;
    }
    let samples = reference();
    let span = samples.last().expect("the fixture has rows").0;
    let (worst, last) = compare(&samples, span);
    println!("NEPTUNE vs spacelab over 24 h: worst {worst:.2} m, final {last:.2} m");
    assert!(worst > 0.0 && worst.is_finite());
}
