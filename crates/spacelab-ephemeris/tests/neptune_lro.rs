//! The selenocentric force model against NEPTUNE, a fourth independent propagator.
//!
//! `lro_arc.rs` measures this model against NASA's reconstruction of LRO, which answers "is it
//! right" but not "is it right for the reasons we think". This answers the second question: NEPTUNE
//! (Space-Systems/neptune, TU Braunschweig ILR/IRAS, Apache-2.0, Fortran-2008) is a propagator of a
//! lineage that shares nothing with Orekit, SPICE or this crate, and it is given the *same* initial
//! state, the *same* gravity field and the *same* set of forces.
//!
//! Making the force models identical is the part that decides whether the residual means anything.
//! What was done, and what it cost:
//!
//!   * NEPTUNE ships AIUB-GRL350A and spacelab embeds GRGM1200A — two different GRAIL solutions, so
//!     a raw residual would be measuring the solutions. `derivations/gen_lunar_gravity.py --icgem`
//!     emits *our* coefficients in ICGEM form and `tools/neptune-oracle/run.sh -d` drops them in
//!     under the filename NEPTUNE fixes in its source. Its reader takes GM and the reference radius
//!     from that header, so both sides then agree on the field, on GM and on Re.
//!   * Drag, solar radiation pressure, Earth albedo and both tide models are off on the NEPTUNE
//!     side, and absent here. What remains is the lunar field plus Earth and Sun as point masses.
//!
//! Three differences remain, all named rather than absorbed:
//!
//!   * **Body-fixed frame.** libslam rotates into the Moon's body-fixed frame with the IAU 2015
//!     analytical series (Archinal et al. 2018); this crate uses the DE440 principal axes from a
//!     frame kernel. The IAU series approximates the mean-Earth frame, and PA against ME is 875 m
//!     on the lunar surface — so a field defined in principal axes is being evaluated by NEPTUNE at
//!     a position expressed in something else. That is a statement about NEPTUNE, and it is the
//!     floor under everything below.
//!   * **Central body.** NEPTUNE integrates in GCRF with the Moon as a perturbing body carrying
//!     harmonics; this crate integrates selenocentrically. Same physics, different cancellation: a
//!     1e-10 relative tolerance on a 384 000 km Earth-centred state is centimetres of truncation
//!     per step that a 1800 km selenocentric state never pays.
//!   * **Planetary ephemeris.** DE421 there, DE440 here, and GCRF against J2000 — a 23 mas frame
//!     bias, 0.2 m at this orbit radius.
//!
//! Gated on `tools/fetch-ephemeris.sh` and `tools/fetch-lunar-kernels.sh`, and skipping cleanly
//! without them. The fixture is committed, so nothing here builds or runs Fortran.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;

use spacelab_ephemeris::frames::{EARTH_J2000, MOON_J2000, MOON_PA_DE440_FRAME, SUN_J2000};
use spacelab_ephemeris::{Ephemeris, Epoch};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{BodyOriented, Newtonian, Plain, ThirdBody};
use spacelab_propagation::{Dop853, Tolerance, lunar};

const SPK: &str = "../../data/ephemeris/de440s.bsp";
const BPC: &str = "../../data/ephemeris/moon_pa_de440_200625.bpc";
const FIXTURE: &str = "tests/fixtures/neptune_lro_matched.txt";

/// Earth and Sun gravitational parameters, m³/s² — DE440's values.
const MU_EARTH: f64 = 3.986_004_355_070_226e14;
const MU_SUN: f64 = 1.327_124_400_419_393_9e20;

/// The degree the fixture was generated at, and the degree the comparison must run at.
const DEGREE: usize = 40;

struct Reference {
    start_et: f64,
    /// `(elapsed_seconds, [x, y, z, vx, vy, vz])` in metres and metres per second.
    samples: Vec<(f64, [f64; 6])>,
}

fn read_reference() -> Reference {
    let text = std::fs::read_to_string(FIXTURE).expect("the NEPTUNE fixture is committed");
    let mut start_et = None;
    let mut samples = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# start_et_seconds ") {
            start_et = Some(rest.trim().parse::<f64>().expect("the header epoch parses"));
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let field: Vec<f64> = line
            .split_whitespace()
            .map(|token| token.parse::<f64>().expect("the fixture is numeric"))
            .collect();
        assert_eq!(field.len(), 7, "expected 7 columns, got {}", field.len());
        // The fixture is kilometres, as NEPTUNE writes it; the propagator is metres throughout.
        let mut state = [0.0; 6];
        for (slot, value) in state.iter_mut().zip(&field[1..]) {
            *slot = value * 1000.0;
        }
        samples.push((field[0], state));
    }
    Reference {
        start_et: start_et.expect("the fixture names its start epoch"),
        samples,
    }
}

fn kernels_present() -> bool {
    for path in [SPK, BPC] {
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present — run tools/fetch-ephemeris.sh");
            return false;
        }
    }
    true
}

/// Propagate the fixture's initial state for `span_seconds` and report the worst position
/// difference against NEPTUNE, together with the difference at the end of the span.
fn compare(reference: &Reference, span_seconds: f64) -> (f64, f64) {
    let eph = Ephemeris::load(SPK)
        .expect("de440s.bsp loads")
        .with_kernel(BPC)
        .expect("lunar PA orientation loads");
    let et = reference.start_et;
    let y0 = reference.samples[0].1;

    let forces = (
        BodyOriented {
            inner: lunar::embedded(DEGREE).expect("degree is inside the embedded model"),
            rotate: |t: f64| {
                eph.rotation(
                    MOON_J2000,
                    MOON_PA_DE440_FRAME,
                    Epoch::from_et_seconds(et + t),
                )
                .expect("the BPC covers the arc")
            },
        },
        (
            ThirdBody {
                mu: MU_EARTH,
                ephemeris: eph.perturbing_body(EARTH_J2000, MOON_J2000, et),
            },
            ThirdBody {
                mu: MU_SUN,
                ephemeris: eph.perturbing_body(SUN_J2000, MOON_J2000, et),
            },
        ),
    );

    let trajectory = integrate(
        &Dop853,
        &Plain(Newtonian(forces)),
        0.0,
        &y0,
        span_seconds,
        10.0,
        Controls::new(Tolerance::new(1e-9, 1e-12), 1e-6),
    )
    .expect("the arc integrates");

    let mut worst: f64 = 0.0;
    let mut last = 0.0;
    // Report every six hours. The elapsed column is a difference of two ET values around 8.1e8 s,
    // so it lands a microsecond off the round number and an equality test on it never fires.
    let mut next_report = 0.0;
    for &(elapsed, want) in &reference.samples {
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
            println!("  +{:>6.1} h  |Δr| = {difference:9.1} m", elapsed / 3600.0);
            next_report += 21_600.0;
        }
    }
    (worst, last)
}

/// A day of the same arc, on the same field, against a propagator that shares no line of code.
///
/// The cap is what a frame difference of the size named in the module docs buys over a day, and no
/// more: the two implementations are not expected to agree to the metre while one evaluates a
/// principal-axis field at a mean-Earth position. What it does catch is the whole class of failure
/// the reconstruction test cannot separate from a truncation — a transposed rotation, a lost origin
/// shift, a third body with the wrong sign — each of which is orders above this.
#[test]
fn a_day_of_the_selenocentric_model_tracks_neptune() {
    if !kernels_present() {
        return;
    }
    let reference = read_reference();
    let (worst, last) = compare(&reference, 86_400.0);
    println!("NEPTUNE vs spacelab over 24 h: worst {worst:.1} m, final {last:.1} m");
    assert!(
        worst > 0.0 && worst < 2_000.0,
        "24 h against NEPTUNE is {worst:.1} m — outside (0, 2000) that is a model difference, not \
         the frame and ephemeris differences the module documents"
    );
}

/// The full seven-day span the fixture holds.
///
/// Ignored by default because a degree-40 field over 89 orbits is minutes, not seconds, in a debug
/// build. Run with
/// `cargo test --release -p spacelab-ephemeris --test neptune_lro -- --ignored --nocapture`.
#[test]
#[ignore = "seven days of degree-40 lunar field; run in release"]
fn the_full_week_against_neptune() {
    if !kernels_present() {
        return;
    }
    let reference = read_reference();
    let span = reference.samples.last().expect("the fixture has rows").0;
    let (worst, last) = compare(&reference, span);
    println!(
        "NEPTUNE vs spacelab over {:.1} days: worst {worst:.1} m, final {last:.1} m",
        span / 86_400.0
    );
    assert!(worst > 0.0 && worst.is_finite());
}
