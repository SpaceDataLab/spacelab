//! The selenocentric force model against a real lunar orbiter: NASA's reconstructed LRO ephemeris.
//!
//! Everything else in `lunar_orientation.rs` checks a piece — the field against a summation, the
//! rotation against a frame kernel, the origin shift against itself. This checks the whole thing at
//! once, in the only way that cannot be satisfied by a consistent mistake: propagate from an LRO
//! state and ask NASA's own reconstruction where the spacecraft actually was.
//!
//! LRO is a hard case on purpose. A ~50 km polar orbit is where the mascons dominate — the field is
//! unstable there on a scale of weeks, which is why lunar missions carry the high-degree solution and
//! not a `J₂` — so this measures the model, not the integrator.
//!
//! Gated on `tools/fetch-ephemeris.sh` and `tools/fetch-lunar-kernels.sh`, and skipping cleanly
//! without them.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;
use std::str::FromStr;

use spacelab_ephemeris::frames::{EARTH_J2000, MOON_J2000, SUN_J2000};
use spacelab_ephemeris::{Ephemeris, Epoch};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{
    BodyOriented, Newtonian, Plain, SolarRadiationPressure, ThirdBody,
};
use spacelab_propagation::{Dop853, Tolerance, lunar};

const SPK: &str = "../../data/ephemeris/de440s.bsp";
const BPC: &str = "../../data/ephemeris/moon_pa_de440_200625.bpc";
const LRO: &str = "../../data/ephemeris/lro_2025349_2026074.bsp";

/// LRO's NAIF body code.
const LRO_ID: i32 = -85;

/// Earth and Sun gravitational parameters, m³/s² — DE440's values.
const MU_EARTH: f64 = 3.986_004_355_070_226e14;
const MU_SUN: f64 = 1.327_124_400_419_393_9e20;

/// LRO's area-to-mass ratio is about 0.0035 m²/kg (≈15 m² of array and bus over ~1000 kg at this
/// point in the mission) and its reflectivity is near 1.3, so `Φ·C_r·A/m ≈ 4.5e-6·1.3·0.0035`. The
/// number is nominal, not fitted: SRP is 2e-8 m/s² against the field's 1.6 m/s², and the point of
/// including it at all is that it is there.
const SRP_REFERENCE: f64 = 4.56e-6 * 1.3 * 0.003_5;
const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

fn kernels_present() -> bool {
    for path in [SPK, BPC, LRO] {
        if !Path::new(path).exists() {
            eprintln!(
                "skipping: {path} not present — run tools/fetch-ephemeris.sh and \
                 tools/fetch-lunar-kernels.sh"
            );
            return false;
        }
    }
    true
}

/// Propagate LRO for `arc_seconds` from `start` and return the position error, metres, against the
/// reconstruction — together with the along-track share of it.
fn arc_error(eph: &Ephemeris, start: Epoch, arc_seconds: f64, degree: usize) -> (f64, f64) {
    let lro = MOON_J2000.with_ephem(LRO_ID);
    let truth = |epoch: Epoch| {
        eph.state(lro, MOON_J2000, epoch)
            .expect("LRO is inside the reconstructed span")
    };

    let initial = truth(start);
    let y0 = [
        initial.position.x * 1000.0,
        initial.position.y * 1000.0,
        initial.position.z * 1000.0,
        initial.velocity.x * 1000.0,
        initial.velocity.y * 1000.0,
        initial.velocity.z * 1000.0,
    ];
    let et = start.to_et_seconds();

    let forces = (
        BodyOriented {
            inner: lunar::embedded(degree).expect("degree is inside the embedded model"),
            rotate: |t: f64| {
                eph.rotation(
                    MOON_J2000,
                    spacelab_ephemeris::frames::MOON_PA_DE440_FRAME,
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
        SolarRadiationPressure {
            reference_acceleration: SRP_REFERENCE,
            astronomical_unit: ASTRONOMICAL_UNIT,
            // The occulting body is the Moon here, not the Earth — the field name is the Earth's
            // because that is what the model was written for, but it is the shadowing radius.
            earth_radius: lunar::MOON_MEAN_RADIUS,
            sun: eph.perturbing_body(SUN_J2000, MOON_J2000, et),
        },
    );

    let trajectory = integrate(
        &Dop853,
        &Plain(Newtonian(forces)),
        0.0,
        &y0,
        arc_seconds,
        10.0,
        Controls::new(Tolerance::new(1e-9, 1e-12), 1e-6),
    )
    .expect("the arc integrates");
    let final_state = trajectory.final_state();

    let want = truth(Epoch::from_et_seconds(et + arc_seconds));
    let error = [
        final_state[0] - want.position.x * 1000.0,
        final_state[1] - want.position.y * 1000.0,
        final_state[2] - want.position.z * 1000.0,
    ];
    let total = (error[0] * error[0] + error[1] * error[1] + error[2] * error[2]).sqrt();

    // The along-track direction is the truth velocity, normalised — a unit vector, so the
    // projection stays in the metres `error` is already in.
    let speed =
        (want.velocity.x.powi(2) + want.velocity.y.powi(2) + want.velocity.z.powi(2)).sqrt();
    let along =
        (error[0] * want.velocity.x + error[1] * want.velocity.y + error[2] * want.velocity.z)
            / speed;
    (total, along.abs())
}

/// The whole selenocentric model, propagated against NASA's reconstruction of LRO.
///
/// The number this prints is the honest one: a degree-40 truncation of a degree-1200 solution, at
/// 50 km altitude, over a fraction of an orbit. It is not a claim of parity with a mission-grade
/// determination — the assertion is a bracket wide enough for a coefficient-set change and tight
/// enough that a wrong frame, a dropped force or a lost origin shift cannot pass it.
#[test]
fn a_reconstructed_lro_arc_is_reproduced_by_the_selenocentric_model() {
    if !kernels_present() {
        return;
    }
    let eph = Ephemeris::load(SPK)
        .expect("de440s.bsp loads")
        .with_kernel(BPC)
        .expect("lunar PA orientation loads")
        .with_kernel(LRO)
        .expect("the LRO reconstruction loads");
    let start = Epoch::from_str("2026-01-15T00:00:00 TDB").expect("literal epoch parses");

    // Measured 4.5 m, 14.2 m and 76.9 m against this reconstruction. The caps are an order of
    // magnitude above that — room for a coefficient set or an integrator tolerance to change, and
    // still far below what any of the failures this test exists to catch would cost: a degree-8
    // field is 1.3 km over the last of these arcs, and a dropped origin shift or a transposed
    // rotation are hundreds of kilometres.
    for (minutes, cap) in [(10.0, 50.0), (30.0, 150.0), (60.0, 800.0)] {
        let (total, along) = arc_error(&eph, start, minutes * 60.0, 40);
        println!(
            "LRO {minutes:>4.0} min: |Δr| = {total:8.1} m, along-track {along:8.1} m \
             ({:.0}%)",
            100.0 * along / total
        );
        assert!(
            total > 0.0 && total < cap,
            "{minutes} min arc is off by {total:.1} m, outside (0, {cap}) — that is a model error, \
             not a truncation"
        );
    }

    // Degree matters here in a way it does not in LEO: the mascons are the reason a lunar mission
    // carries a high-degree field. A degree-8 truncation — the degree an Earth orbit at this
    // altitude would be content with — must be visibly worse over the same arc.
    let (fine, _) = arc_error(&eph, start, 3600.0, 40);
    let (coarse, _) = arc_error(&eph, start, 3600.0, 8);
    println!(
        "one hour: degree 40 {fine:.1} m, degree 8 {coarse:.1} m — ratio {:.1}",
        coarse / fine
    );
    assert!(
        coarse > 3.0 * fine,
        "degree 8 is only {:.1}x worse than degree 40; the field is not being evaluated",
        coarse / fine
    );
}
