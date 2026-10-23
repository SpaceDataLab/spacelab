//! What changing the planetary ephemeris from DE440 to DE442 is worth, measured.
//!
//! DE442 (JPL, kernel published 2025-02-06) is the successor to DE440/DE441 (2021). The currency
//! table in `docs/verification.md` said "Stay" on DE440, and a successor existing is a reason to
//! re-read that decision rather than to assume it. This file is the measurement it needs: the same
//! two bodies the force models actually perturb with — the Sun and the Moon — read from both
//! kernels over their common span, and the Moon at J2000 against the JPL Horizons vector that
//! `moon_horizons.rs` already anchors to.
//!
//! Gated on both kernels: `tools/fetch-ephemeris.sh` downloads them and neither is committed, so
//! CI stays network-free and this skips cleanly without them.
//!
//! **The constraint that decides the ticket is not in the numbers below.** NAIF publishes no lunar
//! principal-axis orientation for DE442: `moon_pa_de440_200625.bpc` and `moon_de440_250416.tf` are
//! the only ones in `generic_kernels/pck/` and `generic_kernels/fk/satellites/`, and `gm_de440.tpc`
//! is likewise the latest mass file. Selenocentric work (`lro_arc.rs`, `neptune_lro.rs`,
//! `lunar_orientation.rs`) therefore cannot move to DE442 without pairing DE442 positions with a
//! DE440 orientation, which is a frame mismatch rather than an upgrade.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;
use std::str::FromStr;

use spacelab_ephemeris::frames::{EARTH_J2000, MOON_J2000, SUN_J2000};
use spacelab_ephemeris::{Ephemeris, Epoch, Frame};

const DE440: &str = "../../data/ephemeris/de440s.bsp";
const DE442: &str = "../../data/ephemeris/de442s.bsp";

/// Both kernels, or `None` when either download is absent.
fn kernels() -> Option<(Ephemeris, Ephemeris)> {
    for path in [DE440, DE442] {
        if !Path::new(path).exists() {
            eprintln!("skipping: {path} not present — run tools/fetch-ephemeris.sh");
            return None;
        }
    }
    Some((
        Ephemeris::load(DE440).expect("de440s.bsp loads"),
        Ephemeris::load(DE442).expect("de442s.bsp loads"),
    ))
}

/// Worst and RMS separation, in metres, between the two kernels' answers for `body` about
/// `EARTH_J2000`, sampled every `step_days` over `[from, to]`. Also the worst speed difference in
/// mm/s, since a third-body term differentiates the position it is given.
struct Spread {
    worst_m: f64,
    rms_m: f64,
    worst_mm_s: f64,
    at: String,
    samples: usize,
}

fn spread(
    de440: &Ephemeris,
    de442: &Ephemeris,
    body: Frame,
    from: &str,
    to: &str,
    step_days: f64,
) -> Spread {
    // Stepped in ET seconds off the start rather than by repeated addition, so the sample grid is
    // exact and the last sample is where it says it is.
    let start = Epoch::from_str(from).expect("a TDB epoch").to_et_seconds();
    let end = Epoch::from_str(to).expect("a TDB epoch").to_et_seconds();
    let step = step_days * 86_400.0;

    let (mut worst_m, mut sum_sq, mut worst_mm_s, mut samples) =
        (0.0_f64, 0.0_f64, 0.0_f64, 0usize);
    let mut at = String::new();

    let mut k = 0_u32;
    let mut samples_f = 0.0_f64;
    loop {
        let seconds = step.mul_add(f64::from(k), start);
        if seconds > end {
            break;
        }
        let t = Epoch::from_et_seconds(seconds);
        let (older, newer) = (
            de440
                .state(body, EARTH_J2000, t)
                .expect("inside the DE440 span"),
            de442
                .state(body, EARTH_J2000, t)
                .expect("inside the DE442 span"),
        );
        let separation = ((older.position.x - newer.position.x).powi(2)
            + (older.position.y - newer.position.y).powi(2)
            + (older.position.z - newer.position.z).powi(2))
        .sqrt()
            * 1000.0;
        let speed_gap = ((older.velocity.x - newer.velocity.x).powi(2)
            + (older.velocity.y - newer.velocity.y).powi(2)
            + (older.velocity.z - newer.velocity.z).powi(2))
        .sqrt()
            * 1e6;

        if separation > worst_m {
            worst_m = separation;
            at = format!("{t}");
        }
        worst_mm_s = worst_mm_s.max(speed_gap);
        sum_sq += separation * separation;
        samples += 1;
        samples_f += 1.0;
        k += 1;
    }

    Spread {
        worst_m,
        rms_m: (sum_sq / samples_f).sqrt(),
        worst_mm_s,
        at,
        samples,
    }
}

/// The delta table `docs/verification.md` §2 carries: DE440 against DE442 over the span the force
/// models are used on, for the only two bodies they perturb with.
#[test]
fn de440_and_de442_differ_by_metres_on_the_bodies_the_force_models_use() {
    let Some((de440, de442)) = kernels() else {
        return;
    };

    let mut worst_overall = 0.0_f64;
    for (name, body, step_days) in [
        // The Moon moves fastest and is sampled finest; the Sun's geocentric position changes
        // slowly enough that a daily sample cannot hide a feature in the difference.
        ("Moon", MOON_J2000, 0.25),
        ("Sun", SUN_J2000, 1.0),
    ] {
        let s = spread(
            &de440,
            &de442,
            body,
            "2000-01-01T00:00:00 TDB",
            "2030-01-01T00:00:00 TDB",
            step_days,
        );
        println!(
            "{name:5} DE440-DE442 over 2000..2030 ({} samples, {step_days} d): \
             worst {:.3} m at {}, RMS {:.3} m, worst |Δv| {:.3} mm/s",
            s.samples, s.worst_m, s.at, s.rms_m, s.worst_mm_s
        );
        worst_overall = worst_overall.max(s.worst_m);
    }

    // Not a tolerance on an implementation: a bound on how much two JPL solutions of the same
    // system disagree. It is here so that a kernel swap that moved a body by kilometres — a wrong
    // file at the path, a truncated download that still parses — fails loudly instead of quietly
    // shifting every third-body term.
    assert!(
        worst_overall < 1000.0,
        "DE440 and DE442 differ by {worst_overall:.1} m, which is not two solutions of one system"
    );
    assert!(
        worst_overall > 0.0,
        "DE440 and DE442 are bit-identical, so one of the two paths is reading the wrong file"
    );
}

/// Both kernels against the same external authority, so "closer to Horizons" is a statement about
/// the ephemerides and not about our reader.
#[test]
fn both_kernels_reproduce_the_horizons_moon_at_j2000() {
    let Some((de440, de442)) = kernels() else {
        return;
    };
    let epoch = Epoch::from_str("2000-01-01T12:00:00 TDB").expect("J2000 TDB epoch");

    // JPL Horizons, geocentric ICRF, J2000 — the vector `moon_horizons.rs` anchors to, printed to
    // about 10 m, which is the floor on both numbers below.
    let reference = [-291_608.38_f64, -266_716.83, -76_102.49];

    for (name, kernel) in [("DE440", &de440), ("DE442", &de442)] {
        let moon = kernel
            .state(MOON_J2000, EARTH_J2000, epoch)
            .expect("Moon w.r.t. Earth");
        let err = ((moon.position.x - reference[0]).powi(2)
            + (moon.position.y - reference[1]).powi(2)
            + (moon.position.z - reference[2]).powi(2))
        .sqrt()
            * 1000.0;
        println!("{name} Moon@J2000 vs Horizons: {err:.1} m");
        assert!(err < 1000.0, "{name} Moon@J2000 off Horizons by {err:.1} m");
    }
}
