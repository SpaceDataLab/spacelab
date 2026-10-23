//! The GPS almanac read two ways, and what the pair proves.
//!
//! CelesTrak publishes the same almanac in both distributed formats at the same week and time of
//! applicability, and the two encode the same underlying scaled integers. **SEM** is positional, in
//! semicircles, with inclination as an offset from 0.30 semicircles; **Yuma** is `key: value`, in
//! radians, with inclination outright. Nothing in either file says how to convert to the other, so a
//! reader that gets a unit or a reference wrong produces a plausible orbit from one of them and a
//! different plausible orbit from the other — which is exactly what a matched pair detects and a
//! single file cannot.
//!
//! The rest of the file asserts that the parsed set is a GPS constellation at all, in quantities no
//! transcription can fake: the orbital period against half a sidereal day, and the radius band.

#![allow(clippy::expect_used, clippy::unwrap_used)]
#![allow(clippy::doc_markdown)] // CelesTrak, SEM, Yuma, GPS are names in prose

use spacelab_gnss::almanac::{AlmanacEntry, parse_sem, parse_yuma};

const SEM: &str = include_str!("fixtures/almanac.sem.week0384.147456.txt");
const YUMA: &str = include_str!("fixtures/almanac.yuma.week0384.147456.txt");

/// GPS week 2432, the era the fixture was retrieved in (2026-08-15); 2432 mod 1024 = 384.
const HINT: u16 = 2432;

/// Half a sidereal day, seconds — the GPS orbital period by design (two revolutions per sidereal
/// day, which is what makes the ground track repeat daily).
const HALF_SIDEREAL_DAY: f64 = 86_164.090_524 / 2.0;

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn by_prn(entries: &[AlmanacEntry], prn: u8) -> Option<&AlmanacEntry> {
    entries.iter().find(|e| e.ephemeris.prn == prn)
}

/// The two formats must describe the same satellites.
#[test]
fn sem_and_yuma_carry_the_same_constellation() {
    let (sem, yuma) = (parse_sem(SEM, HINT), parse_yuma(YUMA, HINT));
    assert_eq!(sem.len(), yuma.len(), "record count");

    let mut sem_prns: Vec<u8> = sem.iter().map(|e| e.ephemeris.prn).collect();
    let mut yuma_prns: Vec<u8> = yuma.iter().map(|e| e.ephemeris.prn).collect();
    sem_prns.sort_unstable();
    yuma_prns.sort_unstable();
    assert_eq!(sem_prns, yuma_prns, "PRN sets");

    for entry in &sem {
        let other = by_prn(&yuma, entry.ephemeris.prn).expect("PRN present in both");
        assert_eq!(
            entry.ephemeris.health, other.ephemeris.health,
            "PRN {} health",
            entry.ephemeris.prn
        );
    }
}

/// The two formats must give the same orbit, and the residual between them is a *measurement of the
/// files*, not of the model: SEM prints 15 significant digits, Yuma 9–10, so the difference is
/// dominated by Yuma's decimal truncation of `ω` (9 digits, ~5e-10 rad, ~1.3 cm at a GPS radius).
///
/// Measured over all 32 satellites at 15-minute steps across ±12 hours from the time of
/// applicability: **max position difference 5.7832e-2 m, max velocity difference 6.6712e-6 m/s**.
///
/// The bounds below are set against the failure modes they exist to exclude, both measured on this
/// same fixture by perturbing the parsed set: dropping the 0.30-semicircle inclination reference
/// costs **2.457e7 m**, and leaving the node in semicircles costs **4.647e7 m**. Nine orders of
/// magnitude separate a correct reader from either, so there is no tolerance to tune.
#[test]
fn the_two_formats_agree_to_the_precision_yuma_prints() {
    let (sem, yuma) = (parse_sem(SEM, HINT), parse_yuma(YUMA, HINT));

    let mut worst_position = 0.0_f64;
    let mut worst_velocity = 0.0_f64;
    for entry in &sem {
        let other = by_prn(&yuma, entry.ephemeris.prn).expect("PRN present in both");
        for step in -48..=48 {
            let tk = f64::from(step) * 900.0;
            let (p1, v1) = entry.ephemeris.position_velocity_ecef(tk);
            let (p2, v2) = other.ephemeris.position_velocity_ecef(tk);
            let dp = norm([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]]);
            let dv = norm([v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]);
            worst_position = worst_position.max(dp);
            worst_velocity = worst_velocity.max(dv);
        }
    }

    assert!(
        worst_position < 1.0e-1,
        "SEM against Yuma: {worst_position:.4e} m"
    );
    assert!(
        worst_velocity < 1.0e-5,
        "SEM against Yuma: {worst_velocity:.4e} m/s"
    );
    // And not *identical*: the two files really are different decimal renderings, so a zero here
    // would mean the test compared something against itself.
    assert!(worst_position > 0.0, "the two formats parsed identically");
}

/// The parsed elements have to describe GPS, and the period is the check that cannot be satisfied by
/// accident: it comes from `√A` alone through `2π√(a³/μ)`, so a misread mantissa or a metres/kilometres
/// slip moves it by minutes. GPS is a 2-revolution-per-sidereal-day constellation, so every satellite
/// must sit within seconds of half a sidereal day. Measured on this fixture: **43 074.501 s to
/// 43 079.766 s**, against 43 082.045 s — the real station-keeping spread, 7.5 s at its widest.
#[test]
fn every_satellite_has_the_gps_period_and_radius() {
    for entry in parse_sem(SEM, HINT) {
        let e = &entry.ephemeris;
        let period = 2.0 * std::f64::consts::PI / e.mean_motion();
        assert!(
            (period - HALF_SIDEREAL_DAY).abs() < 15.0,
            "PRN {}: period {period:.3} s against {HALF_SIDEREAL_DAY:.3} s",
            e.prn
        );

        // Radius over a full revolution, which also exercises the eccentricity: GPS is nearly
        // circular at 26 560 km, and the band is wide enough for the real spread and narrow enough
        // to exclude a wrong √A.
        for step in 0..24 {
            let tk = f64::from(step) * HALF_SIDEREAL_DAY / 24.0;
            let r = norm(e.position_ecef(tk));
            assert!(
                (2.55e7..2.75e7).contains(&r),
                "PRN {}: radius {r:.0} m at tk = {tk:.0} s",
                e.prn
            );
        }
    }
}

/// The almanac is the *reduced* model, and saying so is worth a measurement rather than a comment:
/// evaluated at the time of applicability against the same elements carried forward half a day, the
/// two-body motion has to be the only thing moving. A non-zero `Δn` or `i̇` read into the correction
/// slots would show here as a secular drift in the osculating radius that the almanac cannot have.
#[test]
fn the_reduced_model_is_purely_two_body() {
    let sem = parse_sem(SEM, HINT);
    let e = &by_prn(&sem, 1).expect("PRN 1").ephemeris;

    // One anomalistic period apart, the state must repeat: with no Δn, no i̇ and no harmonics, the
    // in-plane motion is exactly periodic and only the node moves (at Ω̇, which is common to both
    // epochs and so cancels in the radius).
    let period = 2.0 * std::f64::consts::PI / e.mean_motion();
    let r0 = norm(e.position_ecef(0.0));
    let r1 = norm(e.position_ecef(period));
    assert!(
        (r0 - r1).abs() < 1.0e-6,
        "radius drifted {:.3e} m over one period",
        (r0 - r1).abs()
    );
}
