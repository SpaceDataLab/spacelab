//! The official SGP4 verification cases, AIAA 2006-6753.
//!
//! Thirty-three element sets chosen to hit the branches a healthy LEO satellite
//! never touches: deep-space secular resonance, near-critical inclination, the
//! Lyddane singularity, backward propagation, decaying objects, and elements
//! deliberately outside the model's domain.
//!
//! The reference values come from `python-sgp4` (MIT) — and, because that package
//! ships a compiled extension and dispatches to it when present, from Vallado's own
//! `SGP4.cpp`, the companion code to AIAA 2006-6753. The Rust `sgp4` crate this
//! library wraps is a *port* of that same source.
//!
//! So this is a port-versus-source check. It establishes that the transcription is
//! faithful, to the millimetre, over every branch the published set exercises. It
//! does **not** establish independent confirmation, and no test can: every SGP4
//! implementation in circulation descends from the same listing. For SGP4 the
//! specification *is* a particular listing, and its every quirk is part of it.
//!
//! The complementary check is `sgp4_primary_source.rs`, which compares against the
//! numbers Vallado *printed* in 2006 rather than the ones his code produces today.
//!
//! Fixtures regenerate with `derivations/sgp4_verification.py`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::{Convention, Satellite};

const CASES: &str = include_str!("fixtures/sgp4_ver_tle.txt");
const STATES: &str = include_str!("fixtures/sgp4_ver_states.csv");

/// Position agreement, metres.
///
/// Two implementations of the same chain of trigonometric series, in different
/// languages, with different summation orders. Vallado reports cross-implementation
/// agreement below `1e-8` km for near-Earth cases; the deep-space branch, with its
/// resonance integration, accumulates more. This bound is a millimetre, and the worst
/// case observed is far below it — the number is printed, so a regression would be
/// visible before it became a failure.
const TOL_POSITION_M: f64 = 1e-3;

/// Velocity agreement, metres per second.
const TOL_VELOCITY_MS: f64 = 1e-6;

struct Case {
    norad: u32,
    satellite: Option<Satellite>,
}

fn cases() -> Vec<Case> {
    CASES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split('|').collect();
            // The published set deliberately includes malformed and bad-checksum
            // element sets; load them anyway so the propagator itself is tested.
            let satellite = Satellite::from_tle_unchecked(f[0], f[1]).ok();
            let norad = f[1][2..7].trim().parse().expect("a catalog number");
            Case { norad, satellite }
        })
        .collect()
}

#[test]
fn every_verification_element_set_is_accepted_or_explicitly_refused() {
    let all = cases();
    assert_eq!(all.len(), 33, "the published set has 33 element sets");

    let rejected: Vec<u32> = all
        .iter()
        .filter(|c| c.satellite.is_none())
        .map(|c| c.norad)
        .collect();

    // Refusing an element set at construction is a legitimate answer for the cases
    // designed to be outside the model's domain. Silently accepting one and then
    // producing a plausible orbit would not be.
    println!(
        "{} of 33 element sets refused at construction: {rejected:?}",
        rejected.len()
    );
    assert!(
        rejected.len() <= 2,
        "refused too many: {rejected:?} — the crate is rejecting valid elements"
    );
}

#[test]
fn we_agree_with_the_reference_implementation_everywhere_it_succeeds() {
    let by_norad: std::collections::HashMap<u32, &Satellite> = {
        let leaked: &'static [Case] = Box::leak(cases().into_boxed_slice());
        leaked
            .iter()
            .filter_map(|c| c.satellite.as_ref().map(|s| (c.norad, s)))
            .collect()
    };

    let (mut agreed, mut both_failed, mut we_failed, mut they_failed) = (0, 0, 0, 0);
    let (mut worst_r, mut worst_v, mut worst_at) = (0.0_f64, 0.0_f64, (0_u32, 0.0_f64));

    for line in STATES.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let norad: u32 = f[0].parse().unwrap();
        let minutes: f64 = f[1].parse().unwrap();
        let reference_error: i32 = f[2].parse().unwrap();

        let Some(satellite) = by_norad.get(&norad) else {
            continue; // refused at construction; covered by the test above
        };

        match (satellite.propagate_minutes(minutes), reference_error) {
            // Both succeeded: the numbers must match.
            (Ok(state), 0) => {
                let r: [f64; 3] = [
                    f[3].parse().unwrap(),
                    f[4].parse().unwrap(),
                    f[5].parse().unwrap(),
                ];
                let v: [f64; 3] = [
                    f[6].parse().unwrap(),
                    f[7].parse().unwrap(),
                    f[8].parse().unwrap(),
                ];

                let dr = ((state.position.vector.x - r[0] * 1000.0).powi(2)
                    + (state.position.vector.y - r[1] * 1000.0).powi(2)
                    + (state.position.vector.z - r[2] * 1000.0).powi(2))
                .sqrt();
                let dv = ((state.velocity.vector.x - v[0] * 1000.0).powi(2)
                    + (state.velocity.vector.y - v[1] * 1000.0).powi(2)
                    + (state.velocity.vector.z - v[2] * 1000.0).powi(2))
                .sqrt();

                if dr > worst_r {
                    worst_r = dr;
                    worst_at = (norad, minutes);
                }
                worst_v = worst_v.max(dv);

                assert!(
                    dr < TOL_POSITION_M,
                    "sat {norad} at {minutes} min: position off by {dr:.3e} m"
                );
                assert!(
                    dv < TOL_VELOCITY_MS,
                    "sat {norad} at {minutes} min: velocity off by {dv:.3e} m/s"
                );
                agreed += 1;
            }
            // We declined where the reference succeeded. Suspicious; count it.
            (Err(_), 0) => we_failed += 1,
            // Both refused. The reference's error codes are finer than ours; we
            // require only that we also decline to invent an answer.
            (Err(_), _) => both_failed += 1,
            // We answered where the reference declined. Worse: we invented an orbit.
            (Ok(_), _) => they_failed += 1,
        }
    }

    println!(
        "sgp4-ver: {agreed} states agree (worst {worst_r:.2e} m at sat {} t={} min, {worst_v:.2e} m/s); \
         {both_failed} both refused; {we_failed} we refused alone; {they_failed} we answered alone",
        worst_at.0, worst_at.1
    );

    assert!(agreed > 700, "too few agreeing states: {agreed}");
    assert_eq!(
        they_failed, 0,
        "we produced an orbit where the reference refused to"
    );
    assert_eq!(we_failed, 0, "we refused where the reference succeeded");
}

/// The IAU/WGS-84 convention is available, and it is *different*. If this ever stops
/// being true, the AFSPC default has silently stopped meaning anything.
#[test]
fn the_two_conventions_do_not_agree() {
    let line = CASES
        .lines()
        .find(|l| l.starts_with("1 00005"))
        .expect("the first verification case");
    let f: Vec<&str> = line.split('|').collect();

    let elements = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()).unwrap();
    let reference = Satellite::from_elements(&elements, Convention::Reference).unwrap();
    let iau = Satellite::from_elements(&elements, Convention::Iau).unwrap();

    let a = reference.propagate_minutes(1440.0).unwrap().position.vector;
    let i = iau.propagate_minutes(1440.0).unwrap().position.vector;
    let separation = (a - i).norm();

    assert!(
        separation > 1.0,
        "the conventions agree to {separation:.3} m; the default no longer distinguishes them"
    );
    println!("conventions differ by {separation:.1} m after 24 h on sat 00005");
}
