//! The error, convention and batch surface the verification harness does not reach.
//!
//! `sgp4_verification.rs` drives the propagator hard on the published element sets, but
//! it always loads with `from_tle_unchecked` and only compares the two conventions it
//! expects to match. This file exercises the rest of the public contract on real orbits:
//! the checksum gate, the error messages, the accessors, the AFSPC node reduction, and the
//! batch entry points — including their misuse guards. A library headed for real operations
//! must fail exactly where it says it does.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::{NoradId, UtcDateTime};
use spacelab_sgp4::{Convention, PropagationError, Satellite, propagate_many_into};
use spacelab_testkit::nan_max;

/// ISS, epoch 2008-264 — the `sgp4` crate's documented (Vallado) element set, with a
/// valid checksum on both lines.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// The published verification element sets (`line1|line2`), reused for the deep-space
/// node-reduction check.
const CASES: &str = include_str!("fixtures/sgp4_ver_tle.txt");

fn utc(hour: u8, minute: u8) -> UtcDateTime {
    UtcDateTime::new(2008, 9, 20, hour, minute, 0.0).expect("a valid UTC label")
}

#[test]
fn from_tle_rejects_a_corrupted_checksum_but_unchecked_repairs_it() {
    // The genuine element set is accepted by the checksum-validating constructor.
    assert!(Satellite::from_tle(LINE1, LINE2).is_ok());

    // Flipping the column-69 checksum digit must be caught, not propagated.
    let mut corrupted = LINE1.to_owned();
    let last = corrupted.pop().unwrap();
    corrupted.push(if last == '7' { '6' } else { '7' });
    assert_eq!(
        Satellite::from_tle(&corrupted, LINE2).unwrap_err(),
        PropagationError::BadChecksum,
    );

    // The caller who opts out of the check gets the repaired line and a usable satellite.
    assert!(Satellite::from_tle_unchecked(&corrupted, LINE2).is_ok());
}

#[test]
fn short_and_garbage_lines_are_refused_not_crashed() {
    // A line shorter than 69 characters fails the checksum test cleanly (no slice panic).
    assert_eq!(
        Satellite::from_tle("too short", LINE2).unwrap_err(),
        PropagationError::BadChecksum,
    );
    // Unchecked, a short line is returned unrepaired and the parser then rejects it.
    assert_eq!(
        Satellite::from_tle_unchecked("too short", "also short").unwrap_err(),
        PropagationError::MalformedElements,
    );
}

#[test]
fn every_error_prints_a_distinct_non_empty_message() {
    let all = [
        PropagationError::MalformedElements,
        PropagationError::BadChecksum,
        PropagationError::UnusableElements,
        PropagationError::Diverged,
        PropagationError::Decayed,
    ];
    let messages: Vec<String> = all.iter().map(ToString::to_string).collect();
    assert!(messages.iter().all(|m| !m.is_empty()));
    let distinct: std::collections::HashSet<&String> = messages.iter().collect();
    assert_eq!(
        distinct.len(),
        all.len(),
        "two errors share a message: {messages:?}",
    );
}

#[test]
fn accessors_report_the_identity_and_convention() {
    let sat = Satellite::from_tle(LINE1, LINE2).unwrap();
    assert_eq!(sat.norad_id(), NoradId::new(25544));
    assert_eq!(sat.convention(), Convention::Reference);

    // A non-default convention is carried and reported back.
    let elements = sgp4::Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).unwrap();
    let iau = Satellite::from_elements(&elements, Convention::Iau).unwrap();
    assert_eq!(iau.convention(), Convention::Iau);
}

#[test]
fn the_epoch_is_the_zero_of_the_propagation_parameter() {
    let sat = Satellite::from_tle(LINE1, LINE2).unwrap();

    // Columns 19-32 of line 1: 08264.51782528. Decoding is exact in nanoseconds, so the
    // round trip through the accessor must be too.
    let epoch = sat.epoch();
    assert_eq!(
        (
            epoch.year(),
            epoch.month(),
            epoch.day(),
            epoch.hour(),
            epoch.minute()
        ),
        (2008, 9, 20, 12, 25),
    );
    // 0.51782528 d = 44740.104192 s = 12:25:40.104192, exactly.
    assert!((epoch.second() - 40.104_192).abs() < 1e-6, "{epoch}");
    // Exactly +0.0, not merely small: the difference is taken in integer nanoseconds.
    assert_eq!(sat.minutes_since_epoch(epoch).to_bits(), 0_u64);

    // And it is the zero the propagation parameter is measured from, not merely a label.
    assert_eq!(
        sat.propagate_at(epoch).unwrap().position,
        sat.propagate_minutes(0.0).unwrap().position,
    );
}

/// `AfspcNodeReduction` must actually move something, or it is a switch that does nothing.
///
/// The vendor crate's two propagation paths differ in one expression — the node reduced by
/// `rem_euclid` here and by a truncating `%` in [`Convention::Reference`] — reachable only inside
/// the low-inclination Lyddane arm of the deep-space correction, and only where that angle is
/// negative. At least one published verification orbit must land there, or the branch has silently
/// become dead code. Nothing here claims a *mechanism* beyond that one expression, because none is
/// established.
#[test]
fn the_afspc_node_reduction_moves_at_least_one_verification_orbit() {
    let mut max_separation = 0.0_f64;
    for line in CASES.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('|').collect();
        let Ok(elements) = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()) else {
            continue;
        };
        let (Ok(reference), Ok(afspc)) = (
            Satellite::from_elements(&elements, Convention::Reference),
            Satellite::from_elements(&elements, Convention::AfspcNodeReduction),
        ) else {
            continue;
        };
        for &minutes in &[420.0, 720.0, 1440.0, 2880.0] {
            if let (Ok(a), Ok(b)) = (
                reference.propagate_minutes(minutes),
                afspc.propagate_minutes(minutes),
            ) {
                max_separation = nan_max(
                    max_separation,
                    (a.position.vector - b.position.vector).norm(),
                );
            }
        }
    }
    assert!(
        max_separation > 1.0,
        "AfspcNodeReduction never diverges from Reference ({max_separation:.3} m): it is dead code",
    );
    println!(
        "AfspcNodeReduction vs Reference: max separation {max_separation:.1} m \
         across the verification set"
    );
}

#[test]
fn a_single_satellite_batch_takes_the_sequential_path() {
    // One satellite is below the parallel threshold, so this exercises the sequential
    // fallback in `propagate_many_into` (and `propagate_at`/`propagate_into` beneath it).
    let sat = Satellite::from_tle(LINE1, LINE2).unwrap();
    let times = [utc(12, 0), utc(13, 30), utc(15, 0)];
    let mut out = vec![Err(PropagationError::Diverged); times.len()];
    propagate_many_into(std::slice::from_ref(&sat), &times, &mut out);
    assert!(
        out.iter().all(Result::is_ok),
        "the ISS is stable over three hours: {out:?}",
    );
}

#[test]
#[should_panic(expected = "output slice")]
fn propagate_into_panics_when_the_output_is_too_short() {
    let sat = Satellite::from_tle(LINE1, LINE2).unwrap();
    let times = [utc(12, 0), utc(13, 0)];
    let mut out = vec![Err(PropagationError::Diverged); 1]; // one slot for two epochs
    sat.propagate_into(&times, &mut out);
}

#[test]
#[should_panic(expected = "cells")]
fn propagate_many_into_panics_when_the_output_is_too_short() {
    let sat = Satellite::from_tle(LINE1, LINE2).unwrap();
    let times = [utc(12, 0), utc(13, 0)];
    let mut out = vec![Err(PropagationError::Diverged); 1]; // needs one satellite × two epochs
    propagate_many_into(std::slice::from_ref(&sat), &times, &mut out);
}

#[test]
fn either_empty_side_of_the_batch_takes_the_sequential_path() {
    // Both degenerate shapes are refused by the parallel-dispatch guard rather than handled by
    // the parallel path, and in both cases the guard is load-bearing rather than a tuning
    // choice: the parallel path slices by `per * t`, and `chunks(0)` panics. An empty epoch list
    // makes `t` zero; an empty catalogue makes `per` zero. The catalogue is sized off this host's
    // thread count rather than off a literal, so that it is the epoch list and not the batch size
    // that sends the first call down the sequential path on any machine, not merely on a small one.
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    let satellites = vec![Satellite::from_tle(LINE1, LINE2).unwrap(); 2 * threads + 1];
    let mut out = vec![Err(PropagationError::Diverged); 4];
    propagate_many_into(&satellites, &[], &mut out);
    assert!(
        out.iter().all(Result::is_err),
        "no epoch was asked for, so nothing may be written: {out:?}",
    );

    propagate_many_into(&[], &[utc(12, 0), utc(13, 0)], &mut out);
    assert!(
        out.iter().all(Result::is_err),
        "no satellite was given, so nothing may be written: {out:?}",
    );
}
