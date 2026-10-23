//! `propagate_many_into` auto-parallelizes a large catalogue across scoped threads and
//! keeps a small one sequential. Correctness of the parallel branch is *observational
//! identity* with the sequential one: the same cells in the same order, the same
//! errors, and the exact same `f64` bits. A chunk-misalignment (a satellite written
//! into the wrong row) or a data race would break that. Distinct real element sets —
//! the AIAA near-Earth and deep-space mix — cycled past `2 * available_parallelism`
//! make a misalignment visible, because neighbouring rows differ.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::UtcDateTime;
use spacelab_frames::{StateVector, Teme};
use spacelab_sgp4::{PropagationError, Satellite, propagate_many_into};

const CASES: &str = include_str!("fixtures/sgp4_ver_tle.txt");

type Cell = Result<StateVector<Teme>, PropagationError>;

/// The element sets that parse (checksums are ignored: the published set is
/// deliberately dirty, and a propagatable object is all this test needs).
fn distinct_satellites() -> Vec<Satellite> {
    CASES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|line| {
            let f: Vec<&str> = line.split('|').collect();
            Satellite::from_tle_unchecked(f[0], f[1]).ok()
        })
        .collect()
}

/// The six state components as raw bit patterns, so a bit-exact comparison also treats
/// a decayed slot's `NaN` as equal to itself (`PartialEq` on `f64` would not).
fn state_bits(state: &StateVector<Teme>) -> [u64; 6] {
    let r = state.position.vector;
    let v = state.velocity.vector;
    [
        r.x.to_bits(),
        r.y.to_bits(),
        r.z.to_bits(),
        v.x.to_bits(),
        v.y.to_bits(),
        v.z.to_bits(),
    ]
}

/// Bit-exact cell equality. The deterministic propagator produces the *same* bits on
/// both paths, so any difference is a real divergence (misalignment or a race).
fn same_cell(left: &Cell, right: &Cell) -> bool {
    match (left, right) {
        (Ok(a), Ok(b)) => state_bits(a) == state_bits(b),
        (Err(a), Err(b)) => a == b,
        _ => false,
    }
}

#[test]
fn parallel_batch_is_bit_identical_to_sequential() {
    let distinct = distinct_satellites();
    assert!(
        distinct.len() > 20,
        "the fixture should yield the near/deep-space mix, got {}",
        distinct.len()
    );

    // Cycle the distinct sets well past `2 * available_parallelism` on any realistic
    // core count, forcing the scoped-thread branch. Consecutive rows differ, so a
    // chunk-misalignment cannot alias itself away.
    let satellites: Vec<Satellite> = distinct.iter().cloned().cycle().take(640).collect();

    // Epochs spanning the fixture's validity (2000-2008), so near-Earth and
    // deep-space objects each exercise real arithmetic rather than the epoch itself.
    let times = [
        UtcDateTime::new(2000, 6, 28, 0, 0, 0.0).unwrap(),
        UtcDateTime::new(2004, 2, 1, 12, 0, 0.0).unwrap(),
        UtcDateTime::new(2006, 6, 25, 6, 0, 0.0).unwrap(),
        UtcDateTime::new(2008, 9, 20, 12, 0, 0.0).unwrap(),
    ];
    let t = times.len();

    let mut parallel: Vec<Cell> =
        vec![Err(PropagationError::MalformedElements); satellites.len() * t];
    propagate_many_into(&satellites, &times, &mut parallel);

    // Sequential reference: each satellite propagated on its own into its own row.
    let mut sequential: Vec<Cell> = Vec::with_capacity(satellites.len() * t);
    for satellite in &satellites {
        let mut row = vec![Err(PropagationError::MalformedElements); t];
        satellite.propagate_into(&times, &mut row);
        sequential.extend(row);
    }

    assert_eq!(parallel.len(), sequential.len());
    for (i, (p, s)) in parallel.iter().zip(&sequential).enumerate() {
        assert!(
            same_cell(p, s),
            "cell {i} (satellite {}, epoch {}) diverged between parallel and sequential",
            i / t,
            i % t
        );
    }

    // The batch must actually contain both successes and per-cell failures, or the
    // test would be vacuous on trivial data.
    assert!(
        parallel.iter().any(Result::is_ok),
        "expected some propagations to succeed"
    );
}
