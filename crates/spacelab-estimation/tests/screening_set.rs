//! The workspace's conjunction screen, scored on the synthetic set with an exact answer key
//! (plan 18 Ф21.O).
//!
//! `spacelab_testkit::screening_set` holds 100 circular orbits over one day — a Walker shell, planted
//! crossing and coplanar pairs, a background — and every local minimum of every pair's separation
//! below its 10 km threshold, each in closed form and re-derived in
//! `derivations/screening_set.mac` (G3). The screen scored here is the two stages of
//! `src/screening.rs`, as a caller strings them together: the radial filter drops a pair whose
//! shells are further apart than the threshold, `closest_approaches` scans each pair left, and an
//! approach below the threshold is reported. A report recalls a keyed event within 1 ms and 1 mm.

#![expect(
    clippy::expect_used,
    reason = "a test of the committed set fails loudly"
)]

use spacelab_estimation::{closest_approaches, radially_separated};
use spacelab_testkit::screening_set::{
    CircularOrbit, Group, Minimum, PairKind, ScreeningSet, Tolerance, screening_set,
};

const TOLERANCE: Tolerance = Tolerance {
    tca: 1e-3,
    distance: 1e-3,
};

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn relative(orbits: &[CircularOrbit], a: usize, b: usize) -> impl Fn(f64) -> ([f64; 3], [f64; 3]) {
    let (oa, ob) = (orbits[a], orbits[b]);
    move |t| {
        let ((ra, va), (rb, vb)) = (oa.state(t), ob.state(t));
        (sub(rb, ra), sub(vb, va))
    }
}

/// What one run of the screen did besides report.
struct Run {
    reports: Vec<Minimum>,
    scanned: usize,
    approaches: usize,
    /// Approaches emitted on pairs in one plane at one radius, whose separation has no minimum.
    on_comoving: usize,
}

/// The radial filter, then a scan at `step` refined to `tolerance`, over all 4 950 pairs.
fn screen(set: &ScreeningSet, step: f64, tolerance: f64) -> Run {
    let orbits: Vec<CircularOrbit> = set.objects.iter().map(|o| o.orbit(set.mu)).collect();
    let mut run = Run {
        reports: Vec::new(),
        scanned: 0,
        approaches: 0,
        on_comoving: 0,
    };
    let n = set.objects.len();
    for a in 0..n {
        for b in a + 1..n {
            let (ra, rb) = (set.objects[a].radius, set.objects[b].radius);
            if radially_separated(ra, ra, rb, rb, set.threshold) {
                continue;
            }
            run.scanned += 1;
            let (oa, ob) = (set.objects[a], set.objects[b]);
            let comoving =
                (oa.radius, oa.inclination, oa.raan) == (ob.radius, ob.inclination, ob.raan);
            closest_approaches(
                relative(&orbits, a, b),
                set.start,
                set.end,
                step,
                tolerance,
                &mut |c| {
                    run.approaches += 1;
                    run.on_comoving += usize::from(comoving);
                    if c.distance < set.threshold {
                        run.reports.push(Minimum {
                            a,
                            b,
                            tca: c.tca,
                            distance: c.distance,
                        });
                    }
                },
            )
            .expect("a well-formed scan");
        }
    }
    run
}

/// The screen at a 60 s step, inside its documented condition — shorter than the quarter
/// revolution between a crossing pair's extrema — recalls every keyed event and raises no false
/// alarm (G3: the key is the closed form of `derivations/screening_set.mac`).
///
/// The radial filter keeps exactly the 795 pairs within 10 km of each other in radius: the two
/// coplanar hard negatives, 10 001 m and 15 km apart, are dropped by it, and none of the hard
/// negatives the scan does see is reported below the threshold — the nearest is 1 cm beyond it.
///
/// One thing the score does not show. On the 140 shell pairs that share a plane and a radius,
/// `Δr·Δv` is zero in exact arithmetic and rounding noise in `f64`, and the scan emits an approach at
/// each of its sign changes: tens of thousands in the day, every one at the pair's constant 5 302 km
/// or more, so none is reported. A comoving pair inside the threshold would be reported at random
/// epochs. The set keys no such pair, because its separation has no minimum to key.
#[test]
fn the_screen_recalls_every_keyed_event_without_a_false_alarm() {
    let set = screening_set().expect("the committed set parses");
    let run = screen(&set, 60.0, 1e-6);
    let score = set.score(&run.reports, TOLERANCE);
    eprintln!(
        "60 s step: {} pairs scanned, {} approaches ({} on comoving pairs), {} reported; recalled \
         {} of {}, {} false alarms; worst {:.3e} s and {:.3e} m",
        run.scanned,
        run.approaches,
        run.on_comoving,
        run.reports.len(),
        score.recalled(),
        score.keyed,
        score.false_alarms.len(),
        score.worst_tca,
        score.worst_distance
    );
    assert_eq!(run.scanned, 795);
    let missed: Vec<Minimum> = score.misses.iter().map(|&i| set.events[i]).collect();
    let alarms: Vec<Minimum> = score.false_alarms.iter().map(|&j| run.reports[j]).collect();
    assert!(missed.is_empty(), "missed {missed:?}");
    assert!(alarms.is_empty(), "false alarms {alarms:?}");
    assert_eq!(score.recalled(), 179);
    assert!(
        score.worst_tca < 1e-6 && score.worst_distance < 1e-6,
        "{score:?}"
    );
}

/// The same screen at a 2 400 s step, past the quarter revolution, recalls 44 of the 179 keyed
/// events, and every one it misses is a crossing event: the documented condition on the step is a
/// real one, and the set sees it break. A coplanar pair's minima are 10⁶ s apart, and no step this
/// size skips one.
#[test]
fn a_step_past_a_quarter_revolution_misses_crossing_events() {
    let set = screening_set().expect("the committed set parses");
    let run = screen(&set, 2_400.0, 1e-6);
    let score = set.score(&run.reports, TOLERANCE);
    eprintln!(
        "2400 s step: recalled {} of {}, {} false alarms",
        score.recalled(),
        score.keyed,
        score.false_alarms.len()
    );
    assert_eq!(score.recalled(), 44, "{score:?}");
    assert!(
        score
            .misses
            .iter()
            .all(|&i| set.objects[set.events[i].a].group == Group::Crossing),
        "{score:?}"
    );
    assert!(score.false_alarms.is_empty(), "{score:?}");
}

/// S1: a dense scan of the separation itself, at 1 s and sharing nothing with the bracketing
/// search, on every planted pair and every 40th shell pair that is not comoving. Each pair's
/// sampled local minima are as many as its closed-form minima, each within a sample of one, and no
/// sample is closer than the pair's least distance.
#[test]
fn a_dense_scan_of_sampled_pairs_agrees_with_the_key() {
    let set = screening_set().expect("the committed set parses");
    let orbits: Vec<CircularOrbit> = set.objects.iter().map(|o| o.orbit(set.mu)).collect();
    let mut shell = 0;
    let mut checked = 0;
    for pair in &set.pairs {
        let planted = set.objects[pair.a].group != Group::Shell;
        if pair.kind == PairKind::Comoving {
            continue;
        }
        if !planted {
            shell += 1;
            if shell % 40 != 1 {
                continue;
            }
        }
        checked += 1;
        let rel = relative(&orbits, pair.a, pair.b);
        let sep = |t: f64| {
            let d = rel(t).0;
            libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        };
        let samples = 86_400_usize;
        let at = |i: usize| {
            #[expect(clippy::cast_precision_loss, reason = "below 2^53")]
            let t = i as f64;
            (t, sep(t))
        };
        let mut minima = Vec::new();
        let mut least = f64::INFINITY;
        for i in 1..samples {
            let ((_, before), (t, here), (_, after)) = (at(i - 1), at(i), at(i + 1));
            least = least.min(here);
            if here < before && here <= after {
                minima.push(t);
            }
        }
        assert!(
            least >= pair.least_distance - 1e-6,
            "{pair:?}: a sample {least} m apart"
        );
        let keyed: Vec<f64> = set
            .events
            .iter()
            .chain(&set.negatives)
            .filter(|m| (m.a, m.b) == (pair.a, pair.b))
            .map(|m| m.tca)
            .collect();
        if planted {
            assert_eq!(minima.len(), keyed.len(), "{pair:?}: {minima:?} {keyed:?}");
            for (t, k) in minima.iter().zip(&keyed) {
                assert!(
                    libm::fabs(t - k) <= 1.0,
                    "{pair:?}: sampled {t} s, keyed {k} s"
                );
            }
        } else {
            assert!(least >= set.threshold, "{pair:?}");
            assert!(keyed.is_empty(), "{pair:?}");
        }
    }
    assert_eq!(checked, 17 + 16);
}
