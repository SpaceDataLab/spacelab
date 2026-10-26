//! Coverage metrics over a set of access windows — the operational answers the windows exist to
//! produce: how often the target is seen, and the worst-case wait between looks.
//!
//! The headline is the **max gap**: the longest interval the target is *not* covered, counting the
//! stretches before the first pass and after the last. A revisit requirement ("no worse than two
//! hours of latency", `notes/06-coverage-source-material.md`) is a bound on exactly this number.
//!
//! Gate: `tests/metrics.rs::a_periodic_pattern_has_the_exact_rational_metrics` holds every metric to
//! the exact rationals of `derivations/metrics_periodic_gaps.mac`.

// The pass count is a `usize` slice length that comfortably fits `u32` (tens of passes a day).
#![allow(clippy::cast_possible_truncation)]

use crate::access::Window;

/// Coverage statistics over a span `[t0, t1]`, derived from a set of access [`Window`]s that must be
/// sorted and non-overlapping (as [`crate::access_windows`] emits them).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageMetrics {
    /// Number of passes.
    pub passes: u32,
    /// Total time covered, seconds.
    pub covered_seconds: f64,
    /// Fraction of the span covered, in `[0, 1]`.
    pub coverage_fraction: f64,
    /// Longest uncovered interval, seconds — the worst-case revisit latency, including the wait
    /// before the first pass and after the last.
    pub max_gap: f64,
    /// Mean uncovered interval, seconds.
    pub mean_gap: f64,
    /// Time-average of the wait until the next pass, seconds — the Wertz **mean response time**,
    /// `Σ gapᵢ²/(2·span)`. Long gaps count quadratically, so this separates two constellations with the
    /// same mean gap but different worst cases. The final gap's response is taken to decay to zero at
    /// `t1` (the analysis boundary), so no periodicity is assumed.
    pub mean_response_time: f64,
    /// Longest single pass, seconds.
    pub longest_pass: f64,
    /// Shortest single pass, seconds.
    pub shortest_pass: f64,
}

/// Compute [`CoverageMetrics`] from access `windows` over `[t0, t1]`.
///
/// The gaps counted are the leading interval `[t0, first.rise]`, each between-pass interval, and the
/// trailing interval `[last.set, t1]`; with no passes the whole span is one gap.
#[must_use]
pub fn metrics(windows: &[Window], t0: f64, t1: f64) -> CoverageMetrics {
    let span = t1 - t0;
    if windows.is_empty() {
        return CoverageMetrics {
            passes: 0,
            covered_seconds: 0.0,
            coverage_fraction: 0.0,
            max_gap: span,
            mean_gap: span,
            mean_response_time: if span > 0.0 { span / 2.0 } else { 0.0 },
            longest_pass: 0.0,
            shortest_pass: 0.0,
        };
    }

    let mut covered = 0.0;
    let mut longest_pass = 0.0_f64;
    let mut shortest_pass = f64::INFINITY;
    for w in windows {
        let d = w.duration();
        covered += d;
        longest_pass = longest_pass.max(d);
        shortest_pass = shortest_pass.min(d);
    }

    let leading = windows[0].rise - t0;
    let trailing = t1 - windows[windows.len() - 1].set;
    let mut max_gap = leading.max(trailing);
    let mut gap_sum = leading + trailing;
    let mut sq_sum = leading * leading + trailing * trailing;
    let mut gap_count = 2u32;
    for i in 1..windows.len() {
        let gap = windows[i].rise - windows[i - 1].set;
        max_gap = max_gap.max(gap);
        gap_sum += gap;
        sq_sum += gap * gap;
        gap_count += 1;
    }

    CoverageMetrics {
        passes: windows.len() as u32,
        covered_seconds: covered,
        coverage_fraction: covered / span,
        max_gap,
        mean_gap: gap_sum / f64::from(gap_count),
        mean_response_time: if span > 0.0 {
            sq_sum / (2.0 * span)
        } else {
            0.0
        },
        longest_pass,
        shortest_pass,
    }
}
