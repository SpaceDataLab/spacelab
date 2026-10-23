//! Combining the access of several satellites over one cell into a single coverage schedule.
//!
//! A constellation exists to close the revisit gap a single satellite leaves. Answering "how often
//! is this cell seen by *any* satellite" means taking the **union** of every satellite's windows —
//! overlapping or touching passes merge into one covered interval — and then measuring the gaps in
//! that union. `notes/06-coverage-source-material.md`: the operational requirement is a bound on the
//! max gap, and only the combined schedule can meet it.
//!
//! `std`-only: merging sorts the windows, which needs a growable buffer.

use core::cmp::Ordering;

use crate::access::Window;
use crate::metrics::{CoverageMetrics, metrics};

/// Merge a set of windows — from any number of satellites, in any order — into the disjoint,
/// sorted intervals of their union. Windows that overlap or merely touch are fused.
pub fn merge_windows(windows: &mut Vec<Window>) {
    if windows.len() < 2 {
        return;
    }
    windows.sort_by(|a, b| a.rise.partial_cmp(&b.rise).unwrap_or(Ordering::Equal));
    let mut write = 0;
    for read in 1..windows.len() {
        if windows[read].rise <= windows[write].set {
            windows[write].set = windows[write].set.max(windows[read].set);
        } else {
            write += 1;
            windows[write] = windows[read];
        }
    }
    windows.truncate(write + 1);
}

/// The coverage metrics of the *union* of `windows` over `[t0, t1]` — the combined schedule of a
/// constellation. Merges in place, then measures the gaps.
#[must_use]
pub fn combined_metrics(windows: &mut Vec<Window>, t0: f64, t1: f64) -> CoverageMetrics {
    merge_windows(windows);
    metrics(windows, t0, t1)
}

#[cfg(test)]
mod tests {
    use super::{Window, combined_metrics, merge_windows};

    fn w(rise: f64, set: f64) -> Window {
        Window { rise, set }
    }

    #[test]
    fn overlapping_and_touching_windows_fuse() {
        // (0,10) & (5,15) overlap → (0,15); (15,20) touches → (0,20); (30,40) is separate.
        let mut ws = vec![w(30.0, 40.0), w(0.0, 10.0), w(15.0, 20.0), w(5.0, 15.0)];
        merge_windows(&mut ws);
        assert_eq!(ws, vec![w(0.0, 20.0), w(30.0, 40.0)]);
    }

    #[test]
    fn disjoint_windows_are_left_alone_but_sorted() {
        let mut ws = vec![w(50.0, 60.0), w(10.0, 20.0)];
        merge_windows(&mut ws);
        assert_eq!(ws, vec![w(10.0, 20.0), w(50.0, 60.0)]);
    }

    #[test]
    fn the_union_shrinks_the_gap_a_single_satellite_leaves() {
        // One satellite covers the day's start and end but leaves a 22 h gap in the middle.
        let one = vec![w(0.0, 3_600.0), w(82_800.0, 86_400.0)];
        let mut solo = one.clone();
        let solo_gap = combined_metrics(&mut solo, 0.0, 86_400.0).max_gap;

        // A second satellite passing mid-day splits that gap roughly in half.
        let mut both = one;
        both.push(w(43_200.0, 46_800.0));
        let both_gap = combined_metrics(&mut both, 0.0, 86_400.0).max_gap;

        assert!(
            both_gap < solo_gap * 0.6,
            "the second satellite must roughly halve the {solo_gap:.0} s gap, got {both_gap:.0}"
        );
    }
}
