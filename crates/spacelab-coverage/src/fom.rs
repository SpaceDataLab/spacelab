//! Constellation figures of merit — the whole-grid answer a design study wants, aggregated from the
//! per-point [`CoverageMetrics`] that [`crate::revisit_map`] produces for a cell grid.
//!
//! A single point's metrics answer "how is *this* spot served"; a constellation is judged over a grid
//! of points, area-weighted (typically by `cos(latitude)` for a lat/lon grid so equal-area is
//! respected). This reduces that grid to the standard scalars: mean and worst-case coverage, the
//! worst point's revisit gap, and the response-time statistics — the numbers the Walker/flower/street
//! generators are tuned against.
//!
//! Gate: `tests::a_weighted_periodic_grid_has_the_exact_rational_figures_of_merit` below holds the
//! aggregation to the exact rationals of `derivations/fom_weighted_grid.mac`.

// The point count is a grid size (thousands of cells) that comfortably fits `u32`.
#![allow(clippy::cast_possible_truncation)]

use crate::metrics::CoverageMetrics;

/// Whole-constellation figures of merit over a grid of points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstellationFom {
    /// Number of grid points aggregated.
    pub point_count: u32,
    /// Area-weighted mean coverage fraction, in `[0, 1]`.
    pub mean_coverage: f64,
    /// Worst (minimum) coverage fraction of any point.
    pub min_coverage: f64,
    /// Worst-case revisit: the largest [`CoverageMetrics::max_gap`] over all points, seconds.
    pub worst_max_gap: f64,
    /// Area-weighted mean of the per-point [`CoverageMetrics::max_gap`], seconds.
    pub mean_max_gap: f64,
    /// Area-weighted mean of the per-point [`CoverageMetrics::mean_response_time`], seconds.
    pub mean_response_time: f64,
    /// Worst (maximum) [`CoverageMetrics::mean_response_time`] over all points, seconds.
    pub worst_response_time: f64,
    /// Number of points covered essentially the whole span (`coverage_fraction ≈ 1`).
    pub fully_covered_points: u32,
}

/// Aggregate per-point `points` into a [`ConstellationFom`]. `weights` (e.g. `cos(latitude)`) area-
/// weight the means; if `None` — or the wrong length, or non-positive in total — points are weighted
/// equally. Worst-case and counts are weight-independent.
#[must_use]
pub fn constellation_fom(points: &[CoverageMetrics], weights: Option<&[f64]>) -> ConstellationFom {
    if points.is_empty() {
        return ConstellationFom {
            point_count: 0,
            mean_coverage: 0.0,
            min_coverage: 0.0,
            worst_max_gap: 0.0,
            mean_max_gap: 0.0,
            mean_response_time: 0.0,
            worst_response_time: 0.0,
            fully_covered_points: 0,
        };
    }

    let n = points.len();
    let valid_weights = weights.filter(|w| w.len() == n && w.iter().sum::<f64>() > 0.0);
    let weight = |i: usize| valid_weights.map_or(1.0, |w| w[i]);
    let total_w: f64 = (0..n).map(weight).sum();

    let mut cov_w = 0.0;
    let mut gap_w = 0.0;
    let mut resp_w = 0.0;
    let mut min_coverage = f64::INFINITY;
    let mut worst_max_gap = 0.0_f64;
    let mut worst_response = 0.0_f64;
    let mut fully_covered = 0u32;

    for (i, m) in points.iter().enumerate() {
        let w = weight(i);
        cov_w += w * m.coverage_fraction;
        gap_w += w * m.max_gap;
        resp_w += w * m.mean_response_time;
        min_coverage = min_coverage.min(m.coverage_fraction);
        worst_max_gap = worst_max_gap.max(m.max_gap);
        worst_response = worst_response.max(m.mean_response_time);
        if m.coverage_fraction >= 1.0 - 1e-9 {
            fully_covered += 1;
        }
    }

    ConstellationFom {
        point_count: n as u32,
        mean_coverage: cov_w / total_w,
        min_coverage,
        worst_max_gap,
        mean_max_gap: gap_w / total_w,
        mean_response_time: resp_w / total_w,
        worst_response_time: worst_response,
        fully_covered_points: fully_covered,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::manual_midpoint)]
    use super::*;
    use crate::access::Window;
    use crate::metrics::metrics;

    fn point(windows: &[(f64, f64)], t0: f64, t1: f64) -> CoverageMetrics {
        let ws: Vec<Window> = windows
            .iter()
            .map(|&(rise, set)| Window { rise, set })
            .collect();
        metrics(&ws, t0, t1)
    }

    #[test]
    fn empty_grid_is_all_zero() {
        let fom = constellation_fom(&[], None);
        assert_eq!(fom.point_count, 0);
        assert_eq!(fom.mean_coverage, 0.0);
    }

    #[test]
    fn uniform_weighting_averages_and_takes_the_worst_point() {
        let full = point(&[(0.0, 1000.0)], 0.0, 1000.0); // coverage 1, gap 0, response 0
        let half = point(&[(0.0, 500.0)], 0.0, 1000.0); // coverage 0.5, trailing gap 500
        let fom = constellation_fom(&[full, half], None);

        assert_eq!(fom.point_count, 2);
        assert_eq!(
            fom.mean_coverage,
            (full.coverage_fraction + half.coverage_fraction) / 2.0
        );
        assert_eq!(fom.min_coverage, half.coverage_fraction);
        assert_eq!(fom.worst_max_gap, half.max_gap);
        assert_eq!(fom.mean_max_gap, (full.max_gap + half.max_gap) / 2.0);
        assert_eq!(fom.worst_response_time, half.mean_response_time);
        assert_eq!(fom.fully_covered_points, 1);
    }

    #[test]
    fn area_weights_shift_the_means_but_not_the_worst_case() {
        let full = point(&[(0.0, 1000.0)], 0.0, 1000.0);
        let half = point(&[(0.0, 500.0)], 0.0, 1000.0);
        // Weight the fully-covered point 3:1.
        let fom = constellation_fom(&[full, half], Some(&[3.0, 1.0]));
        let expect = (3.0 * full.coverage_fraction + half.coverage_fraction) / 4.0;
        assert!((fom.mean_coverage - expect).abs() < 1e-12);
        // Worst case is weight-independent.
        assert_eq!(fom.worst_max_gap, half.max_gap);
        // A wrong-length weight slice falls back to uniform.
        let uniform = constellation_fom(&[full, half], Some(&[1.0]));
        assert_eq!(
            uniform.mean_coverage,
            (full.coverage_fraction + half.coverage_fraction) / 2.0
        );
    }

    /// A grid point with only the three fields the aggregation reads, so the expected values below
    /// are arithmetic on chosen numbers rather than a second run of `metrics`.
    fn graded(coverage_fraction: f64, max_gap: f64, mean_response_time: f64) -> CoverageMetrics {
        CoverageMetrics {
            passes: 1,
            covered_seconds: 0.0,
            coverage_fraction,
            max_gap,
            mean_gap: 0.0,
            mean_response_time,
            longest_pass: 0.0,
            shortest_pass: 0.0,
        }
    }

    #[test]
    fn the_weighted_mean_response_time_and_the_full_coverage_count_are_the_stated_ones() {
        // The cases above read `worst_response_time` but never `mean_response_time`, and they count
        // one fully covered point out of two — where counting the *other* one gives the same answer.
        //
        //   weights 1, 2, 5 → total 8
        //   coverage (1·1 + 2·1 + 5·0.25)/8 = 4.25/8
        //   max gap  (1·0 + 2·10 + 5·40)/8  = 220/8
        //   response (1·4 + 2·9 + 5·100)/8  = 522/8
        let points = [
            graded(1.0, 0.0, 4.0),
            graded(1.0, 10.0, 9.0),
            graded(0.25, 40.0, 100.0),
        ];
        let fom = constellation_fom(&points, Some(&[1.0, 2.0, 5.0]));

        assert!(
            (fom.mean_coverage - 4.25 / 8.0).abs() < 1e-12,
            "{}",
            fom.mean_coverage
        );
        assert!(
            (fom.mean_max_gap - 220.0 / 8.0).abs() < 1e-12,
            "{}",
            fom.mean_max_gap
        );
        assert!(
            (fom.mean_response_time - 522.0 / 8.0).abs() < 1e-12,
            "mean response time {}",
            fom.mean_response_time
        );
        assert_eq!(fom.min_coverage, 0.25);
        assert_eq!(fom.worst_max_gap, 40.0);
        assert_eq!(fom.worst_response_time, 100.0);
        assert_eq!(
            fom.fully_covered_points, 2,
            "two of the three points are fully covered"
        );
    }

    #[test]
    fn weights_that_sum_to_zero_fall_back_to_uniform_rather_than_dividing_by_them() {
        // The guard is `sum > 0`, and zero is the one total on which it and `>= 0` disagree: an
        // all-zero weight slice would otherwise make every weighted mean 0/0.
        let points = [graded(1.0, 0.0, 4.0), graded(0.25, 40.0, 100.0)];
        let fom = constellation_fom(&points, Some(&[0.0, 0.0]));
        assert_eq!(fom.mean_coverage, (1.0 + 0.25) / 2.0);
        assert_eq!(fom.mean_response_time, (4.0 + 100.0) / 2.0);
    }

    #[test]
    fn response_time_penalises_an_uneven_gap_split() {
        // Same total gap, split evenly vs unevenly: the uneven split has a larger response time,
        // because response weights each gap quadratically.
        let even = point(&[(100.0, 110.0), (210.0, 220.0)], 0.0, 320.0);
        let uneven = point(&[(10.0, 20.0), (30.0, 40.0)], 0.0, 320.0);
        assert!(
            uneven.mean_response_time > even.mean_response_time,
            "uneven {} should exceed even {}",
            uneven.mean_response_time,
            even.mean_response_time
        );
    }

    /// `rises` passes of length `d` every `p` seconds, the first `a` seconds in, over one day.
    fn periodic(p: f64, d: f64, a: f64, rises: u32) -> CoverageMetrics {
        let ws: Vec<(f64, f64)> = (0..rises)
            .map(|i| {
                let rise = a + p * f64::from(i);
                (rise, rise + d)
            })
            .collect();
        point(&ws, 0.0, 86_400.0)
    }

    /// (G3) A four-point grid of periodic access patterns, weighted 1:2:3:4, against the exact
    /// rationals of `derivations/fom_weighted_grid.mac` (re-enumerated in
    /// `derivations/fom_weighted_grid.g`): mean coverage 11/30, mean max gap 5400, mean response
    /// 254315/108, min coverage 1/12, worst max gap 9900, worst response 51575/12.
    ///
    /// The extremes are picks of a single correctly rounded quotient, so they are exact; the weighted
    /// means carry four products and sums of rounded inputs, bound 8·ε relative.
    #[test]
    fn a_weighted_periodic_grid_has_the_exact_rational_figures_of_merit() {
        let points = [
            periodic(5400.0, 600.0, 1000.0, 16),
            periodic(5400.0, 600.0, 0.0, 16),
            point(&[(0.0, 86_400.0)], 0.0, 86_400.0),
            periodic(10_800.0, 900.0, 3000.0, 8),
        ];
        let fom = constellation_fom(&points, Some(&[1.0, 2.0, 3.0, 4.0]));
        let close = |got: f64, exact: f64| (got - exact).abs() <= 8.0 * f64::EPSILON * exact;
        assert!(
            close(fom.mean_coverage, 11.0 / 30.0),
            "{}",
            fom.mean_coverage
        );
        assert!(close(fom.mean_max_gap, 5400.0), "{}", fom.mean_max_gap);
        assert!(
            close(fom.mean_response_time, 254_315.0 / 108.0),
            "{}",
            fom.mean_response_time
        );
        assert_eq!(fom.min_coverage, 1.0 / 12.0);
        assert_eq!(fom.worst_max_gap, 9900.0);
        assert_eq!(fom.worst_response_time, 51_575.0 / 12.0);
        assert_eq!(fom.fully_covered_points, 1);
        assert_eq!(fom.point_count, 4);
    }
}
