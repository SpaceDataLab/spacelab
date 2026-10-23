//! Coverage analysis for constellations — the **α-characteristic** (Mozhaev), the design-time figure
//! of merit a delta-system is chosen by.
//!
//! The α-characteristic of `L`-fold coverage is the smallest sub-satellite viewing-zone angular
//! radius `α_L` at which the constellation keeps every point of the sphere within `α_L` of at least
//! `L` sub-satellite points *at all times* — i.e. the minimum sensor footprint that yields continuous
//! `L`-fold coverage. It depends only on the phase structure and inclination, and a smaller value is
//! better. It is exactly the worst-case `L`-th-nearest satellite angular distance over space and
//! time, which is what [`alpha_characteristic`] evaluates on a grid.
//!
//! This is design-time analysis (it allocates and propagates), so it lives behind the crate's `std`
//! feature; the generators stay `no_std`.

use spacelab_core::{Angle, Vec3};
use spacelab_elements::KeplerianElements;

use crate::walker::{Walker, WalkerPattern};

const TAU: f64 = core::f64::consts::TAU;

/// The `k`-th of `n` near-uniform Fibonacci-sphere directions.
fn fibonacci_dir(k: u32, n: u32) -> Vec3 {
    let ga = core::f64::consts::PI * (3.0 - libm::sqrt(5.0));
    let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(n);
    let r = libm::sqrt((1.0 - z * z).max(0.0));
    let phi = ga * f64::from(k);
    Vec3::new(r * libm::cos(phi), r * libm::sin(phi), z)
}

/// The α-characteristic of `fold`-fold coverage (radians) for the circular delta-system `sats`: the
/// smallest sub-satellite-zone angular radius at which the constellation continuously covers the
/// sphere at least `fold` times.
///
/// Evaluated as the worst case, over `grid_points` Fibonacci directions × `time_samples` instants of
/// one orbital period, of the `fold`-th-nearest satellite angular distance. Returns `π` (uncoverable)
/// when `fold` exceeds the satellite count. Larger grids tighten the (always slightly optimistic)
/// estimate.
#[must_use]
pub fn alpha_characteristic(
    sats: &[KeplerianElements],
    fold: u32,
    grid_points: u32,
    time_samples: u32,
) -> f64 {
    let n = sats.len();
    let fold = fold as usize;
    // An empty grid or an empty time axis runs neither loop and leaves `worst` at its initial 0.0 —
    // "this constellation is covered by a point sensor", which `minimum_walker` would then accept
    // from the very first candidate it tries. Nothing was sampled, so nothing is known: report the
    // same π the other degenerate inputs report.
    if n == 0 || fold == 0 || fold > n || grid_points == 0 || time_samples == 0 {
        return core::f64::consts::PI;
    }
    let period = TAU * libm::sqrt(sats[0].semi_major_axis.powi(3) / sats[0].mu);

    let mut worst = 0.0_f64;
    let mut dirs: Vec<Vec3> = Vec::with_capacity(n);
    let mut angles: Vec<f64> = Vec::with_capacity(n);
    for it in 0..time_samples {
        let t = period * f64::from(it) / f64::from(time_samples);
        dirs.clear();
        for e in sats {
            let r = e.propagate(t).to_state().0;
            let inv = 1.0 / r.norm();
            dirs.push(Vec3::new(r.x * inv, r.y * inv, r.z * inv));
        }
        for ip in 0..grid_points {
            let p = fibonacci_dir(ip, grid_points);
            angles.clear();
            for d in &dirs {
                angles.push(libm::acos(d.dot(p).clamp(-1.0, 1.0)));
            }
            angles.sort_by(f64::total_cmp);
            worst = worst.max(angles[fold - 1]);
        }
    }
    worst
}

/// Search for the Walker Delta system with the fewest satellites whose α-characteristic does not
/// exceed `max_footprint` (radians) — the cheapest constellation that continuously `fold`-fold covers
/// the sphere with a sensor of that footprint, at the common `inclination`, `semi_major_axis` and
/// `mu`. Counts are scanned upward and, within each, all plane factorisations and phasings tried, so
/// the first hit is a minimum-satellite design. Returns `None` if nothing at or below `max_total`
/// covers. The grid controls the α-characteristic estimate as in [`alpha_characteristic`].
#[must_use]
#[allow(clippy::too_many_arguments)] // footprint + the shared orbit + fold + the search + grid bounds
pub fn minimum_walker(
    max_footprint: f64,
    inclination: Angle,
    semi_major_axis: f64,
    mu: f64,
    fold: u32,
    max_total: u32,
    grid_points: u32,
    time_samples: u32,
) -> Option<Walker> {
    let mut sats = Vec::new();
    for total in fold..=max_total {
        for planes in 1..=total {
            if total % planes != 0 {
                continue;
            }
            let per_plane = total / planes;
            for phasing in 0..planes {
                let w = Walker {
                    pattern: WalkerPattern::Delta,
                    inclination,
                    planes,
                    per_plane,
                    phasing,
                    semi_major_axis,
                    mu,
                };
                sats.clear();
                w.generate(|e| sats.push(e));
                if alpha_characteristic(&sats, fold, grid_points, time_samples) <= max_footprint {
                    return Some(w);
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use crate::soc::StreetOfCoverage;
    use crate::walker::{Walker, WalkerPattern};
    use spacelab_core::Angle;

    const MU: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;

    fn walker(planes: u32, per_plane: u32) -> Vec<KeplerianElements> {
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(55.0_f64.to_radians()),
            planes,
            per_plane,
            phasing: 1,
            semi_major_axis: RE + 20_000e3,
            mu: MU,
        };
        let mut v = Vec::new();
        w.generate(|e| v.push(e));
        v
    }

    /// One satellite, circular and equatorial, sampled on the single grid direction and five
    /// instants, has a closed-form α: the grid direction is `+x`, the sub-point sweeps `2πi/5`, and
    /// the worst of `acos(cos 2πi/5)` over `i = 0..5` is `4π/5`.
    fn equatorial_single() -> Vec<KeplerianElements> {
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(0.0),
            planes: 1,
            per_plane: 1,
            phasing: 0,
            semi_major_axis: RE + 20_000e3,
            mu: MU,
        };
        let mut v = Vec::new();
        w.generate(|e| v.push(e));
        v
    }

    #[test]
    fn the_fibonacci_grid_is_unit_length_even_in_z_and_turns_by_the_golden_angle() {
        // The grid feeds `alpha_characteristic`, whose own assertions are all one-sided bounds, so a
        // garbled grid only ever pushes α *down* — straight through an upper bound. Pin the three
        // defining properties instead. The golden angle is written here as τ/φ² rather than the
        // source's π(3−√5); they are the same number — 4π/(3+√5) = 4π(3−√5)/4 — by a different
        // transcription, so agreement is evidence and not a tautology.
        let n = 97_u32;
        let phi = 1.0_f64.midpoint(libm::sqrt(5.0));
        let ga = TAU / (phi * phi);
        for k in 0..n {
            let v = fibonacci_dir(k, n);
            assert!((v.norm() - 1.0).abs() < 1e-15, "k={k}: |v| = {}", v.norm());

            let want_z = 1.0 - (2.0 * f64::from(k) + 1.0) / f64::from(n);
            assert!(
                (v.z - want_z).abs() < 1e-15,
                "k={k}: z {} want {want_z}",
                v.z
            );

            let want = libm::fmod(libm::fmod(ga * f64::from(k), TAU) + TAU, TAU);
            let got = libm::fmod(libm::atan2(v.y, v.x) + TAU, TAU);
            let gap = (got - want).abs();
            assert!(
                gap.min(TAU - gap) < 1e-9,
                "k={k}: azimuth {got} want {want}"
            );
        }
    }

    #[test]
    fn a_degenerate_request_is_uncoverable_rather_than_covered_by_a_point() {
        let sats = walker(2, 2); // 4
        let pi = core::f64::consts::PI;
        for (fold, grid, times, why) in [
            (0, 200, 20, "a zero-fold request"),
            (5, 200, 20, "more folds than satellites"),
            (1, 0, 20, "an empty direction grid"),
            (1, 200, 0, "an empty time axis"),
        ] {
            let a = alpha_characteristic(&sats, fold, grid, times);
            assert!((a - pi).abs() < 1e-15, "{why} gave {a}, not pi");
        }
        assert!(
            (alpha_characteristic(&[], 1, 200, 20) - pi).abs() < 1e-15,
            "an empty constellation is not uncoverable"
        );
        // `fold == n` is the last admissible request and must stay strictly inside the range.
        let full = alpha_characteristic(&sats, 4, 200, 20);
        assert!(full > 0.0 && full < pi, "fold == n gave {full}");
    }

    #[test]
    fn alpha_is_the_worst_sampled_angular_distance_and_its_closed_form_at_one_direction() {
        // Recompute the definition from the primary quantities — Kepler's third law for the sampling
        // period, the propagated position, its own normalisation — so a mutation of any of them
        // moves the implementation without moving the reference. `fibonacci_dir` is shared with the
        // implementation on purpose; the test above is what pins the grid.
        let sats = walker(3, 2); // 6, inclined 55 degrees
        let (grid, times) = (2_u32, 5_u32);
        let period = TAU * libm::sqrt(sats[0].semi_major_axis.powi(3) / sats[0].mu);
        let mut want = 0.0_f64;
        for it in 0..times {
            let t = period * f64::from(it) / f64::from(times);
            let dirs: Vec<Vec3> = sats
                .iter()
                .map(|e| {
                    let r = e.propagate(t).to_state().0;
                    let n = r.norm();
                    Vec3::new(r.x / n, r.y / n, r.z / n)
                })
                .collect();
            for ip in 0..grid {
                let p = fibonacci_dir(ip, grid);
                let mut a: Vec<f64> = dirs
                    .iter()
                    .map(|d| libm::acos(d.dot(p).clamp(-1.0, 1.0)))
                    .collect();
                a.sort_by(f64::total_cmp);
                want = want.max(a[0]);
            }
        }
        let got = alpha_characteristic(&sats, 1, grid, times);
        assert!((got - want).abs() < 1e-15, "alpha {got} against {want}");

        // The human-checkable anchor: one equatorial satellite on the single `+x` direction.
        let closed = 4.0 * core::f64::consts::PI / 5.0;
        let one = alpha_characteristic(&equatorial_single(), 1, 1, 5);
        assert!(
            (one - closed).abs() < 1e-9,
            "alpha {one} against 4pi/5 {closed}"
        );
    }

    #[test]
    fn the_minimum_walker_search_visits_the_smallest_designs_first() {
        // At a footprint of π every constellation qualifies, so the very first candidate the scan
        // reaches — one satellite in one plane — is the answer. Anything that reorders or thins the
        // (total, planes) enumeration returns a larger design and is caught here without needing a
        // coverage oracle at all.
        let w = minimum_walker(
            core::f64::consts::PI,
            Angle::from_radians(70.0_f64.to_radians()),
            RE + 20_000e3,
            MU,
            1,
            12,
            40,
            8,
        )
        .expect("every design covers at a footprint of pi");
        assert_eq!(
            (w.planes, w.per_plane, w.total()),
            (1, 1, 1),
            "the search skipped the single-satellite design"
        );
    }

    #[test]
    fn alpha_is_bounded_by_a_covering_footprint() {
        // A Street-of-Coverage design continuously covers the sphere with footprint θ, so the
        // α-characteristic (the *minimum* covering radius) must not exceed θ — and must be positive.
        let soc = StreetOfCoverage {
            inclination: Angle::from_radians(86.4_f64.to_radians()),
            altitude: 780e3,
            min_elevation: Angle::from_radians(8.2_f64.to_radians()),
            earth_radius: RE,
            mu: MU,
            per_plane: None,
        };
        let g = soc.geometry().expect("feasible");
        let mut sats = Vec::new();
        soc.generate(|e| sats.push(e));

        let alpha1 = alpha_characteristic(&sats, 1, 500, 60);
        assert!(alpha1 > 0.0);
        assert!(
            alpha1 <= g.footprint_half_angle + 1e-3,
            "alpha1 {} exceeds footprint {}",
            alpha1,
            g.footprint_half_angle
        );
    }

    #[test]
    fn alpha_is_monotone_in_fold_and_decreases_with_density() {
        let sparse = walker(4, 4); // 16
        let dense = walker(8, 8); //  64

        let a1 = alpha_characteristic(&sparse, 1, 400, 40);
        let a2 = alpha_characteristic(&sparse, 2, 400, 40);
        assert!(a2 >= a1, "two-fold {a2} below single {a1}");

        let dense1 = alpha_characteristic(&dense, 1, 400, 40);
        assert!(dense1 < a1, "dense {dense1} not below sparse {a1}");
    }

    #[test]
    fn a_single_satellite_cannot_cover_the_sphere() {
        // One satellite leaves the antipode of its sub-point uncovered, so α₁ ≈ π.
        let one = walker(1, 1);
        let a1 = alpha_characteristic(&one, 1, 400, 40);
        assert!(a1 > 3.0, "single-satellite alpha {a1} not near pi");
    }

    #[test]
    fn minimum_walker_finds_a_verified_covering_design() {
        let inc = Angle::from_radians(70.0_f64.to_radians());
        let a = RE + 20_000e3;
        let (grid, times) = (250, 30);

        // A generous footprint is met by some small Walker; the returned design really covers.
        let footprint = 1.2; // rad
        let w = minimum_walker(footprint, inc, a, MU, 1, 12, grid, times)
            .expect("a covering Walker exists at 1.2 rad");
        let mut sats = Vec::new();
        w.generate(|e| sats.push(e));
        assert!(alpha_characteristic(&sats, 1, grid, times) <= footprint);
        assert!(w.total() <= 12);

        // A tighter footprint cannot be met with fewer satellites than a looser one.
        if let Some(tight) = minimum_walker(0.9, inc, a, MU, 1, 12, grid, times) {
            assert!(
                w.total() <= tight.total(),
                "looser footprint needs more satellites"
            );
        }
    }
}
