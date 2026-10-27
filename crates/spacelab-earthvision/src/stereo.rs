//! Dense stereo matching by Semi-Global Matching (SGM, Hirschmüller 2008) — the disparity engine
//! behind a stereo DSM (digital surface model).
//!
//! Given two **rectified** grayscale images (a point in the left image at column `x` matches the right
//! image at `x − d` for disparity `d ≥ 0`), SGM estimates the per-pixel disparity that a global 2-D
//! smoothness would, at a fraction of the cost, by summing several cheap 1-D dynamic-programming passes:
//!
//! 1. a **matching cost** volume `C(y, x, d)` — here the Hamming distance of local **census**
//!    descriptors ([`census_cost_volume`]), robust to illumination and the standard SGM data term;
//! 2. **path aggregation** ([`aggregate_sgm`]): along each of several directions `r`, the recursion
//!    `L_r(p,d) = C(p,d) + min(L_r(p−r,d), L_r(p−r,d±1)+P1, minₖ L_r(p−r,k)+P2) − minₖ L_r(p−r,k)`
//!    with a small penalty `P1` for a ±1 disparity change (slanted surfaces) and a larger `P2` for a
//!    jump (a depth discontinuity). Their sum `S = Σ_r L_r` is the aggregated cost;
//! 3. **winner-take-all** with a parabola sub-pixel refinement ([`disparity_wta`]).
//!
//! [`semi_global_match`] runs all three; [`depth_from_disparity`] turns disparity into depth for a
//! rectified pinhole pair (`Z = f·B/d`), the DSM elevation. `no_std` + `alloc`.
//!
//! Gate: `derivations/stereo_disparity.mac` shows that for two cameras sharing a rotation, with the
//! baseline along the camera `x` axis, a point stays on one row and its disparity is `f·B/z_c`, and
//! that the sub-pixel step is the vertex of the parabola through the three costs;
//! `derivations/stereo_disparity.gp` recomputes the vertices by interpolation. The module tests
//! `depth_from_disparity_inverts_the_disparity_of_a_rotated_rectified_pair` (G1) and
//! `the_subpixel_refinement_is_the_vertex_of_the_interpolating_parabola` (G3) hold the shipped code
//! to them.

// Raster indices cast usize↔isize/f64 in range; single letters (p, d, y, x, r) are the SGM notation.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use alloc::vec;
use alloc::vec::Vec;

/// The eight cardinal + diagonal path directions `(dy, dx)` SGM aggregates over.
pub const DIRECTIONS_8: [(i32, i32); 8] = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

/// The four cardinal path directions (a cheaper aggregation).
pub const DIRECTIONS_4: [(i32, i32); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];

/// The tuning knobs of a Semi-Global Match: the disparity search range, the census window radius, and
/// the two smoothness penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgmConfig {
    /// Number of disparities searched, `d ∈ [0, d_max)`.
    pub d_max: usize,
    /// Census window radius (window side `2·radius+1`).
    pub radius: usize,
    /// Penalty for a ±1 disparity change between adjacent pixels (slanted surfaces).
    pub p1: f64,
    /// Penalty for a larger disparity jump (a depth discontinuity); `p2 ≥ p1`.
    pub p2: f64,
}

/// The census bit-string of pixel `(y, x)`: one bit per neighbour in the `(2·radius+1)²` window, set
/// when the neighbour is brighter than the centre (border neighbours clamped to the edge).
fn census_at(img: &[f64], h: usize, w: usize, y: usize, x: usize, radius: usize) -> u64 {
    let center = img[y * w + x];
    let mut bits = 0u64;
    for dy in -(radius as isize)..=(radius as isize) {
        for dx in -(radius as isize)..=(radius as isize) {
            if dy == 0 && dx == 0 {
                continue;
            }
            let ny = (y as isize + dy).clamp(0, h as isize - 1) as usize;
            let nx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
            bits = (bits << 1) | u64::from(img[ny * w + nx] > center);
        }
    }
    bits
}

/// Census descriptors of a whole image (`h·w`), one `u64` per pixel.
fn census_image(img: &[f64], h: usize, w: usize, radius: usize) -> Vec<u64> {
    let mut out = vec![0u64; h * w];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = census_at(img, h, w, y, x, radius);
        }
    }
    out
}

/// The census-Hamming matching-cost volume `C(y, x, d)` (row-major `h·w·d_max`, `d` the fastest axis)
/// for a rectified pair: `C = popcount(censusL(y,x) XOR censusR(y,x−d))`, or `n_bits` (the maximum) at
/// pixels where `x − d` falls off the right image.
#[must_use]
pub fn census_cost_volume(
    left: &[f64],
    right: &[f64],
    h: usize,
    w: usize,
    cfg: &SgmConfig,
) -> Vec<f64> {
    let (d_max, radius) = (cfg.d_max, cfg.radius);
    let cl = census_image(left, h, w, radius);
    let cr = census_image(right, h, w, radius);
    let n_bits = ((2 * radius + 1) * (2 * radius + 1) - 1) as f64;
    let mut cost = vec![0.0; h * w * d_max];
    for y in 0..h {
        for x in 0..w {
            let base = (y * w + x) * d_max;
            for d in 0..d_max {
                cost[base + d] = if x >= d {
                    f64::from((cl[y * w + x] ^ cr[y * w + (x - d)]).count_ones())
                } else {
                    n_bits
                };
            }
        }
    }
    cost
}

/// Aggregate a cost volume `C` (row-major `h·w·d_max`) along the given `directions`: the sum of the
/// SGM path recursions, `S = Σ_r L_r`. `p1` penalises a ±1 disparity step, `p2 (≥ p1)` a larger jump.
#[must_use]
pub fn aggregate_sgm(
    cost: &[f64],
    h: usize,
    w: usize,
    cfg: &SgmConfig,
    directions: &[(i32, i32)],
) -> Vec<f64> {
    let (d_max, p1, p2) = (cfg.d_max, cfg.p1, cfg.p2);
    let mut agg = vec![0.0; h * w * d_max];
    let mut l = vec![0.0; h * w * d_max];
    for &(dy, dx) in directions {
        // Traverse so the predecessor pixel (y−dy, x−dx) is always visited first.
        let ys: Vec<usize> = if dy >= 0 {
            (0..h).collect()
        } else {
            (0..h).rev().collect()
        };
        let xs: Vec<usize> = if dx >= 0 {
            (0..w).collect()
        } else {
            (0..w).rev().collect()
        };
        for &y in &ys {
            for &x in &xs {
                let p = (y * w + x) * d_max;
                let py = y as isize - dy as isize;
                let px = x as isize - dx as isize;
                if py < 0 || py >= h as isize || px < 0 || px >= w as isize {
                    l[p..p + d_max].copy_from_slice(&cost[p..p + d_max]);
                    continue;
                }
                let q = (py as usize * w + px as usize) * d_max;
                let mut min_prev = f64::INFINITY;
                for k in 0..d_max {
                    min_prev = min_prev.min(l[q + k]);
                }
                for d in 0..d_max {
                    let mut best = l[q + d];
                    if d > 0 {
                        best = best.min(l[q + d - 1] + p1);
                    }
                    if d + 1 < d_max {
                        best = best.min(l[q + d + 1] + p1);
                    }
                    best = best.min(min_prev + p2);
                    l[p + d] = cost[p + d] + best - min_prev;
                }
            }
        }
        for i in 0..agg.len() {
            agg[i] += l[i];
        }
    }
    agg
}

/// Winner-take-all disparity from an aggregated cost volume, with a parabola sub-pixel refinement at
/// interior minima: `d* = d − ½·(S[d+1]−S[d−1])/(S[d+1]−2S[d]+S[d−1])`. One `f64` disparity per pixel.
#[must_use]
pub fn disparity_wta(agg: &[f64], h: usize, w: usize, d_max: usize) -> Vec<f64> {
    let mut disp = vec![0.0; h * w];
    for p in 0..h * w {
        let base = p * d_max;
        let mut best = 0;
        for d in 1..d_max {
            if agg[base + d] < agg[base + best] {
                best = d;
            }
        }
        let mut d = best as f64;
        if best > 0 && best + 1 < d_max {
            let (c0, c1, c2) = (agg[base + best - 1], agg[base + best], agg[base + best + 1]);
            let denom = c0 + c2 - 2.0 * c1;
            if denom.abs() > 1e-12 {
                d -= 0.5 * (c2 - c0) / denom;
            }
        }
        disp[p] = d;
    }
    disp
}

/// Full SGM: census cost, aggregation over [`DIRECTIONS_8`], winner-take-all with sub-pixel. Returns
/// one disparity per pixel (`h·w`).
#[must_use]
pub fn semi_global_match(
    left: &[f64],
    right: &[f64],
    h: usize,
    w: usize,
    cfg: &SgmConfig,
) -> Vec<f64> {
    let cost = census_cost_volume(left, right, h, w, cfg);
    let agg = aggregate_sgm(&cost, h, w, cfg, &DIRECTIONS_8);
    disparity_wta(&agg, h, w, cfg.d_max)
}

/// Depth from disparity for a rectified pinhole stereo pair: `Z = focal·baseline/d` (`focal` in pixels,
/// `baseline` in world units). `0` for a non-positive disparity (no valid intersection).
#[must_use]
pub fn depth_from_disparity(disparity: f64, focal: f64, baseline: f64) -> f64 {
    if disparity > 0.0 {
        focal * baseline / disparity
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal, clippy::float_cmp, clippy::unwrap_used)]
    use super::*;
    use crate::camera::{PinholeCamera, triangulate};
    use spacelab_core::{Mat3, Vec3};

    // A deterministic pseudo-random texture in [0, 1) — a unique local pattern so census matches are
    // unambiguous (no std rng in a no_std crate).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f64) / ((1u64 << 24) as f64)
    }

    #[test]
    fn aggregate_matches_a_hand_computed_single_path() {
        // One left→right path over a 1×3 image, d_max = 2. Border pixel x=0 initialises L = C; then the
        // recursion propagates. Verify against the closed-form values.
        let (h, w, dmax) = (1usize, 3usize, 2usize);
        // C[x][d]: x0 {0,4}, x1 {3,0}, x2 {0,5}
        let cost = vec![0.0, 4.0, 3.0, 0.0, 0.0, 5.0];
        let cfg = SgmConfig {
            d_max: dmax,
            radius: 1,
            p1: 1.0,
            p2: 4.0,
        };
        let agg = aggregate_sgm(&cost, h, w, &cfg, &[(0, 1)]);
        // x0 (border): L = C = {0, 4}.
        // x1: min_prev = 0. d0: C=3 + min(L0[0]=0, L0[1]+P1=5, min_prev+P2=4) − 0 = 3+0 = 3.
        //     d1: C=0 + min(L0[1]=4, L0[0]+P1=1, min_prev+P2=4) − 0 = 0+1 = 1.  ⇒ {3, 1}
        // x2: min_prev = 1. d0: C=0 + min(L1[0]=3, L1[1]+P1=2, 1+P2=5) − 1 = 0 + 2 − 1 = 1.
        //     d1: C=5 + min(L1[1]=1, L1[0]+P1=4, 1+P2=5) − 1 = 5 + 1 − 1 = 5.  ⇒ {1, 5}
        assert_eq!(agg, vec![0.0, 4.0, 3.0, 1.0, 1.0, 5.0]);
    }

    #[test]
    fn recovers_a_constant_disparity_on_random_texture() {
        // Warp a random-texture left image by a constant disparity d0 (right(y,x) = left(y,x+d0)); SGM
        // must recover exactly d0 on the interior, where the texture is present in both images.
        let (h, w) = (40usize, 64usize);
        let d0 = 4usize;
        let d_max = 12usize;
        let mut seed = 20240720u64;
        let left: Vec<f64> = (0..h * w).map(|_| lcg(&mut seed)).collect();
        let mut right = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                right[y * w + x] = if x + d0 < w {
                    left[y * w + (x + d0)]
                } else {
                    lcg(&mut seed) // undefined (occluded) region — random filler
                };
            }
        }
        let cfg = SgmConfig {
            d_max,
            radius: 2,
            p1: 6.0,
            p2: 60.0,
        };
        let disp = semi_global_match(&left, &right, h, w, &cfg);
        // Interior columns (past the census radius and the disparity search), interior rows.
        let (mut ok, mut total) = (0usize, 0usize);
        for y in 3..h - 3 {
            for x in (d0 + 3)..(w - 3) {
                total += 1;
                if (disp[y * w + x].round() as usize) == d0 {
                    ok += 1;
                }
            }
        }
        assert!(
            ok as f64 / total as f64 > 0.98,
            "recovered {ok}/{total} correct disparities"
        );
    }

    #[test]
    fn resolves_a_two_plane_scene_across_the_discontinuity() {
        // Left half disparity 3, right half disparity 7 — a depth step. SGM's P2 term must let the
        // boundary through while keeping each plane flat; most interior pixels land on the right plane.
        let (h, w) = (40usize, 80usize);
        let d_max = 14usize;
        let (dl, dr) = (3usize, 7usize);
        let mut seed = 777u64;
        let left: Vec<f64> = (0..h * w).map(|_| lcg(&mut seed)).collect();
        let truth = |x: usize| if x < w / 2 { dl } else { dr };
        let mut right = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                let d = truth(x);
                right[y * w + x] = if x + d < w {
                    left[y * w + (x + d)]
                } else {
                    lcg(&mut seed)
                };
            }
        }
        // P2 ≈ the census data range (max 24) keeps the depth step sharp; a much larger P2 over-smooths
        // it. A narrow band astride the discontinuity is still mis-assigned (inherent to SGM), so the
        // accuracy is measured off that band.
        let cfg = SgmConfig {
            d_max,
            radius: 2,
            p1: 5.0,
            p2: 30.0,
        };
        let disp = semi_global_match(&left, &right, h, w, &cfg);
        let (mut ok, mut total) = (0usize, 0usize);
        for y in 3..h - 3 {
            for x in (dr + 3)..(w - 3) {
                if (x as isize - w as isize / 2).abs() < 4 {
                    continue; // skip the immediate boundary band
                }
                total += 1;
                if (disp[y * w + x].round() as usize) == truth(x) {
                    ok += 1;
                }
            }
        }
        assert!(
            ok as f64 / total as f64 > 0.95,
            "two-plane recovery {ok}/{total}"
        );
    }

    #[test]
    fn depth_from_disparity_matches_pinhole_triangulation() {
        // A rectified pair: identical rotation, centres 0 and (B,0,0), same intrinsics. The disparity of
        // a world point is f·B/Z, so depth_from_disparity must return its Z — and match triangulate().
        let (focal, cx, cy, baseline) = (1000.0, 320.0, 240.0, 5.0);
        let intr = |center| PinholeCamera {
            rotation: Mat3::IDENTITY,
            center,
            focal,
            principal: [cx, cy],
        };
        let left = intr(Vec3::new(0.0, 0.0, 0.0));
        let right = intr(Vec3::new(baseline, 0.0, 0.0));
        for &z in &[8.0, 15.0, 40.0, 100.0] {
            let world = Vec3::new(1.3, -0.7, z);
            let ul = left.project(world);
            let ur = right.project(world);
            let disparity = ul[0] - ur[0];
            let depth = depth_from_disparity(disparity, focal, baseline);
            assert!((depth - z).abs() < 1e-9, "depth {depth} vs Z {z}");
            let tri = triangulate(&[(&left, ul), (&right, ur)]).unwrap();
            assert!((tri.z - z).abs() < 1e-6, "triangulate Z {} vs {z}", tri.z);
        }
    }
    /// (G1) `Z = f·B/d` inverts the disparity of a rectified pair: two cameras sharing a generic
    /// rotation `R`, the right centre moved by `B Rᵀ e_x`, image a point on one row and `f·B/z_c`
    /// apart — the identity `derivations/stereo_disparity.mac` proves for any `R`. The bound is the
    /// rounding of the two pixels the disparity subtracts, relative to the disparity itself.
    #[test]
    fn depth_from_disparity_inverts_the_disparity_of_a_rotated_rectified_pair() {
        let (focal, baseline) = (1000.0, 5.0);
        let rotation = Mat3::rot1(0.3) * Mat3::rot3(-0.7);
        let left = PinholeCamera {
            rotation,
            center: Vec3::new(1.5, -2.0, 0.25),
            focal,
            principal: [320.0, 240.0],
        };
        let right = PinholeCamera {
            center: left.center + rotation.transpose() * Vec3::new(baseline, 0.0, 0.0),
            ..left
        };
        for world in [
            Vec3::new(3.0, 4.0, 60.0),
            Vec3::new(-20.0, 7.5, 140.0),
            Vec3::new(0.5, -9.0, 25.0),
            Vec3::new(40.0, 30.0, 900.0),
        ] {
            let zc = left.camera_coordinates(world).z;
            assert!(zc > 0.0, "point behind the pair");
            let (ul, ur) = (left.project(world), right.project(world));
            let row_tol = 8.0 * f64::EPSILON * (ul[1].abs() + ur[1].abs());
            assert!(
                (ul[1] - ur[1]).abs() <= row_tol,
                "rows {} vs {}",
                ul[1],
                ur[1]
            );
            let d = ul[0] - ur[0];
            let depth = depth_from_disparity(d, focal, baseline);
            let tol = 8.0 * f64::EPSILON * (1.0 + (ul[0].abs() + ur[0].abs()) / d);
            assert!(
                (depth / zc - 1.0).abs() <= tol,
                "depth {depth} vs z_c {zc} (disparity {d})"
            );
        }
    }

    /// (G3) The refinement returns the vertex of the parabola through the three costs around the
    /// integer winner, exactly as `derivations/stereo_disparity.mac` computes it in rationals and
    /// `derivations/stereo_disparity.gp` re-derives by interpolating: `33/26`, `283/42` and `163/14`.
    /// The costs are dyadic, so the only rounding is the closed form's own: a few ulps of the answer.
    #[test]
    fn the_subpixel_refinement_is_the_vertex_of_the_interpolating_parabola() {
        // (integer winner, costs at winner − 1, winner, winner + 1, exact vertex)
        let rows: [(usize, [f64; 3], f64); 3] = [
            (1, [1.5, 0.25, 0.625], 33.0 / 26.0),
            (7, [5.125, 4.5, 6.5], 283.0 / 42.0),
            (12, [1.0 / 1024.0, 0.0, 3.0 / 512.0], 163.0 / 14.0),
        ];
        for (winner, costs, vertex) in rows {
            // Every other disparity costs more than the three, so the winner is the one intended.
            let d_max = winner + 2;
            let mut volume = vec![100.0; d_max];
            volume[winner - 1..=winner + 1].copy_from_slice(&costs);
            let got = disparity_wta(&volume, 1, 1, d_max)[0];
            assert!(
                (got - vertex).abs() <= 4.0 * f64::EPSILON * vertex,
                "winner {winner}: {got} vs {vertex}"
            );
        }
    }

    // A 3x3 patch of nine distinct brightnesses, chosen so every census bit below is decided by a
    // different neighbour and no two rows or columns are interchangeable.
    const PATCH: [f64; 9] = [1.0, 9.0, 2.0, 8.0, 5.0, 7.0, 3.0, 0.0, 6.0];

    #[test]
    fn the_path_directions_are_the_unit_steps_they_are_named_for() {
        // SGM's aggregation is a sum over paths, and it is only isotropic if the path set is. Both
        // constants are stated here as what they must be -- every non-zero step in {-1,0,1}^2 for the
        // eight, the four axis steps for the four -- because a dropped minus sign turns a path into a
        // duplicate of its opposite, which no smooth-scene test can see.
        assert_eq!(DIRECTIONS_8.len(), 8);
        for (dy, dx) in DIRECTIONS_8 {
            assert!((-1..=1).contains(&dy) && (-1..=1).contains(&dx));
            assert!((dy, dx) != (0, 0));
            assert!(
                DIRECTIONS_8.contains(&(-dy, -dx)),
                "({dy}, {dx}) has no opposite"
            );
        }
        for i in 0..8 {
            for j in i + 1..8 {
                assert_ne!(DIRECTIONS_8[i], DIRECTIONS_8[j], "duplicate direction");
            }
        }
        assert_eq!(DIRECTIONS_4.len(), 4);
        for (dy, dx) in DIRECTIONS_4 {
            assert_eq!(
                dy.abs() + dx.abs(),
                1,
                "({dy}, {dx}) is not a cardinal step"
            );
            assert!(DIRECTIONS_4.contains(&(-dy, -dx)));
        }
    }

    #[test]
    fn the_census_bits_are_the_brighter_neighbours_in_window_scan_order() {
        // Bit by bit, most significant first, scanning dy then dx over the window and skipping the
        // centre. On PATCH the centre pixel is 5, so the neighbours 1, 9, 2, 8, 7, 3, 0, 6 give
        // 0 1 0 1 1 0 0 1.
        assert_eq!(census_at(&PATCH, 3, 3, 1, 1, 1), 0b0101_1001);
        // The corner does two jobs. It pins the clamp -- offsets that leave the image repeat the edge
        // pixel, so (0,0) is its own neighbour three times over -- and it pins the comparison as
        // strict: those three neighbours equal the centre exactly and their bits stay clear.
        assert_eq!(census_at(&PATCH, 3, 3, 0, 0, 1), 0b0010_1111);
    }

    #[test]
    fn a_disparity_that_runs_off_the_image_costs_the_whole_census_word() {
        // Two facts the matcher rests on and no scene test can reach: a pixel whose match falls off
        // the right edge is charged the maximum a census word can cost -- one per bit, (2r+1)^2 - 1 --
        // and the cost is the Hamming *distance*, so identical descriptors cost nothing. Both radii
        // are checked because the arithmetic on n_bits has forms that agree with it at one radius and
        // not the other.
        for (radius, n_bits) in [(1usize, 8.0), (2, 24.0)] {
            let cfg = SgmConfig {
                d_max: 3,
                radius,
                p1: 1.0,
                p2: 4.0,
            };
            let cost = census_cost_volume(&PATCH, &PATCH, 3, 3, &cfg);
            let middle = 3usize; // (y, x) = (1, 0), the first pixel of the middle row
            for (x, d) in [(0usize, 1usize), (0, 2), (1, 2)] {
                assert_eq!(
                    cost[(middle + x) * 3 + d],
                    n_bits,
                    "radius {radius}, x {x}, d {d}"
                );
            }
            assert_eq!(cost[(middle + 1) * 3], 0.0, "radius {radius}");
        }
    }

    #[test]
    fn each_path_is_walked_from_its_own_edge_forwards() {
        // The same closed form as the left-to-right path, turned through ninety degrees: a 3x1 column
        // aggregated downwards. Turning it is the whole point -- with one row, reversing the row
        // traversal, or reading the predecessor from the wrong side of the pixel, changes nothing, so
        // a horizontal fixture cannot see either. C[y][d]: y0 {0,4}, y1 {3,0}, y2 {0,5}.
        let cost = vec![0.0, 4.0, 3.0, 0.0, 0.0, 5.0];
        let cfg = SgmConfig {
            d_max: 2,
            radius: 1,
            p1: 1.0,
            p2: 4.0,
        };
        // y0 (border): L = C = {0, 4}.
        // y1: min_prev = 0. d0: 3 + min(0, 4+1, 0+4) - 0 = 3. d1: 0 + min(4, 0+1, 4) - 0 = 1.
        // y2: min_prev = 1. d0: 0 + min(3, 1+1, 1+4) - 1 = 1. d1: 5 + min(1, 3+1, 5) - 1 = 5.
        assert_eq!(
            aggregate_sgm(&cost, 3, 1, &cfg, &[(1, 0)]),
            vec![0.0, 4.0, 3.0, 1.0, 1.0, 5.0]
        );
    }

    #[test]
    fn the_subpixel_disparity_is_the_vertex_of_the_parabola_through_its_neighbours() {
        // S(d) = 2(d - 1.25)^2 sampled at d = 0, 1, 2, in binary fractions so the arithmetic is exact.
        // The refinement must return the vertex itself.
        let disp = disparity_wta(&[3.125, 0.125, 1.125], 1, 1, 3);
        assert_eq!(disp[0], 1.25);
        // No parabola without two neighbours: at either end of the search range the integer winner
        // stands, and reaching past it would index off the volume.
        assert_eq!(disparity_wta(&[1.0, 5.0, 9.0], 1, 1, 3)[0], 0.0);
        assert_eq!(disparity_wta(&[9.0, 5.0, 1.0], 1, 1, 3)[0], 2.0);
        // Equal costs go to the lower disparity: the winner here is index 0, which is on the boundary
        // and so is returned whole. Taking the later one instead would refine and return 0.5.
        assert_eq!(disparity_wta(&[3.0, 3.0, 9.0], 1, 1, 3)[0], 0.0);
        // The curvature guard is exclusive, and its boundary is exactly reachable: c0 + c2 - 2c1 is
        // 1e-12 to the bit here, so the refinement is declined and the integer winner stands. Admitting
        // it would move the answer by half a disparity.
        assert_eq!(disparity_wta(&[1e-12, 0.0, 0.0], 1, 1, 3)[0], 1.0);
    }

    #[test]
    fn a_vertical_pass_steps_from_the_pixel_directly_above() {
        // Three rows, so the predecessor row is 2 by the last step: an index that folded the row
        // into the column arithmetic would still be right at row 1 and wrong from row 2 on. Two
        // columns, so the fold is observable at all.
        let (h, w, d_max) = (3usize, 2usize, 2usize);
        let cfg = SgmConfig {
            d_max,
            radius: 1,
            p1: 1.0,
            p2: 4.0,
        };
        let cost: Vec<f64> = (0..h * w * d_max).map(|i| ((i * 7) % 11) as f64).collect();
        let got = aggregate_sgm(&cost, h, w, &cfg, &[(1, 0)]);

        // L(y,x,d) = C(y,x,d) + min[L', L'±1 + p1, min_k L'_k + p2] − min_k L'_k, with L' the row
        // above at the same column; the first row has no predecessor and keeps its raw cost.
        let at = |y: usize, x: usize, d: usize| (y * w + x) * d_max + d;
        let mut want = vec![0.0f64; h * w * d_max];
        for x in 0..w {
            for d in 0..d_max {
                want[at(0, x, d)] = cost[at(0, x, d)];
            }
            for y in 1..h {
                let prev: Vec<f64> = (0..d_max).map(|k| want[at(y - 1, x, k)]).collect();
                let m = prev.iter().copied().fold(f64::INFINITY, f64::min);
                for d in 0..d_max {
                    let mut best = prev[d].min(m + cfg.p2);
                    if d > 0 {
                        best = best.min(prev[d - 1] + cfg.p1);
                    }
                    if d + 1 < d_max {
                        best = best.min(prev[d + 1] + cfg.p1);
                    }
                    want[at(y, x, d)] = cost[at(y, x, d)] + best - m;
                }
            }
        }
        assert_eq!(got, want);
    }
}
