//! Dense multi-view stereo by **plane sweeping** — the depth map behind a photogrammetric DSM, from a
//! reference view and one or more calibrated neighbours.
//!
//! For each reference pixel a set of depth hypotheses is swept: back-project the pixel to the 3-D point
//! at that depth (a fronto-parallel plane in the reference camera frame), project it into every
//! neighbour, and score the **photo-consistency** — the zero-mean normalised cross-correlation of the
//! reference patch against the bilinearly-sampled neighbour patch. The depth that maximises the
//! consistency (summed over neighbours) wins. This is the same collinearity model as
//! [`crate::camera::PinholeCamera`], swept over depth instead of triangulated from correspondences.
//!
//! `no_std` + `alloc`.

// Raster indices cast usize↔isize/f64; single letters (u, v, x, y, d) are image/geometry notation;
// plane_sweep_mvs takes the reference, its neighbours, the shape, and the sweep parameters.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::too_many_arguments
)]

use crate::camera::PinholeCamera;
use alloc::vec;
use alloc::vec::Vec;
use spacelab_core::Vec3;

/// Bilinear sample of an `h × w` image at `(x, y)` (column, row), or `None` if the 2×2 support falls
/// outside the image.
fn bilinear(img: &[f64], h: usize, w: usize, x: f64, y: f64) -> Option<f64> {
    if x < 0.0 || y < 0.0 {
        return None;
    }
    let (x0, y0) = (x as usize, y as usize);
    if x0 + 1 >= w || y0 + 1 >= h {
        return None;
    }
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let i = y0 * w + x0;
    let top = img[i] * (1.0 - fx) + img[i + 1] * fx;
    let bot = img[i + w] * (1.0 - fx) + img[i + w + 1] * fx;
    Some(top * (1.0 - fy) + bot * fy)
}

/// The world point on the reference pixel `(u, v)`'s ray at camera-frame depth `depth`.
fn back_project(cam: &PinholeCamera, u: f64, v: f64, depth: f64) -> Vec3 {
    let cam_pt = Vec3::new(
        depth * (u - cam.principal[0]) / cam.focal,
        depth * (v - cam.principal[1]) / cam.focal,
        depth,
    );
    cam.center + cam.rotation.transpose() * cam_pt
}

/// Zero-mean, L2-normalised patch from a sampler `f(dx, dy)`; `None` if any tap is missing.
fn zmn_patch(r: isize, mut f: impl FnMut(isize, isize) -> Option<f64>) -> Option<Vec<f64>> {
    let side = (2 * r + 1) as usize;
    let mut p = vec![0.0; side * side];
    let mut mean = 0.0;
    for dy in -r..=r {
        for dx in -r..=r {
            let val = f(dx, dy)?;
            p[((dy + r) * (2 * r + 1) + (dx + r)) as usize] = val;
            mean += val;
        }
    }
    mean /= (side * side) as f64;
    let mut norm = 0.0;
    for x in &mut p {
        *x -= mean;
        norm += *x * *x;
    }
    let norm = libm::sqrt(norm);
    if norm < 1e-12 {
        return None;
    }
    for x in &mut p {
        *x /= norm;
    }
    Some(p)
}

/// Plane-sweep dense stereo: for each pixel of the `reference` image (`h × w`, taken by `ref_cam`),
/// return the depth hypothesis from `depths` (camera-frame `z`, metres) that maximises the summed
/// zero-mean NCC photo-consistency against the `neighbors` (each an image with its camera), over a
/// `(2·patch_radius+1)²` window. Pixels with no consistent hypothesis get `0`.
#[must_use]
pub fn plane_sweep_mvs(
    reference: &[f64],
    ref_cam: &PinholeCamera,
    neighbors: &[(&[f64], &PinholeCamera)],
    h: usize,
    w: usize,
    depths: &[f64],
    patch_radius: usize,
) -> Vec<f64> {
    let r = patch_radius as isize;
    let mut depth_map = vec![0.0; h * w];
    for vv in 0..h {
        for uu in 0..w {
            // Reference patch (integer grid), zero-mean normalised — constant across depths.
            let Some(ref_patch) = zmn_patch(r, |dx, dy| {
                let (x, y) = (uu as isize + dx, vv as isize + dy);
                if x >= 0 && x < w as isize && y >= 0 && y < h as isize {
                    Some(reference[y as usize * w + x as usize])
                } else {
                    None
                }
            }) else {
                continue;
            };
            let (mut best_score, mut best_depth) = (f64::NEG_INFINITY, 0.0);
            for &depth in depths {
                let world = back_project(ref_cam, uu as f64, vv as f64, depth);
                let mut score = 0.0;
                let mut ok = true;
                for &(img, cam) in neighbors {
                    let c = cam.project(world); // patch centre in the neighbour
                    let Some(np) = zmn_patch(r, |dx, dy| {
                        bilinear(img, h, w, c[0] + dx as f64, c[1] + dy as f64)
                    }) else {
                        ok = false;
                        break;
                    };
                    score += ref_patch.iter().zip(&np).map(|(a, b)| a * b).sum::<f64>();
                }
                if ok && score > best_score {
                    best_score = score;
                    best_depth = depth;
                }
            }
            depth_map[vv * w + uu] = best_depth;
        }
    }
    depth_map
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp, clippy::unreadable_literal)]
    use super::*;
    use spacelab_core::Mat3;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f64) / ((1u64 << 24) as f64)
    }

    #[test]
    fn plane_sweep_recovers_a_fronto_parallel_plane() {
        // A rectified pair viewing a textured fronto-parallel plane at depth Z0: the neighbour is the
        // reference shifted by the disparity d0 = f·B/Z0. Sweeping depths, the true depth maximises the
        // photo-consistency, so the recovered depth is Z0 across the interior.
        let (h, w) = (32usize, 48usize);
        let (focal, cx, cy, baseline) = (1000.0, 24.0, 16.0, 5.0);
        let z0 = 500.0;
        let d0 = (focal * baseline / z0) as usize; // 10 px
        let intr = |center| PinholeCamera {
            rotation: Mat3::IDENTITY,
            center,
            focal,
            principal: [cx, cy],
        };
        let ref_cam = intr(Vec3::new(0.0, 0.0, 0.0));
        let nbr_cam = intr(Vec3::new(baseline, 0.0, 0.0));

        let mut seed = 2024u64;
        let reference: Vec<f64> = (0..h * w).map(|_| lcg(&mut seed)).collect();
        let mut neighbor = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                neighbor[y * w + x] = if x + d0 < w {
                    reference[y * w + (x + d0)]
                } else {
                    lcg(&mut seed)
                };
            }
        }

        let depths: Vec<f64> = (400..=600).step_by(20).map(f64::from).collect();
        assert!(depths.contains(&z0));
        let dm = plane_sweep_mvs(
            &reference,
            &ref_cam,
            &[(&neighbor, &nbr_cam)],
            h,
            w,
            &depths,
            2,
        );

        // Score only where every depth's neighbour patch is on-image (past the largest disparity
        // f·B/min_depth ≈ 12.5 px plus the patch radius); the left border cannot see the plane at Z0.
        let (mut ok, mut total) = (0usize, 0usize);
        for y in 3..h - 3 {
            for x in 16..(w - 4) {
                total += 1;
                if dm[y * w + x] == z0 {
                    ok += 1;
                }
            }
        }
        assert!(
            ok as f64 / total as f64 > 0.95,
            "recovered {ok}/{total} at Z0"
        );
    }

    /// Two depth hypotheses that score bit-identically, so only the tie-break decides between them.
    ///
    /// The neighbour is periodic in `x` with a period of 10 px and the two depths differ in
    /// disparity by exactly that period -- `f·B/500 = 10`, `f·B/250 = 20` -- so each patch reads the
    /// same values at the same integer offsets and the two consistency sums agree to the last bit.
    /// The sweep keeps the first of a tie; nothing else in this file can tell that from keeping the
    /// last, because every other fixture has a strict winner.
    #[test]
    fn a_tie_between_two_depth_hypotheses_is_broken_in_favour_of_the_first() {
        let (h, w) = (24usize, 60usize);
        let (focal, cx, cy, baseline) = (1000.0, 30.0, 12.0, 5.0);
        let intr = |center| PinholeCamera {
            rotation: Mat3::IDENTITY,
            center,
            focal,
            principal: [cx, cy],
        };
        let ref_cam = intr(Vec3::new(0.0, 0.0, 0.0));
        let nbr_cam = intr(Vec3::new(baseline, 0.0, 0.0));

        let mut seed = 7u64;
        let reference: Vec<f64> = (0..h * w).map(|_| lcg(&mut seed)).collect();
        let tile: Vec<f64> = (0..10 * h).map(|_| lcg(&mut seed)).collect();
        let neighbor: Vec<f64> = (0..h * w)
            .map(|i| tile[(i / w) * 10 + (i % w) % 10])
            .collect();

        let dm = plane_sweep_mvs(
            &reference,
            &ref_cam,
            &[(&neighbor, &nbr_cam)],
            h,
            w,
            &[500.0, 250.0],
            1,
        );
        for y in 2..h - 2 {
            for x in 24..w - 2 {
                assert_eq!(dm[y * w + x], 500.0, "at ({x}, {y})");
            }
        }
    }

    /// A 3x3 ramp whose four taps around any cell are distinct, so every weight and every index in
    /// the interpolation moves the answer.
    const RAMP: [f64; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

    /// `bilinear` against hand-computed values, in general position and at the nodes.
    ///
    /// The plane sweep above cannot see this function. It is a winner-take-all search over a handful
    /// of depth hypotheses, and a distorted sampler distorts every hypothesis alike, so the argmax --
    /// the only thing asserted -- survives it: twenty-four mutants of the interpolation did, among
    /// them one that reads the same tap twice and one that reads the tap to the left of the intended
    /// one. Both weights are away from 0, 1/2 and 1 so that no pair of them can be swapped unseen.
    #[test]
    fn bilinear_is_the_weighted_mean_of_the_four_taps_it_straddles() {
        // top = 1*0.75 + 2*0.25 = 1.25, bot = 4*0.75 + 5*0.25 = 4.25,
        // out = 1.25*0.25 + 4.25*0.75 = 3.5, exactly in binary.
        assert_eq!(bilinear(&RAMP, 3, 3, 0.25, 0.75), Some(3.5));
        // At a node the tap itself comes back; on an edge, the mean of the pair it lies on.
        assert_eq!(bilinear(&RAMP, 3, 3, 0.0, 0.0), Some(1.0));
        assert_eq!(bilinear(&RAMP, 3, 3, 1.0, 1.0), Some(5.0));
        assert_eq!(bilinear(&RAMP, 3, 3, 0.5, 0.0), Some(1.5));
        assert_eq!(bilinear(&RAMP, 3, 3, 0.0, 0.5), Some(2.5));
    }

    /// The sampler refuses rather than folds: a coordinate outside the grid, or one whose upper taps
    /// would be, is `None` and not a clamped or wrapped value.
    #[test]
    fn bilinear_refuses_a_sample_whose_four_taps_are_not_all_inside() {
        assert_eq!(bilinear(&RAMP, 3, 3, -1.0, 0.5), None);
        assert_eq!(bilinear(&RAMP, 3, 3, 0.5, -1.0), None);
        // Exactly on the last column and row: the second tap of each pair would be off the grid.
        assert_eq!(bilinear(&RAMP, 3, 3, 2.0, 0.5), None);
        assert_eq!(bilinear(&RAMP, 3, 3, 0.5, 2.0), None);
        assert_eq!(bilinear(&RAMP, 3, 3, 2.5, 0.5), None);
    }

    /// `zmn_patch`'s two defining properties, its scan order, and its refusal on a flat window.
    ///
    /// The sweep normalises every candidate the same way, so a patch that is scaled, offset or laid
    /// out transposed still produces the same argmax and twenty-two mutants lived through it. The
    /// taps here are `10*dy + dx + 100`: distinct per position, so the scan order is observable, and
    /// offset from zero, so subtracting the mean is too.
    fn tap(dx: isize, dy: isize) -> f64 {
        f64::from(i32::try_from(10 * dy + dx).unwrap()) + 100.0
    }

    #[test]
    fn a_zero_mean_normalised_patch_is_the_window_centred_and_scaled_to_unit_length() {
        let p = zmn_patch(1, |dx, dy| Some(tap(dx, dy))).unwrap();
        assert_eq!(p.len(), 9);

        let sum: f64 = p.iter().sum();
        let norm_sq: f64 = p.iter().map(|v| v * v).sum();
        assert!(sum.abs() < 1e-15, "mean not removed: {sum}");
        assert!((norm_sq - 1.0).abs() < 1e-15, "not unit length: {norm_sq}");

        // Scan order: row-major in dy, then dx, and the offset is gone -- so entry k carries
        // 10*(k/3 - 1) + (k%3 - 1), scaled by the length of that vector, sqrt(606).
        let scale = libm::sqrt(606.0);
        for (k, &got) in p.iter().enumerate() {
            let want = f64::from(10 * (k as i32 / 3 - 1) + (k as i32 % 3 - 1));
            assert!(
                (got * scale - want).abs() < 1e-12,
                "tap {k}: {} vs {want}",
                got * scale
            );
        }
    }

    /// A radius-2 window, because a radius-1 one cannot see the side length it is indexed by: with
    /// `r = 1` the stride `2*r + 1` is 3 whether the `2` multiplies or divides. At `r = 2` the two
    /// disagree, 5 against 2, and the taps land in different cells.
    #[test]
    fn the_patch_stride_is_the_side_length_and_not_something_that_agrees_with_it_at_radius_one() {
        let p = zmn_patch(2, |dx, dy| Some(tap(dx, dy))).unwrap();
        assert_eq!(p.len(), 25);
        let scale = norm_of_ramp(2);
        for (k, &got) in p.iter().enumerate() {
            let want = f64::from(10 * (k as i32 / 5 - 2) + (k as i32 % 5 - 2));
            assert!(
                (got * scale - want).abs() < 1e-12,
                "tap {k}: {} vs {want}",
                got * scale
            );
        }
    }

    /// The Euclidean length of the zero-mean `10*dy + dx` ramp over a `(2r+1)²` window.
    fn norm_of_ramp(r: i32) -> f64 {
        let mut s = 0.0;
        for dy in -r..=r {
            for dx in -r..=r {
                let v = f64::from(10 * dy + dx);
                s += v * v;
            }
        }
        libm::sqrt(s)
    }

    /// A window with nothing in it has no direction to normalise to, and a window with a missing tap
    /// cannot be compared at all: both are `None` rather than a patch of zeros or of garbage.
    ///
    /// The third case sits exactly on the contrast floor. Four deviations of `±5e-13` give
    /// `sqrt(4a²) = 2a`, and doubling is exact in binary, so the length is the double nearest
    /// `1e-12` to the bit -- the one value at which an inclusive floor and an exclusive one disagree.
    #[test]
    fn a_patch_with_no_contrast_or_a_missing_tap_is_refused() {
        assert!(zmn_patch(1, |_, _| Some(5.0)).is_none());
        let on_the_floor = |dx: isize, dy: isize| {
            let a = 5e-13;
            Some(match (dy, dx) {
                (-1, -1 | 0) => a,
                (-1, 1) | (0, -1) => -a,
                _ => 0.0,
            })
        };
        assert!(
            zmn_patch(1, on_the_floor).is_some(),
            "a patch exactly at the floor is above it, not on the wrong side of it"
        );
        let one_tap_missing = |dx: isize, dy: isize| {
            if (dx, dy) == (1, 1) {
                None
            } else {
                Some(tap(dx, dy))
            }
        };
        assert!(zmn_patch(1, one_tap_missing).is_none());
    }
}
