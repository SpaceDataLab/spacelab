//! The feature front-end of a photogrammetric pipeline — Harris corner detection and normalised
//! cross-correlation matching — that turns image pairs into the tie-point correspondences bundle
//! adjustment consumes.
//!
//! [`detect_corners`] finds repeatable interest points by the **Harris** response
//! `R = det(M) − k·tr(M)²` of the structure tensor `M = Σ_w [[Ix², IxIy],[IxIy, Iy²]]` (Sobel
//! gradients, a box window), keeping the local maxima above a relative threshold by non-maximum
//! suppression. [`match_ncc`] pairs corners between two images by the **zero-mean normalised
//! cross-correlation** of their patches, with a ratio test against the second-best candidate to reject
//! ambiguous matches. The resulting `(pixel_a, pixel_b)` pairs are the observations of a shared 3-D
//! point in two views — exactly what [`crate::camera::triangulate`] / [`crate::camera::bundle_adjust`]
//! take. `no_std` + `alloc`.

// Physical/tensor single letters (Ix, Iy, M, R, k) and raster casts mirror the vision literature.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::similar_names
)]

use alloc::vec;
use alloc::vec::Vec;

/// A lightweight row-major grayscale image view (`h·w`, `f64`).
#[derive(Clone, Copy, Debug)]
pub struct Image<'a> {
    /// Row-major pixel data (`h·w`).
    pub data: &'a [f64],
    /// Height (rows).
    pub h: usize,
    /// Width (columns).
    pub w: usize,
}

impl Image<'_> {
    /// Sample with replicate (edge-clamp) border, so gradients and patches are defined everywhere.
    #[inline]
    fn at(&self, y: isize, x: isize) -> f64 {
        let yy = y.clamp(0, self.h as isize - 1) as usize;
        let xx = x.clamp(0, self.w as isize - 1) as usize;
        self.data[yy * self.w + xx]
    }
}

/// A detected corner: sub-pixel-free integer location (`x` = column, `y` = row) and its Harris response.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// Column (pixels).
    pub x: f64,
    /// Row (pixels).
    pub y: f64,
    /// Harris response at the corner.
    pub response: f64,
}

/// Harris detector settings.
#[derive(Clone, Copy, Debug)]
pub struct HarrisConfig {
    /// Harris sensitivity `k` (typically 0.04–0.06).
    pub k: f64,
    /// Structure-tensor box-window radius.
    pub window_radius: usize,
    /// Keep only responses above `threshold_rel · max_response`.
    pub threshold_rel: f64,
    /// Non-maximum-suppression radius (Chebyshev, pixels).
    pub nms_radius: usize,
    /// Cap on the number of corners returned (strongest first).
    pub max_corners: usize,
}

/// A corner-to-corner match: indices into the two corner lists and the NCC score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FeatureMatch {
    /// Index into the first image's corner list.
    pub a: usize,
    /// Index into the second image's corner list.
    pub b: usize,
    /// Zero-mean NCC score in `[−1, 1]`.
    pub score: f64,
}

/// NCC matcher settings.
#[derive(Clone, Copy, Debug)]
pub struct NccConfig {
    /// Correlation patch radius (patch side `2·patch_radius+1`).
    pub patch_radius: usize,
    /// Minimum NCC score to accept a match.
    pub min_score: f64,
    /// Reject if the second-best score exceeds `max_ratio · best` (Lowe-style ratio test).
    pub max_ratio: f64,
}

/// The Sobel gradient images `(Ix, Iy)` of `img` (replicate border), row-major `h·w`.
fn sobel(img: Image) -> (Vec<f64>, Vec<f64>) {
    let (h, w) = (img.h, img.w);
    let mut ix = vec![0.0; h * w];
    let mut iy = vec![0.0; h * w];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let p = |dy, dx| img.at(y + dy, x + dx);
            let gx = -p(-1, -1) + p(-1, 1) - 2.0 * p(0, -1) + 2.0 * p(0, 1) - p(1, -1) + p(1, 1);
            let gy = -p(-1, -1) - 2.0 * p(-1, 0) - p(-1, 1) + p(1, -1) + 2.0 * p(1, 0) + p(1, 1);
            ix[y as usize * w + x as usize] = gx;
            iy[y as usize * w + x as usize] = gy;
        }
    }
    (ix, iy)
}

/// The Harris response map `R = det(M) − k·tr(M)²` (row-major `h·w`), with the structure tensor summed
/// over a `(2·window_radius+1)²` box window (replicate border).
#[must_use]
pub fn harris_response(img: Image, k: f64, window_radius: usize) -> Vec<f64> {
    let (h, w) = (img.h, img.w);
    let (ix, iy) = sobel(img);
    let g = |v: &[f64], y: isize, x: isize| -> f64 {
        let yy = y.clamp(0, h as isize - 1) as usize;
        let xx = x.clamp(0, w as isize - 1) as usize;
        v[yy * w + xx]
    };
    let rw = window_radius as isize;
    let mut resp = vec![0.0; h * w];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
            for dy in -rw..=rw {
                for dx in -rw..=rw {
                    let a = g(&ix, y + dy, x + dx);
                    let b = g(&iy, y + dy, x + dx);
                    sxx += a * a;
                    syy += b * b;
                    sxy += a * b;
                }
            }
            let trace = sxx + syy;
            resp[y as usize * w + x as usize] = sxx * syy - sxy * sxy - k * trace * trace;
        }
    }
    resp
}

/// Detect Harris corners: response, relative threshold, then greedy non-maximum suppression keeping the
/// strongest [`HarrisConfig::max_corners`] separated by [`HarrisConfig::nms_radius`].
#[must_use]
pub fn detect_corners(img: Image, cfg: &HarrisConfig) -> Vec<Corner> {
    let (h, w) = (img.h, img.w);
    let resp = harris_response(img, cfg.k, cfg.window_radius);
    let max_r = resp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let thresh = cfg.threshold_rel * max_r;
    let mut cand: Vec<Corner> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let r = resp[y * w + x];
            if r > thresh {
                cand.push(Corner {
                    x: x as f64,
                    y: y as f64,
                    response: r,
                });
            }
        }
    }
    cand.sort_by(|a, b| b.response.total_cmp(&a.response));
    let nms = cfg.nms_radius as f64;
    let mut kept: Vec<Corner> = Vec::new();
    for c in cand {
        if kept.len() >= cfg.max_corners {
            break;
        }
        if kept
            .iter()
            .all(|k| (k.x - c.x).abs().max((k.y - c.y).abs()) > nms)
        {
            kept.push(c);
        }
    }
    kept
}

/// Extract a zero-mean, L2-normalised patch of radius `r` centred at `(cy, cx)` (replicate border). A
/// flat patch (zero variance) returns all zeros (it will never exceed the match threshold).
fn normalized_patch(img: Image, cy: isize, cx: isize, r: isize) -> Vec<f64> {
    let side = (2 * r + 1) as usize;
    let mut patch = vec![0.0; side * side];
    let mut mean = 0.0;
    for dy in -r..=r {
        for dx in -r..=r {
            let v = img.at(cy + dy, cx + dx);
            patch[((dy + r) * (2 * r + 1) + (dx + r)) as usize] = v;
            mean += v;
        }
    }
    mean /= (side * side) as f64;
    let mut norm = 0.0;
    for v in &mut patch {
        *v -= mean;
        norm += *v * *v;
    }
    // A flat patch is already all zeros after the mean subtraction (never exceeds the match threshold).
    let norm = libm::sqrt(norm);
    if norm > 1e-12 {
        for v in &mut patch {
            *v /= norm;
        }
    }
    patch
}

/// Match corners between two images by zero-mean NCC of their patches, with a ratio test. Each `a`
/// corner takes its best `b` corner if the score clears [`NccConfig::min_score`] and beats the
/// second-best by the ratio test.
#[must_use]
pub fn match_ncc(
    a: Image,
    ca: &[Corner],
    b: Image,
    cb: &[Corner],
    cfg: &NccConfig,
) -> Vec<FeatureMatch> {
    let r = cfg.patch_radius as isize;
    let pb: Vec<Vec<f64>> = cb
        .iter()
        .map(|c| normalized_patch(b, c.y as isize, c.x as isize, r))
        .collect();
    let mut matches = Vec::new();
    for (ia, c) in ca.iter().enumerate() {
        let pa = normalized_patch(a, c.y as isize, c.x as isize, r);
        let (mut best, mut best_j) = (f64::NEG_INFINITY, usize::MAX);
        let mut second = f64::NEG_INFINITY;
        for (jb, patch) in pb.iter().enumerate() {
            let score: f64 = pa.iter().zip(patch).map(|(u, v)| u * v).sum();
            if score > best {
                second = best;
                best = score;
                best_j = jb;
            } else if score > second {
                second = score;
            }
        }
        if best >= cfg.min_score && second <= cfg.max_ratio * best {
            matches.push(FeatureMatch {
                a: ia,
                b: best_j,
                score: best,
            });
        }
    }
    matches
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::unwrap_used,
        clippy::cast_lossless
    )]
    use super::*;
    use crate::camera::{PinholeCamera, triangulate};
    use spacelab_core::{Mat3, Vec3};

    // The deterministic non-separable pattern the numpy Harris oracle used (no RNG, identical here).
    fn pattern(h: usize, w: usize) -> Vec<f64> {
        (0..h * w)
            .map(|i| ((i / w * 7 + i % w * 13 + (i / w) * (i % w) * 3) % 19) as f64)
            .collect()
    }

    #[test]
    fn harris_response_matches_the_numpy_oracle() {
        // Same 12×12 pattern, k=0.04, window radius 1; response at six pixels vs the numpy reference
        // (/tmp/harris_oracle.py) — exercises the exact Sobel + box-window + R = det − k·tr² arithmetic.
        let (h, w) = (12usize, 12usize);
        let data = pattern(h, w);
        let img = Image { data: &data, h, w };
        let resp = harris_response(img, 0.04, 1);
        let oracle = [
            (3, 3, 4010823.84),
            (5, 7, 1458992.0),
            (6, 6, 5026402.5600000005),
            (8, 4, 8493806.56),
            (2, 9, 10222759.36),
            (9, 9, 3674266.56),
        ];
        for (y, x, r) in oracle {
            let got = resp[y * w + x];
            assert!(
                (got - r).abs() < 1e-6 * r.abs().max(1.0),
                "R[{y},{x}] {got} vs numpy {r}"
            );
        }
    }

    #[test]
    fn detects_the_four_corners_of_a_square_like_skimage() {
        // A bright square on black at rows 12..27, cols 14..29. skimage corner_peaks locates its four
        // corners at (12,14),(12,29),(27,14),(27,29); our detector must find the same four.
        let (h, w) = (40usize, 40usize);
        let mut data = vec![0.0; h * w];
        for y in 12..28 {
            for x in 14..30 {
                data[y * w + x] = 1.0;
            }
        }
        let img = Image { data: &data, h, w };
        let corners = detect_corners(
            img,
            &HarrisConfig {
                k: 0.04,
                window_radius: 1,
                threshold_rel: 0.2,
                nms_radius: 3,
                max_corners: 8,
            },
        );
        let expect = [(12, 14), (12, 29), (27, 14), (27, 29)];
        for (ey, ex) in expect {
            let hit = corners
                .iter()
                .any(|c| (c.y - ey as f64).abs() <= 1.0 && (c.x - ex as f64).abs() <= 1.0);
            assert!(hit, "no corner near ({ey},{ex}); got {corners:?}");
        }
        assert!(corners.len() <= 8);
    }

    #[test]
    fn ncc_recovers_a_known_shift() {
        // B is A shifted so a feature at (y,x) in A sits at (y,x−d) in B (a rectified stereo pair of a
        // fronto-parallel plane). NCC must pair each corner to the one d columns left, disparity ≈ d.
        let (h, w) = (48usize, 64usize);
        let d = 6usize;
        let mut seed = 12345u64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f64) / ((1u64 << 24) as f64)
        };
        let a: Vec<f64> = (0..h * w).map(|_| rng()).collect();
        let mut b = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                b[y * w + x] = if x + d < w { a[y * w + (x + d)] } else { rng() };
            }
        }
        let ia = Image { data: &a, h, w };
        let ib = Image { data: &b, h, w };
        let cfg = HarrisConfig {
            k: 0.04,
            window_radius: 2,
            threshold_rel: 0.05,
            nms_radius: 4,
            max_corners: 60,
        };
        let ca = detect_corners(ia, &cfg);
        let cb = detect_corners(ib, &cfg);
        let matches = match_ncc(
            ia,
            &ca,
            ib,
            &cb,
            &NccConfig {
                patch_radius: 4,
                min_score: 0.9,
                max_ratio: 0.9,
            },
        );
        assert!(matches.len() >= 10, "only {} matches", matches.len());
        // Every accepted match, on the interior, must show disparity ≈ d.
        let mut checked = 0;
        for m in &matches {
            let (ax, ay) = (ca[m.a].x, ca[m.a].y);
            let bx = cb[m.b].x;
            if ax >= (d + 5) as f64 && ax < (w - 5) as f64 {
                assert!(
                    (ax - bx - d as f64).abs() <= 1.0,
                    "match at ({ax},{ay}) disparity {} vs {d}",
                    ax - bx
                );
                checked += 1;
            }
        }
        assert!(checked >= 5, "only {checked} interior matches checked");
    }

    #[test]
    fn matched_features_triangulate_to_the_correct_depth() {
        // The same shifted-texture pair is a rectified stereo view of a plane at depth Z = f·B/d. Feed
        // the NCC correspondences into the camera model: every point must triangulate to that depth,
        // closing images → features → matches → 3-D structure for bundle adjustment.
        let (h, w) = (48usize, 64usize);
        let d = 6usize;
        let (focal, baseline) = (1000.0, 3.0);
        let z_truth = focal * baseline / d as f64; // 500
        let mut seed = 99u64;
        let mut rng = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f64) / ((1u64 << 24) as f64)
        };
        let a: Vec<f64> = (0..h * w).map(|_| rng()).collect();
        let mut b = vec![0.0; h * w];
        for y in 0..h {
            for x in 0..w {
                b[y * w + x] = if x + d < w { a[y * w + (x + d)] } else { rng() };
            }
        }
        let ia = Image { data: &a, h, w };
        let ib = Image { data: &b, h, w };
        let cfg = HarrisConfig {
            k: 0.04,
            window_radius: 2,
            threshold_rel: 0.05,
            nms_radius: 4,
            max_corners: 60,
        };
        let ca = detect_corners(ia, &cfg);
        let cb = detect_corners(ib, &cfg);
        let matches = match_ncc(
            ia,
            &ca,
            ib,
            &cb,
            &NccConfig {
                patch_radius: 4,
                min_score: 0.9,
                max_ratio: 0.9,
            },
        );
        let cam = |center| PinholeCamera {
            rotation: Mat3::IDENTITY,
            center,
            focal,
            principal: [0.0, 0.0],
        };
        let left = cam(Vec3::new(0.0, 0.0, 0.0));
        let right = cam(Vec3::new(baseline, 0.0, 0.0));
        let mut n = 0;
        for m in &matches {
            let pa = [ca[m.a].x, ca[m.a].y];
            let pb = [cb[m.b].x, cb[m.b].y];
            if pa[0] < (d + 5) as f64 || pa[0] >= (w - 5) as f64 {
                continue;
            }
            let world = triangulate(&[(&left, pa), (&right, pb)]).unwrap();
            assert!(
                (world.z - z_truth).abs() < 0.2 * z_truth,
                "triangulated Z {} vs {z_truth}",
                world.z
            );
            n += 1;
        }
        assert!(n >= 5, "only {n} points triangulated");
    }

    #[test]
    fn sobel_is_the_kernel_it_is_named_after() {
        // `sobel` is private and its only consumer squares everything it returns, so the *sign* of
        // either gradient is invisible downstream: `Ix²`, `Iy²` and `(ΣIxIy)²` are all even in it.
        // State the two kernels here instead. On a field linear in the pixel, `f = a·y + b·x`, the
        // weights on either side of the centre sum to 4, so `gx = 8b` and `gy = 8a` exactly.
        let (h, w) = (5usize, 5usize);
        let (a, b) = (3.0, 2.0);
        let data: Vec<f64> = (0..h * w)
            .map(|k| a * (k / w) as f64 + b * (k % w) as f64)
            .collect();
        let (ix, iy) = sobel(Image { data: &data, h, w });
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                assert!((ix[y * w + x] - 8.0 * b).abs() < 1e-12, "Ix[{y},{x}]");
                assert!((iy[y * w + x] - 8.0 * a).abs() < 1e-12, "Iy[{y},{x}]");
            }
        }
    }

    #[test]
    fn a_normalised_patch_is_the_neighbourhood_centred_and_scaled_in_raster_order() {
        // `match_ncc` only ever forms dot products of two patches, and a patch permuted the same way
        // on both sides gives the same score — so the raster index, its stride, the running mean and
        // the centring were all unobservable. State the patch itself.
        let (h, w) = (5usize, 5usize);
        let data: Vec<f64> = (0..h * w).map(|k| k as f64).collect();
        let img = Image { data: &data, h, w };
        // The 3×3 neighbourhood of (2,2) is 6,7,8 / 11,12,13 / 16,17,18; its mean is 12 and the
        // deviations square-sum to 156.
        let p = normalized_patch(img, 2, 2, 1);
        let want = [-6.0, -5.0, -4.0, -1.0, 0.0, 1.0, 4.0, 5.0, 6.0];
        assert_eq!(p.len(), 9);
        for (k, w0) in want.iter().enumerate() {
            assert!(
                (p[k] - w0 / libm::sqrt(156.0)).abs() < 1e-12,
                "patch[{k}] = {}",
                p[k]
            );
        }
        // A flat neighbourhood is all zeros, as documented — the guard, not a division by zero.
        let flat = vec![4.0; h * w];
        let z = normalized_patch(Image { data: &flat, h, w }, 2, 2, 1);
        assert!(z.iter().all(|&v| v == 0.0), "{z:?}");
    }

    #[test]
    fn the_corner_threshold_is_relative_to_the_strongest_response_and_strict() {
        // Both existing detector tests locate corners to within a pixel and cap the count, which
        // cannot see the threshold being formed by division instead of multiplication, nor whether
        // the comparison against it is strict.
        let (h, w) = (40usize, 40usize);
        let mut data = vec![0.0; h * w];
        for y in 12..28 {
            for x in 14..30 {
                data[y * w + x] = 1.0;
            }
        }
        let img = Image { data: &data, h, w };
        let cfg = |threshold_rel| HarrisConfig {
            k: 0.04,
            window_radius: 1,
            threshold_rel,
            nms_radius: 0,
            max_corners: h * w,
        };
        let max_r = harris_response(img, 0.04, 1)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
        for threshold_rel in [0.2, 0.5, 0.9] {
            let kept = detect_corners(img, &cfg(threshold_rel));
            assert!(!kept.is_empty() && kept.len() < h * w, "{threshold_rel}");
            for c in &kept {
                assert!(
                    c.response > threshold_rel * max_r,
                    "{c:?} at {threshold_rel}"
                );
            }
        }
        // Raising the bar cannot admit more corners, and at the maximum itself nothing is *above* it.
        assert!(detect_corners(img, &cfg(0.9)).len() < detect_corners(img, &cfg(0.2)).len());
        assert_eq!(detect_corners(img, &cfg(1.0)).len(), 0);
    }

    #[test]
    fn suppression_keeps_corners_strictly_further_apart_than_its_radius() {
        // The four corners of the square sit at (12,14), (12,29), (27,14), (27,29) — every pair
        // exactly 15 apart in the Chebyshev metric the suppressor uses. That is the one separation at
        // which `>` and `>=` disagree, and it is also large enough that replacing either difference
        // with a sum turns every pair into a distant one.
        let (h, w) = (40usize, 40usize);
        let mut data = vec![0.0; h * w];
        for y in 12..28 {
            for x in 14..30 {
                data[y * w + x] = 1.0;
            }
        }
        let img = Image { data: &data, h, w };
        let cfg = |nms_radius| HarrisConfig {
            k: 0.04,
            window_radius: 1,
            threshold_rel: 0.2,
            nms_radius,
            max_corners: 64,
        };
        for nms_radius in [3usize, 14, 15] {
            let kept = detect_corners(img, &cfg(nms_radius));
            for (i, ci) in kept.iter().enumerate() {
                for cj in &kept[i + 1..] {
                    let sep = (ci.x - cj.x).abs().max((ci.y - cj.y).abs());
                    assert!(
                        sep > nms_radius as f64,
                        "{ci:?} and {cj:?} at r={nms_radius}"
                    );
                }
            }
        }
        assert_eq!(detect_corners(img, &cfg(14)).len(), 4);
        assert!(detect_corners(img, &cfg(15)).len() < 4);
    }

    #[test]
    fn the_ratio_test_rejects_a_second_candidate_as_good_as_the_best() {
        // Nothing stated the ratio test's arithmetic: the shift fixture's matches are unambiguous, so
        // the second-best score never mattered and neither did what it was compared against.
        //
        // Hand-built patches with an exactly known correlation. `u` is the reference, `m` correlates
        // with it at 6/√(6·12) = 1/√2, and `o` is orthogonal to it. Scores away from 1 matter: at a
        // best of 1 the mutated thresholds `max_ratio·best` and `max_ratio/best` coincide.
        #[rustfmt::skip]
        let u = [ 1.0,  1.0,  1.0,   0.0, 0.0,  0.0,  -1.0, -1.0, -1.0];
        #[rustfmt::skip]
        let m = [ 2.0,  1.0,  0.0,   1.0, 0.0, -1.0,   0.0, -1.0, -2.0];
        #[rustfmt::skip]
        let o = [ 1.0,  0.0, -1.0,   1.0, 0.0, -1.0,   1.0,  0.0, -1.0];
        // Three 3×3 windows side by side, sampled at x = 1, 4, 7 so none of them overlap.
        let tile = |blocks: [&[f64; 9]; 3]| {
            let mut v = vec![0.0; 27];
            for (bi, block) in blocks.iter().enumerate() {
                for r in 0..3 {
                    for c in 0..3 {
                        v[r * 9 + bi * 3 + c] = block[r * 3 + c];
                    }
                }
            }
            v
        };
        let a_data = u.to_vec();
        let a = Image {
            data: &a_data,
            h: 3,
            w: 3,
        };
        let ca = [Corner {
            x: 1.0,
            y: 1.0,
            response: 1.0,
        }];
        let cb: Vec<Corner> = (0..3)
            .map(|i| Corner {
                x: (1 + 3 * i) as f64,
                y: 1.0,
                response: 1.0,
            })
            .collect();
        let cfg = NccConfig {
            patch_radius: 1,
            min_score: 0.5,
            max_ratio: 0.8,
        };
        // One good candidate and two orthogonal ones: best 1/√2, second 0, accepted.
        let clear = tile([&m, &o, &o]);
        let matches = match_ncc(
            a,
            &ca,
            Image {
                data: &clear,
                h: 3,
                w: 9,
            },
            &cb,
            &cfg,
        );
        assert_eq!(matches.len(), 1, "{matches:?}");
        assert_eq!(matches[0].b, 0, "{matches:?}");
        assert!(
            (matches[0].score - 1.0 / libm::sqrt(2.0)).abs() < 1e-12,
            "{matches:?}"
        );
        // Two candidates equally good: best and second both 1/√2, so `second ≤ 0.8·best` fails and
        // the match is thrown away. Under `+` the bound becomes 1.51 and under `/` it becomes 1.13,
        // and both admit it.
        let tied = tile([&m, &m, &o]);
        let matches = match_ncc(
            a,
            &ca,
            Image {
                data: &tied,
                h: 3,
                w: 9,
            },
            &cb,
            &cfg,
        );
        assert!(matches.is_empty(), "{matches:?}");
    }
}
