//! Object-detection post-processing — the head that turns dense per-anchor predictions into a short
//! list of boxes: anchor generation, box decoding, IoU, and non-maximum suppression.
//!
//! A single-stage detector predicts, at every feature-map cell and for every prior box (**anchor**), a
//! class score and a box **delta**. [`generate_anchors`] lays the anchors over the feature map;
//! [`decode_box`] applies a predicted delta to an anchor (the Faster-R-CNN parameterisation
//! `x = xₐ + dx·wₐ`, `w = wₐ·e^{dw}`); and [`nms`] keeps, per the greedy non-maximum-suppression rule,
//! the highest-scoring boxes while discarding those that overlap a kept box by more than an
//! [`BBox::iou`] threshold. `alloc` for the box lists (built on the `std`/`test` side, like
//! [`crate::model`]).

// doc names detector families (Faster-R-CNN, RetinaNet); single letters are box-geometry notation;
// feature-map indices cast to f64 for pixel centres.
#![allow(
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::cast_precision_loss
)]

/// An axis-aligned box in pixel coordinates, corners `(x1, y1)`–`(x2, y2)` with `x1 ≤ x2`, `y1 ≤ y2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BBox {
    /// Left edge.
    pub x1: f64,
    /// Top edge.
    pub y1: f64,
    /// Right edge.
    pub x2: f64,
    /// Bottom edge.
    pub y2: f64,
}

impl BBox {
    /// Construct from corners.
    #[must_use]
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        Self { x1, y1, x2, y2 }
    }

    /// Construct from centre `(cx, cy)` and size `(w, h)`.
    #[must_use]
    pub fn from_center(cx: f64, cy: f64, w: f64, h: f64) -> Self {
        Self {
            x1: cx - 0.5 * w,
            y1: cy - 0.5 * h,
            x2: cx + 0.5 * w,
            y2: cy + 0.5 * h,
        }
    }

    /// Width.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.x2 - self.x1
    }

    /// Height.
    #[must_use]
    pub fn height(&self) -> f64 {
        self.y2 - self.y1
    }

    /// Area (non-negative), `width·height`.
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width().max(0.0) * self.height().max(0.0)
    }

    /// Intersection-over-union with `other`, in `[0, 1]` (the torchvision `box_iou` convention: no
    /// `+1` on the extents).
    #[must_use]
    pub fn iou(&self, other: &Self) -> f64 {
        let ix1 = self.x1.max(other.x1);
        let iy1 = self.y1.max(other.y1);
        let ix2 = self.x2.min(other.x2);
        let iy2 = self.y2.min(other.y2);
        let inter = (ix2 - ix1).max(0.0) * (iy2 - iy1).max(0.0);
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }
}

/// Greedy non-maximum suppression: keep the highest-scoring box, discard every remaining box whose IoU
/// with it exceeds `iou_threshold`, and repeat. Returns the kept indices in descending score order —
/// the `torchvision.ops.nms` contract.
#[must_use]
pub fn nms(boxes: &[BBox], scores: &[f64], iou_threshold: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..boxes.len()).collect();
    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    let mut suppressed = vec![false; boxes.len()];
    let mut keep = Vec::new();
    for i in 0..order.len() {
        let a = order[i];
        if suppressed[a] {
            continue;
        }
        keep.push(a);
        for &b in &order[i + 1..] {
            if !suppressed[b] && boxes[a].iou(&boxes[b]) > iou_threshold {
                suppressed[b] = true;
            }
        }
    }
    keep
}

/// Lay anchor boxes over an `feat_h × feat_w` feature map at pixel `stride`. Each cell centre is
/// `((j+0.5)·stride, (i+0.5)·stride)`; for every `scale` (√area) and aspect `ratio` (= height/width) an
/// anchor of `w = scale/√ratio`, `h = scale·√ratio` is emitted. Row-major over cells, then `scales`,
/// then `aspect_ratios`.
#[must_use]
pub fn generate_anchors(
    feat_h: usize,
    feat_w: usize,
    stride: f64,
    scales: &[f64],
    aspect_ratios: &[f64],
) -> Vec<BBox> {
    let mut anchors = Vec::with_capacity(feat_h * feat_w * scales.len() * aspect_ratios.len());
    for i in 0..feat_h {
        for j in 0..feat_w {
            let cx = (j as f64 + 0.5) * stride;
            let cy = (i as f64 + 0.5) * stride;
            for &scale in scales {
                for &ratio in aspect_ratios {
                    let sr = libm::sqrt(ratio);
                    anchors.push(BBox::from_center(cx, cy, scale / sr, scale * sr));
                }
            }
        }
    }
    anchors
}

/// Decode a predicted box `delta = [dx, dy, dw, dh]` relative to `anchor` — the Faster-R-CNN / SSD
/// parameterisation `x = xₐ + dx·wₐ`, `y = yₐ + dy·hₐ`, `w = wₐ·e^{dw}`, `h = hₐ·e^{dh}` (centre form).
#[must_use]
pub fn decode_box(anchor: &BBox, delta: [f64; 4]) -> BBox {
    let (wa, ha) = (anchor.width(), anchor.height());
    let cx = f64::midpoint(anchor.x1, anchor.x2) + delta[0] * wa;
    let cy = f64::midpoint(anchor.y1, anchor.y2) + delta[1] * ha;
    let w = wa * libm::exp(delta[2]);
    let h = ha * libm::exp(delta[3]);
    BBox::from_center(cx, cy, w, h)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unreadable_literal, clippy::identity_op)]
    use super::*;

    fn boxes() -> Vec<BBox> {
        vec![
            BBox::new(10.0, 10.0, 50.0, 50.0),
            BBox::new(12.0, 11.0, 52.0, 51.0),
            BBox::new(90.0, 90.0, 140.0, 140.0),
            BBox::new(92.0, 93.0, 141.0, 139.0),
            BBox::new(30.0, 30.0, 70.0, 70.0),
            BBox::new(200.0, 200.0, 260.0, 255.0),
        ]
    }

    #[test]
    fn iou_matches_torchvision() {
        // torchvision computes box_iou in float32, so compare at 1e-6 (our f64 value is exact).
        let b = boxes();
        assert!((b[0].iou(&b[1]) - 0.8626309633255005).abs() < 1e-6);
        assert!((b[0].iou(&b[4]) - 0.1428571492433548).abs() < 1e-6);
        assert!((b[2].iou(&b[3]) - 0.8672427535057068).abs() < 1e-6);
        assert_eq!(b[0].iou(&b[5]), 0.0); // disjoint
    }

    #[test]
    fn nms_matches_torchvision() {
        // torchvision.ops.nms on the same boxes/scores keeps [4, 0, 2, 5] at both thresholds.
        let b = boxes();
        let scores = [0.9, 0.75, 0.8, 0.6, 0.95, 0.5];
        assert_eq!(nms(&b, &scores, 0.3), vec![4, 0, 2, 5]);
        assert_eq!(nms(&b, &scores, 0.5), vec![4, 0, 2, 5]);
    }

    #[test]
    fn anchors_match_the_closed_form() {
        // 2×2 map, stride 16, one scale 32, two aspect ratios {1, 2}. Cell (0,0) centre is (8, 8).
        let a = generate_anchors(2, 2, 16.0, &[32.0], &[1.0, 2.0]);
        assert_eq!(a.len(), 2 * 2 * 1 * 2);
        // ratio 1 → 32×32 square about (8,8).
        assert_eq!(a[0], BBox::new(-8.0, -8.0, 24.0, 24.0));
        // ratio 2 → w = 32/√2, h = 32·√2 about (8,8).
        let r2 = 2.0f64.sqrt();
        assert!((a[1].width() - 32.0 / r2).abs() < 1e-12);
        assert!((a[1].height() - 32.0 * r2).abs() < 1e-12);
        assert!((f64::midpoint(a[1].x1, a[1].x2) - 8.0).abs() < 1e-12);
        // The next cell (0,1) is one stride to the right: centre (24, 8).
        assert!((f64::midpoint(a[2].x1, a[2].x2) - 24.0).abs() < 1e-12);
    }

    #[test]
    fn decode_is_the_faster_rcnn_parameterization() {
        let anchor = BBox::new(20.0, 40.0, 60.0, 100.0); // w=40, h=60, centre (40, 70)
        // Zero delta returns the anchor.
        assert_eq!(decode_box(&anchor, [0.0, 0.0, 0.0, 0.0]), anchor);
        // A known delta, computed by hand.
        let d = [0.5, -0.25, libm::log(2.0), libm::log(0.5)];
        let got = decode_box(&anchor, d);
        // cx = 40 + 0.5·40 = 60; cy = 70 − 0.25·60 = 55; w = 40·2 = 80; h = 60·0.5 = 30.
        assert!((got.width() - 80.0).abs() < 1e-9 && (got.height() - 30.0).abs() < 1e-9);
        assert!((f64::midpoint(got.x1, got.x2) - 60.0).abs() < 1e-9);
        assert!((f64::midpoint(got.y1, got.y2) - 55.0).abs() < 1e-9);
    }

    #[test]
    fn iou_of_two_collapsed_boxes_is_zero_rather_than_nan() {
        // `union > 0` is the only thing between `iou` and `0/0`, and it takes *both* boxes having
        // zero area to reach it — one positive area keeps the union positive. `decode_box` emits
        // `w = wa·e^{dw}`, which underflows to zero for a sufficiently negative predicted `dw`, so
        // a collapsed box is an ordinary detector output rather than a contrived input.
        let p = BBox::new(5.0, 5.0, 5.0, 5.0);
        assert_eq!(p.iou(&p), 0.0);
        assert_eq!(p.iou(&BBox::new(9.0, 9.0, 9.0, 9.0)), 0.0);
        // An inverted box clamps to zero area through `area()`, and reaches the same guard.
        let q = BBox::new(10.0, 10.0, 8.0, 8.0);
        assert_eq!(q.iou(&q), 0.0);
    }

    #[test]
    fn nms_suppresses_strictly_above_the_threshold() {
        // B nested in A: inter = 2, union = 4 + 2 − 2 = 4, so the IoU is 2/4 — exactly one half,
        // with no rounding, which is what puts the fixture *on* the cut. The published fixture's
        // fifteen pairwise IoUs are all far from 0.3 and 0.5, so `>` and `>=` decided alike there.
        let a = BBox::new(0.0, 0.0, 4.0, 1.0);
        let b = BBox::new(0.0, 0.0, 2.0, 1.0);
        assert_eq!(a.iou(&b), 0.5);
        // torchvision keeps `b`: suppression needs the IoU strictly greater than the cut.
        assert_eq!(nms(&[a, b], &[0.9, 0.8], 0.5), vec![0, 1]);
        // And a hair below the cut it *is* suppressed, so the line above is not vacuous.
        assert_eq!(nms(&[a, b], &[0.9, 0.8], 0.499), vec![0]);
    }
}
