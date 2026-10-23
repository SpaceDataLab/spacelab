//! Finding what is in the frame, and measuring where it is to a fraction of a pixel.
//!
//! # The three steps, and what each one can get wrong
//!
//! 1. **Background and noise.** Estimated as the median and `1.4826 × MAD` of the whole frame, not
//!    the mean and standard deviation. A mean is pulled by the very sources being looked for — one
//!    saturated star in a sparse field moves it enough to lose the faint ones — while the median of
//!    a frame that is mostly sky is the sky. The `1.4826` makes the MAD agree with a Gaussian σ.
//! 2. **Detection.** Pixels above `background + threshold·σ`, grouped into 8-connected blobs, with
//!    blobs smaller than `min_pixels` discarded as cosmic rays or hot pixels. A single pixel can be
//!    5σ from read noise alone; three adjacent ones cannot, and that is the whole content of the
//!    minimum-size cut.
//! 3. **Centroiding.** The intensity-weighted first moment of the background-subtracted flux. It is
//!    unbiased for a symmetric profile and cheap, and it is *not* what a crowded field or an
//!    undersampled PSF wants — those need a fitted model, which needs a PSF, which is a different
//!    ticket. What is returned alongside is the uncertainty, so that a caller can tell the two cases
//!    apart rather than discovering the difference in the plate residuals.
//!
//! # The centroid uncertainty
//!
//! For a profile of width `σ_psf` measured with total signal-to-noise `Q`, the centroid error is
//! `σ_psf/Q` — the standard result, and the reason a bright star pins a plate a hundred times better
//! than a faint one at the same width. Both factors are measured from the blob itself: `σ_psf` from
//! its second moment and `Q` from the flux against the noise in the pixels the flux was summed over.
//! No PSF model, no assumed gain, and no floor: an under-sampled source reports a small `σ_psf`
//! honestly, and it is the plate residual that will disagree with it.

use alloc::vec;
use alloc::vec::Vec;

use crate::wcs::Pixel;

/// A detected source: where it is, how bright, and how well the position is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Source {
    /// Centroid, in the image's own pixel coordinates — pixel centres at integer values.
    pub centroid: Pixel,
    /// Background-subtracted flux summed over the blob, in the image's own units.
    pub flux: f64,
    /// Brightest single pixel, background-subtracted.
    pub peak: f64,
    /// Pixels above the threshold.
    pub pixel_count: usize,
    /// Gaussian-equivalent profile width from the second moment, pixels.
    pub width: f64,
    /// Centroid standard error, pixels — `width/Q` with `Q` the blob's signal-to-noise.
    pub position_sigma: f64,
}

impl Source {
    /// Signal-to-noise of the summed flux, `flux/(σ√n)`.
    #[must_use]
    pub fn signal_to_noise(&self) -> f64 {
        if self.position_sigma <= 0.0 {
            return f64::INFINITY;
        }
        self.width / self.position_sigma
    }
}

/// A frame: row-major pixels, `width` per row.
#[derive(Debug, Clone, Copy)]
pub struct Frame<'a> {
    /// Pixel values, row-major, `width × height` of them.
    pub pixels: &'a [f64],
    /// Row length.
    pub width: usize,
    /// Number of rows.
    pub height: usize,
}

impl Frame<'_> {
    /// The value at `(x, y)`, or `None` outside the frame.
    #[must_use]
    pub fn at(&self, x: usize, y: usize) -> Option<f64> {
        (x < self.width && y < self.height).then(|| self.pixels[y * self.width + x])
    }

    /// Whether the pixel count matches the stated shape — the one thing a caller can get wrong that
    /// nothing downstream would notice, since a frame one row short simply detects nothing there.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        self.pixels.len() == self.width * self.height
    }
}

/// Median and Gaussian-equivalent noise of a frame: `(median, 1.4826 × MAD)`.
///
/// `None` for an empty frame. The two passes cost a sort each; on a frame large enough for that to
/// matter, a caller samples a subset and calls this on the sample.
#[must_use]
pub fn background(pixels: &[f64]) -> Option<(f64, f64)> {
    if pixels.is_empty() {
        return None;
    }
    let median = |values: &mut Vec<f64>| -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let middle = values.len() / 2;
        if values.len().is_multiple_of(2) {
            f64::midpoint(values[middle - 1], values[middle])
        } else {
            values[middle]
        }
    };
    let mut sorted = pixels.to_vec();
    let centre = median(&mut sorted);
    let mut deviations: Vec<f64> = pixels.iter().map(|p| libm::fabs(p - centre)).collect();
    let mad = median(&mut deviations);
    Some((centre, 1.482_602_218_505_602 * mad))
}

/// Detect and centroid every source above `threshold` sigmas of the background noise, keeping blobs
/// of at least `min_pixels`.
///
/// Returns them brightest first. `None` if the frame's shape does not match its pixel count or the
/// background cannot be estimated.
///
/// The blob search is an explicit stack rather than recursion: a saturated frame is one connected
/// component of a million pixels, and a recursive flood fill would overflow the stack on exactly the
/// image a caller most wants an error from.
#[must_use]
pub fn detect(frame: &Frame, threshold: f64, min_pixels: usize) -> Option<Vec<Source>> {
    if !frame.is_consistent() {
        return None;
    }
    let (level, noise) = background(frame.pixels)?;
    if noise.is_nan() || noise <= 0.0 {
        return None;
    }
    let cut = level + threshold * noise;

    let mut visited = vec![false; frame.pixels.len()];
    let mut sources = Vec::new();
    let mut stack = Vec::new();
    let mut blob = Vec::new();

    for seed in 0..frame.pixels.len() {
        if visited[seed] || frame.pixels[seed] <= cut {
            continue;
        }
        blob.clear();
        stack.clear();
        stack.push(seed);
        visited[seed] = true;
        while let Some(index) = stack.pop() {
            blob.push(index);
            // Clamped in `usize` rather than widened to a signed type and back: the frame is
            // non-empty here, so `width - 1` and `height - 1` cannot underflow, and there is no
            // cast to reason about on a 32-bit target.
            let (x, y) = (index % frame.width, index / frame.width);
            for ny in y.saturating_sub(1)..=(y + 1).min(frame.height - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(frame.width - 1) {
                    let neighbour = ny * frame.width + nx;
                    if !visited[neighbour] && frame.pixels[neighbour] > cut {
                        visited[neighbour] = true;
                        stack.push(neighbour);
                    }
                }
            }
        }
        if blob.len() < min_pixels {
            continue;
        }
        if let Some(source) = centroid(frame, &blob, level, noise) {
            sources.push(source);
        }
    }

    sources.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    Some(sources)
}

/// First and second moments of one blob.
fn centroid(frame: &Frame, blob: &[usize], level: f64, noise: f64) -> Option<Source> {
    let mut flux = 0.0;
    let mut peak: f64 = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for &index in blob {
        let value = frame.pixels[index] - level;
        if value <= 0.0 {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let (x, y) = ((index % frame.width) as f64, (index / frame.width) as f64);
        flux += value;
        peak = peak.max(value);
        sum_x += value * x;
        sum_y += value * y;
    }
    if flux <= 0.0 {
        return None;
    }
    let (cx, cy) = (sum_x / flux, sum_y / flux);

    let mut second = 0.0;
    for &index in blob {
        let value = frame.pixels[index] - level;
        if value <= 0.0 {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let (x, y) = ((index % frame.width) as f64, (index / frame.width) as f64);
        second += value * ((x - cx) * (x - cx) + (y - cy) * (y - cy));
    }
    // Trace of the second-moment matrix is σx² + σy²; the circular-equivalent width is its half.
    let width = libm::sqrt(0.5 * second / flux);

    #[allow(clippy::cast_precision_loss)]
    let signal_to_noise = flux / (noise * libm::sqrt(blob.len() as f64));
    Some(Source {
        centroid: Pixel::new(cx, cy),
        flux,
        peak,
        pixel_count: blob.len(),
        width,
        position_sigma: if signal_to_noise > 0.0 {
            width / signal_to_noise
        } else {
            f64::INFINITY
        },
    })
}
