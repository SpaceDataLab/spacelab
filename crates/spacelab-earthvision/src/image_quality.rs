//! **Image quality** — the numbers an imaging contract is actually written in.
//!
//! The rest of this crate models *where* the light came from and *what* it did on the way. Nothing in
//! it answers "how sharp is this image", which is the question an acceptance review asks first. The
//! metrics here are the standard answers:
//!
//! * [`analyse_slanted_edge`] — the ISO 12233 slanted-edge measurement. A straight edge crossing the
//!   pixel grid at a small angle samples the edge-spread function at a different sub-pixel phase in
//!   every row; binning those rows together reconstructs the ESF at four times the pixel pitch,
//!   differentiating gives the line-spread function, and its Fourier transform is the **MTF**. From
//!   the same profile come the two numbers a satellite-imagery specification is usually written in:
//!   **RER**, the relative edge response over one pixel, and the **FWHM** of the LSF.
//! * [`snr_homogeneous_blocks`] and [`snr_homogeneous_area`] — signal-to-noise from the flattest
//!   parts of the scene, where the local standard deviation is the noise because there is nothing
//!   else for it to be. The two differ in how they find flat: a block grid, or a local-binary-pattern
//!   test that keeps the pixels whose neighbourhood is *not* an ordered ramp.
//! * [`gaussian_blur`], [`resample`], [`quantise`] and [`add_gaussian_noise`] — controlled
//!   degradations. These matter more than they look: a metric can only be verified against an image
//!   whose quality is known, and the way to know it is to take a reference and damage it by a stated
//!   amount.
//! * [`ground_sample_distance`] — the ground spacing of adjacent pixels, from the RPC camera rather
//!   than from the metadata that claims it.
//!
//! Ported from the metric core of satellogic's **iquaflow** (MIT): `metrics/sharpness_metric.py`,
//! `metrics/rer_metric.py`, `metrics/snr_metric.py` and the `datasets/modifier_*.py` degradations.
//! The framework around them — the `MLflow` experiment harness, the dataset tooling and the
//! deep-learning quality regressors — is a Python research platform and is deliberately not here.
//!
//! # What is not measured
//!
//! No window is applied to the LSF before the transform. ISO 12233 applies one because a real edge
//! profile has noisy tails, and a window suppresses them — at the cost of broadening the LSF and
//! biasing the MTF downward by an amount that depends on the window rather than on the image. The
//! caller crops the region of interest to the edge instead, which is the same tail suppression
//! without the bias.

// Short names (x, y, d, f) are the ISO 12233 notation. Every `as f64` below is applied to a pixel
// count or an index, exact for any image that fits in memory; every `as usize` is applied to a
// distance or a count that the line above has already clamped to a non-negative value inside the
// image, so neither the truncation nor the sign can bite.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::features::Image;
use crate::radiative_transfer::Rng;
use crate::rpc::{InverseError, RationalPolynomialCamera};
use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::TAU;

/// Why a patch could not be measured.
///
/// A slanted-edge measurement on an unsuitable patch does not fail loudly — it returns a plausible
/// number that is wrong, which is worse. Each variant below is one of the ways iquaflow's patch
/// rejection says no.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeError {
    /// Fewer than eight rows or columns: too few sub-pixel phases to reconstruct anything.
    TooSmall,
    /// The two sides of the edge differ by less than the configured contrast. The derivative
    /// centroid is then dominated by noise rather than by the edge.
    LowContrast,
    /// A pixel is at or above the configured saturation level. A clipped edge has no measurable
    /// slope: the MTF it reports is the sensor's clipping, not its optics.
    Saturated,
    /// No intensity gradient at all in some row, so the edge does not cross the whole patch.
    NoEdge,
    /// The edge is not straight: the residual of the line fit exceeds half a pixel.
    NotStraight,
    /// The edge angle is outside the configured band. Too close to vertical and the rows repeat the
    /// same sub-pixel phase instead of covering them; too far and the projection smears the profile.
    AngleOutOfRange,
    /// A bin of the supersampled profile received no pixel — the patch is too short for the slant.
    EmptyBin,
}

/// Patch-acceptance thresholds and the supersampling pitch.
#[derive(Debug, Clone, Copy)]
pub struct EdgeConfig {
    /// Width of an ESF bin, in pixels. ISO 12233's quarter-pixel is the default: four bins per pixel
    /// is what a slant of a few degrees over a few tens of rows can actually fill.
    pub bin_width: f64,
    /// Reject the patch if the two plateaux differ by less than this, in the image's own units.
    pub min_contrast: f64,
    /// Reject the patch if any pixel reaches this level.
    pub saturation: f64,
    /// Accepted edge angles from vertical, in degrees.
    pub angle_bounds: (f64, f64),
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            bin_width: 0.25,
            min_contrast: 0.05,
            saturation: f64::INFINITY,
            angle_bounds: (1.0, 15.0),
        }
    }
}

/// A measured edge profile: the supersampled ESF, its derivative, and where the edge sits in them.
#[derive(Debug, Clone)]
pub struct SlantedEdge {
    /// Edge angle from vertical, in degrees, signed by the direction of the slant.
    pub angle_degrees: f64,
    /// The ESF bin width, in pixels.
    pub bin_width: f64,
    /// The edge-spread function, normalised to run from 0 to 1 across the edge.
    pub esf: Vec<f64>,
    /// The line-spread function: the ESF's derivative, per pixel.
    pub lsf: Vec<f64>,
    /// The 50 % crossing of the ESF, in bins — the origin every metric below is measured from.
    pub centre: f64,
}

impl SlantedEdge {
    /// The modulation transfer function at `frequency`, in cycles per pixel.
    ///
    /// The transform is evaluated directly at the requested frequency rather than on an FFT grid:
    /// the LSF is a few hundred samples, the cost is irrelevant, and it removes the zero-padding
    /// that would otherwise decide which frequencies can be asked for. Normalised by the DC term, so
    /// `mtf(0.0)` is 1 by construction.
    ///
    /// The result is divided by the *measurement's* own transfer function. Two steps of this
    /// analysis are box filters one bin wide — averaging the pixels that fall in a bin, and taking
    /// the difference of adjacent bins for the derivative — so each multiplies the answer by
    /// `sinc(f·h)`, exactly and knowably. Leaving them in would report the camera's MTF times the
    /// analysis's: 6 % low at 0.35 cycles per pixel with quarter-pixel bins, which is the difference
    /// between passing and failing a typical sharpness specification. Above `1/h` the correction is
    /// a division by zero and the measurement means nothing; it returns 0 there.
    #[must_use]
    pub fn mtf(&self, frequency: f64) -> f64 {
        let (mut re, mut im, mut dc) = (0.0, 0.0, 0.0);
        for (i, &value) in self.lsf.iter().enumerate() {
            // Sample `i` of the difference sits halfway between ESF bins `i` and `i + 1`.
            let x = (i as f64 + 0.5 - self.centre) * self.bin_width;
            let phase = TAU * frequency * x;
            re += value * libm::cos(phase);
            im -= value * libm::sin(phase);
            dc += value;
        }
        if dc == 0.0 {
            return 0.0;
        }
        let response = sinc(frequency * self.bin_width);
        if libm::fabs(response) < 1e-6 {
            return 0.0;
        }
        libm::sqrt(re * re + im * im) / libm::fabs(dc) / (response * response)
    }

    /// The relative edge response: how much of the full edge step is crossed within one pixel,
    /// `ESF(+½) − ESF(−½)`.
    ///
    /// The number a satellite-imagery specification is usually written in, because it is the one an
    /// interpreter's eye responds to: 1 would be a perfect edge inside a single pixel, and a typical
    /// well-focused sensor lands near 0.5.
    #[must_use]
    pub fn rer(&self) -> f64 {
        let half = 0.5 / self.bin_width;
        self.esf_at(self.centre + half) - self.esf_at(self.centre - half)
    }

    /// The full width at half maximum of the LSF, in pixels.
    ///
    /// Returns `None` if the LSF never falls to half its peak on one side — a profile that wide is
    /// not measured by the patch, it is cut off by it.
    #[must_use]
    pub fn fwhm(&self) -> Option<f64> {
        let (peak_index, peak) =
            self.lsf
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                });
        if peak <= 0.0 {
            return None;
        }
        let half = peak * 0.5;
        let left = crossing(&self.lsf[..=peak_index], half, true)?;
        let right = crossing(&self.lsf[peak_index..], half, false)? + peak_index as f64;
        Some((right - left) * self.bin_width)
    }

    /// The ESF at a fractional bin index, by linear interpolation, clamped at the ends.
    fn esf_at(&self, index: f64) -> f64 {
        let last = self.esf.len() - 1;
        if index <= 0.0 {
            return self.esf[0];
        }
        if index >= last as f64 {
            return self.esf[last];
        }
        let lower = index as usize;
        let t = index - lower as f64;
        self.esf[lower] * (1.0 - t) + self.esf[lower + 1] * t
    }
}

/// The normalised sinc `sin(πx)/(πx)`, with its removable singularity filled in.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let argument = core::f64::consts::PI * x;
    libm::sin(argument) / argument
}

/// The fractional index where `values` crosses `level`, searching from the end that faces the peak.
fn crossing(values: &[f64], level: f64, from_start: bool) -> Option<f64> {
    if from_start {
        // Walking back from the peak (the last element) to the first sample below `level`.
        for i in (1..values.len()).rev() {
            if values[i - 1] <= level && values[i] > level {
                let t = (level - values[i - 1]) / (values[i] - values[i - 1]);
                return Some(i as f64 - 1.0 + t);
            }
        }
    } else {
        for i in 1..values.len() {
            if values[i - 1] > level && values[i] <= level {
                let t = (values[i - 1] - level) / (values[i - 1] - values[i]);
                return Some(i as f64 - 1.0 + t);
            }
        }
    }
    None
}

/// The line the edge runs along, `x = intercept + slope·y`, with the angle it makes in degrees —
/// and every reason to refuse the patch, checked before anything is measured from it.
///
/// The per-row location is the centroid of that row's own gradient magnitude, which is where the
/// edge is to sub-pixel precision for any symmetric point-spread function, whatever its width.
fn fit_edge_line(image: &Image, config: &EdgeConfig) -> Result<(f64, f64, f64), EdgeError> {
    let (h, w) = (image.h, image.w);
    if h < 8 || w < 8 {
        return Err(EdgeError::TooSmall);
    }
    let (low, high) = image
        .data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    if high >= config.saturation {
        return Err(EdgeError::Saturated);
    }
    if high - low < config.min_contrast {
        return Err(EdgeError::LowContrast);
    }

    let mut located = vec![0.0; h];
    for (y, slot) in located.iter_mut().enumerate() {
        let row = &image.data[y * w..(y + 1) * w];
        let (mut moment, mut mass) = (0.0, 0.0);
        for x in 1..w - 1 {
            let gradient = libm::fabs((row[x + 1] - row[x - 1]) * 0.5);
            moment += gradient * x as f64;
            mass += gradient;
        }
        if mass <= 0.0 {
            return Err(EdgeError::NoEdge);
        }
        *slot = moment / mass;
    }

    let n = h as f64;
    let mean_y = (n - 1.0) * 0.5;
    let mean_x = located.iter().sum::<f64>() / n;
    let (mut sxy, mut syy) = (0.0, 0.0);
    for (y, &x) in located.iter().enumerate() {
        let dy = y as f64 - mean_y;
        sxy += dy * (x - mean_x);
        syy += dy * dy;
    }
    let slope = sxy / syy;
    let intercept = mean_x - slope * mean_y;

    let residual = libm::sqrt(
        located
            .iter()
            .enumerate()
            .map(|(y, &x)| {
                let e = x - (intercept + slope * y as f64);
                e * e
            })
            .sum::<f64>()
            / n,
    );
    if residual > 0.5 {
        return Err(EdgeError::NotStraight);
    }
    let angle_degrees = libm::atan(slope) * 180.0 / core::f64::consts::PI;
    let (min_angle, max_angle) = config.angle_bounds;
    if !(min_angle..=max_angle).contains(&libm::fabs(angle_degrees)) {
        return Err(EdgeError::AngleOutOfRange);
    }
    Ok((intercept, slope, angle_degrees))
}

/// Measure a near-vertical slanted edge.
///
/// The patch must contain one straight edge crossing every row, slanted by a few degrees from
/// vertical. For a near-horizontal edge, transpose the patch and read the result the same way.
///
/// # Errors
///
/// Any of [`EdgeError`] — the patch-rejection tests are run before the measurement, because each one
/// of them is a way to get a confident wrong answer.
pub fn analyse_slanted_edge(image: &Image, config: &EdgeConfig) -> Result<SlantedEdge, EdgeError> {
    let (h, w) = (image.h, image.w);
    let (intercept, slope, angle_degrees) = fit_edge_line(image, config)?;
    let angle = libm::atan(slope);
    let n = h as f64;

    // Project every pixel onto the edge normal and bin it. The reach is limited by the row whose
    // edge sits closest to a border, so that every bin is fed from every row.
    let cos_angle = libm::cos(angle);
    let reach = {
        let leftmost = intercept.min(intercept + slope * (n - 1.0));
        let rightmost = intercept.max(intercept + slope * (n - 1.0));
        (leftmost - 1.0).min(w as f64 - 2.0 - rightmost) * cos_angle
    };
    if reach < 2.0 * config.bin_width {
        return Err(EdgeError::TooSmall);
    }
    let bins = (2.0 * reach / config.bin_width) as usize;
    let mut sums = vec![0.0; bins];
    let mut counts = vec![0u32; bins];
    for y in 0..h {
        let edge = intercept + slope * y as f64;
        for x in 0..w {
            let distance = (x as f64 - edge) * cos_angle;
            let index = (distance + reach) / config.bin_width;
            if index < 0.0 {
                continue;
            }
            let index = index as usize;
            if index < bins {
                sums[index] += image.data[y * w + x];
                counts[index] += 1;
            }
        }
    }
    if counts.contains(&0) {
        return Err(EdgeError::EmptyBin);
    }
    let mut esf: Vec<f64> = sums
        .iter()
        .zip(&counts)
        .map(|(&s, &c)| s / f64::from(c))
        .collect();

    // Normalise to a 0 → 1 rise, from the plateaux rather than from the extremes: an overshoot at a
    // sharp edge is part of the response, not part of the step it is measured against.
    let plateau = (bins / 10).max(1);
    let start = esf[..plateau].iter().sum::<f64>() / plateau as f64;
    let end = esf[bins - plateau..].iter().sum::<f64>() / plateau as f64;
    let span = end - start;
    if libm::fabs(span) < config.min_contrast {
        return Err(EdgeError::LowContrast);
    }
    for value in &mut esf {
        *value = (*value - start) / span;
    }

    // The difference of *adjacent* bins, not of the two either side: the second is a box filter two
    // bins wide, and it broadens the line-spread function by four times as much variance as it needs
    // to. Sample `i` of the result belongs at bin `i + ½`, which is where `mtf` reads it from.
    let lsf: Vec<f64> = esf
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) / config.bin_width)
        .collect();

    let centre = crossing_up(&esf, 0.5).ok_or(EdgeError::NoEdge)?;
    Ok(SlantedEdge {
        angle_degrees,
        bin_width: config.bin_width,
        esf,
        lsf,
        centre,
    })
}

/// The fractional index where a rising profile first crosses `level`.
fn crossing_up(values: &[f64], level: f64) -> Option<f64> {
    for i in 1..values.len() {
        if values[i - 1] <= level && values[i] > level {
            let denominator = values[i] - values[i - 1];
            let t = if denominator == 0.0 {
                0.0
            } else {
                (level - values[i - 1]) / denominator
            };
            return Some(i as f64 - 1.0 + t);
        }
    }
    None
}

/// A signal-to-noise estimate and the sample it was made from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SnrEstimate {
    /// Mean over noise standard deviation, on the parts of the scene judged homogeneous.
    pub snr: f64,
    /// The mean signal there.
    pub mean: f64,
    /// The standard deviation there — the noise, if the judgement was right.
    pub sigma: f64,
    /// How many blocks or windows the estimate was made from.
    pub samples: usize,
}

/// Signal-to-noise from the flattest blocks of a regular grid.
///
/// The image is tiled into `block`×`block` squares, and the `keep_fraction` of them with the lowest
/// standard deviation are taken to be scene-free. Their median mean over their median σ is the
/// estimate — medians rather than means because one textured block that survives the ordering should
/// not move the answer.
///
/// Returns `None` if the grid holds fewer than four blocks.
#[must_use]
pub fn snr_homogeneous_blocks(
    image: &Image,
    block: usize,
    keep_fraction: f64,
) -> Option<SnrEstimate> {
    if block < 2 {
        return None;
    }
    let (rows, columns) = (image.h / block, image.w / block);
    if rows * columns < 4 {
        return None;
    }
    let mut statistics = Vec::with_capacity(rows * columns);
    for by in 0..rows {
        for bx in 0..columns {
            let mut values = Vec::with_capacity(block * block);
            for y in by * block..(by + 1) * block {
                values.extend_from_slice(&image.data[y * image.w + bx * block..][..block]);
            }
            statistics.push(moments(&values));
        }
    }
    statistics.sort_by(|a, b| a.1.total_cmp(&b.1));
    let keep = ((statistics.len() as f64 * keep_fraction) as usize).clamp(1, statistics.len());
    let kept = &statistics[..keep];
    let mean = median(&mut kept.iter().map(|s| s.0).collect::<Vec<_>>());
    let sigma = median(&mut kept.iter().map(|s| s.1).collect::<Vec<_>>());
    Some(SnrEstimate {
        snr: if sigma > 0.0 {
            mean / sigma
        } else {
            f64::INFINITY
        },
        mean,
        sigma,
        samples: keep,
    })
}

/// Signal-to-noise from the pixels whose 3×3 neighbourhood carries no ordered structure.
///
/// The homogeneity test is the local binary pattern of Ojala, Pietikäinen & Mäenpää (2002), used the
/// way iquaflow uses it and for the reason that makes it work: an edge or a ramp produces a
/// *uniform* pattern — one contiguous run of neighbours above the centre and one below, at most two
/// transitions around the ring. Noise on a flat surface produces the pattern that is not uniform.
/// So the pixels this keeps are the ones whose neighbourhood is nothing but noise, which is exactly
/// where σ is the noise.
///
/// Returns `None` if the image is smaller than 3×3 or if no window passes.
#[must_use]
pub fn snr_homogeneous_area(image: &Image) -> Option<SnrEstimate> {
    if image.h < 3 || image.w < 3 {
        return None;
    }
    let mut means = Vec::new();
    let mut sigmas = Vec::new();
    for y in 1..image.h - 1 {
        for x in 1..image.w - 1 {
            let centre = image.data[y * image.w + x];
            let ring = [
                image.data[(y - 1) * image.w + x - 1],
                image.data[(y - 1) * image.w + x],
                image.data[(y - 1) * image.w + x + 1],
                image.data[y * image.w + x + 1],
                image.data[(y + 1) * image.w + x + 1],
                image.data[(y + 1) * image.w + x],
                image.data[(y + 1) * image.w + x - 1],
                image.data[y * image.w + x - 1],
            ];
            let mut transitions = 0;
            for i in 0..8 {
                let a = ring[i] >= centre;
                let b = ring[(i + 1) % 8] >= centre;
                if a != b {
                    transitions += 1;
                }
            }
            if transitions <= 2 {
                continue;
            }
            let mut window = ring.to_vec();
            window.push(centre);
            let (mean, sigma) = moments(&window);
            means.push(mean);
            sigmas.push(sigma);
        }
    }
    if means.is_empty() {
        return None;
    }
    let samples = means.len();
    let mean = median(&mut means);
    // The 3×3 sample standard deviation of white noise is biased low; `sqrt(9/8)` is the Bessel
    // correction the nine-pixel window still owes after dividing by `n`.
    let sigma = median(&mut sigmas) * libm::sqrt(9.0 / 8.0);
    Some(SnrEstimate {
        snr: if sigma > 0.0 {
            mean / sigma
        } else {
            f64::INFINITY
        },
        mean,
        sigma,
        samples,
    })
}

/// Mean and population standard deviation.
fn moments(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n;
    (mean, libm::sqrt(variance))
}

/// The median, by sorting in place.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    }
}

/// Blur by a Gaussian of standard deviation `sigma` pixels, separably, with a replicated border.
///
/// The kernel is truncated at four sigma and renormalised, so it sums to one exactly whatever the
/// truncation cost — a kernel that does not is a brightness change wearing a blur's name.
#[must_use]
pub fn gaussian_blur(image: &Image, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return image.data.to_vec();
    }
    let radius = libm::ceil(4.0 * sigma) as usize;
    let mut kernel: Vec<f64> = (0..=2 * radius)
        .map(|i| {
            let d = i as f64 - radius as f64;
            libm::exp(-0.5 * d * d / (sigma * sigma))
        })
        .collect();
    let total: f64 = kernel.iter().sum();
    for value in &mut kernel {
        *value /= total;
    }

    let (h, w) = (image.h, image.w);
    // `saturating_sub` is the replicated border: a tap that would fall off the low side reads the
    // first row or column, and `min` does the same at the high side.
    let clamp = |position: usize, tap: usize, limit: usize| {
        (position + tap).saturating_sub(radius).min(limit - 1)
    };
    let mut horizontal = vec![0.0; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                acc += weight * image.data[y * w + clamp(x, k, w)];
            }
            horizontal[y * w + x] = acc;
        }
    }
    let mut out = vec![0.0; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &weight) in kernel.iter().enumerate() {
                acc += weight * horizontal[clamp(y, k, h) * w + x];
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Resample to `factor` times the pixel count per axis: the ground sample distance changes by
/// `1/factor`.
///
/// Down (`factor < 1`) is an area average over the source footprint, which is what a coarser
/// detector does; up is bilinear, which is what a resampler does and adds no information. Returns
/// the pixels with their new height and width.
#[must_use]
pub fn resample(image: &Image, factor: f64) -> (Vec<f64>, usize, usize) {
    let h = (libm::round(image.h as f64 * factor) as usize).max(1);
    let w = (libm::round(image.w as f64 * factor) as usize).max(1);
    let mut out = vec![0.0; h * w];
    let (scale_y, scale_x) = (image.h as f64 / h as f64, image.w as f64 / w as f64);
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = if factor < 1.0 {
                let y0 = (y as f64 * scale_y) as usize;
                let y1 = (libm::ceil((y + 1) as f64 * scale_y) as usize).min(image.h);
                let x0 = (x as f64 * scale_x) as usize;
                let x1 = (libm::ceil((x + 1) as f64 * scale_x) as usize).min(image.w);
                let mut sum = 0.0;
                for sy in y0..y1 {
                    for sx in x0..x1 {
                        sum += image.data[sy * image.w + sx];
                    }
                }
                sum / ((y1 - y0) * (x1 - x0)) as f64
            } else {
                let sy = ((y as f64 + 0.5) * scale_y - 0.5).clamp(0.0, image.h as f64 - 1.0);
                let sx = ((x as f64 + 0.5) * scale_x - 0.5).clamp(0.0, image.w as f64 - 1.0);
                let (y0, x0) = (sy as usize, sx as usize);
                let (y1, x1) = ((y0 + 1).min(image.h - 1), (x0 + 1).min(image.w - 1));
                let (ty, tx) = (sy - y0 as f64, sx - x0 as f64);
                let top =
                    image.data[y0 * image.w + x0] * (1.0 - tx) + image.data[y0 * image.w + x1] * tx;
                let bottom =
                    image.data[y1 * image.w + x0] * (1.0 - tx) + image.data[y1 * image.w + x1] * tx;
                top * (1.0 - ty) + bottom * ty
            };
        }
    }
    (out, h, w)
}

/// Quantise to `levels` uniform steps over `[0, full_scale]`, the way a shallower analogue-to-digital
/// converter would.
///
/// Values outside the range are clamped, because that is what the converter does too.
#[must_use]
pub fn quantise(image: &Image, levels: u32, full_scale: f64) -> Vec<f64> {
    let steps = f64::from(levels.max(2) - 1);
    image
        .data
        .iter()
        .map(|&v| {
            let t = (v / full_scale).clamp(0.0, 1.0);
            libm::round(t * steps) / steps * full_scale
        })
        .collect()
}

/// Add white Gaussian noise of standard deviation `sigma`, from a seeded stream.
#[must_use]
pub fn add_gaussian_noise(image: &Image, sigma: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(image.data.len());
    let mut spare = None;
    for &value in image.data {
        let deviate = spare.take().unwrap_or_else(|| {
            let radius = libm::sqrt(-2.0 * libm::log(rng.open_unit()));
            let angle = rng.azimuth();
            spare = Some(radius * libm::sin(angle));
            radius * libm::cos(angle)
        });
        out.push(value + sigma * deviate);
    }
    out
}

/// The ground spacing of adjacent pixels at `(row, column)`, in metres, as `(across, along)`.
///
/// Measured through the camera rather than read from the product metadata: the two disagree whenever
/// the look is off-nadir or the terrain height differs from the one the metadata assumed, and the
/// disagreement is the whole reason to compute it.
///
/// # Errors
///
/// [`InverseError`] if any of the three inverse projections fails to converge.
pub fn ground_sample_distance(
    camera: &RationalPolynomialCamera,
    row: f64,
    column: f64,
    height: f64,
) -> Result<(f64, f64), InverseError> {
    let here = camera.inverse(row, column, height)?;
    let across = camera.inverse(row, column + 1.0, height)?;
    let along = camera.inverse(row + 1.0, column, height)?;
    let (metres_per_degree_lat, metres_per_degree_lon) = local_scale(here.latitude);
    let distance = |a: f64, b: f64, c: f64, d: f64| {
        let dy = (a - c) * metres_per_degree_lat;
        let dx = (b - d) * metres_per_degree_lon;
        libm::sqrt(dx * dx + dy * dy)
    };
    Ok((
        distance(
            here.latitude,
            here.longitude,
            across.latitude,
            across.longitude,
        ),
        distance(
            here.latitude,
            here.longitude,
            along.latitude,
            along.longitude,
        ),
    ))
}

/// Metres per degree of latitude and of longitude on the WGS-84 ellipsoid, at `latitude` degrees.
///
/// The meridian and prime-vertical radii of curvature, not a mean sphere: they differ by 0.7 % from
/// each other at the equator, which is a whole pixel in twenty on a metre-class product.
fn local_scale(latitude: f64) -> (f64, f64) {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    let e2 = F * (2.0 - F);
    let sin_lat = libm::sin(latitude * core::f64::consts::PI / 180.0);
    let w = 1.0 - e2 * sin_lat * sin_lat;
    let meridian = A * (1.0 - e2) / (w * libm::sqrt(w));
    let prime_vertical = A / libm::sqrt(w);
    let radians_per_degree = core::f64::consts::PI / 180.0;
    (
        meridian * radians_per_degree,
        prime_vertical * libm::cos(latitude * radians_per_degree) * radians_per_degree,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    const SIGMA: f64 = 0.8;
    const SLANT: f64 = 0.1234; // ≈ 7.0° from vertical.

    /// A synthetic patch whose edge is the ideal step convolved with a Gaussian of `sigma` pixels:
    /// the ESF is the normal CDF, so every metric has a closed form.
    fn gaussian_edge(h: usize, w: usize, sigma: f64) -> Vec<f64> {
        let mut data = vec![0.0; h * w];
        for y in 0..h {
            let edge = w as f64 * 0.5 + SLANT * (y as f64 - h as f64 * 0.5);
            for x in 0..w {
                let d = x as f64 - edge;
                // Φ(d/σ) = ½(1 + erf(d/(σ√2))).
                data[y * w + x] = 0.5 * (1.0 + libm::erf(d / (sigma * core::f64::consts::SQRT_2)));
            }
        }
        data
    }

    /// A Gaussian edge placed at `centre` at mid-patch and leaning by `slant` pixels per row.
    ///
    /// [`gaussian_edge`] is the centred, right-leaning case; the reach is measured from whichever
    /// border the edge leans towards, so telling the two sides apart needs the other lean.
    fn slanted_edge_at(h: usize, w: usize, sigma: f64, centre: f64, slant: f64) -> Vec<f64> {
        let mut data = vec![0.0; h * w];
        for y in 0..h {
            let edge = centre + slant * (y as f64 - h as f64 * 0.5);
            for x in 0..w {
                let d = x as f64 - edge;
                data[y * w + x] = 0.5 * (1.0 + libm::erf(d / (sigma * core::f64::consts::SQRT_2)));
            }
        }
        data
    }

    /// An edge with no point-spread function at all: every pixel is exactly `base` or `base + span`.
    ///
    /// The plateau means are then exactly those two values, so the measured contrast is exactly
    /// `span` — the only way to sit a patch *on* a contrast floor rather than near it.
    fn hard_step_edge(h: usize, w: usize, base: f64, span: f64) -> Vec<f64> {
        let mut data = vec![0.0; h * w];
        for y in 0..h {
            let edge = w as f64 * 0.5 + SLANT * (y as f64 - h as f64 * 0.5);
            for x in 0..w {
                data[y * w + x] = if (x as f64) < edge { base } else { base + span };
            }
        }
        data
    }

    /// A Gaussian edge whose plateaux sit at `base` and `base + span` rather than at 0 and 1.
    ///
    /// The metrics are all differences or ratios and so are blind to the offset — which is exactly
    /// why the ESF's own normalisation needs a fixture that is not already normalised.
    fn offset_edge(h: usize, w: usize, sigma: f64, base: f64, span: f64) -> Vec<f64> {
        gaussian_edge(h, w, sigma)
            .into_iter()
            .map(|v| base + span * v)
            .collect()
    }

    /// A Gaussian edge whose line is bent by `amplitude` pixels over one sine period down the patch.
    fn bent_edge(h: usize, w: usize, sigma: f64, amplitude: f64) -> Vec<f64> {
        let mut data = vec![0.0; h * w];
        for y in 0..h {
            let t = TAU * y as f64 / h as f64;
            let edge =
                w as f64 * 0.5 + SLANT * (y as f64 - h as f64 * 0.5) + amplitude * libm::sin(t);
            for x in 0..w {
                let d = x as f64 - edge;
                data[y * w + x] = 0.5 * (1.0 + libm::erf(d / (sigma * core::f64::consts::SQRT_2)));
            }
        }
        data
    }

    /// The same edge convolved with a box of `width` pixels instead: the ESF is a straight ramp.
    fn box_edge(h: usize, w: usize, width: f64) -> Vec<f64> {
        let mut data = vec![0.0; h * w];
        for y in 0..h {
            let edge = w as f64 * 0.5 + SLANT * (y as f64 - h as f64 * 0.5);
            for x in 0..w {
                let d = x as f64 - edge;
                data[y * w + x] = (d / width + 0.5).clamp(0.0, 1.0);
            }
        }
        data
    }

    /// The measured FWHM, RER and MTF of a Gaussian edge, against their closed forms.
    ///
    /// This is the rare case where the truth is exact rather than another implementation: for a
    /// Gaussian PSF the LSF is Gaussian, so `FWHM = 2√(2 ln 2)·σ`, the MTF is `exp(−2π²σ²f²)`, and
    /// the relative edge response is `erf(½/(σ√2))`.
    ///
    /// The MTF is corrected for the analysis's own two box filters and lands within 1 % of the
    /// closed form. The FWHM and the RER are spatial-domain numbers and are not corrected, so they
    /// keep the discretisation — binning and differencing each add `h²/12` to the variance of the
    /// measured profile, 0.8 % of the width at `σ = 0.8` and `h = ¼`. What is measured is −0.4 %,
    /// and `a_repeating_slant_measures_worse_than_an_irrational_one` is what says the residue is the
    /// sampling rather than an error: the same code on a slant that visits only ten sub-pixel phases
    /// comes back four times worse.
    #[test]
    fn a_gaussian_edge_returns_the_sigma_it_was_built_with() {
        let data = gaussian_edge(64, 48, SIGMA);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");

        let want_angle = libm::atan(SLANT) * 180.0 / core::f64::consts::PI;
        assert!(
            (edge.angle_degrees - want_angle).abs() < 0.05,
            "angle {} vs {want_angle}",
            edge.angle_degrees
        );

        let want_fwhm = 2.0 * libm::sqrt(2.0 * libm::log(2.0)) * SIGMA;
        let got_fwhm = edge.fwhm().expect("the LSF falls off inside the patch");
        assert!(
            (got_fwhm - want_fwhm).abs() < 0.02,
            "FWHM {got_fwhm}, want {want_fwhm}"
        );

        let want_rer = libm::erf(0.5 / (SIGMA * core::f64::consts::SQRT_2));
        assert!(
            (edge.rer() - want_rer).abs() < 0.01,
            "RER {}, want {want_rer}",
            edge.rer()
        );

        for frequency in [0.1, 0.25, 0.4, 0.5] {
            let want = libm::exp(
                -2.0 * core::f64::consts::PI
                    * core::f64::consts::PI
                    * SIGMA
                    * SIGMA
                    * frequency
                    * frequency,
            );
            let got = edge.mtf(frequency);
            assert!(
                (got - want).abs() < 0.01,
                "MTF({frequency}) = {got}, want {want}"
            );
        }
        assert!((edge.mtf(0.0) - 1.0).abs() < 1e-12, "MTF(0) must be 1");
    }

    /// The measurement needs the edge's sub-pixel phases to *cover* the bin, and a slant that
    /// repeats does not provide that.
    ///
    /// A slant of exactly 1/10 of a pixel per row visits ten sub-pixel phases and no others, however
    /// many rows the patch has: each quarter-pixel bin then averages two or three fixed offsets
    /// instead of a spread, and the second moment of that little set alternates from bin to bin. The
    /// ESF picks up a ripple at the bin frequency, and differentiating amplifies it — measured here
    /// as a 1.6 % FWHM error against 0.4 % for a slant with no short period. This is why ISO 12233
    /// specifies a slant that is not a simple fraction, and it is a property of the method rather
    /// than of this implementation.
    #[test]
    fn a_repeating_slant_measures_worse_than_an_irrational_one() {
        let want = 2.0 * libm::sqrt(2.0 * libm::log(2.0)) * SIGMA;
        let error = |slant: f64| {
            let mut data = vec![0.0; 64 * 48];
            for y in 0..64 {
                let edge = 24.0 + slant * (y as f64 - 32.0);
                for x in 0..48 {
                    let d = x as f64 - edge;
                    data[y * 48 + x] =
                        0.5 * (1.0 + libm::erf(d / (SIGMA * core::f64::consts::SQRT_2)));
                }
            }
            let image = Image {
                data: &data,
                h: 64,
                w: 48,
            };
            let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");
            libm::fabs(edge.fwhm().expect("the LSF falls off inside the patch") - want) / want
        };
        let repeating = error(0.1);
        let covering = error(SLANT);
        assert!(
            covering < repeating * 0.5,
            "phase coverage should more than halve the error: {repeating} → {covering}"
        );
        assert!(
            repeating < 0.05,
            "even the bad case stays inside 5 %: {repeating}"
        );
    }

    /// A box PSF's MTF is `|sinc(w f)|`, including the zero — the case an exponential cannot fake.
    #[test]
    fn a_box_psf_transforms_to_a_sinc() {
        let data = box_edge(80, 60, 2.0);
        let image = Image {
            data: &data,
            h: 80,
            w: 60,
        };
        let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");
        for frequency in [0.1, 0.2, 0.35] {
            let argument = core::f64::consts::PI * 2.0 * frequency;
            let want = libm::fabs(libm::sin(argument) / argument);
            let got = edge.mtf(frequency);
            assert!(
                (got - want).abs() < 0.02,
                "MTF({frequency}) = {got}, want {want}"
            );
        }
        // The first zero of a two-pixel box is at 0.5 cycles/pixel.
        assert!(edge.mtf(0.5) < 0.05, "the sinc zero is {}", edge.mtf(0.5));
    }

    /// Blurring a measured edge moves the metrics the way blur does, by the amount blur does.
    ///
    /// Two Gaussians in series are a Gaussian of the quadrature sum, so the degraded edge has a
    /// *predicted* FWHM rather than merely a larger one — which is what makes this a check on the
    /// metric and not just on the sign of a difference.
    ///
    /// The blur is applied to a larger patch and the interior is cropped out afterwards. Within four
    /// sigma of the border the separable blur is reading replicated rows, whose edge sits at the
    /// wrong place for their position; those rows come back sharper than the rest, and mixed into
    /// the same profile they pull the measured width down by 3 %. That is the fixture's border, not
    /// the metric's, and cropping is how the two are told apart.
    #[test]
    fn a_known_blur_moves_the_metrics_by_the_predicted_amount() {
        let data = gaussian_edge(64, 48, SIGMA);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        let sharp = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");

        let added = 1.2;
        let wide = gaussian_edge(96, 72, SIGMA);
        let wide_image = Image {
            data: &wide,
            h: 96,
            w: 72,
        };
        let blurred_wide = gaussian_blur(&wide_image, added);
        let mut cropped = vec![0.0; 64 * 48];
        for y in 0..64 {
            cropped[y * 48..(y + 1) * 48]
                .copy_from_slice(&blurred_wide[(y + 16) * 72 + 12..(y + 16) * 72 + 60]);
        }
        let blurred_image = Image {
            data: &cropped,
            h: 64,
            w: 48,
        };
        let blurred =
            analyse_slanted_edge(&blurred_image, &EdgeConfig::default()).expect("still an edge");

        let combined = libm::sqrt(SIGMA * SIGMA + added * added);
        let want = 2.0 * libm::sqrt(2.0 * libm::log(2.0)) * combined;
        let got = blurred.fwhm().expect("the LSF falls off inside the patch");
        assert!(
            (got - want).abs() < 0.05,
            "blurred FWHM {got}, want {want} from σ = {combined}"
        );
        assert!(
            blurred.rer() < sharp.rer(),
            "blur must lower the edge response: {} → {}",
            sharp.rer(),
            blurred.rer()
        );
        assert!(
            blurred.mtf(0.25) < sharp.mtf(0.25),
            "blur must lower the MTF: {} → {}",
            sharp.mtf(0.25),
            blurred.mtf(0.25)
        );
    }

    /// Each patch-rejection test refuses the patch it exists for.
    #[test]
    fn the_patch_rejection_tests_refuse_what_they_are_for() {
        let flat = vec![0.5; 64 * 48];
        let image = Image {
            data: &flat,
            h: 64,
            w: 48,
        };
        assert_eq!(
            analyse_slanted_edge(&image, &EdgeConfig::default()).unwrap_err(),
            EdgeError::LowContrast
        );

        let small = gaussian_edge(6, 6, SIGMA);
        let image = Image {
            data: &small,
            h: 6,
            w: 6,
        };
        assert_eq!(
            analyse_slanted_edge(&image, &EdgeConfig::default()).unwrap_err(),
            EdgeError::TooSmall
        );

        let data = gaussian_edge(64, 48, SIGMA);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        assert_eq!(
            analyse_slanted_edge(
                &image,
                &EdgeConfig {
                    saturation: 0.9,
                    ..EdgeConfig::default()
                }
            )
            .unwrap_err(),
            EdgeError::Saturated
        );
        assert_eq!(
            analyse_slanted_edge(
                &image,
                &EdgeConfig {
                    angle_bounds: (20.0, 40.0),
                    ..EdgeConfig::default()
                }
            )
            .unwrap_err(),
            EdgeError::AngleOutOfRange
        );
    }

    /// Both SNR estimators return the noise that was added, on an image where nothing else is there.
    #[test]
    fn the_snr_estimators_return_the_noise_that_was_added() {
        const LEVEL: f64 = 0.6;
        const NOISE: f64 = 0.02;
        let flat = vec![LEVEL; 128 * 128];
        let image = Image {
            data: &flat,
            h: 128,
            w: 128,
        };
        let noisy_data = add_gaussian_noise(&image, NOISE, 7);
        let noisy = Image {
            data: &noisy_data,
            h: 128,
            w: 128,
        };

        let want = LEVEL / NOISE;
        let blocks = snr_homogeneous_blocks(&noisy, 16, 0.5).expect("64 blocks");
        assert!(
            (blocks.snr - want).abs() / want < 0.20,
            "block SNR {}, want {want}",
            blocks.snr
        );
        let area = snr_homogeneous_area(&noisy).expect("every window is noise");
        assert!(
            (area.snr - want).abs() / want < 0.20,
            "area SNR {}, want {want}",
            area.snr
        );
    }

    /// The homogeneity test keeps noise and rejects structure — which is the only reason to run it.
    #[test]
    fn the_homogeneity_test_rejects_a_ramp() {
        let mut ramp = vec![0.0; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                ramp[y * 64 + x] = x as f64 / 64.0;
            }
        }
        let image = Image {
            data: &ramp,
            h: 64,
            w: 64,
        };
        assert_eq!(
            snr_homogeneous_area(&image),
            None,
            "a monotone ramp is a uniform pattern everywhere"
        );
    }

    /// Halving the sampling halves the pixel count per axis and averages what it drops.
    #[test]
    fn resampling_down_averages_and_resampling_up_interpolates() {
        let data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let image = Image {
            data: &data,
            h: 2,
            w: 4,
        };
        let (half, h, w) = resample(&image, 0.5);
        assert_eq!((h, w), (1, 2));
        // The four corners of each 2×2 source block, averaged.
        assert_eq!(
            half,
            [
                (1.0 + 3.0 + 9.0 + 11.0) / 4.0,
                (5.0 + 7.0 + 13.0 + 15.0) / 4.0
            ]
        );

        let (doubled, h, w) = resample(&image, 2.0);
        assert_eq!((h, w), (4, 8));
        // The corner output pixel falls outside every source centre and is clamped to the corner
        // source value; interpolation adds nothing anywhere, so the mean is unchanged.
        assert_eq!(doubled[0], 1.0);
        let source_mean = data.iter().sum::<f64>() / data.len() as f64;
        let doubled_mean = doubled.iter().sum::<f64>() / doubled.len() as f64;
        assert!(
            (doubled_mean - source_mean).abs() < 1e-12,
            "upsampling moved the mean: {source_mean} → {doubled_mean}"
        );
    }

    /// Quantisation lands on the level grid and never leaves the range.
    #[test]
    fn quantisation_lands_on_the_grid() {
        let data = [0.0, 0.1, 0.24, 0.26, 0.5, 0.99, 1.4, -0.3];
        let image = Image {
            data: &data,
            h: 1,
            w: 8,
        };
        // Five levels over [0, 1]: the grid is 0, ¼, ½, ¾, 1.
        assert_eq!(
            quantise(&image, 5, 1.0),
            [0.0, 0.0, 0.25, 0.25, 0.5, 1.0, 1.0, 0.0]
        );
    }

    /// The metres-per-degree scale against the published WGS-84 values.
    ///
    /// One degree of latitude is 110 574 m at the equator and 111 694 m at the pole; one degree of
    /// longitude is 111 320 m at the equator and 0 at the pole. Those are the numbers every geodesy
    /// text tabulates, and they are what a ground sample distance is denominated in.
    #[test]
    fn the_local_scale_matches_the_published_wgs84_values() {
        let (lat, lon) = local_scale(0.0);
        assert!((lat - 110_574.0).abs() < 1.0, "equatorial latitude {lat}");
        assert!((lon - 111_320.0).abs() < 1.0, "equatorial longitude {lon}");
        let (lat, lon) = local_scale(90.0);
        assert!((lat - 111_694.0).abs() < 1.0, "polar latitude {lat}");
        assert!(lon.abs() < 1e-6, "polar longitude {lon}");
        let (_, lon) = local_scale(60.0);
        assert!((lon - 55_800.0).abs() < 50.0, "longitude at 60° {lon}");
    }

    /// The ground sample distance a real Pléiades RPC implies, against the published product.
    ///
    /// The fixture is the Réunion scene's `.RPB`, and the product it describes is the 0.5 m
    /// resampled panchromatic — Pléiades' 0.7 m native sampling delivered on a half-metre grid. So
    /// the number this must return is the *product* spacing, and both axes must agree with it,
    /// because a square pixel is what the resampling produces.
    ///
    /// This is the metric's only external anchor: the closed forms behind FWHM, RER and MTF are
    /// synthetic by construction, and `local_scale` alone would still pass with the camera read
    /// backwards. Measured: 0.5078 m across-track and 0.5052 m along-track, 1.6 % and 1.0 % above
    /// the nominal grid — the rational polynomial is a fit to the physical sensor, so it carries the
    /// scene's own look angle rather than the nominal one.
    #[test]
    fn the_ground_sample_distance_matches_the_pleiades_product() {
        let camera = RationalPolynomialCamera::parse_rpb(include_str!(
            "../tests/fixtures/pleiades_reunion.rpb"
        ))
        .expect("the fixture parses");
        let (across, along) =
            ground_sample_distance(&camera, 0.0, 0.0, 0.0).expect("the inverse converges");
        for (name, gsd) in [("across", across), ("along", along)] {
            assert!(
                (gsd - 0.5).abs() < 0.02,
                "{name}-track GSD {gsd} m, want the product's 0.5 m"
            );
        }
    }

    /// The mean and the *population* deviation — the `n` divisor, not `n − 1`.
    #[test]
    fn the_moments_are_the_mean_and_the_population_deviation() {
        let (mean, sigma) = moments(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mean, 2.5);
        assert_eq!(sigma, libm::sqrt(1.25));
    }

    /// The median averages the two middle samples on an even count and takes the middle on an odd.
    #[test]
    fn the_median_is_the_middle_of_the_sorted_sample() {
        assert_eq!(median(&mut [3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&mut [4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    /// The transform is the Fourier sum of the line-spread function, on numbers small enough to do by
    /// hand.
    ///
    /// Three samples at `x = ¼, ¾, 1¼` pixels weighted `1, 2, 1`. At half a cycle per pixel the phases
    /// are `¼π`, `¾π` and `1¼π`, where both the cosine and the sine come to `±√2/2`, so the real and
    /// imaginary sums are each exactly `−√2` and the magnitude is 2 against a DC of 4. What is left is
    /// the division by the analysis's own two box filters.
    #[test]
    fn the_transform_is_the_fourier_sum_of_the_line_spread_function() {
        let edge = SlantedEdge {
            angle_degrees: 5.0,
            bin_width: 0.5,
            esf: vec![0.0, 0.25, 0.75, 1.0],
            lsf: vec![1.0, 2.0, 1.0],
            centre: 0.0,
        };
        let response = sinc(0.25);
        let want = 0.5 / (response * response);
        let got = edge.mtf(0.5);
        assert!((got - want).abs() < 1e-12, "MTF {got}, want {want}");
        // At two cycles per pixel the half-pixel bin is at its own first zero, and there is nothing
        // left to divide by: the measurement means nothing there and must say so.
        assert_eq!(edge.mtf(2.0), 0.0);
    }

    /// The edge response interpolates the ESF between bins and clamps at both ends.
    #[test]
    fn the_edge_response_interpolates_and_clamps() {
        let edge = SlantedEdge {
            angle_degrees: 5.0,
            bin_width: 0.25,
            esf: vec![0.0, 0.0, 1.0, 3.0],
            lsf: vec![0.0, 4.0, 8.0],
            centre: 1.7,
        };
        // Half a pixel is two bins, so both reads fall outside the table and clamp to its ends.
        assert_eq!(edge.rer(), 3.0);
        let wide = SlantedEdge {
            bin_width: 1.0,
            ..edge
        };
        // Half a pixel is half a bin: ESF(2.2) = 1 + ⅕(3 − 1) = 1.4, ESF(1.2) = 0 + ⅕(1 − 0) = 0.2.
        let got = wide.rer();
        assert!((got - 1.2).abs() < 1e-15, "RER {got}");
    }

    /// Two equal maxima are not one wide one — the width is measured from the first.
    ///
    /// The profile rises through half maximum at ½ and falls back through it two thirds of the way
    /// into the following bin, so the width is `7/6` bins. Measured from the *second* maximum it
    /// would be 2 bins, which is the answer a peak search that keeps the last tie returns.
    #[test]
    fn the_width_is_measured_from_the_first_of_two_equal_maxima() {
        let edge = SlantedEdge {
            angle_degrees: 5.0,
            bin_width: 1.0,
            esf: vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            lsf: vec![0.0, 2.0, 0.5, 2.0, 1.5, 0.0],
            centre: 1.0,
        };
        let got = edge.fwhm().expect("the profile has a peak");
        assert!((got - 7.0 / 6.0).abs() < 1e-15, "FWHM {got}");
    }

    /// The 50 % crossing is the *rising* one, and a sample sitting exactly on the level belongs to
    /// the interval that rises through it rather than to the one that starts on it.
    #[test]
    fn the_centre_is_the_first_rising_crossing() {
        assert_eq!(crossing_up(&[0.0, 0.25, 0.75, 1.0], 0.5), Some(1.5));
        assert_eq!(crossing_up(&[0.0, 0.5, 0.5, 1.0], 0.5), Some(2.0));
        assert_eq!(crossing_up(&[0.0, 0.1, 0.2], 0.5), None);
    }

    /// The homogeneity test keeps the neighbourhood that is only noise.
    ///
    /// A checkerboard ring around a mid-level centre alternates eight times, which is as far from a
    /// uniform pattern as a ring gets; an edge through the same window makes one run above and one
    /// below, which is two transitions and is rejected. The kept window's mean is `(4 + ½)/9 = ½` and
    /// its population deviation is `√(2/9)`, which the `√(9/8)` Bessel factor lifts to exactly `½`.
    #[test]
    fn the_homogeneity_test_keeps_the_neighbourhood_that_is_only_noise() {
        let checkerboard = [1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 1.0];
        let image = Image {
            data: &checkerboard,
            h: 3,
            w: 3,
        };
        let got = snr_homogeneous_area(&image).expect("the ring is not uniform");
        assert_eq!(got.samples, 1);
        assert!((got.mean - 0.5).abs() < 1e-15, "mean {}", got.mean);
        assert!((got.sigma - 0.5).abs() < 1e-15, "sigma {}", got.sigma);
        assert!((got.snr - 1.0).abs() < 1e-15, "SNR {}", got.snr);

        let step = [0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.0, 0.0, 1.0];
        let image = Image {
            data: &step,
            h: 3,
            w: 3,
        };
        assert_eq!(snr_homogeneous_area(&image), None, "an edge is not noise");
    }

    /// The block estimate is the median mean over the median σ of the flattest blocks.
    ///
    /// Four 2×2 blocks with deviations 1, 1, 2 and 4; keeping half of them leaves the first two,
    /// whose means are 1 and 2 and whose deviations are both 1.
    #[test]
    fn the_block_estimate_is_the_median_of_the_flattest_blocks() {
        #[rustfmt::skip]
        let data = [
            0.0, 2.0, 1.0, 3.0,
            0.0, 2.0, 1.0, 3.0,
            0.0, 4.0, 0.0, 8.0,
            0.0, 4.0, 0.0, 8.0,
        ];
        let image = Image {
            data: &data,
            h: 4,
            w: 4,
        };
        let got = snr_homogeneous_blocks(&image, 2, 0.5).expect("four blocks");
        assert_eq!(got.samples, 2);
        assert_eq!(got.mean, 1.5);
        assert_eq!(got.sigma, 1.0);
        assert_eq!(got.snr, 1.5);
        assert_eq!(snr_homogeneous_blocks(&image, 1, 0.5), None, "a 1×1 block");
        assert_eq!(snr_homogeneous_blocks(&image, 3, 0.5), None, "one block");
    }

    /// Down-sampling averages the source footprint, and the footprint is rectangular in general.
    ///
    /// A 4×6 patch at 0.4 lands on 2×2 outputs, so each one averages two rows by three columns — the
    /// case that tells a product of the two extents from their sum.
    ///
    /// The field is deliberately *not* a ramp. Bilinear interpolation at the centre of a footprint
    /// reproduces the mean of a linear field exactly, so on a ramp the two branches of `resample`
    /// return the same numbers and the choice between them is unobservable. Here each axis carries a
    /// step, so an interpolated sample and an average are different answers.
    #[test]
    fn down_sampling_averages_a_rectangular_footprint() {
        let rows = [0.0, 0.0, 12.0, 36.0];
        let columns = [0.0, 3.0, 9.0, 0.0, 6.0, 18.0];
        let data: Vec<f64> = rows
            .iter()
            .flat_map(|r| columns.iter().map(move |c| r + c))
            .collect();
        let image = Image {
            data: &data,
            h: 4,
            w: 6,
        };
        let (small, h, w) = resample(&image, 0.4);
        assert_eq!((h, w), (2, 2));
        // Row means 0 and 24, column means 4 and 8, and the field is their sum.
        assert_eq!(small, [4.0, 8.0, 28.0, 32.0]);
    }

    /// Up-sampling is bilinear on the pixel centres, with the border clamped.
    #[test]
    fn up_sampling_interpolates_between_pixel_centres() {
        let data = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        let image = Image {
            data: &data,
            h: 2,
            w: 4,
        };
        let (doubled, h, w) = resample(&image, 2.0);
        assert_eq!((h, w), (4, 8));
        // Output (0,1) sits a quarter of the way between source columns 0 and 1 of row 0.
        assert_eq!(doubled[1], 1.5);
        // Output (1,1) is a further quarter of the way down towards row 1's 9.5.
        assert_eq!(doubled[9], 3.5);
        // The far corner falls outside every source centre and clamps to the corner sample.
        assert_eq!(doubled[31], 15.0);

        // A patch tall enough for an interior output to sit between two interior rows: at (3,3) the
        // sample falls a quarter of the way into both source intervals, and the two interpolations
        // compose to 13.75. The 2×4 case above cannot see the row fraction at all, because at its
        // only clamped row the two taps are the same pixel.
        let tall: Vec<f64> = (0..4)
            .flat_map(|y| (0..4).map(move |x| f64::from(10 * y + x)))
            .collect();
        let (grid, h, w) = resample(
            &Image {
                data: &tall,
                h: 4,
                w: 4,
            },
            2.0,
        );
        assert_eq!((h, w), (8, 8));
        assert_eq!(grid[3 * 8 + 3], 13.75);
    }

    /// The blur kernel is a Gaussian truncated at four sigma and renormalised to sum to one.
    ///
    /// On a unit impulse far from every border the centre of the result is `k(0)²`, and `k(0)` is
    /// `1/Σ`, the sum running over the nine taps `−4σ…4σ` at `σ = 1`. A kernel truncated anywhere
    /// else, or left unnormalised, misses it.
    #[test]
    fn the_blur_kernel_is_truncated_at_four_sigma_and_sums_to_one() {
        let mut data = vec![0.0; 9 * 9];
        data[4 * 9 + 4] = 1.0;
        let image = Image {
            data: &data,
            h: 9,
            w: 9,
        };
        let blurred = gaussian_blur(&image, 1.0);
        let total: f64 = (-4..=4)
            .map(|d| libm::exp(-0.5 * f64::from(d) * f64::from(d)))
            .sum();
        let want = 1.0 / (total * total);
        assert!(
            (blurred[4 * 9 + 4] - want).abs() < 1e-15,
            "peak {}, want {want}",
            blurred[4 * 9 + 4]
        );
        // A blur is not a brightness change: the impulse's energy is conserved, and a flat field
        // comes back flat whatever the border does.
        let sum: f64 = blurred.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "the impulse lost energy: {sum}");
        let flat = vec![0.75; 9 * 9];
        for value in gaussian_blur(
            &Image {
                data: &flat,
                h: 9,
                w: 9,
            },
            1.0,
        ) {
            assert!(
                (value - 0.75).abs() < 1e-15,
                "a flat field moved to {value}"
            );
        }
    }

    /// The noise is Box–Muller on the module's own stream, added to the signal rather than taken
    /// from it.
    #[test]
    fn the_added_noise_is_the_box_muller_pair_of_the_stream() {
        let data = [0.0, 0.0];
        let image = Image {
            data: &data,
            h: 1,
            w: 2,
        };
        let got = add_gaussian_noise(&image, 2.0, 99);
        let mut rng = Rng::new(99);
        let radius = libm::sqrt(-2.0 * libm::log(rng.open_unit()));
        let angle = rng.azimuth();
        assert_eq!(got[0], 2.0 * radius * libm::cos(angle));
        assert_eq!(got[1], 2.0 * radius * libm::sin(angle));
    }

    /// How far the bins reach, and where the ESF's two plateaux put its ends.
    ///
    /// The 64×48 fixture's edge runs from 20.0512 to 27.8254 across the patch, so the reach is
    /// limited by the right side — `48 − 2 − 27.8254 = 18.1746` pixels, times `cos 7.03°` — and
    /// `2·18.0373/¼` is 144 bins. That number is the whole binning geometry in one integer.
    ///
    /// The fixture is offset to `[0.2, 0.8]` because every metric is a difference or a ratio and so
    /// cannot see the normalisation at all: only the ESF's own ends can, and after normalising they
    /// must be 0 and 1 whatever the plateaux were.
    #[test]
    fn the_bins_reach_as_far_as_every_row_can_feed_them() {
        let data = offset_edge(64, 48, SIGMA, 0.2, 0.6);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");
        assert_eq!(edge.esf.len(), 144, "the bin count is the reach");
        assert_eq!(edge.lsf.len(), 143, "one difference fewer than bins");
        assert!(
            edge.esf[0].abs() < 1e-12,
            "the ESF starts at {}",
            edge.esf[0]
        );
        let last = edge.esf[143];
        assert!((last - 1.0).abs() < 1e-12, "the ESF ends at {last}");
        // The edge sits in the middle of a reach that is symmetric about it by construction.
        assert!(
            (edge.centre - 71.5).abs() < 1.0,
            "the 50 % crossing is at bin {}",
            edge.centre
        );
    }

    /// Eight rows and eight columns are enough; seven of either are not.
    ///
    /// The bound is a floor, not a preference — a patch one row short must be refused rather than
    /// measured, and refusing it needs *either* dimension to be short, not both.
    #[test]
    fn the_patch_floor_is_eight_by_eight() {
        let square = gaussian_edge(8, 8, SIGMA);
        assert!(
            analyse_slanted_edge(
                &Image {
                    data: &square,
                    h: 8,
                    w: 8
                },
                &EdgeConfig::default()
            )
            .is_ok(),
            "eight by eight is the smallest patch that is measured"
        );
        let short = gaussian_edge(7, 8, SIGMA);
        assert_eq!(
            analyse_slanted_edge(
                &Image {
                    data: &short,
                    h: 7,
                    w: 8
                },
                &EdgeConfig::default()
            )
            .unwrap_err(),
            EdgeError::TooSmall
        );
        let narrow = gaussian_edge(8, 7, SIGMA);
        assert_eq!(
            analyse_slanted_edge(
                &Image {
                    data: &narrow,
                    h: 8,
                    w: 7
                },
                &EdgeConfig::default()
            )
            .unwrap_err(),
            EdgeError::TooSmall
        );
    }

    /// A patch at exactly the contrast floor is measured, and its edge is located from a gradient
    /// whose total is that contrast rather than one.
    ///
    /// The default floor is 0.05, and a Gaussian edge scaled to span exactly 0.05 sits on it: the
    /// test is `<`, so the patch is kept. It is also the only fixture here whose per-row gradient
    /// mass is not 1, which is what separates dividing the first moment by it from multiplying.
    #[test]
    fn a_patch_at_exactly_the_contrast_floor_is_kept() {
        let data = offset_edge(64, 48, SIGMA, 0.0, 0.05);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        let edge = analyse_slanted_edge(&image, &EdgeConfig::default())
            .expect("0.05 is not below a floor of 0.05");
        let want = libm::erf(0.5 / (SIGMA * core::f64::consts::SQRT_2));
        assert!(
            (edge.rer() - want).abs() < 0.01,
            "RER {} on a 5 % edge, want {want}",
            edge.rer()
        );
    }

    /// An edge that is not a line is refused, and one that only wobbles is not.
    ///
    /// The refusal is on the root-mean-square residual of the straight-line fit, in pixels. A three-
    /// pixel bend gives ≈ 2.1 and is refused; a 0.4-pixel one gives ≈ 0.28 and is measured. Both
    /// halves are needed: a residual that sums the errors rather than their squares is zero for
    /// *any* least-squares fit, and one that multiplies by the row count instead of dividing refuses
    /// the wobble as well.
    #[test]
    fn a_bent_edge_is_refused_and_a_wobbly_one_is_not() {
        let bent = bent_edge(64, 48, SIGMA, 3.0);
        assert_eq!(
            analyse_slanted_edge(
                &Image {
                    data: &bent,
                    h: 64,
                    w: 48
                },
                &EdgeConfig::default()
            )
            .unwrap_err(),
            EdgeError::NotStraight
        );
        let wobbly = bent_edge(64, 48, SIGMA, 0.4);
        assert!(
            analyse_slanted_edge(
                &Image {
                    data: &wobbly,
                    h: 64,
                    w: 48
                },
                &EdgeConfig::default()
            )
            .is_ok(),
            "a 0.4-pixel wobble is inside the half-pixel tolerance"
        );
    }

    /// An edge too close to a border leaves no reach to bin, whatever the patch's size.
    #[test]
    fn an_edge_against_the_border_leaves_nothing_to_bin() {
        let mut data = vec![0.0; 16 * 16];
        for y in 0..16 {
            let edge = 14.5 + SLANT * (y as f64 - 8.0);
            for x in 0..16 {
                let d = x as f64 - edge;
                data[y * 16 + x] = 0.5 * (1.0 + libm::erf(d / (SIGMA * core::f64::consts::SQRT_2)));
            }
        }
        assert_eq!(
            analyse_slanted_edge(
                &Image {
                    data: &data,
                    h: 16,
                    w: 16
                },
                &EdgeConfig::default()
            )
            .unwrap_err(),
            EdgeError::TooSmall
        );
    }

    /// The reach is measured from whichever border the edge leans towards.
    ///
    /// Every other fixture here leans right from the middle of the patch, where the right border is
    /// the near one and the left end of the edge is simply the intercept. That makes the *other*
    /// half of the reach arithmetic — the row at the far end of the patch, and the subtraction of
    /// the border margin from it — unobservable. This edge leans left and starts left of centre, so
    /// the binding side is the one the intercept alone cannot supply.
    ///
    /// The edge runs from 23.9488 down to 16.1746, so the reach is `16.1746 − 1` pixels times
    /// `cos 7.03°`, and `2·15.0603/¼` is 120 bins.
    #[test]
    fn the_reach_is_measured_from_the_border_the_edge_leans_towards() {
        let data = slanted_edge_at(64, 48, SIGMA, 20.0, -SLANT);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");
        assert!(edge.angle_degrees < 0.0, "the slant is to the left");
        assert_eq!(edge.esf.len(), 120, "the bin count is the reach");

        // The LSF is the ESF's own difference quotient. Nothing else asserts its *scale*: RER reads
        // the ESF, FWHM is a fraction of the peak, and MTF divides by its own DC term, so all three
        // are blind to a factor on the derivative.
        let peak = (0..edge.lsf.len())
            .max_by(|&a, &b| edge.lsf[a].total_cmp(&edge.lsf[b]))
            .expect("a non-empty LSF");
        let want = (edge.esf[peak + 1] - edge.esf[peak]) / edge.bin_width;
        assert_eq!(
            edge.lsf[peak], want,
            "the LSF is not the difference quotient"
        );
        // The quotient carries the derivative's own scale: a Gaussian LSF peaks at 1/(σ√2π), which
        // is 0.499 for this edge. A quantity that merely looked like a derivative would not.
        assert!((want - 0.4987).abs() < 0.02, "the LSF peaks at {want}");
    }

    /// The plateaux are what the ESF's zero and one are anchored to, and how many bins they are
    /// matters when the patch is small.
    ///
    /// On a wide patch the leading bins sit tens of sigma into the tail and are flat to machine
    /// precision, so the length of the run being averaged cannot be seen. On a patch this small the
    /// run is still on the rising part of the profile, which is where the choice becomes an answer.
    #[test]
    fn the_plateaux_are_where_the_normalisation_is_anchored() {
        let data = gaussian_edge(8, 8, SIGMA);
        let image = Image {
            data: &data,
            h: 8,
            w: 8,
        };
        let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).expect("a clean edge");
        let bins = edge.esf.len();
        let plateau = (bins / 10).max(1);
        assert_ne!(
            plateau,
            (bins % 10).max(1),
            "with {bins} bins the plateau length would not be observable"
        );
        let head = edge.esf[..plateau].iter().sum::<f64>() / plateau as f64;
        let tail = edge.esf[bins - plateau..].iter().sum::<f64>() / plateau as f64;
        assert!(
            head.abs() < 1e-12,
            "the leading plateau normalised to {head}"
        );
        assert!(
            (tail - 1.0).abs() < 1e-12,
            "the trailing plateau normalised to {tail}"
        );
    }

    /// The reach is a real boundary: a pixel past it is dropped, not folded into the end bin.
    ///
    /// The 8×8 patch's reach is set by its right border, so four more columns on the *left* leave it
    /// exactly where it was and leave every bin fed by exactly the same pixels. The four new columns
    /// all project past the low end of the binning. Dropping them and folding them into the first
    /// bin are 2 % apart in RER and 5 % apart in MTF here — on a patch this small the first bin is
    /// still on the rising part of the profile, where an average over everything to the left of it
    /// is a different number.
    #[test]
    fn a_pixel_beyond_the_reach_is_dropped_not_folded_into_the_end_bin() {
        let narrow = slanted_edge_at(8, 8, SIGMA, 4.0, SLANT);
        let wide = slanted_edge_at(8, 12, SIGMA, 8.0, SLANT);
        let close = analyse_slanted_edge(
            &Image {
                data: &narrow,
                h: 8,
                w: 8,
            },
            &EdgeConfig::default(),
        )
        .expect("a clean edge");
        let padded = analyse_slanted_edge(
            &Image {
                data: &wide,
                h: 8,
                w: 12,
            },
            &EdgeConfig::default(),
        )
        .expect("a clean edge");
        assert_eq!(
            close.esf.len(),
            padded.esf.len(),
            "the same reach means the same bins"
        );
        for (i, (&here, &there)) in close.esf.iter().zip(&padded.esf).enumerate() {
            assert!(
                (here - there).abs() < 1e-9,
                "bin {i} moved from {here} to {there} on columns outside the reach"
            );
        }
        assert!(
            (close.rer() - padded.rer()).abs() < 1e-9,
            "RER {} against {}",
            close.rer(),
            padded.rer()
        );
    }

    /// A step whose contrast is exactly the configured floor is measured, not refused.
    ///
    /// The floor is what the contrast may not fall *below*. Saying so needs a patch that sits on it
    /// exactly, and only a step with no point-spread function does: its plateau means are exactly
    /// the two pixel values, so the measured span is exactly the one it was built with.
    #[test]
    fn a_step_at_exactly_the_contrast_floor_is_measured() {
        let data = hard_step_edge(64, 48, 0.25, 0.5);
        let image = Image {
            data: &data,
            h: 64,
            w: 48,
        };
        let config = EdgeConfig {
            min_contrast: 0.5,
            ..EdgeConfig::default()
        };
        let edge = analyse_slanted_edge(&image, &config).expect("0.5 is not below a floor of 0.5");
        // A step with no blur crosses the whole rise inside one pixel.
        assert!(edge.rer() > 0.99, "RER {} on an unblurred step", edge.rer());
        let config = EdgeConfig {
            min_contrast: 0.5 + f64::EPSILON,
            ..EdgeConfig::default()
        };
        assert_eq!(
            analyse_slanted_edge(&image, &config).unwrap_err(),
            EdgeError::LowContrast
        );
    }

    /// Four blocks is the floor, and it is their product that has to reach it.
    ///
    /// A 2×6 patch in 2×2 blocks is one row of three: enough blocks to sort, not enough to take a
    /// median of the flattest half and call it noise.
    #[test]
    fn a_grid_of_fewer_than_four_blocks_is_not_an_estimate() {
        let data: Vec<f64> = (0..12).map(f64::from).collect();
        let image = Image {
            data: &data,
            h: 2,
            w: 6,
        };
        assert!(
            snr_homogeneous_blocks(&image, 2, 0.5).is_none(),
            "one row of three blocks is not four blocks"
        );
    }

    /// Every block is read from its own rows, and when none is discarded every one of them shows.
    ///
    /// Keeping the flattest half hides a block that was read from the wrong rows, because a block
    /// that straddles two scene levels is exactly the one the ordering throws away. Keeping all of
    /// them is what makes the row range of each block observable.
    #[test]
    fn every_block_is_read_from_its_own_rows() {
        let data = [
            0.0, 0.0, 10.0, 10.0, //
            0.0, 0.0, 10.0, 10.0, //
            4.0, 4.0, 20.0, 26.0, //
            4.0, 8.0, 20.0, 20.0,
        ];
        let image = Image {
            data: &data,
            h: 4,
            w: 4,
        };
        let estimate = snr_homogeneous_blocks(&image, 2, 1.0).expect("four blocks");
        assert_eq!(estimate.samples, 4, "no block was discarded");
        // Means 0, 10, 5 and 21.5; deviations 0, 0, √3 and √6.75.
        assert_eq!(estimate.mean, 7.5, "the median of the four block means");
        let want = libm::sqrt(3.0) * 0.5;
        assert!(
            (estimate.sigma - want).abs() < 1e-15,
            "the median of the four block deviations is {}, want {want}",
            estimate.sigma
        );
    }

    /// A blank image has no noise, so the ratio is infinite by fiat rather than a nought over nought.
    #[test]
    fn a_blank_image_has_infinite_signal_to_noise() {
        let data = [0.0; 16];
        let image = Image {
            data: &data,
            h: 4,
            w: 4,
        };
        let estimate = snr_homogeneous_blocks(&image, 2, 0.5).expect("four blocks");
        assert_eq!(estimate.snr, f64::INFINITY, "a flat field has no noise");
    }

    /// An image with no rows is refused before anything indexes it.
    ///
    /// The guard is on *either* dimension being short of three, not on both: a zero-height patch
    /// with a usable width still has no interior row, and the loop bound that would find one
    /// underflows before it can say so.
    #[test]
    fn an_image_with_no_rows_is_refused_before_it_is_indexed() {
        let image = Image {
            data: &[],
            h: 0,
            w: 4,
        };
        assert!(snr_homogeneous_area(&image).is_none());
        let data = [0.0; 8];
        assert!(
            snr_homogeneous_area(&Image {
                data: &data,
                h: 4,
                w: 2
            })
            .is_none(),
            "two columns leave no interior pixel either"
        );
    }

    /// The eight neighbours are read once each, from their own places around the centre.
    ///
    /// The local binary pattern is a walk around the ring, so the sampling has to be a ring: an
    /// index that lands on the wrong neighbour, or twice on the same one, changes both the count of
    /// transitions and the nine values the moments are taken over. Every neighbour here is a
    /// different number, so any permutation of the eight is a different answer.
    ///
    /// Only the middle window survives the homogeneity test. Its ring alternates above and below the
    /// centre all the way round — eight transitions, the least ordered a 3×3 can be — while the two
    /// windows either side of it are a ramp against a flat border and give two.
    #[test]
    fn the_neighbourhood_ring_is_read_in_order_around_the_centre() {
        let data = [
            0.0, 0.0, 0.0, //
            8.0, 1.0, 7.0, //
            4.0, 4.5, 2.0, //
            5.0, 3.0, 6.0, //
            9.0, 9.0, 9.0,
        ];
        let image = Image {
            data: &data,
            h: 5,
            w: 3,
        };
        let estimate = snr_homogeneous_area(&image).expect("one window is nothing but noise");
        assert_eq!(estimate.samples, 1, "only the middle window is kept");
        assert_eq!(estimate.mean, 4.5, "the mean of 1..8 and the centre");
        // Σ(v − 4.5)² over the nine is 42, so σ is √(42/9) before the 3×3 Bessel factor √(9/8).
        let want = libm::sqrt(42.0 / 8.0);
        assert!(
            (estimate.sigma - want).abs() < 1e-15,
            "sigma {}, want {want}",
            estimate.sigma
        );
        assert!(
            (estimate.snr - 4.5 / want).abs() < 1e-15,
            "snr {}",
            estimate.snr
        );
    }

    /// The deviation is taken about the mean in two passes, which survives an offset that the
    /// one-pass shortcut does not.
    ///
    /// `Σ(v − μ)²/n` and `Σv²/n − μ²` — which is what `Σ(v + μ)(v − μ)/n` is — are the same number
    /// in ℝ and not the same number in f64. The second forms products around `2μ²`, and the spread
    /// it is trying to measure is what falls off the bottom of them. Image data carries this shape
    /// whenever a large pedestal sits under a small noise, which is most of the time.
    ///
    /// The pedestal here is large enough that the products' own last bits are the ones being
    /// asked for: at `2·10¹⁵` an ulp is a quarter, and the products differ from each other by
    /// quarters. A smaller pedestal is not a weaker version of this test but no test at all — at
    /// `10¹⁰` every product below is exact and the two forms agree to the last bit.
    #[test]
    fn the_deviation_survives_a_pedestal_the_shortcut_formula_does_not() {
        let pedestal = 1e15;
        let values = [pedestal, pedestal + 1.0, pedestal + 2.0, pedestal + 3.0];
        let (mean, sigma) = moments(&values);
        assert_eq!(mean, pedestal + 1.5, "the mean is exact in f64 here");
        let want = libm::sqrt(1.25);
        assert!(
            (sigma / want - 1.0).abs() < 1e-15,
            "sigma {sigma}, want {want}"
        );
        // What the shortcut would have returned on the same four numbers.
        let shortcut = libm::sqrt(
            values.iter().map(|&v| (v + mean) * (v - mean)).sum::<f64>() / values.len() as f64,
        );
        assert!(
            (shortcut / want - 1.0).abs() > 1e-9,
            "the pedestal is too small to separate the two forms: {shortcut}"
        );
    }

    /// The kernel is truncated at four sigma, so its reach follows sigma rather than being fixed.
    #[test]
    fn the_blur_reaches_four_sigma_and_no_further() {
        let mut data = vec![0.0; 21 * 21];
        data[10 * 21 + 10] = 1.0;
        let image = Image {
            data: &data,
            h: 21,
            w: 21,
        };
        let blurred = gaussian_blur(&image, 2.0);
        assert!(
            blurred[10 * 21 + 17] > 0.0,
            "seven pixels is inside four sigma of a two-sigma blur"
        );
        assert_eq!(
            blurred[10 * 21 + 19],
            0.0,
            "nine pixels is outside the truncation"
        );
        let sum: f64 = blurred.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "the impulse lost energy: {sum}");
    }

    /// The grid is `levels` steps over `[0, full_scale]`, so the scale is what the levels are counted
    /// against — a unit full scale is the one case where dividing by it and multiplying agree.
    #[test]
    fn quantisation_counts_its_levels_against_the_full_scale() {
        let data = [0.0, 0.9, 1.1, 3.0, 7.5, 9.0, -1.0];
        let image = Image {
            data: &data,
            h: 1,
            w: 7,
        };
        // Five levels over [0, 8]: the grid is 0, 2, 4, 6, 8, and 9 is above full scale.
        assert_eq!(
            quantise(&image, 5, 8.0),
            [0.0, 0.0, 2.0, 4.0, 8.0, 8.0, 0.0]
        );
    }
}
