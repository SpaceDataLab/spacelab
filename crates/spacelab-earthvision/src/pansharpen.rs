//! **Pan-sharpening** — fusing a high-resolution panchromatic band with low-resolution multispectral
//! bands to synthesise high-resolution colour, by component substitution.
//!
//! - [`brovey_pan_sharpen`] — the Brovey transform `MSₖ·PAN/ΣMS`: it moves all the spatial detail of
//!   `PAN` into every band while exactly preserving the inter-band ratios (hue), at the cost of overall
//!   radiometry.
//! - [`gram_schmidt_pan_sharpen`] — the Gram–Schmidt component substitution (Laben & Brower): build a
//!   simulated low-resolution pan as the mean of the bands, Gram–Schmidt-orthogonalise
//!   `[PAN_sim, bands…]`, replace the first component with the real (histogram-matched) high-resolution
//!   pan, and invert — the standard, higher-fidelity method. If the real pan carries no detail beyond
//!   the simulated one, it is the identity.
//!
//! Bands are equal-length row-major `f64` slices (any pixel count); the multispectral bands are taken
//! already upsampled to the pan grid. `no_std` + `alloc`.

// Short names (b, k, j, p) index bands/pixels; the doc names methods (Brovey, Gram–Schmidt).
#![allow(
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]

use alloc::vec;
use alloc::vec::Vec;

/// Brovey pan-sharpen: `outₖ[p] = MSₖ[p]·PAN[p]/Σₖ MSₖ[p]`. Returns one sharpened band per input band.
/// A pixel whose bands sum to zero is left at zero.
#[must_use]
pub fn brovey_pan_sharpen(ms_bands: &[&[f64]], pan: &[f64]) -> Vec<Vec<f64>> {
    let n = ms_bands.len();
    let pixels = pan.len();
    let mut out = vec![vec![0.0; pixels]; n];
    for p in 0..pixels {
        let sum: f64 = ms_bands.iter().map(|b| b[p]).sum();
        if sum.abs() > 1e-300 {
            let scale = pan[p] / sum;
            for k in 0..n {
                out[k][p] = ms_bands[k][p] * scale;
            }
        }
    }
    out
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_dev(v: &[f64], mu: f64) -> f64 {
    libm::sqrt(v.iter().map(|&x| (x - mu) * (x - mu)).sum::<f64>() / v.len() as f64)
}

/// Gram–Schmidt component-substitution pan-sharpen. The simulated low-resolution pan is the per-pixel
/// mean of the bands; the real `pan` is histogram-matched (mean/σ) to it before substitution. Returns
/// one sharpened band per input band.
#[must_use]
pub fn gram_schmidt_pan_sharpen(ms_bands: &[&[f64]], pan: &[f64]) -> Vec<Vec<f64>> {
    let n = ms_bands.len();
    let pixels = pan.len();
    if n == 0 {
        return Vec::new();
    }
    // Simulated low-resolution pan (equal-weight mean of the bands) and the band means.
    let mut sim = vec![0.0; pixels];
    for p in 0..pixels {
        sim[p] = ms_bands.iter().map(|b| b[p]).sum::<f64>() / n as f64;
    }
    let band_mean: Vec<f64> = ms_bands.iter().map(|b| mean(b)).collect();

    // Forward Gram–Schmidt: GS_0 = sim (mean-subtracted), then each band orthogonalised against the
    // earlier GS components. `phi[k][j]` is the projection coefficient of band k onto GS_j.
    let sim_mean = mean(&sim);
    let mut gs: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    gs.push(sim.iter().map(|&x| x - sim_mean).collect());
    let mut phi: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (k, band) in ms_bands.iter().enumerate() {
        let mut comp: Vec<f64> = band.iter().map(|&x| x - band_mean[k]).collect();
        phi[k] = Vec::with_capacity(k + 1);
        for gj in &gs {
            let dot: f64 = comp.iter().zip(gj).map(|(a, b)| a * b).sum();
            let norm: f64 = gj.iter().map(|b| b * b).sum();
            let c = if norm > 1e-300 { dot / norm } else { 0.0 };
            phi[k].push(c);
            for (ci, gjv) in comp.iter_mut().zip(gj) {
                *ci -= c * gjv;
            }
        }
        gs.push(comp);
    }

    // Substitute GS_0 with the histogram-matched high-resolution pan (mean-subtracted).
    let (pmu, psg) = (mean(pan), std_dev(pan, mean(pan)));
    let sim_sg = std_dev(&sim, sim_mean);
    let scale = if psg > 1e-300 { sim_sg / psg } else { 0.0 };
    gs[0] = pan.iter().map(|&x| (x - pmu) * scale).collect();

    // Inverse Gram–Schmidt: MS_k = GS_{k+1} + Σ_{j≤k} phi[k][j]·GS_j + band_mean[k].
    let mut out = vec![vec![0.0; pixels]; n];
    for k in 0..n {
        for p in 0..pixels {
            let mut val = gs[k + 1][p] + band_mean[k];
            for (j, &c) in phi[k].iter().enumerate() {
                val += c * gs[j][p];
            }
            out[k][p] = val;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal, clippy::manual_midpoint)]
    use super::*;

    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f64) / ((1u64 << 24) as f64)
    }

    #[test]
    fn mean_and_std_dev_are_the_first_two_moments() {
        // Neither helper had a test of its own, and the only assertion that mentioned `mean` used it
        // on both sides -- `mean(&out[k]) - mean(band)` is zero for *any* constant `mean`. State the
        // closed forms on 1..=6, whose mean is 3.5 and whose second moment about a given centre is
        // written out below.
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((mean(&v) - 3.5).abs() < 1e-12);
        // Σ(x − 3.5)² = 2(2.5² + 1.5² + 0.5²) = 17.5.
        assert!((std_dev(&v, 3.5) - libm::sqrt(17.5 / 6.0)).abs() < 1e-12);
        // About a centre that is neither zero nor the mean, because those are exactly the two centres
        // at which `(x − μ)(x + μ)` and `(x − μ)²` agree: their means differ by 2μ(μ − x̄).
        // Σ(x − 2)² = 1 + 0 + 1 + 4 + 9 + 16 = 31.
        assert!((std_dev(&v, 2.0) - libm::sqrt(31.0 / 6.0)).abs() < 1e-12);
    }

    #[test]
    fn a_single_band_gram_schmidt_is_the_pan_histogram_matched_to_that_band() {
        // With one band the algorithm has a closed form. The simulated pan *is* the band, so the
        // first Gram-Schmidt component is the band's deviation, the band projects onto it with
        // coefficient exactly 1 and leaves a zero residual, and the inverse is the substituted
        // component alone:
        //
        //     out[p] = mean(b) + (pan[p] − mean(pan))·σ(b)/σ(pan),
        //
        // i.e. the pan matched to the band's mean and standard deviation. That single equation pins
        // the projection coefficient, the histogram-matching scale and the direction it is applied
        // in, none of which the three-band tests can see: the identity test holds for *any* φ,
        // because whatever the forward pass subtracts the inverse adds back.
        let b = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let pan = [0.0, 3.0, 1.0, 5.0, 2.0, 7.0];
        let out = gram_schmidt_pan_sharpen(&[&b], &pan);
        assert_eq!(out.len(), 1);
        let (bmu, bsg) = (mean(&b), std_dev(&b, mean(&b)));
        let (pmu, psg) = (mean(&pan), std_dev(&pan, mean(&pan)));
        for p in 0..b.len() {
            let want = bmu + (pan[p] - pmu) * bsg / psg;
            assert!(
                (out[0][p] - want).abs() < 1e-9,
                "px {p}: {} vs {want}",
                out[0][p]
            );
        }
        // The consequence worth naming: the sharpened band carries the band's radiometry and the
        // pan's spatial structure.
        assert!((mean(&out[0]) - bmu).abs() < 1e-9);
        assert!((std_dev(&out[0], mean(&out[0])) - bsg).abs() < 1e-9);
    }

    #[test]
    fn brovey_carries_pan_intensity_and_preserves_hue() {
        let mut seed = 7u64;
        let pixels = 40;
        let b0: Vec<f64> = (0..pixels).map(|_| 0.2 + lcg(&mut seed)).collect();
        let b1: Vec<f64> = (0..pixels).map(|_| 0.2 + lcg(&mut seed)).collect();
        let b2: Vec<f64> = (0..pixels).map(|_| 0.2 + lcg(&mut seed)).collect();
        let pan: Vec<f64> = (0..pixels).map(|_| 0.5 + lcg(&mut seed)).collect();
        let out = brovey_pan_sharpen(&[&b0, &b1, &b2], &pan);
        for p in 0..pixels {
            // The summed sharpened radiance equals the pan (intensity comes from the pan).
            assert!((out[0][p] + out[1][p] + out[2][p] - pan[p]).abs() < 1e-9);
            // Inter-band ratios (hue) are exactly the input ratios.
            assert!((out[0][p] / out[1][p] - b0[p] / b1[p]).abs() < 1e-9);
            assert!((out[2][p] / out[1][p] - b2[p] / b1[p]).abs() < 1e-9);
        }
    }

    #[test]
    fn gram_schmidt_is_the_identity_when_the_pan_adds_no_detail() {
        // If the real pan equals the simulated low-res pan (the band mean), GS substitution injects no
        // new detail — the sharpened bands must return the inputs exactly.
        let mut seed = 11u64;
        let pixels = 50;
        let b0: Vec<f64> = (0..pixels).map(|_| 0.3 + lcg(&mut seed)).collect();
        let b1: Vec<f64> = (0..pixels).map(|_| 0.3 + lcg(&mut seed)).collect();
        let b2: Vec<f64> = (0..pixels).map(|_| 0.3 + lcg(&mut seed)).collect();
        let bands = [b0.as_slice(), b1.as_slice(), b2.as_slice()];
        let sim: Vec<f64> = (0..pixels).map(|p| (b0[p] + b1[p] + b2[p]) / 3.0).collect();
        let out = gram_schmidt_pan_sharpen(&bands, &sim);
        for (k, band) in bands.iter().enumerate() {
            for p in 0..pixels {
                assert!((out[k][p] - band[p]).abs() < 1e-9, "band {k} px {p}");
            }
        }
    }

    #[test]
    fn gram_schmidt_preserves_band_means_and_injects_pan_detail() {
        // A real pan with genuine high-frequency detail (a per-pixel checkerboard the smooth bands
        // lack): the sharpened bands keep their radiometric mean (spectral fidelity — histogram matching
        // holds the overall variance to the simulated pan) yet take on the pan's spatial pattern.
        let pixels = 64;
        let smooth = |p: usize| 0.5 + 0.1 * ((p / 8) as f64);
        let b0: Vec<f64> = (0..pixels).map(smooth).collect();
        let b1: Vec<f64> = (0..pixels).map(|p| smooth(p) + 0.05).collect();
        let bands = [b0.as_slice(), b1.as_slice()];
        let pan: Vec<f64> = (0..pixels)
            .map(|p| (b0[p] + b1[p]) / 2.0 + if p % 2 == 0 { 0.15 } else { -0.15 })
            .collect();
        let out = gram_schmidt_pan_sharpen(&bands, &pan);
        // The mean of the even minus odd pixels — the checkerboard the pan carries.
        let checker = |v: &[f64]| {
            let e: f64 = v.iter().step_by(2).sum();
            let o: f64 = v.iter().skip(1).step_by(2).sum();
            (e - o) / (v.len() as f64 / 2.0)
        };
        for (k, band) in bands.iter().enumerate() {
            assert!(
                (mean(&out[k]) - mean(band)).abs() < 1e-9,
                "mean drifted, band {k}"
            );
            assert!(
                checker(band).abs() < 1e-9,
                "input band already had detail, {k}"
            );
            assert!(
                checker(&out[k]).abs() > 0.05,
                "no pan detail injected, band {k}"
            );
        }
    }
}
