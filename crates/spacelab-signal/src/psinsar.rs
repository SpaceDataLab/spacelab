//! InSAR time-series — **persistent-scatterer** selection and the **SBAS** velocity/DEM-error
//! inversion over a stack of interferograms.
//!
//! A pixel is a good phase measurement where its brightness is stable across the stack: the **amplitude
//! dispersion** `D_A = σ_A/μ_A` ([`amplitude_dispersion`]) is a proxy for phase stability, and
//! [`select_persistent_scatterers`] keeps the low-`D_A` pixels (the PS-InSAR criterion of Ferretti et
//! al.). For each such scatterer, the unwrapped interferometric phases across the stack are
//! ([`sbas_invert`]) a linear function of the line-of-sight **deformation velocity** `v` and the
//! residual **DEM error** `Δh`:
//!
//! ```text
//! φᵢ = (4π/λ)·v·Δtᵢ  +  (4π·B⊥ᵢ)/(λ·r·sinθ)·Δh  +  noise,
//! ```
//!
//! so a two-parameter least squares over the temporal (`Δtᵢ`) and perpendicular (`B⊥ᵢ`) baselines
//! recovers `(v, Δh)` — the core of the Small-BAseline-Subset method. `alloc` for the buffers.

// Short physical names (v, r, θ) and the SAR terms in the docs mirror the InSAR literature.
#![allow(
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]

use alloc::vec::Vec;

const FOUR_PI: f64 = 4.0 * core::f64::consts::PI;

/// The amplitude dispersion `D_A = σ_A/μ_A` of every pixel over a `stack` of `pixels`-length amplitude
/// images — the PS-InSAR phase-stability proxy. A pixel with zero mean amplitude gets `∞`.
#[must_use]
pub fn amplitude_dispersion(stack: &[&[f64]], pixels: usize) -> Vec<f64> {
    let n = stack.len().max(1) as f64;
    (0..pixels)
        .map(|p| {
            let mean = stack.iter().map(|img| img[p]).sum::<f64>() / n;
            if mean.abs() < 1e-300 {
                return f64::INFINITY;
            }
            let var = stack
                .iter()
                .map(|img| (img[p] - mean) * (img[p] - mean))
                .sum::<f64>()
                / n;
            libm::sqrt(var) / mean
        })
        .collect()
}

/// The pixel indices whose amplitude `dispersion` is below `threshold` — the persistent scatterers.
#[must_use]
pub fn select_persistent_scatterers(dispersion: &[f64], threshold: f64) -> Vec<usize> {
    (0..dispersion.len())
        .filter(|&p| dispersion[p] < threshold)
        .collect()
}

/// Invert one scatterer's unwrapped interferometric `phases` (radians) for the line-of-sight
/// deformation `velocity` and residual DEM error, by least squares over the temporal baselines
/// `temporal_baselines` (same time unit as the returned velocity) and perpendicular baselines
/// `perp_baselines` (m). `wavelength` (m), slant `range` (m), `incidence` angle (radians). Returns
/// `(velocity, dem_error)`; `dem_error` in metres.
#[must_use]
pub fn sbas_invert(
    phases: &[f64],
    temporal_baselines: &[f64],
    perp_baselines: &[f64],
    wavelength: f64,
    range: f64,
    incidence: f64,
) -> (f64, f64) {
    // Design columns: c0 = (4π/λ)·Δt (velocity), c1 = (4π·B⊥)/(λ·r·sinθ) (DEM error).
    let kv = FOUR_PI / wavelength;
    let kh = FOUR_PI / (wavelength * range * libm::sin(incidence));
    let (mut s00, mut s01, mut s11, mut b0, mut b1) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..phases.len() {
        let c0 = kv * temporal_baselines[i];
        let c1 = kh * perp_baselines[i];
        s00 += c0 * c0;
        s01 += c0 * c1;
        s11 += c1 * c1;
        b0 += c0 * phases[i];
        b1 += c1 * phases[i];
    }
    let det = s00 * s11 - s01 * s01;
    if libm::fabs(det) < 1e-300 {
        return (0.0, 0.0);
    }
    ((s11 * b0 - s01 * b1) / det, (s00 * b1 - s01 * b0) / det)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::unreadable_literal
    )]
    use super::*;

    #[test]
    fn amplitude_dispersion_and_ps_selection() {
        // Pixel 0 is perfectly stable (D_A = 0); pixel 1 has a strong amplitude swing (high D_A). Only
        // the stable one is selected as a persistent scatterer.
        let img0 = [10.0, 10.0];
        let img1 = [10.0, 2.0];
        let img2 = [10.0, 18.0];
        let da = amplitude_dispersion(&[&img0, &img1, &img2], 2);
        assert!(da[0] < 1e-12, "stable D_A {}", da[0]);
        assert!(da[1] > 0.5, "unstable D_A {}", da[1]);
        assert_eq!(select_persistent_scatterers(&da, 0.25), vec![0]);
        // The threshold is exclusive, and a perfectly stable pixel sits exactly on zero -- the one
        // place in this module where the boundary is reachable bit for bit.
        assert_eq!(da[0], 0.0);
        assert_eq!(select_persistent_scatterers(&da, 0.0), [] as [usize; 0]);
    }

    /// `D_A` against its closed form, on a stack whose dispersion is known exactly.
    ///
    /// Six mutants of `amplitude_dispersion` survived, because the fixture above only ever compares
    /// the answer to a band. For amplitudes `m(1 ± r)` and `m(1 ± r/2)` the mean is `m`, the
    /// variance is `0.625 m^2 r^2` and the dispersion is `r * sqrt(0.625)` whatever `m` is -- a
    /// number the code never sees.
    ///
    /// The small-`r` rung is the one that matters. `(a - mean) * (a - mean)` and
    /// `(a + mean) * (a - mean)` are the same sum algebraically, since
    /// `sum (a^2 - mean^2) / n = E[a^2] - mean^2 = var`; they differ only in conditioning, and the
    /// second is the textbook naive form that cancels. At `r = 1e-6` it is wrong in the fifth
    /// significant figure -- 7.905245e-7 against 7.905694e-7 -- which is what separates the two.
    #[test]
    fn amplitude_dispersion_is_its_closed_form_and_keeps_its_digits() {
        for m in [1.0f64, 100.0, 1234.5678] {
            for r in [1e-1f64, 1e-3, 1e-6] {
                let imgs = [
                    [m * (1.0 - r)],
                    [m * (1.0 + r)],
                    [m * (1.0 - 0.5 * r)],
                    [m * (1.0 + 0.5 * r)],
                ];
                let stack: [&[f64]; 4] = [&imgs[0], &imgs[1], &imgs[2], &imgs[3]];
                let got = amplitude_dispersion(&stack, 1)[0];
                let want = r * libm::sqrt(0.625);
                assert!(
                    (got - want).abs() < 1e-8 * want,
                    "m {m} r {r:e}: {got:e} vs {want:e}"
                );
            }
        }
        // A pixel with no signal has no dispersion to report, and the guard is what keeps the
        // ratio off `0/0`.
        let dark = [0.0f64, 0.0];
        let stack: [&[f64]; 2] = [&dark, &dark];
        assert_eq!(amplitude_dispersion(&stack, 2), vec![f64::INFINITY; 2]);
        // The divisor is the stack depth, not one: two images of the same pixel differing by 2d
        // around m give sigma = d exactly, so a mis-scaled variance shows up directly.
        let (lo, hi) = ([6.0f64], [10.0f64]);
        let stack: [&[f64]; 2] = [&lo, &hi];
        assert_eq!(amplitude_dispersion(&stack, 1)[0], 0.25);
    }

    fn geometry() -> (f64, f64, f64) {
        (0.056, 800e3, 38.0_f64.to_radians()) // C-band, 800 km slant range, 38° incidence
    }

    #[test]
    fn sbas_recovers_velocity_and_dem_error_exactly() {
        // Noise-free phases built from a known velocity and DEM error must invert back to them.
        let (lam, r, theta) = geometry();
        let (v_true, dh_true) = (0.02, 15.0); // 2 cm/yr subsidence, 15 m residual DEM
        let tb: Vec<f64> = (0..12).map(|i| 0.3 * f64::from(i) - 1.5).collect(); // years
        let pb: Vec<f64> = (0..12).map(|i| 120.0 * (f64::from(i) - 6.0)).collect(); // metres
        // Written out rather than taken from `FOUR_PI`: an oracle that reuses the constant under
        // test cancels it on both sides, and did -- the two-way travel factor could be any number
        // at all and both inversions below still came back exact.
        let four_pi = 4.0 * core::f64::consts::PI;
        let kv = four_pi / lam;
        let kh = four_pi / (lam * r * theta.sin());
        let phases: Vec<f64> = (0..12)
            .map(|i| kv * v_true * tb[i] + kh * pb[i] * dh_true)
            .collect();
        let (v, dh) = sbas_invert(&phases, &tb, &pb, lam, r, theta);
        assert!((v - v_true).abs() < 1e-9, "v {v}");
        assert!((dh - dh_true).abs() < 1e-6, "dem {dh}");
    }

    #[test]
    fn sbas_is_robust_to_moderate_phase_noise() {
        // The same problem with deterministic phase noise: least squares still recovers the parameters
        // closely (an over-determined 12-observation fit).
        let (lam, r, theta) = geometry();
        let (v_true, dh_true) = (0.03, -20.0);
        let tb: Vec<f64> = (0..16).map(|i| 0.25 * f64::from(i) - 2.0).collect();
        // Perpendicular baselines vary independently of time (real acquisitions do), else the two
        // design columns are collinear and the inversion is degenerate.
        let pb: Vec<f64> = (0..16)
            .map(|i| 150.0 * (f64::from((i * 7) % 11) - 5.0))
            .collect();
        // Written out rather than taken from `FOUR_PI`: an oracle that reuses the constant under
        // test cancels it on both sides, and did -- the two-way travel factor could be any number
        // at all and both inversions below still came back exact.
        let four_pi = 4.0 * core::f64::consts::PI;
        let kv = four_pi / lam;
        let kh = four_pi / (lam * r * theta.sin());
        let mut seed = 42u64;
        let mut noise = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 40) as f64 / (1u64 << 24) as f64 * 0.4 - 0.2 // ±0.2 rad
        };
        let phases: Vec<f64> = (0..16)
            .map(|i| kv * v_true * tb[i] + kh * pb[i] * dh_true + noise())
            .collect();
        let (v, dh) = sbas_invert(&phases, &tb, &pb, lam, r, theta);
        assert!((v - v_true).abs() < 0.1 * v_true.abs(), "v {v} vs {v_true}");
        assert!(
            (dh - dh_true).abs() < 0.1 * dh_true.abs(),
            "dem {dh} vs {dh_true}"
        );
    }
}
