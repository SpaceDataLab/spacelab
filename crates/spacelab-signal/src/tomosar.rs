//! SAR tomography (TomoSAR) — resolving scatterers *along elevation* from a multi-baseline stack, the
//! third SAR dimension.
//!
//! `N` acquisitions of the same range-azimuth cell from slightly different orbits (baselines `bₙ`) see
//! a superposition of scatterers at elevations `s`: `yₙ = Σ γ(s)·exp(−j·ξₙ·s)` with the elevation
//! spatial frequency `ξₙ = 4π bₙ /(λ r)`. Recovering the reflectivity profile `γ(s)` is a
//! spectral-estimation problem over the [`steering_vector`] `a(s) = [exp(−j ξₙ s)]`:
//!
//! - [`bartlett_spectrum`] — Bartlett / beamforming, `P(s) = aᴴ R a / aᴴa`: robust but limited to the
//!   Rayleigh resolution `λr/(2·baseline-aperture)`.
//! - [`capon_spectrum`] — Capon / MVDR, `P(s) = 1/(aᴴ R⁻¹ a)`: adaptive, super-resolves scatterers
//!   closer than Rayleigh.
//! - [`sl1mmer_spectrum`] — **SL1MMER** (Zhu & Bamler): compressive sensing on the raw stack `y`, the
//!   `ℓ₁`-regularised inversion `min ½‖y − Aγ‖² + λ‖γ‖₁` over the [`sensing_matrix`] `A`, solved by
//!   complex **FISTA** ([`fista_lasso`]). Its sparsity prior super-resolves a handful of point
//!   scatterers beyond both Rayleigh and Capon. [`sl1mmer`] runs the full three-step method — the
//!   `ℓ₁` scale-down, **model selection** ([`detect_support`]) and a least-squares
//!   **estimation reconstruction** ([`refit_amplitudes`]) that debiases the amplitudes.
//!
//! `R` is the `N×N` sample covariance of the stack ([`sample_covariance`]). `alloc` for the matrices.

// Physical single letters (b, r, s, a, n) mirror the tomography literature; casts are in range.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use crate::fft::Complex;
use alloc::vec;
use alloc::vec::Vec;

/// The elevation steering vector `a(s) = [exp(−j·4π bₙ s /(λ r))]ₙ` for baselines `baselines` (m),
/// `wavelength` (m), slant `range` (m) and elevation `elevation` (m).
#[must_use]
pub fn steering_vector(
    baselines: &[f64],
    wavelength: f64,
    range: f64,
    elevation: f64,
) -> Vec<Complex> {
    baselines
        .iter()
        .map(|&b| {
            let xi = 4.0 * core::f64::consts::PI * b / (wavelength * range);
            Complex::from_polar(1.0, -xi * elevation)
        })
        .collect()
}

/// The `N×N` sample covariance `R = (1/L)·Σ yₗ yₗᴴ` (row-major) from `L` snapshots (multi-look), each an
/// `N`-vector of the co-registered stack.
#[must_use]
pub fn sample_covariance(snapshots: &[&[Complex]], n: usize) -> Vec<Complex> {
    let mut r = vec![Complex::ZERO; n * n];
    for snap in snapshots {
        for i in 0..n {
            for j in 0..n {
                r[i * n + j] = r[i * n + j] + snap[i] * snap[j].conj();
            }
        }
    }
    let scale = 1.0 / snapshots.len().max(1) as f64;
    for x in &mut r {
        *x = x.scale(scale);
    }
    r
}

/// `aᴴ b = Σ conj(aᵢ)·bᵢ`.
fn inner(a: &[Complex], b: &[Complex]) -> Complex {
    let mut s = Complex::ZERO;
    for (x, y) in a.iter().zip(b) {
        s = s + x.conj() * *y;
    }
    s
}

/// `R·v` for a row-major `n×n` `R`.
fn matvec(r: &[Complex], v: &[Complex], n: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; n];
    for i in 0..n {
        let mut s = Complex::ZERO;
        for j in 0..n {
            s = s + r[i * n + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// Solve the complex linear system `A x = b` (`A` row-major `n×n`) by Gaussian elimination with
/// partial pivoting; `None` if singular.
fn solve_complex(a: &[Complex], b: &[Complex], n: usize) -> Option<Vec<Complex>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        let mut best = m[col * n + col].norm_sqr();
        for r in (col + 1)..n {
            let v = m[r * n + col].norm_sqr();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            x.swap(col, piv);
        }
        let d = m[col * n + col];
        let inv = d.conj().scale(1.0 / d.norm_sqr()); // 1/d
        for r in 0..n {
            if r != col {
                let f = m[r * n + col] * inv;
                for c in col..n {
                    m[r * n + c] = m[r * n + c] - f * m[col * n + c];
                }
                x[r] = x[r] - f * x[col];
            }
        }
    }
    for i in 0..n {
        let d = m[i * n + i];
        x[i] = x[i] * d.conj().scale(1.0 / d.norm_sqr());
    }
    Some(x)
}

/// The Bartlett (beamforming) elevation spectrum `P(s) = aᴴ R a / aᴴa` over the `elevations` grid (m),
/// for the given `covariance` (`N×N`), `baselines`, `wavelength` and `range`.
#[must_use]
pub fn bartlett_spectrum(
    covariance: &[Complex],
    baselines: &[f64],
    wavelength: f64,
    range: f64,
    elevations: &[f64],
) -> Vec<f64> {
    let n = baselines.len();
    elevations
        .iter()
        .map(|&s| {
            let a = steering_vector(baselines, wavelength, range, s);
            let ra = matvec(covariance, &a, n);
            inner(&a, &ra).re / inner(&a, &a).re
        })
        .collect()
}

/// The Capon (MVDR) elevation spectrum `P(s) = 1/(aᴴ R⁻¹ a)` over the `elevations` grid (m). `loading`
/// is the diagonal loading added to `R` for numerical stability (a small fraction of its trace).
#[must_use]
pub fn capon_spectrum(
    covariance: &[Complex],
    baselines: &[f64],
    wavelength: f64,
    range: f64,
    elevations: &[f64],
    loading: f64,
) -> Vec<f64> {
    let n = baselines.len();
    let mut r = covariance.to_vec();
    for i in 0..n {
        r[i * n + i] = r[i * n + i] + Complex::new(loading, 0.0);
    }
    elevations
        .iter()
        .map(|&s| {
            let a = steering_vector(baselines, wavelength, range, s);
            match solve_complex(&r, &a, n) {
                // aᴴ R⁻¹ a is real-positive for a Hermitian positive-definite R.
                Some(x) => 1.0 / inner(&a, &x).re,
                None => 0.0,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------
// SL1MMER — compressive-sensing (ℓ₁) super-resolution TomoSAR.
// ---------------------------------------------------------------------------------------------------

/// The `N×G` sensing matrix `A` (row-major) whose column `g` is the steering vector of elevation
/// `elevations[g]`: `A[n, g] = exp(−j·4π·bₙ·s_g /(λ r))`.
#[must_use]
pub fn sensing_matrix(
    baselines: &[f64],
    wavelength: f64,
    range: f64,
    elevations: &[f64],
) -> Vec<Complex> {
    let (n, g) = (baselines.len(), elevations.len());
    let mut a = vec![Complex::ZERO; n * g];
    for (i, &b) in baselines.iter().enumerate() {
        let xi = 4.0 * core::f64::consts::PI * b / (wavelength * range);
        for (j, &s) in elevations.iter().enumerate() {
            a[i * g + j] = Complex::from_polar(1.0, -xi * s);
        }
    }
    a
}

/// Complex soft-thresholding `S_κ(z) = z·max(1 − κ/|z|, 0)` — the proximal operator of `κ|z|`.
fn soft_threshold(z: Complex, kappa: f64) -> Complex {
    let m = z.abs();
    if m <= kappa || m < 1e-300 {
        Complex::ZERO
    } else {
        z.scale(1.0 - kappa / m)
    }
}

/// `A·v` for a row-major `n×g` `A` and `g`-vector `v`.
fn mat_vec(a: &[Complex], v: &[Complex], n: usize, g: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; n];
    for i in 0..n {
        let mut s = Complex::ZERO;
        for j in 0..g {
            s = s + a[i * g + j] * v[j];
        }
        out[i] = s;
    }
    out
}

/// `Aᴴ·r` for a row-major `n×g` `A` and `n`-vector `r`.
fn mat_herm_vec(a: &[Complex], r: &[Complex], n: usize, g: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; g];
    for i in 0..n {
        for j in 0..g {
            out[j] = out[j] + a[i * g + j].conj() * r[i];
        }
    }
    out
}

/// The Lipschitz constant `‖A‖₂² = λ_max(AᴴA)` by power iteration — the largest usable FISTA step is
/// its reciprocal.
fn lipschitz(a: &[Complex], n: usize, g: usize, iters: usize) -> f64 {
    let mut v = vec![Complex::new(1.0, 0.0); g];
    let mut lambda = 1.0;
    for _ in 0..iters {
        let w = mat_herm_vec(a, &mat_vec(a, &v, n, g), n, g); // AᴴA v
        let norm = libm::sqrt(w.iter().map(|c| c.norm_sqr()).sum::<f64>());
        if norm < 1e-300 {
            break;
        }
        lambda = norm;
        let inv = 1.0 / norm;
        for (vi, wi) in v.iter_mut().zip(&w) {
            *vi = wi.scale(inv);
        }
    }
    lambda
}

/// Solve the complex LASSO `min_γ ½‖y − Aγ‖₂² + λ‖γ‖₁` (`A` row-major `n×g`) by FISTA — the accelerated
/// iterative soft-thresholding of Beck & Teboulle (2009). Returns the `g`-vector `γ`.
#[must_use]
pub fn fista_lasso(
    a: &[Complex],
    y: &[Complex],
    n: usize,
    g: usize,
    lambda: f64,
    max_iter: usize,
) -> Vec<Complex> {
    let step = 1.0 / (lipschitz(a, n, g, 40) * 1.01); // 1.01 keeps step ≤ 1/L for convergence
    let mut gamma = vec![Complex::ZERO; g];
    let mut z = gamma.clone();
    let mut t = 1.0f64;
    for _ in 0..max_iter {
        let az = mat_vec(a, &z, n, g);
        let resid: Vec<Complex> = az.iter().zip(y).map(|(p, q)| *p - *q).collect();
        let grad = mat_herm_vec(a, &resid, n, g);
        let g_new: Vec<Complex> = (0..g)
            .map(|j| soft_threshold(z[j] - grad[j].scale(step), lambda * step))
            .collect();
        let t_new = 0.5 * (1.0 + libm::sqrt(1.0 + 4.0 * t * t));
        let mom = (t - 1.0) / t_new;
        for j in 0..g {
            z[j] = g_new[j] + (g_new[j] - gamma[j]).scale(mom);
        }
        gamma = g_new;
        t = t_new;
    }
    gamma
}

/// The SL1MMER elevation spectrum: the magnitude of the sparse reflectivity `γ` recovered by [complex
/// LASSO](fista_lasso) from a single multi-baseline measurement `y` (the `N` co-registered stack
/// samples of one range-azimuth cell), over the `elevations` grid. `lambda` trades sparsity against fit
/// and `max_iter` bounds the FISTA loop.
#[must_use]
pub fn sl1mmer_spectrum(
    y: &[Complex],
    baselines: &[f64],
    wavelength: f64,
    range: f64,
    elevations: &[f64],
    lambda: f64,
    max_iter: usize,
) -> Vec<f64> {
    let (n, g) = (baselines.len(), elevations.len());
    let a = sensing_matrix(baselines, wavelength, range, elevations);
    let gamma = fista_lasso(&a, y, n, g, lambda, max_iter);
    gamma.iter().map(|c| c.abs()).collect()
}

/// SL1MMER tuning: the ℓ₁ penalty, the FISTA iteration cap, and the model-selection threshold.
#[derive(Clone, Copy, Debug)]
pub struct Sl1mmerConfig {
    /// ℓ₁ penalty `λ` (sparsity vs fit).
    pub lambda: f64,
    /// FISTA iteration cap.
    pub max_iter: usize,
    /// Support-detection threshold, relative to the peak magnitude.
    pub threshold_rel: f64,
}

/// A recovered scatterer: its grid elevation (m) and complex reflectivity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scatterer {
    /// Elevation on the grid (m).
    pub elevation: f64,
    /// Debiased complex reflectivity.
    pub reflectivity: Complex,
}

/// Detect the sparse support of a reflectivity vector: the grid indices whose magnitude exceeds
/// `threshold_rel · max|γ|` — the **model-selection** step (how many scatterers, and where).
#[must_use]
pub fn detect_support(gamma: &[Complex], threshold_rel: f64) -> Vec<usize> {
    let peak = gamma.iter().map(|c| c.abs()).fold(0.0, f64::max);
    if peak <= 0.0 {
        return Vec::new();
    }
    let thresh = threshold_rel * peak;
    (0..gamma.len())
        .filter(|&j| gamma[j].abs() > thresh)
        .collect()
}

/// Least-squares refit of the reflectivities on a fixed `support` — the **estimation-reconstruction**
/// step that removes the ℓ₁ amplitude bias. Solves `min ‖y − A_S x‖²` (normal equations
/// `A_Sᴴ A_S x = A_Sᴴ y`) over the selected columns of `A` (`n×g` row-major); returns a full `g`-vector,
/// zero off the support.
#[must_use]
pub fn refit_amplitudes(
    a: &[Complex],
    y: &[Complex],
    n: usize,
    g: usize,
    support: &[usize],
) -> Vec<Complex> {
    let k = support.len();
    let mut out = vec![Complex::ZERO; g];
    if k == 0 {
        return out;
    }
    // Normal equations on the k selected columns.
    let mut ata = vec![Complex::ZERO; k * k];
    let mut aty = vec![Complex::ZERO; k];
    for (p, &sp) in support.iter().enumerate() {
        for i in 0..n {
            aty[p] = aty[p] + a[i * g + sp].conj() * y[i];
        }
        for (q, &sq) in support.iter().enumerate() {
            let mut s = Complex::ZERO;
            for i in 0..n {
                s = s + a[i * g + sp].conj() * a[i * g + sq];
            }
            ata[p * k + q] = s;
        }
    }
    if let Some(x) = solve_complex(&ata, &aty, k) {
        for (p, &sp) in support.iter().enumerate() {
            out[sp] = x[p];
        }
    }
    out
}

/// The full **SL1MMER** (Zhu & Bamler): ℓ₁ scale-down ([`fista_lasso`]) → model selection
/// ([`detect_support`]) → estimation reconstruction ([`refit_amplitudes`]). Returns the detected
/// scatterers with **debiased** complex reflectivities (the ℓ₁ step biases magnitudes low; the LS refit
/// on the selected support restores them).
#[must_use]
pub fn sl1mmer(
    y: &[Complex],
    baselines: &[f64],
    wavelength: f64,
    range: f64,
    elevations: &[f64],
    cfg: &Sl1mmerConfig,
) -> Vec<Scatterer> {
    let (n, g) = (baselines.len(), elevations.len());
    let a = sensing_matrix(baselines, wavelength, range, elevations);
    let gamma = fista_lasso(&a, y, n, g, cfg.lambda, cfg.max_iter);
    let support = detect_support(&gamma, cfg.threshold_rel);
    let refit = refit_amplitudes(&a, y, n, g, &support);
    support
        .iter()
        .map(|&j| Scatterer {
            elevation: elevations[j],
            reflectivity: refit[j],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::unreadable_literal,
        clippy::expect_used
    )]
    use super::*;

    // The ideal covariance of equal-power uncorrelated scatterers: R = Σ_k a(s_k) a(s_k)ᴴ.
    fn ideal_covariance(
        baselines: &[f64],
        wavelength: f64,
        range: f64,
        scatterers: &[f64],
    ) -> Vec<Complex> {
        let n = baselines.len();
        let mut r = vec![Complex::ZERO; n * n];
        for &s in scatterers {
            let a = steering_vector(baselines, wavelength, range, s);
            for i in 0..n {
                for j in 0..n {
                    r[i * n + j] = r[i * n + j] + a[i] * a[j].conj();
                }
            }
        }
        r
    }

    fn argmax(v: &[f64]) -> usize {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }

    const LAMBDA: f64 = 0.056; // C-band
    const RANGE: f64 = 700e3;
    const BASELINES: [f64; 7] = [-100.0, -66.0, -33.0, 0.0, 33.0, 66.0, 100.0]; // 200 m aperture

    #[test]
    fn steering_vector_is_flat_at_zero_elevation() {
        let a = steering_vector(&BASELINES, LAMBDA, RANGE, 0.0);
        assert!(
            a.iter()
                .all(|c| (c.re - 1.0).abs() < 1e-15 && c.im.abs() < 1e-15)
        );
    }

    #[test]
    fn both_estimators_localize_well_separated_scatterers() {
        // Rayleigh elevation resolution ≈ λr/(2·aperture) ≈ 98 m; place scatterers 120 m apart.
        let (s1, s2) = (-60.0, 60.0);
        let r = ideal_covariance(&BASELINES, LAMBDA, RANGE, &[s1, s2]);
        let grid: Vec<f64> = (-200..=200).step_by(2).map(f64::from).collect();
        let bart = bartlett_spectrum(&r, &BASELINES, LAMBDA, RANGE, &grid);
        let cap = capon_spectrum(&r, &BASELINES, LAMBDA, RANGE, &grid, 1e-3);
        for spec in [&bart, &cap] {
            let peak = grid[argmax(spec)];
            assert!(
                (peak - s1).abs() < 25.0 || (peak - s2).abs() < 25.0,
                "peak at {peak}, not near a scatterer"
            );
        }
    }

    #[test]
    fn capon_super_resolves_a_sub_rayleigh_pair() {
        // A pair 60 m apart (< Rayleigh 98 m): Capon must be bimodal (a dip between the two), while
        // Bartlett stays unimodal (one broad lobe peaking at the centre).
        let (s1, s2) = (-30.0, 30.0);
        let r = ideal_covariance(&BASELINES, LAMBDA, RANGE, &[s1, s2]);
        let at = |f: &dyn Fn(&[f64]) -> Vec<f64>, s: f64| f(&[s])[0];
        let cap = |g: &[f64]| capon_spectrum(&r, &BASELINES, LAMBDA, RANGE, g, 1e-3);
        let bart = |g: &[f64]| bartlett_spectrum(&r, &BASELINES, LAMBDA, RANGE, g);
        // Capon: both scatterer elevations exceed the midpoint ⇒ two resolved peaks.
        assert!(
            at(&cap, s1) > at(&cap, 0.0) && at(&cap, s2) > at(&cap, 0.0),
            "Capon not resolving: {} {} {}",
            at(&cap, s1),
            at(&cap, 0.0),
            at(&cap, s2)
        );
        // Bartlett: the midpoint is at least as bright as the scatterers ⇒ unresolved.
        assert!(
            at(&bart, 0.0) >= at(&bart, s1).min(at(&bart, s2)),
            "Bartlett unexpectedly resolved"
        );
    }

    #[test]
    fn sample_covariance_matches_the_ideal_for_coherent_snapshots() {
        // A single scatterer imaged over L looks (same steering vector) gives R = a aᴴ.
        let a = steering_vector(&BASELINES, LAMBDA, RANGE, 40.0);
        let snaps: Vec<&[Complex]> = (0..8).map(|_| a.as_slice()).collect();
        let r = sample_covariance(&snaps, BASELINES.len());
        let ideal = ideal_covariance(&BASELINES, LAMBDA, RANGE, &[40.0]);
        for (x, y) in r.iter().zip(&ideal) {
            assert!((*x - *y).abs() < 1e-12);
        }
    }

    // The stack measurement y = Σ_k γ_k · a(s_k) of scatterers at given elevations/amplitudes.
    fn measure(scatterers: &[(f64, Complex)]) -> Vec<Complex> {
        let mut y = vec![Complex::ZERO; BASELINES.len()];
        for &(s, amp) in scatterers {
            let a = steering_vector(&BASELINES, LAMBDA, RANGE, s);
            for (yi, ai) in y.iter_mut().zip(&a) {
                *yi = *yi + *ai * amp;
            }
        }
        y
    }

    #[test]
    fn fista_matches_the_sklearn_lasso_oracle() {
        // A real over-determined problem (12×5), λ = 0.096 (= m·α, α = 0.008). The complex FISTA on
        // real data must converge to what scikit-learn's Lasso (fit_intercept=False) returns.
        let (m, p) = (12usize, 5usize);
        let x: Vec<Complex> = (0..m * p)
            .map(|k| Complex::new(((k / p * 5 + k % p * 11) % 13) as f64 * 0.1 - 0.6, 0.0))
            .collect();
        let y: Vec<Complex> = (0..m)
            .map(|i| Complex::new(((i * 7 + 3) % 9) as f64 * 0.2 - 0.8, 0.0))
            .collect();
        let gamma = fista_lasso(&x, &y, m, p, 0.096, 40000);
        let w = [
            -0.21333314725829314,
            -0.5945629116886713,
            0.6965613486663732,
            -0.594562911688831,
            -0.5279920732379186,
        ];
        for (g, wj) in gamma.iter().zip(w) {
            assert!(
                (g.re - wj).abs() < 2e-3 && g.im.abs() < 1e-6,
                "{g:?} vs sklearn {wj}"
            );
        }
    }

    #[test]
    fn sl1mmer_recovers_sparse_scatterers() {
        // Two scatterers on the grid (noise-free y = Aγ). The ℓ₁ inversion recovers the support exactly:
        // its two dominant peaks sit at the scatterer grid points, everything else near zero.
        let grid: Vec<f64> = (-200..=200).step_by(10).map(f64::from).collect();
        let (i1, i2) = (14usize, 24usize); // s = −60, +40
        assert_eq!((grid[i1], grid[i2]), (-60.0, 40.0));
        let y = measure(&[
            (grid[i1], Complex::new(1.0, 0.0)),
            (grid[i2], Complex::new(0.8, 0.0)),
        ]);
        let spec = sl1mmer_spectrum(&y, &BASELINES, LAMBDA, RANGE, &grid, 0.4, 5000);
        // The two brightest grid points are the true scatterers.
        let mut order: Vec<usize> = (0..grid.len()).collect();
        order.sort_by(|&a, &b| spec[b].total_cmp(&spec[a]));
        assert!(
            (order[0] == i1 && order[1] == i2) || (order[0] == i2 && order[1] == i1),
            "top peaks at {:?}, expected {i1},{i2}",
            &order[..2]
        );
        // Off-support energy is negligible next to the recovered scatterers.
        let peak = spec[i1].max(spec[i2]);
        for (k, &v) in spec.iter().enumerate() {
            if k != i1 && k != i2 {
                assert!(v < 0.2 * peak, "spurious peak {v} at grid {k}");
            }
        }
        assert!(spec[i1] > spec[i2]); // amplitude ordering (1.0 > 0.8) preserved
    }

    #[test]
    fn sl1mmer_super_resolves_beyond_beamforming() {
        // Two scatterers 60 m apart — below Rayleigh (~98 m). SL1MMER resolves them into two spikes
        // with a deep dip between; the matched-filter (beamforming) response cannot, staying unimodal.
        let grid: Vec<f64> = (-200..=200).step_by(5).map(f64::from).collect();
        let (i1, i2, imid) = (34usize, 46usize, 40usize); // s = −30, +30, midpoint 0
        assert_eq!((grid[i1], grid[i2], grid[imid]), (-30.0, 30.0, 0.0));
        let y = measure(&[
            (grid[i1], Complex::new(1.0, 0.0)),
            (grid[i2], Complex::new(1.0, 0.0)),
        ]);

        let spec = sl1mmer_spectrum(&y, &BASELINES, LAMBDA, RANGE, &grid, 0.3, 8000);
        // SL1MMER: both scatterer bins beat the midpoint (resolved) and off-support is quiet.
        assert!(
            spec[i1] > 3.0 * spec[imid] && spec[i2] > 3.0 * spec[imid],
            "SL1MMER not resolving: {} {} {}",
            spec[i1],
            spec[imid],
            spec[i2]
        );

        // Matched filter / beamforming |a(s)ᴴ y|: a single broad lobe — the midpoint is at least as
        // bright as either scatterer, so the pair is unresolved.
        let a = sensing_matrix(&BASELINES, LAMBDA, RANGE, &grid);
        let mf: Vec<f64> = (0..grid.len())
            .map(|g| {
                let mut s = Complex::ZERO;
                for i in 0..BASELINES.len() {
                    s = s + a[i * grid.len() + g].conj() * y[i];
                }
                s.abs()
            })
            .collect();
        assert!(
            mf[imid] >= mf[i1] && mf[imid] >= mf[i2],
            "beamforming unexpectedly resolved (mid {}, s1 {}, s2 {})",
            mf[imid],
            mf[i1],
            mf[i2]
        );
        // A quick sanity that the beamformer's own peak is broad (near the centre), not at a scatterer.
        assert!((argmax(&mf) as isize - imid as isize).abs() <= 2);
    }

    #[test]
    fn sl1mmer_selects_the_model_and_debiases_amplitudes() {
        // Two scatterers with known complex amplitudes on the grid (noise-free). The ℓ₁ step biases the
        // magnitudes low; model selection recovers the support and the LS refit restores the true
        // complex reflectivities exactly — the full three-step SL1MMER.
        let grid: Vec<f64> = (-200..=200).step_by(10).map(f64::from).collect();
        let (i1, i2) = (14usize, 24usize);
        let (amp1, amp2) = (Complex::new(1.0, 0.0), Complex::from_polar(0.8, 0.7));
        let y = measure(&[(grid[i1], amp1), (grid[i2], amp2)]);

        let cfg = Sl1mmerConfig {
            lambda: 0.4,
            max_iter: 6000,
            threshold_rel: 0.3,
        };
        let mut sc = sl1mmer(&y, &BASELINES, LAMBDA, RANGE, &grid, &cfg);
        assert_eq!(sc.len(), 2, "model selection found {} scatterers", sc.len());
        sc.sort_by(|a, b| a.elevation.total_cmp(&b.elevation));
        assert_eq!((sc[0].elevation, sc[1].elevation), (grid[i1], grid[i2]));
        // Debiased amplitudes match the truth (LS on the exact support is unbiased).
        assert!(
            (sc[0].reflectivity - amp1).abs() < 1e-9,
            "{:?}",
            sc[0].reflectivity
        );
        assert!(
            (sc[1].reflectivity - amp2).abs() < 1e-9,
            "{:?}",
            sc[1].reflectivity
        );

        // The raw ℓ₁ magnitudes were biased low — the bias the refit corrects.
        let a = sensing_matrix(&BASELINES, LAMBDA, RANGE, &grid);
        let gamma = fista_lasso(&a, &y, BASELINES.len(), grid.len(), 0.4, 6000);
        assert!(
            gamma[i1].abs() < amp1.abs() - 0.05,
            "L1 magnitude {} not biased below {}",
            gamma[i1].abs(),
            amp1.abs()
        );
    }

    /// The linear solver on its own: Cramer's rule, a forced pivot swap, and a singular system.
    ///
    /// Every call in this module hands `solve_complex` a diagonally-loaded Hermitian matrix whose
    /// leading entry is already the largest in its column, so the pivot search, the row swap and the
    /// singularity guard are all dead code under the existing tests — eleven mutants survived,
    /// including never swapping a row and never refusing a singular matrix. Cramer's rule is an
    /// independent algorithm for the same answer; the swaps and the refusal need fixtures built to
    /// reach them.
    #[test]
    fn the_complex_solver_pivots_where_it_must_and_refuses_a_singular_matrix() {
        fn det3(m: &[Complex]) -> Complex {
            m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6])
        }
        // Cramer: x_j = det(A with column j replaced by b) / det(A).
        fn cramer3(a: &[Complex], b: &[Complex]) -> Vec<Complex> {
            let det = det3(a);
            let inv_det = det.conj().scale(1.0 / det.norm_sqr());
            (0..3)
                .map(|j| {
                    let mut m = a.to_vec();
                    for i in 0..3 {
                        m[i * 3 + j] = b[i];
                    }
                    det3(&m) * inv_det
                })
                .collect()
        }
        let c = Complex::new;

        // A dense system with no zero entries and a pivot swap forced at the *second* column, so the
        // column-indexed reads inside the search run with `col > 0`.
        let a = [
            c(2.0, 1.0),
            c(1.0, 0.0),
            c(0.0, -1.0),
            c(1.0, 0.0),
            c(0.5, 0.0),
            c(3.0, 1.0),
            c(0.0, 2.0),
            c(4.0, -1.0),
            c(2.0, 0.0),
        ];
        let b = [c(1.0, 0.0), c(0.0, 1.0), c(2.0, -2.0)];
        let got = solve_complex(&a, &b, 3).expect("nonsingular");
        for (g, w) in got.iter().zip(&cramer3(&a, &b)) {
            assert!((*g - *w).abs() < 1e-12, "{g:?} vs Cramer {w:?}");
        }

        // A zero on the leading diagonal: without the swap the first pivot is 0 and there is no
        // answer at all. The exchange makes the system the identity, so the solution is `b` reversed.
        let swap = [c(0.0, 0.0), c(1.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)];
        let rhs = [c(3.0, 1.0), c(5.0, 0.0)];
        let got = solve_complex(&swap, &rhs, 2).expect("nonsingular after the swap");
        assert_eq!((got[0], got[1]), (rhs[1], rhs[0]));

        // Rank deficient by construction: row 1 is twice row 0, so the second pivot is exactly zero.
        assert!(
            solve_complex(
                &[c(1.0, 0.0), c(2.0, 0.0), c(2.0, 0.0), c(4.0, 0.0)],
                &rhs,
                2
            )
            .is_none()
        );
        // ...and a 1x1 zero has nothing to pivot onto.
        assert!(solve_complex(&[Complex::ZERO], &rhs[..1], 1).is_none());
    }

    /// The Bartlett spectrum's height, not only where its peak is.
    ///
    /// `both_estimators_localize_well_separated_scatterers` compares the grid to itself through
    /// `argmax`, so the `aᴴa` normalisation is invisible: for the unit-modulus steering vectors this
    /// module builds, `aᴴa` is the baseline count, and multiplying by it instead of dividing scales
    /// every value by `N²` without moving the peak one bin. For a single unit scatterer at `s₀`,
    /// `R = a₀a₀ᴴ` and `P(s) = |aᴴa₀|²/N`, which is `N` on the scatterer and zero at any `s` whose
    /// steering vector is orthogonal to it.
    #[test]
    fn the_bartlett_spectrum_peaks_at_the_baseline_count() {
        let n = BASELINES.len() as f64;
        let r = ideal_covariance(&BASELINES, LAMBDA, RANGE, &[40.0]);
        let p = bartlett_spectrum(&r, &BASELINES, LAMBDA, RANGE, &[40.0]);
        assert!((p[0] - n).abs() < 1e-9, "P(s₀) = {} vs N = {n}", p[0]);
        // Twice the power is twice the spectrum: linear in R, not quadratic and not affine.
        let r2: Vec<Complex> = r.iter().map(|z| z.scale(2.0)).collect();
        let p2 = bartlett_spectrum(&r2, &BASELINES, LAMBDA, RANGE, &[40.0]);
        assert!((p2[0] - 2.0 * n).abs() < 1e-9, "P₂(s₀) = {}", p2[0]);
    }

    /// Capon's diagonal loading, against the Sherman–Morrison closed form.
    ///
    /// The loading is a small stabiliser in every existing test, so writing it to the wrong index or
    /// subtracting it changes nothing they can see. Make it the whole story instead: for one unit
    /// scatterer, `R = a₀a₀ᴴ` is rank one and `R + δI` inverts in closed form, giving
    /// `P(s₀) = 1/(a₀ᴴ(R+δI)⁻¹a₀) = (δ + N)/N` — a value that moves with δ, so a loading that lands
    /// off the diagonal or arrives with the wrong sign is a different number.
    #[test]
    fn capon_loading_lands_on_the_diagonal_with_the_sign_it_should() {
        let n = BASELINES.len() as f64;
        let r = ideal_covariance(&BASELINES, LAMBDA, RANGE, &[40.0]);
        for delta in [0.25, 2.0, 20.0] {
            let p = capon_spectrum(&r, &BASELINES, LAMBDA, RANGE, &[40.0], delta);
            let want = (delta + n) / n;
            assert!(
                (p[0] - want).abs() < 1e-9 * want,
                "δ={delta}: P(s₀) = {} vs (δ+N)/N = {want}",
                p[0]
            );
        }
    }

    /// Soft-thresholding shrinks the magnitude by `κ` and leaves the phase alone.
    ///
    /// Nothing tested the proximal operator directly. State the closed form —
    /// `|S_κ(z)| = max(|z| − κ, 0)`, `arg S_κ(z) = arg z` — which is the operator's definition rather
    /// than the `1 − κ/|z|` form the code computes it with. The `|z| < 1e-300` half of the guard is
    /// unreachable and cannot be stated: `abs` is `sqrt(re² + im²)`, so the square underflows first
    /// and the smallest non-zero magnitude any `Complex` can report is about 2.2e-162.
    #[test]
    fn soft_thresholding_shrinks_the_magnitude_and_keeps_the_phase() {
        for &(re, im) in &[(3.0, 4.0), (-1.0, 0.5), (0.0, -2.0)] {
            let z = Complex::new(re, im);
            for kappa in [0.0, 0.5, 1.0, z.abs() * 0.999] {
                let s = soft_threshold(z, kappa);
                assert!(
                    (s.abs() - (z.abs() - kappa)).abs() < 1e-12,
                    "{z:?} κ={kappa}"
                );
                assert!((s.arg() - z.arg()).abs() < 1e-12, "{z:?} κ={kappa} phase");
            }
            // At and beyond the threshold the operator collapses to zero, inclusively.
            assert_eq!(soft_threshold(z, z.abs()), Complex::ZERO);
            assert_eq!(soft_threshold(z, z.abs() * 2.0), Complex::ZERO);
        }
        assert_eq!(soft_threshold(Complex::ZERO, 0.0), Complex::ZERO);
    }

    /// FISTA returns a LASSO minimiser, checked against the optimality conditions.
    ///
    /// The existing SL1MMER tests ask where the peaks are and that the ℓ₁ magnitudes are biased low;
    /// both hold for any iterate close enough to the answer, so the step size and the momentum
    /// schedule are free. The subgradient conditions of `min ½‖y − Aγ‖² + λ‖γ‖₁` are not: writing
    /// `c = Aᴴ(y − Aγ)`, a minimiser has `|cⱼ| ≤ λ` everywhere and `cⱼ = λ·γⱼ/|γⱼ|` on the support.
    #[test]
    fn fista_returns_a_point_that_satisfies_the_lasso_optimality_conditions() {
        let grid: Vec<f64> = (-5..=6).map(|k| f64::from(k) * 40.0).collect();
        let (n, g) = (BASELINES.len(), grid.len());
        let a = sensing_matrix(&BASELINES, LAMBDA, RANGE, &grid);
        let truth = Complex::new(2.0, -1.0);
        let y: Vec<Complex> = (0..n).map(|i| a[i * g + 4] * truth).collect();
        let lambda = 0.5;

        let gamma = fista_lasso(&a, &y, n, g, lambda, 20_000);
        let resid: Vec<Complex> = (0..n)
            .map(|i| {
                y[i] - (0..g)
                    .map(|j| a[i * g + j] * gamma[j])
                    .fold(Complex::ZERO, |s, z| s + z)
            })
            .collect();
        let c = mat_herm_vec(&a, &resid, n, g);
        for j in 0..g {
            assert!(
                c[j].abs() <= lambda * (1.0 + 1e-6),
                "|c[{j}]| = {} exceeds λ = {lambda}",
                c[j].abs()
            );
            if gamma[j].abs() > 1e-6 {
                let want = gamma[j].scale(lambda / gamma[j].abs());
                assert!(
                    (c[j] - want).abs() < 1e-6,
                    "support {j}: {:?} vs {want:?}",
                    c[j]
                );
            }
        }
        // Non-vacuous: the support is not empty, so the second condition is actually exercised.
        assert!(
            gamma.iter().any(|z| z.abs() > 1e-6),
            "everything thresholded away"
        );
    }

    /// FISTA is *accelerated*, and how fast it arrives is the only thing its step and its momentum
    /// schedule decide.
    ///
    /// Both converge to the same minimiser for any step in `(0, 2/L)` and any momentum tending to
    /// one, so an optimality check at 20 000 iterations cannot see either — six mutants of those
    /// three lines survived it. Measure the objective gap after thirty instead, against a bound with
    /// two orders of headroom over what the correct schedule reaches. Dropping the `1` from the
    /// `t`-recurrence costs a factor of 1000, and a momentum of `(t + 1)/t'` instead of `(t − 1)/t'`
    /// a factor of 17 000. The step needs a badly scaled `A` to matter: with `‖A‖₂² = 0.0084`,
    /// adding 1.01 to the Lipschitz constant rather than multiplying by it shrinks the step 120-fold
    /// and the gap never falls below 2e-5.
    #[test]
    fn fista_converges_at_the_rate_its_momentum_schedule_promises() {
        let grid: Vec<f64> = (-5..=6).map(|k| f64::from(k) * 40.0).collect();
        let (n, g) = (BASELINES.len(), grid.len());
        let objective = |a: &[Complex], y: &[Complex], gamma: &[Complex], lambda: f64| {
            let fit: f64 = (0..n)
                .map(|i| {
                    ((0..g)
                        .map(|j| a[i * g + j] * gamma[j])
                        .fold(Complex::ZERO, |s, z| s + z)
                        - y[i])
                        .norm_sqr()
                })
                .sum();
            0.5 * fit + lambda * gamma.iter().map(|z| z.abs()).sum::<f64>()
        };
        let truth = Complex::new(2.0, -1.0);

        for (scale, bound) in [(1.0, 1e-3), (0.01, 1e-6)] {
            let a: Vec<Complex> = sensing_matrix(&BASELINES, LAMBDA, RANGE, &grid)
                .iter()
                .map(|z| z.scale(scale))
                .collect();
            let y: Vec<Complex> = (0..n).map(|i| a[i * g + 4] * truth).collect();
            let lambda = 0.5 * scale * scale;
            let best = objective(&a, &y, &fista_lasso(&a, &y, n, g, lambda, 20_000), lambda);
            let early = objective(&a, &y, &fista_lasso(&a, &y, n, g, lambda, 30), lambda);
            assert!(
                early - best < bound,
                "scale {scale}: gap after 30 iterations is {} (bound {bound})",
                early - best
            );
        }
    }

    /// A sensing matrix with no energy in it has no Lipschitz constant to divide by.
    ///
    /// The power iteration's break is unreachable from any real stack, so replacing its `<` by `==`
    /// costs nothing the SL1MMER tests can see — until `‖A‖₂` really is zero, when the reciprocal of
    /// the norm turns the whole iterate into `NaN` and every downstream magnitude with it.
    #[test]
    fn a_sensing_matrix_with_no_energy_still_returns_numbers() {
        let (n, g) = (4usize, 6usize);
        let y = vec![Complex::new(1.0, 1.0); n];
        let gamma = fista_lasso(&vec![Complex::ZERO; n * g], &y, n, g, 0.1, 50);
        assert_eq!(gamma.len(), g);
        assert!(
            gamma.iter().all(|z| z.re == 0.0 && z.im == 0.0),
            "{gamma:?}"
        );
    }

    /// The detected support is what *strictly exceeds* a threshold *proportional* to the peak.
    ///
    /// `sl1mmer_recovers_two_scatterers` reads the support only through the scatterers it produces,
    /// with magnitudes far from the cut, so neither the strictness of the comparison nor whether the
    /// threshold scales with the peak is observable. A geometric ladder with an entry landing exactly
    /// on the cut states both.
    #[test]
    fn the_detected_support_is_what_strictly_exceeds_the_relative_threshold() {
        let gamma: Vec<Complex> = [4.0, 2.0, 1.0, 0.0]
            .iter()
            .map(|&m| Complex::from_polar(m, 0.3))
            .collect();
        // Peak 4, cut 0.5 * 4 = 2: index 1 sits exactly on it and must be excluded.
        assert_eq!(detect_support(&gamma, 0.5), vec![0]);
        assert_eq!(detect_support(&gamma, 0.2), vec![0, 1, 2]);
        assert_eq!(detect_support(&gamma, 1.0), Vec::<usize>::new());
        // Nothing to be relative to.
        assert_eq!(
            detect_support(&[Complex::ZERO; 3], 0.5),
            Vec::<usize>::new()
        );
    }

    /// The sample covariance and the matrix-vector product, written out entry by entry.
    ///
    /// `sample_covariance_matches_the_ideal_for_coherent_snapshots` uses one repeated snapshot, so
    /// the `1/L` average is the same number `L` times and the outer product is symmetric under the
    /// index confusion a wrong stride would cause. Two different snapshots of length two state both.
    #[test]
    fn the_sample_covariance_averages_the_outer_products_it_is_given() {
        let c = Complex::new;
        let s0 = [c(1.0, 2.0), c(3.0, -1.0)];
        let s1 = [c(0.0, 1.0), c(-2.0, 0.5)];
        let r = sample_covariance(&[&s0[..], &s1[..]], 2);
        for i in 0..2 {
            for j in 0..2 {
                let want = (s0[i] * s0[j].conj() + s1[i] * s1[j].conj()).scale(0.5);
                assert!((r[i * 2 + j] - want).abs() < 1e-15, "R[{i}][{j}]");
            }
        }
        // R v, against the sum it is defined as -- a wrong stride or a sum of products both move it.
        let v = [c(2.0, 0.0), c(0.0, -3.0)];
        let rv = matvec(&r, &v, 2);
        for i in 0..2 {
            let want = r[i * 2] * v[0] + r[i * 2 + 1] * v[1];
            assert!((rv[i] - want).abs() < 1e-15, "(Rv)[{i}]");
        }
    }
}
