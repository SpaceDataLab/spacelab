//! Synthetic-aperture-radar focusing — pulse compression of the linear-FM (chirp) signal in range and
//! azimuth, the core of the Range-Doppler algorithm.
//!
//! A SAR transmits a long linear-FM chirp `exp(jπK t²)` and matched-filters the echo to a short pulse,
//! trading a long pulse (for energy) for fine resolution. [`matched_filter`] does the compression by
//! FFT correlation `IFFT(FFT(s)·conj(FFT(replica)))`; the compressed impulse response is a sinc of
//! −3 dB width `≈ 0.886/B` (bandwidth `B = |K|·T`) and peak sidelobe `≈ −13.26 dB` (rectangular
//! window). The same operation, with the transmitted chirp as replica, compresses **range**
//! ([`range_compress`]); with the target's Doppler-history chirp as replica it compresses **azimuth**
//! ([`azimuth_compress`]); doing both focuses a scene ([`focus_2d`]).
//!
//! [`focus_2d`] omits range-cell migration correction — valid only for a scene-centred target and a
//! short aperture, where the migration is sub-resolution. [`range_doppler_focus`] is the full
//! Range-Doppler algorithm **with RCMC**: azimuth FFT, a range-Doppler-domain range shift that
//! straightens the target's range-cell migration `ΔR(f_η) = λ²·r₀·f_η²/(8V²)` by truncated-sinc
//! interpolation, azimuth matched filtering, and azimuth IFFT — which focuses a target whose energy
//! migrates across several range cells over the aperture. [`secondary_range_compression`] adds the SRC
//! step (a range-Doppler-domain range filter `exp(−jπ f_τ²/K_src(f_η))`) that removes the
//! range-azimuth coupling, keeping the range response sharp at squint or wide bandwidth.
//! [`apply_chirp_scaling`] is the interpolation-free differential-RCMC primitive of the Chirp-Scaling
//! Algorithm: a phase multiply that re-scales a range chirp and shifts its compressed position without
//! resampling. `alloc` for the FFT buffers.

// n·dt casts and SAR/LFM/PSLR/FFT doc terms are intentional; short physical names mirror the literature.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use crate::fft::{Complex, fft, ifft, zero_pad};
use alloc::vec;
use alloc::vec::Vec;

/// A linear-FM (chirp) replica: `n` samples of `exp(jπ·chirp_rate·t²)`, `t` centred and sampled at
/// `sample_rate`. Its swept bandwidth is `|chirp_rate|·n/sample_rate`.
#[must_use]
pub fn lfm_chirp(n: usize, chirp_rate: f64, sample_rate: f64) -> Vec<Complex> {
    let dt = 1.0 / sample_rate;
    (0..n)
        .map(|i| {
            let t = (i as f64 - (n as f64 - 1.0) / 2.0) * dt;
            Complex::from_polar(1.0, core::f64::consts::PI * chirp_rate * t * t)
        })
        .collect()
}

/// The linear-correlation length: `signal + replica`, rounded up to a power of two for the FFT.
fn corr_len(signal: usize, replica: usize) -> usize {
    (signal + replica).next_power_of_two()
}

/// Matched-filter (pulse-compress) a `signal` with a `replica` by FFT correlation
/// `IFFT(FFT(s)·conj(FFT(r)))`. The output length is `corr_len`; the target appears as the
/// compressed impulse response at the correlation lag.
#[must_use]
pub fn matched_filter(signal: &[Complex], replica: &[Complex]) -> Vec<Complex> {
    let n = corr_len(signal.len(), replica.len());
    let mut s = zero_pad(signal, n);
    let mut r = zero_pad(replica, n);
    fft(&mut s);
    fft(&mut r);
    for i in 0..n {
        s[i] = s[i] * r[i].conj();
    }
    ifft(&mut s);
    s
}

/// Range-compress a 2-D echo (`rows` azimuth lines × `cols` range samples, row-major) with the range
/// `replica`, line by line. Returns the compressed data and its new column count.
#[must_use]
pub fn range_compress(
    echo: &[Complex],
    rows: usize,
    cols: usize,
    replica: &[Complex],
) -> (Vec<Complex>, usize) {
    let out_cols = corr_len(cols, replica.len());
    let mut out = vec![Complex::ZERO; rows * out_cols];
    for r in 0..rows {
        let comp = matched_filter(&echo[r * cols..r * cols + cols], replica);
        out[r * out_cols..r * out_cols + out_cols].copy_from_slice(&comp);
    }
    (out, out_cols)
}

/// Azimuth-compress a 2-D array (`rows` × `cols`, row-major) with the azimuth `replica`, column by
/// column. Returns the compressed data and its new row count.
#[must_use]
pub fn azimuth_compress(
    data: &[Complex],
    rows: usize,
    cols: usize,
    replica: &[Complex],
) -> (Vec<Complex>, usize) {
    let out_rows = corr_len(rows, replica.len());
    let mut out = vec![Complex::ZERO; out_rows * cols];
    let mut col = vec![Complex::ZERO; rows];
    for c in 0..cols {
        for r in 0..rows {
            col[r] = data[r * cols + c];
        }
        let comp = matched_filter(&col, replica);
        for (r, &v) in comp.iter().enumerate() {
            out[r * cols + c] = v;
        }
    }
    (out, out_rows)
}

/// Focus a 2-D SAR echo: range compression followed by azimuth compression. Returns the focused image
/// and its `(rows, cols)`.
#[must_use]
pub fn focus_2d(
    echo: &[Complex],
    rows: usize,
    cols: usize,
    range_replica: &[Complex],
    azimuth_replica: &[Complex],
) -> (Vec<Complex>, usize, usize) {
    let (rc, cols2) = range_compress(echo, rows, cols, range_replica);
    let (ac, rows2) = azimuth_compress(&rc, rows, cols2, azimuth_replica);
    (ac, rows2, cols2)
}

/// Index and value of the maximum of a slice, or `None` if it is empty.
fn argmax(mag: &[f64]) -> Option<(usize, f64)> {
    let mut best = (0, *mag.first()?);
    for (i, &v) in mag.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    Some(best)
}

/// The −3 dB (half-power) main-lobe width, in samples, of a magnitude impulse response, by linear
/// interpolation at the half-power crossings either side of the peak.
///
/// Returns `NaN` when the main lobe is not bracketed by the samples — a walk that reaches an end of
/// the array still above half power has no crossing on that side, and interpolating with the only
/// neighbour it has left reads the slope backwards (a peak in `mag[0]` came back as a *negative*
/// width). Supply an impulse response whose main lobe is inside the window.
///
/// Returns `NaN` too when the input is empty or its peak does not sit strictly above its own half
/// power. Both walks below start on the peak and each is what puts `l + 1` and `r - 1` in range: the
/// first is only read after the left walk has stepped off the peak, the second after the right walk
/// has. `peak > half` is what makes them step. It fails for an empty slice, for a peak that is NaN,
/// infinite, zero or negative, and for the smallest subnormals, where `peak / √2` rounds back to
/// `peak` — and this is a public entry point reached from Python with an arbitrary `Vec<f64>`, so
/// none of those were unreachable.
#[must_use]
pub fn resolution_3db_samples(mag: &[f64]) -> f64 {
    let Some((pk, peak)) = argmax(mag) else {
        return f64::NAN;
    };
    let half = peak / core::f64::consts::SQRT_2; // −3 dB in power ⇒ 1/√2 in amplitude
    if peak.is_nan() || peak <= half {
        return f64::NAN;
    }
    let mut l = pk;
    while l > 0 && mag[l] > half {
        l -= 1;
    }
    if mag[l] > half {
        return f64::NAN;
    }
    let dl = mag[l + 1] - mag[l];
    let left = if dl.abs() > 1e-300 {
        l as f64 + (half - mag[l]) / dl
    } else {
        l as f64
    };
    let mut r = pk;
    while r + 1 < mag.len() && mag[r] > half {
        r += 1;
    }
    if mag[r] > half {
        return f64::NAN;
    }
    let dr = mag[r - 1] - mag[r];
    let right = if dr.abs() > 1e-300 {
        (r - 1) as f64 + (mag[r - 1] - half) / dr
    } else {
        r as f64
    };
    right - left
}

/// The peak-sidelobe ratio, in decibels, of a magnitude impulse response: the highest sidelobe outside
/// the main lobe (bounded by the first nulls either side of the peak) relative to the peak.
///
/// Returns `NaN` for an empty input, which has no peak to be relative to.
#[must_use]
pub fn peak_sidelobe_ratio_db(mag: &[f64]) -> f64 {
    let Some((pk, peak)) = argmax(mag) else {
        return f64::NAN;
    };
    let mut l = pk;
    while l > 0 && mag[l - 1] < mag[l] {
        l -= 1;
    }
    let mut r = pk;
    while r + 1 < mag.len() && mag[r + 1] < mag[r] {
        r += 1;
    }
    let mut side = 0.0f64;
    for &v in &mag[..l] {
        side = side.max(v);
    }
    for &v in &mag[r + 1..] {
        side = side.max(v);
    }
    20.0 * libm::log10(side / peak)
}

// ---------------------------------------------------------------------------------------------------
// Range-Doppler algorithm with Range Cell Migration Correction (RCMC).
// ---------------------------------------------------------------------------------------------------

/// The geometry of a broadside stripmap acquisition, for the Range-Doppler algorithm.
#[derive(Clone, Copy, Debug)]
pub struct RdaParams {
    /// Pulse repetition frequency (azimuth sample rate, Hz).
    pub prf: f64,
    /// Platform (beam-footprint) velocity (m/s).
    pub velocity: f64,
    /// Radar wavelength (m).
    pub wavelength: f64,
    /// Closest-approach slant range of the reference target (m).
    pub r0_slant: f64,
    /// Range-bin spacing (m/sample).
    pub range_spacing: f64,
}

/// The azimuth FM rate `K_a = 2V²/(λ·r₀)` (Hz/s) — the chirp rate of a point target's Doppler history.
#[must_use]
pub fn azimuth_fm_rate(params: &RdaParams) -> f64 {
    2.0 * params.velocity * params.velocity / (params.wavelength * params.r0_slant)
}

/// The range-cell migration `ΔR(f_η) = λ²·r₀·f_η²/(8V²)` (m) at azimuth frequency `f_eta` (Hz).
#[must_use]
pub fn range_migration(f_eta: f64, params: &RdaParams) -> f64 {
    let RdaParams {
        wavelength: l,
        r0_slant: r0,
        velocity: v,
        ..
    } = *params;
    l * l * r0 * f_eta * f_eta / (8.0 * v * v)
}

/// The azimuth (Doppler) frequency in Hz of FFT bin `k` of an `n`-point azimuth transform at `prf`,
/// mapped to `[−prf/2, prf/2)` (the usual FFT wrap-around).
#[must_use]
pub fn azimuth_frequency(k: usize, n: usize, prf: f64) -> f64 {
    let kk = if 2 * k < n {
        k as f64
    } else {
        k as f64 - n as f64
    };
    kk * prf / n as f64
}

/// Normalised sinc `sin(πx)/(πx)`.
#[inline]
fn sinc(x: f64) -> f64 {
    if libm::fabs(x) < 1e-12 {
        1.0
    } else {
        let px = core::f64::consts::PI * x;
        libm::sin(px) / px
    }
}

/// Shift a complex range line by `delta` samples (positive = towards larger index) by truncated-sinc
/// interpolation with a `2·half_kernel`-tap window: `out[r] = Σ_j in[⌊r−δ⌋+j]·sinc(r−δ−(⌊r−δ⌋+j))`.
#[must_use]
pub fn sinc_shift(line: &[Complex], delta: f64, half_kernel: usize) -> Vec<Complex> {
    let n = line.len();
    let hk = half_kernel as isize;
    let mut out = vec![Complex::ZERO; n];
    for r in 0..n {
        let src = r as f64 - delta;
        let base = libm::floor(src) as isize;
        let mut acc = Complex::ZERO;
        for j in (1 - hk)..=hk {
            let idx = base + j;
            if idx >= 0 && (idx as usize) < n {
                acc = acc + line[idx as usize].scale(sinc(src - idx as f64));
            }
        }
        out[r] = acc;
    }
    out
}

/// Focus a range-compressed SAR echo (`na` azimuth lines × `nr` range bins, row-major) by the
/// Range-Doppler algorithm: azimuth FFT, RCMC (when `apply_rcmc`), azimuth matched filtering, azimuth
/// IFFT. `na` must be a power of two. Returns the focused image (`na × nr`).
///
/// Passing `apply_rcmc = false` skips only the migration correction — the control against which RCMC's
/// effect is measured on a target whose energy migrates across range cells.
#[must_use]
pub fn range_doppler_focus(
    range_compressed: &[Complex],
    na: usize,
    nr: usize,
    params: &RdaParams,
    apply_rcmc: bool,
) -> Vec<Complex> {
    let ka = azimuth_fm_rate(params);
    // Azimuth matched filter: the conjugate spectrum of the azimuth chirp replica exp(−jπ·K_a·η²),
    // η = (a − na/2)/prf — exact matched filtering carried out in the azimuth-frequency domain.
    let mut replica: Vec<Complex> = (0..na)
        .map(|a| {
            let eta = (a as f64 - na as f64 / 2.0) / params.prf;
            Complex::from_polar(1.0, -core::f64::consts::PI * ka * eta * eta)
        })
        .collect();
    fft(&mut replica);
    let mf: Vec<Complex> = replica.iter().map(|c| c.conj()).collect();

    // Transpose into azimuth columns, FFT each, giving the range-Doppler domain rd[k*nr + r].
    let mut rd = vec![Complex::ZERO; na * nr];
    let mut col = vec![Complex::ZERO; na];
    for r in 0..nr {
        for a in 0..na {
            col[a] = range_compressed[a * nr + r];
        }
        fft(&mut col);
        for k in 0..na {
            rd[k * nr + r] = col[k];
        }
    }

    // RCMC: straighten each azimuth-frequency row by −ΔR(f_η) in range bins.
    if apply_rcmc {
        let mut row = vec![Complex::ZERO; nr];
        for k in 0..na {
            let f_eta = azimuth_frequency(k, na, params.prf);
            let shift_bins = range_migration(f_eta, params) / params.range_spacing;
            row.copy_from_slice(&rd[k * nr..k * nr + nr]);
            let shifted = sinc_shift(&row, -shift_bins, 4);
            rd[k * nr..k * nr + nr].copy_from_slice(&shifted);
        }
    }

    // Azimuth matched filter (multiply each range column by the MF spectrum) then azimuth IFFT.
    for k in 0..na {
        for r in 0..nr {
            rd[k * nr + r] = rd[k * nr + r] * mf[k];
        }
    }
    let mut out = vec![Complex::ZERO; na * nr];
    for r in 0..nr {
        for k in 0..na {
            col[k] = rd[k * nr + r];
        }
        ifft(&mut col);
        for a in 0..na {
            out[a * nr + r] = col[a];
        }
    }
    out
}

/// Secondary-range-compression FM rate `K_src(f_η) = 2V²(f₀D₀)³/(R₀·c·f_η²)` (Hz/s), with carrier
/// `f₀ = c/λ` and the range-migration factor `D₀ = √(1 − (λ f_η/2V)²)`. Derived by the principle of
/// stationary phase: it is the residual range-frequency chirp the range-azimuth coupling leaves after
/// range compression, which SRC must remove. Returns `None` at `f_η ≈ 0` (no coupling — SRC vanishes).
#[must_use]
pub fn src_fm_rate(f_eta: f64, params: &RdaParams) -> Option<f64> {
    if libm::fabs(f_eta) < 1e-9 {
        return None;
    }
    let c = 299_792_458.0;
    let f0 = c / params.wavelength;
    let migration = params.wavelength * f_eta / (2.0 * params.velocity);
    let d0 = libm::sqrt((1.0 - migration * migration).max(0.0));
    let fd = f0 * d0;
    Some(
        2.0 * params.velocity * params.velocity * fd * fd * fd
            / (params.r0_slant * c * f_eta * f_eta),
    )
}

/// Apply **secondary range compression** to range-Doppler-domain data (`na` azimuth-frequency rows ×
/// `nr` range bins, row-major): for each row, a range FFT, the SRC filter `exp(−jπ f_τ²/K_src(f_η))`
/// (which cancels the range-azimuth coupling chirp), and a range IFFT. `nr` must be a power of two.
/// This is the RDA step between the azimuth FFT and RCMC that keeps the range response sharp at squint
/// or wide bandwidth, where the coupling is not negligible.
#[must_use]
pub fn secondary_range_compression(
    rd_data: &[Complex],
    na: usize,
    nr: usize,
    params: &RdaParams,
) -> Vec<Complex> {
    let fs_r = 299_792_458.0 / (2.0 * params.range_spacing); // range-frequency sample rate (Hz)
    let mut out = rd_data.to_vec();
    let mut line = vec![Complex::ZERO; nr];
    for k in 0..na {
        let Some(k_src) = src_fm_rate(azimuth_frequency(k, na, params.prf), params) else {
            continue;
        };
        line.copy_from_slice(&rd_data[k * nr..k * nr + nr]);
        fft(&mut line);
        for m in 0..nr {
            let f_tau = azimuth_frequency(m, nr, fs_r);
            line[m] =
                line[m] * Complex::from_polar(1.0, -core::f64::consts::PI * f_tau * f_tau / k_src);
        }
        ifft(&mut line);
        out[k * nr..k * nr + nr].copy_from_slice(&line);
    }
    out
}

/// The **chirp-scaling** phase multiply — the interpolation-free differential range-cell-migration
/// correction at the heart of the Chirp-Scaling Algorithm. Multiplying a range chirp of FM rate
/// `chirp_rate` by `exp(jπ·chirp_rate·scaling·(t − t_ref)²)` (`t` centred on sample `tau_ref` at
/// `sample_rate`) re-scales it to rate `chirp_rate·(1+scaling)` and shifts its compressed position from
/// sample `c` to `(c + scaling·tau_ref)/(1 + scaling)` — a controlled shift by a phase multiply alone,
/// no resampling. CSA applies this per azimuth-frequency row to equalise every range's migration to a
/// reference before the bulk RCMC.
#[must_use]
pub fn apply_chirp_scaling(
    line: &[Complex],
    chirp_rate: f64,
    scaling: f64,
    tau_ref: f64,
    sample_rate: f64,
) -> Vec<Complex> {
    let dt = 1.0 / sample_rate;
    line.iter()
        .enumerate()
        .map(|(m, &v)| {
            let t = (m as f64 - tau_ref) * dt;
            v * Complex::from_polar(1.0, core::f64::consts::PI * chirp_rate * scaling * t * t)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::similar_names,
        clippy::unwrap_used,
        clippy::expect_used
    )]
    use super::*;

    const C: f64 = 299_792_458.0;

    fn magnitude(z: &[Complex]) -> Vec<f64> {
        z.iter().map(|c| c.abs()).collect()
    }

    #[test]
    fn range_compression_matches_impulse_response_theory() {
        // A long chirp: bandwidth B = |K|·T, oversampled 4×. Matched-filtering a delayed copy compresses
        // it to a sinc whose −3 dB width is ~0.886·fs/B samples and whose sidelobes sit at ~−13.26 dB.
        let (fs, bw, tp) = (200e6, 50e6, 5e-6);
        let n = (tp * fs) as usize; // 1000
        let k = bw / tp;
        let chirp = lfm_chirp(n, k, fs);
        let mut signal = vec![Complex::ZERO; 4096];
        signal[1000..1000 + n].copy_from_slice(&chirp);

        let mag = magnitude(&matched_filter(&signal, &chirp));
        let width = resolution_3db_samples(&mag);
        let pslr = peak_sidelobe_ratio_db(&mag);
        let theory = 0.886 * fs / bw; // 3.544 samples
        assert!(
            (width - theory).abs() < 1.0,
            "range width {width} vs theory {theory}"
        );
        assert!((-14.5..-12.0).contains(&pslr), "range PSLR {pslr} dB");
        // Ground-range resolution ≈ 0.886·c/(2B) ≈ 2.66 m.
        let range_res = width * C / (2.0 * fs);
        assert!(
            (range_res - 0.886 * C / (2.0 * bw)).abs() < 1.0,
            "range res {range_res} m"
        );
    }

    #[test]
    fn azimuth_compression_focuses_the_doppler_chirp() {
        // The azimuth signal of a point target is itself a linear-FM chirp (its Doppler history);
        // matched-filtering it is the same pulse compression, so it must show the same sidelobe floor.
        let (prf, ka, na) = (1000.0, 400.0, 512usize);
        let chirp = lfm_chirp(na, ka, prf);
        let mut signal = vec![Complex::ZERO; 2048];
        signal[400..400 + na].copy_from_slice(&chirp);
        let mag = magnitude(&matched_filter(&signal, &chirp));
        assert!(
            (-14.5..-12.0).contains(&peak_sidelobe_ratio_db(&mag)),
            "azimuth PSLR"
        );
        // The compressed pulse is far narrower than the uncompressed chirp (na samples).
        assert!(resolution_3db_samples(&mag) < 0.05 * na as f64);
    }

    #[test]
    fn focus_2d_collapses_a_point_target_to_one_peak() {
        // A separable 2-D point-target echo (range chirp × azimuth chirp) placed in the scene; focusing
        // must collapse it to a single dominant, sharp peak.
        let (rows, cols) = (64usize, 128usize);
        let (rg, az) = (48usize, 32usize);
        let range_chirp = lfm_chirp(rg, 30e6 / (rg as f64 / 100e6), 100e6);
        // Azimuth chirp with real bandwidth: B = Ka·(az/prf) ≈ 288 Hz at prf 1000 (≈3.5× oversampled),
        // so it actually compresses (a near-zero-bandwidth chirp would not focus).
        let az_chirp = lfm_chirp(az, 9000.0, 1000.0);
        let (r0, c0) = (16usize, 40usize);
        let mut echo = vec![Complex::ZERO; rows * cols];
        for a in 0..az {
            for g in 0..rg {
                echo[(r0 + a) * cols + (c0 + g)] = az_chirp[a] * range_chirp[g];
            }
        }
        let (img, out_rows, out_cols) = focus_2d(&echo, rows, cols, &range_chirp, &az_chirp);
        let mag = magnitude(&img);
        let (_pk, peak) = argmax(&mag).expect("a focused image is not empty");
        let mean: f64 = mag.iter().sum::<f64>() / mag.len() as f64;
        assert!(
            peak > 50.0 * mean,
            "peak {peak} not dominant over mean {mean}"
        );
        assert_eq!(mag.len(), out_rows * out_cols);
        // Focusing concentrates the target's energy (spread over rg·az input cells) into a few cells,
        // not a smear: only a handful sit above half the peak.
        let bright = mag.iter().filter(|&&v| v > 0.5 * peak).count();
        assert!(
            bright < 16,
            "point target not focused: {bright} bright cells"
        );
    }

    #[test]
    fn sinc_shift_moves_an_impulse() {
        let mut line = vec![Complex::ZERO; 32];
        line[10] = Complex::new(1.0, 0.0);
        // Integer shift is exact: the impulse lands three samples right.
        let s3 = sinc_shift(&line, 3.0, 6);
        assert!((s3[13].abs() - 1.0).abs() < 1e-9 && s3[10].abs() < 1e-9);
        // Half-sample shift interpolates symmetrically about the two neighbours (sinc(±0.5)=2/π).
        let sh = sinc_shift(&line, 0.5, 8);
        assert!((sh[10].abs() - 2.0 / core::f64::consts::PI).abs() < 1e-3);
        assert!((sh[11].abs() - 2.0 / core::f64::consts::PI).abs() < 1e-3);
    }

    #[test]
    fn rda_formulas_match_hand_values() {
        let p = RdaParams {
            prf: 1600.0,
            velocity: 7000.0,
            wavelength: 0.24,
            r0_slant: 850e3,
            range_spacing: 0.5,
        };
        // K_a = 2·7000²/(0.24·850000).
        assert!((azimuth_fm_rate(&p) - 2.0 * 7000.0 * 7000.0 / (0.24 * 850e3)).abs() < 1e-6);
        // ΔR(100 Hz) = 0.24²·850000·100²/(8·7000²).
        let expect = 0.24 * 0.24 * 850e3 * 1e4 / (8.0 * 7000.0 * 7000.0);
        assert!((range_migration(100.0, &p) - expect).abs() < 1e-9);
        assert_eq!(azimuth_frequency(0, 1024, 1600.0), 0.0);
        assert!((azimuth_frequency(1023, 1024, 1600.0) + 1600.0 / 1024.0).abs() < 1e-9);
    }

    #[test]
    fn rcmc_focuses_a_migrating_point_target() {
        // A point target whose range-compressed energy migrates across ~6 range cells over the
        // aperture (ΔR = V²η²/(2r₀)). The Range-Doppler algorithm with RCMC straightens that curve and
        // integrates coherently into one sharp peak; without RCMC the azimuth compression is defocused,
        // so its peak is far weaker and its range response smeared — the direct demonstration RCMC works.
        let (na, nr, r0_bin) = (1024usize, 96usize, 48usize);
        let p = RdaParams {
            prf: 1600.0,
            velocity: 7000.0,
            wavelength: 0.24,
            r0_slant: 850e3,
            range_spacing: 0.5,
        };
        let ka = azimuth_fm_rate(&p);
        // Sanity: the migration over the processed band is several range cells (RCM is significant).
        let band_edge = ka * (na as f64 / p.prf) / 2.0;
        let migration_cells = range_migration(band_edge, &p) / p.range_spacing;
        assert!(
            migration_cells > 4.0,
            "migration only {migration_cells} cells"
        );

        let mut echo = vec![Complex::ZERO; na * nr];
        for a in 0..na {
            let eta = (a as f64 - na as f64 / 2.0) / p.prf;
            let dr = p.velocity * p.velocity * eta * eta / (2.0 * p.r0_slant); // ΔR(η), metres
            let rc = r0_bin as f64 + dr / p.range_spacing;
            let phase = Complex::from_polar(1.0, -core::f64::consts::PI * ka * eta * eta);
            for r in 0..nr {
                echo[a * nr + r] = phase.scale(sinc(r as f64 - rc));
            }
        }

        let with = range_doppler_focus(&echo, na, nr, &p, true);
        let without = range_doppler_focus(&echo, na, nr, &p, false);
        let mag = |z: &[Complex]| z.iter().map(|c| c.abs()).collect::<Vec<_>>();
        let (mw, mo) = (mag(&with), mag(&without));
        let peak = |m: &[f64]| {
            let (mut bi, mut bv) = (0, 0.0);
            for (i, &v) in m.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            (bi, bv)
        };
        let (pi, pv) = peak(&mw);
        let (pi_no, pv_no) = peak(&mo);
        // RCMC concentrates the energy: the focused peak is markedly stronger than without it.
        assert!(pv > 1.8 * pv_no, "peak with RCMC {pv} vs without {pv_no}");
        // The peak sits at the reference range bin (RCMC brought every azimuth frequency back to it) and
        // at the azimuth correlation origin (index 0, wrapping to na−1 — the matched-filter lag of a
        // fully-illuminated, aperture-centred target).
        let (pa, pr) = (pi / nr, pi % nr);
        assert!(
            (pa <= 1 || pa >= na - 1) && (pr as isize - r0_bin as isize).abs() <= 1,
            "peak at (az {pa}, rg {pr}), expected (~0, {r0_bin})"
        );
        // The RCMC-focused response is tight in range; the un-corrected one is smeared across cells.
        // Count range bins above half the respective peak, each at its own focused azimuth line.
        let bright =
            |m: &[f64], az: usize, half: f64| (0..nr).filter(|&r| m[az * nr + r] > half).count();
        assert!(
            bright(&mw, pa, 0.5 * pv) <= bright(&mo, pi_no / nr, 0.5 * pv_no),
            "RCMC did not tighten the range response"
        );
    }

    #[test]
    fn chirp_scaling_shifts_the_compressed_position_without_interpolation() {
        // A range chirp centred at sample c₀; chirp scaling by q about τ_ref re-scales it to rate
        // K(1+q) and moves its compressed peak to (c₀+q·τ_ref)/(1+q) — verified by the shift between the
        // matched-filter peaks with and without scaling (each matched to its own rate).
        let (n, fs, k, q) = (64usize, 1.0e8, 5.0e13, 0.05);
        let c0 = 200.0 + (n as f64 - 1.0) / 2.0; // lfm_chirp centres at (n−1)/2
        let tau_ref = 100.0;
        let mut sig = vec![Complex::ZERO; 512];
        sig[200..200 + n].copy_from_slice(&lfm_chirp(n, k, fs));

        let mag = |z: &[Complex]| z.iter().map(|c| c.abs()).collect::<Vec<_>>();
        let p0 = argmax(&mag(&matched_filter(&sig, &lfm_chirp(n, k, fs))))
            .expect("a matched filter output is not empty")
            .0;
        let scaled = apply_chirp_scaling(&sig, k, q, tau_ref, fs);
        let p1 = argmax(&mag(&matched_filter(
            &scaled,
            &lfm_chirp(n, k * (1.0 + q), fs),
        )))
        .expect("a matched filter output is not empty")
        .0;

        let measured = p1 as f64 - p0 as f64;
        let predicted = q * (tau_ref - c0) / (1.0 + q); // ≈ −6.26 samples
        assert!(
            (measured - predicted).abs() < 1.5,
            "chirp-scaling shift {measured} vs predicted {predicted}"
        );
    }

    #[test]
    fn src_fm_rate_vanishes_at_zero_doppler_and_matches_the_formula() {
        let p = RdaParams {
            prf: 2000.0,
            velocity: 7500.0,
            wavelength: 0.24,
            r0_slant: 600e3,
            range_spacing: 0.4,
        };
        assert!(src_fm_rate(0.0, &p).is_none()); // no coupling at zero Doppler
        // K_src = 2V²(f₀D₀)³/(R₀·c·f_η²), hand-computed at f_η = 750 Hz.
        let f_eta = 750.0;
        let f0 = C / p.wavelength;
        let mig = p.wavelength * f_eta / (2.0 * p.velocity);
        let fd = f0 * (1.0 - mig * mig).sqrt();
        let expect =
            2.0 * p.velocity * p.velocity * fd * fd * fd / (p.r0_slant * C * f_eta * f_eta);
        assert!((src_fm_rate(f_eta, &p).unwrap() - expect).abs() < 1e-3 * expect);
    }

    #[test]
    fn src_restores_the_range_resolution_of_a_coupled_target() {
        // A range-compressed point (a sinc) carrying the range-azimuth coupling of one azimuth-frequency
        // row — a range chirp of rate K_src(f_η). The coupling broadens the range impulse response;
        // secondary range compression (its exact inverse) restores the original resolution.
        // The spacing is deliberately not 1 m: at 1 m the `2·range_spacing` that sets the range-frequency
        // axis is arithmetically indistinguishable from `2/range_spacing`.
        let (na, nr) = (8usize, 256usize);
        let p = RdaParams {
            prf: 2000.0,
            velocity: 7500.0,
            wavelength: 0.24,
            r0_slant: 600e3,
            range_spacing: 0.4,
        };
        let row = 3usize;
        let f_eta = azimuth_frequency(row, na, p.prf);
        let k_src = src_fm_rate(f_eta, &p).unwrap();
        let fs_r = C / (2.0 * p.range_spacing);
        let pi = core::f64::consts::PI;

        let r0 = 128.0;
        let clean: Vec<Complex> = (0..nr)
            .map(|m| Complex::new(sinc(m as f64 - r0), 0.0))
            .collect();
        // Inject the coupling: range FFT × exp(+jπ f_τ²/K_src) × range IFFT.
        let mut dline = clean.clone();
        fft(&mut dline);
        for m in 0..nr {
            let f_tau = azimuth_frequency(m, nr, fs_r);
            dline[m] = dline[m] * Complex::from_polar(1.0, pi * f_tau * f_tau / k_src);
        }
        ifft(&mut dline);

        let mut rd = vec![Complex::ZERO; na * nr];
        rd[row * nr..row * nr + nr].copy_from_slice(&dline);
        let corrected = secondary_range_compression(&rd, na, nr, &p);

        let mag = |z: &[Complex]| z.iter().map(|c| c.abs()).collect::<Vec<_>>();
        let w_clean = resolution_3db_samples(&mag(&clean));
        let w_dist = resolution_3db_samples(&mag(&dline));
        let w_corr = resolution_3db_samples(&mag(&corrected[row * nr..row * nr + nr]));
        assert!(
            w_dist > 1.5 * w_clean,
            "coupling did not broaden the IPR: {w_dist} vs {w_clean}"
        );
        assert!(
            (w_corr - w_clean).abs() < 0.3 * w_clean,
            "SRC did not restore the resolution: {w_corr} vs clean {w_clean}"
        );
    }

    /// `lfm_chirp` places the time origin at the middle sample, and only a direct look can tell.
    ///
    /// Matched filtering is shift-invariant -- correlating a chirp against itself gives the same
    /// compressed pulse wherever the replica's origin sits -- so every focusing test in this file is
    /// blind to the centring. Three observables pin it: the middle sample of an odd-length chirp is at
    /// `t = 0` and so is exactly `1 + 0j`; `t` enters only as `t^2`, so samples equidistant from the
    /// middle are bit-identical; and at `fs = 1`, `K = 0.1` the phase is `0.1*pi*(i-4)^2`, whose
    /// cosines and sines are the fifth roots of unity.
    #[test]
    fn the_chirp_is_a_quadratic_phase_ramp_centred_on_the_middle_sample() {
        const A: f64 = 0.309_016_994_374_947_4; // cos 72 degrees = sin 18 degrees
        const B: f64 = 0.951_056_516_295_153_5; // cos 18 degrees = sin 72 degrees
        let c = lfm_chirp(9, 0.1, 1.0);
        assert_eq!((c[4].re, c[4].im), (1.0, 0.0));
        for i in 0..4 {
            assert_eq!(
                c[i].re.to_bits(),
                c[8 - i].re.to_bits(),
                "re {i} vs {}",
                8 - i
            );
            assert_eq!(
                c[i].im.to_bits(),
                c[8 - i].im.to_bits(),
                "im {i} vs {}",
                8 - i
            );
        }
        // Phases 1.6*pi, 0.9*pi, 0.4*pi, 0.1*pi at i = 0..4.
        for (i, re, im) in [(0, A, -B), (1, -B, A), (2, A, B), (3, B, A)] {
            assert!(
                (c[i].re - re).abs() < 1e-14 && (c[i].im - im).abs() < 1e-14,
                "sample {i} = {} + {}j, want {re} + {im}j",
                c[i].re,
                c[i].im
            );
        }
    }

    /// The correlation is linear, not circular: its length is the sum of the two, rounded up.
    ///
    /// A zero-padded correlation gives the same compressed pulse whatever the padding, so the only
    /// thing that says which arithmetic produced the transform length is the length itself. 96 + 32 is
    /// already a power of two and 100 + 28 rounds up to the same 128; the products 3072 and 2800 both
    /// round to 4096.
    #[test]
    fn the_correlation_length_is_the_sum_of_the_two_lengths_rounded_up() {
        let one = Complex::new(1.0, 0.0);
        let long = vec![one; 96];
        let short = vec![one; 32];
        assert_eq!(matched_filter(&long, &short).len(), 128);
        assert_eq!(matched_filter(&vec![one; 100], &vec![one; 28]).len(), 128);
        let (out, cols) = range_compress(&vec![one; 2 * 96], 2, 96, &short);
        assert_eq!((cols, out.len()), (128, 256));
    }

    /// Every azimuth column is written back to the column it came from.
    ///
    /// With a one-sample unit replica the matched filter is the identity -- the replica's spectrum is
    /// all ones -- so `azimuth_compress` must reproduce the input in the leading rows of the taller
    /// output and leave the rest zero. The point-target tests only ask whether one peak is dominant,
    /// which stays true when the write-back folds distinct columns on top of each other.
    #[test]
    fn azimuth_compression_writes_each_column_back_where_it_came_from() {
        let (rows, cols) = (4usize, 3usize);
        let data: Vec<Complex> = (0..rows * cols)
            .map(|i| Complex::new((10 * (i / cols) + i % cols + 1) as f64, 0.0))
            .collect();
        let (out, out_rows) = azimuth_compress(&data, rows, cols, &[Complex::new(1.0, 0.0)]);
        assert_eq!((out_rows, out.len()), (8, 24)); // corr_len(4, 1) = 8
        for (i, v) in out.iter().enumerate() {
            let want = if i < rows * cols { data[i].re } else { 0.0 };
            assert!(
                (v.re - want).abs() < 1e-9 && v.im.abs() < 1e-9,
                "cell {i} = {} + {}j, want {want}",
                v.re,
                v.im
            );
        }
    }

    /// The −3 dB width, against hand-computed crossings, ties and both degenerate branches.
    ///
    /// The focusing tests compare the width against sinc theory to within a whole sample, which hides
    /// the interpolation entirely: which neighbour the slope is taken from, which of two equal maxima
    /// wins, and what happens when a sample sits exactly on the half-power level or the slope sits
    /// exactly on the guard threshold.
    #[test]
    fn the_3db_width_is_the_interpolated_distance_between_the_half_power_crossings() {
        let sqrt2 = core::f64::consts::SQRT_2;
        // Peak 5.0 at index 1 over a 1.0 pedestal, shoulder 2.0 on the right. The crossings bracket
        // [0, 1] and [1, 2], so left = (5/sqrt2 − 1)/4, right = 1 + (5 − 5/sqrt2)/3 and the width is
        // (35/12)(1 − 1/sqrt2). The second 5.0 at index 5 makes the tie-break observable: taking the
        // last maximum instead gives 2.58 samples.
        let a = [1.0, 5.0, 2.0, 3.0, 4.0, 5.0, 4.0, 0.0];
        let want = (35.0 / 12.0) * (1.0 - 1.0 / sqrt2);
        let got = resolution_3db_samples(&a);
        assert!((got - want).abs() < 1e-12, "width {got}, want {want}");

        // A sample exactly at the half-power level stops the walk on it. Stepping one further lands on
        // a flat segment whose zero slope sends the interpolation down the degenerate branch, which
        // reports the sample index instead: 4 − 0 rather than 3 − 1.
        let h = 5.0 / sqrt2;
        assert_eq!(resolution_3db_samples(&[h, h, 5.0, h, h]), 2.0);

        // The slope guard, straddled exactly. `mag[1] − mag[0]` is the 1e-300 literal itself, which
        // `> 1e-300` rejects, so both crossings fall back on their sample index and the width is the
        // whole span. Half power here is 8.84e-301, below the 1e-300 shoulders, so the walks do reach
        // the ends.
        assert_eq!(
            resolution_3db_samples(&[0.0, 1e-300, 1.25e-300, 1e-300, 0.0]),
            4.0
        );
    }

    /// An input with no usable peak has no width to report, and reports it rather than panicking.
    ///
    /// Every case here was a panic, not a wrong number, and every one is reachable from the Python
    /// bridge, which hands both functions an unvalidated `Vec<f64>`. The empty slice indexed
    /// `mag[0]` to seed the maximum. The others all fail `peak > half`, so neither walk stepped off
    /// the peak and the interpolation read `mag[pk + 1]` past the end or `mag[pk - 1]` under it —
    /// the latter an unsigned underflow. The subnormal case is the one that a `peak > 0` guard
    /// would have missed: `5e-324 / √2` rounds back to `5e-324`.
    #[test]
    fn an_input_with_no_usable_peak_reports_no_width_rather_than_panicking() {
        let tiny = f64::from_bits(1);
        for mag in [
            &[][..],
            &[0.0],
            &[-2.0, -1.0],         // peak at the last index: read past the end
            &[0.0, -1.0],          // peak at the first index: unsigned underflow
            &[f64::NAN],           // no comparison against a NaN peak is ever true
            &[f64::INFINITY, 1.0], // half power is infinite too
            &[tiny, 0.5 * tiny],   // peak / √2 rounds back to peak
        ] {
            assert!(
                resolution_3db_samples(mag).is_nan(),
                "3 dB width of {mag:?} is not NaN"
            );
        }
        assert!(
            peak_sidelobe_ratio_db(&[]).is_nan(),
            "PSLR of an empty input"
        );
    }

    /// A main lobe that runs off the end of the array has no width to report.
    ///
    /// Both walks stop at an array edge while still above half power, so there is no crossing on that
    /// side. Interpolating anyway took the slope from the only neighbour left -- the one on the *far*
    /// side of the peak -- and the left case came back as −0.348 samples, a negative resolution.
    #[test]
    fn a_main_lobe_that_runs_off_the_array_has_no_width_to_report() {
        assert!(resolution_3db_samples(&[5.0, 4.0, 0.0, 0.0]).is_nan());
        assert!(resolution_3db_samples(&[0.0, 0.0, 4.0, 5.0]).is_nan());
    }

    /// The main lobe is bounded by the first turning point either side, and the sidelobe search starts
    /// one sample outside it.
    ///
    /// The focusing tests only check that the ratio lands in a −14.5..−12 dB band, which every one of
    /// these boundary moves stays inside. Each case here moves exactly one boundary: the descent runs
    /// to the last sample (so the sidelobe set on that side is empty and the boundary sample must stay
    /// out of it), the ascent runs to the first, or a plateau stops the walk where a `<=` would not.
    #[test]
    fn the_peak_sidelobe_ratio_looks_outside_the_main_lobe_and_no_further_in() {
        for (case, mag, want) in [
            (
                "descent reaches the last sample",
                &[2.0, 0.5, 5.0, 3.0][..],
                -7.958_800_173_440_752,
            ),
            (
                "ascent reaches the first sample",
                &[1.0, 2.0, 5.0, 1.0, 4.0, 0.0],
                -1.938_200_260_161_128,
            ),
            (
                "plateau on the left flank",
                &[1.0, 2.0, 2.0, 5.0, 1.0, 1.5, 0.0],
                -7.958_800_173_440_752,
            ),
            (
                "plateau on the right flank",
                &[0.0, 1.5, 1.0, 5.0, 2.0, 2.0, 1.0],
                -7.958_800_173_440_752,
            ),
        ] {
            let got = peak_sidelobe_ratio_db(mag);
            assert!(
                (got - want).abs() < 1e-12,
                "{case}: {got} dB, want {want} dB"
            );
        }
    }

    /// Bin `n/2` folds to the negative Nyquist frequency, not the positive one.
    #[test]
    fn the_bin_at_half_the_transform_length_folds_to_the_negative_frequency() {
        assert_eq!(azimuth_frequency(3, 8, 16.0), 6.0);
        assert_eq!(azimuth_frequency(4, 8, 16.0), -8.0);
        assert_eq!(azimuth_frequency(5, 8, 16.0), -6.0);
    }

    /// The shift kernel is laid down on both sides of the source sample, with its sign.
    ///
    /// A unit impulse at 10 shifted by half a sample with a 4 + 4 tap window: `out[r]` is exactly
    /// `sinc(r − 10.5)` for `r` in 7..=14 and zero outside, because only those outputs have the impulse
    /// inside their window. `sinc_shift_moves_an_impulse` checks two taps by magnitude, which cannot
    /// see the sign of the accumulation, and looks at nothing left of the source sample.
    #[test]
    fn the_shift_kernel_reaches_both_sides_of_the_source_sample_with_its_sign() {
        let mut line = vec![Complex::ZERO; 32];
        line[10] = Complex::new(1.0, 0.0);
        let s = sinc_shift(&line, 0.5, 4);
        let pi = core::f64::consts::PI;
        for (r, want) in [
            (7usize, -1.0 / (3.5 * pi)),
            (8, 1.0 / (2.5 * pi)),
            (13, 1.0 / (2.5 * pi)),
            (14, -1.0 / (3.5 * pi)),
        ] {
            assert!(
                (s[r].re - want).abs() < 1e-12,
                "out[{r}] = {}, want {want}",
                s[r].re
            );
        }
        assert_eq!(s[6].re, 0.0);
        assert_eq!(s[15].re, 0.0);
    }

    /// `K_src` against an independent evaluation, where the migration factor is large enough to see.
    ///
    /// `src_fm_rate_vanishes_at_zero_doppler_and_matches_the_formula` runs at a spaceborne geometry
    /// where `lambda*f_eta/(2V)` is 0.012 and `D0` is 1 to within 1e-4, so its 1e-3 relative tolerance
    /// cannot see any factor of `D0` at all -- eight mutants of the radicand, of `f0*D0` and of the
    /// migration factor outlived it. Here the factor is 0.75, 0.50 and −0.90, so `D0^3` is 0.29, 0.65
    /// and 0.083. References from `derivations/src_fm_rate.mac` at 60 digits.
    #[test]
    fn the_src_rate_matches_the_stationary_phase_value_where_the_migration_factor_bites() {
        fn params(wavelength: f64, velocity: f64, r0_slant: f64) -> RdaParams {
            RdaParams {
                prf: 2000.0,
                velocity,
                wavelength,
                r0_slant,
                range_spacing: 1.0,
            }
        }
        for (lam, f, v, r0, want) in [
            (0.30, 10000.0, 2000.0, 500e3, 1.5412202636373315e11),
            (0.06, 5000.0, 300.0, 800e3, 2.432327551777837e12),
            (0.25, -7200.0, 1000.0, 250e3, 7.35151377608443e10),
        ] {
            let got = src_fm_rate(f, &params(lam, v, r0)).unwrap();
            assert!(
                (got / want - 1.0).abs() < 1e-13,
                "K_src(lambda={lam}, f={f}) = {got}, want {want}"
            );
        }

        // Ratios that hold the migration factor fixed, so D0 cancels and the four remaining exponents
        // stand alone: lambda^-3, f_eta^-2, V^2, R0^-1.
        let base = src_fm_rate(10000.0, &params(0.30, 2000.0, 500e3)).unwrap();
        for (doubled, got, want) in [
            (
                "lambda x2, f_eta /2",
                src_fm_rate(5000.0, &params(0.60, 2000.0, 500e3)).unwrap(),
                0.5,
            ),
            (
                "lambda x2, V x2",
                src_fm_rate(10000.0, &params(0.60, 4000.0, 500e3)).unwrap(),
                0.5,
            ),
            (
                "f_eta x2, V x2",
                src_fm_rate(20000.0, &params(0.30, 4000.0, 500e3)).unwrap(),
                1.0,
            ),
            (
                "R0 x2",
                src_fm_rate(10000.0, &params(0.30, 2000.0, 1000e3)).unwrap(),
                0.5,
            ),
        ] {
            assert!(
                (got / base / want - 1.0).abs() < 1e-13,
                "{doubled} scaled K_src by {}, want {want}",
                got / base
            );
        }

        // The zero-Doppler guard is strict: 1e-9 itself still couples.
        assert!(src_fm_rate(1e-9, &params(0.30, 2000.0, 500e3)).is_some());
        assert!(src_fm_rate(0.0, &params(0.30, 2000.0, 500e3)).is_none());
    }

    /// The focused image is written back in row-major `(azimuth, range)` order.
    ///
    /// Feeding a scene that is constant down azimuth collapses every azimuth transform to its DC bin,
    /// so the output is that same range profile scaled by one complex constant, identically on every
    /// azimuth line. That pins the write-back index for all `na * nr` cells; `rcmc_focuses_a_migrating_
    /// point_target` only locates the single brightest one, which survives a folded layout.
    #[test]
    fn the_focused_image_is_written_back_in_azimuth_major_order() {
        let (na, nr) = (8usize, 4usize);
        let p = RdaParams {
            prf: 1600.0,
            velocity: 7000.0,
            wavelength: 0.24,
            r0_slant: 850e3,
            range_spacing: 0.5,
        };
        let rc: Vec<Complex> = (0..na * nr)
            .map(|i| Complex::new((i % nr + 1) as f64, 0.0))
            .collect();
        let out = range_doppler_focus(&rc, na, nr, &p, false);
        assert_eq!(out.len(), na * nr);
        assert!(
            out[0].abs() > 0.1,
            "degenerate fixture: |out[0]| = {}",
            out[0].abs()
        );
        for a in 0..na {
            for r in 0..nr {
                let want = out[0].scale((r + 1) as f64);
                let got = out[a * nr + r];
                assert!(
                    (got.re - want.re).abs() < 1e-9 && (got.im - want.im).abs() < 1e-9,
                    "cell (az {a}, rg {r}) = {} + {}j, want {} + {}j",
                    got.re,
                    got.im,
                    want.re,
                    want.im
                );
            }
        }
    }
}
