//! Fourier-in-universal-time evaluation of the CCIR coefficients.
//!
//! Ported from Orekit's `FourierTimeSeries`. Given the effective ionisation level
//! `az` and the universal time, it collapses each monthly CCIR coefficient block into
//! the `cf2` (76) and `cm3` (49) spatial-coefficient arrays consumed by the profile
//! parameters, after interpolating between the low/high solar-activity rows using the
//! effective sunspot number `azr`.
//!
//! The collapse is `Σ_d ((1 − azr01)·loᵈ + azr01·hiᵈ)·T_d(h)`, which is **affine in `azr01`** and
//! therefore in `az`. Splitting it at that seam — [`EpochSeries`] holds the two harmonic sums,
//! which depend only on the epoch — turns the per-evaluation cost from `2·depth` multiply-adds per
//! column into one. The Galileo path needs that split: its Az is a function of the local MODIP, so
//! it re-evaluates the series at every Gauss node of every rung of the STEC ladder, where the ITU
//! path evaluates it once per ray.

use std::f64::consts::PI;

use crate::ccir::CcirCoefficients;

/// F2 spatial coefficients per epoch.
const F2_COLUMNS: usize = 76;

/// M(3000)F2 spatial coefficients per epoch.
const FM3_COLUMNS: usize = 49;

/// The epoch-only half of the CCIR collapse: each block's low- and high-solar-activity rows,
/// already summed over their time harmonics and awaiting only the `azr` interpolation.
pub(crate) struct EpochSeries {
    /// foF2 harmonic sums, low then high solar activity (76 values each).
    f2: [Vec<f64>; 2],
    /// M(3000)F2 harmonic sums, low then high solar activity (49 values each).
    m3: [Vec<f64>; 2],
}

impl EpochSeries {
    /// Sum both solar-activity rows of each block over the time harmonics at `hours`.
    pub(crate) fn new(hours: f64, coeffs: &CcirCoefficients) -> Self {
        // Time argument (Eq. 49) and its sine/cosine harmonics up to order 6.
        let t = (15.0 * hours).to_radians() - PI;
        let sc_t = sin_cos(t, 6);

        Self {
            f2: harmonic_sums(&coeffs.flatten_f2, &sc_t, F2_COLUMNS, 13),
            m3: harmonic_sums(&coeffs.flatten_fm3, &sc_t, FM3_COLUMNS, 9),
        }
    }
}

/// Fourier time series evaluated at one epoch and effective ionisation level.
pub(crate) struct FourierTimeSeries {
    /// Effective ionisation level (Az).
    pub az: f64,
    /// Effective sunspot number (Azr, Eq. 19).
    pub azr: f64,
    /// Spatial coefficients for foF2 (76 values).
    pub cf2: Vec<f64>,
    /// Spatial coefficients for M(3000)F2 (49 values).
    pub cm3: Vec<f64>,
}

impl FourierTimeSeries {
    /// Evaluate at `az` against a series whose epoch half is already summed.
    pub(crate) fn at(az: f64, epoch: &EpochSeries) -> Self {
        let mut series = Self {
            az,
            azr: 0.0,
            cf2: vec![0.0; F2_COLUMNS],
            cm3: vec![0.0; FM3_COLUMNS],
        };
        series.set_az(az, epoch);
        series
    }

    /// Re-interpolate in place for a new `az`, reusing the coefficient buffers.
    ///
    /// The Galileo STEC ladder calls this once per Gauss node and can reach 1.7e7 of them, so the
    /// two allocations [`Self::at`] makes are themselves worth hoisting out of the loop.
    pub(crate) fn set_az(&mut self, az: f64, epoch: &EpochSeries) {
        // Effective sunspot number (Eq. 19).
        let azr = libm::sqrt(167_273.0 + (az - 63.7) * 1123.6) - 408.99;
        let azr01 = azr * 0.01;
        let omazr01 = 1.0 - azr01;

        self.az = az;
        self.azr = azr;
        interpolate(&mut self.cf2, &epoch.f2, omazr01, azr01);
        interpolate(&mut self.cm3, &epoch.m3, omazr01, azr01);
    }
}

/// `out[c] = omazr01·rows[0][c] + azr01·rows[1][c]` — Eq. 44 merged with Eq. 50/51.
fn interpolate(out: &mut [f64], rows: &[Vec<f64>; 2], omazr01: f64, azr01: f64) {
    for (c, value) in out.iter_mut().enumerate() {
        *value = omazr01 * rows[0][c] + azr01 * rows[1][c];
    }
}

/// Sum a flattened `[column][harmonic][row]` block over its harmonics, per solar-activity row.
///
/// For each column the constant term plus `depth - 1` sine/cosine harmonics are summed. The two
/// rows are kept apart rather than blended here, because the blend is the only part that depends
/// on the effective ionisation level.
fn harmonic_sums(flatten: &[f64], sc_t: &[f64], columns: usize, depth: usize) -> [Vec<f64>; 2] {
    let stride = 2 * depth;
    let mut rows = [vec![0.0; columns], vec![0.0; columns]];
    for (row, out) in rows.iter_mut().enumerate() {
        for (column, value) in out.iter_mut().enumerate() {
            let base = column * stride + row;
            // Harmonic 0 is the constant term.
            let mut acc = flatten[base];
            for d in 1..depth {
                acc += flatten[base + 2 * d] * sc_t[d - 1];
            }
            *value = acc;
        }
    }
    rows
}

/// Return `[sin(a), cos(a), sin(2a), cos(2a), …, sin(na), cos(na)]`.
pub(crate) fn sin_cos(a: f64, n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * n);
    for k in 1..=n {
        let (s, c) = libm::sincos(k as f64 * a);
        out.push(s);
        out.push(c);
    }
    out
}
