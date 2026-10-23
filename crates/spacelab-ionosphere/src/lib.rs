//! `NeQuick` ionospheric model.
//!
//! This crate computes the ionospheric electron density N(h) and the Slant Total
//! Electron Content (STEC) along a receiver-to-satellite ray using the `NeQuick` model.
//! Every driver shares the same electron-density core:
//!
//! - [`NeQuickGalileo`] — the NeQuick-G single-frequency variant driven by the three
//!   Galileo broadcast effective-ionisation coefficients, with the 5°×10° MODIP grid.
//!
//! # Sources
//!
//! - European Union (2016). *European GNSS (Galileo) Open Service — Ionospheric
//!   Correction Algorithm for Galileo Single Frequency Users*, issue 1.2
//!   (the NeQuick-G / OS-SIS-ICD specification).
//! - ITU-R Recommendation P.531 (ionospheric propagation data and prediction methods),
//!   the `NeQuick` 2 reference model.
//! - Ported from Orekit's `org.orekit.models.earth.ionosphere.nequick` package
//!   (Apache-2.0), classes `NeQuickGalileo` and `NeQuickItu`. NeQuick-G departs from
//!   Orekit where Orekit departs from the ICD: Az is fixed by the MODIP at the receiver
//!   (section 2.5.4.4), the E-layer amplitude floor applies without an F1 layer too
//!   (Annex F), and the slant TEC is integrated with the ICD's Kronrod G7-K15 quadrature
//!   (Annex F.2.6). It reproduces the JRC reference implementation to rounding on the
//!   108 Annex E cases in all twelve months.
//!
//! The monthly CCIR coefficient maps (`assets/ccir11.asc` … `ccir22.asc`) and the
//! Galileo 5°×10° modified-dip latitude grid (`assets/modipNeQG_wrapped.asc`) are the
//! files embedded in the ICD (© European Union, reused with attribution), with line
//! endings normalised. Orekit's copies of the CCIR maps differ from them in the eighth
//! significant digit, and for July to December in the last F2 coefficient as well.
//!
//! # Example
//!
//! ```
//! use spacelab_ionosphere::{DateTimeComponents, GeodeticPoint, NeQuickGalileo};
//!
//! // High solar activity broadcast coefficients.
//! let model = NeQuickGalileo::new([236.831641, -0.39362878, 0.00402826613]);
//! let receiver = GeodeticPoint::new(82.49_f64.to_radians(), 297.66_f64.to_radians(), 78.11);
//! let satellite = GeodeticPoint::new(54.29_f64.to_radians(), 8.23_f64.to_radians(), 20_281_546.18);
//! // 2018-04-02 00:00:00 UTC → April, midnight.
//! let epoch = DateTimeComponents::new(4, 0.0);
//! let stec = model.stec(epoch, receiver, satellite);
//! // ICD Annex E prints 20.40 TECU; the JRC reference implementation gives 20.40224.
//! assert!((stec - 20.40224).abs() < 1.0e-5);
//! ```

// The transcribed ITU-R coefficients and the equation-mirroring variable names below
// are kept identical to the reference implementation; the following lints fight that
// deliberate fidelity rather than a defect, so they are relaxed crate-wide.
#![allow(clippy::unreadable_literal)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::similar_names)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

mod ccir;
mod fourier;
mod galileo;
mod math;
mod model;
mod modip;
mod parameters;
mod ray;
mod types;

pub use galileo::NeQuickGalileo;
pub use types::{DateTimeComponents, GeodeticPoint};

/// The ionospheric group-path delay `Δ = 40.3·TEC/f²` (metres) from a slant TEC.
///
/// `stec_tecu` is the slant TEC in TECU (`10¹⁶` electrons/m², as [`NeQuickGalileo::stec`]
/// return); `frequency_hz` is the carrier frequency in Hz. The delay scales as
/// `1/f²`, so it shrinks on higher bands. This is Orekit's `NeQuickModel.pathDelay` factor
/// (`DELAY_FACTOR = 40.3·10¹⁶`) — the standard first-order (IONEX) ionospheric range error; it turns a
/// TEC into the pseudorange bias a single-frequency receiver must remove.
#[must_use]
pub fn path_delay(stec_tecu: f64, frequency_hz: f64) -> f64 {
    40.3e16 * stec_tecu / (frequency_hz * frequency_hz)
}

#[cfg(test)]
mod delay_tests {
    use super::path_delay;

    #[test]
    fn l1_twenty_tecu_delay_is_about_three_metres_and_scales_as_inverse_square() {
        let l1 = 1_575.42e6; // GPS/Galileo L1, Hz
        let d = path_delay(20.0, l1);
        assert!((3.2..3.3).contains(&d), "L1 20-TECU delay {d}");
        // Inverse-square in frequency: the L2 (1227.60 MHz) delay is larger by (f_L1/f_L2)².
        let l2 = 1_227.60e6;
        let ratio = path_delay(20.0, l2) / d;
        assert!(((l1 / l2).powi(2) - ratio).abs() < 1e-9, "ratio {ratio}");
        // Zero TEC → zero delay.
        assert!(path_delay(0.0, l1) < 1e-12);
    }
}
