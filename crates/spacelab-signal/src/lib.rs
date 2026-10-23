//! **spacesignal** — space signal processing, the second Phase-8 line.
//!
//! The integral-geometry core behind computed tomography and synthetic-aperture-radar
//! back-projection: the **Radon transform** (a field's line integrals over all angles and offsets —
//! the sinogram), its adjoint **back-projection**, and **filtered back-projection**, the ramp-filtered
//! inverse that reconstructs the field. SAR and tomographic reconstruction are the same operation on
//! different data.
//!
//! The Radon core is `no_std` + `libm`, allocation-free (caller-provided row-major slices). The
//! spectral pillars — [`fft`](mod@fft) (complex FFT), [`sar`] (synthetic-aperture-radar focusing), [`insar`]
//! (interferometry) and [`tomosar`] (tomographic elevation estimation) — add `alloc` for their FFT and
//! image buffers.

#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod csa;
pub mod fft;
pub mod insar;
pub mod polsar;
pub mod psinsar;
pub mod radon;
pub mod sar;
pub mod tomosar;

pub use csa::{CsaParams, chirp_scaling_focus};
pub use fft::{Complex, fft, ifft};
pub use insar::{
    coherence, interferogram, remove_reference_phase, residues, unwrap_phase_1d, unwrap_phase_2d,
    unwrap_phase_2d_branch_cut, unwrap_phase_2d_lstsq, wrap_phase,
};
pub use polsar::{cloude_pottier, coherency_matrix, pauli_powers, pauli_vector};
pub use psinsar::{amplitude_dispersion, sbas_invert, select_persistent_scatterers};
pub use radon::{back_project, filtered_back_projection, radon_forward, ramp_filter};
pub use sar::{
    RdaParams, apply_chirp_scaling, azimuth_compress, azimuth_fm_rate, azimuth_frequency, focus_2d,
    lfm_chirp, matched_filter, peak_sidelobe_ratio_db, range_compress, range_doppler_focus,
    range_migration, resolution_3db_samples, secondary_range_compression, sinc_shift, src_fm_rate,
};
pub use tomosar::{
    Scatterer, Sl1mmerConfig, bartlett_spectrum, capon_spectrum, detect_support, fista_lasso,
    refit_amplitudes, sample_covariance, sensing_matrix, sl1mmer, sl1mmer_spectrum,
    steering_vector,
};
