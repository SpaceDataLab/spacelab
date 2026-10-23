//! The empty / degenerate-input guards of the DSP routines: each must return a neutral result rather
//! than index out of bounds or divide by zero.

#![allow(clippy::float_cmp)] // the guarded returns are exact zeros

use spacelab_signal::{
    Complex, amplitude_dispersion, cloude_pottier, fft, residues, sbas_invert, unwrap_phase_1d,
    unwrap_phase_2d, unwrap_phase_2d_branch_cut, unwrap_phase_2d_lstsq,
};

#[test]
fn empty_and_single_element_inputs_return_early() {
    // An FFT of one sample is a no-op.
    let mut one = [Complex::ZERO];
    fft(&mut one);
    assert_eq!(one[0], Complex::ZERO);

    assert_eq!(unwrap_phase_1d(&[]), [] as [f64; 0]);
    assert_eq!(unwrap_phase_2d(&[], 0, 0), [] as [f64; 0]);
    assert_eq!(unwrap_phase_2d_lstsq(&[], 0, 0), [] as [f64; 0]);
    // A residue map needs at least one 2×2 loop.
    assert_eq!(residues(&[0.0], 1, 1), [] as [i8; 0]);
    // Below a 2×2 field the branch-cut unwrapper returns the (zero) output untouched.
    assert_eq!(unwrap_phase_2d_branch_cut(&[0.0], 1, 1), vec![0.0]);
}

#[test]
fn degenerate_polarimetry_and_time_series_return_neutral_values() {
    // A zero coherency matrix carries no entropy/anisotropy/alpha.
    assert_eq!(cloude_pottier(&[Complex::ZERO; 9]), (0.0, 0.0, 0.0));

    // A pixel that is zero in every image has infinite amplitude dispersion (zero mean).
    let zero_img = [0.0_f64];
    let dispersion = amplitude_dispersion(&[&zero_img, &zero_img], 1);
    assert!(dispersion[0].is_infinite());

    // SBAS with all-zero baselines is singular: return the zero solution, not a division by zero.
    assert_eq!(
        sbas_invert(
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            0.056,
            800e3,
            0.6,
        ),
        (0.0, 0.0),
    );
}
