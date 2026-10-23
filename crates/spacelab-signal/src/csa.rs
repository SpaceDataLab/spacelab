//! The **Chirp-Scaling Algorithm** (Raney/Runge/Bamler; Cumming & Wong Ch. 7) — SAR focusing of the
//! raw echo with **no interpolation**, only FFTs and phase multiplies.
//!
//! Unlike the Range-Doppler algorithm, which straightens the range-cell migration by a sinc
//! interpolation ([`crate::sar::range_doppler_focus`]), CSA equalises the migration of every range to
//! a reference by a **chirp-scaling** phase multiply ([`crate::sar::apply_chirp_scaling`]), then does
//! range compression, secondary range compression and a *bulk* migration correction as a single
//! phase multiply in the 2-D frequency domain, and finally azimuth compression with a residual-phase
//! correction. The seven steps:
//!
//! 1. azimuth FFT → range-Doppler domain;
//! 2. chirp scaling by `exp(jπ·K_m·α·(τ−τ_ref)²)` (differential RCMC);
//! 3. range FFT → 2-D frequency domain;
//! 4. range compression + SRC + bulk RCMC: `exp(jπ·D/K_m·f_τ²)·exp(j·4π/c·(1/D−1)·R_ref·f_τ)`;
//! 5. range IFFT → range-Doppler domain;
//! 6. azimuth compression + residual phase correction;
//! 7. azimuth IFFT → focused image.
//!
//! The modified FM rate `K_m` and factors `D(f_η) = √(1−(λf_η/2V)²)`, `α = 1/D − 1` are derived by the
//! principle of stationary phase. `na` (azimuth) and `nr` (range) must be powers of two. `alloc`.

// Physical single letters and SAR doc terms mirror the literature; casts are in range.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use crate::fft::{Complex, fft, ifft};
use crate::sar::azimuth_frequency;
use alloc::vec;
use alloc::vec::Vec;

const C: f64 = 299_792_458.0;
const PI: f64 = core::f64::consts::PI;

/// The geometry and waveform parameters of a Chirp-Scaling focus.
#[derive(Clone, Copy, Debug)]
pub struct CsaParams {
    /// Pulse repetition frequency (azimuth sample rate, Hz).
    pub prf: f64,
    /// Effective (beam-footprint) velocity (m/s).
    pub velocity: f64,
    /// Radar wavelength (m).
    pub wavelength: f64,
    /// Reference (scene-centre) slant range (m).
    pub r_ref: f64,
    /// Range-frequency sample rate (Hz).
    pub range_sample_rate: f64,
    /// Transmitted range chirp FM rate (Hz/s, positive).
    pub range_fm_rate: f64,
}

/// The migration factor `D(f_η) = √(1 − (λ f_η/2V)²)`.
fn migration_factor(f_eta: f64, p: &CsaParams) -> f64 {
    let m = p.wavelength * f_eta / (2.0 * p.velocity);
    libm::sqrt((1.0 - m * m).max(1e-12))
}

/// The range-Doppler-domain modified FM rate `K_m`: `1/K_m = 1/K_r − R_ref·λ³·f_η²/(2V²c²D³)`.
fn modified_fm_rate(f_eta: f64, d: f64, p: &CsaParams) -> f64 {
    let l = p.wavelength;
    let src =
        p.r_ref * l * l * l * f_eta * f_eta / (2.0 * p.velocity * p.velocity * C * C * d * d * d);
    1.0 / (1.0 / p.range_fm_rate - src)
}

/// The chirp-scaling factor `α = 1/D − 1`: the extra migration of a target at azimuth frequency
/// `f_η`, as a fraction of the reference range's own migration.
fn scaling_factor(d: f64) -> f64 {
    1.0 / d - 1.0
}

/// The migration delay of the reference range, `τ_ref = (2·R_ref/c)·α` (s) — the point the chirp
/// scaling re-scales about, and the shift step 4's bulk correction takes back out.
fn reference_migration_delay(alpha: f64, p: &CsaParams) -> f64 {
    (2.0 * p.r_ref / C) * alpha
}

/// The range time of bin `r` of `nr`, relative to the reference delay at bin `nr/2` (s).
fn range_time(r: usize, nr: usize, dt: f64) -> f64 {
    (r as f64 - nr as f64 / 2.0) * dt
}

/// The slant-range offset of bin `r` from the reference range (m).
fn range_offset(r: usize, nr: usize, dt: f64) -> f64 {
    range_time(r, nr, dt) * C / 2.0
}

/// Step 2's chirp-scaling phase `π·K_m·α·(τ − τ_ref)²` (C&W Eq. 7.24).
fn scaling_phase(km: f64, alpha: f64, tau: f64, tau_ref: f64) -> f64 {
    let t = tau - tau_ref;
    PI * km * alpha * t * t
}

/// Step 4's single phase multiply: range compression and secondary range compression at the scaled
/// rate `K_m(1+α)`, plus the bulk migration correction `2π·τ_ref·f_τ` (C&W Eq. 7.26, 7.27).
fn compression_and_bulk_phase(km: f64, alpha: f64, f_tau: f64, p: &CsaParams) -> f64 {
    let ks = km * (1.0 + alpha);
    let comp = PI * f_tau * f_tau / ks; // exp(+jπ f_τ²/K_s) matches the exp(−jπ f_τ²/K_s) chirp
    let bulk = 2.0 * PI * (2.0 * p.r_ref / C) * alpha * f_tau;
    comp + bulk
}

/// Step 6's azimuth matched filter `4π·R_ref·D/λ`, plus the residual phase the chirp scaling leaves
/// at a range `dr` off the reference, `−4π·K_m(1+α)·α·dr²/c²` (C&W Eq. 7.29, 7.31).
fn azimuth_and_residual_phase(km: f64, alpha: f64, d: f64, dr: f64, p: &CsaParams) -> f64 {
    let az = 4.0 * PI * p.r_ref * d / p.wavelength;
    let resid = -4.0 * PI / (C * C) * km * (1.0 + alpha) * alpha * dr * dr;
    az + resid
}

/// Focus a raw SAR echo (`na` azimuth × `nr` range, row-major) by the Chirp-Scaling Algorithm. The
/// range axis is centred on the reference delay `2·r_ref/c` (bin `nr/2`). Returns the focused image.
#[must_use]
pub fn chirp_scaling_focus(raw: &[Complex], na: usize, nr: usize, p: &CsaParams) -> Vec<Complex> {
    let dt = 1.0 / p.range_sample_rate;

    // Step 1 — azimuth FFT (per range column) into the range-Doppler domain.
    let mut rd = vec![Complex::ZERO; na * nr];
    let mut col = vec![Complex::ZERO; na];
    for r in 0..nr {
        for a in 0..na {
            col[a] = raw[a * nr + r];
        }
        fft(&mut col);
        for k in 0..na {
            rd[k * nr + r] = col[k];
        }
    }

    // Precompute the azimuth-frequency-dependent factors.
    let f_eta: Vec<f64> = (0..na).map(|k| azimuth_frequency(k, na, p.prf)).collect();
    let d: Vec<f64> = f_eta.iter().map(|&f| migration_factor(f, p)).collect();
    let km: Vec<f64> = (0..na)
        .map(|k| modified_fm_rate(f_eta[k], d[k], p))
        .collect();

    // Step 2 — chirp scaling: multiply each range-Doppler row by exp(jπ·K_m·α·(τ−τ_ref)²), with
    // α = 1/D − 1 and the reference-range migration delay τ_ref = (2·r_ref/c)·α (bin nr/2 + τ_ref/dt).
    let mut line = vec![Complex::ZERO; nr];
    for k in 0..na {
        let alpha = scaling_factor(d[k]);
        let tau_ref = reference_migration_delay(alpha, p);
        for r in 0..nr {
            let ph = scaling_phase(km[k], alpha, range_time(r, nr, dt), tau_ref);
            rd[k * nr + r] = rd[k * nr + r] * Complex::from_polar(1.0, ph);
        }
    }

    // Step 3 — range FFT (per azimuth-frequency row) into the 2-D frequency domain.
    for k in 0..na {
        line.copy_from_slice(&rd[k * nr..k * nr + nr]);
        fft(&mut line);
        rd[k * nr..k * nr + nr].copy_from_slice(&line);
    }

    // Step 4 — range compression + SRC + bulk RCMC as one phase multiply. The MF+SRC quadratic uses
    // the scaled rate K_m·(1+α); the bulk RCMC is a linear phase moving the reference migration back.
    for k in 0..na {
        let alpha = scaling_factor(d[k]);
        for r in 0..nr {
            let f_tau = azimuth_frequency(r, nr, p.range_sample_rate);
            let ph = compression_and_bulk_phase(km[k], alpha, f_tau, p);
            rd[k * nr + r] = rd[k * nr + r] * Complex::from_polar(1.0, ph);
        }
    }

    // Step 5 — range IFFT back to the range-Doppler domain.
    for k in 0..na {
        line.copy_from_slice(&rd[k * nr..k * nr + nr]);
        ifft(&mut line);
        rd[k * nr..k * nr + nr].copy_from_slice(&line);
    }

    // Step 6 — azimuth compression (matched to the azimuth phase −4π·r_ref·D/λ) plus the residual
    // phase from chirp scaling, then step 7 azimuth IFFT.
    for r in 0..nr {
        let dr = range_offset(r, nr, dt);
        for k in 0..na {
            let alpha = scaling_factor(d[k]);
            let ph = azimuth_and_residual_phase(km[k], alpha, d[k], dr, p);
            col[k] = rd[k * nr + r] * Complex::from_polar(1.0, ph);
        }
        ifft(&mut col);
        for a in 0..na {
            rd[a * nr + r] = col[a];
        }
    }
    rd
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal, clippy::float_cmp)]
    use super::*;

    // A raw point-target echo at the reference range (broadside): the two-way delay curves as
    // R(η)=√(R0²+V²η²), so the range-compressed energy migrates across cells; the azimuth phase is a
    // chirp. This is what CSA must focus.
    fn simulate_raw(na: usize, nr: usize, p: &CsaParams, pulse_dur: f64) -> Vec<Complex> {
        let dt = 1.0 / p.range_sample_rate;
        let mut raw = vec![Complex::ZERO; na * nr];
        for a in 0..na {
            let eta = (a as f64 - na as f64 / 2.0) / p.prf;
            let range = libm::sqrt(p.r_ref * p.r_ref + p.velocity * p.velocity * eta * eta);
            let migration_delay = 2.0 * (range - p.r_ref) / C;
            let az_phase = -4.0 * PI * range / p.wavelength;
            for r in 0..nr {
                let tau = (r as f64 - nr as f64 / 2.0) * dt - migration_delay;
                if libm::fabs(tau) < pulse_dur / 2.0 {
                    let ph = az_phase + PI * p.range_fm_rate * tau * tau;
                    raw[a * nr + r] = Complex::from_polar(1.0, ph);
                }
            }
        }
        raw
    }

    #[test]
    fn chirp_scaling_focuses_a_migrating_point_target() {
        // A point target whose range-compressed energy migrates across ~10 range cells over the
        // aperture. CSA (all phase multiplies + FFTs, no interpolation) must collapse it to one sharp
        // peak — if any phase function's constant were wrong, it would not focus.
        let (na, nr) = (1024usize, 512usize);
        let p = CsaParams {
            prf: 1200.0,
            velocity: 7500.0,
            wavelength: 0.3,
            r_ref: 500e3,
            range_sample_rate: 150e6,
            range_fm_rate: 2.5e13,
        };
        let pulse_dur = 2.0e-6; // 300-sample chirp
        let raw = simulate_raw(na, nr, &p, pulse_dur);

        let img = chirp_scaling_focus(&raw, na, nr, &p);
        let mag: Vec<f64> = img.iter().map(|c| c.abs()).collect();
        let (mut pk, mut peak) = (0usize, 0.0);
        for (i, &v) in mag.iter().enumerate() {
            if v > peak {
                peak = v;
                pk = i;
            }
        }
        let mean: f64 = mag.iter().sum::<f64>() / mag.len() as f64;
        // The focused target is a dominant peak (energy from na·(pulse samples) input cells collapsed).
        assert!(
            peak > 200.0 * mean,
            "peak {peak} not dominant over mean {mean}"
        );
        // And it is concentrated: only a handful of cells exceed half the peak (a focused point, not a
        // smear across the migration path).
        let bright = mag.iter().filter(|&&v| v > 0.5 * peak).count();
        assert!(bright < 12, "not focused: {bright} bright cells");
        // The peak sits at the target's reference range bin (nr/2), where the migration was corrected.
        assert!(
            (pk % nr).abs_diff(nr / 2) <= 2,
            "peak range bin {} not at r_ref {}",
            pk % nr,
            nr / 2
        );
    }

    /// The parameters `modified_fm_rate` reads; the rest do not enter it.
    fn fm_params(r_ref: f64, wavelength: f64, velocity: f64, range_fm_rate: f64) -> CsaParams {
        CsaParams {
            prf: 1200.0,
            velocity,
            wavelength,
            r_ref,
            range_sample_rate: 150e6,
            range_fm_rate,
        }
    }

    /// `K_m` against an independent evaluation, and the five exponents of its correction term.
    ///
    /// The focusing test above cannot see this function. Its secondary-range-compression term is a
    /// sub-percent correction to `K_r` at any real geometry, and a point target collapses to a peak
    /// without it -- eleven mutants of the term outlived that test: the whole of `lambda^3`, all
    /// three factors of `D^3`, the 2 in the denominator, and the sign of the correction itself.
    ///
    /// The three references come from `derivations/csa_fm_rate.mac`, which evaluates Cumming & Wong
    /// Eq. 7.18 in exact rationals at 60 digits. The ratios then restate the five exponents without
    /// restating the formula, so an exponent that moves is caught even where a printed digit is not.
    #[test]
    fn the_modified_fm_rate_matches_the_stationary_phase_value_and_its_exponents() {
        for (r, lam, f, v, kr, d, want) in [
            (
                500e3,
                0.30,
                320.0,
                7500.0,
                2.5e13,
                0.80,
                2.5168019187760119e13,
            ),
            (
                800e3,
                0.06,
                1000.0,
                7000.0,
                1.0e12,
                0.95,
                1.0000228831274068e12,
            ),
            (
                250e3,
                0.25,
                -450.0,
                7600.0,
                5.0e13,
                1.00,
                5.019119830814637e13,
            ),
        ] {
            let got = modified_fm_rate(f, d, &fm_params(r, lam, v, kr));
            assert!(
                (got / want - 1.0).abs() < 1e-13,
                "K_m(f={f}, D={d}) = {got}, want {want}"
            );
        }

        // 1/K_r - 1/K_m is the correction itself, read back out of the function. Doubling each
        // argument in turn must scale it by that argument's exponent: R, lambda^3, f^2, V^-2, D^-3.
        let src = |r, lam, f, v, d| {
            let p = fm_params(r, lam, v, 2.5e13);
            1.0 / p.range_fm_rate - 1.0 / modified_fm_rate(f, d, &p)
        };
        let base = src(500e3, 0.30, 320.0, 7500.0, 0.80);
        for (doubled, got, want) in [
            ("r_ref", src(1000e3, 0.30, 320.0, 7500.0, 0.80), 2.0),
            ("wavelength", src(500e3, 0.60, 320.0, 7500.0, 0.80), 8.0),
            ("f_eta", src(500e3, 0.30, 640.0, 7500.0, 0.80), 4.0),
            ("velocity", src(500e3, 0.30, 320.0, 15000.0, 0.80), 0.25),
            (
                "migration factor",
                src(500e3, 0.30, 320.0, 7500.0, 1.60),
                0.125,
            ),
        ] {
            assert!(
                (got / base / want - 1.0).abs() < 1e-10,
                "doubling {doubled} scaled the correction by {}, want {want}",
                got / base
            );
        }
    }

    fn add_target(
        raw: &mut [Complex],
        na: usize,
        nr: usize,
        p: &CsaParams,
        pulse_dur: f64,
        eta0: f64,
        r0: f64,
    ) {
        let dt = 1.0 / p.range_sample_rate;
        for a in 0..na {
            let eta = (a as f64 - na as f64 / 2.0) / p.prf - eta0;
            let range = libm::sqrt(r0 * r0 + p.velocity * p.velocity * eta * eta);
            let delay = 2.0 * (range - p.r_ref) / C;
            let az_phase = -4.0 * PI * range / p.wavelength;
            for r in 0..nr {
                let tau = (r as f64 - nr as f64 / 2.0) * dt - delay;
                if libm::fabs(tau) < pulse_dur / 2.0 {
                    let ph = az_phase + PI * p.range_fm_rate * tau * tau;
                    raw[a * nr + r] = raw[a * nr + r] + Complex::from_polar(1.0, ph);
                }
            }
        }
    }

    /// Two targets, neither of them where the first test put its one.
    ///
    /// `chirp_scaling_focuses_a_migrating_point_target` places its target at the reference range and
    /// at broadside, which is the one cell where the algorithm's differential half does nothing:
    /// `tau - tau_ref` in the chirp scaling and `dr` in the residual phase are both zero there. It
    /// also never looks at the azimuth coordinate of the peak, only the range one. Here the second
    /// target sits 120 range cells beyond the reference and 0.15 s off broadside, so the bulk
    /// migration correction has to put it back in the right column and the azimuth compression in
    /// the right row.
    #[test]
    fn two_targets_off_the_reference_cell_each_focus_where_they_belong() {
        let (na, nr) = (1024usize, 512usize);
        let p = CsaParams {
            prf: 1200.0,
            velocity: 7500.0,
            wavelength: 0.3,
            r_ref: 500e3,
            range_sample_rate: 150e6,
            range_fm_rate: 2.5e13,
        };
        let (pulse_dur, eta0) = (2.0e-6, 0.15);
        let range_bin = C / (2.0 * p.range_sample_rate);
        let mut raw = vec![Complex::ZERO; na * nr];
        add_target(&mut raw, na, nr, &p, pulse_dur, 0.0, p.r_ref);
        add_target(
            &mut raw,
            na,
            nr,
            &p,
            pulse_dur,
            eta0,
            p.r_ref + 120.0 * range_bin,
        );

        let img = chirp_scaling_focus(&raw, na, nr, &p);
        let mag: Vec<f64> = img.iter().map(|c| c.abs()).collect();
        let mean: f64 = mag.iter().sum::<f64>() / mag.len() as f64;

        // Each target's own cell, and each is the maximum of the 5x5 around it.
        let cells = [(na / 2, nr / 2), (na / 2 + 180, nr / 2 + 120)];
        let mut peaks = [0.0f64; 2];
        for (t, &(ca, cr)) in cells.iter().enumerate() {
            peaks[t] = mag[ca * nr + cr];
            assert!(
                peaks[t] > 200.0 * mean,
                "target {t} at ({ca}, {cr}) is {} against a mean of {mean:e}",
                peaks[t]
            );
            for a in ca - 2..=ca + 2 {
                for r in cr - 2..=cr + 2 {
                    if (a, r) == (ca, cr) {
                        continue;
                    }
                    assert!(
                        mag[a * nr + r] < peaks[t],
                        "target {t} is not the peak of its own neighbourhood: \
                         ({a}, {r}) = {} >= {}",
                        mag[a * nr + r],
                        peaks[t]
                    );
                }
            }
        }

        // The off-reference target is the one the differential correction is for. Measured, it keeps
        // 93.49% of the peak the on-reference target reaches; the band is wide enough for a libm
        // that rounds differently and far too narrow for a phase term that does not belong.
        let kept = peaks[1] / peaks[0];
        assert!(
            (kept - 0.9349).abs() < 2e-3,
            "the off-reference target kept {kept} of the reference peak"
        );
    }

    /// A wide-beam geometry, where the differential half of the algorithm has real work to do.
    ///
    /// The migration factor `alpha = 1/D − 1` is the square of the half-beamwidth sine over two, and
    /// at the spaceborne fixture above that is 7e-5: the chirp-scaling quadratic, the bulk migration
    /// shift and the residual phase are all a hundredth of a milliradian, and twenty-nine mutants of
    /// them outlived a test that only asks where the peak is. Here it is 0.036 -- an airborne/drone
    /// L-band collection at 2 km, 100 m/s, over a 2048-pulse aperture -- so the same three terms are
    /// 2.9, 21.8 range cells and 2.5 radians. CSA is named after the first of them; this is the
    /// regime it exists for, and the observable is the impulse response it produces.
    #[test]
    fn a_wide_beam_geometry_puts_the_differential_correction_to_work() {
        let (na, nr) = (2048usize, 512usize);
        let p = CsaParams {
            prf: 350.0,
            velocity: 100.0,
            wavelength: 0.3,
            r_ref: 2000.0,
            range_sample_rate: 150e6,
            range_fm_rate: 2.5e13,
        };
        let pulse_dur = 2.0e-6;
        let alpha = 1.0 / migration_factor(p.prf / 2.0, &p) - 1.0;
        assert!(
            alpha > 0.03,
            "the fixture is not wide-beam after all: alpha = {alpha}"
        );

        let mut raw = vec![Complex::ZERO; na * nr];
        add_target(&mut raw, na, nr, &p, pulse_dur, 0.0, p.r_ref);
        let img = chirp_scaling_focus(&raw, na, nr, &p);
        let mag: Vec<f64> = img.iter().map(|c| c.abs()).collect();

        // The target sits at the reference range, broadside, so it must focus at the middle cell --
        // the chirp scaling moved it by 21.8 cells and the bulk correction has to bring it back.
        let mut brightest = 0;
        for i in 1..mag.len() {
            if mag[i] > mag[brightest] {
                brightest = i;
            }
        }
        assert_eq!((brightest / nr, brightest % nr), (na / 2, nr / 2));

        // Rectangular-window matched filtering gives a sinc: -3 dB width 0.886*fs/B and first
        // sidelobe -13.26 dB, in range against the transmitted bandwidth and in azimuth against the
        // Doppler bandwidth ka*T. A residual quadratic phase of order a radian broadens the main lobe
        // and lifts the sidelobes off that floor, which is what makes the differential terms visible.
        let ka = 2.0 * p.velocity * p.velocity / (p.wavelength * p.r_ref);
        let cut_range: Vec<f64> = (0..nr).map(|r| mag[na / 2 * nr + r]).collect();
        let cut_azimuth: Vec<f64> = (0..na).map(|a| mag[a * nr + nr / 2]).collect();
        for (axis, cut, theory, tol) in [
            (
                "range",
                &cut_range,
                0.886 * p.range_sample_rate / (p.range_fm_rate * pulse_dur),
                0.10,
            ),
            (
                "azimuth",
                &cut_azimuth,
                0.886 * p.prf * p.prf / (ka * na as f64),
                0.20,
            ),
        ] {
            let width = crate::sar::resolution_3db_samples(cut);
            let pslr = crate::sar::peak_sidelobe_ratio_db(cut);
            assert!(
                (width / theory - 1.0).abs() < tol,
                "{axis} -3 dB width {width} samples against a theoretical {theory}"
            );
            assert!(
                pslr < -13.0,
                "{axis} PSLR {pslr} dB is above the sinc floor"
            );
        }
    }

    /// The wide-beam parameters the phase-term tests below share.
    fn phase_params() -> CsaParams {
        CsaParams {
            prf: 350.0,
            velocity: 100.0,
            wavelength: 0.3,
            r_ref: 2000.0,
            range_sample_rate: 150e6,
            range_fm_rate: 2.5e13,
        }
    }

    /// The four phase terms of the algorithm, against an independent evaluation.
    ///
    /// A focused image cannot separate the factors of these products. At the spaceborne fixture the
    /// scaling factor is 7e-5, so every term that carries it is a microradian; at a beam wide enough
    /// for them to matter, the single-reference azimuth matched filter no longer focuses anything off
    /// the reference range -- and the residual phase is zero *at* the reference range by construction,
    /// so it has nowhere left to show itself. Twenty-eight mutants lived in that gap. The terms were
    /// given names so they could be asked directly; references from `derivations/csa_phases.mac` at
    /// 60 digits.
    #[test]
    fn the_four_phase_terms_match_their_closed_forms() {
        let p = phase_params();
        let (d, km, tau, f_tau, dr) = (0.9645, 2.5e13, 1.3e-6, 41e6, 5000.0);
        let alpha = scaling_factor(d);
        let tau_ref = reference_migration_delay(alpha, &p);
        let dt = 1.0 / p.range_sample_rate;
        let az_only = azimuth_and_residual_phase(km, alpha, d, 0.0, &p);
        for (what, got, want) in [
            ("alpha", alpha, 0.036_806_635_562_467_6),
            ("tau_ref", tau_ref, 4.910_948_835_473_053e-7),
            (
                "scaling phase",
                scaling_phase(km, alpha, tau, tau_ref),
                1.891_520_999_931_377,
            ),
            (
                "compression + bulk",
                compression_and_bulk_phase(km, alpha, f_tau, &p),
                330.252_891_957_561_25,
            ),
            ("azimuth phase", az_only, 80_801.763_050_329_48),
            (
                // The residual is the whole of the range-offset dependence: at dr = 0 it vanishes.
                "residual phase",
                azimuth_and_residual_phase(km, alpha, d, dr, &p) - az_only,
                -3_334.819_747_889_322_6,
            ),
            (
                "range time",
                range_time(300, 512, dt),
                2.933_333_333_333_333e-7,
            ),
            (
                "range offset",
                range_offset(300, 512, dt),
                43.969_560_506_666_67,
            ),
        ] {
            assert!(
                (got / want - 1.0).abs() < 1e-12,
                "{what} = {got}, want {want}"
            );
        }
    }

    /// Assert that `got` is `want` times `base`, to the last few bits.
    fn ratio(got: f64, base: f64, want: f64, what: &str) {
        assert!(
            (got / base / want - 1.0).abs() < 1e-12,
            "{what} scaled the term by {}, want {want}",
            got / base
        );
    }

    /// The exponents of steps 2 and 4, restated as doubling ratios and a parity.
    ///
    /// A value alone leaves a factor free to move if another moves with it; these say which power
    /// belongs to which argument. The parity split is what separates the two halves of step 4: the
    /// compression quadratic is even in `f_tau` and the bulk migration shift is odd, so evaluating
    /// the sum at `+f` and `-f` recovers each of them alone.
    #[test]
    fn the_scaling_and_compression_phases_carry_the_exponents_they_should() {
        let p = phase_params();
        let (d, km, f) = (0.9645, 2.5e13, 41e6);
        let a = scaling_factor(d);

        // Step 2: K_m and alpha linearly, the time from the reference delay quadratically.
        let s = |km: f64, a: f64, t: f64| scaling_phase(km, a, t, 0.0);
        let base = s(km, a, 1.3e-6);
        ratio(
            s(2.0 * km, a, 1.3e-6),
            base,
            2.0,
            "K_m in the scaling phase",
        );
        ratio(
            s(km, 2.0 * a, 1.3e-6),
            base,
            2.0,
            "alpha in the scaling phase",
        );
        ratio(s(km, a, 2.6e-6), base, 4.0, "the time in the scaling phase");

        // Step 4, split by parity into the compression quadratic and the bulk shift.
        let both = |km: f64, a: f64, f: f64, p: &CsaParams| {
            let (plus, minus) = (
                compression_and_bulk_phase(km, a, f, p),
                compression_and_bulk_phase(km, a, -f, p),
            );
            (f64::midpoint(plus, minus), (plus - minus) / 2.0)
        };
        let (comp, bulk) = both(km, a, f, &p);
        assert!(comp > 0.0 && bulk > 0.0, "degenerate split: {comp}, {bulk}");
        ratio(
            both(km, a, 2.0 * f, &p).0,
            comp,
            4.0,
            "f_tau in the compression",
        );
        ratio(
            both(2.0 * km, a, f, &p).0,
            comp,
            0.5,
            "K_m in the compression",
        );
        ratio(
            both(km, a, 2.0 * f, &p).1,
            bulk,
            2.0,
            "f_tau in the bulk shift",
        );
        ratio(
            both(km, 2.0 * a, f, &p).1,
            bulk,
            2.0,
            "alpha in the bulk shift",
        );
        let far = CsaParams {
            r_ref: 2.0 * p.r_ref,
            ..p
        };
        ratio(both(km, a, f, &far).1, bulk, 2.0, "R_ref in the bulk shift");
    }

    /// The exponents of step 6 and of the range axis it is indexed by.
    ///
    /// Same split: the azimuth matched filter does not depend on the range offset at all, so
    /// subtracting the value at `dr = 0` leaves the residual phase alone.
    #[test]
    fn the_azimuth_and_residual_phases_carry_the_exponents_they_should() {
        let p = phase_params();
        let (d, km) = (0.9645, 2.5e13);
        let a = scaling_factor(d);
        let far = CsaParams {
            r_ref: 2.0 * p.r_ref,
            ..p
        };
        let resid = |km: f64, dr: f64| {
            azimuth_and_residual_phase(km, a, d, dr, &p)
                - azimuth_and_residual_phase(km, a, d, 0.0, &p)
        };
        let base = resid(km, 5000.0);
        assert!(
            base < 0.0,
            "the residual phase should subtract, not add: {base}"
        );
        assert_eq!(
            resid(km, -5000.0),
            base,
            "the residual phase is even in the range offset"
        );
        ratio(resid(km, 10000.0), base, 4.0, "the offset in the residual");
        ratio(resid(2.0 * km, 5000.0), base, 2.0, "K_m in the residual");

        // The azimuth filter itself: R_ref and D linearly, the wavelength inversely.
        let az = |d: f64, p: &CsaParams| azimuth_and_residual_phase(km, a, d, 0.0, p);
        let base = az(d, &p);
        ratio(az(2.0 * d, &p), base, 2.0, "D in the azimuth filter");
        ratio(az(d, &far), base, 2.0, "R_ref in the azimuth filter");
        let short = CsaParams {
            wavelength: p.wavelength / 2.0,
            ..p
        };
        ratio(
            az(d, &short),
            base,
            2.0,
            "the wavelength in the azimuth filter",
        );

        // The range axis is linear in the sample interval and zero at the middle bin.
        assert_eq!(range_time(256, 512, 1.0), 0.0);
        assert_eq!(range_offset(256, 512, 1.0), 0.0);
        ratio(
            range_time(300, 512, 2.0),
            range_time(300, 512, 1.0),
            2.0,
            "dt",
        );
        ratio(
            range_offset(300, 512, 1.0),
            range_time(300, 512, 1.0),
            C / 2.0,
            "the range offset against the range time",
        );
    }

    /// The whole pipeline is unitary: FFTs and unit-modulus phase multiplies, nothing else.
    ///
    /// Two azimuth transforms and two range transforms, one forward and one inverse of each, cancel
    /// their `1/N` exactly, and every phase multiply between them has modulus one -- so the focused
    /// image carries the raw echo's energy to the last few bits. That is what says the phase multiplies
    /// are multiplies applied to the cell they were computed for: a write to the wrong row destroys
    /// energy, and adding a phasor rather than multiplying by one creates it. The focusing tests can
    /// see neither, because both leave a peak standing somewhere.
    #[test]
    fn focusing_is_unitary_so_the_image_carries_the_echo_s_energy() {
        let (na, nr) = (64usize, 32usize);
        let p = CsaParams {
            prf: 350.0,
            velocity: 100.0,
            wavelength: 0.3,
            r_ref: 2000.0,
            range_sample_rate: 150e6,
            range_fm_rate: 2.5e13,
        };
        // A deterministic full-rank echo -- every cell carries energy, so no row can be skipped
        // without the total noticing.
        let raw: Vec<Complex> = (0..na * nr)
            .map(|i| Complex::from_polar(1.0 + (i % 7) as f64, (i % 13) as f64))
            .collect();
        let energy = |z: &[Complex]| z.iter().map(|c| c.abs() * c.abs()).sum::<f64>();
        let img = chirp_scaling_focus(&raw, na, nr, &p);
        let (before, after) = (energy(&raw), energy(&img));
        assert!(
            (after / before - 1.0).abs() < 1e-12,
            "energy {before} in, {after} out"
        );

        // Linear, so zero in is zero out and scaling the echo scales the image exactly.
        let zero = chirp_scaling_focus(&vec![Complex::ZERO; na * nr], na, nr, &p);
        assert!(zero.iter().all(|c| c.re == 0.0 && c.im == 0.0));
        let doubled: Vec<Complex> = raw.iter().map(|c| c.scale(2.0)).collect();
        for (i, (a, b)) in chirp_scaling_focus(&doubled, na, nr, &p)
            .iter()
            .zip(&img)
            .enumerate()
        {
            assert_eq!((a.re, a.im), (2.0 * b.re, 2.0 * b.im), "cell {i}");
        }
    }
}
