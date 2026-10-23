//! `NeQuick` profile parameters at a single point and epoch.
//!
//! Ported from Orekit's `NeQuickParameters`. From the Fourier-collapsed CCIR
//! coefficients and the local geometry this derives the critical frequencies
//! (foE, foF2, foF1), the M(3000)F2 factor, the layer peak densities and heights
//! (hmE, hmF1, hmF2), the layer thicknesses, and the three Epstein-layer amplitudes.

use std::f64::consts::PI;

use crate::fourier::{FourierTimeSeries, sin_cos};
use crate::math::{clip_exp, join};
use crate::types::SinCos;

/// Solar zenith angle at the day/night transition, degrees.
const X0: f64 = 86.23292796211615;

/// Ionospheric profile parameters used to evaluate the electron density.
pub(crate) struct NeQuickParameters {
    /// Effective sunspot number (Azr).
    pub azr: f64,
    /// F2 layer maximum density, 10¹¹ m⁻³.
    pub nm_f2: f64,
    /// F2 layer peak height, km.
    pub hm_f2: f64,
    /// F1 layer peak height, km.
    pub hm_f1: f64,
    /// E layer peak height, km.
    pub hm_e: f64,
    /// F2 layer bottom thickness, km.
    pub b2_bot: f64,
    /// F1 layer top thickness, km.
    pub b1_top: f64,
    /// F1 layer bottom thickness, km.
    pub b1_bot: f64,
    /// E layer top thickness, km.
    pub be_top: f64,
    /// E layer bottom thickness, km.
    pub be_bot: f64,
    /// F2, F1 and E layer Epstein amplitudes.
    pub amplitudes: [f64; 3],
}

impl NeQuickParameters {
    /// Derive the profile parameters at `(latitude, longitude)` (radians) with the given
    /// MODIP (degrees), effective ionisation series, calendar month and universal time.
    /// `floor_e_without_f1` selects whether the Eq. 98 E-layer floor also applies without an F1
    /// layer.
    pub(crate) fn new(
        fourier: &FourierTimeSeries,
        month: u32,
        hours: f64,
        (latitude, longitude): (f64, f64),
        modip: f64,
        floor_e_without_f1: bool,
    ) -> Self {
        let azr = fourier.azr;
        let xeff = compute_effective_solar_angle(month, hours, latitude, longitude);

        // E layer (Eq. 78, 36).
        let hm_e = 120.0;
        let fo_e = compute_fo_e(month, fourier.az, xeff, latitude);
        let nm_e = 0.124 * fo_e * fo_e;

        // F2 layer.
        let sc_l = sin_cos(longitude, 8);
        let fo_f2 = compute_fo_f2(modip, &fourier.cf2, latitude, &sc_l);
        let m_f2 = compute_m_f2(modip, &fourier.cm3, latitude, &sc_l);
        let nm_f2 = 0.124 * fo_f2 * fo_f2;
        let hm_f2 = compute_hm_f2(fo_f2, fo_e, m_f2);

        // F1 layer.
        let fo_f1 = compute_fo_f1(fo_e, fo_f2);
        let nm_f1 = if fo_f1 <= 0.0 && fo_e > 2.0 {
            let fo_epopf = fo_e + 0.5;
            0.124 * fo_epopf * fo_epopf
        } else {
            0.124 * fo_f1 * fo_f1
        };
        let hm_f1 = f64::midpoint(hm_f2, hm_e);

        // Thickness parameters (Eq. 85 to 89).
        let a = 0.01 * clip_exp(-3.467 + 0.857 * libm::log(fo_f2 * fo_f2) + 2.02 * libm::log(m_f2));
        let b2_bot = 0.385 * nm_f2 / a;
        let b1_top = 0.3 * (hm_f2 - hm_f1);
        let b1_bot = 0.5 * (hm_f1 - hm_e);
        let be_top = b1_bot.max(7.0);
        let be_bot = 5.0;

        let amplitudes = compute_layer_amplitudes(&LayerInputs {
            nm_e,
            nm_f1,
            fo_f1,
            nm_f2,
            hm_f2,
            hm_e,
            hm_f1,
            b2_bot,
            be_top,
            b1_bot,
            floor_e_without_f1,
        });

        Self {
            azr,
            nm_f2,
            hm_f2,
            hm_f1,
            hm_e,
            b2_bot,
            b1_top,
            b1_bot,
            be_top,
            be_bot,
            amplitudes,
        }
    }

    /// E layer thickness for a height `h` (km), Eq. 109.
    pub(crate) fn be(&self, h: f64) -> f64 {
        if h > self.hm_e {
            self.be_top
        } else {
            self.be_bot
        }
    }

    /// F1 layer thickness for a height `h` (km), Eq. 110.
    pub(crate) fn bf1(&self, h: f64) -> f64 {
        if h > self.hm_f1 {
            self.b1_top
        } else {
            self.b1_bot
        }
    }
}

/// Effective solar zenith angle at a point, radians (Orekit
/// `computeEffectiveSolarAngle`).
fn compute_effective_solar_angle(month: u32, hours: f64, latitude: f64, longitude: f64) -> f64 {
    // Local time (Eq. 4).
    let lt = hours + longitude / 15.0_f64.to_radians();
    // Day of year at mid-month (Eq. 20) and time (Eq. 21).
    let dy = 30.5 * f64::from(month) - 15.0;
    let t = dy + (18.0 - hours) / 24.0;
    // Solar declination (Eq. 22 to 25).
    let am = (0.9856 * t - 3.289).to_radians();
    let al = am + (1.916 * libm::sin(am) + 0.020 * libm::sin(2.0 * am) + 282.634).to_radians();
    let s_dec = 0.39782 * libm::sin(al);
    let c_dec = libm::sqrt(1.0 - s_dec * s_dec);
    // Solar zenith angle (Eq. 26, 27).
    let sc_lat = SinCos::of(latitude);
    let coef = (PI / 12.0) * (12.0 - lt);
    let c_zenith = sc_lat.s * s_dec + sc_lat.c * c_dec * libm::cos(coef);
    let angle = libm::atan2(libm::sqrt(1.0 - c_zenith * c_zenith), c_zenith);
    let x = angle.to_degrees();
    // Effective solar zenith angle (Eq. 28).
    let xeff = join(90.0 - 0.24 * clip_exp(20.0 - 0.2 * x), x, 12.0, x - X0);
    xeff.to_radians()
}

/// E layer critical frequency, MHz (Orekit `computefoE`).
fn compute_fo_e(month: u32, az: f64, xeff: f64, latitude: f64) -> f64 {
    let lat = latitude.to_degrees();
    let sq_az = libm::sqrt(az);
    // seas parameter (Eq. 30 to 32).
    let seas = match month {
        1 | 2 | 11 | 12 => -1.0,
        3 | 4 | 9 | 10 => 0.0,
        _ => 1.0,
    };
    // Latitudinal dependence (Eq. 33, 34).
    let ee = clip_exp(0.3 * lat);
    let seasp = seas * ((ee - 1.0) / (ee + 1.0));
    // Critical frequency (Eq. 35).
    let coef = 1.112 - 0.019 * seasp;
    libm::sqrt(coef * coef * sq_az * libm::pow(libm::cos(xeff), 0.6) + 0.49)
}

/// F2 layer peak height, km (Orekit `computehmF2`).
fn compute_hm_f2(fo_f2: f64, fo_e: f64, m_f2: f64) -> f64 {
    let fo = fo_f2 / fo_e;
    let ratio = join(fo, 1.75, 20.0, fo - 1.75);
    let mut delta_m = -0.012;
    if fo_e >= 1e-30 {
        delta_m += 0.253 / (ratio - 1.215);
    }
    let m_f2_sq = m_f2 * m_f2;
    let temp = libm::sqrt((0.0196 * m_f2_sq + 1.0) / (1.2967 * m_f2_sq - 1.0));
    (1490.0 * m_f2 * temp) / (m_f2 + delta_m) - 176.0
}

/// F2 layer critical frequency, MHz (Orekit `computefoF2`).
fn compute_fo_f2(modip: f64, cf2: &[f64], latitude: f64, sc_l: &[f64]) -> f64 {
    // Legendre grades (Eq. 63).
    let q = [12usize, 12, 9, 5, 2, 1, 1, 1, 1];

    let mut frequency = cf2[0];

    // MODIP powers (Eq. 57).
    let sin_modip = libm::sin(modip.to_radians());
    let mut m = [0.0f64; 12];
    m[0] = 1.0;
    for i in 1..q[0] {
        m[i] = sin_modip * m[i - 1];
        frequency += m[i] * cf2[i];
    }

    // Latitude and longitude terms.
    let mut index = 12usize;
    let cos_lat1 = libm::cos(latitude);
    let mut cos_lat_i = cos_lat1;
    for i in 1..q.len() {
        let c = cos_lat_i * sc_l[2 * i - 1];
        let s = cos_lat_i * sc_l[2 * i - 2];
        for &m_j in m.iter().take(q[i]) {
            frequency += m_j * c * cf2[index];
            index += 1;
            frequency += m_j * s * cf2[index];
            index += 1;
        }
        cos_lat_i *= cos_lat1; // Eq. 58
    }

    frequency
}

/// Maximum usable frequency factor M(3000)F2 (Orekit `computeMF2`).
fn compute_m_f2(modip: f64, cm3: &[f64], latitude: f64, sc_l: &[f64]) -> f64 {
    // Legendre grades (Eq. 71).
    let r = [7usize, 8, 6, 3, 2, 1, 1];

    let mut m3000 = cm3[0];

    // MODIP powers (Eq. 57).
    let sin_modip = libm::sin(modip.to_radians());
    let mut m = [0.0f64; 12];
    m[0] = 1.0;
    for i in 1..12 {
        m[i] = sin_modip * m[i - 1];
        if i < 7 {
            m3000 += m[i] * cm3[i];
        }
    }

    // Latitude and longitude terms.
    let mut index = 7usize;
    let cos_lat1 = libm::cos(latitude);
    let mut cos_lat_i = cos_lat1;
    for i in 1..r.len() {
        let c = cos_lat_i * sc_l[2 * i - 1];
        let s = cos_lat_i * sc_l[2 * i - 2];
        for &m_j in m.iter().take(r[i]) {
            m3000 += m_j * c * cm3[index];
            index += 1;
            m3000 += m_j * s * cm3[index];
            index += 1;
        }
        cos_lat_i *= cos_lat1; // Eq. 58
    }

    m3000
}

/// F1 layer critical frequency, MHz (Orekit `computefoF1`).
fn compute_fo_f1(fo_e: f64, fo_f2: f64) -> f64 {
    let temp = join(1.4 * fo_e, 0.0, 1000.0, fo_e - 2.0);
    let temp2 = join(0.0, temp, 1000.0, fo_e - temp);
    let value = join(temp2, 0.85 * temp2, 60.0, 0.85 * fo_f2 - temp2);
    if value < 1.0e-6 { 0.0 } else { value }
}

/// Inputs to the layer-amplitude computation, grouped to keep the signature small.
struct LayerInputs {
    nm_e: f64,
    nm_f1: f64,
    fo_f1: f64,
    nm_f2: f64,
    hm_f2: f64,
    hm_e: f64,
    hm_f1: f64,
    b2_bot: f64,
    be_top: f64,
    b1_bot: f64,
    floor_e_without_f1: bool,
}

/// F2, F1 and E layer Epstein amplitudes (Eq. 90 to 98).
///
/// The section 2.5.5 equations place the Eq. 98 floor inside the `foF1 >= 0.5` branch only, but
/// the ICD's own pseudo-code (Annex F), the JRC reference implementation that produced the Annex E
/// verification data apply it after both branches. Without it a night-time E layer under a dense
/// F2 peak goes negative: -2.54 against the reference's 0.05 at (5.25 N, 307.19 E), April,
/// 00 UT, high solar activity. `floor_e_without_f1` keeps the section 2.5.5 reading for the ITU
/// variant, which follows Orekit's port and is validated against it.
fn compute_layer_amplitudes(p: &LayerInputs) -> [f64; 3] {
    let mut amplitude = [0.0; 3];

    // F2 layer amplitude (Eq. 90).
    let a1 = 4.0 * p.nm_f2;
    amplitude[0] = a1;

    if p.fo_f1 < 0.5 {
        amplitude[1] = 0.0;
        let a3 = 4.0 * (p.nm_e - epst(a1, p.hm_f2, p.b2_bot, p.hm_e));
        amplitude[2] = if p.floor_e_without_f1 {
            join(a3, 0.05, 60.0, a3 - 0.005)
        } else {
            a3
        };
    } else {
        let mut a2a = 0.0;
        let mut a3a = 4.0 * p.nm_e;
        for _ in 0..5 {
            a2a = 4.0
                * (p.nm_f1
                    - epst(a1, p.hm_f2, p.b2_bot, p.hm_f1)
                    - epst(a3a, p.hm_e, p.be_top, p.hm_f1));
            a2a = join(a2a, 0.8 * p.nm_f1, 1.0, a2a - 0.8 * p.nm_f1);
            a3a = 4.0
                * (p.nm_e
                    - epst(a2a, p.hm_f1, p.b1_bot, p.hm_e)
                    - epst(a1, p.hm_f2, p.b2_bot, p.hm_e));
        }
        amplitude[1] = a2a;
        amplitude[2] = join(a3a, 0.05, 60.0, a3a - 0.005);
    }

    amplitude
}

/// Epstein basis function (reference section 2.5.1).
fn epst(x: f64, y: f64, z: f64, w: f64) -> f64 {
    let ex = clip_exp((w - y) / z);
    let opex = 1.0 + ex;
    x * ex / (opex * opex)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_inputs(nm_e: f64, nm_f1: f64, fo_f1: f64, nm_f2: f64) -> LayerInputs {
        LayerInputs {
            nm_e,
            nm_f1,
            fo_f1,
            nm_f2,
            hm_f2: 300.0,
            hm_e: 120.0,
            hm_f1: 210.0,
            b2_bot: 45.0,
            be_top: 45.0,
            b1_bot: 45.0,
            floor_e_without_f1: true,
        }
    }

    /// Eq. 109/110 put the peak height itself on the *bottom* side of each thickness selector.
    ///
    /// `h` reaches these from `sqrt(s² + rp²)/1000 − RE` for a Gauss node offset `s`, and `hm_f1`
    /// is itself derived from foF2, foE and M(3000)F2, so neither threshold is ever attained: the
    /// closest any sampled node gets to hmE = 120 is 116.4 km. The discontinuity they guard is not
    /// small — `be_top` against `be_bot` is a factor of nine — so which side the peak belongs to is
    /// a real question decided by two operators no STEC fixture can reach.
    #[test]
    #[allow(clippy::float_cmp)] // each side of the selector is a stored constant, returned verbatim
    fn the_layer_thickness_selectors_put_the_peak_height_on_the_bottom_side() {
        let p = NeQuickParameters {
            azr: 50.0,
            nm_f2: 2.0,
            hm_f2: 300.0,
            hm_f1: 210.0,
            hm_e: 120.0,
            b2_bot: 45.0,
            b1_top: 27.0,
            b1_bot: 45.0,
            be_top: 45.0,
            be_bot: 5.0,
            amplitudes: [0.0; 3],
        };
        assert_eq!(p.be(120.0), 5.0);
        assert_eq!(p.be(120.000_000_000_1), 45.0);
        assert_eq!(p.bf1(210.0), 45.0);
        assert_eq!(p.bf1(210.000_000_000_1), 27.0);
    }

    /// Eq. 23 carries the *second harmonic* of the solar mean anomaly, `0.020·sin(2·am)`.
    ///
    /// Its full swing is 0.04°, which propagates to about 1.5e-4 relative on foE and so to ~1.5e-4
    /// TECU of a twenty-TECU column — an order under the 1e-3 the end-to-end fixtures assert. The
    /// references are an independent evaluation of Eq. 20–28 in mpmath at 200 bits; the second
    /// point is where the mutant `sin(2 + am)` shows most, and the first is a saturated one, where
    /// `join` pins xeff at the day/night transition and the harmonic cannot be seen at all.
    #[test]
    fn the_solar_declination_carries_the_second_harmonic_of_the_mean_anomaly() {
        let mid_sky = compute_effective_solar_angle(4, 12.0, 0.6, 0.3);
        assert!(
            (mid_sky - 0.500_672_298_139_212_1).abs() < 1.0e-12,
            "mid-sky xeff {mid_sky}"
        );
        let saturated = compute_effective_solar_angle(1, 0.0, 0.0, 0.0);
        assert!(
            (saturated - 1.570_796_295_428_505_5).abs() < 1.0e-12,
            "saturated xeff {saturated}"
        );
    }

    /// Eq. 33/34's latitudinal factor is `tanh(0.15·φ)` in disguise, and it vanishes at the equator.
    ///
    /// `(e^{0.3φ} − 1)/(e^{0.3φ} + 1)` saturates to ±1 by |φ| ≈ 15°, and every fixture is at
    /// |φ| ≥ 14.3° — where `0.3 + φ` saturates identically — or in a month where `seas = 0` kills
    /// the term outright. Only at the equator does the true factor go to zero while the mutant's
    /// stays at `tanh(0.15)`.
    #[test]
    fn the_seasonal_latitude_factor_vanishes_at_the_equator() {
        // Eq. 35 with seasp = 0: foE = sqrt(1.112²·sqrt(Az)·cos(xeff)^0.6 + 0.49), evaluated in
        // mpmath at 200 bits. `0.3 + φ` gives seasp = tanh(0.15) = 0.1489 and foE = 3.4446314271.
        let got = compute_fo_e(6, 100.0, 0.5, 0.0);
        assert!(
            (got - 3.453_054_218_930_359).abs() < 1.0e-12,
            "foE at the equator: {got}"
        );
    }

    /// Eq. 39 caps foF1 at 85% of foF2, and the cap is only visible when it binds.
    ///
    /// The join at `a = 60` is saturated wherever the fixtures go: they carry foE ≈ 2.7 and
    /// foF2 ≈ 9, so the smoothing argument `0.85·foF2 − 1.4·foE` is about +3.9 and the second
    /// branch is weighted 1.8e-35. Four different ways of writing that argument all collapse to the
    /// same number there. A low-foF2 / high-foE profile puts the argument negative, which is the
    /// only regime that tells them apart.
    #[test]
    fn the_f1_critical_frequency_is_capped_at_eighty_five_percent_of_f2() {
        // foE = 3 > 2 gives temp2 = 1.4·foE = 4.2, above the cap 0.85·foF2 = 3.4, so the argument
        // is −0.8 and the capped branch 0.85·4.2 = 3.57 is selected to within 1/(1 + e^48) = 1e-21.
        assert!((compute_fo_f1(3.0, 4.0) - 3.57).abs() < 1.0e-12);
        // The uncapped regime, for the control: this is where every fixture sits.
        assert!((compute_fo_f1(2.7, 9.0) - 3.78).abs() < 1.0e-12);
    }

    /// Eq. 91–98: the amplitude iteration starts at `4·NmE`, and Eq. 98's floor is biased below.
    ///
    /// The five-pass loop is contractive — its gain is about 0.18 per pass — so after five passes
    /// the initial guess survives with weight 1e-4 and any starting value gives the same answer to
    /// ten digits. A homogeneous fixture removes the contraction instead of fighting it: with
    /// `NmE` = `NmF1` = `NmF2` = 0 every source term of Eq. 91–97 vanishes, A1 = 4·NmF2 = 0, and the map
    /// is linear and homogeneous, so its fixed point is exactly zero — reached on the first pass
    /// from the correct start and only approached from any other.
    #[test]
    #[allow(clippy::float_cmp)] // the homogeneous fixture makes the fixed point exactly zero
    fn the_amplitude_iteration_starts_at_four_times_the_e_layer_peak() {
        let a = compute_layer_amplitudes(&layer_inputs(0.0, 0.0, 4.2, 0.0));
        assert_eq!(a[0], 0.0);
        assert_eq!(a[1], 0.0);
        // Eq. 98 is join(A3, 0.05, 60, A3 − 0.005): at A3 = 0 the smoothing argument is negative,
        // so the answer is the *lower* half, 0.05/(1 + e^{−0.3}) = 0.0287221258405829. Writing the
        // argument as `A3 + 0.005` gives 0.05/(1 + e^{+0.3}) = 0.0212778741594171 and `A3 / 0.005`
        // gives 0.05/2 = 0.025.
        assert_eq!(a[2], 0.05 / (1.0 + libm::exp(-0.3)));

        // Eq. 90/91's threshold is exclusive: foF1 = 0.5 still runs the iteration, which gives the
        // F1 layer an amplitude, rather than the no-F1 arm, which zeroes it.
        let nm_f1 = 0.124 * 0.5 * 0.5;
        let boundary = compute_layer_amplitudes(&layer_inputs(0.0, nm_f1, 0.5, 0.0));
        assert!(
            boundary[1] > 0.0,
            "foF1 = 0.5 took the no-F1 arm: {boundary:?}"
        );
        let below = compute_layer_amplitudes(&layer_inputs(0.0, nm_f1, 0.499, 0.0));
        assert_eq!(below[1], 0.0);
    }

    /// The Eq. 98 floor applies without an F1 layer too, as in the reference implementation.
    ///
    /// A night-time E layer under a dense F2 peak: `NmE` = 0.065 and `NmF2` = 39.6 are the values at
    /// (5.25 N, 307.19 E), April, 00 UT, high solar activity, where the F2 tail at 120 km exceeds
    /// `NmE` and the raw Eq. 92 amplitude is negative.
    #[test]
    #[allow(clippy::float_cmp)] // the unfloored arm returns Eq. 92 itself
    fn the_e_layer_floor_applies_when_there_is_no_f1_layer() {
        let mut inputs = layer_inputs(0.065, 0.0, 0.0, 39.6);
        inputs.hm_f2 = 454.4;
        inputs.b2_bot = 61.8;
        let raw = 4.0
            * (inputs.nm_e - epst(4.0 * inputs.nm_f2, inputs.hm_f2, inputs.b2_bot, inputs.hm_e));
        assert!(raw < -1.0, "the fixture must drive Eq. 92 negative: {raw}");
        let a = compute_layer_amplitudes(&inputs);
        assert!((a[2] - 0.05).abs() < 1.0e-12, "E amplitude {}", a[2]);
        // The ITU variant keeps the unfloored Eq. 92.
        inputs.floor_e_without_f1 = false;
        assert_eq!(compute_layer_amplitudes(&inputs)[2], raw);
    }
}
