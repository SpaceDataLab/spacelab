//! `NeQuick` profile parameters at a single point and epoch.
//!
//! Ported from Orekit's `NeQuickParameters`. From the Fourier-collapsed CCIR
//! coefficients and the local geometry this derives the critical frequencies
//! (foE, foF2, foF1), the M(3000)F2 factor, the layer peak densities and heights
//! (hmE, hmF1, hmF2), the layer thicknesses, and the three Epstein-layer amplitudes.
//!
//! Gate: `derivations/nequick_parameters.mac` shows that the smoothed switches of Eq. 29, 84, 95,
//! 98, 101 and 102 are `NeqJoin`, that the latitude factor, local-time cosine and layer-thickness
//! forms are the ones the code evaluates, that χ0 = 86.23292796211615° (Eq. 28) is the double
//! nearest the fixed point where Eq. 29's two branches meet, and that 0.124 is the
//! plasma-frequency constant (CODATA 2018) to three digits; it evaluates Eq. 20 to 29 at the
//! tests' reference epochs and prints Eq. 20 to 98 at 30 digits for three profiles. The module
//! tests
//! `the_solar_declination_carries_the_second_harmonic_of_the_mean_anomaly` (G2),
//! `chi0_is_the_double_nearest_the_day_night_fixed_point` (G2) and
//! `the_profile_parameters_are_the_icd_equations_evaluated` (G2) hold the code to them.

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
    /// `rules` selects the variant's reading of the E- and F1-layer amplitudes (Eq. 91 to 98).
    pub(crate) fn new(
        fourier: &FourierTimeSeries,
        month: u32,
        hours: f64,
        (latitude, longitude): (f64, f64),
        modip: f64,
        rules: AmplitudeRules,
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
            rules,
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
    rules: AmplitudeRules,
}

/// The two readings of the E- and F1-layer amplitudes (Eq. 91 to 98) the variants follow.
///
/// The section 2.5.5 equations place the Eq. 98 floor inside the `foF1 >= 0.5` branch only, but
/// both reference implementations apply a floor after both branches. Without one a night-time E
/// layer under a dense F2 peak goes negative: -2.54 against the Galileo reference's 0.05 at
/// (5.25 N, 307.19 E), April, 00 UT, high solar activity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AmplitudeRules {
    /// The Galileo ICD's pseudo-code (Annex F) and the JRC implementation that produced the Annex E
    /// data: no F1 layer below foF1 = 0.5, and Eq. 98's floor of 0.05 after either branch.
    Galileo,
}

impl AmplitudeRules {
    /// Whether a profile with this foF1 (MHz) carries an F1 layer (Eq. 91, 93).
    fn has_f1_layer(self, fo_f1: f64) -> bool {
        match self {
            Self::Galileo => fo_f1 >= 0.5,
        }
    }

    /// Whether a negative F1 amplitude out of the Eq. 94 to 96 iteration stands.
    fn keeps_a_negative_f1_amplitude(self) -> bool {
        match self {
            Self::Galileo => true,
        }
    }

    /// The value the E-layer amplitude is floored to (Eq. 98).
    fn e_floor(self) -> f64 {
        match self {
            Self::Galileo => 0.05,
        }
    }
}

/// F2, F1 and E layer Epstein amplitudes (Eq. 90 to 98), under the variant's [`AmplitudeRules`].
fn compute_layer_amplitudes(p: &LayerInputs) -> [f64; 3] {
    // F2 layer amplitude (Eq. 90).
    let a1 = 4.0 * p.nm_f2;
    // No F1 layer (Eq. 91, 92).
    let without_f1 = (0.0, 4.0 * (p.nm_e - epst(a1, p.hm_f2, p.b2_bot, p.hm_e)));

    let (a2, a3) = if p.rules.has_f1_layer(p.fo_f1) {
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
        if a2a < 0.0 && !p.rules.keeps_a_negative_f1_amplitude() {
            without_f1
        } else {
            (a2a, a3a)
        }
    } else {
        without_f1
    };

    // Eq. 98: the smoothing argument is centred on 0.005 under both rules.
    [a1, a2, join(a3, p.rules.e_floor(), 60.0, a3 - 0.005)]
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
            rules: AmplitudeRules::Galileo,
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
    /// references are Eq. 20–29 evaluated at 50 digits in `derivations/nequick_parameters.mac`
    /// (to 1e-15); the second point is where the mutant `sin(2 + am)` shows most, and the first is
    /// a saturated one, where `join` pins xeff at the day/night transition and the harmonic cannot
    /// be seen at all. (G2)
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
        // Eq. 35 with seasp = 0: foE = sqrt(1.112²·sqrt(Az)·cos(xeff)^0.6 + 0.49), evaluated at 50
        // digits in derivations/nequick_parameters.mac (to 1e-15). `0.3 + φ` gives
        // seasp = tanh(0.15) = 0.1489 and foE = 3.4446314271.
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
    fn the_e_layer_floor_applies_when_there_is_no_f1_layer() {
        let mut inputs = layer_inputs(0.065, 0.0, 0.0, 39.6);
        inputs.hm_f2 = 454.4;
        inputs.b2_bot = 61.8;
        let raw = 4.0
            * (inputs.nm_e - epst(4.0 * inputs.nm_f2, inputs.hm_f2, inputs.b2_bot, inputs.hm_e));
        assert!(raw < -1.0, "the fixture must drive Eq. 92 negative: {raw}");
        let a = compute_layer_amplitudes(&inputs);
        assert!((a[2] - 0.05).abs() < 1.0e-12, "E amplitude {}", a[2]);
    }

    /// (G2) χ0 of Eq. 28 is the double nearest the angle at which Eq. 29's two branches meet.
    ///
    /// `derivations/nequick_parameters.mac` solves `χ = 90 − 0.24·exp(20 − 0.2·χ)` at 50 digits:
    /// the root is 5.4e-15 below the printed 86.23292796211615 -- more than half a unit of its last
    /// digit, so the print is not the root rounded to 16 digits but the shortest decimal of the
    /// double nearest it, which is what `X0` stores.
    #[test]
    #[allow(clippy::float_cmp)] // nearest double is the claim
    fn chi0_is_the_double_nearest_the_day_night_fixed_point() {
        const ROOT: f64 = 86.232_927_962_116_144_636_705_101_678_2;
        assert_eq!(X0, ROOT);
    }

    /// (G2) The whole parameter set is ICD Eq. 20 to 98 evaluated as printed.
    ///
    /// The references are `derivations/nequick_parameters.mac`'s 50-digit evaluation of Eq. 20 to
    /// 98 -- the ICD's own Legendre index offsets K and H (Eq. 65/66, 73/74), Annex F's smoothed
    /// foF1, the clipped exponential of F.2.12.2 -- on synthetic CCIR series that keep foF2 near
    /// 7 MHz and M(3000)F2 near 3: `cf2₁ = 7`, `cf2ₗ = (−1)ˡ·0.3/l`, `cm3₁ = 3`, `cm3ₗ = (−1)ˡ/(20·l)`.
    /// A June noon carries an F1 layer through the five-pass amplitude iteration; a November
    /// night has none, and its E-layer floor tells the two rule sets apart.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the derivation's 30-digit values, kept whole"
    )]
    fn the_profile_parameters_are_the_icd_equations_evaluated() {
        // (azr, foF2, NmF2, hmF2, hmF1, hmE, B2bot, B1top, B1bot, BEtop, BEbot, A1, A2, A3).
        const JUNE_NOON: [f64; 14] = [
            105.052_488_516_270_413_559_146_165_446,
            7.043_495_994_349_224_462_687_737_501_02,
            6.151_743_641_979_282_710_258_197_385_15,
            258.176_853_980_994_448_878_864_560_195,
            189.088_426_990_497_224_439_432_280_098,
            120.0,
            28.705_055_760_744_563_039_020_836_398_2,
            20.726_528_097_149_167_331_829_684_029_3,
            34.544_213_495_248_612_219_716_140_048_8,
            34.544_213_495_248_612_219_716_140_048_8,
            5.0,
            24.606_974_567_917_130_841_032_789_540_6,
            4.646_821_197_616_198_222_508_240_059_28,
            4.532_633_015_406_554_047_443_021_166_59,
        ];
        const NOVEMBER_NIGHT: [f64; 14] = [
            136.855_839_042_490_091_234_527_659_509,
            6.978_507_609_248_058_021_530_396_166_25,
            6.038_746_488_089_297_761_351_022_221_77,
            314.628_453_835_405_347_412_176_671_682,
            217.314_226_917_702_673_706_088_335_841,
            120.0,
            28.812_384_488_624_776_953_969_107_835_4,
            29.194_268_075_310_802_111_826_500_752_3,
            48.657_113_458_851_336_853_044_167_920_6,
            48.657_113_458_851_336_853_044_167_920_6,
            5.0,
            24.154_985_952_357_191_045_404_088_887_1,
            0.0,
            0.131_448_870_509_403_342_359_589_165_719,
        ];
        let cf2: Vec<f64> = (1..=76_u32)
            .map(|l| {
                if l == 1 {
                    7.0
                } else {
                    sign(l) * 0.3 / f64::from(l)
                }
            })
            .collect();
        let cm3: Vec<f64> = (1..=49_u32)
            .map(|l| {
                if l == 1 {
                    3.0
                } else {
                    sign(l) / (20.0 * f64::from(l))
                }
            })
            .collect();
        let profile = |month, hours, at, modip, az: f64, rules| {
            let fourier = FourierTimeSeries {
                az,
                azr: libm::sqrt(167_273.0 + (az - 63.7) * 1123.6) - 408.99,
                cf2: cf2.clone(),
                cm3: cm3.clone(),
            };
            NeQuickParameters::new(&fourier, month, hours, at, modip, rules)
        };
        let check = |p: &NeQuickParameters, want: &[f64; 14], tag: &str| {
            let got = [
                p.azr,
                want[1],
                p.nm_f2,
                p.hm_f2,
                p.hm_f1,
                p.hm_e,
                p.b2_bot,
                p.b1_top,
                p.b1_bot,
                p.be_top,
                p.be_bot,
                p.amplitudes[0],
                p.amplitudes[1],
                p.amplitudes[2],
            ];
            for (k, (&x, &y)) in got.iter().zip(want).enumerate() {
                assert!(
                    (x - y).abs() <= 1.0e-12 * y.abs().max(1.0),
                    "{tag}, quantity {k}: {x} vs {y}"
                );
            }
        };
        let galileo = AmplitudeRules::Galileo;
        let june = profile(6, 12.0, (0.6, 0.3), 40.0, 150.0, galileo);
        check(&june, &JUNE_NOON, "june noon");
        let night = profile(11, 2.0, (0.1, -1.0), 5.0, 180.0, galileo);
        check(&night, &NOVEMBER_NIGHT, "november night");
    }

    /// `(−1)ˡ` for the synthetic CCIR series.
    fn sign(l: u32) -> f64 {
        if l.is_multiple_of(2) { 1.0 } else { -1.0 }
    }
}
