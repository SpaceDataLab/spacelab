//! Shared electron-density profile, parameterised over the `NeQuick` variant.
//!
//! Ported from the electron-density core of Orekit's `NeQuickModel`. The bottomside
//! (three Epstein layers with an optional Chapman correction) and topside profiles are
//! identical across the Galileo and ITU variants; only three hooks differ — the
//! bottomside exponential arguments, the topside thickness parameter H0, and whether
//! the Chapman low-altitude correction applies. Those are the abstract methods of
//! `NeQuickModel`, captured here by the [`Variant`] trait (Template Method): the
//! shared profile calls back into the concrete variant for each hook.

use crate::fourier::FourierTimeSeries;
use crate::math::clip_exp;
use crate::parameters::{AmplitudeRules, NeQuickParameters};
use crate::types::GeodeticPoint;

/// Converts the 10¹¹ m⁻³ profile densities to m⁻³.
const DENSITY_FACTOR: f64 = 1.0e11;

/// The variant-specific hooks of the `NeQuick` electron-density profile.
///
/// These are exactly the abstract methods Orekit's `NeQuickModel` defers to its
/// Galileo and ITU subclasses.
pub(crate) trait Variant {
    /// Whether the Chapman correction applies at height `h_km` (km).
    fn apply_chapman_parameters(&self, h_km: f64) -> bool;

    /// Exponential arguments for the F2, F1 and E Epstein layers (Eq. 111 to 113).
    fn compute_exponential_arguments(&self, h: f64, params: &NeQuickParameters) -> [f64; 3];

    /// Topside thickness parameter H0 (km). `month` is consumed by the Galileo variant.
    fn compute_h0(&self, params: &NeQuickParameters, month: u32) -> f64;

    /// The variant's reading of the E- and F1-layer amplitudes (Eq. 91 to 98).
    const AMPLITUDE_RULES: AmplitudeRules;
}

/// Electron density at a point, m⁻³ (Orekit `NeQuickModel.electronDensity`).
///
/// `modip` is supplied by the caller (each variant reads it from its own MODIP grid)
/// and reused for the profile parameters, matching Orekit's single MODIP evaluation.
pub(crate) fn electron_density<V: Variant>(
    variant: &V,
    fourier: &FourierTimeSeries,
    month: u32,
    hours: f64,
    gp: GeodeticPoint,
    modip: f64,
) -> f64 {
    let params = NeQuickParameters::new(
        fourier,
        month,
        hours,
        (gp.latitude, gp.longitude),
        modip,
        V::AMPLITUDE_RULES,
    );
    let h_km = gp.altitude / 1000.0;
    if h_km <= params.hm_f2 {
        bottom_electron_density(variant, h_km, &params)
    } else {
        top_electron_density(variant, h_km, &params, month)
    }
}

/// Bottomside electron density, m⁻³ (Eq. 109 to 121).
fn bottom_electron_density<V: Variant>(variant: &V, h: f64, params: &NeQuickParameters) -> f64 {
    // B parameters for the current height (Eq. 109, 110).
    let ct = [1.0 / params.b2_bot, 1.0 / params.bf1(h), 1.0 / params.be(h)];

    let arguments = variant.compute_exponential_arguments(h, params);
    let amplitudes = params.amplitudes;

    let mut s = [0.0; 3];
    let mut ds = [0.0; 3];
    for i in 0..3 {
        if libm::fabs(arguments[i]) > 25.0 {
            s[i] = 0.0;
            ds[i] = 0.0;
        } else {
            let exp_a = clip_exp(arguments[i]);
            let op_exp_a = 1.0 + exp_a;
            s[i] = amplitudes[i] * (exp_a / (op_exp_a * op_exp_a));
            ds[i] = ct[i] * ((1.0 - exp_a) / (1.0 + exp_a));
        }
    }

    let a_no = s[0] + s[1] + s[2];
    if variant.apply_chapman_parameters(h) {
        // Chapman parameters (Eq. 119, 120).
        let bc = 1.0 - 10.0 * ((s[0] * ds[0] + s[1] * ds[1] + s[2] * ds[2]) / a_no);
        let z = 0.1 * (h - 100.0);
        // Electron density (Eq. 121).
        a_no * clip_exp(1.0 - bc * z - clip_exp(-z)) * DENSITY_FACTOR
    } else {
        a_no * DENSITY_FACTOR
    }
}

/// Topside electron density, m⁻³ (Eq. 122 to 127).
fn top_electron_density<V: Variant>(
    variant: &V,
    h: f64,
    params: &NeQuickParameters,
    month: u32,
) -> f64 {
    // Constant parameters (Eq. 122, 123).
    let g = 0.125;
    let r = 100.0;

    // Arguments deltaH and z (Eq. 124, 125).
    let delta_h = h - params.hm_f2;
    let h0 = variant.compute_h0(params, month);
    let z = delta_h / (h0 * (1.0 + (r * g * delta_h) / (r * h0 + g * delta_h)));

    // Exponential (Eq. 126) and electron density (Eq. 127).
    let ee = clip_exp(z);
    if ee > 1.0e11 {
        (4.0 * params.nm_f2 / ee) * DENSITY_FACTOR
    } else {
        let op_exp_z = 1.0 + ee;
        (4.0 * params.nm_f2 * ee / (op_exp_z * op_exp_z)) * DENSITY_FACTOR
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A variant whose three hooks are dictated by the test rather than by a ray geometry.
    ///
    /// Nothing else in the crate can put a chosen exponential argument into the profile: both real
    /// variants derive theirs from hmF2/hmF1/hmE and a clamped height, and for Galileo the clamp
    /// pins all three near −4 whenever the Chapman branch is live.
    struct Fixed {
        chapman: bool,
        arguments: [f64; 3],
        h0: f64,
    }

    impl Variant for Fixed {
        const AMPLITUDE_RULES: AmplitudeRules = AmplitudeRules::Galileo;

        fn apply_chapman_parameters(&self, _h_km: f64) -> bool {
            self.chapman
        }
        fn compute_exponential_arguments(&self, _h: f64, _p: &NeQuickParameters) -> [f64; 3] {
            self.arguments
        }
        fn compute_h0(&self, _p: &NeQuickParameters, _month: u32) -> f64 {
            self.h0
        }
    }

    fn params(amplitudes: [f64; 3]) -> NeQuickParameters {
        NeQuickParameters {
            azr: 50.0,
            nm_f2: 2.0,
            hm_f2: 300.0,
            hm_f1: 210.0,
            hm_e: 120.0,
            b2_bot: 1.0,
            b1_top: 40.0,
            b1_bot: 4.0,
            be_top: 20.0,
            be_bot: 2.0,
            amplitudes,
        }
    }

    /// Eq. 119–121, with every quantity chosen so the reference is exact arithmetic.
    ///
    /// The derivative term of Eq. 120 is unreachable end to end. `ds` is read only inside
    /// `apply_chapman_parameters`, which the ITU variant answers `false` unconditionally, and which
    /// the Galileo variant gates on h < 100 km — where its own argument clamp freezes all three
    /// exponentials at their h = 100 values, ≈ [−4.4, −2.6, −4.2]. The correction then reaches the
    /// answer through a double exponential that is 1.0 at 100 km and 1.2e-2 at 80, so the whole
    /// term is worth 0.038 TECU of a twenty-TECU column, and only the zenith fixture samples that
    /// window at all.
    ///
    /// At h = 90 the selectors give ct = [1/b2bot, 1/b1bot, 1/bebot] = [1, 1/4, 1/2]. Exponential
    /// arguments ln 3, −ln 3 and ln 9 make the Epstein weights e/(1+e)² equal to 3/16, 3/16 and
    /// 9/100, so amplitudes [16, 16, 100] give s = [3, 3, 9] and `A_no` = 15 exactly. The Eq. 120
    /// terms are ct·(1−e)/(1+e) = [−1/2, 1/8, −2/5], so
    ///   BC = 1 − 10·(3·(−1/2) + 3·(1/8) + 9·(−2/5))/15 = 1 + 3.15 = 4.15,
    ///   z  = 0.1·(90 − 100) = −1,  N = 15·exp(1 + 4.15 − e)·1e11.
    #[test]
    fn the_chapman_bottomside_carries_the_derivative_term_of_equation_120() {
        let v = Fixed {
            chapman: true,
            arguments: [3.0_f64.ln(), -(3.0_f64.ln()), 9.0_f64.ln()],
            h0: 0.0,
        };
        let got = bottom_electron_density(&v, 90.0, &params([16.0, 16.0, 100.0]));
        let want = 15.0 * (1.0 + 4.15 - std::f64::consts::E).exp() * 1.0e11;
        assert!((got - want).abs() < 1.0e-9 * want, "{got} vs {want}");
    }

    /// Eq. 118's ±25 cutoff is strict, and beyond it the layer contributes exactly nothing.
    ///
    /// The reachable Galileo arguments are all near −4, so the guard is never consulted anywhere
    /// near its threshold; and past it the else branch would add at most `amp·1.4e-11` to an `A_no` of
    /// order 30, a relative perturbation of 1e-12 that no TECU assertion can resolve.
    #[test]
    fn the_twenty_five_cutoff_is_strict_and_zeroes_the_layer_beyond_it() {
        let v = Fixed {
            chapman: false,
            arguments: [25.0, -25.0, 30.0],
            h0: 0.0,
        };
        let got = bottom_electron_density(&v, 200.0, &params([1.0e10; 3]));
        // |arg| = 25 belongs to the exponential, and e^{-25}/(1+e^{-25})² = e^{25}/(1+e^{25})², so
        // the first two layers contribute the same weight; |arg| = 30 is past the cutoff and adds 0.
        let w = 25.0_f64.exp() / (1.0 + 25.0_f64.exp()).powi(2);
        let want = 2.0 * 1.0e10 * w * 1.0e11;
        assert!((got - want).abs() < 1.0e-9 * want, "{got} vs {want}");
    }

    /// Eq. 127 in its overflow-guard form keeps the 4, the `NmF2` and the 1e11.
    ///
    /// The branch is reached — the ITU fixture that runs the ray to 205 000 km has z ≈ 40 — but the
    /// density there is 8e-6 m⁻³ over a 2e8 m path, i.e. 1.6e-13 TECU, ten orders under the
    /// assertion. So the coefficients inside it are free.
    ///
    /// With h0 = 50, hmF2 = 300 and h = 175 300 km, Eq. 124/125 give deltaH = 175 000 and
    /// z = 42.478125882020890, so e^z = 2.8e18 and the guard fires. Both forms of Eq. 127 agree to
    /// 2/e^z = 7e-19 there, so a 1e-9 gate tests the coefficients and not which branch was taken.
    #[test]
    fn the_topside_overflow_branch_keeps_the_coefficients_of_equation_127() {
        let v = Fixed {
            chapman: false,
            arguments: [0.0; 3],
            h0: 50.0,
        };
        let p = params([0.0; 3]);
        let got = top_electron_density(&v, 175_300.0, &p, 4);
        let z = 42.478_125_882_020_89_f64;
        let want = 4.0 * 2.0 * z.exp() / (1.0 + z.exp()).powi(2) * 1.0e11;
        assert!((got - want).abs() < 1.0e-9 * want, "{got} vs {want}");
    }
}
