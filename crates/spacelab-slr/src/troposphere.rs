//! Atmospheric refraction at optical wavelengths — the Mendes–Pavlis model.
//!
//! A laser pulse to a satellite and back spends about 2.4 m of extra path in the neutral atmosphere
//! at zenith, and three times that at 15° elevation. Nothing else in an SLR residual is that large,
//! so this is the correction the whole measurement rests on.
//!
//! Two pieces, from two papers by the same authors and both adopted by IERS Conventions (2010)
//! chapter 9:
//!
//! - the **zenith delay**, Mendes & Pavlis (2004), *Geophys. Res. Lett.* **31**, L14602, which is
//!   dispersive — it depends on the laser wavelength, unlike the microwave models in
//!   `spacelab-gnss`;
//! - the **mapping function** FCULa, Mendes, Prates, Pavlis, Pavlis & Langley (2002), *Geophys.
//!   Res. Lett.* **29**(10), 1414, a Marini continued fraction whose coefficients are linear in
//!   surface temperature, latitude and height.
//!
//! Ported from the IERS Conventions Center reference routines `FCULZD_HPA` and `FCUL_A`
//! (`iers-conventions.obspm.fr`, chapter 9), which this file follows term for term. The routines are
//! renamed and re-expressed in Rust as that licence requires; the differences are the units taken at
//! the boundary — radians rather than degrees for the angles, to match the rest of the workspace —
//! and the return of the two zenith components as a struct instead of three out-parameters. Neither
//! changes a number: both are checked against the reference routines' own published test cases and
//! against the compiled Fortran itself in `tools/iers-oracle/`.

// `FCULa` is the model's own name, not an item in this crate, and reads as prose everywhere it
// appears.
#![allow(clippy::doc_markdown)]

/// Zenith delay split into the two components the model computes separately, metres.
///
/// The split is not cosmetic: the hydrostatic part is predictable to a millimetre from surface
/// pressure alone, while the non-hydrostatic part carries almost all of the residual uncertainty
/// from a badly-known water vapour profile — so an error budget wants them apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZenithDelay {
    /// Hydrostatic (dry) zenith delay, metres. About 2.3 m at sea level.
    pub hydrostatic: f64,
    /// Non-hydrostatic (wet) zenith delay, metres. Millimetres at optical wavelengths — an order of
    /// magnitude smaller than the microwave wet delay, which is why SLR is the more accurate
    /// technique in a humid atmosphere.
    pub non_hydrostatic: f64,
}

impl ZenithDelay {
    /// The zenith total delay, metres.
    #[must_use]
    pub const fn total(&self) -> f64 {
        self.hydrostatic + self.non_hydrostatic
    }
}

/// Partial pressure of water vapour, hPa, from the surface observation an SLR station logs.
///
/// IERS Conventions (2010) chapter 9: the saturation pressure over water, times the enhancement
/// factor that accounts for the vapour being mixed with air rather than alone, times the relative
/// humidity. `temperature` is in kelvin, `pressure` and the result in hPa, `relative_humidity` in
/// percent.
#[must_use]
pub fn water_vapour_pressure(temperature: f64, pressure: f64, relative_humidity: f64) -> f64 {
    let saturation = 0.01
        * libm::exp(
            1.237_884_7e-5 * temperature * temperature - 1.912_131_6e-2 * temperature
                + 33.937_110_47
                - 6.343_164_5e3 / temperature,
        );
    let celsius = temperature - 273.15;
    let enhancement = 1.000_62 + 3.14e-6 * pressure + 5.6e-7 * celsius * celsius;
    0.01 * relative_humidity * enhancement * saturation
}

/// Mendes–Pavlis zenith delay, metres, at the laser wavelength.
///
/// `latitude` in radians (geodetic), `height` the ellipsoidal height in metres, `pressure` the
/// total surface pressure and `water_vapour` its water-vapour partial pressure in hPa,
/// `wavelength` in micrometres — 0.532 for a frequency-doubled Nd:YAG, which is most of the
/// network.
///
/// The dispersion equations are fitted over 0.355–1.064 µm; outside that the extrapolation is the
/// caller's problem, and the function does not pretend otherwise by clamping.
#[must_use]
pub fn mendes_pavlis_zenith_delay(
    latitude: f64,
    height: f64,
    pressure: f64,
    water_vapour: f64,
    wavelength: f64,
) -> ZenithDelay {
    // Dispersion-equation constants, Mendes & Pavlis (2004) equations (20) and (32). k1 and k3 are
    // the starred forms.
    const K0: f64 = 238.018_5;
    const K1: f64 = 19_990.975;
    const K2: f64 = 57.362;
    const K3: f64 = 579.551_74;
    const W0: f64 = 295.235;
    const W1: f64 = 2.642_2;
    const W2: f64 = -0.032_380;
    const W3: f64 = 0.004_028;
    /// Atmospheric CO₂ content the dispersion formula is referred to, ppm.
    const CO2_PPM: f64 = 375.0;

    let sigma2 = 1.0 / (wavelength * wavelength);
    let sigma4 = sigma2 * sigma2;
    let sigma6 = sigma4 * sigma2;

    // Equation (24): gravity at the site relative to its value at the reference latitude.
    let f = 1.0 - 0.002_66 * libm::cos(2.0 * latitude) - 0.000_000_28 * height;
    let co2 = 1.0 + 0.534e-6 * (CO2_PPM - 450.0);

    let fh = 0.01
        * co2
        * ((K1 * (K0 + sigma2)) / ((K0 - sigma2) * (K0 - sigma2))
            + K3 * (K2 + sigma2) / ((K2 - sigma2) * (K2 - sigma2)));
    let fnh = 0.003_101 * (W0 + 3.0 * W1 * sigma2 + 5.0 * W2 * sigma4 + 7.0 * W3 * sigma6);

    ZenithDelay {
        hydrostatic: 2.416_579e-3 * fh * pressure / f,
        non_hydrostatic: 1e-4 * (5.316 * fnh - 3.759 * fh) * water_vapour / f,
    }
}

/// The FCULa mapping function — the factor the zenith delay is multiplied by at `elevation`.
///
/// `latitude` and `elevation` in radians, `height` in metres, `temperature` in kelvin. One function
/// for both delay components: FCULa is deliberately not split hydrostatic/wet, because at optical
/// wavelengths the wet part is millimetres and a separate wet mapping buys nothing.
///
/// It grows without bound as the elevation goes to zero, as every Marini fraction does. SLR
/// stations do not track below about 10°, and neither the fit nor this function is meaningful
/// there.
#[must_use]
pub fn fcul_a_mapping(latitude: f64, height: f64, temperature: f64, elevation: f64) -> f64 {
    const A10: f64 = 0.121_008e-2;
    const A11: f64 = 0.172_95e-5;
    const A12: f64 = 0.319_1e-4;
    const A13: f64 = -0.184_78e-7;
    const A20: f64 = 0.304_965e-2;
    const A21: f64 = 0.234_6e-5;
    const A22: f64 = -0.103_5e-3;
    const A23: f64 = -0.185_6e-7;
    const A30: f64 = 0.687_77e-1;
    const A31: f64 = 0.197_2e-4;
    const A32: f64 = -0.345_8e-2;
    const A33: f64 = 0.106_0e-6;

    let celsius = temperature - 273.15;
    let cos_phi = libm::cos(latitude);
    let a1 = A10 + A11 * celsius + A12 * cos_phi + A13 * height;
    let a2 = A20 + A21 * celsius + A22 * cos_phi + A23 * height;
    let a3 = A30 + A31 * celsius + A32 * cos_phi + A33 * height;

    let sine = libm::sin(elevation);
    let numerator = 1.0 + a1 / (1.0 + a2 / (1.0 + a3));
    numerator / (sine + a1 / (sine + a2 / (sine + a3)))
}

/// One-way tropospheric path delay, metres, at `elevation` — the zenith delay mapped down.
///
/// The two-way measurement carries twice this.
#[must_use]
pub fn refraction_delay(
    latitude: f64,
    height: f64,
    pressure: f64,
    temperature: f64,
    relative_humidity: f64,
    wavelength: f64,
    elevation: f64,
) -> f64 {
    let vapour = water_vapour_pressure(temperature, pressure, relative_humidity);
    let zenith = mendes_pavlis_zenith_delay(latitude, height, pressure, vapour, wavelength);
    zenith.total() * fcul_a_mapping(latitude, height, temperature, elevation)
}

#[cfg(test)]
mod tests {
    // The reference routine publishes its test values to nineteen digits; they are kept whole so a
    // future comparison against a higher-precision build has something to compare to.
    #![allow(clippy::excessive_precision)]

    use core::f64::consts::PI;

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    /// Degrees to radians, for transcribing the reference routines' degree-valued test cases.
    const fn radians(degrees: f64) -> f64 {
        degrees * (PI / 180.0)
    }

    /// The IERS routine `FCULZD_HPA` at its own published test point: McDonald Observatory,
    /// 2009 August 14, at the 532 nm second harmonic.
    ///
    /// **The expected values here are the compiled routine's, not the ones printed in its header
    /// comment, because the two disagree.** Built with `gfortran -fdefault-real-8`, `FCULZD_HPA`
    /// returns a zenith hydrostatic delay of `1.932_995_972_236_289_6` m; its header says
    /// `1.932_992_176_591_644_462` m. That is 3.8 µm, two parts in 10⁶, and it is a constant
    /// *ratio* across all three outputs — so it is the `f` gravity factor that divides them all,
    /// and it is a stale value in the IERS distribution's documentation rather than anything a
    /// caller can act on. The build flag is not the cause: a default `gfortran` build differs from
    /// the double one by 1.6e-12 m, six orders below the gap. `FCUL_A`'s published value, by
    /// contrast, matches its own code exactly. This port reproduces what the routine computes, to
    /// one ULP; `tools/iers-oracle/` rebuilds the Fortran and prints both numbers.
    #[test]
    fn the_zenith_delay_reproduces_the_iers_reference_routine() {
        let z =
            mendes_pavlis_zenith_delay(radians(30.671_666_67), 2010.344, 798.4188, 14.322, 0.532);
        assert_within_ulps(
            z.hydrostatic,
            1.932_995_972_236_289_6,
            4,
            "zenith hydrostatic delay",
        );
        assert_within_ulps(
            z.non_hydrostatic,
            0.223_375_273_168_358_32e-2,
            4,
            "zenith non-hydrostatic delay",
        );
        assert_within_ulps(z.total(), 1.935_229_724_967_973_2, 4, "zenith total delay");
    }

    /// (G5) The test case published in the header of the IERS routine `FCUL_A.F`, IERS
    /// Conventions (2010) chapter 9 software: same site, 2009 August 12, 15° elevation.
    #[test]
    fn the_mapping_function_reproduces_the_iers_reference_routine() {
        let m = fcul_a_mapping(radians(30.671_666_67), 2075.0, 300.15, radians(15.0));
        assert_within_ulps(m, 3.800_243_667_312_344_087, 4, "m");
    }

    /// At zenith the continued fraction has to collapse to its own numerator, exactly — the
    /// property that makes the zenith delay a zenith delay.
    #[test]
    fn the_mapping_function_is_one_at_the_zenith() {
        for &(lat, h, t) in &[
            (0.0, 0.0, 300.0),
            (radians(45.0), 1000.0, 280.0),
            (radians(-78.0), 30.0, 240.0),
        ] {
            assert_within_ulps(
                fcul_a_mapping(lat, h, t, PI / 2.0),
                1.0,
                2,
                "fcul_a_mapping(lat",
            );
        }
    }

    /// Refraction grows monotonically as the beam is pushed towards the horizon. A sign slip in any
    /// of the nine `a` coefficients breaks this long before it breaks the single reference point.
    #[test]
    fn the_mapping_function_grows_monotonically_towards_the_horizon() {
        let lat = radians(52.0);
        let mut previous = f64::INFINITY;
        for step in 1..=16 {
            let elevation = radians(f64::from(step) * 5.0);
            let m = fcul_a_mapping(lat, 100.0, 285.0, elevation);
            assert!(
                m < previous,
                "mapping is not monotone between {}° and the step below",
                f64::from(step) * 5.0
            );
            previous = m;
        }
        assert!(previous >= 1.0);
    }

    /// The absolute value, against an independent 60-digit evaluation of the same IERS chapter 9
    /// relations in `tools/gen_slr_vapour_oracle.mac`.
    ///
    /// The linearity check below is a relation between two of this function's own outputs, and
    /// every operator inside the saturation exponent and the enhancement factor is invisible to it:
    /// scale both halves and the ratio is unchanged. These two points fix the scale. They sit on
    /// opposite sides of the 273.15 K offset so the `celsius` term cannot be dropped, and at
    /// different pressures so the enhancement term cannot either.
    ///
    /// The bound is 16 ulps where the rest of this file holds 1, and the four extra bits are the
    /// exponent's own: its four terms are each an order larger than their sum (33.94 and −22.01
    /// against a total near 2.8), so the f64 evaluation gives up about four bits to cancellation
    /// before `exp` amplifies what is left by the argument itself. Worst of the three measured
    /// here is 9 ulps, at ISA sea level.
    #[test]
    fn the_water_vapour_pressure_matches_the_iers_relations_evaluated_in_maxima() {
        assert_within_ulps(
            water_vapour_pressure(288.15, 1013.25, 50.0),
            8.561_841_443_790_938_532_574_358_232_432_802_188_66,
            16,
            "ISA sea level at 50% humidity",
        );
        assert_within_ulps(
            water_vapour_pressure(253.15, 700.0, 90.0),
            1.133_616_434_589_982_627_473_169_121_533_658_818_59,
            16,
            "a cold station at 700 hPa and 90% humidity",
        );
        // The saturation pressure alone, so a failure localises to one of the two factors: at
        // RH = 100% and the enhancement factor's own reference the product is saturation × f.
        assert_within_ulps(
            water_vapour_pressure(288.15, 1013.25, 100.0)
                / (1.000_62 + 3.14e-6 * 1013.25 + 5.6e-7 * 15.0 * 15.0),
            1.705_669_094_295_088_844_095_358_495_980_953_166_17e1,
            16,
            "the saturation pressure over water at 288.15 K",
        );
    }

    /// Water vapour pressure at saturation must equal the enhanced saturation pressure, and zero
    /// humidity must give zero — the two ends the interpolation is anchored on.
    #[test]
    fn the_water_vapour_pressure_is_linear_in_relative_humidity() {
        // Saturation over water at 15 °C is about 17.0 hPa; the enhancement factor adds 0.4%.
        let full = water_vapour_pressure(288.15, 1013.25, 100.0);
        assert!(
            (16.5..17.5).contains(&full),
            "{full} hPa is not plausible at 15 C"
        );
        assert_within_ulps(
            water_vapour_pressure(288.15, 1013.25, 0.0),
            0.0,
            0,
            "water_vapour_pressure(288.15",
        );
        assert_within_ulps(
            water_vapour_pressure(288.15, 1013.25, 50.0),
            full * 0.5,
            2,
            "water_vapour_pressure(288.15",
        );
    }

    /// The zenith delay is dispersive: a longer wavelength refracts less. The 1064 nm fundamental
    /// must come out below the 532 nm harmonic at the same weather.
    #[test]
    fn the_zenith_delay_is_dispersive() {
        let green = mendes_pavlis_zenith_delay(radians(45.0), 100.0, 1013.25, 10.0, 0.532);
        let infrared = mendes_pavlis_zenith_delay(radians(45.0), 100.0, 1013.25, 10.0, 1.064);
        assert!(
            infrared.hydrostatic < green.hydrostatic,
            "{} is not below {}",
            infrared.hydrostatic,
            green.hydrostatic
        );
        // The refractivity of air falls about 4.5% between the second harmonic and the
        // fundamental, so the two colours of the same laser differ by roughly 11 cm at zenith.
        // This is not a small effect that could be folded into a bias — it is why the zenith model
        // takes a wavelength at all.
        let spread = green.hydrostatic - infrared.hydrostatic;
        assert!(
            (0.08..0.14).contains(&spread),
            "dispersion of {spread} m between 532 and 1064 nm is out of family"
        );
    }

    /// The composed one-way delay is the product of its two published parts, and doubles into the
    /// two-way measurement. This pins the composition, which is where a factor of two hides.
    #[test]
    fn the_composed_delay_is_the_zenith_delay_times_the_mapping() {
        let (lat, h, p, t, rh, wl, el) = (
            radians(36.0),
            1740.0,
            810.0,
            288.0,
            42.0,
            0.532,
            radians(24.0),
        );
        let vapour = water_vapour_pressure(t, p, rh);
        let expected = mendes_pavlis_zenith_delay(lat, h, p, vapour, wl).total()
            * fcul_a_mapping(lat, h, t, el);
        assert_within_ulps(
            refraction_delay(lat, h, p, t, rh, wl, el),
            expected,
            0,
            "refraction_delay(lat",
        );
    }
}
