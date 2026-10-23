//! Dual-frequency **linear combinations** of GNSS observables, and the carrier frequencies they need.
//!
//! Each combination exists to cancel something. Writing the observation equations for a code
//! observable `P` and a phase observable `L` (both in metres) on frequency `f`:
//!
//! ```text
//! P = ρ + I/f² + ε_P            L = ρ − I/f² + λ·N + ε_L
//! ```
//!
//! where `ρ` is everything non-dispersive (geometry, clocks, troposphere), `I/f²` the first-order
//! ionospheric delay — opposite in sign for code and phase — and `λ·N` the phase ambiguity. From
//! that pair:
//!
//! * **Geometry-free** `L1 − L2` removes `ρ` outright, leaving ionosphere and ambiguity. It is the
//!   ionospheric observable and the primary cycle-slip statistic ([`crate::cycle_slip`]).
//! * **Ionosphere-free** `(f1²·X1 − f2²·X2)/(f1² − f2²)` removes `I/f²` and keeps `ρ`, at the cost of
//!   amplifying noise roughly threefold on the GPS L1/L2 pair.
//! * **Wide-lane** `(f1·L1 − f2·L2)/(f1 − f2)` keeps `ρ` with an 86 cm wavelength on GPS L1/L2 — a
//!   long enough wavelength for the ambiguity to be resolvable from noisy data.
//! * **Narrow-lane** `(f1·X1 + f2·X2)/(f1 + f2)` keeps `ρ` and *halves* the noise, at a 10.7 cm
//!   wavelength.
//! * **Melbourne–Wübbena** is the wide-lane phase minus the narrow-lane code. Geometry, clocks,
//!   troposphere and the ionosphere all cancel — the ionosphere because both combinations carry the
//!   *same* residual term `+I/(f1·f2)`, the phase's `−I/f²` having changed sign under the wide-lane
//!   difference — leaving the wide-lane ambiguity plus noise. It is the standard wide-lane ambiguity
//!   and slip statistic.
//! * **GRAPHIC** `(P + L)/2` is the single-frequency analogue: the ionosphere cancels between code
//!   and phase, leaving half the ambiguity, at half the code noise.
//!
//! Every function here takes and returns **metres**. RINEX reports carrier phase in cycles
//! ([`crate::rinex_obs`]), so a phase observable needs [`FrequencyPair::wavelength1`] /
//! [`FrequencyPair::wavelength2`] applied before it arrives.

// The doc allow is for spec proper nouns (RINEX, GLONASS, BeiDou, QZSS, NavIC, GRAPHIC).
#![allow(clippy::doc_markdown)]

/// Speed of light in vacuum, m/s — exact by the SI definition of the metre.
const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// The GNSS fundamental frequency, Hz.
///
/// Every CDMA GNSS carrier is an exact integer or half-integer multiple of 10.23 MHz, which is why
/// the frequencies below are written as that multiple rather than as a rounded MHz figure: the
/// products `154 × 10.23e6` and `1575.42e6` differ in the last bits, and a wide-lane wavelength
/// divides by the difference of two of them.
const FUNDAMENTAL_HZ: f64 = 10.23e6;

/// GPS L1 / Galileo E1 / QZSS L1 / BeiDou B1C carrier, Hz (`154 × 10.23 MHz`).
pub const L1_HZ: f64 = 154.0 * FUNDAMENTAL_HZ;
/// GPS L2 / QZSS L2 carrier, Hz (`120 × 10.23 MHz`).
pub const L2_HZ: f64 = 120.0 * FUNDAMENTAL_HZ;
/// GPS L5 / Galileo E5a / QZSS L5 / BeiDou B2a / NavIC L5 carrier, Hz (`115 × 10.23 MHz`).
pub const L5_HZ: f64 = 115.0 * FUNDAMENTAL_HZ;
/// Galileo E6 / QZSS L6 carrier, Hz (`125 × 10.23 MHz`).
pub const E6_HZ: f64 = 125.0 * FUNDAMENTAL_HZ;
/// Galileo E5b / BeiDou B2I / BeiDou B2b carrier, Hz (`118 × 10.23 MHz`).
pub const E5B_HZ: f64 = 118.0 * FUNDAMENTAL_HZ;
/// Galileo E5 AltBOC (E5a+E5b) centre carrier, Hz (`116.5 × 10.23 MHz`).
pub const E5_HZ: f64 = 116.5 * FUNDAMENTAL_HZ;
/// BeiDou B1I carrier, Hz (`152.6 × 10.23 MHz`).
pub const B1I_HZ: f64 = 152.6 * FUNDAMENTAL_HZ;
/// BeiDou B3I carrier, Hz (`124 × 10.23 MHz`).
pub const B3I_HZ: f64 = 124.0 * FUNDAMENTAL_HZ;
/// NavIC S-band carrier, Hz (`243.6 × 10.23 MHz`).
pub const NAVIC_S_HZ: f64 = 243.6 * FUNDAMENTAL_HZ;
/// GLONASS G3 CDMA carrier, Hz (`117.5 × 10.23 MHz`) — the one GLONASS band that is not FDMA.
pub const GLONASS_G3_HZ: f64 = 117.5 * FUNDAMENTAL_HZ;
/// GLONASS G1a CDMA carrier, Hz (`156.5 × 10.23 MHz`).
pub const GLONASS_G1A_HZ: f64 = 156.5 * FUNDAMENTAL_HZ;
/// GLONASS G2a CDMA carrier, Hz (`122 × 10.23 MHz`).
pub const GLONASS_G2A_HZ: f64 = 122.0 * FUNDAMENTAL_HZ;

/// The carrier frequency (Hz) of a RINEX 3 constellation letter and band character — the pair an
/// [`crate::rinex_obs::ObsCode`] carries.
///
/// `None` where the pair does not name one frequency:
///
/// * GLONASS bands `1` and `2` are FDMA, so the frequency is a property of the *satellite*, not of
///   the band — [`crate::glonass_l1_frequency`] takes the channel number the navigation message
///   carries.
/// * Any band a constellation does not broadcast.
///
/// BeiDou follows RINEX 3.04 and later, where band `2` is B1I and band `1` is B1C; files written
/// against RINEX 3.02 put B1I on band `1` and would be read here as B1C, a 14.322 MHz error.
#[must_use]
pub fn carrier_frequency(system: char, band: char) -> Option<f64> {
    // Grouped by carrier rather than by constellation: the bands that share a frequency are the
    // reason interoperable receivers exist, and listing them together says so.
    Some(match (system, band) {
        ('G' | 'J' | 'E' | 'C' | 'S', '1') => L1_HZ,
        ('G' | 'J', '2') => L2_HZ,
        ('G' | 'J' | 'E' | 'C' | 'I' | 'S', '5') => L5_HZ,
        ('J' | 'E', '6') => E6_HZ,
        ('E' | 'C', '7') => E5B_HZ,
        ('E' | 'C', '8') => E5_HZ,
        ('C', '2') => B1I_HZ,
        ('C', '6') => B3I_HZ,
        ('I', '9') => NAVIC_S_HZ,
        ('R', '3') => GLONASS_G3_HZ,
        ('R', '4') => GLONASS_G1A_HZ,
        ('R', '6') => GLONASS_G2A_HZ,
        _ => return None,
    })
}

/// A refusal to build a [`FrequencyPair`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CombinationError {
    /// A frequency was not finite and positive.
    NonPositiveFrequency,
    /// The frequencies were not given high first, or were equal.
    ///
    /// Ordering is an invariant rather than a convention: the wide-lane wavelength is
    /// `c/(f1 − f2)`, so a swapped pair silently returns a negative wavelength and a
    /// sign-flipped ambiguity, and an equal pair divides by zero.
    NotDescending,
}

impl core::fmt::Display for CombinationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveFrequency => {
                write!(f, "a carrier frequency is not finite and positive")
            }
            Self::NotDescending => write!(f, "the carrier frequencies are not strictly descending"),
        }
    }
}

/// Two distinct carrier frequencies, high first — the input every dual-frequency combination needs.
///
/// The type exists to hold `f1 > f2 > 0` once, at construction, rather than to re-check it in each
/// of the six combinations below.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencyPair {
    f1: f64,
    f2: f64,
}

impl FrequencyPair {
    /// The GPS L1/L2 pair — the combinations' reference case.
    pub const GPS_L1_L2: Self = Self {
        f1: L1_HZ,
        f2: L2_HZ,
    };
    /// The GPS L1/L5 pair.
    pub const GPS_L1_L5: Self = Self {
        f1: L1_HZ,
        f2: L5_HZ,
    };
    /// The Galileo E1/E5a pair.
    pub const GALILEO_E1_E5A: Self = Self {
        f1: L1_HZ,
        f2: L5_HZ,
    };

    /// A pair from two frequencies in Hz.
    ///
    /// # Errors
    /// [`CombinationError::NonPositiveFrequency`] if either is not finite and positive, and
    /// [`CombinationError::NotDescending`] if they are not given `f1 > f2`.
    pub fn new(f1: f64, f2: f64) -> Result<Self, CombinationError> {
        if !(f1.is_finite() && f2.is_finite() && f1 > 0.0 && f2 > 0.0) {
            return Err(CombinationError::NonPositiveFrequency);
        }
        if f1 <= f2 {
            return Err(CombinationError::NotDescending);
        }
        Ok(Self { f1, f2 })
    }

    /// The higher carrier frequency, Hz.
    #[must_use]
    pub const fn f1(self) -> f64 {
        self.f1
    }

    /// The lower carrier frequency, Hz.
    #[must_use]
    pub const fn f2(self) -> f64 {
        self.f2
    }

    /// Wavelength of the higher carrier, metres.
    #[must_use]
    pub fn wavelength1(self) -> f64 {
        SPEED_OF_LIGHT / self.f1
    }

    /// Wavelength of the lower carrier, metres.
    #[must_use]
    pub fn wavelength2(self) -> f64 {
        SPEED_OF_LIGHT / self.f2
    }

    /// Wide-lane wavelength `c/(f1 − f2)`, metres — 0.861 m on GPS L1/L2.
    #[must_use]
    pub fn wide_lane_wavelength(self) -> f64 {
        SPEED_OF_LIGHT / (self.f1 - self.f2)
    }

    /// Narrow-lane wavelength `c/(f1 + f2)`, metres — 0.107 m on GPS L1/L2.
    #[must_use]
    pub fn narrow_lane_wavelength(self) -> f64 {
        SPEED_OF_LIGHT / (self.f1 + self.f2)
    }

    /// Ionosphere-free combination `(f1²·x1 − f2²·x2)/(f1² − f2²)`, metres.
    ///
    /// Applies to a code pair or a phase pair alike; on phases the result carries a non-integer
    /// ambiguity, which is why PPP resolves the wide lane first.
    #[must_use]
    pub fn ionosphere_free(self, x1: f64, x2: f64) -> f64 {
        let (a, b) = (self.f1 * self.f1, self.f2 * self.f2);
        (a * x1 - b * x2) / (a - b)
    }

    /// Wide-lane combination `(f1·x1 − f2·x2)/(f1 − f2)`, metres.
    #[must_use]
    pub fn wide_lane(self, x1: f64, x2: f64) -> f64 {
        (self.f1 * x1 - self.f2 * x2) / (self.f1 - self.f2)
    }

    /// Narrow-lane combination `(f1·x1 + f2·x2)/(f1 + f2)`, metres.
    #[must_use]
    pub fn narrow_lane(self, x1: f64, x2: f64) -> f64 {
        (self.f1 * x1 + self.f2 * x2) / (self.f1 + self.f2)
    }

    /// Melbourne–Wübbena combination, metres: the wide-lane phase minus the narrow-lane code.
    ///
    /// `p1`/`p2` are pseudoranges and `l1`/`l2` carrier phases, all metres. What survives is the
    /// wide-lane ambiguity `λ_WL·N_WL` plus noise and the satellite/receiver wide-lane biases — so
    /// it is constant over an unbroken arc, and a step in it is a wide-lane slip.
    #[must_use]
    pub fn melbourne_wubbena(self, p1: f64, p2: f64, l1: f64, l2: f64) -> f64 {
        self.wide_lane(l1, l2) - self.narrow_lane(p1, p2)
    }

    /// Melbourne–Wübbena in **wide-lane cycles** — [`Self::melbourne_wubbena`] over
    /// [`Self::wide_lane_wavelength`]. Over a clean arc this hovers about an integer.
    #[must_use]
    pub fn melbourne_wubbena_cycles(self, p1: f64, p2: f64, l1: f64, l2: f64) -> f64 {
        self.melbourne_wubbena(p1, p2, l1, l2) / self.wide_lane_wavelength()
    }
}

/// Geometry-free combination `x1 − x2`, metres.
///
/// A free function rather than a [`FrequencyPair`] method because it does not use the frequencies:
/// it is the difference of two observables on any two bands, and that is exactly why everything
/// non-dispersive cancels from it.
#[must_use]
pub fn geometry_free(x1: f64, x2: f64) -> f64 {
    x1 - x2
}

/// GRAPHIC combination `(p + l)/2`, metres — the single-frequency ionosphere-free observable.
///
/// The first-order ionosphere enters code and phase with opposite sign and equal magnitude, so the
/// mean of the two cancels it without a second band. What remains carries half the phase ambiguity
/// and half the code noise.
#[must_use]
pub fn graphic(p: f64, l: f64) -> f64 {
    f64::midpoint(p, l)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Every carrier is its exact multiple of 10.23 MHz, checked against the MHz figure the ICDs
    /// print. The tolerance is 1 Hz on numbers of order 1e9 — enough to catch a wrong multiplier
    /// (the smallest step, 0.5 × 10.23 MHz, is five million times this) and not so tight that it
    /// asserts on the last bit of a decimal literal.
    #[test]
    fn the_carriers_are_their_published_frequencies() {
        for (got, want, name) in [
            (L1_HZ, 1_575.42e6, "L1"),
            (L2_HZ, 1_227.60e6, "L2"),
            (L5_HZ, 1_176.45e6, "L5"),
            (E6_HZ, 1_278.75e6, "E6"),
            (E5B_HZ, 1_207.14e6, "E5b"),
            (E5_HZ, 1_191.795e6, "E5"),
            (B1I_HZ, 1_561.098e6, "B1I"),
            (B3I_HZ, 1_268.52e6, "B3I"),
            (NAVIC_S_HZ, 2_492.028e6, "NavIC S"),
            (GLONASS_G3_HZ, 1_202.025e6, "G3"),
            (GLONASS_G1A_HZ, 1_600.995e6, "G1a"),
            (GLONASS_G2A_HZ, 1_248.06e6, "G2a"),
        ] {
            assert!((got - want).abs() < 1.0, "{name}: {got} vs {want}");
        }
    }

    /// Every `(system, band)` pair the lookup accepts, against the MHz figure the constellation's
    /// own ICD prints, and the pairs it must refuse.
    ///
    /// The accepted list is exhaustive because a band nothing asks for is a band whose arm can be
    /// removed with no assertion moving: the pair then falls through to `None`, and a receiver that
    /// tracked it would lose the observable rather than mis-scale it. Sources, all in MHz:
    /// IS-GPS-200/-705 L1 1575.42, L2 1227.60, L5 1176.45; Galileo OS SIS ICD E1 1575.42,
    /// E6 1278.75, E5a 1176.45, E5b 1207.140, E5 (AltBOC centre) 1191.795; BDS-SIS-ICD B1C 1575.42,
    /// B1I 1561.098, B2a 1176.45, B2b 1207.14, B2 (AltBOC centre) 1191.795, B3I 1268.52;
    /// IS-QZSS-PNT L1/L2/L5 as GPS and L6 1278.75; IRNSS SIS ICD L5 1176.45, S 2492.028;
    /// GLONASS CDMA ICD L1OC 1600.995, L2OC 1248.06, L3OC 1202.025.
    ///
    /// The tolerance is 1 Hz on numbers of order 1e9, as in the constants' own test: the smallest
    /// step between two carriers here is 0.5 × 10.23 MHz.
    #[test]
    fn the_band_lookup_resolves_what_it_can_and_refuses_the_rest() {
        for (system, band, mhz, name) in [
            ('G', '1', 1_575.42, "GPS L1"),
            ('J', '1', 1_575.42, "QZSS L1"),
            ('E', '1', 1_575.42, "Galileo E1"),
            ('C', '1', 1_575.42, "BeiDou B1C"),
            ('S', '1', 1_575.42, "SBAS L1"),
            ('G', '2', 1_227.60, "GPS L2"),
            ('J', '2', 1_227.60, "QZSS L2"),
            ('G', '5', 1_176.45, "GPS L5"),
            ('J', '5', 1_176.45, "QZSS L5"),
            ('E', '5', 1_176.45, "Galileo E5a"),
            ('C', '5', 1_176.45, "BeiDou B2a"),
            ('I', '5', 1_176.45, "NavIC L5"),
            ('S', '5', 1_176.45, "SBAS L5"),
            ('J', '6', 1_278.75, "QZSS L6"),
            ('E', '6', 1_278.75, "Galileo E6"),
            ('E', '7', 1_207.14, "Galileo E5b"),
            ('C', '7', 1_207.14, "BeiDou B2b"),
            ('E', '8', 1_191.795, "Galileo E5"),
            ('C', '8', 1_191.795, "BeiDou B2"),
            ('C', '2', 1_561.098, "BeiDou B1I"),
            ('C', '6', 1_268.52, "BeiDou B3I"),
            ('I', '9', 2_492.028, "NavIC S"),
            ('R', '3', 1_202.025, "GLONASS L3OC"),
            ('R', '4', 1_600.995, "GLONASS L1OC"),
            ('R', '6', 1_248.06, "GLONASS L2OC"),
        ] {
            let got = carrier_frequency(system, band)
                .unwrap_or_else(|| panic!("{name}: {system}{band} has no frequency"));
            assert!(
                (got - mhz * 1e6).abs() < 1.0,
                "{name}: {got} vs {}",
                mhz * 1e6
            );
        }

        // The arms are grouped by carrier, and the grouping is the claim: an interoperable receiver
        // combines these across constellations, so they have to be one number and not three that
        // print alike.
        for (a, b) in [
            (('G', '1'), ('E', '1')),
            (('G', '1'), ('C', '1')),
            (('G', '5'), ('E', '5')),
            (('E', '7'), ('C', '7')),
        ] {
            assert_eq!(
                carrier_frequency(a.0, a.1),
                carrier_frequency(b.0, b.1),
                "{}{} and {}{} share a carrier",
                a.0,
                a.1,
                b.0,
                b.1
            );
        }

        // The BeiDou band numbering this file follows is RINEX 3.04+, and the cost of reading a
        // 3.02-era file with it is exactly the B1C - B1I gap the doc comment names.
        let (b1c, b1i) = (
            carrier_frequency('C', '1').unwrap(),
            carrier_frequency('C', '2').unwrap(),
        );
        assert!((b1c - b1i - 14.322e6).abs() < 1.0, "{}", b1c - b1i);

        for (system, band, why) in [
            // FDMA: the frequency belongs to the satellite, not the band.
            ('R', '1', "GLONASS G1 is FDMA"),
            ('R', '2', "GLONASS G2 is FDMA"),
            // Bands the constellation does not broadcast.
            ('G', '6', "no GPS band 6"),
            ('G', '7', "no GPS band 7"),
            ('G', '8', "no GPS band 8"),
            ('J', '7', "no QZSS band 7"),
            ('E', '2', "no Galileo band 2"),
            ('E', '9', "no Galileo band 9"),
            ('C', '9', "no BeiDou band 9"),
            ('I', '1', "no NavIC band 1"),
            ('S', '2', "no SBAS band 2"),
            ('R', '5', "no GLONASS band 5"),
            ('R', '9', "no GLONASS band 9"),
            // Not a constellation letter, not a band character.
            ('Z', '1', "unknown system"),
            ('g', '1', "the letters are upper case"),
            ('G', 'X', "not a band digit"),
            ('G', ' ', "blank band"),
        ] {
            assert_eq!(
                carrier_frequency(system, band),
                None,
                "{system}{band}: {why}"
            );
        }
    }

    #[test]
    fn a_pair_must_be_positive_and_descending() {
        assert_eq!(
            FrequencyPair::new(L2_HZ, L1_HZ),
            Err(CombinationError::NotDescending)
        );
        assert_eq!(
            FrequencyPair::new(L1_HZ, L1_HZ),
            Err(CombinationError::NotDescending)
        );
        assert_eq!(
            FrequencyPair::new(-1.0, L2_HZ),
            Err(CombinationError::NonPositiveFrequency)
        );
        assert_eq!(
            FrequencyPair::new(f64::NAN, L2_HZ),
            Err(CombinationError::NonPositiveFrequency)
        );
        assert_eq!(
            FrequencyPair::new(L1_HZ, L2_HZ),
            Ok(FrequencyPair::GPS_L1_L2)
        );
        assert_ne!(
            CombinationError::NotDescending.to_string(),
            CombinationError::NonPositiveFrequency.to_string()
        );
    }

    /// The GPS L1/L2 wavelengths, against the figures every GNSS text prints.
    #[test]
    fn the_gps_wavelengths_are_the_published_ones() {
        let p = FrequencyPair::GPS_L1_L2;
        for (got, want, name) in [
            (p.wavelength1(), 0.190_293_672_798, "λ1"),
            (p.wavelength2(), 0.244_210_213_425, "λ2"),
            (p.wide_lane_wavelength(), 0.861_918_400_322, "λ_WL"),
            (p.narrow_lane_wavelength(), 0.106_953_378_142, "λ_NL"),
        ] {
            assert!(
                (got - want).abs() < 1e-12,
                "{name}: {got:.12} vs {want:.12}"
            );
        }
    }

    /// Each combination cancels exactly what it is built to cancel, at one synthetic epoch whose
    /// truth is known term by term.
    ///
    /// The observables are assembled from `ρ`, an ionospheric delay `I/f²` and integer ambiguities,
    /// with no noise, so every assertion below is an identity rather than a tolerance on real data:
    /// a wrong sign or a swapped frequency in any of the six moves the result by metres.
    #[test]
    fn each_combination_cancels_what_it_is_built_to_cancel() {
        let p = FrequencyPair::GPS_L1_L2;
        let (f1, f2) = (p.f1(), p.f2());
        let rho = 22_345_678.901_f64; // geometry + clocks + troposphere, metres
        let iono = 4.0e16_f64; // I in the P = ρ + I/f² convention, m·Hz²
        let (n1, n2) = (-1_234_567.0_f64, -962_120.0_f64); // integer cycles

        let (i1, i2) = (iono / (f1 * f1), iono / (f2 * f2));
        let (p1, p2) = (rho + i1, rho + i2);
        let (l1, l2) = (
            rho - i1 + n1 * p.wavelength1(),
            rho - i2 + n2 * p.wavelength2(),
        );

        // Ionosphere-free: ρ exactly, on the code pair.
        assert!(
            (p.ionosphere_free(p1, p2) - rho).abs() < 1e-6,
            "IF code: {}",
            p.ionosphere_free(p1, p2) - rho
        );

        // Geometry-free: the differential ionosphere and the differential ambiguity, no geometry.
        // The tolerance is the cancellation itself: differencing two 2.2e7 m observables to recover
        // a 1e-2 m ionospheric difference costs an ulp of the *inputs*, about 3.7e-9 m.
        let gf_code = geometry_free(p1, p2);
        assert!((gf_code - (i1 - i2)).abs() < 1e-8, "GF code: {gf_code}");
        let gf_phase = geometry_free(l1, l2);
        let want = (i2 - i1) + n1 * p.wavelength1() - n2 * p.wavelength2();
        assert!(
            (gf_phase - want).abs() < 1e-6,
            "GF phase: {gf_phase} vs {want}"
        );

        // Wide-lane phase: ρ, the wide-lane ambiguity and an ionospheric term.
        let wl = p.wide_lane(l1, l2);
        let n_wl = n1 - n2;
        let want = rho - (f1 * i1 - f2 * i2) / (f1 - f2) + n_wl * p.wide_lane_wavelength();
        assert!((wl - want).abs() < 1e-6, "WL: {wl} vs {want}");

        // Narrow-lane code: ρ plus the ionospheric term the wide-lane phase also carries, which is
        // the whole reason Melbourne-Wübbena is ionosphere-free.
        let nl = p.narrow_lane(p1, p2);
        let want = rho + (f1 * i1 + f2 * i2) / (f1 + f2);
        assert!((nl - want).abs() < 1e-6, "NL: {nl} vs {want}");
        // Both residual terms are I/(f1·f2) — measured here as 2.0683e-2 m — so the *difference*
        // removes them. A sign error in either combination leaves twice that behind.
        let wl_iono = -(f1 * i1 - f2 * i2) / (f1 - f2);
        let nl_iono = (f1 * i1 + f2 * i2) / (f1 + f2);
        let closed_form = iono / (f1 * f2);
        assert!(
            (wl_iono - nl_iono).abs() < 1e-12 && (wl_iono - closed_form).abs() < 1e-12,
            "the MW ionosphere does not cancel: {wl_iono} vs {nl_iono} vs {closed_form}"
        );

        // Melbourne-Wübbena: exactly the wide-lane ambiguity, in metres and in cycles.
        let mw = p.melbourne_wubbena(p1, p2, l1, l2);
        let want = n_wl * p.wide_lane_wavelength();
        assert!((mw - want).abs() < 1e-6, "MW: {mw} vs {want}");
        let cycles = p.melbourne_wubbena_cycles(p1, p2, l1, l2);
        assert!(
            (cycles - n_wl).abs() < 1e-6,
            "MW cycles: {cycles} vs {n_wl}"
        );

        // GRAPHIC: ρ plus half the ambiguity, with the ionosphere gone.
        let g = graphic(p1, l1);
        let want = rho + 0.5 * n1 * p.wavelength1();
        assert!((g - want).abs() < 1e-6, "GRAPHIC: {g} vs {want}");
    }

    /// Hand-computed values at one epoch, from the GPS L1/L2 constants, to five decimal places.
    ///
    /// The inputs are round numbers so the arithmetic can be checked by hand:
    /// `P1 = 20 000 000.000 m`, `P2 = 20 000 002.000 m`, `L1 = 19 999 999.000 m`,
    /// `L2 = 19 999 997.000 m`. With `f1 = 154 f0`, `f2 = 120 f0`:
    ///
    /// * `IF(P) = (154² · P1 − 120² · P2)/(154² − 120²) = P1 − (14400/9316)·(P2 − P1)`
    ///   `= 20 000 000 − 2 × 1.545 727 780 163 = 19 999 996.908 544 440 m`
    /// * `WL(L) = (154·L1 − 120·L2)/34 = L1 + (120/34)·(L1 − L2) = 20 000 006.058 823 529 m`
    /// * `NL(P) = (154·P1 + 120·P2)/274 = P1 + (120/274)·(P2 − P1) = 20 000 000.875 912 409 m`
    /// * `MW = WL(L) − NL(P) = 5.182 911 120 m`
    /// * `GF(P) = P1 − P2 = −2 m`, `GRAPHIC(P1,L1) = 19 999 999.5 m`
    #[test]
    fn the_combinations_match_values_computed_by_hand() {
        let p = FrequencyPair::GPS_L1_L2;
        let (p1, p2) = (20_000_000.0_f64, 20_000_002.0_f64);
        let (l1, l2) = (19_999_999.0_f64, 19_999_997.0_f64);

        for (got, want, name) in [
            (p.ionosphere_free(p1, p2), 19_999_996.908_544_44, "IF"),
            (p.wide_lane(l1, l2), 20_000_006.058_823_53, "WL"),
            (p.narrow_lane(p1, p2), 20_000_000.875_912_41, "NL"),
            (p.melbourne_wubbena(p1, p2, l1, l2), 5.182_911_121, "MW"),
            (geometry_free(p1, p2), -2.0, "GF"),
            (graphic(p1, l1), 19_999_999.5, "GRAPHIC"),
        ] {
            assert!((got - want).abs() < 1e-6, "{name}: {got:.9} vs {want:.9}");
        }

        // And the ratios that produced those hand values are the exact rationals they should be:
        // 154² − 120² = 9316, 154 − 120 = 34, 154 + 120 = 274.
        let (f1, f2) = (p.f1() / FUNDAMENTAL_HZ, p.f2() / FUNDAMENTAL_HZ);
        assert!(((f1 * f1 - f2 * f2) - 9316.0).abs() < 1e-9);
        assert!(((f1 - f2) - 34.0).abs() < 1e-9);
        assert!(((f1 + f2) - 274.0).abs() < 1e-9);
    }
}
