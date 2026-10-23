//! When the sky is dark enough and clear enough to observe through — the atmospheric half of an
//! optical tasking gate.
//!
//! [`crate::detectors`] answers *is the object geometrically above the horizon and in sunlight*.
//! That is not an observation. An optical station also needs the Sun far enough below its own
//! horizon, a line of sight that is not looking through so much atmosphere that the object is
//! extinguished, and — in [`crate::photometry`], which builds on this module — a sky background the
//! object can be seen against. This module carries the first two.
//!
//! # True altitude and apparent altitude are not interchangeable
//!
//! Refraction lifts a body near the horizon by about half a degree, and the published air-mass fits
//! disagree about which of the two altitudes they take. Kasten & Young (1989) is a fit in the
//! *apparent* (refracted) altitude; Young (1994) is a fit in the *true* (geometric) zenith angle,
//! and says so in its title — it exists because the earlier fits took the angle an observer points
//! at rather than the one an ephemeris computes.
//!
//! Feeding one the other's argument is a 20 % error at the horizon: `X = 37.92` against `31.73`.
//! So the two altitudes are separate types here, [`TrueAltitude`] and [`ApparentAltitude`], and the
//! only way between them is [`Refraction`]. Routed correctly, the two fits agree to 6.0e-3 relative
//! at the horizon and 6.2e-4 above 10° elevation — which is the check that pins both.
//!
//! # Sign conventions
//!
//! The detectors follow [`crate::detectors`]: `g > 0` when the gate is *open* (dark enough, far
//! enough from the Moon), `g < 0` when it is shut, and [`crate::find_events`] brackets the crossing.

use spacelab_core::Vec3;
use spacelab_sensors::ElevationMask;

use crate::detectors::ElevationDetector;

/// Rate at which a body's local hour angle advances, radians per second of UT1, for a body whose
/// right ascension is fixed.
///
/// This is the Earth's rotation relative to the equinox, `1.002_737_811_911_354_48` revolutions per
/// UT1 day — the IAU 2000 Earth Rotation Angle rate, `2π(0.779_057_273_264_0 + 1.002_737_811_911_354_48 Tu)`,
/// the same constant `spacelab_fa::cio` evaluates ERA with. It is repeated here rather than taken as
/// a dependency, because one scalar does not justify pulling the CIO tables into an access-window
/// crate.
///
/// In the units these calculations are usually quoted in it is **15.041 067 178 670 °/h**, and that
/// is the number to divide an hour angle by. Dividing by 15 °/h instead — the mean *solar* rate —
/// is the classic rise/set bug: it is right only for a body whose right ascension advances at the
/// Sun's own mean rate of 0.0411 °/h, and wrong by 4 s per hour of hour angle for anything else.
// The rate is kept at the width the IAU published it, not at the width `f64` distinguishes.
#[allow(clippy::excessive_precision)]
pub const SIDEREAL_ROTATION_RATE: f64 =
    1.002_737_811_911_354_48 * 2.0 * core::f64::consts::PI / 86_400.0;

/// Mean solar rate, radians per second: `2π/86400`, exactly 15 °/h.
///
/// The correct divisor for the *Sun's* hour angle, whose right ascension advances with it. See
/// [`SIDEREAL_ROTATION_RATE`] for why the two are not interchangeable.
pub const SOLAR_ROTATION_RATE: f64 = 2.0 * core::f64::consts::PI / 86_400.0;

/// Arcminutes, in radians — the unit the standard altitudes below are published in.
const ARCMIN: f64 = core::f64::consts::PI / (180.0 * 60.0);

/// Reference pressure of the refraction formulae, hPa.
const REFRACTION_REFERENCE_PRESSURE: f64 = 1010.0;

/// Reference temperature of the refraction formulae, kelvin.
const REFRACTION_REFERENCE_TEMPERATURE: f64 = 283.0;

/// True altitude below which [`Refraction`] refuses to answer, radians (−1°).
///
/// Both fits are `1/tan(h + c/(h + d))` with poles at `h = −5.11°` and `h = −4.4°`; between the
/// horizon and those poles they are extrapolations of a fit that was never checked there, and below
/// them they change sign. Nothing observable lives there — a body a degree below the true horizon
/// is behind 40 air masses of atmosphere — so the domain is closed rather than extended.
const REFRACTION_FLOOR: f64 = -core::f64::consts::PI / 180.0;

/// Geometric altitude above the horizon, radians — the angle an ephemeris computes, before the
/// atmosphere bends the ray.
///
/// Consumed by [`airmass_young_1994`] and produced by [`Refraction::geometric`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TrueAltitude(pub f64);

/// Refracted altitude above the horizon, radians — the angle a telescope points at.
///
/// Consumed by [`airmass_kasten_young_1989`] and produced by [`Refraction::apparent`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct ApparentAltitude(pub f64);

impl TrueAltitude {
    /// The matching zenith angle, `π/2 − h`.
    #[must_use]
    pub fn zenith_angle(self) -> f64 {
        core::f64::consts::FRAC_PI_2 - self.0
    }
}

impl ApparentAltitude {
    /// The matching zenith angle, `π/2 − h`.
    #[must_use]
    pub fn zenith_angle(self) -> f64 {
        core::f64::consts::FRAC_PI_2 - self.0
    }
}

/// Atmospheric refraction at a stated pressure and temperature.
///
/// Two different published fits, because the inverse of a refraction formula is not a refraction
/// formula: [`Refraction::apparent`] is Sæmundsson (1986), fitted in the true altitude, and
/// [`Refraction::geometric`] is Bennett (1982), fitted in the apparent altitude. Using either one
/// backwards costs a few arcseconds near the horizon, which is why both are here.
///
/// Both are scaled by `(P/1010 hPa)(283 K/T)`, the density ratio the fits were normalised at.
///
/// # Both fits run high, by a few percent, everywhere
///
/// Measured against a ray integrated through the U.S. Standard Atmosphere in
/// `tests/integrated_airmass.rs`: Sæmundsson is +0.43 % to +4.79 % and Bennett +0.58 % to +3.27 %,
/// with the *minimum* at 2°–3° of altitude and the maximum at 40°–50°. Both are horizon-optimised
/// fits and neither was meant for the zenith — at 90° both return a negative refraction.
///
/// It is the fits' own normalisation rather than a modelling choice on either side: above about 20°
/// altitude refraction is `(n₀ − 1) tan z` and nothing else, and the integration reproduces that to
/// 0.13 % against Ciddor's (1996) dry-air refractivity. In absolute terms it is 2.3″ and 1.4″ at 50°
/// altitude and 47″ and 65″ at the horizon — irrelevant to the air mass, which uses refraction only
/// to convert between arguments, and not irrelevant to astrometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refraction {
    /// Station pressure, hPa.
    pub pressure_hpa: f64,
    /// Station temperature, kelvin.
    pub temperature_k: f64,
}

impl Refraction {
    /// The conditions the two fits are published at: 1010 hPa and 283 K.
    pub const STANDARD: Self = Self {
        pressure_hpa: REFRACTION_REFERENCE_PRESSURE,
        temperature_k: REFRACTION_REFERENCE_TEMPERATURE,
    };

    /// Density scaling of the refraction, `(P/1010)(283/T)`.
    fn density_ratio(self) -> f64 {
        (self.pressure_hpa / REFRACTION_REFERENCE_PRESSURE)
            * (REFRACTION_REFERENCE_TEMPERATURE / self.temperature_k)
    }

    /// True → apparent altitude, Sæmundsson (1986): `R = 1.02 / tan(h + 10.3/(h + 5.11))` arcminutes
    /// with `h` in degrees.
    ///
    /// `None` below `REFRACTION_FLOOR`, where the fit is extrapolating toward its pole.
    #[must_use]
    pub fn apparent(self, h: TrueAltitude) -> Option<ApparentAltitude> {
        if h.0.is_nan() || h.0 < REFRACTION_FLOOR || h.0 > core::f64::consts::FRAC_PI_2 {
            return None;
        }
        let deg = h.0.to_degrees();
        let bend = 1.02 / libm::tan((deg + 10.3 / (deg + 5.11)).to_radians());
        Some(ApparentAltitude(h.0 + bend * ARCMIN * self.density_ratio()))
    }

    /// Apparent → true altitude, Bennett (1982): `R = 1 / tan(h + 7.31/(h + 4.4))` arcminutes with
    /// `h` in degrees.
    ///
    /// `None` below the horizon, where there is no ray to have been bent.
    #[must_use]
    pub fn geometric(self, h: ApparentAltitude) -> Option<TrueAltitude> {
        if h.0.is_nan() || h.0 < 0.0 || h.0 > core::f64::consts::FRAC_PI_2 {
            return None;
        }
        let deg = h.0.to_degrees();
        let bend = 1.0 / libm::tan((deg + 7.31 / (deg + 4.4)).to_radians());
        Some(TrueAltitude(h.0 - bend * ARCMIN * self.density_ratio()))
    }

    /// The [`ElevationMask`] equivalent to an apparent one — the whole atmosphere, applied once to
    /// the threshold instead of to every sample.
    ///
    /// A station's "visible above 10°" is an *apparent* elevation; the mask is a geometric one, and
    /// handing one to the other is worth **0.81–1.18 s** of pass edge at a 10° mask and
    /// **9.3–20.3 s** at the horizon, with the same sign on every edge of every pass
    /// (`spacelab/tests/refraction_in_access.rs`). This is the conversion that closes it, and it is
    /// exact rather than an approximation: `h ↦ h + R(h)` is strictly increasing, so
    /// `apparent(h) ≥ mask` and `h ≥ geometric(mask)` are the same condition and one apparent mask is
    /// exactly one geometric mask. The batch kernel never sees the atmosphere and costs nothing.
    ///
    /// `None` when the apparent mask is outside [`Refraction::geometric`]'s domain, or when the
    /// geometric mask it converts to is not one [`ElevationMask::new`] accepts.
    #[must_use]
    pub fn elevation_mask(self, mask: ApparentAltitude) -> Option<ElevationMask> {
        ElevationMask::new(self.geometric(mask)?.0).ok()
    }
}

/// Relative optical air mass, Kasten & Young (1989), Applied Optics **28**, 4735:
/// `X = 1/[sin h + 0.50572 (h + 6.07995)^−1.6364]` with `h` the **apparent** altitude in degrees.
///
/// `f64::INFINITY` below the apparent horizon: there is no line of sight, and the fit does not
/// diverge there — it turns over and comes back down, so the physical limit has to be supplied. At
/// the apparent horizon itself the fit is in its published range and gives 37.92.
///
/// `X(90°) = 0.999 71`, not 1 — the fit does not constrain the zenith exactly, and that 2.9e-4 is
/// the fit residual, not a coding error.
#[must_use]
pub fn airmass_kasten_young_1989(h: ApparentAltitude) -> f64 {
    if h.0.is_nan() || h.0 < 0.0 {
        return f64::INFINITY;
    }
    let deg = h.0.to_degrees();
    1.0 / (libm::sin(h.0) + 0.50572 * libm::pow(deg + 6.07995, -1.6364))
}

/// Relative optical air mass, Young (1994), Applied Optics **33**, 1108, as a rational function of
/// `c = cos z` with `z` the **true** zenith angle:
/// `X = (1.002432 c² + 0.148386 c + 0.0096467)/(c³ + 0.149864 c² + 0.0102963 c + 0.000303978)`.
///
/// `X(0) = 1.000 000 364`, not 1: the fit does not constrain the zenith exactly either, and 3.6e-7
/// is its residual there against Kasten & Young's 2.9e-4.
///
/// `f64::INFINITY` below the true horizon. The denominator has a real root at `c ≈ −0.063`
/// (`z ≈ 93.6°`), so continuing past `z = 90°` returns a finite number on the way to a pole; the
/// half degree of refracted visibility that closure discards is worth 38 air masses, and no
/// photometry survives it.
#[must_use]
pub fn airmass_young_1994(h: TrueAltitude) -> f64 {
    if h.0.is_nan() || h.0 < 0.0 {
        return f64::INFINITY;
    }
    let c = libm::cos(h.zenith_angle());
    let num = 1.002_432 * c * c + 0.148_386 * c + 0.009_646_7;
    let den = c * c * c + 0.149_864 * c * c + 0.010_296_3 * c + 0.000_303_978;
    num / den
}

/// Plane-parallel air mass, `sec z` — the wrong answer near the horizon, kept because it is the
/// baseline every other fit is quoted against.
///
/// Diverges at `z = 90°` where the true air mass is about 38, and is already 3 % high at 10°
/// elevation. `f64::INFINITY` below the horizon, and at it, where `sec z` itself is infinite.
#[must_use]
pub fn airmass_plane_parallel(h: TrueAltitude) -> f64 {
    if h.0.is_nan() || h.0 <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / libm::cos(h.zenith_angle())
}

/// Which air-mass fit a site uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AirmassModel {
    /// Kasten & Young (1989), evaluated at the apparent altitude — the default for photometry.
    KastenYoung1989,
    /// Young (1994), evaluated at the true zenith angle.
    Young1994,
    /// `sec z`. Present as a baseline; do not use it below 20° elevation.
    PlaneParallel,
}

impl AirmassModel {
    /// Air mass for a body at true altitude `h`, converting to the apparent altitude through
    /// `refraction` for the fits that want it.
    ///
    /// `f64::INFINITY` below the horizon, and below `REFRACTION_FLOOR` where the conversion has no
    /// answer — both mean *no line of sight*, which is the value a detector needs.
    #[must_use]
    pub fn at(self, h: TrueAltitude, refraction: Refraction) -> f64 {
        match self {
            Self::KastenYoung1989 => refraction
                .apparent(h)
                .map_or(f64::INFINITY, airmass_kasten_young_1989),
            Self::Young1994 => airmass_young_1994(h),
            Self::PlaneParallel => airmass_plane_parallel(h),
        }
    }
}

/// Extinction in magnitudes over `airmass` air masses at extinction coefficient `k`, `Δm = k X`.
///
/// `k` is a site and season property — order 0.2 mag/airmass in V at a clear dark site, 0.1 at a
/// high dry one, and several tenths through haze — so it is a parameter, never a constant.
#[must_use]
pub fn extinction(k_mag_per_airmass: f64, airmass: f64) -> f64 {
    k_mag_per_airmass * airmass
}

/// Solar altitude thresholds that name a degree of darkness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Twilight {
    /// −6°: the horizon is still distinct, bright planets visible.
    Civil,
    /// −12°: the horizon at sea is no longer distinct.
    Nautical,
    /// −15°: the working threshold for satellite and deep-sky observing.
    Amateur,
    /// −18°: the Sun contributes nothing measurable to the sky background.
    Astronomical,
}

impl Twilight {
    /// The solar altitude that defines this threshold, radians (always negative).
    #[must_use]
    pub fn solar_altitude(self) -> f64 {
        let deg = match self {
            Self::Civil => -6.0,
            Self::Nautical => -12.0,
            Self::Amateur => -15.0,
            Self::Astronomical => -18.0,
        };
        deg * (core::f64::consts::PI / 180.0)
    }
}

/// The altitude at which a body is *defined* to rise or set. Not zero, and the difference between
/// these is minutes of clock time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StandardAltitude {
    /// −34′: horizontal refraction alone, for a point source. The Astronomical Almanac's value for
    /// stars and planets.
    Point,
    /// −50′: 34′ of refraction plus the Sun's 16′ semidiameter — the upper limb touching the
    /// horizon, which is what "sunrise" means.
    SolarUpperLimb,
    /// −35′: the horizontal refraction adopted by the NOAA solar calculator, applied to the Sun's
    /// *centre*. Differs from [`StandardAltitude::SolarUpperLimb`] by the semidiameter.
    SolarCentre,
    /// `0.7275 π − 34′`, where `π` is the Moon's equatorial horizontal parallax: parallax lowers the
    /// Moon as seen from the surface, and it is large enough to matter.
    Moon {
        /// Equatorial horizontal parallax, radians (about 0.95° at mean distance).
        horizontal_parallax: f64,
    },
}

impl StandardAltitude {
    /// The altitude itself, radians.
    #[must_use]
    pub fn altitude(self) -> f64 {
        match self {
            Self::Point => -34.0 * ARCMIN,
            Self::SolarUpperLimb => -50.0 * ARCMIN,
            Self::SolarCentre => -35.0 * ARCMIN,
            Self::Moon {
                horizontal_parallax,
            } => 0.7275 * horizontal_parallax - 34.0 * ARCMIN,
        }
    }
}

/// Whether a body of fixed declination crosses a given altitude from a given latitude, and if so by
/// how much hour angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiurnalArc {
    /// The body rises and sets. `hour_angle` is the semi-diurnal arc in radians, `(0, π)`: the body
    /// is above the standard altitude for hour angles in `(−H, +H)` about upper transit.
    Rises {
        /// Semi-diurnal arc, radians.
        hour_angle: f64,
    },
    /// Circumpolar above the standard altitude — never sets.
    AlwaysAbove,
    /// Never reaches the standard altitude — never rises.
    AlwaysBelow,
}

impl DiurnalArc {
    /// Duration for which the body is above the standard altitude, seconds, at a stated hour-angle
    /// rate — [`SIDEREAL_ROTATION_RATE`] for a body of fixed right ascension,
    /// [`SOLAR_ROTATION_RATE`] for the Sun.
    ///
    /// `f64::INFINITY` when circumpolar, `0.0` when it never rises.
    #[must_use]
    pub fn duration(self, hour_angle_rate: f64) -> f64 {
        match self {
            Self::Rises { hour_angle } => 2.0 * hour_angle / hour_angle_rate,
            Self::AlwaysAbove => f64::INFINITY,
            Self::AlwaysBelow => 0.0,
        }
    }
}

/// The hour angle at which a body of declination `declination` crosses `altitude`, seen from
/// `latitude` — `cos H = (sin h₀ − sin φ sin δ)/(cos φ cos δ)`. All three angles in radians.
///
/// `altitude` is a bare angle rather than a [`StandardAltitude`] because rise/set, twilight and an
/// elevation mask are the same calculation at three thresholds: pass
/// [`StandardAltitude::altitude`] for the first and [`Twilight::solar_altitude`] for the second.
/// Narrowing the parameter to a type only one of them can produce would put the others outside the
/// API. The `|cos H| > 1` branches are not failures — they are the polar day and polar night, and
/// they are the case a solver that returns a bare `f64` turns into a NaN.
///
/// At a geographic pole, or for a body exactly on the celestial pole seen from anywhere, the
/// quotient is `0/0`: the altitude is then `δ` (or `φ`) for every hour angle, constant, so the body
/// is above or below the threshold for all of them and never crosses. That is a degeneracy of the
/// geometry, not of the formula, and it is answered directly rather than divided into.
#[must_use]
pub fn diurnal_arc(latitude: f64, declination: f64, altitude: f64) -> DiurnalArc {
    let (sin_phi, cos_phi) = (libm::sin(latitude), libm::cos(latitude));
    let (sin_dec, cos_dec) = (libm::sin(declination), libm::cos(declination));

    // The degeneracy is read off the angles, not off `cos φ cos δ`. `libm::cos` never returns
    // exactly zero — it bottoms out at 6.123e-17 on the doubles nearest π/2, so a product of two of
    // them is 3.7e-33 — and a guard spelled `denominator == 0.0` is unreachable for every finite
    // input. What then reaches the quotient at a pole is `0/3.7e-33`, which lands inside `(−1, 1)`
    // and reports a body of constant altitude as *rising*: exactly the case the guard exists for.
    //
    // `float_cmp`: the equality is the point. A caller who means a pole passes `FRAC_PI_2`, and one
    // ulp away is the continuous case the quotient already answers correctly — as is any `|φ| > π/2`,
    // where `cos φ < 0` and the altitude formula stays valid, so a `>=` here would be wrong.
    #[allow(clippy::float_cmp)]
    let on_a_pole = libm::fabs(latitude) == core::f64::consts::FRAC_PI_2
        || libm::fabs(declination) == core::f64::consts::FRAC_PI_2;
    if on_a_pole {
        // Altitude is `sin h = sin φ sin δ`, independent of hour angle.
        return if sin_phi * sin_dec >= libm::sin(altitude) {
            DiurnalArc::AlwaysAbove
        } else {
            DiurnalArc::AlwaysBelow
        };
    }

    let denominator = cos_phi * cos_dec;
    let cos_h = (libm::sin(altitude) - sin_phi * sin_dec) / denominator;
    if cos_h <= -1.0 {
        DiurnalArc::AlwaysAbove
    } else if cos_h >= 1.0 {
        DiurnalArc::AlwaysBelow
    } else {
        DiurnalArc::Rises {
            hour_angle: libm::acos(cos_h),
        }
    }
}

/// The observer's own darkness gate: the Sun below a twilight threshold.
///
/// `g > 0` when the sky is dark enough. It is the negated [`ElevationDetector`] evaluated on the
/// Sun at a negative mask, and is written that way rather than recomputing the elevation, so it
/// inherits the `atan2` conditioning that detector documents.
#[derive(Debug, Clone, Copy)]
pub struct DarknessDetector {
    /// The Sun's elevation against the twilight threshold.
    pub sun_elevation: ElevationDetector,
}

impl DarknessDetector {
    /// A gate at `twilight` for a station at `station` with local vertical `up`, both in the same
    /// body-fixed frame the Sun position will be given in.
    #[must_use]
    pub fn new(station: Vec3, up: Vec3, twilight: Twilight) -> Self {
        Self {
            sun_elevation: ElevationDetector::new(station, up, twilight.solar_altitude()),
        }
    }

    /// `g > 0` when the Sun is below the threshold.
    #[must_use]
    pub fn g(&self, sun: Vec3) -> f64 {
        -self.sun_elevation.g(sun)
    }
}

/// A minimum angular separation between the target and the Moon, as seen from the observer.
///
/// `g > 0` when the target is far enough from the Moon. The separation gate is coarse — the
/// scattered-light model in [`crate::photometry`] is the quantitative version — but it is what an
/// observing plan is actually written with.
#[derive(Debug, Clone, Copy)]
pub struct MoonSeparationDetector {
    /// Observer position, metres, in the frame the target and Moon are given in.
    pub observer: Vec3,
    /// Minimum separation, radians.
    pub min_separation: f64,
}

impl MoonSeparationDetector {
    /// A gate at `min_separation` radians.
    #[must_use]
    pub const fn new(observer: Vec3, min_separation: f64) -> Self {
        Self {
            observer,
            min_separation,
        }
    }

    /// `g = separation − min_separation`, positive when the target is clear of the Moon.
    #[must_use]
    pub fn g(&self, target: Vec3, moon: Vec3) -> f64 {
        let a = target - self.observer;
        let b = moon - self.observer;
        libm::atan2(a.cross(b).norm(), a.dot(b)) - self.min_separation
    }
}

#[cfg(test)]
mod tests {
    // `expect` on the refraction conversions: the arguments are literals inside the stated domain.
    // `float_cmp`: the durations of the two polar cases are exactly `INFINITY` and `0.0`, by
    // construction rather than by arithmetic, and comparing them to anything else would weaken the
    // assertion.
    #![allow(clippy::expect_used, clippy::float_cmp)]
    use super::*;

    const DEG: f64 = core::f64::consts::PI / 180.0;

    fn t(deg: f64) -> TrueAltitude {
        TrueAltitude(deg * DEG)
    }

    fn a(deg: f64) -> ApparentAltitude {
        ApparentAltitude(deg * DEG)
    }

    /// The property the mask conversion rests on: `h ↦ h + R(h)` is strictly increasing, so
    /// `apparent(h) ≥ mask` and `h ≥ geometric(mask)` are the same condition and one apparent mask is
    /// exactly one geometric mask. If it were not monotone, a single apparent threshold would
    /// correspond to an *interval* of geometric ones and shifting the mask would be an approximation
    /// rather than an identity — which is what `Refraction::elevation_mask` claims it is not.
    ///
    /// Swept at 0.001° from the fit's floor at −1° to the zenith, 91 001 samples, because
    /// monotonicity is a claim about every point and the interesting region is the arcminute-wide one
    /// just above the horizon where the bend changes fastest — `R` falls by 0.164° over the first
    /// degree while `h` rises by one, and it is that race the property is about.
    ///
    /// The step is formed as `i/1000 − 1` rather than by accumulating `0.001`, so the last sample is
    /// exactly 90° rather than an ulp past `FRAC_PI_2`, where `apparent` closes its domain. The count
    /// is asserted, so a sample silently falling outside the domain fails here rather than quietly
    /// shrinking the sweep this claim rests on.
    #[test]
    fn the_true_to_apparent_map_is_strictly_increasing_over_its_whole_domain() {
        let r = Refraction::STANDARD;
        let mut previous = f64::NEG_INFINITY;
        let mut steps = 0_u32;
        for i in 0..=91_000 {
            let deg = f64::from(i) / 1000.0 - 1.0;
            let apparent = r.apparent(t(deg)).expect("inside the fit's domain");
            assert!(
                apparent.0 > previous,
                "h ↦ h + R(h) is not increasing at {deg}°: {} against the {previous} below it",
                apparent.0
            );
            previous = apparent.0;
            steps += 1;
        }
        assert_eq!(
            steps, 91_001,
            "the domain sampled is not the domain documented"
        );
    }

    /// The conversion is the Bennett bend applied to the threshold and nothing else — asserted
    /// against `geometric` directly rather than against a recomputed formula, because the claim is
    /// that the mask *is* that value and not that it is close to it.
    #[test]
    fn an_apparent_mask_converts_to_exactly_the_geometric_one() {
        let r = Refraction::STANDARD;
        for deg in [0.0_f64, 1.0, 5.0, 10.0, 30.0, 80.0] {
            let mask = r.elevation_mask(a(deg)).expect("inside the domain");
            let geometric = r.geometric(a(deg)).expect("above the horizon");
            assert_eq!(
                mask.sin_min(),
                libm::sin(geometric.0),
                "the {deg}° apparent mask is not the geometric one"
            );
            // And it is *below* the apparent one, by the bend: the atmosphere lifts the satellite,
            // so a geometric mask that reproduces an apparent one has to sit lower.
            assert!(
                geometric.0 < deg * DEG,
                "the geometric mask {} is not below the apparent {deg}°",
                geometric.0
            );
        }
    }

    /// The two ends where there is no mask to build, and they fail for different reasons.
    ///
    /// Below the apparent horizon there is no ray to unbend, so `geometric` refuses. At the zenith
    /// the refusal comes from further along: both published fits return a *negative* refraction at
    /// 90° — documented on [`Refraction`] and measured against an integrated ray — so the geometric
    /// mask lands 3.9e-7 rad *above* `π/2`, where `ElevationMask::new` stops being invertible. The
    /// conversion returns `None` rather than a mask whose predicate no longer means what it says.
    #[test]
    fn a_mask_outside_the_domain_is_refused_at_both_ends() {
        let r = Refraction::STANDARD;
        assert!(r.elevation_mask(a(-0.001)).is_none(), "below the horizon");
        assert!(
            r.elevation_mask(ApparentAltitude(f64::NAN)).is_none(),
            "NaN"
        );

        let zenith = ApparentAltitude(core::f64::consts::FRAC_PI_2);
        let past = r
            .geometric(zenith)
            .expect("the zenith is in the fit's domain");
        assert!(
            past.0 > core::f64::consts::FRAC_PI_2,
            "the zenith bend is not negative, so this boundary is no longer the one being tested"
        );
        assert!(r.elevation_mask(zenith).is_none(), "at the zenith");
    }

    #[test]
    fn the_two_air_mass_fits_agree_once_each_is_given_the_altitude_it_was_fitted_in() {
        // The check that pins both fits *and* the refraction model that joins them. Kasten & Young
        // is a fit in the apparent altitude and Young 1994 a fit in the true zenith angle; routed
        // correctly they are two independent descriptions of the same physical quantity, and they
        // agree to 6.0e-3 relative at the horizon and 6.2e-4 above 10°. A mistyped coefficient in
        // either one, or a sign slip in the refraction, breaks this immediately.
        let r = Refraction::STANDARD;
        for &(elev, tol) in &[
            (60.0, 2.1e-4),
            (30.0, 4.2e-4),
            (20.0, 6.2e-4),
            (10.0, 2.0e-4),
            (5.0, 1.9e-3),
            (1.0, 2.2e-3),
            (0.0, 6.0e-3),
        ] {
            let ky = airmass_kasten_young_1989(r.apparent(t(elev)).expect("above the floor"));
            let y94 = airmass_young_1994(t(elev));
            let rel = ((ky - y94) / y94).abs();
            assert!(
                rel <= tol,
                "elevation {elev}: KY {ky} vs Y94 {y94}, relative {rel} > {tol}"
            );
        }
    }

    #[test]
    fn feeding_kasten_young_the_true_altitude_is_a_twenty_percent_error_at_the_horizon() {
        // The reason the two altitudes are separate types. This is not a hypothetical: it is what
        // happens when the same `elevation` variable is handed to both fits.
        let wrong = airmass_kasten_young_1989(a(0.0));
        let right = airmass_young_1994(t(0.0));
        assert!(
            (wrong - 37.919_608).abs() < 1e-5,
            "KY at apparent horizon: {wrong}"
        );
        assert!(
            (right - 31.734_86).abs() < 1e-4,
            "Y94 at true horizon: {right}"
        );
        assert!((wrong - right) / right > 0.19, "{wrong} vs {right}");
    }

    #[test]
    fn air_mass_is_one_at_the_zenith_and_about_thirty_eight_at_the_horizon() {
        // `sec z` is 1 at the zenith by construction. Neither fit is: each carries its own residual
        // there, and the residuals are the fits' own, not a transcription error — Young's is 700×
        // the tighter, which is the whole reason it exists.
        assert!((airmass_plane_parallel(t(90.0)) - 1.0).abs() < 1e-12);
        let young = airmass_young_1994(t(90.0));
        assert!((young - 1.000_000_364).abs() < 1e-9, "{young}");
        let zenith = airmass_kasten_young_1989(a(90.0));
        assert!((zenith - 0.999_71).abs() < 1e-5, "{zenith}");
        // The published horizon value the fit is quoted for.
        assert!((airmass_kasten_young_1989(a(0.0)) - 37.92).abs() < 0.01);
    }

    #[test]
    fn air_mass_increases_monotonically_all_the_way_down_and_saturates_below_the_horizon() {
        // The property that lets the fits return `INFINITY` below the horizon instead of erroring:
        // the composite `g` of a detector built on them stays monotone through the crossing. Both
        // fits turn over or pole out below zero if evaluated naively, so this is a real guard on a
        // real failure mode, not a tautology.
        let r = Refraction::STANDARD;
        let mut previous = 0.0;
        let mut elev = 90.0;
        while elev > -3.0 {
            for model in [
                AirmassModel::KastenYoung1989,
                AirmassModel::Young1994,
                AirmassModel::PlaneParallel,
            ] {
                let x = model.at(t(elev), r);
                assert!(x > 0.0, "{model:?} at {elev}: {x}");
            }
            let x = AirmassModel::KastenYoung1989.at(t(elev), r);
            assert!(x >= previous, "KY not monotone at {elev}: {x} < {previous}");
            previous = x;
            elev -= 0.25;
        }
        assert!(previous.is_infinite(), "did not saturate: {previous}");
        assert!(AirmassModel::Young1994.at(t(-0.5), r).is_infinite());
        assert!(AirmassModel::PlaneParallel.at(t(-0.5), r).is_infinite());
    }

    #[test]
    fn refraction_lifts_by_the_published_amounts_and_the_two_fits_invert_each_other() {
        let r = Refraction::STANDARD;
        // Sæmundsson at the true horizon: 1.02/tan(10.3/5.11 deg) = 28.98'.
        let lifted = r.apparent(t(0.0)).expect("at the floor's interior");
        assert!(
            (lifted.0 / ARCMIN - 28.982).abs() < 1e-3,
            "{}",
            lifted.0 / ARCMIN
        );
        // Bennett at the apparent horizon: 1/tan(7.31/4.4 deg) = 34.5'.
        let dropped = r.geometric(a(0.0)).expect("at the horizon");
        assert!(
            (dropped.0 / ARCMIN + 34.478).abs() < 1e-3,
            "{}",
            dropped.0 / ARCMIN
        );
        // Round-tripping is not exact — two different fits — but it closes to a few arcseconds
        // above the horizon, which is the agreement that says neither is transcribed wrongly.
        for elev in [5.0, 10.0, 30.0, 60.0, 89.0] {
            let there = r.apparent(t(elev)).expect("above the floor");
            let back = r.geometric(there).expect("above the horizon");
            let miss = (back.0 - elev * DEG).abs() / ARCMIN * 60.0;
            assert!(miss < 4.0, "elevation {elev}: round trip misses {miss}\"");
        }
    }

    #[test]
    fn refraction_scales_with_density_and_closes_its_domain() {
        let cold_dense = Refraction {
            pressure_hpa: 1030.0,
            temperature_k: 253.0,
        };
        let standard = Refraction::STANDARD;
        let a_cold = cold_dense.apparent(t(0.0)).expect("valid").0;
        let a_std = standard.apparent(t(0.0)).expect("valid").0;
        let ratio = (1030.0 / 1010.0) * (283.0 / 253.0);
        assert!((a_cold / a_std - ratio).abs() < 1e-12, "{a_cold} {a_std}");
        // Closed below −1°, where the fit is walking toward its pole at −5.11°.
        assert!(standard.apparent(t(-1.001)).is_none());
        assert!(standard.apparent(t(-0.999)).is_some());
        assert!(standard.geometric(a(-0.001)).is_none());
        assert!(standard.apparent(TrueAltitude(f64::NAN)).is_none());
        assert!(standard.geometric(ApparentAltitude(f64::NAN)).is_none());
    }

    #[test]
    fn the_twilight_and_standard_altitudes_are_the_published_ones() {
        assert!((Twilight::Civil.solar_altitude() + 6.0 * DEG).abs() < 1e-15);
        assert!((Twilight::Nautical.solar_altitude() + 12.0 * DEG).abs() < 1e-15);
        assert!((Twilight::Amateur.solar_altitude() + 15.0 * DEG).abs() < 1e-15);
        assert!((Twilight::Astronomical.solar_altitude() + 18.0 * DEG).abs() < 1e-15);
        assert!((StandardAltitude::Point.altitude() / DEG + 0.566_666_7).abs() < 1e-7);
        assert!((StandardAltitude::SolarUpperLimb.altitude() / DEG + 0.833_333_3).abs() < 1e-7);
        assert!((StandardAltitude::SolarCentre.altitude() / DEG + 0.583_333_3).abs() < 1e-7);
        // The upper limb sits one solar semidiameter, 16', below the 34' point-source value.
        let limb = StandardAltitude::SolarUpperLimb.altitude() - StandardAltitude::Point.altitude();
        assert!((limb / ARCMIN + 16.0).abs() < 1e-12, "{}", limb / ARCMIN);
        // The Moon at mean parallax 57.0' sits above the point-source value: 0.7275·57.0' = 41.5'.
        let moon = StandardAltitude::Moon {
            horizontal_parallax: 57.0 * ARCMIN,
        }
        .altitude();
        assert!((moon / ARCMIN - (0.7275 * 57.0 - 34.0)).abs() < 1e-12);
        assert!(moon > 0.0, "the Moon rises above the geometric horizon");
    }

    #[test]
    fn the_diurnal_arc_is_six_hours_of_hour_angle_on_the_equator_and_at_the_equinox() {
        // Latitude 0, declination 0, threshold 0: the body is up for exactly half the day, so the
        // semi-diurnal arc is π/2 and the duration is half a day at whichever rate is asked for.
        let arc = diurnal_arc(0.0, 0.0, 0.0);
        let DiurnalArc::Rises { hour_angle } = arc else {
            panic!("{arc:?}")
        };
        assert!(
            (hour_angle - core::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "{hour_angle}"
        );
        assert!((arc.duration(SOLAR_ROTATION_RATE) - 43_200.0).abs() < 1e-9);
        let sidereal = arc.duration(SIDEREAL_ROTATION_RATE);
        assert!((sidereal - 43_082.0).abs() < 1.0, "{sidereal}");
    }

    #[test]
    fn the_sidereal_rate_is_the_published_fifteen_point_zero_four_one() {
        let deg_per_hour = SIDEREAL_ROTATION_RATE.to_degrees() * 3600.0;
        assert!(
            (deg_per_hour - 15.041_067_178_670).abs() < 1e-11,
            "{deg_per_hour}"
        );
        // The mean solar rate is exactly 15, and the difference over a 6 h hour angle is 24 s —
        // the "off by a few seconds" a rise/set routine written with 15 °/h carries.
        assert!((SOLAR_ROTATION_RATE.to_degrees() * 3600.0 - 15.0).abs() < 1e-13);
        let drift = 6.0 * 3600.0 * (1.0 - SOLAR_ROTATION_RATE / SIDEREAL_ROTATION_RATE);
        assert!((drift - 59.0).abs() < 1.0, "{drift} s over 6 h");
    }

    #[test]
    fn polar_day_and_polar_night_are_answers_rather_than_nans() {
        // Tromsø, 69.65°N, at the solstices. The Sun does not cross the horizon either way, and the
        // ticket's requirement is that the answer is "never", not a NaN out of `acos`.
        let tromso = 69.65 * DEG;
        let limb = StandardAltitude::SolarUpperLimb.altitude();
        let summer = diurnal_arc(tromso, 23.44 * DEG, limb);
        let winter = diurnal_arc(tromso, -23.44 * DEG, limb);
        assert_eq!(summer, DiurnalArc::AlwaysAbove);
        assert_eq!(winter, DiurnalArc::AlwaysBelow);
        assert!(summer.duration(SOLAR_ROTATION_RATE).is_infinite());
        assert_eq!(winter.duration(SOLAR_ROTATION_RATE), 0.0);
        // And astronomical twilight at midsummer never arrives there either — the same solver at a
        // different threshold, which is the whole reason the parameter is a bare angle.
        let dark = diurnal_arc(tromso, 23.44 * DEG, Twilight::Astronomical.solar_altitude());
        assert_eq!(dark, DiurnalArc::AlwaysAbove);
        // Where the Sun does set, nautical twilight is 48 minutes of hour angle past the horizon
        // crossing: at the equator at the equinox the Sun falls 12° in 48 min of a 15 °/h rotation.
        let DiurnalArc::Rises { hour_angle: h12 } =
            diurnal_arc(0.0, 0.0, Twilight::Nautical.solar_altitude())
        else {
            panic!("the Sun rises at the equator")
        };
        let DiurnalArc::Rises { hour_angle: h0 } = diurnal_arc(0.0, 0.0, 0.0) else {
            panic!("the Sun rises at the equator")
        };
        let minutes = (h12 - h0) / SOLAR_ROTATION_RATE / 60.0;
        assert!((minutes - 48.0).abs() < 0.5, "{minutes} minutes");
    }

    #[test]
    fn the_pole_is_answered_by_declination_alone() {
        // The altitude there is δ for every hour angle, so the body never crosses any threshold.
        // In exact arithmetic the published quotient is 0/0; in `f64` it is `n/3.7e-33`, because
        // `cos(FRAC_PI_2)` is 6.123e-17 rather than zero — which is why the degeneracy is decided
        // on the angles and not on the denominator.
        let north = core::f64::consts::FRAC_PI_2;
        let point = StandardAltitude::Point.altitude();
        assert_eq!(
            diurnal_arc(north, 10.0 * DEG, point),
            DiurnalArc::AlwaysAbove
        );
        assert_eq!(
            diurnal_arc(north, -10.0 * DEG, point),
            DiurnalArc::AlwaysBelow
        );
        // Same degeneracy from the other side: a body on the celestial pole, seen from anywhere.
        assert_eq!(
            diurnal_arc(45.0 * DEG, north, point),
            DiurnalArc::AlwaysAbove
        );
        assert_eq!(
            diurnal_arc(-45.0 * DEG, north, point),
            DiurnalArc::AlwaysBelow
        );

        // The comparison is against the threshold, not against zero: from the pole a body at
        // δ = 30° is above a 20° threshold and below a 40° one, and it is the *product* sin φ sin δ
        // that is compared — a sum or a quotient would put both on the same side.
        assert_eq!(
            diurnal_arc(north, 30.0 * DEG, 20.0 * DEG),
            DiurnalArc::AlwaysAbove
        );
        assert_eq!(
            diurnal_arc(north, 30.0 * DEG, 40.0 * DEG),
            DiurnalArc::AlwaysBelow
        );

        // The two cases the quotient answered wrongly before the guard became reachable: with the
        // threshold *exactly* at the body's constant altitude the numerator vanishes, `cos H` lands
        // at 0 rather than outside [−1, 1], and a body that never moves was reported as rising.
        assert_eq!(
            diurnal_arc(north, 30.0 * DEG, 30.0 * DEG),
            DiurnalArc::AlwaysAbove
        );
        assert_eq!(diurnal_arc(north, north, north), DiurnalArc::AlwaysAbove);
    }

    #[test]
    fn the_apparent_zenith_angle_is_the_complement_of_its_altitude() {
        // `TrueAltitude` and `ApparentAltitude` carry the same conversion and neither may drift into
        // the other's units: this is the one that no other assertion in the module reads back.
        for deg in [-1.0_f64, 0.0, 10.0, 45.0, 89.0, 90.0] {
            let h = deg * DEG;
            assert_eq!(
                ApparentAltitude(h).zenith_angle(),
                core::f64::consts::FRAC_PI_2 - h
            );
            assert_eq!(
                TrueAltitude(h).zenith_angle(),
                core::f64::consts::FRAC_PI_2 - h
            );
        }
        assert_eq!(
            ApparentAltitude(0.0).zenith_angle(),
            core::f64::consts::FRAC_PI_2
        );
        assert_eq!(
            ApparentAltitude(core::f64::consts::FRAC_PI_2).zenith_angle(),
            0.0
        );
    }

    #[test]
    fn the_refraction_domains_are_closed_at_their_endpoints_and_not_inside_them() {
        // `refraction_scales_with_density_and_closes_its_domain` samples 1 mrad either side of the
        // floor, which cannot tell `<` from `<=`. These are the endpoints themselves.
        let r = Refraction::STANDARD;
        assert!(r.apparent(TrueAltitude(REFRACTION_FLOOR)).is_some());
        assert!(
            r.apparent(TrueAltitude(f64::from_bits(REFRACTION_FLOOR.to_bits() + 1)))
                .is_none()
        );
        // Bennett is defined up to and including the zenith, and not past it.
        assert!(
            r.geometric(ApparentAltitude(core::f64::consts::FRAC_PI_2))
                .is_some()
        );
        assert!(
            r.geometric(ApparentAltitude(core::f64::consts::FRAC_PI_2 + 1.0 * DEG))
                .is_none()
        );
        assert!(
            r.apparent(TrueAltitude(core::f64::consts::FRAC_PI_2))
                .is_some()
        );
    }

    #[test]
    fn bennett_scales_with_air_density_the_same_way_saemundsson_does() {
        // Only the Sæmundsson branch had its density scaling read back, and at `Refraction::STANDARD`
        // the ratio is exactly 1 — so the whole factor was multiplied by one in every assertion.
        let h = a(20.0);
        let standard = Refraction::STANDARD
            .geometric(h)
            .expect("inside the domain");
        let thin = Refraction {
            pressure_hpa: REFRACTION_REFERENCE_PRESSURE / 2.0,
            temperature_k: REFRACTION_REFERENCE_TEMPERATURE,
        }
        .geometric(h)
        .expect("inside the domain");
        let (std_bend, thin_bend) = (h.0 - standard.0, h.0 - thin.0);
        assert!(
            std_bend > 0.0,
            "refraction must lift, not lower: {std_bend}"
        );
        assert!(
            (thin_bend - 0.5 * std_bend).abs() <= 1e-15 * std_bend,
            "half the pressure must halve the bend: {thin_bend} against {std_bend}"
        );
        // And temperature enters as the reciprocal: twice the temperature is half the density.
        let hot = Refraction {
            pressure_hpa: REFRACTION_REFERENCE_PRESSURE,
            temperature_k: REFRACTION_REFERENCE_TEMPERATURE * 2.0,
        }
        .geometric(h)
        .expect("inside the domain");
        assert!((h.0 - hot.0 - 0.5 * std_bend).abs() <= 1e-15 * std_bend);
    }

    #[test]
    fn the_air_mass_fits_saturate_rather_than_extrapolate_off_their_domains() {
        // `INFINITY` below the horizon and on a `NaN` altitude, for every fit. The `NaN` disjunct is
        // the one no other assertion reaches: `NaN < 0.0` is false, so a fit that tested only the
        // sign would carry the `NaN` all the way into the photometry.
        assert_eq!(
            airmass_kasten_young_1989(ApparentAltitude(f64::NAN)),
            f64::INFINITY
        );
        assert_eq!(airmass_kasten_young_1989(a(-1.0)), f64::INFINITY);
        assert_eq!(airmass_young_1994(TrueAltitude(f64::NAN)), f64::INFINITY);
        assert_eq!(
            airmass_plane_parallel(TrueAltitude(f64::NAN)),
            f64::INFINITY
        );
        // `sec z` is a secant: 2 at 60° from the zenith, not 0.5.
        let sec60 = airmass_plane_parallel(t(30.0));
        assert!((sec60 - 2.0).abs() < 1e-12, "{sec60}");
        let sec45 = airmass_plane_parallel(t(45.0));
        assert!((sec45 - core::f64::consts::SQRT_2).abs() < 1e-12, "{sec45}");
        // And it diverges at the horizon rather than turning over.
        assert_eq!(airmass_plane_parallel(t(0.0)), f64::INFINITY);
    }

    #[test]
    fn the_darkness_gate_opens_when_the_sun_is_below_the_threshold() {
        let station = Vec3::new(6_378_137.0, 0.0, 0.0);
        let up = Vec3::new(1.0, 0.0, 0.0);
        let det = DarknessDetector::new(station, up, Twilight::Astronomical);
        let au = 1.495_978_707e11;
        // Sun 30° above the local horizon: shut.
        let day = station + Vec3::new(au * libm::sin(30.0 * DEG), au * libm::cos(30.0 * DEG), 0.0);
        assert!(det.g(day) < 0.0, "{}", det.g(day));
        // Sun 30° below: open.
        let night =
            station + Vec3::new(-au * libm::sin(30.0 * DEG), au * libm::cos(30.0 * DEG), 0.0);
        assert!(det.g(night) > 0.0, "{}", det.g(night));
        // Exactly at −18°, the gate is on its boundary.
        let edge =
            station + Vec3::new(-au * libm::sin(18.0 * DEG), au * libm::cos(18.0 * DEG), 0.0);
        assert!(det.g(edge).abs() < 1e-9, "{}", det.g(edge));
    }

    #[test]
    fn the_moon_separation_gate_measures_from_the_observer() {
        let observer = Vec3::new(1.0e6, -2.0e6, 5.0e5);
        let det = MoonSeparationDetector::new(observer, 30.0 * DEG);
        let moon = observer + Vec3::new(0.0, 0.0, 3.844e8);
        // 45° away: 15° of margin.
        let far = observer
            + Vec3::new(
                1.0e6 * libm::sin(45.0 * DEG),
                0.0,
                1.0e6 * libm::cos(45.0 * DEG),
            );
        assert!(
            (det.g(far, moon) - 15.0 * DEG).abs() < 1e-9,
            "{}",
            det.g(far, moon)
        );
        // 10° away: shut by 20°.
        let near = observer
            + Vec3::new(
                1.0e6 * libm::sin(10.0 * DEG),
                0.0,
                1.0e6 * libm::cos(10.0 * DEG),
            );
        assert!(
            (det.g(near, moon) + 20.0 * DEG).abs() < 1e-9,
            "{}",
            det.g(near, moon)
        );
    }

    #[test]
    fn extinction_is_linear_in_air_mass() {
        assert!((extinction(0.2, 1.0) - 0.2).abs() < 1e-15);
        assert!((extinction(0.2, 5.0) - 1.0).abs() < 1e-15);
        assert!(extinction(0.2, f64::INFINITY).is_infinite());
    }
}
