//! How bright the object is, how bright the sky it competes with is, and whether the difference
//! comes out of the noise — the radiometric half of an optical tasking gate.
//!
//! [`crate::observability`] answers *is it dark, and how much atmosphere is in the way*. This module
//! turns that into a number an observing plan can be written against: an apparent magnitude for the
//! object, a sky surface brightness for the background including moonlight, and a signal-to-noise
//! ratio from the two over a stated aperture and exposure.
//!
//! # Sources, and why they are not a tutorial page
//!
//! The moonlight model is Krisciunas & Schaefer (1991), PASP **103**, 1033, equation by equation, and
//! it carries the paper's *own* air mass `X(Z) = (1 − 0.96 sin²Z)^−0.5` rather than the sharper fits
//! in [`crate::observability`]: the scattering coefficients were fitted against that `X`, and it is
//! the one that stays bounded — 5.0 at the horizon — where a Kasten & Young air mass of 38 would
//! drive the extinction term to zero and the model outside the range it was calibrated on.
//!
//! The dark-sky surface brightness is a **parameter**, not a constant. It is a site, season and
//! solar-cycle property; a genuinely dark site is 21.7–22.0 V mag/arcsec² and a suburban one is
//! four magnitudes brighter, and the widely-copied summaries that quote a moonless sky near +20
//! alongside airglow at +23.5 are internally inconsistent by three magnitudes.
//!
//! # Sign conventions
//!
//! Magnitudes are magnitudes: larger is fainter, and a difference of 5 is a factor of 100 in flux.
//! [`DetectabilityDetector`]'s `g` follows [`crate::detectors`] — positive when the object is above
//! the required signal-to-noise.
//!
//! # Gates
//!
//! `derivations/photometry_closed_forms.mac` derives the diffuse-sphere phase law from its Lambert
//! integral, the V-band zero point, the Krisciunas & Schaefer air mass and lunar phase law at exact
//! points, and the closed-form root of the CCD equation; the tests below named `(G2)`/`(G3)` hold
//! the shipped code to them.

use spacelab_core::Vec3;

use crate::observability::{AirmassModel, Refraction, TrueAltitude, extinction};

/// Planck's constant, J·s — SI defining constant, exact.
const PLANCK: f64 = 6.626_070_15e-34;

/// Speed of light in vacuum, m/s — SI defining constant, exact.
const LIGHT_SPEED: f64 = 2.997_924_58e8;

/// Johnson–Cousins V effective wavelength, metres (Bessell 1990).
const V_BAND_WAVELENGTH: f64 = 545e-9;

/// Johnson–Cousins V effective width, metres (Bessell 1990).
const V_BAND_WIDTH: f64 = 84e-9;

/// Flux density of a `V = 0` source, W·m⁻²·m⁻¹ (Bessell, Castelli & Plez 1998: 3.63e-9 erg
/// cm⁻²·s⁻¹·Å⁻¹).
const V_BAND_ZERO_FLUX: f64 = 3.63e-2;

/// Photon flux of a `V = 0` source above the atmosphere, photons·s⁻¹·m⁻².
///
/// `F_λ Δλ λ/(hc)` with the four constants above: **8.366e9**. Published summaries quote
/// "about 1e10", which is the same number at the precision that phrase carries — the spread is the
/// choice of effective width, not a disagreement about the flux.
pub const V_BAND_ZERO_POINT_PHOTONS: f64 =
    V_BAND_ZERO_FLUX * V_BAND_WIDTH * V_BAND_WAVELENGTH / (PLANCK * LIGHT_SPEED);

/// Krisciunas & Schaefer's own optical path length, their Eq. 3: `X(Z) = (1 − 0.96 sin²Z)^−0.5`.
///
/// Bounded at 5.0 on the horizon, unlike the fits in [`crate::observability`]. It is used here and
/// nowhere else, because the scattering and extinction coefficients of the moonlight model were
/// fitted against it.
#[must_use]
pub fn krisciunas_schaefer_airmass(zenith_angle: f64) -> f64 {
    let s = libm::sin(zenith_angle);
    libm::pow(1.0 - 0.96 * s * s, -0.5)
}

/// Surface brightness in nanolamberts from V magnitudes per square arcsecond, the inverse of
/// [`nanolambert_to_v_mag`] — Krisciunas & Schaefer Eq. 27 read backwards.
#[must_use]
pub fn v_mag_to_nanolambert(v_mag_per_arcsec2: f64) -> f64 {
    34.08 * libm::exp(20.7233 - 0.92104 * v_mag_per_arcsec2)
}

/// V magnitudes per square arcsecond from a surface brightness in nanolamberts — Krisciunas &
/// Schaefer Eq. 27, `V = (20.7233 − ln(B/34.08))/0.92104`.
#[must_use]
pub fn nanolambert_to_v_mag(nanolambert: f64) -> f64 {
    (20.7233 - libm::log(nanolambert / 34.08)) / 0.92104
}

/// The Moon's apparent V magnitude at phase angle `alpha` radians, `−12.73 + 0.026|α| + 4e−9 α⁴`
/// with `α` in degrees — the published lunar phase law, and the calibration
/// [`moon_illuminance`] is the illuminance form of.
///
/// `−12.73` at full, about `−11.0` at quarter, and the quartic term is what makes the last few
/// degrees before full so much brighter than a linear law predicts (the opposition surge).
#[must_use]
pub fn lunar_v_magnitude(alpha: f64) -> f64 {
    let deg = libm::fabs(alpha.to_degrees());
    -12.73 + 0.026 * deg + 4.0e-9 * deg * deg * deg * deg
}

/// Krisciunas & Schaefer Eq. 20: the Moon's illuminance outside the atmosphere,
/// `I* = 10^(−0.4(3.84 + 0.026|α| + 4e−9 α⁴))`, `α` the phase angle in degrees.
///
/// The phase-dependent part is [`lunar_v_magnitude`] shifted by a constant, so the *ratio*
/// `I*(α)/I*(0)` is exactly the published phase law and carries no free parameter.
#[must_use]
pub fn moon_illuminance(alpha: f64) -> f64 {
    let deg = libm::fabs(alpha.to_degrees());
    libm::pow(
        10.0,
        -0.4 * (3.84 + 0.026 * deg + 4.0e-9 * deg * deg * deg * deg),
    )
}

/// Krisciunas & Schaefer Eq. 21: the scattering function at angular separation `rho` radians,
/// Rayleigh plus Mie.
///
/// `f(ρ) = 10^5.36 (1.06 + cos²ρ) + 10^(6.15 − ρ/40)` with `ρ` in degrees, and the aureole form
/// `6.2e7/ρ²` replacing the Mie term inside 10°, where the exponential no longer describes the
/// forward-scattering peak. The two branches do not meet — `f` is 1.09e6 at 10°, which the aureole
/// owns, and 1.26e6 just past it — and that step is the paper's, kept rather than smoothed, because
/// smoothing it would be a different model.
#[must_use]
pub fn moon_scattering(rho: f64) -> f64 {
    let deg = libm::fabs(rho.to_degrees());
    let rayleigh = libm::pow(10.0, 5.36) * (1.06 + libm::cos(rho) * libm::cos(rho));
    let mie = if deg > 10.0 {
        libm::pow(10.0, 6.15 - deg / 40.0)
    } else {
        6.2e7 / (deg * deg)
    };
    rayleigh + mie
}

/// A site's sky background: how dark it is at the zenith with no Moon, and how much a magnitude of
/// atmosphere costs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyBackground {
    /// Moonless zenith surface brightness, V mag/arcsec². 21.7–22.0 at a genuinely dark site, 21.6
    /// at Krisciunas & Schaefer's own, 18 in a suburb.
    pub dark_zenith_v_mag: f64,
    /// Extinction coefficient, magnitudes per air mass. A site and season property.
    pub extinction_coefficient: f64,
}

impl SkyBackground {
    /// The moonless background at zenith angle `Z`, nanolamberts — Krisciunas & Schaefer Eq. 2,
    /// `B₀(Z) = B_zenith · 10^(−0.4 k (X − 1)) · X`.
    ///
    /// Two competing effects, and both belong: the airglow is *emitted* by the atmosphere, so it
    /// scales with the path length `X`, and it is then extinguished over the extra `X − 1` air
    /// masses. The net is brighter toward the horizon at ordinary extinction coefficients.
    #[must_use]
    pub fn dark_nanolambert(&self, zenith_angle: f64) -> f64 {
        let x = krisciunas_schaefer_airmass(zenith_angle);
        v_mag_to_nanolambert(self.dark_zenith_v_mag)
            * libm::pow(10.0, -0.4 * self.extinction_coefficient * (x - 1.0))
            * x
    }

    /// The Moon's contribution, nanolamberts — Krisciunas & Schaefer Eq. 15,
    /// `B_moon = f(ρ) I*(α) 10^(−0.4 k X(Z_moon)) (1 − 10^(−0.4 k X(Z)))`.
    ///
    /// Zero when the Moon is at or below the horizon: the model's `X(Z)` is symmetric about `Z = 90°`
    /// and would otherwise report a *set* Moon as brightly as a risen one. That is a domain closure,
    /// not a guard — there is no scattered moonlight from a Moon that is not up.
    #[must_use]
    pub fn moon_nanolambert(
        &self,
        phase_angle: f64,
        separation: f64,
        moon_zenith_angle: f64,
        target_zenith_angle: f64,
    ) -> f64 {
        if moon_zenith_angle.is_nan() || moon_zenith_angle >= core::f64::consts::FRAC_PI_2 {
            return 0.0;
        }
        let k = self.extinction_coefficient;
        moon_scattering(separation)
            * moon_illuminance(phase_angle)
            * libm::pow(
                10.0,
                -0.4 * k * krisciunas_schaefer_airmass(moon_zenith_angle),
            )
            * (1.0
                - libm::pow(
                    10.0,
                    -0.4 * k * krisciunas_schaefer_airmass(target_zenith_angle),
                ))
    }

    /// Dark sky plus moonlight, as a V surface brightness in mag/arcsec² — the number a limiting
    /// magnitude is computed against.
    #[must_use]
    pub fn total_v_mag(
        &self,
        phase_angle: f64,
        separation: f64,
        moon_zenith_angle: f64,
        target_zenith_angle: f64,
    ) -> f64 {
        nanolambert_to_v_mag(
            self.dark_nanolambert(target_zenith_angle)
                + self.moon_nanolambert(
                    phase_angle,
                    separation,
                    moon_zenith_angle,
                    target_zenith_angle,
                ),
        )
    }
}

/// How an object's brightness varies with the angle between the Sun and the observer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseFunction {
    /// Diffuse sphere: `Φ(α) = [sin α + (π − α) cos α]/π`, 1 at opposition and 0 at conjunction.
    /// The default model for an unresolved rocket body or a tumbling fragment.
    DiffuseSphere,
    /// Specular sphere: a mirror ball reflects the same solid angle whatever the phase, so `Φ ≡ 1`.
    /// The two bracket a real object's behaviour.
    SpecularSphere,
    /// `Φ = 10^(−0.4 β α)` with `α` in radians: a magnitude penalty linear in phase angle, the form
    /// a fitted light curve usually comes back in.
    Linear {
        /// Slope, magnitudes per radian.
        beta_mag_per_rad: f64,
    },
}

impl PhaseFunction {
    /// The dimensionless brightness factor at phase angle `alpha` radians, 1 at opposition.
    #[must_use]
    pub fn at(self, alpha: f64) -> f64 {
        let a = libm::fabs(alpha).clamp(0.0, core::f64::consts::PI);
        match self {
            Self::DiffuseSphere => {
                (libm::sin(a) + (core::f64::consts::PI - a) * libm::cos(a)) / core::f64::consts::PI
            }
            Self::SpecularSphere => 1.0,
            Self::Linear { beta_mag_per_rad } => libm::pow(10.0, -0.4 * beta_mag_per_rad * a),
        }
    }
}

/// An object's brightness, as a standard magnitude at a reference range and phase.
///
/// The classical form `m = m₀ + 5 log₁₀(r·R) + f(α)` with the reference folded into `m₀`, which for
/// a resident space object means a magnitude quoted at a stated range — 1000 km is the usual choice
/// — and a stated phase angle, conventionally 90°.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StandardMagnitude {
    /// Apparent V magnitude at `reference_range` and `reference_phase`.
    pub magnitude: f64,
    /// Range the magnitude is quoted at, metres.
    pub reference_range: f64,
    /// Phase angle the magnitude is quoted at, radians.
    pub reference_phase: f64,
    /// How the brightness varies away from `reference_phase`.
    pub phase_function: PhaseFunction,
}

impl StandardMagnitude {
    /// A magnitude quoted at 1000 km and 90° phase off a diffuse sphere — the SSA convention.
    #[must_use]
    pub const fn at_1000_km(magnitude: f64) -> Self {
        Self {
            magnitude,
            reference_range: 1.0e6,
            reference_phase: core::f64::consts::FRAC_PI_2,
            phase_function: PhaseFunction::DiffuseSphere,
        }
    }

    /// Apparent magnitude at `range` metres and phase angle `alpha` radians, above the atmosphere.
    ///
    /// `f64::INFINITY` — infinitely faint — where the phase function vanishes, which for a diffuse
    /// sphere is exact conjunction: the object is then between observer and Sun with its unlit face
    /// turned outward, and no exposure recovers it.
    #[must_use]
    pub fn apparent(&self, range: f64, alpha: f64) -> f64 {
        let phase = self.phase_function.at(alpha) / self.phase_function.at(self.reference_phase);
        if phase.is_nan() || phase <= 0.0 {
            return f64::INFINITY;
        }
        self.magnitude + 5.0 * libm::log10(range / self.reference_range) - 2.5 * libm::log10(phase)
    }
}

/// The instrument: what it collects, what it costs to read, and how big a point source lands.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalSensor {
    /// Clear aperture diameter, metres.
    pub aperture_diameter: f64,
    /// Central obstruction as a fraction of the diameter; the collecting area is
    /// `π D²(1 − ε²)/4`.
    pub obstruction_ratio: f64,
    /// Optical throughput, fraction reaching the detector.
    pub throughput: f64,
    /// Detector quantum efficiency at V.
    pub quantum_efficiency: f64,
    /// Exposure, seconds.
    pub exposure: f64,
    /// Plate scale, arcseconds per pixel.
    pub pixel_scale_arcsec: f64,
    /// Radius of the photometric aperture, arcseconds — usually of order the seeing disc.
    pub photometry_radius_arcsec: f64,
    /// Read noise, electrons RMS per pixel per read.
    pub read_noise: f64,
    /// Dark current, electrons per pixel per second.
    pub dark_current: f64,
}

impl OpticalSensor {
    /// Collecting area, m².
    #[must_use]
    pub fn collecting_area(&self) -> f64 {
        let d = self.aperture_diameter;
        let e = self.obstruction_ratio;
        core::f64::consts::PI * d * d * (1.0 - e * e) / 4.0
    }

    /// Pixels inside the photometric aperture, `π (r/scale)²`.
    #[must_use]
    pub fn pixels_in_aperture(&self) -> f64 {
        let r = self.photometry_radius_arcsec / self.pixel_scale_arcsec;
        core::f64::consts::PI * r * r
    }

    /// Electrons collected from a source of apparent magnitude `v_mag` at the detector, i.e. after
    /// whatever extinction the caller has already added to the magnitude.
    #[must_use]
    pub fn source_electrons(&self, v_mag: f64) -> f64 {
        V_BAND_ZERO_POINT_PHOTONS
            * libm::pow(10.0, -0.4 * v_mag)
            * self.collecting_area()
            * self.throughput
            * self.quantum_efficiency
            * self.exposure
    }

    /// Electrons per pixel from a sky of surface brightness `v_mag_per_arcsec2`.
    #[must_use]
    pub fn sky_electrons_per_pixel(&self, v_mag_per_arcsec2: f64) -> f64 {
        self.source_electrons(v_mag_per_arcsec2) * self.pixel_scale_arcsec * self.pixel_scale_arcsec
    }

    /// The CCD equation: `SNR = S/√(S + n(B + D t + R²))`.
    ///
    /// `v_mag` is the source *at the detector* and `sky_v_mag_per_arcsec2` the background it sits on.
    /// Returns 0 for a source that contributes nothing, which is the value a detector needs at an
    /// infinite magnitude rather than a `NaN` out of `0/0`.
    #[must_use]
    pub fn snr(&self, v_mag: f64, sky_v_mag_per_arcsec2: f64) -> f64 {
        let signal = self.source_electrons(v_mag);
        if signal.is_nan() || signal <= 0.0 {
            return 0.0;
        }
        let pixels = self.pixels_in_aperture();
        let sky = self.sky_electrons_per_pixel(sky_v_mag_per_arcsec2);
        let dark = self.dark_current * self.exposure;
        let variance = signal + pixels * (sky + dark + self.read_noise * self.read_noise);
        signal / libm::sqrt(variance)
    }

    /// The magnitude at which [`OpticalSensor::snr`] equals `target_snr`, by bisection on magnitude.
    ///
    /// Monotone in `v_mag` — a fainter source is fewer electrons against the same background — so
    /// bisection is exact rather than merely convergent. Searched over `[-5, 40]`, which spans
    /// Venus to well past any real detector.
    #[must_use]
    pub fn limiting_magnitude(&self, sky_v_mag_per_arcsec2: f64, target_snr: f64) -> f64 {
        let (mut bright, mut faint) = (-5.0_f64, 40.0_f64);
        // 60 halvings take a 45-magnitude interval to 4e-17, far below the f64 resolution of the
        // answer; the count is fixed so the loop cannot be made non-terminating.
        for _ in 0..60 {
            let mid = f64::midpoint(bright, faint);
            if self.snr(mid, sky_v_mag_per_arcsec2) >= target_snr {
                bright = mid;
            } else {
                faint = mid;
            }
        }
        f64::midpoint(bright, faint)
    }
}

/// The composed gate: is *this* object detectable from *this* site right now, and at what
/// signal-to-noise.
///
/// `g = SNR − required_snr`, so `g > 0` when the object is detectable and [`crate::find_events`]
/// brackets the moment it becomes so. Everything the model needs that is not geometry — the object's
/// standard magnitude, the site's sky and the instrument — is fixed in the detector; the geometry
/// arrives per call.
#[derive(Debug, Clone, Copy)]
pub struct DetectabilityDetector {
    /// The object's brightness law.
    pub target: StandardMagnitude,
    /// The site's background and extinction.
    pub sky: SkyBackground,
    /// The instrument.
    pub sensor: OpticalSensor,
    /// The signal-to-noise the gate opens at.
    pub required_snr: f64,
    /// Which air-mass fit to extinguish through.
    pub airmass: AirmassModel,
    /// Refraction conditions at the site.
    pub refraction: Refraction,
}

/// The geometry one detectability evaluation needs, all in one frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationGeometry {
    /// Observer position.
    pub observer: Vec3,
    /// Observer's local vertical (need not be unit).
    pub up: Vec3,
    /// Target position.
    pub target: Vec3,
    /// Sun position.
    pub sun: Vec3,
    /// Moon position.
    pub moon: Vec3,
}

/// The angle between two vectors, `atan2(|a×b|, a·b)` — accurate at both ends of its range.
fn angle_between(a: Vec3, b: Vec3) -> f64 {
    libm::atan2(a.cross(b).norm(), a.dot(b))
}

/// True altitude of `what` above the horizon of an observer at `observer` with vertical `up`.
fn altitude(observer: Vec3, up: Vec3, what: Vec3) -> f64 {
    let d = what - observer;
    let vertical = d.dot(up) / up.norm();
    let horizontal = (d - up.scale(vertical / up.norm())).norm();
    libm::atan2(vertical, horizontal)
}

impl DetectabilityDetector {
    /// Signal-to-noise for this geometry, with extinction applied to the object and the Moon's
    /// scattered light applied to the background.
    #[must_use]
    pub fn snr(&self, geom: &ObservationGeometry) -> f64 {
        let to_target = geom.target - geom.observer;
        let range = to_target.norm();
        // Phase angle at the object: Sun–object–observer.
        let alpha = angle_between(geom.sun - geom.target, geom.observer - geom.target);
        let target_alt = TrueAltitude(altitude(geom.observer, geom.up, geom.target));
        let x = self.airmass.at(target_alt, self.refraction);
        let above = self.target.apparent(range, alpha);
        let observed = above + extinction(self.sky.extinction_coefficient, x);

        let moon_alt = altitude(geom.observer, geom.up, geom.moon);
        let separation = angle_between(to_target, geom.moon - geom.observer);
        let moon_phase = angle_between(geom.sun - geom.moon, geom.observer - geom.moon);
        let background = self.sky.total_v_mag(
            moon_phase,
            separation,
            core::f64::consts::FRAC_PI_2 - moon_alt,
            target_alt.zenith_angle(),
        );
        self.sensor.snr(observed, background)
    }

    /// `g = SNR − required_snr`, positive when the object is detectable.
    #[must_use]
    pub fn g(&self, geom: &ObservationGeometry) -> f64 {
        self.snr(geom) - self.required_snr
    }
}

#[cfg(test)]
mod tests {
    // `float_cmp`: the model's closed domains return exactly `0.0` and exactly `INFINITY`, by
    // construction rather than by arithmetic.
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::observability::Twilight;

    const DEG: f64 = core::f64::consts::PI / 180.0;

    fn dark_site() -> SkyBackground {
        SkyBackground {
            dark_zenith_v_mag: 21.8,
            extinction_coefficient: 0.172,
        }
    }

    fn metre_class() -> OpticalSensor {
        OpticalSensor {
            aperture_diameter: 1.0,
            obstruction_ratio: 0.3,
            throughput: 0.6,
            quantum_efficiency: 0.85,
            exposure: 30.0,
            pixel_scale_arcsec: 1.0,
            photometry_radius_arcsec: 2.0,
            read_noise: 5.0,
            dark_current: 0.01,
        }
    }

    /// (G2) The diffuse-sphere phase law against the integral that defines it, `∫ cos i cos e` over
    /// the lit and visible lune normalised to opposition, derived in closed form in
    /// `derivations/photometry_closed_forms.mac` and integrated numerically in
    /// `derivations/photometry_closed_forms.gp`; the values are the derivation's, at `α = kπ/6`.
    ///
    /// Bound 4·ε absolute (`Φ ≤ 1`): `sin α` and `(π − α) cos α` are each within an ulp or two, and
    /// near conjunction they cancel, so the error is absolute rather than relative there.
    #[test]
    // The derived values are kept at the derivation's width, not at the width f64 distinguishes.
    // Φ(π/2) = 1/π is one of them, deliberately the derivation's literal and not `FRAC_1_PI`.
    #[allow(clippy::excessive_precision, clippy::approx_constant)]
    fn the_diffuse_sphere_phase_law_is_the_lambert_integral() {
        const EXACT: [f64; 7] = [
            1.0,
            0.880_842_779_578_927_541_405_319_738_999_961_2,
            0.608_997_781_044_229_358_088_996_582_489_818_1,
            0.318_309_886_183_790_671_537_767_526_745_028_7,
            0.108_997_781_044_229_358_088_996_582_489_818_1,
            0.014_817_375_794_488_894_641_596_568_247_025,
            0.0,
        ];
        for (k, exact) in EXACT.into_iter().enumerate() {
            #[allow(clippy::cast_precision_loss)] // k <= 6
            let alpha = k as f64 * core::f64::consts::PI / 6.0;
            let got = PhaseFunction::DiffuseSphere.at(alpha);
            assert!(
                (got - exact).abs() <= 4.0 * f64::EPSILON,
                "Φ({k}π/6) = {got:e} against {exact:e}, error {:e}",
                (got - exact).abs()
            );
        }
    }

    /// (G3) The V-band zero point against its exact rational `F_λ Δλ λ/(hc)` =
    /// 7913400000000000000000000/945926598642347 photons·s⁻¹·m⁻², from
    /// `derivations/photometry_closed_forms.mac` (and `.gp`). Bound 4 ulps: three products and a
    /// quotient of correctly rounded decimal constants.
    #[test]
    // The derived values are kept at the derivation's width, not at the width f64 distinguishes.
    #[allow(clippy::excessive_precision)]
    fn the_v_band_zero_point_is_the_exact_rational() {
        let exact = 8.365_765_389_574_419_916_357_019_585_559_658e9;
        let err = (V_BAND_ZERO_POINT_PHOTONS - exact).abs() / exact;
        assert!(
            err <= 4.0 * f64::EPSILON,
            "{V_BAND_ZERO_POINT_PHOTONS:e}: relative {err:e}"
        );
    }

    /// (G3) Krisciunas & Schaefer's air mass at exact angles: 1 at the zenith, `5/√19` at 60° and
    /// `5/√7` at 30° elevation, 5 on the horizon (`derivations/photometry_closed_forms.mac`). Bound
    /// 4 ulps relative.
    #[test]
    fn the_krisciunas_schaefer_air_mass_has_its_exact_values() {
        let pi = core::f64::consts::PI;
        for (z, exact) in [
            (0.0, 1.0),
            (pi / 6.0, 5.0 / libm::sqrt(19.0)),
            (pi / 3.0, 5.0 / libm::sqrt(7.0)),
            (pi / 2.0, 5.0),
        ] {
            let got = krisciunas_schaefer_airmass(z);
            let err = (got - exact).abs() / exact;
            assert!(
                err <= 4.0 * f64::EPSILON,
                "X({z}) = {got} against {exact}: {err:e}"
            );
        }
    }

    /// (G2) `limiting_magnitude` against the closed-form root of the CCD equation it inverts:
    /// `SNR² (S + nN) = S²` gives `S = (s² + √(s⁴ + 4s²nN))/2` and `V = −2.5 log₁₀(S/rate)`, derived
    /// in `derivations/photometry_closed_forms.mac` and re-solved from `SNR(V) = s` in
    /// `derivations/photometry_closed_forms.gp`, for the metre-class sensor at four (sky, SNR).
    ///
    /// Bound 1e-13 mag: the bisection resolves its interval to an ulp of `V` (3.6e-15 at 21), and the
    /// f64 SNR it compares, a few ulps off, moves the crossing by `δSNR/|dSNR/dV|`, ~1e-15.
    #[test]
    // The derived values are kept at the derivation's width, not at the width f64 distinguishes.
    #[allow(clippy::excessive_precision)]
    fn the_limiting_magnitude_is_the_root_of_the_ccd_equation() {
        let s = metre_class();
        for (sky, snr, exact) in [
            (21.8, 5.0, 21.352_475_800_487_403_109_614_656_579_009_28),
            (19.0, 5.0, 20.058_393_222_308_139_460_639_913_520_528_36),
            (21.8, 10.0, 20.545_861_780_002_524_021_563_358_788_274_08),
            (21.8, 3.0, 21.928_758_371_223_762_274_871_995_827_629_50),
        ] {
            let got = s.limiting_magnitude(sky, snr);
            assert!(
                (got - exact).abs() <= 1e-13,
                "sky {sky}, SNR {snr}: {got} against {exact}, error {:e}",
                (got - exact).abs()
            );
        }
    }

    #[test]
    fn the_v_band_zero_point_is_the_published_flux_turned_into_photons() {
        // 8.366e9 photons/s/m², and the "about 1e10" every summary quotes is the same number at the
        // precision that phrase carries. A unit slip anywhere in the chain moves this by decades,
        // which is what the assertion is for.
        assert!(
            (V_BAND_ZERO_POINT_PHOTONS - 8.366e9).abs() < 2e6,
            "{V_BAND_ZERO_POINT_PHOTONS:e}"
        );
    }

    #[test]
    fn the_lunar_illuminance_carries_exactly_the_published_phase_law() {
        // `I*` is the published magnitude law shifted by a constant, so the ratio to full moon must
        // reproduce `Δm = 0.026α + 4e−9 α⁴` exactly — which pins the phase terms without needing to
        // know what unit `I*` is in.
        for alpha_deg in [0.0_f64, 15.0, 45.0, 90.0, 120.0, 180.0] {
            let alpha = alpha_deg * DEG;
            let ratio = moon_illuminance(alpha) / moon_illuminance(0.0);
            let dm = lunar_v_magnitude(alpha) - lunar_v_magnitude(0.0);
            let expected = libm::pow(10.0, -0.4 * dm);
            assert!(
                (ratio - expected).abs() <= 1e-12 * expected,
                "alpha {alpha_deg}: {ratio:e} vs {expected:e}"
            );
        }
        // And the published anchor the law is quoted with: the full Moon at V = −12.73.
        assert!((lunar_v_magnitude(0.0) + 12.73).abs() < 1e-12);
        // Quarter Moon, α = 90°: −12.73 + 2.34 + 0.2624, and that quartic quarter-magnitude is
        // the term a linear phase law has no way to produce.
        assert!((lunar_v_magnitude(90.0 * DEG) + 10.127_56).abs() < 1e-5);
        // The new Moon is 8.88 magnitudes fainter than the full Moon, a factor of 3.56e3.
        assert!((lunar_v_magnitude(180.0 * DEG) - lunar_v_magnitude(0.0) - 8.88).abs() < 0.01);
    }

    #[test]
    fn the_nanolambert_conversion_inverts_itself() {
        for v in [16.0_f64, 18.0, 21.0, 21.8, 24.0] {
            let round = nanolambert_to_v_mag(v_mag_to_nanolambert(v));
            assert!((round - v).abs() < 1e-12, "{v} -> {round}");
        }
        // Fainter is a smaller brightness, and 5 magnitudes is a factor of 100.
        let a = v_mag_to_nanolambert(21.0);
        let b = v_mag_to_nanolambert(26.0);
        assert!((a / b - 100.0).abs() < 0.02, "{}", a / b);
    }

    #[test]
    fn the_krisciunas_schaefer_path_is_bounded_at_five() {
        assert!((krisciunas_schaefer_airmass(0.0) - 1.0).abs() < 1e-15);
        assert!(
            (krisciunas_schaefer_airmass(core::f64::consts::FRAC_PI_2) - 5.0).abs() < 1e-12,
            "{}",
            krisciunas_schaefer_airmass(core::f64::consts::FRAC_PI_2)
        );
        // Monotone in between, which is what makes it usable as a path length at all.
        let mut previous = 0.0;
        for z_deg in 0..=90 {
            let x = krisciunas_schaefer_airmass(f64::from(z_deg) * DEG);
            assert!(x >= previous, "not monotone at {z_deg}: {x} < {previous}");
            previous = x;
        }
    }

    #[test]
    fn moonlight_brightens_the_sky_by_the_amounts_the_model_is_used_for() {
        // The end-to-end sanity of the moonlight model: a dark site at 21.8 goes to roughly 19 with
        // the full Moon 60° away and both bodies well up, and to roughly 21 at quarter. Those are
        // the magnitudes the model is quoted for; a sign slip or a decade error in `f(ρ)` leaves
        // this range immediately.
        let sky = dark_site();
        let z = 40.0 * DEG;
        let full = sky.total_v_mag(0.0, 60.0 * DEG, z, z);
        let quarter = sky.total_v_mag(90.0 * DEG, 60.0 * DEG, z, z);
        let new = sky.total_v_mag(180.0 * DEG, 60.0 * DEG, z, z);
        assert!((18.0..20.0).contains(&full), "full moon sky {full}");
        assert!(
            (20.0..21.6).contains(&quarter),
            "quarter moon sky {quarter}"
        );
        // A new Moon adds nothing measurable: the sky is the moonless value at that zenith angle.
        let moonless = nanolambert_to_v_mag(sky.dark_nanolambert(z));
        assert!((new - moonless).abs() < 0.01, "{new} vs {moonless}");
        // Brighter with a fuller Moon, and brighter closer to it.
        assert!(full < quarter && quarter < new);
        let near = sky.total_v_mag(0.0, 30.0 * DEG, z, z);
        assert!(near < full, "{near} should be brighter than {full}");
    }

    #[test]
    fn a_moon_below_the_horizon_contributes_nothing() {
        // The model's own `X(Z)` is symmetric about 90°, so without the domain closure a Moon 30°
        // below the horizon would scatter as much light as one 30° above it.
        let sky = dark_site();
        let below = sky.moon_nanolambert(0.0, 60.0 * DEG, 120.0 * DEG, 40.0 * DEG);
        assert_eq!(below, 0.0);
        assert_eq!(
            sky.moon_nanolambert(0.0, 60.0 * DEG, core::f64::consts::FRAC_PI_2, 40.0 * DEG),
            0.0
        );
        // And just above it, the contribution is present and large.
        let above = sky.moon_nanolambert(0.0, 60.0 * DEG, 89.0 * DEG, 40.0 * DEG);
        assert!(above > 100.0, "{above}");
    }

    #[test]
    fn the_diffuse_sphere_phase_function_is_one_at_opposition_and_zero_at_conjunction() {
        let p = PhaseFunction::DiffuseSphere;
        assert!((p.at(0.0) - 1.0).abs() < 1e-15);
        assert!(p.at(core::f64::consts::PI).abs() < 1e-15);
        // At quadrature the classical value is 1/π.
        assert!((p.at(core::f64::consts::FRAC_PI_2) - 1.0 / core::f64::consts::PI).abs() < 1e-15);
        // Monotone decreasing, and the specular sphere is flat.
        let mut previous = f64::INFINITY;
        for deg in 0..=180 {
            let v = p.at(f64::from(deg) * DEG);
            assert!(v <= previous + 1e-15, "not monotone at {deg}");
            previous = v;
            assert!((PhaseFunction::SpecularSphere.at(f64::from(deg) * DEG) - 1.0).abs() < 1e-15);
        }
        // The linear law is a straight line in magnitudes, by construction.
        let lin = PhaseFunction::Linear {
            beta_mag_per_rad: 1.5,
        };
        let dm = -2.5 * libm::log10(lin.at(0.7) / lin.at(0.0));
        assert!((dm - 1.5 * 0.7).abs() < 1e-12, "{dm}");
    }

    #[test]
    fn the_inverse_square_law_is_five_magnitudes_per_decade_of_range() {
        let m = StandardMagnitude::at_1000_km(6.0);
        // At the reference range and phase, the standard magnitude is the apparent magnitude.
        assert!((m.apparent(1.0e6, core::f64::consts::FRAC_PI_2) - 6.0).abs() < 1e-12);
        // Ten times further is five magnitudes fainter; twice as far is 1.505.
        assert!((m.apparent(1.0e7, core::f64::consts::FRAC_PI_2) - 11.0).abs() < 1e-12);
        assert!((m.apparent(2.0e6, core::f64::consts::FRAC_PI_2) - 7.505_15).abs() < 1e-5);
        // Fully lit is brighter than at quadrature by 2.5 log10(π).
        let gain = m.apparent(1.0e6, core::f64::consts::FRAC_PI_2) - m.apparent(1.0e6, 0.0);
        assert!(
            (gain - 2.5 * libm::log10(core::f64::consts::PI)).abs() < 1e-12,
            "{gain}"
        );
        // Exact conjunction has no lit face, but `sin π` is 3.9e-17 rather than zero in `f64`, so
        // the diffuse sphere comes back at V = 45.8 instead of infinite — undetectable by 25
        // magnitudes, and honest about which of the two it is.
        let conjunction = m.apparent(1.0e6, core::f64::consts::PI);
        assert!((45.0..47.0).contains(&conjunction), "{conjunction}");
        // The infinite branch is reachable, through a phase law steep enough to underflow.
        let steep = StandardMagnitude {
            phase_function: PhaseFunction::Linear {
                beta_mag_per_rad: 3000.0,
            },
            ..m
        };
        assert_eq!(steep.apparent(1.0e6, 1.0), f64::INFINITY);
    }

    #[test]
    fn the_signal_to_noise_scales_the_way_the_ccd_equation_says() {
        let sky = dark_site();
        let s = metre_class();
        let background = sky.dark_zenith_v_mag;

        // Four times the exposure is *just over* twice the SNR, and the excess is the read noise:
        // signal, sky and dark all scale with `t` while `R²` does not, so the noise grows more slowly
        // than `√t` and the ratio climbs from 2.006 on a bright source to 2.099 on a faint one, where
        // the read-noise floor is a larger share of the variance. A model that scaled `R²` with
        // exposure would sit at exactly 2 and be wrong.
        let long = OpticalSensor {
            exposure: 120.0,
            ..s
        };
        let bright_ratio = long.snr(16.0, background) / s.snr(16.0, background);
        let faint_ratio = long.snr(23.0, background) / s.snr(23.0, background);
        assert!(
            (bright_ratio - 2.006).abs() < 0.005,
            "bright {bright_ratio}"
        );
        assert!((faint_ratio - 2.099).abs() < 0.005, "faint {faint_ratio}");
        assert!(bright_ratio < faint_ratio);

        // Twice the aperture is twice the SNR by the same argument: signal and sky both scale with
        // collecting area, so the ratio matches the exposure one term for term.
        let big = OpticalSensor {
            aperture_diameter: 2.0,
            ..s
        };
        let ratio = big.snr(16.0, background) / s.snr(16.0, background);
        assert!(
            (ratio - bright_ratio).abs() < 1e-3,
            "aperture scaling {ratio}"
        );

        // Monotone in magnitude, which is what `limiting_magnitude` bisects on.
        let mut previous = f64::INFINITY;
        for deci in 100..300 {
            let v = s.snr(f64::from(deci) / 10.0, background);
            assert!(
                v <= previous,
                "SNR not monotone at V = {}",
                f64::from(deci) / 10.0
            );
            previous = v;
        }
        // An infinitely faint source is exactly zero signal, not a NaN.
        assert_eq!(s.snr(f64::INFINITY, background), 0.0);
    }

    #[test]
    fn a_metre_class_telescope_reaches_the_magnitude_a_metre_class_telescope_reaches() {
        // The coarse end-to-end anchor: 1 m, 30 s, dark sky, 2" aperture radius should reach about
        // V = 21 at SNR 5. Anything outside 19–23 means a unit error somewhere in the chain — the
        // collecting area, the zero point, the pixel scale or the sky conversion — and this is the
        // only assertion in the module that would catch several of them at once.
        let s = metre_class();
        let limit = s.limiting_magnitude(21.8, 5.0);
        assert!((19.0..23.0).contains(&limit), "limiting magnitude {limit}");
        assert!(
            (limit - 21.352_476).abs() < 1e-5,
            "limiting magnitude {limit}"
        );
        // The bisection inverts `snr` exactly.
        assert!(
            (s.snr(limit, 21.8) - 5.0).abs() < 1e-6,
            "{}",
            s.snr(limit, 21.8)
        );
        // A brighter sky costs depth, and the full-moon sky costs about two magnitudes.
        let bright = s.limiting_magnitude(19.0, 5.0);
        assert!(bright < limit - 1.0, "{bright} against {limit}");
    }

    #[test]
    fn the_detectability_gate_opens_on_a_lit_object_over_a_dark_site() {
        let earth = 6_378_137.0_f64;
        let observer = Vec3::new(earth, 0.0, 0.0);
        let up = Vec3::new(1.0, 0.0, 0.0);
        let au = 1.495_978_707e11;
        // Local midnight: the Sun is straight down.
        let sun = observer + Vec3::new(-au, 0.0, 0.0);
        // The Moon 30° below the horizon, so it contributes nothing.
        let moon = observer + Vec3::new(-3.844e8 * 0.5, 0.0, 3.844e8 * 0.866);
        let det = DetectabilityDetector {
            target: StandardMagnitude::at_1000_km(6.0),
            sky: dark_site(),
            sensor: metre_class(),
            required_snr: 5.0,
            airmass: AirmassModel::KastenYoung1989,
            refraction: Refraction::STANDARD,
        };

        // Overhead at 800 km, sunlit from below the horizon — a bright, close, high-elevation pass.
        let overhead = observer + Vec3::new(8.0e5, 0.0, 0.0);
        let geom = ObservationGeometry {
            observer,
            up,
            target: overhead,
            sun,
            moon,
        };
        assert!(det.g(&geom) > 0.0, "SNR {}", det.snr(&geom));

        // The same object below the horizon: infinite air mass, no signal, gate shut by exactly the
        // required SNR.
        let under = observer + Vec3::new(-8.0e5, 0.0, 0.0);
        let below = ObservationGeometry {
            target: under,
            ..geom
        };
        assert_eq!(det.g(&below), -det.required_snr);

        // And an object 18 magnitudes fainter is not detectable: apparent V = 22.4 after extinction
        // against a limiting magnitude of 21.35 for this instrument and sky, so SNR 1.9 against the
        // 5 required.
        let faint = DetectabilityDetector {
            target: StandardMagnitude::at_1000_km(24.0),
            ..det
        };
        assert!(faint.g(&geom) < 0.0, "SNR {}", faint.snr(&geom));
    }

    #[test]
    fn the_moon_costs_depth_when_it_is_up_and_close() {
        // The reason the Moon is in the geometry at all: the same object, the same sky, the same
        // instrument, and a full Moon 20° away instead of below the horizon.
        let earth = 6_378_137.0_f64;
        let observer = Vec3::new(earth, 0.0, 0.0);
        let up = Vec3::new(1.0, 0.0, 0.0);
        let au = 1.495_978_707e11;
        let sun = observer + Vec3::new(-au, 0.0, 0.0);
        let det = DetectabilityDetector {
            target: StandardMagnitude::at_1000_km(11.0),
            sky: dark_site(),
            sensor: metre_class(),
            required_snr: 5.0,
            airmass: AirmassModel::KastenYoung1989,
            refraction: Refraction::STANDARD,
        };
        let target = observer + Vec3::new(8.0e5 * 0.966, 8.0e5 * 0.259, 0.0);
        let moon_down = observer + Vec3::new(-3.844e8, 0.0, 0.0);
        // Full Moon: opposite the Sun, so up when the Sun is down, and near the target.
        let moon_up = observer + Vec3::new(3.844e8 * 0.966, -3.844e8 * 0.259, 0.0);
        let dark = det.snr(&ObservationGeometry {
            observer,
            up,
            target,
            sun,
            moon: moon_down,
        });
        let bright = det.snr(&ObservationGeometry {
            observer,
            up,
            target,
            sun,
            moon: moon_up,
        });
        assert!(
            bright < dark,
            "moonlit {bright} should be worse than {dark}"
        );
        assert!(bright > 0.0, "moonlit SNR {bright}");
    }

    #[test]
    fn the_twilight_gate_and_the_detectability_gate_are_separate_questions() {
        // A reminder in test form of what this module does not do: it says nothing about whether the
        // observer's own sky is dark. The Sun's contribution is not in the background model, so a
        // daylight geometry comes back with the same SNR as a midnight one and only
        // `observability::DarknessDetector` refuses it. Composing the two is the caller's job, and
        // this asserts that the composition is necessary rather than merely tidy.
        assert!(Twilight::Astronomical.solar_altitude() < 0.0);
        let earth = 6_378_137.0_f64;
        let observer = Vec3::new(earth, 0.0, 0.0);
        let up = Vec3::new(1.0, 0.0, 0.0);
        let au = 1.495_978_707e11;
        let target = observer + Vec3::new(8.0e5, 0.0, 0.0);
        let moon = observer + Vec3::new(-3.844e8, 0.0, 0.0);
        let det = DetectabilityDetector {
            target: StandardMagnitude::at_1000_km(6.0),
            sky: dark_site(),
            sensor: metre_class(),
            required_snr: 5.0,
            airmass: AirmassModel::KastenYoung1989,
            refraction: Refraction::STANDARD,
        };
        let midnight = det.snr(&ObservationGeometry {
            observer,
            up,
            target,
            sun: observer + Vec3::new(-au, 0.0, 0.0),
            moon,
        });
        let noon = det.snr(&ObservationGeometry {
            observer,
            up,
            target,
            sun: observer + Vec3::new(au, 0.0, 0.0),
            moon,
        });
        assert!(noon > 0.0 && midnight > 0.0);
    }

    #[test]
    fn the_scattering_function_is_the_published_two_branch_form() {
        // Krisciunas & Schaefer Eq. 21, both branches, evaluated here from the paper rather than
        // read back through the module. `moonlight_brightens_the_sky…` sees only the composed sky
        // brightness at one separation, where the Rayleigh and Mie terms are within a factor of two
        // of each other and either can absorb an operator slip in the other.
        for rho_deg in [0.5_f64, 2.0, 5.0, 9.5, 10.0, 10.5, 30.0, 60.0, 90.0, 150.0] {
            let rho = rho_deg * DEG;
            let cos_rho = libm::cos(rho);
            let rayleigh = libm::pow(10.0, 5.36) * (1.06 + cos_rho * cos_rho);
            // The branch boundary belongs to the aureole: `> 10`, so 10.0 itself is `6.2e7/ρ²`.
            let mie = if rho_deg > 10.0 {
                libm::pow(10.0, 6.15 - rho_deg / 40.0)
            } else {
                6.2e7 / (rho_deg * rho_deg)
            };
            let expected = rayleigh + mie;
            let got = moon_scattering(rho);
            assert!(
                (got - expected).abs() <= 1e-12 * expected,
                "rho {rho_deg}: {got:e} against {expected:e}"
            );
        }
        // The paper's own step at 10°, kept rather than smoothed: 6.2e5 on the aureole side and
        // 7.94e5 just past it, a 28 % jump up. Which side owns the boundary is the whole content of
        // the comparison, and it is not visible from either branch alone.
        let ray10 = libm::pow(10.0, 5.36) * {
            let c = libm::cos(10.0 * DEG);
            1.06 + c * c
        };
        assert!((moon_scattering(10.0 * DEG) - ray10 - 6.2e5).abs() < 1.0);
        assert!((moon_scattering(10.000_001 * DEG) - ray10 - libm::pow(10.0, 5.9)).abs() < 1.0);
        // Scattering is symmetric in the sign of the separation and falls away from the Moon.
        assert_eq!(moon_scattering(-30.0 * DEG), moon_scattering(30.0 * DEG));
        assert!(moon_scattering(30.0 * DEG) > moon_scattering(90.0 * DEG));
    }

    #[test]
    fn the_moonless_background_composes_extinction_with_path_length() {
        // `B₀(Z) = B_zenith · 10^(−0.4 k (X − 1)) · X`. Every other assertion reads it through
        // `total_v_mag`, where moonlight dominates; the two factors are checked here directly, at
        // zenith angles where `X ≠ 1` and with an extinction coefficient that is not zero.
        let sky = dark_site();
        for z_deg in [0.0_f64, 30.0, 60.0, 80.0, 90.0] {
            let z = z_deg * DEG;
            let x = krisciunas_schaefer_airmass(z);
            let expected = v_mag_to_nanolambert(sky.dark_zenith_v_mag)
                * libm::pow(10.0, -0.4 * sky.extinction_coefficient * (x - 1.0))
                * x;
            let got = sky.dark_nanolambert(z);
            assert!(
                (got - expected).abs() <= 1e-12 * expected,
                "z {z_deg}: {got} against {expected}"
            );
        }
        // At the zenith the path is one air mass and neither factor does anything.
        let zenith = v_mag_to_nanolambert(sky.dark_zenith_v_mag);
        assert!((sky.dark_nanolambert(0.0) - zenith).abs() <= 1e-12 * zenith);
        // The documented net at an ordinary coefficient: brighter toward the horizon, because the
        // airglow grows with the path faster than the extinction removes it.
        assert!(sky.dark_nanolambert(80.0 * DEG) > sky.dark_nanolambert(0.0));
    }

    #[test]
    fn the_moon_term_extinguishes_along_both_paths() {
        // Two air masses enter, the Moon's and the target's, and they are not interchangeable: the
        // Moon's path attenuates the light on its way in, the target's decides how much of the
        // column is in front of the object.
        let sky = dark_site();
        let k = sky.extinction_coefficient;
        let (alpha, separation) = (50.0 * DEG, 35.0 * DEG);
        let (moon_z, target_z) = (30.0 * DEG, 60.0 * DEG);
        let expected = moon_scattering(separation)
            * moon_illuminance(alpha)
            * libm::pow(10.0, -0.4 * k * krisciunas_schaefer_airmass(moon_z))
            * (1.0 - libm::pow(10.0, -0.4 * k * krisciunas_schaefer_airmass(target_z)));
        let got = sky.moon_nanolambert(alpha, separation, moon_z, target_z);
        assert!(
            (got - expected).abs() <= 1e-12 * expected,
            "{got} against {expected}"
        );
        assert_ne!(
            sky.moon_nanolambert(alpha, separation, target_z, moon_z),
            got,
            "the two air masses are not interchangeable"
        );
    }

    #[test]
    fn the_pixel_scale_enters_the_aperture_and_the_sky_and_is_not_one() {
        // `metre_class` samples at exactly 1"/px, where `r/scale` and `r·scale` are the same number
        // and `× scale²` is the identity — so nothing about the pixel scale is observable there.
        let s = OpticalSensor {
            pixel_scale_arcsec: 0.5,
            photometry_radius_arcsec: 2.0,
            ..metre_class()
        };
        // 2" at 0.5"/px is 4 pixels of radius: π·16 in the aperture, not π·1.
        assert!((s.pixels_in_aperture() - core::f64::consts::PI * 16.0).abs() < 1e-12);
        // The sky is a surface brightness, so electrons per pixel scale with the pixel's *area*.
        let sky_mag = 21.8;
        let expected = s.source_electrons(sky_mag) * 0.25;
        assert!((s.sky_electrons_per_pixel(sky_mag) - expected).abs() <= 1e-12 * expected);
        // Halving the scale quarters the sky per pixel and quadruples the pixel count, which is the
        // pair of opposite powers that keeps the SNR nearly scale-free.
        let coarse = OpticalSensor {
            pixel_scale_arcsec: 1.0,
            ..s
        };
        let sky_ratio =
            coarse.sky_electrons_per_pixel(sky_mag) / s.sky_electrons_per_pixel(sky_mag);
        assert!((sky_ratio - 4.0).abs() < 1e-12, "{sky_ratio}");
        let pixel_ratio = s.pixels_in_aperture() / coarse.pixels_in_aperture();
        assert!((pixel_ratio - 4.0).abs() < 1e-12, "{pixel_ratio}");
    }

    #[test]
    fn the_signal_to_noise_is_zero_rather_than_nan_on_a_nan_source() {
        // The `NaN` disjunct is separate from the `<= 0` one and unreachable through it: `NaN <= 0`
        // is false, so a guard that tested only the sign would return `NaN/√NaN` and carry it into
        // the detector's `g`, where a negative-versus-positive gate becomes a silent never-opens.
        let s = metre_class();
        assert_eq!(s.snr(f64::NAN, 21.8), 0.0);
        assert_eq!(s.snr(f64::INFINITY, 21.8), 0.0);
    }

    #[test]
    fn the_bisection_bracket_reaches_the_bright_end_it_claims() {
        // The search is documented over [−5, 40], "Venus to well past any real detector", and every
        // other assertion lands near V = 21, where the bright end of the bracket is unobservable. A
        // small, fast instrument against a bright sky puts the answer below zero — where a bracket
        // that began at +5 would report its own endpoint as the limiting magnitude.
        let s = OpticalSensor {
            aperture_diameter: 0.01,
            exposure: 0.001,
            ..metre_class()
        };
        let limit = s.limiting_magnitude(15.0, 100.0);
        assert!(limit < 0.0, "limiting magnitude {limit}");
        assert!(
            limit > -5.0,
            "the answer must be inside the bracket, not at its edge: {limit}"
        );
        // And the bisection still inverts `snr` exactly there.
        let back = s.snr(limit, 15.0);
        assert!((back - 100.0).abs() < 1e-6, "{back}");
    }

    #[test]
    fn the_observation_geometry_is_measured_with_the_vectors_the_names_promise() {
        // `angle_between` and `altitude` are read only through the composed SNR, whose assertions
        // span a factor of several — wide enough for a flipped difference or a dropped
        // normalisation to pass unnoticed. Both have closed forms, and these are them.
        let x = Vec3::new(1.0, 0.0, 0.0);
        assert!(
            (angle_between(x, Vec3::new(0.0, 2.0, 0.0)) - core::f64::consts::FRAC_PI_2).abs()
                < 1e-15
        );
        assert!(angle_between(x, x.scale(3.0)).abs() < 1e-15);
        assert!((angle_between(x, x.scale(-3.0)) - core::f64::consts::PI).abs() < 1e-15);

        // `up` is documented as needing no normalisation, and that is the only thing that makes the
        // two `/up.norm()` divisions visible: at |up| = 1 they are the identity.
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let up = Vec3::new(0.0, 0.0, 3.0);
        assert!(
            (altitude(origin, up, Vec3::new(1.0, 0.0, 1.0)) - core::f64::consts::FRAC_PI_4).abs()
                < 1e-15
        );
        assert!(
            (altitude(origin, up, Vec3::new(0.0, 0.0, 5.0)) - core::f64::consts::FRAC_PI_2).abs()
                < 1e-15
        );
        assert!(altitude(origin, up, Vec3::new(7.0, 0.0, 0.0)).abs() < 1e-15);
        assert!(altitude(origin, up, Vec3::new(1.0, 0.0, -1.0)) < 0.0);
        // Scaling `up` cannot move an altitude.
        let target = Vec3::new(1.0, 2.0, 3.0);
        let scaled = altitude(origin, up.scale(11.0), target);
        let unit = altitude(origin, Vec3::new(0.0, 0.0, 1.0), target);
        assert!((scaled - unit).abs() < 1e-15, "{scaled} against {unit}");
    }

    #[test]
    fn the_detectability_snr_is_the_composition_of_its_named_parts() {
        // Six vector differences build the observation out of `ObservationGeometry`, and reversing
        // any one of them moves the SNR by less than the bands the end-to-end assertions allow.
        // This rebuilds the chain from the same published pieces and demands the same number, on a
        // geometry where the Moon is up and close enough that every term carries weight.
        let earth = 6_378_137.0_f64;
        let observer = Vec3::new(earth, 0.0, 0.0);
        let up = Vec3::new(1.0, 0.0, 0.0);
        let au = 1.495_978_707e11;
        let sun = observer + Vec3::new(-au * 0.5, au * 0.866, 0.0);
        let moon = observer + Vec3::new(3.844e8 * 0.7, -3.844e8 * 0.5, 3.844e8 * 0.51);
        let target = observer + Vec3::new(8.0e5 * 0.9, 8.0e5 * 0.3, 8.0e5 * 0.316);
        let det = DetectabilityDetector {
            target: StandardMagnitude::at_1000_km(9.0),
            sky: dark_site(),
            sensor: metre_class(),
            required_snr: 5.0,
            airmass: AirmassModel::KastenYoung1989,
            refraction: Refraction::STANDARD,
        };
        let geom = ObservationGeometry {
            observer,
            up,
            target,
            sun,
            moon,
        };

        let to_target = geom.target - geom.observer;
        let alpha = angle_between(geom.sun - geom.target, geom.observer - geom.target);
        let target_alt = TrueAltitude(altitude(geom.observer, geom.up, geom.target));
        let x = det.airmass.at(target_alt, det.refraction);
        let at_detector = det.target.apparent(to_target.norm(), alpha)
            + extinction(det.sky.extinction_coefficient, x);
        let moon_alt = altitude(geom.observer, geom.up, geom.moon);
        let background = det.sky.total_v_mag(
            angle_between(geom.sun - geom.moon, geom.observer - geom.moon),
            angle_between(to_target, geom.moon - geom.observer),
            core::f64::consts::FRAC_PI_2 - moon_alt,
            target_alt.zenith_angle(),
        );
        assert_eq!(det.snr(&geom), det.sensor.snr(at_detector, background));

        // The geometry has to be one where each part matters, or the assertion above is vacuous.
        assert!(moon_alt > 0.0, "the Moon must be up: {moon_alt}");
        assert!(
            (10.0 * DEG..80.0 * DEG).contains(&target_alt.0),
            "target altitude {}",
            target_alt.0
        );
        assert!(det.snr(&geom) > 0.0);
    }
}
