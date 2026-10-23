//! The photometric forward model: how bright a box-wing spacecraft looks, from its attitude.
//!
//! # What this is for
//!
//! [`spacelab_coverage::photometry::StandardMagnitude`] answers "how bright is this object" with one
//! fitted number and a phase law. That is the right model for a catalogue and the wrong one for
//! inference: it cannot produce a *light curve*, and a light curve is the observable that carries
//! attitude, spin rate and shape. This module goes the other way round — facets, a reflectance model
//! and a viewing geometry in, radiant intensity and an apparent magnitude out — so that an estimator
//! can compare a predicted brightness against a measured one.
//!
//! # The reflectance model
//!
//! Each facet keeps the coefficients the radiation-pressure model already carries, and they mean the
//! same thing here: absorbed `C_a`, specularly reflected `C_s`, diffusely reflected
//! `C_d = 1 − C_a − C_s`. Sharing them is the point — an object cannot push on sunlight with one
//! albedo and reflect it to a telescope with another, and a light curve that disagreed with the
//! spacecraft's own SRP model would be fitting a second, contradictory object.
//!
//! The radiant intensity a facet of area `A` sends toward the observer, with `θᵢ` the incidence and
//! `θₒ` the emission angle from its normal, is
//!
//! ```text
//! I(ω) = E·A·cos θᵢ · [ C_d/π · cos θₒ  +  C_s·(k+1)/(2π) · cosᵏ ψ ]
//! ```
//!
//! — a Lambertian half and a Phong specular lobe of exponent `k` about the mirror direction, `ψ` the
//! angle off it. Both halves are normalised so that integrating the intensity over the hemisphere
//! returns exactly `E·A·cos θᵢ·C_d` and `E·A·cos θᵢ·C_s`: the surface reflects the fraction of the
//! intercepted power its coefficients say it does, and no more. That is the property the tests pin,
//! and it is what makes `k` a *distribution* parameter rather than a brightness knob.
//!
//! One consequence to keep in view: at oblique incidence part of the lobe points below the surface,
//! and the model has nowhere else to put that power, so the facet reflects *less* than `C_s` allows.
//! Measured at 75° incidence, the loss is nothing at `k = 0`, 10.3% at `k = 8` and 2.2% at `k = 40`
//! — not monotone, because a lobe narrow enough to fit above the horizon clears it again. It never
//! reflects more, which is the direction that would matter.
//!
//! Phong is not a measured BRDF of any real material. It is the isotropic, Fresnel-free limit of the
//! Ashikhmin–Shirley model the resident-space-object literature standardised on (Wetterer & Jah 2009,
//! *J. Guid. Control Dyn.* **32**(5); Linares et al. 2014), chosen here because it adds exactly one
//! parameter to the coefficients the crate already has. A material model with a Fresnel term belongs
//! in the same place if a measurement ever demands one.
//!
//! # What is and is not modelled
//!
//! A facet contributes when it is both lit (`n̂·ŝ > 0`) and visible (`n̂·ô > 0`). For a **convex**
//! body those two tests are exact — no other facet can be in the way — which is why a box-wing is a
//! reasonable object to model this way and a spacecraft with a boom over its bus is not. Self-
//! shadowing of the bus by a solar array, and any inter-facet bounce, are absent. Earth shadow is
//! not computed here either: it enters as `sunlit_fraction`, which [`crate::shadow::lighting_ratio`]
//! produces, so that penumbra is a number rather than a boolean.
//!
//! # Verification status
//!
//! The zero point, the energy budget of the reflectance model and the geometry are pinned against
//! closed forms in `tests/lightcurve.rs`. The comparison this model ultimately wants — published
//! light curves of Telstar 401, Echostar 2, Solidaridad 1, `HGS-1`, ACTS and `GStar` 3 (Earl &
//! Wade, IAC-14-A6.9.1; Earl, `MSc` thesis, RMC 2013) — is **not** done: the repository holds none
//! of that photometry, and the papers' figures are theirs. The mathematics is free to implement
//! with a citation; the data has to be obtained before the residual can be quoted.

use crate::attitude::Vec3;
use crate::spacecraft::BoxWing;

/// Total solar irradiance at 1 au, W·m⁻² — the IAU 2015 Resolution B3 nominal value.
pub const SOLAR_IRRADIANCE: f64 = 1361.0;

/// The Sun's apparent V magnitude, −26.74.
///
/// It is the zero point of every magnitude this module returns: a source delivering the same
/// irradiance as the Sun is that bright, and the rest is `−2.5 log₁₀` of the ratio. Pairing it with
/// [`SOLAR_IRRADIANCE`] keeps the whole chain radiometric, in W·m⁻², with no photometric conversion
/// anywhere in it; that the two routes agree is checked once in the tests rather than relied on.
pub const SUN_V_MAGNITUDE: f64 = -26.74;

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn unit(v: Vec3) -> Option<Vec3> {
    let norm = libm::sqrt(dot(v, v));
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    Some([v[0] / norm, v[1] / norm, v[2] / norm])
}

/// Where the Sun and the observer are, as the body sees them.
///
/// The two directions are unit vectors in the **body** frame, which is what makes this a function of
/// attitude: rotate the same inertial geometry by a different attitude and the light curve changes.
/// [`Self::from_inertial`] does that rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViewGeometry {
    sun_body: Vec3,
    observer_body: Vec3,
    range: f64,
    sunlit_fraction: f64,
}

impl ViewGeometry {
    /// Build a geometry, or reject it. The directions are normalised here — a caller cannot hand in
    /// a non-unit vector and silently scale the brightness — and `None` comes back for a zero or
    /// non-finite direction, a non-positive range, or a `sunlit_fraction` outside `[0, 1]`.
    #[must_use]
    pub fn new(
        sun_body: Vec3,
        observer_body: Vec3,
        range: f64,
        sunlit_fraction: f64,
    ) -> Option<Self> {
        if !range.is_finite() || range <= 0.0 || !(0.0..=1.0).contains(&sunlit_fraction) {
            return None;
        }
        Some(Self {
            sun_body: unit(sun_body)?,
            observer_body: unit(observer_body)?,
            range,
            sunlit_fraction,
        })
    }

    /// The same geometry from inertial directions and an attitude, following
    /// [`BoxWing::drag_acceleration_inertial`]'s convention: `inertial_to_body` is the matrix `M`
    /// with `v_body = M·v_inertial`.
    #[must_use]
    pub fn from_inertial(
        sun_inertial: Vec3,
        observer_inertial: Vec3,
        range: f64,
        sunlit_fraction: f64,
        inertial_to_body: &[[f64; 3]; 3],
    ) -> Option<Self> {
        let rotate = |v: Vec3| {
            [
                dot(inertial_to_body[0], v),
                dot(inertial_to_body[1], v),
                dot(inertial_to_body[2], v),
            ]
        };
        Self::new(
            rotate(sun_inertial),
            rotate(observer_inertial),
            range,
            sunlit_fraction,
        )
    }

    /// Sun–object–observer angle, radians. Zero at opposition, where the object is fully lit as seen
    /// from the observer, and `π` at conjunction, where its lit side is turned away.
    #[must_use]
    pub fn phase_angle(&self) -> f64 {
        libm::acos(dot(self.sun_body, self.observer_body).clamp(-1.0, 1.0))
    }

    /// Unit body-frame direction to the Sun.
    #[must_use]
    pub fn sun_body(&self) -> Vec3 {
        self.sun_body
    }

    /// Unit body-frame direction to the observer.
    #[must_use]
    pub fn observer_body(&self) -> Vec3 {
        self.observer_body
    }

    /// Observer range, metres.
    #[must_use]
    pub fn range(&self) -> f64 {
        self.range
    }

    /// Fraction of the solar disc the body can see, `0` in umbra.
    #[must_use]
    pub fn sunlit_fraction(&self) -> f64 {
        self.sunlit_fraction
    }
}

/// A box-wing spacecraft as an *optical* object: the facets the force models already use, plus the
/// one parameter reflectance needs and radiation pressure does not — how tightly each facet's
/// specular lobe is bunched about the mirror direction.
///
/// The exponent is kept beside the facets for the same reason [`crate::torque::RigidBoxWing`] keeps
/// its arms there: it belongs to the observer's problem, not the propagator's, and adding a field to
/// `Facet` would put it into every constructor of a model verified against Orekit without one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotometricBoxWing<const P: usize> {
    /// The facets and their optical coefficients.
    pub shape: BoxWing<P>,
    /// Phong specular exponent per facet, index for index with `shape.facets`. Large is a mirror
    /// (a narrow glint that flashes as the body turns), small is a rough surface, and `0` spreads
    /// the specular fraction uniformly over the hemisphere.
    pub specular_exponent: [f64; P],
}

impl<const P: usize> PhotometricBoxWing<P> {
    /// Radiant intensity toward the observer, W·sr⁻¹, summed over the facets that are both lit and
    /// visible. Zero in umbra and zero when no facet is both.
    #[must_use]
    pub fn radiant_intensity(&self, view: &ViewGeometry) -> f64 {
        let irradiance = SOLAR_IRRADIANCE * view.sunlit_fraction;
        let mut intensity = 0.0;
        for (facet, exponent) in self.shape.facets.iter().zip(self.specular_exponent) {
            let mut normal = facet.unit_normal(&view.sun_body);
            let mut cos_in = dot(normal, view.sun_body);
            if facet.double_sided && cos_in < 0.0 {
                normal = [-normal[0], -normal[1], -normal[2]];
                cos_in = -cos_in;
            }
            let cos_out = dot(normal, view.observer_body);
            if cos_in <= 0.0 || cos_out <= 0.0 {
                continue;
            }

            let diffuse = 1.0 - facet.absorption - facet.reflection;
            // Mirror direction of the incoming ray: r̂ = 2(n̂·ŝ)n̂ − ŝ.
            let mirror = [
                2.0 * cos_in * normal[0] - view.sun_body[0],
                2.0 * cos_in * normal[1] - view.sun_body[1],
                2.0 * cos_in * normal[2] - view.sun_body[2],
            ];
            let cos_lobe = dot(mirror, view.observer_body).max(0.0);
            let lobe = facet.reflection * (exponent + 1.0) / (2.0 * core::f64::consts::PI)
                * libm::pow(cos_lobe, exponent);
            let lambert = diffuse / core::f64::consts::PI * cos_out;
            intensity += irradiance * facet.area * cos_in * (lambert + lobe);
        }
        intensity
    }

    /// Apparent V magnitude above the atmosphere.
    ///
    /// `m = m_☉ − 2.5 log₁₀(E/E_☉)` with `E = I/d²` the irradiance the object delivers at the
    /// observer and `E_☉` the unshadowed solar constant — the ratio the Sun's own magnitude is the
    /// zero point of. Returns `f64::INFINITY`, infinitely faint, when nothing reflects: in umbra, or
    /// when every facet is lit on the side the observer cannot see.
    ///
    /// Atmospheric extinction is *not* applied; [`spacelab_coverage::observability::extinction`] is
    /// where that belongs, since it depends on the observer's air mass and not on the object.
    #[must_use]
    pub fn apparent_magnitude(&self, view: &ViewGeometry) -> f64 {
        let intensity = self.radiant_intensity(view);
        if intensity <= 0.0 {
            return f64::INFINITY;
        }
        let irradiance = intensity / (view.range * view.range);
        SUN_V_MAGNITUDE - 2.5 * libm::log10(irradiance / SOLAR_IRRADIANCE)
    }
}

/// The body's true rotation period about its spin axis, measured in an inertial frame, seconds.
///
/// Distinct from [`SynodicPeriod`] because they are different numbers and confusing them is a
/// silent error: a spin rate fitted from a light curve is synodic, and a dynamical model integrates
/// a sidereal one. See [`Self::to_synodic`] for what separates them.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SiderealPeriod(f64);

/// The period a light curve repeats with, seconds — what a fit to photometry actually returns.
///
/// It is not the body's rotation period: the illumination and viewing geometry turn too, and the
/// curve repeats when the body returns to the same *aspect*, not to the same inertial orientation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SynodicPeriod(f64);

impl SiderealPeriod {
    /// A period in seconds, or `None` if it is not positive and finite.
    #[must_use]
    pub fn new(seconds: f64) -> Option<Self> {
        (seconds.is_finite() && seconds > 0.0).then_some(Self(seconds))
    }

    /// The period in seconds.
    #[must_use]
    pub fn seconds(self) -> f64 {
        self.0
    }

    /// The synodic period this spin produces, given `bisector_rate`: the rate in rad·s⁻¹ at which
    /// the phase-angle bisector — the direction halfway between the Sun and the observer — turns
    /// about the spin axis, signed positive when it turns the same way the body does.
    ///
    /// `ω_syn = ω_sid − ω_pab`, the standard asteroid-photometry relation, for the same reason a
    /// solar day is longer than a sidereal one: the body has to turn a little further to present the
    /// same face to a geometry that has itself moved on.
    ///
    /// `None` when the two rates cancel, which is not a degenerate case but a real one — a body
    /// spinning at exactly the bisector rate shows the same aspect forever and its light curve has
    /// no period to find. The result carries no sense of direction, because the observable does not:
    /// photometry alone cannot say whether an apparent rotation runs forwards or backwards.
    #[must_use]
    pub fn to_synodic(self, bisector_rate: f64) -> Option<SynodicPeriod> {
        let relative = core::f64::consts::TAU / self.0 - bisector_rate;
        (relative.is_finite() && relative != 0.0)
            .then(|| SynodicPeriod(core::f64::consts::TAU / libm::fabs(relative)))
    }
}

impl SynodicPeriod {
    /// A period in seconds, or `None` if it is not positive and finite.
    #[must_use]
    pub fn new(seconds: f64) -> Option<Self> {
        (seconds.is_finite() && seconds > 0.0).then_some(Self(seconds))
    }

    /// The period in seconds.
    #[must_use]
    pub fn seconds(self) -> f64 {
        self.0
    }

    /// The sidereal period behind this observed one, for a body whose apparent rotation is
    /// **prograde** with respect to `bisector_rate` — the assumption photometry cannot make for
    /// itself, which is why it is stated in the argument rather than hidden in the type.
    #[must_use]
    pub fn to_sidereal(self, bisector_rate: f64) -> Option<SiderealPeriod> {
        let absolute = core::f64::consts::TAU / self.0 + bisector_rate;
        (absolute.is_finite() && absolute > 0.0)
            .then(|| SiderealPeriod(core::f64::consts::TAU / absolute))
    }
}
