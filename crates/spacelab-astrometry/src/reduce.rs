//! The last leg: a measured pixel becomes a *true* topocentric direction with an epoch and a
//! covariance — the thing an angles-only orbit determination is entitled to consume.
//!
//! # Do not double-correct
//!
//! A plate solution fitted against catalogue positions has **already absorbed** whatever part of
//! refraction and aberration is constant across the field, because the reference stars suffered the
//! same displacement as the target and the fit had no way to tell that from a pointing error. What
//! survives the fit is the *differential* across the field: refraction stretches a frame vertically
//! by `dR/dh`, which is 1.6 % at 20° altitude, and differential aberration is a scale change of
//! order `β ≈ 1e-4` — both of which a first-order SIP term absorbs as well. Applying the corrections
//! in this module on top of such a plate subtracts them twice.
//!
//! They belong to the other case, and it is a real one: the pointing comes from the mount or from a
//! separate boresight calibration rather than from stars in the same frame, or the plate was fitted
//! against *apparent* places and the reduction to catalogue place is being done explicitly. Which
//! case a caller is in is not something this crate can detect, so it is stated here rather than
//! guessed at.
//!
//! # Order of operations
//!
//! Going from what the telescope measured to where the object really was:
//!
//! 1. **Refraction**, removed first, because it acts on the ray *in the atmosphere* — the last thing
//!    to happen to the photon and therefore the first to undo.
//! 2. **Aberration**, removed second, because it is a property of the observer's motion and applies
//!    to the direction the ray had before it entered the atmosphere.
//! 3. **Light time** is *not* removed, and that is deliberate: an angles-only IOD wants the
//!    astrometric direction toward where the object *was seen*, and pairs it with the observer's
//!    position at the observation epoch. Removing light time here would produce a geometric
//!    direction that no orbit fit in this repository asks for.
//!
//! Going the other way — predicting where a known orbit will appear — is the same two steps in the
//! opposite order, and both functions are provided so that a residual can be formed in whichever
//! space the caller is working in.
//!
//! # The epoch is part of the measurement
//!
//! A satellite trails. A 10 s exposure on a LEO target draws a streak tens of arcminutes long, and
//! its two ends are separated by 10 s of orbital motion — 70 km along track. Reporting the streak's
//! *centroid* against the shutter-open time is a 5 s error, which is 35 km, which is larger than
//! anything else in the reduction by three orders of magnitude. [`Streak`] exists so that the epoch
//! cannot be separated from the endpoint it belongs to.
//!
//! # Gate
//!
//! `derivations/relativistic_aberration.mac` shows the coded boost obeys
//! `cos θ' = (cos θ + β)/(1 + β cos θ)` in the plane of the direction and the motion, and that the
//! boost by `−β` inverts it exactly; `derivations/relativistic_aberration.gp` recomputes its
//! directions as a Lorentz boost of the photon four-momentum. The gate is
//! `the_boost_is_the_relativistic_aberration_law` (G2) in `tests/derivation_gates.rs`. The
//! refraction removed here is `spacelab-coverage`'s fit; the tilt that applies it is pinned by
//! `tests/reduction.rs`.

use spacelab_core::{Epoch, Tt, Vec3};
use spacelab_coverage::observability::{ApparentAltitude, Refraction, TrueAltitude};

use crate::wcs::{Pixel, PlateSolution};

/// Speed of light in vacuum, m/s — exact by the SI definition of the metre.
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Normalise, or `None` if the vector has no direction.
fn normalize(v: Vec3) -> Option<Vec3> {
    let norm = v.norm();
    (norm > 0.0 && norm.is_finite()).then(|| v.scale(1.0 / norm))
}

/// Rodrigues rotation of `v` about the unit axis `axis` by `angle`, right-handed.
fn rotate(v: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let (sin, cos) = (libm::sin(angle), libm::cos(angle));
    v.scale(cos) + axis.cross(v).scale(sin) + axis.scale(axis.dot(v) * (1.0 - cos))
}

/// Rotate `direction` in the vertical plane through `up` by `angle`, positive **downward** —
/// toward the horizon.
///
/// `None` when the two are parallel, which is the zenith and the nadir. There the vertical plane is
/// undefined, and so is the direction the rotation would move in.
fn tilt(direction: Vec3, up: Vec3, angle: f64) -> Option<Vec3> {
    let axis = normalize(up.cross(direction))?;
    Some(rotate(direction, axis, angle))
}

/// Remove atmospheric refraction: the direction the telescope pointed at → the direction the ray
/// had outside the atmosphere.
///
/// `direction` and `up` need not be normalised. `None` below the apparent horizon, where
/// [`Refraction::geometric`] has no ray to unbend, and for a non-finite or zero-length input.
///
/// **At the zenith the direction is returned unchanged**, exactly. Both published fits leave a
/// residual there rather than vanishing — Bennett gives −0.081″ and Sæmundsson −0.116″ at `h = 90°`,
/// which is the fit's own error and not a refraction — and there is no azimuth in which to apply it,
/// since the vertical plane degenerates. Passing the fit residual through in an arbitrary azimuth
/// would be worse than dropping it, so it is dropped and said so.
#[must_use]
pub fn remove_refraction(direction: Vec3, up: Vec3, refraction: Refraction) -> Option<Vec3> {
    let (direction, up) = (normalize(direction)?, normalize(up)?);
    let apparent = ApparentAltitude(libm::asin(direction.dot(up).clamp(-1.0, 1.0)));
    let true_altitude = refraction.geometric(apparent)?;
    match tilt(direction, up, apparent.0 - true_altitude.0) {
        Some(bent) => normalize(bent),
        None => Some(direction),
    }
}

/// Apply atmospheric refraction: the direction a ray had outside the atmosphere → the direction the
/// telescope has to point at.
///
/// The inverse of [`remove_refraction`] only to the agreement between the two published fits, which
/// is a few arcseconds near the horizon and milliarcseconds above 15°. That is a property of the
/// fits, not of this code: Sæmundsson (1986) is fitted in the true altitude and Bennett (1982) in
/// the apparent one, and neither is the other's algebraic inverse.
#[must_use]
pub fn apply_refraction(direction: Vec3, up: Vec3, refraction: Refraction) -> Option<Vec3> {
    let (direction, up) = (normalize(direction)?, normalize(up)?);
    let true_altitude = TrueAltitude(libm::asin(direction.dot(up).clamp(-1.0, 1.0)));
    let apparent = refraction.apparent(true_altitude)?;
    match tilt(direction, up, true_altitude.0 - apparent.0) {
        Some(bent) => normalize(bent),
        None => Some(direction),
    }
}

/// Stellar aberration for an observer with a stated velocity.
///
/// Exact special relativity, not the first-order series: the map from a natural direction `n` to the
/// apparent one is
///
/// ```text
/// n' ∝ n/γ + β [1 + (n·β)/(1 + 1/γ)],      γ = 1/√(1 − β²)
/// ```
///
/// which is the action of the Lorentz boost on the celestial sphere, in the form SOFA's `iauAb`
/// uses. Because it *is* a boost, its inverse is the boost by `−β` and nothing has to be iterated —
/// [`Aberration::remove`] is as exact as [`Aberration::apply`].
///
/// # What velocity to pass
///
/// The observer's **total** velocity with respect to the frame the catalogue positions are in. For
/// an Earth-bound telescope against ICRS that is the barycentric velocity: the Earth's orbital
/// motion, 29.79 km/s, giving the classical constant of aberration 20.4955″, plus the observer's
/// diurnal rotation, at most 0.465 km/s at the equator and worth 0.32″. Both are the caller's to
/// supply; splitting them into separate "annual" and "diurnal" functions would invite adding the two
/// results, which is not what the relativistic formula does.
///
/// Omitted, and stated rather than hidden: the gravitational term `SRS/em` that SOFA's `iauAb`
/// carries alongside the boost, worth about 4 mas at one astronomical unit from the Sun. It is
/// below the noise of any plate this crate is aimed at and above the noise of a Gaia-grade one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aberration {
    beta: Vec3,
    /// `1/γ = √(1 − β²)`, stored because both directions need it.
    reciprocal_lorentz: f64,
}

impl Aberration {
    /// Construct from the observer's velocity in m/s.
    ///
    /// `None` for a non-finite velocity or one at or above the speed of light.
    #[must_use]
    pub fn new(velocity: Vec3) -> Option<Self> {
        let beta = velocity.scale(1.0 / SPEED_OF_LIGHT);
        let squared = beta.dot(beta);
        if !squared.is_finite() || squared >= 1.0 {
            return None;
        }
        Some(Self {
            beta,
            reciprocal_lorentz: libm::sqrt(1.0 - squared),
        })
    }

    /// The velocity as a fraction of the speed of light.
    #[must_use]
    pub fn beta(&self) -> Vec3 {
        self.beta
    }

    /// The classical constant of aberration for this velocity, `|β|` in radians — 9.9365e-5, or
    /// 20.4955″, for the Earth's mean orbital speed.
    #[must_use]
    pub fn constant(&self) -> f64 {
        self.beta.norm()
    }

    /// The **diurnal** term: an observer carried around by the rotation of the body they stand on,
    /// `v = ω × r`.
    ///
    /// Both vectors must be in the frame the directions are in. For a ground station observing in
    /// TEME that is the station's TEME position and `(0, 0, ω)` with `ω = 7.292_115_146_706_979e-5`
    /// rad/s, `spacelab_frames::EARTH_ROTATION_RATE` — not linked, because taking `ω` as a *vector*
    /// rather than a scalar rate is exactly what keeps this crate off `spacelab-frames`, and it is
    /// also what lets polar motion tilt the pole. The two constants are held equal by a test.
    ///
    /// # Why the diurnal term and not the annual one
    ///
    /// The Earth's 29.8 km/s orbital motion is the famous 20.5″ constant of aberration, and it does
    /// **not** apply to a satellite observation: the station and the satellite share the Earth's
    /// barycentric velocity, and a shared velocity cancels in a geocentric frame. What does not
    /// cancel is the station's own rotation about the Earth's axis — **465.1 m/s** at the equator,
    /// falling as `cos φ` — and that is worth **0.320″** at the equator and **0.262″** at 34.9°,
    /// which is a hundred times an astrometric plate's centroiding error and cannot be absorbed into
    /// a bias that averages out, because it turns with the hour angle.
    ///
    /// `None` for a velocity that is not finite or is at or above `c`, as [`Aberration::new`].
    #[must_use]
    pub fn diurnal(observer: Vec3, angular_velocity: Vec3) -> Option<Self> {
        Self::new(angular_velocity.cross(observer))
    }

    /// The boost by `−β`, which is the inverse map.
    #[must_use]
    fn reversed(&self) -> Self {
        Self {
            beta: -self.beta,
            reciprocal_lorentz: self.reciprocal_lorentz,
        }
    }

    /// Natural direction → apparent direction.
    #[must_use]
    pub fn apply(&self, natural: Vec3) -> Option<Vec3> {
        let natural = normalize(natural)?;
        let projection = natural.dot(self.beta);
        let weight = 1.0 + projection / (1.0 + self.reciprocal_lorentz);
        normalize(natural.scale(self.reciprocal_lorentz) + self.beta.scale(weight))
    }

    /// Apparent direction → natural direction.
    #[must_use]
    pub fn remove(&self, apparent: Vec3) -> Option<Vec3> {
        self.reversed().apply(apparent)
    }
}

/// A trailed image: two measured endpoints, each with the epoch that belongs to it.
///
/// The type exists to make the epoch inseparable from the endpoint. See the module documentation
/// for what separating them costs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Streak {
    start: Pixel,
    end: Pixel,
    open: Epoch<Tt>,
    close: Epoch<Tt>,
}

impl Streak {
    /// Construct from the endpoint measured at shutter open and the one measured at shutter close.
    ///
    /// `None` if the exposure does not run forward. Which endpoint is which is the caller's to
    /// determine — a single frame cannot say, and the usual discriminators are a second frame, a
    /// shutter chop, or a known direction of motion.
    #[must_use]
    pub fn new(start: Pixel, open: Epoch<Tt>, end: Pixel, close: Epoch<Tt>) -> Option<Self> {
        (close > open).then_some(Self {
            start,
            end,
            open,
            close,
        })
    }

    /// Exposure length, seconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // an exposure long enough to lose ns precision is 104 days
    pub fn duration(&self) -> f64 {
        (self.close.ns_since_j2000() - self.open.ns_since_j2000()) as f64 * 1.0e-9
    }

    /// The two endpoints, each paired with its own epoch, in exposure order.
    #[must_use]
    pub fn endpoints(&self) -> [(Pixel, Epoch<Tt>); 2] {
        [(self.start, self.open), (self.end, self.close)]
    }

    /// The midpoint of the streak, at the mid-exposure epoch.
    ///
    /// The pairing is the whole point: the mid-exposure epoch is the *only* one this position can be
    /// reported against, and only for motion that is linear over the exposure — which for a LEO
    /// target over 10 s is true to the second-order term, about 0.4 m for a 7 km/s object on a 6800
    /// km orbit, well under a pixel.
    #[must_use]
    pub fn midpoint(&self) -> (Pixel, Epoch<Tt>) {
        let mid = (self.open.ns_since_j2000() + self.close.ns_since_j2000()).div_euclid(2);
        (
            Pixel::new(
                f64::midpoint(self.start.x, self.end.x),
                f64::midpoint(self.start.y, self.end.y),
            ),
            Epoch::from_ns_since_j2000(mid),
        )
    }

    /// Apparent rate of motion, pixels per second, as `(dx/dt, dy/dt)`.
    #[must_use]
    pub fn rate(&self) -> (f64, f64) {
        let dt = self.duration();
        (
            (self.end.x - self.start.x) / dt,
            (self.end.y - self.start.y) / dt,
        )
    }
}

/// A reduced observation: a direction, when it was measured, and how well it is known.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Astrometric {
    /// Right ascension, radians, in the frame the plate was fitted in.
    pub right_ascension: f64,
    /// Declination, radians.
    pub declination: f64,
    /// The epoch of *this* measurement — for a streak endpoint, that endpoint's own epoch.
    pub epoch: Epoch<Tt>,
    /// Covariance in **great-circle** measure, rad², ordered `(Δα cos δ, Δδ)`.
    ///
    /// Great-circle rather than coordinate measure so that the two components are comparable and
    /// the trace is an angle: a covariance in `Δα` alone would grow as `sec δ` and report a
    /// circumpolar observation as poor when it is not.
    pub covariance: [[f64; 2]; 2],
}

impl Astrometric {
    /// The unit direction.
    #[must_use]
    pub fn direction(&self) -> Vec3 {
        let cos_dec = libm::cos(self.declination);
        Vec3::new(
            cos_dec * libm::cos(self.right_ascension),
            cos_dec * libm::sin(self.right_ascension),
            libm::sin(self.declination),
        )
    }

    /// One-sigma position error, radians — the square root of the covariance's trace, which is the
    /// radius of the error circle of equal area.
    #[must_use]
    pub fn position_sigma(&self) -> f64 {
        libm::sqrt(self.covariance[0][0] + self.covariance[1][1])
    }

    /// Reduce a measured pixel through a fitted plate.
    ///
    /// `centroid_sigma` is the target's *own* centroiding error in pixels — [`crate::sources::Source`]
    /// reports it as `position_sigma`. It is added in quadrature to the plate's contribution
    /// because the two are independent and both are real: the plate covariance from
    /// [`PlateSolution::position_covariance`] says how well the *transformation* is known at that
    /// pixel, and says nothing about how well the target was centroided. A reduction that reports
    /// only the plate term claims a bright reference field makes a faint target well measured.
    ///
    /// The tangent-plane covariance is taken as already being in great-circle measure. That is exact
    /// at the tangent point and rotates by `O(ξ tan δ)` away from it — 0.02 rad of field rotation
    /// one degree from a tangent point at 60° declination, which changes a covariance *eigenvalue*
    /// by nothing and its orientation by that angle. For a diagonal covariance, which is what the
    /// fit produces, the difference is second order.
    #[must_use]
    pub fn from_plate(
        solution: &PlateSolution,
        pixel: Pixel,
        epoch: Epoch<Tt>,
        centroid_sigma: f64,
    ) -> Self {
        let (right_ascension, declination) = solution.wcs.pixel_to_sky(pixel);
        let scale = solution.wcs.plate_scale();
        let centroid_variance = (centroid_sigma * scale) * (centroid_sigma * scale);
        let mut covariance = solution.position_covariance(pixel);
        covariance[0][0] += centroid_variance;
        covariance[1][1] += centroid_variance;
        Self {
            right_ascension,
            declination,
            epoch,
            covariance,
        }
    }

    /// Replace the direction with `direction`, keeping the epoch and covariance.
    ///
    /// The covariance is carried through unchanged, which is right for the corrections in this
    /// module: refraction and aberration are rotations of a few arcseconds, and a rotation that
    /// small changes a covariance by its own magnitude times the rotation — parts in `1e-5`.
    #[must_use]
    pub fn with_direction(self, direction: Vec3) -> Option<Self> {
        let direction = normalize(direction)?;
        Some(Self {
            right_ascension: libm::atan2(direction.y, direction.x),
            declination: libm::asin(direction.z.clamp(-1.0, 1.0)),
            ..self
        })
    }

    /// Undo refraction and then aberration, in that order — see the module documentation for why
    /// this is not always the right thing to call.
    ///
    /// `up` is the local vertical at the station, in the same frame as the plate.
    #[must_use]
    pub fn to_true_direction(
        self,
        up: Vec3,
        refraction: Refraction,
        aberration: &Aberration,
    ) -> Option<Self> {
        let unrefracted = remove_refraction(self.direction(), up, refraction)?;
        self.with_direction(aberration.remove(unrefracted)?)
    }
}
