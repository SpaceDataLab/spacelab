//! The beta angle and the eclipse statistics of a circular orbit (plan 18 Ф31.G): how high the Sun
//! stands above the orbit plane, what fraction of each revolution passes in the Earth's shadow, and
//! over a span of dates the eclipse seasons, the full-sun intervals between them and the longest
//! eclipse.
//!
//! **β.** The plane with node `Ω` and inclination `i` has the normal
//! `ĥ = (sin i sin Ω, −sin i cos Ω, cos i)`, and β is the Sun's elevation above it, `sin β = ĥ·ŝ`:
//! positive on the side `ĥ` points to, from which the satellite is seen to move anticlockwise, and in
//! `[−π/2, π/2]`. It is taken as `atan2(ĥ·S, |ĥ × S|)`, which keeps its digits at both poles. Over
//! time the plane turns about the pole of date at the node rate the caller gives
//! ([`CircularOrbit::node_rate`]: `KeplerianElements::nodal_regression_rate` for `J₂`), and the Sun is
//! [`AnalyticSun`]; `Ω` and `i` are on the J2000 mean equator and equinox, the frame that ephemeris
//! returns.
//!
//! **The cylinder.** A satellite at radius `r` is in the shadow of a body of radius `R` when it is on
//! the night side within `R` of the anti-Sun axis. On the circle that is `cos v > cos β*/cos β`, `v`
//! the angle from the anti-Sun point and `β* = asin(R/r)`, so the fraction of each revolution in
//! shadow is `f = acos(cos β*/cos β)/π` below `β*` and 0 at or above it: `β* = 64.30°` and
//! `f(0) = 0.35724` at 700 km. The code takes `f = atan2(√((ρ − s)(ρ + s)), cos β*)/π` with
//! `ρ = R/r` and `s = sin|β|` ([`eclipse_fraction`]), which needs no `β*` and does not cancel at the
//! onset; at `β = 0` it is `asin(ρ)/π`.
//!
//! **The onset.** `f` leaves 0 as a square root: with `d = β* − |β|`,
//! `πf = √(2 tan β*·d)·(1 − (5 − 2 cos²β*)/(12 sin β* cos β*)·d + O(d²))`. No error bound in `f` is
//! uniform there, so the bound is stated in β: an error of `δ` in β moves `f` by at most
//! `(κ/π)·√(2 tan β*·δ)`, `κ = (β*/2)/sin(β*/2) ≤ π/(2√2)`, at every β, and by at most
//! `(1/π)·√(2 tan β*·δ)` to first order at the onset. At 700 km (`κ = 1.0545`,
//! `tan β* = 2.078`) the analytic Sun's `0.0132°` is at most 0.0104 of `f`, 62 s of the 98.8 min
//! revolution, and only near the onset; the computed `f` is the exact one at a β within
//! `8u·tan β*` of the input, `u = 2⁻⁵³` (the test `the_fraction_is_exact_at_a_beta_within_its_band`).
//!
//! **The cones.** The Sun is a disc. From a point on a line tangent to both the Earth and the Sun the
//! two discs touch, so the umbra's edge is the cone of their external tangents and the penumbra's the
//! cone of their internal ones, the edges of [`crate::shadow::lighting_ratio`]. On the circle each
//! cone is the cylinder's formula with its own onset, `φ_u = β* − asin((R_S − R)/D)` and
//! `φ_p = β* + asin((R_S + R)/D)`, `D` the Sun's distance, so the umbra and the penumbra arcs
//! bracket the cylinder's: `f_u ≤ f ≤ f_p` ([`conical_eclipse`]). At 1 au the onsets are
//! `0.2640°` below and `0.2689°` above `β*`; at 700 km and `β = 0` the cylinder's eclipse is 8.7 s
//! longer than the umbra and 8.9 s shorter than the penumbra, and at `β = β*`, where the cylinder
//! has none, the penumbra lasts 264 s. By the same bound with `φ` for β,
//! `|f_p − f| ≤ (κ/π)·√(2 tan φ_p·(φ_p − β*))`, 0.0472 of a revolution (280 s) at 700 km. The
//! orbit mean of the hidden part of the Sun's disc, `1 − ν`, which lies between the two, is the
//! integral of `lighting_ratio` across the penumbra arcs, by Gauss–Legendre at 40 points after the
//! substitution that makes their `3/2`-power ends analytic (`dsst::gaussian::smoothstep_nodes`).
//!
//! **Seasons.** Over a window [`eclipse_seasons`] returns the intervals with `|β| ≤ β*`, on which
//! every revolution has an eclipse, and the full-sun ones between them, from the rate-bounded scan
//! [`spacelab_coverage::windows`] on `g = ρ − |ĥ·ŝ|`. Its rate bound is `|Ω̇|·|p × ĥ|` for the plane,
//! `p` the pole of date, plus [`SUN_DIRECTION_RATE`] for the Sun, so the scan misses no season or
//! full-sun interval longer than its floor, a billionth of the window. The longest eclipse is
//! `f·T` at the least `|β|` of the window, `T` the orbit's [`CircularOrbit::period`]; that least
//! `|β|` is 0 when β changes sign and otherwise comes from a Lipschitz branch and bound in time with
//! the same rate, so the longest eclipse is an [`Enclosure`] whose upper end is the safe side for a
//! battery.
//!
//! What the numbers rest on: the circular orbit, the mean plane (the node's short periods are not
//! in it), the cylinder or the two cones of a spherical Earth, [`AnalyticSun`] to `0.0132°`, and a
//! pole of date fixed at the epoch, which moves `20″` a year. Every value is **under H-libm, not
//! outward-rounded** ([`EclipseSeasons::LABEL`]).
//!
//! Derived in `derivations/eclipse_beta.mac` (55 identities: the normal and β, the shadow condition
//! on the circle, the `atan2` form, `β = 0` and the onset, the series and the Hölder bound, the cones
//! from their tangent lines, the Sun's rate bound, the numbers above) and re-derived at 60 digits in
//! `derivations/eclipse_beta.gp`: the half-arcs as roots of the shadow conditions themselves, the
//! cones' from `lighting_ratio`'s separation and apparent radii, and the mean of `1 − ν` by `intnum`
//! of the textbook lens; it writes the rows `tests/eclipse.rs` holds the code to.
//!
//! This is design-time analysis (it allocates), so it lives behind the crate's `std` feature.

use core::cmp::Ordering;
use core::f64::consts::{FRAC_PI_2, PI};
use std::collections::BinaryHeap;
use std::vec::Vec;

use libm::{atan2, cos, fabs, sin, sqrt};
use spacelab_coverage::{Window, find_events_rate_bounded, windows};

use crate::dsst::gaussian::{Rule, smoothstep_nodes};
use crate::ephemeris::{AnalyticSun, Ephemeris, j2000_from_of_date};
use crate::shadow::{SUN_RADIUS, lighting_ratio};

/// Seconds in a Julian century.
const JULIAN_CENTURY: f64 = 36_525.0 * 86_400.0;
/// The unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;
/// The rounding allowance on `ĥ·ŝ`, a dot product of two vectors of unit length to a few `u`.
const GAMMA: f64 = 16.0 * U;
/// Sub-intervals the branch and bound for the least `|β|` starts from.
const START_CELLS: u32 = 16;
/// Instants the branch and bound may evaluate before it reports what it has.
const BUDGET: usize = 1 << 16;

/// A bound on the angular rate of [`AnalyticSun`]'s direction within a Julian century of J2000:
/// 37 300° per Julian century, in rad/s. The series and the precession that turns it give at most
/// 37 231.02° (`derivations/eclipse_beta.mac` §9), within 0.2 % of the largest series rate, at
/// perihelion; the test `the_sun_turns_no_faster_than_its_bound` samples it.
pub const SUN_DIRECTION_RATE: f64 = 37_300.0 * (PI / 180.0) / JULIAN_CENTURY;

/// Why an eclipse statistic was not computed: input outside the model's domain is refused, never
/// answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EclipseError {
    /// A radius is not finite and positive, or the orbit is not above the body: `r ≤ R`.
    BadRadius,
    /// β is not finite, or is outside `[−π/2, π/2]`.
    BadBeta,
    /// The node is not finite, or the inclination is outside `[0, π]`.
    BadPlane,
    /// The Sun's position is not finite, or is the origin.
    BadSun,
    /// The Sun's distance is not finite, or does not exceed `R + R_S`, `r + R_S` and `2 R_S`: the
    /// Sun would touch the body or the orbit, or `lighting_ratio` would read the frame as centred on
    /// the Sun.
    BadSunDistance,
    /// The node rate is not finite, or the period is not finite and positive.
    BadRates,
    /// An instant or a window end is not finite, or the window does not have `t1 > t0`.
    BadInterval,
    /// An instant is more than a Julian century from J2000, where [`AnalyticSun`] was not measured
    /// and [`SUN_DIRECTION_RATE`] not derived.
    OutsideSunSpan,
    /// The tolerance is not finite and positive.
    BadTolerance,
}

impl core::fmt::Display for EclipseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::BadRadius => {
                "a radius is not finite and positive, or the orbit is not above the body"
            }
            Self::BadBeta => "beta is not finite or not in [-pi/2, pi/2]",
            Self::BadPlane => "the node is not finite or the inclination is not in [0, pi]",
            Self::BadSun => "the Sun's position is not finite or is the origin",
            Self::BadSunDistance => {
                "the Sun's distance does not clear the body, the orbit and twice the Sun's radius"
            }
            Self::BadRates => {
                "the node rate is not finite, or the period is not finite and positive"
            }
            Self::BadInterval => "an instant is not finite, or the window does not have t1 > t0",
            Self::OutsideSunSpan => "an instant is more than a Julian century from J2000",
            Self::BadTolerance => "the tolerance is not finite and positive",
        })
    }
}

impl core::error::Error for EclipseError {}

/// The β angle of the plane with node `raan` and inclination `inclination` (radians, `[0, π]`), for
/// the Sun at `sun` (any frame the plane is given in, any non-zero length): radians in
/// `[−π/2, π/2]`, positive on the side of the orbit normal (module docs).
///
/// # Errors
/// [`EclipseError::BadPlane`] and [`EclipseError::BadSun`].
pub fn beta_angle(raan: f64, inclination: f64, sun: [f64; 3]) -> Result<f64, EclipseError> {
    let normal = orbit_normal(raan, inclination)?;
    let length = norm(sun);
    if !(length.is_finite() && length > 0.0) {
        return Err(EclipseError::BadSun);
    }
    Ok(atan2(dot(normal, sun), norm(cross(normal, sun))))
}

/// `β* = asin(R/r)`, radians: the β angle at and above which a circular orbit of radius
/// `orbit_radius` about a body of radius `body_radius` is in full sun on every revolution, by the
/// cylinder (64.30° at 700 km above the Earth's equatorial radius).
///
/// # Errors
/// [`EclipseError::BadRadius`].
pub fn critical_beta(orbit_radius: f64, body_radius: f64) -> Result<f64, EclipseError> {
    let onset = cylinder_onset(orbit_radius, body_radius)?;
    Ok(atan2(onset.sin, onset.cos))
}

/// The fraction of each revolution a circular orbit of radius `orbit_radius` spends in the
/// cylindrical shadow of a body of radius `body_radius`, at β angle `beta` (radians):
/// `acos(cos β*/cos β)/π` below `β* = asin(R/r)` and 0 at or above it (module docs).
///
/// # Errors
/// [`EclipseError::BadRadius`] and [`EclipseError::BadBeta`].
pub fn eclipse_fraction(
    orbit_radius: f64,
    body_radius: f64,
    beta: f64,
) -> Result<f64, EclipseError> {
    let onset = cylinder_onset(orbit_radius, body_radius)?;
    let s = sin(check_beta(beta)?);
    Ok(half_arc(onset, s) / PI)
}

/// The eclipse of a circular orbit in the two cones of a spherical body and the Sun.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConicalEclipse {
    /// The fraction of each revolution with the whole Sun hidden.
    pub umbra: f64,
    /// The fraction with some of the Sun hidden, the umbra included.
    pub penumbra: f64,
    /// The orbit mean of the hidden part of the Sun's disc, `1 − ν`, `ν` the
    /// [`lighting_ratio`]: between `umbra` and `penumbra`.
    pub mean_shadow: f64,
    /// `φ_u`, radians: an umbra arc exists while `|β| < φ_u`; at or below 0 the orbit is beyond the
    /// umbra's vertex and never in it.
    pub umbra_onset: f64,
    /// `φ_p`, radians: some shadow while `|β| < φ_p`, full sun at or above it.
    pub penumbra_onset: f64,
}

/// The [`ConicalEclipse`] of a circular orbit of radius `orbit_radius` about a body of radius
/// `body_radius`, the Sun (radius [`SUN_RADIUS`]) at distance `sun_distance` and β angle `beta`
/// (metres and radians; module docs).
///
/// # Errors
/// [`EclipseError::BadRadius`], [`EclipseError::BadBeta`] and [`EclipseError::BadSunDistance`].
pub fn conical_eclipse(
    orbit_radius: f64,
    body_radius: f64,
    sun_distance: f64,
    beta: f64,
) -> Result<ConicalEclipse, EclipseError> {
    let cylinder = cylinder_onset(orbit_radius, body_radius)?;
    let abs_beta = check_beta(beta)?;
    let d = sun_distance;
    if !(d.is_finite()
        && d > body_radius + SUN_RADIUS
        && d > 2.0 * SUN_RADIUS
        && d > orbit_radius + SUN_RADIUS)
    {
        return Err(EclipseError::BadSunDistance);
    }
    let r = body_radius;
    // The cones' half-angles: sin θ = (R_S ∓ R)/D, cos θ with 1 − sin²θ factored.
    let umbra_cone = Onset {
        sin: (SUN_RADIUS - r) / d,
        cos: sqrt(((d - SUN_RADIUS + r) / d) * ((d + SUN_RADIUS - r) / d)),
    };
    let penumbra_cone = Onset {
        sin: (SUN_RADIUS + r) / d,
        cos: sqrt(((d - SUN_RADIUS - r) / d) * ((d + SUN_RADIUS + r) / d)),
    };
    // φ_u = β* − θ_u and φ_p = β* + θ_p, by their sines and cosines.
    let umbra_onset = Onset {
        sin: cylinder.sin * umbra_cone.cos - cylinder.cos * umbra_cone.sin,
        cos: cylinder.cos * umbra_cone.cos + cylinder.sin * umbra_cone.sin,
    };
    let penumbra_onset = Onset {
        sin: cylinder.sin * penumbra_cone.cos + cylinder.cos * penumbra_cone.sin,
        cos: cylinder.cos * penumbra_cone.cos - cylinder.sin * penumbra_cone.sin,
    };
    let s = sin(abs_beta);
    let (umbra, penumbra) = (half_arc(umbra_onset, s), half_arc(penumbra_onset, s));
    let mut lost = umbra;
    if penumbra > umbra {
        let sun = [d * cos(beta), 0.0, d * sin(beta)];
        let partial: f64 = smoothstep_nodes(Rule::P40, umbra, penumbra)
            .map(|(v, w)| {
                let satellite = [-orbit_radius * cos(v), orbit_radius * sin(v), 0.0];
                w * (1.0 - lighting_ratio(satellite, sun, r))
            })
            .sum();
        lost += partial;
    }
    Ok(ConicalEclipse {
        umbra: umbra / PI,
        penumbra: penumbra / PI,
        mean_shadow: lost / PI,
        umbra_onset: atan2(umbra_onset.sin, umbra_onset.cos),
        penumbra_onset: atan2(penumbra_onset.sin, penumbra_onset.cos),
    })
}

/// A circular orbit whose plane turns about the pole of date at a constant rate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircularOrbit {
    /// The orbit radius, metres.
    pub radius: f64,
    /// The inclination on the J2000 mean equator, radians, in `[0, π]`.
    pub inclination: f64,
    /// The right ascension of the ascending node at `t = 0`, the Sun's epoch, from the J2000 mean
    /// equinox, radians.
    pub raan: f64,
    /// The node's rate, rad/s: `KeplerianElements::nodal_regression_rate` for the `J₂` secular
    /// rate, or any mean rate.
    pub node_rate: f64,
    /// One revolution of the argument of latitude, seconds: `2π` over the draconitic mean motion,
    /// or the Keplerian period without `J₂`. An eclipse lasts `f` times it.
    pub period: f64,
}

/// A value known to lie between two ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Enclosure {
    /// The lower end.
    pub lower: f64,
    /// The upper end.
    pub upper: f64,
}

/// The eclipse seasons of a circular orbit over a window (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct EclipseSeasons {
    /// The intervals with `|β| ≤ β*`, on which every revolution has an eclipse, in increasing
    /// time; one open at an end of the window is cut there.
    pub seasons: Vec<Window>,
    /// The intervals between them, `|β| ≥ β*`: full sun on every revolution.
    pub full_sun: Vec<Window>,
    /// The least `|β|` over the window, radians.
    pub least_beta: Enclosure,
    /// The longest eclipse in the window, seconds: the cylinder's `f` at the least `|β|` times the
    /// period. The upper end is the safe side for a battery.
    pub longest_eclipse: Enclosure,
}

impl EclipseSeasons {
    /// What the values rest on: the `libm` elementary functions within their measured error
    /// (H-libm), and round-to-nearest arithmetic rather than outward rounding.
    pub const LABEL: &'static str = "under H-libm, not outward-rounded";
}

/// The β angle of `orbit` at `t` seconds past `sun`'s epoch, radians (module docs).
///
/// # Errors
/// [`EclipseError::BadPlane`], [`EclipseError::BadRates`], [`EclipseError::BadInterval`] for a `t`
/// that is not finite, and [`EclipseError::OutsideSunSpan`].
pub fn beta_at(orbit: &CircularOrbit, sun: AnalyticSun, t: f64) -> Result<f64, EclipseError> {
    let plane = Plane::new(orbit, sun)?;
    if !t.is_finite() {
        return Err(EclipseError::BadInterval);
    }
    if !within_span(sun, t) {
        return Err(EclipseError::OutsideSunSpan);
    }
    let (normal, s) = (plane.normal_at(t), sun.position(t));
    Ok(atan2(dot(normal, s), norm(cross(normal, s))))
}

/// The eclipse seasons, the full-sun intervals and the longest eclipse of `orbit` about a body of
/// radius `body_radius` on `[t0, t1]`, seconds past `sun`'s epoch: the season edges to `tol` seconds
/// and the longest eclipse to within `tol` seconds, or wider after 65 536 instants (module docs).
///
/// # Errors
/// [`EclipseError::BadRadius`], [`EclipseError::BadPlane`], [`EclipseError::BadRates`],
/// [`EclipseError::BadInterval`], [`EclipseError::OutsideSunSpan`] and
/// [`EclipseError::BadTolerance`].
pub fn eclipse_seasons(
    orbit: &CircularOrbit,
    body_radius: f64,
    sun: AnalyticSun,
    t0: f64,
    t1: f64,
    tol: f64,
) -> Result<EclipseSeasons, EclipseError> {
    let onset = cylinder_onset(orbit.radius, body_radius)?;
    let plane = Plane::new(orbit, sun)?;
    if !(t0.is_finite() && t1.is_finite() && t1 > t0) {
        return Err(EclipseError::BadInterval);
    }
    if !(within_span(sun, t0) && within_span(sun, t1)) {
        return Err(EclipseError::OutsideSunSpan);
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(EclipseError::BadTolerance);
    }
    let sine = |t: f64| {
        let s = sun.position(t);
        dot(plane.normal_at(t), s) / norm(s)
    };
    let rate = fabs(orbit.node_rate) * plane.tilt + SUN_DIRECTION_RATE;

    let mut seasons = Vec::new();
    windows(
        |t| onset.sin - fabs(sine(t)),
        rate,
        t0,
        t1,
        tol,
        |w| {
            seasons.push(w);
        },
    );
    let full_sun = complement(&seasons, t0, t1);

    let mut crossings = 0_u32;
    find_events_rate_bounded(sine, rate, t0, t1, tol, |_| crossings += 1)
        .map_err(|_| EclipseError::BadInterval)?;
    let duration = |s: f64| half_arc(onset, s) / PI * orbit.period;
    let (least, most) = if crossings > 0 || sine(t0) == 0.0 {
        (0.0, 0.0)
    } else {
        least_abs(
            |t| fabs(sine(t)),
            rate,
            t0,
            t1,
            |lo, hi| duration(lo) - duration(hi) <= tol,
        )
    };
    Ok(EclipseSeasons {
        seasons,
        full_sun,
        least_beta: Enclosure {
            lower: arcsine(least),
            upper: arcsine(most),
        },
        longest_eclipse: Enclosure {
            lower: duration(most),
            upper: duration(least),
        },
    })
}

/// An onset angle by its sine and cosine.
#[derive(Clone, Copy, Debug)]
struct Onset {
    sin: f64,
    cos: f64,
}

/// `β* = asin(R/r)` as `(R/r, √(1 − (R/r)²))`, the second from `(r − R)/r` and `(r + R)/r`, which
/// neither overflow nor cancel near the surface.
fn cylinder_onset(orbit_radius: f64, body_radius: f64) -> Result<Onset, EclipseError> {
    let (big, small) = (orbit_radius, body_radius);
    if !(small.is_finite() && small > 0.0 && big.is_finite() && big > small) {
        return Err(EclipseError::BadRadius);
    }
    Ok(Onset {
        sin: small / big,
        cos: sqrt(((big - small) / big) * ((big + small) / big)),
    })
}

/// `|β|`, refused outside `[−π/2, π/2]`.
fn check_beta(beta: f64) -> Result<f64, EclipseError> {
    let a = fabs(beta);
    if a <= FRAC_PI_2 {
        Ok(a)
    } else {
        Err(EclipseError::BadBeta)
    }
}

/// Half the arc of the circle within the onset angle `φ` of the anti-Sun axis, radians, for
/// `s = sin|β|`: `acos(cos φ/cos β)` as `atan2(√((sin φ − s)(sin φ + s)), cos φ)`. When
/// `s ≥ sin φ` the circle is all outside (`cos φ ≥ 0`) or all inside (`cos φ < 0`, a penumbra
/// cone that reaches past the terminator for an orbit tens of metres above the surface).
fn half_arc(onset: Onset, s: f64) -> f64 {
    if s >= onset.sin {
        return if onset.cos < 0.0 { PI } else { 0.0 };
    }
    atan2(sqrt((onset.sin - s) * (onset.sin + s)), onset.cos)
}

/// `asin(s)` for `s ∈ [0, 1]`, as `atan2(s, √((1 − s)(1 + s)))`.
fn arcsine(s: f64) -> f64 {
    atan2(s, sqrt((1.0 - s) * (1.0 + s)))
}

/// The orbit plane over time: its normal at `t = 0`, the pole of date it turns about, and the sine
/// of the angle between them.
struct Plane {
    normal: [f64; 3],
    pole: [f64; 3],
    pole_cross_normal: [f64; 3],
    pole_dot_normal: f64,
    tilt: f64,
    node_rate: f64,
}

impl Plane {
    fn new(orbit: &CircularOrbit, sun: AnalyticSun) -> Result<Self, EclipseError> {
        let normal = orbit_normal(orbit.raan, orbit.inclination)?;
        if !(orbit.node_rate.is_finite() && orbit.period.is_finite() && orbit.period > 0.0) {
            return Err(EclipseError::BadRates);
        }
        if !sun.epoch_j2000_seconds.is_finite() {
            return Err(EclipseError::BadInterval);
        }
        let pole = j2000_from_of_date(sun.epoch_j2000_seconds / JULIAN_CENTURY, [0.0, 0.0, 1.0]);
        let pole_cross_normal = cross(pole, normal);
        Ok(Self {
            normal,
            pole,
            pole_cross_normal,
            pole_dot_normal: dot(pole, normal),
            tilt: norm(pole_cross_normal),
            node_rate: orbit.node_rate,
        })
    }

    /// The normal turned by `Ω̇t` about the pole (Rodrigues): exactly the epoch's normal at `t = 0`.
    fn normal_at(&self, t: f64) -> [f64; 3] {
        let angle = self.node_rate * t;
        let (s, c) = (sin(angle), cos(angle));
        let along = self.pole_dot_normal * (1.0 - c);
        core::array::from_fn(|k| {
            self.normal[k] * c + self.pole_cross_normal[k] * s + self.pole[k] * along
        })
    }
}

/// The normal of the plane with node `raan` and inclination `inclination`.
fn orbit_normal(raan: f64, inclination: f64) -> Result<[f64; 3], EclipseError> {
    if !(raan.is_finite() && (0.0..=PI).contains(&inclination)) {
        return Err(EclipseError::BadPlane);
    }
    let (si, ci) = (sin(inclination), cos(inclination));
    Ok([si * sin(raan), -si * cos(raan), ci])
}

/// Whether `t` seconds past the Sun's epoch is within a Julian century of J2000.
fn within_span(sun: AnalyticSun, t: f64) -> bool {
    fabs(sun.epoch_j2000_seconds + t) <= JULIAN_CENTURY
}

/// `[t0, t1]` less the `seasons`, which are in increasing time.
fn complement(seasons: &[Window], t0: f64, t1: f64) -> Vec<Window> {
    let mut out = Vec::new();
    let mut start = t0;
    for w in seasons {
        if w.rise > start {
            out.push(Window {
                rise: start,
                set: w.rise,
            });
        }
        start = w.set;
    }
    if t1 > start {
        out.push(Window {
            rise: start,
            set: t1,
        });
    }
    out
}

/// A sub-interval of the branch and bound, ordered so that the one with the least lower bound is
/// popped first.
#[derive(Clone, Copy, Debug)]
struct Cell {
    bound: f64,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
}

impl Cell {
    /// The least an `L`-Lipschitz function can reach on `[a, b]` from its end values, the crossing
    /// of the two cones of slope `L`, less the rounding allowance.
    fn new(a: f64, b: f64, fa: f64, fb: f64, rate: f64) -> Self {
        let bound = (0.5 * (fa + fb - rate * (b - a)) - GAMMA).max(0.0);
        Self {
            bound,
            a,
            b,
            fa,
            fb,
        }
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Cell {}
impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        other.bound.total_cmp(&self.bound)
    }
}

/// An enclosure `(lower, upper)` of the least value of `f`, `|df/dt| ≤ rate`, on `[t0, t1]`, by
/// branch and bound: it stops when `settled(lower, upper)` holds, or after [`BUDGET`] instants with
/// a wider one. The upper end is the least sample plus the rounding allowance.
fn least_abs<F, S>(f: F, rate: f64, t0: f64, t1: f64, settled: S) -> (f64, f64)
where
    F: Fn(f64) -> f64,
    S: Fn(f64, f64) -> bool,
{
    let mut heap = BinaryHeap::new();
    let step = (t1 - t0) / f64::from(START_CELLS);
    let mut a = t0;
    let mut fa = f(a);
    let mut best = fa;
    for k in 1..=START_CELLS {
        let b = if k == START_CELLS {
            t1
        } else {
            t0 + f64::from(k) * step
        };
        let fb = f(b);
        best = best.min(fb);
        heap.push(Cell::new(a, b, fa, fb, rate));
        (a, fa) = (b, fb);
    }
    let mut evaluations = START_CELLS as usize + 1;
    let floor = 4.0 * f64::EPSILON * fabs(t0).max(fabs(t1));
    let mut dropped = f64::INFINITY;
    loop {
        let lower = heap.peek().map_or(f64::INFINITY, |c| c.bound).min(dropped);
        let upper = best + GAMMA;
        if lower >= best || settled(lower, upper) || evaluations >= BUDGET {
            return (lower.min(best), upper);
        }
        let Some(cell) = heap.pop() else {
            return (lower.min(best), upper);
        };
        if cell.b - cell.a <= floor {
            dropped = dropped.min(cell.bound);
            continue;
        }
        let m = f64::midpoint(cell.a, cell.b);
        let fm = f(m);
        evaluations += 1;
        best = best.min(fm);
        heap.push(Cell::new(cell.a, m, cell.fa, fm, rate));
        heap.push(Cell::new(m, cell.b, fm, cell.fb, rate));
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    sqrt(dot(v, v))
}
