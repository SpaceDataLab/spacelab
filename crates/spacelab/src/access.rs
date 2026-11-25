//! Access windows of a ground station on the WGS-84 ellipsoid, from an SGP4 element set (Ф31.C).
//!
//! This is the Rust API the two bridges return to the bit. It composes three pieces that exist
//! elsewhere: the satellite's ITRF position from [`teme_to_itrf_at`], the elevation constraint
//! [`ElevationMask`] about the station's geodetic normal, and the rate-bounded scan
//! [`spacelab_coverage::windows`].
//!
//! # The model
//!
//! A station is a geodetic latitude, longitude and height on WGS 84. Its zenith is the ellipsoid's
//! outward normal, `(cos φ cos λ, cos φ sin λ, sin φ)`, which is the normal at the foot of the
//! station for every height: `geodetic_to_itrf(φ, λ, h)` is the foot plus `h` times that normal
//! (`derivations/access_station_geometry.mac`, G1). The satellite is visible when its geometric
//! elevation above the plane normal to the zenith is at least the mask, `g = d·ẑ − sin ε_min |d| ≥ 0`
//! for the line of sight `d`. Refraction is the caller's: an apparent mask converts to one geometric
//! mask exactly (`spacelab_coverage::Refraction::geometric`).
//!
//! # The clock
//!
//! A scan runs on the naive UTC coordinate, seconds past the start label, which is the clock SGP4
//! itself measures from its epoch (`Satellite::minutes_since_epoch`). Every UTC day on it is
//! 86 400 s long, so a window that spans a leap second reads one second short, and a `23:59:60`
//! start label is the first second of the next day. Leap seconds end on 2027-05-20.
//!
//! # What a scan guarantees
//!
//! [`spacelab_coverage::windows`] misses no window at least its step floor long — a billionth of
//! the span, raised to four ulps of the bounds — **if** its rate bound bounds `|dg/dt|`. For a
//! station fixed in ITRF, `|dg/dt| ≤ (1 + |sin ε_min|)·|ṙ_ITRF|` (`derivations/step_bound.mac`).
//! For SGP4 the speed bound is a hypothesis, H-rate: the mean orbit's perigee speed plus the
//! Earth's rotation at its apogee, both from the element set, times [`SGP4_RATE_MARGIN`]. SGP4's
//! osculating speed differs from that mean orbit's by its short-period terms (about 0.2 % in low
//! orbit) and by the drag and luni-solar drift over the span; the margin covers both by a wide
//! factor, and `tests/bridge_access_reference.rs` samples `|dg/dt|` on every reference case and
//! holds it under the bound. It is tested there, not proved for SGP4's formulas.
//!
//! Each edge is a Brent root of `g`, to the abscissa tolerance the caller gives.

use core::cell::Cell;
use core::fmt;

use spacelab_core::{Angle, Length, UtcDateTime, Vec3};
use spacelab_coverage::windows;
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sensors::{Constraint, ElevationMask, OrbitBound};
use spacelab_sgp4::{PropagationError, Satellite};

use crate::{EarthOrientationError, EopSource, teme_to_itrf_at};

pub use spacelab_coverage::Window;

/// The factor H-rate puts on the mean orbit's speed bound for SGP4: ten per cent, fifty times the
/// short-period terms' share in low orbit.
pub const SGP4_RATE_MARGIN: f64 = 1.1;

/// The gravitational parameter SGP4's element sets are fitted under, WGS-72, m³/s².
const WGS72_MU: f64 = 3.986_008e14;

/// Nanoseconds in a second, as an exact `f64`.
const NS_PER_SECOND: f64 = 1e9;

/// Why a scan was refused. A refused scan returns no window.
#[derive(Debug, Clone, PartialEq)]
pub enum AccessError {
    /// A station's latitude is outside `[−π/2, π/2]`, or a coordinate is not finite.
    Station,
    /// The mask is not strictly inside `(−π/2, π/2)`.
    Mask,
    /// The span is not finite and positive, or its end is outside the calendar.
    Interval,
    /// The tolerance is not finite and positive.
    Tolerance,
    /// The Earth's orientation could not be established at a probed instant.
    EarthOrientation {
        /// Seconds past the start label.
        at_seconds: f64,
        /// What the frame chain refused.
        error: EarthOrientationError,
    },
    /// SGP4 refused a probed instant: the object decayed, or its elements left the model.
    Propagation {
        /// Seconds past the start label.
        at_seconds: f64,
        /// What SGP4 refused.
        error: PropagationError,
    },
}

impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Station => write!(
                f,
                "a station's latitude is outside [-pi/2, pi/2] or a coordinate is not finite"
            ),
            Self::Mask => write!(f, "the mask is not strictly inside (-pi/2, pi/2)"),
            Self::Interval => write!(
                f,
                "the span is not finite and positive, or it ends outside the calendar"
            ),
            Self::Tolerance => write!(f, "the tolerance is not finite and positive"),
            Self::EarthOrientation { at_seconds, error } => {
                write!(f, "at {at_seconds} s past the start: {error}")
            }
            Self::Propagation { at_seconds, error } => {
                write!(f, "at {at_seconds} s past the start: {error}")
            }
        }
    }
}

impl std::error::Error for AccessError {}

/// A ground station on the WGS-84 ellipsoid: its ITRF position and its geodetic zenith.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Station {
    position: Vec3,
    zenith: Vec3,
}

impl Station {
    /// The station at geodetic `latitude_rad`, `longitude_rad` and `height_m` above WGS 84.
    ///
    /// # Errors
    ///
    /// [`AccessError::Station`] if the latitude is outside `[−π/2, π/2]` or any coordinate is not
    /// finite.
    pub fn geodetic(
        latitude_rad: f64,
        longitude_rad: f64,
        height_m: f64,
    ) -> Result<Self, AccessError> {
        let half_pi = core::f64::consts::FRAC_PI_2;
        if !((-half_pi..=half_pi).contains(&latitude_rad)
            && longitude_rad.is_finite()
            && height_m.is_finite())
        {
            return Err(AccessError::Station);
        }
        let position = geodetic_to_itrf(Geodetic {
            latitude: Angle::from_radians(latitude_rad),
            longitude: Angle::from_radians(longitude_rad),
            height: Length::from_metres(height_m),
        })
        .vector;
        let (sin_lat, cos_lat) = libm::sincos(latitude_rad);
        let (sin_lon, cos_lon) = libm::sincos(longitude_rad);
        Ok(Self {
            position,
            zenith: Vec3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat),
        })
    }

    /// The station's ITRF position, metres.
    #[must_use]
    pub const fn position(&self) -> Vec3 {
        self.position
    }

    /// The outward normal of the ellipsoid at the station, unit.
    #[must_use]
    pub const fn zenith(&self) -> Vec3 {
        self.zenith
    }

    /// The elevation constraint `g = d·ẑ − sin ε_min |d|` of a satellite at ITRF `satellite`:
    /// positive above the mask. The function every scan here refines.
    #[must_use]
    pub fn g(&self, mask: &ElevationMask, satellite: Vec3) -> f64 {
        mask.g(&spacelab_sensors::AccessGeometry {
            satellite,
            cell: self.position,
            zenith: self.zenith,
        })
    }
}

/// The elevation mask of `mask_rad`, or [`AccessError::Mask`].
///
/// # Errors
///
/// [`AccessError::Mask`] if the mask is not strictly inside `(−π/2, π/2)`.
pub fn elevation_mask(mask_rad: f64) -> Result<ElevationMask, AccessError> {
    ElevationMask::new(mask_rad).map_err(|_| AccessError::Mask)
}

/// H-rate's bound on `|dg/dt|` for `satellite` under `mask`: the elevation constraint's rate bound
/// on the element set's mean orbit in the rotating frame, times [`SGP4_RATE_MARGIN`].
#[must_use]
pub fn sgp4_rate_bound(satellite: &Satellite, mask: &ElevationMask) -> f64 {
    let n = satellite.brouwer_mean_motion() / 60.0;
    let orbit = OrbitBound {
        semi_major_axis: libm::cbrt(WGS72_MU / (n * n)),
        eccentricity: satellite.eccentricity(),
        mu: WGS72_MU,
    };
    SGP4_RATE_MARGIN * mask.max_rate(&orbit, EARTH_ROTATION_RATE)
}

/// The satellite's ITRF position `t` seconds past `start` on the naive UTC coordinate: SGP4, then
/// TEME to ITRF with the Earth orientation `eop` names. The function every SGP4 scan evaluates.
///
/// # Errors
///
/// [`AccessError::Interval`] if the instant is outside the calendar,
/// [`AccessError::EarthOrientation`] where the frame chain refuses it, and
/// [`AccessError::Propagation`] where SGP4 does; the frame chain is asked first.
pub fn sgp4_itrf_position(
    satellite: &Satellite,
    eop: EopSource<'_>,
    start: UtcDateTime,
    t: f64,
) -> Result<Vec3, AccessError> {
    // Rounded, not truncated, so that a probe lands on the nanosecond nearest the scan's abscissa.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a validated span keeps t·1e9 far inside i128"
    )]
    let ns = start.naive_ns_since_j2000() + (t * NS_PER_SECOND).round() as i128;
    let label = UtcDateTime::from_naive_ns_since_j2000(ns, false).ok_or(AccessError::Interval)?;
    // The frame first: an instant the Earth's orientation cannot be given for is refused as that,
    // whatever SGP4 would have said there.
    let (transform, _) =
        teme_to_itrf_at(label, eop).map_err(|error| AccessError::EarthOrientation {
            at_seconds: t,
            error,
        })?;
    let teme = satellite
        .propagate_at(label)
        .map_err(|error| AccessError::Propagation {
            at_seconds: t,
            error,
        })?;
    Ok(transform.position(teme.position).vector)
}

/// Check the scan's span and tolerance, and that its end is on the calendar.
fn check_scan(start: UtcDateTime, span_seconds: f64, tol_seconds: f64) -> Result<(), AccessError> {
    if !(span_seconds.is_finite() && span_seconds > 0.0) {
        return Err(AccessError::Interval);
    }
    if !(tol_seconds.is_finite() && tol_seconds > 0.0) {
        return Err(AccessError::Tolerance);
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "a finite f64 times 1e9 saturates in the cast, it never wraps"
    )]
    let end = (span_seconds * NS_PER_SECOND).round() as i128;
    start
        .naive_ns_since_j2000()
        .checked_add(end)
        .and_then(|ns| UtcDateTime::from_naive_ns_since_j2000(ns, false))
        .map(|_| ())
        .ok_or(AccessError::Interval)
}

/// The windows on `[0, span_seconds]` past `start` in which `satellite` stands at least `mask_rad`
/// above the geodetic horizon of `station`, each edge refined to `tol_seconds`.
///
/// Times are seconds past `start` on the naive UTC coordinate (the module's clock). A window open
/// at the start or at the end of the span is clamped to it. The scan is
/// [`spacelab_coverage::windows`] with [`sgp4_rate_bound`]; the module documentation states what
/// it guarantees and under which hypothesis.
///
/// # Errors
///
/// [`AccessError::Mask`], [`AccessError::Interval`] or [`AccessError::Tolerance`] for the
/// arguments, before SGP4 runs; [`AccessError::Propagation`] or [`AccessError::EarthOrientation`]
/// for the first probed instant SGP4 or the frame chain refuses. A refused scan returns no window.
pub fn sgp4_windows(
    satellite: &Satellite,
    eop: EopSource<'_>,
    start: UtcDateTime,
    span_seconds: f64,
    station: &Station,
    mask_rad: f64,
    tol_seconds: f64,
) -> Result<Vec<Window>, AccessError> {
    let mask = elevation_mask(mask_rad)?;
    check_scan(start, span_seconds, tol_seconds)?;
    let refused: Cell<Option<AccessError>> = Cell::new(None);
    let g = |t: f64| match sgp4_itrf_position(satellite, eop, start, t) {
        Ok(r) => station.g(&mask, r),
        Err(e) => {
            // The first refusal is the one reported. Minus infinity makes the next step reach the
            // end of the span, so a refused scan stops at once instead of crawling at its floor.
            let first = refused.take().unwrap_or(e);
            refused.set(Some(first));
            f64::NEG_INFINITY
        }
    };
    let mut found = Vec::new();
    windows(
        g,
        sgp4_rate_bound(satellite, &mask),
        0.0,
        span_seconds,
        tol_seconds,
        |w| found.push(w),
    );
    match refused.into_inner() {
        Some(e) => Err(e),
        None => Ok(found),
    }
}
