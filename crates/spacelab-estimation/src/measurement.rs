//! Measurement models, written once over any [`Scalar`] so their partials come from automatic
//! differentiation — exact, and from the same source that computes the predicted value.
//!
//! A tracking geometry maps a satellite state to a scalar an instrument reads: a range, a
//! range-rate, an elevation. The orbit-determination partial `∂measurement/∂epoch_state` is then
//! `∂measurement/∂state(t) · Φ(t, t₀)` — the first factor from AD here, the second from the state
//! transition matrix (`spacelab_propagation::variational`). Finite-difference partials would cap the
//! whole solution at ~1e-8 (see [`spacelab_autodiff`]); these do not.
//!
//! The station position (and velocity, if the frame rotates) is a constant of the geometry. Work in
//! the frame where the observation is natural: an ITRF station is fixed, so its velocity is zero and
//! range-rate is `Δr·v_sat / ρ`; in an inertial frame the station moves and the caller supplies its
//! velocity.
//!
//! # Every observable here is instantaneous and geometric
//!
//! One state, one station, one instant, and a straight line between them. That is a *convention*,
//! not an approximation that happens to be small, and it is stated because a caller cannot tell
//! from a signature which convention they were handed:
//!
//! - **No light time — *here*.** The satellite is taken where it is at the reception epoch, not
//!   where it was when the signal left. Measured on the ISS day in
//!   `spacelab/tests/light_time_in_measurements.rs`: **51.8 m** on [`range`] at the horizon, and
//!   **5.30″** on the angles at every elevation. That is a choice of epoch and not a missing term,
//!   so these formulae stay as they are and [`observed`](crate::observed) makes the choice — it
//!   solves for the emission epoch, evaluates an observable there, and corrects the partials for
//!   the state's own effect on the flight time. Against Orekit's `Range` the modelled range closes
//!   from **62.5 m** to **6.6e-7 m** (`spacelab/tests/light_time_oracle.rs`).
//! - **No aberration.** These are coordinate directions in the working frame, not the directions an
//!   observer in motion sees. That correction belongs on the observation rather than on the model —
//!   see [`light_time`](crate::light_time) for why, and for the 0.32″ it is worth in a geocentric
//!   frame.
//! - **No troposphere, ionosphere or Shapiro delay.** Not in *these* functions, which are the
//!   geometry alone. Corrections attach through [`modifier`](crate::modifier), which lifts each
//!   observable here into a decorator chain: the Shapiro delay and measurement biases are
//!   implemented there, and the atmospheric models (`spacelab_gnss`) attach through its
//!   [`SlantDelay`](crate::modifier::SlantDelay) trait.

use spacelab_autodiff::Scalar;

/// A ground station: its position, and its velocity in the same frame (zero for a body-fixed frame).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Station {
    /// Position in the working frame, metres.
    pub position: [f64; 3],
    /// Velocity in the working frame, m/s. Zero in a frame that rotates with the station.
    pub velocity: [f64; 3],
}

impl Station {
    /// A station fixed in the working frame (velocity zero).
    #[must_use]
    pub const fn fixed(position: [f64; 3]) -> Self {
        Self {
            position,
            velocity: [0.0, 0.0, 0.0],
        }
    }
}

/// Line-of-sight `Δr = r_sat − r_station` and relative velocity `Δv = v_sat − v_station`, generic.
#[inline]
fn relative<S: Scalar>(sat: &[S; 6], station: &Station) -> ([S; 3], [S; 3]) {
    let dr = [
        sat[0] - S::from_f64(station.position[0]),
        sat[1] - S::from_f64(station.position[1]),
        sat[2] - S::from_f64(station.position[2]),
    ];
    let dv = [
        sat[3] - S::from_f64(station.velocity[0]),
        sat[4] - S::from_f64(station.velocity[1]),
        sat[5] - S::from_f64(station.velocity[2]),
    ];
    (dr, dv)
}

/// Slant range `ρ = |r_sat − r_station|`.
#[must_use]
pub fn range<S: Scalar>(sat: &[S; 6], station: &Station) -> S {
    let (dr, _) = relative(sat, station);
    (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt()
}

/// Range rate `ρ̇ = (Δr · Δv) / ρ` — the radial component of relative velocity, positive receding.
#[must_use]
pub fn range_rate<S: Scalar>(sat: &[S; 6], station: &Station) -> S {
    let (dr, dv) = relative(sat, station);
    let rho2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
    let dot = dr[0] * dv[0] + dr[1] * dv[1] + dr[2] * dv[2];
    dot * rho2.sqrt().recip()
}

/// Topocentric right ascension `α = atan2(Δy, Δx)` of the line of sight, in the working inertial
/// frame — the optical-survey angle observable, paired with [`declination`]. Depends only on the
/// line-of-sight direction, so unlike azimuth/elevation it needs no station orientation.
///
/// The value is cyclic in `2π`; a residual `O − C` must be wrapped to `(−π, π]` before it enters the
/// filter — that belongs to the residual model, not here.
#[must_use]
pub fn right_ascension<S: Scalar>(sat: &[S; 6], station: &Station) -> S {
    let (dr, _) = relative(sat, station);
    dr[1].atan2(dr[0])
}

/// Topocentric declination `δ = atan2(Δz, √(Δx² + Δy²))` of the line of sight, in `[−π/2, π/2]`.
#[must_use]
pub fn declination<S: Scalar>(sat: &[S; 6], station: &Station) -> S {
    let (dr, _) = relative(sat, station);
    let rho_xy = (dr[0] * dr[0] + dr[1] * dr[1]).sqrt();
    dr[2].atan2(rho_xy)
}

/// A station with its local horizon frame — the east, north and up unit vectors in the working
/// frame. Azimuth and elevation are angles *in this frame*, so unlike range or right ascension they
/// need the station's orientation, not merely its position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Topocentric {
    /// Station position in the working (Earth-fixed) frame, metres.
    pub position: [f64; 3],
    /// Local east unit vector.
    pub east: [f64; 3],
    /// Local north unit vector.
    pub north: [f64; 3],
    /// Local up (geodetic normal) unit vector.
    pub up: [f64; 3],
}

impl Topocentric {
    /// The horizon frame for a station at geodetic latitude/longitude (radians) fixed in an
    /// Earth-fixed working frame: up is the geodetic normal, east and north complete the
    /// right-handed triad. `position` (from the ellipsoid) is in the same frame.
    #[must_use]
    pub fn from_geodetic(position: [f64; 3], latitude: f64, longitude: f64) -> Self {
        let (sin_lat, cos_lat) = (libm::sin(latitude), libm::cos(latitude));
        let (sin_lon, cos_lon) = (libm::sin(longitude), libm::cos(longitude));
        Self {
            position,
            east: [-sin_lon, cos_lon, 0.0],
            north: [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            up: [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        }
    }
}

/// The line of sight in the station's local east/north/up components.
#[inline]
fn horizon<S: Scalar>(sat: &[S; 6], topo: &Topocentric) -> (S, S, S) {
    let dr = [
        sat[0] - S::from_f64(topo.position[0]),
        sat[1] - S::from_f64(topo.position[1]),
        sat[2] - S::from_f64(topo.position[2]),
    ];
    let project = |axis: &[f64; 3]| {
        dr[0] * S::from_f64(axis[0]) + dr[1] * S::from_f64(axis[1]) + dr[2] * S::from_f64(axis[2])
    };
    (project(&topo.east), project(&topo.north), project(&topo.up))
}

/// Elevation `el = atan2(u, √(e² + n²))` above the local horizon, in `[−π/2, π/2]`.
#[must_use]
pub fn elevation<S: Scalar>(sat: &[S; 6], topo: &Topocentric) -> S {
    let (e, n, u) = horizon(sat, topo);
    let horizontal = (e * e + n * n).sqrt();
    u.atan2(horizontal)
}

/// Azimuth `az = atan2(e, n)`, measured from local north toward east.
///
/// Cyclic in `2π`; a residual must be wrapped before it enters the filter, as for right ascension.
#[must_use]
pub fn azimuth<S: Scalar>(sat: &[S; 6], topo: &Topocentric) -> S {
    let (e, n, _) = horizon(sat, topo);
    e.atan2(n)
}
