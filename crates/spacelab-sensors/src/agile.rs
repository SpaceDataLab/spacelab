//! Agile optical instrument: a boresight that can slew off-nadir up to a limit.

use crate::cull::{CullBody, Reach};
use crate::{
    AccessGeometry, And, Boresight, Cone, Constraint, ElevationMask, OrbitBound, SensorError,
};

/// An agile optical instrument that can point its boresight up to `max_off_nadir` from the geocentric
/// nadir. A cell is accessible when its off-nadir angle is within that slew limit **and** it is above
/// the local horizon.
///
/// The horizon term is not redundant. A cell on the *far* side of the Earth has a small off-nadir
/// angle too — its direction from the satellite is close to nadir — so the slew limit alone would
/// accept it (a bare [`Cone`] "sees through the Earth"). Only the horizon test rejects the far side.
/// So this is exactly `And(Cone(max_off_nadir), ElevationMask::HORIZON)`, named for what a mission
/// analyst specifies: a maximum pointing angle, not a field-of-view half-angle.
///
/// [`AgileOptical::new`] measures the off-nadir from the *geocentric* nadir, as [`Cone::new`] does.
/// A nadir-pointing attitude built on the local vertical holds the geodetic one instead, and the two
/// differ by up to **0.177 125°** at 550 km — **1.70 km** on the ground straight below, and more off
/// nadir. [`AgileOptical::pointed`] takes the axis explicitly, which an *agile* instrument needs
/// anyway: this crate could not express a commanded pointing at all before, so "slewable" was a name
/// rather than a capability. Use `spacelab_earth::geodetic_nadir` for the geodetic axis, or the
/// commanded boresight for a real slew.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgileOptical {
    cone: Cone,
}

impl AgileOptical {
    /// Build from the maximum off-nadir slew angle in radians, inside `(0, π/2]`, about the
    /// geocentric nadir.
    ///
    /// # Errors
    ///
    /// [`SensorError::HalfAngleOutOfRange`] outside `(0, π/2]`, as [`Cone::new`].
    pub fn new(max_off_nadir: f64) -> Result<Self, SensorError> {
        Self::pointed(max_off_nadir, Boresight::GeocentricNadir)
    }

    /// The same instrument about an explicit [`Boresight`].
    ///
    /// # Errors
    ///
    /// As [`Cone::pointed`]: [`SensorError::HalfAngleOutOfRange`] outside `(0, π/2]`, and
    /// [`SensorError::DegenerateBoresight`] for an axis with no length.
    pub fn pointed(max_off_nadir: f64, boresight: Boresight) -> Result<Self, SensorError> {
        Ok(Self {
            cone: Cone::pointed(max_off_nadir, boresight)?,
        })
    }

    /// The axis the slew limit is measured about.
    #[must_use]
    pub const fn boresight(self) -> Boresight {
        self.cone.boresight()
    }

    /// `cos(max_off_nadir)`.
    #[must_use]
    pub const fn cos_max_off_nadir(self) -> f64 {
        self.cone.cos_half()
    }

    /// The composed constraint this sensor is: the slew cone intersected with visibility.
    #[inline]
    fn composed(self) -> And<Cone, ElevationMask> {
        And(self.cone, ElevationMask::HORIZON)
    }
}

impl Constraint for AgileOptical {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        self.composed().g(geom)
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        self.composed().covered(geom)
    }

    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        self.composed().max_rate(orbit, frame_rate)
    }
}

impl Reach for AgileOptical {
    fn reach(&self, sat_radius_m: f64, body: CullBody) -> f64 {
        self.composed().reach(sat_radius_m, body)
    }
}
