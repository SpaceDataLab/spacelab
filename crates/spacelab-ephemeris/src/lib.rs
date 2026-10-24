//! High-fidelity planetary ephemerides over [ANISE](https://nyxspace.com) (MPL-2.0), the
//! SPICE-replacement that reads JPL DE440 SPK kernels.
//!
//! This is the ephemeris stage of the deep-space pillar: the DE-grade body states that turn a
//! [`spacelab_astro`](../spacelab_astro) transfer into a dated Moon or Mars trajectory. The
//! analytic Sun/Moon of `spacelab-propagation` is a LEO perturbation; here the ephemeris *is* the
//! answer. Only this crate depends on ANISE, so the core build is unaffected.
//!
//! The kernel is not committed (it is tens of MB); fetch it with `tools/fetch-ephemeris.sh`, which
//! pulls the NyxSpace-hosted `de440s.bsp`.

use anise::almanac::Almanac;
use spacelab_core::Vec3;

// Re-export the currency a caller needs: epochs, frames, and the named-frame constants.
pub use anise::constants::frames;
pub use anise::prelude::{Epoch, Frame};

pub mod bridge;
pub use bridge::PerturbingBody;

/// A loaded DE ephemeris: an ANISE almanac backed by one or more SPK kernels.
pub struct Ephemeris {
    almanac: Almanac,
}

/// A Cartesian state, kilometres and kilometres per second, in the requested frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyState {
    /// Position, km.
    pub position: Vec3,
    /// Velocity, km/s.
    pub velocity: Vec3,
}

/// A rotation matrix, row-major — the shape
/// [`BodyOriented`](spacelab_propagation::forces::BodyOriented) consumes.
pub type Matrix3 = [[f64; 3]; 3];

impl Ephemeris {
    /// Load an SPK/BSP kernel (e.g. `de440s.bsp`).
    ///
    /// # Errors
    ///
    /// Propagates ANISE's error if the kernel cannot be read or parsed.
    pub fn load(path: &str) -> Result<Self, Box<dyn core::error::Error>> {
        Ok(Self {
            almanac: Almanac::new(path)?,
        })
    }

    /// Add another kernel to this almanac, guessing its type from its contents.
    ///
    /// Translations need an SPK; orientations need a binary PCK (`moon_pa_de440_200625.bpc` for the
    /// Moon's principal axes) and, for the mean-Earth frame, a set of Euler parameters. Kernels
    /// compose, so a selenocentric problem loads `de440s.bsp` and then the lunar BPC.
    ///
    /// # Errors
    ///
    /// Propagates ANISE's error if the kernel cannot be read or its type recognised.
    pub fn with_kernel(self, path: &str) -> Result<Self, Box<dyn core::error::Error>> {
        Ok(Self {
            almanac: self.almanac.load(path)?,
        })
    }

    /// Add a NAIF **text** frame kernel (`.tf`), for the frames defined as a fixed rotation off
    /// another one rather than by a time series.
    ///
    /// [`with_kernel`](Self::with_kernel) reads binary kernels and cannot parse a text FK, so without
    /// this the Moon's mean-Earth frame is unreachable: `moon_de440_250416.tf` is where the PA→ME
    /// rotation is published. ANISE converts the assignments to Euler parameters at load.
    ///
    /// # Errors
    ///
    /// Propagates ANISE's error if the file cannot be read or its assignments do not parse.
    pub fn with_frame_kernel(self, path: &str) -> Result<Self, Box<dyn core::error::Error>> {
        let euler = anise::naif::kpl::parser::convert_fk(path, false)?;
        Ok(Self {
            almanac: self.almanac.with_euler_parameters(euler),
        })
    }

    /// The state of `target` relative to `observer` at `epoch` (TDB), km and km/s, in `target`'s
    /// J2000 frame. Frames come from [`anise::constants::frames`] (e.g. `MOON_J2000`, `EARTH_J2000`).
    ///
    /// # Errors
    ///
    /// Propagates ANISE's error if the translation is undefined (a body missing from the kernel, an
    /// epoch outside its coverage).
    pub fn state(
        &self,
        target: Frame,
        observer: Frame,
        epoch: Epoch,
    ) -> Result<BodyState, Box<dyn core::error::Error>> {
        let orbit = self.almanac.translate(target, observer, epoch, None)?;
        Ok(BodyState {
            position: Vec3::new(orbit.radius_km.x, orbit.radius_km.y, orbit.radius_km.z),
            velocity: Vec3::new(
                orbit.velocity_km_s.x,
                orbit.velocity_km_s.y,
                orbit.velocity_km_s.z,
            ),
        })
    }

    /// The rotation taking a vector expressed in `from` to the same vector expressed in `to`, at
    /// `epoch`: row-major, so `v_to[i] = Σⱼ m[i][j] · v_from[j]`.
    ///
    /// This is what a body-fixed force model needs about a body that is not the Earth. For the Moon
    /// that is `rotation(MOON_J2000, MOON_PA_DE440_FRAME, epoch)`, and the caller feeds it to
    /// [`BodyOriented`](spacelab_propagation::forces::BodyOriented) as a closure over the
    /// propagation time.
    ///
    /// Which lunar frame is not a detail: the gravity field is fitted in the **principal axes**, the
    /// maps are drawn in the **mean Earth/mean rotation** frame, and the two differ by about 0.03°,
    /// which is ~900 m on the surface. `MOON_PA_DE440_FRAME` needs `moon_pa_de440_200625.bpc`;
    /// `MOON_ME_DE440_ME421_FRAME` additionally needs the Euler parameters built from the DE440
    /// frame kernel. Both come from `tools/fetch-lunar-kernels.sh`.
    ///
    /// # Errors
    ///
    /// Propagates ANISE's error when no orientation path connects the two frames — most often
    /// because the kernel defining one of them has not been loaded.
    pub fn rotation(
        &self,
        from: Frame,
        to: Frame,
        epoch: Epoch,
    ) -> Result<Matrix3, Box<dyn core::error::Error>> {
        let dcm = self.almanac.rotate(from, to, epoch)?;
        let m = dcm.rot_mat;
        Ok([
            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ])
    }

    /// Re-express `state` about a different centre, leaving its axes alone.
    ///
    /// A selenocentric propagation is a geocentric one with the origin moved: same J2000 axes, the
    /// Moon at the centre instead of the Earth. That is a translation by the Moon's own ephemeris
    /// state, and the velocity shift matters — the Moon moves at about 1 km/s, which is a quarter of
    /// a lunar orbital velocity, so dropping it is not a small error.
    ///
    /// # Errors
    ///
    /// Fails if `from_center` and `to_center` do not share axes — the operation is a translation and
    /// cannot absorb a rotation — or propagates ANISE's error if the shift is undefined.
    pub fn recenter(
        &self,
        state: BodyState,
        from_center: Frame,
        to_center: Frame,
        epoch: Epoch,
    ) -> Result<BodyState, Box<dyn core::error::Error>> {
        if from_center.orientation_id != to_center.orientation_id {
            return Err(format!(
                "recenter is a translation, but orientation {} and {} differ",
                from_center.orientation_id, to_center.orientation_id
            )
            .into());
        }
        let shift = self.state(to_center, from_center, epoch)?;
        Ok(BodyState {
            position: state.position - shift.position,
            velocity: state.velocity - shift.velocity,
        })
    }
}
