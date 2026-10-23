//! Environmental torques on a rigid box-wing spacecraft, body frame, N·m.
//!
//! # Why a torque is not the force with an arm bolted on afterwards
//!
//! [`crate::spacecraft::BoxWing`] already knows what each facet does to the body's *momentum*:
//! pressure and shear per facet, summed and divided by the mass. The moment about the centre of mass
//! is the same per-facet force crossed with the arm to that facet's centre of pressure, `Σ dᵢ × Fᵢ`
//! — and that sum is not recoverable from the total force. Two facets whose forces cancel exactly
//! still leave a couple; a symmetric body under a symmetric load has zero torque and a large force.
//! So the facet loop is shared rather than written twice: the force model hands both callers the
//! same per-facet force through `for_each_srp_facet` / `for_each_drag_facet`, and all this module
//! adds is the arm. A second implementation of the panel physics here would be a second thing to
//! keep in step with Orekit.
//!
//! # The four torques
//!
//! | source | expression | scaling |
//! |---|---|---|
//! | solar radiation | `Σ dᵢ × F_srp,ᵢ` | independent of altitude, `∝ area × arm` |
//! | aerodynamic | `Σ dᵢ × F_drag,ᵢ` | `∝ ρ v²`, so it collapses above ~1000 km |
//! | gravity gradient | `3μ/R⁵ · (R × I R)` | `∝ R⁻³`, and it needs an *asymmetric* inertia |
//! | residual magnetic dipole | `m × B` | `∝ R⁻³` through the field |
//!
//! The first two are moments of the forces the drag and SRP models already produce, so they inherit
//! that verification; the last two are closed forms checked here against their analytic limits.
//!
//! Sources: Wertz, *Spacecraft Attitude Determination and Control* (1978) §17.2–17.4; Markley &
//! Crassidis, *Fundamentals of Spacecraft Attitude Determination and Control* (2014) §3.4; Hughes,
//! *Spacecraft Attitude Dynamics* (1986) §7.

use crate::attitude::{Inertia, Vec3, cross};
use crate::spacecraft::BoxWing;

/// A box-wing spacecraft as a *rigid body*: the facets the force models already use, plus the three
/// things a moment needs and a force does not — where each facet pushes, how the mass is
/// distributed, and what magnetic dipole the bus leaves behind.
///
/// The arms live here rather than inside `Facet` deliberately. A facet's force is a function of its
/// normal and area alone; threading a position through the force path would put a new field into
/// every constructor of a model that has been verified against Orekit without one, for the benefit
/// of a caller that model does not have.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RigidBoxWing<const P: usize> {
    /// The facets and their optical and aerodynamic coefficients.
    pub shape: BoxWing<P>,
    /// Vector from the centre of mass to each facet's centre of pressure, body frame, m — index for
    /// index with `shape.facets`. For a facet whose pressure is uniform this is its centroid.
    pub arms: [Vec3; P],
    /// Inertia tensor about the centre of mass, body frame, kg·m².
    pub inertia: Inertia,
    /// Residual magnetic dipole moment of the bus, body frame, A·m².
    pub dipole: Vec3,
}

impl<const P: usize> RigidBoxWing<P> {
    /// Moment of the solar-radiation-pressure force about the centre of mass. Arguments are
    /// [`BoxWing::srp_acceleration`]'s: the radiation flux vector and the unit satellite→Sun
    /// direction, both body frame. Zero in umbra, where the force is zero facet by facet.
    #[must_use]
    pub fn srp_torque(&self, flux_body: &Vec3, sun_dir_body: &Vec3) -> Vec3 {
        let mut total = [0.0; 3];
        self.shape
            .for_each_srp_facet(flux_body, sun_dir_body, |index, force| {
                let moment = cross(self.arms[index], force);
                total = [
                    total[0] + moment[0],
                    total[1] + moment[1],
                    total[2] + moment[2],
                ];
            });
        total
    }

    /// Moment of the aerodynamic force about the centre of mass. Arguments are
    /// [`BoxWing::drag_acceleration`]'s: density, the body-frame velocity of the atmosphere relative
    /// to the spacecraft, and the unit satellite→Sun direction the Sun-tracking panels need.
    #[must_use]
    pub fn drag_torque(&self, rho: f64, v_rel_body: &Vec3, sun_dir_body: &Vec3) -> Vec3 {
        let mut total = [0.0; 3];
        self.shape
            .for_each_drag_facet(rho, v_rel_body, sun_dir_body, |index, force| {
                let moment = cross(self.arms[index], force);
                total = [
                    total[0] + moment[0],
                    total[1] + moment[1],
                    total[2] + moment[2],
                ];
            });
        total
    }

    /// [`gravity_gradient`] on this body's inertia.
    #[must_use]
    pub fn gravity_gradient_torque(&self, centre_to_body: Vec3, mu: f64) -> Vec3 {
        gravity_gradient(&self.inertia, centre_to_body, mu)
    }

    /// [`magnetic`] on this body's residual dipole.
    #[must_use]
    pub fn magnetic_torque(&self, field_body: Vec3) -> Vec3 {
        magnetic(self.dipole, field_body)
    }
}

/// The gravity-gradient torque, `T = 3μ/R⁵ · (R × I R)`, N·m.
///
/// `centre_to_body` is the vector between the spacecraft's centre of mass and the primary's centre,
/// **in the body frame** — that is the one thing here that can be got wrong, since the vector is
/// naturally known in an inertial or rotating frame and has to be rotated by the attitude first.
///
/// The *sense* of that vector cannot be got wrong: both factors of the cross product flip together
/// under `R → −R`, so nadir and zenith give the same torque. Nor can a scale error hide, since the
/// expression is homogeneous of degree −3 in `|R|`.
///
/// It vanishes for a spherically symmetric body (`I = λE` makes `I R ∥ R`) — the torque exists only
/// because the near side is pulled harder than the far side of an *unequal* mass distribution.
#[must_use]
pub fn gravity_gradient(inertia: &Inertia, centre_to_body: Vec3, mu: f64) -> Vec3 {
    let r2 = centre_to_body[0] * centre_to_body[0]
        + centre_to_body[1] * centre_to_body[1]
        + centre_to_body[2] * centre_to_body[2];
    let coefficient = 3.0 * mu / (r2 * r2 * libm::sqrt(r2));
    let moment = cross(centre_to_body, inertia.apply(centre_to_body));
    [
        coefficient * moment[0],
        coefficient * moment[1],
        coefficient * moment[2],
    ]
}

/// The residual-dipole magnetic torque, `T = m × B`, N·m.
///
/// `dipole_body` is the spacecraft's residual magnetic dipole moment in A·m² and `field_body` the
/// geomagnetic flux density at the spacecraft, both body frame. The field is in **tesla**:
/// `spacelab_geomagnetism`'s IGRF returns nanotesla, so a caller scales by `1e-9`. The conversion is
/// left to the caller rather than hidden in a unit-named argument because the dipole's own unit is
/// what fixes it, and a hidden factor of 10⁹ is a wrong answer that looks plausible.
#[must_use]
pub fn magnetic(dipole_body: Vec3, field_body: Vec3) -> Vec3 {
    cross(dipole_body, field_body)
}

/// Arms for [`BoxWing::box6`]'s facet order — `−X, +X, −Y, +Y, −Z, +Z` — for a box whose centre of
/// mass is at its geometric centre: each side's centroid, half its own dimension out along its
/// normal. Offset them yourself if the centre of mass is not the centre of the box; that offset is
/// exactly what makes a box-shaped body feel a net torque under a uniform pressure.
#[must_use]
pub fn box6_arms(x_length: f64, y_length: f64, z_length: f64) -> [Vec3; 6] {
    let (x, y, z) = (0.5 * x_length, 0.5 * y_length, 0.5 * z_length);
    [
        [-x, 0.0, 0.0],
        [x, 0.0, 0.0],
        [0.0, -y, 0.0],
        [0.0, y, 0.0],
        [0.0, 0.0, -z],
        [0.0, 0.0, z],
    ]
}
