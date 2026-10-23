//! Interplanetary transfer metrics from a Lambert arc.
//!
//! Given two bodies' states (from an ephemeris) and a time of flight, the Lambert arc between them
//! fixes the heliocentric departure and arrival velocities; differencing against the bodies' own
//! velocities gives the hyperbolic excess speeds — the launch energy `C3` and the arrival `v∞` that
//! a porkchop plot maps over a grid of departure and arrival dates. See Curtis, *Orbital Mechanics
//! for Engineering Students*, §8.

use spacelab_core::Vec3;

use crate::lambert::lambert;

/// The energy metrics of one transfer leg.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transfer {
    /// Departure hyperbolic excess speed `v∞ = |v₁ − v_body1|` (√C3), length/time.
    pub v_inf_departure: f64,
    /// Characteristic launch energy `C3 = v∞²`, length²/time².
    pub c3: f64,
    /// Arrival hyperbolic excess speed `|v₂ − v_body2|`, length/time.
    pub v_inf_arrival: f64,
    /// The Lambert departure velocity (heliocentric).
    pub v_departure: Vec3,
    /// The Lambert arrival velocity (heliocentric).
    pub v_arrival: Vec3,
}

fn diff(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

/// Metrics of the transfer from `body1` (position `r1`, velocity `v_body1`) to `body2` (`r2`,
/// `v_body2`) over time `tof`, under central gravitational parameter `mu`. `prograde` selects the
/// transfer sense. Units are the caller's, consistent. Returns `None` when the Lambert geometry is
/// degenerate (0° or 180° transfer).
#[must_use]
pub fn transfer(
    r1: Vec3,
    v_body1: Vec3,
    r2: Vec3,
    v_body2: Vec3,
    tof: f64,
    mu: f64,
    prograde: bool,
) -> Option<Transfer> {
    let (v1, v2) = lambert(r1, r2, tof, mu, prograde)?;
    let v_inf_dep = diff(v1, v_body1).norm();
    let v_inf_arr = diff(v2, v_body2).norm();
    Some(Transfer {
        v_inf_departure: v_inf_dep,
        c3: v_inf_dep * v_inf_dep,
        v_inf_arrival: v_inf_arr,
        v_departure: v1,
        v_arrival: v2,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    use super::*;
    use crate::kepler::propagate;
    use spacelab_testkit::assert_within_ulps;

    const MU_SUN_KM: f64 = 1.327_124_400_18e11; // km³/s²
    const AU: f64 = 1.495_978_707e8; // km

    /// The three metrics are exactly what they claim to be: the excess speeds of the very arc the
    /// call returns, and `C3` its square. The geometry is inclined, so no component of the velocity
    /// difference can hide behind a zero the way it does in the coplanar case.
    #[test]
    fn the_metrics_are_the_excess_speeds_of_the_arc_that_comes_back() {
        let (r_e, r_m) = (1.0 * AU, 1.524 * AU);
        let (vc_e, vc_m) = ((MU_SUN_KM / r_e).sqrt(), (MU_SUN_KM / r_m).sqrt());
        let ang = 150f64.to_radians();
        let tilt = 3.4f64.to_radians();
        let r1 = Vec3::new(r_e * 0.9781, r_e * 0.2079, r_e * 0.0139);
        let v_body1 = Vec3::new(-vc_e * 0.2072, vc_e * 0.9739, vc_e * 0.0918);
        let r2 = Vec3::new(
            r_m * ang.cos() * tilt.cos(),
            r_m * ang.sin(),
            r_m * ang.cos() * tilt.sin(),
        );
        let v_body2 = Vec3::new(
            -vc_m * ang.sin(),
            vc_m * ang.cos() * tilt.cos(),
            vc_m * ang.cos() * tilt.sin(),
        );
        let tof = 250.0 * 86_400.0;

        let t = transfer(r1, v_body1, r2, v_body2, tof, MU_SUN_KM, true).expect("solvable");
        let departure = (t.v_departure - v_body1).norm();
        let arrival = (t.v_arrival - v_body2).norm();
        assert_within_ulps(t.v_inf_departure, departure, 2, "departure excess speed");
        assert_within_ulps(t.v_inf_arrival, arrival, 2, "arrival excess speed");
        assert_within_ulps(t.c3, departure * departure, 2, "C3");
        assert!(
            departure > 1.0 && arrival > 1.0,
            "the geometry left an excess speed too small to resolve: {departure} / {arrival}"
        );
        let (v1, v2) = crate::lambert::lambert(r1, r2, tof, MU_SUN_KM, true).expect("arc");
        assert_eq!(
            t.v_departure, v1,
            "the departure velocity is not the Lambert one"
        );
        assert_eq!(
            t.v_arrival, v2,
            "the arrival velocity is not the Lambert one"
        );
    }

    /// A coplanar circular Earth→Mars-like transfer: the metrics are physical (positive C3 of the
    /// right order) and the Lambert leg is self-consistent (departure velocity propagates to r2).
    #[test]
    fn earth_to_mars_like_transfer_is_physical_and_consistent() {
        let (r_e, r_m) = (1.0 * AU, 1.524 * AU);
        let (vc_e, vc_m) = ((MU_SUN_KM / r_e).sqrt(), (MU_SUN_KM / r_m).sqrt());
        // Earth at +x; Mars 150° ahead (a non-degenerate transfer angle).
        let ang = 150f64.to_radians();
        let r1 = Vec3::new(r_e, 0.0, 0.0);
        let v_body1 = Vec3::new(0.0, vc_e, 0.0);
        let r2 = Vec3::new(r_m * ang.cos(), r_m * ang.sin(), 0.0);
        let v_body2 = Vec3::new(-vc_m * ang.sin(), vc_m * ang.cos(), 0.0);
        let tof = 250.0 * 86_400.0; // ~250 days

        let t = transfer(r1, v_body1, r2, v_body2, tof, MU_SUN_KM, true).expect("solvable");
        // Earth-Mars launch energy is a few to a few tens of km²/s².
        assert!(
            (3.0..60.0).contains(&t.c3),
            "implausible C3 {:.2} km²/s²",
            t.c3
        );
        assert!(
            t.v_inf_arrival > 0.0 && t.v_inf_arrival < 10.0,
            "arrival v∞ {:.2}",
            t.v_inf_arrival
        );
        // The Lambert departure velocity, propagated by tof, reaches Mars's position.
        let (rf, _) = propagate(r1, t.v_departure, MU_SUN_KM, tof);
        let miss = ((rf.x - r2.x).powi(2) + (rf.y - r2.y).powi(2) + (rf.z - r2.z).powi(2)).sqrt();
        assert!(miss < 1.0, "transfer arc missed Mars by {miss:.3e} km");
    }
}
