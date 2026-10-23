//! Continuous low-thrust propagation — the primitive electric-propulsion transfers are built on.
//!
//! A low-thrust arc is two-body gravity plus a small continuous acceleration along a steering law.
//! This module supplies the equations of motion and a fixed-step propagator for the common steering
//! laws (tangential for orbit raising/lowering, or a fixed inertial direction). It is the foundation
//! for an optimal-control layer — indirect (Pontryagin costates) or direct (collocation) — which is a
//! later extension ([plans/13-deep-space.md](../plans/13-deep-space.md), Stage D). Units are the
//! caller's, consistent; the thrust is given as an acceleration `T/m` (constant here — a
//! mass-depleting variant is a refinement).

#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]

use spacelab_core::Vec3;

/// A continuous low-thrust steering law: the unit thrust direction at the current state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Steering {
    /// Along the velocity — maximises specific-energy gain (orbit raising).
    Tangential,
    /// Opposite the velocity — orbit lowering / de-orbit.
    Retrograde,
    /// A fixed inertial direction (need not be unit; it is normalised).
    Inertial(Vec3),
}

impl Steering {
    fn direction(self, v: Vec3) -> Vec3 {
        match self {
            Self::Tangential => v.scale(1.0 / v.norm()),
            Self::Retrograde => v.scale(-1.0 / v.norm()),
            Self::Inertial(d) => d.scale(1.0 / d.norm()),
        }
    }
}

/// Acceleration under two-body gravity plus the low thrust: `−μ r/|r|³ + a_T·û`.
#[must_use]
pub fn acceleration(r: Vec3, v: Vec3, mu: f64, thrust_accel: f64, steering: Steering) -> Vec3 {
    let rn = r.norm();
    let g = -mu / (rn * rn * rn);
    let u = steering.direction(v);
    Vec3::new(
        g * r.x + thrust_accel * u.x,
        g * r.y + thrust_accel * u.y,
        g * r.z + thrust_accel * u.z,
    )
}

/// Propagate `(r, v)` under two-body gravity plus a constant-magnitude low thrust for time `dt`, in
/// `steps` fixed RK4 steps. Returns the final `(r, v)`.
#[must_use]
pub fn propagate_lowthrust(
    r: Vec3,
    v: Vec3,
    mu: f64,
    thrust_accel: f64,
    steering: Steering,
    dt: f64,
    steps: usize,
) -> (Vec3, Vec3) {
    let h = dt / steps as f64;
    let (mut r, mut v) = (r, v);
    let deriv = |r: Vec3, v: Vec3| (v, acceleration(r, v, mu, thrust_accel, steering));
    let add = |a: Vec3, b: Vec3, s: f64| Vec3::new(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s);
    for _ in 0..steps {
        let (k1r, k1v) = deriv(r, v);
        let (k2r, k2v) = deriv(add(r, k1r, h / 2.0), add(v, k1v, h / 2.0));
        let (k3r, k3v) = deriv(add(r, k2r, h / 2.0), add(v, k2v, h / 2.0));
        let (k4r, k4v) = deriv(add(r, k3r, h), add(v, k3v, h));
        r = Vec3::new(
            r.x + h / 6.0 * (k1r.x + 2.0 * k2r.x + 2.0 * k3r.x + k4r.x),
            r.y + h / 6.0 * (k1r.y + 2.0 * k2r.y + 2.0 * k3r.y + k4r.y),
            r.z + h / 6.0 * (k1r.z + 2.0 * k2r.z + 2.0 * k3r.z + k4r.z),
        );
        v = Vec3::new(
            v.x + h / 6.0 * (k1v.x + 2.0 * k2v.x + 2.0 * k3v.x + k4v.x),
            v.y + h / 6.0 * (k1v.y + 2.0 * k2v.y + 2.0 * k3v.y + k4v.y),
            v.z + h / 6.0 * (k1v.z + 2.0 * k2v.z + 2.0 * k3v.z + k4v.z),
        );
    }
    (r, v)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_precision_loss, clippy::excessive_precision)]
    use super::*;
    use spacelab_testkit::assert_within_ulps;

    const MU_EARTH_KM: f64 = 398_600.0;

    /// An inclined, eccentric state and an off-axis thrust direction — nothing at zero, nothing
    /// parallel to an axis, and the thrust direction not a unit vector, so the normalisation is
    /// exercised rather than assumed.
    const R: Vec3 = Vec3 {
        x: 6811.0,
        y: -2743.0,
        z: 1902.0,
    };
    const V: Vec3 = Vec3 {
        x: 1.734,
        y: 6.219,
        z: -2.845,
    };
    const DIRECTION: Vec3 = Vec3 {
        x: 0.41,
        y: -0.72,
        z: 0.56,
    };
    const THRUST: f64 = 3.7e-5;

    /// `-mu r/|r|^3 + a_T*u` at `(R, V)` for each steering law, evaluated at 60 decimal digits.
    const ACCELERATION: [[f64; 3]; 3] = [
        [
            -0.006_212_348_343_026_272_914_044_675,
            0.002_538_181_155_593_251_958_807_047,
            -0.001_752_283_544_005_648_307_675_690,
        ],
        [
            -0.006_230_535_615_742_461_862_434_361,
            0.002_472_952_407_599_031_038_092_380,
            -0.001_722_443_410_858_267_937_348_995,
        ],
        [
            -0.006_206_272_737_827_484_630_136_448,
            0.002_478_928_113_496_249_825_082_230,
            -0.001_716_644_513_354_264_596_043_664,
        ],
    ];

    /// Every steering law against the closed form, at a state with no symmetry to hide behind.
    /// `Inertial` is the one the spiral tests never reach at all.
    #[test]
    fn the_acceleration_is_its_closed_form_under_every_steering_law() {
        let laws = [
            Steering::Tangential,
            Steering::Retrograde,
            Steering::Inertial(DIRECTION),
        ];
        for (law, want) in laws.into_iter().zip(ACCELERATION) {
            let got = acceleration(R, V, MU_EARTH_KM, THRUST, law);
            for (axis, (name, got)) in [("x", got.x), ("y", got.y), ("z", got.z)]
                .into_iter()
                .enumerate()
            {
                assert_within_ulps(got, want[axis], 8, &format!("{law:?} {name}"));
            }
        }
    }

    /// The steering direction is a unit vector whichever law produced it, retrograde is tangential
    /// reversed exactly, and an inertial command is scale-invariant.
    #[test]
    fn every_steering_law_returns_a_unit_direction() {
        let unit = |law: Steering| {
            let a = acceleration(R, V, MU_EARTH_KM, 1.0, law);
            let g = acceleration(R, V, MU_EARTH_KM, 0.0, law);
            Vec3::new(a.x - g.x, a.y - g.y, a.z - g.z)
        };
        for law in [
            Steering::Tangential,
            Steering::Retrograde,
            Steering::Inertial(DIRECTION),
        ] {
            let d = unit(law);
            assert_within_ulps(d.norm(), 1.0, 4, &format!("|{law:?}|"));
        }
        let (fwd, back) = (unit(Steering::Tangential), unit(Steering::Retrograde));
        assert_eq!(
            (fwd.x, fwd.y, fwd.z),
            (-back.x, -back.y, -back.z),
            "retrograde is not tangential reversed"
        );
        let scaled = unit(Steering::Inertial(DIRECTION.scale(1000.0)));
        let plain = unit(Steering::Inertial(DIRECTION));
        assert!(
            (scaled - plain).norm() < 8.0 * f64::EPSILON,
            "an inertial command is not scale invariant"
        );
    }

    /// Runge-Kutta is fourth order, so halving the step over a fixed span must cut the difference
    /// between successive solutions by sixteen. A wrong stage weight lowers the order — which no
    /// energy or monotonicity assertion would notice.
    #[test]
    fn the_low_thrust_propagator_converges_at_fourth_order() {
        let span = 600.0;
        let solutions: Vec<Vec3> = [5usize, 10, 20, 40, 80]
            .iter()
            .map(|&n| {
                propagate_lowthrust(
                    R,
                    V,
                    MU_EARTH_KM,
                    THRUST,
                    Steering::Inertial(DIRECTION),
                    span,
                    n,
                )
                .0
            })
            .collect();
        for (k, w) in solutions.windows(3).enumerate() {
            let ratio = (w[0] - w[1]).norm() / (w[1] - w[2]).norm();
            assert!(
                (13.0..19.0).contains(&ratio),
                "halving {k} converged at a ratio of {ratio:.2}, not 16"
            );
        }
    }

    /// Tangential thrust raises the orbit: specific energy increases monotonically and the
    /// semi-major axis grows.
    #[test]
    fn tangential_thrust_raises_the_orbit() {
        let r0 = Vec3::new(7000.0, 0.0, 0.0);
        let vc = (MU_EARTH_KM / 7000.0).sqrt();
        let v0 = Vec3::new(0.0, vc, 0.0);
        let energy = |r: Vec3, v: Vec3| v.dot(v) / 2.0 - MU_EARTH_KM / r.norm();
        let sma = |r: Vec3, v: Vec3| -MU_EARTH_KM / (2.0 * energy(r, v));

        let a0 = sma(r0, v0);
        let e0 = energy(r0, v0);
        // Step through the spiral, checking energy never decreases.
        let (mut r, mut v) = (r0, v0);
        let mut prev_e = e0;
        for _ in 0..40 {
            let (rn, vn) =
                propagate_lowthrust(r, v, MU_EARTH_KM, 1e-4, Steering::Tangential, 250.0, 250);
            r = rn;
            v = vn;
            let e = energy(r, v);
            assert!(e >= prev_e - 1e-9, "energy dropped under tangential thrust");
            prev_e = e;
        }
        assert!(energy(r, v) > e0, "orbit energy did not rise");
        assert!(
            sma(r, v) > a0 + 100.0,
            "semi-major axis did not grow (from {a0:.0} km)"
        );
    }

    /// Retrograde thrust lowers the orbit; zero thrust conserves energy (two-body).
    #[test]
    fn retrograde_lowers_and_coasting_conserves() {
        let r0 = Vec3::new(8000.0, 0.0, 200.0);
        let vc = (MU_EARTH_KM / r0.norm()).sqrt();
        let v0 = Vec3::new(0.0, vc, 0.0);
        let energy = |r: Vec3, v: Vec3| v.dot(v) / 2.0 - MU_EARTH_KM / r.norm();

        let (rr, vr) = propagate_lowthrust(
            r0,
            v0,
            MU_EARTH_KM,
            1e-4,
            Steering::Retrograde,
            2000.0,
            2000,
        );
        assert!(
            energy(rr, vr) < energy(r0, v0),
            "retrograde thrust did not lower the orbit"
        );

        let (rc, vc2) =
            propagate_lowthrust(r0, v0, MU_EARTH_KM, 0.0, Steering::Tangential, 2000.0, 4000);
        assert!(
            (energy(rc, vc2) - energy(r0, v0)).abs() < 1e-6,
            "coasting did not conserve energy"
        );
    }
}
