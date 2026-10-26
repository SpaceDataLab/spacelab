//! Two-tier constellations for continuous surveillance of a spherical *shell* of near-Earth space —
//! space situational awareness, where the targets are other orbiting objects, not the ground.
//!
//! Unlike ground coverage, the target is a sphere in space and a sensor's viewing zone is set by three
//! limits: the half-cone `β` of the onboard sensor (measured from its boresight), the slant range `D`,
//! and the minimum elevation `γ` of the satellite above the *target's* local horizon. A single Walker
//! delta-tier covers one sphere; a **two-tier** system splits a thick shell between an inner and an
//! outer tier, and is chosen to minimise the total launch characteristic velocity (СХС),
//!
//! ```text
//! V(N, H, i) = N · [ √(μ/Rₑ · (Rₑ + 2H)/(Rₑ + H)) − ω Rₑ cos i ],
//! ```
//!
//! the sum over the tiers of the per-satellite launch ΔV (Earth-rotation assist `ω Rₑ cos i`
//! subtracted). This module provides the space-visibility test and the cost; the tier sizing is the
//! optimisation the caller drives with them.
//!
//! The cost's energy balance and the elevation and cone tests are derived in
//! `derivations/shell_surveillance.mac`, the cost at 60 digits in `derivations/shell_surveillance.gp`;
//! the gates are `characteristic_velocity_matches_the_derived_values` and
//! `the_elevation_gate_is_an_angle_and_not_merely_a_sign` below.

use spacelab_core::{Angle, Vec3};

use crate::walker::Walker;

fn scale(v: Vec3, s: f64) -> Vec3 {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

/// Whether a sensor satellite at `sat` observes a target point at `target` (both Earth-centred
/// inertial, metres), under the three limits: the target lies within `half_cone` of the sensor
/// `boresight`, the slant range is at most `range`, and the satellite sits at least `min_elevation`
/// above the target's local horizon.
#[must_use]
pub fn observes(
    sat: Vec3,
    target: Vec3,
    boresight: Vec3,
    half_cone: Angle,
    range: f64,
    min_elevation: Angle,
) -> bool {
    let d = target - sat; // sat → target
    let dist = d.norm();
    if dist > range {
        return false;
    }
    // Sensor half-cone: angle between the boresight and the target direction ≤ β.
    if d.dot(boresight) < libm::cos(half_cone.radians()) * dist * boresight.norm() {
        return false;
    }
    // Elevation of the satellite above the target's horizon: sin(el) = (sat − target)·t̂ / |sat − target|.
    let t_hat = scale(target, 1.0 / target.norm());
    let sin_el = -d.dot(t_hat) / dist;
    sin_el >= libm::sin(min_elevation.radians())
}

/// The launch characteristic velocity (СХС) of a one-tier delta-system of `n` satellites at circular
/// height `height` (m) and inclination — Nguyen's formula (10). `re` is the Earth radius, `earth_rate`
/// its spin rate (rad/s).
#[must_use]
pub fn characteristic_velocity(
    n: u32,
    height: f64,
    inclination: Angle,
    mu: f64,
    re: f64,
    earth_rate: f64,
) -> f64 {
    let dv_star = libm::sqrt(mu / re * (re + 2.0 * height) / (re + height));
    let v_omega = earth_rate * re * libm::cos(inclination.radians());
    f64::from(n) * (dv_star - v_omega)
}

/// One tier of a shell-surveillance system: a Walker delta-system plus its sensor limits. The
/// boresight is the satellite nadir (`−r̂`), so the tier watches the shell below it.
#[derive(Clone, Copy, Debug)]
pub struct ShellTier {
    /// The tier's delta-system (circular, common altitude and inclination).
    pub walker: Walker,
    /// Sensor half-cone `β` from nadir.
    pub half_cone: Angle,
    /// Maximum slant range `D`, metres.
    pub range: f64,
    /// Minimum elevation `γ` of the satellite above a target's horizon.
    pub min_elevation: Angle,
}

impl ShellTier {
    /// Emit the tier's satellite elements (the underlying Walker).
    pub fn generate<E: FnMut(spacelab_elements::KeplerianElements)>(&self, emit: E) {
        self.walker.generate(emit);
    }

    /// This tier's launch characteristic velocity (СХС), taking the circular height as
    /// `semi_major_axis − re`.
    #[must_use]
    pub fn characteristic_velocity(&self, mu: f64, re: f64, earth_rate: f64) -> f64 {
        characteristic_velocity(
            self.walker.total(),
            self.walker.semi_major_axis - re,
            self.walker.inclination,
            mu,
            re,
            earth_rate,
        )
    }
}

/// A two-tier shell-surveillance system: an inner and an outer delta-tier whose union continuously
/// watches a spherical shell of space.
#[derive(Clone, Copy, Debug)]
pub struct TwoTierShell {
    /// The inner (lower-altitude) tier.
    pub inner: ShellTier,
    /// The outer (higher-altitude) tier.
    pub outer: ShellTier,
}

impl TwoTierShell {
    /// Total number of satellites across both tiers.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.inner.walker.total() + self.outer.walker.total()
    }

    /// The combined launch characteristic velocity (СХС) — the optimisation objective.
    #[must_use]
    pub fn total_characteristic_velocity(&self, mu: f64, re: f64, earth_rate: f64) -> f64 {
        self.inner.characteristic_velocity(mu, re, earth_rate)
            + self.outer.characteristic_velocity(mu, re, earth_rate)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::walker::WalkerPattern;
    use spacelab_elements::KeplerianElements;

    const MU: f64 = 3.986_004_415e14;
    const RE: f64 = 6_371e3; // Nguyen's mean Earth radius
    const OMEGA_E: f64 = 7.292_115_9e-5;
    const TAU: f64 = core::f64::consts::TAU;

    /// A unit vector with no zero component, and one perpendicular to it.
    fn frame() -> (Vec3, Vec3) {
        let raw_up = Vec3::new(0.3, 0.5, 0.8);
        let inv_up = 1.0 / raw_up.norm();
        let up = Vec3::new(raw_up.x * inv_up, raw_up.y * inv_up, raw_up.z * inv_up);
        let raw_east = Vec3::new(0.5, -0.3, 0.0); // up · east = 0.15 − 0.15 = 0
        let inv_east = 1.0 / raw_east.norm();
        let east = Vec3::new(
            raw_east.x * inv_east,
            raw_east.y * inv_east,
            raw_east.z * inv_east,
        );
        (up, east)
    }

    /// (G3) The elevation gate at geometries whose elevation is exact by construction, either side
    /// of the mask (`derivations/shell_surveillance.mac`: the code's `sin_el` is the sine of that
    /// angle).
    #[test]
    fn the_elevation_gate_is_an_angle_and_not_merely_a_sign() {
        // The existing coverage test uses `min_elevation = 0`, which reduces the gate to a sign
        // test — and every surviving mutant here rescales `sin_el` by a large *positive* factor, so
        // the sign is exactly what they preserve. Build a geometry whose elevation is known by
        // construction and place the mask strictly between two of them.
        let (up, east) = frame();
        let radius = RE + 500e3;
        let target = Vec3::new(up.x * radius, up.y * radius, up.z * radius);
        let range = 2_000e3;
        let slant = 800e3;
        let mask = Angle::from_radians(10.0_f64.to_radians());

        for (el_deg, want) in [(5.0_f64, false), (15.0_f64, true)] {
            let (sin_el, cos_el) = libm::sincos(el_deg.to_radians());
            // `look` leaves the target at elevation `el_deg`: sin_el = look·û = sin(el), exactly.
            let look = Vec3::new(
                sin_el * up.x + cos_el * east.x,
                sin_el * up.y + cos_el * east.y,
                sin_el * up.z + cos_el * east.z,
            );
            let sat = Vec3::new(
                target.x + slant * look.x,
                target.y + slant * look.y,
                target.z + slant * look.z,
            );
            let boresight = Vec3::new(-look.x, -look.y, -look.z);
            let got = observes(
                sat,
                target,
                boresight,
                Angle::from_radians(60.0_f64.to_radians()),
                range,
                mask,
            );
            assert_eq!(got, want, "elevation {el_deg} deg against a {mask:?} mask");
        }
    }

    #[test]
    fn the_cone_gate_measures_an_angle_against_a_boresight_of_any_length() {
        // The cone assertion in the existing test is vacuous — its "outside" case is refused by the
        // range gate before the cone is ever evaluated — and both call sites pass a unit boresight,
        // which is the one length at which `* boresight.norm()` and `/ boresight.norm()` agree. Use
        // a boresight of length two and straddle the half-cone.
        let (up, east) = frame();
        let radius = RE + 1_200e3;
        let sat = Vec3::new(up.x * radius, up.y * radius, up.z * radius);
        let nadir = Vec3::new(-2.0 * up.x, -2.0 * up.y, -2.0 * up.z); // length two, on purpose
        let half_cone = Angle::from_radians(20.0_f64.to_radians());
        let slant = 900e3;

        for (off_deg, want) in [(12.0_f64, true), (28.0_f64, false)] {
            let (sin_off, cos_off) = libm::sincos(off_deg.to_radians());
            // A direction `off_deg` from nadir, so the cone test is decided at a known angle.
            let los = Vec3::new(
                slant * (-cos_off * up.x + sin_off * east.x),
                slant * (-cos_off * up.y + sin_off * east.y),
                slant * (-cos_off * up.z + sin_off * east.z),
            );
            let target = Vec3::new(sat.x + los.x, sat.y + los.y, sat.z + los.z);
            let got = observes(
                sat,
                target,
                nadir,
                half_cone,
                2_000e3,
                Angle::from_radians(-90.0_f64.to_radians()),
            );
            assert_eq!(got, want, "{off_deg} deg off a 20 deg half-cone");
        }
    }

    #[test]
    fn both_boundaries_are_inclusive_at_the_limit_they_name() {
        // On the +z axis every quantity is exact: the slant range is the coordinate difference, the
        // boresight dot product is `2·L`, and `cos(0)·L·‖b‖` is the same `2·L` bit for bit. So this
        // fixture sits precisely on both thresholds at once and says which way each falls.
        let l = 400e3;
        let sat = Vec3::new(0.0, 0.0, RE + l);
        let target = Vec3::new(0.0, 0.0, RE);
        let nadir = Vec3::new(0.0, 0.0, -2.0);
        let zero_cone = Angle::from_radians(0.0);
        let horizon = Angle::from_radians(0.0);
        assert!(
            observes(sat, target, nadir, zero_cone, l, horizon),
            "a target exactly at the range limit and exactly on the boresight is not observed"
        );
        assert!(
            !observes(sat, target, nadir, zero_cone, l - 1.0, horizon),
            "a target beyond the range limit is observed"
        );
    }

    #[test]
    fn a_tier_reports_the_launch_cost_of_the_walker_it_wraps() {
        // Only ever checked by `chv.is_finite() && chv > 0.0`, which `1.0` also satisfies. Compare
        // against the free function — itself pinned by the energy balance above — at a *literal*
        // altitude, so that reading the height as `a + re` rather than `a − re` is observable.
        let height = 3_000e3;
        let inc = Angle::from_radians(65.0_f64.to_radians());
        let tier = |planes, per_plane, h: f64| ShellTier {
            walker: Walker {
                pattern: WalkerPattern::Delta,
                inclination: inc,
                planes,
                per_plane,
                phasing: 0,
                semi_major_axis: RE + h,
                mu: MU,
            },
            half_cone: Angle::from_radians(30.0_f64.to_radians()),
            range: 5_000e3,
            min_elevation: Angle::from_radians(0.0),
        };

        let inner = tier(4, 3, height);
        let want_inner = characteristic_velocity(12, height, inc, MU, RE, OMEGA_E);
        let got_inner = inner.characteristic_velocity(MU, RE, OMEGA_E);
        assert!(
            (got_inner - want_inner).abs() < 1e-9,
            "tier cost {got_inner} against {want_inner}"
        );

        let outer = tier(6, 2, 8_000e3);
        let want_outer = characteristic_velocity(12, 8_000e3, inc, MU, RE, OMEGA_E);
        let shell = TwoTierShell { inner, outer };
        let total = shell.total_characteristic_velocity(MU, RE, OMEGA_E);
        assert!(
            (total - (want_inner + want_outer)).abs() < 1e-9,
            "shell cost {total} against {want_inner} + {want_outer}"
        );
        assert!(
            total > want_inner && total > want_outer,
            "the total is not above either tier"
        );
    }

    /// (G2) The characteristic velocity satisfies the energy balance that defines `ΔV*`, evaluated
    /// independently in `f64` (derived in `derivations/shell_surveillance.mac`).
    #[test]
    fn characteristic_velocity_matches_the_energy_balance() {
        // ΔV* is defined by equating launch-pad and orbit energy:
        // (ΔV*)²/2 − μ/Rₑ = V_H²/2 − μ/(Rₑ+H), with V_H² = μ/(Rₑ+H).
        let h = 1_000e3;
        let inc = Angle::from_radians(60.0_f64.to_radians());
        let v_h2 = MU / (RE + h);
        let dv_star = (2.0 * (v_h2 / 2.0 - MU / (RE + h) + MU / RE)).sqrt();
        let expect = 12.0 * (dv_star - OMEGA_E * RE * (60.0_f64.to_radians()).cos());
        let got = characteristic_velocity(12, h, inc, MU, RE, OMEGA_E);
        assert!((got - expect).abs() < 1e-6, "{got} vs {expect}");
        // Higher orbits and lower (rotation-assisted) inclinations cost more / less as expected.
        assert!(
            characteristic_velocity(1, 2_000e3, inc, MU, RE, OMEGA_E)
                > characteristic_velocity(1, 500e3, inc, MU, RE, OMEGA_E)
        );
    }

    /// (G3) The characteristic velocity against `derivations/shell_surveillance.gp`, which evaluates
    /// the closed form at 60 digits with these constants (Nguyen's mean radius and rates). Bound:
    /// 8 ulp relative — a square root of a three-term quotient less a term thirty times smaller.
    #[test]
    fn characteristic_velocity_matches_the_derived_values() {
        for (n, h, inc_deg, want) in [
            (12, 1_000e3, 60.0_f64, 9.836_389_213_801_578e4),
            (12, 3_000e3, 65.0, 1.067_014_737_498_713_2e5),
            (12, 8_000e3, 65.0, 1.160_695_006_204_232_4e5),
            (1, 500e3, 60.0, 7.960_244_856_361_561e3),
        ] {
            let got = characteristic_velocity(
                n,
                h,
                Angle::from_radians(inc_deg.to_radians()),
                MU,
                RE,
                OMEGA_E,
            );
            assert!(
                (got - want).abs() <= 8.0 * f64::EPSILON * want,
                "N={n} H={h} i={inc_deg}: {got} against {want}"
            );
        }
    }

    #[test]
    fn observes_honours_cone_range_and_elevation() {
        // Sensor satellite 1500 km up on the +x axis, looking down (nadir = −x̂).
        let sat = Vec3::new(RE + 1_500e3, 0.0, 0.0);
        let nadir = Vec3::new(-1.0, 0.0, 0.0);
        let cone = Angle::from_radians(40.0_f64.to_radians());
        let range = 3_000e3;
        let gamma = Angle::from_radians(5.0_f64.to_radians());

        // Target straight below on a 700 km sphere: in cone, in range, high elevation → visible.
        let below = Vec3::new(RE + 700e3, 0.0, 0.0);
        assert!(observes(sat, below, nadir, cone, range, gamma));
        // Same sphere but pushed sideways out of the cone → not visible.
        let a = 30.0_f64.to_radians();
        let side = Vec3::new((RE + 700e3) * a.cos(), (RE + 700e3) * a.sin(), 0.0);
        assert!(!observes(sat, side, nadir, cone, range, gamma));
        // Straight below but beyond the range gate → not visible.
        assert!(!observes(sat, below, nadir, cone, 500e3, gamma));
    }

    fn fibonacci_sphere(n: u32, radius: f64) -> Vec<Vec3> {
        let ga = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..n)
            .map(|k| {
                let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(n);
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = ga * f64::from(k);
                Vec3::new(radius * r * phi.cos(), radius * r * phi.sin(), radius * z)
            })
            .collect()
    }

    fn tier(planes: u32, per_plane: u32, alt: f64, cone_deg: f64) -> ShellTier {
        ShellTier {
            walker: Walker {
                pattern: WalkerPattern::Delta,
                inclination: Angle::from_radians(70.0_f64.to_radians()),
                planes,
                per_plane,
                phasing: 1,
                semi_major_axis: RE + alt,
                mu: MU,
            },
            half_cone: Angle::from_radians(cone_deg.to_radians()),
            range: 12_000e3,
            min_elevation: Angle::from_radians(0.0),
        }
    }

    fn tier_positions(t: &ShellTier, time: f64) -> Vec<Vec3> {
        let mut v = Vec::new();
        t.generate(|e: KeplerianElements| v.push(e.propagate(time).to_state().0));
        v
    }

    /// (S1) Sampled shell coverage; supporting only.
    #[test]
    fn two_tiers_continuously_watch_a_space_shell() {
        // Two dense delta-tiers above a shell, both looking down, together keep every point of the
        // shell in view of at least one satellite at all sampled times. This exercises the space
        // visibility geometry end-to-end (targets are spheres in space, not the ground).
        let system = TwoTierShell {
            inner: tier(9, 9, 3_000e3, 55.0),
            outer: tier(8, 8, 6_000e3, 55.0),
        };
        // A shell from 600 to 1200 km altitude — below both tiers, so the nadir boresight and the
        // elevation constraint are satisfiable.
        let shell_radii = [RE + 600e3, RE + 900e3, RE + 1_200e3];
        let period = TAU * ((RE + 3_000e3).powi(3) / MU).sqrt(); // inner-tier period
        let tiers = [system.inner, system.outer];

        let mut worst_covered = true;
        for it in 0..40 {
            let time = period * f64::from(it) / 40.0;
            let sats: Vec<(Vec3, &ShellTier)> = tiers
                .iter()
                .flat_map(|t| tier_positions(t, time).into_iter().map(move |p| (p, t)))
                .collect();
            for &radius in &shell_radii {
                for target in fibonacci_sphere(300, radius) {
                    let seen = sats.iter().any(|(p, t)| {
                        let nadir = scale(*p, -1.0 / p.norm());
                        observes(*p, target, nadir, t.half_cone, t.range, t.min_elevation)
                    });
                    if !seen {
                        worst_covered = false;
                    }
                }
            }
        }
        assert!(worst_covered, "a shell point went unseen by both tiers");
        // The cost objective is a positive, finite number and the tiers add up.
        let chv = system.total_characteristic_velocity(MU, RE, OMEGA_E);
        assert!(chv.is_finite() && chv > 0.0);
        assert_eq!(system.total(), 9 * 9 + 8 * 8);
    }
}
