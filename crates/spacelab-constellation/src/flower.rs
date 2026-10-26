//! Flower Constellations — every satellite on one repeating space track (Mortari, Wilkins &
//! Bruccoleri, *The Flower Constellations*, J. Astronautical Sciences, 2004).
//!
//! All `Ns` satellites share a single orbit shape `(a, e, i, ω)` and differ only in right ascension
//! `Ω` and mean anomaly `M`. The semi-major axis is chosen so the ground track *repeats*: in one
//! repetition the satellite completes `Np` revolutions while the rotating frame turns `Nd` times,
//!
//! ```text
//! Np · T_orbit = Nd · T_frame   ⟹   n = (Np/Nd)·ω_frame,   a = (μ / n²)^(1/3).
//! ```
//!
//! Two satellites lie on the *same* repeating track iff their offsets satisfy the compatibility
//! relation `Np·ΔΩ + Nd·ΔM ≡ 0 (mod 2π)` — the reference trajectory time-shifted. Distributing the
//! `Ns` satellites uniformly in time along that one track (shift `k·T_rep/Ns`) gives the closed form
//!
//! ```text
//! Ω_k = Ω_0 − 2π·k·Nd/Ns,   M_k = M_0 + 2π·k·Np/Ns,   k = 0…Ns−1,
//! ```
//!
//! which satisfies the compatibility relation exactly for every `k`. (The later 2-D / 3-D *lattice*
//! Flower theory, Davis–Avendaño–Mortari 2013, generalises this to several interleaved tracks and
//! subsumes Walker; only the single-track form is built here.)
//!
//! The repeat condition, the time-shift phasing and the compatibility relation are derived in
//! `derivations/flower_phasing.mac`; `derivations/flower_phasing.gp` gives the semi-major axis at
//! 60 digits and `derivations/flower_phasing.g` (GAP) the orbit counts. The gates are
//! `every_satellite_is_the_reference_time_shifted_along_the_track`,
//! `the_semi_major_axis_matches_the_derived_value` and
//! `distinct_satellites_and_nodes_are_the_group_orbit_counts` below.

use spacelab_core::Angle;
use spacelab_elements::KeplerianElements;

const TAU: f64 = core::f64::consts::TAU;

/// A single-track Flower Constellation: `satellites` spacecraft sharing one `(Np, Nd)` repeating
/// orbit of shape `(e, i, ω)`, uniformly time-phased along the common ground track.
#[derive(Clone, Copy, Debug)]
pub struct FlowerConstellation {
    /// `Np` — satellite revolutions per repetition (petals, before reduction by `gcd(Np, Nd)`).
    pub petals: u32,
    /// `Nd` — rotating-frame turns per repetition (typically "days").
    pub days: u32,
    /// `Ns` — number of satellites.
    pub satellites: u32,
    /// Common eccentricity `e`.
    pub eccentricity: f64,
    /// Common inclination `i`.
    pub inclination: Angle,
    /// Common argument of periapsis `ω`.
    pub arg_periapsis: Angle,
    /// Right ascension `Ω_0` of the reference satellite.
    pub raan: Angle,
    /// Mean anomaly `M_0` of the reference satellite.
    pub mean_anomaly: Angle,
    /// Rotating-frame angular rate `ω_frame`, rad/s (e.g. Earth's sidereal rate).
    pub frame_rate: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

impl FlowerConstellation {
    /// Total number of satellites.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.satellites
    }

    /// The repeating-orbit mean motion `n = (Np/Nd)·ω_frame`, rad/s. Requires `days ≥ 1`.
    #[must_use]
    pub fn mean_motion(&self) -> f64 {
        f64::from(self.petals) / f64::from(self.days) * self.frame_rate
    }

    /// The repeating-orbit semi-major axis `a = (μ / n²)^(1/3)`, metres. Requires `days ≥ 1`.
    #[must_use]
    pub fn semi_major_axis(&self) -> f64 {
        let n = self.mean_motion();
        libm::cbrt(self.mu / (n * n))
    }

    /// Emit the `Ns` satellites' elements. Every set shares `(a, e, i, ω)`; satellite `k` sits at
    /// `Ω_k = Ω_0 − 2π·k·Nd/Ns` with mean anomaly `M_k = M_0 + 2π·k·Np/Ns`, converted to a true
    /// anomaly via [`KeplerianElements::at_mean_anomaly`]. A degenerate design (any of `Np`, `Nd`,
    /// `Ns` zero) emits nothing. The `Ns` satellites fill `Ns/gcd(Np, Nd, Ns)` distinct positions:
    /// an unreduced `(Np, Nd)` sharing a factor with `Ns` emits coincident satellites.
    pub fn generate<E: FnMut(KeplerianElements)>(&self, mut emit: E) {
        let (np, nd, ns) = (self.petals, self.days, self.satellites);
        if np == 0 || nd == 0 || ns == 0 {
            return;
        }
        let a = self.semi_major_axis();
        let raan0 = self.raan.radians();
        let m0 = self.mean_anomaly.radians();
        let per_sat = TAU / f64::from(ns); // 2π/Ns
        for k in 0..ns {
            let kf = f64::from(k);
            let raan = raan0 - kf * f64::from(nd) * per_sat;
            let mean = m0 + kf * f64::from(np) * per_sat;
            let base = KeplerianElements {
                semi_major_axis: a,
                eccentricity: self.eccentricity,
                inclination: self.inclination,
                raan: Angle::from_radians(raan),
                arg_periapsis: self.arg_periapsis,
                true_anomaly: Angle::from_radians(0.0),
                mu: self.mu,
            };
            emit(base.at_mean_anomaly(Angle::from_radians(mean)));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    use super::*;
    use spacelab_core::Vec3;
    use spacelab_testkit::nan_max;

    const MU_EARTH: f64 = 3.986_004_418e14;
    // Earth's sidereal rotation rate, rad/s (2π / 86 164.0905 s).
    const OMEGA_EARTH: f64 = 7.292_115_146_5e-5;

    fn example() -> FlowerConstellation {
        FlowerConstellation {
            petals: 3, // 3 revs per sidereal day → ~8 h repeating orbit
            days: 1,
            satellites: 6,
            eccentricity: 0.2,
            inclination: Angle::from_radians(63.434_f64.to_radians()), // critical inclination
            arg_periapsis: Angle::from_radians(270.0_f64.to_radians()),
            raan: Angle::from_radians(0.0),
            mean_anomaly: Angle::from_radians(0.0),
            frame_rate: OMEGA_EARTH,
            mu: MU_EARTH,
        }
    }

    fn collect(fc: &FlowerConstellation) -> Vec<KeplerianElements> {
        let mut v = Vec::new();
        fc.generate(|e| v.push(e));
        v
    }

    /// Express an inertial position in the frame rotated by `theta` about z (ECEF convention: a point
    /// at inertial longitude φ maps to φ − theta).
    fn to_rotating(r: Vec3, theta: f64) -> Vec3 {
        let (s, c) = libm::sincos(theta);
        Vec3::new(r.x * c + r.y * s, -r.x * s + r.y * c, r.z)
    }

    fn rot_pos(e: &KeplerianElements, t: f64, omega: f64) -> Vec3 {
        to_rotating(e.propagate(t).to_state().0, omega * t)
    }

    #[test]
    fn a_multi_day_repeat_with_a_fractional_phase_ratio_pins_the_relations() {
        // Every Flower configuration in the repository has `Nd = 1` and `M₀ = 0` — the one place
        // where `Np/Nd` and `Np·Nd` coincide, where `k·Nd` and `k/Nd` coincide, and where `2Np/Ns`
        // is a whole number, so flipping the sign of the mean-anomaly progression moves every
        // satellite by a whole number of turns and lands it back where it started. None of the
        // three relations is observable there. `Np=14, Nd=3, Ns=8` breaks all three degeneracies.
        let fc = FlowerConstellation {
            petals: 14,
            days: 3,
            satellites: 8,
            mean_anomaly: Angle::from_radians(0.9),
            ..example()
        };

        // The repeat condition: `Np` orbits take exactly `Nd` revolutions of the rotating frame.
        let want_n = 14.0 * OMEGA_EARTH / 3.0;
        let got_n = fc.mean_motion();
        assert!(
            (got_n - want_n).abs() < 1e-15 * want_n,
            "mean motion {got_n} against {want_n}"
        );

        // Mortari's compatibility relation, between every consecutive pair: Np·ΔΩ + Nd·ΔM ≡ 0.
        let sats = collect(&fc);
        assert_eq!(sats.len(), 8);
        let tau = core::f64::consts::TAU;
        for k in 1..sats.len() {
            let draan = sats[k].raan.radians() - sats[k - 1].raan.radians();
            let dmean = sats[k].mean_anomaly().radians() - sats[k - 1].mean_anomaly().radians();
            let res = libm::fmod(libm::fmod(14.0 * draan + 3.0 * dmean, tau) + tau, tau);
            assert!(
                res.min(tau - res) < 1e-9,
                "satellite {k}: compatibility residual {}",
                res.min(tau - res)
            );
        }
    }

    #[test]
    fn every_degenerate_count_on_its_own_emits_nothing() {
        // The existing degeneracy test only ever zeroes `Np`, which the first clause of the guard
        // catches; `Nd = 0` then reaches `mean_motion` and divides by zero, giving `a = 0` and a
        // full set of elements at the centre of the Earth.
        for (fc, which) in [
            (
                FlowerConstellation {
                    petals: 0,
                    ..example()
                },
                "petals",
            ),
            (
                FlowerConstellation {
                    days: 0,
                    ..example()
                },
                "days",
            ),
            (
                FlowerConstellation {
                    satellites: 0,
                    ..example()
                },
                "satellites",
            ),
        ] {
            assert!(
                collect(&fc).is_empty(),
                "zero {which} still emitted elements"
            );
        }
    }

    #[test]
    fn semi_major_axis_matches_the_repeat_condition() {
        let fc = example();
        // n = (Np/Nd)·ω_e, and independently a Kepler check n² a³ = μ.
        let n = fc.mean_motion();
        assert!((n - 3.0 * OMEGA_EARTH).abs() < 1e-15);
        let a = fc.semi_major_axis();
        assert!((n * n * a * a * a - MU_EARTH).abs() / MU_EARTH < 1e-12);
        // ~8 h orbit ⟹ a ≈ 20 300 km, and perigee stays above the Earth.
        assert!((a - 20_270e3).abs() < 50e3, "a = {a}");
        assert!(a * (1.0 - fc.eccentricity) > 6_378e3);
    }

    #[test]
    fn reference_ground_track_repeats() {
        let fc = example();
        let sats = collect(&fc);
        assert_eq!(sats.len(), 6);
        let t_rep = f64::from(fc.days) * TAU / fc.frame_rate; // = Np·T_orbit
        for m in 0..20 {
            let t = t_rep * f64::from(m) / 20.0;
            let a = rot_pos(&sats[0], t, fc.frame_rate);
            let b = rot_pos(&sats[0], t + t_rep, fc.frame_rate);
            assert!(
                (a - b).norm() < 1e-3,
                "track not repeating: {} m",
                (a - b).norm()
            );
        }
    }

    #[test]
    fn all_satellites_share_one_space_track() {
        // The defining Flower property: every satellite is the reference, time-shifted by k·T_rep/Ns.
        // If Ω_k / M_k are right, sat k's rotating-frame position at t equals the reference's at
        // t + τ_k — same closed curve. A wrong phasing would show as a multi-thousand-km miss.
        let fc = example();
        let sats = collect(&fc);
        let ns = fc.satellites;
        let t_rep = f64::from(fc.days) * TAU / fc.frame_rate;
        let mut worst = 0.0_f64;
        for k in 1..ns {
            let tau_k = t_rep * f64::from(k) / f64::from(ns);
            for m in 0..24 {
                let t = t_rep * f64::from(m) / 24.0;
                let pk = rot_pos(&sats[k as usize], t, fc.frame_rate);
                let p0 = rot_pos(&sats[0], t + tau_k, fc.frame_rate);
                worst = nan_max(worst, (pk - p0).norm());
            }
        }
        assert!(
            worst < 1.0,
            "satellites not on one track: worst miss {worst:.3} m"
        );
    }

    #[test]
    fn phasing_satisfies_the_compatibility_relation() {
        let fc = example();
        let sats = collect(&fc);
        let (np, nd) = (f64::from(fc.petals), f64::from(fc.days));
        let raan0 = sats[0].raan.radians();
        let m0 = sats[0].mean_anomaly().radians();
        for s in &sats {
            let d_raan = s.raan.radians() - raan0;
            let d_m = s.mean_anomaly().radians() - m0;
            let resid = np * d_raan + nd * d_m; // ≡ 0 (mod 2π)
            let wrapped = resid - TAU * libm::round(resid / TAU);
            assert!(wrapped.abs() < 1e-9, "compatibility residual {wrapped:.2e}");
        }
    }

    /// (G2) Satellite `k` is the reference satellite time-shifted by `k·T_rep/Ns`.
    ///
    /// The defining property of a single-track Flower: a time shift `τ` moves the node relative to
    /// the rotating frame by `−ω·τ` and the mean anomaly by `n·τ`
    /// (`derivations/flower_phasing.mac`, 6 identities). The expected angles are built here from
    /// `τ_k`, `ω` and `n` — not from the `2π·k·Nd/Ns` closed form — and read back from the elements,
    /// the mean anomaly through the elements' own Kepler inversion. Bound: 2e-14 rad, the round-off
    /// of angles up to `2π·Np` plus the Kepler round trip at `e = 0.2`.
    #[test]
    fn every_satellite_is_the_reference_time_shifted_along_the_track() {
        for (np, nd, ns, m0) in [
            (3, 1, 6, 0.0),
            (14, 3, 8, 0.9),
            (4, 2, 6, 0.3),
            (2, 4, 8, 5.0),
        ] {
            let fc = FlowerConstellation {
                petals: np,
                days: nd,
                satellites: ns,
                mean_anomaly: Angle::from_radians(m0),
                raan: Angle::from_radians(0.7),
                ..example()
            };
            let n = fc.mean_motion();
            let t_rep = f64::from(nd) * TAU / OMEGA_EARTH;
            for (k, s) in collect(&fc).iter().enumerate() {
                let tau = t_rep * k as f64 / f64::from(ns);
                for (got, want, what) in [
                    (s.raan.radians(), 0.7 - OMEGA_EARTH * tau, "node"),
                    (s.mean_anomaly().radians(), m0 + n * tau, "mean anomaly"),
                ] {
                    let d = (got - want).rem_euclid(TAU);
                    let d = d.min(TAU - d);
                    assert!(
                        d < 2e-14,
                        "({np},{nd},{ns}) sat {k} {what}: {d:.2e} rad off"
                    );
                }
            }
        }
    }

    /// (G3) The repeating-orbit semi-major axis against `derivations/flower_phasing.gp`.
    ///
    /// `a = (μ/n²)^(1/3)` with `n = (Np/Nd)·ω`, at 60 digits in PARI/GP for the module's fixtures,
    /// Kepler's third law re-checked there at that precision. Bound: 4 ulp relative — five
    /// correctly rounded operations, a third of the error of `μ/n²` surviving the cube root.
    #[test]
    fn the_semi_major_axis_matches_the_derived_value() {
        for (np, nd, want) in [
            (3, 1, 2.027_041_982_595_185e7),
            (14, 3, 1.509_868_273_243_782_3e7),
            (2, 1, 2.656_176_415_793_235_1e7),
            (13, 1, 7.626_313_928_017_769e6),
        ] {
            let fc = FlowerConstellation {
                petals: np,
                days: nd,
                ..example()
            };
            let got = fc.semi_major_axis();
            assert!(
                (got - want).abs() <= 4.0 * f64::EPSILON * want,
                "({np},{nd}): a = {got} against {want}"
            );
        }
    }

    /// (G4) The satellites occupy exactly the positions the cyclic time-shift group predicts.
    ///
    /// Scaled by `Ns`, satellite `k` is the point `(−k·Nd, k·Np) mod Ns` of the node/mean-anomaly
    /// torus. `derivations/flower_phasing.g` (GAP) checks for every `Np, Nd, Ns ≤ 10` that this
    /// orbit has `Ns/gcd(Np, Nd, Ns)` distinct points on `Ns/gcd(Nd, Ns)` distinct nodes, so an
    /// unreduced `(Np, Nd)` sharing a factor with `Ns` stacks satellites on one another. Counted
    /// here from the emitted elements, rounded to that grid.
    #[test]
    fn distinct_satellites_and_nodes_are_the_group_orbit_counts() {
        for (np, nd, ns, sats, nodes) in [
            (3_u32, 1_u32, 6_u32, 6, 6),
            (14, 3, 8, 8, 8),
            (4, 2, 6, 3, 3),
            (2, 4, 8, 4, 2),
            (5, 3, 9, 9, 3),
        ] {
            let fc = FlowerConstellation {
                petals: np,
                days: nd,
                satellites: ns,
                ..example()
            };
            let grid = |angle: f64| {
                (libm::round(angle * f64::from(ns) / TAU) as i64).rem_euclid(i64::from(ns))
            };
            let mut points: Vec<(i64, i64)> = collect(&fc)
                .iter()
                .map(|s| (grid(s.raan.radians()), grid(s.mean_anomaly().radians())))
                .collect();
            points.sort_unstable();
            points.dedup();
            let mut node_set: Vec<i64> = points.iter().map(|p| p.0).collect();
            node_set.dedup();
            assert_eq!(
                (points.len(), node_set.len()),
                (sats, nodes),
                "({np},{nd},{ns})"
            );
        }
    }
}
