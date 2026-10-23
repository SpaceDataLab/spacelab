//! Draim's four-satellite continuous global coverage constellation (Draim, *A common-period
//! four-satellite continuous global coverage constellation*, J. Guidance 1987; US patent 4,854,527).
//!
//! Four common-period elliptical satellites give continuous whole-Earth visibility with the smallest
//! satellite count for which that is possible. Two satellites hold perigee in the southern hemisphere
//! and two in the northern; the four are staggered 90° in right ascension and a quarter revolution in
//! mean anomaly. The patent's worked example (Table I) — period 48 h, `i = 33°`, `e = 0.28`, with
//!
//! ```text
//! satellite:   1     2     3     4
//! ω (deg):   270    90   270    90     (−90 = perigee south, +90 = perigee north)
//! Ω (deg):     0    90   180   270
//! M (deg):     0   270   180    90
//! ```
//!
//! achieves a minimum global look angle of 2.1°, i.e. every point on Earth always sees at least one
//! satellite at least 2.1° above the horizon.

use spacelab_core::Angle;
use spacelab_elements::KeplerianElements;

const TAU: f64 = core::f64::consts::TAU;

/// Right ascension of the four satellites, degrees (90° apart).
const RAAN_DEG: [f64; 4] = [0.0, 90.0, 180.0, 270.0];
/// Argument of perigee, degrees — alternating south/north perigees.
const ARGP_DEG: [f64; 4] = [270.0, 90.0, 270.0, 90.0];
/// Mean anomaly, degrees — quarter-revolution stagger.
const MEAN_DEG: [f64; 4] = [0.0, 270.0, 180.0, 90.0];

/// A Draim four-satellite continuous global coverage constellation. All four share `(a, e, i)`; the
/// per-satellite `(ω, Ω, M)` are the fixed Draim pattern, rigidly offset in right ascension by
/// `raan0`.
#[derive(Clone, Copy, Debug)]
pub struct DraimTetrahedral {
    /// Common semi-major axis, metres.
    pub semi_major_axis: f64,
    /// Common eccentricity `e`.
    pub eccentricity: f64,
    /// Common inclination `i`.
    pub inclination: Angle,
    /// Rigid right-ascension offset applied to all four satellites.
    pub raan0: Angle,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

impl DraimTetrahedral {
    /// Build from an orbital `period` (seconds): `a = (μ (T/2π)²)^(1/3)`.
    #[must_use]
    pub fn from_period(
        period: f64,
        eccentricity: f64,
        inclination: Angle,
        raan0: Angle,
        mu: f64,
    ) -> Self {
        let a = libm::cbrt(mu * (period / TAU) * (period / TAU));
        Self {
            semi_major_axis: a,
            eccentricity,
            inclination,
            raan0,
            mu,
        }
    }

    /// The patent's worked example: 48-hour period, `i = 33°`, `e = 0.28`, zero right-ascension
    /// offset — the configuration with a 2.1° minimum global look angle.
    #[must_use]
    pub fn nominal(mu: f64) -> Self {
        Self::from_period(
            48.0 * 3600.0,
            0.28,
            Angle::from_radians(33.0_f64.to_radians()),
            Angle::from_radians(0.0),
            mu,
        )
    }

    /// Total number of satellites — always 4.
    #[must_use]
    pub const fn total(&self) -> u32 {
        4
    }

    /// Emit the four satellites' elements.
    pub fn generate<E: FnMut(KeplerianElements)>(&self, mut emit: E) {
        let raan0 = self.raan0.radians();
        for k in 0..4 {
            let base = KeplerianElements {
                semi_major_axis: self.semi_major_axis,
                eccentricity: self.eccentricity,
                inclination: self.inclination,
                raan: Angle::from_radians(raan0 + RAAN_DEG[k].to_radians()),
                arg_periapsis: Angle::from_radians(ARGP_DEG[k].to_radians()),
                true_anomaly: Angle::from_radians(0.0),
                mu: self.mu,
            };
            emit(base.at_mean_anomaly(Angle::from_radians(MEAN_DEG[k].to_radians())));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacelab_core::Vec3;

    const MU: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;

    fn collect(d: &DraimTetrahedral) -> Vec<KeplerianElements> {
        let mut v = Vec::new();
        d.generate(|e| v.push(e));
        v
    }

    fn fibonacci_sphere(n: u32) -> Vec<Vec3> {
        let ga = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        (0..n)
            .map(|k| {
                let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(n);
                let r = (1.0 - z * z).max(0.0).sqrt();
                let phi = ga * f64::from(k);
                Vec3::new(r * phi.cos(), r * phi.sin(), z)
            })
            .collect()
    }

    #[test]
    fn nominal_geometry_matches_the_patent() {
        let d = DraimTetrahedral::nominal(MU);
        // 48-hour period ⟹ a ≈ 67 000 km, and perigee well above the Earth at e = 0.28.
        let period = TAU * (d.semi_major_axis.powi(3) / MU).sqrt();
        assert!((period - 48.0 * 3600.0).abs() < 1.0, "period {period}");
        assert!(d.semi_major_axis * (1.0 - d.eccentricity) > RE);
        let sats = collect(&d);
        assert_eq!(sats.len(), 4);
        // Two perigees south (ω = 270°), two north (ω = 90°).
        let north = sats
            .iter()
            .filter(|s| (s.arg_periapsis.radians().to_degrees() - 90.0).abs() < 1e-6)
            .count();
        assert_eq!(north, 2);
    }

    #[test]
    fn four_satellites_give_continuous_global_coverage() {
        // The defining Draim property: at every instant every point on Earth sees at least one
        // satellite above the horizon. In the inertial frame (satellites move, the pattern repeats
        // each period) this is: every direction covered at every sampled time. A ground point at unit
        // direction ĝ sees satellite s at elevation ≥ 0 iff s·ĝ ≥ Re. If the patent's elements were
        // wrong, this would fail — so it doubles as a check on the parameters.
        let d = DraimTetrahedral::nominal(MU);
        let sats = collect(&d);
        let period = TAU * (d.semi_major_axis.powi(3) / MU).sqrt();
        let grid = fibonacci_sphere(600);

        let n_times = 400;
        let mut worst = f64::INFINITY; // min over (t, ĝ) of the best (s·ĝ − Re)
        for it in 0..n_times {
            let t = period * f64::from(it) / f64::from(n_times);
            let pos: Vec<Vec3> = sats.iter().map(|e| e.propagate(t).to_state().0).collect();
            for g in &grid {
                let mut best = f64::NEG_INFINITY;
                for s in &pos {
                    best = best.max(s.dot(*g));
                }
                worst = worst.min(best - RE);
            }
        }
        // A positive worst margin ⟹ continuous global visibility above the horizon. The patent's 2.1°
        // look-angle margin makes this comfortably positive.
        assert!(
            worst > 0.0,
            "coverage gap: worst (s·ĝ − Re) = {worst:.3e} m"
        );
    }
}
