//! Walker constellations (Delta/Star), the standard `i: T/P/F` symmetric pattern (Walker 1984).

use spacelab_core::Angle;
use spacelab_elements::KeplerianElements;

const TAU: f64 = core::f64::consts::TAU;

/// The right-ascension span of a Walker pattern's planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkerPattern {
    /// **Delta** (δ): the planes span the full `2π` of right ascension — the general symmetric case
    /// (GNSS-style, e.g. Galileo 56°: 24/3/1).
    Delta,
    /// **Star**: the planes span `π` (one half-plane of ascending nodes), the near-polar layout where
    /// ascending passes fill one hemisphere of nodes and descending passes the other (e.g. Iridium).
    Star,
}

/// A Walker constellation of circular orbits, notation `i: T/P/F`. `T = planes · per_plane`
/// satellites sit in `planes` equally-spaced planes of `per_plane` each; adjacent planes are phased by
/// `phasing · 2π/T` in argument of latitude. Taking `planes` and `per_plane` (rather than the total
/// `T`) makes `T` divisible by `P` by construction — an illegal Walker cannot be expressed.
#[derive(Clone, Copy, Debug)]
pub struct Walker {
    /// Node spread of the planes.
    pub pattern: WalkerPattern,
    /// Common inclination `i`.
    pub inclination: Angle,
    /// Number of orbital planes `P`.
    pub planes: u32,
    /// Satellites per plane `S`, so the total is `T = P · S`.
    pub per_plane: u32,
    /// Phasing parameter `F` (used mod `P`) — the inter-plane relative phasing.
    pub phasing: u32,
    /// Common semi-major axis, metres (circular, so equal to the orbit radius).
    pub semi_major_axis: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

impl Walker {
    /// Total number of satellites `T = planes · per_plane`.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.planes * self.per_plane
    }

    /// Emit each satellite's circular elements, plane-major (all of plane 0, then plane 1, …). Every
    /// set has `e = 0` and `ω = 0`; the in-plane position (argument of latitude) is carried in the
    /// true anomaly, as [`KeplerianElements::from_state`] does for a circular orbit. A degenerate
    /// pattern (zero planes or zero per plane) emits nothing.
    pub fn generate<E: FnMut(KeplerianElements)>(&self, mut emit: E) {
        let (planes, per_plane) = (self.planes, self.per_plane);
        if planes == 0 || per_plane == 0 {
            return;
        }
        let total = f64::from(planes * per_plane);
        let node_span = match self.pattern {
            WalkerPattern::Delta => TAU,
            WalkerPattern::Star => TAU / 2.0,
        };
        let phasing = f64::from(self.phasing % planes);
        for plane in 0..planes {
            let raan = f64::from(plane) * node_span / f64::from(planes);
            let phase_offset = f64::from(plane) * phasing * TAU / total;
            for slot in 0..per_plane {
                let u = libm::fmod(
                    f64::from(slot) * TAU / f64::from(per_plane) + phase_offset,
                    TAU,
                );
                emit(KeplerianElements {
                    semi_major_axis: self.semi_major_axis,
                    eccentricity: 0.0,
                    inclination: self.inclination,
                    raan: Angle::from_radians(raan),
                    arg_periapsis: Angle::from_radians(0.0),
                    true_anomaly: Angle::from_radians(u),
                    mu: self.mu,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    const MU_EARTH: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;

    fn collect(w: &Walker) -> Vec<KeplerianElements> {
        let mut v = Vec::new();
        w.generate(|e| v.push(e));
        v
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn walker_delta_has_the_right_count_and_plane_spacing() {
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(56.0_f64.to_radians()),
            planes: 3,
            per_plane: 8,
            phasing: 1,
            semi_major_axis: RE + 23_222e3, // Galileo-like MEO
            mu: MU_EARTH,
        };
        let sats = collect(&w);
        assert_eq!(sats.len(), 24);
        assert_eq!(w.total(), 24);
        // Three planes at RAAN 0, 120, 240°.
        for (plane, expected_deg) in [(0usize, 0.0_f64), (8, 120.0), (16, 240.0)] {
            assert!(
                approx(sats[plane].raan.radians(), expected_deg.to_radians(), 1e-12),
                "plane starting at sat {plane} raan"
            );
        }
        // All circular at the same inclination and altitude.
        for s in &sats {
            assert_eq!(s.eccentricity, 0.0);
            assert!(approx(
                s.inclination.radians(),
                56.0_f64.to_radians(),
                1e-12
            ));
            assert!(approx(s.semi_major_axis, RE + 23_222e3, 1e-6));
        }
    }

    #[test]
    fn walker_delta_phasing_matches_the_definition() {
        // i:6/3/1 — hand-computable: planes at {0,120,240}°, in-plane step 180°, inter-plane phase
        // step F·360/T = 60°.
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(1.0),
            planes: 3,
            per_plane: 2,
            phasing: 1,
            semi_major_axis: RE + 800e3,
            mu: MU_EARTH,
        };
        let s = collect(&w);
        let u: Vec<f64> = s
            .iter()
            .map(|e| e.true_anomaly.radians().to_degrees())
            .collect();
        // plane 0: 0,180 ; plane 1: 60,240 ; plane 2: 120,300
        let expect = [0.0, 180.0, 60.0, 240.0, 120.0, 300.0];
        for (got, want) in u.iter().zip(expect) {
            assert!(approx(*got, want, 1e-9), "u {got} vs {want}");
        }
    }

    #[test]
    fn walker_star_spans_half_the_node_circle() {
        let w = Walker {
            pattern: WalkerPattern::Star,
            inclination: Angle::from_radians(86.4_f64.to_radians()),
            planes: 6,
            per_plane: 11,
            phasing: 2,
            semi_major_axis: RE + 780e3, // Iridium-like
            mu: MU_EARTH,
        };
        let sats = collect(&w);
        assert_eq!(sats.len(), 66);
        // Planes span [0, π): last plane's RAAN is 5·(180/6) = 150°, not near 360°.
        let last_plane_raan = sats[5 * 11].raan.radians().to_degrees();
        assert!(
            approx(last_plane_raan, 150.0, 1e-9),
            "star last plane raan {last_plane_raan}"
        );
    }

    #[test]
    fn generated_elements_round_trip_to_a_circular_orbit() {
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(0.9),
            planes: 4,
            per_plane: 5,
            phasing: 1,
            semi_major_axis: RE + 1200e3,
            mu: MU_EARTH,
        };
        for e in collect(&w) {
            let (r, v) = e.to_state();
            // Circular: |r| = a, and the round-trip recovers the same elements.
            assert!(approx(r.norm(), RE + 1200e3, 1e-3), "radius {}", r.norm());
            let back = KeplerianElements::from_state(r, v, MU_EARTH);
            assert!(approx(back.semi_major_axis, e.semi_major_axis, 1.0));
            assert!(back.eccentricity < 1e-9);
        }
    }
}
