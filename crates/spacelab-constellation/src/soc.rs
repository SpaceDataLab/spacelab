//! Street-of-Coverage (`SoC`) near-polar constellations for continuous single-fold coverage.
//!
//! A single plane of `S` satellites paints a *street* — a band of continuous coverage `2·λ_street`
//! wide along its ground track (Lüders 1961). Adams & Rider (1985–87) tile the globe with `P` such
//! near-polar streets: co-rotating neighbours may sit a wide `ΔΩ_co` apart because their footprints
//! phase-mesh, while the one counter-rotating seam (first plane vs last) must sit a tighter
//! `ΔΩ_counter` apart. The geometry:
//!
//! - footprint half-angle `θ = arccos((Re/(Re+h))·cos ε) − ε`
//! - street half-width  `cos λ_street = cos θ / cos(π/S)`, with `S ≥ π/θ`
//! - co-rotating gap    `ΔΩ_co = 2·asin(sin((θ+λ)/2) / sin i)`
//! - counter gap        `ΔΩ_counter = 2·asin(sin λ / sin i)`
//! - node span          `Ω_max = π − ΔΩ_counter`, `(P−1)·ΔΩ = Ω_max`
//! - inter-plane stagger `Δu_inter = π/S − 2·acos(cos(ΔΩ_co/2)/cos((θ+λ)/2))`
//!
//! Both node-spacing forms satisfy `sin(ΔΩ/2) = sin(D/2)/sin i` for their touch distance `D`
//! (`D_co = θ+λ`, `D_op = 2λ`), which is what the plane-count construction is built on; this also
//! gives the physically-required ordering `ΔΩ_co > ΔΩ_counter` whenever `θ > λ_street` (always true
//! here). Verified end-to-end against a direct time-sampled coverage oracle in the tests — the real
//! check on the phasing.

// Casts: every `as u32` below is of a small, positive, integer-valued f64 count (plane / satellite
// tallies), so truncation and sign loss cannot occur.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use spacelab_core::Angle;
use spacelab_elements::KeplerianElements;

const TAU: f64 = core::f64::consts::TAU;
const PI: f64 = core::f64::consts::PI;

/// Design inputs for a near-polar Street-of-Coverage constellation of circular orbits, sized for
/// continuous single-fold coverage above a minimum elevation.
#[derive(Clone, Copy, Debug)]
pub struct StreetOfCoverage {
    /// Common inclination `i` (near-polar; the node-spacing arcsines stop being real well away from
    /// 90°, and [`StreetOfCoverage::geometry`] then returns `None`).
    pub inclination: Angle,
    /// Orbit altitude above the reference sphere, metres (circular).
    pub altitude: f64,
    /// Minimum elevation a ground point must see a satellite at for it to count as covered.
    pub min_elevation: Angle,
    /// Reference Earth radius, metres.
    pub earth_radius: f64,
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Optional override for satellites per plane `S`. `None` picks the minimum `⌈π/θ⌉`; a value
    /// below that minimum makes the street ill-defined and `geometry` returns `None`.
    pub per_plane: Option<u32>,
}

/// The solved geometry of a [`StreetOfCoverage`] design — every quantity the continuity conditions
/// depend on, exposed so it can be inspected and checked independently of the emitted elements. All
/// angles are radians.
#[derive(Clone, Copy, Debug)]
pub struct SocGeometry {
    /// Earth-central footprint half-angle `θ`.
    pub footprint_half_angle: f64,
    /// Satellites per plane `S`.
    pub per_plane: u32,
    /// Number of planes `P`.
    pub planes: u32,
    /// Street half-width `λ_street`.
    pub street_half_width: f64,
    /// Maximum node spacing between co-rotating planes `ΔΩ_co`.
    pub raan_step_co: f64,
    /// Node spacing of the counter-rotating seam `ΔΩ_counter`.
    pub raan_step_counter: f64,
    /// Actual, evenly-filled node spacing between adjacent planes `ΔΩ ≤ ΔΩ_co`.
    pub raan_step: f64,
    /// Ascending-node span of the plane set `Ω_max = π − ΔΩ_counter`.
    pub raan_span: f64,
    /// Inter-plane argument-of-latitude stagger `Δu_inter`.
    pub inter_plane_phase: f64,
}

impl StreetOfCoverage {
    /// Total satellites `P · S`, or 0 when the geometry is infeasible.
    #[must_use]
    pub fn total(&self) -> u32 {
        self.geometry().map_or(0, |g| g.planes * g.per_plane)
    }

    /// Solve the `SoC` geometry (footprint, street, node spacings, plane count, phasing). Returns
    /// `None` when the inputs cannot form a tiled street: a non-positive footprint, an `S` below the
    /// `π/θ` continuity floor, an inclination too far from polar for the node-spacing arcsines to be
    /// real, or a degenerate node span.
    #[must_use]
    pub fn geometry(&self) -> Option<SocGeometry> {
        let sin_i = libm::sin(self.inclination.radians());
        let eps = self.min_elevation.radians();
        let ratio = self.earth_radius / (self.earth_radius + self.altitude);
        // θ = arccos(ratio · cos ε) − ε   (the SMAD Earth-central-angle relation).
        let theta = libm::acos(ratio * libm::cos(eps)) - eps;
        if theta <= 0.0 {
            return None;
        }
        let s_min = libm::ceil(PI / theta);
        let s = self.per_plane.map_or(s_min, f64::from);
        if s < 2.0 || s < s_min {
            return None;
        }
        let half_du = PI / s; // Δu₁₂/2 = π/S
        // λ_street = arccos(cos θ / cos(π/S)); the S ≥ π/θ floor keeps the argument ≤ 1.
        let cos_lambda = libm::cos(theta) / libm::cos(half_du);
        if cos_lambda > 1.0 {
            return None;
        }
        let lambda = libm::acos(cos_lambda);
        // ΔΩ_co = 2·asin(sin((θ+λ)/2) / sin i);
        // ΔΩ_counter = 2·asin(sin λ / sin i).
        let arg_co = libm::sin(f64::midpoint(theta, lambda)) / sin_i;
        let arg_counter = libm::sin(lambda) / sin_i;
        if arg_co > 1.0 || arg_counter > 1.0 {
            return None; // inclination too far from polar
        }
        let raan_step_co = 2.0 * libm::asin(arg_co);
        let raan_step_counter = 2.0 * libm::asin(arg_counter);
        let raan_span = PI - raan_step_counter; // Ω_max, last plane RAAN < 180°
        if raan_span <= 0.0 {
            return None;
        }
        let planes = libm::ceil(raan_span / raan_step_co) as u32 + 1;
        let raan_step = raan_span / f64::from(planes - 1);
        // Δu_inter = π/S − 2·acos(cos(ΔΩ_co/2) / cos((θ+λ)/2)), at the max co spacing —
        // ΔΩ_co/2 ≥ (θ+λ)/2 keeps the ratio ≤ 1. Closer planes (P > P_min) only add margin.
        let phase_arg = libm::cos(0.5 * raan_step_co) / libm::cos(f64::midpoint(theta, lambda));
        let inter_plane_phase = half_du - 2.0 * libm::acos(phase_arg.min(1.0));
        Some(SocGeometry {
            footprint_half_angle: theta,
            per_plane: s as u32,
            planes,
            street_half_width: lambda,
            raan_step_co,
            raan_step_counter,
            raan_step,
            raan_span,
            inter_plane_phase,
        })
    }

    /// Emit the constellation's circular elements, plane-major, or nothing if the geometry is
    /// infeasible. Plane `p` sits at RAAN `p·ΔΩ`; satellite `s` in it is at argument of latitude
    /// `s·2π/S + p·Δu_inter` (carried in the true anomaly of the circular orbit).
    pub fn generate<E: FnMut(KeplerianElements)>(&self, mut emit: E) {
        let Some(g) = self.geometry() else {
            return;
        };
        let a = self.earth_radius + self.altitude;
        let du = TAU / f64::from(g.per_plane);
        for p in 0..g.planes {
            let raan = f64::from(p) * g.raan_step;
            let phase = f64::from(p) * g.inter_plane_phase;
            for s in 0..g.per_plane {
                let u = libm::fmod(f64::from(s) * du + phase, TAU);
                let u = if u < 0.0 { u + TAU } else { u };
                emit(KeplerianElements {
                    semi_major_axis: a,
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
    #![allow(clippy::expect_used)]
    use super::*;
    use spacelab_core::Vec3;

    const MU_EARTH: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;

    fn iridium_like() -> StreetOfCoverage {
        StreetOfCoverage {
            inclination: Angle::from_radians(86.4_f64.to_radians()),
            altitude: 780e3,
            min_elevation: Angle::from_radians(8.2_f64.to_radians()),
            earth_radius: RE,
            mu: MU_EARTH,
            per_plane: None,
        }
    }

    #[test]
    fn geometry_matches_the_hand_computed_street() {
        let soc = iridium_like();
        let g = soc.geometry().expect("feasible near-polar SoC");
        // Cross-check θ through an independent relation: a ground point at central angle θ must see
        // the satellite at exactly ε_min, via tan(el) = (cos θ − Re/(Re+h)) / sin θ.
        let theta = g.footprint_half_angle;
        let ratio = RE / (RE + soc.altitude);
        let el = libm::atan2(libm::cos(theta) - ratio, libm::sin(theta));
        assert!(
            (el - soc.min_elevation.radians()).abs() < 1e-9,
            "footprint edge elevation {} != eps",
            el.to_degrees()
        );
        assert!((15.0..25.0).contains(&theta.to_degrees()));
        // S ≥ ⌈π/θ⌉ = 10; the street is narrower than the footprint.
        assert_eq!(g.per_plane, 10);
        assert!(g.street_half_width < g.footprint_half_angle);
        // cos λ = cos θ / cos(π/S) exactly (eq 7).
        let expect_cos = libm::cos(g.footprint_half_angle) / libm::cos(PI / f64::from(g.per_plane));
        assert!((libm::cos(g.street_half_width) - expect_cos).abs() < 1e-12);
        // Co-rotating planes sit wider apart than the counter-rotating seam.
        assert!(g.raan_step_co > g.raan_step_counter);
        // The evenly-filled spacing never exceeds the co-rotating maximum (continuity).
        assert!(g.raan_step <= g.raan_step_co + 1e-12);
        assert!(g.planes >= 6);
    }

    /// Every field of the solved geometry, against the street-of-coverage relations at 60 digits.
    ///
    /// The test above pins the footprint half-angle through the elevation relation and then checks
    /// the rest only by inequality — that the co-rotating spacing exceeds the counter-rotating one,
    /// that the filled spacing does not exceed the maximum. Inequalities are satisfied by a great
    /// many wrong numbers: the two node spacings, the node span and the inter-plane stagger were
    /// unpinned, and with them every arithmetic operator that builds them. Reference values from
    /// `tools/gen_soc_oracle.mac`, transcribed independently from the construction (Lüders 1961;
    /// Rider 1986; SMAD ch. 7). The second case forces `S` above the floor, which is the only way
    /// to move `π/S` and everything downstream of it.
    #[test]
    fn every_solved_quantity_matches_the_street_of_coverage_relations() {
        struct Case {
            per_plane: Option<u32>,
            planes: u32,
            satellites: u32,
            footprint_half_angle: f64,
            street_half_width: f64,
            raan_step_co: f64,
            raan_step_counter: f64,
            raan_step: f64,
            raan_span: f64,
            inter_plane_phase: f64,
        }
        const CASES: [Case; 2] = [
            Case {
                per_plane: None,
                planes: 7,
                satellites: 70,
                footprint_half_angle: 3.477_523_587_149_594_6e-1,
                street_half_width: 1.516_523_759_528_832_5e-1,
                raan_step_co: 5.004_132_563_157_829e-1,
                raan_step_counter: 3.039_090_918_276_771_7e-1,
                raan_step: 4.729_472_602_936_86e-1,
                raan_span: 2.837_683_561_762_116e0,
                inter_plane_phase: 2.820_682_740_153_738e-1,
            },
            Case {
                per_plane: Some(12),
                planes: 6,
                satellites: 72,
                footprint_half_angle: 3.477_523_587_149_594_6e-1,
                street_half_width: 2.315_771_215_736_105e-1,
                raan_step_co: 5.805_081_676_819_301e-1,
                raan_step_counter: 4.640_867_591_701_013_4e-1,
                raan_step: 5.355_011_788_839_383e-1,
                raan_span: 2.677_505_894_419_691_7e0,
                inter_plane_phase: 2.242_940_448_497_582e-1,
            },
        ];

        for case in CASES {
            let mut soc = iridium_like();
            soc.per_plane = case.per_plane;
            let g = soc.geometry().expect("feasible near-polar SoC");
            assert_eq!(g.planes, case.planes, "planes");
            assert_eq!(soc.total(), case.satellites, "P · S");
            let close = |got: f64, want: f64, what: &str| {
                assert!(
                    (got - want).abs() < 1e-12,
                    "{what}: {got:.17e} vs {want:.17e}"
                );
            };
            close(g.footprint_half_angle, case.footprint_half_angle, "θ");
            close(g.street_half_width, case.street_half_width, "λ");
            close(g.raan_step_co, case.raan_step_co, "ΔΩ_co");
            close(g.raan_step_counter, case.raan_step_counter, "ΔΩ_counter");
            close(g.raan_step, case.raan_step, "ΔΩ");
            close(g.raan_span, case.raan_span, "Ω_max");
            close(g.inter_plane_phase, case.inter_plane_phase, "Δu_inter");
        }
    }

    /// The emitted elements are the geometry the solver reported, laid out plane-major.
    ///
    /// The coverage oracle below scores the constellation as a set and is indifferent to how the
    /// satellites are labelled: a stagger applied with the wrong sign, or a semi-major axis built by
    /// subtraction, moves every satellite together and can still cover the sphere. This reads the
    /// stream back against `raan = p·ΔΩ` and `u = s·2π/S + p·Δu_inter`, and requires the reported
    /// anomaly to be the representative in `[0, 2π)` — which is what the wrap exists to produce, and
    /// which is not vacuous here because the stagger is large enough to send `fmod` negative.
    #[test]
    fn the_emitted_elements_lay_out_the_reported_geometry() {
        // A near-polar case with a positive stagger, and a low-inclination one whose stagger is
        // negative — the only way `fmod` returns a negative angle and the wrap does anything.
        let low = StreetOfCoverage {
            inclination: Angle::from_radians(60.0_f64.to_radians()),
            altitude: 300e3,
            min_elevation: Angle::from_radians(0.0),
            earth_radius: RE,
            mu: MU_EARTH,
            per_plane: Some(21),
        };

        let mut wrapped = 0;
        for soc in [iridium_like(), low] {
            let g = soc.geometry().expect("feasible");
            let du = TAU / f64::from(g.per_plane);

            let mut got = Vec::new();
            soc.generate(|e| got.push(e));
            assert_eq!(got.len() as u32, g.planes * g.per_plane);

            for (index, e) in got.iter().enumerate() {
                let p = index as u32 / g.per_plane;
                let s = index as u32 % g.per_plane;
                assert!(
                    (e.semi_major_axis - (soc.earth_radius + soc.altitude)).abs() < 1e-6,
                    "a[{index}] = {}",
                    e.semi_major_axis
                );
                assert!(
                    e.eccentricity.abs() < f64::EPSILON,
                    "e[{index}] must be circular"
                );
                assert!(
                    (e.raan.radians() - f64::from(p) * g.raan_step).abs() < 1e-12,
                    "Ω[{index}]"
                );

                let raw = libm::fmod(f64::from(s) * du + f64::from(p) * g.inter_plane_phase, TAU);
                if raw < 0.0 {
                    wrapped += 1;
                }
                let want = if raw < 0.0 { raw + TAU } else { raw };
                assert!(
                    (e.true_anomaly.radians() - want).abs() < 1e-12,
                    "u[{index}] = {} vs {want}",
                    e.true_anomaly.radians()
                );
                assert!(
                    (0.0..TAU).contains(&e.true_anomaly.radians()),
                    "u[{index}] = {} is not the representative in [0, 2π)",
                    e.true_anomaly.radians()
                );
            }
        }
        assert!(
            wrapped > 0,
            "no satellite exercised the negative-angle wrap"
        );
    }

    #[test]
    fn below_the_per_plane_floor_is_infeasible() {
        let mut soc = iridium_like();
        soc.per_plane = Some(4); // < ⌈π/θ⌉ = 10 → no continuous street
        assert!(soc.geometry().is_none());
        assert_eq!(soc.total(), 0);
    }

    /// The node-spacing guard refuses an arcsine that would be imaginary, and nothing else.
    ///
    /// Both node spacings divide a sine by `sin i`, and both arcsines are real only while the
    /// quotient stays at or below 1. Every other fixture here is near-polar, where the quotients
    /// sit far below the guard, so its *shape* went unpinned: which comparison it uses, and which
    /// side of the boundary is feasible. Two cases pin both. Away from polar the co-rotating
    /// quotient alone exceeds 1, so the guard must fire on it. On the boundary itself — where
    /// `sin i` is bit-for-bit the numerator, making the quotient exactly 1 — the design is still
    /// feasible: `ΔΩ_co = 2·asin(1) = π` is one co-rotating seam per hemisphere, not zero of them.
    #[test]
    #[allow(clippy::float_cmp)] // the boundary case is exact by construction, see below
    fn the_node_spacing_guard_refuses_only_an_arcsine_that_would_be_imaginary() {
        let mut soc = iridium_like();
        // sin((θ+λ)/2)/sin 10° = 1.42 > 1, while sin λ/sin 10° = 0.87 is still real.
        soc.inclination = Angle::from_radians(10.0_f64.to_radians());
        assert!(soc.geometry().is_none());
        assert_eq!(soc.total(), 0);

        // Rebuild θ and λ exactly as `geometry` does, then put the inclination on (θ+λ)/2: the
        // quotient is then a number divided by itself, i.e. exactly 1.0 whatever the two round to.
        let eps = soc.min_elevation.radians();
        let theta = libm::acos(RE / (RE + soc.altitude) * libm::cos(eps)) - eps;
        let lambda = libm::acos(libm::cos(theta) / libm::cos(PI / libm::ceil(PI / theta)));
        soc.inclination = Angle::from_radians(f64::midpoint(theta, lambda));
        let g = soc
            .geometry()
            .expect("the boundary belongs to the feasible side");
        assert_eq!(g.raan_step_co, PI);
        assert!(g.raan_step_counter < PI, "{}", g.raan_step_counter);
        assert!(g.raan_span > 0.0);
    }

    // Fibonacci-sphere sampling — a near-uniform point set on the unit sphere (the grid the thesis
    // itself uses to score coverage).
    fn fibonacci_sphere(n: u32) -> Vec<Vec3> {
        let ga = PI * (3.0 - libm::sqrt(5.0)); // golden angle
        (0..n)
            .map(|k| {
                let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(n);
                let r = libm::sqrt((1.0 - z * z).max(0.0));
                let phi = ga * f64::from(k);
                Vec3::new(r * libm::cos(phi), r * libm::sin(phi), z)
            })
            .collect()
    }

    #[test]
    fn constellation_gives_continuous_global_coverage() {
        // The real oracle: build the constellation, then over one orbital period check that at every
        // sampled instant every sampled inertial direction lies within the footprint θ of some
        // satellite. Streets are fixed in inertial space (nodes fixed, all periods equal), so
        // covering every inertial direction at all times ⟹ covering every rotating ground point.
        let soc = iridium_like();
        let g = soc.geometry().expect("feasible");
        let mut sats = Vec::new();
        soc.generate(|e| sats.push(e));
        assert_eq!(sats.len() as u32, g.planes * g.per_plane);

        let a = soc.earth_radius + soc.altitude;
        let period = TAU * libm::sqrt(a * a * a / MU_EARTH);
        let cos_theta = libm::cos(g.footprint_half_angle);
        let grid = fibonacci_sphere(700);

        let n_times: u32 = 120;
        let mut worst_margin = f64::INFINITY; // min over (t, point) of (best cos_angle − cos θ)
        for it in 0..n_times {
            let t = period * f64::from(it) / f64::from(n_times);
            let dirs: Vec<Vec3> = sats.iter().map(|e| e.propagate(t).to_state().0).collect();
            for gpt in &grid {
                // best coverage of this direction = max cos(central angle) over all satellites.
                let mut best = -1.0_f64;
                for d in &dirs {
                    best = best.max(d.dot(*gpt) / d.norm());
                }
                worst_margin = worst_margin.min(best - cos_theta);
            }
        }
        // A positive margin everywhere ⟹ continuous global coverage. A small negative tolerance
        // absorbs the finite time/grid sampling landing just inside a footprint edge.
        assert!(
            worst_margin > -2e-3,
            "coverage gap: worst (cosθ_best − cosθ) = {worst_margin:.2e}"
        );
    }
}
