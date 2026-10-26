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
//! - counter gap        `ΔΩ_counter = 2·asin(√(sin²λ − cos²i) / sin i)`, real only for `λ ≥ π/2 − i`
//! - node span          `Ω_max = π − ΔΩ_counter`, `(P−1)·ΔΩ = Ω_max`
//! - inter-plane stagger `Δu_inter = π/S − 2·atan(cos i · tan(ΔΩ_co/2))`
//!
//! Each node spacing makes the dihedral angle of its pair of planes — the largest distance between
//! their tracks — equal to the touch distance: `θ+λ` for co-rotating neighbours, whose footprints
//! phase-mesh, and `2λ` at the seam. Co-rotating planes `ΔΩ` apart meet at `sin(a/2) = sin i ·
//! sin(ΔΩ/2)`; the seam pairs one plane's ascending pass with the other's descending pass, which
//! cross at `2(π/2 − i)` on the equator, so there `sin²(a/2) = cos²i + sin²i·sin²(ΔΩ/2)`. Every
//! relation is derived from vectors in `derivations/soc_street_geometry.mac` (18 identities), with
//! the reference values re-computed in `derivations/soc_street_geometry.gp`; the gate is
//! `every_solved_quantity_matches_the_derived_street_geometry` and
//! `the_emitted_planes_touch_at_the_derived_distances` below.

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
    /// Common inclination `i` (near-polar; below `π/2 − λ_street` the seam cannot close, and
    /// [`StreetOfCoverage::geometry`] then returns `None`).
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
    /// real (the seam needs `λ_street ≥ π/2 − i`), or a degenerate node span.
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
        // ΔΩ_counter = 2·asin(√(sin²λ − cos²i) / sin i), the factored difference avoiding the
        // cancellation of two nearly equal squares near the λ = π/2 − i limit.
        let arg_co = libm::sin(f64::midpoint(theta, lambda)) / sin_i;
        let (sin_lambda, cos_i) = (libm::sin(lambda), libm::cos(self.inclination.radians()));
        let seam_sq = (sin_lambda - cos_i.abs()) * (sin_lambda + cos_i.abs());
        if arg_co > 1.0 || seam_sq < 0.0 {
            return None; // inclination too far from polar
        }
        let arg_counter = libm::sqrt(seam_sq) / sin_i;
        let raan_step_co = 2.0 * libm::asin(arg_co);
        let raan_step_counter = 2.0 * libm::asin(arg_counter);
        let raan_span = PI - raan_step_counter; // Ω_max, last plane RAAN < 180°
        if raan_span <= 0.0 {
            return None;
        }
        let planes = libm::ceil(raan_span / raan_step_co) as u32 + 1;
        let raan_step = raan_span / f64::from(planes - 1);
        // Δu_inter = π/S − 2b, b = atan(cos i · tan(ΔΩ_co/2)) the signed argument of latitude at
        // which a plane is farthest from its neighbour, at the max co spacing. Its cosine is
        // cos(ΔΩ_co/2)/cos((θ+λ)/2), but an arccosine drops the sign that a retrograde plane needs
        // and loses digits as b → 0. Closer planes (ΔΩ < ΔΩ_co) are not re-phased.
        let foot =
            libm::atan(libm::cos(self.inclination.radians()) * libm::tan(0.5 * raan_step_co));
        let inter_plane_phase = half_du - 2.0 * foot;
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

    /// (G1) Every field of the solved geometry, against `derivations/soc_street_geometry.mac`.
    ///
    /// The derivation proves each relation from vectors on the sphere (18 identities) and prints
    /// these reference values at 60 digits; `derivations/soc_street_geometry.gp` re-computes them in
    /// PARI/GP and agrees to all twenty digits printed. Five cases: the Iridium-like street at the
    /// floor `S` and at a forced `S = 12` (the only way to move `π/S` and everything downstream), a
    /// second near-polar shell, a retrograde one (the stagger's sign), and the polar limit, where
    /// the seam spacing is `2λ` exactly and the stagger `π/S`. Bound: 8 ulp of the larger of the
    /// value and 1 — the chain is a dozen correctly rounded operations deep, and the arcsine of
    /// the seam is evaluated far from its singularity.
    #[test]
    #[allow(clippy::too_many_lines)] // thirty-five reference values: five designs of seven fields
    fn every_solved_quantity_matches_the_derived_street_geometry() {
        struct Case {
            soc: StreetOfCoverage,
            planes: u32,
            per_plane: u32,
            want: [f64; 7],
        }
        let design =
            |altitude: f64, inc_deg: f64, eps_deg: f64, per_plane: Option<u32>| StreetOfCoverage {
                inclination: Angle::from_radians(inc_deg.to_radians()),
                altitude,
                min_elevation: Angle::from_radians(eps_deg.to_radians()),
                earth_radius: RE,
                mu: MU_EARTH,
                per_plane,
            };
        // [θ, λ, ΔΩ_co, ΔΩ_counter, ΔΩ, Ω_max, Δu_inter]
        let cases = [
            Case {
                soc: design(780e3, 86.4, 8.2, None),
                planes: 7,
                per_plane: 10,
                want: [
                    0.347_752_358_714_959_45,
                    0.151_652_375_952_883_33,
                    0.500_413_256_315_782_9,
                    0.276_229_923_140_010_9,
                    0.477_560_455_074_963_73,
                    2.865_362_730_449_782,
                    0.282_068_274_015_367_53,
                ],
            },
            Case {
                soc: design(780e3, 86.4, 8.2, Some(12)),
                planes: 6,
                per_plane: 12,
                want: [
                    0.347_752_358_714_959_45,
                    0.231_577_121_573_610_4,
                    0.580_508_167_681_93,
                    0.446_075_417_634_700_5,
                    0.539_103_447_191_018_5,
                    2.695_517_235_955_093,
                    0.224_294_044_849_767_42,
                ],
            },
            Case {
                soc: design(600e3, 88.0, 5.0, None),
                planes: 8,
                per_plane: 10,
                want: [
                    0.338_943_788_815_770_3,
                    0.129_390_480_824_320_23,
                    0.468_625_080_066_241_24,
                    0.249_236_766_196_201_64,
                    0.413_193_698_199_084_5,
                    2.892_355_887_393_591_6,
                    0.297_498_843_516_534_5,
                ],
            },
            Case {
                soc: design(780e3, 98.0, 8.2, None),
                planes: 7,
                per_plane: 10,
                want: [
                    0.347_752_358_714_959_45,
                    0.151_652_375_952_883_33,
                    0.504_418_894_403_781,
                    0.118_758_181_089_779_73,
                    0.503_805_745_416_668_9,
                    3.022_834_472_500_013_5,
                    0.385_857_448_251_552_41,
                ],
            },
            Case {
                soc: design(1_200e3, 90.0, 10.0, None),
                planes: 6,
                per_plane: 8,
                want: [
                    0.419_190_804_896_360_87,
                    0.150_627_124_044_383_9,
                    0.569_817_928_940_744_8,
                    0.301_254_248_088_767_8,
                    0.568_067_681_100_205_1,
                    2.840_338_405_501_025_5,
                    core::f64::consts::FRAC_PI_8, // the polar stagger π/S
                ],
            },
        ];

        for case in cases {
            let g = case.soc.geometry().expect("feasible near-polar SoC");
            assert_eq!(
                (g.planes, g.per_plane),
                (case.planes, case.per_plane),
                "P, S"
            );
            assert_eq!(case.soc.total(), case.planes * case.per_plane, "P · S");
            let got = [
                g.footprint_half_angle,
                g.street_half_width,
                g.raan_step_co,
                g.raan_step_counter,
                g.raan_step,
                g.raan_span,
                g.inter_plane_phase,
            ];
            for (k, (got, want)) in got.iter().zip(case.want).enumerate() {
                let bound = 8.0 * f64::EPSILON * want.abs().max(1.0);
                assert!(
                    (got - want).abs() <= bound,
                    "field {k} of {:?}: {got:.17e} vs {want:.17e}",
                    case.soc.inclination
                );
            }
        }
    }

    /// (G2) The emitted planes satisfy the touch conditions that define the street tiling.
    ///
    /// Read back from the satellites themselves — each plane's normal is `r × v` of its first
    /// satellite, not the solver's angles — two great circles are at most their dihedral angle
    /// apart (`derivations/soc_street_geometry.mac` §3). Co-rotating neighbours must not exceed the
    /// `θ + λ` their phase-meshed footprints close, and the counter-rotating seam (the last plane
    /// against the first, as point sets) must close at exactly `2λ`: wider leaves a strip covered
    /// by neither street. The seam spacing this module shipped before used the co-rotating form for
    /// that pair and opened the seam to 0.329 rad against `2λ` = 0.303 rad on the Iridium-like
    /// design — a 0.3° coverage hole at ±24° latitude. Bound: 1e-13 rad, the round-off of a
    /// normalised `r × v` taken through `acos` well away from 0 and π.
    #[test]
    fn the_emitted_planes_touch_at_the_derived_distances() {
        let mut forced = iridium_like();
        forced.per_plane = Some(12);
        let shell = StreetOfCoverage {
            inclination: Angle::from_radians(88.0_f64.to_radians()),
            altitude: 600e3,
            min_elevation: Angle::from_radians(5.0_f64.to_radians()),
            ..iridium_like()
        };
        let retro = StreetOfCoverage {
            inclination: Angle::from_radians(98.0_f64.to_radians()),
            ..iridium_like()
        };
        for soc in [iridium_like(), forced, shell, retro] {
            let g = soc.geometry().expect("feasible");
            let mut normals = Vec::new();
            let mut index = 0_u32;
            soc.generate(|e| {
                if index.is_multiple_of(g.per_plane) {
                    let (r, v) = e.to_state();
                    let h = r.cross(v);
                    normals.push(Vec3::new(h.x / h.norm(), h.y / h.norm(), h.z / h.norm()));
                }
                index += 1;
            });
            assert_eq!(normals.len() as u32, g.planes);
            let dihedral = |a: Vec3, b: Vec3| libm::acos(a.dot(b).abs().min(1.0));
            let touch_co = g.footprint_half_angle + g.street_half_width;
            for pair in normals.windows(2) {
                let a = dihedral(pair[0], pair[1]);
                assert!(
                    a <= touch_co + 1e-13,
                    "co-rotating gap {a} above θ+λ = {touch_co}"
                );
            }
            let seam = dihedral(normals[0], normals[normals.len() - 1]);
            let want = 2.0 * g.street_half_width;
            assert!(
                (seam - want).abs() < 1e-13,
                "seam dihedral {seam:.15} against 2λ = {want:.15}"
            );
        }
    }

    /// (G2) The stagger puts each co-rotating neighbour's satellite where the street gap needs it.
    ///
    /// Two planes at the maximum co-rotating spacing are swapped by the half-turn about the
    /// midpoint `E` of their nodes, which carries plane 1 at `u` onto plane 2 at `−u`
    /// (`derivations/soc_street_geometry.mac` §6). The gap that matters is plane 1's midpoint
    /// between two satellites at the foot `b` — the point of plane 1 farthest from plane 2 — and
    /// the stagger is right iff the half-turn of that gap point is a plane-2 satellite. `b` is
    /// found here by bisecting `E·v = 0` on states from the propagator, not from its closed form.
    /// The retrograde 98° case is the one the former arccosine stagger got wrong: it returned
    /// `|b|` and missed by 0.14 rad, opening a 0.5° hole. Bound: 1e-12 rad on the sphere.
    #[test]
    fn the_stagger_meets_the_gap_under_the_half_turn() {
        let retro = StreetOfCoverage {
            inclination: Angle::from_radians(98.0_f64.to_radians()),
            ..iridium_like()
        };
        let mut forced = iridium_like();
        forced.per_plane = Some(12);
        for soc in [iridium_like(), forced, retro] {
            let geom = soc.geometry().expect("feasible");
            let radius = soc.earth_radius + soc.altitude;
            let at = |raan: f64, u: f64| {
                KeplerianElements {
                    semi_major_axis: radius,
                    eccentricity: 0.0,
                    inclination: soc.inclination,
                    raan: Angle::from_radians(raan),
                    arg_periapsis: Angle::from_radians(0.0),
                    true_anomaly: Angle::from_radians(u),
                    mu: MU_EARTH,
                }
                .to_state()
            };
            let half = 0.5 * geom.raan_step_co;
            let node_mid = Vec3::new(libm::cos(half), libm::sin(half), 0.0);
            // E·r(u) on plane 1 rises then falls on (−π/2, π/2); its maximum is where E·v = 0.
            let (mut lo, mut hi) = (-0.5 * PI, 0.5 * PI);
            for _ in 0..200 {
                let mid = f64::midpoint(lo, hi);
                if at(0.0, mid).1.dot(node_mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let gap = at(0.0, lo).0;
            let gap = Vec3::new(gap.x / radius, gap.y / radius, gap.z / radius);
            let twice = 2.0 * node_mid.dot(gap);
            let image = Vec3::new(
                twice * node_mid.x - gap.x,
                twice * node_mid.y - gap.y,
                twice * node_mid.z - gap.z,
            );
            // Satellites run together, so shift time until plane 1's gap (u = π/S) sits at b.
            let shift = lo - PI / f64::from(geom.per_plane);
            let nearest = (0..geom.per_plane)
                .map(|s| {
                    let u = f64::from(s) * TAU / f64::from(geom.per_plane)
                        + geom.inter_plane_phase
                        + shift;
                    let r = at(geom.raan_step_co, u).0;
                    // The chord, not acos of the dot product, which cannot resolve below 1e-8.
                    Vec3::new(
                        image.x - r.x / radius,
                        image.y - r.y / radius,
                        image.z - r.z / radius,
                    )
                    .norm()
                })
                .fold(f64::INFINITY, f64::min);
            assert!(
                nearest < 1e-12,
                "{:?}: the gap's half-turn misses plane 2 by {nearest:.3e} rad",
                soc.inclination
            );
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
        // A near-polar case with a positive stagger, and a wide-footprint 70° one whose stagger is
        // negative (−1.27 rad) — the only way `fmod` returns a negative angle and the wrap does
        // anything. Its λ = 1.09 rad clears the seam's π/2 − i floor.
        let low = StreetOfCoverage {
            inclination: Angle::from_radians(70.0_f64.to_radians()),
            altitude: 8_000e3,
            min_elevation: Angle::from_radians(0.0),
            earth_radius: RE,
            mu: MU_EARTH,
            per_plane: Some(10),
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

    /// The node-spacing guards refuse an arcsine that would be imaginary, and nothing else.
    ///
    /// The co-rotating spacing divides `sin((θ+λ)/2)` by `sin i`; the seam needs
    /// `sin²λ − cos²i ≥ 0`. Each fires alone somewhere: a narrow street (Iridium-like, λ = 0.15)
    /// is refused by the seam at 80° while its co-rotating quotient is 0.25, and admitted at 82°; a
    /// wide one (8 000 km, S = 12, λ = 1.09) passes the seam at 30° while its co-rotating quotient
    /// is 1.78. On the co-rotating boundary itself — where `sin i` is bit-for-bit the numerator, so
    /// the quotient is exactly 1 — the design is still feasible: `ΔΩ_co = 2·asin(1) = π`.
    #[test]
    #[allow(clippy::float_cmp)] // the boundary case is exact by construction, see below
    fn the_node_spacing_guards_refuse_only_an_arcsine_that_would_be_imaginary() {
        let mut narrow = iridium_like();
        narrow.inclination = Angle::from_radians(80.0_f64.to_radians());
        assert!(narrow.geometry().is_none(), "the seam cannot close at 80°");
        assert_eq!(narrow.total(), 0);
        narrow.inclination = Angle::from_radians(82.0_f64.to_radians());
        assert!(narrow.geometry().is_some(), "the seam closes at 82°");

        let mut wide = StreetOfCoverage {
            inclination: Angle::from_radians(30.0_f64.to_radians()),
            altitude: 8_000e3,
            min_elevation: Angle::from_radians(0.0),
            earth_radius: RE,
            mu: MU_EARTH,
            per_plane: Some(12),
        };
        assert!(wide.geometry().is_none(), "co-rotating quotient above 1");

        // Rebuild θ and λ exactly as `geometry` does, then put the inclination on (θ+λ)/2: the
        // quotient is then a number divided by itself, i.e. exactly 1.0 whatever the two round to.
        let theta = libm::acos(RE / (RE + wide.altitude));
        let lambda = libm::acos(libm::cos(theta) / libm::cos(PI / 12.0));
        wide.inclination = Angle::from_radians(f64::midpoint(theta, lambda));
        let g = wide
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

    /// (S1) A sampled coverage check of the whole design; supporting only, the gate is above.
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
        // Sampling can only miss a gap, never invent one, so a covering design leaves no sampled
        // point outside every footprint: the bound is round-off. The −2e-3 this once allowed hid
        // the seam hole that `the_emitted_planes_touch_at_the_derived_distances` now decides.
        assert!(
            worst_margin > -1e-12,
            "coverage gap: worst (cosθ_best − cosθ) = {worst_margin:.2e}"
        );
    }
}
