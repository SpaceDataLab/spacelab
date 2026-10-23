//! The accessor, reach, free-function, empty-grid and error branches the geometry tests skip.
//!
//! `predicate.rs` and `constraint_algebra.rs` cover the constraint mathematics; this file covers the
//! surrounding contract that had no test: the agile-optical accessor and its culling reach, the
//! standalone `elevation` helper, `is_empty` on both grids, and every construction/output error
//! (`RaggedGrid`, `OutputLengthMismatch`, `IncidenceBandOutOfRange`) plus the SAR rate bound.

// The empty-grid tests deliberately compare `len()` against 0 alongside `is_empty()` to pin their
// agreement — that is the contract under test.
// `float_cmp`: the knife-edge link test needs bit-exact zero — that is the property under test.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::len_zero,
    clippy::float_cmp
)]

use core::f64::consts::FRAC_PI_2;

use spacelab_core::Vec3;
use spacelab_sensors::elevation::elevation;
use spacelab_sensors::ellipsoid_grid::{elevation_accumulate_into, elevation_covered_into};
use spacelab_sensors::{
    AccessGeometry, AgileOptical, CellGrid, CommsAntenna, CommsBudget, Cone, Constraint, CullBody,
    ElevationMask, EllipsoidGrid, Or, OrbitBound, Reach, SensorError, SideLookingSAR, WGS84_GAMMA,
    look_side,
};

const MU_EARTH: f64 = 3.986_004_418e14;
const EARTH_ROTATION_RATE: f64 = 7.292_115e-5;
const RE: f64 = 6_371_000.0;
const R_ORBIT: f64 = 7_000_000.0;

fn earth() -> CullBody {
    CullBody {
        min_radius_m: 6_356_752.314,
        deflection_margin_rad: 0.003,
    }
}

#[test]
fn agile_optical_reports_its_cosine_and_a_wider_cone_reaches_further() {
    let narrow = AgileOptical::new(5.0_f64.to_radians()).unwrap();
    let wide = AgileOptical::new(80.0_f64.to_radians()).unwrap();

    // reach() (the Reach trait) borrows, so it may precede the by-value accessor.
    let (rn, rw) = (narrow.reach(R_ORBIT, earth()), wide.reach(R_ORBIT, earth()));
    assert!(
        rn.is_finite() && rn > 0.0,
        "reach must be a real positive angle: {rn}"
    );
    assert!(
        rw >= rn,
        "a wider slew cone must reach at least as far: {rw} < {rn}"
    );

    assert!((narrow.cos_max_off_nadir() - 5.0_f64.to_radians().cos()).abs() < 1e-15);
}

#[test]
fn the_elevation_helper_is_the_angle_above_the_local_horizon() {
    let cell = Vec3::new(0.0, 0.0, RE);
    let zenith = Vec3::new(0.0, 0.0, 1.0);

    // Straight overhead is +90°, a target level with the horizon is 0°, and one below is negative.
    let overhead = elevation(Vec3::new(0.0, 0.0, R_ORBIT), cell, zenith);
    assert!(
        (overhead - FRAC_PI_2).abs() < 1e-12,
        "overhead should be +pi/2, got {overhead}"
    );
    let level = elevation(Vec3::new(RE, 0.0, RE), cell, zenith);
    assert!(
        level.abs() < 1e-9,
        "a horizontal line of sight should be 0, got {level}"
    );
    let below = elevation(Vec3::new(0.0, 0.0, RE - 100.0), cell, zenith);
    assert!(
        below < 0.0,
        "a target below the horizon should be negative, got {below}"
    );
}

#[test]
fn empty_grids_report_empty_and_populated_ones_do_not() {
    let cell_grid = CellGrid::new(&[], &[], &[], &[], &[], &[]).unwrap();
    assert!(cell_grid.is_empty() && cell_grid.len() == 0);

    let empty = EllipsoidGrid::new(&[], &[], &[], WGS84_GAMMA).unwrap();
    assert!(empty.is_empty() && empty.len() == 0);

    let one = EllipsoidGrid::new(&[RE], &[0.0], &[0.0], WGS84_GAMMA).unwrap();
    assert!(!one.is_empty() && one.len() == 1);
}

#[test]
fn ragged_and_mismatched_grids_are_refused_with_the_right_error() {
    // A grid whose coordinate slices differ in length cannot be built.
    assert!(matches!(
        EllipsoidGrid::new(&[RE, RE], &[0.0, 0.0], &[0.0], WGS84_GAMMA),
        Err(SensorError::RaggedGrid { .. }),
    ));

    // Both batch coverage functions reject an output buffer that does not match the grid.
    let grid = EllipsoidGrid::new(&[RE], &[0.0], &[0.0], WGS84_GAMMA).unwrap();
    let sat = Vec3::new(0.0, 0.0, R_ORBIT);
    let mut mask_out = [0u8; 2]; // the grid has one cell
    assert!(matches!(
        elevation_covered_into(ElevationMask::HORIZON, sat, &grid, &mut mask_out),
        Err(SensorError::OutputLengthMismatch {
            cells: 1,
            buffer: 2
        }),
    ));
    let mut counters = [0u32; 3];
    assert!(matches!(
        elevation_accumulate_into(ElevationMask::HORIZON, sat, &grid, &mut counters),
        Err(SensorError::OutputLengthMismatch {
            cells: 1,
            buffer: 3
        }),
    ));
}

#[test]
fn side_looking_sar_validates_its_band_and_bounds_its_rate() {
    // near >= far, or far >= pi/2, is outside the admissible incidence band.
    assert!(matches!(
        SideLookingSAR::new(0.6, 0.4),
        Err(SensorError::IncidenceBandOutOfRange { .. }),
    ));
    assert!(matches!(
        SideLookingSAR::new(0.3, FRAC_PI_2),
        Err(SensorError::IncidenceBandOutOfRange { .. }),
    ));

    // A valid band's rate bound is (1 + cos(near)) * v_rel, since near < far makes cos(near) larger.
    let near = 20.0_f64.to_radians();
    let sar = SideLookingSAR::new(near, 45.0_f64.to_radians()).unwrap();
    let orbit = OrbitBound {
        semi_major_axis: R_ORBIT,
        eccentricity: 0.1,
        mu: MU_EARTH,
    };
    let expected = (1.0 + near.cos()) * orbit.max_relative_speed(EARTH_ROTATION_RATE);
    assert!((sar.max_rate(&orbit, EARTH_ROTATION_RATE) - expected).abs() <= 1e-9 * expected);
}

fn orbit() -> OrbitBound {
    OrbitBound {
        semi_major_axis: R_ORBIT,
        eccentricity: 0.1,
        mu: MU_EARTH,
    }
}

#[test]
fn the_cone_rate_bound_is_the_sum_of_its_two_named_terms() {
    // `Cone::max_rate` is called in exactly one test, `constraint_algebra`'s combinator identity,
    // which puts the mutated value on both sides of the comparison — the canonical "two internal
    // quantities agree with each other" gap. A finite-difference bound would not close it either:
    // the tightest surviving mutant lands only 1.29x below the true bound, and the bound is
    // documented as deliberately loose. Assert the closed form, as `SideLookingSAR` already does.
    let half = 30.0_f64.to_radians();
    let cone = Cone::new(half).unwrap();
    let orbit = orbit();
    let v_rel = orbit.max_relative_speed(EARTH_ROTATION_RATE);
    let expected =
        (1.0 + half.cos()) * v_rel + 2.0 * orbit.apogee_radius() * v_rel / orbit.perigee_radius();
    let got = cone.max_rate(&orbit, EARTH_ROTATION_RATE);
    assert!(
        (got - expected).abs() <= 1e-9 * expected,
        "cone rate bound {got} against {expected}"
    );
    assert!(got > 0.0 && got.is_finite());
}

#[test]
fn the_elevation_rate_bound_is_one_plus_the_masks_sine_times_the_relative_speed() {
    // The finite-difference test in `predicate.rs` bounds this from one side only, and both
    // surviving mutants make the bound *smaller* — 0.913 v_rel and v_rel + 1.087 against the true
    // 1.087 v_rel — which leaves it a valid bound that the measured worst case (0.714 of it) still
    // sits under. No tightening of that ratio can separate them; only the closed form can.
    let mask = ElevationMask::new(5.0_f64.to_radians()).unwrap();
    let orb = orbit();
    let expected = (1.0 + mask.sin_min().abs()) * orb.max_relative_speed(EARTH_ROTATION_RATE);
    let got = mask.max_rate(&orb, EARTH_ROTATION_RATE);
    assert!(
        (got - expected).abs() <= 1e-9 * expected,
        "elevation rate bound {got} against {expected}"
    );
    // A mask below the horizon has the same bound as its mirror above it: the sine enters as |s|.
    let below = ElevationMask::new(-5.0_f64.to_radians()).unwrap();
    assert!(
        (below.max_rate(&orb, EARTH_ROTATION_RATE) - got).abs() <= 1e-9 * expected,
        "the bound is not symmetric in the sign of the mask"
    );
}

#[test]
fn look_side_is_the_cells_signed_distance_from_the_orbit_plane() {
    // The existing fixture puts the satellite on `+x` with velocity along `+z`, which makes the
    // starboard vector exactly `(0, 1, 0)`: two of the three line-of-sight components never reach
    // the answer, the third differences against a zero, and the assertion is only on the sign, so
    // every rescaling of the normalisations survives. For a circular orbit `d̂ × v̂ = −(r̂ × v̂)`, so
    // `look_side` is minus the signed perpendicular distance of the cell from the orbit plane —
    // computable here without touching the implementation's algebra.
    let dir = Vec3::new(0.36, 0.48, 0.8);
    let sat = Vec3::new(dir.x * R_ORBIT, dir.y * R_ORBIT, dir.z * R_ORBIT);
    let raw = dir.cross(Vec3::new(0.2, 0.9, 0.35)); // perpendicular to `dir`, no zero component
    let inv = 1.0 / raw.norm();
    let vel = Vec3::new(
        raw.x * inv * 7500.0,
        raw.y * inv * 7500.0,
        raw.z * inv * 7500.0,
    );

    let n = sat.cross(vel);
    let ninv = 1.0 / n.norm();
    let normal = Vec3::new(n.x * ninv, n.y * ninv, n.z * ninv);

    for cell_dir in [Vec3::new(0.5, 0.3, 0.81), Vec3::new(-0.4, 0.7, 0.59)] {
        let cell = Vec3::new(cell_dir.x * RE, cell_dir.y * RE, cell_dir.z * RE);
        let want = -cell.dot(normal);
        let got = look_side(sat, vel, cell);
        assert!(
            (got - want).abs() <= 1e-6 * want.abs().max(1.0),
            "look_side {got} against the plane distance {want}"
        );
        // The side depends on the direction of travel, not on how fast it is travelled.
        let faster = Vec3::new(vel.x * 3.0, vel.y * 3.0, vel.z * 3.0);
        let scaled = look_side(sat, faster, cell);
        assert!(
            (scaled - got).abs() <= 1e-6 * got.abs().max(1.0),
            "rescaling the velocity moved look_side from {got} to {scaled}"
        );
    }
}

#[test]
fn the_line_of_sight_is_the_satellite_minus_the_cell_in_every_component() {
    // Its only caller sizes a tolerance with `.norm()`, from a fixture whose cell is on the pole —
    // so two components difference against zero and a mutation that *enlarges* the norm only makes
    // that tolerance looser. Use six distinct non-zero coordinates and assert the vector itself.
    let geom = AccessGeometry {
        satellite: Vec3::new(1_100_000.0, -2_300_000.0, 6_700_000.0),
        cell: Vec3::new(-370_000.0, 910_000.0, 6_310_000.0),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    };
    let los = geom.line_of_sight();
    assert!((los.x - 1_470_000.0).abs() < 1e-9, "x = {}", los.x);
    assert!((los.y + 3_210_000.0).abs() < 1e-9, "y = {}", los.y);
    assert!((los.z - 390_000.0).abs() < 1e-9, "z = {}", los.z);
}

#[test]
fn an_or_is_covered_when_exactly_one_side_is() {
    // The combinator test iterates over two geometries, one where both operands are covered and
    // one where neither is; `a || b` and `a && b` agree on both. A geometry that separates them is
    // the whole test.
    let cell = Vec3::new(0.0, 0.0, RE);
    let zenith = Vec3::new(0.0, 0.0, 1.0);
    let (sin30, cos30) = (0.5_f64, 3.0_f64.sqrt() / 2.0);
    let slant = 900_000.0;
    let geom = AccessGeometry {
        satellite: Vec3::new(slant * cos30, 0.0, RE + slant * sin30),
        cell,
        zenith,
    };
    let strict = ElevationMask::new(80.0_f64.to_radians()).unwrap();
    let lenient = ElevationMask::new(5.0_f64.to_radians()).unwrap();
    assert!(!strict.covered(&geom), "the 80 deg mask must reject 30 deg");
    assert!(lenient.covered(&geom), "the 5 deg mask must accept 30 deg");
    assert!(
        Or(strict, lenient).covered(&geom),
        "an Or with one covered side must be covered"
    );
}

#[test]
fn the_elevation_helper_differences_all_three_coordinates() {
    // The existing helper test puts the cell on the pole, where the x and y subtractions are
    // against zero. Take a cell off every axis and check against the closed form.
    let cell = Vec3::new(0.36 * RE, 0.48 * RE, 0.8 * RE);
    let zenith = Vec3::new(0.36, 0.48, 0.8);
    let sat = Vec3::new(1_200_000.0, -800_000.0, 7_100_000.0);
    let d = Vec3::new(sat.x - cell.x, sat.y - cell.y, sat.z - cell.z);
    let want = (d.dot(zenith) / d.norm()).asin();
    let got = elevation(sat, cell, zenith);
    assert!((got - want).abs() < 1e-15, "elevation {got} against {want}");
    assert!(got.abs() < FRAC_PI_2);
}

#[test]
fn a_populated_cell_grid_is_not_empty() {
    // The non-empty half of the emptiness contract was only ever checked on `EllipsoidGrid`.
    let one = CellGrid::new(&[RE], &[0.0], &[0.0], &[1.0], &[0.0], &[0.0]).unwrap();
    assert!(!one.is_empty() && one.len() == 1);
}

#[test]
fn the_agile_reach_is_the_cone_and_horizon_reach_it_delegates_to() {
    // `reach` was only asserted to be finite, positive and monotone in the slew cone — which the
    // constant `1.0` also satisfies. Pin it against the constraint pair it is defined as.
    let half = 40.0_f64.to_radians();
    let agile = AgileOptical::new(half).unwrap();
    let cone = Cone::new(half).unwrap();
    let got = agile.reach(R_ORBIT, earth());
    let want = cone
        .reach(R_ORBIT, earth())
        .min(ElevationMask::HORIZON.reach(R_ORBIT, earth()));
    assert!(
        (got - want).abs() < 1e-12,
        "agile reach {got} against the cone/horizon reach {want}"
    );
}

fn budget() -> CommsBudget {
    CommsBudget {
        eirp_dbw: 20.0,
        g_over_t_db: 10.0,
        frequency_hz: 8.2e9,
        required_cn0_dbhz: 70.0,
        zenith_atmo_db: 0.5,
        reference_altitude_m: 700e3,
        floor_elevation_rad: 5.0_f64.to_radians(),
    }
}

#[test]
fn the_slant_range_closes_the_ground_satellite_triangle() {
    // The break-even test bisects `margin_db`, which is built on `slant_range`, so a wrong range
    // merely means the bisection solves a different equation — and the assertion still holds. The
    // triangle (centre, satellite, ground station) has interior angle 90 deg + eps at the station,
    // so the law of cosines gives d² + 2·Re·d·sin(eps) − (rs² − Re²) = 0 independently of how the
    // quadratic was solved.
    let re = 6_378_137.0_f64;
    let b = budget();
    let rs = re + b.reference_altitude_m;
    for eps_deg in [5.0_f64, 20.0, 45.0, 90.0] {
        let eps = eps_deg.to_radians();
        let d = b.slant_range(eps);
        let residual = d * d + 2.0 * re * d * eps.sin() - (rs * rs - re * re);
        assert!(
            residual.abs() < 1e-6 * rs * rs * f64::EPSILON.sqrt(),
            "at {eps_deg} deg the range {d} leaves a residual of {residual}"
        );
    }
}

#[test]
fn a_comms_antenna_is_the_elevation_mask_at_its_break_even_angle() {
    // Nothing calls `Constraint::{g, covered, max_rate}` on a `CommsAntenna`, so all three could be
    // replaced by a constant. They are defined as delegation; assert the delegation.
    let antenna = CommsAntenna::from_budget(&budget()).unwrap();
    let mask = ElevationMask::new(antenna.break_even_elevation()).unwrap();
    let orb = orbit();
    assert!(
        (antenna.max_rate(&orb, EARTH_ROTATION_RATE) - mask.max_rate(&orb, EARTH_ROTATION_RATE))
            .abs()
            < 1e-15
    );
    let cell = Vec3::new(0.0, 0.0, RE);
    let zenith = Vec3::new(0.0, 0.0, 1.0);
    for el_deg in [-5.0_f64, 2.0, 20.0, 70.0] {
        let (s, c) = (el_deg.to_radians().sin(), el_deg.to_radians().cos());
        let geom = AccessGeometry {
            satellite: Vec3::new(900_000.0 * c, 0.0, RE + 900_000.0 * s),
            cell,
            zenith,
        };
        assert!(
            (antenna.g(&geom) - mask.g(&geom)).abs() < 1e-15,
            "g disagrees at {el_deg} deg"
        );
        assert_eq!(
            antenna.covered(&geom),
            mask.covered(&geom),
            "covered disagrees at {el_deg} deg"
        );
    }
}

#[test]
fn a_link_that_closes_exactly_at_zenith_still_closes() {
    // `if self.margin_db(hi) < 0.0 { return None }` is the one branch a link on the knife edge
    // exercises. Requiring exactly the C/N0 available overhead makes the margin bit-exactly zero,
    // since it is computed as `x - x`.
    let mut b = budget();
    b.required_cn0_dbhz = b.cn0_dbhz(FRAC_PI_2);
    assert_eq!(
        b.margin_db(FRAC_PI_2),
        0.0,
        "the fixture is not on the edge"
    );
    assert!(
        b.break_even_elevation().is_some(),
        "a link that exactly closes overhead was reported as never closing"
    );
}
