//! The constraints, checked against the geometry two computer algebra systems proved.
//!
//! `derivations/footprint.mac` (Maxima) and `derivations/footprint.input` (FriCAS) independently
//! reduce `rho_ours^2 - rho_savi^2` to zero. These tests assert that the code implements the
//! geometry that was proved, not merely a plausible one.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::naive_bytecount
)]

use spacelab_core::Vec3;
use spacelab_sensors::{
    And, CellGrid, Cone, Constraint, ElevationMask, OrbitBound, SensorError,
    kernel::{
        cone_covered_into, cone_covered_one, elevation_accumulate_into, elevation_covered_into,
        elevation_covered_one,
    },
};
use spacelab_testkit::nan_max;

/// Radius of the reference sphere. The derivation is spherical; so is this test.
const R: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;
const EARTH_ROTATION_RATE: f64 = 7.292_115_146_706_979e-5;

/// `splitmix64`, so the test is deterministic without a dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// A point uniform on the unit sphere, by the Marsaglia z/theta method.
    fn direction(&mut self) -> Vec3 {
        let z = 2.0f64.mul_add(self.unit(), -1.0);
        let theta = 2.0 * std::f64::consts::PI * self.unit();
        let r = (1.0 - z * z).sqrt();
        Vec3::new(r * theta.cos(), r * theta.sin(), z)
    }
}

/// A ground cell at the north pole of the sphere, so its zenith is `+z`.
fn polar_cell() -> (Vec3, Vec3) {
    (Vec3::new(0.0, 0.0, R), Vec3::new(0.0, 0.0, 1.0))
}

/// Satellite at radius `L*R` and Earth-central angle `lam` from the polar cell.
fn satellite_at(l: f64, lam: f64) -> Vec3 {
    Vec3::new(L_R(l) * lam.sin(), 0.0, L_R(l) * lam.cos())
}

#[allow(non_snake_case)]
fn L_R(l: f64) -> f64 {
    l * R
}

// ---------------------------------------------------------------------------------------------
// 1. `covered` is exactly `g >= 0`, which is the trait's stated contract.
// ---------------------------------------------------------------------------------------------

#[test]
fn covered_agrees_with_g_sign_for_elevation_over_random_geometry() {
    let mut rng = Rng(0x5EED_1234);
    for &mask_deg in &[-10.0_f64, 0.0, 5.0, 30.0, 80.0] {
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");
        for _ in 0..200_000 {
            let u = rng.direction();
            let cell = u.scale(R);
            let alt = 300e3 + rng.unit() * 35_500e3;
            let sat = rng.direction().scale(R + alt);
            let geom = spacelab_sensors::AccessGeometry {
                satellite: sat,
                cell,
                zenith: u,
            };
            assert_eq!(
                mask.covered(&geom),
                mask.g(&geom) >= 0.0,
                "mask {mask_deg}°, g = {}",
                mask.g(&geom)
            );
        }
    }
}

#[test]
fn covered_agrees_with_g_sign_for_cone_over_random_geometry() {
    let mut rng = Rng(0xC0FF_EE01);
    for &half_deg in &[1.0_f64, 20.0, 60.0, 90.0] {
        let cone = Cone::new(half_deg.to_radians()).expect("half-angle in range");
        for _ in 0..200_000 {
            let u = rng.direction();
            let alt = 300e3 + rng.unit() * 35_500e3;
            let geom = spacelab_sensors::AccessGeometry {
                satellite: rng.direction().scale(R + alt),
                cell: u.scale(R),
                zenith: u,
            };
            assert_eq!(
                cone.covered(&geom),
                cone.g(&geom) >= 0.0,
                "half {half_deg}°"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 2. The footprint edge is where the CAS said it is.
// ---------------------------------------------------------------------------------------------

/// `cos(lam*) = (s*u + 1 - s^2)/L`, `u = sqrt(L^2 - 1 + s^2)` — the root Maxima and FriCAS both
/// confirm equals SaVi's published mask-elevation footprint radius.
fn boundary_lambda_elevation(l: f64, eps_min: f64) -> f64 {
    let s = eps_min.sin();
    let u = (l * l - 1.0 + s * s).sqrt();
    ((s * u + 1.0 - s * s) / l).acos()
}

#[test]
fn g_vanishes_on_the_derived_footprint_edge() {
    let (cell, zenith) = polar_cell();
    for &alt_km in &[500.0_f64, 800.0, 2000.0, 20_200.0, 35_786.0] {
        for &mask_deg in &[0.0_f64, 5.0, 10.0, 25.0] {
            let l = (R + alt_km * 1e3) / R;
            let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");
            let lam = boundary_lambda_elevation(l, mask_deg.to_radians());
            let geom = spacelab_sensors::AccessGeometry {
                satellite: satellite_at(l, lam),
                cell,
                zenith,
            };
            let g = mask.g(&geom);

            // `g` has the dimension of length. Its inputs are dots and a norm of vectors whose
            // magnitude is |d| <= (L+1)R. Each carries a few ulp, so |g| should not exceed a
            // small multiple of eps * |d|. At GEO |d| ~ 4.2e7 m and eps*|d| ~ 9e-9 m.
            let d_mag = geom.line_of_sight().norm();
            let tol = 64.0 * f64::EPSILON * d_mag;
            assert!(
                g.abs() < tol,
                "alt {alt_km} km, mask {mask_deg}°: |g| = {g:.3e} m exceeds {tol:.3e} m"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 3. `Cone ∧ Horizon` reproduces SaVi's horizon-limited branch — without a branch.
// ---------------------------------------------------------------------------------------------

/// SaVi's two closed forms for the footprint radius on the unit sphere, `rho = sin(lam_max)`.
fn savi_cone_rho(l: f64, alpha: f64) -> f64 {
    let (s, c) = (alpha.sin(), alpha.cos());
    if l >= 1.0 / s {
        // horizon-limited: the cone is wider than the limb
        (l * l - 1.0).sqrt() / l
    } else {
        s * (l * c - (1.0 - l * l * s * s).sqrt())
    }
}

#[test]
fn cone_and_horizon_boundary_matches_savi_in_both_branches() {
    let (cell, zenith) = polar_cell();
    let horizon = ElevationMask::HORIZON;

    // (L, alpha_deg, which branch SaVi takes)
    let cases = [
        (1.05_f64, 30.0_f64, "cone-limited"), // 1/sin a = 2.0  > L
        (1.08, 20.0, "cone-limited"),         // 1/sin a = 2.92 > L
        (2.0, 40.0, "horizon-limited"),       // 1/sin a = 1.56 < L
        (6.61, 60.0, "horizon-limited"),      // GEO, wide cone
    ];

    for (l, alpha_deg, branch) in cases {
        let alpha = alpha_deg.to_radians();
        let cone = Cone::new(alpha).expect("half-angle in range");
        let combined = And(cone, horizon);

        let lam_edge = savi_cone_rho(l, alpha).asin();

        // Just inside the edge: covered. Just outside: not. The margin is chosen so the angular
        // step moves the satellite far more than the rounding of `g` (metres, not nanometres).
        let margin = 1e-7; // rad; ~0.6 m of sub-satellite motion, ~1e5 x the g rounding
        for (dl, want) in [(-margin, true), (margin, false)] {
            let geom = spacelab_sensors::AccessGeometry {
                satellite: satellite_at(l, lam_edge + dl),
                cell,
                zenith,
            };
            assert_eq!(
                combined.covered(&geom),
                want,
                "L={l}, alpha={alpha_deg}° ({branch}), lam={} rad, dl={dl}: g={}",
                lam_edge + dl,
                combined.g(&geom)
            );
        }
    }
}

#[test]
fn cone_alone_sees_through_the_earth_where_the_composite_does_not() {
    // A 60° cone at GEO contains the antipode direction geometrically, but the antipodal cell is
    // behind the limb. `Cone` says yes; `Cone ∧ Horizon` says no. This is the failure mode SaVi
    // avoids with a branch, and the composition avoids with `min`.
    let (cell, zenith) = polar_cell();
    let l = 6.61;
    let cone = Cone::new(60.0_f64.to_radians()).expect("half-angle in range");
    let lam_beyond_limb = (1.0_f64 / l).acos() + 0.05;
    let geom = spacelab_sensors::AccessGeometry {
        satellite: satellite_at(l, lam_beyond_limb),
        cell,
        zenith,
    };
    assert!(cone.covered(&geom), "the bare cone contains the cell");
    assert!(
        !And(cone, ElevationMask::HORIZON).covered(&geom),
        "the cell is behind the limb, so the composite must reject it"
    );
}

// ---------------------------------------------------------------------------------------------
// 4. The batch kernels agree with the scalar predicate, bit for bit.
// ---------------------------------------------------------------------------------------------

/// Six owned coordinate streams: `px, py, pz, zx, zy, zz`.
type OwnedColumns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
fn random_grid(rng: &mut Rng, n: usize) -> OwnedColumns {
    let mut px = Vec::with_capacity(n);
    let mut py = Vec::with_capacity(n);
    let mut pz = Vec::with_capacity(n);
    let mut zx = Vec::with_capacity(n);
    let mut zy = Vec::with_capacity(n);
    let mut zz = Vec::with_capacity(n);
    for _ in 0..n {
        let u = rng.direction();
        px.push(u.x * R);
        py.push(u.y * R);
        pz.push(u.z * R);
        zx.push(u.x);
        zy.push(u.y);
        zz.push(u.z);
    }
    (px, py, pz, zx, zy, zz)
}

#[test]
fn batch_elevation_kernel_matches_the_scalar_predicate_exactly() {
    let mut rng = Rng(0xBEEF_0007);
    let n = 4099; // prime, so the vectorised body and its scalar tail both run
    let (px, py, pz, zx, zy, zz) = random_grid(&mut rng, n);
    let grid = CellGrid::new(&px, &py, &pz, &zx, &zy, &zz).expect("rectangular");

    for &mask_deg in &[-5.0_f64, 0.0, 10.0] {
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");
        for _ in 0..64 {
            let sat = rng.direction().scale(R + 300e3 + rng.unit() * 35_500e3);
            let mut out = vec![0u8; n];
            elevation_covered_into(mask, sat, &grid, &mut out).expect("lengths match");
            for i in 0..n {
                let want = elevation_covered_one(
                    mask,
                    sat,
                    Vec3::new(px[i], py[i], pz[i]),
                    Vec3::new(zx[i], zy[i], zz[i]),
                );
                assert_eq!(out[i] == 1, want, "cell {i}, mask {mask_deg}°");
            }
        }
    }
}

#[test]
fn batch_cone_kernel_matches_the_scalar_predicate_exactly() {
    let mut rng = Rng(0xFEED_0009);
    let n = 2053;
    let (px, py, pz, zx, zy, zz) = random_grid(&mut rng, n);
    let grid = CellGrid::new(&px, &py, &pz, &zx, &zy, &zz).expect("rectangular");
    let cone = Cone::new(25.0_f64.to_radians()).expect("half-angle in range");

    for _ in 0..64 {
        let sat = rng.direction().scale(R + 300e3 + rng.unit() * 35_500e3);
        let mut out = vec![0u8; n];
        cone_covered_into(cone, sat, &grid, &mut out).expect("lengths match");
        for i in 0..n {
            let want = cone_covered_one(cone, sat, Vec3::new(px[i], py[i], pz[i]));
            assert_eq!(out[i] == 1, want, "cell {i}");
        }
    }
}

#[test]
fn accumulate_counts_exactly_the_epochs_a_cell_was_covered() {
    let mut rng = Rng(0x1234_ABCD);
    let n = 512;
    let (px, py, pz, zx, zy, zz) = random_grid(&mut rng, n);
    let grid = CellGrid::new(&px, &py, &pz, &zx, &zy, &zz).expect("rectangular");

    // A negative mask takes the accumulator's other branch — the one that keeps a cell whose
    // horizon test *fails*, because the satellite may sit below the local horizon and still be in
    // view. At a single positive mask that branch is dead code, which is why the `EllipsoidGrid`
    // twin (which sweeps -3, 0 and 12 degrees) caught mutants of it and this one did not.
    for mask_deg in [-3.0_f64, 0.0, 5.0, 12.0] {
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");

        let epochs = 37;
        let mut counters = vec![0u32; n];
        let mut expected = vec![0u32; n];
        for _ in 0..epochs {
            let sat = rng.direction().scale(R + 700e3);
            elevation_accumulate_into(mask, sat, &grid, &mut counters).expect("lengths match");
            let mut once = vec![0u8; n];
            elevation_covered_into(mask, sat, &grid, &mut once).expect("lengths match");
            for i in 0..n {
                expected[i] += u32::from(once[i]);
            }
        }
        assert_eq!(counters, expected, "at a {mask_deg} degree mask");
        assert!(
            counters.iter().any(|&c| c > 0),
            "some cell must be seen at a {mask_deg} degree mask"
        );
        assert!(counters.iter().all(|&c| c <= epochs), "never over-counted");
    }
}

// ---------------------------------------------------------------------------------------------
// 5. `max_rate` really bounds `|dg/dt|`. A bound that is not a bound loses passes.
// ---------------------------------------------------------------------------------------------

#[test]
fn max_rate_bounds_the_finite_difference_of_g_along_a_real_orbit() {
    // Circular 700 km orbit, sampled in a frame rotating with the Earth. The satellite state is
    // built directly rather than propagated: this test is about the bound, not about SGP4.
    let a = R + 700e3;
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };
    let mask = ElevationMask::new(5.0_f64.to_radians()).expect("mask in range");
    let bound = mask.max_rate(&orbit, EARTH_ROTATION_RATE);

    let n_mean = (MU / (a * a * a)).sqrt();
    let inc = 51.6_f64.to_radians();
    let (cell, _) = polar_cell();
    let cell = Vec3::new(cell.x, cell.y, cell.z * 0.5); // move it off the pole, still on axis-ish
    let cell = cell.scale(R / cell.norm());

    let dt = 0.05_f64;
    let mut worst: f64 = 0.0;
    for k in 0..200_000 {
        let t = f64::from(k) * 0.5;
        let sample = |t: f64| {
            let u = n_mean * t;
            // inertial circular orbit at inclination `inc`
            let p = Vec3::new(
                a * u.cos(),
                a * u.sin() * inc.cos(),
                a * u.sin() * inc.sin(),
            );
            // rotate into the body-fixed frame
            let th = EARTH_ROTATION_RATE * t;
            let sat = Vec3::new(
                p.x * th.cos() + p.y * th.sin(),
                -p.x * th.sin() + p.y * th.cos(),
                p.z,
            );
            mask.g(&spacelab_sensors::AccessGeometry {
                satellite: sat,
                cell,
                zenith: cell.scale(1.0 / R),
            })
        };
        let rate = (sample(t + dt) - sample(t - dt)) / (2.0 * dt);
        worst = nan_max(worst, rate.abs());
    }
    assert!(
        worst < bound,
        "measured |dg/dt| = {worst:.1} m/s exceeds the bound {bound:.1} m/s"
    );
    // And it must not be absurdly loose, or the coarse step wastes evaluations. Measured here:
    // worst/bound = 0.713996. Note this direction cannot pin the bound's *form* — a mutation that
    // makes the bound smaller leaves it a valid bound and only raises this ratio, which is why
    // `sensor_api.rs` asserts the closed form as well.
    assert!(
        worst > 0.05 * bound,
        "bound {bound:.1} m/s is more than 20x the observed maximum {worst:.1} m/s"
    );
}

// ---------------------------------------------------------------------------------------------
// 6. Illegal states are rejected at the boundary, not tolerated in the loop.
// ---------------------------------------------------------------------------------------------

#[test]
fn out_of_range_sensors_are_rejected() {
    assert!(matches!(
        ElevationMask::new(std::f64::consts::FRAC_PI_2),
        Err(SensorError::ElevationOutOfRange { .. })
    ));
    assert!(matches!(
        ElevationMask::new(f64::NAN),
        Err(SensorError::ElevationOutOfRange { .. })
    ));
    assert!(matches!(
        Cone::new(0.0),
        Err(SensorError::HalfAngleOutOfRange { .. })
    ));
    assert!(matches!(
        Cone::new(2.0),
        Err(SensorError::HalfAngleOutOfRange { .. })
    ));
    assert!(
        Cone::new(std::f64::consts::FRAC_PI_2).is_ok(),
        "pi/2 allowed"
    );
}

#[test]
fn ragged_grids_and_mismatched_buffers_are_rejected() {
    let three = [0.0, 1.0, 2.0];
    let two = [0.0, 1.0];
    assert!(matches!(
        CellGrid::new(&three, &two, &three, &three, &three, &three),
        Err(SensorError::RaggedGrid {
            shortest: 2,
            longest: 3
        })
    ));

    let grid = CellGrid::new(&three, &three, &three, &three, &three, &three).expect("rectangular");
    let mut out = [0u8; 2];
    assert!(matches!(
        elevation_covered_into(
            ElevationMask::HORIZON,
            Vec3::new(0.0, 0.0, 1e7),
            &grid,
            &mut out
        ),
        Err(SensorError::OutputLengthMismatch {
            cells: 3,
            buffer: 2
        })
    ));
}

// ---------------------------------------------------------------------------------------------
// 7. The boundary itself. Random geometry never lands on `g == 0`, so it is constructed.
// ---------------------------------------------------------------------------------------------

#[test]
fn a_cell_exactly_on_the_horizon_is_covered() {
    // Satellite in the cell's horizon plane: d = (X, 0, 0), zenith = (0,0,1), so t == 0.0 exactly
    // and g == 0.0 exactly. The contract says `covered == (g >= 0)`, so this must be `true`.
    // A `>` in place of `>=` inside `covered_from` passes every random test and fails this one.
    let (cell, zenith) = polar_cell();
    let geom = spacelab_sensors::AccessGeometry {
        satellite: Vec3::new(1_000_000.0, 0.0, R),
        cell,
        zenith,
    };
    let g = ElevationMask::HORIZON.g(&geom);
    assert_eq!(
        g, 0.0,
        "the construction must put the satellite exactly on the horizon"
    );
    assert!(
        ElevationMask::HORIZON.covered(&geom),
        "g == 0.0 is covered: the boundary belongs to the set"
    );
}

#[test]
fn a_cell_behind_a_sub_surface_satellite_is_not_in_the_cone() {
    // `AccessGeometry` does not forbid a decayed satellite. With |r| < |c| along the same ray the
    // cell sits *behind* the sensor, and `utilde = n2 - c·r` goes negative. Without the
    // `utilde >= 0` guard the squared comparison would report it covered: the cone would see
    // through its own back.
    let cell = Vec3::new(0.0, 0.0, R);
    let sat = Vec3::new(0.0, 0.0, R / 2.0);
    let geom = spacelab_sensors::AccessGeometry {
        satellite: sat,
        cell,
        zenith: Vec3::new(0.0, 0.0, 1.0),
    };
    let cone = Cone::new(80.0_f64.to_radians()).expect("half-angle in range");

    let n2 = sat.dot(sat);
    assert!(
        n2 - cell.dot(sat) < 0.0,
        "the construction must drive utilde negative"
    );
    assert!(
        !cone.covered(&geom),
        "a cell behind the sensor is never in the cone"
    );
    assert!(cone.g(&geom) < 0.0, "and its g must agree");
}
