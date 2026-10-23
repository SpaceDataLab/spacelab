//! The three-stream grid must decide the same coverage as the six-stream one.
//!
//! It cannot be bit-identical: a stored unit zenith rounds once at build time, a derived one
//! rounds differently. What must hold is that every disagreement lies within a few ulp of
//! `g = 0` — a set of measure zero, and orders of magnitude below the along-track error of any
//! real ephemeris. These tests measure the disagreement rate and how close to the boundary it
//! occurs, rather than asserting a tolerance nobody derived.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::naive_bytecount
)]

use spacelab_core::Vec3;
use spacelab_earth::{EQUATORIAL_RADIUS, FLATTENING, POLAR_RADIUS};
use spacelab_sensors::{
    AccessGeometry, CellGrid, Constraint, ElevationMask, EllipsoidGrid, WGS84_GAMMA,
    ellipsoid_grid, kernel,
};

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
    fn direction(&mut self) -> Vec3 {
        let z = 2.0f64.mul_add(self.unit(), -1.0);
        let th = 2.0 * std::f64::consts::PI * self.unit();
        let r = (1.0 - z * z).sqrt();
        Vec3::new(r * th.cos(), r * th.sin(), z)
    }
}

/// A point on the WGS-84 ellipsoid in the given direction, and its exact unit zenith.
fn on_ellipsoid(dir: Vec3) -> (Vec3, Vec3) {
    let a = EQUATORIAL_RADIUS;
    let b = POLAR_RADIUS;
    // scale the direction until it lands on the ellipsoid
    let k = 1.0 / ((dir.x * dir.x + dir.y * dir.y) / (a * a) + dir.z * dir.z / (b * b)).sqrt();
    let p = dir.scale(k);
    let n = Vec3::new(p.x / (a * a), p.y / (a * a), p.z / (b * b));
    (p, n.scale(1.0 / n.norm()))
}

#[test]
fn wgs84_gamma_is_exactly_one_over_one_minus_f_squared() {
    let f = FLATTENING;
    let recomputed = 1.0 / ((1.0 - f) * (1.0 - f));
    assert_eq!(
        f64::to_bits(WGS84_GAMMA),
        f64::to_bits(recomputed),
        "WGS84_GAMMA = {WGS84_GAMMA:?} but 1/(1-f)^2 = {recomputed:?}"
    );

    // and the same number by the other route, a^2/b^2
    let a = EQUATORIAL_RADIUS;
    let b = POLAR_RADIUS;
    let by_axes = (a * a) / (b * b);
    assert!(
        (by_axes - WGS84_GAMMA).abs() < 4.0 * f64::EPSILON,
        "a^2/b^2 = {by_axes:?} disagrees with 1/(1-f)^2 = {WGS84_GAMMA:?}"
    );
}

#[test]
fn derived_zenith_matches_the_exact_ellipsoid_normal() {
    let mut rng = Rng(0x2026_0710);
    for _ in 0..100_000 {
        let (p, exact) = on_ellipsoid(rng.direction());
        let px = [p.x];
        let py = [p.y];
        let pz = [p.z];
        let grid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).expect("rectangular");
        let got = grid.zenith(0);
        // both unit vectors: the angle between them is ~ |a - b|
        let dx = got.x - exact.x;
        let dy = got.y - exact.y;
        let dz = got.z - exact.z;
        let angle = (dx * dx + dy * dy + dz * dz).sqrt();
        // 1 nrad of zenith error moves the footprint edge ~5 mm (derivations/predicate_*.m)
        assert!(
            angle < 1e-14,
            "derived zenith is {angle:.3e} rad off the exact normal"
        );
    }
}

#[test]
fn ellipsoid_kernel_decides_the_same_coverage_as_the_stored_zenith_kernel() {
    let mut rng = Rng(0xE111_0501);
    let n = 8192;
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    let (mut zx, mut zy, mut zz) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let (p, z) = on_ellipsoid(rng.direction());
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
        zx.push(z.x);
        zy.push(z.y);
        zz.push(z.z);
    }
    let stored = CellGrid::new(&px, &py, &pz, &zx, &zy, &zz).expect("rectangular");
    let derived = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).expect("rectangular");

    let mut worst_g: f64 = f64::INFINITY;
    let mut disagreements = 0usize;
    let mut total = 0usize;

    for &mask_deg in &[0.0_f64, 5.0, 30.0] {
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");
        for _ in 0..40 {
            let sat = rng
                .direction()
                .scale(EQUATORIAL_RADIUS + 400e3 + rng.unit() * 35_000e3);
            let mut a = vec![0u8; n];
            let mut b = vec![0u8; n];
            kernel::elevation_covered_into(mask, sat, &stored, &mut a).expect("lengths match");
            ellipsoid_grid::elevation_covered_into(mask, sat, &derived, &mut b)
                .expect("lengths match");

            for i in 0..n {
                total += 1;
                if a[i] != b[i] {
                    disagreements += 1;
                    // how far from the boundary was it?
                    let g = mask.g(&AccessGeometry {
                        satellite: sat,
                        cell: Vec3::new(px[i], py[i], pz[i]),
                        zenith: Vec3::new(zx[i], zy[i], zz[i]),
                    });
                    worst_g = worst_g.min(g.abs());
                    // Any disagreement must sit within a nanometre of g = 0. `g` is in metres and
                    // |d| is up to 4e7 m, so one ulp of the dot products is ~1e-8 m.
                    assert!(
                        g.abs() < 1e-6,
                        "the two grids disagree {} m from the boundary, which is not rounding",
                        g.abs()
                    );
                }
            }
        }
    }
    println!(
        "  {disagreements} disagreements in {total} decisions ({:.2e}); \
         closest was |g| = {:.3e} m from the boundary",
        disagreements as f64 / total as f64,
        if worst_g.is_finite() { worst_g } else { 0.0 }
    );
    // The point is not that there are none — it is that they are all rounding.
    assert!(
        disagreements * 1_000_000 < total,
        "{disagreements} of {total} is too many to be rounding"
    );
}

#[test]
fn ellipsoid_accumulate_matches_its_own_covered_kernel() {
    let mut rng = Rng(0xACC0_2026);
    let n = 1024;
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        let (p, _) = on_ellipsoid(rng.direction());
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
    }
    let grid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).expect("rectangular");

    for &mask_deg in &[-3.0_f64, 0.0, 12.0] {
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");
        let mut counters = vec![0u32; n];
        let mut expected = vec![0u32; n];
        for _ in 0..25 {
            let sat = rng.direction().scale(EQUATORIAL_RADIUS + 600e3);
            ellipsoid_grid::elevation_accumulate_into(mask, sat, &grid, &mut counters)
                .expect("lengths match");
            let mut once = vec![0u8; n];
            ellipsoid_grid::elevation_covered_into(mask, sat, &grid, &mut once)
                .expect("lengths match");
            for i in 0..n {
                expected[i] += u32::from(once[i]);
            }
        }
        assert_eq!(counters, expected, "mask {mask_deg}°");
    }
}
