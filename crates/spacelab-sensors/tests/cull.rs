//! The cull must never drop a covered cell. This test tries hard to make it.
//!
//! A false negative in the coarse cull is a wrong answer — a covered cell reported as not covered —
//! and no amount of downstream care recovers it. So the guarantee is not "usually keeps covered
//! cells" but "never drops one", and this test is an adversarial search for a counterexample:
//! largest footprints (mask 0, low LEO), the latitude of maximum vertical deflection (45°), and
//! cells densely sampled straight through the footprint edge.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::doc_markdown,
    clippy::float_cmp
)]

use spacelab_core::Vec3;
use spacelab_earth::{EQUATORIAL_RADIUS, FLATTENING, POLAR_RADIUS};
use spacelab_sensors::{
    And, Cone, Constraint, CullBody, ElevationMask, Not, Or, Reach, WGS84_GAMMA, cap_cosine,
    cull_into, ellipsoid_grid::EllipsoidGrid, reachable_band,
};
use spacelab_testkit::nan_max;

const B: f64 = POLAR_RADIUS;

/// A point on the WGS-84 ellipsoid in a given geocentric direction, and its exact geodetic zenith.
fn on_ellipsoid(dir: Vec3) -> (Vec3, Vec3) {
    let a = EQUATORIAL_RADIUS;
    let k = 1.0 / ((dir.x * dir.x + dir.y * dir.y) / (a * a) + dir.z * dir.z / (B * B)).sqrt();
    let p = dir.scale(k);
    let n = Vec3::new(p.x / (a * a), p.y / (a * a), p.z / (B * B));
    (p, n.scale(1.0 / n.norm()))
}

fn unit(v: Vec3) -> Vec3 {
    v.scale(1.0 / v.norm())
}

/// A unit vector at Earth-central angle `ang` from `axis`, rotated by `azimuth` about it.
fn at_angle(axis: Vec3, perp: Vec3, ang: f64, azimuth: f64) -> Vec3 {
    let p2 = axis.cross(perp);
    let r = Vec3::new(
        azimuth.cos() * perp.x + azimuth.sin() * p2.x,
        azimuth.cos() * perp.y + azimuth.sin() * p2.y,
        azimuth.cos() * perp.z + azimuth.sin() * p2.z,
    );
    Vec3::new(
        ang.cos() * axis.x + ang.sin() * r.x,
        ang.cos() * axis.y + ang.sin() * r.y,
        ang.cos() * axis.z + ang.sin() * r.z,
    )
}

#[test]
fn wgs84_cull_body_matches_the_polar_radius_and_bounds_the_deflection() {
    let body = CullBody::WGS84;
    assert!(
        (body.min_radius_m - POLAR_RADIUS).abs() < 1e-6,
        "cull min radius {} != POLAR_RADIUS {POLAR_RADIUS}",
        body.min_radius_m
    );
    // margin must exceed the exact max vertical deflection atan(e²/(2√(1−e²))).
    let e2 = FLATTENING * (2.0 - FLATTENING);
    let delta_max = (e2 / (2.0 * (1.0 - e2).sqrt())).atan();
    assert!(
        body.deflection_margin_rad > delta_max,
        "cull margin {} does not exceed the max deflection {delta_max}",
        body.deflection_margin_rad
    );
    // ...but is not absurdly loose (within 20% of the exact bound).
    assert!(body.deflection_margin_rad < 1.2 * delta_max);
}

/// The heart of it: sweep the adversarial configurations and assert no covered cell escapes the cap.
#[test]
fn cull_never_drops_a_covered_cell_for_the_elevation_mask() {
    let body = CullBody::WGS84;
    let mut covered_total = 0u64;
    let mut escapes = 0u64;
    let mut worst_excess_deg = 0.0_f64;

    // Adversarial: mask at and below the horizon (largest footprints), low LEO, sub-point marched
    // through 45° latitude where the vertical deflection peaks.
    for &mask_deg in &[-2.0_f64, 0.0, 3.0, 10.0] {
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask");
        for alt_km in (300..=2000).step_by(85) {
            for lat_deg in (0..=90).step_by(3) {
                let lat = (lat_deg as f64).to_radians();
                // sub-satellite geocentric direction at this latitude
                let sub = unit(Vec3::new(lat.cos(), 0.0, lat.sin()));
                let r_sat = EQUATORIAL_RADIUS + alt_km as f64 * 1e3;
                let sat = sub.scale(r_sat);

                let lam_cap = mask.reach(r_sat, body);
                let cos_cap = cap_cosine(lam_cap);

                // a vector perpendicular to sub, to sweep azimuth
                let seed = if sub.z.abs() < 0.9 {
                    Vec3::new(0.0, 0.0, 1.0)
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let perp = unit(seed.cross(sub));

                // march a cell from nadir out past the cap edge, several azimuths
                for az_step in 0..8 {
                    let azimuth = az_step as f64 * core::f64::consts::TAU / 8.0;
                    for a_step in 0..400 {
                        let ang = (lam_cap + 0.02) * (a_step as f64 / 399.0);
                        let dir = at_angle(sub, perp, ang, azimuth);
                        let (cellp, zen) = on_ellipsoid(dir);

                        // exact coverage with the true geodetic zenith
                        let los = Vec3::new(sat.x - cellp.x, sat.y - cellp.y, sat.z - cellp.z);
                        let sin_el = los.dot(zen) / los.norm();
                        if sin_el < mask.sin_min() {
                            continue; // not covered; the cull may include or exclude it freely
                        }
                        covered_total += 1;

                        // the cull's decision for this one cell, via the same square-free test
                        let dot = sat.dot(cellp);
                        let inside = dot >= 0.0
                            && dot * dot >= cos_cap * cos_cap * sat.dot(sat) * cellp.dot(cellp);
                        if !inside {
                            escapes += 1;
                            // how far past the cap did this covered cell sit?
                            let geo = (sat.dot(cellp) / (sat.norm() * cellp.norm()))
                                .clamp(-1.0, 1.0)
                                .acos();
                            worst_excess_deg =
                                nan_max(worst_excess_deg, (geo - lam_cap).to_degrees());
                        }
                    }
                }
            }
        }
    }

    println!(
        "elevation cull: checked {covered_total} covered cells across mask×alt×lat, {escapes} escaped"
    );
    assert_eq!(
        escapes, 0,
        "the cull dropped {escapes} covered cells (worst {worst_excess_deg:.4}° past the cap) — false negative"
    );
    assert!(
        covered_total > 1_000_000,
        "the sweep did not exercise enough covered cells"
    );
}

#[test]
fn cull_never_drops_a_covered_cell_for_a_nadir_cone() {
    let body = CullBody::WGS84;
    let mut covered_total = 0u64;
    let mut escapes = 0u64;

    for &half_deg in &[10.0_f64, 30.0, 55.0] {
        // A physical optical sensor is the cone AND the horizon — the bare cone matches far-side
        // cells (it sees through the Earth), which no real instrument does. The cull for the
        // composite is min(cone_cap, horizon_cap).
        let cone = Cone::new(half_deg.to_radians()).expect("half-angle");
        let sensor = And(cone, ElevationMask::HORIZON);
        for alt_km in (300..=2000).step_by(120) {
            for lat_deg in (0..=90).step_by(5) {
                let lat = (lat_deg as f64).to_radians();
                let sub = unit(Vec3::new(lat.cos(), 0.0, lat.sin()));
                let r_sat = EQUATORIAL_RADIUS + alt_km as f64 * 1e3;
                let sat = sub.scale(r_sat);
                let lam_cap = sensor.reach(r_sat, body);
                let cos_cap = cap_cosine(lam_cap);
                let seed = if sub.z.abs() < 0.9 {
                    Vec3::new(0.0, 0.0, 1.0)
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                };
                let perp = unit(seed.cross(sub));

                for az_step in 0..6 {
                    let azimuth = az_step as f64 * core::f64::consts::TAU / 6.0;
                    for a_step in 0..300 {
                        let ang = (lam_cap + 0.02) * (a_step as f64 / 299.0);
                        let dir = at_angle(sub, perp, ang, azimuth);
                        let (cellp, zen) = on_ellipsoid(dir);
                        // physical coverage: inside the cone AND above the horizon
                        let geom = spacelab_sensors::AccessGeometry {
                            satellite: sat,
                            cell: cellp,
                            zenith: zen,
                        };
                        if !sensor.covered(&geom) {
                            continue;
                        }
                        let n2 = sat.dot(sat);
                        covered_total += 1;
                        let dot = sat.dot(cellp);
                        let inside =
                            dot >= 0.0 && dot * dot >= cos_cap * cos_cap * n2 * cellp.dot(cellp);
                        if !inside {
                            escapes += 1;
                        }
                    }
                }
            }
        }
    }
    println!("cone cull: checked {covered_total} covered cells, {escapes} escaped");
    assert_eq!(escapes, 0, "the cone cull dropped {escapes} covered cells");
}

/// AND composes to the smaller cap, and the smaller cap must still contain the intersection.
#[test]
fn and_cull_contains_the_intersection() {
    let body = CullBody::WGS84;
    let mask = ElevationMask::new(5.0_f64.to_radians()).expect("mask");
    let cone = Cone::new(20.0_f64.to_radians()).expect("cone");
    let both = And(cone, mask);

    let r_sat = EQUATORIAL_RADIUS + 700e3;
    // AND's cap is the min of the two, per set logic.
    let expected = cone.reach(r_sat, body).min(mask.reach(r_sat, body));
    assert!((both.reach(r_sat, body) - expected).abs() < 1e-15);

    // and a cell covered by BOTH must be inside the AND cap
    let sub = unit(Vec3::new(1.0, 0.2, 0.3));
    let sat = sub.scale(r_sat);
    let cos_cap = cap_cosine(both.reach(r_sat, body));
    let seed = Vec3::new(0.0, 0.0, 1.0);
    let perp = unit(seed.cross(sub));
    let mut n = 0;
    for a_step in 0..500 {
        let ang = 0.5 * (a_step as f64 / 499.0);
        let dir = at_angle(sub, perp, ang, 0.7);
        let (cellp, zen) = on_ellipsoid(dir);
        let geom = spacelab_sensors::AccessGeometry {
            satellite: sat,
            cell: cellp,
            zenith: zen,
        };
        if both.covered(&geom) {
            let dot = sat.dot(cellp);
            assert!(
                dot >= 0.0 && dot * dot >= cos_cap * cos_cap * sat.dot(sat) * cellp.dot(cellp),
                "AND-covered cell escaped the AND cap"
            );
            n += 1;
        }
    }
    assert!(n > 0, "no AND-covered cells were exercised");
}

/// The cull kernel over an `EllipsoidGrid` agrees with the per-cell test and eliminates most cells.
#[test]
fn cull_kernel_matches_per_cell_and_reduces_the_set() {
    // a small global grid
    let n = 40_000usize;
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let th = golden * i as f64;
        let (p, _) = on_ellipsoid(Vec3::new(r * th.cos(), r * th.sin(), z));
        px.push(p.x);
        py.push(p.y);
        pz.push(p.z);
    }
    let grid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).expect("rectangular");
    let mask = ElevationMask::new(5.0_f64.to_radians()).expect("mask");
    let r_sat = EQUATORIAL_RADIUS + 700e3;
    // Off every axis, with three distinct non-zero components. On the pole the kernel's hoisted
    // `rx*rx + ry*ry + rz*rz` collapses to `rz*rz` and its dot product to `rz*cz`, so a whole
    // column of arithmetic reduces to identities and nine mutants of it become unobservable.
    let dir = Vec3::new(0.36, 0.48, 0.8); // unit: 0.36² + 0.48² + 0.8² = 1
    let sat = Vec3::new(dir.x * r_sat, dir.y * r_sat, dir.z * r_sat);
    let cos_cap = cap_cosine(mask.reach(r_sat, CullBody::WGS84));

    let mut out = vec![0u8; n];
    cull_into(sat, cos_cap, &grid, &mut out).expect("lengths");

    let mut kept = 0usize;
    for i in 0..n {
        let cell = Vec3::new(px[i], py[i], pz[i]);
        let dot = sat.dot(cell);
        let want = dot >= 0.0 && dot * dot >= cos_cap * cos_cap * sat.dot(sat) * cell.dot(cell);
        assert_eq!(
            out[i] == 1,
            want,
            "kernel disagreed with the per-cell test at {i}"
        );
        kept += usize::from(out[i]);
    }
    // a 700 km / 5° footprint is a small cap; the cull must throw away the vast majority.
    let fraction = kept as f64 / n as f64;
    println!(
        "cull kept {kept}/{n} = {:.1}% of the sphere for a 700 km, 5° mask",
        fraction * 100.0
    );
    assert!(
        fraction < 0.15,
        "cull kept {fraction:.3} of cells — far too many for this footprint"
    );
    assert!(kept > 0, "cull kept nothing");
}

/// OR grows the cap to the larger operand; NOT gives up (no cull).
#[test]
fn or_and_not_compose_caps_correctly() {
    let body = CullBody::WGS84;
    let r_sat = EQUATORIAL_RADIUS + 700e3;
    let narrow = Cone::new(10.0_f64.to_radians()).expect("cone");
    let wide = ElevationMask::new(5.0_f64.to_radians()).expect("mask");

    // OR reaches as far as its widest operand.
    let either = Or(narrow, wide);
    let expected = narrow.reach(r_sat, body).max(wide.reach(r_sat, body));
    assert!((either.reach(r_sat, body) - expected).abs() < 1e-15);
    assert!(either.reach(r_sat, body) >= narrow.reach(r_sat, body));

    // NOT of anything culls nothing: the complement of a cap is not a cap.
    let complement = Not(wide);
    assert_eq!(complement.reach(r_sat, body), core::f64::consts::PI);

    // and cos(π) < 0, so the kernel keeps every cell for a NOT.
    assert!(cap_cosine(complement.reach(r_sat, body)) < 0.0);
}

/// The reachable latitude band widens with inclination and footprint, never excludes the sub-satellite
/// latitude, and saturates at the pole so a polar orbit culls nothing.
#[test]
fn reachable_band_bounds_the_covered_latitudes() {
    use core::f64::consts::FRAC_PI_2;
    let r_sat = 6_378_137.0 + 550e3;
    let mask = ElevationMask::new(5.0_f64.to_radians()).expect("mask in range");
    let lam = mask.reach(r_sat, CullBody::WGS84); // footprint cap half-angle

    // A low-inclination orbit reaches only a band well short of the pole (a real cull).
    let inc_lo = 25.0_f64.to_radians();
    let band_lo = reachable_band(inc_lo, r_sat, &mask, CullBody::WGS84);
    assert!(
        band_lo < FRAC_PI_2,
        "low-inc band should cull, got {band_lo}"
    );
    assert!(
        band_lo >= inc_lo,
        "band must contain the sub-satellite latitude"
    );
    assert!((band_lo - (inc_lo + lam)).abs() < 1e-12, "band = inc + cap");

    // Monotone in inclination, and a near-polar orbit saturates at the pole (no cull).
    let band_mid = reachable_band(53.0_f64.to_radians(), r_sat, &mask, CullBody::WGS84);
    assert!(band_mid > band_lo, "band grows with inclination");
    let band_polar = reachable_band(98.0_f64.to_radians(), r_sat, &mask, CullBody::WGS84);
    assert_eq!(
        band_polar, FRAC_PI_2,
        "near-polar orbit reaches the pole → cull nothing"
    );

    // A retrograde orbit reaches the same latitudes as its prograde mirror: `min(i, π − i)`. Every
    // sample above is prograde, where `π − i`, `π + i` and `π / i` all exceed `i` and the `min`
    // returns `i` whichever way the expression is written. At 120° they separate — and the band is
    // still short of the pole, so the saturation does not hide the difference either.
    let inc_retro = 120.0_f64.to_radians();
    let mirror = core::f64::consts::PI - inc_retro; // 60 degrees
    let band_retro = reachable_band(inc_retro, r_sat, &mask, CullBody::WGS84);
    assert!(
        (band_retro - (mirror + lam)).abs() < 1e-12,
        "retrograde band {band_retro} is not the mirrored inclination plus the cap"
    );
    assert!(
        band_retro < FRAC_PI_2,
        "the retrograde case must not saturate"
    );
}

/// `reach` is only ever checked *relatively* — `And` takes the smaller, `Or` the larger, a wide cone
/// reaches further than a narrow one. Nothing pins the number itself, so cargo-mutants can replace
/// the whole of `<Cone as Reach>::reach` with the constant `1.0` and every one of those relations
/// still holds. Twenty-six survivors sit in that one function.
///
/// The oracle is the sine rule on the triangle (Earth centre, satellite, footprint edge), which is
/// where the implementation's algebra comes from but is not the algebra it uses:
///
/// * cone-limited, `l·sin a < 1` — the angle at the surface point is obtuse, so
///   `λ = asin(l·sin a) − a`;
/// * horizon-limited, `l·sin a ≥ 1` — the ray grazes the limb and the cone stops mattering, so
///   `λ = acos(1/l)`, the plain horizon half-angle.
///
/// The implementation instead builds `ρ = sin λ` directly and takes `asin`; the two agree only if
/// its algebra is right. `ElevationMask` gets the same treatment from the same triangle with the
/// angle at the surface point fixed at `90° + ε`: `λ = π/2 − ε − asin(cos ε / l)`.
#[test]
fn the_footprint_half_angle_is_the_one_the_sine_rule_gives() {
    let body = CullBody::WGS84;
    let margin = body.deflection_margin_rad;

    // Cone: half-angles from narrow to past the horizon limit, at radii from low LEO to GEO.
    for half_angle_deg in [1.0_f64, 5.0, 20.0, 45.0, 60.0, 80.0] {
        let a = half_angle_deg.to_radians();
        let cone = Cone::new(a).expect("a valid half-angle");
        for altitude_km in [400.0, 800.0, 2_000.0, 20_000.0, 35_786.0] {
            let r_sat = body.min_radius_m + altitude_km * 1000.0;
            let l = r_sat / body.min_radius_m;
            let want = if l * a.sin() >= 1.0 {
                (1.0 / l).acos() // horizon-limited: the cone is wider than the visible cap
            } else {
                (l * a.sin()).asin() - a
            } + margin;
            let got = cone.reach(r_sat, body);
            assert!(
                (got - want).abs() < 1e-12,
                "cone {half_angle_deg}° at {altitude_km} km: reach {got:.15} rad, sine rule gives {want:.15}"
            );
        }
    }

    // The two branches must meet: at exactly l = 1/sin a the cone edge grazes the limb.
    for half_angle_deg in [10.0_f64, 30.0, 55.0] {
        let a = half_angle_deg.to_radians();
        let cone = Cone::new(a).expect("a valid half-angle");
        let r_sat = body.min_radius_m / a.sin();
        let got = cone.reach(r_sat, body);
        let want = core::f64::consts::FRAC_PI_2 - a + margin; // acos(1/l) = acos(sin a) at l = 1/sin a
        assert!(
            (got - want).abs() < 1e-9,
            "cone {half_angle_deg}° at the grazing radius: {got:.15} vs {want:.15}"
        );
    }

    // Elevation mask, same triangle with the angle at the surface point pinned to 90° + ε.
    for elevation_deg in [0.0_f64, 5.0, 10.0, 30.0, 70.0] {
        let eps = elevation_deg.to_radians();
        let mask = ElevationMask::new(eps).expect("a valid elevation");
        for altitude_km in [400.0, 1_200.0, 35_786.0] {
            let r_sat = body.min_radius_m + altitude_km * 1000.0;
            let l = r_sat / body.min_radius_m;
            let want = core::f64::consts::FRAC_PI_2 - eps - (eps.cos() / l).asin() + margin;
            let got = mask.reach(r_sat, body);
            assert!(
                (got - want).abs() < 1e-12,
                "mask {elevation_deg}° at {altitude_km} km: reach {got:.15} rad, sine rule gives {want:.15}"
            );
        }
    }
}
