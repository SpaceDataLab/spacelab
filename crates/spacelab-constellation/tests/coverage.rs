//! End-to-end: a generated constellation drives the coverage engine — *generate → propagate →
//! revisit*. Each satellite's elements are propagated into the Earth-fixed frame and fed to
//! `access_windows` with an elevation mask; the per-satellite windows merge into one revisit
//! statistic. The check is monotone and physical: adding satellites cannot lengthen the worst-case
//! gap, and a substantially denser Walker must shorten it.

#![allow(clippy::expect_used)]

use spacelab_constellation::{FlowerConstellation, Walker, WalkerPattern};
use spacelab_core::{Angle, Vec3};
use spacelab_coverage::{Window, access_windows, combined_metrics};
use spacelab_elements::KeplerianElements;
use spacelab_sensors::{ElevationMask, OrbitBound};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const OMEGA_E: f64 = 7.292_115_146_5e-5; // Earth sidereal rate, rad/s
const DAY: f64 = 86_400.0;

/// Earth-fixed (ECEF) satellite position at time `t`: propagate in inertial coordinates, then undo
/// the frame's rotation `ω⊕·t` about z.
fn ecef_at(el: &KeplerianElements, t: f64) -> Vec3 {
    let r = el.propagate(t).to_state().0;
    let (s, c) = (OMEGA_E * t).sin_cos();
    Vec3::new(r.x * c + r.y * s, -r.x * s + r.y * c, r.z)
}

fn walker(
    pattern: WalkerPattern,
    planes: u32,
    per_plane: u32,
    inc_deg: f64,
) -> Vec<KeplerianElements> {
    let w = Walker {
        pattern,
        inclination: Angle::from_radians(inc_deg.to_radians()),
        planes,
        per_plane,
        phasing: 1,
        semi_major_axis: RE + 1_200e3,
        mu: MU,
    };
    let mut v = Vec::new();
    w.generate(|e| v.push(e));
    v
}

/// Worst-case revisit gap (seconds) at a spherical-Earth ground site over `[0, span]`, above a 5°
/// mask, for the whole constellation.
fn revisit_gap(
    sats: &[KeplerianElements],
    lat_deg: f64,
    span: f64,
) -> spacelab_coverage::CoverageMetrics {
    let lat = lat_deg.to_radians();
    let cell = Vec3::new(RE * lat.cos(), 0.0, RE * lat.sin());
    let n = cell.norm();
    let zenith = Vec3::new(cell.x / n, cell.y / n, cell.z / n); // spherical Earth: radial = normal
    let sensor = ElevationMask::new(5.0_f64.to_radians()).expect("valid mask");

    let mut windows: Vec<Window> = Vec::new();
    for s in sats {
        let orbit = OrbitBound {
            semi_major_axis: s.semi_major_axis,
            eccentricity: s.eccentricity,
            mu: MU,
        };
        access_windows(
            &sensor,
            &orbit,
            OMEGA_E,
            cell,
            zenith,
            |t| ecef_at(s, t),
            0.0,
            span,
            1e-3,
            |w| windows.push(w),
        );
    }
    combined_metrics(&mut windows, 0.0, span)
}

#[test]
fn denser_walker_shortens_the_revisit_gap() {
    let sparse = walker(WalkerPattern::Star, 3, 4, 70.0); // 12 satellites
    let dense = walker(WalkerPattern::Star, 6, 8, 70.0); //  48 satellites

    let g_sparse = revisit_gap(&sparse, 45.0, DAY);
    let g_dense = revisit_gap(&dense, 45.0, DAY);

    // The physically monotone metrics: adding satellites cannot lengthen the worst-case gap or lower
    // the covered fraction, and 4× the satellites must materially improve both. (`passes` is not
    // monotone — denser coverage merges overlapping windows into fewer, longer intervals.)
    assert!(
        g_dense.max_gap < g_sparse.max_gap,
        "dense gap {:.0}s not shorter than sparse {:.0}s",
        g_dense.max_gap,
        g_sparse.max_gap
    );
    assert!(
        g_dense.coverage_fraction > g_sparse.coverage_fraction,
        "dense fraction {:.3} not above sparse {:.3}",
        g_dense.coverage_fraction,
        g_sparse.coverage_fraction
    );
    // 48 satellites at 1200 km revisit a mid-latitude site well inside a day.
    assert!(
        g_dense.max_gap < DAY / 4.0,
        "dense max gap {:.0}s too long",
        g_dense.max_gap
    );
    assert!(g_dense.coverage_fraction > 0.0 && g_dense.coverage_fraction <= 1.0);
}

#[test]
fn flower_repeat_orbit_is_visible_from_the_ground() {
    // A Flower's satellites are elliptical repeating-ground-track orbits; confirm they close windows
    // at a site under the track (a sanity wiring of the elliptical generator through the engine).
    let fc = FlowerConstellation {
        petals: 14,
        days: 1,
        satellites: 8,
        eccentricity: 0.05,
        inclination: Angle::from_radians(63.434_f64.to_radians()),
        arg_periapsis: Angle::from_radians(270.0_f64.to_radians()),
        raan: Angle::from_radians(0.0),
        mean_anomaly: Angle::from_radians(0.0),
        frame_rate: OMEGA_E,
        mu: MU,
    };
    let mut sats = Vec::new();
    fc.generate(|e| sats.push(e));
    assert_eq!(sats.len(), 8);

    let m = revisit_gap(&sats, 0.0, DAY);
    assert!(m.passes > 0, "no passes from the flower constellation");
    assert!(m.max_gap < DAY, "site never covered");
}
