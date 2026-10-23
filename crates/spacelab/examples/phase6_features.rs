//! Phase-6 ML gate on the REAL coverage engine: emit per-satellite coverage features as CSV.
//!
//! For each synthetic satellite this drives the production access-window engine
//! (`spacelab_coverage::revisit_map`) over a global grid under a real sensor `Constraint`, reduces the
//! per-cell coverage fractions to nine latitude-band means, and prints one CSV row. The companion
//! `experiments/phase6_real_engine.py` then clusters the rows and reports whether sensor-aware coverage
//! carries information beyond the orbital elements — the same null-hypothesis test the numpy notebook
//! answered, now on the engine that will actually serve it.
//!
//! Orbits are analytic circular (Earth-fixed via a `Mat3::rot3` rotation); the *coverage* computation
//! — event detection, sensor constraints, revisit metrics — is the real engine, which is the point.
//!
//! ```text
//! cargo run --release -p spacelab --example phase6_features > /tmp/phase6.csv
//! uv run --no-project --with numpy --with scikit-learn python experiments/phase6_real_engine.py /tmp/phase6.csv
//! ```

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::unwrap_used
)]

use core::f64::consts::TAU;

use spacelab_core::{Mat3, Vec3};
use spacelab_coverage::revisit_map;
use spacelab_frames::EARTH_ROTATION_RATE;
use spacelab_sensors::{AgileOptical, Constraint, ElevationMask, OrbitBound, SideLookingSAR};

const RE: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;
const LAT_BANDS: [f64; 9] = [-80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0];
const LONS: usize = 12;
const HORIZON: f64 = 86_400.0; // one day
const TOL: f64 = 1.0; // access-edge tolerance, seconds

/// splitmix64 → uniform in [0, 1), for reproducible jitter (no `rand` dependency).
fn u01(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn jitter(seed: u64, spread: f64) -> f64 {
    (u01(seed) - 0.5) * 2.0 * spread
}

/// A global grid of `(centre, zenith)` cells (spherical Earth), latitude-major.
fn global_grid() -> Vec<(Vec3, Vec3)> {
    let mut cells = Vec::with_capacity(LAT_BANDS.len() * LONS);
    for &lat in &LAT_BANDS {
        let phi = lat.to_radians();
        for j in 0..LONS {
            let lam = TAU * j as f64 / LONS as f64;
            let up = Vec3::new(phi.cos() * lam.cos(), phi.cos() * lam.sin(), phi.sin());
            cells.push((up.scale(RE), up));
        }
    }
    cells
}

/// Earth-fixed satellite position for a circular orbit `(a, inc, raan, phase0)`.
fn sat_closure(a: f64, inc: f64, raan: f64, phase0: f64) -> impl Fn(f64) -> Vec3 + Sync {
    let n = (MU / (a * a * a)).sqrt();
    move |t: f64| {
        let u = phase0 + n * t;
        let (cu, su) = (u.cos(), u.sin());
        let (cr, sr) = (raan.cos(), raan.sin());
        let (ci, si) = (inc.cos(), inc.sin());
        let eci = Vec3::new(
            a * (cu * cr - su * sr * ci),
            a * (cu * sr + su * cr * ci),
            a * (su * si),
        );
        Mat3::rot3(EARTH_ROTATION_RATE * t) * eci
    }
}

/// Per-latitude-band coverage and revisit features for one satellite under one sensor: nine mean
/// coverage fractions followed by nine mean max-gap ratios (worst-case revisit latency / horizon,
/// clamped to 1 for never-covered bands). The gap half captures *cadence*, which the plain coverage
/// fraction misses and which the sensor strongly shapes.
fn band_features<C: Constraint + Sync>(
    cells: &[(Vec3, Vec3)],
    sensor: &C,
    a: f64,
    sat: impl Fn(f64) -> Vec3 + Sync,
) -> [f64; 18] {
    let orbit = OrbitBound {
        semi_major_axis: a,
        eccentricity: 0.0,
        mu: MU,
    };
    let m = revisit_map(
        cells,
        sensor,
        &orbit,
        EARTH_ROTATION_RATE,
        sat,
        0.0,
        HORIZON,
        TOL,
        8,
    );
    let mut f = [0.0; 18];
    for b in 0..9 {
        let (mut cov, mut gap) = (0.0, 0.0);
        for j in 0..LONS {
            let cell = &m[b * LONS + j];
            cov += cell.coverage_fraction;
            gap += (cell.max_gap / HORIZON).min(1.0);
        }
        f[b] = cov / LONS as f64;
        f[9 + b] = gap / LONS as f64;
    }
    f
}

fn emit(id: usize, regime: &str, sensor: &str, a: f64, inc: f64, raan: f64, f: [f64; 18]) {
    print!(
        "{id},{regime},{sensor},{:.3},0.0,{:.4},{:.4}",
        a / 1000.0,
        inc.to_degrees(),
        raan.to_degrees()
    );
    for v in f {
        print!(",{v:.6}");
    }
    println!();
}

fn main() {
    let cells = global_grid();
    let wide = ElevationMask::new(5.0_f64.to_radians()).unwrap();
    let sar = SideLookingSAR::new(20.0_f64.to_radians(), 46.0_f64.to_radians()).unwrap();
    let agile = AgileOptical::new(20.0_f64.to_radians()).unwrap();

    print!("sat_id,regime,sensor,a_km,ecc,inc_deg,raan_deg");
    for i in 0..18 {
        print!(",c{i}");
    }
    println!();
    let mut id = 0;

    // PART A: four regimes, all under the wide geometric sensor.
    let regimes = [
        ("Starlink-LEO", RE + 550e3, 53.0),
        ("GNSS-MEO", RE + 20_180e3, 55.0),
        ("SSO-EO", RE + 700e3, 98.0),
        ("GEO-comms", 42_164e3, 0.1),
    ];
    for (name, a0, inc0) in regimes {
        for k in 0..12 {
            let seed = id as u64 * 97 + 1;
            let a = a0 * (1.0 + jitter(seed, 0.003));
            let inc = (inc0 + jitter(seed + 1, 1.0)).to_radians();
            let raan = u01(seed + 2) * TAU;
            let phase = u01(seed + 3) * TAU;
            let f = band_features(&cells, &wide, a, sat_closure(a, inc, raan, phase));
            emit(id, name, "wide", a, inc, raan, f);
            let _ = k;
            id += 1;
        }
    }

    // PART B: one LEO family, three sensor types — elements identical, coverage sensor-dependent.
    // Enough samples (60) for a supervised sensor classifier with a RAAN-grouped geo-holdout.
    for k in 0..60 {
        let seed = id as u64 * 97 + 1;
        let a = (RE + 550e3) * (1.0 + jitter(seed, 0.001));
        let inc = (53.0 + jitter(seed + 1, 0.5)).to_radians();
        let raan = u01(seed + 2) * TAU;
        let phase = u01(seed + 3) * TAU;
        let sat = sat_closure(a, inc, raan, phase);
        let (label, f) = match k % 3 {
            0 => ("wide", band_features(&cells, &wide, a, sat)),
            1 => ("sar", band_features(&cells, &sar, a, sat)),
            _ => ("agile", band_features(&cells, &agile, a, sat)),
        };
        emit(id, "LEO-family", label, a, inc, raan, f);
        id += 1;
    }
}
