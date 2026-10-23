//! Throughput of the coverage kernels, and evidence that they vectorise.
//!
//! `cargo run -p spacelab-sensors --release --example coverage_bench`
//!
//! Nothing here is a microbenchmark harness. It measures the one loop that dominates a
//! catalog-scale coverage query, on a grid large enough to leave cache, and reports cells per
//! second so the number can be compared against the 5e11 evaluations the MVP needs.

#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::doc_markdown,
    clippy::naive_bytecount
)]

use std::hint::black_box;
use std::time::Instant;

use spacelab_core::Vec3;
use spacelab_sensors::{
    CellGrid, Cone, ElevationMask, EllipsoidGrid, WGS84_GAMMA, ellipsoid_grid,
    kernel::{cone_covered_into, elevation_accumulate_into, elevation_covered_into},
};

const R: f64 = 6_378_137.0;

/// Six owned coordinate streams: `px, py, pz, zx, zy, zz`.
type OwnedColumns = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);
fn fibonacci_sphere(n: usize) -> OwnedColumns {
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut px = Vec::with_capacity(n);
    let mut py = Vec::with_capacity(n);
    let mut pz = Vec::with_capacity(n);
    let mut zx = Vec::with_capacity(n);
    let mut zy = Vec::with_capacity(n);
    let mut zz = Vec::with_capacity(n);
    for i in 0..n {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let th = golden * i as f64;
        let (x, y) = (r * th.cos(), r * th.sin());
        px.push(x * R);
        py.push(y * R);
        pz.push(z * R);
        zx.push(x);
        zy.push(y);
        zz.push(z);
    }
    (px, py, pz, zx, zy, zz)
}

fn main() {
    let n = 1 << 20; // ~1.05e6 cells, 48 MB of grid: well past the 16 MB L3
    let (px, py, pz, zx, zy, zz) = fibonacci_sphere(n);
    let grid = CellGrid::new(&px, &py, &pz, &zx, &zy, &zz).expect("rectangular");

    let mask = ElevationMask::new(5.0_f64.to_radians()).expect("mask in range");
    let cone = Cone::new(25.0_f64.to_radians()).expect("half-angle in range");
    let sat = Vec3::new(0.0, 0.0, R + 700e3);

    let mut out = vec![0u8; n];
    let mut counters = vec![0u32; n];

    let reps = 40;
    let bench = |name: &str, f: &mut dyn FnMut()| {
        for _ in 0..4 {
            f(); // warm the page tables and the branch predictors
        }
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        let dt = t0.elapsed().as_secs_f64();
        let cells = (n * reps) as f64;
        println!(
            "  {name:<26} {:>8.1} ms/pass   {:>7.2} Gcell/s   {:>6.2} ns/cell",
            dt / reps as f64 * 1e3,
            cells / dt / 1e9,
            dt / cells * 1e9
        );
    };

    println!(
        "grid: {n} cells ({:.0} MB), 40 passes each\n",
        (n * 48) as f64 / 1e6
    );
    bench("elevation_covered_into", &mut || {
        elevation_covered_into(mask, black_box(sat), &grid, &mut out).expect("lengths match");
        black_box(&out);
    });
    bench("elevation_accumulate_into", &mut || {
        elevation_accumulate_into(mask, black_box(sat), &grid, &mut counters)
            .expect("lengths match");
        black_box(&counters);
    });
    bench("cone_covered_into", &mut || {
        cone_covered_into(cone, black_box(sat), &grid, &mut out).expect("lengths match");
        black_box(&out);
    });

    let egrid = EllipsoidGrid::new(&px, &py, &pz, WGS84_GAMMA).expect("rectangular");
    println!("\n  same query, zenith derived instead of stored (24 B/cell):");
    bench("ellipsoid/elevation", &mut || {
        ellipsoid_grid::elevation_covered_into(mask, black_box(sat), &egrid, &mut out)
            .expect("lengths match");
        black_box(&out);
    });
    bench("ellipsoid/accumulate", &mut || {
        ellipsoid_grid::elevation_accumulate_into(mask, black_box(sat), &egrid, &mut counters)
            .expect("lengths match");
        black_box(&counters);
    });

    // ---- thread scaling, cells partitioned, no shared writes, no atomics ----
    println!("\n  thread scaling of ellipsoid/elevation (cells partitioned):");
    println!(
        "  {:>7}  {:>10}  {:>12}  {:>10}  {:>8}",
        "threads", "ms/pass", "Gcell/s", "GB/s", "speedup"
    );
    let mut base = 0.0_f64;
    for &nt in &[1usize, 2, 4, 8, 16] {
        let chunk = n.div_ceil(nt);
        let t0 = Instant::now();
        for _ in 0..reps {
            std::thread::scope(|scope| {
                for (k, obuf) in out.chunks_mut(chunk).enumerate() {
                    let lo = k * chunk;
                    let hi = (lo + obuf.len()).min(n);
                    let (a, b, c) = (&px[lo..hi], &py[lo..hi], &pz[lo..hi]);
                    scope.spawn(move || {
                        let sub = EllipsoidGrid::new(a, b, c, WGS84_GAMMA).expect("rectangular");
                        ellipsoid_grid::elevation_covered_into(mask, sat, &sub, obuf)
                            .expect("lengths match");
                    });
                }
            });
        }
        let dt = t0.elapsed().as_secs_f64();
        let cells = (n * reps) as f64;
        let gcs = cells / dt / 1e9;
        if nt == 1 {
            base = gcs;
        }
        println!(
            "  {nt:>7}  {:>10.2}  {gcs:>12.2}  {:>10.1}  {:>7.2}x",
            dt / reps as f64 * 1e3,
            gcs * 24.0,
            gcs / base
        );
    }

    let seen = out.iter().filter(|&&c| c == 1).count();
    println!(
        "\n  cells covered by the last kernel: {seen} ({:.2}% of the sphere)",
        100.0 * seen as f64 / n as f64
    );
}
