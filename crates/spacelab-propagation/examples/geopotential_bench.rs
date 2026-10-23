//! Throughput of the tesseral geopotential acceleration, by degree.
//!
//! `cargo run -p spacelab-propagation --release --example geopotential_bench`
//!
//! Not a microbenchmark harness — a warm loop over one force evaluation, the cost that dominates a
//! high-fidelity propagation (one call per integrator stage). Reports µs per evaluation so the degree
//! trade-off is visible and any optimisation can be measured against it.

#![allow(
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::cast_lossless
)]

use std::hint::black_box;
use std::time::Instant;

use spacelab_propagation::GeoPotential;
use spacelab_propagation::forces::Acceleration;

fn main() {
    let r = [6_778_000.0_f64, 1_200_000.0, 900_000.0];
    let v = [0.0; 3];
    println!("tesseral geopotential acceleration (XGM2019e), warm loop — scalar vs f64x4 SIMD:");
    for &degree in &[4usize, 8, 20, 40, 70, 100, 120] {
        let g = GeoPotential::embedded(degree).expect("degree within model");
        for _ in 0..1000 {
            black_box(g.acceleration(black_box(0.0), black_box(&r), &v));
            black_box(g.acceleration_simd(black_box(&r)));
        }
        let reps = 20_000;
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(g.acceleration(black_box(0.0), black_box(&r), &v));
        }
        let scalar = t0.elapsed().as_secs_f64() / reps as f64;
        let t0 = Instant::now();
        for _ in 0..reps {
            black_box(g.acceleration_simd(black_box(&r)));
        }
        let simd = t0.elapsed().as_secs_f64() / reps as f64;
        let terms = (degree + 1) * (degree + 2) / 2;
        println!(
            "  degree {degree:>3}: scalar {:>7.2} µs   SIMD {:>7.2} µs   {:>4.2}× faster   ({:>6.0} → {:>6.0} evals/s, {terms} terms)",
            scalar * 1e6,
            simd * 1e6,
            scalar / simd,
            1.0 / scalar,
            1.0 / simd,
        );
    }
    println!(
        "\nThe per-stage scratch is three rolling harmonic rows (O(degree)), not the full DIM×DIM\n\
         array, so low degrees no longer pay the old zero-init fixed cost; per-term cost is flat past\n\
         degree ~40 — the recursion is work-bound."
    );
}
