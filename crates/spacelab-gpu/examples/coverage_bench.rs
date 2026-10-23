//! CPU (f64) vs GPU (f32) for the instantaneous constellation coverage count, across grid sizes —
//! shows where the GPU overtakes the O(ground × satellites) CPU sweep.
//!
//! `cargo run -p spacelab-gpu --release --example coverage_bench`

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::time::Instant;

use spacelab_gpu::{Ecef, Gpu, coverage_counts_cpu};

fn sphere_points(n: usize, radius: f32, seed: u64) -> Vec<Ecef> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32
    };
    (0..n)
        .map(|_| {
            let lon = next() * core::f32::consts::TAU;
            let colat = (2.0 * next() - 1.0).acos();
            Ecef::new([
                radius * colat.sin() * lon.cos(),
                radius * colat.sin() * lon.sin(),
                radius * colat.cos(),
            ])
        })
        .collect()
}

fn main() {
    let gpu = Gpu::new();
    let sats = sphere_points(600, 7.0e6, 0x1234); // a 600-satellite mega-constellation
    let mask = 10.0_f32.to_radians();
    println!(
        "instantaneous coverage of a {}-satellite constellation — CPU (f64) vs GPU (f32):",
        sats.len()
    );
    if gpu.is_none() {
        println!("  (no GPU reachable; CPU only)");
    }
    for &g in &[10_000usize, 100_000, 1_000_000] {
        let grid = sphere_points(g, 6.371e6, 0x9abc);
        let work = (g as f64) * (sats.len() as f64) / 1e6;

        let cpu_ms = if g <= 100_000 {
            let t = Instant::now();
            let _ = std::hint::black_box(coverage_counts_cpu(&sats, &grid, mask));
            t.elapsed().as_secs_f64() * 1e3
        } else {
            f64::NAN
        };

        let gpu_ms = gpu.as_ref().map_or(f64::NAN, |ctx| {
            let _ = ctx.coverage_counts(&sats, &grid, mask); // warm
            let t = Instant::now();
            let _ = std::hint::black_box(ctx.coverage_counts(&sats, &grid, mask));
            t.elapsed().as_secs_f64() * 1e3
        });

        let speedup = if cpu_ms.is_finite() && gpu_ms.is_finite() {
            format!("{:.1}× GPU", cpu_ms / gpu_ms)
        } else {
            "—".to_string()
        };
        println!(
            "  grid = {g:>8} pts ({work:>7.1} M point·sat):  CPU {cpu_ms:>9.2} ms   GPU {gpu_ms:>7.2} ms   {speedup}"
        );
    }
}
