//! CPU (f64) vs GPU (f32) for the all-pairs closest-approach conjunction screen, across catalogue
//! sizes — shows where the GPU overtakes the O(N²) CPU sweep.
//!
//! `cargo run -p spacelab-gpu --release --example conjunction_bench`

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::time::Instant;

use spacelab_gpu::{Gpu, ScreenState, pairwise_min_approach_cpu};

fn catalogue(n: usize) -> Vec<ScreenState> {
    let mut s = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = || {
        s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32
    };
    (0..n)
        .map(|_| {
            let radius = 6.9e6 + next() * 3.0e5;
            let (a, b) = (
                next() * core::f32::consts::TAU,
                next() * core::f32::consts::TAU,
            );
            let r = [
                radius * a.cos() * b.sin(),
                radius * a.sin() * b.sin(),
                radius * b.cos(),
            ];
            let speed = 7.5e3 + next() * 3.0e2;
            let v = [
                speed * (next() - 0.5),
                speed * (next() - 0.5),
                speed * (next() - 0.5),
            ];
            ScreenState::new(r, v)
        })
        .collect()
}

fn main() {
    let gpu = Gpu::new();
    println!("all-pairs closest-approach conjunction screen — CPU (f64) vs GPU (f32):");
    if gpu.is_none() {
        println!("  (no GPU reachable; CPU only)");
    }
    for &n in &[1_000usize, 4_000, 16_000, 32_000] {
        let cat = catalogue(n);
        let pairs = (n as f64) * (n as f64) / 1e6;

        let cpu_ms = if n <= 16_000 {
            let t = Instant::now();
            let _ = std::hint::black_box(pairwise_min_approach_cpu(&cat, 300.0));
            t.elapsed().as_secs_f64() * 1e3
        } else {
            f64::NAN
        };

        let gpu_ms = gpu.as_ref().map_or(f64::NAN, |g| {
            let _ = g.pairwise_min_approach(&cat, 300.0); // warm (compile + allocate)
            let t = Instant::now();
            let _ = std::hint::black_box(g.pairwise_min_approach(&cat, 300.0));
            t.elapsed().as_secs_f64() * 1e3
        });

        let speedup = if cpu_ms.is_finite() && gpu_ms.is_finite() {
            format!("{:.1}× GPU", cpu_ms / gpu_ms)
        } else {
            "—".to_string()
        };
        println!(
            "  N = {n:>6} ({pairs:>6.1} M pairs):  CPU {cpu_ms:>9.2} ms   GPU {gpu_ms:>7.2} ms   {speedup}"
        );
    }
}
