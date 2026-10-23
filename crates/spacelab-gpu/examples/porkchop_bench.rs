//! CPU (f64) vs GPU (f32) for a porkchop grid of Lambert transfers, across grid sizes — the speedup
//! and the f32-vs-f64 C3 agreement on converged cells.
//!
//! `cargo run -p spacelab-gpu --release --example porkchop_bench`

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::time::Instant;

use spacelab_gpu::{Gpu, PORKCHOP_NO_SOLUTION, ScreenState, porkchop_c3_cpu};

const MU_SUN: f64 = 1.327_124_400_18e20;
const AU: f64 = 1.495_978_707e11;
const DAY: f64 = 86_400.0;

fn circular(radius: f64, theta: f64) -> ScreenState {
    let speed = (MU_SUN / radius).sqrt();
    ScreenState::new(
        [
            (radius * theta.cos()) as f32,
            (radius * theta.sin()) as f32,
            0.0,
        ],
        [
            (-speed * theta.sin()) as f32,
            (speed * theta.cos()) as f32,
            0.0,
        ],
    )
}

fn grid(n: usize) -> (Vec<ScreenState>, Vec<ScreenState>, Vec<f32>, Vec<f32>) {
    // n departures over ~2 years, n arrivals 150–500 days later; Earth → Mars circular orbits.
    let dep: Vec<ScreenState> = (0..n)
        .map(|k| circular(AU, k as f64 / n as f64 * 4.0))
        .collect();
    let dep_t: Vec<f32> = (0..n)
        .map(|k| (k as f64 / n as f64 * 730.0 * DAY) as f32)
        .collect();
    let arr: Vec<ScreenState> = (0..n)
        .map(|k| circular(1.524 * AU, 0.4 + k as f64 / n as f64 * 5.0))
        .collect();
    let arr_t: Vec<f32> = (0..n)
        .map(|k| ((150.0 + k as f64 / n as f64 * 350.0) * DAY) as f32)
        .collect();
    (dep, arr, dep_t, arr_t)
}

fn main() {
    let gpu = Gpu::new();
    let mu = MU_SUN as f32;
    println!("porkchop grid of Lambert transfers — CPU (f64) vs GPU (f32):");
    if gpu.is_none() {
        println!("  (no GPU reachable; CPU only)");
    }
    for &n in &[64usize, 256, 1024] {
        let (dep, arr, dep_t, arr_t) = grid(n);
        let cells = (n * n) as f64 / 1e3;

        let t = Instant::now();
        let c = porkchop_c3_cpu(&dep, &arr, &dep_t, &arr_t, mu, true);
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

        let (gpu_ms, worst, compared) = gpu.as_ref().map_or((f64::NAN, 0.0, 0), |ctx| {
            let g = ctx.porkchop_c3(&dep, &arr, &dep_t, &arr_t, mu, true); // warm
            let t = Instant::now();
            let g2 = std::hint::black_box(ctx.porkchop_c3(&dep, &arr, &dep_t, &arr_t, mu, true));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            // The useful porkchop band: low departure energy (≤ ~45 km²/s²), where the transfer is
            // well conditioned. The high-C3 ridges are near-singular geometry that f32 cannot resolve
            // to f64 — visually masked, and refined on the CPU anyway.
            let (mut w, mut cmp) = (0.0f64, 0u32);
            for (gi, ci) in g.iter().zip(&c) {
                if *gi < 1e29 && *ci > 2.0e6 && *ci < 4.5e7 {
                    w = w.max((f64::from(*gi) - f64::from(*ci)).abs() / f64::from(*ci));
                    cmp += 1;
                }
            }
            let _ = g2;
            (ms, w, cmp)
        });

        let solved = c.iter().filter(|&&x| x < PORKCHOP_NO_SOLUTION).count();
        let speedup = if gpu_ms.is_finite() {
            format!(
                "{:.1}× GPU, worst C3 err {:.2}% ({compared} cells)",
                cpu_ms / gpu_ms,
                worst * 100.0
            )
        } else {
            "—".to_string()
        };
        println!(
            "  {n}×{n} = {cells:>6.1}k cells ({solved} solved):  CPU {cpu_ms:>8.2} ms   GPU {gpu_ms:>6.2} ms   {speedup}"
        );
    }
}
