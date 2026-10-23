//! CPU (f64) vs GPU (f32) for batch near-Earth SGP4 over a catalogue × a set of epochs — the speedup
//! and the f32-vs-f64 position agreement.
//!
//! `cargo run -p spacelab-gpu --release --example sgp4_bench`

#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

use spacelab_gpu::batch_sgp4::Sgp4Sat;
use spacelab_gpu::{Gpu, sgp4_propagate_cpu};

// A handful of real near-Earth element sets, replicated to a synthetic catalogue.
const TLES: &[(&str, &str)] = &[
    (
        "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992",
        "2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008",
    ),
    (
        "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
    ),
    (
        "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
        "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
    ),
];

fn catalogue(n: usize) -> Vec<Sgp4Sat> {
    let base: Vec<Sgp4Sat> = TLES
        .iter()
        .filter_map(|(l1, l2)| {
            sgp4::Elements::from_tle(None, l1.as_bytes(), l2.as_bytes())
                .ok()
                .and_then(|e| Sgp4Sat::from_elements(&e))
        })
        .collect();
    (0..n).map(|k| base[k % base.len()]).collect()
}

fn main() {
    let gpu = Gpu::new();
    let times: Vec<f32> = (0..96).map(|k| k as f32 * 15.0).collect(); // 96 epochs over 24 h
    println!(
        "batch near-Earth SGP4 over {} epochs — CPU (f64) vs GPU (f32):",
        times.len()
    );
    if gpu.is_none() {
        println!("  (no GPU reachable; CPU only)");
    }
    for &n in &[1_000usize, 10_000, 30_000] {
        let sats = catalogue(n);
        let props = (n * times.len()) as f64 / 1e6;

        let t = Instant::now();
        let c = sgp4_propagate_cpu(&sats, &times);
        let cpu_ms = t.elapsed().as_secs_f64() * 1e3;

        let (gpu_ms, worst) = gpu.as_ref().map_or((f64::NAN, 0.0), |ctx| {
            let init: Vec<_> = sats.iter().map(|s| s.to_gpu()).collect();
            let g = ctx.sgp4_propagate(&init, &times); // km
            let t = Instant::now();
            let _ = std::hint::black_box(ctx.sgp4_propagate(&init, &times));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            // GPU (km) vs the CPU oracle (m) → compare in km.
            let mut w = 0.0f64;
            for (gi, ci) in g.iter().zip(&c) {
                let dr = ((f64::from(gi.r[0]) - f64::from(ci.r[0]) / 1000.0).powi(2)
                    + (f64::from(gi.r[1]) - f64::from(ci.r[1]) / 1000.0).powi(2)
                    + (f64::from(gi.r[2]) - f64::from(ci.r[2]) / 1000.0).powi(2))
                .sqrt();
                w = w.max(dr);
            }
            (ms, w)
        });

        let speedup = if gpu_ms.is_finite() {
            format!(
                "{:.0}× GPU, worst {:.0} m vs f64",
                cpu_ms / gpu_ms,
                worst * 1000.0
            )
        } else {
            "—".to_string()
        };
        println!(
            "  {n:>6} sats ({props:>5.1} M states):  CPU {cpu_ms:>8.1} ms   GPU {gpu_ms:>6.1} ms   {speedup}"
        );
    }
}
