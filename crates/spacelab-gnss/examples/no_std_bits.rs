//! This crate's fused multiply-add kernels on a fixed sweep, printed as bit patterns.
//!
//! `just no-std` runs it with `std` and again without, and the two outputs must be equal. With `std`
//! the kernels call `f64::mul_add`, without it `libm::fma`; both are IEEE 754 fusedMultiplyAdd. The
//! route taken goes to stderr, so the recipe can check that the two runs took different ones.

use spacelab_gnss::{ClockCorrection, OrbitCorrection};

const SAMPLES: u32 = 20_000;

/// `SplitMix64`, as a fixed stream of uniform doubles in `[−1, 1)`.
fn stream(mut state: u64) -> impl FnMut() -> f64 {
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        2.0 * (f64::from_bits(0x3FF0_0000_0000_0000 | (z >> 12)) - 1.0) - 1.0
    }
}

fn bits(values: &[f64]) -> String {
    let words: Vec<String> = values
        .iter()
        .map(|v| format!(" {:016x}", v.to_bits()))
        .collect();
    words.concat()
}

fn main() {
    eprintln!(
        "route: {}",
        if cfg!(feature = "std") { "std" } else { "libm" }
    );
    let mut next = stream(0x6_7155);
    for i in 0..SAMPLES {
        // SSR magnitudes: metres and millimetres per second, an update interval of tens of seconds.
        let reference = 302_400.0 * (next() + 1.0);
        let orbit = OrbitCorrection {
            prn: 1,
            iode: 0,
            reference,
            delta: [2.0 * next(), 2.0 * next(), 2.0 * next()],
            delta_rate: [1e-3 * next(), 1e-3 * next(), 1e-3 * next()],
        };
        let clock = ClockCorrection {
            prn: 1,
            reference,
            coefficients: [5.0 * next(), 1e-3 * next(), 1e-6 * next()],
        };
        let tow = reference + 60.0 * next();
        println!(
            "gnss {i}{}{}",
            bits(&orbit.at(tow)),
            bits(&[clock.at(tow), clock.seconds_at(tow)]),
        );
    }
}
