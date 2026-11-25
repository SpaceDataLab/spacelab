//! This crate's kernels that call a `std` float method outside the `std`-gated modules, on a fixed
//! sweep, printed as bit patterns.
//!
//! `just no-std` runs it with `std` and again without, and the two outputs must be equal. With `std`
//! the kernels call `f64::mul_add` and `f64::floor`, without it `libm::fma` and `libm::floor`; each
//! pair is the same IEEE 754 operation, fusedMultiplyAdd and the exact roundToIntegralTowardNegative.
//! The route taken goes to stderr, so the recipe can check that the two runs took different ones.

use spacelab_propagation::attitude::{Inertia, exp_so3, integrate_attitude_torqued};
use spacelab_propagation::space_weather::{
    FluxQuality, SpaceWeatherDay, nrlmsise_ap_array, sdtm_ap_history,
};

const RUNS: u32 = 1_000;
const STEPS: usize = 25;

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
    let mut next = stream(0x9_7091);
    let mut vector = |scale: f64| [scale * next(), scale * next(), scale * next()];

    // The fixed-step time grid `t = t0 + k·dt`, read through the moment's stage times. Its one rounding
    // shows only where `t0` does not swamp `k·dt`: an epoch of the order of the span, and one that
    // cancels a grid point, where `t` is the product's own rounding error. A distant epoch is kept too.
    for i in 0..RUNS {
        let r0 = exp_so3(vector(3.2));
        let w0 = vector(0.05);
        let Some(inertia) = Inertia::principal(vector(4.0).map(|m| 10.0 + m)) else {
            continue;
        };
        let [t0, dt, k] = vector(1.0);
        let dt = 0.1 + 0.05 * dt;
        let t0 = match i % 3 {
            0 => 1e5 * (t0 + 1.0),
            1 => 3.0 * t0,
            _ => -(dt * f64::from(i % 25)),
        };
        let mut times = Vec::with_capacity(4 * STEPS);
        let (r, w) = integrate_attitude_torqued(&r0, w0, t0, &inertia, dt, STEPS, |_, w, t| {
            times.push(t);
            [-k * w[0], -k * w[1], -k * w[2]]
        });
        println!(
            "attitude {i}{}{}{}",
            bits(&r.concat()),
            bits(&w),
            bits(&times)
        );
    }

    // The 3-hourly interval of a UT second, across every boundary of the day and an ulp either side.
    let days: Vec<SpaceWeatherDay> = (1..=6)
        .map(|day| SpaceWeatherDay {
            year: 2024,
            month: 5,
            day,
            ap: 10.0,
            ap3h: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0].map(|k| f64::from(day) * 100.0 + k),
            kp3h: [1.0; 8],
            f107_observed: 150.0,
            f107_adjusted: 150.0,
            f107_center81: 150.0,
            f107_last81: 150.0,
            quality: FluxQuality::Observed,
        })
        .collect();
    let mut seconds = Vec::new();
    for boundary in 0..=8 {
        let sec = 10_800.0 * f64::from(boundary);
        seconds.extend([sec.next_down(), sec, sec.next_up()]);
    }
    seconds.extend((0..1_000).map(|_| 43_200.0 * (next() + 1.0)));
    for (i, &sec) in seconds.iter().enumerate() {
        let ap = nrlmsise_ap_array(&days, 2024, 5, 6, sec).unwrap_or([f64::NAN; 7]);
        let history = sdtm_ap_history::<8>(&days, 2024, 5, 6, sec).unwrap_or([f64::NAN; 8]);
        println!(
            "space_weather {i}{}{}{}",
            bits(&[sec]),
            bits(&ap),
            bits(&history)
        );
    }
}
