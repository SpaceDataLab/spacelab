//! This crate's fused multiply-add kernels on a fixed sweep, printed as bit patterns.
//!
//! `just no-std` runs it with `std` and again without, and the two outputs must be equal. With `std`
//! the kernels call `f64::mul_add`, without it `libm::fma`; both are IEEE 754 fusedMultiplyAdd. The
//! route taken goes to stderr, so the recipe can check that the two runs took different ones.

use spacelab_control::{AttitudeError, ControlLaw, MrpLyapunov, QuaternionPd};
use spacelab_propagation::attitude::{Inertia, exp_so3};

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
    let mut next = stream(0xC0_7201);
    let mut vector = |scale: f64| [scale * next(), scale * next(), scale * next()];
    for i in 0..SAMPLES {
        let actual = exp_so3(vector(3.2));
        let reference = exp_so3(vector(3.2));
        let (omega, omega_reference) = (vector(0.1), vector(0.1));
        let moments = vector(5.0).map(|m| 10.0 + m);
        let gains = vector(1.0).map(|g| 0.5 + 0.25 * g);
        let (Some(inertia), Some(pd), Some(mrp)) = (
            Inertia::principal(moments),
            QuaternionPd::new(gains[0], gains[1]),
            MrpLyapunov::new(gains[1], gains[2]),
        ) else {
            continue;
        };
        let error = AttitudeError::new(&actual, &reference, omega, omega_reference);
        println!(
            "control {i}{}{}{}{}{}",
            bits(&error.quaternion()),
            bits(&[error.angle()]),
            bits(&pd.torque(&error, &inertia)),
            bits(&mrp.torque(&error, &inertia)),
            bits(&[mrp.lyapunov(&error, &inertia)]),
        );
    }
}
