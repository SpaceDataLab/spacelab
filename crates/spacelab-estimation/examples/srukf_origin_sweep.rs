//! How accurate the SR-UKF's posterior is as the state mean grows away from its uncertainty.
//!
//! A sigma-point filter perturbs its mean by `γS`, which for the Van der Merwe defaults is a
//! *thousandth* of the standard deviation. When the mean is an ECI position in metres and the
//! covariance is metre-level, that perturbation is `1e-6` on a number of order `1e7`, and every
//! quantity built from the cloud carries the mean's rounding rather than the perturbation's.
//!
//! The problem is linear and its Kalman gain is closed-form, so the filter's own error is the whole
//! difference. Run it as
//!
//! ```text
//! cargo run -p spacelab-estimation --example srukf_origin_sweep
//! ```

// Matrix code here indexes by one i/j/k as the crate's own does, and an example that cannot even
// build its prior should say so loudly.
#![allow(clippy::needless_range_loop, clippy::expect_used)]

use spacelab_estimation::{SquareRootUnscentedKalman, UnscentedParams, unscented_transform};

/// `P₀ = diag(1e-6, 4e-6)`, `σ = 1e-3`, `h(x) = x₀ + x₁` — millimetre-scale uncertainty, the regime
/// an orbit-determination filter is in once it has converged.
const P0: [[f64; 2]; 2] = [[1e-6, 0.0], [0.0, 4e-6]];
const SIGMA: f64 = 1e-3;
const INNOVATION: f64 = 1e-3;

fn main() {
    println!("            ---------- SR-UKF update ----------   plain transform");
    println!("   |x̂|       mean      covariance   ulp(x̂)/|Kν|     covariance");
    for exponent in 0..=9 {
        let mean = libm::pow(10.0, f64::from(exponent));
        let (mean_error, covariance_error, floor) = departure(mean);
        let transform = transform_departure(mean);
        println!(
            "  1e{exponent}   {mean_error:9.2e}   {covariance_error:9.2e}   {floor:9.2e}   {transform:9.2e}"
        );
    }
}

/// The plain (covariance-form) transform's departure from affine exactness at state `mean`, as the
/// worst entry of `|P_ut − A P Aᵀ|` scaled by `√(Pᵢᵢ Pⱼⱼ)`.
///
/// Same defect, one amplification worse: the center covariance weight is `W₀ᶜ ≈ −1/α² = −1e6`, and
/// it multiplies the *square* of the mean offset — a quantity that is zero in exact arithmetic for a
/// symmetric cloud, so what it actually multiplies is rounding.
fn transform_departure(mean: f64) -> f64 {
    let x = [mean, -2.0 * mean, 0.5 * mean];
    let s = [[2e-3, 0.0, 0.0], [5e-4, 1.5e-3, 0.0], [-3e-4, 4e-4, 1e-3]];
    let a = [[1.0, 0.2, -0.1], [0.0, 1.3, 0.4], [0.5, -0.2, 0.9]];

    let (_, cov) = unscented_transform::<3, 3, 7>(&UnscentedParams::gaussian(), &x, &s, |v| {
        let mut y = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                y[i] += a[i][j] * v[j];
            }
        }
        y
    });

    // (A S)(A S)ᵀ, so the reference never squares the mean.
    let mut asx = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                asx[i][j] += a[i][k] * s[k][j];
            }
        }
    }
    let mut want = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                want[i][j] += asx[i][k] * asx[j][k];
            }
        }
    }
    let mut worst: f64 = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let scale = libm::sqrt(want[i][i] * want[j][j]);
            worst = worst.max(libm::fabs(cov[i][j] - want[i][j]) / scale);
        }
    }
    worst
}

/// The relative departure of one SR-UKF update from the closed-form Kalman answer at state `mean`,
/// as `(posterior mean, posterior covariance, the floor set by storing the mean at all)`.
fn departure(mean: f64) -> (f64, f64, f64) {
    let x0 = [mean, mean];
    let mut filter =
        SquareRootUnscentedKalman::<2, 5>::from_covariance(x0, &P0, UnscentedParams::gaussian())
            .expect("P0 is positive-definite");

    // `z − h(x̂)` is exact by Sterbenz, so the reference uses the innovation the filter actually
    // sees, not the one this line asked for. Quantizing `z` is the caller's floor, not the filter's.
    let predicted = x0[0] + x0[1];
    let z = predicted + INNOVATION;
    let innovation = z - predicted;

    filter.update(|x| [x[0] + x[1]], &[z], &[[SIGMA]]);

    // K = P Hᵀ (H P Hᵀ + R)⁻¹ with H = [1, 1].
    let hph_r = P0[0][0] + P0[1][1] + SIGMA * SIGMA;
    let gain = [P0[0][0] / hph_r, P0[1][1] / hph_r];
    let correction = [gain[0] * innovation, gain[1] * innovation];

    let mut mean_error: f64 = 0.0;
    for a in 0..2 {
        let measured = filter.x[a] - x0[a];
        mean_error =
            mean_error.max(libm::fabs(measured - correction[a]) / libm::fabs(correction[a]));
    }

    // P⁺ = P⁻ − K (H P Hᵀ + R) Kᵀ.
    let posterior = filter.covariance();
    let mut covariance_error: f64 = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            let reference = if a == b { P0[a][a] } else { 0.0 } - gain[a] * hph_r * gain[b];
            covariance_error = covariance_error
                .max(libm::fabs(posterior[a][b] - reference) / libm::fabs(reference));
        }
    }

    // Below this the answer cannot be read back out of `x` at all: the correction is stored into a
    // number of magnitude `mean`, so it is quantized to that number's ulp.
    let floor = ulp(mean) / libm::fabs(correction[0]);
    (mean_error, covariance_error, floor)
}

fn ulp(x: f64) -> f64 {
    f64::from_bits(x.to_bits() + 1) - x
}
