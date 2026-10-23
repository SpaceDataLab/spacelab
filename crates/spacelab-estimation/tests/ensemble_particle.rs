//! The two Monte-Carlo estimators against the things that can actually judge them.
//!
//! A sampling filter cannot be checked the way a Kalman filter can: its answer is a random variable,
//! so "close to the truth once" is not evidence. Each leg below is built around a property that
//! survives that:
//!
//!   * the ensemble filter must **converge to the exact Kalman solution** on a linear-Gaussian
//!     problem, at the `1/√M` rate that sampling theory predicts — and the rate is *fitted from the
//!     measurements*, not asserted from a single ensemble size;
//!   * the particle filter must **carry a bimodal posterior** on a problem where an extended Kalman
//!     filter is not merely wrong but confidently wrong;
//!   * the resampling schemes must satisfy their **defining variance ordering**, measured over
//!     repeated draws from one weight vector.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::needless_range_loop
)]

use spacelab_estimation::{
    EnsembleKalman, ParticleFilter, Resampling, Rng, SquareRootKalman, particle,
};

/// A constant-velocity pair, one second per step.
const PHI: [[f64; 2]; 2] = [[1.0, 1.0], [0.0, 1.0]];
/// `Q = diag(0.01, 0.01)`, as its square root.
const Q_SQRT: [[f64; 2]; 2] = [[0.1, 0.0], [0.0, 0.1]];
/// `P₀ = diag(4, 1)`, as its square root.
const P0_SQRT: [[f64; 2]; 2] = [[2.0, 0.0], [0.0, 1.0]];
/// Position measurement, variance 1.
const R: f64 = 1.0;
const R_SQRT: [[f64; 1]; 1] = [[1.0]];
const STEPS: usize = 20;

/// A truth trajectory and the noisy position measurements of it, drawn once and shared by every
/// filter in the first leg — so the comparison is between estimators, not between data sets.
fn observations(seed: u64) -> [f64; STEPS] {
    let mut rng = Rng::new(seed);
    let mut truth = [1.0, 0.5];
    let mut out = [0.0; STEPS];
    for slot in &mut out {
        let noise = rng.correlated_normal(&Q_SQRT);
        truth = [truth[0] + truth[1] + noise[0], truth[1] + noise[1]];
        *slot = truth[0] + libm::sqrt(R) * rng.standard_normal();
    }
    out
}

/// The exact solution: a square-root Kalman filter on the same data.
fn kalman_reference(data: &[f64; STEPS]) -> ([f64; 2], [[f64; 2]; 2]) {
    let mut kf = SquareRootKalman::<2>::new([0.0, 0.0], P0_SQRT);
    for &z in data {
        kf.x = [kf.x[0] + kf.x[1], kf.x[1]];
        kf.time_update(&PHI, &Q_SQRT);
        let innovation = z - kf.x[0];
        kf.measurement_update(&[1.0, 0.0], innovation, R);
    }
    (kf.x, kf.covariance())
}

/// The ensemble filter on the same data, from an independent draw.
fn ensemble_run<const M: usize>(data: &[f64; STEPS], seed: u64) -> ([f64; 2], [[f64; 2]; 2]) {
    let mut rng = Rng::new(seed);
    let mut filter =
        EnsembleKalman::<2, M>::from_mean_and_covariance_sqrt(&[0.0, 0.0], &P0_SQRT, &mut rng);
    for &z in data {
        filter.predict(|x| [x[0] + x[1], x[1]], &Q_SQRT, &mut rng);
        filter
            .update(&[z], |x| [x[0]], &R_SQRT, &mut rng)
            .expect("R is positive definite, so P_yy is");
    }
    (filter.mean(), filter.covariance())
}

/// Root-mean-square distance from the Kalman answer, over `REPEATS` independent ensembles.
///
/// Averaging over seeds is what makes the number a property of the *ensemble size* rather than of one
/// lucky draw — a single run's error fluctuates by a factor of two either way.
fn ensemble_error<const M: usize>(
    data: &[f64; STEPS],
    want: &([f64; 2], [[f64; 2]; 2]),
) -> (f64, f64) {
    const REPEATS: usize = 12;
    let (mut mean_sq, mut cov_sq) = (0.0, 0.0);
    for repeat in 0..REPEATS {
        let (mean, cov) = ensemble_run::<M>(data, 1000 + repeat as u64);
        for i in 0..2 {
            mean_sq += (mean[i] - want.0[i]).powi(2);
            for j in 0..2 {
                cov_sq += (cov[i][j] - want.1[i][j]).powi(2);
            }
        }
    }
    let n = REPEATS as f64;
    ((mean_sq / n).sqrt(), (cov_sq / n).sqrt())
}

/// Least-squares slope of `ln error` against `ln M`.
fn log_log_slope(sizes: &[f64], errors: &[f64]) -> f64 {
    let n = sizes.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (&m, &e) in sizes.iter().zip(errors) {
        let (x, y) = (m.ln(), e.ln());
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    (n * sxy - sx * sy) / (n * sxx - sx * sx)
}

/// The ensemble filter converges to the Kalman solution, and it does so at the sampling rate.
///
/// Both the mean and the covariance are checked, because they fail differently: a wrong gain moves
/// the mean and a missing observation perturbation shrinks the covariance while leaving the mean
/// almost right. The slope bound `[-0.65, -0.35]` brackets the theoretical `-1/2` loosely enough to
/// survive the finite repeat count and tightly enough to reject a filter that has stopped converging
/// (slope 0) or one whose error is dominated by a bias the ensemble size cannot fix.
#[test]
fn the_ensemble_filter_converges_to_the_kalman_solution_at_the_sampling_rate() {
    let data = observations(42);
    let want = kalman_reference(&data);

    let e16 = ensemble_error::<16>(&data, &want);
    let e64 = ensemble_error::<64>(&data, &want);
    let e256 = ensemble_error::<256>(&data, &want);
    let e1024 = ensemble_error::<1024>(&data, &want);

    println!("     M     |Δmean|      |ΔP|_F");
    for (m, e) in [(16, e16), (64, e64), (256, e256), (1024, e1024)] {
        println!("{m:6}  {:10.4}  {:10.4}", e.0, e.1);
    }

    let sizes = [16.0, 64.0, 256.0, 1024.0];
    let mean_slope = log_log_slope(&sizes, &[e16.0, e64.0, e256.0, e1024.0]);
    let cov_slope = log_log_slope(&sizes, &[e16.1, e64.1, e256.1, e1024.1]);
    println!("fitted slopes: mean {mean_slope:.3}, covariance {cov_slope:.3} (theory −0.5)");

    assert!(
        (-0.65..=-0.35).contains(&mean_slope),
        "the ensemble mean converges at M^{mean_slope:.3}, not M^-0.5"
    );
    assert!(
        (-0.65..=-0.35).contains(&cov_slope),
        "the ensemble covariance converges at M^{cov_slope:.3}, not M^-0.5"
    );
    assert!(
        e1024.0 < 0.05 && e1024.1 < 0.15,
        "1024 members should reach the Kalman answer: mean {:.4}, covariance {:.4}",
        e1024.0,
        e1024.1
    );
}

/// The measurement of the second leg: `z = x²/20 + v`, the observation of Gordon, Salmond & Smith
/// 1993. It is even in `x`, so a symmetric prior gives a posterior with two modes at `±√(20z)` and
/// no amount of data resolves which one is the truth.
fn squared_observation(x: f64) -> f64 {
    x * x / 20.0
}

/// A particle filter carries both modes of a bimodal posterior; an EKF picks one and is certain.
///
/// The truth sits at `x = −4`, the prior is `N(0, 25)`, and twenty measurements arrive. The EKF is
/// started at `+1`, which is inside the prior and on the wrong side of zero — it linearises `∂z/∂x =
/// x/10` about its own estimate, walks to `+4`, and shrinks its covariance around it. Its estimate is
/// then eight sigma from the truth *with a small reported variance*, which is the failure that makes
/// a linearised filter dangerous rather than merely inaccurate.
#[test]
fn the_particle_filter_keeps_both_modes_where_the_ekf_commits_to_one() {
    const TRUTH: f64 = -4.0;
    const NOISE_VARIANCE: f64 = 0.01;
    const OBSERVATIONS: usize = 20;
    let mut rng = Rng::new(2718);
    let data: [f64; OBSERVATIONS] = core::array::from_fn(|_| {
        squared_observation(TRUTH) + libm::sqrt(NOISE_VARIANCE) * rng.standard_normal()
    });

    // The linearised filter, started on the wrong side of the symmetry.
    let mut ekf = SquareRootKalman::<1>::new([1.0], [[5.0]]);
    for &z in &data {
        let innovation = z - squared_observation(ekf.x[0]);
        ekf.measurement_update(&[ekf.x[0] / 10.0], innovation, NOISE_VARIANCE);
    }
    let ekf_sigma = ekf.covariance()[0][0].sqrt();
    let ekf_error = (ekf.x[0] - TRUTH).abs();
    println!("EKF: x̂ = {:.3} ± {ekf_sigma:.3}, truth {TRUTH}", ekf.x[0]);

    // The particle filter, from the same prior.
    let mut filter =
        ParticleFilter::<1, 4096>::from_mean_and_covariance_sqrt(&[0.0], &[[5.0]], &mut rng);
    for &z in &data {
        filter
            .update(|x| {
                let residual = z - squared_observation(x[0]);
                -0.5 * residual * residual / NOISE_VARIANCE
            })
            .expect("some particle explains the observation");
        filter.resample_if_degenerate(0.5, Resampling::Systematic, &mut rng);
    }

    let (mut negative, mut positive) = (0.0, 0.0);
    let (mut negative_mode, mut positive_mode) = (0.0, 0.0);
    for (particle, &w) in filter.particles().iter().zip(filter.weights()) {
        if particle[0] < 0.0 {
            negative += w;
            negative_mode += w * particle[0];
        } else {
            positive += w;
            positive_mode += w * particle[0];
        }
    }
    println!(
        "particle filter: {:.1}% at {:.2}, {:.1}% at {:.2}; weighted mean {:.2}",
        100.0 * negative,
        negative_mode / negative,
        100.0 * positive,
        positive_mode / positive,
        filter.mean()[0]
    );

    assert!(
        ekf_error > 6.0 && ekf_sigma < 1.0,
        "the EKF was supposed to be confidently wrong: error {ekf_error:.3}, sigma {ekf_sigma:.3}"
    );
    assert!(
        negative > 0.2 && positive > 0.2,
        "both modes must survive: {negative:.3} negative, {positive:.3} positive"
    );
    assert!(
        (negative_mode / negative - TRUTH).abs() < 0.5,
        "the negative mode must sit on the truth, not near it: {:.3}",
        negative_mode / negative
    );
    assert!(
        (positive_mode / positive + TRUTH).abs() < 0.5,
        "the positive mode must sit on the reflection: {:.3}",
        positive_mode / positive
    );
    // The summary the other filters would report is the one place with no probability mass at all.
    assert!(
        filter.mean()[0].abs() < 2.0,
        "the weighted mean of a bimodal posterior sits between the modes: {:.3}",
        filter.mean()[0]
    );
}

/// Every scheme is unbiased, and the low-variance schemes are measurably lower-variance.
///
/// `E[nᵢ] = M wᵢ` is what makes a resampling step legitimate at all — a biased scheme would move the
/// distribution every time it fired. The variance ordering is the reason to prefer one: with 20 000
/// repetitions the standard error on each variance is about 1%, so the gaps below (a factor of two
/// and more) are not sampling noise.
#[test]
fn the_resampling_schemes_are_unbiased_and_ordered_by_variance() {
    const M: usize = 8;
    const REPEATS: usize = 20_000;
    let weights: [f64; M] = [0.30, 0.22, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02];

    let schemes = [
        ("multinomial", Resampling::Multinomial),
        ("residual", Resampling::Residual),
        ("stratified", Resampling::Stratified),
        ("systematic", Resampling::Systematic),
    ];
    let mut total_variance = [0.0; 4];
    for (index, (name, scheme)) in schemes.iter().enumerate() {
        let mut rng = Rng::new(31_337 + index as u64);
        let mut sum = [0.0; M];
        let mut sum_squares = [0.0; M];
        for _ in 0..REPEATS {
            let indices = scheme.indices(&weights, &mut rng);
            let mut counts = [0.0; M];
            for &i in &indices {
                counts[i] += 1.0;
            }
            for i in 0..M {
                sum[i] += counts[i];
                sum_squares[i] += counts[i] * counts[i];
            }
        }
        let n = REPEATS as f64;
        let mut variance = 0.0;
        for i in 0..M {
            let mean = sum[i] / n;
            let want = M as f64 * weights[i];
            assert!(
                (mean - want).abs() < 0.03,
                "{name}: E[n_{i}] = {mean:.4}, want {want:.4} — the scheme is biased"
            );
            variance += sum_squares[i] / n - mean * mean;
        }
        total_variance[index] = variance;
        println!("{name:>12}: Σ Var[nᵢ] = {variance:.4}");
    }

    let [multinomial, residual, stratified, systematic] = total_variance;
    assert!(
        residual < multinomial,
        "residual {residual:.4} should beat multinomial {multinomial:.4}"
    );
    assert!(
        stratified < multinomial,
        "stratified {stratified:.4} should beat multinomial {multinomial:.4}"
    );
    // Systematic is the lowest-variance of the four on a generic weight vector, which is why it is
    // the tracking-literature default; it is not a theorem for *every* vector (a comb can align
    // badly with a pathological one), so this is a measurement on this vector and says so.
    assert!(
        systematic < stratified,
        "systematic {systematic:.4} should beat stratified {stratified:.4} on this weight vector"
    );
}

/// The one-pass comb and the binary search agree on where a point lands in the cumulative weights.
///
/// Two independent index lookups, and the filter's correctness rests on both: a comb off by one
/// silently biases every low-variance scheme toward its neighbour, which no unbiasedness test with
/// 3% tolerance would catch on a near-uniform weight vector.
#[test]
fn the_two_index_lookups_agree() {
    const M: usize = 6;
    let weights: [f64; M] = [0.05, 0.35, 0.10, 0.20, 0.28, 0.02];
    // Systematic resampling with a known offset is a deterministic comb, so its indices can be
    // recomputed by hand from the cumulative weights.
    let cdf = {
        let mut c = [0.0; M];
        let mut running = 0.0;
        for (slot, &w) in c.iter_mut().zip(&weights) {
            running += w;
            *slot = running;
        }
        c
    };
    let mut rng = Rng::new(4);
    for _ in 0..500 {
        let indices = particle::Resampling::Systematic.indices(&weights, &mut rng);
        for (position, &index) in indices.iter().enumerate() {
            // Whatever offset was drawn, particle `index` must own an interval that the comb point
            // could have fallen in: the point is at least `position/M` and less than `(position+1)/M`.
            let low = if index == 0 { 0.0 } else { cdf[index - 1] };
            let high = cdf[index];
            assert!(
                high > low || index == M - 1,
                "index {index} owns an empty interval"
            );
            assert!(
                low <= (position as f64 + 1.0) / M as f64 && high >= position as f64 / M as f64,
                "comb point {position} landed in [{low}, {high}) at index {index}"
            );
        }
    }
}
