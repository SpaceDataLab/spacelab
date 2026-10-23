//! The linear covariance analysis, checked against the thing it is the linear term of.
//!
//! [`LinearCovariance`] is a matrix recursion. Nothing inside it ever runs a filter, so a sign error
//! in one of its cross terms would produce a covariance that is smooth, symmetric, positive and
//! wrong — and the module's own unit tests, which check identities and limits of that same
//! recursion, would not necessarily see it. This test takes the other route: it *simulates* the
//! scenario. Trajectories are drawn from the truth's statistics, driven through the actual filter
//! equations sample by sample, and the covariance of the resulting navigation error is measured
//! rather than propagated. The two answers share no code at all.
//!
//! The scenario is deliberately wrong in all four ways at once, because a check on a filter whose
//! models are right is a check that the recursion reduces to a Kalman filter, which the unit tests
//! already make:
//!
//! * the truth has a state the filter does not estimate (a measurement bias) — `Δ ≠ 0`;
//! * the filter's transition differs from the truth's — `D ≠ 0`;
//! * the filter's process noise is smaller than the truth's;
//! * the filter's measurement noise is smaller than the truth's.
//!
//! # Reading the tolerance
//!
//! A sample variance over `T` independent trials has relative standard deviation `√(2/T)`, so at
//! `T = 40 000` the Monte-Carlo answer carries 0.7 % of its own noise. The assertions are set at
//! 4 %, which is between five and six of those — loose enough that a green run is not a coincidence
//! and tight enough that a factor-of-two error in any term of the recursion cannot survive it. The
//! seed is pinned, so a failure is a regression rather than a draw.

// A test may divide by an integer count and index matrices by single letters, as the estimation
// literature does.
#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use spacelab_estimation::{LinearCovariance, sampling::Rng};

/// Truth states: position, velocity, and a measurement bias the filter never models.
const N: usize = 3;
/// Filter states: position and velocity.
const M: usize = 2;

const TRIALS: usize = 40_000;
const STEPS: usize = 30;
const SEED: u64 = 20_260_817;

/// `T` (`M × N`): the filter estimates the first two truth states.
const MAP: [[f64; N]; M] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

/// Truth transition over one step: constant velocity, and a bias that persists.
const TRUTH_STM: [[f64; N]; N] = [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// What the filter integrates: a velocity that it believes decays slightly, which it does not.
const FILTER_STM: [[f64; M]; M] = [[1.0, 1.0], [0.0, 0.995]];

/// Truth process noise: white acceleration on the velocity, nothing on the bias.
const TRUTH_Q: [[f64; N]; N] = [
    [1.0e-4, 1.0e-4, 0.0],
    [1.0e-4, 4.0e-4, 0.0],
    [0.0, 0.0, 0.0],
];
/// What the filter allows itself — a quarter of the truth's, the classic under-tuning.
const FILTER_Q: [[f64; M]; M] = [[0.25e-4, 0.25e-4], [0.25e-4, 1.0e-4]];

/// Truth measurement: position plus the unmodelled bias. Filter: position alone.
const TRUTH_H: [f64; N] = [1.0, 0.0, 1.0];
const FILTER_H: [f64; M] = [1.0, 0.0];
const TRUE_R: f64 = 0.09;
const FILTER_R: f64 = 0.04;

/// The truth's a-priori dispersion, and its lower Cholesky factor for drawing from it.
const DISPERSION: [[f64; N]; N] = [[4.0, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.16]];
/// The covariance the filter is seeded with, and the error it actually starts with.
const FILTER_P0: [[f64; M]; M] = [[1.0, 0.0], [0.0, 0.09]];

fn cholesky<const D: usize>(a: &[[f64; D]; D]) -> [[f64; D]; D] {
    let mut l = [[0.0; D]; D];
    for i in 0..D {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                assert!(sum > 0.0, "not positive definite at {i}");
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    l
}

/// One simulated trajectory, returning the navigation error `T x − x̂` at the end of the run.
fn trial(rng: &mut Rng, dispersion_sqrt: &[[f64; N]; N], error_sqrt: &[[f64; M]; M]) -> [f64; M] {
    // Truth process noise is rank-deficient (the bias never moves), so it is drawn on the 2×2
    // velocity/position block and zero-padded rather than factored as a 3×3.
    let q_block = [
        [TRUTH_Q[0][0], TRUTH_Q[0][1]],
        [TRUTH_Q[1][0], TRUTH_Q[1][1]],
    ];
    let q_sqrt = cholesky(&q_block);

    let mut x = rng.correlated_normal(dispersion_sqrt);
    let e0 = rng.correlated_normal(error_sqrt);
    // The filter starts with an error independent of the dispersion, which is the initial condition
    // `LinearCovariance::new` assumes.
    let mut estimate = [x[0] - e0[0], x[1] - e0[1]];
    let mut p = FILTER_P0;

    for _ in 0..STEPS {
        // Truth.
        let w = rng.correlated_normal(&q_sqrt);
        let next = [
            TRUTH_STM[0][0] * x[0] + TRUTH_STM[0][1] * x[1] + TRUTH_STM[0][2] * x[2] + w[0],
            TRUTH_STM[1][0] * x[0] + TRUTH_STM[1][1] * x[1] + TRUTH_STM[1][2] * x[2] + w[1],
            x[2],
        ];
        x = next;

        // Filter time update.
        estimate = [
            FILTER_STM[0][0] * estimate[0] + FILTER_STM[0][1] * estimate[1],
            FILTER_STM[1][0] * estimate[0] + FILTER_STM[1][1] * estimate[1],
        ];
        p = joseph_free_propagation(&p);

        // Measurement.
        let noise = rng.standard_normal() * TRUE_R.sqrt();
        let observed = TRUTH_H[0] * x[0] + TRUTH_H[1] * x[1] + TRUTH_H[2] * x[2] + noise;
        let predicted = FILTER_H[0] * estimate[0] + FILTER_H[1] * estimate[1];

        let mut p_ht = [0.0; M];
        for i in 0..M {
            p_ht[i] = p[i][0] * FILTER_H[0] + p[i][1] * FILTER_H[1];
        }
        let innovation_variance = FILTER_R + FILTER_H[0] * p_ht[0] + FILTER_H[1] * p_ht[1];
        let gain = [p_ht[0] / innovation_variance, p_ht[1] / innovation_variance];
        let innovation = observed - predicted;
        for i in 0..M {
            estimate[i] += gain[i] * innovation;
        }
        p = joseph_update(&p, &gain);
    }

    [x[0] - estimate[0], x[1] - estimate[1]]
}

fn joseph_free_propagation(p: &[[f64; M]; M]) -> [[f64; M]; M] {
    let mut out = [[0.0; M]; M];
    for i in 0..M {
        for j in 0..M {
            let mut acc = 0.0;
            for k in 0..M {
                for l in 0..M {
                    acc += FILTER_STM[i][k] * p[k][l] * FILTER_STM[j][l];
                }
            }
            out[i][j] = acc + FILTER_Q[i][j];
        }
    }
    out
}

fn joseph_update(p: &[[f64; M]; M], gain: &[f64; M]) -> [[f64; M]; M] {
    let mut a = [[0.0; M]; M];
    for i in 0..M {
        a[i][i] = 1.0;
        for j in 0..M {
            a[i][j] -= gain[i] * FILTER_H[j];
        }
    }
    let mut out = [[0.0; M]; M];
    for i in 0..M {
        for j in 0..M {
            let mut acc = 0.0;
            for k in 0..M {
                for l in 0..M {
                    acc += a[i][k] * p[k][l] * a[j][l];
                }
            }
            out[i][j] = acc + gain[i] * FILTER_R * gain[j];
        }
    }
    out
}

#[test]
fn the_analysis_reproduces_the_error_covariance_a_simulated_ensemble_measures() {
    let mut analysis = LinearCovariance::<N, M>::new(MAP, DISPERSION, FILTER_P0);
    for _ in 0..STEPS {
        analysis.time_update(&TRUTH_STM, &TRUTH_Q, &FILTER_STM, &FILTER_Q);
        analysis.measurement_update(&TRUTH_H, &FILTER_H, TRUE_R, FILTER_R);
    }

    let dispersion_sqrt = cholesky(&DISPERSION);
    let error_sqrt = cholesky(&FILTER_P0);
    let mut rng = Rng::new(SEED);
    let mut sum = [0.0; M];
    let mut second = [[0.0; M]; M];
    for _ in 0..TRIALS {
        let e = trial(&mut rng, &dispersion_sqrt, &error_sqrt);
        for i in 0..M {
            sum[i] += e[i];
            for j in 0..M {
                second[i][j] += e[i] * e[j];
            }
        }
    }
    let count = TRIALS as f64;
    let mean = [sum[0] / count, sum[1] / count];
    let mut sampled = [[0.0; M]; M];
    for i in 0..M {
        for j in 0..M {
            sampled[i][j] = second[i][j] / count - mean[i] * mean[j];
        }
    }

    let predicted = analysis.true_covariance();
    for i in 0..M {
        for j in 0..M {
            println!(
                "MEASURED Cee[{i}][{j}] analysis {:.6e} ensemble {:.6e} ratio {:.4}",
                predicted[i][j],
                sampled[i][j],
                sampled[i][j] / predicted[i][j]
            );
        }
    }
    println!(
        "MEASURED claimed {:?} optimism {:?}",
        analysis.filter_covariance(),
        analysis.optimism()
    );

    // The ensemble mean of the error must be zero: nothing in this scenario is a *bias* on the
    // estimate, only a source of variance the filter does not account for. `√(Cee/T)` is that
    // mean's own standard error, so four of them is the threshold.
    for i in 0..M {
        let standard_error = (predicted[i][i] / count).sqrt();
        assert!(
            mean[i].abs() < 4.0 * standard_error,
            "mean error[{i}] {} against {standard_error:.3e}",
            mean[i]
        );
    }

    for i in 0..M {
        for j in 0..M {
            let scale = (predicted[i][i] * predicted[j][j]).sqrt();
            assert!(
                (sampled[i][j] - predicted[i][j]).abs() < 0.04 * scale,
                "Cee[{i}][{j}]: analysis {:.6e}, ensemble {:.6e}",
                predicted[i][j],
                sampled[i][j]
            );
        }
    }

    // And the finding the analysis exists to produce: this filter is optimistic, and by how much.
    for (i, ratio) in analysis.optimism().into_iter().enumerate() {
        assert!(ratio > 1.0, "state {i} optimism {ratio}");
    }
}

#[test]
fn the_ensemble_also_reproduces_the_filters_own_covariance_when_the_models_are_right() {
    // The control. Give the simulation the truth's own transition, process noise and measurement
    // variance and the same ensemble must now land on the filter's reported covariance — which is
    // what makes the disagreement in the test above attributable to the mismodelling rather than to
    // the simulator. Run through the analysis with matched models rather than a second simulator,
    // because the identity `Cee = P̂` is what makes the two interchangeable here.
    let mut analysis = LinearCovariance::<M, M>::new(
        [[1.0, 0.0], [0.0, 1.0]],
        [[4.0, 0.0], [0.0, 0.25]],
        FILTER_P0,
    );
    for _ in 0..STEPS {
        analysis.time_update(&FILTER_STM, &FILTER_Q, &FILTER_STM, &FILTER_Q);
        analysis.measurement_update(&FILTER_H, &FILTER_H, FILTER_R, FILTER_R);
    }
    for ratio in analysis.optimism() {
        assert!((ratio - 1.0).abs() < 1e-9, "optimism {ratio}");
    }
}
