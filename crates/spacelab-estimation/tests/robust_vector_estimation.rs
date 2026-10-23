//! What robustness buys a *vector*-measurement filter, measured on a star-tracker arc with wrong-star
//! matches — and the parameter that decides which policy wins, which neither this file nor
//! `robust_estimation.rs` could have found alone.
//!
//! # The experiment
//!
//! A body slewing at a constant `[2.0, −1.0, 1.5] mrad/s` with a constant gyro bias, observed once a
//! second by a star tracker reporting two non-parallel catalogue directions in the body frame at
//! `σ = 1e-4 rad` (20.6″) per axis. The [`MultiplicativeEkf`] is correctly specified by construction:
//! the truth is propagated by the same rotation the filter integrates, and the gyro carries exactly the
//! `σ_v` rate noise and `σ_u` bias walk the filter's `Q` assumes.
//!
//! A **wrong-star match** rotates the measured direction by a fixed angle about a random axis before
//! the noise is added — a real star, correctly measured, attributed to the wrong catalogue entry. Its
//! magnitude is a property of the catalogue rather than of the sensor, which is why it is an angle and
//! not a multiple of `σ`. At the headline 5° it is **870σ**; the control below repeats the whole thing
//! at 3′, which is 8.7σ.
//!
//! Every estimator sees byte-identical data: the random stream is consumed by the truth, the gyro, the
//! measurement noise and the outlier draw, never by the filter.
//!
//! # Measured — 40 runs × 300 steps × 2 stars = 24 000 vector updates per cell
//!
//! Attitude RMS in arcseconds, NEES against six degrees of freedom (consistent ≈ 6), and the longest
//! unbroken run of rejections within one arc on one star:
//!
//! ```text
//!                     0 %                  1 %                   5 %                  20 %
//!                 RMS  NEES  run     RMS    NEES  run     RMS    NEES  run      RMS    NEES  run
//!   plain        26.3   6.2    0   1186.6  6343.8   0   2523.8 28333.0   0   5170.3 118004.8   0
//!   gate 3σ      26.5   6.2    2     27.3     5.9   2     28.6     6.1   3     53.3      5.9   5
//!   inflate 3σ   26.3   6.2    0     60.2     5.9   0     60.8     6.1   0    612.3     66.7   0
//!   Huber        26.4   6.2    0    130.5     6.8   0    144.8     9.5   0    660.0    229.4   0
//!   Student-t    26.9   6.5    0     42.2     6.2   0     43.8     6.3   0    507.5     28.4   0
//! ```
//!
//! Clean-data efficiency loss in RMS: **0.76 % for the gate, 0.00 % for inflation, 0.38 % for Huber,
//! 2.28 % for the Student-t** — smaller than the scalar arc's 1.4–2.4 % for three of the four, because
//! `δ²` is χ²(3) and its relative spread is narrower, so a threshold calibrated on the tail clips less
//! of the bulk.
//!
//! # The influence function predicts this ranking exactly, and predicted the scalar one backwards
//!
//! `robust.rs` records that the scalar table *inverts* the influence-function ordering: the gate has
//! the best influence function and was the worst estimator, at 932 m against 4.4 m. Here the ordering
//! is obeyed to the letter — gate (zero influence) 53.3″, Student-t and inflation (redescending, so the
//! increment falls as `1/|ν|`) 507″ and 612″, Huber (bounded but non-zero in the one-step form, since
//! `S_eff ∝ |ν|` makes `Kν` tend to a constant) 660″, plain (linear, unbounded) 5170″.
//!
//! The two results are not in conflict, and the thing that reconciles them is **whether the filter can
//! recover from a bad update**:
//!
//! * The scalar lockout needed a biased state the gate then protects. The only thing that could widen
//!   that covariance again was `Q` per step, and it was far too small.
//! * Here `σ_v² dt = 1e-8 rad²` per second *is* the steady-state variance, so a filter that has just
//!   taken a bad update is back to a gate several σ wide within a few steps. The longest rejection run
//!   at 20 % contamination is 5, against 299 there.
//!
//! **So the influence function ranks the policies correctly exactly when the process noise can undo a
//! mistake, and backwards when it cannot.** Neither file could have found that; it needs both, and it
//! is the thing to check before choosing a policy for a new filter — not the measurement dimension, and
//! not the estimator.
//!
//! # Two controls, because a single arc cannot separate three explanations
//!
//! [`the_ranking_inverts_when_the_mismatch_is_a_near_neighbour_rather_than_a_gross_error`] repeats the
//! 20 % cell at 3′ (8.7σ) instead of 5° (870σ), changing nothing else: **plain 57.6″, gate 767.9″,
//! inflate 49.4″, Huber 46.4″, Student-t 41.2″**. The gate is now 13× *worse* than doing nothing, and
//! Huber and the Student-t are the best two — the scalar file's conclusion, reproduced on this filter by
//! moving one parameter. An outlier the gate cannot cleanly separate from the bulk is one it starves the
//! filter to reject.
//!
//! [`the_gate_locks_out_on_the_linearisation_transient_with_no_outliers_at_all`] starts the same arc
//! from a coarse 2.9° attitude with **no contamination whatsoever**: plain settles at 37.3″, the gate at
//! **16 368″**, rejecting 62.7 % of all observations with a run covering an entire 300-step arc. A MEKF
//! started far from truth is transiently overconfident — `Φ = I + F dt` under-predicts how fast a large
//! error rotates, so the covariance contracts before the error does, and measured NEES over the first 30
//! steps is 61 against 6. The gate reads that as contamination and locks out.
//!
//! That is a sharper statement than #277's, and the one to carry into an operations manual: **the gate
//! does not need an outlier to fail. It needs only a stretch in which the filter's own covariance is
//! wrong, and acquisition is exactly that.** Unlike the scalar case, NEES does flag this one — 28.3
//! against 11.6 — but the rejection run flags it first and unambiguously.

// Counts turned into exact fractions are compared against the literals they were built from.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::similar_names
)]

use core::fmt::Write as _;
use spacelab_estimation::{
    InvariantEkf, MultiplicativeEkf, NavigationEkf, RobustPolicy, RobustTally,
    SquareRootUnscentedKalman, UnitQuaternion, UnscentedParams, chi_square_gate,
    normalized_squared, process_noise_sqrt,
};

const STEP: f64 = 1.0;
const STEPS: usize = 300;
const RUNS: usize = 40;
/// Per-axis star-tracker noise on the measured unit vector (rad). 1e-4 ≈ 20.6″.
const STAR_SIGMA: f64 = 1e-4;
/// Gyro rate white noise (rad·s^−1ᐟ²) and bias random walk (rad·s^−3ᐟ²).
const GYRO_SIGMA_V: f64 = 1e-4;
const GYRO_SIGMA_U: f64 = 1e-7;
/// A wrong-star match displaces the measured direction by the angular distance to the wrong star.
const MISMATCH_ANGLE: f64 = 5.0 * core::f64::consts::PI / 180.0;
const CONTAMINATION: [f64; 4] = [0.0, 0.01, 0.05, 0.20];
const TUNING: f64 = 1.345;
const STUDENT_DOF: f64 = 5.0;
const ARCSEC: f64 = core::f64::consts::PI / 180.0 / 3600.0;

const OMEGA: [f64; 3] = [2e-3, -1e-3, 1.5e-3];
const BIAS: [f64; 3] = [2e-5, -3e-5, 1e-5];
/// Initial attitude uncertainty (rad) and gyro-bias uncertainty (rad/s), post-acquisition: 5 mrad is
/// 0.29°, small enough that the MEKF's `Φ = I + F dt` is a fair model of the error dynamics from the
/// first step. [`ACQUISITION_SIGMA`] is the coarse-acquisition value the transient test uses instead.
const ATTITUDE_SIGMA_0: f64 = 5e-3;
const BIAS_SIGMA_0: f64 = 1e-4;
/// A coarse initial attitude — sun sensor and magnetometer, before the tracker has a solution.
const ACQUISITION_SIGMA: f64 = 0.05;

/// `SplitMix64` with a Box–Muller transform — the same deterministic source as
/// `robust_estimation.rs`, so the two tables are drawn from streams of the same quality.
struct Gaussian {
    state: u64,
    spare: Option<f64>,
}

impl Gaussian {
    const fn seeded(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let radius = libm::sqrt(-2.0 * libm::log(u1));
        let angle = core::f64::consts::TAU * u2;
        self.spare = Some(radius * libm::sin(angle));
        radius * libm::cos(angle)
    }

    /// A direction drawn uniformly on the sphere, by normalising three independent normals.
    fn direction(&mut self) -> [f64; 3] {
        loop {
            let v = [self.normal(), self.normal(), self.normal()];
            let n = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if n > 1e-6 {
                return [v[0] / n, v[1] / n, v[2] / n];
            }
        }
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn references() -> [[f64; 3]; 2] {
    [unit([1.0, 0.3, -0.2]), unit([-0.2, 1.0, 0.4])]
}

struct Estimator {
    name: &'static str,
    policy: RobustPolicy,
}

fn estimators() -> [Estimator; 5] {
    let gate = chi_square_gate(3, 3.0);
    [
        Estimator {
            name: "plain",
            policy: RobustPolicy::Plain,
        },
        Estimator {
            name: "gate 3σ",
            policy: RobustPolicy::ChiSquareGate { threshold: gate },
        },
        Estimator {
            name: "inflate 3σ",
            policy: RobustPolicy::ChiSquareInflation { threshold: gate },
        },
        Estimator {
            name: "Huber",
            policy: RobustPolicy::Huber { tuning: TUNING },
        },
        Estimator {
            name: "Student-t",
            policy: RobustPolicy::StudentT {
                degrees_of_freedom: STUDENT_DOF,
            },
        },
    ]
}

struct Report {
    /// Attitude RMS in arcseconds.
    rms: f64,
    /// Mean NEES over the six-state error `[δθ, δb]`.
    nees: f64,
    rejected: f64,
    longest_rejection_run: u32,
}

/// The six-state MEKF error `[δθ, δb]` against truth. `δθ` is the body-frame multiplicative error the
/// filter's covariance is defined on — `q_true = q̂ ⊗ exp(δθ)` — so the same rotation the reset applies.
fn mekf_error(filter: &MultiplicativeEkf, q_true: UnitQuaternion, bias: [f64; 3]) -> [f64; 6] {
    let dtheta = filter
        .attitude()
        .conjugate()
        .mul(q_true)
        .to_rotation_vector();
    [
        dtheta[0],
        dtheta[1],
        dtheta[2],
        bias[0] - filter.bias()[0],
        bias[1] - filter.bias()[1],
        bias[2] - filter.bias()[2],
    ]
}

fn rotate_by(v: [f64; 3], axis: [f64; 3], angle: f64) -> [f64; 3] {
    UnitQuaternion::from_rotation_vector([axis[0] * angle, axis[1] * angle, axis[2] * angle])
        .rotate(v)
}

fn run(policy: RobustPolicy, contamination: f64, seed: u64) -> Report {
    run_with(
        policy,
        contamination,
        MISMATCH_ANGLE,
        ATTITUDE_SIGMA_0,
        seed,
    )
}

/// The arc, with the wrong-star displacement and the initial attitude uncertainty as parameters. The
/// headline table fixes both; the two control tests move one each, which is what makes them controls.
fn run_with(
    policy: RobustPolicy,
    contamination: f64,
    mismatch: f64,
    attitude_sigma: f64,
    seed: u64,
) -> Report {
    let refs = references();
    let mut source = Gaussian::seeded(seed);
    // One tally per star *per run*: the run counter is a statement about one filter's arc, and letting
    // it span a fresh filter would report a lockout that never happened.
    let mut seen = 0u32;
    let mut rejected = 0u32;
    let mut longest = 0u32;
    let (mut squared_error, mut nees_total, mut samples) = (0.0, 0.0, 0usize);

    let mut initial = [[0.0; 6]; 6];
    for axis in 0..3 {
        initial[axis][axis] = attitude_sigma * attitude_sigma;
        initial[axis + 3][axis + 3] = BIAS_SIGMA_0 * BIAS_SIGMA_0;
    }

    for _ in 0..RUNS {
        let mut tally = [RobustTally::default(); 2];
        let mut q_true = UnitQuaternion::identity();
        let offset: [f64; 3] = core::array::from_fn(|_| attitude_sigma * source.normal());
        let mut filter = MultiplicativeEkf::new(
            q_true.mul(UnitQuaternion::from_rotation_vector(offset)),
            core::array::from_fn(|i| BIAS[i] + BIAS_SIGMA_0 * source.normal()),
            initial,
        );

        for _ in 0..STEPS {
            // Truth: the exact rotation the filter integrates, so the model is correctly specified.
            q_true = q_true
                .mul(UnitQuaternion::from_rotation_vector([
                    OMEGA[0] * STEP,
                    OMEGA[1] * STEP,
                    OMEGA[2] * STEP,
                ]))
                .normalized();

            // gyro = ω + bias + white noise, at the σ_v the filter's Q assumes.
            let gyro: [f64; 3] = core::array::from_fn(|i| {
                OMEGA[i] + BIAS[i] + GYRO_SIGMA_V / libm::sqrt(STEP) * source.normal()
            });
            filter.predict(gyro, STEP, GYRO_SIGMA_V, GYRO_SIGMA_U);

            for (star, reference) in refs.iter().enumerate() {
                let mut body = q_true.rotate(*reference);
                if source.uniform() < contamination {
                    // A wrong-star match: a real direction, attributed to the wrong catalogue entry.
                    let axis = source.direction();
                    body = rotate_by(body, axis, mismatch);
                }
                let measured: [f64; 3] =
                    core::array::from_fn(|i| body[i] + STAR_SIGMA * source.normal());
                let outcome = filter
                    .update_robust(measured, *reference, STAR_SIGMA, policy)
                    .expect("two non-parallel references keep the innovation covariance regular");
                tally[star].record(&outcome);
            }

            let error = mekf_error(&filter, q_true, BIAS);
            squared_error += error[0] * error[0] + error[1] * error[1] + error[2] * error[2];
            let sqrt = process_noise_sqrt(&filter.covariance())
                .expect("the Joseph update keeps the covariance positive-definite");
            nees_total += normalized_squared(&sqrt, &error);
            samples += 1;
        }

        for t in &tally {
            seen += t.seen;
            rejected += t.rejected;
            longest = longest.max(t.longest_rejection_run);
        }
    }

    let n = samples as f64;
    Report {
        rms: libm::sqrt(squared_error / n) / ARCSEC,
        nees: nees_total / n,
        rejected: f64::from(rejected) / f64::from(seen),
        longest_rejection_run: longest,
    }
}

/// The headline table: cost on clean data, benefit on dirty data, for all five policies, at the
/// gross-outlier magnitude a 5° star confusion produces.
#[test]
fn the_gate_wins_on_gross_outliers_where_the_scalar_arc_said_it_would_lose() {
    let names = estimators();
    let mut rms = [[0.0; CONTAMINATION.len()]; 5];
    let mut nees = [[0.0; CONTAMINATION.len()]; 5];
    let mut runs = [[0u32; CONTAMINATION.len()]; 5];

    for (c, &contamination) in CONTAMINATION.iter().enumerate() {
        // Seeded per contamination level, not per estimator: a paired comparison.
        let seed = 0x5EED_0000 + c as u64;
        for (e, estimator) in names.iter().enumerate() {
            let report = run(estimator.policy, contamination, seed);
            rms[e][c] = report.rms;
            nees[e][c] = report.nees;
            runs[e][c] = report.longest_rejection_run;
        }
    }

    let mut table = String::new();
    writeln!(
        table,
        "attitude RMS (arcsec) / NEES(6) / longest rejection run"
    )
    .unwrap();
    for (e, estimator) in names.iter().enumerate() {
        write!(table, "  {:<12}", estimator.name).unwrap();
        for c in 0..CONTAMINATION.len() {
            write!(
                table,
                " {:7.1} {:6.1} {:4}",
                rms[e][c], nees[e][c], runs[e][c]
            )
            .unwrap();
        }
        table.push('\n');
    }
    std::eprint!("{table}");

    // 1. The plain filter degrades badly, and confidently: NEES grows faster than RMS².
    assert!(
        rms[0][3] > 10.0 * rms[0][0],
        "plain 20% RMS {:.1}″ against clean {:.1}″",
        rms[0][3],
        rms[0][0]
    );
    assert!(nees[0][3] > 100.0, "plain 20% NEES {:.1}", nees[0][3]);

    // 2. Every robust policy beats it at every non-zero contamination level.
    for e in 1..5 {
        for c in 1..CONTAMINATION.len() {
            assert!(
                rms[e][c] < rms[0][c],
                "{} at {:.0}%: {:.1}″ against plain {:.1}″",
                names[e].name,
                CONTAMINATION[c] * 100.0,
                rms[e][c],
                rms[0][c]
            );
        }
    }

    // 3. The gate does *not* lock out here — the finding this test was written to settle. In the
    //    scalar arc it was the worst of the five by two orders of magnitude; here it is the best, and
    //    the longest rejection run stays in single digits rather than reaching the length of the arc.
    for c in 0..CONTAMINATION.len() {
        assert!(
            runs[1][c] < 20,
            "gate rejection run {} at {:.0}% — the scalar lockout has reproduced after all",
            runs[1][c],
            CONTAMINATION[c] * 100.0
        );
        assert!(
            rms[1][c] <= rms[0][c] * 1.02,
            "gate {:.1}″ against plain {:.1}″ at {:.0}%",
            rms[1][c],
            rms[0][c],
            CONTAMINATION[c] * 100.0
        );
    }

    // 4. Clean-data efficiency: the price is real and larger than the scalar case's, because δ² is
    //    χ²(3) and the discount lands on the whole observation rather than on one axis.
    for e in 1..5 {
        let loss = rms[e][0] / rms[0][0] - 1.0;
        assert!(
            (0.0..0.15).contains(&loss),
            "{} clean-data loss {:.1}%",
            names[e].name,
            loss * 100.0
        );
    }
}

/// The invariant filter's Jacobian `H = [r]×` does not depend on the estimate, so `δ²` measures the
/// *observation* rather than how far the filter has drifted. That is the property the scalar lockout
/// exploited in reverse, and it should make the invariant filter's gate at least as safe as the MEKF's.
#[test]
fn the_invariant_filter_gates_on_the_observation_rather_than_on_its_own_drift() {
    let refs = references();
    let gate = chi_square_gate(3, 3.0);
    let mut source = Gaussian::seeded(0x1E7F_0001);
    let mut tally = [RobustTally::default(); 2];

    let mut q_true = UnitQuaternion::identity();
    let offset = [4e-3, -3e-3, 5e-3];
    let mut filter = InvariantEkf::new(
        q_true.mul(UnitQuaternion::from_rotation_vector(offset)),
        [
            [ATTITUDE_SIGMA_0 * ATTITUDE_SIGMA_0, 0.0, 0.0],
            [0.0, ATTITUDE_SIGMA_0 * ATTITUDE_SIGMA_0, 0.0],
            [0.0, 0.0, ATTITUDE_SIGMA_0 * ATTITUDE_SIGMA_0],
        ],
    );

    let mut rejected_outliers = 0u32;
    let mut accepted_outliers = 0u32;
    for _ in 0..STEPS {
        q_true = q_true
            .mul(UnitQuaternion::from_rotation_vector([
                OMEGA[0] * STEP,
                OMEGA[1] * STEP,
                OMEGA[2] * STEP,
            ]))
            .normalized();
        filter.predict(OMEGA, STEP, GYRO_SIGMA_V);

        for (star, reference) in refs.iter().enumerate() {
            let mut body = q_true.rotate(*reference);
            let corrupted = source.uniform() < 0.05;
            if corrupted {
                body = rotate_by(body, source.direction(), MISMATCH_ANGLE);
            }
            let measured: [f64; 3] =
                core::array::from_fn(|i| body[i] + STAR_SIGMA * source.normal());
            let outcome = filter
                .update_robust(
                    measured,
                    *reference,
                    STAR_SIGMA,
                    RobustPolicy::ChiSquareGate { threshold: gate },
                )
                .unwrap();
            if corrupted {
                if outcome.accepted {
                    accepted_outliers += 1;
                } else {
                    rejected_outliers += 1;
                }
            }
            tally[star].record(&outcome);
        }
    }

    let residual = filter
        .attitude()
        .conjugate()
        .mul(q_true)
        .to_rotation_vector();
    let error = libm::sqrt(residual.iter().map(|c| c * c).sum::<f64>());
    let longest = tally.iter().map(|t| t.longest_rejection_run).max().unwrap();
    std::eprintln!(
        "LIEKF gate at 5%: {:.1}″ attitude error, {rejected_outliers} of {} outliers rejected, \
         longest rejection run {longest}",
        error / ARCSEC,
        rejected_outliers + accepted_outliers
    );

    // A 5° mismatch is 87 mrad of innovation against a steady-state σ of order 0.1 mrad, so once the
    // filter has converged the gate cannot miss one. The ones it does admit arrive early, while the
    // 50 mrad prior is still wider than the outlier.
    assert!(
        rejected_outliers > 0 && accepted_outliers <= 2,
        "{rejected_outliers} rejected, {accepted_outliers} accepted"
    );
    assert!(longest < 20, "rejection run {longest}");
    assert!(
        error < 200.0 * ARCSEC,
        "attitude error {:.1}″",
        error / ARCSEC
    );
}

/// The `D`-dimensional path: an SR-UKF observing position with a contaminated stream. The sigma cloud
/// is propagated once and the noise factor is re-folded at the policy's scale, so the robust update
/// must agree with the plain one whenever the policy does not discount.
#[test]
fn the_unscented_filter_agrees_with_the_plain_update_when_the_policy_does_not_discount() {
    let params = UnscentedParams::gaussian();
    let mut p0 = [[0.0; 6]; 6];
    for axis in 0..3 {
        p0[axis][axis] = 100.0;
        p0[axis + 3][axis + 3] = 1.0;
    }
    let x0 = [10.0, -5.0, 2.0, 0.3, -0.1, 0.2];
    let dynamics = |x: &[f64; 6]| {
        let mut next = *x;
        for axis in 0..3 {
            next[axis] += x[axis + 3] * STEP;
        }
        next
    };
    let position = |x: &[f64; 6]| [x[0], x[1], x[2]];
    let noise_sqrt = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]];
    let mut process = [[0.0; 6]; 6];
    for i in 0..6 {
        process[i][i] = 1e-3;
    }

    let mut plain = SquareRootUnscentedKalman::<6, 13>::from_covariance(x0, &p0, params).unwrap();
    let mut huber = plain.clone();
    let mut source = Gaussian::seeded(0x5C1E_0002);
    let mut tally = RobustTally::default();
    let mut discounted = false;
    let mut truth = x0;

    for step in 0..120 {
        truth = dynamics(&truth);
        plain.predict(dynamics, &process);
        huber.predict(dynamics, &process);

        let clean: [f64; 3] = core::array::from_fn(|i| truth[i] + 3.0 * source.normal());
        // One gross outlier, at a step where the covariance has already contracted.
        let z: [f64; 3] = if step == 80 {
            [clean[0] + 300.0, clean[1], clean[2]]
        } else {
            clean
        };

        plain.update(position, &z, &noise_sqrt);
        let outcome = huber.update_robust(
            position,
            &z,
            &noise_sqrt,
            RobustPolicy::Huber { tuning: TUNING },
        );
        tally.record(&outcome);
        // Once one observation has been discounted the two filters are different filters; the
        // equality only means anything while the robust path has taken every decision the plain one did.
        if outcome.weight < 1.0 {
            discounted = true;
        }

        if outcome.weight == 1.0 && !discounted {
            // Undiscounted: the two must be the same filter, not merely close — the robust path takes
            // the same factor object the plain one built.
            for a in 0..6 {
                assert!(
                    (plain.x[a] - huber.x[a]).abs() < 1e-12,
                    "step {step} state {a}: {} against {}",
                    plain.x[a],
                    huber.x[a]
                );
            }
        }
    }

    let plain_miss = libm::sqrt((0..3).map(|i| (plain.x[i] - truth[i]).powi(2)).sum::<f64>());
    let huber_miss = libm::sqrt((0..3).map(|i| (huber.x[i] - truth[i]).powi(2)).sum::<f64>());
    std::eprintln!(
        "SR-UKF after a 100σ spike: plain {plain_miss:.2} m, Huber {huber_miss:.2} m, \
         mean weight {:.3}",
        tally.mean_weight()
    );
    assert!(
        huber_miss < 0.5 * plain_miss,
        "Huber {huber_miss:.2} m against plain {plain_miss:.2} m"
    );
    assert!(tally.downweighted > 0, "nothing was ever down-weighted");
}

/// The SE(2)3 navigation filter takes three measurement streams, and the ticket's rule is one tally
/// each. This checks the plumbing on the position stream: a rejected fix leaves the state bit-identical,
/// which is the property "unchanged" has to mean for a filter whose reset is a group composition.
#[test]
fn a_rejected_position_fix_leaves_the_navigation_state_bit_identical() {
    let mut p0 = [[0.0; 15]; 15];
    for i in 0..15 {
        p0[i][i] = if i < 3 { 1e-4 } else { 1e-2 };
    }
    let mut filter = NavigationEkf::new(
        spacelab_estimation::Se23::identity(),
        [0.0; 3],
        [0.0; 3],
        p0,
        [0.0, 0.0, -9.80665],
    );

    let gate = chi_square_gate(3, 3.0);
    let before = filter.pose().position();
    let covariance_before = filter.covariance();
    let outcome = filter
        .update_position_robust(
            [500.0, 0.0, 0.0],
            1.0,
            RobustPolicy::ChiSquareGate { threshold: gate },
        )
        .unwrap();
    assert!(!outcome.accepted, "a 500σ fix passed a 3σ gate");
    assert!(outcome.nominal_normalized_squared > gate);
    assert_eq!(filter.pose().position(), before);
    assert_eq!(filter.covariance(), covariance_before);

    // And the same fix at the nominal magnitude is taken, so the gate is not simply refusing everything.
    let outcome = filter
        .update_position_robust(
            [0.5, 0.0, 0.0],
            1.0,
            RobustPolicy::ChiSquareGate { threshold: gate },
        )
        .unwrap();
    assert!(outcome.accepted);
    assert_eq!(outcome.weight, 1.0);
    assert!(filter.pose().position()[0] > 0.0);
}

/// The mechanism control. The headline table's outliers are ~870σ — a 5° star confusion against a
/// 20.6″ tracker — which is so gross that discarding is obviously right. A *near-neighbour* match at
/// 3′ is 8.7σ: the regime where discarding costs information, and where the scalar file's ranking
/// should reappear. Same arc, same seed, same 20 % rate; only the displacement changes.
#[test]
fn the_ranking_inverts_when_the_mismatch_is_a_near_neighbour_rather_than_a_gross_error() {
    let near = 3.0 / 60.0 * core::f64::consts::PI / 180.0;
    let names = estimators();
    let mut rms = [0.0; 5];
    for (e, estimator) in names.iter().enumerate() {
        rms[e] = run_with(estimator.policy, 0.20, near, ATTITUDE_SIGMA_0, 0x5EED_1234).rms;
    }
    std::eprintln!(
        "3′ near-neighbour at 20%: plain {:.1}″  gate {:.1}″  inflate {:.1}″  Huber {:.1}″  \
         Student-t {:.1}″",
        rms[0],
        rms[1],
        rms[2],
        rms[3],
        rms[4]
    );

    // The three continuous policies beat the plain filter, as they do in the scalar file.
    for e in [2, 3, 4] {
        assert!(
            rms[e] < rms[0],
            "{} {:.1}″ against plain {:.1}″",
            names[e].name,
            rms[e],
            rms[0]
        );
    }

    // And the gate is now *worse than doing nothing* — 13× the plain filter's error, on the same arc
    // whose 5° version it won by 100×. That is the whole finding: what ranks the policies is how large
    // the outlier is against the innovation it is measured with, not the filter, the measurement
    // dimension, or the policy's influence function taken on its own.
    assert!(
        rms[1] > 5.0 * rms[0],
        "gate {:.1}″ against plain {:.1}″ — the gate's failure at 8.7σ did not reproduce",
        rms[1],
        rms[0]
    );
}

/// The lockout the scalar file found does reproduce here — but on the **linearisation transient**,
/// with no outliers at all. A MEKF started from a coarse 2.9° attitude and an unknown gyro bias is
/// transiently overconfident: `Φ = I + F dt` under-predicts how fast a large error rotates, so the
/// covariance contracts before the error does, and a χ² gate reads the resulting innovations as
/// contamination. Measured NEES over the first 30 steps of that arc is 61 against 6.
///
/// That is a sharper statement than #277's. The gate does not need an outlier to lock out; it needs
/// only a stretch during which the filter's own covariance is wrong, and acquisition is exactly that.
/// **Do not gate a filter that has not converged** — the same arc from a 0.29° start (the headline
/// table's 0 % column) shows no rejection run at all.
#[test]
fn the_gate_locks_out_on_the_linearisation_transient_with_no_outliers_at_all() {
    let gate = chi_square_gate(3, 3.0);
    let plain = run_with(
        RobustPolicy::Plain,
        0.0,
        MISMATCH_ANGLE,
        ACQUISITION_SIGMA,
        0xACC0_0001,
    );
    let gated = run_with(
        RobustPolicy::ChiSquareGate { threshold: gate },
        0.0,
        MISMATCH_ANGLE,
        ACQUISITION_SIGMA,
        0xACC0_0001,
    );
    std::eprintln!(
        "acquisition from 2.9°, 0% contamination: plain {:.1}″ NEES {:.1} run {} | gate {:.1}″ \
         NEES {:.1} run {} rejecting {:.1}%",
        plain.rms,
        plain.nees,
        plain.longest_rejection_run,
        gated.rms,
        gated.nees,
        gated.longest_rejection_run,
        gated.rejected * 100.0
    );

    assert_eq!(
        plain.longest_rejection_run, 0,
        "the plain filter rejects nothing, by construction"
    );
    assert!(
        gated.longest_rejection_run > 100,
        "rejection run {} — the transient lockout did not reproduce, and the module documentation \
         above is now wrong",
        gated.longest_rejection_run
    );
    assert!(
        gated.rms > 100.0 * plain.rms,
        "gate {:.1}″ against plain {:.1}″ — a locked-out filter should be far worse",
        gated.rms,
        plain.rms
    );
}
