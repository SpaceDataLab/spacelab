//! The RTS smoother, checked against the textbook recursion it deliberately does not use, and
//! against the statistics it claims — never against itself.
//!
//! * **Cross-check**: the square-root array form and the covariance-form recursion
//!   `P̂ₖ = P⁺ₖ + Cₖ(P̂ₖ₊₁ − P⁻ₖ₊₁)Cₖᵀ` are two implementations of the same estimator. The second
//!   lives here, in the test, and only here — it is the thing being agreed with, not the thing
//!   being shipped.
//! * **Ordering**: `P⁺ₖ − P̂ₖ` is positive-semidefinite. Conditioning on the future cannot make an
//!   estimate worse, and a smoother that got a sign wrong would still return a symmetric matrix.
//! * **Consistency**: over many realisations of a linear-Gaussian system the smoothed normalised
//!   squared error averages to the state dimension. This is the check that the covariance is the
//!   *right* one and not merely a small one — a smoother that over-trusts the future passes the
//!   ordering test and fails this.
//! * **Refusal**: a singular predicted covariance is reported, not papered over.

#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::excessive_precision
)]

use proptest::prelude::*;
use spacelab_estimation::consistency::{is_consistent, normalized_squared};
use spacelab_estimation::kalman::SquareRootKalman;
use spacelab_estimation::rts::{RtsError, RtsSmoother, SmoothingStep};

const N: usize = 3;

fn matmul(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

fn transpose(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut t = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn matvec(a: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
    let mut y = [0.0; N];
    for i in 0..N {
        for j in 0..N {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

fn add(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = *a;
    for i in 0..N {
        for j in 0..N {
            c[i][j] += b[i][j];
        }
    }
    c
}

fn sub(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut c = *a;
    for i in 0..N {
        for j in 0..N {
            c[i][j] -= b[i][j];
        }
    }
    c
}

/// `A⁻¹` by Gauss–Jordan with partial pivoting. Only the textbook recursion needs an explicit
/// inverse — the shipped smoother solves a triangular system instead, which is the point.
fn inverse(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut m = *a;
    let mut inv = [[0.0; N]; N];
    for i in 0..N {
        inv[i][i] = 1.0;
    }
    for col in 0..N {
        let mut pivot = col;
        for r in (col + 1)..N {
            if m[r][col].abs() > m[pivot][col].abs() {
                pivot = r;
            }
        }
        m.swap(col, pivot);
        inv.swap(col, pivot);
        let d = m[col][col];
        assert!(d != 0.0, "singular matrix in the reference inverse");
        for j in 0..N {
            m[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..N {
            if r == col {
                continue;
            }
            let f = m[r][col];
            for j in 0..N {
                m[r][j] -= f * m[col][j];
                inv[r][j] -= f * inv[col][j];
            }
        }
    }
    inv
}

/// Positive-semidefinite to within `slack` on the diagonal, by Cholesky with a tolerated negative
/// pivot of that size. Returns the most negative pivot seen, so a caller can report it.
fn cholesky_pivot_floor(p: &[[f64; N]; N]) -> f64 {
    let mut l = [[0.0; N]; N];
    let mut worst = f64::INFINITY;
    for i in 0..N {
        for j in 0..=i {
            let mut sum = p[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                worst = worst.min(sum);
                l[i][j] = if sum > 0.0 { libm::sqrt(sum) } else { 0.0 };
            } else if l[j][j] != 0.0 {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    worst
}

/// One recorded interval plus the predicted covariance the textbook form needs but the array form
/// derives for itself.
struct Reference {
    step: SmoothingStep<N>,
    predicted_covariance: [[f64; N]; N],
}

/// The textbook covariance-form RTS backward step, for cross-checking only.
///
/// `Cₖ = P⁺ₖ Φₖᵀ (P⁻ₖ₊₁)⁻¹`, `x̂ₖ = x⁺ₖ + Cₖ(x̂ₖ₊₁ − x⁻ₖ₊₁)`,
/// `P̂ₖ = P⁺ₖ + Cₖ(P̂ₖ₊₁ − P⁻ₖ₊₁)Cₖᵀ` — including the covariance subtraction the shipped form avoids.
fn textbook_step(
    reference: &Reference,
    x_ahead: &[f64; N],
    p_ahead: &[[f64; N]; N],
) -> ([f64; N], [[f64; N]; N]) {
    let s = reference.step.filtered_sqrt;
    let p_filtered = matmul(&s, &transpose(&s));
    let gain = matmul(
        &matmul(&p_filtered, &transpose(&reference.step.stm)),
        &inverse(&reference.predicted_covariance),
    );

    let mut lookahead = [0.0; N];
    for i in 0..N {
        lookahead[i] = x_ahead[i] - reference.step.predicted[i];
    }
    let correction = matvec(&gain, &lookahead);
    let mut x = reference.step.filtered;
    for i in 0..N {
        x[i] += correction[i];
    }

    let spread = sub(p_ahead, &reference.predicted_covariance);
    let p = add(
        &p_filtered,
        &matmul(&matmul(&gain, &spread), &transpose(&gain)),
    );
    (x, p)
}

/// A deterministic standard-normal stream: xorshift64 for the uniforms, Box–Muller for the shape.
struct Normal(u64);

impl Normal {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        // Open at zero: Box-Muller takes a logarithm of this.
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn draw(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(core::f64::consts::TAU * u2)
    }
}

/// The linear-Gaussian system the tests simulate, filter and smooth.
struct System {
    stm: [[f64; N]; N],
    process_noise_sqrt: [[f64; N]; N],
    partial: [f64; N],
    noise_variance: f64,
}

/// A constant-turn-rate system with process noise on every axis: `Φ` mixes all three states so no
/// direction is trivially decoupled, and only the first is measured, so the other two are
/// observable solely through the dynamics — the regime where smoothing earns the most.
fn system() -> System {
    System {
        stm: [[0.98, 0.20, 0.0], [-0.20, 0.98, 0.05], [0.0, 0.0, 0.995]],
        process_noise_sqrt: [[0.05, 0.0, 0.0], [0.01, 0.04, 0.0], [0.0, 0.0, 0.03]],
        partial: [1.0, 0.0, 0.0],
        noise_variance: 0.25,
    }
}

/// Run the forward filter over `measurements`, recording what the backward pass needs.
fn forward(measurements: &[f64], p0: &[[f64; N]; N]) -> (Vec<Reference>, SquareRootKalman<N>) {
    let System {
        stm,
        process_noise_sqrt,
        partial,
        noise_variance,
    } = system();
    let mut kf = SquareRootKalman::<N>::from_covariance([0.0; N], p0).unwrap();
    let mut record = Vec::new();

    for &z in measurements {
        let filtered = kf.x;
        let filtered_sqrt = kf.covariance_sqrt();

        kf.x = matvec(&stm, &kf.x);
        kf.time_update(&stm, &process_noise_sqrt);
        let predicted = kf.x;
        let predicted_covariance = kf.covariance();

        let innovation = z - (0..N).map(|i| partial[i] * kf.x[i]).sum::<f64>();
        kf.measurement_update(&partial, innovation, noise_variance);

        record.push(Reference {
            step: SmoothingStep {
                filtered,
                filtered_sqrt,
                stm,
                process_noise_sqrt,
                predicted,
            },
            predicted_covariance,
        });
    }
    (record, kf)
}

#[test]
fn the_square_root_form_matches_the_textbook_covariance_recursion() {
    // Two implementations of one estimator, over a whole arc: the shipped array form and the
    // covariance-form recursion above. Worst measured departure over the 12 epochs of this arc is
    // 2.237e-15 relative on the state and 7.105e-15 absolute on the covariance; the bounds are
    // 1e-12, some two decades above.
    let measurements: Vec<f64> = (0..12).map(|k| libm::sin(0.4 * f64::from(k))).collect();
    let p0 = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
    let (record, kf) = forward(&measurements, &p0);

    let mut smoother = RtsSmoother::new(kf.x, kf.covariance_sqrt());
    let mut x_reference = kf.x;
    let mut p_reference = kf.covariance();

    for (epoch, reference) in record.iter().enumerate().rev() {
        smoother.step(&reference.step).expect("non-singular");
        let (x_want, p_want) = textbook_step(reference, &x_reference, &p_reference);
        x_reference = x_want;
        p_reference = p_want;

        let got_x = smoother.state();
        let got_p = smoother.covariance();
        let scale = (0..N).map(|i| x_want[i].abs()).fold(1.0, f64::max);
        for i in 0..N {
            assert!(
                (got_x[i] - x_want[i]).abs() / scale < 1e-12,
                "epoch {epoch} state[{i}]: {} vs {}",
                got_x[i],
                x_want[i]
            );
            for j in 0..N {
                assert!(
                    (got_p[i][j] - p_want[i][j]).abs() < 1e-12,
                    "epoch {epoch} P̂[{i}][{j}]: {} vs {}",
                    got_p[i][j],
                    p_want[i][j]
                );
            }
        }
    }
}

#[test]
fn smoothing_never_increases_the_covariance() {
    // P⁺ₖ − P̂ₖ ⪰ 0 at every epoch: the future can only inform. A smoother whose gain carried the
    // wrong sign returns a symmetric matrix of plausible size and fails exactly here.
    let measurements: Vec<f64> = (0..20).map(|k| 0.3 * f64::from(k % 5) - 0.6).collect();
    let p0 = [[9.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 9.0]];
    let (record, kf) = forward(&measurements, &p0);

    let mut smoother = RtsSmoother::new(kf.x, kf.covariance_sqrt());
    let mut shrank = false;
    for (epoch, reference) in record.iter().enumerate().rev() {
        smoother.step(&reference.step).expect("non-singular");
        let s = reference.step.filtered_sqrt;
        let filtered = matmul(&s, &transpose(&s));
        let refined = smoother.covariance();
        let difference = sub(&filtered, &refined);

        // The Cholesky pivots of P⁺ − P̂ may sit a rounding below zero on a direction the arc did not
        // shrink; anything worse is a sign error, not arithmetic.
        let floor = cholesky_pivot_floor(&difference);
        let scale = (0..N).map(|i| filtered[i][i]).fold(0.0, f64::max);
        assert!(
            floor > -1e-12 * scale,
            "epoch {epoch}: P⁺ − P̂ has pivot {floor:e} against trace scale {scale:e}"
        );
        if (0..N).any(|i| refined[i][i] < 0.9 * filtered[i][i]) {
            shrank = true;
        }
    }
    assert!(
        shrank,
        "no epoch shrank by even 10%: the smoother is returning the filter"
    );
}

#[test]
fn the_smoothed_estimate_is_consistent_over_many_realisations() {
    // The ordering test above passes for any smoother that merely shrinks the covariance. This one
    // asks whether the covariance it reports is the right one: simulate the linear-Gaussian system
    // it assumes, and the normalised squared error of the smoothed estimate must average to N.
    // With 400 runs `chi_square_mean_interval(3, 400, 4.0)` is [2.5348, 3.5152], so a covariance
    // wrong by even 20% in scale fails.
    const RUNS: usize = 400;
    const ARC: usize = 15;
    const EPOCH: usize = 7; // mid-arc, where smoothing differs most from filtering

    let System {
        stm,
        process_noise_sqrt,
        partial,
        noise_variance,
    } = system();
    let p0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut rng = Normal(0x9E37_79B9_7F4A_7C15);
    let mut total = 0.0;

    for _ in 0..RUNS {
        // Truth starts from the prior the filter is given, so the prior is honest.
        let mut truth = [rng.draw(), rng.draw(), rng.draw()];
        let mut truths = Vec::with_capacity(ARC);
        let mut measurements = Vec::with_capacity(ARC);
        for _ in 0..ARC {
            let w = [rng.draw(), rng.draw(), rng.draw()];
            truth = matvec(&stm, &truth);
            let noise = matvec(&process_noise_sqrt, &w);
            for i in 0..N {
                truth[i] += noise[i];
            }
            truths.push(truth);
            let clean: f64 = (0..N).map(|i| partial[i] * truth[i]).sum();
            measurements.push(clean + libm::sqrt(noise_variance) * rng.draw());
        }

        let (record, kf) = forward(&measurements, &p0);
        let mut smoother = RtsSmoother::new(kf.x, kf.covariance_sqrt());
        for reference in record.iter().rev().take(ARC - 1 - EPOCH) {
            smoother.step(&reference.step).expect("non-singular");
        }
        // `record[k].filtered` is the estimate *before* epoch k's own update, so stepping back
        // ARC-1-EPOCH intervals from the end lands on the estimate at truths[EPOCH].
        let estimate = smoother.state();
        let mut residual = [0.0; N];
        for i in 0..N {
            residual[i] = estimate[i] - truths[EPOCH][i];
        }
        total += normalized_squared(&smoother.covariance_sqrt(), &residual);
    }

    let mean = total / RUNS as f64;
    assert!(
        is_consistent(mean, N, RUNS, 4.0),
        "smoothed NEES {mean:.4} is not consistent with {N} degrees of freedom over {RUNS} runs"
    );
}

#[test]
fn a_singular_prediction_is_reported_not_papered_over() {
    // Zero process noise and a rank-deficient Φ: the predicted covariance has no third direction, so
    // the smoother gain does not exist there. Returning a fabricated estimate would be worse than
    // returning nothing.
    let step = SmoothingStep::<N> {
        filtered: [0.0; N],
        filtered_sqrt: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        stm: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        process_noise_sqrt: [[0.0; N]; N],
        predicted: [0.0; N],
    };
    let mut smoother = RtsSmoother::new(
        [0.0; N],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
    let outcome = smoother.step(&step);
    assert_eq!(outcome, Err(RtsError::SingularPrediction { direction: 2 }));

    // The index is the whole content of the report: a caller that is told only "singular" has to
    // re-derive which state it lost, and the variant is the crate's public way of saying so.
    let reported = outcome.unwrap_err().to_string();
    assert!(
        reported.contains("singular") && reported.contains('2'),
        "the message must name the direction that failed, got {reported:?}"
    );
}

/// The stored factor stays lower-triangular. This is load-bearing, not cosmetic:
/// `consistency::normalized_squared` forward-substitutes through whatever `covariance_sqrt`
/// returns, and the consistency test above would silently measure the wrong quantity if the final
/// triangularising sweep were skipped — `covariance()` is `Ŝ Ŝᵀ` and cannot see the difference.
/// The sweep annihilates analytically rather than to the bit; the worst above-diagonal leftover
/// measured over this arc is 1.383e-17 against a factor whose largest entry is 1.633, so the bound
/// is 1e-14.
#[test]
fn the_smoothed_factor_stays_lower_triangular() {
    let measurements: Vec<f64> = (0..10).map(|k| libm::cos(0.7 * f64::from(k))).collect();
    let p0 = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]];
    let (record, kf) = forward(&measurements, &p0);

    let mut smoother = RtsSmoother::new(kf.x, kf.covariance_sqrt());
    for (epoch, reference) in record.iter().enumerate().rev() {
        smoother.step(&reference.step).expect("non-singular");
        let s = smoother.covariance_sqrt();
        for i in 0..N {
            for j in (i + 1)..N {
                assert!(
                    s[i][j].abs() < 1e-14,
                    "epoch {epoch}: Ŝ[{i}][{j}] = {:e} above the diagonal",
                    s[i][j]
                );
            }
        }
        let p = smoother.covariance();
        for i in 0..N {
            for j in 0..N {
                let gram: f64 = (0..N).map(|k| s[i][k] * s[j][k]).sum();
                assert!(
                    (p[i][j] - gram).abs() < 1e-12,
                    "epoch {epoch}: P̂ ≠ Ŝ Ŝᵀ at [{i}][{j}]"
                );
            }
        }
    }
}

proptest! {
    /// The two forms agree on arbitrary systems, not only the one hand-picked above.
    ///
    /// `Φ` is drawn freely and `S_Q`, `S₀` are drawn as lower-triangular factors with a positive
    /// diagonal, so `Q` and `P₀` are positive-definite by construction and the predicted covariance
    /// is never singular — the case the shipped form is *supposed* to reject, and which the
    /// reference inverse could not evaluate anyway. Worst measured departure over 20 000 draws
    /// (`PROPTEST_CASES=20000`; the test itself runs proptest's default 256): 1.543e-13 relative on
    /// the state and 1.609e-13 on the covariance, so the bound is 1e-11.
    #[test]
    fn the_two_forms_agree_on_arbitrary_systems(
        phi in prop::array::uniform3(prop::array::uniform3(-1.2_f64..1.2)),
        q_diag in prop::array::uniform3(0.02_f64..1.0),
        q_off in prop::array::uniform3(-0.3_f64..0.3),
        s_diag in prop::array::uniform3(0.1_f64..3.0),
        s_off in prop::array::uniform3(-1.0_f64..1.0),
        x_filtered in prop::array::uniform3(-5.0_f64..5.0),
        x_predicted in prop::array::uniform3(-5.0_f64..5.0),
        x_ahead in prop::array::uniform3(-5.0_f64..5.0),
    ) {
        let lower = |diag: [f64; N], off: [f64; N]| {
            let mut m = [[0.0; N]; N];
            m[0][0] = diag[0];
            m[1][0] = off[0];
            m[1][1] = diag[1];
            m[2][0] = off[1];
            m[2][1] = off[2];
            m[2][2] = diag[2];
            m
        };
        let filtered_sqrt = lower(s_diag, s_off);
        let process_noise_sqrt = lower(q_diag, q_off);

        // P⁻ = Φ P⁺ Φᵀ + Q, formed here only so the reference recursion has it.
        let p_filtered = matmul(&filtered_sqrt, &transpose(&filtered_sqrt));
        let q = matmul(&process_noise_sqrt, &transpose(&process_noise_sqrt));
        let predicted_covariance = add(
            &matmul(&matmul(&phi, &p_filtered), &transpose(&phi)),
            &q,
        );

        // The look-ahead smoothed covariance has to be a real covariance, and no larger than the
        // prediction it refines, or the pair being compared is not one an arc could produce.
        let mut p_ahead = predicted_covariance;
        for i in 0..N {
            for j in 0..N {
                p_ahead[i][j] *= 0.5;
            }
        }
        let mut s_ahead = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..=i {
                let mut sum = p_ahead[i][j];
                for k in 0..j {
                    sum -= s_ahead[i][k] * s_ahead[j][k];
                }
                if i == j {
                    prop_assume!(sum > 1e-12);
                    s_ahead[i][j] = libm::sqrt(sum);
                } else {
                    s_ahead[i][j] = sum / s_ahead[j][j];
                }
            }
        }

        let reference = Reference {
            step: SmoothingStep {
                filtered: x_filtered,
                filtered_sqrt,
                stm: phi,
                process_noise_sqrt,
                predicted: x_predicted,
            },
            predicted_covariance,
        };

        let mut smoother = RtsSmoother::new(x_ahead, s_ahead);
        smoother.step(&reference.step).expect("Q is positive-definite");
        let (x_want, p_want) = textbook_step(&reference, &x_ahead, &p_ahead);

        let got_x = smoother.state();
        let got_p = smoother.covariance();
        let x_scale = (0..N).map(|i| x_want[i].abs()).fold(1.0, f64::max);
        let p_scale = (0..N).map(|i| p_want[i][i].abs()).fold(1.0, f64::max);
        for i in 0..N {
            prop_assert!(
                (got_x[i] - x_want[i]).abs() / x_scale < 1e-11,
                "state[{}]: {} vs {}", i, got_x[i], x_want[i]
            );
            for j in 0..N {
                prop_assert!(
                    (got_p[i][j] - p_want[i][j]).abs() / p_scale < 1e-11,
                    "P̂[{}][{}]: {} vs {}", i, j, got_p[i][j], p_want[i][j]
                );
            }
        }
    }
}

/// The joint posterior of the four-epoch arc of `derivations/rts_smoother.mac`, from the inverse of
/// its block-tridiagonal information matrix in exact rationals: `(x̂, P̂)` at epochs 1 and 0.
const JOINT_EPOCH_1: ([f64; 2], [[f64; 2]; 2]) = (
    [
        8.522_036_876_279_131_786_144_677e-1,
        6.814_801_548_688_902_322_293_66e-1,
    ],
    [
        [
            1.166_256_804_073_796_065_676_383e-1,
            -8.211_563_153_291_958_509_734_778e-2,
        ],
        [
            -8.211_563_153_291_958_509_734_778e-2,
            2.742_566_761_824_024_071_608_762e-1,
        ],
    ],
);
const JOINT_EPOCH_0: ([f64; 2], [[f64; 2]; 2]) = (
    [
        6.684_961_323_778_554_565_379_192e-1,
        3.149_556_632_598_205_730_376_299e-1,
    ],
    [
        [
            2.097_033_427_242_537_922_125_578e-1,
            -1.647_966_084_820_253_359_673_447e-1,
        ],
        [
            -1.647_966_084_820_253_359_673_447e-1,
            3.305_303_941_963_770_943_361_815e-1,
        ],
    ],
);

/// (G2) The smoother computes what a smoother is defined to be: the mean and covariance of each
/// state conditioned on every measurement of the arc. `derivations/rts_smoother.mac` forms that
/// conditional directly — the joint Gaussian over `(x₀ … x₃)`, its information matrix inverted in
/// exact rationals — and shows the array form's blocks are the textbook recursion's. Dyadic inputs,
/// measurement first at every epoch; the bound is 16 ulps of the prior's scale, 4.
#[test]
fn the_smoother_is_the_joint_gaussian_conditional() {
    let stm = [[1.0, 0.5], [0.0, 1.0]];
    let sq = [[0.25, 0.0], [0.125, 0.5]];
    let h = [1.0, 0.25];
    let mut kf = SquareRootKalman::new([0.5, -1.0], [[2.0, 0.0], [0.5, 1.0]]);
    let mut record = Vec::new();
    for (k, z) in [0.75, 0.5, 1.75, 2.5].into_iter().enumerate() {
        let innovation = z - (h[0] * kf.x[0] + h[1] * kf.x[1]);
        kf.measurement_update(&h, innovation, 0.25);
        if k == 3 {
            break;
        }
        let filtered = kf.x;
        kf.x = [
            stm[0][0] * filtered[0] + stm[0][1] * filtered[1],
            filtered[1],
        ];
        record.push(SmoothingStep {
            filtered,
            filtered_sqrt: kf.covariance_sqrt(),
            stm,
            process_noise_sqrt: sq,
            predicted: kf.x,
        });
        kf.time_update(&stm, &sq);
    }
    let mut smoother = RtsSmoother::new(kf.x, kf.covariance_sqrt());
    let tol = 16.0 * f64::EPSILON * 4.0;
    let mut worst = 0.0_f64;
    smoother.step(&record[2]).unwrap();
    for (back, want) in [(1, &JOINT_EPOCH_1), (0, &JOINT_EPOCH_0)] {
        smoother.step(&record[back]).unwrap();
        let (x, p) = (smoother.state(), smoother.covariance());
        for i in 0..2 {
            worst = worst.max((x[i] - want.0[i]).abs());
            assert!(
                (x[i] - want.0[i]).abs() <= tol,
                "epoch {back}: x̂[{i}] = {}",
                x[i]
            );
            for j in 0..2 {
                worst = worst.max((p[i][j] - want.1[i][j]).abs());
                assert!(
                    (p[i][j] - want.1[i][j]).abs() <= tol,
                    "epoch {back}: P̂[{i}][{j}] = {}",
                    p[i][j]
                );
            }
        }
    }
    println!("RTS vs the exact joint conditional: worst {worst:.2e}");
}
