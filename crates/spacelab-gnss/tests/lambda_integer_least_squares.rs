//! LAMBDA against an exhaustive integer search, on random positive-definite covariances.
//!
//! **No published test vector was reachable.** The Teunissen/Delft papers that define the method
//! state the algorithm and plot the decorrelated variance spectrum but print no worked numerical
//! example; the LAMBDA MATLAB toolbox, which ships one, is not in this repository and could not be
//! retrieved. So the oracle here is not a transcription — it is a *different algorithm for the same
//! problem*, and it is one that cannot be wrong:
//!
//! For a float vector `â` and covariance `Q`, write `f(a) = (a − â)ᵀQ⁻¹(a − â)`. Take the `n + 1`
//! integer vectors made of `round(â)` and of `round(â)` with one coordinate moved to its
//! next-nearest integer. The `p`-th smallest `f` over that set is an upper bound on the `p`-th
//! smallest `f` over all of `Zⁿ`, simply because those `n + 1` vectors are `n + 1` distinct members
//! of `Zⁿ`. Since `xᵀQ⁻¹x ≥ ‖x‖²/λ_max(Q)`, every one of the `p` best therefore lies inside the box
//! `|aᵢ − âᵢ| ≤ sqrt(ceilingₚ · λ_max(Q))`, and `λ_max(Q)` is bounded above by the matrix's maximum
//! absolute row sum. Enumerating that box and taking the smallest `f` values is a *proof* of the
//! answer, not an approximation of it — no decorrelation, no factorisation, no search heuristic, and
//! `Q⁻¹` formed by Gauss-Jordan rather than by the `LᵀDL` factors LAMBDA uses.
//!
//! The bound has to be sized for `p`, not for the optimum: a box that provably holds the best vector
//! need not hold the second best, and sizing it for the optimum alone made this oracle wrongly
//! accuse a correct runner-up on its first trial.
//!
//! Two things are compared per trial: the best integer vector, which must be identical, and the
//! second best, which must also be identical — the second is what the ratio test consumes, and a
//! search that shrinks its ellipsoid too eagerly gets the first right and the second wrong.

#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(clippy::doc_markdown)] // LAMBDA, Gauss-Jordan, Teunissen are names in prose
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use spacelab_gnss::lambda::Reduction;

/// SplitMix64 — a deterministic generator, so a failing trial is reproducible from its seed alone.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// `Q = B Bᵀ + δI` — symmetric positive definite by construction, and elongated because the entries
/// of `B` are correlated by a shared column, which is what makes the trial worth running.
fn random_covariance(rng: &mut SplitMix, n: usize) -> Vec<f64> {
    let mut b = vec![0.0_f64; n * n];
    for entry in &mut b {
        *entry = rng.signed();
    }
    // A shared component in every row: without it the random matrix is nearly isotropic and
    // rounding solves the problem, which would test nothing.
    let shared: Vec<f64> = (0..n).map(|_| rng.signed()).collect();
    let weight = 2.0 + rng.signed();
    for i in 0..n {
        for j in 0..n {
            b[i * n + j] += weight * shared[j];
        }
    }
    let mut q = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let dot: f64 = (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum();
            q[i * n + j] = dot;
        }
        q[i * n + i] += 0.05;
    }
    q
}

/// `Q⁻¹` by Gauss-Jordan with partial pivoting — formed independently of anything LAMBDA computes.
fn invert(q: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = q.to_vec();
    let mut inverse = vec![0.0_f64; n * n];
    for i in 0..n {
        inverse[i * n + i] = 1.0;
    }
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))?;
        if a[pivot * n + k].abs() < 1e-12 {
            return None;
        }
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
                inverse.swap(k * n + c, pivot * n + c);
            }
        }
        let head = a[k * n + k];
        for c in 0..n {
            a[k * n + c] /= head;
            inverse[k * n + c] /= head;
        }
        for i in 0..n {
            if i == k {
                continue;
            }
            let factor = a[i * n + k];
            if factor == 0.0 {
                continue;
            }
            for c in 0..n {
                a[i * n + c] -= factor * a[k * n + c];
                inverse[i * n + c] -= factor * inverse[k * n + c];
            }
        }
    }
    Some(inverse)
}

fn objective(inverse: &[f64], n: usize, a: &[i64], float: &[f64]) -> f64 {
    let r: Vec<f64> = (0..n).map(|i| a[i] as f64 - float[i]).collect();
    (0..n)
        .map(|i| r[i] * (0..n).map(|j| inverse[i * n + j] * r[j]).sum::<f64>())
        .sum()
}

/// The two best integer vectors, found by enumerating the whole box the argument above proves the
/// minimiser lies in. `None` where that box is larger than `budget` points.
fn brute_force(
    float: &[f64],
    q: &[f64],
    n: usize,
    want: usize,
    budget: u128,
) -> Option<Vec<(Vec<i64>, f64)>> {
    assert!(want <= n + 1, "the seed set holds only n + 1 vectors");
    let inverse = invert(q, n)?;
    let rounded: Vec<i64> = float.iter().map(|v| v.round() as i64).collect();

    // round(â), then round(â) with one coordinate moved to its next-nearest integer.
    let mut seeds = vec![rounded.clone()];
    for i in 0..n {
        let mut moved = rounded.clone();
        moved[i] += if float[i] >= rounded[i] as f64 { 1 } else { -1 };
        seeds.push(moved);
    }
    let mut ceilings: Vec<f64> = seeds
        .iter()
        .map(|s| objective(&inverse, n, s, float))
        .collect();
    ceilings.sort_by(f64::total_cmp);
    let ceiling = ceilings[want - 1];

    // λ_max(Q) is at most the maximum absolute row sum, so this radius is an over-estimate of the
    // true one and the box is guaranteed to contain every minimiser.
    let row_sum_bound = (0..n)
        .map(|i| (0..n).map(|j| q[i * n + j].abs()).sum::<f64>())
        .fold(0.0_f64, f64::max);
    let radius = (ceiling * row_sum_bound).sqrt();
    if !radius.is_finite() {
        return None;
    }

    let ranges: Vec<(i64, i64)> = (0..n)
        .map(|i| {
            (
                (float[i] - radius).ceil() as i64,
                (float[i] + radius).floor() as i64,
            )
        })
        .collect();
    let points: u128 = ranges
        .iter()
        .map(|&(lo, hi)| u128::try_from((hi - lo + 1).max(0)).unwrap_or(0))
        .product();
    if points == 0 || points > budget {
        return None;
    }

    let mut best: Vec<(Vec<i64>, f64)> = Vec::new();
    let mut candidate: Vec<i64> = ranges.iter().map(|&(lo, _)| lo).collect();
    loop {
        let value = objective(&inverse, n, &candidate, float);
        best.push((candidate.clone(), value));
        best.sort_by(|a, b| a.1.total_cmp(&b.1));
        best.truncate(want);

        let mut axis = 0;
        loop {
            if axis == n {
                return Some(best);
            }
            candidate[axis] += 1;
            if candidate[axis] <= ranges[axis].1 {
                break;
            }
            candidate[axis] = ranges[axis].0;
            axis += 1;
        }
    }
}

/// LAMBDA's best two integer vectors are the best two, on every trial the exhaustive search can
/// afford to confirm.
///
/// Measured over the seeds below: of 600 trials in dimensions 2 to 5, **548 verified** and 52
/// skipped because their proof box exceeded the enumeration budget, with **0 disagreements** on the
/// optimum and **0** on the runner-up. Rounding the float vector — the estimator LAMBDA exists to
/// replace — was *not* the optimum on **315** of the 548, which is what says the trials are hard
/// enough to be worth running: a test where rounding always wins cannot distinguish LAMBDA from
/// rounding.
#[test]
fn lambda_returns_the_exhaustive_optimum() {
    const BUDGET: u128 = 400_000;
    let mut rng = SplitMix(0x5EED_1234_ABCD_0001);

    let (mut verified, mut skipped, mut rounding_was_wrong) = (0_u32, 0_u32, 0_u32);
    for trial in 0..600_u32 {
        let n = 2 + (trial as usize % 4);
        let q = random_covariance(&mut rng, n);
        let float: Vec<f64> = (0..n).map(|_| rng.signed() * 6.0).collect();

        let Some(truth) = brute_force(&float, &q, n, 2, BUDGET) else {
            skipped += 1;
            continue;
        };
        let reduction = Reduction::new(&float, &q).expect("a positive-definite covariance");
        let found = reduction
            .search(2)
            .expect("a search inside its node budget");
        assert_eq!(found.len(), 2, "trial {trial}");

        assert_eq!(
            found[0].ambiguities, truth[0].0,
            "trial {trial}, n = {n}: LAMBDA {:?} at {:.6}, exhaustive {:?} at {:.6}",
            found[0].ambiguities, found[0].squared_norm, truth[0].0, truth[0].1
        );
        assert!(
            (found[0].squared_norm - truth[0].1).abs() < 1e-9 * truth[0].1.max(1.0),
            "trial {trial}: objective {} vs {}",
            found[0].squared_norm,
            truth[0].1
        );
        assert_eq!(
            found[1].ambiguities, truth[1].0,
            "trial {trial}, n = {n}: runner-up {:?} at {:.9}, exhaustive {:?} at {:.9}",
            found[1].ambiguities, found[1].squared_norm, truth[1].0, truth[1].1
        );
        assert!(
            (found[1].squared_norm - truth[1].1).abs() < 1e-9 * truth[1].1.max(1.0),
            "trial {trial}: runner-up objective {} vs {}",
            found[1].squared_norm,
            truth[1].1
        );

        // The bootstrapped estimate is never better than the optimum, and the optimum is never
        // beaten by rounding — the two inequalities that order the three estimators.
        let bootstrap = reduction.bootstrap();
        assert!(
            bootstrap.squared_norm >= found[0].squared_norm - 1e-12,
            "trial {trial}: bootstrap beat the ILS optimum"
        );
        let rounded: Vec<i64> = float.iter().map(|v| v.round() as i64).collect();
        if rounded != truth[0].0 {
            rounding_was_wrong += 1;
        }

        verified += 1;
    }

    assert_eq!(
        (verified, skipped),
        (548, 52),
        "the trial set changed: {verified} verified, {skipped} skipped"
    );
    assert_eq!(
        rounding_was_wrong, 315,
        "trials where rounding was not the optimum: {rounding_was_wrong}"
    );
}

/// The transformation is unimodular on every trial, which is the property that makes the answer
/// transferable: `z = Zᵀa` is a bijection of the integers, so an optimum found in `z` really is an
/// optimum in `a`. Computed here from the returned `Z` by LU determinant, not from the reduction's
/// own bookkeeping.
///
/// Measured over 600 trials in dimensions 2 to 5: the worst `||det Z| − 1|` is **8.8818e-16**, one
/// ulp — the entries really are integers and the determinant really is ±1.
#[test]
fn every_transformation_is_unimodular() {
    let mut rng = SplitMix(0x5EED_1234_ABCD_0002);
    let mut worst = 0.0_f64;
    for trial in 0..600_u32 {
        let n = 2 + (trial as usize % 4);
        let q = random_covariance(&mut rng, n);
        let float: Vec<f64> = (0..n).map(|_| rng.signed() * 6.0).collect();
        let reduction = Reduction::new(&float, &q).expect("a positive-definite covariance");
        let z = reduction.transformation();

        for entry in z {
            assert!(
                (entry - entry.round()).abs() < 1e-9,
                "trial {trial}: non-integer Z entry {entry}"
            );
        }
        let det = determinant(z, n);
        worst = worst.max((det.abs() - 1.0).abs());
    }
    assert!(worst < 1e-12, "worst |det Z| deviation {worst:.6e}");
}

/// Determinant by Gaussian elimination with partial pivoting.
fn determinant(m: &[f64], n: usize) -> f64 {
    let mut a = m.to_vec();
    let mut det = 1.0_f64;
    for k in 0..n {
        let Some(pivot) = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
        else {
            return 0.0;
        };
        if a[pivot * n + k] == 0.0 {
            return 0.0;
        }
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
            }
            det = -det;
        }
        det *= a[k * n + k];
        for i in k + 1..n {
            let factor = a[i * n + k] / a[k * n + k];
            for c in k..n {
                a[i * n + c] -= factor * a[k * n + c];
            }
        }
    }
    det
}

/// The decorrelation, measured rather than asserted: over the same trials the reduction lowers the
/// conditional-variance spread `max(d)/min(d)`, which is what sets how many integer vectors the
/// search has to visit.
///
/// Measured over 600 trials in dimensions 2 to 5: the median reduction factor is **4.1644e0** and
/// the largest **3.3234e2**. The spread never grows.
#[test]
fn the_reduction_never_increases_the_variance_spread() {
    let mut rng = SplitMix(0x5EED_1234_ABCD_0003);
    let mut factors = Vec::new();
    for trial in 0..600_u32 {
        let n = 2 + (trial as usize % 4);
        let q = random_covariance(&mut rng, n);
        let float: Vec<f64> = (0..n).map(|_| rng.signed() * 6.0).collect();
        let reduction = Reduction::new(&float, &q).expect("a positive-definite covariance");

        // The spread before reduction is the spread of Q's own conditional variances, which for the
        // identity transformation are the pivots of its LᵀDL factorisation — recomputed here by
        // reducing the already-diagonalised problem is circular, so it is taken from the Cholesky
        // pivots of Q directly.
        let before = spread(&cholesky_pivots(&q, n).expect("positive definite"));
        let after = spread(reduction.conditional_variances());
        assert!(
            after <= before * (1.0 + 1e-9),
            "trial {trial}: spread grew {before:.4e} -> {after:.4e}"
        );
        factors.push(before / after);
    }
    factors.sort_by(f64::total_cmp);
    let median = factors[factors.len() / 2];
    let largest = *factors.last().unwrap();
    assert!(
        (median - 4.164_381).abs() < 1e-5,
        "median factor {median:.6e}"
    );
    assert!(
        (largest - 3.323_396e2).abs() < 1e-2,
        "largest factor {largest:.6e}"
    );
}

fn spread(d: &[f64]) -> f64 {
    let hi = d.iter().fold(f64::MIN, |m, v| m.max(*v));
    let lo = d.iter().fold(f64::MAX, |m, v| m.min(*v));
    hi / lo
}

/// The pivots of the Cholesky factorisation of `Q`, which are its conditional variances in the
/// original parametrisation.
fn cholesky_pivots(q: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut a = q.to_vec();
    let mut pivots = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let pivot = a[i * n + i];
        if !(pivot.is_finite() && pivot > 0.0) {
            return None;
        }
        pivots[i] = pivot;
        for j in 0..i {
            let factor = a[i * n + j] / pivot;
            for k in 0..=j {
                a[j * n + k] -= factor * a[i * n + k];
            }
        }
    }
    Some(pivots)
}
