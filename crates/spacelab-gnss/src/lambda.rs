//! **LAMBDA** — integer least-squares estimation of carrier-phase ambiguities.
//!
//! Given a float ambiguity vector `â` and its covariance `Q`, the integer least-squares (ILS)
//! problem is
//!
//! ```text
//! min over a in Z^n of (a − â)ᵀ Q⁻¹ (a − â)
//! ```
//!
//! which is NP-hard in general, so every method for it searches. Rounding `â` entry by entry solves
//! it only when `Q` is diagonal; on real GNSS covariances, which are extremely elongated, rounding
//! is wrong far more often than not.
//!
//! The Least-squares AMBiguity Decorrelation Adjustment (Teunissen 1993) makes the search cheap in
//! two stages, and both are here:
//!
//! 1. **Reduction.** Find a unimodular integer `Z` — `det Z = ±1`, so `z = Zᵀa` is integer exactly
//!    when `a` is — that makes `Q_z = ZᵀQZ` as close to diagonal as integer arithmetic allows. It is
//!    built from integer Gauss transformations (which drive the off-diagonal entries of the `LᵀDL`
//!    factor below ½ in magnitude) interleaved with permutations (which push the conditional
//!    variances into decreasing order). This is [`Reduction`].
//! 2. **Search.** Enumerate integer vectors inside the ellipsoid `(z − ẑ)ᵀQ_z⁻¹(z − ẑ) ≤ χ²`,
//!    level by level, shrinking `χ²` as candidates are found. Decorrelated, the levels are nearly
//!    independent and the first candidates tried are almost always the answer. This is
//!    [`Reduction::search`].
//!
//! [`Reduction::bootstrap`] is the cheap non-searching estimator on the same factorisation: fix the
//! last ambiguity by rounding, condition the rest on it, repeat. It is the Babai point of the
//! lattice, it is what the search starts from, and on a reduced `Q` it very often *is* the ILS
//! solution — but "very often" is not "always", which is why it is offered separately rather than
//! substituted for the search.
//!
//! Implemented from Chang, Yang & Zhou (2005), *MLAMBDA: a modified LAMBDA method for integer
//! least-squares estimation*, J. Geod. 79:552–565 — Algorithms 2.1 (integer Gauss transformation),
//! 2.2 (permutation), 2.3 (reduction) and the search bounds of its §2.2, which restate the original
//! LAMBDA. The modifications that paper proposes on top of them (symmetric pivoting, greedy
//! selection, lazy transformation) are *not* implemented: they change the cost, not the answer.
//!
//! Verified against a brute-force integer search over the full box the ellipsoid inscribes: see
//! `tests/lambda_integer_least_squares.rs`.

// The doc allow is for method and author names (LAMBDA, Teunissen, Babai, MLAMBDA); the single
// letters (l, d, n, z, q) are the source's own matrix notation.
#![allow(clippy::doc_markdown, clippy::many_single_char_names)]

use std::fmt;

/// A refusal to reduce a covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambdaError {
    /// The covariance is not square, or its side does not match the float vector's length.
    DimensionMismatch {
        /// Entries the covariance carries.
        covariance: usize,
        /// Entries the float ambiguity vector carries.
        ambiguities: usize,
    },
    /// The problem has no ambiguities to resolve.
    Empty,
    /// The covariance carries a non-finite entry, or the float vector does.
    NotFinite,
    /// The covariance is not symmetric to within its own scale.
    ///
    /// Checked rather than assumed: the factorisation reads only the lower triangle, so an
    /// asymmetric input would be silently reinterpreted as the symmetric matrix built from that
    /// half — a different problem, solved without complaint.
    NotSymmetric {
        /// Row of the worst offending pair.
        row: usize,
        /// Column of the worst offending pair.
        column: usize,
    },
    /// The covariance is not positive definite, so the ILS objective is not a distance.
    NotPositiveDefinite {
        /// The pivot, in elimination order, that came out non-positive.
        pivot: usize,
    },
    /// The enumeration reached [`MAX_SEARCH_NODES`] without exhausting the ellipsoid.
    SearchExhausted {
        /// Candidates held when the budget ran out. Fewer than were asked for, and not provably the
        /// best ones, which is why they are reported as a count rather than returned.
        found: usize,
    },
}

impl fmt::Display for LambdaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::DimensionMismatch {
                covariance,
                ambiguities,
            } => write!(
                f,
                "a covariance of {covariance} entries does not go with {ambiguities} ambiguities"
            ),
            Self::Empty => write!(f, "there are no ambiguities to resolve"),
            Self::NotFinite => write!(f, "the covariance or the float vector is not finite"),
            Self::NotSymmetric { row, column } => write!(
                f,
                "the covariance is not symmetric between entry ({row}, {column}) and its transpose"
            ),
            Self::NotPositiveDefinite { pivot } => write!(
                f,
                "the covariance is not positive definite: pivot {pivot} is not positive"
            ),
            Self::SearchExhausted { found } => write!(
                f,
                "the enumeration visited {MAX_SEARCH_NODES} nodes holding {found} candidates \
                 without exhausting the ellipsoid"
            ),
        }
    }
}

impl std::error::Error for LambdaError {}

/// Tree nodes [`Reduction::search`] will visit before it refuses to keep going.
///
/// The enumeration terminates for every positive-definite factorisation — it walks outwards from
/// the conditional float at each level, so the partial cost grows without bound along any branch
/// and every branch is eventually pruned — but *how many* nodes that takes is a property of the
/// conditioning, not of the dimension, and a caller whose covariance is badly conditioned has no
/// other way to be told that the answer is not coming. A search that stops without saying so is
/// worse than one that refuses.
///
/// The value is a budget rather than a bound on anything the mathematics promises, so it is public
/// and reachable. A unit covariance with every float at a half-integer is the cheapest way to reach
/// it: every level ties, so both integers either side of each conditional float cost the same and
/// the ellipsoid never shrinks past them, and the tree doubles with every ambiguity added. Measured
/// on that family, 18 ambiguities finish inside this budget and 19 exceed it.
pub const MAX_SEARCH_NODES: usize = 1_000_000;

/// One integer candidate and the value of the ILS objective at it.
#[derive(Debug, Clone, PartialEq)]
pub struct Candidate {
    /// The integer ambiguity vector, in the caller's original parametrisation.
    pub ambiguities: Vec<i64>,
    /// `(a − â)ᵀ Q⁻¹ (a − â)` — smaller is better, and the ratio of the second to the first is the
    /// classical acceptance test.
    pub squared_norm: f64,
}

/// A row-major square matrix of the side the reduction was built at.
#[derive(Debug, Clone, PartialEq)]
struct Square {
    n: usize,
    data: Vec<f64>,
}

impl Square {
    fn identity(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self { n, data }
    }

    fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    fn swap_columns(&mut self, a: usize, b: usize) {
        for i in 0..self.n {
            self.data.swap(i * self.n + a, i * self.n + b);
        }
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        for j in 0..self.n {
            self.data.swap(a * self.n + j, b * self.n + j);
        }
    }
}

/// A decorrelated ILS problem: the transformation, the factorisation and the transformed float
/// vector the search runs on.
///
/// Building one is the expensive half of LAMBDA and depends on `Q` alone, so a caller holding `Q`
/// fixed across epochs — the usual case, since the ambiguity covariance changes far more slowly
/// than the float estimate — builds it once.
#[derive(Debug, Clone, PartialEq)]
pub struct Reduction {
    n: usize,
    /// Unit lower triangular `L̄` of `Q_z = L̄ᵀ D̄ L̄`.
    l: Square,
    /// Conditional variances `d̄`, strictly positive.
    d: Vec<f64>,
    /// The unimodular transformation `Z`, integer-valued.
    z: Square,
    /// `Z⁻¹`, integer-valued — carried along rather than recomputed, since every operation applied
    /// to `Z` has a known inverse and integer matrix inversion after the fact does not.
    z_inverse: Square,
    /// `ẑ = Zᵀâ`, the float vector in the decorrelated parametrisation.
    z_hat: Vec<f64>,
}

/// `Q = LᵀDL` with `L` unit lower triangular and `D` diagonal — the factorisation LAMBDA is built
/// on, in which `d[i]` is the variance of `â[i]` conditioned on `a[i+1..n]` being fixed.
///
/// Eliminates from the last index backwards, which is what makes the conditioning run in that
/// direction.
fn ltdl(q: &[f64], n: usize) -> Result<(Square, Vec<f64>), LambdaError> {
    let mut work = q.to_vec();
    let mut l = Square::identity(n);
    let mut d = vec![0.0; n];

    for i in (0..n).rev() {
        let pivot = work[i * n + i];
        if !(pivot.is_finite() && pivot > 0.0) {
            return Err(LambdaError::NotPositiveDefinite { pivot: i });
        }
        d[i] = pivot;
        for j in 0..=i {
            l.set(i, j, work[i * n + j] / pivot.sqrt());
        }
        for j in 0..i {
            let scale = l.at(i, j);
            for k in 0..=j {
                work[j * n + k] -= l.at(i, k) * scale;
            }
        }
        let diagonal = l.at(i, i);
        for j in 0..=i {
            l.set(i, j, l.at(i, j) / diagonal);
        }
    }
    Ok((l, d))
}

impl Reduction {
    /// Decorrelate the problem given by the float ambiguities `float` and their covariance
    /// `covariance` (row-major, `n × n`).
    ///
    /// # Errors
    /// [`LambdaError`] if the dimensions disagree, the problem is empty, an entry is not finite, the
    /// covariance is not symmetric, or it is not positive definite.
    pub fn new(float: &[f64], covariance: &[f64]) -> Result<Self, LambdaError> {
        let n = float.len();
        if n == 0 {
            return Err(LambdaError::Empty);
        }
        if covariance.len() != n * n {
            return Err(LambdaError::DimensionMismatch {
                covariance: covariance.len(),
                ambiguities: n,
            });
        }
        if !float.iter().chain(covariance).all(|v| v.is_finite()) {
            return Err(LambdaError::NotFinite);
        }
        // Symmetry to the matrix's own scale: an absolute tolerance would reject a covariance in
        // squared cycles of a long arc and accept one in squared metres that is visibly asymmetric.
        let scale = covariance.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for i in 0..n {
            for j in 0..i {
                let asymmetry = (covariance[i * n + j] - covariance[j * n + i]).abs();
                if asymmetry > 1e-9 * scale.max(f64::MIN_POSITIVE) {
                    return Err(LambdaError::NotSymmetric { row: i, column: j });
                }
            }
        }

        let (mut l, mut d) = ltdl(covariance, n)?;
        let mut z = Square::identity(n);
        let mut z_inverse = Square::identity(n);
        let mut z_hat = float.to_vec();

        // Algorithm 2.3: sweep the columns from the second to last towards the first, driving the
        // sub-diagonal entries below 1/2 and permuting whenever that lowers the conditional
        // variance of the pair. A permutation restarts the sweep, because it changes the columns to
        // its right; `settled` records how far the previous sweep got so those columns are not
        // re-transformed for nothing.
        // A single ambiguity has no pair to decorrelate: the sweep below starts at `n − 2`, which
        // does not exist, and the answer is already the rounded float.
        if n >= 2 {
            let mut k = n - 2;
            let mut settled = k;
            loop {
                if k <= settled {
                    for i in k + 1..n {
                        gauss(&mut l, &mut z, &mut z_inverse, &mut z_hat, i, k);
                    }
                }
                let candidate = d[k] + l.at(k + 1, k) * l.at(k + 1, k) * d[k + 1];
                if candidate < d[k + 1] {
                    permute(
                        &mut l,
                        &mut d,
                        &mut z,
                        &mut z_inverse,
                        &mut z_hat,
                        k,
                        candidate,
                    );
                    settled = k;
                    k = n - 2;
                } else if k == 0 {
                    break;
                } else {
                    k -= 1;
                }
            }
        }

        Ok(Self {
            n,
            l,
            d,
            z,
            z_inverse,
            z_hat,
        })
    }

    /// The number of ambiguities.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.n
    }

    /// Whether there are no ambiguities. Never true — [`Reduction::new`] refuses an empty problem —
    /// and present because a type with `len` and without `is_empty` is a trap for the reader.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The unimodular transformation `Z`, row-major, with `z = Zᵀa`.
    ///
    /// Integer-valued by construction: every entry is an exact integer stored in an `f64`, which is
    /// lossless well past any magnitude a GNSS reduction produces.
    #[must_use]
    pub fn transformation(&self) -> &[f64] {
        &self.z.data
    }

    /// The conditional variances `d̄` of the decorrelated problem, in index order.
    ///
    /// Their spread is what the reduction exists to shrink: the ratio of the largest to the smallest
    /// is the search's difficulty, and on GNSS covariances the reduction lowers it by orders of
    /// magnitude.
    #[must_use]
    pub fn conditional_variances(&self) -> &[f64] {
        &self.d
    }

    /// The bootstrapped (Babai) estimate: round the last ambiguity, condition the rest on it, work
    /// backwards, and map back through `Z`.
    ///
    /// Cheap — no search — and on a reduced problem usually the ILS solution, but not guaranteed to
    /// be. Use it where an answer is needed within a fixed budget, or as the starting point it is
    /// for [`Reduction::search`].
    #[must_use]
    pub fn bootstrap(&self) -> Candidate {
        let mut z = vec![0_i64; self.n];
        let mut conditioned = vec![0.0_f64; self.n];
        for i in (0..self.n).rev() {
            conditioned[i] = self.conditional_float(&z, &conditioned, i);
            z[i] = round_half_away(conditioned[i]);
        }
        let squared_norm = self.objective(&z, &conditioned);
        Candidate {
            ambiguities: self.back_transform(&z),
            squared_norm,
        }
    }

    /// The `count` best integer vectors, in increasing order of the ILS objective.
    ///
    /// The first is the integer least-squares solution. The second is what the standard ratio test
    /// needs, which is why asking for two is the usual call. Returns fewer than `count` only when
    /// `count` exceeds the number of integer vectors that exist, which for `n ≥ 1` it does not.
    ///
    /// Enumerates in the Schnorr–Euchner order — at every level the nearest integer first, then
    /// alternating outwards — so the first full vector found is the bootstrapped estimate and the
    /// ellipsoid can be shrunk to it immediately. Once `count` candidates are held, the bound is the
    /// worst of them, and the search is exhaustive within that bound: any vector it discards is
    /// provably worse than every one it kept.
    ///
    /// `count` is therefore also a work bound — the ellipsoid stays unbounded until that many
    /// candidates are held, so asking for a large `count` asks for a large enumeration.
    ///
    /// The projections are carried between levels rather than rebuilt at each node, after
    /// Ghasemmehdi and Agrell (*Faster projection in sphere decoding*, IEEE Trans. Inf. Theory
    /// **57**(6):3530–3536, 2011, §III-B). A node revisited after a climb costs one multiply–add
    /// per level the climb passed through instead of a fresh inner product over every level above
    /// it, which is what LAMBDA 4.0 adopts and where its speed-up at high dimension comes from.
    ///
    /// # Errors
    /// [`LambdaError::SearchExhausted`] if the enumeration visits [`MAX_SEARCH_NODES`] nodes
    /// without emptying the ellipsoid.
    pub fn search(&self, count: usize) -> Result<Vec<Candidate>, LambdaError> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let n = self.n;
        let mut best: Vec<(Vec<i64>, f64)> = Vec::with_capacity(count + 1);
        let mut bound = f64::INFINITY;

        let mut z = vec![0_i64; n];
        let mut conditioned = vec![0.0_f64; n];
        // `z[i] - conditioned[i]`, held rather than recomputed because the projections below read
        // it once per level they fold in.
        let mut offset = vec![0.0_f64; n];
        // Partial objective accumulated from level `i` upwards, so `partial[i]` is the cost of the
        // levels above `i` and `partial[0]` plus level 0 is the full objective.
        let mut partial = vec![0.0_f64; n + 1];
        // The Schnorr-Euchner state at each level: the current integer and the step that walks
        // outwards from the conditional float, alternating sign and growing by one each turn.
        let mut step = vec![0.0_f64; n];
        // `projected[i * stride + j]` is `Σ_{k≥j} offset[k] · L̄(k, i)`, the part of level `i`'s
        // conditioning contributed by the levels at or above `j`. Row `n` is the empty sum, so a
        // column is `n + 1` long, and the column is the contiguous axis because folding walks a
        // column downwards. The stride is written once and read by both the allocation and the
        // slicing: a layout stated twice is a layout that can disagree with itself.
        let stride = n + 1;
        let mut projected = vec![0.0_f64; n * stride];
        // The smallest row of each column that still holds. Row `n` always does, so that is where
        // every column starts and where an invalidated column is measured from.
        let mut valid = vec![n; n];
        // Counted down rather than up, so the budget is spent rather than compared against.
        let mut budget = MAX_SEARCH_NODES;

        let mut level = n - 1;
        let mut deepest = level;
        self.enter(
            level,
            stride,
            &mut conditioned,
            &mut projected,
            &mut valid,
            &offset,
        );
        z[level] = round_half_away(conditioned[level]);
        offset[level] = float_of(z[level]) - conditioned[level];
        step[level] = if offset[level] > 0.0 { -1.0 } else { 1.0 };

        loop {
            let Some(left) = budget.checked_sub(1) else {
                return Err(LambdaError::SearchExhausted { found: best.len() });
            };
            budget = left;
            let cost = partial[level + 1] + offset[level] * offset[level] / self.d[level];
            if cost < bound {
                if level > 0 {
                    partial[level] = cost;
                    level -= 1;
                    deepest = level;
                    self.enter(
                        level,
                        stride,
                        &mut conditioned,
                        &mut projected,
                        &mut valid,
                        &offset,
                    );
                    z[level] = round_half_away(conditioned[level]);
                    offset[level] = float_of(z[level]) - conditioned[level];
                    step[level] = if offset[level] > 0.0 { -1.0 } else { 1.0 };
                    continue;
                }
                best.push((z.clone(), cost));
                best.sort_by(|a, b| a.1.total_cmp(&b.1));
                best.truncate(count);
                if best.len() == count {
                    bound = best[best.len() - 1].1;
                }
                // Fall through to the next integer at level 0. Nothing is conditioned on level 0,
                // so no column of `projected` goes stale.
                advance(&mut z, &mut step, level);
                offset[level] = float_of(z[level]) - conditioned[level];
            } else {
                // The enumeration at this level is outward from the conditional float, so every
                // integer still to come here is at least as far — the level is exhausted.
                if level == n - 1 {
                    break;
                }
                level += 1;
                advance(&mut z, &mut step, level);
                offset[level] = float_of(z[level]) - conditioned[level];
                invalidate(&mut valid, deepest, level);
                deepest = level;
            }
        }

        Ok(best
            .into_iter()
            .map(|(z, squared_norm)| Candidate {
                ambiguities: self.back_transform(&z),
                squared_norm,
            })
            .collect())
    }

    /// Bring column `i` of the projections up to date and write level `i`'s conditional float.
    ///
    /// Folds in one level per row the column is stale by, which after a descent from the level
    /// immediately above is a single multiply–add. `conditional_float` computes the same number
    /// from scratch, and does so in the opposite summation order.
    fn enter(
        &self,
        i: usize,
        stride: usize,
        conditioned: &mut [f64],
        projected: &mut [f64],
        valid: &mut [usize],
        offset: &[f64],
    ) {
        // The row the column is current from once the fold below has run — the first level that
        // conditions `i`, the last row the fold writes, and the row the conditional float reads.
        // One binding for all three, because they are one number.
        let current = i + 1;
        let column = &mut projected[i * stride..][..=self.n];
        for j in (current..valid[i]).rev() {
            column[j] = column[j + 1] + offset[j] * self.l.at(j, i);
        }
        valid[i] = current;
        conditioned[i] = self.z_hat[i] + column[current];
    }

    /// `ẑ_i` conditioned on the integers already fixed at levels above `i` (Chang et al. Eqn. 17).
    fn conditional_float(&self, z: &[i64], conditioned: &[f64], i: usize) -> f64 {
        let mut acc = self.z_hat[i];
        for j in i + 1..self.n {
            acc += (float_of(z[j]) - conditioned[j]) * self.l.at(j, i);
        }
        acc
    }

    /// The ILS objective at a fully fixed integer vector (Chang et al. Eqn. 18).
    fn objective(&self, z: &[i64], conditioned: &[f64]) -> f64 {
        (0..self.n)
            .map(|i| {
                let r = float_of(z[i]) - conditioned[i];
                r * r / self.d[i]
            })
            .sum()
    }

    /// `a = Z⁻ᵀ z` — the decorrelated integers back in the caller's parametrisation.
    fn back_transform(&self, z: &[i64]) -> Vec<i64> {
        (0..self.n)
            .map(|i| {
                let value: f64 = (0..self.n)
                    .map(|j| self.z_inverse.at(j, i) * float_of(z[j]))
                    .sum();
                round_half_away(value)
            })
            .collect()
    }
}

/// Nearest integer, ties away from zero. `i64` saturating, which no reduced ambiguity approaches.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn round_half_away(x: f64) -> i64 {
    let r = x.round();
    if r >= i64::MAX as f64 {
        i64::MAX
    } else if r <= i64::MIN as f64 {
        i64::MIN
    } else {
        r as i64
    }
}

#[allow(clippy::cast_precision_loss)]
fn float_of(x: i64) -> f64 {
    x as f64
}

/// Take the next integer at `level` in the Schnorr–Euchner order: the conditional float's nearest
/// integer, then one below, one above, two below, … so the enumeration is outward by distance.
fn advance(z: &mut [i64], step: &mut [f64], level: usize) {
    z[level] += round_half_away(step[level]);
    step[level] = -step[level] - step[level].signum();
}

/// Mark every column whose conditioning read level `level` as stale, after a climb from `deepest`.
///
/// A column at or above row `level + 1` sums only levels the climb did not touch, so it still
/// holds; anything at or below row `level` folded in the integer that has just moved. The columns
/// the climb passed through are stale by construction, and the ones below them are stale only
/// until the first that a previous climb already lifted — below that, `valid` is the flat block
/// that climb left behind, so the scan can stop.
fn invalidate(valid: &mut [usize], deepest: usize, level: usize) {
    for entry in &mut valid[deepest..level] {
        *entry = level + 1;
    }
    for entry in valid[..deepest].iter_mut().rev() {
        if *entry > level {
            break;
        }
        *entry = level + 1;
    }
}

/// Algorithm 2.1: the integer Gauss transformation `Z_ij`, applied to `L`, `Z`, `Z⁻¹` and `ẑ`.
///
/// Drives `L(i, j)` into `[−½, ½]` by subtracting an integer multiple of column `i` from column `j`.
fn gauss(
    l: &mut Square,
    z: &mut Square,
    z_inverse: &mut Square,
    z_hat: &mut [f64],
    i: usize,
    j: usize,
) {
    let mu = float_of(round_half_away(l.at(i, j)));
    if mu == 0.0 {
        return;
    }
    let n = l.n;
    for k in i..n {
        l.set(k, j, l.at(k, j) - mu * l.at(k, i));
    }
    for k in 0..n {
        z.set(k, j, z.at(k, j) - mu * z.at(k, i));
        // Z <- Z·(I − µ eᵢ eⱼᵀ) inverts to Z⁻¹ <- (I + µ eᵢ eⱼᵀ)·Z⁻¹, which is a row operation.
        z_inverse.set(i, k, z_inverse.at(i, k) + mu * z_inverse.at(j, k));
    }
    z_hat[j] -= mu * z_hat[i];
}

/// Algorithm 2.2: interchange ambiguities `k` and `k + 1`, updating the factorisation in place.
///
/// `delta` is the post-swap conditional variance `d̄[k+1]` the caller has already evaluated as the
/// reason to permute at all.
fn permute(
    l: &mut Square,
    d: &mut [f64],
    z: &mut Square,
    z_inverse: &mut Square,
    z_hat: &mut [f64],
    k: usize,
    delta: f64,
) {
    let n = l.n;
    let eta = d[k] / delta;
    let lambda = d[k + 1] * l.at(k + 1, k) / delta;
    d[k] = eta * d[k + 1];
    d[k + 1] = delta;

    for j in 0..k {
        let (a, b) = (l.at(k, j), l.at(k + 1, j));
        l.set(k, j, (-l.at(k + 1, k)).mul_add(a, b));
        l.set(k + 1, j, eta.mul_add(a, lambda * b));
    }
    l.set(k + 1, k, lambda);
    for i in k + 2..n {
        let (a, b) = (l.at(i, k), l.at(i, k + 1));
        l.set(i, k, b);
        l.set(i, k + 1, a);
    }
    z.swap_columns(k, k + 1);
    z_inverse.swap_rows(k, k + 1);
    z_hat.swap(k, k + 1);
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use proptest::prelude::*;

    use super::*;

    /// `Q` rebuilt from a reduction's own factors: `L̄ᵀ D̄ L̄`.
    fn from_factors(r: &Reduction) -> Vec<f64> {
        let n = r.n;
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = (0..n).map(|k| r.l.at(k, i) * r.d[k] * r.l.at(k, j)).sum();
            }
        }
        out
    }

    /// `ZᵀQZ` for the reduction's own `Z`.
    fn transformed(q: &[f64], r: &Reduction) -> Vec<f64> {
        let n = r.n;
        let mut out = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] = (0..n)
                    .map(|a| {
                        r.z.at(a, i) * (0..n).map(|b| q[a * n + b] * r.z.at(b, j)).sum::<f64>()
                    })
                    .sum();
            }
        }
        out
    }

    /// A three-dimensional covariance with the elongation a real GNSS ambiguity problem has.
    fn elongated() -> (Vec<f64>, Vec<f64>) {
        let q = vec![
            6.290, 5.978, 0.544, //
            5.978, 6.292, 2.340, //
            0.544, 2.340, 6.288,
        ];
        let float = vec![5.45, 3.10, 2.97];
        (float, q)
    }

    /// The factorisation is what it claims: `Q = LᵀDL`, `L` unit lower triangular, `D > 0`.
    #[test]
    fn the_factorisation_reproduces_the_matrix() {
        let (_, q) = elongated();
        let (l, d) = ltdl(&q, 3).unwrap();
        for (i, &di) in d.iter().enumerate() {
            assert!((l.at(i, i) - 1.0).abs() < 1e-12, "unit diagonal at {i}");
            assert!(di > 0.0, "d[{i}] = {di}");
            for j in i + 1..3 {
                assert!(l.at(i, j).abs() < 1e-15, "upper entry ({i}, {j})");
            }
        }
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let rebuilt: f64 = (0..3).map(|k| l.at(k, i) * d[k] * l.at(k, j)).sum();
                worst = worst.max((rebuilt - q[i * 3 + j]).abs());
            }
        }
        assert!(worst < 1e-12, "LᵀDL residual {worst:.3e}");
    }

    /// The reduction's three defining properties, each of which a wrong bookkeeping step breaks on
    /// its own: `Z` is unimodular, `Q_z = ZᵀQZ` really is what the stored factors factorise, and the
    /// off-diagonal entries of `L̄` are bounded by ½.
    #[test]
    fn the_reduction_is_unimodular_consistent_and_bounded() {
        let (float, q) = elongated();
        let r = Reduction::new(&float, &q).unwrap();

        // Integer entries, and det Z = ±1 by the 3x3 rule.
        for v in r.transformation() {
            assert!((v - v.round()).abs() < 1e-12, "non-integer Z entry {v}");
        }
        let z = |i: usize, j: usize| r.z.at(i, j);
        let det = z(0, 0) * (z(1, 1) * z(2, 2) - z(1, 2) * z(2, 1))
            - z(0, 1) * (z(1, 0) * z(2, 2) - z(1, 2) * z(2, 0))
            + z(0, 2) * (z(1, 0) * z(2, 1) - z(1, 1) * z(2, 0));
        assert!((det.abs() - 1.0).abs() < 1e-9, "det Z = {det}");

        // Z·Z⁻¹ = I, which is what back-transforming a candidate depends on.
        for i in 0..3 {
            for j in 0..3 {
                let entry: f64 = (0..3).map(|k| r.z.at(i, k) * r.z_inverse.at(k, j)).sum();
                let want = f64::from(u8::from(i == j));
                assert!((entry - want).abs() < 1e-9, "ZZ⁻¹ ({i}, {j}) = {entry}");
            }
        }

        // The stored factors factorise the transformed covariance, not something adjacent to it.
        let want = transformed(&q, &r);
        let got = from_factors(&r);
        let worst = want
            .iter()
            .zip(&got)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-9, "ZᵀQZ against L̄ᵀD̄L̄: {worst:.3e}");

        // Bounded off-diagonals — the point of the integer Gauss transformations.
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    r.l.at(i, j).abs() <= 0.5 + 1e-12,
                    "L̄({i}, {j}) = {}",
                    r.l.at(i, j)
                );
            }
        }
    }

    /// Decorrelation is the whole justification for the transformation, so it is measured rather
    /// than asserted in prose.
    ///
    /// On the covariance above the conditional-variance spread `max(d)/min(d)` falls from
    /// **6.9977e1 to 6.8852e0**, a factor of 10.2 — that ratio is what sets how many integer
    /// candidates the search has to visit.
    #[test]
    fn the_reduction_flattens_the_conditional_variances() {
        let (float, q) = elongated();
        let (_, before) = ltdl(&q, 3).unwrap();
        let r = Reduction::new(&float, &q).unwrap();
        let after = r.conditional_variances();

        let spread = |d: &[f64]| {
            let hi = d.iter().fold(f64::MIN, |m, v| m.max(*v));
            let lo = d.iter().fold(f64::MAX, |m, v| m.min(*v));
            hi / lo
        };
        let (was, now) = (spread(&before), spread(after));
        assert!(
            now < was / 5.0,
            "spread {was:.4e} -> {now:.4e} is not a decorrelation"
        );
        assert!((was - 6.9977e1).abs() < 1e-2, "before {was:.4e}");
        assert!((now - 6.8852e0).abs() < 1e-3, "after {now:.4e}");
    }

    /// A diagonal covariance is the one case where the answer is known without a search: the
    /// ambiguities are independent, so integer least squares *is* rounding, and the reduction must
    /// not move the answer.
    #[test]
    fn a_diagonal_covariance_resolves_to_the_rounded_float() {
        let float = vec![1.4, -2.6, 0.5, 7.49];
        let q = vec![
            0.20, 0.0, 0.0, 0.0, //
            0.0, 0.11, 0.0, 0.0, //
            0.0, 0.0, 0.37, 0.0, //
            0.0, 0.0, 0.0, 0.05,
        ];
        let r = Reduction::new(&float, &q).unwrap();
        let best = r.search(1).unwrap();
        assert_eq!(best[0].ambiguities, vec![1, -3, 1, 7]);
        assert_eq!(r.bootstrap().ambiguities, vec![1, -3, 1, 7]);
    }

    /// An integer float vector is already the solution, and the objective there is exactly zero —
    /// the one value in the whole method that admits no tolerance.
    #[test]
    fn an_integer_float_vector_resolves_to_itself() {
        let (_, q) = elongated();
        let float = vec![4.0, -7.0, 11.0];
        let r = Reduction::new(&float, &q).unwrap();
        let best = r.search(2).unwrap();
        assert_eq!(best[0].ambiguities, vec![4, -7, 11]);
        assert!(
            best[0].squared_norm.abs() < 1e-18,
            "{}",
            best[0].squared_norm
        );
        assert!(
            best[1].squared_norm > 0.0,
            "the second candidate is the first"
        );
    }

    /// The candidates come back ordered, and the objective reported for each is the objective of
    /// the vector reported with it — computed here in the *original* parametrisation from `Q⁻¹`,
    /// which the search never forms, so agreement is a check on the transformation and not on
    /// arithmetic reused from it.
    #[test]
    fn the_reported_objective_is_the_objective_in_the_original_problem() {
        let (float, q) = elongated();
        let r = Reduction::new(&float, &q).unwrap();
        let candidates = r.search(5).unwrap();
        assert_eq!(candidates.len(), 5);

        // Q⁻¹ by Gauss-Jordan on the 3x3, independent of the reduction's factorisation.
        let inverse = invert3(&q);
        let mut previous = f64::NEG_INFINITY;
        for c in &candidates {
            let residual: Vec<f64> = (0..3)
                .map(|i| float_of(c.ambiguities[i]) - float[i])
                .collect();
            let quadratic: f64 = (0..3)
                .map(|i| {
                    residual[i]
                        * (0..3)
                            .map(|j| inverse[i * 3 + j] * residual[j])
                            .sum::<f64>()
                })
                .sum();
            assert!(
                (quadratic - c.squared_norm).abs() < 1e-9 * quadratic.max(1.0),
                "{:?}: {quadratic} vs {}",
                c.ambiguities,
                c.squared_norm
            );
            assert!(c.squared_norm >= previous, "candidates out of order");
            previous = c.squared_norm;
        }
    }

    fn invert3(q: &[f64]) -> Vec<f64> {
        let m = |i: usize, j: usize| q[i * 3 + j];
        let c = [
            m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1),
            m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2),
            m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1),
            m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2),
            m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0),
            m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2),
            m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0),
            m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1),
            m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0),
        ];
        let det = m(0, 0) * c[0] + m(0, 1) * c[3] + m(0, 2) * c[6];
        c.iter().map(|v| v / det).collect()
    }

    #[test]
    fn a_covariance_it_cannot_use_is_refused() {
        assert_eq!(Reduction::new(&[], &[]), Err(LambdaError::Empty));
        assert_eq!(
            Reduction::new(&[1.0, 2.0], &[1.0, 0.0, 0.0]),
            Err(LambdaError::DimensionMismatch {
                covariance: 3,
                ambiguities: 2,
            })
        );
        assert_eq!(
            Reduction::new(&[f64::NAN], &[1.0]),
            Err(LambdaError::NotFinite)
        );
        assert_eq!(
            Reduction::new(&[1.0, 2.0], &[1.0, 0.5, 0.4, 1.0]),
            Err(LambdaError::NotSymmetric { row: 1, column: 0 })
        );
        // Symmetric, but indefinite: the off-diagonal exceeds the geometric mean of the diagonal.
        assert_eq!(
            Reduction::new(&[1.0, 2.0], &[1.0, 2.0, 2.0, 1.0]),
            Err(LambdaError::NotPositiveDefinite { pivot: 0 })
        );
        assert!(r_messages_are_distinct());
    }

    fn r_messages_are_distinct() -> bool {
        let messages: Vec<String> = [
            LambdaError::Empty,
            LambdaError::DimensionMismatch {
                covariance: 3,
                ambiguities: 2,
            },
            LambdaError::NotFinite,
            LambdaError::NotSymmetric { row: 1, column: 0 },
            LambdaError::NotPositiveDefinite { pivot: 0 },
            LambdaError::SearchExhausted { found: 1 },
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        let distinct: std::collections::HashSet<&String> = messages.iter().collect();
        distinct.len() == messages.len() && messages.iter().all(|m| !m.is_empty())
    }

    /// Asking for nothing returns nothing rather than searching for it.
    #[test]
    fn a_zero_candidate_search_is_empty() {
        let (float, q) = elongated();
        assert_eq!(
            Reduction::new(&float, &q).unwrap().search(0).unwrap().len(),
            0
        );
    }

    /// A zero on the diagonal is not positive definite, and the refusal has to be strict.
    ///
    /// `pivot > 0.0` and `pivot >= 0.0` differ on exactly one input, and it is the one that matters:
    /// at `pivot == 0.0` the accepting form stores `d[i] = 0`, divides the whole row by `sqrt(0)` and
    /// hands the search a factorisation of infinities. A zero variance is not a contrived input
    /// either — it is what a caller writes for an ambiguity it considers already fixed.
    #[test]
    fn a_zero_variance_is_refused_rather_than_factorised() {
        assert_eq!(
            Reduction::new(&[1.0, 2.0], &[1.0, 0.0, 0.0, 0.0]),
            Err(LambdaError::NotPositiveDefinite { pivot: 1 })
        );
        // The sweep runs from the last index down, so the index named is the last row reached, and
        // with both variances zero it is still 1 rather than 0.
        assert_eq!(
            Reduction::new(&[1.0, 2.0], &[0.0, 0.0, 0.0, 0.0]),
            Err(LambdaError::NotPositiveDefinite { pivot: 1 })
        );
    }

    /// The bootstrapped objective, against the quadratic form written out by hand.
    ///
    /// A diagonal covariance is the one case that needs no factorisation to check: the reduction
    /// leaves it alone, so the conditional floats *are* the input floats and the objective is
    /// `Σ (round(fᵢ) − fᵢ)² / qᵢᵢ`, formed here from the inputs rather than from anything the module
    /// produced. What existed before asserted the integers `bootstrap` returns and, in the
    /// integration test, only that its objective is no *better* than the optimum — an inequality wide
    /// enough to survive any mutation that makes the number larger.
    #[test]
    fn the_bootstrapped_objective_is_the_quadratic_form_at_the_rounded_float() {
        let float = [1.4, -2.6, 0.5, 7.49];
        let variance = [0.20, 0.11, 0.37, 0.05];
        let mut q = vec![0.0; 16];
        for (i, v) in variance.iter().enumerate() {
            q[i * 4 + i] = *v;
        }
        let candidate = Reduction::new(&float, &q).unwrap().bootstrap();
        assert_eq!(candidate.ambiguities, vec![1, -3, 1, 7]);

        let want: f64 = (0..4)
            .map(|i| {
                let residual = float[i].round() - float[i];
                residual * residual / variance[i]
            })
            .sum();
        // 0.8 + 1.45454… + 0.67567… + 4.802
        assert!((want - 7.732_221_130_221_133).abs() < 1e-15, "{want}");
        assert!(
            (candidate.squared_norm - want).abs() < 1e-15,
            "{} vs {want}",
            candidate.squared_norm
        );
    }

    /// The enumeration leaves each level's conditional float in one direction first, and which one is
    /// observable exactly where the conditional float is an integer.
    ///
    /// `float_of(z) > conditioned` chooses the first step's sign, and against `>=` the two differ on
    /// one input: equality. That is not a measure-zero curiosity here. An integer float vector makes
    /// `ẑ = Zᵀa` an exact integer vector — `Z` is unimodular with exactly-representable entries — and
    /// at the solution every residual above a level is exactly zero, so *every* level ties. The two
    /// neighbours of the solution then cost the same to the last bit, and which one the search
    /// reports second is decided by the step's sign and by nothing else: the sort is stable, so it is
    /// the push order that survives into the answer.
    #[test]
    fn the_enumeration_walks_outwards_from_the_conditional_float() {
        let (_, q) = elongated();
        let r = Reduction::new(&[4.0, -7.0, 11.0], &q).unwrap();
        assert_eq!(r.len(), 3);

        // Fourteen, and the depth is the measurement rather than a margin. The sign is set at two
        // places: once per descent for every level below the top, and exactly once before the loop
        // for the top itself. The first speaks immediately — the two runners-up below are a ± pair
        // and their order is its doing. The second says nothing until the search has exhausted the
        // top level's first integer and climbed back, which on this covariance takes thirteen
        // candidates; asking for five leaves that statement unread. Measured: flipping the top
        // level's initialisation alone changes candidate 13 from `[4, -8, 8]` to `[4, -6, 14]`, its
        // ± partner at the identical cost 1.638185311189144e0, and changes nothing before it.
        let found = r.search(14).unwrap();
        let want = [
            (0, vec![4, -7, 11]),
            (1, vec![5, -6, 11]),
            (2, vec![3, -8, 11]),
            (3, vec![5, -6, 12]),
            (4, vec![3, -8, 10]),
            (13, vec![4, -8, 8]),
        ];
        for (index, want) in &want {
            assert_eq!(&found[*index].ambiguities, want, "candidate {index}");
        }
        // Each pair ties to the last bit, which is what leaves its order to the step's sign alone.
        for pair in [(1, 2), (3, 4), (11, 12)] {
            assert_eq!(
                found[pair.0].squared_norm.to_bits(),
                found[pair.1].squared_norm.to_bits(),
                "candidates {pair:?} must tie: {} vs {}",
                found[pair.0].squared_norm,
                found[pair.1].squared_norm
            );
        }
    }

    /// One ambiguity is a degenerate case the loops must still handle: the reduction sweep starts at
    /// `n − 2`, which for `n = 1` does not exist.
    #[test]
    fn a_single_ambiguity_resolves() {
        let r = Reduction::new(&[3.4], &[0.25]).unwrap();
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
        assert_eq!(r.search(1).unwrap()[0].ambiguities, vec![3]);
        // (3 − 3.4)² / 0.25 = 0.64
        assert!((r.search(1).unwrap()[0].squared_norm - 0.64).abs() < 1e-12);
    }

    /// The enumeration as it was written before the conditioning was carried down the tree: every
    /// conditional float recomputed from scratch out of the integers fixed above it.
    ///
    /// This is the oracle for [`Reduction::search`] and not a second implementation of it. It walks
    /// the identical tree in the identical order and differs in one thing only — where each
    /// conditional float comes from — so any disagreement is a disagreement about the bookkeeping,
    /// which is the whole of what the rewrite changed.
    fn recomputed_search(r: &Reduction, count: usize) -> Vec<Candidate> {
        let n = r.n;
        // The same budget the method under test carries. An oracle without one is a way for a
        // mutation of the shared code beneath both to stop the suite instead of failing it, and a
        // wall clock cannot tell those apart.
        let mut budget = MAX_SEARCH_NODES;
        let mut best: Vec<(Vec<i64>, f64)> = Vec::with_capacity(count + 1);
        let mut bound = f64::INFINITY;
        let mut z = vec![0_i64; n];
        let mut conditioned = vec![0.0_f64; n];
        let mut partial = vec![0.0_f64; n + 1];
        let mut step = vec![0.0_f64; n];

        let mut level = n - 1;
        conditioned[level] = r.conditional_float(&z, &conditioned, level);
        z[level] = round_half_away(conditioned[level]);
        step[level] = if float_of(z[level]) > conditioned[level] {
            -1.0
        } else {
            1.0
        };

        loop {
            assert!(budget > 0, "the recomputing oracle outran the node budget");
            budget -= 1;
            let residual = float_of(z[level]) - conditioned[level];
            let cost = partial[level + 1] + residual * residual / r.d[level];
            if cost < bound {
                if level > 0 {
                    partial[level] = cost;
                    level -= 1;
                    conditioned[level] = r.conditional_float(&z, &conditioned, level);
                    z[level] = round_half_away(conditioned[level]);
                    step[level] = if float_of(z[level]) > conditioned[level] {
                        -1.0
                    } else {
                        1.0
                    };
                    continue;
                }
                best.push((z.clone(), cost));
                best.sort_by(|a, b| a.1.total_cmp(&b.1));
                best.truncate(count);
                if best.len() == count {
                    bound = best[best.len() - 1].1;
                }
                advance(&mut z, &mut step, level);
            } else {
                if level == n - 1 {
                    break;
                }
                level += 1;
                advance(&mut z, &mut step, level);
            }
        }

        best.into_iter()
            .map(|(z, squared_norm)| Candidate {
                ambiguities: r.back_transform(&z),
                squared_norm,
            })
            .collect()
    }

    /// `Q = B Bᵀ + δI`, positive definite by construction and elongated by a column every row
    /// shares — without that the draw is nearly isotropic, rounding solves it, and the search never
    /// climbs far enough to leave a projection stale.
    fn covariance_from(entries: &[f64], n: usize, elongation: f64) -> Vec<f64> {
        let mut b = vec![0.0_f64; n * n];
        b.copy_from_slice(&entries[..n * n]);
        let shared = &entries[n * n..n * n + n];
        for i in 0..n {
            for j in 0..n {
                b[i * n + j] += elongation * shared[j];
            }
        }
        let mut q = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                q[i * n + j] = (0..n).map(|k| b[i * n + k] * b[j * n + k]).sum();
            }
            q[i * n + i] += 0.05;
        }
        q
    }

    proptest! {
        /// The carried projections against recomputation from scratch, candidate for candidate.
        ///
        /// The rewrite's risk is not that a conditional float comes out wrong once — it is that a
        /// column of projections goes stale, which needs the tree to descend, climb several levels
        /// and descend again before it is observable at all. A fixture reaches that by luck; a
        /// sweep over elongated covariances reaches it by construction, and the levels below the
        /// climb are exactly the ones `invalidate`'s second loop decides about.
        ///
        /// The two summation orders differ — scratch sums a column downwards from the top level,
        /// recomputation sums it upwards from the level below — so the objectives are compared to a
        /// tolerance rather than bit-for-bit. The integer vectors are not: those are the answer.
        #[test]
        fn the_carried_projections_reproduce_the_recomputed_search(
            n in 2_usize..=10,
            entries in prop::collection::vec(-1.0_f64..1.0, 110),
            floats in prop::collection::vec(-20.0_f64..20.0, 10),
            count in 1_usize..=6,
        ) {
            let q = covariance_from(&entries, n, 2.5);
            let Ok(r) = Reduction::new(&floats[..n], &q) else {
                return Ok(());
            };
            let got = r.search(count).unwrap();
            let want = recomputed_search(&r, count);
            prop_assert_eq!(got.len(), want.len(), "candidate count");
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                prop_assert_eq!(
                    &a.ambiguities, &b.ambiguities,
                    "candidate {} differs: {:?} against {:?}", i, a, b
                );
                let scale = b.squared_norm.abs().max(1.0);
                prop_assert!(
                    (a.squared_norm - b.squared_norm).abs() < 1e-9 * scale,
                    "candidate {} objective {} against {}", i, a.squared_norm, b.squared_norm
                );
            }
        }
    }

    /// The node budget is a number a caller can reach, and the refusal names what it was holding.
    ///
    /// A unit covariance with every float at a half-integer ties at every level: both integers
    /// either side of each conditional float cost the same, so the first candidate found never
    /// shrinks the ellipsoid past its own siblings and the tree doubles with each ambiguity. The
    /// two dimensions here sit either side of [`MAX_SEARCH_NODES`], which is what makes the
    /// constant's *value* observable rather than merely its existence — #315's rule, that a limit
    /// no test can reach is a limit nothing has checked the sense of.
    #[test]
    fn the_node_budget_is_reachable_and_refuses_rather_than_truncates() {
        let tied = |n: usize| {
            let mut q = vec![0.0_f64; n * n];
            for i in 0..n {
                q[i * n + i] = 1.0;
            }
            Reduction::new(&vec![0.5_f64; n], &q).unwrap().search(1)
        };
        assert_eq!(tied(18).unwrap().len(), 1);
        assert_eq!(tied(19), Err(LambdaError::SearchExhausted { found: 1 }));
    }

    /// The reduced conditioning against recomputation from scratch, in wall-clock seconds.
    ///
    /// `#[ignore]`d because it is a measurement and not a check: a timing assertion on a shared
    /// machine is a flake generator, and the number belongs in the doc comment where it can be
    /// read. Run it with `cargo test -p spacelab-gnss --release --lib the_reduced_conditioning --
    /// --ignored --nocapture`.
    ///
    /// Measured, alternating A-B-A-B over four rounds so startup drift cannot be mistaken for the
    /// effect, on random elongated covariances at `count = 1`, µs per search:
    ///
    /// | `n` | carried | recomputed | ratio |
    /// |----:|--------:|-----------:|------:|
    /// | 5   |    0.32 |       0.24 | 0.73  |
    /// | 10  |    0.67 |       0.48 | 0.71  |
    /// | 20  |    39.0 |       35.2 | 0.90  |
    /// | 30  |   1 174 |      1 292 | 1.10  |
    /// | 40  |  17 141 |     19 531 | 1.14  |
    ///
    /// The two sub-microsecond rows move by ±0.06 in the ratio between runs, which is the timer and
    /// not the algorithm; the rows from `n = 20` up repeat to three figures. At `n = 50` this draw
    /// no longer fits inside [`MAX_SEARCH_NODES`], which is the other half of the same measurement:
    /// what grows with the dimension is the tree, and the per-node saving is second-order against
    /// it.
    ///
    /// So the crossover is near thirty ambiguities, and the vendor's "~20× at dimension 200" —
    /// measured on their MATLAB, where every saved flop is an interpreted operation — does not
    /// survive translation into compiled code. The reason is visible in the two loops: the flops
    /// the method saves are in `conditional_float`'s summation, which is contiguous and which the
    /// compiler vectorises, and it replaces them with a serial recurrence that it cannot. What is
    /// left is a real but modest win exactly where the wall clock is — at `n = 40` the saving is
    /// 2.4 ms per search — and a loss of a few hundred nanoseconds where a search costs a
    /// microsecond anyway.
    #[test]
    #[ignore = "a measurement, not a check: see the table in the doc comment"]
    #[expect(
        clippy::cast_precision_loss,
        reason = "a draw in [-1, 1) and a repetition count, neither near 2^53"
    )]
    fn the_reduced_conditioning_pays_above_thirty_ambiguities() {
        fn splitmix(s: &mut u64) -> f64 {
            *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = *s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            let u = z ^ (z >> 31);
            (u >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
        }
        for n in [5_usize, 10, 20, 30, 40, 50, 60, 70, 80] {
            let mut s = 0x5EED_0000_0000_0001_u64;
            let entries: Vec<f64> = (0..n * n + n).map(|_| splitmix(&mut s)).collect();
            let q = covariance_from(&entries, n, 0.6);
            let floats: Vec<f64> = (0..n).map(|_| splitmix(&mut s) * 20.0).collect();
            let r = Reduction::new(&floats, &q).unwrap();
            if r.search(1).is_err() {
                println!("n={n:3}  the ellipsoid outgrew the node budget");
                break;
            }
            let reps = 20_usize.max(2000 / n);
            let mut ta = std::time::Duration::ZERO;
            let mut tb = std::time::Duration::ZERO;
            let mut rounds = 0;
            for _ in 0..4 {
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(r.search(1).unwrap());
                }
                ta += t.elapsed();
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    std::hint::black_box(recomputed_search(&r, 1));
                }
                tb += t.elapsed();
                rounds += reps;
            }
            let a = ta.as_secs_f64() / rounds as f64;
            let b = tb.as_secs_f64() / rounds as f64;
            println!(
                "n={n:3}  new {:9.3} us   old {:9.3} us   x{:.2}",
                a * 1e6,
                b * 1e6,
                b / a
            );
        }
    }
}
