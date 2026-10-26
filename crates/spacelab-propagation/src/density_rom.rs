//! Reduced-order thermospheric density modelling (ML-ROM) — Proper Orthogonal Decomposition of a set
//! of density-field snapshots plus a learned linear map from space-weather drivers to the reduced
//! coordinates. This is the Mehta–Linares / Gondelach–Linares reduced-order-model approach, built on
//! **openly published accelerometer densities** (CHAMP / GRACE / GRACE-FO / Swarm): compress the
//! high-dimensional log-density field to a handful of energetic spatial modes, then predict the mode
//! amplitudes from drivers (F10.7, Ap, …). No NRLMSIS 2.x code, data, or patented method is used.
//!
//! [`DensityRom::fit`] runs POD by the method of snapshots (a small `m×m` symmetric eigenproblem via
//! cyclic Jacobi), keeping the top-`rank` modes; [`DensityRom::project`] / [`DensityRom::reconstruct`]
//! move between the full field and its reduced coordinates. [`ReducedDynamics::fit`] learns the
//! driver → reduced-coordinate map by least squares. Everything is a plain linear-algebra regression,
//! so it is exact and verifiable: a synthetic low-rank field round-trips through the POD to machine
//! precision, and a known driver map is recovered by the dynamics fit.

// Index-heavy linear algebra over dynamically-sized matrices; the maths is standard POD / normal
// equations, so short index names and index loops are the clearest form.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::doc_markdown
)]

use alloc::vec;
use alloc::vec::Vec;

/// A POD reduced-order model of a density field: the mean field plus the top spatial modes.
#[derive(Clone, Debug)]
pub struct DensityRom {
    mean: Vec<f64>,
    modes: Vec<Vec<f64>>, // `rank` orthonormal spatial modes, each of length `n`
    eigenvalues: Vec<f64>, // captured mode energies (descending)
    total_energy: f64,    // sum of all snapshot-covariance eigenvalues
}

impl DensityRom {
    /// Fit a POD basis to `snapshots` (each a length-`n` field sample, e.g. log-density over a grid),
    /// keeping the `rank` most energetic modes.
    ///
    /// # Errors
    /// Returns `None` if there are fewer than two snapshots, the snapshots differ in length, or
    /// `rank` exceeds the number of snapshots.
    #[must_use]
    pub fn fit(snapshots: &[Vec<f64>], rank: usize) -> Option<Self> {
        let m = snapshots.len();
        if m < 2 || rank == 0 || rank > m {
            return None;
        }
        let n = snapshots[0].len();
        if n == 0 || snapshots.iter().any(|s| s.len() != n) {
            return None;
        }

        let mut mean = vec![0.0; n];
        for s in snapshots {
            for i in 0..n {
                mean[i] += s[i];
            }
        }
        for mi in &mut mean {
            *mi /= m as f64;
        }

        // Centred snapshot columns X̃ (n×m), stored column-major as `centred[j][i]`.
        let centred: Vec<Vec<f64>> = snapshots
            .iter()
            .map(|s| (0..n).map(|i| s[i] - mean[i]).collect())
            .collect();

        // Method of snapshots: C = X̃ᵀ X̃ (m×m symmetric), eigendecomposed by Jacobi.
        let mut cov = vec![vec![0.0; m]; m];
        for a in 0..m {
            for b in a..m {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += centred[a][i] * centred[b][i];
                }
                cov[a][b] = acc;
                cov[b][a] = acc;
            }
        }
        let (evals, evecs) = jacobi_symmetric(&cov);
        let total_energy: f64 = evals.iter().sum();

        // Descending eigenvalue order.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| {
            evals[b]
                .partial_cmp(&evals[a])
                .unwrap_or(core::cmp::Ordering::Equal)
        });

        // Keep only genuinely energetic modes: a requested rank beyond the data's numerical rank
        // yields fewer modes rather than garbage from dividing by a near-zero eigenvalue.
        let lambda_max = evals[order[0]].max(0.0);
        let cutoff = 1e-12 * lambda_max;

        let mut modes = Vec::with_capacity(rank);
        let mut eigenvalues = Vec::with_capacity(rank);
        for &e in order.iter().take(rank) {
            let lambda = evals[e];
            if lambda <= cutoff {
                break;
            }
            // Spatial mode φ = X̃ v / √λ (length n), orthonormal.
            let scale = 1.0 / libm::sqrt(lambda);
            let mut phi = vec![0.0; n];
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..m {
                    acc += centred[j][i] * evecs[j][e];
                }
                phi[i] = acc * scale;
            }
            modes.push(phi);
            eigenvalues.push(lambda);
        }
        if modes.is_empty() {
            return None;
        }

        Some(Self {
            mean,
            modes,
            eigenvalues,
            total_energy,
        })
    }

    /// The number of retained modes.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.modes.len()
    }

    /// The field dimension `n`.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.mean.len()
    }

    /// The mean field.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Project a full field onto the reduced coordinates `αₗ = φₗ·(field − mean)`.
    #[must_use]
    pub fn project(&self, field: &[f64]) -> Vec<f64> {
        self.modes
            .iter()
            .map(|phi| {
                let mut acc = 0.0;
                for i in 0..self.mean.len() {
                    acc += phi[i] * (field[i] - self.mean[i]);
                }
                acc
            })
            .collect()
    }

    /// Reconstruct a full field from reduced coordinates: `mean + Σ αₗ φₗ`.
    #[must_use]
    pub fn reconstruct(&self, coords: &[f64]) -> Vec<f64> {
        let mut out = self.mean.clone();
        for (phi, &a) in self.modes.iter().zip(coords) {
            for i in 0..out.len() {
                out[i] += a * phi[i];
            }
        }
        out
    }

    /// The fraction of total field variance captured by the retained modes, in `[0, 1]`.
    #[must_use]
    pub fn energy_fraction(&self) -> f64 {
        if self.total_energy <= 0.0 {
            return 0.0;
        }
        self.eigenvalues.iter().sum::<f64>() / self.total_energy
    }
}

/// A learned linear map from space-weather drivers to POD reduced coordinates: `α = W · d`. Fit by
/// least squares against paired driver / reduced-coordinate rows — the "dynamics" layer of the ROM.
#[derive(Clone, Debug)]
pub struct ReducedDynamics {
    weights: Vec<Vec<f64>>, // `rank × drivers`
}

impl ReducedDynamics {
    /// Fit `W` so that `coord_rows[k] ≈ W · driver_rows[k]` in the least-squares sense. Include a
    /// constant `1.0` in each driver row for an intercept.
    ///
    /// # Errors
    /// Returns `None` on shape mismatch or a singular normal-equations system (too few / degenerate
    /// driver rows).
    #[must_use]
    pub fn fit(driver_rows: &[Vec<f64>], coord_rows: &[Vec<f64>]) -> Option<Self> {
        let k = driver_rows.len();
        if k == 0 || k != coord_rows.len() {
            return None;
        }
        let p = driver_rows[0].len();
        let r = coord_rows[0].len();
        if p == 0
            || r == 0
            || driver_rows.iter().any(|d| d.len() != p)
            || coord_rows.iter().any(|c| c.len() != r)
        {
            return None;
        }

        // Normal equations DᵀD (p×p) shared across all `r` coordinate regressions.
        let mut dtd = vec![vec![0.0; p]; p];
        for row in driver_rows {
            for a in 0..p {
                for b in 0..p {
                    dtd[a][b] += row[a] * row[b];
                }
            }
        }
        let mut weights = Vec::with_capacity(r);
        for c in 0..r {
            let mut dtc = vec![0.0; p];
            for (row, coord) in driver_rows.iter().zip(coord_rows) {
                for a in 0..p {
                    dtc[a] += row[a] * coord[c];
                }
            }
            let w = solve_linear(&dtd, &dtc)?;
            weights.push(w);
        }
        Some(Self { weights })
    }

    /// Predict the reduced coordinates for a driver row.
    #[must_use]
    pub fn predict(&self, drivers: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .map(|w| {
                let mut acc = 0.0;
                for a in 0..w.len().min(drivers.len()) {
                    acc += w[a] * drivers[a];
                }
                acc
            })
            .collect()
    }
}

/// Dynamic Mode Decomposition with control (Proctor, Brunton & Kutz 2016): a reduced-order **linear
/// dynamical system** `aₖ₊₁ = A aₖ + B uₖ` for the POD reduced coordinates `a` driven by controls `u`
/// (space-weather drivers). Where [`ReducedDynamics`] is a memoryless map `u → a`, DMDc carries the
/// previous state, so it captures the thermosphere's **inertia** — density relaxes toward the driver
/// forcing with a lag rather than tracking it instantly, which `A` (the free-decay operator) encodes.
///
/// `[A | B]` is fit by least squares from snapshot triples `(aₖ, uₖ, aₖ₊₁)`: regress `aₖ₊₁` on the
/// stacked regressor `[aₖ; uₖ]` through the shared normal equations, the same orthogonal-free path as
/// [`ReducedDynamics`]. Exact and verifiable — a trajectory from a known `(A, B)` recovers it.
#[derive(Clone, Debug)]
pub struct DmdcModel {
    state_dim: usize,        // r — reduced-coordinate dimension
    control_dim: usize,      // p — control (driver) dimension
    operator: Vec<Vec<f64>>, // `[A | B]`, `r` rows × `(r + p)` columns
}

impl DmdcModel {
    /// Fit `[A | B]` so that `next_coords[k] ≈ A·prev_coords[k] + B·controls[k]` in the least-squares
    /// sense. The three slices must have equal length `k`, with `prev_coords`/`next_coords` rows of the
    /// reduced dimension `r` and `controls` rows of the driver dimension `p` (add a constant `1.0`
    /// control for an affine offset).
    ///
    /// # Errors
    /// Returns `None` on a length or width mismatch, or a singular normal-equations system (too few or
    /// degenerate samples — `k` must give a full-rank `[aₖ; uₖ]` regressor, so `k ≥ r + p`).
    #[must_use]
    pub fn fit(
        prev_coords: &[Vec<f64>],
        controls: &[Vec<f64>],
        next_coords: &[Vec<f64>],
    ) -> Option<Self> {
        let k = prev_coords.len();
        if k == 0 || controls.len() != k || next_coords.len() != k {
            return None;
        }
        let r = prev_coords[0].len();
        let p = controls[0].len();
        if r == 0
            || p == 0
            || prev_coords.iter().any(|a| a.len() != r)
            || next_coords.iter().any(|a| a.len() != r)
            || controls.iter().any(|u| u.len() != p)
        {
            return None;
        }
        let m = r + p;

        // Stacked regressors zₖ = [aₖ; uₖ] and the shared normal matrix ZᵀZ (m×m).
        let regressors: Vec<Vec<f64>> = (0..k)
            .map(|i| prev_coords[i].iter().chain(&controls[i]).copied().collect())
            .collect();
        let mut ztz = vec![vec![0.0; m]; m];
        for z in &regressors {
            for a in 0..m {
                for b in 0..m {
                    ztz[a][b] += z[a] * z[b];
                }
            }
        }

        // One right-hand side per output coordinate; solve for that row of `[A | B]`.
        let mut operator = Vec::with_capacity(r);
        for c in 0..r {
            let mut ztc = vec![0.0; m];
            for (z, next) in regressors.iter().zip(next_coords) {
                for a in 0..m {
                    ztc[a] += z[a] * next[c];
                }
            }
            operator.push(solve_linear(&ztz, &ztc)?);
        }
        Some(Self {
            state_dim: r,
            control_dim: p,
            operator,
        })
    }

    /// The reduced state one step ahead: `A·state + B·control`.
    #[must_use]
    pub fn predict_next(&self, state: &[f64], control: &[f64]) -> Vec<f64> {
        self.operator
            .iter()
            .map(|w| {
                let mut acc = 0.0;
                for a in 0..self.state_dim {
                    acc += w[a] * state.get(a).copied().unwrap_or(0.0);
                }
                for b in 0..self.control_dim {
                    acc += w[self.state_dim + b] * control.get(b).copied().unwrap_or(0.0);
                }
                acc
            })
            .collect()
    }

    /// The free-dynamics operator `A` (`r × r`): its spectrum is the set of DMD modes (growth/decay
    /// rates and oscillation of the reduced coordinates under zero forcing).
    #[must_use]
    pub fn system_matrix(&self) -> Vec<Vec<f64>> {
        self.operator
            .iter()
            .map(|row| row[..self.state_dim].to_vec())
            .collect()
    }

    /// The control operator `B` (`r × p`): how the drivers force the reduced coordinates.
    #[must_use]
    pub fn control_matrix(&self) -> Vec<Vec<f64>> {
        self.operator
            .iter()
            .map(|row| row[self.state_dim..].to_vec())
            .collect()
    }
}

/// Cyclic Jacobi eigendecomposition of a symmetric matrix `a` (n×n). Returns `(eigenvalues,
/// eigenvectors)`, the eigenvectors as columns `evecs[i][k]` (row `i`, eigenvector `k`).
#[must_use]
pub fn jacobi_symmetric(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p][q] * m[p][q];
            }
        }
        if off < 1e-300 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if libm::fabs(m[p][q]) < 1e-300 {
                    continue;
                }
                let theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
                let t = libm::copysign(1.0, theta)
                    / (libm::fabs(theta) + libm::sqrt(theta * theta + 1.0));
                let c = 1.0 / libm::sqrt(t * t + 1.0);
                let s = t * c;
                for i in 0..n {
                    let mip = m[i][p];
                    let miq = m[i][q];
                    m[i][p] = c * mip - s * miq;
                    m[i][q] = s * mip + c * miq;
                }
                for i in 0..n {
                    let mpi = m[p][i];
                    let mqi = m[q][i];
                    m[p][i] = c * mpi - s * mqi;
                    m[q][i] = s * mpi + c * mqi;
                }
                for i in 0..n {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }
    let evals = (0..n).map(|i| m[i][i]).collect();
    (evals, v)
}

/// Solve `a x = b` for a dense `n×n` system by Gaussian elimination with partial pivoting. `None` if
/// the matrix is singular.
fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = a[i].clone();
            row.push(b[i]);
            row
        })
        .collect();
    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if libm::fabs(aug[r][col]) > libm::fabs(aug[pivot][col]) {
                pivot = r;
            }
        }
        if libm::fabs(aug[pivot][col]) < 1e-300 {
            return None;
        }
        aug.swap(col, pivot);
        let inv = 1.0 / aug[col][col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = aug[r][col] * inv;
            if f != 0.0 {
                for k in col..=n {
                    aug[r][k] -= f * aug[col][k];
                }
            }
        }
    }
    Some((0..n).map(|i| aug[i][n] / aug[i][i]).collect())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp, clippy::cast_lossless)]

    use super::*;

    // Two orthonormal length-n patterns (sine/cosine), for a synthetic rank-2 field.
    fn modes(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut p1: Vec<f64> = (0..n)
            .map(|i| libm::sin(core::f64::consts::TAU * i as f64 / n as f64))
            .collect();
        let mut p2: Vec<f64> = (0..n)
            .map(|i| libm::cos(core::f64::consts::TAU * i as f64 / n as f64))
            .collect();
        for p in [&mut p1, &mut p2] {
            let norm = libm::sqrt(p.iter().map(|v| v * v).sum::<f64>());
            for v in p.iter_mut() {
                *v /= norm;
            }
        }
        (p1, p2)
    }

    /// The Jacobi eigensolver against a spectrum known in closed form, not against itself.
    ///
    /// The `[−1, 2, −1]` tridiagonal Toeplitz has eigenvalues `2 − 2 cos(kπ/(n+1))`, which for
    /// `n = 3` is `{2 − √2, 2, 2 + √2}` — confirmed against `charpoly` in Maxima at 60 digits. Its
    /// uniform diagonal is what makes the rotation angle observable: the true `θ` is
    /// `(m[q][q] − m[p][p]) / 2m[p][q] = 0`, so any sign or operator slip in that numerator moves
    /// the very first rotation off 45°. Every other test in this file reaches the solver through
    /// `fit`, and asserts only properties invariant under the choice of basis inside a correct
    /// invariant subspace, so none of them can see the angle at all.
    #[test]
    fn the_jacobi_eigensolver_matches_a_spectrum_known_in_closed_form() {
        let a = vec![
            vec![2.0, -1.0, 0.0],
            vec![-1.0, 2.0, -1.0],
            vec![0.0, -1.0, 2.0],
        ];
        let root2 = libm::sqrt(2.0);
        let (evals, evecs) = jacobi_symmetric(&a);

        let mut sorted = evals.clone();
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (got, want) in sorted.iter().zip([2.0 - root2, 2.0, 2.0 + root2]) {
            assert!((got - want).abs() < 1e-13, "eigenvalue {got}, want {want}");
        }

        // Each column is an eigenvector of `a` for its own eigenvalue, and the columns are
        // orthonormal — a rotation with the wrong angle keeps the second property and loses the
        // first.
        for k in 0..3 {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i][j] * evecs[j][k]).sum();
                let lv = evals[k] * evecs[i][k];
                assert!(
                    (av - lv).abs() < 1e-13,
                    "(A v - lambda v)[{i}] for mode {k}"
                );
            }
            for l in 0..3 {
                let dot: f64 = (0..3).map(|i| evecs[i][k] * evecs[i][l]).sum();
                let want = f64::from(u8::from(k == l));
                assert!((dot - want).abs() < 1e-13, "v{k}.v{l} = {dot}, want {want}");
            }
        }
    }

    /// The retained energy of a field whose rank exceeds the modes kept is a fraction the
    /// construction fixes exactly, not "very nearly one".
    ///
    /// Four snapshots built from three orthonormal spatial patterns with mutually orthogonal,
    /// mean-zero temporal patterns of squared norm 4 give `C = Σ aₖ² tₖtₖᵀ` and hence eigenvalues
    /// `4aₖ²` — here 16, 9 and 6.25, every one of them a dyadic rational and so exact in binary.
    /// Keeping two of the three retains 25 of 31.25, or 0.8. The other two energy assertions in
    /// this file both use a field that lies *entirely* in the retained subspace, where the true
    /// fraction is 1.0 and a function replaced by the constant `1.0` is indistinguishable from it.
    #[test]
    fn the_retained_energy_is_the_fraction_the_construction_fixes() {
        let psi = [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
        ];
        let t = [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ];
        let amp = [2.0, 1.5, 1.25]; // eigenvalues 4a² = 16, 9, 6.25
        let mean = [3.0, -1.0, 0.5, 2.0];

        let snapshots: Vec<Vec<f64>> = (0..4)
            .map(|j| {
                (0..4)
                    .map(|i| mean[i] + (0..3).map(|k| amp[k] * t[k][j] * psi[k][i]).sum::<f64>())
                    .collect()
            })
            .collect();

        let rom = DensityRom::fit(&snapshots, 2).unwrap();
        assert_eq!(rom.rank(), 2);
        let frac = rom.energy_fraction();
        assert!(
            (frac - 0.8).abs() < 1e-12,
            "energy fraction {frac}, want 0.8"
        );

        // And the mode that was dropped is genuinely missing from the reconstruction, so the
        // fraction is reporting a real loss rather than an accounting one.
        let residual: f64 = snapshots
            .iter()
            .map(|s| {
                let recon = rom.reconstruct(&rom.project(s));
                s.iter()
                    .zip(&recon)
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum::<f64>()
            })
            .sum();
        assert!(
            (residual - 6.25).abs() < 1e-10,
            "dropped energy {residual}, want 6.25"
        );
    }

    #[test]
    fn pod_recovers_a_low_rank_field_exactly() {
        let n = 40;
        let (phi1, phi2) = modes(n);
        let mean: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * i as f64).collect();

        let snapshots: Vec<Vec<f64>> = (0..12)
            .map(|j| {
                let a = libm::sin(0.5 * j as f64);
                let b = libm::cos(0.3 * j as f64 + 1.0);
                (0..n)
                    .map(|i| mean[i] + a * phi1[i] + b * phi2[i])
                    .collect()
            })
            .collect();

        let rom = DensityRom::fit(&snapshots, 2).unwrap();
        assert_eq!(rom.rank(), 2);
        // The field lives exactly in a rank-2 affine subspace → the top 2 modes capture all variance.
        assert!(
            rom.energy_fraction() > 1.0 - 1e-9,
            "energy {}",
            rom.energy_fraction()
        );
        // Every snapshot reconstructs to machine precision.
        for s in &snapshots {
            let recon = rom.reconstruct(&rom.project(s));
            let err: f64 = s.iter().zip(&recon).map(|(a, b)| (a - b) * (a - b)).sum();
            assert!(err < 1e-18, "reconstruction error {err}");
        }
    }

    /// The energetic-mode cutoff is a fraction of the largest eigenvalue, so the rank a field
    /// reduces to cannot depend on the units it is measured in. Written as `1e-12 / λ_max` it does:
    /// scaling the field by 1e-4 scales every eigenvalue by 1e-8, and an inverse cutoff climbs to
    /// 1e-4 — above every mode a scaled-down field has. Densities arrive in kg/m³, of order 1e-12,
    /// which is exactly the regime where an absolute cutoff would swallow the whole spectrum.
    #[test]
    fn the_mode_cutoff_is_relative_so_the_rank_does_not_depend_on_the_units() {
        let n = 30;
        let (phi1, phi2) = modes(n);
        let snapshots: Vec<Vec<f64>> = (0..8)
            .map(|j| {
                let a = libm::sin(0.5 * j as f64);
                let b = libm::cos(0.4 * j as f64 + 0.7);
                (0..n).map(|i| a * phi1[i] + b * phi2[i]).collect()
            })
            .collect();
        let unscaled = DensityRom::fit(&snapshots, 2).unwrap();
        assert_eq!(unscaled.rank(), 2, "the fixture must be genuinely rank 2");

        for scale in [1.0e-4_f64, 1.0e4] {
            let rescaled: Vec<Vec<f64>> = snapshots
                .iter()
                .map(|s| s.iter().map(|v| v * scale).collect())
                .collect();
            let rom = DensityRom::fit(&rescaled, 2).unwrap();
            assert_eq!(rom.rank(), 2, "rank changed at scale {scale:e}");
        }
    }

    #[test]
    fn pod_modes_are_orthonormal() {
        let n = 30;
        let (phi1, phi2) = modes(n);
        // Affinely-independent coefficients so the centred data is genuinely rank 2.
        let snapshots: Vec<Vec<f64>> = (0..8)
            .map(|j| {
                let a = libm::sin(0.5 * j as f64);
                let b = libm::cos(0.4 * j as f64 + 0.7);
                (0..n).map(|i| a * phi1[i] + b * phi2[i]).collect()
            })
            .collect();
        let rom = DensityRom::fit(&snapshots, 2).unwrap();
        let m = &rom.modes;
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        assert!((dot(&m[0], &m[0]) - 1.0).abs() < 1e-9);
        assert!((dot(&m[1], &m[1]) - 1.0).abs() < 1e-9);
        assert!(dot(&m[0], &m[1]).abs() < 1e-9);
    }

    #[test]
    fn reduced_dynamics_recovers_a_known_linear_map() {
        // coords = W·drivers, W = [[2, -1, 0.5], [0, 3, -2]] (last driver is the intercept 1.0).
        let w = [[2.0, -1.0, 0.5], [0.0, 3.0, -2.0]];
        let drivers: Vec<Vec<f64>> = (0..20)
            .map(|k| vec![libm::sin(k as f64), libm::cos(0.7 * k as f64), 1.0])
            .collect();
        let coords: Vec<Vec<f64>> = drivers
            .iter()
            .map(|d| {
                vec![
                    w[0][0] * d[0] + w[0][1] * d[1] + w[0][2] * d[2],
                    w[1][0] * d[0] + w[1][1] * d[1] + w[1][2] * d[2],
                ]
            })
            .collect();

        let dyn_model = ReducedDynamics::fit(&drivers, &coords).unwrap();
        let pred = dyn_model.predict(&[0.4, -0.2, 1.0]);
        let expect = [2.0 * 0.4 - 1.0 * -0.2 + 0.5, 3.0 * -0.2 - 2.0];
        assert!(
            (pred[0] - expect[0]).abs() < 1e-9 && (pred[1] - expect[1]).abs() < 1e-9,
            "{pred:?}"
        );
    }

    /// Least squares on data no linear map fits: the residual is orthogonal to every driver column,
    /// the defining equation `Dᵀ(y − D w) = 0`. (G2)
    #[test]
    fn reduced_dynamics_leaves_a_residual_orthogonal_to_every_driver() {
        let drivers: Vec<Vec<f64>> = (0..200)
            .map(|k| {
                let t = f64::from(k);
                vec![
                    1.0,
                    libm::sin(0.3 * t),
                    libm::cos(1.1 * t),
                    (t / 200.0).powi(2),
                ]
            })
            .collect();
        // A term outside the span: the fit cannot reach it, so the residual is not zero.
        let coords: Vec<Vec<f64>> = drivers
            .iter()
            .enumerate()
            .map(|(k, d)| vec![0.5 - 2.0 * d[1] + 3.0 * d[3] + libm::sin(7.3 * k as f64)])
            .collect();
        let model = ReducedDynamics::fit(&drivers, &coords).unwrap();
        let residual: Vec<f64> = drivers
            .iter()
            .zip(&coords)
            .map(|(d, c)| c[0] - model.predict(d)[0])
            .collect();
        assert!(residual.iter().map(|r| r * r).sum::<f64>() > 1.0);
        for a in 0..4 {
            let along: f64 = drivers.iter().zip(&residual).map(|(d, r)| d[a] * r).sum();
            let scale: f64 = drivers
                .iter()
                .zip(&coords)
                .map(|(d, c)| (d[a] * c[0]).abs())
                .sum();
            assert!(
                along.abs() <= 1e-12 * scale,
                "column {a}: {along:e} of {scale:e}"
            );
        }
    }

    #[test]
    fn dmdc_recovers_a_known_linear_system() {
        // aₖ₊₁ = A aₖ + B uₖ, A = [[0.9, 0.1], [-0.2, 0.8]], B = [[0.5], [-0.3]], driven by sin(0.6k).
        let a = [[0.9, 0.1], [-0.2, 0.8]];
        let b = [0.5, -0.3];
        let mut state = vec![1.0, 0.0];
        let (mut prev, mut ctrl, mut next) = (Vec::new(), Vec::new(), Vec::new());
        for k in 0..30 {
            let u = libm::sin(0.6 * k as f64);
            let n0 = a[0][0] * state[0] + a[0][1] * state[1] + b[0] * u;
            let n1 = a[1][0] * state[0] + a[1][1] * state[1] + b[1] * u;
            prev.push(state.clone());
            ctrl.push(vec![u]);
            next.push(vec![n0, n1]);
            state = vec![n0, n1];
        }

        let model = DmdcModel::fit(&prev, &ctrl, &next).unwrap();
        let (am, bm) = (model.system_matrix(), model.control_matrix());
        assert!(
            (am[0][0] - 0.9).abs() < 1e-9 && (am[0][1] - 0.1).abs() < 1e-9,
            "{am:?}"
        );
        assert!(
            (am[1][0] + 0.2).abs() < 1e-9 && (am[1][1] - 0.8).abs() < 1e-9,
            "{am:?}"
        );
        assert!(
            (bm[0][0] - 0.5).abs() < 1e-9 && (bm[1][0] + 0.3).abs() < 1e-9,
            "{bm:?}"
        );

        // One-step prediction reproduces the true update, memory (A) included.
        let pred = model.predict_next(&[0.3, -0.4], &[0.7]);
        let truth = [
            0.9 * 0.3 + 0.1 * -0.4 + 0.5 * 0.7,
            -0.2 * 0.3 + 0.8 * -0.4 - 0.3 * 0.7,
        ];
        assert!(
            (pred[0] - truth[0]).abs() < 1e-9 && (pred[1] - truth[1]).abs() < 1e-9,
            "{pred:?}"
        );
    }

    #[test]
    fn jacobi_recovers_known_eigenvalues() {
        // Symmetric matrix with eigenvalues 2 and 4 (a[0][0]=a[1][1]=3, a[0][1]=1).
        let a = vec![vec![3.0, 1.0], vec![1.0, 3.0]];
        let (mut evals, _) = jacobi_symmetric(&a);
        evals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!(
            (evals[0] - 2.0).abs() < 1e-12 && (evals[1] - 4.0).abs() < 1e-12,
            "{evals:?}"
        );
    }

    #[test]
    fn jacobi_returns_eigenvectors_and_survives_a_repeated_eigenvalue() {
        // Eigenvalues alone say almost nothing about the rotation: every sweep is an orthogonal
        // similarity, so the spectrum is preserved however the angle is chosen. The eigen*vector*
        // relation is what pins it.
        let a = vec![
            vec![4.0, 1.0, -2.0],
            vec![1.0, 3.0, 0.5],
            vec![-2.0, 0.5, 6.0],
        ];
        let (evals, evecs) = jacobi_symmetric(&a);
        for k in 0..3 {
            // A v = λ v, with v the k-th column of the accumulated rotation.
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i][j] * evecs[j][k]).sum();
                assert!(
                    (av - evals[k] * evecs[i][k]).abs() < 1e-10,
                    "A v ≠ λ v at ({i},{k}): {av} vs {}",
                    evals[k] * evecs[i][k]
                );
            }
            let norm: f64 = (0..3).map(|i| evecs[i][k] * evecs[i][k]).sum();
            assert!((norm - 1.0).abs() < 1e-12, "column {k} norm {norm}");
        }

        // An already-diagonal matrix with a repeated eigenvalue: the sweep's `off` is zero and it
        // returns before touching a rotation at all.
        let (evals, evecs) = jacobi_symmetric(&[vec![2.0, 0.0], vec![0.0, 2.0]]);
        assert!(evals.iter().all(|l| (l - 2.0).abs() < 1e-12), "{evals:?}");
        assert!(evecs.iter().flatten().all(|x| x.is_finite()), "{evecs:?}");

        // The per-pair skip is a separate guard, and reaching it needs a matrix that is *not*
        // diagonal: here `off` is nonzero because of the (0,2) coupling, yet the sweep still visits
        // (0,1), where the off-diagonal is exactly zero and the two diagonal entries are equal. The
        // rotation angle there is `0.0 / 0.0`, and one NaN poisons the whole accumulated basis.
        let a = vec![
            vec![2.0, 0.0, 1.0],
            vec![0.0, 2.0, 0.0],
            vec![1.0, 0.0, 5.0],
        ];
        let (mut evals, evecs) = jacobi_symmetric(&a);
        assert!(evecs.iter().flatten().all(|x| x.is_finite()), "{evecs:?}");
        evals.sort_by(|x, y| x.partial_cmp(y).unwrap());
        let root13 = libm::sqrt(13.0);
        for (got, want) in
            evals
                .iter()
                .zip([f64::midpoint(7.0, -root13), 2.0, f64::midpoint(7.0, root13)])
        {
            assert!((got - want).abs() < 1e-12, "{evals:?}");
        }
    }

    #[test]
    fn solve_linear_pivots_and_refuses_a_singular_system() {
        // A zero in the first pivot position: only a search for the *largest* magnitude finds the
        // row that can lead. Picking the smallest, or only an exact tie, leaves a zero pivot and
        // refuses a perfectly well-posed system.
        let a = vec![vec![0.0, 2.0], vec![4.0, 1.0]];
        let x = solve_linear(&a, &[6.0, 5.0]).unwrap();
        assert!(
            (x[0] - 0.5).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12,
            "{x:?}"
        );
        // Rank-deficient: the second row is twice the first, so no pivot survives elimination.
        assert!(solve_linear(&[vec![1.0, 2.0], vec![2.0, 4.0]], &[3.0, 6.0]).is_none());
        // And an outright zero column in the first position, which the guard must reject rather
        // than divide by.
        assert!(solve_linear(&[vec![0.0, 0.0], vec![0.0, 1.0]], &[1.0, 1.0]).is_none());
    }

    #[test]
    fn every_degenerate_pod_input_is_refused_and_the_boundaries_are_not() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 1.0, 0.5, 2.0];
        // One reason at a time, so a rule that demanded all of them together would let each pass.
        assert!(DensityRom::fit(&[], 1).is_none(), "no snapshots");
        assert!(
            DensityRom::fit(core::slice::from_ref(&a), 1).is_none(),
            "one snapshot"
        );
        assert!(
            DensityRom::fit(&[a.clone(), b.clone()], 0).is_none(),
            "rank 0"
        );
        assert!(
            DensityRom::fit(&[a.clone(), b.clone()], 3).is_none(),
            "rank above the snapshot count"
        );
        assert!(
            DensityRom::fit(&[vec![], vec![]], 1).is_none(),
            "empty fields"
        );
        assert!(
            DensityRom::fit(&[a.clone(), vec![1.0, 2.0]], 1).is_none(),
            "ragged fields"
        );
        // The two boundaries the guards must not reject: the smallest usable snapshot count, and a
        // rank equal to it.
        assert!(DensityRom::fit(&[a.clone(), b.clone()], 2).is_some());
        assert!(DensityRom::fit(&[a, b], 1).is_some());
    }

    #[test]
    fn every_degenerate_dynamics_input_is_refused() {
        let d = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let c = vec![vec![2.0], vec![3.0], vec![5.0]];
        assert!(
            ReducedDynamics::fit(&d, &c).is_some(),
            "the well-posed case"
        );
        assert!(ReducedDynamics::fit(&[], &[]).is_none(), "no rows");
        assert!(
            ReducedDynamics::fit(&d, &c[..2]).is_none(),
            "row counts differ"
        );
        assert!(
            ReducedDynamics::fit(&[vec![], vec![]], &[vec![1.0], vec![2.0]]).is_none(),
            "no drivers"
        );
        assert!(
            ReducedDynamics::fit(&[vec![1.0], vec![1.0]], &[vec![], vec![]]).is_none(),
            "no coordinates"
        );
        let ragged_d = vec![vec![1.0, 0.0], vec![0.0], vec![1.0, 1.0]];
        assert!(
            ReducedDynamics::fit(&ragged_d, &c).is_none(),
            "ragged drivers"
        );
        let ragged_c = vec![vec![2.0], vec![3.0, 0.0], vec![5.0]];
        assert!(
            ReducedDynamics::fit(&d, &ragged_c).is_none(),
            "ragged coords"
        );

        let prev = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let ctrl = vec![vec![1.0], vec![2.0], vec![0.5]];
        let next = vec![vec![0.5, 0.5], vec![0.25, 1.0], vec![0.75, 1.2]];
        assert!(
            DmdcModel::fit(&prev, &ctrl, &next).is_some(),
            "the well-posed case"
        );
        assert!(DmdcModel::fit(&[], &[], &[]).is_none(), "no rows");
        assert!(
            DmdcModel::fit(&prev, &ctrl[..2], &next).is_none(),
            "control count"
        );
        assert!(
            DmdcModel::fit(&prev, &ctrl, &next[..2]).is_none(),
            "next count"
        );
        assert!(
            DmdcModel::fit(
                &[vec![], vec![]],
                &[vec![1.0], vec![1.0]],
                &[vec![], vec![]]
            )
            .is_none(),
            "no coordinates"
        );
        assert!(
            DmdcModel::fit(&prev[..2], &[vec![], vec![]], &next[..2]).is_none(),
            "no controls"
        );
        let ragged_prev = vec![vec![1.0, 0.0], vec![0.0], vec![1.0, 1.0]];
        assert!(
            DmdcModel::fit(&ragged_prev, &ctrl, &next).is_none(),
            "ragged prev"
        );
        let ragged_next = vec![vec![0.5, 0.5], vec![0.25], vec![0.75, 1.2]];
        assert!(
            DmdcModel::fit(&prev, &ctrl, &ragged_next).is_none(),
            "ragged next"
        );
        let ragged_ctrl = vec![vec![1.0], vec![2.0, 0.0], vec![0.5]];
        assert!(
            DmdcModel::fit(&prev, &ragged_ctrl, &next).is_none(),
            "ragged control"
        );
    }

    #[test]
    fn the_rom_reports_the_field_it_was_fitted_to() {
        let (p1, p2) = modes(16);
        let snaps: Vec<Vec<f64>> = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.5), (0.3, -0.8)]
            .iter()
            .map(|&(a, b)| (0..16).map(|i| 10.0 + a * p1[i] + b * p2[i]).collect())
            .collect();
        let rom = DensityRom::fit(&snaps, 2).unwrap();

        assert_eq!(rom.dimension(), 16);
        assert_eq!(rom.mean().len(), 16);
        // The snapshots share the constant offset 10 and their mode amplitudes sum to (0.3, 0.7)/4.
        for (i, mi) in rom.mean().iter().enumerate() {
            let want = 10.0 + 0.3 / 4.0 * p1[i] + 0.7 / 4.0 * p2[i];
            assert!((mi - want).abs() < 1e-12, "mean[{i}] = {mi} vs {want}");
        }
        // Zero reduced coordinates reconstruct the mean itself, component by component.
        let at_origin = rom.reconstruct(&[0.0, 0.0]);
        assert_eq!(at_origin.len(), 16);
        assert!(
            at_origin
                .iter()
                .zip(rom.mean())
                .all(|(a, b)| (a - b).abs() < 1e-15)
        );
        // A rank-2 field is captured entirely, so the retained energy is the whole of it.
        let frac = rom.energy_fraction();
        assert!((frac - 1.0).abs() < 1e-10, "energy fraction {frac}");

        // Asking for more modes than the data has stops at its numerical rank rather than dividing
        // by a near-zero eigenvalue, and the cutoff that decides that scales *with* the largest
        // eigenvalue.
        let over = DensityRom::fit(&snaps, 4).unwrap();
        assert_eq!(over.rank(), 2, "eigenvalues {:?}", over.energy_fraction());
    }

    #[test]
    fn dmdc_reads_each_control_component_from_its_own_column() {
        // `[A | B]` with two controls, so the second control column is not the first: an operator
        // index that walked backwards from the state block would read `A` again.
        let model = DmdcModel {
            state_dim: 2,
            control_dim: 2,
            operator: vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        };
        let pred = model.predict_next(&[7.0, -3.0], &[2.0, 5.0]);
        assert_eq!(pred, vec![2.0, 5.0], "{pred:?}");
    }
}
