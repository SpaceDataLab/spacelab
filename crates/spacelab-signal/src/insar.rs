//! SAR interferometry (InSAR) — the phase difference of two co-registered SAR acquisitions, its
//! coherence, and phase unwrapping.
//!
//! Two passes over the same scene give complex images `s₁`, `s₂`; their **interferogram**
//! `s₁·conj(s₂)` carries a phase proportional to the range difference (topography / deformation).
//! That phase is measured modulo `2π`, so it must be **unwrapped**; and its reliability is the
//! **coherence** `γ = |⟨s₁ conj(s₂)⟩| / √(⟨|s₁|²⟩⟨|s₂|²⟩) ∈ [0, 1]`. [`remove_reference_phase`]
//! subtracts a modelled flat-earth/topographic phase before unwrapping.
//!
//! [`unwrap_phase_1d`] is Itoh integration; [`unwrap_phase_2d`] a path-following unwrap (unwrap the
//! first column, then each row) — exact for a residue-free (well-sampled) field, but path-dependent
//! (and so wrong) when the field has **residues**. [`residues`] locates them (the Goldstein loop
//! integral around every 2×2 cell), and [`unwrap_phase_2d_lstsq`] is the residue-robust alternative:
//! the unweighted least-squares (minimum-Lᵨ-norm) unwrap of Ghiglia & Romero (1994), which solves the
//! discrete Poisson equation `∇²φ = ρ` (`ρ` the divergence of the wrapped gradients, Neumann boundary)
//! by the discrete cosine transform. [`unwrap_phase_2d_branch_cut`] is the other classic residue-aware
//! unwrapper — Goldstein's: join opposite residues with branch cuts, then flood-fill the phase without
//! crossing one, so no integration path can loop a residue. `alloc` for the output buffers.
//!
//! Gate: `derivations/insar_unwrap.mac` derives the coherence of a signal in orthogonal noise
//! (`1/√(1 + |b|²)`), the `+1` residue of a phase vortex around the loop [`residues`] walks and the
//! zero loop sum of any gradient field, that the Neumann Poisson equation is the normal equation of
//! the least-squares fit, that the DCT-II basis diagonalises the Neumann Laplacian with the divisor
//! the code uses, and that the coded inverse DCT undoes the forward one;
//! `derivations/insar_unwrap.gp` re-checks the last two and the vortex at 100 digits. The module
//! tests `the_coherence_of_a_signal_in_orthogonal_noise_is_its_closed_form` (G3),
//! `residues_vanish_on_a_smooth_field_and_ring_a_singularity` (G3),
//! `lstsq_solves_the_discrete_poisson_equation` (G2) and
//! `the_inverse_dct_undoes_the_forward_one_at_every_length` (G1) hold the shipped code to them.

// Grid indices cast usize↔isize/f64 in range; short names (r, c) are the standard raster notation.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]

use crate::fft::Complex;
use alloc::vec;
use alloc::vec::Vec;

const PI: f64 = core::f64::consts::PI;
const TAU: f64 = core::f64::consts::TAU;

/// Wrap a phase to `(−π, π]`.
#[must_use]
pub fn wrap_phase(phi: f64) -> f64 {
    let x = libm::fmod(phi + PI, TAU);
    let x = if x <= 0.0 { x + TAU } else { x };
    x - PI
}

/// The interferogram `s₁·conj(s₂)` (elementwise); its argument is the interferometric phase.
#[must_use]
pub fn interferogram(s1: &[Complex], s2: &[Complex]) -> Vec<Complex> {
    s1.iter().zip(s2).map(|(a, b)| *a * b.conj()).collect()
}

/// Per-pixel complex coherence magnitude over a `(2·half_window+1)²` boxcar, in `[0, 1]`. `1` means the
/// two images are identical up to a constant phase; `0` means uncorrelated.
#[must_use]
pub fn coherence(
    s1: &[Complex],
    s2: &[Complex],
    rows: usize,
    cols: usize,
    half_window: usize,
) -> Vec<f64> {
    let hw = half_window as isize;
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let mut cross = Complex::ZERO;
            let (mut p1, mut p2) = (0.0, 0.0);
            for dr in -hw..=hw {
                for dc in -hw..=hw {
                    let (rr, cc) = (r as isize + dr, c as isize + dc);
                    if rr >= 0 && rr < rows as isize && cc >= 0 && cc < cols as isize {
                        let idx = rr as usize * cols + cc as usize;
                        cross = cross + s1[idx] * s2[idx].conj();
                        p1 += s1[idx].norm_sqr();
                        p2 += s2[idx].norm_sqr();
                    }
                }
            }
            let denom = libm::sqrt(p1 * p2);
            out[r * cols + c] = if denom > 0.0 {
                cross.abs() / denom
            } else {
                0.0
            };
        }
    }
    out
}

/// The residual wrapped phase after subtracting a modelled `reference_phase` (flat-earth / topographic)
/// from an interferogram.
#[must_use]
pub fn remove_reference_phase(ifg: &[Complex], reference_phase: &[f64]) -> Vec<f64> {
    ifg.iter()
        .zip(reference_phase)
        .map(|(z, &r)| wrap_phase(z.arg() - r))
        .collect()
}

/// 1-D Itoh phase unwrapping: integrate the wrapped sample-to-sample differences.
#[must_use]
pub fn unwrap_phase_1d(wrapped: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; wrapped.len()];
    if wrapped.is_empty() {
        return out;
    }
    out[0] = wrapped[0];
    for i in 1..wrapped.len() {
        out[i] = out[i - 1] + wrap_phase(wrapped[i] - wrapped[i - 1]);
    }
    out
}

/// 2-D path-following phase unwrapping (row-major `rows × cols`): unwrap the first column by Itoh, then
/// each row from it. Exact for a residue-free field; residues (aliasing/noise) need a
/// least-squares/branch-cut unwrapper instead.
#[must_use]
pub fn unwrap_phase_2d(wrapped: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    if rows == 0 || cols == 0 {
        return out;
    }
    out[0] = wrapped[0];
    for r in 1..rows {
        out[r * cols] =
            out[(r - 1) * cols] + wrap_phase(wrapped[r * cols] - wrapped[(r - 1) * cols]);
    }
    for r in 0..rows {
        for c in 1..cols {
            let i = r * cols + c;
            out[i] = out[i - 1] + wrap_phase(wrapped[i] - wrapped[i - 1]);
        }
    }
    out
}

/// The Goldstein residue map of a wrapped field (row-major `rows × cols`): for each 2×2 loop with
/// top-left `(r, c)`, the sum of the four wrapped phase differences around it, in units of `2π`
/// (`+1`, `−1` or `0`). A non-zero residue is a branch point where path-following unwrapping becomes
/// path-dependent. Returned as a `(rows−1)·(cols−1)` grid indexed by the loop's top-left corner.
#[must_use]
pub fn residues(wrapped: &[f64], rows: usize, cols: usize) -> Vec<i8> {
    if rows < 2 || cols < 2 {
        return Vec::new();
    }
    let mut res = vec![0i8; (rows - 1) * (cols - 1)];
    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            let (p00, p01) = (wrapped[r * cols + c], wrapped[r * cols + c + 1]);
            let (p10, p11) = (wrapped[(r + 1) * cols + c], wrapped[(r + 1) * cols + c + 1]);
            // Clockwise loop: (r,c)→(r,c+1)→(r+1,c+1)→(r+1,c)→(r,c).
            let loop_sum = wrap_phase(p01 - p00)
                + wrap_phase(p11 - p01)
                + wrap_phase(p10 - p11)
                + wrap_phase(p00 - p10);
            res[r * (cols - 1) + c] = libm::round(loop_sum / TAU) as i8;
        }
    }
    res
}

/// 1-D DCT-II: `F[k] = Σ_n x[n]·cos(π(2n+1)k/2N)`.
fn dct1_forward(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut s = 0.0;
            for (nn, &v) in x.iter().enumerate() {
                s += v * libm::cos(PI * (2 * nn + 1) as f64 * k as f64 / (2 * n) as f64);
            }
            s
        })
        .collect()
}

/// 1-D inverse DCT (DCT-III): `x[n] = (1/N)[F[0] + 2·Σ_{k≥1} F[k]·cos(π(2n+1)k/2N)]` — the exact
/// inverse of [`dct1_forward`].
fn dct1_inverse(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    (0..n)
        .map(|nn| {
            let mut s = f[0];
            for k in 1..n {
                s += 2.0 * f[k] * libm::cos(PI * (2 * nn + 1) as f64 * k as f64 / (2 * n) as f64);
            }
            s / n as f64
        })
        .collect()
}

/// Separable 2-D DCT (or its inverse when `inverse`): the 1-D transform along every row, then every
/// column. Row-major `rows × cols`.
fn dct2(a: &[f64], rows: usize, cols: usize, inverse: bool) -> Vec<f64> {
    let tx = if inverse { dct1_inverse } else { dct1_forward };
    let mut b = vec![0.0; rows * cols];
    for r in 0..rows {
        let row = tx(&a[r * cols..r * cols + cols]);
        b[r * cols..r * cols + cols].copy_from_slice(&row);
    }
    let mut out = b.clone();
    let mut col = vec![0.0; rows];
    for c in 0..cols {
        for r in 0..rows {
            col[r] = b[r * cols + c];
        }
        let tc = tx(&col);
        for r in 0..rows {
            out[r * cols + c] = tc[r];
        }
    }
    out
}

/// Unweighted least-squares (minimum-norm) 2-D phase unwrapping (Ghiglia & Romero 1994): the phase
/// whose gradients best fit the wrapped gradients in the L² sense. It solves the discrete Poisson
/// equation `∇²φ = ρ`, with `ρ` the divergence of the wrapped gradients and Neumann boundaries, by the
/// discrete cosine transform (`φ̂ = ρ̂ / (2cos(πi/M)+2cos(πj/N)−4)`, the mean fixed to zero). Unlike
/// [`unwrap_phase_2d`], the result is path-independent and well-defined even when the field has
/// residues. Row-major `rows × cols`.
///
/// Complexity is `O(MN·(M+N))` (a direct DCT); an FFT-based DCT would drop it to `O(MN·log MN)`.
#[must_use]
pub fn unwrap_phase_2d_lstsq(wrapped: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    // Wrapped gradients (Neumann: zero past the far edge).
    let idx = |r: usize, c: usize| r * cols + c;
    let mut rho = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let dx = if c + 1 < cols {
                wrap_phase(wrapped[idx(r, c + 1)] - wrapped[idx(r, c)])
            } else {
                0.0
            };
            let dx_prev = if c > 0 {
                wrap_phase(wrapped[idx(r, c)] - wrapped[idx(r, c - 1)])
            } else {
                0.0
            };
            let dy = if r + 1 < rows {
                wrap_phase(wrapped[idx(r + 1, c)] - wrapped[idx(r, c)])
            } else {
                0.0
            };
            let dy_prev = if r > 0 {
                wrap_phase(wrapped[idx(r, c)] - wrapped[idx(r - 1, c)])
            } else {
                0.0
            };
            rho[idx(r, c)] = (dx - dx_prev) + (dy - dy_prev);
        }
    }
    let rho_hat = dct2(&rho, rows, cols, false);
    let mut phi_hat = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            if i == 0 && j == 0 {
                continue; // the constant (mean) is undetermined; fix it to zero
            }
            let lambda = 2.0 * libm::cos(PI * i as f64 / rows as f64)
                + 2.0 * libm::cos(PI * j as f64 / cols as f64)
                - 4.0;
            phi_hat[idx(i, j)] = rho_hat[idx(i, j)] / lambda;
        }
    }
    dct2(&phi_hat, rows, cols, true)
}

/// Mark the barrier pixels of a straight branch cut between two dual-grid nodes (residue loop corners)
/// by a Bresenham line, so the flood-fill integration cannot pass between the two residues it joins.
fn draw_cut(barrier: &mut [bool], cols: usize, a: (usize, usize), b: (usize, usize)) {
    let (mut r, mut c) = (a.0 as isize, a.1 as isize);
    let (tr, tc) = (b.0 as isize, b.1 as isize);
    let (dr, dc) = ((tr - r).abs(), -(tc - c).abs());
    let (sr, sc) = (if r < tr { 1 } else { -1 }, if c < tc { 1 } else { -1 });
    let mut err = dr + dc;
    loop {
        barrier[r as usize * cols + c as usize] = true;
        if r == tr && c == tc {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dc {
            err += dc;
            r += sr;
        }
        if e2 <= dr {
            err += dr;
            c += sc;
        }
    }
}

/// Goldstein **branch-cut** 2-D phase unwrapping (row-major `rows × cols`): connect opposite-sign
/// [`residues`] with branch cuts (nearest-neighbour greedy, unmatched charges cut to the image border),
/// then flood-fill the phase from a seed by Itoh integration **without crossing a cut**. The cuts
/// enclose every residue, so the integration path can no longer loop around one — making the result
/// path-independent (the alternative to [`unwrap_phase_2d_lstsq`] for a residue-bearing field). Cut
/// pixels are filled from a neighbour at the end. Returns the unwrapped phase.
#[must_use]
pub fn unwrap_phase_2d_branch_cut(wrapped: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    if rows < 2 || cols < 2 {
        return out;
    }
    // Charged residue nodes on the dual grid, addressed by their loop's top-left pixel.
    let res = residues(wrapped, rows, cols);
    let mut pos: Vec<(usize, usize)> = Vec::new();
    let mut neg: Vec<(usize, usize)> = Vec::new();
    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            match res[r * (cols - 1) + c].cmp(&0) {
                core::cmp::Ordering::Greater => pos.push((r, c)),
                core::cmp::Ordering::Less => neg.push((r, c)),
                core::cmp::Ordering::Equal => {}
            }
        }
    }
    // Branch cuts: greedily join each positive residue to its nearest free negative one.
    let mut barrier = vec![false; rows * cols];
    let mut used = vec![false; neg.len()];
    for &p in &pos {
        let mut best = None;
        let mut best_d = usize::MAX;
        for (k, &q) in neg.iter().enumerate() {
            if used[k] {
                continue;
            }
            let d = p.0.abs_diff(q.0) + p.1.abs_diff(q.1);
            if d < best_d {
                best_d = d;
                best = Some(k);
            }
        }
        if let Some(k) = best {
            used[k] = true;
            draw_cut(&mut barrier, cols, p, neg[k]);
        } else {
            draw_cut(&mut barrier, cols, p, (p.0, 0)); // ground to the left border
        }
    }
    for (k, &q) in neg.iter().enumerate() {
        if !used[k] {
            draw_cut(&mut barrier, cols, q, (q.0, 0));
        }
    }

    // Flood-fill (BFS) Itoh integration from the first non-barrier pixel, never stepping onto a cut.
    let mut visited = vec![false; rows * cols];
    let seed = (0..rows * cols).find(|&i| !barrier[i]).unwrap_or(0);
    out[seed] = wrapped[seed];
    visited[seed] = true;
    let mut queue = vec![seed];
    let mut head = 0;
    while head < queue.len() {
        let i = queue[head];
        head += 1;
        let (r, c) = (i / cols, i % cols);
        let step =
            |nr: usize, nc: usize, out: &mut [f64], visited: &mut [bool], q: &mut Vec<usize>| {
                let ni = nr * cols + nc;
                if !visited[ni] && !barrier[ni] {
                    out[ni] = out[i] + wrap_phase(wrapped[ni] - wrapped[i]);
                    visited[ni] = true;
                    q.push(ni);
                }
            };
        if r > 0 {
            step(r - 1, c, &mut out, &mut visited, &mut queue);
        }
        if r + 1 < rows {
            step(r + 1, c, &mut out, &mut visited, &mut queue);
        }
        if c > 0 {
            step(r, c - 1, &mut out, &mut visited, &mut queue);
        }
        if c + 1 < cols {
            step(r, c + 1, &mut out, &mut visited, &mut queue);
        }
    }
    // Fill the cut pixels from any unwrapped 4-neighbour (they were never integrated across).
    for r in 0..rows {
        for c in 0..cols {
            let i = r * cols + c;
            if visited[i] {
                continue;
            }
            for (nr, nc) in [
                (r.wrapping_sub(1), c),
                (r + 1, c),
                (r, c.wrapping_sub(1)),
                (r, c + 1),
            ] {
                if nr < rows && nc < cols && visited[nr * cols + nc] {
                    let ni = nr * cols + nc;
                    out[i] = out[ni] + wrap_phase(wrapped[i] - wrapped[ni]);
                    visited[i] = true;
                    break;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_precision_loss,
        clippy::unreadable_literal
    )]
    use super::*;
    use spacelab_testkit::nan_max;

    fn unit(phase: f64) -> Complex {
        Complex::from_polar(1.0, phase)
    }

    #[test]
    fn interferogram_phase_is_the_difference() {
        let s1: Vec<Complex> = (0..10).map(|k| unit(0.4 * f64::from(k))).collect();
        let s2: Vec<Complex> = (0..10).map(|k| unit(0.1 * f64::from(k))).collect();
        let ifg = interferogram(&s1, &s2);
        // The per-element loops in this file all iterate the *result*, so an empty one satisfies them
        // without running once. One value per input pixel is the part they cannot state.
        assert_eq!(ifg.len(), s1.len());
        for (k, z) in ifg.iter().enumerate() {
            assert!((wrap_phase(z.arg() - 0.3 * k as f64)).abs() < 1e-12);
        }
    }

    /// (G3) The coherence of a signal in noise orthogonal to it over the window is `1/√(1 + |b|²)`.
    ///
    /// `s₂ = s₁ + b·n` with unit-modulus `s₁` and `n = s₁·e^{2πik/9}` over the nine pixels of a 3 × 3
    /// window: the ninth roots of unity sum to zero, so `Σ s₁ n̄ = 0` and
    /// `derivations/insar_unwrap.mac` gives `γ = 1/√(1 + |b|²)` exactly — `4/5` for `|b| = 3/4`,
    /// whatever the phase of `b` or of `s₁`. Only the centre pixel of a 3 × 3 image has its whole
    /// window inside, so it alone carries the closed form.
    #[test]
    fn the_coherence_of_a_signal_in_orthogonal_noise_is_its_closed_form() {
        let s1: Vec<Complex> = (0..9).map(|k| unit(0.7 * f64::from(k * k) - 1.1)).collect();
        for b in [
            Complex::from_polar(0.75, 0.0),
            Complex::from_polar(0.75, 2.3),
        ] {
            let s2: Vec<Complex> = s1
                .iter()
                .enumerate()
                .map(|(k, &z)| {
                    let n = z * unit(core::f64::consts::TAU * k as f64 / 9.0);
                    z + b * n
                })
                .collect();
            let gamma = coherence(&s1, &s2, 3, 3, 1)[4];
            assert!((gamma - 0.8).abs() < 1e-15, "b = {b:?}: coherence {gamma}");
        }
    }

    #[test]
    fn coherence_is_one_for_identical_and_low_for_random() {
        let mut seed = 1u64;
        let mut rng = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            (seed >> 33) as f64 / (1u64 << 31) as f64 * TAU
        };
        let n = 8 * 8;
        let s1: Vec<Complex> = (0..n).map(|_| unit(rng())).collect();
        let identical = coherence(&s1, &s1, 8, 8, 2);
        assert!(identical.iter().all(|&g| (g - 1.0).abs() < 1e-9));
        let s2: Vec<Complex> = (0..n).map(|_| unit(rng())).collect(); // independent random phases
        let cross = coherence(&s1, &s2, 8, 8, 2);
        // A 5×5 boxcar of independent phasors averages down to well below full coherence.
        assert!(
            cross.iter().all(|&g| g < 0.75),
            "max {}",
            cross.iter().copied().fold(0.0, f64::max)
        );
    }

    /// The boxcar must contain exactly the pixels it claims to, and a window of nothing must not
    /// divide by nothing.
    ///
    /// The fixture above measures identical images, where every pixel of every window is perfectly
    /// correlated and the coherence is 1 whatever the window covers — so the window arithmetic is
    /// unobservable, and eight mutants of the offset, the row-major index and the empty-window guard
    /// outlived it. Against a constant reference and with unit amplitudes the coherence becomes a
    /// count, `γ = |Σ e^{−iφ}| / N = |n₀ − n_π| / N` over the window, so a `π` mask makes every
    /// pixel's answer a rational fixed by which pixels the window covers.
    ///
    /// The mask is scattered rather than periodic on purpose. A checkerboard was measured not to
    /// separate `rr * cols` from `rr + cols`: with an odd `cols` the parity of the row-major index
    /// *is* the parity of `r + c`, so an index shift flips every phase at once, `cross` becomes
    /// `−cross`, and only `|cross|` is read. The expected table is generated from the boxcar
    /// definition rather than from this implementation.
    #[test]
    fn coherence_counts_exactly_the_pixels_its_window_covers() {
        let (rows, cols) = (5usize, 6usize);
        let phase_pi = [
            (0, 0),
            (1, 3),
            (1, 4),
            (2, 1),
            (2, 4),
            (3, 2),
            (3, 3),
            (4, 5),
        ];
        let flat: Vec<Complex> = vec![unit(0.0); rows * cols];
        let masked: Vec<Complex> = (0..rows * cols)
            .map(|i| {
                unit(if phase_pi.contains(&(i / cols, i % cols)) {
                    PI
                } else {
                    0.0
                })
            })
            .collect();
        #[rustfmt::skip]
        let want = [
            [1.0 / 2.0, 2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
            [1.0 / 3.0, 5.0 / 9.0, 5.0 / 9.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [2.0 / 3.0, 5.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 3.0],
            [2.0 / 3.0, 5.0 / 9.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0,       2.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0],
        ];
        let g = coherence(&flat, &masked, rows, cols, 1);
        assert_eq!(g.len(), rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                assert!(
                    (g[r * cols + c] - want[r][c]).abs() < 1e-15,
                    "({r},{c}): {} vs {}",
                    g[r * cols + c],
                    want[r][c]
                );
            }
        }
        // A window with no signal in it has no coherence to report. `denom > 0.0` is what keeps the
        // ratio off `0/0`; asked as `>= 0.0` every pixel here comes back NaN.
        let dark = vec![Complex::ZERO; rows * cols];
        let none = coherence(&dark, &dark, rows, cols, 1);
        assert!(none.iter().all(|&g| g == 0.0), "{none:?}");
    }

    /// A grid too small to hold a 2×2 loop has no residues, and must say so rather than index into
    /// the arithmetic that would compute them.
    ///
    /// `rows < 2 || cols < 2` is the whole guard, and five mutants of it survived because no fixture
    /// ever passed a degenerate shape: `<=` swallows the smallest grid that *does* have a loop,
    /// while `==` and `&&` let a zero-row grid through to `rows - 1`, which underflows.
    #[test]
    fn residues_refuses_a_grid_too_small_to_hold_a_loop() {
        assert_eq!(residues(&[], 0, 0), [] as [i8; 0]);
        assert_eq!(residues(&[], 0, 5), [] as [i8; 0]);
        assert_eq!(residues(&[], 5, 0), [] as [i8; 0]);
        assert_eq!(residues(&[0.0; 5], 1, 5), [] as [i8; 0]);
        assert_eq!(residues(&[0.0; 5], 5, 1), [] as [i8; 0]);
        // Two rows and three columns is the smallest grid with loops in it, and it must be measured,
        // not waved through: two loops, both residue-free on a linear ramp.
        assert_eq!(residues(&[0.0, 1.0, 2.0, 0.5, 1.5, 2.5], 2, 3), vec![0, 0]);
    }

    /// The branch cut is a Bresenham line and covers exactly the pixels one covers.
    ///
    /// Nine mutants of `draw_cut` — the two step directions, the negated column delta, the row-major
    /// index and the termination test — survived every unwrapping fixture, because the only one with
    /// residues in it checked congruence, which every Itoh integration satisfies by construction
    /// whatever the barrier looks like. The oracle here is the definition of a Bresenham line: the
    /// pixel nearest the ideal line at each step along the major axis. Every case below is chosen so
    /// no ideal point lands on a half-integer, where the definition itself is a convention.
    #[test]
    fn draw_cut_lays_the_bresenham_pixels_and_nothing_else() {
        let (rows, cols) = (8usize, 8usize);
        let cases: [(_, _, &[(usize, usize)]); 5] = [
            // Horizontal, vertical and the exact diagonal.
            ((2, 1), (2, 5), &[(2, 1), (2, 2), (2, 3), (2, 4), (2, 5)]),
            ((1, 3), (5, 3), &[(1, 3), (2, 3), (3, 3), (4, 3), (5, 3)]),
            ((0, 0), (4, 4), &[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]),
            // Shallow, slope 2/5: r = 0.4c gives 0, .4, .8, 1.2, 1.6, 2.
            (
                (0, 0),
                (2, 5),
                &[(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (2, 5)],
            ),
            // Steep, slope 5/2: c = 0.4r gives the same six, transposed.
            (
                (0, 0),
                (5, 2),
                &[(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)],
            ),
        ];
        let lay = |from: (usize, usize), to: (usize, usize)| -> Vec<(usize, usize)> {
            let mut barrier = vec![false; rows * cols];
            draw_cut(&mut barrier, cols, from, to);
            (0..rows * cols)
                .filter(|&i| barrier[i])
                .map(|i| (i / cols, i % cols))
                .collect()
        };
        for (a, b, want) in cases {
            for (from, to) in [(a, b), (b, a)] {
                assert_eq!(lay(from, to), want, "cut {from:?} -> {to:?}");
            }
        }

        // Where the ideal line does pass through a half-integer, the choice is a convention and
        // this one is pinned as measured, not derived: slope 1/2 puts the midpoint of every pair of
        // columns exactly between two rows, and the algorithm takes the diagonal step, advancing
        // both axes at once. That makes the cut direction-dependent -- `a -> b` and `b -> a` are
        // reflections of each other -- which is harmless for a barrier but is the observable that
        // separates the two tie forms of the Bresenham decision from the forms that ship.
        assert_eq!(
            lay((0, 0), (2, 4)),
            [(0, 0), (1, 1), (1, 2), (2, 3), (2, 4)]
        );
        assert_eq!(
            lay((2, 4), (0, 0)),
            [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4)]
        );
        assert_eq!(
            lay((0, 0), (4, 2)),
            [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2)]
        );
        assert_eq!(
            lay((4, 2), (0, 0)),
            [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]
        );
    }

    /// What the branch cuts buy: the 2π discontinuity sits *on the cut*, not wherever the
    /// integration order happened to leave it.
    ///
    /// Congruence — `wrap(out − wrapped) = 0` — is satisfied by any Itoh integration whatsoever,
    /// cuts or no cuts, because every step adds a wrapped difference; the fixture that checked it
    /// could not fail, and 46 mutants of this function survived behind it. The observable that does
    /// separate them is *where* the unwrapped field jumps by 2π. Count the 4-neighbour pairs whose
    /// output difference is not the wrapped input difference:
    ///
    /// - with cuts, there are exactly 16 of them — the Manhattan distance between the two residues,
    ///   one per step of the Bresenham staircase joining them — and every one lies inside the
    ///   rectangle the two residues span;
    /// - without cuts, `unwrap_phase_2d` leaves 23, in two ghost lines running from each residue to
    ///   the right border, which is the path dependence the cuts exist to remove.
    #[test]
    fn the_branch_cut_confines_the_discontinuity_to_the_cut_it_drew() {
        let (rows, cols) = (24usize, 24usize);
        // Two opposite-charge branch points at (8.5, 7.5) and (15.5, 16.5), so the residues land on
        // the dual-grid loops with top-left pixels (8, 7) and (15, 16).
        let wrapped: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as f64, (i % cols) as f64);
                wrap_phase(libm::atan2(r - 8.5, c - 7.5) - libm::atan2(r - 15.5, c - 16.5))
            })
            .collect();
        let res = residues(&wrapped, rows, cols);
        let charged = |sign: i8| -> Vec<(usize, usize)> {
            (0..(rows - 1) * (cols - 1))
                .filter(|&i| res[i].signum() == sign)
                .map(|i| (i / (cols - 1), i % (cols - 1)))
                .collect()
        };
        assert_eq!(charged(1), vec![(8, 7)]);
        assert_eq!(charged(-1), vec![(15, 16)]);

        // Every 4-neighbour pair whose unwrapped difference is not the wrapped input difference.
        let jumps = |un: &[f64]| -> Vec<(usize, usize)> {
            let mut out = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    let i = r * cols + c;
                    for j in [
                        (c + 1 < cols).then_some(i + 1),
                        (r + 1 < rows).then_some(i + cols),
                    ]
                    .into_iter()
                    .flatten()
                    {
                        if ((un[j] - un[i]) - wrap_phase(wrapped[j] - wrapped[i])).abs() > 1.0 {
                            out.push((r, c));
                        }
                    }
                }
            }
            out
        };

        let cut = jumps(&unwrap_phase_2d_branch_cut(&wrapped, rows, cols));
        assert_eq!(cut.len(), 8_usize.abs_diff(15) + 7_usize.abs_diff(16));
        for &(r, c) in &cut {
            assert!(
                (8..=15).contains(&r) && (7..=16).contains(&c),
                "({r},{c}) is outside the rectangle the residues span"
            );
        }

        let plain = jumps(&unwrap_phase_2d(&wrapped, rows, cols));
        assert_eq!(plain.len(), 23, "{plain:?}");
        assert!(
            plain.iter().any(|&(_, c)| c == cols - 1),
            "the path-following ghost line should reach the border"
        );
    }

    #[test]
    fn unwrap_1d_recovers_a_wrapped_ramp() {
        // A steep ramp (0.9 rad/sample over 60 samples ⇒ many 2π wraps).
        let truth: Vec<f64> = (0..60).map(|k| 0.9 * f64::from(k)).collect();
        let wrapped: Vec<f64> = truth.iter().map(|&p| wrap_phase(p)).collect();
        let un = unwrap_phase_1d(&wrapped);
        assert_eq!(un.len(), wrapped.len());
        for (u, t) in un.iter().zip(&truth) {
            assert!((u - t).abs() < 1e-9, "{u} vs {t}");
        }
    }

    #[test]
    fn unwrap_2d_recovers_a_smooth_phase_up_to_a_constant() {
        let (rows, cols) = (24usize, 32usize);
        let truth: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = (i / cols, i % cols);
                0.35 * r as f64 + 0.22 * c as f64 // a residue-free ramp
            })
            .collect();
        let wrapped: Vec<f64> = truth.iter().map(|&p| wrap_phase(p)).collect();
        let un = unwrap_phase_2d(&wrapped, rows, cols);
        let offset = un[0] - truth[0];
        for (u, t) in un.iter().zip(&truth) {
            assert!((u - offset - t).abs() < 1e-9, "{} vs {t}", u - offset);
        }
    }

    #[test]
    fn remove_reference_leaves_the_residual_topographic_phase() {
        let reference: Vec<f64> = (0..20).map(|k| 0.5 * f64::from(k)).collect();
        let topo: Vec<f64> = (0..20).map(|k| 0.05 * f64::from(k * k)).collect();
        let ifg: Vec<Complex> = reference
            .iter()
            .zip(&topo)
            .map(|(&r, &t)| unit(r + t))
            .collect();
        let residual = remove_reference_phase(&ifg, &reference);
        assert_eq!(residual.len(), ifg.len());
        for (res, t) in residual.iter().zip(&topo) {
            assert!((wrap_phase(res - t)).abs() < 1e-12);
        }
    }

    /// (G1) The coded DCT-III undoes the coded DCT-II at every length up to 16, the identity
    /// `derivations/insar_unwrap.mac` proves for `N ≤ 3` in closed form and checks to 60 digits for
    /// `N ≤ 8` (the `.gp` to 100 digits for `N ≤ 16`). Odd and even lengths both, since the cosines
    /// of an odd `N` include `cos(π/2)`.
    #[test]
    fn the_inverse_dct_undoes_the_forward_one_at_every_length() {
        for n in 1..=16u32 {
            let x: Vec<f64> = (0..n).map(|k| (f64::from(k) * 0.91).cos() - 0.3).collect();
            let rt = dct1_inverse(&dct1_forward(&x));
            for (k, (a, b)) in rt.iter().zip(&x).enumerate() {
                assert!((a - b).abs() < 1e-14, "N = {n}, sample {k}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn dct_matches_scipy_and_round_trips() {
        // dct1_forward = scipy.fft.dct(x, type=2)/2 (our forward drops scipy's factor-2 convention).
        let f = dct1_forward(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let oracle = [15.0, -4.979796569765561, 0.0, -0.4490279765795853, 0.0];
        for (g, o) in f.iter().zip(oracle) {
            assert!((g - o).abs() < 1e-9, "{g} vs scipy {o}");
        }
        // The inverse is exact.
        let x: Vec<f64> = (0..16).map(|k| (f64::from(k) * 0.37).sin() + 0.5).collect();
        let rt = dct1_inverse(&dct1_forward(&x));
        for (a, b) in rt.iter().zip(&x) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    /// (G3) A smooth field has no residues — the four steps around a cell of a gradient telescope —
    /// and the vortex `atan2(r − r₀, c − c₀)` has exactly one, `+1` on the loop the code walks:
    /// four quarter turns (`derivations/insar_unwrap.mac`, re-checked in the `.gp`).
    #[test]
    fn residues_vanish_on_a_smooth_field_and_ring_a_singularity() {
        let (rows, cols) = (12usize, 12usize);
        // A residue-free ramp: no branch points.
        let ramp: Vec<f64> = (0..rows * cols)
            .map(|i| wrap_phase(0.35 * (i / cols) as f64 + 0.22 * (i % cols) as f64))
            .collect();
        assert!(residues(&ramp, rows, cols).iter().all(|&r| r == 0));
        // A phase singularity φ = atan2(y−yc, x−xc): exactly one ±1 residue, at the loop that encloses
        // the branch point (5.5, 5.5).
        let spiral: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as f64, (i % cols) as f64);
                libm::atan2(r - 5.5, c - 5.5)
            })
            .collect();
        let res = residues(&spiral, rows, cols);
        let nonzero: Vec<usize> = (0..res.len()).filter(|&k| res[k] != 0).collect();
        assert_eq!(nonzero.len(), 1, "expected one residue, got {nonzero:?}");
        assert_eq!(
            nonzero[0],
            5 * (cols - 1) + 5,
            "residue not at the singularity"
        );
        assert_eq!(res[nonzero[0]], 1, "the vortex winds the positive way");
    }

    /// (G2) The least-squares unwrap satisfies its defining equation. `derivations/insar_unwrap.mac`
    /// shows that the Neumann Poisson equation `∇²φ = ρ` checked here is the normal equation of
    /// `min Σ (φ_q − φ_p − g_pq)²`, and that the DCT-II basis diagonalises that Laplacian with the
    /// divisor `2cos(πi/M) + 2cos(πj/N) − 4` the solver uses.
    #[test]
    fn lstsq_solves_the_discrete_poisson_equation() {
        // On a field WITH a residue (where path-following is ambiguous), the LS unwrap must satisfy
        // ∇²φ = ρ exactly — the normal equations that define the least-squares solution.
        let (rows, cols) = (16usize, 20usize);
        let wrapped: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as f64, (i % cols) as f64);
                wrap_phase(0.4 * r + 0.3 * c + libm::atan2(r - 8.3, c - 9.7))
            })
            .collect();
        let phi = unwrap_phase_2d_lstsq(&wrapped, rows, cols);
        let idx = |r: usize, c: usize| r * cols + c;
        let wgrad_x = |r: usize, c: usize| wrap_phase(wrapped[idx(r, c + 1)] - wrapped[idx(r, c)]);
        let wgrad_y = |r: usize, c: usize| wrap_phase(wrapped[idx(r + 1, c)] - wrapped[idx(r, c)]);
        let mut max_err = 0.0f64;
        for r in 0..rows {
            for c in 0..cols {
                // ρ: divergence of the wrapped gradients (Neumann).
                let rho = (if c + 1 < cols { wgrad_x(r, c) } else { 0.0 })
                    - (if c > 0 { wgrad_x(r, c - 1) } else { 0.0 })
                    + (if r + 1 < rows { wgrad_y(r, c) } else { 0.0 })
                    - (if r > 0 { wgrad_y(r - 1, c) } else { 0.0 });
                // ∇²φ: divergence of the *unwrapped* gradients (Neumann), same scheme.
                let lap = (if c + 1 < cols {
                    phi[idx(r, c + 1)] - phi[idx(r, c)]
                } else {
                    0.0
                }) - (if c > 0 {
                    phi[idx(r, c)] - phi[idx(r, c - 1)]
                } else {
                    0.0
                }) + (if r + 1 < rows {
                    phi[idx(r + 1, c)] - phi[idx(r, c)]
                } else {
                    0.0
                }) - (if r > 0 {
                    phi[idx(r, c)] - phi[idx(r - 1, c)]
                } else {
                    0.0
                });
                max_err = nan_max(max_err, (lap - rho).abs());
            }
        }
        assert!(max_err < 1e-9, "Poisson residual {max_err}");
    }

    #[test]
    fn branch_cut_recovers_a_residue_free_ramp() {
        // No residues ⇒ no cuts ⇒ the flood-fill is plain path-following, exact up to a constant.
        let (rows, cols) = (16usize, 20usize);
        let truth: Vec<f64> = (0..rows * cols)
            .map(|i| 0.3 * (i / cols) as f64 + 0.2 * (i % cols) as f64)
            .collect();
        let wrapped: Vec<f64> = truth.iter().map(|&p| wrap_phase(p)).collect();
        assert!(residues(&wrapped, rows, cols).iter().all(|&r| r == 0));
        let un = unwrap_phase_2d_branch_cut(&wrapped, rows, cols);
        let off = un[0] - truth[0];
        for (u, t) in un.iter().zip(&truth) {
            assert!((u - off - t).abs() < 1e-9, "{} vs {t}", u - off);
        }
    }

    #[test]
    fn branch_cut_is_congruent_on_a_residue_dipole() {
        // A residue dipole: φ = atan2 around a + centre minus atan2 around a − centre gives exactly one
        // +1 and one −1 residue. Branch cuts join them; the flood-fill result stays congruent to the
        // wrapped input (wrap(out − wrapped) = 0) everywhere — a consistent unwrap despite the residues.
        let (rows, cols) = (24usize, 24usize);
        let field: Vec<f64> = (0..rows * cols)
            .map(|i| {
                let (r, c) = ((i / cols) as f64, (i % cols) as f64);
                libm::atan2(r - 8.5, c - 7.5) - libm::atan2(r - 15.5, c - 16.5)
            })
            .collect();
        let wrapped: Vec<f64> = field.iter().map(|&p| wrap_phase(p)).collect();
        let res = residues(&wrapped, rows, cols);
        assert_eq!(res.iter().filter(|&&r| r > 0).count(), 1);
        assert_eq!(res.iter().filter(|&&r| r < 0).count(), 1);

        let un = unwrap_phase_2d_branch_cut(&wrapped, rows, cols);
        // Congruent everywhere: the unwrapped phase equals the wrapped one modulo 2π.
        for (u, w) in un.iter().zip(&wrapped) {
            assert!(wrap_phase(u - w).abs() < 1e-9, "not congruent: {u} vs {w}");
        }
    }

    #[test]
    fn lstsq_recovers_a_residue_free_field_up_to_a_constant() {
        // Where the true gradients are below π (no aliasing, no residues), LS is exact up to a constant.
        let (rows, cols) = (18usize, 24usize);
        let truth: Vec<f64> = (0..rows * cols)
            .map(|i| 0.3 * (i / cols) as f64 + 0.2 * (i % cols) as f64)
            .collect();
        let wrapped: Vec<f64> = truth.iter().map(|&p| wrap_phase(p)).collect();
        let phi = unwrap_phase_2d_lstsq(&wrapped, rows, cols);
        let off = phi[0] - truth[0];
        for (p, t) in phi.iter().zip(&truth) {
            assert!((p - off - t).abs() < 1e-9, "{} vs {t}", p - off);
        }
    }

    #[test]
    fn a_grid_too_small_to_unwrap_is_answered_with_its_own_shape_and_nothing_else() {
        // Every unwrapper here guards its degenerate shapes, and the guard is a disjunction: one
        // vanishing side is enough. Read as a conjunction, a 0 x 5 field walks into indexing an empty
        // vector; read with the comparison shifted by one, a field of exactly the smallest workable
        // size is refused instead of unwrapped. Both directions are checked below, and the smallest
        // workable size -- two rows, two columns, one residue loop -- is checked to actually work.
        let ramp: Vec<f64> = (0..6).map(|k| wrap_phase(1.7 * f64::from(k))).collect();
        assert_eq!(unwrap_phase_2d(&[], 0, 5), [] as [f64; 0]);
        assert_eq!(unwrap_phase_2d(&[], 5, 0), [] as [f64; 0]);
        assert_eq!(unwrap_phase_2d_lstsq(&[], 0, 5), [] as [f64; 0]);
        assert_eq!(unwrap_phase_2d_lstsq(&[], 5, 0), [] as [f64; 0]);
        assert_eq!(residues(&ramp, 1, 6), [] as [i8; 0]);
        assert_eq!(residues(&ramp, 6, 1), [] as [i8; 0]);
        assert_eq!(residues(&ramp[..4], 2, 2).len(), 1);
        assert_eq!(residues(&ramp, 2, 3).len(), 2);

        // A single row has no 2-D loop to integrate around, so the branch-cut unwrapper declines it
        // and returns a zero field of the right size rather than falling through to its flood fill.
        assert_eq!(unwrap_phase_2d_branch_cut(&ramp, 1, 6), vec![0.0; 6]);
        assert_eq!(unwrap_phase_2d_branch_cut(&ramp, 6, 1), vec![0.0; 6]);
        // Two by two is the smallest field it does accept, and there it must unwrap: the truth is a
        // ramp of 1.7 rad per column and 2.3 per row, which wraps, so the answer cannot be the input.
        let truth = [0.0, 1.7, 2.3, 4.0];
        let wrapped: Vec<f64> = truth.iter().map(|&p| wrap_phase(p)).collect();
        let out = unwrap_phase_2d_branch_cut(&wrapped, 2, 2);
        for (o, t) in out.iter().zip(truth) {
            assert!((o - out[0] - (t - truth[0])).abs() < 1e-12, "{o} vs {t}");
        }
    }

    // Two fixtures, because one field cannot carry both halves of the requirement. RICH keeps the
    // flood fill reaching nearly every pixel, so that the up-step, the left-step and three of the
    // four fill directions each own a pixel nothing else reaches; POCKET walls the seed in, so that
    // most of the grid falls to the cut-pixel fill instead. Both come from
    // `tools/gen_branchcut_oracle.py` -- Goldstein's algorithm written from its description rather
    // than from this code, which also searched for the two layouts. The wrapped fields are embedded
    // verbatim so neither side re-evaluates a transcendental.
    #[rustfmt::skip]
    const RICH_WRAPPED: [f64; 64] = [
        1.423696631076531, 3.064351863956986, -2.5720037049763054, -2.736588582173977, 2.071740193274552, 1.7141956593435186, 1.8876203585714704, 2.319229110046095,
        0.07074802638355049, -1.4792446144806821, -0.7370212784776582, -0.3544862624937708, 0.9438365436966123, 1.2017125225228886, 1.4833619825903588, 2.845165588265222,
        -0.260669859299993, -2.029961602060652, 1.7675464398369574, 0.9728337889921139, 0.8607378001719113, 0.6660214340226265, -0.015896218002016127, -1.8532700254170464,
        0.28998024045353965, -2.1692049656065606, 2.58417093573264, 1.6362617734876252, 1.0219434677462331, 0.4697046444694344, -0.35255471886553025, -1.4597573598032811,
        2.2855047000259923, -3.0919306736596326, 2.8071541236421265, 1.9684413434446721, 1.1054130486097566, 0.5452079523789628, -0.07890971713105266, -1.754374105304783,
        -2.8920713527517687, -2.7128132693286675, -2.9678831417763973, 2.428295315151103, 0.6867196624905576, 0.5254691873625168, 1.2965632731469139, 3.0974724295270235,
        -1.9768914229618488, -1.7540024880218645, -1.8297326361924044, -2.70349456352075, -1.2993890242289554, -0.2510199019434003, 1.828009998134089, 2.92484844472152,
        -1.0878627754494037, -0.6202392736501228, 0.2799725100812327, 2.4724861376062552, -2.601309012431135, -2.19967386648461, -3.1135848706317057, -3.010248087592788,
    ];
    #[rustfmt::skip]
    const RICH_EXPECTED: [f64; 64] = [
        0.0, 0.0, -2.5720037049763054, -2.736588582173977, 2.071740193274552, 1.7141956593435186, 1.8876203585714704, 2.319229110046095,
        0.0, 4.803940692698904, -0.7370212784776582, -0.3544862624937708, 0.9438365436966123, 1.2017125225228886, 1.4833619825903588, 2.845165588265222,
        6.022515447879593, 4.253223705118934, 1.7675464398369574, 0.9728337889921139, 0.8607378001719113, 0.6660214340226265, -0.015896218002016127, 4.42991528176254,
        6.573165547633126, 4.113980341573026, 2.58417093573264, 1.6362617734876252, 1.0219434677462331, 0.4697046444694344, -0.35255471886553025, 4.823427947376305,
        2.2855047000259923, 3.1912546335199536, 2.8071541236421265, 1.9684413434446721, 1.1054130486097566, 0.5452079523789628, -0.07890971713105266, 4.528811201874803,
        3.3911139544278175, 3.5703720378509187, 3.315302165403189, 2.428295315151103, 0.6867196624905576, 0.5254691873625168, 1.2965632731469139, 3.0974724295270235,
        4.306293884217737, 4.529182819157722, 4.453452670987182, 3.579690743658836, 4.983796282950631, -0.2510199019434003, 1.828009998134089, 2.92484844472152,
        -1.0878627754494028, -0.6202392736501219, 0.2799725100812336, 2.472486137606256, 3.6818762947484514, 4.083511440694976, 3.1696004365478805, 3.2729372195867983,
    ];
    #[rustfmt::skip]
    const POCKET_WRAPPED: [f64; 64] = [
        2.303730699217482, 2.0575643265847017, 1.5735773929669854, 0.8535629527657882, -0.8599690913332374, -1.3785282494693796, -1.7480749199140924, 2.987296600549975,
        -2.8521442770916923, -2.893230137933415, 2.4377924739614976, 2.7500293178832003, -2.052242371033751, -1.5411966673994497, 0.014415161506827978, 1.5060566680498315,
        -1.6896385875165847, -0.7809865493529973, 2.3258251542938204, -2.644993095797756, -1.9409429361458344, -2.9026223574339918, 1.9854147751757152, 0.9252895077272472,
        -0.7562034296067566, -0.700091663783259, -2.465400263730153, -1.8191582277274367, -0.17528119666676467, 2.302403431716529, -2.6733449345523934, -0.5608698738122366,
        0.2082950601785587, 1.0568774535835495, 3.060865674497336, -1.6953271394323801, 0.9729636283916747, 2.7155389689454275, -2.2011852308745787, -0.9363182675135668,
        1.0102771114252214, 1.8208172077721532, 2.8052476081474262, -2.859537612699714, 2.906283092326447, -2.761805775504297, -1.8231204703548203, -0.8872433723412532,
        1.6026126861822565, 2.2489246611038816, 2.9014091814784013, -2.9039407076203503, -2.598255945090842, -2.113932808023849, -1.4351178626200491, -0.7053398896946677,
        2.0973586910606485, 2.644483530401094, -3.0981807214082275, -2.612479629805255, -2.1619116844809585, -1.662157714607008, -1.0868672497586482, -0.4763204278118698,
    ];
    #[rustfmt::skip]
    const POCKET_EXPECTED: [f64; 64] = [
        0.0, 0.0, -4.709607914212601, -5.429622354413798, -0.8599690913332374, -1.3785282494693796, -1.7480749199140924, -3.2958887066296114,
        0.0, -2.893230137933415, -3.8453928332180887, -3.533155989296386, -2.052242371033751, -1.5411966673994497, 0.014415161506827978, -4.777128639129755,
        0.0, -0.7809865493529973, -3.957360152885766, -2.644993095797756, -1.9409429361458344, -2.9026223574339918, 1.9854147751757152, -5.357895799452339,
        0.0, -0.700091663783259, -2.465400263730153, -1.8191582277274367, -0.17528119666676467, -3.980781875463056, 3.6098403726271924, -6.844055180991823,
        0.0, 1.0568774535835495, -3.2223196326822503, -1.6953271394323801, 0.9729636283916747, -3.567646338234158, 4.0820000763050075, -7.219503574693153,
        0.0, 1.8208172077721532, -3.47793769903216, -2.859537612699714, 2.906283092326447, -2.761805775504296, 4.460064836824766, -7.170428679520839,
        0.0, 2.2489246611038816, -3.381776125701185, -2.9039407076203503, 3.684929362088744, -2.113932808023848, 4.848067444559537, -6.988525196874253,
        0.0, 2.644483530401094, -3.0981807214082275, -2.612479629805255, 4.121273622698627, -1.662157714607007, 5.196318057420938, -6.759505734991455,
    ];

    fn assert_branch_cut(
        wrapped: &[f64; 64],
        expected: &[f64; 64],
        positive: usize,
        negative: usize,
    ) -> Vec<f64> {
        let res = residues(wrapped, 8, 8);
        assert_eq!(res.iter().filter(|&&v| v > 0).count(), positive);
        assert_eq!(res.iter().filter(|&&v| v < 0).count(), negative);
        let out = unwrap_phase_2d_branch_cut(wrapped, 8, 8);
        for (i, (&got, &want)) in out.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "pixel {i} ({}, {}): {got} vs {want}",
                i / 8,
                i % 8
            );
        }
        out
    }

    /// Every pixel of a residue-bearing field, against an independent implementation.
    ///
    /// The two tests above check the branch-cut unwrapper the only ways a smooth fixture can: that a
    /// residue-free ramp comes back, and that the jumps a dipole leaves behind sit on the cut. Neither
    /// reaches the machinery in between -- which residue pairs with which, where the cut is drawn,
    /// where the fill starts, which neighbour each blocked pixel is finally filled from -- and
    /// twenty-four mutants of it outlived them.
    ///
    /// Four positive charges against five negative ones give the pairing a real choice: candidates
    /// that tie on L1 distance, and candidates sharing a positive's row or column, so that both the
    /// metric and its first-wins tie-break change the cuts if either is altered. The seventeen
    /// barrier pixels leave pockets the fill can only enter by stepping up or left, and cut pixels
    /// whose only integrated neighbour lies in one particular direction.
    #[test]
    fn the_branch_cut_unwraps_every_pixel_of_a_nine_residue_field() {
        assert_branch_cut(&RICH_WRAPPED, &RICH_EXPECTED, 4, 5);
    }

    /// The same unwrapper with the seed walled in, which reaches a different half of the code.
    ///
    /// Here the first free pixel is not pixel 0 and its component is small, so the flood fill
    /// integrates 46 of the 64 pixels and the rest is left to the cut-pixel fill. That fill sweeps
    /// row-major and can only read a neighbour it has already written, so it propagates right and
    /// down and never reaches the bottom left. Reading one row past the grid -- the `<=` form of the
    /// row bound -- is attempted only from a bottom-row pixel still unfilled when the sweep arrives,
    /// which is exactly this shape and no smoother one.
    #[test]
    fn a_seed_walled_into_a_corner_leaves_the_rest_of_the_grid_to_the_fill() {
        let out = assert_branch_cut(&POCKET_WRAPPED, &POCKET_EXPECTED, 3, 6);
        // Column 0 is neither integrated nor filled, so it comes back as the zero the output buffer
        // was created with -- which a caller cannot tell from a phase that really is zero.
        assert!(out.iter().step_by(8).all(|&v| v == 0.0));
    }
}
