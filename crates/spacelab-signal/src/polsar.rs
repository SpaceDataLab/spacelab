//! Polarimetric SAR decomposition — turning a scattering matrix into physical mechanisms.
//!
//! - [`pauli_powers`] — the **Pauli** decomposition: the powers of single-, double- and cross-polar
//!   bounce, `|S_hh+S_vv|²/2` (odd/surface), `|S_hh−S_vv|²/2` (even/dihedral), `2|S_hv|²` (volume).
//! - [`cloude_pottier`] — the **H/A/ᾱ** decomposition (Cloude & Pottier): the eigen-decomposition of the
//!   3×3 Hermitian **coherency matrix** `T₃ = ⟨k_P k_Pᴴ⟩` (`k_P` the Pauli scattering vector) gives
//!   scattering entropy `H = −Σ pᵢ log₃ pᵢ`, anisotropy `A = (p₂−p₃)/(p₂+p₃)` and mean scattering angle
//!   `ᾱ = Σ pᵢ αᵢ` (`αᵢ = arccos|eᵢ[0]|`), with `pᵢ = λᵢ/Σλ` the eigenvalue probabilities.
//!
//! The Hermitian eigen-problem is solved by embedding `T₃` in a real symmetric `6×6` and running a
//! cyclic Jacobi sweep. `alloc` for the matrices.

// Physical single letters (h, v, a, s) mirror the polarimetry literature; the doc uses SAR terms;
// the Jacobi rotation's akp/akq bindings only look alike.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::doc_markdown
)]

use crate::fft::Complex;
use alloc::vec;
use alloc::vec::Vec;

/// The Pauli scattering vector `k_P = (1/√2)[S_hh+S_vv, S_hh−S_vv, 2·S_hv]` (reciprocal/monostatic).
#[must_use]
pub fn pauli_vector(s_hh: Complex, s_hv: Complex, s_vv: Complex) -> [Complex; 3] {
    let r = core::f64::consts::FRAC_1_SQRT_2;
    [
        (s_hh + s_vv).scale(r),
        (s_hh - s_vv).scale(r),
        s_hv.scale(2.0 * r),
    ]
}

/// The Pauli bounce powers `[odd, even, volume]` = `[|S_hh+S_vv|²/2, |S_hh−S_vv|²/2, 2|S_hv|²]`.
#[must_use]
pub fn pauli_powers(s_hh: Complex, s_hv: Complex, s_vv: Complex) -> [f64; 3] {
    [
        (s_hh + s_vv).norm_sqr() / 2.0,
        (s_hh - s_vv).norm_sqr() / 2.0,
        2.0 * s_hv.norm_sqr(),
    ]
}

/// The `3×3` Hermitian coherency matrix `T₃ = (1/L)·Σ k_P k_Pᴴ` (row-major) averaged over the looks
/// `samples`, each a scattering triple `(S_hh, S_hv, S_vv)`.
#[must_use]
pub fn coherency_matrix(samples: &[(Complex, Complex, Complex)]) -> Vec<Complex> {
    let mut t = vec![Complex::ZERO; 9];
    for &(hh, hv, vv) in samples {
        let k = pauli_vector(hh, hv, vv);
        for i in 0..3 {
            for j in 0..3 {
                t[i * 3 + j] = t[i * 3 + j] + k[i] * k[j].conj();
            }
        }
    }
    let scale = 1.0 / samples.len().max(1) as f64;
    for x in &mut t {
        *x = x.scale(scale);
    }
    t
}

/// Cyclic Jacobi eigen-decomposition of a real symmetric `n×n` matrix `a` (row-major, consumed).
/// Returns `(eigenvalues, eigenvectors_row_major)` where column `k` (i.e. `vec[i*n+k]`) is the
/// eigenvector of `eigenvalues[k]`.
fn jacobi_symmetric(mut a: Vec<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if libm::fabs(apq) < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = libm::copysign(1.0, theta)
                    / (libm::fabs(theta) + libm::sqrt(theta * theta + 1.0));
                let c = 1.0 / libm::sqrt(t * t + 1.0);
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let eig = (0..n).map(|i| a[i * n + i]).collect();
    (eig, v)
}

/// The Cloude–Pottier `(H, A, ᾱ)` of a `3×3` Hermitian coherency matrix `t3` (row-major, complex): the
/// scattering entropy `H ∈ [0, 1]`, anisotropy `A ∈ [0, 1]`, and mean scattering angle `ᾱ` (radians).
#[must_use]
pub fn cloude_pottier(t3: &[Complex]) -> (f64, f64, f64) {
    // Embed the Hermitian T₃ = Re + i·Im in the real symmetric 6×6 [[Re, −Im],[Im, Re]]; its
    // eigenvalues are those of T₃, each doubled, with eigenvectors [v_re; v_im].
    let n = 6;
    let mut m = vec![0.0; n * n];
    for i in 0..3 {
        for j in 0..3 {
            let z = t3[i * 3 + j];
            m[i * n + j] = z.re;
            m[(i + 3) * n + (j + 3)] = z.re;
            m[i * n + (j + 3)] = -z.im;
            m[(i + 3) * n + j] = z.im;
        }
    }
    let (eig, vecs) = jacobi_symmetric(m, n);

    // Pick one column per distinct T₃ eigenvalue: sort the 6 by eigenvalue, take every other.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eig[b].total_cmp(&eig[a]));
    let mut lambdas = [0.0f64; 3];
    let mut alphas = [0.0f64; 3];
    for (idx, &col) in order.iter().step_by(2).take(3).enumerate() {
        lambdas[idx] = eig[col].max(0.0);
        // The complex eigenvector's first component: e[0] = v_re[0] + i·v_im[0] (rows 0 and 3).
        let e0 = libm::sqrt(vecs[col] * vecs[col] + vecs[3 * n + col] * vecs[3 * n + col]);
        alphas[idx] = libm::acos(e0.clamp(0.0, 1.0));
    }

    let total: f64 = lambdas.iter().sum();
    if total <= 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let p: [f64; 3] = [lambdas[0] / total, lambdas[1] / total, lambdas[2] / total];
    let log3 = |x: f64| libm::log(x) / libm::log(3.0);
    let h = -p
        .iter()
        .filter(|&&pi| pi > 1e-30)
        .map(|&pi| pi * log3(pi))
        .sum::<f64>();
    let a = if p[1] + p[2] > 1e-30 {
        (p[1] - p[2]) / (p[1] + p[2])
    } else {
        0.0
    };
    let alpha_bar = p[0] * alphas[0] + p[1] * alphas[1] + p[2] * alphas[2];
    (h, a, alpha_bar)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;

    fn c(re: f64) -> Complex {
        Complex::new(re, 0.0)
    }

    fn diag(a: f64, b: f64, d: f64) -> Vec<Complex> {
        let mut t = vec![Complex::ZERO; 9];
        t[0] = c(a);
        t[4] = c(b);
        t[8] = c(d);
        t
    }

    #[test]
    fn pauli_powers_of_canonical_scatterers() {
        // Trihedral / sphere (odd bounce): S_hh = S_vv = 1, S_hv = 0 → all power in the odd term.
        let p = pauli_powers(c(1.0), c(0.0), c(1.0));
        assert!((p[0] - 2.0).abs() < 1e-12 && p[1].abs() < 1e-12 && p[2].abs() < 1e-12);
        // Dihedral (even bounce): S_hh = 1, S_vv = −1 → all power in the even term.
        let d = pauli_powers(c(1.0), c(0.0), c(-1.0));
        assert!(d[0].abs() < 1e-12 && (d[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn h_a_alpha_of_a_pure_surface() {
        // Deterministic odd-bounce target: T₃ = diag(1,0,0) → one eigenvalue, zero entropy, α = 0.
        let (h, a, alpha) = cloude_pottier(&diag(1.0, 0.0, 0.0));
        assert!(h < 1e-9, "H {h}");
        assert!(alpha < 1e-6, "alpha {alpha}");
        assert!(a.abs() < 1e-9, "A {a}");
    }

    #[test]
    fn h_a_alpha_of_a_pure_dihedral() {
        // T₃ = diag(0,1,0): the dominant eigenvector is [0,1,0] → α = arccos(0) = 90°, entropy 0.
        let (h, _a, alpha) = cloude_pottier(&diag(0.0, 1.0, 0.0));
        assert!(h < 1e-9, "H {h}");
        assert!(
            (alpha - core::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "alpha {alpha}"
        );
    }

    #[test]
    fn h_a_alpha_of_a_fully_random_target() {
        // T₃ = (1/3)·I: three equal eigenvalues → maximum entropy H = 1 and zero anisotropy.
        let (h, a, _alpha) = cloude_pottier(&diag(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0));
        assert!((h - 1.0).abs() < 1e-9, "H {h}");
        assert!(a.abs() < 1e-9, "A {a}");
    }

    /// The Jacobi rotation against the definition of an eigen-decomposition, on a matrix that is not
    /// already diagonal.
    ///
    /// Every fixture above hands `cloude_pottier` a diagonal `T₃`, whose 6×6 embedding is diagonal
    /// too — so `off < 1e-30` breaks the sweep loop on entry and not one line of the rotation runs.
    /// That is why 82 mutants of `jacobi_symmetric` survive with every assertion in this file green.
    ///
    /// The oracle is the definition rather than a second Jacobi: `V` orthogonal and `A vₖ = λₖ vₖ`
    /// against the *original* `A` is exactly what it means for `(λ, V)` to be the decomposition, and
    /// neither statement mentions a rotation. Trace and Frobenius norm are added because they are
    /// free: a similarity transform preserves `Σλ` and `Σλ²`, and a mutated rotation that is no
    /// longer a similarity breaks them.
    #[test]
    fn jacobi_returns_an_orthogonal_basis_that_diagonalises_the_matrix() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for case in 0..200 {
            let n = 3 + case % 5;
            // Every fourth case is scaled so two diagonal entries are within 1e-9 of each other: a
            // near-degenerate pair is where `theta` is large, and the branch of the rotation formula
            // that matters there is not the one a generic matrix exercises.
            let squeeze = case % 4 == 0;
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in i..n {
                    let x = next() * if i == j { 5.0 } else { 2.0 };
                    a[i * n + j] = x;
                    a[j * n + i] = x;
                }
            }
            if squeeze && n >= 2 {
                a[n + 1] = a[0] + 1e-9;
            }
            check_eigen(&a, n);
        }
    }

    /// `(λ, V)` against the definition, for one matrix.
    fn check_eigen(a: &[f64], n: usize) {
        let (eig, vecs) = jacobi_symmetric(a.to_vec(), n);
        let scale: f64 = a.iter().fold(0.0, |m, x| {
            if m >= libm::fabs(*x) {
                m
            } else {
                libm::fabs(*x)
            }
        });

        // Vᵀ V = I.
        for p in 0..n {
            for q in 0..n {
                let dot: f64 = (0..n).map(|i| vecs[i * n + p] * vecs[i * n + q]).sum();
                let want = f64::from(u8::from(p == q));
                assert!(
                    libm::fabs(dot - want) < 1e-11,
                    "V column {p}·{q} = {dot}, expected {want}"
                );
            }
        }

        // A vₖ = λₖ vₖ, against the matrix as it was handed in.
        for k in 0..n {
            for i in 0..n {
                let av: f64 = (0..n).map(|j| a[i * n + j] * vecs[j * n + k]).sum();
                let lv = eig[k] * vecs[i * n + k];
                assert!(
                    libm::fabs(av - lv) < 1e-11 * scale,
                    "(A v)[{i}] = {av} but λ v[{i}] = {lv} for eigenvalue {k}"
                );
            }
        }

        let trace: f64 = (0..n).map(|i| a[i * n + i]).sum();
        let sum: f64 = eig.iter().sum();
        assert!(
            libm::fabs(sum - trace) < 1e-11,
            "Σλ = {sum}, tr A = {trace}"
        );
        let frob: f64 = a.iter().map(|x| x * x).sum();
        let sq: f64 = eig.iter().map(|x| x * x).sum();
        assert!(libm::fabs(sq - frob) < 1e-10, "Σλ² = {sq}, ‖A‖_F² = {frob}");
    }

    /// `(H, A, ᾱ)` of a target whose eigenvectors are complex and off-axis, against their closed form.
    ///
    /// Built as `T₃ = Σ λₖ uₖ uₖᴴ` from a unitary chosen so that every quantity is known by hand:
    /// `u₁ = (cos θ, i sin θ, 0)`, `u₂ = (i sin θ, cos θ, 0)`, `u₃ = (0, 0, 1)`. The imaginary parts
    /// put the −Im/+Im blocks of the 6×6 embedding to work, which a real `T₃` leaves at zero, and
    /// `|u₁[0]| = cos θ` makes `ᾱ` a weighted sum of `θ`, `π/2 − θ` and `π/2` — no arccos of a number
    /// the code itself produced.
    #[test]
    fn h_a_alpha_of_a_target_with_complex_eigenvectors() {
        // The three sum to 5 and not to 1, so `λₖ / total` and `λₖ · total` are different numbers.
        // With unit trace they are the same one, and the three mutants of that division survive.
        let (l1, l2, l3) = (3.0, 1.5, 0.5);
        let theta = 0.4_f64;
        let (ct, st) = (theta.cos(), theta.sin());
        let mut t = vec![Complex::ZERO; 9];
        t[0] = c(l1 * ct * ct + l2 * st * st);
        t[4] = c(l1 * st * st + l2 * ct * ct);
        t[8] = c(l3);
        t[1] = Complex::new(0.0, -st * ct * (l1 - l2));
        t[3] = t[1].conj();

        let (h, a, alpha) = cloude_pottier(&t);
        let total = l1 + l2 + l3;
        let p = [l1 / total, l2 / total, l3 / total];
        let log3 = |x: f64| x.ln() / 3.0_f64.ln();
        let want_h = -p.iter().map(|&pi| pi * log3(pi)).sum::<f64>();
        let want_a = (p[1] - p[2]) / (p[1] + p[2]);
        let half_pi = core::f64::consts::FRAC_PI_2;
        let want_alpha = p[0] * theta + p[1] * (half_pi - theta) + p[2] * half_pi;
        println!("polsar rotated target: H = {h}, A = {a}, alpha = {alpha}");
        assert!(libm::fabs(h - want_h) < 1e-12, "H {h} vs {want_h}");
        assert!(libm::fabs(a - want_a) < 1e-12, "A {a} vs {want_a}");
        assert!(
            libm::fabs(alpha - want_alpha) < 1e-9,
            "alpha {alpha} vs {want_alpha}"
        );
    }

    /// Anisotropy when the third mechanism is absent, which is where the guard on it is decided.
    ///
    /// `A = (p₂ − p₃)/(p₂ + p₃)` is guarded by `p₂ + p₃ > 1e-30`, and the sum tells a target with two
    /// mechanisms apart from one with none. Every other fixture here has all three eigenvalues equal
    /// or only one non-zero, and on both of those a product stands in for the sum unnoticed: with
    /// `λ₃ = 0` and `λ₂ > 0` the sum is 0.3 and the product is 0, so one form reports full anisotropy
    /// and the other reports none.
    #[test]
    fn anisotropy_of_a_target_with_only_two_mechanisms() {
        let (h, a, alpha) = cloude_pottier(&diag(0.7, 0.3, 0.0));
        let log3 = |x: f64| x.ln() / 3.0_f64.ln();
        let want_h = -(0.7 * log3(0.7) + 0.3 * log3(0.3));
        let want_alpha = 0.3 * core::f64::consts::FRAC_PI_2;
        println!("polsar two-mechanism target: H = {h}, A = {a}, alpha = {alpha}");
        assert!(libm::fabs(h - want_h) < 1e-12, "H {h} vs {want_h}");
        assert!(
            libm::fabs(a - 1.0) < 1e-12,
            "A {a}, expected full anisotropy"
        );
        assert!(
            libm::fabs(alpha - want_alpha) < 1e-9,
            "alpha {alpha} vs {want_alpha}"
        );
    }

    #[test]
    fn coherency_matrix_is_hermitian_and_averages() {
        // A single look gives a rank-1 T₃ (entropy 0); its trace equals the total Pauli power.
        let (hh, hv, vv) = (
            Complex::new(1.0, 0.2),
            Complex::new(0.1, -0.3),
            Complex::new(0.6, 0.4),
        );
        let t = coherency_matrix(&[(hh, hv, vv)]);
        // Two looks of the same scatterer must average back to the one-look matrix. One look alone
        // cannot see the `1/n` scale at all -- `1/1` and `1*1` are the same number.
        let twice = coherency_matrix(&[(hh, hv, vv), (hh, hv, vv)]);
        for i in 0..9 {
            assert!((twice[i] - t[i]).abs() < 1e-12, "two looks differ at {i}");
        }
        // And two different looks average, rather than sum or take the last.
        let other = (
            Complex::new(-0.4, 0.9),
            Complex::ZERO,
            Complex::new(0.2, -0.1),
        );
        let mixed = coherency_matrix(&[(hh, hv, vv), other]);
        let alone = coherency_matrix(&[other]);
        for i in 0..9 {
            let want = (t[i] + alone[i]).scale(0.5);
            assert!((mixed[i] - want).abs() < 1e-12, "mixed look differs at {i}");
        }
        // Hermitian: T[i,j] = conj(T[j,i]).
        for i in 0..3 {
            for j in 0..3 {
                assert!((t[i * 3 + j] - t[j * 3 + i].conj()).abs() < 1e-12);
            }
        }
        let trace = t[0].re + t[4].re + t[8].re;
        let total: f64 = pauli_powers(hh, hv, vv).iter().sum();
        assert!((trace - total).abs() < 1e-9, "{trace} vs {total}");
        let (h, _, _) = cloude_pottier(&t);
        assert!(h < 1e-6, "single-look entropy {h}");
    }
}
