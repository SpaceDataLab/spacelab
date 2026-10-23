//! The square-root information filter, and the reason it exists: it keeps the digits the normal
//! equations throw away on an ill-conditioned problem.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::float_cmp
)]

use spacelab_estimation::{Srif, SrifError};

/// Naive normal-equations solve — form `HᵀH`, invert by Gaussian elimination — the thing we do NOT
/// do, here only to measure how much worse it is.
fn normal_equations_solve<const N: usize>(rows: &[([f64; N], f64)]) -> [f64; N] {
    let mut ata = [[0.0; N]; N];
    let mut atb = [0.0; N];
    for (h, z) in rows {
        for i in 0..N {
            for j in 0..N {
                ata[i][j] += h[i] * h[j];
            }
            atb[i] += h[i] * z;
        }
    }
    solve_linear(&mut ata, &mut atb)
}

/// Gaussian elimination with partial pivoting, solving `A x = b` in place.
fn solve_linear<const N: usize>(a: &mut [[f64; N]; N], b: &mut [f64; N]) -> [f64; N] {
    for col in 0..N {
        let mut piv = col;
        for r in (col + 1)..N {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        a.swap(piv, col);
        b.swap(piv, col);
        for r in (col + 1)..N {
            let f = a[r][col] / a[col][col];
            for c in col..N {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0; N];
    for i in (0..N).rev() {
        let mut acc = b[i];
        for j in (i + 1)..N {
            acc -= a[i][j] * x[j];
        }
        x[i] = acc / a[i][i];
    }
    x
}

#[test]
fn solves_a_well_conditioned_least_squares_problem() {
    // Fit x = [3, -2, 5] to noiseless rows; the SRIF must recover it exactly.
    let x_true = [3.0, -2.0, 5.0];
    let hs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 2.0],
    ];
    let mut srif = Srif::<3>::new();
    for h in hs {
        let z = h[0] * x_true[0] + h[1] * x_true[1] + h[2] * x_true[2];
        srif.add_row(&h, z, 1.0);
    }
    let x = srif.solve().expect("full rank");
    for i in 0..3 {
        assert!(
            (x[i] - x_true[i]).abs() < 1e-13,
            "x[{i}] = {} vs {}",
            x[i],
            x_true[i]
        );
    }
}

#[test]
fn qr_keeps_the_digits_the_normal_equations_lose() {
    // A Vandermonde fit at tightly-clustered points is ill-conditioned. The SRIF (Givens QR) works
    // at cond(H); the normal equations work at cond(H)², so they lose about twice as many digits.
    // This is the whole argument of notes/14 §1, in Rust.
    const N: usize = 4;
    let x_true = [1.0, -2.0, 3.0, -1.0];
    // 12 points in [1.00, 1.11]. `cond(H) = 3.405e5`, measured — not the `1e8+` this comment used to
    // claim, and the errors below are what settles it: `ε·cond(H) = 7.6e-11` and `ε·cond(H)² = 2.6e-5`
    // bracket the QR and normal-equations errors this test actually prints (6.1e-12 and 1.6e-5). At
    // `cond(H) = 1e8` the normal equations would have no correct digits at all, which is not what is
    // observed here. `derivations/normal_equations_vs_qr.m` is where that regime is measured: it
    // builds `H` from prescribed singular values rather than hoping a Vandermonde reaches them.
    let mut rows: Vec<([f64; N], f64)> = Vec::new();
    for k in 0..12 {
        let t = 1.0 + 0.01 * f64::from(k);
        let h = [1.0, t, t * t, t * t * t];
        let z = h[0] * x_true[0] + h[1] * x_true[1] + h[2] * x_true[2] + h[3] * x_true[3];
        rows.push((h, z));
    }

    let mut srif = Srif::<N>::new();
    for (h, z) in &rows {
        srif.add_row(h, *z, 1.0);
    }
    let x_qr = srif.solve().expect("full rank");
    let x_ne = normal_equations_solve(&rows);

    let err = |x: &[f64; N]| (0..N).map(|i| (x[i] - x_true[i]).abs()).fold(0.0, f64::max);
    let (e_qr, e_ne) = (err(&x_qr), err(&x_ne));
    println!("ill-conditioned Vandermonde: QR error {e_qr:.2e}, normal-equations error {e_ne:.2e}");

    // QR must be far more accurate — orders of magnitude, the cond vs cond² gap.
    assert!(e_qr < 1e-4, "QR should stay accurate, got {e_qr:.2e}");
    assert!(
        e_ne > 100.0 * e_qr,
        "normal equations should be far worse; QR {e_qr:.2e}, NE {e_ne:.2e}"
    );
}

#[test]
fn the_covariance_is_the_inverse_information_matrix() {
    // P = (HᵀWH)⁻¹. Build a small weighted problem, compare the SRIF covariance to the explicit
    // inverse of the information matrix.
    let hs = [
        ([1.0, 0.5], 1.0, 2.0),
        ([0.3, 1.0], 2.0, 1.0),
        ([1.0, 1.0], 0.5, 0.5),
        ([-0.5, 1.0], 1.5, 3.0),
    ];
    let mut srif = Srif::<2>::new();
    let mut info = [[0.0; 2]; 2];
    for (h, z, w) in hs {
        srif.add_row(&h, z, w);
        for i in 0..2 {
            for j in 0..2 {
                info[i][j] += w * h[i] * h[j];
            }
        }
    }
    let p = srif.covariance().expect("invertible");
    // invert the 2×2 information matrix analytically
    let det = info[0][0] * info[1][1] - info[0][1] * info[1][0];
    let inv = [
        [info[1][1] / det, -info[0][1] / det],
        [-info[1][0] / det, info[0][0] / det],
    ];
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (p[i][j] - inv[i][j]).abs() < 1e-12,
                "P[{i}][{j}] {} vs {}",
                p[i][j],
                inv[i][j]
            );
        }
    }
}

#[test]
fn sequential_equals_batch_regardless_of_row_order() {
    // The information square root is the Cholesky factor of HᵀWH, which is unique with a positive
    // diagonal — so folding the rows in any order gives the same R.
    let hs = [
        ([1.0, 2.0, 0.5], 3.0, 1.0),
        ([0.0, 1.0, 1.0], 1.0, 2.0),
        ([2.0, -1.0, 0.0], 0.5, 0.5),
        ([1.0, 1.0, 1.0], 2.0, 1.5),
        ([-1.0, 0.5, 2.0], 1.0, 0.8),
    ];
    let mut a = Srif::<3>::new();
    for (h, z, w) in hs {
        a.add_row(&h, z, w);
    }
    let mut b = Srif::<3>::new();
    for (h, z, w) in hs.iter().rev() {
        b.add_row(h, *z, *w);
    }
    let (ra, rb) = (a.information_sqrt(), b.information_sqrt());
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (ra[i][j] - rb[i][j]).abs() < 1e-11,
                "R[{i}][{j}] order-dependent"
            );
        }
    }
    let (xa, xb) = (a.solve().unwrap(), b.solve().unwrap());
    for i in 0..3 {
        assert!((xa[i] - xb[i]).abs() < 1e-12);
    }
}

#[test]
fn an_unobserved_direction_is_reported_not_invented() {
    // Only the first component is ever excited; the second is unobservable.
    let mut srif = Srif::<2>::new();
    srif.add_row(&[1.0, 0.0], 5.0, 1.0);
    srif.add_row(&[2.0, 0.0], 10.0, 1.0);
    assert_eq!(srif.solve(), Err(SrifError::RankDeficient { direction: 1 }));
}

#[test]
fn an_a_priori_estimate_is_folded_in_like_a_measurement() {
    // With no data, the solution is the prior. Give a prior x0 = [2, -1] with unit information; solve
    // before any measurement returns the prior.
    let r0 = [[1.0, 0.0], [0.0, 1.0]]; // chol(P0⁻¹) = I, i.e. unit prior covariance
    let x0 = [2.0, -1.0];
    let srif = Srif::<2>::with_prior(&x0, &r0);
    let x = srif.solve().expect("prior is full rank");
    assert!(
        (x[0] - 2.0).abs() < 1e-14 && (x[1] + 1.0).abs() < 1e-14,
        "prior not recovered: {x:?}"
    );
}
