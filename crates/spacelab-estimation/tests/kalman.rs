//! The square-root Kalman filter, checked against the batch SRIF it shares a kernel with and against
//! the textbook Kalman formulas — never against itself.
//!
//! * **Sequential = batch**: for a static state with no process noise, folding measurements one at a
//!   time through the filter gives exactly the batch [`Srif`] estimate and covariance. This is the
//!   duality of the two forms, made a test.
//! * **Time / measurement updates** reproduce `Φ P Φᵀ + Q` and the direct Kalman gain / Joseph
//!   covariance to rounding.
//! * **Positive-definiteness** survives a run of updates — the property the square root exists to
//!   guarantee.

#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use spacelab_estimation::kalman::SquareRootKalman;
use spacelab_estimation::srif::Srif;

fn dot<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    (0..N).map(|i| a[i] * b[i]).sum()
}

fn matvec<const N: usize>(a: &[[f64; N]; N], x: &[f64; N]) -> [f64; N] {
    let mut y = [0.0; N];
    for i in 0..N {
        for j in 0..N {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

fn matmul<const N: usize>(a: &[[f64; N]; N], b: &[[f64; N]; N]) -> [[f64; N]; N] {
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

fn transpose<const N: usize>(a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut t = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            t[i][j] = a[j][i];
        }
    }
    t
}

#[test]
fn the_sequential_filter_matches_the_batch_srif() {
    // Static state (Φ = I, Q = 0): the covariance-form filter and the information-form batch solver
    // must land on the same estimate and the same covariance, from the same weak prior.
    let x_true = [1.0, 2.0, 3.0];
    let rows: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.4, 0.7],
    ];
    let noise = 0.01_f64; // measurement variance
    let p0 = [[1e6, 0.0, 0.0], [0.0, 1e6, 0.0], [0.0, 0.0, 1e6]];

    let mut kf = SquareRootKalman::from_covariance([0.0; 3], &p0).unwrap();
    // prior information square root = chol(P₀⁻¹) = diag(1e-3), upper triangular
    let r0 = [[1e-3, 0.0, 0.0], [0.0, 1e-3, 0.0], [0.0, 0.0, 1e-3]];
    let mut srif = Srif::<3>::with_prior(&[0.0; 3], &r0);

    for h in &rows {
        let z = dot(h, &x_true);
        kf.measurement_update(h, z - dot(h, &kf.x), noise);
        srif.add_row(h, z, 1.0 / noise);
    }

    let x_srif = srif.solve().unwrap();
    let p_srif = srif.covariance().unwrap();
    let p_kf = kf.covariance();
    for i in 0..3 {
        assert!(
            (kf.x[i] - x_srif[i]).abs() < 1e-8,
            "state[{i}]: filter {} vs SRIF {}",
            kf.x[i],
            x_srif[i]
        );
        for j in 0..3 {
            assert!(
                (p_kf[i][j] - p_srif[i][j]).abs() < 1e-10,
                "cov[{i}][{j}]: {} vs {}",
                p_kf[i][j],
                p_srif[i][j]
            );
        }
    }
}

#[test]
fn the_time_update_reproduces_phi_p_phi_transpose_plus_q() {
    let s0 = [[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]]; // a lower-triangular square root
    let stm = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.05, 0.0, 1.0]];
    let sq = [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]];

    let mut kf = SquareRootKalman::new([0.0; 3], s0);
    kf.time_update(&stm, &sq);
    let got = kf.covariance();

    // direct: Φ (S S ᵀ) Φᵀ + S_Q S_Qᵀ
    let p = matmul(&s0, &transpose(&s0));
    let phi_p = matmul(&stm, &p);
    let mut want = matmul(&phi_p, &transpose(&stm));
    let q = matmul(&sq, &transpose(&sq));
    for i in 0..3 {
        for j in 0..3 {
            want[i][j] += q[i][j];
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (got[i][j] - want[i][j]).abs() < 1e-10,
                "P⁻[{i}][{j}]: {} vs {}",
                got[i][j],
                want[i][j]
            );
        }
    }
}

#[test]
fn the_measurement_update_reproduces_the_direct_kalman_gain() {
    let s0 = [[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]];
    let x0 = [0.1, 0.2, 0.3];
    let h = [1.0, 0.5, 0.2];
    let r = 0.5_f64;
    let innovation = 1.3_f64;

    let mut kf = SquareRootKalman::new(x0, s0);
    let f = kf.measurement_update(&h, innovation, r);

    // direct Kalman: f = hPhᵀ+R, K = Phᵀ/f, x⁺ = x + Kν, P⁺ = (I − Kh)P
    let p = matmul(&s0, &transpose(&s0));
    let p_ht = matvec(&p, &h);
    let f_want = dot(&h, &p_ht) + r;
    assert!(
        (f - f_want).abs() < 1e-10,
        "innovation variance {f} vs {f_want}"
    );

    let k = [p_ht[0] / f_want, p_ht[1] / f_want, p_ht[2] / f_want];
    let x_want = [
        x0[0] + k[0] * innovation,
        x0[1] + k[1] * innovation,
        x0[2] + k[2] * innovation,
    ];
    for i in 0..3 {
        assert!(
            (kf.x[i] - x_want[i]).abs() < 1e-10,
            "x⁺[{i}]: {} vs {}",
            kf.x[i],
            x_want[i]
        );
    }

    // P⁺ = (I − K h) P
    let mut i_minus_kh = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            i_minus_kh[i][j] = f64::from(u8::from(i == j)) - k[i] * h[j];
        }
    }
    let p_want = matmul(&i_minus_kh, &p);
    let p_got = kf.covariance();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (p_got[i][j] - p_want[i][j]).abs() < 1e-10,
                "P⁺[{i}][{j}]: {} vs {}",
                p_got[i][j],
                p_want[i][j]
            );
        }
    }
}

#[test]
fn the_covariance_stays_symmetric_and_positive_definite() {
    let s0 = [[3.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.5, 0.3, 1.5]];
    let stm = [[1.0, 0.2, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.0]];
    let sq = [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]];
    let mut kf = SquareRootKalman::new([0.0; 3], s0);

    for step in 0..25 {
        kf.time_update(&stm, &sq);
        // alternate measurements along different axes so every direction is observed
        let h = match step % 3 {
            0 => [1.0, 0.0, 0.0],
            1 => [0.0, 1.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };
        kf.measurement_update(&h, 0.1, 0.05);

        let p = kf.covariance();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (p[i][j] - p[j][i]).abs() < 1e-12,
                    "asymmetry at step {step}: {} vs {}",
                    p[i][j],
                    p[j][i]
                );
            }
        }
        // positive-definite ⇔ a Cholesky factor exists
        assert!(
            SquareRootKalman::from_covariance([0.0; 3], &p).is_some(),
            "covariance lost positive-definiteness at step {step}"
        );
    }
}

/// Seeding from a covariance takes its Cholesky factor, on a factor whose entries are exact.
///
/// `L = [[2,0,0],[1,3,0],[-1,4,5]]` gives `P = L Lᵀ = [[4,2,-2],[2,10,11],[-2,11,42]]`, and every
/// intermediate — 2, 9, 25 — is a perfect square, so the recovered factor must be `L` to the bit.
/// Every other caller of `from_covariance` passes a diagonal matrix, where the inner accumulation is
/// empty and the off-diagonal division has a zero numerator: a sign flip and a remainder both leave
/// such a seed exactly where it was.
#[test]
fn seeding_from_a_full_covariance_recovers_its_cholesky_factor() {
    let l = [[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 4.0, 5.0]];
    let p = matmul(&l, &transpose(&l));
    let kf = SquareRootKalman::from_covariance([0.0; 3], &p).expect("positive-definite");
    let s = kf.covariance_sqrt();
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (s[i][j] - l[i][j]).abs() < 1e-12,
                "S[{i}][{j}] = {}, want {}",
                s[i][j],
                l[i][j]
            );
        }
    }
}

/// The stored factor stays lower-triangular with a non-negative diagonal through both updates.
///
/// This is load-bearing, not cosmetic: `consistency::normalized_squared` forward-substitutes through
/// whatever `covariance_sqrt` returns, and `new` documents its argument as a lower-triangular root.
/// Both updates end in a triangularising sweep whose only observable is this shape — `covariance()`
/// is `S Sᵀ` and cannot see it, which is why skipping the sweep entirely goes unnoticed. The sweep
/// annihilates analytically, not to the bit: the worst above-diagonal leftover here is 1.1e-16, about
/// an ulp of the factor's largest entry, so the bound is 1e-14.
#[test]
fn both_updates_leave_the_factor_lower_triangular_with_a_non_negative_diagonal() {
    let s0 = [[2.0, 0.0, 0.0], [0.5, 3.0, 0.0], [0.1, 0.2, 4.0]];
    let stm = [[1.0, 0.1, 0.0], [0.0, 1.0, 0.1], [0.05, 0.0, 1.0]];
    let sq = [[0.1, 0.0, 0.0], [0.03, 0.2, 0.0], [0.02, 0.01, 0.3]];

    let mut kf = SquareRootKalman::new([0.0; 3], s0);
    let check = |kf: &SquareRootKalman<3>, stage: &str| {
        let s = kf.covariance_sqrt();
        for i in 0..3 {
            assert!(s[i][i] >= 0.0, "{stage}: S[{i}][{i}] = {}", s[i][i]);
            for j in (i + 1)..3 {
                assert!(
                    s[i][j].abs() < 1e-14,
                    "{stage}: S[{i}][{j}] = {:e} above the diagonal",
                    s[i][j]
                );
            }
        }
        let p = kf.covariance();
        for i in 0..3 {
            for j in 0..3 {
                let ss: f64 = (0..3).map(|k| s[i][k] * s[j][k]).sum();
                assert!(
                    (p[i][j] - ss).abs() < 1e-12,
                    "{stage}: P ≠ S Sᵀ at [{i}][{j}]"
                );
            }
        }
    };

    kf.time_update(&stm, &sq);
    check(&kf, "time update");
    kf.measurement_update(&[1.0, -0.5, 0.25], 0.7, 0.04);
    check(&kf, "measurement update");
}
