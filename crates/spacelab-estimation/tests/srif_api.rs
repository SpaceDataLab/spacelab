//! The SRIF solve/covariance/diagnostics surface on a small closed-form least-squares problem.
//!
//! The existing srif tests hammer the numerical conditioning; this covers the rest of the contract —
//! `Default`, `residual_stats`, the covariance rank-deficient branch, and the `SrifError` message.

#![allow(clippy::float_cmp, clippy::unwrap_used)] // the small systems here have exact rational answers

use spacelab_estimation::{Srif, SrifError};

#[test]
fn default_is_the_empty_filter() {
    let d = Srif::<2>::default();
    assert_eq!(d.residual_stats(), (0.0, 0));
    assert_eq!(d.information_sqrt(), [[0.0; 2]; 2]);
}

#[test]
fn a_consistent_system_solves_with_no_residual_and_the_closed_form_covariance() {
    // Measure x0, x1, and x0 + x1 with unit weight; the data is exactly consistent (3 + 5 = 8).
    let mut f = Srif::<2>::new();
    f.add_row(&[1.0, 0.0], 3.0, 1.0);
    f.add_row(&[0.0, 1.0], 5.0, 1.0);
    f.add_row(&[1.0, 1.0], 8.0, 1.0);

    let x = f.solve().unwrap();
    assert!(
        (x[0] - 3.0).abs() < 1e-12 && (x[1] - 5.0).abs() < 1e-12,
        "{x:?}"
    );

    // HᵀH = [[2,1],[1,2]], so P = (HᵀH)⁻¹ = [[2,-1],[-1,2]] / 3.
    let p = f.covariance().unwrap();
    assert!((p[0][0] - 2.0 / 3.0).abs() < 1e-12);
    assert!((p[0][1] + 1.0 / 3.0).abs() < 1e-12);
    assert!((p[1][0] + 1.0 / 3.0).abs() < 1e-12);
    assert!((p[1][1] - 2.0 / 3.0).abs() < 1e-12);

    let (rss, rows) = f.residual_stats();
    assert!(
        rss < 1e-18,
        "consistent data leaves no post-fit residual, got {rss}"
    );
    assert_eq!(rows, 3);
}

#[test]
fn an_unobserved_direction_is_reported_rather_than_invented() {
    // Only x0 is measured, so x1's information diagonal is zero.
    let mut g = Srif::<2>::new();
    g.add_row(&[1.0, 0.0], 1.0, 1.0);

    assert_eq!(g.solve(), Err(SrifError::RankDeficient { direction: 1 }));
    assert_eq!(
        g.covariance(),
        Err(SrifError::RankDeficient { direction: 1 })
    );
    assert_ne!(SrifError::RankDeficient { direction: 1 }.to_string(), "");
}

/// The residual sum of squares is the post-fit cost, on a system whose cost is known in closed form.
///
/// Three unit-weight measurements of one scalar — 1, 2 and 6 — have the least-squares answer 3 and
/// a cost of `4 + 1 + 9 = 14`, which the orthogonal sweep accumulates as the part of each `z` the
/// rotation leaves behind. Nothing else here reads it: the other tests all use exactly consistent
/// data, where the cost is zero and a sign or a product is invisible.
#[test]
fn the_residual_sum_of_squares_is_the_post_fit_cost() {
    let mut f = Srif::<1>::new();
    for z in [1.0, 2.0, 6.0] {
        f.add_row(&[1.0], z, 1.0);
    }
    let (ss, rows) = f.residual_stats();
    assert!((ss - 14.0).abs() < 1e-12, "residual sum of squares {ss}");
    assert_eq!(rows, 3);
    assert!((f.solve().unwrap()[0] - 3.0).abs() < 1e-12);
}

/// `information_sqrt` returns a factor of the information matrix the design implies, not zeros.
///
/// `RᵀR = Σ wⱼ hⱼ hⱼᵀ` by construction, so measuring `x₀`, `x₁` and `x₀ + x₁` at unit weight must
/// give `[[2, 1], [1, 2]]`. The accessor's only other test is on a default-constructed filter, where
/// the honest answer and an all-zero one are the same matrix.
#[test]
fn the_information_square_root_factors_the_information_matrix() {
    let rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let mut f = Srif::<2>::new();
    for h in &rows {
        f.add_row(h, 0.0, 1.0);
    }

    let r = f.information_sqrt();
    assert_eq!(r[1][0], 0.0, "R is upper-triangular");
    let mut want = [[0.0; 2]; 2];
    for h in &rows {
        for i in 0..2 {
            for j in 0..2 {
                want[i][j] += h[i] * h[j];
            }
        }
    }
    for i in 0..2 {
        for j in 0..2 {
            let got: f64 = (0..2).map(|k| r[k][i] * r[k][j]).sum();
            assert!(
                (got - want[i][j]).abs() < 1e-12,
                "(RᵀR)[{i}][{j}] = {got}, want {}",
                want[i][j]
            );
        }
    }
}
