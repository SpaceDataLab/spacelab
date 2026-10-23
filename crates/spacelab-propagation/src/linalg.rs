//! Small dense linear algebra for the implicit integrators: LU factorisation with partial pivoting.
//!
//! Both [`crate::radau`] (a `3N×3N` Newton system, refactored each iteration) and
//! [`crate::rosenbrock`] (an `N×N` system factored once per step and reused across four stage
//! solves) need to solve dense linear systems. Separating factor from solve is what lets Rosenbrock
//! pay for one `O(N³)` factorisation and four `O(N²)` back-substitutions per step — the whole reason
//! a linearly-implicit method is cheaper than a Newton-iterated one.

// Indices here run over rows/columns of a dense matrix; range loops read more clearly than iterators.
#![allow(clippy::needless_range_loop)]

/// An in-place LU factorisation `PA = LU` (Doolittle, unit lower triangle) with the pivot record.
pub struct Lu<const K: usize> {
    /// `factors[r][c]` holds `U` on and above the diagonal, the `L` multipliers below it.
    factors: [[f64; K]; K],
    /// `piv[c]` is the row swapped into position `c` during elimination.
    piv: [usize; K],
    /// A zero pivot was hit: the matrix is singular to working precision.
    singular: bool,
}

/// Factor `m` into `PA = LU` by Gaussian elimination with partial pivoting.
#[must_use]
pub fn lu_factor<const K: usize>(mut m: [[f64; K]; K]) -> Lu<K> {
    let mut piv = [0usize; K];
    let mut singular = false;
    for col in 0..K {
        let mut p = col;
        for r in (col + 1)..K {
            if m[r][col].abs() > m[p][col].abs() {
                p = r;
            }
        }
        piv[col] = p;
        m.swap(col, p);
        if m[col][col] == 0.0 {
            singular = true;
            continue;
        }
        let d = m[col][col];
        for r in (col + 1)..K {
            let factor = m[r][col] / d;
            m[r][col] = factor;
            for c in (col + 1)..K {
                m[r][c] -= factor * m[col][c];
            }
        }
    }
    Lu {
        factors: m,
        piv,
        singular,
    }
}

impl<const K: usize> Lu<K> {
    /// Solve `A x = b` using the stored factors. Returns `None` if the factorisation was singular.
    #[must_use]
    pub fn solve(&self, mut b: [f64; K]) -> Option<[f64; K]> {
        if self.singular {
            return None;
        }
        for col in 0..K {
            b.swap(col, self.piv[col]);
        }
        // Forward substitution with the unit lower triangle.
        for r in 1..K {
            for c in 0..r {
                b[r] -= self.factors[r][c] * b[c];
            }
        }
        // Back substitution with the upper triangle.
        for r in (0..K).rev() {
            for c in (r + 1)..K {
                b[r] -= self.factors[r][c] * b[c];
            }
            b[r] /= self.factors[r][r];
        }
        Some(b)
    }
}

/// Factor-and-solve in one call, for callers that solve a system only once.
#[must_use]
pub fn lu_solve<const K: usize>(m: [[f64; K]; K], b: [f64; K]) -> Option<[f64; K]> {
    lu_factor(m).solve(b)
}

#[cfg(test)]
mod tests {
    use super::{lu_factor, lu_solve};

    /// Partial pivoting, exercised by a matrix that cannot be factored without it.
    ///
    /// Every other user of this module feeds it a Newton or Rosenbrock system `I − γhJ`, which is
    /// diagonally dominant for the step sizes those methods take, so the pivot search never actually
    /// moves a row. Two mutants of the search survived on that: `>` weakened to `==` never fires
    /// without an exact tie, leaving `p = col` and no pivoting at all; `>` widened to `>=` takes the
    /// *last* maximal row instead of the first, which is invisible unless two rows tie. Both matrices
    /// below are singular-looking to an unpivoted factorisation and ordinary to a pivoted one.
    #[test]
    fn the_pivot_search_moves_the_row_it_has_to() {
        // A zero on the diagonal: without pivoting the first elimination divides by it.
        let swap = [[0.0, 2.0], [3.0, 4.0]];
        let x = lu_solve(swap, [2.0, 7.0]).unwrap_or([f64::NAN; 2]);
        assert!(
            (x[0] - 1.0).abs() < 1e-14 && (x[1] - 1.0).abs() < 1e-14,
            "wrong solution {x:?} for a system that needs a row swap"
        );
        assert!(!lu_factor(swap).singular, "pivoting was not applied");

        // Growth: the sub-diagonal entry is a thousand times the diagonal one. Unpivoted this is
        // finite but loses every digit; pivoted it is exact.
        let growth = [[1.0e-3, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 2.0]];
        let want = [1.0, 2.0, 3.0];
        let b = [
            growth[0][0] * want[0] + growth[0][1] * want[1] + growth[0][2] * want[2],
            growth[1][0] * want[0] + growth[1][1] * want[1] + growth[1][2] * want[2],
            growth[2][0] * want[0] + growth[2][1] * want[1] + growth[2][2] * want[2],
        ];
        let got = lu_solve(growth, b).unwrap_or([f64::NAN; 3]);
        let mut worst = 0.0_f64;
        for i in 0..3 {
            worst = worst.max((got[i] - want[i]).abs());
        }
        println!("LU with pivoting: worst residual {worst:.3e}");
        assert!(
            worst < 1e-14,
            "pivoted solve is off by {worst:.3e}: {got:?}"
        );

        // A tie in the first column. `>` keeps the first maximal row, `>=` takes the last, and the
        // recorded pivot is where the two differ -- the solution does not, so it is the pivot record
        // that has to be the assertion.
        let tie = [[1.0, 2.0], [-1.0, 5.0]];
        assert_eq!(
            lu_factor(tie).piv,
            [0, 1],
            "a tie in the pivot column must keep the first row, not the last"
        );
    }

    /// A genuinely singular matrix is reported, not silently divided by.
    #[test]
    fn a_singular_matrix_is_reported() {
        let m = [[1.0, 2.0], [2.0, 4.0]];
        assert!(lu_factor(m).singular);
        assert!(lu_solve(m, [1.0, 2.0]).is_none());
    }
}
