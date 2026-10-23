//! Orbit control: the optimal gain for a relative-motion plant, and the element feedback that holds
//! an absolute orbit.
//!
//! [`crate::attitude`] closes the loop on where the vehicle points. This closes it on where the
//! vehicle *is*. Two laws, because rendezvous and station-keeping are different problems:
//!
//! [`lqr`] is the linear-quadratic regulator: given a linear plant and a quadratic cost it returns
//! the gain that minimises the cost over an infinite horizon, and nothing about it is specific to
//! orbits. [`clohessy_wiltshire_plant`] supplies the relative-motion plant it is usually pointed at.
//! This is the right tool when the target is a *state* — a rendezvous, a hold point, a formation
//! geometry — and the errors are small enough that a linearisation is honest.
//!
//! [`ElementFeedback`] is the Naasz / Gurfil Lyapunov law: given a desired set of orbital *elements*
//! it produces the thrust direction that drives them there, with a Lyapunov function that is
//! guaranteed to decrease. This is the right tool for station-keeping, where the error is a slow
//! secular drift in elements rather than a position offset, and where the linearisation LQR needs
//! would be about a point the vehicle is never at.
//!
//! # Solving the Riccati equation without an eigensolver
//!
//! The optimal gain is `K = R⁻¹BᵀP` where `P` solves the continuous-time algebraic Riccati equation
//!
//! ```text
//! AᵀP + PA − P B R⁻¹Bᵀ P + Q = 0
//! ```
//!
//! which has up to `2ⁿ` solutions, exactly one of them stabilising. The textbook route selects it
//! from the stable invariant subspace of the Hamiltonian
//!
//! ```text
//! H = [  A   −G ]        G = B R⁻¹Bᵀ
//!     [ −Q   −Aᵀ]
//! ```
//!
//! by a Schur decomposition, which this workspace does not have and which is not worth acquiring for
//! a `12×12` matrix. [`lqr`] uses the **matrix sign function** instead (Roberts 1980; Byers 1987):
//! Newton's iteration `S ← ½(S + S⁻¹)` converges to `sign(H)`, whose `−1` eigenspace is exactly the
//! stable invariant subspace, and `P` falls out of one overdetermined linear system. It needs
//! nothing but LU — [`spacelab_propagation::linalg`] — and it converges quadratically.
//!
//! What it does not do is *promise* it converged to the right thing, so [`lqr`] finishes by
//! substituting `P` back into the Riccati equation and refusing the answer if the relative residual
//! is not at rounding level. An iteration that returns a matrix solving a different equation is the
//! failure mode that matters here, and it is cheap to rule out.
//!
//! `derivations/lqr_riccati.mac` supplies the oracles: exact closed forms for the scalar case, the
//! double integrator and the Clohessy–Wiltshire cross-track block, and — for the full `6×6` — the
//! closed-loop characteristic polynomial derived from the Hamiltonian's spectrum alone, never from
//! `P`. That last one is what makes the check independent: the optimal closed loop's eigenvalues are
//! the stable half of `H`'s, so a gain that solved the wrong branch would reproduce `P`'s own
//! arithmetic and still fail the polynomial.

// The matrices (A, B, Q, R, G, H, P, K, S) and the row/column indices are single letters because
// that is exactly the notation of the equations, and range loops over a dense matrix's indices read
// more clearly than iterator chains -- the same two allows `spacelab_propagation::linalg` and
// `spacelab_astro::cw` carry, for the same reason.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

use spacelab_propagation::{
    dsst::{auxiliary::DsstAuxiliary, gaussian::variational_matrix},
    linalg::lu_factor,
};

/// Why an optimal gain could not be produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LqrError {
    /// An entry of `A`, `B`, `Q` or `R` is not finite.
    NonFinite,
    /// `Q` or `R` is not symmetric. Both are quadratic-form weights; an asymmetric one means the
    /// caller wrote a different cost from the one they meant, and only the symmetric part would
    /// have any effect.
    Asymmetric,
    /// `R` is singular, so `R⁻¹BᵀP` does not exist. Physically: some control direction is free.
    SingularControlWeight,
    /// The sign iteration hit a singular matrix, which for a Hamiltonian means an eigenvalue exactly
    /// on the imaginary axis. `sign(H)` is undefined there and no amount of iterating produces one.
    ImaginaryAxisEigenvalue,
    /// The sign iteration did not converge within [`MAX_SIGN_ITERATIONS`].
    NoConvergence,
    /// The stable invariant subspace is not a graph over the state space, so no `P` exists at all.
    /// This is what an **unstabilisable** plant looks like from here: a mode the control cannot reach
    /// and the dynamics do not damp has infinite cost, so the optimal `P` is unbounded and the
    /// subspace has no component along that state. Measured on `A = diag(1, −1)`, `B = [0; 1]`.
    NoRiccatiSolution,
    /// A `P` was produced and it does not solve the Riccati equation. The iteration converged to
    /// something, and that something is not the answer. This is also the second net under
    /// [`LqrError::NoRiccatiSolution`]: a *nearly* rank-deficient extraction has no zero pivot to
    /// trip on and instead yields a large `P` with a residual nowhere near rounding.
    RiccatiResidual,
}

/// Iteration cap for the matrix sign function. Newton converges quadratically once it is near the
/// answer, so a well-posed problem takes well under 20; this bound exists to bound a pathological
/// one rather than to be approached.
pub const MAX_SIGN_ITERATIONS: usize = 100;

/// Relative Frobenius change at which the sign iteration is called converged.
const SIGN_TOLERANCE: f64 = 1e-14;

/// The largest relative Riccati residual [`lqr`] will hand back. Set well above the `1e-15` a
/// converged solve actually achieves and well below anything that would change a gain's behaviour,
/// so it separates "converged" from "converged to the wrong thing" and not much else.
const RESIDUAL_TOLERANCE: f64 = 1e-9;

/// The infinite-horizon optimal state feedback for `ẋ = Ax + Bu` under `J = ∫ (xᵀQx + uᵀRu) dt`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lqr<const N: usize, const M: usize> {
    gain: [[f64; N]; M],
    riccati: [[f64; N]; N],
    residual: f64,
}

impl<const N: usize, const M: usize> Lqr<N, M> {
    /// `K`, `M × N`: the optimal control is `u = −K x`.
    #[must_use]
    pub const fn gain(&self) -> &[[f64; N]; M] {
        &self.gain
    }

    /// `P`, the solution of the Riccati equation. `xᵀPx` is the optimal cost-to-go from `x`.
    #[must_use]
    pub const fn riccati(&self) -> &[[f64; N]; N] {
        &self.riccati
    }

    /// The relative Riccati residual this solution was accepted at — the measured accuracy of the
    /// gain, not a promise about it.
    #[must_use]
    pub const fn residual(&self) -> f64 {
        self.residual
    }

    /// `u = −K x`.
    #[must_use]
    pub fn control(&self, state: &[f64; N]) -> [f64; M] {
        let mut u = [0.0; M];
        for (i, ui) in u.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..N {
                acc -= self.gain[i][j] * state[j];
            }
            *ui = acc;
        }
        u
    }

    /// The closed-loop dynamics matrix `A − BK`.
    #[must_use]
    pub fn closed_loop(&self, a: &[[f64; N]; N], b: &[[f64; M]; N]) -> [[f64; N]; N] {
        let mut cl = *a;
        for i in 0..N {
            for j in 0..N {
                for k in 0..M {
                    cl[i][j] -= b[i][k] * self.gain[k][j];
                }
            }
        }
        cl
    }
}

/// The optimal gain for `ẋ = Ax + Bu`, `J = ∫ (xᵀQx + uᵀRu) dt`.
///
/// `b` is `N × M` in row-major order: `b[state][control]`. `N2` must be `2 * N` — the Hamiltonian is
/// twice the state dimension and stable Rust cannot compute that from `N`, so the caller states it
/// and this checks it, exactly as [`spacelab_propagation`]'s Van Loan discretisation does.
///
/// # Errors
///
/// Every variant of [`LqrError`]. The one worth planning for is [`LqrError::NoRiccatiSolution`],
/// which is what an unstabilisable plant looks like from here — no gain exists, and none is
/// invented.
///
/// # Panics
///
/// If `N2 != 2 * N`. That is a mismatch between two const parameters at the call site, so it fails
/// at once rather than being threaded through the error type.
pub fn lqr<const N: usize, const M: usize, const N2: usize>(
    a: &[[f64; N]; N],
    b: &[[f64; M]; N],
    q: &[[f64; N]; N],
    r: &[[f64; M]; M],
) -> Result<Lqr<N, M>, LqrError> {
    assert!(N2 == 2 * N, "lqr: N2 must equal 2 * N");

    let finite = |m: &[f64]| m.iter().all(|x| x.is_finite());
    if !(a.iter().all(|row| finite(row))
        && b.iter().all(|row| finite(row))
        && q.iter().all(|row| finite(row))
        && r.iter().all(|row| finite(row)))
    {
        return Err(LqrError::NonFinite);
    }
    if !is_symmetric(q) || !is_symmetric(r) {
        return Err(LqrError::Asymmetric);
    }

    let (ham, g, r_inv_bt) = hamiltonian::<N, M, N2>(a, b, q, r)?;

    let (sign, _iterations) = matrix_sign::<N2>(&ham)?;
    let p = stable_subspace_solution::<N, N2>(&sign)?;

    let residual = riccati_residual(a, q, &g, &p);
    if !(residual.is_finite() && residual <= RESIDUAL_TOLERANCE) {
        return Err(LqrError::RiccatiResidual);
    }

    // K = (R⁻¹Bᵀ) P.
    let mut gain = [[0.0; N]; M];
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += r_inv_bt[i][k] * p[k][j];
            }
            gain[i][j] = acc;
        }
    }

    Ok(Lqr {
        gain,
        riccati: p,
        residual,
    })
}

/// The Hamiltonian, the `G` that built it, and the `R⁻¹Bᵀ` that built `G`.
type HamiltonianParts<const N: usize, const M: usize, const N2: usize> =
    ([[f64; N2]; N2], [[f64; N]; N], [[f64; N]; M]);

/// The Hamiltonian `H = [[A, −G], [−Q, −Aᵀ]]` of the LQR problem, with `G = B R⁻¹Bᵀ` and the
/// `R⁻¹Bᵀ` that both `G` and the gain are built from.
///
/// Separate from [`lqr`] so a test can drive [`matrix_sign`] with the same matrix production uses,
/// rather than a transcription of this layout that would then have to be kept in step with it.
///
/// # Errors
///
/// [`LqrError::SingularControlWeight`] if `R` cannot be factored.
fn hamiltonian<const N: usize, const M: usize, const N2: usize>(
    a: &[[f64; N]; N],
    b: &[[f64; M]; N],
    q: &[[f64; N]; N],
    r: &[[f64; M]; M],
) -> Result<HamiltonianParts<N, M, N2>, LqrError> {
    // R⁻¹Bᵀ, M × N — the only place R is inverted, and it is reused for both G and the gain.
    let r_lu = lu_factor(*r);
    let mut r_inv_bt = [[0.0; N]; M];
    for j in 0..N {
        let mut col = [0.0; M];
        for (k, ck) in col.iter_mut().enumerate() {
            *ck = b[j][k];
        }
        let solved = r_lu.solve(col).ok_or(LqrError::SingularControlWeight)?;
        for (k, sk) in solved.iter().enumerate() {
            r_inv_bt[k][j] = *sk;
        }
    }

    // G = B (R⁻¹Bᵀ), N × N, symmetric positive semi-definite.
    let mut g = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..M {
                acc += b[i][k] * r_inv_bt[k][j];
            }
            g[i][j] = acc;
        }
    }

    let mut ham = [[0.0; N2]; N2];
    for i in 0..N {
        for j in 0..N {
            ham[i][j] = a[i][j];
            ham[i][N + j] = -g[i][j];
            ham[N + i][j] = -q[i][j];
            ham[N + i][N + j] = -a[j][i];
        }
    }

    Ok((ham, g, r_inv_bt))
}

/// `sign(H)` by Newton's iteration `S ← ½(cS + c⁻¹S⁻¹)`, with Higham's norm scaling
/// `c = √(‖S⁻¹‖/‖S‖)`, switched off near convergence.
///
/// The scaling is kept for accuracy, not for speed, because that is the way the measurement went.
/// Against `c ≡ 1` on the two Hamiltonians the tests solve, it *costs* four steps on the `12 × 12`
/// Clohessy–Wiltshire one (15 against 11) and saves three on the `4 × 4` double integrator (3
/// against 6); the Riccati residual it leaves is the better one in both directions — `8.89e-14`
/// against `9.27e-14`, and `1.51e-16` against `3.08e-16`. So the textbook rationale, that scaling
/// buys iterations when the spectrum is badly spread, does not survive contact with a CW
/// Hamiltonian whose eigenvalues span `n` against `1/√ρ`. That case is exactly where it is
/// slowest.
///
/// The switch-off is Higham's safeguard against perturbing an already-converged matrix, and on
/// both plants here it never fires before the convergence test does — `change` falls from `1e-1`
/// to below the tolerance inside two quadratic steps. Removing it was measured to change neither
/// step count. It stays because a plant that approaches the answer linearly is exactly the one
/// this function will meet outside its own tests.
///
/// Returns the iteration count with the matrix, because every claim above is about cost and there
/// is no other way to check one: scaled or unscaled, the iteration lands on the same `sign(H)`,
/// and only the number of steps to get there separates them.
fn matrix_sign<const K: usize>(h: &[[f64; K]; K]) -> Result<([[f64; K]; K], usize), LqrError> {
    let mut s = *h;
    for iteration in 0..MAX_SIGN_ITERATIONS {
        let inv = invert::<K>(&s).ok_or(LqrError::ImaginaryAxisEigenvalue)?;
        let change = frobenius_difference(&s, &inv);
        let scale = if change > 1e-2 {
            libm::sqrt(frobenius(&inv) / frobenius(&s))
        } else {
            1.0
        };
        let mut next = [[0.0; K]; K];
        for i in 0..K {
            for j in 0..K {
                next[i][j] = f64::midpoint(scale * s[i][j], inv[i][j] / scale);
            }
        }
        let step = frobenius_difference(&next, &s);
        let size = frobenius(&next);
        s = next;
        if !size.is_finite() {
            return Err(LqrError::ImaginaryAxisEigenvalue);
        }
        if step <= SIGN_TOLERANCE * size {
            return Ok((s, iteration + 1));
        }
    }
    Err(LqrError::NoConvergence)
}

/// Recover `P` from `sign(H)`.
///
/// The stable invariant subspace of `H` is the `−1` eigenspace of `S = sign(H)`, i.e. the null space
/// of `S + I`, and it is spanned by `[I; P]`. So `(S + I) [I; P] = 0`, which splits into
///
/// ```text
/// [ S₁₂     ] P = − [ S₁₁ + I ]
/// [ S₂₂ + I ]       [ S₂₁     ]
/// ```
///
/// — `2N` equations in `N` unknown columns, consistent in exact arithmetic and overdetermined in
/// `f64`. Solved in the least-squares sense through the normal equations, which is the standard
/// treatment (Byers 1987) and is well conditioned here because the two blocks together have full
/// column rank whenever `H` has no imaginary-axis eigenvalue.
fn stable_subspace_solution<const N: usize, const N2: usize>(
    sign: &[[f64; N2]; N2],
) -> Result<[[f64; N]; N], LqrError> {
    let mut w = [[0.0; N]; N2];
    let mut v = [[0.0; N]; N2];
    for i in 0..N {
        for j in 0..N {
            w[i][j] = sign[i][N + j];
            w[N + i][j] = sign[N + i][N + j] + if i == j { 1.0 } else { 0.0 };
            v[i][j] = -(sign[i][j] + if i == j { 1.0 } else { 0.0 });
            v[N + i][j] = -sign[N + i][j];
        }
    }

    let mut wtw = [[0.0; N]; N];
    let mut wtv = [[0.0; N]; N];
    for i in 0..N {
        for j in 0..N {
            let mut nn = 0.0;
            let mut nv = 0.0;
            for k in 0..N2 {
                nn += w[k][i] * w[k][j];
                nv += w[k][i] * v[k][j];
            }
            wtw[i][j] = nn;
            wtv[i][j] = nv;
        }
    }

    let lu = lu_factor(wtw);
    let mut p = [[0.0; N]; N];
    for j in 0..N {
        let mut rhs = [0.0; N];
        for (i, ri) in rhs.iter_mut().enumerate() {
            *ri = wtv[i][j];
        }
        let col = lu.solve(rhs).ok_or(LqrError::NoRiccatiSolution)?;
        for (i, ci) in col.iter().enumerate() {
            p[i][j] = *ci;
        }
    }

    // The exact solution is symmetric; the normal equations do not enforce it, and the asymmetry
    // that survives is a direct measure of the solve's own error. Averaging is the standard
    // projection onto the symmetric matrices and can only reduce the residual.
    for i in 0..N {
        for j in (i + 1)..N {
            let mean = f64::midpoint(p[i][j], p[j][i]);
            p[i][j] = mean;
            p[j][i] = mean;
        }
    }
    Ok(p)
}

/// `‖AᵀP + PA − PGP + Q‖_F` relative to the size of the terms that cancel to produce it.
///
/// Normalising by the sum of the term norms rather than by `‖Q‖` alone is what makes this a
/// *relative* residual: the four terms are individually enormous compared with their sum, and a
/// residual measured against the smallest of them would read as a failure on a correct solve.
fn riccati_residual<const N: usize>(
    a: &[[f64; N]; N],
    q: &[[f64; N]; N],
    g: &[[f64; N]; N],
    p: &[[f64; N]; N],
) -> f64 {
    let mut scale = 0.0;
    let mut residual = 0.0;
    for i in 0..N {
        for j in 0..N {
            let mut atp = 0.0;
            let mut pa = 0.0;
            for k in 0..N {
                atp += a[k][i] * p[k][j];
                pa += p[i][k] * a[k][j];
            }
            let mut pgp = 0.0;
            for k in 0..N {
                let mut row = 0.0;
                for m in 0..N {
                    row += p[i][m] * g[m][k];
                }
                pgp += row * p[k][j];
            }
            let term = atp + pa - pgp + q[i][j];
            residual += term * term;
            scale += atp * atp + pa * pa + pgp * pgp + q[i][j] * q[i][j];
        }
    }
    if scale == 0.0 {
        return 0.0;
    }
    libm::sqrt(residual / scale)
}

/// Exact equality is the predicate on purpose: the question is whether the caller wrote a symmetric
/// weight matrix, not whether one is symmetric to some tolerance. A near-symmetric `Q` is a typo
/// with a plausible-looking cost attached, and it should be reported rather than quietly symmetrised.
#[allow(clippy::float_cmp)]
fn is_symmetric<const K: usize>(m: &[[f64; K]; K]) -> bool {
    for i in 0..K {
        for j in (i + 1)..K {
            if m[i][j] != m[j][i] {
                return false;
            }
        }
    }
    true
}

fn frobenius<const K: usize>(m: &[[f64; K]; K]) -> f64 {
    let mut acc = 0.0;
    for row in m {
        for x in row {
            acc += x * x;
        }
    }
    libm::sqrt(acc)
}

fn frobenius_difference<const K: usize>(a: &[[f64; K]; K], b: &[[f64; K]; K]) -> f64 {
    let mut acc = 0.0;
    for i in 0..K {
        for j in 0..K {
            let d = a[i][j] - b[i][j];
            acc += d * d;
        }
    }
    libm::sqrt(acc)
}

fn invert<const K: usize>(m: &[[f64; K]; K]) -> Option<[[f64; K]; K]> {
    let lu = lu_factor(*m);
    let mut inv = [[0.0; K]; K];
    for j in 0..K {
        let mut e = [0.0; K];
        e[j] = 1.0;
        let col = lu.solve(e)?;
        for (i, ci) in col.iter().enumerate() {
            inv[i][j] = *ci;
        }
    }
    Some(inv)
}

/// The monic characteristic polynomial of `m`: coefficients of `λ⁰ … λ^{N−1}`, with the `λ^N`
/// coefficient equal to 1 and therefore not returned.
///
/// Faddeev–LeVerrier, which needs only matrix products and traces — no eigensolver, no root finding.
/// That is what lets the closed-loop spectrum be compared against a CAS by *coefficients*, which are
/// real and order-free, rather than by eigenvalues, which are neither.
///
/// The recurrence divides by `k`, so it loses roughly `N` digits of the smallest coefficient on an
/// `N × N` matrix whose eigenvalues are widely spread. On the `6 × 6` closed loops here that is
/// measured at `1e-12` relative against a 40-digit reference; for a much larger or much stiffer
/// matrix it would be the wrong tool.
#[must_use]
pub fn characteristic_polynomial<const N: usize>(m: &[[f64; N]; N]) -> [f64; N] {
    let mut coefficients = [0.0; N];
    let mut work = [[0.0; N]; N];
    let mut c = 1.0;
    for k in 1..=N {
        // work ← m · work + c I, i.e. the next Faddeev matrix.
        let mut next = [[0.0; N]; N];
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for l in 0..N {
                    acc += m[i][l] * work[l][j];
                }
                next[i][j] = acc;
            }
            next[i][i] += c;
        }
        // work ← m · next; its trace gives the next coefficient.
        let mut product = [[0.0; N]; N];
        let mut trace = 0.0;
        for i in 0..N {
            for j in 0..N {
                let mut acc = 0.0;
                for l in 0..N {
                    acc += m[i][l] * next[l][j];
                }
                product[i][j] = acc;
            }
            trace += product[i][i];
        }
        // `k` runs to `N`, the matrix dimension: exactly representable, and a matrix wider than
        // 2^53 does not fit in memory.
        #[allow(clippy::cast_precision_loss)]
        let step = k as f64;
        c = -trace / step;
        coefficients[N - k] = c;
        work = next;
    }
    coefficients
}

/// A linear time-invariant plant `ẋ = Ax + Bu`: the pair `(A, B)`, with `B` in row-major
/// `b[state][control]` order.
pub type LinearPlant<const N: usize, const M: usize> = ([[f64; N]; N], [[f64; M]; N]);

/// The Clohessy–Wiltshire plant `(A, B)` for a chief at mean motion `n`, in the LVLH frame of
/// `spacelab_astro::cw`: `x` radial, `y` along-track, `z` cross-track.
///
/// ```text
/// ẍ = 3n²x + 2nẏ        ÿ = −2nẋ        z̈ = −n²z
/// ```
///
/// `B` is `[0; I₃]`: the control is an acceleration in the same length unit as the state, so a state
/// in kilometres gives a control in km/s² and `R` must be weighted to match.
///
/// # The curvilinear question
///
/// The *dynamics* are the same in the rectilinear and curvilinear forms — the curvilinear form
/// (`cw_propagate_curvilinear`) is a change of coordinates that removes the chief orbit's curvature,
/// not a different linearisation. So there is one gain, and the choice is about which coordinates
/// the state fed to it is expressed in. Past a few kilometres of along-track separation that should
/// be the curvilinear one, which is measured two orders more accurate at 100 km; the gain does not
/// change.
///
/// `None` unless `n` is finite and strictly positive.
#[must_use]
pub fn clohessy_wiltshire_plant(mean_motion: f64) -> Option<LinearPlant<6, 3>> {
    if !(mean_motion.is_finite() && mean_motion > 0.0) {
        return None;
    }
    let n = mean_motion;
    let a = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [3.0 * n * n, 0.0, 0.0, 0.0, 2.0 * n, 0.0],
        [0.0, 0.0, 0.0, -2.0 * n, 0.0, 0.0],
        [0.0, 0.0, -n * n, 0.0, 0.0, 0.0],
    ];
    let b = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    Some((a, b))
}

/// The Naasz / Gurfil Lyapunov element feedback: the thrust direction that drives a set of
/// equinoctial elements towards their targets.
///
/// With `V = ½ Δœᵀ W Δœ` and the Gauss variational equations `œ̇ = B(œ) u`,
///
/// ```text
/// V̇ = Δœᵀ W B u = dᵀu        d = Bᵀ W Δœ
/// ```
///
/// so *any* `u` with a negative component along `d` reduces `V`. This takes `u = −c d` clipped to
/// the available acceleration, giving `V̇ = −c‖d‖²` — non-positive for every element error and every
/// orbit, not only small ones, and zero only where `d` vanishes.
///
/// # Elements, in the DSST's order, naming and units
///
/// `[a, k, h, q, p, λ]` with `k = eₓ`, `h = e_y` the eccentricity-vector components, `q = hₓ`,
/// `p = h_y` the inclination-vector components, and `λ` the mean longitude, with `a` in metres to
/// match [`spacelab_propagation::dsst`]. Equinoctial because it
/// is non-singular at zero eccentricity and zero inclination, which is where station-keeping
/// problems live and where classical elements stop being coordinates at all.
///
/// # Weights carry the units
///
/// `a` is a length, `k h q p` are dimensionless and `λ` is an angle, so `Δœ` is not a vector in any
/// single unit and `W` is what makes the sum meaningful. Bryson's rule — weight each element by the
/// reciprocal square of the error the mission is willing to tolerate in it — turns `V` into a
/// dimensionless count of tolerances and is the intended way to set them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElementFeedback {
    weights: [f64; 6],
    proportional: f64,
    max_acceleration: f64,
}

impl ElementFeedback {
    /// `None` unless every weight is finite and non-negative with at least one strictly positive,
    /// and both the proportional gain and the acceleration limit are finite and strictly positive.
    ///
    /// A zero weight is admitted: it is how an element is declared uncontrolled, and it is the usual
    /// case — a ground-track hold cares about `a` and `λ` and not about the eccentricity vector.
    #[must_use]
    pub fn new(weights: [f64; 6], proportional: f64, max_acceleration: f64) -> Option<Self> {
        let usable = |x: f64| x.is_finite() && x > 0.0;
        let ok = weights.iter().all(|w| w.is_finite() && *w >= 0.0)
            && weights.iter().any(|w| *w > 0.0)
            && usable(proportional)
            && usable(max_acceleration);
        ok.then_some(Self {
            weights,
            proportional,
            max_acceleration,
        })
    }

    /// `V = ½ Δœᵀ W Δœ`.
    #[must_use]
    pub fn lyapunov(&self, error: &[f64; 6]) -> f64 {
        let mut acc = 0.0;
        for (w, e) in self.weights.iter().zip(error) {
            acc += w * e * e;
        }
        0.5 * acc
    }

    /// The commanded acceleration in the orbit (inertial) frame, m/s² — the frame every DSST force
    /// returns its acceleration in, so this can be summed with them directly.
    ///
    /// `error` is `œ − œ_target`, in the DSST's `[a, k, h, q, p, λ]` order. The mean-longitude
    /// component is wrapped to `(−π, π]` before use: an unwrapped `λ` error of `2π − ε` commands
    /// nearly a full revolution of drift to reach an orbit that is `ε` away, which is the same
    /// unwinding failure a quaternion law has and is fixed the same way.
    #[must_use]
    pub fn acceleration(
        &self,
        aux: &DsstAuxiliary,
        true_longitude: f64,
        error: &[f64; 6],
    ) -> [f64; 3] {
        let Some(e) = usable_error(error) else {
            return [0.0; 3];
        };

        let variational = variational_matrix(aux, true_longitude);
        // d = Bᵀ W Δœ.
        let mut d = [0.0; 3];
        for (i, di) in d.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..6 {
                acc += variational[j][i] * self.weights[j] * e[j];
            }
            *di = acc;
        }

        // Below the smallest normal, the reciprocal is either infinite or has lost every bit of
        // its mantissa, so the direction it would produce is noise rather than a small command.
        let magnitude = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if !magnitude.is_finite() || magnitude < f64::MIN_POSITIVE {
            return [0.0; 3];
        }
        // Saturation scales d without rotating it, so the sign of V̇ = −c‖d‖² survives it.
        let commanded = self.proportional * magnitude;
        let scale = -if commanded > self.max_acceleration {
            self.max_acceleration / magnitude
        } else {
            self.proportional
        };
        [scale * d[0], scale * d[1], scale * d[2]]
    }

    /// `V̇` along the closed loop, which this law makes non-positive by construction.
    ///
    /// Computed as `dᵀu` from the same `B(œ)` the control used, so a test can measure the decrease
    /// rather than take the derivation's word for it.
    #[must_use]
    pub fn decrease_rate(&self, aux: &DsstAuxiliary, true_longitude: f64, error: &[f64; 6]) -> f64 {
        let Some(e) = usable_error(error) else {
            return 0.0;
        };
        let u = self.acceleration(aux, true_longitude, error);
        let variational = variational_matrix(aux, true_longitude);
        let mut rate = 0.0;
        for j in 0..6 {
            let mut element_rate = 0.0;
            for i in 0..3 {
                element_rate += variational[j][i] * u[i];
            }
            rate += self.weights[j] * e[j] * element_rate;
        }
        rate
    }
}

/// The element error the law will act on, with the mean longitude wrapped — or `None` if it is not
/// an error at all.
///
/// One decision for both `acceleration` and `decrease_rate`, because they disagreed when it lived in
/// only one of them. `acceleration` refused a non-finite error at its `‖d‖` guard and returned no
/// command; `decrease_rate` recomputed `dᵀu` from that zero command and a still-NaN `e`, and
/// `NaN · 0` is `NaN`, so the law reported an undefined Lyapunov rate for an input it had already
/// declined to act on. An estimator that diverges is the way that input arrives.
fn usable_error(error: &[f64; 6]) -> Option<[f64; 6]> {
    if !error.iter().all(|e| e.is_finite()) {
        return None;
    }
    let mut e = *error;
    e[5] = wrap_to_pi(e[5]);
    Some(e)
}

/// Wrap an angle to `(−π, π]`.
fn wrap_to_pi(angle: f64) -> f64 {
    let tau = core::f64::consts::TAU;
    let wrapped = angle - tau * libm::floor(angle / tau + 0.5);
    // `floor` puts the boundary at −π; the interval is closed at +π by convention, and the two
    // differ by a full turn, so the correction is exact.
    if wrapped <= -core::f64::consts::PI {
        wrapped + tau
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    // Reference values are Maxima's 40-digit bigfloats truncated to what `f64` can hold, carried at
    // the CAS's precision so that regenerating one is a copy rather than a rounding decision.
    #![allow(clippy::excessive_precision, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    /// `derivations/lqr_riccati.mac`: the closed-loop characteristic polynomial of the
    /// Clohessy–Wiltshire LQR at `a = 6878.137 km`, `Q = I₆`, `R = 10⁶ I₃`, coefficients of
    /// `λ⁰ … λ⁵`. Derived from the Hamiltonian's spectrum and never from `P`, and licensed by a
    /// spectral-factorisation residual of `8.85e-43`.
    const CW_CLOSED_LOOP: [f64; 6] = [
        1.000_007_502_734_555_784_921_474_484_580_675_525_588e-9,
        1.343_078_007_866_673_089_694_552_053_617_477_373_173e-7,
        9.010_375_244_710_102_790_098_023_183_288_469_246_899e-6,
        3.581_248_034_029_908_353_996_214_623_509_071_817_562e-4,
        9.005_460_356_628_364_205_561_602_829_029_956_212_357e-3,
        1.341_976_931_056_134_221_576_552_349_332_354_188_333e-1,
    ];

    /// The same run's exact cross-track gain, from the closed-form `2 × 2` solution.
    const CW_CROSS_TRACK_GAIN: [f64; 2] = [
        9.987_757_806_778_943_818_874_793_953_051_262_467_004e-4,
        4.470_516_258_057_662_811_629_826_635_814_179_390_474e-2,
    ];

    const CW_MEAN_MOTION: f64 = 1.106_783_446_334_940_580_568_226_918_118_439_966_965e-3;
    const CW_CONTROL_WEIGHT: f64 = 1.0e6;

    fn diagonal<const N: usize>(d: [f64; N]) -> [[f64; N]; N] {
        let mut m = [[0.0; N]; N];
        for (i, di) in d.into_iter().enumerate() {
            m[i][i] = di;
        }
        m
    }

    /// What the norm scaling in [`matrix_sign`] does to the step count, which is the only thing it
    /// can do that an assertion on the answer would see: switched on or off, the iteration lands on
    /// the same `sign(H)`, so the count is what [`matrix_sign`] returns it for.
    ///
    /// The double-integrator bound is the discriminating one. Scaled it takes 3 steps; with the
    /// predicate inverted so the scaling is off where it matters, 5, and with it switched on only
    /// at one exact value, 6. The Clohessy–Wiltshire bound is a ceiling and not a discriminator —
    /// there the scaling costs steps rather than saving them (15 against 11), which is the finding
    /// [`matrix_sign`]'s doc comment records against the textbook rationale.
    #[test]
    fn the_norm_scaling_costs_the_iterations_the_measurement_says_it_does() {
        let (a, b) = clohessy_wiltshire_plant(CW_MEAN_MOTION).expect("positive mean motion");
        let (cw, ..) = hamiltonian::<6, 3, 12>(
            &a,
            &b,
            &diagonal([1.0; 6]),
            &diagonal([CW_CONTROL_WEIGHT; 3]),
        )
        .expect("invertible control weight");
        let (_, cw_steps) = matrix_sign::<12>(&cw).expect("no imaginary-axis eigenvalue");

        let (di, ..) = hamiltonian::<2, 1, 4>(
            &[[0.0, 1.0], [0.0, 0.0]],
            &[[0.0], [1.0]],
            &diagonal([1.0, 1.0]),
            &[[1.0]],
        )
        .expect("invertible control weight");
        let (_, di_steps) = matrix_sign::<4>(&di).expect("no imaginary-axis eigenvalue");

        println!("sign iterations: CW 12x12 {cw_steps}, double integrator 4x4 {di_steps}");
        // Bounded on both sides: the upper bound is the cost claim, the lower one says the count is
        // the number of iterations run and not the index of the last one.
        assert!(
            (15..=16).contains(&cw_steps),
            "CW sign iteration reported {cw_steps} steps"
        );
        assert!(
            (3..=4).contains(&di_steps),
            "double-integrator sign iteration reported {di_steps} steps"
        );
    }

    /// The double integrator, whose optimal solution Maxima derives in closed form:
    /// `P = [[√3, 1], [1, √3]]`, `K = [1, √3]`, closed loop `λ² + √3λ + 1`.
    #[test]
    fn the_double_integrator_matches_its_closed_form() {
        let a = [[0.0, 1.0], [0.0, 0.0]];
        let b = [[0.0], [1.0]];
        let solution =
            lqr::<2, 1, 4>(&a, &b, &diagonal([1.0, 1.0]), &[[1.0]]).expect("well-posed problem");

        let root3 = 3.0_f64.sqrt();
        let p = solution.riccati();
        println!(
            "double integrator: P = {p:?}, K = {:?}, residual {:.3e}",
            solution.gain(),
            solution.residual()
        );
        for (got, want) in [
            (p[0][0], root3),
            (p[0][1], 1.0),
            (p[1][0], 1.0),
            (p[1][1], root3),
        ] {
            assert!((got - want).abs() < 1e-13, "P entry {got} against {want}");
        }
        assert!((solution.gain()[0][0] - 1.0).abs() < 1e-13);
        assert!((solution.gain()[0][1] - root3).abs() < 1e-13);

        let poly = characteristic_polynomial(&solution.closed_loop(&a, &b));
        assert!((poly[0] - 1.0).abs() < 1e-13, "constant term {}", poly[0]);
        assert!((poly[1] - root3).abs() < 1e-13, "linear term {}", poly[1]);
    }

    /// The scalar case, where the Riccati equation is a quadratic with a closed-form positive root.
    /// Both a stable (`a < 0`) and a marginally-stable (`a = 0`) plant, because they exercise
    /// different signs in the same formula.
    #[test]
    fn the_scalar_case_matches_the_quadratic_root() {
        for (a, expected) in [(-1.0, core::f64::consts::SQRT_2 - 1.0), (0.0, 1.0)] {
            let solution =
                lqr::<1, 1, 2>(&[[a]], &[[1.0]], &[[1.0]], &[[1.0]]).expect("well-posed problem");
            let p = solution.riccati()[0][0];
            println!("scalar a = {a}: P = {p:.17e} against {expected:.17e}");
            assert!((p - expected).abs() < 1e-14, "P = {p} against {expected}");
            assert!((solution.gain()[0][0] - expected).abs() < 1e-14);
        }
    }

    /// The full `6 × 6`, against the closed-loop polynomial the CAS derived from the Hamiltonian
    /// alone. This is the check that cannot be passed by a `P` that solves the wrong branch: the
    /// coefficients come from `H`'s stable spectrum and never touch `P`'s arithmetic.
    #[test]
    fn the_clohessy_wiltshire_closed_loop_has_the_spectrum_the_hamiltonian_says() {
        let (a, b) = clohessy_wiltshire_plant(CW_MEAN_MOTION).expect("positive mean motion");
        let solution = lqr::<6, 3, 12>(
            &a,
            &b,
            &diagonal([1.0; 6]),
            &diagonal([CW_CONTROL_WEIGHT; 3]),
        )
        .expect("well-posed problem");
        println!("CW LQR: Riccati residual {:.3e}", solution.residual());

        let poly = characteristic_polynomial(&solution.closed_loop(&a, &b));
        let mut worst = 0.0_f64;
        for (j, (got, want)) in poly.iter().zip(&CW_CLOSED_LOOP).enumerate() {
            let deviation = (got - want).abs() / want.abs();
            println!("  c{j}: {got:.17e} against {want:.17e}   rel {deviation:.3e}");
            worst = worst.max(deviation);
        }
        assert!(
            worst < 1e-11,
            "closed-loop polynomial deviates from the CAS by {worst:.3e}"
        );
    }

    /// The cross-track axis decouples from the in-plane pair entirely, so the `6 × 6` answer must
    /// contain the exact `2 × 2` one. A gain that is right on average and wrong in the coupling
    /// would not reproduce it.
    #[test]
    fn the_cross_track_block_is_the_exact_two_by_two_solution() {
        let (a, b) = clohessy_wiltshire_plant(CW_MEAN_MOTION).expect("positive mean motion");
        let solution = lqr::<6, 3, 12>(
            &a,
            &b,
            &diagonal([1.0; 6]),
            &diagonal([CW_CONTROL_WEIGHT; 3]),
        )
        .expect("well-posed problem");

        let k = solution.gain();
        for (index, (got, want)) in [
            (k[2][2], CW_CROSS_TRACK_GAIN[0]),
            (k[2][5], CW_CROSS_TRACK_GAIN[1]),
        ]
        .into_iter()
        .enumerate()
        {
            let deviation = (got - want).abs() / want;
            println!(
                "  cross-track gain {index}: {got:.17e} against {want:.17e}  rel {deviation:.3e}"
            );
            assert!(deviation < 1e-12, "cross-track gain {got} against {want}");
        }
        // And the cross-track control must not react to the in-plane state at all.
        for j in [0, 1, 3, 4] {
            assert!(
                k[2][j].abs() < 1e-14,
                "cross-track gain reads state {j}: {}",
                k[2][j]
            );
        }
    }

    /// Integrate `ẋ = (A − BK)x` and accumulate `∫(xᵀQx + uᵀRu) dt`, which is what the LQR claims to
    /// minimise. RK4 at a step far below the closed-loop time constant, with the running cost taken
    /// by Simpson over the midpoint state RK4 already produced. Every gain compared goes through
    /// this identical integration, so the comparison between them is exact even where the absolute
    /// integral carries truncation.
    fn closed_loop_cost(
        a: &[[f64; 6]; 6],
        b: &[[f64; 3]; 6],
        q: &[[f64; 6]; 6],
        r: &[[f64; 3]; 3],
        gain: &[[f64; 6]; 3],
        x0: &[f64; 6],
    ) -> f64 {
        let mut cl = *a;
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..3 {
                    cl[i][j] -= b[i][k] * gain[k][j];
                }
            }
        }
        let running = |x: &[f64; 6]| {
            let mut c = 0.0;
            for i in 0..6 {
                c += q[i][i] * x[i] * x[i];
            }
            for k in 0..3 {
                let mut u = 0.0;
                for j in 0..6 {
                    u -= gain[k][j] * x[j];
                }
                c += r[k][k] * u * u;
            }
            c
        };
        let deriv = |x: &[f64; 6]| {
            let mut d = [0.0; 6];
            for i in 0..6 {
                for j in 0..6 {
                    d[i] += cl[i][j] * x[j];
                }
            }
            d
        };
        // 5000 s is 112 closed-loop time constants, so the horizon truncation is far below the RK4
        // truncation and the integral has converged.
        let (dt, steps) = (1.0, 5_000);
        let mut x = *x0;
        let mut cost = 0.0;
        for _ in 0..steps {
            let k1 = deriv(&x);
            let mut x2 = x;
            for i in 0..6 {
                x2[i] += 0.5 * dt * k1[i];
            }
            let k2 = deriv(&x2);
            let mut x3 = x;
            for i in 0..6 {
                x3[i] += 0.5 * dt * k2[i];
            }
            let k3 = deriv(&x3);
            let mut x4 = x;
            for i in 0..6 {
                x4[i] += dt * k3[i];
            }
            let k4 = deriv(&x4);
            let start = running(&x);
            for i in 0..6 {
                x[i] += dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
            }
            cost += dt * (start + 4.0 * running(&x2) + running(&x)) / 6.0;
        }
        cost
    }

    /// The gain is optimal, not merely stabilising: the cost it achieves from a given initial state
    /// must be no worse than the cost of any other stabilising gain. Measured by integrating the
    /// closed loop and accumulating `xᵀQx + uᵀRu`, against `x₀ᵀPx₀` which is what `P` claims.
    #[test]
    fn the_cost_to_go_is_what_the_riccati_matrix_claims_and_no_detuned_gain_beats_it() {
        let (a, b) = clohessy_wiltshire_plant(CW_MEAN_MOTION).expect("positive mean motion");
        let q = diagonal([1.0; 6]);
        let r = diagonal([CW_CONTROL_WEIGHT; 3]);
        let solution = lqr::<6, 3, 12>(&a, &b, &q, &r).expect("well-posed problem");

        let x0 = [1.0, -2.0, 0.5, 1.0e-3, 2.0e-3, -1.0e-3];
        let predicted = {
            let p = solution.riccati();
            let mut acc = 0.0;
            for i in 0..6 {
                for j in 0..6 {
                    acc += x0[i] * p[i][j] * x0[j];
                }
            }
            acc
        };

        let cost_of = |gain: &[[f64; 6]; 3]| closed_loop_cost(&a, &b, &q, &r, gain, &x0);

        let optimal = cost_of(solution.gain());
        println!(
            "cost: integrated {optimal:.9e}, x0' P x0 = {predicted:.9e}, rel {:.3e}",
            (optimal - predicted).abs() / predicted
        );
        assert!(
            (optimal - predicted).abs() / predicted < 1e-6,
            "integrated cost {optimal} against the Riccati prediction {predicted}"
        );

        for factor in [0.5, 0.8, 1.25, 2.0] {
            let mut detuned = *solution.gain();
            for row in &mut detuned {
                for g in row.iter_mut() {
                    *g *= factor;
                }
            }
            let cost = cost_of(&detuned);
            println!(
                "  gain x{factor}: cost {cost:.9e}, {:.4}x optimal",
                cost / optimal
            );
            assert!(
                cost > optimal,
                "a gain scaled by {factor} achieved {cost}, below the optimum {optimal}"
            );
        }
    }

    /// Symmetry is a statement about the off-diagonal pairs, so the diagonal is not compared at
    /// all. That is invisible on ordinary input — `m[i][i] != m[i][i]` is false — and visible on a
    /// NaN, which is not equal to itself. A loop that included the diagonal would call this matrix
    /// asymmetric and `lqr` would return the wrong error for it.
    #[test]
    fn a_matrix_is_symmetric_about_its_off_diagonal_pairs_and_not_its_diagonal() {
        assert!(is_symmetric(&[[f64::NAN]]));
        assert!(is_symmetric(&[[f64::NAN, 2.0], [2.0, 1.0]]));
        assert!(!is_symmetric(&[[1.0, 2.0], [2.5, 1.0]]));
        assert!(!is_symmetric(&[[1.0, f64::NAN], [f64::NAN, 1.0]]));
    }

    #[test]
    fn an_ill_posed_problem_is_refused_rather_than_answered() {
        let a = [[0.0, 1.0], [0.0, 0.0]];
        let b = [[0.0], [1.0]];
        let q = diagonal([1.0, 1.0]);
        assert_eq!(
            lqr::<2, 1, 4>(&a, &b, &q, &[[0.0]]),
            Err(LqrError::SingularControlWeight)
        );
        assert_eq!(
            lqr::<2, 1, 4>(&a, &b, &[[1.0, 0.5], [0.4, 1.0]], &[[1.0]]),
            Err(LqrError::Asymmetric)
        );
        assert_eq!(
            lqr::<2, 1, 4>(&[[f64::NAN, 0.0], [0.0, 0.0]], &b, &q, &[[1.0]]),
            Err(LqrError::NonFinite)
        );
        // An uncontrollable unstable mode: B reaches only the second state, and the first is a pure
        // unstable integrator nothing can touch. Its cost is infinite, so no finite P exists — and
        // the stable subspace, having no component along that state, is not a graph over it.
        assert_eq!(
            lqr::<2, 1, 4>(&[[1.0, 0.0], [0.0, -1.0]], &[[0.0], [1.0]], &q, &[[1.0]]),
            Err(LqrError::NoRiccatiSolution)
        );
        // A plant with almost no control authority: B reaches the second state through a 1e-8
        // actuator and the control weight is 1e12 on top of that. The sign iteration converges —
        // it returns a finite P in 51 steps — and P is nowhere near solving the Riccati equation.
        // That is the only way to reach `RiccatiResidual`, and it is the case that separates the
        // two conjuncts of its guard: the residual is finite, so a guard that accepted on either
        // condition alone would hand this P back as an answer.
        let weak = lqr::<2, 1, 4>(&a, &[[0.0], [1e-8]], &q, &[[1.0e12]]);
        assert_eq!(weak, Err(LqrError::RiccatiResidual));

        assert!(clohessy_wiltshire_plant(0.0).is_none());
        assert!(clohessy_wiltshire_plant(-1e-3).is_none());
        assert!(clohessy_wiltshire_plant(f64::INFINITY).is_none());
    }

    #[test]
    fn the_characteristic_polynomial_is_the_one_the_definition_gives() {
        // A companion matrix's characteristic polynomial is its last column, by construction.
        let m = [[0.0, 0.0, -6.0], [1.0, 0.0, -11.0], [0.0, 1.0, -6.0]];
        let poly = characteristic_polynomial(&m);
        for (got, want) in poly.iter().zip(&[6.0, 11.0, 6.0]) {
            assert!((got - want).abs() < 1e-12, "{got} against {want}");
        }
        // (λ−1)(λ−2)(λ−3) = λ³ − 6λ² + 11λ − 6, so a diagonal matrix must give the same up to sign.
        let d = diagonal([1.0, 2.0, 3.0]);
        let poly = characteristic_polynomial(&d);
        for (got, want) in poly.iter().zip(&[-6.0, 11.0, -6.0]) {
            assert!((got - want).abs() < 1e-12, "{got} against {want}");
        }
    }

    // ---------------------------------------------------------------- element feedback

    /// Earth's gravitational parameter in SI, m³/s² — the units `KeplerianElements` is written in,
    /// so the whole element-feedback block works in metres and the oracle needs no rescaling.
    const MU: f64 = 3.986_004_418e14;

    /// Keplerian `[a, e, i, Ω, ω, ν]` (km, rad) to the DSST's equinoctial `[a, k, h, q, p, λ]` and
    /// the true longitude, written out here rather than borrowed so the oracle below shares no code
    /// with what it checks. The convention is read off `DsstAuxiliary` itself: `B = √(1 − k² − h²)`
    /// is `√(1 − e²)`, and `γ = (1 − p² − q²)/(1 + p² + q²)` is `cos i` exactly when `p² + q²` is
    /// `tan²(i/2)`.
    fn equinoctial(kepler: [f64; 6]) -> ([f64; 6], f64) {
        let [a, e, inc, raan, argp, nu] = kepler;
        let perigee_longitude = argp + raan;
        let tan_half = libm::tan(0.5 * inc);
        let eccentric = 2.0 * libm::atan(libm::sqrt((1.0 - e) / (1.0 + e)) * libm::tan(0.5 * nu));
        let mean = eccentric - e * libm::sin(eccentric);
        (
            [
                a,
                e * libm::cos(perigee_longitude),
                e * libm::sin(perigee_longitude),
                tan_half * libm::cos(raan),
                tan_half * libm::sin(raan),
                mean + perigee_longitude,
            ],
            nu + perigee_longitude,
        )
    }

    /// The same element set as an inertial state, by the classical rotation sequence.
    fn cartesian(kepler: [f64; 6]) -> ([f64; 3], [f64; 3]) {
        let [a, e, inc, raan, argp, nu] = kepler;
        let p = a * (1.0 - e * e);
        let r = p / (1.0 + e * libm::cos(nu));
        let (sn, cn) = libm::sincos(nu);
        let perifocal_r = [r * cn, r * sn, 0.0];
        let speed = libm::sqrt(MU / p);
        let perifocal_v = [-speed * sn, speed * (e + cn), 0.0];
        let (so, co) = libm::sincos(argp);
        let (sr, cr) = libm::sincos(raan);
        let (si, ci) = libm::sincos(inc);
        // R3(-Ω) R1(-i) R3(-ω), column by column.
        let rot = [
            [co * cr - so * ci * sr, -so * cr - co * ci * sr, si * sr],
            [co * sr + so * ci * cr, -so * sr + co * ci * cr, -si * cr],
            [so * si, co * si, ci],
        ];
        let apply = |x: [f64; 3]| {
            let mut out = [0.0; 3];
            for i in 0..3 {
                for j in 0..3 {
                    out[i] += rot[i][j] * x[j];
                }
            }
            out
        };
        (apply(perifocal_r), apply(perifocal_v))
    }

    /// Inertial state back to the equinoctial set, through `KeplerianElements::from_state` — the
    /// crate's own verified conversion, and a completely different code path from the DSST's
    /// `δaᵢ/δv` assembly.
    fn equinoctial_from_state(r: [f64; 3], v: [f64; 3]) -> [f64; 6] {
        use spacelab_core::Vec3;
        use spacelab_elements::KeplerianElements;
        let els = KeplerianElements::from_state(
            Vec3::new(r[0], r[1], r[2]),
            Vec3::new(v[0], v[1], v[2]),
            MU,
        );
        equinoctial([
            els.semi_major_axis,
            els.eccentricity,
            els.inclination.radians(),
            els.raan.radians(),
            els.arg_periapsis.radians(),
            els.true_anomaly.radians(),
        ])
        .0
    }

    /// A moderately eccentric, moderately inclined orbit — every element non-degenerate, so a sign
    /// error in any column of the variational matrix has somewhere to show.
    const TEST_ORBIT: [f64; 6] = [7.200e6, 0.02, 0.9, 1.3, 0.7, 2.1];

    fn test_auxiliary() -> (DsstAuxiliary, f64) {
        let (oe, true_longitude) = equinoctial(TEST_ORBIT);
        (
            DsstAuxiliary::new(oe[0], oe[1], oe[2], oe[3], oe[4], oe[5], MU),
            true_longitude,
        )
    }

    /// The Gauss variational matrix against a central difference of the real element response to a
    /// real velocity increment.
    ///
    /// `B(œ) = ∂œ/∂v` is by definition what an impulse does to the elements, so applying one and
    /// differencing is the definition rather than a rearrangement of the DSST's algebra. The path
    /// runs state → `KeplerianElements::from_state` → equinoctial, and touches none of the
    /// `linear_combination`/`dot_ext` machinery the matrix is built from.
    #[test]
    fn the_variational_matrix_is_what_an_impulse_actually_does_to_the_elements() {
        use spacelab_propagation::dsst::gaussian::variational_matrix;

        let (aux, true_longitude) = test_auxiliary();
        let analytic = variational_matrix(&aux, true_longitude);
        let (r, v) = cartesian(TEST_ORBIT);
        // The inertial axes, because that is the frame the matrix's acceleration argument is in.
        // `f/g/w` are a basis *expressed in* that frame, not a frame to resolve into — and pointing
        // the difference along them instead is exactly the mistake the sparsity pattern catches:
        // only an out-of-plane impulse can turn the orbit plane, so differencing along `w` puts the
        // whole of `∂(q, p)/∂v` in one column where the inertial matrix spreads it over three.
        let basis = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // 0.1 m/s against a 7.4 km/s orbital speed: the central difference's O(ε²) truncation and
        // the O(ulp/ε) rounding are both around 1e-11 relative there.
        let step = 0.1;
        let mut worst = 0.0_f64;
        for (axis, direction) in basis.iter().enumerate() {
            let shifted = |sign: f64| {
                let mut moved = v;
                for i in 0..3 {
                    moved[i] += sign * step * direction[i];
                }
                equinoctial_from_state(r, moved)
            };
            let (plus, minus) = (shifted(1.0), shifted(-1.0));
            for element in 0..6 {
                let numeric = (plus[element] - minus[element]) / (2.0 * step);
                let want = analytic[element][axis];
                let deviation = (numeric - want).abs() / want.abs().max(1.0e-12);
                println!(
                    "  B[{element}][{axis}]: analytic {want:+.12e}  finite difference \
                     {numeric:+.12e}  rel {deviation:.3e}"
                );
                worst = worst.max(deviation);
            }
        }
        assert!(
            worst < 1e-6,
            "the variational matrix deviates from a measured impulse by {worst:.3e}"
        );
    }

    /// The Lyapunov function decreases along the law's own control, for every element error and
    /// every point on the orbit — the claim that makes this a law rather than a heuristic. Measured
    /// from `V̇ = Δœᵀ W B u`, not argued from the derivation.
    #[test]
    fn the_element_feedback_never_increases_its_lyapunov_function() {
        let (aux, _) = test_auxiliary();
        // Bryson scaling for a 1 km semi-major-axis tolerance, 1e-4 on the eccentricity and
        // inclination vectors and 0.01 rad on the mean longitude.
        let weights = [1.0e-6, 1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e4];
        let law = ElementFeedback::new(weights, 1.0e-3, 1.0e-5).expect("valid gains");

        let errors = [
            [1.0e4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0e-3, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0e-3, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0e-4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -2.0e-4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-2],
            [-2.5e4, 5.0e-4, -3.0e-4, 1.0e-4, 2.0e-4, -0.05],
        ];
        let mut worst_rate = f64::NEG_INFINITY;
        let mut best_rate = f64::INFINITY;
        let mut positions = 0;
        for error in &errors {
            for step in 0..24 {
                let longitude = f64::from(step) * core::f64::consts::TAU / 24.0;
                let rate = law.decrease_rate(&aux, longitude, error);
                assert!(
                    rate <= 0.0,
                    "V rose at longitude {longitude}: dV/dt = {rate:e} for error {error:?}"
                );
                worst_rate = worst_rate.max(rate);
                best_rate = best_rate.min(rate);
                positions += 1;
            }
        }
        println!(
            "dV/dt <= 0 at all {positions} (error, longitude) pairs; \
             most negative {best_rate:.3e}, least negative {worst_rate:.3e}"
        );
        // A law that commanded nothing would also satisfy `dV/dt <= 0`, so the decrease has to be
        // strict somewhere. It is not strict everywhere: an error whose steepest-descent direction
        // `d` vanishes at a particular longitude genuinely has no useful thrust to apply there.
        assert!(best_rate < 0.0, "no case produced any decrease at all");

        // At the target the law commands nothing, and V is exactly zero.
        let none = law.acceleration(&aux, 0.4, &[0.0; 6]);
        assert_eq!(none, [0.0; 3]);
        assert_eq!(law.lyapunov(&[0.0; 6]), 0.0);
    }

    /// Saturation clips the magnitude and leaves the direction alone, which is what preserves the
    /// sign of `V̇` through the limiter.
    #[test]
    fn saturation_scales_the_command_without_rotating_it() {
        let (aux, longitude) = test_auxiliary();
        let weights = [1.0e-6, 1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e4];
        let error = [-5.0e5, 5.0e-3, -3.0e-3, 1.0e-3, 2.0e-3, -0.5];

        let limit = 1.0e-6;
        let saturated = ElementFeedback::new(weights, 1.0e-3, limit).expect("valid");
        let generous = ElementFeedback::new(weights, 1.0e-3, 1.0e3).expect("valid");
        let a = saturated.acceleration(&aux, longitude, &error);
        let b = generous.acceleration(&aux, longitude, &error);

        let norm = |x: &[f64; 3]| libm::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        let (na, nb) = (norm(&a), norm(&b));
        println!("saturated |u| = {na:.3e} against limit {limit:.3e}, unsaturated {nb:.3e}");
        assert!(
            na <= limit * (1.0 + 1e-12),
            "the limit was exceeded: {na:e}"
        );
        assert!(
            nb > limit,
            "the unsaturated command should have been larger"
        );
        for i in 0..3 {
            let deviation = (a[i] / na - b[i] / nb).abs();
            assert!(
                deviation < 1e-12,
                "saturation rotated axis {i} by {deviation:e}"
            );
        }
        assert!(saturated.decrease_rate(&aux, longitude, &error) < 0.0);
    }

    /// A mean-longitude error of nearly a full turn is a small error the other way round, and the
    /// law has to see it that way — the same unwinding failure a quaternion law has.
    #[test]
    fn a_mean_longitude_error_takes_the_short_way_round() {
        let (aux, longitude) = test_auxiliary();
        let law =
            ElementFeedback::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1.0e-3, 1.0e3).expect("valid");
        let small = law.acceleration(&aux, longitude, &[0.0, 0.0, 0.0, 0.0, 0.0, -0.01]);
        let wrapped = law.acceleration(
            &aux,
            longitude,
            &[0.0, 0.0, 0.0, 0.0, 0.0, core::f64::consts::TAU - 0.01],
        );
        for i in 0..3 {
            assert!(
                (small[i] - wrapped[i]).abs() < 1e-18,
                "axis {i}: {} against {}",
                small[i],
                wrapped[i]
            );
        }
        // And the unwrapped reading would have gone the other way, so the wrap is doing work.
        let long_way = law.acceleration(&aux, longitude, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.01]);
        assert!(long_way[0] * small[0] < 0.0 || long_way[1] * small[1] < 0.0);
    }

    #[test]
    fn element_feedback_gains_that_cannot_control_anything_are_refused() {
        let good = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert!(ElementFeedback::new([0.0; 6], 1.0, 1.0).is_none());
        assert!(ElementFeedback::new([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1.0, 1.0).is_none());
        assert!(ElementFeedback::new([f64::NAN, 0.0, 0.0, 0.0, 0.0, 1.0], 1.0, 1.0).is_none());
        assert!(ElementFeedback::new(good, 0.0, 1.0).is_none());
        assert!(ElementFeedback::new(good, 1.0, 0.0).is_none());
        assert!(ElementFeedback::new(good, 1.0, f64::INFINITY).is_none());
        assert!(ElementFeedback::new(good, 1.0, 1.0).is_some());
    }

    #[test]
    fn an_angle_wraps_to_the_short_way_round() {
        let pi = core::f64::consts::PI;
        let tau = core::f64::consts::TAU;
        for (input, expected) in [
            (0.0, 0.0),
            (0.5, 0.5),
            (-0.5, -0.5),
            (tau - 0.25, -0.25),
            (-tau + 0.25, 0.25),
            (7.0 * tau + 1.0, 1.0),
            (pi, pi),
        ] {
            let got = wrap_to_pi(input);
            assert!(
                (got - expected).abs() < 1e-12,
                "wrap({input}) = {got}, expected {expected}"
            );
        }
    }

    /// `riccati_residual` on a case small enough to evaluate by hand, because everywhere else it is
    /// only ever consulted as an accept/reject predicate on a converged solve — where it reads
    /// `1e-16` and nothing distinguishes that from a residual that is wrong by construction.
    ///
    /// `N = 1`, `A = 3`, `Q = 5`, `G = 7`, `P = 2`. Then `AᵀP = PA = 6`, `PGP = 28`, so the equation
    /// evaluates to `6 + 6 − 28 + 5 = −11` and the normalising sum of squares is
    /// `36 + 36 + 784 + 25 = 881`. The residual is `√(121/881)`. None of the four terms is `0`, `1`
    /// or `2`, so no operator swap inside the sum is masked by an arithmetic coincidence.
    #[test]
    fn the_riccati_residual_is_the_normalised_norm_of_the_equation_it_names() {
        let got = riccati_residual(&[[3.0]], &[[5.0]], &[[7.0]], &[[2.0]]);
        let want = libm::sqrt(121.0 / 881.0);
        println!("riccati residual {got:.17e} against {want:.17e}");
        assert!((got - want).abs() < 1e-15, "{got} against {want}");

        // The exact scalar solution: `2ap − gp² + q = 0` at `a = 3, g = 7, q = 5`.
        let root = (3.0 + libm::sqrt(9.0 + 35.0)) / 7.0;
        let converged = riccati_residual(&[[3.0]], &[[5.0]], &[[7.0]], &[[root]]);
        println!("at the exact root the residual is {converged:.3e}");
        assert!(
            converged < 1e-15,
            "the exact solution is not a solution: {converged:e}"
        );

        // The all-zero problem is the one case where the normalisation has nothing to divide by.
        assert_eq!(
            riccati_residual(&[[0.0]], &[[0.0]], &[[0.0]], &[[0.0]]),
            0.0
        );
    }

    /// `control` is `u = −Kx` and `residual` is the number the solve was accepted at. Both are
    /// read off `Lqr` everywhere else through their effect on a trajectory, which is a place a
    /// sign, a factor or a stale field can hide.
    ///
    /// The state is `(2, 5)` against the double integrator's `K = [1, √3]`, so `u = −(2 + 5√3)`.
    /// Neither product equals its sum and neither equals its quotient, so `*`→`+` and `*`→`/`
    /// separate, and the single component is far from zero, so `−=`→`+=` and `−=`→`/=` do too.
    #[test]
    fn the_control_is_minus_the_gain_times_the_state_and_the_residual_is_the_accepted_one() {
        let a = [[0.0, 1.0], [0.0, 0.0]];
        let b = [[0.0], [1.0]];
        let solution =
            lqr::<2, 1, 4>(&a, &b, &diagonal([1.0, 1.0]), &[[1.0]]).expect("well-posed problem");

        let state = [2.0, 5.0];
        let u = solution.control(&state);
        let k = solution.gain();
        let want = -(k[0][0] * state[0] + k[0][1] * state[1]);
        println!("u = {:.17e} against −Kx = {want:.17e}", u[0]);
        assert!((u[0] - want).abs() < 1e-15, "{} against {want}", u[0]);
        // And against the closed form, so the test does not merely restate `gain()`.
        let closed = -(2.0 + 5.0 * 3.0_f64.sqrt());
        assert!((u[0] - closed).abs() < 1e-12, "{} against {closed}", u[0]);

        // The residual is the solve's own measured error: strictly positive, because a `6 × 6`
        // Newton iteration does not land on the exact `P`, and far below the tolerance it was
        // accepted at.
        let r = solution.residual();
        println!("residual {r:.3e} against tolerance {RESIDUAL_TOLERANCE:.0e}");
        assert!(
            r > 0.0,
            "a floating-point solve reported an exact residual: {r:e}"
        );
        assert!(r < RESIDUAL_TOLERANCE, "accepted at {r:e}");
    }

    /// `V = ½ Δœᵀ W Δœ` on a case whose value is written out, rather than only through the sign of
    /// its difference along a trajectory.
    ///
    /// `w = (2, 3, 0, 0, 0, 5)` against `e = (7, 11, 0, 0, 0, 13)` gives
    /// `½(2·49 + 3·121 + 5·169) = ½(98 + 363 + 845) = 653`. Every `w·e·e` differs from `w + e·e`
    /// and from `w·e + e`, so the two operator swaps in the accumulation separate.
    #[test]
    fn the_lyapunov_function_is_half_the_weighted_square_of_the_error() {
        let law = ElementFeedback::new([2.0, 3.0, 0.0, 0.0, 0.0, 5.0], 1.0e-3, 1.0).expect("valid");
        let got = law.lyapunov(&[7.0, 11.0, 0.0, 0.0, 0.0, 13.0]);
        println!("V = {got:.17e} against 653");
        assert!((got - 653.0).abs() < 1e-12, "{got} against 653");
        assert_eq!(law.lyapunov(&[0.0; 6]), 0.0);
    }

    /// A non-finite element error commands nothing, rather than commanding a non-finite thrust.
    ///
    /// The guard is `‖d‖` non-finite **or** below the smallest normal, and the two disjuncts are
    /// reached by different inputs: this is the first one. It matters operationally — an estimator
    /// that diverges hands the law a NaN, and a NaN acceleration would propagate into the state.
    #[test]
    fn a_non_finite_element_error_commands_nothing() {
        let (aux, longitude) = test_auxiliary();
        let law = ElementFeedback::new([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1.0e-3, 1.0).expect("valid");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let u = law.acceleration(&aux, longitude, &[bad, 0.0, 0.0, 0.0, 0.0, 0.0]);
            println!("error {bad} commands {u:?}");
            assert_eq!(u, [0.0; 3], "a {bad} error produced a command");
            assert_eq!(
                law.decrease_rate(&aux, longitude, &[bad, 0.0, 0.0, 0.0, 0.0, 0.0]),
                0.0
            );
        }
    }

    /// The second disjunct of the same guard: `‖d‖` reads exactly zero for a `d` that is not zero,
    /// because every square underflows before the sum is taken.
    ///
    /// `w = e = 1e-100` on the semi-major axis puts `d` at about `1e-197`, and `1e-197²` is below
    /// the smallest subnormal, so the sum of squares is `0` and its root is `0`. The command that
    /// would follow is `−p·d ≈ 1e-200` — finite, non-zero, and carrying none of the direction
    /// information it appears to, since the magnitude it was scaled against was lost. A square root
    /// cannot land strictly between zero and the smallest normal, so this is the *only* way the
    /// second disjunct is ever true.
    #[test]
    fn a_command_whose_magnitude_underflowed_is_refused_rather_than_scaled() {
        let (aux, longitude) = test_auxiliary();
        let law =
            ElementFeedback::new([1.0e-100, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0e-3, 1.0).expect("valid");
        let error = [1.0e-100, 0.0, 0.0, 0.0, 0.0, 0.0];
        let u = law.acceleration(&aux, longitude, &error);
        println!("underflowed |d|: command {u:?}");
        assert_eq!(u, [0.0; 3], "a command was issued from a magnitude of zero");

        // The same law one hundred decades up is not refused, so the test is about the underflow
        // and not about the weight being small.
        let scaled = law.acceleration(
            &aux,
            longitude,
            &[1.0e-100 * 1.0e100, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert!(
            scaled.iter().any(|x| *x != 0.0),
            "the representable case was refused too: {scaled:?}"
        );
    }

    /// Saturation clips to the limit *exactly*, and the branch is taken on `p‖d‖` and not on
    /// `p/‖d‖`.
    ///
    /// `saturation_scales_the_command_without_rotating_it` asserts the direction survives and the
    /// magnitude does not exceed the limit; neither pins `‖d‖` itself, so the sum under the square
    /// root can become a product and the clip still lands under the bound. Here the saturated
    /// command's magnitude is the limit to the ulp, which it can only be if `‖d‖` is the norm.
    #[test]
    fn the_saturated_command_is_exactly_the_limit_and_the_branch_reads_the_product() {
        let (aux, longitude) = test_auxiliary();
        let weights = [1.0e-6, 1.0e8, 1.0e8, 1.0e8, 1.0e8, 1.0e4];
        let error = [-5.0e5, 5.0e-3, -3.0e-3, 1.0e-3, 2.0e-3, -0.5];
        let limit = 1.0e-6;
        // The gain is small enough that `p‖d‖` and `p/‖d‖` land on opposite sides of the limit —
        // `9.8e-2` against `1.0e-7` — so the branch is decided by the product and cannot be read
        // as a quotient.
        let law = ElementFeedback::new(weights, 1.0e-4, limit).expect("valid");

        let u = law.acceleration(&aux, longitude, &error);
        let magnitude = libm::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        println!("saturated |u| = {magnitude:.17e} against the limit {limit:.17e}");
        assert!(
            (magnitude - limit).abs() <= 8.0 * f64::EPSILON * limit,
            "|u| = {magnitude:e} is not the limit {limit:e}"
        );

        // The same geometry below the limit is not clipped, so the branch is exercised from both
        // sides with `‖d‖` unchanged — the only thing that differs is which side of it `p‖d‖` is.
        let generous = ElementFeedback::new(weights, 1.0e-4, 1.0e3).expect("valid");
        let v = generous.acceleration(&aux, longitude, &error);
        let unsaturated = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        println!("unsaturated |u| = {unsaturated:.3e}");
        assert!(unsaturated > limit, "the case does not saturate at all");
        let d_magnitude = unsaturated / 1.0e-4;
        assert!(
            1.0e-4 / d_magnitude < limit,
            "the quotient reading of the branch is on the same side as the product"
        );
    }

    /// `V̇` is zero when there is no error, and negative when there is one — the second half is
    /// already asserted elsewhere, the first is what separates the rate from any constant.
    #[test]
    fn the_decrease_rate_is_zero_when_there_is_nothing_to_correct() {
        let (aux, longitude) = test_auxiliary();
        let law = ElementFeedback::new([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1.0e-3, 1.0).expect("valid");
        assert_eq!(law.decrease_rate(&aux, longitude, &[0.0; 6]), 0.0);
        assert!(law.decrease_rate(&aux, longitude, &[1.0e3, 0.0, 0.0, 0.0, 0.0, 0.0]) < 0.0);
    }
}
