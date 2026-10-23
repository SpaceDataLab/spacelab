//! Bracket-preserving root refinement — Brent's method, returning a bracket, not a point.
//!
//! # Provenance
//!
//! This is `zeroin`, as published in Forsythe, Malcolm & Moler, *Computer Methods for Mathematical
//! Computations* (1977), which descends from Dekker (1969) and Brent, *Algorithms for Minimization
//! without Derivatives* (1973), ch. 4. It is on Netlib. It is **not** taken from *Numerical
//! Recipes*, whose `zbrent` is one licensed expression of the same public algorithm; see
//! `spacelab/CONTRIBUTING.md`. The structure below is derived from the algorithm's description —
//! inverse quadratic interpolation, guarded by the acceptance test that keeps every step inside the
//! bracket — not transcribed from any one listing.
//!
//! # Why it returns a bracket
//!
//! A root finder that returns a single `f64` has thrown away what its caller needs. An event
//! detector that resumes exactly at the root re-detects the same event forever; it must resume on
//! the far side, and it must know the sign there. So [`brent`] returns a [`Bracket`] — two abscissae
//! straddling the root, narrower than the tolerance, with the sign at each — exactly as Orekit's
//! `EventState.findRoot` does ([notes/13-orekit.md](../notes/13-orekit.md) §3).
//!
//! # The guarantees
//!
//! * The bracket is a **precondition**, checked, not hoped for: `f(a)` and `f(b)` must differ in
//!   sign. `brent` returns [`RootError::NotBracketed`] otherwise rather than extrapolating, which is
//!   the failure mode `notes/11-gmat.md` §3 records in GMAT.
//! * Convergence is **guaranteed**: the interpolated step is accepted only when it lands inside the
//!   bracket and beats bisection, so the interval never stops shrinking and the worst case is
//!   bisection — `⌈log2((b−a)/tol)⌉` evaluations, bounded, no watchdog needed.
//! * The tolerance is on the **abscissa** (the units of `x` — seconds, for a time root), never on
//!   `f`. This is the dimension three other implementations get wrong
//!   ([notes/13-orekit.md](../notes/13-orekit.md) §3).

#![allow(clippy::doc_markdown)] // program names and math notation, not identifiers
#![allow(clippy::float_cmp)] // exact `== 0.0` is the exact-root test; `a == c` is a zeroin invariant

/// A root, returned as the narrow interval that brackets it.
///
/// `lo < hi`, and `f(lo)`, `f(hi)` have opposite signs (or one is exactly zero). The width is
/// `hi - lo ≤ tol + 4·ε·|x|` — the `zeroin` convergence test stops at `|xm| ≤ 2ε|b| + tol/2`, so the
/// floating-point floor `4·ε·|x|` adds to the requested `tol` rather than being absorbed by it. The
/// best point estimate is [`Bracket::estimate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bracket {
    /// Lower abscissa of the bracket.
    pub lo: f64,
    /// `f(lo)`.
    pub f_lo: f64,
    /// Upper abscissa of the bracket.
    pub hi: f64,
    /// `f(hi)`.
    pub f_hi: f64,
    /// Number of function evaluations spent (including the two endpoints).
    pub evaluations: u32,
}

impl Bracket {
    /// The endpoint with the smaller `|f|` — the best single-point estimate of the root.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        if self.f_lo.abs() <= self.f_hi.abs() {
            self.lo
        } else {
            self.hi
        }
    }

    /// The width of the bracket, an upper bound on the distance from [`Bracket::estimate`] to the
    /// true root.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }
}

/// Why a root could not be refined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootError {
    /// `f(a)` and `f(b)` have the same sign, so no root is bracketed. The refiner does not guess.
    NotBracketed,
    /// The tolerance was not a positive, finite number.
    BadTolerance,
    /// The bracket endpoints were not finite, or `a == b`.
    BadInterval,
}

/// Refine a root of `f` known to lie in `[a, b]`, to an abscissa tolerance `tol`.
///
/// `f` must be continuous on `[a, b]` and change sign across it. Returns the narrowest bracket the
/// method reached — width `≤ tol + 4·ε·|x|` (see [`Bracket`]).
///
/// # Errors
///
/// * [`RootError::NotBracketed`] if `f(a)` and `f(b)` share a sign (neither is zero).
/// * [`RootError::BadTolerance`] if `tol` is not positive and finite.
/// * [`RootError::BadInterval`] if `a`, `b` are not finite or are equal.
///
/// # Panics
///
/// Does not panic. A non-finite `f` value ends the loop at the current bracket rather than looping.
#[allow(clippy::many_single_char_names)] // a, b, c, d, e, p, q, r, s are the algorithm's own names
pub fn brent<F: FnMut(f64) -> f64>(
    mut f: F,
    a: f64,
    b: f64,
    tol: f64,
) -> Result<Bracket, RootError> {
    if !(tol.is_finite() && tol > 0.0) {
        return Err(RootError::BadTolerance);
    }
    if !(a.is_finite() && b.is_finite()) || a == b {
        return Err(RootError::BadInterval);
    }

    let (mut a, mut b) = (a, b);
    let mut fa = f(a);
    let mut fb = f(b);
    let mut evals = 2u32;

    // Endpoint roots: report a degenerate bracket rather than iterating toward one.
    if fa == 0.0 {
        return Ok(bracket_of(a, fa, a, fa, evals));
    }
    if fb == 0.0 {
        return Ok(bracket_of(b, fb, b, fb, evals));
    }
    // Precondition: opposite signs. `signum` handles ±0 and large magnitudes without overflow.
    if fa.signum() == fb.signum() {
        return Err(RootError::NotBracketed);
    }

    // `c` is the previous `b`, kept as the point of opposite sign to the current `b`.
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    loop {
        // Keep `b` as the best (smallest |f|) point, `c` as the opposite-sign counterpart.
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol1 = 2.0 * f64::EPSILON * b.abs() + 0.5 * tol;
        let xm = 0.5 * (c - b);

        if fb == 0.0 {
            // Exact root: a zero-width bracket at b, not the wide [b, c] whose far end is stale.
            return Ok(bracket_of(b, 0.0, b, 0.0, evals));
        }
        if xm.abs() <= tol1 {
            // Converged to a narrow bracket. Return [b, c], ordered, with the sign at each end.
            return Ok(bracket_of(b, fb, c, fc, evals));
        }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Attempt interpolation.
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                // Secant (linear).
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                // Inverse quadratic interpolation.
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * qq * (qq - r) - (b - a) * (r - 1.0));
                q = (qq - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            }
            p = p.abs();
            // Accept the interpolated step only if it stays in the bracket and beats bisection.
            let min1 = 3.0 * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if 2.0 * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            // Bisection forced.
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        if d.abs() > tol1 {
            b += d;
        } else {
            b += if xm > 0.0 { tol1 } else { -tol1 };
        }
        fb = f(b);
        evals += 1;

        // A non-finite evaluation cannot be bracketed against; stop at the current interval.
        if !fb.is_finite() {
            return Ok(bracket_of(b.midpoint(c), fc, c, fc, evals));
        }

        // Re-establish `c` as the opposite-sign point.
        if (fb > 0.0) == (fc > 0.0) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
    }
}

/// Raise a scan step to the smallest increment that provably advances a walk over `[t0, t1]`.
///
/// A step floor expressed as a fraction of the *span* does not by itself guarantee progress. The
/// walk advances by `t + h`, which rounds back to `t` whenever `h` falls below half an ulp of `t` —
/// and an ulp is set by the *magnitude* of `t`, not by the width of the interval. At a
/// seconds-since-J2000 epoch (`8.6e8` s) one ulp is `0.12 µs`, so a nanosecond floor over a
/// ten-second span never moves and the scan loops forever. `4·ε·|t|` is between four and eight ulps
/// at either endpoint, the same floating-point floor [`brent`] already carries in its own
/// convergence test.
#[must_use]
pub(crate) fn advancing_step(t0: f64, t1: f64, h: f64) -> f64 {
    h.max(4.0 * f64::EPSILON * t0.abs().max(t1.abs()))
}

/// Order two abscissae into a [`Bracket`] with `lo < hi`.
fn bracket_of(x0: f64, f0: f64, x1: f64, f1: f64, evaluations: u32) -> Bracket {
    if x0 <= x1 {
        Bracket {
            lo: x0,
            f_lo: f0,
            hi: x1,
            f_hi: f1,
            evaluations,
        }
    } else {
        Bracket {
            lo: x1,
            f_lo: f1,
            hi: x0,
            f_hi: f0,
            evaluations,
        }
    }
}
