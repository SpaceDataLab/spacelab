//! The full tesseral geopotential `C̄_nm, S̄_nm` to high degree/order — longitude-dependent gravity,
//! the piece the zonal-only [`crate::forces::Zonal`] cannot represent.
//!
//! # Provenance and verification
//!
//! The acceleration is `a = ∇U` of the geopotential
//! `U = (μ/r) Σₙ (Rₑ/r)ⁿ Σₘ P̄_nm(sinφ)[C̄_nm cos mλ + S̄_nm sin mλ]`, evaluated by a **fully-normalized**
//! Cunningham/Gottlieb body-fixed recursion: normalized solid-harmonic functions `V̄_nm, W̄_nm` are
//! built by three-term recurrences in `(x, y, z)` — no pole singularity, no factorial overflow — and
//! the acceleration is a linear combination of them with the normalized coefficients directly.
//!
//! The normalized recursion is *derived*, not transcribed: substituting `V_nm = V̄_nm / N_nm` into the
//! unnormalized Cunningham recursion (with `N_nm = √[(2−δ₀ₘ)(2n+1)(n−m)!/(n+m)!]`) turns its integer
//! coefficients into the Holmes–Featherstone normalized constants
//! `a_nm = √[(2n−1)(2n+1)/((n−m)(n+m))]`, `b_nm = √[(2n+1)(n+m−1)(n−m−1)/((2n−3)(n−m)(n+m))]`, and the
//! acceleration's degree-mixing picks up the closed-form ratios `N_nm/N_{n+1,m'}` below. Everything is
//! checked: the tests reproduce the symbolic Maxima gradient oracle (`derivations/geopotential.mac`) at
//! two points to 1e-9, cross-check a zonal-only model against the verified [`crate::forces::Zonal`],
//! confirm a near-pole evaluation is finite, and confirm a degree-70 field is stable and genuinely
//! differs from degree 20 (where all modern models still agree) — and, one octave up, that a degree-120
//! field is finite and differs from degree 70 by the short-wavelength terms.
//!
//! Written over [`Scalar`], so the STM Jacobian comes from AD and stays symplectic. The normalized form
//! keeps every intermediate `O(1)`, so it is sound far past the ~degree-30 ceiling of the unnormalized
//! recursion; [`GeoPotential::MAX_DEGREE`] caps it at 120 for the embedded model and the array sizes.
//!
//! # Memory
//!
//! A `no_std` spherical-harmonic model to degree `N` needs `O(N²)` storage for the coefficient triangle
//! and the precomputed normalization tables (recursion and acceleration constants, so no `sqrt` runs per
//! call), with no `alloc` to spill it to the heap — so the struct is large (~0.8 MB at degree 120) and
//! held by reference. The per-call solid-harmonic scratch, by contrast, is only `O(N)`: the Gottlieb
//! recurrence for row `n` reads only rows `n−1`/`n−2`, so [`GeoPotential::acceleration`] keeps just three
//! rolling rows (`~3·DIM`, a few kB at `f64`) instead of the full `O(N²)` array — which also drops the
//! per-call zero-fill from the whole array to the rows actually computed. That leanness matters most for
//! STM via AD, where each scratch element carries `N+1` dual partials. The precompute makes evaluation
//! ~2× faster at high degree (≈8 ns/term).

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::similar_names
)]

use crate::forces::Acceleration;
use alloc::boxed::Box;
use spacelab_autodiff::Scalar;

/// Highest degree/order supported (array bound and embedded-model cap).
const MAX_DEGREE: usize = 120;
/// Array dimension: solid harmonics run to degree `MAX_DEGREE + 1`.
const DIM: usize = MAX_DEGREE + 2;

/// The tesseral geopotential as an [`Acceleration`]. The `C̄₀₀ = 1` term *is* the point mass, so this
/// is a complete gravity model — use it alone, not added to a separate `PointMass`.
///
/// Coefficients are stored **fully normalized** (the EGM/ICGEM convention), so no de-normalization —
/// and no factorial overflow — happens at construction or evaluation.
#[derive(Clone)]
pub struct GeoPotential {
    mu: f64,
    re: f64,
    degree: usize,
    // The coefficient tables are ≈0.8 MB in total, so they live on the heap: a `GeoPotential` built by
    // value on the stack (and a few nested in a call chain) overflows a thread stack. Indexing is
    // unchanged — `Box<[[f64; DIM]; DIM]>` derefs to the array.
    cbar: Box<[[f64; DIM]; DIM]>,
    sbar: Box<[[f64; DIM]; DIM]>,
    // Precomputed, position-independent constants (Gottlieb's normalization tables): the recursion
    // coefficients and the acceleration's degree-mixing ratios, so no `sqrt` runs in the hot loop.
    rec_a: Box<[[f64; DIM]; DIM]>, // ζ-coefficient of the V̄/W̄ recursion
    rec_b: Box<[[f64; DIM]; DIM]>, // η-coefficient
    sec: [f64; DIM],               // sectorial diagonal coefficient c_m
    acc_alpha: Box<[[f64; DIM]; DIM]>,
    acc_beta: Box<[[f64; DIM]; DIM]>,  // [n][0] holds β₀(n)
    acc_gamma: Box<[[f64; DIM]; DIM]>, // [n][0] holds γ₀(n)
}

/// A heap-allocated zeroed `DIM×DIM` table, built without a giant stack temporary (`Box::new([[…]])`
/// would construct the array on the stack first).
fn boxed_zeros() -> Box<[[f64; DIM]; DIM]> {
    let rows: Box<[[f64; DIM]]> = alloc::vec![[0.0f64; DIM]; DIM].into_boxed_slice();
    match rows.try_into() {
        Ok(table) => table,
        Err(_) => unreachable!("built with exactly DIM rows"),
    }
}

impl core::fmt::Debug for GeoPotential {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GeoPotential")
            .field("mu", &self.mu)
            .field("re", &self.re)
            .field("degree", &self.degree)
            .finish_non_exhaustive()
    }
}

/// `(n−m)! / (n+m)! = 1 / Π_{k=n−m+1}^{n+m} k`, formed as a product (no factorial overflow).
fn ratio_factorial(n: usize, m: usize) -> f64 {
    let mut denom = 1.0;
    for k in (n - m + 1)..=(n + m) {
        denom *= k as f64;
    }
    1.0 / denom
}

/// `N_nm = √[(2 − δ₀ₘ)(2n + 1)(n−m)!/(n+m)!]` — the factor relating normalized to unnormalized
/// coefficients (`C_nm = C̄_nm · N_nm`). Only used to *accept* unnormalized input at low degree.
fn normalization(n: usize, m: usize) -> f64 {
    let delta = if m == 0 { 1.0 } else { 2.0 };
    libm::sqrt(delta * (2 * n + 1) as f64 * ratio_factorial(n, m))
}

impl GeoPotential {
    /// The maximum degree/order this type supports.
    pub const MAX_DEGREE: usize = MAX_DEGREE;

    fn empty(mu: f64, re: f64, degree: usize) -> Self {
        Self {
            mu,
            re,
            degree,
            cbar: boxed_zeros(),
            sbar: boxed_zeros(),
            rec_a: boxed_zeros(),
            rec_b: boxed_zeros(),
            sec: [0.0; DIM],
            acc_alpha: boxed_zeros(),
            acc_beta: boxed_zeros(),
            acc_gamma: boxed_zeros(),
        }
    }

    /// Fill the position-independent constant tables (Gottlieb's normalization factors). Called once
    /// per model, so no `sqrt` runs in [`Acceleration::acceleration`].
    fn precompute(&mut self) {
        let top = self.degree + 1;
        for n in 1..=top {
            for m in 0..=n {
                if m == 0 || n > m {
                    self.rec_a[n][m] =
                        libm::sqrt(((2 * n - 1) * (2 * n + 1)) as f64 / ((n - m) * (n + m)) as f64);
                }
                if n >= m + 2 {
                    self.rec_b[n][m] = libm::sqrt(
                        ((2 * n + 1) * (n + m - 1) * (n - m - 1)) as f64
                            / ((2 * n - 3) * (n - m) * (n + m)) as f64,
                    );
                }
            }
        }
        for m in 1..=top {
            self.sec[m] = if m == 1 {
                libm::sqrt(3.0)
            } else {
                libm::sqrt((2 * m + 1) as f64 / (2 * m) as f64)
            };
        }
        for n in 0..=self.degree {
            let nf = (2 * n + 1) as f64 / (2 * n + 3) as f64;
            self.acc_gamma[n][0] = libm::sqrt(0.5 * nf * ((n + 2) * (n + 1)) as f64);
            self.acc_beta[n][0] = libm::sqrt(nf);
            for m in 1..=n {
                self.acc_gamma[n][m] = libm::sqrt(nf * ((n + m + 2) * (n + m + 1)) as f64);
                self.acc_beta[n][m] = libm::sqrt(nf * (n + m + 1) as f64 / (n + 1 - m) as f64);
                self.acc_alpha[n][m] = if m == 1 {
                    libm::sqrt(2.0 * nf / ((n + 1) * n) as f64)
                } else {
                    libm::sqrt(nf / ((n - m + 2) * (n - m + 1)) as f64)
                };
            }
        }
    }

    /// Build from **normalized** coefficients `(n, m, C̄_nm, S̄_nm)` (the EGM/ICGEM convention). The
    /// `C̄₀₀ = 1` point-mass term is set automatically. `None` if `degree > MAX_DEGREE`.
    #[must_use]
    pub fn from_normalized(
        mu: f64,
        re: f64,
        degree: usize,
        coeffs: &[(usize, usize, f64, f64)],
    ) -> Option<Self> {
        if degree > MAX_DEGREE {
            return None;
        }
        let mut g = Self::empty(mu, re, degree);
        g.cbar[0][0] = 1.0;
        for &(n, m, cbar, sbar) in coeffs {
            if n > degree || m > n {
                continue;
            }
            g.cbar[n][m] = cbar;
            g.sbar[n][m] = sbar;
        }
        g.precompute();
        Some(g)
    }

    /// Build from **unnormalized** coefficients `(n, m, C_nm, S_nm)`, normalizing on the way in. For
    /// cross-checking against a hand-derived potential at low degree; production data is normalized.
    #[must_use]
    pub fn from_unnormalized(
        mu: f64,
        re: f64,
        degree: usize,
        coeffs: &[(usize, usize, f64, f64)],
    ) -> Option<Self> {
        if degree > MAX_DEGREE {
            return None;
        }
        let mut g = Self::empty(mu, re, degree);
        g.cbar[0][0] = 1.0;
        for &(n, m, cnm, snm) in coeffs {
            if n > degree || m > n {
                continue;
            }
            let inv = 1.0 / normalization(n, m);
            g.cbar[n][m] = cnm * inv;
            g.sbar[n][m] = snm * inv;
        }
        g.precompute();
        Some(g)
    }

    /// The embedded gravity field ([`crate::geopotential_coeffs::MODEL_NAME`] — currently `XGM2019e`)
    /// truncated to `degree` (≤ [`Self::MAX_DEGREE`]), using the model's own `GM` and reference radius.
    #[must_use]
    pub fn embedded(degree: usize) -> Option<Self> {
        use crate::geopotential_coeffs::{MODEL_COEFFS, MODEL_GM, MODEL_MAX_DEGREE, MODEL_RE};
        if degree > MODEL_MAX_DEGREE {
            return None;
        }
        Self::from_normalized(MODEL_GM, MODEL_RE, degree, MODEL_COEFFS)
    }

    /// The gravitational parameter the coefficients are referenced to, m³/s².
    #[must_use]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// The reference radius the coefficients are referenced to, m.
    #[must_use]
    pub fn reference_radius(&self) -> f64 {
        self.re
    }

    /// The highest degree this field was built with.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// The stored **fully normalized** `(C̄_nm, S̄_nm)`, or `None` outside the field's degree.
    ///
    /// Normalized is the only form kept, which is the point of the accessor rather than an accident
    /// of it: the textbook lunar `C₂₂ = 2.238e-5` and the stored `C̄₂₂ = 3.467e-5` differ by
    /// `N₂₂ = √(5/12)`, and a caller comparing the stored number against a published one without
    /// de-normalizing is out by 55 %.
    #[must_use]
    pub fn normalized(&self, n: usize, m: usize) -> Option<(f64, f64)> {
        if n > self.degree || m > n {
            return None;
        }
        Some((self.cbar[n][m], self.sbar[n][m]))
    }

    /// The r-derived quantities the Gottlieb recursion needs: `re_inv_r = Rₑ/r`, `zeta = z·Rₑ/r²`,
    /// `eta = Rₑ²/r²`, `xi = x·Rₑ/r²`, `ups = y·Rₑ/r²`.
    fn harmonic_quantities<S: Scalar>(&self, r: &[S; 3]) -> (S, S, S, S, S) {
        let (x, y, z) = (r[0], r[1], r[2]);
        let inv_r = (x * x + y * y + z * z).sqrt().recip();
        let inv_r2 = inv_r * inv_r;
        let re = S::from_f64(self.re);
        (
            z * re * inv_r2,  // zeta = z·Rₑ/r²
            re * re * inv_r2, // eta  = Rₑ²/r²
            x * re * inv_r2,  // xi   = x·Rₑ/r²
            y * re * inv_r2,  // ups  = y·Rₑ/r²
            re * inv_r,       // re_inv_r = Rₑ/r
        )
    }

    /// One row `n` of the normalized solid harmonics `V̄, W̄` (Gottlieb body-fixed recursion),
    /// returned in a fresh pair. Every element of row `n` reads only rows `n-1`/`n-2` (the sectorial
    /// diagonal reads the row-`n-1` diagonal), so the full `DIM×DIM` scratch the old `fill_harmonics`
    /// zeroed and held is unnecessary — three rolling rows suffice, cutting the per-call scratch from
    /// `2·DIM²` to `~3·DIM` (238 KB → 6 KB at f64, far more for the AD/STM `Dual` path) and the
    /// zero-fill from the whole array to only the rows actually computed. **Bit-identical** to that
    /// fill: the per-element formula and evaluation order are unchanged. `vp1/wp1` are row `n-1`,
    /// `vp2/wp2` row `n-2` (both unread for `n < 2`). Quantities from [`Self::harmonic_quantities`].
    #[allow(clippy::too_many_arguments)] // the recursion state: index, five r-quantities, two prior rows
    fn geo_row<S: Scalar>(
        &self,
        n: usize,
        zeta: S,
        eta: S,
        xi: S,
        ups: S,
        re_inv_r: S,
        vp1: &[S; DIM],
        wp1: &[S; DIM],
        vp2: &[S; DIM],
        wp2: &[S; DIM],
    ) -> ([S; DIM], [S; DIM]) {
        let zero = S::from_f64(0.0);
        let mut vn = [zero; DIM];
        let mut wn = [zero; DIM];
        if n == 0 {
            vn[0] = re_inv_r; // V̄₀₀ = Rₑ/r; W̄₀₀ = 0
            return (vn, wn);
        }
        // zonal column m = 0 (W̄ₙ₀ ≡ 0, left zero)
        vn[0] = S::from_f64(self.rec_a[n][0]) * zeta * vp1[0]
            - if n >= 2 {
                S::from_f64(self.rec_b[n][0]) * eta * vp2[0]
            } else {
                zero
            };
        // off-diagonal 1 ≤ m < n
        for m in 1..n {
            let anm = S::from_f64(self.rec_a[n][m]);
            let (p2v, p2w) = if n >= m + 2 {
                (vp2[m], wp2[m])
            } else {
                (zero, zero)
            };
            let bnm = if n >= m + 2 {
                S::from_f64(self.rec_b[n][m])
            } else {
                zero
            };
            vn[m] = anm * zeta * vp1[m] - bnm * eta * p2v;
            wn[m] = anm * zeta * wp1[m] - bnm * eta * p2w;
        }
        // sectorial diagonal m = n
        let cm = S::from_f64(self.sec[n]);
        vn[n] = cm * (xi * vp1[n - 1] - ups * wp1[n - 1]);
        wn[n] = cm * (xi * wp1[n - 1] + ups * vp1[n - 1]);
        (vn, wn)
    }

    /// f64-specialised acceleration with a `wide::f64x4` accumulation of the acceleration sum — the
    /// flop-bound hot path (the geopotential is flop-bound; the row-major layout already makes the
    /// order `m` contiguous, so lane-parallelism across `m` is the win). Matches the `Scalar`-generic
    /// [`Acceleration::acceleration`] to within the model's 1e-9 validation tolerance: the recursion
    /// is shared and per-lane arithmetic is identical; only the cross-lane summation order differs.
    /// SSE2 baseline; AVX2 with `target-feature=+avx2` / `target-cpu=native`.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::similar_names)]
    pub fn acceleration_simd(&self, r: &[f64; 3]) -> [f64; 3] {
        use wide::f64x4;
        let (zeta, eta, xi, ups, re_inv_r) = self.harmonic_quantities(r);
        // Three rolling rows of V̄/W̄: the acceleration sum for degree n reads row n+1, and the
        // Gottlieb recurrence for a row reads only the two preceding rows, so the old full DIM×DIM
        // scratch is unnecessary. Seed rows 0 and 1.
        let mut vr = [[0.0f64; DIM]; 3];
        let mut wr = [[0.0f64; DIM]; 3];
        let (v0, w0) = self.geo_row(
            0, zeta, eta, xi, ups, re_inv_r, &vr[0], &wr[0], &vr[0], &wr[0],
        );
        vr[0] = v0;
        wr[0] = w0;
        let (v1, w1) = self.geo_row(
            1, zeta, eta, xi, ups, re_inv_r, &vr[0], &wr[0], &vr[0], &wr[0],
        );
        vr[1] = v1;
        wr[1] = w1;

        let load4 = |s: &[f64; DIM], i: usize| f64x4::new([s[i], s[i + 1], s[i + 2], s[i + 3]]);
        let zero4 = f64x4::splat(0.0);
        let half4 = f64x4::splat(0.5);

        let (mut ax, mut ay, mut az) = (0.0f64, 0.0f64, 0.0f64);
        for n in 0..=self.degree {
            // Advance the ring to hold row n+1 (rows 0/1 were seeded above).
            if n >= 1 {
                let j = n + 1;
                let (vn, wn) = self.geo_row(
                    j,
                    zeta,
                    eta,
                    xi,
                    ups,
                    re_inv_r,
                    &vr[(j - 1) % 3],
                    &wr[(j - 1) % 3],
                    &vr[(j - 2) % 3],
                    &wr[(j - 2) % 3],
                );
                vr[j % 3] = vn;
                wr[j % 3] = wn;
            }
            let vn1 = &vr[(n + 1) % 3];
            let wn1 = &wr[(n + 1) % 3];

            // m = 0 zonal boundary (scalar)
            let cn0 = self.cbar[n][0];
            let g0 = self.acc_gamma[n][0];
            ax -= cn0 * g0 * vn1[1];
            ay -= cn0 * g0 * wn1[1];
            az -= (n + 1) as f64 * self.acc_beta[n][0] * cn0 * vn1[0];

            // m = 1..=n, four orders at a time
            let (mut axv, mut ayv, mut azv) = (zero4, zero4, zero4);
            let mut m = 1usize;
            while m + 4 <= n + 1 {
                let cnm = load4(&self.cbar[n], m);
                let snm = load4(&self.sbar[n], m);
                let gamma = load4(&self.acc_gamma[n], m);
                let beta = load4(&self.acc_beta[n], m);
                let alpha = load4(&self.acc_alpha[n], m);
                let (vp1, vm1, v0) = (load4(vn1, m + 1), load4(vn1, m - 1), load4(vn1, m));
                let (wp1, wm1, w0) = (load4(wn1, m + 1), load4(wn1, m - 1), load4(wn1, m));
                let nm1 = f64x4::new([
                    (n - m + 1) as f64,
                    (n - m) as f64,
                    (n - m - 1) as f64,
                    (n - m - 2) as f64,
                ]);
                let fac = nm1 * (nm1 + f64x4::splat(1.0));
                axv += (gamma * (zero4 - cnm * vp1 - snm * wp1)
                    + fac * alpha * (cnm * vm1 + snm * wm1))
                    * half4;
                ayv += (gamma * (snm * vp1 - cnm * wp1) + fac * alpha * (snm * vm1 - cnm * wm1))
                    * half4;
                azv += nm1 * beta * (zero4 - cnm * v0 - snm * w0);
                m += 4;
            }
            ax += axv.reduce_add();
            ay += ayv.reduce_add();
            az += azv.reduce_add();

            // remainder m..=n (scalar)
            while m <= n {
                let (cnm, snm) = (self.cbar[n][m], self.sbar[n][m]);
                let (gamma, beta, alpha) = (
                    self.acc_gamma[n][m],
                    self.acc_beta[n][m],
                    self.acc_alpha[n][m],
                );
                let fac = ((n - m + 1) * (n - m + 2)) as f64;
                ax += 0.5
                    * (gamma * (-cnm * vn1[m + 1] - snm * wn1[m + 1])
                        + fac * alpha * (cnm * vn1[m - 1] + snm * wn1[m - 1]));
                ay += 0.5
                    * (gamma * (snm * vn1[m + 1] - cnm * wn1[m + 1])
                        + fac * alpha * (snm * vn1[m - 1] - cnm * wn1[m - 1]));
                az += (n - m + 1) as f64 * beta * (-cnm * vn1[m] - snm * wn1[m]);
                m += 1;
            }
        }
        let mu_re2 = self.mu / (self.re * self.re);
        [mu_re2 * ax, mu_re2 * ay, mu_re2 * az]
    }
}

impl Acceleration for GeoPotential {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let zero = S::from_f64(0.0);
        let (zeta, eta, xi, ups, re_inv_r) = self.harmonic_quantities(r);
        // Three rolling rows of V̄/W̄ (the degree-n sum reads row n+1; the recurrence needs only the
        // two preceding rows), seeded at 0 and 1 — no full DIM×DIM scratch, which matters most here on
        // the AD/STM `Dual` path where every element carries N+1 partials.
        let mut vr = [[zero; DIM]; 3];
        let mut wr = [[zero; DIM]; 3];
        let (v0, w0) = self.geo_row(
            0, zeta, eta, xi, ups, re_inv_r, &vr[0], &wr[0], &vr[0], &wr[0],
        );
        vr[0] = v0;
        wr[0] = w0;
        let (v1, w1) = self.geo_row(
            1, zeta, eta, xi, ups, re_inv_r, &vr[0], &wr[0], &vr[0], &wr[0],
        );
        vr[1] = v1;
        wr[1] = w1;

        // acceleration = (μ/Rₑ²) Σ contributions, with the closed-form normalization ratios
        // α = N_nm/N_{n+1,m-1}, β = N_nm/N_{n+1,m}, γ = N_nm/N_{n+1,m+1}.
        let mu_re2 = S::from_f64(self.mu / (self.re * self.re));
        let half = S::from_f64(0.5);
        let (mut ax, mut ay, mut az) = (zero, zero, zero);
        for n in 0..=self.degree {
            if n >= 1 {
                let j = n + 1;
                let (vn, wn) = self.geo_row(
                    j,
                    zeta,
                    eta,
                    xi,
                    ups,
                    re_inv_r,
                    &vr[(j - 1) % 3],
                    &wr[(j - 1) % 3],
                    &vr[(j - 2) % 3],
                    &wr[(j - 2) % 3],
                );
                vr[j % 3] = vn;
                wr[j % 3] = wn;
            }
            let vn1 = &vr[(n + 1) % 3];
            let wn1 = &wr[(n + 1) % 3];
            for m in 0..=n {
                let cnm = S::from_f64(self.cbar[n][m]);
                let snm = S::from_f64(self.sbar[n][m]);
                if m == 0 {
                    let gamma0 = S::from_f64(self.acc_gamma[n][0]);
                    let beta0 = S::from_f64(self.acc_beta[n][0]);
                    ax = ax - cnm * gamma0 * vn1[1];
                    ay = ay - cnm * gamma0 * wn1[1];
                    az = az - S::from_f64((n + 1) as f64) * beta0 * cnm * vn1[0];
                } else {
                    let gamma = S::from_f64(self.acc_gamma[n][m]);
                    let beta = S::from_f64(self.acc_beta[n][m]);
                    let alpha = S::from_f64(self.acc_alpha[n][m]);
                    let fac = S::from_f64(((n - m + 1) * (n - m + 2)) as f64);
                    ax = ax
                        + half
                            * (gamma * (zero - cnm * vn1[m + 1] - snm * wn1[m + 1])
                                + fac * alpha * (cnm * vn1[m - 1] + snm * wn1[m - 1]));
                    ay = ay
                        + half
                            * (gamma * (zero - cnm * wn1[m + 1] + snm * vn1[m + 1])
                                + fac * alpha * (zero - cnm * wn1[m - 1] + snm * vn1[m - 1]));
                    az = az
                        + S::from_f64((n - m + 1) as f64)
                            * beta
                            * (zero - cnm * vn1[m] - snm * wn1[m]);
                }
            }
        }
        [mu_re2 * ax, mu_re2 * ay, mu_re2 * az]
    }

    /// f64 fast path: the `wide::f64x4` [`Self::acceleration_simd`]. Matches the generic sum to
    /// 4.5e-15 — far inside the model's 1e-9 tolerance — so a plain f64 propagation is transparently
    /// faster with no accuracy change. (Velocity is unused by the conservative gravity field.)
    fn acceleration_f64(&self, _t: f64, r: &[f64; 3], _v: &[f64; 3]) -> [f64; 3] {
        self.acceleration_simd(r)
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::unwrap_used,
        clippy::cast_precision_loss
    )]
    use super::*;
    use crate::forces::Zonal;
    use spacelab_testkit::nan_max;

    const MU: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;

    // The same unnormalized coefficient set the Maxima oracle uses (derivations/geopotential.mac).
    fn oracle_model() -> GeoPotential {
        let coeffs = [
            (2, 0, -1.0826e-3, 0.0),
            (2, 1, 1.0e-6, 2.0e-6),
            (2, 2, 1.5e-6, -0.9e-6),
            (3, 0, 2.5e-6, 0.0),
            (3, 1, 1.1e-6, -0.7e-6),
            (3, 2, 0.8e-6, 0.5e-6),
            (3, 3, -0.6e-6, 0.4e-6),
            (4, 0, -1.6e-6, 0.0),
            (4, 2, 0.3e-6, -0.2e-6),
        ];
        GeoPotential::from_unnormalized(MU, RE, 4, &coeffs).unwrap()
    }

    /// The normalized recursion reproduces the symbolic-gradient oracle at two off-axis points to 1e-9.
    #[test]
    fn matches_the_maxima_gradient_oracle() {
        let g = oracle_model();
        let cases = [
            (
                [7_000_123.0, 1_200_456.0, 900_789.0],
                [
                    -7.613_097_285_278_765,
                    -1.305_432_785_993_487_8,
                    -0.982_071_225_354_376_7,
                ],
            ),
            (
                [-4_200_000.0, 5_100_000.0, -2_600_000.0],
                [
                    4.679_392_436_932_629,
                    -5.682_393_986_354_161,
                    2.904_389_678_644_647_7,
                ],
            ),
        ];
        for (r, expect) in cases {
            let acc = g.acceleration(0.0, &r, &[0.0; 3]);
            for i in 0..3 {
                let rel = (acc[i] - expect[i]).abs() / expect[i].abs();
                assert!(
                    rel < 1e-9,
                    "axis {i}: {} vs oracle {} (rel {rel:.2e})",
                    acc[i],
                    expect[i]
                );
            }
        }
    }

    /// A zonal-only geopotential reproduces `PointMass + Zonal` exactly.
    #[test]
    fn zonal_only_matches_zonal_force() {
        let (j2, j3, j4) = (1.082_626e-3, -2.532_7e-6, -1.619_9e-6);
        let geo = GeoPotential::from_unnormalized(
            MU,
            RE,
            4,
            &[(2, 0, -j2, 0.0), (3, 0, -j3, 0.0), (4, 0, -j4, 0.0)],
        )
        .unwrap();
        let zonal = Zonal::<3> {
            mu: MU,
            re: RE,
            j: [j2, j3, j4],
        };
        let pm = crate::forces::PointMass { mu: MU };
        let r = [6_800_000.0, 900_000.0, 2_100_000.0];
        let a_geo = geo.acceleration(0.0, &r, &[0.0; 3]);
        let a_zon = zonal.acceleration(0.0, &r, &[0.0; 3]);
        let a_pm = pm.acceleration(0.0, &r, &[0.0; 3]);
        for i in 0..3 {
            let want = a_pm[i] + a_zon[i];
            assert!(
                (a_geo[i] - want).abs() < 1e-9 * want.abs().max(1e-6),
                "axis {i}: {} vs {want}",
                a_geo[i]
            );
        }
    }

    /// The embedded field (degree 12) is dominated by the point mass with the J₂ oblateness bump.
    #[test]
    fn embedded_field_is_sane() {
        let g = GeoPotential::embedded(12).unwrap();
        let r = [6_778_000.0, 0.0, 0.0];
        let a = g.acceleration(0.0, &r, &[0.0; 3]);
        let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let newton = crate::geopotential_coeffs::MODEL_GM / (6_778_000.0_f64 * 6_778_000.0);
        println!(
            "{} (12) |a| = {mag:.6} m/s²  (point mass {newton:.6})",
            crate::geopotential_coeffs::MODEL_NAME
        );
        assert!(
            (mag - newton).abs() / newton < 2e-3,
            "|a| {mag} far from point mass {newton}"
        );
        assert!(
            a[0] < 0.0 && a[0].abs() > 8.0,
            "radial accel wrong: {}",
            a[0]
        );
    }

    /// The f64 SIMD fast path reproduces the `Scalar`-generic acceleration to far inside the model's
    /// own 1e-9 validation tolerance (per-lane arithmetic is identical; only the sum's cross-lane
    /// order differs) — so `acceleration_simd` preserves accuracy while parallelising the flops.
    #[test]
    fn simd_matches_scalar_within_tolerance() {
        for degree in [8usize, 20, 70, 100, 120] {
            let g = GeoPotential::embedded(degree).unwrap();
            let r = [5_000_000.0, 3_500_000.0, 2_600_000.0];
            let scal = g.acceleration(0.0, &r, &[0.0; 3]);
            let simd = g.acceleration_simd(&r);
            let mag = (scal[0] * scal[0] + scal[1] * scal[1] + scal[2] * scal[2]).sqrt();
            let mut max_rel = 0.0f64;
            for i in 0..3 {
                max_rel = nan_max(max_rel, (scal[i] - simd[i]).abs() / mag);
            }
            std::eprintln!("degree {degree}: SIMD vs scalar max rel {max_rel:.3e}");
            assert!(
                max_rel < 1e-12,
                "degree {degree}: SIMD rel err {max_rel:.3e} exceeds 1e-12"
            );
        }
    }

    /// End-to-end wiring: a plain f64 propagation system `Plain(Newtonian(geo))` routes through the
    /// f64 fast path, so its derivative's acceleration is *bit-for-bit* the SIMD output — proving the
    /// `eval_f64`/`acceleration_f64` chain reaches `acceleration_simd`, not the scalar default.
    #[test]
    fn f64_propagation_routes_through_simd() {
        use crate::DynamicalSystem;
        use crate::forces::{Newtonian, Plain};
        let g = GeoPotential::embedded(70).unwrap();
        let r = [5_000_000.0, 3_500_000.0, 2_600_000.0];
        let sys = Plain(Newtonian(g.clone()));
        let y = [r[0], r[1], r[2], 1.0, 2.0, 3.0];
        let mut dydt = [0.0; 6];
        sys.derivative(0.0, &y, &mut dydt);
        let simd = g.acceleration_simd(&r);
        for i in 0..3 {
            assert!(
                (dydt[3 + i] - simd[i]).abs() < 1e-30,
                "f64 propagation did not route to acceleration_simd"
            );
        }
    }

    /// High degree is stable and carries information degree 20 does not: the embedded field at degree
    /// 70 is finite, physical, and differs from degree 20 by the short-wavelength terms.
    #[test]
    fn high_degree_is_stable_and_distinct() {
        let g70 = GeoPotential::embedded(70).unwrap();
        let g20 = GeoPotential::embedded(20).unwrap();
        let r = [5_000_000.0, 3_500_000.0, 2_600_000.0]; // ~660 km, off-axis
        let a70 = g70.acceleration(0.0, &r, &[0.0; 3]);
        let a20 = g20.acceleration(0.0, &r, &[0.0; 3]);
        let mag70 = (a70[0] * a70[0] + a70[1] * a70[1] + a70[2] * a70[2]).sqrt();
        let diff =
            ((a70[0] - a20[0]).powi(2) + (a70[1] - a20[1]).powi(2) + (a70[2] - a20[2]).powi(2))
                .sqrt();
        println!("degree70 |a| = {mag70:.6}, |a70 - a20| = {diff:.3e} m/s²");
        assert!(
            mag70.is_finite() && (8.0..12.0).contains(&mag70),
            "degree-70 |a| unphysical: {mag70}"
        );
        assert!(
            diff > 1e-8 && diff < 1e-2,
            "degree 70 vs 20 delta off: {diff:.3e}"
        );
    }

    /// The lifted ceiling is real: a degree-120 field is finite, physical, and carries short-wavelength
    /// information degree 70 cannot — the two differ by the degree 71..120 terms. This is
    /// [`high_degree_is_stable_and_distinct`] one octave up, proving `MAX_DEGREE = 120` genuinely
    /// evaluates past 70 (the old ceiling) rather than merely accepting the higher-degree coefficients.
    #[test]
    fn degree_120_is_finite_and_distinct_from_70() {
        let g120 = GeoPotential::embedded(120).unwrap();
        let g70 = GeoPotential::embedded(70).unwrap();
        let r = [5_000_000.0, 3_500_000.0, 2_600_000.0]; // ~660 km, off-axis
        let a120 = g120.acceleration(0.0, &r, &[0.0; 3]);
        let a70 = g70.acceleration(0.0, &r, &[0.0; 3]);
        let mag120 = (a120[0] * a120[0] + a120[1] * a120[1] + a120[2] * a120[2]).sqrt();
        let diff =
            ((a120[0] - a70[0]).powi(2) + (a120[1] - a70[1]).powi(2) + (a120[2] - a70[2]).powi(2))
                .sqrt();
        println!("degree120 |a| = {mag120:.6}, |a120 - a70| = {diff:.3e} m/s²");
        assert!(
            mag120.is_finite() && (8.0..12.0).contains(&mag120),
            "degree-120 |a| unphysical: {mag120}"
        );
        assert!(
            diff > 1e-9 && diff < 1e-3,
            "degree 120 vs 70 delta off: {diff:.3e}"
        );
    }

    /// The degree-70 field matches an INDEPENDENT implementation: `derivations/geopotential_validate.py`
    /// evaluates the same `XGM2019e` coefficients with `SciPy` associated-Legendre functions and a numerical
    /// gradient (nothing shared with the Cartesian solid-harmonic recursion here), and agrees to ~1e-8.
    #[test]
    fn validated_against_scipy_at_degree_70() {
        let g = GeoPotential::embedded(70).unwrap();
        let cases = [
            (
                [5_000_000.0, 3_500_000.0, 2_600_000.0],
                [-6.8286333121, -4.7801292054, -3.5615291595],
            ),
            (
                [6_600_000.0, 1_200_000.0, 900_000.0],
                [-8.4960579611, -1.5448160693, -1.1618676633],
            ),
        ];
        for (r, expect) in cases {
            let a = g.acceleration(0.0, &r, &[0.0; 3]);
            for i in 0..3 {
                let rel = (a[i] - expect[i]).abs() / expect[i].abs();
                assert!(
                    rel < 1e-6,
                    "axis {i}: {} vs SciPy {} (rel {rel:.2e})",
                    a[i],
                    expect[i]
                );
            }
        }
    }

    /// A near-polar evaluation is finite (the Cartesian solid-harmonic form is pole-nonsingular).
    #[test]
    fn near_pole_is_finite() {
        let g = GeoPotential::embedded(30).unwrap();
        let r = [1.0, 0.0, 6_900_000.0]; // essentially over the pole
        let a = g.acceleration(0.0, &r, &[0.0; 3]);
        assert!(
            a.iter().all(|c| c.is_finite()),
            "non-finite near the pole: {a:?}"
        );
        assert!(
            a[2] < 0.0 && a[2].abs() > 8.0,
            "polar radial accel wrong: {}",
            a[2]
        );
    }

    /// Both constructors take a whole model and a requested degree, and drop the rows above it — that
    /// is how a degree-120 table is used at degree 20. The guard is `n > degree || m > n`, and with
    /// `&&` in its place a row is dropped only when it fails *both* halves, so an above-degree row is
    /// written instead of skipped. Rows between `degree` and `DIM` land in a corner of the table
    /// nobody reads, which is why an accuracy test cannot see this; a row above `DIM` indexes off the
    /// end of it. Asserted here as the promise itself: the model truncated by the caller and the
    /// model truncated by the constructor are the same field.
    // The two fields must agree to the bit: an ignored row leaves no trace at all.
    #[allow(clippy::float_cmp)]
    #[test]
    fn rows_above_the_requested_degree_are_ignored_not_written() {
        let in_range = [
            (2, 0, -1.0826e-3, 0.0),
            (2, 1, 1.0e-6, 2.0e-6),
            (3, 2, 0.8e-6, 0.5e-6),
        ];
        let with_extra = [
            in_range[0],
            in_range[1],
            // Just past the requested degree — inside the table, outside the sum.
            (4, 1, 7.0e-6, -3.0e-6),
            in_range[2],
            // Past `DIM` as well: writing this one is an out-of-bounds index, not a stray value.
            (MAX_DEGREE + 40, 5, 9.0e-6, 4.0e-6),
            // Order above degree: never a coefficient at all.
            (3, 7, -5.0e-6, 6.0e-6),
        ];
        let r = [5_600_000.0, -3_100_000.0, 2_400_000.0];
        for (label, want, got) in [
            (
                "normalized",
                GeoPotential::from_normalized(MU, RE, 3, &in_range).unwrap(),
                GeoPotential::from_normalized(MU, RE, 3, &with_extra).unwrap(),
            ),
            (
                "unnormalized",
                GeoPotential::from_unnormalized(MU, RE, 3, &in_range).unwrap(),
                GeoPotential::from_unnormalized(MU, RE, 3, &with_extra).unwrap(),
            ),
        ] {
            let (want, got) = (
                want.acceleration(0.0, &r, &[0.0; 3]),
                got.acceleration(0.0, &r, &[0.0; 3]),
            );
            for i in 0..3 {
                assert_eq!(got[i], want[i], "{label} axis {i}");
            }
        }
    }

    /// `degree > MAX_DEGREE` is the one refusal both constructors make. `from_normalized`'s is
    /// asserted by `degree_120_is_finite_and_distinct_from_70`; `from_unnormalized`'s was not, so
    /// tightening it to `>=` (refusing the top degree the crate supports) or loosening it to `==`
    /// (accepting a degree whose recursion runs off the end of the table) went unnoticed.
    #[test]
    fn the_degree_bound_admits_max_degree_and_refuses_one_more() {
        let coeffs = [(2, 0, -1.0826e-3, 0.0)];
        for (label, at_max, past_max) in [
            (
                "normalized",
                GeoPotential::from_normalized(MU, RE, MAX_DEGREE, &coeffs).is_some(),
                GeoPotential::from_normalized(MU, RE, MAX_DEGREE + 1, &coeffs).is_some(),
            ),
            (
                "unnormalized",
                GeoPotential::from_unnormalized(MU, RE, MAX_DEGREE, &coeffs).is_some(),
                GeoPotential::from_unnormalized(MU, RE, MAX_DEGREE + 1, &coeffs).is_some(),
            ),
        ] {
            assert!(at_max, "{label}: degree {MAX_DEGREE} must be buildable");
            assert!(
                !past_max,
                "{label}: degree {} must be refused",
                MAX_DEGREE + 1
            );
        }
    }

    /// The `Debug` impl names the field and its three scalars. It is public API — anything that logs
    /// a propagator configuration goes through it — and nothing asserted it, so a body replaced by
    /// `Ok(())` printed an empty string and no test noticed.
    #[test]
    fn the_debug_impl_names_the_field_and_its_scalars() {
        let shown = alloc::format!("{:?}", GeoPotential::embedded(7).unwrap());
        for expected in ["GeoPotential", "mu", "re", "degree: 7"] {
            assert!(shown.contains(expected), "Debug lost {expected:?}: {shown}");
        }
    }
}
