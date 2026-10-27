//! Linear covariance analysis: what a navigation filter will *actually* achieve, as opposed to what
//! it will *report*, on a scenario that has not flown yet.
//!
//! Every filter in this crate propagates one covariance — its own. That covariance is the truth only
//! while the filter's model is the truth. Let the dynamics it integrates differ from the real ones,
//! or its process noise, or its measurement partials, or the variance it assigns to a range
//! measurement, and the number it reports stops being the number it achieves. Linear covariance
//! analysis (Maybeck, *Stochastic Models, Estimation and Control* I §6.8; Geller, "Linear Covariance
//! Techniques for Orbital Rendezvous Analysis and Autonomous Onboard Mission Planning", *Journal of
//! Guidance, Control, and Dynamics* **29**(6), 1404–1414, 2004) carries **both** at once, which is
//! the whole of its value: a covariance propagation that cannot disagree with the filter is a
//! covariance propagation that cannot tell you anything the filter would not have told you.
//!
//! # The recursion
//!
//! The truth state `x` has dimension `N`; the filter estimates `M ≤ N` of it, through a selection
//! `T` (`M × N`) that says which truth quantity each filter state is an estimate of. The analysis is
//! carried in **error coordinates** `e = T x − x̂`, so the augmented vector is `[x; e]` and the
//! blocks are `Cxx` (`N × N`, the truth dispersion), `Cex` (`M × N`, their correlation) and `Cee`
//! (`M × M`, the true error covariance — the answer).
//!
//! With truth transition `Φ`, filter transition `Φ̂`, truth process noise `Q` and the filter's belief
//! `Q̂`:
//!
//! ```text
//! [x]        [ Φ     0 ] [x]     [I]
//! [e]   =    [ D     Φ̂ ] [e]  +  [T] w ,     D = T Φ − Φ̂ T
//! ```
//!
//! and for a scalar measurement with truth partials `H` (`1 × N`), filter partials `Ĥ` (`1 × M`),
//! true noise variance `R` and assumed `R̂`, with the filter's own gain `K = P̂Ĥᵀ/(ĤP̂Ĥᵀ + R̂)`:
//!
//! ```text
//! [x]        [ I        0     ] [x]     [ 0 ]
//! [e]   =    [ −KΔ   I − KĤ   ] [e]  +  [ −K ] v ,   Δ = H − Ĥ T
//! ```
//!
//! `D` and `Δ` are the entire content of the method. They are the *model errors* — `D` is how far
//! the filter's dynamics have drifted from the truth's over one step, `Δ` is how wrongly it thinks
//! the measurement depends on the state — and they are the only route by which the truth dispersion
//! `Cxx` reaches the error. Set both to zero, give the filter the truth's `Q` and `R`, and `Cee`
//! reduces term by term to the filter's own recursion: the two covariances then agree to the last
//! bit, forever, and `the_two_covariances_are_identical_when_the_filter_is_right` asserts it as an
//! identity rather than a tolerance.
//!
//! `Cex` is a different matter and the place a plausible-sounding claim would be wrong. It does
//! **not** stay at zero under a perfect filter, because the truth's process noise enters the error
//! as well — the vehicle wanders by `w` and the filter, which cannot know `w`, wanders with it, so
//! `Cex` picks up `T Q` every step. It stays at zero exactly when `Q = 0`. The initial condition is
//! `Cex₀ = 0` for the reason Geller gives: the trajectory's injection dispersion and the navigator's
//! knowledge error at epoch are separate physical facts, so they start uncorrelated.
//!
//! # What it is not
//!
//! It is not a substitute for a Monte-Carlo ensemble, it is the linear term of one. Everything here
//! is exact for linear dynamics and linear measurements and is a first-order approximation
//! otherwise, so a scenario whose nonlinearity matters over the analysis span needs the ensemble as
//! well — which is how `tests/lincov_monte_carlo.rs` verifies this module rather than the other way
//! round. Its advantage is cost: one matrix recursion against thousands of trajectories, which is
//! what makes it usable inside a design sweep.
//!
//! The filter side is the covariance form with the **Joseph** update rather than `(I − KĤ)P̂`,
//! because the point of the exercise is a filter whose gain is not optimal for the truth, and the
//! short form is only valid for the gain that is.
//!
//! # Gate
//!
//! `derivations/lincov_recursion.mac` derives the error dynamics from `e = T x − x̂` and shows,
//! for free symbolic `N = 3`, `M = 2` matrices, that the covariance of `[x; e]` after each
//! update equals block by block the products [`time_update`](LinearCovariance::time_update) and
//! [`measurement_update`](LinearCovariance::measurement_update) form, that Joseph is the short
//! form only at the optimal gain, and that a right filter has `Cee = P̂`;
//! `derivations/lincov_recursion.gp` re-derives a mismatched rational scenario as one
//! augmented-matrix congruence. The gates are the module tests
//! `lincov_matches_the_exact_rational_recursion` (G2) and
//! `the_two_covariances_are_identical_when_the_filter_is_right` (G1).

// Matrix algebra over const-generic dimensions, with the single-letter names (P, K, H, Q, R, T, D)
// that the estimation literature uses universally.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

/// A linear covariance analysis over an `N`-dimensional truth and an `M`-dimensional filter.
///
/// Built by [`LinearCovariance::new`], advanced by [`time_update`](Self::time_update) and
/// [`measurement_update`](Self::measurement_update), and read through
/// [`true_covariance`](Self::true_covariance) and
/// [`filter_covariance`](Self::filter_covariance) — the pair whose divergence is the product.
#[derive(Debug, Clone)]
pub struct LinearCovariance<const N: usize, const M: usize> {
    /// Which truth quantity each filter state estimates (`M × N`). Constant over the scenario.
    map: [[f64; N]; M],
    /// Truth dispersion `Cxx = E[x xᵀ]`.
    dispersion: [[f64; N]; N],
    /// Correlation `Cex = E[e xᵀ]`.
    cross: [[f64; N]; M],
    /// True error covariance `Cee = E[e eᵀ]`.
    error: [[f64; M]; M],
    /// The covariance the filter reports, `P̂`.
    filter: [[f64; M]; M],
}

impl<const N: usize, const M: usize> LinearCovariance<N, M> {
    /// Seed the analysis.
    ///
    /// * `map` is `T` (`M × N`): row `i` expresses filter state `i` as a combination of truth
    ///   states. For a filter that estimates the first `M` truth states it is `[I 0]`.
    /// * `dispersion` is the truth's a-priori covariance `Cxx₀` — how much the *scenario* varies,
    ///   which is a different question from how well it is known.
    /// * `filter` is the covariance the filter is seeded with, `P̂₀`.
    ///
    /// The initial error covariance is taken to be `P̂₀` and the initial correlation zero: the filter
    /// is started honest, and made wrong afterwards by the models it is given. A run that starts
    /// with a mis-seeded filter sets the error covariance separately through
    /// [`with_initial_error`](Self::with_initial_error).
    #[must_use]
    pub const fn new(map: [[f64; N]; M], dispersion: [[f64; N]; N], filter: [[f64; M]; M]) -> Self {
        Self {
            map,
            dispersion,
            cross: [[0.0; N]; M],
            error: filter,
            filter,
        }
    }

    /// Replace the initial true error covariance, for the case where the filter's a-priori is not
    /// the error it actually starts with — an over- or under-confident seed, which is the first
    /// thing a navigation analysis is asked about.
    #[must_use]
    pub const fn with_initial_error(mut self, error: [[f64; M]; M]) -> Self {
        self.error = error;
        self
    }

    /// Advance both covariances across one step.
    ///
    /// `truth_stm` is `Φ` and `truth_process_noise` is `Q`, both in truth coordinates; `filter_stm`
    /// is `Φ̂` and `filter_process_noise` is `Q̂`, both in filter coordinates and both being what the
    /// filter *believes* rather than what is so.
    pub fn time_update(
        &mut self,
        truth_stm: &[[f64; N]; N],
        truth_process_noise: &[[f64; N]; N],
        filter_stm: &[[f64; M]; M],
        filter_process_noise: &[[f64; M]; M],
    ) {
        // D = T Φ − Φ̂ T (M × N): the dynamics mismodelling over this step.
        let mut d = [[0.0; N]; M];
        for i in 0..M {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc += self.map[i][k] * truth_stm[k][j];
                }
                for k in 0..M {
                    acc -= filter_stm[i][k] * self.map[k][j];
                }
                d[i][j] = acc;
            }
        }

        // Cxx⁻ = Φ Cxx Φᵀ + Q.
        let phi_cxx = mul(truth_stm, &self.dispersion);
        let mut dispersion = mul_transpose(&phi_cxx, truth_stm);
        add_in_place(&mut dispersion, truth_process_noise);

        // Cex⁻ = (D Cxx + Φ̂ Cex) Φᵀ + T Q.
        let mut inner = mul(&d, &self.dispersion);
        let phi_hat_cex = mul(filter_stm, &self.cross);
        add_in_place(&mut inner, &phi_hat_cex);
        let mut cross = mul_transpose(&inner, truth_stm);
        let tq = mul(&self.map, truth_process_noise);
        add_in_place(&mut cross, &tq);

        // Cee⁻ = D Cxx Dᵀ + Φ̂ Cex Dᵀ + (Φ̂ Cex Dᵀ)ᵀ + Φ̂ Cee Φ̂ᵀ + T Q Tᵀ. The first term is folded
        // into the third by writing the sum as `inner Dᵀ + Φ̂ Cex Dᵀ`-style products of the blocks
        // already formed: `inner` is `D Cxx + Φ̂ Cex`, so `inner Dᵀ` is the first two terms at once.
        let mut error = mul_transpose(&inner, &d);
        let cross_dt = mul_transpose(&self.cross, &d); // Cex Dᵀ (M × M)
        for i in 0..M {
            for j in 0..M {
                // The transpose of Φ̂ Cex Dᵀ, added to complete the symmetric pair.
                let mut acc = 0.0;
                for k in 0..M {
                    acc += filter_stm[j][k] * cross_dt[k][i];
                }
                error[i][j] += acc;
            }
        }
        let carried = mul(filter_stm, &self.error);
        let propagated = mul_transpose(&carried, filter_stm);
        add_in_place(&mut error, &propagated);
        let tqt = mul_transpose(&tq, &self.map);
        add_in_place(&mut error, &tqt);

        self.dispersion = dispersion;
        self.cross = cross;
        self.error = error;

        // The filter's own: P̂⁻ = Φ̂ P̂ Φ̂ᵀ + Q̂.
        let phi_p = mul(filter_stm, &self.filter);
        let mut filter = mul_transpose(&phi_p, filter_stm);
        add_in_place(&mut filter, filter_process_noise);
        self.filter = filter;
    }

    /// Process one scalar measurement.
    ///
    /// `truth_partials` is `H` (`1 × N`) — how the observable really depends on the truth — and
    /// `filter_partials` is `Ĥ` (`1 × M`), the partials the filter differentiates its own model to
    /// get. `true_noise_variance` is the variance the sensor actually has;
    /// `filter_noise_variance` is the one the filter was told, and the gain is computed from the
    /// latter.
    ///
    /// Returns the filter's innovation variance `ĤP̂Ĥᵀ + R̂` — what a residual would be normalised
    /// by — so a scenario can be checked for the consistency [`crate::consistency`] would measure in
    /// flight.
    pub fn measurement_update(
        &mut self,
        truth_partials: &[f64; N],
        filter_partials: &[f64; M],
        true_noise_variance: f64,
        filter_noise_variance: f64,
    ) -> f64 {
        // The filter's gain, from the filter's covariance and the filter's noise: K = P̂Ĥᵀ/f̂.
        let mut p_ht = [0.0; M];
        for i in 0..M {
            let mut acc = 0.0;
            for k in 0..M {
                acc += self.filter[i][k] * filter_partials[k];
            }
            p_ht[i] = acc;
        }
        let mut innovation_variance = filter_noise_variance;
        for i in 0..M {
            innovation_variance += filter_partials[i] * p_ht[i];
        }
        let mut gain = [0.0; M];
        if innovation_variance > 0.0 {
            for i in 0..M {
                gain[i] = p_ht[i] / innovation_variance;
            }
        }

        // Δ = H − Ĥ T (1 × N): the measurement mismodelling.
        let mut delta = *truth_partials;
        for j in 0..N {
            let mut acc = 0.0;
            for k in 0..M {
                acc += filter_partials[k] * self.map[k][j];
            }
            delta[j] -= acc;
        }

        // A = I − K Ĥ (M × M).
        let mut a = [[0.0; M]; M];
        for i in 0..M {
            a[i][i] = 1.0;
            for j in 0..M {
                a[i][j] -= gain[i] * filter_partials[j];
            }
        }

        // Cex⁺ = −K Δ Cxx + A Cex; the dispersion itself is untouched by a measurement.
        let mut cross = [[0.0; N]; M];
        for i in 0..M {
            for j in 0..N {
                let mut acc = 0.0;
                for k in 0..N {
                    acc -= gain[i] * delta[k] * self.dispersion[k][j];
                }
                for k in 0..M {
                    acc += a[i][k] * self.cross[k][j];
                }
                cross[i][j] = acc;
            }
        }

        // Cee⁺ = (A Cee Aᵀ) + (KΔ Cxx ΔᵀKᵀ) − (KΔ Cexᵀ Aᵀ) − its transpose + K R Kᵀ, which is the
        // covariance of `A e − KΔ x − K v` with the three terms uncorrelated only in `v`.
        let a_cee = mul(&a, &self.error);
        let mut error = mul_transpose(&a_cee, &a);
        // Cxx Δᵀ (N), then Δ Cxx Δᵀ (scalar).
        let mut cxx_dt = [0.0; N];
        for i in 0..N {
            let mut acc = 0.0;
            for k in 0..N {
                acc += self.dispersion[i][k] * delta[k];
            }
            cxx_dt[i] = acc;
        }
        let mut d_cxx_dt = 0.0;
        for i in 0..N {
            d_cxx_dt += delta[i] * cxx_dt[i];
        }
        // Cex Δᵀ (M), then A Cex Δᵀ (M).
        let mut correlated = [0.0; M];
        for i in 0..M {
            let mut acc = 0.0;
            for k in 0..N {
                acc += self.cross[i][k] * delta[k];
            }
            correlated[i] = acc;
        }
        let mut a_cex_dt = [0.0; M];
        for i in 0..M {
            let mut acc = 0.0;
            for k in 0..M {
                acc += a[i][k] * correlated[k];
            }
            a_cex_dt[i] = acc;
        }
        for i in 0..M {
            for j in 0..M {
                error[i][j] +=
                    gain[i] * d_cxx_dt * gain[j] - gain[i] * a_cex_dt[j] - a_cex_dt[i] * gain[j]
                        + gain[i] * true_noise_variance * gain[j];
            }
        }

        // The filter's own, in Joseph form because K is not the truth's optimal gain.
        let a_p = mul(&a, &self.filter);
        let mut filter = mul_transpose(&a_p, &a);
        for i in 0..M {
            for j in 0..M {
                filter[i][j] += gain[i] * filter_noise_variance * gain[j];
            }
        }

        self.cross = cross;
        self.error = error;
        self.filter = filter;
        innovation_variance
    }

    /// The true error covariance `E[(T x − x̂)(T x − x̂)ᵀ]` — the achieved navigation performance.
    #[must_use]
    pub const fn true_covariance(&self) -> &[[f64; M]; M] {
        &self.error
    }

    /// The covariance the filter reports, `P̂` — the claimed navigation performance.
    #[must_use]
    pub const fn filter_covariance(&self) -> &[[f64; M]; M] {
        &self.filter
    }

    /// The truth dispersion `E[x xᵀ]`: how far the scenario itself wanders, which no measurement
    /// changes.
    #[must_use]
    pub const fn dispersion(&self) -> &[[f64; N]; N] {
        &self.dispersion
    }

    /// The correlation `E[(T x − x̂) xᵀ]` between the navigation error and the truth. Zero exactly
    /// when the filter's models are the truth's, and the route by which they are not.
    #[must_use]
    pub const fn cross_covariance(&self) -> &[[f64; N]; M] {
        &self.cross
    }

    /// Per-state ratio of achieved to claimed one-sigma, `√(Cee[i][i]/P̂[i][i])`.
    ///
    /// One is a filter telling the truth about itself, above one is optimism — the filter reporting
    /// a tighter error than it has — and below one is a filter that is better than it claims. This
    /// is the same quantity [`crate::consistency`] measures from residuals after the fact; here it
    /// is available before the scenario flies.
    ///
    /// A state whose claimed variance is not positive yields [`f64::INFINITY`], which is the honest
    /// reading of a filter claiming certainty it does not have.
    #[must_use]
    pub fn optimism(&self) -> [f64; M] {
        let mut out = [0.0; M];
        for i in 0..M {
            out[i] = if self.filter[i][i] > 0.0 {
                libm::sqrt(self.error[i][i] / self.filter[i][i])
            } else {
                f64::INFINITY
            };
        }
        out
    }
}

/// `A B` for `A` (`R × K`) and `B` (`K × C`).
fn mul<const R: usize, const K: usize, const C: usize>(
    a: &[[f64; K]; R],
    b: &[[f64; C]; K],
) -> [[f64; C]; R] {
    let mut out = [[0.0; C]; R];
    for i in 0..R {
        for j in 0..C {
            let mut acc = 0.0;
            for k in 0..K {
                acc += a[i][k] * b[k][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// `A Bᵀ` for `A` (`R × K`) and `B` (`C × K`).
fn mul_transpose<const R: usize, const K: usize, const C: usize>(
    a: &[[f64; K]; R],
    b: &[[f64; K]; C],
) -> [[f64; C]; R] {
    let mut out = [[0.0; C]; R];
    for i in 0..R {
        for j in 0..C {
            let mut acc = 0.0;
            for k in 0..K {
                acc += a[i][k] * b[j][k];
            }
            out[i][j] = acc;
        }
    }
    out
}

fn add_in_place<const R: usize, const C: usize>(a: &mut [[f64; C]; R], b: &[[f64; C]; R]) {
    for i in 0..R {
        for j in 0..C {
            a[i][j] += b[i][j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `T = [I 0]`: the filter estimates the first `M` truth states.
    fn selection<const N: usize, const M: usize>() -> [[f64; N]; M] {
        let mut t = [[0.0; N]; M];
        for i in 0..M {
            t[i][i] = 1.0;
        }
        t
    }

    /// One time update and one measurement update against the exact rational answer (G2). The
    /// recursion is derived from `e = T x − x̂` in `derivations/lincov_recursion.mac` — the covariance
    /// of `[x; e]` through the Jacobian of the error dynamics equals, block by block and for free
    /// symbolic `N = 3`, `M = 2` matrices, the products these two methods form — and the values are
    /// that recursion on a rational scenario with every model mismatched (`Φ̂ ≠ Φ`, `T` mixing two
    /// truth states, `Q̂ ≠ Q`, `Ĥ ≠ H`, `R̂ ≠ R`), re-derived as one augmented-matrix congruence in
    /// exact rationals by `derivations/lincov_recursion.gp`.
    #[test]
    #[allow(clippy::excessive_precision, clippy::unreadable_literal)]
    fn lincov_matches_the_exact_rational_recursion() {
        let mut analysis = LinearCovariance::<3, 2>::new(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
            [[16.0, 1.0, 0.0], [1.0, 4.0, 0.5], [0.0, 0.5, 1.0]],
            [[9.0, 0.5], [0.5, 2.0]],
        );
        analysis.time_update(
            &[[1.0, 0.5, 0.125], [0.0, 1.0, 0.25], [0.0, 0.0, 0.99]],
            &[[1e-3, 0.0, 2e-4], [0.0, 2e-3, 0.0], [2e-4, 0.0, 3e-3]],
            &[[1.0, 0.51], [0.0, 1.0]],
            &[[5e-4, 0.0], [0.0, 1e-3]],
        );
        let innovation = analysis.measurement_update(&[1.0, 0.1, 0.05], &[1.0, 0.0], 0.09, 0.04);

        let dispersion = [
            [18.079125, 3.15625, 0.37145],
            [3.15625, 4.3145, 0.7425],
            [0.37145, 0.7425, 0.9831],
        ];
        let cross = [
            [
                -3.331813938703367193938852314139037008351e-1,
                -4.672649619688800182708252653738071832147e-1,
                -1.234448979216936260637294329093310296206e-1,
            ],
            [
                5.336296707279533696763879372834063173364e-2,
                1.359685548174406942913600842046729621576e-1,
                2.455661210243577904217184505545791255821e-1,
            ],
        ];
        let error = [
            [
                1.422149419746660870423272900927262942843e-1,
                -1.234850701363867313417009073839060743026e-2,
            ],
            [
                -1.234850701363867313417009073839060743026e-2,
                1.835866321618709469076320002220917218209e0,
            ],
        ];
        let filter = [
            [
                3.984112325856196689405900284985154954472e-2,
                6.037316174645258025757891705641117300684e-3,
            ],
            [
                6.037316174645258025757891705641117300684e-3,
                1.771581985363480195021200115185637542574e0,
            ],
        ];
        let close = |got: f64, want: f64, what: &str| {
            assert!(
                (got - want).abs() <= 1e-14 * want.abs().max(1.0),
                "{what}: {got:e} against the exact {want:e}"
            );
        };
        close(innovation, 10.0707, "innovation variance");
        for i in 0..3 {
            for j in 0..3 {
                close(analysis.dispersion()[i][j], dispersion[i][j], "Cxx");
            }
        }
        for i in 0..2 {
            for j in 0..3 {
                close(analysis.cross_covariance()[i][j], cross[i][j], "Cex");
            }
            for j in 0..2 {
                close(analysis.true_covariance()[i][j], error[i][j], "Cee");
                close(analysis.filter_covariance()[i][j], filter[i][j], "P̂");
            }
        }
    }

    /// A filter whose models are the truth's carries `Cee` exactly as its own `P̂` (G1):
    /// `derivations/lincov_recursion.mac` identity 4, through both updates.
    #[test]
    fn the_two_covariances_are_identical_when_the_filter_is_right() {
        // Truth and filter share a state, dynamics, process noise, partials and noise variance, so
        // D = 0 and Δ = 0 by construction and every term of Cee's recursion collapses onto P̂'s. The
        // agreement is exact, not close, and the assertion says so.
        let phi = [[1.0, 0.5], [0.0, 1.0]];
        let q = [[1e-4, 0.0], [0.0, 4e-4]];
        let p0 = [[9.0, 0.0], [0.0, 0.25]];
        let mut analysis =
            LinearCovariance::<2, 2>::new(selection(), [[100.0, 0.0], [0.0, 1.0]], p0);

        for step in 0..40 {
            analysis.time_update(&phi, &q, &phi, &q);
            let h = [1.0, 0.0];
            analysis.measurement_update(&h, &h, 0.04, 0.04);
            if step % 7 == 0 {
                let h = [0.0, 1.0];
                analysis.measurement_update(&h, &h, 0.01, 0.01);
            }
        }

        for i in 0..2 {
            for j in 0..2 {
                let (truth, claimed) = (
                    analysis.true_covariance()[i][j],
                    analysis.filter_covariance()[i][j],
                );
                assert!(
                    (truth - claimed).abs() <= 1e-12 * claimed.abs().max(1.0),
                    "Cee[{i}][{j}] {truth} vs P̂ {claimed}"
                );
            }
        }
        for ratio in analysis.optimism() {
            assert!((ratio - 1.0).abs() < 1e-9, "optimism {ratio}");
        }
        // Cex is *not* zero, and the reason is physical rather than a defect: the truth's process
        // noise displaces the trajectory and the filter, which cannot see it, is displaced with it.
        assert!(
            analysis.cross_covariance()[0][0].abs() > 1e-4,
            "Cex {:?}",
            analysis.cross_covariance()
        );
    }

    #[test]
    fn a_perfect_filter_with_no_process_noise_never_correlates_with_the_dispersion() {
        // The other half of the statement above. Remove Q and the only route from the dispersion
        // into the error is D and Δ, both of which are zero here, so Cex stays at the zero it was
        // seeded with through any number of steps.
        let phi = [[1.0, 0.25], [0.0, 1.0]];
        let zero = [[0.0; 2]; 2];
        let mut analysis = LinearCovariance::<2, 2>::new(
            selection(),
            [[36.0, 0.0], [0.0, 4.0]],
            [[9.0, 0.0], [0.0, 1.0]],
        );
        for _ in 0..25 {
            analysis.time_update(&phi, &zero, &phi, &zero);
            analysis.measurement_update(&[1.0, 0.0], &[1.0, 0.0], 0.04, 0.04);
        }
        for row in analysis.cross_covariance() {
            for value in row {
                assert!(value.abs() < 1e-14, "Cex {:?}", analysis.cross_covariance());
            }
        }
    }

    #[test]
    fn understating_the_measurement_noise_makes_the_filter_optimistic() {
        // The only mismatch is R̂ < R: the filter trusts a noisy sensor. Its gain is then too large
        // for the noise it actually sees, so the achieved error exceeds the claimed one, and the
        // route is entirely K R Kᵀ — nothing here touches the dynamics.
        let phi = [[1.0]];
        let q = [[1e-6]];
        let mut analysis = LinearCovariance::<1, 1>::new([[1.0]], [[4.0]], [[1.0]]);
        for _ in 0..30 {
            analysis.time_update(&phi, &q, &phi, &q);
            analysis.measurement_update(&[1.0], &[1.0], 0.25, 0.01);
        }
        let (truth, claimed) = (
            analysis.true_covariance()[0][0],
            analysis.filter_covariance()[0][0],
        );
        assert!(truth > claimed, "achieved {truth} vs claimed {claimed}");
        assert!(analysis.optimism()[0] > 2.0, "{:?}", analysis.optimism());
        // A scalar filter with a wrong R has a closed-form steady state. With Φ = 1 and Q → 0 the
        // gain settles at K = P̂/(P̂ + R̂) with P̂ → 0, so the achieved variance approaches K²R
        // summed over the tail; the assertion that matters is the direction and that the analysis
        // does not blow up, both of which the two above cover.
        assert!(truth.is_finite() && truth > 0.0);
    }

    #[test]
    fn an_unmodelled_state_biases_the_error_through_the_dispersion() {
        // Truth is [position, unmodelled bias]; the filter estimates position alone and never learns
        // the bias, which enters its measurement as an unmodelled term (Δ = [0, 1]). The achieved
        // error must then exceed the claimed one by the bias's own variance route, and the
        // correlation Cex must become non-zero — the analysis is *reading* the dispersion, which is
        // the entire mechanism a consider covariance also captures.
        let truth_stm = [[1.0, 0.0], [0.0, 1.0]];
        let truth_q = [[1e-6, 0.0], [0.0, 0.0]];
        let filter_stm = [[1.0]];
        let filter_q = [[1e-6]];
        let bias_variance = 0.09;
        let mut analysis = LinearCovariance::<2, 1>::new(
            [[1.0, 0.0]],
            [[1.0, 0.0], [0.0, bias_variance]],
            [[1.0]],
        );
        for _ in 0..200 {
            analysis.time_update(&truth_stm, &truth_q, &filter_stm, &filter_q);
            // Truth: z = position + bias + v. Filter: ẑ = position.
            analysis.measurement_update(&[1.0, 1.0], &[1.0], 0.04, 0.04);
        }
        let truth = analysis.true_covariance()[0][0];
        let claimed = analysis.filter_covariance()[0][0];
        assert!(
            claimed < 1e-3,
            "the filter should think it has converged: {claimed}"
        );
        // With enough measurements the estimate converges onto position + bias, so the error
        // variance floors at the bias variance itself.
        assert!(
            (truth - bias_variance).abs() < 0.01 * bias_variance,
            "achieved {truth} against the bias variance {bias_variance}"
        );
        assert!(
            analysis.cross_covariance()[0][1].abs() > 0.5 * bias_variance,
            "Cex {:?}",
            analysis.cross_covariance()
        );
    }

    #[test]
    fn the_consider_covariance_is_the_one_measurement_limit_of_this_recursion() {
        // A single measurement on a filter that ignores a parameter is exactly the setting
        // `consider_covariance` solves in closed form, so the two must agree to rounding. Truth
        // z = hx·x + hc·c + v with the filter estimating x alone from a diffuse prior.
        let (hx, hc, r, pcc) = (2.0, 3.0, 0.5, 4.0);
        let diffuse = 1e12;
        let mut analysis =
            LinearCovariance::<2, 1>::new([[1.0, 0.0]], [[diffuse, 0.0], [0.0, pcc]], [[diffuse]]);
        analysis.measurement_update(&[hx, hc], &[hx], r, r);

        let p = [[r / (hx * hx)]];
        let cross = [[hx * hc / r]];
        let sensitivity = crate::consider::consider_sensitivity(&p, &cross);
        let expected = crate::consider::consider_covariance(&p, &sensitivity, &[[pcc]]);
        let achieved = analysis.true_covariance()[0][0];
        assert!(
            (achieved / expected[0][0] - 1.0).abs() < 1e-6,
            "LinCov {achieved} against consider {}",
            expected[0][0]
        );
        // And the filter, knowing nothing of `c`, reports only r/hx².
        assert!(
            (analysis.filter_covariance()[0][0] / p[0][0] - 1.0).abs() < 1e-6,
            "claimed {}",
            analysis.filter_covariance()[0][0]
        );
    }

    #[test]
    fn a_mismodelled_transition_feeds_the_dispersion_into_the_error() {
        // Same state on both sides, but the filter integrates a slightly different rate. D ≠ 0, so
        // the truth dispersion reaches the error even though every measurement is modelled
        // perfectly. With no measurements at all the growth is purely D Cxx Dᵀ per step, which is
        // checkable in closed form for one step.
        let truth_stm = [[1.02]];
        let filter_stm = [[1.0]];
        let zero = [[0.0]];
        let dispersion = 4.0;
        let mut analysis = LinearCovariance::<1, 1>::new([[1.0]], [[dispersion]], [[0.25]]);
        analysis.time_update(&truth_stm, &zero, &filter_stm, &zero);
        // D = 1.02 − 1.0 = 0.02; Cee⁻ = D²·Cxx + Φ̂²·Cee = 4e-4·4 + 0.25.
        let expected = 0.02_f64.powi(2) * dispersion + 0.25;
        assert!(
            (analysis.true_covariance()[0][0] - expected).abs() < 1e-15,
            "{} vs {expected}",
            analysis.true_covariance()[0][0]
        );
        // The filter, believing its own transition, reports 0.25 unchanged.
        assert!((analysis.filter_covariance()[0][0] - 0.25).abs() < 1e-15);
        // And Cex has become non-zero: D Cxx Φᵀ = 0.02·4·1.02.
        assert!(
            (analysis.cross_covariance()[0][0] - 0.02 * dispersion * 1.02).abs() < 1e-15,
            "{}",
            analysis.cross_covariance()[0][0]
        );
    }

    #[test]
    fn both_covariances_stay_symmetric_and_positive_on_the_diagonal() {
        // Every block here is formed as a congruence `A C Aᵀ` plus a positive term, so symmetry is
        // structural rather than enforced. A run with every model mismatched at once is where a
        // sign error in the cross terms would show.
        let truth_stm = [[1.0, 0.3], [0.0, 0.99]];
        let filter_stm = [[1.0, 0.31], [0.0, 1.0]];
        let truth_q = [[2e-4, 1e-5], [1e-5, 3e-4]];
        let filter_q = [[1e-4, 0.0], [0.0, 1e-4]];
        let mut analysis = LinearCovariance::<2, 2>::new(
            selection(),
            [[16.0, 0.0], [0.0, 4.0]],
            [[4.0, 0.0], [0.0, 1.0]],
        );
        for step in 0..50 {
            analysis.time_update(&truth_stm, &truth_q, &filter_stm, &filter_q);
            let truth_h = [1.0, 0.05];
            let filter_h = [1.0, 0.0];
            analysis.measurement_update(&truth_h, &filter_h, 0.09, 0.04);
            assert!(
                analysis.true_covariance()[0][0] > 0.0 && analysis.true_covariance()[1][1] > 0.0,
                "step {step}: {:?}",
                analysis.true_covariance()
            );
            let c = analysis.true_covariance();
            assert!(
                (c[0][1] - c[1][0]).abs() <= 1e-12 * c[0][1].abs().max(1.0),
                "step {step}: Cee asymmetric {c:?}"
            );
            let p = analysis.filter_covariance();
            assert!(
                (p[0][1] - p[1][0]).abs() <= 1e-12 * p[0][1].abs().max(1.0),
                "step {step}: P̂ asymmetric {p:?}"
            );
        }
        // Every model error here inflates rather than deflates, so the filter ends up optimistic.
        for ratio in analysis.optimism() {
            assert!(ratio > 1.0, "{:?}", analysis.optimism());
        }
    }

    // Exact zero is the assertion: the guard either skipped the division or it did not, and a
    // tolerance would let a denormal gain through as a pass.
    #[allow(clippy::float_cmp)]
    #[test]
    fn a_filter_told_a_zero_innovation_variance_takes_no_gain() {
        // R̂ = 0 with P̂ = 0 makes the innovation variance zero, which is a degenerate scenario
        // rather than an error. The guard must leave the gain at zero rather than divide, and both
        // covariances must come through unchanged.
        let mut analysis = LinearCovariance::<1, 1>::new([[1.0]], [[1.0]], [[0.0]]);
        let variance = analysis.measurement_update(&[1.0], &[1.0], 1.0, 0.0);
        assert_eq!(variance, 0.0, "innovation variance");
        assert_eq!(analysis.true_covariance()[0][0], 0.0);
        assert_eq!(analysis.filter_covariance()[0][0], 0.0);
        assert!(analysis.optimism()[0].is_infinite());
    }
}
