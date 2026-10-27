//! Gaseous band absorption as a k-distribution, and how it composes with the scattering slab.
//!
//! [`crate::radiative_transfer::Atmosphere`] is grey: one optical depth, one single-scattering albedo,
//! one phase function. That is a fair model of Rayleigh scattering and of an aerosol, whose extinction
//! varies slowly across a 10 nm channel. It is not a model of a molecular band. At 10 nm resolution
//! each of water vapour, carbon dioxide, nitrous oxide, methane and oxygen is thousands of individual
//! lines, and the channel-averaged transmittance
//!
//! ```text
//!     T(u) = <exp(-k(v) u)>
//! ```
//!
//! is **not** an exponential in the absorber column `u`: the average of exponentials is not the
//! exponential of the average, and the gap is not small — it is the difference between a band that
//! saturates in its line cores while its wings still transmit, and a grey slab that does neither. So a
//! single "effective gaseous optical depth" cannot be right at more than one column, and fitting one
//! at the scene's column silently rescales every other column in the same retrieval.
//!
//! What is right is the **k-distribution**: sort the channel by absorption coefficient rather than by
//! wavenumber, and the average becomes a sum,
//!
//! ```text
//!     T(u) = sum_j w_j exp(-k_j u),   sum_j w_j = 1,   w_j >= 0,   k_j >= 0,
//! ```
//!
//! with `w_j` the fraction of the channel's width absorbing at `k_j`. The reordering approximates
//! nothing — an exponential sum does not care what order its terms are in — and unlike a single
//! optical depth this form **composes with a scattering calculation**: each term is a grey absorber,
//! so the transport runs once per term and the results are weight-summed. That is the whole reason to
//! carry the table. [`KDistribution`] is one absorber's row of it; [`gas_optical_depths`] combines the
//! absorbers acting in one channel; [`crate::radiative_transfer::Atmosphere::with_absorber`] builds the
//! slab one term sees.
//!
//! The coefficients themselves are not in this crate's source. They are fitted line-by-line from
//! HITRAN by `tools/gen_gas_transmittance.py`, which writes
//! `tests/fixtures/gas_transmittance.txt` — HITRAN grants no redistribution, so what is committed is
//! the fit and never a line parameter. That fixture carries, per channel and per gas, one shared
//! weight vector and one rate vector per surface-pressure node, plus the column `u_max` past which the
//! channel is opaque and must be refused rather than extrapolated.
//!
//! Gate: `derivations/gas_k_distribution.mac` shows the multiplication rule is the product of the
//! two transmittances, that an air mass multiplies the column, and that the resort's merge keeps
//! the zeroth and first moments and — Jensen — can only lose transmittance;
//! `derivations/gas_k_distribution.gp` recomputes its literals and samples the sign. The module test
//! `the_band_transmittance_is_its_defining_sum` (G2) holds the shipped sums to that file's 40-digit
//! values.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

/// Why a k-distribution was rejected.
///
/// Each names a violated precondition of the sum, not a modelling preference. A term list that is not
/// a probability distribution over absorption coefficients does not merely fit worse: it makes
/// `T(0) != 1`, so the channel gains or loses photons at zero absorber, and nothing downstream can
/// tell that from a real band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GasError {
    /// No terms at all. A channel with no absorption is one term of unit weight at zero rate, which
    /// is what the fixture writes; an empty list is a parse that went wrong.
    NoTerms,
    /// A negative or non-finite weight, or a negative or non-finite rate.
    NotADistribution,
    /// The weights do not sum to 1.
    WeightsDoNotSumToOne,
    /// A negative or non-finite absorber column.
    ColumnNotPhysical,
}

/// How far the weights may sum from 1 before the table is rejected. The fixture writes ten decimal
/// places per weight and up to sixteen weights, so honest rounding cannot exceed ~1e-9.
const WEIGHT_TOLERANCE: f64 = 1.0e-8;

/// One absorber's band absorption over one channel, at one scene's vertical column.
///
/// `terms` are `(weight, rate)` pairs straight out of the fixture row for this channel, this gas and
/// this surface pressure. `column` is the absorber's **vertical** column in the units the rates were
/// tabulated against — precipitable centimetres for water, `P / 1013.25 hPa` for every well-mixed gas.
///
/// Vertical, not slant, and the distinction is the one place this type is easy to get wrong. The
/// fixture tabulates `T(u)` against the *slant* column `u = column * air_mass`, because that is what a
/// two-way transmittance factor needs and what `radcalnet_toa.txt` carries per row. A rate is a
/// coefficient either way — `k * u` is the slant optical depth and `k * column` the vertical one — so
/// the transport is handed the vertical depth and works the geometry out itself, photon by photon,
/// which is exactly what makes this better than a two-way factor: the diffuse path that scatters at
/// 30 km and arrives at the sensor never had an air mass of `1/mu`.
#[derive(Clone, Copy, Debug)]
pub struct KDistribution<'a> {
    terms: &'a [(f64, f64)],
    column: f64,
}

impl<'a> KDistribution<'a> {
    /// Wrap a fixture row's `(weight, rate)` terms at a vertical absorber `column`.
    ///
    /// Zero-weight padding terms are accepted and carried: they contribute nothing to any sum, and
    /// rejecting them would force every caller to trim the fixed-width rows the fixture writes.
    ///
    /// # Errors
    ///
    /// [`GasError::NoTerms`], [`GasError::NotADistribution`], [`GasError::WeightsDoNotSumToOne`] or
    /// [`GasError::ColumnNotPhysical`].
    pub fn new(terms: &'a [(f64, f64)], column: f64) -> Result<Self, GasError> {
        if terms.is_empty() {
            return Err(GasError::NoTerms);
        }
        if !column.is_finite() || column < 0.0 {
            return Err(GasError::ColumnNotPhysical);
        }
        let mut total = 0.0;
        for &(weight, rate) in terms {
            if !weight.is_finite() || weight < 0.0 || !rate.is_finite() || rate < 0.0 {
                return Err(GasError::NotADistribution);
            }
            total += weight;
        }
        if libm::fabs(total - 1.0) > WEIGHT_TOLERANCE {
            return Err(GasError::WeightsDoNotSumToOne);
        }
        Ok(Self { terms, column })
    }

    /// The direct-beam transmittance along a path of `air_mass` times the vertical column — the
    /// fixture's own `T(u)` at `u = column * air_mass`.
    ///
    /// This is the quantity the table was fitted to, so it is the one that reproduces it exactly, and
    /// it is what a comparison against the table should use. It is *not* what the transport needs:
    /// a scattered photon's path is not one air mass, which is why [`gas_optical_depths`] exists.
    ///
    /// Accurate only over the fit's own domain, `u <= u_max` from the same fixture row. Past that the
    /// channel is opaque and a caller must refuse it; this function will keep returning the sum's
    /// asymptote, which is a number and not an answer.
    #[must_use]
    pub fn transmittance(&self, air_mass: f64) -> f64 {
        let u = self.column * air_mass;
        self.terms
            .iter()
            .map(|&(weight, rate)| weight * libm::exp(-rate * u))
            .sum()
    }

    /// The vertical gaseous optical depth of term `j`, `k_j * column`.
    fn depth(&self, index: usize) -> f64 {
        self.terms[index].1 * self.column
    }
}

/// Combine the absorbers acting in one channel into `(weight, vertical gaseous optical depth)` pairs
/// — one grey absorber per term, which is what the transport adds to the scattering slab.
///
/// # The multiplication rule, and what it assumes
///
/// Gases' k-distributions are combined by taking every combination of terms: weights multiply and
/// optical depths add, because `exp(-tau_1) exp(-tau_2) = exp(-(tau_1 + tau_2))`. That is exact if the
/// absorbers' line positions inside the channel are **statistically independent** — if knowing that a
/// wavenumber sits in a strong water line tells you nothing about whether it also sits in a strong
/// methane line. It is the standard assumption (MODTRAN, 6S, libRadtran's REPTRAN all make it), it is
/// not derivable, and where two bands genuinely overlap it is the error term.
///
/// It is exactly right, rather than merely standard, in the case this crate mostly meets: a channel
/// where one gas absorbs and the rest are transparent. Each transparent one is a single term of unit
/// weight at zero rate, the product is the absorbing one unchanged, and nothing has been assumed.
///
/// # Cost
///
/// The product of an `n`-term and an `m`-term distribution has `n*m` terms, and the transport runs
/// once per term. Two 16-term gases in one channel is 256 transport runs where one gas is 16, and the
/// SWIR channels where water, carbon dioxide, nitrous oxide and methane all act multiply four of them.
/// Callers that cannot afford the product for such a channel have to say so with a measured pruning
/// error; this function does not prune, because a silent cap on the terms is indistinguishable in the
/// output from a band that was never there.
#[must_use]
pub fn gas_optical_depths(gases: &[KDistribution<'_>]) -> Vec<(f64, f64)> {
    let mut terms = Vec::from([(1.0, 0.0)]);
    for gas in gases {
        let mut next = Vec::with_capacity(terms.len() * gas.terms.len());
        for &(weight, depth) in &terms {
            for (index, &(gas_weight, _)) in gas.terms.iter().enumerate() {
                if gas_weight > 0.0 {
                    next.push((weight * gas_weight, depth + gas.depth(index)));
                }
            }
        }
        terms = next;
    }
    terms
}

/// The same product, resorted onto at most `bins` terms — for the channels whose exact product a
/// caller cannot afford.
///
/// # Why this is needed and not a premature optimisation
///
/// Measured on the shipped table: the median `RadCalNet` channel is 18816 product terms and the worst
/// is 368640 at 2170 nm, so one sweep over the 211 published wavelengths is 5575735 transport runs
/// per scene. That is not a slow test, it is a test that never finishes, and no term count small
/// enough to run is reachable by choosing better gases.
///
/// # What is preserved exactly, and what is not
///
/// This is the classical correlated-k resort: order the product terms by optical depth, cut the
/// ordering into `bins` intervals of equal weight, and replace each interval by its total weight at
/// its weight-averaged depth. Two things survive exactly, up to floating-point summation:
///
///   * `sum_j w_j`, so the output is still a distribution and a transparent channel is still
///     transparent;
///   * `sum_j w_j tau_j`, the mean optical depth, so the weak-absorption limit is untouched — and
///     since `1 - <exp(-tau)> <= <tau>` by Jensen, a channel this crate calls transparent is still
///     provably transparent afterwards.
///
/// What is *not* preserved is the transmittance itself. Merging terms of unequal depth replaces
/// `<exp(-tau)>` by `exp(<tau>)` within each bin, and `exp` is convex, so every bin transmits at most
/// what it should: the resort is biased, one-signed, and its size is the within-bin depth spread. It
/// is therefore a quantity to measure against [`gas_optical_depths`], not to assume small — which is
/// what `the_resort_preserves_the_two_moments_it_claims_to` and the `RadCalNet` sweep do.
///
/// Ordering by depth is what makes the bias small rather than merely bounded: terms that merge are
/// then adjacent in depth, so the within-bin spread is as small as any partition into `bins` parts
/// can make it. Binning the terms in their tabulated order instead would merge a line core with a
/// window and lose the whole point of the k-distribution.
#[must_use]
pub fn gas_optical_depths_binned(
    gases: &[KDistribution<'_>],
    bins: NonZeroUsize,
) -> Vec<(f64, f64)> {
    let mut terms = Vec::from([(1.0, 0.0)]);
    for gas in gases {
        let mut next = Vec::with_capacity(terms.len() * gas.terms.len());
        for &(weight, depth) in &terms {
            for (index, &(gas_weight, _)) in gas.terms.iter().enumerate() {
                if gas_weight > 0.0 {
                    next.push((weight * gas_weight, depth + gas.depth(index)));
                }
            }
        }
        terms = resort(next, bins.get());
    }
    terms
}

/// Cut `terms` into `bins` equal-weight intervals of the depth ordering, each collapsed to its total
/// weight at its weight-averaged depth.
///
/// Resorting after every gas rather than once at the end is what bounds the transient: the working
/// set never exceeds `bins` times one gas's term count, where the full product would be their
/// product. It costs one extra merge per gas, which the measured error pays for.
fn resort(mut terms: Vec<(f64, f64)>, bins: usize) -> Vec<(f64, f64)> {
    if terms.len() <= bins {
        return terms;
    }
    terms.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    // Bins carry equal shares of *transmitted* weight, `w exp(-tau)`, not of weight. Equal weight is
    // the obvious rule and it is the wrong one: in a strong band most of the channel is opaque --
    // measured on the shipped table, 49 % of the product's weight sits past tau = 20 at 1390 nm,
    // where exp(-tau) < 2e-9 -- so equal-weight bins spend half their number resolving depths that
    // all transmit the same nothing, and starve the small-depth end where the whole transmittance
    // lives. Weighting by exp(-tau) gives each depth range bins in proportion to what it contributes,
    // which is the quantity the bias is measured in. It needs no threshold: the opaque tail advances
    // the measure so little that it falls into one trailing bin on its own.
    let mut share: Vec<f64> = terms
        .iter()
        .map(|&(weight, depth)| weight * libm::exp(-depth))
        .collect();
    let mut total: f64 = share.iter().sum();
    if total <= 0.0 {
        // Every term saturated, so no depth is distinguishable from any other by transmittance and
        // there is nothing for the measure to resolve. Fall back to weight, which is at least a
        // partition.
        share = terms.iter().map(|&(weight, _)| weight).collect();
        total = share.iter().sum();
    }
    let mut out: Vec<(f64, f64)> = Vec::with_capacity(bins);
    // `spent` is the measure banked by the bins already closed; `cumulative` includes the bin
    // still being filled.
    let (mut cumulative, mut spent, mut weight, mut moment) = (0.0, 0.0, 0.0, 0.0);
    for (index, &(term_weight, depth)) in terms.iter().enumerate() {
        cumulative += share[index];
        weight += term_weight;
        moment += term_weight * depth;
        // The last term never closes a bin, so the trailing bin always holds it and its weight is
        // never zero -- which is what makes the division below safe without a guard.
        let last = index + 1 == terms.len();
        // Each bin still open claims an equal share of the measure *still unspent*, not of the total.
        // The difference is not cosmetic. Against a fixed ladder of `k * total / bins` a single heavy
        // term crosses many boundaries at once, and since one term can close only one bin, the early
        // bins end up holding one term each while the budget runs out before the depths that carry
        // the variation -- measured, 2.1e-2 against 9.0e-4 on the four-band stand-in. Dividing the
        // remainder puts the overshoot back into the bins that are left.
        #[expect(clippy::cast_precision_loss, reason = "bin counts are far below 2^53")]
        let boundary = spent + (total - spent) / (bins - out.len()) as f64;
        if !last && cumulative >= boundary && out.len() + 1 < bins {
            out.push((weight, moment / weight));
            spent = cumulative;
            weight = 0.0;
            moment = 0.0;
        }
    }
    out.push((weight, moment / weight));
    out
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::excessive_precision,
        clippy::unreadable_literal
    )]

    use super::*;

    /// A four-term stand-in for a water band: most of the channel nearly transparent, a little of it
    /// in line cores three orders of magnitude stronger. Any exponential sum with that spread has the
    /// property the module exists for.
    const BAND: [(f64, f64); 4] = [(0.55, 0.004), (0.30, 0.09), (0.12, 1.4), (0.03, 26.0)];

    /// A window: one term, unit weight, zero rate — what the fixture writes for a transparent channel.
    const WINDOW: [(f64, f64); 1] = [(1.0, 0.0)];

    /// Almost all the weight in one strong term, with two whispers in the wings. Transmitted weight
    /// — which is what the resort bins on — is concentrated at the far end of the sorted order here,
    /// so no bin boundary falls before the last term.
    const CORE_HEAVY: [(f64, f64); 3] = [(0.001, 0.0), (0.001, 0.1), (0.998, 5.0)];

    /// The reason the table is a distribution and not a number.
    ///
    /// Take the band's transmittance at one column, back out the single optical depth that would
    /// reproduce it, and ask that grey slab about a different column. It is wrong, and not slightly:
    /// a band saturates in its cores while its wings keep transmitting, so its transmittance falls
    /// far more slowly with column than any exponential can. Fitting the grey depth at `u = 1` and
    /// using it at `u = 8` is a factor-of-several error in the quantity an atmospheric correction
    /// divides by, which is the whole argument for carrying sixteen terms instead of one.
    #[test]
    fn a_band_is_not_an_exponential_in_the_column_and_the_gap_is_not_small() {
        let band = KDistribution::new(&BAND, 1.0).unwrap();
        let grey = -libm::log(band.transmittance(1.0));
        let mut worst: f64 = 0.0;
        for step in 1..=16 {
            let u = f64::from(step) * 0.5;
            let exact = band.transmittance(u);
            let ratio = libm::exp(-grey * u) / exact;
            worst = worst.max(libm::fabs(ratio - 1.0));
        }
        assert!(
            worst > 0.5,
            "a grey depth calibrated at one column tracked the band to {worst:.3} over the rest, \
             which would make the k-distribution pointless"
        );
    }

    /// (G1) The multiplication rule is exactly the product of the two direct transmittances.
    ///
    /// This is what "weights multiply, depths add" *means*, and it is the identity that says the
    /// composition in [`gas_optical_depths`] is arithmetic rather than a model. Asserted at several
    /// columns, because a single column is satisfied by any pair of numbers whose product is right.
    #[test]
    fn the_multiplication_rule_reproduces_the_product_of_the_two_transmittances() {
        for scale in [0.1, 1.0, 3.0, 9.0] {
            let water = KDistribution::new(&BAND, 2.0 * scale).unwrap();
            let oxygen = KDistribution::new(&[(0.7, 0.02), (0.3, 5.0)], 0.9 * scale).unwrap();
            let joint = gas_optical_depths(&[water, oxygen]);
            let summed: f64 = joint
                .iter()
                .map(|&(weight, depth)| weight * libm::exp(-depth))
                .sum();
            let product = water.transmittance(1.0) * oxygen.transmittance(1.0);
            assert!(
                libm::fabs(summed - product) < 1e-15,
                "at scale {scale}: {summed:.17e} against {product:.17e}"
            );
        }
    }

    /// Combining an absorber with a window is exact, not an assumption.
    ///
    /// The independence the multiplication rule needs is vacuous when one of the two gases has a
    /// single term at zero rate: the product is the other distribution term for term, weight for
    /// weight. That is the case 197 of `RadCalNet`'s 211 wavelengths are in for at least one gas, so
    /// it is worth pinning rather than reasoning about.
    #[test]
    fn a_window_gas_leaves_the_other_distribution_exactly_alone() {
        let band = KDistribution::new(&BAND, 1.7).unwrap();
        let window = KDistribution::new(&WINDOW, 4.2).unwrap();
        let alone = gas_optical_depths(&[band]);
        let with_window = gas_optical_depths(&[band, window]);
        assert_eq!(alone, with_window);
        for (index, &(weight, depth)) in alone.iter().enumerate() {
            assert_eq!(weight, BAND[index].0);
            assert_eq!(depth, BAND[index].1 * 1.7);
        }
    }

    /// Zero-weight padding is carried in and dropped out.
    ///
    /// The fixture pads every row to a fixed width, so the terms a caller hands over are `TERMS`
    /// long whatever the fit kept. Those terms must not reach the transport — each one would be a
    /// full Monte-Carlo run contributing exactly nothing — and must not change any sum.
    #[test]
    fn padding_terms_cost_nothing_and_produce_no_transport_run() {
        let padded = [(0.55, 0.004), (0.45, 0.7), (0.0, 0.0), (0.0, 0.0)];
        let trimmed = [(0.55, 0.004), (0.45, 0.7)];
        let with_padding = KDistribution::new(&padded, 1.3).unwrap();
        let without = KDistribution::new(&trimmed, 1.3).unwrap();
        assert_eq!(with_padding.transmittance(1.6), without.transmittance(1.6));
        assert_eq!(
            gas_optical_depths(&[with_padding]),
            gas_optical_depths(&[without])
        );
    }

    /// The binned product drops the padding too.
    ///
    /// The test above asserts this of the exact product and stops there, which left the binned
    /// path's own filter with nothing holding it. The bin count is deliberately above the term
    /// count: below it the padding is invisible in the answer, because a zero weight adds nothing
    /// to either moment and a bin that swallows one reports exactly what it would have without it.
    /// Above it the resort hands the product back untouched, and the padding is the difference
    /// between two terms and four.
    #[test]
    fn the_binned_product_drops_the_padding_as_well() {
        let padded = [(0.55, 0.004), (0.45, 0.7), (0.0, 0.0), (0.0, 0.0)];
        let trimmed = [(0.55, 0.004), (0.45, 0.7)];
        let bins = NonZeroUsize::new(8).unwrap();
        assert_eq!(
            gas_optical_depths_binned(&[KDistribution::new(&padded, 1.3).unwrap()], bins),
            gas_optical_depths_binned(&[KDistribution::new(&trimmed, 1.3).unwrap()], bins)
        );
    }

    /// Every bin the resort returns is a distribution term: a weight above zero, and a finite depth.
    ///
    /// The trailing `moment / weight` has no guard and does not need one — the last term can never
    /// close a bin, so the trailing bin always holds it. That argument is load-bearing and nothing
    /// was checking it. `CORE_HEAVY` is the shape that exercises it: the measure is so concentrated
    /// in the last term that no boundary is crossed until the loop reaches it, which is exactly when
    /// the last-term test is the only thing standing between the caller and `0 / 0`.
    #[test]
    fn every_bin_the_resort_returns_carries_weight() {
        let binned = gas_optical_depths_binned(
            &[KDistribution::new(&CORE_HEAVY, 1.0).unwrap()],
            NonZeroUsize::new(2).unwrap(),
        );
        assert_ne!(binned.len(), 0);
        for (weight, depth) in binned {
            assert!(
                weight > 0.0 && depth.is_finite(),
                "the resort returned a term of weight {weight} at depth {depth}"
            );
        }
    }

    /// Air mass multiplies the column, and the direction is not free.
    ///
    /// Every other call to [`KDistribution::transmittance`] here passes an air mass of 1, or a gas
    /// for which the answer does not depend on it — and at air mass 1 a slant path and a vertical
    /// one are the same number, so nothing said which way the scaling goes. This is the identity
    /// that does: asking a column `u` about a slant path `m` is asking a column `u m` about the
    /// vertical. It needs no tolerance, because multiplying by 1.0 is exact.
    #[test]
    fn the_air_mass_multiplies_the_column_rather_than_dividing_it() {
        for &air_mass in &[0.5, 1.6, 3.65] {
            let slant = KDistribution::new(&BAND, 1.3 * air_mass)
                .unwrap()
                .transmittance(1.0);
            let vertical = KDistribution::new(&BAND, 1.3)
                .unwrap()
                .transmittance(air_mass);
            assert_eq!(
                vertical, slant,
                "air mass {air_mass} is not acting as a column multiplier"
            );
        }
    }

    /// (G2) The band transmittance is its defining exponential sum.
    ///
    /// `T(u) = Σ wⱼ exp(−kⱼ u)` at `u = column × air mass`, and the product of two gases the product
    /// of their sums, against `derivations/gas_k_distribution.mac`'s 40-digit values for the
    /// four-term stand-in and the two-term oxygen band; `derivations/gas_k_distribution.gp`
    /// recomputes them. The same file shows the product's mean depth is the sum of the two means,
    /// `3317/1000` here, which is the first moment the resort is held to.
    #[test]
    fn the_band_transmittance_is_its_defining_sum() {
        let band = KDistribution::new(&BAND, 1.3).unwrap();
        for (got, want) in [
            (
                band.transmittance(1.6),
                8.007503692183774790270023815564327570129e-1,
            ),
            (
                KDistribution::new(&BAND, 1.0).unwrap().transmittance(1.0),
                8.515753853937098157215998250902171928408e-1,
            ),
        ] {
            assert!(
                (got - want).abs() < 4.0 * f64::EPSILON,
                "{got:e} against {want:e}"
            );
        }
        let water = KDistribution::new(&BAND, 2.0).unwrap();
        let oxygen = KDistribution::new(&[(0.7, 0.02), (0.3, 5.0)], 0.9).unwrap();
        let joint = gas_optical_depths(&[water, oxygen]);
        let (summed, mean): (f64, f64) = joint
            .iter()
            .fold((0.0, 0.0), |(t, first), &(weight, depth)| {
                (t + weight * libm::exp(-depth), first + weight * depth)
            });
        let want = 5.55091411363238279191508539011546061539e-1;
        assert!(
            (summed - want).abs() < 4.0 * f64::EPSILON,
            "{summed:e} against {want:e}"
        );
        assert!(
            (mean - 3.317).abs() < 8.0 * f64::EPSILON * 3.317,
            "mean depth {mean}"
        );
    }

    /// No absorber is unit transmittance, and no column is too.
    #[test]
    fn a_transparent_channel_and_a_zero_column_both_transmit_everything() {
        assert_eq!(
            KDistribution::new(&WINDOW, 3.0).unwrap().transmittance(2.0),
            1.0
        );
        assert_eq!(
            KDistribution::new(&BAND, 0.0).unwrap().transmittance(2.0),
            1.0
        );
    }

    /// (G4) The resort keeps the zeroth and first moments and can only lose transmittance.
    ///
    /// A term list that is not a distribution is not a worse fit — it is a channel that gains or
    /// loses photons at zero absorber, which nothing downstream can distinguish from a real band.
    /// The resort exists to make an unaffordable product affordable, so the thing to pin is what it is
    /// allowed to move. Two quantities are exact by construction and asserted as such; the
    /// transmittance is not exact, and Jensen fixes its sign — merging terms of unequal depth can only
    /// ever transmit *less* than the product it replaced. A resort that got the weighted mean wrong
    /// would break the sign before it broke any tolerance.
    #[test]
    fn the_resort_preserves_the_two_moments_it_claims_to() {
        let columns = [0.7, 1.3, 0.4, 2.1];
        let gases: Vec<_> = columns
            .iter()
            .map(|&column| KDistribution::new(&BAND, column).unwrap())
            .collect();
        let exact = gas_optical_depths(&gases);
        assert_eq!(
            exact.len(),
            BAND.len().pow(4),
            "the exact product changed shape"
        );

        let moments = |terms: &[(f64, f64)]| {
            terms
                .iter()
                .fold((0.0, 0.0), |(weight, first), &(w, depth)| {
                    (weight + w, first + w * depth)
                })
        };
        let transmittance = |terms: &[(f64, f64)], u: f64| -> f64 {
            terms
                .iter()
                .map(|&(w, depth)| w * libm::exp(-depth * u))
                .sum()
        };
        let (exact_weight, exact_first) = moments(&exact);

        let mut ladder = Vec::new();
        for bins in [4_usize, 8, 16, 32, 64] {
            let mut worst = 0.0_f64;
            let binned = gas_optical_depths_binned(&gases, NonZeroUsize::new(bins).unwrap());
            assert!(
                binned.len() <= bins,
                "{bins} bins produced {} terms",
                binned.len()
            );

            let (weight, first) = moments(&binned);
            assert!(
                libm::fabs(weight - exact_weight) < 1.0e-12,
                "{bins} bins moved the total weight by {}",
                weight - exact_weight
            );
            assert!(
                libm::fabs(first - exact_first) < 1.0e-12,
                "{bins} bins moved the mean optical depth by {}",
                first - exact_first
            );

            for &u in &[0.05, 0.2, 1.0, 3.0] {
                let (whole, cut) = (transmittance(&exact, u), transmittance(&binned, u));
                assert!(
                    cut <= whole + 1.0e-15,
                    "{bins} bins transmitted {cut} where the exact product transmits {whole}, and \
                     merging by Jensen can only lose transmittance"
                );
                worst = worst.max((whole - cut) / whole);
            }
            ladder.push((bins, worst));
        }

        // The bias is a knob and not a fixed cost, which is the property a caller needs to be able to
        // buy accuracy with transport runs. Measured here, on a deliberately harsh stand-in -- four
        // gases each spanning rate 0.004 to 26.0, over a 60-fold column range:
        //   4 bins 3.66e-1   8 1.82e-1   16 1.06e-1   32 4.59e-2   64 1.12e-2
        // The factor of 30 across the ladder is the claim. The exact numbers move with the fit and
        // are not worth pinning, but a resort whose bias stopped falling would be a broken one.
        //
        // This stand-in is the case the shipped binning rule is *worst* on, deliberately: it asks for
        // the worst error over u in [0.05, 3.0], and at u = 0.05 nothing is saturated, so weighting
        // the bins by exp(-tau) buys nothing and costs resolution. Equal-weight bins reach 9.0e-4
        // here. They also reach 1.0e-1 on the shipped table, against this rule's 1.9e-2, because a
        // real strong band *is* half saturated. Keeping the harder number visible is the point: the
        // rule is chosen for the regime the table is used in, not because it wins everywhere.
        let (coarsest, finest) = (ladder[0].1, ladder[ladder.len() - 1].1);
        assert!(
            coarsest > finest * 30.0,
            "16 times the transport runs bought only {coarsest:.3e} -> {finest:.3e}, so the bin \
             count is not buying accuracy: {ladder:?}"
        );
        assert!(
            finest < 2.0e-2,
            "the resort's bias at 64 bins is now {finest:.3e} on the four-band stand-in"
        );
    }

    /// Asking for more bins than the product has terms must hand back the product, not a re-derived
    /// approximation of it — otherwise a caller pays the bias on the channels that never needed it.
    #[test]
    fn enough_bins_is_the_exact_product_untouched() {
        let gases = [
            KDistribution::new(&BAND, 1.0).unwrap(),
            KDistribution::new(&WINDOW, 1.0).unwrap(),
        ];
        let exact = gas_optical_depths(&gases);
        let binned = gas_optical_depths_binned(&gases, NonZeroUsize::new(64).unwrap());
        assert_eq!(exact, binned);
    }

    #[test]
    fn a_term_list_that_is_not_a_distribution_is_refused() {
        assert_eq!(KDistribution::new(&[], 1.0).err(), Some(GasError::NoTerms));
        assert_eq!(
            KDistribution::new(&[(0.5, 1.0), (0.4, 2.0)], 1.0).err(),
            Some(GasError::WeightsDoNotSumToOne)
        );
        assert_eq!(
            KDistribution::new(&[(1.5, 1.0), (-0.5, 2.0)], 1.0).err(),
            Some(GasError::NotADistribution)
        );
        assert_eq!(
            KDistribution::new(&[(1.0, -1.0)], 1.0).err(),
            Some(GasError::NotADistribution)
        );
        assert_eq!(
            KDistribution::new(&WINDOW, -1.0).err(),
            Some(GasError::ColumnNotPhysical)
        );
        assert_eq!(
            KDistribution::new(&WINDOW, f64::NAN).err(),
            Some(GasError::ColumnNotPhysical)
        );
    }

    /// Nothing at all is one unit-weight term at zero depth, so a caller need not special-case it.
    #[test]
    fn no_gases_is_one_transparent_term() {
        assert_eq!(gas_optical_depths(&[]), Vec::from([(1.0, 0.0)]));
    }
}
