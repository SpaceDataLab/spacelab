//! How wrong SDTM-1 is, as a function of the regime it is being asked about.
//!
//! A density model that reports only a density is unusable in a covariance: the drag acceleration
//! carries the density's error straight into the state, and a conjunction probability computed from a
//! point density is a number without a meaning. What this supplies is the other half — the spread of
//! `ln(ρ_model/ρ_measured)` measured on a **held-out** split, tabulated by the two drivers that decide
//! it, and turned into a multiplicative interval.
//!
//! Three choices make it honest, and each was measured rather than assumed (`docs/verification.md`
//! §10.12):
//!
//! * **the spread comes from the calendar-month split, not the satellite-month one.** The leaky split
//!   (§10.10) gives a σ 1.6 to 2.3 times smaller in the solar-minimum cell, and an interval built from
//!   it covers only part of what it claims when the month is genuinely new. An interval that does not
//!   cover is worse than no interval;
//! * **the spread is taken about zero, not about each cell's mean.** A caller handed an interval wants
//!   it to contain the measurement, so a cell whose residual is biased must carry that bias as part of
//!   its width;
//! * **it is a function of the model's own inputs.** The regime is decided by the 81-day mean flux and
//!   by the storm kernel's response — both already in hand to evaluate the model at all, so asking for
//!   the uncertainty costs a caller nothing it has not already fetched.
//!
//! The distribution is log-normal because the fit's own residual is a log residual and is symmetric in
//! it: the interval is `ρ · exp(±z σ)`, multiplicative, and never negative. That is a property of the
//! parameterisation and not a fitted claim about the tails — [`DensityUncertainty::coverage`] is what
//! measures whether the claim holds, and `tests/measured_density.rs` runs it against measured density.
//!
//! Gate: `derivations/sdtm_uncertainty.mac` derives the coverage `P(|u| ≤ zσ) = erf(z/√2)` of the
//! log-symmetric interval and holds each quantile literal to half an ulp of `√2·erf⁻¹(p)` at 60
//! digits, and `derivations/sdtm_uncertainty.gp` recomputes the quantiles at 100 digits by
//! root-finding `erfc`. The module tests `the_normal_quantiles_are_the_ones_the_cas_printed` (G3) and
//! `the_interval_is_log_symmetric_and_monotone_in_the_quantile` (G1) tie the code to them.

use super::SdtmDrivers;

/// Upper edges of the flux regimes, in units of the 81-day mean F10.7.
///
/// Three cells, chosen to split the corpus of §10.10 into comparable thirds rather than to match the
/// literature's `F10.7 ≥ 175` bin, which that corpus does not reach: its 81-day mean spans 69.8 to
/// 165.7.
pub const FLUX_EDGES: [f64; 2] = [100.0, 150.0];

/// Upper edges of the storm regimes, in units of the kernel's `ap` response — the literature's
/// quiet/active boundary at `Ap` 15.
pub const STORM_EDGES: [f64; 1] = [15.0];

/// Number of regimes: `(FLUX_EDGES + 1) × (STORM_EDGES + 1)`.
pub const REGIMES: usize = (FLUX_EDGES.len() + 1) * (STORM_EDGES.len() + 1);

/// Two-sided standard-normal 90 % quantile, `√2 · erf⁻¹(0.9)`.
///
/// Maxima at 60 digits — `fpprec:60$ bfloat(sqrt(2)*inverse_erf(9/10));`, cut here to 30 —
/// `1.64485362695147271486384890799`, whose nearest `f64` this is, 7.9e-17 (0.355 ulp) away.
pub const NORMAL_90: f64 = 1.644_853_626_951_472_7;

/// Two-sided standard-normal 95 % quantile, `√2 · erf⁻¹(0.95)`.
///
/// Maxima, same form: `1.95996398454005423552459443052`, 0.162 ulp from this `f64`.
pub const NORMAL_95: f64 = 1.959_963_984_540_054_3;

/// Two-sided standard-normal 99 % quantile, `√2 · erf⁻¹(0.99)`.
///
/// Maxima, same form: `2.57582930354890076097857674860`, 0.151 ulp from this `f64`.
pub const NORMAL_99: f64 = 2.575_829_303_548_901;

/// The spread of `ln(ρ_model/ρ_measured)` by regime.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DensityUncertainty {
    log_sigma: [f64; REGIMES],
}

impl DensityUncertainty {
    /// Wrap a measured table. Returns `None` unless every entry is finite and strictly positive: a
    /// zero σ is a claim that the model is exact in that regime, which no measurement can make.
    #[must_use]
    pub fn from_log_sigma(log_sigma: [f64; REGIMES]) -> Option<Self> {
        log_sigma
            .iter()
            .all(|s| s.is_finite() && *s > 0.0)
            .then_some(Self { log_sigma })
    }

    /// The same, unchecked, for the generated table — see [`super::Thermosphere::from_fitted`].
    pub(crate) const fn from_fitted(log_sigma: [f64; REGIMES]) -> Self {
        Self { log_sigma }
    }

    /// Which regime a set of drivers falls in, `flux × (STORM_EDGES + 1) + storm`.
    #[must_use]
    pub fn regime(drivers: &SdtmDrivers) -> usize {
        let bin = |value: f64, edges: &[f64]| edges.iter().filter(|e| value >= **e).count();
        bin(drivers.f107a, &FLUX_EDGES) * (STORM_EDGES.len() + 1) + bin(drivers.storm, &STORM_EDGES)
    }

    /// Standard deviation of `ln(ρ_model/ρ_measured)` in that regime.
    #[must_use]
    pub fn log_sigma(&self, drivers: &SdtmDrivers) -> f64 {
        self.log_sigma[Self::regime(drivers)]
    }

    /// The table itself, regime-major.
    #[must_use]
    pub const fn table(&self) -> &[f64; REGIMES] {
        &self.log_sigma
    }

    /// A multiplicative interval on a density: `(ρ·e^{−zσ}, ρ·e^{+zσ})`.
    ///
    /// `z` is the standard-normal quantile the caller wants — [`NORMAL_90`], [`NORMAL_95`] and
    /// [`NORMAL_99`] are the usual three. Asymmetric in `ρ` and symmetric in its logarithm, which is
    /// the shape the residual actually has; an additive `±σρ` interval would reach below zero at the
    /// widths this model needs in the solar-minimum cell, where `σ` is 0.78.
    #[must_use]
    pub fn interval(&self, density: f64, drivers: &SdtmDrivers, z: f64) -> (f64, f64) {
        let half = z * self.log_sigma(drivers);
        (density * libm::exp(-half), density * libm::exp(half))
    }

    /// The fraction of a sample of `(modelled, measured)` pairs whose interval at `z` contains the
    /// measurement — the number that says whether the table above is a claim or a decoration.
    ///
    /// Returns `None` for an empty sample, and skips no pair: a non-finite or non-positive density is
    /// counted as *not* covered, because a caller would have been handed one.
    #[must_use]
    pub fn coverage<'a>(
        &self,
        sample: impl IntoIterator<Item = &'a (f64, f64, SdtmDrivers)>,
        z: f64,
    ) -> Option<f64> {
        // Counted in `f64` rather than cast from `usize` at the end: both counts are exact integers
        // far below 2^53, and the alternative is a lossy-cast the lint is right to flag.
        let (mut inside, mut n) = (0.0_f64, 0.0_f64);
        for (modelled, measured, drivers) in sample {
            let (low, high) = self.interval(*modelled, drivers, z);
            n += 1.0;
            if *measured >= low && *measured <= high {
                inside += 1.0;
            }
        }
        (n > 0.0).then(|| inside / n)
    }
}

#[cfg(test)]
mod tests {
    // `excessive_precision`: the comparison *is* that the constant is what the CAS printed, rounded by
    // the compiler. Trimming the literal to 17 digits would make the assertion compare a constant with
    // itself as written rather than with the value it claims to be.
    #![allow(clippy::expect_used, clippy::float_cmp, clippy::excessive_precision)]

    use super::*;

    fn drivers(f107a: f64, storm: f64) -> SdtmDrivers {
        SdtmDrivers {
            f107: f107a,
            f107a,
            storm,
        }
    }

    /// Every regime is reachable and each boundary belongs to the cell above it.
    ///
    /// An off-by-one here is silent: it reports a neighbouring regime's σ, which is the right order of
    /// magnitude and the wrong number.
    #[test]
    fn the_regime_edges_are_closed_below_and_cover_every_cell() {
        let seen: Vec<usize> = [69.8, 100.0, 149.9, 150.0, 400.0]
            .iter()
            .flat_map(|f| [0.0, 14.999, 15.0, 400.0].iter().map(move |s| (*f, *s)))
            .map(|(f, s)| DensityUncertainty::regime(&drivers(f, s)))
            .collect();
        assert_eq!(DensityUncertainty::regime(&drivers(99.999, 0.0)), 0);
        assert_eq!(DensityUncertainty::regime(&drivers(99.999, 15.0)), 1);
        assert_eq!(DensityUncertainty::regime(&drivers(100.0, 0.0)), 2);
        assert_eq!(DensityUncertainty::regime(&drivers(150.0, 1e9)), 5);
        for cell in 0..REGIMES {
            assert!(seen.contains(&cell), "regime {cell} is unreachable");
        }
        assert!(
            seen.iter().all(|c| *c < REGIMES),
            "a regime indexes past the table"
        );
    }

    /// (G1) The interval is multiplicative, centred on the density in the logarithm, and widens with
    /// `z`: the geometric mean of its ends is `ρ` and `ln(high/ρ) = zσ`, the identities
    /// `derivations/sdtm_uncertainty.mac` proves.
    #[test]
    fn the_interval_is_log_symmetric_and_monotone_in_the_quantile() {
        let u = DensityUncertainty::from_log_sigma([0.78, 0.60, 0.32, 0.36, 0.31, 0.42])
            .expect("a usable table");
        let d = drivers(120.0, 3.0);
        let rho = 4.0e-12;
        let (low, high) = u.interval(rho, &d, NORMAL_90);
        assert!(low > 0.0 && low < rho && high > rho);
        // Symmetric in the logarithm: the geometric mean of the two ends is the density itself.
        assert!(((low * high).sqrt() - rho).abs() < 1e-15 * rho);
        for z in [NORMAL_95, NORMAL_99] {
            let (l, h) = u.interval(rho, &d, z);
            assert!(l < low && h > high, "z = {z} did not widen the interval");
        }
        // And the width is the regime's σ, not some other cell's.
        assert!((libm::log(high / rho) - NORMAL_90 * 0.32).abs() < 1e-15);
    }

    /// The constructor refuses a table that claims the model is exact, or that carries a `NaN`.
    #[test]
    fn the_constructor_refuses_a_table_that_is_not_a_spread() {
        assert!(DensityUncertainty::from_log_sigma([0.3; REGIMES]).is_some());
        let mut zero = [0.3; REGIMES];
        zero[4] = 0.0;
        assert!(DensityUncertainty::from_log_sigma(zero).is_none());
        let mut negative = [0.3; REGIMES];
        negative[0] = -0.3;
        assert!(DensityUncertainty::from_log_sigma(negative).is_none());
        let mut nan = [0.3; REGIMES];
        nan[5] = f64::NAN;
        assert!(DensityUncertainty::from_log_sigma(nan).is_none());
    }

    /// Coverage counts what it says it counts, including the pair a caller would have been handed and
    /// could not use.
    #[test]
    fn coverage_is_the_fraction_of_pairs_the_interval_contains() {
        let u = DensityUncertainty::from_log_sigma([0.3; REGIMES]).expect("usable");
        let d = drivers(120.0, 3.0);
        let inside = 1.0e-12;
        let (low, high) = u.interval(inside, &d, NORMAL_90);
        let sample = vec![
            (inside, inside, d),       // dead centre
            (inside, high * 0.999, d), // just inside the upper end
            (inside, high * 1.001, d), // just outside it
            (inside, low * 0.999, d),  // and below the lower
            (inside, f64::NAN, d),     // not covered, and not skipped
        ];
        let covered = u.coverage(&sample, NORMAL_90).expect("a non-empty sample");
        assert!((covered - 2.0 / 5.0).abs() < 1e-15, "covered {covered}");
        assert!(u.coverage(&[], NORMAL_90).is_none());
    }

    /// (G3) The three quantiles against the CAS's own digits, which is what the doc comments claim:
    /// `derivations/sdtm_uncertainty.mac` (60 digits) and `.gp` (100 digits) put each literal within
    /// half an ulp of `√2·erf⁻¹(p)`, so it is the nearest `f64`.
    #[test]
    fn the_normal_quantiles_are_the_ones_the_cas_printed() {
        // fpprec:60$ bfloat(sqrt(2)*inverse_erf(9/10)); and the same at 95 % and 99 %, cut to 30.
        assert_eq!(NORMAL_90, 1.644_853_626_951_472_714_863_848_907_99_f64);
        assert_eq!(NORMAL_95, 1.959_963_984_540_054_235_524_594_430_52_f64);
        assert_eq!(NORMAL_99, 2.575_829_303_548_900_760_978_576_748_60_f64);
        // And each is the point where the Gaussian CDF reaches its coverage, to `erf`'s own accuracy.
        for (z, p) in [(NORMAL_90, 0.90), (NORMAL_95, 0.95), (NORMAL_99, 0.99)] {
            let covered = libm::erf(z / core::f64::consts::SQRT_2);
            assert!(
                (covered - p).abs() < 1e-15,
                "z = {z} covers {covered}, not {p}"
            );
        }
    }
}
