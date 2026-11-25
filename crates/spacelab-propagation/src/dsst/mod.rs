//! DSST (Draper Semi-analytical Satellite Theory) — central-body **zonal** mean-element rates.
//!
//! A faithful port of Orekit's `DSSTZonal` mean-element-rate path (J2..Jₙ), validated against Orekit
//! 13 (`tools/orekit-oracle`, `orekit_dsst_zonal.csv`). Scope is the analytic **mean-element rates**
//! `dy/dt = [da, dk, dh, dq, dp, dλ]` and their semi-analytical propagation; the DSST **short-period
//! corrections** are deliberately out of scope (a follow-on).
//!
//! The pieces, mirroring Orekit's class split:
//! * [`auxiliary::DsstAuxiliary`] — equinoctial auxiliary elements `(a, k, h, q, p, λ)` and the
//!   derived gravity-context factors (A, B, C, χ, the direction cosines α/β/γ, …);
//! * `hansen::HansenZonal` — the zonal Hansen kernels K₀^{-n-1,s} and their χ-derivatives;
//! * `coefficients` — Vₙₛ, Qₙₛ, GₛHₛ, the `UpperBounds` D/R functions and the eccentricity-power
//!   truncation `maxEccPowMeanElements`;
//! * [`zonal::DsstZonal`] — the potential derivatives (Orekit's `UAnddU`), the six mean-element
//!   rates and the DOP853 mean propagation.
//!
//! # Frame convention
//!
//! Built as Orekit's `new DSSTZonal(provider)` (null central-body frame): the pole is the orbit
//! frame's +Z, so the direction cosines α/β/γ are pure functions of `(p, q)` — no frame transform.
//! For an inertial orbit frame this is J-about-inertial-Z, the standard mean-element convention.

// Faithful Orekit port: single-letter math names (A, B, C, χ, k, h, p, q, s, n), Orekit-mirrored
// helper names, dense recurrences and full-precision golden fixtures. These allows keep the port
// auditable against the Java rather than renamed to satisfy lints.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::struct_field_names,
    clippy::doc_markdown,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    // Danielson's `(a + b) / 2` averages are ported verbatim; `f64::midpoint` would change the
    // rounding away from Orekit's exact form (the values are bounded, so overflow is impossible).
    clippy::manual_midpoint
)]

pub mod auxiliary;
pub(crate) mod coefficients;
pub mod drag;
pub mod gaussian;
pub mod gaussian_short_period;
pub(crate) mod hansen;
pub mod propagator;
pub mod short_period;
pub mod srp;
pub mod tesseral;
pub mod tesseral_short_period;
pub mod third_body;
pub mod third_body_short_period;
pub mod zonal;

pub use auxiliary::DsstAuxiliary;
pub use drag::DsstAtmosphericDrag;
pub use propagator::{
    DsstMeanPropagator, MEAN_FROM_OSCULATING_PASSES, MEAN_FROM_OSCULATING_TOLERANCE,
};
pub use srp::DsstSolarRadiationPressure;
pub use tesseral::DsstTesseral;
pub use third_body::DsstThirdBody;
pub use zonal::DsstZonal;

use crate::driver::IntegrationError;

/// Why a DSST computation refused its input or could not finish.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum DsstError {
    /// A mean equinoctial state `[a, k, h, q, p, λ]` that is not a finite ellipse: a component is
    /// not finite, `a ≤ 0`, or `e = √(k² + h²) ≥ 1`.
    MeanState {
        /// The state as given.
        state: [f64; 6],
    },
    /// A span that is not finite or is negative: the mean-element integration runs forward only.
    Span {
        /// The span as given, seconds.
        dt: f64,
    },
    /// The equinoctial Kepler equation `λ = F − k·sin F + h·cos F` gave no eccentric longitude `F`:
    /// `e = √(k² + h²)` is not below 1, or the Halley iteration did not converge within its 50
    /// steps.
    Kepler {
        /// `k = e·cos ϖ`.
        k: f64,
        /// `h = e·sin ϖ`.
        h: f64,
        /// The mean longitude `λ`, radians.
        lambda: f64,
    },
    /// The mean-element integration failed.
    Integration(IntegrationError),
    /// A Gauss–Legendre order that is not one of [`gaussian::GAUSS_ORDER`].
    GaussOrder {
        /// The order asked for.
        order: usize,
    },
    /// A zonal field without J2: `Cₙ₀` needs three entries (degrees 0, 1 and 2), `Jₙ` one.
    ZonalField {
        /// How many coefficients were given.
        len: usize,
    },
    /// A tesseral field whose `Cₙₘ` and `Sₙₘ` tables do not reach degree 2, differ in degree, or
    /// have a row `n ≥ 2` shorter than `n + 1`.
    TesseralField {
        /// The first degree the two tables do not hold.
        degree: usize,
    },
    /// A third-body truncation past the precomputed tables: `max_ar3_pow` must be below 22, and
    /// `max_ecc_pow` at most `max_ar3_pow`.
    Truncation {
        /// The power of `a/R₃` asked for.
        max_ar3_pow: usize,
        /// The power of `e` asked for.
        max_ecc_pow: usize,
    },
    /// [`DsstMeanPropagator::mean_from_osculating`] did not converge within
    /// [`MEAN_FROM_OSCULATING_PASSES`] passes.
    MeanFromOsculating {
        /// The last pass's largest correction (relative in `a`, absolute elsewhere); NaN when a
        /// correction was not finite.
        step: f64,
    },
}

impl core::fmt::Display for DsstError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MeanState { state } => write!(
                f,
                "the mean equinoctial state {state:?} is not a finite ellipse: a > 0 and e < 1"
            ),
            Self::Span { dt } => write!(
                f,
                "the span {dt} s is not finite and non-negative: DSST propagates forward only"
            ),
            Self::Kepler { k, h, lambda } => write!(
                f,
                "no eccentric longitude for k = {k}, h = {h}, λ = {lambda}: e is not below 1, or \
                 the equinoctial Kepler equation did not converge"
            ),
            Self::Integration(e) => write!(f, "the mean-element integration failed: {e:?}"),
            Self::GaussOrder { order } => write!(
                f,
                "Gauss order {order} is not one of {:?}",
                gaussian::GAUSS_ORDER
            ),
            Self::ZonalField { len } => {
                write!(
                    f,
                    "a zonal field needs J2, and {len} coefficients do not reach it"
                )
            }
            Self::TesseralField { degree } => write!(
                f,
                "the tesseral Cnm and Snm tables do not hold degree {degree}"
            ),
            Self::Truncation {
                max_ar3_pow,
                max_ecc_pow,
            } => write!(
                f,
                "the third-body truncation ({max_ar3_pow}, {max_ecc_pow}) is past the tables: the \
                 a/R3 power below 22, the e power at most the a/R3 power"
            ),
            Self::MeanFromOsculating { step } => write!(
                f,
                "the mean state did not converge: after {MEAN_FROM_OSCULATING_PASSES} passes the \
                 last correction was {step:e}, above {MEAN_FROM_OSCULATING_TOLERANCE:e}"
            ),
        }
    }
}

impl core::error::Error for DsstError {}

/// `state` if it is a mean equinoctial ellipse, else [`DsstError::MeanState`].
pub(crate) fn mean_state(state: [f64; 6]) -> Result<[f64; 6], DsstError> {
    let [a, k, h, ..] = state;
    if state.iter().all(|x| x.is_finite()) && a > 0.0 && k * k + h * h < 1.0 {
        Ok(state)
    } else {
        Err(DsstError::MeanState { state })
    }
}

/// `dt` if it is a finite forward span, else [`DsstError::Span`].
pub(crate) fn span(dt: f64) -> Result<f64, DsstError> {
    if dt.is_finite() && dt >= 0.0 {
        Ok(dt)
    } else {
        Err(DsstError::Span { dt })
    }
}

/// Prints one row of `docs/comparisons.md` for `tools/compare.py`: the `n` distances to Orekit
/// 13.1.7 (`tools/orekit-oracle`) of one quantity, their largest and their root mean square. A NaN
/// distance is the largest.
#[cfg(test)]
pub(crate) fn comparison_row(quantity: &str, unit: &str, distances: &[f64]) {
    let max = distances
        .iter()
        .copied()
        .fold(0.0, spacelab_testkit::nan_max);
    let rms = libm::sqrt(distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64);
    std::println!(
        "comparison | Orekit | 13.1.7 | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`, orekit_dsst_zonal.csv): zonal field to
    // degree 6, MEAN equinoctial state, direct getMeanElementRate and a MEAN DSSTPropagator run.
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const C20: f64 = -1.08262689216735860e-03;
    const C30: f64 = 2.53209950653743350e-06;
    const C40: f64 = 1.62004577259847830e-06;
    const C50: f64 = 2.27515624171733840e-07;
    const C60: f64 = -5.41217294926390600e-07;

    const MAX_ECC_POW: usize = 2;

    // Initial MEAN equinoctial state (a, ex=k, ey=h, hx=q, hy=p, lM=λ).
    const A0: f64 = 7.20000000000000000e+06;
    const EX0: f64 = 3.09016994374947470e-03;
    const EY0: f64 = 9.51056516295153600e-03;
    const HX0: f64 = 8.54890329218965000e-01;
    const HY0: f64 = 7.69746709859868000e-01;
    const LM0: f64 = 1.43116998663535020e+00;
    const INIT: [f64; 6] = [A0, EX0, EY0, HX0, HY0, LM0];

    // getMeanElementRate at the initial state: [da, dk, dh, dq, dp, dM].
    const RATES: [f64; 6] = [
        0.00000000000000000e+00,
        3.42263402763264900e-09,
        -1.70976565822581990e-09,
        -1.40912873324375400e-07,
        1.56498213547341530e-07,
        -1.03110702784141430e-06,
    ];

    // MEAN propagation samples: (t_s, [a, ex, ey, hx, hy, lM]).
    const SAMPLES: [(f64, [f64; 6]); 5] = [
        (0.0, INIT),
        (
            21600.0,
            [
                7.20000000000000000e+06,
                3.16395301514028070e-03,
                9.47328593348677900e-03,
                8.51839935608425400e-01,
                7.73121044973265600e-01,
                2.37304247212616130e+01,
            ],
        ),
        (
            43200.0,
            [
                7.20000000000000000e+06,
                3.23744043048822130e-03,
                9.43531196934242000e-03,
                8.48776222732037000e-01,
                7.76483292346584100e-01,
                4.60296794537376600e+01,
            ],
        ),
        (
            64800.0,
            [
                7.20000000000000000e+06,
                3.31062619455574970e-03,
                9.39664608263022400e-03,
                8.45699238487331300e-01,
                7.79833399409175500e-01,
                6.83289341840520300e+01,
            ],
        ),
        (
            86400.0,
            [
                7.20000000000000000e+06,
                3.38350433649739800e-03,
                9.35729113913850800e-03,
                8.42609030979382800e-01,
                7.83171313780110800e-01,
                9.06281889121936600e+01,
            ],
        ),
    ];

    fn zonal() -> DsstZonal {
        DsstZonal::new(MU, AE, std::vec![0.0, 0.0, C20, C30, C40, C50, C60]).unwrap()
    }

    /// The two gates every entry point passes first, at their edges: `a` must be positive and not
    /// merely non-negative, `e = 1` is not an ellipse while `e = 1 − 1e-12` is, and one non-finite
    /// component anywhere refuses the state; a span may be zero, of either sign, but not below it.
    #[test]
    fn the_gates_are_a_finite_ellipse_and_a_forward_span() {
        let ok = [7.0e6, 0.1, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(mean_state(ok), Ok(ok));
        let mut tiny = ok;
        tiny[0] = f64::MIN_POSITIVE;
        assert_eq!(mean_state(tiny), Ok(tiny));
        let mut grazing = ok;
        grazing[1] = 1.0 - 1e-12;
        grazing[2] = 0.0;
        assert_eq!(mean_state(grazing), Ok(grazing));
        let mut refused = std::vec::Vec::new();
        for (i, bad) in [(0, 0.0), (0, -7.0e6), (1, 1.0), (2, -1.0)] {
            let mut s = [7.0e6, 0.0, 0.0, 0.3, 0.4, 0.5];
            s[i] = bad;
            refused.push(s);
        }
        for i in 0..6 {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut s = ok;
                s[i] = bad;
                refused.push(s);
            }
        }
        for s in refused {
            assert!(
                matches!(mean_state(s), Err(DsstError::MeanState { .. })),
                "{s:?} passed"
            );
        }

        for dt in [0.0, -0.0, f64::MIN_POSITIVE, 3600.0, f64::MAX] {
            assert_eq!(span(dt).map(f64::to_bits), Ok(dt.to_bits()));
        }
        for dt in [-1e-300, -3600.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(span(dt), Err(DsstError::Span { .. })),
                "{dt} passed"
            );
        }
    }

    /// A refusal says what it refused, in the numbers it was given: the message is what reaches a C
    /// caller through `sl_last_error` and a Python caller as the `ValueError`.
    #[test]
    fn a_refusal_names_what_it_refused() {
        let cases = [
            (
                DsstError::MeanState {
                    state: [-7.5e6, 0.0, 0.0, 0.0, 0.0, 0.0],
                },
                ["-7500000", "not a finite ellipse"],
            ),
            (DsstError::Span { dt: -60.5 }, ["-60.5", "forward only"]),
            (
                DsstError::Kepler {
                    k: 1.25,
                    h: 0.5,
                    lambda: 3.5,
                },
                ["k = 1.25, h = 0.5, λ = 3.5", "e is not below 1"],
            ),
            (
                DsstError::Integration(IntegrationError::StepLimit { steps: 2e5 }),
                ["integration failed", "StepLimit"],
            ),
            (
                DsstError::GaussOrder { order: 13 },
                ["order 13", "[12, 16, 20, 24, 32, 40, 48]"],
            ),
            (
                DsstError::ZonalField { len: 2 },
                ["needs J2", "2 coefficients"],
            ),
            (
                DsstError::TesseralField { degree: 4 },
                ["Cnm and Snm", "degree 4"],
            ),
            (
                DsstError::Truncation {
                    max_ar3_pow: 30,
                    max_ecc_pow: 2,
                },
                ["(30, 2)", "below 22"],
            ),
            (
                DsstError::MeanFromOsculating { step: 1.5e-3 },
                ["did not converge", "1.5e-3"],
            ),
        ];
        for (e, says) in cases {
            let message = std::format!("{e}");
            for part in says {
                assert!(message.contains(part), "{message:?} lacks {part:?}");
            }
        }
    }

    /// Reduce an angle difference to (−π, π] so an accumulated mean longitude compares wrap-safe.
    fn wrap(d: f64) -> f64 {
        use core::f64::consts::TAU;
        d - TAU * libm::round(d / TAU)
    }

    /// `from_j` is `new` with the sign convention `Cₙ₀ = −Jₙ` and the degree offset applied, and it
    /// is the constructor the Python bridge calls. Nothing in Rust reached it, so its index
    /// arithmetic and its sign were both unasserted: `c_n0[i + 2] = −jₙ` survived becoming
    /// `c_n0[i * 2]`, which files `J3` where `J2` belongs, and `c_n0[i + 2] = +jₙ`, which inverts
    /// the whole field. Both show up as different rates, so the two constructors are compared
    /// through the answer rather than through the coefficient vector.
    #[test]
    #[allow(clippy::float_cmp)] // both paths must produce bit-identical rates
    fn from_j_is_new_with_the_sign_convention() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let j = [-C20, -C30, -C40, -C50, -C60];
        let from_j = DsstZonal::from_j(MU, AE, &j)
            .unwrap()
            .mean_element_rates(&aux);
        let explicit = zonal().mean_element_rates(&aux);
        assert_eq!(from_j, explicit, "from_j must reproduce the explicit Cₙ₀");
        // A J2-only field is not the degree-6 one: without this the comparison above would pass on
        // a `from_j` that dropped every coefficient past the first.
        let j2_only = DsstZonal::from_j(MU, AE, &j[..1])
            .unwrap()
            .mean_element_rates(&aux);
        assert_ne!(
            j2_only, explicit,
            "J2 alone cannot equal the degree-6 field"
        );
        assert_eq!(DsstZonal::from_j(MU, AE, &j).unwrap().max_degree(), 6);
        assert_eq!(DsstZonal::from_j(MU, AE, &j[..1]).unwrap().max_degree(), 2);
    }

    /// The distance to Orekit's `maxEccPowMeanElements` at the initial state (2 here): the series cut
    /// is Orekit's heuristic, which no mathematics fixes; what it drops is measured against the
    /// orbit average in `tests/dsst_short_period_identity.rs`.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn truncation_order_distance_to_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let ours = zonal().max_ecc_pow(&aux);
        comparison_row(
            "DSST zonal eccentricity cut of the mean rates",
            "powers",
            &[ours.abs_diff(MAX_ECC_POW) as f64],
        );
    }

    /// The distance to Orekit's `getMeanElementRate` at a fixed state, relative, absolute where
    /// Orekit's is zero (`da`). `tests/dsst_short_period_identity.rs` decides: the rates are the
    /// orbit average of Gauss's rates.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn fixed_state_rates_distance_to_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let got = zonal().mean_element_rates(&aux);
        let distances: std::vec::Vec<f64> = (0..6)
            .map(|i| {
                if RATES[i] == 0.0 {
                    got[i].abs()
                } else {
                    (got[i] - RATES[i]).abs() / RATES[i].abs()
                }
            })
            .collect();
        comparison_row(
            "DSST zonal mean element rates at a fixed state",
            "relative",
            &distances,
        );
    }

    /// The distance to Orekit's MEAN `DSSTPropagator` at every sample. The propagator's own test
    /// decides: the propagated elements solve the ODE of the summed mean rates (`propagator.rs`).
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn mean_propagation_distance_to_orekit() {
        let z = zonal();
        let (mut a, mut ehpq, mut along) = (
            std::vec::Vec::new(),
            std::vec::Vec::new(),
            std::vec::Vec::new(),
        );
        for (t, want) in SAMPLES {
            let got = z.propagate_mean(INIT, t).unwrap();
            a.push((got[0] - want[0]).abs() / want[0].abs());
            ehpq.extend((1..=4).map(|i| (got[i] - want[i]).abs()));
            along.push(want[0] * wrap(got[5] - want[5]).abs());
        }
        comparison_row("DSST zonal mean propagation, a", "relative", &a);
        comparison_row("DSST zonal mean propagation, k, h, q, p", "1", &ehpq);
        comparison_row("DSST zonal mean propagation, along-track", "m", &along);
    }

    /// Orekit's DSSTPropagator run in `PropagationType.OSCULATING` (`orekit_dsst_zonal_sp.csv`), for
    /// the distance below. Orekit returns the raw mean state at the initial epoch, so `t = 0` is
    /// excluded — our osculating applies the short period there too. The short periods are decided
    /// by their defining equation in `tests/dsst_short_period_identity.rs`.
    #[rustfmt::skip]
    const OSC_GOLDEN: [(f64, [f64; 6]); 4] = [
        (10800.0, [7.20080406051416800e+06, 2.62505970455380930e-03, 9.07352971461781100e-03, 8.53294337752836500e-01, 7.71500600121665100e-01, 1.25799728461532960e+01]),
        (21600.0, [7.19653519386523100e+06, 3.37069108060647200e-03, 1.05265141886316960e-02, 8.51934578177427900e-01, 7.73077799984410100e-01, 2.37311878101715340e+01]),
        (32400.0, [7.20618577009010100e+06, 2.99398570992457000e-03, 9.69571419768334100e-03, 8.50206466868549800e-01, 7.74813287148281200e-01, 3.48794458105977440e+01]),
        (43200.0, [7.19190590428175800e+06, 3.77434602397508840e-03, 7.98874607136180700e-03, 8.48872684251018700e-01, 7.76510924445086500e-01, 4.60300713996471700e+01]),
    ];

    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn osculating_short_period_distance_to_orekit() {
        let z = zonal();
        let (mut da, mut rest) = (std::vec::Vec::new(), std::vec::Vec::new());
        for (t, g) in OSC_GOLDEN {
            let o = z.propagate_osculating(INIT, t).unwrap();
            da.push((o[0] - g[0]).abs());
            rest.extend((1..6).map(|i| (o[i] - g[i]).abs()));
        }
        comparison_row("DSST zonal osculating a", "m", &da);
        comparison_row("DSST zonal osculating k, h, q, p, λ", "1", &rest);
    }
}
