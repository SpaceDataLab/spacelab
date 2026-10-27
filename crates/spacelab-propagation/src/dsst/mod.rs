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
//! * [`hansen::HansenZonal`] — the zonal Hansen kernels K₀^{-n-1,s} and their χ-derivatives;
//! * [`coefficients`] — Vₙₛ, Qₙₛ, GₛHₛ, the `UpperBounds` D/R functions and the eccentricity-power
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
pub mod coefficients;
pub mod drag;
pub mod gaussian;
pub mod gaussian_short_period;
pub mod hansen;
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
    MeanFromOsculatingError,
};
pub use srp::DsstSolarRadiationPressure;
pub use tesseral::DsstTesseral;
pub use third_body::DsstThirdBody;
pub use zonal::DsstZonal;

#[cfg(test)]
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
        DsstZonal::new(MU, AE, std::vec![0.0, 0.0, C20, C30, C40, C50, C60])
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
        let from_j = DsstZonal::from_j(MU, AE, &j).mean_element_rates(&aux);
        let explicit = zonal().mean_element_rates(&aux);
        assert_eq!(from_j, explicit, "from_j must reproduce the explicit Cₙ₀");
        // A J2-only field is not the degree-6 one: without this the comparison above would pass on
        // a `from_j` that dropped every coefficient past the first.
        let j2_only = DsstZonal::from_j(MU, AE, &j[..1]).mean_element_rates(&aux);
        assert_ne!(
            j2_only, explicit,
            "J2 alone cannot equal the degree-6 field"
        );
        assert_eq!(DsstZonal::from_j(MU, AE, &j).max_degree(), 6);
        assert_eq!(DsstZonal::from_j(MU, AE, &j[..1]).max_degree(), 2);
    }

    /// The eccentricity-power truncation the port fixes at the initial state must equal Orekit's
    /// own `maxEccPowMeanElements` (2 here) — else different terms are retained and the rates drift.
    #[test]
    fn truncation_order_matches_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        assert_eq!(zonal().max_ecc_pow(&aux), MAX_ECC_POW);
    }

    /// The six fixed-state mean-element rates match Orekit's `getMeanElementRate` to < 1e-10
    /// relative (`da` is identically zero, checked absolutely).
    #[test]
    fn fixed_state_rates_match_orekit() {
        let aux = DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU);
        let got = zonal().mean_element_rates(&aux);
        let mut max_rel = 0.0_f64;
        for i in 0..6 {
            let err = if RATES[i] == 0.0 {
                got[i].abs()
            } else {
                (got[i] - RATES[i]).abs() / RATES[i].abs()
            };
            max_rel = max_rel.max(err);
            assert!(
                err < 1e-10,
                "rate[{i}] got {} want {} rel {err:.3e}",
                got[i],
                RATES[i]
            );
        }
        std::println!("max fixed-state rate rel error: {max_rel:.3e}");
    }

    /// The MEAN semi-analytical propagation matches Orekit's `DSSTPropagator` at every sample:
    /// `a` to < 1e-9 relative, the equinoctial e/i components to < 1e-9 absolute, and the mean
    /// longitude to < 1 m along-track.
    #[test]
    fn mean_propagation_matches_orekit() {
        let z = zonal();
        let (mut max_a_rel, mut max_eh_abs, mut max_along) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (t, want) in SAMPLES {
            let got = z.propagate_mean(INIT, t);
            let a_rel = (got[0] - want[0]).abs() / want[0].abs();
            max_a_rel = max_a_rel.max(a_rel);
            for i in 1..=4 {
                max_eh_abs = max_eh_abs.max((got[i] - want[i]).abs());
            }
            let along = want[0] * wrap(got[5] - want[5]).abs();
            max_along = max_along.max(along);

            assert!(a_rel < 1e-9, "t={t}: a rel {a_rel:.3e}");
            for i in 1..=4 {
                let e = (got[i] - want[i]).abs();
                assert!(e < 1e-9, "t={t}: comp[{i}] abs {e:.3e}");
            }
            assert!(along < 1.0, "t={t}: along-track {along:.3e} m");
        }
        std::println!(
            "max mean-prop errors: a_rel {max_a_rel:.3e}, e/i_abs {max_eh_abs:.3e}, along {max_along:.3e} m"
        );
    }

    /// The osculating output (mean + short-period) matches Orekit's DSSTPropagator run in
    /// `PropagationType.OSCULATING` (`orekit_dsst_zonal_sp.csv`) to round-off. Orekit returns the raw
    /// mean state at the initial epoch, so `t = 0` is excluded — our osculating consistently applies
    /// the short-period there too. Cross-checked first-principles: the osculating tracks a numerical
    /// J2–J5 propagation to ~230 m over 12 h (the DSST truncation residual).
    #[rustfmt::skip]
    const OSC_GOLDEN: [(f64, [f64; 6]); 4] = [
        (10800.0, [7.20080406051416800e+06, 2.62505970455380930e-03, 9.07352971461781100e-03, 8.53294337752836500e-01, 7.71500600121665100e-01, 1.25799728461532960e+01]),
        (21600.0, [7.19653519386523100e+06, 3.37069108060647200e-03, 1.05265141886316960e-02, 8.51934578177427900e-01, 7.73077799984410100e-01, 2.37311878101715340e+01]),
        (32400.0, [7.20618577009010100e+06, 2.99398570992457000e-03, 9.69571419768334100e-03, 8.50206466868549800e-01, 7.74813287148281200e-01, 3.48794458105977440e+01]),
        (43200.0, [7.19190590428175800e+06, 3.77434602397508840e-03, 7.98874607136180700e-03, 8.48872684251018700e-01, 7.76510924445086500e-01, 4.60300713996471700e+01]),
    ];

    #[test]
    fn osculating_short_period_matches_orekit() {
        let z = zonal();
        let mut max_da = 0.0f64;
        let mut max_comp = 0.0f64;
        for (t, g) in OSC_GOLDEN {
            let o = z.propagate_osculating(INIT, t);
            let da = (o[0] - g[0]).abs();
            assert!(da < 1e-2, "t={t}: da = {da:.3e} m");
            max_da = max_da.max(da);
            for i in 1..6 {
                let d = (o[i] - g[i]).abs();
                assert!(d < 1e-9, "t={t} comp {i}: {d:.3e}");
                max_comp = max_comp.max(d);
            }
        }
        std::println!("DSST osculating vs Orekit: max da {max_da:.2e} m, max comp {max_comp:.2e}");
    }
}
