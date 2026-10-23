//! Where the DSST cuts its eccentricity series, against Orekit's own choice.
//!
//! `compute_max_ecc_pow` is the one output of `dsst::coefficients` that never appears in an answer.
//! It decides how many powers of the eccentricity the mean-element expansion keeps, and a cut one
//! term too early costs accuracy quietly: the propagated orbit still looks right, because the terms
//! that were dropped are the small ones. Every DSST test up to now compared a trajectory, and a
//! trajectory cannot see this — which is why 70 mutations of this function survived the suite.
//!
//! The fixture is `tools/orekit-oracle/DsstTruncationOracle.java` against **Orekit 13.1.7 /
//! Hipparchus 4.0.3**. Orekit computes the same quantity in the private
//! `DSSTZonal.computeMeanElementsTruncations` and stores it in the private field
//! `maxEccPowMeanElements`; the tool reaches it through the public `initializeShortPeriodTerms`
//! and reads the field reflectively. No `orekit-data` is involved — the gravity field is supplied
//! by the tool and the frame is EME2000, so the run is reproducible from the jars alone.
//!
//! The grid is a coarse sweep over LEO, MEO, GEO, GTO and Molniya states crossed with degrees
//! 2..12, plus two dense eccentricity sweeps (MEO to `e = 0.75`, LEO to `e = 0.06`, a `100 000 km` HEO to `e = 0.93`) whose step is
//! finer than any transition the coarse grid can resolve — so the fixture pins *where* the order
//! steps up, not merely that it does. Over the 1134 states it takes every value from 0 to 6.
//!
//! The field is `spacelab`'s own embedded `XGM2019e` zonals. The `cn0` rows exist so the
//! normalization step is asserted rather than assumed: the Java side hard-codes the fully
//! normalized `C̄ₙ₀` and multiplies by `√(2n+1)`, this side derives the same numbers from
//! `geopotential_coeffs::MODEL_COEFFS`, and the two must agree bit for bit before any truncation
//! order is compared.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

use std::vec::Vec;

use spacelab_propagation::dsst::{DsstAuxiliary, DsstZonal};
use spacelab_propagation::geopotential_coeffs::{MODEL_COEFFS, MODEL_GM, MODEL_RE};

const FIXTURE: &str = include_str!("fixtures/orekit_dsst_truncation.csv");

/// Degrees the oracle emits coefficients for, and the number of `trunc` rows it emits.
const MAX_DEGREE: usize = 12;
const CN0_ROWS: usize = 11;
const TRUNC_ROWS: usize = 1134;

/// Unnormalized `Cₙ₀ = C̄ₙ₀·√(2n+1)`, the `m = 0` case of the Kaula normalization factor.
fn unnormalized_c_n0(max_degree: usize) -> Vec<f64> {
    let mut c = std::vec![0.0; max_degree + 1];
    for &(n, m, cbar, _) in MODEL_COEFFS {
        if m == 0 && n <= max_degree {
            c[n] = cbar * ((2 * n + 1) as f64).sqrt();
        }
    }
    c
}

#[test]
fn the_eccentricity_truncation_order_matches_orekit() {
    let c_n0 = unnormalized_c_n0(MAX_DEGREE);
    let mut cn0_rows = 0_usize;
    let mut trunc_rows = 0_usize;
    let mut mismatches = Vec::new();

    for line in FIXTURE.lines() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        match f[0] {
            "cn0" => {
                let n: usize = f[1].parse().expect("degree");
                let want: f64 = f[2].parse().expect("coefficient");
                assert_eq!(
                    c_n0[n], want,
                    "unnormalized C_{n},0 differs from the oracle's own input"
                );
                cn0_rows += 1;
            }
            "trunc" => {
                let v: Vec<f64> = f[1..7]
                    .iter()
                    .map(|s| s.parse().expect("equinoctial element"))
                    .collect();
                let max_degree: usize = f[7].parse().expect("max degree");
                let want: usize = f[8].parse().expect("truncation order");
                let aux = DsstAuxiliary::new(v[0], v[1], v[2], v[3], v[4], v[5], MODEL_GM);
                let zonal = DsstZonal::new(MODEL_GM, MODEL_RE, c_n0[..=max_degree].to_vec());
                let got = zonal.max_ecc_pow(&aux);
                if got != want {
                    mismatches.push(std::format!(
                        "a={:.0} e={:.4} gamma={:+.4} n_max={max_degree}: got {got}, Orekit {want}",
                        aux.sma,
                        aux.ecc,
                        aux.gamma
                    ));
                }
                trunc_rows += 1;
            }
            other => panic!("unknown row kind {other}"),
        }
    }

    assert_eq!(cn0_rows, CN0_ROWS);
    assert_eq!(trunc_rows, TRUNC_ROWS);
    assert!(
        mismatches.is_empty(),
        "{} of {TRUNC_ROWS} states truncate at a different power:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}
