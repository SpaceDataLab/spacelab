//! The GOT table is computed here, so it is checked against atlases computed elsewhere.
//!
//! `ocean_tide_fes2004` and `ocean_tide_fes2014` are transcriptions: their correctness is
//! provenance, and `generated_tables.rs` seals the bytes. `ocean_tide_got` is not — it is this
//! repository's own spherical-harmonic analysis of GOT5.5's elevation grids
//! (`derivations/tide_to_stokes.py`), so a seal says only that the committed numbers are the ones
//! the generator produced, not that the generator is right. What can say that is a comparison with
//! two independent groups' answers to the same question.
//!
//! The comparison is arranged so that it needs no invented tolerance. FES2004 and FES2014b are both
//! published, ten years apart, from different data and different software, so the difference
//! *between them* is the atlas-to-atlas spread at this degree — the noise floor of the whole
//! exercise. A pipeline with a convention error lands outside it; one that is working lands inside.
//! The assertion is therefore relative to that spread rather than to a number chosen by hand.
//!
//! It caught a real defect. IERS Eq. 6.20 pairs C̄ with sin ε and S̄ with cos ε, so the quantity
//! Eq. 6.21 rotates is `S̄ + iC̄ = iA`; the other pairing, `C̄ + iS̄ = conj(A)`, is equally true and
//! was what the generator first used. Every C came out as its own S. The table sealed, the
//! quadrature self-test passed at 1e-15, and the round trip was exact — because all three use the
//! generator's own convention. Only the published atlases disagreed.

// The tables publish exact decimals and the sums here are over a few hundred of them; the
// tolerances below are stated in the same 10⁻¹¹ units the tables carry. A missing wave or a missing
// cell is a defect in the tables rather than a case to handle, so the lookups panic on purpose.
// `FES2014b`, `C̄`, `S̄` and the rest are the conventions' own spelling, not identifiers.
#![allow(clippy::cast_precision_loss, clippy::doc_markdown, clippy::expect_used)]

use spacelab_propagation::ocean_tide_fes2004::{FES2004_WAVES, OceanTideWave};
use spacelab_propagation::ocean_tide_fes2014::FES2014B_WAVES;
use spacelab_propagation::ocean_tide_got::GOT_WAVES;

/// Degree 6 is where the GOT table stops (IERS §6.3.1 publishes no `k′n` above it), so every
/// comparison is over the triangle both sides fill.
const MAX_DEGREE: u8 = 6;

fn wave(waves: &'static [OceanTideWave], doodson: u32) -> Option<&'static OceanTideWave> {
    waves.iter().find(|w| w.doodson == doodson)
}

/// RMS of `a − b` over the `(n, m)` cells both waves carry, in units of the tables' 10⁻¹¹.
fn rms(a: &OceanTideWave, b: &OceanTideWave) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &(n, m, cp, sp, cm, sm) in a.coeffs {
        if n > MAX_DEGREE {
            continue;
        }
        let Some(&(_, _, cp2, sp2, cm2, sm2)) =
            b.coeffs.iter().find(|&&(n2, m2, ..)| (n2, m2) == (n, m))
        else {
            continue;
        };
        for (x, y) in [(cp, cp2), (sp, sp2), (cm, cm2), (sm, sm2)] {
            sum += (x - y) * (x - y);
            count += 1;
        }
    }
    assert!(count > 0, "no shared cells");
    (sum / count as f64).sqrt()
}

/// The Doodson numbers all three tables carry.
fn shared_by_all_three() -> Vec<u32> {
    GOT_WAVES
        .iter()
        .map(|w| w.doodson)
        .filter(|&d| wave(FES2004_WAVES, d).is_some() && wave(FES2014B_WAVES, d).is_some())
        .collect()
}

#[test]
fn our_analysis_of_got_is_closer_to_fes2014b_than_fes2004_is_on_every_shared_wave() {
    let shared = shared_by_all_three();
    assert_eq!(shared.len(), 9, "waves in GOT, FES2004 and FES2014b alike");
    for doodson in shared {
        let f14 = wave(FES2014B_WAVES, doodson).expect("in FES2014b");
        let ours = rms(wave(GOT_WAVES, doodson).expect("in GOT"), f14);
        let theirs = rms(wave(FES2004_WAVES, doodson).expect("in FES2004"), f14);
        assert!(
            ours < theirs,
            "Doodson {doodson}: our GOT analysis is {ours:.3}e-11 from FES2014b, \
             FES2004 is {theirs:.3}e-11 — a pipeline landing outside the spread between two \
             published atlases is a convention error, not a model difference"
        );
    }
}

#[test]
fn the_got_table_agrees_with_fes2014b_across_every_wave_the_two_share() {
    // Measured 2026-09-05: worst rms 0.228e-11 (J1), worst single cell 0.721e-11 (S2), median
    // 0.050e-11 against FES2004's 0.185e-11. The bounds sit just above the worst of those.
    let mut compared = 0usize;
    let mut worst_rms: f64 = 0.0;
    for got in GOT_WAVES {
        let Some(f14) = wave(FES2014B_WAVES, got.doodson) else {
            continue;
        };
        compared += 1;
        worst_rms = worst_rms.max(rms(got, f14));
    }
    assert_eq!(compared, 17, "waves shared by GOT and FES2014b");
    assert!(
        worst_rms < 0.30,
        "worst GOT-vs-FES2014b rms is {worst_rms:.3}e-11, was 0.228e-11 when measured"
    );
}

#[test]
fn the_disagreement_with_fes2014b_does_not_grow_with_degree() {
    // A quadrature or truncation error in the analysis would grow with degree; an atlas difference
    // does not. M2 is the wave with the signal to see it: its own rms falls from 18.9e-11 at degree
    // 2 to 9.0e-11 at degree 6, while the difference stays flat between 0.087 and 0.160e-11. That
    // flatness is the evidence that what is left is two atlases disagreeing, not this pipeline
    // losing accuracy — which is the claim the degree cap rests on.
    let got = wave(GOT_WAVES, 255_555).expect("M2 in GOT");
    let f14 = wave(FES2014B_WAVES, 255_555).expect("M2 in FES2014b");
    for degree in 2..=MAX_DEGREE {
        let mut sum = 0.0;
        let mut count = 0usize;
        for &(n, m, cp, sp, cm, sm) in got.coeffs {
            if n != degree {
                continue;
            }
            let &(_, _, cp2, sp2, cm2, sm2) = f14
                .coeffs
                .iter()
                .find(|&&(n2, m2, ..)| (n2, m2) == (n, m))
                .expect("FES2014b fills the triangle to degree 6");
            for (x, y) in [(cp, cp2), (sp, sp2), (cm, cm2), (sm, sm2)] {
                sum += (x - y) * (x - y);
                count += 1;
            }
        }
        let diff = (sum / count as f64).sqrt();
        assert!(
            diff < 0.20,
            "M2 degree {degree}: difference from FES2014b is {diff:.3}e-11, \
             which is no longer flat in degree"
        );
    }
}
