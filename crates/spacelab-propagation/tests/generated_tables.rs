//! The generated coefficient tables are sealed, because nothing else in the suite can check them.
//!
//! `geopotential_coeffs`, `ocean_tide_fes2004` and `ocean_tide_fes2014` are transcribed from
//! published models by the scripts under `derivations/`, and their correctness is a matter of
//! provenance: the generator parses the authoritative file, tallies what it emitted and refuses to
//! drop a row. What no accuracy test here can do is notice that one of those numbers has *since*
//! changed. Flip the sign of a degree-85 gravity coefficient and the acceleration moves orders of
//! magnitude below the agreement the field oracles already demand — `cargo mutants` generates 8619
//! such sign deletions across these tables and the whole suite catches none of them.
//!
//! So the assertion is over the bytes rather than the physics: fold the table the crate compiled
//! into an FNV-1a seal and compare it with the one `derivations/table_seals.py` computes by reading
//! the same source text with Python's float parser. Agreement says two independent parsers landed
//! on the same doubles in the same order. When a table is legitimately regenerated, re-run that
//! script and update the constant here — the diff to the table is what review reads, and the seal
//! is what makes an *un*reviewed change to it impossible to ship.

// Degree and order reach 120; every one of them is exact in an f64.
#![allow(clippy::cast_precision_loss)]

use spacelab_propagation::geopotential_coeffs::MODEL_COEFFS;
use spacelab_propagation::ocean_tide_eot20::EOT20_WAVES;
use spacelab_propagation::ocean_tide_fes2004::{FES2004_WAVES, OceanTideWave};
use spacelab_propagation::ocean_tide_fes2014::FES2014B_WAVES;
use spacelab_propagation::ocean_tide_got::GOT_WAVES;
use spacelab_testkit::fnv1a_f64;

/// A tide table flattened the way `table_seals.py` reads it: Doodson number, then each
/// `(n, m, C+, S+, C−, S−)` row in file order.
fn tide_values(waves: &'static [OceanTideWave]) -> impl Iterator<Item = f64> {
    waves.iter().flat_map(|wave| {
        std::iter::once(f64::from(wave.doodson)).chain(
            wave.coeffs
                .iter()
                .flat_map(|&(n, m, cp, sp, cm, sm)| [f64::from(n), f64::from(m), cp, sp, cm, sm]),
        )
    })
}

#[test]
fn the_gravity_field_is_the_table_the_generator_wrote() {
    // Degrees 2..=120 fill the triangle: one row per order, so (3 + 121)·119/2.
    assert_eq!(MODEL_COEFFS.len(), 7378, "XGM2019e rows to degree 120");
    let seal = fnv1a_f64(
        MODEL_COEFFS
            .iter()
            .flat_map(|&(n, m, c, s)| [n as f64, m as f64, c, s]),
    );
    assert_eq!(seal, 0x0efb_b311_1ef0_b35d, "XGM2019e coefficients moved");
}

#[test]
fn the_fes2004_tide_table_is_the_one_the_generator_wrote() {
    assert_eq!(FES2004_WAVES.len(), 18, "FES2004 waves");
    let rows: usize = FES2004_WAVES.iter().map(|w| w.coeffs.len()).sum();
    assert_eq!(rows, 402, "FES2004 rows to degree/order 6");
    let seal = fnv1a_f64(tide_values(FES2004_WAVES));
    assert_eq!(seal, 0x9ddd_af25_5856_7b7c, "FES2004 coefficients moved");
}

#[test]
fn the_fes2014b_tide_table_is_the_one_the_generator_wrote() {
    // This one is `include!`d from a sibling `.in` file, so the seal is also the only check that
    // the module still pulls in the full table rather than a stale or partial copy of it.
    assert_eq!(FES2014B_WAVES.len(), 35, "FES2014b waves");
    let rows: usize = FES2014B_WAVES.iter().map(|w| w.coeffs.len()).sum();
    assert_eq!(rows, 175_033, "FES2014b rows to degree/order 100");
    let seal = fnv1a_f64(tide_values(FES2014B_WAVES));
    assert_eq!(seal, 0x4a82_951c_1210_44f4, "FES2014b coefficients moved");
}

#[test]
fn the_got_tide_table_is_the_one_the_generator_wrote() {
    // This one is not transcribed but computed, from GOT5.5's elevation grids, so the seal answers a
    // narrower question than it does for the FES tables: it says the committed numbers are still the
    // ones `derivations/tide_to_stokes.py` produced. Whether the generator is *right* is
    // `tests/ocean_tide_atlas.rs`, which compares it with two published atlases.
    assert_eq!(GOT_WAVES.len(), 31, "GOT5.5 waves that Eq. 6.21 applies to");
    let rows: usize = GOT_WAVES.iter().map(|w| w.coeffs.len()).sum();
    assert_eq!(rows, 775, "GOT rows to degree/order 6");
    let seal = fnv1a_f64(tide_values(GOT_WAVES));
    assert_eq!(seal, 0x48b3_0d5a_3de3_3ddd, "GOT coefficients moved");
}

#[test]
fn the_eot20_tide_table_is_the_one_the_generator_wrote() {
    // Computed from EOT20's elevation grids by the same generator as GOT, so the seal answers the
    // same narrower question; `tests/ocean_tide_atlas.rs` is what says the analysis is right.
    assert_eq!(
        EOT20_WAVES.len(),
        16,
        "EOT20 waves that Eq. 6.21 applies to"
    );
    let rows: usize = EOT20_WAVES.iter().map(|w| w.coeffs.len()).sum();
    assert_eq!(rows, 400, "EOT20 rows to degree/order 6");
    let seal = fnv1a_f64(tide_values(EOT20_WAVES));
    assert_eq!(seal, 0xac3c_117a_5c9e_1c4f, "EOT20 coefficients moved");
}
