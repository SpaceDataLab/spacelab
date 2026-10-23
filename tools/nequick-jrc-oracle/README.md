# NeQuick-G JRC oracle

The reference implementation of the Galileo single-frequency ionospheric correction algorithm:
NeQuickG JRC 1.0 (European Union, 2019-12-10, EUPL-1.2), the code that the Galileo ICD's Annex E
verification data come from. It is fetched from a mirror (`odrisci/NeQuickJRC`, commit `02701f2`,
the 2019 release with ANSI-C build fixes) rather than vendored; what is committed here is the
driver that calls it and this record.

`run.sh` builds the library with `stec.c`, feeds it the inputs of
`crates/spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv`, and compares its output with
the committed column. The inputs are the 108 Annex E cases of the ICD (issue 1.2), each repeated
for all twelve months, plus an anti-meridian and a zenith geometry.

## Agreement

`NeQuickGalileo::stec` reproduces the reference to **1.7e-13 relative** on all 1298 rows. The
reference itself reproduces its own 576-case benchmark exactly at the five decimals it prints.

## What it found

Before this oracle the crate was validated against Orekit's `NeQuickGalileoTest`, which pins
Orekit's own output. Against the reference the same code was out by up to **18.8 TECU**, for four
reasons, all inherited from Orekit 13.1.7. Each effect below is that one cause restored alone, the
largest change it makes over the 1298 rows:

| cause | largest effect | where the ICD says so |
|---|---|---|
| Az re-evaluated from the MODIP at every integration point | 18.8 TECU | §2.5.4.4: "Az is not updated with MODIP along the ray" |
| CCIR maps for July to December with a corrupt last F2 coefficient | 4.2 TECU | the files embedded in the ICD |
| E-layer amplitude floor (Eq. 98) skipped when there is no F1 layer | 2.9 TECU | Annex F pseudo-code; §2.5.5 places it in one branch only |
| doubling two-point Gauss rule over one segment instead of Kronrod G7-K15 | 2.6 TECU | §2.5.8.2.7 splits at 1000 and 2000 km; Annex F.2.6 |

The first error vanishes on a vertical ray, which is why zenith checks never saw it: along a
vertical ray the MODIP does not change. The third appears at night under a dense F2 layer, where
the F2 tail at 120 km exceeds NmE and the unfloored Eq. 92 goes negative (−2.54 against the
reference's 0.05 at 5.25° N, 307.19° E, April, 00 UT, high solar activity).

## The Annex E table is not the reference rounded

Annex E prints two decimals. 39 of its 108 values differ from the reference implementation by more
than the 0.005 TECU rounding, up to 0.038 TECU. The fixture therefore pins the reference at 17
digits, and checks the printed table at 0.04 TECU only.
