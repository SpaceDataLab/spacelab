# SDTM-1's selection ledger

Every number SDTM-1 quotes up to the baseline in `docs/sdtm1-baseline.json` is a **validation** number.
The mission-family folds that report it also chose the form it reports it for, and a held-out family
still saw the storms and the driver history of the families that fitted it, because the missions overlap
in time. There is no test number yet. This ledger says what was chosen on what, so that a later test can
be read for what it is. `tools/check-sdtm-freeze.py` fails on a change to the field, the form or the data
that does not come with a new row here and a new baseline.

A row is a choice of form or of what ships. A parameter fitted inside each fold is not a choice, and the
ledger says so where a scan could have made it one.

| # | Choice | Section, commit | Read on | Rejected on the same data | What it quotes |
|---|---|---|---|---|---|
| 1 | Folds by mission family, the eight satellites in five families | §10.21, `16fb332` | The twins: GRACE A/B and Swarm A/C fly the same orbit, so holding out one scores its twin. No error was read to choose this | Folds by satellite | The bar: 25.5 % pooled |
| 2 | USGS `Dst` does not drive the storm kernel | `08cedc3` | Family folds, the corpus storms | `Dst`-driven kernel | Refuted, nothing ships |
| 3 | Heating with nitric-oxide cooling as the storm state | §10.24, §10.25, `6f160b9`, `98c9739` | Family folds, both thinning phases, the 630 storm comparisons. The sign was read on §10.22's per-phase O/C first | The two-constant kernel | 23.1 % pooled |
| 4 | The daily flux excess saturates as `D tanh(d/D)`, `D` fitted in each fold | §10.26, `f5d2437` | Family folds. A scan of fixed `D` kept improving, so `D` was left free to keep the scan from choosing it. The *form* was still chosen here | Linear; fixed `D` = 1, 0.5, 0.35, 0.25 | 21.5 % pooled |
| 5 | The storm drive gains `(F10.7a/150)^κ` inside `a`, with a linear `s` (`storm-log1-lin`) | §10.27, §10.28, `3432ce6`, `9cf925d` | Family folds, both phases, the 630 and the 16 comparisons | `lean`, `storm-log1-geo`; `+ a·f`, `× exp(κf)`, `D·exp(κf)`, the power gain without linear `s`; a fitted knee, `[a, a²]`, a capped cooling; row weights 5 and 20 | 20.8 % pooled, 23.9 % at `Ap ≥ 50`, 357 peaks within ±25 % |
| 6 | The 3-hourly `ap` stays the default drive, `ap60` is measured and not shipped | §10.30, `20b9cfc`, D-26 | Family folds and storms, then what a forecast can supply | `mix`, `fast`, `ap60` | 0.635 → 0.707 at the largest storms |
| 7 | An opt-in solar-wind field: half the drive from the wind, the share fitted | §10.31, §10.32, `3c70317`, `e06f607`, D-27 | The same folds and storms. Four drives were fitted there: the share from one start, fixed at 1, fixed at ½, and from two starts. The optimiser's start was chosen by the training cost | `s = 1`, and shipping it as the default | 20.45 % pooled, 0.941 at the largest storms |
| 8 | Inside the envelope SDTM-1, outside NRLMSISE-00 | §10.28, §10.29, `3dbad7b`, D-5′ | Family folds, by flux and storm bin | NRLMSISE-00 everywhere | 20.8 % against 50.1 % |

**What the table adds up to.** At least 25 forms were read on one set of five folds, and the number in the
last column is each time the best of what was tried at that step. Each row was a real gain and the steps
are not independent of what went before, so no correction has been applied to them; the honest statement
is that the optimism is unmeasured until out-of-time and leave-one-storm-out folds exist (`plans/17` Ф30.B).

**Known leaks in the baseline, kept as they are.** The `CouplingMap` of row 7 is a quantile table built over
the whole span, so each fold's map saw later storms. The scales and `DensityUncertainty` cells are also
built over the whole span. These stay in the frozen field; the harness rebuilds them inside each fold.

**Read but not chosen on.** §10.33's storm phases and §10.34's named storms chose nothing, so they are not
rows. They have been seen, though: a later change designed on them is selected on them and enters here. The
same holds for §10.36, which runs the frozen field and chooses nothing either.

## After the baseline

Each change to the form, the coefficients, the corpus or a driver after 2026-11-01 adds a row and a new
baseline. It names the harness split that decided it, and that split is then no longer a test for that row.

| # | Choice | Commit | Decided on | Rejected | What it quotes |
|---|---|---|---|---|---|
