# SDTM-1: the prospective blind test

The protocol of `plans/17` Ф30.J, registered by the commit that adds this file, on 2026-11-12. At that date
no density row after 2026-04-30 had been read by any tool in this repository: the corpus ends on
2026-04-30 for GRACE-FO C and Swarm A, B and C (`data/validation/tudelft-v02/corpus/manifest.json`), and
TU Delft's server has refused connections since 2026-09-05 (`docs/provenance.md`, `tudelft-thermosphere`).
Everything below is fixed before the first blind row is read. A change to it after that is printed beside
the result it changed, with its date.

## 1 · What is frozen

The model is SDTM-1 as shipped. The baseline `docs/sdtm1-baseline.json` pins it: the coefficient tables
`sdtm/coefficients.rs` (sha256 `7cc5d588…062c938f`) and `sdtm/solar_wind_coefficients.rs` (`363ef5c1…602f763f`),
the model code `sdtm/mod.rs` (`b0d14895…87d1397f`) and `sdtm/atmosphere.rs` (`f4014e91…d332bb7c`), emitted at
`e06f607` from the corpus and the drivers the same file pins. `just freeze` and `just freeze-data` pass at
every read. No coefficient, form, scale or envelope of SDTM-1 changes on the evidence of a blind row before
the verdicts of §5 are printed.

The one rival of §5 (B5) is not shipped. The harness fits it on every row of the frozen corpus, with
`tools/evaluate-sdtm.py`'s field `flux_boundary_latitude`, at the commit of the runner (§7). The fit is
deterministic, so that commit and the frozen corpus fix it.

## 2 · The blind rows

- **Source.** TU Delft's thermosphere densities (CC BY 4.0), version 02 or a later version, which the report
  names. Every month from 2026-05 onward, on every mission the source publishes. At registration that is
  GRACE-FO C and Swarm A, B and C.
- **Build.** `tools/build-density-corpus.py`, with the rules it applies to the frozen corpus: the newest
  reprocessing of each satellite-month, nominal rows only, one row per 120 s, UTC. The blind rows go to a
  directory of their own. The frozen corpus stays byte-identical, so `just freeze-data` still passes.
- **Scored rows.** The harness's cadence: every 60th corpus row from the first (`STEP, PHASE = 60, 0`),
  about one row per 2 h, as in §10.38 of `docs/verification.md`. No row is excluded by any other rule.
- **Drivers.** Measured indices, fetched at the read by the scripts that fetched the frozen ones
  (`tools/fetch-density-data.sh`, `tools/fetch-omni.sh`). The report prints each file's sha256. The rows up
  to 2026-04-30 of each driver file may be revised by their provider; the report prints the largest change
  to a value the frozen field was fitted on.
- **Other sources.** Each is scored in a table of its own and never pooled with TU Delft's rows: ESA's
  Swarm `DNSxPOD` products, only after the user decides register row `esa-swarm-dns`; TianMu-1's densities,
  only where their terms allow it (`plans/17` Ф23.P). Starlink's public ephemerides test an orbit, not a
  density, and take a protocol of their own (Ф23.M).

## 3 · Models and modes

Every model reads measured indices: this is reconstruction (`plans/17` rule 12). Nowcast and forecast are
not run here (Ф30.H).

| model | input mode | licence of its number |
|---|---|---|
| SDTM-1 | F10.7 of the day before, its trailing 81-day mean, 3-hourly `ap` | ours: public |
| SDTM-1 with the solar wind (D-27) | as above, with hourly `ap60` and OMNI's hourly solar wind | ours, OMNI CC0: public |
| NRLMSISE-00 | F10.7 of the day before, its centred 81-day mean, the storm-time `ap[]` | NRL's public-domain code, our port: public |
| JB2008, clean-room (Ф18.A′) | SET's F10/S10/M10/Y10 at their lags, centred 81-day means, hourly `ΔTc` from `Dst` | our code: public |

The two SDTM-1 fields are two rows, never one pooled number.

## 4 · Metrics

As §10.38, on the rows every model answers:

- MAPE, `100 · mean |M/O − 1|`; the same with each satellite's mean `ln M/O` removed; the same on the days
  of daily `Ap` ≥ 50; `100 · mean |ln M/O|`; the median of `M/O`. Pooled and by satellite.
- Paired differences, SDTM-1 minus each model, on the same rows and the same 4 000 draws of
  `tools/cluster-intervals.py`, at its levels `day` and `storm`. A level with fewer than 10 clusters prints
  its point value and no interval.

## 5 · Hypotheses and their decision rules

Each verdict is one of four words. **Holds:** the 95 % interval lies on the stated side of zero at both
levels. **Fails:** it lies on the other side at both levels. **Not shown:** anything else. **Not resolved:**
a level has fewer than 10 clusters.

- **B1.** SDTM-1's MAPE is below NRLMSISE-00's, on all rows.
- **B2.** SDTM-1's MAPE is below the clean-room JB2008's, on all rows. On 2020–2026 with the fit frozen at
  2020 it was −5.24, storm interval [−7.79, −2.92] (§10.38).
- **B3.** SDTM-1's level: the pooled median `M/O` lies in [0.95, 1.05]. A point check, no interval; §10.38
  read 0.980.
- **B4** (Ф23.W, from §10.40). In the onset phase, in the dayside magnetic local-time sector (9–15 h), at
  `|MLat|` ≥ 60°, SDTM-1's mean `100 · ln M/O` is below zero; and the onset contrast, `|MLat|` ≥ 60° minus
  `|MLat|` < 40°, is below zero on every mission family present. Phases, the dipole and the sectors are
  `tools/residual-maps.py`'s; the interval resamples whole storms. Only the `storm` level applies. If it
  holds, an onset `Q_auroral` term may be proposed through the selection ledger
  (`docs/sdtm-selection-ledger.md`); this test does not add it.
- **B5** (Ф23.X, from §10.49). The field `flux_boundary_latitude` minus SDTM-1: its MAPE difference is below
  zero. Its storm-out folds (`tools/evaluate-sdtm.py run --kind storm --field flux_boundary_latitude`) are
  printed beside it in the same report. If it holds, it enters the ledger; a change to the shipped form
  needs a freeze re-baseline, which is the user's decision.

The solar-wind field's difference to SDTM-1 is printed, at all rows and at `Ap` ≥ 50, with no hypothesis.

## 6 · When it is read, and what is printed

- **First read:** when three calendar months of blind rows exist for at least one satellite. Then after
  every further three months, cumulatively.
- **Every read is printed in full**, whatever it shows, as a section of `docs/verification.md`, with this
  file's commit, the runner's commit and the sha256 of every input.
- **The verdicts of §5 are read once,** at the first read with 180 or more days of blind rows. Later reads
  print the same table and do not replace them.
- A change to SDTM-1 made after the verdicts enters the ledger, and the months after it are scored under a
  new protocol.

## 7 · The runner

The scoring code is written and committed before the first blind month is read, and its first run reproduces
§10.38's public table, to its printed digits, when pointed at the 2020 cut of the frozen corpus. It reads
the blind rows only through the paths of §2, and it scores the shipped field as its coefficient table
holds it, not a refit. Its commit is the one §1 and §6 cite.

Added on 2026-11-12, with the runner and before any blind row is read: three conventions that §10.38 applied and
this file did not name. Each satellite's log scale from the field's table (`SATELLITE_SCALE`) multiplies SDTM-1's
density, as the harness scored §10.38's folds; GRACE-FO C is the reference, at zero. The storm state runs from 120
days before the row, as the fit runs it, not over the tables' shorter `HISTORY`. The day of year is the day and the
whole hour, as the fit read it.

The runner is `tools/score-blind.py`, with `crates/spacelab-propagation/examples/blind_rows.rs` for SDTM-1's two
fields and the clean-room JB2008. On the same day, `tools/score-blind.py cut` printed §10.38's public table to its
printed digits: the four models' five columns, and the five paired differences with their intervals at days, years
and storms. Row by row, SDTM-1's two fields agree with the harness's residuals to 2.5e-14 in `ln`, and NRLMSISE-00
exactly. Over the tables' 15-day `HISTORY`, the fields frozen at 2020 (nitric oxide 40 h, against the shipped 30 h)
differed by up to 4.9e-4 and moved one interval bound by 0.01; hence the 120 days.

## 8 · What a reader can regenerate

Every public number above, from a clean checkout, the TU Delft data and the drivers of §2. The private rows
cannot be regenerated by a reader: DTM2020's licence forbids dissemination, and SET's agreement forbids giving
its program to anyone. The clean-room JB2008 stands for SET's program in public.
