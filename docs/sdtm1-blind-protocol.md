# SDTM-1: the prospective blind test

The protocol of `plans/17` Ф30.J, registered by the commit that adds this file, on 2026-11-12. At that date
no density row after 2026-04-30 had been read by any tool in this repository: the corpus ends on
2026-04-30 for GRACE-FO C and Swarm A, B and C (`data/validation/tudelft-v02/corpus/manifest.json`), and
TU Delft's server has refused connections since 2026-09-05 (`docs/provenance.md`, `tudelft-thermosphere`).
Everything below is fixed before the first blind row is read. A change to it after that is printed beside
the result it changed, with its date. §9 holds the amendments of 2026-11-24 (`plans/18` Ф30.N, D-40), made
before any blind file was fetched; a rule above that one of them changes says so.

## 1 · What is frozen

The model is SDTM-1 as shipped. The baseline `docs/sdtm1-baseline.json` pins it: the coefficient tables
`sdtm/coefficients.rs` (sha256 `7cc5d588…062c938f`) and `sdtm/solar_wind_coefficients.rs` (`363ef5c1…602f763f`),
the model code `sdtm/mod.rs` (`b0d14895…87d1397f`) and `sdtm/atmosphere.rs` (`f4014e91…d332bb7c`), emitted at
`e06f607` from the corpus and the drivers the same file pins. `just freeze` and `just freeze-data` pass at
every read. No coefficient, form, scale or envelope of SDTM-1 changes on the evidence of a blind row before
the verdicts of §5 are printed. The tag of §9 A6 pins this file and the baseline; a second model enters only as
§9 A5 says.

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
- **Other sources** (amended, §9 A8). This protocol scores TU Delft's densities only. Another source enters
  only by a protocol of its own, registered and tagged before its first file is fetched, and is never pooled
  with TU Delft's rows. None is registered.
- **Swarm A, B and C of 2026-03** (§9 A4) are not blind rows: `tools/score-blind.py gap` scores them as a row
  of their own.

## 3 · Models and modes

Every model reads measured indices: this is reconstruction (`plans/17` rule 12). Nowcast and forecast are
not run here (Ф30.H).

| model | input mode | licence of its number |
|---|---|---|
| SDTM-1 | F10.7 of the day before, its trailing 81-day mean, 3-hourly `ap` | ours: public |
| SDTM-1 with the solar wind (D-27) | as above, with hourly `ap60` and OMNI's hourly solar wind | ours, OMNI CC0: public |
| NRLMSISE-00 | F10.7 of the day before, its centred 81-day mean, the storm-time `ap[]` | NRL's public-domain code, our port: public |
| JB2008, clean-room (Ф18.A′) | SET's F10/S10/M10/Y10 at their lags, centred 81-day means, hourly `ΔTc` from `Dst` | our code on SET's index files: public under §7 of SET's agreement, with its acknowledgement; provisional (D-38) |

The two SDTM-1 fields are two rows, never one pooled number. The clean-room JB2008's row is computed in the
private tree: its drivers are SET's index files, and the path that reads them is not in the public cut. The
public runner prints it as a named skip with that reason (§9 A3).

**JB2008 acknowledgement (interim).** This document prints JB2008 numbers computed with the index files
`SOLFSMY.TXT` and `DTCFILE.TXT` of Space Environment Technologies (SET) as input. JB2008 is the model of
Bowman et al. (2008), "A New Empirical Thermospheric Density Model JB2008 Using New Solar and Geomagnetic
Indices", AIAA 2008-6438. The numbers are published under §7 of SET's JB2008 License Agreement of 2026-10-15,
which allows them in publications and reports with the acknowledgement and citation of SET's JB2008
Publication Acknowledgement (Notice to Users). That text was requested from SET on 2026-10-26; it replaces
this note when it arrives.

## 4 · Metrics

As §10.38, on the rows every model answers:

- MAPE, `100 · mean |M/O − 1|`; the same with each satellite's mean `ln M/O` removed; the same on the days
  of daily `Ap` ≥ 50; `100 · mean |ln M/O|`; the median of `M/O`. Pooled and by satellite.
- Paired differences, SDTM-1 minus each model, on the same rows and the same 4 000 draws of
  `tools/cluster-intervals.py`, at its levels `day` and `storm`. A level with fewer than 10 clusters prints
  its point value and no interval. B2's and B5's words are read on §9 A2's statistic, not on these
  intervals, which are still printed.

## 5 · Hypotheses and their decision rules

Each verdict is one of four words. **Holds:** the 95 % interval lies on the stated side of zero at both
levels. **Fails:** it lies on the other side at both levels. **Not shown:** anything else. **Not resolved:**
a level has fewer than 10 clusters. For B2 and B5 the same words come from §9 A2's rule. Which word withdraws
the recommendation of `plans/17` D-5′ is §9 A7.

- **B1.** SDTM-1's MAPE is below NRLMSISE-00's, on all rows.
- **B2.** SDTM-1's MAPE is below the clean-room JB2008's, on all rows. On 2020–2026 with the fit frozen at
  2020 it was −5.24, storm interval [−7.79, −2.92] (§10.38). Read by §9 A2's rule. Computed in the private
  tree; published under §7 of SET's agreement with the acknowledgement of §3 (§9 A3).
- **B3.** SDTM-1's level: the pooled median `M/O` lies in [0.95, 1.05]. A point check, no interval; §10.38
  read 0.980.
- **B4** (Ф23.W, from §10.40). In the onset phase, in the dayside magnetic local-time sector (9–15 h), at
  `|MLat|` ≥ 60°, SDTM-1's mean `100 · ln M/O` is below zero; and the onset contrast, `|MLat|` ≥ 60° minus
  `|MLat|` < 40°, is below zero on every mission family present. Phases, the dipole and the sectors are
  `tools/residual-maps.py`'s; the interval resamples whole storms. Only the `storm` level applies. Read once,
  at the first read with 10 storms with onset rows (§9 A1). If it holds, an onset `Q_auroral` term may be
  proposed through the selection ledger (`docs/sdtm-selection-ledger.md`); this test does not add it.
- **B5** (Ф23.X, from §10.49). The field `flux_boundary_latitude` minus SDTM-1: its MAPE difference is below
  zero. Read by §9 A2's rule. Its storm-out folds (`tools/evaluate-sdtm.py run --kind storm --field
  flux_boundary_latitude`) are printed beside it in the same report. If it holds, it enters the ledger; a
  change to the shipped form needs a freeze re-baseline, which is the user's decision.

The solar-wind field's difference to SDTM-1 is printed, at all rows and at `Ap` ≥ 50, with no hypothesis.

## 6 · When it is read, and what is printed

- **First read:** when three calendar months of blind rows exist for at least one satellite. Then after
  every further three months, cumulatively.
- **Every read is printed in full**, whatever it shows, as a section of `docs/verification.md`, with this
  file's commit, the runner's commit and the sha256 of every input.
- **B1 and B3 are read once,** at the first read with 180 or more days of blind rows. **B4** is read once,
  at the first read with 10 storms with onset rows (§9 A1). **B2 and B5** are read at the looks of §9 A2,
  each look once, at the first read that completes it. Later reads print the same table and do not replace
  a word already read.
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
and storms. In the public tree the clean-room JB2008's row and difference print as a named skip (§8, §9 A3). Row by
row, SDTM-1's two fields agree with the harness's residuals to 2.5e-14 in `ln`, and NRLMSISE-00 exactly. Over the tables' 15-day `HISTORY`, the fields frozen at 2020 (nitric oxide 40 h, against the shipped 30 h)
differed by up to 4.9e-4 and moved one interval bound by 0.01; hence the 120 days.

## 8 · What a reader can regenerate

Every public number above, from a clean checkout, the TU Delft data and the drivers of §2, except the rows
below. The private rows cannot be regenerated by a reader: DTM2020's licence forbids dissemination, and SET's
agreement forbids giving its program to anyone.

Nor can the clean-room JB2008's row, B2's difference and B2's word, though they are printed under §7 of SET's
agreement: the code is ours, but its drivers are SET's index files, which SET's agreement does not let us
redistribute, and the path that reads them is not in the public cut (§9 A3). The public runner prints them as
a named skip with that reason.

## 9 · Amendments of 2026-11-24, before any blind file is fetched

Made under `plans/18` Ф30.N, which carries out decision D-40 (a) and D-38's line on B2. At this date no blind
file has been fetched or read: the corpus still ends on 2026-04-30 (its manifest is pinned by
`docs/sdtm1-baseline.json`), and no blind directory exists. A1 to A6 are D-40's six; A7 and A8 settle two
gaps the review of 2026-11-15 found (`plans/18` I3 X13, X14). Each names the rules above that it changes;
`92ba8ca` (public `3651f45`) holds the text as first registered.

**A1 (2026-11-24). B4 is read at ten storms.** The window to 2026-10-31 holds six storms with onset rows, and
B4 needs ten, so as registered its only possible word at the 180-day read was "not resolved" (`plans/18` F122).
B4's word is read once, at the first read whose blind rows hold 10 or more storms with onset rows; until then
a read prints B4's cells and "not read". If the rows complete A2's last look with fewer than 10 such storms,
B4 is "not resolved". The rule for its words is §5's; only the time of the read changes (§5, §6).

**A2 (2026-11-24). B2 and B5 by a group-sequential rule.** As registered, B2 and B5 were read once, at
180 days, on about 14 storm-level clusters, where a real difference of B2's size is most likely "not shown"
(F122). They are now read at up to three looks. Look k reads the blind rows before 2026-11-01, 2027-05-01 and
2027-11-01, cumulatively. A look is complete when every satellite with a row in its six months has a row in
their last month; a satellite with no row there counts as ended once some satellite has rows 183 days past the
look's end. Each look is read once, at the first read that completes it. At each look, and at each of the two
registered levels, `day` and `storm`, the rule has three parts, each with its derivation.

- *(b) The statistic and its reference law.* `T = D / s`. `D` is the row mean of the paired MAPE difference
  (B2: SDTM-1 minus the clean-room JB2008; B5: `flux_boundary_latitude` minus SDTM-1) over the look's rows.
  `s` is its CR2 standard error over the level's clusters: with `w_j = n_j / N` a cluster's share of the rows
  and `m_j` its mean, `s² = Σ c_j (m_j − D)²` with `c_j = w_j² / (1 − 2 w_j + Σ w²)`. Under the working model
  that the clusters' means are independent with one variance `τ²`, `s²` is unbiased for `Var D = τ² Σ w²`, and
  it is a Gaussian quadratic form `m′Am` with mean `τ² tr A` and variance `2 τ⁴ tr A²`. Satterthwaite's
  two-moment match gives `ν = (tr A)² / tr(A²)` (Bell and McCaffrey 2002), and `T` is read against
  Student's t with `ν` degrees of freedom. Equal clusters give the one-sample t with `K − 1`.
  The working model fits these levels: on §10.38's rows the differences to NRLMSISE-00 spread a day inside
  its storm-level cluster about 0.7 as wide as a block's mean, so the clusters' shared parts dominate. A percentile interval of 4 000 draws,
  or a normal bound on `T`, does not hold the first look's two-sided level of 0.00052 (`plans/18` V2 4).
- *(a) The bounds, and how they follow the realised cluster counts.* The registered bounds are O'Brien and
  Fleming's `C·√(3/k)` for three looks at equal information and a two-sided 0.05: `C = 2.004036`, rounded up
  to 3.4711, 2.4545 and 2.0041. At equal information they spend 0.000518, 0.014317 and 0.049992 by looks 1, 2
  and 3. That is the schedule: by look k, the rule may have spent in all what the registered bounds spend by
  look k (Lan and DeMets 1983, with the spend fixed by look, not by information). Look k's bound `z_k` on the
  normal scale solves `P0(|Z_j| ≥ z_j for some j ≤ k) = spend_k` under the canonical joint law of the looks
  read: `Z_j = S_j / √V_j`, where `S_j` sums the differences of the rows read by look j and `V_j = Σ n_c²` sums
  the squared row counts of the level's clusters realised at look j. Under the working model `Var S_j = τ² V_j`
  and the clusters' sums are independent, so `corr(Z_i, Z_j) = √(V_i / V_j)`; with equal clusters `V_j` is
  proportional to the cluster count. The bounds follow the realised clusters through this correlation and
  through nothing else; `z_k` is rounded up to four decimals, and the bound on `T` is Student's quantile at
  `z_k`'s nominal two-sided level. Under the canonical law the level is at most 0.049992 at every
  information fraction, and equal counts give the registered bounds. Fixed bounds would give a level anywhere
  from 0.0451 to 0.0590 (0.0554 at fractions 1/10 and 3/10): the last bound's own level and three
  independent looks, the limits in which the union bound and Šidák's inequality become equalities. A look at which a level holds fewer than 10 clusters takes no decision, and its
  spend passes to the next look.
- *(c) The two levels at a look.* A look reads **holds** when `T ≤ −bound` at both levels, **fails** when
  `T ≥ bound` at both, and continues otherwise; the first word stops the rule. At look 3 a continuation reads
  **not shown**, and a look 3 that cannot decide reads **not resolved**. The event that both levels cross at
  some look lies inside the event that the storm level crosses at some look, so the rule's level is at most
  the storm level's own; the day level can only lower it, and the bounds need no adjustment for the
  conjunction (the intersection-union principle).

Derivations: `derivations/group_sequential_obf.mac` (15 identities: the correlation, Isserlis's variance,
Satterthwaite's `ν`, the CR2 mean, `ν = K − 1`; the constant, the bounds and every level above) and
`derivations/group_sequential_obf.gp` (the same at 30 digits, and the CR2 trace in exact rationals); the two
agree to nine digits.

The error rate, by simulation (S2). `tools/score-blind.py level --reps 100000` runs the rule as code under the
null. Each calendar of the three looks adds 10 to 14 storm-level clusters per look (every combination of 10
and 14, and 12 at each), with the information fractions they realise. Each cluster's and each day's mean
difference is drawn with mean zero, normal or heavy-tailed (Student's t with 3 degrees of freedom), at a
storm's spread of 0.6 and 1.4 of a block's and a day's of 0.4 and 0.8, around the 0.6 and 0.7 that §10.38's
differences to NRLMSISE-00 show.
Over the 72 gated cells, at 100 000 replicates a cell and seed 20261124, the rule's level lies between 0.0471
and 0.0515 under the normal nulls (pooled 0.0494, standard error 0.0001) and between 0.0391 and 0.0428 under
the heavy-tailed ones. Every cell's Monte Carlo interval, 95 % jointly over the 72 cells (Bonferroni), reaches
down to 0.05. "Holds" and "fails" share the level evenly, and in every cell it equals the storm level's own,
the most that (c) allows. Read against the normal bounds instead of Student's, the same statistics reach
0.0620 to 0.0802. Two results of the first runs changed the rule or its gate. Looks weighted by their
cluster counts put three normal cells above 0.05 (at most 0.0522); the squared row counts of (a) replaced
them. A gate on per-cell 95 % intervals then failed once, at 0.0515 [0.0502, 0.0529] (10, 10 and 14
clusters; normal; 1.4 and 0.8); 72 such intervals put one wholly above a true 0.05 with probability 0.84.
Rerun with fresh seeds at 400 000 replicates, that calendar's four normal cells gave 0.0496 to 0.0502, each
interval reaching 0.05. `tools/test_score_blind.py` holds the gate on the 18 gated cells at a storm's 1.4
and a day's 0.8, at 20 000 replicates a cell, and fails when the statistics are read against the normal
bounds.

*Scope.* The guarantee holds when the clusters' differences are symmetric about zero, as they are when the two
models' errors are exchangeable within each cluster. A null with mean zero and a skewed spread is not covered.
Under a null with mean zero and skewness +2 (an exponential less 1) the level is 0.0826 to 0.0984, nearly
all of it "holds" (0.0773 to 0.0931; "fails" 0.0048 to 0.0058): a cluster mean that falls low comes with a
small spread. A skewness of −2 moves the same excess to "fails". On §10.38's rows the storm-level differences to NRLMSISE-00 are skewed by −1.4: the long tail is
where SDTM-1 is far better. Under a null of that shape, "fails" would be the word that rises.

**A3 (2026-11-24). B2 in the public tree (D-38).** B2 stays registered. Its column is computed in the private
tree, because its drivers are SET's index files `SOLFSMY.TXT` and `DTCFILE.TXT`, and the path that reads them,
`space_weather::jb2008_drivers`, is not in the public cut (`docs/provenance.md`, `set-jb2008-indices`). The
public runner prints the clean-room JB2008's row, its paired difference and B2's word as a named skip with
that reason; §8 lists them among the rows a reader cannot regenerate. Their numbers are published under §7 of
SET's agreement with the acknowledgement of §3, in the class `number-public`, provisional until SET answers.
If SET's answer makes the class `number-private`, a dated amendment says that B2's word stays in the private
record (§3, §5, §7, §8).

**A4 (2026-11-24). Swarm of 2026-03.** The frozen corpus lacks 2026-03 for Swarm A, B and C (GRACE-FO C has
it): the archive of 2026-10-25 holds no member for that month. No fit has read it. If it appears, `blind`
refuses it, since it precedes `FROZEN_END`. `tools/score-blind.py gap DIR` scores it as a labelled row of its
own, with §4's table and no hypothesis; it is never pooled with the blind months (§2).

**A5 (2026-11-24). A second model.** The model under test is the frozen baseline of §1. If the re-baseline R1
(`plans/18` D-39) is selected by its rule on data before `FROZEN_END`, a dated amendment registers it beside
the baseline, as a second model with its own copies of the hypotheses, and that amendment is tagged and pushed
before the first blind file is fetched. The baseline stays registered and is scored as a rival row. Nothing
selected after the first fetch enters this protocol (§1).

**A6 (2026-11-24). The time stamp.** The blind months precede the freeze and the registration: the test is
blind because no file of them was ever in hand, not because they did not exist. The evidence is the corpus
manifest that the baseline pins and a dated, public tag. The tag `sdtm1-blind-protocol-1` marks the commit
that adds these amendments, with this file and `docs/sdtm1-baseline.json` as they stand there, and it is
pushed to the public repository before any blind file is fetched. A blind file fetched before that push voids
the test for the months it holds.

**A7 (2026-11-24). D-5′'s tripwire.** `plans/17` D-5′ recommends SDTM-1 for drag inside its fitted envelope,
and withdraws the recommendation "if SDTM-1 is not below NRLMSISE-00" on months no fit saw. Those are B1's
months. **Fails** withdraws the recommendation. **Not shown** does not: the report then says that the blind
months did not confirm it. **Not resolved** leaves it to the next protocol. A withdrawal needs evidence
against the model, not the absence of evidence for it: B1's chance of "holds" at its read is about 0.80
(F122), so a withdrawal on "not shown" would follow, about one time in five, from the number of storms in six
months rather than from the model (§5).

**A8 (2026-11-24). Other sources.** `plans/17` Ф30.J named three more blind sources: ESA's Swarm density
products of version 0301, Starlink's public ephemerides and TianMu-1's densities. None has a protocol or a
runner. ESA's terms wait on the user's decision (register row `esa-swarm-dns`); Starlink's terms are not in
the register; the register's TianMu-1 record (`tianmu-ngms`) is composition of October 2024, not a density of
a blind month. They are removed from this test.
Another source enters only by a protocol of its own, registered and tagged before its first file is fetched
(§2).
