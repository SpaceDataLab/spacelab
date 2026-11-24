# Provenance register — what each ported module descends from, and whether we may ship it

`CONTRIBUTING.md` holds the *rule*: science is free, expression is not, and translating an
implementation between languages is a derivative work. This file holds the *findings* — one row per
module in this workspace that describes itself as a port, transcription or translation, with the
upstream artefact, that artefact's licence as read from the artefact itself, and the verdict.

Two things make a register necessary rather than a paragraph in `CONTRIBUTING.md`. The first is that
`cargo deny` cannot see any of this: it reads the licence metadata of *dependencies*, and every entry
below is code we wrote, so the whole class is invisible to the gate. The second is that the answer
changes with what we intend to do. Reading a licensed implementation is always fine; running it as an
oracle is usually fine; shipping a translation of it under `MIT OR Apache-2.0` is a different question
with a different answer, and this workspace is intended to be published.

**Every licence below was read from the artefact on 2026-09-04**, not from memory or from a summary —
the retrieval is named in each row so a re-check knows where to look. A licence that has changed since
is a finding, not a footnote.

## The register

| Module | Upstream artefact | Upstream licence, as read | Verdict |
|---|---|---|---|
| `spacelab-propagation/src/dtm2020/{mod,tables}.rs` | CNES `dtm2020_F107_Kp-subr.f90` and `DTM_2020_F107_Kp.dat`, from `swami-h2020-eu/mcm` | **Proprietary.** `LICENSE` in that repository: academic and non-commercial only; no modification; no dissemination "or any of its supporting data"; use is acceptance | **Must not ship as it stands.** See §1 |
| `spacelab-propagation/src/nrlmsise00/**` | **`nrlmsise00_sub.for`**, the NRL FORTRAN by Picone, Hedin & Drob | **US Government work**, not subject to domestic copyright (17 U.S.C. §105) | **Ship. Re-based, and the re-basing is measured.** It descended from Brodowski's C, which carries **no grant of any kind** — warranty disclaimer only, GitHub `license: null`. See §2 |
| `tools/jb2008-set-driver/` | Bruce R. Bowman's `JB2008.for`, Space Environment Technologies — compiled unmodified, never committed; the directory holds only our driver program and its build script | **SET's JB2008 License Agreement** (the PDF dated 2026-10-15, read 2026-10-25): own research, educational, engineering and internal use only; no modification or translation except to compile and interface; no embedding in a library given to others | **Development tree only.** No translation exists any more; the driver is excluded from the public tree. See §3 |
| **28 files** — all 14 of `spacelab-propagation/src/dsst/`, `spacelab-propagation/src/{earth_radiation,spacecraft}.rs`, all 8 of `spacelab-ionosphere/src/` (the NeQuick chain), `spacelab-gnss/src/{gpt3,itu834_mapping}.rs`, `spacelab-elements/src/{brouwer_lyddane,eckstein_hechler}.rs` | Orekit (CS GROUP, formerly CS-SI) | **Apache-2.0** — `LICENSE.txt` at `CS-SI/Orekit`; copyright line read from `DSSTPropagator.java` | **Ship, with attribution owed.** See §4. The full list is in `NOTICE`, and it is 28 rather than the 15 a first pass counted — the NeQuick and GPT3 chains say "Ported from Orekit's …" in their own module docs and had not been counted |
| `spacelab-sensors/src/{geoloc,scan}.rs` | `pyorbital.geoloc` and `pyorbital.geoloc_instrument_definitions` (pytroll) | **Apache-2.0** — GitHub licence API | **Ship, with attribution owed.** Same conditions as Orekit; in `NOTICE` |
| `spacelab-earthvision/src/image_quality.rs` | the metric core of Satellogic's `iquaflow` | **MIT** — GitHub licence API | **Ship, with attribution owed.** In `NOTICE` |
| `spacelab-slr/src/{displacement,loading,troposphere}.rs` | IERS Conventions Centre reference routines `DEHANTTIDEINEL`, `HARDISP` (with `ADMINT`/`TDFRPH`/`SPLINE`/`EVAL`), `FCULZD_HPA`, `FCUL_A` | IERS Conventions software, published by the Conventions Centre as the reference realisation of the Conventions | **Ship.** `CONTRIBUTING.md`'s first paragraph is about exactly this material: the IERS Conventions and everything the IERS publishes are ours to implement. Orekit's own `NOTICE` credits the same routines |
| `spacelab-propagation/src/ias15.rs` | Everhart (1985), Rein & Spiegel (2015); REBOUND used only as a numerical check | REBOUND is **GPL-3.0** — `LICENSE` at `hannorein/rebound` | **Clean, and the reason the derive-first rule pays.** The nodes are the roots of `d⁷/dx⁷[x⁸(x−1)⁷]` and the `c`/`d` conversion matrices come from `derivations/ias15_nodes.mac`; REBOUND's hard-coded arrays are the agreement check at ~1e-33, not the source. Deriving first is what makes a GPL implementation safe to *compare against* |
| `spacelab-gpu/src/batch_sgp4.rs` | the `sgp4` crate 2.4 | **MIT** — `cargo metadata` | **Ship, with attribution owed.** The MIT notice has to travel with a substantial portion; §4's `NOTICE` carries it |
| `spacelab-sgp4/src/lib.rs` | — | — | **Not a port.** The crate *depends on* `sgp4` 2.4 (MIT) and adds typing and batching around it. The word "port" in its docs describes the upstream crate, not this one |
| `spacelab-eop/src/subdaily_tables.rs`, `spacelab-slr/src/loading_tables.rs`, `spacelab-ionosphere/src/lib.rs`, `spacelab-propagation/src/{rkn1210,rkn1210_consts}.rs` | IERS Conventions software tables (`ORTHO_EOP`/`CNMTX`, `PMSDNUT2`, `UTLIBR`, `ADMINT.F`), the Galileo ICD (Annex C) CCIR maps and MODIP grid and the ITU-R MODIP grid, Dormand–El-Mikkawy–Prince (1987) | published tables | **Clean.** Measured constants and published coefficient tables are facts, and every one of these is *machine-read* into the tree by a script in `derivations/` or `tools/` rather than hand-copied — which is a correctness argument first and a provenance argument second |
| `spacelab-astro/src/richardson.rs`, `spacelab-propagation/src/{adams_nordsieck,geopotential,ias15,radau,rosenbrock}.rs` | Maxima derivations in `derivations/*.mac` | — | **Clean, and the pattern to copy.** Each of these says in its own documentation that the coefficients are *derived* rather than transcribed, and names the `.mac` file that derives them; the published listing is the agreement check. `rosenbrock.rs` is the one that also names a reference listing (Hairer & Wanner's `ros4.f`, BSD-style), and a Butcher tableau is facts in any case |
| `spacelab-propagation/src/ocean_tide_got.rs` | GOT5.5 **cotidal grids** (amplitude and phase), R. D. Ray, NASA GSFC | **No stated terms.** `earth.gsfc.nasa.gov` serves the atlas with no licence file, no registration and no click-through; the `#README` names papers to cite and warns against navigational use, nothing more | **Ship, with the finding recorded.** And it is not a port: the coefficients are *computed here* by `derivations/tide_to_stokes.py` from the elevation grids. See §5 |
| `spacelab-gnss/tests/fixtures/{rinex3_nav_igs_brdc.24n,rinex4_nav_brd400dlr.rnx}` | 40 lines of `BRDC00WRD_R_20240010000_01D_MN.rnx` and 219 of `BRD400DLR_S_20262000000_01D_MN.rnx` — IGS multi-GNSS broadcast navigation via the BKG Frankfurt data centre, the second one DLR/GSOC's BCEmerge product (doi:10.57677/BRD400DLR) | **Open, and the terms say so.** *IGS Data and Product Disclaimer and Terms of Use*, 2020-08-05: IGS data are "provided openly for the benefit of all scientific, educational, and commercial users" and "made openly available for use without restriction". Attribution is required and use is acceptance | **Ship, with attribution in the file itself.** See §6 |
| `spacelab-coverage/src/root.rs`, `spacelab-earth/src/lib.rs`, `spacelab-estimation/src/hinf.rs`, `spacelab-fa/src/{fundamental,lib}.rs`, `spacelab-propagation/src/lib.rs` | — | — | **Not ports — the opposite.** Each of these matched the sweep because its documentation says it is *not* transcribed from an implementation: "not transcribed from any one listing", "never transcribed from another implementation's source", "Nothing here is ported from it". They are listed so that the sweep below is a closed set rather than a filter someone has to re-run judgement on |

## The sweep, and why the count moved

The register is built from a mechanical sweep, not from recollection: `tools/check-provenance.py`
lists every `crates/*/src/**/*.rs` whose **module documentation** claims a port, a translation or a
transcription, and fails if any of them is missing from the covered set at the end of this file. That
is **56 files**, and every one is accounted for below — as a derivation with a verdict, as facts, or
as a documented *denial* of derivation.

Two counts moved on the way here, and both moved the same way. A first pass done by reading the
modules that came to mind found 15 Orekit-derived files; the real number is **28**, the missing ones
being the whole NeQuick chain in `spacelab-ionosphere`, `gpt3.rs`, `itu834_mapping.rs` — which that
pass had actively mis-filed as a published ITU table — and three more under `dsst/`. Then the first
*mechanical* pass found 47 rather than 56, because its pattern required "transcribed **from**" and
nine modules write "transcribed verbatim from", "transcribed byte-for-byte", "transcribing 342 rows".

Neither miss was harmless in the same way. A `NOTICE` naming 15 of 28 files discharges Apache-2.0's
attribution condition for 15 of 28 files, and a regex that reads as exhaustive while missing a sixth
of the tree is worse than no regex, because it is trusted. The remedy is that the sweep is now a
checked command rather than a paragraph describing one.

## 1 · DTM2020 — the one that blocks

`swami-h2020-eu/mcm`'s `LICENSE`, verbatim:

> CNES, the Met Office and Elecnor Deimos grant to the Licensee a free non-exclusive and
> non-transferable license for the MCM software for academic, non-commercial, purposes only. […] A
> user of MCM or DTM software shall not:
> - use it for any non-academic, commercial purposes,
> - make any modification or improvement to the Software,
> - disseminate it or partly disseminate it, or any of its supporting data
> - copy it except for archival or emergency repair
>
> […] THE USE OF THE MCM OR DTM SOFTWARE CONSTITUTES AN IMPLICIT ACCEPTATION BY THE USER OF THE TERMS
> OF THE PRESENT LICENSE.

Three of those four clauses are engaged by what is in the tree. `dtm2020/mod.rs` describes itself as
"a faithful f64 port of the reference Fortran `dtm2020_F107_Kp-subr.f90`", which is a modification.
`dtm2020/tables.rs` holds "verbatim `.dat` digits" from `DTM_2020_F107_Kp.dat`, which is the
supporting data. Publishing the workspace is dissemination of both.

**Asking CNES for a commercial licence is worth doing and does not solve this.** The licence on offer
is non-exclusive and non-transferable — it would legitimise *our* use, including running the Fortran
as an oracle, which a commercial venture is otherwise outside the grant for. It would not give us the
right to sub-license the model to everyone who downloads a permissively-licensed crate. Redistribution
is the clause an agency is least likely to relax, and the open-source release is exactly the act it
forbids.

**The remedy separates three things that are currently one module**, and each lands somewhere
different:

* **The mathematics.** Bruinsma & Boniface, *The operational and research DTM-2020 thermosphere
  models*, J. Space Weather Space Clim. 11, 47 (2021) is **CC BY 4.0** — an explicit grant to reuse
  with attribution. The Bates–Walker profile, the spherical-harmonic `gldtm` expansion and the driver
  parameterisation are all in the paper. An implementation written from it is unencumbered, and the CC
  BY licence removes even the derivative-of-expression question, provided the code is written from the
  paper and not transcribed from the Fortran.
* **The coefficient set.** It does not ship. The engine gains a reader for `DTM_2020_F107_Kp.dat` and
  `tools/fetch-density-data.sh` gains the download, so each user obtains the file from CNES under
  their own acceptance of CNES's terms. This is not a workaround invented for this case — it is the
  pattern this repository already uses five times, for `finals2000A`, ICGEM `.gfc` fields, the FES
  ocean-tide atlases, HWM14 and the MT_CKD continuum. The library stays permissive and the data stays
  its author's.
* **The Fortran.** Oracle only, and never read while writing the replacement.

`tables.rs` has to leave the **history**, not just the tree: a tag cut from the current history
distributes those digits to everyone who clones it. That is the one irreversible-in-the-wrong-direction
step here, and it is cheaper now — before any public tag — than after.

The longer answer, and the one that would make the model ours outright, is to fit our own coefficient
set. `density_calibration.rs` and the 348 693 rows of TU Delft accelerometer and POD density in §10.2
are most of what a DTM-class fit needs. That is research rather than a ticket, and it is the same line
as the open density-calibration layer that already stands in for HASDM.

**Its numbers stay in the development tree too (`plans/18` D-32).** A number computed by running
DTM2020 is class `number-private`: a score, a ratio, a correlation, a table column, a ranking against
it, and the stretch target its MAPE set for SDTM-1 with that target's verdict. The development tree
keeps each behind a private marker, a table keeps a DTM2020 column behind a column marker the public
cut drops, and the public tree prints none. `tools/check-classes.py --check` fails where the public
view holds one, and the public cut refuses its own output where it holds one. A line that names
DTM2020 with no number stays, and so does a number a third party published about it, cited.

**What release 0.3.0 already carries (D-58).** Read on 2026-11-24 at tag `v0.3.0` (public `8da2422`)
with `tools/check-dtm-numbers.py` (now part of `tools/check-classes.py`), whose wide pattern is an
upper bound: `docs/verification.md` holds
15 lines that name DTM2020 beside a percentage and 50 by the wide pattern, `docs/limitations.md` one
number, the comments of `crates/spacelab-propagation/tests/measured_density.rs` 3 and 8, and a
comment of `.cargo/mutants.toml` the model's measured density floor. One line is inside a published
package: line 8 of `src/sdtm/mod.rs` in `spacelab-propagation` 0.3.0 on crates.io
(`spacelab-propagation-0.3.0.crate`, sha256
`94c25a7b00d10408c573dbcbfadfd65fd1581719d5fd722fc2c5bae0544645c1`), a module doc that prints
DTM2020's pooled MAPE, of the kind docs.rs renders; no other file of that package prints a DTM2020
number. PyPI's `spacedatalab` 0.3.0 prints none: its source archive (`spacedatalab-0.3.0.tar.gz`,
sha256 `b06061fd4538951d831fa25b620365dc2a5ad22bba6b6d0e9ffb2c197ece8529`) holds the wrapper crate and
its READMEs only, and the extension module of the `cp311-abi3-manylinux_2_28_x86_64` wheel holds no
string naming DTM2020. Nothing is yanked: a yank stops new resolutions and removes no stored file and
nothing from the tag, and it is the owner's act. The 0.4.0 release notes are to say so (`plans/18`
Ф32.A).

## 2 · NRLMSISE-00 — re-based on the source of record, and the re-basing is a measurement

Brodowski's C package states no terms at all. There is no `COPYING`, the source header names the
authors and gives a contact address, `DOCUMENTATION` offers only a warranty disclaimer, and GitHub
reports `license: null`. No licence is a *worse* position than a copyleft licence, not a better one:
GPL grants permission on conditions, and silence grants nothing. (The `nrlmsise00` PyPI wrapper
declares GPL-2.0, but that is the wrapper author's choice for their own wrapper and establishes
nothing about the C.)

A data point that is not a defence but is worth having: **Orekit ships Apache-2.0 translations of
both this C and of CNES work.** Its `NOTICE.txt` says so in as many words — "software translated from
original work developed by Dominik Brodowski" and "software translated from original work developed by
R. Biancale, S. Bruinsma (CNES)" — and Orekit is redistributed by everyone. That establishes what the
field's practice is, not what the licences permit, and the two are different questions.

**The exit is that the C is not the model's source of record, and taking the real one cost a header
and a generator rather than a rewrite.** NRLMSISE-00 was developed by Picone, Hedin and Drob at the
**U.S. Naval Research Laboratory**, and a work of the United States Government is not subject to
domestic copyright (17 U.S.C. §105). `nrlmsise00_sub.for` is that work; the C is a transcription of
it. `tools/fetch-nrlmsise00-fortran.sh` pins it by commit into gitignored space with a digest.

What was done, and what each step is worth:

* **The tables are re-derived from the FORTRAN and the result is checked, not asserted.**
  `derivations/nrlmsise00_tables.py` now reads `BLOCK DATA GTD7BK` instead of `nrlmsise-00_data.c`,
  and regenerating reproduces the shipped `tables.rs` **byte for byte** — all 3 290 values, the same
  decimal digits in the same order, the only diff being the header naming the new source. That is the
  strongest form the claim can take: the numbers we ship are the FORTRAN's, demonstrably, and not
  merely "the same numbers as far as anyone checked".
* The generator reads `COMMON/PARM7/`'s 64 chunk names out of the FORTRAN's own declaration rather
  than hardcoding them, because that order *is* the storage layout — `PT(150)`, `PD(150,9)`, `PS(150)`,
  `PDL(25,2)`, `PTL(100,4)`, `PMA(100,10)`, `SAM(100)` are 3 200 reals aliased over 64 arrays of 50,
  and getting the order wrong would shift every table after the first silently.
* **The module header now names the FORTRAN**, function by function: `glatf`, `ccor`, `ccor2`, `dnet`,
  `scalh`, `spline`, `splint`, `splini`, `densm`, `densu`, `globe7`, `glob7s`, `gts7`, `gtd7`, `gtd7d`
  each carry the name of the `SUBROUTINE` or `FUNCTION` they implement, and `g0`/`sumex`/`sg0`/`zeta`
  are its statement functions promoted to real ones.
* **The one thing the old header credited to the C turns out to be a departure from both.** It said the
  "C file-scope statics that the original uses to shuttle values between routines" were made explicit
  here. Those statics are not the C's invention — they are the FORTRAN's eight `COMMON` blocks,
  `/PARMB/`, `/MESO7/`, `/LPOLY/`, `/DMIX/` and four more, which the C transcribes as `static mut`
  because that is what a straight translation of `COMMON` produces. Threading them as locals, small
  structs and a return record is a rewrite *away* from both, and it is why this model is
  `forbid(unsafe_code)`-clean and reentrant. The header says that now.

Nothing else changed: the code is untouched, and the 17 canonical vectors pass at the same tolerance.
The zero-based indexing is the remaining trace of the C's convention, and an array base is not
expression — Rust has no other option.

## 3 · JB2008 — a licence agreement that forbids the port

Space Environment Technologies now serves a *JB2008 License Agreement* (the PDF on spacewx.com dated
2026-10-15, read 2026-10-25). It covers the "Software": the source code and the index files
`SOLFSMY.TXT` and `DTCFILE.TXT` "and any successor". It allows use only for the licensee's own
research, educational, engineering and internal purposes (§3); forbids embedding "any portion" of it
in a library given to third parties (§4); forbids modification or translation except to compile and
interface (§5); and forbids redistributing the index files "in whole or in part, as standalone data products or as part of any commercial product or service without the written permission of SpaceWx" (§6). Outputs may be
published with the acknowledgement it gives; commercial Outputs need a licence (§7).

The development tree carried a Rust translation of `JB2008.for` and a script that extracted its
tables; §5 forbids the translation even for our own use, so both are deleted. What remains is what §5
allows: `tools/jb2008-set-driver/` compiles SET's unmodified file, checked by its SHA-256, together
with a nineteen-line driver program, and the density campaign talks to that program over a pipe. SET's
file is never committed. Over the 1 440 fixture rows the driver and the deleted translation agree to
5.2e-8 relative. The two validation fixtures are excerpts of the index files, which §6 forbids
redistributing. The driver and the fixtures are excluded from the public tree (below). The earlier
reading of this section — the file states no terms, so it ships — read the file alone, and no stated
grant was never a permission.

The model itself is published (Bowman et al. 2008, AIAA 2008-6438). An implementation written from the
papers alone, with every number taken from a printed source, is a different piece of work from this
code. `spacelab-propagation/src/jacchia_bowman/` is that implementation (`plans/17` Ф18.A′, 2026-10-26):
a census mapped every number the model needs to the page that prints it, or marked it "code only"
without recording its value; a fresh agent whose only inputs were the five papers (Jacchia 1970 and
1971, Bowman et al. 2008 ×2, Tobiska et al. 2008) and that census wrote the model, with no access to
SET's file, the deleted translation or any other implementation. The code-only items are left out.
`PROVENANCE.md` beside the module gives every number's document, page and equation. It was held out
of the public tree until it was measured against the TU Delft accelerometer densities: over the
campaign's rows its MAPE is 25.5 %. It ships.

**JB2008 acknowledgement (interim).** This document prints JB2008 numbers computed with the index
files `SOLFSMY.TXT` and `DTCFILE.TXT` of Space Environment Technologies (SET) as input. JB2008 is
the model of Bowman et al. (2008), "A New Empirical Thermospheric Density Model JB2008 Using New
Solar and Geomagnetic Indices", AIAA 2008-6438. The numbers are published under §7 of SET's JB2008
License Agreement of 2026-10-15, which allows them in publications and reports with the
acknowledgement and citation of SET's JB2008 Publication Acknowledgement (Notice to Users). That
text was requested from SET on 2026-10-26; it replaces this note when it arrives.

## 4 · Orekit and `sgp4` — permissive, and the obligation is paperwork

Twenty-eight modules descend from Orekit and two from pytroll's `pyorbital`, both Apache-2.0; one
descends from the `sgp4` crate and one from Satellogic's `iquaflow`, both MIT. All four licences
permit exactly what this workspace does. Both attach conditions to *distribution*,
and neither condition was being met, because the repository root carried **no licence text at all** —
no `LICENSE-MIT`, no `LICENSE-APACHE`, no `NOTICE` — only the `license = "MIT OR Apache-2.0"` string
in `Cargo.toml`.

Apache-2.0 §4 requires a recipient to be given a copy of the licence, to have the original copyright
and attribution notices retained, and to be told that files were modified; where the original work
ships a `NOTICE` file, its attribution notices travel too. MIT requires its copyright and permission
notice to accompany substantial portions. `LICENSE-APACHE`, `LICENSE-MIT` and `NOTICE` now exist at
the root and the derived modules are named in the `NOTICE`.

One subtlety is worth stating rather than leaving implicit. `MIT OR Apache-2.0` is an offer of a
*choice*, and for the Orekit-derived files we cannot offer the MIT branch free of Apache-2.0's
attribution conditions — those are not ours to waive. The `NOTICE` says so: the dual offer stands for
the workspace, and the Apache-2.0 conditions ride along with the named files whichever branch a
downstream takes. That is the ordinary reading, and it costs a downstream nothing beyond keeping the
notice.

## 5 · GOT5.5 — the atlas we do not transcribe

The only entry here that describes work rather than an inheritance, and it is recorded because the
sweep catches its module documentation: `ocean_tide_got.rs` uses the word "transcribe" to say what it
is **not** doing.

`crates/spacelab-propagation/src/ocean_tide_got.rs` holds ocean-tide geopotential coefficients, the
same quantity as the FES2004 and FES2014b tables beside it. Those two are transcriptions of published
`ΔC̄nm/ΔS̄nm` files. This one is not: no such file is published for GOT, so
`derivations/tide_to_stokes.py` reads the *elevation* grids the atlas actually ships and performs the
spherical-harmonic analysis and the IERS 2010 Eq. 6.21 loading conversion here. The numbers are ours;
the input is a grid of measurements.

**The upstream terms, as read on 2026-09-05.** GOT5.5 and its GOT5.6 supplement are served from
`earth.gsfc.nasa.gov/geo/data/ocean-tide-models` as plain `.tar.gz` files — no licence file in the
archive, no registration, no acceptance step. The `#README.txt` asks for three papers to be cited
(Schrama & Ray 1994; Ray 1999; Ray 2013) and warns that the data must not be used for coastal
navigation. That is an attribution request and a safety warning, not a licence condition, and the
citations are in place at every use site. No stated terms means nothing to breach and nothing to
rely on either; JB2008 was once read the same way, until SET published an agreement (§3).

**Why that matters more than usual here.** The obvious alternative atlas is FES2022b, and it is a
different question: the FES grids require an AVISO account, and the terms attach at registration. §1
is what a licence accepted by use costs, so `tools/fetch-tide-atlas.sh` fetches GOT and only *prints*
what obtaining FES would involve. Registering is a decision for the user; a `curl` in a fetch script
is not the place to take it.

**What the atlas is not.** The grids are never committed — the two archives are about 1 GB and land
in `~/.cache/spacelab/tides`, outside the repository, with a `manifest.json` recording the URL, the
sha256 and the retrieval time. What this repository commits is the generator, the Maxima file that
proves its conventions, and the table it produced. `docs/verification.md` §10.8 measures that table
against both FES atlases.

## 6 · IGS broadcast navigation — a data fixture, and terms that are actually stated

`crates/spacelab-gnss/tests/fixtures/rinex3_nav_igs_brdc.24n` is not code and does not appear in the
sweep, which is exactly why it is recorded here: the sweep reads `//!` blocks in `crates/*/src`, so a
vendored *data* file is invisible to it. The rule the tree has settled on — GFZ's Hpo indices, and now
this — is that a vendored data file carries its attribution in its own bytes, where it cannot be
separated from the data.

**What they are.** Two excerpts, each verbatim and each checked byte for byte against its original
before committing — the point of these fixtures is that their columns are the format's, so one edited
by hand would defeat its own purpose.

* `rinex3_nav_igs_brdc.24n` — forty lines of `BRDC00WRD_R_20240010000_01D_MN.rnx` (2024-01-01, RINEX
  3.05 mixed navigation): the first broadcast record of each of Galileo, GPS, BeiDou and GLONASS.
* `rinex4_nav_brd400dlr.rnx` — 219 lines of `BRD400DLR_S_20262000000_01D_MN.rnx` (2026-07-19, RINEX
  4.02): one record of each of the 38 (record type, constellation, message, subtype) shapes the
  whole 148 789-line product contains. This one is DLR/GSOC's BCEmerge product, distributed through
  the same IGS channel and carrying its own DOI.
* `igs_sp3c_2026224.sp3` — the 22 header lines and first three epochs of
  `IGS0OPSRAP_20262240000_01D_15M_ORB.SP3` (2026-08-12, SP3-c, 32 GPS satellites), the IGS rapid
  orbit combination. Header verbatim, including the product's own `/*` comment block naming the
  eleven analysis centres it averages. `tests/sp3d.rs` derives an SP3-d file from it in memory and
  never writes one; the 140-satellite case in that file relabels these records across five
  constellations, so its coordinates are these coordinates and only the identifiers are the test's.

Each keeps its file's own header lines, and adds `COMMENT` lines naming the source and carrying the
acknowledgement. `BRD400DLR`'s `DOI` record and the `COMMENT` naming O. Montenbruck and
P. Steigenberger are the file's own and are kept verbatim.

**The upstream terms, as read on 2026-09-05.** The *IGS Data and Product Disclaimer and Terms of Use*
(5 August 2020) is the strongest position of anything in this register, and it is worth quoting rather
than summarising:

> The IGS products and station data are provided openly for the benefit of all scientific,
> educational, and commercial users. For 25 years, IGS data and products have been made openly
> available for use without restriction, and continue to be offered free of cost or obligation.

> **Attribution.** By accessing data, products, and any other information from the IGS, users agree to
> appropriately cite and attribute these resources to providers and their sponsors, acknowledgment of
> IGS and its contributing organizations, and to adherence to professional and ethical standards.

> **Terms of Use.** Access to, and use of IGS data, products, and other information constitutes
> acceptance of the aforementioned information.

Note the last line: this is a licence **accepted by use**, structurally the same instrument as §1's,
which is what makes it worth reading rather than assuming. The difference is entirely in what is
accepted — §1 accepts a bar on commercial use and on dissemination, and this accepts an attribution
duty and nothing else. "Commercial users" is named explicitly, so the question §1 turns on does not
arise. Attribution is discharged in the fixture's own `COMMENT` lines.

**What the content is, separately from the terms.** The records are broadcast navigation messages —
what the satellites themselves transmit. The IGS's contribution is collection and concatenation, not
authorship, so even without the terms above the copyright question would be thin. That is an argument
worth having in reserve and not one worth relying on, since the stated terms already permit the use.


## The data register

Code is half of provenance. The other half is *data*: the coefficient tables the crates embed, the
fixtures the tests read, and the products the `tools/fetch-*.sh` scripts retrieve at run time. A
licence on a table binds the crate that embeds it exactly as a licence on code does, and a
non-commercial clause on a *run-time input* binds the product that consumes it even when nothing is
committed. `cargo deny` sees none of this. The rows below are the reading of every such source, from
the artefact, with the date; `tools/check-provenance.py --data --check` fails when a fetch script
reads a host with no row, when a committed fixture or asset matches no row, or when a row matches
nothing, and `--public` additionally fails on any row still marked unread.

Verdicts: **ship** (ours, public domain, or terms with no condition); **ship+attr** (ships, and
`NOTICE` carries the attribution the terms ask for); **internal-only** (fetched into a cache outside
the tree, run as a comparator, never committed and never a dependency); **restricted** (committed or
used under terms that bind — the row says what was decided); **user-decision** (registration or an
agreement the user has to accept in person); **unread** (a claim, not a reading, until it is read).

Classes (`plans/18` §1.7, rule 18) say what may be *published*, where the verdict says what may be
committed: **`ship`** (the data may be in the public tree and the wheel; every `ship` or `ship+attr`
verdict, and only those); **`internal`** (used in the development tree, nothing derived from it is
public); **`number-public`** (the source stays out, and a number computed from it may be printed on
the terms read); **`number-private`** (the source stays out, and so does every number computed from
it). A row whose terms are unread, or wait on the user's decision, says why instead ("not read:",
"open:") and has no class until they are read. The block below repeats the class in its `class`
field, and a `number-private` row adds `names`, a case-insensitive regular expression for the words
a line names the source by. `tools/check-classes.py --check` holds the two halves equal and refuses
every public line that names a `number-private` source beside a number of ours; the public cut runs
the same refusal over the tree it writes.

| Key | What it is | Terms, as read | Verdict | Class |
|---|---|---|---|---|
| `tudelft-thermosphere` | Thermosphere neutral density and crosswind data, TU Delft (Siemes et al. 2023): the CHAMP / GRACE-FO fixtures and the satellite-months the density campaign reads | CC BY 4.0, stated in every file's own header (`gracefo_density_20190101.txt` lines 11–16), and confirmed by the data's contact, Christian Siemes, by email on 2026-10-24: CC BY 4.0 for the density and wind datasets. The same email puts TU Delft's satellite geometry, aerodynamic and radiation-pressure models under CC BY-NC-SA 4.0; spacelab uses none of them, and they are not admitted. The server has refused connections since 2026-09-05. The complete version 02 archive came from Siemes on 2026-10-25 (`Archive.zip`, 23 035 131 104 bytes, sha256 `f84072af…3a6f`); `tools/build-density-corpus.py` reads it in place into the git-ignored corpus (`verification.md` §10.20), and the twelve satellite-months fetched earlier are the same bytes | **ship+attr** | `ship` |
| `sdtm-reference` | `crates/spacelab-propagation/src/sdtm/coefficients.rs` and `crates/spacelab-propagation/tests/fixtures/sdtm_reference.csv` — SDTM-1's fitted coefficients, and this model's own values at a sample of the geometries it was fitted on, both written by `tools/fit-sdtm-corpus.py`; and the same three for the solar-wind field, `sdtm/solar_wind_coefficients.rs`, `sdtm_solar_wind_reference.csv` and `sdtm_solar_wind_hours.csv`, the last closed-form synthetic hours the generator writes | Our own numbers, fitted on `tudelft-thermosphere` (CC BY 4.0). Neither file contains a measured density, a driver series or any line of TU Delft's files; what does carry over is the attribution CC BY 4.0 §3(a)(1) requires of adapted material, discharged in `coefficients.rs`'s own header, in this row and in `docs/verification.md` §10.25. The solar-wind field's coupling map is a derived table: one knot per `ap60` class of `gfz-hpo` (CC BY 4.0), placed at the quantile of `nasa-omni`'s (CC0) solar wind. Its module header, `NOTICE` and §10.32 carry both citations | **ship+attr** | `ship` |
| `celestrak` | CelesTrak: `SW-All.csv` space-weather series, GP and SupGP element sets, the ISS ground-track TLE, satellite.js | Usage policy (read 2026-09-06, 2026-11-12, and again 2026-11-24 on the page stamped "Updated 2026 Nov 16"): rate limits — GP no more than every two hours, space weather "updates every 3 hours" and is downloaded "once per update", directories hourly, stop on any non-200 and report it to a person — and nothing about reuse or redistribution; no licence text exists. Read 2026-11-12: since 2026-03-26 a second GP download inside one two-hour update answers HTTP 403 with a message saying so, and fifty 301/403/404 answers in two hours put the address in the firewall; FORMAT defaults to OMM CSV since 2026-05-09; five-digit catalog numbers ran out on 2026-07-11, so TLE, 3LE and 2LE omit every object numbered 100000 or above; SupGP is selected by `SOURCE`; `SW-All.csv` takes its sources from the primary providers since 2026-07-06 (`tests/sw_all_reseal.rs`). `tools/fetch-supgp.sh` encodes all of it. Read again 2026-11-24: the usage policy (`https://celestrak.org/usage-policy.php`, updated 2026-11-16) and *A New Way to Obtain GP Data* (`https://celestrak.org/NORAD/documentation/gp-data-formats.php`): "CelesTrak will now (as of 2026 Mar 26) simply enforce the one-download-per-update policy for all users, starting with the Active and Starlink GROUPs", and "we now set a limit on HTTP errors (301, 403, or 404) of 50 in a 2-hour period, at which point the IP address is sent to the firewall". Seven pages of the site state nothing on reuse, redistribution, copyright or citation. The GP data are "(which we get from Space Track)" (`space-track`), and several SupGP sets are published "with permission from" their operator. Read 2026-11-24: the page asks for `SW-Last5Years.csv` in place of `SW-All.csv` ("10.8 times the size"); our scripts still fetch `SW-All.csv`. Every script in `tools/` that requests this host does it through `tools/celestrak.sh` (`solar_proxy_admissibility.py` through `fetch-density-data.sh --sw-all`): no redirect is followed, and a 301, a 403 or any other non-200 stops the script with the status, the `Location` and CelesTrak's message. `tools/fetch-density-data.sh` asks for `SW-All.csv` at most once per three-hour update, sends the copy's `ETag` and `Last-Modified` as `If-None-Match` and `If-Modified-Since`, and keeps the copy on a 304. `tools/test_celestrak_client.py` holds all of it against a local server, and fails on a script that names this host without the client | **ship** | `ship` |
| `vallado-sgp4-verification` | The AIAA 2006-6753 verification element sets and `tcppver.out`, Vallado's own SGP4 distribution as served by CelesTrak, plus python-sgp4's rendering of them | Published test cases of a public-domain algorithm; python-sgp4 is MIT | **ship** | `ship` |
| `gfz-hpo` | GFZ Hp30 / Hp60 / ap30 / ap60 — the Gannon-storm excerpts committed as fixtures, the **complete hourly series** (`Hp60_ap60_complete_series.txt`, 1985-01-01 →, 365 544 rows) fetched into the git-ignored cache for the `--ap60` fit of `verification.md` §10.15 and the solar-wind field of §10.32, whose shipped coupling map takes its knot values from the `ap60` classes (`sdtm-reference`), and the Kp/ap feed `fetch-density-data.sh` reads | `# LICENSE: CC BY 4.0` in the file header; citation lines in the same header (Matzka et al. 2024, doi:10.5880/Hpo.0003; Yamazaki et al. 2024). Re-read 2026-11-12 on the live file: the first 2 KiB are byte for byte those of the 2026-09-14 copy, version 3.0, which is the one every `ap60` fit has read (it is in the SDTM-1 data freeze) | **ship+attr** | `ship` |
| `set-jb2008-indices` | Space Environment Technologies `SOLFSMY.TXT` (F10/S10/M10/Y10) and `DTCFILE.TXT` (Dst-derived ΔTc): JB2008's drivers; 25-day excerpts committed as fixtures in the development tree | SET's JB2008 License Agreement (2026-10-15, read 2026-10-25, and again in full at source on 2026-11-24: the 3-page PDF `https://spacewx.com/wp-content/uploads/2026/10/JB2008_License_Agreement.pdf`, sha256 `71cd2c6c31f2f86ac7925c4918b4a69f202acbae168c3bb4c62c439e9a8953b4`): §1 makes the index files part of the licensed "Software", §3 licenses it for the licensee's own research, educational, engineering and internal use, and §6 forbids redistributing them "in whole or in part, as standalone data products or as part of any commercial product or service without the written permission of SpaceWx". The files' own headers say nothing about terms: `SOLFSMY.TXT` opens with four lines about its data, `DTCFILE.TXT` has no header (both read at source 2026-11-24). §1 defines "Outputs" as "the thermospheric mass densities, exospheric and local temperatures, and any other values you compute by running the Software, together with plots, tables, and analyses derived from those values". §7 lets Outputs be published "in scientific and engineering publications, reports, theses, and presentations, provided that you include the acknowledgement and citation set forth in the JB2008 Publication Acknowledgement (Notice to Users) that accompanies the Software". A commercial Output is "Redistribution of Outputs as a commercial data product or service, or the systematic redistribution of Outputs as a substitute for the Software itself", and it "requires a separate license from SpaceWx". The Notice to Users is not in the agreement, on spacewx.com/jb2008 or in the `jb2008.zip` SET serves there (read 2026-11-24). Its text was asked of SET on 2026-10-26, with our reading that the validation numbers in our documents are publications and reports under §7, not a commercial product; SET has not answered. **A number computed with these files as input is `number-public` under §7, provisional (D-38):** an interim acknowledgement in our own words ("JB2008 acknowledgement (interim)") stands once in each public document and comment block that prints one, and `tools/check-set-numbers.py --check` fails where it is missing; SET's text replaces it when it comes; no new public line of this kind is added. If SET reads these numbers as commercial Outputs, or objects, the class becomes `number-private` and the lines leave the public tree. The excerpts are excluded from the public tree, and its fetch script does not retrieve the files. Open series rebuilt to the published definitions replace them as drivers (`plans/16` Ф18.A) | **restricted** | `number-public`, provisional (D-38) |
| `iers-eop` | `finals2000A.all` (`finals.all.iau2000.txt`), IERS Earth Orientation Centre, and six excerpts of it, committed from 2026-07-10 (`spacelab-eop`'s) to 2026-09-04; an excerpt of the EOP 20 C04 series `eopc04.1962-now`, EOP Product Centre (Paris Observatory), header verbatim — cut on 2026-10-24 from the copy `astropy-iers-data` 0.2026.10.19 ships, which its `ReadMe.eopc04` says was taken from hpiers.obspm.fr, because that host timed out and datacenter.iers.org refused the request that day. Series versions: the C04 rows are the fourth update of EOP 20 C04, effective 2026-02-05 (IERS Message 547; IERS keeps the earlier series as `eopc04_20_v3`). Every `finals2000A` cut pre-dates Bulletin A's move to a USNO Kalman smoother and a weighted cubic spline on 2026-10-15 (Message 566; past values move by less than 0.05 mas and 20 µs), so its Bulletin A digits are the earlier processing's. No fixture holds a row inside the rapid solution's +30 µas x-pole bias, 2025-04-02 to 2025-11-12 (Message 546), checked 2026-11-12. `crates/spacelab-eop/tests/fixture_cut.rs` pins both `spacelab-eop` excerpts' row counts and spans, so a re-cut is a diff with a reason | IERS products are published for free use with no licence text; attribution is by citation (IERS EOP 20 C04, Bulletin A) | **ship** | `ship` |
| `iers-leap-seconds` | `leap-seconds.list`, the IERS Earth Orientation Centre's list of leap seconds (Paris Observatory), updated through Bulletin C 72 and valid until 2027-06-28, vendored as `crates/spacelab-fa/tests/fixtures/leap-seconds.list`. Taken on 2026-10-25 from the copy the IANA time-zone database serves (`data.iana.org/time-zones/data/leap-seconds.list`), byte-identical to Fedora's `tzdata-2026e`, because hpiers.obspm.fr timed out that day. sha256 `db5a895f16853b03bfc865e8d68f9fc8710ef1740e3400c701cd46a5bbbc3433`; the file's own SHA-1 line (`#h`, over the digits of its `#$`, `#@` and data lines) matches its data | The file's own words: "COPYRIGHT STATUS OF THIS FILE … This file is in the public domain." | **ship** | `ship` |
| `naif-spice` | DE440 / DE442 short-span kernels, lunar PA kernels, the LRO SPK, from NAIF | NASA/JPL, a US Government work; fetched, never vendored | **ship** | `ship` |
| `igs-bkg` | IGS final / rapid SP3 products and the multi-GNSS broadcast files via BKG; DLR/GSOC's BCEmerge merged navigation file (doi:10.57677/BRD400DLR) | *IGS Data and Product Disclaimer and Terms of Use*, 2020-08-05: open "for the benefit of all scientific, educational, and commercial users", attribution required (§6) | **ship+attr** | `ship` |
| `ilrs` | ILRS normal points (CRD), the `ilrsa` combined LAGEOS orbits, data-handling and eccentricity SINEX files, from EDC (DGFI-TUM) and ILRS/GSFC | ILRS data are open; the network asks to be acknowledged (Pearlman et al. 2019, doi:10.1007/s00190-019-01241-1). Read again 2026-11-24 for the v190 / v90 series, which add LARES-2 (IERS Message No. 567, 2026-10-29, `https://datacenter.iers.org/data/2/message_567.txt`, no terms of its own): the ILRS citation page (`https://ilrs.gsfc.nasa.gov/about/cite.html`): "The data and products are not copyrighted; however, in the event that you publish data or results using these data, … we request that you include the following citation", Pearlman et al. (2019) | **ship+attr** | `ship` |
| `itrf-ign` | ITRF2020 SLR station positions and velocities (IGN) | ITRF products are freely available; cite Altamimi et al. 2023 (doi:10.1007/s00190-023-01738-w) | **ship+attr** | `ship` |
| `icgem-xgm2019e` | XGM2019e static gravity field, the one the geopotential embeds, fetched by `fetch-specs.sh` for `spec_seals.py` | ICGEM model page: CC BY 4.0 (read 2026-09-06) | **ship+attr** | `ship` |
| `icgem-eigen-goco` | EIGEN-6S2 time-variable excerpt (fixture), EIGEN-6C4 and GOCO06s (spec seals) | CC BY 4.0 — the GFZ Data Services DOI pages of all three (10.5880/icgem.2016.008, 10.5880/icgem.2015.1, 10.5880/ICGEM.2019.002) declare `spdx.org/licenses/CC-BY-4.0` in their metadata, read 2026-09-06 | **ship+attr** | `ship` |
| `ngdc-igrf` | IGRF-14 coefficients (NCEI) | "The software code is in the public domain … may be used freely by the public" — NCEI IGRF page, read 2026-09-06 | **ship** | `ship` |
| `sedoo-fes2014b` | The FES2014b geopotential-coefficient table (GRGS / SEDOO) that `derivations/ocean_tide_fes2014.py` transcribes into `ocean_tide_fes2014.rs`, the default ocean-tide table | AVISO+ Licence Agreement, Issue 20, read 2026-09-06 from the PDF: use "for any purpose, including commercial purposes"; §3.2 restricts only re-sharing the *original, unmodified* grids on mass-distribution platforms and "does not restrict Sharing Adapted Material (Value-Added Products or Derivative Works)"; Annex A makes only the *current fields of the latest FES* non-commercial. Credit owed: "Noveltis, Legos and CLS", AVISO+, CNES | **ship+attr** | `ship` |
| `aviso-fes` | FES2014b / FES2022 elevation grids from AVISO+, for `tide_to_stokes.py`; FES2022b's heights and loading are doi:10.24400/527896/A01-2024.004 | AVISO+ Licence Agreement, Issue 20, effective 2026-08-10 (`https://www.aviso.altimetry.fr/fileadmin/documents/data/License_Aviso.pdf`, read again in full on 2026-11-24): §2.1 B grants to "produce, reproduce, and Share Adapted Material (Value-Added Products or Derivative Works), for any purpose, including commercial purposes"; §3.2: "You must not Share the Licensed Material, in its original, unmodified form, through any operational platform intended for onward mass distribution — including but not limited to a web portal, public API, or cloud data infrastructure — without prior written authorization from CNES", which does not restrict adapted material, nor a publication's figures and supplementary data; Annex A makes non-commercial only the "Currents from the latest released version of the FES ocean tide model (currently FES2022)". The FES2022 page (`…/global-tide-fes/release-fes22.html`): "The heights products can be used for any purpose (scientific, commercial,…)", and "To access data, registration is required", which is the user's act, not a script's. So the grids stay out of the public tree and the wheel, and coefficients computed from them ship with §3.1's credit | **user-decision** | `number-public` |
| `gsfc-got` | GOT5.5 / GOT5.6 cotidal grids (R. D. Ray, NASA GSFC) from which `ocean_tide_got.rs` is *computed* | No stated terms, no registration, no click-through; `#README` asks for citation and warns against navigational use (§5) | **ship+attr** | `ship` |
| `seanoe-eot20` | EOT20 cotidal grids (DGFI-TUM), the final version `85762.zip` from SEANOE (doi:10.17882/79489), from which `ocean_tide_eot20.rs` is *computed*. The archive also carries the load-tide grids, which the generator refuses by their title | SEANOE dataset page, read 2026-10-24: Creative Commons Attribution 4.0. The page also withdraws an older, corrupted release ("should not be used for any application"), offered only on request; the final version is open access. Credit owed: the dataset citation and Hart-Davis et al. (2021), ESSD 13, 3869 | **ship+attr** | `ship` |
| `tuwien-vmf` | VMF3 5°×5° grid fixture and the GPT3 grid `assets/gpt3_5.grd` (via Orekit's Apache-2.0 copy), held byte for byte against TU Wien's own `codes/gpt3_5.grd`, which `tools/fetch-specs.sh` fetches from vmf.geo.tuwien.ac.at into the git-ignored `data/specs` (Ф17.M) | VMF Data Server terms, read 2026-09-06: free of charge; products older than 24 h under CC BY 4.0; **forecast products for non-commercial scientific use only** — so an operational path uses GPT3, never the VMF forecasts | **ship+attr** | `ship` |
| `iers-conventions` | The IERS Conventions (2010) tables 5.2a, 5.2b, 5.2d, 5.3a and 5.3b, the chapter 6 FES2004 Cnm-Snm file, chapter 6 itself as updated 2018-02-01 (`icc6.pdf`) and Desai's ocean pole-tide coefficients (`desaiscopolecoef.txt.gz`), which `tools/fetch-specs.sh` fetches from the Conventions Centre (iers-conventions.obspm.fr) into the git-ignored `data/specs`; the committed nutation, CIO-locator, FES2004 and Step-2 tables are held against them (Ф17.C, Ф19.G) | The Centre's home page and chapter 5 page carry no licence, copyright or terms text (read 2026-11-13); the tables are those IERS Technical Note 36 publishes, cited as Petit & Luzum (2010) | **ship** | `ship` |
| `uber-h3` | Uber H3 v4.5.0 `src/h3lib/lib/faceijk.c` and `src/h3lib/include/constants.h`, which `tools/fetch-specs.sh` fetches into the git-ignored `data/specs`; `derivations/h3_constants.py` reads the 20 face centres, their axis azimuths and three constants from them into `derivations/h3_constants.mac`, against which `h3_definition.mac` checks the grid `h3o` builds (Ф17.P) | Apache-2.0 (`LICENSE` at `uber/h3`, read 2026-11-12). What is committed is numbers, cited with the release, not code; `h3o` itself is a dependency, under `cargo deny` | **ship** | `ship` |
| `itu-r-grids` | The ITU 1°×2° MODIP grid (`spacelab-ionosphere/assets/modip.asc`, read only by NeQuick-ITU) and the ITU-R P.834 mapping coefficients (`spacelab-gnss/assets`). The CCIR maps and the Galileo MODIP grid left this row on 2026-10-20 for `galileo-icd` | Byte-identical to the files Orekit distributes under Apache-2.0. Re-read 2026-09-14, unchanged: ITU's site-wide copyright page says "ITU holds copyright in the information available on this Web site, unless otherwise stated" and directs reproduction requests to jur@itu.int, newsroom material being the only stated exception; and the Study Group 3 ionospheric/tropospheric databank page that offers these files carries **warranty disclaimers only** — "provided 'as is' without any warranty of any kind" — and **no grant**, so the free-of-charge reading the 2026-09-06 pass left open is not supported by the page. Orekit's distribution is the lineage relied on; the question for ITU-R is in `plans/15`. The alternative — regenerate the grids from an open source — is **measured shut**: neither MODIP grid reproduces from the vendored DGRF series at any epoch (`verification.md` §5.10) | **unread** | open: ITU-R's grant is asked (`plans/15`); the files stay out meanwhile |
| `galileo-icd` | The CCIR monthly foF2/M(3000)F2 coefficient maps and the Galileo 5°×10° MODIP grid (`spacelab-ionosphere/assets/ccir*.asc`, `modipNeQG_wrapped.asc`), the files embedded in Annex C of the Galileo *Ionospheric Correction Algorithm for Galileo Single Frequency Users*, issue 1.2, © European Commission 2016; and `spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv`, whose inputs are its Annex E cases and whose STEC column is the output of `nequick-jrc` | The GSC terms of use, read 2026-10-20 at `gsc-europa.eu/terms-of-use`: "Unless otherwise stated, downloading, reproduction and use of all the materials and documents published on the Website are authorised provided the source is acknowledged as follow: © EU 2011-2026", except material identified as a third party's copyright, and logos. The ICD's own terms page is disclaimers only and identifies no third-party copyright in Annex C. The maps descend from CCIR's (ITU's predecessor) foF2 coefficients, and ITU's own P.531 copies hold the same values. Orekit's copies, which this crate embedded until 2026-10-20, differ from both in the eighth digit and carry a corrupt last F2 coefficient in July to December (up to 4.2 TECU, `tools/nequick-jrc-oracle/README.md`) | **ship+attr** | `ship` |
| `nequick-jrc` | NeQuickG JRC 1.0, the reference implementation of the Galileo ICD (European Union, 2019), cloned by `tools/nequick-jrc-oracle/run.sh` from the `odrisci/NeQuickJRC` mirror | EUPL-1.2, copyleft: run as an oracle, never linked or vendored; only its outputs are committed | **internal-only** | `number-public` |
| `ccsds-examples` | The example OPM / OEM / CDM messages from the CCSDS Blue Books (502.0-B-3 annex G, 508.0-B-1) | The standard's own worked examples, reproduced as format-conformance fixtures. The front matter of 502.0-B-3 (read 2026-09-06 from `public.ccsds.org`) carries no copyright line and no reproduction restriction — only "published and maintained by the CCSDS Secretariat"; the same position as GOT, recorded rather than assumed | **ship** | `ship` |
| `gps-almanac` | SEM and YUMA almanacs (week 384) | US Coast Guard Navigation Center, a US Government work | **ship** | `ship` |
| `oracle-outputs` | Fixtures *we generated* by running an oracle: Basilisk (ISC), NASA 42 (NOSA 1.3, run unmodified), pyorbital (Apache-2.0), Orekit (Apache-2.0), pyerfa (BSD-3), skyfield (MIT), python-sgp4 (MIT), qhull, NEPTUNE (Apache-2.0), GRGM1200A (NASA PDS), and our own generators (air mass, twilight, slanted edge, gas transmittance, the ONNX cloud fixture, the hand-built RINEX observation file, the DSST coefficients' closed forms of `derivations/dsst_coefficients.mac`), and the validated enclosures TaylorModels.jl (MIT) computes in `derivations/taylor_model_enclosures.jl` | The numbers are the output of a run and are ours; each file names its generator and the oracle's licence in its header where the format allows a comment | **ship** | `ship` |
| `pyspectral-lut` | libRadtran Rayleigh reflectance LUT via pyspectral v1.0.1, and `rayleigh_doubling.txt`, our solution on its geometry grid and depths | CC BY 4.0, Zenodo doi:10.5281/zenodo.1288441 (file header) | **ship+attr** | `ship` |
| `aeronet` | AERONET Version 3 Level 2.0 almucantar inversions (42 sites) | AERONET Data Use Policy: acknowledge the site PIs and the network, and contact the PI before publishing on the data (file header names the PIs) | **ship+attr** | `ship` |
| `radcalnet` | RadCalNet surface/TOA reflectance and atmospheres, 42 scenes plus hold-out days | RadCalNet Data Policy v1.1, quoted in full in the file header: "freely and publically available"; acknowledge "the CEOS WGCV RadCalNet service … and in particular from site X and Y". Portal credentials are read from the environment by the generator and are never in the tree | **ship+attr** | `ship` |
| `mpi-mainz-ozone` | Ozone absorption cross-section, MPI-Mainz UV/VIS Spectral Atlas | Citation requested, no other condition (read 2026-09-06) | **ship+attr** | `ship` |
| `pleiades-s2p-rpc` | One real RPC00B camera model (Pleiades over Réunion) taken from the s2p test image's metadata | s2p (Centre Borelli) is **BSD-3-Clause** — GitHub licence API, 2026-09-06 — not AGPL as a first pass assumed; its `tests/data/input_pair` holds the images and a config with no separate notice. The ninety coefficients are the camera's facts read from the image header; the BSD notice travels in `NOTICE` for the distribution they were read from | **ship+attr** | `ship` |
| `cara-cdms` | NASA CARA's eleven "Alfano" sample CDMs | NASA Open Source Agreement notices in the repository, no LICENSE file; fetched, never vendored (script header) | **internal-only** | `number-public` |
| `hwm14` | NRL Horizontal Wind Model 14, software and coefficients | CC BY-NC-ND (Drob et al. 2015 supporting information): no derivatives, no commercial use — run as an oracle, never ported or linked (script header) | **internal-only** | `number-public` |
| `neptune` | NEPTUNE propagator (TU Braunschweig), a fourth independent oracle | Apache-2.0; deliberately not linked (script header) | **internal-only** | `number-public` |
| `nrlmsise00-fortran` | `nrlmsise00_sub.for`, the NRL FORTRAN that `derivations/nrlmsise00_tables.py` reads | US Government work, 17 U.S.C. §105 (§2) | **ship** | `ship` |
| `hitran` | HITRAN line parameters behind the gas-transmittance fit | Free with citation; the lines are cached under `~/.cache/spacelab/hitran` and never enter the tree — only our exponential-sum fit does | **internal-only** | `number-public` |
| `mt-ckd` | AER MT_CKD water-vapour continuum | Research-only; "may not be incorporated into proprietary software … or commercial software … without the express written consent of AER" — oracle, cached outside the tree, nothing derived from it committed | **internal-only** | `number-public` |
| `nrlmsis2` | NRLMSIS 2.0 / 2.1 via `pymsis` | NRL licence: no non-academic commercial use, no modification, no dissemination; US 10,641,925 B2 (family: EP withdrawn, AU granted, US to 2039) — validation only | **internal-only** | `number-private` (D-30 (c): unpublished unless its terms are read to allow it) |
| `dtm2020` | DTM2020 code and `DTM_2020_F107_Kp.dat` | §1 — excluded from the public tree; research use only. A number computed from it is `number-private` (D-32): the public tree prints none, and `tools/check-classes.py --check` fails where it would. Tag `v0.3.0` already carries some, one line of them in the published `spacelab-propagation` 0.3.0 (D-58, §1) | **restricted** | `number-private` (D-32) |
| `wdc-kyoto` | WDC Kyoto Dst / AE / SYM-H | "The WDC Kyoto does not allow commercial applications of the geomagnetic indices" (read 2026-09-05). Not fetched by any script | **restricted** | `internal` |
| `esa-swarm-dns` | ESA's own Swarm Level-2 thermospheric density products (`SW_OPER_DNSxPOD_2_`, `DNSxACC_2_`) on `swarm-diss.eo.esa.int` — 4608 daily files per satellite, 2013-11-28 → 2026-07-31, 643 MB for Swarm A alone. The fallback for `tudelft-thermosphere`, whose server has refused connections on five probes between 2026-09-05 and 2026-09-14 | Read 2026-09-14 from the two documents the download page itself links. The *Data Policy* (ESA/PB-EO(2010)54) is permissive in intent — use "be it scientific, for public good or **commercial**", "full and open access … free of charge" — but access is "granted upon electronic on-line registration and the user accepting the ESA Terms & Conditions", and those (ESA-EOPG-PDGS-PR-1 Issue 1.1, 2023-09-20) add three clauses no register row can decide: **B.6(a)** "Use all ESA EO data accessed exclusively for their own EO research and applications purposes"; **C.2** mark "all ESA EO data and analysed information … © ESA (year of reception)"; **C.5** any IP emerging from direct use carries a "worldwide, free of charge, irrevocable, and non-exclusive licence" to ESA, with patent filings notified within two months. A fitted coefficient set is plausibly "analysed information" and certainly "own intellectual property" | **user-decision** | open: the user's decision on ESA's Terms & Conditions (B.6(a), C.2, C.5) |
| `usgs-dst` | USGS Geomagnetism Program web service, virtual observatory `USGS`, elements `Dst3` ("3-channel Dst") and `Dst4` ("4-channel Dst") as `/ws/elements/` lists them; method Rigler (2017), doi:10.3133/ofr20171037. The 2026-09-06 probes asked for `DST`, which is not an element, and read its all-`null` answer as an empty index. `tools/fetch-usgs-dst.sh` fetches the hourly series from the filter endpoint, `/ws/algorithms/filter/` with `type=variation`, `input_sampling_period=60` and `output_sampling_period=3600`, one calendar month per request, sequentially, into the git-ignored `data/validation/density/usgs-dst/` — the route and the load USGS (E. J. Rigler) gave and agreed to on 2026-10-27; `/ws/data/` holds the 1-minute series only. The Gannon-storm week is committed as `usgs_dst_gannon.txt` | U.S. public domain: "USGS-authored or produced data and information are considered to be in the U.S. Public Domain", and credit to "U.S. Geological Survey" is asked (usgs.gov, *Copyrights and Credits*, read 2026-10-25); the fixture header carries the credit and the citation. Fetched 2026-10-28, 2015-09 → 2026-10, 97 896 hours; values from 2015-09-24. The `variation` product carries unscreened artefacts (`verification.md` §10.7), reported to USGS | **ship** | `ship` |
| `nasa-omni` | NASA/GSFC SPDF's OMNI2 hourly solar wind and IMF at the bow-shock nose, merged from ACE, Wind, IMP 8 and others and time-shifted from L1 (King & Papitashvili 2005, doi:10.1029/2004JA010649). `tools/fetch-omni.sh` fetches the yearly `omni2_YYYY.dat` files into a temporary directory and writes only the IMF and plasma columns to the git-ignored `data/validation/density/omni2_sw_hourly.txt`, for the solar-wind driver experiment of `verification.md` §10.31 and the solar-wind field of §10.32, whose shipped coupling map places its knots at the quantiles of these hours (`sdtm-reference`). The same files carry WDC Kyoto's Dst, AE, AL and AU (`wdc-kyoto`), GFZ's Kp and ap, SILSO's sunspot number, NRC's F10.7 and DTU's PCN: none is written | The SPASE record `spase://NASA/NumericalData/OMNI/PT1H` (doi:10.48322/1shr-ht18; read 2026-10-30 in the `hpde/NASA` repository on GitHub) lists `CC0-1.0` for the `low_res_omni` files, "the Creative Commons license applicable to all publicly available NASA Heliophysics data products". NASA's Heliophysics data policy (HPD-SDMP v2.2, effective 2025-02-14, section 2, item 2 g) releases data "with a Creative Commons Zero license" where nothing else restricts them. OMNIWeb asks for an acknowledgement and the King & Papitashvili citation, which the file header carries. The grant covers NASA's columns, not the indices other providers supplied, so those are dropped at fetch | **ship** | `ship` |
| `gcat` | J. McDowell's General Catalog of Artificial Space Objects, `satcat.tsv`: the decay dates, types and launch tags of `tools/lifetime-campaign.py` (`docs/verification.md` §10.37). Fetched by `tools/fetch-reentry-elements.py` under gitignored `data/`, never vendored | The page (read 2026-11-01): "GCAT is released under the Creative Commons CC-BY licence", the licence linked as CC-BY-4.0; "You are free to use the data in GCAT as long as you include a citation to it". Cite: McDowell, J. C., 2020, General Catalog of Artificial Space Objects, https://planet4589.org/space/gcat | **ship+attr** | `ship` |
| `mcdowell-elements` | McDowell's historical TLE archive, `planet4589.org/space/elements`, one file per object: the element sets the same campaign starts from. Fetched by the same script at one request a second, under gitignored `data/` | No licence stated on the archive's pages (read 2026-11-01), and the GCAT notice names GCAT. No terms is no grant, so the files stay on this machine; only numbers derived from them leave it | **internal-only** | `number-public` |
| `space-track` | Space-Track.org GP catalogue | User agreement binds per account; never vendored, never used by a fetch script — CelesTrak serves anything committed | **user-decision** | not read: the terms bind per account |
| `cddis` | NASA CDDIS (IGS/ILRS archive) | Earthdata login; the terms attach at registration | **user-decision** | not read: the terms attach at Earthdata registration |
| `noaa-wam-ipe` | WAM-IPE neutral density on the NOAA Open Data Dissemination bucket `noaa-nws-wam-ipe-pds`: the operational `v1.2/wfs` `wam_fixed_height` files `tools/fetch-wam-ipe.sh` selects and `tools/wam-ipe.sha256` pins, read by `wam_ipe.rs` and scored in `measured_density.rs`. Nothing from them is committed | The AWS Open Data registry page, read 2026-10-24: "NOAA data disseminated through NODD are open to the public and can be used as desired"; NOAA "requests attribution for the use or dissemination of unaltered NOAA data", and "it is not permissible to state or imply endorsement by or affiliation with NOAA". The bucket's `NON-OPERATIONAL_RESEARCH/` reruns ask users to "contact the WAM-IPE team … before accessing such files", so the fetch script does not read them. Credit: `wam_ipe.rs`'s header and `docs/verification.md` §10.18 | **ship+attr** | `ship` |
| `noaa-goes-euvs` | GOES-R EUVS daily science products (NOAA/NCEI): He II 30.4 nm, H I Lyman-α 121.6 nm and the Mg II core-to-wing index, GOES-16/17/18/19 | The files' own `license` global attribute, read 2026-09-14: *"These data may be redistributed and used without restriction."* LISIRD's catalogue states CC0 1.0 for the same products. A grant with no condition, so nothing is owed but the citation we give anyway | **ship** | `ship` |
| `sdo-eve` | SDO/EVE Level-3 merged daily product (LASP/CU for NASA): the ESP 27.16–33.8 nm band, the MEGS-P Lyman-α diode, the He II 30.4 nm line | `license` attribute of the file, read 2026-09-14: `CC0 1.0 https://creativecommons.org/publicdomain/zero/1.0/`; LISIRD's catalogue agrees. Acknowledgement asked for in the file (`Courtesy of NASA/SDO and the AIA, EVE, and HMI science teams.`) and given in `docs/verification.md` §10.9; it joins `NOTICE` when something derived from it ships, which is `NOTICE`'s own scope | **ship** | `ship` |
| `lisird-lyman-alpha` | The composite solar Lyman-α irradiance (Machol et al. 2019, version 4; LASP, with NOAA NCEI), as served by LISIRD: `composite_lyman_alpha` | **CC0 1.0**, stated three ways at source (read 2026-10-31): the catalogue (`https://lasp.colorado.edu/lemr/lisird/lisird`) carries `lds:composite_lyman_alpha lds:license lds:cc0` ("Creative Commons - CC0 1.0 Universal", acknowledgement "Machol et al. 2019, doi:10.1029/2019EA000648"); the site's release notes read "Version 4.26.1 October 28, 2026 … Add CC0 license to Composite Solar Lyman-alpha"; the data directory `https://lasp.colorado.edu/data/timed_see/composite_lya/` holds `LICENSE.txt` — the CC0 1.0 legal code, 7 049 bytes, sha256 `7179683e8000e6bdc9bbc60d85edf0a4ac8e76f951857f54fcb775d5886f1309` — and `README.md` with the citation (doi:10.25980/ZR1T-6Y72). The grant followed our letter of 2026-10-24 (`plans/15`: Machol, Woodraska; the file was placed 2026-10-26). CC0 owes nothing; the citation and the acknowledgement are given anyway, and `tools/solar_proxy_admissibility.py` prints the licence's URL and digest beside the series | **ship** | `ship` |
| `lisird-composites` | As served by LISIRD and **not** admitted: the Viereck composite Mg II, SORCE Mg II, SOHO/SEM, and the Penticton / NOAA / CLS radio-flux series (the composite Lyman-α left this row on 2026-10-31, see `lisird-lyman-alpha`) | LISIRD publishes a machine-readable catalogue (`https://lasp.colorado.edu/lemr/lisird/lisird`) in which each dataset either names a licence or does not. Read 2026-09-14: these name **none** — 354 of the 420 datasets name none, against 41 CC0 and 25 CC BY 4.0. No licence is not a permissive licence, so they are read as oracles and nothing derived from them ships. (`lasp_filtered_drao_f107` *is* CC BY 4.0 and is the admissible alternative source of F10.7 if CelesTrak's ever closes) | **internal-only** | `internal` |
| `bremen-mgii` | The Bremen composite Mg II index (1978–) and its F30-extended form (1947–), IUP Bremen | The page (read 2026-09-14, moved to `https://www.iup.uni-bremen.de/UVSAT/data/`) and both data files carry citations and a contact address and **no terms of any kind**; LISIRD's catalogue also states none for its copy. Oracle only, and the letter asking IUP for a licence is in `plans/16` Ф14.G | **internal-only** | `internal` |
| `gold-on2` | NASA GOLD Level-2 column O/N₂ (`ON2`, V04 R02, netCDF; DOI 10.48322/wjrk-dk56; ATBD Correira et al. 2021, doi:10.1029/2021JA029517), from `https://spdf.gsfc.nasa.gov/pub/data/gold/level2/on2/`: the validation of SDTM-1's column O/N₂ (Ф23.P), never training | SPDF data-use policy (`https://spdf.gsfc.nasa.gov/data_use_policy.html`, read 2026-11-11): "public domain and made available with a Creative Commons Zero (CC0) dedication … even for commercial purposes, all without asking permission"; the SPASE record names CC0 1.0 and the acknowledgement "National Aeronautics and Space Administration; Dr. Richard Eastes". The mission's Rules of the Road (`https://gold.cs.ucf.edu/data/gold-data-usage/`) are norms, not a licence: acknowledge the source, send the PI each manuscript at submission, label a value-added product as different from the PI's. Release Notes Rev 5.12 flag ON2 discontinuities within ±10 days of the 2024-10-07 grating shift and contamination poleward of 60° magnetic latitude | **ship+attr** | `ship` |
| `guvi-on2` | TIMED/GUVI column O/N₂, Level 3 only (`timed_guvi_l3-on2_*_Av0100r000.nc`, v01 r00, spectrograph mode since the 2007-12-07 scan-mirror failure; DOI 10.48322/nvfx-3509), from `https://guvitimed.jhuapl.edu/data/level3/level3_on2_spect/netcdf/`: Ф23.P's second validation source | The GUVI Data Usage Policy (`https://guvitimed.jhuapl.edu/home_guvi-datausage`, read 2026-11-11) is the only text and is not a licence: "Mission scientific and model results are open to all"; co-authorship is to be offered to the PI and may be declined; the acknowledgement it requires names NASA MO&DA, The Aerospace Corporation, The Johns Hopkins University and the PI, Dr. Larry J. Paxton. The SPASE record holds no rights statement and the product is not at SPDF. Redistribution of the files is not granted, so they are read as an oracle under gitignored `data/` and only statistics against them are published | **internal-only** | `number-public` |
| `tianmu-ngms` | TianMu-1 NGMS in-situ O, N₂ and O/N₂ near 500 km for October 2024, the plot data of Ai et al. (2026, *Earth Space Sci.* 13(9), e2026EA005176) on Zenodo (`https://zenodo.org/records/20195710`, doi:10.5281/zenodo.20195710, eight `.mat` files): Ф23.P's in-situ check | The record's licence field is `cc-by-4.0` (read 2026-11-11); the article itself is CC BY-NC-ND 4.0, which binds its text and figures, not the record. `figure10-11.mat` also holds NRLMSIS 2.0 output, which `nrlmsis2` forbids us to disseminate: those columns are never read or copied. The NGMS O/N₂ is a local ratio, not the column O/N₂ | **ship+attr** | `ship` |
| `storm-ai` | The STORM-AI challenge dataset (Sanchez-Hurtado et al., Harvard Dataverse doi:10.7910/DVN/U6K6MJ, version 1.1, the twelve `private_eval__*` archives; `tools/fetch-storm-ai.sh`): the initial states of its 4 557 evaluation cases and the TU Delft orbit-mean densities they are scored on, for Ф25.E. `docs/storm-ai-split.json` names each case — id, satellite, span, subsets — and the archives' MD5s, never a density | The record's licence is CC BY 4.0 (read 2026-10-31, again at the Dataverse API 2026-11-12); the devkit is MIT. The archives and the targets stay under `data/`. The manifest is our list of the cases with the dataset's DOI | **ship+attr** | `ship` |
| `aether-p3` | The twelve AETHER-P3 Nowcast benchmark cases (Wang & Bai 2026, arXiv:2609.06777, Table 7, dates read as whole UTC days) in `docs/aether-p3-cases.json`, and its Table 9 relative errors quoted beside ours by `tools/benchmark-splits.py` | The paper is CC BY 4.0 (read 2026-11-12); its code is MIT and is not used. The cases are dates and satellites; the quoted numbers are cited, not derived | **ship+attr** | `ship` |
| `pyfes` | PyFES / LIBFES, CNES's tide-prediction library (`CNES/aviso-fes` on GitHub; `pyfes` 2026.10.0 on PyPI): the comparator of `ocean_tide_admittance.rs` (`experiments/ocean_tide_admittance_verify.py`) and of the inferred minor amplitudes `ocean_tide.rs` holds as test literals. Not vendored | `LICENSE` at `https://raw.githubusercontent.com/CNES/aviso-fes/main/LICENSE` (read 2026-11-24): "BSD 3-Clause License / Copyright (c) 2025, CNES / All rights reserved. / Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met"; PyPI's metadata (`https://pypi.org/pypi/pyfes/json`) says `BSD-3-Clause`. The "GPL/AGPL" label of `plans/17` §7 was wrong twice over: the legacy C tag 2.9.7 (2024-07-22) carries `COPYING`, the GNU LGPL 3.0, and BSD-3 holds from tag 2024.11.0. A port would carry CNES's notice in `NOTICE` | **ship** | `ship` |
| `tiegcm` | NCAR HAO's thermosphere-ionosphere GCM (`NCAR/tiegcm` on GitHub): a candidate comparator, not run (`verification.md`, "Not run, each with its reason") | `LICENSE` at `https://raw.githubusercontent.com/NCAR/tiegcm/master/LICENSE` (read 2026-11-24), the "NCAR TIE-GCM OPEN SOURCE ACADEMIC RESEARCH LICENSE AGREEMENT": §1 grants use, reproduction and derivative works "for research, academic, and non-profit purposes only"; §3a: "You shall not sell, or license, or transfer for a fee the Software, or any work that in any manner contains the Software, or any data products generated by the Software", and "The Software may not be used for operational purposes, i.e., any applications intended to create forecast, nowcast, or hindcast products"; §3d binds notification, citation and co-authorship to the CEDAR and TIMED "rules of the road". Stricter than the 2026-11-11 summary: every operational use is out, not only hindcasts, and so is any fee for its output | **user-decision** | open: research, academic and non-profit use only; whether a comparison by this project is that is the user's decision |
| `tudelft-lambda` | TU Delft's LAMBDA 4.0 MATLAB toolbox for integer ambiguity resolution, and its paper (Massarweh, Verhagen & Teunissen, GPS Solut. 29:14, 2025, doi:10.1007/s10291-024-01738-z): `spacelab-gnss`'s `lambda.rs` works from the paper (Ф24.D); the toolbox is neither run nor vendored | The terms page (`https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/research/lambda/terms-and-conditions`, read 2026-11-24): "LAMBDA and Ps-LAMBDA are available free of charge to be used by any individual and for any purpose except where specific restrictions are indicated", and "we do not allow the redistribution of LAMBDA and Ps-LAMBDA, which must be obtained exclusively from this website"; it names MATLAB 3.0 and Python 1.0, not 4.0. The download page offers 4.0 through a form and names no licence, and a licence file inside the package is unread. The paper is CC BY 4.0, which covers its text, not the code | **user-decision** | not read: the 4.0 package's own terms sit behind TU Delft's download form, the user's act |
| `ckdmip` | ECMWF's CKDMIP line-by-line spectra and fluxes (Hogan & Matricardi 2020, GMD 13:6501, doi:10.5194/gmd-13-6501-2020): LBLRTM v12.8 output with AER's line parameters and MT_CKD 3.2, the open gas-optics route of Ф26.D. Not fetched | The dataset's `README` (`https://aux.ecmwf.int/ecpds/home/ckdmip/README`, read 2026-11-24, 821 bytes): "The contents of the directories "concentrations", "lw_spectra", "sw_spectra", "lw_fluxes" and "sw_fluxes" are free to use for any purpose, but it is requested that any forthcoming publications cite the Hogan and Matricardi (2020) paper. The contents of the "results" directory are for use in the CKDMIP project only." Neither the README, the project page nor the paper states AER's terms on the spectra, which are the output of LBLRTM and MT_CKD, whose notice (`mt-ckd`) reaches "any modified version" | **internal-only** | `number-public` (D-46 (a): comparator only; shipping is the user's decision with these texts) |
| `rrtmgp-data` | RRTMGP's k-distribution tables (`earth-system-radiation/rrtmgp-data`, Zenodo doi:10.5281/zenodo.7988260), whose shortwave minor absorbers include the water continuum, `h2o_self` and `h2o_frgn`. Not fetched | `LICENSE` at `https://raw.githubusercontent.com/earth-system-radiation/rrtmgp-data/main/LICENSE` (read 2026-11-24): "BSD 3-Clause License / Copyright (c) 2015-2018, Atmospheric and Environmental Research and Regents of the University of Colorado": AER, which owns the continuum, licenses its k-space form under BSD-3. The code, `rte-rrtmgp`, is BSD-3 too (copyright 2015-2025, Columbia added) | **internal-only** | `number-public` (D-46 (a): comparator only; shipping is the user's decision with these texts) |
| `ecckd` | ECMWF's ecCKD and the CKD definition files ecRad ships (`ecckd-*.nc` in `ecmwf-ifs/ecrad`'s `data/`), fitted to CKDMIP's spectra. Not fetched | `LICENSE` of `ecmwf-ifs/ecckd` (read 2026-11-24): Apache License 2.0; its `NOTICE`: "(C) Copyright 2019- ECMWF. (C) Copyright 2003-2015 The University of Reading." ecRad's `LICENSE` is Apache-2.0 too, and its `data/` holds the `ecckd-*.nc` files with no licence or README of their own: they are Apache-2.0 because the repository is, and no notice ties them to CKDMIP, LBLRTM or MT_CKD | **internal-only** | `number-public` (D-46 (a): comparator only; shipping is the user's decision with these texts) |
| `tsis1-hsrs` | The TSIS-1 Hybrid Solar Reference Spectrum, version 2 (LASP; Coddington et al. 2023, doi:10.1029/2022EA002637), as LISIRD serves it: the solar spectrum of the radiance and SNR chain. Not fetched | LISIRD's catalogue (`https://lasp.colorado.edu/lemr/lisird/lisird`, read 2026-11-24): `lds:tsis1_hsrs_p1nm … lds:license lds:cc_by_4.0 … lds:version "2"`, and `lds:cc_by_4.0` is "Creative Commons - Attribution 4.0 International". The dataset pages are a JavaScript shell with no terms. Credit: Coddington et al. (2023) and (2021, doi:10.1029/2020GL091709) | **ship+attr** | `ship` |
| `gpw-v4` | CIESIN's Gridded Population of the World v4.11, population density (doi:10.7927/H49C6VHW) and count (doi:10.7927/H4JW8BX5): the population table of the casualty expectation (Ф31.R). Not fetched | NASA's CMR record (`https://cmr.earthdata.nasa.gov/search/collections.umm_json?doi=10.7927/H49C6VHW`, read 2026-11-24), `UseConstraints`: "This work is licensed under the Creative Commons Attribution 4.0 International License (https://creativecommons.org/licenses/by/4.0). Users are free to use, copy, distribute, transmit, and adapt the work for commercial and non-commercial purposes, without restriction, as long as clear attribution of the source is provided." The count record says the same, and the Earthdata page "openly shared, without restriction". SEDAC's own record refused the connection on 2026-11-15 and again on 2026-11-24 | **ship+attr** | `ship` |
| `silso` | WDC-SILSO's sunspot number (Royal Observatory of Belgium), and the 13-month smoothed Rz12 that IRI's `ig_rz.dat` carries; OMNI's copy is dropped at fetch (`nasa-omni`). Never vendored: Rz12 and IG12 are caller inputs (Ф14.U) | `https://www.sidc.be/SILSO/datafiles` (read 2026-11-24): "our sunspot data can be freely downloaded (Licence: CC BY-NC : https://creativecommons.org/licenses/by-nc/4.0)", the 13-month smoothed series included, with credit asked in any publication ("Source: WDC-SILSO, Royal Observatory of Belgium, Brussels", doi:10.24414/qnza-ac80). The Observatory's data policy (`https://robinfo.oma.be/en/legal-notices/data-policy`) is stricter: "Data presented by the ROB, may only be used for private, educational or scientific purposes", and commercial use is "only possible after the express written prior consent of the Director general of the ROB" | **restricted** | `internal` |
| `goco2025s` | GOCO2025s, the GOCO consortium's satellite-only gravity field to degree 300, with secular, annual and semi-annual terms to degree 200 in the same file (TU Graz; ICGEM's longtime table): Ф20.E's field. Not fetched | TU Graz's repository record (`https://repository.tugraz.at/doi/10.3217/f48p8-8h651`, read 2026-11-24): "Creative Commons Attribution 4.0 International"; DataCite's metadata for doi:10.3217/f48p8-8h651 gives `cc-by-4.0`. ICGEM's table has no licence column. Credit: Oehlinger et al. (2025) | **ship+attr** | `ship` |
| `gfz-aod1b` | GFZ's AOD1B Release 07, the non-tidal atmosphere and ocean de-aliasing product of GRACE and GRACE-FO (dataset doi:10.5880/GFZ.1.3.2022.003; Shihora et al. 2022, doi:10.1029/2022JB024360): Ф20.L's input. Not fetched | DataCite's metadata for the dataset DOI (`https://api.datacite.org/dois/10.5880/gfz.1.3.2022.003`, read 2026-11-24): rights "Creative Commons Attribution 4.0 International", `cc-by-4.0`. GFZ Data Services' landing page rendered its title only, and ISDC's download page (`https://isdc.gfz.de/esmdata/aod1b/`) states no licence: the reading rests on the DOI's registered metadata | **ship+attr** | `ship` |
| `gfz-hpo-forecast` | GFZ's experimental hourly 72-hour Hp30 / Hp60 forecast (`https://isdc-data.gfz.de/geomagnetism/HpoForecast/`, v0102), which Ф18.B-7′ would archive. Not fetched | Read 2026-11-24: the v0101 `Readme.md` and the v0102 listings state no licence, copyright, citation or DOI. `https://spaceweather.gfz.de`'s disclaimer: "The data/data products are provided “as-is” without warranty of any kind either expressed or implied", "The entire risk as to the quality and performance of the Data/data products is with the Licensee", and its footer "© 2025 spaceweather.gfz.de | Alle Rechte vorbehalten". No grant, so the issues stay on this machine and only scores computed against them leave it, as for `mcdowell-elements`. The Hpo index's CC BY 4.0 (`gfz-hpo`) does not cover the forecast | **internal-only** | `number-public` |
| `igs-antex` | The IGS antenna phase-centre file `igs20_2444.atx` (IGSMAIL-8778, 2026-11-09; `https://files.igs.org/pub/station/general/`), for Ф24.I-1's ANTEX reader. Not fetched | The mail (`https://lists.igs.org/pipermail/igsmail/2026/008774.html`) and the directory carry no terms. *IGS Data and Product Disclaimer and Terms of Use*, 2020-08-05 (`https://igs.org/wp-content/uploads/2020/09/IGS-Data-and-Product-Disclaimer-and-Terms-of-Use-200805.pdf`, read 2026-11-24): "The IGS products and station data are provided openly for the benefit of all scientific, educational, and commercial users"; users "agree to appropriately cite and attribute these resources to providers and their sponsors"; "IGS data and products should not be used for any life-critical functions" | **ship+attr** | `ship` |
| `imcce-vsop87` | VSOP87 (Bretagnon & Francou 1988, A&A 202, 309), IMCCE / Paris Observatory (`https://ftp.imcce.fr/pub/ephem/planets/vsop87/`), for Ф7.L. Not fetched | IMCCE's legal notice (`https://www.imcce.fr/mentions-legales`, read 2026-11-24): "The data contained in the Web and FTP servers of the IMCCE can be used as an example to illustrate documents if the source is clearly identified"; "Requests for the reproduction of data or publications from the site of the IMCCE are made using the form provided for this purpose"; its software "can be freely used for private purposes or for educational purposes. Caution, any commercial or professional use is subject to authorization". The VSOP87 `README` states no terms. So the series files stay out until IMCCE authorises a reproduction, which is the user's request, and numbers computed from them may be shown with the source | **restricted** | `number-public` |
| `elpmpp02` | ELP/MPP02, the lunar series of Chapront & Francou, for Ф7.L | Not read on 2026-11-24: the series is no longer under `https://ftp.imcce.fr/pub/ephem/moon/` (HTTP 404 on `…/elpmpp02/`), and the location search results give, `cyrano-se.obspm.fr/pub/2_lunar_solutions/2_elpmpp02/`, refused the connection. IMCCE's legal notice (`imcce-vsop87`) is the likely frame, but no page ties it to this series | **unread** | not read: no page that states its terms loaded |
| `gaia` | ESA's Gaia data releases, DR4 among them, for `spacelab-astrometry` (Ф26.A); UCAC4 is the default and the commercial catalogue (D-20). Not fetched | ESA's Gaia licence page (`https://www.cosmos.esa.int/web/gaia-users/license`, read 2026-11-24): "Gaia data are distributed under the CC BY-NC 3.0 IGO license"; ESA's archive terms (`https://www.cosmos.esa.int/web/esdc/terms-and-conditions`): "Prior to any commercial use by the User of any Data or Data Product, including any use or application that directly or indirectly generates a financial gain, a detailed request for authorisation/licence shall be made by the User"; credit "Credit: ESA, Gaia DPAC". The release page: "Gaia DR4 (based on 66 months of data) 2 December 2026" | **restricted** | `number-public` |
| `itu-r-recs` | ITU-R Recommendations P.618, P.676, P.838 and P.840 (Earth–space attenuation) and Study Group 3's validation examples `CG-3M3J-13-ValEx-Rev8.3.0.xlsx`, for the atmospheric loss of a link (Ф26.J). Not fetched | Read 2026-11-24. Free to read: "Free online access to all current ITU-R Recommendations and Reports is now provided to the general public" (`https://www.itu.int/pub/R-REC`). Not free to reproduce: P.618-14's front matter, "No part of this publication may be reproduced, by any means whatsoever, without written permission of ITU"; ITU's website terms (`https://www.itu.int/en/about/Pages/terms-of-use.aspx`) allow download, copy and use "for personal, educational, or non-commercial purposes provided that you acknowledge the source", and "You may not modify, reproduce, distribute, sell, transmit, create derivative works or use the content for any commercial purpose without obtaining prior written permission from ITU"; the SG3 page (`https://www.itu.int/en/ITU-R/study-groups/rsg3/Pages/iono-tropo-spheric.aspx`): "These validation examples are provided "as is" without any warranty of any kind", under "© ITU All Rights Reserved", and no grant. So the texts, their coefficient tables, the digital maps and the spreadsheet stay out of the tree until ITU's written permission, the user's request (as for `itu-r-grids`); the terms say nothing of numbers computed by the formulas, which are not ITU's content | **restricted** | `number-public` |

```text
tudelft-thermosphere | ship+attr | read 2026-10-24 | hosts thermosphere.tudelft.nl | paths crates/spacelab-propagation/tests/fixtures/champ_density_sample.txt crates/spacelab-propagation/tests/fixtures/gracefo_density_*.txt crates/spacelab-propagation/tests/fixtures/gracefo_crosswind_*.txt | class ship
sdtm-reference | ship+attr | read 2026-09-14 | hosts - | paths crates/spacelab-propagation/tests/fixtures/sdtm_reference.csv crates/spacelab-propagation/tests/fixtures/sdtm_solar_wind_reference.csv crates/spacelab-propagation/tests/fixtures/sdtm_solar_wind_hours.csv | class ship
celestrak | ship | read 2026-11-24 | hosts celestrak.org | paths crates/spacelab-propagation/tests/fixtures/celestrak_sw_*.csv crates/spacelab-sgp4/tests/fixtures/omm_stations.* crates/spacelab-sgp4/tests/fixtures/supgp_paired.tle crates/spacelab/tests/fixtures/supgp_gps_paired.tle crates/spacelab-sgp4/tests/fixtures/sgp4_catalogue_oracle.txt crates/spacelab-sgp4/tests/fixtures/iss_groundtrack_skyfield.csv crates/spacelab/tests/fixtures/iss_groundtrack_eop_skyfield.csv | class ship
vallado-sgp4-verification | ship | read 2026-09-06 | hosts - | paths crates/spacelab-sgp4/tests/fixtures/sgp4_ver_*.* crates/spacelab-sgp4/tests/fixtures/tcppver.out | class ship
gfz-hpo | ship+attr | read 2026-11-12 | hosts kp.gfz.de | paths crates/spacelab-propagation/tests/fixtures/gfz_hp*_gannon.txt | class ship
set-jb2008-indices | restricted | read 2026-11-24 | hosts sol.spacenvironment.net | paths - | class number-public
iers-eop | ship | read 2026-09-06 | hosts datacenter.iers.org | paths crates/spacelab/tests/fixtures/finals2000A_*.txt crates/spacelab-eop/tests/fixtures/finals2000A_excerpt.txt crates/spacelab-eop/tests/fixtures/eopc04_excerpt.txt | class ship
iers-leap-seconds | ship | read 2026-10-25 | hosts - | paths crates/spacelab-fa/tests/fixtures/leap-seconds.list | class ship
naif-spice | ship | read 2026-09-06 | hosts naif.jpl.nasa.gov | paths - | class ship
igs-bkg | ship+attr | read 2026-09-06 | hosts igs.bkg.bund.de | paths crates/spacelab-gnss/tests/fixtures/igs_sp3c_2026224.sp3 crates/spacelab/tests/fixtures/igs20_*.sp3 crates/spacelab-gnss/tests/fixtures/rinex3_nav_igs_brdc.24n crates/spacelab-gnss/tests/fixtures/rinex4_nav_brd400dlr.rnx | class ship
ilrs | ship+attr | read 2026-11-24 | hosts edc.dgfi.tum.de ilrs.gsfc.nasa.gov | paths crates/spacelab/tests/fixtures/ilrsa_lageos*.sp3 crates/spacelab/tests/fixtures/slr_lageos*.npt crates/spacelab/tests/fixtures/ilrs_*.snx | class ship
itrf-ign | ship+attr | read 2026-09-06 | hosts itrf.ign.fr | paths crates/spacelab/tests/fixtures/itrf2020_*.snx | class ship
icgem-xgm2019e | ship+attr | read 2026-09-06 | hosts icgem.gfz-potsdam.de | paths - | class ship
icgem-eigen-goco | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-propagation/tests/fixtures/eigen6s2_lowdeg_timevariable.gfc | class ship
ngdc-igrf | ship | read 2026-09-06 | hosts www.ngdc.noaa.gov | paths crates/spacelab-gnss/tests/fixtures/igrf14coeffs.txt | class ship
sedoo-fes2014b | ship+attr | read 2026-09-06 | hosts api.sedoo.fr | paths - | class ship
aviso-fes | user-decision | read 2026-11-24 | hosts www.aviso.altimetry.fr | paths - | class number-public
gsfc-got | ship+attr | read 2026-09-06 | hosts earth.gsfc.nasa.gov | paths - | class ship
seanoe-eot20 | ship+attr | read 2026-10-24 | hosts www.seanoe.org | paths - | class ship
tuwien-vmf | ship+attr | read 2026-09-06 | hosts vmf.geo.tuwien.ac.at | paths crates/spacelab-gnss/tests/fixtures/vmf3_5x5_20181125.H00 crates/spacelab-gnss/assets/gpt3_5.grd | class ship
iers-conventions | ship | read 2026-11-13 | hosts iers-conventions.obspm.fr | paths - | class ship
uber-h3 | ship | read 2026-11-12 | hosts raw.githubusercontent.com | paths - | class ship
galileo-icd | ship+attr | read 2026-10-20 | hosts - | paths crates/spacelab-ionosphere/assets/ccir*.asc crates/spacelab-ionosphere/assets/modipNeQG_wrapped.asc crates/spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv | class ship
nequick-jrc | internal-only | read 2026-10-20 | hosts github.com | paths - | class number-public
ccsds-examples | ship | read 2026-09-06 | hosts - | paths crates/spacelab-ndm/tests/fixtures/ccsds_*.* | class ship
gps-almanac | ship | read 2026-09-06 | hosts - | paths crates/spacelab-gnss/tests/fixtures/almanac.*.txt | class ship
oracle-outputs | ship | read 2026-09-06 | hosts - | paths crates/spacelab-control/tests/fixtures/*.txt crates/spacelab-coverage/tests/fixtures/*.txt crates/spacelab-earth/tests/fixtures/*_pyerfa.csv crates/spacelab-fa/tests/fixtures/*_pyerfa.csv crates/spacelab-frames/tests/fixtures/*_pyerfa.csv crates/spacelab-frames/tests/fixtures/orekit_teme_itrf.* crates/spacelab-geometry/tests/fixtures/qhull_oracle.txt crates/spacelab-ephemeris/tests/fixtures/neptune_*.txt crates/spacelab/tests/fixtures/neptune_*.txt crates/spacelab/tests/fixtures/orekit_*.csv crates/spacelab-propagation/tests/fixtures/orekit_dsst_*.csv crates/spacelab-propagation/tests/fixtures/lunar_gravity_oracle.txt crates/spacelab-sensors/tests/fixtures/pyorbital_*.txt crates/spacelab-earthvision/tests/fixtures/slanted_edge.txt crates/spacelab-earthvision/tests/fixtures/gas_transmittance.txt crates/spacelab-vision/tests/fixtures/* crates/spacelab-gnss/tests/fixtures/rinex3_obs_handbuilt.24o crates/spacelab-propagation/tests/fixtures/dsst_coefficients.csv crates/spacelab-propagation/tests/fixtures/taylor_model_enclosures.txt | class ship
pyspectral-lut | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/pyspectral_rayleigh.txt crates/spacelab-earthvision/tests/fixtures/rayleigh_doubling.txt | class ship
aeronet | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/aeronet_*.txt | class ship
radcalnet | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/radcalnet_*.txt | class ship
mpi-mainz-ozone | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/ozone_cross_section.txt | class ship
pleiades-s2p-rpc | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/pleiades_reunion.rpb | class ship
cara-cdms | internal-only | read 2026-09-06 | hosts github.com raw.githubusercontent.com | paths - | class number-public
hwm14 | internal-only | read 2026-09-06 | hosts codeload.github.com | paths - | class number-public
neptune | internal-only | read 2026-09-06 | hosts codeload.github.com | paths - | class number-public
nrlmsise00-fortran | ship | read 2026-09-06 | hosts api.github.com raw.githubusercontent.com | paths - | class ship
hitran | internal-only | read 2026-09-06 | hosts - | paths - | class number-public
mt-ckd | internal-only | read 2026-09-06 | hosts - | paths - | class number-public
nrlmsis2 | internal-only | read 2026-09-06 | hosts - | paths - | class number-private | names NRLMSIS[ -]?2(\.[0-9x])?|pymsis
dtm2020 | restricted | read 2026-09-04 | hosts - | paths - | class number-private | names DTM-?2020
wdc-kyoto | restricted | read 2026-09-05 | hosts - | paths - | class internal
usgs-dst | ship | read 2026-10-25 | hosts geomag.usgs.gov | paths crates/spacelab-propagation/tests/fixtures/usgs_dst_gannon.txt | class ship
nasa-omni | ship | read 2026-10-30 | hosts spdf.gsfc.nasa.gov | paths - | class ship
gcat | ship+attr | read 2026-11-01 | hosts planet4589.org | paths - | class ship
mcdowell-elements | internal-only | read 2026-11-01 | hosts planet4589.org | paths - | class number-public
esa-swarm-dns | user-decision | read 2026-09-14 | hosts swarm-diss.eo.esa.int earth.esa.int | paths - | class -
space-track | user-decision | read 2026-09-06 | hosts - | paths - | class -
cddis | user-decision | read 2026-09-06 | hosts - | paths - | class -
noaa-wam-ipe | ship+attr | read 2026-10-24 | hosts noaa-nws-wam-ipe-pds.s3.amazonaws.com | paths - | class ship
noaa-goes-euvs | ship | read 2026-09-14 | hosts data.ngdc.noaa.gov www.ncei.noaa.gov | paths - | class ship
sdo-eve | ship | read 2026-09-14 | hosts lasp.colorado.edu | paths - | class ship
lisird-lyman-alpha | ship | read 2026-10-31 | hosts lasp.colorado.edu | paths - | class ship
lisird-composites | internal-only | read 2026-09-14 | hosts lasp.colorado.edu | paths - | class internal
bremen-mgii | internal-only | read 2026-09-14 | hosts www.iup.uni-bremen.de | paths - | class internal
gold-on2 | ship+attr | read 2026-11-11 | hosts spdf.gsfc.nasa.gov | paths - | class ship
guvi-on2 | internal-only | read 2026-11-11 | hosts guvitimed.jhuapl.edu | paths - | class number-public
tianmu-ngms | ship+attr | read 2026-11-11 | hosts zenodo.org | paths - | class ship
storm-ai | ship+attr | read 2026-11-12 | hosts dataverse.harvard.edu | paths docs/storm-ai-split.json | class ship
aether-p3 | ship+attr | read 2026-11-12 | hosts - | paths docs/aether-p3-cases.json | class ship
pyfes | ship | read 2026-11-24 | hosts - | paths - | class ship
tiegcm | user-decision | read 2026-11-24 | hosts - | paths - | class -
tudelft-lambda | user-decision | read 2026-11-24 | hosts - | paths - | class -
ckdmip | internal-only | read 2026-11-24 | hosts - | paths - | class number-public
rrtmgp-data | internal-only | read 2026-11-24 | hosts - | paths - | class number-public
ecckd | internal-only | read 2026-11-24 | hosts - | paths - | class number-public
tsis1-hsrs | ship+attr | read 2026-11-24 | hosts - | paths - | class ship
gpw-v4 | ship+attr | read 2026-11-24 | hosts - | paths - | class ship
silso | restricted | read 2026-11-24 | hosts - | paths - | class internal
goco2025s | ship+attr | read 2026-11-24 | hosts - | paths - | class ship
gfz-aod1b | ship+attr | read 2026-11-24 | hosts - | paths - | class ship
gfz-hpo-forecast | internal-only | read 2026-11-24 | hosts - | paths - | class number-public
igs-antex | ship+attr | read 2026-11-24 | hosts - | paths - | class ship
imcce-vsop87 | restricted | read 2026-11-24 | hosts - | paths - | class number-public
elpmpp02 | unread | read 2026-11-24 | hosts - | paths - | class -
gaia | restricted | read 2026-11-24 | hosts - | paths - | class number-public
itu-r-recs | restricted | read 2026-11-24 | hosts - | paths - | class number-public
```

**The storm driver, decided by the rows above.** JB2008's storm term is Dst-driven and DTM2020's is
Kp-driven; the NRLMSISE-00 and JB2008 validation runs use ap and SET's Dst-derived ΔTc respectively.
Kyoto's Dst is closed to commercial applications, and SET's ΔTc is for the licensee's own research
and internal use, with no redistribution (`set-jb2008-indices`). USGS's Dst4 and
Dst3 are public domain but start in September 2015, after the 2003-10 and 2015-03 storms. The one open, continuous, commercially usable storm index from 1995 is GFZ's Hpo family (CC BY
4.0), which is why our own thermosphere model (`plans/16` Ф15) is driven by Hp60/ap, why a Dst term
in it is fitted on USGS Dst (Ф23.A) and not on Kyoto's, and why a Dst-driven ΔTc rebuilt from Kyoto
data (Ф18.A) is a *validation* input and nothing sold depends on it.

## Not for public release

Four models are kept in the development tree and are not part of the public release. The public
tree is cut from the development tree by a script, never edited by hand.

* **DTM2020.** §1's licence forbids dissemination of the code and of its coefficients, and asking
  CNES cannot fix that: the licence on offer is non-transferable, and a permissive crate is a
  sub-licence to everyone who downloads it. Commercial use is a separate question and is **not**
  resolved by excluding the module: "academic, non-commercial purposes only" binds our own use too.
* **JB2008.** §3: SET's licence agreement forbids a translation and the distribution of the code,
  and the validation fixtures are excerpts of index files it forbids redistributing. The development
  tree runs SET's own compiled file as a comparator and ships none of it.
* **NeQuick-ITU and the ITU-R P.834 mapping function.** Their data, the ITU 1°×2° MODIP grid and the
  P.834 coefficient grid (`itu-r-grids` in the data register), carry ITU copyright and no grant.
  NeQuick-G does not need them: its CCIR maps and MODIP grid are the Galileo ICD's own.

Where this documentation quotes a number measured with one of these models, the number is the
development tree's and is reproducible only there. The public tree quotes none measured with DTM2020
(§1, D-32).


## The covered set

The machine-readable half of this file. `tools/check-provenance.py --check` compares it against the
sweep and fails on a module that is missing from it or on a line that no longer claims anything --
the second direction matters too, because a stale line is a claim that somebody looked at code that
has since been rewritten. Regenerate it by running the script without `--check`.

```text
crates/spacelab-astro/src/richardson.rs
crates/spacelab-coverage/src/root.rs
crates/spacelab-earth/src/lib.rs
crates/spacelab-earthvision/src/image_quality.rs
crates/spacelab-elements/src/brouwer_lyddane.rs
crates/spacelab-elements/src/eckstein_hechler.rs
crates/spacelab-eop/src/subdaily_tables.rs
crates/spacelab-estimation/src/hinf.rs
crates/spacelab-fa/src/fundamental.rs
crates/spacelab-fa/src/lib.rs
crates/spacelab-gnss/src/gpt3.rs
crates/spacelab-gpu/src/batch_sgp4.rs
crates/spacelab-ionosphere/src/ccir.rs
crates/spacelab-ionosphere/src/fourier.rs
crates/spacelab-ionosphere/src/galileo.rs
crates/spacelab-ionosphere/src/lib.rs
crates/spacelab-ionosphere/src/model.rs
crates/spacelab-ionosphere/src/modip.rs
crates/spacelab-ionosphere/src/parameters.rs
crates/spacelab-ionosphere/src/ray.rs
crates/spacelab-propagation/src/adams_nordsieck.rs
crates/spacelab-propagation/src/dsst/auxiliary.rs
crates/spacelab-propagation/src/dsst/coefficients.rs
crates/spacelab-propagation/src/dsst/drag.rs
crates/spacelab-propagation/src/dsst/gaussian.rs
crates/spacelab-propagation/src/dsst/gaussian_short_period.rs
crates/spacelab-propagation/src/dsst/hansen.rs
crates/spacelab-propagation/src/dsst/mod.rs
crates/spacelab-propagation/src/dsst/short_period.rs
crates/spacelab-propagation/src/dsst/srp.rs
crates/spacelab-propagation/src/dsst/tesseral.rs
crates/spacelab-propagation/src/dsst/tesseral_short_period.rs
crates/spacelab-propagation/src/dsst/third_body.rs
crates/spacelab-propagation/src/dsst/third_body_short_period.rs
crates/spacelab-propagation/src/dsst/zonal.rs
crates/spacelab-propagation/src/earth_radiation.rs
crates/spacelab-propagation/src/geopotential.rs
crates/spacelab-propagation/src/ias15.rs
crates/spacelab-propagation/src/lib.rs
crates/spacelab-propagation/src/nrlmsise00/mod.rs
crates/spacelab-propagation/src/ocean_tide_got.rs
crates/spacelab-propagation/src/radau.rs
crates/spacelab-propagation/src/rkn1210.rs
crates/spacelab-propagation/src/rkn1210_consts.rs
crates/spacelab-propagation/src/rosenbrock.rs
crates/spacelab-propagation/src/sdtm/atmosphere.rs
crates/spacelab-propagation/src/spacecraft.rs
crates/spacelab-sensors/src/geoloc.rs
crates/spacelab-sensors/src/scan.rs
crates/spacelab-sgp4/src/lib.rs
crates/spacelab-slr/src/displacement.rs
crates/spacelab-slr/src/loading.rs
crates/spacelab-slr/src/loading_tables.rs
crates/spacelab-slr/src/troposphere.rs
```
