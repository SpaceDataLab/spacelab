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
| `spacelab-propagation/src/jb2008/**` | Bruce R. Bowman's `JB2008.for`, Space Environment Technologies | **SET's JB2008 License Agreement** (the PDF dated 2026-10-15, read 2026-10-25): own research, educational, engineering and internal use only; no modification or translation except to compile and interface; no embedding in a library given to others | **Must not ship.** Excluded from the public tree. See §3 |
| **28 files** — all 14 of `spacelab-propagation/src/dsst/`, `spacelab-propagation/src/{earth_radiation,spacecraft}.rs`, all 8 of `spacelab-ionosphere/src/` (the NeQuick chain), `spacelab-gnss/src/{gpt3,itu834_mapping}.rs`, `spacelab-elements/src/{brouwer_lyddane,eckstein_hechler}.rs` | Orekit (CS GROUP, formerly CS-SI) | **Apache-2.0** — `LICENSE.txt` at `CS-SI/Orekit`; copyright line read from `DSSTPropagator.java` | **Ship, with attribution owed.** See §4. The full list is in `NOTICE`, and it is 28 rather than the 15 a first pass counted — the NeQuick and GPT3 chains say "Ported from Orekit's …" in their own module docs and had not been counted |
| `spacelab-sensors/src/{geoloc,scan}.rs` | `pyorbital.geoloc` and `pyorbital.geoloc_instrument_definitions` (pytroll) | **Apache-2.0** — GitHub licence API | **Ship, with attribution owed.** Same conditions as Orekit; in `NOTICE` |
| `spacelab-earthvision/src/image_quality.rs` | the metric core of Satellogic's `iquaflow` | **MIT** — GitHub licence API | **Ship, with attribution owed.** In `NOTICE` |
| `spacelab-slr/src/{displacement,loading,troposphere}.rs` | IERS Conventions Centre reference routines `DEHANTTIDEINEL`, `HARDISP` (with `ADMINT`/`TDFRPH`/`SPLINE`/`EVAL`), `FCULZD_HPA`, `FCUL_A` | IERS Conventions software, published by the Conventions Centre as the reference realisation of the Conventions | **Ship.** `CONTRIBUTING.md`'s first paragraph is about exactly this material: the IERS Conventions and everything the IERS publishes are ours to implement. Orekit's own `NOTICE` credits the same routines |
| `spacelab-propagation/src/ias15.rs` | Everhart (1985), Rein & Spiegel (2015); REBOUND used only as a numerical check | REBOUND is **GPL-3.0** — `LICENSE` at `hannorein/rebound` | **Clean, and the reason the derive-first rule pays.** The nodes are the roots of `d⁷/dx⁷[x⁸(x−1)⁷]` and the `c`/`d` conversion matrices come from `derivations/ias15_nodes.mac`; REBOUND's hard-coded arrays are the agreement check at ~1e-33, not the source. Deriving first is what makes a GPL implementation safe to *compare against* |
| `spacelab-gpu/src/batch_sgp4.rs` | the `sgp4` crate 2.4 | **MIT** — `cargo metadata` | **Ship, with attribution owed.** The MIT notice has to travel with a substantial portion; §4's `NOTICE` carries it |
| `spacelab-sgp4/src/lib.rs` | — | — | **Not a port.** The crate *depends on* `sgp4` 2.4 (MIT) and adds typing and batching around it. The word "port" in its docs describes the upstream crate, not this one |
| `spacelab-eop/src/subdaily_tables.rs`, `spacelab-slr/src/loading_tables.rs`, `spacelab-ionosphere/src/lib.rs`, `spacelab-propagation/src/{rkn1210,rkn1210_consts}.rs` | IERS Conventions software tables (`ORTHO_EOP`/`CNMTX`, `PMSDNUT2`, `UTLIBR`, `ADMINT.F`), the Galileo ICD (Annex C) CCIR maps and MODIP grid and the ITU-R MODIP grid, Dormand–El-Mikkawy–Prince (1987) | published tables | **Clean.** Measured constants and published coefficient tables are facts, and every one of these is *machine-read* into the tree by a script in `derivations/` or `tools/` rather than hand-copied — which is a correctness argument first and a provenance argument second |
| `spacelab-astro/src/richardson.rs`, `spacelab-propagation/src/{adams_nordsieck,geopotential,ias15,radau,rosenbrock}.rs` | Maxima derivations in `derivations/*.mac` | — | **Clean, and the pattern to copy.** Each of these says in its own documentation that the coefficients are *derived* rather than transcribed, and names the `.mac` file that derives them; the published listing is the agreement check. `rosenbrock.rs` is the one that also names a reference listing (Hairer & Wanner's `ros4.f`, BSD-style), and a Butcher tableau is facts in any case |
| `spacelab-propagation/src/ocean_tide_got.rs` | GOT5.5 **cotidal grids** (amplitude and phase), R. D. Ray, NASA GSFC | **No stated terms.** `earth.gsfc.nasa.gov` serves the atlas with no licence file, no registration and no click-through; the `#README` names papers to cite and warns against navigational use, nothing more. Same position as JB2008 | **Ship, with the finding recorded.** And it is not a port: the coefficients are *computed here* by `derivations/tide_to_stokes.py` from the elevation grids. See §5 |
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
interface (§5); and forbids redistributing the index files "in whole or in part" (§6). Outputs may be
published with the acknowledgement it gives; commercial Outputs need a licence (§7).

`jb2008/` is a translation of `JB2008.for` into Rust, so §5 forbids it and §4 forbids shipping it;
`derivations/jb2008_tables.py` extracts its tables from that file; and the two validation fixtures are
excerpts of the index files, which §6 forbids redistributing. All four are excluded from the public
tree (below). The earlier reading of this section — the file states no terms, so it ships — read the
file alone, and no stated grant was never a permission.

The model itself is published (Bowman et al. 2008, AIAA 2008-6438). An implementation written from the
papers alone, with every number taken from a printed source, is a different piece of work from this
code and is not in this repository.

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
citations are in place at every use site. The position is JB2008's, recorded the same way: no stated
terms, so nothing to breach and nothing to rely on either.

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

| Key | What it is | Terms, as read | Verdict |
|---|---|---|---|
| `tudelft-thermosphere` | Thermosphere neutral density and crosswind data, TU Delft (Siemes et al. 2023): the CHAMP / GRACE-FO fixtures and the satellite-months the density campaign reads | CC BY 4.0, stated in every file's own header (`gracefo_density_20190101.txt` lines 11–16), and confirmed by the data's contact, Christian Siemes, by email on 2026-10-24: CC BY 4.0 for the density and wind datasets. The same email puts TU Delft's satellite geometry, aerodynamic and radiation-pressure models under CC BY-NC-SA 4.0; spacelab uses none of them, and they are not admitted. The server has refused connections since 2026-09-05 | **ship+attr** |
| `sdtm-reference` | `crates/spacelab-propagation/src/sdtm/coefficients.rs` and `crates/spacelab-propagation/tests/fixtures/sdtm_reference.csv` — SDTM-1's fitted coefficients, and this model's own values at a sample of the geometries it was fitted on, both written by `tools/gen_sdtm_coefficients.py` | Our own numbers, fitted on `tudelft-thermosphere` (CC BY 4.0). Neither file contains a measured density, a driver series or any line of TU Delft's files; what does carry over is the attribution CC BY 4.0 §3(a)(1) requires of adapted material, discharged in `coefficients.rs`'s own header, in this row and in `docs/verification.md` §10.10 | **ship+attr** |
| `celestrak` | CelesTrak: `SW-All.csv` space-weather series, GP and SupGP element sets, the ISS ground-track TLE, satellite.js | Usage policy (read 2026-09-06): rate limits — GP no more than every two hours, directories hourly, stop on a non-200 — and nothing about reuse or redistribution; no licence text exists | **ship** |
| `vallado-sgp4-verification` | The AIAA 2006-6753 verification element sets and `tcppver.out`, Vallado's own SGP4 distribution as served by CelesTrak, plus python-sgp4's rendering of them | Published test cases of a public-domain algorithm; python-sgp4 is MIT | **ship** |
| `gfz-hpo` | GFZ Hp30 / Hp60 / ap30 / ap60 — the Gannon-storm excerpts committed as fixtures, the **complete hourly series** (`Hp60_ap60_complete_series.txt`, 1985-01-01 →, 365 544 rows) fetched into the git-ignored cache for the `--ap60` fit of `verification.md` §10.15, and the Kp/ap feed `fetch-density-data.sh` reads | `# LICENSE: CC BY 4.0` in the file header; citation lines in the same header (Matzka et al. 2024, doi:10.5880/Hpo.0003; Yamazaki et al. 2024) | **ship+attr** |
| `set-jb2008-indices` | Space Environment Technologies `SOLFSMY.TXT` (F10/S10/M10/Y10) and `DTCFILE.TXT` (Dst-derived ΔTc): JB2008's drivers; 25-day excerpts committed as fixtures in the development tree | SET's JB2008 License Agreement (2026-10-15, read 2026-10-25, §3): the index files are part of the licensed Software, for the licensee's own research, educational, engineering and internal use, and §6 forbids redistributing them "in whole or in part". The excerpts are excluded from the public tree, and its fetch script does not retrieve the files. Open series rebuilt to the published definitions replace them as drivers (`plans/16` Ф18.A) | **restricted** |
| `iers-eop` | `finals2000A.all` and its excerpts (five fixtures), IERS Earth Orientation Centre; an excerpt of the EOP 20 C04 series `eopc04.1962-now`, EOP Product Centre (Paris Observatory), header verbatim — cut on 2026-10-24 from the copy `astropy-iers-data` 0.2026.10.19 ships, which its `ReadMe.eopc04` says was taken from hpiers.obspm.fr, because that host timed out and datacenter.iers.org refused the request that day | IERS products are published for free use with no licence text; attribution is by citation (IERS EOP 20 C04, Bulletin A) | **ship** |
| `iers-leap-seconds` | `leap-seconds.list`, the IERS Earth Orientation Centre's list of leap seconds (Paris Observatory), updated through Bulletin C 72 and valid until 2027-06-28, vendored as `crates/spacelab-fa/tests/fixtures/leap-seconds.list`. Taken on 2026-10-25 from the copy the IANA time-zone database serves (`data.iana.org/time-zones/data/leap-seconds.list`), byte-identical to Fedora's `tzdata-2026e`, because hpiers.obspm.fr timed out that day. sha256 `db5a895f16853b03bfc865e8d68f9fc8710ef1740e3400c701cd46a5bbbc3433`; the file's own SHA-1 line (`#h`, over the digits of its `#$`, `#@` and data lines) matches its data | The file's own words: "COPYRIGHT STATUS OF THIS FILE … This file is in the public domain." | **ship** |
| `naif-spice` | DE440 / DE442 short-span kernels, lunar PA kernels, the LRO SPK, from NAIF | NASA/JPL, a US Government work; fetched, never vendored | **ship** |
| `igs-bkg` | IGS final / rapid SP3 products and the multi-GNSS broadcast files via BKG; DLR/GSOC's BCEmerge merged navigation file (doi:10.57677/BRD400DLR) | *IGS Data and Product Disclaimer and Terms of Use*, 2020-08-05: open "for the benefit of all scientific, educational, and commercial users", attribution required (§6) | **ship+attr** |
| `ilrs` | ILRS normal points (CRD), the `ilrsa` combined LAGEOS orbits, data-handling and eccentricity SINEX files, from EDC (DGFI-TUM) and ILRS/GSFC | ILRS data are open; the network asks to be acknowledged (Pearlman et al. 2019, doi:10.1007/s00190-019-01241-1) | **ship+attr** |
| `itrf-ign` | ITRF2020 SLR station positions and velocities (IGN) | ITRF products are freely available; cite Altamimi et al. 2023 (doi:10.1007/s00190-023-01738-w) | **ship+attr** |
| `icgem-xgm2019e` | XGM2019e static gravity field, the one the geopotential embeds, fetched by `fetch-specs.sh` for `spec_seals.py` | ICGEM model page: CC BY 4.0 (read 2026-09-06) | **ship+attr** |
| `icgem-eigen-goco` | EIGEN-6S2 time-variable excerpt (fixture), EIGEN-6C4 and GOCO06s (spec seals) | CC BY 4.0 — the GFZ Data Services DOI pages of all three (10.5880/icgem.2016.008, 10.5880/icgem.2015.1, 10.5880/ICGEM.2019.002) declare `spdx.org/licenses/CC-BY-4.0` in their metadata, read 2026-09-06 | **ship+attr** |
| `ngdc-igrf` | IGRF-14 coefficients (NCEI) | "The software code is in the public domain … may be used freely by the public" — NCEI IGRF page, read 2026-09-06 | **ship** |
| `sedoo-fes2014b` | The FES2014b geopotential-coefficient table (GRGS / SEDOO) that `derivations/ocean_tide_fes2014.py` transcribes into `ocean_tide_fes2014.rs`, the default ocean-tide table | AVISO+ Licence Agreement, Issue 20, read 2026-09-06 from the PDF: use "for any purpose, including commercial purposes"; §3.2 restricts only re-sharing the *original, unmodified* grids on mass-distribution platforms and "does not restrict Sharing Adapted Material (Value-Added Products or Derivative Works)"; Annex A makes only the *current fields of the latest FES* non-commercial. Credit owed: "Noveltis, Legos and CLS", AVISO+, CNES | **ship+attr** |
| `aviso-fes` | FES2014b / FES2022 elevation grids from AVISO+, for `tide_to_stokes.py` | Same licence; registration is mandatory before download, which is the user's act, not a script's | **user-decision** |
| `gsfc-got` | GOT5.5 / GOT5.6 cotidal grids (R. D. Ray, NASA GSFC) from which `ocean_tide_got.rs` is *computed* | No stated terms, no registration, no click-through; `#README` asks for citation and warns against navigational use (§5) | **ship+attr** |
| `seanoe-eot20` | EOT20 cotidal grids (DGFI-TUM), the final version `85762.zip` from SEANOE (doi:10.17882/79489), from which `ocean_tide_eot20.rs` is *computed*. The archive also carries the load-tide grids, which the generator refuses by their title | SEANOE dataset page, read 2026-10-24: Creative Commons Attribution 4.0. The page also withdraws an older, corrupted release ("should not be used for any application"), offered only on request; the final version is open access. Credit owed: the dataset citation and Hart-Davis et al. (2021), ESSD 13, 3869 | **ship+attr** |
| `tuwien-vmf` | VMF3 5°×5° grid fixture and the GPT3 grid `assets/gpt3_5.grd` (via Orekit's Apache-2.0 copy) | VMF Data Server terms, read 2026-09-06: free of charge; products older than 24 h under CC BY 4.0; **forecast products for non-commercial scientific use only** — so an operational path uses GPT3, never the VMF forecasts | **ship+attr** |
| `itu-r-grids` | The ITU 1°×2° MODIP grid (`spacelab-ionosphere/assets/modip.asc`, read only by NeQuick-ITU) and the ITU-R P.834 mapping coefficients (`spacelab-gnss/assets`). The CCIR maps and the Galileo MODIP grid left this row on 2026-10-20 for `galileo-icd` | Byte-identical to the files Orekit distributes under Apache-2.0. Re-read 2026-09-14, unchanged: ITU's site-wide copyright page says "ITU holds copyright in the information available on this Web site, unless otherwise stated" and directs reproduction requests to jur@itu.int, newsroom material being the only stated exception; and the Study Group 3 ionospheric/tropospheric databank page that offers these files carries **warranty disclaimers only** — "provided 'as is' without any warranty of any kind" — and **no grant**, so the free-of-charge reading the 2026-09-06 pass left open is not supported by the page. Orekit's distribution is the lineage relied on; the question for ITU-R is in `plans/15`. The alternative — regenerate the grids from an open source — is **measured shut**: neither MODIP grid reproduces from the vendored DGRF series at any epoch (`verification.md` §5.10) | **unread** |
| `galileo-icd` | The CCIR monthly foF2/M(3000)F2 coefficient maps and the Galileo 5°×10° MODIP grid (`spacelab-ionosphere/assets/ccir*.asc`, `modipNeQG_wrapped.asc`), the files embedded in Annex C of the Galileo *Ionospheric Correction Algorithm for Galileo Single Frequency Users*, issue 1.2, © European Commission 2016; and `spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv`, whose inputs are its Annex E cases and whose STEC column is the output of `nequick-jrc` | The GSC terms of use, read 2026-10-20 at `gsc-europa.eu/terms-of-use`: "Unless otherwise stated, downloading, reproduction and use of all the materials and documents published on the Website are authorised provided the source is acknowledged as follow: © EU 2011-2026", except material identified as a third party's copyright, and logos. The ICD's own terms page is disclaimers only and identifies no third-party copyright in Annex C. The maps descend from CCIR's (ITU's predecessor) foF2 coefficients, and ITU's own P.531 copies hold the same values. Orekit's copies, which this crate embedded until 2026-10-20, differ from both in the eighth digit and carry a corrupt last F2 coefficient in July to December (up to 4.2 TECU, `tools/nequick-jrc-oracle/README.md`) | **ship+attr** |
| `nequick-jrc` | NeQuickG JRC 1.0, the reference implementation of the Galileo ICD (European Union, 2019), cloned by `tools/nequick-jrc-oracle/run.sh` from the `odrisci/NeQuickJRC` mirror | EUPL-1.2, copyleft: run as an oracle, never linked or vendored; only its outputs are committed | **internal-only** |
| `ccsds-examples` | The example OPM / OEM / CDM messages from the CCSDS Blue Books (502.0-B-3 annex G, 508.0-B-1) | The standard's own worked examples, reproduced as format-conformance fixtures. The front matter of 502.0-B-3 (read 2026-09-06 from `public.ccsds.org`) carries no copyright line and no reproduction restriction — only "published and maintained by the CCSDS Secretariat"; the same position as JB2008 and GOT, recorded rather than assumed | **ship** |
| `gps-almanac` | SEM and YUMA almanacs (week 384) | US Coast Guard Navigation Center, a US Government work | **ship** |
| `oracle-outputs` | Fixtures *we generated* by running an oracle: Basilisk (ISC), NASA 42 (NOSA 1.3, run unmodified), pyorbital (Apache-2.0), Orekit (Apache-2.0), pyerfa (BSD-3), skyfield (MIT), python-sgp4 (MIT), qhull, NEPTUNE (Apache-2.0), GRGM1200A (NASA PDS), and our own generators (air mass, twilight, slanted edge, gas transmittance, the ONNX cloud fixture, the hand-built RINEX observation file) | The numbers are the output of a run and are ours; each file names its generator and the oracle's licence in its header where the format allows a comment | **ship** |
| `pyspectral-lut` | libRadtran Rayleigh reflectance LUT via pyspectral v1.0.1 | CC BY 4.0, Zenodo doi:10.5281/zenodo.1288441 (file header) | **ship+attr** |
| `aeronet` | AERONET Version 3 Level 2.0 almucantar inversions (42 sites) | AERONET Data Use Policy: acknowledge the site PIs and the network, and contact the PI before publishing on the data (file header names the PIs) | **ship+attr** |
| `radcalnet` | RadCalNet surface/TOA reflectance and atmospheres, 42 scenes plus hold-out days | RadCalNet Data Policy v1.1, quoted in full in the file header: "freely and publically available"; acknowledge "the CEOS WGCV RadCalNet service … and in particular from site X and Y". Portal credentials are read from the environment by the generator and are never in the tree | **ship+attr** |
| `mpi-mainz-ozone` | Ozone absorption cross-section, MPI-Mainz UV/VIS Spectral Atlas | Citation requested, no other condition (read 2026-09-06) | **ship+attr** |
| `pleiades-s2p-rpc` | One real RPC00B camera model (Pleiades over Réunion) taken from the s2p test image's metadata | s2p (Centre Borelli) is **BSD-3-Clause** — GitHub licence API, 2026-09-06 — not AGPL as a first pass assumed; its `tests/data/input_pair` holds the images and a config with no separate notice. The ninety coefficients are the camera's facts read from the image header; the BSD notice travels in `NOTICE` for the distribution they were read from | **ship+attr** |
| `cara-cdms` | NASA CARA's eleven "Alfano" sample CDMs | NASA Open Source Agreement notices in the repository, no LICENSE file; fetched, never vendored (script header) | **internal-only** |
| `hwm14` | NRL Horizontal Wind Model 14, software and coefficients | CC BY-NC-ND (Drob et al. 2015 supporting information): no derivatives, no commercial use — run as an oracle, never ported or linked (script header) | **internal-only** |
| `neptune` | NEPTUNE propagator (TU Braunschweig), a fourth independent oracle | Apache-2.0; deliberately not linked (script header) | **internal-only** |
| `nrlmsise00-fortran` | `nrlmsise00_sub.for`, the NRL FORTRAN that `derivations/nrlmsise00_tables.py` reads | US Government work, 17 U.S.C. §105 (§2) | **ship** |
| `hitran` | HITRAN line parameters behind the gas-transmittance fit | Free with citation; the lines are cached under `~/.cache/spacelab/hitran` and never enter the tree — only our exponential-sum fit does | **internal-only** |
| `mt-ckd` | AER MT_CKD water-vapour continuum | Research-only; "may not be incorporated into proprietary software … or commercial software … without the express written consent of AER" — oracle, cached outside the tree, nothing derived from it committed | **internal-only** |
| `nrlmsis2` | NRLMSIS 2.0 / 2.1 via `pymsis` | NRL licence: no non-academic commercial use, no modification, no dissemination; US 10,641,925 B2 (family: EP withdrawn, AU granted, US to 2039) — validation only | **internal-only** |
| `dtm2020` | DTM2020 code and `DTM_2020_F107_Kp.dat` | §1 — excluded from the public tree; research use only | **restricted** |
| `wdc-kyoto` | WDC Kyoto Dst / AE / SYM-H | "The WDC Kyoto does not allow commercial applications of the geomagnetic indices" (read 2026-09-05). Not fetched by any script | **restricted** |
| `esa-swarm-dns` | ESA's own Swarm Level-2 thermospheric density products (`SW_OPER_DNSxPOD_2_`, `DNSxACC_2_`) on `swarm-diss.eo.esa.int` — 4608 daily files per satellite, 2013-11-28 → 2026-07-31, 643 MB for Swarm A alone. The fallback for `tudelft-thermosphere`, whose server has refused connections on five probes between 2026-09-05 and 2026-09-14 | Read 2026-09-14 from the two documents the download page itself links. The *Data Policy* (ESA/PB-EO(2010)54) is permissive in intent — use "be it scientific, for public good or **commercial**", "full and open access … free of charge" — but access is "granted upon electronic on-line registration and the user accepting the ESA Terms & Conditions", and those (ESA-EOPG-PDGS-PR-1 Issue 1.1, 2023-09-20) add three clauses no register row can decide: **B.6(a)** "Use all ESA EO data accessed exclusively for their own EO research and applications purposes"; **C.2** mark "all ESA EO data and analysed information … © ESA (year of reception)"; **C.5** any IP emerging from direct use carries a "worldwide, free of charge, irrevocable, and non-exclusive licence" to ESA, with patent filings notified within two months. A fitted coefficient set is plausibly "analysed information" and certainly "own intellectual property" | **user-decision** |
| `usgs-dst` | USGS Geomagnetism Program web service, virtual observatory `USGS`, elements `Dst3` ("3-channel Dst") and `Dst4` ("4-channel Dst") as `/ws/elements/` lists them. The 2026-09-06 probes asked for `DST`, which is not an element, and read its all-`null` answer as an empty index | U.S. public domain: "USGS-authored or produced data and information are considered to be in the U.S. Public Domain", and credit to "U.S. Geological Survey" is asked (usgs.gov, *Copyrights and Credits*, read 2026-10-25). **Not fetched.** On 2026-10-25 `/ws/health`, `/ws/elements/` and `/ws/observatories/USGS` answer 200 JSON, and every `/ws/data/` request answers a 404 HTML page, the documented `?id=BOU` example included; it was not worked around. A letter to the Geomagnetism Program asks for the access route or the hourly `Dst4` archive. Until then the open storm driver is GFZ Hp30/Hp60 (CC BY, since 1995) | **ship** |
| `space-track` | Space-Track.org GP catalogue | User agreement binds per account; never vendored, never used by a fetch script — CelesTrak serves anything committed | **user-decision** |
| `cddis` | NASA CDDIS (IGS/ILRS archive) | Earthdata login; the terms attach at registration | **user-decision** |
| `noaa-wam-ipe` | WAM-IPE neutral density on the NOAA Open Data Dissemination bucket `noaa-nws-wam-ipe-pds`: the operational `v1.2/wfs` `wam_fixed_height` files `tools/fetch-wam-ipe.sh` selects and `tools/wam-ipe.sha256` pins, read by `wam_ipe.rs` and scored in `measured_density.rs`. Nothing from them is committed | The AWS Open Data registry page, read 2026-10-24: "NOAA data disseminated through NODD are open to the public and can be used as desired"; NOAA "requests attribution for the use or dissemination of unaltered NOAA data", and "it is not permissible to state or imply endorsement by or affiliation with NOAA". The bucket's `NON-OPERATIONAL_RESEARCH/` reruns ask users to "contact the WAM-IPE team … before accessing such files", so the fetch script does not read them. Credit: `wam_ipe.rs`'s header and `docs/verification.md` §10.18 | **ship+attr** |
| `noaa-goes-euvs` | GOES-R EUVS daily science products (NOAA/NCEI): He II 30.4 nm, H I Lyman-α 121.6 nm and the Mg II core-to-wing index, GOES-16/17/18/19 | The files' own `license` global attribute, read 2026-09-14: *"These data may be redistributed and used without restriction."* LISIRD's catalogue states CC0 1.0 for the same products. A grant with no condition, so nothing is owed but the citation we give anyway | **ship** |
| `sdo-eve` | SDO/EVE Level-3 merged daily product (LASP/CU for NASA): the ESP 27.16–33.8 nm band, the MEGS-P Lyman-α diode, the He II 30.4 nm line | `license` attribute of the file, read 2026-09-14: `CC0 1.0 https://creativecommons.org/publicdomain/zero/1.0/`; LISIRD's catalogue agrees. Acknowledgement asked for in the file (`Courtesy of NASA/SDO and the AIA, EVE, and HMI science teams.`) and given in `docs/verification.md` §10.9; it joins `NOTICE` when something derived from it ships, which is `NOTICE`'s own scope | **ship** |
| `lisird-composites` | As served by LISIRD and **not** admitted: the Machol et al. (2019) composite Lyman-α, the Viereck composite Mg II, SORCE Mg II, SOHO/SEM, and the Penticton / NOAA / CLS radio-flux series | LISIRD publishes a machine-readable catalogue (`https://lasp.colorado.edu/lemr/lisird/lisird`) in which each dataset either names a licence or does not. Read 2026-09-14: these name **none** — 354 of the 420 datasets name none, against 41 CC0 and 25 CC BY 4.0. No licence is not a permissive licence, so they are read as oracles and nothing derived from them ships. (`lasp_filtered_drao_f107` *is* CC BY 4.0 and is the admissible alternative source of F10.7 if CelesTrak's ever closes) | **internal-only** |
| `bremen-mgii` | The Bremen composite Mg II index (1978–) and its F30-extended form (1947–), IUP Bremen | The page (read 2026-09-14, moved to `https://www.iup.uni-bremen.de/UVSAT/data/`) and both data files carry citations and a contact address and **no terms of any kind**; LISIRD's catalogue also states none for its copy. Oracle only, and the letter asking IUP for a licence is in `plans/16` Ф14.G | **internal-only** |

```text
tudelft-thermosphere | ship+attr | read 2026-10-24 | hosts thermosphere.tudelft.nl | paths crates/spacelab-propagation/tests/fixtures/champ_density_sample.txt crates/spacelab-propagation/tests/fixtures/gracefo_density_*.txt crates/spacelab-propagation/tests/fixtures/gracefo_crosswind_*.txt
sdtm-reference | ship+attr | read 2026-09-14 | hosts - | paths crates/spacelab-propagation/tests/fixtures/sdtm_reference.csv
celestrak | ship | read 2026-09-06 | hosts celestrak.org | paths crates/spacelab-propagation/tests/fixtures/celestrak_sw_*.csv crates/spacelab-sgp4/tests/fixtures/omm_stations.* crates/spacelab-sgp4/tests/fixtures/supgp_paired.tle crates/spacelab/tests/fixtures/supgp_gps_paired.tle crates/spacelab-sgp4/tests/fixtures/sgp4_catalogue_oracle.txt crates/spacelab-sgp4/tests/fixtures/iss_groundtrack_skyfield.csv crates/spacelab/tests/fixtures/iss_groundtrack_eop_skyfield.csv
vallado-sgp4-verification | ship | read 2026-09-06 | hosts - | paths crates/spacelab-sgp4/tests/fixtures/sgp4_ver_*.* crates/spacelab-sgp4/tests/fixtures/tcppver.out
gfz-hpo | ship+attr | read 2026-09-06 | hosts kp.gfz.de | paths crates/spacelab-propagation/tests/fixtures/gfz_hp*_gannon.txt
set-jb2008-indices | restricted | read 2026-10-25 | hosts sol.spacenvironment.net | paths -
iers-eop | ship | read 2026-09-06 | hosts datacenter.iers.org | paths crates/spacelab/tests/fixtures/finals2000A_*.txt crates/spacelab-eop/tests/fixtures/finals2000A_excerpt.txt crates/spacelab-eop/tests/fixtures/eopc04_excerpt.txt
iers-leap-seconds | ship | read 2026-10-25 | hosts - | paths crates/spacelab-fa/tests/fixtures/leap-seconds.list
naif-spice | ship | read 2026-09-06 | hosts naif.jpl.nasa.gov | paths -
igs-bkg | ship+attr | read 2026-09-06 | hosts igs.bkg.bund.de | paths crates/spacelab-gnss/tests/fixtures/igs_sp3c_2026224.sp3 crates/spacelab/tests/fixtures/igs20_*.sp3 crates/spacelab-gnss/tests/fixtures/rinex3_nav_igs_brdc.24n crates/spacelab-gnss/tests/fixtures/rinex4_nav_brd400dlr.rnx
ilrs | ship+attr | read 2026-09-06 | hosts edc.dgfi.tum.de ilrs.gsfc.nasa.gov | paths crates/spacelab/tests/fixtures/ilrsa_lageos*.sp3 crates/spacelab/tests/fixtures/slr_lageos*.npt crates/spacelab/tests/fixtures/ilrs_*.snx
itrf-ign | ship+attr | read 2026-09-06 | hosts itrf.ign.fr | paths crates/spacelab/tests/fixtures/itrf2020_*.snx
icgem-xgm2019e | ship+attr | read 2026-09-06 | hosts icgem.gfz-potsdam.de | paths -
icgem-eigen-goco | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-propagation/tests/fixtures/eigen6s2_lowdeg_timevariable.gfc
ngdc-igrf | ship | read 2026-09-06 | hosts www.ngdc.noaa.gov | paths crates/spacelab-gnss/tests/fixtures/igrf14coeffs.txt
sedoo-fes2014b | ship+attr | read 2026-09-06 | hosts api.sedoo.fr | paths -
aviso-fes | user-decision | read 2026-09-06 | hosts www.aviso.altimetry.fr | paths -
gsfc-got | ship+attr | read 2026-09-06 | hosts earth.gsfc.nasa.gov | paths -
seanoe-eot20 | ship+attr | read 2026-10-24 | hosts www.seanoe.org | paths -
tuwien-vmf | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-gnss/tests/fixtures/vmf3_5x5_20181125.H00 crates/spacelab-gnss/assets/gpt3_5.grd
galileo-icd | ship+attr | read 2026-10-20 | hosts - | paths crates/spacelab-ionosphere/assets/ccir*.asc crates/spacelab-ionosphere/assets/modipNeQG_wrapped.asc crates/spacelab-ionosphere/tests/fixtures/nequick_g_jrc_stec.tsv
nequick-jrc | internal-only | read 2026-10-20 | hosts github.com | paths -
ccsds-examples | ship | read 2026-09-06 | hosts - | paths crates/spacelab-ndm/tests/fixtures/ccsds_*.*
gps-almanac | ship | read 2026-09-06 | hosts - | paths crates/spacelab-gnss/tests/fixtures/almanac.*.txt
oracle-outputs | ship | read 2026-09-06 | hosts - | paths crates/spacelab-control/tests/fixtures/*.txt crates/spacelab-coverage/tests/fixtures/*.txt crates/spacelab-earth/tests/fixtures/*_pyerfa.csv crates/spacelab-fa/tests/fixtures/*_pyerfa.csv crates/spacelab-frames/tests/fixtures/*_pyerfa.csv crates/spacelab-frames/tests/fixtures/orekit_teme_itrf.* crates/spacelab-geometry/tests/fixtures/qhull_oracle.txt crates/spacelab-ephemeris/tests/fixtures/neptune_*.txt crates/spacelab/tests/fixtures/neptune_*.txt crates/spacelab/tests/fixtures/orekit_*.csv crates/spacelab-propagation/tests/fixtures/orekit_dsst_*.csv crates/spacelab-propagation/tests/fixtures/lunar_gravity_oracle.txt crates/spacelab-sensors/tests/fixtures/pyorbital_*.txt crates/spacelab-earthvision/tests/fixtures/slanted_edge.txt crates/spacelab-earthvision/tests/fixtures/gas_transmittance.txt crates/spacelab-vision/tests/fixtures/* crates/spacelab-gnss/tests/fixtures/rinex3_obs_handbuilt.24o
pyspectral-lut | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/pyspectral_rayleigh.txt
aeronet | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/aeronet_*.txt
radcalnet | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/radcalnet_*.txt
mpi-mainz-ozone | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/ozone_cross_section.txt
pleiades-s2p-rpc | ship+attr | read 2026-09-06 | hosts - | paths crates/spacelab-earthvision/tests/fixtures/pleiades_reunion.rpb
cara-cdms | internal-only | read 2026-09-06 | hosts github.com raw.githubusercontent.com | paths -
hwm14 | internal-only | read 2026-09-06 | hosts codeload.github.com | paths -
neptune | internal-only | read 2026-09-06 | hosts codeload.github.com | paths -
nrlmsise00-fortran | ship | read 2026-09-06 | hosts api.github.com raw.githubusercontent.com | paths -
hitran | internal-only | read 2026-09-06 | hosts - | paths -
mt-ckd | internal-only | read 2026-09-06 | hosts - | paths -
nrlmsis2 | internal-only | read 2026-09-06 | hosts - | paths -
dtm2020 | restricted | read 2026-09-04 | hosts - | paths -
wdc-kyoto | restricted | read 2026-09-05 | hosts - | paths -
usgs-dst | ship | read 2026-10-25 | hosts - | paths -
esa-swarm-dns | user-decision | read 2026-09-14 | hosts swarm-diss.eo.esa.int earth.esa.int | paths -
space-track | user-decision | read 2026-09-06 | hosts - | paths -
cddis | user-decision | read 2026-09-06 | hosts - | paths -
noaa-wam-ipe | ship+attr | read 2026-10-24 | hosts noaa-nws-wam-ipe-pds.s3.amazonaws.com | paths -
noaa-goes-euvs | ship | read 2026-09-14 | hosts data.ngdc.noaa.gov www.ncei.noaa.gov | paths -
sdo-eve | ship | read 2026-09-14 | hosts lasp.colorado.edu | paths -
lisird-composites | internal-only | read 2026-09-14 | hosts lasp.colorado.edu | paths -
bremen-mgii | internal-only | read 2026-09-14 | hosts www.iup.uni-bremen.de | paths -
```

**The storm driver, decided by the rows above.** JB2008's storm term is Dst-driven and DTM2020's is
Kp-driven; the NRLMSISE-00 and JB2008 validation runs use ap and SET's Dst-derived ΔTc respectively.
Kyoto's Dst is closed to commercial applications, USGS's is empty, and SET's ΔTc has no stated terms.
The one open, continuous, commercially usable storm index is GFZ's Hpo family (CC BY 4.0, from 1995),
which is why our own thermosphere model (`plans/16` Ф15) is driven by Hp60/ap and not by Dst, and why
a Dst-driven ΔTc rebuilt from Kyoto data (Ф18.A) is a *validation* input and nothing sold depends on it.

## Not for public release

Four models are kept in the development tree and are not part of the public release. The public
tree is cut from the development tree by a script, never edited by hand.

* **DTM2020.** §1's licence forbids dissemination of the code and of its coefficients, and asking
  CNES cannot fix that: the licence on offer is non-transferable, and a permissive crate is a
  sub-licence to everyone who downloads it. Commercial use is a separate question and is **not**
  resolved by excluding the module: "academic, non-commercial purposes only" binds our own use too.
* **JB2008.** §3: SET's licence agreement forbids the translation and its distribution, and the
  validation fixtures are excerpts of index files it forbids redistributing.
* **NeQuick-ITU and the ITU-R P.834 mapping function.** Their data, the ITU 1°×2° MODIP grid and the
  P.834 coefficient grid (`itu-r-grids` in the data register), carry ITU copyright and no grant.
  NeQuick-G does not need them: its CCIR maps and MODIP grid are the Galileo ICD's own.

Where this documentation quotes a number measured with one of these models, the number is the
development tree's and is reproducible only there.


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
