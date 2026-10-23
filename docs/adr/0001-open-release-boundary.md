# 1. The open-release boundary is the whole library minus excluded roots

*Status:* accepted, 2026-09-06 (`plans/16` D-8). *Supersedes:* nothing. *Depends on:* ADR is the
record; `docs/provenance.md`'s release block is the enforcement.

## Context

spacelab is 32 crates in one workspace and has never been published — no tags, no crates.io
namespace, no public history. The open release therefore has a free choice of boundary, and that
choice is made once: what is left out on the first day is a capability the public library is
documented never to have had, and what is put in is a surface that Hyrum's law will make permanent.

Two facts constrain it.

**One module cannot ship.** DTM2020's CNES licence is academic and non-commercial, forbids
modification, and forbids dissemination of the code "or any of its supporting data". The licence on
offer is non-transferable, and a permissive crate is a sub-licence to everyone who downloads it, so
asking CNES cannot fix it (`plans/16` D-2). The same shape reaches a handful of oracles — HITRAN,
MT_CKD, HWM14, NRLMSIS, HASDM — but those are *run* rather than vendored, so they never enter the
tree and never reach this decision.

**The verification page is the product's argument.** `docs/verification.md` is 8 000 lines of
measured agreement against agency products, and its value is that a reader can run what it claims.
Every number in it names a test file, and a number whose test is not public is a claim rather than a
verification.

## Decision

**The public tree is the whole workspace minus a small set of named excluded roots, published as a
new repository whose initial commit is the cleaned tree.**

The boundary is expressed as data, not as judgement, in `docs/provenance.md`'s release block:

* `exclude:` names a restricted root. Today: `crates/spacelab-propagation/src/dtm2020` and
  `derivations/dtm2020_tables.py`, and since 2026-10-20 NeQuick-ITU and ITU-R P.834 with their ITU
  grids.
* `name:` adds an identifier a single-file root is reached through, since `mod itu;` never spells
  `itu.rs`.
* `depends:` names every file *outside* those roots whose text mentions one, because each has to be
  edited or dropped to cut a tree that compiles.

`tools/check-provenance.py --check` compares sets of files: a listed file may gain or lose mentions
freely, but a file that *starts* naming a restricted module fails the build. That is what turns "we
will remember to strip this" into a gate, and it fires on the commit that creates the coupling rather
than on release day.

History is not rewritten. A new repository with a cleaned initial commit costs nothing today, because
nothing has been published, and costs more with every month it is deferred.

## Consequences

* **The verified pillars are public**, which is the point: SGP4, DSST, the tide models, the
  integrators, the GNSS stack, the frames, the estimators and SDTM-1 all ship with the tests that
  produce the published numbers.
* **§10.2's three-model density comparison becomes a two-model comparison** in the public tree, and
  its recorded numbers change with it. The private tree keeps DTM2020 and the non-commercial oracles
  (`plans/16` Ф16.D), and publications may cite pre-2026-09-05 numbers with their date (D-6).
* **The measured coupling is small, which is why this is affordable**: two lines of
  `spacelab-propagation/src/lib.rs`, two intra-doc links, one enum variant in `measured_density.rs`,
  and the documentation. Neither the C ABI nor the Python bridge exposes DTM2020 at all.
* **The boundary can move outward but not inward.** Adding a crate later is a release; removing one
  is a breaking change for every downstream user. So a root is excluded only when a licence forces
  it, never because the code is unpolished.
* **The ITU grids are excluded, not waited for** (`plans/16` D-17, 2026-10-20). `itu-r-grids` is
  still unread and §5.10 has measured regeneration from the open DGRF series shut. NeQuick-G no longer
  needs those files: its CCIR maps and MODIP grid are now the Galileo ICD's own, © EU with
  attribution. So the cut removes only NeQuick-ITU and `itu834_mapping`, and `--public` passes
  because no file of that row reaches the public tree. An ITU grant would bring both back.

## Alternatives considered

**A curated core — publish a subset chosen for polish.** Rejected. The verification page is the
moat, and it cites test files across nearly every crate; a subset makes most of it unreproducible by
a reader, which converts the strongest asset into marketing copy. It also front-loads a judgement
call per crate that nobody has a criterion for, where the licence rule is mechanical.

**AGPL over `MIT OR Apache-2.0`.** Rejected as a separate decision (D-1) and worth restating here
because it is the usual reflex for a boundary question: the defensible position is the verification,
not copyleft, and a copyleft licence would keep exactly the users who cannot adopt it.

**Rewrite history to remove DTM2020, then publish this repository.** Rejected. A filtered history is
a claim about every commit in it, and the claim is hard to check and easy to get wrong; a new
repository with one honest initial commit makes the same guarantee trivially verifiable.

**Ship DTM2020 and ask CNES for permission.** Rejected on the licence text: the offer is
non-transferable, and a permissive crate sub-licences every downloader. Commercial use of DTM2020 in
our *own* private work is a separate question the letter in `plans/15` addresses either way.
