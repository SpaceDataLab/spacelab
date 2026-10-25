# Contributing

## Provenance: derive from the mathematics

**Science is free. Source code is expression.** Both halves matter.

Equations, algorithms and measured constants are not copyrightable. The IERS
Conventions, the primary literature, the textbooks, and every coefficient table
IERS publishes are ours to read and implement without restriction.

Implementations are a different thing, and translating one between languages is a
derivative work in the same sense a translated novel is.

### The rule

**Read anything.** Papers, conventions, errata, test data, expected-output files, and
other people's source. Reading is not the hazard, and pretending otherwise makes
debugging dishonest: when our TEME vector disagreed with Skyfield's by twenty metres,
the answer was in the `sgp4` crate's source, and going and looking was the right move.

**The hazard is what you write next.** Copyright does not ask whether you copied
literally. Substantial similarity of *expression* is enough, and "I read it, then wrote
it from memory" is precisely the fact pattern that loses. So:

| Activity | |
|---|---|
| Reading source to understand, diagnose, or find which paper a routine implements | **Fine.** |
| Reading data, test vectors, expected outputs, coefficient tables | **Fine.** They are facts, and facts are not copyrightable. |
| Writing a routine here **derived from published mathematics** | **This is the method.** |
| Writing a routine here that is the **expression of an AGPL or SOFA-licensed implementation** | **No.** It relicenses the whole workspace, and that door opens one way. |

The consequence is concrete, not theoretical. CelesTrak's
`fundamentals-of-astrodynamics` is AGPL-3.0, including `valladopy`. If a routine in this
permissive workspace is a derivative of it, the remedy is to relicense everything as AGPL
or to rip the routine out and have someone who never saw it write it again.

**And the engineering reason outranks the legal one.** Transcribing C produces a C API:
out-parameters, integer status codes, frames tracked in the programmer's head. That is
`sofars`, which is MIT and already exists. The typed frame and time-scale algebra that
justifies this workspace is reachable only by deriving from the mathematics. Porting the
code would forfeit the reason for the project.

`cargo deny` cannot detect a human transcribing a licensed implementation. Only this rule
can.

### About Numerical Recipes specifically

*Numerical Recipes* (Press et al., CUP) carries the most restrictive licence of any source
this project has touched — stricter than AGPL, stricter than SOFA. Its own words:

> Without an additional license to use the contained software, this book is intended as a
> text and reference book, **for reading and study purposes only.** … a restricted, limited
> free license for use of the software by the individual owner of a copy of this book **who
> personally keyboards one or more routines into a single computer** is granted …
> **Unlicensed transfer of Numerical Recipes programs to any other format, or to any
> computer except one that is specifically licensed, is strictly prohibited.**

Committing a routine from it to a git repository is exactly that prohibited transfer.

This matters because the coverage engine's event detector refines every sign change with
**Brent's method**, which appears in that book as `zbrent`. The algorithm is not theirs:
it descends from Dekker (1969) and is published in Brent, R. P. (1973), *Algorithms for
Minimization without Derivatives*, ch. 4, as `zeroin`; in Forsythe, Malcolm & Moler (1977);
and on Netlib. Their *expression* of it is theirs.

Take it from the `roots` crate (BSD-2-Clause), or derive it from Brent. Read the book for
why bracketing beats Newton when you have no derivative, and why inverse quadratic
interpolation is safe only inside a bracket. Type nothing out of it.

### About SOFA specifically

A correction to a claim that circulated in early drafts of the design docs: SOFA's
licence **does** permit derivative works. Clause 2 grants use "for any purpose,
including commercial applications, free of charge"; clause 3 grants permission to
"use and adapt its code and algorithms in your own software, on a world-wide,
royalty-free basis."

We decline anyway, for two reasons that have nothing to do with permission.

**The obligations ride along.** A derived work must carry a statement that it is
derived from SOFA and not endorsed by SOFA (3a); its source must describe how it
differs from the original (3b) — forever; routine names may not contain `iau` or
`sofa` (3c); and these requirements propagate to anyone granted a right to modify
our source (3e). Clause 4 forbids bringing SOFA "into disrepute", a term that is
not OSI-approvable and cannot be expressed as an SPDX identifier — which means
every downstream user's licence audit special-cases us, forever. Clause 6 reserves
the right to change the terms of future versions.

**Transcription defeats the purpose.** Porting C yields a C API: out-parameters,
integer status codes, reference frames tracked in the programmer's head. The typed
frame and time-scale algebra that justifies this crate's existence is reachable
only by deriving from the mathematics. Port `sofars` and you have re-derived
`sofars`, which is MIT-licensed and already exists.

SOFA remains useful to us as a **comparator**: its test program `t_sofa_c.c` and
the values it prints are distances we publish, not a proof. Running a program and
recording its output creates no derivative work and incurs no obligation. See
[`tools/sofa-oracle/`](tools/sofa-oracle/).

## A second implementation is a comparison, not a proof

`pyerfa` wraps ERFA. ERFA is a fork of SOFA. `sofars` is a transliteration of SOFA.
Compiled and run over the same 45 epochs spanning ±5000 years, SOFA and ERFA agree
**bit for bit**: one program in three wrappers.

Agreement with any implementation shows a distance, not correctness — whether the
other code descends from the same source or not. This document used to ask for a
second, independent lineage to close that gap; it no longer does. What decides is
the mathematics: the formula derived in Maxima and the shipped expression
subtracted to zero, the same identity in a second engine (FriCAS, PARI/GP), and the
code evaluated against exact or certified values at its limits and special cases —
the gate kinds of `docs/verification.md`, "How to read the *Check* column". SOFA,
ERFA, Orekit and Vallado's code are comparisons, published as distances.

## Comparators under non-commercial terms

Several of the best references this workspace measures itself against are licensed for research
or non-commercial use only: NRLMSIS 2.x and HWM14 (NRL), DTM2020 (CNES), the AER MT_CKD continuum,
SET's HASDM database. `docs/provenance.md` names each with the clause. They are used, and the rule
for using them is fixed so that a paid product can never be found to have depended on one:

- **Read and run, never port, vendor or link.** A `tools/fetch-*.sh` script downloads the reference
  into `~/.cache/spacelab/` (gitignored, outside the tree); a test or a generator runs it and
  records the residual; no line of it and no table from it enters a crate. Where the reference's
  mathematics is published, our implementation is written from the paper with the citation, and the
  reference is a comparison — the position `docs/provenance.md` §2 records for NRLMSISE-00.
- **Its output is a number in `docs/verification.md`, dated.** A verification row may cite an
  NC-licensed comparator's result with the date it was run. That row is a publication of a
  comparison, which every such licence permits, and it is the *only* thing the comparator's output
  becomes.
- **Nothing sold consumes it.** No default, no fallback, no calibration table and no shipped
  coefficient may be fitted to, initialised from or corrected by an NC-licensed reference. A model
  that needs a reference to be useful (a calibration layer, a learned corrector) is fitted to
  open data — TU Delft's densities, GFZ's indices, IGS products — with the NC reference cited as
  a comparison only. The reason is that a licence clause on a run-time input binds the product
  that consumes it, and excluding a file from a tree does not discharge it.
- **A new NC comparator is a register row first.** Before the fetch script, the row in
  `docs/provenance.md` with the clause quoted; `tools/check-provenance.py --data --check` fails on
  a fetch host without one.

## Tolerances are derived, never tuned

A test tolerance is a named constant with a comment deriving it from the
conditioning of the problem. If a test fails, the first question is whether the
tolerance was ever justifiable — not how much to loosen it.

See `crates/spacelab-fa/tests/gmst82_oracle.rs` for the shape of this: the
tolerance is a first-order rounding-error model over the polynomial's
intermediates, and the worst observed deviation uses about 10% of it.

## Before opening a pull request

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo deny check
```

If a commit implements an equation from a book, state in the commit body which
equation, and whether the errata touch it.
