#!/usr/bin/env python3
"""The public lines that print a number computed from DTM-2020 (`plans/18` Ф14.Z, D-32).

CNES's licence for DTM-2020 forbids disseminating the model "or any of its supporting data"
(`docs/provenance.md` §1). The register gives every number computed from it the class
`number-private` (D-32): the development tree keeps those numbers behind private markers, a table
keeps a column of them behind a column marker, and the public tree prints none. A line may name the
model with no number, and a number a third party published about it stays as a citation.

    uv run --no-project python -I tools/check-dtm-numbers.py            # the list, narrow and wide
    uv run --no-project python -I tools/check-dtm-numbers.py --check    # fail on a number in public

The list is read in the public view of `tools/check-set-numbers.py`: the files git tracks or would
track, without the release block's excluded roots, without the lines a private marker removes and
without the table columns a column marker removes, as the public cut writes them. In the public tree
the markers are gone and the view is the tree itself. Three patterns, the first two that check's:

  * narrow -- a line names the model beside a percentage;
  * wide -- a line names the model and still holds a digit once names, sections, dates and tickets
    are cut out of it, or it is a data row of a table whose header names the model in a column;
  * paragraph -- a paragraph, list item or comment block names the model on one line and holds such
    a digit on another, as "DTM-2020 reads" above "20.2 %" does.

The wide and paragraph patterns are upper bounds. None sees a number that stands in a list "in the
table's order" under a table whose row names the model: that is read by hand. `--check` fails when

  * a wide hit stands in the public view and is not in REVIEWED, the hits that were read by hand and
    print no number of ours computed from the model;
  * a REVIEWED fragment no longer matches a hit, so the reading is stale;
  * the register row no longer records the class and the decision.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "check_set_numbers", HERE / "check-set-numbers.py"
)
assert _spec is not None and _spec.loader is not None
view = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(view)

ROOT = HERE.parent
REGISTER = "docs/provenance.md"
# The register row's key, spelled so that this file names no restricted module for the cut to find.
ROW = "| `dtm" + "2020` |"
ROW_RECORDS = ("`number-private`", "D-32")
NAME = re.compile(r"(?i)DTM-?2020")
# A list item's marker: `- `, `* `, `+ ` or `1. `.
ITEM = re.compile(r"([-*+]|\d+\.)\s")
# This check and its test plant the numbers they refuse.
SELF = ("tools/check-dtm-numbers.py", "tools/test_check_dtm_numbers.py")

# Wide hits read by hand on 2026-11-24 that print no number of ours computed from the model, each by
# its file and a fragment of its line. A fragment rather than the line, so that this file prints no
# such number either.
REVIEWED: list[tuple[str, str, str]] = [
    (".github/workflows/ci.yml", "first draft named 15 Orekit-derived", "a file count"),
    ("CONTRIBUTING.md", "NRLMSIS 2.x and HWM14 (NRL)", "model names"),
    (
        "crates/spacelab-propagation/src/sdtm/mod.rs",
        "51.7 % mean absolute error on the twelve satellite-months",
        "NRLMSISE-00's number",
    ),
    (
        "crates/spacelab-propagation/src/sdtm/mod.rs",
        "4, GRS80). Must exceed",
        "a column number and a datum",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "daily F10.7 solar flux and Ap geomagnetic index",
        "an index name",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "the radio flux at 1 AU",
        "a wavelength and a unit",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "81 day AVERAGE OF F10.7 FLUX",
        "the source's own words",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "last 81 days at t",
        "a window length",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "F10 is the familiar 10.7 cm radio flux",
        "index names",
    ),
    (
        "docs/adr/0001-open-release-boundary.md",
        "MT_CKD, HWM14, NRLMSIS, HASDM",
        "model names",
    ),
    (
        "docs/adr/0001-open-release-boundary.md",
        "`, and since 2026-10-20 ",
        "a recommendation's number",
    ),
    (
        "docs/limitations.md",
        "was fitted on 365 214 rows of the TU Delft v02 corpus",
        "SDTM-1's and NRLMSISE-00's numbers",
    ),
    (
        "docs/limitations.md",
        "it reads 21.2 % MAPE against",
        "SDTM-1's, NRLMSISE-00's and the clean-room JB2008's numbers",
    ),
    (
        "docs/limitations.md",
        "moves the real density by a factor of 2.7",
        "the measured bulge, and a count of models",
    ),
    ("docs/provenance.md", "— the one that blocks", "a section number"),
    ("docs/provenance.md", "a faithful f64 port of the reference Fortran", "a type"),
    (
        "docs/provenance.md",
        "J. Space Weather Space Clim. 11, 47 (2021)",
        "a citation",
    ),
    (
        "docs/provenance.md",
        "beside a percentage and 50 by the wide pattern",
        "counts of lines at tag v0.3.0",
    ),
    ("docs/provenance.md", "USGS's Dst4 and", "an index name"),
    ("docs/verification.md", "(T₁₂₀/T)^(1 + α + γ)", "an exponent"),
    (
        "docs/verification.md",
        "added 14 and 23 lines above every anchored line",
        "line counts",
    ),
    ("docs/verification.md", "| 1 219 | 0 | 1 —", "mutant counts"),
    ("docs/verification.md", "Each model wants its own 81-day mean", "a window length"),
    ("docs/verification.md", "the radio flux at 1 AU", "a unit"),
    ("docs/verification.md", "are *fits* to 3-hourly", "a cadence"),
    (
        "docs/verification.md",
        "whose honest error is **50.1 %**",
        "SDTM-1's number",
    ),
    (
        "docs/verification.md",
        "Bruinsma & Laurens report",
        "Bruinsma & Laurens (2024)'s numbers, cited",
    ),
    (
        "docs/verification.md",
        "on single-peak storms, with StD",
        "Bruinsma & Laurens (2024)'s numbers, cited",
    ),
    (
        "docs/verification.md",
        "6356.766 km sits",
        "the reference radius and a column number",
    ),
    (
        "docs/verification.md",
        "its EGU25-1673 abstract",
        "Wang et al. (2026)'s ranking, cited",
    ),
    (
        "docs/verification.md",
        "They find the MSIS models low by about 20–30 %",
        "Wang et al. (2026)'s and NRLMSISE-00's numbers",
    ),
    (
        "docs/verification.md",
        "fall in density after Halloween 2003",
        "the measurement's and SDTM-1's numbers",
    ),
    (
        "docs/verification.md",
        "before 2020-01-01 (255 993 rows)",
        "row counts, and a fit window cited from Bruinsma & Boniface (2021)",
    ),
    (
        "docs/verification.md",
        "doi:10.1029/2025EA004454) drive",
        "Zhu et al. (2026)'s numbers, cited",
    ),
    (
        "docs/verification.md",
        "and loses in super storms",
        "Zhu et al. (2026)'s numbers, cited",
    ),
    ("docs/verification.md", "319 modules, 101 name a check", "module counts"),
    (
        "tools/fetch-density-data.sh",
        "J. Space Weather Space Clim. 2023, CC BY 4.0",
        "a citation",
    ),
    ("tools/fetch-density-data.sh", "it is ~30 MB a month", "a download size"),
]


def units(rel: str, lines: list[str | None]) -> list[list[int]]:
    """The prose units of a file's public view, as 0-based line indices.

    A unit is a paragraph or a list item of a document, or a run of comment lines in code. A blank
    line, a table row, a heading, a fence, a line of code or a new list item ends one, and a line
    inside a document's fence is a unit of its own. A line the cut deletes ends nothing, since the
    lines around it meet in the public tree.
    """
    suffix = pathlib.PurePosixPath(rel).suffix
    prefixes = view.COMMENT.get(suffix)
    found: list[list[int]] = []
    current: list[int] = []
    fenced = False
    for i, line in enumerate(lines):
        if line is None:
            continue
        stripped = line.strip()
        if prefixes is not None:
            if not stripped.startswith(prefixes):
                stripped = ""
            stripped = stripped.lstrip("/!#*").strip()
        elif stripped.startswith("```"):
            fenced = not fenced
        ends = fenced or not stripped or stripped.startswith(("|", "#", "```"))
        if ends or ITEM.match(stripped):
            if current:
                found.append(current)
            current = []
        if not ends:
            current.append(i)
    if current:
        found.append(current)
    return found


def paragraph_hits(rel: str, lines: list[str | None]) -> list[tuple[int, str]]:
    """`(line number, unit text)` where a unit names the model and a line not naming it holds a digit."""
    found = []
    for unit in units(rel, lines):
        text = [lines[i] or "" for i in unit]
        if not any(NAME.search(x) for x in text):
            continue
        digits = [
            i
            for i, x in zip(unit, text, strict=True)
            if not NAME.search(x) and view.DIGIT.search(view.NOT_A_NUMBER.sub("", x))
        ]
        if digits:
            found.append((digits[0] + 1, " ".join(x.strip() for x in text)))
    return found


def all_hits(rel: str, lines: list[str | None]) -> list[tuple[int, str, bool, str]]:
    """`(line number, kind, narrow, text)` for every hit of the three patterns, in line order."""
    found = [
        (n, kind, narrow, (lines[n - 1] or "").strip())
        for n, kind, narrow in view.hits(lines, NAME)
    ]
    found += [(n, "paragraph", False, text) for n, text in paragraph_hits(rel, lines)]
    return sorted(found)


def problems_of(
    files: dict[str, list[str | None]], reviewed: list[tuple[str, str, str]]
) -> list[str]:
    problems = []
    used = set()
    for rel, lines in files.items():
        for n, kind, _, text in all_hits(rel, lines):
            # A paragraph's text holds its lines', so one entry may answer a line and its paragraph.
            known = {(r, f) for r, f, _ in reviewed if r == rel and f in text}
            used |= known
            if known:
                continue
            problems.append(
                f"{rel}:{n}: a number of the model ({kind}): {(lines[n - 1] or '').strip()[:100]}"
            )
    for rel, text in sorted({(r, f) for r, f, _ in reviewed} - used):
        problems.append(f"{rel}: REVIEWED fragment matches no hit, re-read it: {text}")
    register = files.get(REGISTER)
    row = next((x for x in register or [] if x and x.startswith(ROW)), None)
    if row is None:
        problems.append(f"{REGISTER}: no register row {ROW}")
    else:
        problems += [
            f"{REGISTER}: row {ROW} does not record '{w}'"
            for w in ROW_RECORDS
            if w not in row
        ]
    return problems


def main() -> int:
    check = "--check" in sys.argv[1:]
    files = view.public_files(ROOT, SELF)
    narrow: dict[str, int] = {}
    wide: dict[str, int] = {}
    paragraph: dict[str, int] = {}
    for rel, lines in files.items():
        for n, kind, is_narrow, text in all_hits(rel, lines):
            if kind == "paragraph":
                paragraph[rel] = paragraph.get(rel, 0) + 1
            else:
                narrow[rel] = narrow.get(rel, 0) + is_narrow
                wide[rel] = wide.get(rel, 0) + 1
            if not check:
                mark = "reviewed" if view.reviewed_as(rel, text, REVIEWED) else kind
                pattern = (
                    "narrow" if is_narrow else "wide" if kind != "paragraph" else "unit"
                )
                print(
                    f"{rel}:{n}: {pattern:6} {mark:9} {(lines[n - 1] or '').strip()[:110]}"
                )
    print()
    for rel in sorted(set(wide) | set(paragraph)):
        print(
            f"{rel}: {narrow.get(rel, 0)} narrow, {wide.get(rel, 0)} wide, "
            f"{paragraph.get(rel, 0)} paragraph"
        )
    print(
        f"total: {sum(narrow.values())} narrow, {sum(wide.values())} wide, "
        f"{sum(paragraph.values())} paragraph"
    )
    if not check:
        return 0
    problems = problems_of(files, REVIEWED)
    for p in problems:
        print(f"refused: {p}")
    print(f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
