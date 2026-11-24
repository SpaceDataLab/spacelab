#!/usr/bin/env python3
"""Every data source has a publication class, and no public line prints a number its class forbids
(`plans/18` Ф14.X, rule 18, §1.7).

`docs/provenance.md`'s data register gives each source a class, in the prose table's Class column and
in the `class` field of the block below it:

  * `ship` -- the data may be in the public tree and the wheel (the `ship` and `ship+attr` verdicts);
  * `internal` -- used in the development tree; nothing derived from it is public;
  * `number-public` -- the source stays out; a number computed from it may be printed, on its terms;
  * `number-private` -- the source stays out, and so does every number computed from it;

or `-` while its terms are unread or wait on the user's decision, when the prose cell says why
("not read:", "open:"). A `number-private` row adds `names`, a case-insensitive regular expression
for the words a line names the source by, and this check refuses every public line that names it
beside a number of ours: Ф14.Z's DTM-2020 gate (D-32), read from the register instead of written for
one model.

    uv run --no-project python -I tools/check-classes.py            # the list, by source
    uv run --no-project python -I tools/check-classes.py --check    # fail on a class or a number

The list is read in the public view of `tools/check-set-numbers.py`, over every text file: the files
git tracks or would track, without the release block's excluded roots, without the lines a private
marker removes and without the table columns a column marker removes, as the public cut writes them.
The public cut runs the same refusal (`refused_in`) over the tree it has written, where the markers
are gone and the view is the tree itself. Three patterns, for each scanned source:

  * narrow -- a line names the source beside a percentage;
  * wide -- a line names the source and still holds a digit once its names, model names, sections,
    dates and tickets are cut out of it, or it is a data row of a table whose header names the source
    in a column;
  * paragraph -- a paragraph, list item or comment block names the source on one line and holds such
    a digit on another, as "DTM-2020 reads" above "20.2 %" does.

The wide and paragraph patterns are upper bounds. None sees a number that stands in a list "in the
table's order" under a table whose row names the source: that is read by hand. `--check` fails when

  * a row has no class or one outside the four, `-` under a verdict other than `unread` or
    `user-decision`, `ship` without a shipping verdict or the reverse, or a prose Class cell that does
    not say the block's class;
  * a `number-private` row has no `names` that compile, or a row of another class has names;
  * a hit stands in the public view and is not in REVIEWED, the hits that were read by hand and print
    no number of ours computed from a scanned source;
  * a REVIEWED fragment no longer matches a hit, so the reading is stale.

`internal` rows carry no names and are not scanned: today's are index series and composites that
other open products share a name with, so a name finds the open product's numbers too. A
`number-public` row's own terms may set a condition, which is its own check (`check-set-numbers.py`).
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
DATA = "## The data register"
CLASSES = ("ship", "internal", "number-public", "number-private")
UNCLASSED = "-"
# A class may wait while the terms are unread or the user decides; the prose cell says which.
WAITING = ("unread", "user-decision")
WAITING_CELL = ("not read:", "open:")
SHIPPING = ("ship", "ship+attr")
SCANNED = "number-private"
# A list item's marker: `- `, `* `, `+ ` or `1. `.
ITEM = re.compile(r"([-*+]|\d+\.)\s")
# This check and its test plant the numbers they refuse.
SELF = ("tools/check-classes.py", "tools/test_check_classes.py")

# Wide hits read by hand that print no number of ours computed from a scanned source, each by its file
# and a fragment of its line. A fragment rather than the line, so that this file prints no such number
# either. The DTM-2020 entries were read on 2026-11-24 for Ф14.Z, the rest on 2026-11-24 for Ф14.X.
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
    # Read for Ф14.X: the NRLMSIS 2 hits (D-30 (c)) and the files beyond documents and comments.
    (
        "crates/spacelab-propagation/src/density_rom.rs",
        "amplitudes from drivers (F10.7, Ap, …)",
        "an index name",
    ),
    (
        "crates/spacelab-propagation/tests/measured_density.rs",
        "read 1.36 and 2.03 on the quiet day",
        "JB2008's and NRLMSISE-00's numbers",
    ),
    (
        "derivations/libm_pow_sites.tsv",
        "(fixed test inputs, gamma bound estimated",
        "line numbers and resource bounds",
    ),
    ("docs/limitations.md", "- **NRLMSIS 2.0 / 2.1.** The NRL licence", "versions"),
    ("docs/limitations.md", "US 10,641,925 B2 claims", "a patent number"),
    ("docs/limitations.md", "remains a dated oracle number", "a ticket family, Ф15"),
    (
        "docs/provenance.md",
        "| `nrlmsis2` | NRLMSIS 2.0 / 2.1 via",
        "versions, a patent number",
    ),
    (
        "docs/provenance.md",
        "| `tianmu-ngms` | TianMu-1 NGMS",
        "TianMu-1's record, cited",
    ),
    (
        "docs/provenance.md",
        "| class number-private | names",
        "the register's names field",
    ),
    (
        "docs/verification.md",
        "| NRLMSIS 2.0 (2020) / 2.1 (2022) |",
        "versions, years, a count",
    ),
    (
        "docs/verification.md",
        "read 1.31 and 1.85 in the F10.7 < 100 bin",
        "JB2008's and NRLMSISE-00's numbers",
    ),
    (
        "docs/verification.md",
        "Ranjan et al. (2024), arXiv:2411.14071",
        "Ranjan et al.'s, Lei et al.'s and Licata et al.'s numbers, cited",
    ),
    (
        "docs/verification.md",
        "most of all in the recovery of 2–3 November",
        "Licata et al. (2022)'s finding, cited",
    ),
    (
        "docs/verification.md",
        "**TianMu-1.** One storm, 10–12 October 2024",
        "dates of the record's storm",
    ),
    ("docs/verification.md", "columns are not read.", "dates"),
    (
        "docs/verification.md",
        "RMSE of the orbit means in 1e-13 kg/m³",
        "a unit",
    ),
    ("justfile", "jb2008 and harris_priester mutants", "mutant names"),
    ("justfile", "359 logs of 364", "a count of logs"),
    (
        "tools/neptune-oracle/decks/lro-matched/neptune.inp",
        "4 = NRLMSIS2.0",
        "an option of NEPTUNE's deck",
    ),
    (
        "tools/neptune-oracle/decks/sentinel1b-matched/neptune.inp",
        "4 = NRLMSIS2.0",
        "an option of NEPTUNE's deck",
    ),
    ("tools/tianmu-on2.py", "recovery 15:00 UT on 11 October", "times and dates"),
]


def rows_of(register: str) -> list[dict]:
    """The data register's block rows, parsed by `check-provenance.py`."""
    spec = importlib.util.spec_from_file_location(
        "check_provenance", HERE / "check-provenance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.data_rows(register)


def class_cells(register: str) -> dict[str, str]:
    """Each prose row's Class cell, by its key."""
    prose = register.split(DATA, 1)[1].split("```")[0] if DATA in register else ""
    cells = {}
    for line in prose.splitlines():
        found = re.match(r"\| `([^`]+)` \|", line)
        if found:
            parts = view.BORDER.split(line.strip())
            cells[found.group(1)] = parts[-2].strip() if len(parts) > 2 else ""
    return cells


def class_problems(register: str) -> list[str]:
    problems = []
    cells = class_cells(register)
    for row in rows_of(register):
        key, verdict, kind, names = (
            row["key"],
            row["verdict"],
            row["class"],
            row["names"],
        )
        where = f"{REGISTER}: row {key}"
        if kind not in (*CLASSES, UNCLASSED):
            problems.append(f"{where}: class '{kind}' is not one of {CLASSES} or '-'")
        if kind == UNCLASSED and verdict not in WAITING:
            problems.append(f"{where}: no class, under the verdict '{verdict}'")
        elif kind != UNCLASSED and (kind == "ship") != (verdict in SHIPPING):
            problems.append(f"{where}: class '{kind}' under the verdict '{verdict}'")
        cell = cells.get(key)
        if cell is None:
            problems.append(f"{where}: no prose row")
        elif kind == UNCLASSED and not cell.startswith(WAITING_CELL):
            problems.append(
                f"{where}: no class, and the Class cell says no reason: {cell}"
            )
        elif kind != UNCLASSED and not cell.startswith(f"`{kind}`"):
            problems.append(f"{where}: the Class cell does not say `{kind}`: {cell}")
        if kind == SCANNED and not names:
            problems.append(f"{where}: class {SCANNED} with no names to find it by")
        if names and kind != SCANNED:
            problems.append(f"{where}: names under class '{kind}'")
        if names:
            try:
                re.compile(names, re.I)
            except re.error as e:
                problems.append(f"{where}: names do not compile: {e}")
    return problems


def scanned(register: str) -> list[tuple[str, re.Pattern[str]]]:
    """`(key, names)` of every row whose numbers no public line may print."""
    found = []
    for row in rows_of(register):
        if row["class"] == SCANNED and row["names"]:
            try:
                found.append((str(row["key"]), re.compile(str(row["names"]), re.I)))
            except re.error:
                continue
    return found


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


def paragraph_hits(
    rel: str, lines: list[str | None], name: re.Pattern[str]
) -> list[tuple[int, str]]:
    """`(line number, unit text)` where a unit names the source and a line not naming it holds a digit."""
    found = []
    for unit in units(rel, lines):
        text = [lines[i] or "" for i in unit]
        if not any(name.search(x) for x in text):
            continue
        digits = [
            i
            for i, x in zip(unit, text, strict=True)
            if not name.search(x) and view.DIGIT.search(view.NOT_A_NUMBER.sub("", x))
        ]
        if digits:
            found.append((digits[0] + 1, " ".join(x.strip() for x in text)))
    return found


def all_hits(
    rel: str, lines: list[str | None], name: re.Pattern[str]
) -> list[tuple[int, str, bool, str]]:
    """`(line number, kind, narrow, text)` for every hit of the three patterns, in line order."""
    found = [
        (n, kind, narrow, (lines[n - 1] or "").strip())
        for n, kind, narrow in view.hits(lines, name, own=True)
    ]
    found += [
        (n, "paragraph", False, text) for n, text in paragraph_hits(rel, lines, name)
    ]
    return sorted(found)


def refusals(
    files: dict[str, list[str | None]],
    sources: list[tuple[str, re.Pattern[str]]],
    reviewed: list[tuple[str, str, str]],
) -> tuple[list[str], set[tuple[str, str]]]:
    """The hits no REVIEWED entry answers, and the entries that answered one."""
    problems = []
    used: set[tuple[str, str]] = set()
    for key, name in sources:
        for rel, lines in files.items():
            for n, kind, _, text in all_hits(rel, lines, name):
                # A paragraph's text holds its lines', so one entry may answer a line and its paragraph.
                known = {(r, f) for r, f, _ in reviewed if r == rel and f in text}
                used |= known
                if known:
                    continue
                problems.append(
                    f"{rel}:{n}: a number of {key}, class {SCANNED} ({kind}): "
                    f"{(lines[n - 1] or '').strip()[:100]}"
                )
    return problems, used


def register_text(files: dict[str, list[str | None]]) -> str:
    lines = files.get(REGISTER)
    if lines is None:
        raise SystemExit(f"{REGISTER} is not in the tree")
    return "\n".join(x for x in lines if x is not None) + "\n"


def problems_of(
    files: dict[str, list[str | None]], reviewed: list[tuple[str, str, str]]
) -> list[str]:
    register = register_text(files)
    problems = class_problems(register)
    refused, used = refusals(files, scanned(register), reviewed)
    problems += refused
    for rel, text in sorted({(r, f) for r, f, _ in reviewed} - used):
        problems.append(f"{rel}: REVIEWED fragment matches no hit, re-read it: {text}")
    return problems


def tree_files(tree: pathlib.Path) -> dict[str, list[str | None]]:
    """Every text file under `tree`, as it is: the public tree carries no markers to apply."""
    files: dict[str, list[str | None]] = {}
    for path in sorted(p for p in tree.rglob("*") if p.is_file()):
        rel = path.relative_to(tree).as_posix()
        data = path.read_bytes()
        if rel in SELF or rel.startswith(".git/") or b"\0" in data:
            continue
        files[rel] = list(data.decode("utf-8", errors="replace").splitlines())
    return files


def refused_in(tree: pathlib.Path) -> list[str]:
    """What the public cut refuses in the tree it wrote: a missing class, or a forbidden number.

    A stale REVIEWED entry is not the cut's to refuse: `--check` reports it in the development tree.
    """
    files = tree_files(tree)
    register = register_text(files)
    return class_problems(register) + refusals(files, scanned(register), REVIEWED)[0]


def main() -> int:
    check = "--check" in sys.argv[1:]
    files = view.public_files(ROOT, SELF, every=True)
    register = register_text(files)
    for key, name in scanned(register):
        counts = {"narrow": 0, "wide": 0, "paragraph": 0}
        for rel, lines in files.items():
            for n, kind, is_narrow, text in all_hits(rel, lines, name):
                pattern = (
                    "unit" if kind == "paragraph" else "narrow" if is_narrow else "wide"
                )
                counts["paragraph" if kind == "paragraph" else "wide"] += 1
                counts["narrow"] += is_narrow
                if not check:
                    mark = "reviewed" if view.reviewed_as(rel, text, REVIEWED) else kind
                    print(
                        f"{key}: {rel}:{n}: {pattern:6} {mark:9} "
                        f"{(lines[n - 1] or '').strip()[:110]}"
                    )
        print(
            f"{key}: {counts['narrow']} narrow, {counts['wide']} wide, "
            f"{counts['paragraph']} paragraph"
        )
    rows = rows_of(register)
    tally = {k: sum(r["class"] == k for r in rows) for k in (*CLASSES, UNCLASSED)}
    print(f"{len(rows)} sources: " + ", ".join(f"{v} {k}" for k, v in tally.items()))
    if not check:
        return 0
    problems = problems_of(files, REVIEWED)
    for p in problems:
        print(f"refused: {p}")
    print(f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
