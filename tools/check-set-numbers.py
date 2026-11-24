#!/usr/bin/env python3
"""The public lines that print a number computed with SET's index files as input (`plans/18` Ф14.V).

SET's JB2008 License Agreement (2026-10-15) makes the index files `SOLFSMY.TXT` and `DTCFILE.TXT` part
of its Software. Its §7 lets a number computed with them be published in a report "provided that you
include the acknowledgement and citation set forth in the JB2008 Publication Acknowledgement (Notice
to Users)". JB2008 runs here on those files (`space_weather::jb2008_drivers`), and the register row
`set-jb2008-indices` gives a number computed with them the class `number-public`, provisional (D-38).
Until SET sends the Notice's text, an interim acknowledgement in our own words stands in its place.

    uv run --no-project python -I tools/check-set-numbers.py            # the list, narrow and wide
    uv run --no-project python -I tools/check-set-numbers.py --check    # fail on a missing acknowledgement

The list is read in the public view: the files git tracks or would track, without the release block's
excluded roots and without the lines a private marker removes, as the public cut writes them. Two
patterns:

  * narrow -- a line names JB2008 beside a percentage;
  * wide -- a line names JB2008 and still holds a digit once names, sections, dates and tickets are
    cut out of it, or it is a data row of a table whose header names JB2008 in a column.

The wide pattern is an upper bound: it also catches "JB2008's S10, Y10 and M10". `--check` fails when

  * a wide hit stands in a unit with no acknowledgement (a unit is the whole document for Markdown and
    TeX, the comment block for a comment in code, the whole file for anything else), and it is not in
    REVIEWED, the hits that were read by hand and print no number of ours with SET's files as input;
  * a REVIEWED fragment no longer matches a hit, so the reading is stale;
  * a public line says that SET's index files state no terms, against the register;
  * the register row no longer records §7, the Notice and the class.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTER = "docs/provenance.md"
ROW = "`set-jb2008-indices`"
# Replaced by SET's own wording when the Notice to Users arrives (D-38).
ACK = "JB2008 acknowledgement (interim)"
ROW_RECORDS = ("§7", "Notice to Users", "`number-public`", "provisional")
# Spelled in parts: the cut deletes every line that carries a marker, and this file ships.
LINE = "@" + "private"
BEGIN, END = LINE + "-begin", LINE + "-end"
# In a table's header cell: the cut drops that column from the header and every row below it.
COLUMN = LINE + "-column"
# A cell border; `\|` is a pipe inside a cell.
BORDER = re.compile(r"(?<!\\)\|")
# This check and its test quote the sentences and plant the numbers they refuse.
SELF = ("tools/check-set-numbers.py", "tools/test_check_set_numbers.py")
# The files the public cut deletes besides the release block's roots.
CUT = ("tools/cut-public-tree.py", "CHANGELOG.md")
# The registers whose lines naming a deleted root the cut drops.
REGISTERS = ("docs/provenance.md", "docs/verification.md")
# The per-site ledgers whose rows, data or comment, in a deleted root the cut drops.
SITE_LEDGERS = ("derivations/libm_pow_sites.tsv",)

DOCUMENT = {".md", ".tex"}
COMMENT = {
    ".rs": ("//", "/*", "*"),
    ".c": ("//", "/*", "*"),
    ".h": ("//", "/*", "*"),
    ".cpp": ("//", "/*", "*"),
    ".hpp": ("//", "/*", "*"),
    ".java": ("//", "/*", "*"),
    ".py": ("#",),
    ".sh": ("#",),
    ".toml": ("#",),
    ".yml": ("#",),
    ".yaml": ("#",),
    ".jl": ("#",),
    ".mac": ("/*", "*"),
}

NAME = re.compile(r"JB2008")
# `\s` is Unicode-aware, so it also takes the no-break spaces the documents put before `%`.
PERCENT = re.compile(r"\d\s?%")
# What a line may name without printing a number: model names, sections, tickets, decisions,
# findings, dates, paths and code spans, counts with a unit, versions: `plans/18` M0-11's probe, and
# a paper's AIAA number, which the acknowledgement itself cites.
NOT_A_NUMBER = re.compile(
    r"(?i:dtm)-?20\d\d|JB2008|NRLMSISE?-?\d*(\.\d)?|SDTM-\d|GRACE-FO|§\s?\d+(\.\d+)*|Ф\d+\.[A-Z]+(-\d)?|D-\d+"
    r"|AIAA \d{4}-\d+|F\d{2,3}\b|\b(19|20)\d\d(-\d\d){0,2}\b|plans/\d+|`[^`]*`"
    r"|\b\d+ (km|h|days?|months?|storms?|rows|files|satellites?)\b"
    r"|Orekit \d+(\.\d+)*|v\d+(\.\d+)+|\b0\.\d\.\d\b|#\d+|\bL\d\b|\bB\d\b|\(\d\)|\b\d\.\s"
)
DIGIT = re.compile(r"\d")
RULE = re.compile(r"\|\s*:?-{2,}")
# A sentence that gives SET's index files no terms: the subject and the claim on one line.
SUBJECT = re.compile(
    r"\bSET\b|SOLFSMY|DTCFILE|ΔTc|set-jb2008-indices|Space Environment"
)
NO_TERMS = re.compile(
    r"\b(states?|stated|has|have|with) no (stated )?terms|\bno terms (are )?stated",
    re.I,
)

# Wide hits read by hand on 2026-11-24 that print no number computed with SET's files as input, each
# by its file and a fragment of its line. A fragment rather than the line, so that this file does not
# name a model the public cut has to look for.
REVIEWED = [
    (
        "crates/spacelab-propagation/src/jacchia_bowman/constants.rs",
        "B08 = Bowman et al. (2008)",
        "a citation's short name, B08",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "the radio flux at 1 AU",
        "a unit, 1 AU",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "their 81-day centred means",
        "a window length",
    ),
    (
        "crates/spacelab-propagation/src/space_weather.rs",
        "a bare linear F10.7 term misses",
        "an index name, F10.7",
    ),
    (
        "docs/sdtm1-blind-protocol.md",
        "| JB2008, clean-room (Ф18.A′) | SET's F10/S10/M10/Y10",
        "index names and a window length",
    ),
    (
        "tools/benchmark-splits.py",
        "its Table 9 RE for each model",
        "a paper's table, cited",
    ),
    (
        "tools/benchmark-splits.py",
        "1 - RMSE/JB2008 (persistence, reconstruction)",
        "a column header in code",
    ),
    ("tools/fetch-density-data.sh", "it is ~30 MB a month", "a download size"),
    (
        "tools/hpo-driver-table.py",
        "find JB2008 3-9 % better",
        "Zhu et al. (2026)'s numbers, cited",
    ),
    (
        "tools/solar_proxy_admissibility.py",
        "stand in for JB2008's S10, Y10 and M10",
        "index names",
    ),
]


def load_checker(root: pathlib.Path):
    spec = importlib.util.spec_from_file_location(
        "check_provenance", root / "tools" / "check-provenance.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    module.REGISTER = root / REGISTER
    return module


def without_columns(line: str, columns: set[int]) -> str:
    """A table row without the cells at `columns`, counted from the text before its first border."""
    start = line.index("|")
    cells = BORDER.split(line[start:])
    return line[:start] + "|".join(c for k, c in enumerate(cells) if k not in columns)


def public_lines(text: str, where: str = "") -> list[str | None]:
    """The file's lines as the public cut writes them, `None` where it deletes one."""
    out: list[str | None] = []
    inside = False
    columns: set[int] = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.lstrip("/!#* ").startswith("|"):
            columns = set()
        if BEGIN in line:
            inside = True
            out.append(None)
        elif END in line:
            inside = False
            out.append(None)
        elif inside:
            out.append(None)
        elif COLUMN in line:
            if not line.lstrip("/!#* ").startswith("|"):
                raise SystemExit(f"{where}:{n}: {COLUMN} outside a table row")
            start = line.index("|")
            cells = BORDER.split(line[start:])
            columns = {k for k, c in enumerate(cells) if COLUMN in c}
            out.append(without_columns(line, columns))
        elif LINE in line:
            out.append(None)
        else:
            out.append(without_columns(line, columns) if columns else line)
    return out


def public_files(
    root: pathlib.Path, skip: tuple[str, ...] = SELF, every: bool = False
) -> dict[str, list[str | None]]:
    """The public view of each file, by its path; `every` takes every text file, not the suffixes."""
    listed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    excluded = tuple(load_checker(root).restricted()[0])
    files = {}
    for rel in sorted(p for p in listed if p):
        path = root / rel
        suffix = path.suffix
        if rel.startswith(excluded) or rel in skip + CUT or not path.is_file():
            continue
        if not every and suffix not in DOCUMENT and suffix not in COMMENT:
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        lines = public_lines(data.decode("utf-8", errors="replace"), rel)
        if rel in REGISTERS:
            lines = [
                None if x is not None and x.strip().startswith(excluded) else x
                for x in lines
            ]
        if rel in SITE_LEDGERS:
            lines = [
                None
                if x is not None and x.lstrip("#").lstrip().startswith(excluded)
                else x
                for x in lines
            ]
        files[rel] = lines
    return files


def hits(
    lines: list[str | None], name: re.Pattern[str] = NAME, own: bool = False
) -> list[tuple[int, str, bool]]:
    """`(line number, kind, narrow)` for every wide hit of `name`; kind is `named` or `column`.

    `own` also cuts what `name` matches before looking for a digit, for a name that holds one.
    """

    def has_number(text: str) -> bool:
        return bool(
            DIGIT.search(NOT_A_NUMBER.sub("", name.sub("", text) if own else text))
        )

    found = []
    column = None
    for i, line in enumerate(lines):
        if line is None:
            continue
        body = line.lstrip("/!#* ")
        is_row = body.startswith("|")
        if not is_row:
            column = None
        narrow = bool(name.search(line) and PERCENT.search(line))
        # The next public line, read only for a row that names the model: copying the rest of the
        # list on every line is quadratic, and a data file runs to 10^5 lines.
        if (
            is_row
            and name.search(line)
            and RULE.search(next((x for x in lines[i + 1 :] if x is not None), ""))
        ):
            cells = BORDER.split(body)
            column = next((k for k, c in enumerate(cells) if name.search(c)), None)
            continue
        if is_row and column is not None and not RULE.search(line):
            cells = BORDER.split(body)
            if column < len(cells) and has_number(cells[column]):
                found.append((i + 1, "column", narrow))
                continue
        if name.search(line) and has_number(line):
            found.append((i + 1, "named", narrow))
    return found


def unit(rel: str, lines: list[str | None], n: int) -> list[str]:
    """The lines an acknowledgement for line `n` (1-based) may stand on."""
    present = [x for x in lines if x is not None]
    suffix = pathlib.PurePosixPath(rel).suffix
    prefixes = COMMENT.get(suffix)
    line = lines[n - 1] or ""
    if suffix in DOCUMENT or not prefixes or not line.lstrip().startswith(prefixes):
        return present

    def is_comment(x: str | None) -> bool:
        return x is not None and x.lstrip().startswith(prefixes)

    lo = n - 1
    while lo > 0 and (is_comment(lines[lo - 1]) or lines[lo - 1] is None):
        lo -= 1
    hi = n - 1
    while hi + 1 < len(lines) and (is_comment(lines[hi + 1]) or lines[hi + 1] is None):
        hi += 1
    return [x for x in lines[lo : hi + 1] if x is not None]


def false_terms(lines: list[str | None]) -> list[int]:
    return [
        i + 1
        for i, line in enumerate(lines)
        if line is not None and SUBJECT.search(line) and NO_TERMS.search(line)
    ]


def reviewed_as(
    rel: str, text: str, reviewed: list[tuple[str, str, str]]
) -> tuple[str, str] | None:
    return next(((r, f) for r, f, _ in reviewed if r == rel and f in text), None)


def problems_of(
    files: dict[str, list[str | None]], reviewed: list[tuple[str, str, str]]
) -> list[str]:
    problems = []
    used = set()
    for rel, lines in files.items():
        for n, kind, _ in hits(lines):
            text = (lines[n - 1] or "").strip()
            known = reviewed_as(rel, text, reviewed)
            if known is not None:
                used.add(known)
                continue
            if not any(ACK in x for x in unit(rel, lines, n)):
                problems.append(
                    f"{rel}:{n}: a JB2008 number ({kind}) with no '{ACK}' in its "
                    f"{'document' if rel.endswith(tuple(DOCUMENT)) else 'unit'}: {text[:100]}"
                )
        for n in false_terms(lines):
            problems.append(
                f"{rel}:{n}: says SET's index files state no terms, against the register row {ROW}: "
                f"{(lines[n - 1] or '').strip()[:100]}"
            )
    for rel, text in sorted({(r, f) for r, f, _ in reviewed} - used):
        problems.append(f"{rel}: REVIEWED fragment matches no hit, re-read it: {text}")
    register = files.get(REGISTER)
    row = next((x for x in register or [] if x and x.startswith(f"| {ROW}")), None)
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
    files = public_files(ROOT)
    narrow: dict[str, int] = {}
    wide: dict[str, int] = {}
    for rel, lines in files.items():
        found = hits(lines)
        for n, kind, is_narrow in found:
            narrow[rel] = narrow.get(rel, 0) + is_narrow
            wide[rel] = wide.get(rel, 0) + 1
            if not check:
                text = (lines[n - 1] or "").strip()
                mark = "reviewed" if reviewed_as(rel, text, REVIEWED) else kind
                print(
                    f"{rel}:{n}: {'narrow' if is_narrow else 'wide  '} {mark:8} {text[:110]}"
                )
    print()
    for rel in sorted(wide):
        print(f"{rel}: {narrow[rel]} narrow, {wide[rel]} wide")
    print(f"total: {sum(narrow.values())} narrow, {sum(wide.values())} wide")
    if not check:
        return 0
    problems = problems_of(files, REVIEWED)
    for p in problems:
        print(f"refused: {p}")
    print(f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
