#!/usr/bin/env python3
"""Reconcile the mutation table in `docs/verification.md` against the runs it reports.

Every row's four numbers are copied by hand out of what `just mutants-verdict` printed, and until
this script existed nothing compared them with the run again. Ten of sixty-two rows disagreed on the
first pass, and the failure had a direction: eight carried the *unviable* count in the Timeouts
column. That is the one substitution that flatters the seal, because a timeout is scored as a
detection and an unviable mutant is not scored at all -- so the wrong column overstates what the
suite caught, in a place a reader has no way to check.

The run's own `mutants.out/{caught,missed,timeout,unviable}.txt` are the authority: one mutant per
line, written by cargo-mutants, never edited. `outcomes.json` carries a `Success` entry for the
unmutated baseline, which is why the line counts and not that file's length are what a row is
compared against.

The cache is a local artefact -- `~/.cache/spacelab-mutants/`, outside the repo, and the seal blocks
that fill it are run by hand. So this cannot run on a CI runner, and a version that "passed"
there would pass by finding nothing to check. It refuses to exit 0 without having verified a row,
for that reason.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "verification.md"
CACHE = pathlib.Path(
    os.environ.get("MUTANTS_CACHE", "~/.cache/spacelab-mutants")
).expanduser()

# A pipe inside a backtick span is never a separator, and the table's prose is full of `||` and
# `|x|`. The rows escape those as `\|`, so splitting on an unescaped pipe is what the rendered
# document does too -- a row this disagrees with is a row GFM would truncate.
SEPARATOR = re.compile(r"(?<!\\)\|")
# `| `block-name` | crate | caught | survivors | timeouts | prose |`, with the survivor count bold.
ROW = re.compile(r"^\| `([a-z0-9-]+)` \|")
# The document holds 220 backticked table rows and only this table's are block verdicts. Scoping to
# the heading rather than to a column count means another six-column table cannot start being read
# as mutation results by coincidence.
HEADING = re.compile(r"^#+ Verdict, as measured")

COLUMNS = {3: "caught", 4: "missed", 5: "timeout"}


def published() -> tuple[dict[str, dict[str, int]], list[str]]:
    """The counts each row claims, keyed by block name, and the rows this could not read.

    The unread rows are returned rather than skipped because a row nothing checks looks exactly like
    a row that passed. Two kinds occur, and only one of them is benign: a row covering several
    blocks at once has no single cache to compare against, while a row whose count column is an em
    dash is a number that was never written down.
    """
    rows: dict[str, dict[str, int]] = {}
    unread: list[str] = []
    inside = False
    for line in DOC.read_text().splitlines():
        if HEADING.match(line):
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = SEPARATOR.split(line.rstrip())
        if len(cells) != 8 or not cells[1].strip().startswith("`"):
            continue
        name = ROW.match(line)
        if not name:
            unread.append(cells[1].strip())
            continue
        try:
            # The survivor cell is bold (`**4**`), the other two are bare, so strip to the digits.
            rows[name.group(1)] = {
                key: int(re.sub(r"\D", "", cells[index]))
                for index, key in COLUMNS.items()
            }
        except ValueError:
            unread.append(cells[1].strip())
    return rows, unread


def measured(block: str) -> dict[str, int] | None:
    """What the run recorded, or None when its cache is gone."""
    out = CACHE / block / "mutants.out"
    if not out.is_dir():
        return None
    counts = {}
    for key in (*COLUMNS.values(), "unviable"):
        path = out / f"{key}.txt"
        if not path.is_file():
            return None
        counts[key] = sum(1 for line in path.read_text().splitlines() if line.strip())
    return counts


def main(wanted: list[str]) -> int:
    rows, unread = published()
    for row in unread:
        print(f"  not read: {row}")
    for block in wanted:
        if block not in rows:
            # Not fatal: `just mutants` calls this before anyone has written the row.
            print(f"  no table row yet for {block}")

    checked = bad = uncached = 0
    for block, claim in sorted(rows.items()):
        if wanted and block not in wanted:
            continue
        run = measured(block)
        if run is None:
            uncached += 1
            continue
        checked += 1
        wrong = {
            key: (claim[key], run[key])
            for key in COLUMNS.values()
            if claim[key] != run[key]
        }
        if wrong:
            bad += 1
            detail = ", ".join(
                f"{k} says {c}, run has {m}" for k, (c, m) in wrong.items()
            )
            print(f"  {block}: {detail} (unviable {run['unviable']})")

    print(
        f"{len(rows) + len(unread)} rows, {checked} checked, {bad} disagree, "
        f"{uncached} no longer cached, {len(unread)} not read"
    )
    if not checked:
        print(
            "nothing was checked -- this reconciliation needs the run cache, not just the repo"
        )
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
