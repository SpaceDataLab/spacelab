#!/usr/bin/env python3
"""Reconcile `.cargo/mutants.toml` against the mutants cargo-mutants actually generates.

An exclusion in that file is a *regex* carrying a `file:line:col` anchor, so it fails in two silent
directions. It goes stale whenever a line is inserted above the code it names -- a justified mutant
then reappears as a survivor, and the seal that finds it reads as a regression. And an unescaped
metacharacter changes what it matches: `replace + with -` quantifies the space and matches nothing,
while `replace && with ||` is an alternation with an empty branch, which matches *every* mutant and
switches the file off entirely. Both have happened.

Two `cargo mutants --list` calls answer it, with the config and without: every entry must claim
exactly one mutant, and every excluded mutant must be claimed by exactly one entry. Both halves are
needed, because per-file counting lets an entry matching *two* mutants cancel against one matching
none.

`--rewrite` moves a stale anchor to where its code went, and only when its proof provably went with
it: the function cargo-mutants names is byte-identical to the one in the commit that last wrote the
entry, and the anchor sits at the same place inside it. A function that changed is reported, never
followed -- a proof does not survive the code it was written against.

`--since REV` catches the drift no count can see: an anchor that moved onto *another* mutant of the
same shape still claims exactly one, so it reconciles, while the proof above it now excuses code it
was never about. Every entry that stood in REV's ledger is compared with what it named there; one that
now names code no entry excluded at REV is reported. `--since HEAD` before each commit keeps it so.

Standard library only, so it runs the same under `uv run python` locally and under a bare `python3`
on a CI runner that has neither `uv` nor `just`.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER = ROOT / ".cargo" / "mutants.toml"

# The characters that mean something to the regex engine and nothing to a `file:line:col: replace X
# with Y in f` anchor. An entry carrying one unescaped is a bug whatever it happens to match today.
METACHARACTER = re.compile(r"(?<!\\)[|+*?(){}\[\]^$]")
# One TOML basic string on a line of its own, which is how every entry in the ledger is written.
ENTRY = re.compile(r'^\s*"((?:[^"\\]|\\.)*)"\s*,\s*$', re.M)
ANCHOR = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?P<tail>.+)$")


def mutant_list(*args: str) -> list[str]:
    """`cargo mutants --list`, as an argv list rather than a shell string.

    `CARGO_TERM_COLOR` is forced off rather than inherited. Cargo emits SGR sequences on `always`
    even into a pipe, and this parser matches anchors against the whole line -- so under the CI job's
    own `env: CARGO_TERM_COLOR: always` every one of the entries matched nothing and the run reported
    every anchor stale, which is the opposite of what the ledger says when it is genuinely broken.
    """
    done = subprocess.run(
        ["cargo", "mutants", "--list", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
    )
    if done.returncode != 0:
        sys.stderr.write(done.stderr)
        raise SystemExit(
            f"cargo mutants --list {' '.join(args)} exited {done.returncode}"
        )
    return done.stdout.splitlines()


def git(*args: str) -> str | None:
    done = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return done.stdout if done.returncode == 0 else None


@functools.cache
def function_spans(path: str) -> dict[str, tuple[int, int]]:
    """Each unconfigured mutant of `path` -> the 1-based line span of the function it mutates."""
    listed = json.loads("\n".join(mutant_list("--no-config", "--json", "-f", path)))
    return {
        m["name"]: (
            m["function"]["span"]["start"]["line"],
            m["function"]["span"]["end"]["line"],
        )
        for m in listed
        if m.get("function")
    }


@functools.cache
def lines_at(rev: str | None, path: str) -> list[str] | None:
    """`path` at `rev`, or in the working tree when `rev` is None."""
    if rev is None:
        return (ROOT / path).read_text().splitlines()
    shown = git("show", f"{rev}:{path}")
    return None if shown is None else shown.splitlines()


def named(lines: list[str] | None, pattern: str) -> tuple[str, int, str] | None:
    """The code an anchor names in `lines`: the line's text, the column past its indent, the tail."""
    anchor = ANCHOR.match(pattern)
    if anchor is None or lines is None or int(anchor["line"]) > len(lines):
        return None
    text = lines[int(anchor["line"]) - 1]
    indent = len(text) - len(text.lstrip())
    return text.strip(), int(anchor["col"]) - indent, anchor["tail"]


def transfers(rev: str, entries: list[str], every: list[str]) -> int:
    """Entries kept since `rev` whose anchor now names code that no entry excluded there."""
    shown = git("show", f"{rev}:{LEDGER.relative_to(ROOT)}")
    if shown is None:
        raise SystemExit(f"no ledger at {rev}")
    then = [raw.replace("\\\\", "\\") for raw in ENTRY.findall(shown)]
    kept_entries = set(then)
    excluded_then: dict[str, set[tuple[str, int, str] | None]] = {}
    moved = 0
    for pattern in entries:
        hits = [line for line in every if re.search(pattern, line)]
        if pattern not in kept_entries or len(hits) != 1:
            continue
        path = hits[0].split(":", 1)[0]
        now, before = (
            named(lines_at(None, path), pattern),
            named(lines_at(rev, path), pattern),
        )
        if before is None or now == before:
            continue
        if path not in excluded_then:
            excluded_then[path] = {
                named(lines_at(rev, path), entry)
                for entry in then
                if (anchor := ANCHOR.match(entry))
                and re.search(anchor["file"] + "$", path)
            }
        if now not in excluded_then[path]:
            print(f"  moved onto other code since {rev}: {pattern}")
            print(f"      then: {before[0]}")
            print(f"      now:  {now[0] if now else None}")
            moved += 1
    return moved


def relocate(
    pattern: str, every: list[str], claimed: set[str]
) -> tuple[str | None, str]:
    """Where a stale anchor's mutant went, if its function is byte-identical; else why not."""
    anchor = ANCHOR.match(pattern)
    if anchor is None:
        return None, "not a file:line:col anchor"
    line, col = int(anchor["line"]), int(anchor["col"])
    moved = re.compile(anchor["file"] + r":(\d+):(\d+): " + anchor["tail"])
    raw = pattern.replace("\\", "\\\\")
    log = git(
        "log", "-1", "--format=%H", "-S", raw, "--", str(LEDGER.relative_to(ROOT))
    )
    if not log:
        return None, "no commit wrote this entry"
    commit = log.strip()
    found: list[str] = []
    for name in every:
        hit = moved.search(name)
        if hit is None or name in claimed or int(hit[2]) != col:
            continue
        path = name.split(":", 1)[0]
        span = function_spans(path).get(name)
        old = git("show", f"{commit}:{path}")
        if span is None or old is None:
            continue
        body = (ROOT / path).read_text().splitlines()[span[0] - 1 : span[1]]
        start = line - (int(hit[1]) - span[0]) - 1
        if start >= 0 and old.splitlines()[start : start + len(body)] == body:
            found.append(f"{anchor['file']}:{hit[1]}:{col}: {anchor['tail']}")
    if len(found) == 1:
        return found[0], f"function unchanged since {commit[:10]}"
    if not found:
        return None, f"its function changed since {commit[:10]}: re-read the proof"
    return None, f"{len(found)} identical functions could hold it"


def main(move_stale: bool, since: str | None) -> int:
    every = mutant_list("--no-config")
    kept = set(mutant_list())
    entries = [raw.replace("\\\\", "\\") for raw in ENTRY.findall(LEDGER.read_text())]

    claims: dict[str, list[str]] = {}
    stale: list[str] = []
    bad = 0
    for pattern in entries:
        if METACHARACTER.search(pattern):
            print(f"  unescaped metacharacter: {pattern}")
            bad += 1
            continue
        hits = [line for line in every if re.search(pattern, line)]
        if not hits:
            print(f"  stale anchor, matches no mutant: {pattern}")
            stale.append(pattern)
            bad += 1
        elif len(hits) > 1:
            print(f"  ambiguous anchor, matches {len(hits)}: {pattern}")
            bad += 1
        elif hits[0] in kept:
            print(f"  entry did not exclude its mutant: {pattern}")
            bad += 1
        for hit in hits:
            claims.setdefault(hit, []).append(pattern)

    for hit, patterns in claims.items():
        if len(patterns) > 1:
            print(f"  {len(patterns)} entries claim one mutant: {hit}")
            bad += 1

    if since is not None:
        bad += transfers(since, entries, every)

    excluded = len(every) - len(kept)
    print(
        f"{len(entries)} entries, {excluded} mutants excluded, {len(claims)} claimed, {bad} broken"
    )
    if stale and move_stale:
        return rewrite(stale, every, set(claims))
    return 1 if bad or len(entries) != excluded or len(claims) != excluded else 0


def rewrite(stale: list[str], every: list[str], claimed: set[str]) -> int:
    """Move each stale anchor `relocate` can place, then reconcile again from scratch."""
    text = LEDGER.read_text()
    moved = 0
    for pattern in stale:
        new, why = relocate(pattern, every, claimed)
        if new is None:
            print(f"  left for a hand: {pattern} -- {why}")
            continue
        old_entry = '"' + pattern.replace("\\", "\\\\") + '",'
        assert text.count(old_entry) == 1, old_entry
        text = text.replace(old_entry, '"' + new.replace("\\", "\\\\") + '",')
        claimed.add(next(name for name in every if re.search(new, name)))
        print(f"  moved: {pattern} -> {new.split(': ', 1)[0]} ({why})")
        moved += 1
    LEDGER.write_text(text)
    print(f"{moved} of {len(stale)} stale anchors moved; reconciling again")
    return main(move_stale=False, since=None)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    cli.add_argument(
        "--rewrite",
        action="store_true",
        help="move stale anchors whose function is unchanged",
    )
    cli.add_argument(
        "--since",
        metavar="REV",
        help="report anchors that moved onto other code since REV",
    )
    args = cli.parse_args()
    sys.exit(main(move_stale=args.rewrite, since=args.since))
