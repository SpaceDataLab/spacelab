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

Standard library only, so it runs the same under `uv run python` locally and under a bare `python3`
on a CI runner that has neither `uv` nor `just`.
"""

from __future__ import annotations

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


def main() -> int:
    every = mutant_list("--no-config")
    kept = set(mutant_list())
    entries = [raw.replace("\\\\", "\\") for raw in ENTRY.findall(LEDGER.read_text())]

    claims: dict[str, list[str]] = {}
    bad = 0
    for pattern in entries:
        if METACHARACTER.search(pattern):
            print(f"  unescaped metacharacter: {pattern}")
            bad += 1
            continue
        hits = [line for line in every if re.search(pattern, line)]
        if not hits:
            print(f"  stale anchor, matches no mutant: {pattern}")
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

    excluded = len(every) - len(kept)
    print(
        f"{len(entries)} entries, {excluded} mutants excluded, {len(claims)} claimed, {bad} broken"
    )
    return 1 if bad or len(entries) != excluded or len(claims) != excluded else 0


if __name__ == "__main__":
    sys.exit(main())
