"""Classify the survivors of a `cargo mutants` run by what they actually change.

A survivor is a mutation the test suite did not notice. Deciding what it means -- a test
gap, a provable equivalence, or a defect -- starts with the same mechanical step every
time: apply the mutation, run the tests, read the numbers they print, put the source back.
Doing that by hand is where most of the triage time goes, and a hand-rolled probe per file
is a fresh chance to point it at the wrong directory and read every mutant as caught.

`cargo mutants` already writes the patch for every mutant under `mutants.out/diff/`, so
this drives those patches directly. The one wrinkle is that its `+++` line carries the
mutation description rather than a path, which no patch tool will accept; the path is on
the `---` line and is copied over before applying.

Classification, given that the tests pass on the unmutated tree:

    IDENTICAL  every diagnostic the tests print is byte-for-byte what it was, so the
               mutation moved no number the suite can see -- an equivalence candidate,
               still owed a written reason before it goes in `.cargo/mutants.toml`
    MOVED      the tests still pass but a printed diagnostic changed; the delta is the
               size of the hole, and it is what a new assertion has to be tighter than
    CAUGHT     the tests fail on the mutation. Against a fresh run that listed the mutant as
               missed, the run and the tree have diverged and the run is stale; against a test
               just written to close the gap, this is the proof that the test is not a decoration
    TIMEOUT    the tests never finished. The suite neither passed nor failed, so this is not a
               verdict: the mutant survives until the loop it landed in is bounded, or until a
               longer budget turns it into one of the three above

Diagnostics are collected with `--test-threads=1`, since spacelab's tests print their
measured residuals and parallel runs interleave those lines nondeterministically.

    uv run python tools/mutant_triage.py \
        --out ~/.cache/spacelab-mutants/dsst-rest-1/mutants.out \
        --package spacelab-propagation --only dsst/gaussian.rs

Add `--list timeout.txt` to sweep the mutants a run only ever detected by wall clock, which is
not a verdict: what finishes under a longer budget is an ordinary survivor and is triaged as one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Exit code for a test run the budget cut short. Cargo reports a failing suite as 101 and never
# uses this, so it cannot collide with a real verdict.
TIMED_OUT = 124

# `path:line:col: description`, the format of `missed.txt`.
MISSED = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+):(?P<col>\d+): (?P<desc>.+)$")

# A line worth comparing carries a measured number; bare progress chatter does not.
NUMERIC = re.compile(r"\d[eE.\d]*[-+]?\d")

# Cargo's own output and libtest's bookkeeping, none of which reflects the mutation.
#
# The doctest wall-clock summary belongs here for a reason worth stating: it is the only line cargo
# prints that carries a *measurement of its own runtime*, so it differs between any two runs. Left
# in, it made every survivor of a whfast replay read as MOVED with the delta
# `'all doctests ran in 0.15s' -> '0.14s'` -- a survivor dressed as a diagnostic change. The filter
# is an enumeration of known noise, so any new noise source becomes signal by default; timings are
# the class to watch, since they are the only cargo output guaranteed to vary run to run.
CHATTER = re.compile(
    r"^\s*(Compiling|Finished|Running|Fresh|warning|error|Blocking)\b"
    r"|^running \d+ tests?$"
    r"|^test result:"
    r"|^\s*Doc-tests\b"
    r"|^all doctests ran in "
)

# libtest prefixes a test's own `println!` with `test <name> ... `, and those printed
# residuals are precisely the numbers this exists to compare -- strip the prefix, keep
# the diagnostic, and drop the bare `... ok` lines that carry nothing.
TEST_PREFIX = re.compile(r"^test [\w:]+ \.\.\. ")


@dataclass(frozen=True)
class Mutant:
    path: str
    line: int
    col: int
    desc: str
    diff: Path

    @property
    def anchor(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.desc}"


def load_mutants(out: Path, only: str | None, lists: list[str]) -> list[Mutant]:
    """Pair every entry of the named survivor lists with the diff that produces it."""
    missed = [ln for name in lists for ln in (out / name).read_text().splitlines()]
    by_desc: dict[tuple[str, int, int, str], Path] = {}
    for diff in (out / "diff").glob("*.diff"):
        head = diff.read_text().splitlines()[:2]
        if (
            len(head) < 2
            or not head[0].startswith("--- ")
            or not head[1].startswith("+++ ")
        ):
            continue
        # The filename encodes the site; the header's second line is the description.
        stem = re.match(
            r"^(?P<p>.+)_line_(?P<l>\d+)_col_(?P<c>\d+)(?:_\d+)?$", diff.stem
        )
        if stem is None:
            continue
        by_desc[(head[0][4:], int(stem["l"]), int(stem["c"]), head[1][4:])] = diff

    mutants = []
    for raw in missed:
        m = MISSED.match(raw.strip())
        if m is None:
            continue
        if only is not None and only not in m["path"]:
            continue
        key = (m["path"], int(m["line"]), int(m["col"]), m["desc"])
        diff = by_desc.get(key)
        if diff is None:
            print(f"no diff for {raw}", file=sys.stderr)
            continue
        mutants.append(
            Mutant(m["path"], int(m["line"]), int(m["col"]), m["desc"], diff)
        )
    return mutants


@contextmanager
def held(targets: list[Path], lock_dir: Path) -> Iterator[None]:
    """Declare, for the whole run, that this process is putting mutations in the tree.

    A concurrent `cargo mutants` copies the working tree into $TMPDIR once per invocation, so a
    batch that starts while a mutation is applied bakes it into its own baseline and every result
    in that batch is meaningless. Neither `git diff` nor cargo-mutants' `changed by cargo-mutants`
    marker can be the interlock: the campaign carries thousands of uncommitted test lines under
    `crates/*/src`, and the patches applied here do not carry the marker.

    So the lock is a declaration, and the pristine bytes go to disk beside it. `finally` and the
    handlers below cover every signal that can be caught; SIGKILL -- which is what the OOM-killer
    sends -- cannot be, and then the lock and the backups outlive the process on purpose. A stale
    lock blocks the queue and names what to restore, which is the failure this is built for.
    """
    lock = lock_dir / "replay.lock"
    if lock.exists():
        raise SystemExit(f"another replayer holds the tree: {lock.read_text()}")
    backups = {t: lock_dir / f"pristine-{t.name}" for t in targets}
    for target, backup in backups.items():
        backup.write_bytes(target.read_bytes())
    lock.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "restore": {str(t): str(b) for t, b in backups.items()},
            },
            indent=2,
        )
    )

    def bail(signum: int, _frame: object) -> None:
        raise SystemExit(f"replayer received signal {signum}")

    previous = [
        (sig, signal.signal(sig, bail))
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    ]
    try:
        yield
    finally:
        for target, backup in backups.items():
            target.write_bytes(backup.read_bytes())
            backup.unlink()
            # `patch` drops a `.rej` beside the file whenever a hunk misses, and a stray `.rej`
            # under `crates/*/src` ends up inside the tree `cargo mutants` snapshots for its next
            # batch. Nothing else in the repo produces one, so removing it here is unambiguous.
            target.with_suffix(target.suffix + ".rej").unlink(missing_ok=True)
        lock.unlink()
        for sig, handler in previous:
            signal.signal(sig, handler)


def patched(diff: Path) -> str:
    """The diff with its `+++` line pointed at the file the `---` line names."""
    lines = diff.read_text().splitlines(keepends=True)
    lines[1] = "+++ " + lines[0][4:]
    return "".join(lines)


def apply(diff_text: str, *, revert: bool) -> bool:
    argv = ["patch", "-p0", "--silent", "--no-backup-if-mismatch"]
    if revert:
        argv.append("-R")
    done = subprocess.run(
        argv, cwd=ROOT, input=diff_text, text=True, capture_output=True
    )
    if done.returncode != 0:
        print(done.stdout + done.stderr, file=sys.stderr)
    return done.returncode == 0


def run_tests(package: str, filt: str | None, timeout: int) -> tuple[int, list[str]]:
    """Return the exit code and the diagnostic lines the tests printed.

    Every target of the package, not just `--lib`: several modules here are tested exclusively
    from `tests/` -- `dopri5.rs` from `tests/integrators.rs`, `ark.rs` from `tests/ark.rs` -- and
    a lib-only run reports every one of their mutants as surviving a suite that in fact kills it.
    """
    argv = ["cargo", "test", "-p", package]
    if filt is not None:
        argv.append(filt)
    argv += ["--", "--nocapture", "--test-threads=1"]
    # Its own session, so the budget can reap the test binary and not only the cargo that
    # spawned it: `subprocess.run`'s timeout kills the direct child, and the orphaned binary
    # left behind holds a core at 100 % for as long as the mutation keeps it spinning.
    with subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return TIMED_OUT, []
        code = proc.returncode
    kept = []
    for raw in out.splitlines():
        if CHATTER.search(raw):
            continue
        ln = TEST_PREFIX.sub("", raw).rstrip()
        if ln in ("ok", "FAILED", "ignored") or not NUMERIC.search(ln):
            continue
        kept.append(ln)
    return code, kept


def classify(base: list[str], got: list[str], code: int) -> tuple[str, str]:
    if code == TIMED_OUT:
        # A suite that never finished neither passed nor failed, and reading it as a failure
        # would report every mutation that hangs the tests as one the tests kill.
        return "TIMEOUT", "the suite did not finish inside the budget"
    if code != 0:
        # Whether this is good news depends on when it is run. Verifying a test written to close
        # the gap, it is the whole point; triaging a survivor of a fresh run, it means the run and
        # the working tree have diverged and the run is stale. State the fact, not the reading.
        return "CAUGHT", "the suite fails on this mutation"
    if got == base:
        return "IDENTICAL", ""
    moved = [f"{b!r} -> {g!r}" for b, g in zip(base, got, strict=False) if b != g]
    if len(base) != len(got):
        moved.append(f"line count {len(base)} -> {len(got)}")
    return "MOVED", " | ".join(moved[:4])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, required=True, help="a run's mutants.out directory"
    )
    ap.add_argument("--package", required=True, help="cargo package whose tests to run")
    ap.add_argument(
        "--filter",
        default=None,
        help="test-name filter passed to cargo test. Do not set it to the module name to save "
        "time: a test need not be named after what it covers, and every one it excludes turns "
        "a caught mutant into a reported survivor. `--filter whfast` hid the two tests that "
        "cover `stumpff` and reported all 99 of that file's mutants as surviving",
    )
    ap.add_argument(
        "--only", default=None, help="restrict to survivors whose path contains this"
    )
    ap.add_argument("--timeout", type=int, default=900, help="seconds per test run")
    ap.add_argument("--report", type=Path, default=None, help="write a TSV here")
    ap.add_argument(
        "--list",
        dest="lists",
        action="append",
        default=None,
        help="survivor list to read, repeatable (default: missed.txt)",
    )
    ap.add_argument(
        "--lock-dir",
        type=Path,
        default=Path.home() / ".cache/spacelab-mutants",
        help="where the replay lock and the pristine backups live",
    )
    args = ap.parse_args()

    mutants = load_mutants(args.out, args.only, args.lists or ["missed.txt"])
    if not mutants:
        print("no survivors matched", file=sys.stderr)
        return 1

    code, base = run_tests(args.package, args.filter, args.timeout)
    if code != 0:
        print(f"baseline is not green (rc={code}) -- fix that first", file=sys.stderr)
        return 1
    print(f"baseline: {len(base)} diagnostic lines, {len(mutants)} survivors\n")

    args.lock_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted({ROOT / m.path for m in mutants})
    rows = []
    with held(targets, args.lock_dir):
        for i, mut in enumerate(mutants, 1):
            text = patched(mut.diff)
            # Restore from a byte copy rather than with `patch -R`: an editor's format-on-save can
            # rewrap the mutated line while the run is in flight, after which the reverse patch no
            # longer matches its context and the mutation silently stays in the tree.
            target = ROOT / mut.path
            pristine = target.read_bytes()
            if not apply(text, revert=False):
                rows.append(("SKIPPED", mut.anchor, "patch did not apply"))
                continue
            try:
                rc, got = run_tests(args.package, args.filter, args.timeout)
                verdict, detail = classify(base, got, rc)
            finally:
                target.write_bytes(pristine)
            rows.append((verdict, mut.anchor, detail))
            print(f"[{i}/{len(mutants)}] {verdict:9s} {mut.anchor}")
            if detail:
                print(f"            {detail}")

    tally: dict[str, int] = {}
    for verdict, _, _ in rows:
        tally[verdict] = tally.get(verdict, 0) + 1
    print("\n" + "  ".join(f"{k} {v}" for k, v in sorted(tally.items())))
    # The verdict is binary and the three labels are not: IDENTICAL and MOVED are both survivors,
    # differing only in whether the hole is measurable yet. Printed apart they read as three
    # outcomes, and a large MOVED count reads as progress. Say the count that matters.
    survived = len(rows) - tally.get("CAUGHT", 0)
    print(f"caught {tally.get('CAUGHT', 0)}/{len(rows)}, {survived} still surviving")

    if args.report is not None:
        args.report.write_text("".join(f"{v}\t{a}\t{d}\n" for v, a, d in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
