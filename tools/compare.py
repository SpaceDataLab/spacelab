#!/usr/bin/env python3
"""The comparator suite (`plans/17` Ф19.C): every parked comparator test, run, its distances written down.

Another code's output is a comparator (`plans/17` P0, `docs/verification.md` §12.1): it is recorded and
published as a distance, and it never decides a test. A comparator test is parked with
`#[ignore = "comparator: ..."]`, so `cargo test` skips it and nothing in CI can fail on its distance. This
script finds every parked test in the committed tree, runs each one alone
(`cargo test -- --ignored --exact`), and reads the rows it prints, one line per quantity:

    comparison | <code> | <version> | <quantity> | <unit> | <n> | <max> | <RMS>

`max` and `RMS` are the largest and the root-mean-square distance over the `n` samples. It writes them all to
`docs/comparisons.md` with the date, and lists the parked tests that print no row, so that no comparator is
missing from the table without a word. A test that fails stops the script: its distance is not judged, but
a comparator that cannot run has no distance to record. A row whose test lies inside a licence-restricted

    uv run --no-project python -I tools/compare.py
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "comparisons.md"
PARKED = re.compile(r'#\[ignore\s*=\s*"comparator:')
FN = re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)")
PREFIX = "comparison | "
# A parked attribute sits on the test's own item: its `fn` follows within the attribute block.
ATTRIBUTE_SPAN = 6
TIMEOUT_SECONDS = 1800


@dataclasses.dataclass(frozen=True, slots=True)
class Parked:
    path: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"


@dataclasses.dataclass(frozen=True, slots=True)
class Row:
    code: str
    version: str
    quantity: str
    unit: str
    n: int
    max: float
    rms: float


def parse_row(line: str) -> Row:
    """One `comparison | ...` line; anything else in its place is a malformed row, not a skipped one."""
    fields = line.removeprefix(PREFIX).split(" | ")
    if not line.startswith(PREFIX) or len(fields) != 7:
        raise ValueError(f"not a comparison row of 8 fields: {line!r}")
    # A field is a Markdown table cell: a `|` inside one splits the cell where the table is read.
    if any("|" in f for f in fields):
        raise ValueError(f"a comparison field holds '|': {line!r}")
    code, version, quantity, unit, n, worst, rms = (f.strip() for f in fields)
    return Row(code, version, quantity, unit, int(n), float(worst), float(rms))


def parked_tests(root: pathlib.Path, paths: list[str]) -> list[Parked]:
    """Every `#[ignore = "comparator: ..."]` test in `paths`, in file order."""
    found = []
    for path in paths:
        lines = (root / path).read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not PARKED.search(line):
                continue
            names = [
                m.group(1)
                for m in map(FN.match, lines[i + 1 : i + 1 + ATTRIBUTE_SPAN])
                if m
            ]
            if not names:
                raise ValueError(
                    f"{path}:{i + 1}: a comparator attribute with no fn after it"
                )
            found.append(Parked(path, names[0]))
    return found


def target(path: str) -> tuple[str, list[str], str]:
    """The crate directory, the `cargo test` target arguments and the module path of a test file."""
    parts = pathlib.PurePosixPath(path).parts
    if len(parts) < 4 or parts[0] != "crates" or parts[2] not in ("src", "tests"):
        raise ValueError(
            f"{path}: not under crates/<crate>/src or crates/<crate>/tests"
        )
    crate, kind, rest = parts[1], parts[2], list(parts[3:])
    rest[-1] = rest[-1].removesuffix(".rs")
    if kind == "tests":
        if len(rest) != 1:
            raise ValueError(f"{path}: a module of an integration test, not its root")
        return crate, ["--test", rest[0]], ""
    if rest[-1] == "mod" or rest == ["lib"]:
        rest.pop()
    return crate, ["--lib"], "::".join(rest)


def package(crate: str) -> str:
    with (ROOT / "crates" / crate / "Cargo.toml").open("rb") as f:
        return tomllib.load(f)["package"]["name"]


def cargo(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", "test", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        env={**os.environ, "CARGO_TERM_COLOR": "never"},
        check=False,
    )


INLINE_MOD = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*\{", re.M)


def full_name(test: Parked, listed: list[str], module: str) -> str:
    """The one listed test path that is `test`: its name, inside its file's module.

    Between the file's module and the name only the file's own inline modules (`mod tests { ... }`)
    may stand: `dsst::drag::tests::x` is not in `dsst/mod.rs`, though it starts with `dsst::`.
    """
    inline = set(INLINE_MOD.findall((ROOT / test.path).read_text()))
    prefix = module + "::" if module else ""
    hits = [
        name
        for name in listed
        if name.startswith(prefix)
        and name.rsplit("::", 1)[-1] == test.name
        and all(seg in inline for seg in name[len(prefix) :].split("::")[:-1])
    ]
    if len(hits) != 1:
        raise ValueError(f"{test.key}: {len(hits)} ignored tests match, {hits}")
    return hits[0]


def fail(what: str, done: subprocess.CompletedProcess[str]) -> None:
    tail = "\n".join((done.stdout + done.stderr).splitlines()[-40:])
    sys.exit(f"{what} failed (exit {done.returncode}):\n{tail}")


def run(tests: list[Parked]) -> list[tuple[Parked, list[Row]]]:
    listed: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    results = []
    for test in tests:
        crate, args, module = target(test.path)
        selector = ("-p", package(crate), *args)
        if selector not in listed:
            done = cargo([*selector, "--", "--ignored", "--list"])
            if done.returncode != 0:
                fail(f"listing {' '.join(selector)}", done)
            listed[selector] = [
                line.removesuffix(": test")
                for line in done.stdout.splitlines()
                if line.endswith(": test")
            ]
        name = full_name(test, listed[selector], module)
        done = cargo([*selector, "--", "--ignored", "--exact", "--nocapture", name])
        if done.returncode != 0:
            fail(test.key, done)
        rows = [
            parse_row(line)
            for line in done.stdout.splitlines()
            if line.startswith(PREFIX)
        ]
        print(f"{test.key}: {len(rows)} rows", file=sys.stderr)
        results.append((test, rows))
    return results


def restricted_roots() -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "check_provenance", ROOT / "tools" / "check-provenance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restricted()[0]


def table(
    results: list[tuple[Parked, list[Row]]], day: datetime.date, restricted: list[str]
) -> str:
    def private(test: Parked) -> str:
        return (
            if any(test.path.startswith(r) for r in restricted)
            else ""
        )

    out = [
        "# Comparisons",
        "",
        "Distances between this library and other codes, and between other codes and the mathematics,",
        "measured by the parked comparator tests. Another code's output is a comparator: it is recorded",
        "here as a distance and decides no test (`docs/verification.md` §12.1). The mathematics decides,",
        "through each module's gate. Regenerated by one command, which runs every test parked with",
        '`#[ignore = "comparator: ..."]` and reads the rows it prints:',
        "",
        "    uv run --no-project python -I tools/compare.py",
        "",
        f"Generated {day.isoformat()}. `n` is the number of samples; `max` and `RMS` are the largest and the",
        "root-mean-square distance over them.",
        "",
        "| Code | Version | Quantity | Unit | n | max | RMS | Test |",
        "|---|---|---|---|--:|--:|--:|---|",
    ]
    out += [
        f"| {r.code} | {r.version} | {r.quantity} | {r.unit} | {r.n} | {r.max:.3e} | {r.rms:.3e} | "
        f"`{test.key}` |{private(test)}"
        for test, rows in results
        for r in rows
    ]
    silent = [test for test, rows in results if not rows]
    if silent:
        out += ["", "Parked comparator tests that print no row yet:", ""]
        out += [f"- `{test.key}`{private(test)}" for test in silent]
    return "\n".join(out) + "\n"


def committed_rust() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "crates/*.rs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args()
    results = run(parked_tests(ROOT, committed_rust()))
    OUT.write_text(
        table(results, datetime.datetime.now(datetime.UTC).date(), restricted_roots())
    )
    rows = sum(len(r) for _, r in results)
    print(f"{len(results)} parked tests, {rows} rows -> {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
