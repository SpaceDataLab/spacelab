#!/usr/bin/env python3
"""Every module that says it is a port must appear in `docs/provenance.md`.

`cargo deny` reads the licence metadata of *dependencies*. Every provenance question in this
workspace is about code we wrote — a translation of somebody else's implementation carries their
licence and no metadata says so — which makes the whole class invisible to the gate that exists to
catch licence problems. This script is the gate for that class.

It is deliberately crude. It does not judge whether a module is a derivative work; it finds every
module whose own documentation *claims* a port, a translation or a transcription, and requires that
the claim has been looked at and recorded. A module may end up in the register as "clean" or even as
"not a port at all" — several are — but it may not be absent.

Two failure modes it was written against, both of which had already happened:

  * A register assembled by reading the modules that came to mind named 15 Orekit-derived files where
    there were 28. Apache-2.0's attribution condition is per file.
  * The first mechanical pass matched `transcribed from` and so missed `transcribed verbatim from`,
    `transcribed byte-for-byte` and `transcribing 342 rows` — nine files, a sixth of the set. A
    pattern that reads as exhaustive and is not is worse than no pattern, because it is trusted.

The covered set is a fenced block at the end of the register rather than the prose table, because the
table groups files by upstream and a machine cannot read `{mod,tables}.rs` or `dsst/*.rs`.

    uv run python tools/check-provenance.py            # list the sweep
    uv run python tools/check-provenance.py --check    # fail on anything uncovered

The second half of the register is *data*: every host a `tools/fetch-*.sh` script reads, every
committed fixture and asset, with the terms as read and a verdict. A licence that binds a table
binds the crate that embeds it just as a licence on code does, and `cargo deny` sees neither.

    uv run python tools/check-provenance.py --data --check     # every host and file has a row
    uv run python tools/check-provenance.py --data --check --public   # ...and no row is unread
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REGISTER = ROOT / "docs" / "provenance.md"
MARKER = "## The covered set"
RELEASE = "## Not for public release"
DATA = "## The data register"
VERDICTS = {
    "ship",
    "ship+attr",
    "internal-only",
    "restricted",
    "user-decision",
    "unread",
}
# Everything committed that is not code: test fixtures and the assets `include_str!` embeds.
DATA_GLOBS = ("crates/*/tests/fixtures/**/*", "crates/*/assets/**/*")
HOST = re.compile(r"https?://([A-Za-z0-9.-]+)")

# A module may be licence-restricted without being removable, so the second half of this script
# tracks *coupling* rather than presence: the set of files that name an excluded module has to be
# the set the register records. A file already listed may gain or lose mentions freely; a file that
# starts naming one has to be added deliberately, which is the moment to ask whether the public tree
# can still be cut. Counting files rather than mentions is what makes the gate quiet enough to keep.
# Plus the root-level markdown -- README.md and CHANGELOG.md both name models.
SCAN = ("crates", "derivations", "tools", "docs", ".github", ".cargo")
# The C ABI headers, their C/C++ tests and the Java oracles count too: a bridge that exports a
# restricted model is coupled to it like any other file.
SCAN_SUFFIXES = {
    ".rs",
    ".toml",
    ".py",
    ".sh",
    ".md",
    ".yml",
    ".yaml",
    ".tex",
    ".in",
    ".csv",
}
SCAN_SUFFIXES |= {".c", ".h", ".cpp", ".hpp", ".java"}

# A claim of derivation, in any of the shapes the module docs actually use. `transcrib\w+` rather
# than `transcribed from` is the whole point: the adverb goes in the middle.
CLAIM = re.compile(
    r"\b(port of|ported from|a faithful port|direct port|translation of|translated from|transcrib\w+)\b",
    re.I,
)


def module_doc(path: pathlib.Path) -> str:
    """The `//!` block at the top of the file, as one string.

    Stops at the first item, so a `//!` comment further down (there are none, but the rule should
    not depend on that) cannot widen the sweep, and an ordinary `//` comment does not end it.
    """
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip()
        if stripped.startswith("//!"):
            lines.append(stripped[3:].strip())
        elif stripped.startswith("//") or not stripped:
            continue
        else:
            break
    return " ".join(lines)


def sweep() -> list[str]:
    hits = []
    for path in sorted(ROOT.glob("crates/*/src/**/*.rs")):
        if CLAIM.search(module_doc(path)):
            hits.append(str(path.relative_to(ROOT)))
    return hits


def covered() -> set[str]:
    text = REGISTER.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"{REGISTER.name} has no '{MARKER}' section")
    block = text.split(MARKER, 1)[1]
    body = block.split("```")[1] if "```" in block else ""
    # The opening fence carries a language tag (```text), which is not a path.
    return {line.strip() for line in body.splitlines() if line.strip().endswith(".rs")}


def restricted() -> tuple[list[str], list[str], set[str]]:
    """The excluded roots, the extra names they go by, and the files recorded as naming them.

    A root's basename is the name searched for. That is enough for a directory, but not for a
    single file: a module is pulled in by `mod <stem>;`, which never spells the file name, so
    `name:` lines add the identifiers a restricted file is reached through.
    """
    text = REGISTER.read_text(encoding="utf-8")
    if RELEASE not in text:
        raise SystemExit(f"{REGISTER.name} has no '{RELEASE}' section")
    # The public tree keeps the section's prose and none of its block: no roots are restricted there.
    section = text.split(RELEASE, 1)[1].split("\n## ", 1)[0]
    if "```" not in section:
        return [], [], set()
    body = section.split("```")[1]
    excl, names, deps = [], [], set()
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("exclude:"):
            excl.append(line.removeprefix("exclude:").strip())
        elif line.startswith("name:"):
            names.append(line.removeprefix("name:").strip())
        elif line.startswith("depends:"):
            deps.add(line.removeprefix("depends:").strip())
    return excl, names, deps


def mentions(excluded: list[str], names: list[str]) -> set[str]:
    """Files outside the excluded roots whose text names one of them."""
    needles = [pathlib.PurePosixPath(e).name for e in excluded] + names
    found = set()
    roots = [ROOT.glob("*.md")] + [(ROOT / top).rglob("*") for top in SCAN]
    for group in roots:
        for path in sorted(group):
            if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
                continue
            rel = str(path.relative_to(ROOT))
            if any(rel.startswith(e) for e in excluded):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(n.lower() in text.lower() for n in needles):
                found.add(rel)
    return found


def data_rows() -> list[dict]:
    """The machine-readable half of the data register: one row per source."""
    text = REGISTER.read_text(encoding="utf-8")
    if DATA not in text:
        raise SystemExit(f"{REGISTER.name} has no '{DATA}' section")
    body = text.split(DATA, 1)[1].split("```")[1]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line == "text":
            continue
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) != 5:
            raise SystemExit(
                f"malformed data-register row (want 5 ' | ' fields): {line}"
            )
        key, verdict, read, hosts, paths = parts
        rows.append(
            {
                "key": key,
                "verdict": verdict,
                "read": read.removeprefix("read").strip(),
                "hosts": []
                if hosts == "hosts -"
                else hosts.removeprefix("hosts").split(),
                "paths": []
                if paths == "paths -"
                else paths.removeprefix("paths").split(),
            }
        )
    return rows


def fetch_hosts() -> dict[str, set[str]]:
    return {
        str(p.relative_to(ROOT)): set(HOST.findall(p.read_text(encoding="utf-8")))
        for p in sorted(ROOT.glob("tools/fetch-*.sh"))
    }


def data_check() -> int:
    rows = data_rows()
    prose = REGISTER.read_text(encoding="utf-8").split(DATA, 1)[1].split("```")[0]
    problems = []
    for r in rows:
        if r["verdict"] not in VERDICTS:
            problems.append(
                f"{r['key']}: verdict '{r['verdict']}' is not one of {sorted(VERDICTS)}"
            )
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", r["read"]):
            problems.append(f"{r['key']}: 'read' must be a date, got '{r['read']}'")
        if f"`{r['key']}`" not in prose:
            problems.append(f"{r['key']}: no row in the prose table above the block")
    registered = {h for r in rows for h in r["hosts"]}
    for script, hosts in fetch_hosts().items():
        for h in sorted(hosts - registered):
            problems.append(
                f"{script}: fetches from {h}, which has no data-register row"
            )
    files = {p for g in DATA_GLOBS for p in ROOT.glob(g) if p.is_file()}
    covered: set[pathlib.Path] = set()
    for r in rows:
        matched = {p for g in r["paths"] for p in ROOT.glob(g) if p.is_file()}
        if r["paths"] and not matched:
            problems.append(f"{r['key']}: none of its paths matches a committed file")
        covered |= matched
    for f in sorted(files - covered):
        problems.append(
            f"{f.relative_to(ROOT)}: committed data with no data-register row"
        )
    unread = sorted(r["key"] for r in rows if r["verdict"] == "unread")
    if "--public" in sys.argv:
        # A row whose every file is cut from the public tree, and which no shipped script fetches,
        # does not reach that tree, so its terms do not hold the release up.
        excluded = restricted()[0]
        shipped = [
            r["key"]
            for r in rows
            if r["verdict"] == "unread"
            and (
                r["hosts"] or not all(p.startswith(tuple(excluded)) for p in r["paths"])
            )
        ]
        problems += [f"{k}: terms unread -- not releasable until read" for k in shipped]
    for p in problems:
        print(p)
    print(
        f"{len(rows)} data rows, {len(registered)} hosts, {len(files)} committed data files, "
        f"{len(unread)} unread: {', '.join(unread) or 'none'}"
    )
    return 1 if problems else 0


def main() -> int:
    if "--data" in sys.argv:
        return data_check()
    hits = sweep()
    if "--check" not in sys.argv:
        print("\n".join(hits))
        print(f"{len(hits)} modules claim a port, a translation or a transcription")
        return 0

    have = covered()
    missing = [h for h in hits if h not in have]
    stale = sorted(have - set(hits))
    for m in missing:
        print(f"not in docs/provenance.md: {m}")
    for s in stale:
        print(f"in docs/provenance.md but no longer claims a port: {s}")
    if missing or stale:
        print(
            f"\n{len(hits)} modules swept, {len(have)} covered. "
            "Read the module, decide what it descends from, and record it -- "
            "a module may be recorded as clean, but not left out."
        )
        return 1
    excluded, names, recorded = restricted()
    found = mentions(excluded, names)
    added = sorted(found - recorded)
    gone = sorted(recorded - found)
    for a in added:
        print(f"newly depends on a licence-restricted module: {a}")
    for g in gone:
        print(
            f"recorded as depending on a licence-restricted module, no longer does: {g}"
        )
    if added or gone:
        print(
            f"\n{len(excluded)} restricted roots, {len(found)} dependent files against "
            f"{len(recorded)} recorded. Every one of these has to be edited or dropped to cut a "
            "public tree, so the list is the release checklist -- update it deliberately."
        )
        return 1

    print(f"docs/provenance.md covers all {len(hits)} modules that claim a port")
    print(
        f"and records all {len(found)} files coupled to the {len(excluded)} "
        "licence-restricted roots"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
