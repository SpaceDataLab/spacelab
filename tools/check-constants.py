#!/usr/bin/env python3
"""Every long literal in library code names where it comes from (`plans/17` Ф17.D).

A literal with more than six significant digits in a crate's `src/`, outside `#[cfg(test)]`, is a
number someone transcribed or computed. A wrong digit there is invisible to most tests: the
mutation runs found tables where no test could see a flipped sign. So each such literal must say
where it comes from, in one of three ways:

  * **sealed** -- it sits in a table `derivations/table_seals.py` folds, which re-reads the emitted
    text against the compiled table (and `spec_seals.py` against the agency's file);
  * **derived** -- its line, the comment block right above it, or the one above its item (or above
    the first item of a run of items with no blank line between them) names a file under
    `derivations/` or `tools/` that exists: the derivation that checks it or the generator that
    emitted it; a file whose header says it is generated and names its generator counts whole;
  * **cited** -- the same comments cite a document: a standard (IERS, IAU, ICD, CODATA, WGS 84,
    ...), a paper (a year, "et al."), a URL or a DOI. A code is not a document (`plans/17` §1.1),
    so naming Orekit or SOFA does not count.

The rest are the ledger in `docs/verification.md` §12.2, one line per item with its count. The
ledger can only shrink: `--check` fails when an item gains a literal without a source, when a new
item has one, or when an entry no longer matches what the sweep finds.

    uv run --no-project python tools/check-constants.py            # the count, by class and crate
    uv run --no-project python tools/check-constants.py --check    # fail on any ledger mismatch
    uv run --no-project python tools/check-constants.py --ledger   # a ledger line per open item
    uv run --no-project python tools/check-constants.py --report   # each open literal and its line

Standard library only, like the other `check-*` tools, so it runs under a bare `python3` as well.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
LEDGER_DOC = ROOT / "docs" / "verification.md"
MARKER = "### 12.2 · Long literals without a named source"
# Spelled in two pieces: the public cut deletes every line that carries the whole token.
PRIVATE = "<!-- @" + "private -->"
DIGITS = 6

# A float literal, or an integer with an exponent; `_` separators and an `f64`/`f32` suffix allowed.
# Not after a word character or a dot, so neither `x2.5` nor a tuple field `t.0` is one.
NUMBER = re.compile(
    r"(?<![\w.])(\d[\d_]*\.[\d_]*(?:[eE][+-]?\d+)?|\d[\d_]*[eE][+-]?\d+)(?:_?f(?:32|64))?(?![\w.])"
)
ITEM = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const\s+fn|const|static|(?:async\s+|unsafe\s+)*fn)\s+"
    r"([A-Za-z_]\w*)"
)
CFG_TEST = re.compile(r"^\s*#\[cfg\((?:all\()?test\b")
ATTR = re.compile(r"^\s*#!?\[")
COMMENT = re.compile(r"^\s*//")
PATH = re.compile(r"((?:derivations|tools)/[\w./-]*\w)")
# Documents, not codes. A year counts after a capitalised word or an opening bracket, as in
# "Knocke (1988)", "Vallado 2013" or "(2010)", so that "n = 2000" in a comment does not.
CITE = re.compile(
    r"(?:\b[A-Z][\w'&.-]*,? \(?|\()(?:1[89]|20)\d\d[a-z]?\b(?!-\d)|et al\b|https?://|\b(?:doi|DOI)\b|"
    r"arXiv|\bIERS\b|\bIAU\b|\bICD\b|IS-GPS|IS-QZSS|CODATA|\bNIST\b|\bSI\b(?!\)| units)|\bWGS ?84\b|\bEGM\d|"
    r"\bGRS ?80\b|\bTN ?\d|\bRFC ?\d|\bISO ?\d|CCSDS|\bNGA\b|ITU-R|Conventions|\bTable \d|"
    r"\bEqs?\.? ?\(?\d"
)
STRING = re.compile(r'"(?:[^"\\\n]|\\.)*"')
CHAR = re.compile(r"'(?:[^'\\\n]|\\.)'")


@dataclass(frozen=True)
class Literal:
    path: str
    item: str
    line: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.item}"


def significant(token: str) -> int:
    mantissa = token.lower().replace("_", "").split("e")[0]
    digits = mantissa.replace(".", "").lstrip("0").rstrip("0")
    return len(digits)


def code_of(line: str) -> tuple[str, str]:
    """The line's code with strings and chars blanked, and its trailing `//` comment."""
    code = CHAR.sub("' '", STRING.sub('""', line))
    if "//" in code:
        at = code.index("//")
        return code[:at], line[at:]
    return code, ""


def files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "crates/*/src/*.rs", "crates/*/src/**/*.rs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted({line for line in out.splitlines() if line})


def sealed() -> set[tuple[str, str]]:
    spec = importlib.util.spec_from_file_location(
        "table_seals", ROOT / "derivations" / "table_seals.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {(f"crates/{path}", name) for path, names in module.TABLES for name in names}


def test_lines(lines: list[str]) -> set[int]:
    """The lines of every item under `#[cfg(test)]`, found by matching braces."""
    out: set[int] = set()
    i = 0
    while i < len(lines):
        if not CFG_TEST.match(lines[i]):
            i += 1
            continue
        start = i
        i += 1
        while i < len(lines) and (ATTR.match(lines[i]) or COMMENT.match(lines[i])):
            i += 1
        depth, opened = 0, False
        while i < len(lines):
            code, _ = code_of(lines[i])
            depth += code.count("{") - code.count("}")
            opened = opened or "{" in code
            if (opened and depth <= 0) or (not opened and code.rstrip().endswith(";")):
                break
            i += 1
        out.update(range(start, i + 1))
        i += 1
    return out


def block_above(lines: list[str], at: int) -> str:
    """The contiguous comment and attribute lines right above line `at`."""
    out = []
    j = at - 1
    while j >= 0 and (COMMENT.match(lines[j]) or ATTR.match(lines[j])):
        out.append(lines[j])
        j -= 1
    return "\n".join(out)


def exists(path: str) -> bool:
    """A named file, or a stem a comment gives with its extensions in braces (`x.{gp,mac}`)."""
    target = ROOT / path
    return target.exists() or any(target.parent.glob(target.name + ".*"))


def group_comment(lines: list[str], at: int, current: str) -> str:
    """The comment that covers the item at `at`: its own, or its group's.

    An item whose attributes follow the previous item's last line, with no blank line or comment
    between them, belongs to that item's group, and the comment above the group's first item covers
    it too (`#[rustfmt::skip]` tables under one generator note).
    """
    j = at - 1
    while j >= 0 and ATTR.match(lines[j]):
        j -= 1
    if j >= 0 and lines[j].strip() and not COMMENT.match(lines[j]):
        return current
    return block_above(lines, at)


def header(lines: list[str]) -> str:
    out = []
    for line in lines:
        if line.strip() and not line.lstrip().startswith(("//", "#!")):
            break
        out.append(line)
    return "\n".join(out)


def owners(lines: list[str], skip: set[int]) -> list[tuple[str, str]]:
    """Each line's item and the comment that covers it.

    An attribute, however many lines it spans, belongs to the item below it: the defaults in a
    `#[pyo3(signature = ...)]` are the next function's, not the last one's.
    """
    out = [("<module>", "")] * len(lines)
    item, comment = "<module>", ""
    pending: list[int] = []
    depth = 0
    for i, line in enumerate(lines):
        if i in skip:
            continue
        if depth > 0 or ATTR.match(line):
            code, _ = code_of(line)
            depth += code.count("[") - code.count("]")
            pending.append(i)
            continue
        if pending and COMMENT.match(line):
            continue
        found = ITEM.match(line)
        if found:
            start = pending[0] if pending else i
            item, comment = found.group(1), group_comment(lines, start, comment)
        for j in [*pending, i]:
            out[j] = (item, comment)
        pending = []
    return out


def scan(path: str, seals: set[tuple[str, str]]) -> tuple[Counter, list[Literal]]:
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    skip = test_lines(lines)
    owner = owners(lines, skip)
    head = header(lines)
    generated = "generat" in head.lower() and any(exists(p) for p in PATH.findall(head))
    tally: Counter = Counter()
    open_: list[Literal] = []
    for i, line in enumerate(lines):
        if i in skip:
            continue
        item, item_comment = owner[i]
        code, trailing = code_of(line)
        tokens = [t for t in NUMBER.findall(code) if significant(t) > DIGITS]
        if not tokens:
            continue
        if (path, item) in seals:
            kind = "sealed"
        elif generated:
            kind = "derived"
        else:
            context = "\n".join((trailing, block_above(lines, i), item_comment))
            kind = (
                "cited"
                if CITE.search(context)
                else "derived"
                if any(exists(p) for p in PATH.findall(context))
                else "open"
            )
        tally[kind] += len(tokens)
        if kind == "open":
            open_ += [Literal(path, item, i + 1, t) for t in tokens]
    return tally, open_


def restricted_roots() -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "check_provenance", ROOT / "tools" / "check-provenance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restricted()[0]


def ledger() -> dict[str, int]:
    text = LEDGER_DOC.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"{LEDGER_DOC.name} has no '{MARKER}' section")
    block = text.split(MARKER, 1)[1].split("```")[1]
    out = {}
    for line in block.splitlines():
        line = line.replace(PRIVATE, "").strip()
        if not line or line == "text":
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 2 or "::" not in parts[0] or not parts[1].isdigit():
            raise SystemExit(f"malformed ledger line: {line}")
        out[parts[0]] = int(parts[1])
    return out


def main() -> int:
    seals = sealed()
    tally: Counter = Counter()
    found: list[Literal] = []
    for path in files():
        t, o = scan(path, seals)
        tally += t
        found += o
    by_key = Counter(lit.key for lit in found)

    if "--ledger" in sys.argv:
        private = restricted_roots()
        for key, n in sorted(by_key.items()):
            mark = f" {PRIVATE}" if any(key.startswith(r) for r in private) else ""
            print(f"{key} | {n}{mark}")
        return 0
    if "--report" in sys.argv:
        for lit in found:
            print(f"{lit.path}:{lit.line}  {lit.item}  {lit.text}")
        return 0

    total = sum(tally.values())
    print(
        f"{total} literals with more than {DIGITS} significant digits in library code"
    )
    for kind in ("sealed", "derived", "cited", "open"):
        print(f"  {kind:8s} {tally[kind]}")
    by_crate = Counter(pathlib.PurePosixPath(lit.path).parts[1] for lit in found)
    for crate, n in sorted(by_crate.items()):
        print(f"  {crate:32s} {n}")
    if "--check" not in sys.argv:
        return 0

    book = ledger()
    bad = False
    for key in sorted(by_key.keys() | book.keys()):
        have, listed = by_key.get(key, 0), book.get(key, 0)
        if have > listed:
            print(
                f"{key}: {have} literals without a source, the ledger allows {listed}"
            )
            bad = True
        elif have < listed:
            print(
                f"{key}: the ledger lists {listed}, the sweep finds {have} -- lower the entry"
            )
            bad = True
    if bad:
        print(
            "\nThe ledger in docs/verification.md §12.2 lists the literals that name no source. Name "
            "the derivation, the generator or the document beside a new one; lower an entry when "
            "its literals gain one."
        )
        return 1
    print(f"ledger consistent: {len(book)} items, {sum(book.values())} literals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
