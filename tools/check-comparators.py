#!/usr/bin/env python3
"""Every test that another implementation still decides is in the comparator ledger.

P0 (`plans/17` §1.1): another code's output is a comparator. It is recorded and published as a
distance, and it never decides a test. This script finds the tests where one still does -- a
*comparator-decided* test -- and holds them to the ledger in `docs/verification.md` §12.1. The
ledger can only shrink: a test leaves it when it is parked as a comparator
(`#[ignore = "comparator: ..."]`) or rewritten against its module's math gate (`plans/17` Ф19.D).

A `#[test]` is comparator-decided when one of these holds and it is not parked:

  * **name** -- its name has another code's name as a word: `orekit`, `erfa`, `sofa`, ...;
  * **file** -- it is in `tests/` in a file whose name has one, or the word `oracle`;
  * **fixture** -- it reads a fixture whose file name has one, or `oracle`: by path, through a
    constant the file binds to the fixture, or through a helper in the same file that does;
  * **doc** -- its own doc comment names one.

The last three say where an expected value may have come from, and a test they catch can still
check mathematics: a test whose own doc comment names its check kind -- a gate `(G1)` ... `(G5)`,
or a supporting `(S1)` recomputation or `(S2)` property (`plans/17` §1.2) -- is not counted under
them. The tag is a reviewed claim about what the test asserts against, so it is written only after
reading the test's body. The name rule has no such exception -- a test named
after another code is renamed when it stops being one. A document is not a code: SGP4-VER's
`tcppver.out`, an ICD's tables and a textbook's worked example are specification (`plans/17`
§1.6), and agency products (IGS, ILRS, IERS, CelesTrak) are data.

Three failure modes, all mechanical:

  * a comparator-decided test that is not in the ledger -- a new one, which the ratchet refuses;
  * a ledger entry that is no longer comparator-decided -- the ledger is stale and the count wrong;
  * a ledger entry whose file or test no longer exists.

    uv run --no-project python tools/check-comparators.py            # the count, by crate and rule
    uv run --no-project python tools/check-comparators.py --check    # fail on the three cases above
    uv run --no-project python tools/check-comparators.py --ledger   # a ledger line per test found
    uv run --no-project python tools/check-comparators.py --report   # each test, its module, its gate
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
MARKER = "### 12.1 · Comparator-decided tests"
# Spelled in two pieces: the public cut deletes every line that carries the whole token.
PRIVATE = "<!-- @" + "private -->"

# Other implementations, by the names our tests and fixtures give them.
CODES = (
    "astropy",
    "basilisk",
    "disort",
    "erfa",
    "gmat",
    "hapsira",
    "hwm14",
    "jrc",
    "modtran",
    "nasa42",
    "neptune",
    "orekit",
    "poliastro",
    "pyerfa",
    "pyfes",
    "pymsis",
    "pyorbital",
    "pyspectral",
    "qhull",
    "scipy",
    "skyfield",
    "sofa",
    "spiceypy",
    "stk",
    "tudat",
)
CODE = re.compile(r"(?<![a-z0-9])(" + "|".join(CODES) + r")(?![a-z0-9])")
ORACLE = re.compile(r"(?<![a-z0-9])oracle(?![a-z0-9])")
GATE_TAG = re.compile(r"\((?:G[1-5]|S[12])\)")
# What a module doc names when its formulas have a math gate of their own: a derivation, a seal.
GATE = re.compile(
    r"derivations/[\w./-]+\.(?:mac|py|m|gp|input|sh|jl|c|g)|\b(?:spec_seals|table_seals)\b"
)

TEST_ATTR = re.compile(r"^\s*#\[(?:\w+::)*test\]")
ATTR = re.compile(r"^\s*#\[")
DOC = re.compile(r"^\s*///(.*)$")
FN = re.compile(
    r"^(\s*)(?:pub(?:\([^)]*\))?\s+)?(?:const\s+|async\s+|unsafe\s+)*fn\s+([A-Za-z_]\w*)"
)
CONST = re.compile(
    r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+([A-Z_][A-Z0-9_]*)\s*:"
)
LITERAL = re.compile(r'"([^"\\\n]*)"')
WORD = re.compile(r"[A-Za-z_]\w*")
DEF = re.compile(
    r"^\s*pub(?:\([^)]*\))?\s+(?:const\s+|async\s+|unsafe\s+)*"
    r"(?:fn|struct|enum|trait|type|const|static)\s+([A-Za-z_]\w*)"
)
USE = re.compile(r"^\s*use\s+(spacelab\w*)::(.*?);", re.S | re.M)


@dataclass(frozen=True)
class Test:
    path: str
    name: str
    line: int
    rule: str
    code: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.name}"


def files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "crates/*.rs"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(line for line in out.splitlines() if line)


def fixture_names() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "crates/*/tests/fixtures/*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {pathlib.PurePosixPath(line).name for line in out.splitlines() if line}


def body(lines: list[str], start: int, indent: str) -> tuple[str, int]:
    """The text of the item whose signature opens at `start`, to its closing brace at `indent`."""
    if lines[start].rstrip().endswith("}") and "{" in lines[start]:
        return lines[start], start
    for end in range(start + 1, len(lines)):
        if lines[end].rstrip() == indent + "}":
            return "\n".join(lines[start : end + 1]), end
    return "\n".join(lines[start:]), len(lines) - 1


def header(lines: list[str], fn_line: int) -> tuple[list[str], str]:
    """The attributes and doc comment directly above a function.

    Plain `//` comments may sit among them, and an attribute may span lines (`#[expect(` ...
    `)]`, an `ignore` string continued on the next line); the header ends at a blank line or at
    the end of the previous item.
    """
    attrs, docs = [], []
    i = fn_line - 1
    while i >= 0:
        stripped = lines[i].strip()
        if m := DOC.match(lines[i]):
            docs.append(m.group(1))
        elif stripped.startswith("//"):
            pass
        elif not stripped or (
            not ATTR.match(lines[i]) and stripped.endswith(("}", "{", ";"))
        ):
            break
        else:
            attrs.append(stripped)
        i -= 1
    return list(reversed(attrs)), " ".join(reversed(docs))


def comparator_fixture(name: str) -> str | None:
    low = name.lower()
    if m := CODE.search(low):
        return m.group(1)
    return "oracle" if ORACLE.search(low) else None


def scan(
    path: str, fixtures: set[str]
) -> tuple[list[Test], list[tuple[str, str, str]]]:
    """The comparator-decided tests of one file, and every test with its body (for `--report`)."""
    text = (ROOT / path).read_text(encoding="utf-8")
    lines = text.splitlines()

    def refs(chunk: str) -> str | None:
        for lit in LITERAL.findall(chunk):
            base = pathlib.PurePosixPath(lit).name
            if base in fixtures and (code := comparator_fixture(base)):
                return code
        return None

    bound: dict[str, str] = {}
    for i, line in enumerate(lines):
        if m := CONST.match(line):
            stmt = line
            j = i
            while ";" not in stmt and j + 1 < len(lines):
                j += 1
                stmt += lines[j]
            if code := refs(stmt):
                bound[m.group(1)] = code

    items = []
    for i, line in enumerate(lines):
        if m := FN.match(line):
            attrs, doc = header(lines, i)
            chunk, _ = body(lines, i, m.group(1))
            is_test = any(TEST_ATTR.match(a) for a in attrs)
            items.append((m.group(2), i + 1, attrs, doc, chunk, is_test))

    def reads(chunk: str, helpers: dict[str, str]) -> str | None:
        if code := refs(chunk):
            return code
        words = set(WORD.findall(chunk))
        for name, code in {**bound, **helpers}.items():
            if name in words:
                return code
        return None

    helpers: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for name, _, _, _, chunk, is_test in items:
            if is_test or name in helpers:
                continue
            inner = chunk.split("\n", 1)[1] if "\n" in chunk else ""
            if code := reads(inner, helpers):
                helpers[name] = code
                changed = True

    in_tests = "/tests/" in path
    file_code = None
    if in_tests:
        stem = pathlib.PurePosixPath(path).stem.lower()
        file_code = (m.group(1) if (m := CODE.search(stem)) else None) or (
            "oracle" if ORACLE.search(stem) else None
        )

    found, every = [], []
    for name, line, attrs, doc, chunk, is_test in items:
        if not is_test:
            continue
        every.append((path, name, chunk))
        if 'ignore = "comparator' in " ".join(attrs):
            continue
        tagged = bool(GATE_TAG.search(doc))
        if m := CODE.search(name):
            found.append(Test(path, name, line, "name", m.group(1)))
        elif tagged:
            continue
        elif file_code:
            found.append(Test(path, name, line, "file", file_code))
        elif code := reads(chunk, helpers):
            found.append(Test(path, name, line, "fixture", code))
        elif m := CODE.search(doc.lower()):
            found.append(Test(path, name, line, "doc", m.group(1)))
    return found, every


def restricted_roots() -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "check_provenance", ROOT / "tools" / "check-provenance.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restricted()[0]


def ledger() -> dict[str, str]:
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
        if len(parts) != 2 or "::" not in parts[0]:
            raise SystemExit(f"malformed ledger line: {line}")
        out[parts[0]] = parts[1]
    return out


def module_doc(path: pathlib.Path) -> str:
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


def report(found: list[Test], every: dict[str, tuple[str, str, str]]) -> None:
    """Each test with the module it exercises and whether that module names a math gate."""
    defs: dict[str, dict[str, list[str]]] = {}
    for path in files():
        parts = pathlib.PurePosixPath(path).parts
        if len(parts) < 3 or parts[2] != "src":
            continue
        crate = parts[1].replace("-", "_")
        for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
            if m := DEF.match(line):
                defs.setdefault(crate, {}).setdefault(m.group(1), []).append(path)
    for t in found:
        if "/src/" in t.path:
            module = t.path
        else:
            text = (ROOT / t.path).read_text(encoding="utf-8")
            imported = {
                (crate, w)
                for crate, names in USE.findall(text)
                for w in WORD.findall(names)
            }
            words = set(WORD.findall(every[t.key][2]))
            votes = Counter(
                f
                for crate, w in imported
                if w in words
                for f in defs.get(crate, {}).get(w, [])
            )
            module = votes.most_common(1)[0][0] if votes else "-"
        gate = "-"
        if module != "-":
            gate = "gate" if GATE.search(module_doc(ROOT / module)) else "no gate"
        print(f"{t.key} | {t.rule}:{t.code} | {module} | {gate}")


def main() -> int:
    fixtures = fixture_names()
    found: list[Test] = []
    every: dict[str, tuple[str, str, str]] = {}
    for path in files():
        f, e = scan(path, fixtures)
        found += f
        every.update({f"{p}::{n}": (p, n, c) for p, n, c in e})

    if "--ledger" in sys.argv:
        private = restricted_roots()
        for t in sorted(found, key=lambda t: t.key):
            mark = f" {PRIVATE}" if any(t.path.startswith(r) for r in private) else ""
            print(f"{t.key} | {t.rule}:{t.code}{mark}")
        return 0
    if "--report" in sys.argv:
        report(sorted(found, key=lambda t: t.key), every)
        return 0

    by_crate = Counter(pathlib.PurePosixPath(t.path).parts[1] for t in found)
    by_rule = Counter(t.rule for t in found)
    print(f"{len(every)} tests; {len(found)} comparator-decided")
    for rule in ("name", "file", "fixture", "doc"):
        print(f"  rule {rule:8s} {by_rule[rule]}")
    for crate, n in sorted(by_crate.items()):
        print(f"  {crate:32s} {n}")
    if "--check" not in sys.argv:
        return 0

    book = ledger()
    keys = {t.key for t in found}
    new = sorted(keys - book.keys())
    stale = sorted(book.keys() - keys)
    for k in new:
        print(f"comparator-decided and not in the ledger: {k}")
    for k in stale:
        why = (
            "test is gone"
            if k not in every
            else "no longer comparator-decided -- remove it"
        )
        print(f"stale ledger entry: {k} ({why})")
    if new or stale:
        print(
            "\nThe ledger in docs/verification.md §12.1 lists the tests that another implementation "
            'still decides. Park a new one as a comparator (#[ignore = "comparator: ..."]) or give '
            "its module a math gate; take an entry out when its test stops being one."
        )
        return 1
    print(f"ledger consistent: {len(book)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
