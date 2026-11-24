#!/usr/bin/env python3
"""Every counted statement of the README and the ledger, compared with the tree (`plans/18` Ф32.K, rule 17).

A count, a size or a version that a document states about this repository goes stale the day the tree
moves, and nothing reads the sentence again. The README said "about 0.84 MB" of stack after the tables
had moved to the heap; §8 said 26 crates of 32 when there were 33; §7 counted 102 `#[test]` in the
bridge's `lib.rs` when there were 113. Each claim below names its sentence by a pattern, reads the values the sentence
states, and reads the same values off the tree. `--check` fails when:

  * a stated value differs from the tree's;
  * a pattern matches nowhere: the sentence was reworded or deleted, and its claim must follow it;
  * a README badge shows a workflow of this repository that no push starts, so the badge reports a
    run somebody chose to dispatch, not `main`.

Everything is read in the public view: the lines the public cut keeps, and no file under a root that
`docs/provenance.md`'s release block excludes. So one sentence holds in both trees, and a count only the
private tree has goes on a line the cut deletes, which this check does not read.

What it does not cover:

  * measured numbers -- residuals, scores, the mutants of a dated run. Their tools print them again,
    or they stand as a dated record;
  * the crate table and the figure captions of the README, which `tools/gen-readme.py` and
    `tools/gen-figures.py` write;
  * scope words such as "every" and "all", which name a set rather than count one: rule 17 leaves them
    to review;
  * any sentence that is not in CLAIMS. A new counted sentence about the tree needs its row here.

    uv run --no-project python -I tools/check-claims.py           # each claim, stated against the tree
    uv run --no-project python -I tools/check-claims.py --check   # exit 1 on a stale or missing claim

Standard library only.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import re
import sys
import tomllib
from collections.abc import Callable

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The public cut's own reading of a file: `public_lines` drops what the cut drops.
view = load("check_set_numbers", "check-set-numbers.py")

README = "README.md"
LEDGER = "docs/verification.md"
CONTRIBUTING = "CONTRIBUTING.md"
MANIFEST = "Cargo.toml"
WORKFLOWS = ".github/workflows"
CI = f"{WORKFLOWS}/ci.yml"
WEEKLY = f"{WORKFLOWS}/weekly.yml"
REPOSITORY = "github.com/SpaceDataLab/spacelab/"

# fmt: off
WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
)
# fmt: on
NUMBER = {w: n for n, w in enumerate(WORDS)} | {"no": 0, "none": 0}


class Stale(Exception):
    """A fact the sentence asserts and the tree contradicts, in words."""


class View:
    """The tree as the public cut sees it; `overrides` replaces a file's text, for the negative tests."""

    def __init__(
        self, root: pathlib.Path, overrides: dict[str, str] | None = None
    ) -> None:
        self.root = root
        self.overrides = overrides or {}
        self.excluded = tuple(view.load_checker(root).restricted()[0])
        self.read: dict[str, tuple[str | None, ...]] = {}

    def public(self, rel: str) -> bool:
        """Whether the cut keeps `rel`: it deletes each excluded root, a file or a whole directory."""
        return not any(rel == r or rel.startswith(f"{r}/") for r in self.excluded)

    def numbered(self, rel: str) -> tuple[str | None, ...]:
        """The file's lines, `None` where the cut deletes one, so an index is a line number."""
        if rel not in self.read:
            if rel in self.overrides:
                text = self.overrides[rel]
            else:
                text = (self.root / rel).read_text(encoding="utf-8")
            self.read[rel] = tuple(view.public_lines(text, rel))
        return self.read[rel]

    def lines(self, rel: str) -> tuple[str, ...]:
        return tuple(x for x in self.numbered(rel) if x is not None)

    def text(self, rel: str) -> str:
        return "\n".join(self.lines(rel))

    def files(self, pattern: str) -> list[str]:
        found = (p.relative_to(self.root).as_posix() for p in self.root.glob(pattern))
        return sorted(rel for rel in found if self.public(rel))


def one(pattern: str, text: str, where: str) -> re.Match[str]:
    found = re.search(pattern, text, re.M)
    if found is None:
        raise SystemExit(
            f"{where}: no match for {pattern!r} -- has the file's layout changed?"
        )
    return found


def listing(names: list[str]) -> str:
    """`a`, `b` and `c`, as the ledger writes a list in prose."""
    quoted = [f"`{n}`" for n in names]
    return (
        quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
    )


# ── What the tree says ────────────────────────────────────────────────────────────────────────────


def workspace(v: View) -> dict:
    return tomllib.loads(v.text(MANIFEST))["workspace"]


def members(v: View) -> list[str]:
    names = []
    for pattern in workspace(v)["members"]:
        names += [
            pathlib.PurePosixPath(rel).parent.name
            for rel in v.files(f"{pattern}/Cargo.toml")
        ]
    return sorted(set(names))


def rust_version(v: View) -> tuple[str]:
    return (workspace(v)["package"]["rust-version"],)


def minor_version(v: View) -> tuple[str]:
    return (".".join(workspace(v)["package"]["version"].split(".")[:2]),)


def python_floors(v: View) -> tuple[str, str]:
    text = v.text("crates/spacelab-python/Cargo.toml")
    abi3 = one(r'"abi3-py3(\d+)"', text, "spacelab-python/Cargo.toml").group(1)
    abi3t = one(r'"abi3t-py3(\d+)"', text, "spacelab-python/Cargo.toml").group(1)
    return (f"3.{abi3}", f"3.{abi3t}")


def job(text: str, name: str, where: str) -> str:
    """One job's block of a workflow: from `  name:` to the next job at the same indent."""
    start = one(rf"^  {re.escape(name)}:\n", text, where).end()
    end = re.search(r"^  [\w-]+:\n", text[start:], re.M)
    return text[start : start + end.start()] if end else text[start:]


def sealed(v: View) -> list[str]:
    """The crates `ci.yml`'s whole-crate mutation job names in its `-p` lists."""
    return sorted(
        set(re.findall(r"-p (spacelab[\w-]*)", job(v.text(CI), "mutants-sealed", CI)))
    )


def mutation_gate(v: View) -> tuple[int, int, int]:
    n, total = len(sealed(v)), len(members(v))
    return (n, total, total - n)


def mutation_gate_ledger(v: View) -> tuple[int, int]:
    return mutation_gate(v)[:2]


def triggers(text: str) -> set[str]:
    """The events of a workflow's `on:` block."""
    block = one(r"^on:\n((?:[ #].*\n|\n)*)", text + "\n", "workflow").group(1)
    return set(re.findall(r"^  ([\w-]+):", block, re.M))


def started_by(v: View, events: set[str] | None) -> list[str]:
    """The workflows that start on one of `events`, or on anything but a dispatch for `None`."""
    out = []
    for rel in v.files(f"{WORKFLOWS}/*.y*ml"):
        on = triggers(v.text(rel))
        if (on - {"workflow_dispatch"}) if events is None else (on & events):
            out.append(rel)
    return out


def only_dispatch(v: View) -> tuple[()]:
    if started := started_by(v, None):
        raise Stale(f"{', '.join(started)} start(s) on more than a dispatch")
    return ()


def nothing_on_push(v: View) -> tuple[()]:
    if started := started_by(v, {"push", "pull_request", "pull_request_target"}):
        raise Stale(f"{', '.join(started)} start(s) on a push or a pull request")
    return ()


GEO = "crates/spacelab-propagation/src/geopotential.rs"


def geopotential_tables(v: View) -> tuple[int, int]:
    text = v.text(GEO)
    degree = int(one(r"^const MAX_DEGREE: usize = (\d+);", text, GEO).group(1))
    dim = degree + int(
        one(r"^const DIM: usize = MAX_DEGREE \+ (\d+);", text, GEO).group(1)
    )
    body = one(r"(?s)^pub struct GeoPotential \{\n(.*?)^\}", text, GEO)
    tables = len(re.findall(r": Box<\[\[f64; DIM\]; DIM\]>", body.group(1)))
    return (tables, tables * dim * dim * 8)


FES = "crates/spacelab-propagation/src/ocean_tide_fes2014_table.in"


def fes_degree(v: View) -> tuple[int]:
    return (max(int(n) for n in re.findall(r"^\s*\((\d+), \d+, ", v.text(FES), re.M)),)


def fes_waves(v: View) -> tuple[int, int]:
    return (v.text(FES).count("OceanTideWave {"), fes_degree(v)[0])


COEFFS = "crates/spacelab-propagation/src/geopotential_coeffs.rs"


def geopotential_rows(v: View) -> tuple[int, int]:
    text = v.text(COEFFS)
    degree = int(
        one(r"^pub const MODEL_MAX_DEGREE: usize = (\d+);", text, COEFFS).group(1)
    )
    body = text.split("pub const MODEL_COEFFS", 1)[1].split("\n];", 1)[0]
    return (degree, len(re.findall(r"^\s*\(\d+, \d+, ", body, re.M)))


NUT = "crates/spacelab-fa/src/nut_tables.rs"


def nutation_terms(v: View) -> tuple[int, int]:
    text = v.text(NUT)
    luni = one(r"XLS_MULT: \[\[i8; \d+\]; (\d+)\]", text, NUT).group(1)
    planetary = one(r"XPL_MULT: \[\[i8; \d+\]; (\d+)\]", text, NUT).group(1)
    return (int(luni), int(planetary))


PYTHON = "crates/spacelab-python/src/lib.rs"


def interpreter_split(lines: tuple[str, ...]) -> int:
    """The line that opens the tests needing an interpreter; every `#[test]` after it is one."""
    for k, line in enumerate(lines):
        if "mod interpreter_tests" in line:
            return k
    raise SystemExit(
        f"{PYTHON}: no `mod interpreter_tests` -- has the file's layout changed?"
    )


def python_tests(v: View) -> tuple[int, int, int, int]:
    lines = v.lines(PYTHON)
    split = interpreter_split(lines)
    tests = [k for k, line in enumerate(lines) if line.strip() == "#[test]"]
    gated = [k for k in tests if k > split]
    return (len(tests), len(gated), len(tests) - len(gated), len(gated))


def python_onnx(v: View) -> tuple[int, int]:
    lines = v.lines(PYTHON)
    split = interpreter_split(lines)
    onnx = [
        k
        for k, line in enumerate(lines)
        if k > split
        and line.strip() == "#[test]"
        and any('feature = "onnx"' in x for x in lines[max(k - 3, 0) : k + 3])
    ]
    return (len(onnx), python_tests(v)[1])


# A test set aside, with a reason, and not a parked comparator (`#[ignore = "comparator: ..."]`).
IGNORED = re.compile(r'^\s*#\[ignore(?:\]|\s*=\s*"(?!comparator:))')


def ignored(v: View) -> dict[str, int]:
    """Each crate file's count of set-aside tests."""
    counts = {}
    for rel in v.files("crates/**/*.rs"):
        n = sum(1 for line in v.lines(rel) if IGNORED.match(line))
        if n:
            counts[rel] = n
    return counts


def weekly_jobs(v: View) -> list[str]:
    text = v.text(WEEKLY)
    block = one(r"^jobs:\n", text, WEEKLY).end()
    return re.findall(r"^  ([\w-]+):\n", text[block:], re.M)


def weekly_files(v: View) -> set[str]:
    """The test files a weekly job runs with `--ignored`."""
    runs = re.findall(
        r"cargo test .*-p ([\w-]+) --test (\w+) -- --ignored", v.text(WEEKLY)
    )
    return {f"crates/{crate}/tests/{name}.rs" for crate, name in runs}


def weekly(v: View) -> tuple[int, int, int, str]:
    counts, run = ignored(v), weekly_files(v)
    jobs = weekly_jobs(v)
    covered = sum(n for rel, n in counts.items() if rel in run)
    return (sum(counts.values()), covered, len(jobs), ", ".join(f"`{j}`" for j in jobs))


LAMBDA = "crates/spacelab-gnss/src/lambda.rs"


def by_hand(v: View) -> tuple[int, str]:
    counts, run = ignored(v), weekly_files(v)
    rest = {rel: n for rel, n in counts.items() if rel not in run and rel != LAMBDA}
    names = sorted(pathlib.PurePosixPath(rel).stem for rel in rest)
    return (sum(rest.values()), listing(names))


def toolchain(v: View) -> tuple[str]:
    return (
        one(
            r'^channel = "([^"]+)"',
            v.text("rust-toolchain.toml"),
            "rust-toolchain.toml",
        ).group(1),
    )


def msrv_job(v: View) -> tuple[str]:
    return (
        one(r'RUSTUP_TOOLCHAIN: "([^"]+)"', job(v.text(CI), "msrv", CI), CI).group(1),
    )


def unpublished(v: View) -> tuple[int]:
    n = 0
    for name in members(v):
        package = tomllib.loads(v.text(f"crates/{name}/Cargo.toml"))["package"]
        n += package.get("publish") is False
    return (n,)


def block(v: View, marker: str) -> list[str]:
    """The lines of the ```text block that follows `marker` in the ledger."""
    text = v.text(LEDGER)
    if marker not in text:
        raise SystemExit(f"{LEDGER}: no '{marker}' section")
    body = text.split(marker, 1)[1].split("```")[1]
    return [line.strip() for line in body.splitlines()[1:] if line.strip()]


COMPARATORS = "### 12.1 · Comparator-decided tests"


def comparator_decided(v: View) -> tuple[int, int, int, int, int, int, int]:
    entries = [line.split("|") for line in block(v, COMPARATORS)]
    rules = [e[1].strip().split(":")[0] for e in entries]
    codes = [e[1].strip().split(":")[1] for e in entries]
    propagation = sum(e[0].startswith("crates/spacelab-propagation/") for e in entries)
    return (
        len(entries),
        *(rules.count(r) for r in ("name", "file", "fixture", "doc")),
        codes.count("orekit"),
        propagation,
    )


CENSUS = "## 12 · Formula provenance census"


def census(v: View) -> tuple[int, int, int, int, int, int, int]:
    derivations = load("check_derivations", "check-derivations.py")
    derivations.ROOT = v.root
    modules = [m for m in derivations.modules() if v.public(m)]
    entries = [line.split("|")[1].strip() for line in block(v, CENSUS)]
    named = len(modules) - len(entries)
    kinds = (entries.count(k) for k in ("physics", "parser", "generated", "infra"))
    return (len(modules), named, len(entries), *kinds)


LITERALS = "### 12.2 · Long literals without a named source"


def open_literals(v: View) -> tuple[int, int]:
    entries = [int(line.split("|")[1]) for line in block(v, LITERALS)]
    return (sum(entries), len(entries))


def open_literals_prose(v: View) -> tuple[int]:
    return (open_literals(v)[0],)


def h3o(v: View) -> tuple[str]:
    lock = v.text("Cargo.lock")
    full = one(r'^name = "h3o"\nversion = "([^"]+)"', lock, "Cargo.lock").group(1)
    return (".".join(full.split(".")[:2]),)


# ── The sentences ─────────────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Claim:
    """A sentence of `doc`, found by `pattern` (whitespace in the document folds to one space), whose
    groups state the values `truth` reads off the tree, in the same order."""

    name: str
    doc: str
    pattern: str
    truth: Callable[[View], tuple]


CLAIMS = (
    Claim("msrv-badge", README, r"badge/MSRV-(\d+\.\d+)-", rust_version),
    Claim("dependency", README, r'spacelab = "(\d+\.\d+)"', minor_version),
    Claim(
        "python",
        README,
        r"CPython (3\.\d+)\+ and free-threaded (3\.\d+)t",
        python_floors,
    ),
    Claim(
        "mutation-gate",
        README,
        r"(\d+) of the (\d+) crates are in the whole-crate mutation gate, .*? The other (\d+) are "
        r"outside it",
        mutation_gate,
    ),
    Claim(
        "geopotential-heap",
        README,
        r"A degree-120 geopotential allocates its (\w+) tables once, ([\d ]+) bytes on the heap",
        geopotential_tables,
    ),
    Claim("fes2014-degree", README, r"FES2014b ocean tide to degree (\d+)", fes_degree),
    Claim(
        "mutation-gate",
        LEDGER,
        r"names the crates taken to zero survivors — \*\*(\d+)\*\* of the workspace's (\d+),",
        mutation_gate_ledger,
    ),
    Claim(
        "no-push",
        LEDGER,
        r"Every trigger of `ci\.yml` and `weekly\.yml` is `workflow_dispatch`",
        only_dispatch,
    ),
    Claim(
        "no-push",
        CONTRIBUTING,
        r"No CI job runs on a push or a pull request",
        nothing_on_push,
    ),
    Claim(
        "geopotential-rows",
        LEDGER,
        r"`geopotential_coeffs\.rs` — XGM2019e to degree (\d+), ([\d ]+) rows",
        geopotential_rows,
    ),
    Claim(
        "fes2014-waves",
        LEDGER,
        r"`ocean_tide_fes2014_table\.in` — FES2014b, (\d+) waves to degree (\d+)",
        fes_waves,
    ),
    Claim(
        "nutation-terms",
        LEDGER,
        r"`nut_tables\.rs` — IAU 2000A nutation, (\d+) \+ (\d+) terms",
        nutation_terms,
    ),
    Claim(
        "python-tests",
        LEDGER,
        r"(\d+) `#\[test\]` in `lib\.rs`, .*? (\d+) of them need an interpreter .*? the other (\d+) "
        r"run in an ordinary `cargo test --workspace`\. One of the (\d+) also needs `onnx`",
        python_tests,
    ),
    Claim("python-onnx", LEDGER, r"(\w+) of the (\d+) also needs `onnx`", python_onnx),
    Claim(
        "ignored-tests",
        LEDGER,
        r"(\d+) tests carry an `#\[ignore\]` that does not park a comparator .*? runs (\d+) of "
        r"them in (\d+) independent jobs \(([^)]*)\)",
        weekly,
    ),
    Claim(
        "ignored-by-hand",
        LEDGER,
        r"The other (\w+), in (.*?), are run by hand",
        by_hand,
    ),
    Claim(
        "toolchain-pin",
        LEDGER,
        r"`rust-toolchain\.toml` names `(\d+\.\d+\.\d+)`",
        toolchain,
    ),
    Claim(
        "toolchain-pin", LEDGER, r"the toolchain pin \(`(\d+\.\d+\.\d+)`\)", toolchain
    ),
    Claim("msrv-job", LEDGER, r'`RUSTUP_TOOLCHAIN: "(\d+\.\d+\.\d+)"`', msrv_job),
    Claim("unpublished", LEDGER, r"(\w+) crates stay `publish = false`", unpublished),
    Claim(
        "comparator-decided",
        LEDGER,
        r"\*\*Today: (\d+) comparator-decided tests\*\* — name (\d+), file (\d+), fixture (\d+), "
        r"doc (\d+); (\w+) of them Orekit's, (\d+) in `spacelab-propagation`",
        comparator_decided,
    ),
    Claim(
        "census",
        LEDGER,
        r"\*\*Today, in the public tree: (\d+) modules, (\d+) name a check, (\d+) do not\*\* — "
        r"(\w+) physics module, (\d+) parsers, (\d+) generated and (\d+) infrastructure",
        census,
    ),
    Claim(
        "open-literals",
        LEDGER,
        r"\*\*Today: (\d+) open literals in (\d+) items\.\*\*",
        open_literals,
    ),
    Claim(
        "open-literals",
        LEDGER,
        r"In the public tree (\w+) literals stay open",
        open_literals_prose,
    ),
    Claim("h3o", MANIFEST, r"(\d+\.\d+), the version in the lock", h3o),
)

BADGE = re.compile(re.escape(REPOSITORY) + r"actions/workflows/([\w.-]+)/badge\.svg")


def value(stated: str) -> int | str:
    """A stated value as the tree's functions return it: a count as an int, anything else as text."""
    folded = " ".join(stated.split())
    if re.fullmatch(r"\d{1,3}(?: \d{3})*|\d+", folded):
        return int(folded.replace(" ", ""))
    return NUMBER.get(folded.lower(), folded)


def folded(v: View, rel: str) -> tuple[str, list[tuple[int, int]]]:
    """The document with each run of whitespace folded to one space, and where each line starts."""
    starts, parts, at = [], [], 0
    for n, line in enumerate(v.numbered(rel), 1):
        if line is None:
            continue
        part = " ".join(line.split())
        starts.append((at, n))
        parts.append(part)
        at += len(part) + 1
    return " ".join(parts), starts


def line_of(starts: list[tuple[int, int]], offset: int) -> int:
    return max(n for s, n in starts if s <= offset)


def evaluate(v: View) -> dict[str, tuple | Stale]:
    """Each claim's truth, read once per tree."""
    out: dict[str, tuple | Stale] = {}
    for claim in CLAIMS:
        key = claim.truth.__name__
        if key not in out:
            try:
                out[key] = claim.truth(v)
            except Stale as stale:
                out[key] = stale
    return out


def problems_of(v: View, truths: dict[str, tuple | Stale] | None = None) -> list[str]:
    truths = evaluate(v) if truths is None else truths
    found: list[str] = []
    docs = {rel: folded(v, rel) for rel in {claim.doc for claim in CLAIMS}}
    for claim in CLAIMS:
        text, starts = docs[claim.doc]
        matches = list(re.finditer(claim.pattern, text))
        if not matches:
            found.append(
                f"{claim.name}: {claim.doc} no longer states it -- update the claim"
            )
            continue
        truth = truths[claim.truth.__name__]
        for m in matches:
            where = f"{claim.doc}:{line_of(starts, m.start())}"
            if isinstance(truth, Stale):
                found.append(f"{claim.name}: {where} -- {truth}")
                continue
            stated = tuple(value(g) for g in m.groups())
            if stated != truth:
                found.append(
                    f"{claim.name}: {where} states {stated}, the tree says {truth}"
                )
    readme = v.text(README)
    for workflow in BADGE.findall(readme):
        rel = f"{WORKFLOWS}/{workflow}"
        if not (v.root / rel).exists() and rel not in v.overrides:
            found.append(
                f"ci-badge: {README} shows {workflow}, which is not in {WORKFLOWS}"
            )
        elif "push" not in triggers(v.text(rel)):
            found.append(
                f"ci-badge: {README} shows {workflow}, which no push starts: the badge is the last "
                "dispatched run, not `main`"
            )
    return found


def main() -> int:
    v = View(ROOT)
    truths = evaluate(v)
    for claim in CLAIMS:
        truth = truths[claim.truth.__name__]
        shown = truth if not isinstance(truth, Stale) else f"stale: {truth}"
        print(f"{claim.name:20s} {claim.doc:22s} {shown}")
    found = problems_of(v, truths)
    if "--check" not in sys.argv[1:]:
        return 0
    for p in found:
        print(p)
    print(f"{len(CLAIMS)} claims; {len(found)} stale or missing")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
