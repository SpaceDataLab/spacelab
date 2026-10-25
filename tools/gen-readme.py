#!/usr/bin/env python3
"""Regenerate the crate table in README.md from the workspace itself.

The table this writes was hand-maintained until it described seven crates and the workspace had
thirty-two. A list of what a repository contains is not prose and should never be typed: it is a
projection of `cargo metadata` and of `docs/verification.md`, and the only way it stays true is to
be derived from both every time.

Two columns are derived rather than transcribed:

* **What** is the crate's own `description` in its `Cargo.toml` — the same string `cargo publish`
  would send to crates.io, so the README and the registry cannot disagree.
* **Verified in** links the section of `docs/verification.md` that covers the crate, from two signals
  that are read rather than declared here: the crate list each `## N · ...` heading opens with, and
  the rows whose first table column is the crate's short name. Both are needed — §2, §6 and §7 key
  their tables on the component rather than on the crate, so they have no matching rows at all, while
  §4 verifies two crates its opening list does not name. A crate neither signal reaches is printed as
  `--` rather than dropped, because a crate that ships with no verification section is the finding,
  not a formatting problem.

Each published crate's own `README.md` is the same projection for one crate: crates.io renders it
from the package, where no relative link resolves, so its links are absolute.

    python3 tools/gen-readme.py            # rewrite README.md and the crate READMEs in place
    python3 tools/gen-readme.py --check    # exit 1 if it is stale (this is what CI runs)

Standard library only, plus `cargo metadata`, so CI needs nothing the runner does not already have.
"""

from __future__ import annotations

import bisect
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
VERIFICATION = ROOT / "docs" / "verification.md"

BEGIN = "<!-- BEGIN GENERATED CRATE TABLE — tools/gen-readme.py -->"
END = "<!-- END GENERATED CRATE TABLE -->"

# The one crate that legitimately has no verification section: `spacelab-testkit` is a dev-only helper
# that exists to *write* the other crates' checks rather than to be one. Anything else the two signals
# do not reach is a gap, and the table says `--` so that it shows.
NOT_TABLED = {
    "spacelab-testkit": "dev-only: the comparison helpers the other crates' tests are written with",
}


def sections() -> list[tuple[int, int, str]]:
    """`(line, number, title)` for every `## N · ...` heading of the verification document."""
    found = []
    for i, line in enumerate(VERIFICATION.read_text().splitlines(), 1):
        m = re.match(r"## (\d+) · (.+)$", line)
        if m:
            found.append((i, int(m.group(1)), m.group(2)))
    if not found:
        raise SystemExit(
            f"{VERIFICATION}: no '## N · ...' headings — has the layout changed?"
        )
    return found


def declared_crates() -> dict[str, int]:
    """Crate -> section, from the crate names each section's opening paragraph puts in backticks."""
    lines = VERIFICATION.read_text().splitlines()
    declared: dict[str, int] = {}
    for line_no, number, _ in sections():
        # The opening list runs from the heading to the first blank line after it.
        opening = []
        for line in lines[line_no : line_no + 6]:
            if opening and not line.strip():
                break
            if line.strip():
                opening.append(line)
        for name in re.findall(r"`(spacelab(?:-[a-z0-9]+)*)`", " ".join(opening)):
            declared.setdefault(name, number)
    return declared


def anchor(number: int, title: str) -> str:
    """GitHub's heading anchor for `## N · Title`, by `github-slugger`'s actual rule.

    The slugger strips its punctuation set and then replaces spaces **one at a time**, so a space that
    stood beside a stripped character survives as a hyphen of its own: `coverage & constellations`
    anchors as `coverage--constellations`, not `coverage-constellations`. Collapsing the run is the
    obvious-looking simplification and it produces a link that lands silently at the top of the file --
    the same failure mode as an unresolved intra-doc link, and just as invisible from the rendered page.
    """
    slug = f"{number} · {title}".lower()
    return "#" + re.sub(r"[^\w\- ]", "", slug).strip().replace(" ", "-")


def rows_per_crate() -> dict[str, dict[int, int]]:
    """How many verification rows each crate has, per section number."""
    heads = sections()
    starts = [line for line, _, _ in heads]
    counts: dict[str, dict[int, int]] = {}
    for i, line in enumerate(VERIFICATION.read_text().splitlines(), 1):
        m = re.match(r"\| ([a-z0-9-]+) \|", line)
        if not m:
            continue
        section = heads[bisect.bisect_right(starts, i) - 1][1]
        counts.setdefault(m.group(1), {}).setdefault(section, 0)
        counts[m.group(1)][section] += 1
    return counts


def packages() -> list[dict]:
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    return sorted(meta["packages"], key=lambda p: p["name"])


def verified_in(name: str, docs: str) -> str:
    """The `Verified in` cell for one crate, its link resolved against `docs`."""
    counts = rows_per_crate()
    titles = {number: title for _, number, title in sections()}
    short = name.removeprefix("spacelab-")
    where = counts.get(short, {}) if short != name else {}
    section = declared_crates().get(name)
    if where:
        # A crate's rows can straddle two sections (`coverage` is in both §3 and §4); prefer the
        # one holding most of them, and let the count say how many there are in total.
        section = max(where, key=lambda s: where[s])
    if section is not None:
        link = (
            f"[§{section} {titles[section]}]({docs}{anchor(section, titles[section])})"
        )
        return f"{link} ({sum(where.values())} rows)" if where else link
    if name in NOT_TABLED:
        return NOT_TABLED[name]
    return "**--**"


def description(pkg: dict) -> str:
    text = (pkg.get("description") or "").strip().rstrip(".")
    if not text:
        raise SystemExit(
            f"{pkg['name']}: no `description` in Cargo.toml — the README is built from it"
        )
    return text


def table() -> str:
    lines = [
        BEGIN,
        "",
        "| Crate | What | Verified in |",
        "|---|---|---|",
    ]
    for pkg in packages():
        name = pkg["name"]
        verified = verified_in(name, "docs/verification.md")
        lines += [f"| [`{name}`](crates/{name}) | {description(pkg)} | {verified} |"]
    lines += ["", END]
    return "\n".join(lines)


def crate_readme(pkg: dict) -> str:
    """The README `cargo publish` uploads: crates.io renders it, so every link is absolute."""
    name = pkg["name"]
    repo = pkg["repository"]
    verified = verified_in(name, f"{repo}/blob/main/docs/verification.md")
    return "\n".join(
        [
            f"<!-- Generated by tools/gen-readme.py from {name}'s Cargo.toml. Do not edit. -->",
            "",
            f"# {name}",
            "",
            f"{description(pkg)}.",
            "",
            f"Part of [spacelab]({repo}), astrodynamics in Rust held to one rule: mathematics",
            "decides — a derivation, an exact value, the equation that defines the quantity — and",
            "another code's output is a comparison, never the proof. This crate is verified in",
            f"{verified}.",
            "",
            f"- API documentation: <https://docs.rs/{name}>",
            f"- Homepage: <{pkg['homepage']}>",
            "",
            "## Licence",
            "",
            f"Licensed under either of [Apache-2.0]({repo}/blob/main/LICENSE-APACHE) or",
            f"[MIT]({repo}/blob/main/LICENSE-MIT) at your option. Third-party attributions are in",
            f"[NOTICE]({repo}/blob/main/NOTICE).",
            "",
        ]
    )


def main() -> int:
    check = "--check" in sys.argv[1:]
    text = README.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{README}: the generated block's markers are missing")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    targets = {README: head + table() + tail}
    # `publish = false` comes through `cargo metadata` as an empty registry list.
    for pkg in packages():
        if pkg["publish"] != []:
            targets[Path(pkg["manifest_path"]).parent / "README.md"] = crate_readme(pkg)
    stale = [
        p for p, body in targets.items() if not p.exists() or p.read_text() != body
    ]
    if not stale:
        print(f"README.md and {len(targets) - 1} crate READMEs are current")
        return 0
    if check:
        for p in stale:
            print(f"{p.relative_to(ROOT)} is stale — run `python3 tools/gen-readme.py`")
        return 1
    for p in stale:
        p.write_text(targets[p])
        print(f"{p.relative_to(ROOT)} rewritten")
    return 0


if __name__ == "__main__":
    sys.exit(main())
