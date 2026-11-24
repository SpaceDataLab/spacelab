#!/usr/bin/env python3
"""The claims check passes the tree as it is, and refuses each planted stale value.

Every value every claim states is planted stale in turn, in the document, and must come back as a
mismatch; then the tree side moves under an unchanged sentence, and the sentence must go stale.

uv run --no-project python -I tools/test_check_claims.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("check_claims", HERE / "check-claims.py")
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["check_claims"] = tool
spec.loader.exec_module(tool)

REAL = tool.View(tool.ROOT)
TRUTHS = tool.evaluate(REAL)


def raw(rel: str) -> str:
    return (tool.ROOT / rel).read_text(encoding="utf-8")


def planted(overrides: dict[str, str], same_tree: bool = True) -> list[str]:
    """The problems of the tree with `overrides`; a document-only plant keeps the tree's truths."""
    v = tool.View(tool.ROOT, overrides)
    return tool.problems_of(v, TRUTHS if same_tree else None)


def stale(stated: str) -> str:
    """Another value of the same shape, so the sentence still matches and only the value is wrong."""
    folded = " ".join(stated.split())
    if re.fullmatch(r"\d{1,3}(?: \d{3})*|\d+", folded):
        return str(int(folded.replace(" ", "")) + 1)
    if folded.lower() in tool.NUMBER:
        word = tool.WORDS[tool.NUMBER[folded.lower()] + 1]
        return word.capitalize() if folded[0].isupper() else word
    if re.fullmatch(r"\d+(?:\.\d+)+", folded):
        return re.sub(r"\d+$", lambda m: str(int(m.group()) + 1), folded)
    return folded.replace("`", "`x", 1)


def test_the_tree_as_it_is_passes() -> None:
    found = tool.problems_of(REAL, TRUTHS)
    assert found == [], found


def test_each_stated_value_planted_stale_is_refused() -> None:
    checked = 0
    for claim in tool.CLAIMS:
        text = raw(claim.doc)
        m = re.search(claim.pattern.replace(" ", r"\s+"), text, re.S)
        assert m is not None, f"{claim.name}: the raw {claim.doc} has no match"
        for k in range(1, len(m.groups()) + 1):
            new = stale(m.group(k))
            assert new != m.group(k), (claim.name, k)
            found = planted({claim.doc: text[: m.start(k)] + new + text[m.end(k) :]})
            mine = f"{claim.name}: {claim.doc}:"
            assert any(p.startswith(mine) and " states " in p for p in found), (
                claim.name,
                k,
                found,
            )
            assert all(" states " in p for p in found), (claim.name, k, found)
            checked += 1
    assert checked >= 50, checked


def test_a_reworded_sentence_is_reported_missing() -> None:
    text = raw(tool.LEDGER).replace(
        "crates stay `publish = false`", "crates are unpublished"
    )
    found = planted({tool.LEDGER: text})
    assert found == [
        f"unpublished: {tool.LEDGER} no longer states it -- update the claim"
    ], found


def test_a_stale_sentence_on_a_private_line_is_not_read() -> None:
    line = "\n9 of the 40 crates are in the whole-crate mutation gate, here. The other 31 are outside it."
    assert any(
        p.startswith("mutation-gate: README.md")
        for p in planted({"README.md": raw("README.md") + line})
    )
    private = f"{line} <!-- {tool.view.LINE} -->\n"
    assert planted({"README.md": raw("README.md") + private}) == []


def test_a_badge_needs_a_workflow_that_a_push_starts() -> None:
    badge = (
        f"\n[![ci](https://{tool.REPOSITORY}actions/workflows/ci.yml/badge.svg)](x)\n"
    )
    readme = raw("README.md") + badge
    found = [p for p in planted({"README.md": readme}) if p.startswith("ci-badge")]
    assert len(found) == 1 and "no push starts" in found[0], found
    ci = raw(tool.CI).replace("\non:\n", "\non:\n  push:\n", 1)
    found = planted({"README.md": readme, tool.CI: ci}, same_tree=False)
    assert not [p for p in found if p.startswith("ci-badge")], found
    gone = readme.replace("ci.yml/badge", "nightly.yml/badge")
    found = [p for p in planted({"README.md": gone}) if p.startswith("ci-badge")]
    assert len(found) == 1 and "is not in" in found[0], found


def test_a_push_or_a_schedule_trigger_breaks_the_dispatch_claims() -> None:
    on = "\non:\n"
    weekly = raw(tool.WEEKLY)
    assert on in weekly
    found = planted(
        {tool.WEEKLY: weekly.replace(on, on + "  push:\n", 1)}, same_tree=False
    )
    assert sorted(p.split(":")[1].strip() for p in found) == [
        tool.CONTRIBUTING,
        tool.LEDGER,
    ], found
    found = planted(
        {tool.WEEKLY: weekly.replace(on, on + "  schedule:\n", 1)}, same_tree=False
    )
    assert len(found) == 1 and found[0].startswith(f"no-push: {tool.LEDGER}:"), found


def test_the_tree_moving_under_a_sentence_makes_it_stale() -> None:
    def moved(rel: str, old: str, new: str) -> list[str]:
        text = raw(rel)
        assert text.count(old) == 1, (rel, old)
        return planted({rel: text.replace(old, new)}, same_tree=False)

    def names(found: list[str]) -> list[str]:
        return sorted(p.split(":")[0] for p in found)

    found = moved(tool.MANIFEST, 'rust-version = "1.94"', 'rust-version = "1.95"')
    assert names(found) == ["msrv-badge"], found
    found = moved(tool.CI, "packages: -p spacelab-estimation\n", "packages: ''\n")
    assert names(found) == ["mutation-gate", "mutation-gate"], found
    found = moved(
        "Cargo.lock", 'name = "h3o"\nversion = "0.11.', 'name = "h3o"\nversion = "0.12.'
    )
    assert names(found) == ["h3o"], found
    found = moved(tool.GEO, "    sec: [f64; DIM],", "    sec: Box<[[f64; DIM]; DIM]>,")
    assert names(found) == ["geopotential-heap"], found


def test_a_set_aside_test_counts_unless_the_cut_deletes_its_file() -> None:
    def with_ignore(rel: str) -> list[str]:
        return planted({rel: raw(rel) + "\n#[ignore]\nfn f() {}\n"}, same_tree=False)

    found = with_ignore("crates/spacelab-core/src/lib.rs")
    assert sorted(p.split(":")[0] for p in found) == [
        "ignored-by-hand",
        "ignored-tests",
    ], found
    restricted = [
        r for r in REAL.excluded if r.startswith("crates/") and r.endswith(".rs")
    ]
    if not restricted:
        assert not REAL.excluded, (
            "a restricted tree without a restricted crate file to plant in"
        )
        return
    assert with_ignore(restricted[0]) == []


def test_values_are_read_as_the_tree_writes_them() -> None:
    assert tool.value("833 504") == 833504
    assert tool.value("Seven") == 7
    assert tool.value("no") == 0
    assert tool.value("1.94") == "1.94"
    assert tool.value("`a`,\n  `b`") == "`a`, `b`"


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
