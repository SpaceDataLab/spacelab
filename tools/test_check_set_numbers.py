#!/usr/bin/env python3
"""The SET-number check refuses each planted defect, and passes the same text once it is fixed.

uv run --no-project python -I tools/test_check_set_numbers.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "check_set_numbers", HERE / "check-set-numbers.py"
)
assert spec is not None and spec.loader is not None
tool = importlib.util.module_from_spec(spec)
sys.modules["check_set_numbers"] = tool
spec.loader.exec_module(tool)

ROW = (
    "| `set-jb2008-indices` | SOLFSMY, DTCFILE | §7 and the Notice to Users; class "
    "`number-public`, provisional | **restricted** |"
)
ACK_MD = "**JB2008 acknowledgement (interim).** SET, JB2008, Bowman et al. (2008)."
ACK_RS = "//! JB2008 acknowledgement (interim). SET, JB2008, Bowman et al. (2008)."


def problems(files: dict[str, str], reviewed=None) -> list[str]:
    tree = {rel: tool.public_lines(text) for rel, text in files.items()}
    tree.setdefault(tool.REGISTER, tool.public_lines(ROW))
    return tool.problems_of(tree, reviewed or [])


def test_a_score_in_a_document_needs_the_acknowledgement() -> None:
    score = "Pooled, SDTM-1 reads 20.3 % against JB2008's 25.5 %.\n"
    assert len(problems({"docs/x.md": score})) == 1
    assert problems({"docs/x.md": f"{ACK_MD}\n\n{score}"}) == []


def test_a_ratio_and_a_table_column_count_too() -> None:
    ratio = "JB2008 reads 1.116 and 1.139 on the two satellites.\n"
    assert len(problems({"docs/x.md": ratio})) == 1
    table = "| model | bias |\n|---|---|\n| a | 1.0 |\n\n| day | JB2008 |\n|---|---|\n| quiet | 1.359 |\n"
    found = problems({"docs/x.md": table})
    assert len(found) == 1 and "(column)" in found[0], found
    assert problems({"docs/x.md": ACK_MD + "\n\n" + table}) == []


def test_a_comment_block_needs_its_own_acknowledgement() -> None:
    first = f"{ACK_RS}\n//! JB2008 at 1.116.\n\nfn f() {{}}\n"
    second = "/// JB2008 at 0.978 and 8.2 %.\nfn g() {}\n"
    assert problems({"tests/x.rs": first}) == []
    found = problems({"tests/x.rs": first + second})
    assert len(found) == 1 and "tests/x.rs:5:" in found[0], found


def test_a_private_line_is_not_public() -> None:
    hidden = (
        f"JB2008 reads 26.5 %. <!-- {tool.LINE} -->\n<!-- {tool.BEGIN} -->\n"
        f"| JB2008 | 26.5 % |\n<!-- {tool.END} -->\n"
    )
    assert problems({"docs/x.md": hidden}) == []


def test_a_name_with_no_number_passes() -> None:
    plain = "JB2008: §10.38, Ф18.A′, D-38, AIAA 2008-6438, 2026-10-15, `ΔTc` over 25 days.\n"
    assert problems({"docs/x.md": plain}) == []
    # The wide pattern is an upper bound: index names read as numbers, and REVIEWED holds them.
    assert len(problems({"docs/x.md": "JB2008's S10, Y10 and M10.\n"})) == 1


def test_a_sentence_giving_sets_files_no_terms_is_refused() -> None:
    for false in (
        "| JB2008, clean-room | our code; SET's indices state no terms: public |",
        "| SET SOLFSMY / DTCFILE (S10, M10, Y10, ΔTc) | no terms stated anywhere |",
        "Kyoto's Dst is closed, and SET's ΔTc has no stated terms.",
    ):
        found = problems({"docs/x.md": false + "\n"})
        assert len(found) == 1 and "state no terms" in found[0], (false, found)
    for true in (
        "The composites that would cover them exist and state no terms.",
        "SET's agreement covers the index files; their headers say nothing about terms.",
    ):
        assert problems({"docs/x.md": true + "\n"}) == [], true


def test_a_reviewed_line_passes_and_a_stale_one_is_refused() -> None:
    line = "# The columns that stand in for JB2008's S10, Y10 and M10."
    reviewed = [("tools/x.py", "stand in for JB2008's S10", "index names")]
    assert len(problems({"tools/x.py": line + "\n"})) == 1
    assert problems({"tools/x.py": line + "\n"}, reviewed) == []
    found = problems({"tools/x.py": "pass\n"}, reviewed)
    assert len(found) == 1 and "REVIEWED fragment matches no hit" in found[0], found


def test_the_register_row_must_record_the_class() -> None:
    bare = "| `set-jb2008-indices` | SOLFSMY, DTCFILE | §3, §6 | **restricted** |"
    found = problems({tool.REGISTER: bare})
    assert len(found) == len(tool.ROW_RECORDS), found
    assert len(problems({tool.REGISTER: "no row here"})) == 1


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
