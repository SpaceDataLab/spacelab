#!/usr/bin/env python3
"""A tag's section of the changelog holds exactly the commits that tag released.

    python3 tools/test_gen_changelog.py

Builds a throwaway repository with two tags and a commit past the second, runs
`tools/gen-changelog.py` there, and checks each section. The generator once bounded every section
by `HEAD` instead of by its tag, so the first release also listed everything committed after it.
Standard library only, like the generator.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

GENERATOR = Path(__file__).resolve().parent / "gen-changelog.py"


def git(repo: Path, when: str, *args: str) -> None:
    """Run git with a fixed identity and clock, so tag order never depends on the second it ran."""
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@invalid",
        "GIT_AUTHOR_DATE": when,
        "GIT_COMMITTER_DATE": when,
    }
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


def sections(text: str) -> dict[str, str]:
    """`## title` -> the text under it, up to the next `## `."""
    out: dict[str, str] = {}
    title = None
    for line in text.splitlines():
        if line.startswith("## "):
            title = line[3:]
            out[title] = ""
        elif title is not None:
            out[title] += line + "\n"
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        (repo / "tools").mkdir()
        shutil.copyfile(GENERATOR, repo / "tools" / GENERATOR.name)
        git(repo, "2026-01-01T00:00:00Z", "init", "-q", "-b", "main")
        git(
            repo,
            "2026-01-01T00:00:00Z",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "feat: first",
        )
        git(repo, "2026-01-02T00:00:00Z", "tag", "-a", "v0.1.0", "-m", "v0.1.0")
        git(
            repo,
            "2026-01-03T00:00:00Z",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fix: second",
        )
        git(repo, "2026-01-04T00:00:00Z", "tag", "-a", "v0.2.0", "-m", "v0.2.0")
        git(
            repo,
            "2026-01-05T00:00:00Z",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "feat: third",
        )
        subprocess.run(
            [sys.executable, str(repo / "tools" / GENERATOR.name)],
            cwd=repo,
            check=True,
            capture_output=True,
        )
        found = sections((repo / "CHANGELOG.md").read_text(encoding="utf-8"))

    expected = {"Unreleased": "third", "v0.2.0": "second", "v0.1.0": "first"}
    failures = [
        f"{title}: expected only `{mine}`, got:\n{found.get(title, '(no section)')}"
        for title, mine in expected.items()
        if title not in found
        or mine not in found[title]
        or any(other in found[title] for other in expected.values() if other != mine)
    ]
    for failure in failures:
        print(failure, file=sys.stderr)
    if not failures:
        print("each section holds exactly the commits its release made")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
