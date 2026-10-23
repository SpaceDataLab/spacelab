#!/usr/bin/env python3
"""Publish the workspace to crates.io in dependency order, resumably.

Three things make a bare `cargo publish --workspace` the wrong tool for the first release:

* **crates.io rate-limits new crates.** A burst of a few, then one every few minutes, and this
  workspace creates 28 at once. The 429 names the time to retry after; this script sleeps until
  then instead of failing two hours in.
* **A failed run has to resume.** Every crate already on the registry at this version is skipped,
  read from the sparse index rather than remembered, so rerunning after any failure is safe.
* **The development tree must never be published.** It carries the models `docs/provenance.md`
  excludes from the public release. The cut tree has no release checklist in that file, so a tree
  that still has one is refused before anything is uploaded.

    python3 tools/publish-crates.py             # print the order and what is already published
    python3 tools/publish-crates.py --publish   # upload, in that order

The token is cargo's own (`cargo login` or `CARGO_REGISTRY_TOKEN`); this script never reads it.
Standard library only.
"""

from __future__ import annotations

import email.utils
import importlib.util
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = "https://index.crates.io"
RETRY_AFTER = re.compile(r"try again after (.+? GMT)")
FALLBACK_WAIT = 600


def restricted_roots() -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "provenance", ROOT / "tools" / "check-provenance.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.restricted()[0]


def publish_order() -> list[tuple[str, str]]:
    """`(name, version)` of every publishable package, each after everything it depends on.

    Dev-dependencies do not count: the workspace's are path-only, and `cargo publish` drops them.
    """
    meta = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    packages = {p["name"]: p for p in meta["packages"] if p["publish"] != []}
    deps = {
        name: {
            d["name"]
            for d in p["dependencies"]
            if d["kind"] != "dev" and d["name"] in packages
        }
        for name, p in packages.items()
    }
    order: list[str] = []
    while deps:
        ready = sorted(n for n, d in deps.items() if d <= set(order))
        if not ready:
            raise SystemExit(f"dependency cycle among {sorted(deps)}")
        order += ready
        for n in ready:
            del deps[n]
    return [(n, packages[n]["version"]) for n in order]


def index_path(name: str) -> str:
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def published(name: str, version: str) -> bool:
    try:
        with urllib.request.urlopen(f"{INDEX}/{index_path(name)}", timeout=30) as reply:
            lines = reply.read().decode().splitlines()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise
    return any(json.loads(line)["vers"] == version for line in lines if line)


def wait_seconds(stderr: str) -> float:
    m = RETRY_AFTER.search(stderr)
    if not m:
        return FALLBACK_WAIT
    until = email.utils.parsedate_to_datetime(m.group(1))
    return max(0.0, (until - datetime.now(UTC)).total_seconds()) + 5


def publish(name: str) -> None:
    while True:
        done = subprocess.run(
            ["cargo", "publish", "-p", name],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0:
            return
        if "429 Too Many Requests" not in done.stderr:
            sys.stderr.write(done.stderr)
            raise SystemExit(
                f"cargo publish -p {name} failed; rerun to resume after the fix"
            )
        wait = wait_seconds(done.stderr)
        print(f"  rate-limited; retrying {name} in {wait / 60:.1f} min", flush=True)
        time.sleep(wait)


def main() -> int:
    roots = restricted_roots()
    if roots:
        raise SystemExit(
            f"this is the development tree: {len(roots)} restricted roots are still present. "
            "Publish from the public tree that tools/cut-public-tree.py makes."
        )
    order = publish_order()
    pending = [(n, v) for n, v in order if not published(n, v)]
    for n, v in order:
        print(f"{'pending' if (n, v) in pending else 'on crates.io'}  {n} {v}")
    print(f"{len(pending)} of {len(order)} to publish")
    if "--publish" not in sys.argv[1:]:
        return 0
    for i, (n, v) in enumerate(pending, 1):
        print(f"[{i}/{len(pending)}] {n} {v}", flush=True)
        publish(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
