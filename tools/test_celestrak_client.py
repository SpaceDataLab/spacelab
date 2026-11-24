#!/usr/bin/env python3
"""Every script in tools/ that asks CelesTrak for data keeps its usage policy, against a local server.

    uv run --no-project python -I tools/test_celestrak_client.py

A local server plays CelesTrak: a 200 with validators, a 304 to a request that carries them, a 301
with a `Location` and a message, a 403 with a message. `fetch-density-data.sh --sw-all` must
download `SW-All.csv` once, not ask again inside one three-hour update, ask after it with the copy's
validators, keep the copy on a 304, and stop on a 301 or a 403 with the provider's text, its
`Location` unfollowed and no partial file left. `fetch-supgp.sh` and `fetch-satellite-js.sh` must
stop on the same two answers (rule 26 of `plans/18`: every fetch path, the same gate).

Rule 16: the same scenarios run against the `fetch()` and driver loop `fetch-density-data.sh` had at
a37ecb5, kept verbatim below, and that code must fail each one it was wrong on. Standard library only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.server
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

TOOLS = pathlib.Path(__file__).resolve().parent
UPDATE = dt.timedelta(hours=3)
SW_PATH = "/SpaceData/SW-All.csv"
MOVED_PATH = "/moved/SW-All.csv"
SW_V1 = b"DATE,BSRN,ND,KP1\n2026-11-23,2636,1,20\n"
SW_V2 = b"DATE,BSRN,ND,KP1\n2026-11-23,2636,1,20\n2026-11-24,2636,2,7\n"
ETAG = '"5b2d-6448e1c3"'
MODIFIED = "Mon, 23 Nov 2026 21:00:00 GMT"
TEXT_301 = "Legacy query: this URL moved in 2026 and re-requesting it will not change the answer."
TEXT_403 = (
    "You have already downloaded this file once in its current update. Please wait."
)
CLIENTS = {"fetch-density-data.sh", "fetch-supgp.sh", "fetch-satellite-js.sh"}

# What `tools/fetch-density-data.sh` ran for `SW-All.csv` at a37ecb5, lines 66-90 and 132-141 with
# the drivers array cut to CelesTrak's entry: the planted defect of rule 16.
OLD_SW_ALL = r"""
set -euo pipefail
DEST="${DENSITY_DIR:-data/validation/density}"
mkdir -p "$DEST"
fetch() {
  local url="$1" out="$2"
  local tmp
  tmp="$(mktemp)"
  if ! curl -sS -L --fail --max-time 900 -o "$tmp" "$url"; then
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$out"
}

manifest() {
  local path="$1" url="$2" extra="$3"
  local sha bytes
  sha="$(sha256sum "$path" | cut -d' ' -f1)"
  bytes="$(stat -c%s "$path")"
  cat >"${path}.json" <<EOF
{
  "url": "$url",
  "retrieved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "sha256": "$sha",
  "bytes": $bytes$extra
}
EOF
}
drivers=(
  "SW-All.csv|${SW_ALL_URL}|F10.7 and 3-hourly ap, CelesTrak"
)
for d in "${drivers[@]}"; do
  IFS='|' read -r name url what <<<"$d"
  if fetch "$url" "$DEST/$name"; then
    manifest "$DEST/$name" "$url" ",
  \"contents\": \"$what\""
    echo "$name  $(stat -c%s "$DEST/$name") bytes"
  else
    echo "could not fetch $name from $url" >&2
  fi
done
"""


@dataclass
class Answer:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)


Route = Callable[[dict[str, str]], Answer]


@dataclass
class Request:
    path: str
    headers: dict[str, str]
    body_bytes: int


class Provider:
    """CelesTrak, played on 127.0.0.1: each path answers through a route the scenario sets."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.log: list[Request] = []
        provider = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                route = provider.routes.get(path)
                answer = (
                    route(dict(self.headers)) if route else Answer(404, b"no route")
                )
                provider.log.append(
                    Request(self.path, dict(self.headers), len(answer.body))
                )
                self.send_response(answer.status)
                for key, value in answer.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(answer.body)))
                self.end_headers()
                self.wfile.write(answer.body)

            def log_message(self, *_: object) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take(self) -> list[Request]:
        log, self.log = self.log, []
        return log


def served(body: bytes) -> Route:
    """A file with validators, answering 304 to a request that carries either one of them."""

    def route(headers: dict[str, str]) -> Answer:
        validators = {"ETag": ETAG, "Last-Modified": MODIFIED}
        if (
            headers.get("If-None-Match") == ETAG
            or headers.get("If-Modified-Since") == MODIFIED
        ):
            return Answer(304, b"", {"ETag": ETAG})
        return Answer(200, body, validators)

    return route


def fixed(status: int, text: str, headers: dict[str, str] | None = None) -> Route:
    return lambda _: Answer(status, text.encode(), headers or {})


def run(
    argv: list[str], env: dict[str, str], scratch: pathlib.Path
) -> subprocess.CompletedProcess:
    # A bare environment: no proxy, no ~/.curlrc, and mktemp inside the scratch directory, so a
    # temporary file a script leaves behind is seen.
    base = {
        "PATH": os.environ["PATH"],
        "HOME": str(scratch),
        "TMPDIR": str(scratch),
        "LC_ALL": "C",
    }
    return subprocess.run(
        argv, env=base | env, capture_output=True, text=True, timeout=120, check=False
    )


def age_clock(dest: pathlib.Path, by: dt.timedelta) -> None:
    """Move the time of CelesTrak's last answer back by `by`, as if that much time had passed."""
    manifest = dest / "SW-All.csv.json"
    if not manifest.exists():
        return
    then = (dt.datetime.now(dt.UTC) - by).strftime("%Y-%m-%dT%H:%M:%SZ")
    text = re.sub(r'("checked_utc": ")[^"]*"', rf'\g<1>{then}"', manifest.read_text())
    manifest.write_text(text)


def leftovers(*dirs: pathlib.Path, keep: set[str]) -> list[str]:
    return sorted(
        str(p)
        for d in dirs
        if d.exists()
        for p in d.iterdir()
        if p.name not in keep and p.is_file()
    )


def sw_all_scenarios(
    target: Callable[[pathlib.Path, str], subprocess.CompletedProcess],
) -> dict[str, list[str]]:
    """Each scenario of the policy against one SW-All.csv client; the failures of each, by name."""
    provider = Provider()
    url = provider.base + SW_PATH
    failures: dict[str, list[str]] = {}
    keep = {"SW-All.csv", "SW-All.csv.json", "SW-All.csv.http"}
    with tempfile.TemporaryDirectory() as root:
        dest = pathlib.Path(root, "density")
        scratch = pathlib.Path(root, "scratch")
        scratch.mkdir()

        def scenario(name: str) -> list[str]:
            return failures.setdefault(name, [])

        def check(name: str, holds: bool, what: str) -> None:
            if not holds:
                scenario(name).append(what)

        sw = dest / "SW-All.csv"

        # 200: written once, from one unconditional request.
        provider.routes = {SW_PATH: served(SW_V1)}
        done = target(dest, url)
        log = provider.take()
        name = "200 writes the file once"
        scenario(name)
        check(
            name, done.returncode == 0, f"exit {done.returncode}: {done.stderr.strip()}"
        )
        check(
            name,
            sw.exists() and sw.read_bytes() == SW_V1,
            "the file is not the served body",
        )
        check(name, len(log) == 1, f"{len(log)} requests, not 1")
        check(
            name,
            leftovers(dest, scratch, keep=keep) == [],
            f"left {leftovers(dest, scratch, keep=keep)}",
        )
        written = sw.stat().st_mtime_ns if sw.exists() else 0

        # Inside one update: no request at all.
        age_clock(dest, UPDATE - dt.timedelta(minutes=1))
        done = target(dest, url)
        log = provider.take()
        name = "inside one three-hour update, no request"
        scenario(name)
        check(name, done.returncode == 0, f"exit {done.returncode}")
        check(name, log == [], f"{len(log)} requests inside the update")

        # After the update: a conditional request, answered 304, and the copy kept untouched.
        age_clock(dest, UPDATE + dt.timedelta(seconds=1))
        done = target(dest, url)
        log = provider.take()
        name = "after the update, conditional; 304 keeps the copy"
        scenario(name)
        check(
            name, done.returncode == 0, f"exit {done.returncode}: {done.stderr.strip()}"
        )
        check(name, len(log) == 1, f"{len(log)} requests, not 1")
        if log:
            check(
                name,
                log[0].headers.get("If-None-Match") == ETAG,
                f"If-None-Match {log[0].headers.get('If-None-Match')!r}",
            )
            check(
                name,
                log[0].headers.get("If-Modified-Since") == MODIFIED,
                f"If-Modified-Since {log[0].headers.get('If-Modified-Since')!r}",
            )
        check(
            name,
            sum(r.body_bytes for r in log) == 0,
            f"{sum(r.body_bytes for r in log)} body bytes downloaded again",
        )
        check(name, sw.exists() and sw.read_bytes() == SW_V1, "the copy changed")
        check(
            name,
            sw.exists() and sw.stat().st_mtime_ns == written,
            "the copy was rewritten",
        )
        check(
            name,
            leftovers(dest, scratch, keep=keep) == [],
            f"left {leftovers(dest, scratch, keep=keep)}",
        )

        # 301: stop, print the status, the Location and the message; follow nothing.
        age_clock(dest, UPDATE + dt.timedelta(seconds=1))
        provider.routes = {
            SW_PATH: fixed(301, TEXT_301, {"Location": provider.base + MOVED_PATH}),
            MOVED_PATH: served(SW_V2),
        }
        done = target(dest, url)
        log = provider.take()
        name = "301 stops with the provider's text"
        scenario(name)
        check(name, done.returncode != 0, "exit 0 on a 301")
        check(name, "301" in done.stderr, "the status is not printed")
        check(name, TEXT_301 in done.stderr, "the provider's message is not printed")
        check(
            name,
            provider.base + MOVED_PATH in done.stderr,
            "the Location is not printed",
        )
        check(
            name,
            [r.path for r in log] == [SW_PATH],
            f"requests {[r.path for r in log]}",
        )
        check(name, sw.exists() and sw.read_bytes() == SW_V1, "the copy changed")
        check(
            name,
            leftovers(dest, scratch, keep=keep) == [],
            f"left {leftovers(dest, scratch, keep=keep)}",
        )

        # 403: the same stop, the copy untouched.
        age_clock(dest, UPDATE + dt.timedelta(seconds=1))
        provider.routes = {SW_PATH: fixed(403, TEXT_403)}
        done = target(dest, url)
        log = provider.take()
        name = "403 stops with the provider's text"
        scenario(name)
        check(name, done.returncode != 0, "exit 0 on a 403")
        check(name, "403" in done.stderr, "the status is not printed")
        check(name, TEXT_403 in done.stderr, "the provider's message is not printed")
        check(name, len(log) == 1, f"{len(log)} requests, not 1")
        check(name, sw.exists() and sw.read_bytes() == SW_V1, "the copy changed")
        check(
            name,
            leftovers(dest, scratch, keep=keep) == [],
            f"left {leftovers(dest, scratch, keep=keep)}",
        )

        # 403 and a body that is not the CSV, with no copy yet: nothing is written at all.
        for status, text, name in (
            (403, TEXT_403, "403 with no copy writes nothing"),
            (200, "<html>maintenance</html>", "200 that is not SW-All writes nothing"),
        ):
            fresh = pathlib.Path(root, f"fresh-{status}")
            provider.routes = {SW_PATH: fixed(status, text)}
            done = target(fresh, url)
            provider.take()
            scenario(name)
            check(name, done.returncode != 0, f"exit 0 on {status} {text[:20]!r}")
            check(
                name,
                leftovers(fresh, scratch, keep=set()) == [],
                f"left {leftovers(fresh, scratch, keep=set())}",
            )
    provider.server.shutdown()
    return failures


def new_sw_all(dest: pathlib.Path, url: str) -> subprocess.CompletedProcess:
    return run(
        [str(TOOLS / "fetch-density-data.sh"), "--sw-all"],
        {"DENSITY_DIR": str(dest), "SW_ALL_URL": url},
        dest.parent / "scratch",
    )


def old_sw_all(dest: pathlib.Path, url: str) -> subprocess.CompletedProcess:
    return run(
        ["bash", "-c", OLD_SW_ALL],
        {"DENSITY_DIR": str(dest), "SW_ALL_URL": url},
        dest.parent / "scratch",
    )


def report(failures: dict[str, list[str]], red: str = "FAIL") -> None:
    for name, found in failures.items():
        print(
            f"  {'ok  ' if not found else red} {name}"
            + "".join(f"\n         {f}" for f in found)
        )


def test_sw_all_keeps_the_policy() -> None:
    failures = sw_all_scenarios(new_sw_all)
    report(failures)
    assert not any(failures.values()), (
        "fetch-density-data.sh --sw-all breaks CelesTrak's policy"
    )


def test_the_old_fetch_fails_what_it_was_wrong_on() -> None:
    failures = sw_all_scenarios(old_sw_all)
    report(failures, red="red, as planted:")
    wrong = {
        "inside one three-hour update, no request",
        "after the update, conditional; 304 keeps the copy",
        "301 stops with the provider's text",
        "403 stops with the provider's text",
        "403 with no copy writes nothing",
        "200 that is not SW-All writes nothing",
    }
    missed = sorted(name for name in wrong if not failures.get(name))
    assert missed == [], f"the scenarios no longer see the old defect in: {missed}"


def stops(argv: list[str], env: dict[str, str], route_path: str, label: str) -> None:
    """One more fetch path: a 301 and a 403 each stop it with the provider's text and no file."""
    for status, text in ((301, TEXT_301), (403, TEXT_403)):
        provider = Provider()
        headers = {"Location": provider.base + "/moved"} if status == 301 else {}
        provider.routes = {
            route_path: fixed(status, text, headers),
            "/moved": fixed(200, "x"),
        }
        with tempfile.TemporaryDirectory() as root:
            dest = pathlib.Path(root, "dest")
            scratch = pathlib.Path(root, "scratch")
            scratch.mkdir()
            filled = {
                k: v.replace("{base}", provider.base).replace("{dest}", str(dest))
                for k, v in env.items()
            }
            done = run(
                [a.replace("{base}", provider.base) for a in argv], filled, scratch
            )
            log = provider.take()
            assert done.returncode != 0, f"{label}: exit 0 on a {status}"
            assert str(status) in done.stderr and text in done.stderr, (
                f"{label} {status}: {done.stderr}"
            )
            if status == 301:
                assert provider.base + "/moved" in done.stderr, (
                    f"{label}: the Location is not printed"
                )
            assert [r.path.split("?")[0] for r in log] == [route_path], (
                f"{label} {status}: {log}"
            )
            assert leftovers(dest, scratch, keep=set()) == [], (
                f"{label} {status}: {leftovers(dest, scratch, keep=set())}"
            )
        provider.server.shutdown()
        print(f"  ok   {label}: {status} stops with the provider's text")


def test_supgp_stops_on_301_and_403() -> None:
    env = {
        "SUPGP_DIR": "{dest}",
        "SUPGP_URL": "{base}/sup-gp.php",
        "GP_URL": "{base}/gp.php",
    }
    stops(
        [str(TOOLS / "fetch-supgp.sh"), "GPS-A"], env, "/sup-gp.php", "fetch-supgp.sh"
    )


def test_supgp_still_fetches_on_200() -> None:
    provider = Provider()
    csv = "OBJECT_NAME,OBJECT_ID\nGPS BIIR-2,1997-035A\n"
    provider.routes = {"/sup-gp.php": fixed(200, csv), "/gp.php": fixed(200, csv)}
    with tempfile.TemporaryDirectory() as root:
        dest = pathlib.Path(root)
        env = {
            "SUPGP_DIR": str(dest),
            "SUPGP_URL": provider.base + "/sup-gp.php",
            "GP_URL": provider.base + "/gp.php",
        }
        done = run([str(TOOLS / "fetch-supgp.sh"), "GPS-A"], env, dest)
        assert done.returncode == 0, done.stderr
        assert (dest / "sup-GPS-A.csv").read_text() == csv
        assert (dest / "gp-active.csv").read_text() == csv
        assert sorted(p.name for p in dest.iterdir()) == [
            "gp-active.csv",
            "gp-active.csv.json",
            "sup-GPS-A.csv",
            "sup-GPS-A.csv.json",
        ], sorted(p.name for p in dest.iterdir())
    provider.server.shutdown()
    print("  ok   fetch-supgp.sh: 200 still writes both files")


def test_satellite_js_stops_on_301_and_403() -> None:
    url = "{base}/software/SGP4/satellite.js"
    stops(
        [str(TOOLS / "fetch-satellite-js.sh"), url],
        {"SATELLITE_JS_DIR": "{dest}"},
        "/software/SGP4/satellite.js",
        "fetch-satellite-js.sh",
    )


def test_satellite_js_still_fetches_on_200() -> None:
    provider = Provider()
    body = "/* satellite-js v6.0.1 */\n"
    provider.routes = {"/satellite.js": fixed(200, body)}
    with tempfile.TemporaryDirectory() as root:
        dest = pathlib.Path(root, "dest")
        env = {
            "SATELLITE_JS_DIR": str(dest),
            "SATELLITE_JS_SHA256": hashlib.sha256(body.encode()).hexdigest(),
        }
        done = run(
            [str(TOOLS / "fetch-satellite-js.sh"), provider.base + "/satellite.js"],
            env,
            pathlib.Path(root),
        )
        assert done.returncode == 0, done.stderr
        assert (dest / "satellite.js").read_text() == body
        assert sorted(p.name for p in dest.iterdir()) == [
            "manifest.json",
            "satellite.js",
        ]
    provider.server.shutdown()
    print("  ok   fetch-satellite-js.sh: 200 still writes the file")


def test_every_celestrak_fetch_goes_through_the_client() -> None:
    """A script that names celestrak.org outside a comment is one of the clients, and sources it."""
    naming = set()
    for path in sorted(TOOLS.rglob("*")):
        if path.suffix not in {".sh", ".py"} or path.name in {
            "celestrak.sh",
            pathlib.Path(__file__).name,
        }:
            continue
        code = [
            line
            for line in path.read_text(errors="replace").splitlines()
            if not line.lstrip().startswith("#")
        ]
        if any("celestrak.org" in line for line in code):
            naming.add(path.name)
    assert naming == CLIENTS, (
        f"names celestrak.org outside the client: {sorted(naming - CLIENTS)}"
    )
    for name in CLIENTS:
        text = (TOOLS / name).read_text()
        assert re.search(
            r'^source "\$\(dirname "\$\{BASH_SOURCE\[0\]\}"\)/celestrak\.sh"$',
            text,
            re.M,
        ), name
    print("  ok   every celestrak.org fetch in tools/ sources tools/celestrak.sh")


if __name__ == "__main__":
    red = []
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            try:
                test()
            except AssertionError as error:
                print(f"  FAIL {error}")
                red.append(name)
    print(f"{len(red)} red: {red}" if red else "all green")
    sys.exit(1 if red else 0)
