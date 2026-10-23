#!/usr/bin/env python3
"""Goldstein branch-cut phase unwrapping, and the fixture search behind insar.rs's test.

Written from the algorithm description (Goldstein 1988) rather than from the Rust: residues on the
dual grid; greedy nearest-neighbour (L1) pairing of opposite charges, unmatched ones grounded to the
left border; Bresenham cuts; breadth-first Itoh integration from the first pixel not on a cut, never
stepping onto one; cut pixels filled at the end from any already integrated 4-neighbour.

`emit` prints the wrapped field and its answer as Rust consts, so the test embeds both and neither
side re-evaluates a transcendental.

`search` is why this file is more than a transcription. A field that merely unwraps correctly leaves
fourteen mutations of the machinery alive: with one negative residue the pairing metric and its
tie-break are unreachable, with no cut at the origin the seed search is unreachable, and with every
pixel reachable by a down-and-right walk the up-step, the left-step and three of the four fill
directions are unreachable too. MUTANTS lists those fourteen; `search` hill-climbs over dislocation
layouts until the answer changes under every one of them.
"""

from __future__ import annotations

import math
import random
import sys

PI = math.pi
TAU = 2.0 * math.pi

# Each is one cargo-mutants survivor of `unwrap_phase_2d_branch_cut`, keyed by its file:line:col.
MUTANTS = {
    "pair_metric": "332:39 replace + with * -- L1 distance becomes a product",
    "pair_tie": "333:18 replace < with <= -- keep the last nearest rather than the first",
    "ground_used": "346:12 delete ! -- ground the paired negatives instead of the odd ones",
    "seed_add": "353:25 replace * with + -- search rows+cols pixels for the seed, not rows*cols",
    "seed_div": "353:25 replace * with / -- search rows/cols pixels for the seed",
    "up_skip": "371:14 replace > with < -- never step up",
    "up_self": "372:20 replace - with / -- the up-step addresses its own pixel",
    "left_skip": "377:14 replace > with < -- never step left",
    "left_self": "378:23 replace - with / -- the left-step addresses its own pixel",
    "fill_down_self": "393:20 replace + with * -- the fill's down-neighbour is the pixel itself",
    "fill_right_self": "395:23 replace + with * -- the fill's right-neighbour is the pixel itself",
    "fill_row_le": "397:23 replace < with <= -- the fill reads one row past the grid",
    "fill_col_le": "397:36 replace < with <= -- the fill reads one column past the grid",
    "fill_index_div": "397:57 replace * with / -- the fill's guard reads the wrong pixel",
}


def wrap_phase(phi: float) -> float:
    x = math.fmod(phi + PI, TAU)
    if x <= 0.0:
        x += TAU
    return x - PI


def round_half_away(x: float) -> float:
    return math.copysign(math.floor(abs(x) + 0.5), x)


def residues(w: list[float], rows: int, cols: int) -> list[int]:
    if rows < 2 or cols < 2:
        return []
    res = [0] * ((rows - 1) * (cols - 1))
    for r in range(rows - 1):
        for c in range(cols - 1):
            p00, p01 = w[r * cols + c], w[r * cols + c + 1]
            p10, p11 = w[(r + 1) * cols + c], w[(r + 1) * cols + c + 1]
            s = (
                wrap_phase(p01 - p00)
                + wrap_phase(p11 - p01)
                + wrap_phase(p10 - p11)
                + wrap_phase(p00 - p10)
            )
            res[r * (cols - 1) + c] = int(round_half_away(s / TAU))
    return res


def draw_cut(
    barrier: list[bool], cols: int, a: tuple[int, int], b: tuple[int, int]
) -> None:
    r, c = a
    tr, tc = b
    dr, dc = abs(tr - r), -abs(tc - c)
    sr = 1 if r < tr else -1
    sc = 1 if c < tc else -1
    err = dr + dc
    while True:
        barrier[r * cols + c] = True
        if r == tr and c == tc:
            return
        e2 = 2 * err
        if e2 >= dc:
            err += dc
            r += sr
        if e2 <= dr:
            err += dr
            c += sc


def unwrap_branch_cut(
    w: list[float], rows: int, cols: int, mutant: str | None = None
) -> tuple[list[float], list[bool], list[int]]:
    """The Rust function, optionally with one of MUTANTS applied. Raises IndexError where it panics."""
    out = [0.0] * (rows * cols)
    if rows < 2 or cols < 2:
        return out, [], []
    res = residues(w, rows, cols)
    pos, neg = [], []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = res[r * (cols - 1) + c]
            if v > 0:
                pos.append((r, c))
            elif v < 0:
                neg.append((r, c))

    barrier = [False] * (rows * cols)
    used = [False] * len(neg)
    for p in pos:
        best, best_d = None, None
        for k, q in enumerate(neg):
            if used[k]:
                continue
            dr, dc = abs(p[0] - q[0]), abs(p[1] - q[1])
            d = dr * dc if mutant == "pair_metric" else dr + dc
            if best_d is None or (d <= best_d if mutant == "pair_tie" else d < best_d):
                best_d, best = d, k
        if best is not None:
            used[best] = True
            draw_cut(barrier, cols, p, neg[best])
        else:
            draw_cut(barrier, cols, p, (p[0], 0))
    for k, q in enumerate(neg):
        if used[k] if mutant == "ground_used" else not used[k]:
            draw_cut(barrier, cols, q, (q[0], 0))

    span = rows * cols
    if mutant == "seed_add":
        span = rows + cols
    elif mutant == "seed_div":
        span = rows // cols
    visited = [False] * (rows * cols)
    seed = next((i for i in range(span) if not barrier[i]), 0)
    out[seed] = w[seed]
    visited[seed] = True
    queue = [seed]
    head = 0
    while head < len(queue):
        i = queue[head]
        head += 1
        r, c = divmod(i, cols)

        def step(nr: int, nc: int, here: int = i) -> None:
            ni = nr * cols + nc
            if not visited[ni] and not barrier[ni]:
                out[ni] = out[here] + wrap_phase(w[ni] - w[here])
                visited[ni] = True
                queue.append(ni)

        if r > 0 and mutant != "up_skip":
            step(r if mutant == "up_self" else r - 1, c)
        if r + 1 < rows:
            step(r + 1, c)
        if c > 0 and mutant != "left_skip":
            step(r, c if mutant == "left_self" else c - 1)
        if c + 1 < cols:
            step(r, c + 1)

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if visited[i]:
                continue
            down = (r, c) if mutant == "fill_down_self" else (r + 1, c)
            right = (r, c) if mutant == "fill_right_self" else (r, c + 1)
            row_hi = rows + 1 if mutant == "fill_row_le" else rows
            col_hi = cols + 1 if mutant == "fill_col_le" else cols
            for nr, nc in ((r - 1, c), down, (r, c - 1), right):
                if not (0 <= nr < row_hi and 0 <= nc < col_hi):
                    continue
                probe = nr // cols if mutant == "fill_index_div" else nr * cols
                if visited[probe + nc]:
                    ni = nr * cols + nc
                    out[i] = out[ni] + wrap_phase(w[i] - w[ni])
                    visited[i] = True
                    break
    return out, barrier, res


Layout = tuple[tuple[int, int, int], ...]


def field(
    rows: int, cols: int, layout: Layout, ramp: tuple[float, float]
) -> list[float]:
    """The wrapped phase of a ramp plus one screw dislocation per (row, col, charge) in `layout`."""

    def phase(r: int, c: int) -> float:
        s = ramp[0] * r + ramp[1] * c
        for yr, xc, sign in layout:
            s += sign * math.atan2(r - (yr + 0.5), c - (xc + 0.5))
        return s

    return [wrap_phase(phase(r, c)) for r in range(rows) for c in range(cols)]


def killed(w: list[float], rows: int, cols: int) -> set[str]:
    """The mutants whose answer differs from the reference -- a panic counts as a difference."""
    truth, _, _ = unwrap_branch_cut(w, rows, cols)
    reference = tuple(repr(v) for v in truth)
    out = set()
    for name in MUTANTS:
        try:
            got, _, _ = unwrap_branch_cut(w, rows, cols, name)
        except IndexError:
            out.add(name)
            continue
        if tuple(repr(v) for v in got) != reference:
            out.add(name)
    return out


def search(
    rows: int,
    cols: int,
    ramp: tuple[float, float],
    seed: int,
    target: set[str],
    rare: set[str] = frozenset(),
) -> tuple[Layout, set[str]]:
    """Hill-climb over dislocation layouts for the one killing most of `target`.

    A layout killing a mutant in `rare` scores far above one that kills more of the rest, because a
    mutant only a handful of layouts in a quarter-million reach is not something a wider search finds
    again once the climb has wandered off it.
    """
    rng = random.Random(seed)
    sites = [(r, c) for r in range(rows - 1) for c in range(cols - 1)]
    best_layout: Layout = ()
    best_killed: set[str] = set()
    reached: dict[str, int] = dict.fromkeys(MUTANTS, 0)

    def score(hit: set[str]) -> int:
        return len(hit) + 50 * len(hit & rare)

    for _ in range(400):
        layout = tuple(
            (r, c, rng.choice((-1, 1))) for r, c in rng.sample(sites, rng.randint(4, 8))
        )
        hit = killed(field(rows, cols, layout, ramp), rows, cols) & target
        for _ in range(600):
            if score(hit) > score(best_killed):
                best_layout, best_killed = layout, set(hit)
            for name in hit:
                reached[name] += 1
            if len(best_killed) == len(target):
                return best_layout, best_killed
            cand = list(layout)
            move = rng.randrange(3)
            if move == 0 or len(cand) <= 4:
                r, c = rng.choice(sites)
                cand[rng.randrange(len(cand))] = (r, c, rng.choice((-1, 1)))
            elif move == 1 and len(cand) < 9:
                r, c = rng.choice(sites)
                cand.append((r, c, rng.choice((-1, 1))))
            else:
                cand.pop(rng.randrange(len(cand)))
            if len({(r, c) for r, c, _ in cand}) != len(cand):
                continue
            trial = killed(field(rows, cols, tuple(cand), ramp), rows, cols) & target
            if score(trial) >= score(hit):
                layout, hit = tuple(cand), trial
    for name, n in sorted(reached.items(), key=lambda kv: kv[1]):
        print(f"    reached {n:7}  {name}: {MUTANTS[name]}", file=sys.stderr)
    return best_layout, best_killed


def emit(
    rows: int, cols: int, layout: Layout, ramp: tuple[float, float], tag: str
) -> None:
    w = field(rows, cols, layout, ramp)
    answer, barrier, res = unwrap_branch_cut(w, rows, cols)
    hit = killed(w, rows, cols)
    print(f"// {rows}x{cols} layout {layout} ramp {ramp}", file=sys.stderr)
    print(
        f"// {sum(v > 0 for v in res)} positive and {sum(v < 0 for v in res)} negative residues,"
        f" {sum(barrier)} barrier pixels, seed {barrier.index(False)},"
        f" {sum(not v for v in barrier)} integrated",
        file=sys.stderr,
    )
    print(f"// kills: {sorted(hit)}", file=sys.stderr)
    for name, values in ((f"{tag}_WRAPPED", w), (f"{tag}_EXPECTED", answer)):
        print(f"    const {name}: [f64; {rows * cols}] = [")
        for r in range(rows):
            print(
                "        " + " ".join(f"{values[r * cols + c]!r}," for c in range(cols))
            )
        print("    ];")


RAMP = (0.31, 0.17)

# Two fixtures, because one field cannot carry both halves of the requirement. `RICH` needs the flood
# fill to reach nearly every pixel, so that the up-step, the left-step and three of the four fill
# directions each own a pixel nothing else reaches. `POCKET` needs the opposite: the seed walled into
# a corner, so most of the grid is left to the fill loop and the bottom row is still unfilled when the
# loop reaches it -- which is the only way the row test's `<=` form ever indexes past the grid.
# seed_add and fill_col_le are in neither, because no field reaches them; the proofs are in
# .cargo/mutants.toml.
FIXTURES = {
    "RICH": (
        8,
        8,
        (
            (4, 6, -1),
            (0, 0, 1),
            (1, 6, 1),
            (1, 1, 1),
            (0, 3, -1),
            (5, 3, -1),
            (3, 0, -1),
            (6, 5, 1),
            (6, 2, -1),
        ),
    ),
    "POCKET": (
        8,
        8,
        (
            (2, 6, -1),
            (4, 3, 1),
            (3, 1, -1),
            (0, 3, -1),
            (1, 1, -1),
            (2, 4, -1),
            (2, 1, 1),
            (1, 5, 1),
            (0, 6, -1),
        ),
    ),
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        rows, cols = (int(v) for v in sys.argv[2].split("x"))
        rare = set(sys.argv[3:])
        want = set(MUTANTS) - {"seed_add", "fill_col_le"}
        found, hit = search(rows, cols, RAMP, 20260807, want, rare)
        print(f"// killed {len(hit)}/{len(want)}: {sorted(hit)}", file=sys.stderr)
        emit(rows, cols, found, RAMP, "FOUND")
    else:
        for tag, (rows, cols, layout) in FIXTURES.items():
            emit(rows, cols, layout, RAMP, tag)
