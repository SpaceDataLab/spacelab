"""Convex hulls and Delaunay triangulations from qhull, as an oracle for `spacelab-geometry`.

qhull (Barber, Dobkin & Huhdanpaa, 1996) is the reference implementation of the quickhull family and
shares no code and no arithmetic with this workspace: it works in floating point with facet merging
and a joggle fallback, where `spacelab-geometry` decides every sign exactly by expansion arithmetic.
Two implementations that reach the same combinatorial answer by different means is the comparison
worth making; the crate's own tests already check the defining predicates.

qhull is an **oracle, never a dependency**. Its licence is permissive but bespoke — copyright notices
kept intact, the licence text redistributed with any copy, modifications annotated with name, date and
reason — which is fine for a cross-check harness run by hand and needless friction inside a shipped
library.

    uv run python tools/gen_qhull_oracle.py

writes `crates/spacelab-geometry/tests/fixtures/qhull_oracle.txt`, which
`crates/spacelab-geometry/tests/qhull_cross_check.rs` reads. Coordinates are written with 17
significant digits, so both codes see bit-identical doubles and no residual is a parsing artefact.

Which answer is compared depends on whether the input has one. For points in general position the
Delaunay triangulation is unique and the triangle sets must match exactly. On a regular grid every
cell is cocircular, and both codes produce *a* valid triangulation of it — there the invariants are
compared instead: the triangle count, which Euler's formula fixes, and the hull.
"""

import math
import random
import subprocess
from pathlib import Path

FIXTURE = Path("crates/spacelab-geometry/tests/fixtures/qhull_oracle.txt")


def run_qhull(
    program: str, options: list[str], dimension: int, points: list[list[float]]
) -> str:
    """Feed points to a qhull program on stdin and return its stdout."""
    payload = [str(dimension), str(len(points))]
    payload += [" ".join(f"{c:.17g}" for c in p) for p in points]
    # A fixed argv and no shell: `program` and `options` are literals from this file.
    finished = subprocess.run(
        [program, *options],
        input="\n".join(payload) + "\n",
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return finished.stdout


def extreme_points(dimension: int, points: list[list[float]]) -> list[int]:
    """`qconvex Fx`: the indices of the extreme points, one per line after a count."""
    lines = run_qhull("qconvex", ["Fx"], dimension, points).split()
    count = int(lines[0])
    return sorted(int(value) for value in lines[1 : 1 + count])


def facets(dimension: int, points: list[list[float]]) -> list[tuple[int, ...]]:
    """`qconvex Qt i`: the hull's simplicial facets as point indices."""
    lines = run_qhull("qconvex", ["Qt", "i"], dimension, points).splitlines()
    return sorted(
        tuple(sorted(int(v) for v in line.split()))
        for line in lines[1:]
        if line.strip()
    )


def delaunay(points: list[list[float]]) -> list[tuple[int, ...]]:
    """`qdelaunay Qt i`: the Delaunay triangles as point indices."""
    lines = run_qhull("qdelaunay", ["Qt", "i"], 2, points).splitlines()
    return sorted(
        tuple(sorted(int(v) for v in line.split()))
        for line in lines[1:]
        if line.strip()
    )


def scattered(count: int, dimension: int, seed: int) -> list[list[float]]:
    """Points in general position: no two equal, no three collinear, no four cocircular."""
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(dimension)] for _ in range(count)]


def grid(side: int, dimension: int) -> list[list[float]]:
    if dimension == 2:
        return [[float(x), float(y)] for x in range(side) for y in range(side)]
    return [
        [float(x), float(y), float(z)]
        for x in range(side)
        for y in range(side)
        for z in range(side)
    ]


def fibonacci_sphere(count: int) -> list[list[float]]:
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(1.0 - z * z)
        theta = golden * index
        out.append([radius * math.cos(theta), radius * math.sin(theta), z])
    return out


def with_duplicates(points: list[list[float]], seed: int) -> list[list[float]]:
    """Every fourth point repeated, in place, so the duplicate is not adjacent to its original."""
    rng = random.Random(seed)
    out = list(points)
    for _ in range(len(points) // 4):
        out.insert(rng.randrange(len(out)), list(rng.choice(points)))
    return out


def emit(
    handle, name: str, kind: str, dimension: int, points: list[list[float]], **sections
):
    handle.write(f"case {name} {kind}\n")
    handle.write(f"points {dimension} {len(points)}\n")
    for p in points:
        handle.write(" ".join(f"{c:.17g}" for c in p) + "\n")
    for key, value in sections.items():
        if isinstance(value, int):
            handle.write(f"{key} {value}\n")
        elif value and isinstance(value[0], tuple):
            handle.write(f"{key} {len(value)}\n")
            for row in value:
                handle.write(" ".join(str(v) for v in row) + "\n")
        else:
            handle.write(f"{key} {len(value)}\n")
            handle.write(" ".join(str(v) for v in value) + "\n")
    handle.write("end\n\n")


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w") as handle:
        handle.write(
            "# qhull cross-check fixture -- generated by tools/gen_qhull_oracle.py\n"
        )
        handle.write(
            f"# qhull version: {run_qhull('qconvex', ['-V'], 2, [[0, 0]]).strip()}\n\n"
        )

        planar = scattered(80, 2, 20_260_814)
        emit(
            handle,
            "hull2_scattered",
            "hull2",
            2,
            planar,
            vertices=extreme_points(2, planar),
        )

        planar_grid = grid(6, 2)
        emit(
            handle,
            "hull2_grid",
            "hull2",
            2,
            planar_grid,
            vertices=extreme_points(2, planar_grid),
        )

        emit(
            handle,
            "delaunay2_scattered",
            "delaunay2",
            2,
            planar,
            triangles=delaunay(planar),
        )

        emit(
            handle,
            "delaunay2_grid",
            "delaunay2_count",
            2,
            planar_grid,
            triangle_count=len(delaunay(planar_grid)),
            vertices=extreme_points(2, planar_grid),
        )

        spatial = scattered(60, 3, 20_260_815)
        emit(
            handle,
            "hull3_scattered",
            "hull3",
            3,
            spatial,
            vertices=extreme_points(3, spatial),
            facets=facets(3, spatial),
        )

        cube = grid(3, 3)
        emit(
            handle,
            "hull3_cube",
            "hull3_vertices",
            3,
            cube,
            vertices=extreme_points(3, cube),
        )

        repeated = with_duplicates(spatial[:30], 20_260_816)
        emit(
            handle,
            "hull3_duplicates",
            "hull3_vertices",
            3,
            repeated,
            vertices=extreme_points(3, repeated),
        )

        sphere = fibonacci_sphere(150)
        emit(
            handle,
            "sphere_fibonacci",
            "hull3",
            3,
            sphere,
            vertices=extreme_points(3, sphere),
            facets=facets(3, sphere),
        )
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
