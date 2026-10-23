#!/usr/bin/env -S uv run --quiet --with mpmath python
"""High-precision oracle for the lunar gravity field, against the Gottlieb recursion in Rust.

The Rust evaluator sums Gottlieb's solid harmonics `V/W` and differentiates them in closed form. This
oracle does neither: it evaluates the potential directly as

    U(r) = (mu/r) * sum_n (Re/r)^n * sum_m Pbar_nm(sin phi) * [Cbar_nm cos(m lam) + Sbar_nm sin(m lam)]

with 60-digit arithmetic and takes the gradient by central differences. Same coefficients, different
algebra, different arithmetic -- so a mistake in the normalisation ratios, the degree-mixing, or the
Cartesian gradient of the Rust path has nowhere to hide.

What it does NOT independently check is the Legendre recursion coefficients themselves, which are the
same textbook `a_nm`/`b_nm` in both. Those are already pinned by the Earth field's oracle comparisons;
what is new here is the lunar constants (`GM`, `Re`) and the truncation.

The step size is the honest part of a numerical gradient. At 60 digits with h = 1e-6 m over a
1.8e6 m radius, the central-difference truncation error is O(h^2 * U''') ~ 1e-12 relative and the
cancellation error is ~1e-60/h -- so the answer is good to about 12 digits, which is stated in the
emitted header rather than implied.

Run:  timeout 600 uv run --no-project --with mpmath python derivations/gen_lunar_oracle.py
"""

from __future__ import annotations

import pathlib
import re
import sys

from mpmath import mp, mpf, sqrt

SRC = (
    pathlib.Path(__file__).parent.parent
    / "crates/spacelab-propagation/src/lunar_coeffs.rs"
)
OUT = (
    pathlib.Path(__file__).parent.parent
    / "crates/spacelab-propagation/tests/fixtures/lunar_gravity_oracle.txt"
)

DEGREE = 20
# Points chosen to exercise the field rather than to be convenient: a low polar pass, a low equatorial
# one over the near side, a mid-latitude point at a longitude where the tesseral terms are large, and
# one far out where the point mass has to dominate. Metres, principal-axis frame.
POINTS = [
    (1.8e6, 0.0, 0.0),
    (0.0, 0.0, 1.85e6),
    (1.2e6, 9.0e5, 1.0e6),
    (-1.5e6, 4.0e5, -8.0e5),
    (2.5e7, -1.1e7, 6.0e6),
]


def load_model() -> tuple[mpf, mpf, dict[tuple[int, int], tuple[mpf, mpf]]]:
    """Read the generated Rust table -- the same literals the Rust build compiles."""
    text = SRC.read_text()
    gm = mpf(re.search(r"MODEL_GM: f64 = ([0-9.eE+-]+);", text).group(1))
    re_ = mpf(re.search(r"MODEL_RE: f64 = ([0-9.eE+-]+);", text).group(1))
    coeffs: dict[tuple[int, int], tuple[mpf, mpf]] = {(0, 0): (mpf(1), mpf(0))}
    for n_s, m_s, c_s, s_s in re.findall(
        r"\((\d+), (\d+), ([0-9.eE+-]+), ([0-9.eE+-]+)\),", text
    ):
        coeffs[(int(n_s), int(m_s))] = (mpf(c_s), mpf(s_s))
    return gm, re_, coeffs


def legendre_normalized(degree: int, t: mpf) -> dict[tuple[int, int], mpf]:
    """Fully normalized (geodesy 4pi) associated Legendre functions at sin(phi) = t."""
    u = sqrt(1 - t * t)
    p: dict[tuple[int, int], mpf] = {
        (0, 0): mpf(1),
        (1, 0): sqrt(3) * t,
        (1, 1): sqrt(3) * u,
    }
    for n in range(2, degree + 1):
        p[(n, n)] = sqrt(mpf(2 * n + 1) / mpf(2 * n)) * u * p[(n - 1, n - 1)]
        for m in range(n):
            a = sqrt(mpf((2 * n - 1) * (2 * n + 1)) / mpf((n - m) * (n + m)))
            term = a * t * p[(n - 1, m)]
            if n - 2 >= m:
                b = sqrt(
                    mpf((2 * n + 1) * (n + m - 1) * (n - m - 1))
                    / mpf((n - m) * (n + m) * (2 * n - 3))
                )
                term -= b * p[(n - 2, m)]
            p[(n, m)] = term
    return p


def potential(
    xyz: tuple[mpf, mpf, mpf],
    gm: mpf,
    ref: mpf,
    coeffs: dict[tuple[int, int], tuple[mpf, mpf]],
    degree: int,
) -> mpf:
    x, y, z = xyz
    r = sqrt(x * x + y * y + z * z)
    p = legendre_normalized(degree, z / r)
    # cos(m lam) and sin(m lam) by Chebyshev recursion on (x/rho, y/rho) -- no atan2, so the branch
    # cut never enters.
    rho = sqrt(x * x + y * y)
    cl, sl = (x / rho, y / rho) if rho > 0 else (mpf(1), mpf(0))
    cm, sm = [mpf(1)], [mpf(0)]
    for _ in range(degree):
        cm.append(cm[-1] * cl - sm[-1] * sl)
        sm.append(sm[-1] * cl + cm[-2] * sl)
    total = mpf(0)
    ratio = mpf(1)
    for n in range(degree + 1):
        inner = mpf(0)
        for m in range(n + 1):
            c, s = coeffs.get((n, m), (mpf(0), mpf(0)))
            inner += p[(n, m)] * (c * cm[m] + s * sm[m])
        total += ratio * inner
        ratio *= ref / r
    return gm / r * total


def main() -> int:
    mp.dps = 60
    gm, ref, coeffs = load_model()
    h = mpf("1e-6")
    rows = []
    for point in POINTS:
        base = tuple(mpf(v) for v in point)
        acc = []
        for axis in range(3):
            plus = list(base)
            minus = list(base)
            plus[axis] += h
            minus[axis] -= h
            up = potential(tuple(plus), gm, ref, coeffs, DEGREE)
            down = potential(tuple(minus), gm, ref, coeffs, DEGREE)
            acc.append((up - down) / (2 * h))
        magnitude = sqrt(sum(a * a for a in acc))
        print(f"  {point} -> |a| = {mp.nstr(magnitude, 12)} m/s^2")
        rows.append((point, acc))

    lines = [
        "# Lunar gravity oracle: GRGM1200A truncated to degree "
        f"{DEGREE}, principal-axis frame.",
        "#",
        "# GENERATED by derivations/gen_lunar_oracle.py. The accelerations are the gradient of the",
        "# potential summed directly in the Legendre form at 60 decimal digits, differenced centrally",
        "# with h = 1e-6 m -- independent of the Gottlieb solid-harmonic recursion the crate uses.",
        "# Accurate to about 12 significant figures, the central-difference floor at this step.",
        "#",
        "# x y z ax ay az   (metres, m/s^2)",
    ]
    for point, acc in rows:
        values = " ".join(f"{v:.1f}" for v in point)
        grads = " ".join(mp.nstr(a, 17) for a in acc)
        lines.append(f"{values} {grads}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"  wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
