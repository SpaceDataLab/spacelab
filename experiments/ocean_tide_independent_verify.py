"""Independent verification of the FES2004 ocean-tide ΔC̄nm/ΔS̄nm (IERS 2010 Eq. 6.15).

A from-scratch numpy re-implementation that shares NO code with the Rust port or its generated table:
it parses the raw `fes2004_Cnm-Snm.dat` itself, re-derives the Doodson->Delaunay multipliers, and sums
Eq. 6.15 with Orekit's own emitted GMST + Delaunay arguments. It then checks that result against the
ΔC̄/ΔS̄ the Rust port reproduces bit-for-bit from Orekit (`OREKIT_DCS` below -- exactly the values the
`ocean_tide::tests::delta_cs_matches_orekit` test asserts, so numpy == Orekit == Rust port).

Eq. 6.15, per wave, for each degree n >= 2 and order m <= n:
    ΔC̄nm += (C⁺ + C⁻)·cos θ + (S⁺ + S⁻)·sin θ
    ΔS̄nm += (S⁺ − S⁻)·cos θ − (C⁺ − C⁻)·sin θ
with θ = c_γ·(GMST+π) + c_l·l + c_l′·l′ + c_F·F + c_D·D + c_Ω·Ω, the Delaunay multipliers coming from
the Doodson number d via c_γ = c_τ, c_l = −c_p, c_l′ = −c_ps, c_F = −c_τ+c_s+c_h+c_p+c_ps,
c_D = −c_h−c_ps, c_Ω = −c_τ+c_s+c_h+c_p−c_n′+c_ps (IERS Conventions 2010 §6.3).

    uv run --no-project python experiments/ocean_tide_independent_verify.py [path/to/fes2004_Cnm-Snm.dat]
"""

from __future__ import annotations

import sys

import numpy as np

# FES2004 coefficients are tabulated in units of 10^-11 (file header; Orekit FESCnmSnmReader scale).
SCALE = 1.0e-11
MAX_DEGREE = 6
DEFAULT_DAT = "/tmp/orekit-bench/orekit-data-main/fes2004_Cnm-Snm.dat"

# Orekit's own values at 2024-01-01T00:00:00 UTC (tools/orekit-oracle -> orekit_ocean_tide.csv).
GMST = 1.58827626212885320e02
DELAUNAY = np.array([  # l, l', F, D, Omega  (radians)
    2.00112767583961430e03,
    1.57023942002898050e02,
    2.02554451620882600e03,
    1.87022276577018720e03,
    -5.91878837663082800e00,
])
# ΔC̄nm, ΔS̄nm keyed by (n, m) -- what the Rust port reproduces bit-for-bit.
OREKIT_DCS = {
    (2, 0): (8.66613090676338700e-11, -5.77956367038604860e-11),
    (2, 1): (-3.58949120317954750e-11, 3.21567019225667160e-11),
    (2, 2): (-2.36808947482612250e-10, -4.07525148682983650e-10),
    (3, 0): (1.63235886979017050e-10, 1.21402485974879260e-10),
    (3, 1): (-5.76705446519916100e-10, -8.38219496737743700e-11),
    (3, 2): (4.44967062669770100e-11, -2.56070860162123820e-11),
    (3, 3): (-6.18601228399051600e-10, 1.35709790371800830e-11),
    (4, 0): (-7.74756821510940900e-11, -7.08801068334570900e-11),
    (4, 1): (-5.64833347015401900e-10, -3.90461432273574200e-11),
    (4, 2): (1.63279777500894770e-10, 3.46141388456912350e-10),
    (4, 3): (2.42684484155221950e-10, 2.86380562771294630e-10),
    (4, 4): (9.35405423862152400e-11, -3.65889367127821340e-10),
    (5, 0): (-9.08857701907423500e-11, -6.65554371038115000e-11),
    (5, 1): (5.69828729637418100e-10, -1.19445493612052600e-10),
    (5, 2): (7.24413110210773400e-11, 2.98800119430674860e-10),
    (5, 3): (-6.14825563043263000e-11, -2.36171448526757850e-10),
    (5, 4): (-2.74289015400280660e-10, 2.72315101609209970e-10),
    (5, 5): (1.82921179385129340e-10, -2.22620150290379120e-10),
    (6, 0): (-3.92468243582551440e-11, 1.06891938587308280e-10),
    (6, 1): (8.29136620575438900e-11, -8.61270489422525400e-12),
    (6, 2): (-1.48214916865007900e-10, -1.12871705590579910e-10),
    (6, 3): (-1.35387189555665600e-10, 4.47037390005829540e-11),
    (6, 4): (-3.06536122494912500e-11, 5.50494139703270450e-11),
    (6, 5): (-3.05242549791334700e-11, -1.82726913994146440e-10),
    (6, 6): (-3.57398100596411200e-10, 2.64639622446727300e-10),
}


def parse_dat(path: str) -> dict[int, list[tuple[int, int, float, float, float, float]]]:
    """Group {doodson: [(n, m, C+, S+, C-, S-)]}, degree n = 2..6, order m <= n. Independent parse."""
    waves: dict[int, list[tuple[int, int, float, float, float, float]]] = {}
    with open(path) as fh:
        for lineno, line in enumerate(fh):
            if lineno < 4:  # four header lines
                continue
            t = line.split()
            if len(t) < 8:
                continue
            try:
                doodson = int(t[0].replace(".", "").replace(",", ""))
                n, m = int(t[2]), int(t[3])
                cp, sp, cm, sm = (float(x) for x in t[4:8])
            except ValueError:
                continue
            if 2 <= n <= MAX_DEGREE and 0 <= m <= n:
                waves.setdefault(doodson, []).append((n, m, cp, sp, cm, sm))
    return waves


def theta(doodson: int) -> float:
    """Doodson number -> tidal argument θ, via the IERS 2010 Doodson->Delaunay multipliers."""
    c_ps = (doodson % 10) - 5
    c_n_prime = (doodson // 10) % 10 - 5
    c_p = (doodson // 100) % 10 - 5
    c_h = (doodson // 1000) % 10 - 5
    c_s = (doodson // 10000) % 10 - 5
    c_tau = (doodson // 100000) % 10
    c_gamma = c_tau
    c_l = -c_p
    c_l_prime = -c_ps
    c_f = -c_tau + c_s + c_h + c_p + c_ps
    c_d = -c_h - c_ps
    c_omega = -c_tau + c_s + c_h + c_p - c_n_prime + c_ps
    gamma = GMST + np.pi
    return (c_gamma * gamma + c_l * DELAUNAY[0] + c_l_prime * DELAUNAY[1]
            + c_f * DELAUNAY[2] + c_d * DELAUNAY[3] + c_omega * DELAUNAY[4])


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DAT
    waves = parse_dat(path)
    dcs: dict[tuple[int, int], np.ndarray] = {nm: np.zeros(2) for nm in OREKIT_DCS}
    for doodson, rows in waves.items():
        cos, sin = np.cos(theta(doodson)), np.sin(theta(doodson))
        for n, m, cp, sp, cm, sm in rows:
            cp, sp, cm, sm = (v * SCALE for v in (cp, sp, cm, sm))
            dcs[(n, m)] += [
                (cp + cm) * cos + (sp + sm) * sin,
                (sp - sm) * cos - (cp - cm) * sin,
            ]

    max_rel = 0.0
    for (n, m), (want_c, want_s) in sorted(OREKIT_DCS.items()):
        got_c, got_s = dcs[(n, m)]
        rel_c = abs(got_c - want_c) / abs(want_c)
        rel_s = abs(got_s - want_s) / abs(want_s)
        max_rel = max(max_rel, rel_c, rel_s)
        print(f"    ({n},{m})  ΔC̄ num {got_c:+.6e} port {want_c:+.6e} rel {rel_c:.2e}   "
              f"ΔS̄ num {got_s:+.6e} port {want_s:+.6e} rel {rel_s:.2e}")
    print(f"\n{len(waves)} waves parsed independently; max relative error "
          f"(numpy Eq. 6.15 vs Rust port / Orekit): {max_rel:.3e}")


if __name__ == "__main__":
    main()
