# 2. The lattice configuration number is Walker's F, not the literature's N_c′

*Status:* accepted, 2026-10-26 (`plans/17` Ф17.B batch 1). *Supersedes:* nothing.

## Context

`LatticeFlower` places satellite `(i, j)` at `Ω = Ω₀ + 2π·i/No`,
`M = 2π·(Nc·i/(No·Nso) + j/Nso)`. The 2-D lattice theory (Avendaño, Davis & Mortari, CMDA 2013;
restated in Casanova, Avendaño & Mortari, IEEE Trans. AES 50(2), 2014, p. 1348, eqs. 3–4, and Arnas
et al., CMDA 121, 2015, eq. 3) defines the lattice as the solutions of
`[No 0; Nc′ Nso]·[ΔΩ; ΔM] = 2π·[i; j]`, which gives `M = 2π·(j/Nso − Nc′·i/(No·Nso))`. The two
conventions describe the same set of lattices with `Nc = −Nc′ mod No`
(`derivations/lattice_flower.mac`, checked as a point set in `derivations/lattice_flower.g`). The
Ф17.B gate found the sign difference; the code had documented neither convention.

## Decision

Keep `Nc` with Walker's sign: a circular lattice with `(No, Nso, Nc) = (P, S, F)` is the Walker
`i: T/P/F` satellite for satellite. The convention is stated on the type, on the field, in the Python
binding and in the C header, with the conversion `Nc = −Nc′ mod No`.

## Consequences

- No caller's answer moves. The Rust, Python and C APIs keep the meaning their tests pin
  (lattice (4, 6, 3) equals Walker 4/6/3).
- A configuration copied from the lattice papers must be negated mod `No`. The documentation says so
  where the number is entered.

## Alternatives

- **Adopt the papers' `Nc′`.** Rejected: it silently changes the satellites every existing call
  produces, in three published APIs (crates.io, PyPI, the C header), and breaks the identity with
  Walker's F, the convention most users bring.
- **A second constructor taking `Nc′`.** Rejected for now: a second entry point for one sign is API
  surface without a caller. Revisit if a user brings literature tables.
