# Design

What the README summarises in one line each, argued in full.

**Time scales are types.** An `Epoch<Tai>` cannot be passed where an `Epoch<Ut1>` is expected. Those
two differ by tens of seconds of Earth rotation, and no comment has ever prevented that mistake.
Conversions that are definitionally exact return a value; conversions that need measured data take it
as an argument and return a `Result`. The asymmetry is visible in the signatures.

**Frames live in the type system.** `Rotation<Teme, Itrf>` composes with `Rotation<Gcrf, Teme>` and
with nothing else; the compiler checks the chain and then erases it. A `Rotation` applies to a
`Position`. Only a `Transform`, which carries the angular velocity, applies to a `StateVector` —
because moving a velocity between rotating frames needs the `ω × r` term, and forgetting it costs
465 m/s at the equator while still looking like a plausible orbit.

**UTC is a calendar, not a scale.** Every other scale — TAI, TT, TDB, UT1 — ticks without
interruption, so an integer nanosecond count names each of its moments once. UTC does not: a day with
a leap second holds 86 401 of them, and `23:59:60` exists. There is no `Epoch<Utc>`, because it would
be a lie the type system then enforced. UTC is `UtcDateTime`, and converting it to TAI is fallible.

**Epochs are exact.** An `i128` of nanoseconds since J2000, not a floating Julian Date. An `f64` JD
near 2460000 resolves to about 20 µs, which is why SOFA threads a two-part `(d1, d2)` split through
all 192 of its routines. Fixed point solves it once, and the API never mentions it.

**Every routine is derived, not transcribed.** From the primary literature — see
[`derivations/BIBLIOGRAPHY.md`](../derivations/BIBLIOGRAPHY.md) — with the symbolic derivation committed
beside the code in [`derivations/`](../derivations/) and re-proved on every run by
[`derivations/verify.sh`](../derivations/verify.sh), which asserts 49 identities in Maxima and fails if
one of them stops holding. Constants are machine-generated at the source's own precision and emitted
correctly rounded; `derivations/oracle_constants.py` checks that every reference value a generator
prints is still pasted somewhere in the tree.

**Models ship at full resolution.** The runtime degree, order and term count are the caller's choice,
never baked low in the data: the geopotential carries XGM2019e, the ocean tide carries FES2014b to
degree 100 with 35 waves, the nutation series is the full IAU 2000A. Truncating for convenience is
treated as a defect, and an audit that found two of them is recorded in the history.

**Nothing is a single source of truth.** Two oracles for numbers, and the second one is chosen for
*lineage*, not for language: `pyerfa` and `sofars` are both SOFA descendants and therefore one oracle,
not two. Where a genuinely independent implementation exists it is run — Orekit for the IERS
Conventions chain, NEPTUNE for numerical propagation, NASA's reconstructed LRO trajectory for the
lunar arc, the IGS combination for the whole thing.

**Would the tests notice a wrong answer?** A green suite says the code passes as written, not that a
wrong version would fail. `cargo mutants` rewrites one operator at a time and asks; every crate has
been through it, every survivor is either killed by a new test or recorded in `.cargo/mutants.toml`
with the argument for why it is equivalent, and `docs/verification.md` §8 is the ledger. It has found
production defects the suite did not.

**Speed and footprint are properties of the design, not a later pass.** The core is `no_std` with
const-generic, stack-allocated state, so a degree-120 geopotential propagation holds about 0.84 MB
and allocates nothing. Where a profile named a hot spot it was fixed and measured: the CIO frame
transform went from 41.9 µs to 0.218 µs per call by exact interpolation of the slow CIP scalars —
192×, at the `f64` floor — and the geopotential's scratch went from `O(n²)` to `O(n)` rolling rows,
bit-identically. `cargo bench` carries the numbers.
