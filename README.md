# spacelab

Astrodynamics in Rust, with the verification as the deliverable.

Thirty-two crates: fundamental astronomy and reference frames, SGP4, numerical and semi-analytical
propagation with the full IERS 2010 force stack, orbit determination, GNSS, satellite laser ranging,
closed-loop control, coverage and constellation design, and Earth-observation geometry. `no_std` at
the core, `#![forbid(unsafe_code)]` workspace-wide, `MIT OR Apache-2.0`, and every ported module's
ancestry read from the upstream licence itself and written down in
[`docs/provenance.md`](docs/provenance.md) — including the one place the answer is *no* and what
is being done about it.

The claim it is organised around is not "fast" or "complete". It is that **every number it produces
has been checked against something that did not come from here**, and that the check is written down
— which oracle, which lineage, what residual, in which test — in
[`docs/verification.md`](docs/verification.md), 1 026 rows of it.

```sh
tools/fetch-eop.sh
cargo run -p spacelab --example ground_track
cargo test --workspace                 # 3 078 tests
```

## The number that matters most

Every other test compares one piece against a reference for that piece. One test runs the whole
chain against a product nobody here produced: fit a seven-parameter arc to the **IGS final precise
orbit** for 2025-01-01 — the weighted combination of ten analysis centres, in IGS20 / ITRF2020, at
the satellite centre of mass — and report the post-fit residual in metres of real orbit.

| Third bodies | radial | along | cross | 3D RMS |
|---|---|---|---|---|
| analytic Sun + Moon (committed fixture, runs in CI) | 0.175 | 0.221 | 0.105 | **0.300 m** |
| DE440 kernel | 0.054 | 0.063 | 0.045 | **0.094 m** |
| Orekit 13.1.7, its own fit of the same file | 0.053 | 0.062 | 0.065 | **0.105 m** |

Measured Earth orientation, the CIO frame chain, the geopotential in the true rotating frame,
luni-solar attraction, solar radiation pressure, the Schwarzschild term, the automatic-differentiation
variational equations and the square-root batch estimator all have to be right *together* for that to
come out small. Two independent implementations land within a centimetre of each other, on
solar-pressure scales agreeing to 0.3 %. `crates/spacelab/tests/igs_orbit_fit.rs`, and the other half
of it is `tools/orekit-oracle/IgsOrbitFitOracle.java`.

## Crates

<!-- BEGIN GENERATED CRATE TABLE — tools/gen-readme.py -->

| Crate | What | Verified in |
|---|---|---|
| [`spacelab`](crates/spacelab) | Space data computation: time, Earth orientation, frames, propagation, geodesy | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) |
| [`spacelab-astro`](crates/spacelab-astro) | Two-body and conic mission design: universal-variable Kepler, Lambert transfers, patched conics | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (21 rows) |
| [`spacelab-astrometry`](crates/spacelab-astrometry) | Astrometric plate solving: source detection, TAN/SIP world coordinates, blind asterism matching and reduction to a true topocentric direction | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (10 rows) |
| [`spacelab-autodiff`](crates/spacelab-autodiff) | Forward-mode automatic differentiation: dual numbers for exact Jacobians and sensitivities | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (6 rows) |
| [`spacelab-capi`](crates/spacelab-capi) | C ABI for the spacelab space-data computation core, and the C++ header that wraps it | [§6 The C and C++ boundary](docs/verification.md#6--the-c-and-c-boundary) |
| [`spacelab-constellation`](crates/spacelab-constellation) | Constellation design: generators (Walker, Street-of-Coverage, Flower, 2-D Lattice, kinematically-regular, Draim, two-tier shell/SSA) plus the Mozhaev group classification and the alpha-characteristic coverage figure of merit | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (6 rows) |
| [`spacelab-control`](crates/spacelab-control) | Closed-loop guidance, navigation and control: feedback laws, and the sensor/compute/actuator latency the loop actually runs at | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (26 rows) |
| [`spacelab-core`](crates/spacelab-core) | Typed epochs, angles and units for space data computation | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (6 rows) |
| [`spacelab-coverage`](crates/spacelab-coverage) | Access-window detection: sign-change bracketing and bracket-preserving root refinement | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (12 rows) |
| [`spacelab-earth`](crates/spacelab-earth) | The WGS-84 ellipsoid: geodetic and Cartesian coordinates | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (7 rows) |
| [`spacelab-earthvision`](crates/spacelab-earthvision) | Earth-observation foundations: viewing geometry (sun/sensor angles) and Monte Carlo atmospheric radiative transfer (Marchuk's method) | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (72 rows) |
| [`spacelab-elements`](crates/spacelab-elements) | Osculating Keplerian orbital elements: state vector to elements and back, singular cases handled | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (9 rows) |
| [`spacelab-eop`](crates/spacelab-eop) | IERS Earth-orientation parameters: parsing, provenance, interpolation | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (5 rows) |
| [`spacelab-ephemeris`](crates/spacelab-ephemeris) | High-fidelity planetary ephemerides over ANISE (JPL DE440), for cislunar and deep space | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (3 rows) |
| [`spacelab-estimation`](crates/spacelab-estimation) | Orbit determination and estimation: square-root information and Kalman-family filters, smoothers, and conjunction probability | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (51 rows) |
| [`spacelab-fa`](crates/spacelab-fa) | Fundamental astronomy: time scales, Earth rotation, reference frames | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (12 rows) |
| [`spacelab-frames`](crates/spacelab-frames) | Type-level reference frames: rotations that compose only when they should | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (15 rows) |
| [`spacelab-geomagnetism`](crates/spacelab-geomagnetism) | IGRF-14 geomagnetic field and centred-dipole magnetic coordinates | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (2 rows) |
| [`spacelab-geometry`](crates/spacelab-geometry) | Computational geometry: exact predicates, convex hull, Delaunay triangulation and spherical Voronoi | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (7 rows) |
| [`spacelab-gnss`](crates/spacelab-gnss) | Multi-constellation GNSS: GPS, Galileo, BeiDou, QZSS, NavIC and GLONASS broadcast orbits, RINEX 3/4 and SP3 parsing, observation processing, tropospheric mapping and DOP | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (23 rows) |
| [`spacelab-gpu`](crates/spacelab-gpu) | Host-side GPU acceleration (wgpu compute) for spacelab batch operations, with CPU fallback | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (3 rows) |
| [`spacelab-grid`](crates/spacelab-grid) | H3 geodesic grid: tessellate a region into equal-area cells and place each centre in ITRF | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (2 rows) |
| [`spacelab-ionosphere`](crates/spacelab-ionosphere) | NeQuick-G (Galileo) ionospheric electron-density and slant-TEC model, verified against the JRC reference implementation | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (4 rows) |
| [`spacelab-ndm`](crates/spacelab-ndm) | CCSDS Navigation Data Messages: OPM and OEM readers and writers, CDM reader, in KVN | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (2 rows) |
| [`spacelab-propagation`](crates/spacelab-propagation) | Orbit propagation: verified integrators (Runge-Kutta, multistep, symplectic, Taylor), the IERS 2010 force model, atmospheric density models, DSST and attitude | [§2 Numerical propagation, forces & DSST](docs/verification.md#2--numerical-propagation-forces--dsst) |
| [`spacelab-python`](crates/spacelab-python) | Python bindings (PyO3) for the spacelab space-data computation core | [§7 The Python boundary](docs/verification.md#7--the-python-boundary) |
| [`spacelab-sensors`](crates/spacelab-sensors) | Coverage constraints: continuous scalar predicates over (satellite, cell, sensor) | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (13 rows) |
| [`spacelab-sgp4`](crates/spacelab-sgp4) | Typed, batch-shaped SGP4 propagation for many satellites | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (8 rows) |
| [`spacelab-signal`](crates/spacelab-signal) | Space signal processing: the Radon transform and filtered back-projection (tomography / SAR back-projection) | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (4 rows) |
| [`spacelab-slr`](crates/spacelab-slr) | Satellite laser ranging: CRD normal points, SINEX station coordinates, Mendes-Pavlis refraction, IERS station displacement and the two-way range observation model | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (12 rows) |
| [`spacelab-testkit`](crates/spacelab-testkit) | Reductions and comparisons for the workspace's accuracy tests. Dev-only; never a runtime dependency | dev-only: the comparison helpers the other crates' tests are written with |
| [`spacelab-vision`](crates/spacelab-vision) | deepvision: pure-Rust CNN inference primitives (conv2d, relu, pooling, dense, softmax) for on-board / offline image models | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (4 rows) |

<!-- END GENERATED CRATE TABLE -->

## What is different about this

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
[`derivations/BIBLIOGRAPHY.md`](derivations/BIBLIOGRAPHY.md) — with the symbolic derivation committed
beside the code in [`derivations/`](derivations/) and re-proved on every run by
[`derivations/verify.sh`](derivations/verify.sh), which asserts 49 identities in Maxima and fails if
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

## What it does not claim

`docs/limitations.md` states the uncertainty budget in two halves — the error the engine introduces,
and the error it inherits — and says which one dominates. It is worth reading before quoting any
accuracy number from here, because for TLE-driven work the second is a thousand times the first: our
footprint-edge predicate places a boundary to 1.4 nm and the TLE feeding it is a kilometre out.
Capabilities deliberately not built are listed there too, each with the condition that would reopen
it.

## Provenance

We read papers, not implementations. Equations, algorithms and measured constants are free; source
code is expression, and translating a program between languages is a derivative work. The rules — and
why SOFA's permissive-but-encumbered licence is *declined* rather than forbidden — are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). The same register covers *data*: every embedded table, every
fixture and every product a fetch script reads has a row with its terms as read and the date, and a
CI step fails on a host or a file without one.

## Licence

`MIT OR Apache-2.0`. Thirty-two files are translations of permissively licensed code — Orekit and
pyorbital (Apache-2.0), iquaflow and the `sgp4` crate (MIT) — and carry those notices in
[`NOTICE`](NOTICE); every other module is written from the published mathematics.
[`docs/provenance.md`](docs/provenance.md) records each reading, module by module and data source by
data source, and says which one thing in this tree cannot be released.
