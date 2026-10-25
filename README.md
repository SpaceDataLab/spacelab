# spacelab

[![crates.io](https://img.shields.io/crates/v/spacelab)](https://crates.io/crates/spacelab)
[![docs.rs](https://img.shields.io/docsrs/spacelab)](https://docs.rs/spacelab)
[![PyPI](https://img.shields.io/pypi/v/spacedatalab)](https://pypi.org/project/spacedatalab/)
[![Python](https://img.shields.io/pypi/pyversions/spacedatalab)](https://pypi.org/project/spacedatalab/)
[![ci](https://github.com/SpaceDataLab/spacelab/actions/workflows/ci.yml/badge.svg)](https://github.com/SpaceDataLab/spacelab/actions/workflows/ci.yml)
[![wheels](https://github.com/SpaceDataLab/spacedatalab/actions/workflows/wheels.yml/badge.svg)](https://github.com/SpaceDataLab/spacedatalab/actions/workflows/wheels.yml)
[![MSRV 1.94](https://img.shields.io/badge/MSRV-1.94-orange)](Cargo.toml)
[![no_std core](https://img.shields.io/badge/core-no__std-informational)](crates/spacelab-core)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#licence)

[Website](https://spacedatalab.org) · [API reference](https://docs.rs/spacelab) ·
[Verification ledger](docs/verification.md) · [Limitations](docs/limitations.md) ·
[Design](docs/design.md) · [Changelog](CHANGELOG.md)

> **Astrodynamics in Rust, with the verification as the deliverable.** Time scales and reference
> frames, SGP4, numerical and semi-analytical propagation with the full IERS 2010 force model, orbit
> determination, GNSS, satellite laser ranging, coverage and constellation design, and
> Earth-observation geometry. Mathematics decides what is correct — a derivation re-checked in a
> second engine, an exact value, the equation that defines the quantity — and every check is written
> down with its residual. Other codes are comparisons, never the proof.

```toml
[dependencies]
spacelab = "0.2"    # the assembled API, or any single crate, e.g. spacelab-sgp4
```

```sh
pip install spacedatalab    # CPython 3.11+ and free-threaded 3.15t; `import spacedatalab`
```

C and C++20: build [`crates/spacelab-capi`](crates/spacelab-capi) with CMake. From a checkout:
`tools/fetch-eop.sh && cargo run -p spacelab --example ground_track`.

## At a glance

The whole chain against a product nobody here produced: a seven-parameter fit to the **IGS final
precise orbit** for 2025-01-01, post-fit residual in metres of real orbit.

| Third bodies | radial | along | cross | 3D RMS |
|---|---|---|---|---|
| analytic Sun + Moon (committed fixture, runs in CI) | 0.175 | 0.221 | 0.105 | **0.300 m** |
| DE440 kernel | 0.054 | 0.063 | 0.045 | **0.094 m** |
| Orekit 13.1.7, its own fit of the same file | 0.053 | 0.062 | 0.065 | **0.105 m** |

Earth orientation, the CIO frame chain, the geopotential, luni-solar attraction, solar pressure,
relativity, the variational equations and the estimator all have to be right together for that to
come out small. Two independent implementations agree within a centimetre
([`igs_orbit_fit.rs`](crates/spacelab/tests/igs_orbit_fit.rs)).

Four more comparisons, each against something that did not come from here, regenerated through
the Python bindings by [`tools/gen-figures.py`](tools/gen-figures.py):

<!-- BEGIN GENERATED FIGURES — tools/gen-figures.py -->

| ![SGP4 against python-sgp4](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/sgp4-reference.png) | ![GCRF to ITRF against ERFA](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/frames-erfa.png) |
|:--:|:--:|
| SGP4 against python-sgp4 over 24 h: at most **250 nm** (ISS) and **240 nm** (GPS); 3 % of the samples agree to the bit. Both run Vallado's SGP4, so this pins units, epochs and frames. | The CIO chain against ERFA's identical route, 1900–2100: at most **0.012 µas**, 360 nm on the ground. ERFA's matrix route differs by up to 3.7 µas: two formulations of one model, not an error. |
| ![Brute-force sampling against the event engine](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/access-windows.png) | ![Coverage map with the analytic reach](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/coverage-map.png) |
| 108 cells, 24 h: at a 1 s step brute force finds **417 of 417** passes and agrees on coverage to **0.0045 %**; at 600 s it misses 79. The engine needs no step: it brackets every edge and refines it to 1 ms. | The engine's coverage of 10368 cells ends at **71.25°** latitude, the last cell inside the analytic reach of the 5° mask, **71.5°** (dashed). |

<!-- END GENERATED FIGURES -->

## Why it is different

- **Time scales and frames are types.** `Epoch<Tai>` is not `Epoch<Ut1>`; `Rotation<Teme, Itrf>`
  composes only with a matching frame, and only a `Transform` carries the `ω × r` a velocity needs.
- **Epochs are exact.** An `i128` of nanoseconds since J2000, not a floating Julian date. UTC is a
  calendar, not a scale, so converting it is fallible.
- **Derived, not transcribed.** From the primary literature, with the symbolic derivations
  committed in [`derivations/`](derivations/) and re-proved in Maxima on every run.
- **Full-resolution models.** XGM2019e geopotential, FES2014b ocean tide to degree 100, the full
  IAU 2000A nutation. Truncation is the caller's choice, never the data's.
- **Mathematics decides; other codes compare.** A formula is verified by its derivation, an exact
  value or the equation that defines it. Orekit, ERFA and NEPTUNE are comparisons, published as
  distances, and the ledger names the code behind every row that still rests on one.
- **Mutation-tested.** Every crate has been through `cargo mutants`; each survivor is killed or
  argued equivalent in a ledger.
- **Lean by design.** A `no_std`, stack-allocated core: a degree-120 geopotential propagation holds
  about 0.84 MB and allocates nothing.

Each point is argued in full in [`docs/design.md`](docs/design.md).

## Crates

<details>
<summary>Every crate, what it does, and the section of the verification ledger that covers it</summary>

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
| [`spacelab-eop`](crates/spacelab-eop) | IERS Earth-orientation parameters: parsing, provenance, interpolation | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (7 rows) |
| [`spacelab-ephemeris`](crates/spacelab-ephemeris) | High-fidelity planetary ephemerides over ANISE (JPL DE440), for cislunar and deep space | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (3 rows) |
| [`spacelab-estimation`](crates/spacelab-estimation) | Orbit determination and estimation: square-root information and Kalman-family filters, smoothers, and conjunction probability | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (51 rows) |
| [`spacelab-fa`](crates/spacelab-fa) | Fundamental astronomy: time scales, Earth rotation, reference frames | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (12 rows) |
| [`spacelab-frames`](crates/spacelab-frames) | Type-level reference frames: rotations that compose only when they should | [§1 Time, Earth orientation & geodesy](docs/verification.md#1--time-earth-orientation--geodesy) (15 rows) |
| [`spacelab-geomagnetism`](crates/spacelab-geomagnetism) | IGRF-14 geomagnetic field and centred-dipole magnetic coordinates | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (2 rows) |
| [`spacelab-geometry`](crates/spacelab-geometry) | Computational geometry: exact predicates, convex hull, Delaunay triangulation and spherical Voronoi | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (7 rows) |
| [`spacelab-gnss`](crates/spacelab-gnss) | Multi-constellation GNSS: GPS, Galileo, BeiDou, QZSS, NavIC and GLONASS broadcast orbits, RINEX 3/4 and SP3 parsing, observation processing, tropospheric mapping and DOP | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (22 rows) |
| [`spacelab-gpu`](crates/spacelab-gpu) | Host-side GPU acceleration (wgpu compute) for spacelab batch operations, with CPU fallback | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (3 rows) |
| [`spacelab-grid`](crates/spacelab-grid) | H3 geodesic grid: tessellate a region into equal-area cells and place each centre in ITRF | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (2 rows) |
| [`spacelab-ionosphere`](crates/spacelab-ionosphere) | NeQuick-G (Galileo) ionospheric electron-density and slant-TEC model, verified against the JRC reference implementation | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (2 rows) |
| [`spacelab-ndm`](crates/spacelab-ndm) | CCSDS Navigation Data Messages: OPM and OEM readers and writers, CDM reader, in KVN | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (2 rows) |
| [`spacelab-propagation`](crates/spacelab-propagation) | Orbit propagation: verified integrators (Runge-Kutta, multistep, symplectic, Taylor), the IERS 2010 force model, atmospheric density models, DSST and attitude | [§2 Numerical propagation, forces & DSST](docs/verification.md#2--numerical-propagation-forces--dsst) |
| [`spacelab-python`](crates/spacelab-python) | The PyO3 bindings of the spacelab crates, added to a Python module by `register`; the `spacedatalab` wheel on PyPI is built from them | [§7 The Python boundary](docs/verification.md#7--the-python-boundary) |
| [`spacelab-python-harness`](crates/spacelab-python-harness) | Loads spacelab-python into CPython as `spacedatalab`, for this repository's Python tests, figures and experiments | [§7 The Python boundary](docs/verification.md#7--the-python-boundary) |
| [`spacelab-sensors`](crates/spacelab-sensors) | Coverage constraints: continuous scalar predicates over (satellite, cell, sensor) | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (13 rows) |
| [`spacelab-sgp4`](crates/spacelab-sgp4) | Typed, batch-shaped SGP4 propagation for many satellites | [§3 Orbital mechanics & automatic differentiation](docs/verification.md#3--orbital-mechanics--automatic-differentiation) (8 rows) |
| [`spacelab-signal`](crates/spacelab-signal) | Space signal processing: the Radon transform and filtered back-projection (tomography / SAR back-projection) | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (4 rows) |
| [`spacelab-slr`](crates/spacelab-slr) | Satellite laser ranging: CRD normal points, SINEX station coordinates, Mendes-Pavlis refraction, IERS station displacement and the two-way range observation model | [§4 Estimation, control, sensors, coverage & constellations](docs/verification.md#4--estimation-control-sensors-coverage--constellations) (12 rows) |
| [`spacelab-testkit`](crates/spacelab-testkit) | Reductions and comparisons for the workspace's accuracy tests. Dev-only; never a runtime dependency | dev-only: the comparison helpers the other crates' tests are written with |
| [`spacelab-vision`](crates/spacelab-vision) | deepvision: pure-Rust CNN inference primitives (conv2d, relu, pooling, dense, softmax) for on-board / offline image models | [§5 GNSS, ionosphere & Phase-8 (EO / SAR / vision)](docs/verification.md#5--gnss-ionosphere--phase-8-eo--sar--vision) (4 rows) |

<!-- END GENERATED CRATE TABLE -->

</details>

## Limits and provenance

[`docs/limitations.md`](docs/limitations.md) splits the error budget into what the engine adds and
what it inherits; for TLE-driven work the second dominates by a factor of a thousand. Equations come
from papers, not from other implementations: the rules are in [`CONTRIBUTING.md`](CONTRIBUTING.md),
and every module and data source has its licence reading in
[`docs/provenance.md`](docs/provenance.md).

## Licence

Licensed under either of [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT) at your option. Files
translated from permissively licensed code carry their notices in [`NOTICE`](NOTICE).
