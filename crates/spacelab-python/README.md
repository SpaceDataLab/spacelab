# spacedatalab

[![PyPI](https://img.shields.io/pypi/v/spacedatalab)](https://pypi.org/project/spacedatalab/)
[![Python](https://img.shields.io/pypi/pyversions/spacedatalab)](https://pypi.org/project/spacedatalab/)
[![crates.io](https://img.shields.io/crates/v/spacelab)](https://crates.io/crates/spacelab)
[![docs.rs](https://img.shields.io/docsrs/spacelab)](https://docs.rs/spacelab)
[![ci](https://github.com/SpaceDataLab/spacelab/actions/workflows/ci.yml/badge.svg)](https://github.com/SpaceDataLab/spacelab/actions/workflows/ci.yml)
[![wheels](https://github.com/SpaceDataLab/spacelab/actions/workflows/wheels.yml/badge.svg)](https://github.com/SpaceDataLab/spacelab/actions/workflows/wheels.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](https://github.com/SpaceDataLab/spacelab#licence)
[![Rust core · PyO3](https://img.shields.io/badge/Rust%20core-PyO3-orange)](https://github.com/SpaceDataLab/spacelab/tree/main/crates/spacelab-python)

[Website](https://spacedatalab.org) · [Source](https://github.com/SpaceDataLab/spacelab) ·
[Verification ledger](https://github.com/SpaceDataLab/spacelab/blob/main/docs/verification.md) ·
[Changelog](https://github.com/SpaceDataLab/spacelab/blob/main/CHANGELOG.md)

> **Python bindings for [spacelab](https://github.com/SpaceDataLab/spacelab), astrodynamics in
> Rust with the verification as the deliverable.** SGP4, numerical and semi-analytical propagation,
> GNSS, orbit determination, coverage and constellation design, Earth observation and SAR. Every
> number is checked against a lineage that did not come from the project, and each check is written
> down with its oracle and residual. One abi3 wheel per platform, no runtime dependencies.

```sh
pip install spacedatalab
```

```python
import spacedatalab as sdl

l1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
l2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
r, v = sdl.sgp4_propagate(l1, l2, 0.0)  # TEME position (m) and velocity (m/s)
```

`spacedatalab` is the Python entry point of [SpaceDataLab](https://spacedatalab.org): its top level
is the spacelab core, and later SpaceDataLab products arrive as submodules.

## Checked against independent references

<!-- BEGIN GENERATED FIGURES — tools/gen-figures.py -->

| ![SGP4 against python-sgp4](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/sgp4-reference.png) | ![GCRF to ITRF against ERFA](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/frames-erfa.png) |
|:--:|:--:|
| SGP4 against python-sgp4 over 24 h: at most **250 nm** (ISS) and **240 nm** (GPS); 3 % of the samples agree to the bit. Same lineage, so this pins units, epochs and frames. | The CIO chain against ERFA's identical route, 1900–2100: at most **0.012 µas**, 360 nm on the ground. ERFA's matrix route differs by up to 3.7 µas: two formulations of one model, not an error. |
| ![Brute-force sampling against the event engine](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/access-windows.png) | ![Coverage map with the analytic reach](https://raw.githubusercontent.com/SpaceDataLab/spacelab/main/docs/figures/coverage-map.png) |
| 108 cells, 24 h: at a 1 s step brute force finds **417 of 417** passes and agrees on coverage to **0.0045 %**; at 600 s it misses 79. The engine needs no step: it brackets every edge and refines it to 1 ms. | The engine's coverage of 10368 cells ends at **71.25°** latitude, the last cell inside the analytic reach of the 5° mask, **71.5°** (dashed). |

<!-- END GENERATED FIGURES -->

## What is exposed

- **Propagation**: SGP4, Kepler, Lambert, Brouwer-Lyddane, DSST (mean and osculating), and
  numerical propagation with the geopotential, tides, relativity, drag and luni-solar forces.
- **Environment**: NRLMSISE-00 and SDTM-1 density, space weather, IGRF, NeQuick-G, Klobuchar,
  Saastamoinen, and the Niell, GMF and VMF mapping functions.
- **Frames and astrometry**: TEME, GCRF, ITRF and geodetic transforms, aberration, refraction.
- **Measurements and estimation**: range, range-rate, TDOA/FDOA, DSN observables, Gauss-Newton,
  GNSS combinations, cycle slips and ambiguity resolution, collision probability.
- **Mission design**: Walker, Street-of-Coverage, Flower and Draim constellations, coverage
  metrics, access windows, sun-synchronous and repeat-ground-track orbits, the CR3BP libration
  points and halo seeds.
- **GNSS**: orbits from RINEX navigation files, DOP, GLONASS frequencies.
- **Earth observation and SAR**: swath geometry, atmospheric correction, BRDF kernels, phase
  unwrapping, interferometry, chirp-scaling focusing, PS/SBAS, stereo and pan-sharpening.
- **Control**: attitude laws and margins, LQR, Clohessy-Wiltshire, consensus on graphs.

## Wheels

One `abi3` wheel per platform covers CPython 3.11 to 3.14; an `abi3.abi3t` wheel covers 3.15 and
the free-threaded 3.15t. Built for Linux (x86_64, aarch64; manylinux 2.28), macOS (arm64, x86_64)
and Windows (x64). Other platforms build from the sdist with a Rust toolchain.

## Licence

Licensed under either of
[Apache-2.0](https://github.com/SpaceDataLab/spacelab/blob/main/LICENSE-APACHE) or
[MIT](https://github.com/SpaceDataLab/spacelab/blob/main/LICENSE-MIT) at your option. Third-party
attributions are in [NOTICE](https://github.com/SpaceDataLab/spacelab/blob/main/NOTICE).
