# Orekit TEME→ITRF oracle

A third-lineage numerical oracle for the frame chain. Every other reference `spacelab` validates
against descends from SOFA (`pyerfa`) or Vallado (`skyfield`, `python-sgp4`); Orekit implements the
IERS Conventions directly and descends from neither. See `plans/04-validation-and-risks.md` §1.2
and `notes/13-orekit.md`.

`OrekitTemeItrfOracle.java` produces `crates/spacelab-frames/tests/fixtures/orekit_teme_itrf.csv`
(and a richer `.json`), consumed by `crates/spacelab-frames/tests/orekit_oracle.rs`.

## Provenance

- **Orekit 13.1.7**, **Hipparchus 4.0.3**, from Maven Central, published SHA-1s verified.
  - `orekit-13.1.7.jar` → `dc3434a6936ae5decf96fcf58c02722c88ed50ff`
  - `hipparchus-core-4.0.3.jar` → `a5fb9ddc2460b99d3e46204be3d546399abe98af`
  - (geometry, ode, fitting, optim, filtering, stat at the same version)
- **orekit-data** from `https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip`.

## What it guarantees

The EOP (`xp`, `yp`, `UT1−UTC`) are **injected** per epoch, not read from `orekit-data`, and the
tool asserts at run time that Orekit used exactly the injected values — so the Rust test can feed
its chain the identical inputs. Nutation corrections are zero (pure IAU model), LOD is zero
(nominal Earth rate `ω⊕ = 7.292115146706979e-5` rad/s), and no ITRF datum (Helmert) step is applied
— this is the raw CIO-based terrestrial frame, which is what a from-scratch TEME→ITRF routine
produces. It emits both `IERS_2010` and `IERS_1996`; the difference between them (0.83–1.95 m on the
test vector) is the CIRF precession-nutation model sensitivity and is recorded per epoch.

## Regenerate

Needs a JDK (built with 25) and network. No Maven. From `/tmp/orekit-oracle/` (or any scratch dir):

```sh
# download the 8 jars + orekit-data (see provenance above), then:
javac -cp "jars/orekit-13.1.7.jar:jars/hipparchus-core-4.0.3.jar:jars/hipparchus-geometry-4.0.3.jar:jars/hipparchus-ode-4.0.3.jar:jars/hipparchus-fitting-4.0.3.jar:jars/hipparchus-optim-4.0.3.jar:jars/hipparchus-filtering-4.0.3.jar:jars/hipparchus-stat-4.0.3.jar" OrekitTemeItrfOracle.java
java  -cp "jars/*:." OrekitTemeItrfOracle
```

Then re-derive the CSV the test reads from the JSON, and re-run `cargo test -p spacelab-frames`.

## What our chain measures against it

- **GMST82**: 3.9×10⁻¹² rad worst over ten non-leap epochs (25 µm at the equator).
- **TEME→ITRF**: 0.89 m, flat across 2000–2030 — the TEME-definition convention floor of the
  GMST82 equinox path against Orekit's full CIO chain.
- **Inside a leap second**: exactly `ω⊕ × 1 s` — the naive coordinate's documented aliasing, pinned.

The Java source is committed as provenance, not as a dependency. We derive from mathematics and cite
Orekit; we do not port it (`spacelab/CONTRIBUTING.md`).

---

# IGS precise-orbit fit oracle (`IgsOrbitFitOracle.java`)

An independent-lineage cross-check for `crates/spacelab/tests/igs_orbit_fit.rs`: the same SP3 file,
the same seven estimated parameters, solved by Orekit's `BatchLSEstimator` on Orekit's own SP3
parser, frame chain, DE440 ephemerides and Holmes-Featherstone geopotential. Nothing is shared but
the product file.

## Provenance

- **Orekit 13.1.7**, **Hipparchus 4.0.3** from Maven Central — the same jars as the frame oracle above.
- **orekit-data** from `https://gitlab.orekit.org/orekit/orekit-data/-/archive/main/orekit-data-main.zip`
  (21.1 MB), which supplies the EOP, the DE-440 ephemerides and `Potential/eigen-6s.gfc`.
- Input: `data/validation/IGS0OPSFIN_20250010000_01D_15M_ORB.SP3`, sha256
  `2c4fe3942890ab256122f43a426af61b248e5731dafe2adc047ee538eb358410`, fetched by
  `tools/fetch-validation-data.sh`.

## Two deliberate model differences

Orekit reads **EIGEN-6S** where the Rust side uses XGM2019e — identical to ~1e-9 relative below
degree 12 — and its `IsotropicRadiationSingleCoefficient` is parameterised as
`pRef · cr · area / mass` where ours is a single reference acceleration, so the 5-argument
constructor is used with `pRef = SRP_NOMINAL · MASS`, unit area and unit coefficient. The estimated
reflection coefficient is then directly the Rust side's SRP scale. The shadow body is a sphere
(`flattening = 0`), because the Rust conical shadow uses one equatorial radius and a flattening
difference would show up as penumbra geometry rather than as the physics under test.

## Run

```sh
javac -cp "jars/*" IgsOrbitFitOracle.java
java -cp "jars/*:." IgsOrbitFitOracle path/to/file.SP3 [G02]
```

`-Dorekit.data=<dir>` overrides the `orekit-data` location. Omit the satellite id for all of them;
one that does not converge is reported and skipped rather than aborting the run.

## What it measured

2025-01-01, all 32 GPS satellites, 96 position observations each: mean 3D post-fit RMS **0.255 m**
against the Rust side's **0.249 m** on DE440. Per satellite the two differ by **≤ 11 mm on 31 of the
32** (G19 is the exception, where Orekit lands 104 mm worse), the same G31 is worst in both at
0.62 m, and the estimated solar-pressure scales agree to 0.3%. Full table and interpretation in
`docs/verification.md` §10.1.

---

# Measured-density oracle (`DensityOracle.java`)

An independent-lineage baseline for `crates/spacelab-propagation/tests/measured_density.rs`: the
same TU Delft neutral-density files, the same days, three empirical thermosphere models — through
Orekit's own index parsers, geodetic-to-inertial chain, solar position and model code. Nothing is
shared but the measurement.

## Provenance

- **Orekit 13.1.7**, **Hipparchus 4.0.3**, same jars as above.
- **orekit-data**, which already carries `CSSI-Space-Weather-Data/SpaceWeather-All-v1.2.txt` and
  `Space-Environment-Data/{SOLFSMY.TXT,DTCFILE.TXT}`. Orekit reads the fixed-width layouts of the
  same CelesTrak and Space Environment Technologies series that the Rust side reads as CSV and
  free-form text: same publishers, separately parsed, so a driver disagreement shows up as one.
- Input: the TU Delft products fetched by `tools/fetch-density-data.sh`, or the two committed
  fixture days under `crates/spacelab-propagation/tests/fixtures/`.

## One deliberate model difference

**Orekit ships DTM2000, not DTM2020.** There is no DTM2020 in Orekit 13.1.7, so that column is a
generation comparison rather than a parity check — and it is the most lopsided number in the file.

## Run

```sh
javac -cp "jars/*" DensityOracle.java
java -cp "jars/*:." DensityOracle path/to/*_DNS_*.txt
```

`-Dorekit.data=<dir>` overrides the `orekit-data` location. Several files may be passed at once;
each is reported separately.

## What it measured

On the two committed fixture days, driven from the daily `Ap` as Orekit's `NRLMSISE00` does,
**the two implementations agree to 1e-3 relative** — bias 2.019 against 2.019 on 2019-01-01 and
2.086 against 2.087 on 2024-05-10, with the RMS matching to the same three decimals. JB2008 agrees
to 5e-3. That parity is what licenses the rest: the difference between the Rust side's 1.661 and
Orekit's 2.086 on the storm day is entirely the 3-hourly `ap` array, not the model.

Across all twelve downloaded product files — six months of GRACE-FO C and six of Swarm A, 2.1 M rows
at full resolution here against one row in six on the Rust side — the largest disagreement in bias is
**0.008 on NRLMSISE-00 and 0.002 on JB2008**, part of which is the decimation rather than either
implementation. The Swarm files exercise the UTC branch of the time-scale column and the nine-column
form of the product; the GRACE-FO files exercise GPS time and twelve columns. Full tables and
interpretation in `docs/verification.md` §10.2.

---

# Light-time oracle (`OrekitLightTimeOracle.java`)

A third-lineage check on the epoch a measurement's geometry belongs to, for
`spacelab_estimation::observed`. Orekit's `Range` and `RangeRate` measurement models and its public
`AbstractMeasurement.signalTimeOfFlightAdjustableEmitter` are the reference; the Rust side is
`crates/spacelab/tests/light_time_oracle.rs`, reading
`crates/spacelab/tests/fixtures/orekit_light_time.csv`.

## The experiment isolates one variable

The orbit is **two-body Keplerian**, propagated in closed form on both sides from the same Cartesian
epoch state with the same `μ = 3.986004418e14`. Not for convenience: with SGP4 or a numerical orbit
the two trajectories would differ by metres, and metres is the size of the effect under test, so a
residual could not be attributed. Everything is in **EME2000** and the station's inertial position
and velocity are written per epoch, so the Rust side needs no frame chain — a frame difference is
worth about a metre here and is not what is being measured. The frame chain has its own oracle
(`OrekitTemeItrfOracle.java`).

Two flight times per epoch. `tau_resolved` re-evaluates the propagator at every iterate, which is the
true fixed point; `tau_shifted` is what Orekit's own `Range` uses internally, the emitter
Keplerian-shifted from the reception state. The difference is Orekit's own shortcut and is reported
rather than assumed negligible: **8.01e-8 m**, so "agrees with Orekit" is unambiguous about which
Orekit.

`GroundStation` carries estimable Earth-orientation parameters whose linear models refuse to evaluate
without a reference date. They are all left at zero — this oracle is about light time, not EOP — but
the dates have to be set or `estimateWithoutDerivatives` throws
`no reference date set for parameter prime-meridian-offset`.

## Regenerate

Needs a JDK and the same eight jars as above, plus `orekit-data`. From `/tmp/orekit-oracle/`:

```sh
javac -cp "jars/*" OrekitLightTimeOracle.java
java  -cp "jars/*:." OrekitLightTimeOracle          # writes orekit_light_time.csv, 470 rows
```

## What it measured

Over 470 above-the-horizon epochs of a 550 km, 53° orbit seen from a 35°N station:

| | |
|---|---|
| The two propagations are one orbit | 8.83e-7 m at reception, 8.38e-7 m at emission |
| Flight time against Orekit's fixed point | 2.08e-15 s = 6.23e-7 m |
| Modelled range | **6.61e-7 m**, against the instantaneous convention's **62.52 m** |
| Modelled range rate | 3.30e-9 m/s, against 0.169 m/s |
| Two-way minus one-way | 3.33 m — a different observable, not a refinement |

The range residual is smaller than the orbit agreement that bounds it: the light-time model agrees
with Orekit's to better than the two Kepler solvers agree with each other. Interpretation in
`docs/verification.md` §4.21.
