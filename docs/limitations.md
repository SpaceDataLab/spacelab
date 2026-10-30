# Limitations and the uncertainty budget

An access window printed to the millisecond is a lie if the orbit it came from is uncertain by
kilometres. This document separates the two uncertainties `spacelab` actually has — the **engine**,
which is what this codebase controls, and the **input ephemeris**, which it does not — so that no
result is quoted to more precision than it holds. Without this separation, validation is theatre:
you can match a reference to nanoseconds and still be seconds wrong about reality.

## The two budgets

| Source | Magnitude | What sets it |
|---|---|---|
| **Engine — frame chain** | ground track **2.3 mm** vs Skyfield over 1.6 h, against an 11 mm bar (`spacelab-sgp4/tests/ground_track.rs`, C1) | our TEME→ITRF and geodesy vs an independent lineage |
| **Engine — event edges** | access-window edges **82 µs** vs Orekit (`spacelab/tests/access_oracle.rs`, C2) | the TEME→ITRF path and the two SGP4 transcriptions. *Not* Brent convergence: dropping the abscissa tolerance from `1e-4` to `1e-9` moves the residual to 107 µs rather than reducing it |
| **Engine — integrator** | DOP853 order-8 verified to **~1e-29** in CAS; energy bounded, not drifting, for the symplectic methods | the numerical method, measured, not assumed |
| **Earth orientation** (if zeroed) | **38.0 ms** of pass-edge shift, measured — 464× the residual that remains once it is supplied | `ΔUT1 = −0.481 s` at the fixture epoch turns the Earth 7.2″, ~223 m at the equator, which a ground track sweeping at ~6.9 km/s crosses in tens of ms. Every coverage test now names an [`EopSource`]; zeroing it is `EopSource::NoEop` and is a declaration |
| **Model — atmospheric refraction** | pass edges **0.8 s** at a 10° mask to **20 s** at the horizon, one-signed (`spacelab/tests/refraction_in_access.rs`) | the elevation geometry is geometric by contract; **the conversion is now a constructor** — `Refraction::elevation_mask`, `ElevationDetector::apparent`, `Sensor.elevation_apparent`, `elevation_mask_apparent` — and a caller who states a geometric mask still gets one |
| **Model — light time** | range **51.8 m** at the horizon, line of sight **1.79″–5.30″** with a range-independent `|v|/c` ceiling of 5.30″ (`spacelab/tests/light_time_in_measurements.rs`) | the measurement geometry is instantaneous; the solver exists and nothing calls it by default |
| **Input — TLE / SGP4** | **1–3 km/day**, along-track dominant | the element set's own accuracy and SGP4's mean-element theory |

The engine terms are three to five orders of magnitude below the input term. **The TLE error is the
floor**, and everything the *engine* does is comfortably inside it.

The refraction and light-time rows are the exceptions, and they are a different kind of term: not
engine limits but models left out, biases rather than scatter, and each overtakes the TLE floor in
the regime it belongs to — refraction as the mask approaches the horizon, light time whenever the
question is a *measurement* rather than a pass time. They are listed here rather than under
"modelling simplifications" alone because a budget that omits them is wrong about which term
dominates.

The two are not the same size against the same yardstick, and it matters which question is being
asked. For a *pass time* the light-time term is negligible: 51.8 m of range at the horizon is 7 ms
of pass time at 7.5 km/s, well inside the TLE floor. For a *measurement residual* it is the
dominant modelling term by two orders of magnitude over the engine — which is why it sits in this
table under the estimation path and not under coverage.

## What the TLE error means for an access window

A LEO satellite moves at ~7.5 km/s. A **1 km** along-track position error is therefore about
**0.13 s** of pass-time uncertainty; a few days off the TLE epoch, at a few km/day, is **0.5–2 s**.
This is the *real* uncertainty on a rise/set time — three orders of magnitude larger than the engine's
82 µs agreement with Orekit, and larger still than the millisecond a root-finder can resolve.

So a coverage product should report access windows to about a **second**, and a max-gap revisit to
about **a minute**, when its input is a TLE. Reporting them tighter is false precision: the extra
digits are the engine's, not the world's.

That second holds at a working mask of 10° or so. Take the mask down toward the horizon and the
refraction row above takes over — 20 s on a grazing pass — and the answer is no longer uncertain by a
second in either direction but systematically late to open and early to close.

## The consequence for the strong claims

- **"Access-window edges to the millisecond"** — true of the *engine* (C2: 82 µs), false of the
  *answer* when the orbit is a TLE. Quote the second.
- **"Revisit gap of 1.2 h"** (`constellation` tests) — the *hours* digit is real; do not defend the
  minutes against a TLE-sourced ephemeris.
- **The only way past this floor is better input**, not a better engine: orbit determination from
  tracking observations, which is exactly what `spacelab-estimation` exists to do (square-root
  information filter, `notes/14`). With a precise state and a real force model, the engine's
  millimetres and milliseconds finally become the answer's, and this budget is rewritten around the
  covariance of the fit rather than the TLE.

## Numerical floors that depend on where the state sits

Most of this document is about the *world* being less certain than the engine. This section is the
one place where the engine's own arithmetic is the limit, and it is reachable from the crate's own
domain rather than hypothetical.

**A sigma-point filter's accuracy depends on the magnitude of its state.** The scaled unscented
transform perturbs the mean by `γS`, and at the Van der Merwe default `α = 1e-3` that is a
*thousandth* of a standard deviation. An ECI position in metres against a converged metre-level
covariance therefore asks `f64` to resolve a `1e-6` perturbation on a number of order `1e7` — and the
weights that then combine the cloud are `±1/α² = ±1e6`.

Measured by `spacelab-estimation/examples/srukf_origin_sweep.rs`: one SR-UKF update on a linear
problem with `P₀ = diag(1e-6, 4e-6)`, `σ = 1e-3`, `h(x) = x₀ + x₁`, against the closed-form Kalman
gain, sweeping `|x̂|`. Relative error in the posterior correction and in the posterior covariance;
the last column is the covariance-form `unscented_transform` on the same decade.

| `\|x̂\|` | mean (before) | mean (now) | covariance (now) | plain transform (before → now) |
|---|---|---|---|---|
| 1e0 | 1.8e-7 | 5.6e-8 | 8.6e-11 | 8.2e-11 → 8.2e-11 |
| 1e3 | 2.4e-4 | 1.3e-9 | 1.8e-8 | 2.7e-7 → 4.6e-8 |
| 1e5 | 7.6e-3 | 8.7e-8 | 7.7e-7 | 3.6e-4 → 7.6e-5 |
| 1e6 | **1.1e-1** | 2.3e-7 | 7.7e-7 | 4.9e-2 → 2.6e-3 |
| 1e7 | **9.9e-1** | 9.9e-4 | 2.0e-3 | 3.2e-1 → 1.6e-1 |
| 1e9 | **9.9e-1** | 3.8e-3 | 5.4e-2 | 5.2e4 → 2.6e3 |

"Before" is the direct `Σ Wᵢᵐ Yᵢ`, whose intermediates are a million times its own answer; "now"
accumulates every cloud from its own centre point (`unscented::centered_residuals`), which is the
same number in exact arithmetic. That buys five orders of magnitude at `1e6` — a converged LEO fit in
metres — and turns a filter that was returning noise at `1e7` into one that is right to 1e-3.

What is left is not the accumulation. `sigma_points` can only offset the mean by `γS`, and on a state
of 1e7 that offset is itself quantized at `ulp(1e7) = 1.9e-9`, so the measurement function is handed
a perturbation already `ulp(x̂)/γS ≈ 1e-3` wrong. **Past `|x̂| ≈ 1e6` with a metre-level covariance,
the state has to be re-centred by the caller** — filter a deviation from a reference trajectory, not
an absolute position — because no arrangement of the arithmetic inside the filter reaches past that.

The covariance-form `unscented_transform` gives out two decades earlier than the square-root filter
(2.6e-3 against 7.7e-7 at `1e6`): its centre weight `W₀ᶜ ≈ −1e6` multiplies the *square* of a mean
offset that is zero in exact arithmetic, so what it multiplies is rounding. Prefer
`SquareRootUnscentedKalman` when the state is large.

## Modelling simplifications currently in force

- **Two nadirs, now both expressible.** Cell zeniths use the geodetic normal, while `Cone` and
  `AgileOptical` measure off-nadir from the *geocentric* one. The deflection is
  `atan(e²/(2√(1−e²)))` = **0.192 424 3°** at the surface — at geodetic latitude **45.096 2°**, not
  45° — falling to **0.177 125°** at 550 km, which is **1.70 km** of ground straight below.
  `derivations/nadir_deflection.mac` derives both the closed form and the values at height.

  `Boresight::Fixed` now takes any axis and `spacelab_earth::geodetic_nadir` produces the geodetic
  one, so the convention is a choice a caller states rather than one the crate makes silently. It is
  more general than a nadir switch on purpose: an *agile* instrument slews, and the crate could not
  express a commanded pointing at all before. **What remains** is that `Cone::new` still measures from
  the geocentric nadir, because changing what a bare half-angle means would silently move every
  existing footprint — and that no attitude *error* is modelled, this being the difference between two
  exact conventions rather than between a convention and a spacecraft.
- **SDTM-1's shipped field is fitted on eight satellites, and its envelope is theirs.**
  `sdtm::coefficients` was fitted on 365 214 rows of the TU Delft v02 corpus — CHAMP, GRACE A/B, GOCE,
  GRACE-FO C and Swarm A/B/C, 2000-07 to 2026-04, **225 to 545 km**, F10.7a **65.8 to 231.9**
  (`docs/verification.md` §10.25–§10.28). With each mission family held out it holds **20.8 %** MAPE
  pooled, against NRLMSISE-00's 50.1 % on the same rows, and is below NRLMSISE-00 in every flux and
  storm bin (§10.28). Outside 225–545 km the profile is an extrapolation of the Bates form. The boundary
  scale sits on its fit bound, as in every fold. The levels of the eight instruments differ by up to
  9.4 % (`SATELLITE_SCALE`); the field is GRACE-FO C's. `sdtm::coefficients::UNCERTAINTY` is the width
  that goes with this — `σ` of `ln(model/measured)` 0.23 to 0.29 by regime, whose 90 % interval covers
  87.8 to 99.6 % of each family held out, and 89.0 to 90.9 % of each regime. The daily flux excess
  saturates at about 26 sfu (`sdtm::FluxExcess`, §10.26), which took the 0.76 % of rows with an excess
  of 70 sfu or more from 167 % to 25 % MAPE. **One regime is still known to be poor: high flux in storm
  time.** There the 90 % interval covers 80.8 % of the §10.2 campaign, and on 2024-05-10 (233 against
  163 sfu) the model is 1.22–1.41 times too dense before the storm begins and 36.2 % MAPE over the
  day. **Inside 225–545 km and F10.7a 65.8–231.9 sfu SDTM-1 is the recommended model; NRLMSISE-00 is
  the baseline everywhere and the model to use outside that envelope** (`docs/verification.md`
  §10.28). The envelope is `sdtm::coefficients::ENVELOPE`, `sdtm1_envelope()` and `sl_sdtm1_envelope`
  in the bridges, and each bridge has an SDTM-1 drag constructor beside the NRLMSISE-00 one; neither
  refuses an arc that leaves the envelope (§10.29). The library has no default atmosphere: every drag
  path names its model. The calibrated
  NRLMSISE-00 of §10.2 (28.9 %) is a correction fitted on the caller's own densities, and the same
  layer applies to SDTM-1.
  One convention is load-bearing and easy to get wrong: `SdtmPoint::altitude` is **geodetic** height
  above the reference ellipsoid, because that is the column the fit read (TU Delft's, GRS80) and what
  JB2008, DTM2020 and NRLMSISE-00 all take. `sdtm::REFERENCE_RADIUS` is the effective radius of the
  geopotential reduction and *not* a datum: a sphere of that radius is 21.371 km below the GRS80
  surface at the equator, so reading the two as interchangeable is a factor of 1.44 in equatorial
  density at 400 km. `sdtm::SdtmAtmosphere` takes the same geodetic closure the NRLMSISE-00 wrapper
  does, which is the way to avoid deciding this by hand.
- **SDTM-1 under-reaches the peak of the largest storms.** Its storm driver is heating with
  nitric-oxide cooling (§10.24), driven by 3-hourly `ap` over fifteen days of history
  (`coefficients::HISTORY`), and scaled by `(F10.7a/150)^κ`, `κ` = −1.30 (`sdtm::StormGain`,
  §10.27–§10.28): a storm counts 2.5 times as much at F10.7a = 75 as at 150. Scored on Bruinsma &
  Laurens (2024)'s protocol over the 204 storms the corpus covers since 2000, each satellite by the
  fit that held its family out (630 comparisons), the peak is within ±25 % on 57 %, with median peak
  ratio 1.00 and 1.03 for single-peak storms; the onset is within ±3 h on 68 % and the recovery within
  ±6 h on 80 %. The largest storms (`ap` ≥ 300) reach only 0.63 of the observed peak enhancement:
  2024-05-10 0.58–0.69. **A drag prediction through a severe storm should be expected to run low at
  the peak by a third.** JB2008, driven by hourly `Dst`, remains the reference on storm peaks in the
  published assessment (23.7 % mean error; SDTM-1 26.2 % on the same 16 comparisons, in sample).
  SDTM-1's deficit is at the peak, not in the timing or the storm's mean: the model's peak comes a
  median 1.5 h after the measured one, and over the storm it is a tenth low. Between storms that `ap`
  rates alike the ratio runs from 0.365 to 1.51 (storm medians at a first-day mean `ap` of 120 or
  more), so one severe storm can be off by more, either way (§10.30). USGS `Dst` as SDTM-1's driver
  made it worse (§10.15). Hourly `ap60` made it worse under the old kernel (§10.15); under the cooling
  state it recovers a fifth of the large-storm gap (0.64 → 0.71) and sharpens the peak's timing. It is
  not the shipped driver: GFZ's forecast of it reaches 72 h and is described as under development.
  Neither the nitric oxide's drive nor a fast uncooled mode lifts the largest storms (§10.30). Driven
  half by the measured solar wind and half by `ap60`, a held-out fit lifts them to 0.94 of the
  observed peak at the median, with the pooled error 20.75 → 20.45 % and the storm-day rows 23.4 →
  23.7 % (§10.31). That field ships opt-in as `sdtm::solar_wind_coefficients` (§10.32), for
  reconstruction and nowcasting. It cannot drive a prediction: the solar wind is measured less than
  an hour before it reaches the Earth.
- **SDTM-1's solar-wind field reconstructs and nowcasts; it does not forecast.**
  `sdtm::solar_wind_coefficients` (§10.32) runs its storm state hourly on `(1 − s)·ap60 + s·N`,
  `s` = 0.500, over 336 hours (fourteen days) of history that end at the hour before the sample. It
  was fitted on OMNI's hourly averages, time-shifted from L1 to the bow-shock nose, and on GFZ's
  `ap60`. A caller with real-time L1 data shifts it to the bow shock and averages it by the hour
  first. No real-time series was in the fit, and their error is not in the uncertainty table:
  whether a storm reconstructed on real-time solar wind scores better than on `ap60` alone is not
  measured. An hour without the solar wind drives by its `ap60`, so a gap degrades to the `ap60`
  drive and does not fail; the Halloween 2003 storm, whose plasma ACE did not measure, is such a
  storm throughout. The map from Newell's coupling onto `ap60`'s scale holds its end values, so a
  coupling above the strongest knot, 120 848, reads as `ap60` 534 however strong the solar wind is.
  The field moves the storm peaks, not the storm-day rows (§10.31). Its 90 % interval covers 78.2 %
  of the worst family's rows in the low-flux active cell, and no campaign test runs it (§10.32).
  Each field's storm driver is its own state: the 3-hourly field's driver is not the solar-wind
  field's input, and the reverse.
- **In the drag path an empirical atmosphere contributes a radial density gradient and no horizontal one.**
  `Drag`'s state-transition matrix needs `∂ρ/∂r`, and four of the five atmospheres exist only in `f64`,
  so each wrapper lifts the value through a local exponential `ρ = ρ₀·exp(−(|r| − |r₀|)/H)`. Two things
  that costs, the same for all of them: the along-track and cross-track derivative of the density is
  **zero** in the STM, although the diurnal bulge moves the real density by a factor of 2.7 across half
  a day — defensible because the radial gradient is a factor of `e` every scale height and so dominates
  the partials, but it is a modelling choice and not an approximation with a bound; and the lift is
  local, measured at 0.01 % over 1 km and 3.4 % over 50 km, which only matters to a caller that
  evaluates it once and reuses it rather than letting an integrator re-evaluate it every step.
  Where the wrappers differ is `H`: JB2008, DTM2020 and NRLMSISE-00 read it from a second model
  evaluation 10 km higher, so it is a finite difference of a table and it has a fallback arm for the
  storm-lifted layer where the 10 km step is not monotone; `sdtm::SdtmAtmosphere` has a closed form and
  uses the model's own analytic `dρ/dz`, so its radial term is exact and it needs neither the second
  evaluation nor the fallback. `d z/d|r|` is taken as one in all four, which is exact at the pole and
  the equator and low by 5.4e-6 at 45°.
- **Past the last published index day there is a baseline, not a forecast.**
  `space_weather::extend_forecast` carries the feed forward by persistence or by 27-day recurrence, and
  both are what a forecast has to beat rather than one themselves. Measured over 420 origins in the last
  fifteen years (`docs/verification.md` §10.13): persistence is worth 4.2 flux units of F10.7 at one day
  and degrades to a plateau near 18, recurrence sits flat at about 14.5 from the start and overtakes it
  at a lead of 8 days — 2 days for `Ap`. NOAA's own `PRD` forecast is in the same file and is *not*
  scored anywhere here, because a single snapshot carries it only for days that have not happened yet;
  doing so needs an archive of past snapshots this repository does not keep. A caller propagating more
  than a week past the last measurement is inside that, and `FluxQuality::Extrapolated` marks every day
  it applies to.
- **Analytic low-precision ephemerides** — the Sun is the Astronomical Almanac series and the Moon
  Montenbruck–Gill's, and both are perturbation-grade rather than pointing-grade. Measured over
  ±1 century of J2000 at 512 epochs (`spacelab-propagation/src/ephemeris.rs`, generated by
  `derivations/analytic_ephemeris.py`): the Sun within **47.4″** (0.0132°) and 9.9e-5 AU of ERFA
  `epv00`, the Moon within **278″** (4.6′) and **458 km** of ERFA `moon98`. That is why they enter as
  third-body and radiation-pressure perturbations and never as the answer.
- **The Moon's single-epoch Horizons check is a sample, not a bound** — 145 km and 0.0039° at J2000
  (`spacelab-propagation/tests/ephemeris.rs`), where the Moon was near apogee. The ±1-century sweep
  above is the bound, and it is a factor of three larger.
- **The measurement geometry is instantaneous unless a caller says otherwise, and now they can.**
  `spacelab_estimation::measurement` still forms every observable from one state, one station and one
  instant — that is the right shape for a geometry that is differentiated — and
  `spacelab_estimation::observed` is what chooses the epoch. `Signal::DownLeg` solves
  `c τ = |r_sat(t_r − τ) − r_station|` and evaluates the observable there; `Signal::Instantaneous` is
  the old convention, now stated rather than inherited. Against Orekit's own `Range` on a two-body
  orbit the modelled range closes from **62.5 m** to **6.6e-7 m** and the range rate from 0.169 m/s to
  **3.3e-9 m/s** (`spacelab/tests/light_time_oracle.rs`, §4.21). On the ISS day the untreated error is
  **51.8 m** of range at the horizon and up to **5.30″** of angular lag — the latter bounded by
  `|v|/c` whatever the range, and spanning 1.79″–5.30″ over the day as the transverse fraction of
  the velocity changes, against a factor of 1461 in the range error over the same samples
  (`spacelab/tests/light_time_in_measurements.rs`).
  Three things remain deliberately outside it. The **partial** correction needs the satellite's
  acceleration at emission and takes it as an argument, because this crate has no force model; the
  term reaches only observables whose partial has non-zero velocity rows, so a range and an angle are
  exactly unaffected by it. The **state transition matrix must reach the emission epoch** — chaining
  `Φ(t_r, t₀)` instead is worth `τ/t_r`, which over a 100 s arc is 2.3e-4, an order of magnitude
  *more* than the light-time correction on the row itself. And the **two-way** solution stays in
  `spacelab_slr::two_way_light_time`, whose iterates are pinned to the ULP against ILRS normal points;
  the oracle measures the gap between the two observables at **3.33 m**, which is what a second
  implementation would have to be worth.
- **Aberration is modelled, on the observation rather than on the model, and the term is the diurnal
  one** — 0.262″ at this station's latitude, the observer's 381.9 m/s over `c`. The 20.5″ annual term
  does *not* apply: station and satellite share the Earth's barycentric velocity, which cancels in a
  geocentric frame. `Aberration::diurnal(observer, ω)` builds the velocity as `ω × r`, verified
  against ERFA `ab` to **6.206e-17 rad** over 20 station-and-direction pairs (§4.20), and it is
  reachable from C and from Python. What no code does is apply it *automatically*: it stays off
  `spacelab_estimation::measurement`, whose functions are differentiated by `spacelab-autodiff` and
  have no use for a term that would be differentiated to zero. A caller who wants aberrated angles
  composes the two, as `tests/reduction.rs` does.
- **The chain's terrestrial realization is unnamed, and cannot be named from `finals2000A`.** Where a
  *product* states its frame the workspace now reads it — `spacelab_gnss::ReferenceFrame` takes SP3
  line 1 columns 47–51, and the four tests that difference an orbit against station coordinates
  assert the two agree — but `finals2000A` has no header at all, so the frame the CIO chain lands in
  is whatever the file's vintage realizes and nothing can recover which. The IERS C04 product says it
  in its own second line; reading that is what would close this, and §9's "EOP series" row now
  carries the reason. Until then a station and a satellite are checked against each other, and
  neither is checked against the chain.
- **Neither bridge can read an EOP series.** `spacelab-eop` is unreachable from C and from Python:
  the bridges expose `teme_to_itrf` with `xp`, `yp` and `lod_seconds` as arguments defaulting to
  zero, so a caller outside Rust either supplies Earth orientation from somewhere else or ships the
  38.0 ms silently. `EopSource` closed that inside the workspace; it does not reach the bridges, and
  that is the #319 drift pattern at crate granularity again, stated rather than closed in passing.
- **`spacelab-astrometry` is otherwise Rust-only.** Exposing aberration in both bridges was the first
  time either depended on the crate at all; plate solving, TAN/SIP world coordinates, blind asterism
  matching and streak reduction are still unreachable from C and from Python. That is the #319 drift
  pattern at crate granularity, and it is stated here rather than closed in passing.
- **Atmospheric refraction in the access geometry is a stated convention, no longer an omission.**
  `ElevationMask` and `ElevationDetector` compare against the *geometric* elevation and always will:
  that is the angle the predicate measures, and the batch kernel must never carry an atmosphere.
  What was missing was the conversion at the boundary where a caller states a threshold, and its
  absence cost **0.81–1.18 s** of pass edge at a 10° mask, **2.0–8.6 s** at 5° and **9.3–20.3 s** at
  the horizon on the ISS day of `spacelab/tests/refraction_in_access.rs` — always opening early and
  closing late, so a *bias* that, unlike the TLE error it is the size of, does not average out.

  It is now a constructor on every surface: `Refraction::elevation_mask` and
  `ElevationDetector::apparent` in Rust, `Sensor.elevation_apparent` plus `refraction_apparent` /
  `refraction_geometric` in Python, `sl_refraction_apparent` / `sl_refraction_geometric` in C, and
  `elevation_mask_apparent` in C++. Before that neither bridge exposed refraction at all, so a
  Python or C caller could not correct the bias even knowing it was there.

  The correction stays exact rather than approximate, and that is a property and not a hope:
  `h ↦ h + R(h)` is strictly increasing — swept at 0.001° from the fit's −1° floor to the zenith,
  91 001 samples, in `observability.rs` — so an apparent mask is exactly one geometric mask and shifting the threshold
  once gives the same edges as refracting every sample. What separates the two routes is the 3.7″ by
  which the Sæmundsson and Bennett fits disagree, measured in the same test, not an approximation in
  the shift. **What remains** is that the geometric mask is still the default when a caller passes a
  bare number, because changing that would silently move every existing caller's answer.

## What cannot ship, and why

Some of the best models in this domain are not in the tree, or are in it and cannot leave it, for
reasons that are legal rather than technical. Each is named here with the clause that binds and the
thing that stands in for it, so that a reader who wonders why the library does not simply use the
best model finds the answer instead of a gap. `docs/provenance.md` holds the readings.

- **NRLMSIS 2.0 / 2.1.** The NRL licence forbids "any non-academic, commercial purposes", "any
  modification or improvement" and dissemination; a Fortran-to-Rust port is a prohibited
  modification, and US 10,641,925 B2 claims the model's estimation method. Substitute: NRLMSISE-00
  (a US Government work, re-based on the NRL FORTRAN) with the calibration layer, and our own
  thermosphere model on open data (`plans/16` Ф15). NRLMSIS 2.1 remains a dated oracle number.
- **DTM2020.** CNES / Met Office / Elecnor Deimos licence: academic, non-commercial only; no
  modification; no dissemination "or any of its supporting data". It is the most accurate model in
  the density campaign (20.2 % against 25.5 % for JB2008) and it is the one file this repository
  cannot publish. Substitute: the same as above; DTM2020's numbers are cited with their date.
- **SGP4-XP.** No public algorithm specification exists, so there is nothing to implement from.
  Substitute: SGP4 with SupGP element sets, and a learned corrector on open data (`plans/16` Ф8.D).
- **HWM14.** CC BY-NC-ND: no derivatives, no commercial use. Substitute: our own wind model fitted
  to TU Delft's CC BY 4.0 crosswinds; HWM14 is run as an oracle.
- **MT_CKD.** Research-only; "may not be incorporated into proprietary software … or commercial
  software" without AER's written consent, and the clause reaches "any modified version" — so even
  our own re-interpolated coefficients could not ship. Substitute: our band model fitted to HITRAN
  lines (free with citation), with the MT_CKD gap measured and recorded.
- **JB2008, SET's solar indices and HASDM.** SET's JB2008 License Agreement (2026-10-15) covers the
  code and the index files `SOLFSMY.TXT` / `DTCFILE.TXT`: the licensee's own research and internal use
  only, no translation except to compile and interface, no redistribution of the index files. So the
  public tree carries no JB2008 port and no excerpt of those files, and every JB2008 number in these
  documents was measured in the development tree. The HASDM database is "for research purposes only"
  with commercial use "only through an explicit agreement with SET". Substitute: SDTM-1 and
  NRLMSISE-00 with the calibration layer, and JB2008's indices rebuilt from open series to the
  published definitions (`plans/16` Ф18.A). NOAA's public WAM-IPE nowcast, measured as a feed, does
  not help as it stands (`docs/verification.md` §10.18).
- **Kyoto Dst, and forecast troposphere products.** WDC Kyoto "does not allow commercial
  applications" of its indices; TU Wien's VMF *forecast* products are non-commercial. Substitute:
  GFZ's Hp30 / Hp60 (CC BY 4.0) as the storm driver, and GPT3 for a troposphere without a grid.

## Capabilities deliberately deferred

These are differences against Orekit 13.1.7, GMAT R2025a, NEPTUNE, ODTBX, Basilisk and 42 that were
examined and decided against rather than missed. Each names the condition that would reopen it, so
that a future review starts from the decision instead of rediscovering the gap.

**Four entries on this list shipped the day after it was written, and the list went on saying they
had not.** `1f011b7` (2026-08-16) recorded RTCM/SSR/NTRIP, SINEX code biases, a mission-sequence and
targeting language, and LinCov as deferred; `b4aff69` (2026-08-17) shipped all four. That is the
failure mode a "deliberately deferred" list is *most* prone to, because nothing recomputes it: it is
prose asserting the absence of files, and a file appearing does not disturb it. Each is now a row in
`docs/verification.md` with a measured residual and a mutation verdict —
`spacelab-gnss/src/{rtcm,ntrip,sinex_bias}.rs` (`reseal-rtcm`, 415 caught, 0 survivors),
`spacelab-astro/src/targeting.rs` (`seal-targeting`), `spacelab-estimation/src/lincov.rs`
(`seal-lincov`, clean on the first pass). The entries are gone from the list below rather than
restated, and this paragraph is what remains of them.

- **IIRV.** A legacy NASA fixed-column state format. The CCSDS family in `spacelab-ndm` covers every
  interface that is currently exchanged; no consumer here asks for IIRV. Still the only survivor of
  the original list.

### The boundary the control pillar draws

`spacelab-control` closed the feedback ring the workspace did not have: attitude laws, latency as
three separate numbers, delayed-argument integration by the method of steps, the delay margin, LQR
and element feedback on the orbit, and distributed consensus over delayed links. What it stops short
of is worth stating with the same discipline, because a loop that exists invites the assumption that
every loop exists.

- **The loop commands an ideal body torque, and no actuator produces one.** There is no reaction
  wheel, no momentum management or desaturation, no thruster on/off modulation, no magnetorquer and
  no gimbal. `Latency::actuator` is one number, a pure transport delay — not the actuator's
  dynamics, its saturation or its quantisation. Reopens when a mission needs a wheel-sizing or a
  desaturation-cadence answer, which is the first question a real ADCS asks after the gains.
- **`closed_loop` does not run on `delayed`.** The fixed-step loop makes the delayed lookup exact
  when the latency is a whole number of steps — the case a fixed-rate ADCS is actually in — and
  `Latency::steps` refuses a non-commensurate delay rather than rounding one silently. A loop with a
  non-commensurate latency therefore cannot be simulated end to end with the estimator in it. Neither
  published delay result needs it: both linearise to a plant the DDE integrator drives directly.
- **The delay margin is the *linearised* loop's.** `margin` solves the characteristic equation of a
  second-order small-angle model, and §4.13 measures it against a nonlinear integration of the same
  loop. A margin for a large-angle slew, a saturating actuator or a gain-scheduled controller is not
  what it computes and would not be conservative.
- **No orbit-control or formation scenario has been run in Basilisk or 42.** §4.16 compares the
  *attitude* pillar against both; §4.14's station-keeping budget and §4.15's consensus bound stand on
  their own oracles — a physics floor and a published theorem — and neither has a second
  implementation behind it.
- **42 is used as an opinion on the plant, never on the controller.** Its `ThreeAxisFSW` is a
  quaternion PD with wheels and a solar-array gimbal, not the bare MRP Lyapunov law, so running a
  like-for-like controller in it needs a source change — a NOSA v1.3 *Modification*, with §3.C's
  changelog and §3.F's registration attached. Reopens if a controller comparison is worth those
  obligations; the plant comparison, which needs no source change, is what is published instead.

Checked in the same review and found **not** to be gaps, so they need no re-checking:
manoeuvres and empirical accelerations (`spacelab-propagation/src/maneuver.rs`); multi-body harmonic
gravity (Earth plus the lunar field in `lunar.rs`); Taylor-algebra / field propagation (`stt.rs` plus
`spacelab-autodiff`); attitude providers and event detectors; box-wing drag/SRP and Earth albedo; and
the tide set, which is ahead of Orekit's.
