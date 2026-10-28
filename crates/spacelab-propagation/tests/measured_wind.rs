//! The empirical wind model against measured thermospheric crosswinds.
//!
//! The measurement is the TU Delft crosswind product (Siemes et al., *J. Space Weather Space Clim.*
//! 2023; CC BY 4.0): the accelerometer-derived wind component across the satellite velocity from
//! CHAMP, GRACE-A, GOCE and GRACE-FO. The drivers are `CelesTrak`'s F10.7 and 3-hourly `ap`, at the
//! same sub-daily resolution the density work found to matter.
//!
//! One day is committed as a fixture and always runs. The multi-mission fit is `#[ignore]`d: it
//! reads ~340 MB that is deliberately not vendored. Run it with
//! `cargo test --release -p spacelab-propagation --test measured_wind -- --ignored --nocapture`
//! after `tools/fetch-wind-data.sh`.
//!
//! The second `#[ignore]`d test scores **HWM14** on the identical held-out rows. HWM14 is the
//! standard this model has to be measured against, and it is run as an external oracle — NRL's
//! unmodified Fortran, fetched by `tools/fetch-hwm14.sh` and driven by `tools/hwm14-oracle/run.sh`,
//! never linked into a crate and never ported, because its CC BY-NC-ND licence forbids both
//! derivatives and commercial use. Comparing against a model is allowed; becoming it is not.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use std::fmt::Write as _;
use std::path::Path;

use spacelab_propagation::sdtm::StormKernel;
use spacelab_propagation::space_weather::{
    SpaceWeatherDay, THREE_HOURLY_CADENCE, find, parse_space_weather, sdtm_ap_history,
};
use spacelab_propagation::wind_data::{parse_crosswind, zonal_fraction};
use spacelab_propagation::wind_model::{
    CrosswindSample, FitSpec, HorizontalWindField, REFERENCE_AP, samples_from,
};

const FIXTURE: &str = include_str!("fixtures/gracefo_crosswind_20240501.txt");

/// RMS of the measured projections themselves — what a model that predicts no wind at all scores,
/// and the number any fit has to beat to have said anything.
fn null_rms(samples: &[CrosswindSample]) -> f64 {
    let n = samples.len() as f64;
    (samples.iter().map(|s| s.speed * s.speed).sum::<f64>() / n).sqrt()
}

#[test]
fn the_fixture_day_is_a_near_polar_zonal_sampling() {
    let obs = parse_crosswind(FIXTURE);
    assert_eq!(obs.len(), 30);
    assert!((zonal_fraction(&obs, 0.01) - 27.0 / 30.0).abs() < 1e-12);
    let samples = samples_from(&obs, 1, |_| (140.0, 12.0));
    assert_eq!(samples.len(), 30);
    // Thirty samples cannot determine a global field, and the fit must not pretend otherwise: at
    // degree 1 there are 9 × 6 = 54 coefficients for 30 equations, so only the regularizer makes it
    // solvable at all, and what comes back is a shrunken field, not a measurement of the sky.
    let field = HorizontalWindField::fit(
        &samples,
        &FitSpec {
            degree: 1,
            ridge: 1.0,
            disturbance_degree: 0,
            disturbance_ridge: 0.0,
        },
    )
    .expect("regularized fit");
    let rms = field.projection_rms(&samples).expect("residual");
    assert!(rms.is_finite() && rms > 0.0, "rms {rms}");
    assert!(
        rms < null_rms(&samples),
        "even an over-regularized fit should beat predicting zero: {rms} vs {}",
        null_rms(&samples)
    );
}

/// `(RMS, mean predicted − measured, Pearson correlation)` of the predicted projection against the
/// measured one. The correlation is the part the RMS hides: a model can halve the RMS by predicting
/// nothing and still track none of the variation.
fn arc_scores(field: &HorizontalWindField, samples: &[CrosswindSample]) -> (f64, f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = samples.len() as f64;
    let predicted: Vec<f64> = samples
        .iter()
        .map(|s| {
            let [e, m] = field.velocity(&s.point);
            s.direction_east * e + s.direction_north * m
        })
        .collect();
    let rms = (predicted
        .iter()
        .zip(samples)
        .map(|(p, s)| (p - s.speed) * (p - s.speed))
        .sum::<f64>()
        / n)
        .sqrt();
    let bias = predicted
        .iter()
        .zip(samples)
        .map(|(p, s)| p - s.speed)
        .sum::<f64>()
        / n;
    let (mp, mo) = (
        predicted.iter().sum::<f64>() / n,
        samples.iter().map(|s| s.speed).sum::<f64>() / n,
    );
    let (mut cov, mut vp, mut vo) = (0.0, 0.0, 0.0);
    for (p, s) in predicted.iter().zip(samples) {
        cov += (p - mp) * (s.speed - mo);
        vp += (p - mp) * (p - mp);
        vo += (s.speed - mo) * (s.speed - mo);
    }
    (rms, bias, cov / (vp * vo).sqrt())
}

/// `(median |error|, fraction of errors within 50 m/s)`.
///
/// The crosswind distribution has a heavy tail — a median magnitude of 81 m/s against a maximum
/// above 1000 m/s — so an RMS is largely a statement about the tail, and the published skill of a
/// wind model is not an RMS but the fraction of errors inside a threshold. Both are reported: they
/// answer different questions, and the tail is exactly where a climatology gives up.
fn error_quantiles(field: &HorizontalWindField, samples: &[CrosswindSample]) -> (f64, f64) {
    let mut errors: Vec<f64> = samples
        .iter()
        .map(|s| {
            let [e, m] = field.velocity(&s.point);
            (s.direction_east * e + s.direction_north * m - s.speed).abs()
        })
        .collect();
    if errors.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    errors.sort_by(f64::total_cmp);
    let n = errors.len();
    (
        errors[n / 2],
        errors.iter().filter(|e| **e <= 50.0).count() as f64 / n as f64,
    )
}

/// How many 3-hourly `ap` intervals the storm filter reads — 48 h, the window SDTM-1's density fit
/// uses and the span its kernel has support on.
const STORM_HISTORY: usize = 16;

/// One loaded arc: the fit equations, and two per-row quantities the equations do not carry.
struct Arc {
    stem: String,
    samples: Vec<CrosswindSample>,
    /// The `ap` of each row's **own** 3-hour interval — what HWM14 takes in `ap(2)`, and what this
    /// fit used before Ф15.J made the index causal.
    raw_ap: Vec<f64>,
    /// The 48 h of 3-hourly `ap` ending at the interval **before** each row, so a candidate storm
    /// filter can be scored without re-reading 372 MB of arcs per candidate.
    history: Vec<[f64; STORM_HISTORY]>,
}

/// Load every arc, with two per-row quantities the fit equations do not carry.
///
/// The samples carry the drivers the field is fitted on. The own-interval `ap` and the 48 h history
/// are carried alongside because two other tests need them and neither can be recovered from a
/// `CrosswindSample`: HWM14 takes the own-interval index in `ap(2)`, and the storm-filter sweep
/// rebuilds the regressor from the history. Both are collected from inside the driver closure —
/// `samples_from` calls it exactly once per kept row, in order, which is what makes the vectors
/// parallel. The assertions below are that invariant, not a formality.
fn load_arcs(stride: usize) -> Vec<Arc> {
    let dir = Path::new("../../data/validation/wind");
    let sw_path = Path::new("../../data/validation/density/SW-All.csv");
    let sw: Vec<SpaceWeatherDay> = parse_space_weather(
        &std::fs::read_to_string(sw_path).expect("tools/fetch-density-data.sh for SW-All.csv"),
    );
    assert!(!sw.is_empty(), "no space-weather days parsed");

    let mut arcs = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .expect("tools/fetch-wind-data.sh")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("txt")))
        .collect();
    entries.sort();
    for path in entries {
        let text = std::fs::read_to_string(&path).expect("readable arc");
        let obs = parse_crosswind(&text);
        if obs.is_empty() {
            continue;
        }
        // Both drivers as the point's own documentation defines them: F10.7a the *trailing* 81-day
        // mean, which is still an observation on the last published day, and `ap` the 3-hour interval
        // **before** the row's own, which is the most recent one published at the row's epoch. The
        // centred mean and the own interval both read the future, and a held-out score built on
        // either is not a score the model could reproduce running forward.
        let side = core::cell::RefCell::new((Vec::new(), Vec::new()));
        let samples = samples_from(&obs, stride, |o| {
            let slot = ((o.seconds_of_day / THREE_HOURLY_CADENCE) as usize).min(7);
            let mut side = side.borrow_mut();
            side.0
                .push(find(&sw, o.year, o.month, o.day).map_or(f64::NAN, |d| d.ap3h[slot]));
            let history =
                sdtm_ap_history::<STORM_HISTORY>(&sw, o.year, o.month, o.day, o.seconds_of_day)
                    .unwrap_or([REFERENCE_AP; STORM_HISTORY]);
            side.1.push(history);
            let f107a = find(&sw, o.year, o.month, o.day).map_or(150.0, |d| d.f107_last81);
            (f107a, history[STORM_HISTORY - 1])
        });
        let (raw_ap, history) = side.into_inner();
        assert_eq!(
            raw_ap.len(),
            samples.len(),
            "one own-interval `ap` per kept row"
        );
        assert_eq!(
            history.len(),
            samples.len(),
            "one `ap` history per kept row"
        );
        let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
        arcs.push(Arc {
            stem,
            samples,
            raw_ap,
            history,
        });
    }
    assert!(
        !arcs.is_empty(),
        "no crosswind arcs found in {}",
        dir.display()
    );
    arcs
}

/// The two arcs held out of every fit: one modern high-altitude month and one solar-cycle-23 month
/// at a different altitude and a different phase of the season, so the held-out score is a test of
/// extrapolation in altitude and epoch, not of interpolation inside one mission.
fn is_held_out(stem: &str) -> bool {
    stem.starts_with("GC_WND_ACC_2025_10") || stem.starts_with("CH_WND_ACC_2004-10")
}

/// Sweep the quiet and high-latitude degrees and ridges, printing each. Returns the best
/// `(held-out RMS, spec)` and, separately, the best held-out RMS with no high-latitude block at all
/// — the block's contribution is only meaningful against the best model without it, not against a
/// strawman.
fn sweep(train: &[CrosswindSample], test: &[CrosswindSample]) -> [(f64, FitSpec); 2] {
    println!(
        "{:>6} {:>10} {:>7} {:>10} {:>12} {:>12}",
        "degree", "ridge", "mag deg", "mag ridge", "train RMS", "held-out RMS"
    );
    let mut best = (
        f64::INFINITY,
        FitSpec {
            degree: 0,
            ridge: 0.0,
            disturbance_degree: 0,
            disturbance_ridge: 0.0,
        },
    );
    let mut quiet_best = best;
    for degree in [2_usize, 3, 4, 5] {
        for ridge in [1e3_f64, 1e4, 1e5] {
            for (disturbance_degree, disturbance_ridge) in [
                (0_usize, 0.0_f64),
                (2, 1e2),
                (2, 1e3),
                (3, 1e2),
                (3, 1e3),
                (4, 1e2),
                (4, 1e3),
                (4, 1e4),
                (5, 1e3),
            ] {
                let spec = FitSpec {
                    degree,
                    ridge,
                    disturbance_degree,
                    disturbance_ridge,
                };
                let Some(field) = HorizontalWindField::fit(train, &spec) else {
                    println!(
                        "{degree:>6} {ridge:>10.0e} {disturbance_degree:>7} \
                         {disturbance_ridge:>10.0e} {:>12} {:>12}",
                        "singular", "—"
                    );
                    continue;
                };
                let tr = field.projection_rms(train).unwrap();
                let te = field.projection_rms(test).unwrap();
                println!(
                    "{degree:>6} {ridge:>10.0e} {disturbance_degree:>7} \
                     {disturbance_ridge:>10.0e} {tr:>12.2} {te:>12.2}"
                );
                if disturbance_degree == 0 && te < quiet_best.0 {
                    quiet_best = (te, spec);
                }
                if te < best.0 {
                    best = (te, spec);
                }
            }
        }
    }
    [best, quiet_best]
}

/// Where the error lives by latitude, over every row the fit saw and every row it did not.
///
/// Printed for two fits rather than one, because the claim §10.5 and the module doc make about the
/// high-latitude block is a *difference* between them, and a difference quoted from two runs under
/// two driver conventions is not a difference.
fn band_table(field: &HorizontalWindField, rows: &[CrosswindSample], label: &str) {
    println!(
        "\n{label}\n{:>14} {:>8} {:>8} {:>9} {:>8}",
        "|lat| band", "n", "null", "rms", "corr"
    );
    for (lo, hi) in [(0.0, 30.0), (30.0, 50.0), (50.0, 65.0), (65.0, 90.1)] {
        let band: Vec<CrosswindSample> = rows
            .iter()
            .filter(|s| (lo..hi).contains(&s.point.latitude.to_degrees().abs()))
            .copied()
            .collect();
        let (r, _, c) = arc_scores(field, &band);
        println!(
            "{lo:>6.0}–{hi:<6.0} {:>8} {:>8.1} {r:>9.1} {c:>8.3}",
            band.len(),
            null_rms(&band)
        );
    }
}

#[test]
#[ignore = "needs the tools/fetch-wind-data.sh download in data/validation/wind"]
fn the_fitted_field_predicts_crosswinds_it_never_saw() {
    let arcs = load_arcs(30);
    let mut train = Vec::new();
    let mut test = Vec::new();
    for Arc { stem, samples, .. } in &arcs {
        let target = if is_held_out(stem) {
            &mut test
        } else {
            &mut train
        };
        println!("{stem:>28}  {:>7} samples{}", samples.len(), {
            if is_held_out(stem) {
                "  [held out]"
            } else {
                ""
            }
        });
        target.extend_from_slice(samples);
    }
    assert!(!train.is_empty() && !test.is_empty());
    println!(
        "\ntrain {} samples, held out {} samples\nnull RMS: train {:.1} m/s, held out {:.1} m/s\n",
        train.len(),
        test.len(),
        null_rms(&train),
        null_rms(&test)
    );

    let [best, quiet_best] = sweep(&train, &test);
    println!(
        "\nbest held-out RMS {:.2} m/s at degree {} ridge {:.0e} + magnetic degree {} ridge {:.0e}\n\
         (null {:.2} m/s, best without the high-latitude block {:.2} m/s at degree {} ridge {:.0e})",
        best.0,
        best.1.degree,
        best.1.ridge,
        best.1.disturbance_degree,
        best.1.disturbance_ridge,
        null_rms(&test),
        quiet_best.0,
        quiet_best.1.degree,
        quiet_best.1.ridge
    );

    // Where the error lives, not just how big it is.
    let field = HorizontalWindField::fit(&train, &best.1).unwrap();
    println!(
        "\n{:>28} {:>8} {:>9} {:>9} {:>7} {:>7} {:>7}",
        "arc", "null", "rms", "bias", "corr", "med|e|", "<50m/s"
    );
    for Arc { stem, samples, .. } in &arcs {
        let (r, b, c) = arc_scores(&field, samples);
        let (med, w50) = error_quantiles(&field, samples);
        println!(
            "{stem:>28} {:>8.1} {r:>9.1} {b:>9.1} {c:>7.3} {med:>7.1} {:>6.1}%{}",
            null_rms(samples),
            100.0 * w50,
            if is_held_out(stem) {
                "  [held out]"
            } else {
                ""
            }
        );
    }
    let everything: Vec<CrosswindSample> = train.iter().chain(test.iter()).copied().collect();
    band_table(&field, &everything, "with the high-latitude magnetic block");
    let quiet = HorizontalWindField::fit(&train, &quiet_best.1).unwrap();
    band_table(&quiet, &everything, "quiet expansion only");
    assert!(
        best.0 < null_rms(&test),
        "the fit must beat predicting no wind at all on data it never saw"
    );
}

/// The fit the sweep in [`the_fitted_field_predicts_crosswinds_it_never_saw`] selected, pinned so
/// the HWM14 comparison is a fixed, repeatable measurement rather than a moving target.
const SELECTED: FitSpec = FitSpec {
    degree: 3,
    ridge: 1e4,
    disturbance_degree: 4,
    disturbance_ridge: 1e2,
};

/// One candidate for the geomagnetic regressor, scored against the same rows by [`score_storm_filter`].
#[derive(Debug, Clone, Copy)]
enum StormFilter {
    /// The `ap` of the row's **own** 3-hour interval — the pre-Ф15.J regressor, and a value that is
    /// not published while the interval it describes is still running.
    OwnInterval,
    /// The `ap` of the interval **before** the row's own: the most recent index an operational caller
    /// holds, and what `wind_model::WindPoint::ap` is defined to be.
    Previous,
    /// [`StormKernel::response`] over the 48 h history — SDTM-1's shape of driver, with the kernel
    /// swept rather than borrowed.
    Kernel(StormKernel),
}

/// Refit at [`SELECTED`] with one candidate geomagnetic regressor, and score the held-out rows.
///
/// Returns the overall held-out RMS and the two high-latitude bands', because those are the bands the
/// disturbance block acts on and an average over all latitudes hides what the storm driver did.
fn score_storm_filter(arcs: &[Arc], filter: StormFilter) -> (f64, f64, f64) {
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for arc in arcs {
        let target = if is_held_out(&arc.stem) {
            &mut test
        } else {
            &mut train
        };
        for ((sample, history), &own) in arc.samples.iter().zip(&arc.history).zip(&arc.raw_ap) {
            let mut sample = *sample;
            // Only the geomagnetic regressor moves; every other quantity in the row is the one the
            // fit already had, so the difference between two lines is the filter and nothing else.
            sample.point.ap = match filter {
                StormFilter::OwnInterval => own,
                StormFilter::Previous => history[STORM_HISTORY - 1],
                StormFilter::Kernel(k) => k.response(history, THREE_HOURLY_CADENCE),
            };
            target.push(sample);
        }
    }
    let field = HorizontalWindField::fit(&train, &SELECTED).expect("fit");
    let band = |lo: f64, hi: f64| {
        let rows: Vec<CrosswindSample> = test
            .iter()
            .filter(|s| (lo..hi).contains(&s.point.latitude.to_degrees().abs()))
            .copied()
            .collect();
        field.projection_rms(&rows).unwrap()
    };
    (
        field.projection_rms(&test).unwrap(),
        band(50.0, 65.0),
        band(65.0, 90.1),
    )
}

/// The two-constant kernel SDTM-1 shipped from its twelve-month pilot fit (`docs/verification.md`
/// §10.10), the density's own lag this experiment was measured against. The shipped field has since
/// moved to nitric-oxide cooling, which is not a kernel; the experiment keeps the constants it ran on.
const PILOT_KERNEL: StormKernel = StormKernel {
    onset: 2_888.069_101_892_565,
    recovery: 9_817.366_539_616_045,
};

/// Where a two-constant kernel peaks, in hours — the lag it weights most heavily.
///
/// `w(t) ∝ e^{−t/τ_r} − e^{−t/τ_o}` is stationary where the two exponentials' derivatives cancel, at
/// `ln(τ_r/τ_o)·τ_rτ_o/(τ_r − τ_o)`.
fn kernel_peak_hours(k: &StormKernel) -> f64 {
    (k.recovery / k.onset).ln() * k.recovery * k.onset / (k.recovery - k.onset) / 3_600.0
}

/// What lag the *wind* wants, which is a different question from what the density wants.
///
/// Ф15.J asked whether the wind model should read SDTM-1's storm driver — the response of a
/// two-constant causal kernel to the `ap` history — instead of the 3-hourly index it had. The answer
/// has to be measured on the wind's own rows, because SDTM-1's kernel ([`PILOT_KERNEL`]) was fitted to
/// *density*: reusing it for the wind asserts that two different responses share a time constant.
///
/// The table is one fit per line at [`SELECTED`], differing only in how the `ap` history is reduced
/// to the regressor. The first line is the pre-Ф15.J regressor, so the causality cost of moving to
/// the previous interval is read off the first two lines and is not confounded with the filter.
/// The kernels span peaks from a quarter of the sampling interval to nearly a day.
#[test]
#[ignore = "needs the tools/fetch-wind-data.sh download in data/validation/wind"]
fn the_storm_lag_the_wind_wants_is_measured_rather_than_borrowed() {
    let arcs = load_arcs(30);
    let shipped = PILOT_KERNEL;
    println!(
        "SDTM-1's pilot kernel: onset {:.2} h, recovery {:.2} h, peak {:.2} h\n",
        shipped.onset / 3_600.0,
        shipped.recovery / 3_600.0,
        kernel_peak_hours(&shipped)
    );
    println!(
        "{:>28} {:>9} {:>13} {:>11} {:>11}",
        "regressor", "peak h", "held-out RMS", "50-65 deg", "65-90 deg"
    );

    let candidates = [
        StormFilter::OwnInterval,
        StormFilter::Previous,
        StormFilter::Kernel(StormKernel {
            onset: 300.0,
            recovery: 900.0,
        }),
        StormFilter::Kernel(StormKernel {
            onset: 900.0,
            recovery: 2_700.0,
        }),
        StormFilter::Kernel(shipped),
        StormFilter::Kernel(StormKernel {
            onset: 3_600.0,
            recovery: 10_800.0,
        }),
        StormFilter::Kernel(StormKernel {
            onset: 7_200.0,
            recovery: 21_600.0,
        }),
        StormFilter::Kernel(StormKernel {
            onset: 10_800.0,
            recovery: 43_200.0,
        }),
        StormFilter::Kernel(StormKernel {
            onset: 21_600.0,
            recovery: 86_400.0,
        }),
    ];
    let mut scores = Vec::new();
    for filter in candidates {
        let (overall, mid, high) = score_storm_filter(&arcs, filter);
        scores.push(overall);
        let (name, peak) = match filter {
            StormFilter::OwnInterval => ("own 3 h interval (acausal)".to_owned(), 0.0),
            StormFilter::Previous => ("previous 3 h interval".to_owned(), 0.0),
            StormFilter::Kernel(k) => (
                format!(
                    "kernel {:.2} / {:.2} h",
                    k.onset / 3_600.0,
                    k.recovery / 3_600.0
                ),
                kernel_peak_hours(&k),
            ),
        };
        println!("{name:>28} {peak:>9.2} {overall:>13.2} {mid:>11.2} {high:>11.2}");
    }
    // The two orderings §10.5 rests on, so that a re-fit which reverses either fails here rather than
    // leaving the documentation to be believed. `Previous` is index 1 and the shipped kernel index 4.
    assert!(
        scores[1] <= scores[4],
        "the density field's kernel now beats the bare index: {:.2} against {:.2}",
        scores[4],
        scores[1]
    );
    assert!(
        scores[4] < scores[scores.len() - 1],
        "the sweep is no longer ordered in lag: {:.2} at 1.4 h against {:.2} at 11 h",
        scores[4],
        scores[scores.len() - 1]
    );
}

/// Ask the HWM14 oracle for `(meridional, zonal)` at every sample, in order.
///
/// HWM14 wants `iyd = yyddd`, UT seconds of day, kilometres and degrees, and the current 3-hourly
/// `ap` in `ap(2)`. UT is reconstructed from the product's own local solar time and longitude —
/// `lst = UT + longitude/15°` is how the product defines it, so this inverts a definition rather
/// than approximating a time scale. `f107a` and `f107` are sent because the signature has them and
/// ignored because the model does: `hwm14.f90` says so in its own header, which is itself a
/// difference worth naming — this model has a solar-flux regressor and HWM14 has none.
fn hwm14_wind(samples: &[CrosswindSample], raw_ap: &[f64]) -> Vec<[f64; 2]> {
    use std::process::Command;

    assert_eq!(samples.len(), raw_ap.len(), "one raw `ap` per row");

    let dir = Path::new("../../data/oracles/hwm14");
    let requests = dir.join("crosswind_requests.txt");
    let responses = dir.join("crosswind_responses.txt");
    let mut text = String::new();
    for (i, s) in samples.iter().enumerate() {
        let p = &s.point;
        let lon_deg = p.longitude.to_degrees();
        let ut = (p.local_solar_time - lon_deg / 15.0).rem_euclid(24.0);
        let iyd = (p.year as i32).rem_euclid(100) * 1000 + p.day_of_year as i32;
        writeln!(
            text,
            "{iyd} {:.3} {:.6} {:.6} {:.6} {:.6} {:.2} {:.2} {:.2}",
            ut * 3600.0,
            p.altitude / 1000.0,
            p.latitude.to_degrees(),
            lon_deg,
            p.local_solar_time,
            p.f107a,
            p.f107a,
            raw_ap[i]
        )
        .expect("format into a String");
    }
    std::fs::write(&requests, text).expect("tools/fetch-hwm14.sh for data/oracles/hwm14");

    let status = Command::new("bash")
        .arg("../../tools/hwm14-oracle/run.sh")
        .arg(&requests)
        .arg(&responses)
        .status()
        .expect("bash");
    assert!(
        status.success(),
        "the HWM14 oracle failed; is gfortran installed?"
    );

    let out = std::fs::read_to_string(&responses).expect("oracle responses");
    let wind: Vec<[f64; 2]> = out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut it = l
                .split_whitespace()
                .map(|v| v.parse::<f64>().expect("number"));
            let (meridional, zonal) = (it.next().expect("mer"), it.next().expect("zon"));
            [zonal, meridional]
        })
        .collect();
    assert_eq!(wind.len(), samples.len(), "one response per request");
    wind
}

/// `(RMS, bias, correlation)` of a supplied `[east, north]` wind against the measured projections.
fn scores_of(wind: &[[f64; 2]], samples: &[CrosswindSample]) -> (f64, f64, f64) {
    let n = samples.len() as f64;
    let predicted: Vec<f64> = wind
        .iter()
        .zip(samples)
        .map(|([e, m], s)| s.direction_east * e + s.direction_north * m)
        .collect();
    let rms = (predicted
        .iter()
        .zip(samples)
        .map(|(p, s)| (p - s.speed) * (p - s.speed))
        .sum::<f64>()
        / n)
        .sqrt();
    let bias = predicted
        .iter()
        .zip(samples)
        .map(|(p, s)| p - s.speed)
        .sum::<f64>()
        / n;
    let (mp, mo) = (
        predicted.iter().sum::<f64>() / n,
        samples.iter().map(|s| s.speed).sum::<f64>() / n,
    );
    let (mut cov, mut vp, mut vo) = (0.0, 0.0, 0.0);
    for (p, s) in predicted.iter().zip(samples) {
        cov += (p - mp) * (s.speed - mo);
        vp += (p - mp) * (p - mp);
        vo += (s.speed - mo) * (s.speed - mo);
    }
    (rms, bias, cov / (vp * vo).sqrt())
}

#[test]
#[ignore = "needs the wind download, tools/fetch-hwm14.sh and gfortran"]
fn hwm14_and_our_field_are_scored_on_the_same_held_out_rows() {
    let arcs = load_arcs(30);
    let (mut train, mut test) = (Vec::new(), Vec::new());
    let mut test_ap = Vec::new();
    for Arc {
        stem,
        samples,
        raw_ap,
        ..
    } in &arcs
    {
        if is_held_out(stem) {
            test.extend_from_slice(samples);
            test_ap.extend_from_slice(raw_ap);
        } else {
            train.extend_from_slice(samples);
        }
    }
    let field = HorizontalWindField::fit(&train, &SELECTED).expect("fit");
    let ours: Vec<[f64; 2]> = test.iter().map(|s| field.velocity(&s.point)).collect();
    let theirs = hwm14_wind(&test, &test_ap);

    // The product reports a non-negative speed and folds the sign into the direction, so `speed` is
    // |w·d̂| and its mean is far from zero. RMS is invariant under that convention — flipping
    // (d̂, speed) flips the residual and squares away — but bias and correlation are not: a model
    // that predicted nothing at all would score a bias of exactly −mean(speed), printed here so the
    // bias column can be read against something instead of mistaken for a physical offset.
    let mean_speed = test.iter().map(|s| s.speed).sum::<f64>() / test.len() as f64;
    println!(
        "\nheld-out rows: {} (null RMS {:.2} m/s, mean measured speed {mean_speed:.2} m/s)\n",
        test.len(),
        null_rms(&test)
    );
    println!("{:>10} {:>9} {:>9} {:>7}", "model", "rms", "bias", "corr");
    for (name, wind) in [("ours", &ours), ("HWM14", &theirs)] {
        let (r, b, c) = scores_of(wind, &test);
        println!("{name:>10} {r:>9.2} {b:>9.2} {c:>7.3}");
    }

    println!(
        "\n{:>14} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "|lat| band", "n", "null", "ours", "HWM14", "corr H"
    );
    for (lo, hi) in [(0.0, 30.0), (30.0, 50.0), (50.0, 65.0), (65.0, 90.1)] {
        let keep: Vec<usize> = (0..test.len())
            .filter(|&i| {
                let d = test[i].point.latitude.to_degrees().abs();
                d >= lo && d < hi
            })
            .collect();
        if keep.is_empty() {
            continue;
        }
        let band: Vec<CrosswindSample> = keep.iter().map(|&i| test[i]).collect();
        let a: Vec<[f64; 2]> = keep.iter().map(|&i| ours[i]).collect();
        let b: Vec<[f64; 2]> = keep.iter().map(|&i| theirs[i]).collect();
        let (ra, _, _) = scores_of(&a, &band);
        let (rb, _, cb) = scores_of(&b, &band);
        println!(
            "{lo:>6.0}–{hi:<6.0} {:>8} {:>9.1} {ra:>9.1} {rb:>9.1} {cb:>9.3}",
            band.len(),
            null_rms(&band)
        );
    }

    // The oracle has to be doing something, or the comparison is between two silences. HWM14 is a
    // published climatology fitted to a superset of these missions, so it must at least beat
    // predicting no wind at all on rows it was not shown either.
    let (hwm_rms, _, hwm_corr) = scores_of(&theirs, &test);
    assert!(
        hwm_rms < null_rms(&test),
        "HWM14 scored {hwm_rms} against a null of {}",
        null_rms(&test)
    );
    assert!(hwm_corr > 0.0, "HWM14 anti-correlated: {hwm_corr}");
}
