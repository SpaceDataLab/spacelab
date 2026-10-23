//! Fitting an element set back out of the ephemeris it produced.
//!
//! The self-consistency leg of propagator conversion, and the only one with an exact answer in it.
//! A known TLE is propagated onto a grid, the grid is handed to the converter with the element set
//! itself thrown away, and the fit has to walk back to the numbers it came from. There is no model
//! error anywhere in that loop — the same SGP4 listing produces the samples and evaluates the
//! trial vectors — so anything left over is the converter's own: the finite-difference step, the
//! Gauss-Newton stopping rule, the parameter scaling.
//!
//! ## Measured
//!
//! ISS element set (25544, epoch 2008-264), 97 samples on a 15-minute grid over 24 h, position
//! only, from a guess displaced by `1e-3` rev/day, `1e-4` in eccentricity and `0.5°` in all four
//! angles. 13 iterations:
//!
//! | parameter | absolute error of the recovered value |
//! |---|---|
//! | mean motion | 0 rev/day |
//! | eccentricity | 3.7e-16 |
//! | inclination | 0 ° |
//! | right ascension | 2.8e-14 ° |
//! | argument of perigee | 8.8e-12 ° |
//! | mean anomaly | 8.7e-12 ° |
//!
//! Post-fit 3-D position RMS **5.4e-8 m**, worst single epoch 1.1e-7 m, velocity RMS 6.2e-11 m/s.
//! Two of the six come back bit-identical and the rest within a few ulps of their own magnitude.
//! The two that recover worst are the two a near-circular orbit separates worst — `ω` and `M`
//! differ by a rotation of an ellipse the ISS orbit barely has, which the formal sigmas say
//! outright: `8.1e-4°` against `1.2e-6°` for the inclination, three decades of difference in how
//! well the arc determines them.
//!
//! Adding `B*` as a seventh parameter recovers it to 4.4e-15 earth radii⁻¹ on the same arc. Asking
//! for it at GNSS altitude instead returns `RankDeficient { direction: 6 }`: the drag column of the
//! Jacobian is identically zero at 26 560 km, and an unobserved parameter is reported rather than
//! invented.
//!
//! ## What the format then costs
//!
//! The fitted numbers are not what a TLE carries: 69 columns hold the mean motion to `1e-8`
//! rev/day, the angles to `1e-4°` and the eccentricity to `1e-7`. Displacing the source element set
//! by half a column in each of six fields — the worst case rounding can produce — and printing the
//! fit of *that* moves the propagated position by **6.2 m** RMS and 8.2 m at worst over the same
//! 24 h. That is the format's quantisation, not the converter's error, and it is the floor under
//! any TLE anybody publishes.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use spacelab_estimation::{
    ConverterError, PropagatorConverter, SampledPropagator, TrajectorySample, residual_statistics,
};
use spacelab_sgp4::{
    Convention, Satellite, TLE_FIT_STEPS, TLE_FIT_STEPS_WITH_DRAG, TleBuilder, TleModel,
};

/// The `sgp4` crate's documented ISS element set: a LEO with a real, negative `B*`.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// NAVSTAR 48, the standard catalogue record out of `fixtures/supgp_paired.tle`: a 12-hour orbit
/// on the deep-space branch, with a `B*` of exactly zero because there is no drag to fit at
/// 26 560 km.
const GNSS_LINE1: &str = "1 26407U 00040A   26227.13601475  .00000027  00000+0  00000+0 0  9996";
const GNSS_LINE2: &str = "2 26407  54.8447 212.8214 0119539 303.2108  63.5863  2.00558113191139";

/// Sampling grid: 24 h at 15 minutes, the cadence an IGS precise product is quoted on.
const SPAN_SECONDS: f64 = 86_400.0;
const STEP_SECONDS: f64 = 900.0;

fn elements(line1: &str, line2: &str) -> sgp4::Elements {
    sgp4::Elements::from_tle(None, line1.as_bytes(), line2.as_bytes()).expect("a valid element set")
}

/// Propagate an element set onto the grid and keep the states as the reference trajectory.
fn reference(satellite: &Satellite, span: f64, step: f64) -> Vec<TrajectorySample> {
    let count = (span / step).round() as usize + 1;
    (0..count)
        .map(|i| {
            let t = i as f64 * step;
            TrajectorySample {
                t,
                state: satellite.state_at(t).expect("SGP4 carries the ISS a day"),
            }
        })
        .collect()
}

/// Position-only least squares with one metre of assumed uncertainty: plain least squares, and the
/// covariance then reads in metres.
fn converter() -> PropagatorConverter {
    PropagatorConverter {
        position_sigma: 1.0,
        ..PropagatorConverter::default()
    }
}

/// The measurement: the six mean elements come back out of their own ephemeris.
#[test]
fn a_tle_fitted_to_its_own_ephemeris_recovers_the_element_set_it_came_from() {
    let truth = elements(LINE1, LINE2);
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the element set loads");
    let samples = reference(&satellite, SPAN_SECONDS, STEP_SECONDS);
    assert_eq!(samples.len(), 97);

    let model = TleModel::new(truth.clone(), Convention::Reference);
    let exact = TleModel::parameters(&truth);
    // A guess displaced far enough to be a guess: a mean motion wrong in the eighth column of the
    // TLE moves the object by kilometres over the arc, and half a degree of angle by tens.
    let guess = [
        exact[0] + 1e-3,
        exact[1] + 1e-4,
        exact[2] + 0.5,
        exact[3] + 0.5,
        exact[4] + 0.5,
        exact[5] + 0.5,
    ];

    let fit = converter()
        .fit_by_finite_difference(&model, &samples, &guess, &TLE_FIT_STEPS)
        .expect("the arc is observable in all six elements");

    let names = [
        "mean motion",
        "eccentricity",
        "inclination",
        "right ascension",
        "argument of perigee",
        "mean anomaly",
    ];
    println!(
        "ISS 25544, {} samples over {} h, {} iterations, converged {}",
        samples.len(),
        SPAN_SECONDS / 3600.0,
        fit.report.iterations,
        fit.report.converged
    );
    for k in 0..6 {
        println!(
            "  {:<20} {:>20.12} recovered, error {:.3e}, sigma {:.3e}",
            names[k],
            fit.parameters[k],
            (fit.parameters[k] - exact[k]).abs(),
            fit.covariance[k][k].sqrt()
        );
    }
    println!(
        "  post-fit position RMS {:.4e} m, worst {:.4e} m, velocity RMS {:.4e} m/s",
        fit.residuals.position_rms, fit.residuals.worst_position, fit.residuals.velocity_rms
    );

    assert!(fit.report.converged, "{:?}", fit.report);
    // Measured: 0, 3.7e-16, 0, 2.8e-14, 8.8e-12, 8.8e-12 in the order above. The bounds are set
    // decades above what was seen, because the failure this guards against is a converter that
    // stops early or steps in the wrong units, and that is orders, not factors.
    let bounds = [1e-10, 1e-10, 1e-8, 1e-8, 1e-7, 1e-7];
    for k in 0..6 {
        assert!(
            (fit.parameters[k] - exact[k]).abs() < bounds[k],
            "{} recovered {} for {}",
            names[k],
            fit.parameters[k],
            exact[k]
        );
    }
    // Measured 5.4e-8 m. Ten nanometres against an orbit of 6.8 million metres is the statement
    // that nothing but round-off separates the fit from the element set that produced the samples.
    assert!(
        fit.residuals.position_rms < 1e-4,
        "post-fit RMS {:.4e} m",
        fit.residuals.position_rms
    );
}

/// `B*` is a seventh parameter where there is drag to observe it with.
#[test]
fn the_drag_term_is_recovered_alongside_the_elements_on_a_low_orbit() {
    let truth = elements(LINE1, LINE2);
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the element set loads");
    let samples = reference(&satellite, SPAN_SECONDS, STEP_SECONDS);

    let model = TleModel::new(truth.clone(), Convention::Reference);
    let exact = TleModel::parameters_with_drag(&truth);
    let mut guess = exact;
    guess[0] += 1e-4;
    guess[5] += 0.1;
    // A drag term displaced by half its own size, so the fit has to find it rather than confirm it.
    guess[6] = 0.5 * exact[6];

    let fit = converter()
        .fit_by_finite_difference(&model, &samples, &guess, &TLE_FIT_STEPS_WITH_DRAG)
        .expect("drag is observable on a 400 km orbit over a day");

    println!(
        "B* recovered {:.6e} for {:.6e} (error {:.3e}), sigma {:.3e}, post-fit RMS {:.4e} m",
        fit.parameters[6],
        exact[6],
        (fit.parameters[6] - exact[6]).abs(),
        fit.covariance[6][6].sqrt(),
        fit.residuals.position_rms
    );
    assert!(fit.report.converged, "{:?}", fit.report);
    // Measured 4.4e-15 earth radii⁻¹ against a B* of −1.1606e-5: ten significant figures.
    assert!(
        (fit.parameters[6] - exact[6]).abs() < 1e-10,
        "B* recovered {} for {}",
        fit.parameters[6],
        exact[6]
    );
    assert!(fit.residuals.position_rms < 1e-4);
}

/// What the 69 columns cost, once the answer is not already sitting on their grid.
///
/// The reference ephemeris of the tests above came out of a TLE, so its elements were *already*
/// rounded to the columns and printing the fit was free. A real fit lands wherever the trajectory
/// puts it. Displacing each element by half a column — the worst case rounding can produce —
/// and regenerating the reference from *that* measures the quantisation on its own, with the
/// converter's own error nine decades below and unable to contribute.
#[test]
fn printing_a_fit_that_is_not_on_the_column_grid_costs_metres() {
    let on_grid = elements(LINE1, LINE2);
    let mut off_grid = on_grid.clone();
    off_grid.mean_motion += 5e-9;
    off_grid.eccentricity += 5e-8;
    off_grid.inclination += 5e-5;
    off_grid.right_ascension += 5e-5;
    off_grid.argument_of_perigee += 5e-5;
    off_grid.mean_anomaly += 5e-5;

    let model = TleModel::new(off_grid.clone(), Convention::Reference);
    let source = Satellite::from_elements(&off_grid, Convention::Reference)
        .expect("the displaced element set is still an orbit");
    let samples = reference(&source, SPAN_SECONDS, STEP_SECONDS);

    let exact = TleModel::parameters(&off_grid);
    let guess = [
        exact[0] + 1e-4,
        exact[1],
        exact[2] + 0.1,
        exact[3] + 0.1,
        exact[4] + 0.1,
        exact[5] + 0.1,
    ];
    let fit = converter()
        .fit_by_finite_difference(&model, &samples, &guess, &TLE_FIT_STEPS)
        .expect("the arc is observable");

    let fitted = model.elements(&fit.parameters);
    let tle = TleBuilder::from_elements(&fitted)
        .element_set_number(1)
        .revolution_number(56_353)
        .build()
        .expect("every fitted field fits its columns");

    // The parser this crate reads catalogues with is the arbiter of whether the columns were right.
    let printed = Satellite::from_tle(tle.line1(), tle.line2()).expect("the printed lines parse");
    let rounded = residual_statistics(&printed, &samples).expect("the printed elements propagate");

    println!("{tle}");
    println!(
        "fitted   position RMS {:.4e} m, worst {:.4e} m\n\
         printed  position RMS {:.4} m, worst {:.4} m",
        fit.residuals.position_rms,
        fit.residuals.worst_position,
        rounded.position_rms,
        rounded.worst_position
    );

    assert!(
        rounded.position_rms > 1.0e5 * fit.residuals.position_rms,
        "printing an off-grid fit cost {:.4e} m against a fit residual of {:.4e} m, \
         which would mean the columns carried the answer exactly",
        rounded.position_rms,
        fit.residuals.position_rms
    );
    // Measured: 6.19 m RMS, 8.24 m worst, from half a column in each of six fields at once. The
    // sizes that make it up: 5e-5° on a 6.8 Mm radius is 5.9 m, and 5e-9 rev/day is 0.2 m of
    // along-track per revolution growing to ~3 m across the day.
    assert!(
        rounded.position_rms < 30.0,
        "printed-element RMS {:.4} m",
        rounded.position_rms
    );
}

/// Fitting velocity alongside position changes the answer very little and the covariance a lot.
#[test]
fn fitting_velocity_as_well_as_position_lands_on_the_same_elements() {
    let truth = elements(LINE1, LINE2);
    let satellite = Satellite::from_tle(LINE1, LINE2).expect("the element set loads");
    let samples = reference(&satellite, SPAN_SECONDS, STEP_SECONDS);
    let model = TleModel::new(truth.clone(), Convention::Reference);
    let exact = TleModel::parameters(&truth);
    let guess = [
        exact[0] + 1e-4,
        exact[1],
        exact[2] + 0.1,
        exact[3] + 0.1,
        exact[4] + 0.1,
        exact[5] + 0.1,
    ];

    let position_only = converter()
        .fit_by_finite_difference(&model, &samples, &guess, &TLE_FIT_STEPS)
        .expect("observable");
    let with_velocity = PropagatorConverter {
        position_sigma: 1.0,
        // A metre and a millimetre per second is the ratio an SP3 product's own records carry.
        velocity_sigma: Some(1e-3),
        ..PropagatorConverter::default()
    }
    .fit_by_finite_difference(&model, &samples, &guess, &TLE_FIT_STEPS)
    .expect("observable");

    println!(
        "position only  RMS {:.4e} m\nwith velocity  RMS {:.4e} m, velocity RMS {:.4e} m/s",
        position_only.residuals.position_rms,
        with_velocity.residuals.position_rms,
        with_velocity.residuals.velocity_rms
    );
    for k in 0..6 {
        assert!(
            (position_only.parameters[k] - with_velocity.parameters[k]).abs()
                < 1e-6 * position_only.parameters[k].abs().max(1e-3),
            "parameter {k}: {} against {}",
            position_only.parameters[k],
            with_velocity.parameters[k]
        );
    }
}

/// Where there is no drag, a seventh parameter is not free information.
#[test]
fn asking_for_a_drag_term_a_gnss_orbit_cannot_observe_is_refused_rather_than_invented() {
    let truth = elements(GNSS_LINE1, GNSS_LINE2);
    let satellite =
        Satellite::from_tle(GNSS_LINE1, GNSS_LINE2).expect("the GNSS element set loads");
    assert!(
        satellite.regime().is_deep_space(),
        "a 12-hour orbit takes the deep-space branch"
    );
    let samples = reference(&satellite, SPAN_SECONDS, STEP_SECONDS);
    let model = TleModel::new(truth.clone(), Convention::Reference);

    // Six parameters: fine, and exact.
    let exact = TleModel::parameters(&truth);
    let guess = [
        exact[0] + 1e-5,
        exact[1],
        exact[2] + 0.05,
        exact[3] + 0.05,
        exact[4] + 0.05,
        exact[5] + 0.05,
    ];
    let fit = converter()
        .fit_by_finite_difference(&model, &samples, &guess, &TLE_FIT_STEPS)
        .expect("the six elements are observable at GNSS altitude too");
    println!(
        "GNSS 26407, six parameters: post-fit RMS {:.4e} m in {} iterations",
        fit.residuals.position_rms, fit.report.iterations
    );
    assert!(fit.residuals.position_rms < 1e-4);

    // Seven: the drag column of the Jacobian is identically zero at 26 560 km, and the solver says
    // so rather than returning whatever B* the guess held.
    let seven = TleModel::parameters_with_drag(&truth);
    let outcome =
        converter().fit_by_finite_difference(&model, &samples, &seven, &TLE_FIT_STEPS_WITH_DRAG);
    println!("GNSS 26407, seven parameters: {outcome:?}");
    assert!(
        matches!(outcome, Err(ConverterError::Solver(_))),
        "expected an unobserved direction, got {outcome:?}"
    );
}
