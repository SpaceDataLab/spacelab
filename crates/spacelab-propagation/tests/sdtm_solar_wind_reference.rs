//! The shipped SDTM-1 solar-wind field against the generator that fitted it.
//!
//! `tools/fit-sdtm-corpus.py --ap60 --solar-wind fit --emit` writes `sdtm/solar_wind_coefficients.rs`
//! and two fixtures: the field at a sample of the geometries it fitted on, as `sdtm_reference.rs`
//! checks the 3-hourly field, and closed-form synthetic hours run through the coupling, the map,
//! the drive and the storm state — the hourly path the 3-hourly field does not have. Neither fixture
//! holds measured data: the first is our own model's values, the second the generator's
//! `synthetic_hours`.
//!
//! A mismatch is a transcription failure, so the tolerances are what two correct implementations
//! of the same arithmetic differ by: `libm` against numpy, an ulp apiece.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_propagation::sdtm::{
    self, CouplingMap, SdtmDrivers, SdtmPoint, SolarWind, SolarWindDrive, StormHour,
    solar_wind_coefficients as coefficients,
};

const REFERENCE: &str = include_str!("fixtures/sdtm_solar_wind_reference.csv");
const HOURS: &str = include_str!("fixtures/sdtm_solar_wind_hours.csv");

fn records(text: &str, fields: usize) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<f64> = line
                .split(',')
                .map(|v| v.parse().expect("a number in every field"))
                .collect();
            assert_eq!(f.len(), fields, "{fields} fields: {line}");
            f
        })
        .collect()
}

/// `(point, drivers, T∞, ρ)` of each sampled row.
fn rows() -> Vec<(SdtmPoint, SdtmDrivers, f64, f64)> {
    records(REFERENCE, 9)
        .into_iter()
        .map(|f| {
            let point = SdtmPoint {
                altitude: f[0],
                latitude: f[1],
                local_solar_time: f[2],
                day_of_year: f[3],
            };
            let drivers = SdtmDrivers {
                f107: f[4],
                f107a: f[5],
                storm: f[6],
            };
            (point, drivers, f[7], f[8])
        })
        .collect()
}

#[test]
fn the_evaluator_reproduces_the_generators_own_values() {
    let field = coefficients::FIELD;
    let reference = rows();
    assert!(reference.len() >= 32, "{} rows is too few", reference.len());
    let (mut worst_temperature, mut worst_density) = (0.0_f64, 0.0_f64);
    for (point, drivers, temperature, density) in &reference {
        let t = field.exospheric_temperature(point, drivers);
        let rho = field.density(point, drivers);
        worst_temperature = worst_temperature.max(((t - temperature) / temperature).abs());
        worst_density = worst_density.max(((rho - density) / density).abs());
    }
    println!("worst T∞ {worst_temperature:.3e}, worst ρ {worst_density:.3e} relative");
    assert!(
        worst_temperature < 1.0e-12,
        "worst T∞ mismatch {worst_temperature:e}"
    );
    assert!(
        worst_density < 1.0e-11,
        "worst ρ mismatch {worst_density:e}"
    );
}

/// Every synthetic hour's coupling and drive, and every window's storm driver, are the
/// generator's: the coupling's arithmetic, the map's interpolation and end values, the gap
/// fallback, the share and the one-hour cooling recursion over [`coefficients::HISTORY`] hours.
#[test]
fn the_hourly_path_reproduces_the_generators_own_values() {
    let records = records(HOURS, 7);
    let hours: Vec<StormHour> = records
        .iter()
        .map(|f| StormHour {
            ap60: f[0],
            solar_wind: (!f[1].is_nan()).then_some(SolarWind {
                speed: f[1],
                by: f[2],
                bz: f[3],
            }),
        })
        .collect();
    let relative = |got: f64, want: f64| {
        if want == 0.0 {
            got.abs()
        } else {
            ((got - want) / want).abs()
        }
    };
    let (mut worst_coupling, mut worst_drive, mut worst_storm) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut windows = 0;
    for (end, (hour, f)) in hours.iter().zip(&records).enumerate() {
        match hour.solar_wind {
            Some(wind) => worst_coupling = worst_coupling.max(relative(wind.coupling(), f[4])),
            None => assert!(f[4].is_nan(), "a gap has no coupling"),
        }
        worst_drive = worst_drive.max(relative(coefficients::DRIVE.drive(hour), f[5]));
        if !f[6].is_nan() {
            let window = &hours[end + 1 - coefficients::HISTORY..=end];
            let storm = coefficients::DRIVE.storm(coefficients::FIELD.storm(), window);
            worst_storm = worst_storm.max(relative(storm, f[6]));
            windows += 1;
        }
    }
    println!(
        "{} hours, {windows} windows: worst coupling {worst_coupling:.3e}, drive {worst_drive:.3e}, \
         storm {worst_storm:.3e} relative",
        hours.len()
    );
    assert_eq!(
        windows, 49,
        "the windows ending in the last 48 hours and the one before"
    );
    assert!(
        hours.iter().filter(|h| h.solar_wind.is_none()).count() >= 25,
        "the gaps are there"
    );
    assert!(
        worst_coupling < 1.0e-13,
        "worst coupling mismatch {worst_coupling:e}"
    );
    assert!(
        worst_drive < 1.0e-13,
        "worst drive mismatch {worst_drive:e}"
    );
    assert!(
        worst_storm < 1.0e-12,
        "worst storm mismatch {worst_storm:e}"
    );
}

/// The synthetic hours reach both ends of the map and its zero, so the clamps are exercised by the
/// comparison above and not only by the unit tests.
#[test]
fn the_synthetic_hours_reach_both_ends_of_the_map() {
    let (couplings, ap60) = coefficients::DRIVE.map().knots();
    let winds: Vec<f64> = records(HOURS, 7)
        .iter()
        .filter(|f| !f[1].is_nan())
        .map(|f| f[4])
        .collect();
    assert!(
        winds
            .iter()
            .any(|&c| c > couplings[coefficients::KNOTS - 1])
    );
    assert!(winds.iter().any(|&c| c == 0.0 && c < couplings[0]));
    assert_eq!(
        ap60[0], 0.0,
        "the quietest class an hour with the solar wind has had"
    );
}

/// The generated literals are a field, a map and a drive the checked constructors accept, which is
/// where their refusals bind, and the map is `ap60`'s classes in order.
#[test]
fn the_generated_literals_pass_the_checked_constructors() {
    let field = coefficients::FIELD;
    assert!(
        sdtm::Thermosphere::from_coefficients(
            *field.coefficients(),
            field.sigma(),
            field.boundary_scale(),
            *field.storm(),
        )
        .and_then(|f| f.with_flux_excess(*field.flux_excess()))
        .and_then(|f| f.with_storm_gain(*field.storm_gain()))
        .is_some(),
        "the generated literals are not a field the checked constructor accepts"
    );
    let drive = coefficients::DRIVE;
    let (couplings, ap60) = drive.map().knots();
    let map = CouplingMap::new(*couplings, *ap60).expect("a map the checked constructor accepts");
    assert_eq!(SolarWindDrive::new(drive.share(), map), Some(drive));
    assert!(
        ap60.iter().all(|v| v.fract() == 0.0),
        "ap60 classes are integers"
    );
    assert_eq!(coefficients::HARMONICS, (coefficients::DEGREE + 1).pow(2));
    assert_eq!(
        coefficients::HISTORY % 24,
        0,
        "a whole number of days of hours"
    );
}

/// Every sampled row is a physical atmosphere and inside the envelope the generator wrote, and the
/// envelope is the one the docs quote, rounded.
#[test]
fn the_field_is_physical_and_inside_its_envelope_everywhere_it_was_sampled() {
    let (field, envelope) = (coefficients::FIELD, coefficients::ENVELOPE);
    for (point, drivers, _, _) in &rows() {
        let t = field.exospheric_temperature(point, drivers);
        assert!((500.0..2500.0).contains(&t), "T∞ = {t} K at {point:?}");
        let (rho, gradient) = field.density_gradient(point, drivers);
        assert!(
            rho > 0.0 && gradient < 0.0,
            "ρ = {rho:e}, dρ/dζ = {gradient:e}"
        );
        assert!(
            envelope.contains(point, drivers),
            "{point:?} {drivers:?} outside {envelope:?}"
        );
    }
    let km = |m: f64| (m / 1000.0).round();
    let tenth = |x: f64| (x * 10.0).round() / 10.0;
    assert_eq!(
        (km(*envelope.altitude.start()), km(*envelope.altitude.end())),
        (225.0, 545.0)
    );
    assert_eq!(
        (tenth(*envelope.f107a.start()), tenth(*envelope.f107a.end())),
        (65.8, 231.9)
    );
}
