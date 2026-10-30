//! The shipped SDTM-1 evaluator against the generator that fitted it.
//!
//! `tools/fit-sdtm-corpus.py` does the search in Python and writes `sdtm/coefficients.rs`. Two
//! implementations of one model is a transcription risk exactly like the ported atmospheres of §2 —
//! the fitted numbers can be right and the Rust still wrong about which driver multiplies which
//! harmonic, what the geopotential conversion is, or where the storm response is sampled. So the
//! generator also writes down what *it* computes at a sample of the geometries it fitted on, and this
//! is the comparison.
//!
//! The fixture is our own model's output at our own sample of points: no measured density, no driver
//! series and nothing from TU Delft's files is in it, which is why it can be committed while the
//! 23 GB of measurements it was fitted on cannot (`docs/provenance.md`, `sdtm-reference`).
//!
//! A mismatch here is a *transcription* failure, so the tolerance is what two correct implementations
//! of the same arithmetic differ by and not a physical tolerance: every field of the fixture is written
//! at round-trip precision, which leaves only `libm`'s `exp` against numpy's, an ulp apiece over the
//! dozen operations of the profile. Anything looser would accept a wrong harmonic ordering.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_propagation::sdtm::{self, SdtmDrivers, SdtmPoint, coefficients, geopotential_height};

const REFERENCE: &str = include_str!("fixtures/sdtm_reference.csv");

/// One row of the generator's output.
struct Row {
    point: SdtmPoint,
    drivers: SdtmDrivers,
    exospheric_temperature: f64,
    density: f64,
}

fn rows() -> Vec<Row> {
    REFERENCE
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<f64> = line
                .split(',')
                .map(|v| v.parse().expect("a number in every field"))
                .collect();
            assert_eq!(f.len(), 9, "nine fields: {line}");
            Row {
                point: SdtmPoint {
                    altitude: f[0],
                    latitude: f[1],
                    local_solar_time: f[2],
                    day_of_year: f[3],
                },
                drivers: SdtmDrivers {
                    f107: f[4],
                    f107a: f[5],
                    storm: f[6],
                },
                exospheric_temperature: f[7],
                density: f[8],
            }
        })
        .collect()
}

/// Every fitted coefficient reaches the evaluator in the place the generator put it.
///
/// `T∞` is checked separately from `ρ` because they fail differently: a wrong coefficient index moves
/// `T∞`, while a wrong boundary constant or geopotential conversion moves only the density.
#[test]
fn the_evaluator_reproduces_the_generators_own_values() {
    let field = coefficients::FIELD;
    let reference = rows();
    assert!(reference.len() >= 32, "{} rows is too few", reference.len());
    let mut worst_temperature = 0.0_f64;
    let mut worst_density = 0.0_f64;
    for row in &reference {
        let t = field.exospheric_temperature(&row.point, &row.drivers);
        let rho = field.density(&row.point, &row.drivers);
        worst_temperature = worst_temperature
            .max(((t - row.exospheric_temperature) / row.exospheric_temperature).abs());
        worst_density = worst_density.max(((rho - row.density) / row.density).abs());
    }
    println!("worst T∞ {worst_temperature:.3e}, worst ρ {worst_density:.3e} relative");
    // Round-tripped inputs, so this is the arithmetic's own disagreement: ~1e-15 in T∞, and that
    // amplified by the eight or nine scale heights between the boundary and the fitted band in ρ.
    assert!(
        worst_temperature < 1.0e-12,
        "worst T∞ mismatch {worst_temperature:e}"
    );
    assert!(
        worst_density < 1.0e-11,
        "worst ρ mismatch {worst_density:e}"
    );
}

/// The fitted field is a physical atmosphere at every point the generator sampled, not only on average.
///
/// The fit minimises a log residual and could in principle buy that with an unphysical profile —
/// `T∞` below the boundary, a density that rises with altitude, a scale height of 300 km at 450. This
/// is the assertion the fit itself cannot make, because the fit's objective does not contain it.
#[test]
fn the_fitted_field_is_physical_everywhere_the_generator_sampled() {
    let field = coefficients::FIELD;
    for row in &rows() {
        let t = field.exospheric_temperature(&row.point, &row.drivers);
        assert!(
            (500.0..2500.0).contains(&t),
            "T∞ = {t} K at {:?}",
            row.point
        );
        let (rho, gradient) = field.density_gradient(&row.point, &row.drivers);
        assert!(
            rho > 0.0 && gradient < 0.0,
            "ρ = {rho:e}, dρ/dζ = {gradient:e}"
        );
        let height = field.scale_height(&row.point, &row.drivers);
        assert!(
            (20_000.0..150_000.0).contains(&height),
            "scale height {height:e} m at {} m",
            row.point.altitude
        );
    }
}

/// The generator's own altitudes are inside the band it was fitted on, and the shape parameter it
/// chose is the one `coefficients.rs` carries.
///
/// A tripwire for the fit's coverage rather than for the code: the corpus spans GOCE's orbit near
/// 250 km to Swarm B's above 500 km, which is what identifies `sigma` (`docs/verification.md`
/// §10.21). If a later corpus narrows or widens the band, this test is what says the note is stale.
#[test]
fn the_fitted_band_is_the_one_the_verdict_names() {
    let altitudes: Vec<f64> = rows().iter().map(|r| r.point.altitude).collect();
    let low = altitudes.iter().copied().fold(f64::INFINITY, f64::min);
    let high = altitudes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (150_000.0..300_000.0).contains(&low) && (450_000.0..=560_000.0).contains(&high),
        "sampled altitudes {low:e} .. {high:e}"
    );
    let field = coefficients::FIELD;
    assert!(
        field.sigma() > 0.0 && field.sigma() < 1.0e-4,
        "sigma {:e}",
        field.sigma()
    );
    // The generated table is built by the unchecked constructor, so this is where the checked one's
    // refusals bind: the same six arguments, and they have to accept them.
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
    assert_eq!(coefficients::HARMONICS, (coefficients::DEGREE + 1).pow(2));
    // The geopotential conversion the generator used is the one the crate exports.
    assert!(geopotential_height(sdtm::BOUNDARY_ALTITUDE).abs() < 1.0e-9);

    // `SATELLITE_SCALE` is reported and never applied, so no density can be wrong because of it and
    // no other assertion in this file reads it. That is exactly why it needs one of its own: a number
    // nothing consumes can be regenerated wrong and stay wrong indefinitely. Each is the log offset
    // the fit gives one satellite's rows against GRACE-FO C's, and each must sit well inside the
    // fit's bound of 0.3: near the bound the scale is clamped, not a cross-calibration.
    assert_eq!(coefficients::SATELLITE_SCALE.len(), 7);
    for (name, scale) in coefficients::SATELLITE_SCALE {
        assert!(
            scale.abs() < 0.2,
            "{name} reads {:.2} % against GRACE-FO C, which is not a cross-calibration",
            100.0 * (libm::exp(scale) - 1.0)
        );
    }
}

/// Every row the generator sampled is inside the envelope it wrote, which holds the envelope to the
/// units of `SdtmPoint` and `SdtmDrivers`: an envelope written in kilometres would hold none of them.
/// And the envelope is the one the docs quote, rounded, so a refit that moves it fails here until
/// `docs/limitations.md` and `docs/verification.md` §10.28 move with it.
#[test]
fn the_envelope_holds_every_sampled_row_and_is_the_one_the_docs_quote() {
    let envelope = coefficients::ENVELOPE;
    for row in &rows() {
        assert!(
            envelope.contains(&row.point, &row.drivers),
            "{:?} {:?} outside {envelope:?}",
            row.point,
            row.drivers
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
