//! The tropospheric and ionospheric corrections composed onto an observable, and what their
//! elevation derivative is worth.
//!
//! `spacelab_estimation::modifier` is the composition layer: [`AtmosphericDelay`] hangs any
//! caller-supplied [`SlantDelay`] on any observable, and the layer's own tests prove the algebra
//! against a stand-in model. What they cannot prove is that a **real** model fits through the seam,
//! because the delay models live in `spacelab_gnss` and the estimation crate does not depend on it.
//! This file is where the two meet — the only place in the workspace that sees both.
//!
//! # The thing being measured
//!
//! [`SlantDelay::delay`] is stated over [`Scalar`] rather than `f64` on purpose. Until this ticket
//! every delay model in the workspace was `f64`-only, so the seam existed and nothing could pass
//! through it *with a derivative*: an `f64` delay lifted into a chain contributes a constant, and
//! a constant has partial **exactly zero**. A filter given such a correction would move its residual
//! and not its Jacobian, which is a silent modelling error rather than a loud one.
//!
//! The mapping functions now take their elevation over [`Scalar`], so the correction differentiates
//! with the geometry. Measured here, at 35°N 33°E on day 200 with GPT3's own weather:
//!
//! | elevation | slant delay | `∂delay/∂ε` |
//! |---|---|---|
//! | 5° | 25.520 m | −4.081 m/deg |
//! | 10° | 13.962 m | −1.287 m/deg |
//! | 30° | 5.008 m | −0.150 m/deg |
//! | 80° | 2.552 m | −0.0078 m/deg |
//!
//! Metres per degree at low elevation, which is the claim `modifier.rs` makes in prose and this
//! file turns into a number: between 80° and 5° the derivative grows by a factor of **521**, and
//! that growth is the whole reason the trait is generic.
//!
//! Chain-ruled onto the state the number is smaller, because `∂ε/∂r` carries a `1/ρ`: at this
//! file's 25 000 km slant range the two corrections move the range partial by **2.3e-6** per metre
//! of state, against a geometric partial of order one. Small — and ten orders of magnitude above
//! the `2.2e-16` an `f64` model's exactly-zero partial would leave behind. The comparison that
//! matters is with zero, not with one.
//!
//! # What is composed
//!
//! GPT3 supplies the surface weather **and** the VMF3 `a` coefficients at the station; Saastamoinen
//! turns the weather into the two zenith delays; VMF3 maps them onto the slant path. Nothing here
//! is a hand-invented constant — the whole chain descends from the station's coordinates and the
//! day of year. The ionospheric term is a vertical delay projected by the single-layer obliquity,
//! which is the shape every GNSS ionosphere model (Klobuchar, NeQuick-G, an IONEX map) shares.

#![allow(clippy::expect_used, clippy::unwrap_used)]
#![allow(clippy::doc_markdown)] // GPT3, VMF3, Saastamoinen, IONEX are names in prose

use spacelab_autodiff::{Dual, Scalar, jacobian};
use spacelab_core::{Angle, Length};
use spacelab_earth::{Geodetic, geodetic_to_itrf};
use spacelab_estimation::{
    AtmosphericDelay, Modified, Observable, Range, SlantDelay, Station, Topocentric, elevation,
    partials,
};
use spacelab_gnss::{
    gpt3, ionospheric_mapping_factor, saastamoinen_zenith_hydrostatic, saastamoinen_zenith_wet,
    vmf3_mapping,
};

/// Station: 35°N, 33°E, on the ellipsoid — the geometry `light_time_in_measurements.rs` uses.
const STATION_LATITUDE: f64 = 35.0;
const STATION_LONGITUDE: f64 = 33.0;
/// Mid-July, where the wet delay is largest and the seasonal terms are furthest from their mean.
const DAY_OF_YEAR: f64 = 200.0;
/// A vertical ionospheric delay of 5 m — mid-latitude daytime at L1, within a factor of two of any
/// real value. Its magnitude does not matter here; its *obliquity* does.
const VERTICAL_IONOSPHERE: f64 = 5.0;
const SHELL_HEIGHT: f64 = 350e3;
const EARTH_RADIUS: f64 = 6_378_137.0;
/// A slant range typical of a medium-Earth-orbit link, so the geometry is a real one.
const SLANT: f64 = 25_000e3;

const DEGREE: f64 = core::f64::consts::PI / 180.0;

/// GPT3 weather and VMF3 `a` coefficients, Saastamoinen zenith delays, VMF3 mapping — the
/// composition Orekit calls a `ViennaThree` estimated-troposphere model, as one [`SlantDelay`].
///
/// Everything that is a constant of the observation is resolved once, in [`Self::at`]: the station
/// coordinates, the day of year, the weather, the two zenith delays. Only the elevation reaches
/// [`SlantDelay::delay`], which is the layer's own rule.
struct ViennaTroposphere {
    latitude: f64,
    longitude: f64,
    height: f64,
    day_of_year: f64,
    ah: f64,
    aw: f64,
    zenith_hydrostatic: f64,
    zenith_wet: f64,
}

impl ViennaTroposphere {
    fn at(latitude: f64, longitude: f64, height: f64, day_of_year: f64) -> Self {
        let weather = gpt3(latitude, longitude, day_of_year, height);
        Self {
            latitude,
            longitude,
            height,
            day_of_year,
            ah: weather.ah,
            aw: weather.aw,
            // GPT3 reports pressures in pascals; Saastamoinen's coefficients are per hectopascal.
            zenith_hydrostatic: saastamoinen_zenith_hydrostatic(
                weather.pressure / 100.0,
                latitude,
                height,
            ),
            zenith_wet: saastamoinen_zenith_wet(
                weather.temperature,
                weather.water_vapor_pressure / 100.0,
            ),
        }
    }
}

impl SlantDelay for ViennaTroposphere {
    fn delay<S: Scalar>(&self, elevation: S, _azimuth: S) -> S {
        let (hydrostatic, wet) = vmf3_mapping(
            self.ah,
            self.aw,
            self.latitude,
            self.longitude,
            self.height,
            self.day_of_year,
            elevation,
        );
        hydrostatic * S::from_f64(self.zenith_hydrostatic) + wet * S::from_f64(self.zenith_wet)
    }
}

/// A vertical ionospheric delay projected onto the slant path by the single-layer obliquity — the
/// shape Klobuchar, NeQuick-G and an IONEX map all share once the vertical value is in hand.
struct ThinShellIonosphere {
    vertical: f64,
    shell_height: f64,
    earth_radius: f64,
}

impl SlantDelay for ThinShellIonosphere {
    fn delay<S: Scalar>(&self, elevation: S, _azimuth: S) -> S {
        ionospheric_mapping_factor(elevation, self.shell_height, self.earth_radius)
            * S::from_f64(self.vertical)
    }
}

fn station() -> (Topocentric, Station) {
    let geodetic = Geodetic {
        latitude: Angle::from_degrees(STATION_LATITUDE),
        longitude: Angle::from_degrees(STATION_LONGITUDE),
        height: Length::from_metres(0.0),
    };
    let itrf = geodetic_to_itrf(geodetic).vector;
    let position = [itrf.x, itrf.y, itrf.z];
    (
        Topocentric::from_geodetic(
            position,
            STATION_LATITUDE * DEGREE,
            STATION_LONGITUDE * DEGREE,
        ),
        Station::fixed(position),
    )
}

/// A satellite placed on a line of sight of exactly this elevation and azimuth, `SLANT` metres out.
///
/// Built in the station's own horizon triad, so the elevation is the one asked for rather than one
/// solved for — and every test reads it back through `spacelab_estimation::elevation` to confirm
/// that the construction and the measurement agree on which frame they are in.
fn satellite_at(topocentric: &Topocentric, elevation_rad: f64, azimuth_rad: f64) -> [f64; 6] {
    let (sin_e, cos_e) = elevation_rad.sin_cos();
    let (sin_a, cos_a) = azimuth_rad.sin_cos();
    let mut state = [0.0_f64; 6];
    for (axis, slot) in state.iter_mut().enumerate().take(3) {
        *slot = topocentric.position[axis]
            + SLANT
                * (cos_e * (cos_a * topocentric.north[axis] + sin_a * topocentric.east[axis])
                    + sin_e * topocentric.up[axis]);
    }
    state
}

/// The chain is the two models evaluated at the line of sight the state implies and added to the
/// range, to the bit — no reordering, no second geometry, no arithmetic the decorator invented.
#[test]
fn the_composed_correction_is_the_two_models_summed_at_the_line_of_sight() {
    let (topocentric, fixed) = station();
    let satellite = satellite_at(&topocentric, 12.0 * DEGREE, 47.0 * DEGREE);
    let none: [f64; 0] = [];

    let observed = elevation(&satellite, &topocentric);
    assert!(
        (observed - 12.0 * DEGREE).abs() < 1e-12,
        "the construction placed the satellite at {} degrees",
        observed / DEGREE
    );

    let troposphere = ViennaTroposphere::at(
        STATION_LATITUDE * DEGREE,
        STATION_LONGITUDE * DEGREE,
        0.0,
        DAY_OF_YEAR,
    );
    let ionosphere = ThinShellIonosphere {
        vertical: VERTICAL_IONOSPHERE,
        shell_height: SHELL_HEIGHT,
        earth_radius: EARTH_RADIUS,
    };
    let azimuth = spacelab_estimation::azimuth(&satellite, &topocentric);

    let chain = Modified::new(
        Range { station: fixed },
        AtmosphericDelay {
            topocentric,
            model: troposphere,
        },
    )
    .with(AtmosphericDelay {
        topocentric,
        model: ionosphere,
    });

    let by_hand = spacelab_estimation::range(&satellite, &fixed)
        + chain.observable.modifier.model.delay(observed, azimuth)
        + chain.modifier.model.delay(observed, azimuth);
    let value: f64 = chain.value(&satellite, &none);
    assert_eq!(
        value.to_bits(),
        by_hand.to_bits(),
        "the chained composition {value} against the hand-written one {by_hand}"
    );
}

/// **The point of the whole change.** A correction that reaches the filter as a constant moves the
/// residual and leaves the Jacobian alone; this one moves both, and by an amount nothing could
/// mistake for rounding.
///
/// The state partials of the decorated range are compared against the bare range's. Their
/// difference is what the two corrections contribute, and it is checked twice: against a central
/// difference of the composed observable, which is an independent route, and against zero, which is
/// what an `f64` delay model would have produced.
///
/// Measured at 12° elevation and a 25 000 km slant range: the largest component of the correction's
/// own partial is **2.3e-6**. The chain rule divides `∂delay/∂ε` by the range, so the number is
/// small in absolute terms and that is not the point — an `f64` model gives exactly `0`, and the
/// `f64` noise floor on a geometric partial of order one is `2.2e-16`. This is ten decades clear of
/// it, and it is a term a filter can see.
#[test]
fn the_correction_carries_a_partial_an_f64_model_would_have_lost() {
    let (topocentric, fixed) = station();
    let satellite = satellite_at(&topocentric, 12.0 * DEGREE, 47.0 * DEGREE);
    let none: [f64; 0] = [];

    let chain = Modified::new(
        Range { station: fixed },
        AtmosphericDelay {
            topocentric,
            model: ViennaTroposphere::at(
                STATION_LATITUDE * DEGREE,
                STATION_LONGITUDE * DEGREE,
                0.0,
                DAY_OF_YEAR,
            ),
        },
    )
    .with(AtmosphericDelay {
        topocentric,
        model: ThinShellIonosphere {
            vertical: VERTICAL_IONOSPHERE,
            shell_height: SHELL_HEIGHT,
            earth_radius: EARTH_RADIUS,
        },
    });

    let decorated = partials(&chain, &satellite, &none);
    let bare = partials(&Range { station: fixed }, &satellite, &none);

    let step = 100.0;
    let mut largest = 0.0_f64;
    for i in 0..3 {
        let (mut low, mut high) = (satellite, satellite);
        low[i] -= step;
        high[i] += step;
        let central = (chain.value(&high, &none) - chain.value(&low, &none)) / (2.0 * step);
        assert!(
            (decorated.state[i] - central).abs() <= 1e-6 * central.abs(),
            "∂/∂r[{i}] = {} against the central difference {central}",
            decorated.state[i]
        );
        largest = largest.max((decorated.state[i] - bare.state[i]).abs());
    }
    assert!(
        largest > 1e-9,
        "the corrections contributed {largest} to the state partial — an f64 model contributes 0"
    );

    // A delay is a function of position only, so the velocity partials are untouched — which is
    // also how a mistakenly velocity-dependent model would announce itself.
    for i in 3..6 {
        assert_eq!(
            decorated.state[i].to_bits(),
            bare.state[i].to_bits(),
            "a velocity partial moved"
        );
    }
}

/// `∂delay/∂ε` from automatic differentiation against a central difference, across the elevations a
/// pass actually spans, and the growth toward the horizon that makes the term matter.
///
/// The central difference is an independent route with its own `√ε` floor; the bound below is the
/// floor, not the agreement, which is measured at better than 1e-8 relative everywhere on the
/// sweep. The monotonicity assertion is the physical claim: a mapping function's derivative is
/// unbounded as `ε → 0`, so a correction that is a rounding term at the zenith is metres per degree
/// on the horizon, and an estimator that drops it drops the largest term in its own Jacobian.
#[test]
fn the_elevation_derivative_grows_without_bound_toward_the_horizon() {
    let troposphere = ViennaTroposphere::at(
        STATION_LATITUDE * DEGREE,
        STATION_LONGITUDE * DEGREE,
        0.0,
        DAY_OF_YEAR,
    );
    let azimuth = 47.0 * DEGREE;

    let mut previous = f64::INFINITY;
    for degrees in [80.0, 30.0, 10.0, 5.0] {
        let radians = degrees * DEGREE;
        let (_, jacobian_rows) = jacobian::<1, 1, _>(
            |e| [troposphere.delay(e[0], Dual::<1>::constant(azimuth))],
            &[radians],
        );
        let analytic = jacobian_rows[0][0];

        let step = 1e-5;
        let central = (troposphere.delay(radians + step, azimuth)
            - troposphere.delay(radians - step, azimuth))
            / (2.0 * step);
        assert!(
            (analytic - central).abs() <= 1e-6 * central.abs(),
            "at {degrees}° the AD derivative {analytic} disagrees with the difference {central}"
        );

        let per_degree = analytic * DEGREE;
        assert!(
            per_degree < 0.0,
            "at {degrees}° the delay grows with elevation ({per_degree} m/deg)"
        );
        assert!(
            per_degree.abs() > previous.abs() || previous.is_infinite(),
            "at {degrees}° the derivative {per_degree} m/deg did not exceed the {previous} above it"
        );
        previous = per_degree;
    }

    // The prose claim in `spacelab_estimation::modifier`, as a number: metres per degree at low
    // elevation, which is where a correction stops being a correction and starts being the model.
    let (_, low) = jacobian::<1, 1, _>(
        |e| [troposphere.delay(e[0], Dual::<1>::constant(azimuth))],
        &[5.0 * DEGREE],
    );
    assert!(
        (low[0][0] * DEGREE).abs() > 1.0,
        "at 5° the derivative is {} m/degree, not the metres per degree claimed",
        low[0][0] * DEGREE
    );
}
