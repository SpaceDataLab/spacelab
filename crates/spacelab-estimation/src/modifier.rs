//! Measurement corrections as a composable layer: one correction is a decorator over an observable,
//! and a decorated observable is itself an observable.
//!
//! [`measurement`](crate::measurement) computes geometry and nothing else — a range, an elevation,
//! a line-of-sight angle. What an instrument reads is that geometry plus every correction the
//! signal picked up on the way: tropospheric and ionospheric delay, the gravitational (Shapiro)
//! delay, a station or transponder bias. Before this module each caller added those at its own call
//! site, in its own order, and no two callers wrote it the same way.
//!
//! # Shape
//!
//! [`Observable`] is what an instrument reads, as a function of the six-element satellite state and
//! `K` **estimated measurement parameters** — the biases and offsets the filter solves for
//! alongside the orbit. [`Modifier`] is one correction, and it returns the *delta* it contributes
//! rather than a rewritten value. [`Modified`] joins the two and is itself an [`Observable`], so
//! corrections chain to any depth:
//!
//! ```text
//! Modified::new(Range{station}, ShapiroDelay{..}).with(EstimatedBias::<1>::new(0)?)
//!     = range(sat, station) + shapiro(sat, station) + parameters[0]
//! ```
//!
//! Composition is **static** — `Modified<Modified<Range, ShapiroDelay>, EstimatedBias<1>>` is one
//! monomorphised type — for the same reason the force models compose in tuples rather than in a
//! `Vec<Box<dyn Acceleration>>`: a trait object cannot carry a generic method, and every value here
//! is generic over [`Scalar`] so that the partials come from automatic differentiation rather than
//! from a hand-written Jacobian that can fall out of step with the value.
//!
//! # What this is not
//!
//! An options struct on each measurement function — `range(sat, station, &RangeOptions { troposphere,
//! ionosphere, bias, .. })` — was rejected. It puts every correction's parameters in every
//! measurement's signature, so adding one correction is a breaking change to all six observables:
//! the divergent-change smell, and precisely what the decorator avoids.
//!
//! # Where the atmospheric models are
//!
//! They are not here and they are not reachable from here. Saastamoinen, VMF1/VMF3, GPT3, Niell,
//! Klobuchar and NeQuick-G live in `spacelab_gnss`, which this crate does not depend on, so
//! [`AtmosphericDelay`] is generic over a caller-supplied [`SlantDelay`]: the caller owns the model
//! and the dependency edge, this layer owns the composition. The trait is stated over [`Scalar`]
//! rather than over `f64` on purpose — a delay evaluated at `f64` would enter the filter with a
//! zero partial, and the tropospheric mapping function's own elevation derivative is metres per
//! degree at low elevation, not a rounding term.
//!
//! # Gate
//!
//! `derivations/shapiro_delay.mac` reduces the Shapiro integral to the IERS closed form, and
//! `derivations/measurement_modifiers.mac` shows the PPN form is that formula at `γ = 1`, its
//! `γ` partial is `Δρ/(1 + γ)`, its position gradient is the closed form the AD is held to,
//! and the bias and drift columns are `1` and `Δt`; `derivations/measurement_modifiers.gp`
//! evaluates the defining integral along the module's leg. The gates are the module tests
//! `the_shapiro_modifier_is_the_path_integral_along_the_leg` (G2),
//! `the_shapiro_delay_matches_the_maxima_closed_form` (G3),
//! `the_shapiro_partials_are_the_closed_form_derivative` (G1) and
//! `an_estimated_gamma_is_the_delay_divided_by_one_plus_gamma` (G1).

use spacelab_autodiff::{Dual, Scalar, jacobian};

use crate::light_time::SPEED_OF_LIGHT;
use crate::measurement::{
    Station, Topocentric, azimuth, declination, elevation, range, range_rate, right_ascension,
};
use crate::radiometric::{CoronaSeries, PlasmaSignature, RadiometricError, solar_corona_delay};

/// Why a modifier could not be built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifierError {
    /// An estimated parameter was addressed past the end of the parameter vector. Reported at
    /// construction, so an assembled chain can only read a parameter that exists.
    ParameterOutOfRange {
        /// The index asked for.
        index: usize,
        /// The number of estimated measurement parameters the chain carries.
        parameters: usize,
    },
}

impl core::fmt::Display for ModifierError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ParameterOutOfRange { index, parameters } => write!(
                f,
                "estimated parameter {index} is past the {parameters} the measurement carries"
            ),
        }
    }
}

impl core::error::Error for ModifierError {}

/// What an instrument reads, as a function of the satellite state and the `K` estimated measurement
/// parameters.
///
/// Written once over any [`Scalar`], so [`partials`] gets the value and every partial from the same
/// source. `satellite` is the six-element state in the working frame, at the epoch the geometry
/// belongs to — light time is a choice of *epoch*, settled before this is called by
/// [`observed::reduce`](crate::observed::reduce), whose
/// [`partials_at_emission`](crate::observed::partials_at_emission) also corrects the row this
/// observable's partials produce for the state's own effect on the flight time.
pub trait Observable<const K: usize> {
    /// The observable's value at this state and these parameters.
    fn value<S: Scalar>(&self, satellite: &[S; 6], parameters: &[S; K]) -> S;
}

/// One correction to an observable: the amount it adds, in the observable's own units.
///
/// A delta rather than a rewritten value, because every correction in this layer is additive in the
/// observable — a delay adds metres to a range, a bias adds its own units to whatever it biases —
/// and a delta can be compared directly against the model it came from, which is how the
/// no-arithmetic-of-its-own property is tested.
pub trait Modifier<const K: usize> {
    /// The correction's contribution at this state and these parameters.
    fn delta<S: Scalar>(&self, satellite: &[S; 6], parameters: &[S; K]) -> S;
}

/// An observable decorated with one correction — itself an observable, which is what makes the
/// chain compose.
///
/// The sum is taken in application order. Floating-point addition is not associative, so the order
/// a caller writes the chain in is the order it is summed in, and a chain reproduces a hand-written
/// composition of the same terms in the same order to the bit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Modified<O, M> {
    /// The decorated observable.
    pub observable: O,
    /// The correction applied to it.
    pub modifier: M,
}

impl<O, M> Modified<O, M> {
    /// Decorate `observable` with one correction.
    #[must_use]
    pub const fn new(observable: O, modifier: M) -> Self {
        Self {
            observable,
            modifier,
        }
    }

    /// Add one more correction to the chain, applied after the ones already in it.
    ///
    /// Chaining lives here rather than on [`Observable`] so that a chain can be built without
    /// naming `K`: every observable and every modifier in this module is written for all `K`, so a
    /// method on the trait would be ambiguous at the point it is called and would have to be
    /// turbofished. Assembled this way, `K` is fixed once — where the chain meets the parameter
    /// vector it is evaluated at.
    #[must_use]
    pub const fn with<N>(self, modifier: N) -> Modified<Self, N> {
        Modified::new(self, modifier)
    }
}

impl<const K: usize, O: Observable<K>, M: Modifier<K>> Observable<K> for Modified<O, M> {
    fn value<S: Scalar>(&self, satellite: &[S; 6], parameters: &[S; K]) -> S {
        self.observable.value(satellite, parameters) + self.modifier.delta(satellite, parameters)
    }
}

/// Slant range to a station — [`measurement::range`](crate::measurement::range) lifted into the
/// layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    /// The observing station.
    pub station: Station,
}

impl<const K: usize> Observable<K> for Range {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        range(satellite, &self.station)
    }
}

/// Range rate — [`measurement::range_rate`](crate::measurement::range_rate) lifted into the layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeRate {
    /// The observing station.
    pub station: Station,
}

impl<const K: usize> Observable<K> for RangeRate {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        range_rate(satellite, &self.station)
    }
}

/// Topocentric right ascension — [`measurement::right_ascension`](crate::measurement::right_ascension)
/// lifted into the layer. Cyclic in `2π`; the residual still has to be wrapped by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RightAscension {
    /// The observing station.
    pub station: Station,
}

impl<const K: usize> Observable<K> for RightAscension {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        right_ascension(satellite, &self.station)
    }
}

/// Topocentric declination — [`measurement::declination`](crate::measurement::declination) lifted
/// into the layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Declination {
    /// The observing station.
    pub station: Station,
}

impl<const K: usize> Observable<K> for Declination {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        declination(satellite, &self.station)
    }
}

/// Elevation above the local horizon — [`measurement::elevation`](crate::measurement::elevation)
/// lifted into the layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elevation {
    /// The station and its horizon frame.
    pub topocentric: Topocentric,
}

impl<const K: usize> Observable<K> for Elevation {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        elevation(satellite, &self.topocentric)
    }
}

/// Azimuth from local north — [`measurement::azimuth`](crate::measurement::azimuth) lifted into the
/// layer. Cyclic in `2π`; the residual still has to be wrapped by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Azimuth {
    /// The station and its horizon frame.
    pub topocentric: Topocentric,
}

impl<const K: usize> Observable<K> for Azimuth {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        azimuth(satellite, &self.topocentric)
    }
}

/// Shapiro gravitational delay of one signal leg, metres — IERS Conventions (2010) §11.
///
/// `r_emitter` and `r_receiver` are the geocentric distances of the two endpoints and `path` the
/// straight-line distance between them, all metres; `gm` is the central body's gravitational
/// parameter, m³/s².
///
/// # Derivation
///
/// The delay is the coordinate-time excess `(2GM/c²)∫dl/r` along the leg. `derivations/shapiro_delay.mac`
/// verifies in Maxima that the antiderivative `log(l + b + √(l² + 2bl + r₁²))` differentiates back
/// to `1/r(l)` (residual exactly `0`) and that evaluating it between the endpoints reduces to
/// `log((r₁ + r₂ + p)/(r₁ + r₂ − p))` (difference from the IERS form exactly `0`). The same script
/// integrates the three-dimensional geometry numerically: quadrature minus closed form is `0.0 m`
/// at a 500 km zenith pass, `1.1e-19 m` at 10° elevation and `-1.1e-18 m` at LAGEOS.
///
/// # Size, measured
///
/// One leg, from the same script: **0.669 mm** to a 500 km satellite at the zenith, **2.281 mm** at
/// 10° elevation, **9.516 mm** to LAGEOS at 12 270 km, **12.66 mm** to GPS at 20 200 km. A two-way
/// system doubles those in flight time, but a two-way range published as a one-way range —
/// `c(τ_up + τ_down)/2`, which is how an ILRS normal point is formed — carries one leg. The
/// commonly quoted "~2 cm two-way" belongs to GNSS altitude (25.3 mm), not to LEO, where the
/// two-way delay is 1.34 mm at the zenith and 4.56 mm at 10°.
#[must_use]
pub fn shapiro_delay<S: Scalar>(r_emitter: S, r_receiver: S, path: S, gm: f64) -> S {
    let sum = r_emitter + r_receiver;
    S::from_f64(2.0 * gm / (SPEED_OF_LIGHT * SPEED_OF_LIGHT)) * ((sum + path) / (sum - path)).ln()
}

/// The Shapiro delay with the PPN parameter `γ` written out rather than assumed.
///
/// ```text
/// Δρ = (1 + γ) GM/c² · ln((r_e + r_r + s)/(r_e + r_r − s))
/// ```
///
/// [`shapiro_delay`] is this at `γ = 1`, which is general relativity and what every other use in
/// this workspace wants. It is separate because `γ` is the one parameter of the measurement model
/// that is *solved for* rather than adopted: it is the quantity Cassini measured at superior
/// conjunction — `γ − 1 = (2.1 ± 2.3) × 10⁻⁵`, Bertotti, Iess & Tortora, *Nature* **425** (2003) —
/// and that measurement is exactly `∂ρ/∂γ` over a tracking arc. A `γ` baked into the constant `2`
/// has partial zero and cannot be estimated at all.
///
/// The two functions are one formula and the tests hold them to that: `shapiro_delay_ppn(.., 1.0)`
/// is bit-identical to `shapiro_delay`, and the partial is the delay itself divided by `1 + γ`.
#[must_use]
pub fn shapiro_delay_ppn<S: Scalar>(r_emitter: S, r_receiver: S, path: S, gm: f64, gamma: S) -> S {
    let sum = r_emitter + r_receiver;
    (gamma + S::from_f64(1.0))
        * S::from_f64(gm / (SPEED_OF_LIGHT * SPEED_OF_LIGHT))
        * ((sum + path) / (sum - path)).ln()
}

/// The Shapiro delay as a modifier on a range: [`shapiro_delay`] over the leg between the station
/// and the satellite.
///
/// The state and the station must be in the same **geocentric** frame; only the two geocentric
/// distances and the straight-line path enter, so a rotating Earth-fixed frame and an inertial one
/// give the same delay.
///
/// `spacelab_slr::shapiro_delay` is the same equation at `f64`, for the two-way SLR model. This one
/// is generic over [`Scalar`] because it is differentiated: an `f64` delay would enter the filter
/// with a zero partial. They are two implementations of one formula and can drift; what stops it is
/// that `derivations/shapiro_delay.mac` pins this one to a bigfloat reference and computes the
/// 9.516 mm LAGEOS leg the SLR side's own test brackets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapiroDelay {
    /// Station position in the working frame, metres.
    pub station: [f64; 3],
    /// Gravitational parameter of the central body, m³/s².
    pub gm: f64,
}

impl<const K: usize> Modifier<K> for ShapiroDelay {
    fn delta<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        let r_satellite = (satellite[0] * satellite[0]
            + satellite[1] * satellite[1]
            + satellite[2] * satellite[2])
            .sqrt();
        let r_station = S::from_f64(libm::sqrt(
            self.station[0] * self.station[0]
                + self.station[1] * self.station[1]
                + self.station[2] * self.station[2],
        ));
        let path = range(satellite, &Station::fixed(self.station));
        shapiro_delay(r_satellite, r_station, path, self.gm)
    }
}

/// A bias of known size, applied rather than solved for.
///
/// This is the ILRS Data Handling File case — a station whose range bias the network publishes, in
/// the observable's own units. Where the network instead flags a bias to be *estimated*, the
/// modifier is [`EstimatedBias`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bias {
    /// The bias added to the observable, in its units.
    pub value: f64,
}

impl<const K: usize> Modifier<K> for Bias {
    fn delta<S: Scalar>(&self, _satellite: &[S; 6], _parameters: &[S; K]) -> S {
        S::from_f64(self.value)
    }
}

/// A bias the filter solves for: one of the `K` estimated measurement parameters, added to the
/// observable.
///
/// The partial `∂observable/∂parameter` is then exactly `1`, from AD, and the parameter gets a
/// column in the Jacobian row like any state direction. An estimator that can only subtract a fixed
/// bias cannot detect one that changed; this is the difference between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EstimatedBias<const K: usize> {
    index: usize,
}

impl<const K: usize> EstimatedBias<K> {
    /// The bias carried by estimated parameter `index`.
    ///
    /// # Errors
    ///
    /// [`ModifierError::ParameterOutOfRange`] if `index` is not one of the `K` parameters. Checked
    /// once here so that evaluation cannot address a parameter that does not exist.
    pub const fn new(index: usize) -> Result<Self, ModifierError> {
        if index < K {
            Ok(Self { index })
        } else {
            Err(ModifierError::ParameterOutOfRange {
                index,
                parameters: K,
            })
        }
    }

    /// Which estimated parameter this bias reads.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }
}

impl<const K: usize> Modifier<K> for EstimatedBias<K> {
    fn delta<S: Scalar>(&self, _satellite: &[S; 6], parameters: &[S; K]) -> S {
        parameters[self.index]
    }
}

/// The Shapiro delay with `γ` taken from an estimated parameter, so the arc measures it.
///
/// The geometry is [`ShapiroDelay`]'s; the only difference is where `γ` comes from, and that
/// difference is the whole point — with `γ` in the parameter vector the Jacobian gains a column
/// `∂ρ/∂γ = Δρ/(1 + γ)`, which at Cassini's 2002 conjunction geometry is **19.7 km** per unit `γ`
/// on the one-way range, measured by this module's own test. That is what makes the parameter
/// observable at all: Bertotti's `± 2.3 × 10⁻⁵` is 45 cm of range against it.
///
/// This is Tudat's `ppn_parameter_gamma`, and it is the only one of its ~80 estimable parameters
/// that belongs in *this* layer: the rest — `ppn_parameter_beta`, the Love numbers, the free-core-
/// nutation rate, the Yarkovsky coefficient, the gas–surface accommodation coefficients — enter
/// through the force model or the Earth-orientation model, not through the measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimatedShapiroDelay<const K: usize> {
    /// Station position in the working frame, metres.
    pub station: [f64; 3],
    /// Gravitational parameter of the central body, m³/s².
    pub gm: f64,
    index: usize,
}

impl<const K: usize> EstimatedShapiroDelay<K> {
    /// The Shapiro delay whose `γ` is estimated parameter `index`.
    ///
    /// # Errors
    ///
    /// [`ModifierError::ParameterOutOfRange`] if `index` is not one of the `K` parameters.
    pub const fn new(station: [f64; 3], gm: f64, index: usize) -> Result<Self, ModifierError> {
        if index < K {
            Ok(Self { station, gm, index })
        } else {
            Err(ModifierError::ParameterOutOfRange {
                index,
                parameters: K,
            })
        }
    }

    /// Which estimated parameter carries `γ`.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }
}

impl<const K: usize> Modifier<K> for EstimatedShapiroDelay<K> {
    fn delta<S: Scalar>(&self, satellite: &[S; 6], parameters: &[S; K]) -> S {
        let r_satellite = (satellite[0] * satellite[0]
            + satellite[1] * satellite[1]
            + satellite[2] * satellite[2])
            .sqrt();
        let r_station = S::from_f64(libm::sqrt(
            self.station[0] * self.station[0]
                + self.station[1] * self.station[1]
                + self.station[2] * self.station[2],
        ));
        let path = range(satellite, &Station::fixed(self.station));
        shapiro_delay_ppn(
            r_satellite,
            r_station,
            path,
            self.gm,
            parameters[self.index],
        )
    }
}

/// A clock offset that drifts across an arc: `bias + rate · Δt`, both estimated.
///
/// The arcwise counterpart of [`EstimatedBias`], and Tudat's arcwise clock/time-bias parameters.
/// A station clock is not a constant over a pass — it is a polynomial, and the linear term is the
/// one a single arc can separate from the orbit, because a bias that is constant over the arc is
/// degenerate with the along-track state and a bias that ramps is not.
///
/// `elapsed` is a *constant of the observation*, like the meteorology [`AtmosphericDelay`] carries:
/// the modifier is built per observation with the seconds since the arc's own reference epoch
/// already in it, so that [`Observable::value`] keeps its signature and the layer keeps its rule
/// that everything not a function of the state lives in the implementor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EstimatedDrift<const K: usize> {
    /// Seconds from the arc's reference epoch to this observation.
    pub elapsed: f64,
    bias: usize,
    rate: usize,
}

impl<const K: usize> EstimatedDrift<K> {
    /// The drift carried by parameters `bias` and `rate`, evaluated `elapsed` seconds into the arc.
    ///
    /// # Errors
    ///
    /// [`ModifierError::ParameterOutOfRange`] naming whichever index is past the `K` parameters.
    /// The two are checked in order, so a call with both wrong reports the first.
    pub const fn new(elapsed: f64, bias: usize, rate: usize) -> Result<Self, ModifierError> {
        if bias >= K {
            return Err(ModifierError::ParameterOutOfRange {
                index: bias,
                parameters: K,
            });
        }
        if rate >= K {
            return Err(ModifierError::ParameterOutOfRange {
                index: rate,
                parameters: K,
            });
        }
        Ok(Self {
            elapsed,
            bias,
            rate,
        })
    }

    /// Which estimated parameter carries the constant term.
    #[must_use]
    pub const fn bias_index(&self) -> usize {
        self.bias
    }

    /// Which estimated parameter carries the rate.
    #[must_use]
    pub const fn rate_index(&self) -> usize {
        self.rate
    }
}

impl<const K: usize> Modifier<K> for EstimatedDrift<K> {
    fn delta<S: Scalar>(&self, _satellite: &[S; 6], parameters: &[S; K]) -> S {
        parameters[self.bias] + parameters[self.rate] * S::from_f64(self.elapsed)
    }
}

/// The solar-corona range correction as a modifier — [`solar_corona_delay`] on the station-to-
/// satellite leg, with both endpoints referred to the Sun.
///
/// [`crate::radiometric`] computes the path integral; this is what puts it in the chain beside the
/// Shapiro delay and the troposphere, which is what #325 is for. The series and the frequency are
/// validated once at construction, so evaluation carries no error the caller has to handle on every
/// observation.
///
/// # Occultation
///
/// A line of sight whose closest approach to the Sun's centre is inside
/// [`SOLAR_RADIUS`](crate::radiometric::SOLAR_RADIUS) is not a
/// link, and the correction has no value there. [`Self::occulted`] answers that question directly
/// and [`Modifier::delta`] returns NaN — the same "no answer" this workspace uses everywhere a
/// total function has no total domain. Screen with `occulted` before scheduling the observation;
/// a NaN reaching the residual is a scheduling bug, and it is visible rather than silent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarCoronaDelay<'a> {
    /// The station, in the same geocentric frame as the state, metres.
    pub station: [f64; 3],
    /// The Sun, in that frame at this observation's epoch, metres.
    pub sun: [f64; 3],
    series: CoronaSeries<'a>,
    frequency: f64,
    signature: PlasmaSignature,
}

impl<'a> SolarCoronaDelay<'a> {
    /// The corona correction for a link on `frequency` hertz through `series`.
    ///
    /// # Errors
    ///
    /// [`RadiometricError::NonPositiveFrequency`] if the frequency is not positive and finite. The
    /// series validates itself, at [`CoronaSeries::new`].
    pub const fn new(
        station: [f64; 3],
        sun: [f64; 3],
        series: CoronaSeries<'a>,
        frequency: f64,
        signature: PlasmaSignature,
    ) -> Result<Self, RadiometricError> {
        if frequency <= 0.0 || !frequency.is_finite() {
            return Err(RadiometricError::NonPositiveFrequency { hertz: frequency });
        }
        Ok(Self {
            station,
            sun,
            series,
            frequency,
            signature,
        })
    }

    /// Whether the Sun stands between the station and the satellite closely enough that there is no
    /// link — the geometry [`solar_corona_delay`] refuses.
    #[must_use]
    pub fn occulted(&self, satellite: &[f64; 6]) -> bool {
        let station = self.heliocentric(&[self.station[0], self.station[1], self.station[2]]);
        let target = self.heliocentric(&[satellite[0], satellite[1], satellite[2]]);
        solar_corona_delay(
            &self.series,
            &target,
            &station,
            self.frequency,
            self.signature,
        )
        .is_err()
    }

    /// A geocentric position referred to the Sun.
    fn heliocentric<S: Scalar>(&self, position: &[S; 3]) -> [S; 3] {
        [
            position[0] - S::from_f64(self.sun[0]),
            position[1] - S::from_f64(self.sun[1]),
            position[2] - S::from_f64(self.sun[2]),
        ]
    }
}

impl<const K: usize> Modifier<K> for SolarCoronaDelay<'_> {
    fn delta<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        let station = self.heliocentric(&[
            S::from_f64(self.station[0]),
            S::from_f64(self.station[1]),
            S::from_f64(self.station[2]),
        ]);
        let target = self.heliocentric(&[satellite[0], satellite[1], satellite[2]]);
        solar_corona_delay(
            &self.series,
            &target,
            &station,
            self.frequency,
            self.signature,
        )
        .unwrap_or_else(|_| S::from_f64(f64::NAN))
    }
}

/// A one-way path delay along the line of sight, metres — the interface the tropospheric and
/// ionospheric models are attached through.
///
/// Generic over [`Scalar`] because the delay is a function of the *state*: elevation and azimuth
/// both move with the orbit, and a mapping function's elevation derivative is the largest term in
/// the correction's own Jacobian at low elevation. Everything that is a constant of the observation
/// — the epoch, the meteorology, the station's coordinates, the mapping-function coefficients —
/// belongs in the implementor, which the caller constructs per observation.
pub trait SlantDelay {
    /// Delay in metres for a line of sight at `elevation` above the horizon and `azimuth` from
    /// north, both radians.
    fn delay<S: Scalar>(&self, elevation: S, azimuth: S) -> S;
}

/// A [`SlantDelay`] attached to an observable, evaluated at the line of sight the state implies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtmosphericDelay<D> {
    /// The station and its horizon frame — the delay is a function of the local elevation.
    pub topocentric: Topocentric,
    /// The delay model.
    pub model: D,
}

impl<const K: usize, D: SlantDelay> Modifier<K> for AtmosphericDelay<D> {
    fn delta<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        self.model.delay(
            elevation(satellite, &self.topocentric),
            azimuth(satellite, &self.topocentric),
        )
    }
}

/// One observation's value and its partials, from automatic differentiation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Partials<const K: usize> {
    /// The modified observable's value.
    pub value: f64,
    /// `∂value/∂state(t)` — chain it through the state transition matrix to reach the epoch state.
    pub state: [f64; 6],
    /// `∂value/∂parameter` for each estimated measurement parameter. These do not go through the
    /// STM: a bias is not propagated, so the column is the partial itself.
    pub parameters: [f64; K],
}

/// Evaluate a modified observable and its partials at one observation.
///
/// Two AD passes, not one: the seeded dimension is a const generic, `6 + K` cannot be spelled on
/// stable Rust, and a second evaluation of a measurement geometry is nothing against the force
/// evaluation that produced the state it is given. Both passes read the same source, so the value
/// they agree on is the value the partials belong to.
#[must_use]
pub fn partials<const K: usize, O: Observable<K>>(
    observable: &O,
    satellite: &[f64; 6],
    parameters: &[f64; K],
) -> Partials<K> {
    let mut lifted_parameters = [Dual::<6>::constant(0.0); K];
    for (slot, p) in lifted_parameters.iter_mut().zip(parameters) {
        *slot = Dual::constant(*p);
    }
    let (value, state_jacobian) =
        jacobian::<6, 1, _>(|s| [observable.value(s, &lifted_parameters)], satellite);

    let mut lifted_state = [Dual::<K>::constant(0.0); 6];
    for (slot, x) in lifted_state.iter_mut().zip(satellite) {
        *slot = Dual::constant(*x);
    }
    let (_, parameter_jacobian) =
        jacobian::<K, 1, _>(|p| [observable.value(&lifted_state, p)], parameters);

    let mut state = [0.0; 6];
    for (slot, row) in state.iter_mut().zip(&state_jacobian) {
        *slot = row[0];
    }
    let mut columns = [0.0; K];
    for (slot, row) in columns.iter_mut().zip(&parameter_jacobian) {
        *slot = row[0];
    }
    Partials {
        value: value[0],
        state,
        parameters: columns,
    }
}

#[cfg(test)]
mod tests {
    // The Maxima reference below is written at the precision it was computed to, not at the
    // shortest `f64` that round-trips: a truncated constant cannot be checked against the source.
    #![allow(
        clippy::excessive_precision,
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::many_single_char_names
    )]

    use spacelab_testkit::assert_within_ulps;

    use super::*;
    use crate::radiometric::{AKSIM_PAVLOV_2022, SOLAR_RADIUS};

    /// IERS 2010 conventional GM, m³/s² — the value `derivations/shapiro_delay.mac` uses.
    const EARTH_GM: f64 = 3.986_004_418e14;
    const EARTH_RADIUS: f64 = 6_378_137.0;

    fn station() -> Station {
        Station::fixed([4_075_530.0, 931_781.0, 4_801_618.0])
    }

    fn topocentric() -> Topocentric {
        Topocentric::from_geodetic([4_075_530.0, 931_781.0, 4_801_618.0], 0.8977, 0.2251)
    }

    fn satellite() -> [f64; 6] {
        [
            3_961_000.0,
            2_614_000.0,
            5_215_000.0,
            -3_140.0,
            6_190.0,
            -0.7e3,
        ]
    }

    /// Lifting an observable into the layer adds no arithmetic: every one of the six reproduces the
    /// free function it wraps to the bit, at a state where no component is zero or shared.
    #[test]
    fn every_observable_reproduces_the_free_function_to_the_ulp() {
        let (sat, st, tp) = (satellite(), station(), topocentric());
        let none: [f64; 0] = [];
        assert_within_ulps(
            Range { station: st }.value(&sat, &none),
            range(&sat, &st),
            0,
            "range",
        );
        assert_within_ulps(
            RangeRate { station: st }.value(&sat, &none),
            range_rate(&sat, &st),
            0,
            "range rate",
        );
        assert_within_ulps(
            RightAscension { station: st }.value(&sat, &none),
            right_ascension(&sat, &st),
            0,
            "right ascension",
        );
        assert_within_ulps(
            Declination { station: st }.value(&sat, &none),
            declination(&sat, &st),
            0,
            "declination",
        );
        assert_within_ulps(
            Elevation { topocentric: tp }.value(&sat, &none),
            elevation(&sat, &tp),
            0,
            "elevation",
        );
        assert_within_ulps(
            Azimuth { topocentric: tp }.value(&sat, &none),
            azimuth(&sat, &tp),
            0,
            "azimuth",
        );
    }

    /// A chain is its terms summed in the order they were added, and nothing else. Written against
    /// the hand-composed expression the layer exists to replace, at zero ulps: if the decorator
    /// carried any arithmetic of its own — a reordering, a fused multiply, a second `sqrt` — this is
    /// where it would show.
    #[test]
    fn a_chain_is_the_hand_written_composition_to_the_ulp() {
        let (sat, st) = (satellite(), station());
        let none: [f64; 0] = [];
        let shapiro = ShapiroDelay {
            station: st.position,
            gm: EARTH_GM,
        };
        let bias = Bias { value: -0.012 };

        let chained = Modified::new(Range { station: st }, shapiro)
            .with(bias)
            .value(&sat, &none);
        let by_hand = range(&sat, &st) + shapiro.delta(&sat, &none) + bias.delta(&sat, &none);
        assert_within_ulps(chained, by_hand, 0, "the chained composition");
    }

    /// The closed form against Maxima at 40 digits (G3).
    ///
    /// `derivations/shapiro_delay.mac` proves the antiderivative and the reduction to the IERS form
    /// symbolically (both residuals exactly 0) and then evaluates it as a bigfloat at
    /// `(r₁, r₂, p) = (7000000, 7500000, 900000)` m with the same GM:
    /// `1.102527717326717568931198e-3` m.
    #[test]
    fn the_shapiro_delay_matches_the_maxima_closed_form() {
        let maxima = 1.102_527_717_326_717_568_931_198e-3;
        assert_within_ulps(
            shapiro_delay(7.0e6, 7.5e6, 9.0e5, EARTH_GM),
            maxima,
            1,
            "the Shapiro delay against Maxima",
        );
    }

    /// Coincident endpoints have no path and no delay — `log(1)`, exactly zero, not a small
    /// residue that a cancellation happened to leave behind.
    #[test]
    fn the_shapiro_delay_vanishes_when_the_endpoints_coincide() {
        assert_within_ulps(
            shapiro_delay(EARTH_RADIUS, EARTH_RADIUS, 0.0, EARTH_GM),
            0.0,
            0,
            "the delay over no path",
        );
    }

    /// The modifier on the module's three-dimensional leg is the defining integral (G2):
    /// `(2GM/c²) ∫ dl/r` along the straight station–satellite line, taken by `intnum` at 60
    /// digits in `derivations/measurement_modifiers.gp` with no logarithm in it, and equal to the
    /// closed form of `derivations/measurement_modifiers.mac` to 1e-79 m. Measured: 2 ulps, the
    /// rounding of the ratio and the logarithm.
    #[test]
    #[allow(clippy::unreadable_literal)]
    fn the_shapiro_modifier_is_the_path_integral_along_the_leg() {
        let none: [f64; 0] = [];
        let modifier = ShapiroDelay {
            station: station().position,
            gm: EARTH_GM,
        };
        assert_within_ulps(
            modifier.delta(&satellite(), &none),
            2.3082308593837045843267125633210189848631e-3,
            2,
            "the Shapiro modifier against the path integral",
        );
    }

    /// The modifier is the closed form evaluated at the leg's own geometry, to the bit: the two
    /// geocentric distances and the straight-line path, and no fourth quantity.
    #[test]
    fn the_shapiro_modifier_is_the_closed_form_at_the_leg_geometry() {
        let (sat, st) = (satellite(), station());
        let none: [f64; 0] = [];
        let modifier = ShapiroDelay {
            station: st.position,
            gm: EARTH_GM,
        };
        let r_sat = libm::sqrt(sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2]);
        let p = st.position;
        let r_sta = libm::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        assert_within_ulps(
            modifier.delta(&sat, &none),
            shapiro_delay(r_sat, r_sta, range(&sat, &st), EARTH_GM),
            0,
            "the modifier against the closed form",
        );
    }

    /// The correction is differentiated, not pasted on (G1): the closed-form gradient below is
    /// `derivations/measurement_modifiers.mac` identity 3.
    ///
    /// `∂D/∂rᵢ = k[(∂s + ∂p)/(s + p) − (∂s − ∂p)/(s − p)]` with `s = r_sat + r_sta`,
    /// `∂s/∂rᵢ = rᵢ/r_sat`, `∂p/∂rᵢ = (rᵢ − stationᵢ)/p` and `k = 2GM/c²`. The AD partial of the
    /// delay agrees with it to **6.7e-16 relative** on a partial of size 1.9e-10 per metre.
    ///
    /// The chain's own partial is then checked to carry it, and *that* comparison has a floor the
    /// correction has nothing to do with: the geometric partial is of order 1, the delay's is
    /// 1e-10, and differencing two `f64`s ten decades apart leaves about `ε/2` of the larger —
    /// 3.4e-17 was measured here, which is why the bound below is 1e-16 and not the 1e-15 relative
    /// the AD itself reaches.
    #[test]
    fn the_shapiro_partials_are_the_closed_form_derivative() {
        let (sat, st) = (satellite(), station());
        let none: [f64; 0] = [];
        let modifier = ShapiroDelay {
            station: st.position,
            gm: EARTH_GM,
        };

        let r_sat = libm::sqrt(sat[0] * sat[0] + sat[1] * sat[1] + sat[2] * sat[2]);
        let q = st.position;
        let r_sta = libm::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
        let path = range(&sat, &st);
        let s = r_sat + r_sta;
        let k = 2.0 * EARTH_GM / (SPEED_OF_LIGHT * SPEED_OF_LIGHT);

        let (_, delay_jacobian) = jacobian::<6, 1, _>(|z| [modifier.delta(z, &[])], &sat);
        let bare = partials(&Range { station: st }, &sat, &none);
        let decorated = partials(&Modified::new(Range { station: st }, modifier), &sat, &none);

        for i in 0..3 {
            let ds = sat[i] / r_sat;
            let dp = (sat[i] - q[i]) / path;
            let want = k * ((ds + dp) / (s + path) - (ds - dp) / (s - path));
            assert!(
                want.abs() > 1e-11,
                "the delay partial {want} is too small to test"
            );
            assert!(
                (delay_jacobian[i][0] - want).abs() <= 1e-14 * want.abs(),
                "∂(Shapiro)/∂r[{i}] = {} against the closed form {want}",
                delay_jacobian[i][0]
            );
            let carried = decorated.state[i] - bare.state[i];
            assert!(
                (carried - want).abs() < 1e-16,
                "the chain carried {carried} of the delay partial {want}"
            );
        }
        // A delay does not depend on velocity, so those partials are untouched.
        for i in 3..6 {
            assert_within_ulps(decorated.state[i], bare.state[i], 0, "a velocity partial");
        }
    }

    /// An estimated bias is a unit column in the Jacobian and changes nothing else: the value moves
    /// by the bias, the state partials do not move at all.
    #[test]
    fn an_estimated_bias_is_a_unit_column_and_leaves_the_state_partials_alone() {
        let (sat, st) = (satellite(), station());
        let bias = EstimatedBias::<2>::new(1).expect("index 1 of 2");
        assert_eq!(bias.index(), 1);
        let observable = Modified::new(Range { station: st }, bias);

        let p = partials(&observable, &sat, &[0.0, 4.75]);
        let bare = partials(&Range { station: st }, &sat, &[0.0, 4.75]);
        assert_within_ulps(p.value, bare.value + 4.75, 0, "the biased value");
        assert_within_ulps(p.parameters[1], 1.0, 0, "∂range/∂bias");
        assert_within_ulps(p.parameters[0], 0.0, 0, "∂range/∂(unused parameter)");
        for i in 0..6 {
            assert_within_ulps(p.state[i], bare.state[i], 0, "a state partial");
        }
    }

    /// A parameter index past the vector is refused at construction, so evaluation cannot read one
    /// that does not exist.
    #[test]
    fn an_estimated_parameter_past_the_vector_is_refused() {
        assert_eq!(
            EstimatedBias::<2>::new(2),
            Err(ModifierError::ParameterOutOfRange {
                index: 2,
                parameters: 2,
            })
        );
        assert!(EstimatedBias::<2>::new(1).is_ok());
        assert!(EstimatedBias::<0>::new(0).is_err());
    }

    /// The index a bias reports is the parameter it reads, at every index the vector has.
    ///
    /// Checked across the whole vector rather than at one index: a reader that answered with a fixed
    /// index would agree with a single-index test and disagree with the parameter actually read.
    #[test]
    fn an_estimated_bias_reports_the_parameter_it_reads() {
        let parameters = [-2.0, 3.5, 7.25];
        for index in 0..parameters.len() {
            let bias = EstimatedBias::<3>::new(index).expect("index within the vector");
            assert_eq!(bias.index(), index);
            assert_within_ulps(
                bias.delta(&[0.0; 6], &parameters),
                parameters[index],
                0,
                "the bias read from its own parameter",
            );
        }
    }

    /// The one refusal says which index was asked for and how many the chain carries — the two
    /// numbers a caller needs to fix the call.
    #[test]
    fn the_out_of_range_refusal_names_both_numbers() {
        let error = ModifierError::ParameterOutOfRange {
            index: 3,
            parameters: 2,
        };
        let expected = "estimated parameter 3 is past the 2 the measurement carries";
        assert_eq!(format!("{error}"), expected);
        let dynamic: &dyn core::error::Error = &error;
        assert_eq!(format!("{dynamic}"), expected);
    }

    /// A zenith delay mapped by `1/sin(elevation)` with an azimuthal gradient — a stand-in for the
    /// models in `spacelab_gnss`, which this crate cannot reach, exercising both arguments.
    struct MappedDelay {
        zenith: f64,
        gradient: f64,
    }

    impl SlantDelay for MappedDelay {
        fn delay<S: Scalar>(&self, elevation: S, azimuth: S) -> S {
            (S::from_f64(self.zenith) + S::from_f64(self.gradient) * azimuth.cos())
                * elevation.sin().recip()
        }
    }

    /// A supplied delay model reaches the observable through the line of sight it implies, and its
    /// elevation dependence survives into the partials.
    ///
    /// The value is checked to the bit against the model called directly. The partial is checked
    /// against a central difference, which is an independent route with its own `√ε` floor: measured
    /// **1.8e-10 relative**, on partials of order 1e-5 m/m and a delay of 8.23 m at the 16.3°
    /// elevation of this geometry. The bound is left at 1e-7, the floor the difference itself has.
    #[test]
    fn a_supplied_delay_reaches_the_observable_through_the_line_of_sight() {
        let (sat, tp) = (satellite(), topocentric());
        let none: [f64; 0] = [];
        let modifier = AtmosphericDelay {
            topocentric: tp,
            model: MappedDelay {
                zenith: 2.31,
                gradient: 0.04,
            },
        };
        assert_within_ulps(
            modifier.delta(&sat, &none),
            modifier
                .model
                .delay(elevation(&sat, &tp), azimuth(&sat, &tp)),
            0,
            "the delay at the line of sight",
        );

        let observable = Modified::new(Elevation { topocentric: tp }, modifier);
        let p = partials(&observable, &sat, &none);
        let h = 1.0;
        for i in 0..3 {
            let (mut lo, mut hi) = (sat, sat);
            lo[i] -= h;
            hi[i] += h;
            let central = (observable.value(&hi, &none) - observable.value(&lo, &none)) / (2.0 * h);
            assert!(
                (p.state[i] - central).abs() <= 1e-7 * central.abs(),
                "∂/∂r[{i}] = {} against the central difference {central}",
                p.state[i]
            );
        }
    }

    /// IAU 2015 Resolution B3 nominal solar mass parameter, m³/s² — an exact defining constant.
    const SUN_GM: f64 = 1.327_124_4e20;
    /// IAU 2012 Resolution B2 astronomical unit, metres — exact.
    const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

    /// Earth and a deep-space probe at superior conjunction, both referred to the Sun at the origin,
    /// on a line of sight passing `impact` metres from the Sun's centre.
    ///
    /// The probe is at Cassini's heliocentric distance during the 2002 conjunction, 8.43 AU. The
    /// separation angle is solved for rather than assumed: `b = r_e r_p sin δ / p` with `p` the
    /// straight-line path, whose own dependence on `δ` the fixed point below removes. Every test
    /// that uses this reads the impact parameter back out of the geometry rather than trusting it.
    ///
    /// **The conjunction plane is deliberately not a coordinate plane.** Written on the axes — Earth
    /// on `x`, probe in `xy` — it is the same conjunction up to a rotation and gives the same delay,
    /// and it leaves `station[1] = station[2] = satellite[2] = 0` through every evaluation. Seven of
    /// `seal-estimation-325`'s nine survivors were that fixture: with a zero component, `y·y` and
    /// `z·z` are both `0`, so `*` → `+` and `+` → `-` in the two geocentric radii change nothing at
    /// all. `seal-radiometric` had found the identical gap one module over, in `angle_between`, and
    /// the remedy is the same one — build on two orthonormal directions with no zero component,
    /// `u = (2,3,6)/7` and `v = (3,6,−4)/√61`, which are exactly orthogonal (`6 + 18 − 24 = 0`).
    /// A rotation leaves every distance and the impact parameter invariant, so nothing else moves.
    fn conjunction(impact: f64) -> ([f64; 3], [f64; 6]) {
        const U: [f64; 3] = [2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0];
        let root61 = libm::sqrt(61.0);
        let v = [3.0 / root61, 6.0 / root61, -4.0 / root61];

        let (r_earth, r_probe) = (ASTRONOMICAL_UNIT, 8.43 * ASTRONOMICAL_UNIT);
        let mut delta = libm::asin(impact * (r_earth + r_probe) / (r_earth * r_probe));
        for _ in 0..4 {
            let path = libm::sqrt(
                r_earth * r_earth + r_probe * r_probe + 2.0 * r_earth * r_probe * libm::cos(delta),
            );
            delta = libm::asin(impact * path / (r_earth * r_probe));
        }
        let (sin_delta, cos_delta) = (libm::sin(delta), libm::cos(delta));

        let mut earth = [0.0_f64; 3];
        let mut probe = [0.0_f64; 6];
        for i in 0..3 {
            earth[i] = r_earth * U[i];
            probe[i] = r_probe * (v[i] * sin_delta - U[i] * cos_delta);
        }
        (earth, probe)
    }

    /// The perpendicular distance from the origin to the line through two points.
    fn impact_parameter(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let separation = libm::sqrt(
            (a[0] - b[0]) * (a[0] - b[0])
                + (a[1] - b[1]) * (a[1] - b[1])
                + (a[2] - b[2]) * (a[2] - b[2]),
        );
        libm::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]) / separation
    }

    /// Writing `γ` out changes nothing at `γ = 1`: general relativity is the one-parameter family's
    /// own member, to the bit, and not a value the second implementation happens to come close to.
    ///
    /// Checked over four decades of path geometry, because the two differ in *where* the factor two
    /// sits — `2GM/c²` against `(1 + γ) · GM/c²` — and a scaling that is exact at one magnitude and
    /// rounds at another would show as a one-ulp disagreement somewhere along this sweep.
    #[test]
    fn the_ppn_shapiro_delay_is_the_relativistic_one_at_unit_gamma() {
        for path in [9.0e5, 9.0e6, 6.0e7, 1.2e9] {
            let (r1, r2) = (7.0e6 + path, 7.5e6 + path);
            assert_within_ulps(
                shapiro_delay_ppn(r1, r2, path, EARTH_GM, 1.0),
                shapiro_delay(r1, r2, path, EARTH_GM),
                0,
                "the PPN form at γ = 1",
            );
        }
    }

    /// `γ` scales the delay and nothing else (G1, `derivations/measurement_modifiers.mac`
    /// identity 2): the value is linear in `1 + γ` and the partial is the
    /// slope of that line, `Δρ/(1 + γ)`, which is what makes a tracking arc a measurement of `γ`.
    ///
    /// The geometry is Cassini's 2002 superior conjunction — 1 AU to 8.43 AU with the ray passing
    /// 1.6 solar radii from the Sun's centre. The measured one-way delay there is **39 333.7 m**
    /// and the partial is half of it, **19 666.8 m per unit `γ`**. Two independent checks stand
    /// behind that. The closed form is compared against the near-conjunction asymptote
    /// `(1 + γ)(GM/c²) ln(4 r_e r_p / b²)`, which is a different expression of the same geometry
    /// and agrees to better than 1e-5 relative; and the AD partial is compared against
    /// `Δρ/(1 + γ)`, which is analytic and shares no code with the AD pass.
    ///
    /// Bertotti, Iess & Tortora, *Nature* **425**, 374 (2003) measured `γ − 1 = (2.1 ± 2.3) × 10⁻⁵`
    /// from this geometry; against a 19.7 km partial that uncertainty is 45 cm of range, which is
    /// the scale a Ka-band tracking system reaches and the reason the parameter is estimable at all.
    #[test]
    fn an_estimated_gamma_is_the_delay_divided_by_one_plus_gamma() {
        let (earth, probe) = conjunction(1.6 * SOLAR_RADIUS);
        let impact = impact_parameter(&earth, &[probe[0], probe[1], probe[2]]);
        assert!(
            (impact - 1.6 * SOLAR_RADIUS).abs() < 1e-6 * SOLAR_RADIUS,
            "the conjunction geometry was placed at {impact} m from the Sun"
        );

        let gamma = 1.0;
        let modifier = EstimatedShapiroDelay::<1>::new(earth, SUN_GM, 0).expect("index 0 of 1");
        assert_eq!(modifier.index(), 0);
        let observable = Modified::new(
            Range {
                station: Station::fixed(earth),
            },
            modifier,
        );
        let p = partials(&observable, &probe, &[gamma]);

        let delay = modifier.delta(&probe, &[gamma]);
        let r_probe = libm::sqrt(probe[0] * probe[0] + probe[1] * probe[1] + probe[2] * probe[2]);
        let r_earth = libm::sqrt(earth[0] * earth[0] + earth[1] * earth[1] + earth[2] * earth[2]);
        let asymptote = libm::log(4.0 * r_earth * r_probe / (impact * impact));
        let approximate = (1.0 + gamma) * SUN_GM / (SPEED_OF_LIGHT * SPEED_OF_LIGHT) * asymptote;
        assert!(
            (delay - approximate).abs() < 1e-5 * delay,
            "the closed form {delay} m against the near-conjunction asymptote {approximate} m"
        );
        assert!(
            (39_300.0..39_400.0).contains(&delay),
            "the conjunction delay is {delay} m, not the 39.33 km the doc comment claims"
        );

        assert!(
            (p.parameters[0] - delay / (1.0 + gamma)).abs() < 1e-12 * delay,
            "∂ρ/∂γ = {} against Δρ/(1 + γ) = {}",
            p.parameters[0],
            delay / (1.0 + gamma)
        );
        assert!(
            (19_650.0..19_690.0).contains(&p.parameters[0]),
            "∂ρ/∂γ is {} m, not the 19.67 km the doc comment claims",
            p.parameters[0]
        );
        // The whole family, not one member: doubling `1 + γ` doubles the delay exactly.
        assert_within_ulps(
            modifier.delta(&probe, &[3.0]),
            2.0 * delay,
            0,
            "the delay at γ = 3",
        );
        assert_within_ulps(
            modifier.delta(&probe, &[-1.0]),
            0.0,
            0,
            "the delay at γ = −1",
        );
    }

    /// An out-of-range `γ` index is refused at construction, like every other estimated parameter.
    #[test]
    fn an_estimated_gamma_past_the_vector_is_refused() {
        assert_eq!(
            EstimatedShapiroDelay::<2>::new([0.0, 0.0, 0.0], SUN_GM, 2),
            Err(ModifierError::ParameterOutOfRange {
                index: 2,
                parameters: 2,
            })
        );
        assert!(EstimatedShapiroDelay::<2>::new([0.0, 0.0, 0.0], SUN_GM, 1).is_ok());
    }

    /// `γ` is read from the slot the chain was built with, at every slot the vector has.
    ///
    /// Checked across the whole vector rather than at one index, for the reason
    /// [`an_estimated_bias_reports_the_parameter_it_reads`] is: a reader answering with a fixed
    /// index agrees with a single-index test and disagrees with the parameter actually read. The
    /// delay assertion is the same fact from the other side — wherever `γ = 1` is placed the answer
    /// is general relativity's, and the two slots holding nonsense are not read.
    #[test]
    fn an_estimated_gamma_reports_and_reads_its_own_slot() {
        let (earth, probe) = conjunction(1.6 * SOLAR_RADIUS);
        let relativity = ShapiroDelay {
            station: earth,
            gm: SUN_GM,
        };
        let none: [f64; 0] = [];
        let expected: f64 = relativity.delta(&probe, &none);

        for index in 0..3 {
            let modifier =
                EstimatedShapiroDelay::<3>::new(earth, SUN_GM, index).expect("within the vector");
            assert_eq!(modifier.index(), index);
            let mut parameters = [-7.5_f64; 3];
            parameters[index] = 1.0;
            assert_within_ulps(
                modifier.delta(&probe, &parameters),
                expected,
                0,
                "the delay at γ = 1 read from its own slot",
            );
        }
    }

    /// A drifting clock is two unit-and-ramp columns: `∂/∂bias = 1` and `∂/∂rate = Δt`, exactly,
    /// with the state partials untouched — which is the whole difference from [`EstimatedBias`],
    /// whose second column does not exist.
    #[test]
    fn a_drifting_clock_contributes_a_unit_column_and_a_ramp_column() {
        let (sat, st) = (satellite(), station());
        let elapsed = 1_837.5;
        // Neither index is zero and neither is the other: an accessor answering with a fixed slot,
        // or with the wrong one of the two, agrees with no assertion below.
        let drift =
            EstimatedDrift::<3>::new(elapsed, 2, 1).expect("both indices within the vector");
        assert_eq!(drift.bias_index(), 2);
        assert_eq!(drift.rate_index(), 1);

        let parameters = [0.0, 1.25e-3, -0.4];
        let observable = Modified::new(Range { station: st }, drift);
        let p = partials(&observable, &sat, &parameters);
        let bare = partials(&Range { station: st }, &sat, &parameters);

        assert_within_ulps(
            p.value,
            bare.value + parameters[2] + parameters[1] * elapsed,
            0,
            "the drifted range",
        );
        assert_within_ulps(p.parameters[2], 1.0, 0, "∂range/∂bias");
        assert_within_ulps(p.parameters[1], elapsed, 0, "∂range/∂rate");
        assert_within_ulps(p.parameters[0], 0.0, 0, "∂range/∂(unused parameter)");
        for i in 0..6 {
            assert_within_ulps(p.state[i], bare.state[i], 0, "a state partial");
        }
        // Every pairing of the three slots, so that neither accessor can answer with a constant.
        for (bias, rate) in [(0, 1), (1, 2), (2, 0)] {
            let placed = EstimatedDrift::<3>::new(7.0, bias, rate).expect("within the vector");
            assert_eq!((placed.bias_index(), placed.rate_index()), (bias, rate));
        }
        // At the arc's own reference epoch the rate is unobservable and the bias is all there is.
        let at_epoch = EstimatedDrift::<3>::new(0.0, 2, 1).expect("both indices within the vector");
        assert_within_ulps(
            at_epoch.delta(&sat, &parameters),
            parameters[2],
            0,
            "the drift at Δt = 0",
        );
    }

    /// Both indices are checked, and the refusal names whichever one is past the vector — a drift
    /// whose rate index is wrong is as unevaluable as one whose bias index is.
    #[test]
    fn a_drift_names_whichever_index_is_past_the_vector() {
        assert_eq!(
            EstimatedDrift::<2>::new(0.0, 2, 1),
            Err(ModifierError::ParameterOutOfRange {
                index: 2,
                parameters: 2,
            })
        );
        assert_eq!(
            EstimatedDrift::<2>::new(0.0, 1, 7),
            Err(ModifierError::ParameterOutOfRange {
                index: 7,
                parameters: 2,
            })
        );
        // Both wrong reports the first, which is the one the caller fixes first.
        assert_eq!(
            EstimatedDrift::<2>::new(0.0, 5, 7),
            Err(ModifierError::ParameterOutOfRange {
                index: 5,
                parameters: 2,
            })
        );
        assert!(EstimatedDrift::<2>::new(0.0, 1, 0).is_ok());
    }

    /// The corona modifier is [`solar_corona_delay`] at the leg's own geometry, to the bit, and it
    /// refers both endpoints to the Sun rather than to the frame origin.
    ///
    /// The second half is the part a value comparison alone would miss: the delay is a function of
    /// the three positions *relative to the Sun*, so translating station, Sun and satellite together
    /// must leave it unchanged. A sign slip in the subtraction passes the first assertion at one
    /// frame origin and fails this one at any other.
    #[test]
    fn the_corona_modifier_is_the_path_integral_referred_to_the_sun() {
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).expect("the published one-term series");
        let (earth_from_sun, probe_from_sun) = conjunction(4.0 * SOLAR_RADIUS);
        let sun = [1.0e10, -2.0e10, 3.0e10];
        let station = [
            sun[0] + earth_from_sun[0],
            sun[1] + earth_from_sun[1],
            sun[2] + earth_from_sun[2],
        ];
        let mut satellite = [0.0; 6];
        for i in 0..3 {
            satellite[i] = sun[i] + probe_from_sun[i];
        }

        let frequency = 8.4e9;
        let none: [f64; 0] = [];
        let modifier =
            SolarCoronaDelay::new(station, sun, series, frequency, PlasmaSignature::GroupDelay)
                .expect("a positive X-band frequency");
        let direct = solar_corona_delay(
            &series,
            &[probe_from_sun[0], probe_from_sun[1], probe_from_sun[2]],
            &earth_from_sun,
            frequency,
            PlasmaSignature::GroupDelay,
        )
        .expect("a line of sight outside the Sun");
        assert_within_ulps(
            modifier.delta(&satellite, &none),
            direct,
            0,
            "the corona modifier against the path integral",
        );

        let shifted_sun = [sun[0] - 7.0e11, sun[1] + 5.0e11, sun[2] - 1.0e11];
        let shifted_station = [
            station[0] - 7.0e11,
            station[1] + 5.0e11,
            station[2] - 1.0e11,
        ];
        let mut shifted_satellite = satellite;
        shifted_satellite[0] -= 7.0e11;
        shifted_satellite[1] += 5.0e11;
        shifted_satellite[2] -= 1.0e11;
        let shifted = SolarCoronaDelay::new(
            shifted_station,
            shifted_sun,
            series,
            frequency,
            PlasmaSignature::GroupDelay,
        )
        .expect("a positive X-band frequency");
        // Not zero ulps, and the reason is the translation and not the model: shifting a 1.5e11 m
        // coordinate by 7e11 m and subtracting it back rounds off three bits, which reaches the
        // impact parameter — a difference of two nearly-parallel vectors — as ~1e-14 relative.
        // Measured 6 ulps on a 112 m delay; the bound is 32, still five orders below any sign slip.
        assert_within_ulps(
            shifted.delta(&shifted_satellite, &none),
            direct,
            32,
            "the corona delay under a common translation",
        );

        // A group delay is late and a phase advance is early, by the same column.
        let phase = SolarCoronaDelay::new(
            station,
            sun,
            series,
            frequency,
            PlasmaSignature::PhaseAdvance,
        )
        .expect("a positive X-band frequency");
        assert_within_ulps(
            phase.delta(&satellite, &none),
            -direct,
            0,
            "the phase-advance signature",
        );
        assert!(
            direct > 0.0 && direct.is_finite(),
            "the X-band corona delay at four solar radii is {direct} m"
        );
    }

    /// [`SolarCoronaDelay::occulted`] answers the question [`Modifier::delta`]'s NaN encodes, and it
    /// answers it at the same geometries — checked by walking the line of sight across the limb from
    /// an eighth of a solar radius out to nearly four, rather than at a single point on either side.
    ///
    /// The sweep steps by 0.25 R☉ off a half-step offset, so it never lands on the limb. That is
    /// than convenient: [`conjunction`] places the ray by a fixed point that converges to about
    /// 1e-9 R☉, and `solar_corona_delay`'s test is the strict `impact < SOLAR_RADIUS`, so at
    /// exactly one solar radius the verdict is decided by the last bit of the placement and the
    /// assertion would be measuring the generator. Either side of it is a real question.
    #[test]
    fn occultation_is_exactly_where_the_corona_delay_has_no_value() {
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).expect("the published one-term series");
        let none: [f64; 0] = [];
        let mut inside = 0;
        let mut outside = 0;
        for step in 0..16 {
            let radii = 0.25 * f64::from(step) + 0.125;
            let (earth_from_sun, probe) = conjunction(radii * SOLAR_RADIUS);
            let modifier = SolarCoronaDelay::new(
                earth_from_sun,
                [0.0, 0.0, 0.0],
                series,
                8.4e9,
                PlasmaSignature::GroupDelay,
            )
            .expect("a positive X-band frequency");
            let value: f64 = modifier.delta(&probe, &none);
            assert_eq!(
                modifier.occulted(&probe),
                value.is_nan(),
                "at {radii} solar radii the verdict and the value disagree ({value} m)"
            );
            if radii < 1.0 {
                assert!(modifier.occulted(&probe), "{radii} solar radii is inside");
                inside += 1;
            } else {
                assert!(
                    !modifier.occulted(&probe) && value > 0.0,
                    "{radii} solar radii is a link, and the delay is {value} m"
                );
                outside += 1;
            }
        }
        assert_eq!(
            (inside, outside),
            (4, 12),
            "the sweep crossed the limb once"
        );
    }

    /// A frequency that is not positive and finite is refused once, at construction, so evaluation
    /// carries no error the caller has to handle on every observation.
    #[test]
    fn a_corona_link_without_a_frequency_is_refused_at_construction() {
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).expect("the published one-term series");
        for frequency in [0.0, -8.4e9, f64::NAN, f64::INFINITY] {
            assert!(
                matches!(
                    SolarCoronaDelay::new(
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        series,
                        frequency,
                        PlasmaSignature::GroupDelay,
                    ),
                    Err(RadiometricError::NonPositiveFrequency { .. })
                ),
                "{frequency} Hz was accepted as a carrier"
            );
        }
        assert!(
            SolarCoronaDelay::new(
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                series,
                8.4e9,
                PlasmaSignature::GroupDelay,
            )
            .is_ok()
        );
    }

    /// `partials` is the same two quantities a caller would get from AD by hand, at zero ulps —
    /// it exists to keep the chain rule in one place, not to compute anything of its own.
    #[test]
    fn partials_agree_with_a_direct_ad_pass() {
        let (sat, st) = (satellite(), station());
        let observable = Modified::new(
            Range { station: st },
            ShapiroDelay {
                station: st.position,
                gm: EARTH_GM,
            },
        )
        .with(EstimatedBias::<1>::new(0).expect("index 0 of 1"));
        let parameters = [-0.031];

        let p = partials(&observable, &sat, &parameters);
        let lifted = [Dual::<6>::constant(parameters[0]); 1];
        let (value, jac) = jacobian::<6, 1, _>(|s| [observable.value(s, &lifted)], &sat);
        assert_within_ulps(p.value, value[0], 0, "the value");
        for (i, row) in jac.iter().enumerate() {
            assert_within_ulps(p.state[i], row[0], 0, "a state partial");
        }
    }
}
