//! **SDTM** — a semi-empirical total-density model of the thermosphere, on open data only.
//!
//! This is our own model. It exists because the two models that measure best against accelerometer
//! densities cannot be used: DTM2020's licence is academic and non-commercial and forbids
//! dissemination of the coefficients, and NRLMSIS 2.x adds a patent to the same restrictions
//! (`docs/provenance.md`, `docs/limitations.md`). What remains in the open — NRLMSISE-00 — is
//! 51.7 % mean absolute error on the twelve satellite-months of `docs/verification.md` §10.2 against
//! DTM2020's 20.2 %, and at 400 km a fractional density error `ε` puts `Δs ≈ 1.5·ε·a_drag·t²` of
//! along-track error into a prediction, which is kilometres a day. So the gap is worth closing
//! rather than inheriting.
//!
//! # The form
//!
//! A Bates temperature profile above a fixed lower boundary, with each species in diffusive
//! equilibrium above it — the Jacchia construction, whose mathematics is published and whose
//! parameters here are fitted to open measurements:
//!
//! ```text
//! T(ζ)  = T∞ − (T∞ − T₀)·exp(−σζ)                     ζ = geopotential height above 120 km
//! n_i(ζ) = n_i(0)·(T₀/T)^{1+α_i}·exp(−m_i g₀ I₁ / k)   I₁ = ∫₀^ζ dζ'/T(ζ') closed form below
//! ρ(ζ)  = Σ_i m_i n_i(ζ)                               i ∈ {N₂, O₂, O, Ar, He}
//! ```
//!
//! `derivations/sdtm_bates.mac` is the proof that this is consistent and that the closed form is
//! the right one: that `d/dζ (σζ + ln(T/T₀))/(σT∞) = 1/T` with the boundary value zero, that the
//! species law satisfies `d ln n_i/dζ = −m_i g₀/(kT) − (1+α_i) d ln T/dζ`, that [`Thermosphere::
//! density_gradient`]'s expression is the derivative of the shipped density, and that the geopotential
//! Jacobian is `((R+z₀)/(R+z))²`. Symbolic integration of `1/T` is what hangs Maxima on this
//! integrand, so the identities are differentiations and the integral is additionally checked against
//! `quad_qags`; `verify.sh` runs all of it, and the tests here re-check the closed form against
//! Simpson quadrature and the analytic gradient against a Richardson-extrapolated difference.
//!
//! Two things are deliberately *not* in that list, and both are licence-driven rather than physical:
//! there is **no effective-mass profile**, and there is **no multi-source data collator or sequential
//! ensemble estimator**. US 10,641,925 B2 (in force in the United States to 2039-06-13; the European
//! member of the family was withdrawn) claims a density estimator whose parameters include an
//! effective mass profile, fitted from collated and weighted multi-source data. Bates plus diffusive
//! equilibrium above a fixed boundary is prior art by decades, and it is what this uses.
//!
//! # What is fitted
//!
//! The exospheric temperature, the profile's shape parameter, the storm response's time constants,
//! and one scale on the boundary's *total* density. The boundary *composition* stays literature
//! ([`BOUNDARY`], Jacchia 1971): the fit may move the level at 120 km, and does, because that level is
//! a 1971 table rather than a measurement of this atmosphere — but it may not move one species against
//! another, which is the freedom that would let it buy a density at 450 km with a composition at 120
//! that nothing in the data constrains.
//!
//! ```text
//! T∞ = T₀ + exp(Σ_g Σ_k c[g][k] · driver_g · Y_k(φ, λ))
//! ```
//!
//! with `Y_k` the fully-normalised real spherical harmonics of the *sun-fixed* frame (latitude φ,
//! local solar time λ) — the same basis, and the same verified Legendre recursion, that
//! [`crate::density_calibration::DensityCorrection`] uses — and `driver_g` the
//! [`DRIVERS`] global regressors: solar activity, its square, the daily excess over the 81-day mean,
//! the geomagnetic storm response in two scales with a quadratic in each, the annual and semiannual
//! harmonics, and the products of activity with those four seasonal terms. The coefficients are
//! therefore `DRIVERS × K` numbers, and the activity-by-geography cross terms the literature models
//! all carry come out of that product rather than being listed by hand.
//!
//! Writing `T∞ = T₀ + exp(…)` rather than `T∞ = exp(…)` is what makes `T∞ ≥ T₀` hold by construction
//! at every point of parameter space. It is not cosmetic: the pilot fit of this form (`plans/16` §2)
//! drove `T∞` to 549 K at the 2019 solar minimum, and a form that can cross `T₀` has a Bates profile
//! that is no longer monotone and an `I₁` that is no longer the integral of a positive function. The
//! bound is not strict only because `exp` underflows below an exponent of −745, and that limit is
//! the exact isothermal barometric profile at `T₀` rather than a singularity — the profile degrades
//! into physics, which is the property worth having at the edge of parameter space.
//!
//! # What the caller owns
//!
//! The indices, exactly as JB2008 and [`crate::nrlmsise00`] do: nothing here reads a data
//! file. The one piece of state a memoryless index cannot express is the storm response, because the
//! thermosphere's reaction to geomagnetic forcing rises over hours and decays over most of a day.
//! [`StormResponse`] is that state — the two-constant [`StormKernel`] or heating with
//! [`NitricOxideCooling`], every constant fitted; it is a property of the
//! model, so the field owns it and exposes it through [`Thermosphere::storm`], while the *history*
//! stays with the caller who fetched it. One call per epoch feeds many density evaluations.

use crate::density_calibration::DensityCorrection;

/// Boltzmann's constant, J/K (SI defining constant, exact).
const BOLTZMANN: f64 = 1.380_649e-23;
/// Atomic mass constant, kg (CODATA 2022, `m_u = 1.66053906892(52)e-27`).
const ATOMIC_MASS: f64 = 1.660_539_068_92e-27;

/// Altitude of the lower boundary, m. Above it the species are in diffusive equilibrium; below it
/// nothing here applies.
pub const BOUNDARY_ALTITUDE: f64 = 120_000.0;

/// Temperature at the lower boundary, K — Jacchia (1971) `T₁₂₀`. Held fixed: at 120 km the
/// temperature is far better constrained by the literature than by the orbital drag this is fitted to.
pub const BOUNDARY_TEMPERATURE: f64 = 380.0;

/// Effective Earth radius for the geopotential-height reduction, m. The value JB2008's `XGRAV`
/// uses (6356.766 km), so the two models' altitude arguments mean the same thing.
pub const REFERENCE_RADIUS: f64 = 6_356_766.0;

/// Standard gravity at the boundary, m/s² — `9.80665·(R/(R+z₀))²` with the radius above. Constant
/// with latitude: the geopotential form absorbs the radial dependence, and the ~0.3 % equator-to-pole
/// variation of `g` is a factor the fitted latitude harmonics carry.
pub const BOUNDARY_GRAVITY: f64 = 9.806_65
    * (REFERENCE_RADIUS / (REFERENCE_RADIUS + BOUNDARY_ALTITUDE))
    * (REFERENCE_RADIUS / (REFERENCE_RADIUS + BOUNDARY_ALTITUDE));

/// One species at the lower boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundarySpecies {
    /// Molecular mass, kg.
    pub mass: f64,
    /// Number density at [`BOUNDARY_ALTITUDE`], m⁻³.
    pub number_density: f64,
    /// Thermal-diffusion factor `α_i`. Zero for every species heavier than helium.
    pub thermal_diffusion: f64,
}

/// The boundary composition: N₂, O₂, O, Ar, He at 120 km, from Jacchia (1971, SAO Special Report 332,
/// Table 6) — the same boundary values the Jacchia-family models are built on, in SI units.
///
/// Helium's thermal-diffusion factor is −0.38 (Jacchia 1971 §3), the one species light enough for the
/// term to matter; it is what tilts the helium bulge towards the winter pole.
pub static BOUNDARY: [BoundarySpecies; 5] = [
    BoundarySpecies {
        mass: 28.013_4 * ATOMIC_MASS,
        number_density: 4.0e17,
        thermal_diffusion: 0.0,
    },
    BoundarySpecies {
        mass: 31.998_8 * ATOMIC_MASS,
        number_density: 7.5e16,
        thermal_diffusion: 0.0,
    },
    BoundarySpecies {
        mass: 15.999_4 * ATOMIC_MASS,
        number_density: 7.6e16,
        thermal_diffusion: 0.0,
    },
    BoundarySpecies {
        mass: 39.948 * ATOMIC_MASS,
        number_density: 1.8e15,
        thermal_diffusion: 0.0,
    },
    BoundarySpecies {
        mass: 4.002_6 * ATOMIC_MASS,
        number_density: 3.4e13,
        thermal_diffusion: -0.38,
    },
];

/// Total mass density at the boundary, kg/m³ — `Σ m_i n_i(0)`, an invariant of the form that no
/// coefficient can move.
#[must_use]
pub fn boundary_density() -> f64 {
    BOUNDARY.iter().map(|s| s.mass * s.number_density).sum()
}

/// Geopotential height above the boundary, m, for a geometric altitude `z` above the same reference
/// radius: `ζ = (z − z₀)·(R + z₀)/(R + z)`.
///
/// This is what makes a constant `g₀` exact rather than approximate — the definition of geopotential
/// height is that `g₀·ζ = ∫ g dz`.
#[must_use]
pub fn geopotential_height(altitude: f64) -> f64 {
    (altitude - BOUNDARY_ALTITUDE) * (REFERENCE_RADIUS + BOUNDARY_ALTITUDE)
        / (REFERENCE_RADIUS + altitude)
}

/// `dζ/dz = ((R + z₀)/(R + z))²`, the Jacobian that turns a gradient in geopotential height into one
/// in geometric altitude. Derived in `derivations/sdtm_bates.mac`.
#[must_use]
pub fn geopotential_jacobian(altitude: f64) -> f64 {
    let q = (REFERENCE_RADIUS + BOUNDARY_ALTITUDE) / (REFERENCE_RADIUS + altitude);
    q * q
}

/// The causal response of the thermosphere to geomagnetic forcing: a difference of two exponentials,
/// `w(τ) = (e^{−τ/τ_r} − e^{−τ/τ_o})/(τ_r − τ_o)`, convolved with the index history.
///
/// Two time constants rather than one because the measured response has both a rise and a decay: the
/// heating is deposited over a few hours and the density relaxes over most of a day, so a single
/// exponential is either too slow to see the onset or too fast to carry the recovery. `w(0) = 0`,
/// `∫₀^∞ w = 1` (so a constant index reproduces itself), and the peak sits at
/// `τ* = ln(τ_r/τ_o)·τ_rτ_o/(τ_r − τ_o)`.
///
/// Both constants are fitted, not assumed. The literature's range is the prior: an onset of a few
/// hours and a recovery of 12–24 h.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StormKernel {
    /// Onset time constant `τ_o`, seconds.
    pub onset: f64,
    /// Recovery time constant `τ_r`, seconds. It must be separated from [`Self::onset`] by more than
    /// [`KERNEL_SEPARATION`] of itself, because the weight divides by `τ_r − τ_o`.
    pub recovery: f64,
}

impl StormKernel {
    /// The kernel weight at lag `τ` seconds. Zero for a negative lag, which is what causal means.
    #[must_use]
    pub fn weight(&self, lag: f64) -> f64 {
        if lag < 0.0 {
            return 0.0;
        }
        (libm::exp(-lag / self.recovery) - libm::exp(-lag / self.onset))
            / (self.recovery - self.onset)
    }

    /// The response at the end of `history`, a series of index values sampled every `cadence`
    /// seconds and held constant over each interval, with `history.last()` the interval ending at the
    /// evaluation epoch.
    ///
    /// Exact for a piecewise-constant input rather than a quadrature of it: over one interval each
    /// first-order state has the closed form `y ← a + (y − a)·e^{−h/τ}`, so the result does not
    /// depend on the cadence being small. Before the series starts the index is taken to be its
    /// first value, which is the assumption that leaves no transient at the start of a fetched file.
    #[must_use]
    pub fn response(&self, history: &[f64], cadence: f64) -> f64 {
        let Some(&first) = history.first() else {
            return 0.0;
        };
        let (decay_o, decay_r) = (
            libm::exp(-cadence / self.onset),
            libm::exp(-cadence / self.recovery),
        );
        let (mut y_o, mut y_r) = (first, first);
        for &a in history {
            y_o = a + (y_o - a) * decay_o;
            y_r = a + (y_r - a) * decay_r;
        }
        (self.recovery * y_r - self.onset * y_o) / (self.recovery - self.onset)
    }
}

/// Heating with nitric-oxide cooling, after Weimer et al. (2015, 2020, 2023), in the storm index's
/// own units:
///
/// `dD/dt = (H − D)/τ_h − D·N/(τ_h·H_s)`, `dN/dt = (H − N)/τ_NO`,
///
/// with `H` the index, `D` the storm response and `N` the nitric-oxide state. Two states rather than
/// one kernel because the measured storm has a fast peak and a slow storm-day level, and one linear
/// kernel carries either but not both (`docs/verification.md` §10.23): the nitric oxide a storm
/// produces builds up over a day and then cools the thermosphere it heated, which lowers the later
/// peaks of a long storm and shortens its recovery. Without cooling (`H_s → ∞`) it is the unit-gain
/// first-order filter; in steady state `D = H/(1 + H/H_s)`.
///
/// All three constants are fitted. Derived in `derivations/sdtm_no_cooling.mac`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NitricOxideCooling {
    /// Heating time constant `τ_h`, seconds.
    pub heating: f64,
    /// Nitric-oxide time constant `τ_NO`, seconds.
    pub nitric_oxide: f64,
    /// Saturation level `H_s`, in index units: the index at which cooling halves the response.
    pub saturation: f64,
}

impl NitricOxideCooling {
    /// `D` in steady state under a constant index, `H/(1 + H/H_s)`.
    #[must_use]
    pub fn steady_state(&self, index: f64) -> f64 {
        index / (1.0 + index / self.saturation)
    }

    /// The response at the end of `history`, with [`StormKernel::response`]'s conventions: values
    /// every `cadence` seconds, each held over its interval, `history.last()` the interval ending at
    /// the evaluation epoch.
    ///
    /// `N` takes its exact step. `D`'s rate depends on `N`, and the exact step of that system is not
    /// elementary, so `D` takes the exact step of the system with `N` frozen at its exact mean over
    /// the interval: the local error starts at `h³`, and on a realistic storm at a 3 h cadence it is
    /// 0.32 % of `D` against a fine Runge–Kutta solution. Before the series starts both states are
    /// taken to be in steady state at its first value, which leaves no transient at the start of a
    /// fetched file that began quiet.
    #[must_use]
    pub fn response(&self, history: &[f64], cadence: f64) -> f64 {
        let Some(&first) = history.first() else {
            return 0.0;
        };
        let decay = libm::exp(-cadence / self.nitric_oxide);
        let mean_weight = self.nitric_oxide / cadence * (1.0 - decay);
        let (mut d, mut n) = (self.steady_state(first), first);
        for &drive in history {
            let n_mean = drive + (n - drive) * mean_weight;
            let rate = (1.0 + n_mean / self.saturation) / self.heating;
            let target = drive / (self.heating * rate);
            d = target + (d - target) * libm::exp(-rate * cadence);
            n = drive + (n - drive) * decay;
        }
        d
    }
}

/// How a fitted field turns the index history into its storm driver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StormResponse {
    /// The two-constant linear kernel.
    Kernel(StormKernel),
    /// Heating with nitric-oxide cooling.
    Cooling(NitricOxideCooling),
}

impl StormResponse {
    /// The storm driver at the end of `history` — [`StormKernel::response`] or
    /// [`NitricOxideCooling::response`], which share their conventions.
    #[must_use]
    pub fn response(&self, history: &[f64], cadence: f64) -> f64 {
        match self {
            Self::Kernel(k) => k.response(history, cadence),
            Self::Cooling(c) => c.response(history, cadence),
        }
    }

    /// Whether the constants describe a response: a kernel's positive and separated by more than
    /// [`KERNEL_SEPARATION`] (it divides by their difference), the cooling's finite and positive.
    fn usable(&self) -> bool {
        match self {
            Self::Kernel(k) => {
                k.onset > 0.0
                    && k.recovery > 0.0
                    && (k.recovery - k.onset).abs() > KERNEL_SEPARATION * k.recovery
            }
            Self::Cooling(c) => [c.heating, c.nitric_oxide, c.saturation]
                .iter()
                .all(|x| x.is_finite() && *x > 0.0),
        }
    }
}

/// How the daily flux excess `d = (F10.7 − F10.7a)/100` of [`driver_terms`] enters `T∞`.
///
/// Linear, a day 200 sfu over its 81-day mean weighs eight times a day 25 over it, and the corpus
/// says the thermosphere does not (`docs/verification.md` §10.26): the 0.8 % of rows with `d ≥ 0.7`
/// — flare-contaminated radio flux, and the first weeks of a rising active region, where the
/// trailing mean lags — carried a held-out error of 167 % under the linear form and 25 % saturated.
/// `F10.7` is a proxy for the EUV that heats, and its daily excess over the mean outruns the EUV's.
///
/// Derived in `derivations/sdtm_flux_excess.mac`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FluxExcess {
    /// `d` itself.
    Linear,
    /// `D·tanh(d/D)`: `d` where `|d| ≪ D`, bounded by `±D`. `scale` is `D`, in units of
    /// [`F107_SCALE`], and fitted.
    Saturated {
        /// `D`, in units of [`F107_SCALE`].
        scale: f64,
    },
}

impl FluxExcess {
    /// The excess `T∞` sees for a raw excess `d`.
    #[must_use]
    pub fn excess(&self, d: f64) -> f64 {
        match *self {
            Self::Linear => d,
            Self::Saturated { scale } => scale * libm::tanh(d / scale),
        }
    }

    /// `drivers` with `F10.7` moved to the flux whose raw excess is [`Self::excess`] — the form the
    /// generator fits, so [`driver_terms`] stays the one definition of the regressors.
    fn drivers(&self, drivers: &SdtmDrivers) -> SdtmDrivers {
        match self {
            Self::Linear => *drivers,
            Self::Saturated { .. } => SdtmDrivers {
                f107: drivers.f107a
                    + F107_SCALE * self.excess((drivers.f107 - drivers.f107a) / F107_SCALE),
                ..*drivers
            },
        }
    }

    /// Whether this is a form: a saturation scale finite and positive.
    fn usable(&self) -> bool {
        match *self {
            Self::Linear => true,
            Self::Saturated { scale } => scale.is_finite() && scale > 0.0,
        }
    }
}

/// How the storm driver of [`driver_terms`] scales with the 81-day mean flux.
///
/// `T∞ − T₀` is the exponential of the regression, so a storm term with no flux in it adds kelvins in
/// proportion to the quiet `T∞ − T₀`, which rises with the flux. The measured storm response does
/// not (`docs/verification.md` §10.27): held out on the corpus, the median storm-peak ratio ran from
/// 0.71 below F10.7a = 80 to 2.36 above 200. Scaling the storm driver by `(F10.7a/150)^κ` before it
/// enters `a = ln(1 + storm/15)` is a gain on a minor storm and a shift by the log of the gain on a
/// large one, and it takes the trend out; `κ` is −1.26 to −1.32 on every held-out fold.
///
/// It scales the storm state's output. While the gain holds over the history that is the state run
/// on the gained index with the gained saturation, because the state is homogeneous of degree one.
/// Derived in `derivations/sdtm_storm_gain.mac`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StormGain {
    /// The storm driver as the state gives it.
    Unity,
    /// `(F10.7a/150)^κ`, with `exponent` the fitted `κ`.
    FluxPower {
        /// `κ`.
        exponent: f64,
    },
}

impl StormGain {
    /// The factor on the storm driver at the 81-day mean flux `f107a`.
    #[must_use]
    pub fn gain(&self, f107a: f64) -> f64 {
        match *self {
            Self::Unity => 1.0,
            Self::FluxPower { exponent } => libm::pow(f107a / REFERENCE_F107, exponent),
        }
    }

    /// `drivers` with the storm driver scaled by [`Self::gain`].
    fn drivers(&self, drivers: &SdtmDrivers) -> SdtmDrivers {
        match self {
            Self::Unity => *drivers,
            Self::FluxPower { .. } => SdtmDrivers {
                storm: drivers.storm * self.gain(drivers.f107a),
                ..*drivers
            },
        }
    }

    /// Whether this is a form: an exponent that is finite.
    fn usable(&self) -> bool {
        match *self {
            Self::Unity => true,
            Self::FluxPower { exponent } => exponent.is_finite(),
        }
    }
}

/// Where and when to evaluate the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdtmPoint {
    /// **Geodetic** altitude above the reference ellipsoid, m — the altitude JB2008, DTM2020 and
    /// NRLMSISE-00 all take, and the one the TU Delft product this was fitted to reports (its column
    /// 4, GRS80). Must exceed [`BOUNDARY_ALTITUDE`].
    ///
    /// [`REFERENCE_RADIUS`] is the *effective* radius of the geopotential reduction, not a datum this
    /// is measured from, and reading it as one is not a small error: a sphere of that radius sits
    /// 21.371 km below the GRS80 surface at the equator and 13.7 m above it at the pole, so at a
    /// 400 km scale height of 58 km the equatorial confusion is a factor of 1.44 in density.
    pub altitude: f64,
    /// Latitude, radians.
    pub latitude: f64,
    /// Local solar time, hours in `[0, 24)`.
    pub local_solar_time: f64,
    /// Day of year, `1.0` at the start of 1 January.
    pub day_of_year: f64,
}

/// The indices the model consumes, in the caller's hands exactly as JB2008's are.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdtmDrivers {
    /// 10.7 cm solar flux of the previous day, 1e-22 W/(m²·Hz), at the Earth's actual distance from
    /// the Sun.
    pub f107: f64,
    /// 10.7 cm flux, 81-day **trailing** mean — the window ending on the sample's day, not the
    /// centred one NRLMSISE-00 specifies.
    ///
    /// That is the fit's convention and it was chosen for the operational path: a centred mean needs
    /// forty days of the future, so within forty days of the present it is half forecast, and this
    /// model's driver stays an observation to the last published day.
    /// [`crate::space_weather::sdtm_drivers`] applies it, and the two means are not
    /// interchangeable: over the index feed's 25 125 observed days they differ by 6.7 flux units on
    /// average, by 31.6 at the 99th percentile and by 42.6 at the worst (2001-09-17), and by 9.3 on
    /// average across the six months this model was fitted on.
    pub f107a: f64,
    /// Geomagnetic storm response — [`StormResponse::response`] over the index history, in the units
    /// of that index (`ap` or `ap60`).
    pub storm: f64,
}

/// The region a fitted field's rows span: geodetic altitude and the 81-day mean flux, each from its
/// lowest row to its highest.
///
/// Inside it the field interpolates between measurements. Outside it the field extrapolates where no
/// row tests it, and `docs/verification.md` §10.28 recommends NRLMSISE-00 there instead. The storm
/// driver is not bounded here: its fitted range is whatever storms the corpus span brought, and the
/// largest of those are a known limitation of their own (`docs/limitations.md`). Nor is the daily
/// flux, whose excess over the mean the shipped field saturates ([`FluxExcess`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Envelope {
    /// Geodetic altitude, m, as [`SdtmPoint::altitude`].
    pub altitude: core::ops::RangeInclusive<f64>,
    /// 81-day trailing mean F10.7, as [`SdtmDrivers::f107a`].
    pub f107a: core::ops::RangeInclusive<f64>,
}

impl Envelope {
    /// Whether a point and its drivers are inside, both ends included. A NaN is outside.
    #[must_use]
    pub fn contains(&self, point: &SdtmPoint, drivers: &SdtmDrivers) -> bool {
        self.altitude.contains(&point.altitude) && self.f107a.contains(&drivers.f107a)
    }
}

/// Number of global regressors the exospheric-temperature coefficients are affine in.
pub const DRIVERS: usize = 16;

/// Reference flux the activity regressor is centred on, and its scale — so that the fitted
/// coefficients are all of order one and the constant term is the mid-cycle temperature.
pub const REFERENCE_F107: f64 = 150.0;
/// Scale of the activity regressor.
pub const F107_SCALE: f64 = 100.0;
/// Scale of the linear storm regressor, in index units.
pub const STORM_SCALE: f64 = 100.0;
/// Index at which the log-activity storm regressor is zero — the quiet/active boundary, the same
/// value [`crate::wind_model::REFERENCE_AP`] uses and for the same reason.
pub const STORM_REFERENCE: f64 = 15.0;

/// Smallest relative separation of the storm kernel's two time constants that
/// [`Thermosphere::from_coefficients`] accepts.
///
/// The kernel weight divides by `τ_r − τ_o`, so equal constants are not a model and nearly equal
/// ones are a difference of two nearly equal exponentials divided by a nearly zero number — the
/// limit is finite (`τ·e^{−τ/τ_c}/τ_c²`) but this expression does not compute it. One part in a
/// thousand keeps the cancellation four orders of magnitude clear of f64's floor, and no fitted pair
/// is anywhere near it: the physical constants differ by a factor of four to eight.
pub const KERNEL_SEPARATION: f64 = 1.0e-3;

/// The [`DRIVERS`] global regressors at a point:
/// `[1, f, f², d, s, s², a, a², cos Ω, sin Ω, cos 2Ω, sin 2Ω, f cos Ω, f sin Ω, f cos 2Ω, f sin 2Ω]`
/// with `f = (F10.7a − 150)/100`, `d = (F10.7 − F10.7a)/100`, `s = storm/100`,
/// `a = ln(1 + storm/15)` and `Ω = 2π·doy/365.25`.
///
/// The quadratic in `f` is there because the exospheric temperature saturates at high activity; the
/// daily excess `d` is separate from the mean because the response to a flare-day flux is not the
/// response to a cycle-high mean, and a field may saturate it before it gets here ([`FluxExcess`]). The last four are the seasonal terms multiplied by activity,
/// because the amplitude of the semiannual density variation is itself a function of the cycle —
/// Bowman et al. (2008) make the semiannual term explicitly activity-dependent for that reason, and
/// a form with only additive seasonal terms cannot represent it at either end of a cycle.
///
/// The storm response appears in two scales because which of them a fit should use is a measured
/// question, not a derivable one, and `docs/verification.md` §10.15 is the measurement. `s` is linear
/// in the index; `a` is the log-activity form
/// [`crate::wind_model::disturbance_drivers`] already carries, near-linear in `Kp` over 3..8, and both
/// carry a quadratic because a storm response linear in either scale under-reaches the measured peak
/// enhancement by about half. A fit admits a subset; the pairs it does not admit are zeros in the
/// coefficient array, so this list is the union of the candidate forms rather than the cost of any one
/// of them.
#[must_use]
pub fn driver_terms(point: &SdtmPoint, drivers: &SdtmDrivers) -> [f64; DRIVERS] {
    let f = (drivers.f107a - REFERENCE_F107) / F107_SCALE;
    let d = (drivers.f107 - drivers.f107a) / F107_SCALE;
    let s = drivers.storm / STORM_SCALE;
    // `max` rather than a comparison: it reads the same, and it takes NaN to the quiet end too.
    let a = libm::log1p(drivers.storm.max(0.0) / STORM_REFERENCE);
    let omega = core::f64::consts::TAU * point.day_of_year / 365.25;
    let (sin1, cos1) = libm::sincos(omega);
    let (sin2, cos2) = libm::sincos(2.0 * omega);
    [
        1.0,
        f,
        f * f,
        d,
        s,
        s * s,
        a,
        a * a,
        cos1,
        sin1,
        cos2,
        sin2,
        f * cos1,
        f * sin1,
        f * cos2,
        f * sin2,
    ]
}

mod atmosphere;
pub mod coefficients;
mod uncertainty;

pub use atmosphere::SdtmAtmosphere;

pub use uncertainty::{
    DensityUncertainty, FLUX_EDGES, NORMAL_90, NORMAL_95, NORMAL_99, REGIMES, STORM_EDGES,
};

/// A fitted SDTM field: the exospheric-temperature coefficients, the profile shape parameter and the
/// storm response.
///
/// `K` is the number of sun-fixed spherical harmonics, `(degree + 1)²` — 1 for a globally uniform
/// temperature, 9 for degree 2, 16 for degree 3. The degree is a fit decision, not a constant of the
/// model: `plans/16` Ф15.D chooses it by held-out error, the way
/// [`crate::wind_model::HorizontalWindField`]'s was chosen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Thermosphere<const K: usize> {
    coeffs: [[f64; K]; DRIVERS],
    sigma: f64,
    boundary_scale: f64,
    storm: StormResponse,
    flux_excess: FluxExcess,
    storm_gain: StormGain,
}

impl<const K: usize> Thermosphere<K> {
    /// Wrap known coefficients. `coeffs` is driver-major, `sigma` is the Bates shape parameter in
    /// m⁻¹ (~2e-5, i.e. 0.02 km⁻¹), and `boundary_scale` multiplies [`BOUNDARY`]'s total density.
    ///
    /// `boundary_scale` is the one thing about the boundary a fit is allowed to move, and it exists
    /// because [`BOUNDARY`] is Jacchia's 1971 literature table rather than a measurement of this
    /// atmosphere: the 120 km *total* it sums to is not a fitted quantity anywhere else in the form, so
    /// without it a fit can only match the measured level by distorting `T∞` — which then distorts the
    /// altitude dependence, the one thing the profile is for. Species *ratios* stay literature, which
    /// is also what keeps the form clear of the effective-mass-profile parameterisation
    /// US 10 641 925 B2 claims (`docs/provenance.md`).
    ///
    /// Returns `None` unless `sigma` and `boundary_scale` are finite and strictly positive, `K` is a
    /// perfect square, and the storm response's constants describe one — a kernel's positive and
    /// separated by more than [`KERNEL_SEPARATION`], the cooling's finite and positive: a non-positive
    /// `sigma` is not a profile that rises to `T∞`, a `K` that is not `(degree+1)²` does not index a
    /// spherical-harmonic basis, and an unseparated kernel is a `0/0`.
    #[must_use]
    pub fn from_coefficients(
        coeffs: [[f64; K]; DRIVERS],
        sigma: f64,
        boundary_scale: f64,
        storm: StormResponse,
    ) -> Option<Self> {
        let square = {
            let root = isqrt(K);
            root * root == K
        };
        let usable = sigma.is_finite()
            && sigma > 0.0
            && boundary_scale.is_finite()
            && boundary_scale > 0.0
            && square
            && storm.usable();
        usable.then_some(Self {
            coeffs,
            sigma,
            boundary_scale,
            storm,
            flux_excess: FluxExcess::Linear,
            storm_gain: StormGain::Unity,
        })
    }

    /// The same field with the daily flux excess entering as `flux_excess` says;
    /// [`Self::from_coefficients`] makes it [`FluxExcess::Linear`]. `None` unless a saturation scale
    /// is finite and positive.
    #[must_use]
    pub fn with_flux_excess(self, flux_excess: FluxExcess) -> Option<Self> {
        flux_excess.usable().then_some(Self {
            flux_excess,
            ..self
        })
    }

    /// The same field with the storm driver scaled as `storm_gain` says;
    /// [`Self::from_coefficients`] makes it [`StormGain::Unity`]. `None` unless an exponent is
    /// finite.
    #[must_use]
    pub fn with_storm_gain(self, storm_gain: StormGain) -> Option<Self> {
        storm_gain.usable().then_some(Self { storm_gain, ..self })
    }

    /// The same field without the checks — for the generated table, and nothing else.
    ///
    /// `coefficients.rs` is written by `tools/gen_sdtm_coefficients.py`, so
    /// [`Self::from_coefficients`]'s refusals are properties of that file rather than of a caller's
    /// input: a `const` field needs no fallible path, and checking literals on every construction buys
    /// nothing. `sdtm_reference.rs` puts the same literals through [`Self::from_coefficients`] and
    /// asserts it accepts them, so the refusals still bind — at test time, where the answer cannot
    /// differ between runs.
    pub(crate) const fn from_fitted(
        coeffs: [[f64; K]; DRIVERS],
        sigma: f64,
        boundary_scale: f64,
        storm: StormResponse,
        flux_excess: FluxExcess,
        storm_gain: StormGain,
    ) -> Self {
        Self {
            coeffs,
            sigma,
            boundary_scale,
            storm,
            flux_excess,
            storm_gain,
        }
    }

    /// The spherical-harmonic degree, `√K − 1`.
    #[must_use]
    pub const fn degree() -> usize {
        isqrt(K) - 1
    }

    /// The packed coefficients, driver-major.
    #[must_use]
    pub const fn coefficients(&self) -> &[[f64; K]; DRIVERS] {
        &self.coeffs
    }

    /// The Bates shape parameter `σ`, m⁻¹.
    #[must_use]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }

    /// The fitted multiplier on [`boundary_density`].
    #[must_use]
    pub const fn boundary_scale(&self) -> f64 {
        self.boundary_scale
    }

    /// The fitted storm response, for the caller to run over its own index history.
    #[must_use]
    pub const fn storm(&self) -> &StormResponse {
        &self.storm
    }

    /// How the fitted field takes the daily flux excess.
    #[must_use]
    pub const fn flux_excess(&self) -> &FluxExcess {
        &self.flux_excess
    }

    /// How the fitted field scales the storm driver with the 81-day mean flux.
    #[must_use]
    pub const fn storm_gain(&self) -> &StormGain {
        &self.storm_gain
    }

    /// Exospheric temperature `T∞`, K — at or above [`BOUNDARY_TEMPERATURE`] by construction, and
    /// equal to it only where the exponent underflows, which is the isothermal limit.
    #[must_use]
    pub fn exospheric_temperature(&self, point: &SdtmPoint, drivers: &SdtmDrivers) -> f64 {
        let angular = DensityCorrection::<K>::basis(point.latitude, point.local_solar_time);
        let g = driver_terms(
            point,
            &self.storm_gain.drivers(&self.flux_excess.drivers(drivers)),
        );
        let mut sum = 0.0;
        for (row, gi) in self.coeffs.iter().zip(g) {
            for (c, y) in row.iter().zip(angular) {
                sum += c * gi * y;
            }
        }
        BOUNDARY_TEMPERATURE + libm::exp(sum)
    }

    /// Temperature at the point, K — the Bates profile between [`BOUNDARY_TEMPERATURE`] and `T∞`.
    #[must_use]
    pub fn temperature(&self, point: &SdtmPoint, drivers: &SdtmDrivers) -> f64 {
        let t_inf = self.exospheric_temperature(point, drivers);
        bates(t_inf, self.sigma, geopotential_height(point.altitude))
    }

    /// Total mass density, kg/m³.
    ///
    /// Below [`BOUNDARY_ALTITUDE`] the model is not defined: this returns [`boundary_density`] and a
    /// zero gradient rather than continuing a diffusive-equilibrium profile down through the
    /// turbopause, where it is wrong by orders of magnitude — and where the Bates temperature itself
    /// goes negative a few scale heights down, which would be a `NaN` rather than a bad answer.
    /// `docs/limitations.md` states the envelope.
    #[must_use]
    pub fn density(&self, point: &SdtmPoint, drivers: &SdtmDrivers) -> f64 {
        self.density_gradient(point, drivers).0
    }

    /// Total mass density and its derivative with respect to *geometric* altitude, `(ρ, dρ/dz)` in
    /// kg/m³ and kg/m⁴.
    ///
    /// Analytic, from the same closed form the density is — not a finite difference of it. The
    /// species derivative is `ρ_i·(−m_i g₀/(kT) − (1+α_i)·T′/T)` with `T′ = σ(T∞ − T)`, proved in
    /// `derivations/sdtm_bates.mac`, and the geometric-altitude Jacobian is
    /// [`geopotential_jacobian`].
    #[must_use]
    pub fn density_gradient(&self, point: &SdtmPoint, drivers: &SdtmDrivers) -> (f64, f64) {
        if point.altitude <= BOUNDARY_ALTITUDE {
            return (self.boundary_scale * boundary_density(), 0.0);
        }
        let t_inf = self.exospheric_temperature(point, drivers);
        let (rho, d_zeta) = profile(t_inf, self.sigma, geopotential_height(point.altitude));
        (
            self.boundary_scale * rho,
            self.boundary_scale * d_zeta * geopotential_jacobian(point.altitude),
        )
    }

    /// Local density scale height `H = −ρ/(dρ/dz)`, m.
    ///
    /// The exact local scale height of this profile, which is what the drag path's lift to an
    /// automatic-differentiation scalar needs; the other empirical models in this crate obtain theirs
    /// from a second model evaluation 10 km higher because they have no closed form to differentiate.
    /// Infinite at and below the boundary, where the gradient is zero by the rule above.
    #[must_use]
    pub fn scale_height(&self, point: &SdtmPoint, drivers: &SdtmDrivers) -> f64 {
        let (rho, grad) = self.density_gradient(point, drivers);
        -rho / grad
    }
}

/// The Bates profile at geopotential height `zeta`.
fn bates(t_inf: f64, sigma: f64, zeta: f64) -> f64 {
    t_inf - (t_inf - BOUNDARY_TEMPERATURE) * libm::exp(-sigma * zeta)
}

/// `(ρ, dρ/dζ)` at geopotential height `zeta ≥ 0`, in kg/m³ and kg/m³ per metre of geopotential
/// height.
///
/// One pass over the species, sharing the temperature and the integral. `I₁ = (σζ + ln(T/T₀))/(σT∞)`
/// is the closed form of `∫₀^ζ dζ′/T`, and it is exact at `ζ = 0` where both terms vanish — so the
/// boundary density is `Σ m_i n_i` with no cancellation.
fn profile(t_inf: f64, sigma: f64, zeta: f64) -> (f64, f64) {
    let t = bates(t_inf, sigma, zeta);
    let integral = (sigma * zeta + libm::log(t / BOUNDARY_TEMPERATURE)) / (sigma * t_inf);
    let t_prime = sigma * (t_inf - t);
    let ratio = BOUNDARY_TEMPERATURE / t;
    let (mut rho, mut grad) = (0.0, 0.0);
    for s in &BOUNDARY {
        let barometric = s.mass * BOUNDARY_GRAVITY / BOLTZMANN;
        let partial = s.mass
            * s.number_density
            * libm::pow(ratio, 1.0 + s.thermal_diffusion)
            * libm::exp(-barometric * integral);
        rho += partial;
        grad += partial * (-barometric / t - (1.0 + s.thermal_diffusion) * t_prime / t);
    }
    (rho, grad)
}

/// `⌊√n⌋` for `degree()` in a const context, never below 1.
///
/// The clamp is what makes [`Thermosphere::degree`] total: it is `isqrt(K) - 1` on a `usize`, so a
/// true `⌊√0⌋` would underflow, and `K = 0` is a type the language will happily form. Clamped, the
/// same expression that refuses every other non-square `K` refuses `0` as well — `1·1 ≠ 0` — so the
/// constructor needs no separate clause for it. [`crate::density_calibration`]'s copy of this helper
/// has always started at 1 for the same reason; this one did not, and a mutation run found the two
/// disagreeing at the one argument where it mattered.
const fn isqrt(n: usize) -> usize {
    let mut r = 1;
    while (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::float_cmp)]

    use super::*;

    /// A field with enough structure to exercise every term: a mid-cycle constant, activity and
    /// storm sensitivity, a diurnal bulge and an annual term. Not a fit — `plans/16` Ф15.D fits.
    fn field() -> Thermosphere<9> {
        let mut coeffs = [[0.0; 9]; DRIVERS];
        coeffs[0][0] = libm::log(700.0); // constant: T∞ ≈ 380 + 700 K at zero activity
        coeffs[1][0] = 0.35; // linear in (F10.7a − 150)/100
        coeffs[2][0] = -0.05; // saturating
        coeffs[3][0] = 0.08; // daily excess
        coeffs[4][0] = 0.20; // storm
        coeffs[0][2] = 0.06; // Y₁₁ᶜ — the diurnal bulge
        coeffs[0][3] = 0.02; // Y₁₁ˢ
        coeffs[0][4] = -0.03; // Y₂₀ — pole-to-equator
        coeffs[8][0] = 0.015; // annual
        coeffs[14][0] = 0.008; // semiannual amplitude growing with activity
        Thermosphere::from_coefficients(coeffs, 2.0e-5, 1.0, StormResponse::Kernel(kernel()))
            .expect("a usable field")
    }

    fn kernel() -> StormKernel {
        StormKernel {
            onset: 3.0 * 3600.0,
            recovery: 18.0 * 3600.0,
        }
    }

    fn point(altitude: f64) -> SdtmPoint {
        SdtmPoint {
            altitude,
            latitude: 0.42,
            local_solar_time: 14.3,
            day_of_year: 212.4,
        }
    }

    fn drivers() -> SdtmDrivers {
        SdtmDrivers {
            f107: 128.0,
            f107a: 141.0,
            storm: 22.0,
        }
    }

    /// Every slot of [`driver_terms`], by name and by value.
    ///
    /// The order of this list is shared with `tools/gen_sdtm_coefficients.py` and with every emitted
    /// coefficient table, and nothing else pins it: a driver inserted in the middle re-points every
    /// slot after it, the arrays still have the right length, and the result is a model that evaluates
    /// a semiannual amplitude where a flux cross-term belongs. That is not hypothetical — adding `s²`
    /// and the log-activity pair moved the annual term from 5 to 8 and the `f·cos 2Ω` term from 11 to
    /// 14, and the only thing that caught it was reading the diff.
    #[test]
    fn the_regressors_are_in_the_order_the_coefficient_tables_are_written_in() {
        let drivers = SdtmDrivers {
            f107: 128.0,
            f107a: 141.0,
            storm: 22.0,
        };
        let point = point(400_000.0);
        let terms = driver_terms(&point, &drivers);
        let f = (141.0 - REFERENCE_F107) / F107_SCALE;
        let d = (128.0 - 141.0) / F107_SCALE;
        let s = 22.0 / STORM_SCALE;
        let a = libm::log1p(22.0 / STORM_REFERENCE);
        let omega = core::f64::consts::TAU * 212.4 / 365.25;
        let (sin1, cos1) = libm::sincos(omega);
        let (sin2, cos2) = libm::sincos(2.0 * omega);
        let expected = [
            ("constant", 1.0),
            ("f", f),
            ("f²", f * f),
            ("d", d),
            ("s", s),
            ("s²", s * s),
            ("a", a),
            ("a²", a * a),
            ("cos Ω", cos1),
            ("sin Ω", sin1),
            ("cos 2Ω", cos2),
            ("sin 2Ω", sin2),
            ("f cos Ω", f * cos1),
            ("f sin Ω", f * sin1),
            ("f cos 2Ω", f * cos2),
            ("f sin 2Ω", f * sin2),
        ];
        assert_eq!(expected.len(), DRIVERS, "the list is the whole list");
        for (k, (name, want)) in expected.iter().enumerate() {
            assert_eq!(terms[k], *want, "slot {k} should be {name}");
        }
    }

    /// The log-activity regressor is the one that reaches a storm without overwhelming quiet time.
    ///
    /// `ap` is quasi-exponential in `Kp` — 4, 7, 15, 27, 48, 80, 132, 207, 400 — so a regressor linear
    /// in `ap` gives one storm hour the leverage of a hundred quiet ones, and a least-squares fit over
    /// a mostly quiet corpus puts its slope where the quiet data is. The statement is about *steps*
    /// rather than range: over `Kp` 3 to 8 the log regressor's step per unit of `Kp` varies by a factor
    /// of 1.30 while the linear one's varies by 6.25, so only the first is a `Kp`-like driver. That is
    /// arithmetic, not opinion, which is why it is asserted as a ratio of extremes and not as a
    /// tolerance somebody chose.
    #[test]
    fn the_log_activity_regressor_steps_evenly_in_kp_where_the_linear_one_does_not() {
        let ladder = [4.0, 7.0, 15.0, 27.0, 48.0, 80.0, 132.0, 207.0, 400.0];
        let at = |storm: f64| {
            let drivers = SdtmDrivers {
                f107: 141.0,
                f107a: 141.0,
                storm,
            };
            let g = driver_terms(&point(400_000.0), &drivers);
            (g[4], g[6])
        };
        // Kp 3..8 is indices 2..=7, five steps.
        let spread = |f: fn((f64, f64)) -> f64| {
            let steps: Vec<f64> = (2..7)
                .map(|k| f(at(ladder[k + 1])) - f(at(ladder[k])))
                .collect();
            let lo = steps.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = steps.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            hi / lo
        };
        let linear = spread(|(s, _)| s);
        let log = spread(|(_, a)| a);
        assert!(
            log < 1.35 && linear > 5.0,
            "step spread over Kp 3..8: linear ×{linear:.2}, log ×{log:.2}"
        );
        // And negative or absent activity is the quiet end, not a NaN.
        assert_eq!(at(0.0).1, 0.0);
        assert_eq!(at(-5.0).1, 0.0);
        assert_eq!(at(f64::NAN).1, 0.0);
    }

    /// The closed form of `∫₀^ζ dζ′/T(ζ′)` is the integral, measured against Simpson's rule on the
    /// same integrand rather than trusted from the algebra.
    ///
    /// This is the identity `derivations/sdtm_bates.mac` proves symbolically; here it is the shipped
    /// expression, at the shipped precision, over the whole altitude range the model is used on. The
    /// Maxima script is the proof that the form is right and this is the proof that the code is the
    /// form.
    #[test]
    fn the_closed_form_integral_equals_a_quadrature_of_one_over_the_temperature() {
        let f = field();
        let t_inf = f.exospheric_temperature(&point(400_000.0), &drivers());
        for &zeta in &[1.0e3, 10.0e3, 50.0e3, 120.0e3, 300.0e3, 600.0e3, 900.0e3] {
            let closed = (f.sigma * zeta
                + libm::log(bates(t_inf, f.sigma, zeta) / BOUNDARY_TEMPERATURE))
                / (f.sigma * t_inf);
            // Simpson on 2·n intervals; n is large enough that the O(h⁴) error is below the 1e-12
            // the assertion asks for -- h⁴ at 4000 intervals over 900 km is ~1e-13 of the integral.
            let n = 4000;
            let h = zeta / (2.0 * f64::from(n));
            let mut sum = 1.0 / bates(t_inf, f.sigma, 0.0) + 1.0 / bates(t_inf, f.sigma, zeta);
            for i in 1..2 * n {
                let w = if i % 2 == 1 { 4.0 } else { 2.0 };
                sum += w / bates(t_inf, f.sigma, f64::from(i) * h);
            }
            let quadrature = sum * h / 3.0;
            assert!(
                (closed - quadrature).abs() < 1e-12 * closed,
                "ζ = {zeta}: closed {closed:e} vs quadrature {quadrature:e}"
            );
        }
    }

    /// The value `derivations/sdtm_bates.mac` prints, to the digits Maxima printed them.
    ///
    /// A literal from the CAS, so that a change to the closed form in the code that the quadrature
    /// above would also accept -- because both moved -- still fails here.
    #[test]
    fn the_integral_matches_the_value_maxima_printed() {
        // Tinf = 1000 K, T0 = 380 K, s = 0.02 /km, x = 280 km, in the script's own units.
        let (t_inf, sigma, zeta) = (1000.0, 0.02, 280.0);
        let t = t_inf - (t_inf - 380.0) * libm::exp(-sigma * zeta);
        let closed = (sigma * zeta + libm::log(t / 380.0)) / (sigma * t_inf);
        assert!(
            (closed - 0.328_264_435_927_652).abs() < 1e-15,
            "closed form {closed:.15} against Maxima's 0.328264435927652"
        );
    }

    /// The analytic density gradient is the derivative of the density, measured against a
    /// Richardson-extrapolated central difference of the shipped `density`.
    ///
    /// The gradient is what the drag path's scale height and the state-transition matrix will use, so
    /// an error in it is invisible in the density and fatal in a covariance.
    #[test]
    fn the_analytic_gradient_is_the_derivative_of_the_shipped_density() {
        let (f, d) = (field(), drivers());
        for &z in &[150_000.0, 250_000.0, 400_000.0, 550_000.0, 800_000.0] {
            let (rho, grad) = f.density_gradient(&point(z), &d);
            // Richardson: the h and h/2 central differences combine to O(h⁴). h = 200 m keeps the
            // difference of two nearly equal densities well above the f64 cancellation floor.
            let h = 200.0;
            let diff = |step: f64| {
                (f.density(&point(z + step), &d) - f.density(&point(z - step), &d)) / (2.0 * step)
            };
            let (d1, d2) = (diff(h), diff(h / 2.0));
            let numeric = (4.0 * d2 - d1) / 3.0;
            assert!(
                (grad - numeric).abs() < 1e-9 * grad.abs(),
                "z = {z}: analytic {grad:e} vs numeric {numeric:e} (ρ = {rho:e})"
            );
        }
    }

    /// The boundary density is `Σ m_i n_i`, whatever the coefficients say.
    ///
    /// The one altitude where the answer is not ours to choose: the composition is literature, so a
    /// fit that moved the density at 120 km would be a fit that had redefined the boundary. It holds
    /// **bit for bit** at `ζ = 0`, because `I₁(0) = 0` and `T(0) = T₀` are exact in the closed form —
    /// so the identity is asserted on the profile itself, at every exospheric temperature, rather
    /// than through an altitude a millimetre above the boundary where the real profile has already
    /// moved the density by 1e-13 of itself.
    #[test]
    fn the_density_at_the_boundary_is_the_boundary_composition_for_every_field() {
        let expected = boundary_density();
        for t_inf in [380.0, 500.0, 1000.0, 2000.0] {
            let (rho, grad) = profile(t_inf, 2.0e-5, 0.0);
            assert_eq!(rho, expected, "ρ(ζ=0) at T∞ = {t_inf}");
            assert!(grad < 0.0, "the profile still falls at the boundary");
        }
        // And the public path is continuous into it: a millimetre up, the difference is the profile
        // over that millimetre (one part in 1e10 of a 10 km scale height) and nothing else.
        for storm in [0.0, 50.0, 400.0] {
            for f107a in [65.0, 150.0, 250.0] {
                let d = SdtmDrivers {
                    f107: f107a,
                    f107a,
                    storm,
                };
                let rho = field().density(&point(BOUNDARY_ALTITUDE + 1e-3), &d);
                assert!(
                    (rho - expected).abs() < 1e-6 * expected,
                    "ρ(120 km + 1 mm) = {rho:e} against Σ m_i n_i = {expected:e}"
                );
            }
        }
        // 2.4e-8 kg/m³ is the Jacchia-71 boundary value this composition is; a transcription error
        // in one species mass or density would still satisfy the identity above.
        assert!(
            (expected - 2.4e-8).abs() < 0.2e-8,
            "boundary density {expected:e}"
        );
    }

    /// The boundary scale multiplies the density and its gradient by exactly itself, and leaves the
    /// scale height and the temperature alone.
    ///
    /// Which is the whole of what it is allowed to do: it moves the *level* the profile starts from,
    /// and a parameter that also moved the shape would be a second, unidentified, shape parameter.
    #[test]
    fn the_boundary_scale_multiplies_the_density_and_nothing_else() {
        let f = field();
        let scale = 0.75;
        let scaled = Thermosphere::from_coefficients(*f.coefficients(), f.sigma, scale, *f.storm())
            .expect("usable");
        let d = drivers();
        for &z in &[BOUNDARY_ALTITUDE, 200_000.0, 450_000.0, 800_000.0] {
            let p = point(z);
            let (rho, grad) = f.density_gradient(&p, &d);
            let (scaled_rho, scaled_grad) = scaled.density_gradient(&p, &d);
            assert!(
                (scaled_rho - scale * rho).abs() <= f64::EPSILON * scale * rho,
                "z = {z}: {scaled_rho:e} against {:e}",
                scale * rho
            );
            assert!(
                (scaled_grad - scale * grad).abs() <= f64::EPSILON * (scale * grad).abs(),
                "z = {z}: gradient {scaled_grad:e} against {:e}",
                scale * grad
            );
            assert_eq!(
                scaled.exospheric_temperature(&p, &d),
                f.exospheric_temperature(&p, &d)
            );
            if z > BOUNDARY_ALTITUDE {
                let h = f.scale_height(&p, &d);
                assert!(
                    (scaled.scale_height(&p, &d) - h).abs() < 1e-9 * h,
                    "z = {z}: scale height moved"
                );
            }
        }
    }

    /// Isothermal limit: as `T∞ → T₀` the profile is the barometric law at `T₀`, term for term.
    ///
    /// The one case with an independent closed form, so it is the one case where the whole species
    /// sum can be checked against something that is not itself.
    #[test]
    fn the_isothermal_limit_is_the_barometric_law() {
        // ln(T∞ − T₀) = −30 puts T∞ − T₀ at 1e-13 K: the profile is isothermal to any precision the
        // comparison can see, and σζ/(σT∞) is still evaluated the general way.
        let mut coeffs = [[0.0; 1]; DRIVERS];
        coeffs[0][0] = -30.0;
        let f = Thermosphere::<1>::from_coefficients(
            coeffs,
            2.0e-5,
            1.0,
            StormResponse::Kernel(kernel()),
        )
        .expect("usable");
        let d = drivers();
        for &z in &[200_000.0, 400_000.0, 700_000.0] {
            let zeta = geopotential_height(z);
            let barometric: f64 = BOUNDARY
                .iter()
                .map(|s| {
                    s.mass
                        * s.number_density
                        * libm::exp(
                            -s.mass * BOUNDARY_GRAVITY * zeta / (BOLTZMANN * BOUNDARY_TEMPERATURE),
                        )
                })
                .sum();
            let rho = f.density(&point(z), &d);
            assert!(
                (rho - barometric).abs() < 1e-9 * barometric,
                "z = {z}: {rho:e} against the barometric {barometric:e}"
            );
        }
    }

    /// Density falls monotonically with altitude, the scale height stays physical, and both are
    /// continuous — the `C¹` claim, measured over the range the model is for rather than argued.
    #[test]
    fn the_profile_is_monotone_continuous_and_physically_scaled() {
        let (f, d) = (field(), drivers());
        let mut previous = f64::INFINITY;
        let mut previous_scale = f64::NAN;
        let step = 1_000.0;
        let mut z = BOUNDARY_ALTITUDE + 100.0;
        while z <= 1_000_000.0 {
            let (rho, grad) = f.density_gradient(&point(z), &d);
            let scale = -rho / grad;
            assert!(
                rho > 0.0 && rho < previous,
                "ρ not decreasing at {z}: {rho:e}"
            );
            assert!(grad < 0.0, "gradient not negative at {z}");
            // 5 to 400 km spans the measured range of this profile: ~9 km a kilometre above the
            // boundary, where the temperature gradient steepens the falloff of an N₂/O₂ mixture,
            // to ~218 km at 1000 km, where what is left is hot helium. A band that excluded either
            // end would be a band fitted to the middle.
            assert!(
                (5_000.0..400_000.0).contains(&scale),
                "scale height {scale:e} m at {z} is not thermospheric"
            );
            if previous_scale.is_finite() {
                // `|dH/dz| < 1` -- the scale height varies by less than a metre per metre of
                // altitude. Dimensionless on purpose: it is the WKB condition that makes a *local*
                // scale height meaningful at all, so it is a physical statement rather than a
                // tolerance, and it is the same number at 120 km and at 1000 km where a
                // percentage-per-kilometre bound is not (the relative rate runs from 4.3%/km just
                // above the boundary to 0.29%/km at the top, while `dH/dz` peaks at 0.61 near
                // 951 km). A discontinuity in either the density or its gradient breaks it.
                assert!(
                    (scale - previous_scale).abs() < step,
                    "scale height moves by {:e} m over {step:e} m at {z}",
                    (scale - previous_scale).abs()
                );
            }
            previous = rho;
            previous_scale = scale;
            z += step;
        }
        // Below the boundary the model is the boundary and nothing else.
        let (rho, grad) = f.density_gradient(&point(BOUNDARY_ALTITUDE - 10_000.0), &d);
        assert!((rho - boundary_density()).abs() < f64::EPSILON * rho && grad == 0.0);

        // Two absolute anchors, because every assertion above is internal: this field at 400 km
        // gives 3.5e-12 kg/m³ and a 57 km scale height, which is what the literature reports for
        // moderate solar activity at that altitude. A unit error -- kilometres for metres, a wrong
        // Boltzmann constant, a molar mass where a molecular mass belongs -- moves these by orders
        // of magnitude while leaving the model perfectly self-consistent.
        let (rho, _) = f.density_gradient(&point(400_000.0), &d);
        assert!(
            (2.0e-12..6.0e-12).contains(&rho),
            "ρ(400 km) = {rho:e} kg/m³"
        );
        let h = f.scale_height(&point(400_000.0), &d);
        assert!((45_000.0..70_000.0).contains(&h), "H(400 km) = {h:e} m");
    }

    /// A hotter exosphere is a denser thermosphere above the boundary, and `T∞ > T₀` always.
    #[test]
    fn raising_the_exospheric_temperature_raises_the_density_above_the_boundary() {
        let f = field();
        let quiet = SdtmDrivers {
            f107: 70.0,
            f107a: 70.0,
            storm: 0.0,
        };
        let active = SdtmDrivers {
            f107: 220.0,
            f107a: 200.0,
            storm: 120.0,
        };
        for &z in &[200_000.0, 400_000.0, 800_000.0] {
            let (tq, ta) = (
                f.exospheric_temperature(&point(z), &quiet),
                f.exospheric_temperature(&point(z), &active),
            );
            assert!(tq > BOUNDARY_TEMPERATURE && ta > tq, "T∞ {tq} then {ta}");
            assert!(
                f.density(&point(z), &active) > f.density(&point(z), &quiet),
                "at {z}"
            );
        }
        // The bound survives coefficients that drive the exponent past the underflow of `exp`, and
        // what it degrades into there is the isothermal barometric profile rather than a NaN.
        let mut coeffs = [[0.0; 1]; DRIVERS];
        coeffs[0][0] = -800.0;
        let cold = Thermosphere::<1>::from_coefficients(
            coeffs,
            2.0e-5,
            1.0,
            StormResponse::Kernel(kernel()),
        )
        .expect("usable");
        let t = cold.exospheric_temperature(&point(400_000.0), &quiet);
        assert_eq!(
            t, BOUNDARY_TEMPERATURE,
            "T∞ = {t} where the exponent underflows"
        );
        let rho = cold.density(&point(400_000.0), &quiet);
        let zeta = geopotential_height(400_000.0);
        let barometric: f64 = BOUNDARY
            .iter()
            .map(|s| {
                s.mass
                    * s.number_density
                    * libm::exp(
                        -s.mass * BOUNDARY_GRAVITY * zeta / (BOLTZMANN * BOUNDARY_TEMPERATURE),
                    )
            })
            .sum();
        assert!(
            rho.is_finite() && (rho - barometric).abs() < 1e-12 * barometric,
            "{rho:e}"
        );
    }

    /// The equinox-to-solstice swing of `T∞`, at a fixed place and a flux held at its own 81-day
    /// mean. `Ω = 2π·doy/365.25`, so `cos 2Ω` peaks at day 1 and troughs a quarter-year later.
    fn seasonal_swing(field: &Thermosphere<9>, f107a: f64) -> f64 {
        let d = SdtmDrivers {
            f107: f107a,
            f107a,
            storm: 0.0,
        };
        let at = |doy: f64| {
            field.exospheric_temperature(
                &SdtmPoint {
                    day_of_year: doy,
                    ..point(400_000.0)
                },
                &d,
            )
        };
        (at(1.0) - at(365.25 / 4.0)).abs()
    }

    /// The seasonal amplitude is a function of activity, which is what the last four regressors buy:
    /// with only additive seasonal terms the equinox-to-solstice swing would grow with activity only
    /// as fast as `T∞` itself does.
    ///
    /// Measured on [`field`]: 1.0 K at F10.7a 70 against 20.6 K at 220, a factor of 20; with the four
    /// cross terms zeroed, 6.8 K against 11.6 K, a factor of 1.7 which is exactly the ratio of the two
    /// exponentials. The second half of the test is what makes the first half a measurement of the
    /// cross terms rather than of the exponential's curvature.
    #[test]
    fn the_seasonal_amplitude_scales_with_activity() {
        let f = field();
        let (quiet, active) = (seasonal_swing(&f, 70.0), seasonal_swing(&f, 220.0));
        assert!(
            active > 5.0 * quiet,
            "semiannual swing {quiet} K quiet against {active} K active"
        );

        let mut flat = *f.coefficients();
        for row in flat.iter_mut().skip(9) {
            *row = [0.0; 9];
        }
        let additive = Thermosphere::from_coefficients(flat, f.sigma, f.boundary_scale, *f.storm())
            .expect("usable");
        let (q, a) = (
            seasonal_swing(&additive, 70.0),
            seasonal_swing(&additive, 220.0),
        );
        assert!(a < 2.0 * q, "without the cross terms: {q} K against {a} K");
    }

    /// The storm kernel: unit gain on a constant index, zero weight at zero lag, and the peak where
    /// the algebra says.
    #[test]
    fn the_storm_kernel_has_unit_gain_a_delayed_onset_and_the_peak_the_algebra_gives() {
        let k = kernel();
        assert_eq!(
            k.weight(0.0),
            0.0,
            "a causal kernel with a rise starts at zero"
        );
        assert_eq!(k.weight(-3600.0), 0.0, "no weight before the event");
        // Constant index in, same value out: ∫w = 1. Ten recovery times of history is enough that the
        // initial condition has decayed below the tolerance.
        let history = [17.0; 200];
        let response = k.response(&history, 3600.0);
        assert!((response - 17.0).abs() < 1e-9, "unit gain: {response}");
        let peak = libm::log(k.recovery / k.onset) * k.recovery * k.onset / (k.recovery - k.onset);
        let (a, b, c) = (
            k.weight(peak - 600.0),
            k.weight(peak),
            k.weight(peak + 600.0),
        );
        assert!(b > a && b > c, "the peak is not at τ* = {peak}");
        assert!(
            (6.0 * 3600.0..8.0 * 3600.0).contains(&peak),
            "τ* = {peak} s"
        );
    }

    /// The recursive response is the convolution, not an approximation of it.
    ///
    /// Exactness for a piecewise-constant history is the property that lets the caller use the
    /// index's own cadence -- three hours for `ap` -- instead of resampling it. Measured against a
    /// direct quadrature of `∫ w(τ)·a(t−τ) dτ` over the same step function.
    #[test]
    fn the_recursive_response_equals_a_direct_convolution_of_the_same_step_function() {
        let k = kernel();
        let cadence = 3.0 * 3600.0;
        // A storm: quiet, a three-hour-resolution rise to ap ≈ 64, and a decay.
        let mut history = [4.0_f64; 40];
        let mut age = 0.0_f64;
        for h in &mut history {
            *h = 4.0 + 60.0 * libm::exp(-(age - 28.0).abs() / 3.0);
            age += 1.0;
        }
        let recursive = k.response(&history, cadence);

        // Midpoint rule, 400 sub-steps per interval, walking the lag down one interval at a time --
        // then the pre-history tail held at the first value out to twenty recovery times, which is
        // the same assumption `response` makes.
        let sub = 400_u32;
        let dt = cadence / f64::from(sub);
        let mut lag_start = cadence * f64::from(u32::try_from(history.len()).expect("short"));
        let tail_start = lag_start;
        let mut direct = 0.0;
        for &a in &history {
            for step in 0..sub {
                direct += k.weight(lag_start - (f64::from(step) + 0.5) * dt) * a * dt;
            }
            lag_start -= cadence;
        }
        let mut lag = tail_start;
        let tail_end = tail_start + 20.0 * k.recovery;
        while lag < tail_end {
            direct += k.weight(lag + 0.5 * dt) * history[0] * dt;
            lag += dt;
        }
        assert!(
            (recursive - direct).abs() < 1e-6 * recursive,
            "recursive {recursive} against the convolution {direct}"
        );
    }

    /// The constants `derivations/sdtm_no_cooling.mac` §7 steps with: `τ_h` = 6 h, `τ_NO` = 20 h,
    /// `H_s` = 300.
    fn cooling() -> NitricOxideCooling {
        NitricOxideCooling {
            heating: 6.0 * 3600.0,
            nitric_oxide: 20.0 * 3600.0,
            saturation: 300.0,
        }
    }

    /// The step is the one Maxima proved: from rest, `H` = 0, 80, 200, 50 at a 3 h hold reach the
    /// values `sdtm_no_cooling.mac` prints at 30 digits (and `sdtm_no_cooling.gp` confirms at 60).
    /// A history that starts at zero starts from rest, because the pre-history steady state at 0 is 0.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the CAS's digits as it printed them"
    )]
    fn the_cooling_step_reproduces_the_cas_reference_values() {
        let history = [0.0, 80.0, 200.0, 50.0];
        let want = [
            0.0,
            31.340_590_579_752_310_568_477_078_14,
            95.476_858_372_459_638_463_825_058_898_3,
            73.433_019_279_766_652_034_989_509_123_7,
        ];
        for (n, d) in want.iter().enumerate() {
            let got = cooling().response(&history[..=n], 3.0 * 3600.0);
            assert!(
                (got - d).abs() <= 1e-13 * d.max(1.0),
                "after {} steps: {got} against the CAS's {d}",
                n + 1
            );
        }
    }

    /// The steady state is `H/(1 + H/H_s)`, a constant history sits on it from the first step (no
    /// start transient), and without cooling the response is the unit-gain first-order filter.
    #[test]
    fn the_cooling_state_holds_its_fixed_point_and_reduces_to_the_heating_filter() {
        let c = cooling();
        let cadence = 3.0 * 3600.0;
        let want = 50.0 / (1.0 + 50.0 / 300.0);
        assert!((c.steady_state(50.0) - want).abs() < 1e-14 * want);
        for len in [1, 2, 40] {
            let got = c.response(&vec![50.0; len], cadence);
            assert!(
                (got - want).abs() < 1e-12 * want,
                "{len} steps at 50: {got}"
            );
        }
        let none = NitricOxideCooling {
            saturation: f64::MAX,
            ..c
        };
        let mut history = vec![0.0; 9];
        history[1..].fill(100.0);
        let filter = 100.0 * (1.0 - libm::exp(-8.0 * cadence / c.heating));
        let got = none.response(&history, cadence);
        assert!(
            (got - filter).abs() < 1e-12 * filter,
            "no cooling: {got} against the first-order filter's {filter}"
        );
        // Cooling lowers the same storm, and an empty history is quiet.
        assert!(c.response(&history, cadence) < got);
        assert_eq!(c.response(&[], cadence), 0.0);
    }

    /// The enum is the two responses and nothing more.
    #[test]
    fn the_storm_response_is_the_variant_it_holds() {
        let history = [4.0, 30.0, 90.0, 120.0, 40.0, 9.0];
        let cadence = 3.0 * 3600.0;
        assert_eq!(
            StormResponse::Kernel(kernel()).response(&history, cadence),
            kernel().response(&history, cadence)
        );
        assert_eq!(
            StormResponse::Cooling(cooling()).response(&history, cadence),
            cooling().response(&history, cadence)
        );
    }

    /// The constructor refuses cooling constants that are not finite and positive, one at a time.
    #[test]
    fn the_constructor_refuses_a_cooling_that_is_not_a_response() {
        let build = |c| {
            Thermosphere::<9>::from_coefficients(
                [[0.0; 9]; DRIVERS],
                2e-5,
                1.0,
                StormResponse::Cooling(c),
            )
        };
        assert!(build(cooling()).is_some());
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            for c in [
                NitricOxideCooling {
                    heating: bad,
                    ..cooling()
                },
                NitricOxideCooling {
                    nitric_oxide: bad,
                    ..cooling()
                },
                NitricOxideCooling {
                    saturation: bad,
                    ..cooling()
                },
            ] {
                assert!(build(c).is_none(), "{c:?} accepted");
            }
        }
    }

    /// The saturation is the one Maxima proved: at `D` = 0.27 the four excesses of
    /// `sdtm_flux_excess.mac` reach its 30-digit values (and `sdtm_flux_excess.gp` confirms at 60),
    /// and the linear form is the identity.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the CAS's digits as it printed them"
    )]
    fn the_flux_excess_reproduces_the_cas_reference_values() {
        let saturated = FluxExcess::Saturated { scale: 0.27 };
        for (d, want) in [
            (8.236, 0.269_999_999_999_999_999_999_999_998_273),
            (0.712, 0.267_247_841_689_291_188_733_474_356_195),
            (-0.13, -0.120_805_909_547_456_328_426_797_669_961),
            (0.001, 0.000_999_995_427_551_380_603_497_317_931_118),
        ] {
            let got = saturated.excess(d);
            assert!(
                (got - want).abs() <= 1e-15 * want.abs(),
                "d = {d}: {got} against the CAS's {want}"
            );
            assert_eq!(FluxExcess::Linear.excess(d), d);
        }
    }

    /// The field reads the excess through its own form and nowhere else: a saturated field at a raw
    /// flux is the linear field at the flux whose raw excess is the saturated one, a linear field is
    /// unchanged by the rewrite, and the saturation is what the field reports.
    #[test]
    fn the_field_takes_the_daily_excess_through_its_form() {
        let linear = field();
        let excess = FluxExcess::Saturated { scale: 0.27 };
        let saturated = linear.with_flux_excess(excess).expect("a usable scale");
        assert_eq!(*linear.flux_excess(), FluxExcess::Linear);
        assert_eq!(*saturated.flux_excess(), excess);
        let raw = SdtmDrivers {
            f107: 233.2,
            f107a: 162.0,
            storm: 0.0,
        };
        let moved = SdtmDrivers {
            f107: 162.0 + F107_SCALE * excess.excess((233.2 - 162.0) / F107_SCALE),
            ..raw
        };
        let here = point(400_000.0);
        let want = linear.exospheric_temperature(&here, &moved);
        let got = saturated.exospheric_temperature(&here, &raw);
        assert!((got - want).abs() < 1e-12 * want, "{got} against {want}");
        // The saturated excess is three-eighths of the raw one here, so the two fields cannot agree.
        let unsaturated = linear.exospheric_temperature(&here, &raw);
        assert!(
            (got - unsaturated).abs() > 1.0,
            "{got} against {unsaturated}"
        );
        assert_eq!(
            FluxExcess::Linear.drivers(&raw),
            raw,
            "the linear form rewrites the flux"
        );
    }

    /// A saturation scale that is not finite and positive is not a form.
    #[test]
    fn the_constructor_refuses_a_flux_excess_that_is_not_a_form() {
        assert!(field().with_flux_excess(FluxExcess::Linear).is_some());
        for bad in [0.0, -0.27, f64::NAN, f64::INFINITY] {
            assert!(
                field()
                    .with_flux_excess(FluxExcess::Saturated { scale: bad })
                    .is_none(),
                "scale {bad} accepted"
            );
        }
    }

    /// The gain is the one Maxima proved: at `κ` = −1.3 the four mean fluxes of
    /// `sdtm_storm_gain.mac` reach its 30-digit values (and `sdtm_storm_gain.gp` confirms at 60), the
    /// reference flux gives exactly 1, and the unity form is 1 everywhere.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the CAS's digits as it printed them"
    )]
    fn the_storm_gain_reproduces_the_cas_reference_values() {
        let power = StormGain::FluxPower { exponent: -1.3 };
        for (f107a, want) in [
            (75.0, 2.462_288_826_689_832_568_998_786_138_34),
            (150.0, 1.0),
            (225.0, 0.590_311_662_197_037_410_598_387_887_71),
            (300.0, 0.406_126_198_178_117_761_304_854_691_387),
        ] {
            let got = power.gain(f107a);
            assert!(
                (got - want).abs() <= 1e-15 * want,
                "F10.7a = {f107a}: {got} against the CAS's {want}"
            );
            assert_eq!(StormGain::Unity.gain(f107a), 1.0);
        }
        assert_eq!(power.gain(REFERENCE_F107), 1.0, "not 1 at the reference");
    }

    /// The field reads the storm driver through its gain and nowhere else: a gained field is the
    /// unity field at the gained driver, the unity form rewrites nothing, and the gain is what the
    /// field reports.
    #[test]
    fn the_field_takes_the_storm_driver_through_its_gain() {
        let unity = field();
        let gain = StormGain::FluxPower { exponent: -1.3 };
        let gained = unity.with_storm_gain(gain).expect("a usable exponent");
        assert_eq!(*unity.storm_gain(), StormGain::Unity);
        assert_eq!(*gained.storm_gain(), gain);
        let raw = SdtmDrivers {
            f107: 230.0,
            f107a: 225.0,
            storm: 40.0,
        };
        let moved = SdtmDrivers {
            storm: 40.0 * gain.gain(225.0),
            ..raw
        };
        let here = point(400_000.0);
        let want = unity.exospheric_temperature(&here, &moved);
        let got = gained.exospheric_temperature(&here, &raw);
        assert!((got - want).abs() < 1e-12 * want, "{got} against {want}");
        // The gain is 0.59 here, so the gained field reads a smaller storm than the raw driver.
        let ungained = unity.exospheric_temperature(&here, &raw);
        assert!(ungained - got > 1.0, "{got} against {ungained}");
        assert_eq!(
            StormGain::Unity.drivers(&raw),
            raw,
            "the unity form rewrites the drivers"
        );
        // At the reference flux the gain is 1 and the two fields agree to the bit.
        let reference = SdtmDrivers {
            f107a: REFERENCE_F107,
            ..raw
        };
        assert_eq!(
            gained.exospheric_temperature(&here, &reference),
            unity.exospheric_temperature(&here, &reference)
        );
    }

    /// An exponent that is not finite is not a form; zero is, and it is the unity gain.
    #[test]
    fn the_constructor_refuses_a_storm_gain_that_is_not_a_form() {
        assert!(field().with_storm_gain(StormGain::Unity).is_some());
        let zero = StormGain::FluxPower { exponent: 0.0 };
        assert!(field().with_storm_gain(zero).is_some());
        assert_eq!(zero.gain(73.0), 1.0);
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                field()
                    .with_storm_gain(StormGain::FluxPower { exponent: bad })
                    .is_none(),
                "exponent {bad} accepted"
            );
        }
    }

    /// Both ends are inside and the next double past either is not, on each axis alone; a NaN on
    /// either axis is outside, and the axes the envelope does not bound move nothing.
    #[test]
    fn the_envelope_is_closed_on_each_axis_and_refuses_a_nan() {
        let envelope = Envelope {
            altitude: 225.0e3..=545.0e3,
            f107a: 65.8..=231.9,
        };
        let inside = |altitude: f64, f107a: f64| {
            envelope.contains(&point(altitude), &SdtmDrivers { f107a, ..drivers() })
        };
        assert!(inside(400.0e3, 141.0));
        for altitude in [225.0e3, 545.0e3] {
            assert!(inside(altitude, 141.0), "{altitude}");
        }
        for f107a in [65.8, 231.9] {
            assert!(inside(400.0e3, f107a), "{f107a}");
        }
        for altitude in [f64::next_down(225.0e3), f64::next_up(545.0e3), f64::NAN] {
            assert!(!inside(altitude, 141.0), "{altitude}");
        }
        for f107a in [f64::next_down(65.8), f64::next_up(231.9), f64::NAN] {
            assert!(!inside(400.0e3, f107a), "{f107a}");
        }
        let storm = SdtmDrivers {
            f107: 400.0,
            storm: 300.0,
            ..drivers()
        };
        assert!(envelope.contains(&point(400.0e3), &storm));
    }

    /// The constructor refuses what is not a model: a non-positive shape parameter or boundary scale,
    /// a `K` that is not a perfect square, and a kernel with equal or non-positive time constants.
    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "one refusal per clause, each beside its reason"
    )]
    fn the_constructor_refuses_parameters_that_are_not_a_profile() {
        let ok = StormResponse::Kernel(kernel());
        assert!(Thermosphere::<9>::from_coefficients([[0.0; 9]; DRIVERS], 0.0, 1.0, ok).is_none());
        assert!(
            Thermosphere::<9>::from_coefficients([[0.0; 9]; DRIVERS], -2e-5, 1.0, ok).is_none()
        );
        assert!(
            Thermosphere::<9>::from_coefficients([[0.0; 9]; DRIVERS], f64::NAN, 1.0, ok).is_none()
        );
        assert!(Thermosphere::<8>::from_coefficients([[0.0; 8]; DRIVERS], 2e-5, 1.0, ok).is_none());
        // A zero boundary scale is not a thin atmosphere, it is no atmosphere: every density becomes
        // exactly zero and every scale height a `0/0`.
        assert!(Thermosphere::<9>::from_coefficients([[0.0; 9]; DRIVERS], 2e-5, 0.0, ok).is_none());
        assert!(
            Thermosphere::<9>::from_coefficients([[0.0; 9]; DRIVERS], 2e-5, -1.0, ok).is_none()
        );
        assert!(
            Thermosphere::<9>::from_coefficients([[0.0; 9]; DRIVERS], 2e-5, f64::INFINITY, ok)
                .is_none()
        );
        let same = StormKernel {
            onset: 3600.0,
            recovery: 3600.0,
        };
        assert!(
            Thermosphere::<9>::from_coefficients(
                [[0.0; 9]; DRIVERS],
                2e-5,
                1.0,
                StormResponse::Kernel(same)
            )
            .is_none()
        );
        // And a pair that is merely too close, which is the numerically dangerous case rather than
        // the impossible one.
        let close = StormKernel {
            onset: 3600.0,
            recovery: 3600.0 * (1.0 + 1e-4),
        };
        assert!(
            Thermosphere::<9>::from_coefficients(
                [[0.0; 9]; DRIVERS],
                2e-5,
                1.0,
                StormResponse::Kernel(close)
            )
            .is_none()
        );
        let negative = StormKernel {
            onset: -3600.0,
            recovery: 3600.0,
        };
        assert!(
            Thermosphere::<9>::from_coefficients(
                [[0.0; 9]; DRIVERS],
                2e-5,
                1.0,
                StormResponse::Kernel(negative)
            )
            .is_none()
        );
        // Zero, separately from negative, and one constant at a time. A zero time constant is not the
        // limit of a small one: `exp(-lag/0)` is `exp(-inf) = 0` at every positive lag and `exp(NaN)`
        // at the origin, so the weight is a `NaN` at zero rather than a narrowing kernel. Each is
        // paired with a partner far enough away that the separation clause cannot be what rejects it,
        // so each tests the clause it is aimed at.
        for zeroed in [
            StormKernel {
                onset: 0.0,
                recovery: 3600.0,
            },
            StormKernel {
                onset: 3600.0,
                recovery: 0.0,
            },
        ] {
            assert!(
                Thermosphere::<9>::from_coefficients(
                    [[0.0; 9]; DRIVERS],
                    2e-5,
                    1.0,
                    StormResponse::Kernel(zeroed)
                )
                .is_none(),
                "{zeroed:?} accepted"
            );
        }
        // The separation is a *strict* inequality, stated from both sides of the exact boundary.
        // 2000 and 1998 are chosen so that `recovery - onset` and `KERNEL_SEPARATION * recovery` are
        // both exactly 2.0 in binary: a kernel that only just clears the boundary would leave the
        // comparison untested at the one argument that distinguishes `>` from `>=`.
        let at_boundary = StormKernel {
            onset: 1998.0,
            recovery: 2000.0,
        };
        assert_eq!(
            at_boundary.recovery - at_boundary.onset,
            KERNEL_SEPARATION * at_boundary.recovery,
            "the boundary case is not exact, so it tests nothing"
        );
        assert!(
            Thermosphere::<9>::from_coefficients(
                [[0.0; 9]; DRIVERS],
                2e-5,
                1.0,
                StormResponse::Kernel(at_boundary)
            )
            .is_none()
        );
        let past_boundary = StormKernel {
            onset: 1997.0,
            recovery: 2000.0,
        };
        assert!(
            Thermosphere::<9>::from_coefficients(
                [[0.0; 9]; DRIVERS],
                2e-5,
                1.0,
                StormResponse::Kernel(past_boundary)
            )
            .is_some()
        );
        assert_eq!(Thermosphere::<16>::degree(), 3);
        assert_eq!(Thermosphere::<1>::degree(), 0);
        // Zero terms is not a degenerate model, it is the absence of one: without `Y₀₀` the
        // exospheric temperature is `380 + e⁰` whatever the drivers say.
        assert!(Thermosphere::<0>::from_coefficients([[0.0; 0]; DRIVERS], 2e-5, 1.0, ok).is_none());
    }

    /// `isqrt` is `⌊√n⌋` clamped below at 1, and the constructor's perfect-square test is sound only
    /// because of the first half: `root·root == K` accepts exactly the squares precisely when `root`
    /// is the *floor*. Anything that overshoots — a bound tested against `r·(r+1)` rather than
    /// `(r+1)²`, say — agrees on every square and disagrees everywhere else, which is invisible to a
    /// test that only ever asks about squares.
    #[test]
    fn the_integer_square_root_is_the_floor_of_the_real_one_and_never_zero() {
        for n in 0..=200_usize {
            #[expect(clippy::cast_precision_loss, clippy::cast_sign_loss)]
            #[expect(clippy::cast_possible_truncation)]
            let expected = libm::floor((n as f64).sqrt()).max(1.0) as usize;
            assert_eq!(isqrt(n), expected, "isqrt({n})");
        }
    }

    /// The accessors return what the constructor was given, and the Bates profile runs between the
    /// two temperatures that define it.
    ///
    /// Both halves exist because a mutation run found them unsaid: every other test reaches the
    /// parameters through a density, where a wrong `boundary_scale` is a multiplicative constant the
    /// ratio tests divide out and the temperature never appears on its own at all. The anchors here
    /// are values rather than comparisons — `ζ(120 km) = 0` makes `T(120 km) = T₀` identically, and
    /// far above, `e^{−σζ}` is small enough that what is left of `T∞ − T` is a stated bound.
    #[test]
    fn the_accessors_report_the_fit_and_the_profile_spans_its_two_temperatures() {
        let f = Thermosphere::<9>::from_coefficients(
            [[0.0; 9]; DRIVERS],
            3.5e-5,
            1.75,
            StormResponse::Kernel(kernel()),
        )
        .expect("a usable field");
        assert_eq!(f.sigma(), 3.5e-5);
        assert_eq!(f.boundary_scale(), 1.75);
        assert_eq!(*f.storm(), StormResponse::Kernel(kernel()));

        let f = field();
        let (p, d) = (point(BOUNDARY_ALTITUDE), drivers());
        let t_inf = f.exospheric_temperature(&p, &d);
        assert!(
            (f.temperature(&p, &d) - BOUNDARY_TEMPERATURE).abs() < 1e-9 * BOUNDARY_TEMPERATURE,
            "at the boundary the profile is {} K, not {BOUNDARY_TEMPERATURE}",
            f.temperature(&p, &d)
        );
        // 2000 km is `σζ = 29` above the boundary — geopotential, so 1 458 km rather than 1 880 —
        // and `e^{−29}` leaves the profile a tenth of a nanokelvin short of `T∞`. Below it, never
        // above: the exponential is positive and the coefficient `T∞ − T₀` is positive with it.
        let short = t_inf - f.temperature(&point(2_000_000.0), &d);
        assert!(
            (0.0..1e-9).contains(&short),
            "at 2000 km the profile is {short} K from T∞ = {t_inf}"
        );
        // And in between it is strictly rising, which is the sign of the exponential's coefficient.
        let mid = f.temperature(&point(300_000.0), &d);
        assert!(
            BOUNDARY_TEMPERATURE < mid && mid < t_inf,
            "{BOUNDARY_TEMPERATURE} < {mid} < {t_inf} is false"
        );
    }
}
