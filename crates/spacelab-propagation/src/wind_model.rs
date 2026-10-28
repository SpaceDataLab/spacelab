//! An empirical **horizontal thermospheric wind** model, fitted here to measured crosswinds.
//!
//! This is our own model. No HWM14 code, coefficients or derived table is used: the published
//! HWM14 software and its coefficients are the Supporting Information of Drob et al. (2015,
//! *Earth and Space Science* 2, 301–319, doi:10.1002/2014EA000089), which carries a
//! **CC BY-NC-ND** licence — non-commercial, no derivatives — so it can be read and compared
//! against but neither redistributed nor ported. What is used is open data:
//! [`crate::wind_data`]'s TU Delft crosswind product (CC BY 4.0).
//!
//! # The field
//!
//! The horizontal wind is expanded in **vector spherical harmonics** in the sun-fixed
//! (latitude φ, local-solar-time λ) frame, `λ = 2π·lst/24`:
//!
//! ```text
//! (u, v) = Σ_{n,m} [ s_nm · (∂Y_nm/∂λ / cos φ ,  ∂Y_nm/∂φ)      (spheroidal)
//!                  + t_nm · (−∂Y_nm/∂φ        ,  ∂Y_nm/∂λ / cos φ) ]   (toroidal)
//! ```
//!
//! with `u` eastward, `v` northward and `Y_nm = P̄_nm(sin φ)·{cos mλ, sin mλ}` the fully-normalized
//! real surface harmonics. Two properties are worth the extra algebra over expanding `u` and `v`
//! separately in scalar harmonics:
//!
//! * **It is regular at the poles.** A scalar expansion of the eastward component admits a non-zero
//!   `u` at the geographic pole, where "east" is not a direction; the gradient forms cannot, because
//!   they are the components of a vector field that is smooth there. Near-polar orbits sample the
//!   poles more densely than anywhere else, so this is exactly where an artefact would be fitted.
//! * **It couples the two components.** The crosswind sampling is zonal-dominated — `|d̂ · ê| ≥ 0.99`
//!   in 79–93 % of the near-polar missions' rows — but not zonal-only: the polar passes swing the
//!   cross-track direction through north-south, and GOCE's sun-synchronous inclination carries a
//!   meridional projection in every row it ever recorded. A coupled basis turns that minority of
//!   rows into a constraint on the whole field instead of a residual. It is also the basis HWM14
//!   itself uses, so a comparison against its published figures is like for like.
//!
//! The coefficients are affine in a small set of global regressors ([`DRIVERS`]): altitude, the
//! 81-day mean solar flux, the geomagnetic index, and annual and semiannual harmonics of the day of
//! year. So the fitted object is `DRIVERS × angular_terms(degree)` numbers.
//!
//! # The high-latitude block
//!
//! That expansion alone leaves a measured gap. Fitted to the arcs `tools/fetch-wind-data.sh`
//! downloads, its held-out error is 55.8 m/s against a 101.9 m/s null inside 30° of the equator but
//! 171.3 m/s against 208.7 m/s poleward of 65°, and raising the degree does not close it — degrees
//! 4 to 6 at every ridge tried are worse out of sample than degree 3. Angular resolution is not what
//! is missing.
//!
//! What is missing is a coordinate. Above the auroral oval the wind is driven by ion drag from
//! magnetospheric convection, which is organized about the **magnetic** pole and in **magnetic local
//! time** — a two-cell pattern that is nearly stationary in those coordinates and that a
//! geographic/solar-time expansion can only smear around a circle of latitude, because the two poles
//! are 9.2° apart. So the field carries a second vector-harmonic expansion in
//! (magnetic latitude, magnetic local time) from [`spacelab_geomagnetism::dipole`], rotated onto the
//! geographic axes by [`CentredDipole::meridian_rotation`], with:
//!
//! * a **smooth high-latitude window** ([`disturbance_window`]) that is zero equatorward of 40°
//!   magnetic and one poleward of 55°. Without it the two expansions are nearly collinear — the
//!   frames differ by only 9.2° — and the normal equations cannot tell them apart. The window is
//!   what makes the second block identifiable rather than a reparameterization of the first.
//! * **saturating activity drivers** ([`disturbance_drivers`]): `[1, a, a²]` with
//!   `a = ln(1 + ap/15)`. The constant term is the convection pattern that is there even when quiet;
//!   `a` is nearly linear in `Kp` above `Kp ≈ 2` (the `ap`↔`Kp` table is quasi-exponential), so this
//!   is a `Kp`-like driver without importing the table, and it stops a single storm hour with
//!   `ap = 400` from carrying 27× the leverage of a `Kp = 5` hour.
//!
//! The dipole is evaluated at the point's own year. IGRF-14 in this workspace carries the 2025.0
//! coefficients and the 2025–2030 secular variation, so reaching 2002 is a 23-year extrapolation of
//! a five-year trend — but it is still the better of the two options available, measured against the
//! full IGRF-14 series: the true pole moved 1.179° between 2002.0 and 2025.0, and the linear
//! extrapolation lands 0.243° from the true 2002 pole where freezing at 2025.0 would be off by the
//! whole 1.179°.
//!
//! **It closes about half the gap it was aimed at**, measured on the same held-out arcs
//! (`tests/measured_wind.rs`): 116.64 → 103.07 m/s overall against a 133.87 m/s null, which is 24 %
//! of the variance explained becoming 41 %. Poleward of 65° the error falls from 171.4 to 149.0 m/s
//! and the correlation rises from 0.313 to 0.473; between 50° and 65°, 114.1 → 101.5 m/s and 0.282 →
//! 0.460. The equatorial band improves too, 55.8 → 52.9 m/s, which is the expected side effect: the
//! quiet block no longer has to spend coefficients smearing a polar pattern around a latitude
//! circle. What is left is still a climatology — half the variance of an individual pass is weather
//! no set of global harmonics in these six drivers can carry.
//!
//! Those figures are 0.6 m/s worse than the ones this module reported before Ф15.J, and every part of
//! the difference is the drivers rather than the field: [`WindPoint::ap`] is now the 3-hour interval
//! *before* the sample instead of the one containing it, and [`WindPoint::f107a`] the trailing 81-day
//! mean instead of the centred one. They pull opposite ways and were separated by measurement: the
//! trailing mean is worth 0.3 m/s in *our* favour and the causal index costs 0.9, which is the price
//! of no longer reading an index that is not published until after the sample it describes. The old
//! number was the better score of a model that could not be run forward.
//!
//! # The observation
//!
//! An accelerometer measures the wind component *across* the satellite velocity; the along-track
//! component is inseparable from the density. Each row is therefore one scalar equation
//! `w · d̂ = speed` ([`CrosswindSample`]), and the fit is a ridge-regularized weighted least squares
//! on those equations — never a fit to a `(u, v)` pair, which the data do not contain.
//!
//! The vertical component of `d̂` is not dropped from the geometry, it is simply not modelled: the
//! field is horizontal, so a row contributes `d̂_e·u + d̂_n·v` and its `|d̂_d| ≲ 0.02` share of a
//! vertical wind of order 10 m/s — under 0.2 m/s — lands in the residual. That is an order of
//! magnitude below the product's own uncertainty.
//!
//! # Gate
//!
//! `derivations/wind_model_vsh.mac` proves the three recursions of [`angular_basis`] are Rodrigues'
//! `P̄ₙₘ`, its `φ`-derivative and `P̄ₙₘ/cos φ`; that every toroidal term is divergence-free, every
//! spheroidal one curl-free with divergence `−n(n+1)Y`; the window's end values and slopes; the
//! declination as the obliquity rotation; that the fit's direction rotation is the transpose of the
//! evaluator's; and that the force adapter's east and north are the sphere's tangents.
//! `derivations/wind_model_vsh.gp` evaluates the basis from Rodrigues at 100 digits. The module tests
//! `the_basis_is_the_vector_harmonic_rodrigues_gives` (G1) and
//! `the_fitted_field_is_the_stationary_point_of_its_ridge_objective` (G2) hold the code to them.

// Spherical-harmonic index maths uses short symbols (n, m, x, s, p, q) and index-parallel loops over
// the coefficient/basis vectors; degree indices are tiny in f64.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use alloc::vec;
use alloc::vec::Vec;

use spacelab_autodiff::Scalar;
use spacelab_geomagnetism::dipole::CentredDipole;

use crate::forces::Wind;
use crate::wind_data::CrosswindObservation;

/// Altitude at which the driver regressor is zero, metres — the middle of the sampled range.
pub const REFERENCE_ALTITUDE: f64 = 400_000.0;
/// Altitude scale of the driver regressor, metres.
pub const ALTITUDE_SCALE: f64 = 100_000.0;
/// 81-day mean F10.7 at which the flux regressor is zero, solar flux units.
pub const REFERENCE_F107: f64 = 100.0;
/// F10.7 scale of the flux regressor, solar flux units.
pub const F107_SCALE: f64 = 100.0;
/// Geomagnetic `ap` at which the activity regressor is zero.
pub const REFERENCE_AP: f64 = 15.0;
/// `ap` scale of the activity regressor.
pub const AP_SCALE: f64 = 50.0;

/// The number of global regressors each angular term is multiplied by: a constant, altitude and its
/// square, solar flux, geomagnetic activity, and the cosine/sine of the annual and semiannual
/// cycles.
///
/// Altitude gets a quadratic term because the data span 225 km (GOCE) to 500 km (GRACE-FO) and the
/// thermospheric wind does not grow linearly across that range — it strengthens with height through
/// the lower thermosphere and flattens above roughly 300 km, which a straight line through 275 km of
/// altitude cannot represent at either end.
pub const DRIVERS: usize = 9;

/// The number of activity regressors the high-latitude block's angular terms are multiplied by:
/// a constant, and the first two powers of the saturating activity index.
///
/// The constant is not redundant with the quiet block's: it multiplies harmonics of *magnetic*
/// latitude and local time, and so carries the part of the convection pattern that is present at
/// every level of activity.
pub const DISTURBANCE_DRIVERS: usize = 3;

/// Magnetic latitude at which the high-latitude window starts to open, radians.
pub const DISTURBANCE_ONSET: f64 = 40.0 * core::f64::consts::PI / 180.0;
/// Magnetic latitude at which the high-latitude window is fully open, radians.
pub const DISTURBANCE_FULL: f64 = 55.0 * core::f64::consts::PI / 180.0;

/// Where and when a wind is wanted.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindPoint {
    /// Geodetic latitude, radians.
    pub latitude: f64,
    /// Geodetic longitude, radians — needed for the magnetic coordinates, which are Earth-fixed.
    pub longitude: f64,
    /// Local solar time, hours in `[0, 24)`.
    pub local_solar_time: f64,
    /// Geodetic altitude, metres.
    pub altitude: f64,
    /// Calendar year, e.g. `2024.0`; the epoch the geomagnetic dipole is evaluated at.
    pub year: f64,
    /// Day of year, `1..=366` (fractional days allowed).
    pub day_of_year: f64,
    /// **Trailing** 81-day mean F10.7, solar flux units — the mean of the 81 days ending at this
    /// one, [`crate::space_weather::SpaceWeatherDay::f107_last81`].
    ///
    /// Not the centred mean, which is what NRLMSISE-00 asks for and what this fit used until Ф15.J.
    /// The centred one reads 40 days of flux that have not happened yet, so a model driven by it
    /// cannot be run forward and a held-out score built on it is not a forecast score. Here the two
    /// conventions were measured against each other on the same rows and the trailing mean was also
    /// the *better* one — 102.17 m/s held-out against 102.47 — so nothing was traded for the change.
    pub f107a: f64,
    /// Geomagnetic `ap` of the 3-hour interval **before** the one this point falls in: the most
    /// recent index published at the point's own epoch.
    ///
    /// The same exclusion [`crate::space_weather::sdtm_ap_history`] makes for the density field, for
    /// the same reason — `ap` for the interval a sample is inside is not available while that
    /// interval is still running. It costs 0.90 m/s of held-out RMS (102.17 → 103.07) against the
    /// own-interval index the fit used until Ф15.J, and that is the price of a number the model can
    /// reproduce running forward rather than a defect.
    ///
    /// It is the *instantaneous* index and not a filtered one. Ф15.J asked whether it should instead
    /// be SDTM-1's storm driver — then [`crate::sdtm::StormKernel::response`] over the 48 h
    /// history — and measured seven two-constant causal kernels with peaks from 0.14 h to 11 h
    /// against it. The held-out RMS was monotone in the peak lag with the zero-lag end best, so the
    /// crosswind carries no storm lag this data can resolve and the density field's storm state is
    /// not borrowed for it.
    /// `the_storm_lag_the_wind_wants_is_measured_rather_than_borrowed` is that table.
    pub ap: f64,
}

/// One crosswind equation: `w · d̂ = speed` at a point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrosswindSample {
    /// Where and when.
    pub point: WindPoint,
    /// Northward component of the crosswind direction `d̂`.
    pub direction_north: f64,
    /// Eastward component of the crosswind direction `d̂`.
    pub direction_east: f64,
    /// The measured projection along `d̂`, m/s.
    pub speed: f64,
    /// Observation weight (e.g. `1/σ²`); non-positive weights drop the sample.
    pub weight: f64,
}

/// Turn parsed observations into fit samples, keeping every `stride`-th nominal row.
///
/// The product is sampled at 10 s, far finer than the field varies and far finer than the noise
/// decorrelates, so every row is *not* an independent observation: a stride buys conditioning and
/// wall-clock rather than losing information.
///
/// `drivers` maps an observation to its `(f107a, ap)`. The product carries neither, and they are not
/// constants over an arc — `ap` changes on a three-hour cadence and a storm is the whole reason to
/// have it as a regressor — so they are read per row from a space-weather feed
/// ([`crate::space_weather`]) rather than passed once for the file.
///
/// Returns an empty vector for `stride == 0`.
#[must_use]
pub fn samples_from<D: Fn(&CrosswindObservation) -> (f64, f64)>(
    observations: &[CrosswindObservation],
    stride: usize,
    drivers: D,
) -> Vec<CrosswindSample> {
    if stride == 0 {
        return Vec::new();
    }
    observations
        .iter()
        .filter(|o| o.nominal)
        .step_by(stride)
        .map(|o| {
            let (f107a, ap) = drivers(o);
            CrosswindSample {
                point: WindPoint {
                    latitude: o.latitude,
                    longitude: o.longitude,
                    local_solar_time: o.local_solar_time,
                    altitude: o.altitude,
                    year: f64::from(o.year),
                    day_of_year: f64::from(o.day_of_year()),
                    f107a,
                    ap,
                },
                direction_north: o.direction[0],
                direction_east: o.direction[1],
                speed: o.speed,
                weight: 1.0,
            }
        })
        .collect()
}

/// The number of vector-spherical-harmonic terms at `degree`: `2·degree·(degree + 2)`, being a
/// spheroidal and a toroidal term for each of the `(degree + 1)² − 1` real harmonics of degree ≥ 1.
///
/// Degree 0 is excluded on purpose: a constant `Y₀₀` has no gradient, so it contributes no wind.
#[must_use]
pub const fn angular_terms(degree: usize) -> usize {
    2 * degree * (degree + 2)
}

/// The `DRIVERS` global regressors at a point.
#[must_use]
pub fn driver_terms(point: &WindPoint) -> [f64; DRIVERS] {
    let annual = core::f64::consts::TAU * point.day_of_year / 365.25;
    let (sin_a, cos_a) = libm::sincos(annual);
    let (sin_s, cos_s) = libm::sincos(2.0 * annual);
    let z = (point.altitude - REFERENCE_ALTITUDE) / ALTITUDE_SCALE;
    [
        1.0,
        z,
        z * z,
        (point.f107a - REFERENCE_F107) / F107_SCALE,
        (point.ap - REFERENCE_AP) / AP_SCALE,
        cos_a,
        sin_a,
        cos_s,
        sin_s,
    ]
}

/// The `DISTURBANCE_DRIVERS` activity regressors at a point: `[1, a, a²]` with
/// `a = ln(1 + ap/REFERENCE_AP)`.
///
/// The logarithm is what makes this usable as a regressor. `ap` is quasi-exponential in `Kp` — 4, 7,
/// 15, 27, 48, 80, 132, 207, 400 for `Kp = 1..9` — so a linear `ap` term would give one storm hour
/// the leverage of a hundred quiet ones and fit the storm at the expense of everything else. `a` is
/// 0.24, 0.38, 0.69, 1.03, 1.44, 1.85, 2.28, 2.69, 3.32 on that same ladder, a near-constant 0.41
/// per unit of `Kp` from 3 to 8 — so this is a `Kp`-like driver, which is the scale the convection
/// pattern is published in, without importing the `ap`↔`Kp` table.
#[must_use]
pub fn disturbance_drivers(point: &WindPoint) -> [f64; DISTURBANCE_DRIVERS] {
    // `max` rather than a comparison: it reads the same, and it takes NaN to the quiet end too.
    let a = libm::log1p(point.ap.max(0.0) / REFERENCE_AP);
    [1.0, a, a * a]
}

/// A smooth `0 → 1` window on `|magnetic latitude|` between [`DISTURBANCE_ONSET`] and
/// [`DISTURBANCE_FULL`].
///
/// It is the cubic smoothstep, so the window and its first derivative are continuous — a hard cut
/// would put a discontinuity in the wind, and through the drag force a discontinuity in the
/// acceleration, at a fixed magnetic latitude every satellite crosses four times a revolution.
#[must_use]
pub fn disturbance_window(magnetic_latitude: f64) -> f64 {
    let x = (libm::fabs(magnetic_latitude) - DISTURBANCE_ONSET)
        / (DISTURBANCE_FULL - DISTURBANCE_ONSET);
    if x.is_nan() || x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * (3.0 - 2.0 * x)
    }
}

/// A point's centred-dipole magnetic coordinates and the rotation from magnetic to geographic
/// horizontal components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MagneticPoint {
    /// Centred-dipole magnetic latitude, radians.
    pub latitude: f64,
    /// Magnetic local time, hours in `[0, 24)`.
    pub local_time: f64,
    /// Cosine of the angle from geographic north to magnetic north.
    pub cos_rotation: f64,
    /// Sine of the angle from geographic north to magnetic north, positive toward geographic east.
    pub sin_rotation: f64,
}

/// The magnetic coordinates of a [`WindPoint`].
///
/// The sub-solar point needed for magnetic local time is reconstructed from the point itself rather
/// than taken as an input: local solar time *is* the Sun's hour angle, so the sub-solar longitude is
/// `longitude − (lst − 12)·15°` exactly, and only the sub-solar latitude needs a formula. It enters
/// only through the 9.2° dipole tilt, so `solar_declination`'s ~2° worst-case error is worth under
/// a minute of magnetic local time — three orders below the `24 h / 2·degree` the expansion itself
/// resolves.
#[must_use]
pub fn magnetic_point(point: &WindPoint) -> MagneticPoint {
    let dipole = CentredDipole::at_year(point.year);
    let sub_solar_longitude =
        point.longitude - (point.local_solar_time - 12.0) * (core::f64::consts::PI / 12.0);
    let sub_solar_latitude = solar_declination(point.day_of_year);
    let (cos_rotation, sin_rotation) = dipole.meridian_rotation(point.latitude, point.longitude);
    MagneticPoint {
        latitude: dipole.magnetic_latitude(point.latitude, point.longitude),
        local_time: dipole.magnetic_local_time(
            point.latitude,
            point.longitude,
            sub_solar_latitude,
            sub_solar_longitude,
        ),
        cos_rotation,
        sin_rotation,
    }
}

/// Solar declination from day of year, radians.
///
/// `asin(sin ε · sin L)` — the obliquity rotation exactly, with the Sun's ecliptic longitude `L`
/// approximated as linear in the day of year from the March equinox. Dropping the equation of centre
/// costs up to ~2° near the equinoxes, which is why this is private: it is fit for placing the
/// sub-solar point in [`magnetic_point`] and for nothing that needs the Sun's actual position.
fn solar_declination(day_of_year: f64) -> f64 {
    /// Mean obliquity of the ecliptic at J2000.0, radians (IAU 2006: 84381.406″).
    const OBLIQUITY: f64 = 84_381.406 / 3600.0 * (core::f64::consts::PI / 180.0);
    /// Day of year of the March equinox, to the day.
    const EQUINOX: f64 = 79.0;
    let l = core::f64::consts::TAU * (day_of_year - EQUINOX) / 365.25;
    libm::asin(libm::sin(OBLIQUITY) * libm::sin(l))
}

/// The `angular_terms(degree)` vector spherical harmonics at `(latitude, local_solar_time)`, each as
/// its `[eastward, northward]` pair.
///
/// Packed by degree `n = 1..=degree`, then order `m = 0..=n`, then `cos mλ` before `sin mλ` (no sine
/// term at `m = 0`), then spheroidal before toroidal.
#[must_use]
pub fn angular_basis(degree: usize, latitude: f64, local_solar_time: f64) -> Vec<[f64; 2]> {
    let (x, s) = libm::sincos(latitude); // x = sin φ, s = cos φ
    let lambda = local_solar_time * (core::f64::consts::TAU / 24.0);
    let n_max = degree;

    // Fully-normalized P̄_nm(sin φ), its derivative with respect to φ, and P̄_nm/cos φ. The last is
    // what keeps the `∂/∂λ / cos φ` component finite at the poles: for m ≥ 1, P̄_nm carries a factor
    // cos^m φ, so the quotient is a polynomial and satisfies the same recursion. The derivative
    // recursion is the term-by-term derivative of the value recursion (dx/dφ = cos φ, ds/dφ = −sin φ)
    // rather than a closed form from the literature, so the two cannot drift apart.
    let w = n_max + 1;
    let idx = |n: usize, m: usize| n * w + m;
    let mut p = vec![0.0_f64; w * w];
    let mut dp = vec![0.0_f64; w * w];
    let mut q = vec![0.0_f64; w * w];
    p[idx(0, 0)] = 1.0;
    if n_max >= 1 {
        let r3 = libm::sqrt(3.0);
        p[idx(1, 1)] = r3 * s;
        dp[idx(1, 1)] = -r3 * x;
        q[idx(1, 1)] = r3;
        p[idx(1, 0)] = r3 * x;
        dp[idx(1, 0)] = r3 * s;
    }
    for n in 2..=n_max {
        let nf = n as f64;
        let sect = libm::sqrt((2.0 * nf + 1.0) / (2.0 * nf));
        p[idx(n, n)] = sect * s * p[idx(n - 1, n - 1)];
        dp[idx(n, n)] = sect * (s * dp[idx(n - 1, n - 1)] - x * p[idx(n - 1, n - 1)]);
        q[idx(n, n)] = sect * s * q[idx(n - 1, n - 1)];
        for m in 0..n {
            let mf = m as f64;
            let a = libm::sqrt((2.0 * nf - 1.0) * (2.0 * nf + 1.0) / ((nf - mf) * (nf + mf)));
            let b = if m + 2 <= n {
                libm::sqrt(
                    (2.0 * nf + 1.0) * (nf - mf - 1.0) * (nf + mf - 1.0)
                        / ((2.0 * nf - 3.0) * (nf - mf) * (nf + mf)),
                )
            } else {
                0.0
            };
            let (p2, dp2, q2) = if m + 2 <= n {
                (p[idx(n - 2, m)], dp[idx(n - 2, m)], q[idx(n - 2, m)])
            } else {
                (0.0, 0.0, 0.0)
            };
            p[idx(n, m)] = a * x * p[idx(n - 1, m)] - b * p2;
            dp[idx(n, m)] = a * (x * dp[idx(n - 1, m)] + s * p[idx(n - 1, m)]) - b * dp2;
            q[idx(n, m)] = a * x * q[idx(n - 1, m)] - b * q2;
        }
    }

    let mut basis = Vec::with_capacity(angular_terms(degree));
    for n in 1..=n_max {
        for m in 0..=n {
            let (sin_ml, cos_ml) = libm::sincos(m as f64 * lambda);
            // (trig(mλ), d trig(mλ)/dλ / m) for the cosine term and then the sine term.
            let forms: [(f64, f64); 2] = [(cos_ml, -sin_ml), (sin_ml, cos_ml)];
            for (k, (trig, dtrig)) in forms.iter().enumerate() {
                if m == 0 && k == 1 {
                    continue; // sin 0 ≡ 0 — the term is identically zero, not merely small
                }
                let d_lambda = m as f64 * q[idx(n, m)] * dtrig; // (1/cos φ) ∂Y/∂λ
                let d_phi = dp[idx(n, m)] * trig; // ∂Y/∂φ
                basis.push([d_lambda, d_phi]); // spheroidal
                basis.push([-d_phi, d_lambda]); // toroidal
            }
        }
    }
    basis
}

/// What to fit and how hard to shrink it.
///
/// A parameter object rather than four positional arguments: `degree` and `disturbance_degree` are
/// both small integers and `ridge` and `disturbance_ridge` are both large floats, so a call site that
/// swapped a pair would still compile and would still return a field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FitSpec {
    /// Degree of the quiet expansion in (latitude, local solar time).
    pub degree: usize,
    /// Ridge on the quiet block.
    pub ridge: f64,
    /// Degree of the high-latitude expansion in (magnetic latitude, magnetic local time). Zero
    /// disables the block entirely.
    pub disturbance_degree: usize,
    /// Ridge on the high-latitude block. It is separate because the block sees only the fraction of
    /// the samples the window admits, so the same shrinkage means something different there.
    pub disturbance_ridge: f64,
}

/// A fitted horizontal wind field: a quiet expansion in geographic latitude and local solar time,
/// plus an optional high-latitude expansion in magnetic latitude and magnetic local time.
#[derive(Clone, Debug, PartialEq)]
pub struct HorizontalWindField {
    degree: usize,
    coeffs: Vec<f64>,
    disturbance_degree: usize,
    disturbance: Vec<f64>,
}

impl HorizontalWindField {
    /// Wrap known coefficient vectors, each packed driver-major (`coeffs[g * angular_terms + j]`).
    ///
    /// Returns `None` if either length does not match its `drivers · angular_terms(degree)`.
    #[must_use]
    pub fn from_coefficients(
        degree: usize,
        coeffs: Vec<f64>,
        disturbance_degree: usize,
        disturbance: Vec<f64>,
    ) -> Option<Self> {
        let ok = coeffs.len() == DRIVERS * angular_terms(degree)
            && disturbance.len() == DISTURBANCE_DRIVERS * angular_terms(disturbance_degree);
        ok.then_some(Self {
            degree,
            coeffs,
            disturbance_degree,
            disturbance,
        })
    }

    /// The expansion degree.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// The high-latitude expansion degree; zero when there is no such block.
    #[must_use]
    pub const fn disturbance_degree(&self) -> usize {
        self.disturbance_degree
    }

    /// The packed coefficient vector.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coeffs
    }

    /// The packed high-latitude coefficient vector.
    #[must_use]
    pub fn disturbance_coefficients(&self) -> &[f64] {
        &self.disturbance
    }

    /// The horizontal wind at a point as `[eastward, northward]`, m/s.
    #[must_use]
    pub fn velocity(&self, point: &WindPoint) -> [f64; 2] {
        let basis = angular_basis(self.degree, point.latitude, point.local_solar_time);
        let g = driver_terms(point);
        let j_max = basis.len();
        let mut east = 0.0;
        let mut north = 0.0;
        for gi in 0..DRIVERS {
            let block = gi * j_max;
            for j in 0..j_max {
                let c = self.coeffs[block + j] * g[gi];
                east += c * basis[j][0];
                north += c * basis[j][1];
            }
        }
        if self.disturbance_degree > 0 {
            let magnetic = magnetic_point(point);
            let window = disturbance_window(magnetic.latitude);
            if window > 0.0 {
                let basis = angular_basis(
                    self.disturbance_degree,
                    magnetic.latitude,
                    magnetic.local_time,
                );
                let g = disturbance_drivers(point);
                let j_max = basis.len();
                let (mut magnetic_east, mut magnetic_north) = (0.0, 0.0);
                for gi in 0..DISTURBANCE_DRIVERS {
                    let block = gi * j_max;
                    for j in 0..j_max {
                        let c = self.disturbance[block + j] * g[gi] * window;
                        magnetic_east += c * basis[j][0];
                        magnetic_north += c * basis[j][1];
                    }
                }
                east +=
                    magnetic.cos_rotation * magnetic_east + magnetic.sin_rotation * magnetic_north;
                north +=
                    magnetic.cos_rotation * magnetic_north - magnetic.sin_rotation * magnetic_east;
            }
        }
        [east, north]
    }

    /// Fit the field to crosswind equations by ridge-regularized weighted least squares:
    /// `(BᵀW B + ridge·I) c = BᵀW y`, with row `B_i = g(x_i) ⊗ (d̂_e·u_j + d̂_n·v_j)` and `y_i` the
    /// measured speed.
    ///
    /// `ridge` is not optional in practice. Crosswind sampling is zonal-dominated, so the meridional
    /// directions of the design are weakly excited rather than absent; with `ridge = 0` the solver
    /// puts whatever the noise leaves there into a large meridional wind no measurement supports, and
    /// on a single-mission arc the equations can be singular outright. A positive `ridge` shrinks the
    /// weakly-observed directions toward zero and leaves the well-observed ones alone, which is what
    /// "the data say little about this" should look like in the coefficients.
    ///
    /// The two blocks are solved together, in one normal-equation system, not in sequence: fitting
    /// the quiet field first and the high-latitude field to its residual would hand the quiet block
    /// every high-latitude row to absorb before the block that should own it is ever consulted.
    ///
    /// # Errors
    /// Returns `None` if `spec.degree` is zero, if no sample carries a positive weight, or if the
    /// regularized normal equations are not positive definite.
    #[must_use]
    pub fn fit(samples: &[CrosswindSample], spec: &FitSpec) -> Option<Self> {
        let j_max = angular_terms(spec.degree);
        // `angular_terms` is zero exactly at degree zero, so this is the whole of the documented
        // rejection and not half of it.
        if j_max == 0 {
            return None;
        }
        let jd_max = angular_terms(spec.disturbance_degree);
        let quiet = DRIVERS * j_max;
        let k = quiet + DISTURBANCE_DRIVERS * jd_max;
        let mut ata = vec![0.0_f64; k * k];
        let mut atb = vec![0.0_f64; k];
        let mut row = vec![0.0_f64; k];
        let mut used = 0_usize;
        for sample in samples {
            if sample.weight.is_nan() || sample.weight <= 0.0 {
                continue;
            }
            let basis = angular_basis(
                spec.degree,
                sample.point.latitude,
                sample.point.local_solar_time,
            );
            let g = driver_terms(&sample.point);
            for gi in 0..DRIVERS {
                let block = gi * j_max;
                for j in 0..j_max {
                    row[block + j] = g[gi]
                        * (sample.direction_east * basis[j][0]
                            + sample.direction_north * basis[j][1]);
                }
            }
            if jd_max > 0 {
                let magnetic = magnetic_point(&sample.point);
                let window = disturbance_window(magnetic.latitude);
                // The measured direction resolved on the magnetic axes, so the block's own basis
                // needs no rotation: d̂ · (u_m ê_m + v_m n̂_m) = (d̂ · ê_m) u_m + (d̂ · n̂_m) v_m.
                let magnetic_east = magnetic.cos_rotation * sample.direction_east
                    - magnetic.sin_rotation * sample.direction_north;
                let magnetic_north = magnetic.sin_rotation * sample.direction_east
                    + magnetic.cos_rotation * sample.direction_north;
                let basis = angular_basis(
                    spec.disturbance_degree,
                    magnetic.latitude,
                    magnetic.local_time,
                );
                let g = disturbance_drivers(&sample.point);
                for gi in 0..DISTURBANCE_DRIVERS {
                    let block = quiet + gi * jd_max;
                    for j in 0..jd_max {
                        row[block + j] = g[gi]
                            * window
                            * (magnetic_east * basis[j][0] + magnetic_north * basis[j][1]);
                    }
                }
            }
            let wy = sample.weight * sample.speed;
            for i in 0..k {
                if row[i] == 0.0 {
                    continue;
                }
                atb[i] += wy * row[i];
                let wri = sample.weight * row[i];
                for j in i..k {
                    ata[i * k + j] += wri * row[j];
                }
            }
            used += 1;
        }
        if used == 0 {
            return None;
        }
        for i in 0..k {
            ata[i * k + i] += if i < quiet {
                spec.ridge
            } else {
                spec.disturbance_ridge
            };
            for j in i + 1..k {
                ata[j * k + i] = ata[i * k + j];
            }
        }
        let mut coeffs = solve_symmetric_positive_definite(&mut ata, &atb, k)?;
        let disturbance = coeffs.split_off(quiet);
        Some(Self {
            degree: spec.degree,
            coeffs,
            disturbance_degree: spec.disturbance_degree,
            disturbance,
        })
    }

    /// Weighted RMS of `w · d̂ − speed` over `samples`, m/s — the residual of the equations the field
    /// was fitted to, or, on samples it was not fitted to, its prediction error.
    ///
    /// Returns `None` if no sample carries a positive weight.
    #[must_use]
    pub fn projection_rms(&self, samples: &[CrosswindSample]) -> Option<f64> {
        let mut sum = 0.0;
        let mut total = 0.0;
        for sample in samples {
            if sample.weight.is_nan() || sample.weight <= 0.0 {
                continue;
            }
            let [east, north] = self.velocity(&sample.point);
            let predicted = sample.direction_east * east + sample.direction_north * north;
            let r = predicted - sample.speed;
            sum += sample.weight * r * r;
            total += sample.weight;
        }
        (total > 0.0).then(|| libm::sqrt(sum / total))
    }
}

/// Solve `A c = b` for symmetric positive-definite `A` (row-major, `n × n`) by Cholesky. `a` is
/// overwritten. Returns `None` if a pivot is not positive, which for regularized normal equations
/// means the ridge was too small to make them definite.
fn solve_symmetric_positive_definite(a: &mut [f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d.is_nan() || d <= 0.0 {
            return None;
        }
        let l = libm::sqrt(d);
        a[j * n + j] = l;
        for i in j + 1..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = v / l;
        }
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let mut v = b[i];
        for k in 0..i {
            v -= a[i * n + k] * y[k];
        }
        y[i] = v / a[i * n + i];
    }
    let mut c = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut v = y[i];
        for k in i + 1..n {
            v -= a[k * n + i] * c[k];
        }
        c[i] = v / a[i * n + i];
    }
    Some(c)
}

/// A [`HorizontalWindField`] wrapped as a [`Wind`] for the drag force.
///
/// Mirrors [`crate::nrlmsise00::Nrlmsise00Atmosphere`]: the epoch (day of year and UT seconds at
/// `t = 0`) and the space-weather drivers are fields, and the inertial→geodetic conversion is
/// injected as a closure so this crate stays independent of the frames crates.
///
/// The wind is a horizontal vector, so its inertial components are `u·ê + v·n̂` with `ê` and `n̂` the
/// local east and north directions. Those two are computed **in `S`**, from the position, so the
/// drag Jacobian sees how the wind vector turns as the satellite moves along its orbit — the part of
/// `∂w/∂r` that is of the same order as the co-rotation term `ω`. The two scalar amplitudes `u` and
/// `v` are evaluated in f64 and lifted, exactly as the empirical density models lift their table
/// lookups, so the Jacobian does not see the field's own horizontal gradient. That gradient is the
/// wind changing over ~1000 km, an order below the geometric term.
///
/// `ê` and `n̂` are geocentric — they are built from the inertial position, not from the ellipsoid
/// normal — so they lean up to 0.19° away from their geodetic counterparts. At that angle the error
/// is 0.3 % of the wind vector, well inside the fit residual.
///
/// **This allocates.** [`angular_basis`] returns a `Vec`, and a field with a high-latitude block
/// builds two of them, so a propagation pays two allocations per force evaluation inside the
/// window and one outside it. Removing that means handing the basis a scratch buffer, which changes
/// the shape of the [`Wind`] trait and is a separate decision from whether the wind is right.
pub struct EmpiricalWind<G> {
    /// The fitted field.
    pub field: HorizontalWindField,
    /// Calendar year at propagation time `t = 0`; the epoch the geomagnetic dipole is evaluated at.
    pub year: f64,
    /// Day of year (1..=366) at propagation time `t = 0`.
    pub doy0: i32,
    /// UT seconds into the day at `t = 0`.
    pub sec0: f64,
    /// Trailing 81-day mean F10.7, as [`WindPoint::f107a`] defines it.
    pub f107a: f64,
    /// Geomagnetic `ap` of the interval before the epoch, as [`WindPoint::ap`] defines it.
    ///
    /// Held fixed across the propagation, unlike the fit, which reads it per row. Over an arc long
    /// enough for `ap` to move this is the caller's approximation to make, and the field is public so
    /// that a caller who has the feed can step it.
    pub ap: f64,
    /// Maps `(t, inertial position in metres)` to `(altitude_km, geodetic_latitude_deg,
    /// geodetic_longitude_deg)` — the same provider the empirical atmospheres take.
    pub geodetic: G,
}

/// `hours` reduced to the `[0, 24)` [`WindPoint::local_solar_time`] is documented to hold.
///
/// Named rather than inline because the range *is* the contract: the expansion reads local solar
/// time as an angle and would be just as happy with an unreduced one, so nothing downstream would
/// notice a reduction that removed the wrong number of days.
fn hours_into_day(hours: f64) -> f64 {
    hours - 24.0 * libm::floor(hours / 24.0)
}

impl<G: Fn(f64, [f64; 3]) -> (f64, f64, f64)> Wind for EmpiricalWind<G> {
    fn velocity<S: Scalar>(&self, t: f64, r: &[S; 3]) -> [S; 3] {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let (alt_km, lat_deg, lon_deg) = (self.geodetic)(t, rf);
        let total_sec = self.sec0 + t;
        let days = libm::floor(total_sec / 86_400.0);
        let sec = total_sec - 86_400.0 * days;
        let lst_raw = sec / 3_600.0 + lon_deg / 15.0;
        let point = WindPoint {
            latitude: lat_deg.to_radians(),
            longitude: lon_deg.to_radians(),
            local_solar_time: hours_into_day(lst_raw),
            altitude: alt_km * 1000.0,
            year: self.year,
            day_of_year: f64::from(self.doy0) + days,
            f107a: self.f107a,
            ap: self.ap,
        };
        let [east, north] = self.field.velocity(&point);

        // Local east ê = ẑ × r̂ / ‖ẑ × r̂‖ and north n̂ = r̂ × ê, in S. On the polar axis ẑ × r is the
        // zero vector and neither direction exists; the wind is then reported as zero rather than as
        // a division by a norm of zero. The exact pole is a measure-zero point of the orbit, and the
        // field itself vanishes there — the vector-harmonic basis is built so that it does.
        let horizontal = libm::sqrt(rf[0] * rf[0] + rf[1] * rf[1]);
        if horizontal.is_nan() || horizontal <= 0.0 {
            return [S::from_f64(0.0); 3];
        }
        let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let hn = (r[0] * r[0] + r[1] * r[1]).sqrt();
        let inv_h = hn.recip();
        let east_dir = [-r[1] * inv_h, r[0] * inv_h, S::from_f64(0.0)];
        let inv_r = rn.recip();
        let up = [r[0] * inv_r, r[1] * inv_r, r[2] * inv_r];
        // n̂ = r̂ × ê
        let north_dir = [
            up[1] * east_dir[2] - up[2] * east_dir[1],
            up[2] * east_dir[0] - up[0] * east_dir[2],
            up[0] * east_dir[1] - up[1] * east_dir[0],
        ];
        let (e, n) = (S::from_f64(east), S::from_f64(north));
        [
            east_dir[0] * e + north_dir[0] * n,
            east_dir[1] * e + north_dir[1] * n,
            east_dir[2] * e + north_dir[2] * n,
        ]
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    /// A point with everything but the geometry held at its reference value.
    fn at(latitude: f64, longitude: f64, local_solar_time: f64) -> WindPoint {
        WindPoint {
            latitude,
            longitude,
            local_solar_time,
            altitude: REFERENCE_ALTITUDE,
            year: 2024.0,
            day_of_year: 80.0,
            f107a: REFERENCE_F107,
            ap: REFERENCE_AP,
        }
    }

    /// A spec with no high-latitude block.
    const fn quiet_spec(degree: usize, ridge: f64) -> FitSpec {
        FitSpec {
            degree,
            ridge,
            disturbance_degree: 0,
            disturbance_ridge: 0.0,
        }
    }

    /// A numerically-differentiated surface harmonic, for checking the analytic basis.
    fn harmonic(degree: usize, n: usize, m: usize, sine: bool, phi: f64, lambda: f64) -> f64 {
        // Rebuild P̄_nm at `phi` from the same value recursion the basis uses, then apply the trig.
        let (x, s) = libm::sincos(phi);
        let w = degree + 1;
        let mut p = vec![0.0_f64; w * w];
        p[0] = 1.0;
        if degree >= 1 {
            p[w + 1] = libm::sqrt(3.0) * s;
            p[w] = libm::sqrt(3.0) * x;
        }
        for nn in 2..=degree {
            let nf = nn as f64;
            p[nn * w + nn] =
                libm::sqrt((2.0 * nf + 1.0) / (2.0 * nf)) * s * p[(nn - 1) * w + (nn - 1)];
            for mm in 0..nn {
                let mf = mm as f64;
                let a = libm::sqrt((2.0 * nf - 1.0) * (2.0 * nf + 1.0) / ((nf - mf) * (nf + mf)));
                let b = if mm + 2 <= nn {
                    libm::sqrt(
                        (2.0 * nf + 1.0) * (nf - mf - 1.0) * (nf + mf - 1.0)
                            / ((2.0 * nf - 3.0) * (nf - mf) * (nf + mf)),
                    )
                } else {
                    0.0
                };
                let p2 = if mm + 2 <= nn {
                    p[(nn - 2) * w + mm]
                } else {
                    0.0
                };
                p[nn * w + mm] = a * x * p[(nn - 1) * w + mm] - b * p2;
            }
        }
        let arg = m as f64 * lambda;
        p[n * w + m] * if sine { libm::sin(arg) } else { libm::cos(arg) }
    }

    /// Index of the `(n, m, sine, toroidal)` term in the packed basis.
    fn term(degree: usize, n: usize, m: usize, sine: bool, toroidal: bool) -> usize {
        let mut k = 0;
        for nn in 1..=degree {
            for mm in 0..=nn {
                for s in [false, true] {
                    if mm == 0 && s {
                        continue;
                    }
                    for t in [false, true] {
                        if (nn, mm, s, t) == (n, m, sine, toroidal) {
                            return k;
                        }
                        k += 1;
                    }
                }
            }
        }
        unreachable!("term not in basis")
    }

    /// `angular_basis(6, 0.625, 5.5)` from `derivations/wind_model_vsh.gp`.
    #[expect(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        reason = "verbatim 25-digit derivations/wind_model_vsh.gp output"
    )]
    const WIND_BASIS_MID: [[f64; 2]; 96] = [
        [0.0, 1.404629326047588612772934e0],
        [-1.404629326047588612772934e0, 0.0],
        [
            -1.717232872802521920707960e0,
            -1.322776193080313767365866e-1,
        ],
        [1.322776193080313767365866e-1, -1.717232872802521920707960e0],
        [2.260779966436308676876376e-1, -1.004748270880471098384954e0],
        [1.004748270880471098384954e0, 2.260779966436308676876376e-1],
        [0.0, 3.182991177721280167158033e0],
        [-3.182991177721280167158033e0, 0.0],
        [-2.246685433964105851043604e0, 1.594035796419923689569368e-1],
        [
            -1.594035796419923689569368e-1,
            -2.246685433964105851043604e0,
        ],
        [2.957817486745968516275933e-1, 1.210790395648774600376153e0],
        [-1.210790395648774600376153e0, 2.957817486745968516275933e-1],
        [-8.129109323794436617309965e-1, 1.775082676544971536104957e0],
        [
            -1.775082676544971536104957e0,
            -8.129109323794436617309965e-1,
        ],
        [
            -3.033824901668271719065091e0,
            -4.756319696787009224869347e-1,
        ],
        [4.756319696787009224869347e-1, -3.033824901668271719065091e0],
        [0.0, 2.290523463825059058319148e0],
        [-2.290523463825059058319148e0, 0.0],
        [-1.143211493061337609794373e0, 7.256919742818993684282448e-1],
        [
            -7.256919742818993684282448e-1,
            -1.143211493061337609794373e0,
        ],
        [1.505066483321409611546634e-1, 5.512177798223371229746050e0],
        [-5.512177798223371229746050e0, 1.505066483321409611546634e-1],
        [-1.258403907361656564229711e0, 1.084271195519019689714392e-1],
        [
            -1.084271195519019689714392e-1,
            -1.258403907361656564229711e0,
        ],
        [
            -4.696427318716901030374602e0,
            -2.905295912156492788534533e-2,
        ],
        [2.905295912156492788534533e-2, -4.696427318716901030374602e0],
        [3.812657887519453102348782e0, 9.240175630733321726572644e-1],
        [-9.240175630733321726572644e-1, 3.812657887519453102348782e0],
        [-1.579254605699311794424053e0, 2.230775732642575309548556e0],
        [-2.230775732642575309548556e0, -1.579254605699311794424053e0],
        [0.0, -2.148127284390490340311876e0],
        [2.148127284390490340311876e0, 0.0],
        [8.304767403711241557802019e-1, 9.168424106953754985564914e-1],
        [
            -9.168424106953754985564914e-1,
            8.304767403711241557802019e-1,
        ],
        [-1.093343370580977955040283e-1, 6.964109511760239900553575e0],
        [
            -6.964109511760239900553575e0,
            -1.093343370580977955040283e-1,
        ],
        [
            -9.830478193852659338715339e-1,
            -4.930408645726850272549650e0,
        ],
        [4.930408645726850272549650e0, -9.830478193852659338715339e-1],
        [-3.668784408215605555314815e0, 1.321099014977934902596096e0],
        [-1.321099014977934902596096e0, -3.668784408215605555314815e0],
        [6.692327197927725928645667e0, 5.833060195138807600462601e-1],
        [-5.833060195138807600462601e-1, 6.692327197927725928645667e0],
        [-2.772052689219996517971793e0, 1.408225303324276165971399e0],
        [-1.408225303324276165971399e0, -2.772052689219996517971793e0],
        [2.366456892769419266767362e0, -2.398210494357255401558466e0],
        [2.398210494357255401558466e0, 2.366456892769419266767362e0],
        [4.098823572198208698333796e0, 1.384607474490546886637842e0],
        [-1.384607474490546886637842e0, 4.098823572198208698333796e0],
        [0.0, -6.715562169824313753905112e0],
        [6.715562169824313753905112e0, 0.0],
        [2.119845485387684234244233e0, 2.242958037967497127208872e-1],
        [-2.242958037967497127208872e-1, 2.119845485387684234244233e0],
        [-2.790829526506540906028472e-1, 1.703695774156155043565938e0],
        [
            -1.703695774156155043565938e0,
            -2.790829526506540906028472e-1,
        ],
        [
            -5.637866520970707610915052e-2,
            -8.985676310888147070147415e0,
        ],
        [8.985676310888147070147415e0, -5.637866520970707610915052e-2],
        [-2.104080430255426601039538e-1, 2.407704710949948900459577e0],
        [
            -2.407704710949948900459577e0,
            -2.104080430255426601039538e-1,
        ],
        [6.578798306072172514556988e0, -1.428815827393694759535531e0],
        [1.428815827393694759535531e0, 6.578798306072172514556988e0],
        [-2.725027482492237881393611e0, -3.449466548627193112609083e0],
        [3.449466548627193112609083e0, -2.725027482492237881393611e0],
        [4.592223474806729712397734e0, -2.418738615635713239317500e0],
        [2.418738615635713239317500e0, 4.592223474806729712397734e0],
        [7.953964378075752029010631e0, 1.396459390836621803864542e0],
        [-1.396459390836621803864542e0, 7.953964378075752029010631e0],
        [-3.992111914247440429880212e0, -1.792302270179873781395360e0],
        [1.792302270179873781395360e0, -3.992111914247440429880212e0],
        [3.063255210834410086080803e0, -2.335773794299305501602875e0],
        [2.335773794299305501602875e0, 3.063255210834410086080803e0],
        [0.0, -6.299342985943726116214737e0],
        [6.299342985943726116214737e0, 0.0],
        [1.680556315021658511438697e0, -1.018195298513384344000296e0],
        [1.018195298513384344000296e0, 1.680556315021658511438697e0],
        [
            -2.212494362088717630813180e-1,
            -7.733961126240451387235583e0,
        ],
        [7.733961126240451387235583e0, -2.212494362088717630813180e-1],
        [8.873727002741028901504663e-1, -6.460002824013650918112885e0],
        [6.460002824013650918112885e0, 8.873727002741028901504663e-1],
        [3.311720002672540992469153e0, 1.730952539797229861787511e0],
        [-1.730952539797229861787511e0, 3.311720002672540992469153e0],
        [2.666872102228439158494715e0, -3.838396317725032995009214e0],
        [3.838396317725032995009214e0, 2.666872102228439158494715e0],
        [-1.104654593857466698669160e0, -9.266708448014722304930567e0],
        [9.266708448014722304930567e0, -1.104654593857466698669160e0],
        [5.276726523822799804450645e0, 1.646199494119574120103499e0],
        [-1.646199494119574120103499e0, 5.276726523822799804450645e0],
        [9.139558436907395029854986e0, -9.504337210697618740141918e-1],
        [9.504337210697618740141918e-1, 9.139558436907395029854986e0],
        [-8.421752183231222714622096e0, -2.328300465398103852933381e0],
        [2.328300465398103852933381e0, -8.421752183231222714622096e0],
        [6.462237736264047455245846e0, -3.034300242104796774717114e0],
        [3.034300242104796774717114e0, 6.462237736264047455245846e0],
        [-3.603997313243659465912131e0, 2.108688999663617703928167e0],
        [-2.108688999663617703928167e0, -3.603997313243659465912131e0],
        [-3.603997313243659465912131e0, -2.108688999663617703928167e0],
        [2.108688999663617703928167e0, -3.603997313243659465912131e0],
    ];
    /// `angular_basis(6, -1.5, 19.25)` from `derivations/wind_model_vsh.gp`, 4.1° from the pole.
    #[expect(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        reason = "verbatim 25-digit derivations/wind_model_vsh.gp output"
    )]
    const WIND_BASIS_POLAR: [[f64; 2]; 96] = [
        [0.0, 1.225204272737073590909257e-1],
        [-1.225204272737073590909257e-1, 0.0],
        [1.640131095503299317551307e0, 5.553548205435788886306645e-1],
        [-5.553548205435788886306645e-1, 1.640131095503299317551307e0],
        [5.567494854628491285592345e-1, -1.636022545137956671278721e0],
        [1.636022545137956671278721e0, 5.567494854628491285592345e-1],
        [0.0, -4.733308965107719802953850e-1],
        [4.733308965107719802953850e-1, 0.0],
        [-3.658257623650689171012927e0, -1.232471057768706099311872e0],
        [1.232471057768706099311872e0, -3.658257623650689171012927e0],
        [-1.241811130367639102999125e0, 3.630742657038671457129588e0],
        [-3.630742657038671457129588e0, -1.241811130367639102999125e0],
        [
            1.667787185819312538051999e-1,
            -2.168057923987503477955952e-1,
        ],
        [2.168057923987503477955952e-1, 1.667787185819312538051999e-1],
        [
            -2.173502577059158887461428e-1,
            -1.663609356577248798135786e-1,
        ],
        [
            1.663609356577248798135786e-1,
            -2.173502577059158887461428e-1,
        ],
        [0.0, 1.115894759102938030651213e0],
        [-1.115894759102938030651213e0, 0.0],
        [6.098424795508302252370473e0, 2.038956717187831705375106e0],
        [-2.038956717187831705375106e0, 6.098424795508302252370473e0],
        [2.070136269193304955231916e0, -6.006572797215882965874506e0],
        [6.006572797215882965874506e0, 2.070136269193304955231916e0],
        [
            -4.401496636263575991146268e-1,
            5.707385826669615248754024e-1,
        ],
        [
            -5.707385826669615248754024e-1,
            -4.401496636263575991146268e-1,
        ],
        [5.736142094853911815971081e-1, 4.379431175612204499346275e-1],
        [
            -4.379431175612204499346275e-1,
            5.736142094853911815971081e-1,
        ],
        [
            -1.744395680525249377473553e-2,
            -2.604132857937700748031500e-2,
        ],
        [
            2.604132857937700748031500e-2,
            -1.744395680525249377473553e-2,
        ],
        [
            -2.610672627842875572554245e-2,
            1.740025945977704040566371e-2,
        ],
        [
            -1.740025945977704040566371e-2,
            -2.610672627842875572554245e-2,
        ],
        [0.0, -2.098264222042702081764729e0],
        [2.098264222042702081764729e0, 0.0],
        [-8.882398128663279672299398e0, -2.939351738584620567705372e0],
        [2.939351738584620567705372e0, -8.882398128663279672299398e0],
        [-3.015167873694829861266913e0, 8.659050996816794537163147e0],
        [-8.659050996816794537163147e0, -3.015167873694829861266913e0],
        [8.615486382398366809514802e-1, -1.113402928923660941217830e0],
        [1.113402928923660941217830e0, 8.615486382398366809514802e-1],
        [
            -1.122792045290944257199954e0,
            -8.543441158579452161448878e-1,
        ],
        [8.543441158579452161448878e-1, -1.122792045290944257199954e0],
        [5.220077837933112121699112e-2, 7.779765253132103579958044e-2],
        [
            -7.779765253132103579958044e-2,
            5.220077837933112121699112e-2,
        ],
        [
            7.812398573813102244094500e-2,
            -5.198272950169698746039611e-2,
        ],
        [5.198272950169698746039611e-2, 7.812398573813102244094500e-2],
        [
            -3.033979802352486235228089e-3,
            8.109159811454981830960223e-4,
        ],
        [
            -8.109159811454981830960223e-4,
            -3.033979802352486235228089e-3,
        ],
        [8.129524378926859696937556e-4, 3.026379642304564967272748e-3],
        [
            -3.026379642304564967272748e-3,
            8.129524378926859696937556e-4,
        ],
        [0.0, 3.457731631312391434779597e0],
        [-3.457731631312391434779597e0, 0.0],
        [1.195131555020934099446545e1, 3.903585154258161255365622e0],
        [-3.903585154258161255365622e0, 1.195131555020934099446545e1],
        [4.056924962538650401848693e0, -1.149959104159975645814726e1],
        [1.149959104159975645814726e1, 4.056924962538650401848693e0],
        [-1.448854486045380803211431e0, 1.864474726587073718880915e0],
        [-1.864474726587073718880915e0, -1.448854486045380803211431e0],
        [1.888183927769144939223639e0, 1.430661776114954887566803e0],
        [-1.430661776114954887566803e0, 1.888183927769144939223639e0],
        [
            -1.150587613707469426565252e-1,
            -1.711179920873891844022179e-1,
        ],
        [
            1.711179920873891844022179e-1,
            -1.150587613707469426565252e-1,
        ],
        [
            -1.721976053126132346946610e-1,
            1.143373868764370754405373e-1,
        ],
        [
            -1.143373868764370754405373e-1,
            -1.721976053126132346946610e-1,
        ],
        [
            1.003736574669422776886188e-2,
            -2.679393950649996469799648e-3,
        ],
        [2.679393950649996469799648e-3, 1.003736574669422776886188e-2],
        [
            -2.689504045962531288300224e-3,
            -9.999634357318483878602509e-3,
        ],
        [
            9.999634357318483878602509e-3,
            -2.689504045962531288300224e-3,
        ],
        [
            -1.905117059062938594258184e-5,
            2.899365754699383271774375e-4,
        ],
        [
            -2.899365754699383271774375e-4,
            -1.905117059062938594258184e-5,
        ],
        [2.906646944231968606534764e-4, 1.900344715309141507188060e-5],
        [
            -1.900344715309141507188060e-5,
            2.906646944231968606534764e-4,
        ],
        [0.0, -5.222815901111764851985917e0],
        [5.222815901111764851985917e0, 0.0],
        [-1.525685891428323741972377e1, -4.904255967280034180151168e0],
        [4.904255967280034180151168e0, -1.525685891428323741972377e1],
        [-5.179005735331104967804221e0, 1.444747219758379371724484e1],
        [-1.444747219758379371724484e1, -5.179005735331104967804221e0],
        [2.216309269191887859138636e0, -2.837511226670137462172268e0],
        [2.837511226670137462172268e0, 2.216309269191887859138636e0],
        [-2.888350473674018308636379e0, -2.177298942917281585233451e0],
        [2.177298942917281585233451e0, -2.888350473674018308636379e0],
        [2.158339118337660893270973e-1, 3.201808518767109387931318e-1],
        [
            -3.201808518767109387931318e-1,
            2.158339118337660893270973e-1,
        ],
        [
            3.230182762290494133984036e-1,
            -2.139380054948215184600076e-1,
        ],
        [2.139380054948215184600076e-1, 3.230182762290494133984036e-1],
        [
            -2.432609222402915550790865e-2,
            6.483836218520547987342421e-3,
        ],
        [
            -6.483836218520547987342421e-3,
            -2.432609222402915550790865e-2,
        ],
        [6.518156766433625920310864e-3, 2.419800619547394666191071e-2],
        [
            -2.419800619547394666191071e-2,
            6.518156766433625920310864e-3,
        ],
        [
            6.851790312104126781536469e-5,
            -1.041713702784926920649693e-3,
        ],
        [1.041713702784926920649693e-3, 6.851790312104126781536469e-5],
        [
            -1.045381189489297326427681e-3,
            -6.827752334260798552617562e-5,
        ],
        [
            6.827752334260798552617562e-5,
            -1.045381189489297326427681e-3,
        ],
        [2.377654015706728770290327e-5, 9.823894611109532795860927e-6],
        [
            -9.823894611109532795860927e-6,
            2.377654015706728770290327e-5,
        ],
        [
            9.848565399365790123334693e-6,
            -2.371697960546459638260579e-5,
        ],
        [2.371697960546459638260579e-5, 9.848565399365790123334693e-6],
    ];

    /// (G1) The basis is the vector spherical harmonic `derivations/wind_model_vsh.mac` derives —
    /// `(∂Y/∂λ / cos φ, ∂Y/∂φ)` and its rotation by 90° — evaluated independently of the module's
    /// three recursions: `derivations/wind_model_vsh.gp` builds `P̄ₙₘ` from Rodrigues' formula and
    /// differentiates it in closed form at 100 digits. Degree 6 at a mid-latitude point and at one
    /// 4.1° from the south pole, where the `P̄/cos φ` recursion carries the `1/cos φ` the
    /// direct quotient would lose digits to.
    ///
    /// The bound is 64 ulps of the term's own size plus one: the recursions run at most six steps,
    /// each a two-term combination with factors under 4, so the rounding they can accumulate is a
    /// few dozen ulps, while a wrong normalisation or a sign is off at order one.
    #[test]
    fn the_basis_is_the_vector_harmonic_rodrigues_gives() {
        let mut worst = 0.0_f64;
        for (phi, lst, want) in [
            (0.625, 5.5, &WIND_BASIS_MID),
            (-1.5, 19.25, &WIND_BASIS_POLAR),
        ] {
            let got = angular_basis(6, phi, lst);
            assert_eq!(got.len(), want.len());
            for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                for c in 0..2 {
                    let bound = 64.0 * f64::EPSILON * (1.0 + w[c].abs());
                    let miss = (g[c] - w[c]).abs();
                    assert!(
                        miss <= bound,
                        "φ={phi} lst={lst}: term {k} component {c} = {:e}, want {:e}",
                        g[c],
                        w[c]
                    );
                    worst = worst.max(miss / bound);
                }
            }
        }
        println!("vector basis vs Rodrigues: worst {worst:.3} of the bound");
    }

    /// (G2) The fitted field is the stationary point of the ridge objective it is defined by,
    /// `Σ wᵢ (d̂ᵢ·w(xᵢ) − yᵢ)² + λ_q |c_q|² + λ_d |c_d|²` — with the quiet and the high-latitude
    /// blocks solved together.
    ///
    /// The design rows are not taken from `fit`: each column is the projection the *evaluator*
    /// [`HorizontalWindField::velocity`] gives for a unit coefficient vector, so the test holds the
    /// fit's assembly (the driver products, the window, and the transposed magnetic rotation that
    /// `derivations/wind_model_vsh.mac` shows is the evaluator's rotation seen from the data side)
    /// against the evaluator, and its Cholesky solve against the normal equations. Each gradient
    /// component is held to 1e-12 of `Σ|w r y|`, the size of the terms it sums; the measured worst
    /// is 2.4e-15, and the fit's rotation taken untransposed fails it at 4.3e-2 on the first one.
    #[test]
    fn the_fitted_field_is_the_stationary_point_of_its_ridge_objective() {
        let spec = FitSpec {
            degree: 1,
            ridge: 0.3,
            disturbance_degree: 1,
            disturbance_ridge: 1.7,
        };
        let samples: Vec<CrosswindSample> = (0..160)
            .map(|i| {
                let f = f64::from(i);
                let lat = 1.45 * libm::sin(0.37 * f + 0.2);
                let heading = 0.91 * f;
                CrosswindSample {
                    point: WindPoint {
                        latitude: lat,
                        longitude: -3.0 + 0.0377 * f,
                        local_solar_time: (0.613 * f) % 24.0,
                        altitude: 250_000.0 + 1_500.0 * f,
                        year: 2015.0,
                        day_of_year: 1.0 + (2.3 * f) % 365.0,
                        f107a: 80.0 + 0.5 * f,
                        ap: (7.0 * f) % 90.0,
                    },
                    direction_north: libm::sin(heading),
                    direction_east: libm::cos(heading),
                    speed: 40.0 * libm::sin(0.13 * f) + 15.0 * libm::cos(0.71 * f),
                    weight: 0.5 + f64::from(i % 4) * 0.25,
                }
            })
            .collect();
        let field = HorizontalWindField::fit(&samples, &spec).unwrap();
        let (nq, nd) = (
            DRIVERS * angular_terms(spec.degree),
            DISTURBANCE_DRIVERS * angular_terms(spec.disturbance_degree),
        );
        assert!(
            samples
                .iter()
                .any(|s| disturbance_window(magnetic_point(&s.point).latitude) > 0.0),
            "the sample must reach the high-latitude block"
        );
        let project = |coeffs: Vec<f64>, disturbance: Vec<f64>, s: &CrosswindSample| {
            let unit = HorizontalWindField::from_coefficients(
                spec.degree,
                coeffs,
                spec.disturbance_degree,
                disturbance,
            )
            .unwrap();
            let [e, n] = unit.velocity(&s.point);
            s.direction_east * e + s.direction_north * n
        };
        let rows: Vec<Vec<f64>> = samples
            .iter()
            .map(|s| {
                (0..nq + nd)
                    .map(|j| {
                        let mut cq = vec![0.0; nq];
                        let mut cd = vec![0.0; nd];
                        if j < nq {
                            cq[j] = 1.0;
                        } else {
                            cd[j - nq] = 1.0;
                        }
                        project(cq, cd, s)
                    })
                    .collect()
            })
            .collect();
        let c: Vec<f64> = field
            .coefficients()
            .iter()
            .chain(field.disturbance_coefficients())
            .copied()
            .collect();
        let mut worst = 0.0_f64;
        for j in 0..nq + nd {
            let (mut grad, mut scale) = (0.0, 0.0);
            for (s, row) in samples.iter().zip(&rows) {
                let predicted: f64 = row.iter().zip(&c).map(|(r, ck)| r * ck).sum();
                grad += s.weight * row[j] * (predicted - s.speed);
                scale += (s.weight * row[j] * s.speed).abs();
            }
            grad += if j < nq {
                spec.ridge
            } else {
                spec.disturbance_ridge
            } * c[j];
            let rel = grad.abs() / scale.max(1.0);
            assert!(rel < 1e-12, "∂J/∂c[{j}] = {grad:e} (relative {rel:.3e})");
            worst = worst.max(rel);
        }
        println!("ridge objective gradient at the fit: worst {worst:.3e} relative");
    }

    #[test]
    fn the_basis_is_the_gradient_of_its_harmonic() {
        // The whole construction rests on `(u, v)` being `(∂Y/∂λ / cos φ, ∂Y/∂φ)`. Both derivatives
        // come from hand-differentiated recursions, so they are checked against central differences
        // of the value recursion itself — a wrong normalization factor in one and not the other would
        // otherwise be invisible, and would read as a plausible but wrong wind pattern.
        //
        // Degree 6 rather than the 4 the fits use: the three-term recursion only reaches back two
        // degrees, so the `n − 2` back-reference and the `m + 2 ≤ n` guard that selects it are both
        // degenerate below degree 5 — the terms they would pick up are still zero there, and an
        // index or a bound off by one reads as correct.
        let degree = 6;
        let h = 1e-6;
        for &(phi, lst) in &[
            (0.0, 0.0),
            (0.7, 6.0),
            (-1.1, 13.5),
            (0.31, 21.0),
            (-0.05, 3.25),
        ] {
            let lambda = lst * (core::f64::consts::TAU / 24.0);
            let basis = angular_basis(degree, phi, lst);
            for n in 1..=degree {
                for m in 0..=n {
                    for sine in [false, true] {
                        if m == 0 && sine {
                            continue;
                        }
                        let d_phi = (harmonic(degree, n, m, sine, phi + h, lambda)
                            - harmonic(degree, n, m, sine, phi - h, lambda))
                            / (2.0 * h);
                        let d_lambda = (harmonic(degree, n, m, sine, phi, lambda + h)
                            - harmonic(degree, n, m, sine, phi, lambda - h))
                            / (2.0 * h * libm::cos(phi));
                        let sph = basis[term(degree, n, m, sine, false)];
                        let tor = basis[term(degree, n, m, sine, true)];
                        let scale = 1.0 + d_phi.abs() + d_lambda.abs();
                        assert!(
                            (sph[0] - d_lambda).abs() < 1e-6 * scale
                                && (sph[1] - d_phi).abs() < 1e-6 * scale,
                            "spheroidal ({n},{m},{sine}) at φ={phi}: {sph:?} vs [{d_lambda},{d_phi}]"
                        );
                        assert!(
                            (tor[0] + d_phi).abs() < 1e-6 * scale
                                && (tor[1] - d_lambda).abs() < 1e-6 * scale,
                            "toroidal ({n},{m},{sine}) at φ={phi}: {tor:?} vs [{},{d_lambda}]",
                            -d_phi
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn every_basis_term_is_finite_and_bounded_at_the_pole() {
        // A scalar expansion of the eastward wind would have a non-zero, direction-dependent value at
        // the pole; the gradient forms must not. Approaching the pole, the `1/cos φ` factor and the
        // vanishing `P̄_nm` have to cancel exactly — if the `P̄/cos φ` recursion were wrong this would
        // blow up, and the near-polar orbits that dominate the data are exactly where it would bite.
        let degree = 6;
        for &eps in &[1e-3, 1e-6, 1e-9, 0.0] {
            let phi = core::f64::consts::FRAC_PI_2 - eps;
            for term in angular_basis(degree, phi, 7.5) {
                assert!(
                    term[0].is_finite() && term[1].is_finite(),
                    "non-finite basis at φ = π/2 − {eps}: {term:?}"
                );
                assert!(
                    term[0].abs() < 1e3 && term[1].abs() < 1e3,
                    "unbounded basis at φ = π/2 − {eps}: {term:?}"
                );
            }
        }
    }

    #[test]
    fn every_driver_is_the_regressor_its_documentation_names() {
        // The fit only ever sees these nine numbers multiplied by an angular basis, so a driver that
        // is scaled or offset wrongly is absorbed into the coefficients and never shows up as a
        // residual. Each one is pinned here at a point chosen to make it exactly a half.
        let point = WindPoint {
            latitude: 0.4,
            longitude: 1.1,
            local_solar_time: 9.0,
            altitude: REFERENCE_ALTITUDE + 0.5 * ALTITUDE_SCALE,
            year: 2024.0,
            // A quarter of the tropical year: the annual angle is exactly π/2 and the semiannual π.
            day_of_year: 365.25 / 4.0,
            f107a: REFERENCE_F107 + 0.5 * F107_SCALE,
            ap: REFERENCE_AP + 0.5 * AP_SCALE,
        };
        let d = driver_terms(&point);
        let expected = [1.0, 0.5, 0.25, 0.5, 0.5, 0.0, 1.0, -1.0, 0.0];
        for (k, (&got, &want)) in d.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-12, "driver {k}: {got} vs {want}");
        }
        // Altitude enters as a signed offset from the reference and quadratically, so a point below
        // the reference has a negative linear term and the same positive square.
        let low = WindPoint {
            altitude: REFERENCE_ALTITUDE - 0.5 * ALTITUDE_SCALE,
            ..point
        };
        let l = driver_terms(&low);
        assert!((l[1] + 0.5).abs() < 1e-12 && (l[2] - 0.25).abs() < 1e-12);
    }

    #[test]
    fn the_activity_driver_is_the_kp_like_ladder_it_claims_to_be() {
        // The argument for the logarithm is the ladder it produces on the `ap` values that stand for
        // Kp = 1..9, so that ladder is the thing to pin — a driver rescaled anywhere along it would
        // be absorbed by the coefficients and leave no residual behind.
        let ladder = [4.0, 7.0, 15.0, 27.0, 48.0, 80.0, 132.0, 207.0, 400.0];
        let expected = [0.24, 0.38, 0.69, 1.03, 1.44, 1.85, 2.28, 2.69, 3.32];
        for (&ap, &want) in ladder.iter().zip(expected.iter()) {
            let g = disturbance_drivers(&WindPoint {
                ap,
                ..at(0.0, 0.0, 12.0)
            });
            assert_eq!(g[0], 1.0);
            assert!((g[1] - want).abs() < 0.005, "ap {ap}: {} vs {want}", g[1]);
            assert!((g[2] - g[1] * g[1]).abs() < 1e-12);
        }
        // Near-constant per unit of Kp over the range a storm model has to span, which is what makes
        // a quadratic in `a` enough.
        for k in 2..7 {
            let step = disturbance_drivers(&WindPoint {
                ap: ladder[k + 1],
                ..at(0.0, 0.0, 12.0)
            })[1]
                - disturbance_drivers(&WindPoint {
                    ap: ladder[k],
                    ..at(0.0, 0.0, 12.0)
                })[1];
            assert!((0.33..0.45).contains(&step), "Kp step {step} at rung {k}");
        }
        // Quiet and unphysical inputs both land on the zero of the driver rather than on a negative
        // logarithm or a NaN.
        for ap in [0.0, -1.0, -1000.0] {
            let g = disturbance_drivers(&WindPoint {
                ap,
                ..at(0.0, 0.0, 12.0)
            });
            assert_eq!(g, [1.0, 0.0, 0.0], "ap {ap}");
        }
    }

    #[test]
    fn the_solar_declination_is_the_obliquity_rotation_and_costs_what_it_claims() {
        // Structure first: zero at both equinoxes and ±ε at both solstices, with ε the IAU 2006
        // obliquity. A quarter of the year past the March equinox is the June solstice.
        const EQUINOX: f64 = 79.0;
        const YEAR: f64 = 365.25;
        let obliquity = (84_381.406 / 3600.0_f64).to_radians();
        assert!(solar_declination(EQUINOX).abs() < 1e-12);
        assert!(solar_declination(EQUINOX + YEAR / 2.0).abs() < 1e-12);
        assert!((solar_declination(EQUINOX + YEAR / 4.0) - obliquity).abs() < 1e-12);
        assert!((solar_declination(EQUINOX + 3.0 * YEAR / 4.0) + obliquity).abs() < 1e-12);
        // Then the size of the approximation, measured rather than asserted away. Reference is the
        // geometric declination of date from `pyerfa` (`epv00` for the Sun's geocentric direction,
        // `pnm06a` onto the equator of date) at 12:00 UTC on every day of 2023-2026: the worst
        // departure over those four years is 1.727° on 2024-09-25, doy 269. That is the September
        // equinox, where the equation of centre this model drops is largest and the declination it
        // is being compared against is itself passing through zero. Away from the equinoxes the
        // same comparison is within a twentieth of a degree, which the solstices below show.
        let deg = core::f64::consts::PI / 180.0;
        for (doy, truth_deg, low, high) in [
            (269.0, -1.158_316, 1.70, 1.75),
            (172.0, 23.437_781, 0.0, 0.05),
            (355.0, -23.435_297, 0.0, 0.05),
        ] {
            let error = (solar_declination(doy) - truth_deg * deg).abs() / deg;
            assert!(
                (low..high).contains(&error),
                "doy {doy}: {error:.4}° from the true declination"
            );
        }
    }

    #[test]
    fn the_magnetic_coordinates_track_the_dipole_and_the_sun() {
        let dipole = CentredDipole::at_year(2024.0);
        let (pole_latitude, pole_longitude) = dipole.pole();
        // At the geomagnetic pole the magnetic latitude is 90°, and the rotation to geographic
        // components is a rotation wherever it is asked for.
        let m = magnetic_point(&WindPoint {
            latitude: pole_latitude,
            longitude: pole_longitude,
            ..at(0.0, 0.0, 12.0)
        });
        assert!((m.latitude - core::f64::consts::FRAC_PI_2).abs() < 1e-9);
        // The rotation is checked away from the pole, where it means something. At the pole itself
        // the magnetic meridian is undefined and the pair is only as good as the ~1e-9 rad round
        // trip through `asin`/`atan2` leaves it — `c² + s²` comes back 1 − 1.1e-7 there, which is
        // harmless because a wind resolved on an undefined pair of axes is arbitrary anyway.
        let m = magnetic_point(&at(0.5, 1.0, 6.0));
        let unit = m
            .cos_rotation
            .mul_add(m.cos_rotation, m.sin_rotation * m.sin_rotation);
        assert!((unit - 1.0).abs() < 1e-12, "not a rotation: {unit}");
        // Magnetic local time is a clock. The only thing local solar time moves is the sub-solar
        // longitude, so an hour of it is an hour of magnetic local time — to within the few minutes
        // the 9.2° dipole tilt bends the meridian by.
        let mut previous = magnetic_point(&at(0.5, 1.0, 0.0)).local_time;
        for k in 1..24 {
            let now = magnetic_point(&at(0.5, 1.0, f64::from(k))).local_time;
            let step = (now - previous).rem_euclid(24.0);
            assert!(
                (0.9..1.1).contains(&step),
                "MLT stepped {step} h at lst {k}"
            );
            previous = now;
        }
        // Local solar time is an angle, so a whole day of it is a whole turn and changes nothing.
        // `24 · (π/12)` is exactly `fl(2π)`, and the only residue is that `fl(2π)` is not the true
        // period of `sin`/`cos`.
        let mut worst: f64 = 0.0;
        for j in 0..8 {
            for k in 0..12 {
                let point = at(-1.2 + 0.35 * f64::from(j), 0.0, 2.0 * f64::from(k));
                let shifted = WindPoint {
                    local_solar_time: point.local_solar_time + 24.0,
                    ..point
                };
                let step =
                    (magnetic_point(&point).local_time - magnetic_point(&shifted).local_time).abs();
                worst = worst.max(step.min(24.0 - step));
            }
        }
        println!("worst |MLT(lst) − MLT(lst + 24 h)|: {worst:e} h");
        assert!(
            worst < 1e-12,
            "a whole day of local solar time moved MLT by {worst} h"
        );
        // The magnetic latitude is the point's, not the sub-solar point's: it does not move when
        // only the clock does.
        assert_eq!(
            magnetic_point(&at(0.5, 1.0, 3.0)).latitude,
            magnetic_point(&at(0.5, 1.0, 19.0)).latitude
        );
    }

    #[test]
    fn a_coefficient_vector_of_the_wrong_length_is_rejected() {
        // The packing is `drivers · angular_terms(degree)` per block and nothing checks it later, so
        // a wrong length here would read past the end of one block into the next.
        let j = angular_terms(2);
        let k = angular_terms(3);
        // A well-formed one is accepted, and what it was handed is what it reports back: the two
        // degrees are what the caller must pass to `angular_terms` to slice either block, so an
        // accessor that quietly returned something else would mis-slice the vector it accompanies.
        let coeffs: Vec<f64> = (0..DRIVERS * j).map(|i| i as f64 - 3.0).collect();
        let disturbance: Vec<f64> = (0..DISTURBANCE_DRIVERS * k)
            .map(|i| 2.0 - i as f64)
            .collect();
        let field =
            HorizontalWindField::from_coefficients(2, coeffs.clone(), 3, disturbance.clone())
                .unwrap();
        assert_eq!(field.degree(), 2);
        assert_eq!(field.disturbance_degree(), 3);
        assert_eq!(field.coefficients(), coeffs.as_slice());
        assert_eq!(field.disturbance_coefficients(), disturbance.as_slice());
        assert!(
            HorizontalWindField::from_coefficients(2, vec![0.0; DRIVERS * j - 1], 0, Vec::new())
                .is_none()
        );
        assert!(
            HorizontalWindField::from_coefficients(2, vec![0.0; DRIVERS * j], 3, Vec::new())
                .is_none()
        );
        assert!(HorizontalWindField::from_coefficients(2, Vec::new(), 0, Vec::new()).is_none());
    }

    /// A point where no driver and no magnetic term is incidentally zero.
    fn excited(latitude: f64, longitude: f64, local_solar_time: f64) -> WindPoint {
        WindPoint {
            latitude,
            longitude,
            local_solar_time,
            altitude: 550e3,
            year: 2024.0,
            day_of_year: 50.0,
            f107a: 175.0,
            ap: 40.0,
        }
    }

    #[test]
    fn every_global_coefficient_contributes_its_driver_times_its_basis_term() {
        // `velocity` is the only place the packing that `from_coefficients` documents is actually
        // dereferenced, and the fit inverts the same expression — a transposed block stride or a
        // sign on one accumulator would round-trip perfectly and still be wrong for any coefficient
        // vector a caller wrote by hand. Setting one coefficient at a time and reading the answer
        // off `driver_terms` and `angular_basis`, each pinned by its own test, checks the packing
        // against its documentation instead of against itself.
        let degree = 2;
        let j_max = angular_terms(degree);
        let point = excited(0.4, -0.9, 15.5);
        let drivers = driver_terms(&point);
        let basis = angular_basis(degree, point.latitude, point.local_solar_time);
        assert!(
            drivers.iter().all(|g| g.abs() > 1e-3),
            "the point must excite every driver: {drivers:?}"
        );
        for (gi, &g) in drivers.iter().enumerate() {
            for (j, b) in basis.iter().enumerate() {
                let mut c = vec![0.0; DRIVERS * j_max];
                c[gi * j_max + j] = 2.5;
                let field =
                    HorizontalWindField::from_coefficients(degree, c, 0, Vec::new()).unwrap();
                let w = field.velocity(&point);
                let expect = [2.5 * g * b[0], 2.5 * g * b[1]];
                assert!(
                    (w[0] - expect[0]).abs() < 1e-12 && (w[1] - expect[1]).abs() < 1e-12,
                    "coefficient ({gi},{j}): {w:?} vs {expect:?}"
                );
            }
        }
    }

    #[test]
    fn every_high_latitude_coefficient_is_rotated_from_magnetic_to_geographic() {
        // The high-latitude block is expanded on magnetic coordinates and has to come back as
        // geographic components. The rotation is the one place in the model where a sign error
        // stays plausible — it mirrors the wind about the magnetic meridian, which at 60° N is a
        // rotation of tens of degrees, not a small perturbation.
        let degree = 2;
        let j_max = angular_terms(degree);
        let point = excited(1.047, 0.0, 3.0);
        let magnetic = magnetic_point(&point);
        let window = disturbance_window(magnetic.latitude);
        assert!(
            (window - 1.0).abs() < f64::EPSILON,
            "the point must sit inside the fully-open window: magnetic latitude {} deg",
            magnetic.latitude.to_degrees()
        );
        let drivers = disturbance_drivers(&point);
        let basis = angular_basis(degree, magnetic.latitude, magnetic.local_time);
        assert!(
            magnetic.sin_rotation.abs() > 1e-2 && magnetic.cos_rotation.abs() > 1e-2,
            "the rotation must be a real one: {magnetic:?}"
        );
        for (gi, &g) in drivers.iter().enumerate() {
            for (j, b) in basis.iter().enumerate() {
                let mut d = vec![0.0; DISTURBANCE_DRIVERS * j_max];
                d[gi * j_max + j] = 2.5;
                let field = HorizontalWindField::from_coefficients(
                    degree,
                    vec![0.0; DRIVERS * j_max],
                    degree,
                    d,
                )
                .unwrap();
                let w = field.velocity(&point);
                let (me, mn) = (2.5 * g * b[0], 2.5 * g * b[1]);
                let expect = [
                    magnetic.cos_rotation * me + magnetic.sin_rotation * mn,
                    magnetic.cos_rotation * mn - magnetic.sin_rotation * me,
                ];
                assert!(
                    (w[0] - expect[0]).abs() < 1e-12 && (w[1] - expect[1]).abs() < 1e-12,
                    "high-latitude coefficient ({gi},{j}): {w:?} vs {expect:?}"
                );
            }
        }
    }

    #[test]
    fn the_basis_counts_what_it_says_it_does() {
        for degree in 1..=6 {
            assert_eq!(
                angular_basis(degree, 0.3, 11.0).len(),
                angular_terms(degree)
            );
            assert_eq!(angular_terms(degree), 2 * ((degree + 1) * (degree + 1) - 1));
        }
    }

    /// A field whose only non-zero coefficient is the `n = 1, m = 0` toroidal constant term, which is
    /// a pure zonal flow varying as `cos φ`-ish — the simplest non-trivial wind there is.
    fn zonal_field(degree: usize, amplitude: f64) -> HorizontalWindField {
        let j = angular_terms(degree);
        let mut c = vec![0.0; DRIVERS * j];
        c[term(degree, 1, 0, false, true)] = amplitude;
        HorizontalWindField::from_coefficients(degree, c, 0, Vec::new()).unwrap()
    }

    /// A field with every coefficient excited, so neither horizontal component is incidentally
    /// zero. [`zonal_field`] carries one toroidal `(1, 0)` term, whose `∂Y/∂λ` is identically zero
    /// — its wind is purely eastward, and anything that only touches the northward half of an
    /// expression is invisible to it.
    fn mixed_field(degree: usize) -> HorizontalWindField {
        let j = angular_terms(degree);
        let c = (0..DRIVERS * j).map(|i| 3.0 - (i % 7) as f64).collect();
        HorizontalWindField::from_coefficients(degree, c, 0, Vec::new()).unwrap()
    }

    #[test]
    fn a_known_field_round_trips_through_the_fit() {
        // The fit must recover a field it was given. The samples are crosswind *projections*, so this
        // also checks that the design row is built from `d̂` and not from the field's own components.
        let degree = 2;
        let truth = zonal_field(degree, 40.0);
        let mut samples = Vec::new();
        for i in 0..24 {
            for k in 0..17 {
                let point = at(-1.4 + 0.175 * f64::from(k), 0.0, f64::from(i));
                // Two independent directions per point, so the pair determines the vector: a purely
                // zonal sampling would leave the meridional half of the design null by construction
                // and could not tell a right answer from a regularized one.
                for &(dn, de) in &[(0.0, 1.0), (0.8, 0.6)] {
                    let [east, north] = truth.velocity(&point);
                    samples.push(CrosswindSample {
                        point,
                        direction_north: dn,
                        direction_east: de,
                        speed: de * east + dn * north,
                        weight: 1.0,
                    });
                }
            }
        }
        let fitted = HorizontalWindField::fit(&samples, &quiet_spec(degree, 1e-9)).unwrap();
        assert!(fitted.projection_rms(&samples).unwrap() < 1e-6);
        for point in [at(0.42, 0.0, 9.3), at(-0.9, 0.0, 20.1)] {
            let want = truth.velocity(&point);
            let got = fitted.velocity(&point);
            assert!(
                (got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6,
                "{got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn the_fit_separates_one_driver_block_from_another() {
        // The round trip above samples a single point in driver space — one altitude, one flux, one
        // `ap`, one day — so every driver column is a constant multiple of every other and the fit
        // has no way to tell which block a wind belongs to. It recovers the *velocity* regardless,
        // which is why it passes. This one varies altitude and `ap` across the samples so the blocks
        // are separately identifiable, and then asks for the coefficients back rather than the wind:
        // a mutation of the driver-major packing moves a coefficient between blocks without moving
        // the fitted velocity at the sampled points at all.
        let degree = 1;
        let j = angular_terms(degree);
        let zonal = term(degree, 1, 0, false, true);
        let mut c = vec![0.0; DRIVERS * j];
        c[zonal] = 30.0; // constant
        c[j + zonal] = 12.0; // × (altitude − 400 km)/100 km
        c[4 * j + zonal] = -8.0; // × (ap − 15)/50
        let truth = HorizontalWindField::from_coefficients(degree, c, 0, Vec::new()).unwrap();

        // A factorial design in the drivers, not a line through them: five (altitude, flux, ap)
        // states that are not collinear, crossed with six days spread round the year so the annual
        // and semiannual pairs are separately excited too. Nine drivers need nine independent rows.
        let states = [
            (250e3, 80.0, 4.0),
            (400e3, 150.0, 60.0),
            (500e3, 100.0, 15.0),
            (330e3, 200.0, 120.0),
            (450e3, 120.0, 30.0),
        ];
        let mut samples = Vec::new();
        for i in 0..12 {
            for k in 0..9 {
                for (day, &(alt, f107a, ap)) in
                    (0..6).flat_map(|d| states.iter().map(move |s| (d, s)))
                {
                    let point = WindPoint {
                        latitude: -1.2 + 0.3 * f64::from(k),
                        altitude: alt,
                        f107a,
                        ap,
                        day_of_year: 15.0 + 60.0 * f64::from(day),
                        ..at(0.0, 0.0, 2.0 * f64::from(i))
                    };
                    for &(dn, de) in &[(0.0, 1.0), (0.8, 0.6)] {
                        let [east, north] = truth.velocity(&point);
                        samples.push(CrosswindSample {
                            point,
                            direction_north: dn,
                            direction_east: de,
                            speed: de * east + dn * north,
                            weight: 1.0,
                        });
                    }
                }
            }
        }
        let fitted = HorizontalWindField::fit(&samples, &quiet_spec(degree, 1e-9)).unwrap();
        assert!(fitted.projection_rms(&samples).unwrap() < 1e-6);
        for (block, want) in [(0, 30.0), (1, 12.0), (4, -8.0)] {
            let got = fitted.coefficients()[block * j + zonal];
            assert!((got - want).abs() < 1e-4, "block {block}: {got} vs {want}");
        }
    }

    #[test]
    fn the_design_row_of_one_sample_is_the_row_the_documentation_describes() {
        // A single sample makes the normal equations rank one plus the ridge, and that has a closed
        // form: for `A = w r rᵀ + D` and `b = w y r`, Sherman-Morrison gives
        // `c = w y D⁻¹r / (1 + w rᵀD⁻¹r)`. So the row, the driver-major packing, the magnetic
        // rotation, the window, the per-block ridge and the Cholesky can all be checked at once
        // against an expression built here out of `angular_basis`, `driver_terms` and
        // `magnetic_point`, each pinned by its own test. The round trips above cannot do this: they
        // solve for a field and then compare fields, so a design row and a `velocity` that are
        // wrong in the same way agree with each other.
        let (degree, dd) = (1, 1);
        let point = excited(1.047, 0.0, 3.0);
        let (de, dn) = (0.6, -0.8);
        let (speed, weight) = (37.5, 2.5);
        let spec = FitSpec {
            degree,
            ridge: 2.0,
            disturbance_degree: dd,
            // Deliberately not `ridge`: the two blocks are shrunk by different amounts, so a fit
            // that applied one of them to both would land somewhere else entirely.
            disturbance_ridge: 5.0,
        };
        let sample = CrosswindSample {
            point,
            direction_north: dn,
            direction_east: de,
            speed,
            weight,
        };
        let fitted = HorizontalWindField::fit(&[sample], &spec).unwrap();

        let mut row = Vec::new();
        let basis = angular_basis(degree, point.latitude, point.local_solar_time);
        for g in driver_terms(&point) {
            row.extend(basis.iter().map(|b| g * (de * b[0] + dn * b[1])));
        }
        let magnetic = magnetic_point(&point);
        let window = disturbance_window(magnetic.latitude);
        assert!(
            window > 0.0,
            "the sample must sit inside the window: {magnetic:?}"
        );
        let magnetic_basis = angular_basis(dd, magnetic.latitude, magnetic.local_time);
        let me = magnetic.cos_rotation * de - magnetic.sin_rotation * dn;
        let mn = magnetic.sin_rotation * de + magnetic.cos_rotation * dn;
        for g in disturbance_drivers(&point) {
            row.extend(
                magnetic_basis
                    .iter()
                    .map(|b| g * window * (me * b[0] + mn * b[1])),
            );
        }

        let quiet = DRIVERS * angular_terms(degree);
        assert_eq!(row.len(), quiet + DISTURBANCE_DRIVERS * angular_terms(dd));
        let shrink = |i: usize| {
            if i < quiet {
                spec.ridge
            } else {
                spec.disturbance_ridge
            }
        };
        let denom = 1.0
            + weight
                * row
                    .iter()
                    .enumerate()
                    .map(|(i, r)| r * r / shrink(i))
                    .sum::<f64>();
        let got: Vec<f64> = fitted
            .coefficients()
            .iter()
            .chain(fitted.disturbance_coefficients())
            .copied()
            .collect();
        assert_eq!(got.len(), row.len());
        for (i, (&g, &r)) in got.iter().zip(&row).enumerate() {
            let want = weight * speed * r / shrink(i) / denom;
            assert!(
                (g - want).abs() < 1e-9 * (1.0 + want.abs()),
                "coefficient {i}: {g} vs {want}"
            );
        }
    }

    #[test]
    fn the_projection_residual_weights_what_it_counts_and_drops_what_it_cannot_use() {
        // The weighted RMS is what every fit above is judged by, so a weight that is silently taken
        // at face value — a NaN, a zero, a negative — poisons the judge rather than the fit. The
        // expected value is accumulated here from the usable rows only.
        let field = mixed_field(1);
        let row = |latitude, lst, dn: f64, de: f64, speed, weight| CrosswindSample {
            point: excited(latitude, 0.0, lst),
            direction_north: dn,
            direction_east: de,
            speed,
            weight,
        };
        let usable = [
            row(0.4, 3.0, -0.8, 0.6, 12.0, 3.0),
            row(-0.7, 15.5, 0.6, 0.8, -40.0, 1.0),
        ];
        let mut samples = usable.to_vec();
        samples.push(row(0.1, 7.0, 0.0, 1.0, 1e6, 0.0));
        samples.push(row(0.2, 9.0, 0.0, 1.0, 1e6, -4.0));
        samples.push(row(0.3, 11.0, 0.0, 1.0, 1e6, f64::NAN));

        let (mut sum, mut total) = (0.0, 0.0);
        for s in &usable {
            let [east, north] = field.velocity(&s.point);
            let r = s.direction_east * east + s.direction_north * north - s.speed;
            sum += s.weight * r * r;
            total += s.weight;
        }
        // 4.0 rather than 1.0, so the mean is not the sum and `sum / total` is not `sum % total`.
        assert!((total - 4.0).abs() < f64::EPSILON);
        let want = libm::sqrt(sum / total);
        assert!(
            want > 1.0,
            "the residual must be large enough to compare: {want}"
        );
        let got = field.projection_rms(&samples).unwrap();
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        assert!(field.projection_rms(&samples[2..]).is_none());
    }

    #[test]
    fn a_fit_with_nothing_to_fit_declines_rather_than_returning_a_field() {
        // Every way there can be no usable equation. A field returned here would be a field of
        // whatever the regularizer alone produced, and nothing downstream could tell the difference.
        let point = at(0.3, 0.0, 8.0);
        let good = CrosswindSample {
            point,
            direction_north: 0.0,
            direction_east: 1.0,
            speed: 25.0,
            weight: 1.0,
        };
        assert!(HorizontalWindField::fit(&[], &quiet_spec(2, 1.0)).is_none());
        assert!(HorizontalWindField::fit(&[good], &quiet_spec(0, 1.0)).is_none());
        for weight in [0.0, -1.0, f64::NAN] {
            let dropped = CrosswindSample { weight, ..good };
            assert!(
                HorizontalWindField::fit(&[dropped], &quiet_spec(2, 1.0)).is_none(),
                "weight {weight} was used"
            );
            assert!(
                zonal_field(2, 10.0).projection_rms(&[dropped]).is_none(),
                "weight {weight} counted toward the residual"
            );
        }
        // And a sample that *is* usable makes both of them answer.
        assert!(HorizontalWindField::fit(&[good], &quiet_spec(2, 1.0)).is_some());
        assert!(zonal_field(2, 10.0).projection_rms(&[good]).is_some());
    }

    #[test]
    fn samples_from_keeps_every_stride_th_nominal_row_and_nothing_for_a_zero_stride() {
        let observations = crate::wind_data::parse_crosswind(include_str!(
            "../tests/fixtures/gracefo_crosswind_20240501.txt"
        ));
        assert_eq!(observations.len(), 30);
        let drivers = |_: &CrosswindObservation| (140.0, 22.0);
        assert_eq!(samples_from(&observations, 0, drivers).len(), 0);
        assert_eq!(samples_from(&observations, 1, drivers).len(), 30);
        assert_eq!(samples_from(&observations, 3, drivers).len(), 10);
        let first = samples_from(&observations, 3, drivers)[1];
        let source = observations[3];
        assert_eq!(first.point.latitude, source.latitude);
        assert_eq!(first.point.local_solar_time, source.local_solar_time);
        assert_eq!(first.point.altitude, source.altitude);
        assert_eq!(first.point.day_of_year, f64::from(source.day_of_year()));
        assert_eq!(first.point.year, f64::from(source.year));
        assert_eq!((first.point.f107a, first.point.ap), (140.0, 22.0));
        assert_eq!(first.direction_north, source.direction[0]);
        assert_eq!(first.direction_east, source.direction[1]);
        assert_eq!((first.speed, first.weight), (source.speed, 1.0));
    }

    #[test]
    fn a_purely_zonal_sampling_leaves_the_meridional_wind_to_the_regularizer() {
        // The claim the module docs make about the data, made falsifiable: fit the same truth field
        // from crosswinds that are exactly east-west, and the eastward component still comes back
        // while the northward one is shrunk to nothing. Without the ridge these equations are
        // singular and there is no solution at all.
        let degree = 2;
        let truth = zonal_field(degree, 40.0);
        let mut samples = Vec::new();
        for i in 0..24 {
            for k in 0..17 {
                let point = at(-1.4 + 0.175 * f64::from(k), 0.0, f64::from(i));
                samples.push(CrosswindSample {
                    point,
                    direction_north: 0.0,
                    direction_east: 1.0,
                    speed: truth.velocity(&point)[0],
                    weight: 1.0,
                });
            }
        }
        assert!(HorizontalWindField::fit(&samples, &quiet_spec(degree, 0.0)).is_none());
        let fitted = HorizontalWindField::fit(&samples, &quiet_spec(degree, 1e-6)).unwrap();
        let point = at(0.42, 0.0, 9.3);
        let want = truth.velocity(&point);
        let got = fitted.velocity(&point);
        assert!(
            (got[0] - want[0]).abs() < 1e-3 * want[0].abs().max(1.0),
            "the observed component should survive: {got:?} vs {want:?}"
        );
        // Zero to the precision a wind is measured in: a thousandth of a millimetre per second
        // against an eastward component of tens of metres per second.
        assert!(
            got[1].abs() < 1e-4,
            "the unobserved component should be shrunk to zero, got {}",
            got[1]
        );
    }

    /// A field whose only non-zero coefficient is the high-latitude block's `n = 1, m = 0` toroidal
    /// constant — a zonal flow about the *magnetic* pole, inside the window.
    fn polar_field(degree: usize, amplitude: f64) -> HorizontalWindField {
        let j = angular_terms(degree);
        let mut d = vec![0.0; DISTURBANCE_DRIVERS * j];
        d[term(degree, 1, 0, false, true)] = amplitude;
        HorizontalWindField::from_coefficients(degree, vec![0.0; DRIVERS * j], degree, d).unwrap()
    }

    #[test]
    fn the_high_latitude_window_opens_once_and_is_smooth() {
        assert_eq!(disturbance_window(0.0), 0.0);
        assert_eq!(disturbance_window(DISTURBANCE_ONSET), 0.0);
        assert_eq!(disturbance_window(DISTURBANCE_FULL), 1.0);
        assert_eq!(disturbance_window(core::f64::consts::FRAC_PI_2), 1.0);
        // Where the window sits is a claim about where the auroral oval is, so it is also pinned in
        // degrees rather than only against its own two constants — those are self-consistent under
        // any rescaling of themselves, and the mid-latitudes have to stay strictly outside.
        let deg = core::f64::consts::PI / 180.0;
        assert_eq!(disturbance_window(20.0 * deg), 0.0);
        assert_eq!(disturbance_window(39.0 * deg), 0.0);
        let half = disturbance_window(47.5 * deg);
        assert!(
            (0.0..1.0).contains(&half) && half > 0.0,
            "partly open {half}"
        );
        assert_eq!(disturbance_window(60.0 * deg), 1.0);
        // Symmetric in the hemisphere, monotone, and with a continuous first derivative at both
        // ends — a step there would put a jump in the drag acceleration four times a revolution.
        // The comparison is against the window's own mean slope: a central difference at a point
        // where the true derivative vanishes still returns the O(h) curvature term, so the test is
        // "negligible next to how fast the window opens", not "zero".
        let d = |x: f64| (disturbance_window(x + 1e-6) - disturbance_window(x - 1e-6)) / 2e-6;
        let slope = 1.0 / (DISTURBANCE_FULL - DISTURBANCE_ONSET);
        assert!(
            d(DISTURBANCE_ONSET).abs() < 1e-4 * slope,
            "kink at the onset"
        );
        assert!(
            d(DISTURBANCE_FULL).abs() < 1e-4 * slope,
            "kink at full opening"
        );
        let mut previous = 0.0;
        for k in 0..=90 {
            let x = f64::from(k) * core::f64::consts::PI / 180.0;
            let w = disturbance_window(x);
            assert_eq!(w, disturbance_window(-x));
            assert!(w >= previous, "not monotone at {k}°");
            previous = w;
        }
    }

    #[test]
    fn the_high_latitude_block_is_confined_to_high_latitude() {
        // The window is what makes the two blocks distinguishable, so it has to actually be zero
        // where it claims to be — not merely small. Equatorward of 40° magnetic the field is the
        // quiet block alone, which here is identically zero.
        let field = polar_field(2, 60.0);
        let dipole = CentredDipole::at_year(2024.0);
        let (mut inside, mut outside) = (0, 0);
        for k in 0..72 {
            for i in 0..37 {
                let (lat, lon) = (
                    (f64::from(i) * 5.0 - 90.0) * core::f64::consts::PI / 180.0,
                    f64::from(k) * 5.0 * core::f64::consts::PI / 180.0,
                );
                let [east, north] = field.velocity(&at(lat, lon, 9.0));
                if libm::fabs(dipole.magnetic_latitude(lat, lon)) <= DISTURBANCE_ONSET {
                    assert_eq!(
                        (east, north),
                        (0.0, 0.0),
                        "wind below the window at {lat} {lon}"
                    );
                    outside += 1;
                } else if libm::fabs(dipole.magnetic_latitude(lat, lon)) >= DISTURBANCE_FULL {
                    inside += 1;
                }
            }
        }
        assert!(inside > 100 && outside > 100, "{inside} in, {outside} out");
    }

    #[test]
    fn a_high_latitude_field_round_trips_through_the_fit() {
        // The sharpest available check that `fit` builds its high-latitude design row with the same
        // magnetic rotation `velocity` evaluates it with: a sign error in either would leave a
        // residual on the very samples the coefficients were solved from, because `projection_rms`
        // goes back through `velocity`.
        //
        // The sampling is global, not polar. The two frames are only 9.2° apart, so a high-latitude
        // sampling alone would let the quiet block impersonate the disturbance; the low-latitude
        // rows, where the truth is zero, are what pin it down.
        let degree = 2;
        let truth = polar_field(degree, 60.0);
        let mut samples = Vec::new();
        for i in 0..12 {
            for k in 0..19 {
                for lon in 0..8 {
                    let point = at(
                        (f64::from(k) * 10.0 - 90.0) * core::f64::consts::PI / 180.0,
                        f64::from(lon) * 45.0 * core::f64::consts::PI / 180.0,
                        f64::from(i) * 2.0,
                    );
                    for &(dn, de) in &[(0.0, 1.0), (0.8, 0.6)] {
                        let [east, north] = truth.velocity(&point);
                        samples.push(CrosswindSample {
                            point,
                            direction_north: dn,
                            direction_east: de,
                            speed: de * east + dn * north,
                            weight: 1.0,
                        });
                    }
                }
            }
        }
        let fitted = HorizontalWindField::fit(
            &samples,
            &FitSpec {
                degree,
                ridge: 1e-6,
                disturbance_degree: degree,
                disturbance_ridge: 1e-6,
            },
        )
        .unwrap();
        let rms = fitted.projection_rms(&samples).unwrap();
        assert!(
            rms < 1e-3,
            "training residual {rms} m/s on an exactly-representable field"
        );
        for &(lat, lon) in &[(1.35, 0.4), (-1.4, 3.0), (1.2, 5.5), (-1.25, 1.1)] {
            let point = at(lat, lon, 4.5);
            let want = truth.velocity(&point);
            let got = fitted.velocity(&point);
            assert!(
                (got[0] - want[0]).abs() < 1e-2 && (got[1] - want[1]).abs() < 1e-2,
                "at ({lat}, {lon}): {got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn the_force_adapter_turns_the_field_into_an_inertial_vector() {
        // A wind of `u` m/s east at a point on the equator at longitude 0 must come out along −y in
        // an inertial frame whose x axis passes through that point... which is the geometry, not the
        // field: `ê = ẑ × r̂` at `r = (R, 0, 0)` is `+ŷ`. So the test is that the adapter produces
        // exactly `u·ê + v·n̂` and not some other pairing of the two amplitudes.
        let degree = 2;
        let field = zonal_field(degree, 40.0);
        let r = [6.8e6, 0.0, 0.0];
        let wind = EmpiricalWind {
            field: field.clone(),
            year: 2024.0,
            doy0: 80,
            sec0: 0.0,
            f107a: REFERENCE_F107,
            ap: REFERENCE_AP,
            geodetic: |_t: f64, _r: [f64; 3]| (400.0, 0.0, 0.0),
        };
        let got = wind.velocity::<f64>(0.0, &r);
        let point = at(0.0, 0.0, 0.0);
        let [east, north] = field.velocity(&point);
        // ê = +ŷ, n̂ = r̂ × ê = x̂ × ŷ = +ẑ.
        assert!((got[0] - 0.0).abs() < 1e-9, "{got:?}");
        assert!((got[1] - east).abs() < 1e-9, "{got:?} vs east {east}");
        assert!((got[2] - north).abs() < 1e-9, "{got:?} vs north {north}");
    }

    #[test]
    fn the_adapter_is_the_field_resolved_on_the_local_horizontal_frame() {
        // The test above sits at `r = (R, 0, 0)` with `sec0 = t = 0` and longitude 0, where every
        // term of the time bookkeeping and two thirds of the frame construction are multiplied by a
        // zero — an arithmetic slip in either is invisible there. These epochs and positions make
        // each of them carry a value: a clock that crosses day boundaries, a longitude that pushes
        // local solar time off both ends of `[0, 24)`, and positions where the operands of the
        // degeneracy guard differ in sign from the ones the frame is actually built from.
        let field = mixed_field(2);
        let (year, doy0, f107a, ap) = (2024.0, 80, 175.0, 40.0);
        // `(sec0, t, day_of_year, sec)` — the day of year the clock lands on and the seconds into
        // that day are written out rather than recomputed from `sec0 + t`, so the mutation under
        // test cannot move both sides of the comparison at once.
        let epochs = [
            (80_000.0, 12_345.0, 81.0, 5_945.0),
            (31_200.0, 300_000.0, 83.0, 72_000.0),
        ];
        let geodetics: [(f64, f64, f64); 2] = [(512.5, 23.75, -47.5), (250.0, -61.0, 170.0)];
        let positions = [
            [4.1e6, -3.2e6, 2.7e6],
            [3.5e6, 3.5e6, -4.0e6], // |x| = |y|, so x² − y² vanishes where x² + y² does not
            [0.0, -5.1e6, 3.3e6],   // x = 0, so x/x is NaN where x·x is not
            [-5.0e6, 100.0, 2.0e6], // 2x + y² < 0 where x² + y² > 0
            [100.0, -5.0e6, 2.0e6], // x² + 2y < 0, the same trap on the other operand
        ];
        for (&(sec0, t, doy, sec), &(alt_km, lat_deg, lon_deg)) in epochs.iter().zip(&geodetics) {
            let lst_raw = sec / 3_600.0 + lon_deg / 15.0;
            let point = WindPoint {
                latitude: lat_deg.to_radians(),
                longitude: lon_deg.to_radians(),
                local_solar_time: lst_raw - 24.0 * libm::floor(lst_raw / 24.0),
                altitude: alt_km * 1_000.0,
                year,
                day_of_year: doy,
                f107a,
                ap,
            };
            let [east, north] = field.velocity(&point);
            assert!(
                east.abs() > 1.0 && north.abs() > 1.0,
                "both components must be excited: east {east}, north {north}"
            );
            for r in positions {
                let wind = EmpiricalWind {
                    field: field.clone(),
                    year,
                    doy0,
                    sec0,
                    f107a,
                    ap,
                    geodetic: move |_t: f64, _r: [f64; 3]| (alt_km, lat_deg, lon_deg),
                };
                let got = wind.velocity::<f64>(t, &r);
                let rn = libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                let hn = libm::sqrt(r[0] * r[0] + r[1] * r[1]);
                let e = [-r[1] / hn, r[0] / hn, 0.0];
                let up = [r[0] / rn, r[1] / rn, r[2] / rn];
                let n = [
                    up[1] * e[2] - up[2] * e[1],
                    up[2] * e[0] - up[0] * e[2],
                    up[0] * e[1] - up[1] * e[0],
                ];
                for i in 0..3 {
                    let want = e[i] * east + n[i] * north;
                    assert!(
                        (got[i] - want).abs() < 1e-9,
                        "component {i} at r = {r:?}, t = {t}: {} vs {want}",
                        got[i]
                    );
                }
            }
        }
    }

    #[test]
    fn the_local_time_reduction_lands_in_the_day_it_claims() {
        // Every way of getting the number of days wrong still produces a valid *angle*, because the
        // expansion reads local solar time modulo a turn — which is why this checks the range and
        // the whole-day offset directly rather than through a wind.
        for &h in &[
            -1.515_277_777_777_777_7,
            0.0,
            23.999,
            24.0,
            31.333_333_333_333_332,
            -0.000_1,
            1e5,
            -1e5,
        ] {
            let r = hours_into_day(h);
            assert!((0.0..24.0).contains(&r), "{h} h reduced to {r}");
            let days = (r - h) / 24.0;
            assert!(
                libm::fabs(days - libm::round(days)) < 1e-12,
                "{h} h moved by {days} days, which is not a whole number of them"
            );
        }
    }

    #[test]
    fn a_field_of_zeros_is_no_wind_at_all_rather_than_a_nan() {
        // The horizontal amplitudes multiply the frame vectors, and the vertical component of east
        // is an exact zero — so a `*` that became a `/` there is invisible until an amplitude is
        // zero too, and a field that has not been fitted yet is exactly that. What it produces
        // would go straight into the drag acceleration.
        let j = angular_terms(2);
        let field =
            HorizontalWindField::from_coefficients(2, vec![0.0; DRIVERS * j], 0, Vec::new())
                .unwrap();
        let wind = EmpiricalWind {
            field,
            year: 2024.0,
            doy0: 80,
            sec0: 31_200.0,
            f107a: 175.0,
            ap: 40.0,
            geodetic: |_t: f64, _r: [f64; 3]| (450.0, 12.5, 88.0),
        };
        assert_eq!(
            wind.velocity::<f64>(9_876.0, &[4.1e6, -3.2e6, 2.7e6]),
            [0.0; 3]
        );
    }

    #[test]
    fn the_adapter_reports_no_wind_on_the_polar_axis() {
        // `ê = ẑ × r̂` does not exist there. Returning zero is a choice; dividing by a zero norm is
        // a NaN that would propagate into every subsequent state of the trajectory.
        let wind = EmpiricalWind {
            field: zonal_field(2, 40.0),
            year: 2024.0,
            doy0: 80,
            sec0: 0.0,
            f107a: REFERENCE_F107,
            ap: REFERENCE_AP,
            geodetic: |_t: f64, _r: [f64; 3]| (400.0, 90.0, 0.0),
        };
        assert_eq!(wind.velocity::<f64>(0.0, &[0.0, 0.0, 6.8e6]), [0.0; 3]);
    }
}
