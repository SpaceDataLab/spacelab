//! The JB2008 evaluation: indices and geometry in, density and temperatures out.
//!
//! Gate: `derivations/jacchia_bowman_full_model_terms.mac`, term by term, in the `(G2)` tests of
//! `tests/jacchia_bowman_g2_model.rs`.

use super::calendar::year_and_day_of_year;
use super::constants::Z0;
use super::profile::Profile;
use super::static_model::{NumberDensities, static_state};
use super::variations::{
    SolarIndices, delta_tc_b06, diurnal_ratio, high_altitude_factor, local_solar_time_hours,
    seasonal_latitudinal_log10, semiannual_log10, tc, wrap_pi,
};

/// Model input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Input {
    /// Modified Julian date, UT (MJD = JD - 2400000.5, SR332 p. 44).
    pub mjd: f64,
    /// Right ascension of the sun, rad. With `sat_ra` it gives the hour angle of the sun at
    /// the point, H = sat_ra - sun_ra (SR313 p. 18, eq. (16)).
    pub sun_ra: f64,
    /// Declination of the sun, rad (SR313 p. 18, eq. (15)).
    pub sun_dec: f64,
    /// Right ascension of the point (satellite), rad, same frame as `sun_ra`.
    pub sat_ra: f64,
    /// Geocentric latitude of the point, rad. Used as the latitude of SR313 eqs. (15) and
    /// (24) and of B06 eqs. (7)-(8); the papers say "latitude" / "geographic latitude".
    pub sat_lat: f64,
    /// Height above the reference surface, km; >= 90 km (SR313 p. 9).
    pub height_km: f64,
    /// Solar indices, sfu, already lagged by the caller (B08 p. 5: F10 and S10 1 day, M10
    /// 2 days, Y10 5 days); bars are 81-day centred averages (B08 p. 5).
    pub indices: SolarIndices,
    /// Storm-time exospheric temperature change dTc, K, an input as in B08 (p. 13-15, 17).
    pub dtc_storm: f64,
}

/// Model output.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Output {
    /// Total mass density, kg m^-3.
    pub density: f64,
    /// Exospheric temperature T_inf, K.
    pub t_inf: f64,
    /// Local temperature T(z) of the profile at the height, K.
    pub temperature: f64,
    /// Number densities of the static model at T_inf, m^-3. The semiannual, seasonal-
    /// latitudinal and high-altitude terms correct the mass density only.
    pub number_densities: NumberDensities,
    /// Intermediate terms.
    pub terms: Terms,
}

/// Intermediate terms of one evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Terms {
    /// Tc of B08 eq. (2), K.
    pub tc: f64,
    /// B06 Delta Tc, K (0 outside 200-700 km).
    pub delta_tc_b06: f64,
    /// Diurnal ratio T_l / Tc of SR313 eq. (17).
    pub diurnal_ratio: f64,
    /// Local solar time, h.
    pub local_solar_time: f64,
    /// Day of year t (1.0 at 0h UT January 1).
    pub day_of_year: f64,
    /// Semiannual Delta log10 rho, B08 eq. (3).
    pub semiannual_log10: f64,
    /// Seasonal-latitudinal Delta log10 rho.
    pub seasonal_latitudinal_log10: f64,
    /// High-altitude density factor, B06 eqs. (10)-(11).
    pub high_altitude_factor: f64,
    /// Static-model mass density at T_inf before the corrections, kg m^-3.
    pub static_density: f64,
}

/// Why an input cannot be evaluated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    /// An input field is NaN or infinite.
    NonFiniteInput,
    /// The height is below the lower boundary z_0 = 90 km of the model.
    BelowLowerBoundary {
        /// The height given, km.
        height_km: f64,
    },
    /// T_inf does not give T_0 < T_x < T_inf, so eqs. (11) and (13) define no profile.
    NoTemperatureProfile {
        /// The exospheric temperature reached, K.
        t_inf: f64,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NonFiniteInput => write!(f, "an input is not finite"),
            Error::BelowLowerBoundary { height_km } => {
                write!(f, "height {height_km} km is below the 90 km lower boundary")
            }
            Error::NoTemperatureProfile { t_inf } => {
                write!(
                    f,
                    "exospheric temperature {t_inf} K gives no temperature profile"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

fn all_finite(i: &Input) -> bool {
    let ix = &i.indices;
    [
        i.mjd,
        i.sun_ra,
        i.sun_dec,
        i.sat_ra,
        i.sat_lat,
        i.height_km,
        i.dtc_storm,
        ix.f10,
        ix.f10_bar,
        ix.s10,
        ix.s10_bar,
        ix.m10,
        ix.m10_bar,
        ix.y10,
        ix.y10_bar,
    ]
    .iter()
    .all(|v| v.is_finite())
}

/// Evaluates the model.
///
/// Modelling decisions (REPORT.md, "Decisions"): B06 Delta Tc is added to Tc before the
/// diurnal factor (B06 p. 784); the storm dTc is added to T_l after it, in the place of the
/// J70 geomagnetic Delta T_g (SR313 p. 21, p. 36; B08 p. 15).
pub fn evaluate(input: &Input) -> Result<Output, Error> {
    if !all_finite(input) {
        return Err(Error::NonFiniteInput);
    }
    let z = input.height_km;
    if z < Z0 {
        return Err(Error::BelowLowerBoundary { height_km: z });
    }
    let ix = &input.indices;
    let hour_angle = wrap_pi(input.sat_ra - input.sun_ra);
    let lst = local_solar_time_hours(hour_angle);

    let tc_b08 = tc(ix);
    let dtc_b06 = delta_tc_b06(ix.f10, lst, input.sat_lat, z);
    let ratio = diurnal_ratio(input.sat_lat, input.sun_dec, hour_angle);
    let t_inf = (tc_b08 + dtc_b06) * ratio + input.dtc_storm;

    let profile = Profile::new(t_inf).ok_or(Error::NoTemperatureProfile { t_inf })?;
    let state = static_state(&profile, z);

    let (_, doy) = year_and_day_of_year(input.mjd);
    let sa = semiannual_log10(ix, z, doy);
    let sl = seasonal_latitudinal_log10(z, input.sat_lat, input.mjd);
    let fac = high_altitude_factor(ix.f10_bar, z);
    let density = state.density * 10f64.powf(sa + sl) * fac;

    Ok(Output {
        density,
        t_inf,
        temperature: state.temperature,
        number_densities: state.n,
        terms: Terms {
            tc: tc_b08,
            delta_tc_b06: dtc_b06,
            diurnal_ratio: ratio,
            local_solar_time: lst,
            day_of_year: doy,
            semiannual_log10: sa,
            seasonal_latitudinal_log10: sl,
            high_altitude_factor: fac,
            static_density: state.density,
        },
    })
}
