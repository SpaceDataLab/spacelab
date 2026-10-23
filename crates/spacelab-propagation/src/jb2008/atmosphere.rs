//! JB2008 as a [`crate::forces::Atmosphere`] for the [`crate::forces::Drag`] force.
//!
//! The [`super`] model is f64 and needs the epoch (MJD), the Sun's right ascension/declination, and
//! the satellite's right ascension, geocentric latitude and altitude — none of which the bare
//! inertial position carries. This wrapper supplies them: the epoch (MJD at `t = 0`) and the constant
//! space-weather drivers ([`Jb2008Input`]) are held as fields; the two frame-dependent conversions are
//! injected as closures, so this crate stays independent of the frames/ephemeris crates. `geometry`
//! maps `(t, inertial position in metres)` to `(altitude_km, geocentric_latitude_rad,
//! right_ascension_rad)`; `sun` maps `t` to `(sun_ra_rad, sun_dec_rad)`. Only the hour angle
//! `sat_ra − sun_ra` and the Sun declination enter the model, so any consistent frame works.
//!
//! The [`crate::forces::Atmosphere`] density is written over [`Scalar`] so its gradient can enter the
//! state-transition matrix, but the empirical model is defined only in f64. The value is therefore
//! evaluated in f64 and lifted to `S` through a **local exponential** `ρ = ρ₀·exp(−(|r|−|r₀|)/H)`,
//! with the scale height `H` read from a second model evaluation 10 km higher — the same device
//! operational orbit determination uses for tabulated density, identical to the NRLMSISE-00 wrapper.

use spacelab_autodiff::Scalar;

use crate::forces::Atmosphere;

use super::{Jb2008Input, jb2008};

/// JB2008 wrapped as a drag [`Atmosphere`]. See the module docs for the injected geometry/Sun
/// closures and the local-exponential lift.
pub struct Jb2008Atmosphere<G, S> {
    /// Modified Julian Date (`JD − 2400000.5`) at propagation time `t = 0`.
    pub mjd0: f64,
    /// Constant space-weather drivers for the propagation span.
    pub input: Jb2008Input,
    /// Maps `(t, inertial position in metres)` to `(altitude_km, geocentric_latitude_rad,
    /// right_ascension_rad)`; the caller supplies the Earth-rotation and ellipsoid model.
    pub geometry: G,
    /// Maps `t` to the Sun's `(right_ascension_rad, declination_rad)`; the caller supplies the
    /// ephemeris.
    pub sun: S,
}

impl<G, S> Jb2008Atmosphere<G, S>
where
    G: Fn(f64, [f64; 3]) -> (f64, f64, f64),
    S: Fn(f64) -> (f64, f64),
{
    /// Total mass density (kg/m³) at the point, and 10 km higher (for the local scale height).
    fn density_pair(&self, t: f64, rf: [f64; 3]) -> (f64, f64) {
        let (alt, lat, ra) = (self.geometry)(t, rf);
        let (sun_ra, sun_dec) = (self.sun)(t);
        let mjd = self.mjd0 + t / 86_400.0;
        (
            jb2008(mjd, sun_ra, sun_dec, ra, lat, alt, &self.input).rho,
            jb2008(mjd, sun_ra, sun_dec, ra, lat, alt + 10.0, &self.input).rho,
        )
    }
}

impl<G, S> Atmosphere for Jb2008Atmosphere<G, S>
where
    G: Fn(f64, [f64; 3]) -> (f64, f64, f64),
    S: Fn(f64) -> (f64, f64),
{
    fn density<Sc: Scalar>(&self, t: f64, r: &[Sc; 3]) -> Sc {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let (rho0, rho_up) = self.density_pair(t, rf);
        // Local scale height H = −Δh / ln(ρ_up/ρ₀), Δh = 10 km; fall back to 50 km if degenerate.
        let h_scale = if rho0 > 0.0 && rho_up > 0.0 {
            let hh = -10_000.0 / libm::log(rho_up / rho0);
            if hh.is_finite() && hh > 1_000.0 {
                hh
            } else {
                50_000.0
            }
        } else {
            50_000.0
        };
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let r0 = Sc::from_f64(libm::sqrt(rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2]));
        Sc::from_f64(rho0) * ((r0 - r_mag) * Sc::from_f64(1.0 / h_scale)).exp()
    }
}
