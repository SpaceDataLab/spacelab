//! Modified Harris-Priester as a [`crate::forces::Atmosphere`] for the [`crate::forces::Drag`] force.
//!
//! The [`super`] model is f64 and needs the Sun's direction and the satellite's position in a common
//! Earth-centred frame — neither of which the bare inertial position carries on its own. This wrapper
//! supplies them through injected closures, so this crate stays independent of the frames/ephemeris
//! crates: `geometry` maps `(t, inertial position in metres)` to the position in the Earth frame, and
//! `sun` maps `t` to the Sun's position in that same frame. Because the density depends only on `|r|`
//! (altitude) and the direction of `r` relative to the diurnal bulge — both invariant under a shared
//! rotation about `+z` — any consistent Earth-centred frame the caller supplies works.
//!
//! The [`crate::forces::Atmosphere`] density is written over [`Scalar`] so its gradient can enter the
//! state-transition matrix, but the empirical model is defined only in f64. The value is therefore
//! evaluated in f64 and lifted to `S` through a **local exponential** `ρ = ρ₀·exp(−(|r|−|r₀|)/H)`,
//! with the scale height `H` read from a second model evaluation 10 km higher (radially, so the
//! diurnal-bulge angle is unchanged) — identical to the JB2008 and NRLMSISE-00 wrappers, giving a
//! uniform drag-path treatment of all three empirical models. The lift captures the dominant radial
//! density gradient that decay depends on; the far smaller horizontal (bulge) gradient is not
//! propagated into the STM, a deliberate omission on a mean-activity model good only to a factor.

use spacelab_autodiff::Scalar;

use crate::forces::Atmosphere;

use super::{HarrisPriester, harris_priester};

/// Modified Harris-Priester wrapped as a drag [`Atmosphere`]. See the module docs for the injected
/// geometry/Sun closures and the local-exponential lift.
pub struct HarrisPriesterAtmosphere<G, S> {
    /// The model configuration (ellipsoid + cosine exponent).
    pub model: HarrisPriester,
    /// Maps `(t, inertial position in metres)` to the satellite position in the Earth frame (metres);
    /// the caller supplies the Earth-rotation model.
    pub geometry: G,
    /// Maps `t` to the Sun's position in the same Earth frame (metres); the caller supplies the
    /// ephemeris.
    pub sun: S,
}

impl<G, S> HarrisPriesterAtmosphere<G, S>
where
    G: Fn(f64, [f64; 3]) -> [f64; 3],
    S: Fn(f64) -> [f64; 3],
{
    /// Total mass density (kg/m³) at the point, and 10 km higher radially (for the local scale height).
    fn density_pair(&self, t: f64, rf: [f64; 3]) -> (f64, f64) {
        let pos = (self.geometry)(t, rf);
        let sun = (self.sun)(t);
        let p_mag = libm::sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
        let up = (p_mag + 10_000.0) / p_mag; // scale radially by +10 km, same bulge angle
        let pos_up = [pos[0] * up, pos[1] * up, pos[2] * up];
        (
            harris_priester(sun, pos, &self.model),
            harris_priester(sun, pos_up, &self.model),
        )
    }
}

impl<G, S> Atmosphere for HarrisPriesterAtmosphere<G, S>
where
    G: Fn(f64, [f64; 3]) -> [f64; 3],
    S: Fn(f64) -> [f64; 3],
{
    fn density<Sc: Scalar>(&self, t: f64, r: &[Sc; 3]) -> Sc {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let (rho0, rho_up) = self.density_pair(t, rf);
        // Local scale height H = −Δh / ln(ρ_up/ρ₀), Δh = 10 km; fall back to 50 km if degenerate
        // (e.g. ρ₀ = 0 above the table ceiling, or ρ_up = ρ₀ inside the 100–120 km band).
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
