//! Jacchia temperature profile, SR313 p. 9-10, eqs. (9)-(13).
//!
//! Gate: `derivations/jacchia_bowman_quartic.mac` and the profile `(G2)` tests in
//! `tests/jacchia_bowman_g1_g2_static.rs`.

use super::constants::{
    GX_FACTOR, PROFILE_B, PROFILE_N, QUARTIC_C3_OVER_G, QUARTIC_C4_OVER_G, T0, TX_A, TX_B, TX_C,
    TX_K, Z0, ZX,
};
use std::f64::consts::FRAC_2_PI;

/// Inflection-point temperature T_x(T_inf), K (SR313 p. 9, eq. (9)).
pub fn inflection_temperature(t_inf: f64) -> f64 {
    TX_A + TX_B * t_inf + TX_C * (TX_K * t_inf).exp()
}

/// The temperature profile T(z) for one exospheric temperature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Profile {
    /// Exospheric temperature T_inf, K.
    pub t_inf: f64,
    /// Inflection temperature T_x at z_x = 125 km, K.
    pub tx: f64,
    /// Gradient G_x = (dT/dz) at z_x, K/km.
    pub gx: f64,
    /// A = (2/pi)(T_inf - T_x), K (SR313 p. 10).
    a: f64,
}

impl Profile {
    /// Builds the profile. Returns `None` unless T_0 < T_x < T_inf, the range in which eqs. (11)
    /// and (13) define a monotonic profile (G_x > 0 and A > 0).
    pub fn new(t_inf: f64) -> Option<Self> {
        let tx = inflection_temperature(t_inf);
        if !(t_inf.is_finite() && tx > T0 && t_inf > tx) {
            return None;
        }
        let gx = GX_FACTOR * (tx - T0) / (ZX - Z0);
        Some(Self {
            t_inf,
            tx,
            gx,
            a: FRAC_2_PI * (t_inf - tx),
        })
    }

    /// Temperature T(z), K, for z >= z_0 (km).
    pub fn temperature(&self, z: f64) -> f64 {
        let d = z - ZX;
        if d <= 0.0 {
            // eq. (10) with c1 = G_x, c2 = 0, c3, c4 from eq. (11).
            let d2 = d * d;
            self.tx + self.gx * d * (1.0 + d2 * (QUARTIC_C3_OVER_G + QUARTIC_C4_OVER_G * d))
        } else {
            // eq. (13)
            self.tx + self.a * (self.gx / self.a * d * (1.0 + PROFILE_B * d.powf(PROFILE_N))).atan()
        }
    }

    /// Analytic dT/dz, K/km.
    pub fn gradient(&self, z: f64) -> f64 {
        let d = z - ZX;
        if d <= 0.0 {
            self.gx * (1.0 + d * d * (3.0 * QUARTIC_C3_OVER_G + 4.0 * QUARTIC_C4_OVER_G * d))
        } else {
            let dn = d.powf(PROFILE_N);
            let u = self.gx / self.a * d * (1.0 + PROFILE_B * dn);
            self.gx * (1.0 + PROFILE_B * (PROFILE_N + 1.0) * dn) / (1.0 + u * u)
        }
    }
}
