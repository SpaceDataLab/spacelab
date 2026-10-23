//! Communications antenna: access is where the link budget closes.
//!
//! A ground-station or relay link is available when the received carrier-to-noise density exceeds
//! what the waveform needs — `C/N₀ ≥ C/N₀_req`, i.e. link margin `≥ 0`. As the satellite rises the
//! slant range shortens (free-space path loss drops) and the atmospheric slant path thins, so the
//! margin is **monotone increasing in elevation**. That monotonicity is the whole design: the
//! budget has a single break-even elevation `ε_link` where the margin crosses zero, and access is
//! exactly `ε ≥ ε_link`. So a [`CommsAntenna`] evaluates the budget once, at construction, to find
//! `ε_link`, and *is* an [`ElevationMask`] at it — inheriting the mask's exact, square-root-free
//! `g`, `covered` and derived rate bound rather than carrying a second, harder-to-bound switching
//! function.
//!
//! The slant range uses a reference altitude (a spherical `d(ε)`); for a near-circular orbit that
//! is the operating altitude. Eccentric orbits vary it, which shifts `ε_link` by the margin's slope
//! there — a documented approximation, not a hidden one.

use crate::{AccessGeometry, Constraint, ElevationMask, OrbitBound, SensorError};

/// Speed of light, m/s.
const C_LIGHT: f64 = 299_792_458.0;
/// `−10·log₁₀(k)` with `k = 1.380649e−23 J/K`: the Boltzmann term of the C/N₀ equation, dBW/(K·Hz).
const BOLTZMANN_DB: f64 = 228.6;
/// WGS-84 equatorial radius, m — the reference sphere for the slant-range model.
const EARTH_RADIUS: f64 = 6_378_137.0;

/// A one-way communications link budget. All powers in dB; the constraint it produces is an
/// elevation mask at the elevation where the margin reaches zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommsBudget {
    /// Transmitter effective isotropic radiated power, dBW.
    pub eirp_dbw: f64,
    /// Receiver figure of merit `G/T`, dB/K.
    pub g_over_t_db: f64,
    /// Carrier frequency, Hz.
    pub frequency_hz: f64,
    /// Required carrier-to-noise density for the waveform, dBHz.
    pub required_cn0_dbhz: f64,
    /// Atmospheric loss at zenith, dB; scaled by `1/sin ε` along the slant path.
    pub zenith_atmo_db: f64,
    /// Reference altitude for the slant-range model, m.
    pub reference_altitude_m: f64,
    /// Hard elevation floor, rad: no link below it whatever the budget says (horizon obstruction).
    pub floor_elevation_rad: f64,
}

impl CommsBudget {
    /// Slant range to the satellite at elevation `eps`, m, on the reference sphere:
    /// `d = √((Rₑ+h)² − (Rₑ cos ε)²) − Rₑ sin ε`.
    #[must_use]
    pub fn slant_range(&self, eps: f64) -> f64 {
        let rs = EARTH_RADIUS + self.reference_altitude_m;
        let horizontal = EARTH_RADIUS * libm::cos(eps);
        libm::sqrt(rs * rs - horizontal * horizontal) - EARTH_RADIUS * libm::sin(eps)
    }

    /// Received `C/N₀` at elevation `eps`, dBHz:
    /// `EIRP + G/T − FSPL(d, f) − L_atm/sin ε + 228.6`.
    #[must_use]
    pub fn cn0_dbhz(&self, eps: f64) -> f64 {
        let d = self.slant_range(eps);
        let fspl =
            20.0 * libm::log10(4.0 * core::f64::consts::PI * d * self.frequency_hz / C_LIGHT);
        let atmo = self.zenith_atmo_db / libm::sin(eps);
        self.eirp_dbw + self.g_over_t_db - fspl - atmo + BOLTZMANN_DB
    }

    /// Link margin at elevation `eps`, dB. Monotone increasing in `eps` on `(0, π/2]`.
    #[must_use]
    pub fn margin_db(&self, eps: f64) -> f64 {
        self.cn0_dbhz(eps) - self.required_cn0_dbhz
    }

    /// The lowest elevation at which the margin is non-negative, in `[floor, π/2]`, by bisection
    /// on the monotone margin. `None` if the link does not close even at zenith.
    #[must_use]
    pub fn break_even_elevation(&self) -> Option<f64> {
        let hi = core::f64::consts::FRAC_PI_2;
        let lo = self.floor_elevation_rad.max(1e-4); // sin ε must stay away from 0
        if self.margin_db(hi) < 0.0 {
            return None; // never closes, even overhead
        }
        if self.margin_db(lo) >= 0.0 {
            return Some(lo); // closes already at the floor
        }
        let (mut a, mut b) = (lo, hi);
        for _ in 0..60 {
            let m = f64::midpoint(a, b);
            if self.margin_db(m) >= 0.0 {
                b = m;
            } else {
                a = m;
            }
        }
        Some(b)
    }
}

/// A communications-antenna access constraint: the elevation mask the [`CommsBudget`] implies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommsAntenna {
    mask: ElevationMask,
    break_even_rad: f64,
}

impl CommsAntenna {
    /// Build from a link budget: solve for the break-even elevation and mask there.
    ///
    /// # Errors
    ///
    /// [`SensorError::LinkNeverCloses`] if the margin is negative even at zenith;
    /// [`SensorError::ElevationOutOfRange`] if the resulting elevation is not inside `(-π/2, π/2)`.
    pub fn from_budget(budget: &CommsBudget) -> Result<Self, SensorError> {
        let eps = budget
            .break_even_elevation()
            .ok_or(SensorError::LinkNeverCloses)?;
        Ok(Self {
            mask: ElevationMask::new(eps)?,
            break_even_rad: eps,
        })
    }

    /// The break-even elevation, rad — the mask this antenna imposes.
    #[must_use]
    pub const fn break_even_elevation(self) -> f64 {
        self.break_even_rad
    }
}

impl Constraint for CommsAntenna {
    fn g(&self, geom: &AccessGeometry) -> f64 {
        self.mask.g(geom)
    }

    fn covered(&self, geom: &AccessGeometry) -> bool {
        self.mask.covered(geom)
    }

    fn max_rate(&self, orbit: &OrbitBound, frame_rate: f64) -> f64 {
        self.mask.max_rate(orbit, frame_rate)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    // An X-band LEO downlink: high-gain satellite EIRP, a mid-size ground station.
    fn x_band() -> CommsBudget {
        CommsBudget {
            eirp_dbw: 18.0,
            g_over_t_db: 30.0,
            frequency_hz: 8.2e9,
            // High enough that the link fails near the horizon and closes only higher up, so the
            // break-even elevation is a real crossing rather than the floor.
            required_cn0_dbhz: 95.0,
            zenith_atmo_db: 0.5,
            reference_altitude_m: 700e3,
            floor_elevation_rad: 5f64.to_radians(),
        }
    }

    #[test]
    fn margin_increases_with_elevation() {
        let b = x_band();
        let mut prev = f64::NEG_INFINITY;
        for deg in [5.0_f64, 10.0, 20.0, 40.0, 70.0, 89.0] {
            let m = b.margin_db(deg.to_radians());
            assert!(m > prev, "margin not monotone at {deg}°: {m} <= {prev}");
            prev = m;
        }
    }

    #[test]
    fn break_even_is_where_the_margin_crosses_zero() {
        let b = x_band();
        let eps = b.break_even_elevation().unwrap();
        assert!(
            b.margin_db(eps).abs() < 1e-3,
            "margin at break-even is {}",
            b.margin_db(eps)
        );
        assert!(
            b.margin_db(eps + 1e-3) > 0.0 && b.margin_db(eps - 1e-3) < 0.0,
            "not a sign change"
        );
    }

    #[test]
    fn a_stronger_link_lowers_the_required_elevation() {
        let weak = x_band();
        let strong = CommsBudget {
            eirp_dbw: 28.0,
            ..weak
        };
        let (ew, es) = (
            CommsAntenna::from_budget(&weak)
                .unwrap()
                .break_even_elevation(),
            CommsAntenna::from_budget(&strong)
                .unwrap()
                .break_even_elevation(),
        );
        assert!(
            es < ew,
            "10 dB more EIRP should lower the mask: {es} vs {ew}"
        );
    }

    #[test]
    fn a_hopeless_link_is_rejected() {
        let b = CommsBudget {
            eirp_dbw: -50.0,
            ..x_band()
        };
        assert_eq!(
            CommsAntenna::from_budget(&b),
            Err(SensorError::LinkNeverCloses)
        );
    }
}
