//! GLONASS broadcast ephemeris → PZ-90 position/velocity by numerical integration (GLONASS ICD 5.1,
//! Appendix A.3.1.2).
//!
//! Unlike the Keplerian systems ([`crate::gps`]), GLONASS broadcasts the satellite's Cartesian PZ-90
//! state — position, velocity and the *luni-solar acceleration* — at a reference epoch, and the user
//! integrates the equations of motion (central term + `J₂` + the Earth-rotation Coriolis/centrifugal
//! terms of the rotating PZ-90 frame + the broadcast luni-solar acceleration, held constant over the
//! ≤15-minute validity) with a 4th-order Runge–Kutta step. This module does exactly that.

// The single letters (a, b, c, r, x, y, z) are the ICD A.3.1.2 integration notation; the sub-step
// count is a small non-negative integer from a bounded interval.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// PZ-90.11 gravitational parameter `μ`, m³/s² (GLONASS ICD).
pub const MU: f64 = 3.986_004_418e14;

/// PZ-90 Earth equatorial radius `aₑ`, m (GLONASS ICD).
pub const EARTH_RADIUS: f64 = 6_378_136.0;

/// Second zonal harmonic `J₂ = −C̄₂₀·√5` in unnormalised form (GLONASS ICD), dimensionless.
pub const J2: f64 = 1.082_625_7e-3;

/// PZ-90 Earth-rotation rate `Ω̇ₑ`, rad/s (GLONASS ICD).
pub const EARTH_ROTATION_RATE: f64 = 7.292_115e-5;

/// Default RK4 sub-step, seconds — the ICD-recommended integration step.
const DEFAULT_STEP: f64 = 60.0;

/// A GLONASS broadcast-ephemeris record: the PZ-90 Cartesian state and luni-solar acceleration at the
/// reference epoch. Position is metres, velocity m/s, acceleration m/s² (the broadcast km/s² × 10³).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GlonassEphemeris {
    /// Satellite slot (PRN).
    pub slot: u8,
    /// Reference epoch, seconds of day (Moscow time) — the instant the broadcast state is valid.
    pub toe: f64,
    /// PZ-90 position `[x, y, z]`, m.
    pub position: [f64; 3],
    /// PZ-90 velocity `[vx, vy, vz]`, m/s.
    pub velocity: [f64; 3],
    /// PZ-90 luni-solar acceleration `[ax, ay, az]`, m/s².
    pub luni_solar: [f64; 3],
    /// Clock bias `−τₙ`, s.
    pub tau: f64,
    /// Relative frequency offset `+γₙ`, dimensionless.
    pub gamma: f64,
    /// Frequency channel number (`−7 … +6`).
    pub freq_channel: i8,
    /// SV health (0 = healthy).
    pub health: u8,
}

/// The PZ-90 equations of motion (ICD A.3.1.2): `d/dt [r, v] = [v, a(r, v)]`, with the central term,
/// `j2` oblateness, the rotating-frame Coriolis/centrifugal terms at rate `omega`, and the constant
/// luni-solar acceleration `acc`. Parameterised in `mu`/`j2`/`omega`/`acc` so the two-body reduction
/// (`j2 = omega = 0`, `acc = 0`) is testable against the analytic Kepler flow.
fn deriv(s: [f64; 6], mu: f64, j2: f64, omega: f64, acc: [f64; 3]) -> [f64; 6] {
    let (x, y, z) = (s[0], s[1], s[2]);
    let (vx, vy) = (s[3], s[4]);
    let r2 = x * x + y * y + z * z;
    let r = libm::sqrt(r2);
    let r3 = r2 * r;
    // a = 3/2 · J₂ · μ · aₑ² / r⁵ ; c = −μ/r³ − a(1 − 5z²/r²).
    let a = 1.5 * j2 * mu * EARTH_RADIUS * EARTH_RADIUS / (r2 * r3);
    let b = 5.0 * z * z / r2;
    let c = -mu / r3 - a * (1.0 - b);
    let om2 = omega * omega;
    [
        s[3],
        s[4],
        s[5],
        (c + om2) * x + 2.0 * omega * vy + acc[0],
        (c + om2) * y - 2.0 * omega * vx + acc[1],
        (c - 2.0 * a) * z + acc[2],
    ]
}

fn rk4_step(s: [f64; 6], h: f64, mu: f64, j2: f64, omega: f64, acc: [f64; 3]) -> [f64; 6] {
    let add = |a: [f64; 6], k: [f64; 6], f: f64| {
        let mut o = [0.0; 6];
        for i in 0..6 {
            o[i] = a[i] + k[i] * f;
        }
        o
    };
    let k1 = deriv(s, mu, j2, omega, acc);
    let k2 = deriv(add(s, k1, h / 2.0), mu, j2, omega, acc);
    let k3 = deriv(add(s, k2, h / 2.0), mu, j2, omega, acc);
    let k4 = deriv(add(s, k3, h), mu, j2, omega, acc);
    let mut o = [0.0; 6];
    for i in 0..6 {
        o[i] = s[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    o
}

/// Integrate the 6-state `s0` by `dt` seconds with RK4, in `⌈|dt|/max_step⌉` equal sub-steps.
fn integrate(
    s0: [f64; 6],
    dt: f64,
    mu: f64,
    j2: f64,
    omega: f64,
    acc: [f64; 3],
    max_step: f64,
) -> [f64; 6] {
    if dt == 0.0 {
        return s0;
    }
    let n = libm::ceil(libm::fabs(dt) / max_step).max(1.0);
    let h = dt / n;
    let steps = n as usize;
    let mut s = s0;
    for _ in 0..steps {
        s = rk4_step(s, h, mu, j2, omega, acc);
    }
    s
}

/// GLONASS FDMA nominal **L1** carrier frequency for channel `k`, Hz: `(1602.0 + k·0.5625)` MHz
/// (GLONASS ICD). Unlike the CDMA constellations, GLONASS satellites transmit on per-satellite
/// frequencies set by their channel number — so the ionospheric delay is per satellite.
#[must_use]
pub fn glonass_l1_frequency(channel: i8) -> f64 {
    (1602.0 + f64::from(channel) * 0.5625) * 1.0e6
}

/// GLONASS FDMA nominal **L2** carrier frequency for channel `k`, Hz: `(1246.0 + k·0.4375)` MHz
/// (GLONASS ICD).
#[must_use]
pub fn glonass_l2_frequency(channel: i8) -> f64 {
    (1246.0 + f64::from(channel) * 0.4375) * 1.0e6
}

impl GlonassEphemeris {
    /// This satellite's FDMA **L1** carrier frequency (Hz), from its [`Self::freq_channel`].
    #[must_use]
    pub fn l1_frequency(&self) -> f64 {
        glonass_l1_frequency(self.freq_channel)
    }

    /// This satellite's FDMA **L2** carrier frequency (Hz), from its [`Self::freq_channel`].
    #[must_use]
    pub fn l2_frequency(&self) -> f64 {
        glonass_l2_frequency(self.freq_channel)
    }

    /// PZ-90 position and velocity (`[m]`, `[m/s]`) at `dt` seconds from [`Self::toe`], by integrating
    /// the ICD equations of motion with the default 60-second RK4 step.
    #[must_use]
    pub fn position_velocity_pz90(&self, dt: f64) -> ([f64; 3], [f64; 3]) {
        self.position_velocity_pz90_with_step(dt, DEFAULT_STEP)
    }

    /// [`Self::position_velocity_pz90`] with an explicit RK4 sub-step (for convergence studies).
    #[must_use]
    pub fn position_velocity_pz90_with_step(&self, dt: f64, max_step: f64) -> ([f64; 3], [f64; 3]) {
        let s0 = [
            self.position[0],
            self.position[1],
            self.position[2],
            self.velocity[0],
            self.velocity[1],
            self.velocity[2],
        ];
        let s = integrate(
            s0,
            dt,
            MU,
            J2,
            EARTH_ROTATION_RATE,
            self.luni_solar,
            max_step,
        );
        ([s[0], s[1], s[2]], [s[3], s[4], s[5]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacelab_core::Vec3;
    use spacelab_elements::KeplerianElements;

    /// A representative GLONASS state: near-circular at the ~25 510 km orbit radius, in the xy-plane.
    fn sample() -> GlonassEphemeris {
        let a = 25_510_000.0;
        let vc = libm::sqrt(MU / a);
        GlonassEphemeris {
            slot: 1,
            toe: 43_200.0,
            position: [a, 0.0, 0.0],
            velocity: [0.0, vc * 0.9, vc * 0.4], // inclined, still near-circular
            luni_solar: [1.0e-6, -2.0e-6, 3.0e-6],
            tau: 1.0e-4,
            gamma: 0.0,
            freq_channel: 0,
            health: 0,
        }
    }

    #[test]
    fn glonass_orbit_stays_near_the_medium_earth_orbit_radius() {
        let eph = sample();
        for &dt in &[0.0, 300.0, 900.0, -600.0] {
            let (p, _) = eph.position_velocity_pz90(dt);
            let r = libm::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            // GLONASS sits near 25 510 km; the small perturbations keep it within a few percent.
            assert!((24.5e6..26.5e6).contains(&r), "dt={dt}: r={r}");
        }
    }

    #[test]
    fn reduced_two_body_integration_matches_the_kepler_flow() {
        // Turn off J₂, the frame rotation and the luni-solar term: the RK4 integration must then
        // reproduce the analytic two-body flow (spacelab-elements), a fully independent oracle.
        let a = 25_510_000.0;
        let vc = libm::sqrt(MU / a);
        let s0 = [a, 0.0, 0.0, 0.0, vc * 0.9, vc * 0.4];
        let dt = 1200.0;
        let s = integrate(s0, dt, MU, 0.0, 0.0, [0.0; 3], 30.0);

        let kep = KeplerianElements::from_state(
            Vec3::new(s0[0], s0[1], s0[2]),
            Vec3::new(s0[3], s0[4], s0[5]),
            MU,
        );
        let (r_oracle, _) = kep.propagate(dt).to_state();
        let diff = (Vec3::new(s[0], s[1], s[2]) - r_oracle).norm();
        // RK4 at 30-s steps over 20 min: sub-metre against the exact Kepler position.
        assert!(diff < 1.0, "two-body RK4 vs Kepler: {diff} m");
    }

    /// The full ICD right-hand side, against the published equations evaluated in Maxima.
    ///
    /// The Kepler test above is the only external oracle `deriv` has, and it reaches it through
    /// `j2 = 0, omega = 0, acc = 0` — so the oblateness term, both rotating-frame terms and the
    /// broadcast luni-solar acceleration are switched off in the one place they could be checked.
    /// The reference here is `tools/gen_glonass_oracle.mac` at 60 digits, which transcribes ICD
    /// A.3.1.2 in the ICD's own grouping (an explicit `5z²/r² − 3` on the `z` row) rather than the
    /// `c − 2a` factoring this module uses, so the two share no arithmetic. The state has no zero
    /// or repeated coordinate and the three luni-solar components differ in sign and size, so no
    /// term is annihilated. Tolerance is set by the rounding of the decimal constants into doubles
    /// (~1e-16 relative), not by the formula.
    #[test]
    // The first three rows are copied out of the state, so equality there is exact by construction.
    #[allow(clippy::float_cmp)]
    fn the_icd_right_hand_side_matches_the_published_equations() {
        const STATE: [f64; 6] = [1.5e7, -8.0e6, 1.9e7, 1500.0, 3000.0, -900.0];
        const ACC: [f64; 3] = [1.0e-6, -2.0e-6, 3.0e-6];
        const WANT: [f64; 3] = [
            1.565_617_251_861_625_7e-1,
            -6.891_682_343_262_004e-2,
            -4.570_127_719_221_374_6e-1,
        ];
        let d = deriv(STATE, MU, J2, EARTH_ROTATION_RATE, ACC);
        assert_eq!([d[0], d[1], d[2]], [STATE[3], STATE[4], STATE[5]]);
        for (k, (&got, &want)) in d[3..].iter().zip(WANT.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-13 * want.abs(),
                "row {k}: {got:.17e} vs Maxima {want:.17e}",
            );
        }
    }

    /// The sub-step count is `⌈|dt|/max_step⌉` — a division, and a bound the caller sets.
    ///
    /// [`rk4_converges_at_fourth_order_in_the_step`] compares three integrations to a fourth, so a
    /// change that makes *every* step finer leaves the ratio to floating-point noise and the test
    /// passes or fails at random. Anchoring the coarse error to an absolute band instead pins the
    /// step count from below: the reference is the analytic Kepler flow, and a 300-second step over
    /// twenty minutes must be wrong by a specific, measurable amount.
    #[test]
    fn the_sub_step_count_is_the_interval_divided_by_the_bound() {
        let a = 25_510_000.0;
        let vc = libm::sqrt(MU / a);
        let s0 = [a, 0.0, 0.0, 0.0, vc * 0.9, vc * 0.4];
        let dt = 1200.0;
        let kep = KeplerianElements::from_state(
            Vec3::new(s0[0], s0[1], s0[2]),
            Vec3::new(s0[3], s0[4], s0[5]),
            MU,
        );
        let (r_oracle, _) = kep.propagate(dt).to_state();
        let err = |max_step: f64| {
            let s = integrate(s0, dt, MU, 0.0, 0.0, [0.0; 3], max_step);
            (Vec3::new(s[0], s[1], s[2]) - r_oracle).norm()
        };
        let (e300, e150) = (err(300.0), err(150.0));
        // Measured: 4 × 300 s misses by 0.174317 m, 8 × 150 s by 0.010837 m — a ratio of 16.09,
        // the fourth order. The lower bound is the load-bearing half: it fails for any step count
        // materially larger than the four the division asks for.
        assert!(
            (0.05..0.5).contains(&e300),
            "4 × 300 s should miss Kepler by ~0.17 m, not {e300} m",
        );
        assert!(
            (12.0..20.0).contains(&(e300 / e150)),
            "ratio {}",
            e300 / e150
        );
    }

    #[test]
    fn rk4_converges_at_fourth_order_in_the_step() {
        // Halving the step should cut the error by ~16× (order 4). Compare coarse/fine to a very fine
        // reference over a fixed interval with the full dynamics.
        let eph = sample();
        let dt = 900.0;
        let reference = eph.position_velocity_pz90_with_step(dt, 5.0).0;
        let coarse = eph.position_velocity_pz90_with_step(dt, 120.0).0;
        let fine = eph.position_velocity_pz90_with_step(dt, 60.0).0;
        let err = |a: [f64; 3]| {
            libm::sqrt(
                (a[0] - reference[0]).powi(2)
                    + (a[1] - reference[1]).powi(2)
                    + (a[2] - reference[2]).powi(2),
            )
        };
        let (ec, ef) = (err(coarse), err(fine));
        // Order 4 ⇒ ratio ≈ 16; allow a wide band, but it must clearly improve and be >8×.
        assert!(ef < ec, "finer step not better: {ef} vs {ec}");
        assert!(
            ec / ef.max(1e-12) > 8.0,
            "convergence ratio {} too low",
            ec / ef.max(1e-12)
        );
    }

    #[test]
    fn fdma_carrier_frequencies_follow_the_channel_relation() {
        // Channel 0 is the nominal centre: L1 = 1602.000 MHz, L2 = 1246.000 MHz.
        assert!((glonass_l1_frequency(0) - 1_602.0e6).abs() < 1e-3);
        assert!((glonass_l2_frequency(0) - 1_246.0e6).abs() < 1e-3);
        // 0.5625 / 0.4375 MHz per channel step.
        assert!((glonass_l1_frequency(1) - glonass_l1_frequency(0) - 0.5625e6).abs() < 1e-3);
        assert!((glonass_l2_frequency(-7) - (1246.0 - 7.0 * 0.4375) * 1e6).abs() < 1e-3);
        // The ephemeris methods read the record's own channel.
        let mut eph = sample();
        eph.freq_channel = 3;
        assert!((eph.l1_frequency() - glonass_l1_frequency(3)).abs() < 1e-9);
        assert!((eph.l2_frequency() - glonass_l2_frequency(3)).abs() < 1e-9);
    }
}
