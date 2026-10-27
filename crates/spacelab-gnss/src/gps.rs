//! GPS broadcast ephemeris → ECEF position/velocity, per IS-GPS-200 (Table 20-IV / 30-II) with the
//! analytic velocity of Remondi (2004).
//!
//! The broadcast model is a reference Keplerian orbit ([`BroadcastEphemeris::sqrt_a`], `e`, `i0`,
//! `omega0`, `omega`, `m0`) plus a secular mean-motion / node / inclination drift and once-per-orbit
//! harmonic corrections to the argument of latitude, radius and inclination. The receiver evaluates it
//! at a time `tk` seconds from the ephemeris reference epoch [`BroadcastEphemeris::toe`] (already
//! reduced for the GPS-week rollover).
//!
//! Gate: `derivations/gps_broadcast_orbit.mac` shows that the atan2 true anomaly used here is the
//! ICD's half-angle form, that the Remondi velocity is the time derivative of the ICD position,
//! and that `F_RELATIVITY` is `−2√μ/c²` to the ten digits IS-GPS-200 prints; it evaluates the
//! algorithm exactly on the module's sample ephemeris, re-computed independently by
//! `derivations/gps_broadcast_orbit.gp`. The module test
//! `the_broadcast_orbit_is_the_icd_algorithm_evaluated_exactly` (G2) holds the shipped position
//! and velocity, GPS and BeiDou GEO, to those values.

// The single letters (a, e, i, n, r, u, x, y, z) are the IS-GPS-200 orbit notation; the doc allow is
// for constellation/spec proper nouns (BeiDou, IS-GPS-200, RINEX), not identifiers.
#![allow(clippy::many_single_char_names, clippy::doc_markdown)]

/// A GNSS constellation whose broadcast ephemeris uses the Keplerian-plus-corrections model. Each
/// carries the gravitational parameter and Earth-rotation rate fixed by its interface spec — the
/// broadcast elements are consistent with *those* values, so the algorithm must use them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GnssSystem {
    /// GPS (IS-GPS-200).
    #[default]
    Gps,
    /// Galileo (OS-SIS-ICD).
    Galileo,
    /// BeiDou (BDS-SIS-ICD); GEO satellites (PRN ≤ 5 / ≥ 59) get the extra `Rz(Ω̇ₑ·tk)·Rx(−5°)`
    /// rotation automatically (see [`BroadcastEphemeris::is_beidou_geo`]).
    Beidou,
    /// QZSS (IS-QZSS-PNT) — the GPS algorithm and constants.
    Qzss,
    /// NavIC / IRNSS (IRNSS-ICD-SPS) — the GPS algorithm and constants.
    Navic,
}

impl GnssSystem {
    /// The constellation's gravitational parameter `μ`, m³/s², from its ICD.
    #[must_use]
    pub const fn gravitational_parameter(self) -> f64 {
        match self {
            // IS-GPS-200 §20.3.3.4.3; IS-QZSS and IRNSS-ICD adopt the same WGS-84 value.
            Self::Gps | Self::Qzss | Self::Navic => 3.986_005e14,
            // Galileo OS-SIS-ICD / BDS-SIS-ICD both adopt the GTRF/CGCS2000 value.
            Self::Galileo | Self::Beidou => 3.986_004_418e14,
        }
    }

    /// The constellation's Earth-rotation rate `Ω̇ₑ`, rad/s, from its ICD.
    #[must_use]
    pub const fn rotation_rate(self) -> f64 {
        match self {
            Self::Gps | Self::Galileo | Self::Qzss | Self::Navic => 7.292_115_146_7e-5,
            Self::Beidou => 7.292_115e-5,
        }
    }
}

/// GPS gravitational parameter `μ` (IS-GPS-200 §20.3.3.4.3), m³/s².
pub const EARTH_GM: f64 = GnssSystem::Gps.gravitational_parameter();

/// GPS Earth-rotation rate `Ω̇ₑ` (IS-GPS-200 §20.3.3.4.3), rad/s.
pub const EARTH_ROTATION_RATE: f64 = GnssSystem::Gps.rotation_rate();

/// Relativistic clock-correction constant `F = −2√μ/c²` (IS-GPS-200 §20.3.3.3.3.1), s/√m. This is
/// GPS's defined value; Galileo/BeiDou differ by well under a femtosecond in the clock term.
pub const F_RELATIVITY: f64 = -4.442_807_633e-10;

/// Solve Kepler's equation `M = E − e·sin E` for the eccentric anomaly `E`, by Newton's method.
///
/// Converges quadratically for the near-circular GPS orbits (`e ≲ 0.02`); iterates to a `1e-14`
/// residual (a handful of steps) with a hard cap so a pathological input cannot spin.
#[must_use]
pub fn kepler_solve(m: f64, e: f64) -> f64 {
    let mut ek = m;
    for _ in 0..30 {
        let d = (ek - e * libm::sin(ek) - m) / (1.0 - e * libm::cos(ek));
        ek -= d;
        if libm::fabs(d) < 1e-14 {
            break;
        }
    }
    ek
}

/// A GNSS broadcast-ephemeris parameter set (one subframe 1–3 / I-NAV / D1 record). Angles are
/// radians, lengths metres, times seconds of the system week; harmonic-correction amplitudes carry
/// their ICD units (`c**` in metres or radians as appropriate). The [`Self::system`] selects the μ and
/// `Ω̇ₑ` the orbit is evaluated with.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BroadcastEphemeris {
    /// Which constellation these parameters belong to.
    pub system: GnssSystem,
    /// Satellite PRN.
    pub prn: u8,
    /// GPS week number.
    pub week: u16,
    /// Time of ephemeris `toe`, s of GPS week.
    pub toe: f64,
    /// Time of clock `toc`, s of GPS week.
    pub toc: f64,
    /// Square root of the semi-major axis `√a`, √m.
    pub sqrt_a: f64,
    /// Eccentricity `e`.
    pub e: f64,
    /// Inclination at `toe`, `i0`, rad.
    pub i0: f64,
    /// Longitude of ascending node at weekly epoch, `Ω0`, rad.
    pub omega0: f64,
    /// Argument of perigee, `ω`, rad.
    pub omega: f64,
    /// Mean anomaly at `toe`, `M0`, rad.
    pub m0: f64,
    /// Mean-motion correction `Δn`, rad/s.
    pub delta_n: f64,
    /// Rate of inclination `IDOT`, rad/s.
    pub idot: f64,
    /// Rate of right ascension `Ω̇`, rad/s.
    pub omega_dot: f64,
    /// Cosine harmonic correction to the argument of latitude `Cuc`, rad.
    pub cuc: f64,
    /// Sine harmonic correction to the argument of latitude `Cus`, rad.
    pub cus: f64,
    /// Cosine harmonic correction to the radius `Crc`, m.
    pub crc: f64,
    /// Sine harmonic correction to the radius `Crs`, m.
    pub crs: f64,
    /// Cosine harmonic correction to the inclination `Cic`, rad.
    pub cic: f64,
    /// Sine harmonic correction to the inclination `Cis`, rad.
    pub cis: f64,
    /// Clock bias `af0`, s.
    pub af0: f64,
    /// Clock drift `af1`, s/s.
    pub af1: f64,
    /// Clock drift rate `af2`, s/s².
    pub af2: f64,
    /// Group delay `TGD`, s.
    pub tgd: f64,
    /// SV health (0 = healthy).
    pub health: u8,
}

impl BroadcastEphemeris {
    /// Semi-major axis `a = (√a)²`, metres.
    #[must_use]
    pub fn semi_major_axis(&self) -> f64 {
        self.sqrt_a * self.sqrt_a
    }

    /// Corrected mean motion `n = √(μ/a³) + Δn`, rad/s, in the constellation's μ.
    #[must_use]
    pub fn mean_motion(&self) -> f64 {
        let a = self.semi_major_axis();
        libm::sqrt(self.system.gravitational_parameter() / (a * a * a)) + self.delta_n
    }

    /// The eccentric anomaly at `tk` seconds from [`Self::toe`].
    #[must_use]
    pub fn eccentric_anomaly(&self, tk: f64) -> f64 {
        kepler_solve(self.m0 + self.mean_motion() * tk, self.e)
    }

    /// Whether this is a BeiDou GEO satellite (PRN C01–C05 or C59–C63), which the BDS-SIS-ICD
    /// propagates with an extra `Rz(Ω̇ₑ·tk)·Rx(−5°)` frame rotation rather than the MEO/IGSO path.
    #[must_use]
    pub fn is_beidou_geo(&self) -> bool {
        self.system == GnssSystem::Beidou && (self.prn <= 5 || self.prn >= 59)
    }

    /// ECEF position of a BeiDou GEO satellite (BDS-SIS-ICD): the in-plane position carried to a
    /// custom inertial frame with `Ωk = Ω0 + Ω̇·tk − Ω̇ₑ·toe` (no `−Ω̇ₑ·tk`), then rotated to ECEF by
    /// `Rz(Ω̇ₑ·tk)·Rx(−5°)`. Velocity is taken by finite difference (the standard GEO treatment).
    fn geo_position(&self, tk: f64) -> [f64; 3] {
        let a = self.semi_major_axis();
        let e = self.e;
        let ek = kepler_solve(self.m0 + self.mean_motion() * tk, e);
        let (sin_ek, cos_ek) = libm::sincos(ek);
        let nu = libm::atan2(libm::sqrt(1.0 - e * e) * sin_ek, cos_ek - e);
        let phi = nu + self.omega;
        let (s2, c2) = libm::sincos(2.0 * phi);
        let u = phi + self.cus * s2 + self.cuc * c2;
        let r = a * (1.0 - e * cos_ek) + self.crs * s2 + self.crc * c2;
        let i = self.i0 + self.idot * tk + self.cis * s2 + self.cic * c2;
        let (su, cu) = libm::sincos(u);
        let (xp, yp) = (r * cu, r * su);

        let omega_e = self.system.rotation_rate();
        let omega_k = self.omega0 + self.omega_dot * tk - omega_e * self.toe;
        let (so, co) = libm::sincos(omega_k);
        let (si, ci) = libm::sincos(i);
        let (xg, yg, zg) = (xp * co - yp * ci * so, xp * so + yp * ci * co, yp * si);

        // Rotate the custom-frame point to ECEF: Rz(Ω̇ₑ·tk)·Rx(−5°); the −5° is computed, not tabled.
        let (sino, coso) = libm::sincos(omega_e * tk);
        let (s5, c5) = libm::sincos(-5.0 * core::f64::consts::PI / 180.0);
        [
            xg * coso + yg * sino * c5 + zg * sino * s5,
            -xg * sino + yg * coso * c5 + zg * coso * s5,
            -yg * s5 + zg * c5,
        ]
    }

    /// ECEF position `[x, y, z]` (metres) at `tk` seconds from [`Self::toe`], per IS-GPS-200.
    #[must_use]
    pub fn position_ecef(&self, tk: f64) -> [f64; 3] {
        self.position_velocity_ecef(tk).0
    }

    /// ECEF position and velocity (`[m]`, `[m/s]`) at `tk` seconds from [`Self::toe`].
    ///
    /// Position is IS-GPS-200 Table 20-IV; velocity is the analytic derivative (Remondi 2004),
    /// including the `−Ω̇ₑ` Earth-rotation transport, so it is the true ECEF velocity.
    #[must_use]
    pub fn position_velocity_ecef(&self, tk: f64) -> ([f64; 3], [f64; 3]) {
        if self.is_beidou_geo() {
            // GEO position is analytic; its velocity is a central finite difference of it. The step
            // balances the O(h²·|p‴|) truncation against the O(ulp(|p|)/h) cancellation: at a GEO
            // radius, |p‴| = Ω̇ₑ³|p| ≈ 1.6e-5 m/s³ and ulp(|p|) ≈ 7.5e-9 m, so the optimum sits near
            // 0.1 s and costs ~3e-8 m/s. At 1e-3 s the cancellation term alone is ~3.7e-6 m/s.
            let p = self.geo_position(tk);
            let h = 0.1;
            let (pp, pm) = (self.geo_position(tk + h), self.geo_position(tk - h));
            let v = [
                (pp[0] - pm[0]) / (2.0 * h),
                (pp[1] - pm[1]) / (2.0 * h),
                (pp[2] - pm[2]) / (2.0 * h),
            ];
            return (p, v);
        }

        let a = self.semi_major_axis();
        let e = self.e;
        let n = self.mean_motion();

        let ek = kepler_solve(self.m0 + n * tk, e);
        let (sin_ek, cos_ek) = libm::sincos(ek);
        let one_minus_ecos = 1.0 - e * cos_ek;

        // True anomaly and argument of latitude.
        let sqrt_1me2 = libm::sqrt(1.0 - e * e);
        let nu = libm::atan2(sqrt_1me2 * sin_ek, cos_ek - e);
        let phi = nu + self.omega;
        let (sin2phi, cos2phi) = libm::sincos(2.0 * phi);

        // Second-harmonic corrections.
        let du = self.cus * sin2phi + self.cuc * cos2phi;
        let dr = self.crs * sin2phi + self.crc * cos2phi;
        let di = self.cis * sin2phi + self.cic * cos2phi;

        let u = phi + du;
        let r = a * one_minus_ecos + dr;
        let i = self.i0 + self.idot * tk + di;

        let (sin_u, cos_u) = libm::sincos(u);
        let xp = r * cos_u;
        let yp = r * sin_u;

        let omega_e = self.system.rotation_rate();
        let omega_k = self.omega0 + (self.omega_dot - omega_e) * tk - omega_e * self.toe;
        let (sin_o, cos_o) = libm::sincos(omega_k);
        let (sin_i, cos_i) = libm::sincos(i);

        let position = [
            xp * cos_o - yp * cos_i * sin_o,
            xp * sin_o + yp * cos_i * cos_o,
            yp * sin_i,
        ];

        // Analytic velocity (Remondi): differentiate each stage.
        let ek_dot = n / one_minus_ecos;
        let nu_dot = ek_dot * sqrt_1me2 / one_minus_ecos;
        let u_dot = nu_dot * (1.0 + 2.0 * (self.cus * cos2phi - self.cuc * sin2phi));
        let r_dot =
            a * e * sin_ek * ek_dot + 2.0 * nu_dot * (self.crs * cos2phi - self.crc * sin2phi);
        let i_dot = self.idot + 2.0 * nu_dot * (self.cis * cos2phi - self.cic * sin2phi);

        let xp_dot = r_dot * cos_u - yp * u_dot;
        let yp_dot = r_dot * sin_u + xp * u_dot;
        let omega_k_dot = self.omega_dot - omega_e;

        let velocity = [
            xp_dot * cos_o - yp_dot * cos_i * sin_o + yp * sin_i * sin_o * i_dot
                - position[1] * omega_k_dot,
            xp_dot * sin_o + yp_dot * cos_i * cos_o - yp * sin_i * cos_o * i_dot
                + position[0] * omega_k_dot,
            yp_dot * sin_i + yp * cos_i * i_dot,
        ];

        (position, velocity)
    }

    /// SV clock correction (seconds) at `tk_clock` seconds from [`Self::toc`], including the
    /// relativistic eccentricity term but *not* the group delay `TGD` (subtract [`Self::tgd`] for the
    /// L1 single-frequency user). `ek` is the eccentric anomaly at the same instant.
    #[must_use]
    pub fn clock_correction(&self, tk_clock: f64, ek: f64) -> f64 {
        let relativistic = F_RELATIVITY * self.e * self.sqrt_a * libm::sin(ek);
        self.af0 + self.af1 * tk_clock + self.af2 * tk_clock * tk_clock + relativistic
    }
}

#[cfg(test)]
mod tests {
    // The references are carried at the precision Maxima printed them to.
    #![allow(clippy::excessive_precision)]
    use super::*;
    use spacelab_core::{Angle, Vec3};
    use spacelab_elements::KeplerianElements;

    /// A realistic GPS ephemeris (Block-II order-of-magnitude values), enough to exercise every term.
    fn sample() -> BroadcastEphemeris {
        BroadcastEphemeris {
            system: GnssSystem::Gps,
            prn: 1,
            week: 2300,
            toe: 216_000.0,
            toc: 216_000.0,
            sqrt_a: 5153.65,
            e: 0.006_2,
            i0: 0.959_9,
            omega0: -1.234,
            omega: 0.567,
            m0: 0.321,
            delta_n: 4.5e-9,
            idot: -2.1e-10,
            omega_dot: -8.1e-9,
            cuc: 1.2e-6,
            cus: 8.3e-6,
            crc: 210.0,
            crs: 35.0,
            cic: -1.1e-7,
            cis: 9.0e-8,
            af0: 1.5e-4,
            af1: 3.4e-12,
            af2: 0.0,
            tgd: -1.0e-8,
            health: 0,
        }
    }

    #[test]
    fn kepler_equation_residual_is_tiny() {
        for &e in &[0.0, 0.006_2, 0.02] {
            for k in 0..12 {
                let m = -3.0 + 0.5 * f64::from(k);
                let ek = kepler_solve(m, e);
                assert!((ek - e * ek.sin() - m).abs() < 1e-13, "M={m} e={e}");
            }
        }
    }

    /// The analytic (Remondi) velocity must be the derivative of the position the same call returns —
    /// checked against a fourth-order central difference of `position_ecef`, which shares no algebra
    /// with the differentiation. Every previous assertion on this function was rotation-invariant
    /// (radius, speed, vis-viva), so a wrong `Ω̇ₖ` or a swapped `cos_i` was free to pass; this pins all
    /// three components.
    ///
    /// It found two. For a BeiDou MEO/IGSO the residual was **2.99e-5 m/s** because `Ω̇ₖ` used the GPS
    /// `EARTH_ROTATION_RATE` constant while the position used the per-system `rotation_rate()`
    /// (CGCS2000's 7.292115e-5, 1.467e-12 rad/s apart) — 850× the numerical floor, and invisible
    /// because the only BeiDou fixture had `prn = 1`, which routes to the GEO branch and never reaches
    /// that line. And the GEO branch's own difference step was 1e-3 s, where `ulp(|p|)/2h` cancellation
    /// alone contributes ~3.7e-6 m/s.
    #[test]
    fn the_analytic_velocity_is_the_derivative_of_the_position_it_returns() {
        // Fourth order: (−p₊₂ + 8p₊₁ − 8p₋₁ + p₋₂)/12h. At h = 1 s the truncation is below 1e-10 m/s
        // and the cancellation below 4e-9 m/s, so anything above ~1e-7 is the model, not the stencil.
        fn central_difference(eph: &BroadcastEphemeris, tk: f64) -> [f64; 3] {
            let h = 1.0;
            let (p1, p2) = (eph.position_ecef(tk + h), eph.position_ecef(tk + 2.0 * h));
            let (m1, m2) = (eph.position_ecef(tk - h), eph.position_ecef(tk - 2.0 * h));
            let mut d = [0.0; 3];
            for i in 0..3 {
                d[i] = (-p2[i] + 8.0 * p1[i] - 8.0 * m1[i] + m2[i]) / (12.0 * h);
            }
            d
        }

        let mut worst = 0.0_f64;
        for (system, prn) in [
            (GnssSystem::Gps, 1),
            (GnssSystem::Galileo, 11),
            (GnssSystem::Qzss, 3),
            (GnssSystem::Navic, 7),
            (GnssSystem::Beidou, 20), // MEO/IGSO: the analytic branch
            (GnssSystem::Beidou, 1),  // C01: the GEO branch, velocity by difference
        ] {
            let mut eph = sample();
            eph.system = system;
            eph.prn = prn;
            for tk in [-7200.0_f64, -1800.0, 0.0, 1800.0, 7200.0] {
                let (_, v) = eph.position_velocity_ecef(tk);
                let fd = central_difference(&eph, tk);
                let d = [v[0] - fd[0], v[1] - fd[1], v[2] - fd[2]];
                let err = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                worst = worst.max(err);
                assert!(
                    err < 1.0e-6,
                    "{system:?} PRN {prn} at tk = {tk}: analytic v {v:?} vs d/dt of position {fd:?}, \
                     |Δ| = {err:e} m/s"
                );
            }
        }
        println!("worst |v − dp/dt| over all constellations = {worst:.3e} m/s");
    }

    /// `Ωₖ = Ω₀ + (Ω̇ − Ω̇ₑ)·tₖ − Ω̇ₑ·t_oe`: the `t_oe` term is the Earth rotation accumulated between
    /// the ephemeris reference epoch and the frame's own origin, so advancing `t_oe` by Δ must rotate
    /// the whole ECEF position *backwards* by exactly `Ω̇ₑ·Δ` about the pole. Sign and coefficient both
    /// ride on that; the radius/speed assertions elsewhere are invariant under any rotation about ẑ and
    /// cannot see either.
    #[test]
    fn shifting_the_reference_epoch_rotates_the_ecef_position_backwards_by_the_earth_rate() {
        let base = sample();
        let omega_e = base.system.rotation_rate();
        for delta in [900.0_f64, -3600.0, 43_200.0] {
            let mut shifted = base;
            shifted.toe = base.toe + delta;
            // Same tₖ (seconds *from* t_oe), so the orbit is at the same point of its own path and only
            // the Earth-rotation term differs.
            let (p, q) = (base.position_ecef(600.0), shifted.position_ecef(600.0));
            let want = -omega_e * delta;
            let got = libm::atan2(q[1], q[0]) - libm::atan2(p[1], p[0]);
            let got = libm::atan2(libm::sin(got), libm::cos(got)); // wrap to (−π, π]
            let want = libm::atan2(libm::sin(want), libm::cos(want));
            assert!(
                (got - want).abs() < 1e-12,
                "Δt_oe = {delta}: rotated by {got} rad, expected −Ω̇ₑ·Δ = {want}"
            );
            // A pure rotation about the pole: radius and z untouched.
            let (rp, rq) = (libm::hypot(p[0], p[1]), libm::hypot(q[0], q[1]));
            assert!((rp - rq).abs() < 1e-6 && (p[2] - q[2]).abs() < 1e-6);
        }
    }

    #[test]
    fn gps_orbit_radius_and_inertial_speed_obey_vis_viva() {
        let eph = sample();
        let (p, v) = eph.position_velocity_ecef(120.0);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        // GPS orbits sit near 26 560 km radius (~20 200 km altitude).
        assert!((26.0e6..27.2e6).contains(&r), "radius {r}");

        // Undo the Earth-rotation carry to recover the inertial velocity: v_i = v_ecef + Ωₑ ẑ × r.
        let vi = [
            v[0] - EARTH_ROTATION_RATE * p[1],
            v[1] + EARTH_ROTATION_RATE * p[0],
            v[2],
        ];
        let speed = (vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2]).sqrt();
        // Vis-viva: v² = μ(2/r − 1/a). This exercises the whole velocity chain, Earth rotation included.
        let expected = (EARTH_GM * (2.0 / r - 1.0 / eph.semi_major_axis())).sqrt();
        assert!(
            (speed - expected).abs() < 1e-2 * expected,
            "speed {speed} vs vis-viva {expected}"
        );
    }

    #[test]
    fn galileo_and_beidou_orbits_use_their_own_constants_and_obey_vis_viva() {
        // Galileo √a ≈ 5440.6 (a ≈ 29 600 km); BeiDou MEO √a ≈ 5282.6 (a ≈ 27 906 km).
        for (system, sqrt_a, r_lo, r_hi) in [
            (GnssSystem::Galileo, 5440.6, 29.0e6, 30.2e6),
            (GnssSystem::Beidou, 5282.6, 27.3e6, 28.5e6),
        ] {
            let eph = BroadcastEphemeris {
                system,
                sqrt_a,
                ..sample()
            };
            let (p, v) = eph.position_velocity_ecef(200.0);
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r_lo..r_hi).contains(&r), "{system:?} radius {r}");

            let omega_e = system.rotation_rate();
            let vi = [v[0] - omega_e * p[1], v[1] + omega_e * p[0], v[2]];
            let speed = (vi[0] * vi[0] + vi[1] * vi[1] + vi[2] * vi[2]).sqrt();
            let expected =
                (system.gravitational_parameter() * (2.0 / r - 1.0 / eph.semi_major_axis())).sqrt();
            assert!(
                (speed - expected).abs() < 1e-2 * expected,
                "{system:?} speed {speed} vs vis-viva {expected}"
            );
        }
    }

    #[test]
    fn reduced_two_body_orbit_matches_the_keplerian_state_oracle() {
        // With every correction, drift and Earth rotation removed and evaluated at tk = toe = 0, the
        // IS-GPS-200 transform collapses to the classical Keplerian element → ECEF conversion. Cross-
        // check the position against the independently-tested spacelab-elements to_state().
        let mut eph = sample();
        eph.delta_n = 0.0;
        eph.idot = 0.0;
        eph.omega_dot = 0.0;
        eph.cuc = 0.0;
        eph.cus = 0.0;
        eph.crc = 0.0;
        eph.crs = 0.0;
        eph.cic = 0.0;
        eph.cis = 0.0;
        eph.toe = 0.0;

        let p = eph.position_ecef(0.0);

        // Build the same orbit as Keplerian elements: true anomaly from M0 via Kepler's equation.
        let ek = kepler_solve(eph.m0, eph.e);
        let (sin_ek, cos_ek) = ek.sin_cos();
        let nu = libm::atan2((1.0 - eph.e * eph.e).sqrt() * sin_ek, cos_ek - eph.e);
        let kep = KeplerianElements {
            semi_major_axis: eph.semi_major_axis(),
            eccentricity: eph.e,
            inclination: Angle::from_radians(eph.i0),
            raan: Angle::from_radians(eph.omega0),
            arg_periapsis: Angle::from_radians(eph.omega),
            true_anomaly: Angle::from_radians(nu),
            mu: EARTH_GM,
        };
        let (r_oracle, _) = kep.to_state();
        let diff = (Vec3::new(p[0], p[1], p[2]) - r_oracle).norm() / r_oracle.norm();
        assert!(diff < 1e-9, "relative position error {diff:e}");
    }

    #[test]
    fn the_relativity_constant_is_minus_two_root_mu_over_c_squared() {
        // `F_RELATIVITY` is a tabled number used in exactly one place, and no test called that place,
        // so even its sign was unconstrained. IS-GPS-200 §20.3.3.3.3.1 defines it as −2√μ/c².
        const C: f64 = 299_792_458.0;
        let want = -2.0 * EARTH_GM.sqrt() / (C * C);
        assert!(
            (F_RELATIVITY - want).abs() < 1e-19,
            "F = {F_RELATIVITY} vs −2√μ/c² = {want}"
        );
    }

    /// (G2) Position and velocity are the IS-GPS-200 Table 20-IV algorithm evaluated exactly: the
    /// references are `derivations/gps_broadcast_orbit.mac` at 40 digits, from this sample's doubles
    /// as exact binary rationals, and agree with the independent PARI/GP route of
    /// `derivations/gps_broadcast_orbit.gp` to 1e-30. The same derivation shows the code's atan2
    /// true anomaly is the ICD's half-angle form and that the Remondi velocity is the symbolic time
    /// derivative of the ICD position, so the velocity references are that derivative, not a
    /// difference quotient.
    ///
    /// Three epochs a quarter-orbit or more apart put every harmonic factor at a different phase,
    /// and the BeiDou GEO case carries the BDS-SIS-ICD `Rz(Ω̇ₑ·tk)·Rx(−5°)` path. The residual is
    /// the f64 evaluation: Kepler's equation is solved to a step below 1e-14 rad, which is 3e-7 m at
    /// these radii, and each trigonometric call rounds at 1e-16 relative.
    #[test]
    fn the_broadcast_orbit_is_the_icd_algorithm_evaluated_exactly() {
        let eph = sample();
        for (tk, want_p, want_v) in [
            (
                0.0,
                [
                    -1.608_349_047_629_949_522_608_307_177_496_015_129_125e7,
                    1.245_509_414_443_383_025_514_302_169_956_380_305_266e7,
                    1.683_329_974_311_097_511_202_157_632_207_521_635_224e7,
                ],
                [
                    4.386_156_686_763_534_240_872_302_271_518_575_758_199e2,
                    -2.132_868_891_403_938_462_852_896_741_488_921_059_079e3,
                    2.009_134_984_342_218_859_848_980_124_961_002_776_766e3,
                ],
            ),
            (
                1800.0,
                [
                    -1.541_415_114_366_485_200_684_389_006_242_593_955_029e7,
                    8.228_261_553_544_978_826_839_208_835_223_353_847_592e6,
                    1.982_090_795_702_325_132_508_994_429_627_142_134_235e7,
                ],
                [
                    2.838_977_683_361_724_098_871_682_382_161_423_540_394e2,
                    -2.535_947_046_765_945_106_357_328_123_330_572_673_875e3,
                    1.291_283_960_177_011_331_596_598_677_536_856_884_211e3,
                ],
            ),
            (
                -7200.0,
                [
                    -1.727_221_023_687_484_819_761_309_720_069_030_491_116e7,
                    1.967_569_356_933_729_596_983_169_688_362_624_037_098e7,
                    -3.674_667_302_932_292_817_188_733_379_826_896_841_437e6,
                ],
                [
                    -5.396_257_837_432_511_153_667_658_796_995_594_911_98e2,
                    9.182_671_115_107_704_676_621_603_652_503_407_311_922e1,
                    3.144_095_093_111_117_689_203_125_539_001_010_362_227e3,
                ],
            ),
        ] {
            let (p, v) = eph.position_velocity_ecef(tk);
            for k in 0..3 {
                assert!(
                    (p[k] - want_p[k]).abs() < 1e-6,
                    "tk = {tk}, axis {k}: position {} vs {}, |d| = {:.2e} m",
                    p[k],
                    want_p[k],
                    (p[k] - want_p[k]).abs()
                );
                assert!(
                    (v[k] - want_v[k]).abs() < 1e-9,
                    "tk = {tk}, axis {k}: velocity {} vs {}, |d| = {:.2e} m/s",
                    v[k],
                    want_v[k],
                    (v[k] - want_v[k]).abs()
                );
            }
        }

        let geo = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 1,
            sqrt_a: 6493.4,
            e: 0.000_4,
            i0: 0.02,
            ..sample()
        };
        let want = [
            -3.899_982_835_142_000_127_548_946_848_277_510_675_753e7,
            1.586_104_032_190_708_188_350_330_687_563_835_373_037e7,
            1.980_797_178_700_234_885_081_610_668_226_004_636_799e6,
        ];
        let p = geo.position_ecef(300.0);
        for k in 0..3 {
            assert!(
                (p[k] - want[k]).abs() < 1e-6,
                "GEO axis {k}: {} vs {}, |d| = {:.2e} m",
                p[k],
                want[k],
                (p[k] - want[k]).abs()
            );
        }
    }

    #[test]
    fn kepler_newton_converges_where_a_fixed_point_iteration_would_not() {
        // At GPS eccentricities the Newton denominator `1 − e·cos E` is within 2 % of one, so dropping
        // it, inverting it or flipping its sign still converges and `kepler_equation_residual_is_tiny`
        // cannot tell Newton from a plain fixed-point iteration. Eccentric orbits separate them: with
        // the real derivative the residual is at the rounding floor in a handful of steps at every
        // eccentricity below, while the same loop without it does not reach 1e-14 in its 30-iteration
        // cap once e passes ~0.2.
        for &e in &[0.2_f64, 0.5, 0.7, 0.9] {
            for k in 0..24 {
                let m = -3.0 + 0.25 * f64::from(k);
                let ek = kepler_solve(m, e);
                let residual = ek - e * ek.sin() - m;
                assert!(
                    residual.abs() < 1e-14,
                    "e = {e}, M = {m}: residual {residual:e}"
                );
            }
        }
    }

    #[test]
    fn mean_motion_is_the_keplerian_rate_plus_the_broadcast_correction() {
        // `Δn` is 4.5e-9 rad/s here — about 29 m of along-track over two minutes, far inside the 1 %
        // vis-viva band, and the Keplerian-oracle test zeroes it outright. So its sign was free.
        let eph = sample();
        let a = eph.semi_major_axis();
        let n0 = (EARTH_GM / (a * a * a)).sqrt();
        // Bound the residual against `Δn` itself, not against `n`: `n` is 3e4 times larger, so the
        // rounding of `n₀ + Δn` is an ulp of `n` and would swamp an absolute tolerance. Nine digits of
        // margin on `Δn` still leaves a sign flip — a change of 2·Δn — caught by ten orders.
        assert!(
            (eph.mean_motion() - n0 - eph.delta_n).abs() < 1e-9 * eph.delta_n.abs(),
            "n = {} vs √(μ/a³) + Δn = {}",
            eph.mean_motion(),
            n0 + eph.delta_n
        );
    }

    #[test]
    fn the_eccentric_anomaly_solves_keplers_equation_at_the_propagated_mean_anomaly() {
        // Nothing called `eccentric_anomaly` at all: the propagator inlines `kepler_solve` instead, so
        // the accessor could return a constant. Its contract is Kepler's equation at M(tk) = M₀ + n·tk.
        let eph = sample();
        for tk in [0.0, 1.0, 120.0, -1800.0, 7200.0] {
            let ek = eph.eccentric_anomaly(tk);
            let m = eph.m0 + eph.mean_motion() * tk;
            let residual = ek - eph.e * ek.sin() - m;
            assert!(residual.abs() < 1e-13, "tk = {tk}: residual {residual:e}");
        }
    }

    #[test]
    fn the_clock_correction_is_the_icd_polynomial_plus_the_relativistic_term() {
        // No test called `clock_correction`, so returning a constant survived. IS-GPS-200 §20.3.3.3.3.1:
        // Δtsv = af0 + af1·(t − toc) + af2·(t − toc)² + F·e·√A·sin E.
        let eph = BroadcastEphemeris {
            af2: 1.7e-19, // the sample carries af2 = 0, which hides the quadratic term entirely
            ..sample()
        };

        // The polynomial alone, with the relativistic term switched off by a circular orbit.
        let clock_only = BroadcastEphemeris { e: 0.0, ..eph };
        assert!(
            clock_only.clock_correction(0.0, 0.0).to_bits() == eph.af0.to_bits(),
            "at tk = 0 and E = 0 the correction is af0"
        );
        for t in [-302_400.0, -3600.0, 1.0, 7200.0, 302_400.0] {
            let want = eph.af0 + eph.af1 * t + eph.af2 * t * t;
            let got = clock_only.clock_correction(t, 0.7);
            assert!(
                (got - want).abs() < 1e-18,
                "tk = {t}: {got} vs af0 + af1·t + af2·t² = {want}"
            );
        }

        // The relativistic term alone. Its amplitude |F|·e·√A is the textbook GPS figure: 14.20 ns of
        // clock, 4.256 m of range, for this ephemeris's e = 0.0062.
        let rel_only = BroadcastEphemeris {
            af0: 0.0,
            af1: 0.0,
            af2: 0.0,
            ..eph
        };
        let amplitude = rel_only.clock_correction(0.0, core::f64::consts::FRAC_PI_2);
        assert!(
            (amplitude.abs() - 1.419_593_884_584_247_749_902_824_751_776_936_381_639e-8).abs()
                < 1e-22,
            "relativistic amplitude {amplitude} s"
        );
        assert!(amplitude < 0.0, "F < 0, so the term is negative at E = π/2");
        // It is odd in E and vanishes at perigee and apogee.
        for ek in [0.3_f64, 1.9, -2.6] {
            let plus = rel_only.clock_correction(0.0, ek);
            assert!(
                (plus + rel_only.clock_correction(0.0, -ek)).abs() < 1e-24,
                "the relativistic term is odd in E"
            );
            assert!(
                (plus - amplitude * ek.sin()).abs() < 1e-24,
                "the relativistic term goes as sin E"
            );
        }
        assert!(rel_only.clock_correction(0.0, 0.0).abs() < 1e-24);
    }

    #[test]
    fn the_beidou_geo_path_shares_the_in_plane_radius_with_the_meo_path() {
        // Both IS-GPS-200 paths solve the same in-plane conic and then rotate it into ECEF by different
        // orthonormal chains, so |r⃗| is identical along either. The MEO path is anchored against
        // `spacelab-elements` by `reduced_two_body_orbit_matches_the_keplerian_state_oracle`; this
        // carries that anchor across to the GEO branch, whose only assertions were a 1300 km-wide radius
        // band and two speed inequalities.
        let geo = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 1,
            sqrt_a: 6493.4,
            e: 0.000_4,
            i0: 0.02,
            ..sample()
        };
        // The eccentric variant is not a real BeiDou GEO — e = 0.5 is far outside the ~4e-4 of the
        // constellation. It is here because `√(1 − e²)` inside the true anomaly is what sets the shape
        // of the conic, and at a realistic e that whole factor is 1 − 8e-8: mutating it moves the
        // radius by 1e-12 of itself, under any tolerance worth asserting. At e = 0.5 the same mutation
        // moves it by 1.7e-6 — measured — which is what makes the term observable at all.
        for e in [0.000_4, 0.5] {
            let geo = BroadcastEphemeris { e, ..geo };
            let meo = BroadcastEphemeris { prn: 20, ..geo };
            assert!(geo.is_beidou_geo() && !meo.is_beidou_geo());
            for tk in [0.0, 120.0, 900.0, 3600.0, 21_600.0, -1800.0] {
                let norm = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                let (tilted, plain) = (norm(geo.geo_position(tk)), norm(meo.position_ecef(tk)));
                assert!(
                    (tilted - plain).abs() < 1e-9 * plain,
                    "e = {e}, tk = {tk}: GEO radius {tilted} vs MEO radius {plain}"
                );
            }
        }
    }

    #[test]
    fn the_geo_path_turns_by_exactly_the_angle_each_broadcast_term_implies() {
        // Matching the radius pins `r`, but the argument of latitude, the inclination and the node
        // only *turn* the vector and leave |r⃗| alone, so they stayed invisible. The GEO rotation chain
        // is orthonormal, so the angle between two evaluations that differ in one broadcast term is
        // exactly the angle that term turns the in-plane solution through — and that has a closed form
        // which needs no knowledge of the rotation itself.
        let base = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 1,
            sqrt_a: 6493.4,
            e: 0.01,
            i0: 0.02,
            idot: 0.0,
            omega_dot: 0.0,
            delta_n: 0.0,
            cuc: 0.0,
            cus: 0.0,
            crc: 0.0,
            crs: 0.0,
            cic: 0.0,
            cis: 0.0,
            ..sample()
        };
        let tk = 3600.0;
        let angle = |a: [f64; 3], b: [f64; 3]| {
            let cross = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            let n = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            libm::atan2(n, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])
        };

        // The in-plane solution at `tk`, from Kepler's equation and the conic — with every correction
        // zeroed in `base`, the argument of latitude is ν + ω and the inclination is i0.
        let ek = base.eccentric_anomaly(tk);
        let (sin_ek, cos_ek) = ek.sin_cos();
        let nu = libm::atan2((1.0 - base.e * base.e).sqrt() * sin_ek, cos_ek - base.e);
        let u = nu + base.omega;
        let sin2phi = (2.0 * u).sin();
        let reference = base.geo_position(tk);

        // Cus turns the point inside the orbit plane by exactly Cus·sin 2φ.
        let cus = 1.0e-3;
        let got = angle(
            BroadcastEphemeris { cus, ..base }.geo_position(tk),
            reference,
        );
        assert!(
            (got - (cus * sin2phi).abs()).abs() < 1e-9 * got,
            "Cus: turned {got} rad, expected {}",
            (cus * sin2phi).abs()
        );

        // Cis tilts the plane about the node by Δi = Cis·sin 2φ. A point at argument of latitude u
        // then moves through cos θ = cos²u + sin²u·cos Δi.
        let cis = 1.0e-3;
        let di = cis * sin2phi;
        let want = (u.cos().powi(2) + u.sin().powi(2) * di.cos()).acos();
        let got = angle(
            BroadcastEphemeris { cis, ..base }.geo_position(tk),
            reference,
        );
        assert!(
            (got - want).abs() < 1e-7 * want,
            "Cis: turned {got}, expected {want}"
        );

        // Ω̇ turns the custom inertial frame about its z axis by Ω̇·tk. A point at latitude ψ in that
        // frame, sin ψ = sin i·sin u, moves through cos θ = sin²ψ + cos²ψ·cos δ.
        let omega_dot = 1.0e-7;
        let delta = omega_dot * tk;
        let sin_psi = base.i0.sin() * u.sin();
        let want = (sin_psi * sin_psi + (1.0 - sin_psi * sin_psi) * delta.cos()).acos();
        let got = angle(
            BroadcastEphemeris { omega_dot, ..base }.geo_position(tk),
            reference,
        );
        assert!(
            (got - want).abs() < 1e-7 * want,
            "Ω̇: turned {got}, expected {want}"
        );
    }

    #[test]
    fn the_beidou_geo_frame_is_tilted_five_degrees_from_the_ecef_equator() {
        // The BDS-SIS-ICD's `Rx(−5°)` is the whole reason the GEO branch exists, and nothing measured
        // it. Strip the orbit to a circular equatorial one with no corrections: then the in-plane
        // solution is `r·(cos(u + Ωk), sin(u + Ωk), 0)` in the custom frame, and at the instant
        // `u + Ωk = π/2` the whole position collapses to a closed form in which the tilt is the only
        // free quantity — the satellite sits exactly 5° off the ECEF equator.
        let geo = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 1,
            sqrt_a: 6493.4,
            e: 0.0,
            i0: 0.0,
            idot: 0.0,
            omega_dot: 0.0,
            delta_n: 0.0,
            cuc: 0.0,
            cus: 0.0,
            crc: 0.0,
            crs: 0.0,
            cic: 0.0,
            cis: 0.0,
            ..sample()
        };
        let r = geo.semi_major_axis();
        let omega_e = GnssSystem::Beidou.rotation_rate();
        let omega_k = geo.omega0 - omega_e * geo.toe;
        // At e = 0 the argument of latitude is M₀ + n·tk + ω, so this is the quarter-turn instant.
        let tk = (core::f64::consts::FRAC_PI_2 - geo.m0 - geo.omega - omega_k) / geo.mean_motion();

        let p = geo.geo_position(tk);
        let (sin_o, cos_o) = (omega_e * tk).sin_cos();
        let (s5, c5) = 5.0_f64.to_radians().sin_cos();
        let want = [r * sin_o * c5, r * cos_o * c5, r * s5];
        for axis in 0..3 {
            assert!(
                (p[axis] - want[axis]).abs() < 1e-6,
                "axis {axis}: {} vs {}",
                p[axis],
                want[axis]
            );
        }
        let norm = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        let latitude = (p[2] / norm).asin().to_degrees();
        assert!(
            (latitude - 5.0).abs() < 1e-9,
            "geocentric latitude {latitude}°, not the BDS-SIS-ICD 5° tilt"
        );
    }

    #[test]
    // One case per broadcast correction, each with its own closed-form shift: splitting the table
    // would separate a correction from the parameter it is checked against.
    #[allow(clippy::too_many_lines)]
    fn each_broadcast_correction_is_exactly_a_shift_of_the_parameter_it_corrects() {
        // The turn-angle test above measures how far a correction moves the satellite but not which
        // way, so flipping `+ Cus·sin 2φ` to `−` turns it the same amount backwards and survives.
        // Every correction is an additive term on a base parameter, so at an instant where its
        // harmonic factor is exactly one, the corrected orbit is the *same orbit* as one carrying the
        // shift in the base parameter outright — and that equality carries the sign.
        let base = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 1,
            sqrt_a: 6493.4,
            e: 0.0,
            i0: 0.02,
            idot: 0.0,
            omega_dot: 0.0,
            delta_n: 0.0,
            cuc: 0.0,
            cus: 0.0,
            crc: 0.0,
            crs: 0.0,
            cic: 0.0,
            cis: 0.0,
            ..sample()
        };
        // A circular orbit makes ν = M exactly, so φ = M₀ + n·tk + ω is available in closed form and
        // so are the two harmonic factors. Evaluate at φ = π/6: `sin 2φ = √3/2` and `cos 2φ = 1/2`,
        // both away from 0 and ±1, and `sin u = 1/2`, so the inclination actually reaches the
        // position. An instant where a factor is exactly one is the wrong choice — there `c·cos 2φ`
        // and `c/cos 2φ` are the same number — and one where `sin u = 0` puts the satellite on the
        // node, where the inclination has no effect at all.
        let n = base.mean_motion();
        let phi = core::f64::consts::FRAC_PI_6;
        let tk = (phi - base.omega - base.m0) / n;
        let (sin2phi, cos2phi) = (2.0 * phi).sin_cos();
        let x = 1.0e-3; // 1 mrad of correction — about 42 km at this radius

        // A tenth of a millimetre, against a shift the correction is worth 42 km of. Only the
        // rounding of two different expression paths lives under it.
        let same = |label: &str, a: [f64; 3], b: [f64; 3]| {
            for axis in 0..3 {
                assert!(
                    (a[axis] - b[axis]).abs() < 1e-4,
                    "{label}: axis {axis}, {} vs {}",
                    a[axis],
                    b[axis]
                );
            }
        };

        // Shifting the base parameter by the correction's *value at this instant* reproduces the
        // corrected orbit exactly, sign and all — the shifted ephemeris carries no harmonic term, so
        // its own φ moving does not matter.
        for (label, corrected, shifted) in [
            (
                "Cus",
                BroadcastEphemeris { cus: x, ..base },
                BroadcastEphemeris {
                    omega: base.omega + x * sin2phi,
                    ..base
                },
            ),
            (
                "Cuc",
                BroadcastEphemeris { cuc: x, ..base },
                BroadcastEphemeris {
                    omega: base.omega + x * cos2phi,
                    ..base
                },
            ),
            (
                "Cis",
                BroadcastEphemeris { cis: x, ..base },
                BroadcastEphemeris {
                    i0: base.i0 + x * sin2phi,
                    ..base
                },
            ),
            (
                "Cic",
                BroadcastEphemeris { cic: x, ..base },
                BroadcastEphemeris {
                    i0: base.i0 + x * cos2phi,
                    ..base
                },
            ),
            (
                "IDOT",
                BroadcastEphemeris {
                    idot: x / tk,
                    ..base
                },
                BroadcastEphemeris {
                    i0: base.i0 + x,
                    ..base
                },
            ),
            (
                "Ω̇",
                BroadcastEphemeris {
                    omega_dot: x / tk,
                    ..base
                },
                BroadcastEphemeris {
                    omega0: base.omega0 + x,
                    ..base
                },
            ),
        ] {
            same(label, corrected.geo_position(tk), shifted.geo_position(tk));
        }

        // Crs and Crc shift the radius rather than an angle, so they show up as a change of |r⃗|.
        for (label, crs, crc, factor) in
            [("Crs", 500.0, 0.0, sin2phi), ("Crc", 0.0, 500.0, cos2phi)]
        {
            let norm = |p: [f64; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let moved = BroadcastEphemeris { crc, crs, ..base }.geo_position(tk);
            let want = norm(base.geo_position(tk)) + (crs + crc) * factor;
            assert!(
                (norm(moved) - want).abs() < 1e-6,
                "{label}: radius {} vs {want}",
                norm(moved)
            );
        }
    }

    #[test]
    fn beidou_geo_sits_at_geostationary_radius_and_is_nearly_fixed_in_ecef() {
        // A BeiDou GEO PRN (C01): √a for the ~42 164 km geostationary radius, small e/i.
        let geo = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 1,
            sqrt_a: 6493.4,
            e: 0.000_4,
            i0: 0.02,
            ..sample()
        };
        assert!(geo.is_beidou_geo());
        let (p, v) = geo.position_velocity_ecef(300.0);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((41.5e6..42.8e6).contains(&r), "GEO radius {r}");
        // The defining GEO signature: the Rz(Ω̇ₑ·tk) rotation removes Earth-rotation motion, so the
        // satellite is nearly fixed in ECEF — tens of m/s, not the ~3 km/s of a MEO. A wrong-sign
        // rotation would leave the full inertial speed here.
        let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!(speed < 500.0, "GEO ECEF speed {speed}");

        // A BeiDou MEO/IGSO PRN is not GEO and takes the MEO path (fast in ECEF).
        let meo = BroadcastEphemeris {
            system: GnssSystem::Beidou,
            prn: 20,
            sqrt_a: 5282.6,
            ..sample()
        };
        assert!(!meo.is_beidou_geo());
        let (_, vm) = meo.position_velocity_ecef(300.0);
        let vmag = (vm[0] * vm[0] + vm[1] * vm[1] + vm[2] * vm[2]).sqrt();
        assert!(vmag > 1000.0, "MEO ECEF speed {vmag}");
    }
}
