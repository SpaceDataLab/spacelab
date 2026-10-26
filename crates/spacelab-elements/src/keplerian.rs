//! Osculating classical (Keplerian) orbital elements, and the two conversions that matter:
//! a state vector to the elements that instantaneously fit it, and back.
//!
//! The four angles are typed [`Angle`] — the degree/radian confusion is the single most common
//! elements bug, and it is preventable at compile time. The semi-major axis, eccentricity and `μ`
//! are bare `f64` (metres, dimensionless, m³/s²): they arrive metres-by-construction from a [`Vec3`]
//! state and are in constant arithmetic with `μ`, where a `Length` wrapper would be unwrapped on
//! every line without guarding a real bug in this crate.
//!
//! The element set carries the `μ` it was computed with. An osculating set is meaningless without
//! the gravitational parameter that produced it, and querying a set built with WGS-84 `μ` for its
//! period under WGS-72 `μ` is a wrong answer that looks plausible — so it is made unrepresentable:
//! [`period`](KeplerianElements::period) and [`mean_motion`](KeplerianElements::mean_motion) take no
//! argument, they read the `μ` frozen into the set.
//!
//! Elliptical orbits only (`0 ≤ e < 1`). The geometry (`a, e, i, Ω, ω, ν`) is defined for any conic,
//! but the mean-anomaly machinery below assumes a bound orbit; hyperbolic anomaly is not modelled.
//!
//! `derivations/keplerian_elements.mac` proves `from_state` inverts `to_state` branch by branch and
//! derives the anomaly maps; `derivations/j2_secular_rates.mac` derives the `J₂` rates. The gates
//! are `kepler_s_equation_is_solved_to_its_conditioning` (G2) and
//! `the_conversions_match_the_element_map_at_60_digits` (G1) in `tests/keplerian.rs`, against
//! `derivations/keplerian_elements.gp`.

// The classical elements are single letters (a, e, i, r, v, h, n, m, p, w) in every reference on
// the subject; they are the ubiquitous language of the domain, not terse local names.
#![allow(clippy::many_single_char_names)]

use spacelab_core::{Angle, Mat3, Vec3};

/// Below this eccentricity the orbit is treated as circular: the eccentricity vector has no
/// reliable direction, so the argument of periapsis is undefined and folded into the anomaly.
const CIRCULAR: f64 = 1e-11;
/// Below this `sin i` the orbit is treated as equatorial: the node vector vanishes, so the right
/// ascension of the ascending node is undefined and folded into the following angle.
const EQUATORIAL: f64 = 1e-11;

/// The classical orbital elements of an osculating Keplerian orbit.
///
/// "Osculating" means these are the elements of the two-body orbit that instantaneously matches the
/// true state — position, velocity, and their first derivative — at one epoch. Under any
/// perturbation they change with time; under pure two-body motion all but the anomaly are constant.
///
/// This is deliberately **not** the mean-element set an OMM/TLE carries. Those are Brouwer-mean
/// elements fitted for SGP4 and must be propagated by SGP4, never fed to [`to_state`]; converting
/// them as if osculating is a kilometre-scale error. Keeping the two types distinct keeps that
/// mistake unrepresentable.
///
/// [`to_state`]: KeplerianElements::to_state
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerianElements {
    /// Semi-major axis `a`, metres.
    pub semi_major_axis: f64,
    /// Eccentricity `e`, dimensionless, in `[0, 1)`.
    pub eccentricity: f64,
    /// Inclination `i`, in `[0, π]`.
    pub inclination: Angle,
    /// Right ascension of the ascending node `Ω`.
    pub raan: Angle,
    /// Argument of periapsis `ω`.
    pub arg_periapsis: Angle,
    /// True anomaly `ν`.
    pub true_anomaly: Angle,
    /// Gravitational parameter `μ`, m³/s² — the field these elements were computed in.
    pub mu: f64,
}

/// Wrap an angle into `[0, 2π)`. `f64::rem_euclid` is a `std` method; the `%` operator (`fmod`) is
/// available in `no_std`, so the reduction is spelled out.
#[inline]
fn wrap_tau(x: f64) -> f64 {
    let r = x % Angle::TAU;
    if r < 0.0 { r + Angle::TAU } else { r }
}

/// Kepler's equation `M = E − e sin E` solved for `E` by Newton's method — quadratically convergent,
/// started at `M` (or `π` for high eccentricity, where `M` is a poor guess).
///
/// Returns the root and the number of steps taken. Newton is self-correcting on a monotone `f`: a
/// different guess or a different stopping rule still lands on the same root, so the returned angle
/// records nothing about the path. The count is the only place a difference between them survives.
fn eccentric_from_mean(m: f64, e: f64) -> (f64, u32) {
    let mut big_e = if e < 0.8 { m } else { core::f64::consts::PI };
    let mut steps = 0;
    // Newton: E ← E − (E − e sin E − M)/(1 − e cos E). ~1e-15 in a handful of steps.
    for _ in 0..64 {
        let (sin_e, cos_e) = libm::sincos(big_e);
        let f = big_e - e * sin_e - m;
        let step = f / (1.0 - e * cos_e);
        big_e -= step;
        steps += 1;
        if libm::fabs(step) < 1e-15 {
            break;
        }
    }
    (big_e, steps)
}

impl KeplerianElements {
    /// The osculating elements of the state `(r, v)` in a field of parameter `μ`.
    ///
    /// Vallado's RV2COE. The singular cases are handled so the round-trip through `to_state`
    /// reconstructs the state exactly:
    ///
    /// * **circular** (`e ≈ 0`): `ω` is undefined; it is set to zero and the argument of latitude is
    ///   carried in `ν`.
    /// * **equatorial** (`i ≈ 0` *or* `i ≈ π`): `Ω` is undefined; it is set to zero and the
    ///   longitude of periapsis is carried in `ω`, measured in the direction of motion — clockwise
    ///   seen from `+ẑ` on the retrograde branch.
    /// * **circular equatorial**: both are zero and the true longitude is carried in `ν`, with the
    ///   same sense convention.
    ///
    /// A state with `r ∥ v` is outside all three: it has no orbital plane at all, and every angle
    /// comes back `NaN`. `a` and `e` are still the energy and the eccentricity vector.
    #[must_use]
    pub fn from_state(r: Vec3, v: Vec3, mu: f64) -> Self {
        let r_mag = r.norm();
        let v2 = v.dot(v);

        let h = r.cross(v);
        let h_mag = h.norm();
        // node vector n = ẑ × h = (−h_y, h_x, 0); its length is h·sin i.
        let node = Vec3::new(-h.y, h.x, 0.0);
        let n_mag = node.norm();

        // eccentricity vector e = [(v² − μ/r) r − (r·v) v] / μ
        let rv = r.dot(v);
        let e_vec = (r.scale(v2 - mu / r_mag) - v.scale(rv)).scale(1.0 / mu);
        let e = e_vec.norm();

        let energy = v2 / 2.0 - mu / r_mag;
        let a = -mu / (2.0 * energy);

        // `r ∥ v` spans no plane. `a` and `e` survive that — the energy still fixes the semi-major
        // axis, and the eccentricity vector reduces to −r̂, the rectilinear ellipse's `e = 1` — but
        // the orientation does not: a line lies in every plane containing it, so `i` and `Ω` are not
        // small here, they are undefined. The branches below would answer `0` for both and report a
        // fall straight down the pole as an equatorial orbit, at three angles that no longer
        // round-trip: `p = a(1 − e²)` is zero, so `to_state` returns `NaN`. Say so here instead.
        //
        // The same guard catches `h_mag` overflowing to infinity, where `equatorial` degenerates the
        // other way — `∞ ≤ EQUATORIAL·∞` is true — and every numerator below is infinite.
        if !(h_mag.is_finite() && h_mag > 0.0) {
            return Self {
                semi_major_axis: a,
                eccentricity: e,
                inclination: Angle::from_radians(f64::NAN),
                raan: Angle::from_radians(f64::NAN),
                arg_periapsis: Angle::from_radians(f64::NAN),
                true_anomaly: Angle::from_radians(f64::NAN),
                mu,
            };
        }

        // Every angle is a two-argument arctangent of a sine and a cosine numerator, never an `acos`
        // of the cosine alone. `acos` loses half the significant digits as its argument approaches
        // ±1 — which is perigee, apogee, and the node, the three places an orbit is most often
        // sampled: the state round-trips to 1.5e-8 relative there against 2e-16 here. Each pair of
        // numerators below shares one positive scale factor, so dropping it leaves the quadrant
        // intact and no sign test is needed to recover it.
        let inclination = libm::atan2(n_mag, h.z);
        let equatorial = n_mag <= EQUATORIAL * h_mag;
        let circular = e <= CIRCULAR;

        // An equatorial orbit has no node, so the substitute angles below are measured from x̂ in the
        // equatorial plane — and *in the direction the satellite travels*, which for `i ≈ π` is
        // clockwise seen from `+ẑ`. `to_state` composes `R3(−Ω) R1(−i) R3(−ω)`, and at `i = π` the
        // `R1` flips the sense of the plane, so an angle measured anticlockwise here comes back as
        // its own negative. Without this factor the retrograde equatorial round trip returns the
        // state reflected in y — not a tolerance failure, a mirrored orbit.
        let sense = if h.z < 0.0 { -1.0 } else { 1.0 };

        // Ω: angle of the node in the equatorial plane, measured from x̂.
        let raan = if equatorial {
            0.0
        } else {
            libm::atan2(node.y, node.x)
        };

        // ω and ν, routing the geometry through whichever angle stays defined:
        // `sin ω = e_z/(e sin i)`, `cos ω = (n·e)/(n e)`; `sin ν = (r·v) h/(μ e r)`,
        // `cos ν = (e·r)/(e r)`; `sin u = r_z/(r sin i)`, `cos u = (n·r)/(n r)`, and `sin i = n/h`.
        let (arg_periapsis, true_anomaly) = match (equatorial, circular) {
            (false, false) => {
                // general case
                (
                    libm::atan2(e_vec.z * h_mag, node.dot(e_vec)),
                    libm::atan2(rv * h_mag / mu, e_vec.dot(r)),
                )
            }
            (false, true) => {
                // circular inclined: ω = 0, ν = argument of latitude u (node → satellite)
                (0.0, libm::atan2(r.z * h_mag, node.dot(r)))
            }
            (true, false) => {
                // elliptical equatorial: Ω = 0, ω = longitude of periapsis ϖ (x̂ → periapsis)
                (
                    libm::atan2(sense * e_vec.y, e_vec.x),
                    libm::atan2(rv * h_mag / mu, e_vec.dot(r)),
                )
            }
            (true, true) => {
                // circular equatorial: Ω = ω = 0, ν = true longitude λ (x̂ → satellite)
                (0.0, libm::atan2(sense * r.y, r.x))
            }
        };

        Self {
            semi_major_axis: a,
            eccentricity: e,
            inclination: Angle::from_radians(inclination),
            raan: Angle::from_radians(wrap_tau(raan)),
            arg_periapsis: Angle::from_radians(wrap_tau(arg_periapsis)),
            true_anomaly: Angle::from_radians(wrap_tau(true_anomaly)),
            mu,
        }
    }

    /// The inertial state `(r, v)` these elements represent.
    ///
    /// Builds the state in the perifocal frame and rotates it to inertial by
    /// `R3(−Ω) R1(−i) R3(−ω)`. With `ω = 0` (circular) the perifocal x-axis sits on the node, and
    /// with `Ω = 0` too (equatorial) it sits on x̂ — which is exactly what makes the singular-case
    /// choices in `from_state` round-trip.
    #[must_use]
    pub fn to_state(&self) -> (Vec3, Vec3) {
        let e = self.eccentricity;
        let nu = self.true_anomaly.radians();
        let p = self.semi_latus_rectum();

        let (sin_nu, cos_nu) = libm::sincos(nu);
        let r_pf = p / (1.0 + e * cos_nu);
        let pos_pf = Vec3::new(r_pf * cos_nu, r_pf * sin_nu, 0.0);
        let speed = libm::sqrt(self.mu / p);
        let vel_pf = Vec3::new(-speed * sin_nu, speed * (e + cos_nu), 0.0);

        let rot = Mat3::rot3(-self.raan.radians())
            * Mat3::rot1(-self.inclination.radians())
            * Mat3::rot3(-self.arg_periapsis.radians());
        (rot * pos_pf, rot * vel_pf)
    }

    /// Semi-latus rectum `p = a(1 − e²)`.
    #[must_use]
    pub fn semi_latus_rectum(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity * self.eccentricity)
    }

    /// Periapsis radius `r_p = a(1 − e)`.
    #[must_use]
    pub fn periapsis_radius(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity)
    }

    /// Apoapsis radius `r_a = a(1 + e)`.
    #[must_use]
    pub fn apoapsis_radius(&self) -> f64 {
        self.semi_major_axis * (1.0 + self.eccentricity)
    }

    /// Mean motion `n = √(μ / a³)`, rad/s — read against the `μ` frozen into the set.
    #[must_use]
    pub fn mean_motion(&self) -> f64 {
        libm::sqrt(self.mu / (self.semi_major_axis * self.semi_major_axis * self.semi_major_axis))
    }

    /// Orbital period `T = 2π / n`, seconds.
    #[must_use]
    pub fn period(&self) -> f64 {
        Angle::TAU / self.mean_motion()
    }

    /// First-order secular nodal regression rate `Ω̇ = −³⁄₂ n J₂ (Rₑ/p)² cos i`, rad/s — the `J₂`
    /// effect that Sun-synchronous and repeat-ground-track design are built on. Prograde orbits
    /// (`i < 90°`) regress (`Ω̇ < 0`); retrograde ones advance.
    #[must_use]
    pub fn nodal_regression_rate(&self, re: f64, j2: f64) -> f64 {
        let p = self.semi_latus_rectum();
        -1.5 * self.mean_motion() * j2 * (re / p) * (re / p) * libm::cos(self.inclination.radians())
    }

    /// First-order secular apsidal precession rate `ω̇ = ³⁄₄ n J₂ (Rₑ/p)² (5 cos²i − 1)`, rad/s. Zero
    /// at the critical inclination `i = 63.43°`, where the perigee neither advances nor regresses.
    #[must_use]
    pub fn apsidal_precession_rate(&self, re: f64, j2: f64) -> f64 {
        let p = self.semi_latus_rectum();
        let cos_i = libm::cos(self.inclination.radians());
        0.75 * self.mean_motion() * j2 * (re / p) * (re / p) * (5.0 * cos_i * cos_i - 1.0)
    }

    /// Anomalistic mean motion `Ṁ = n [1 + ³⁄₄ J₂ (Rₑ/p)² √(1−e²)(3 cos²i − 1)]`, rad/s — the rate the
    /// mean anomaly advances, so `2π/Ṁ` is the perigee-to-perigee period.
    #[must_use]
    pub fn anomalistic_mean_motion(&self, re: f64, j2: f64) -> f64 {
        let n = self.mean_motion();
        let p = self.semi_latus_rectum();
        let cos_i = libm::cos(self.inclination.radians());
        let e = self.eccentricity;
        n * (1.0
            + 0.75
                * j2
                * (re / p)
                * (re / p)
                * libm::sqrt(1.0 - e * e)
                * (3.0 * cos_i * cos_i - 1.0))
    }

    /// Draconitic (nodal) mean motion `Ṁ + ω̇`, rad/s — the rate of successive ascending-node
    /// passages, whose `2π` is the nodal period the repeat-ground-track condition counts.
    #[must_use]
    pub fn draconitic_mean_motion(&self, re: f64, j2: f64) -> f64 {
        self.anomalistic_mean_motion(re, j2) + self.apsidal_precession_rate(re, j2)
    }

    /// Eccentric anomaly `E`, from the true anomaly: `E = atan2(√(1−e²) sin ν, e + cos ν)`.
    #[must_use]
    pub fn eccentric_anomaly(&self) -> Angle {
        let e = self.eccentricity;
        let nu = self.true_anomaly.radians();
        let (sin_nu, cos_nu) = libm::sincos(nu);
        let big_e = libm::atan2(libm::sqrt(1.0 - e * e) * sin_nu, e + cos_nu);
        Angle::from_radians(wrap_tau(big_e))
    }

    /// Mean anomaly `M = E − e sin E`, Kepler's equation forward.
    #[must_use]
    pub fn mean_anomaly(&self) -> Angle {
        let big_e = self.eccentric_anomaly().radians();
        let m = big_e - self.eccentricity * libm::sin(big_e);
        Angle::from_radians(wrap_tau(m))
    }

    /// The same orbit with its anomaly advanced to mean anomaly `M`.
    ///
    /// Solves Kepler's equation `M = E − e sin E` for `E` by Newton's method — quadratically
    /// convergent, started at `M` (or `π` for high eccentricity, where `M` is a poor guess) — then
    /// maps `E` back to the true anomaly. This is the one place the elements need an iterative
    /// solve; every other conversion here is closed-form.
    #[must_use]
    pub fn at_mean_anomaly(&self, m: Angle) -> Self {
        let e = self.eccentricity;
        let m = wrap_tau(m.radians());
        let (big_e, _) = eccentric_from_mean(m, e);
        let (sin_e, cos_e) = libm::sincos(big_e);
        let nu = libm::atan2(libm::sqrt(1.0 - e * e) * sin_e, cos_e - e);
        Self {
            true_anomaly: Angle::from_radians(wrap_tau(nu)),
            ..*self
        }
    }

    /// Two-body propagation by `dt` seconds in closed form: advance the mean anomaly at the mean
    /// motion, `M(t₀+dt) = M(t₀) + n·dt`, and solve back to the true anomaly.
    ///
    /// This is the analytic Kepler flow; it agrees with numerically integrating the two-body ODE to
    /// the integrator's tolerance, which is how the two are cross-checked.
    #[must_use]
    pub fn propagate(&self, dt: f64) -> Self {
        let m = self.mean_anomaly().radians() + self.mean_motion() * dt;
        self.at_mean_anomaly(Angle::from_radians(m))
    }
}

#[cfg(test)]
mod tests {
    // Goldens are kept at the precision the CAS produced them; they are not f64 literals typed
    // by hand, and truncating them would be discarding the oracle's digits.
    #![allow(clippy::unwrap_used, clippy::excessive_precision)]
    use super::*;
    use spacelab_testkit::nan_max;

    /// `wrap_tau` has two obligations and every anomaly here leans on both: land in `[0, 2π)`, and
    /// move the angle by a whole number of turns. Neither is visible downstream, where the wrapped
    /// value is only ever fed to `sin`/`cos` or differenced modulo a turn.
    #[test]
    fn wrap_tau_lands_in_the_turn_a_whole_number_of_turns_away() {
        for k in -9..=9 {
            for &frac in &[0.0, 0.25, 0.5, 0.999_999] {
                let x = Angle::TAU * (f64::from(k) + frac);
                let got = wrap_tau(x);
                assert!(
                    (0.0..Angle::TAU).contains(&got),
                    "x={x}: {got} is outside [0, 2π)"
                );
                let turns = (got - x) / Angle::TAU;
                assert!(
                    libm::fabs(turns - libm::round(turns)) < 1e-12,
                    "x={x}: moved {turns} turns"
                );
            }
        }
    }

    /// The Newton solve behind [`KeplerianElements::at_mean_anomaly`]. The root is all the elements
    /// report, and Newton reaches the same root from a different guess or under a different stopping
    /// rule, so neither is observable in a propagated state. The residual pins the root — written
    /// from Kepler's equation, not from the solver — and the two-sided budget pins the path to it. A
    /// budget with only a ceiling is satisfied by a loop that stops at once; here the ceiling is the
    /// `for` bound itself, so the mean and the stall count carry the statement.
    ///
    /// The grid straddles the `e = 0.8` guess switch on both sides and on the value itself.
    ///
    /// The stall is real and benign. The loop exits on the *absolute* Newton step, whose round-off
    /// floor is `ulp(E)/(1 − e cos E)`; approaching perigee at `e → 1` that floor rises above the
    /// `10⁻¹⁵` threshold, so 9 of these 2169 calls oscillate on it and run out the 64-step cap. They
    /// land on the root anyway: over a 1446-sample `(e ≤ 0.999, M)` grid the true anomaly this solve
    /// produces is within `5.6e-15` rad of an 80-digit bracketed solve of the same equation.
    #[test]
    fn the_kepler_newton_reaches_its_root_within_a_measured_number_of_steps() {
        let (mut least, mut worst, mut total, mut calls, mut stalled) =
            (u32::MAX, 0, 0u32, 0u32, 0);
        let mut residual = 0.0;
        for &e in &[0.0, 0.05, 0.3, 0.79, 0.8, 0.81, 0.95, 0.99, 0.999] {
            for k in 0..241 {
                let m = Angle::TAU * f64::from(k) / 241.0;
                let (big_e, steps) = eccentric_from_mean(m, e);
                residual = nan_max(residual, libm::fabs(big_e - e * libm::sin(big_e) - m));
                least = least.min(steps);
                worst = worst.max(steps);
                total += steps;
                calls += 1;
                stalled += u32::from(steps == 64);
            }
        }
        let mean = f64::from(total) / f64::from(calls);
        std::eprintln!(
            "Kepler Newton: {least}..{worst} steps, mean {mean:.3}, {stalled}/{calls} stalled, \
             residual {residual:.3e}"
        );
        assert!(residual < 4e-15, "worst Kepler residual {residual:.3e}");
        // e = 0 is solved by the guess itself: the first step is exactly zero.
        assert_eq!(least, 1, "the shortest solve took {least} steps");
        assert!(
            stalled <= 9,
            "{stalled} of {calls} calls ran out the step cap"
        );
        assert!((5.05..5.07).contains(&mean), "mean {mean:.3} steps");
    }

    // Ṁ from derivations/j2_secular_rates.mac, which averages the J₂ disturbing function over one
    // revolution and pushes it through Lagrange's planetary equations; 40 significant digits.
    // Columns: a (m), e, i (deg), Ṁ (rad/s). No row sits at i = 60°, where the Ṁ and ω̇ corrections
    // cancel and the node-crossing test in tests/design.rs cannot see this formula at all.
    const GOLDEN_MDOT: [(f64, f64, f64, f64); 5] = [
        (
            7_278_137.0,
            0.0,
            20.0,
            0.001_017_852_645_556_038_255_251_011_321_160_569_240_749,
        ),
        (
            7_278_137.0,
            0.3,
            20.0,
            0.001_018_011_535_771_300_884_902_022_012_858_880_909_886,
        ),
        (
            7_278_137.0,
            0.3,
            98.0,
            0.001_016_119_086_446_529_027_605_828_724_849_997_618_784,
        ),
        (
            26_560_000.0,
            0.72,
            116.5,
            0.000_145_848_604_271_408_341_272_424_860_601_491_824_479_2,
        ),
        (
            42_164_000.0,
            0.001,
            0.5,
            0.000_072_924_308_068_845_662_414_156_666_372_023_563_052_88,
        ),
    ];

    /// (G1) The anomalistic mean motion against the closed form `derivations/j2_secular_rates.mac`
    /// derives from the averaged disturbing function, evaluated there at 40 digits.
    ///
    /// The `√(1−e²)` factor in the anomalistic mean motion is invisible to every numerical oracle in
    /// this crate: the trajectory-based check runs at the inclination where the `Ṁ` and `ω̇`
    /// corrections cancel, and even off it the O(J₂) osculating-versus-mean floor of a propagated
    /// orbit is two decades above what the factor is worth. The CAS derivation is the oracle instead.
    #[test]
    fn the_anomalistic_mean_motion_matches_the_averaged_disturbing_function() {
        const MU: f64 = 3.986_004_418e14;
        const RE: f64 = 6_378_137.0;
        const J2: f64 = 1.082_626_68e-3;
        let mut worst = 0.0;
        for (a, e, i_deg, want) in GOLDEN_MDOT {
            let els = KeplerianElements {
                semi_major_axis: a,
                eccentricity: e,
                inclination: Angle::from_degrees(i_deg),
                raan: Angle::from_radians(0.0),
                arg_periapsis: Angle::from_radians(0.0),
                true_anomaly: Angle::from_radians(0.0),
                mu: MU,
            };
            let got = els.anomalistic_mean_motion(RE, J2);
            let rel = libm::fabs(got - want) / want;
            assert!(rel < 5e-15, "a={a}, e={e}, i={i_deg}°: Ṁ off by {rel:.3e}");
            worst = nan_max(worst, rel);
        }
        std::eprintln!("anomalistic mean motion vs CAS: worst rel {worst:.3e}");
    }
}
