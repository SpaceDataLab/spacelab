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

/// Above this eccentricity Kepler's equation is evaluated as `(1 − e)·sin E + (E − sin E)`, and the
/// eccentric anomaly from the true one by half angles. Near perigee `E` and `e·sin E` agree to about
/// `1 − e` of their size, and their difference loses those digits: 180 u in `E` at `e = 0.99`, 1.8e-10
/// relative at `1 − 1e-6` (C3 row 12). `1 − e` is exact there (Sterbenz) and `E − sin E` has a series
/// with no cancellation, so the rearranged form loses nothing.
const NEAR_PARABOLIC: f64 = 0.99;

/// `x − sin x`. For `|x| ≤ 1` its alternating series `x³/3! − x⁵/5! + … + x²¹/21!`, nested as
/// `(x³/6)(1 − x²/(4·5)(1 − x²/(6·7)(… (1 − x²/(20·21)))))`; the first term left out, `x²³/23!`, is
/// under `2.5e-22` of the sum (`derivations/keplerian_elements.mac` 6c). Above, the plain difference,
/// which keeps its digits there: `|sin x|` is under `0.85·|x|`.
fn x_minus_sin(x: f64) -> f64 {
    if libm::fabs(x) <= 1.0 {
        let x2 = x * x;
        let mut q = 1.0;
        for n in [20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0] {
            q = 1.0 - x2 * q / (n * (n + 1.0));
        }
        x * x2 / 6.0 * q
    } else {
        x - libm::sin(x)
    }
}

/// Kepler's equation forward, `M = E − e·sin E`; above [`NEAR_PARABOLIC`] as
/// `(1 − e)·sin E + (E − sin E)`, the same function (`derivations/keplerian_elements.mac` 6b).
fn mean_from_eccentric(big_e: f64, e: f64) -> f64 {
    if e > NEAR_PARABOLIC {
        (1.0 - e) * libm::sin(big_e) + x_minus_sin(big_e)
    } else {
        big_e - e * libm::sin(big_e)
    }
}

/// Kepler's equation `M = E − e sin E` solved for `E` by Newton's method — quadratically convergent,
/// started at `M` (or `π` for high eccentricity, where `M` is a poor guess).
///
/// Returns the root and the number of steps taken. Newton is self-correcting on a monotone `f`: a
/// different guess or a different stopping rule still lands on the same root, so the returned angle
/// records nothing about the path. The count is the only place a difference between them survives.
///
/// The loop stops on a step under `1e-15`, an absolute size, and its round-off floor is
/// `ulp(E)/(1 − e cos E)`. Two things kept it there:
///
/// * `M` is folded into `[0, π]` by `E(2π − M) = 2π − E(M)` (`keplerian_elements.mac` 6a), solved,
///   and unfolded. Unfolded, `E ≥ 4` has an ulp of 8.9e-16, so the floor passed the stop wherever
///   `1 − e cos E < 0.89`, which every `e > 0.11` reaches: about half a percent of the calls with
///   `M` in `(π, 2π)` ran out the cap (M0-10). `2π − M` is exact for `M ∈ [π, 2π]` by Sterbenz.
/// * Near perigee above `e = 0.99`, `1 − e cos E` falls to `1 − e` and the floor passes the stop
///   on any `E`. There the residual and its slope are the rearranged forms of
///   [`mean_from_eccentric`] and `(1 − e) + 2e·sin²(E/2)` (6b), and the loop also ends on a step
///   below `1e-9` that is no smaller than the one before, without taking it. In exact arithmetic
///   Newton's next step is at most `C·step²` with `C ≤ e/(2√(1 − e²))` (6d), under `3.4e7` for every
///   double `e < 1`, so below `1e-9` an honest step at least halves: one that does not is round-off.
fn eccentric_from_mean(m: f64, e: f64) -> (f64, u32) {
    let upper = m > core::f64::consts::PI;
    let m = if upper { Angle::TAU - m } else { m };
    let mut big_e = if e < 0.8 { m } else { core::f64::consts::PI };
    let mut steps = 0;
    let mut last = f64::INFINITY;
    // Newton: E ← E − (E − e sin E − M)/(1 − e cos E). ~1e-15 in a handful of steps.
    for _ in 0..64 {
        let (f, slope) = if e > NEAR_PARABOLIC {
            let half = libm::sin(0.5 * big_e);
            (
                mean_from_eccentric(big_e, e) - m,
                (1.0 - e) + 2.0 * e * half * half,
            )
        } else {
            let (sin_e, cos_e) = libm::sincos(big_e);
            (big_e - e * sin_e - m, 1.0 - e * cos_e)
        };
        let step = f / slope;
        let size = libm::fabs(step);
        if steps >= 2 && size >= last && size < 1e-9 {
            break;
        }
        big_e -= step;
        steps += 1;
        last = size;
        if size < 1e-15 {
            break;
        }
    }
    (if upper { Angle::TAU - big_e } else { big_e }, steps)
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

    /// Eccentric anomaly `E`, from the true anomaly: `E = atan2(√(1−e²) sin ν, e + cos ν)`, and
    /// above `e = 0.99` its half-angle form `E = 2·atan2(√(1 − e)·sin(ν/2), √(1 + e)·cos(ν/2))`
    /// (`derivations/keplerian_elements.mac` 6e). Near apoapsis `e + cos ν` cancels to `e − 1`; the
    /// half angles keep `1 − e` as a factor, which is exact.
    #[must_use]
    pub fn eccentric_anomaly(&self) -> Angle {
        let e = self.eccentricity;
        let nu = self.true_anomaly.radians();
        let big_e = if e > NEAR_PARABOLIC {
            let (sin_half, cos_half) = libm::sincos(0.5 * nu);
            2.0 * libm::atan2(
                libm::sqrt(1.0 - e) * sin_half,
                libm::sqrt(1.0 + e) * cos_half,
            )
        } else {
            let (sin_nu, cos_nu) = libm::sincos(nu);
            libm::atan2(libm::sqrt(1.0 - e * e) * sin_nu, e + cos_nu)
        };
        Angle::from_radians(wrap_tau(big_e))
    }

    /// Mean anomaly `M = E − e sin E`, Kepler's equation forward, in the rearranged form above
    /// `e = 0.99` that [`at_mean_anomaly`](Self::at_mean_anomaly) solves.
    #[must_use]
    pub fn mean_anomaly(&self) -> Angle {
        let big_e = self.eccentric_anomaly().radians();
        let m = mean_from_eccentric(big_e, self.eccentricity);
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
    // by hand, and truncating them would be discarding the oracle's digits. The decade table's rows
    // near 2π are the CAS's values, not a hand-typed `TAU`.
    #![allow(
        clippy::unwrap_used,
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::approx_constant
    )]
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
    /// 9 of these 2169 calls once ran out the 64-step cap, oscillating on the round-off floor
    /// `ulp(E)/(1 − e cos E)` of the loop's absolute stop, a step under `10⁻¹⁵`, for two causes.
    /// Approaching perigee at `e → 1`, `1 − e cos E` falls towards `1 − e` and lifts the floor over
    /// the stop at any `E`. And with `M` in `(π, 2π)` the root has `E ≥ 4`, whose ulp is 8.9e-16, so
    /// the floor passes `10⁻¹⁵` wherever `1 − e cos E < 0.89`, at modest eccentricities too: all 9
    /// stalls had `M > 5.6`, among them `e = 0.79`, `0.8` and `0.81`. Folding `M` into `[0, π]` keeps
    /// the second floor under the stop; the rearranged residual above `e = 0.99` and the stop on a
    /// step that no longer shrinks end the first. None stalls now, and the most any takes is 14.
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
        assert_eq!(
            stalled, 0,
            "{stalled} of {calls} calls ran out the step cap"
        );
        assert!(worst <= 14, "the longest solve took {worst} steps");
        assert!((4.81..4.82).contains(&mean), "mean {mean:.3} steps");
    }

    /// (S2) The stop on a step that no longer shrinks ends what the fold leaves. Just below the
    /// rearranged residual's threshold the classic one still has a round-off floor above `10⁻¹⁵`
    /// near perigee — `ulp(E)/(1 − e cos E)` is 1.4e-15 at `e = 0.99`, `E = 1/4` — and these four
    /// calls, the only ones of 4 000 000 uniform draws with `e ∈ [0.5, 1)` (a 64-bit LCG, seed
    /// 20 261 115) that still ran out the 64-step cap with `M` folded, oscillate on it. With the floor
    /// stop each settles in 11 or 12 steps, on a root that gives back its `M` to an ulp of `2π`.
    #[test]
    fn the_floor_stop_ends_what_the_fold_leaves() {
        for (e, m) in [
            (0.988_381_326_727_344, 6.281_100_217_002_651),
            (0.983_117_975_118_906_1, 6.275_407_458_850_654),
            (0.988_401_725_203_388_8, 6.280_874_462_468_717),
            (0.985_046_749_280_648_1, 0.009_233_568_790_492_28),
        ] {
            let (big_e, steps) = eccentric_from_mean(m, e);
            let back = turn_distance(mean_from_eccentric(big_e, e), m);
            std::println!("e {e}, M {m}: {steps} steps, residual {back:.3e}");
            assert!(steps <= 12, "e {e}, M {m}: {steps} steps");
            assert!(
                back <= 4.0 * f64::EPSILON,
                "e {e}, M {m}: residual {back:.3e}"
            );
        }
    }

    /// (S2) The solve settles on every ellipse up to `e = 1 − 1e-6` and every mean anomaly, and the
    /// root it returns gives back its `M`.
    ///
    /// 136 eccentricities — every hundredth from 0 to 0.99, and `1 − j·10⁻ᵏ` for `j = 1…9`,
    /// `k = 3…6` — by 2006 mean anomalies: a grid of 2000 across the turn, offset by the golden
    /// ratio's fraction so no node sits on a special angle, plus `0`, `1e-12`, `1e-6`, `π`,
    /// `2π − 1e-6` and `2π − 1e-12`. On the solver before the fold and the floor stop, 979 of these
    /// 272 816 calls ran out the 64-step cap. None does now; the longest takes 22 steps, `M` at or
    /// within 1e-12 of perigee at `e = 1 − 1e-6`, where Newton from the guess `π` walks the cubic
    /// `E³/6` down by thirds before its quadratic phase begins. Every root gives back its `M` through
    /// [`mean_from_eccentric`] to two ulps of `2π`.
    #[test]
    fn kepler_s_equation_settles_on_every_ellipse_and_anomaly() {
        let mut eccs: std::vec::Vec<f64> = (0..100).map(|k| f64::from(k) / 100.0).collect();
        for decade in [1e-3, 1e-4, 1e-5, 1e-6] {
            for j in 1..=9 {
                eccs.push(1.0 - f64::from(j) * decade);
            }
        }
        let n = 2000;
        let mut means: std::vec::Vec<f64> = (0..n)
            .map(|k| Angle::TAU * (f64::from(k) + 0.618_033_988_749_894_8) / f64::from(n))
            .collect();
        means.extend([
            0.0,
            1e-12,
            1e-6,
            core::f64::consts::PI,
            Angle::TAU - 1e-6,
            Angle::TAU - 1e-12,
        ]);
        let (mut most, mut calls, mut residual) = (0, 0u32, 0.0_f64);
        for &e in &eccs {
            for &m in &means {
                let (big_e, steps) = eccentric_from_mean(m, e);
                most = most.max(steps);
                calls += 1;
                residual = nan_max(residual, turn_distance(mean_from_eccentric(big_e, e), m));
            }
        }
        std::println!("Kepler sweep: {calls} calls, at most {most} steps, residual {residual:.3e}");
        assert_eq!(calls, 272_816);
        assert!(most <= 22, "a solve took {most} steps");
        assert!(residual <= 8.0 * f64::EPSILON, "residual {residual:.3e}");
    }

    /// `(e, x, E(M = x), M(E = x), E(ν = x))` from `derivations/keplerian_elements.gp` §3, at 100
    /// digits: eleven eccentricities from 0 to the double nearest `1 − 1e-12`, seventeen anomalies
    /// from 1e-12 at perigee through apoapsis to `2π − 1e-6`.
    #[rustfmt::skip]
    const DECADES: [(f64, f64, f64, f64, f64); 187] = [
        (0.0, 9.999999999999999798866476e-13, 9.999999999999999798866476e-13, 9.999999999999999798866476e-13, 9.999999999999999798866476e-13),
        (0.0, 1.000000000000000062281591e-9, 1.000000000000000062281591e-9, 1.000000000000000062281591e-9, 1.000000000000000062281591e-9),
        (0.0, 9.999999999999999547481118e-7, 9.999999999999999547481118e-7, 9.999999999999999547481118e-7, 9.999999999999999547481118e-7),
        (0.0, 0.0001000000000000000047921736, 0.0001000000000000000047921736, 0.0001000000000000000047921736, 0.0001000000000000000047921736),
        (0.0, 0.001000000000000000020816682, 0.001000000000000000020816682, 0.001000000000000000020816682, 0.001000000000000000020816682),
        (0.0, 0.01000000000000000020816682, 0.01000000000000000020816682, 0.01000000000000000020816682, 0.01000000000000000020816682),
        (0.0, 0.1000000000000000055511151, 0.1000000000000000055511151, 0.1000000000000000055511151, 0.1000000000000000055511151),
        (0.0, 0.5000000000000000000000000, 0.5000000000000000000000000, 0.5000000000000000000000000, 0.5000000000000000000000000),
        (0.0, 1.000000000000000000000000, 1.000000000000000000000000, 1.000000000000000000000000, 1.000000000000000000000000),
        (0.0, 2.000000000000000000000000, 2.000000000000000000000000, 2.000000000000000000000000, 2.000000000000000000000000),
        (0.0, 3.000000000000000000000000, 3.000000000000000000000000, 3.000000000000000000000000, 3.000000000000000000000000),
        (0.0, 3.141591653589793420309206, 3.141591653589793420309206, 3.141591653589793420309206, 3.141591653589793420309206),
        (0.0, 3.500000000000000000000000, 3.500000000000000000000000, 3.500000000000000000000000, 3.500000000000000000000000),
        (0.0, 5.000000000000000000000000, 5.000000000000000000000000, 5.000000000000000000000000, 5.000000000000000000000000),
        (0.0, 6.183185307179586587267295, 6.183185307179586587267295, 6.183185307179586587267295, 6.183185307179586587267295),
        (0.0, 6.282185307179586786219261, 6.282185307179586786219261, 6.282185307179586786219261, 6.282185307179586786219261),
        (0.0, 6.283184307179586092217960, 6.283184307179586092217960, 6.283184307179586092217960, 6.283184307179586092217960),
        (0.5000000000000000000000000, 9.999999999999999798866476e-13, 1.999999999999999959773294e-12, 4.999999999999999899433239e-13, 5.773502691896257528966994e-13),
        (0.5000000000000000000000000, 1.000000000000000062281591e-9, 2.000000000000000123229850e-9, 5.000000000000000312241291e-10, 5.773502691896258004995174e-10),
        (0.5000000000000000000000000, 9.999999999999999547481118e-7, 1.999999999998666576162893e-6, 5.000000000000833107073892e-7, 5.773502691896578133979139e-7),
        (0.5000000000000000000000000, 0.0001000000000000000047921736, 0.0001999999986666667055843462, 5.000000008333333568775348e-5, 5.773502695103759418869432e-5),
        (0.5000000000000000000000000, 0.001000000000000000020816682, 0.001999998666669600033074579, 0.0005000000833333291770751119, 0.0005773503012646423352576626),
        (0.5000000000000000000000000, 0.01000000000000000020816682, 0.01999866695991444169973720, 0.005000083332916667762817394, 0.005773534767071588201105105),
        (0.5000000000000000000000000, 0.1000000000000000055511151, 0.1986951717258994539697902, 0.05008329167658592663601669, 0.05776711797524048047999723),
        (0.5000000000000000000000000, 0.5000000000000000000000000, 0.8878622115708660240357015, 0.2602872306978984998633560, 0.2927349208849687398625995),
        (0.5000000000000000000000000, 1.000000000000000000000000, 1.498701133517848314057985, 0.5792645075960517466737488, 0.6110637027332448632257096),
        (0.5000000000000000000000000, 2.000000000000000000000000, 2.354242758222780914149219, 1.545351286587159152301990, 1.464712442519596403380908),
        (0.5000000000000000000000000, 3.000000000000000000000000, 3.047150774702394435198273, 2.929439995970066388949628, 2.897160747576054382162679),
        (0.5000000000000000000000000, 3.141591653589793420309206, 3.141591986923126693010558, 3.141591153589793511315821, 3.141590921538985984841511),
        (0.5000000000000000000000000, 3.500000000000000000000000, 3.381293824239091633553143, 3.675391613844809924060184, 3.749649583947418535703844),
        (0.5000000000000000000000000, 5.000000000000000000000000, 4.510186665492470084303061, 5.479462137331569234446577, 5.468806868040803745366622),
        (0.5000000000000000000000000, 6.183185307179586587267295, 6.084490135453687250269337, 6.233102015503000608525323, 6.225418189204346063467827),
        (0.5000000000000000000000000, 6.282185307179586786219261, 6.281185308512917495520556, 6.282685307096253302405684, 6.282607956878322013173037),
        (0.5000000000000000000000000, 6.283184307179586092217960, 6.283183307179585708843966, 6.283184807179586284488290, 6.283184729829317065156569),
        (0.9000000000000000222044605, 9.999999999999999798866476e-13, 1.000000000000000201931103e-11, 9.999999999999997578420442e-14, 2.294157338705617344820869e-13),
        (0.9000000000000000222044605, 1.000000000000000062281591e-9, 1.000000000000000134326196e-8, 9.999999999999998417369865e-11, 2.294157338705617534028952e-10),
        (0.9000000000000000222044605, 9.999999999999999547481118e-7, 9.999999998500001768609666e-6, 1.000000000001499732703507e-7, 2.294157338705798404833748e-7),
        (0.9000000000000000222044605, 0.0001000000000000000047921736, 0.0009999985000068252285632581, 1.000000014999999818377134e-5, 2.294157340516794348918796e-5),
        (0.9000000000000000222044605, 0.001000000000000000020816682, 0.009998500682086272127244950, 0.0001000001499999924798773993, 0.0002294157519823318755268337),
        (0.9000000000000000222044605, 0.01000000000000000020816682, 0.09856437752097706674545686, 0.001000149999250001584496950, 0.002294175450640901329924369),
        (0.9000000000000000222044605, 0.1000000000000000055511151, 0.6308435275631534993158328, 0.01014992501785466128719079, 0.02295970185248207924877812),
        (0.9000000000000000222044605, 0.5000000000000000000000000, 1.384412720202162603112589, 0.06851701525621728910865543, 0.1170251714238100638968644),
        (0.9000000000000000222044605, 1.000000000000000000000000, 1.862086686874532271834067, 0.2426761136728931253283387, 0.2493605701194347613484081),
        (0.9000000000000000222044605, 2.000000000000000000000000, 2.522365434000244892094955, 1.181632315856886453953123, 0.6863156677906752558674564),
        (0.9000000000000000222044605, 3.000000000000000000000000, 3.067037496630688558913037, 2.872991992746119496975836, 2.542004493231661424161351),
        (0.9000000000000000222044605, 3.141591653589793420309206, 3.141592127274003860475646, 3.141590753589793584121091, 3.141588294690850496977721),
        (0.9000000000000000222044605, 3.500000000000000000000000, 3.330761568928410759417552, 3.815704904920657871097284, 4.478328665555109125328387),
        (0.9000000000000000222044605, 5.000000000000000000000000, 4.210843490070336040730856, 5.863031847196824643296235, 5.943725716496051321425379),
        (0.9000000000000000222044605, 6.183185307179586587267295, 5.652341779616433401780270, 6.273035382161731827748493, 6.260225605327104424327282),
        (0.9000000000000000222044605, 6.282185307179586786219261, 6.273186806497503296555087, 6.283085307029586515377028, 6.282955891427604216011456),
        (0.9000000000000000222044605, 6.283184307179586092217960, 6.283175307179584129849799, 6.283185207179586438304576, 6.283185077763852518087522),
        (0.9899999999999999911182158, 9.999999999999999798866476e-13, 9.999999999999990915432279e-11, 1.000000000000000868065084e-14, 7.088812050083362028958943e-14),
        (0.9899999999999999911182158, 1.000000000000000062281591e-9, 9.999999999998341741031718e-8, 1.000000000000000966960011e-11, 7.088812050083362613628979e-11),
        (0.9899999999999999911182158, 9.999999999999999547481118e-7, 9.999998350000808240162878e-5, 1.000000000016500842926531e-8, 7.088812050083949616579836e-8),
        (0.9899999999999999911182158, 0.0001000000000000000047921736, 0.009983581221411523283641093, 1.000000165000000853600178e-6, 7.088812055961020749561432e-6),
        (0.9899999999999999911182158, 0.001000000000000000020816682, 0.08854859633018195792511722, 1.000016499999175909015627e-5, 7.088812637849243900770867e-5),
        (0.9899999999999999911182158, 0.01000000000000000020816682, 0.3422703164917751040067935, 0.0001001649991750020551913201, 0.0007088870827249028401100787),
        (0.9899999999999999911182158, 0.1000000000000000055511151, 0.8316604237910567594717854, 0.001164917519640130185919086, 0.007094695547518510154708969),
        (0.9899999999999999911182158, 0.5000000000000000000000000, 1.486483282761429479893488, 0.02536871678183903398759912, 0.03619746491273219219112689),
        (0.9899999999999999911182158, 1.000000000000000000000000, 1.927635550695834916937208, 0.1669437250401824658877864, 0.07741402885963007798206066),
        (0.9899999999999999911182158, 2.000000000000000000000000, 2.551156310065828151547966, 1.099795547442575129634124, 0.2199128266842687450905602),
        (0.9899999999999999911182158, 3.000000000000000000000000, 3.070410669117501748561458, 2.860291192020731451373660, 1.570419412228413667941184),
        (0.9899999999999999911182158, 3.141591653589793420309206, 3.141592151077230515761950, 3.141590663589793600502313, 3.141578546853816370607110),
        (0.9899999999999999911182158, 3.500000000000000000000000, 3.322184394558885540552052, 3.847275395412723646523584, 5.537166487107691054655647),
        (0.9899999999999999911182158, 5.000000000000000000000000, 4.158184058596671431001635, 5.949335031916507075687264, 6.177374126471282456451193),
        (0.9899999999999999911182158, 6.183185307179586587267295, 5.451524883388530065390054, 6.282020389659946348471492, 6.276090611632067975006493),
        (0.9899999999999999911182158, 6.282185307179586786219261, 6.194636710849426806035186, 6.283175307014586488259498, 6.283114419053208006413029),
        (0.9899999999999999911182158, 6.283184307179586092217960, 6.283085307196047998124722, 6.283185297179586472913205, 6.283185236291465948814608),
        (0.9989999999999999991118216, 9.999999999999999798866476e-13, 9.999999999999989252082279e-10, 1.000000000000000868065234e-15, 2.236627204212923119839021e-14),
        (0.9989999999999999991118216, 1.000000000000000062281591e-9, 9.999999998334991741863474e-7, 1.000000000000001116960011e-12, 2.236627204212923304312086e-11),
        (0.9989999999999999991118216, 9.999999999999999547481118e-7, 0.0009998335831197161691357885, 1.000000000166500842926523e-9, 2.236627204213109355974419e-8),
        (0.9989999999999999991118216, 0.0001000000000000000047921736, 0.06142309442589368102984001, 1.000001665000000103600394e-7, 2.236627206075846883181590e-6),
        (0.9989999999999999991118216, 0.001000000000000000020816682, 0.1708509563235790123615538, 1.000166499991675909203566e-6, 2.236627390505302757834767e-5),
        (0.9989999999999999991118216, 0.01000000000000000020816682, 0.3874611232377607019610076, 1.016649916750199124030500e-5, 0.0002236645833635170784089479),
        (0.9989999999999999991118216, 0.1000000000000000055511151, 0.8515505079998896110696269, 0.0002664167698186759674183384, 0.002238491991231632920921750),
        (0.9989999999999999991118216, 0.5000000000000000000000000, 1.496223515511909722494319, 0.02105388693440120315280077, 0.01142196956879144219458535),
        (0.9989999999999999991118216, 1.000000000000000000000000, 1.933873556963495609973829, 0.1593704861769113906015266, 0.02443628416376029651037885),
        (0.9989999999999999991118216, 2.000000000000000000000000, 2.553893356599584302857120, 1.091611870601143987106995, 0.06963865308111168967829331),
        (0.9989999999999999991118216, 3.000000000000000000000000, 3.070731281645106685838526, 2.859021111948192645246696, 0.6110424292802833421478346),
        (0.9989999999999999991118216, 3.141591653589793420309206, 3.141592153339668266889716, 3.141590654589793602140424, 3.141547943411996596805073),
        (0.9989999999999999991118216, 3.500000000000000000000000, 3.321369141586789300403702, 3.850432444461930227960690, 6.037488177684316230515655),
        (0.9989999999999999991118216, 5.000000000000000000000000, 4.153175196314488414470803, 5.957965350388475329572565, 6.249772208313440662585180),
        (0.9989999999999999991118216, 6.183185307179586587267295, 5.431634799179697204881731, 6.282918890409767801652165, 6.280946815188354846602950),
        (0.9989999999999999991118216, 6.282185307179586786219261, 6.112334350856027362591874, 6.283184307013086485558847, 6.283162940905681430815929),
        (0.9989999999999999991118216, 6.283184307179586092217960, 6.282185473596082245385582, 6.283185306179586476374079, 6.283185284813314426189723),
        (0.9999000000000000110134124, 9.999999999999999798866476e-13, 9.999999999999434640106905e-9, 9.999999999998898457642713e-17, 7.071244595189784625822984e-15),
        (0.9999000000000000110134124, 1.000000000000000062281591e-9, 9.999998333501935137773960e-6, 9.999999999998915946575486e-14, 7.071244595189785209047025e-12),
        (0.9999000000000000110134124, 9.999999999999999547481118e-7, 0.008846308180180548821628683, 1.000000001666389820623985e-10, 7.071244595190373688980188e-9),
        (0.9999000000000000110134124, 0.0001000000000000000047921736, 0.08198421852346165576968320, 1.000016664999881581300275e-8, 7.071244601082194292208561e-7),
        (0.9999000000000000110134124, 0.001000000000000000020816682, 0.1807151554330339482674418, 1.001666499916564888798996e-7, 7.071245184430761775013353e-6),
        (0.9999000000000000110134124, 0.01000000000000000020816682, 0.3919903597837197685038732, 1.166649166751873824751386e-6, 7.071303519870781608061821e-5),
        (0.9999000000000000110134124, 0.1000000000000000055511151, 0.8535302901646385361071134, 0.0001765666948365294371946857, 0.0007077142902303962094324134),
        (0.9999000000000000110134124, 0.5000000000000000000000000, 1.497192750166677703575325, 0.02062240394965741474662822, 0.003611166436422549116851693),
        (0.9999000000000000110134124, 1.000000000000000000000000, 1.934494276402445623926721, 0.1586131622905842737306959, 0.007726038625383013504184330),
        (0.9999000000000000110134124, 2.000000000000000000000000, 2.554165706830330726487585, 1.090793502917000862759052, 0.02202473153853977600650163),
        (0.9999000000000000110134124, 3.000000000000000000000000, 3.070763184187473264118417, 2.858894103940938763067252, 0.1987721198893564464276754),
        (0.9999000000000000110134124, 3.141591653589793420309206, 3.141592153564792079317553, 3.141590653689793602304224, 3.141451235769395411523281),
        (0.9999000000000000110134124, 3.500000000000000000000000, 3.321288019847724520943658, 3.850748149366850889998877, 6.205153013062899969966439),
        (0.9999000000000000110134124, 5.000000000000000000000000, 4.152676786328916507397996, 5.958828382235672165607294, 6.272620650678853983842121),
        (0.9999000000000000110134124, 6.183185307179586587267295, 5.429655017014948278956367, 6.283008740484749948078606, 6.282477592889356081537415),
        (0.9999000000000000110134124, 6.282185307179586786219261, 6.102470151746571408927921, 6.283185207012936485299884, 6.283178235934402048350753),
        (0.9999000000000000110134124, 6.283184307179586092217960, 6.274338998996640724248530, 6.283185307079586476720177, 6.283185300108341879014553),
        (0.9999900000000000455102622, 9.999999999999999798866476e-13, 9.999999998378860061895095e-8, 9.999999999954489536807707e-18, 2.236073567685608385382072e-15),
        (0.9999900000000000455102622, 1.000000000000000062281591e-9, 9.998334182807864662416250e-5, 9.999999999954657025590480e-15, 2.236073567685608569809567e-12),
        (0.9999900000000000455102622, 9.999999999999999547481118e-7, 0.01707212148462387374720282, 1.000000016661948928524734e-11, 2.236073567685794667702729e-9),
        (0.9999900000000000455102622, 0.0001000000000000000047921736, 0.08411631152670069625426217, 1.000166664995365689230758e-9, 2.236073569548993862095130e-7),
        (0.9999900000000000455102622, 0.001000000000000000020816682, 0.1817025000884011322367298, 1.016666499162124016234537e-8, 2.236073754024159382442337e-6),
        (0.9999900000000000455102622, 0.01000000000000000020816682, 0.3924430882721442975376155, 2.666641666765290218397249e-7, 2.236092201725174679418742e-5),
        (0.9999900000000000455102622, 0.1000000000000000055511151, 0.8537281709820479126426715, 0.0001675816873383114590515868, 0.0002237938818267121338544861),
        (0.9999900000000000455102622, 0.5000000000000000000000000, 1.497289625687316006783004, 0.02057925565118301993793282, 0.001141926517439840205378273),
        (0.9999900000000000455102622, 1.000000000000000000000000, 1.934556317627265868931975, 0.1585374299019515340169990, 0.002443143899749600044256630),
        (0.9999900000000000455102622, 2.000000000000000000000000, 2.554192928372479053447426, 1.090711666148586520038570, 0.006964928338429418806499919),
        (0.9999900000000000455102622, 3.000000000000000000000000, 3.070766372862526914414105, 2.858881403140213370149068, 0.06304273683528020584445660),
        (0.9999900000000000455102622, 3.141591653589793420309206, 3.141592153587293316875912, 3.141590653599793602320570, 3.141145441119862439756276),
        (0.9999900000000000455102622, 3.500000000000000000000000, 3.321279911689352086319549, 3.850779719857342967886124, 6.258498609740123473779530),
        (0.9999900000000000455102622, 5.000000000000000000000000, 4.152626969998549465912292, 5.958914685420391881149361, 6.279844516660116179499169),
        (0.9999900000000000455102622, 6.183185307179586587267295, 5.429457136197538902332064, 6.283017725492248166046371, 6.282961513297759765051227),
        (0.9999900000000000455102622, 6.282185307179586786219261, 6.101482807091204122488180, 6.283185297012921485307294, 6.283183071105832453457555),
        (0.9999900000000000455102622, 6.283184307179586092217960, 6.266113185692492154249334, 6.283185307169586476754820, 6.283185304943512908379258),
        (0.9999989999999999712443355, 9.999999999999999798866476e-13, 9.999998333048276676611299e-7, 1.000000000028755644569525e-18, 7.071069579734757684171896e-16),
        (0.9999989999999999712443355, 1.000000000000000062281591e-9, 0.0008846222865528374386417366, 1.000000000028922393297803e-15, 7.071069579734758267381532e-13),
        (0.9999989999999999712443355, 9.999999999999999547481118e-7, 0.01806124662152221616916929, 1.000000166695255619255962e-12, 7.071069579735346761919243e-10),
        (0.9999989999999999712443355, 0.0001000000000000000047921736, 0.08432957381940450901412758, 1.001666665027922380129828e-10, 7.071069585627313207977124e-8),
        (0.9999989999999999712443355, 0.001000000000000000020816682, 0.1818012310059310447765084, 1.166666491695430889361109e-9, 7.071070168990320582286529e-7),
        (0.9999989999999999712443355, 0.01000000000000000020816682, 0.3924883589086551731507470, 1.766656666697716847132592e-7, 7.071128505874387316424490e-6),
        (0.9999989999999999712443355, 0.1000000000000000055511151, 0.8537479580848768817306265, 0.0001666831865884974198523221, 7.076968033290948202269443e-5),
        (0.9999989999999999712443355, 0.5000000000000000000000000, 1.497299312759878151830879, 0.02057494082133561771591229, 0.0003611080943914067793720410),
        (0.9999989999999999712443355, 1.000000000000000000000000, 1.934562521442650305745617, 0.1585298566630883254410617, 0.0007725885450241593671458131),
        (0.9999989999999999712443355, 2.000000000000000000000000, 2.554195650391947638574918, 1.090703482471745156433127, 0.002202506786639806399261453),
        (0.9999989999999999712443355, 3.000000000000000000000000, 3.070766691714248328016344, 2.858880133060140841824477, 0.01994176343766897677357110),
        (0.9999989999999999712443355, 3.141591653589793420309206, 3.141592153589543329250546, 3.141590653590793602322283, 3.140178440616952976261553),
        (0.9999989999999999712443355, 3.500000000000000000000000, 3.321279100913650241768213, 3.850782876906392148413516, 6.275378348724033858780075),
        (0.9999989999999999712443355, 5.000000000000000000000000, 4.152621988612100539947425, 5.958923315738863778180181, 6.282128857949566173941687),
        (0.9999989999999999712443355, 6.183185307179586587267295, 5.429437349094709933235235, 6.283018623992997980084533, 6.283114537499253567525418),
        (0.9999989999999999712443355, 6.282185307179586786219261, 6.101384076173674199701172, 6.283185306012919985230320, 6.283184600072569578111947),
        (0.9999989999999999712443355, 6.283184307179586092217960, 6.265124060555719909547018, 6.283185307178586476758207, 6.283185306472479518679723),
        (0.9999999899999999497524072, 9.999999999999999798866476e-13, 8.846221980637337245840037e-5, 1.000000005024759271882730e-20, 7.071067847308351595740255e-17),
        (0.9999999899999999497524072, 1.000000000000000062281591e-9, 0.001806114407609856040462127, 1.000000005041426004111007e-17, 7.071067847308352178949749e-14),
        (0.9999999899999999497524072, 9.999999999999999547481118e-7, 0.01817020532365334610734764, 1.000016671691259229241094e-14, 7.071067847308940673634959e-11),
        (0.9999999899999999497524072, 0.0001000000000000000047921736, 0.08435303262264850368308386, 1.166666669941426006357262e-12, 7.071067853200908592674382e-9),
        (0.9999999899999999497524072, 0.001000000000000000020816682, 0.1818120913540779616443936, 1.766666567169145434078962e-10, 7.071068436564061806814306e-8),
        (0.9999999899999999497524072, 0.01000000000000000020816682, 0.3924933386538288841247720, 1.667658316691616021718680e-7, 7.071126773462713102940502e-7),
        (0.9999999899999999497524072, 0.1000000000000000055511151, 0.8537501346553166743373882, 0.0001665843515060192055886964, 7.076966302343531404841934e-6),
        (0.9999999899999999497524072, 0.5000000000000000000000000, 1.497300378332537447139271, 0.02057446619005240985872129, 3.611080098039622286244970e-5),
        (0.9999999899999999497524072, 1.000000000000000000000000, 1.934563203858933905587923, 0.1585290236068133837083541, 7.725883937834911695017464e-5),
        (0.9999999899999999497524072, 2.000000000000000000000000, 2.554195949812593565830140, 1.090702582267292618550804, 0.0002202507128487683326664265),
        (0.9999999899999999497524072, 3.000000000000000000000000, 3.070766726787762489962200, 2.858879993351332865588868, 0.001994241282871993732759427),
        (0.9999999899999999497524072, 3.141591653589793420309206, 3.141592153589790829375484, 3.141590653589803602322484, 3.127450753734704468441384),
        (0.9999999899999999497524072, 3.500000000000000000000000, 3.321279011728768520766029, 3.850783224181787553598158, 6.282404607599739335003641),
        (0.9999999899999999497524072, 5.000000000000000000000000, 4.152621440662328184714640, 5.958924265073895674078133, 6.283079662272740199875538),
        (0.9999999899999999497524072, 6.183185307179586587267295, 5.429435172524270140627497, 6.283018722828080458298682, 6.283178230213284133402097),
        (0.9999999899999999497524072, 6.282185307179586786219261, 6.101373215825527281706096, 6.283185307002919820208530, 6.283185236468902111306541),
        (0.9999999899999999497524072, 6.283184307179586092217960, 6.265015101853602752240233, 6.283185307179576476758566, 6.283185307108875798424994),
        (0.9999999998999999917259629, 9.999999999999999798866476e-13, 0.0001806114304211083619602022, 1.000000082740372645643686e-22, 7.071068104573632824727854e-18),
        (0.9999999998999999917259629, 1.000000000000000062281591e-9, 0.001817010628617888795882946, 1.000000084407037727871970e-19, 7.071068104573633407937368e-15),
        (0.9999999998999999917259629, 9.999999999999999547481118e-7, 0.01817129492367488775150256, 1.001666749406870870265099e-16, 7.071068104574221902646906e-12),
        (0.9999999998999999917259629, 0.0001000000000000000047921736, 0.08435326721057090978821190, 1.766666673940703997468242e-13, 7.071068110466190065217887e-10),
        (0.9999999998999999917259629, 0.001000000000000000020816682, 0.1818121999575057318983959, 1.667666583249409120440202e-10, 7.071068693829367391393736e-9),
        (0.9999999998999999917259629, 0.01000000000000000020816682, 0.3924933884512559401035707, 1.666668333187336236284739e-7, 7.071127030730429919554153e-8),
        (0.9999999998999999917259629, 0.1000000000000000055511151, 0.8537501564210102833137315, 0.0001665833631551902116264642, 7.076966559852656182499254e-7),
        (0.9999999998999999917259629, 0.5000000000000000000000000, 1.497300388988258758174078, 0.02057446144373955755391706, 3.611080229809321632975733e-6),
        (0.9999999998999999917259629, 1.000000000000000000000000, 1.934563210683093358927709, 0.1585290152762505987906495, 7.725884222728746940936336e-6),
        (0.9999999998999999917259629, 2.000000000000000000000000, 2.554195952806798540908912, 1.090702573265248054810109, 2.202507217435737744128047e-5),
        (0.9999999998999999917259629, 3.000000000000000000000000, 3.070766727138497457726268, 2.858879991954244779872874, 0.0001994242009743634583145604),
        (0.9999999998999999917259629, 3.141591653589793420309206, 3.141592153589793304375506, 3.141590653589793702322444, 3.000406300872313543544508),
        (0.9999999998999999917259629, 3.500000000000000000000000, 3.321279010836920145635244, 3.850783227654541522449013, 6.283107237214835760660964),
        (0.9999999998999999917259629, 5.000000000000000000000000, 4.152621435182833177286213, 5.958924274567246033492666, 6.283174742688507755953325),
        (0.9999999998999999917259629, 6.183185307179586587267295, 5.429435150758576531651144, 6.283018723816431287292643, 6.283184599482930491660490),
        (0.9999999998999999917259629, 6.282185307179586786219261, 6.101373107222099511440822, 6.283185307012819818600501, 6.283185300108517783098107),
        (0.9999999998999999917259629, 6.283184307179586092217960, 6.265014012253581350331025, 6.283185307179586376758612, 6.283185307172515408817992),
        (0.9999999999990000221217201, 9.999999999999999798866476e-13, 0.0001817010532025818137748445, 9.999778782800451425133241e-25, 7.070989599343144393832958e-19),
        (0.9999999999990000221217201, 1.000000000000000062281591e-9, 0.001817119592214449068715182, 9.999780449465452247398135e-22, 7.070989599343144977035997e-16),
        (0.9999999999990000221217201, 9.999999999999999547481118e-7, 0.01817130581967836133214972, 1.166644544946370098436354e-18, 7.070989599343733465211910e-13),
        (0.9999999999990000221217201, 0.0001000000000000000047921736, 0.08435326955645083243158788, 1.667666643709946821888006e-13, 7.070989605235636213583860e-11),
        (0.9999999999990000221217201, 0.001000000000000000020816682, 0.1818122010435403330096920, 1.666676583110451590615056e-10, 7.070990188592336886468890e-10),
        (0.9999999999990000221217201, 0.01000000000000000020816682, 0.3924933889492303589544198, 1.666658433349295886223698e-7, 7.071048524845727609306782e-9),
        (0.9999999999990000221217201, 0.1000000000000000055511151, 0.8537501566386672842192481, 0.0001665833532716789290781816, 7.076887989135947464410071e-8),
        (0.9999999999990000221217201, 0.5000000000000000000000000, 1.497300389094816003015949, 0.02057446139627641465959868, 3.611040138460448772296596e-7),
        (0.9999999999990000221217201, 1.000000000000000000000000, 1.934563210751334973782675, 0.1585290151929449457175200, 7.725798447561028855802070e-7),
        (0.9999999999990000221217201, 2.000000000000000000000000, 2.554195952836740599576100, 1.090702573175227581915583, 2.202482764593406756610117e-6),
        (0.9999999999990000221217201, 3.000000000000000000000000, 3.070766727142004808448358, 2.858879991940273894785498, 1.994219875582333326139246e-5),
        (0.9999999999990000221217201, 3.141591653589793420309206, 3.141592153589793329125514, 3.141590653589793603322414, 1.910622808007177968271407),
        (0.9999999999990000221217201, 3.500000000000000000000000, 3.321279010828001659228114, 3.850783227689269072652608, 6.283177500269783219338917),
        (0.9999999999990000221217201, 5.000000000000000000000000, 4.152621435128038210894618, 5.958924274662179565831546, 6.283184250742207626625566),
        (0.9999999999990000221217201, 6.183185307179586587267295, 5.429435150540919530745627, 6.283018723826314798575191, 6.283185236410706585565894),
        (0.9999999999990000221217201, 6.282185307179586786219261, 6.101373106136064910329413, 6.283185307012918818614396, 6.283185306472487458066272),
        (0.9999999999990000221217201, 6.283184307179586092217960, 6.265014001357577878147728, 6.283185307179586475758642, 6.283185307178879377965080),
    ];

    /// The distance between two angles, folded into `[0, π]`.
    fn turn_distance(a: f64, b: f64) -> f64 {
        let d = libm::fabs(a - b) % Angle::TAU;
        d.min(Angle::TAU - d)
    }

    /// The decade table's bounds, `(e, M → E, E → M, ν → E)` in `u`: what each map measures on its
    /// row, rounded up to a tenth.
    const DECADE_BOUNDS: [(f64, f64, f64, f64); 11] = [
        (0.0, 0.1, 0.1, 0.8),
        (0.5, 0.7, 0.6, 0.9),
        (0.9, 3.9, 4.9, 1.4),
        (0.99, 28.1, 40.8, 1.7),
        (0.999, 1.4, 1.4, 1.4),
        (0.999_9, 1.4, 1.4, 1.4),
        (0.999_99, 1.4, 1.4, 1.4),
        (0.999_999, 1.4, 1.4, 1.4),
        (0.999_999_99, 1.4, 1.4, 1.4),
        (0.999_999_999_9, 1.4, 1.4, 1.4),
        (0.999_999_999_999, 1.4, 1.4, 1.4),
    ];

    /// (G2) The decade table: each of the three anomaly maps against its 100-digit value, in units
    /// of `u = 2⁻⁵³` of the floor its own input sets — the reference's size plus the input's size
    /// carried through the map's derivative, which is what one rounding of the input moves the answer
    /// by. `dE/dM = 1/(1 − e cos E)`, `dM/dE = 1 − e cos E`, `dE/dν = √(1 − e²)/(1 + e cos ν)`.
    ///
    /// Above `e = 0.99`, where the rearranged forms run, every map holds 1.4 u on every row. Below,
    /// the classic forms keep their measured cost, which grows with `1/(1 − e)` near perigee: 3.9 u
    /// and 4.9 u at `e = 0.9`, 28 u and 41 u at the double nearest 0.99. With the classic forms on
    /// every row, as before the change, the rows above 0.99 measured 385 u to 1.2e7 u for `M → E`,
    /// 509 u to 3.8e11 u for `E → M` and 33 u to 1.2e6 u for `ν → E`.
    #[test]
    fn the_anomaly_maps_hold_their_decade_table() {
        let u = f64::EPSILON / 2.0;
        let mut rows = [(0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64); 11];
        let mut row = 0;
        for (k, &(e, x, e_of_m, m_of_e, e_of_nu)) in DECADES.iter().enumerate() {
            #[expect(
                clippy::float_cmp,
                reason = "the table's rows are grouped by one exact eccentricity"
            )]
            let next = k > 0 && e != DECADES[k - 1].0;
            if next {
                row += 1;
            }
            let els = KeplerianElements {
                semi_major_axis: 1e7,
                eccentricity: e,
                inclination: Angle::from_radians(0.5),
                raan: Angle::from_radians(0.0),
                arg_periapsis: Angle::from_radians(0.0),
                true_anomaly: Angle::from_radians(x),
                mu: 3.986_004_418e14,
            };
            let slope = |big_e: f64| {
                let half = libm::sin(0.5 * big_e);
                (1.0 - e) + 2.0 * e * half * half
            };
            let (solved, _) = eccentric_from_mean(x, e);
            let m_to_e = turn_distance(solved, e_of_m) / (u * (e_of_m + x / slope(e_of_m)));
            let forward = wrap_tau(mean_from_eccentric(x, e));
            let e_to_m = turn_distance(forward, m_of_e) / (u * (m_of_e + x * slope(x)));
            let dedn = libm::sqrt(1.0 - e * e) / (1.0 + e * libm::cos(x));
            let nu_to_e = turn_distance(els.eccentric_anomaly().radians(), e_of_nu)
                / (u * (e_of_nu + x * dedn));
            let r = &mut rows[row];
            *r = (e, r.1.max(m_to_e), r.2.max(e_to_m), r.3.max(nu_to_e));
        }
        for (e, a, b, c) in rows {
            std::println!("e = {e:<18} M→E {a:>9.3} u   E→M {b:>9.3} u   ν→E {c:>9.3} u");
        }
        for ((e, a, b, c), (_, ba, bb, bc)) in rows.into_iter().zip(DECADE_BOUNDS) {
            assert!(
                a <= ba && b <= bb && c <= bc,
                "e = {e}: ({a:.3}, {b:.3}, {c:.3}) u against ({ba}, {bb}, {bc})"
            );
        }
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
