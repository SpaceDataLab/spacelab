//! The circular restricted three-body problem (CR3BP) — the arena of libration orbits and low-energy
//! transfers.
//!
//! Two primaries on a circular orbit, a massless third body; the equations are written in the
//! co-rotating (synodic) frame and nondimensionalised so the primaries sit at `(−μ, 0, 0)` and
//! `(1−μ, 0, 0)`, the separation and mean motion are 1, and the only parameter is the mass ratio
//! `μ = m₂/(m₁+m₂)`. This is the model for Earth–Moon halo orbits (Gateway), Sun–Earth L1/L2 (JWST,
//! SOHO), and ballistic lunar transfers. See Szebehely, *Theory of Orbits*; Koon, Lo, Marsden & Ross,
//! *Dynamical Systems, the Three-Body Problem and Space Mission Design*.
//!
//! This slice covers the invariants and equilibria — pseudo-potential, Jacobi constant, the five
//! libration points, the equations of motion. Periodic orbits (halo/Lyapunov by differential
//! correction) and their manifolds build on the equations of motion here.
//!
//! Gate: `derivations/cr3bp.mac` shows that the coded acceleration is `∇Ω` plus the Coriolis term,
//! that the Jacobi constant has zero rate along it, that the collinear points are the roots of the
//! three classical quintics and that L4/L5 are exact for every μ; `derivations/cr3bp.gp` roots the
//! quintics and evaluates the closed forms at 100 digits. The module tests
//! `the_synodic_acceleration_is_its_closed_form_in_all_three_components` (G1),
//! `the_collinear_points_are_the_correctly_rounded_roots_of_the_gradient` (G2),
//! `the_triangular_points_are_the_exact_equilateral_solution` (G3) and
//! `the_jacobi_constant_has_zero_rate_along_the_vector_field` (G4) hold the code to them.

// r1/r2, L1..L5, a1/a2 are the standard notation of the three-body problem.
#![allow(clippy::similar_names, clippy::many_single_char_names)]

use spacelab_core::Vec3;

/// A CR3BP system, parameterised by its mass ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cr3bp {
    /// Mass parameter `μ = m₂/(m₁+m₂) ∈ (0, ½]`. The secondary sits at `(1−μ, 0, 0)`.
    pub mu: f64,
}

impl Cr3bp {
    /// Earth–Moon mass ratio, `r/(1 + r)` with the IAU 2009 best estimate `r = M_M/M_E =
    /// 0.012 300 0371` (Luzum et al. 2011), correctly rounded from the exact ratio in
    /// `derivations/cr3bp.gp`. The common eight-digit `0.012 150 585` is 4.4e-8 above it.
    pub const EARTH_MOON: Self = Self {
        mu: 0.012_150_584_460_351,
    };
    /// Sun–Earth mass ratio, the Earth alone: `1/(1 + GM_S/GM_E)` with the IAU 2009 best estimate
    /// `GM_S/GM_E = 332 946.0487` (Luzum et al. 2011), correctly rounded from the exact ratio in
    /// `derivations/cr3bp.gp`. Sun to Earth+Moon, the other common choice, is `3.040_423_4e-6`.
    pub const SUN_EARTH: Self = Self {
        mu: 3.003_480_595_201_323_4e-6,
    };

    /// Distances to the primary (at `−μ`) and the secondary (at `1−μ`).
    fn radii(self, p: Vec3) -> (f64, f64) {
        let (dx1, dx2) = (p.x + self.mu, p.x - 1.0 + self.mu);
        let r1 = libm::sqrt(dx1 * dx1 + p.y * p.y + p.z * p.z);
        let r2 = libm::sqrt(dx2 * dx2 + p.y * p.y + p.z * p.z);
        (r1, r2)
    }

    /// The effective (pseudo-)potential `Ω = ½(x²+y²) + (1−μ)/r₁ + μ/r₂`.
    #[must_use]
    pub fn pseudo_potential(self, p: Vec3) -> f64 {
        let (r1, r2) = self.radii(p);
        0.5 * (p.x * p.x + p.y * p.y) + (1.0 - self.mu) / r1 + self.mu / r2
    }

    /// The Jacobi constant `C = 2Ω − v²`, the CR3BP's single integral of motion.
    #[must_use]
    pub fn jacobi(self, p: Vec3, v: Vec3) -> f64 {
        2.0 * self.pseudo_potential(p) - v.dot(v)
    }

    /// Synodic-frame acceleration `(ẍ, ÿ, z̈)` for position `p` and velocity `v`, including the
    /// Coriolis and centrifugal terms. The full state derivative is `(v, acceleration(p, v))`.
    #[must_use]
    pub fn acceleration(self, p: Vec3, v: Vec3) -> Vec3 {
        let (r1, r2) = self.radii(p);
        let (a1, a2) = ((1.0 - self.mu) / (r1 * r1 * r1), self.mu / (r2 * r2 * r2));
        Vec3::new(
            p.x + 2.0 * v.y - a1 * (p.x + self.mu) - a2 * (p.x - 1.0 + self.mu),
            p.y - 2.0 * v.x - a1 * p.y - a2 * p.y,
            -a1 * p.z - a2 * p.z,
        )
    }

    /// The triangular points `L4` (`+y`) and `L5` (`−y`), which are exact: `(½−μ, ±√3/2, 0)`.
    #[must_use]
    pub fn triangular_points(self) -> (Vec3, Vec3) {
        let y = libm::sqrt(0.75);
        (
            Vec3::new(0.5 - self.mu, y, 0.0),
            Vec3::new(0.5 - self.mu, -y, 0.0),
        )
    }

    /// The collinear points `[L1, L2, L3]` on the x-axis, by Newton on `dΩ/dx = 0`. L1 lies between
    /// the primaries, L2 beyond the secondary, L3 beyond the primary.
    #[must_use]
    pub fn collinear_points(self) -> [Vec3; 3] {
        let mu = self.mu;
        // dΩ/dx on the x-axis, and its derivative (the |·|³ derivative is −2/r³ regardless of sign).
        let d_omega = |x: f64| {
            let (r1, r2) = (libm::fabs(x + mu), libm::fabs(x - 1.0 + mu));
            x - (1.0 - mu) * (x + mu) / (r1 * r1 * r1) - mu * (x - 1.0 + mu) / (r2 * r2 * r2)
        };
        let d2_omega = |x: f64| {
            let (r1, r2) = (libm::fabs(x + mu), libm::fabs(x - 1.0 + mu));
            1.0 + 2.0 * (1.0 - mu) / (r1 * r1 * r1) + 2.0 * mu / (r2 * r2 * r2)
        };
        let newton = |x0: f64| {
            let mut x = x0;
            for _ in 0..100 {
                let dx = d_omega(x) / d2_omega(x);
                x -= dx;
                if libm::fabs(dx) < 1e-14 {
                    break;
                }
            }
            x
        };
        let gamma = libm::cbrt(mu / 3.0); // first-order distance of L1/L2 from the secondary
        [
            Vec3::new(newton(1.0 - mu - gamma), 0.0, 0.0),
            Vec3::new(newton(1.0 - mu + gamma), 0.0, 0.0),
            Vec3::new(newton(-1.0 - mu), 0.0, 0.0),
        ]
    }

    /// All five libration points, `[L1, L2, L3, L4, L5]`.
    #[must_use]
    pub fn libration_points(self) -> [Vec3; 5] {
        let [l1, l2, l3] = self.collinear_points();
        let (l4, l5) = self.triangular_points();
        [l1, l2, l3, l4, l5]
    }
}

#[cfg(test)]
mod tests {
    // The references below are carried at the precision they were computed to, so that
    // regenerating one is a copy rather than a rounding decision.
    #![allow(clippy::excessive_precision)]

    use spacelab_testkit::{assert_within_ulps, ulps_between};

    use super::*;

    /// A field point with no symmetry left in it: the three coordinates are distinct and nonzero,
    /// it sits on neither primary, and `r₁ ≠ r₂`. Every arithmetic slip inside `radii` moves at
    /// least one radius, and the potential, the Jacobi constant and the acceleration all read them.
    const P: Vec3 = Vec3::new(0.31, -0.47, 0.23);
    /// Likewise for the velocity — three distinct nonzero components, so the two rotation terms
    /// `+2ẏ` and `−2ẋ` cannot be swapped, dropped or confused with each other unnoticed.
    const V: Vec3 = Vec3::new(0.17, -0.29, 0.41);

    /// `got` is the closed form to within a few rounding steps.
    ///
    /// Four ulps is what the longest of these chains — the acceleration, eight operations deep —
    /// costs in accumulated rounding; measured, the worst is one. The tolerance is not what makes
    /// these assertions strong. Every one of them compares against a value computed independently
    /// at 60 decimal digits from the definition, and any operator that is not the one written
    /// moves the result by an O(1) amount, not by an ulp.
    #[track_caller]
    fn assert_closed_form(got: f64, want: f64, what: &str) {
        assert_within_ulps(got, want, 4, what);
    }

    /// The pseudo-potential is `½(x²+y²) + (1−μ)/r₁ + μ/r₂`, evaluated at 60 digits from that
    /// definition and rounded once. The `z` coordinate reaches it only through `r₁` and `r₂`,
    /// which is why the reference point has one.
    #[test]
    fn the_pseudo_potential_is_its_closed_form_at_a_point_with_no_symmetry() {
        assert_closed_form(
            Cr3bp::EARTH_MOON.pseudo_potential(P),
            1.780_317_787_318_750_078_456_715_328_779_219,
            "Omega",
        );
    }

    /// `C = 2Ω − v·v`. The velocity is what separates this from the potential test above: with
    /// `v = 0` the Jacobi constant is `2Ω` and the second term is invisible.
    #[test]
    fn the_jacobi_constant_is_twice_the_potential_less_the_squared_speed() {
        assert_closed_form(
            Cr3bp::EARTH_MOON.jacobi(P, V),
            3.279_535_574_637_500_184_380_348_286_784_810,
            "C",
        );
    }

    /// (G1) All three components of the synodic acceleration against `∇Ω + (2ẏ, −2ẋ, 0)`, the
    /// identity `derivations/cr3bp.mac` shows the coded expression to be, evaluated at 100 digits
    /// in `derivations/cr3bp.gp` at the f64 field point. `z̈` carries no rotation term at all, so
    /// a Coriolis or centrifugal error that survives `ẍ` and `ÿ` has to be caught by the two of
    /// them agreeing with different arithmetic, not by the third.
    #[test]
    fn the_synodic_acceleration_is_its_closed_form_in_all_three_components() {
        let a = Cr3bp::EARTH_MOON.acceleration(P, V);
        assert_closed_form(a.x, -1.628_505_363_613_640_063_773_917_538_670_481, "a.x");
        assert_closed_form(a.y, 1.200_216_013_413_323_035_419_682_785_396_339, "a.y");
        assert_closed_form(a.z, -0.983_722_729_968_222_008_369_268_859_907_687_4, "a.z");
    }

    /// (G3) L4 and L5 close an equilateral triangle with the primaries, so they are exact — `∇Ω`
    /// vanishes there identically in μ (`derivations/cr3bp.mac`): `x = ½−μ` and `y = ±√3/2`,
    /// with no iteration and no truncation. Both points are asserted in full —
    /// `l5.x` and the two `z` components included, since a sign or a component dropped there is
    /// invisible to an assertion that only reads `l4.x`, `l4.y` and `l5.y`.
    #[test]
    fn the_triangular_points_are_the_exact_equilateral_solution() {
        let (l4, l5) = Cr3bp::EARTH_MOON.triangular_points();
        // ½ − μ at the f64 μ the constant carries, and √3/2, both to 34 digits.
        let (x, y) = (
            0.487_849_415_539_649_000_628_230_623_988_201_8,
            0.866_025_403_784_438_646_763_723_170_752_936_2,
        );
        assert_closed_form(l4.x, x, "L4 x");
        assert_closed_form(l4.y, y, "L4 y");
        assert_closed_form(l5.x, x, "L5 x");
        assert_closed_form(l5.y, -y, "L5 y");
        assert!(l4.z == 0.0 && l5.z == 0.0, "L4 {l4:?}, L5 {l5:?}");
    }

    /// (G2) The collinear points to the last bit, at three mass ratios spanning six orders of
    /// magnitude: the references are the real roots of the three quintics
    /// `derivations/cr3bp.mac` derives from `dΩ/dx = 0`, solved at 100 digits in
    /// `derivations/cr3bp.gp` for the f64 mass ratios the systems carry.
    ///
    /// Newton is self-correcting: the second derivative sets only the step length, so corrupting
    /// it changes how many iterations the loop needs and not where it lands — an assertion at
    /// `1e-4`, or at any tolerance the iteration can still reach, cannot see the difference. What
    /// does see it is the last bit. With `d²Ω/dx²` as written the loop converges quadratically and
    /// returns the correctly rounded root; with it scaled or sign-flipped, convergence turns linear
    /// and the `|dx| < 1e-14` break stops the walk about two decimal digits short.
    ///
    /// `μ = 0.4639` is here because the starting point and the break test have no other observable.
    /// Both are private to the iteration, both leave the answer a root, and neither shifts it by
    /// more than a single ulp; at every one of the fourteen real systems checked — Sun–Jupiter down
    /// to Mars–Phobos, Earth–Moon and Sun–Earth included — corrupting either changes nothing at all.
    /// They separate at roughly one mass ratio in twenty, and this is one where the iteration also
    /// reaches the correctly rounded root in all three points, so the assertion stays a statement
    /// about the mathematics rather than a photograph of the current arithmetic.
    ///
    /// All nine are correctly rounded and asserted as exact. (Sun–Earth L3, at `x ≈ −1 − 1.25e-6`
    /// and a part in 8e5 from a cancellation against the `−1` in `x − 1 + μ`, was one ulp off at
    /// the mass ratio's former eight-digit rounding and is exact at the full-precision one.) That is
    /// the algorithm's behaviour at these three, not its guarantee: the error is bounded in absolute terms, near `1e-16`, so where a root approaches
    /// the origin — `L1 → 0` as `μ → ½` — the same absolute error is hundreds of ulps. The bound
    /// that holds everywhere is the residual one, asserted over the whole μ range below.
    #[test]
    fn the_collinear_points_are_the_correctly_rounded_roots_of_the_gradient() {
        // Roots of the quintics at 100 digits (derivations/cr3bp.gp), from the f64 mass ratios.
        for (name, sys, want, tol) in [
            (
                "Earth-Moon",
                Cr3bp::EARTH_MOON,
                [
                    0.836_915_131_427_382_160_046_693_265_085_142_7,
                    1.155_682_161_024_676_951_248_163_210_236_372,
                    -1.005_062_645_331_442_005_708_330_094_870_261,
                ],
                [0, 0, 0],
            ),
            (
                "Sun-Earth",
                Cr3bp::SUN_EARTH,
                [
                    0.990_026_593_870_022_409_736_302_340_399_785_0,
                    1.010_034_116_422_946_031_576_144_465_434_425,
                    -1.000_001_251_450_247_999_078_835_863_664_773,
                ],
                [0, 0, 0],
            ),
            (
                "near equal mass",
                Cr3bp { mu: 0.4639 },
                [
                    0.050_985_075_567_607_166_483_247_148_334_657_92,
                    1.210_662_680_043_992_900_440_520_787_331_401,
                    -1.185_643_513_328_397_515_409_973_511_296_275,
                ],
                [0, 0, 0],
            ),
        ] {
            let got = sys.collinear_points();
            for i in 0..3 {
                let d = ulps_between(got[i].x, want[i]);
                assert!(
                    d <= tol[i],
                    "{name} L{}: got {:.17e}, want {:.17e}, {d} ulps (allowed {})",
                    i + 1,
                    got[i].x,
                    want[i],
                    tol[i]
                );
                assert!(
                    got[i].y == 0.0 && got[i].z == 0.0,
                    "{name} L{} off-axis: {:?}",
                    i + 1,
                    got[i]
                );
            }
        }
    }

    /// Newton converges from anywhere in a root's basin, so a wrong *starting* point is invisible
    /// to any test that only asks where the iteration ended — for Earth–Moon and Sun–Earth every
    /// plausible corruption of `γ = ∛(μ/3)` still lands on the intended root. It stops being
    /// invisible at large μ: at the equal-mass limit, `∛(3μ)` puts the L1 guess at `−0.64`, past
    /// the primary and inside L3's basin, and the point that comes back is not between the
    /// primaries at all.
    ///
    /// So the property is asserted across the whole parameter range rather than at two systems:
    /// for every mass ratio from `1e-9` to the equal-mass limit, L1 lies strictly between the
    /// primaries, L2 beyond the secondary, L3 beyond the primary, and all five points are
    /// equilibria. The residual bound is the measured floor over the sweep — worst `1.1e-15`.
    #[test]
    fn every_mass_ratio_puts_the_three_roots_in_their_own_regions() {
        for k in 0..=48 {
            let mu = 0.5 * libm::pow(2e-9, f64::from(48 - k) / 48.0);
            let sys = Cr3bp { mu };
            let [l1, l2, l3] = sys.collinear_points();
            assert!(l1.x > -mu && l1.x < 1.0 - mu, "mu {mu:e}: L1 {}", l1.x);
            assert!(l2.x > 1.0 - mu, "mu {mu:e}: L2 {}", l2.x);
            assert!(l3.x < -mu, "mu {mu:e}: L3 {}", l3.x);
            for (i, lp) in sys.libration_points().into_iter().enumerate() {
                let r = sys.acceleration(lp, Vec3::ZERO).norm();
                assert!(r < 2e-15, "mu {mu:e}: L{} residual {r:.3e}", i + 1);
            }
        }
    }

    /// Each libration point is an equilibrium: the synodic acceleration vanishes there. This is the
    /// one check on `collinear_points` that needs no reference value — it asks the equations of
    /// motion themselves, through the public API, whether the point Newton returned is a root.
    /// The residual is not merely small, it is at the representable floor, so the bound is set
    /// there: the loosest of the ten is 7.5e-16, roughly three ulps of the coordinate.
    #[test]
    fn every_libration_point_is_an_equilibrium_to_machine_precision() {
        for (name, sys) in [
            ("Earth-Moon", Cr3bp::EARTH_MOON),
            ("Sun-Earth", Cr3bp::SUN_EARTH),
        ] {
            for (i, lp) in sys.libration_points().into_iter().enumerate() {
                let a = sys.acceleration(lp, Vec3::ZERO);
                assert!(
                    a.norm() < 1e-15,
                    "{name} L{}: acceleration {:.3e} at {lp:?} is not zero",
                    i + 1,
                    a.norm()
                );
            }
        }
    }

    /// (G4) The Jacobi constant has zero rate along the shipped vector field, `∇C·(v, a) = 0`,
    /// which `derivations/cr3bp.mac` shows symbolically (`jacobi_rate_vanishes`): the gyroscopic
    /// `(2ẏ, −2ẋ, 0)` does no work and the rest of `a` is `∇Ω`.
    ///
    /// Two independent readings of that rate. The first is exact algebra on `acceleration` alone,
    /// with `∇Ω = acceleration(p, 0)`: `2∇Ω·v − 2v·a` cancels to rounding, a few ulps of `|v||a|`.
    /// The second differences `jacobi` itself along the direction `(v, a)` in state space, by a
    /// fourth-order central difference at `h = 1e-3`; truncation (`h⁴`) and cancellation
    /// (`ε·C/h`) both sit near `1e-12` there, and the measured worst is below `2e-12`. A Coriolis
    /// sign, a dropped centrifugal term or a wrong factor in `jacobi` moves either by O(1).
    #[test]
    fn the_jacobi_constant_has_zero_rate_along_the_vector_field() {
        let sys = Cr3bp::EARTH_MOON;
        for (p, v) in [
            (P, V),
            (Vec3::new(0.82, 0.01, 0.02), Vec3::new(0.003, 0.12, -0.04)),
            (Vec3::new(-0.6, 0.75, -0.1), Vec3::new(0.4, 0.2, 0.15)),
        ] {
            let a = sys.acceleration(p, v);
            let grad = sys.acceleration(p, Vec3::ZERO);
            let rate = 2.0 * grad.dot(v) - 2.0 * v.dot(a);
            let floor = 8.0 * f64::EPSILON * v.norm() * (a.norm() + grad.norm());
            assert!(
                rate.abs() <= floor,
                "{p:?}: 2∇Ω·v − 2v·a = {rate:.3e} (floor {floor:.3e})"
            );

            let h = 1e-3;
            let at = |k: f64| sys.jacobi(p + v.scale(k * h), v + a.scale(k * h));
            let fd = (at(-2.0) - 8.0 * at(-1.0) + 8.0 * at(1.0) - at(2.0)) / (12.0 * h);
            println!("{p:?}: algebraic rate {rate:.3e} (floor {floor:.3e}), differenced {fd:.3e}");
            assert!(
                fd.abs() < 1e-10,
                "{p:?}: dC/dt by differencing `jacobi` = {fd:.3e}"
            );
        }
    }

    /// The Jacobi constant is conserved along a trajectory (integrated with a local RK4).
    #[test]
    fn jacobi_constant_is_conserved() {
        let sys = Cr3bp::EARTH_MOON;
        // A state near L1 with a small transverse velocity.
        let mut p = Vec3::new(0.82, 0.0, 0.02);
        let mut v = Vec3::new(0.0, 0.12, 0.0);
        let c0 = sys.jacobi(p, v);
        let h = 1e-4;
        let deriv = |p: Vec3, v: Vec3| (v, sys.acceleration(p, v));
        for _ in 0..20_000 {
            let (k1p, k1v) = deriv(p, v);
            let (k2p, k2v) = deriv(add(p, k1p, h / 2.0), add(v, k1v, h / 2.0));
            let (k3p, k3v) = deriv(add(p, k2p, h / 2.0), add(v, k2v, h / 2.0));
            let (k4p, k4v) = deriv(add(p, k3p, h), add(v, k3v, h));
            p = rk_step(p, k1p, k2p, k3p, k4p, h);
            v = rk_step(v, k1v, k2v, k3v, k4v, h);
        }
        let drift = (sys.jacobi(p, v) - c0).abs();
        assert!(drift < 1e-6, "Jacobi constant drifted by {drift:.3e}");
    }

    fn add(a: Vec3, b: Vec3, s: f64) -> Vec3 {
        Vec3::new(a.x + b.x * s, a.y + b.y * s, a.z + b.z * s)
    }
    fn rk_step(y: Vec3, k1: Vec3, k2: Vec3, k3: Vec3, k4: Vec3, h: f64) -> Vec3 {
        Vec3::new(
            y.x + h / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
            y.y + h / 6.0 * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
            y.z + h / 6.0 * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
        )
    }
}
