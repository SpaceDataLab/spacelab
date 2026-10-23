//! Combined DSST mean-element propagator.
//!
//! Sums the ported mean-element rates -- zonal, Sun + Moon third-body, solar radiation pressure
//! and the (resonant) tesseral geopotential -- into a single right-hand side and integrates the
//! equinoctial mean state `[a, k, h, q, p, λ]` with DOP853, adding the Keplerian mean-longitude
//! rate `n` (Orekit's auto-inserted `DSSTNewtonianAttraction`). Each contribution is individually
//! bit-exact against Orekit's `getMeanElementRate` (see the per-module tests); this only assembles
//! them. Atmospheric drag is a follow-on.
//!
//! The time-dependent inputs -- the geocentric Sun and Moon positions and the Greenwich sidereal
//! angle -- are supplied as closures of the seconds-since-epoch integration variable, so the caller
//! owns the ephemeris and Earth-rotation model rather than this module baking one in.

use super::auxiliary::DsstAuxiliary;
use super::tesseral::DsstTesseral;
use super::{DsstSolarRadiationPressure, DsstThirdBody, DsstZonal};
use crate::driver::{Controls, integrate_dop853};
use crate::{DynamicalSystem, Tolerance};

/// A combined DSST mean-element propagator over the ported conservative contributions.
///
/// `ephemeris(t)` returns the geocentric `(sun, moon)` positions in metres and `gmst(t)` the
/// Greenwich sidereal angle in radians, both as functions of seconds since the initial epoch. The
/// forces are borrowed so the caller keeps ownership and can reuse them across propagations.
pub struct DsstMeanPropagator<'a, E, G> {
    /// Central-body zonal geopotential (J2..Jₙ).
    pub zonal: &'a DsstZonal,
    /// Sun third-body attraction.
    pub sun: &'a DsstThirdBody,
    /// Moon third-body attraction.
    pub moon: &'a DsstThirdBody,
    /// Solar radiation pressure.
    pub srp: &'a DsstSolarRadiationPressure,
    /// Tesseral geopotential (contributes to the mean rate only near a resonance).
    pub tesseral: &'a DsstTesseral,
    /// `t -> (sun_position, moon_position)` in metres, `t` seconds since the initial epoch.
    pub ephemeris: E,
    /// `t -> Greenwich sidereal angle` in radians.
    pub gmst: G,
    /// Central-body gravitational parameter `μ` (m³/s²).
    pub mu: f64,
}

impl<E, G> DsstMeanPropagator<'_, E, G>
where
    E: Fn(f64) -> ([f64; 3], [f64; 3]),
    G: Fn(f64) -> f64,
{
    /// Propagate a MEAN equinoctial state `[a, k, h, q, p, λ]` forward by `dt` seconds.
    ///
    /// # Panics
    ///
    /// If the DOP853 driver fails, which a smooth mean-element ODE with `dt > 0` does not produce
    /// (only non-finite controls or a step underflow would trip it).
    #[must_use]
    #[allow(clippy::expect_used)] // the invariant above holds; a smooth mean-element ODE cannot fail
    pub fn propagate(&self, initial: [f64; 6], dt: f64) -> [f64; 6] {
        if dt == 0.0 {
            return initial;
        }
        // Match DsstZonal::propagate_mean so an only-zonal configuration reproduces it.
        let tol = Tolerance::new(1e-14, 1e-12);
        integrate_dop853(self, 0.0, &initial, dt, 60.0, Controls::new(tol, 1e-6))
            .expect("DSST combined mean propagation")
            .final_state()
    }

    /// Propagate a MEAN state forward by `dt` and return the OSCULATING equinoctial state: the
    /// combined mean propagation plus the sum of each contribution's short-period correction (zonal +
    /// Sun + Moon + SRP + tesseral). The short-period corrections are first-order generating-function
    /// terms and superpose linearly, so `osculating = mean + Σᵢ (oscᵢ(mean) − mean)`. Drag's
    /// short-period is not included (drag is not a mean-rate contribution of this propagator).
    #[must_use]
    pub fn propagate_osculating(&self, initial: [f64; 6], dt: f64) -> [f64; 6] {
        let mean = self.propagate(initial, dt);
        let (sun_pos, moon_pos) = (self.ephemeris)(dt);
        let gmst = (self.gmst)(dt);
        let contributions = [
            self.zonal.osculating(mean),
            self.sun.osculating(mean, sun_pos),
            self.moon.osculating(mean, moon_pos),
            self.srp.osculating(mean, sun_pos),
            self.tesseral.osculating(mean, gmst),
        ];
        let mut osc = mean;
        for c in contributions {
            for i in 0..6 {
                osc[i] += c[i] - mean[i];
            }
        }
        osc
    }
}

impl<E, G> DynamicalSystem<6> for DsstMeanPropagator<'_, E, G>
where
    E: Fn(f64) -> ([f64; 3], [f64; 3]),
    G: Fn(f64) -> f64,
{
    fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let aux = DsstAuxiliary::new(y[0], y[1], y[2], y[3], y[4], y[5], self.mu);
        let (sun_pos, moon_pos) = (self.ephemeris)(t);
        let gmst = (self.gmst)(t);

        let z = self.zonal.mean_element_rates(&aux);
        let s = self.sun.mean_element_rates(&aux, sun_pos);
        let m = self.moon.mean_element_rates(&aux, moon_pos);
        let r = self.srp.mean_element_rates(&aux, sun_pos);
        let te = self.tesseral.mean_element_rates(&aux, gmst);

        for i in 0..6 {
            dydt[i] = z[i] + s[i] + m[i] + r[i] + te[i];
        }
        // Keplerian contribution (DSSTNewtonianAttraction): only the mean longitude advances at n.
        dydt[5] += aux.n;
    }
}

#[cfg(test)]
mod tests {
    use spacelab_testkit::nan_max;

    use super::*;
    use std::vec;

    // Zonal field to degree 6, same as the module-level zonal fixture (Orekit-validated).
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const C20: f64 = -1.08262689216735860e-03;
    const C30: f64 = 2.53209950653743350e-06;
    const C40: f64 = 1.62004577259847830e-06;
    const C50: f64 = 2.27515624171733840e-07;
    const C60: f64 = -5.41217294926390600e-07;
    // WGS84 Earth rotation rate (rad/s), for the tesseral resonance test.
    const OMEGA_EARTH: f64 = 7.292115e-5;

    fn zonal() -> DsstZonal {
        DsstZonal::new(MU, AE, vec![0.0, 0.0, C20, C30, C40, C50, C60])
    }

    /// With every non-central contribution nulled (zero third-body mass, zero area-to-mass, zero
    /// tesseral coefficients), the combined right-hand side collapses to the zonal one, so the
    /// propagation must reproduce the Orekit-validated `DsstZonal::propagate_mean` to integrator
    /// round-off. This certifies the assembly (summation, the Keplerian term, the DOP853 driver)
    /// independently of the individual force validations.
    #[test]
    fn combined_reduces_to_zonal_when_only_zonal_active() {
        let z = zonal();
        let sun = DsstThirdBody::new(MU, 0.0);
        let moon = DsstThirdBody::new(MU, 0.0);
        let srp = DsstSolarRadiationPressure::new(1.5, 0.0, MU, AE);
        let tess = DsstTesseral::new(
            MU,
            AE,
            OMEGA_EARTH,
            vec![vec![0.0; 4]; 4],
            vec![vec![0.0; 4]; 4],
        );
        let prop = DsstMeanPropagator {
            zonal: &z,
            sun: &sun,
            moon: &moon,
            srp: &srp,
            tesseral: &tess,
            ephemeris: |_t| ([1.5e11, 0.0, 0.0], [3.8e8, 0.0, 0.0]),
            gmst: |_t| 0.0,
            mu: MU,
        };
        // The LEO mean state from the zonal fixture.
        let init = [
            7.20000000000000000e+06,
            3.09016994374947470e-03,
            9.51056516295153600e-03,
            8.54890329218965000e-01,
            7.69746709859868000e-01,
            1.43116998663535020e+00,
        ];
        let mut max_a = 0.0f64;
        let mut max_comp = 0.0f64;
        for &t in &[3600.0, 21600.0, 86400.0] {
            let got = prop.propagate(init, t);
            let want = z.propagate_mean(init, t);
            max_a = max_a.max((got[0] - want[0]).abs());
            for i in 1..6 {
                max_comp = max_comp.max((got[i] - want[i]).abs());
            }
            assert!(
                (got[0] - want[0]).abs() < 1e-3,
                "t={t}: a diff {} m",
                got[0] - want[0]
            );
            for i in 1..6 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-9,
                    "t={t}: comp[{i}] diff {}",
                    got[i] - want[i]
                );
            }
        }
        std::println!("combined vs zonal-only: max da {max_a:.2e} m, max comp {max_comp:.2e}");
    }

    /// The only-zonal OSCULATING propagation reproduces the Orekit-validated
    /// `DsstZonal::propagate_osculating` (mean + zonal short-period): the nulled Sun/Moon/SRP/tesseral
    /// add no short-period delta, so `osculating = zonal mean + zonal short-period`. Certifies the
    /// short-period summation independently of the individual short-period force validations.
    #[test]
    fn combined_osculating_reduces_to_zonal() {
        let z = zonal();
        let sun = DsstThirdBody::new(MU, 0.0);
        let moon = DsstThirdBody::new(MU, 0.0);
        let srp = DsstSolarRadiationPressure::new(1.5, 0.0, MU, AE);
        let tess = DsstTesseral::new(
            MU,
            AE,
            OMEGA_EARTH,
            vec![vec![0.0; 4]; 4],
            vec![vec![0.0; 4]; 4],
        );
        let prop = DsstMeanPropagator {
            zonal: &z,
            sun: &sun,
            moon: &moon,
            srp: &srp,
            tesseral: &tess,
            ephemeris: |_t| ([1.5e11, 0.0, 0.0], [3.8e8, 0.0, 0.0]),
            gmst: |_t| 0.0,
            mu: MU,
        };
        let init = [
            7.20000000000000000e+06,
            3.09016994374947470e-03,
            9.51056516295153600e-03,
            8.54890329218965000e-01,
            7.69746709859868000e-01,
            1.43116998663535020e+00,
        ];
        let mut max_da = 0.0f64;
        for &t in &[3600.0, 21600.0, 86400.0] {
            let got = prop.propagate_osculating(init, t);
            let want = z.propagate_osculating(init, t);
            max_da = max_da.max((got[0] - want[0]).abs());
            assert!(
                (got[0] - want[0]).abs() < 1e-3,
                "t={t}: a diff {} m",
                got[0] - want[0]
            );
            for i in 1..6 {
                assert!(
                    (got[i] - want[i]).abs() < 1e-9,
                    "t={t}: comp[{i}] diff {}",
                    got[i] - want[i]
                );
            }
        }
        std::println!("combined osculating vs zonal-only: max da {max_da:.2e} m");
    }

    /// Sun and Moon on coarse circular orbits, and a uniformly rotating Earth. Plain `fn` items so
    /// every propagator below has the same closure types and can go through one helper.
    fn eph(t: f64) -> ([f64; 3], [f64; 3]) {
        let (ws, wm) = (1.99e-7, 2.66e-6);
        (
            [
                1.496e11 * libm::cos(ws * t),
                1.496e11 * libm::sin(ws * t),
                0.0,
            ],
            [3.84e8 * libm::cos(wm * t), 3.84e8 * libm::sin(wm * t), 0.0],
        )
    }

    fn gmst(t: f64) -> f64 {
        OMEGA_EARTH * t
    }

    /// The right-hand side is the sum of the five contributions, each entering once and with a plus.
    ///
    /// Every force is validated on its own against Orekit's `getMeanElementRate`, and the assembled
    /// propagation is validated against `DsstZonal` with the other four nulled — but that
    /// configuration only ever exercises `z[i] + 0 + 0 + 0 + 0`, so which sign each of the four
    /// additions carries, and whether it is an addition at all, was never observed. Turning any one
    /// of them into `-` or `*` survived the whole suite.
    ///
    /// The test removes one force at a time and asks that the derivative moves by exactly that
    /// force's own rate. The oracle is the per-force `mean_element_rates` — the same functions the
    /// individual Orekit fixtures pin — so this asserts the wiring, not the physics. The bound is
    /// the cancellation error of the difference: subtracting two sums of the same five terms leaves
    /// a few ulps of the largest of them, not of the term being recovered.
    #[test]
    fn the_derivative_is_the_sum_of_the_five_contributions() {
        const T: f64 = 4321.0;
        let z = zonal();
        let sun = DsstThirdBody::new(MU, 1.32712440018e20);
        let moon = DsstThirdBody::new(MU, 4.902800066e12);
        let srp = DsstSolarRadiationPressure::new(1.5, 0.02, MU, AE);
        let mut cnm = vec![vec![0.0; 3]; 3];
        let mut snm = vec![vec![0.0; 3]; 3];
        cnm[2][2] = 1.5e-6;
        snm[2][2] = -9.0e-7;
        let tess = DsstTesseral::new(MU, AE, OMEGA_EARTH, cnm, snm);
        // Nulled counterparts: same shapes, no force.
        let z0 = DsstZonal::new(MU, AE, vec![0.0; 7]);
        let tb0 = DsstThirdBody::new(MU, 0.0);
        let srp0 = DsstSolarRadiationPressure::new(1.5, 0.0, MU, AE);
        let tess0 = DsstTesseral::new(
            MU,
            AE,
            OMEGA_EARTH,
            vec![vec![0.0; 3]; 3],
            vec![vec![0.0; 3]; 3],
        );

        let init = [4.2164172e7, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 0.5];
        let deriv = |zonal: &DsstZonal,
                     sun: &DsstThirdBody,
                     moon: &DsstThirdBody,
                     srp: &DsstSolarRadiationPressure,
                     tesseral: &DsstTesseral| {
            let prop = DsstMeanPropagator {
                zonal,
                sun,
                moon,
                srp,
                tesseral,
                ephemeris: eph,
                gmst,
                mu: MU,
            };
            let mut d = [0.0f64; 6];
            prop.derivative(T, &init, &mut d);
            d
        };

        let all = deriv(&z, &sun, &moon, &srp, &tess);
        let aux = DsstAuxiliary::new(init[0], init[1], init[2], init[3], init[4], init[5], MU);
        let (sun_pos, moon_pos) = eph(T);
        let rates = [
            ("zonal", z.mean_element_rates(&aux)),
            ("sun", sun.mean_element_rates(&aux, sun_pos)),
            ("moon", moon.mean_element_rates(&aux, moon_pos)),
            ("srp", srp.mean_element_rates(&aux, sun_pos)),
            ("tesseral", tess.mean_element_rates(&aux, gmst(T))),
        ];
        let without = [
            deriv(&z0, &sun, &moon, &srp, &tess),
            deriv(&z, &tb0, &moon, &srp, &tess),
            deriv(&z, &sun, &tb0, &srp, &tess),
            deriv(&z, &sun, &moon, &srp0, &tess),
            deriv(&z, &sun, &moon, &srp, &tess0),
        ];
        let mut worst = 0.0f64;
        for (k, (name, rate)) in rates.iter().enumerate() {
            for i in 0..6 {
                let recovered = all[i] - without[k][i];
                // The two derivatives also carry the Keplerian `n` in the mean longitude, four
                // orders above any perturbing rate, and it cancels here — so the floor is its ulp,
                // not the recovered term's.
                let scale = (libm::fabs(all[i]) + libm::fabs(without[k][i])).max(f64::MIN_POSITIVE);
                let miss = libm::fabs(recovered - rate[i]) / (f64::EPSILON * scale);
                worst = nan_max(worst, miss);
                assert!(
                    miss < 64.0,
                    "{name} rate[{i}]: removing it moved the derivative by {recovered:e}, \
                     its own rate is {:e} ({miss:.1} ulps of the total)",
                    rate[i]
                );
            }
        }
        // Every contribution must be big enough for its own sign to be observable at all.
        for (name, rate) in &rates {
            let scale: f64 = (0..6).fold(0.0f64, |s, i| s + libm::fabs(rate[i]));
            assert!(scale > 0.0, "{name} contributes nothing at this state");
        }
        std::println!("leave-one-out superposition: worst {worst:.1} ulps of the total rate");
    }

    /// Smoke test: the full force set integrates a near-geostationary orbit stably. GEO is the
    /// canonical DSST case -- the 1:1 tesseral resonance, lunisolar third-body and SRP all act. The
    /// mean semi-major axis is (by construction of the mean elements) free of the short-period
    /// tesseral libration, so it stays within a few km over a day; this checks only that the
    /// assembled RHS is finite and bounded, not a reference trajectory.
    #[test]
    fn combined_runs_under_all_forces_at_geo() {
        let z = zonal();
        let sun = DsstThirdBody::new(MU, 1.32712440018e20);
        let moon = DsstThirdBody::new(MU, 4.902800066e12);
        let srp = DsstSolarRadiationPressure::new(1.5, 0.02, MU, AE);
        // A small unnormalized C22/S22 to exercise the 1:1 resonance path.
        let mut cnm = vec![vec![0.0; 3]; 3];
        let mut snm = vec![vec![0.0; 3]; 3];
        cnm[2][2] = 1.5e-6;
        snm[2][2] = -9.0e-7;
        let tess = DsstTesseral::new(MU, AE, OMEGA_EARTH, cnm, snm);
        let prop = DsstMeanPropagator {
            zonal: &z,
            sun: &sun,
            moon: &moon,
            srp: &srp,
            tesseral: &tess,
            ephemeris: |t| {
                // Coarse circular Sun/Moon motion -- enough to exercise the time-dependent path.
                let ws = 1.99e-7; // ~1 rev/year
                let wm = 2.66e-6; // ~1 rev/27.3 d
                (
                    [
                        1.496e11 * libm::cos(ws * t),
                        1.496e11 * libm::sin(ws * t),
                        0.0,
                    ],
                    [3.84e8 * libm::cos(wm * t), 3.84e8 * libm::sin(wm * t), 0.0],
                )
            },
            gmst: |t| OMEGA_EARTH * t,
            mu: MU,
        };
        // Near-circular, near-equatorial GEO mean state.
        let a0 = 4.2164172e7;
        let init = [a0, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 0.5];
        let got = prop.propagate(init, 86400.0);
        assert!(
            got.iter().all(|v| v.is_finite()),
            "non-finite state: {got:?}"
        );
        let da_rel = (got[0] - a0).abs() / a0;
        assert!(da_rel < 1e-2, "mean a drifted {da_rel:.3e} over 24 h");
        std::println!("GEO all-forces 24 h: a drift {da_rel:.3e} rel, state {got:?}");
    }
}
