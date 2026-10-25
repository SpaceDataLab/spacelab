//! The DSST **atmospheric drag** mean-element rates (numerically averaged).
//!
//! Direct port of Orekit's `DSSTAtmosphericDrag` fed into the shared Gauss-quadrature averaging of
//! [`super::gaussian`]. Restricted to the analytic **mean-element rates**; the short-period
//! corrections are out of scope.
//!
//! The perturbing acceleration is `DragForce.acceleration` with an `IsotropicDrag` cannonball:
//! `q = ½·ρ·(Cd·A/m)·|v_rel|·v_rel`, where `v_rel = v_atm − v_sat` is the velocity of the satellite
//! **relative to the atmosphere**. The atmosphere co-rotates with the central body, so `v_atm` is
//! non-zero even for a "static" model (Orekit's `Atmosphere.getVelocity` transforms a body-frame
//! zero-velocity point back to the inertial frame, i.e. `v_atm = ω_body × r`). This co-rotation is
//! the one piece SRP did not need; here it enters through the caller-supplied [`Atmosphere`].
//!
//! # Integration limits (`getLLimits`)
//!
//! Drag has **no shadow**: the quadrature runs the full `[-π + L_v, π + L_v]` when the whole orbit
//! lies inside the atmosphere (`apogee < rbar`). Orekit's three branches are reproduced faithfully —
//! trajectory entirely out of the atmosphere (`perigee > rbar` ⇒ skip), entirely within (full circle)
//! and partially within (the `acos` entry/exit true-longitude arc). For a LEO with `apogee < rbar`
//! (the validated case) it is the full circle, so no umbra/shadow arithmetic is involved.
//!
//! # Scoping: the density + velocity law is the caller's
//!
//! Orekit gets `ρ` and `v_atm` from an `org.orekit.models.earth.atmosphere.Atmosphere`
//! (`getDensity`/`getVelocity`), which for a realistic model (NRLMSISE-00, DTM, Harris-Priester)
//! couples in a full ITRF transform and a density table. To keep this module a pure mean-element-rate
//! engine (and the gate reproducible), the density + co-rotation law is injected via the
//! [`Atmosphere`] trait, evaluated at the inertial node position. The NRLMSISE-00 / body-frame
//! coupling is therefore the caller's responsibility, exactly as `propagator.rs` takes ephemeris and
//! GMST as closures. The validation gate drives a constant-density, purely co-rotating atmosphere
//! (`ρ = const`, `v_atm = ω·ẑ × r`) on both the Java and Rust sides.
//!
//! # Checked against the average that defines the rates
//!
//! `tests/dsst_short_period_identity.rs` checks [`DsstAtmosphericDrag::mean_element_rates`] against
//! `(1/2π)∫F dλ`, taken by Simpson's rule in the mean longitude. F is Gauss's rate from a drag law and
//! an exponential co-rotating atmosphere written in the test. The arc below `rbar` comes from a
//! bisection on |r|, not from the conic. Measured 1.8e-10 for an orbit inside the atmosphere and
//! 2.2e-10 for one with e = 0.07 that leaves it.

use libm::{acos, atan2};

use super::auxiliary::DsstAuxiliary;
use super::gaussian::{self, NodeState};
use super::gaussian_short_period;
use super::srp::{mean_to_true, normalize_angle};

/// Default upper altitude for atmospheric drag (m) — `DSSTAtmosphericDrag.DEFAULT_MAX_ATMOSPHERE_ALTITUDE`.
pub const DEFAULT_MAX_ATMOSPHERE_ALTITUDE: f64 = 1_000_000.0;

/// WGS84 Earth equatorial radius (m) — `Constants.WGS84_EARTH_EQUATORIAL_RADIUS`, used for the default
/// critical distance `rbar`.
pub const WGS84_EARTH_EQUATORIAL_RADIUS: f64 = 6_378_137.0;

/// Threshold for the Gauss-quadrature order selection — `DSSTAtmosphericDrag.GAUSS_THRESHOLD`.
pub const GAUSS_THRESHOLD: f64 = 6.0e-10;

/// Local density and co-rotating atmosphere velocity at an inertial position — the reduced form of
/// Orekit's `org.orekit.models.earth.atmosphere.Atmosphere` interface the DSST mean-rate path needs.
///
/// `getDensity(date, position, frame)` and `getVelocity(date, position, frame)` become [`Self::density`]
/// and [`Self::velocity`] with the date and frame fixed at the propagation epoch/frame. The caller owns
/// the density law and the co-rotation kinematics (`v_atm = ω_body × r`); a realistic model plugs a
/// NRLMSISE-00 / DTM table plus the ITRF transform behind this trait.
pub trait Atmosphere {
    /// Local atmospheric density `ρ` at the inertial `position` (kg/m³).
    fn density(&self, position: [f64; 3]) -> f64;

    /// Inertial velocity of the atmosphere molecules at `position` (m/s) — for a co-rotating
    /// atmosphere `ω_body × r`, defined in the same frame as `position`.
    fn velocity(&self, position: [f64; 3]) -> [f64; 3];
}

/// Atmospheric-drag contribution to the DSST mean-element rates (cannonball spacecraft).
///
/// Holds the drag coefficient `Cd`, the area-to-mass ratio `S/m` (m²/kg), the central attraction `µ`,
/// and the critical distance `rbar` above which the atmosphere is neglected. Matches Orekit's
/// `DSSTAtmosphericDrag` built from an `IsotropicDrag(area, cd)`: the acceleration scale is
/// `dragCoeff · (S/m)` with `dragCoeff = globalDragFactor · Cd = Cd`.
#[derive(Debug, Clone, Copy)]
pub struct DsstAtmosphericDrag {
    /// Drag coefficient `Cd`.
    pub cd: f64,
    /// Area-to-mass ratio `S/m`, m²/kg.
    pub area_over_mass: f64,
    /// Central attraction coefficient `µ`, m³/s².
    pub mu: f64,
    /// Critical distance from the central-body centre beyond which drag is neglected (m) —
    /// Orekit's `rbar`.
    pub rbar: f64,
}

impl DsstAtmosphericDrag {
    /// Build with Orekit's default critical distance
    /// (`DEFAULT_MAX_ATMOSPHERE_ALTITUDE + WGS84_EARTH_EQUATORIAL_RADIUS`).
    #[must_use]
    pub fn new(cd: f64, area_over_mass: f64, mu: f64) -> Self {
        Self {
            cd,
            area_over_mass,
            mu,
            rbar: DEFAULT_MAX_ATMOSPHERE_ALTITUDE + WGS84_EARTH_EQUATORIAL_RADIUS,
        }
    }

    /// Build with a custom critical distance `rbar` (Orekit's `setRbar`).
    #[must_use]
    pub fn with_rbar(cd: f64, area_over_mass: f64, mu: f64, rbar: f64) -> Self {
        Self {
            cd,
            area_over_mass,
            mu,
            rbar,
        }
    }

    /// The six mean-element rates `[da, dk, dh, dq, dp, dλ]` (Orekit equinoctial ordering) at the given
    /// auxiliary state, with the density + co-rotation law supplied by `atmosphere`.
    ///
    /// Mirrors a fixed-state Orekit `getMeanElementRate`: the integration limits are set by
    /// [`Self::l_limits`], then the drag acceleration is averaged by order-48 Gauss quadrature. Returns
    /// zeros when the whole orbit is above the atmosphere (`L_low ≥ L_high`).
    #[must_use]
    pub fn mean_element_rates<A: Atmosphere>(
        &self,
        aux: &DsstAuxiliary,
        atmosphere: &A,
    ) -> [f64; 6] {
        let ll = self.l_limits(aux);
        let accel = self.acceleration(atmosphere);
        gaussian::mean_element_rates(aux, ll[0], ll[1], accel)
    }

    /// The Gauss-quadrature order Orekit would fix for this state (`getMeanElementRate`'s `isDirty`
    /// selection). The returned mean rate is always the order-48 result; a propagation caches this
    /// order for later steps. Exposed for validation against Orekit.
    #[must_use]
    pub fn gauss_order<A: Atmosphere>(&self, aux: &DsstAuxiliary, atmosphere: &A) -> usize {
        let ll = self.l_limits(aux);
        let accel = self.acceleration(atmosphere);
        gaussian::select_order(aux, ll[0], ll[1], accel, GAUSS_THRESHOLD)
    }

    /// The osculating equinoctial state `mean + short-period` for a **mean** equinoctial state
    /// `[a, k, h, q, p, λ]` and the density + co-rotation law supplied by `atmosphere`. Mirrors
    /// [`super::third_body::DsstThirdBody::osculating`]: the same atmospheric-arc limits the mean rates
    /// use feed the shared Gaussian short-period engine
    /// (`gaussian_short_period::short_period_correction`).
    #[must_use]
    pub fn osculating<A: Atmosphere>(&self, mean: [f64; 6], atmosphere: &A) -> [f64; 6] {
        let aux = DsstAuxiliary::new(
            mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], self.mu,
        );
        let ll = self.l_limits(&aux);
        let sp =
            gaussian_short_period::short_period_correction(&aux, ll, self.acceleration(atmosphere));
        let mut osc = mean;
        for i in 0..6 {
            osc[i] += sp[i];
        }
        osc
    }

    /// The per-node drag acceleration closure `q(node) = ½·ρ·dragCoeff·(S/m)·|v_rel|·v_rel`, with
    /// `v_rel = v_atm − v_sat` (Orekit's `DragForce.acceleration` + `IsotropicDrag.dragAcceleration`).
    fn acceleration<'a, A: Atmosphere>(
        &self,
        atmosphere: &'a A,
    ) -> impl Fn(&NodeState) -> [f64; 3] + 'a {
        // IsotropicDrag: dragCoeff = globalDragFactor(1) · Cd = Cd; the acceleration scale is
        // dragCoeff · crossSection / (2·mass) = Cd · (S/m) / 2.
        let scale_sc = self.cd * self.area_over_mass;
        move |node: &NodeState| -> [f64; 3] {
            let rho = atmosphere.density(node.pos);
            let v_atm = atmosphere.velocity(node.pos);
            // relativeVelocity = vAtm − vSat (Orekit's DragForce sign convention).
            let rel = [
                v_atm[0] - node.vel[0],
                v_atm[1] - node.vel[1],
                v_atm[2] - node.vel[2],
            ];
            let s = gaussian::norm(rel) * rho * scale_sc / 2.0;
            [s * rel[0], s * rel[1], s * rel[2]]
        }
    }

    /// Integration limits `[L_low, L_high]` in true longitude — Orekit's `getLLimits`.
    ///
    /// Three branches, faithful to Orekit: perigee above `rbar` ⇒ `[0, 0]` (the whole orbit is above
    /// the atmosphere, quadrature skipped); apogee below `rbar` ⇒ the full circle `[-π + L_v, π + L_v]`
    /// (the whole orbit is inside); otherwise the atmospheric arc `[ω − f_b, ω + f_b]` with
    /// `f_b = acos(((a(1−e²)/rbar) − 1)/e)` and `ω = atan2(h, k)`.
    #[must_use]
    pub fn l_limits(&self, aux: &DsstAuxiliary) -> [f64; 2] {
        use core::f64::consts::PI;

        let perigee = aux.sma * (1.0 - aux.ecc);
        // Trajectory entirely out of the atmosphere.
        if perigee > self.rbar {
            return [0.0, 0.0];
        }
        let apogee = aux.sma * (1.0 + aux.ecc);
        // Trajectory entirely within the atmosphere: full circle centred on the true longitude.
        if apogee < self.rbar {
            let lvn = normalize_angle(mean_to_true(aux.k, aux.h, aux.lambda), 0.0);
            return [-PI + lvn, PI + lvn];
        }
        // Trajectory partially within the atmosphere: the arc between atmosphere entry and exit.
        // The cosine is mathematically in `[-1, 1]` on this branch — it reaches an endpoint exactly at
        // grazing tangency (`rbar = a(1 ∓ e)`) — but the f64 evaluation overshoots there: at
        // `a = 7.2e6, e = 0.05, rbar = a(1 − e)` it lands on `1.0000000000000009`, and an unclamped
        // `acos` then returns NaN for both limits, which the quadrature carries into every rate.
        let cos_fb = ((aux.sma * (1.0 - aux.ecc * aux.ecc) / self.rbar) - 1.0) / aux.ecc;
        let fb = acos(cos_fb.clamp(-1.0, 1.0));
        let ww = atan2(aux.h, aux.k);
        [ww - fb, ww + fb]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`, orekit_dsst_drag.csv): a cannonball
    // IsotropicDrag (cd = 2.2, area = 10 m², mass = 1000 kg ⇒ S/m = 0.01) on a MEAN equinoctial LEO
    // (a = 7.0e6, e = 0.01, i = 98°, where apogee < rbar so the whole orbit is inside the atmosphere
    // and the quadrature is the full circle). The atmosphere is a CONSTANT-density, purely co-rotating
    // model: ρ = RHO everywhere and v_atm = ω·ẑ × r (a fixed ω about +Z, the ITRF/GMST rotation reduced
    // to a pure Z-spin, driven identically on both sides). This isolates the co-rotation kinematics +
    // Gauss quadrature with no NRLMSISE-00 or frame table. Direct fixed-state getMeanElementRate, plus
    // the integration limits getLLimits used and the Gauss order the isDirty pass selected.
    const MU: f64 = 3.98600441500000000e+14;
    const CD: f64 = 2.20000000000000000e+00;
    const AREA: f64 = 1.00000000000000000e+01;
    const MASS: f64 = 1.00000000000000000e+03;
    const RHO: f64 = 1.00000000000000004e-12;
    const OMEGA: f64 = 7.29211514670697900e-05;
    const RBAR: f64 = 7.37813700000000000e+06;

    const GAUSS_ORDER: usize = 12;
    const LL_LOW: f64 = -1.70690644724420150e+00;
    const LL_HIGH: f64 = 4.57627885993538450e+00;

    // Initial MEAN equinoctial state (a, ex=k, ey=h, hx=q, hy=p, lM=λ).
    const A0: f64 = 7.00000000000000000e+06;
    const EX0: f64 = 3.09016994374947470e-03;
    const EY0: f64 = 9.51056516295153600e-03;
    const HX0: f64 = 8.54890329218965000e-01;
    const HY0: f64 = 7.69746709859868000e-01;
    const LM0: f64 = 1.43116998663535020e+00;

    // getMeanElementRate: [da, dk, dh, dq, dp, dM].
    const RATES: [f64; 6] = [
        -1.18546389946857460e-03,
        -2.47235538001681400e-13,
        -7.67429932128747400e-13,
        -2.42676254946154250e-12,
        -2.18560101295547930e-12,
        -3.79325964131381240e-16,
    ];

    /// A constant-density, purely co-rotating atmosphere: `ρ = RHO` everywhere and `v_atm = ω·ẑ × r`.
    /// The cross product is Hipparchus `Vector3D.crossProduct((0,0,ω), r)` (extended-precision
    /// `linearCombination`), so the Rust and Java velocities are bit-identical.
    struct CoRotatingConstant {
        rho: f64,
        omega: f64,
    }

    impl Atmosphere for CoRotatingConstant {
        fn density(&self, _position: [f64; 3]) -> f64 {
            self.rho
        }
        fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
            gaussian::cross([0.0, 0.0, self.omega], position)
        }
    }

    fn drag() -> DsstAtmosphericDrag {
        DsstAtmosphericDrag::with_rbar(CD, AREA / MASS, MU, RBAR)
    }

    fn aux() -> DsstAuxiliary {
        DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU)
    }

    fn atmosphere() -> CoRotatingConstant {
        CoRotatingConstant {
            rho: RHO,
            omega: OMEGA,
        }
    }

    /// The full-circle integration limits `[-π + L_v, π + L_v]` the port computes must match Orekit's
    /// `getLLimits` (which then feeds the quadrature) — else a different 2π window is averaged and the
    /// discrete Gauss nodes land elsewhere.
    #[test]
    fn integration_limits_match_orekit() {
        let ll = drag().l_limits(&aux());
        let e_low = (ll[0] - LL_LOW).abs();
        let e_high = (ll[1] - LL_HIGH).abs();
        assert!(
            e_low < 1e-12,
            "L_low got {} want {LL_LOW} abs {e_low:.3e}",
            ll[0]
        );
        assert!(
            e_high < 1e-12,
            "L_high got {} want {LL_HIGH} abs {e_high:.3e}",
            ll[1]
        );
        std::println!("drag integration-limit abs errors: low {e_low:.3e}, high {e_high:.3e}");
    }

    /// The three branches of `l_limits`, against the conic rather than against a golden pair. Orekit's
    /// fixture is fully immersed (`apogee = 7.07e6 < rbar = 7.378e6`), so it exercises the full-circle
    /// arm alone and pins nothing about the entry/exit arc or either branch boundary. The oracle is the
    /// polar equation `r(L) = a(1 − k² − h²)/(1 + k cos L + h sin L)`: whatever `f_b` is, the two
    /// returned longitudes have to be where the radius crosses `rbar`, with the interior below it. (G2)
    ///
    /// Both boundaries are struck exactly — `rbar` is built from the fixture's own `a` and `e` — which
    /// is where the `acos` argument leaves `[-1, 1]` by 9e-16 and used to return NaN for both limits.
    #[test]
    #[allow(clippy::float_cmp)] // the branch returns are exact: `ww ∓ 0.0`, `ww ∓ π`, `0.0`
    fn the_integration_limits_are_the_atmospheric_arc() {
        use core::f64::consts::PI;
        use libm::{cos, fabs, sin};
        use spacelab_testkit::nan_max;

        let radius = |aux: &DsstAuxiliary, l: f64| {
            aux.sma * (1.0 - aux.ecc * aux.ecc) / (1.0 + aux.k * cos(l) + aux.h * sin(l))
        };
        let low = DsstAuxiliary::new(7.2e6, 0.04, 0.03, HX0, HY0, 3.0, MU);
        let ww = atan2(low.h, low.k);

        // Entirely above the atmosphere: the quadrature is skipped.
        let high = DsstAuxiliary::new(4.2164e7, 1.0e-3, -5.0e-4, HX0, HY0, 3.0, MU);
        assert_eq!(drag().l_limits(&high), [0.0, 0.0], "perigee above rbar");

        // Partially within: the arc from atmosphere entry to exit.
        let ll = drag().l_limits(&low);
        let mut worst = 0.0_f64;
        for l in ll {
            worst = nan_max(worst, fabs(radius(&low, l) - RBAR) / RBAR);
        }
        assert!(
            worst < 1e-15,
            "arc ends off rbar by {worst:.3e} relative, {ll:?}"
        );
        assert!(
            radius(&low, 0.5 * (ll[0] + ll[1])) < RBAR && radius(&low, ww + PI) > RBAR,
            "the arc must straddle perigee, not apogee"
        );
        std::println!("drag atmospheric-arc endpoint rel error: {worst:.3e}");

        // Grazing at perigee: a zero-width arc there, not the `[0, 0]` skip.
        let graze = DsstAtmosphericDrag::with_rbar(CD, AREA / MASS, MU, low.sma * (1.0 - low.ecc));
        assert_eq!(graze.l_limits(&low), [ww, ww], "grazing perigee");

        // Grazing at apogee: the full circle centred on perigee, not on the current true longitude.
        let dip = DsstAtmosphericDrag::with_rbar(CD, AREA / MASS, MU, low.sma * (1.0 + low.ecc));
        assert_eq!(dip.l_limits(&low), [ww - PI, ww + PI], "grazing apogee");
    }

    /// The Gauss-order the port selects must equal Orekit's cached order (drag's smooth full-circle
    /// integrand converges early, unlike SRP's shadow-clipped one).
    #[test]
    fn gauss_order_matches_orekit() {
        assert_eq!(drag().gauss_order(&aux(), &atmosphere()), GAUSS_ORDER);
    }

    /// The six fixed-state drag mean-element rates match Orekit's `getMeanElementRate` to < 1e-9
    /// relative. Unlike SRP, drag has no shadow terminator and no near-zero `da` cancellation: drag
    /// removes energy secularly, so the dominant `da` is well-conditioned and matches **bit-exactly**
    /// (0.0 relative); the equinoctial e/i components agree to ~4e-14 and the tiny along-track `dm`
    /// (~-3.8e-16 rad/s, round-off level) to 1.6e-11. The dominant `da` is additionally cross-checked
    /// **independently of Orekit** against a first-principles numerical orbit-average of the Gauss
    /// `a`-equation (200k mean-anomaly samples): -1.185464e-3 m/s, agreeing to 3.5e-15 relative — so
    /// this validates Orekit's own value, not merely that the port reproduces it.
    #[test]
    fn fixed_state_rates_match_orekit() {
        let got = drag().mean_element_rates(&aux(), &atmosphere());
        let mut errs = [0.0_f64; 6];
        for i in 0..6 {
            errs[i] = if RATES[i] == 0.0 {
                got[i].abs()
            } else {
                (got[i] - RATES[i]).abs() / RATES[i].abs()
            };
        }
        std::println!(
            "per-component drag rate rel errors: da {:.3e} dk {:.3e} dh {:.3e} dq {:.3e} dp {:.3e} dm {:.3e}",
            errs[0],
            errs[1],
            errs[2],
            errs[3],
            errs[4],
            errs[5]
        );
        // The doc comment above claims `da` matches bit-exactly, and nothing enforced it: the `1e-9`
        // bound below leaves seven orders of magnitude of slack in the one component that has none.
        // Kahan-compensating the quadrature the wrong way lands here at 1.829e-16 relative.
        assert!(
            errs[0] == 0.0,
            "da is documented as bit-exact against Orekit; got {} want {} rel {:.3e}",
            got[0],
            RATES[0],
            errs[0]
        );
        for i in 0..6 {
            assert!(
                errs[i] < 1e-9,
                "rate[{i}] got {} want {} rel {:.3e}",
                got[i],
                RATES[i],
                errs[i]
            );
        }
    }
}
