//! The DSST **solar radiation pressure** mean-element rates (cannonball SRP, numerically averaged).
//!
//! Direct port of Orekit's `DSSTSolarRadiationPressure` fed into the shared Gauss-quadrature
//! averaging of [`super::gaussian`]. Restricted to the analytic **mean-element rates**; the
//! short-period corrections are out of scope.
//!
//! The perturbing acceleration is the cannonball model of `SolarRadiationPressure`:
//! `q = C_R·(S/m)·λ·P_ref·d_ref²·(r − r☉)/|r − r☉|³`, where `λ ∈ [0, 1]` is the lighting ratio
//! (Sun visibility). Two shadow models appear, exactly as in Orekit:
//! * **Integration limits** (`getLLimits`) restrict the quadrature to the sunlit true-longitude arc
//!   `[L_exit, L_entry]` using the **cylindrical** Earth-shadow (the quartic in `cos L`,
//!   Danielson 3.5); and
//! * **the acceleration** weights each node by the **conical** umbra/penumbra lighting ratio,
//!   [`crate::shadow::lighting_ratio`]: the lens without cancellation, where Orekit's `acos`
//!   arrangement of the same area is up to 6.5e-7 off near a tangency (`shadow.rs`).
//!
//! # Independent validation
//!
//! Beyond the Orekit gate, the mean rates are cross-checked against a first-principles numerical
//! orbit-average that shares no code with this port or Orekit
//! (`experiments/dsst_srp_independent_verify.py`, the same `∂(element)/∂v · a` scheme as the
//! third-body check). An independent *geometric* cylindrical shadow reproduces Orekit's `getLLimits`
//! sunlit arc to ~3e-4 rad; the dominant `dk/dh/dq/dp/dλ` agree to 0.09–0.3 % and `da` (the
//! terminator-sensitive near-cancellation) to ~1.4 % — the residual being precisely the conical
//! penumbra refinement this port applies but the binary-cylindrical check omits.
//!
//! # Scoping: spherical central body
//!
//! Orekit's conical lighting ratio uses `OneAxisEllipsoid.pointOnLimb`, whose limb radius depends on
//! the body-frame (ITRF) orientation for a flattened ellipsoid. Built with **flattening = 0** (a
//! sphere of radius `ae`, the oracle's configuration) the occultation angles are rotation-invariant,
//! so the limb radius reduces to `ρ_occ = asin(ae / |r|)` and the whole ratio is computable directly
//! from the inertial (GCRF) satellite and Sun positions — no Earth-orientation transform. This is the
//! only deviation from the general model and it is what the oracle validates against.
//!
//! # Checked against the average that defines the rates
//!
//! `tests/dsst_short_period_identity.rs` checks [`DsstSolarRadiationPressure::mean_element_rates`]
//! against `(1/2π)∫νF dλ`, taken by Simpson's rule in the mean longitude. F is Gauss's rate from a
//! cannonball law and `ν` the lens of the two discs, both written in the test, and the shadow
//! crossings are bisections, not the quartic. Measured 1.9e-10 with no shadow, and 3.8e-5 over the
//! arc outside the cylinder, where the order-48 Gauss rule steps across the corner of `ν`. Against
//! the average over the whole orbit the rates are 1.6e-3 off: the cylinder cut drops the light of
//! the inner half of each penumbra and keeps the deficit of the outer half, where the cylinder
//! alone, `ν = 1` on its arc, is 4.2e-5 off.

use libm::{atan, atan2, cbrt, copysign, cos, fabs, floor, sin, sqrt};

use super::auxiliary::DsstAuxiliary;
use super::gaussian::{self, NodeState};
use super::gaussian_short_period;

/// Default reference distance for the solar radiation pressure (m) — `DSSTSolarRadiationPressure.D_REF`.
pub const D_REF: f64 = 149_597_870_000.0;

/// Default reference solar radiation pressure at [`D_REF`] (N/m²) — `DSSTSolarRadiationPressure.P_REF`.
pub const P_REF: f64 = 4.56e-6;

/// Sun radius used for the occultation apparent radius (m) — `Constants.SUN_RADIUS`.
pub const SUN_RADIUS: f64 = 6.957e8;

/// Threshold for the shadow equation `|S| < S_ZERO` (`DSSTSolarRadiationPressure.S_ZERO`).
const S_ZERO: f64 = 1.0e-6;

/// Solar-radiation-pressure contribution to the DSST mean-element rates (cannonball spacecraft).
///
/// Holds the reflection coefficient `cr`, the area-to-mass ratio `S/m` (m²/kg), the reference
/// pressure/distance `P_ref`/`d_ref`, the central attraction `µ`, and the central-body equatorial
/// radius `ae` (for the Earth-shadow). Matches Orekit's `IsotropicRadiationSingleCoefficient` with a
/// unit global factor: the acceleration scale is `cr · (S/m)`.
#[derive(Debug, Clone, Copy)]
pub struct DsstSolarRadiationPressure {
    /// Radiation pressure coefficient `C_R`.
    pub cr: f64,
    /// Area-to-mass ratio `S/m`, m²/kg.
    pub area_over_mass: f64,
    /// Reference solar radiation pressure at `d_ref`, N/m².
    pub p_ref: f64,
    /// Reference distance, m.
    pub d_ref: f64,
    /// Central attraction coefficient `µ`, m³/s².
    pub mu: f64,
    /// Central-body equatorial radius `ae`, m (Earth-shadow geometry).
    pub ae: f64,
}

impl DsstSolarRadiationPressure {
    /// Build with Orekit's default reference values ([`D_REF`], [`P_REF`]).
    #[must_use]
    pub fn new(cr: f64, area_over_mass: f64, mu: f64, ae: f64) -> Self {
        Self {
            cr,
            area_over_mass,
            p_ref: P_REF,
            d_ref: D_REF,
            mu,
            ae,
        }
    }

    /// Build with custom reference pressure/distance.
    #[must_use]
    pub fn with_reference(
        cr: f64,
        area_over_mass: f64,
        mu: f64,
        ae: f64,
        d_ref: f64,
        p_ref: f64,
    ) -> Self {
        Self {
            cr,
            area_over_mass,
            p_ref,
            d_ref,
            mu,
            ae,
        }
    }

    /// The six mean-element rates `[da, dk, dh, dq, dp, dλ]` (Orekit equinoctial ordering) at the
    /// given auxiliary state, with the Sun at geocentric inertial position `sun_pos` (m).
    ///
    /// Mirrors a fixed-state Orekit `getMeanElementRate`: the sunlit-arc limits are set by the
    /// cylindrical shadow, then the cannonball acceleration (conical lighting ratio) is averaged by
    /// order-48 Gauss quadrature. Returns zeros when the whole orbit is eclipsed (`L_exit ≥ L_entry`).
    #[must_use]
    pub fn mean_element_rates(&self, aux: &DsstAuxiliary, sun_pos: [f64; 3]) -> [f64; 6] {
        let ll = self.l_limits(aux, sun_pos);
        gaussian::mean_element_rates(aux, ll[0], ll[1], self.acceleration(sun_pos))
    }

    /// The per-node cannonball SRP acceleration closure `q(node) = scale·λ·k_ref·(r − r☉)/|r − r☉|³`,
    /// with `scale = cr·(S/m)`, `k_ref = P_ref·d_ref²` and `λ` the conical lighting ratio
    /// ([`crate::shadow::lighting_ratio`]) — Orekit's `SolarRadiationPressure.acceleration` +
    /// `IsotropicRadiationSingleCoefficient`. Shared by the mean-rate averaging, the Gauss-order
    /// selection and the short-period Fourier quadrature so all three see the identical acceleration.
    fn acceleration(&self, sun_pos: [f64; 3]) -> impl Fn(&NodeState) -> [f64; 3] {
        let k_ref = self.p_ref * self.d_ref * self.d_ref;
        // IsotropicRadiationSingleCoefficient: (global=1)·area·cr / mass = cr·(S/m).
        let scale_sc = self.cr * self.area_over_mass;
        let ae = self.ae;
        move |node: &NodeState| -> [f64; 3] {
            let sun_sat = [
                node.pos[0] - sun_pos[0],
                node.pos[1] - sun_pos[1],
                node.pos[2] - sun_pos[2],
            ];
            let r2 = gaussian::dot(sun_sat, sun_sat);
            let ratio = crate::shadow::lighting_ratio(node.pos, sun_pos, ae);
            let raw_p = ratio * k_ref / r2;
            let flux_scalar = raw_p / sqrt(r2);
            let flux = [
                flux_scalar * sun_sat[0],
                flux_scalar * sun_sat[1],
                flux_scalar * sun_sat[2],
            ];
            [scale_sc * flux[0], scale_sc * flux[1], scale_sc * flux[2]]
        }
    }

    /// The Gauss-quadrature order Orekit would fix for this state (`getMeanElementRate`'s `isDirty`
    /// selection): the returned mean rate is always the order-48 result, but a propagation caches this
    /// order for later steps. Exposed for validation against Orekit.
    #[must_use]
    pub fn gauss_order(&self, aux: &DsstAuxiliary, sun_pos: [f64; 3]) -> usize {
        let ll = self.l_limits(aux, sun_pos);
        // GAUSS_THRESHOLD for SRP (DSSTSolarRadiationPressure.GAUSS_THRESHOLD).
        gaussian::select_order(aux, ll[0], ll[1], self.acceleration(sun_pos), 1.0e-15)
    }

    /// The osculating equinoctial state `mean + short-period` for a **mean** equinoctial state
    /// `[a, k, h, q, p, λ]` and this SRP model's Sun position `sun_pos` (m). Mirrors
    /// [`super::third_body::DsstThirdBody::osculating`]: the same cylindrical-shadow limits the mean
    /// rates use feed the shared Gaussian short-period engine
    /// (`gaussian_short_period::short_period_correction`).
    #[must_use]
    pub fn osculating(&self, mean: [f64; 6], sun_pos: [f64; 3]) -> [f64; 6] {
        let aux = DsstAuxiliary::new(
            mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], self.mu,
        );
        let ll = self.l_limits(&aux, sun_pos);
        let sp =
            gaussian_short_period::short_period_correction(&aux, ll, self.acceleration(sun_pos));
        let mut osc = mean;
        for i in 0..6 {
            osc[i] += sp[i];
        }
        osc
    }

    /// Integration limits `[L_exit, L_entry]` in true longitude — Orekit's `getLLimits`.
    ///
    /// Without an eclipse the bounds are the full orbit `[-π + L_v, π + L_v]`. When the perigee is
    /// close enough to enter the **cylindrical** shadow, the quartic in `cos L` (Danielson 3.5-(2)) is
    /// solved and the sunlit arc `[L_exit, L_entry]` returned instead.
    #[must_use]
    pub fn l_limits(&self, aux: &DsstAuxiliary, sun_pos: [f64; 3]) -> [f64; 2] {
        use core::f64::consts::PI;

        // Default bounds without shadow: [-π, π] shifted by the normalized true longitude.
        let lv = mean_to_true(aux.k, aux.h, aux.lambda);
        let lvn = normalize_angle(lv, 0.0);
        let mut ll = [-PI + lvn, PI + lvn];

        // Direction cosines of the Sun in the equinoctial frame.
        let sun_dir = normalize(sun_pos);
        let alpha = gaussian::dot_ext(sun_dir, aux.f_vec);
        let beta = gaussian::dot_ext(sun_dir, aux.g_vec);
        let gamma = gaussian::dot_ext(sun_dir, aux.w_vec);

        // Only compute limits if the perigee is close enough to be in the shadow.
        if fabs(gamma * aux.sma * (1.0 - aux.ecc)) < self.ae {
            // Coefficients of the quartic equation in cos(L) (Danielson 3.5-(2)).
            let bet2 = beta * beta;
            let h2 = aux.h * aux.h;
            let k2 = aux.k * aux.k;
            // `m = ae/p` with `p = a·B²` the semi-latus rectum. That is Escobal (1976) eq. (A.1)
            // and its equinoctial form in Valk & Lemaître (2008), `ae²(1 + k cos L + h sin L)² +
            // p² cos²Φ − p² = 0`, and it is what Danielson's own eq. 2.1.4-6
            // `r = a(1 − h² − k²)/(1 + h sin L + k cos L)` forces. Danielson 3.5 nevertheless
            // prints `M = ae²/(a²(1 − h² − k²))`, one power of `B²` short, and Orekit transcribes
            // that as `ae / (a * B)` — a cylinder of radius `ae·B`, narrowing with eccentricity.
            // Measured at Orekit's own golden limits for the LEO fixture, the perpendicular
            // distance to the Sun line is `6 377 817.545 m = ae·B`, not `ae = 6 378 136.46 m`; the
            // sunlit arc that costs is `5.016238 → 4.412925 rad` on a GTO (`e = 0.731`), and
            // `+5.2 %` on `da`.
            let m = self.ae / (aux.sma * aux.big_b * aux.big_b);
            let m2 = m * m;
            let m4 = m2 * m2;
            let bb = alpha * beta + m2 * aux.h * aux.k;
            let b2 = bb * bb;
            let cc = alpha * alpha - bet2 + m2 * (k2 - h2);
            let dd = 1.0 - bet2 - m2 * (1.0 + h2);
            let a = [
                4.0 * b2 + cc * cc,
                8.0 * bb * m2 * aux.h + 4.0 * cc * m2 * aux.k,
                -4.0 * b2 + 4.0 * m4 * h2 - 2.0 * cc * dd + 4.0 * m4 * k2,
                -8.0 * bb * m2 * aux.h - 4.0 * dd * m2 * aux.k,
                -4.0 * m4 * h2 + dd * dd,
            ];
            let mut roots = [0.0_f64; 4];
            let nb_roots = real_quartic_roots(&a, &mut roots);
            if nb_roots > 0 {
                let mut entry_found = false;
                let mut exit_found = false;
                // `S` and half its derivative at a true longitude — the shadow equation 3.5-1 in the
                // variable the answer is actually wanted in.
                let shadow_at = |l: f64| {
                    let (sin_l, cos_l) = (sin(l), cos(l));
                    let c_phi = alpha * cos_l + beta * sin_l;
                    let range = 1.0 + aux.k * cos_l + aux.h * sin_l;
                    let s = 1.0 - m2 * range * range - c_phi * c_phi;
                    let half_ds_dl = m2 * range * (aux.k * sin_l - aux.h * cos_l)
                        + c_phi * (alpha * sin_l - beta * cos_l);
                    (c_phi, s, half_ds_dl)
                };
                for &cos_l in roots.iter().take(nb_roots) {
                    let s_l = sqrt((1.0 - cos_l) * (1.0 + cos_l));
                    // Check both angles: L and -L.
                    for j in [-1.0_f64, 1.0] {
                        // Refine `L` before testing it. Eliminating `sin L` collapses the two
                        // crossings onto one variable, so wherever they are nearly symmetric the
                        // quartic's roots nearly coincide and `cos L` comes back with only half its
                        // digits — measured at `2.4e-6` on a `e = 0.0011` GEO with the Sun in the
                        // `f`–`w` plane, which lands `|S|` at `4.7e-6` and has `S_ZERO = 1e-6`
                        // discard a crossing that is real. `L` itself is well conditioned there
                        // (`dS/dL` is `O(1)`), so Newton on 3.5-1 recovers the full precision.
                        let seed = atan2(j * s_l, cos_l);
                        let mut l = seed;
                        for _ in 0..8 {
                            let slope = 2.0 * shadow_at(l).2;
                            if slope == 0.0 {
                                break;
                            }
                            let step = shadow_at(l).1 / slope;
                            l -= step;
                            if fabs(step) <= f64::EPSILON * PI {
                                break;
                            }
                        }
                        // A seed is already within `√ε` of its root; an iteration that travelled
                        // further has left the basin and landed on somebody else's.
                        if fabs(normalize_angle(l - seed, 0.0)) > 1.0e-3 {
                            l = seed;
                        }
                        let (c_phi, s, half_ds_dl) = shadow_at(l);
                        // On the shadow side of the central body (eq. 3.5-3), and 3.5-1 satisfied?
                        if c_phi < 0.0 && fabs(s) < S_ZERO {
                            // Entry or exit? (sign of dS/dL).
                            if half_ds_dl > 0.0 {
                                exit_found = true;
                                ll[0] = l;
                            } else {
                                entry_found = true;
                                ll[1] = l;
                            }
                        }
                    }
                }
                // Must be one entry and one exit, or none.
                if entry_found != exit_found {
                    ll[0] = -PI;
                    ll[1] = PI;
                }
                // Quadrature runs from exit to entry, so L_exit must be below L_entry.
                if ll[0] > ll[1] {
                    if ll[1] < 0.0 {
                        ll[1] += 2.0 * PI;
                    } else {
                        ll[0] -= 2.0 * PI;
                    }
                }
            }
        }
        ll
    }
}

/// Normalize a 3-vector to unit length (Hipparchus `Vector3D.normalize`).
fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = gaussian::norm(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// Equinoctial mean → true longitude: solve Kepler for the eccentric longitude, then convert to true
/// (`EquinoctialLongitudeArgumentUtility.meanToEccentric` + `eccentricToTrue`). `ex = k`, `ey = h`.
///
/// Shared with [`super::drag`] (both need Orekit's `EquinoctialOrbit.getLv()` for the full-circle
/// `getLLimits` centre).
pub(super) fn mean_to_true(ex: f64, ey: f64, lm: f64) -> f64 {
    let le = mean_to_eccentric(ex, ey, lm);
    eccentric_to_true(ex, ey, le)
}

/// Equinoctial Kepler solve (`EquinoctialLongitudeArgumentUtility.meanToEccentric`).
fn mean_to_eccentric(ex: f64, ey: f64, lm: f64) -> f64 {
    const TOLERANCE_CONVERGENCE: f64 = 1.0e-11;
    const MAXIMUM_ITERATION: usize = 50;

    let mut le = lm;
    let mut le_m_lm = 0.0_f64;
    let mut iter = 0;
    loop {
        let sin_le = sin(le);
        let cos_le = cos(le);
        let f2 = ex * sin_le - ey * cos_le;
        let f1 = 1.0 - ex * cos_le - ey * sin_le;
        let f0 = le_m_lm - f2;

        let f12 = 2.0 * f1;
        let shift = f0 * f12 / (f1 * f12 - f0 * f2);

        le_m_lm -= shift;
        le = lm + le_m_lm;

        iter += 1;
        if fabs(shift) <= TOLERANCE_CONVERGENCE || iter >= MAXIMUM_ITERATION {
            break;
        }
    }
    le
}

/// Equinoctial eccentric → true longitude (`EquinoctialLongitudeArgumentUtility.eccentricToTrue`).
fn eccentric_to_true(ex: f64, ey: f64, le: f64) -> f64 {
    let epsilon = sqrt(1.0 - ex * ex - ey * ey);
    let sin_le = sin(le);
    let cos_le = cos(le);
    let num = ex * sin_le - ey * cos_le;
    let den = epsilon + 1.0 - ex * cos_le - ey * sin_le;
    le + 2.0 * atan(num / den)
}

/// Reduce an angle to `[center − π, center + π)` (Hipparchus `MathUtils.normalizeAngle`). Shared with
/// [`super::drag`].
pub(super) fn normalize_angle(a: f64, center: f64) -> f64 {
    use core::f64::consts::{PI, TAU};
    a - TAU * floor((a + PI - center) / TAU)
}

/// Real roots of `a₀x⁴ + a₁x³ + a₂x² + a₃x + a₄` — Orekit's `realQuarticRoots`.
fn real_quartic_roots(a: &[f64; 5], y: &mut [f64; 4]) -> usize {
    // Degenerate quartic → cubic.
    if precision_equals(a[0], 0.0) {
        let aa = [a[1], a[2], a[3], a[4]];
        let mut y3 = [0.0_f64; 3];
        let n = real_cubic_roots(&aa, &mut y3);
        y[..3].copy_from_slice(&y3);
        return n;
    }

    // Biquadratic — solve for `x²` instead of going through Ferrari.
    //
    // With no odd powers the resolvent cubic `(z − c)(z² − 4e)` has a double root, and every quantity
    // Ferrari then derives from it is a cancellation of two numbers that agree in the reals: `p`, `q`,
    // and the discriminants of both factors. So the roots survive only if the rounding happens to
    // fall the right way, and `l_limits` builds exactly this quartic for every circular orbit —
    // `h = k = 0` zeroes `a[1]` and `a[3]` identically — where losing them silently removes the
    // eclipse and averages the whole revolution as sunlit. Orekit 13.1.7 has no such branch.
    //
    // Solving `a₀u² + a₂u + a₄` for `u = x²` leaves one discriminant instead of four, measured at
    // `≤ 1 ulp` of `c²` across that family, which the tie window in `real_quadratic_roots` covers.
    if a[1] == 0.0 && a[3] == 0.0 {
        let mut u = [0.0_f64; 2];
        let nu = real_quadratic_roots(&[a[0], a[2], a[4]], &mut u);
        let mut n = 0;
        for &ui in u.iter().take(nu) {
            if ui > 0.0 {
                let s = sqrt(ui);
                y[n] = s;
                y[n + 1] = -s;
                n += 2;
            } else if ui == 0.0 {
                y[n] = 0.0;
                n += 1;
            }
        }
        return n;
    }

    let b = a[1] / a[0];
    let c = a[2] / a[0];
    let d = a[3] / a[0];
    let e = a[4] / a[0];
    let bh = b * 0.5;

    // Resolvent cubic.
    let mut z3 = [0.0_f64; 3];
    let i3 = real_cubic_roots(
        &[1.0, -c, b * d - 4.0 * e, e * (4.0 * c - b * b) - d * d],
        &mut z3,
    );
    if i3 == 0 {
        return 0;
    }

    let z = z3[0];
    let zh = z * 0.5;
    let p = (z + bh * bh - c).max(0.0);
    let q = (zh * zh - e).max(0.0);
    let r = bh * z - d;
    let pp = sqrt(p);
    let qq = copysign(sqrt(q), r);

    let mut y1 = [0.0_f64; 2];
    let n1 = real_quadratic_roots(&[1.0, bh - pp, zh - qq], &mut y1);
    let mut y2 = [0.0_f64; 2];
    let n2 = real_quadratic_roots(&[1.0, bh + pp, zh + qq], &mut y2);

    if n1 == 2 {
        if n2 == 2 {
            y[0] = y1[0];
            y[1] = y1[1];
            y[2] = y2[0];
            y[3] = y2[1];
            4
        } else {
            y[0] = y1[0];
            y[1] = y1[1];
            2
        }
    } else if n2 == 2 {
        y[0] = y2[0];
        y[1] = y2[1];
        2
    } else {
        0
    }
}

/// Real roots of `a₀x³ + a₁x² + a₂x + a₃` (descending order) — Orekit's `realCubicRoots`.
fn real_cubic_roots(a: &[f64; 4], y: &mut [f64; 3]) -> usize {
    use core::f64::consts::PI;
    if precision_equals(a[0], 0.0) {
        let aa = [a[1], a[2], a[3]];
        let mut y2 = [0.0_f64; 2];
        let n = real_quadratic_roots(&aa, &mut y2);
        y[..2].copy_from_slice(&y2);
        return n;
    }

    let b = -a[1] / (3.0 * a[0]);
    let c = a[2] / a[0];
    let d = a[3] / a[0];
    let b2 = b * b;
    let p = b2 - c / 3.0;
    let q = b * (b2 - c * 0.5) - d * 0.5;

    let disc = p * p * p - q * q;

    if disc < 0.0 {
        // One real root.
        let alpha = q + copysign(sqrt(-disc), q);
        let cbrt_al = cbrt(alpha);
        let cbrt_be = p / cbrt_al;

        if p < 0.0 {
            y[0] = b + 2.0 * q / (cbrt_al * cbrt_al + cbrt_be * cbrt_be - p);
        } else if p > 0.0 {
            y[0] = b + cbrt_al + cbrt_be;
        } else {
            y[0] = b + cbrt_al;
        }
        1
    } else if disc > 0.0 {
        // Three distinct real roots.
        let phi = atan2(sqrt(disc), q) / 3.0;
        let sq_p = 2.0 * sqrt(p);
        y[0] = b + sq_p * cos(phi);
        y[1] = b - sq_p * cos(PI / 3.0 + phi);
        y[2] = b - sq_p * cos(PI / 3.0 - phi);
        3
    } else {
        // One distinct and two equal real roots.
        let cbrt_q = cbrt(q);
        let root1 = b + 2.0 * cbrt_q;
        let root2 = b - cbrt_q;
        if q < 0.0 {
            y[0] = root2;
            y[1] = root2;
            y[2] = root1;
        } else {
            y[0] = root1;
            y[1] = root2;
            y[2] = root2;
        }
        3
    }
}

/// Real roots of `a₀x² + a₁x + a₂` (descending order) — Orekit's `realQuadraticRoots`.
fn real_quadratic_roots(a: &[f64; 3], y: &mut [f64; 2]) -> usize {
    if precision_equals(a[0], 0.0) {
        if precision_equals(a[1], 0.0) {
            return 0;
        }
        y[0] = -a[2] / a[1];
        return 1;
    }

    let b = -0.5 * a[1] / a[0];
    let c = a[2] / a[0];
    // At a double root `b²` and `c` agree in the reals, so `b² − c` is pure cancellation: the sign of
    // the difference is then decided by the rounding already carried in `b` and `c`, not by the
    // polynomial, and a double root reports "no real roots" about half the time. Ferrari routes
    // every perfect-square quartic through exactly this configuration, and `l_limits` builds one
    // for every circular orbit whose Sun lies along an equinoctial axis — where losing the pair
    // loses the eclipse itself and the whole revolution is averaged as sunlit. The `fma` makes the
    // subtraction correctly rounded; the window below covers the error `b` and `c` arrived with.
    // Measured over that family: `|d|` peaks at `0.43` ulp of `max(b², |c|)` with the `fma` and
    // `0.61` without, against the `4` ulp admitted here. Orekit (13.1.7) tests `d < 0` exactly.
    let d = libm::fma(b, b, -c);
    let tie = 4.0 * f64::EPSILON * fabs(b * b).max(fabs(c));

    if d < -tie {
        0
    } else if d > 0.0 {
        let y0 = b + copysign(sqrt(d), b);
        let y1 = c / y0;
        y[0] = y0.max(y1);
        y[1] = y0.min(y1);
        2
    } else {
        y[0] = b;
        y[1] = b;
        2
    }
}

/// Hipparchus `Precision.equals(x, y)` (`maxUlps = 1`): true iff `x` and `y` are within one ULP.
fn precision_equals(x: f64, y: f64) -> bool {
    if x.is_nan() || y.is_nan() {
        return false;
    }
    let x_int = x.to_bits() as i64;
    let y_int = y.to_bits() as i64;
    let sgn_mask = i64::MIN; // 0x8000_0000_0000_0000
    if ((x_int ^ y_int) & sgn_mask) == 0 {
        (x_int - y_int).abs() <= 1
    } else {
        let neg_zero = (-0.0_f64).to_bits() as i64;
        let (delta_plus, delta_minus) = if x_int < y_int {
            (y_int, x_int - neg_zero)
        } else {
            (x_int, y_int - neg_zero)
        };
        if delta_plus > 1 {
            false
        } else {
            delta_minus <= (1 - delta_plus)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // the umbra and full-Sun answers are exact by construction

    use spacelab_testkit::nan_max;

    use super::*;

    // Golden fixtures from Orekit 13 (`tools/orekit-oracle`, orekit_dsst_srp.csv): a cannonball SRP
    // (cr = 1.5, area = 10 m², mass = 1000 kg ⇒ S/m = 0.01) on the same MEAN equinoctial LEO as the
    // zonal/third-body cases (a = 7.2e6, e = 0.01, i = 98°), Sun at epoch 2024-01-01T00:00:00 UTC in
    // GCRF, SPHERICAL Earth of radius ae. Direct fixed-state getMeanElementRate, plus the integration
    // limits getLLimits used and the Gauss order the isDirty pass selected.
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const CR: f64 = 1.50000000000000000e+00;
    const AREA: f64 = 1.00000000000000000e+01;
    const MASS: f64 = 1.00000000000000000e+03;

    const GAUSS_ORDER: usize = 48;
    const LL_LOW: f64 = -4.39803656243010300e+00;
    const LL_HIGH: f64 = 6.34076980768220600e-01;

    // Initial MEAN equinoctial state (a, ex=k, ey=h, hx=q, hy=p, lM=λ).
    const A0: f64 = 7.20000000000000000e+06;
    const EX0: f64 = 3.09016994374947470e-03;
    const EY0: f64 = 9.51056516295153600e-03;
    const HX0: f64 = 8.54890329218965000e-01;
    const HY0: f64 = 7.69746709859868000e-01;
    const LM0: f64 = 1.43116998663535020e+00;

    // Sun geocentric position in the propagation frame (GCRF) at the epoch, m.
    const SUN_POS: [f64; 3] = [
        2.48130578062755620e+10,
        -1.33033126695834660e+11,
        -5.76679652360023650e+10,
    ];

    // getMeanElementRate: [da, dk, dh, dq, dp, dM].
    const RATES: [f64; 6] = [
        -1.66681113612731410e-09,
        6.42088118316530040e-12,
        -2.06453695796115100e-12,
        5.54383062682849500e-13,
        1.72237499933817420e-12,
        2.94388190334062430e-12,
    ];

    fn srp() -> DsstSolarRadiationPressure {
        DsstSolarRadiationPressure::new(CR, AREA / MASS, MU, AE)
    }

    fn aux() -> DsstAuxiliary {
        DsstAuxiliary::new(A0, EX0, EY0, HX0, HY0, LM0, MU)
    }

    /// The cylindrical-shadow integration limits `[L_exit, L_entry]`, against Orekit's `getLLimits`
    /// and the deliberate deviation from it.
    ///
    /// Orekit sets `m = ae / (a·B)` in the shadow equation `S = 1 − m²·range² − cos²Φ`. With the
    /// equinoctial radius `r = a·B²/range` that reads `S = 1 − (ae·B/r)² − cos²Φ`, so `S = 0` puts
    /// the boundary at a perpendicular distance `ae·B` from the anti-Sun axis: a cylinder that
    /// narrows as the orbit gets more eccentric. Checked against Orekit's own golden limits below,
    /// the distance at both crossings is `6 377 817.545 m`, and `ae` is `6 378 136.46 m`.
    ///
    /// The constant is a transcription error, not a modelling choice. Danielson 3.5 prints
    /// `M = ae²/(a²(1 − h² − k²))`; his own eq. 2.1.4-6 gives
    /// `r = a(1 − h² − k²)/(1 + h sin L + k cos L)`, so the shadow equation needs
    /// `M = (ae/(a(1 − h² − k²)))²` — one more power of `B²`. Escobal (1976) writes the same
    /// cylinder as `ae²(1 + e cos f)² + p²cos²Φ − p² = 0` with the semi-latus rectum
    /// `p = a(1 − e²)`, and Valk & Lemaître (2008) restate it in `h`, `k`; both divide by `p²`,
    /// not by `a²B²`. Orekit follows the printed formula.
    ///
    /// `l_limits` casts the physical `ae`, so it no longer reproduces those numbers, and the gap is
    /// asserted rather than the agreement — the geometry itself is pinned in
    /// [`the_shadow_limits_are_the_roots_of_the_shadow_equation`], to `3.6e-15 rad` against a
    /// bisection that shares no algebra with the code. A single `ae → ae·B` substitution would
    /// reproduce Orekit exactly here, but `ae` is also the occulting limb radius the penumbra
    /// lighting ratio uses, so the substitution moves more than the cylinder.
    ///
    /// Third-party confirmation, from `experiments/dsst_srp_independent_verify.py` — a VOP
    /// orbit-average that builds the cylinder from the geometry and never sees this code. Bisecting
    /// the crossings instead of scanning for them, it puts the entry at `0.63381256040` and the
    /// exit at `1.88541311219`, against `0.6338126` / `1.8854132` here and `0.6340770` /
    /// `1.8851490` from Orekit: `4e-8 rad` away, against `2.7e-4 rad` from Orekit, a factor of
    /// `6700`.
    #[test]
    fn integration_limits_match_orekit() {
        let a = aux();
        let ll = srp().l_limits(&a, SUN_POS);
        // Perpendicular distance from the anti-Sun axis at a true longitude, in metres — the one
        // quantity the two models disagree about, written from the geometry rather than either code.
        let dir = normalize(SUN_POS);
        let alpha = gaussian::dot_ext(dir, a.f_vec);
        let beta = gaussian::dot_ext(dir, a.g_vec);
        let perpendicular = |l: f64| {
            let (c, s) = (cos(l), sin(l));
            let r = a.sma * a.big_b * a.big_b / (1.0 + a.k * c + a.h * s);
            r * sqrt(1.0 - (alpha * c + beta * s) * (alpha * c + beta * s))
        };
        for l in ll {
            let miss = libm::fabs(perpendicular(l) - AE);
            assert!(
                miss < 1.0e-6,
                "crossing at {l} sits {miss:.3e} m off the ae cylinder"
            );
        }
        let orekit_radius = AE * a.big_b;
        for l in [LL_LOW, LL_HIGH] {
            let miss = libm::fabs(perpendicular(l) - orekit_radius);
            assert!(
                miss < 1.0e-3,
                "Orekit's crossing at {l} sits {miss:.3e} m off the ae*B cylinder"
            );
        }
        let d_low = (ll[0] - LL_LOW).abs();
        let d_high = (ll[1] - LL_HIGH).abs();
        assert!(
            (2.6e-4..2.7e-4).contains(&d_low) && (2.6e-4..2.7e-4).contains(&d_high),
            "the deviation from Orekit moved: exit {d_low:.6e}, entry {d_high:.6e} rad"
        );
        let here = AE;
        std::println!(
            "SRP limits: {here:.1} m cylinder here against Orekit's {orekit_radius:.1} m, \
             {d_low:.3e} / {d_high:.3e} rad apart"
        );
    }

    /// The Gauss-order the port selects must equal Orekit's cached order (48 here — SRP's shadow makes
    /// lower orders miss the 1e-15 threshold).
    #[test]
    fn gauss_order_matches_orekit() {
        assert_eq!(srp().gauss_order(&aux(), SUN_POS), GAUSS_ORDER);
    }

    /// The six fixed-state SRP mean-element rates against Orekit's `getMeanElementRate`, and the
    /// deviation the shadow-radius correction accounts for.
    ///
    /// Orekit's cylindrical shadow has radius `ae·B` rather than `ae` (see
    /// [`integration_limits_match_orekit`]), which on this `e = 0.01` LEO lengthens the sunlit arc
    /// by `5.3e-4 rad` and moves every rate with it. The two models therefore no longer agree to
    /// round-off, and pretending otherwise by loosening the gate to `1e-3` would let a real
    /// regression through — so the deviation is bracketed instead of bounded. Measured, and asserted
    /// to stay put within a factor of two:
    ///
    /// | `da` | `dk` | `dh` | `dq` | `dp` | `dM` |
    /// |---|---|---|---|---|---|
    /// | `3.455e-4` | `5.311e-5` | `5.311e-5` | `2.049e-4` | `2.050e-4` | `2.080e-4` |
    ///
    /// What the fixture still pins is everything downstream of the limits: the Gauss order, the
    /// quadrature, the equinoctial rate assembly, the penumbra lighting ratio. What it no longer
    /// pins is the shadow arc — [`the_shadow_limits_are_the_roots_of_the_shadow_equation`] does
    /// that, to `3.6e-15 rad` over 9408 states against a construction that shares no algebra with
    /// the code. **The accuracy statement is no longer here** but in
    /// [`fixed_state_rates_match_the_independent_vop_average`], which is an oracle rather than a
    /// second reading of the same convention.
    #[test]
    fn fixed_state_rates_match_orekit() {
        let got = srp().mean_element_rates(&aux(), SUN_POS);
        // Half and twice the measured deviation: tight enough that any other change is a failure.
        let want_dev = [3.455e-4, 5.311e-5, 5.311e-5, 2.049e-4, 2.050e-4, 2.080e-4];
        let mut errs = [0.0_f64; 6];
        for i in 0..6 {
            errs[i] = if RATES[i] == 0.0 {
                libm::fabs(got[i])
            } else {
                libm::fabs((got[i] - RATES[i]) / RATES[i])
            };
        }
        std::println!(
            "SRP rate deviation from Orekit's shadow radius: da {:.3e} dk {:.3e} dh {:.3e} \
             dq {:.3e} dp {:.3e} dm {:.3e}",
            errs[0],
            errs[1],
            errs[2],
            errs[3],
            errs[4],
            errs[5]
        );
        for i in 0..6 {
            assert!(
                errs[i] > 0.5 * want_dev[i] && errs[i] < 2.0 * want_dev[i],
                "rate[{i}] got {} want {} rel {:.3e}, outside [{:.3e}, {:.3e}]",
                got[i],
                RATES[i],
                errs[i],
                0.5 * want_dev[i],
                2.0 * want_dev[i]
            );
        }
    }

    /// The same six rates against an independent variation-of-parameters orbit-average.
    ///
    /// `experiments/dsst_srp_independent_verify.py` shares no algebra with this module: it converts
    /// state to classical elements with a textbook `rv → (a, e, i, Ω, ω, M)`, takes `∂el/∂v` by
    /// central differences, dots that with the SRP acceleration, averages over the mean anomaly and
    /// pushes the classical rates through a numerical Jacobian into equinoctial. No Gauss VOP
    /// matrix, no Hansen kernel, no `l_limits`, no Orekit. Three things had to be right before it
    /// could bound anything, and each is convergence-tested by `--converge`:
    ///
    /// * the shadow crossings are found by bisection rather than by grid search, which put the
    ///   entry and exit within `4e-8 rad` of [`l_limits`] instead of the `7.6e-6 rad` a `8e5`-point
    ///   scan managed;
    /// * the quadrature runs Gauss-Legendre over the *lit arc*, split again at the penumbra kink.
    ///   A uniform grid is only `O(1/N)` on a discontinuous integrand, and `da` pays for it: a
    ///   constant force over a closed orbit has zero mean `da/dt` exactly, so the whole of this
    ///   rate is the shadow and the quadrature error is measured against cancelling halves. `8e5`
    ///   uniform points gave `da` to `1.4e-2`; the split rule converges by order 16;
    /// * the lighting is the conical penumbra, not the binary cylinder. That is the difference
    ///   between `3.2e-3` and `4.5e-4` against Orekit, and it is what makes the comparison a
    ///   comparison of implementations rather than of shadow models.
    ///
    /// The measured agreement, all six the same sign as the `ae`-vs-`ae·B` correction predicts:
    ///
    /// | | `da` | `dk` | `dh` | `dq` | `dp` | `dM` |
    /// |---|---|---|---|---|---|---|
    /// | vs the VOP average | `1.044e-4` | `2.063e-5` | `2.062e-5` | `8.256e-5` | `8.267e-5` | `8.389e-5` |
    /// | vs Orekit | `3.455e-4` | `5.311e-5` | `5.311e-5` | `2.049e-4` | `2.050e-4` | `2.080e-4` |
    ///
    /// So a third implementation, told nothing about either, lands between the two and `2.5×`
    /// closer to this one. What is left is the DSST scheme's own approximation, and most of it is
    /// identifiable: Orekit's order-48 Gauss rule integrates straight *across* the penumbra kink,
    /// and re-running the script that way reproduces the non-monotone signature of a Gauss rule
    /// spanning a `C⁰` corner — `dq` moves `+3.0e-4, +4.1e-5, −9.3e-5, −8.7e-5, −8.2e-5` at orders
    /// `32, 48, 64, 128, 256`, settling on the split answer only at high order.
    ///
    /// Bound at `1.5×` the measured residual, per component.
    #[test]
    fn fixed_state_rates_match_the_independent_vop_average() {
        // experiments/dsst_srp_independent_verify.py, "conical ratio, cylinder arc".
        const VOP_RATES: [f64; 6] = [
            -1.66756115236162085e-09,
            6.42067258043714643e-12,
            -2.06446987717553016e-12,
            5.54450853483786070e-13,
            1.72258563249958469e-12,
            2.94424724419369266e-12,
        ];
        const WANT: [f64; 6] = [1.044e-4, 2.063e-5, 2.062e-5, 8.256e-5, 8.267e-5, 8.389e-5];
        let got = srp().mean_element_rates(&aux(), SUN_POS);
        for i in 0..6 {
            let rel = libm::fabs((got[i] - VOP_RATES[i]) / VOP_RATES[i]);
            assert!(
                rel < 1.5 * WANT[i],
                "rate[{i}] got {} vs the VOP average {} — rel {:.3e}, above {:.3e}",
                got[i],
                VOP_RATES[i],
                rel,
                1.5 * WANT[i]
            );
        }
    }

    /// Horner evaluation of `a₀xⁿ + … + aₙ`, for the residual half of the root check.
    fn horner(a: &[f64], x: f64) -> f64 {
        a.iter().fold(0.0, |acc, &c| acc.mul_add(x, c))
    }

    /// `got[..n]` equals `want` as a multiset within `tol`, and each returned value really is a
    /// root of `a`. Two halves, because either alone is weak: matching the expected list without
    /// the residual would pass on a solver that returns a memorised answer, and the residual
    /// without the list would pass on one that returns the same root three times.
    fn assert_roots(a: &[f64], got: &[f64], n: usize, want: &[f64], tol: f64, what: &str) {
        assert_eq!(
            n,
            want.len(),
            "{what}: root count {n}, expected {}",
            want.len()
        );
        let mut left = want.to_vec();
        for &y in &got[..n] {
            // Backward error, `|p(y)| / Σ|aᵢ||y|ⁿ⁻ⁱ`, not `|p(y)|` itself: the second grows with
            // the conditioning of the root and would force a tolerance loose enough to accept a
            // wrong answer on the well-conditioned cases. The backward error stays at machine
            // epsilon for any root a stable solver returns, however ill-conditioned it is.
            let scale = a
                .iter()
                .fold(0.0_f64, |acc, &c| acc.mul_add(libm::fabs(y), libm::fabs(c)));
            let residual = libm::fabs(horner(a, y)) / if scale == 0.0 { 1.0 } else { scale };
            assert!(
                residual < 1e-14,
                "{what}: backward error at {y} is {residual:.3e}, not a root"
            );
            let (i, gap) =
                left.iter()
                    .enumerate()
                    .fold((usize::MAX, f64::INFINITY), |best, (i, &w)| {
                        let g = libm::fabs(y - w) / if w == 0.0 { 1.0 } else { libm::fabs(w) };
                        if g < best.1 { (i, g) } else { best }
                    });
            assert!(
                gap < tol,
                "{what}: root {y} matches nothing left in {left:?} (best {gap:.3e})"
            );
            left.remove(i);
        }
    }

    /// Every branch of the quadratic solver, on polynomials built from their own roots.
    ///
    /// The three solvers below are pure functions of their coefficients, reached in production only
    /// from inside `l_limits`, where the shadow geometry drives one discriminant arm per call. That
    /// is why 78 mutations of them survived a suite whose only SRP oracle is a propagated rate: the
    /// other arms are never entered, so nothing can notice them being wrong.
    #[test]
    fn the_quadratic_solver_finds_the_roots_on_every_branch() {
        let mut y = [0.0_f64; 2];
        // a0 == 0 and a1 == 0: not an equation, no roots.
        assert_eq!(real_quadratic_roots(&[0.0, 0.0, 5.0], &mut y), 0);
        assert_eq!(real_quadratic_roots(&[0.0, 0.0, 0.0], &mut y), 0);
        // a0 == 0: linear, one root at -a2/a1.
        let n = real_quadratic_roots(&[0.0, 2.0, -6.0], &mut y);
        assert_roots(&[2.0, -6.0], &y, n, &[3.0], 1e-15, "linear");
        // Negative discriminant.
        assert_eq!(real_quadratic_roots(&[1.0, 0.0, 1.0], &mut y), 0);
        assert_eq!(real_quadratic_roots(&[1.0, 2.0, 5.0], &mut y), 0);
        // Two distinct roots, both signs of a1 so the `copysign` in the stable form is exercised.
        for (a, want) in [
            ([1.0, -5.0, 6.0], [3.0, 2.0]),
            ([1.0, 5.0, 6.0], [-2.0, -3.0]),
            ([1.0, 0.0, -4.0], [2.0, -2.0]),
            ([3.0, -15.0, 18.0], [3.0, 2.0]),
        ] {
            let n = real_quadratic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &want, 1e-15, "distinct");
            assert!(y[0] >= y[1], "roots must come back descending, got {y:?}");
        }
        // Zero discriminant: one root, reported twice. Also at the origin, where the stable form
        // divides `c / y0` by a zero it just produced.
        for (a, want) in [
            ([2.0, -4.0, 2.0], [1.0, 1.0]),
            ([1.0, 0.0, 0.0], [0.0, 0.0]),
        ] {
            let n = real_quadratic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &want, 1e-15, "double");
        }
        // The tie window admits a double root's rounding and nothing else. `x² − 2·10⁻⁴x +
        // (10⁻⁸ + 10⁻²⁰)` misses by `d = −10⁻²⁰`, which is `1126` times the window, and its roots
        // are genuinely complex. Scaling the window by anything other than `max(b², |c|)` swallows
        // that miss: `4ε/max(…)` widens it to `8.9e−8`, `max(|b+b|, |c|)` to `1.8e−19` and
        // `max(|b/b|, |c|)` to `8.9e−16` — each reports a double root at `10⁻⁴` that is not there.
        assert_eq!(
            real_quadratic_roots(&[1.0, -2.0e-4, 1.0e-8 + 1.0e-20], &mut y),
            0,
            "a discriminant three orders below the tie window is not a tie"
        );
        // Widely separated roots, which is the whole reason for the `b + copysign(√d, b)` form:
        // the other sign is the subtractive one, and it loses every digit of the small root.
        // `(x − 1e8)(x − 1e−8)` has a discriminant whose square root agrees with `b` to 16 digits.
        let a = [1.0, -(1e8 + 1e-8), 1.0];
        let n = real_quadratic_roots(&a, &mut y);
        assert_roots(&a, &y, n, &[1e8, 1e-8], 1e-9, "ill-conditioned");
    }

    /// Every branch of the cubic solver: the quadratic hand-off, both signs of `p` and the `p == 0`
    /// case under a negative discriminant, three distinct roots, and both orderings of the double
    /// root at a zero discriminant.
    #[test]
    fn the_cubic_solver_finds_the_roots_on_every_branch() {
        let mut y = [0.0_f64; 3];
        // a0 == 0: hands off to the quadratic.
        let a = [0.0, 1.0, -3.0, 2.0];
        let n = real_cubic_roots(&a, &mut y);
        assert_roots(
            &[1.0, -3.0, 2.0],
            &y,
            n,
            &[2.0, 1.0],
            1e-15,
            "cubic -> quadratic",
        );
        // Three distinct real roots (positive discriminant). Roots `{1, 2, 3}` are symmetric about
        // their mean, which makes `q` vanish and `cos(π/3 + φ)` equal `cos(π/3 − φ)` — so a
        // symmetric case alone cannot tell the second root's formula from the third's. `{0, 1, 3}`
        // is deliberately lopsided.
        for (a, want) in [
            ([1.0, -6.0, 11.0, -6.0], [1.0, 2.0, 3.0]),
            ([2.0, -12.0, 22.0, -12.0], [1.0, 2.0, 3.0]),
            ([1.0, -4.0, 3.0, 0.0], [0.0, 1.0, 3.0]),
            ([1.0, -8.0, 17.0, -10.0], [1.0, 2.0, 5.0]),
        ] {
            let n = real_cubic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &want, 1e-13, "three distinct");
        }
        // One real root, negative discriminant. `p = b² − c/3` picks the arm: `(x−1)(x²+x+3)` has
        // `p < 0`, `(x−5)(x²+5x+8)` has `p > 0`, and `x³ − 8` puts it exactly at zero.
        for (a, want) in [
            ([1.0, 0.0, 2.0, -3.0], 1.0),
            ([1.0, 0.0, -17.0, -40.0], 5.0),
            ([1.0, 0.0, 0.0, -8.0], 2.0),
        ] {
            let n = real_cubic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &[want], 1e-14, "one real");
        }
        // Zero discriminant: a double root and a single one, on both sides of `q == 0`. `q` decides
        // only where the single root goes in the triple, so the order is asserted exactly —
        // `assert_roots` pairs each root with its nearest partner and cannot see a permutation.
        for (a, want) in [
            ([1.0, 0.0, -3.0, 2.0], [1.0, 1.0, -2.0]),
            ([1.0, 0.0, -3.0, -2.0], [2.0, -1.0, -1.0]),
            ([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ] {
            let n = real_cubic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &want, 1e-14, "double root");
            assert_eq!(&y[..n], &want[..n], "double-root ordering of {a:?}");
        }
    }

    /// Every branch of the quartic solver: the cubic hand-off, four real roots, two real roots out
    /// of each of the two resolvent quadratics, none, and a pair of double roots.
    #[test]
    fn the_quartic_solver_finds_the_roots_on_every_branch() {
        let mut y = [0.0_f64; 4];
        // a0 == 0: hands off to the cubic.
        let a = [0.0, 1.0, -6.0, 11.0, -6.0];
        let n = real_quartic_roots(&a, &mut y);
        assert_roots(
            &[1.0, -6.0, 11.0, -6.0],
            &y,
            n,
            &[1.0, 2.0, 3.0],
            1e-13,
            "quartic -> cubic",
        );
        // Four real roots, and the same polynomial scaled, so the division by a0 is exercised.
        for a in [
            [1.0, -10.0, 35.0, -50.0, 24.0],
            [-2.0, 20.0, -70.0, 100.0, -48.0],
        ] {
            let n = real_quartic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &[1.0, 2.0, 3.0, 4.0], 1e-12, "four real");
        }
        // Two real roots: one conjugate pair. Both geometries, so neither resolvent quadratic is
        // the only one ever asked for its answer.
        for (a, want) in [
            ([1.0, 0.0, 0.0, 0.0, -1.0], [1.0, -1.0]),
            ([1.0, 0.0, -8.0, 0.0, -9.0], [3.0, -3.0]),
            ([1.0, -2.0, 2.0, 2.0, -3.0], [1.0, -1.0]),
        ] {
            let n = real_quartic_roots(&a, &mut y);
            assert_roots(&a, &y, n, &want, 1e-12, "two real");
        }
        // No real roots at all.
        for a in [[1.0, 0.0, 5.0, 0.0, 4.0], [1.0, 0.0, 2.0, 0.0, 1.0]] {
            assert_eq!(
                real_quartic_roots(&a, &mut y),
                0,
                "expected no real roots of {a:?}"
            );
        }
        // Two double roots: `(x−1)²(x−2)²`. Half the significant digits are gone by construction.
        let a = [1.0, -6.0, 13.0, -12.0, 4.0];
        let n = real_quartic_roots(&a, &mut y);
        assert_roots(&a, &y, n, &[1.0, 1.0, 2.0, 2.0], 1e-7, "two double roots");
        // The biquadratic shortcut needs *both* odd powers absent. `x⁴ − 5x² + 4x = x(x−1)(x²+x−4)`
        // has `a₁ = 0` and `a₃ = 4`; solving it in `x²` would drop the `4x` and answer `{±√5, 0}`.
        let a = [1.0, 0.0, -5.0, 4.0, 0.0];
        let n = real_quartic_roots(&a, &mut y);
        let s17 = sqrt(17.0);
        assert_roots(
            &a,
            &y,
            n,
            &[0.0, 1.0, 0.5 * (s17 - 1.0), -0.5 * (s17 + 1.0)],
            1e-12,
            "one odd power present",
        );
        // A biquadratic through the origin: `x⁴ − x² = x²(x−1)(x+1)`. `u = x² = 0` is a single root
        // of the quartic, not the `±√u` pair the positive branch emits, so the count is three.
        let a = [1.0, 0.0, -1.0, 0.0, 0.0];
        let n = real_quartic_roots(&a, &mut y);
        assert_eq!(n, 3, "x⁴ − x² has three distinct real roots, got {n}");
        assert_roots(
            &a,
            &y,
            n,
            &[1.0, -1.0, 0.0],
            1e-15,
            "biquadratic at the origin",
        );
    }

    /// `precision_equals` is Hipparchus `Precision.equals(x, y, 1)`: within one ULP, counting the
    /// step across zero as one. The table is bit-level on purpose — the function is integer
    /// arithmetic on the bit patterns, and only exact neighbours can separate its branches.
    #[test]
    fn precision_equals_is_a_one_ulp_neighbourhood() {
        let up = |x: f64| f64::from_bits(x.to_bits() + 1);
        let sub = f64::from_bits(1);
        let cases: [(f64, f64, bool); 17] = [
            (1.0, 1.0, true),
            (1.0, up(1.0), true),
            (up(1.0), 1.0, true),
            (1.0, up(up(1.0)), false),
            (-1.0, -up(1.0), true),
            (-1.0, -up(up(1.0)), false),
            (f64::NAN, 1.0, false),
            (1.0, f64::NAN, false),
            (f64::NAN, f64::NAN, false),
            // Across the sign boundary: +0, −0 and the two smallest subnormals are consecutive.
            (0.0, -0.0, true),
            (0.0, sub, true),
            (-0.0, -sub, true),
            (0.0, -sub, true),
            // One step positive and none negative: the branch that adds the two half-gaps has to
            // let this through, and it is the only shape that separates `> 1` from `>= 1`.
            (sub, -0.0, true),
            (sub, -sub, false),
            (0.0, up(sub), false),
            (-1.0, 1.0, false),
        ];
        for (x, y, want) in cases {
            assert_eq!(
                precision_equals(x, y),
                want,
                "precision_equals({x:e}, {y:e}) should be {want}"
            );
        }
        // Infinity is one ULP past the largest finite double, by the same bit arithmetic.
        assert!(precision_equals(f64::INFINITY, f64::MAX));
        assert!(precision_equals(f64::INFINITY, f64::INFINITY));
        assert!(!precision_equals(f64::INFINITY, f64::NEG_INFINITY));
        // The NaN guard has to fire when *either* side is NaN, and the pair that proves it is the
        // smallest signalling NaN against infinity: their bit patterns are adjacent, so an `and`
        // guard would fall through to the integer arithmetic and call them equal.
        let smallest_nan = f64::from_bits(0x7FF0_0000_0000_0001);
        assert!(smallest_nan.is_nan());
        assert!(!precision_equals(smallest_nan, f64::INFINITY));
        assert!(!precision_equals(f64::INFINITY, smallest_nan));
    }

    /// `normalize_angle` lands in `[center − π, center + π)` and moves by a whole number of turns.
    #[test]
    fn normalize_angle_shifts_by_whole_turns_into_the_window() {
        use core::f64::consts::{PI, TAU};
        for i in -40..=40_i32 {
            for j in -6..=6_i32 {
                let a = f64::from(i) * 0.37 + f64::from(j) * TAU;
                for &center in &[0.0, PI, -PI, 3.0, -7.5] {
                    let r = normalize_angle(a, center);
                    // Closed, not half-open: `(a + π − center)` is rounded before the
                    // division, so an `a` that is an exact multiple of `2π` from `center − π` can
                    // land the quotient one ulp below the integer and leave `r` at `center + π`.
                    // Hipparchus has the same edge, and matching it is the contract.
                    assert!(
                        r >= center - PI && r <= center + PI,
                        "normalize_angle({a}, {center}) = {r} outside the window"
                    );
                    let turns = (a - r) / TAU;
                    assert!(
                        libm::fabs(turns - libm::round(turns)) < 1e-9,
                        "normalize_angle({a}, {center}) moved by {turns} turns"
                    );
                }
            }
        }
    }

    /// The equinoctial Kepler solve and the eccentric→true step, against the classical scalar forms.
    ///
    /// Equinoctial longitudes are the classical anomalies plus the longitude of perihelion:
    /// `λ = M + ϖ`, `L = E + ϖ`, `Lᵥ = ν + ϖ`, with `k = e·cos ϖ`, `h = e·sin ϖ`. So Kepler's own
    /// `M = E − e·sin E` and the half-angle identity `tan(ν/2) = √((1+e)/(1−e))·tan(E/2)` pin both
    /// functions without re-deriving either of them from the code under test.
    #[test]
    fn the_equinoctial_kepler_solve_matches_the_classical_form() {
        use core::f64::consts::{PI, TAU};
        let mut worst_e = 0.0_f64;
        let mut worst_v = 0.0_f64;
        for ei in 0..=18_u32 {
            let e = f64::from(ei) * 0.05;
            for pi_ in 0..8_u32 {
                let peri = f64::from(pi_) * PI / 4.0;
                let (ex, ey) = (e * cos(peri), e * sin(peri));
                for mi in 0..48_u32 {
                    let ecc_anom = f64::from(mi) * (TAU / 48.0);
                    let mean_anom = ecc_anom - e * sin(ecc_anom);
                    let le = mean_to_eccentric(ex, ey, mean_anom + peri);
                    worst_e = nan_max(
                        worst_e,
                        libm::fabs(normalize_angle(le - (ecc_anom + peri), 0.0)),
                    );
                    let true_anom =
                        2.0 * atan(sqrt((1.0 + e) / (1.0 - e)) * libm::tan(ecc_anom / 2.0));
                    let lv = eccentric_to_true(ex, ey, ecc_anom + peri);
                    worst_v = nan_max(
                        worst_v,
                        libm::fabs(normalize_angle(lv - (true_anom + peri), 0.0)),
                    );
                }
            }
        }
        // The solver's own convergence threshold is 1e-11 on the shift, which is a Halley step, so
        // the standing error is far below it; the true-longitude step is closed form.
        assert!(
            worst_e < 1e-13,
            "worst eccentric-longitude miss {worst_e:.3e} rad"
        );
        assert!(
            worst_v < 1e-13,
            "worst true-longitude miss {worst_v:.3e} rad"
        );
        std::println!("equinoctial Kepler: worst L {worst_e:.3e} rad, worst Lv {worst_v:.3e} rad");
    }

    /// Classical true longitude for equinoctial `(k, h, λ)`, used as the full-orbit centre.
    ///
    /// `λ = M + ϖ`, `L = ν + ϖ` with `ϖ = atan2(h, k)`; Kepler and the half-angle identity are
    /// written out here rather than reached through `mean_to_true`, and the mean anomaly is folded
    /// into `(−π, π]` by plain arithmetic rather than through `normalize_angle`.
    fn true_longitude_classical(k: f64, h: f64, lm: f64) -> f64 {
        use core::f64::consts::TAU;
        let e = sqrt(k * k + h * h);
        let peri = atan2(h, k);
        let mean_anom = (lm - peri) - TAU * libm::round((lm - peri) / TAU);
        let mut ecc_anom = mean_anom;
        for _ in 0..200 {
            ecc_anom -= (ecc_anom - e * sin(ecc_anom) - mean_anom) / (1.0 - e * cos(ecc_anom));
        }
        2.0 * atan(sqrt((1.0 + e) / (1.0 - e)) * libm::tan(ecc_anom / 2.0)) + peri
    }

    /// The sunlit-arc endpoints `(L_exit, L_entry)` in `(−π, π]`, by bracketing and bisecting the
    /// shadow function directly. `None` when the orbit never enters the shadow.
    ///
    /// This is the independent construction: it shares the *condition* with the code, as it must,
    /// and none of the *solution* — no quartic, no resolvent cubic, no discriminant branch.
    fn shadow_roots_by_bisection(
        aux: &DsstAuxiliary,
        sun_dir: [f64; 3],
        ae: f64,
    ) -> Option<(f64, f64)> {
        use core::f64::consts::{PI, TAU};
        /// Samples per revolution — fine enough that no shadow arc here falls between two of them.
        const N: usize = 8192;
        let alpha = gaussian::dot_ext(sun_dir, aux.f_vec);
        let beta = gaussian::dot_ext(sun_dir, aux.g_vec);
        // How far outside the shadow cylinder the satellite is, in metres: the perpendicular distance
        // from the anti-Sun axis, `r·sin Φ`, less the cylinder radius. Written from the geometry —
        // `r = a·B²/(1 + k cos L + h sin L)` is the equinoctial radius at true longitude `L` — so it
        // borrows no algebra from the code, not even the code's `m`. On the sunlit side it is
        // positive anyway (`r > ae` for every orbit here), so pinning it to `+1` there adds no sign
        // change and every bracket found below lies strictly inside `cos Φ < 0`.
        let shadow = |l: f64| {
            let (c, s) = (cos(l), sin(l));
            let c_phi = alpha * c + beta * s;
            if c_phi >= 0.0 {
                return 1.0;
            }
            let r = aux.sma * aux.big_b * aux.big_b / (1.0 + aux.k * c + aux.h * s);
            r * sqrt((1.0 - c_phi * c_phi).max(0.0)) - ae
        };
        let mut roots = std::vec::Vec::new();
        let mut prev = shadow(-PI);
        for i in 1..=N {
            let l = -PI + TAU * i as f64 / N as f64;
            let cur = shadow(l);
            if (prev < 0.0) != (cur < 0.0) {
                let (mut lo, mut hi) = (l - TAU / N as f64, l);
                for _ in 0..200 {
                    let mid = 0.5 * (lo + hi);
                    if (shadow(lo) < 0.0) == (shadow(mid) < 0.0) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                // Rising through zero — the shadow is being left behind, so this is the exit.
                roots.push((0.5 * (lo + hi), prev < 0.0));
            }
            prev = cur;
        }
        match roots.len() {
            0 => None,
            2 => {
                let (exit, entry) = if roots[0].1 {
                    (roots[0], roots[1])
                } else {
                    (roots[1], roots[0])
                };
                assert!(
                    exit.1 && !entry.1,
                    "the two crossings must be one exit and one entry"
                );
                Some((exit.0, entry.0))
            }
            n => panic!("{n} shadow crossings: the sweep resolved a geometry it should not have"),
        }
    }

    /// The one-sided reset: an accepted shadow root with no partner on the other side.
    ///
    /// `l_limits` sorts each accepted root into an entry or an exit by the sign of `dS/dL`, and an
    /// eclipse needs one of each. At the tangency — the Sun just far enough out of plane that the
    /// orbit grazes the cylinder — the two crossings merge, and the `|S| < S_ZERO` window can admit
    /// one of them while rejecting the other. The guard throws both away and hands back the whole
    /// revolution, which is what the geometry says: the bisection oracle finds no eclipse at any of
    /// these states, so the orbit is sunlit throughout. Dropping the sign from `ll[0] = -PI`
    /// returns `[π, π]` instead, a zero-width sunlit arc that deletes the SRP force outright.
    ///
    /// The window is thin. A scan over five decades of `γ` around the tangency of every orbit in
    /// [`the_shadow_limits_are_the_roots_of_the_shadow_equation`] found 16 such states out of
    /// 1 404 288, all of them this near-circular GEO with the Sun anti-aligned with `f`. So the
    /// sweep below covers a window instead of pinning a value, and the count is asserted: if the
    /// branch stops being reached the test says so rather than passing on an unexecuted line.
    #[test]
    fn the_shadow_limits_fall_back_to_the_full_orbit_at_the_tangency() {
        use core::f64::consts::{PI, TAU};
        let model = srp();
        let (a, k, h) = (4.216_4e7, 0.001, -0.0005);
        let mut hits = 0_u32;
        for li in 0..4_u32 {
            let aux = DsstAuxiliary::new(a, k, h, 0.0, 0.0, f64::from(li) * (TAU / 4.0), MU);
            // The grazing `γ` is the sine of the Sun's elevation at which the cylinder is tangent
            // to the orbit, `ae / r_perigee` to first order in the eccentricity.
            let g0 = AE / (a * (1.0 - aux.ecc));
            let lv = true_longitude_classical(k, h, aux.lambda);
            let lvn = lv - TAU * libm::round(lv / TAU);
            assert!(
                libm::fabs(lvn) > 1e-6,
                "the reset is only observable where it differs from the default bounds"
            );
            for gi in -400..=400_i32 {
                for sgn in [-1.0_f64, 1.0] {
                    let gamma = sgn * (g0 + f64::from(gi) * 1e-6 * g0);
                    let planar = sqrt(1.0 - gamma * gamma);
                    // `φ = π`: the Sun's in-plane part points along `−f`.
                    let mut sun = [0.0_f64; 3];
                    for j in 0..3 {
                        sun[j] = 1.495_978_707e11 * (-planar * aux.f_vec[j] + gamma * aux.w_vec[j]);
                    }
                    // `[-π, π]` is the reset alone: with `Lᵥ ≠ 0` the shadow-free default carries
                    // the `Lᵥ` shift, and a real eclipse returns its own arc.
                    if model.l_limits(&aux, sun) == [-PI, PI] {
                        hits += 1;
                        assert!(
                            shadow_roots_by_bisection(&aux, normalize(sun), AE).is_none(),
                            "the reset fired where there is a real eclipse, γ={gamma:.17e}"
                        );
                    }
                }
            }
        }
        // Two grazing `γ` per sign per mean longitude: `g₀` itself and `g₀(1 − 10⁻⁵)`.
        assert_eq!(
            hits, 16,
            "the one-sided reset is no longer reached by this sweep"
        );
    }

    /// The shadow limits, against a bisection on the shadow function itself.
    ///
    /// `l_limits` reduces the cylindrical-shadow condition to a quartic in `cos L`, takes its real
    /// roots, keeps those that satisfy the condition, and sorts them into an exit and an entry by
    /// the sign of `dS/dL`. One Orekit golden pair was its only test, and a single geometry visits
    /// one discriminant branch, one root count and one wrap-around — so 27 mutations of the quartic
    /// coefficients, of the classification and of the ordering survived it.
    ///
    /// What the quadrature consumes is the *arc*, and its integrand is `2π`-periodic in `L`: the
    /// answer is fixed by the exit longitude modulo `2π` and by the arc length, and the choice
    /// between the two representatives — shift the entry up or the exit down — cannot be seen in
    /// it. The rule is still asserted, because both branches are reachable and a slip in either
    /// turns the sunlit arc into its complement.
    #[test]
    fn the_shadow_limits_are_the_roots_of_the_shadow_equation() {
        use core::f64::consts::{PI, TAU};
        let model = srp();
        // (a, k, h, q, p): LEO through a deep HEO, and inclinations from equatorial to retrograde.
        let orbits: &[[f64; 5]] = &[
            [
                7.2e6,
                3.090_169_943_749_474_7e-3,
                9.510_565_162_951_536e-3,
                0.854_890_329_218_965,
                0.769_746_709_859_868,
            ],
            [7.078_137e6, 0.0, 0.0, 0.0, 0.0],
            [8.0e6, 0.05, -0.02, 0.3, -0.15],
            [2.656e7, 0.1, 0.25, 0.5, 0.2],
            [2.439_6e7, 0.70, 0.21, 0.061, 0.0],
            [4.216_4e7, 0.001, -0.0005, 0.0, 0.0],
            [1.0e8, 0.55, -0.45, 0.17, 0.4],
        ];
        let mut worst = 0.0_f64;
        let (mut full, mut plain, mut wrap_entry, mut wrap_exit) = (0_u32, 0_u32, 0_u32, 0_u32);
        for orbit in orbits {
            let [a, k, h, q, p] = *orbit;
            for li in 0..12_u32 {
                let aux = DsstAuxiliary::new(a, k, h, q, p, f64::from(li) * (TAU / 12.0), MU);
                assert!(
                    a * (1.0 - aux.ecc) > AE,
                    "a test orbit must clear the surface at perigee"
                );
                // Sun directions on the equinoctial triad, so the geometry is controlled directly:
                // `γ` sets how far out of plane the Sun is (and therefore whether a shadow exists
                // at all), `φ` turns it around the plane and moves the arc past the ±π seam.
                for gi in -3..=3_i32 {
                    let gamma = f64::from(gi) * 0.3;
                    let planar = sqrt(1.0 - gamma * gamma);
                    for pi_ in 0..16_u32 {
                        let phi = f64::from(pi_) * (TAU / 16.0);
                        let (ca, cb) = (planar * cos(phi), planar * sin(phi));
                        let mut sun = [0.0_f64; 3];
                        for j in 0..3 {
                            sun[j] = 1.495_978_707e11
                                * (ca * aux.f_vec[j] + cb * aux.g_vec[j] + gamma * aux.w_vec[j]);
                        }
                        let got = model.l_limits(&aux, sun);
                        let want = match shadow_roots_by_bisection(&aux, normalize(sun), AE) {
                            None => {
                                full += 1;
                                let lv = true_longitude_classical(k, h, aux.lambda);
                                let lvn = lv - TAU * libm::round(lv / TAU);
                                [-PI + lvn, PI + lvn]
                            }
                            Some((exit, entry)) => {
                                let mut w = [exit, entry];
                                if w[0] > w[1] {
                                    if w[1] < 0.0 {
                                        wrap_entry += 1;
                                        w[1] += TAU;
                                    } else {
                                        wrap_exit += 1;
                                        w[0] -= TAU;
                                    }
                                } else {
                                    plain += 1;
                                }
                                w
                            }
                        };
                        let miss =
                            nan_max(libm::fabs(got[0] - want[0]), libm::fabs(got[1] - want[1]));
                        if miss > 1e-5 {
                            std::println!(
                                "MISS {miss:.3e} a={a:.4e} k={k} h={h} q={q} p={p} gam={gamma:.2} phi={phi:.4} got={got:?} want={want:?}"
                            );
                        }
                        worst = nan_max(worst, miss);
                        assert!(
                            got[1] > got[0] && got[1] - got[0] <= TAU + 1e-12,
                            "the quadrature window {got:?} is not a forward arc of at most one turn"
                        );
                    }
                }
            }
        }
        // Every branch of the wrap-around, and both the shadowed and the shadow-free case, are
        // reached — otherwise the sweep proves nothing about the code it does not execute.
        assert!(
            full > 0 && plain > 0 && wrap_entry > 0 && wrap_exit > 0,
            "branch coverage: full {full}, plain {plain}, wrap-entry {wrap_entry}, wrap-exit {wrap_exit}"
        );
        // The quartic reaches `cos L` in closed form and the bisection reaches `L` directly, so
        // without the Newton pass the two part company at the grazing geometries — `2.816e-6 rad`
        // at `γ = ∓0.9`, against `~2e-7` where the crossing is transverse. With it the sweep closes
        // to `3.553e-15 rad`, and the bound is set just above that: it is what makes the Newton
        // pass load-bearing rather than decorative. Breaking its step formula lands at `3.0e-6`
        // (`2·S′ → 2/S′`), `1.7e-6` (`S/slope → S·slope`) or `7.3e-8` (either way of cutting the
        // loop to one iteration) — all of which a bound of `1e-5` would have accepted.
        assert!(
            worst < 1e-12,
            "worst shadow-limit disagreement {worst:.3e} rad"
        );
        std::println!(
            "shadow limits vs bisection: worst {worst:.3e} rad over {} states \
             (full {full}, plain {plain}, wrap-entry {wrap_entry}, wrap-exit {wrap_exit})",
            full + plain + wrap_entry + wrap_exit
        );
    }
}
