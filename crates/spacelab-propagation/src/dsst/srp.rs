//! The DSST **solar radiation pressure** contribution: a cannonball in the conical shadow of a
//! spherical central body, averaged over the orbit and expanded in its short periods.
//!
//! The mean-element rates are the orbit average `(1/2π)∫νF dλ`. `F` is Gauss's rate of the
//! cannonball law `q = C_R·(S/m)·P_ref·d_ref²·(r − r☉)/|r − r☉|³`, and `ν ∈ [0, 1]` is the fraction
//! of the Sun's disc in view, [`crate::shadow::lighting_ratio`]. `ν` leaves 1 where the orbit
//! crosses the penumbra cone and reaches 0 where it crosses the umbra cone, and at each such
//! contact it behaves like `(L − L₀)^{3/2}`. So the average is taken piece by piece:
//! * the **contacts** are the roots of Danielson's shadow function (3.5) carried from the cylinder
//!   to the two cones, `S(L) = 1 − cos²Φ − (m·w − τ·cos Φ)²`: a quartic in `tan((L − L₀)/2)`,
//!   `L₀` the anti-Sun direction in the orbit plane, whose roots Newton polishes on `S` itself;
//! * between two contacts the orbit is lit (`ν ≡ 1`), in the penumbra, or in the umbra. A lit piece
//!   takes Gauss–Legendre at order 48; a penumbra piece takes it at order 40 after a substitution
//!   that makes the `3/2` ends analytic (`gaussian::smoothstep_nodes`), cut first at its deepest
//!   point when both its ends are on the penumbra cone; the umbra is left out;
//! * the short-period Fourier coefficients ([`DsstSolarRadiationPressure::osculating`]) are taken
//!   on the same pieces. An orbit that never meets the penumbra cone takes the whole turn.
//!
//! The structure and the Gauss averaging are ported from Orekit's
//! `DSSTSolarRadiationPressure` (Apache-2.0, in `NOTICE`). Orekit cuts the average at a cylinder
//! and weights it by the conical `ν` inside, which drops the light of the inner half of each
//! penumbra and keeps the deficit of the outer half; its cylinder also has radius `ae·B` where the
//! geometry has `ae` (Danielson 3.5 prints `M` one power of `B²` short).
//!
//! # Checked against the average that defines the rates
//!
//! `derivations/dsst_srp_conical_shadow.mac` proves the cone equations, their quartic, the bound
//! that skips it, and the substitution. `derivations/dsst_srp_conical_average.gp` takes the average
//! at 80 digits by routes that share nothing with this module: the contacts by Brent's method on
//! the angles themselves, `F` from exact derivatives of a Cartesian → equinoctial conversion, and
//! the integral by double-exponential quadrature, once in the true longitude and once in the mean
//! longitude. `tests/dsst_srp_conical_reference.rs` holds the rates to it, and the tests below hold
//! the contacts to a bisection on the same angles and the short-period Fourier coefficients to the
//! same reference.
//!
//! # Scoping: spherical central body
//!
//! Orekit's conical lighting ratio uses `OneAxisEllipsoid.pointOnLimb`, whose limb radius depends on
//! the body-frame (ITRF) orientation for a flattened ellipsoid. With **flattening = 0** (a sphere of
//! radius `ae`) the occultation angles are rotation-invariant, so the limb radius reduces to
//! `ρ_occ = asin(ae / |r|)` and the whole ratio is computable directly from the inertial (GCRF)
//! satellite and Sun positions — no Earth-orientation transform.

use core::f64::consts::{PI, TAU};
use std::vec::Vec;

use libm::{atan, atan2, cbrt, copysign, cos, fabs, floor, sin, sqrt};

use super::auxiliary::DsstAuxiliary;
use super::gaussian::{self, NodeState};
use super::gaussian_short_period;
use crate::shadow;

/// Default reference distance for the solar radiation pressure (m) — `DSSTSolarRadiationPressure.D_REF`.
pub const D_REF: f64 = 149_597_870_000.0;

/// Default reference solar radiation pressure at [`D_REF`] (N/m²) — `DSSTSolarRadiationPressure.P_REF`.
pub const P_REF: f64 = 4.56e-6;

/// Sun radius used for the occultation apparent radius (m) — `Constants.SUN_RADIUS`.
pub const SUN_RADIUS: f64 = 6.957e8;

/// Threshold for the shadow equation `|S| < S_ZERO` (`DSSTSolarRadiationPressure.S_ZERO`).
const S_ZERO: f64 = 1.0e-6;

/// Gauss–Legendre order on a penumbra piece, after [`gaussian::smoothstep_nodes`]: every penumbra
/// piece measured reaches the rounding of its integrand by order 32, and 40 keeps a step of margin
/// (`the_penumbra_rule_has_converged`).
const PENUMBRA_ORDER: usize = 40;

/// Solar-radiation-pressure contribution to the DSST mean-element rates (cannonball spacecraft).
///
/// Holds the reflection coefficient `cr`, the area-to-mass ratio `S/m` (m²/kg), the reference
/// pressure/distance `P_ref`/`d_ref`, the central attraction `µ`, and the central-body equatorial
/// radius `ae` (for the shadow). Matches Orekit's `IsotropicRadiationSingleCoefficient` with a unit
/// global factor: the acceleration scale is `cr · (S/m)`.
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
    /// Central-body equatorial radius `ae`, m (shadow geometry).
    pub ae: f64,
}

/// An arc of the orbit between two consecutive shadow contacts, in true longitude, with the Sun at
/// least partly in view.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Piece {
    low: f64,
    high: f64,
    /// Part of the Sun's disc is hidden, so `ν` has a `(L − L₀)^{3/2}` end at each contact.
    penumbra: bool,
}

/// The arcs of the orbit the averages run over.
#[derive(Debug, Clone, PartialEq)]
enum Window {
    /// The orbit never meets the penumbra cone: the whole turn `[L − π, L + π]`, `L` the true
    /// longitude of the state.
    FullTurn([f64; 2]),
    /// The lit and penumbra pieces between the contacts; the umbra is left out.
    Pieces(Vec<Piece>),
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
    /// The average `(1/2π)∫νF dλ` over the whole orbit, taken in the true longitude on the pieces
    /// between the shadow contacts (module doc). Zero when no piece sees the Sun.
    #[must_use]
    pub fn mean_element_rates(&self, aux: &DsstAuxiliary, sun_pos: [f64; 3]) -> [f64; 6] {
        let accel = self.acceleration(sun_pos);
        match self.window(aux, sun_pos) {
            Window::FullTurn(ll) => gaussian::mean_element_rates(aux, ll[0], ll[1], accel),
            Window::Pieces(pieces) => {
                let (nodes, weights) = rule(&pieces);
                let mut rate = gaussian::integrate_rule(
                    aux,
                    nodes.iter().copied().zip(weights.iter().copied()),
                    &accel,
                );
                let coef = 1.0 / (2.0 * PI * aux.big_b);
                for r in &mut rate {
                    *r *= coef;
                }
                rate
            }
        }
    }

    /// The per-node cannonball SRP acceleration closure `q(node) = scale·λ·k_ref·(r − r☉)/|r − r☉|³`,
    /// with `scale = cr·(S/m)`, `k_ref = P_ref·d_ref²` and `λ` the conical lighting ratio
    /// ([`crate::shadow::lighting_ratio`]) — Orekit's `SolarRadiationPressure.acceleration` +
    /// `IsotropicRadiationSingleCoefficient`. Shared by the mean-rate averaging and the short-period
    /// Fourier quadrature, so both see the identical acceleration.
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
            let ratio = shadow::lighting_ratio(node.pos, sun_pos, ae);
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

    /// The osculating equinoctial state `mean + short-period` for a **mean** equinoctial state
    /// `[a, k, h, q, p, λ]` and this SRP model's Sun position `sun_pos` (m). The short-period
    /// Fourier coefficients are taken on the pieces the mean rates integrate, by the shared
    /// Gaussian engine ([`gaussian_short_period`]).
    #[must_use]
    pub fn osculating(&self, mean: [f64; 6], sun_pos: [f64; 3]) -> [f64; 6] {
        let aux = DsstAuxiliary::new(
            mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], self.mu,
        );
        let accel = self.acceleration(sun_pos);
        let sp = match self.window(&aux, sun_pos) {
            Window::FullTurn(ll) => gaussian_short_period::short_period_correction(&aux, ll, accel),
            Window::Pieces(pieces) => {
                let (nodes, weights) = rule(&pieces);
                gaussian_short_period::short_period_correction_on(&aux, &nodes, &weights, accel)
            }
        };
        let mut osc = mean;
        for i in 0..6 {
            osc[i] += sp[i];
        }
        osc
    }

    /// The pieces of the orbit the averages run over: between consecutive contacts, each lit,
    /// penumbra or umbra by the lighting ratio at its middle. A penumbra piece between two contacts
    /// with the penumbra cone is cut again where it runs deepest, at the least `S` of the umbra cone
    /// ([`Cone::dip`]): as the orbit nears that cone `ν` there tends to `|L − L*|³`, and short of
    /// it the umbra cone's complex contacts sit just off `L*`, so only the end of a piece can carry
    /// that point at the rule's convergence (`gaussian::smoothstep_nodes`).
    ///
    /// A contact the quartic admits in error can only split a piece in two of the same kind; only a
    /// missed one merges two kinds, and the quartic misses a pair only where it merges into a double
    /// root, at a graze whose depth is at the rounding.
    fn window(&self, aux: &DsstAuxiliary, sun_pos: [f64; 3]) -> Window {
        let lvn = normalize_angle(mean_to_true(aux.k, aux.h, aux.lambda), 0.0);
        let full = Window::FullTurn([-PI + lvn, PI + lvn]);
        let Some((mut contacts, umbra)) = self.contacts(aux, sun_pos) else {
            return full;
        };
        let first = contacts[0].l;
        for c in &mut contacts {
            c.l = first + normalize_angle(c.l - first, PI);
        }
        contacts.sort_by(|x, y| x.l.total_cmp(&y.l));
        let wrap = Contact {
            l: first + TAU,
            ..contacts[0]
        };
        let mut pieces = Vec::with_capacity(contacts.len() + 1);
        let mut shadowed = false;
        for (i, &low) in contacts.iter().enumerate() {
            let high = contacts.get(i + 1).copied().unwrap_or(wrap);
            if high.l > low.l {
                let nu =
                    shadow::lighting_ratio(position(aux, 0.5 * (low.l + high.l)), sun_pos, self.ae);
                shadowed |= nu < 1.0;
                if nu > 0.0 {
                    let penumbra = nu < 1.0;
                    let dip = if penumbra && !low.umbra && !high.umbra {
                        umbra.dip(low.l, high.l)
                    } else {
                        None
                    };
                    let piece = |low, high| Piece {
                        low,
                        high,
                        penumbra,
                    };
                    match dip {
                        Some(l) => pieces.extend([piece(low.l, l), piece(l, high.l)]),
                        None => pieces.push(piece(low.l, high.l)),
                    }
                }
            }
        }
        if shadowed {
            Window::Pieces(pieces)
        } else {
            full
        }
    }

    /// The crossings of the orbit with the penumbra cone and the umbra cone, in no particular order,
    /// and the umbra cone; `None` when the orbit does not meet the penumbra.
    fn contacts(&self, aux: &DsstAuxiliary, sun_pos: [f64; 3]) -> Option<(Vec<Contact>, Cone)> {
        let distance = gaussian::norm(sun_pos);
        // `lighting_ratio`'s own short circuit: a Sun within two of its radii of the centre leaves
        // the whole orbit lit.
        if distance < 2.0 * shadow::SUN_RADIUS {
            return None;
        }
        let sun_dir = normalize(sun_pos);
        let alpha = gaussian::dot_ext(sun_dir, aux.f_vec);
        let beta = gaussian::dot_ext(sun_dir, aux.g_vec);
        let gamma = gaussian::dot_ext(sun_dir, aux.w_vec);
        // `sin θ` of the penumbra cone, whose vertex is the inner centre of similitude of the two
        // spheres. A point of the orbit plane at distance `r` stands at least `|γ|·r` off the shadow
        // axis and at most `r·√(1 − γ²)` behind the centre, where the cone is
        // `ae/cos θ + tan θ·depth` wide; so an orbit whose perigee has
        // `r_p·(|γ| cos θ − √(1 − γ²) sin θ) ≥ ae` never enters the penumbra.
        let penumbra = (self.ae + shadow::SUN_RADIUS) / distance;
        let cos_p = sqrt((1.0 - penumbra) * (1.0 + penumbra));
        let planar = sqrt((1.0 - gamma) * (1.0 + gamma));
        if aux.sma * (1.0 - aux.ecc) * (fabs(gamma) * cos_p - planar * penumbra) >= self.ae {
            return None;
        }
        let mut out = Vec::new();
        Cone::new(self.ae, aux, [alpha, beta], penumbra, false).contacts(&mut out);
        if out.is_empty() {
            return None;
        }
        // The umbra cone's vertex is the outer centre of similitude, behind the central body:
        // `sin θ` signed, negative for a Sun larger than the body, and the cone narrows.
        let umbra = Cone::new(
            self.ae,
            aux,
            [alpha, beta],
            (self.ae - shadow::SUN_RADIUS) / distance,
            true,
        );
        umbra.contacts(&mut out);
        Some((out, umbra))
    }
}

/// A crossing of the orbit with a shadow cone.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Contact {
    /// The true longitude.
    l: f64,
    /// On the umbra cone, where `ν` reaches 0; else on the penumbra cone, where it leaves 1.
    umbra: bool,
}

/// One of the two shadow cones, along the Keplerian orbit of a state.
///
/// The cone's generators touch the central body and the Sun, on opposite sides for the penumbra and
/// on the same side for the umbra. With `ŝ` the Sun's direction and `θ` the half-angle, signed, its
/// surface is `|r_⊥| = ae/cos θ − tan θ·(r·ŝ)`, and on the orbit `r = p/w·(cos L·f + sin L·g)`, with
/// `w = 1 + k cos L + h sin L` and `p = a·B²`, it is `S(L) = 1 − cos²Φ − (m·w − τ·cos Φ)² = 0`:
/// `cos Φ = α cos L + β sin L` the cosine of the Sun's angle from the radius, `m = ae/(p cos θ)`,
/// `τ = tan θ`. `τ = 0` is Danielson's cylinder (3.5-1). `S` is positive outside the cone.
#[derive(Debug, Clone, Copy)]
struct Cone {
    /// `ŝ·f` and `ŝ·g`.
    alpha: f64,
    beta: f64,
    k: f64,
    h: f64,
    m: f64,
    tau: f64,
    umbra: bool,
}

impl Cone {
    /// The cone of half-angle `asin(sine)` on the orbit of `aux`, with `α = ŝ·f` and `β = ŝ·g`.
    fn new(ae: f64, aux: &DsstAuxiliary, [alpha, beta]: [f64; 2], sine: f64, umbra: bool) -> Self {
        let cosine = sqrt((1.0 - sine) * (1.0 + sine));
        // `m = ae/(p cos θ)` with `p = a·B²` the semi-latus rectum. That is Escobal (1976) eq.
        // (A.1) for the cylinder and its equinoctial form in Valk & Lemaître (2008),
        // `ae²(1 + k cos L + h sin L)² + p² cos²Φ − p² = 0`, and it is what Danielson's own eq.
        // 2.1.4-6 `r = a(1 − h² − k²)/(1 + h sin L + k cos L)` forces. Danielson 3.5 nevertheless
        // prints `M = ae²/(a²(1 − h² − k²))`, one power of `B²` short, and Orekit transcribes that
        // as `ae / (a * B)` — a cylinder of radius `ae·B`, narrowing with eccentricity.
        Self {
            alpha,
            beta,
            k: aux.k,
            h: aux.h,
            m: ae / (aux.sma * aux.big_b * aux.big_b * cosine),
            tau: sine / cosine,
            umbra,
        }
    }

    /// `cos Φ`, the cone's radius over `r`, `S` and half its derivative at a true longitude.
    fn at(&self, l: f64) -> (f64, f64, f64, f64) {
        let (sin_l, cos_l) = (sin(l), cos(l));
        let c_phi = self.alpha * cos_l + self.beta * sin_l;
        let radius = self.m * (1.0 + self.k * cos_l + self.h * sin_l) - self.tau * c_phi;
        let s = 1.0 - c_phi * c_phi - radius * radius;
        let c_phi_dl = self.beta * cos_l - self.alpha * sin_l;
        let radius_dl = self.m * (self.h * cos_l - self.k * sin_l) - self.tau * c_phi_dl;
        (c_phi, radius, s, -c_phi * c_phi_dl - radius * radius_dl)
    }

    /// Half the second derivative of `S` at a true longitude. With `cos Φ'' = −cos Φ` and
    /// `w'' = 1 − w` it is `cos²Φ − cos Φ'² − R'² − R·(m(1 − w) + τ·cos Φ)`, `R` the radius over `r`.
    fn half_curvature(&self, l: f64) -> f64 {
        let (sin_l, cos_l) = (sin(l), cos(l));
        let w = 1.0 + self.k * cos_l + self.h * sin_l;
        let c_phi = self.alpha * cos_l + self.beta * sin_l;
        let c_phi_dl = self.beta * cos_l - self.alpha * sin_l;
        let radius = self.m * w - self.tau * c_phi;
        let radius_dl = self.m * (self.h * cos_l - self.k * sin_l) - self.tau * c_phi_dl;
        c_phi * c_phi
            - c_phi_dl * c_phi_dl
            - radius_dl * radius_dl
            - radius * (self.m * (1.0 - w) + self.tau * c_phi)
    }

    /// The contacts with this cone, appended to `out`: the real roots of `S` by a quartic in
    /// `tan((L − L₀)/2)`, polished by Newton on `S`. They count on the shadow side, `cos Φ < 0`
    /// (3.5-3), and on the nappe that holds the central body, `m·w − τ·cos Φ > 0`. Both conditions
    /// are exact for a satellite more than `ae(1/cos θ − 1)`, about 70 m for the Earth, above the
    /// surface and inside the umbra cone's vertex (`derivations/dsst_srp_conical_shadow.mac`).
    fn contacts(&self, out: &mut Vec<Contact>) {
        let Self {
            alpha,
            beta,
            k,
            h,
            m,
            tau,
            umbra,
        } = *self;
        // Measure the longitude from `L₀`, the anti-Sun direction in the orbit plane, and solve in
        // `t = tan((L − L₀)/2)`. There `cos Φ = −ρ·cos u`, `ρ = √(α² + β²)` and `u = L − L₀`, so the
        // night side is `|t| < 1` and the crossings either side of the shadow axis are two roots in
        // `t`, where a quartic in `cos L` would fold them onto one: a double root whenever the
        // geometry is symmetric about the axis, which a circular orbit always is.
        let planar = libm::hypot(alpha, beta);
        let (cos_0, sin_0) = (-alpha / planar, -beta / planar);
        let (ku, hu) = (k * cos_0 + h * sin_0, h * cos_0 - k * sin_0);
        // S = a·C² + b·C·Σ + c·Σ² + d·C + e·Σ + f in C = cos u, Σ = sin u, with `α` → `−ρ`, `β` → 0.
        let (m2, mt, t1) = (m * m, 2.0 * m * tau, 1.0 + tau * tau);
        let sa = -m2 * ku * ku - mt * ku * planar - t1 * planar * planar;
        let sb = -2.0 * m2 * ku * hu - mt * hu * planar;
        let sc = -m2 * hu * hu;
        let sd = -2.0 * m2 * ku - mt * planar;
        let se = -2.0 * m2 * hu;
        let sf = 1.0 - m2;
        // S·(1 + t²)² with C = (1 − t²)/(1 + t²) and Σ = 2t/(1 + t²).
        let quartic = [
            sa - sd + sf,
            2.0 * (se - sb),
            2.0 * (2.0 * sc - sa + sf),
            2.0 * (sb + se),
            sa + sd + sf,
        ];
        let l_0 = atan2(sin_0, cos_0);
        let mut roots = [0.0_f64; 4];
        let count = real_quartic_roots(&quartic, &mut roots);
        for &t in roots.iter().take(count) {
            // Ferrari's closed form leaves the roots a few digits short; `L` is well conditioned at a
            // crossing that is not a graze (`dS/dL` is `O(1)`), so Newton on `S` recovers them.
            let seed = l_0 + 2.0 * atan(t);
            let mut l = seed;
            for _ in 0..8 {
                let (_, _, s, half_slope) = self.at(l);
                if half_slope == 0.0 {
                    break;
                }
                let step = s / (2.0 * half_slope);
                l -= step;
                if fabs(step) <= f64::EPSILON * PI {
                    break;
                }
            }
            // A seed is already within `√ε` of its root; an iteration that travelled further has
            // left the basin and landed on somebody else's.
            if fabs(normalize_angle(l - seed, 0.0)) > 1.0e-3 {
                l = seed;
            }
            let (c_phi, radius, s, _) = self.at(l);
            if c_phi < 0.0 && radius > 0.0 && fabs(s) < S_ZERO {
                out.push(Contact { l, umbra });
            }
        }
    }

    /// Where `S` is least on `(low, high)`: Newton on `dS/dL`, kept inside its sign change by
    /// bisection. `None` when `dS/dL` does not run from negative to positive across the arc.
    fn dip(&self, low: f64, high: f64) -> Option<f64> {
        let (mut lo, mut hi) = (low, high);
        if !(self.at(lo).3 < 0.0 && self.at(hi).3 > 0.0) {
            return None;
        }
        let mut l = 0.5 * (lo + hi);
        for _ in 0..64 {
            let half_slope = self.at(l).3;
            if half_slope < 0.0 {
                lo = l;
            } else if half_slope > 0.0 {
                hi = l;
            } else {
                break;
            }
            let curvature = self.half_curvature(l);
            let newton = l - half_slope / curvature;
            let next = if curvature > 0.0 && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            let step = next - l;
            l = next;
            if fabs(step) <= f64::EPSILON * fabs(l).max(1.0) {
                break;
            }
        }
        Some(l)
    }
}

/// The quadrature on the pieces, as true longitudes and weights: Gauss–Legendre at order 48 on a
/// lit piece, and at [`PENUMBRA_ORDER`] after the smoothstep substitution on a penumbra piece.
fn rule(pieces: &[Piece]) -> (Vec<f64>, Vec<f64>) {
    let mut out = (Vec::new(), Vec::new());
    for piece in pieces {
        if piece.penumbra {
            out.extend(gaussian::smoothstep_nodes(
                PENUMBRA_ORDER,
                piece.low,
                piece.high,
            ));
        } else {
            out.extend(gaussian::gauss_nodes(
                gaussian::MAX_GAUSS_ORDER,
                piece.low,
                piece.high,
            ));
        }
    }
    out
}

/// The position at true longitude `l` on the Keplerian orbit of `aux`.
fn position(aux: &DsstAuxiliary, l: f64) -> [f64; 3] {
    let (sin_l, cos_l) = (sin(l), cos(l));
    let r = aux.sma * aux.big_b * aux.big_b / (1.0 + aux.k * cos_l + aux.h * sin_l);
    core::array::from_fn(|i| r * (cos_l * aux.f_vec[i] + sin_l * aux.g_vec[i]))
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
    // fall the right way, and `Cone::contacts` builds exactly this quartic for every circular orbit,
    // and one within rounding of it whenever the line of apsides lies along the Sun: `a[1]` and
    // `a[3]` carry the eccentricity across the shadow axis. Losing the roots silently removes the
    // eclipse and averages the whole revolution as sunlit. Orekit 13.1.7 has no such branch.
    //
    // Solving `a₀u² + a₂u + a₄` for `u = x²` leaves one discriminant instead of four.
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
    // every perfect-square quartic through exactly this configuration. The `fma` makes the
    // subtraction correctly rounded; the window below admits `4` ulp of `max(b², |c|)` for the error
    // `b` and `c` arrive with. Orekit (13.1.7) tests `d < 0` exactly.
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
    // GCRF, SPHERICAL Earth of radius ae. Direct fixed-state getMeanElementRate, and the integration
    // limits getLLimits used. The two tests that read them are comparators.
    const MU: f64 = 3.98600441500000000e+14;
    const AE: f64 = 6.37813646000000000e+06;
    const CR: f64 = 1.50000000000000000e+00;
    const AREA: f64 = 1.00000000000000000e+01;
    const MASS: f64 = 1.00000000000000000e+03;

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

    /// Orekit's two shadow crossings on its fixture, against the cylinder they lie on.
    ///
    /// Orekit sets `m = ae / (a·B)` in the shadow equation `S = 1 − m²·range² − cos²Φ`. With the
    /// equinoctial radius `r = a·B²/range` that reads `S = 1 − (ae·B/r)² − cos²Φ`, so `S = 0` puts
    /// the boundary at a perpendicular distance `ae·B` from the anti-Sun axis: a cylinder that
    /// narrows as the orbit gets more eccentric. Danielson 3.5 prints `M = ae²/(a²(1 − h² − k²))`;
    /// his own eq. 2.1.4-6 gives `r = a(1 − h² − k²)/(1 + h sin L + k cos L)`, so the shadow
    /// equation needs `M = (ae/(a(1 − h² − k²)))²`, one more power of `B²`. Escobal (1976) writes the
    /// same cylinder as `ae²(1 + e cos f)² + p²cos²Φ − p² = 0` with the semi-latus rectum
    /// `p = a(1 − e²)`, and Valk & Lemaître (2008) restate it in `h`, `k`; both divide by `p²`, not
    /// by `a²B²`. Measured: both crossings 6 377 817.545 m from the Sun line, where `ae·B` is
    /// 6 377 817.545 m and `ae` is 6 378 136.46 m.
    #[test]
    #[ignore = "comparator: the distance of Orekit's limits, not a verdict"]
    fn orekit_limits_sit_on_the_ae_b_cylinder() {
        let a = aux();
        let dir = normalize(SUN_POS);
        let alpha = gaussian::dot_ext(dir, a.f_vec);
        let beta = gaussian::dot_ext(dir, a.g_vec);
        for l in [LL_LOW, LL_HIGH] {
            let c_phi = alpha * cos(l) + beta * sin(l);
            let perpendicular = gaussian::norm(position(&a, l)) * sqrt(1.0 - c_phi * c_phi);
            std::println!(
                "Orekit's crossing at L = {l}: {perpendicular:.3} m from the Sun line; ae·B = {:.3} m, \
                 ae = {AE:.3} m",
                AE * a.big_b
            );
        }
    }

    /// The six fixed-state rates against Orekit's `getMeanElementRate` on its fixture. Orekit cuts the
    /// average at the cylinder of [`orekit_limits_sit_on_the_ae_b_cylinder`] and weighs it by the
    /// conical `ν` inside, which drops the light of the inner half of each penumbra and keeps the
    /// deficit of the outer half; this module averages over the whole orbit. Measured: from 8.3e-4
    /// apart (`dk`, `dh`) to 3.3e-3 (`dλ`), relative.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn fixed_state_rates_distance_to_orekit() {
        let got = srp().mean_element_rates(&aux(), SUN_POS);
        let rel: [f64; 6] = core::array::from_fn(|i| libm::fabs((got[i] - RATES[i]) / RATES[i]));
        std::println!(
            "SRP mean rates vs Orekit 13: da {:.3e} dk {:.3e} dh {:.3e} dq {:.3e} dp {:.3e} dλ {:.3e}",
            rel[0],
            rel[1],
            rel[2],
            rel[3],
            rel[4],
            rel[5]
        );
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
    /// from inside `Cone::contacts`, where the shadow geometry drives one discriminant arm per call. That
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

    /// The contacts of the orbit with the penumbra cone (`side = 1`) or the umbra cone (`side = −1`),
    /// as true longitudes in `[−π, π)`, by bracketing and bisecting `d − ρ_E − side·ρ_S`: the angle
    /// between the centres of the two discs seen from the satellite, less the sum or the difference
    /// of their angular radii. That is the condition `ν` leaves 1 and reaches 0 at, written from the
    /// geometry; it shares none of the solution with the code: no cone, no quartic, no Newton step.
    fn contacts_by_bisection(aux: &DsstAuxiliary, sun: [f64; 3], side: f64) -> Vec<f64> {
        /// Samples per revolution — fine enough that no shadow arc here falls between two of them.
        const N: usize = 8192;
        let edge = |l: f64| {
            let (c, s) = (cos(l), sin(l));
            let rn = aux.sma * aux.big_b * aux.big_b / (1.0 + aux.k * c + aux.h * s);
            let r: [f64; 3] = core::array::from_fn(|i| rn * (c * aux.f_vec[i] + s * aux.g_vec[i]));
            let to_sun: [f64; 3] = core::array::from_fn(|i| sun[i] - r[i]);
            let sep = atan2(
                gaussian::norm(gaussian::cross(r, to_sun)),
                -gaussian::dot(r, to_sun),
            );
            sep - libm::asin(AE / rn)
                - side * libm::asin(shadow::SUN_RADIUS / gaussian::norm(to_sun))
        };
        let mut out = Vec::new();
        let mut prev = edge(-PI);
        for i in 1..=N {
            let l = -PI + TAU * i as f64 / N as f64;
            let cur = edge(l);
            if (prev < 0.0) != (cur < 0.0) {
                let (mut lo, mut hi) = (l - TAU / N as f64, l);
                for _ in 0..64 {
                    let mid = 0.5 * (lo + hi);
                    if (edge(lo) < 0.0) == (edge(mid) < 0.0) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                out.push(normalize_angle(0.5 * (lo + hi), 0.0));
            }
            prev = cur;
        }
        out
    }

    /// `(a, k, h, q, p)` of the sweeps: LEO through a deep HEO, and inclinations from equatorial to
    /// retrograde. Every perigee clears the surface.
    const ORBITS: [[f64; 5]; 7] = [
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

    /// A Sun at one astronomical unit, `γ` the sine of its angle out of the orbit plane of `aux` and
    /// its part in the plane at `φ` from `f`.
    fn sun_on_the_triad(aux: &DsstAuxiliary, gamma: f64, phi: f64) -> [f64; 3] {
        let planar = sqrt((1.0 - gamma) * (1.0 + gamma));
        let (ca, cb) = (planar * cos(phi), planar * sin(phi));
        core::array::from_fn(|j| {
            1.495_978_707e11 * (ca * aux.f_vec[j] + cb * aux.g_vec[j] + gamma * aux.w_vec[j])
        })
    }

    /// The Sun directions of the sweeps for one orbit: `γ` on the triad's own grid, which sets how
    /// far out of plane the Sun is, and around `ae/r_p`, where the orbit grazes the cones and meets
    /// the penumbra without the umbra; `φ` turns the Sun around the plane and moves the contacts past
    /// the ±π seam, and puts it on the line of apsides, where the crossings either side of the
    /// shadow axis mirror each other and the quartic's odd terms are rounding.
    fn suns(aux: &DsstAuxiliary) -> Vec<[f64; 3]> {
        let graze = AE / (aux.sma * (1.0 - aux.ecc));
        let mut gammas: Vec<f64> = (-3..=3_i32).map(|i| f64::from(i) * 0.3).collect();
        for step in [-0.03, -0.01, -0.003, 0.0, 0.003, 0.01] {
            let g = graze * (1.0 + step);
            if g < 1.0 {
                gammas.extend([g, -g]);
            }
        }
        let perigee = atan2(aux.h, aux.k);
        let mut out = Vec::new();
        for gamma in gammas {
            for i in 0..16_u32 {
                out.push(sun_on_the_triad(aux, gamma, f64::from(i) * (TAU / 16.0)));
            }
            out.push(sun_on_the_triad(aux, gamma, perigee));
            out.push(sun_on_the_triad(aux, gamma, perigee + PI));
        }
        out
    }

    /// The contacts of the orbit with both shadow cones, against a bisection on the angles that
    /// define them ([`contacts_by_bisection`]).
    ///
    /// `contacts` reduces each cone to a quartic in `tan((L − L₀)/2)`, takes its real roots, polishes
    /// them by Newton on `S` and keeps those on the shadow side and on the nappe that holds the central
    /// body. Every crossing the bisection finds must be among them, and nothing else. The sweep
    /// reaches orbits that miss the shadow, orbits that meet the penumbra alone, and orbits that meet
    /// both cones — otherwise it proves nothing about the code it does not execute. Measured 6.2e-14
    /// rad at worst over 2394 states.
    #[test]
    fn the_contacts_are_the_crossings_of_both_cones() {
        let model = srp();
        let mut worst = 0.0_f64;
        let (mut none, mut penumbra_only, mut both) = (0_u32, 0_u32, 0_u32);
        for [a, k, h, q, p] in ORBITS {
            let aux = DsstAuxiliary::new(a, k, h, q, p, 0.0, MU);
            for sun in suns(&aux) {
                let want_p = contacts_by_bisection(&aux, sun, 1.0);
                let want_u = contacts_by_bisection(&aux, sun, -1.0);
                match (want_p.len(), want_u.len()) {
                    (0, 0) => none += 1,
                    (2, 0) => penumbra_only += 1,
                    (2, 2) => both += 1,
                    counts => {
                        panic!("{counts:?} crossings: the sweep met a graze it cannot resolve")
                    }
                }
                let got = model.contacts(&aux, sun).map_or_else(Vec::new, |(c, _)| c);
                let want: Vec<Contact> = want_p
                    .iter()
                    .map(|&l| Contact { l, umbra: false })
                    .chain(want_u.iter().map(|&l| Contact { l, umbra: true }))
                    .collect();
                assert_eq!(
                    got.len(),
                    want.len(),
                    "a={a:e} k={k} h={h} q={q} p={p} sun={sun:?}: {got:?} against {want:?}"
                );
                for w in &want {
                    let miss = got
                        .iter()
                        .filter(|g| g.umbra == w.umbra)
                        .map(|g| libm::fabs(normalize_angle(g.l - w.l, 0.0)))
                        .fold(f64::INFINITY, f64::min);
                    worst = nan_max(worst, miss);
                }
            }
        }
        assert!(
            none > 0 && penumbra_only > 0 && both > 0,
            "coverage: none {none}, penumbra only {penumbra_only}, both {both}"
        );
        assert!(worst < 1e-12, "worst contact disagreement {worst:.3e} rad");
        std::println!(
            "contacts vs bisection: worst {worst:.3e} rad over {} states (none {none}, penumbra \
             only {penumbra_only}, both {both})",
            none + penumbra_only + both
        );
    }

    /// Whether the window's pieces agree with the lighting ratio `ν` along the orbit: a lit piece is
    /// lit throughout, a penumbra piece is in the penumbra throughout, and what the pieces leave out
    /// of the turn is umbra. A full turn is lit everywhere and centred on the state's true
    /// longitude. Each piece and gap is sampled at seven interior points, away from its ends.
    fn assert_window(model: &DsstSolarRadiationPressure, aux: &DsstAuxiliary, sun: [f64; 3]) {
        let nu = |l: f64| shadow::lighting_ratio(position(aux, l), sun, AE);
        let interior =
            |low: f64, high: f64| (1..8_u32).map(move |i| low + (high - low) * f64::from(i) / 8.0);
        match model.window(aux, sun) {
            Window::FullTurn(ll) => {
                let lv = true_longitude_classical(aux.k, aux.h, aux.lambda);
                let lvn = lv - TAU * libm::round(lv / TAU);
                assert!(
                    libm::fabs(ll[0] - (lvn - PI)) < 1e-12
                        && libm::fabs(ll[1] - (lvn + PI)) < 1e-12,
                    "the full turn {ll:?} is not centred on the true longitude {lvn}"
                );
                for l in
                    interior(ll[0], ll[1]).chain((0..64_u32).map(|i| f64::from(i) * TAU / 64.0))
                {
                    assert!(nu(l) == 1.0, "a full turn with ν({l}) = {}", nu(l));
                }
            }
            Window::Pieces(pieces) => {
                assert!(!pieces.is_empty(), "no piece sees the Sun");
                for pair in pieces.windows(2) {
                    assert!(
                        pair[0].high <= pair[1].low,
                        "pieces out of order: {pieces:?}"
                    );
                }
                let (first, last) = (pieces[0], pieces[pieces.len() - 1]);
                assert!(
                    last.high <= first.low + TAU * (1.0 + 4.0 * f64::EPSILON),
                    "the pieces span more than a turn: {pieces:?}"
                );
                for piece in &pieces {
                    assert!(piece.low < piece.high, "an empty piece: {piece:?}");
                    for l in interior(piece.low, piece.high) {
                        let v = nu(l);
                        let fits = if piece.penumbra {
                            v > 0.0 && v < 1.0
                        } else {
                            v == 1.0
                        };
                        assert!(fits, "ν({l}) = {v} inside {piece:?}");
                    }
                }
                let gaps = pieces
                    .windows(2)
                    .map(|pair| (pair[0].high, pair[1].low))
                    .chain(core::iter::once((last.high, first.low + TAU)));
                for (low, high) in gaps {
                    if high - low > 1e-9 {
                        for l in interior(low, high) {
                            assert!(nu(l) == 0.0, "ν({l}) = {} in a gap between pieces", nu(l));
                        }
                    }
                }
            }
        }
    }

    /// The window over the same sweep as the contacts, at two mean longitudes for the centre of the
    /// full turn.
    #[test]
    fn the_pieces_follow_the_lighting_ratio() {
        let model = srp();
        let (mut full, mut pieces) = (0_u32, 0_u32);
        for [a, k, h, q, p] in ORBITS {
            for lambda in [0.0, 2.5] {
                let aux = DsstAuxiliary::new(a, k, h, q, p, lambda, MU);
                for sun in suns(&aux) {
                    assert_window(&model, &aux, sun);
                    match model.window(&aux, sun) {
                        Window::FullTurn(_) => full += 1,
                        Window::Pieces(_) => pieces += 1,
                    }
                }
            }
        }
        assert!(
            full > 0 && pieces > 0,
            "coverage: full {full}, pieces {pieces}"
        );
    }

    /// The rule `rule` of the mean-rate integrand over the pieces, unscaled and summed with Kahan's
    /// compensation, and the sum of its absolute terms, component by component: the scale of the
    /// sum's rounding.
    fn integral_and_magnitude(
        aux: &DsstAuxiliary,
        accel: &impl Fn(&NodeState) -> [f64; 3],
        rule: impl Iterator<Item = (f64, f64)>,
    ) -> ([f64; 6], [f64; 6]) {
        let (mut sum, mut carry, mut magnitude) = ([0.0_f64; 6], [0.0_f64; 6], [0.0_f64; 6]);
        for (x, w) in rule {
            let v = gaussian::value(aux, x, accel);
            for i in 0..6 {
                let y = w * v[i] - carry[i];
                let t = sum[i] + y;
                carry[i] = (t - sum[i]) - y;
                sum[i] = t;
                magnitude[i] += libm::fabs(w * v[i]);
            }
        }
        (sum, magnitude)
    }

    /// The largest difference of two sets of rates against the largest magnitude, `da` over `a` so
    /// that all six are rates of angles (Orekit's `getRatesDiff` weighting). A component the force
    /// does not reach, `dq` and `dp` with the Sun in an equatorial orbit's plane, is rounding on both
    /// sides, and would be noise against its own magnitude.
    fn rate_distance(a: f64, got: [f64; 6], want: [f64; 6], magnitude: [f64; 6]) -> f64 {
        let weight = |i: usize| if i == 0 { 1.0 / a } else { 1.0 };
        let scale = (0..6).fold(0.0_f64, |m, i| nan_max(m, weight(i) * magnitude[i]));
        (0..6).fold(0.0_f64, |m, i| {
            nan_max(m, weight(i) * libm::fabs(got[i] - want[i]))
        }) / scale
    }

    /// `piece` in `parts` equal parts, each under the smoothstep substitution at order 48, the parts
    /// sharing their edges exactly: a rule with `parts` times the nodes of the shipped one and
    /// `parts` times finer at the ends.
    fn in_parts(piece: &Piece, parts: u32) -> impl Iterator<Item = (f64, f64)> + '_ {
        let width = (piece.high - piece.low) / f64::from(parts);
        let edge = move |i: u32| {
            if i == parts {
                piece.high
            } else {
                piece.low + width * f64::from(i)
            }
        };
        (0..parts).flat_map(move |i| gaussian::smoothstep_nodes(48, edge(i), edge(i + 1)))
    }

    /// Every penumbra piece of the sweep at [`PENUMBRA_ORDER`] against the same piece in eight parts
    /// ([`in_parts`]). Measured 6.6e-14 at worst over 2048 pieces; at order 24 the worst is 9.0e-13.
    ///
    /// The floor is the integrand's rounding, not the rule's. A shift `δd` of the separation moves
    /// `ν` by up to `2δd/(π r_S)`, the chord `2 r_S` over the disc `π r_S²`: 137 `δd` for the Sun,
    /// so the last bit of a 1-rad angle moves `ν` by 3e-14. On the denser band
    /// `γ ∈ [0.94, 1.06]·ae/r_p` any two node sets differ by up to 1.05e-13 from order 32 up, so the
    /// bound is twice that.
    #[test]
    fn the_penumbra_rule_has_converged() {
        let model = srp();
        let (mut worst, mut worst_24, mut count) = (0.0_f64, 0.0_f64, 0_u32);
        for [a, k, h, q, p] in ORBITS {
            let aux = DsstAuxiliary::new(a, k, h, q, p, 0.0, MU);
            for sun in suns(&aux) {
                let Window::Pieces(pieces) = model.window(&aux, sun) else {
                    continue;
                };
                let accel = model.acceleration(sun);
                for piece in pieces.iter().filter(|x| x.penumbra) {
                    count += 1;
                    let (fine, magnitude) =
                        integral_and_magnitude(&aux, &accel, in_parts(piece, 8));
                    let miss = |order| {
                        let rule = gaussian::smoothstep_nodes(order, piece.low, piece.high);
                        rate_distance(
                            a,
                            integral_and_magnitude(&aux, &accel, rule).0,
                            fine,
                            magnitude,
                        )
                    };
                    worst = nan_max(worst, miss(PENUMBRA_ORDER));
                    worst_24 = nan_max(worst_24, miss(24));
                }
            }
        }
        assert!(count > 0, "the sweep reached no penumbra");
        assert!(
            worst < 2e-13,
            "penumbra rule at order {PENUMBRA_ORDER}: {worst:.3e}"
        );
        std::println!(
            "penumbra rule over {count} pieces: order {PENUMBRA_ORDER} {worst:.3e}, order 24 \
             {worst_24:.3e}"
        );
    }

    /// The low LEO of the sweep near its tangency with the umbra cone, from both sides: the shipped
    /// rates against each penumbra piece in 64 parts ([`in_parts`]), where a kink left inside one
    /// part costs little. Short of the tangency `ν` dips towards 0 inside a piece between two
    /// penumbra contacts, and the umbra cone's complex contacts close in on its deepest point.
    /// Measured 3.8e-16 at worst, the reference settled to 2.0e-16; without the cut at the deepest
    /// point the same states are 1.9e-7 off.
    #[test]
    fn the_deepest_point_carries_the_umbra_graze() {
        let model = srp();
        let aux = DsstAuxiliary::new(7.078_137e6, 0.001, -0.0005, 0.2, -0.1, 0.0, MU);
        let phi = 2.0;
        let umbra_contacts = |gamma: f64| {
            model
                .contacts(&aux, sun_on_the_triad(&aux, gamma, phi))
                .map_or(0, |(c, _)| c.iter().filter(|x| x.umbra).count())
        };
        let (mut lo, mut hi) = (0.85, 0.9011);
        assert!(umbra_contacts(lo) == 2 && umbra_contacts(hi) == 0);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if umbra_contacts(mid) == 2 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let composite = |pieces: &[Piece], accel: &dyn Fn(&NodeState) -> [f64; 3], m: u32| {
            let rule: Vec<(f64, f64)> = pieces
                .iter()
                .flat_map(|piece| -> Vec<(f64, f64)> {
                    if piece.penumbra {
                        in_parts(piece, m).collect()
                    } else {
                        gaussian::gauss_nodes(48, piece.low, piece.high).collect()
                    }
                })
                .collect();
            integral_and_magnitude(&aux, &accel, rule.into_iter())
        };
        let (mut worst, mut settle) = (0.0_f64, 0.0_f64);
        for delta in [1e-4, 1e-6, 1e-8, 1e-10, -1e-6, -1e-10] {
            let sun = sun_on_the_triad(&aux, hi * (1.0 + delta), phi);
            let Window::Pieces(pieces) = model.window(&aux, sun) else {
                panic!("δ = {delta:e}: the orbit misses the shadow");
            };
            let accel = model.acceleration(sun);
            let (reference, magnitude) = composite(&pieces, &accel, 64);
            let (coarse, _) = composite(&pieces, &accel, 32);
            let got = model.mean_element_rates(&aux, sun);
            let coef = 1.0 / (2.0 * PI * aux.big_b);
            settle = nan_max(settle, rate_distance(aux.sma, coarse, reference, magnitude));
            let scaled = |x: [f64; 6]| x.map(|v| coef * v);
            worst = nan_max(
                worst,
                rate_distance(aux.sma, got, scaled(reference), scaled(magnitude)),
            );
        }
        assert!(settle < 1e-13, "the reference itself moved {settle:.3e}");
        assert!(worst < 1e-13, "the umbra graze: {worst:.3e}");
        std::println!(
            "umbra graze at γ = {hi:.17e}: worst {worst:.3e}, the reference settled to {settle:.3e}"
        );
    }

    /// The orbits of [`FOURIER`]: `(name, [a, k, h, q, p], Sun (m))`, two of the rate fixtures of
    /// `tests/dsst_srp_conical_reference.rs`, the Orekit LEO across both cones and the low LEO `1e-6`
    /// short of the umbra cone.
    #[expect(clippy::unreadable_literal, reason = "printed by the script and checked verbatim")]
    #[rustfmt::skip]
    const FOURIER_ORBITS: [(&str, [f64; 5], [f64; 3]); 2] = [
        ("leo", [7.2e6, 3.0901699437494747e-3, 9.510565162951536e-3, 8.54890329218965e-1, 7.69746709859868e-1], [2.4813057806275562e10, -1.3303312669583466e11, -5.7667965236002365e10]),
        ("graze", [7.078137e6, 1e-3, -5e-4, 2e-1, -1e-1], [-5.4535206342468445e10, 4.553954408493722e9, 1.3922893264836755e11]),
    ];

    /// `(name, j, Cⱼ, Sⱼ)` with `Cⱼ, Sⱼ = (1/π)∫νFᵢ·(cos jL, sin jL) dL` over the whole orbit, the
    /// coefficients of the short periods, at 80 digits by `derivations/dsst_srp_conical_average.gp`,
    /// which `derivations/verify.sh` re-runs to check that every row here is what it prints.
    #[expect(clippy::unreadable_literal, reason = "printed by the script and checked verbatim")]
    #[rustfmt::skip]
    const FOURIER: [(&str, usize, [f64; 6], [f64; 6]); 24] = [
        ("leo", 1, [6.885761492184194e-5, -7.559999616290803e-13, 5.659902996215306e-14, -8.491416113628664e-12, 8.01473324009817e-13, 4.110871438556841e-12], [-2.2142899111736048e-5, -1.931251930126904e-12, 6.806367455256701e-13, 8.01473324009817e-13, -6.257264039969068e-12, -1.1685234154944791e-11]),
        ("leo", 2, [-3.7794895923137854e-6, 3.799905898827013e-12, 1.4742454333396586e-13, -2.0550604065390683e-13, -2.2069386890845112e-12, -3.991350895022062e-12], [-5.2724122342910835e-6, -1.875062459807219e-12, 2.511972624795217e-12, 1.1121742756455758e-12, 1.2733156289682416e-12, 1.996362905620292e-12]),
        ("leo", 3, [4.690781001487515e-6, 9.305614646441193e-13, 1.592976754947631e-14, -9.791197743659195e-13, -1.2094275727463895e-12, -1.91246276130659e-12], [-6.333383090962327e-6, 5.063470059164826e-13, -5.884541073369704e-13, 3.9351907527324445e-13, -1.2550322992936769e-12, -2.421672742548453e-12]),
        ("leo", 4, [7.222934928954809e-6, -1.0824325913099048e-15, 4.779805685102623e-13, -7.330659480357931e-13, 5.474870799167515e-13, 1.230292776725416e-12], [2.4433035081413927e-6, 4.3896530923308825e-13, 2.0415371887284493e-14, -5.47277333522184e-13, -7.457557215863555e-13, -1.1961547880910761e-12]),
        ("leo", 5, [-8.820416041697522e-8, 1.3082454510735074e-13, -4.8044717041126044e-14, 5.661723806750525e-14, 1.4154398641298504e-13, 2.406255550131764e-13], [5.919356373019493e-6, 1.1277623672428082e-13, 3.635514051703783e-13, -6.743645110601599e-13, 2.192952868602521e-13, 6.1092195025408e-13]),
        ("leo", 6, [-3.1779773569002155e-6, 7.835267789191588e-14, -2.516513819647005e-13, 3.278191791962461e-13, -2.2359067719939307e-13, -5.117524013204706e-13], [9.706756921826684e-7, 4.819996832350882e-13, -8.58572724275412e-14, -2.2338093080482553e-13, -3.1512940564568363e-13, -5.094771826500792e-13]),
        ("leo", 7, [-3.2556880931824213e-7, -5.001444710221476e-13, 1.2598304898000931e-13, 1.9396509745260333e-13, 4.766258149615886e-13, 8.142139051998644e-13], [-4.671069132107512e-7, 3.2669571811535336e-13, -1.5514710600517034e-13, -5.6194709685586274e-14, -3.566431462453502e-13, -6.367757998067612e-13]),
        ("leo", 8, [-9.957773200061642e-7, -4.08069103402506e-13, 8.075245456557722e-14, 2.563999887811116e-13, 4.068839604370762e-13, 6.666986597469464e-13], [1.3030684719740677e-6, -2.8310688564021325e-13, 1.5709723760915528e-13, -4.008764756714239e-14, 3.8654859606081815e-13, 7.21220417729054e-13]),
        ("leo", 9, [-2.6674211694473834e-6, 4.301473919987593e-14, -1.6940183358985529e-13, 2.6843604208334946e-13, -2.104250786836321e-13, -4.68901033345156e-13], [-9.467917234507466e-7, -2.5691768642449865e-13, 2.740292244719979e-14, 2.1000602659237022e-13, 2.82172201614604e-13, 4.521701715520324e-13]),
        ("leo", 10, [8.529661808648494e-8, -1.9455640771969032e-14, 1.1469496704528833e-14, -2.3695597668408247e-14, -4.028055248280058e-14, -6.620756410729447e-14], [-2.864188194968126e-6, -4.883356570588243e-14, -1.5923531216917413e-13, 3.2651576038713326e-13, -1.064530096112983e-13, -2.9630069208865404e-13]),
        ("leo", 11, [1.8742972470257471e-6, -3.400837150291042e-14, 1.3054793635606687e-13, -1.951508892768414e-13, 1.2698926363753354e-13, 2.9337002522030636e-13], [-5.421160355559852e-7, -2.415016520177493e-13, 4.307421811576523e-14, 1.2657021154627167e-13, 1.8141472974558686e-13, 2.937447677702342e-13]),
        ("leo", 12, [3.0158904924083734e-7, 2.9104641558155813e-13, -7.083568884689961e-14, -1.2638154423012884e-13, -2.7521199324610815e-13, -4.656763070763955e-13], [4.471811650378663e-7, -1.822917096795955e-13, 9.239234525249083e-14, 1.1023214658224514e-14, 2.091389561730189e-13, 3.8018904116251697e-13]),
        ("graze", 1, [-5.1129592294997334e-5, 1.282883971334243e-13, 5.695286656928286e-14, -4.240624684185944e-12, -1.3645816517270594e-13, -3.888590876636255e-12], [-2.3419056765372168e-5, -2.8096830676580774e-13, -1.282504543335676e-13, -1.3645816517270594e-13, -4.007758842594176e-12, 5.0702289573932625e-12]),
        ("graze", 2, [4.229654357463031e-8, -2.0030027021502364e-12, 7.342509139351634e-13, -9.135534392056594e-14, 2.135627671946987e-13, -3.6493024633574194e-13], [-5.8955275199585175e-8, -1.0596525815958016e-12, -1.909377527774876e-12, -1.1820386641900004e-13, 2.477122157936612e-13, -4.3608485306164755e-13]),
        ("graze", 3, [-4.3974118556584925e-8, -2.868813861311177e-13, -1.2853029705226371e-13, -1.4344292607356165e-13, 3.059216961967661e-13, -5.347570548000452e-13], [-9.331579297121985e-8, 7.779487838677163e-14, 4.403708608822404e-14, 3.300536585135421e-14, -8.942291551820694e-14, 1.47757961230335e-13]),
        ("graze", 4, [-1.3418238027169975e-7, -4.9652733023858254e-14, -1.2410369950371393e-14, -3.16850223658259e-14, 4.4671572663412364e-14, -8.838272320454962e-14], [-4.5465935283227655e-9, 2.8642055106478165e-13, 1.3160655637920162e-13, 1.4003047343911105e-13, -3.073825373484013e-13, 5.333825738591474e-13]),
        ("graze", 5, [-7.719649548425132e-8, 2.3074139431985543e-13, 1.1169189075855933e-13, 1.0882019104298128e-13, -2.524272690635482e-13, 4.321017529936819e-13], [1.441180289112444e-7, 1.6039220136172467e-13, 6.239165489288039e-14, 8.64997929845721e-14, -1.6284020159833598e-13, 2.94375271357774e-13]),
        ("graze", 6, [1.1205842633637355e-7, 2.3286138125044163e-13, 9.80722749708775e-14, 1.2030796586664153e-13, -2.427500004947888e-13, 4.30549642382009e-13], [1.5387450074751698e-7, -1.3563524929773335e-13, -7.401997442168572e-14, -5.80479543922654e-14, 1.5538954911593386e-13, -2.5742296351340154e-13]),
        ("graze", 7, [2.1079302630079727e-7, -2.3376220952256743e-14, -2.696540259172742e-14, 9.992155541602781e-17, 3.856978710793599e-14, -4.998634606291635e-14], [-3.935675908694406e-8, -2.5521404207358615e-13, -1.1391999840069438e-13, -1.273576889710401e-13, 2.7156047108590123e-13, -4.747182162252162e-13]),
        ("graze", 8, [5.989915549493044e-8, -2.2737065995029735e-13, -1.0881335011660384e-13, -1.0834209965808535e-13, 2.4819681444721534e-13, -4.261499676487611e-13], [-2.2764841296249848e-7, -8.170188376534003e-14, -1.988621869294679e-14, -5.260114043983884e-14, 7.326051640879303e-14, -1.4555783025429537e-13]),
        ("graze", 9, [-1.9393664121547054e-7, -1.5895223976831442e-13, -5.788416946289637e-14, -8.875011157854487e-14, 1.585982973731917e-13, -2.9102443667300967e-13], [-1.624181936810951e-7, 1.6018375039782674e-13, 8.593072113039809e-14, 6.981039551008756e-14, -1.8271043795194032e-13, 3.0417545397985383e-13]),
        ("graze", 10, [-2.4206379685718076e-7, 7.202041789793732e-14, 5.1673924269910156e-14, 2.2161846597722356e-14, -9.30574349662237e-14, 1.4208050212385973e-13], [1.1251418114063332e-7, 1.9587442086121085e-13, 8.108135824991324e-14, 1.0253823904115282e-13, -2.0376446266460072e-13, 3.6289082580883684e-13]),
        ("graze", 11, [-6.696435983688188e-10, 1.902478425738491e-13, 8.723200853251229e-14, 9.369469680887996e-14, -2.0523149558957403e-13, 3.56316998939048e-13], [2.771742292401327e-7, 1.621038714488984e-14, -1.3953480257523305e-14, 2.317719729370522e-14, 2.656295645154638e-16, 2.187581080387827e-14]),
        ("graze", 12, [2.567042295784907e-7, 8.62150399148976e-14, 1.9701047560252295e-14, 5.660974596404005e-14, -7.668844495549323e-14, 1.5385187972086533e-13], [1.2060124059589271e-7, -1.495695789382154e-13, -7.784389317856392e-14, -6.720764088056411e-14, 1.6931656329828302e-13, -2.8428806094594365e-13]),
    ];

    /// `(name, (1/π)∫|νFᵢ| dL)`: the scale of the rounding of the coefficients.
    #[rustfmt::skip]
    const FOURIER_SCALE: [(&str, [f64; 6]); 2] = [
        ("leo", [8.755e-5, 1.281e-11, 4.175e-12, 1.037e-11, 8.467e-12, 1.610e-11]),
        ("graze", [7.128e-5, 1.054e-11, 4.823e-12, 5.331e-12, 5.153e-12, 8.271e-12]),
    ];

    /// The short-period Fourier coefficients on the pieces and the rule of the mean rates, against
    /// [`FOURIER`], each element weighted as in [`rate_distance`] against the largest of
    /// [`FOURIER_SCALE`]. Measured 3.4e-15 at worst over the twelve harmonics.
    #[test]
    fn the_fourier_coefficients_are_the_conical_average() {
        let model = srp();
        let mut worst = 0.0_f64;
        for (name, [a, k, h, q, p], sun) in FOURIER_ORBITS {
            let aux = DsstAuxiliary::new(a, k, h, q, p, 0.0, MU);
            let Window::Pieces(pieces) = model.window(&aux, sun) else {
                panic!("{name}: the orbit misses the shadow");
            };
            let (nodes, weights) = rule(&pieces);
            let accel = model.acceleration(sun);
            let (c, s) = gaussian_short_period::fourier_coefficients(&nodes, &weights, |l| {
                gaussian::element_derivatives(&aux, l, &accel).0
            });
            let weight = |i: usize| if i == 0 { 1.0 / a } else { 1.0 };
            let Some(&(_, magnitude)) = FOURIER_SCALE.iter().find(|x| x.0 == name) else {
                panic!("{name}: no scale row");
            };
            let scale = (0..6).fold(0.0_f64, |m, i| nan_max(m, weight(i) * magnitude[i]));
            let rows: Vec<_> = FOURIER.iter().filter(|x| x.0 == name).collect();
            assert_eq!(rows.len(), 12, "{name}: twelve harmonics");
            for &&(_, j, want_c, want_s) in &rows {
                for i in 0..6 {
                    let miss = weight(i) * fabs(c[j][i] - want_c[i]).max(fabs(s[j][i] - want_s[i]));
                    worst = nan_max(worst, miss / scale);
                }
            }
        }
        assert!(worst < 2e-14, "Fourier coefficients: {worst:.3e}");
        std::println!("Fourier coefficients against the 80-digit average: worst {worst:.3e}");
    }
}
