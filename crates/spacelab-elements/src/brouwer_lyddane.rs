//! Brouwer–Lyddane analytical propagator: mean-element theory for the zonal geopotential J2–J5.
//!
//! Brouwer's 1959 artificial-satellite theory in Lyddane's 1963 singularity-free variables, with
//! Warren Phipps' 1992 `T2` approximation removing the critical-inclination (`i ≈ 63.4°`) singularity.
//! A faithful port of Orekit's `BrouwerLyddanePropagator.BLModel` (secular rates + long- and
//! short-period corrections), validated against Orekit's own propagator on two orbits —
//! `tools/orekit-oracle/BlOracle.java` emits both, along with the mean elements Orekit's own
//! `computeMeanOrbit` converges to.
//!
//! # Model
//!
//! The orbit is carried as *mean* elements whose only motion is a secular drift of the mean anomaly,
//! argument of perigee and node under J2/J2²/J4. Converting to *osculating* elements at a target time
//! adds the analytic long- and short-period corrections. The osculating→mean conversion at
//! construction is the fixed point of that map, iterated to convergence.
//!
//! The reported osculating Cartesian state is the standard Keplerian conversion of the osculating
//! elements — the same state Orekit's propagator returns (its `KeplerianOrbit` carries element rates
//! for the state-transition matrix, but the velocity `getPVCoordinates` reports is the Keplerian
//! velocity of the osculating ellipse). This port matches Orekit to 4.1×10⁻⁶ m in position and
//! 4.2×10⁻⁹ m/s in velocity across a 24 h propagation of a near-circular LEO, and to 2.2×10⁻⁴ m
//! and 3.4×10⁻⁷ m/s across eight weeks at `e = 0.3`.
//!
//! Construction is fallible. The fixed point does not converge for a near-equatorial, near-circular
//! orbit — Orekit refuses the same inputs — so it is reported as [`BrouwerLyddaneError`] rather than
//! returned as a set of mean elements the theory never reached.
//!
//! The un-normalized zonal coefficients are `C_{n,0} = −J_n`: `c20 ≈ −1.0826×10⁻³` for Earth.
//!
//! # What checks the theory, as distinct from the port
//!
//! The Orekit comparison says the two implementations agree; it cannot say either is right.
//! `derivations/zonal_secular.mac` closes that gap for the part of the model that dominates every
//! answer it gives. It averages the J2 disturbing function over the mean anomaly — a definite
//! integral the CAS evaluates, not a result quoted from a textbook — applies Lagrange's planetary
//! equations, and subtracts this module's own first-order `dsl`, `dsg` and `dsh` in their own
//! `yp2`/`η`/`cos i` notation. All three residuals are **identically zero**, at symbolic `a`, `e`
//! and `i`.
//!
//! The J2², J4 and J5 terms are second order, need Brouwer's von Zeipel transformation rather than
//! an average, and are **not** derived there. Those stay on the Orekit comparison.

// Dense astrodynamics algebra ported verbatim from Orekit; single-letter names and long expressions
// are the source's, kept for auditability.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::suboptimal_flops,
    clippy::doc_markdown
)]

use core::f64::consts::{PI, TAU};
use core::fmt;

use spacelab_core::{Angle, Vec3};

use crate::KeplerianElements;

/// Phipps 1992 `T2` series parameter, `β = 100·2⁻¹¹` (Orekit `BrouwerLyddanePropagator.BETA`).
const BETA: f64 = 0.048_828_125;
/// Eccentricity cap applied by the model (Orekit `MAX_ECC`).
const MAX_ECC: f64 = 0.999_999;
/// Corrections the osculating→mean fixed point is allowed before it is declared divergent
/// (Orekit `BrouwerLyddanePropagator.MAX_ITERATIONS_DEFAULT`).
pub const MAX_CORRECTIONS: u32 = 200;

#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo {
        lo
    } else if x > hi {
        hi
    } else {
        x
    }
}

/// Reduce an angle to `[−π, π]` (Orekit `MathUtils.normalizeAngle(x, 0)`).
#[inline]
fn normalize_pi(x: f64) -> f64 {
    x - TAU * libm::round(x / TAU)
}

/// True anomaly from mean anomaly, for eccentricity `e` — bracketed Newton on Kepler's equation.
///
/// `E = M + e sin E` puts the root within `e` of `M`, and `f(E) = E − e sin E − M` increases
/// monotonically, so `[M − e, M + e]` brackets it for any `M`. Newton on its own does not stay
/// there: near periapsis at high eccentricity the derivative `1 − e cos E` collapses, a single step
/// throws the iterate several revolutions away, and with no bracket to fall back on the loop returns
/// whatever it happens to hold when the step budget runs out. Measured before this was bracketed:
/// `e = 0.99`, `M = −2.7925` came back **0.194 rad** from the root, silently. Bisecting whenever a
/// Newton step leaves the bracket keeps the quadratic rate where it works and the bracket's
/// guarantee where it does not.
///
/// Returns the anomaly and the number of steps it took. Newton is self-correcting: a different
/// starting guess, a scaled step, or a denominator that is not quite the derivative all still land
/// on the same root, which is the only thing the returned angle records. What separates them is how
/// many steps they need, so the count is returned rather than discarded.
fn true_from_mean(m: f64, e: f64) -> (f64, u32) {
    let (mut lo, mut hi) = (m - e, m + e);
    let mut big_e = m + e * libm::sin(m);
    let mut steps = 0;
    for _ in 0..64 {
        let (s, c) = libm::sincos(big_e);
        let f = big_e - e * s - m;
        if f > 0.0 {
            hi = big_e;
        } else {
            lo = big_e;
        }
        let newton = big_e - f / (1.0 - e * c);
        let next = if newton > lo && newton < hi {
            newton
        } else {
            f64::midpoint(lo, hi)
        };
        let moved = next - big_e;
        big_e = next;
        steps += 1;
        if libm::fabs(moved) < 1e-15 {
            break;
        }
    }
    let (s, c) = libm::sincos(big_e);
    (libm::atan2(libm::sqrt(1.0 - e * e) * s, c - e), steps)
}

/// Phipps' `T2(cos i)` ≈ `(1 − 5 cos² i)⁻¹`, regular at the critical inclination (Eq. 2.47–2.48).
fn t2(cos_i: f64) -> f64 {
    let x = 1.0 - 5.0 * cos_i * cos_i;
    let x2 = x * x;
    let mut sum = 0.0;
    let mut fact = 1.0; // (i+1)!
    for i in 0..=12 {
        fact *= f64::from(i + 1);
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        sum += sign * libm::pow(BETA, f64::from(i)) * libm::pow(x2, f64::from(i)) / fact;
    }
    let mut product = 1.0;
    for i in 0..=10 {
        product *= 1.0 + libm::exp(-libm::pow(2.0, f64::from(i)) * BETA * x2);
    }
    BETA * x * sum * product
}

/// Raw Keplerian elements in radians (`pa` = argument of perigee, `m` = mean anomaly).
#[derive(Debug, Clone, Copy)]
struct Els {
    a: f64,
    e: f64,
    i: f64,
    pa: f64,
    raan: f64,
    m: f64,
}

/// The Brouwer–Lyddane model built from a fixed set of *mean* elements: secular rates and the long-
/// and short-period correction coefficients (all functions of `a, e, i` only, so computed once).
struct BlModel {
    mean: Els,
    mu: f64,
    n0: f64,
    dsl: f64,
    dsg: f64,
    dsh: f64,
    a_rate: f64,
    e_rate: f64,
    m2: f64,
    n: f64,
    n2: f64,
    n3: f64,
    t8: f64,
    yp2: f64,
    ci: f64,
    si: f64,
    one_m_ci2: f64,
    ci2x3m1: f64,
    vle1: f64,
    vle2: f64,
    vle3: f64,
    vli1: f64,
    vli2: f64,
    vli3: f64,
    vll2: f64,
    vlh1i: f64,
    vlh2i: f64,
    vlh3i: f64,
    vls1: f64,
    vls2: f64,
    vls3: f64,
}

impl BlModel {
    fn new(mean: Els, re: f64, mu: f64, c: [f64; 6], m2: f64) -> Self {
        let epp = mean.e;
        let epp2 = epp * epp;
        let n2 = 1.0 - epp2;
        let n = libm::sqrt(n2);
        let n3 = n2 * n;
        let t8 = n + 1.0 / (1.0 + n);
        let app = mean.a;
        let n0 = libm::sqrt(mu / app) / app;
        let q = re / app;

        let mut ql = q * q;
        let mut nl = n2 * n2;
        let yp2 = -0.5 * c[2] * ql / nl;
        let yp22 = yp2 * yp2;
        ql *= q;
        nl *= n2;
        let yp3 = c[3] * ql / nl;
        ql *= q;
        nl *= n2;
        let yp4 = 0.375 * c[4] * ql / nl;
        ql *= q;
        nl *= n2;
        let yp5 = c[5] * ql / nl;

        let (si, ci) = libm::sincos(mean.i);
        let ci2 = ci * ci;
        let one_m_ci2 = 1.0 - ci2;
        let ci2x3m1 = 3.0 * ci2 - 1.0;
        let ci2x5m1 = 5.0 * ci2 - 1.0;

        let dsl = 1.5
            * yp2
            * n
            * (ci2x3m1
                + 0.0625
                    * yp2
                    * (-15.0
                        + n * (16.0 + 25.0 * n)
                        + ci2
                            * (30.0 - n * (96.0 + 90.0 * n)
                                + ci2 * (105.0 + n * (144.0 + 25.0 * n)))))
            + 0.9375 * yp4 * n * epp2 * (3.0 - ci2 * (30.0 - 35.0 * ci2));
        let dsg = 1.5 * yp2 * ci2x5m1
            + 0.09375
                * yp22
                * (-35.0
                    + n * (24.0 + 25.0 * n)
                    + ci2
                        * (90.0 - n * (192.0 + 126.0 * n)
                            + ci2 * (385.0 + n * (360.0 + 45.0 * n))))
            + 0.3125
                * yp4
                * (21.0 - 9.0 * n2 + ci2 * (-270.0 + 126.0 * n2 + ci2 * (385.0 - 189.0 * n2)));
        let dsh = (-3.0 * yp2
            + 0.375 * yp22 * (-5.0 + n * (12.0 + 9.0 * n) - ci2 * (35.0 + n * (36.0 + 5.0 * n)))
            + 1.25 * yp4 * (5.0 - 3.0 * n2) * (3.0 - 7.0 * ci2))
            * ci;

        // Secular rates driven by the empirical along-track term M2 (Phipps 1992, Eqs 2.41 & 2.45).
        let coef = -4.0 / (3.0 * n0 * (1.0 + dsl));
        let a_rate = coef * app;
        let e_rate = coef * epp * n2;

        let t2v = t2(ci);
        let fs12 = yp3 / yp2;
        let fs13 = 10.0 * yp4 / (3.0 * yp2);
        let fs14 = yp5 / yp2;

        let ci2xt2 = ci2 * t2v;
        let ca = 1.0 - ci2 * (11.0 + 40.0 * ci2xt2);
        let cb = 1.0 - ci2 * (3.0 + 8.0 * ci2xt2);
        let cc = 1.0 - ci2 * (9.0 + 24.0 * ci2xt2);
        let cd = 1.0 - ci2 * (5.0 + 16.0 * ci2xt2);
        let ce = 1.0 - ci2 * (33.0 + 200.0 * ci2xt2);
        let cf = 1.0 - ci2 * (9.0 + 40.0 * ci2xt2);

        let p5p = 1.0 + ci2xt2 * (8.0 + 20.0 * ci2xt2);
        let p5p2 = 1.0 + 2.0 * p5p;
        let p5p4 = 1.0 + 4.0 * p5p;
        let p5p10 = 1.0 + 10.0 * p5p;

        let e2x3p4 = 4.0 + 3.0 * epp2;
        let ci_o1pci = ci / (1.0 + ci);

        let q1 = 0.125 * (yp2 * ca - fs13 * cb);
        let q2 = 0.125 * epp2 * ci * (yp2 * p5p10 - fs13 * p5p2);
        let q5 = 0.25 * (fs12 + 0.3125 * e2x3p4 * fs14 * cc);
        let p2 = 0.46875 * p5p2 * epp * ci * si * e2x3p4 * fs14;
        let p3 = 0.15625 * epp * si * fs14 * cc;
        let kf = 35.0 / 1152.0;
        let p4 = kf * epp * fs14 * cd;
        let p5 = 2.0 * kf * epp * epp2 * ci * si * fs14 * p5p4;

        let vle1 = epp * n2 * q1;
        let vle2 = n2 * si * q5;
        let vle3 = -3.0 * epp * n2 * si * p4;
        let vli1 = -epp * q1 / si;
        let vli2 = -epp * ci * q5;
        let vli3 = -3.0 * epp2 * ci * p4;
        let vll2 = vle2 + 3.0 * epp * n2 * p3;
        let vlh1i = -si * q2;
        let vlh2i = epp * ci * q5 + si * p2;
        let vlh3i = -epp2 * ci * p4 - si * p5;
        let vls1 = (n3 - 1.0) * q1 - q2 + 25.0 * epp2 * ci2 * ci2xt2 * ci2xt2 * (yp2 - 0.2 * fs13)
            - 0.0625 * epp2 * (yp2 * ce - fs13 * cf);
        let vls2 =
            epp * si * (t8 + ci_o1pci) * q5 + (11.0 + 3.0 * (epp2 - n3)) * p3 + (1.0 - ci) * p2;
        let vls3 = si * p4 * (3.0 * (n3 - 1.0) - epp2 * (2.0 + ci_o1pci)) - (1.0 - ci) * p5;

        Self {
            mean,
            mu,
            n0,
            dsl,
            dsg,
            dsh,
            a_rate,
            e_rate,
            m2,
            n,
            n2,
            n3,
            t8,
            yp2,
            ci,
            si,
            one_m_ci2,
            ci2x3m1,
            vle1,
            vle2,
            vle3,
            vli1,
            vli2,
            vli3,
            vll2,
            vlh1i,
            vlh2i,
            vlh3i,
            vls1,
            vls2,
            vls3,
        }
    }

    /// Osculating elements at `dt` seconds from the mean epoch (Orekit `propagateParameters`).
    fn osculating(&self, dt: f64) -> Els {
        let not = dt * self.n0;
        // Empirical along-track drag term M2 (Phipps 1992): drifts a, e and the mean anomaly.
        let dt_m2 = dt * self.m2;
        let dt2_m2 = dt * dt_m2;
        let app = self.mean.a + dt_m2 * self.a_rate;
        let epp = clamp(self.mean.e + dt_m2 * self.e_rate, 0.0, MAX_ECC);
        let gpp = normalize_pi(self.mean.pa + self.dsg * not);
        let hpp = normalize_pi(self.mean.raan + self.dsh * not);
        let lpp = normalize_pi(self.mean.m + (1.0 + self.dsl) * not + dt2_m2);

        // Long-period corrections.
        let (sgpp, cgpp) = libm::sincos(gpp);
        let (s2gpp, c2gpp) = libm::sincos(2.0 * gpp);
        let (s3gpp, c3gpp) = libm::sincos(3.0 * gpp);
        let d1e = self.vle1 * c2gpp + self.vle2 * sgpp + self.vle3 * s3gpp;
        let mut d1i = self.vli2 * sgpp + self.vli3 * s3gpp;
        if self.vli1.is_finite() {
            d1i += self.vli1 * c2gpp; // pseudo-singular term, skipped when sin i = 0
        }
        let eppd1l = self.n * (self.vle1 * s2gpp - self.vll2 * cgpp - self.vle3 * c3gpp);
        let sippd1h = self.vlh1i * s2gpp + self.vlh2i * cgpp + self.vlh3i * c3gpp;
        let d1z = self.vls1 * s2gpp + self.vls2 * cgpp + self.vls3 * c3gpp;

        // Short-period corrections.
        let (fpp, _) = true_from_mean(lpp, epp);
        let (sfpp, cfpp) = libm::sincos(fpp);
        let eppcfpp = epp * cfpp;
        let eppcfpp_p1 = eppcfpp + 1.0;
        let eppcfpp_p3 = eppcfpp + 3.0;
        let eppcfpp_p1_3 = eppcfpp_p1 * eppcfpp_p1 * eppcfpp_p1;
        let g2 = 2.0 * gpp;
        let (s2gf, c2gf) = libm::sincos(g2 + fpp);
        let eppc2gf = epp * c2gf;
        let epps2gf = epp * s2gf;
        let (s2g2f, c2g2f) = libm::sincos(g2 + 2.0 * fpp);
        let (s2g3f, c2g3f) = libm::sincos(g2 + 3.0 * fpp);
        let eppc2g3f = epp * c2g3f;
        let epps2g3f = epp * s2g3f;
        let w17 = fpp + epp * sfpp - lpp;
        let w20 = cfpp * (eppcfpp_p3 * eppcfpp + 3.0);
        let w21 = 3.0 * (s2g2f + epps2gf) + epps2g3f;
        let w22 = eppcfpp_p1 * (eppcfpp + 2.0) / self.n2;
        let (sio2, cio2) = libm::sincos(0.5 * self.mean.i);

        let yp2 = self.yp2;
        let n2 = self.n2;
        let n3 = self.n3;
        let t8 = self.t8;
        let ci = self.ci;
        let si = self.si;
        let ci2x3m1 = self.ci2x3m1;
        let one_m_ci2 = self.one_m_ci2;

        let d2a = app
            * (yp2 / n2)
            * ((eppcfpp_p1_3 - n3) * ci2x3m1 + eppcfpp_p1_3 * c2g2f * 3.0 * one_m_ci2);
        let d2e = 0.5
            * yp2
            * ((w20 + epp * t8) * ci2x3m1 + (w20 + epp * c2g2f) * 3.0 * one_m_ci2
                - (3.0 * eppc2gf + eppc2g3f) * n2 * one_m_ci2);
        let d2i = 0.5 * yp2 * ci * si * (3.0 * (c2g2f + eppc2gf) + eppc2g3f);
        let eppd2l = -0.25
            * yp2
            * n3
            * ((w22 + 1.0) * sfpp * 2.0 * one_m_ci2
                + ((1.0 - w22) * s2gf + (w22 + 1.0 / 3.0) * s2g3f) * 3.0 * one_m_ci2);
        let sippd2h = 0.5 * yp2 * ci * si * (w21 - 6.0 * w17);
        let d2z = -(epp * eppd2l * (t8 - 1.0) / n3
            + 0.25
                * yp2
                * (w17 * 6.0 * (1.0 + ci * (2.0 - 5.0 * ci))
                    - w21 * (3.0 + ci * (2.0 - 5.0 * ci))));

        // Assemble osculating elements (Lyddane variables).
        let de = epp + d1e + d2e;
        let dl = eppd1l + eppd2l;
        let dh = (sippd1h + sippd2h) / (2.0 * cio2);
        let di = f64::midpoint(d1i, d2i) * cio2 + sio2;
        let z = lpp + gpp + hpp + d1z + d2z;

        let a = app + d2a;
        let e = libm::sqrt(de * de + dl * dl);
        let (slpp, clpp) = libm::sincos(lpp);
        let l = libm::atan2(de * slpp + dl * clpp, de * clpp - dl * slpp);
        let i = libm::acos(clamp(1.0 - 2.0 * (di * di + dh * dh), -1.0, 1.0));
        let (shpp, chpp) = libm::sincos(hpp);
        let h = libm::atan2(di * shpp + dh * chpp, di * chpp - dh * shpp);
        let g = z - l - h;

        Els {
            a,
            e,
            i,
            pa: g,
            raan: h,
            m: l,
        }
    }
}

/// Equinoctial elements `[a, eₓ, e_y, hₓ, h_y, λ]` — the non-singular set the fixed point iterates
/// in, with `eₓ + i e_y = e e^{i(ω+Ω)}`, `hₓ + i h_y = tan(i/2) e^{iΩ}` and `λ = M + ω + Ω`.
fn to_equinoctial(e: Els) -> [f64; 6] {
    let (sp, cp) = libm::sincos(e.pa + e.raan);
    let (sr, cr) = libm::sincos(e.raan);
    let t = libm::tan(0.5 * e.i);
    [e.a, e.e * cp, e.e * sp, t * cr, t * sr, e.m + e.pa + e.raan]
}

/// Inverse of [`to_equinoctial`]. The eccentricity comes back as a hypotenuse and the inclination as
/// `2 atan` of one, so both are non-negative by construction.
fn from_equinoctial(q: [f64; 6]) -> Els {
    let raan = libm::atan2(q[4], q[3]);
    let pa = libm::atan2(q[2], q[1]) - raan;
    Els {
        a: q[0],
        e: libm::hypot(q[1], q[2]),
        i: 2.0 * libm::atan(libm::hypot(q[3], q[4])),
        pa,
        raan,
        m: q[5] - pa - raan,
    }
}

/// Osculating→mean conversion: the fixed point of the mean→osculating map, iterated undamped.
///
/// The correction is applied in equinoctial elements, not Keplerian ones, because `(e, ω)` is polar
/// and degenerates at `e = 0`: the short-period amplitude in `e` is `O(J₂) ≈ 10⁻³`, so for a
/// near-circular orbit the correction is larger than the element it corrects, and since the model
/// returns `e` as a magnitude the iteration cannot converge onto it. Measured on the Keplerian form
/// at `a = 7200 km, i = 28°`: `e = 0` and `e = 10⁻⁴` both ran to the iteration cap and came back with
/// `e ≈ −0.18`, silently, for every inclination tried. In `(eₓ, e_y)` the same correction is a small
/// Cartesian displacement and the map contracts; Orekit's `FixedPointConverter` does the same.
///
/// Returns the converged elements and the number of corrections applied. An undamped fixed-point
/// iteration reaches its fixed point from wherever it is pushed, so the converged elements cannot
/// tell a sound correction from a scaled one, nor a working exit test from one that never fires:
/// both land on the same point, one of them later. The count is the only place that difference is
/// recorded, so it is returned rather than discarded.
fn osculating_to_mean(
    osc: Els,
    re: f64,
    mu: f64,
    c: [f64; 6],
) -> Result<(Els, u32), BrouwerLyddaneError> {
    let target = to_equinoctial(osc);
    let mut q = target;
    let mut steps = 0;
    for _ in 0..MAX_CORRECTIONS {
        // M2 is irrelevant here: osculating(0) has dt = 0, so the drag terms vanish.
        let model = BlModel::new(from_equinoctial(q), re, mu, c, 0.0);
        let pred = to_equinoctial(model.osculating(0.0));
        let da = target[0] - pred[0];
        let dex = target[1] - pred[1];
        let dey = target[2] - pred[2];
        let dhx = target[3] - pred[3];
        let dhy = target[4] - pred[4];
        let dl = normalize_pi(target[5] - pred[5]);
        q[0] += da;
        q[1] += dex;
        q[2] += dey;
        q[3] += dhx;
        q[4] += dhy;
        q[5] += dl;
        steps += 1;
        if libm::fabs(da) / osc.a < 1e-13
            && libm::hypot(dex, dey) < 1e-13
            && libm::hypot(dhx, dhy) < 1e-13
            && libm::fabs(dl) < 1e-13
        {
            return Ok((from_equinoctial(q), steps));
        }
    }
    Err(BrouwerLyddaneError::MeanElementsDiverged {
        corrections: steps,
        inclination_deg: osc.i * 180.0 / PI,
        eccentricity: osc.e,
    })
}

/// The osculating→mean fixed point does not converge everywhere the theory is otherwise defined.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum BrouwerLyddaneError {
    /// The fixed point did not converge within [`MAX_CORRECTIONS`]. Measured: any orbit with
    /// `i ≲ 2°` and `e ≲ 10⁻³`, where the short-period corrections in the node and in the
    /// eccentricity vector are both larger than the elements they correct. Orekit's
    /// `BrouwerLyddanePropagator` refuses the same inputs.
    MeanElementsDiverged {
        /// Corrections applied before giving up.
        corrections: u32,
        /// Osculating inclination, degrees.
        inclination_deg: f64,
        /// Osculating eccentricity.
        eccentricity: f64,
    },
}

impl fmt::Display for BrouwerLyddaneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::MeanElementsDiverged {
            corrections,
            inclination_deg,
            eccentricity,
        } = self;
        write!(
            f,
            "Brouwer-Lyddane mean elements did not converge in {corrections} corrections \
             (osculating i = {inclination_deg:.6}°, e = {eccentricity:.9})"
        )
    }
}

impl core::error::Error for BrouwerLyddaneError {}

/// A Brouwer–Lyddane analytical propagator, built from an osculating state under the zonal field
/// J2–J5. Cheap to evaluate at any epoch — a mean-element drift plus closed-form periodic terms.
pub struct BrouwerLyddane {
    model: BlModel,
    re: f64,
    c: [f64; 6],
}

impl BrouwerLyddane {
    /// Build from initial *osculating* elements and the un-normalized zonal coefficients
    /// `c20..c50 = −J2..−J5`, with the central-body reference radius `re` (m).
    ///
    /// `m2` is the empirical along-track secular coefficient (rad/s², like a TLE `B*`): with `m2 = 0`
    /// the theory has a known along-track drift (~tens of km/day for a LEO — the orbit *plane* is
    /// exact but the in-track phase leads/lags), which is corrected by fitting `m2` in orbit
    /// determination. Use [`BrouwerLyddane::from_osculating`] for the pure `m2 = 0` case.
    ///
    /// # Errors
    ///
    /// Returns [`BrouwerLyddaneError::MeanElementsDiverged`] when the osculating→mean fixed point
    /// does not converge — a near-equatorial, near-circular orbit.
    pub fn from_osculating_with_drag(
        osculating: &KeplerianElements,
        re: f64,
        c20: f64,
        c30: f64,
        c40: f64,
        c50: f64,
        m2: f64,
    ) -> Result<Self, BrouwerLyddaneError> {
        let c = [0.0, 0.0, c20, c30, c40, c50];
        let osc = Els {
            a: osculating.semi_major_axis,
            e: osculating.eccentricity,
            i: osculating.inclination.radians(),
            pa: osculating.arg_periapsis.radians(),
            raan: osculating.raan.radians(),
            m: osculating.mean_anomaly().radians(),
        };
        let (mean, _) = osculating_to_mean(osc, re, osculating.mu, c)?;
        Ok(Self {
            model: BlModel::new(mean, re, osculating.mu, c, m2),
            re,
            c,
        })
    }

    /// Build from initial *osculating* elements with no empirical drag (`m2 = 0`); see
    /// [`BrouwerLyddane::from_osculating_with_drag`].
    ///
    /// # Errors
    ///
    /// As [`BrouwerLyddane::from_osculating_with_drag`].
    pub fn from_osculating(
        osculating: &KeplerianElements,
        re: f64,
        c20: f64,
        c30: f64,
        c40: f64,
        c50: f64,
    ) -> Result<Self, BrouwerLyddaneError> {
        Self::from_osculating_with_drag(osculating, re, c20, c30, c40, c50, 0.0)
    }

    /// The converged mean elements (secular set the theory carries).
    #[must_use]
    pub fn mean_elements(&self) -> KeplerianElements {
        els_to_keplerian(self.model.mean, self.model.mu)
    }

    /// Osculating elements at `dt` seconds from the initial epoch.
    #[must_use]
    pub fn osculating_at(&self, dt: f64) -> KeplerianElements {
        els_to_keplerian(self.model.osculating(dt), self.model.mu)
    }

    /// Osculating Cartesian state `(r, v)` at `dt` seconds — the standard Keplerian conversion of the
    /// osculating elements, which is exactly the state Orekit's propagator reports (matched to
    /// ~10⁻⁹ m/s in velocity, ~10⁻⁶ m in position).
    #[must_use]
    pub fn propagate(&self, dt: f64) -> (Vec3, Vec3) {
        els_to_keplerian(self.model.osculating(dt), self.model.mu).to_state()
    }

    /// The reference radius the model was built with (m).
    #[must_use]
    pub fn reference_radius(&self) -> f64 {
        self.re
    }

    /// The zonal coefficient `c20 = −J2` the model was built with.
    #[must_use]
    pub fn c20(&self) -> f64 {
        self.c[2]
    }
}

/// Convert raw elements (mean anomaly) into [`KeplerianElements`] (true anomaly).
fn els_to_keplerian(e: Els, mu: f64) -> KeplerianElements {
    KeplerianElements {
        semi_major_axis: e.a,
        eccentricity: e.e,
        inclination: Angle::from_radians(e.i),
        raan: Angle::from_radians(e.raan),
        arg_periapsis: Angle::from_radians(e.pa),
        true_anomaly: Angle::from_radians(0.0),
        mu,
    }
    .at_mean_anomaly(Angle::from_radians(e.m))
}

#[cfg(test)]
mod tests {
    // `clamp` is asserted with `==` on purpose: clipping is exact or it is not clipping.
    #![allow(clippy::excessive_precision, clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    const MU: f64 = 3.986_004_415e14;
    const RE: f64 = 6_378_136.3;
    const C20: f64 = -1.082_626_68e-3;
    const C30: f64 = 2.532_656_49e-6;
    const C40: f64 = 1.619_621_59e-6;
    const C50: f64 = 2.272_960_83e-7;

    const R0: [f64; 3] = [
        4_485_195.611_065_954,
        3_180_300.412_195_459,
        4_537_882.996_315_523,
    ];
    const V0: [f64; 3] = [
        -3_045.222_561_050_032,
        -3_821.479_786_492_346,
        5_708.385_836_216_403,
    ];

    // Orekit BrouwerLyddanePropagator, osculating (r, v) in GCRF at t = 0, 10800, …, 86400 s.
    const GOLDEN: [(f64, [f64; 6]); 9] = [
        (
            0.0,
            [
                4_485_195.611_065_566,
                3_180_300.412_194_986,
                4_537_882.996_316_178,
                -3_045.222_561_050_350,
                -3_821.479_786_492_581,
                5_708.385_836_216_108,
            ],
        ),
        (
            10_800.0,
            [
                3_480_185.561_746_166,
                4_053_041.088_610_102,
                -4_780_737.551_198_219,
                4_217.953_936_935_101,
                2_770.003_906_579_405,
                5_511.009_852_106_562,
            ],
        ),
        (
            21_600.0,
            [
                -3_477_944.813_004_522,
                -2_012_883.216_603_487,
                -6_023_622.765_082_777,
                4_240.014_232_282_869,
                4_597.619_677_204_930,
                -3_947.135_262_308_269,
            ],
        ),
        (
            32_400.0,
            [
                -4_668_164.435_740_918,
                -4_782_070.365_904_355,
                2_773_089.613_722_521,
                -2_790.921_914_449_317,
                -1_261.919_659_802_093,
                -6_743.518_250_279_493,
            ],
        ),
        (
            43_200.0,
            [
                1_904_370.585_736_934,
                431_496.024_805_355,
                6_876_798.067_139_263,
                -5_185.625_682_977_900,
                -5_089.847_811_121_574,
                1_812.486_980_311_353,
            ],
        ),
        (
            54_000.0,
            [
                5_158_082.364_881_260,
                4_883_698.623_881_557,
                -761_564.113_002_942,
                1_273.505_093_586_494,
                -239.627_136_594_450,
                7_389.679_472_689_270,
            ],
        ),
        (
            64_800.0,
            [
                -461_921.221_913_646,
                939_843.735_989_677,
                -7_138_813.413_172_542,
                5_481.706_480_260_566,
                4_986.959_351_483_390,
                358.760_348_626_752,
            ],
        ),
        (
            75_600.0,
            [
                -5_349_597.140_119_108,
                -4_658_225.850_972_317,
                -1_539_586.006_637_013,
                469.555_621_499_080,
                1_802.535_316_674_720,
                -7_144.766_395_049_131,
            ],
        ),
        (
            86_400.0,
            [
                -1_341_011.515_050_056,
                -2_514_216.816_814_949,
                6_601_023.662_441_909,
                -5_332.828_878_487_882,
                -4_438.712_829_323_134,
                -2_704.780_309_897_100,
            ],
        ),
    ];

    #[test]
    fn matches_orekit_reference() {
        let osc = KeplerianElements::from_state(
            Vec3::new(R0[0], R0[1], R0[2]),
            Vec3::new(V0[0], V0[1], V0[2]),
            MU,
        );
        let bl = BrouwerLyddane::from_osculating(&osc, RE, C20, C30, C40, C50).unwrap();

        let mut max_dr = 0.0f64;
        let mut max_dv = 0.0f64;
        for (t, g) in GOLDEN {
            let (r, v) = bl.propagate(t);
            let dr = libm::sqrt((r.x - g[0]).powi(2) + (r.y - g[1]).powi(2) + (r.z - g[2]).powi(2));
            let dv = libm::sqrt((v.x - g[3]).powi(2) + (v.y - g[4]).powi(2) + (v.z - g[5]).powi(2));
            assert!(dr < 6e-6, "t={t}: position error {dr:.3e} m");
            assert!(dv < 6e-9, "t={t}: velocity error {dv:.3e} m/s");
            max_dr = max_dr.max(dr);
            max_dv = max_dv.max(dv);
        }
        std::eprintln!(
            "Brouwer-Lyddane vs Orekit: max |Δr| = {max_dr:.3e} m, |Δv| = {max_dv:.3e} m/s"
        );
    }

    // Orekit BrouwerLyddanePropagator with M2 = 1e-14 rad/s² (orekit_brouwer_lyddane_m2.csv).
    #[rustfmt::skip]
    const GOLDEN_M2: [(f64, [f64; 6]); 9] = [
        (0.0, [4485195.611065566, 3180300.412194986, 4537882.996316178, -3045.222561050350, -3821.479786492581, 5708.385836216108]),
        (10800.0, [3480189.848672734, 4053043.657975114, -4780730.660046607, 4217.949979496102, 2769.999148942743, 5511.016081256761]),
        (21600.0, [-3477924.679840436, -2012861.880208605, -6023638.878091190, 4240.031293179847, 4597.629855737079, -3947.107282110258]),
        (32400.0, [-4668190.904304733, -4782081.233062908, 2773019.958001296, -2790.872519744493, -1261.868736003863, -6743.549332662690]),
        (43200.0, [1904276.310435564, 431404.260513090, 6876826.939766615, -5185.664344202376, -5089.857576124217, 1812.352469036013]),
        (54000.0, [5158114.868391702, 4883688.589632006, -761355.226641077, 1273.346165561900, -239.778073098427, 7389.705394248614]),
        (64800.0, [-461699.439071649, 940044.392439776, -7138793.114165714, 5481.728437535921, 4986.921047043666, 359.066780507761]),
        (75600.0, [-5349565.718227993, -4658121.450710819, -1539978.871138249, 469.864433063093, 1802.804958633172, -7144.681282004450]),
        (86400.0, [-1341393.148104778, -2514532.844971872, 6600821.920710811, -5332.728721797692, -4438.521729085552, -2705.291258844486]),
    ];

    /// The empirical along-track drag term M2 matches Orekit's `BrouwerLyddanePropagator` with
    /// M2 = 1e-14 rad/s² to round-off — validating the M2 code path (which is inert at M2 = 0).
    #[test]
    fn matches_orekit_with_drag() {
        let osc = KeplerianElements::from_state(
            Vec3::new(R0[0], R0[1], R0[2]),
            Vec3::new(V0[0], V0[1], V0[2]),
            MU,
        );
        let bl =
            BrouwerLyddane::from_osculating_with_drag(&osc, RE, C20, C30, C40, C50, 1e-14).unwrap();
        let mut max_dr = 0.0f64;
        for (t, g) in GOLDEN_M2 {
            let (r, v) = bl.propagate(t);
            let dr = libm::sqrt((r.x - g[0]).powi(2) + (r.y - g[1]).powi(2) + (r.z - g[2]).powi(2));
            let dv = libm::sqrt((v.x - g[3]).powi(2) + (v.y - g[4]).powi(2) + (v.z - g[5]).powi(2));
            assert!(dr < 6e-6, "t={t}: position error {dr:.3e} m");
            assert!(dv < 6e-9, "t={t}: velocity error {dv:.3e} m/s");
            max_dr = max_dr.max(dr);
        }
        std::eprintln!("Brouwer-Lyddane M2=1e-14 vs Orekit: max |Δr| = {max_dr:.3e} m");
    }
    // Orekit `BrouwerLyddanePropagator.computeMeanOrbit` (epsilon 1e-13, 200 iterations) on the same
    // osculating states, as `a, e, i, ω, Ω, M` — see `tools/orekit-oracle/BlOracle.java`.
    const MEAN_LEO: [f64; 6] = [
        7.198_490_295_258_542_5e6,
        8.273_013_730_741_43e-3,
        1.710_435_650_741_820_8,
        3.522_339_037_227_751_3e-1,
        7.331_282_838_534_002e-1,
        3.415_168_215_717_842_6e-1,
    ];
    const MEAN_ECC: [f64; 6] = [
        7.191_240_497_515_992e6,
        2.976_908_861_906_024e-1,
        5.234_995_751_632_592e-1,
        4.348_034_769_312_009_7e-1,
        6.967_987_201_897_041e-1,
        9.185_099_211_467_751e-2,
    ];

    /// The long-period corrections are driven by the argument of perigee and carry powers of `e`, so
    /// on the near-circular reference orbit their third harmonic moves the state by less than a
    /// micrometre over a day — under the port's own agreement floor with Orekit, where no assertion
    /// can reach it. At `e = 0.3` over eight weeks the same terms are worth hundreds of metres.
    fn eccentric() -> KeplerianElements {
        KeplerianElements {
            semi_major_axis: 7_200_000.0,
            eccentricity: 0.3,
            inclination: Angle::from_degrees(30.0),
            raan: Angle::from_degrees(40.0),
            arg_periapsis: Angle::from_degrees(25.0),
            true_anomaly: Angle::from_degrees(10.0),
            mu: MU,
        }
    }

    #[rustfmt::skip]
    const GOLDEN_ECC: [(f64, [f64; 6]); 8] = [
        (0.0, [1558863.015603414700, 4587663.242400180000, 1450497.924849931600, -8921.851548963321, 2134.252558236289, 4254.949053099550]),
        (691200.0, [-1364093.140547424800, 4324833.146036033500, 2204154.750675403500, -9740.164872987878, -1559.430879688616, -2323.752906270762]),
        (1382400.0, [-3180485.397127992000, 3795997.516659331000, -969107.324663845600, -6351.307858654335, -6335.804048542088, -4707.129112420556]),
        (2073600.0, [-4244564.719677700000, 1233913.731405319000, -2418212.257441471300, -3532.529601853416, -9397.979339313633, 1414.272328373298]),
        (2764800.0, [-4966692.641992023000, -780293.686894361400, 427349.645837749240, 1934.240386712639, -8607.972447419128, 4982.635992212286]),
        (3456000.0, [-3366109.829322452700, -2786306.106077112300, 2511278.870998758400, 6507.643325048334, -7758.427136487936, -454.659924164764]),
        (4147200.0, [-1869212.724852824000, -4704249.977180431000, 116796.070059525970, 8235.625675894426, -2953.307911616923, -5056.624527272788]),
        (4838400.0, [411552.473590451200, -4392657.814281062000, -2502973.278218806700, 9898.477168024183, 1867.588160342446, -518.332110160760]),
    ];

    #[test]
    fn matches_orekit_on_an_eccentric_orbit_over_eight_weeks() {
        let bl = BrouwerLyddane::from_osculating(&eccentric(), RE, C20, C30, C40, C50).unwrap();
        let (mut max_dr, mut max_dv) = (0.0f64, 0.0f64);
        for (t, g) in GOLDEN_ECC {
            let (r, v) = bl.propagate(t);
            let dr = libm::sqrt((r.x - g[0]).powi(2) + (r.y - g[1]).powi(2) + (r.z - g[2]).powi(2));
            let dv = libm::sqrt((v.x - g[3]).powi(2) + (v.y - g[4]).powi(2) + (v.z - g[5]).powi(2));
            assert!(dr < 3e-4, "t={t}: position error {dr:.3e} m");
            assert!(dv < 5e-7, "t={t}: velocity error {dv:.3e} m/s");
            max_dr = max_dr.max(dr);
            max_dv = max_dv.max(dv);
        }
        std::eprintln!(
            "Brouwer-Lyddane e=0.3 vs Orekit: max |Δr| = {max_dr:.3e} m, |Δv| = {max_dv:.3e} m/s"
        );
    }

    /// Every other test here reads the mean elements only through the osculating state, and that
    /// state is invariant under `(i, Ω, ω) → (−i, Ω ± π, ω ± π)` — the sign gauge of a Keplerian
    /// element set. The propagator's entire internal state can therefore be reflected without moving
    /// a single reported position. Orekit's own `computeMeanOrbit` fixes both the gauge and the
    /// values; the angles are compared modulo 2π, which the reflection does not survive.
    #[test]
    fn the_converged_mean_elements_match_orekit() {
        let leo = KeplerianElements::from_state(
            Vec3::new(R0[0], R0[1], R0[2]),
            Vec3::new(V0[0], V0[1], V0[2]),
            MU,
        );
        for (osc, want, what) in [(leo, MEAN_LEO, "leo"), (eccentric(), MEAN_ECC, "eccentric")] {
            let mean = BrouwerLyddane::from_osculating(&osc, RE, C20, C30, C40, C50)
                .unwrap()
                .mean_elements();
            let got = [
                mean.semi_major_axis,
                mean.eccentricity,
                mean.inclination.radians(),
                mean.arg_periapsis.radians(),
                mean.raan.radians(),
                mean.mean_anomaly().radians(),
            ];
            assert!(
                libm::fabs(got[0] - want[0]) / want[0] < 1e-14,
                "{what}: mean a {:.17e} vs {:.17e}",
                got[0],
                want[0]
            );
            assert!(
                libm::fabs(got[1] - want[1]) < 1e-11,
                "{what}: mean e {:.17e} vs {:.17e}",
                got[1],
                want[1]
            );
            for (k, name) in [(2, "i"), (3, "pa"), (4, "raan"), (5, "m")] {
                let d = libm::fabs(normalize_pi(got[k] - want[k]));
                assert!(d < 1e-11, "{what}: mean {name} off by {d:.3e} rad");
            }
        }
    }

    /// The elements the fixed point converges to are the same from any intermediate state, so they
    /// record nothing about the corrections that produced them: a correction that is negated or
    /// rescaled, or an exit test that never fires, lands on the same point a few passes later. The
    /// count is where that shows. Both ends are asserted, because a budget with only a ceiling is
    /// also satisfied by a loop that stops immediately.
    #[test]
    fn the_fixed_point_converges_within_a_measured_number_of_corrections() {
        let c = [0.0, 0.0, C20, C30, C40, C50];
        let (mut least, mut worst, mut total, mut cases) = (u32::MAX, 0, 0u32, 0u32);
        for &e in &[0.0, 1e-4, 0.01, 0.1, 0.3, 0.6] {
            for &i_deg in &[28.0, 63.4, 88.0, 98.0, 145.0] {
                let osc = Els {
                    a: 7_200_000.0,
                    e,
                    i: i_deg * PI / 180.0,
                    pa: 0.4,
                    raan: 0.7,
                    m: 1.1,
                };
                let (_, steps) = osculating_to_mean(osc, RE, MU, c).unwrap();
                least = least.min(steps);
                worst = worst.max(steps);
                total += steps;
                cases += 1;
            }
        }
        let mean = f64::from(total) / f64::from(cases);
        std::eprintln!(
            "Brouwer-Lyddane osculating→mean: {least}..{worst} corrections, mean {mean:.2}"
        );
        assert!(
            least >= 5,
            "the fixed point exited after {least} corrections"
        );
        assert!(worst <= 96, "the fixed point took {worst} corrections");
        assert!(
            (29.5..30.5).contains(&mean),
            "mean {mean:.2} corrections per case"
        );
    }

    /// The theory has a regime where the fixed point simply does not converge, and returning the
    /// last iterate there is how a caller ends up propagating a negative eccentricity. Orekit's
    /// `BrouwerLyddanePropagator` refuses the same inputs.
    #[test]
    fn a_near_equatorial_near_circular_orbit_is_refused() {
        let c = [0.0, 0.0, C20, C30, C40, C50];
        for &(e, i_deg) in &[
            (0.0, 0.5),
            (1e-6, 0.5),
            (1e-4, 0.5),
            (1e-3, 0.5),
            (0.0, 1.0),
        ] {
            let osc = Els {
                a: 7_200_000.0,
                e,
                i: i_deg * PI / 180.0,
                pa: 0.4,
                raan: 0.7,
                m: 1.1,
            };
            let err = osculating_to_mean(osc, RE, MU, c).unwrap_err();
            let BrouwerLyddaneError::MeanElementsDiverged {
                corrections,
                inclination_deg,
                eccentricity,
            } = err;
            assert_eq!(corrections, MAX_CORRECTIONS, "e = {e}, i = {i_deg}°");
            assert!(
                libm::fabs(inclination_deg - i_deg) < 1e-12,
                "reported i = {inclination_deg} for {i_deg}°"
            );
            assert_eq!(eccentricity, e, "reported e = {eccentricity} for {e}");
            // The report is what a caller sees; an empty one is the same as no report.
            let text = std::format!("{err}");
            assert!(text.contains(&std::format!("{corrections}")), "{text}");
            assert!(text.contains(&std::format!("{i_deg:.6}")), "{text}");
            assert!(text.contains(&std::format!("{e:.9}")), "{text}");
        }
    }

    /// Newton reaches the root of `E − e sin E = M` from any starting guess and with any step close
    /// enough to the derivative, so the returned anomaly cannot distinguish them. Kepler's equation
    /// pins the root — written here from the definition, not taken from the solver — and the
    /// two-sided budget pins the path to it.
    #[test]
    fn the_kepler_newton_reaches_its_root_within_a_measured_number_of_steps() {
        let (mut least, mut worst, mut total, mut calls) = (u32::MAX, 0, 0u32, 0u32);
        for &e in &[
            0.0, 1e-6, 0.01, 0.3, 0.7, 0.79, 0.8, 0.9, 0.99, 0.999, MAX_ECC,
        ] {
            for k in 0..73 {
                let m = -PI + TAU * f64::from(k) / 72.0;
                let (nu, steps) = true_from_mean(m, e);
                let (snu, cnu) = libm::sincos(nu / 2.0);
                let big_e = 2.0 * libm::atan2(libm::sqrt(1.0 - e) * snu, libm::sqrt(1.0 + e) * cnu);
                // `E` has to be recovered from the reported `ν`, and near apoapsis at high `e` that
                // recovery is ill-conditioned — `dE/dν = (1 − e cos E)/√(1−e²)` — after which the
                // residual picks up a further factor `f′ = 1 − e cos E`. The bound tracks that
                // product, so it states something about the solver rather than about the angle it
                // reports through. The slope is set from a 201×721 sweep of the whole `(e, M)`
                // square, not from this grid: the residual runs at ≈ 2×10⁻¹⁶·amp through
                // `e = 0.999` and reaches 0.82 of the bound at `e = MAX_ECC`, where `amp` is 2828.
                let amp = (1.0 - e * libm::cos(big_e)).powi(2) / libm::sqrt(1.0 - e * e);
                let bound = 1e-15 + 5e-14 * amp;
                let residual = libm::fabs(normalize_pi(big_e - e * libm::sin(big_e) - m));
                assert!(
                    residual < bound,
                    "e={e}, M={m}: Kepler residual {residual:.3e} over {bound:.3e}"
                );
                // Through `e = 0.99` — where the model is actually used, `MAX_ECC` being a clamp
                // rather than a design point — the recovery is well enough conditioned for a flat
                // statement. Worst measured there is 7.33×10⁻¹⁵.
                assert!(
                    e > 0.99 || residual < 1e-14,
                    "e={e}, M={m}: Kepler residual {residual:.3e}"
                );
                least = least.min(steps);
                worst = worst.max(steps);
                total += steps;
                calls += 1;
            }
        }
        let mean = f64::from(total) / f64::from(calls);
        std::eprintln!("Brouwer-Lyddane Kepler Newton: {least}..{worst} steps, mean {mean:.2}");
        assert!(worst <= 56, "the solve took {worst} steps");
        assert!(
            (19.5..20.3).contains(&mean),
            "mean {mean:.2} steps per solve"
        );
    }

    /// `clamp` is the model's only guard against an eccentricity driven out of range by the drag
    /// term and against `acos` of an argument that rounding pushed past 1; nothing downstream
    /// re-checks either. Both ends have to clip, not merely pass the value through.
    #[test]
    fn clamp_clips_at_both_ends() {
        assert_eq!(clamp(-2.0, -1.0, 1.0), -1.0);
        assert_eq!(clamp(2.0, -1.0, 1.0), 1.0);
        assert_eq!(clamp(0.25, -1.0, 1.0), 0.25);
        assert_eq!(clamp(-1.0, -1.0, 1.0), -1.0);
        assert_eq!(clamp(1.0, -1.0, 1.0), 1.0);
    }
}
