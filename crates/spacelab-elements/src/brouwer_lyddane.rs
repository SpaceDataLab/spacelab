//! Brouwer–Lyddane analytical propagator: mean-element theory for the zonal geopotential J2–J5.
//!
//! Brouwer's 1959 artificial-satellite theory in Lyddane's 1963 singularity-free variables, with
//! Warren Phipps' 1992 `T2` approximation removing the critical-inclination (`i ≈ 63.4°`) singularity.
//! The code is a port of Orekit's `BrouwerLyddanePropagator.BLModel`, in its layout and notation
//! (secular rates + long- and short-period corrections). Four of its terms are not Orekit's, and
//! neither is the inclination's assembly: the derivation below decides each, and Orekit's terms
//! break the equations of motion they approximate (see *What checks the theory*).
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
//! velocity of the osculating ellipse). The osculating `a` is not `a'' + δa₁`: it is the one the
//! energy integral fixes, `−μ/(2a) + R = K''`, the J2–J5 potential at the osculating position
//! against the mean Hamiltonian, which carries the second-order generator's J2², J3–J5 and
//! long-period cross terms of `a`. The mean motion the fixed point recovers is then right to
//! `O(J2²)`. Against an integration of the J2–J5 equations of motion the state stays within 66 m
//! over three revolutions of a near-circular LEO and 125 m at `e = 0.3`, the second-order periodic
//! terms of the other elements, and drifts along track by 2 m a revolution at most, a third-order
//! drift; with `a'' + δa₁` the drift was 111 and 527 m a revolution.
//!
//! Construction is fallible. The fixed point does not converge within about 8° of retrograde
//! equatorial, where Lyddane's `sin(i/2)` node variables divide by `cos(i/2)`, so that is reported
//! as [`BrouwerLyddaneError`] rather than returned as a set of mean elements the theory never
//! reached.
//!
//! The un-normalized zonal coefficients are `C_{n,0} = −J_n`: `c20 ≈ −1.0826×10⁻³` for Earth.
//!
//! # What checks the theory
//!
//! An agreement with another implementation cannot say either is right. The derivations decide,
//! in Delaunay variables from the zonal Hamiltonian, each against this module's own expressions
//! read from the source:
//!
//! * `derivations/zonal_secular.mac`: the first-order J2 rates `dsl`, `dsg`, `dsh`, by averaging
//!   the disturbing function and Lagrange's planetary equations. Residuals identically zero.
//! * `derivations/brouwer_lyddane.mac` (1): their J2² part by a Lie–Deprit normalisation and the
//!   J4 part by a first-order average. Residuals identically zero.
//! * (2): the short-period generator solves the homological equation, and the six short-period
//!   terms are its Lie brackets, so Hamilton's equations hold to first order in J2.
//! * (3): the long-period generator divides the `g`-dependent part of the averaged J2², J3, J4
//!   and J5 Hamiltonian by the J2 rate of `g`, and the long-period coefficients are its brackets,
//!   with Phipps' `T2` standing for `1/(1 − 5 cos² i)`.
//! * (4): every long-period `δi` is `−e cos i/(η² sin i)·δe`, the conservation of `G cos i`.
//! * (5): the osculating `L` to second order from the generators — the J2² short-period bracket,
//!   the J3–J5 short-period terms and the long-period cross term — is the energy integral's, the
//!   shipped `kq` is the mean Hamiltonian's J2, J2² and J4 part, and at sample elements the energy
//!   at the first-order osculating position gives the generator's `a` to third order.
//!
//! Orekit's `BLModel` fails (2) in `δe` and `e·δl`, and (4) in two `δi` coefficients. Its
//! propagation then drifts from the equations of motion at first order in J2: 38 km over three
//! LEO revolutions, where this one stays within 25 m under J2 alone. Halving J2 halves Orekit's
//! residual and quarters this one's.

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
    re: f64,
    j2: f64,
    j3: f64,
    j4: f64,
    j5: f64,
    kq: f64,
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
        // The mean Hamiltonian whose rates these are, K'' = −μ(1 − 2 kq)/(2a''): kq is its J2,
        // J2² and J4 part over μ/a'' (derivations/brouwer_lyddane.mac (5)).
        let kq = -0.5 * yp2 * n * ci2x3m1
            - 0.03125
                * yp22
                * n
                * (-15.0
                    + n * (12.0 + 15.0 * n)
                    + ci2
                        * (30.0 - n * (72.0 + 54.0 * n) + ci2 * (105.0 + n * (108.0 + 15.0 * n))))
            - 0.0625 * yp4 * n * (2.0 + 3.0 * epp2) * (3.0 - ci2 * (30.0 - 35.0 * ci2));

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
        // Every long-period δi is −e cos i/(η² sin i)·δe: the generator is free of l and h, so it
        // keeps L and G cos i. Orekit's vli1 = −e q1/sin i and vli3 = −3e² cos i p4 break that
        // (derivations/brouwer_lyddane.mac (4)).
        let vli1 = -epp2 * ci * q1 / si;
        let vli2 = -epp * ci * q5;
        let vli3 = 3.0 * epp2 * ci * p4;
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
            re,
            j2: -c[2],
            j3: -c[3],
            j4: -c[4],
            j5: -c[5],
            kq,
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

    /// Mean elements at `dt` seconds from the mean epoch: Brouwer's secular rates in the angles, and
    /// Phipps' along-track drag term (1992, Eqs 2.38, 2.41, 2.45) in `a`, `e` and the mean anomaly.
    fn secular(&self, dt: f64) -> Els {
        let not = dt * self.n0;
        let dt_m2 = dt * self.m2;
        Els {
            a: self.mean.a + dt_m2 * self.a_rate,
            e: clamp(self.mean.e + dt_m2 * self.e_rate, 0.0, MAX_ECC),
            i: self.mean.i,
            pa: normalize_pi(self.mean.pa + self.dsg * not),
            raan: normalize_pi(self.mean.raan + self.dsh * not),
            m: normalize_pi(self.mean.m + (1.0 + self.dsl) * not + dt * dt_m2),
        }
    }

    /// Osculating elements at `dt` seconds from the mean epoch (Orekit `propagateParameters`).
    fn osculating(&self, dt: f64) -> Els {
        let Els {
            a: app,
            e: epp,
            pa: gpp,
            raan: hpp,
            m: lpp,
            ..
        } = self.secular(dt);

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
        // Brouwer's δe and e·δl, which Lagrange's equations for e and l decide at first order in J2
        // (derivations/brouwer_lyddane.mac (2)). Orekit's BLModel multiplies only the e of
        // (w20 + e) by cos(2g+2f), carries an extra e on cos(2g+f) and cos(2g+3f), and writes
        // sin²i for 3cos²i − 1 on the sin f term of e·δl: 38 km over three LEO revolutions.
        let d2e = 0.5
            * yp2
            * ((w20 + epp * t8) * ci2x3m1 + (w20 + epp) * c2g2f * 3.0 * one_m_ci2
                - (3.0 * c2gf + c2g3f) * n2 * one_m_ci2);
        let d2i = 0.5 * yp2 * ci * si * (3.0 * (c2g2f + eppc2gf) + eppc2g3f);
        let eppd2l = -0.25
            * yp2
            * n3
            * ((w22 + 1.0) * sfpp * 2.0 * ci2x3m1
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

        let e = libm::sqrt(de * de + dl * dl);
        let (slpp, clpp) = libm::sincos(lpp);
        let l = libm::atan2(de * slpp + dl * clpp, de * clpp - dl * slpp);
        // sin(i/2) is the hypotenuse itself; Orekit's acos(1 − 2 sin²(i/2)) loses half the digits
        // of a small i, and at 0.01° and 10⁻⁴° (measured) its fixed point stalls short of its
        // tolerance.
        let i = 2.0 * libm::asin(clamp(libm::sqrt(di * di + dh * dh), 0.0, 1.0));
        let (shpp, chpp) = libm::sincos(hpp);
        let h = libm::atan2(di * shpp + dh * chpp, di * chpp - dh * shpp);
        let g = z - l - h;

        // The osculating a is the one the energy integral fixes, −μ/(2a) + R = K'': the J2–J5
        // potential R at the osculating position, which the first-order elements give to O(J2²),
        // against the mean Hamiltonian. To O(J2²) this is a'' + d2a plus the second-order
        // generator's J2², J3–J5 and long-period cross terms (derivations/brouwer_lyddane.mac (5));
        // a'' + d2a alone leaves O(J2²) in a'' and so in the mean motion, an along-track drift.
        let (fo, _) = true_from_mean(l, e);
        let r = (app + d2a) * (1.0 - e * e) / (1.0 + e * libm::cos(fo));
        let sphi = libm::sin(i) * libm::sin(g + fo);
        let s2 = sphi * sphi;
        let q = self.re / r;
        let pot = (q * q / r)
            * (self.j2 * 0.5 * (3.0 * s2 - 1.0)
                + q * (self.j3 * 0.5 * sphi * (5.0 * s2 - 3.0)
                    + q * (self.j4 * 0.125 * (3.0 + s2 * (35.0 * s2 - 30.0))
                        + q * self.j5 * 0.125 * sphi * (15.0 + s2 * (63.0 * s2 - 70.0)))));
        let a = 1.0 / ((1.0 - 2.0 * self.kq) / app + 2.0 * pot);

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
    /// The fixed point did not converge within [`MAX_CORRECTIONS`]. Measured over `a` from 6 600
    /// to 42 164 km, `e` to 0.8 and four `(ω, M)`: most orbits at `i ≥ 172°` (every one at
    /// `e ≤ 0.01`) and none at `i ≤ 170°`, prograde equatorial included. Near retrograde
    /// equatorial Lyddane's `sin(i/2)` node variables divide by `cos(i/2)`.
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
    /// The term is Phipps' (1992, NPS thesis, Eqs 2.38–2.45): the mean anomaly gains `m2·t²`, and
    /// with `m = n₀(1 + δₛl)` the mean `a` and `e` drift at `ȧ = −4a·m2/(3m)` (Kepler's third law to
    /// first order) and `ė = −4eη²·m2/(3m)` (his `σ = 1`). His cubic term `M3·t³` is not carried.
    /// `derivations/brouwer_m2_phipps.mac` derives the rates.
    ///
    /// # Errors
    ///
    /// Returns [`BrouwerLyddaneError::MeanElementsDiverged`] when the osculating→mean fixed point
    /// does not converge — an orbit near retrograde equatorial.
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
    /// osculating elements, the state Orekit's propagator reports too.
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

    fn leo() -> KeplerianElements {
        KeplerianElements::from_state(
            Vec3::new(R0[0], R0[1], R0[2]),
            Vec3::new(V0[0], V0[1], V0[2]),
            MU,
        )
    }

    /// Position and velocity distances from `bl` to a table of Orekit states, as comparison rows.
    fn state_rows(bl: &BrouwerLyddane, golden: &[(f64, [f64; 6])], what: &str) {
        let (dr, dv): (std::vec::Vec<f64>, std::vec::Vec<f64>) = golden
            .iter()
            .map(|&(t, g)| {
                let (r, v) = bl.propagate(t);
                let dr = (r.x - g[0]).hypot(r.y - g[1]).hypot(r.z - g[2]);
                let dv = (v.x - g[3]).hypot(v.y - g[4]).hypot(v.z - g[5]);
                (dr, dv)
            })
            .unzip();
        comparison_row(&std::format!("Brouwer-Lyddane {what}, position"), "m", &dr);
        comparison_row(
            &std::format!("Brouwer-Lyddane {what}, velocity"),
            "m/s",
            &dv,
        );
    }

    fn comparison_row(quantity: &str, unit: &str, d: &[f64]) {
        let max = d.iter().copied().fold(0.0, f64::max);
        #[allow(clippy::cast_precision_loss)]
        let rms = (d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt();
        std::println!(
            "comparison | Orekit | 13.1.7 | {quantity} | {unit} | {} | {max:.3e} | {rms:.3e}",
            d.len()
        );
    }

    /// The distance to Orekit's `BrouwerLyddanePropagator` on the near-circular LEO over a day.
    /// Orekit's short-period δe and e·δl break Lagrange's equations at first order in J2, so the
    /// two differ by tens of kilometres; [`the_propagation_follows_the_zonal_equations_of_motion`]
    /// and `derivations/brouwer_lyddane.mac` decide.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn reference_distance_to_orekit() {
        let bl = BrouwerLyddane::from_osculating(&leo(), RE, C20, C30, C40, C50).unwrap();
        state_rows(&bl, &GOLDEN, "LEO, 1 day");
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

    /// The distance to Orekit's `BrouwerLyddanePropagator` with M2 = 1e-14 rad/s². The M2 path is
    /// decided by [`the_m2_drift_is_phipps`] and `derivations/brouwer_m2_phipps.mac`.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn drag_distance_to_orekit() {
        let bl = BrouwerLyddane::from_osculating_with_drag(&leo(), RE, C20, C30, C40, C50, 1e-14)
            .unwrap();
        state_rows(&bl, &GOLDEN_M2, "LEO with M2 = 1e-14 rad/s2, 1 day");
    }

    /// The M2 drift is Phipps 1992's, read off the mean elements a day out: the mean anomaly gains
    /// exactly `M2·t²` (Eq. 2.38); `ȧ/a = −(2/3)·ṁ/m` with `ṁ = 2·M2` (Kepler's third law, Eqs 2.40–2.41);
    /// `ė/e = η²·ȧ/a` (Eq. 2.44 with his `σ = 1`, Eq. 2.45); and the angles' Brouwer rates do not see it.
    #[test]
    fn the_m2_drift_is_phipps() {
        let mean = Els {
            a: MEAN_LEO[0],
            e: MEAN_LEO[1],
            i: MEAN_LEO[2],
            pa: MEAN_LEO[3],
            raan: MEAN_LEO[4],
            m: MEAN_LEO[5],
        };
        let c = [0.0, 0.0, C20, C30, C40, C50];
        let m2 = 1e-14;
        let drag = BlModel::new(mean, RE, MU, c, m2);
        let free = BlModel::new(mean, RE, MU, c, 0.0);
        let t = 86_400.0;
        let (with, without) = (drag.secular(t), free.secular(t));
        let m = drag.n0 * (1.0 + drag.dsl);

        let dl = normalize_pi(with.m - without.m);
        assert!(
            (dl - m2 * t * t).abs() < 1e-13,
            "Δl = {dl}, M2 t² = {}",
            m2 * t * t
        );

        let adot_over_a = (with.a - mean.a) / (t * mean.a);
        let kepler = -(2.0 / 3.0) * (2.0 * m2) / m;
        assert!(
            ((adot_over_a - kepler) / kepler).abs() < 1e-9,
            "ȧ/a = {adot_over_a}, Kepler {kepler}"
        );

        let eta2 = 1.0 - mean.e * mean.e;
        let edot_over_e = (with.e - mean.e) / (t * mean.e);
        assert!(
            ((edot_over_e - eta2 * adot_over_a) / (eta2 * adot_over_a)).abs() < 1e-8,
            "ė/e = {edot_over_e}, η² ȧ/a = {}",
            eta2 * adot_over_a
        );

        assert_eq!(
            (with.pa, with.raan, with.i),
            (without.pa, without.raan, without.i)
        );
        assert_eq!((without.a, without.e), (mean.a, mean.e));
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
    /// micrometre over a day. At `e = 0.3` the same terms are worth hundreds of metres.
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

    /// The distance to Orekit's `BrouwerLyddanePropagator` at `e = 0.3` over eight weeks, where
    /// Orekit's long-period δi also departs from the conservation of `G cos i`.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn eccentric_distance_to_orekit() {
        let bl = BrouwerLyddane::from_osculating(&eccentric(), RE, C20, C30, C40, C50).unwrap();
        state_rows(&bl, &GOLDEN_ECC, "e = 0.3, 8 weeks");
    }

    /// The distance to the mean elements Orekit's `computeMeanOrbit` converges to. Orekit's map
    /// differs from Brouwer's, so its fixed point does too;
    /// [`the_mean_elements_are_the_fixed_point_in_the_canonical_gauge`] decides.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn mean_elements_distance_to_orekit() {
        for (osc, want, what) in [(leo(), MEAN_LEO, "LEO"), (eccentric(), MEAN_ECC, "e = 0.3")] {
            let mean = BrouwerLyddane::from_osculating(&osc, RE, C20, C30, C40, C50)
                .unwrap()
                .mean_elements();
            comparison_row(
                &std::format!("Brouwer-Lyddane mean a, {what}"),
                "m",
                &[libm::fabs(mean.semi_major_axis - want[0])],
            );
            comparison_row(
                &std::format!("Brouwer-Lyddane mean e, {what}"),
                "1",
                &[libm::fabs(mean.eccentricity - want[1])],
            );
            let got = [
                mean.inclination.radians(),
                mean.arg_periapsis.radians(),
                mean.raan.radians(),
                mean.mean_anomaly().radians(),
            ];
            let d: std::vec::Vec<f64> = (0..4)
                .map(|k| libm::fabs(normalize_pi(got[k] - want[k + 2])))
                .collect();
            comparison_row(
                &std::format!("Brouwer-Lyddane mean (i, w, raan, M), {what}"),
                "rad",
                &d,
            );
        }
    }

    /// The mean elements are defined as the fixed point of the mean→osculating map: their
    /// osculating image at the epoch is the state they were built from (G2). The state is
    /// invariant under `(i, Ω, ω) → (−i, Ω ± π, ω ± π)`, so the internal elements are also pinned
    /// to the canonical gauge `0 ≤ i ≤ π`, `0 ≤ e < 1`, which that reflection leaves.
    #[test]
    fn the_mean_elements_are_the_fixed_point_in_the_canonical_gauge() {
        let mut worst = 0.0f64;
        for osc in orbits() {
            let bl = BrouwerLyddane::from_osculating(&osc, RE, C20, C30, C40, C50).unwrap();
            let mean = bl.model.mean;
            assert!((0.0..=PI).contains(&mean.i), "mean i = {}", mean.i);
            assert!((0.0..1.0).contains(&mean.e), "mean e = {}", mean.e);
            let (r, v) = bl.propagate(0.0);
            let (r0, v0) = osc.to_state();
            worst = worst.max((r.x - r0.x).hypot(r.y - r0.y).hypot(r.z - r0.z));
            worst = worst.max(((v.x - v0.x).hypot(v.y - v0.y).hypot(v.z - v0.z)) * 1e3);
        }
        std::eprintln!("Brouwer-Lyddane fixed point: {worst:.3e} m (velocity as mm/s)");
        // Measured 4.2e-9.
        assert!(
            worst < 2e-8,
            "the osculating image is {worst:.3e} from the state"
        );
    }

    /// Orbits across the regimes the theory covers: the near-circular sun-synchronous LEO, the
    /// eccentric one, near-circular near-equatorial, and a moderate eccentricity at 63°, beside the
    /// critical inclination where Phipps' `T2` stands for `1/(1 − 5 cos² i)`.
    fn orbits() -> [KeplerianElements; 4] {
        let base = eccentric();
        [
            leo(),
            base,
            KeplerianElements {
                eccentricity: 1e-3,
                inclination: Angle::from_degrees(0.01),
                ..base
            },
            KeplerianElements {
                eccentricity: 0.05,
                inclination: Angle::from_degrees(63.0),
                ..base
            },
        ]
    }

    /// The worst distance from the propagated state to an integration of the same zonal field —
    /// point mass plus `J2..J5` by Legendre recurrence, Dormand–Prince 8(5,3) at 10⁻¹³ — sampled
    /// each minute over three revolutions, m; and the along-track drift, the mean along-track
    /// offset over the third revolution less that over the first, per revolution, m. A mean motion
    /// off by δn drifts by `a δn` times the period each revolution; the periodic offsets average out.
    fn distance_to_the_equations_of_motion(osc: &KeplerianElements, c: [f64; 4]) -> (f64, f64) {
        use spacelab_propagation::driver::{Controls, integrate};
        use spacelab_propagation::forces::{Newtonian, Plain, PointMass, Zonal};
        use spacelab_propagation::{Dop853, Tolerance};
        let bl = BrouwerLyddane::from_osculating(osc, RE, c[0], c[1], c[2], c[3]).unwrap();
        let (r0, v0) = osc.to_state();
        let field = Newtonian((
            PointMass { mu: MU },
            Zonal {
                mu: MU,
                re: RE,
                j: [-c[0], -c[1], -c[2], -c[3]],
            },
        ));
        let period = osc.period();
        let minutes = (3.0 * period / 60.0).floor();
        let traj = integrate(
            &Dop853,
            &Plain(field),
            0.0,
            &[r0.x, r0.y, r0.z, v0.x, v0.y, v0.z],
            60.0 * minutes,
            60.0,
            Controls::new(Tolerance::new(1e-13, 1e-13), 1e-3),
        )
        .unwrap();
        let mut worst = 0.0f64;
        // per revolution: the summed along-track offset and the sample count
        let mut along = [(0.0f64, 0.0f64); 3];
        let mut t = 60.0;
        while t <= 60.0 * minutes {
            let y = traj.eval(t);
            let (r, _) = bl.propagate(t);
            let d = Vec3::new(r.x - y[0], r.y - y[1], r.z - y[2]);
            worst = worst.max(d.norm());
            let rr = Vec3::new(y[0], y[1], y[2]);
            let track = rr.cross(Vec3::new(y[3], y[4], y[5])).cross(rr);
            let rev = if t < period {
                0
            } else if t < 2.0 * period {
                1
            } else {
                2
            };
            along[rev].0 += d.dot(track) / track.norm();
            along[rev].1 += 1.0;
            t += 60.0;
        }
        (
            worst,
            0.5 * (along[2].0 / along[2].1 - along[0].0 / along[0].1),
        )
    }

    /// Brouwer's theory is the solution of the zonal equations of motion to first order in J2 in
    /// its periodic terms, so the propagated state follows a numerical integration of the same
    /// field to `O(J2²)`: tens of metres over three revolutions. (S1)
    #[test]
    fn the_propagation_follows_the_zonal_equations_of_motion() {
        let c = [C20, C30, C40, C50];
        let d: std::vec::Vec<f64> = orbits()
            .iter()
            .map(|osc| distance_to_the_equations_of_motion(osc, c).0)
            .collect();
        std::eprintln!("Brouwer-Lyddane against the J2-J5 equations of motion: {d:?} m");
        // Measured 65.6, 124.5, 92.5, 47.0 m: the second-order periodic terms of e, i, l, g and h
        // the theory leaves out. With a'' + δa₁ for the osculating a, the mean motion drifted at
        // O(J2²) and these were 366, 2230, 388, 458 m; a first-order defect is tens of kilometres.
        for (k, bound) in [250.0, 500.0, 400.0, 200.0].into_iter().enumerate() {
            assert!(
                d[k] < bound,
                "orbit {k}: {:.3e} m from the equations of motion",
                d[k]
            );
        }
    }

    /// The osculating `a` is the one the energy integral fixes, so the mean `a` the fixed point
    /// recovers, and with it the mean motion, is right to `O(J2²)`: the along-track drift is third
    /// order, and halving J2 divides it by eight. An `a` short of its second-order terms drifts at
    /// `O(J2²)` and divides by four. (S1, S2)
    #[test]
    fn the_along_track_drift_is_third_order_in_j2() {
        let c = [C20, C30, C40, C50];
        let drift: std::vec::Vec<f64> = orbits()
            .iter()
            .map(|osc| distance_to_the_equations_of_motion(osc, c).1)
            .collect();
        std::eprintln!("Brouwer-Lyddane along-track drift, J2-J5: {drift:?} m per revolution");
        // Measured -0.044, 2.04, -1.93, 0.55 m; a'' + δa₁ drifted -111, -527, -113, -140 m.
        for (k, bound) in [0.2, 8.0, 8.0, 2.5].into_iter().enumerate() {
            assert!(
                drift[k].abs() < bound,
                "orbit {k}: {:.3e} m per revolution",
                drift[k]
            );
        }
        for osc in [eccentric(), orbits()[2]] {
            let full = distance_to_the_equations_of_motion(&osc, [C20, 0.0, 0.0, 0.0]).1;
            let half = distance_to_the_equations_of_motion(&osc, [0.5 * C20, 0.0, 0.0, 0.0]).1;
            let ratio = full / half;
            std::eprintln!(
                "Brouwer-Lyddane J2-only drift: {full:.3e} m, at J2/2 {half:.3e} m, ratio {ratio:.3}"
            );
            // Measured 7.90 at e = 0.3 and 8.05 near-equatorial; 4.00 and 4.02 with a'' + δa₁.
            assert!(
                (7.5..8.5).contains(&ratio),
                "e = {}: drift ratio {ratio:.3}",
                osc.eccentricity
            );
        }
    }

    /// The residual of [`the_propagation_follows_the_zonal_equations_of_motion`] is second order in
    /// J2: halving J2 quarters it. A periodic term that breaks the equations of motion at first
    /// order — Orekit's δe, e·δl — halves it instead. (S2)
    #[test]
    fn the_residual_is_second_order_in_j2() {
        for osc in [eccentric(), leo()] {
            let full = distance_to_the_equations_of_motion(&osc, [C20, 0.0, 0.0, 0.0]).0;
            let half = distance_to_the_equations_of_motion(&osc, [0.5 * C20, 0.0, 0.0, 0.0]).0;
            let ratio = full / half;
            std::eprintln!(
                "Brouwer-Lyddane J2 only: {full:.3e} m, at J2/2 {half:.3e} m, ratio {ratio:.3}"
            );
            // Measured 4.175 at e = 0.3 and 4.011 on the LEO (3.988 and 4.016 while the drift,
            // also second order, dominated); Orekit's map gives 1.97 at e = 0.3.
            assert!(
                (3.8..4.2).contains(&ratio),
                "e = {}: ratio {ratio:.3}",
                osc.eccentricity
            );
        }
    }

    /// The state's `J2..J5` energy `v²/2 − μ/r + (μ/r) Σ Jn (Re/r)^n Pn(z/r)`, J/kg.
    fn zonal_energy(r: Vec3, v: Vec3, c: [f64; 4]) -> f64 {
        let rn = r.norm();
        let s = r.z / rn;
        let s2 = s * s;
        let q = RE / rn;
        let p = [
            0.5 * (3.0 * s2 - 1.0),
            0.5 * s * (5.0 * s2 - 3.0),
            0.125 * (3.0 + s2 * (35.0 * s2 - 30.0)),
            0.125 * s * (15.0 + s2 * (63.0 * s2 - 70.0)),
        ];
        let mut sum = 0.0;
        let mut qn = q * q;
        for k in 0..4 {
            sum -= c[k] * qn * p[k];
            qn *= q;
        }
        0.5 * v.dot(v) - MU / rn + MU / rn * sum
    }

    /// The worst relative change of [`zonal_energy`] along the propagated state, each minute over
    /// three revolutions.
    fn energy_change(osc: &KeplerianElements, c: [f64; 4]) -> f64 {
        let bl = BrouwerLyddane::from_osculating(osc, RE, c[0], c[1], c[2], c[3]).unwrap();
        let (r0, v0) = bl.propagate(0.0);
        let e0 = zonal_energy(r0, v0, c);
        let minutes = (3.0 * osc.period() / 60.0).floor();
        let mut worst = 0.0f64;
        let mut t = 60.0;
        while t <= 60.0 * minutes {
            let (r, v) = bl.propagate(t);
            worst = worst.max(((zonal_energy(r, v, c) - e0) / e0).abs());
            t += 60.0;
        }
        worst
    }

    /// The propagated state keeps the `J2..J5` energy to third order: its osculating `a` is the
    /// one `−μ/(2a) + R = K''` fixes, and the first-order elements put the potential `R` within
    /// `O(J2³)` of its value at the state. Halving J2 divides the change by eight. (G4)
    #[test]
    fn the_state_keeps_the_zonal_energy_to_third_order() {
        let c = [C20, C30, C40, C50];
        let d: std::vec::Vec<f64> = orbits().iter().map(|osc| energy_change(osc, c)).collect();
        std::eprintln!("Brouwer-Lyddane relative energy change, J2-J5: {d:?}");
        // Measured 3.2e-8, 5.1e-8, 3.4e-11, 1.6e-8; with a'' + δa₁, 5.8e-6, 1.1e-5, 1.0e-8, 5.2e-6.
        for (k, bound) in [1.3e-7, 2e-7, 1.5e-10, 6e-8].into_iter().enumerate() {
            assert!(
                d[k] < bound,
                "orbit {k}: relative energy change {:.3e}",
                d[k]
            );
        }
        for osc in [eccentric(), leo()] {
            let full = energy_change(&osc, [C20, 0.0, 0.0, 0.0]);
            let half = energy_change(&osc, [0.5 * C20, 0.0, 0.0, 0.0]);
            let ratio = full / half;
            std::eprintln!(
                "Brouwer-Lyddane J2-only energy change: {full:.3e}, at J2/2 {half:.3e}, ratio {ratio:.3}"
            );
            // Measured 7.99 and 7.98; 4.00 and 4.00 with a'' + δa₁.
            assert!(
                (7.5..8.5).contains(&ratio),
                "e = {}: energy ratio {ratio:.3}",
                osc.eccentricity
            );
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
        assert!(worst <= 12, "the fixed point took {worst} corrections");
        assert!(
            (5.3..5.5).contains(&mean),
            "mean {mean:.2} corrections per case"
        );
    }

    /// The theory has a regime where the fixed point simply does not converge, and returning the
    /// last iterate there is how a caller ends up propagating a state the theory never reached:
    /// near retrograde equatorial, where Lyddane's `sin(i/2)` node variables divide by
    /// `cos(i/2)`. (S2)
    #[test]
    fn a_retrograde_near_equatorial_orbit_is_refused() {
        let c = [0.0, 0.0, C20, C30, C40, C50];
        for &(e, i_deg) in &[
            (0.0, 175.0),
            (1e-6, 178.0),
            (1e-3, 179.0),
            (0.01, 179.9),
            (0.1, 180.0),
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
