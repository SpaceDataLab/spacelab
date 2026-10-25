//! Eckstein–Hechler analytical propagator: zonal geopotential J2–J6 in circular elements.
//!
//! Eckstein & Hechler's mean-element satellite theory (a faithful port of Orekit's
//! `EcksteinHechlerPropagator.EHModel`): the orbit is carried as *mean* circular elements
//! `(a, eₓ, e_y, i, Ω, α_M)` whose only motion is the secular drift of the node, latitude argument
//! and eccentricity vector under J2/J2²/J4/J6; converting to *osculating* elements at a target time
//! adds the analytic long- and short-period corrections in the latitude argument. Working in the
//! circular (equinoctial-like) set `eₓ = e·cos ω`, `e_y = e·sin ω`, `α = ω + M` removes the small-
//! eccentricity singularity that classical `(e, ω, M)` theory carries.
//!
//! # Validity
//!
//! The theory is only defined for orbits that are neither near-equatorial (`sin²i < 10⁻¹⁰`), near the
//! critical inclination (`|sin²i − 4/5| < 10⁻³`, i.e. `i ≈ 63.4°`/`116.6°`, where the eccentricity
//! long-period denominator `rdpom` collapses), nor too eccentric (`e > 0.1`, where the series
//! diverges). Those are returned as [`EcksteinHechlerError`] at construction rather than silently
//! producing garbage — the same three guards Orekit raises.
//!
//! # Velocity
//!
//! Unlike [`crate::brouwer_lyddane`], the reported velocity is *not* the Keplerian velocity of the
//! osculating ellipse: Orekit's Eckstein–Hechler evaluates the circular elements as explicit
//! functions of time and differentiates them (its `UnivariateDerivative2`), so the velocity includes
//! the time variation of the short-period terms (~10 m/s for a LEO). This port carries a first-order
//! univariate dual `Ud` (value + d/dt) through the whole element→Cartesian chain to reproduce that
//! exact velocity. Matched against Orekit's own propagator (`tools/orekit-oracle`,
//! `orekit_eckstein_hechler.csv`) to ~10⁻⁶ m in position and ~10⁻⁹ m/s in velocity over a LEO orbit.
//!
//! The un-normalized zonal coefficients are `C_{n,0} = −J_n`; denormalize the embedded field's
//! `C̄_{n,0}` by `√(2n+1)`.
//!
//! # What checks the theory, as distinct from the port
//!
//! `derivations/zonal_secular.mac` derives the first-order J2 secular rates from the averaged
//! disturbing function and Lagrange's planetary equations, then subtracts this module's own
//! `omm_d`, `rdpom` and `rdl` at `e = 0` — the regime the circular theory is written for, since it
//! refuses `e > 0.1`. All three residuals are **identically zero**. The latitude-argument check is
//! the interesting one: `rdl` has to equal `1 + ω̇/n + Ṁ/n − 1` because `α = ω + M`, and it does,
//! which ties this propagator's variables to [`crate::brouwer_lyddane`]'s through one derivation
//! rather than through a shared ancestor.
//!
//! The J2², J4 and J6 terms are second order and are **not** derived there; they stay on the Orekit
//! comparison.

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

/// First-order univariate dual number `value + d·ε`, carrying a quantity and its time derivative.
///
/// Mirrors the value and first-derivative channels of Orekit's `UnivariateDerivative2`; the second
/// derivative (acceleration) is not needed for `(r, v)` and is dropped. Forward-mode chain rule.
#[derive(Clone, Copy)]
struct Ud {
    v: f64,
    d: f64,
}

impl Ud {
    #[inline]
    const fn cst(v: f64) -> Self {
        Self { v, d: 0.0 }
    }
    #[inline]
    const fn new(v: f64, d: f64) -> Self {
        Self { v, d }
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        Self {
            v: self.v + o.v,
            d: self.d + o.d,
        }
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self {
            v: self.v - o.v,
            d: self.d - o.d,
        }
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self {
            v: self.v * o.v,
            d: self.v * o.d + self.d * o.v,
        }
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        Self {
            v: self.v / o.v,
            d: (self.d * o.v - self.v * o.d) / (o.v * o.v),
        }
    }
    #[inline]
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: -self.d,
        }
    }
    #[inline]
    fn recip(self) -> Self {
        let r = 1.0 / self.v;
        Self {
            v: r,
            d: -self.d * r * r,
        }
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = libm::sqrt(self.v);
        Self {
            v: s,
            d: self.d / (2.0 * s),
        }
    }
    #[inline]
    fn square(self) -> Self {
        Self {
            v: self.v * self.v,
            d: 2.0 * self.v * self.d,
        }
    }
    #[inline]
    fn sin(self) -> Self {
        let (s, c) = libm::sincos(self.v);
        Self {
            v: s,
            d: self.d * c,
        }
    }
    #[inline]
    fn cos(self) -> Self {
        let (s, c) = libm::sincos(self.v);
        Self {
            v: c,
            d: -self.d * s,
        }
    }
    #[inline]
    fn mulf(self, k: f64) -> Self {
        Self {
            v: self.v * k,
            d: self.d * k,
        }
    }
    #[inline]
    fn addf(self, k: f64) -> Self {
        Self {
            v: self.v + k,
            d: self.d,
        }
    }
    #[inline]
    fn subf(self, k: f64) -> Self {
        Self {
            v: self.v - k,
            d: self.d,
        }
    }
}

/// Circular (equinoctial-like) orbital elements: `eₓ = e·cos ω`, `e_y = e·sin ω`, `α_M = ω + M`.
#[derive(Clone, Copy)]
struct Circular {
    a: f64,
    ex: f64,
    ey: f64,
    i: f64,
    raan: f64,
    alpha_m: f64,
}

/// Why an Eckstein–Hechler model could not be built — the three regimes where the theory is invalid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EcksteinHechlerError {
    /// Orbit is (almost) equatorial (`sin²i < 10⁻¹⁰`); the model is singular there.
    AlmostEquatorial {
        /// Mean inclination, degrees.
        inclination_deg: f64,
    },
    /// Orbit is (almost) critically inclined (`i ≈ 63.4°`/`116.6°`); the long-period denominator
    /// `rdpom = −¾ g2 (4 − 5 sin²i)` collapses.
    AlmostCriticallyInclined {
        /// Mean inclination, degrees.
        inclination_deg: f64,
    },
    /// Eccentricity exceeds `0.1`; the mean→osculating series diverges.
    TooLargeEccentricity {
        /// Mean eccentricity.
        eccentricity: f64,
    },
}

impl fmt::Display for EcksteinHechlerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlmostEquatorial { inclination_deg } => {
                write!(
                    f,
                    "almost equatorial orbit (i = {inclination_deg}°): Eckstein-Hechler singular"
                )
            }
            Self::AlmostCriticallyInclined { inclination_deg } => {
                write!(
                    f,
                    "almost critically inclined orbit (i = {inclination_deg}°)"
                )
            }
            Self::TooLargeEccentricity { eccentricity } => {
                write!(
                    f,
                    "eccentricity {eccentricity} too large for Eckstein-Hechler (> 0.1)"
                )
            }
        }
    }
}

impl core::error::Error for EcksteinHechlerError {}

/// Reduce `a` into the `2π`-interval centred on `center` (Orekit `MathUtils.normalizeAngle`).
#[inline]
fn normalize_angle(a: f64, center: f64) -> f64 {
    a - TAU * libm::floor((a + PI - center) / TAU)
}

/// The Eckstein–Hechler model built from a fixed set of *mean* circular elements: the secular rates
/// and the long- and short-period coefficients (functions of `a, e, i` only, computed once).
struct EhModel {
    mean: Circular,
    mu: f64,
    xnot_dot: f64,
    rdpom: f64,
    rdpomp: f64,
    eps1: f64,
    eps2: f64,
    omm_d: f64,
    a_md: f64,
    kh: f64,
    kl: f64,
    ax1: f64,
    ay1: f64,
    as1: f64,
    ac2: f64,
    axy3: f64,
    as3: f64,
    ac4: f64,
    as5: f64,
    ac6: f64,
    ex1: f64,
    exx2: f64,
    exy2: f64,
    ex3: f64,
    ex4: f64,
    ey1: f64,
    eyx2: f64,
    eyy2: f64,
    ey3: f64,
    ey4: f64,
    rx1: f64,
    ry1: f64,
    r2: f64,
    r3: f64,
    rl: f64,
    iy1: f64,
    ix1: f64,
    i2: f64,
    i3: f64,
    ih: f64,
    lx1: f64,
    ly1: f64,
    l2: f64,
    l3: f64,
    ll: f64,
}

impl EhModel {
    /// Build the model from mean circular elements. Returns [`EcksteinHechlerError`] in the three
    /// regimes where the theory is invalid (verbatim from Orekit `EHModel`).
    fn new(mean: Circular, re: f64, mu: f64, ck0: [f64; 7]) -> Result<Self, EcksteinHechlerError> {
        // preliminary processing
        let mut q = re / mean.a;
        let mut ql = q * q;
        let g2 = ck0[2] * ql;
        ql *= q;
        let g3 = ck0[3] * ql;
        ql *= q;
        let g4 = ck0[4] * ql;
        ql *= q;
        let g5 = ck0[5] * ql;
        ql *= q;
        let g6 = ck0[6] * ql;

        let (sin_i1, cos_i1) = libm::sincos(mean.i);
        let sin_i2 = sin_i1 * sin_i1;
        let sin_i4 = sin_i2 * sin_i2;
        let sin_i6 = sin_i2 * sin_i4;

        if sin_i2 < 1.0e-10 {
            return Err(EcksteinHechlerError::AlmostEquatorial {
                inclination_deg: mean.i.to_degrees(),
            });
        }
        if libm::fabs(sin_i2 - 4.0 / 5.0) < 1.0e-3 {
            return Err(EcksteinHechlerError::AlmostCriticallyInclined {
                inclination_deg: mean.i.to_degrees(),
            });
        }
        let ecc = libm::sqrt(mean.ex * mean.ex + mean.ey * mean.ey);
        if ecc > 0.1 {
            return Err(EcksteinHechlerError::TooLargeEccentricity { eccentricity: ecc });
        }

        let xnot_dot = libm::sqrt(mu / mean.a) / mean.a;

        let rdpom = -0.75 * g2 * (4.0 - 5.0 * sin_i2);
        let rdpomp = 7.5 * g4 * (1.0 - 31.0 / 8.0 * sin_i2 + 49.0 / 16.0 * sin_i4)
            - 13.125 * g6 * (1.0 - 8.0 * sin_i2 + 129.0 / 8.0 * sin_i4 - 297.0 / 32.0 * sin_i6);

        q = 3.0 / (32.0 * rdpom);
        let eps1 = q * g4 * sin_i2 * (30.0 - 35.0 * sin_i2)
            - 175.0 * q * g6 * sin_i2 * (1.0 - 3.0 * sin_i2 + 2.0625 * sin_i4);
        q = 3.0 * sin_i1 / (8.0 * rdpom);
        let eps2 = q * g3 * (4.0 - 5.0 * sin_i2) - q * g5 * (10.0 - 35.0 * sin_i2 + 26.25 * sin_i4);

        let omm_d = cos_i1
            * (1.50 * g2 - 2.25 * g2 * g2 * (2.5 - 19.0 / 6.0 * sin_i2)
                + 0.9375 * g4 * (7.0 * sin_i2 - 4.0)
                + 3.28125 * g6 * (2.0 - 9.0 * sin_i2 + 8.25 * sin_i4));

        let rdl = 1.0 - 1.50 * g2 * (3.0 - 4.0 * sin_i2);
        let a_md = rdl
            + 2.25 * g2 * g2 * (9.0 - 263.0 / 12.0 * sin_i2 + 341.0 / 24.0 * sin_i4)
            + 15.0 / 16.0 * g4 * (8.0 - 31.0 * sin_i2 + 24.5 * sin_i4)
            + 105.0 / 32.0 * g6 * (-10.0 / 3.0 + 25.0 * sin_i2 - 48.75 * sin_i4 + 27.5 * sin_i6);

        let qq = -1.5 * g2 / rdl;
        let q_a = 0.75 * g2 * g2 * sin_i2;
        let q_b = 0.25 * g4 * sin_i2;
        let q_c = 105.0 / 16.0 * g6 * sin_i2;
        let q_d = -0.75 * g3 * sin_i1;
        let q_e = 3.75 * g5 * sin_i1;
        let kh = 0.375 / rdpom;
        let kl = kh / sin_i1;

        let ax1 = qq * (2.0 - 3.5 * sin_i2);
        let ay1 = qq * (2.0 - 2.5 * sin_i2);
        let as1 = q_d * (4.0 - 5.0 * sin_i2) + q_e * (2.625 * sin_i4 - 3.5 * sin_i2 + 1.0);
        let ac2 = qq * sin_i2
            + q_a * 7.0 * (2.0 - 3.0 * sin_i2)
            + q_b * (15.0 - 17.5 * sin_i2)
            + q_c * (3.0 * sin_i2 - 1.0 - 33.0 / 16.0 * sin_i4);
        let axy3 = qq * 3.5 * sin_i2;
        let as3 = q_d * 5.0 / 3.0 * sin_i2 + q_e * 7.0 / 6.0 * sin_i2 * (1.0 - 1.125 * sin_i2);
        let ac4 = q_a * sin_i2 + q_b * 4.375 * sin_i2 + q_c * 0.75 * (1.1 * sin_i4 - sin_i2);
        let as5 = q_e * 21.0 / 80.0 * sin_i4;
        let ac6 = q_c * -11.0 / 80.0 * sin_i4;

        let ex1 = qq * (1.0 - 1.25 * sin_i2);
        let exx2 = qq * 0.5 * (3.0 - 5.0 * sin_i2);
        let exy2 = qq * (2.0 - 1.5 * sin_i2);
        let ex3 = qq * 7.0 / 12.0 * sin_i2;
        let ex4 = qq * 17.0 / 8.0 * sin_i2;

        let ey1 = qq * (1.0 - 1.75 * sin_i2);
        let eyx2 = qq * (1.0 - 3.0 * sin_i2);
        let eyy2 = qq * (2.0 * sin_i2 - 1.5);
        let ey3 = qq * 7.0 / 12.0 * sin_i2;
        let ey4 = qq * 17.0 / 8.0 * sin_i2;

        q = -qq * cos_i1;
        let rx1 = 3.5 * q;
        let ry1 = -2.5 * q;
        let r2 = -0.5 * q;
        let r3 = 7.0 / 6.0 * q;
        let rl = g3 * cos_i1 * (4.0 - 15.0 * sin_i2)
            - 2.5 * g5 * cos_i1 * (4.0 - 42.0 * sin_i2 + 52.5 * sin_i4);

        q = 0.5 * qq * sin_i1 * cos_i1;
        let iy1 = q;
        let ix1 = -q;
        let i2 = q;
        let i3 = q * 7.0 / 3.0;
        let ih = -g3 * cos_i1 * (4.0 - 5.0 * sin_i2)
            + 2.5 * g5 * cos_i1 * (4.0 - 14.0 * sin_i2 + 10.5 * sin_i4);

        let lx1 = qq * (7.0 - 77.0 / 8.0 * sin_i2);
        let ly1 = qq * (55.0 / 8.0 * sin_i2 - 7.50);
        let l2 = qq * (1.25 * sin_i2 - 0.5);
        let l3 = qq * (77.0 / 24.0 * sin_i2 - 7.0 / 6.0);
        let ll = g3 * (53.0 * sin_i2 - 4.0 - 57.5 * sin_i4)
            + 2.5 * g5 * (4.0 - 96.0 * sin_i2 + 269.5 * sin_i4 - 183.75 * sin_i6);

        Ok(Self {
            mean,
            mu,
            xnot_dot,
            rdpom,
            rdpomp,
            eps1,
            eps2,
            omm_d,
            a_md,
            kh,
            kl,
            ax1,
            ay1,
            as1,
            ac2,
            axy3,
            as3,
            ac4,
            as5,
            ac6,
            ex1,
            exx2,
            exy2,
            ex3,
            ex4,
            ey1,
            eyx2,
            eyy2,
            ey3,
            ey4,
            rx1,
            ry1,
            r2,
            r3,
            rl,
            iy1,
            ix1,
            i2,
            i3,
            ih,
            lx1,
            ly1,
            l2,
            l3,
            ll,
        })
    }

    /// Osculating circular elements `[a, eₓ, e_y, i, Ω, α_M]` as value+derivative duals at `dt`
    /// seconds from the mean epoch (Orekit `EHModel.propagateParameters`).
    fn propagate_parameters(&self, dt: f64) -> [Ud; 6] {
        // Keplerian evolution
        let dt_ud = Ud::new(dt, 1.0);
        let xnot = dt_ud.mulf(self.xnot_dot);

        // secular effects — eccentricity
        let x = xnot.mulf(self.rdpom + self.rdpomp);
        let cx = x.cos();
        let sx = x.sin();
        let exm = cx
            .mulf(self.mean.ex)
            .add(sx.mulf(self.eps2 - (1.0 - self.eps1) * self.mean.ey));
        let eym = sx
            .mulf((1.0 + self.eps1) * self.mean.ex)
            .add(cx.mulf(self.mean.ey - self.eps2))
            .addf(self.eps2);

        // no secular effect on inclination

        // right ascension of ascending node
        let omm = Ud::new(
            normalize_angle(self.mean.raan + self.omm_d * xnot.v, PI),
            self.omm_d * self.xnot_dot,
        );

        // latitude argument
        let xlm = Ud::new(
            normalize_angle(self.mean.alpha_m + self.a_md * xnot.v, PI),
            self.a_md * self.xnot_dot,
        );

        // periodical terms
        let cl1 = xlm.cos();
        let sl1 = xlm.sin();
        let cl2 = cl1.mul(cl1).sub(sl1.mul(sl1));
        let sl2 = cl1.mul(sl1).add(sl1.mul(cl1));
        let cl3 = cl2.mul(cl1).sub(sl2.mul(sl1));
        let sl3 = cl2.mul(sl1).add(sl2.mul(cl1));
        let cl4 = cl3.mul(cl1).sub(sl3.mul(sl1));
        let sl4 = cl3.mul(sl1).add(sl3.mul(cl1));
        let cl5 = cl4.mul(cl1).sub(sl4.mul(sl1));
        let sl5 = cl4.mul(sl1).add(sl4.mul(cl1));
        let cl6 = cl5.mul(cl1).sub(sl5.mul(sl1));

        let qh = eym.subf(self.eps2).mulf(self.kh);
        let ql = exm.mulf(self.kl);

        let exm_cl1 = exm.mul(cl1);
        let exm_sl1 = exm.mul(sl1);
        let eym_cl1 = eym.mul(cl1);
        let eym_sl1 = eym.mul(sl1);
        let exm_cl2 = exm.mul(cl2);
        let exm_sl2 = exm.mul(sl2);
        let eym_cl2 = eym.mul(cl2);
        let eym_sl2 = eym.mul(sl2);
        let exm_cl3 = exm.mul(cl3);
        let exm_sl3 = exm.mul(sl3);
        let eym_cl3 = eym.mul(cl3);
        let eym_sl3 = eym.mul(sl3);
        let exm_cl4 = exm.mul(cl4);
        let exm_sl4 = exm.mul(sl4);
        let eym_cl4 = eym.mul(cl4);
        let eym_sl4 = eym.mul(sl4);

        // semi major axis
        let rda = exm_cl1
            .mulf(self.ax1)
            .add(eym_sl1.mulf(self.ay1))
            .add(sl1.mulf(self.as1))
            .add(cl2.mulf(self.ac2))
            .add(exm_cl3.add(eym_sl3).mulf(self.axy3))
            .add(sl3.mulf(self.as3))
            .add(cl4.mulf(self.ac4))
            .add(sl5.mulf(self.as5))
            .add(cl6.mulf(self.ac6));

        // eccentricity
        let rdex = cl1
            .mulf(self.ex1)
            .add(exm_cl2.mulf(self.exx2))
            .add(eym_sl2.mulf(self.exy2))
            .add(cl3.mulf(self.ex3))
            .add(exm_cl4.add(eym_sl4).mulf(self.ex4));
        let rdey = sl1
            .mulf(self.ey1)
            .add(exm_sl2.mulf(self.eyx2))
            .add(eym_cl2.mulf(self.eyy2))
            .add(sl3.mulf(self.ey3))
            .add(exm_sl4.sub(eym_cl4).mulf(self.ey4));

        // ascending node
        let rdom = exm_sl1
            .mulf(self.rx1)
            .add(eym_cl1.mulf(self.ry1))
            .add(sl2.mulf(self.r2))
            .add(eym_cl3.sub(exm_sl3).mulf(self.r3))
            .add(ql.mulf(self.rl));

        // inclination
        let rdxi = eym_sl1
            .mulf(self.iy1)
            .add(exm_cl1.mulf(self.ix1))
            .add(cl2.mulf(self.i2))
            .add(exm_cl3.add(eym_sl3).mulf(self.i3))
            .add(qh.mulf(self.ih));

        // latitude argument
        let rdxl = exm_sl1
            .mulf(self.lx1)
            .add(eym_cl1.mulf(self.ly1))
            .add(sl2.mulf(self.l2))
            .add(exm_sl3.sub(eym_cl3).mulf(self.l3))
            .add(ql.mulf(self.ll));

        // osculating parameters
        [
            rda.addf(1.0).mulf(self.mean.a),
            rdex.add(exm),
            rdey.add(eym),
            rdxi.addf(self.mean.i),
            rdom.add(omm),
            rdxl.add(xlm),
        ]
    }

    /// Cartesian `(r, v)` from the circular parameters *with derivatives* — the value channel is the
    /// position, the derivative channel the velocity (Orekit `toCartesian`).
    fn to_cartesian(p: &[Ud; 6]) -> (Vec3, Vec3) {
        let cos_omega = p[4].cos();
        let sin_omega = p[4].sin();
        let cos_i = p[3].cos();
        let sin_i = p[3].sin();
        let (alpha_e, _) = mean_to_eccentric(p[5], p[1], p[2]);
        let cos_ae = alpha_e.cos();
        let sin_ae = alpha_e.sin();
        let ex2 = p[1].square();
        let ey2 = p[2].square();
        let exy = p[1].mul(p[2]);
        let q = ex2.add(ey2).subf(1.0).neg().sqrt();
        let beta = q.addf(1.0).recip();
        let bx2 = beta.mul(ex2);
        let by2 = beta.mul(ey2);
        let bxy = beta.mul(exy);
        let u = bxy.mul(sin_ae).sub(p[1].add(by2.subf(1.0).mul(cos_ae)));
        let v = bxy.mul(cos_ae).sub(p[2].add(bx2.subf(1.0).mul(sin_ae)));
        let x = p[0].mul(u);
        let y = p[0].mul(v);

        let px = x.mul(cos_omega).sub(y.mul(cos_i.mul(sin_omega)));
        let py = x.mul(sin_omega).add(y.mul(cos_i.mul(cos_omega)));
        let pz = y.mul(sin_i);

        (Vec3::new(px.v, py.v, pz.v), Vec3::new(px.d, py.d, pz.d))
    }
}

/// Eccentric latitude argument `α_E` from the mean latitude argument (generalized Kepler equation in
/// circular parameters, Halley iteration — Orekit `meanToEccentric`).
///
/// Returns the argument and the number of Halley steps. The iteration is self-correcting: a scaled
/// shift or a corrupted denominator still lands on the same root of
/// `α_E − α_M = eₓ sin α_E − e_y cos α_E`, which is all the returned angle records. The count is the
/// only place a difference between them survives.
fn mean_to_eccentric(alpha_m: Ud, ex: Ud, ey: Ud) -> (Ud, u32) {
    let mut alpha_e = alpha_m;
    let mut alpha_e_m_alpha_m = Ud::cst(0.0);
    let mut cos_ae = alpha_e.cos();
    let mut sin_ae = alpha_e.sin();
    let mut iter = 0u32;
    loop {
        let f2 = ex.mul(sin_ae).sub(ey.mul(cos_ae));
        let f1 = Ud::cst(1.0).sub(ex.mul(cos_ae)).sub(ey.mul(sin_ae));
        let f0 = alpha_e_m_alpha_m.sub(f2);
        let f12 = f1.mulf(2.0);
        let shift = f0.mul(f12).div(f1.mul(f12).sub(f0.mul(f2)));
        alpha_e_m_alpha_m = alpha_e_m_alpha_m.sub(shift);
        alpha_e = alpha_m.add(alpha_e_m_alpha_m);
        cos_ae = alpha_e.cos();
        sin_ae = alpha_e.sin();
        iter += 1;
        if iter >= 50 || libm::fabs(shift.v) <= 1.0e-12 {
            break;
        }
    }
    (alpha_e, iter)
}

/// Osculating→mean conversion: the fixed point of the mean→osculating map at `dt = 0`, iterated
/// undamped (Orekit's `FixedPointConverter` with `EcksteinHechlerTheory`).
///
/// Returns the converged elements and the number of corrections applied. An undamped fixed point
/// reaches the same place from any intermediate state, so the elements record nothing about the
/// corrections that produced them; the count does.
fn osculating_to_mean(
    osc: Circular,
    re: f64,
    mu: f64,
    ck0: [f64; 7],
) -> Result<(Circular, u32), EcksteinHechlerError> {
    let mut mean = osc;
    let mut steps = 0;
    for _ in 0..100 {
        let model = EhModel::new(mean, re, mu, ck0)?;
        let p = model.propagate_parameters(0.0);
        let da = osc.a - p[0].v;
        let dex = osc.ex - p[1].v;
        let dey = osc.ey - p[2].v;
        let di = osc.i - p[3].v;
        let draan = normalize_angle(osc.raan - p[4].v, 0.0);
        let dalpha = normalize_angle(osc.alpha_m - p[5].v, 0.0);
        mean.a += da;
        mean.ex += dex;
        mean.ey += dey;
        mean.i += di;
        mean.raan += draan;
        mean.alpha_m += dalpha;
        steps += 1;
        if libm::fabs(da) / osc.a < 1.0e-13
            && libm::fabs(dex) < 1.0e-13
            && libm::fabs(dey) < 1.0e-13
            && libm::fabs(di) < 1.0e-13
            && libm::fabs(draan) < 1.0e-13
            && libm::fabs(dalpha) < 1.0e-13
        {
            break;
        }
    }
    Ok((mean, steps))
}

/// `KeplerianElements` → circular set: `eₓ = e·cos ω`, `e_y = e·sin ω`, `α_M = ω + M`.
fn keplerian_to_circular(k: &KeplerianElements) -> Circular {
    let e = k.eccentricity;
    let argp = k.arg_periapsis.radians();
    let m = k.mean_anomaly().radians();
    let (s, c) = libm::sincos(argp);
    Circular {
        a: k.semi_major_axis,
        ex: e * c,
        ey: e * s,
        i: k.inclination.radians(),
        raan: k.raan.radians(),
        alpha_m: argp + m,
    }
}

/// Circular set → `KeplerianElements` (mean anomaly carried through `at_mean_anomaly`).
fn circular_to_keplerian(c: Circular, mu: f64) -> KeplerianElements {
    let e = libm::sqrt(c.ex * c.ex + c.ey * c.ey);
    let argp = libm::atan2(c.ey, c.ex);
    let m = c.alpha_m - argp;
    KeplerianElements {
        semi_major_axis: c.a,
        eccentricity: e,
        inclination: Angle::from_radians(c.i),
        raan: Angle::from_radians(c.raan),
        arg_periapsis: Angle::from_radians(argp),
        true_anomaly: Angle::from_radians(0.0),
        mu,
    }
    .at_mean_anomaly(Angle::from_radians(m))
}

/// An Eckstein–Hechler analytical propagator, built from an osculating state under the zonal field
/// J2–J6. Cheap to evaluate at any epoch — a mean-element secular drift plus closed-form periodic
/// terms in the latitude argument.
pub struct EcksteinHechler {
    model: EhModel,
    re: f64,
}

impl EcksteinHechler {
    /// Build from initial *osculating* Keplerian elements and the un-normalized zonal coefficients
    /// `c20..c60 = −J2..−J6`, with the central-body reference radius `re` (m).
    ///
    /// # Errors
    ///
    /// Returns [`EcksteinHechlerError`] when the mean orbit is near-equatorial, near the critical
    /// inclination, or has eccentricity `> 0.1` — the regimes where the theory is invalid.
    pub fn from_osculating(
        osculating: &KeplerianElements,
        re: f64,
        c20: f64,
        c30: f64,
        c40: f64,
        c50: f64,
        c60: f64,
    ) -> Result<Self, EcksteinHechlerError> {
        let ck0 = [0.0, 0.0, c20, c30, c40, c50, c60];
        let osc = keplerian_to_circular(osculating);
        let (mean, _) = osculating_to_mean(osc, re, osculating.mu, ck0)?;
        let model = EhModel::new(mean, re, osculating.mu, ck0)?;
        Ok(Self { model, re })
    }

    /// The converged mean elements (the secular set the theory carries).
    #[must_use]
    pub fn mean_elements(&self) -> KeplerianElements {
        circular_to_keplerian(self.model.mean, self.model.mu)
    }

    /// Osculating elements at `dt` seconds from the initial epoch.
    #[must_use]
    pub fn osculating_at(&self, dt: f64) -> KeplerianElements {
        let p = self.model.propagate_parameters(dt);
        circular_to_keplerian(
            Circular {
                a: p[0].v,
                ex: p[1].v,
                ey: p[2].v,
                i: p[3].v,
                raan: p[4].v,
                alpha_m: p[5].v,
            },
            self.model.mu,
        )
    }

    /// Osculating Cartesian state `(r, v)` at `dt` seconds — position and its consistent time
    /// derivative (the velocity Orekit's propagator reports, including short-period time variation),
    /// not the Keplerian velocity of the osculating ellipse.
    #[must_use]
    pub fn propagate(&self, dt: f64) -> (Vec3, Vec3) {
        let p = self.model.propagate_parameters(dt);
        EhModel::to_cartesian(&p)
    }

    /// The reference radius the model was built with (m).
    #[must_use]
    pub fn reference_radius(&self) -> f64 {
        self.re
    }
}

#[cfg(test)]
mod tests {
    // The value slot is asserted with `==`: it is the function itself, exactly, or it is wrong.
    #![allow(clippy::excessive_precision, clippy::unwrap_used, clippy::float_cmp)]
    use super::*;

    const MU: f64 = 3.98600441500000000e14;
    const RE: f64 = 6_378_136.3;
    // Un-normalized zonal coefficients = embedded XGM2019e C̄_{n,0} · √(2n+1) (verbatim from the
    // Orekit oracle run, which denormalized the same C̄ values — so both sides use identical Cn0).
    const C20: f64 = -1.08263590288727700e-03;
    const C30: f64 = 2.53244065127664100e-06;
    const C40: f64 = 1.61993350893732940e-06;
    const C50: f64 = 2.27646142664844100e-07;
    const C60: f64 = -5.40773886151106700e-07;

    fn initial() -> KeplerianElements {
        KeplerianElements {
            semi_major_axis: 7_200_000.0,
            eccentricity: 1.0e-3,
            inclination: Angle::from_radians(98.0_f64.to_radians()),
            raan: Angle::from_radians(30.0_f64.to_radians()),
            arg_periapsis: Angle::from_radians(40.0_f64.to_radians()),
            true_anomaly: Angle::from_radians(0.0),
            mu: MU,
        }
        .at_mean_anomaly(Angle::from_radians(50.0_f64.to_radians()))
    }

    // Orekit EcksteinHechlerPropagator, osculating (r, v) in GCRF at t = 0, 900, …, 6300 s
    // (tools/orekit-oracle EhOracle, same Cn0 as above; a=7200 km, e=1e-3, i=98°, Ω=30°, ω=40°, M=50°).
    const GOLDEN: [(f64, [f64; 6]); 8] = [
        (
            0.0,
            [
                4.91146122546749800e+05,
                -8.72755661417147000e+05,
                7.12534287616728800e+06,
                -6.44833162985526000e+03,
                -3.72202861556182700e+03,
                -5.65336658869827200e+00,
            ],
        ),
        (
            900.0,
            [
                -4.71082838128842200e+06,
                -3.41182390555831840e+06,
                4.25756031535515900e+06,
                -4.26970448467034400e+03,
                -1.50966802153456230e+03,
                -5.90027249901970400e+03,
            ],
        ),
        (
            1800.0,
            [
                -6.14106882703341600e+06,
                -3.21875077496804060e+06,
                -2.02993235674086770e+06,
                1.31774883586857000e+03,
                1.90475335086773980e+03,
                -7.05319592565899600e+03,
            ],
        ),
        (
            2700.0,
            [
                -2.67415888889320600e+06,
                -4.57882008605393230e+05,
                -6.70121665432052000e+06,
                5.82503926538884500e+03,
                3.78545254365606430e+03,
                -2.58170121386130450e+03,
            ],
        ),
        (
            3600.0,
            [
                2.91492156916821330e+06,
                2.66721912836028500e+06,
                -6.04735679729033000e+06,
                5.70065727527476200e+03,
                2.65845514782463670e+03,
                3.93362327652895240e+03,
            ],
        ),
        (
            4500.0,
            [
                6.18056665093668800e+06,
                3.66879335699910400e+06,
                -5.77156699355832300e+05,
                1.02198443989693800e+03,
                -5.99664775451252600e+02,
                7.34382422205461500e+03,
            ],
        ),
        (
            5400.0,
            [
                4.49141629763128400e+06,
                1.73015481374044950e+06,
                5.35087687987657200e+06,
                -4.50335492183132400e+03,
                -3.39474890922495520e+03,
                4.86433742410145100e+03,
            ],
        ),
        (
            6300.0,
            [
                -8.06766597043414000e+05,
                -1.59898049227516630e+06,
                6.97061625246435200e+06,
                -6.41032687984834600e+03,
                -3.46336455439094240e+03,
                -1.52571927951536780e+03,
            ],
        ),
    ];

    #[test]
    fn matches_orekit_reference() {
        let eh = EcksteinHechler::from_osculating(&initial(), RE, C20, C30, C40, C50, C60).unwrap();
        let mut max_dr = 0.0f64;
        let mut max_dv = 0.0f64;
        for (t, g) in GOLDEN {
            let (r, v) = eh.propagate(t);
            let dr = libm::sqrt((r.x - g[0]).powi(2) + (r.y - g[1]).powi(2) + (r.z - g[2]).powi(2));
            let dv = libm::sqrt((v.x - g[3]).powi(2) + (v.y - g[4]).powi(2) + (v.z - g[5]).powi(2));
            assert!(dr < 1e-3, "t={t}: position error {dr:.3e} m");
            assert!(dv < 1e-6, "t={t}: velocity error {dv:.3e} m/s");
            max_dr = max_dr.max(dr);
            max_dv = max_dv.max(dv);
        }
        std::eprintln!(
            "Eckstein-Hechler vs Orekit: max |Δr| = {max_dr:.3e} m, |Δv| = {max_dv:.3e} m/s"
        );
    }

    /// Non-Orekit anchor: the model's secular nodal rate `Ω̇ = ommD·xnotDot` reproduces the classic
    /// J2 nodal-regression formula `Ω̇ = −³⁄₂ n J2 (Rₑ/p)² cos i` to the J2²/J4 level (~10⁻³ rel). (G3)
    #[test]
    fn secular_node_rate_matches_j2_formula() {
        let eh = EcksteinHechler::from_osculating(&initial(), RE, C20, C30, C40, C50, C60).unwrap();
        let node_rate = eh.model.omm_d * eh.model.xnot_dot;

        let mean = eh.mean_elements();
        let a = mean.semi_major_axis;
        let e = mean.eccentricity;
        let i = mean.inclination.radians();
        let n = libm::sqrt(MU / (a * a * a));
        let p = a * (1.0 - e * e);
        let j2 = -C20;
        let analytic = -1.5 * n * j2 * (RE / p) * (RE / p) * libm::cos(i);

        let rel = libm::fabs(node_rate - analytic) / libm::fabs(analytic);
        std::eprintln!(
            "EH Ω̇ = {node_rate:.6e} rad/s, J2 analytic = {analytic:.6e}, rel = {rel:.3e}"
        );
        assert!(rel < 5e-3, "node rate rel err {rel:.3e}");
        assert!(node_rate > 0.0, "i=98° is retrograde: node should advance");
    }

    #[test]
    fn rejects_critical_inclination() {
        let osc = KeplerianElements {
            inclination: Angle::from_radians(63.434949_f64.to_radians()),
            ..initial()
        };
        let result = EcksteinHechler::from_osculating(&osc, RE, C20, C30, C40, C50, C60);
        assert!(
            matches!(
                result,
                Err(EcksteinHechlerError::AlmostCriticallyInclined { .. })
            ),
            "expected AlmostCriticallyInclined at the critical inclination"
        );
    }

    #[test]
    fn rejects_high_eccentricity() {
        let osc = KeplerianElements {
            eccentricity: 0.2,
            ..initial()
        };
        let result = EcksteinHechler::from_osculating(&osc, RE, C20, C30, C40, C50, C60);
        assert!(
            matches!(
                result,
                Err(EcksteinHechlerError::TooLargeEccentricity { .. })
            ),
            "expected TooLargeEccentricity at e = 0.2"
        );
    }
    /// Central difference of `f` at `0`, as an oracle for the dual number's derivative slot. The
    /// step is chosen where the truncation error `h²f‴/6` and the cancellation error `ε/h` cross for
    /// the O(1) arguments below.
    fn slope(f: impl Fn(f64) -> f64) -> f64 {
        let h = 1e-5;
        (f(h) - f(-h)) / (2.0 * h)
    }

    fn close(got: f64, want: f64, what: &str) {
        let rel = libm::fabs(got - want) / libm::fabs(want).max(1.0);
        assert!(rel < 1e-7, "{what}: derivative {got:.12e} vs {want:.12e}");
    }

    /// Every velocity the model reports is a derivative slot of [`Ud`], and the slot is carried, not
    /// checked: it is exercised only through orbits where most operands are constants with `d = 0`,
    /// which leaves whole operators free to be wrong. A finite difference of the value function is
    /// an oracle from outside the arithmetic under test.
    #[test]
    fn the_dual_number_carries_the_derivative_of_its_value() {
        let (a, da, b, db) = (1.7, 0.9, 0.6, -1.3);
        let (x, y) = (Ud::new(a, da), Ud::new(b, db));
        close(x.add(y).d, slope(|t| (a + da * t) + (b + db * t)), "add");
        close(x.sub(y).d, slope(|t| (a + da * t) - (b + db * t)), "sub");
        close(x.mul(y).d, slope(|t| (a + da * t) * (b + db * t)), "mul");
        close(x.div(y).d, slope(|t| (a + da * t) / (b + db * t)), "div");
        close(x.neg().d, slope(|t| -(a + da * t)), "neg");
        close(x.recip().d, slope(|t| 1.0 / (a + da * t)), "recip");
        close(x.sqrt().d, slope(|t| libm::sqrt(a + da * t)), "sqrt");
        close(
            x.square().d,
            slope(|t| (a + da * t) * (a + da * t)),
            "square",
        );
        close(x.sin().d, slope(|t| libm::sin(a + da * t)), "sin");
        close(x.cos().d, slope(|t| libm::cos(a + da * t)), "cos");
        close(x.mulf(-2.5).d, slope(|t| (a + da * t) * -2.5), "mulf");
        close(x.addf(-2.5).d, slope(|t| (a + da * t) - 2.5), "addf");
        close(x.subf(-2.5).d, slope(|t| (a + da * t) + 2.5), "subf");
        // The value slot has to be the function itself, not merely something with the right slope.
        assert_eq!(x.div(y).v, a / b);
        assert_eq!(x.recip().v, 1.0 / a);
        assert_eq!(x.neg().v, -a);
    }

    /// `normalize_angle` has two obligations and the model relies on both: land inside the `2π`
    /// window centred on `center`, and move the angle by a whole number of turns. Neither is visible
    /// in a propagated state, where the angle is only ever fed to `sin` and `cos`.
    #[test]
    fn normalize_angle_lands_in_the_window_a_whole_number_of_turns_away() {
        for &center in &[0.0, PI, -2.5, 7.3] {
            for k in -9..=9 {
                for &frac in &[0.0, 0.37, 0.5, 0.999] {
                    let a = center + TAU * (f64::from(k) + frac) - PI;
                    let got = normalize_angle(a, center);
                    // The reduction is `a − 2π⌊(a+π−c)/2π⌋`, so when `a` sits on the window edge
                    // the quotient rounds either way and the result can land a few ulps outside.
                    // The slack is 10⁻⁹ rad, fifteen orders under the π-scale shifts that a wrong
                    // reduction produces.
                    assert!(
                        got >= center - PI - 1e-9 && got < center + PI + 1e-9,
                        "a={a}, center={center}: {got} outside the window"
                    );
                    let turns = (got - a) / TAU;
                    assert!(
                        libm::fabs(turns - libm::round(turns)) < 1e-12,
                        "a={a}, center={center}: moved {turns} turns"
                    );
                }
            }
        }
    }

    /// Halley reaches the root of `α_E − α_M = eₓ sin α_E − e_y cos α_E` from a corrupted shift just
    /// as well as from the right one, so the returned angle cannot separate them. The residual pins
    /// the root — written from the definition, not from the solver — and the two-sided budget pins
    /// the path to it. A budget with only a ceiling is satisfied by a loop that stops at once.
    #[test]
    fn the_latitude_argument_solve_reaches_its_root_within_a_measured_number_of_steps() {
        let (mut least, mut worst, mut total, mut calls) = (u32::MAX, 0, 0u32, 0u32);
        for &e in &[0.0, 1e-6, 1e-3, 0.01, 0.05, 0.1] {
            for j in 0..8 {
                let w = TAU * f64::from(j) / 8.0;
                let (sw, cw) = libm::sincos(w);
                for k in 0..24 {
                    let am = -PI + TAU * f64::from(k) / 24.0;
                    let (ae, steps) =
                        mean_to_eccentric(Ud::new(am, 0.0), Ud::cst(e * cw), Ud::cst(e * sw));
                    let (sae, cae) = libm::sincos(ae.v);
                    let residual = libm::fabs(normalize_angle(
                        ae.v - am - e * cw * sae + e * sw * cae,
                        0.0,
                    ));
                    assert!(residual < 1e-15, "e={e}, aM={am}: residual {residual:.3e}");
                    least = least.min(steps);
                    worst = worst.max(steps);
                    total += steps;
                    calls += 1;
                }
            }
        }
        let mean = f64::from(total) / f64::from(calls);
        std::eprintln!("Eckstein-Hechler mean→eccentric: {least}..{worst} steps, mean {mean:.3}");
        assert!(worst <= 3, "the solve took {worst} steps");
        assert!((2.20..2.25).contains(&mean), "mean {mean:.3} steps");
    }

    /// As above for the osculating→mean fixed point: the converged elements are the same from any
    /// intermediate state, and the cost is the only record of how they were reached.
    #[test]
    fn the_fixed_point_converges_within_a_measured_number_of_corrections() {
        let ck0 = [0.0, 0.0, C20, C30, C40, C50, C60];
        let (mut least, mut worst, mut total, mut cases) = (u32::MAX, 0, 0u32, 0u32);
        for &e in &[0.0, 1e-4, 1e-3, 0.01, 0.05, 0.09] {
            for &i_deg in &[20.0, 50.0, 75.0, 98.0, 130.0] {
                let osc = Circular {
                    a: 7_200_000.0,
                    ex: e * libm::cos(0.7),
                    ey: e * libm::sin(0.7),
                    i: i_deg * PI / 180.0,
                    raan: 0.5,
                    alpha_m: 1.1,
                };
                let (_, steps) = osculating_to_mean(osc, RE, MU, ck0).unwrap();
                least = least.min(steps);
                worst = worst.max(steps);
                total += steps;
                cases += 1;
            }
        }
        let mean = f64::from(total) / f64::from(cases);
        std::eprintln!(
            "Eckstein-Hechler osculating→mean: {least}..{worst} corrections, mean {mean:.3}"
        );
        assert!(
            least >= 5,
            "the fixed point exited after {least} corrections"
        );
        assert!(worst <= 6, "the fixed point took {worst} corrections");
        assert!((5.00..5.06).contains(&mean), "mean {mean:.3} corrections");
    }

    /// The loop exits on a conjunction — every one of the six corrections has to have collapsed —
    /// and the returned elements record none of that: they are the last iterate whatever stopped it.
    /// The defining property is what has to be asserted, so run the conversion back out through the
    /// public mean→osculating map and require the input to reappear in all six components at once.
    /// The grid reaches the near-equatorial edge of the model's domain (`sin²i > 10⁻¹⁰`), where the
    /// inclination correction starts orders below the eccentricity correction and collapses first.
    #[test]
    fn the_conversion_reproduces_the_osculating_input_in_every_component() {
        // The bar is the noise floor of the converged map, not the loop's own 10⁻¹³ exit tolerance:
        // a fixed point left one correction short still round-trips inside that tolerance. The two
        // angles sit two decades above the rest because they carry the whole periodic series and
        // come back some tens of ulps away; a, eₓ, e_y and i are measured relative or are O(10⁻³).
        const TOL: [f64; 6] = [1.0e-15, 1.0e-15, 1.0e-15, 1.0e-15, 1.0e-13, 1.0e-13];
        let mut worst = [0.0f64; 6];
        for &e in &[0.0, 1.0e-4, 0.01, 0.09] {
            for &i_deg in &[0.01, 1.0, 30.0, 98.0, 150.0] {
                let osc = KeplerianElements {
                    eccentricity: e,
                    inclination: Angle::from_radians(i_deg * PI / 180.0),
                    ..initial()
                };
                let eh =
                    EcksteinHechler::from_osculating(&osc, RE, C20, C30, C40, C50, C60).unwrap();
                let (want, got) = (keplerian_to_circular(&osc), {
                    let back = eh.osculating_at(0.0);
                    keplerian_to_circular(&back)
                });
                let residual = [
                    libm::fabs(got.a - want.a) / want.a,
                    libm::fabs(got.ex - want.ex),
                    libm::fabs(got.ey - want.ey),
                    libm::fabs(got.i - want.i),
                    libm::fabs(normalize_angle(got.raan - want.raan, 0.0)),
                    libm::fabs(normalize_angle(got.alpha_m - want.alpha_m, 0.0)),
                ];
                for (w, r) in worst.iter_mut().zip(residual) {
                    *w = spacelab_testkit::nan_max(*w, r);
                }
            }
        }
        std::eprintln!("Eckstein-Hechler round trip: {worst:?}");
        for ((name, r), tol) in ["a", "ex", "ey", "i", "raan", "alphaM"]
            .iter()
            .zip(worst)
            .zip(TOL)
        {
            assert!(r < tol, "{name} did not come back: residual {r:.3e}");
        }
    }
}
