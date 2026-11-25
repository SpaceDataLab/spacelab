//! Outward rounding for the enclosures of this crate (plan 18 Ф28.V): intervals whose ends are the
//! exact results rounded down and up, and the `libm` functions widened by their H-libm constants.
//!
//! `+ − × ÷ √` round to nearest, so the exact result lies between the doubles just below and just
//! above the rounded one. That needs no hypothesis. Each operation also has an exact residual: the
//! two-sum error, `fma(a, b, −p)` for a product, `a − q·b` for a quotient and `a − s²` for a root,
//! exact away from underflow (`derivations/outward_rounding.gp`). Its sign says on which side of the
//! rounded result the exact one lies, so that result is already one end and only the other moves one
//! double (`next_down`, `next_up`); a zero residual gives a point. Where the residual is not exact,
//! below `2⁻⁹⁰⁰` or past overflow, both ends move. These are the directed roundings RD and RU,
//! one double wide at most. `libm` 0.2.16 is not
//! correctly rounded: on each row's domain it is within a measured number of ulps of the true value,
//! a tested hypothesis named H-libm (`derivations/h_libm.txt`, `docs/verification.md` §1.1). Every
//! interval that passes through [`sin`], [`cos`], [`atan2_pt`] or [`acos`] therefore holds **under
//! H-libm**.
//!
//! The constant `c` of a row bounds `|r − f| ≤ c·ulp(f)`, with `ulp(f)` taken at the true value
//! `f`. [`libm_iv`] turns it into a bound at the returned `r`: `ulp(f) ≤ 2⁻⁵²|f| + 2⁻¹⁰⁷⁴` and
//! `|f| ≤ |r| + |r − f|`, so `|r − f| ≤ (c·2⁻⁵²|r| + c·2⁻¹⁰⁷⁴)/(1 − c·2⁻⁵²)`, which for `c ≤ 2` is at
//! most `(c + 2⁻⁴⁰)·2⁻⁵²·|r| + 2⁻¹⁰⁷²` (`derivations/outward_rounding.mac`).

use spacelab_core::Vec3;

/// `sin_all` of `derivations/h_libm.txt`: libm's `sin` within this many ulps of `sin x`.
pub(crate) const H_SIN: f64 = 0.843_75;
/// `cos_all`: libm's `cos`.
pub(crate) const H_COS: f64 = 0.843_75;
/// `atan2_all`: libm's `atan2`, for `|y/x|` in `[2⁻⁶⁰, 2⁶⁰]`.
pub(crate) const H_ATAN2: f64 = 1.5625;
/// `acos_all`: libm's `acos` on `[−1, 1]`.
pub(crate) const H_ACOS: f64 = 0.9375;

/// `2⁻⁵²`.
const TWO_M52: f64 = f64::EPSILON;
/// `2⁻⁴⁰`, the slack on an H-libm constant that absorbs `1/(1 − c·2⁻⁵²)`.
const TWO_M40: f64 = 4096.0 * f64::EPSILON;
/// `2⁻¹⁰⁷²`, at least `c·2⁻¹⁰⁷⁴/(1 − c·2⁻⁵²)` for `c ≤ 2`: an ulp at a subnormal true value.
const TINY: f64 = f64::from_bits(4);
/// Below this `|y/x|`, `atan` is enclosed by its series instead of libm, whose row starts at `2⁻⁶⁰`.
const SMALL_RATIO: f64 = 4.0 * f64::EPSILON;

#[cfg(test)]
std::thread_local! {
    /// Rule 16's planted defect, per test thread: every operation rounded to nearest and nothing
    /// widened, neither the directed roundings nor libm's H-libm allowance.
    pub(crate) static NEAREST: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[cfg(test)]
fn planted() -> bool {
    NEAREST.with(core::cell::Cell::get)
}

#[cfg(not(test))]
const fn planted() -> bool {
    false
}

/// The double just below a value rounded to nearest: at most the exact value.
pub(crate) fn down(x: f64) -> f64 {
    if planted() {
        return x;
    }
    x.next_down()
}

/// The double just above a value rounded to nearest: at least the exact value.
pub(crate) fn up(x: f64) -> f64 {
    if planted() {
        return x;
    }
    x.next_up()
}

/// A closed interval of reals with double ends, `lo ≤ hi`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Iv {
    pub(crate) lo: f64,
    pub(crate) hi: f64,
}

/// `π`, between the double `PI` (below it) and the next.
pub(crate) const PI: Iv = Iv {
    lo: core::f64::consts::PI,
    hi: f64::from_bits(core::f64::consts::PI.to_bits() + 1),
};
/// `2π`, between `TAU` and the next double.
pub(crate) const TAU: Iv = Iv {
    lo: core::f64::consts::TAU,
    hi: f64::from_bits(core::f64::consts::TAU.to_bits() + 1),
};
/// `π/2`, between `FRAC_PI_2` and the next double.
pub(crate) const HALF_PI: Iv = Iv {
    lo: core::f64::consts::FRAC_PI_2,
    hi: f64::from_bits(core::f64::consts::FRAC_PI_2.to_bits() + 1),
};

impl Iv {
    pub(crate) const fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// The intersection of two intervals that both hold the same real.
    pub(crate) fn meet(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// `max |x|` over the interval.
    pub(crate) fn mag(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }

    /// `min |x|` over the interval: 0 when it holds 0.
    pub(crate) fn mig(self) -> f64 {
        if self.lo > 0.0 {
            self.lo
        } else if self.hi < 0.0 {
            -self.hi
        } else {
            0.0
        }
    }

    #[cfg(test)]
    pub(crate) fn contains(self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// A double inside the interval.
    pub(crate) fn mid(self) -> f64 {
        self.lo.midpoint(self.hi).clamp(self.lo, self.hi)
    }

    /// `{x² : x in self}`.
    #[expect(
        clippy::float_cmp,
        reason = "one product when both ends square to the same value"
    )]
    pub(crate) fn sq(self) -> Self {
        let (a, b) = (self.mig(), self.mag());
        if a == b {
            return product(a, a);
        }
        Self {
            lo: product(a, a).lo.max(0.0),
            hi: product(b, b).hi,
        }
    }

    /// `√x` of the interval's non-negative part; the real it holds is known to be non-negative.
    pub(crate) fn sqrt(self) -> Self {
        Self {
            lo: root(self.lo.max(0.0)).lo.max(0.0),
            hi: root(self.hi.max(0.0)).hi,
        }
    }

    /// `max(x, 0)`.
    pub(crate) fn max0(self) -> Self {
        Self {
            lo: self.lo.max(0.0),
            hi: self.hi.max(0.0),
        }
    }

    /// `|x|`.
    pub(crate) fn abs(self) -> Self {
        Self {
            lo: self.mig(),
            hi: self.mag(),
        }
    }
}

/// Below this magnitude an error-free transform may lose its exactness to underflow, and the
/// rounding direction is not read from it: `2⁻⁹⁰⁰`.
const EXACT_FLOOR: f64 = f64::from_bits(0x07B0_0000_0000_0000);

/// The exact value `r + e`, of a rounded `r` and its residual's sign `e`, between doubles: `r`
/// itself on the side the residual says the value is not on. An unknown residual (`None`, NaN or an
/// overflowed `r`) widens to both neighbours.
#[inline]
fn bracket(r: f64, residual: Option<f64>) -> Iv {
    if planted() {
        return Iv::point(r);
    }
    if !r.is_finite() {
        return Iv {
            lo: down(r),
            hi: up(r),
        };
    }
    match residual {
        Some(0.0) => Iv::point(r),
        Some(e) if e > 0.0 => Iv { lo: r, hi: up(r) },
        Some(e) if e < 0.0 => Iv { lo: down(r), hi: r },
        _ => Iv {
            lo: down(r),
            hi: up(r),
        },
    }
}

/// `a + b` between doubles: Knuth's two-sum residual is exact while the sum is finite.
#[inline]
fn sum(a: f64, b: f64) -> Iv {
    let s = a + b;
    let bb = s - a;
    bracket(s, Some((a - (s - bb)) + (b - bb)))
}

/// `a·b` between doubles: `fma(a, b, −p)` is the exact residual unless the product underflows.
#[inline]
fn product(a: f64, b: f64) -> Iv {
    let p = a * b;
    if a == 0.0 || b == 0.0 {
        return Iv::point(p);
    }
    let exact = p.abs() >= EXACT_FLOOR;
    bracket(p, exact.then(|| a.mul_add(b, -p)))
}

/// `a/b` between doubles, `b ≠ 0`: for `q` the rounded quotient, `a − q·b` is a double and its
/// sign times `b`'s is the side, away from underflow.
#[inline]
fn quotient(a: f64, b: f64) -> Iv {
    let q = a / b;
    if a == 0.0 && b.is_finite() {
        return Iv::point(q);
    }
    let exact = a.abs() >= EXACT_FLOOR && q.abs() >= EXACT_FLOOR && b.is_finite();
    bracket(
        q,
        exact.then(|| {
            let r = (-q).mul_add(b, a);
            if b > 0.0 { r } else { -r }
        }),
    )
}

/// `√a` between doubles, `a ≥ 0`: `a − s²` is a double, its sign the side, away from underflow.
#[inline]
fn root(a: f64) -> Iv {
    let s = libm::sqrt(a);
    if a == 0.0 {
        return Iv::point(s);
    }
    let exact = a >= EXACT_FLOOR;
    bracket(s, exact.then(|| (-s).mul_add(s, a)))
}

impl core::ops::Add for Iv {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            lo: sum(self.lo, o.lo).lo,
            hi: sum(self.hi, o.hi).hi,
        }
    }
}

impl core::ops::Sub for Iv {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self + (-o)
    }
}

impl core::ops::Neg for Iv {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

/// The hull of `f` at the four corners, each between doubles; once for two points.
#[expect(
    clippy::float_cmp,
    reason = "a point interval has equal ends, not nearly equal ones"
)]
fn corners(a: Iv, b: Iv, f: impl Fn(f64, f64) -> Iv) -> Iv {
    if a.lo == a.hi && b.lo == b.hi {
        return f(a.lo, b.lo);
    }
    let c = [f(a.lo, b.lo), f(a.lo, b.hi), f(a.hi, b.lo), f(a.hi, b.hi)];
    Iv {
        lo: c[0].lo.min(c[1].lo).min(c[2].lo).min(c[3].lo),
        hi: c[0].hi.max(c[1].hi).max(c[2].hi).max(c[3].hi),
    }
}

impl core::ops::Mul for Iv {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        corners(self, o, product)
    }
}

impl core::ops::Div for Iv {
    type Output = Self;
    /// The whole line when the divisor may be 0.
    fn div(self, o: Self) -> Self {
        if o.lo <= 0.0 && o.hi >= 0.0 {
            return Self {
                lo: f64::NEG_INFINITY,
                hi: f64::INFINITY,
            };
        }
        corners(self, o, quotient)
    }
}

/// An interval that holds `f(x)` when `r` is libm's answer within `c` ulp of it (H-libm), `c ≤ 2`.
pub(crate) fn libm_iv(r: f64, c: f64) -> Iv {
    if planted() {
        return Iv::point(r);
    }
    let delta = up(up(r.abs() * ((c + TWO_M40) * TWO_M52)) + TINY);
    Iv {
        lo: down(r - delta),
        hi: up(r + delta),
    }
}

/// `sin` over an interval, under H-libm.
pub(crate) fn sin(x: Iv) -> Iv {
    lipschitz(
        x,
        |t| libm_iv(libm::sin(t), H_SIN),
        |t| libm_iv(libm::cos(t), H_COS),
    )
}

/// `cos` over an interval, under H-libm.
pub(crate) fn cos(x: Iv) -> Iv {
    lipschitz(
        x,
        |t| libm_iv(libm::cos(t), H_COS),
        |t| -libm_iv(libm::sin(t), H_SIN),
    )
}

/// A 1-Lipschitz function bounded by 1, from its enclosures at the two ends: between them where its
/// derivative, `slope`, keeps one sign over an interval shorter than π (whose zeros, π apart, would
/// change it), and otherwise within the interval's width of each.
#[expect(
    clippy::float_cmp,
    reason = "a point interval has equal ends, not nearly equal ones"
)]
fn lipschitz(x: Iv, at: impl Fn(f64) -> Iv, slope: impl Fn(f64) -> Iv) -> Iv {
    let unit = Iv { lo: -1.0, hi: 1.0 };
    if x.lo == x.hi {
        return at(x.lo).meet(unit);
    }
    let w = up(x.hi - x.lo);
    let (a, b) = (at(x.lo), at(x.hi));
    if w < 3.0 {
        let (sa, sb) = (slope(x.lo), slope(x.hi));
        if sa.lo > 0.0 && sb.lo > 0.0 {
            return Iv { lo: a.lo, hi: b.hi }.meet(unit);
        }
        if sa.hi < 0.0 && sb.hi < 0.0 {
            return Iv { lo: b.lo, hi: a.hi }.meet(unit);
        }
    }
    let widen = |e: Iv| Iv {
        lo: down(e.lo - w),
        hi: up(e.hi + w),
    };
    widen(a).meet(widen(b)).meet(unit)
}

/// `atan2(y, x)` at doubles not both zero, under H-libm; a zero `y` is taken as `+0`, so the
/// negative `x` axis is `π`. Outside the ratios libm's row covers, `atan t` is enclosed by its
/// series: within `|t|³/3` of `t`.
pub(crate) fn atan2_pt(y: f64, x: f64) -> Iv {
    let full = Iv {
        lo: -PI.hi,
        hi: PI.hi,
    };
    if y == 0.0 {
        return if x > 0.0 {
            Iv::point(0.0)
        } else if x < 0.0 {
            PI
        } else {
            full
        };
    }
    if x == 0.0 {
        return if y > 0.0 { HALF_PI } else { -HALF_PI };
    }
    if !(x.is_finite() && y.is_finite()) {
        return full;
    }
    let series = |t: Iv| {
        let tail = up(t.mag() * SMALL_RATIO * SMALL_RATIO);
        Iv {
            lo: down(t.lo - tail),
            hi: up(t.hi + tail),
        }
    };
    let ratio = Iv::point(y) / Iv::point(x);
    if ratio.mag() <= SMALL_RATIO {
        let a = series(ratio);
        return if x > 0.0 {
            a
        } else if y > 0.0 {
            PI + a
        } else {
            a - PI
        };
    }
    let inverse = Iv::point(x) / Iv::point(y);
    if inverse.mag() <= SMALL_RATIO {
        let b = series(inverse);
        return if y > 0.0 { HALF_PI - b } else { -HALF_PI - b };
    }
    libm_iv(libm::atan2(y, x), H_ATAN2)
}

/// `atan2` over a box that holds no point of the closed negative `x` axis's branch cut and not
/// the origin: `x > 0` throughout, or `y ≥ 0` throughout. On such a box the angle is monotone in
/// each coordinate, so its extremes are at corners. `None` for any other box.
pub(crate) fn atan2(y: Iv, x: Iv) -> Option<Iv> {
    let (lo, hi) = if x.lo > 0.0 {
        // Increasing in y; decreasing in x where y > 0, increasing where y < 0.
        let lo = atan2_pt(y.lo, if y.lo >= 0.0 { x.hi } else { x.lo });
        let hi = atan2_pt(y.hi, if y.hi > 0.0 { x.lo } else { x.hi });
        (lo, hi)
    } else if y.lo >= 0.0 {
        if y.lo == 0.0 && x.lo <= 0.0 && x.hi >= 0.0 {
            // The origin is in the box.
            return None;
        }
        // Decreasing in x; increasing in y where x > 0, decreasing where x < 0.
        let lo = atan2_pt(if x.hi > 0.0 { y.lo } else { y.hi }, x.hi);
        let hi = atan2_pt(if x.lo > 0.0 { y.hi } else { y.lo }, x.lo);
        (lo, hi)
    } else {
        return None;
    };
    Some(Iv {
        lo: lo.lo,
        hi: hi.hi,
    })
}

/// `atan2(y, x)` for `y, x ≥ 0` (the first quadrant), the origin allowed: `[0, π/2]` there.
pub(crate) fn atan2_first_quadrant(y: Iv, x: Iv) -> Iv {
    let (y, x) = (y.max0(), x.max0());
    let lo = if y.lo == 0.0 {
        0.0
    } else {
        atan2_pt(y.lo, x.hi).lo.max(0.0)
    };
    let hi = if x.lo == 0.0 {
        HALF_PI.hi
    } else {
        atan2_pt(y.hi, x.lo).hi.min(HALF_PI.hi)
    };
    Iv { lo, hi }
}

/// The angle of a box that may straddle the branch cut, as a centre and a radius on the circle:
/// every point of the box is within `radius` of `centre`, mod 2π. `None` when the box may hold the
/// origin. The angle moves at most `|Δp|/r` along a segment at distance at least `r` from the origin.
pub(crate) fn angle_ball(y: Iv, x: Iv) -> Option<(f64, f64)> {
    let r_lo = (Iv::point(x.mig()).sq() + Iv::point(y.mig()).sq())
        .sqrt()
        .lo;
    if r_lo <= 0.0 {
        return None;
    }
    let (xm, ym) = (x.mid(), y.mid());
    let at = atan2_pt(ym, xm);
    let dx = up(xm - x.lo).max(up(x.hi - xm));
    let dy = up(ym - y.lo).max(up(y.hi - ym));
    let reach = (Iv::point(dx).sq() + Iv::point(dy).sq()).sqrt().hi;
    let centre = at.mid();
    let radius = up(up(reach / r_lo) + up(at.hi - at.lo));
    Some((centre, radius))
}

/// An upper bound of the distance on the circle from the reals of `d` to 0: `min_k |d − 2πk|`, for
/// `|d| ≤ 4π`.
pub(crate) fn circle_distance(d: Iv) -> f64 {
    let mut best = d.mag();
    for k in [-2.0, -1.0, 1.0, 2.0] {
        let shifted = d - Iv::point(k) * TAU;
        best = best.min(shifted.mag());
    }
    best
}

/// `acos` over an interval, under H-libm. The real it holds is a cosine, so the interval is first
/// cut to `[−1, 1]`; `acos` decreases there.
pub(crate) fn acos(c: Iv) -> Iv {
    let lo_arg = c.lo.clamp(-1.0, 1.0);
    let hi_arg = c.hi.clamp(-1.0, 1.0);
    Iv {
        lo: libm_iv(libm::acos(hi_arg), H_ACOS).lo.max(0.0),
        hi: libm_iv(libm::acos(lo_arg), H_ACOS).hi.min(PI.hi),
    }
}

/// A vector of intervals: a box that holds a vector of reals.
#[derive(Clone, Copy, Debug)]
pub(crate) struct IvVec {
    pub(crate) x: Iv,
    pub(crate) y: Iv,
    pub(crate) z: Iv,
}

impl IvVec {
    pub(crate) const fn of(v: Vec3) -> Self {
        Self {
            x: Iv::point(v.x),
            y: Iv::point(v.y),
            z: Iv::point(v.z),
        }
    }

    pub(crate) fn dot(self, o: Self) -> Iv {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub(crate) fn cross(self, o: Self) -> Self {
        Self {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    pub(crate) fn add(self, o: Self) -> Self {
        Self {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }

    pub(crate) fn sub(self, o: Self) -> Self {
        Self {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }

    pub(crate) fn scale(self, k: Iv) -> Self {
        Self {
            x: self.x * k,
            y: self.y * k,
            z: self.z * k,
        }
    }

    /// `|v|²`, each square taken as one non-negative interval.
    pub(crate) fn norm2(self) -> Iv {
        self.x.sq() + self.y.sq() + self.z.sq()
    }

    pub(crate) fn norm(self) -> Iv {
        self.norm2().sqrt()
    }

    /// `v/|v|`; the whole space when `|v|` may be 0.
    pub(crate) fn unit(self) -> Self {
        let inv = Iv::point(1.0) / self.norm();
        self.scale(inv)
    }

    /// An upper bound of `|p − w|` over every `w` in the box.
    pub(crate) fn reach(self, p: Vec3) -> f64 {
        let far = |c: Iv, q: f64| up(q - c.lo).max(up(c.hi - q)).max(0.0);
        let (a, b, c) = (far(self.x, p.x), far(self.y, p.y), far(self.z, p.z));
        (Iv::point(a).sq() + Iv::point(b).sq() + Iv::point(c).sq())
            .sqrt()
            .hi
    }
}

/// `2⁻⁴⁴⁹`: from here up a computed length's square is at least `2⁻⁸⁹⁸`, where three underflowed
/// squares cost a relative `2⁻¹⁷³` at most.
const LENGTH_FLOOR: f64 = f64::from_bits(0x23E0_0000_0000_0000);

/// The length of the double vector `v` as an interval, from `n = √((x² + y²) + z²)` in doubles:
/// three products and two sums of non-negative terms, each within `u` of its result, and the square
/// root, put `n` within `2.6u` of the true length while `n ≥ 2⁻⁴⁴⁹`, so the length is in
/// `[n(1 − 4u), n(1 + 4u)]` (`derivations/outward_rounding.mac`, `norm_relative`). Cheaper than
/// [`IvVec::norm`] for a point; below the floor that is the answer.
pub(crate) fn length(v: Vec3) -> Iv {
    let n = libm::sqrt((v.x * v.x + v.y * v.y) + v.z * v.z);
    if !(n >= LENGTH_FLOOR && n.is_finite()) {
        return IvVec::of(v).norm();
    }
    Iv {
        lo: down(n * (1.0 - 2.0 * f64::EPSILON)),
        hi: up(n * (1.0 + 2.0 * f64::EPSILON)),
    }
}

/// An upper bound of the angle between two unit vectors whose chord is at most `chord`:
/// `2·asin(c/2) ≤ (π/2)·c`, since `asin` is convex on `[0, 1]`.
pub(crate) fn angle_of_chord(chord: f64) -> f64 {
    if chord >= 2.0 {
        PI.hi
    } else {
        up(chord * HALF_PI.hi).min(PI.hi)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "a test fails loudly on a refused box")]
    use super::*;

    #[test]
    fn the_constants_hold_their_reals() {
        // π = 3.14159265358979323846..., the double PI = 3.14159265358979311600...
        assert_eq!(PI.lo.next_up(), PI.hi);
        assert_eq!(TAU.lo.next_up(), TAU.hi);
        assert_eq!(HALF_PI.lo.next_up(), HALF_PI.hi);
        assert_eq!(PI.lo, 884_279_719_003_555.0 / 281_474_976_710_656.0);
        // `powi` of a subnormal result is not exact without constant folding: 2⁻¹⁰²² · 2⁻⁵⁰ is.
        assert_eq!(TINY, f64::MIN_POSITIVE * 2.0_f64.powi(-50));
        assert_eq!(SMALL_RATIO, 2.0_f64.powi(-50));
        assert_eq!(TWO_M40, 2.0_f64.powi(-40));
        assert_eq!(EXACT_FLOOR, 2.0_f64.powi(-900));
        assert_eq!(LENGTH_FLOOR, 2.0_f64.powi(-449));
    }

    /// The constants are the rows of `derivations/h_libm.txt`, which verify.sh §9 measures.
    #[test]
    fn the_h_libm_constants_are_the_table_rows() {
        let table = include_str!("../../../derivations/h_libm.txt");
        for (row, constant) in [
            ("sin_all", H_SIN),
            ("cos_all", H_COS),
            ("atan2_all", H_ATAN2),
            ("acos_all", H_ACOS),
        ] {
            let listed = table
                .lines()
                .find(|l| l.split_whitespace().next() == Some(row))
                .and_then(|l| l.split_whitespace().nth(2))
                .map(str::parse::<f64>);
            assert_eq!(listed, Some(Ok(constant)), "{row}");
        }
    }

    /// A computed length holds the true one: `√2`'s double is above √2, and a vector of tiny
    /// components falls back to the interval norm.
    #[test]
    fn lengths_hold_the_true_length() {
        let l = length(Vec3::new(1.0, 1.0, 0.0));
        assert!(l.lo < core::f64::consts::SQRT_2 && l.hi >= core::f64::consts::SQRT_2);
        assert!(l.hi - l.lo < 2e-15);
        let tiny = length(Vec3::new(1e-300, 0.0, 0.0));
        assert!(tiny.lo <= 1e-300 && 1e-300 <= tiny.hi);
    }

    #[test]
    fn each_operation_holds_its_exact_result() {
        // Each end is the directed rounding itself. The doubles 0.1 and 0.2 sum exactly to
        // 0.30000000000000001665..., between the double 0.29999999999999998889... and the rounded
        // 0.30000000000000004440....
        let s = Iv::point(0.1) + Iv::point(0.2);
        assert_eq!(
            s,
            Iv {
                lo: 0.3,
                hi: 0.300_000_000_000_000_04
            }
        );
        assert_eq!(Iv::point(3.0) * Iv::point(-2.0), Iv::point(-6.0));
        // The double 1/3 = 0.33333333333333331482... is below 1/3; SQRT_2 = 1.41421356237309514547...
        // is above √2 = 1.41421356237309504880....
        let third = 1.0_f64 / 3.0;
        let q = Iv::point(1.0) / Iv::point(3.0);
        assert_eq!(
            q,
            Iv {
                lo: third,
                hi: third.next_up()
            }
        );
        let r2 = core::f64::consts::SQRT_2;
        assert_eq!(
            Iv::point(2.0).sqrt(),
            Iv {
                lo: r2.next_down(),
                hi: r2
            }
        );
        assert_eq!(Iv { lo: -1.0, hi: 2.0 }.sq(), Iv { lo: 0.0, hi: 4.0 });
        // An underflowed product has no exact residual and widens to both neighbours of 0.
        let under = Iv::point(1e-200) * Iv::point(1e-200);
        assert!(under.lo < 0.0 && under.hi > 0.0 && under.hi < 1e-320);
        let neg = Iv::point(-1.0) / Iv::point(3.0);
        assert_eq!(
            neg,
            Iv {
                lo: (-third).next_down(),
                hi: -third
            }
        );
        let whole = Iv::point(1.0) / Iv { lo: -1.0, hi: 1.0 };
        assert!(whole.lo.is_infinite() && whole.hi.is_infinite());
    }

    /// Rule 16's planted defect: under [`NEAREST`] every operation rounds to nearest and nothing
    /// widens, and the results above miss their exact values. S2's planted run refutes the same
    /// defect in the k-fold areas and α (`derivations/verify.sh`); there the a-posteriori margins
    /// alone can hide it, here nothing does.
    #[test]
    fn the_planted_round_to_nearest_misses_exact_values() {
        NEAREST.with(|p| p.set(true));
        let s = Iv::point(0.1) + Iv::point(0.2);
        let r2 = Iv::point(2.0).sqrt();
        let third = Iv::point(1.0) / Iv::point(3.0);
        NEAREST.with(|p| p.set(false));
        // 0.30000000000000001665... (exact) is below the point; √2 below SQRT_2; 1/3 above its double.
        assert_eq!(s, Iv::point(0.300_000_000_000_000_04));
        assert_eq!(r2, Iv::point(core::f64::consts::SQRT_2));
        assert_eq!(third, Iv::point(1.0 / 3.0));
    }

    #[test]
    fn libm_intervals_hold_the_exact_values_at_known_points() {
        // sin(π/6 as a double) = 0.49999999999999994448884876874217... (the double below 1/2).
        let s = sin(Iv::point(core::f64::consts::FRAC_PI_6));
        assert!(s.contains(0.5 - 5.551_115_123_125_783e-17));
        assert!(s.hi - s.lo < 4e-16);
        // atan2(1, 1) = π/4 exactly.
        assert!(atan2_pt(1.0, 1.0).contains(core::f64::consts::FRAC_PI_4));
        assert_eq!(atan2_pt(0.0, -1.0), PI);
        // A ratio below 2^-50: the series, 1e-300 within its interval.
        let tiny = atan2_pt(1e-300, 1.0);
        assert!(tiny.contains(1e-300) && tiny.hi - tiny.lo < 1e-314);
        let left = atan2_pt(1e-300, -1.0);
        assert!(left.lo < PI.lo && left.hi >= PI.lo);
        let steep = atan2_pt(-1.0, 1e-300);
        assert!(steep.contains(-core::f64::consts::FRAC_PI_2));
        // acos clips to [−1, 1] and holds 0 and π at the ends.
        assert_eq!(acos(Iv { lo: 0.9, hi: 1.5 }).lo, 0.0);
        assert!(acos(Iv::point(-1.0)).contains(core::f64::consts::PI));
    }

    #[test]
    fn boxes_give_angles_at_their_corners() {
        let quarter = atan2(Iv { lo: 1.0, hi: 2.0 }, Iv { lo: 1.0, hi: 2.0 }).unwrap();
        assert!(quarter.lo < libm::atan2(1.0, 2.0) && quarter.hi > libm::atan2(2.0, 1.0));
        let left = atan2(Iv { lo: 0.0, hi: 1.0 }, Iv { lo: -2.0, hi: -1.0 }).unwrap();
        assert!(left.hi >= PI.hi - 1e-15 && left.lo < libm::atan2(1.0, -1.0));
        assert!(atan2(Iv { lo: -1.0, hi: 1.0 }, Iv { lo: -1.0, hi: 1.0 }).is_none());
        let q = atan2_first_quadrant(Iv { lo: 0.0, hi: 1e-9 }, Iv { lo: 0.0, hi: 1.0 });
        assert_eq!(
            q,
            Iv {
                lo: 0.0,
                hi: HALF_PI.hi
            }
        );
        let (c, r) = angle_ball(
            Iv {
                lo: -1e-3,
                hi: 1e-3,
            },
            Iv {
                lo: -1.0,
                hi: -0.999,
            },
        )
        .unwrap();
        assert!(circle_distance(Iv::point(c) - PI) <= r && r < 2e-3);
        assert!(angle_ball(Iv { lo: -1.0, hi: 1.0 }, Iv { lo: -1.0, hi: 1.0 }).is_none());
    }
}
