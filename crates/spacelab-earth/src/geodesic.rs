//! Geodesics on the WGS-84 ellipsoid: the direct and the inverse problem, to round-off.
//!
//! Written from the mathematics of C. F. F. Karney, "Algorithms for geodesics", J. Geodesy 87
//! (2013) 43–55 (doi:10.1007/s00190-012-0578-z): the auxiliary sphere, the Fourier series of the
//! distance and longitude integrals to sixth order in the third flattening and in `ε`, and the inverse
//! problem as Newton's method on the azimuth, started from Helmert's astroid near the antipode. The
//! coefficients are derived and the code is ours; nothing is ported from `GeographicLib`, which is
//! a comparator only.
//!
//! The gates: `derivations/ellipsoid_geodesic.mac` derives the integrands from the metric and
//! Clairaut's relation, expands the series, checks them against the elliptic-integral form at 50
//! digits (G1) and emits `geodesic_series.rs`; `derivations/ellipsoid_geodesic.gp` derives the same
//! coefficients by another route and solves reference geodesics at 50 digits by quadrature and root
//! finding. `tests/geodesic.rs` holds this code against those references (G2), against closed forms
//! — the equator, the quarter meridian, the poles (G3) — and against its own inverse (G4).

// The bindings are the paper's symbols: `sb1, cb1` are sin β1 and cos β1, `ss12, cs12` sin σ12 and
// cos σ12, and so on, so pairs differ in one letter by construction; the astroid's `p … w` are
// Vermeille's.
#![expect(
    clippy::similar_names,
    clippy::many_single_char_names,
    reason = "the bindings are the paper's symbols"
)]

use core::f64::consts::{PI, TAU};
use core::fmt;

use spacelab_core::{Angle, Length};

use crate::geodesic_series as series;
use crate::wgs84::{EQUATORIAL_RADIUS, FLATTENING, POLAR_RADIUS};

/// `1 − f`.
const F1: f64 = 1.0 - FLATTENING;
/// `e² = f(2 − f)`.
const E2: f64 = FLATTENING * (2.0 - FLATTENING);
/// `e′² = e²/(1 − e²)`, and `1 − e² = (1 − f)²` without the subtraction.
const EP2: f64 = E2 / (F1 * F1);
/// The third flattening `n = f/(2 − f)`, the variable of the `I3` coefficients.
const N: f64 = FLATTENING / (2.0 - FLATTENING);

/// Horner's rule over ascending coefficients.
const fn horner(c: &[f64], x: f64) -> f64 {
    let mut i = c.len();
    let mut r = 0.0;
    while i > 0 {
        i -= 1;
        r = r * x + c[i];
    }
    r
}

/// `A3`'s coefficients of `ε⁰ … ε⁵`, evaluated at WGS-84's `n` once, at compile time.
const A3X: [f64; 6] = {
    let mut out = [0.0; 6];
    let mut j = 0;
    while j < 6 {
        out[j] = horner(series::A3[j], N);
        j += 1;
    }
    out
};

/// `C3ₗ = εˡ Σ_k C3X[l − 1][k] εᵏ`, `l = 1 … 5`, evaluated at WGS-84's `n`.
const C3X: [[f64; 5]; 5] = {
    let mut out = [[0.0; 5]; 5];
    let mut l = 0;
    while l < 5 {
        let mut k = 0;
        while k < series::C3[l].len() {
            out[l][k] = horner(series::C3[l][k], N);
            k += 1;
        }
        l += 1;
    }
    out
};

/// Inverse iterations before the bracket is accepted as the answer. Newton takes at most 6 after
/// either start (measured over 40 000 pairs, half of them nearly antipodal); bisection halves a
/// bracket of width π, so 64 more reach a double's resolution.
const MAX_ITERATIONS: usize = 100;

/// A Newton correction of α1 this small (2⁻⁴⁰ rad) is the last one. Measured against iterating to
/// the noise floor, a threshold of 3·10⁻¹⁰ already agrees to one ulp of s12 and 3·10⁻⁹ does not, so
/// 2⁻⁴⁰ leaves a factor of 300 in the step and 10⁵ in its quadratic remainder.
const NEWTON_LAST: f64 = 4096.0 * f64::EPSILON;

/// Where the astroid start is used instead of the spherical one, in the astroid's units (Karney
/// eq. 53): `x = (λ12 − π)/(f π A3 cos β1)` and `y = (β1 + β2)/(f π A3 cos² β1)`. The box is the
/// one of those tried (−3 … −8 on each axis) that took the fewest iterations on nearly antipodal
/// pairs.
const ASTROID_X: f64 = -6.0;
const ASTROID_Y: f64 = -6.0;

/// A point on the surface of the ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePoint {
    /// Geodetic latitude, positive north, in `[−π/2, π/2]`.
    pub latitude: Angle,
    /// Longitude, positive east. Any finite value; it is reduced.
    pub longitude: Angle,
}

/// The solution of the inverse problem: the shortest geodesic between two points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicInverse {
    /// The length of the geodesic.
    pub distance: Length,
    /// Its azimuth at the first point, clockwise from north, in `(−π, π]`.
    pub azimuth1: Angle,
    /// Its forward azimuth at the second point (not the back azimuth), in `(−π, π]`.
    pub azimuth2: Angle,
}

/// The solution of the direct problem: where a geodesic of given start, azimuth and length ends.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicDirect {
    /// The end point; its longitude is in `(−π, π]`.
    pub point: SurfacePoint,
    /// The forward azimuth at the end point, in `(−π, π]`.
    pub azimuth: Angle,
}

/// Why a geodesic problem has no answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeodesicError {
    /// A latitude, longitude, azimuth or distance is NaN or infinite.
    NonFinite,
    /// A latitude lies outside `[−π/2, π/2]`.
    LatitudeOutOfRange,
}

impl fmt::Display for GeodesicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NonFinite => write!(f, "a geodesic input is not finite"),
            Self::LatitudeOutOfRange => write!(f, "a latitude lies outside [-pi/2, pi/2]"),
        }
    }
}

impl core::error::Error for GeodesicError {}

fn check_point(p: SurfacePoint) -> Result<(), GeodesicError> {
    let (lat, lon) = (p.latitude.radians(), p.longitude.radians());
    if !lat.is_finite() || !lon.is_finite() {
        return Err(GeodesicError::NonFinite);
    }
    if lat.abs() > core::f64::consts::FRAC_PI_2 {
        return Err(GeodesicError::LatitudeOutOfRange);
    }
    Ok(())
}

/// The series coefficients of one geodesic, from its `ε`.
struct Coefficients {
    /// `A1`.
    a1: f64,
    /// `C1ₗ`, index `l` (`[0]` unused).
    c1: [f64; 7],
    /// `C1′ₗ`.
    c1p: [f64; 7],
    /// `A2`.
    a2: f64,
    /// `C2ₗ`.
    c2: [f64; 7],
    /// `A3`.
    a3: f64,
    /// `C3ₗ`, `l = 1 … 5`.
    c3: [f64; 6],
}

impl Coefficients {
    /// From `k² = e′² cos² α0`.
    fn new(k2: f64) -> Self {
        // (√(1 + k²) − 1)/(√(1 + k²) + 1) without the subtraction (ellipsoid_geodesic.mac, §2).
        let eps = k2 / (2.0 * (1.0 + libm::sqrt(1.0 + k2)) + k2);
        let eps2 = eps * eps;
        let mut c1 = [0.0; 7];
        let mut c1p = [0.0; 7];
        let mut c2 = [0.0; 7];
        let mut c3 = [0.0; 6];
        let mut epsl = 1.0;
        for l in 1..=6 {
            epsl *= eps;
            c1[l] = epsl * horner(series::C1[l - 1], eps2);
            c1p[l] = epsl * horner(series::C1P[l - 1], eps2);
            c2[l] = epsl * horner(series::C2[l - 1], eps2);
            if l <= 5 {
                c3[l] = epsl * horner(&C3X[l - 1][..=5 - l], eps);
            }
        }
        Self {
            a1: horner(&series::A1, eps2) / (1.0 - eps),
            c1,
            c1p,
            a2: horner(&series::A2, eps2) * (1.0 - eps),
            c2,
            a3: horner(&A3X, eps),
            c3,
        }
    }
}

/// `Σ_{l ≥ 1} c[l] sin 2lσ` by Clenshaw's recurrence, from `sin σ`, `cos σ`.
fn sin_series(c: &[f64], s: f64, co: f64) -> f64 {
    let two_cos2 = 2.0 * (co - s) * (co + s);
    let (mut b1, mut b2) = (0.0, 0.0);
    for &ck in c[1..].iter().rev() {
        let b0 = ck + two_cos2 * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    b1 * 2.0 * s * co
}

/// `(y, x)` scaled to a unit vector; the zero vector becomes `(0, 1)`, the angle 0.
fn unit(y: f64, x: f64) -> (f64, f64) {
    let h = libm::hypot(y, x);
    if h == 0.0 { (0.0, 1.0) } else { (y / h, x / h) }
}

/// The reduced latitude `tan β = (1 − f) tan φ`, as `(sin β, cos β)`.
fn reduced(lat: f64) -> (f64, f64) {
    let (s, c) = libm::sincos(lat);
    unit(F1 * s, c)
}

/// `x + y` reduced to `(−π, π]`, with 2π carried to twice double precision so that the reduction
/// adds one rounding, not the 2.4e-16 by which `TAU` misses 2π.
fn reduce_sum(x: f64, y: f64) -> f64 {
    // Knuth's two-sum: s + e = x + y exactly.
    let s = x + y;
    let yv = s - x;
    let e = (x - (s - yv)) + (y - yv);
    let k = libm::round(s / TAU);
    let mut r = libm::fma(-k, TAU, s) - k * series::TAU_LO + e;
    if r > PI {
        r = (r - TAU) - series::TAU_LO;
    } else if r < -PI {
        // -PI itself is the double just above -π, inside the range.
        r = (r + TAU) + series::TAU_LO;
    }
    r
}

/// The azimuth with sine `s` and cosine `c`, in `(−π, π]`: a sine of `−0` is read as `+0`, so due
/// south is `π`, never `−π`.
fn azimuth(s: f64, c: f64) -> Angle {
    Angle::from_radians(libm::atan2(if s == 0.0 { 0.0 } else { s }, c))
}

/// A non-negative quantity that rounding may have pushed to `−0` or below; `+0` in its place.
fn non_negative(x: f64) -> f64 {
    if x <= 0.0 { 0.0 } else { x }
}

/// A geodesic from a point at a given azimuth, ready to give any point along it.
///
/// The series coefficients depend only on the geodesic, so a line evaluates them once; each
/// [`GeodesicLine::position`] then costs one reversion of the distance series, Clenshaw sums and a
/// handful of trigonometric calls. This is the shape for densifying a polygon edge or a swath
/// boundary.
#[derive(Debug, Clone, Copy)]
pub struct GeodesicLine {
    lon1: f64,
    sin_alpha0: f64,
    cos_alpha0: f64,
    sin_sigma1: f64,
    cos_sigma1: f64,
    sin_omega1: f64,
    cos_omega1: f64,
    sin_tau1: f64,
    cos_tau1: f64,
    /// `B1(σ1)`, the periodic part of the distance at the start.
    b11: f64,
    /// `B3(σ1)`.
    b31: f64,
    /// `b·A1`, metres per unit of `τ`.
    b_a1: f64,
    /// `−f sin α0 A3`, the longitude correction per unit of `σ`.
    a3c: f64,
    c1p: [f64; 7],
    c3: [f64; 6],
}

impl GeodesicLine {
    /// The geodesic leaving `from` with forward azimuth `azimuth` (clockwise from north).
    ///
    /// # Errors
    ///
    /// [`GeodesicError::NonFinite`] or [`GeodesicError::LatitudeOutOfRange`] for an input outside
    /// the domain.
    pub fn new(from: SurfacePoint, azimuth: Angle) -> Result<Self, GeodesicError> {
        check_point(from)?;
        if !azimuth.radians().is_finite() {
            return Err(GeodesicError::NonFinite);
        }
        let (sb1, cb1) = reduced(from.latitude.radians());
        let (sa1, ca1) = libm::sincos(azimuth.radians());
        // Clairaut, eq. (10): cos α0 as a hypotenuse, accurate where α0 is near π/2.
        let sin_alpha0 = sa1 * cb1;
        let cos_alpha0 = libm::hypot(ca1, sa1 * sb1);
        // Eq. (11); on the equator heading east the node is the start, σ1 = 0.
        let (sin_sigma1, cos_sigma1) = unit(sb1, ca1 * cb1);
        let coef = Coefficients::new(EP2 * cos_alpha0 * cos_alpha0);
        let b11 = sin_series(&coef.c1, sin_sigma1, cos_sigma1);
        let (sb, cb) = libm::sincos(b11);
        Ok(Self {
            lon1: from.longitude.radians(),
            sin_alpha0,
            cos_alpha0,
            sin_sigma1,
            cos_sigma1,
            // Eq. (12), unnormalised: only angle differences are taken.
            sin_omega1: sin_alpha0 * sin_sigma1,
            cos_omega1: cos_sigma1,
            sin_tau1: sin_sigma1 * cb + cos_sigma1 * sb,
            cos_tau1: cos_sigma1 * cb - sin_sigma1 * sb,
            b11,
            b31: sin_series(&coef.c3, sin_sigma1, cos_sigma1),
            b_a1: POLAR_RADIUS * coef.a1,
            a3c: -FLATTENING * sin_alpha0 * coef.a3,
            c1p: coef.c1p,
            c3: coef.c3,
        })
    }

    /// The point at `distance` along the line (negative goes backwards), and the azimuth there.
    ///
    /// # Errors
    ///
    /// [`GeodesicError::NonFinite`] for a NaN or infinite distance.
    pub fn position(&self, distance: Length) -> Result<GeodesicDirect, GeodesicError> {
        let s12 = distance.metres();
        if !s12.is_finite() {
            return Err(GeodesicError::NonFinite);
        }
        // s = b A1 τ (Karney §2), then σ from τ by the reversion (20).
        let tau12 = s12 / self.b_a1;
        let (st12, ct12) = libm::sincos(tau12);
        let st2 = self.sin_tau1 * ct12 + self.cos_tau1 * st12;
        let ct2 = self.cos_tau1 * ct12 - self.sin_tau1 * st12;
        let sig12 = tau12 + self.b11 + sin_series(&self.c1p, st2, ct2);
        let (ss12, cs12) = libm::sincos(sig12);
        let ss2 = self.sin_sigma1 * cs12 + self.cos_sigma1 * ss12;
        let cs2 = self.cos_sigma1 * cs12 - self.sin_sigma1 * ss12;
        // Eqs. (13), (14), (12) at the end.
        let sb2 = self.cos_alpha0 * ss2;
        let cb2 = libm::hypot(self.sin_alpha0, self.cos_alpha0 * cs2);
        let (so2, co2) = (self.sin_alpha0 * ss2, cs2);
        // ω12 modulo 2π, which is all the reduced longitude needs.
        let omg12 = libm::atan2(
            so2 * self.cos_omega1 - co2 * self.sin_omega1,
            co2 * self.cos_omega1 + so2 * self.sin_omega1,
        );
        // Eq. (8): λ = ω − f sin α0 I3(σ).
        let lam12 = omg12 + self.a3c * (sig12 + (sin_series(&self.c3, ss2, cs2) - self.b31));
        Ok(GeodesicDirect {
            point: SurfacePoint {
                latitude: Angle::from_radians(libm::atan2(sb2, F1 * cb2)),
                longitude: Angle::from_radians(reduce_sum(self.lon1, lam12)),
            },
            azimuth: azimuth(self.sin_alpha0, self.cos_alpha0 * cs2),
        })
    }

    /// Points at many distances along the line, into a caller-provided slice — the batch shape of
    /// this workspace (`plans/08-parallelism.md`). A non-finite distance yields `Err` in its own
    /// slot.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `distances`.
    pub fn positions_into(
        &self,
        distances: &[Length],
        out: &mut [Result<GeodesicDirect, GeodesicError>],
    ) {
        assert!(
            out.len() >= distances.len(),
            "output slice holds {} of {} distances",
            out.len(),
            distances.len()
        );
        for (d, slot) in distances.iter().zip(out.iter_mut()) {
            *slot = self.position(*d);
        }
    }
}

/// The direct problem: where the geodesic from `from` at `azimuth` ends after `distance`.
///
/// # Errors
///
/// [`GeodesicError::NonFinite`] or [`GeodesicError::LatitudeOutOfRange`] for an input outside the
/// domain.
pub fn geodesic_direct(
    from: SurfacePoint,
    azimuth: Angle,
    distance: Length,
) -> Result<GeodesicDirect, GeodesicError> {
    GeodesicLine::new(from, azimuth)?.position(distance)
}

/// The two ends of the problem in Karney's canonical order (eq. 44): `β1 ≤ 0`, `|β2| ≤ |β1|`,
/// `0 ≤ λ12 ≤ π`.
struct Canonical {
    sb1: f64,
    cb1: f64,
    sb2: f64,
    cb2: f64,
    /// `λ12`, and its sine and cosine.
    lam12: f64,
    sl12: f64,
    cl12: f64,
}

/// One evaluation of the hybrid problem (Karney §4): the geodesic leaving point 1 at `α1`, followed
/// to its first northward crossing of `β2`.
struct Hybrid {
    /// `λ12(α1) − λ12`.
    v: f64,
    /// `dλ12/dα1`.
    dv: f64,
    sig12: f64,
    ss1: f64,
    cs1: f64,
    ss2: f64,
    cs2: f64,
    sa2: f64,
    ca2: f64,
    coef: Coefficients,
}

impl Canonical {
    // `cb2 == cb1` is meant exactly: both come from `|φ|`, so equal or opposite latitudes give the
    // same bits.
    #[expect(
        clippy::float_cmp,
        reason = "equal or opposite latitudes give the same bits"
    )]
    fn hybrid(&self, sa1: f64, ca1: f64) -> Hybrid {
        let (sb1, cb1, sb2, cb2) = (self.sb1, self.cb1, self.sb2, self.cb2);
        let sa0 = sa1 * cb1;
        let ca0 = libm::hypot(ca1, sa1 * sb1);
        // Eq. (45), with cos²β2 − cos²β1 in whichever form cancels less; |cos α1| on the parallel
        // of opposite or equal latitude, where the formula is exact but the subtraction is not.
        let ca2 = if cb2 == cb1 {
            libm::fabs(ca1)
        } else {
            let gap = if cb1 < -sb1 {
                (cb2 - cb1) * (cb2 + cb1)
            } else {
                (sb1 - sb2) * (sb1 + sb2)
            };
            libm::sqrt(non_negative((ca1 * cb1) * (ca1 * cb1) + gap)) / cb2
        };
        let sa2 = sa0 / cb2;
        let (ss1, cs1) = unit(sb1, ca1 * cb1);
        let (ss2, cs2) = unit(sb2, ca2 * cb2);
        let sig12 = libm::atan2(non_negative(cs1 * ss2 - ss1 * cs2), cs1 * cs2 + ss1 * ss2);
        // ω12 − λ12 as one angle: (ω1, ω2) unnormalised, rotated by −λ12.
        let (so1, co1, so2, co2) = (sa0 * ss1, cs1, sa0 * ss2, cs2);
        let somg12 = non_negative(co1 * so2 - so1 * co2);
        let comg12 = co1 * co2 + so1 * so2;
        let eta = libm::atan2(
            somg12 * self.cl12 - comg12 * self.sl12,
            comg12 * self.cl12 + somg12 * self.sl12,
        );
        let k2 = EP2 * ca0 * ca0;
        let coef = Coefficients::new(k2);
        let b312 = sin_series(&coef.c3, ss2, cs2) - sin_series(&coef.c3, ss1, cs1);
        let v = eta - FLATTENING * sa0 * coef.a3 * (sig12 + b312);
        let dv = if ca2 == 0.0 {
            // Eq. (47): m12 and cos α2 vanish together; ellipsoid_geodesic.mac §8 takes the limit.
            -2.0 * F1 * libm::sqrt(1.0 + EP2 * sb1 * sb1) / sb1
        } else {
            // Eq. (46) with the reduced length (38): dλ12/dα1 = m12/(a cos α2 cos β2).
            let d1 = libm::sqrt(1.0 + k2 * ss1 * ss1);
            let d2 = libm::sqrt(1.0 + k2 * ss2 * ss2);
            let j = |s: f64, c: f64| {
                coef.a1 * sin_series(&coef.c1, s, c) - coef.a2 * sin_series(&coef.c2, s, c)
            };
            let j12 = (coef.a1 - coef.a2) * sig12 + (j(ss2, cs2) - j(ss1, cs1));
            let m12b = d2 * cs1 * ss2 - d1 * ss1 * cs2 - cs1 * cs2 * j12;
            F1 * m12b / (ca2 * cb2)
        };
        Hybrid {
            v,
            dv,
            sig12,
            ss1,
            cs1,
            ss2,
            cs2,
            sa2,
            ca2,
            coef,
        }
    }

    /// The first azimuth to try: Helmert's astroid near the antipode (Karney §5), elsewhere the
    /// great circle of the auxiliary sphere with `ω12 = λ12/w̄` (eqs. 48–49).
    fn start(&self) -> (f64, f64) {
        let (sb1, cb1, sb2, cb2) = (self.sb1, self.cb1, self.sb2, self.cb2);
        // sin(β1 + β2): the offset from the antipodal parallel.
        let sb12a = sb2 * cb1 + cb2 * sb1;
        let a3 = Coefficients::new(EP2 * sb1 * sb1).a3;
        let lamscale = FLATTENING * PI * a3 * cb1;
        let x = (self.lam12 - PI) / lamscale;
        let y = sb12a / (lamscale * cb1);
        if x >= ASTROID_X && y >= ASTROID_Y && x <= 0.0 && y <= 0.0 {
            if y == 0.0 && x * x <= 1.0 {
                // Eq. (57), the limit y → 0 inside the astroid: the branch with cos α1 ≤ 0, whose
                // first northward crossing of β2 = −β1 is at σ12 = π.
                return unit(-x, -libm::sqrt(1.0 - x * x));
            }
            let mu = astroid(x, y);
            // Eq. (56).
            return unit(-x / (1.0 + mu), y / mu);
        }
        let cbm = 0.5 * (cb1 + cb2);
        let wbar = libm::sqrt(1.0 - E2 * cbm * cbm);
        // Past π the great circle would leave westward, outside the bracket [0, π]; π itself
        // starts on the meridian through the pole the geodesic passes nearest.
        let (so12, co12) = libm::sincos((self.lam12 / wbar).min(PI));
        unit(cb2 * so12, cb1 * sb2 - sb1 * cb2 * co12)
    }
}

/// The positive root `μ` of Karney's quartic (55), `μ⁴ + 2μ³ + (1 − x² − y²)μ² − 2y²μ − y² = 0`,
/// for `y ≠ 0` or outside the unit circle: Vermeille's closed form of the same quartic, with its
/// trigonometric branch. `ellipsoid_geodesic.mac` §7 checks both branches at 50 digits.
fn astroid(x: f64, y: f64) -> f64 {
    let p = x * x;
    let q = y * y;
    let r = (p + q - 1.0) / 6.0;
    let s = p * q / 4.0;
    let r2 = r * r;
    let r3 = r * r2;
    let disc = s * (s + 2.0 * r3);
    let mut u = r;
    if disc >= 0.0 {
        let mut t3 = s + r3;
        // Add the root with the sign of t3, so that nothing cancels.
        t3 += if t3 < 0.0 {
            -libm::sqrt(disc)
        } else {
            libm::sqrt(disc)
        };
        let t = libm::cbrt(t3);
        u += t + if t == 0.0 { 0.0 } else { r2 / t };
    } else {
        let ang = libm::atan2(libm::sqrt(-disc), -(s + r3));
        u += 2.0 * r * libm::cos(ang / 3.0);
    }
    let v = libm::sqrt(u * u + q);
    let uv = if u < 0.0 { q / (v - u) } else { u + v };
    let w = (uv - q) / (2.0 * v);
    uv / (libm::sqrt(uv + w * w) + w)
}

/// The inverse problem: the shortest geodesic from `from` to `to` on WGS 84.
///
/// It is unique except between antipodal points and on the cut locus near the antipode, where it
/// has a mirror image of the same length; the one whose canonical form arrives heading north is
/// returned. Between coincident points the distance is 0 and both azimuths are 0, by convention.
///
/// # Errors
///
/// [`GeodesicError::NonFinite`] or [`GeodesicError::LatitudeOutOfRange`] for an input outside the
/// domain.
// Coincident points are the same two doubles; nothing near them is special.
#[expect(
    clippy::float_cmp,
    reason = "coincident points are the same two doubles"
)]
pub fn geodesic_inverse(
    from: SurfacePoint,
    to: SurfacePoint,
) -> Result<GeodesicInverse, GeodesicError> {
    check_point(from)?;
    check_point(to)?;
    let (mut lat1, mut lon1) = (from.latitude.radians(), from.longitude.radians());
    let (mut lat2, mut lon2) = (to.latitude.radians(), to.longitude.radians());
    // Canonical order. The swap first: it changes the sign of λ12, which the reflection in
    // longitude then makes non-negative.
    let swapped = lat1.abs() < lat2.abs();
    if swapped {
        core::mem::swap(&mut lat1, &mut lat2);
        core::mem::swap(&mut lon1, &mut lon2);
    }
    let mut lam12 = reduce_sum(lon2, -lon1);
    let lon_flip = lam12 < 0.0;
    if lon_flip {
        lam12 = -lam12;
    }
    let lat_flip = lat1 > 0.0;
    // β from |φ|, so that equal and opposite latitudes give equal |sin β| and cos β exactly.
    let (sb1a, cb1) = reduced(lat1.abs());
    let (sb2a, cb2) = reduced(lat2.abs());
    let sb1 = -sb1a;
    let sb2 = if (lat2 < 0.0) == lat_flip {
        sb2a
    } else {
        -sb2a
    };
    let (sl12, cl12) = libm::sincos(lam12);
    let c = Canonical {
        sb1,
        cb1,
        sb2,
        cb2,
        lam12,
        sl12,
        cl12,
    };

    if lam12 == 0.0 && lat1 == lat2 {
        // Coincident points: every azimuth is a geodesic of length 0; 0 by convention.
        return Ok(GeodesicInverse {
            distance: Length::from_metres(0.0),
            azimuth1: Angle::from_radians(0.0),
            azimuth2: Angle::from_radians(0.0),
        });
    }
    let (s12, mut sa1, mut ca1, mut sa2, mut ca2) = if lam12 == 0.0 {
        // The meridian, northward: σ = β at both ends.
        meridian(&c)
    } else if sb1 == 0.0 && sb2 == 0.0 && lam12 <= F1 * PI {
        // The equator, up to its conjugate point at λ12 = (1 − f)π.
        (EQUATORIAL_RADIUS * lam12, 1.0, 0.0, 1.0, 0.0)
    } else {
        general(&c)
    };

    // Undo the canonical order: the latitude reflection negates cos α, the longitude reflection
    // sin α, and the swap exchanges the ends and reverses the direction.
    if lat_flip {
        ca1 = -ca1;
        ca2 = -ca2;
    }
    if lon_flip {
        sa1 = -sa1;
        sa2 = -sa2;
    }
    if swapped {
        (sa1, ca1, sa2, ca2) = (-sa2, -ca2, -sa1, -ca1);
    }
    Ok(GeodesicInverse {
        distance: Length::from_metres(s12),
        azimuth1: azimuth(sa1, ca1),
        azimuth2: azimuth(sa2, ca2),
    })
}

/// `(s12, sin α1, cos α1, sin α2, cos α2)` along the meridian, `α1 = α2 = 0`.
fn meridian(c: &Canonical) -> (f64, f64, f64, f64, f64) {
    let coef = Coefficients::new(EP2);
    let sig12 = libm::atan2(
        non_negative(c.cb1 * c.sb2 - c.sb1 * c.cb2),
        c.cb1 * c.cb2 + c.sb1 * c.sb2,
    );
    let db1 = sin_series(&coef.c1, c.sb2, c.cb2) - sin_series(&coef.c1, c.sb1, c.cb1);
    (POLAR_RADIUS * coef.a1 * (sig12 + db1), 0.0, 1.0, 0.0, 1.0)
}

/// The general case: Newton's method on `α1` (eq. 46), inside a bracket that bisection falls back
/// on. `λ12(α1)` increases on `[0, π]` from 0 to π (Karney Fig. 4), so the bracket always holds the
/// root and the iteration cannot fail. It stops after a Newton step below [`NEWTON_LAST`], or when a
/// step no longer moves `α1` or the residual stops falling, which is round-off.
// A step that leaves (sin α1, cos α1) bit for bit unchanged is the test, not a tolerance.
#[expect(
    clippy::float_cmp,
    reason = "an unchanged step is the stop, not a tolerance"
)]
fn general(c: &Canonical) -> (f64, f64, f64, f64, f64) {
    let (mut sa1, mut ca1) = c.start();
    // The bracket, as unit vectors: α = 0 and α = π.
    let (mut lo_s, mut lo_c) = (0.0, 1.0);
    let (mut hi_s, mut hi_c) = (0.0, -1.0);
    let mut best = c.hybrid(sa1, ca1);
    let mut best_angle = (sa1, ca1);
    let mut h_v = best.v;
    let mut h_dv = best.dv;
    for _ in 0..MAX_ITERATIONS {
        if h_v == 0.0 {
            break;
        }
        if h_v > 0.0 {
            (hi_s, hi_c) = (sa1, ca1);
        } else {
            (lo_s, lo_c) = (sa1, ca1);
        }
        // A Newton step, kept only if it lands strictly inside the bracket.
        let mut next = None;
        let mut last = false;
        if h_dv > 0.0 && h_dv.is_finite() {
            let d = -h_v / h_dv;
            if d.abs() < 1.0 {
                let (sd, cd) = libm::sincos(d);
                let (ns, nc) = unit(sa1 * cd + ca1 * sd, ca1 * cd - sa1 * sd);
                if d.abs() <= NEWTON_LAST {
                    // At round-off of α the bracket test is noise: take the step and stop.
                    last = true;
                    next = Some((ns, nc));
                } else if ns * lo_c - nc * lo_s > 0.0 && hi_s * nc - hi_c * ns > 0.0 {
                    // Inside (lo, hi) on [0, π]: sin(α − lo) > 0 and sin(hi − α) > 0.
                    next = Some((ns, nc));
                }
            }
        }
        let (ns, nc) = next.unwrap_or_else(|| {
            let (ms, mc) = (lo_s + hi_s, lo_c + hi_c);
            if ms == 0.0 && mc == 0.0 {
                (1.0, 0.0)
            } else {
                unit(ms, mc)
            }
        });
        if ns == sa1 && nc == ca1 {
            break;
        }
        (sa1, ca1) = (ns, nc);
        let h = c.hybrid(sa1, ca1);
        let improved = h.v.abs() < best.v.abs();
        h_v = h.v;
        h_dv = h.dv;
        if improved {
            best = h;
            best_angle = (sa1, ca1);
        }
        if last {
            break;
        }
        if !improved && next.is_some() && best.v.abs() < 1e-12 {
            // A Newton step from within round-off of the root that did not reduce the residual:
            // the residual is noise now.
            break;
        }
    }
    let (sa1, ca1) = best_angle;
    let coef = &best.coef;
    let db1 = sin_series(&coef.c1, best.ss2, best.cs2) - sin_series(&coef.c1, best.ss1, best.cs1);
    let s12 = POLAR_RADIUS * coef.a1 * (best.sig12 + db1);
    (s12, sa1, ca1, best.sa2, best.ca2)
}

/// The inverse problem for many pairs, into a caller-provided slice. A pair outside the domain
/// yields `Err` in its own slot; the rest of the batch is unaffected.
///
/// # Panics
///
/// If `out` is shorter than `pairs`.
pub fn geodesic_inverse_into(
    pairs: &[(SurfacePoint, SurfacePoint)],
    out: &mut [Result<GeodesicInverse, GeodesicError>],
) {
    assert!(
        out.len() >= pairs.len(),
        "output slice holds {} of {} pairs",
        out.len(),
        pairs.len()
    );
    for (&(from, to), slot) in pairs.iter().zip(out.iter_mut()) {
        *slot = geodesic_inverse(from, to);
    }
}
