//! The exact area of the sphere covered at least `k` times by spherical caps, with no grid: the
//! instantaneous `k`-fold coverage of sub-satellite footprints (plan 18 Ф31.D, F133 (a)).
//!
//! **The form.** Pick a unit vector `n` whose antipode `s = −n` lies on no cap's circle. In a frame
//! with `n` as the `z` axis the 1-form `ω = (x dy − y dx)/(1 + z) = (1 − cos θ) dφ` has `dω` equal to
//! the area form on the sphere less `s`, so by Stokes the area of a region `R` bounded by circle arcs
//! is the integral of `ω` over its boundary plus `4π` when `s ∈ R`. Along circle `i`,
//! `X(ψ) = cos ρ c + sin ρ (cos ψ u + sin ψ v)` with the cap on the left, the integral over
//! `[ψ₁, ψ₂]` is `(σ − cos ρ)(ψ₂ − ψ₁) − 2σ (G(X₂) − G(X₁))`, where with `A = n·c`, `B` the length
//! of `n` across `c`, `p = 1 + A cos ρ`, `q = B sin ρ`, `P = p + q`, `e = |cos ρ + A|` and
//! `σ = sgn(cos ρ + A)`,
//! `G(X) = atan2(2P (c × n)·X, (P + e)² + 2P (n·X − A cos ρ))`. The second argument is at least
//! `2e(e + P)`, which is positive while `s` is off the circle, so `G` is continuous along it. Every
//! step is derived in `derivations/kfold_coverage_area.mac` and re-evaluated at 60 digits in
//! `derivations/kfold_coverage_area.gp`.
//!
//! **The arcs and their depths.** The crossing points of two circles cut each into arcs. An arc of
//! circle `i` whose points lie in `d` other caps bounds the set covered at least `d + 1` times, with
//! that set on the cap's side, so `A_k` is the sum of the arc integrals at depth `k − 1`, plus `4π`
//! when `s` is covered at least `k` times. A depth is never tested at a point: each pair of circles
//! is classified once — crossing, nested, disjoint, or each inside the other's cap — and the class
//! sets the pair's contribution on both circles; crossings add an interval on each circle, bounded by
//! the same two points of the sphere, so the arcs of every level close up. The points and the arcs'
//! lengths come from the spherical half-angle formula. A crossing whose two points round to one, or
//! past each other, leaves an arc of length 0 on the side its length puts it, which is the tangency it
//! nearly is, so no threshold tells the two apart.
//!
//! **Coincident circles.** Several caps can share one circle, on the same side or on opposite sides
//! (complementary caps). An arc of such a circle with `P` caps on its left and `Q` on its right, `d`
//! more caps around it, bounds `{depth ≥ k}` positively for `d + Q < k ≤ d + P` and negatively for
//! `d + P < k ≤ d + Q`. Two caps on one circle need no special case: their pair is nested (the lower
//! index inside) or apart, which is that rule. Three or more caps whose circles chain within
//! [`COINCIDENT_ANGLE`] of each other are moved onto the first one's circle. Left apart, circles
//! `δ` apart that pass through one point cross there at an angle of order `δ`, so rounding moves
//! their crossings by about `2⁻⁵³/δ` along them and can swap their order; `δ` is a set's largest
//! `angle(c_i, c_j) + |ρ_i − ρ_j|`. Each move changes every `A_k` by at most the area of the band the
//! circle swept, and [`KfoldAreas::snap_bound`] is the sum of those bands. Measured against the same
//! arcs at 60 digits on 336 random sets of three or four such circles, `δ` from about `10⁻⁶` to
//! `10⁻¹²` (the test `near_coincident_corpus_within_its_bounds`, which prints the table again; the
//! sets are in `tests/fixtures/kfold_near_coincident.txt`): left apart, the areas were off by up to
//! `6.3 · 2⁻⁵³/δ`, and by up to `1.4 · 2⁻⁵³/δ` in the decades `10⁻⁶` to `10⁻⁹`; moved, by up to
//! `3.3 δ`, at most 0.23 of the bound. The two errors meet between `δ = 7·10⁻⁹` and `1.5·10⁻⁸`, at
//! 2·10⁻⁸ to 5·10⁻⁸ sr.
//!
//! **Zones.** A region bounded by circle arcs — a latitude band, a geodesic polygon with `ρ = π/2`
//! edges, a swath edge — enters as more circles, which carry a region count beside the cap counts
//! `P` and `Q` and add their crossings to the same per-circle intervals. A geodesic edge from `a` to
//! `b` lies on the circle `c = a × b/|a × b|`, `ρ = π/2`, run with `ψ` increasing, so a zone's
//! interior, on the left of each directed edge, is that circle's cap side here. Not built here.
//!
//! The cost is `O(m² log m)` for `m` caps: every pair of circles, then a sort per circle.
//!
//! **The enclosure.** [`KfoldAreas::lower`] and [`KfoldAreas::upper`] hold each `A_k` of the input,
//! **outward-rounded under H-libm** ([`KfoldAreas::LABEL`], plan 18 Ф28.V). The arcs the doubles
//! chose are integrated again on the exact circles in intervals rounded down and up at every
//! operation, the direction read from the operation's exact residual (`outward`); `libm`'s `sin`,
//! `cos` and `atan2` enter widened by their measured H-libm constants (`docs/verification.md` §1.1),
//! which is the hypothesis the label names. What the doubles may have chosen wrong is
//! bounded, not assumed away: where an arc's ends sit is compared with an interval of where the
//! exact arc's ends sit, by the extended half-angle formula, which is 0 or π exactly where the pair
//! does not cross; along circle `i` the depth then differs only on the two arcs' symmetric
//! difference, of measure at most `2(|Δc| + |Δh|)` for centres `c` and half-lengths `h`, and there
//! `|dω/dψ| ≤ sin ρ·tan(θ/2) ≤ 2 sin ρ/κ`, `κ` the pole's clearance from the circle
//! (`derivations/outward_rounding.mac`). Every pair of caps within [`COINCIDENT_ANGLE`] is moved
//! onto one circle for the enclosure, two included, and pays its band. The bound is first
//! order in the ends' error and takes the largest `|dω/dψ|` along each circle, so it is wider than
//! the error of [`KfoldAreas::at_least`]: `6.7·10⁻¹²` sr for twelve caps in general position, about
//! 500 times the round-off; `1.4·10⁻⁸` sr at a tangency or concentric pair, where the ends' error is
//! `√u`-sized; up to `10⁻⁵` sr for the near-coincident corpus, whose snapped pairs pay their band
//! (the tests measure each).
//!
//! This is design-time analysis (it allocates), so it lives behind the crate's `std` feature.

use spacelab_core::Vec3;

use crate::outward::{self, Iv, IvVec};

const PI: f64 = core::f64::consts::PI;
const TAU: f64 = core::f64::consts::TAU;

/// Three or more caps whose circles chain within this angle of each other, in centre and in
/// half-angle (or of each other's complement), are moved onto one circle. A pair within `2⁻²⁸` in
/// both is at most `δ = 2⁻²⁷ ≈ 7.5·10⁻⁹` apart, where the measured cost of a move, `3.3 δ`, meets
/// that of leaving circles apart, `1.4 … 6.3 · 2⁻⁵³/δ` (module docs); the power of two is checked in
/// `derivations/kfold_coverage_area.mac`.
pub const COINCIDENT_ANGLE: f64 = 1.0 / 268_435_456.0;

/// A spherical cap: the points within `half_angle` of `centre`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalCap {
    /// The cap's centre, a direction of any non-zero length.
    pub centre: Vec3,
    /// The cap's angular radius, radians, in `(0, π)`.
    pub half_angle: f64,
}

/// Why the areas were not computed: input outside the domain is refused, never answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapError {
    /// A centre is not finite or is the zero vector, so it names no point of the sphere.
    BadCentre {
        /// Index of the offending cap in the caller's slice.
        cap: usize,
    },
    /// A half-angle is not in `(0, π)`: NaN, infinite, at most 0 or at least π.
    BadHalfAngle {
        /// Index of the offending cap in the caller's slice.
        cap: usize,
    },
}

impl core::fmt::Display for CapError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BadCentre { cap } => {
                write!(f, "cap {cap}: the centre is not a finite non-zero vector")
            }
            Self::BadHalfAngle { cap } => write!(f, "cap {cap}: the half-angle is not in (0, pi)"),
        }
    }
}

impl core::error::Error for CapError {}

/// The areas covered at least `k` times, steradians.
#[derive(Clone, Debug, PartialEq)]
pub struct KfoldAreas {
    /// `at_least[k − 1]` is the area covered by at least `k` caps, `k = 1, …, n`; one entry per cap.
    /// Rounded to nearest, not a bound; it lies in `[lower[k − 1], upper[k − 1]]`.
    pub at_least: Vec<f64>,
    /// How far any `at_least` may be from the input's because caps were moved onto a shared circle:
    /// the sum over moved caps of the band each swept. 0 when no cap was moved.
    pub snap_bound: f64,
    /// `lower[k − 1] ≤ A_k`, outward-rounded under H-libm ([`KfoldAreas::LABEL`]).
    pub lower: Vec<f64>,
    /// `A_k ≤ upper[k − 1]`, outward-rounded under H-libm.
    pub upper: Vec<f64>,
}

impl KfoldAreas {
    /// What `[lower, upper]` rests on: outward rounding of every `+ − × ÷ √`, and the `libm`
    /// elementary functions within their measured error (H-libm, a tested hypothesis).
    pub const LABEL: &'static str = "outward-rounded under H-libm";
}

/// The area of the sphere covered at least `k` times by `caps`, for every `k` from 1 to the number of
/// caps (module docs): the value rounded to nearest, and its enclosure.
///
/// # Errors
/// [`CapError::BadCentre`] for a centre that is not finite or is zero, and
/// [`CapError::BadHalfAngle`] for a half-angle outside `(0, π)`.
pub fn kfold_areas(caps: &[SphericalCap]) -> Result<KfoldAreas, CapError> {
    areas_with(caps, COINCIDENT_ANGLE)
}

/// [`kfold_areas`] with the coincidence angle as a parameter, so a test can measure both sides of it.
fn areas_with(caps: &[SphericalCap], coincident: f64) -> Result<KfoldAreas, CapError> {
    let mut unit_caps = Vec::with_capacity(caps.len());
    for (i, cap) in caps.iter().enumerate() {
        let len = cap.centre.norm();
        if !(len.is_finite() && len > 0.0) {
            return Err(CapError::BadCentre { cap: i });
        }
        if !(cap.half_angle > 0.0 && cap.half_angle < PI) {
            return Err(CapError::BadHalfAngle { cap: i });
        }
        unit_caps.push(Cap {
            unit: cap.centre.scale(1.0 / len),
            rho: cap.half_angle,
            raw: cap.centre,
        });
    }
    let point = circles_of(&unit_caps, coincident, 3);
    let mut at_least = vec![0.0; caps.len()];
    if point.circles.is_empty() {
        return Ok(KfoldAreas {
            lower: at_least.clone(),
            upper: at_least.clone(),
            at_least,
            snap_bound: point.band,
        });
    }
    let arrangement = arrange(&point.circles);
    let s = arrangement.s;
    for (circle, sweep) in point.circles.iter().zip(&arrangement.sweeps) {
        sweep.integrate(circle, -s, &mut at_least);
    }
    let covered_at_s: usize = point
        .circles
        .iter()
        .map(|c| if dot(s, c.c) > c.cos { c.plus } else { c.minus })
        .sum();
    for area in &mut at_least[..covered_at_s] {
        *area += 2.0 * TAU;
    }
    // The enclosure moves pairs onto one circle too; when no class is a pair, the arrangement is
    // the same.
    let paired = circles_of(&unit_caps, coincident, 2);
    let enclosure = if paired.pairs {
        let arrangement = arrange(&paired.circles);
        enclose(&paired.circles, &arrangement, caps.len())
    } else {
        enclose(&point.circles, &arrangement, caps.len())
    };
    let (mut lower, mut upper) = (
        Vec::with_capacity(caps.len()),
        Vec::with_capacity(caps.len()),
    );
    let whole = outward::up(2.0 * outward::TAU.hi);
    for (k, &value) in at_least.iter().enumerate() {
        let (lo, hi) = enclosure.as_ref().map_or((0.0, whole), |sums| {
            let r = outward::up(sums.radius + paired.band_up);
            (
                outward::down(sums.levels[k].lo - r).max(0.0),
                outward::up(sums.levels[k].hi + r).min(whole),
            )
        });
        // Widening an enclosure keeps it one; the value rounded to nearest is inside it.
        lower.push(lo.min(value));
        upper.push(hi.max(value));
    }
    Ok(KfoldAreas {
        at_least,
        snap_bound: point.band,
        lower,
        upper,
    })
}

/// One input cap: its centre normalised in doubles, its half-angle, and the centre as given, whose
/// exact direction the enclosure takes.
#[derive(Clone, Copy)]
struct Cap {
    unit: Vec3,
    rho: f64,
    raw: Vec3,
}

/// One circle of the arrangement with its frame and the caps it bounds.
struct Circle {
    c: Vec3,
    u: Vec3,
    v: Vec3,
    rho: f64,
    cos: f64,
    sin: f64,
    /// The centre as the input gave it, of the cap whose circle this is.
    raw: Vec3,
    /// The coordinate axis `u` was built from.
    axis: Vec3,
    /// Caps on the `c` side, the left of `ψ` increasing.
    plus: usize,
    /// Caps on the `−c` side.
    minus: usize,
}

impl Circle {
    fn new(cap: Cap, plus: usize, minus: usize) -> Self {
        let c = cap.unit;
        // A coordinate axis at least √(2/3) across `c` makes `u` well defined; (u, v, c) is right-handed.
        let (ax, ay, az) = (c.x.abs(), c.y.abs(), c.z.abs());
        let axis = if ax <= ay && ax <= az {
            Vec3::new(1.0, 0.0, 0.0)
        } else if ay <= az {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
        let u = unit(axis.cross(c));
        let v = c.cross(u);
        let (sin, cos) = libm::sincos(cap.rho);
        Self {
            c,
            u,
            v,
            rho: cap.rho,
            cos,
            sin,
            raw: cap.raw,
            axis,
            plus,
            minus,
        }
    }

    /// The angle `ψ` of a point of the circle.
    fn psi(&self, x: Vec3) -> f64 {
        libm::atan2(dot(x, self.v), dot(x, self.u))
    }
}

/// The circles of an input and what moving caps onto shared circles cost.
struct Snapped {
    circles: Vec<Circle>,
    /// The swept-band bound, rounded to nearest: [`KfoldAreas::snap_bound`].
    band: f64,
    /// The same bound outward-rounded, for the enclosure.
    band_up: f64,
    /// Whether some class of exactly two caps was moved.
    pairs: bool,
}

/// The circles of `caps`: one per cap, except that `min_group` or more caps whose circles chain
/// within `coincident` share their first cap's circle.
fn circles_of(caps: &[Cap], coincident: f64, min_group: usize) -> Snapped {
    let m = caps.len();
    let mut parent: Vec<usize> = (0..m).collect();
    for i in 0..m {
        for j in (i + 1)..m {
            if same_circle(caps[i], caps[j], coincident) {
                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                // The smaller index is the root, so a class's circle is its first cap's.
                let (lo, hi) = if ri < rj { (ri, rj) } else { (rj, ri) };
                parent[hi] = lo;
            }
        }
    }
    let mut size = vec![0_usize; m];
    for i in 0..m {
        size[root(&mut parent, i)] += 1;
    }
    let mut counts = vec![(0_usize, 0_usize); m];
    let (mut band, mut band_up) = (0.0, 0.0);
    let mut circles = Vec::with_capacity(m);
    for i in 0..m {
        let r = root(&mut parent, i);
        if size[r] < min_group {
            circles.push(Circle::new(caps[i], 1, 0));
            continue;
        }
        let rep = caps[r];
        let cap = caps[i];
        let parallel = dot(cap.unit, rep.unit) >= 0.0;
        let (target, target_rho) = if parallel {
            (rep.unit, rep.rho)
        } else {
            (-rep.unit, PI - rep.rho)
        };
        if parallel {
            counts[r].0 += 1;
        } else {
            counts[r].1 += 1;
        }
        band += band_area(
            cap.rho,
            angle(cap.unit, target) + (cap.rho - target_rho).abs(),
        );
        band_up = outward::up(band_up + band_bound(cap, rep, parallel));
    }
    for (r, &(plus, minus)) in counts.iter().enumerate() {
        if plus + minus > 0 {
            circles.push(Circle::new(caps[r], plus, minus));
        }
    }
    Snapped {
        circles,
        band,
        band_up,
        pairs: size.contains(&2),
    }
}

/// Whether two caps' circles are within `coincident` of one circle, on either side.
fn same_circle(a: Cap, b: Cap, coincident: f64) -> bool {
    let d = angle(a.unit, b.unit);
    (d <= coincident && (a.rho - b.rho).abs() <= coincident)
        || (PI - d <= coincident && (a.rho + b.rho - PI).abs() <= coincident)
}

/// The area within `delta` of a circle of radius `rho`: the band a move of at most `delta` sweeps.
fn band_area(rho: f64, delta: f64) -> f64 {
    let lo = (rho - delta).max(0.0);
    let hi = (rho + delta).min(PI);
    2.0 * TAU * libm::sin(0.5 * (hi + lo)) * libm::sin(0.5 * (hi - lo))
}

/// An outward-rounded bound of the band `cap`'s circle sweeps when moved onto `rep`'s (or onto its
/// complement's): the exact move `δ` is at most the angle between the exact centres plus the
/// difference of half-angles, and the band within `δ` of a circle of radius `ρ` is at most
/// `2π·2δ·min(1, sin ρ + δ)`, since `sin` is 1-Lipschitz.
fn band_bound(cap: Cap, rep: Cap, parallel: bool) -> f64 {
    let c = IvVec::of(cap.raw).unit();
    let r = IvVec::of(rep.raw).unit();
    let (target, target_rho) = if parallel {
        (r, Iv::point(rep.rho))
    } else {
        (r.scale(Iv::point(-1.0)), outward::PI - Iv::point(rep.rho))
    };
    let between =
        outward::atan2(c.cross(target).norm(), c.dot(target)).map_or(outward::PI.hi, |a| a.hi);
    let delta = outward::up(between + (Iv::point(cap.rho) - target_rho).mag());
    let sin = outward::sin(Iv::point(cap.rho)).hi;
    let reach = outward::up(sin + delta).min(1.0);
    outward::up(outward::up(outward::up(2.0 * outward::TAU.hi) * delta) * reach)
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// A direction as far as the candidates allow from every circle: the six axes, the eight cube
/// diagonals, and each circle's centre and its antipode. The arc integrals' sensitivity to rounding
/// grows as the inverse of that distance.
fn clear_point(circles: &[Circle]) -> Vec3 {
    let clearance = |s: Vec3| {
        circles.iter().fold(f64::INFINITY, |acc, c| {
            acc.min((angle(s, c.c) - c.rho).abs())
        })
    };
    let mut best = (f64::NEG_INFINITY, Vec3::new(0.0, 0.0, 1.0));
    let mut consider = |s: Vec3| {
        let clear = clearance(s);
        if clear > best.0 {
            best = (clear, s);
        }
    };
    for k in 0..3 {
        let mut e = [0.0; 3];
        e[k] = 1.0;
        let axis = Vec3::new(e[0], e[1], e[2]);
        consider(axis);
        consider(-axis);
    }
    for signs in 0..8_u8 {
        let sign = |bit: u8| if signs & bit == 0 { 1.0 } else { -1.0 };
        consider(unit(Vec3::new(sign(1), sign(2), sign(4))));
    }
    for c in circles {
        consider(c.c);
        consider(-c.c);
    }
    best.1
}

/// How two circles meet.
enum Relation {
    /// They do not cross: whether each circle lies inside the other's cap.
    Apart { i_inside_j: bool, j_inside_i: bool },
    /// They cross: circle `i` enters `j`'s cap at `enter` and leaves it at `exit`, both on circle `i`;
    /// the arc of circle `i` inside `j`'s cap subtends `2 h_i` at `c_i`, that of `j` inside `i`'s
    /// `2 h_j` at `c_j`.
    Cross {
        enter: Vec3,
        exit: Vec3,
        h_i: f64,
        h_j: f64,
    },
}

/// The relation of circles `i` and `j`, `i` listed first.
fn relation(ci: &Circle, cj: &Circle) -> Relation {
    let cr = ci.c.cross(cj.c);
    let d = libm::atan2(norm(cr), dot(ci.c, cj.c));
    let (ri, rj) = (ci.rho, cj.rho);
    // Positive when the circles are not nested, when the caps overlap, and when their complements
    // overlap; all three, and only then, the circles cross.
    let not_nested = d - (ri - rj).abs();
    let caps_meet = ri + rj - d;
    let complements_meet = (TAU - (ri + rj)) - d;
    let least = not_nested.min(caps_meet).min(complements_meet);
    if least <= 0.0 {
        return if not_nested <= caps_meet.min(complements_meet) {
            // The smaller cap lies in the larger; equal caps on one circle nest by order.
            let i_inner = ri <= rj;
            Relation::Apart {
                i_inside_j: i_inner,
                j_inside_i: !i_inner,
            }
        } else {
            // Disjoint caps leave each circle outside; disjoint complements put each inside.
            let inside = complements_meet < caps_meet;
            Relation::Apart {
                i_inside_j: inside,
                j_inside_i: inside,
            }
        };
    }
    // The triangle (c_i, c_j, x) has sides d, ρ_i, ρ_j; its angles at c_i and c_j by the half-angle
    // formula, with s = (d + ρ_i + ρ_j)/2: s − d, s − ρ_i, s − ρ_j and π − s, each positive.
    let sin_s_d = libm::sin(0.5 * caps_meet);
    let sin_pi_s = libm::sin(0.5 * complements_meet);
    // sin(s − ρ_i) and sin(s − ρ_j).
    let sin_s_less = [
        libm::sin(0.5 * (d + (rj - ri))),
        libm::sin(0.5 * (d + (ri - rj))),
    ];
    let h_i = 2.0
        * libm::atan2(
            libm::sqrt(sin_s_d * sin_s_less[0]),
            libm::sqrt(sin_pi_s * sin_s_less[1]),
        );
    let h_j = 2.0
        * libm::atan2(
            libm::sqrt(sin_s_d * sin_s_less[1]),
            libm::sqrt(sin_pi_s * sin_s_less[0]),
        );
    // The direction towards c_j across c_i, (c_i × c_j) × c_i, and the one a quarter turn on.
    let w = unit(cr.cross(ci.c));
    let t = ci.c.cross(w);
    let (sh, ch) = libm::sincos(h_i);
    let foot = ci.c.scale(ci.cos) + w.scale(ci.sin * ch);
    let side = t.scale(ci.sin * sh);
    Relation::Cross {
        enter: foot - side,
        exit: foot + side,
        h_i,
        h_j,
    }
}

/// The part of one circle the doubles put inside another cap: nothing, everything, or the arc from
/// `from` counterclockwise to `to`, `from ≠ to`.
#[derive(Clone, Copy, Debug)]
enum Chosen {
    Empty,
    Full,
    Arc { from: f64, to: f64 },
}

/// A point where the depth along a circle changes: the count goes up by `add`, then down by `sub`.
struct Event {
    psi: f64,
    point: Vec3,
    add: usize,
    sub: usize,
}

/// One circle's depth: the count other caps give at `ψ = −π`, and where it changes.
#[derive(Default)]
struct Sweep {
    base: usize,
    events: Vec<Event>,
}

impl Sweep {
    /// The circle runs inside a cap set on the arc `(from, to)`, `length` radians long: `inside` caps
    /// cover it there, `outside` caps elsewhere. Returns the arc the doubles chose, or `None` when
    /// `inside = outside` and the arc changes nothing.
    #[expect(
        clippy::float_cmp,
        reason = "ends on one angle are one point, not nearly one; `==` also holds -0 equal to +0, \
                  which the sort orders apart"
    )]
    fn interval(
        &mut self,
        circle: &Circle,
        (from, to): (Vec3, Vec3),
        length: f64,
        inside: usize,
        outside: usize,
    ) -> Option<Chosen> {
        if inside == outside {
            self.base += outside;
            return None;
        }
        let a = circle.psi(from);
        let mut b = circle.psi(to);
        // Two ends that nearly coincide can round onto one angle or past each other, which turns an
        // arc of almost nothing into almost the whole circle or back. Their angles then differ from
        // the half-angle formula's length by more than π; the ends are taken as one point, and the
        // length says which side of it the arc lies on.
        let measured = if b < a { b - a + TAU } else { b - a };
        if b == a || (measured - length).abs() > PI {
            b = a;
        }
        let wraps = if b == a { length > PI } else { a > b };
        let enter = Event {
            psi: a,
            point: from,
            add: inside,
            sub: outside,
        };
        let leave = Event {
            psi: b,
            point: to,
            add: outside,
            sub: inside,
        };
        // The stable sort keeps this order for equal angles, so the count at `ψ = −π` and the order
        // of the two events agree.
        if wraps {
            self.base += inside;
            self.events.extend([leave, enter]);
        } else {
            self.base += outside;
            self.events.extend([enter, leave]);
        }
        Some(if b != a {
            Chosen::Arc { from: a, to: b }
        } else if wraps {
            Chosen::Full
        } else {
            Chosen::Empty
        })
    }

    /// Add this circle's arcs to the areas, with `n` the pole of the form; the events are sorted.
    fn integrate(&self, circle: &Circle, n: Vec3, at_least: &mut [f64]) {
        let form = ArcForm::new(circle, n);
        if self.events.is_empty() {
            add_arc(circle, self.base, form.integral(TAU, 0.0), at_least);
            return;
        }
        let g: Vec<f64> = self.events.iter().map(|e| form.g(e.point)).collect();
        let last = self.events.len() - 1;
        let mut depth = self.base;
        for (k, event) in self.events.iter().enumerate() {
            depth += event.add;
            depth -= event.sub;
            let (psi_next, g_next) = if k < last {
                (self.events[k + 1].psi, g[k + 1])
            } else {
                (self.events[0].psi + TAU, g[0])
            };
            let arc = form.integral(psi_next - event.psi, g_next - g[k]);
            add_arc(circle, depth, arc, at_least);
        }
    }
}

/// The arcs of every circle with their depths, the pole's antipode `s`, and per circle the arc the
/// doubles chose inside each other circle's caps.
struct Arrangement {
    s: Vec3,
    sweeps: Vec<Sweep>,
    chosen: Vec<Vec<(usize, Chosen)>>,
}

/// Classify every pair once, collect the events and sort them along each circle.
fn arrange(circles: &[Circle]) -> Arrangement {
    let s = clear_point(circles);
    let mut sweeps: Vec<Sweep> = circles.iter().map(|_| Sweep::default()).collect();
    let mut chosen: Vec<Vec<(usize, Chosen)>> = circles.iter().map(|_| Vec::new()).collect();
    for i in 0..circles.len() {
        for j in (i + 1)..circles.len() {
            let (ci, cj) = (&circles[i], &circles[j]);
            match relation(ci, cj) {
                Relation::Apart {
                    i_inside_j,
                    j_inside_i,
                } => {
                    sweeps[i].base += if i_inside_j { cj.plus } else { cj.minus };
                    sweeps[j].base += if j_inside_i { ci.plus } else { ci.minus };
                    let whole = |inside: bool| if inside { Chosen::Full } else { Chosen::Empty };
                    if cj.plus != cj.minus {
                        chosen[i].push((j, whole(i_inside_j)));
                    }
                    if ci.plus != ci.minus {
                        chosen[j].push((i, whole(j_inside_i)));
                    }
                }
                Relation::Cross {
                    enter,
                    exit,
                    h_i,
                    h_j,
                } => {
                    // Circle i runs inside j's cap from `enter` to `exit`; circle j runs inside i's
                    // cap from `exit` to `enter`, the same two points crossed the other way.
                    if let Some(arc) =
                        sweeps[i].interval(ci, (enter, exit), 2.0 * h_i, cj.plus, cj.minus)
                    {
                        chosen[i].push((j, arc));
                    }
                    if let Some(arc) =
                        sweeps[j].interval(cj, (exit, enter), 2.0 * h_j, ci.plus, ci.minus)
                    {
                        chosen[j].push((i, arc));
                    }
                }
            }
        }
    }
    for sweep in &mut sweeps {
        sweep.events.sort_by(|a, b| a.psi.total_cmp(&b.psi));
    }
    Arrangement { s, sweeps, chosen }
}

/// Add the integral of an arc at `depth` (caps of other circles around it) to the levels it bounds.
fn add_arc(circle: &Circle, depth: usize, integral: f64, at_least: &mut [f64]) {
    let (plus, minus) = (circle.plus, circle.minus);
    if plus > minus {
        for area in &mut at_least[depth + minus..depth + plus] {
            *area += integral;
        }
    } else {
        for area in &mut at_least[depth + plus..depth + minus] {
            *area -= integral;
        }
    }
}

/// The integral of `ω` along arcs of one circle, for the pole `n`.
struct ArcForm {
    n: Vec3,
    c_cross_n: Vec3,
    cos: f64,
    sigma: f64,
    a_cos: f64,
    two_p: f64,
    shift: f64,
}

impl ArcForm {
    fn new(circle: &Circle, n: Vec3) -> Self {
        // A, B, p, q, P and e of the module docs.
        let along = dot(n, circle.c);
        let across = libm::hypot(dot(n, circle.u), dot(n, circle.v));
        let big_p = (1.0 + along * circle.cos) + across * circle.sin;
        let gap = (circle.cos + along).abs();
        Self {
            n,
            c_cross_n: circle.c.cross(n),
            cos: circle.cos,
            sigma: if circle.cos + along < 0.0 { -1.0 } else { 1.0 },
            a_cos: along * circle.cos,
            two_p: 2.0 * big_p,
            shift: (big_p + gap) * (big_p + gap),
        }
    }

    /// `G` at a point of the circle.
    fn g(&self, x: Vec3) -> f64 {
        libm::atan2(
            self.two_p * dot(self.c_cross_n, x),
            self.shift + self.two_p * (dot(self.n, x) - self.a_cos),
        )
    }

    /// The integral over an arc of `dpsi` radians whose `G` rises by `dg`.
    fn integral(&self, dpsi: f64, dg: f64) -> f64 {
        (self.sigma - self.cos) * dpsi - 2.0 * self.sigma * dg
    }
}

// ── The enclosure ─────────────────────────────────────────────────────────────────────────────────

/// One circle of the arrangement as an exact circle, in outward-rounded intervals: the exact unit
/// centre of the input's vector, a frame of it, and the form for the exact pole `n`.
struct Exact {
    c: IvVec,
    u: IvVec,
    v: IvVec,
    rho: f64,
    n: IvVec,
    c_cross_n: IvVec,
    cos: Iv,
    sin: Iv,
    /// `σ`, certain: the pole's antipode is off the circle.
    sigma: f64,
    a_cos: Iv,
    two_p: Iv,
    shift: Iv,
    /// `σ − cos ρ`.
    slope: Iv,
    /// A bound of `|dF/dψ|` along the circle: `2 sin ρ/κ`, `κ` the antipode's clearance.
    lipschitz: f64,
}

impl Exact {
    /// `None` when the pole's antipode `s` cannot be shown off the circle.
    fn new(circle: &Circle, pole: IvVec, antipode: IvVec) -> Option<Self> {
        let c = IvVec::of(circle.raw).unit();
        let u = IvVec::of(circle.axis).cross(c).unit();
        let v = c.cross(u);
        let rho = Iv::point(circle.rho);
        let (cos, sin) = (outward::cos(rho), outward::sin(rho).max0());
        let along = pole.dot(c);
        let across = (pole.dot(u).sq() + pole.dot(v).sq()).sqrt();
        let big_p = (Iv::point(1.0) + along * cos) + across * sin;
        let gap_signed = cos + along;
        if gap_signed.lo <= 0.0 && gap_signed.hi >= 0.0 {
            return None;
        }
        let sigma = if gap_signed.hi < 0.0 { -1.0 } else { 1.0 };
        let clear = (outward::atan2(antipode.cross(c).norm(), antipode.dot(c))? - rho).mig();
        if clear <= 0.0 {
            return None;
        }
        let two_p = Iv::point(2.0) * big_p;
        Some(Self {
            c,
            u,
            v,
            rho: circle.rho,
            n: pole,
            c_cross_n: c.cross(pole),
            cos,
            sin,
            sigma,
            a_cos: along * cos,
            two_p,
            shift: (big_p + gap_signed.abs()).sq(),
            slope: Iv::point(sigma) - cos,
            lipschitz: outward::up(outward::up(2.0 * sin.hi) / clear),
        })
    }

    /// `F(ψ) = (σ − cos ρ)ψ − 2σ G(X(ψ))` at the exact circle's point of parameter `psi`.
    fn antiderivative(&self, psi: f64) -> Option<Iv> {
        let p = Iv::point(psi);
        let x = self.c.scale(self.cos).add(
            self.u
                .scale(self.sin * outward::cos(p))
                .add(self.v.scale(self.sin * outward::sin(p))),
        );
        let num = self.two_p * self.c_cross_n.dot(x);
        let den = self.shift + self.two_p * (self.n.dot(x) - self.a_cos);
        let g = outward::atan2(num, den)?;
        Some(self.slope * p - Iv::point(2.0 * self.sigma) * g)
    }
}

/// Where the arc of one circle inside another cap really is: its centre on the circle as a ball
/// (none when the centres are within rounding of equal or antipodal), and an interval of its
/// half-length, in `[0, π]`.
struct Truth {
    centre: Option<(f64, f64)>,
    half: Iv,
}

/// The true arcs of circle `i` inside `j`'s cap and of `j` inside `i`'s, by the extended half-angle
/// formula: each sine factor is taken as `max(0, ·)`, which gives 0 or π exactly where the pair does
/// not cross, as the relation's three cases do.
fn truths(ei: &Exact, ej: &Exact) -> (Truth, Truth) {
    let full = Iv {
        lo: 0.0,
        hi: outward::PI.hi,
    };
    let d = outward::atan2(ei.c.cross(ej.c).norm(), ei.c.dot(ej.c))
        .unwrap_or(full)
        .meet(full);
    let (ri, rj, half) = (Iv::point(ei.rho), Iv::point(ej.rho), Iv::point(0.5));
    let factor = |x: Iv| outward::sin(x * half).max0();
    let sd = factor(ri + rj - d);
    let sp = factor(outward::TAU - (ri + rj) - d);
    let s1 = factor(d + rj - ri);
    let s2 = factor(d + ri - rj);
    let h = |num: Iv, den: Iv| {
        let two = Iv::point(2.0);
        (two * outward::atan2_first_quadrant(num.sqrt(), den.sqrt())).meet(full)
    };
    (
        Truth {
            centre: outward::angle_ball(ej.c.dot(ei.v), ej.c.dot(ei.u)),
            half: h(sd * s1, sp * s2),
        },
        Truth {
            centre: outward::angle_ball(ei.c.dot(ej.v), ei.c.dot(ej.u)),
            half: h(sd * s2, sp * s1),
        },
    )
}

/// An upper bound of the measure of `ψ` where the chosen arc and the true one differ: at most
/// `|A| + |B|`, at most `|Aᶜ| + |Bᶜ|`, and at most `2(|Δc| + |Δh|)`.
fn sweep_bound(chosen: Chosen, truth: &Truth) -> f64 {
    use outward::up;
    let h = truth.half;
    let (half, centre) = match chosen {
        Chosen::Empty => (Iv::point(0.0), None),
        Chosen::Full => (outward::PI, None),
        Chosen::Arc { from, to } => {
            let len = if to > from {
                Iv::point(to) - Iv::point(from)
            } else {
                Iv::point(to) - Iv::point(from) + outward::TAU
            };
            let half = len * Iv::point(0.5);
            (half, Some(Iv::point(from) + half))
        }
    };
    let apart = up(up(2.0 * half.hi) + up(2.0 * h.hi));
    let together =
        up(up(2.0 * outward::TAU.hi) - outward::down(2.0 * half.lo) - outward::down(2.0 * h.lo));
    let mut best = apart.min(together);
    if let (Some(c), Some((mid, radius))) = (centre, truth.centre) {
        let dc = up(outward::circle_distance(c - Iv::point(mid)) + radius);
        let dh = up(half.hi - h.lo).max(up(h.hi - half.lo));
        best = best.min(up(2.0 * up(dc + dh)));
    }
    best
}

/// The interval sums of every level and the radius every level shares.
struct Sums {
    levels: Vec<Iv>,
    radius: f64,
}

/// The arcs the doubles chose, integrated on the exact circles in intervals, and the bound on what
/// the choice may have got wrong (module docs). `None` when the pole cannot be shown clear of a
/// circle or a denominator positive; the caller then reports `[0, 4π]`.
fn enclose(circles: &[Circle], arrangement: &Arrangement, caps: usize) -> Option<Sums> {
    let s = IvVec::of(arrangement.s).unit();
    let n = s.scale(Iv::point(-1.0));
    let exact: Vec<Exact> = circles
        .iter()
        .map(|c| Exact::new(c, n, s))
        .collect::<Option<_>>()?;
    let mut levels = vec![Iv::point(0.0); caps];
    let mut radius = 0.0;
    let mut truth_cache: Vec<Vec<Option<Truth>>> = circles
        .iter()
        .map(|_| circles.iter().map(|_| None).collect())
        .collect();
    for i in 0..circles.len() {
        for j in (i + 1)..circles.len() {
            let (ti, tj) = truths(&exact[i], &exact[j]);
            truth_cache[i][j] = Some(ti);
            truth_cache[j][i] = Some(tj);
        }
    }
    for (i, (circle, sweep)) in circles.iter().zip(&arrangement.sweeps).enumerate() {
        let ex = &exact[i];
        let mut measure = 0.0;
        for &(j, chosen) in &arrangement.chosen[i] {
            let truth = truth_cache[i][j].as_ref()?;
            measure = outward::up(measure + sweep_bound(chosen, truth));
        }
        radius = outward::up(radius + outward::up(ex.lipschitz * measure));
        if sweep.events.is_empty() {
            add_arc_enclosed(circle, sweep.base, ex.slope * outward::TAU, &mut levels);
            continue;
        }
        let f: Vec<Iv> = sweep
            .events
            .iter()
            .map(|e| ex.antiderivative(e.psi))
            .collect::<Option<_>>()?;
        let last = sweep.events.len() - 1;
        let mut depth = sweep.base;
        for (k, event) in sweep.events.iter().enumerate() {
            depth += event.add;
            depth -= event.sub;
            let next = if k < last {
                f[k + 1]
            } else {
                f[0] + ex.slope * outward::TAU
            };
            add_arc_enclosed(circle, depth, next - f[k], &mut levels);
        }
    }
    let covered: usize = exact
        .iter()
        .zip(circles)
        .map(|(e, c)| if e.sigma < 0.0 { c.plus } else { c.minus })
        .sum();
    let four_pi = Iv::point(2.0) * outward::TAU;
    for level in &mut levels[..covered] {
        *level = *level + four_pi;
    }
    Some(Sums { levels, radius })
}

/// [`add_arc`] in intervals.
fn add_arc_enclosed(circle: &Circle, depth: usize, integral: Iv, levels: &mut [Iv]) {
    let (plus, minus) = (circle.plus, circle.minus);
    if plus > minus {
        for level in &mut levels[depth + minus..depth + plus] {
            *level = *level + integral;
        }
    } else {
        for level in &mut levels[depth + plus..depth + minus] {
            *level = *level - integral;
        }
    }
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn norm(v: Vec3) -> f64 {
    libm::sqrt(dot(v, v))
}

fn unit(v: Vec3) -> Vec3 {
    v.scale(1.0 / norm(v))
}

/// The angle between two unit vectors, accurate at 0 and at π.
fn angle(a: Vec3, b: Vec3) -> f64 {
    libm::atan2(norm(a.cross(b)), dot(a, b))
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "a test fails loudly on a refused input")]
    use super::*;

    fn cap(x: f64, y: f64, z: f64, half_angle: f64) -> SphericalCap {
        SphericalCap {
            centre: Vec3::new(x, y, z),
            half_angle,
        }
    }

    fn areas(caps: &[SphericalCap]) -> Vec<f64> {
        kfold_areas(caps).unwrap().at_least
    }

    /// `2π(1 − cos ρ)` as `4π sin²(ρ/2)`, which keeps its digits for small caps.
    fn cap_area(rho: f64) -> f64 {
        let s = libm::sin(0.5 * rho);
        2.0 * TAU * s * s
    }

    fn worst_gap(got: &[f64], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        got.iter().zip(want).fold(0.0, |acc, (g, w)| {
            spacelab_testkit::nan_max(acc, (g - w).abs())
        })
    }

    /// `SplitMix64`: a fixed stream, so a failing set is reproducible from its seed alone.
    struct SplitMix(u64);

    impl SplitMix {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            #[expect(clippy::cast_precision_loss, reason = "53 bits are exact in an f64")]
            let unit = (z >> 11) as f64 / 9_007_199_254_740_992.0;
            unit
        }

        /// A direction uniform on the sphere (Archimedes: `z` uniform in `[−1, 1]`).
        fn direction(&mut self) -> Vec3 {
            let z = 2.0 * self.next() - 1.0;
            let phi = TAU * self.next();
            let r = libm::sqrt((1.0 - z * z).max(0.0));
            Vec3::new(r * libm::cos(phi), r * libm::sin(phi), z)
        }
    }

    fn random_caps(rng: &mut SplitMix, m: usize, lo: f64, hi: f64) -> Vec<SphericalCap> {
        (0..m)
            .map(|_| SphericalCap {
                centre: rng.direction(),
                half_angle: lo + (hi - lo) * rng.next(),
            })
            .collect()
    }

    // ── S1: the area by quadrature over colatitude ────────────────────────────────────────────

    /// Gauss–Legendre nodes and weights on `[−1, 1]`, by Newton's method on `P_n`.
    fn gauss_legendre(n: usize) -> Vec<(f64, f64)> {
        #[expect(clippy::cast_precision_loss, reason = "a node count is far below 2^52")]
        let nf = n as f64;
        (1..=n)
            .map(|i| {
                #[expect(clippy::cast_precision_loss, reason = "a node index is far below 2^52")]
                let mut x = libm::cos(PI * (i as f64 - 0.25) / (nf + 0.5));
                let mut dp = 0.0;
                for _ in 0..100 {
                    let (mut p0, mut p1) = (1.0, x);
                    for k in 2..=n {
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "a degree is far below 2^52"
                        )]
                        let kf = k as f64;
                        let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                        p0 = p1;
                        p1 = p2;
                    }
                    dp = nf * (x * p1 - p0) / (x * x - 1.0);
                    let step = p1 / dp;
                    x -= step;
                    if step.abs() < 1e-16 {
                        break;
                    }
                }
                (x, 2.0 / ((1.0 - x * x) * dp * dp))
            })
            .collect()
    }

    /// The length of longitude covered at least `k` times, `k = 1, …, m`, on the circle of
    /// colatitude `theta` about `axis`; each cap given as `(colatitude, longitude, ρ)`.
    fn covered_longitude(polar: &[(f64, f64, f64)], theta: f64) -> Vec<f64> {
        let m = polar.len();
        let (st, ct) = libm::sincos(theta);
        let mut full = 0_usize;
        let mut events: Vec<(f64, i32)> = Vec::new();
        for &(ti, phi, rho) in polar {
            let (si, ci) = libm::sincos(ti);
            let w = (libm::cos(rho) - ct * ci) / (st * si);
            if w <= -1.0 {
                full += 1;
            } else if w < 1.0 {
                let h = libm::acos(w);
                let start = (phi - h).rem_euclid(TAU);
                let end = start + 2.0 * h;
                if end <= TAU {
                    events.push((start, 1));
                    events.push((end, -1));
                } else {
                    events.push((start, 1));
                    events.push((TAU, -1));
                    events.push((0.0, 1));
                    events.push((end - TAU, -1));
                }
            }
        }
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut by_count = vec![0.0; m + 1];
        let mut count = i64::try_from(full).unwrap();
        let mut at = 0.0;
        for (psi, step) in events {
            by_count[usize::try_from(count).unwrap()] += psi - at;
            at = psi;
            count += i64::from(step);
        }
        by_count[usize::try_from(count).unwrap()] += TAU - at;
        let mut at_least = vec![0.0; m];
        let mut acc = 0.0;
        for k in (1..=m).rev() {
            acc += by_count[k];
            at_least[k - 1] = acc;
        }
        at_least
    }

    /// The crossing points of two caps' circles by the vector formula
    /// `x = αc_i + βc_j ± γ c_i × c_j` (not the half-angle route of the code).
    fn crossings(ci: Vec3, ri: f64, cj: Vec3, rj: f64) -> Vec<Vec3> {
        let g = ci.dot(cj);
        let (ai, aj) = (libm::cos(ri), libm::cos(rj));
        let k = 1.0 - g * g;
        let det = k - ai * ai - aj * aj + 2.0 * ai * aj * g;
        if k <= 0.0 || det <= 0.0 {
            return Vec::new();
        }
        let (alpha, beta, gamma) = ((ai - aj * g) / k, (aj - ai * g) / k, libm::sqrt(det) / k);
        let mid = ci.scale(alpha) + cj.scale(beta);
        let off = ci.cross(cj).scale(gamma);
        vec![mid + off, mid - off]
    }

    /// `A_k` as `∫ L_k(θ) sin θ dθ` over colatitude about `axis`, split at every colatitude where a
    /// latitude circle touches a cap or passes a crossing, with `θ = a + (b − a)(1 − cos τ)/2` on each
    /// piece so that the square-root ends of a touching piece are smooth in `τ`.
    fn colatitude_areas(caps: &[SphericalCap], axis: Vec3, nodes: usize) -> Vec<f64> {
        let ez = unit(axis);
        let ex = unit(
            if ez.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            }
            .cross(ez),
        );
        let ey = ez.cross(ex);
        let units: Vec<(Vec3, f64)> = caps
            .iter()
            .map(|c| (unit(c.centre), c.half_angle))
            .collect();
        let polar: Vec<(f64, f64, f64)> = units
            .iter()
            .map(|&(c, rho)| {
                (
                    libm::acos(c.dot(ez).clamp(-1.0, 1.0)),
                    libm::atan2(c.dot(ey), c.dot(ex)),
                    rho,
                )
            })
            .collect();
        let mut cuts = vec![0.0, PI];
        for &(ti, _, rho) in &polar {
            // A cap over a pole touches its latitude circles at |θ_i − ρ| or 2π − θ_i − ρ.
            let touch = [(ti - rho).abs(), (ti + rho).min(TAU - ti - rho)];
            cuts.extend(touch.into_iter().filter(|t| *t > 0.0 && *t < PI));
        }
        for i in 0..units.len() {
            for j in (i + 1)..units.len() {
                for p in crossings(units[i].0, units[i].1, units[j].0, units[j].1) {
                    cuts.push(libm::acos((p.dot(ez) / p.norm()).clamp(-1.0, 1.0)));
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        let rule = gauss_legendre(nodes);
        let mut total = vec![0.0; caps.len()];
        for pair in cuts.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if b - a <= 0.0 {
                continue;
            }
            for &(t, w) in &rule {
                let tau = 0.5 * PI * (t + 1.0);
                let (st, ct) = libm::sincos(tau);
                let theta = a + 0.5 * (b - a) * (1.0 - ct);
                let jac = 0.5 * (b - a) * st * 0.5 * PI;
                let lengths = covered_longitude(&polar, theta);
                for (acc, l) in total.iter_mut().zip(&lengths) {
                    *acc += w * jac * l * libm::sin(theta);
                }
            }
        }
        total
    }

    // ── S2: the sampled sphere ──────────────────────────────────────────────────────────────────

    /// The fraction of `n` uniform points covered at least `k` times, as an area, with the binomial
    /// standard deviation of each.
    fn sampled_areas(caps: &[SphericalCap], n: u32, seed: u64) -> Vec<(f64, f64)> {
        let mut rng = SplitMix(seed);
        let units: Vec<(Vec3, f64)> = caps
            .iter()
            .map(|c| (unit(c.centre), libm::cos(c.half_angle)))
            .collect();
        let mut hits = vec![0_u32; caps.len() + 1];
        for _ in 0..n {
            let p = rng.direction();
            let depth = units.iter().filter(|(c, cos)| p.dot(*c) >= *cos).count();
            hits[depth] += 1;
        }
        let nf = f64::from(n);
        let mut at_least = vec![(0.0, 0.0); caps.len()];
        let mut acc = 0_u32;
        for k in (1..=caps.len()).rev() {
            acc += hits[k];
            let p = f64::from(acc) / nf;
            at_least[k - 1] = (2.0 * TAU * p, 2.0 * TAU * libm::sqrt(p * (1.0 - p) / nf));
        }
        at_least
    }

    // ── The comparator: the M0-1 prototype's labelling ────────────────────────────────────────

    /// The M0-1 prototype's labelling, kept to show why this module never tests a depth at a point:
    /// each circle is cut where `acos` of the law of cosines puts the crossings, and each arc's depth
    /// is the number of other caps that strictly contain its midpoint. On a shared circle that
    /// midpoint is on the other cap's boundary, and the count goes wrong by a whole cap.
    fn midpoint_labelled_areas(caps: &[SphericalCap]) -> Vec<f64> {
        let circles: Vec<Circle> = caps
            .iter()
            .map(|c| {
                let cap = Cap {
                    unit: unit(c.centre),
                    rho: c.half_angle,
                    raw: c.centre,
                };
                Circle::new(cap, 1, 0)
            })
            .collect();
        let s = clear_point(&circles);
        let inside = |c: &Circle, x: Vec3| dot(c.c, x) > c.cos;
        let mut at_least = vec![0.0; caps.len()];
        for (i, ci) in circles.iter().enumerate() {
            let mut cuts = Vec::new();
            for (j, cj) in circles.iter().enumerate() {
                let sd = norm(ci.c.cross(cj.c));
                let w = (cj.cos - ci.cos * dot(ci.c, cj.c)) / (ci.sin * sd);
                if j != i && w.abs() < 1.0 {
                    let centre = ci.psi(cj.c);
                    let h = libm::acos(w);
                    cuts.push((centre - h).rem_euclid(TAU));
                    cuts.push((centre + h).rem_euclid(TAU));
                }
            }
            cuts.sort_by(f64::total_cmp);
            let point = |psi: f64| {
                ci.c.scale(ci.cos)
                    + (ci.u.scale(libm::cos(psi)) + ci.v.scale(libm::sin(psi))).scale(ci.sin)
            };
            let form = ArcForm::new(ci, -s);
            let depth_at = |x: Vec3| {
                circles
                    .iter()
                    .enumerate()
                    .filter(|&(j, cj)| j != i && inside(cj, x))
                    .count()
            };
            if cuts.is_empty() {
                let depth = depth_at(point(0.0));
                at_least[depth] += form.integral(TAU, 0.0);
                continue;
            }
            for k in 0..cuts.len() {
                let a = cuts[k];
                let b = if k + 1 < cuts.len() {
                    cuts[k + 1]
                } else {
                    cuts[0] + TAU
                };
                let depth = depth_at(point(0.5 * (a + b)));
                at_least[depth] += form.integral(b - a, form.g(point(b)) - form.g(point(a)));
            }
        }
        let covered = circles.iter().filter(|c| inside(c, s)).count();
        for area in &mut at_least[..covered] {
            *area += 2.0 * TAU;
        }
        at_least
    }

    // ── Closed forms ─────────────────────────────────────────────────────────────────────────────

    /// Two caps `d` apart in the `x`–`z` plane.
    fn pair(d: f64, r1: f64, r2: f64) -> Vec<SphericalCap> {
        vec![
            cap(0.0, 0.0, 1.0, r1),
            cap(libm::sin(d), 0.0, libm::cos(d), r2),
        ]
    }

    /// The lens of two crossing caps by Gauss–Bonnet, `2π − 2γ − 2h₁ cos ρ₁ − 2h₂ cos ρ₂`: `γ` the
    /// angle at a crossing of the triangle (c₁, c₂, x), `h_i` its angle at `c_i`
    /// (`derivations/kfold_coverage_area.mac` §5; against a one-dimensional integral at 60 digits in
    /// `derivations/kfold_coverage_area.gp`). The angles are the half-angle formula's, which keep
    /// their digits when the circles nearly touch or nearly coincide.
    fn lens(d: f64, r1: f64, r2: f64) -> f64 {
        let s = 0.5 * (d + r1 + r2);
        let half = |a: f64, b: f64, c: f64| {
            // the angle opposite side `a`, between sides `b` and `c`
            2.0 * libm::atan2(
                libm::sqrt(libm::sin(s - b) * libm::sin(s - c)),
                libm::sqrt(libm::sin(s) * libm::sin(s - a)),
            )
        };
        let gamma = half(d, r1, r2);
        let h1 = half(r2, d, r1);
        let h2 = half(r1, d, r2);
        2.0 * (PI - gamma - h1 * libm::cos(r1) - h2 * libm::cos(r2))
    }

    /// (G3) One cap: `2π(1 − cos ρ)`, from a hair's breadth to nearly the whole sphere, about
    /// centres on the axes, the diagonals and at random.
    #[test]
    fn one_cap_is_two_pi_one_minus_cos_rho() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0001);
        let mut centres = vec![
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
        ];
        centres.extend((0..5).map(|_| rng.direction()));
        let mut worst = 0.0_f64;
        for rho in [
            1e-6,
            1e-3,
            0.1,
            1.0,
            0.5 * PI,
            2.0,
            3.0,
            PI - 1e-3,
            PI - 1e-6,
        ] {
            for &c in &centres {
                let got = areas(&[SphericalCap {
                    centre: c,
                    half_angle: rho,
                }]);
                worst = spacelab_testkit::nan_max(worst, (got[0] - cap_area(rho)).abs());
            }
        }
        println!("one cap: worst |A_1 - 2 pi (1 - cos rho)| = {worst:.2e}");
        assert!(worst < 4e-15, "{worst:e}");
    }

    /// (G3) Two crossing caps cover their lens twice and the rest once, the lens by Gauss–Bonnet;
    /// small and large caps, each pair also turned to a random place on the sphere.
    #[test]
    fn two_crossing_caps_cover_their_lens_twice() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0002);
        let mut worst = 0.0_f64;
        for (d, r1, r2) in [
            (0.9, 0.7, 0.5),
            (1.5, 1.2, 0.4),
            (2.5, 1.9, 1.6),
            (1.2, 2.0, 1.0),
            (1.2, 0.8, 0.8),
            (2.9, 1.7, 1.5),
        ] {
            let l = lens(d, r1, r2);
            let want = [cap_area(r1) + cap_area(r2) - l, l];
            let caps = pair(d, r1, r2);
            worst = worst.max(worst_gap(&areas(&caps), &want));
            let turn = Rotation::random(&mut rng);
            let turned: Vec<SphericalCap> = caps.iter().map(|c| turn.apply(*c)).collect();
            worst = worst.max(worst_gap(&areas(&turned), &want));
        }
        println!("two caps: worst gap to the lens closed form = {worst:.2e}");
        assert!(worst < 1e-14, "{worst:e}");
    }

    /// A rotation of the sphere, from a random unit quaternion.
    struct Rotation([[f64; 3]; 3]);

    impl Rotation {
        fn random(rng: &mut SplitMix) -> Self {
            let ex = unit(rng.direction());
            let ez = unit(ex.cross(rng.direction()));
            let ey = ez.cross(ex);
            Self([[ex.x, ey.x, ez.x], [ex.y, ey.y, ez.y], [ex.z, ey.z, ez.z]])
        }

        fn apply(&self, cap: SphericalCap) -> SphericalCap {
            let m = &self.0;
            let c = cap.centre;
            SphericalCap {
                centre: Vec3::new(
                    m[0][0] * c.x + m[0][1] * c.y + m[0][2] * c.z,
                    m[1][0] * c.x + m[1][1] * c.y + m[1][2] * c.z,
                    m[2][0] * c.x + m[2][1] * c.y + m[2][2] * c.z,
                ),
                half_angle: cap.half_angle,
            }
        }
    }

    /// (G3) Pairs that do not cross: a cap inside another covers it twice; disjoint caps cover their
    /// sum once; caps whose complements are disjoint cover the sphere once and the rest twice.
    #[test]
    fn nested_disjoint_and_co_disjoint_pairs() {
        let (c1, c2) = (cap_area(1.2), cap_area(0.5));
        let mut worst = worst_gap(&areas(&pair(0.3, 1.2, 0.5)), &[c1, c2]);
        worst = worst.max(worst_gap(&areas(&pair(0.3, 0.5, 1.2)), &[c1, c2]));
        let (c1, c2) = (cap_area(0.7), cap_area(0.9));
        worst = worst.max(worst_gap(&areas(&pair(2.0, 0.7, 0.9)), &[c1 + c2, 0.0]));
        let (c1, c2) = (cap_area(2.0), cap_area(1.8));
        worst = worst.max(worst_gap(
            &areas(&pair(3.0, 2.0, 1.8)),
            &[2.0 * TAU, c1 + c2 - 2.0 * TAU],
        ));
        println!("nested, disjoint, co-disjoint: worst gap = {worst:.2e}");
        assert!(worst < 4e-15, "{worst:e}");
    }

    /// (G3) Three hemispheres meet in the triangle of Girard's theorem, with angles `π − θ_ij`:
    /// `A₃ = 2π − Σθ_ij`, the complement of the union is its antipode, `A₂ = 2π`
    /// (`derivations/kfold_coverage_area.mac` §6; at 60 digits in the `.gp`).
    #[test]
    fn three_hemispheres_meet_in_girards_triangle() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0003);
        let mut worst = 0.0_f64;
        for _ in 0..20 {
            let c: Vec<Vec3> = (0..3).map(|_| rng.direction()).collect();
            let theta = angle(c[0], c[1]) + angle(c[0], c[2]) + angle(c[1], c[2]);
            let caps: Vec<SphericalCap> = c
                .iter()
                .map(|&centre| SphericalCap {
                    centre,
                    half_angle: 0.5 * PI,
                })
                .collect();
            let want = [TAU + theta, TAU, TAU - theta];
            worst = worst.max(worst_gap(&areas(&caps), &want));
        }
        println!("Girard: worst gap = {worst:.2e}");
        assert!(worst < 2e-14, "{worst:e}");
    }

    fn tetrahedron() -> Vec<Vec3> {
        vec![
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ]
    }

    fn octahedron() -> Vec<Vec3> {
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ]
    }

    fn cube() -> Vec<Vec3> {
        (0..8_u8)
            .map(|k| {
                let sign = |bit: u8| if k & bit == 0 { 1.0 } else { -1.0 };
                Vec3::new(sign(1), sign(2), sign(4))
            })
            .collect()
    }

    fn icosahedron() -> Vec<Vec3> {
        let g = 0.5 * (1.0 + libm::sqrt(5.0));
        let mut v = Vec::new();
        for (a, b) in [(1.0, g), (-1.0, g), (1.0, -g), (-1.0, -g)] {
            v.push(Vec3::new(0.0, a, b));
            v.push(Vec3::new(a, b, 0.0));
            v.push(Vec3::new(b, 0.0, a));
        }
        v
    }

    fn caps_at(sites: &[Vec3], half_angle: f64) -> Vec<SphericalCap> {
        sites
            .iter()
            .map(|&centre| SphericalCap { centre, half_angle })
            .collect()
    }

    /// (G3) Caps at the vertices of a regular polyhedron with the face's circumradius cover the
    /// sphere once, touching three or more at every face centre; adjacent caps overlap in lenses and
    /// nothing is covered three times, so `A₂` is the caps' total less `4π`
    /// (`derivations/kfold_coverage_area.mac` §5, where the lenses add up to the same). Tetrahedron
    /// at `acos(1/3)`: `(4π, 4π/3, 0, 0)`; at `π − acos(1/3)`, caps larger than a hemisphere:
    /// `(4π, 4π, 8π/3, 0)`. Octahedron at `acos(1/√3)`: `(4π, 8π − 4√3π, 0, …)`; at `π/2`, three
    /// great circles each shared by complementary caps: `(4π, 4π, 4π, 0, 0, 0)`. Cube at
    /// `acos(1/√3)`, its face-diagonal caps tangent: `(4π, 12π − 16π/√3, 0, …)`. Icosahedron at its
    /// face's circumradius: `(4π, 12·2π(1 − cos ρ) − 4π, 0, …)`.
    #[test]
    fn the_regular_polyhedra_have_their_exact_areas() {
        let four_pi = 2.0 * TAU;
        let r3 = libm::sqrt(3.0);
        let face = |v: &[Vec3], i: [usize; 3]| {
            let m = v[i[0]] + v[i[1]] + v[i[2]];
            libm::acos(m.dot(v[i[0]]) / (m.norm() * v[i[0]].norm()))
        };
        let ico = icosahedron();
        // vertices 0, 3 and 2 are (0, 1, g), (0, -1, g) and (g, 0, 1): one face
        let r_ico = face(&ico, [0, 3, 2]);
        let pad = |mut a: Vec<f64>, n: usize| {
            a.resize(n, 0.0);
            a
        };
        let cases = [
            (
                "tetrahedron at acos(1/3)",
                caps_at(&tetrahedron(), libm::acos(1.0 / 3.0)),
                vec![four_pi, four_pi / 3.0, 0.0, 0.0],
            ),
            (
                "tetrahedron at pi - acos(1/3)",
                caps_at(&tetrahedron(), PI - libm::acos(1.0 / 3.0)),
                vec![four_pi, four_pi, 2.0 * four_pi / 3.0, 0.0],
            ),
            (
                "octahedron at acos(1/sqrt 3)",
                caps_at(&octahedron(), libm::acos(1.0 / r3)),
                pad(vec![four_pi, 2.0 * four_pi - r3 * four_pi], 6),
            ),
            (
                "octahedron at pi/2",
                caps_at(&octahedron(), 0.5 * PI),
                vec![four_pi, four_pi, four_pi, 0.0, 0.0, 0.0],
            ),
            (
                "cube at acos(1/sqrt 3)",
                caps_at(&cube(), libm::acos(1.0 / r3)),
                pad(vec![four_pi, 3.0 * four_pi - 4.0 * four_pi / r3], 8),
            ),
            (
                "icosahedron at its face circumradius",
                caps_at(&ico, r_ico),
                pad(vec![four_pi, 12.0 * cap_area(r_ico) - four_pi], 12),
            ),
        ];
        for (name, caps, want) in cases {
            let got = areas(&caps);
            let gap = worst_gap(&got, &want);
            println!("{name}: worst gap = {gap:.2e}");
            assert!(gap < 2e-14, "{name}: {got:?} against {want:?}");
        }
    }

    // ── The invariant, a second method, the sampler ─────────────────────────────────────────────

    /// (G4) `Σ_k A_k = Σ_i 2π(1 − cos ρ_i)` on random sets of small, mixed and large caps. It holds
    /// for any depth labelling, so it gates the arc set: an arc dropped or counted twice. Beside it,
    /// what any labelling must also meet and a wrong one can break: `4π ≥ A_1 ≥ A_2 ≥ … ≥ 0`.
    #[test]
    fn the_areas_add_up_to_the_caps() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0004);
        for (m, lo, hi) in [
            (12, 0.1, 0.9),
            (12, 0.1, 3.0),
            (66, 0.1, 0.45),
            (300, 0.05, 0.25),
            (40, 1.6, 3.0),
        ] {
            let caps = random_caps(&mut rng, m, lo, hi);
            let got = areas(&caps);
            let sum: f64 = got.iter().sum();
            let want: f64 = caps.iter().map(|c| cap_area(c.half_angle)).sum();
            let gap = (sum - want).abs();
            println!("{m} caps in [{lo}, {hi}]: |sum A_k - sum caps| = {gap:.2e} of {want:.3}");
            assert!(gap < 1e-15 * want.max(1.0) * 8.0, "{gap:e}");
            let slack = 1e-12;
            assert!(got[0] <= 2.0 * TAU + slack, "{got:?}");
            assert!(got[m - 1] >= -slack, "{got:?}");
            assert!(got.windows(2).all(|w| w[0] + slack >= w[1]), "{got:?}");
        }
    }

    /// Sites of a Walker 24/3/1 at 55° at one instant.
    fn walker_sites() -> Vec<Vec3> {
        use crate::walker::{Walker, WalkerPattern};
        use spacelab_core::Angle;
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(55_f64.to_radians()),
            planes: 3,
            per_plane: 8,
            phasing: 1,
            semi_major_axis: 26_560e3,
            mu: 3.986_004_418e14,
        };
        let mut sites = Vec::new();
        w.generate(|e| sites.push(e.propagate(1_234.5).to_state().0));
        sites
    }

    /// (S1) A different method: `A_k = ∫ L_k(θ) sin θ dθ` over colatitude, `L_k` the longitude
    /// covered at least `k` times, by Gauss–Legendre on pieces split at every touching and crossing
    /// colatitude. Random sets of small, mixed and large caps, a Walker instant, and three circles
    /// through one point.
    #[test]
    fn colatitude_quadrature_agrees_with_the_arcs() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0005);
        let mut sets: Vec<(String, Vec<SphericalCap>)> = [
            (5, 0.1, 0.9),
            (12, 0.1, 0.9),
            (12, 0.1, 2.5),
            (30, 1.6, 3.0),
            (66, 0.1, 0.45),
        ]
        .into_iter()
        .map(|(m, lo, hi)| {
            (
                format!("{m} caps in [{lo}, {hi}]"),
                random_caps(&mut rng, m, lo, hi),
            )
        })
        .collect();
        sets.push((
            "Walker 24/3/1 at 0.35 rad".into(),
            caps_at(&walker_sites(), 0.35),
        ));
        sets.push(("three circles through one point".into(), triple_point()));
        for (name, caps) in sets {
            let got = areas(&caps);
            let s1 = colatitude_areas(&caps, rng.direction(), 96);
            let gap = worst_gap(&got, &s1);
            println!("{name}: worst |arcs - quadrature| = {gap:.2e}");
            assert!(gap < 1e-12, "{name}: {got:?} against {s1:?}");
        }
    }

    /// (S2) The fraction of uniform random points covered at least `k` times is within five of its
    /// binomial standard deviations of `A_k/4π`, for every `k`.
    #[test]
    fn the_sampled_sphere_agrees_within_its_error_for_every_k() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0006);
        let sets = [
            random_caps(&mut rng, 12, 0.1, 0.9),
            random_caps(&mut rng, 30, 0.2, 1.2),
            random_caps(&mut rng, 8, 1.6, 3.0),
            caps_at(&walker_sites(), 0.35),
        ];
        let n = 200_000_u32;
        for (s, caps) in sets.iter().enumerate() {
            let got = areas(caps);
            let sampled = sampled_areas(caps, n, 0x5A3F + s as u64);
            let mut worst_z = 0.0_f64;
            for (k, (&exact, &(estimate, _))) in got.iter().zip(&sampled).enumerate() {
                let p = (exact / (2.0 * TAU)).clamp(0.0, 1.0);
                let sigma = 2.0 * TAU * libm::sqrt(p * (1.0 - p) / f64::from(n));
                let gap = (estimate - exact).abs();
                assert!(
                    gap <= 5.0 * sigma + 1e-12,
                    "set {s}, k = {}: sampled {estimate} against {exact} (sigma {sigma:e})",
                    k + 1
                );
                if sigma > 0.0 {
                    worst_z = worst_z.max(gap / sigma);
                }
            }
            println!("set {s}: largest |sampled - exact| / sigma over k = {worst_z:.2}");
        }
    }

    /// Three caps whose circles pass through the pole `(0, 0, 1)`: half-angles 1/2, 11/16, 5/8 at
    /// azimuths 0, 2, 4 rad, as in `derivations/kfold_coverage_area.gp`.
    fn triple_point() -> Vec<SphericalCap> {
        [(0.5, 0.0), (11.0 / 16.0, 2.0), (10.0 / 16.0, 4.0)]
            .into_iter()
            .map(|(r, az): (f64, f64)| {
                cap(
                    libm::sin(r) * libm::cos(az),
                    libm::sin(r) * libm::sin(az),
                    libm::cos(r),
                    r,
                )
            })
            .collect()
    }

    /// Twelve caps with integer centres and half-angles in sixteenths, one larger than a hemisphere,
    /// exact in binary64, and their areas at 60 digits (`derivations/kfold_coverage_area.gp`).
    #[expect(
        clippy::excessive_precision,
        reason = "23 digits as derivations/kfold_coverage_area.gp prints them"
    )]
    fn twelve_caps() -> (Vec<SphericalCap>, Vec<f64>) {
        let twelve: Vec<SphericalCap> = [
            ([3, 1, 2], 12),
            ([-1, 4, 1], 10),
            ([2, -3, 1], 14),
            ([-2, -1, 3], 9),
            ([1, 2, -4], 13),
            ([-3, 2, -1], 11),
            ([4, -1, -2], 15),
            ([0, -2, -3], 8),
            ([-4, -3, 1], 12),
            ([2, 4, 3], 16),
            ([1, -4, 4], 10),
            ([-1, 1, -1], 32),
        ]
        .into_iter()
        .map(|(c, r): ([i32; 3], i32)| {
            cap(
                f64::from(c[0]),
                f64::from(c[1]),
                f64::from(c[2]),
                f64::from(r) / 16.0,
            )
        })
        .collect();
        let mut want = vec![
            1.253_495_720_079_655_397_687_8e1,
            1.102_525_104_558_359_198_908_9e1,
            3.745_051_614_872_408_539_649_6,
            1.755_771_533_122_061_516_078_3e-1,
        ];
        want.resize(12, 0.0);
        (twelve, want)
    }

    /// The f64 code against the same algorithm at 60 digits in `derivations/kfold_coverage_area.gp`,
    /// whose values a different method there, the colatitude integral, reproduces to 1.7e-55 and
    /// 7.6e-66: round-off, on twelve caps with integer centres and half-angles in sixteenths (one
    /// larger than a hemisphere), and on three circles through one point, whose centres are exact
    /// there and rounded to binary64 here, so its gap holds both.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "23 digits as derivations/kfold_coverage_area.gp prints them"
    )]
    fn the_f64_areas_hold_the_60_digit_values() {
        let (twelve, want) = twelve_caps();
        let gap = worst_gap(&areas(&twelve), &want);
        println!("twelve caps: worst |f64 - 60 digits| = {gap:.2e}");
        assert!(gap < 1e-14, "{gap:e}");
        let want = [
            3.130_303_349_535_592_169_239_9,
            2.539_419_776_167_129_666_451_8e-1,
            0.0,
        ];
        let gap = worst_gap(&areas(&triple_point()), &want);
        println!("triple point: worst |f64 - 60 digits| = {gap:.2e}");
        assert!(gap < 1e-14, "{gap:e}");
    }

    // ── Degenerate input ─────────────────────────────────────────────────────────────────────────

    /// Caps on one circle each count once: two equal caps cover their cap twice (the M0-1 prototype
    /// gave twice the cap once and nothing twice, V2); so do three, which are merged onto one circle
    /// at no move; a cap and its complement cover the sphere once; three equal caps and a complement
    /// cover it once and the cap three times; a crossing cap adds its lens to each level.
    #[test]
    fn caps_on_one_circle_count_once_each() {
        let four_pi = 2.0 * TAU;
        let c = cap_area(0.5);
        let a = cap(0.3, -0.2, 0.9, 0.5);
        let complement = SphericalCap {
            centre: -a.centre,
            half_angle: PI - 0.5,
        };
        let crossing = cap(0.9, 0.2, 0.4, 0.6);
        let d = angle(unit(a.centre), unit(crossing.centre));
        let l = lens(d, 0.5, 0.6);
        let cc = cap_area(0.6);
        let cases: [(&str, Vec<SphericalCap>, Vec<f64>); 6] = [
            ("two equal caps", vec![a, a], vec![c, c]),
            ("three equal caps", vec![a, a, a], vec![c, c, c]),
            (
                "a cap and its complement",
                vec![a, complement],
                vec![four_pi, 0.0],
            ),
            (
                "three equal caps and a complement",
                vec![a, complement, a, a],
                vec![four_pi, c, c, 0.0],
            ),
            (
                "two equal caps and a crossing one",
                vec![a, crossing, a],
                vec![c + cc - l, c, l],
            ),
            (
                "a cap, its complement and a crossing one",
                vec![a, complement, crossing],
                vec![four_pi, cc, 0.0],
            ),
        ];
        for (name, caps, want) in cases {
            let result = kfold_areas(&caps).unwrap();
            let gap = worst_gap(&result.at_least, &want);
            println!(
                "{name}: worst gap = {gap:.2e}, snap bound {:.1e}",
                result.snap_bound
            );
            assert!(
                gap < 4e-15,
                "{name}: {:?} against {want:?}",
                result.at_least
            );
            assert!(result.snap_bound < 1e-15, "{name}: {}", result.snap_bound);
        }
    }

    /// Circles `10⁻ᵐ` apart, `m = 1, …, 16`: concentric caps whose half-angles differ by `δ`
    /// (`A₁` the larger cap, `A₂` the smaller), and equal caps whose centres are `δ` apart (`A₂` the
    /// lens). Three caps `δ` apart in a chain are merged onto one circle once `δ ≤ 2⁻²⁸`, and stay
    /// within the band they span plus [`KfoldAreas::snap_bound`].
    #[test]
    fn circles_ten_to_the_minus_m_apart() {
        let rho = 0.6;
        let mut worst = 0.0_f64;
        for m in 1..=16 {
            let delta = 10_f64.powi(-m);
            let concentric = [cap(0.0, 0.0, 1.0, rho), cap(0.0, 0.0, 1.0, rho + delta)];
            let gap = worst_gap(&areas(&concentric), &[cap_area(rho + delta), cap_area(rho)]);
            worst = worst.max(gap);
            let l = lens(delta, rho, rho);
            let gap = worst_gap(
                &areas(&pair(delta, rho, rho)),
                &[2.0 * cap_area(rho) - l, l],
            );
            worst = worst.max(gap);
            let chain: Vec<SphericalCap> = (0..3)
                .map(|k| {
                    let t = f64::from(k) * delta;
                    cap(libm::sin(t), 0.0, libm::cos(t), rho)
                })
                .collect();
            let result = kfold_areas(&chain).unwrap();
            let band = 2.0 * TAU * libm::sin(rho) * 2.0 * delta;
            for (k, &a) in result.at_least.iter().enumerate() {
                assert!(
                    (a - cap_area(rho)).abs() <= band + result.snap_bound + 1e-14,
                    "m = {m}, k = {}: {a} is not within {band:e} of the cap",
                    k + 1
                );
            }
            let snapped = result.snap_bound > 0.0;
            assert_eq!(snapped, delta <= COINCIDENT_ANGLE, "m = {m}");
        }
        println!("circles 1e-m apart, m = 1..16: worst gap to the closed forms = {worst:.2e}");
        assert!(worst < 4e-15, "{worst:e}");
    }

    /// Tangent caps share no area: externally, internally, and complements touching; and caps that
    /// overlap by `δ = 10⁻ᵐ` past a tangency share the lens, `m = 1, …, 15`.
    #[test]
    fn tangent_caps_share_no_area_and_near_tangent_ones_their_lens() {
        let four_pi = 2.0 * TAU;
        let mut worst = worst_gap(
            &areas(&pair(1.2, 0.7, 0.5)),
            &[cap_area(0.7) + cap_area(0.5), 0.0],
        );
        worst = worst.max(worst_gap(
            &areas(&pair(0.4, 0.7, 0.3)),
            &[cap_area(0.7), cap_area(0.3)],
        ));
        worst = worst.max(worst_gap(
            &areas(&pair(TAU - 4.5, 2.0, 2.5)),
            &[four_pi, cap_area(2.0) + cap_area(2.5) - four_pi],
        ));
        println!("tangent pairs: worst gap = {worst:.2e}");
        assert!(worst < 4e-15, "{worst:e}");
        let mut worst = 0.0_f64;
        for m in 1..=15 {
            let delta = 10_f64.powi(-m);
            let d = 1.2 - delta;
            let l = lens(d, 0.7, 0.5);
            let want = [cap_area(0.7) + cap_area(0.5) - l, l];
            worst = worst.max(worst_gap(&areas(&pair(d, 0.7, 0.5)), &want));
        }
        println!("overlap 1e-m past tangency: worst gap = {worst:.2e}");
        assert!(worst < 4e-15, "{worst:e}");
    }

    /// Three great circles through `±z`, the hemispheres' centres in the `x`–`y` plane: each lune of
    /// dihedral angle `Δφ` has area `2Δφ`, so `A_k` is twice the longitude covered `k` times.
    #[test]
    fn three_great_circles_through_one_point() {
        let az = [0.3, 1.7, 4.0];
        let caps: Vec<SphericalCap> = az
            .iter()
            .map(|&a: &f64| cap(libm::cos(a), libm::sin(a), 0.0, 0.5 * PI))
            .collect();
        // the longitude covered at least k times, from the six ends a_i ± π/2
        let mut ends: Vec<f64> = az
            .iter()
            .flat_map(|&a| {
                [
                    (a - 0.5 * PI).rem_euclid(TAU),
                    (a + 0.5 * PI).rem_euclid(TAU),
                ]
            })
            .collect();
        ends.sort_by(f64::total_cmp);
        let mut want = [0.0; 3];
        for k in 0..ends.len() {
            let (a, b) = (
                ends[k],
                if k + 1 < ends.len() {
                    ends[k + 1]
                } else {
                    ends[0] + TAU
                },
            );
            let mid = 0.5 * (a + b);
            let depth = az.iter().filter(|&&c| libm::cos(mid - c) > 0.0).count();
            for w in &mut want[..depth] {
                *w += 2.0 * (b - a);
            }
        }
        let gap = worst_gap(&areas(&caps), &want);
        println!("three great circles through one point: worst gap = {gap:.2e}");
        assert!(gap < 2e-14, "{gap:e}");
    }

    /// The complements `(−c_i, π − ρ_i)` of `m` caps cover a point `m − d` times where the caps
    /// cover it `d` times, so `A_k` of the complements is `4π − A_{m+1−k}` of the caps: small caps
    /// against caps larger than a hemisphere.
    #[test]
    fn complements_mirror_the_levels() {
        let mut rng = SplitMix(0x5EED_31D0_0000_0007);
        let mut worst = 0.0_f64;
        for (m, lo, hi) in [(6, 0.2, 0.9), (15, 0.1, 1.4), (30, 0.3, 0.6)] {
            let caps = random_caps(&mut rng, m, lo, hi);
            let complements: Vec<SphericalCap> = caps
                .iter()
                .map(|c| SphericalCap {
                    centre: -c.centre,
                    half_angle: PI - c.half_angle,
                })
                .collect();
            let a = areas(&caps);
            let b = areas(&complements);
            let mirrored: Vec<f64> = (0..m).map(|k| 2.0 * TAU - a[m - 1 - k]).collect();
            worst = worst.max(worst_gap(&b, &mirrored));
        }
        println!("complements: worst gap to 4 pi - A_(m+1-k) = {worst:.2e}");
        assert!(worst < 5e-14, "{worst:e}");
    }

    /// The exact covering radius of `spacelab::analysis` and the exact area agree: caps of the upper
    /// end cover the sphere `k` times, and caps `10⁻⁴` rad short of the lower end leave a hole.
    #[test]
    fn the_covering_radius_covers_and_less_does_not() {
        let sites = walker_sites();
        for fold in 1..=2_u32 {
            let k = fold as usize;
            let alpha = crate::analysis::covering_radius(&sites, fold).unwrap();
            let full = areas(&caps_at(&sites, alpha.upper));
            assert!(
                (full[k - 1] - 2.0 * TAU).abs() < 1e-13,
                "k = {k}: {}",
                full[k - 1]
            );
            let short = areas(&caps_at(&sites, alpha.lower - 1e-4));
            assert!(
                short[k - 1] < 2.0 * TAU - 1e-10,
                "k = {k}: {}",
                short[k - 1]
            );
            println!(
                "k = {k}: alpha = {:.12}; hole at alpha - 1e-4 = {:.3e} sr",
                alpha.upper,
                2.0 * TAU - short[k - 1]
            );
        }
    }

    /// (S2) Input outside the domain is refused with the offending cap's index; no caps give no
    /// areas.
    #[test]
    fn bad_input_is_refused() {
        let good = cap(0.0, 0.0, 1.0, 0.5);
        for (bad, err) in [
            (cap(f64::NAN, 0.0, 1.0, 0.5), CapError::BadCentre { cap: 1 }),
            (
                cap(f64::INFINITY, 0.0, 1.0, 0.5),
                CapError::BadCentre { cap: 1 },
            ),
            (cap(0.0, 0.0, 0.0, 0.5), CapError::BadCentre { cap: 1 }),
            (cap(0.0, 0.0, 1.0, 0.0), CapError::BadHalfAngle { cap: 1 }),
            (cap(0.0, 0.0, 1.0, -0.1), CapError::BadHalfAngle { cap: 1 }),
            (cap(0.0, 0.0, 1.0, PI), CapError::BadHalfAngle { cap: 1 }),
            (cap(0.0, 0.0, 1.0, 4.0), CapError::BadHalfAngle { cap: 1 }),
            (
                cap(0.0, 0.0, 1.0, f64::NAN),
                CapError::BadHalfAngle { cap: 1 },
            ),
            (
                cap(0.0, 0.0, 1.0, f64::INFINITY),
                CapError::BadHalfAngle { cap: 1 },
            ),
        ] {
            assert_eq!(kfold_areas(&[good, bad]), Err(err), "{bad:?}");
        }
        let none = kfold_areas(&[]).unwrap();
        assert_eq!(none.at_least, [] as [f64; 0]);
        assert_eq!(none.snap_bound, 0.0);
    }

    /// Rule 16's kept negative case: the invariant holds for any depth labelling, so it cannot see
    /// the M0-1 prototype's wrong one on two equal caps (V2: `A₁` twice the cap, `A₂ = 0`); the
    /// closed form `(cap, cap)` does, and the shipped code meets it.
    #[test]
    fn the_invariant_cannot_see_the_prototypes_labels_and_the_closed_form_does() {
        let a = cap(0.3, -0.2, 0.9, 0.5);
        let c = cap_area(0.5);
        let proto = midpoint_labelled_areas(&[a, a]);
        assert!((proto[0] + proto[1] - 2.0 * c).abs() < 1e-14, "{proto:?}");
        assert!(
            (proto[0] - 2.0 * c).abs() < 1e-14 && proto[1].abs() < 1e-14,
            "{proto:?}"
        );
        let got = areas(&[a, a]);
        assert!(worst_gap(&got, &[c, c]) < 4e-15, "{got:?}");
        println!("two equal caps of 0.5 rad: prototype {proto:?}, shipped {got:?}, cap {c}");
    }

    /// Circles `δ` apart that pass through one point cross there at an angle of order `δ`, against
    /// the 60-digit values of `derivations/kfold_coverage_area.gp`, where the colatitude integral
    /// agrees to 1e-30: four circles `10⁻⁸` apart are left apart and stay within `2⁻⁵³/δ`; three
    /// `10⁻¹²` apart are moved onto one circle and stay within the snap bound, and the areas still
    /// fall with `k`.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the inputs and values as derivations/kfold_coverage_area.gp states them"
    )]
    fn near_coincident_circles_through_one_point() {
        let apart = [
            cap(
                -8.760_328_444_897_426e-1,
                -2.458_666_227_100_355e-1,
                4.148_687_252_762_870_4e-1,
                8.736_044_060_868_489e-1,
            ),
            cap(
                -8.760_328_451_510_717e-1,
                -2.458_666_193_293_822_5e-1,
                4.148_687_258_833_314e-1,
                8.736_044_091_587_652e-1,
            ),
            cap(
                -8.760_328_449_406_597e-1,
                -2.458_666_284_711_224_8e-1,
                4.148_687_209_099_009_6e-1,
                8.736_043_988_481_17e-1,
            ),
            cap(
                -8.760_328_443_534_356e-1,
                -2.458_666_313_321_250_8e-1,
                4.148_687_204_543_413e-1,
                8.736_043_962_833_798e-1,
            ),
        ];
        let moved = [
            cap(
                -7.977_735_752_844_216e-1,
                -5.192_310_457_563_937e-1,
                3.065_231_536_126_320_5e-1,
                9.655_900_539_024_349e-1,
            ),
            cap(
                -7.977_735_752_842_509e-1,
                -5.192_310_457_563_492e-1,
                3.065_231_536_131_519_7e-1,
                9.655_900_539_025_168e-1,
            ),
            cap(
                -7.977_735_752_838_844e-1,
                -5.192_310_457_572_321e-1,
                3.065_231_536_126_101e-1,
                9.655_900_539_014_648e-1,
            ),
        ];
        let result = kfold_areas(&apart).unwrap();
        let want = [
            2.248_956_788_970_055_999_099_2,
            2.248_956_773_793_924_097_037_6,
            2.248_956_739_225_869_225_396_5,
            2.248_956_726_502_549_438_311_5,
        ];
        let gap = worst_gap(&result.at_least, &want);
        let allowed = 0.5 * f64::EPSILON / 1e-8;
        println!(
            "four circles 1e-8 apart: worst gap {gap:.2e}, allowed 2^-53/delta = {allowed:.2e}"
        );
        assert_eq!(result.snap_bound, 0.0);
        assert!(gap <= allowed, "{:?}", result.at_least);
        let result = kfold_areas(&moved).unwrap();
        let want = [
            2.708_481_857_876_315_593_662_9,
            2.708_481_857_874_510_849_671_5,
            2.708_481_857_870_156_499_279_7,
        ];
        let gap = worst_gap(&result.at_least, &want);
        println!(
            "three circles 1e-12 apart: worst gap {gap:.2e}, snap bound {:.2e}",
            result.snap_bound
        );
        assert!(gap <= result.snap_bound, "{:?}", result.at_least);
        assert!(
            result.at_least.windows(2).all(|w| w[0] >= w[1]),
            "{:?}",
            result.at_least
        );
    }

    /// A cap within a few ulps of `π` and a circle through its hole: the circle crosses the hole's
    /// circle where the two nearly touch, so its arc outside the large cap is a few `10⁻¹⁶` long and
    /// its two ends can round to one point or past each other. The areas are the whole sphere once
    /// and the small cap twice, to the hole's area of about `10⁻³⁰` sr, in 4000 random frames.
    #[test]
    fn a_circle_through_the_hole_of_a_cap_an_ulp_short_of_pi() {
        let four_pi = 2.0 * TAU;
        let mut rng = SplitMix(0x5EED_31D0_0000_0008);
        let (mut worst, mut wrong) = (0.0_f64, 0);
        for trial in 0..4000_u64 {
            let rot = Rotation::random(&mut rng);
            let rho_j = f64::from_bits(PI.to_bits() - (1 + trial % 8));
            let rho_i = 0.05 + 3.0 * rng.next();
            // the circle's centre `rho_i` from the hole's, give or take 2e-15 rad
            let t = rho_i + (rng.next() - 0.5) * 4e-15;
            let caps = [
                rot.apply(cap(libm::sin(t), 0.0, libm::cos(t), rho_i)),
                rot.apply(cap(0.0, 0.0, -1.0, rho_j)),
            ];
            let gap = worst_gap(&areas(&caps), &[four_pi, cap_area(rho_i)]);
            if gap > 1e-13 {
                wrong += 1;
            }
            worst = worst.max(gap);
        }
        println!(
            "hole an ulp short of pi: {wrong} of 4000 frames off by more than 1e-13, worst {worst:.2e}"
        );
        assert_eq!(wrong, 0, "worst {worst:e}");
    }

    /// The near-coincident corpus of `derivations/kfold_coverage_area.gp`, section (f): per set its
    /// decade, the binary64 caps, and the areas at 60 digits.
    fn corpus() -> Vec<(i32, Vec<SphericalCap>, Vec<f64>)> {
        const CORPUS: &str = include_str!("../tests/fixtures/kfold_near_coincident.txt");
        let mut lines = CORPUS.lines().filter(|l| !l.starts_with('#'));
        let mut sets = Vec::new();
        while let Some(head) = lines.next() {
            let head: Vec<&str> = head.split_whitespace().collect();
            assert_eq!(head[0], "set");
            let decade: i32 = head[1].parse().unwrap();
            let m: usize = head[2].parse().unwrap();
            let caps: Vec<SphericalCap> = (0..m)
                .map(|_| {
                    let t: Vec<i64> = lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .skip(1)
                        .map(|w| w.parse().unwrap())
                        .collect();
                    #[expect(clippy::cast_precision_loss, reason = "|M| < 2^53 is exact")]
                    let x = |k: usize| {
                        libm::scalbn(t[2 * k] as f64, i32::try_from(t[2 * k + 1]).unwrap())
                    };
                    cap(x(0), x(1), x(2), x(3))
                })
                .collect();
            let want: Vec<f64> = lines
                .next()
                .unwrap()
                .split_whitespace()
                .skip(1)
                .map(|w| w.parse().unwrap())
                .collect();
            sets.push((decade, caps, want));
        }
        sets
    }

    /// The near-coincident corpus of `derivations/kfold_coverage_area.gp`, section (f): 336 sets of
    /// three or four circles through one point, `10⁻⁶ … 10⁻¹²` apart, as binary64 inputs with their
    /// areas at 60 digits. A set's `δ` is its largest `angle(c_i, c_j) + |ρ_i − ρ_j|`. The shipped code
    /// stays within [`KfoldAreas::snap_bound`] where it moved caps and within `4 · 2⁻⁵³/δ` where it did
    /// not. For the docs the test prints, per decade, the worst errors of never moving (against
    /// `2⁻⁵³/δ`) and of always moving (against `δ` and the bound), whose crossing sets
    /// [`COINCIDENT_ANGLE`]; always moving stays within the bound in every set.
    #[test]
    fn near_coincident_corpus_within_its_bounds() {
        let eps = 0.5 * f64::EPSILON;
        // per decade: never moved, err·δ/2⁻⁵³; always moved, err/δ and err/bound; shipped, worst
        // error and worst error over its allowance; sets the shipped code moved; sets
        let mut table = std::collections::BTreeMap::<i32, [f64; 7]>::new();
        for (decade, caps, want) in corpus() {
            let m = caps.len();
            let mut delta = 0.0_f64;
            for i in 0..m {
                for j in (i + 1)..m {
                    let (a, b) = (caps[i], caps[j]);
                    delta = delta.max(
                        angle(unit(a.centre), unit(b.centre)) + (a.half_angle - b.half_angle).abs(),
                    );
                }
            }
            let shipped = kfold_areas(&caps).unwrap();
            let apart = areas_with(&caps, 0.0).unwrap();
            let moved = areas_with(&caps, 1e-4).unwrap();
            assert!(
                apart.snap_bound == 0.0 && moved.snap_bound > 0.0,
                "{caps:?}"
            );
            let (e_ship, e_apart, e_moved) = (
                worst_gap(&shipped.at_least, &want),
                worst_gap(&apart.at_least, &want),
                worst_gap(&moved.at_least, &want),
            );
            let allowance = if shipped.snap_bound > 0.0 {
                shipped.snap_bound
            } else {
                4.0 * eps / delta
            };
            assert!(
                e_ship <= allowance,
                "shipped {e_ship:e} over {allowance:e}: {caps:?}"
            );
            assert!(e_moved <= moved.snap_bound, "moved {e_moved:e}: {caps:?}");
            let row = table.entry(decade).or_insert([0.0; 7]);
            for (slot, value) in [
                e_apart * delta / eps,
                e_moved / delta,
                e_moved / moved.snap_bound,
                e_ship,
                e_ship / allowance,
            ]
            .into_iter()
            .enumerate()
            {
                row[slot] = row[slot].max(value);
            }
            row[5] += if shipped.snap_bound > 0.0 { 1.0 } else { 0.0 };
            row[6] += 1.0;
        }
        println!("decade   never moved   always moved          shipped");
        println!("         err.d/2^-53   err/d   err/bound     worst err  /allowance  moved");
        for (decade, r) in &table {
            println!(
                "1e-{decade:<5} {:>10.2}  {:>7.2}  {:>9.3}  {:>12.2e}  {:>10.3}  {:>2}/{}",
                r[0], r[1], r[2], r[3], r[4], r[5], r[6]
            );
        }
        assert_eq!(table.values().map(|r| r[6]).sum::<f64>(), 336.0);
    }

    /// Tangent and nearly tangent pairs in random frames: caps touching outside, one inside another,
    /// complements touching, each exact and `10⁻⁸…10⁻¹⁵` either side of the tangency, and caps of
    /// `10⁻⁴…10⁻¹³` rad nearly touching, against their closed forms.
    #[test]
    fn tangencies_hold_in_every_frame() {
        let mut rng = SplitMix(2026);
        let four_pi = 2.0 * TAU;
        let mut worst = [0.0_f64; 4];
        for trial in 0..900_i32 {
            let rot = Rotation::random(&mut rng);
            let turned = |caps: Vec<SphericalCap>| -> Vec<f64> {
                areas(&caps.into_iter().map(|c| rot.apply(c)).collect::<Vec<_>>())
            };
            let (r1, r2) = (0.05 + 1.5 * rng.next(), 0.05 + 1.5 * rng.next());
            let (lo, hi) = if r1 < r2 { (r1, r2) } else { (r2, r1) };
            let m = trial % 9;
            let sign = if trial % 2 == 0 { 1.0 } else { -1.0 };
            let eta = if m == 0 {
                0.0
            } else {
                sign * 10_f64.powi(-(7 + m))
            };
            let crossing = |d: f64, a: f64, b: f64| {
                let l = lens(d, a, b);
                [cap_area(a) + cap_area(b) - l, l]
            };
            let d = r1 + r2 - eta;
            let want = if eta <= 0.0 {
                [cap_area(r1) + cap_area(r2), 0.0]
            } else {
                crossing(d, r1, r2)
            };
            worst[0] = worst[0].max(worst_gap(&turned(pair(d, r1, r2)), &want));
            let d = hi - lo + eta;
            let want = if eta <= 0.0 {
                [cap_area(hi), cap_area(lo)]
            } else {
                crossing(d, lo, hi)
            };
            worst[1] = worst[1].max(worst_gap(&turned(pair(d, lo, hi)), &want));
            let (c1, c2) = (PI - lo, PI - 0.5 * hi);
            let d = TAU - c1 - c2 - eta;
            let want = if eta <= 0.0 {
                [four_pi, cap_area(c1) + cap_area(c2) - four_pi]
            } else {
                crossing(d, c1, c2)
            };
            worst[2] = worst[2].max(worst_gap(&turned(pair(d, c1, c2)), &want));
            let rho = 10_f64.powi(-(4 + trial % 10));
            let d = 2.0 * rho * (1.0 - 10_f64.powi(-(1 + trial % 13)));
            worst[3] = worst[3].max(worst_gap(
                &turned(pair(d, rho, rho)),
                &crossing(d, rho, rho),
            ));
        }
        println!(
            "worst gap: touching outside {:.2e}, inside {:.2e}, complements {:.2e}, tiny caps {:.2e}",
            worst[0], worst[1], worst[2], worst[3]
        );
        assert!(worst.iter().all(|w| *w < 1e-14), "{worst:?}");
    }

    // ── The enclosure (plan 18 Ф28.V) ────────────────────────────────────────────────────────────

    /// The tightest margin and the widest interval of `r` against references printed to more digits
    /// than a double holds; a reference within one ulp of an end counts as held, for the rounding of
    /// its decimal. `None` when one is outside.
    fn enclosure_margin(r: &KfoldAreas, want: &[f64]) -> Option<(f64, f64)> {
        let (mut margin, mut width) = (f64::INFINITY, 0.0_f64);
        for ((&lo, &hi), &w) in r.lower.iter().zip(&r.upper).zip(want) {
            if lo > w.next_up() || hi < w.next_down() {
                return None;
            }
            margin = margin.min((w - lo).min(hi - w));
            width = width.max(hi - lo);
        }
        Some((margin, width))
    }

    /// (S2) The enclosures hold the 60-digit areas: the twelve caps, exact in binary64, and the 336
    /// near-coincident sets of the corpus, whose pairs within [`COINCIDENT_ANGLE`] the enclosure
    /// moves onto one circle and pays for with their band. The margins and widths are printed.
    #[test]
    fn the_enclosure_holds_the_60_digit_areas() {
        let (twelve, want) = twelve_caps();
        let r = kfold_areas(&twelve).unwrap();
        let Some((margin, width)) = enclosure_margin(&r, &want) else {
            panic!("twelve caps: {:?} {:?} against {want:?}", r.lower, r.upper);
        };
        println!("twelve caps: tightest margin {margin:.2e}, widest {width:.2e}");
        // First order in the ends' error, and at the largest |dω/dψ| along each circle: about 500
        // times the round-off of `at_least` here (module docs).
        assert!(width < 1e-10, "{width:e}");
        let (mut margin, mut width) = (f64::INFINITY, 0.0_f64);
        for (decade, caps, want) in corpus() {
            let r = kfold_areas(&caps).unwrap();
            let Some((m, w)) = enclosure_margin(&r, &want) else {
                panic!(
                    "1e-{decade}: {caps:?}: {:?} {:?} against {want:?}",
                    r.lower, r.upper
                );
            };
            margin = margin.min(m);
            width = width.max(w);
        }
        println!("near-coincident corpus: tightest margin {margin:.2e}, widest {width:.2e}");
    }

    /// Rule 16 at the level of [`kfold_areas`]: one cap's enclosure, which carries no sweep
    /// allowance, holds its area `2π(1 − cos ρ)`, given as a double-double `hi + lo` (to 1e-32) by
    /// `derivations/binary64.gp`'s `f64` at 60 digits. Under the planted round to nearest of
    /// `outward::NEAREST` the enclosure shrinks to the one-ulp width of `2π` between doubles, and the
    /// same check, over the same three caps, refutes it.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "an end equal to `hi` is decided by the sign of `lo`"
    )]
    fn one_caps_enclosure_holds_its_area_and_the_planted_rounding_is_refuted() {
        let rows = [
            (0.5, 0.769_171_448_472_976_5, 5.062_423_379_803_522_5e-17),
            (1.5, 5.838_730_360_990_076, 1.817_012_926_745_579e-16),
            (3.0, 12.503_491_616_037_541, 7.542_614_871_023_817e-16),
        ];
        let holds = |r: &KfoldAreas, hi: f64, lo: f64| {
            let (lower, upper) = (r.lower[0], r.upper[0]);
            (lower < hi || (lower == hi && lo > 0.0)) && (upper > hi || (upper == hi && lo < 0.0))
        };
        let mut refuted = 0;
        for (rho, hi, lo) in rows {
            let caps = [cap(0.0, 0.0, 1.0, rho)];
            let r = kfold_areas(&caps).unwrap();
            outward::NEAREST.with(|p| p.set(true));
            let planted = kfold_areas(&caps);
            outward::NEAREST.with(|p| p.set(false));
            let planted = planted.unwrap();
            println!(
                "rho {rho}: [{:e}, {:e}]; planted [{:e}, {:e}]; exact {hi:e} + {lo:e}",
                r.lower[0], r.upper[0], planted.lower[0], planted.upper[0]
            );
            assert!(holds(&r, hi, lo), "rho {rho} misses its area");
            refuted += usize::from(!holds(&planted, hi, lo));
        }
        assert!(refuted > 0, "the planted rounding holds every area");
    }

    /// The enclosures of degenerate sets hold their closed forms, each itself an outward interval:
    /// equal caps; a cap and its complement, which `PI − 0.5` makes overlap by a sliver or miss by
    /// one, both inside the closed form; concentric caps `10⁻ᵐ` apart; and a circle through the
    /// hole of a cap an ulp short of π, whose areas are the circle's cap and 4π to within the hole.
    #[test]
    fn the_enclosure_holds_closed_forms_at_degenerate_sets() {
        let area = |rho: Iv| {
            let s = outward::sin(rho * Iv::point(0.5));
            Iv::point(2.0) * outward::TAU * s.sq()
        };
        let four_pi = Iv::point(2.0) * outward::TAU;
        let mut widest = 0.0_f64;
        let mut check = |name: &str, caps: &[SphericalCap], want: &[Iv]| {
            let r = kfold_areas(caps).unwrap();
            for (k, w) in want.iter().enumerate() {
                assert!(
                    r.lower[k] <= w.hi && w.lo <= r.upper[k],
                    "{name}, k = {}: [{}, {}] misses {w:?}",
                    k + 1,
                    r.lower[k],
                    r.upper[k]
                );
                widest = widest.max(r.upper[k] - r.lower[k]);
            }
        };
        let base = cap(0.3, -0.2, 0.9, 0.5);
        let base_area = area(Iv::point(0.5));
        check("two equal caps", &[base, base], &[base_area, base_area]);
        check(
            "three equal caps",
            &[base, base, base],
            &[base_area, base_area, base_area],
        );
        let rho_c = PI - 0.5;
        let complement = SphericalCap {
            centre: -base.centre,
            half_angle: rho_c,
        };
        // Coaxial: the caps leave the band cos 0.5 + cos ρ_c > 0 uncovered, or overlap where it is
        // negative.
        let band = outward::TAU * (outward::cos(Iv::point(0.5)) + outward::cos(Iv::point(rho_c)));
        check(
            "a cap and its complement",
            &[base, complement],
            &[four_pi - band.max0(), (-band).max0()],
        );
        let polar = cap(0.0, 0.0, 1.0, 0.6);
        for m in 1..=16 {
            let rho = 0.6 + 10_f64.powi(-m);
            let wider = SphericalCap {
                half_angle: rho,
                ..polar
            };
            check(
                &format!("concentric 1e-{m}"),
                &[polar, wider],
                &[area(Iv::point(rho)), area(Iv::point(0.6))],
            );
        }
        let mut rng = SplitMix(0x5EED_F28E_0000_0002);
        for k in 1..=8_u64 {
            let rho_j = f64::from_bits(PI.to_bits() - k);
            let hole = area(outward::PI - Iv::point(rho_j));
            let rho_i = 0.05 + 3.0 * rng.next();
            let t = rho_i + (rng.next() - 0.5) * 4e-15;
            let ci = area(Iv::point(rho_i));
            let below = |x: Iv| Iv {
                lo: (x - hole).lo,
                hi: x.hi,
            };
            check(
                &format!("hole {k} ulp short of pi"),
                &[
                    cap(libm::sin(t), 0.0, libm::cos(t), rho_i),
                    cap(0.0, 0.0, -1.0, rho_j),
                ],
                &[below(four_pi), below(ci)],
            );
        }
        println!("degenerate closed forms: widest enclosure {widest:.2e}");
    }
}
