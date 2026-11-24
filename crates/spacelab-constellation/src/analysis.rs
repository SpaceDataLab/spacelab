//! Coverage analysis for constellations — the **α-characteristic** (Mozhaev), the design-time figure
//! of merit a delta-system is chosen by.
//!
//! The α-characteristic of `k`-fold coverage, `α_k`, is the smallest sub-satellite zone angular
//! radius at which every point of the sphere is within `α_k` of at least `k` sub-satellite points at
//! every instant: the worst case over space and time of `d_k(x, t)`, the `k`-th smallest angular
//! distance from `x` to the satellites. A smaller value is better.
//!
//! **Space: exact, from a finite candidate set.** At a fixed instant `d_k` is an order statistic of
//! distance functions, so its maximum over the sphere sits where the gradients of the equal distances
//! have the origin in their convex hull. For distinct sites that is one of: (a) a pole of the plane
//! through three sites (three distances equal); (b) the antipode of the midpoint of two sites that are
//! not antipodal (two distances equal, gradients opposed); (c) the antipode of a site; (d) a point of
//! the great circle half-way between an antipodal pair, which is a ridge of value π/2 that ends at a
//! pole of kind (a) unless every other site sits on the pair's axis. Every candidate's value is a
//! lower bound, and the maximiser is a candidate, so the largest candidate value is `α_k(t)`. One
//! site, and `k` equal to the number of sites, give π; two sites give `π − d/2` for `k = 1`. The
//! candidate values on the regular polyhedra, the two-site form, the ridge's end points and the
//! Walker symmetry below are derived in `derivations/analysis_alpha_exact.mac` and re-evaluated at 60
//! digits in `derivations/analysis_alpha_exact.gp`; the gate is
//! `the_regular_polyhedra_have_their_exact_alpha` below.
//!
//! **Near-degenerate sites.** Two sites within [`SNAP_CHORD`] of each other or of each other's
//! antipode are moved onto one representative (or its antipode) before the candidates are built:
//! near such a pair a candidate's location is ill-conditioned (an anti-midpoint at `π − δ` moves by
//! `ε/δ`), while `α` itself is 1-Lipschitz in the sites. The move `Δ` widens the upper end by `2Δ`.
//! Every other candidate carries a first-order bound `λ` on its distance from the exact critical point
//! it computes, and every dot product a rounding allowance `γ = 32u`: the lower end is the best
//! candidate, the upper end adds `λ`, `γ` and `2Δ`.
//!
//! **Time: Lipschitz branch and bound.** Each direction turns about its fixed orbit normal `ĥ_i` at its
//! true-anomaly rate `ν̇_i`. In a frame turning at `W`, the mean of the `ν̇_i ĥ_i`, site `i` moves at
//! most `|ν̇_i ĥ_i − W|`, so `α_k(t)` is Lipschitz with the largest of these, `L` (zero for one plane,
//! which turns rigidly). On `[a, b]`, `α_k` is at most the crossing of the two cones of slope `L` on
//! the end values, so branch and bound closes an enclosure of the worst case. A plane's satellites
//! alone cover no better than all of them, and turn rigidly, so the α of any plane with more than `k`
//! satellites caps the whole span without splitting it: that closes plateaus such as Walker 4/2/1's
//! π/2. A first-order bound pays for a smooth peak of curvature `κ` about `L/√(κm)` instants to
//! settle a margin `m`, and for a plateau no single plane holds about `L·span/m`; past its budget the
//! enclosure is reported wider, and the decision declines.
//!
//! **The decision.** [`minimum_walker`] accepts a design only on the upper end, and compares dot
//! products with cosines, so no inverse function is in the decision: `α_k ≤ θ` on a time interval of
//! half-width `h` holds when at every candidate the `k`-th largest dot product with the sites is at
//! least `cos(θ − Lh − λ − 2Δ) + γ`. It refuses on a candidate whose dot product plus `γ` is below
//! `cos θ`, and does not accept a design it can decide neither way within its budget.
//!
//! Every enclosure here is **under H-libm, not outward-rounded** ([`AlphaEnclosure::LABEL`]): the
//! sites and `θ` pass through `libm`'s `sin` and `cos`, the reported ends through `acos`, and the
//! arithmetic rounds to nearest with the first-order allowances above.
//!
//! This is design-time analysis (it allocates), so it lives behind the crate's `std` feature; the
//! generators stay `no_std`.

use core::cmp::Ordering;
use core::ops::ControlFlow;
use std::collections::{BinaryHeap, VecDeque};

use spacelab_core::{Angle, Vec3};
use spacelab_elements::KeplerianElements;

use crate::walker::{Walker, WalkerPattern};

const PI: f64 = core::f64::consts::PI;
/// The unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;
/// How far from unit length a vector normalised here may be.
const ETA: f64 = 6.0 * U;
/// The rounding allowance on a dot product of two vectors normalised here.
const GAMMA: f64 = 32.0 * U;
/// Two sites closer than this chord to each other, or to each other's antipode, are moved onto one
/// representative. `2^-25` balances the move it costs against the `ε/δ` location error it avoids.
pub const SNAP_CHORD: f64 = 1.0 / 33_554_432.0;
/// Candidates whose location bound is at most this share one cosine in the decision and one `acos`
/// in the enclosure.
const SMALL_LAMBDA: f64 = 1e-12;
/// Coarse sub-intervals a branch and bound starts from.
const START_INTERVALS: u32 = 16;
/// Instants the enclosure branch and bound may evaluate before it reports what it has.
const ENCLOSURE_BUDGET: usize = 1 << 16;
/// Instants the decision may evaluate on one design before it declines to accept it. Near a smooth
/// peak of `α_k(t)` a margin `m` costs about `L/√(κm)` instants, `κ` the peak's curvature.
const DECISION_BUDGET: usize = 1 << 16;
/// A slack `L·h` below this does not split further: the decision declines the design.
const DECISION_RESOLUTION: f64 = 1e-12;

/// Why an α-characteristic was not computed: input outside its domain is refused, never answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlphaError {
    /// `fold` is zero: no point is covered zero times.
    ZeroFold,
    /// `fold` exceeds the number of satellites, so no point is ever covered that often.
    FoldAboveSites {
        /// The fold asked for.
        fold: u32,
        /// The number of satellites or sites given.
        sites: usize,
    },
    /// A site is not finite or is the zero vector, so it names no point of the sphere.
    BadSite {
        /// Index of the offending site in the caller's slice.
        site: usize,
    },
    /// An element set is not a two-body ellipse: a non-finite field, an eccentricity outside
    /// `[0, 1)`, or a semi-major axis or `mu` that is not positive.
    BadOrbit {
        /// Index of the offending element set in the caller's slice.
        satellite: usize,
    },
    /// The tolerance is not finite and positive.
    BadTolerance,
    /// The footprint is NaN.
    BadFootprint,
}

impl core::fmt::Display for AlphaError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroFold => write!(f, "a fold of zero covers nothing"),
            Self::FoldAboveSites { fold, sites } => {
                write!(f, "fold {fold} exceeds the {sites} satellites given")
            }
            Self::BadSite { site } => write!(f, "site {site} is not a finite non-zero vector"),
            Self::BadOrbit { satellite } => {
                write!(f, "satellite {satellite} is not on a two-body ellipse")
            }
            Self::BadTolerance => write!(f, "the tolerance is not finite and positive"),
            Self::BadFootprint => write!(f, "the footprint is NaN"),
        }
    }
}

impl core::error::Error for AlphaError {}

/// An enclosure of an α-characteristic, radians: `lower ≤ α ≤ upper`, [`AlphaEnclosure::LABEL`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlphaEnclosure {
    /// The lower end: the value of the best candidate found.
    pub lower: f64,
    /// The upper end, the one an acceptance compares.
    pub upper: f64,
}

impl AlphaEnclosure {
    /// What the enclosure rests on: the `libm` elementary functions within their measured error
    /// (H-libm), and round-to-nearest arithmetic with first-order allowances rather than outward
    /// rounding.
    pub const LABEL: &'static str = "under H-libm, not outward-rounded";
}

/// The `k`-fold covering radius of a fixed set of sites: the α-characteristic at one instant.
///
/// `sites` are directions (any non-zero length); the answer is the smallest angular radius at which
/// caps about the sites cover every point of the sphere at least `fold` times.
///
/// # Errors
/// [`AlphaError::ZeroFold`], [`AlphaError::FoldAboveSites`] (an empty set included), and
/// [`AlphaError::BadSite`] for a site that is not finite or is zero.
pub fn covering_radius(sites: &[Vec3], fold: u32) -> Result<AlphaEnclosure, AlphaError> {
    let k = check_fold(fold, sites.len())?;
    let mut unit_sites = Vec::with_capacity(sites.len());
    for (i, &s) in sites.iter().enumerate() {
        let len = norm(s);
        if !(len.is_finite() && len > 0.0) {
            return Err(AlphaError::BadSite { site: i });
        }
        unit_sites.push(s.scale(1.0 / len));
    }
    let (lower, upper) = instant_enclosure(&unit_sites, k, &mut Workspace::default());
    Ok(AlphaEnclosure { lower, upper })
}

/// The α-characteristic of `fold`-fold coverage for the satellites `sats` (radians): the smallest
/// sub-satellite zone angular radius at which they cover the sphere at least `fold` times at every
/// instant of `[0, T₀]`, `T₀` the period of `sats[0]` — every instant when the satellites share that
/// period, as a delta-system's do.
///
/// Exact in space, a Lipschitz branch and bound in time (module docs). The branch and bound stops
/// when the enclosure is narrower than `tolerance` or after 65 536 instants; it is an enclosure
/// either way, only wider in the second case.
///
/// # Errors
/// [`AlphaError::ZeroFold`], [`AlphaError::FoldAboveSites`] (an empty constellation included),
/// [`AlphaError::BadOrbit`], and [`AlphaError::BadTolerance`].
pub fn alpha_characteristic(
    sats: &[KeplerianElements],
    fold: u32,
    tolerance: f64,
) -> Result<AlphaEnclosure, AlphaError> {
    let k = check_fold(fold, sats.len())?;
    check_orbits(sats)?;
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(AlphaError::BadTolerance);
    }
    if k == sats.len() {
        return Ok(AlphaEnclosure {
            lower: PI,
            upper: PI.next_up(),
        });
    }
    let motion = Motion::new(sats);
    Ok(motion.enclose(k, sats[0].period(), tolerance))
}

/// Search for the Walker Delta system with the fewest satellites whose α-characteristic is proven not
/// to exceed `max_footprint` (radians): the cheapest constellation that continuously `fold`-fold
/// covers the sphere with a sensor of that footprint, at the common `inclination`, `semi_major_axis`
/// and `mu`. Counts are scanned upward and, within each, all plane factorisations and phasings
/// tried.
///
/// A design is accepted only on the upper end of its enclosure, decided on dot products against
/// `cos θ` over one symmetry period of the pattern (module docs). A design is skipped when its
/// α-characteristic is proven above the footprint, or when the decision does not close within 65 536
/// instants or a slack of 1e-12 rad: when the α-characteristic sits close to the footprint, by a
/// margin the module docs price. The first design accepted is therefore proven to cover; a smaller
/// design was either proven not to or left undecided that close to the footprint, never accepted on
/// a sample. `Ok(None)` if no design of at most `max_total` satellites is accepted.
///
/// # Errors
/// [`AlphaError::ZeroFold`], [`AlphaError::BadFootprint`] for a NaN footprint, and
/// [`AlphaError::BadOrbit`] for a semi-major axis or `mu` that is not finite and positive.
pub fn minimum_walker(
    max_footprint: f64,
    inclination: Angle,
    semi_major_axis: f64,
    mu: f64,
    fold: u32,
    max_total: u32,
) -> Result<Option<Walker>, AlphaError> {
    if fold == 0 {
        return Err(AlphaError::ZeroFold);
    }
    if max_footprint.is_nan() {
        return Err(AlphaError::BadFootprint);
    }
    let finite_positive = |x: f64| x.is_finite() && x > 0.0;
    if !(finite_positive(semi_major_axis)
        && finite_positive(mu)
        && inclination.radians().is_finite())
    {
        return Err(AlphaError::BadOrbit { satellite: 0 });
    }
    let mut sats = Vec::new();
    for total in fold..=max_total {
        for planes in 1..=total {
            if total % planes != 0 {
                continue;
            }
            let per_plane = total / planes;
            for phasing in 0..planes {
                let w = Walker {
                    pattern: WalkerPattern::Delta,
                    inclination,
                    planes,
                    per_plane,
                    phasing,
                    semi_major_axis,
                    mu,
                };
                if walker_verdict(&w, fold, max_footprint, &mut sats) == Verdict::Covers {
                    return Ok(Some(w));
                }
            }
        }
    }
    Ok(None)
}

/// Whether the Walker Delta `w` covers the sphere `fold` times with footprint `theta`.
fn walker_verdict(w: &Walker, fold: u32, theta: f64, sats: &mut Vec<KeplerianElements>) -> Verdict {
    sats.clear();
    w.generate(|e| sats.push(e));
    let k = fold as usize;
    if theta > PI {
        // Every k-th distance is at most π, which is below any footprint above the double nearest π.
        return Verdict::Covers;
    }
    if theta <= 0.0 || k >= sats.len() {
        // α is positive; and for k = n it is π itself, above every footprint not above `PI`.
        return Verdict::Exceeds;
    }
    let motion = Motion::new(sats);
    motion.decide(k, walker_symmetry_span(w), theta)
}

/// The time after which a Walker Delta pattern repeats up to a rotation about the polar axis, seconds.
///
/// With `T = P·S` and `F` taken mod `P`, advancing every argument of latitude by `2π/S` relabels the
/// slots, and by `2πF/T` gives the pattern turned by `2π/P` with the planes relabelled
/// (`derivations/analysis_alpha_exact.mac`, and `walker_shift_is_a_rotation` in its `.gp`). Their
/// integer combinations include `2π·gcd(P, F)/T`, and α is invariant under rotation, so one such
/// span is every instant.
fn walker_symmetry_span(w: &Walker) -> f64 {
    debug_assert_eq!(w.pattern, WalkerPattern::Delta);
    let (planes, total) = (w.planes, w.total());
    let g = gcd(planes, w.phasing % planes);
    let n = libm::sqrt(w.mu / (w.semi_major_axis * w.semi_major_axis * w.semi_major_axis));
    core::f64::consts::TAU * f64::from(g) / (f64::from(total) * n)
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn check_fold(fold: u32, sites: usize) -> Result<usize, AlphaError> {
    if fold == 0 {
        return Err(AlphaError::ZeroFold);
    }
    let k = fold as usize;
    if k > sites {
        return Err(AlphaError::FoldAboveSites { fold, sites });
    }
    Ok(k)
}

fn check_orbits(sats: &[KeplerianElements]) -> Result<(), AlphaError> {
    for (i, e) in sats.iter().enumerate() {
        let angles = [
            e.inclination.radians(),
            e.raan.radians(),
            e.arg_periapsis.radians(),
            e.true_anomaly.radians(),
        ];
        let ok = e.semi_major_axis.is_finite()
            && e.semi_major_axis > 0.0
            && e.mu.is_finite()
            && e.mu > 0.0
            && (0.0..1.0).contains(&e.eccentricity)
            && angles.iter().all(|a| a.is_finite());
        if !ok {
            return Err(AlphaError::BadOrbit { satellite: i });
        }
    }
    Ok(())
}

/// The outcome of a decision `α_k ≤ θ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Proven: `α_k ≤ θ` at every instant.
    Covers,
    /// Proven: some point is farther than `θ` from its `k`-th nearest satellite at some instant.
    Exceeds,
    /// Neither within the budget or the resolution: not accepted.
    Undecided,
}

/// What one instant says about `α_k ≤ θ'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Instant {
    /// A candidate is farther than `θ` from its `k`-th nearest site.
    Hole,
    /// Every candidate is within `θ'` with its allowances.
    Holds,
    /// Neither.
    Unknown,
}

/// The satellites as moving sites, with the Lipschitz constant of `α_k` in time.
struct Motion<'a> {
    sats: &'a [KeplerianElements],
    /// rad/s.
    lipschitz: f64,
    /// Satellites that turn together — one plane, one rate — with their own relative rate (rad/s).
    groups: Vec<(Vec<usize>, f64)>,
}

impl<'a> Motion<'a> {
    fn new(sats: &'a [KeplerianElements]) -> Self {
        let turns: Vec<Turn> = sats.iter().map(Turn::of).collect();
        let mut members: Vec<Vec<usize>> = Vec::new();
        for (i, t) in turns.iter().enumerate() {
            // Any partition is sound, since each group carries its own rate; this one puts a plane's
            // satellites together, whose group then turns rigidly at a rate of rounding size.
            let v = t.mean_velocity();
            match members.iter_mut().find(|g| {
                let first = turns[g[0]].mean_velocity();
                norm(v - first) <= 1e-9 * norm(first)
            }) {
                Some(g) => g.push(i),
                None => members.push(vec![i]),
            }
        }
        let groups = members
            .into_iter()
            .map(|g| {
                let own: Vec<Turn> = g.iter().map(|&i| turns[i]).collect();
                let rate = relative_rate(&own);
                (g, rate)
            })
            .collect();
        Self {
            sats,
            lipschitz: relative_rate(&turns),
            groups,
        }
    }

    fn sites_at(&self, t: f64, out: &mut Vec<Vec3>) {
        out.clear();
        for e in self.sats {
            out.push(unit(e.propagate(t).to_state().0));
        }
    }

    /// The groups with more than `k` members, each with its sites at `t = 0` and its rate.
    fn rigid_groups(&self, k: usize) -> impl Iterator<Item = (Vec<Vec3>, f64)> + '_ {
        self.groups
            .iter()
            .filter(move |(g, _)| g.len() > k)
            .map(|(g, rate)| {
                let sites = g.iter().map(|&i| unit(self.sats[i].to_state().0)).collect();
                (sites, *rate)
            })
    }

    /// An upper bound of `α_k` on `[0, span]` that does not split time. A subset `G` of the sites can
    /// only raise every `k`-th distance, so `α_k(S) ≤ α_k(G)`; and `α_k(G)` moves at most at `G`'s own
    /// relative rate, which is of rounding size for one plane. This closes the plateaus a Lipschitz
    /// bound cannot, such as Walker 4/2/1, whose antipodal pairs hold `α₁` at π/2 at every instant.
    fn ceiling(&self, k: usize, span: f64, ws: &mut Workspace) -> f64 {
        self.rigid_groups(k)
            .fold(PI.next_up(), |best, (sites, rate)| {
                best.min(instant_enclosure(&sites, k, ws).1 + rate * span)
            })
    }

    /// Best-first branch and bound for the worst case of `α_k` over `[0, span]`.
    fn enclose(&self, k: usize, span: f64, tolerance: f64) -> AlphaEnclosure {
        let mut ws = Workspace::default();
        let mut sites = Vec::with_capacity(self.sats.len());
        let mut eval = |t: f64, ws: &mut Workspace| {
            self.sites_at(t, &mut sites);
            instant_enclosure(&sites, k, ws)
        };
        let mut heap = BinaryHeap::new();
        let ceiling = self.ceiling(k, span, &mut ws);
        let step = span / f64::from(START_INTERVALS);
        let (mut lower, mut ua) = eval(0.0, &mut ws);
        let mut a = 0.0;
        let mut evaluations = 1;
        for j in 1..=START_INTERVALS {
            let b = if j == START_INTERVALS {
                span
            } else {
                step * f64::from(j)
            };
            let (lo, ub) = eval(b, &mut ws);
            evaluations += 1;
            lower = lower.max(lo);
            heap.push(Piece::new(a, b, ua, ub, self.lipschitz));
            (a, ua) = (b, ub);
        }
        while let Some(top) = heap.peek() {
            if top.bound.min(ceiling) <= lower + tolerance || evaluations >= ENCLOSURE_BUDGET {
                break;
            }
            let Some(p) = heap.pop() else { break };
            let m = p.a.midpoint(p.b);
            let (lo, um) = eval(m, &mut ws);
            evaluations += 1;
            lower = lower.max(lo);
            heap.push(Piece::new(p.a, m, p.ua, um, self.lipschitz));
            heap.push(Piece::new(m, p.b, um, p.ub, self.lipschitz));
        }
        let top = heap.peek().map_or(lower, |p| p.bound);
        AlphaEnclosure {
            lower,
            upper: top.min(ceiling).max(lower),
        }
    }

    /// Breadth-first decision of `α_k ≤ theta` over `[0, span]`, for `0 < theta ≤ PI`.
    fn decide(&self, k: usize, span: f64, theta: f64) -> Verdict {
        let mut ws = Workspace::default();
        let mut sites = Vec::with_capacity(self.sats.len());
        let cos_theta = libm::cos(theta);
        for (group, rate) in self.rigid_groups(k) {
            if instant_verdict(&group, k, cos_theta, theta - rate * span, &mut ws) == Instant::Holds
            {
                return Verdict::Covers;
            }
        }
        let step = span / f64::from(START_INTERVALS);
        let mut queue: VecDeque<(f64, f64)> = (0..START_INTERVALS)
            .map(|j| {
                let b = if j + 1 == START_INTERVALS {
                    span
                } else {
                    step * f64::from(j + 1)
                };
                (step * f64::from(j), b)
            })
            .collect();
        let mut evaluations = 0;
        while let Some((a, b)) = queue.pop_front() {
            let m = a.midpoint(b);
            let slack = self.lipschitz * (b - a) * 0.5;
            self.sites_at(m, &mut sites);
            evaluations += 1;
            match instant_verdict(&sites, k, cos_theta, theta - slack, &mut ws) {
                Instant::Hole => return Verdict::Exceeds,
                Instant::Holds => {}
                Instant::Unknown => {
                    if evaluations >= DECISION_BUDGET || slack <= DECISION_RESOLUTION {
                        return Verdict::Undecided;
                    }
                    queue.push_back((a, m));
                    queue.push_back((m, b));
                }
            }
        }
        Verdict::Covers
    }
}

/// How one satellite's direction turns: about its fixed orbit normal, at a true-anomaly rate between
/// `slow` and `fast`, `n(1 ∓ e)²/(1 − e²)^{3/2}` (`derivations/analysis_alpha_exact.mac`,
/// `true_anomaly_rate`).
#[derive(Clone, Copy)]
struct Turn {
    normal: Vec3,
    slow: f64,
    fast: f64,
}

impl Turn {
    fn of(e: &KeplerianElements) -> Self {
        let (r, v) = e.to_state();
        let ecc = e.eccentricity;
        let q = 1.0 - ecc * ecc;
        let scale = e.mean_motion() / (q * libm::sqrt(q));
        Self {
            normal: unit(r.cross(v)),
            slow: scale * (1.0 - ecc) * (1.0 - ecc),
            fast: scale * (1.0 + ecc) * (1.0 + ecc),
        }
    }

    fn mean_velocity(self) -> Vec3 {
        self.normal.scale(0.5 * (self.slow + self.fast))
    }
}

/// The Lipschitz constant of `t ↦ α_k(t)` for these satellites, rad/s: in a frame turning at the
/// mean `W` of their angular velocities, a direction turning at `ν̇ ĥ` moves at `|(ν̇ ĥ − W) × s| ≤
/// |ν̇ ĥ − W|` (`derivations/analysis_alpha_exact.mac`, `lagrange_identity`), the larger at one of the
/// two ends of `ν̇`, and α is 1-Lipschitz in the sites and invariant under the frame's rotation.
fn relative_rate(turns: &[Turn]) -> f64 {
    #[allow(clippy::cast_precision_loss)] // a satellite count is far below 2^52
    let inv = 1.0 / turns.len() as f64;
    let w = turns
        .iter()
        .fold(Vec3::ZERO, |acc, t| acc + t.mean_velocity().scale(inv));
    let worst = turns.iter().fold(0.0_f64, |acc, t| {
        acc.max(norm(t.normal.scale(t.slow) - w))
            .max(norm(t.normal.scale(t.fast) - w))
    });
    // The constant is a sum of rounded terms; a relative margin far above their rounding keeps it
    // from landing below the exact value.
    worst * (1.0 + 1e-12)
}

/// A piece of the time axis with its end uppers and the bound they give on its interior.
struct Piece {
    bound: f64,
    a: f64,
    b: f64,
    ua: f64,
    ub: f64,
}

impl Piece {
    /// On `[a, b]` an `L`-Lipschitz function under `ua` at `a` and `ub` at `b` stays under both cones:
    /// at most where they cross, or the lower end plus `L(b − a)` when they cross outside.
    fn new(a: f64, b: f64, ua: f64, ub: f64, lipschitz: f64) -> Self {
        let rise = lipschitz * (b - a);
        let bound = if (ua - ub).abs() <= rise {
            0.5 * (ua + ub + rise)
        } else {
            ua.min(ub) + rise
        };
        Self {
            bound: bound.min(PI.next_up()),
            a,
            b,
            ua,
            ub,
        }
    }
}

impl PartialEq for Piece {
    fn eq(&self, other: &Self) -> bool {
        self.bound.total_cmp(&other.bound) == Ordering::Equal
    }
}
impl Eq for Piece {}
impl PartialOrd for Piece {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Piece {
    fn cmp(&self, other: &Self) -> Ordering {
        self.bound.total_cmp(&other.bound)
    }
}

/// Buffers one instant reuses.
#[derive(Default)]
struct Workspace {
    scratch: Vec<f64>,
    parent: Vec<usize>,
    minus: Vec<bool>,
    /// Distinct points after the move, each with its cluster.
    points: Vec<(Vec3, usize)>,
    /// Representatives whose antipode is also a point.
    axes: Vec<Vec3>,
}

/// `α_k` at one instant, `(lower, upper)`, for unit `sites` and `1 ≤ k ≤ sites.len()`.
fn instant_enclosure(sites: &[Vec3], k: usize, ws: &mut Workspace) -> (f64, f64) {
    let n = sites.len();
    if k == n {
        return (PI, PI.next_up());
    }
    if n == 2 {
        let c = two_site_dot(sites[0], sites[1]);
        return (acos_low(c + GAMMA), acos_high(c - GAMMA));
    }
    let moved = snap(sites, ws);
    let (mut min_all, mut min_small, mut loose) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let mut small_lambda = 0.0_f64;
    let mut scratch = core::mem::take(&mut ws.scratch);
    let _ = for_each_candidate(ws, |x, lambda| {
        let c = kth_dot(x, sites, k, &mut scratch);
        min_all = min_all.min(c);
        if lambda <= SMALL_LAMBDA {
            min_small = min_small.min(c);
            small_lambda = small_lambda.max(lambda);
        } else {
            loose = loose.max(acos_high(c - GAMMA) + lambda);
        }
        ControlFlow::Continue(())
    });
    ws.scratch = scratch;
    let tight = acos_high(min_small - GAMMA) + small_lambda;
    let upper = (tight.max(loose) + 2.0 * moved).min(PI.next_up());
    (acos_low(min_all + GAMMA), upper)
}

/// Whether `α_k ≤ theta_slack` at one instant, refusing on a candidate beyond `θ` itself.
fn instant_verdict(
    sites: &[Vec3],
    k: usize,
    cos_theta: f64,
    theta_slack: f64,
    ws: &mut Workspace,
) -> Instant {
    let cos_or_never = |angle: f64| {
        if angle >= 0.0 {
            libm::cos(angle)
        } else {
            f64::INFINITY
        }
    };
    if sites.len() == 2 {
        let c = two_site_dot(sites[0], sites[1]);
        return if c + GAMMA < cos_theta {
            Instant::Hole
        } else if c - GAMMA >= cos_or_never(theta_slack) {
            Instant::Holds
        } else {
            Instant::Unknown
        };
    }
    let moved = snap(sites, ws);
    let base = theta_slack - 2.0 * moved;
    let need_small = cos_or_never(base - SMALL_LAMBDA);
    let mut holds = true;
    let mut scratch = core::mem::take(&mut ws.scratch);
    let flow = for_each_candidate(ws, |x, lambda| {
        let c = kth_dot(x, sites, k, &mut scratch);
        if c + GAMMA < cos_theta {
            return ControlFlow::Break(());
        }
        if holds {
            let need = if lambda <= SMALL_LAMBDA {
                need_small
            } else {
                cos_or_never(base - lambda)
            };
            holds = c - GAMMA >= need;
        }
        ControlFlow::Continue(())
    });
    ws.scratch = scratch;
    if flow.is_break() {
        Instant::Hole
    } else if holds {
        Instant::Holds
    } else {
        Instant::Unknown
    }
}

/// The dot product of the anti-midpoint of two sites with either of them, `−|a + b|/2 = −cos(d/2)`,
/// which is the `k = 1` maximum `π − d/2` in dot-product form and needs no location
/// (`derivations/analysis_alpha_exact.mac`, `two_sites`).
fn two_site_dot(a: Vec3, b: Vec3) -> f64 {
    -0.5 * norm(a + b)
}

/// Move every site within [`SNAP_CHORD`] of another, or of its antipode, onto its cluster's first
/// site or that site's antipode; fill `ws.points` and `ws.axes`; return the largest move (radians).
fn snap(sites: &[Vec3], ws: &mut Workspace) -> f64 {
    let n = sites.len();
    ws.parent.clear();
    ws.parent.extend(0..n);
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (sites[i], sites[j]);
            if norm(a - b) <= SNAP_CHORD || norm(a + b) <= SNAP_CHORD {
                let (ri, rj) = (root(&mut ws.parent, i), root(&mut ws.parent, j));
                // The smaller index is the root, so a cluster's representative is its first site.
                let (lo, hi) = if ri < rj { (ri, rj) } else { (rj, ri) };
                ws.parent[hi] = lo;
            }
        }
    }
    ws.minus.clear();
    ws.minus.resize(n, false);
    let mut moved = 0.0_f64;
    for i in 0..n {
        let r = root(&mut ws.parent, i);
        if r == i {
            continue;
        }
        let rep = sites[r];
        let same = dot(sites[i], rep) >= 0.0;
        let target = if same { rep } else { -rep };
        if !same {
            ws.minus[r] = true;
        }
        // Chord c between the exact unit vectors is at most the computed one plus 2η, and the angle
        // 2·asin(c/2) is at most c(1 + c).
        let c = norm(sites[i] - target) + 2.0 * ETA;
        moved = moved.max(c * (1.0 + c));
    }
    ws.points.clear();
    ws.axes.clear();
    for (i, &site) in sites.iter().enumerate() {
        if root(&mut ws.parent, i) == i {
            ws.points.push((site, i));
            if ws.minus[i] {
                ws.points.push((-site, i));
                ws.axes.push(site);
            }
        }
    }
    moved
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Visit every candidate built from `ws.points` and `ws.axes` with its location bound (radians):
/// the antipodes, the anti-midpoints of pairs from different clusters, both poles of every triple, and
/// two points of each axis's equator. A location that cannot be bounded is visited with an infinite
/// bound at a valid point, which keeps the lower end and gives up the upper.
fn for_each_candidate(
    ws: &Workspace,
    mut visit: impl FnMut(Vec3, f64) -> ControlFlow<()>,
) -> ControlFlow<()> {
    let points = &ws.points;
    let m = points.len();
    for &(p, _) in points {
        visit(-p, 0.0)?;
    }
    for a in 0..m {
        for b in (a + 1)..m {
            if points[a].1 == points[b].1 {
                continue;
            }
            let s = points[a].0 + points[b].0;
            let len = norm(s);
            // |s − (p̂_a + p̂_b)| ≤ 2η + 2u|s|.
            let lambda = location_bound(2.0 * ETA + 2.0 * U * len, len);
            let x = if lambda.is_finite() {
                s.scale(-1.0 / len)
            } else {
                points[a].0
            };
            visit(x, lambda)?;
        }
    }
    for a in 0..m {
        for b in (a + 1)..m {
            for c in (b + 1)..m {
                let (pa, pb, pc) = (points[a].0, points[b].0, points[c].0);
                let pole = (pb - pa).cross(pc - pa);
                let len = norm(pole);
                // Each difference is within 16u of the exact one and at most 2 long; the cross
                // product's own rounding is at most 3u|A||B|: 80u in all.
                let lambda = location_bound(80.0 * U, len);
                if lambda.is_finite() {
                    let x = pole.scale(1.0 / len);
                    visit(x, lambda)?;
                    visit(-x, lambda)?;
                } else {
                    visit(pa, lambda)?;
                }
            }
        }
    }
    for &r in &ws.axes {
        // r × e, with e the axis of r's smallest component, is exact and at least √(2/3) long.
        let (ax, ay, az) = (r.x.abs(), r.y.abs(), r.z.abs());
        let e = if ax <= ay && ax <= az {
            Vec3::new(1.0, 0.0, 0.0)
        } else if ay <= az {
            Vec3::new(0.0, 1.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };
        let u1 = unit(r.cross(e));
        let u2 = unit(r.cross(u1));
        visit(u1, 8.0 * U)?;
        visit(u2, 8.0 * U)?;
    }
    ControlFlow::Continue(())
}

/// The angle between a computed direction `s` of length `len` and an exact one within `err` of it,
/// plus the rounding of the normalisation: `asin(err/len) ≤ (π/2)·err/len`, while `err < len/2`.
fn location_bound(err: f64, len: f64) -> f64 {
    if len.is_finite() && err < 0.5 * len {
        2.0 * err / len + 4.0 * U
    } else {
        f64::INFINITY
    }
}

/// The `k`-th largest dot product of `x` with `sites`.
fn kth_dot(x: Vec3, sites: &[Vec3], k: usize, scratch: &mut Vec<f64>) -> f64 {
    if k == 1 {
        return sites
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &s| acc.max(dot(x, s)));
    }
    scratch.clear();
    scratch.extend(sites.iter().map(|&s| dot(x, s)));
    *scratch
        .select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a))
        .1
}

/// The plain three-term dot product: one rounding per operation, inside [`GAMMA`].
fn dot(a: Vec3, b: Vec3) -> f64 {
    a.x * b.x + a.y * b.y + a.z * b.z
}

fn norm(v: Vec3) -> f64 {
    libm::sqrt(dot(v, v))
}

fn unit(v: Vec3) -> Vec3 {
    v.scale(1.0 / norm(v))
}

/// A lower bound of the angle whose cosine is at most `c`.
fn acos_low(c: f64) -> f64 {
    if c >= 1.0 { 0.0 } else { libm::acos(c) }
}

/// An upper bound of the angle whose cosine is at least `c`; `PI` itself is below π.
fn acos_high(c: f64) -> f64 {
    if c <= -1.0 {
        PI.next_up()
    } else {
        libm::acos(c)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use crate::soc::StreetOfCoverage;

    const MU: f64 = 3.986_004_418e14;
    const RE: f64 = 6_378_137.0;
    const TAU: f64 = core::f64::consts::TAU;

    fn walker_at(
        inclination_deg: f64,
        planes: u32,
        per_plane: u32,
        phasing: u32,
    ) -> (Walker, Vec<KeplerianElements>) {
        let w = Walker {
            pattern: WalkerPattern::Delta,
            inclination: Angle::from_radians(inclination_deg.to_radians()),
            planes,
            per_plane,
            phasing,
            semi_major_axis: RE + 20_000e3,
            mu: MU,
        };
        let mut v = Vec::new();
        w.generate(|e| v.push(e));
        (w, v)
    }

    fn walker(planes: u32, per_plane: u32) -> Vec<KeplerianElements> {
        walker_at(55.0, planes, per_plane, 1).1
    }

    /// The enclosure over one symmetry span, the path `minimum_walker` decides on.
    fn walker_enclosure(
        w: &Walker,
        sats: &[KeplerianElements],
        k: usize,
        tol: f64,
    ) -> AlphaEnclosure {
        Motion::new(sats).enclose(k, walker_symmetry_span(w), tol)
    }

    fn contains(e: AlphaEnclosure, x: f64) -> bool {
        e.lower <= x && x <= e.upper
    }

    // ── The test-only comparator: the sampled α of the earlier code ─────────────────────────────

    /// The `k`-th of `n` near-uniform Fibonacci-sphere directions.
    fn fibonacci_dir(k: u32, n: u32) -> Vec3 {
        let ga = core::f64::consts::PI * (3.0 - libm::sqrt(5.0));
        let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(n);
        let r = libm::sqrt((1.0 - z * z).max(0.0));
        let phi = ga * f64::from(k);
        Vec3::new(r * libm::cos(phi), r * libm::sin(phi), z)
    }

    /// The worst `fold`-th-nearest distance over `grid_points` Fibonacci directions × `time_samples`
    /// instants of one period: a maximum over a subset, so never above the true α. It is a
    /// comparator, and the acceptance `minimum_walker` used to make on it is the defect the
    /// planted case below shows.
    fn sampled_alpha(
        sats: &[KeplerianElements],
        fold: u32,
        grid_points: u32,
        time_samples: u32,
    ) -> f64 {
        let fold = fold as usize;
        let period = TAU * libm::sqrt(sats[0].semi_major_axis.powi(3) / sats[0].mu);
        let mut worst = 0.0_f64;
        let mut angles = Vec::with_capacity(sats.len());
        for it in 0..time_samples {
            let t = period * f64::from(it) / f64::from(time_samples);
            let dirs: Vec<Vec3> = sats
                .iter()
                .map(|e| unit(e.propagate(t).to_state().0))
                .collect();
            for ip in 0..grid_points {
                let p = fibonacci_dir(ip, grid_points);
                angles.clear();
                angles.extend(dirs.iter().map(|d| libm::acos(d.dot(p).clamp(-1.0, 1.0))));
                angles.sort_by(f64::total_cmp);
                worst = worst.max(angles[fold - 1]);
            }
        }
        worst
    }

    #[test]
    fn the_fibonacci_grid_is_unit_length_even_in_z_and_turns_by_the_golden_angle() {
        // The golden angle is written here as τ/φ² rather than the comparator's π(3−√5); they are the
        // same number — 4π/(3+√5) = 4π(3−√5)/4 — by a different transcription.
        let n = 97_u32;
        let phi = 1.0_f64.midpoint(libm::sqrt(5.0));
        let ga = TAU / (phi * phi);
        for k in 0..n {
            let v = fibonacci_dir(k, n);
            assert!((v.norm() - 1.0).abs() < 1e-15, "k={k}: |v| = {}", v.norm());
            let want_z = 1.0 - (2.0 * f64::from(k) + 1.0) / f64::from(n);
            assert!(
                (v.z - want_z).abs() < 1e-15,
                "k={k}: z {} want {want_z}",
                v.z
            );
            let want = libm::fmod(libm::fmod(ga * f64::from(k), TAU) + TAU, TAU);
            let got = libm::fmod(libm::atan2(v.y, v.x) + TAU, TAU);
            let gap = (got - want).abs();
            assert!(
                gap.min(TAU - gap) < 1e-9,
                "k={k}: azimuth {got} want {want}"
            );
        }
    }

    /// The comparator equals its own closed form on configurations small enough to evaluate exactly
    /// (`derivations/analysis_alpha_closed_form.mac`, re-evaluated in
    /// `derivations/analysis_alpha_closed_form.gp`): one satellite `4π/5`; three equatorial
    /// satellites over five instants `4π/15`, `2π/3`, `14π/15`; one polar satellite on the two-point
    /// grid over four instants `5π/6`.
    #[test]
    fn the_sampler_equals_its_closed_form_on_small_configurations() {
        let pi = core::f64::consts::PI;
        let ring = |per_plane: u32, inclination: f64| {
            let w = Walker {
                pattern: WalkerPattern::Delta,
                inclination: Angle::from_radians(inclination),
                planes: 1,
                per_plane,
                phasing: 0,
                semi_major_axis: RE + 20_000e3,
                mu: MU,
            };
            let mut v = Vec::new();
            w.generate(|e| v.push(e));
            v
        };
        for (sats, fold, grid, times, want) in [
            (ring(1, 0.0), 1, 1, 5, 4.0 * pi / 5.0),
            (ring(3, 0.0), 1, 1, 5, 4.0 * pi / 15.0),
            (ring(3, 0.0), 2, 1, 5, 2.0 * pi / 3.0),
            (ring(3, 0.0), 3, 1, 5, 14.0 * pi / 15.0),
            (ring(1, 0.5 * pi), 1, 2, 4, 5.0 * pi / 6.0),
        ] {
            let got = sampled_alpha(&sats, fold, grid, times);
            assert!(
                (got - want).abs() < 1e-14,
                "{} sats, fold {fold}: {got} against {want}",
                sats.len()
            );
        }
    }

    // ── Space: the exact covering radius ───────────────────────────────────────────────────────

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

    /// (G3) The regular polyhedra have closed-form α for every fold, derived from the candidate set in
    /// exact arithmetic in `derivations/analysis_alpha_exact.mac` and re-run at 60 digits in
    /// `derivations/analysis_alpha_exact.gp`. Tetrahedron: `acos(1/3)`, `π − acos(1/3)` (at a vertex
    /// the other three are equidistant), `π − acos(1/√3)`, `π`. Octahedron: `acos(1/√3)`, `π/2`,
    /// `π/2`, `π − acos(1/√3)` (at a face centre), `3π/4`, `π`. Each enclosure contains its value and
    /// is narrower than 1e-13 rad; `k = n` is `[PI, next_up(PI)]`, which contains π.
    #[test]
    fn the_regular_polyhedra_have_their_exact_alpha() {
        let pi = PI;
        let a3 = libm::acos(1.0 / 3.0);
        let a_r3 = libm::acos(1.0 / libm::sqrt(3.0));
        let cases: [(Vec<Vec3>, Vec<f64>); 2] = [
            (tetrahedron(), vec![a3, pi - a3, pi - a_r3, pi]),
            (
                octahedron(),
                vec![a_r3, pi / 2.0, pi / 2.0, pi - a_r3, 0.75 * pi, pi],
            ),
        ];
        for (sites, want) in cases {
            for (i, &w) in want.iter().enumerate() {
                let k = u32::try_from(i + 1).unwrap();
                let e = covering_radius(&sites, k).unwrap();
                assert!(
                    contains(e, w),
                    "{} sites, k = {k}: {e:?} misses {w}",
                    sites.len()
                );
                assert!(e.upper - e.lower < 1e-13, "k = {k}: {e:?} is wide");
            }
        }
    }

    /// (G3) One site: π. Two sites `d` apart: `π − d/2` for `k = 1`
    /// (`derivations/analysis_alpha_exact.mac`, `two_sites`), and π for `k = 2`. Antipodal: π/2 —
    /// the case the candidate set of plan 18's first draft answered with 0. At `π − 10⁻ᵐ`, `m = 3…13`,
    /// the closed form is `π/2 + δ/2` with `δ = atan2(s, c)` the separation the stored vectors
    /// `(1, 0, 0)` and `(−c, s, 0)` really have.
    #[test]
    fn one_and_two_sites_have_their_closed_forms() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let one = covering_radius(&[axis], 1).unwrap();
        assert!(one.lower == PI && one.upper > PI, "{one:?}");

        let antipodal = covering_radius(&[axis, -axis], 1).unwrap();
        assert!(contains(antipodal, PI / 2.0), "{antipodal:?}");
        assert!(antipodal.upper - antipodal.lower < 1e-14, "{antipodal:?}");
        let both = covering_radius(&[axis, -axis], 2).unwrap();
        assert!(both.lower == PI, "{both:?}");

        let sixty = Vec3::new(0.5, libm::sqrt(3.0) / 2.0, 0.0);
        let got = covering_radius(&[axis, sixty], 1).unwrap();
        assert!(contains(got, PI - PI / 6.0), "{got:?}");

        for m in 3..=13 {
            let delta = libm::pow(10.0, -f64::from(m));
            let (sin, cos) = libm::sincos(delta);
            let other = Vec3::new(-cos, sin, 0.0);
            let want = PI / 2.0 + 0.5 * libm::atan2(sin, cos);
            let got = covering_radius(&[axis, other], 1).unwrap();
            assert!(contains(got, want), "m = {m}: {got:?} misses {want}");
            assert!(got.upper - got.lower < 1e-14, "m = {m}: {got:?} is wide");
        }
    }

    /// (G3) Walker 2/1/0 holds two satellites at opposite ends of one orbit, so α₁ is π/2 at every
    /// instant; one plane turns rigidly, so the time axis costs nothing.
    #[test]
    fn walker_2_1_0_is_a_right_angle_at_every_instant() {
        let (_, sats) = walker_at(55.0, 1, 2, 0);
        let e = alpha_characteristic(&sats, 1, 1e-12).unwrap();
        assert!(contains(e, PI / 2.0), "{e:?}");
        assert!(e.upper - e.lower < 1e-12, "{e:?}");
        let motion = Motion::new(&sats);
        let mut sites = Vec::new();
        for j in 0..7 {
            motion.sites_at(f64::from(j) * 1234.5, &mut sites);
            let at = covering_radius(&sites, 1).unwrap();
            assert!(contains(at, PI / 2.0), "instant {j}: {at:?}");
        }
    }

    /// (S2) Input outside the domain is a typed refusal, never a number.
    #[test]
    fn a_fold_above_the_sites_a_zero_fold_and_bad_input_are_refused() {
        let sats = walker(2, 2);
        assert_eq!(
            alpha_characteristic(&sats, 5, 1e-6),
            Err(AlphaError::FoldAboveSites { fold: 5, sites: 4 })
        );
        assert_eq!(
            alpha_characteristic(&sats, 0, 1e-6),
            Err(AlphaError::ZeroFold)
        );
        assert_eq!(
            alpha_characteristic(&[], 1, 1e-6),
            Err(AlphaError::FoldAboveSites { fold: 1, sites: 0 })
        );
        for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                alpha_characteristic(&sats, 1, tol),
                Err(AlphaError::BadTolerance)
            );
        }
        let mut broken = sats.clone();
        broken[2].eccentricity = 1.0;
        assert_eq!(
            alpha_characteristic(&broken, 1, 1e-6),
            Err(AlphaError::BadOrbit { satellite: 2 })
        );
        broken[2].eccentricity = 0.0;
        broken[2].raan = Angle::from_radians(f64::NAN);
        assert_eq!(
            alpha_characteristic(&broken, 1, 1e-6),
            Err(AlphaError::BadOrbit { satellite: 2 })
        );

        let tet = tetrahedron();
        assert_eq!(
            covering_radius(&tet, 5),
            Err(AlphaError::FoldAboveSites { fold: 5, sites: 4 })
        );
        assert_eq!(covering_radius(&tet, 0), Err(AlphaError::ZeroFold));
        let mut bad = tet.clone();
        bad[1] = Vec3::ZERO;
        assert_eq!(
            covering_radius(&bad, 1),
            Err(AlphaError::BadSite { site: 1 })
        );
        bad[1] = Vec3::new(f64::INFINITY, 0.0, 0.0);
        assert_eq!(
            covering_radius(&bad, 1),
            Err(AlphaError::BadSite { site: 1 })
        );

        let inc = Angle::from_radians(1.0);
        assert_eq!(
            minimum_walker(1.0, inc, RE + 20_000e3, MU, 0, 4).unwrap_err(),
            AlphaError::ZeroFold
        );
        assert_eq!(
            minimum_walker(f64::NAN, inc, RE + 20_000e3, MU, 1, 4).unwrap_err(),
            AlphaError::BadFootprint
        );
        assert_eq!(
            minimum_walker(1.0, inc, -1.0, MU, 1, 4).unwrap_err(),
            AlphaError::BadOrbit { satellite: 0 }
        );
    }

    /// A deterministic stream of unit vectors (`SplitMix64`, uniform in the cube, normalised).
    struct Stream(u64);
    impl Stream {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            #[allow(clippy::cast_precision_loss)]
            let x = (z >> 11) as f64 / (1u64 << 53) as f64;
            2.0 * x - 1.0
        }
        fn unit(&mut self) -> Vec3 {
            loop {
                let v = Vec3::new(self.next_f64(), self.next_f64(), self.next_f64());
                let n = v.norm();
                if n > 0.1 && n <= 1.0 {
                    return v.scale(1.0 / n);
                }
            }
        }
    }

    /// The largest `k`-th distance on a Fibonacci grid, then a pattern-search ascent from the best
    /// ten points: a lower estimate of the true maximum from a different method.
    fn grid_and_ascent(sites: &[Vec3], k: usize, grid: u32) -> f64 {
        let dk = |x: Vec3| {
            let mut d: Vec<f64> = sites
                .iter()
                .map(|s| libm::acos(dot(x, *s).clamp(-1.0, 1.0)))
                .collect();
            d.sort_by(f64::total_cmp);
            d[k - 1]
        };
        let mut scored: Vec<(f64, Vec3)> = (0..grid)
            .map(|i| {
                let p = fibonacci_dir(i, grid);
                (dk(p), p)
            })
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut best = scored[0].0;
        for &(f0, x0) in scored.iter().take(10) {
            let (mut at, mut value, mut step) = (x0, f0, 2e-3);
            while step > 1e-13 {
                let east = unit(at.cross(if at.z.abs() < 0.9 {
                    Vec3::new(0.0, 0.0, 1.0)
                } else {
                    Vec3::new(1.0, 0.0, 0.0)
                }));
                let north = at.cross(east);
                let mut improved = false;
                for dir in [
                    east,
                    -east,
                    north,
                    -north,
                    unit(east + north),
                    unit(east - north),
                    -unit(east + north),
                    -unit(east - north),
                ] {
                    let next = unit(at + dir.scale(step));
                    let next_value = dk(next);
                    if next_value > value {
                        (at, value, improved) = (next, next_value, true);
                    }
                }
                if !improved {
                    step *= 0.5;
                }
            }
            best = best.max(value);
        }
        best
    }

    /// (S1) The candidate set is complete: on random sets with planted antipodal, coincident and
    /// near-antipodal (chord 1e-12) sites, a grid with ascent never finds a point above the upper
    /// end, which is within 1e-10 of the lower in the dot products. A near-antipodal pair's true
    /// anti-midpoint is not a candidate after the move onto its representative; its value lies within
    /// the move's `2Δ`, inside the enclosure.
    #[test]
    fn the_candidates_are_never_below_sampling_with_ascent() {
        let mut rng = Stream(20_261_124);
        for (n, antipodal, coincident, near) in [
            (3, 1, 0, 0),
            (4, 2, 0, 0),
            (5, 1, 1, 1),
            (6, 2, 1, 0),
            (7, 0, 0, 2),
            (8, 3, 1, 1),
        ] {
            let mut sites = Vec::new();
            for _ in 0..antipodal {
                let s = rng.unit();
                sites.extend([s, -s]);
            }
            for _ in 0..coincident {
                let s = rng.unit();
                sites.extend([s, s]);
            }
            for _ in 0..near {
                let s = rng.unit();
                let t = unit(s.cross(Vec3::new(0.3, -0.2, 0.9)));
                sites.extend([s, unit(-s + t.scale(1e-12))]);
            }
            while sites.len() < n {
                sites.push(rng.unit());
            }
            for k in 1..sites.len() {
                let e = covering_radius(&sites, u32::try_from(k).unwrap()).unwrap();
                let sampled = grid_and_ascent(&sites, k, 1500);
                // The upper end holds only if the candidate set is complete, and it is tight: a
                // missing critical point shows as a sampled point above it. Tightness is read in the
                // dot products the method decides on, since next to π acos is flat and the γ
                // allowance widens the ends to √(2γ) there.
                assert!(
                    sampled <= e.upper,
                    "n = {}, k = {k}: sampling found {sampled} above {e:?}",
                    sites.len()
                );
                let (c_lower, c_upper) = (libm::cos(e.lower), libm::cos(e.upper));
                assert!(
                    c_lower - c_upper < 1e-10,
                    "n = {}, k = {k}: {e:?}",
                    sites.len()
                );
            }
        }
    }

    // ── Time: the enclosure and the decision ───────────────────────────────────────────────────

    /// Regression rows, not a gate: the six Walker cases of the pass-12 digest (M0-2), certified there
    /// by the same method in Julia, agree with this enclosure to 5e-8°.
    #[test]
    fn the_six_walker_cases_agree_with_the_pass_12_table() {
        let tol = (5e-8_f64).to_radians();
        for (planes, per_plane, phasing, inc, k, lo, hi) in [
            (5, 1, 1, 43.66, 1, 69.154_390_663, 69.154_390_714),
            (7, 1, 5, 55.69, 2, 81.179_332_989, 81.179_333_045),
            (6, 1, 4, 53.13, 1, 66.421_855_025, 66.421_855_066),
            (3, 8, 1, 55.0, 1, 38.440_187_621, 38.440_187_676),
            (6, 4, 1, 55.0, 4, 57.558_073_465, 57.558_073_514),
            (6, 3, 2, 55.0, 1, 39.668_454_079, 39.668_454_128),
        ] {
            let (w, sats) = walker_at(inc, planes, per_plane, phasing);
            let e = walker_enclosure(&w, &sats, k, 0.5 * tol);
            let (got_lo, got_hi) = (e.lower.to_degrees(), e.upper.to_degrees());
            assert!(
                got_hi - got_lo <= 5e-8
                    && (got_lo - lo).abs() <= 5e-8
                    && (got_hi - hi).abs() <= 5e-8,
                "{}/{planes}/{phasing} k = {k}: [{got_lo}, {got_hi}] against [{lo}, {hi}]",
                planes * per_plane
            );
        }
    }

    /// (G3) The f64 code against the mathematics at sample points: α at the instants `u₀ = 0` and
    /// `u₀ = 3/10` (radians of argument of latitude) of the six Walker cases, evaluated by the
    /// candidate set at 60 digits in `derivations/analysis_alpha_exact.gp`, which checks these
    /// literals against its own values to 1e-21. Each enclosure contains its value and is narrower
    /// than 1e-12: the candidates of small location bound share their largest one.
    #[test]
    #[allow(clippy::excessive_precision)] // 22 digits as derivations/analysis_alpha_exact.gp holds them
    fn the_f64_code_holds_the_60_digit_values_of_six_walker_instants() {
        for (planes, per_plane, phasing, inc, k, at_0, at_3) in [
            (
                5,
                1,
                1,
                43.66,
                1,
                1.180_161_571_420_008_205_745,
                1.206_842_792_871_513_127_863,
            ),
            (
                7,
                1,
                5,
                55.69,
                2,
                1.360_449_745_023_500_894_703,
                1.400_138_747_559_720_321_611,
            ),
            (
                6,
                1,
                4,
                53.13,
                1,
                0.974_053_858_253_563_771_150_3,
                1.055_324_877_647_860_854_325,
            ),
            (
                3,
                8,
                1,
                55.0,
                1,
                0.660_583_505_607_649_021_361_7,
                0.659_622_607_312_107_632_511_4,
            ),
            (
                6,
                4,
                1,
                55.0,
                4,
                1.004_577_893_062_186_243_963,
                0.993_566_963_103_728_838_394_3,
            ),
            (
                6,
                3,
                2,
                55.0,
                1,
                0.649_868_304_158_779_344_489_2,
                0.655_389_278_908_896_844_123_4,
            ),
        ] {
            let (_, sats) = walker_at(inc, planes, per_plane, phasing);
            let motion = Motion::new(&sats);
            let n = sats[0].mean_motion();
            let mut sites = Vec::new();
            for (u0, want) in [(0.0, at_0), (0.3, at_3)] {
                motion.sites_at(u0 / n, &mut sites);
                let e = covering_radius(&sites, k).unwrap();
                assert!(
                    contains(e, want) && e.upper - e.lower < 1e-12,
                    "{}/{planes}/{phasing} at u0 = {u0}: {e:?} against {want}",
                    planes * per_plane
                );
            }
        }
    }

    /// (S1) One symmetry span of a Walker Delta is every instant: the enclosure over it and over a
    /// whole period agree, on patterns with and without antipodal pairs. A span too short would miss
    /// the worst instant and leave the span's enclosure below the period's.
    #[test]
    fn one_symmetry_span_is_every_instant() {
        for (planes, per_plane, phasing, k) in [
            (5, 1, 1, 1),
            (3, 2, 1, 1),
            (7, 1, 3, 2),
            (4, 2, 2, 1),
            (3, 3, 2, 1),
        ] {
            let (w, sats) = walker_at(52.0, planes, per_plane, phasing);
            let span = walker_enclosure(&w, &sats, k, 1e-8);
            let full = alpha_characteristic(&sats, u32::try_from(k).unwrap(), 1e-8).unwrap();
            assert!(
                span.lower <= full.upper && full.lower <= span.upper,
                "{planes}x{per_plane}/{phasing}: span {span:?} against period {full:?}"
            );
            // Both contain the worst case, so their best candidates are within either's width.
            let width = (span.upper - span.lower).max(full.upper - full.lower);
            assert!(
                (span.lower - full.lower).abs() <= width,
                "{span:?} against {full:?}"
            );
        }
    }

    /// (S2) The decision agrees with the enclosure: a footprint above the upper end is accepted, one
    /// below the lower end refused; and one between the ends, which neither side proves, is not
    /// accepted.
    #[test]
    fn the_decision_agrees_with_the_enclosure() {
        let mut sats = Vec::new();
        for (planes, per_plane, phasing, k) in [
            (5, 1, 1, 1u32),
            (2, 2, 1, 1),
            (3, 3, 1, 2),
            (4, 2, 1, 1),
            (6, 1, 4, 1),
        ] {
            let (w, generated) = walker_at(61.0, planes, per_plane, phasing);
            let e = walker_enclosure(&w, &generated, k as usize, 1e-6);
            assert_eq!(
                walker_verdict(&w, k, e.upper + 1e-5, &mut sats),
                Verdict::Covers,
                "{e:?}"
            );
            assert_eq!(
                walker_verdict(&w, k, e.lower - 1e-5, &mut sats),
                Verdict::Exceeds,
                "{e:?}"
            );
        }
        // Walker 4/2/1 holds α₁ at π/2 at every instant: at a footprint of exactly that value the
        // decision has nothing to spare and must not accept.
        let (w, _) = walker_at(61.0, 2, 2, 1);
        assert_ne!(walker_verdict(&w, 1, PI / 2.0, &mut sats), Verdict::Covers);
    }

    /// The planted unsafe cases (rule 16), regression rows. (a) Walker 7/7/5 at 55.69° for two-fold
    /// coverage has α between 81.1793329° and 81.1793335°, and the sampled estimate the search used
    /// to accept on (400 directions × 40 instants) reads 79.6°: at a footprint of 80.75°, between
    /// the two, the old acceptance passes and the exact one refuses. (b) The search itself: at 89.99°
    /// the old scan stopped at 4/1/0, whose sampled α₂ is 89.98° although one plane of four leaves its
    /// pole 90° from all four at every instant; the exact scan refuses it and every design up to
    /// seven satellites but 7/7/2, at 77.56°.
    #[test]
    fn the_sampled_acceptance_passes_designs_the_exact_one_refuses() {
        let theta = 80.75_f64.to_radians();
        let (w, sats) = walker_at(55.69, 7, 1, 5);
        assert!(
            sampled_alpha(&sats, 2, 400, 40) <= theta,
            "the comparator no longer under-reads this case"
        );
        let e = walker_enclosure(&w, &sats, 2, 1e-9);
        assert!(e.lower > theta, "{e:?}");
        let mut buf = Vec::new();
        assert_eq!(walker_verdict(&w, 2, theta, &mut buf), Verdict::Exceeds);

        let theta = 89.99_f64.to_radians();
        let (plane, ring) = walker_at(55.69, 1, 4, 0);
        assert!(sampled_alpha(&ring, 2, 400, 40) <= theta);
        assert_eq!(walker_verdict(&plane, 2, theta, &mut buf), Verdict::Exceeds);
        let d = minimum_walker(theta, w.inclination, w.semi_major_axis, MU, 2, 7)
            .unwrap()
            .expect("7/7/2 covers twice at 77.56 degrees");
        let (_, generated) = walker_at(55.69, d.planes, d.per_plane, d.phasing);
        let found = walker_enclosure(&d, &generated, 2, 1e-9);
        assert!(
            found.upper <= theta,
            "{}/{}/{} was accepted with α in {found:?}, above {theta}",
            d.total(),
            d.planes,
            d.phasing
        );
        assert_eq!((d.total(), d.planes, d.phasing), (7, 7, 2));
    }

    #[test]
    fn the_minimum_walker_search_visits_the_smallest_designs_first() {
        // Above π every design covers, so the first one the scan reaches — one satellite in one
        // plane — is the answer; anything that reorders or thins the enumeration returns more.
        let inc = Angle::from_radians(70.0_f64.to_radians());
        let w = minimum_walker(3.2, inc, RE + 20_000e3, MU, 1, 12)
            .unwrap()
            .expect("every design covers above pi");
        assert_eq!((w.planes, w.per_plane, w.total()), (1, 1, 1));
        // `PI` is below π, where one satellite's α sits; two in one plane, α = π/2, are the answer.
        let w = minimum_walker(PI, inc, RE + 20_000e3, MU, 1, 12)
            .unwrap()
            .expect("two satellites cover at PI");
        assert_eq!((w.planes, w.per_plane, w.total()), (1, 2, 2));
    }

    #[test]
    fn minimum_walker_finds_a_proven_covering_design() {
        let inc = Angle::from_radians(70.0_f64.to_radians());
        let a = RE + 20_000e3;
        let footprint = 1.2; // rad
        let w = minimum_walker(footprint, inc, a, MU, 1, 12)
            .unwrap()
            .expect("a covering Walker exists at 1.2 rad");
        let mut sats = Vec::new();
        w.generate(|e| sats.push(e));
        let e = alpha_characteristic(&sats, 1, 1e-9).unwrap();
        assert!(e.upper <= footprint, "{e:?}");
        assert!(w.total() <= 12);
        // A tighter footprint cannot be met with fewer satellites than a looser one.
        if let Some(tight) = minimum_walker(0.9, inc, a, MU, 1, 12).unwrap() {
            assert!(
                w.total() <= tight.total(),
                "looser footprint needs more satellites"
            );
        }
        // Nothing of twelve satellites covers with a pinhole.
        assert!(minimum_walker(0.05, inc, a, MU, 1, 12).unwrap().is_none());
    }

    #[test]
    fn alpha_is_bounded_by_a_covering_footprint() {
        // A Street-of-Coverage design continuously covers the sphere with footprint θ, so the
        // α-characteristic (the *minimum* covering radius) must not exceed θ at any instant.
        let soc = StreetOfCoverage {
            inclination: Angle::from_radians(86.4_f64.to_radians()),
            altitude: 780e3,
            min_elevation: Angle::from_radians(8.2_f64.to_radians()),
            earth_radius: RE,
            mu: MU,
            per_plane: None,
        };
        let g = soc.geometry().expect("feasible");
        let mut sats = Vec::new();
        soc.generate(|e| sats.push(e));
        let motion = Motion::new(&sats);
        let mut sites = Vec::new();
        for j in 0..3 {
            motion.sites_at(f64::from(j) * 997.0, &mut sites);
            let e = covering_radius(&sites, 1).unwrap();
            assert!(e.lower > 0.0);
            assert!(
                e.lower <= g.footprint_half_angle + 1e-12,
                "{e:?} exceeds {}",
                g.footprint_half_angle
            );
        }
    }

    #[test]
    fn alpha_is_monotone_in_fold_and_decreases_with_density() {
        let sparse = walker(2, 2); // 4
        let dense = walker(4, 4); // 16
        let a1 = alpha_characteristic(&sparse, 1, 1e-6).unwrap();
        let a2 = alpha_characteristic(&sparse, 2, 1e-6).unwrap();
        assert!(a2.upper >= a1.lower, "two-fold {a2:?} below single {a1:?}");
        let dense1 = alpha_characteristic(&dense, 1, 1e-6).unwrap();
        assert!(
            dense1.upper < a1.lower,
            "dense {dense1:?} not below sparse {a1:?}"
        );
    }

    #[test]
    fn a_single_satellite_cannot_cover_the_sphere() {
        // One satellite leaves the antipode of its sub-point uncovered: α₁ is π.
        let e = alpha_characteristic(&walker(1, 1), 1, 1e-6).unwrap();
        assert!(e.lower == PI && e.upper > PI, "{e:?}");
    }
}
