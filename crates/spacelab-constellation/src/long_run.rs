//! Long-run coverage statistics of a ground point under one circular orbit: the fraction of time
//! covered, the passes per revolution, the mean pass and the mean gap (plan 18 Ф31.F, F133 (c),
//! M0-3).
//!
//! **The torus.** Take the argument of latitude `u` as the clock (one revolution is `2π`) and the
//! longitude of the ascending node east of the ground point, `x = x₀ − κu`, with
//! `κ = (ω⊕ − Ω̇)/u̇ > 0`: the node drifts west relative to the Earth. The sub-satellite point has
//! latitude `φ_s(u) = asin(sin i sin u)` and lies `α(u) = atan2(cos i sin u, cos u)` east of the
//! node, `α` continued through the whole revolution. A footprint of Earth-central half-angle `λ`
//! covers a ground point at latitude `φ` when the arc between them is at most `λ`, which at the
//! sub-satellite latitude is a longitude half-width `Δ` with
//! `cos Δ = (cos λ − sin φ sin φ_s)/(cos φ cos φ_s)`: `Δ = 0` when `|φ − φ_s| ≥ λ` and `Δ = π` when
//! `|φ + φ_s| ≥ π − λ`. The code takes it in the half-angle form
//! `tan²(Δ/2) = sin a sin b / (cos c cos d)`, `a, b = (λ ± (φ − φ_s))/2`, `c, d = (φ + φ_s ± λ)/2`,
//! which needs no division at a pole and keeps its digits at the band edges. The point is covered at
//! `u` exactly when `h₋(u) ≤ x₀ ≤ h₊(u)` modulo `2π`, with `h±(u) = κu − α(u) ± Δ(u)`: the covered
//! set of the torus `(u, x₀)` lies between the two boundary curves
//! (`derivations/long_run_coverage.mac`).
//!
//! **Incommensurate rates: a space average.** When `κ` is irrational the line `x = x₀ − κu` is
//! equidistributed on the torus. Weyl's criterion, re-derived: the time average of `e^{i(mu + nx)}`
//! over `[0, T]` is `e^{inx₀}(e^{i(m − nκ)T} − 1)/(i(m − nκ)T)` when `m ≠ nκ`, which tends to 0;
//! for irrational `κ` and integers `m, n` the case `m = nκ` is `m = n = 0` alone; trigonometric
//! polynomials approximate the indicator of a set whose boundary has zero area from above and below
//! in mean, so its time average tends to its area over `4π²`, from every start. Hence:
//!
//! - the fraction covered is `P = (1/2π²) ∫₀^{2π} Δ(u) du`;
//! - a pass is entered and left through the boundary curves, and the line meets the level
//!   `x₀ + 2πℤ` of a curve `h` as often, in the long run, as the total variation of `h` over a
//!   revolution divided by `2π`; so the passes per revolution are M0-3's
//!   `ν = (TV(h₊) + TV(h₋))/(4π)`, i.e. `(1/4π) ∫ (|κ − α′ + Δ′| + |κ − α′ − Δ′|) du` over the `u`
//!   with `0 < Δ < π`;
//! - the mean pass is `P/ν` and the mean gap `(1 − P)/ν`, in revolutions.
//!
//! A floating-point `κ` is rational, so no input is irrational: [`Rates::Incommensurate`] is the
//! caller's declaration that the track does not repeat over the time that matters. `P` does not
//! depend on `κ`; `ν` does, and `|ν(κ) − ν(κ′)| ≤ |κ − κ′| L/π`, `L ≤ π` the length of the `u` with
//! `0 < Δ < π` over half a revolution, because adding `(κ − κ′)u` moves a total variation by at most
//! that of the added term.
//!
//! **Commensurate rates: a finite sum.** A repeat track ([`RepeatTrack`]) of `Np` revolutions in
//! `Nd` days has `κ = Nd/Np` with `gcd(Np, Nd) = 1`. Revolution `k` of a cycle meets the covered set
//! where `h₋(u) ≤ x₀ − 2πkNd/Np ≤ h₊(u)` modulo `2π`, `u` over one revolution, and as `k` runs over
//! the cycle, `kNd` runs over every residue modulo `Np`: at `u` the revolutions covered are the
//! points of the lattice `x₀ + gℤ`, `g = 2π/Np`, in `[h₋(u), h₊(u)]` (an interval shorter than `2π`,
//! so no point is counted twice). With `H = (h − x₀)/g`:
//!
//! - the time covered over the cycle is `∫ (⌊H₊⌋ − ⌈H₋⌉ + 1) du` over the `u` with `0 < Δ < π`,
//!   plus `Np` times the length of the `u` with `Δ = π`; on a monotone run of `H` the integral of
//!   `⌊H⌋` is the floor at the run's start times its length, plus or minus the length after each
//!   level the run crosses, at the root of `H = k`;
//! - each pass enters and leaves once, and the state "level ≤ `h₊`" changes exactly
//!   where `⌊H₊⌋` does (and "`h₋` ≤ level" where `⌈H₋⌉` does), so the passes over the cycle are
//!   half of `Σ (⌊max H₊⌋ − ⌊min H₊⌋) + Σ (⌈max H₋⌉ − ⌈min H₋⌉)` over the monotone runs.
//!
//! The answer depends on the ground point's longitude east of an ascending node, modulo one slot
//! `g`. Averaged over that longitude it is the incommensurate answer at `κ = Nd/Np` exactly: a
//! lattice of spacing `g` puts on average `ℓ/g` points in an interval of length `ℓ`. At any
//! longitude the count differs from `2Δ/g` by at most one, so `|P_rep − P| ≤ 1/Np`, and a run's
//! crossings differ from its range over `g` by less than one, so `|ν_rep − ν(Nd/Np)| < R/(2Np)`,
//! `R` the monotone runs of `h₊` and `h₋` over a revolution. Along a sequence `Nd/Np → κ*` the
//! commensurate answer tends to the incommensurate one at `κ*`, within `1/Np` for `P` and
//! `R/(2Np) + |Nd/Np − κ*| L/π` for `ν`.
//!
//! **Evaluation.** With `σ` the sign of `cos i`, `α(π − v) = σπ − α(v)`, so
//! `h±(π − v) = (κ − σ)π − h∓(v)` and both halves of a revolution come from `v ∈ [−π/2, π/2]`. That
//! interval is cut at the band edges (`φ_s = φ ± λ` and `±(π − λ) − φ`), where `Δ` has a square-root
//! end, and, to help the search below, where `α′ = κ` and where `Δ` is widest
//! (`sin φ_s = sin φ / cos λ`). `∫ Δ` is 64-point Gauss–Legendre on each piece after
//! `v = m + r sin θ`, which makes the square-root ends analytic. At a band edge `Δ` is set to its
//! exact 0 or `π` rather than evaluated at the rounded cut, where its square root would turn the
//! cut's rounding `ε` into an error of `√ε` in `h±` (2e-8 in `ν` on G3's first row). The
//! extrema of `h±` come from 64 samples per piece, clustered the same way, each sampled turn refined
//! by a golden-section search; a total variation over a partition that holds every extremum is
//! exact. Two extrema closer together than the samples are missed, which costs twice the height of
//! that wiggle; the gates bound what that has cost on their cases. Every number is under H-libm, not
//! outward-rounded ([`LongRun::LABEL`]).
//!
//! The gates are in `tests/long_run.rs`: the integrals and the finite sums against
//! `derivations/long_run_coverage.gp` at 60 digits (G2), a polar orbit over an equatorial point in
//! closed form (G3), and simulations along the trajectory within their sampling error (S1).

use crate::repeat_track::RepeatTrack;

const PI: f64 = core::f64::consts::PI;
const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
const TAU: f64 = core::f64::consts::TAU;
/// Gauss–Legendre nodes per piece for `∫ Δ`.
const QUADRATURE_NODES: usize = 64;
/// Sample intervals per piece from which the runs of `h±` are found.
const RUN_SAMPLES: usize = 64;
/// At most two ends, four band edges, two points with `α′ = κ` and the widest latitude.
const MAX_CUTS: usize = 9;

/// The ground point and the orbit's geometry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LongRunGeometry {
    /// The orbit's inclination `i`, radians, in `[0, π]`.
    pub inclination: f64,
    /// The ground point's latitude `φ`, radians, in `[−π/2, π/2]`.
    pub latitude: f64,
    /// The footprint's Earth-central half-angle `λ`, radians, in `(0, π)`.
    pub footprint: f64,
}

/// How the orbit's rates stand to each other; the case decides which formula applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rates {
    /// The rate ratio `κ = (ω⊕ − Ω̇)/u̇ > 0`, declared irrational: the track does not repeat, and the
    /// statistics are the same at every ground longitude (the integrals).
    Incommensurate {
        /// `κ`.
        kappa: f64,
    },
    /// A repeat track, `κ = Nd/Np` exactly: the statistics of one cycle at one ground longitude
    /// (the finite sums).
    Commensurate {
        /// The track.
        track: RepeatTrack,
        /// The ground point's longitude east of an ascending node of the track, radians; only its
        /// value modulo one slot `2π/Np` matters.
        ground_offset: f64,
    },
}

/// The long-run statistics, in revolutions of the argument of latitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LongRunStats {
    /// The fraction of time the ground point is covered.
    pub coverage: f64,
    /// Passes (entries into coverage) per revolution.
    pub passes_per_revolution: f64,
    /// The mean pass, `coverage / passes`, revolutions; `None` without passes.
    pub mean_pass: Option<f64>,
    /// The mean gap, `(1 − coverage) / passes`, revolutions; `None` without passes.
    pub mean_gap: Option<f64>,
}

/// The answer, with the case that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LongRun {
    /// From the integrals: the time average equals the space average.
    Incommensurate(LongRunStats),
    /// From the finite sums over one repeat cycle.
    Commensurate {
        /// The passes over one cycle of `Np` revolutions.
        passes_per_cycle: u64,
        /// The statistics.
        stats: LongRunStats,
    },
}

impl LongRun {
    /// What the numbers rest on: the `libm` elementary functions within their measured error
    /// (H-libm), round-to-nearest arithmetic, and the sampled search for the extrema of `h±`.
    pub const LABEL: &'static str = "under H-libm, not outward-rounded";

    /// The statistics, whichever the case.
    #[must_use]
    pub const fn stats(&self) -> &LongRunStats {
        match self {
            Self::Incommensurate(stats) | Self::Commensurate { stats, .. } => stats,
        }
    }
}

/// Why the statistics were not computed: input outside the domain is refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LongRunError {
    /// The inclination is not finite or not in `[0, π]`.
    BadInclination,
    /// The latitude is not finite or not in `[−π/2, π/2]`.
    BadLatitude,
    /// The footprint is not finite or not in `(0, π)`.
    BadFootprint,
    /// `κ` is not finite and positive.
    BadRateRatio,
    /// The ground offset is not finite.
    BadOffset,
}

impl core::fmt::Display for LongRunError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::BadInclination => "the inclination is not finite and in [0, pi]",
            Self::BadLatitude => "the latitude is not finite and in [-pi/2, pi/2]",
            Self::BadFootprint => "the footprint is not finite and in (0, pi)",
            Self::BadRateRatio => "the rate ratio kappa is not finite and positive",
            Self::BadOffset => "the ground offset is not finite",
        })
    }
}

impl core::error::Error for LongRunError {}

/// The long-run coverage statistics of a ground point (module docs).
///
/// # Errors
/// [`LongRunError`] for an input outside its domain.
pub fn long_run_coverage(
    geometry: &LongRunGeometry,
    rates: Rates,
) -> Result<LongRun, LongRunError> {
    let LongRunGeometry {
        inclination,
        latitude,
        footprint,
    } = *geometry;
    if !(inclination.is_finite() && (0.0..=PI).contains(&inclination)) {
        return Err(LongRunError::BadInclination);
    }
    if !(latitude.is_finite() && (-HALF_PI..=HALF_PI).contains(&latitude)) {
        return Err(LongRunError::BadLatitude);
    }
    if !(footprint.is_finite() && footprint > 0.0 && footprint < PI) {
        return Err(LongRunError::BadFootprint);
    }
    match rates {
        Rates::Incommensurate { kappa } => {
            if !(kappa.is_finite() && kappa > 0.0) {
                return Err(LongRunError::BadRateRatio);
            }
            let torus = Torus::new(inclination, latitude, footprint, kappa);
            Ok(LongRun::Incommensurate(torus.integrals()))
        }
        Rates::Commensurate {
            track,
            ground_offset,
        } => {
            if !ground_offset.is_finite() {
                return Err(LongRunError::BadOffset);
            }
            let torus = Torus::new(inclination, latitude, footprint, track.rate_ratio());
            let (passes_per_cycle, stats) = torus.finite_sums(track.revolutions(), ground_offset);
            Ok(LongRun::Commensurate {
                passes_per_cycle,
                stats,
            })
        }
    }
}

/// What `Δ` is on a piece of `v ∈ [−π/2, π/2]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Band {
    /// `Δ = 0`: the ground point is out of reach at that latitude.
    Out,
    /// `0 < Δ < π`: the boundary curves exist.
    Edge,
    /// `Δ = π`: covered at every longitude.
    Full,
}

/// A point of a boundary curve.
#[derive(Clone, Copy, Debug, Default)]
struct Knot {
    v: f64,
    h: f64,
}

/// The ends and extrema of one boundary curve on one piece, in order.
struct Knots {
    items: [Knot; RUN_SAMPLES + 1],
    len: usize,
}

impl Knots {
    fn as_slice(&self) -> &[Knot] {
        &self.items[..self.len]
    }
}

/// The ascending half `[−π/2, π/2]` cut into pieces.
struct Cuts {
    items: [f64; MAX_CUTS],
    len: usize,
}

impl Cuts {
    fn push(&mut self, v: f64) {
        self.items[self.len] = v;
        self.len += 1;
    }
}

/// A piece between consecutive cuts, its band, and `Δ` at its ends where an end is a band edge:
/// 0 next to [`Band::Out`] and `π` next to [`Band::Full`]. There `Δ` has a square-root end, so the
/// cut's rounding, `ε`, would move it by `√ε`.
#[derive(Clone, Copy, Debug)]
struct Piece {
    a: f64,
    b: f64,
    band: Band,
    end_delta: [Option<f64>; 2],
}

/// The pieces of the ascending half, in order.
struct Pieces {
    items: [Piece; MAX_CUTS - 1],
    len: usize,
}

impl Pieces {
    fn as_slice(&self) -> &[Piece] {
        &self.items[..self.len]
    }
}

struct Torus {
    sin_i: f64,
    cos_i: f64,
    /// The sign of `cos i`: the continued `α` gains `2πσ` per revolution.
    sigma: f64,
    phi: f64,
    lambda: f64,
    kappa: f64,
}

impl Torus {
    fn new(inclination: f64, latitude: f64, footprint: f64, kappa: f64) -> Self {
        let (sin_i, cos_i) = libm::sincos(inclination);
        Self {
            sin_i,
            cos_i,
            sigma: if cos_i < 0.0 { -1.0 } else { 1.0 },
            phi: latitude,
            lambda: footprint,
            kappa,
        }
    }

    /// `(sin v, cos v)`, exact at the turning points `±π/2`, which the double `π/2` misses by 6e-17.
    fn trig(v: f64) -> (f64, f64) {
        if v == HALF_PI {
            (1.0, 0.0)
        } else if v == -HALF_PI {
            (-1.0, 0.0)
        } else {
            libm::sincos(v)
        }
    }

    /// `Δ` at `v`: the sub-satellite latitude by `atan2`, which keeps its digits at the turning
    /// point, then the half-angle form.
    fn delta(&self, sin_v: f64, cos_v: f64) -> f64 {
        let phi_s = libm::atan2(self.sin_i * sin_v, libm::hypot(cos_v, self.cos_i * sin_v));
        let (phi, lambda) = (self.phi, self.lambda);
        let near =
            libm::sin(0.5 * (lambda + phi - phi_s)) * libm::sin(0.5 * (lambda - phi + phi_s));
        if near <= 0.0 {
            return 0.0;
        }
        let far = libm::cos(0.5 * (phi + phi_s + lambda)) * libm::cos(0.5 * (phi + phi_s - lambda));
        if far <= 0.0 {
            return PI;
        }
        2.0 * libm::atan2(libm::sqrt(near), libm::sqrt(far))
    }

    fn band(&self, v: f64) -> Band {
        let (s, c) = Self::trig(v);
        let d = self.delta(s, c);
        if d <= 0.0 {
            Band::Out
        } else if d >= PI {
            Band::Full
        } else {
            Band::Edge
        }
    }

    /// `h_sign(v) = κv − α(v) + sign·Δ(v)` on the ascending half.
    fn h(&self, sign: f64, v: f64) -> f64 {
        let (s, c) = Self::trig(v);
        self.kappa * v - libm::atan2(self.cos_i * s, c) + sign * self.delta(s, c)
    }

    /// `h_sign` at a piece's end, with `Δ` there when the end is a band edge.
    fn h_end(&self, sign: f64, v: f64, delta: Option<f64>) -> f64 {
        delta.map_or_else(
            || self.h(sign, v),
            |delta| {
                let (s, c) = Self::trig(v);
                self.kappa * v - libm::atan2(self.cos_i * s, c) + sign * delta
            },
        )
    }

    /// The pieces between consecutive distinct cuts, with their bands and edge values.
    fn pieces(&self) -> Pieces {
        let cuts = self.cuts();
        let blank = Piece {
            a: 0.0,
            b: 0.0,
            band: Band::Out,
            end_delta: [None; 2],
        };
        let mut pieces = Pieces {
            items: [blank; MAX_CUTS - 1],
            len: 0,
        };
        for w in cuts.items[..cuts.len].windows(2) {
            if w[1] > w[0] {
                pieces.items[pieces.len] = Piece {
                    a: w[0],
                    b: w[1],
                    band: self.band(0.5 * (w[0] + w[1])),
                    end_delta: [None; 2],
                };
                pieces.len += 1;
            }
        }
        let edge = |band: Band| match band {
            Band::Out => Some(0.0),
            Band::Full => Some(PI),
            Band::Edge => None,
        };
        for k in 0..pieces.len {
            if k > 0 {
                pieces.items[k].end_delta[0] = edge(pieces.items[k - 1].band);
            }
            if k + 1 < pieces.len {
                pieces.items[k].end_delta[1] = edge(pieces.items[k + 1].band);
            }
        }
        pieces
    }

    fn cuts(&self) -> Cuts {
        let mut cuts = Cuts {
            items: [0.0; MAX_CUTS],
            len: 0,
        };
        cuts.push(-HALF_PI);
        cuts.push(HALF_PI);
        if self.sin_i > 0.0 {
            let mut at_sine = |s: f64| {
                if s.abs() < 1.0 {
                    cuts.push(libm::asin(s));
                }
            };
            let (phi, lambda) = (self.phi, self.lambda);
            for edge in [
                phi - lambda,
                phi + lambda,
                PI - lambda - phi,
                lambda - PI - phi,
            ] {
                if edge.abs() < HALF_PI {
                    at_sine(libm::sin(edge) / self.sin_i);
                }
            }
            // α′(v) = cos i / (1 − sin²i sin²v) = κ.
            if self.cos_i > 0.0 && self.cos_i < self.kappa {
                let s = libm::sqrt((1.0 - self.cos_i / self.kappa) / (self.sin_i * self.sin_i));
                at_sine(s);
                at_sine(-s);
            }
            let cos_lambda = libm::cos(lambda);
            if cos_lambda > 0.0 {
                at_sine(libm::sin(phi) / (cos_lambda * self.sin_i));
            }
        }
        cuts.items[..cuts.len].sort_unstable_by(f64::total_cmp);
        cuts
    }

    /// The ends and extrema of `h_sign` on `[a, b]`, in order: consecutive knots bound monotone
    /// runs.
    fn knots(&self, sign: f64, piece: &Piece) -> Knots {
        let (a, b) = (piece.a, piece.b);
        let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
        let sample = |j: usize| -> f64 {
            match j {
                0 => a,
                j if j == RUN_SAMPLES => b,
                #[expect(clippy::cast_precision_loss, reason = "a sample index is below 2^52")]
                j => mid + half * libm::sin(-HALF_PI + PI * j as f64 / RUN_SAMPLES as f64),
            }
        };
        let mut values = [0.0; RUN_SAMPLES + 1];
        for (j, value) in values.iter_mut().enumerate() {
            *value = self.h(sign, sample(j));
        }
        values[0] = self.h_end(sign, a, piece.end_delta[0]);
        values[RUN_SAMPLES] = self.h_end(sign, b, piece.end_delta[1]);
        let mut knots = Knots {
            items: [Knot::default(); RUN_SAMPLES + 1],
            len: 1,
        };
        knots.items[0] = Knot { v: a, h: values[0] };
        let mut rising: Option<bool> = None;
        for j in 1..RUN_SAMPLES {
            let before = values[j] - values[j - 1];
            if before != 0.0 {
                rising = Some(before > 0.0);
            }
            let after = values[j + 1] - values[j];
            if let Some(up) = rising
                && after != 0.0
                && up != (after > 0.0)
            {
                let turn = self.extremum(sign, up, sample(j - 1), sample(j + 1), sample(j));
                knots.items[knots.len] = turn;
                knots.len += 1;
            }
        }
        knots.items[knots.len] = Knot {
            v: b,
            h: values[RUN_SAMPLES],
        };
        knots.len += 1;
        knots
    }

    /// A golden-section search for the maximum (`up`) or minimum of `h_sign` on `[lo, hi]`, never
    /// worse than the sample at `start`.
    fn extremum(&self, sign: f64, up: bool, lo: f64, hi: f64, start: f64) -> Knot {
        let orient = if up { 1.0 } else { -1.0 };
        let f = |v: f64| orient * self.h(sign, v);
        let ratio = 0.5 * (libm::sqrt(5.0) - 1.0);
        let (mut lo, mut hi) = (lo, hi);
        let mut left = hi - ratio * (hi - lo);
        let mut right = lo + ratio * (hi - lo);
        let (mut f_left, mut f_right) = (f(left), f(right));
        // 0.618^200 is far below one ulp of any bracket; the two probes meet long before.
        for _ in 0..200 {
            if left >= right {
                break;
            }
            if f_left > f_right {
                hi = right;
                right = left;
                f_right = f_left;
                left = hi - ratio * (hi - lo);
                f_left = f(left);
            } else {
                lo = left;
                left = right;
                f_left = f_right;
                right = lo + ratio * (hi - lo);
                f_right = f(right);
            }
        }
        let mut best = (start, f(start));
        for candidate in [(left, f_left), (right, f_right)] {
            if candidate.1 > best.1 {
                best = candidate;
            }
        }
        Knot {
            v: best.0,
            h: orient * best.1,
        }
    }

    /// `∫ Δ dv` over `[a, b]`: Gauss–Legendre after `v = mid + half·sin θ`.
    fn integral_of_delta(&self, a: f64, b: f64, rule: &[(f64, f64); QUADRATURE_NODES]) -> f64 {
        let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
        let mut sum = 0.0;
        for &(node, weight) in rule {
            let (sin_t, cos_t) = libm::sincos(HALF_PI * node);
            let (sin_v, cos_v) = Self::trig(mid + half * sin_t);
            sum += weight * self.delta(sin_v, cos_v) * cos_t;
        }
        HALF_PI * half * sum
    }

    /// The integrals of the incommensurate case.
    fn integrals(&self) -> LongRunStats {
        let rule = gauss_legendre();
        let mut delta_integral = 0.0;
        let mut variation = 0.0;
        for piece in self.pieces().as_slice() {
            let (a, b) = (piece.a, piece.b);
            match piece.band {
                Band::Out => {}
                Band::Full => delta_integral += PI * (b - a),
                Band::Edge => {
                    delta_integral += self.integral_of_delta(a, b, &rule);
                    for sign in [1.0, -1.0] {
                        let knots = self.knots(sign, piece);
                        variation += knots
                            .as_slice()
                            .windows(2)
                            .map(|w| (w[1].h - w[0].h).abs())
                            .sum::<f64>();
                    }
                }
            }
        }
        // Over both halves of a revolution: P = (1/π²) ∫ Δ dv and ν = (TV₊ + TV₋)/(2π), the
        // integral and the variations taken over the ascending half.
        stats(delta_integral / (PI * PI), variation / TAU)
    }

    /// The finite sums of the commensurate case: the passes over one cycle and the statistics.
    fn finite_sums(&self, revolutions: u32, ground_offset: f64) -> (u64, LongRunStats) {
        let np = f64::from(revolutions);
        let slot = TAU / np;
        // x₀, the node east of the ground point, modulo one slot.
        let x0 = {
            let r = libm::fmod(-ground_offset, slot);
            if r < 0.0 { r + slot } else { r }
        };
        let mirror = (self.kappa - self.sigma) * PI;
        let mut crossings: i64 = 0;
        let mut covered = 0.0;
        for piece in self.pieces().as_slice() {
            let (a, b) = (piece.a, piece.b);
            match piece.band {
                Band::Out => {}
                Band::Full => covered += 2.0 * np * (b - a),
                Band::Edge => {
                    let plus = self.knots(1.0, piece);
                    let minus = self.knots(-1.0, piece);
                    // The ascending half, then the descending one: there h±(π − v) =
                    // (κ − σ)π − h∓(v), over the same measure in v.
                    for descending in [false, true] {
                        let level = |h: f64| {
                            if descending {
                                (mirror - h - x0) / slot
                            } else {
                                (h - x0) / slot
                            }
                        };
                        // The upper curve comes from h₊ on the ascending half and from h₋ on the
                        // descending one.
                        let (upper_sign, upper, lower) = if descending {
                            (-1.0, &minus, &plus)
                        } else {
                            (1.0, &plus, &minus)
                        };
                        let lower_sign = -upper_sign;
                        for w in upper.as_slice().windows(2) {
                            let (ha, hb) = (level(w[0].h), level(w[1].h));
                            crossings += floor_i64(ha.max(hb)) - floor_i64(ha.min(hb));
                            covered += integral_of_floor(w[0].v, w[1].v, ha, hb, |v| {
                                level(self.h(upper_sign, v))
                            });
                        }
                        for w in lower.as_slice().windows(2) {
                            let (ha, hb) = (level(w[0].h), level(w[1].h));
                            crossings += ceil_i64(ha.max(hb)) - ceil_i64(ha.min(hb));
                            // −∫⌈H⌉ = ∫⌊−H⌋.
                            covered += integral_of_floor(w[0].v, w[1].v, -ha, -hb, |v| {
                                -level(self.h(lower_sign, v))
                            });
                        }
                        covered += b - a;
                    }
                }
            }
        }
        let passes = u64::try_from(crossings / 2).unwrap_or(0);
        #[expect(clippy::cast_precision_loss, reason = "a pass count is far below 2^52")]
        let per_revolution = passes as f64 / np;
        (passes, stats(covered / (TAU * np), per_revolution))
    }
}

fn stats(coverage: f64, passes_per_revolution: f64) -> LongRunStats {
    let has_passes = passes_per_revolution > 0.0;
    LongRunStats {
        coverage,
        passes_per_revolution,
        mean_pass: has_passes.then(|| coverage / passes_per_revolution),
        mean_gap: has_passes.then(|| (1.0 - coverage) / passes_per_revolution),
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a level index is far below 2^53"
)]
fn floor_i64(x: f64) -> i64 {
    libm::floor(x) as i64
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "a level index is far below 2^53"
)]
fn ceil_i64(x: f64) -> i64 {
    libm::ceil(x) as i64
}

/// `∫ ⌊H(v)⌋ dv` over a monotone run from `(v0, h0)` to `(v1, h1)`: the floor at the start over the
/// whole run, plus or minus the length after each integer level the run crosses, found by bisection.
fn integral_of_floor(v0: f64, v1: f64, h0: f64, h1: f64, level: impl Fn(f64) -> f64) -> f64 {
    let mut sum = libm::floor(h0) * (v1 - v0);
    let rising = h1 >= h0;
    let (from, to) = if rising {
        (floor_i64(h0) + 1, floor_i64(h1))
    } else {
        (floor_i64(h1) + 1, floor_i64(h0))
    };
    for k in from..=to {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a level index is far below 2^53"
        )]
        let target = k as f64;
        let (mut lo, mut hi) = (v0, v1);
        loop {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            if (level(mid) < target) == rising {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let root = 0.5 * (lo + hi);
        // Rising, ⌊H⌋ gains one after the root; falling, it loses one there.
        sum += if rising { v1 - root } else { -(v1 - root) };
    }
    sum
}

/// Gauss–Legendre nodes and weights on `[−1, 1]`, by Newton's method on `P_n`.
fn gauss_legendre() -> [(f64, f64); QUADRATURE_NODES] {
    #[expect(clippy::cast_precision_loss, reason = "a node count is far below 2^52")]
    let n = QUADRATURE_NODES as f64;
    let mut rule = [(0.0, 0.0); QUADRATURE_NODES];
    for (k, node) in rule.iter_mut().enumerate() {
        #[expect(clippy::cast_precision_loss, reason = "a node index is far below 2^52")]
        let mut x = libm::cos(PI * (k as f64 + 0.75) / (n + 0.5));
        let mut slope = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, x);
            for j in 2..=QUADRATURE_NODES {
                #[expect(clippy::cast_precision_loss, reason = "a degree is far below 2^52")]
                let j = j as f64;
                let p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j;
                p0 = p1;
                p1 = p2;
            }
            slope = n * (x * p1 - p0) / (x * x - 1.0);
            let step = p1 / slope;
            x -= step;
            if step.abs() <= 1e-16 {
                break;
            }
        }
        *node = (x, 2.0 / ((1.0 - x * x) * slope * slope));
    }
    rule
}
