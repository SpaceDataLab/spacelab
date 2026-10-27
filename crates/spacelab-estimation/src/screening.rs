//! Finding the conjunctions that [`conjunction`](crate::conjunction) then puts a probability on.
//!
//! `conjunction.rs` starts from a relative state *at the time of closest approach*. Something has to
//! produce that epoch, and this is it: the local minima of the separation, and the cheap filter that
//! decides which pairs are worth looking for them in.
//!
//! # The minimum is a sign change, not a search over distances
//!
//! Minimising `|Δr|` directly means a one-dimensional optimisation with no bracket and no guarantee
//! of which basin it lands in. Its derivative is better behaved: with `f(t) = ½|Δr|²`,
//!
//! ```text
//! f'(t) = Δr · Δv
//! ```
//!
//! so every closest approach is a zero of `Δr · Δv` where the sign goes **negative to positive** —
//! approaching, then receding. A maximum is the same zero with the sign the other way, and it is
//! discarded by the direction rather than by comparing distances afterwards. That turns the problem
//! into a bracketed root find, which cannot converge to the wrong kind of stationary point and
//! cannot fail to terminate.
//!
//! # Nothing here reports an empty sky by accident
//!
//! A screen has one failure mode that matters more than the rest: *no conjunctions found* is the
//! answer a caller acts on by doing nothing. A malformed step or an ephemeris that runs out mid-span
//! must therefore not be able to produce it, which is why [`closest_approaches`] returns
//! [`ScreeningError`] rather than quietly returning with nothing emitted. Every rejected argument
//! names itself and its value.
//!
//! # What the step size can and cannot promise
//!
//! The coarse scan sees a sign change only if the two samples straddle it. Two minima closer
//! together than `step` are one bracket, and the search reports one approach where there were two.
//! **This is not defended against, because it cannot be**: no sampled scan can bound what it did not
//! sample. The step is the caller's, and it must be shorter than the interval between successive
//! extrema of the separation — for two low-Earth orbits that is a few minutes, and for a co-orbiting
//! pair with a slow relative drift it can be much longer. `a_step_too_coarse_to_resolve_two_minima_
//! reports_one` pins the failure mode rather than pretending it is absent.
//!
//! # Gate
//!
//! `derivations/screening_tca.mac` shows that `Δr · Δv` is the derivative of `½|Δr|²`, that on
//! the coplanar circular fixture its negative-to-positive zeros are `(phase + 2πj)/(n₁ − n₂)` at
//! distance `|r₂ − r₁|`, and that the radial filter's bound is the shells' exact closest
//! distance; `derivations/screening_tca.gp` finds every such zero of the coplanar and the
//! 63.4°-inclined fixture at 60 digits. The gates are the module tests
//! `every_closest_approach_is_the_root_of_the_range_rate` (G2),
//! `a_rectilinear_pass_is_found_where_the_closed_form_puts_it` (G3) and
//! `the_radial_filter_never_drops_a_pair_that_could_conjunct` (G3).

/// One closest approach: when, how close, and how fast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloseApproach {
    /// Time of closest approach, in the caller's own timescale.
    pub tca: f64,
    /// Separation at `tca`, metres.
    pub distance: f64,
    /// Relative speed at `tca`, m/s — the short-encounter model's own validity depends on this
    /// being large enough that the relative motion is rectilinear over the encounter.
    pub relative_speed: f64,
}

/// Why a scan could not be run to the end of its span. Each variant names the argument that was
/// wrong and its value, so that "nothing found" is never the report for a malformed request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScreeningError {
    /// The sample interval is not a positive, finite number of seconds.
    Step(f64),
    /// The span needs more samples at this step than one scan will take. The step is well formed and
    /// the *pair* is not, which is why it is not [`ScreeningError::Step`].
    TooManySamples {
        /// The requested sample interval, seconds.
        step: f64,
        /// `ceil((end - start) / step)`, against a ceiling of [`MAX_SAMPLES`].
        requested: f64,
    },
    /// The refinement tolerance is not a positive, finite number of seconds.
    Tolerance(f64),
    /// `end - start` is zero, negative or not finite — which also covers either endpoint being NaN
    /// or infinite, since the difference is then never a finite positive number.
    Span {
        /// The requested start of the scan.
        start: f64,
        /// The requested end of the scan.
        end: f64,
    },
    /// The relative state was not finite at this epoch: the ephemeris stopped covering the span.
    /// Approaches already reported through `emit` before it stand; the span past it was not scanned.
    Coverage(f64),
}

impl core::fmt::Display for ScreeningError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Step(step) => {
                write!(f, "sample interval {step} is not positive and finite")
            }
            Self::TooManySamples { step, requested } => {
                write!(
                    f,
                    "a {step} s step over this span is {requested} samples, past the {MAX_SAMPLES} a scan will take"
                )
            }
            Self::Tolerance(tolerance) => {
                write!(
                    f,
                    "refinement tolerance {tolerance} is not positive and finite"
                )
            }
            Self::Span { start, end } => {
                write!(
                    f,
                    "scan span [{start}, {end}] is empty, inverted or not finite"
                )
            }
            Self::Coverage(epoch) => {
                write!(f, "the relative state is not finite at epoch {epoch}")
            }
        }
    }
}

impl core::error::Error for ScreeningError {}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[f64; 3]) -> f64 {
    libm::sqrt(dot(a, a))
}

/// Report every closest approach of `relative` in `[start, end]`, in time order.
///
/// `relative` returns the relative position and velocity `(Δr, Δv)` — object 2 minus object 1, in
/// one common inertial frame, the same convention [`ConjunctionGeometry`](crate::conjunction::ConjunctionGeometry)
/// takes. The scan samples every `step` seconds and refines each bracket by bisection until it is
/// narrower than `tolerance`; `emit` is called once per approach found.
///
/// Bisection rather than a secant or Brent step, deliberately: the bracket is the guarantee, and
/// halving it a fixed number of times makes the cost `ceil(log2(step/tolerance))` evaluations per
/// approach — 26 for a 60 s step refined to a microsecond — with no dependence on how the function
/// behaves inside. A superlinear step would be fewer evaluations on a smooth pair and unbounded on a
/// pathological one, and a screening run's cost has to be predictable.
///
/// # Errors
///
/// Returns [`ScreeningError`] for a step or tolerance that is not positive and finite, for a span
/// that is not a finite positive interval, for a step too small to advance the clock at this epoch,
/// and for a non-finite sample. The last of these ends the scan where the ephemeris did rather than
/// propagating a NaN into a bracket: an ephemeris that stops covering the span is a fact about the
/// input, and continuing past it would invent approaches out of comparisons against NaN. Whatever
/// was emitted before that epoch is still the caller's to keep.
pub fn closest_approaches<F>(
    relative: F,
    start: f64,
    end: f64,
    step: f64,
    tolerance: f64,
    emit: &mut dyn FnMut(CloseApproach),
) -> Result<(), ScreeningError>
where
    F: Fn(f64) -> ([f64; 3], [f64; 3]),
{
    // Written against `is_finite` rather than as a negated comparison so that NaN is rejected by
    // name: `!(step > 0.0)` and `step <= 0.0` agree on every value except NaN, which is exactly the
    // argument a caller passes by accident.
    if !step.is_finite() || step <= 0.0 {
        return Err(ScreeningError::Step(step));
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(ScreeningError::Tolerance(tolerance));
    }
    // The span rather than the two endpoints: `end - start` is NaN or infinite whenever either of
    // them is, so one test rejects an inverted span, an empty one, and every non-finite endpoint.
    let span = end - start;
    if !span.is_finite() || span <= 0.0 {
        return Err(ScreeningError::Span { start, end });
    }

    // The scan is a counted number of samples, not an accumulation. Two reasons, and the second is
    // the one that decides it. `t += step` drifts by an ulp per step — 2.9e6 steps in a day at 30 s
    // — while `start + i·step` commits one rounding per sample; and at a large epoch `t + step` can
    // round back to `t`, which makes the loop's termination depend on the same arithmetic it is
    // scanning with. The count is *clamped* rather than checked, because a bound that a comparison
    // can be mutated out of is not a bound; the comparison below only decides what to report.
    let requested = libm::ceil(span / step);
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [1, MAX_SAMPLES] on the line above, so neither can happen"
    )]
    let count = requested.clamp(1.0, MAX_SAMPLES) as usize;
    if requested > MAX_SAMPLES {
        return Err(ScreeningError::TooManySamples { step, requested });
    }

    let rate = |t: f64| {
        let (dr, dv) = relative(t);
        dot(&dr, &dv)
    };

    let mut previous_t = start;
    let mut previous = rate(start);
    if !previous.is_finite() {
        return Err(ScreeningError::Coverage(start));
    }

    for i in 1..=count {
        #[expect(
            clippy::cast_precision_loss,
            reason = "i <= MAX_SAMPLES = 1e6, exactly representable in f64"
        )]
        let t = (start + i as f64 * step).min(end);
        let current = rate(t);
        if !current.is_finite() {
            return Err(ScreeningError::Coverage(t));
        }
        // Negative to non-negative is a minimum; the other direction is a maximum and is skipped by
        // this test rather than by measuring distances afterwards.
        if previous < 0.0 && current >= 0.0 {
            emit(refine(&relative, previous_t, t, tolerance));
        }
        previous_t = t;
        previous = current;
    }
    Ok(())
}

/// The most samples one scan will take: eleven days at a one-second step, or a year at thirty
/// seconds. Past any screening window, and past it the answer a caller wants is a coarser step
/// rather than a longer wait.
///
/// Chosen low enough that its own boundary is reachable from a test. A ceiling no test can stand on
/// is a ceiling nothing checks the sense of, and `>` against `>=` at the boundary is exactly the
/// difference between refusing the largest legal scan and running it.
pub const MAX_SAMPLES: f64 = 1.0e6;

/// The most halvings any `f64` bracket can need. `f64::MAX` down to the smallest subnormal is 2 098
/// doublings, so no interval outlives this count whatever its width — and the loop is counted rather
/// than left to its own two exit tests, because a termination condition one wrong operator can
/// disable is a hang rather than a wrong answer, and a wrong answer is the only kind a test can see.
const MAX_BISECTIONS: usize = 2_100;

/// Bisect `[a, b]`, where `Δr·Δv` is negative at `a` and non-negative at `b`, to `tolerance`.
fn refine<F>(relative: &F, mut a: f64, mut b: f64, tolerance: f64) -> CloseApproach
where
    F: Fn(f64) -> ([f64; 3], [f64; 3]),
{
    for _ in 0..MAX_BISECTIONS {
        if b - a <= tolerance {
            break;
        }
        let mid = f64::midpoint(a, b);
        // A midpoint that is not strictly inside is the interval exhausting `f64`, not a failure to
        // converge; stopping there is the only honest answer and it terminates.
        if mid <= a || mid >= b {
            break;
        }
        let (dr, dv) = relative(mid);
        if dot(&dr, &dv) < 0.0 {
            a = mid;
        } else {
            b = mid;
        }
    }
    let tca = f64::midpoint(a, b);
    let (dr, dv) = relative(tca);
    CloseApproach {
        tca,
        distance: norm(&dr),
        relative_speed: norm(&dv),
    }
}

/// Whether two orbits are far enough apart in radius that no conjunction is possible — the first and
/// cheapest stage of an all-on-all screen (Hoots, Crawford & Roehrich 1984, filter 1).
///
/// All four arguments are geocentric distances in metres. The pair is separable when one object's
/// perigee clears the other's apogee by more than `threshold`, because the two spherical shells the
/// orbits are confined to do not then come within `threshold` of each other at any epoch, whatever
/// the phasing or the plane. `threshold` is the screening distance — the miss distance below which a
/// pair is worth propagating.
///
/// This is a *geometric certainty*, not a heuristic, which is what makes it safe to apply before any
/// propagation: it discards pairs that cannot conjunct rather than pairs that probably will not. The
/// converse does not hold and is not claimed — a pair that survives the filter usually still does not
/// conjunct, and that is what the later stages are for.
///
/// Returns `false` for any non-finite input, so a corrupt element set is propagated and examined
/// rather than silently dropped.
#[must_use]
pub fn radially_separated(
    perigee_a: f64,
    apogee_a: f64,
    perigee_b: f64,
    apogee_b: f64,
    threshold: f64,
) -> bool {
    if !perigee_a.is_finite()
        || !apogee_a.is_finite()
        || !perigee_b.is_finite()
        || !apogee_b.is_finite()
        || !threshold.is_finite()
    {
        return false;
    }
    perigee_a - apogee_b > threshold || perigee_b - apogee_a > threshold
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]
    // The dense-scan oracle indexes 200 000 samples; the counts are exact in `f64` a decade over.
    #![allow(clippy::cast_precision_loss)]

    use super::*;

    /// The separation, computed without [`norm`] or [`dot`].
    ///
    /// `norm(a)` is `sqrt(dot(a, a))`, so an oracle built on it shares the inner product with the
    /// search it is supposed to be checking — and a wrong `dot` would move both sides together. This
    /// is the same arithmetic written out separately, which is what makes the comparison mean
    /// something.
    fn separation(d: &[f64; 3]) -> f64 {
        libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
    }

    /// Two objects passing on straight lines: the minimum is at a time with a closed form.
    ///
    /// Object 2 crosses object 1's track at `closest` metres, moving at `speed`. Then
    /// `Δr(t) = (v t − offset, closest, 0)` and the minimum is at `t = offset / v`, exactly.
    fn rectilinear(closest: f64, speed: f64, offset: f64) -> impl Fn(f64) -> ([f64; 3], [f64; 3]) {
        move |t: f64| ([speed * t - offset, closest, 0.0], [speed, 0.0, 0.0])
    }

    fn collect<F>(
        relative: F,
        start: f64,
        end: f64,
        step: f64,
        tolerance: f64,
    ) -> Result<Vec<CloseApproach>, ScreeningError>
    where
        F: Fn(f64) -> ([f64; 3], [f64; 3]),
    {
        let mut out = Vec::new();
        closest_approaches(relative, start, end, step, tolerance, &mut |c| out.push(c))?;
        Ok(out)
    }

    /// A geometry with a closed-form answer, to the refinement tolerance and no worse (G3):
    /// `t = x₀/v` and the offset `b`, `derivations/screening_tca.mac` identity 2.
    #[test]
    fn a_rectilinear_pass_is_found_where_the_closed_form_puts_it() {
        let (closest, speed, offset) = (1_500.0, 12_000.0, 600_000.0);
        let found = collect(rectilinear(closest, speed, offset), 0.0, 120.0, 5.0, 1e-9)
            .expect("well-formed arguments");
        assert_eq!(found.len(), 1, "{found:?}");
        let expected = offset / speed;
        assert!(
            libm::fabs(found[0].tca - expected) < 1e-9,
            "{} s against the closed-form {expected} s",
            found[0].tca
        );
        assert!(
            libm::fabs(found[0].distance - closest) < 1e-3,
            "{} m against the {closest} m the geometry has",
            found[0].distance
        );
        assert!(libm::fabs(found[0].relative_speed - speed) < 1e-9);
    }

    const MU: f64 = 3.986_004_418e14;

    fn mean_motion(r: f64) -> f64 {
        libm::sqrt(MU / (r * r * r))
    }

    /// Two circular co-planar orbits, the second started `phase` radians ahead.
    ///
    /// Their separation is `√(r₁² + r₂² − 2r₁r₂ cos Δφ)` with `Δφ = (n₂ − n₁)t + phase`, so the
    /// minima are where the two align and the beat period is `2π/|n₂ − n₁|`. That period is what
    /// makes the geometry usable or not: a 40 km altitude difference beats over **7.9 days**, and a
    /// span short enough to scan at 30 s contains no minimum at all. The pair below is 500 km apart
    /// for that reason and no other.
    fn co_planar(r1: f64, r2: f64, phase: f64) -> impl Fn(f64) -> ([f64; 3], [f64; 3]) {
        let (n1, n2) = (mean_motion(r1), mean_motion(r2));
        move |t: f64| {
            let (s1, c1) = (libm::sin(n1 * t), libm::cos(n1 * t));
            let (s2, c2) = (libm::sin(n2 * t + phase), libm::cos(n2 * t + phase));
            (
                [r2 * c2 - r1 * c1, r2 * s2 - r1 * s1, 0.0],
                [
                    -r2 * n2 * s2 + r1 * n1 * s1,
                    r2 * n2 * c2 - r1 * n1 * c1,
                    0.0,
                ],
            )
        }
    }

    /// The same pair with object 2's plane tilted by `inc`, so that `Δr` and `Δv` are genuinely
    /// three-dimensional.
    ///
    /// Every other geometry here is planar, which leaves the third component of the inner product
    /// multiplying zero by zero — unexercised, in a screen that will be handed real orbits, no two
    /// of which share a plane.
    fn inclined(r1: f64, r2: f64, inc: f64, phase: f64) -> impl Fn(f64) -> ([f64; 3], [f64; 3]) {
        let (n1, n2) = (mean_motion(r1), mean_motion(r2));
        let (si, ci) = (libm::sin(inc), libm::cos(inc));
        move |t: f64| {
            let (s1, c1) = (libm::sin(n1 * t), libm::cos(n1 * t));
            let (s2, c2) = (libm::sin(n2 * t + phase), libm::cos(n2 * t + phase));
            (
                [r2 * c2 - r1 * c1, r2 * s2 * ci - r1 * s1, r2 * s2 * si],
                [
                    -r2 * n2 * s2 + r1 * n1 * s1,
                    r2 * n2 * c2 * ci - r1 * n1 * c1,
                    r2 * n2 * c2 * si,
                ],
            )
        }
    }

    /// The beat period of the co-planar pair, seconds.
    fn beat(r1: f64, r2: f64) -> f64 {
        core::f64::consts::TAU / libm::fabs(mean_motion(r2) - mean_motion(r1))
    }

    /// Every local minimum of the separation in `[0, span]`, by dense sampling of `|Δr|` itself.
    ///
    /// The oracle shares nothing with the search: it samples the separation rather than its
    /// derivative, and computes it with [`separation`] rather than [`norm`].
    fn dense_minima<F>(motion: &F, span: f64, samples: usize) -> Vec<f64>
    where
        F: Fn(f64) -> ([f64; 3], [f64; 3]),
    {
        let at = |i: usize| separation(&motion(span * i as f64 / samples as f64).0);
        let mut minima = Vec::new();
        for i in 1..samples {
            let (before, here, after) = (at(i - 1), at(i), at(i + 1));
            if here < before && here <= after {
                minima.push(span * i as f64 / samples as f64);
            }
        }
        minima
    }

    const DENSE_SAMPLES: usize = 200_000;

    /// Every minimum in the span is found, in order, and where a dense scan puts it.
    #[test]
    fn every_minimum_in_the_span_is_found_and_agrees_with_a_dense_scan() {
        let (r1, r2) = (7_000_000.0_f64, 7_500_000.0_f64);
        let motion = co_planar(r1, r2, 1.0);
        // Two full beats, so there are two alignments to find and the phase offset puts neither on
        // a boundary. The 500 km separation is what makes that span 118 000 s rather than 16 days.
        let span = 2.0 * beat(r1, r2);

        let found = collect(&motion, 0.0, span, 30.0, 1e-6).expect("well-formed arguments");
        assert!(found.len() >= 2, "only {} minima found", found.len());

        let minima = dense_minima(&motion, span, DENSE_SAMPLES);
        assert_eq!(
            found.len(),
            minima.len(),
            "search found {} minima, the dense scan {}",
            found.len(),
            minima.len()
        );

        // The dense scan resolves a time to half its own sample spacing, and no better.
        let resolution = span / DENSE_SAMPLES as f64;
        for (c, &t) in found.iter().zip(&minima) {
            assert!(
                libm::fabs(c.tca - t) <= resolution,
                "{} s against the dense scan's {t} s, past its own {resolution} s resolution",
                c.tca
            );
        }
        // In time order, which the caller is entitled to assume.
        for pair in found.windows(2) {
            assert!(pair[0].tca < pair[1].tca, "{found:?} is not in time order");
        }
    }

    /// The same agreement on a pair whose planes differ, where every component of `Δr · Δv` and of
    /// the reported distance carries weight.
    #[test]
    fn an_inclined_pair_exercises_the_out_of_plane_component() {
        let (r1, r2) = (7_000_000.0_f64, 7_500_000.0_f64);
        let motion = inclined(r1, r2, 63.4_f64.to_radians(), 1.0);
        // Four revolutions of the lower object: long enough for several encounters near the line of
        // nodes, short enough that a 15 s step resolves each of them.
        let span = 4.0 * core::f64::consts::TAU / mean_motion(r1);

        // The geometry has to be three-dimensional for this test to mean anything at all.
        let (dr, dv) = motion(0.3 * span);
        assert!(
            libm::fabs(dr[2]) > 1.0e5 && libm::fabs(dv[2]) > 1.0e2,
            "the out-of-plane components are {} m and {} m/s, too small to exercise anything",
            dr[2],
            dv[2]
        );

        let found = collect(&motion, 0.0, span, 15.0, 1e-6).expect("well-formed arguments");
        let minima = dense_minima(&motion, span, DENSE_SAMPLES);
        assert_eq!(
            found.len(),
            minima.len(),
            "search found {} minima, the dense scan {}",
            found.len(),
            minima.len()
        );

        let resolution = span / DENSE_SAMPLES as f64;
        for (c, &t) in found.iter().zip(&minima) {
            assert!(
                libm::fabs(c.tca - t) <= resolution,
                "{} s against the dense scan's {t} s, past its own {resolution} s resolution",
                c.tca
            );
            // And the distance it reports is the separation at its own epoch, computed the other
            // way — the check that the inner product behind `norm` is the right one.
            let (dr, dv) = motion(c.tca);
            assert!(
                libm::fabs(c.distance - separation(&dr)) < 1e-6,
                "reported {} m against {} m at the same epoch",
                c.distance,
                separation(&dr)
            );
            assert!(
                libm::fabs(c.relative_speed - separation(&dv)) < 1e-6,
                "reported {} m/s against {} m/s at the same epoch",
                c.relative_speed,
                separation(&dv)
            );
        }
    }

    /// Every reported TCA is the root of its defining equation `Δr · Δv = 0` (G2), against
    /// `derivations/screening_tca.gp`'s 60-digit roots for both fixtures — found there by bracketing
    /// and `solve`, and for the coplanar pair also equal to the closed form
    /// `(phase + 2πj)/(n₁ − n₂)` of `derivations/screening_tca.mac` to 1e-71 s. Refined to a
    /// nanosecond, the bisection lands within 1e-8 s of each root: the rate function's own `f64`
    /// rounding at `t ~ 10⁴ s` moves the zero by ~1e-9 s, and the bracket by half its width.
    #[test]
    #[allow(clippy::unreadable_literal, clippy::excessive_precision)]
    fn every_closest_approach_is_the_root_of_the_range_rate() {
        let (r1, r2) = (7_000_000.0_f64, 7_500_000.0_f64);
        let coplanar = [
            (
                9.4354218670352426058885362843425348310256e3,
                5.0000000000000000000000000000000000000000e5,
            ),
            (
                6.8719925909032060769792449923725679806299e4,
                5.0000000000000000000000000000000000000000e5,
            ),
        ];
        let inclined_roots = [
            (
                2.6208869132350397883093677246973521134988e3,
                4.3980727602879593264684420990878909719510e6,
            ),
            (
                5.6677604005698032884871739712022854621531e3,
                2.5039620872073022240860801215894125003317e6,
            ),
            (
                8.7120299743995419899117520498921000130110e3,
                6.8908334471539223905552161740778914161901e5,
            ),
            (
                1.1755789251315146736691887758014835668289e4,
                1.5976790330830187261311451494314740592132e6,
            ),
            (
                1.4801173619825771414046408603549989213674e4,
                3.5057096677496224934478107814990302103870e6,
            ),
            (
                1.7850183461553879822895935330511437398677e4,
                5.3560344705264782699330176744253306670609e6,
            ),
            (
                2.0904498389848473123263924361215582690967e4,
                7.0749034488644479457643868105362138545155e6,
            ),
        ];
        let span = 2.0 * beat(r1, r2);
        let found = collect(co_planar(r1, r2, 1.0), 0.0, span, 30.0, 1e-9).expect("well-formed");
        let inc = 63.4_f64.to_radians();
        let span_inclined = 4.0 * core::f64::consts::TAU / mean_motion(r1);
        let found_inclined = collect(inclined(r1, r2, inc, 1.0), 0.0, span_inclined, 15.0, 1e-9)
            .expect("well-formed");
        for (found, roots, label) in [
            (&found, &coplanar[..], "coplanar"),
            (&found_inclined, &inclined_roots[..], "inclined"),
        ] {
            assert_eq!(found.len(), roots.len(), "{label}: {found:?}");
            for (c, &(tca, distance)) in found.iter().zip(roots) {
                assert!(
                    libm::fabs(c.tca - tca) < 1e-8,
                    "{label}: TCA {} s against the root {tca} s",
                    c.tca
                );
                assert!(
                    libm::fabs(c.distance - distance) < 1e-12 * distance,
                    "{label}: {} m against {distance} m at the root",
                    c.distance
                );
            }
        }
    }

    /// A maximum of the separation is a zero of the same function with the other sign, and it must
    /// not be reported. Starting the same geometry half a period later puts a maximum first.
    #[test]
    fn a_maximum_of_the_separation_is_not_reported_as_an_approach() {
        let (closest, speed, offset) = (1_500.0, 12_000.0, 600_000.0);
        // Receding throughout: the derivative is positive everywhere, so there is no bracket.
        let found = collect(rectilinear(closest, speed, offset), 60.0, 120.0, 5.0, 1e-9)
            .expect("well-formed arguments");
        assert!(found.is_empty(), "{found:?}");
        // And approaching throughout, where it is negative everywhere.
        let found = collect(rectilinear(closest, speed, offset), 0.0, 40.0, 5.0, 1e-9)
            .expect("well-formed arguments");
        assert!(found.is_empty(), "{found:?}");
    }

    /// The documented failure mode, pinned: two minima closer together than the step are one bracket
    /// and are reported as one approach. A scan cannot bound what it did not sample, and a test that
    /// asserted otherwise would be asserting something false.
    #[test]
    fn a_step_too_coarse_to_resolve_two_minima_reports_one() {
        let (r1, r2) = (7_000_000.0_f64, 7_500_000.0_f64);
        let motion = co_planar(r1, r2, 1.0);
        let span = 2.0 * beat(r1, r2);

        let fine = collect(&motion, 0.0, span, 30.0, 1e-6).expect("well-formed arguments");
        let coarse = collect(&motion, 0.0, span, 0.75 * span, 1e-6).expect("well-formed arguments");
        assert!(
            coarse.len() < fine.len(),
            "a one-sample scan found {} of the {} minima, so this test no longer demonstrates the \
             limitation it documents",
            coarse.len(),
            fine.len()
        );
    }

    /// Every rejected argument names itself, and none of them is reported as an empty sky.
    ///
    /// The geometry is deliberately one that runs out of coverage at 40 s: under a *correct* guard
    /// it is never called, because each row is rejected before the scan starts. It is there so that
    /// a guard which stops rejecting cannot scan an unbounded span, and so that what it returns
    /// instead is a different error rather than a hang.
    #[test]
    fn every_malformed_argument_is_reported_as_itself() {
        let good = rectilinear(1_500.0, 12_000.0, 600_000.0);
        let covers_40_s = |t: f64| {
            if t > 40.0 {
                ([f64::NAN; 3], [f64::NAN; 3])
            } else {
                good(t)
            }
        };
        let cases = [
            (0.0, 120.0, 0.0, 1e-9, ScreeningError::Step(0.0)),
            (0.0, 120.0, -5.0, 1e-9, ScreeningError::Step(-5.0)),
            (0.0, 120.0, f64::NAN, 1e-9, ScreeningError::Step(f64::NAN)),
            (0.0, 120.0, 5.0, 0.0, ScreeningError::Tolerance(0.0)),
            (0.0, 120.0, 5.0, -1e-9, ScreeningError::Tolerance(-1e-9)),
            (
                0.0,
                120.0,
                5.0,
                f64::NAN,
                ScreeningError::Tolerance(f64::NAN),
            ),
            (
                0.0,
                0.0,
                5.0,
                1e-9,
                ScreeningError::Span {
                    start: 0.0,
                    end: 0.0,
                },
            ),
            (
                120.0,
                0.0,
                5.0,
                1e-9,
                ScreeningError::Span {
                    start: 120.0,
                    end: 0.0,
                },
            ),
            (
                0.0,
                f64::NAN,
                5.0,
                1e-9,
                ScreeningError::Span {
                    start: 0.0,
                    end: f64::NAN,
                },
            ),
            (
                f64::NAN,
                120.0,
                5.0,
                1e-9,
                ScreeningError::Span {
                    start: f64::NAN,
                    end: 120.0,
                },
            ),
            (
                0.0,
                f64::INFINITY,
                5.0,
                1e-9,
                ScreeningError::Span {
                    start: 0.0,
                    end: f64::INFINITY,
                },
            ),
            (
                f64::NEG_INFINITY,
                120.0,
                5.0,
                1e-9,
                ScreeningError::Span {
                    start: f64::NEG_INFINITY,
                    end: 120.0,
                },
            ),
        ];
        for (start, end, step, tolerance, expected) in cases {
            let outcome = collect(covers_40_s, start, end, step, tolerance);
            // NaN payloads are not equal to themselves, so the variant is compared by its debug
            // form; the finite rows still pin the value.
            assert_eq!(
                format!("{outcome:?}"),
                format!("{:?}", Err::<Vec<CloseApproach>, _>(expected)),
                "({start}, {end}, {step}, {tolerance})"
            );
        }
    }

    /// A step below the epoch's own resolution still scans, because the samples are indexed rather
    /// than accumulated.
    ///
    /// `ulp(1e16)` is 2 s, so `t += 1.0` at that epoch never moves at all and an accumulating scan
    /// would sit on its first sample forever. `start + i·step` reaches every representable epoch in
    /// the span, and the pass is still found — to the 2 s the type has left, which is the honest
    /// answer rather than an error.
    #[test]
    fn a_step_below_the_epochs_resolution_still_advances_because_the_scan_is_indexed() {
        let start = 1.0e16_f64;
        assert_eq!(
            (start + 1.0).to_bits(),
            start.to_bits(),
            "the epoch is not past f64's reach, so this test demonstrates nothing"
        );

        // A pass whose closest approach is 30 s into the span, expressed against `start` so that the
        // geometry itself carries no cancellation.
        let motion = move |t: f64| {
            let dt = t - start;
            (
                [12_000.0 * dt - 360_000.0, 1_500.0, 0.0],
                [12_000.0, 0.0, 0.0],
            )
        };
        let found = collect(motion, start, start + 60.0, 1.0, 1e-9).expect("well-formed arguments");
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            libm::fabs(found[0].tca - (start + 30.0)) <= 2.0,
            "{} s against {} s, past the 2 s ulp that is all the type has left",
            found[0].tca,
            start + 30.0
        );
    }

    /// A step so fine that the span would take more samples than a scan will run is refused by
    /// name, rather than started and abandoned.
    #[test]
    fn a_step_that_asks_for_more_samples_than_a_scan_will_take_is_refused() {
        let outcome = collect(
            rectilinear(1_500.0, 12_000.0, 600_000.0),
            0.0,
            60.0,
            1e-9,
            1e-12,
        );
        assert_eq!(
            outcome,
            Err(ScreeningError::TooManySamples {
                step: 1e-9,
                requested: 6.0e10
            }),
            "{outcome:?}"
        );
    }

    /// The ceiling is a ceiling and not a fence one short of it: exactly [`MAX_SAMPLES`] runs, and
    /// one more does not.
    ///
    /// The accepting half is the half worth paying for. Refusing the largest legal scan and running
    /// it are the same code with one character changed, and nothing but a scan that actually reaches
    /// the boundary separates them — which is the whole reason `MAX_SAMPLES` is a number a test can
    /// stand on.
    #[test]
    fn the_sample_ceiling_admits_exactly_its_own_count_and_no_more() {
        let evaluations = core::cell::Cell::new(0_usize);
        let end = MAX_SAMPLES;
        let at_the_ceiling = collect(counted_exact_pass(&evaluations), 0.0, end, 1.0, 1.0);
        assert!(at_the_ceiling.is_ok(), "{at_the_ceiling:?}");
        // One sample before the loop, then exactly `MAX_SAMPLES` of them; the pass at 3 s adds its
        // own bisection on top, and only the leading term is the point.
        #[expect(
            clippy::cast_precision_loss,
            reason = "the count is bounded by MAX_SAMPLES plus a bisection, well under 2^53"
        )]
        let taken = evaluations.get() as f64;
        assert!(
            taken > MAX_SAMPLES,
            "{taken} evaluations, so the scan did not reach the ceiling"
        );

        let one_past = collect(
            rectilinear(1_500.0, 12_000.0, 0.0),
            0.0,
            end + 1.0,
            1.0,
            1.0,
        );
        assert_eq!(
            one_past,
            Err(ScreeningError::TooManySamples {
                step: 1.0,
                requested: MAX_SAMPLES + 1.0
            }),
            "{one_past:?}"
        );
    }

    /// An ephemeris that stops covering the span ends the scan there, naming the epoch, and does not
    /// turn NaN comparisons into approaches. Both the first sample and a later one are checked,
    /// because the first is guarded before the loop and the rest inside it.
    #[test]
    fn a_non_finite_sample_ends_the_scan_rather_than_inventing_an_approach() {
        let good = rectilinear(1_500.0, 12_000.0, 600_000.0);
        let stops_at_40 = |t: f64| {
            if t > 40.0 {
                ([f64::NAN; 3], [f64::NAN; 3])
            } else {
                good(t)
            }
        };
        assert_eq!(
            collect(stops_at_40, 0.0, 120.0, 5.0, 1e-9),
            Err(ScreeningError::Coverage(45.0))
        );

        let bad_from_the_start = |_: f64| ([f64::NAN; 3], [f64::NAN; 3]);
        assert_eq!(
            collect(bad_from_the_start, 0.0, 120.0, 5.0, 1e-9),
            Err(ScreeningError::Coverage(0.0))
        );
    }

    /// Each error prints what it is, both directly and through `dyn Error`.
    #[test]
    fn every_error_names_its_own_argument() {
        let messages = [
            (
                ScreeningError::Step(-5.0),
                "sample interval -5 is not positive and finite",
            ),
            (
                ScreeningError::TooManySamples {
                    step: 1e-9,
                    requested: 6.0e10,
                },
                "a 0.000000001 s step over this span is 60000000000 samples, past the 1000000 a scan will take",
            ),
            (
                ScreeningError::Tolerance(0.0),
                "refinement tolerance 0 is not positive and finite",
            ),
            (
                ScreeningError::Span {
                    start: 120.0,
                    end: 0.0,
                },
                "scan span [120, 0] is empty, inverted or not finite",
            ),
            (
                ScreeningError::Coverage(45.0),
                "the relative state is not finite at epoch 45",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(format!("{error}"), expected);
            let dynamic: &dyn core::error::Error = &error;
            assert_eq!(format!("{dynamic}"), expected);
        }
    }

    /// A geometry whose whole bisection is exact in `f64`: the TCA is at 3 s and every intermediate
    /// midpoint is a power of two.
    fn counted_exact_pass(
        evaluations: &core::cell::Cell<usize>,
    ) -> impl Fn(f64) -> ([f64; 3], [f64; 3]) {
        move |t: f64| {
            evaluations.set(evaluations.get() + 1);
            ([1024.0 * t - 3072.0, 512.0, 0.0], [1024.0, 0.0, 0.0])
        }
    }

    /// The refinement costs what the cost model says it costs, and stops where it says it stops.
    ///
    /// Bisection was chosen over a superlinear step *because* the cost is `ceil(log2(step/tolerance))`
    /// evaluations and nothing else, so that claim is worth an assertion rather than a comment. The
    /// bracket here is `[0, 8]` at a 1 s tolerance, so three halvings leave `[2, 3]` and the answer
    /// at its centre — and both the epoch and the count are exact, which is what makes a loop that
    /// halves once more, or one that halves until `f64` runs out, visible at all.
    #[test]
    fn the_refinement_costs_what_the_cost_model_says() {
        let evaluations = core::cell::Cell::new(0_usize);
        let found = collect(counted_exact_pass(&evaluations), 0.0, 8.0, 8.0, 1.0)
            .expect("well-formed arguments");
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(
            found[0].tca.to_bits(),
            2.5_f64.to_bits(),
            "{} s, where three exact halvings of [0, 8] put the bracket centre at 2.5",
            found[0].tca
        );
        // Two samples for the scan, `ceil(log2(8/1)) = 3` midpoints, one final state at the answer.
        assert_eq!(evaluations.get(), 6);
    }

    /// A tolerance finer than the type stops where the type stops, in a bounded number of steps.
    ///
    /// Asking for more digits than `f64` holds is a caller's mistake that the loop has to survive
    /// rather than diagnose, and the midpoint test is what ends it. The count is asserted alongside
    /// the answer because a midpoint test that stops working is not a wrong answer — it is a loop
    /// that runs to `MAX_BISECTIONS`, and only the count says so.
    #[test]
    fn a_tolerance_finer_than_the_type_stops_at_the_types_own_resolution() {
        let evaluations = core::cell::Cell::new(0_usize);
        let found = collect(counted_exact_pass(&evaluations), 0.0, 8.0, 8.0, 1e-30)
            .expect("well-formed arguments");
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(
            libm::fabs(found[0].tca - 3.0) <= 4.5e-16,
            "{} s against the exact 3 s, further than one ulp",
            found[0].tca
        );
        // 8 s halved down to ulp(3) is 54 steps; with the scan's two samples and the final state,
        // 57. The bound is the assertion — an unbounded loop here would be 2 103.
        assert!(
            evaluations.get() < 64,
            "{} evaluations to exhaust the bracket",
            evaluations.get()
        );
    }

    /// A sample landing exactly on the minimum is still a bracket. `Δr·Δv` is exactly zero there, so
    /// the test that opens a bracket has to accept zero on the right — `previous < 0 && current >= 0`
    /// and not `> 0`, which would step over its own answer.
    #[test]
    fn a_sample_exactly_at_the_minimum_still_opens_a_bracket() {
        let (closest, speed, offset) = (1_500.0, 12_000.0, 600_000.0);
        let exact = offset / speed;
        // A step that lands on `exact` to the bit: it is a sample, not an interior point.
        let found = collect(
            rectilinear(closest, speed, offset),
            0.0,
            2.0 * exact,
            exact,
            1e-9,
        )
        .expect("well-formed arguments");
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(libm::fabs(found[0].tca - exact) < 1e-9);
    }

    /// The radial filter discards only what it is entitled to, checked against the exact answer (G3):
    /// `|a − b|² − (|a| − |b|)² = 2|a||b|(1 − cos θ) ≥ 0` (`derivations/screening_tca.mac`,
    /// identity 5) makes `max(p_a − A_b, p_b − A_a)` the shells' exact closest distance.
    ///
    /// For two concentric circular orbits the closest approach over *all* time is `|r₂ − r₁|`
    /// exactly, reached whenever the two align. That closed form is the oracle here rather than a
    /// scan, and deliberately: at a 1 km altitude difference the two beat against each other with a
    /// period of 315 days, so a scan short enough to run would sample a fraction of one beat and
    /// report whatever separation the pair happened to start at.
    #[test]
    fn the_radial_filter_never_drops_a_pair_that_could_conjunct() {
        let threshold = 5_000.0;
        let r1 = 7_000_000.0_f64;
        for gap in [1.0_f64, 4_000.0, 4_999.0, 5_000.0, 5_001.0, 10_000.0, 5.0e5] {
            let r2 = r1 + gap;
            // Circular, so perigee and apogee are both the radius.
            let separated = radially_separated(r1, r1, r2, r2, threshold);
            assert_eq!(
                separated,
                gap > threshold,
                "at a {gap} m gap the filter said {separated}, and the exact closest approach of \
                 two concentric circles is the gap itself"
            );
        }

        // And the closed form is the one the search agrees with, on the one pair whose beat period
        // is short enough to scan: two alignments in the span, both at |r₂ − r₁|.
        let r2 = r1 + 5.0e5;
        let closest = collect(co_planar(r1, r2, 1.0), 0.0, 2.0 * beat(r1, r2), 30.0, 1e-6)
            .expect("well-formed arguments")
            .into_iter()
            .map(|c| c.distance)
            .fold(f64::INFINITY, f64::min);
        assert!(
            libm::fabs(closest - 5.0e5) < 1.0,
            "the search's closest approach is {closest} m against the exact 500000 m"
        );
    }

    /// The filter is symmetric in its two objects, needs *either* ordering to separate, and refuses
    /// to act on a non-finite input.
    #[test]
    fn the_radial_filter_is_symmetric_and_refuses_non_finite_input() {
        let (lo_p, lo_a, hi_p, hi_a) = (7.0e6, 7.1e6, 7.2e6, 7.3e6);
        assert!(radially_separated(lo_p, lo_a, hi_p, hi_a, 50_000.0));
        assert!(radially_separated(hi_p, hi_a, lo_p, lo_a, 50_000.0));
        // Overlapping shells cannot be separated at any threshold.
        assert!(!radially_separated(7.0e6, 7.3e6, 7.1e6, 7.2e6, 0.0));
        // Exactly at the threshold is not past it: the comparison is strict.
        assert!(!radially_separated(hi_p, hi_a, lo_p, lo_a, hi_p - lo_a));
        assert!(radially_separated(
            hi_p,
            hi_a,
            lo_p,
            lo_a,
            hi_p - lo_a - 1.0
        ));
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(!radially_separated(bad, lo_a, hi_p, hi_a, 1.0));
            assert!(!radially_separated(lo_p, bad, hi_p, hi_a, 1.0));
            assert!(!radially_separated(lo_p, lo_a, bad, hi_a, 1.0));
            assert!(!radially_separated(lo_p, lo_a, hi_p, bad, 1.0));
            assert!(!radially_separated(lo_p, lo_a, hi_p, hi_a, bad));
        }
    }
}
