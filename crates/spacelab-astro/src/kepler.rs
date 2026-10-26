//! Universal-variable two-body propagation.
//!
//! One formulation advances any conic — ellipse, parabola, hyperbola — through a single branchless
//! path, by solving the universal Kepler equation for the universal anomaly `χ` and forming the
//! Lagrange coefficients `f, g, ḟ, ġ`. This is the primitive Lambert and patched conics build on, and
//! the two-body reference a high-fidelity propagator is checked against. See Vallado,
//! *Fundamentals of Astrodynamics and Applications*, Algorithm 8; Curtis, *Orbital Mechanics for
//! Engineering Students*, §3.7.
//!
//! `derivations/kepler_universal.mac` shows on all three conics that `dF/dχ` is the coded `fp` and
//! the new radius, and that the Lagrange coefficients keep `fġ − ḟg = 1`; the Stumpff reductions
//! are in `derivations/stumpff_quartering.mac`. The gates are the module tests
//! `the_universal_anomaly_is_the_root_of_kepler_s_equation` (G2), against
//! `derivations/kepler_universal.gp`'s 100-digit roots, and
//! `the_lagrange_coefficients_keep_their_wronskian` (G4).

// Single-letter names are the standard notation of the universal-variable formulation.
#![allow(clippy::many_single_char_names)]

use spacelab_core::Vec3;

/// Below this the five-term series is exact to within half an ulp; `1/32` is a power of four, so the
/// quartering that reduces `z` into this window is itself exact.
const SERIES_WINDOW: f64 = 1.0 / 32.0;

/// Above this the closed forms have stopped cancelling and take over. See [`stumpff`].
const CLOSED_FORM_FLOOR: f64 = 1.0;

/// Both Stumpff functions, `C(z) = (1 − cos√z)/z` and `S(z) = (√z − sin√z)/√z³`.
///
/// Near zero the closed forms cannot be used, and not just at zero: `1 − cos√z` subtracts two numbers
/// that agree to `1 − z/2`, so it keeps only `z/2` of its significand and the quotient carries a
/// relative error of about `2ε/z`. Measured against 60-digit values over 600 samples a decade, the
/// closed forms are out by 9.6e5 ulps on `|z| ∈ [1e−6, 1e−2]` and 232 ulps on `[1e−2, 0.5]` — and
/// that band is exactly where a near-parabolic arc lives. On `e = 1.0001` at a hundred seconds,
/// `z = −1.16e−6` and the propagated position was 4.4e-9 km out, against 5.4e-13 km with `C` and `S`
/// evaluated exactly.
///
/// Argument reduction has no such term. `z` is quartered until it is inside the series window and the
/// two identities
///
/// ```text
/// C(4z) = (1 − z·S(z))²/2        S(4z) = [S(z) + C(z)·(1 − z·S(z))]/4
/// ```
///
/// unfold it again; every operation there is a square or an addition of like-signed quantities, so
/// nothing cancels. That costs a doubling of the relative error per fold, which is free while `z` is
/// small and is why the closed forms take back over at `|z| > 1`: the same 600-sample sweep puts
/// both methods within 2.1 ulps at the seam, and beyond it the unfolding is the worse of the two
/// (14 ulps against 1.9 for `S` on `[1e3, 1e7]`) because it has fifteen foldings to amplify.
///
/// Zero is not the only place `1 − cos√z` cancels. `C` vanishes to second order at every
/// `z = (2kπ)²` — one whole revolution of the universal anomaly, `z = 4π²` — and there the
/// subtraction keeps nothing at all: measured against 60-digit values, the difference form is out by
/// 5.2e-6 relative a part in 1e5 below `4π²`, by 100 % a part in 1e9 below it, and returns a bare
/// `0.0` at the ceiling the Lambert sweep probes. The half-angle identity `1 − cos θ = 2 sin²(θ/2)`
/// removes the subtraction outright and carries 1.8e-10 and 2.2e-7 at those two points, four to
/// seven orders better, with the same behaviour at `16π²`, `36π²`, `64π²` and `100π²`. It costs
/// nothing anywhere else: swept 600 samples a decade over `z ∈ [1, 1e7]` the two forms agree to
/// within a factor of 1.3 in worst relative error, and on `[1, 10)` they differ by one ulp.
///
/// That mattered outside this function. `lambert`'s bracket probes `z = 4π²(1 − 1e−9)`, where the
/// difference form made `C = 0`, `y` `NaN` and the residual `NaN` — read as *below the root*, so a
/// long-way transfer whose true residual there is `+2.8e28` was reported as having no root at all
/// and refused.
///
/// That last comparison is the circular branch. On the hyperbolic one the closed form has no
/// cancellation to escape and instead inherits the error of `√−z`, amplified by `sinh`, so the two
/// methods converge: swept the same way over `z ∈ [−1e7, −0.1]`, the worst ulp count per decade
/// differs by less than a factor of 1.5 either way, and beyond `|z| ≈ 5e5` both overflow with
/// `cosh`. The threshold is symmetric because nothing in the measurement asks it not to be.
///
/// Battin, *An Introduction to the Mathematics and Methods of Astrodynamics*, §4.5; the identities
/// are derived in `derivations/stumpff_quartering.mac`.
#[inline]
pub(crate) fn stumpff(z: f64) -> (f64, f64) {
    if z > CLOSED_FORM_FLOOR {
        let x = libm::sqrt(z);
        let half = libm::sin(0.5 * x);
        return (2.0 * half * half / z, (x - libm::sin(x)) / (x * x * x));
    }
    if z < -CLOSED_FORM_FLOOR {
        let x = libm::sqrt(-z);
        return (
            (libm::cosh(x) - 1.0) / (-z),
            (libm::sinh(x) - x) / (x * x * x),
        );
    }
    let mut w = z;
    let mut folds = 0u32;
    while libm::fabs(w) > SERIES_WINDOW {
        w *= 0.25;
        folds += 1;
    }
    // `C = Σ (−z)^k/(2k+2)!`, `S = Σ (−z)^k/(2k+3)!`, both to `z⁴`. Inside the window the first
    // dropped term is 1.2e-16 of `C` and 2.9e-17 of `S`.
    let mut c = 0.5 - w / 24.0 + w * w / 720.0 - w * w * w / 40_320.0 + w * w * w * w / 3_628_800.0;
    let mut s = 1.0 / 6.0 - w / 120.0 + w * w / 5040.0 - w * w * w / 362_880.0
        + w * w * w * w / 39_916_800.0;
    for _ in 0..folds {
        let t = 1.0 - w * s;
        s = (s + c * t) * 0.25;
        c = t * t * 0.5;
        w *= 4.0;
    }
    (c, s)
}

/// Stumpff function `C(z) = (1 − cos√z)/z`, with the removable singularity at `z = 0` and the
/// hyperbolic branch for `z < 0`.
pub(crate) fn stumpff_c(z: f64) -> f64 {
    stumpff(z).0
}

/// Stumpff function `S(z) = (√z − sin√z)/√z³`, with the `z = 0` limit `1/6` and the hyperbolic branch.
pub(crate) fn stumpff_s(z: f64) -> f64 {
    stumpff(z).1
}

/// `a·u + b·w`.
#[inline]
fn comb(a: f64, u: Vec3, b: f64, w: Vec3) -> Vec3 {
    Vec3::new(a * u.x + b * w.x, a * u.y + b * w.y, a * u.z + b * w.z)
}

/// The universal anomaly `χ` solving Kepler's equation for the span `dt`, and the number of solver
/// iterations it took.
///
/// The count is not diagnostics. The loop below is safeguarded, so its answer does not depend on the
/// initial guess, on the Newton step, or on which steps get accepted --- corrupt any of them and
/// bisection still walks the same bracket onto the same root. What those corruptions change is the
/// *cost*, and the count is the only place it shows: it is what
/// `the_solver_reaches_its_root_by_newton_and_not_by_bisection` asserts, and without it two dozen
/// mutations of this function are indistinguishable from the function.
pub(crate) fn universal_anomaly(r: Vec3, v: Vec3, mu: f64, dt: f64) -> (f64, u32) {
    let r0 = r.norm();
    let v0 = v.norm();
    let vr0 = r.dot(v) / r0;
    let alpha = 2.0 / r0 - v0 * v0 / mu; // reciprocal semi-major axis, 1/a
    let sqmu = libm::sqrt(mu);

    let guess = if alpha < -1e-9 {
        // Hyperbolic guess: a = 1/alpha < 0.
        let a = 1.0 / alpha;
        let s = if dt >= 0.0 { 1.0 } else { -1.0 };
        s * libm::sqrt(-a)
            * libm::log(
                -2.0 * mu * alpha * dt / (r.dot(v) + s * libm::sqrt(-mu * a) * (1.0 - r0 * alpha)),
            )
    } else {
        // Elliptic guess `√μ·α·dt`, which for `α ≥ 0` is this expression bit for bit, and the
        // near-parabolic fallback for the sliver below `1e-9`.
        sqmu * libm::fabs(alpha) * dt
    };

    // The universal Kepler equation as a root problem in `χ`. Its derivative is `r(χ)`, the radius,
    // which is positive on every conic --- so `F` is strictly increasing, its root is unique, and a
    // bracket around it can be walked out and then never lost.
    let kepler = |chi: f64| -> (f64, f64) {
        let z = alpha * chi * chi;
        let (c, s) = stumpff(z);
        let f =
            (r0 * vr0 / sqmu) * chi * chi * c + (1.0 - alpha * r0) * chi * chi * chi * s + r0 * chi
                - sqmu * dt;
        let fp = (r0 * vr0 / sqmu) * chi * (1.0 - z * s) + (1.0 - alpha * r0) * chi * chi * c + r0;
        (f, fp)
    };

    // Bracket the root. `F(0) = −√μ·dt`, so the root lies on the side of zero that `dt` points to;
    // double outward from the guess until `F` changes sign. `F` grows without bound on both
    // branches, so a hundred doublings from a magnitude of at least one cannot fail to bracket.
    let dir = if dt >= 0.0 { 1.0 } else { -1.0 };
    let mut inner = 0.0;
    let mut outer = dir * f64::max(libm::fabs(guess), 1.0);
    let mut steps = 0;
    for _ in 0..100 {
        steps += 1;
        if dir * kepler(outer).0 >= 0.0 {
            break;
        }
        inner = outer;
        outer *= 2.0;
    }
    let (mut lo, mut hi) = (f64::min(inner, outer), f64::max(inner, outer));

    // Safeguarded Newton: keep the quadratic rate where the tangent behaves, bisect where it does
    // not. Plain Newton from the guess above is not safe on a near-parabolic arc --- there the
    // elliptic guess is the small-eccentricity approximation `√a·M`, while the root is `√a·ΔE ≫
    // √a·M`, and the first step overshoots by four orders. From there the iterates wander over six
    // decades of `z`, and on a thin set of spans they do not come back inside a hundred steps: at
    // `e = 0.999` and `dt = 1e7 s` the unsafeguarded loop returned a position 6.6e20 km from the
    // true one, having conserved neither energy nor angular momentum.
    //
    // The bracket cannot be lost, whatever the step does: `lo` and `hi` are reassigned by the *sign*
    // of `F` and not by position, so `F(lo) ≤ 0 ≤ F(hi)` survives a step that lands outside the
    // interval as well as one that lands inside it. The stop is the fixed point and not a tolerance,
    // for the same reason as in `lambert`: a step floor stops short of the root by whatever that
    // floor happens to be worth in the state.
    let mut chi = guess.clamp(lo, hi);
    for _ in 0..100 {
        steps += 1;
        let (f, fp) = kepler(chi);
        if f < 0.0 {
            lo = chi;
        } else {
            hi = chi;
        }
        let newton = chi - f / fp;
        // The fixed-point test comes before the containment test, and has to. `χ` has just become
        // one end of the bracket, so a converged Newton step --- which lands on `χ` itself --- fails
        // the strict `newton > lo && newton < hi` and would be replaced by a bisection of the whole
        // remaining interval, throwing the answer away and starting again. Measured: 19.4 solver
        // iterations per call with the two tests the other way round, 4.3 with them in this order.
        #[expect(
            clippy::float_cmp,
            reason = "the fixed point is exactly where the step stops moving `χ`"
        )]
        let settled = newton == chi;
        if settled {
            break;
        }
        let next = if newton > lo && newton < hi {
            newton
        } else {
            f64::midpoint(lo, hi)
        };
        // The second stop: the bracket has closed on two adjacent doubles and the midpoint of those
        // is one of them, so bisection has nothing left to give. Without this the loop spins its full
        // hundred iterations whenever Newton's step is half an ulp --- which a circular orbit over
        // one period does exactly, `F` being linear there and its two terms cancelling to the ulp of
        // 3.7e6.
        #[expect(
            clippy::float_cmp,
            reason = "a bisection that returns its own left endpoint has bottomed out"
        )]
        let bottomed = next == chi;
        if bottomed {
            break;
        }
        chi = next;
    }
    (chi, steps)
}

/// Propagate a two-body state `(r, v)` under gravitational parameter `mu` by time `dt`, returning
/// the new `(r, v)`. Units are the caller's, consistent (`mu` in length³/time²); `dt` may be negative.
///
/// The universal anomaly comes from `universal_anomaly`; the state is then reconstructed from the
/// Lagrange coefficients, whose identity `f·ġ − ḟ·g = 1` the tests check.
///
/// # Accuracy
///
/// The result cannot be better than `ε·|dt|·|v|`, one unit in the last place of the span expressed
/// as a displacement: `dt` is a single `f64`, so a span of `N` revolutions carries the epoch only to
/// a relative `ε`, and position moves at the orbital speed. That floor, not the solver, is what
/// limits a long propagation --- `holds_the_ulp_floor_over_many_revolutions` measures the solver
/// against exact periodicity and pins it to a small multiple of the floor. Callers needing better
/// over long spans must carry the epoch in something wider than one double, not ask this function
/// for more.
#[must_use]
pub fn propagate(r: Vec3, v: Vec3, mu: f64, dt: f64) -> (Vec3, Vec3) {
    let r0 = r.norm();
    let v0 = v.norm();
    let alpha = 2.0 / r0 - v0 * v0 / mu;
    let sqmu = libm::sqrt(mu);
    let (chi, _) = universal_anomaly(r, v, mu, dt);

    let z = alpha * chi * chi;
    let (c, s) = stumpff(z);

    let lf = 1.0 - chi * chi / r0 * c;
    let lg = dt - chi * chi * chi / sqmu * s;
    let r_new = comb(lf, r, lg, v);
    let rn = r_new.norm();
    let lfd = sqmu / (rn * r0) * (alpha * chi * chi * chi * s - chi);
    let lgd = 1.0 - chi * chi / rn * c;
    let v_new = comb(lfd, r, lgd, v);

    (r_new, v_new)
}

#[cfg(test)]
mod tests {
    // The references below are carried at the precision they were computed to, so that
    // regenerating one is a copy rather than a rounding decision.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]

    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// The Stumpff functions satisfy `cos√z = 1 − z·C(z)` and `sin√z/√z = 1 − z·S(z)` by
        /// definition, with the hyperbolic forms below zero. That is not an oracle — it is the
        /// identity the closed forms were derived from, so it holds for the series branch and the
        /// quartering recursion just as strictly, and it is the only check that crosses both
        /// thresholds (`|z| = 1` and `|z| = 1/32`) on every run.
        ///
        /// Worst measured over 20 000 draws on `|z| ≤ 1e5`: 3.331e-16 on the `C` identity and
        /// 5.219e-16 on the `S` identity — two ulps, across both thresholds and both branches. The
        /// quartering recursion doubles its relative error per fold and still lands there, which is
        /// the claim the module doc makes and nothing else in this file measures over the whole
        /// range rather than at chosen points.
        #[test]
        fn the_stumpff_functions_satisfy_their_defining_identities(z in -1e5_f64..1e5) {
            let (c, s) = stumpff(z);
            let (c0, c1) = if z > 0.0 {
                let x = libm::sqrt(z);
                (libm::cos(x), libm::sin(x) / x)
            } else if z < 0.0 {
                let x = libm::sqrt(-z);
                (libm::cosh(x), libm::sinh(x) / x)
            } else {
                (1.0, 1.0)
            };

            let dc = ((1.0 - z * c) - c0).abs() / c0.abs().max(1.0);
            let ds = ((1.0 - z * s) - c1).abs() / c1.abs().max(1.0);
            prop_assert!(dc < 1e-14, "C identity off by {dc:.3e} at z {z:.6}");
            prop_assert!(ds < 1e-14, "S identity off by {ds:.3e} at z {z:.6}");
        }

        /// Both functions are finite and positive for every argument the solver can hand them.
        /// `C` and `S` are sums of positive terms for `z < 0` and bounded oscillations above, so a
        /// negative or non-finite value means the branch selection, not the mathematics, went
        /// wrong — the failure mode a fixture on one side of a threshold cannot see.
        #[test]
        fn the_stumpff_functions_stay_finite_and_positive(z in -1e5_f64..1e5) {
            let (c, s) = stumpff(z);
            prop_assert!(c.is_finite() && s.is_finite(), "C {c} S {s} at z {z}");
            prop_assert!(c > 0.0 && s > 0.0, "C {c} S {s} at z {z} is not positive");
        }
    }

    const MU_EARTH_KM: f64 = 398_600.0; // km³/s²

    /// `(z, C(z), S(z))` at 60 decimal digits, spanning both closed-form branches, the series
    /// branch and the two thresholds exactly.
    const STUMPFF: [(f64, f64, f64); 19] = [
        (0.0, 0.5, 0.166_666_666_666_666_666_666_666_7),
        (
            1e-9,
            0.499_999_999_958_333_333_334_722_2,
            0.166_666_666_658_333_333_333_531_7,
        ),
        (
            -1e-9,
            0.500_000_000_041_666_666_668_055_6,
            0.166_666_666_675_000_000_000_198_4,
        ),
        (
            5e-7,
            0.499_999_979_166_667_013_888_886_7,
            0.166_666_662_500_000_049_603_174_4,
        ),
        (
            -5e-7,
            0.500_000_020_833_333_680_555_557_7,
            0.166_666_670_833_333_382_936_508_1,
        ),
        (
            1e-6,
            0.499_999_958_333_334_722_222_199_3,
            0.166_666_658_333_333_531_746_029_4,
        ),
        (
            -1e-6,
            0.500_000_041_666_668_055_555_578_5,
            0.166_666_675_000_000_198_412_700_8,
        ),
        (
            1.000_001e-6,
            0.499_999_958_333_293_055_558_309_3,
            0.166_666_658_333_325_198_413_092_6,
        ),
        (
            -1.000_001e-6,
            0.500_000_041_666_709_722_225_024_1,
            0.166_666_675_000_008_531_746_431_2,
        ),
        (
            1e-4,
            0.499_995_833_347_222_197_420_462_8,
            0.166_665_833_335_317_457_561_691_0,
        ),
        (
            -1e-4,
            0.500_004_166_680_555_580_357_370_1,
            0.166_667_500_001_984_129_739_901_3,
        ),
        (
            0.01,
            0.499_583_472_197_423_390_435_133_4,
            0.166_583_353_171_847_693_184_067_7,
        ),
        (
            -0.01,
            0.500_416_805_580_359_898_806_523_7,
            0.166_750_019_844_025_823_731_119_1,
        ),
        (
            1.0,
            0.459_697_694_131_860_282_599_063_4,
            0.158_529_015_192_103_493_347_497_7,
        ),
        (
            -1.0,
            0.543_080_634_815_243_778_477_905_6,
            0.175_201_193_643_801_456_882_381_9,
        ),
        (
            6.25,
            0.288_182_978_487_509_394_373_360_4,
            0.121_697_782_777_346_784_380_681_3,
        ),
        (
            -6.25,
            0.821_166_316_746_189_778_659_176_4,
            0.227_213_086_786_546_388_572_820_7,
        ),
        (
            39.0,
            0.000_018_693_504_459_147_547_324_240_80,
            0.025_797_778_901_917_277_349_332_58,
        ),
        (
            -39.0,
            6.582_437_607_415_163_741_313_936,
            1.032_490_440_167_838_877_812_653,
        ),
    ];

    /// Both Stumpff functions across both branches and the crossover, against `C` and `S` evaluated
    /// at 60 decimal digits from their definitions.
    ///
    /// Neither function was called by any test: `propagate` reaches them, but through a Newton
    /// iteration that corrects whatever they return, and twenty-one mutations of them survived. The
    /// series branch is the one that matters here. It is not an approximation for convenience — the
    /// closed forms are `(1 − cos√z)/z` and `(√z − sin√z)/√z³`, and as `z → 0` both subtract two
    /// nearly equal numbers, so the closed form loses a digit for every decade `z` falls. The series
    /// has no cancellation at all and is what makes the small-`z` answer correct.
    ///
    /// So the tolerance is not one number, but it is now a tight one on the whole reduced range. Up
    /// to `|z| = 1` the answer is the series unfolded by quartering, and four ulps covers it —
    /// where the old `|z| > 1e-6` cutoff onto the closed form left nine hundred thousand ulps of
    /// cancellation in `[1e−6, 1e−2]`, and this test allowed it. Past `|z| = 1` the closed form is
    /// back and so is its `1 − cos√z`, whose absolute error is one ulp of unity however small the
    /// difference is; `4ε/|z|` is that error carried through the division, and it is what lets the
    /// table hold `z = 39`, a hair from the first zero of `C` at `4π²`, to a relative accuracy that
    /// no method has there.
    #[test]
    fn the_stumpff_functions_match_their_definitions_across_both_branches() {
        for (z, want_c, want_s) in STUMPFF {
            for (name, got, want) in [("C", stumpff_c(z), want_c), ("S", stumpff_s(z), want_s)] {
                // Deliberately the literal `1.0` and not `CLOSED_FORM_FLOOR`: a tolerance written
                // in terms of the implementation's own threshold moves whenever that threshold
                // does, and then the test cannot see the move. This is the accuracy required, not
                // the accuracy the current branch structure happens to deliver.
                let cancellation = if z.abs() > 1.0 {
                    4.0 * f64::EPSILON / z.abs()
                } else {
                    0.0
                };
                let tol = cancellation + 4.0 * f64::EPSILON * want.abs();
                let d = (got - want).abs();
                assert!(
                    d <= tol,
                    "{name}({z:e}): got {got:.17e}, want {want:.17e}, off by {d:.3e} (allowed {tol:.3e})"
                );
            }
        }
    }

    /// The solver stays on the conic across the family and over spans of very different length.
    ///
    /// The initial guess for `χ` and the loop's exit test are self-correcting: Newton lands on the
    /// same root from anywhere in its basin, so no accuracy assertion at a single state can see
    /// either. What can see them is a guess that leaves the basin or an exit that fires before
    /// convergence, and both surface the same way. `g = dt − χ³S/√μ` mixes the true span with
    /// whatever `χ` the loop returned, so an unconverged `χ` moves the point off the orbit
    /// altogether — and then the two integrals of two-body motion stop being integrals.
    ///
    /// Nine conics from circular to a fast hyperbola, twelve spans from a ten-thousandth of a
    /// circular period to five hundred of them, both signs. The bounds are a little over twice what
    /// the hundred and eight states measure: 2.1e-13 of the periapsis potential in energy and
    /// 1.3e-12 of `|h|` in angular momentum. The round trip is the loosest of the three at four
    /// thousand ulps of the span-displacement, and it is here to catch a `χ` that never converged
    /// rather than to measure accuracy — the oracle table below does that.
    #[test]
    fn the_solver_stays_on_the_conic_across_the_family_and_over_long_spans() {
        // Periapsis states at a common radius, tilted out of plane so no component is zero, from
        // circular through the parabolic limit to a fast hyperbola. `e = 1` puts `α` at the rounding
        // of zero, which is the only way to reach the near-parabolic fallback guess.
        let rp = 7000.0;
        let tc = core::f64::consts::TAU * (rp * rp * rp / MU_EARTH_KM).sqrt();
        for &e in &[0.0, 0.3, 0.7, 0.95, 0.999, 1.0, 1.000_1, 1.2, 3.0] {
            let vp = (MU_EARTH_KM * (1.0 + e) / rp).sqrt();
            let r = Vec3::new(rp, 0.0, 0.0);
            let v = Vec3::new(0.0, vp * 0.6, vp * 0.8);
            let (e0, h0) = (v.dot(v) / 2.0 - MU_EARTH_KM / r.norm(), r.cross(v));
            // The potential at periapsis is the scale to measure energy against: at `e = 1` the
            // energy is zero and a relative bound would be a division by it.
            let scale = MU_EARTH_KM / rp;
            for &k in &[1e-4, 1e-2, 1.0, 5.0, 50.0, 500.0] {
                for dt in [k * tc, -k * tc] {
                    let (rf, vf) = propagate(r, v, MU_EARTH_KM, dt);
                    let de = (vf.dot(vf) / 2.0 - MU_EARTH_KM / rf.norm() - e0).abs();
                    let dh = (rf.cross(vf) - h0).norm();
                    assert!(
                        de < 5e-13 * scale,
                        "e {e} dt {dt:e}: energy off by {de:.3e}"
                    );
                    assert!(dh < 3e-12 * h0.norm(), "e {e} dt {dt:e}: h off by {dh:.3e}");
                    // Back again: the round trip has no truncation error of its own, so what is left
                    // is the two representation floors.
                    let (rb, _) = propagate(rf, vf, MU_EARTH_KM, -dt);
                    let back = (rb - r).norm();
                    let tol = 4096.0 * f64::EPSILON * (rf.norm() + dt.abs() * vf.norm());
                    assert!(
                        back < tol,
                        "e {e} dt {dt:e}: round trip off by {back:.3e} km"
                    );
                }
            }
        }
    }

    /// The solver reaches its root by Newton and not by bisection.
    ///
    /// Everything the safeguarded loop does other than keep its bracket is invisible in the answer.
    /// Corrupt the initial guess, the Newton step, the containment test, the doubling that walks the
    /// bracket out --- the loop still ends on the same root, because `lo` and `hi` are moved by the
    /// sign of `F` and bisection alone converges. Two dozen mutations of this function survived every
    /// accuracy assertion in this file for exactly that reason. What they change is how many
    /// evaluations it costs, and that is what this asserts.
    ///
    /// The budget is measured, not chosen, and it is two-sided. Over the nine conics and twelve spans
    /// below the worst call takes 19 steps and the mean is 8.28, the hard end being the near-parabolic
    /// arcs over hundreds of revolutions; bisection alone over the same brackets would need fifty to
    /// sixty. The floor matters as much: a one-sided budget passes when the counter stops counting,
    /// and `steps += 1` mutated to `steps *= 1` in either loop did exactly that until this asserted
    /// the low end. Two steps is the structural minimum — the bracket is tested at least once, and at
    /// least one Newton step has to run to reach the fixed point.
    ///
    /// The margins are small on purpose. Every transcendental here goes through `libm`, a
    /// deterministic pure-Rust implementation, and every other operation is IEEE-754, so the counts
    /// are reproducible bit for bit and there is nothing to pad against. Measured against the mutants
    /// that survive the accuracy assertions, `< 8.5` is what separates them: corrupting the hyperbolic
    /// guess costs 10.15 steps per call, its logarithm 8.72 or 9.32, its argument 8.81, and the
    /// elliptic guess 8.86. A change that legitimately moves these numbers should move them here too.
    #[test]
    fn the_solver_reaches_its_root_by_newton_and_not_by_bisection() {
        let rp = 7000.0;
        let tc = core::f64::consts::TAU * (rp * rp * rp / MU_EARTH_KM).sqrt();
        let (mut worst, mut least, mut total, mut calls) = (0, u32::MAX, 0u32, 0u32);
        for &e in &[0.0, 0.3, 0.7, 0.95, 0.999, 1.0, 1.000_1, 1.2, 3.0] {
            let vp = (MU_EARTH_KM * (1.0 + e) / rp).sqrt();
            let r = Vec3::new(rp, 0.0, 0.0);
            let v = Vec3::new(0.0, vp * 0.6, vp * 0.8);
            for &k in &[1e-4, 1e-2, 1.0, 5.0, 50.0, 500.0] {
                for dt in [k * tc, -k * tc] {
                    let (_, steps) = universal_anomaly(r, v, MU_EARTH_KM, dt);
                    assert!(
                        steps >= 2,
                        "e {e}, dt {dt:e}: the solve reported {steps} steps"
                    );
                    worst = worst.max(steps);
                    least = least.min(steps);
                    total += steps;
                    calls += 1;
                }
            }
        }
        let mean = f64::from(total) / f64::from(calls);
        println!("STEPS least {least} worst {worst} mean {mean:.2}");
        assert!(worst <= 19, "worst call took {worst} steps");
        assert!(mean < 8.5, "mean {mean:.2} steps per call");
    }

    /// The propagated state against the exact two-body solution, over eight decades of span and on
    /// both conics.
    ///
    /// The oracle is classical: orbital elements, Kepler's equation solved at 60 digits, and the
    /// Lagrange coefficients in terms of the eccentric (or hyperbolic) anomaly difference — a
    /// different formulation from the universal-variable one under test, so the two agreeing is
    /// evidence and not a tautology.
    ///
    /// Two floors bound the answer and the tolerance is their sum. One is the position itself: the
    /// state comes back in doubles, so `ε·|r|` is unavoidable, and it is what dominates for a short
    /// span. The other is the span: `dt` is one double, so a long propagation carries its epoch only
    /// to `ε·|dt|`, and the state moves at `|v|` while that error sits there. The constant in front
    /// is measured, not chosen: the worst of the forty-six comparisons reaches 0.31 of the bound.
    #[test]
    fn the_propagated_state_matches_the_exact_two_body_solution() {
        for (name, r, v, table) in [
            (
                "ellipse",
                Vec3::new(7000.0, 1200.0, -800.0),
                Vec3::new(-1.3, 6.9, 2.4),
                &ELLIPSE[..],
            ),
            (
                "hyperbola",
                Vec3::new(8000.0, 1500.0, -900.0),
                Vec3::new(1.1, 9.5, 3.2),
                &HYPERBOLA[..],
            ),
            (
                "near-parabolic ellipse",
                NEAR_PARABOLIC_ELLIPSE_STATE.0,
                NEAR_PARABOLIC_ELLIPSE_STATE.1,
                &NEAR_PARABOLIC_ELLIPSE[..],
            ),
            (
                "near-parabolic hyperbola",
                NEAR_PARABOLIC_HYPERBOLA_STATE.0,
                NEAR_PARABOLIC_HYPERBOLA_STATE.1,
                &NEAR_PARABOLIC_HYPERBOLA[..],
            ),
        ] {
            for &(dt, want) in table {
                let (rf, vf) = propagate(r, v, MU_EARTH_KM, dt);
                let want_r = Vec3::new(want[0], want[1], want[2]);
                let want_v = Vec3::new(want[3], want[4], want[5]);
                let acc = MU_EARTH_KM / (r.norm() * r.norm());
                let tol_r = 8.0 * f64::EPSILON * (want_r.norm() + dt.abs() * v.norm());
                let tol_v = 8.0 * f64::EPSILON * (want_v.norm() + dt.abs() * acc);
                let (dr, dv) = ((rf - want_r).norm(), (vf - want_v).norm());
                println!(
                    "RATIO {name} dt {dt} r {:.3} v {:.3}",
                    dr / tol_r,
                    dv / tol_v
                );
                assert!(
                    dr <= tol_r,
                    "{name} dt {dt}: position off by {dr:.3e} km (allowed {tol_r:.3e})"
                );
                assert!(
                    dv <= tol_v,
                    "{name} dt {dt}: velocity off by {dv:.3e} km/s (allowed {tol_v:.3e})"
                );
            }
        }
    }

    /// Two near-parabolic arcs, `e = 0.999` and `e = 1.0001`, both from a periapsis at 7000 km
    /// rotated 3-1-3 by `(0.7, 0.5, 1.1)` rad so no component is zero. This is the regime the two
    /// tables above miss entirely, and it is where the universal formulation is hardest: `α → 0`
    /// drives `z = αχ²` into the band where the Stumpff closed forms cancel, and the elliptic
    /// initial guess `√μ·α·dt` --- which is `√a·M` where the root is `√a·ΔE ≫ √a·M` --- lands far
    /// outside the tangent's reach. Both defects lived here unseen: at `e = 0.999, dt = 1e7 s` the
    /// solver returned a position 6.6e20 km from this table.
    ///
    /// Generated by `derivations/kepler_two_body.py`, which propagates the double-rounded epoch
    /// state --- the only state a caller can actually pass --- through the classical elements at 60
    /// digits, and first reproduces the committed `ELLIPSE` row to 4e-22 to show the formulation is
    /// the right one.
    const NEAR_PARABOLIC_ELLIPSE_STATE: (Vec3, Vec3) = (
        Vec3 {
            x: -1_098.428_673_272_579_089_825_740,
            y: 6_566.528_239_227_161_066_082_772,
            z: 2_161.980_881_775_988_109_438_913,
        },
        Vec3 {
            x: -9.499_769_365_557_943_601_174_884,
            y: -2.877_149_163_557_400_690_649_501,
            z: 3.912_181_759_634_796_751_612_384,
        },
    );

    const NEAR_PARABOLIC_ELLIPSE: [(f64, [f64; 6]); 6] = [
        (
            1e-3,
            [
                -1_098.438_173_041_306_404_429_153,
                6_566.525_362_074_182_030_877_595,
                2_161.984_793_956_491_525_432_058,
                -9.499_768_089_069_423_280_050_507,
                -2.877_156_794_512_452_405_203_488,
                3.912_179_247_196_431_815_558_505,
            ],
        ),
        (
            1.0,
            [
                -1_107.927_802_556_939_294_815_911,
                6_563.647_275_143_973_898_144_941,
                2_165.891_806_560_339_880_635_820,
                -9.498_487_363_714_154_047_686_049,
                -2.884_778_442_608_729_541_542_649,
                3.909_667_052_318_733_773_319_467,
            ],
        ),
        (
            100.0,
            [
                -2_040.218_387_474_563_483_989_079,
                6_241.359_280_368_281_236_576_757,
                2_539.931_817_023_650_058_241_989,
                -9.318_429_305_421_454_560_273_932,
                -3.617_839_033_841_698_236_753_653,
                3.640_349_480_889_821_607_876_084,
            ],
        ),
        (
            5828.5,
            [
                -21_237.163_622_391_291_870_730_46,
                -25_888.415_638_234_164_013_630_25,
                3_924.545_091_745_940_492_848_511,
                -1.285_974_817_813_406_793_415_772,
                -4.653_761_645_611_996_406_268_557,
                -0.527_104_333_208_584_204_910_131_4,
            ],
        ),
        (
            291_426.0,
            [
                -25_102.774_998_415_731_167_517_99,
                -509_857.946_233_217_771_042_134_1,
                -114_121.393_208_481_327_589_791_5,
                0.071_183_272_767_943_933_186_252_37,
                -1.165_109_820_959_555_636_083_921,
                -0.323_371_932_739_541_576_343_908_3,
            ],
        ),
        (
            1e7,
            [
                539_664.443_847_897_769_032_417_3,
                -4_914_917.666_792_550_075_557_752,
                -1_480_665.771_725_738_635_448_211,
                0.045_609_572_893_810_882_229_216_52,
                -0.293_935_334_179_153_264_688_047_4,
                -0.095_043_256_790_086_884_667_745_86,
            ],
        ),
    ];

    const NEAR_PARABOLIC_HYPERBOLA_STATE: (Vec3, Vec3) = (
        Vec3 {
            x: -1_098.428_673_272_579_089_825_740,
            y: 6_566.528_239_227_161_066_082_772,
            z: 2_161.980_881_775_988_109_438_913,
        },
        Vec3 {
            x: -9.502_382_749_534_582_728_756_504,
            y: -2.877_940_666_512_167_844_359_738,
            z: 3.913_257_999_776_099_449_462_663,
        },
    );

    const NEAR_PARABOLIC_HYPERBOLA: [(f64, [f64; 6]); 6] = [
        (
            1e-3,
            [
                -1_098.438_175_654_690_381_067_775,
                6_566.525_361_282_679_076_110_594,
                2_161.984_795_032_731_666_734_547,
                -9.502_381_473_046_060_889_127_326,
                -2.877_948_297_467_219_099_006_400,
                3.913_255_487_337_733_888_061_526,
            ],
        ),
        (
            1.0,
            [
                -1_107.930_415_940_409_970_724_288,
                6_563.646_483_641_173_650_735_160,
                2_165.892_882_800_273_135_727_654,
                -9.501_100_746_173_110_189_305_340,
                -2.885_569_945_098_718_948_004_316,
                3.910_743_291_836_296_224_070_492,
            ],
        ),
        (
            100.0,
            [
                -2_040.479_253_145_184_396_152_601,
                6_241.280_399_495_615_177_769_665,
                2_540.039_277_644_890_806_151_290,
                -9.321_028_965_920_365_539_353_053,
                -3.618_621_367_174_963_518_503_119,
                3.641_421_311_736_534_103_454_170,
            ],
        ),
        (
            5828.5,
            [
                -21_268.644_066_335_099_812_767_37,
                -25_895.242_024_148_780_897_149_02,
                3_938.180_354_561_661_678_984_124,
                -1.293_103_817_182_546_487_961_915,
                -4.656_813_721_280_045_532_386_152,
                -0.524_389_752_300_236_083_803_355_1,
            ],
        ),
        (
            291_426.0,
            [
                -27_115.586_641_247_477_676_572_14,
                -514_340.906_402_333_218_694_898_1,
                -114_252.298_645_248_169_662_875_3,
                0.064_937_257_883_420_055_861_029_11,
                -1.185_996_436_635_390_512_630_623,
                -0.325_506_660_756_437_960_184_070_3,
            ],
        ),
        (
            1e7,
            [
                526_770.563_878_702_140_814_167_6,
                -5_422_254.179_289_454_047_186_949,
                -1_600_106.481_982_643_425_133_447,
                0.047_273_057_351_238_774_417_831_39,
                -0.362_145_625_930_673_580_355_544_5,
                -0.112_755_712_867_788_667_638_642_4,
            ],
        ),
    ];

    /// `a = 7057.006 km`, `e = 0.0532`, period 5901 s: a span of `1e-3 s` and one of `50 000 s` are
    /// eight and a half decades apart and land on opposite sides of the Stumpff crossover.
    const ELLIPSE: [(f64, [f64; 6]); 7] = [
        (
            1e-3,
            [
                6_999.998_699_996_178_542_779_229,
                1_200.006_899_999_344_891_752_203,
                -799.997_599_999_563_262_441_660_8,
                -1.300_007_642_914_409_864_244_779,
                6.899_998_689_782_213_000_870_119,
                2.400_000_873_474_703_356_454_070,
            ],
        ),
        (
            0.5,
            [
                6_999.349_044_639_655_448_131_406,
                1_203.449_836_061_912_428_929_756,
                -798.799_890_867_175_408_936_444_8,
                -1.303_821_433_398_705_901_009_674,
                6.899_343_924_947_589_643_173_470,
                2.400_436_427_952_730_283_680_852,
            ],
        ),
        (
            10.0,
            [
                6_986.617_889_254_209_879_192_318,
                1_268.933_198_087_324_952_945_802,
                -775.956_740_230_227_072_418_640_9,
                -1.376_418_310_671_478_505_243_972,
                6.886_510_450_948_617_942_820_685,
                2.408_610_479_674_902_282_850_029,
            ],
        ),
        (
            300.0,
            [
                6_269.772_172_689_816_261_998_951,
                3_175.978_207_698_139_362_578_484,
                -52.400_114_265_378_726_726_863_44,
                -3.546_028_261_742_284_818_817_987,
                6.156_187_251_101_626_490_995_061,
                2.543_284_165_182_447_174_076_657,
            ],
        ),
        (
            5000.0,
            [
                5_248.764_651_863_748_007_998_531,
                -4_658.001_912_521_062_078_266_628,
                -2_356.593_125_090_000_270_865_653,
                4.810_250_034_342_690_153_906_033,
                5.230_534_794_616_643_452_657_361,
                0.842_902_689_022_857_943_996_880_1,
            ],
        ),
        (
            50000.0,
            [
                -6_755.973_136_479_606_751_164_495,
                -1_469.718_050_812_532_646_028_601,
                673.634_534_007_685_171_746_654_6,
                0.920_840_335_584_485_985_626_389_8,
                -7.179_813_086_431_391_417_591_499,
                -2.424_567_048_958_451_670_803_638,
            ],
        ),
        (
            -3000.0,
            [
                -6_626.269_318_801_166_858_173_316,
                -2_169.027_941_502_567_149_222_793,
                430.741_715_199_545_624_657_012_0,
                1.700_897_421_246_237_599_759_840,
                -6.967_828_161_871_409_462_130_803,
                -2.488_979_345_558_800_608_929_078,
            ],
        ),
    ];

    /// `a = −91 630.1 km`, so `α < 0` and the hyperbolic initial guess is the one exercised — the
    /// branch that carries a logarithm and a sign chosen from `dt`, which is why one span is
    /// backwards.
    const HYPERBOLA: [(f64, [f64; 6]); 4] = [
        (
            1.0,
            [
                8_001.097_097_373_498_948_337_082,
                1_509.499_454_633_739_812_503_521,
                -896.799_673_856_423_259_333_402_6,
                1.094_195_488_149_384_341_842_812,
                9.498_908_283_161_953_732_582_195,
                3.200_651_802_050_868_637_238_535,
            ],
        ),
        (
            100.0,
            [
                8_081.749_567_588_654_570_974_811,
                2_443.631_243_096_794_064_973_539,
                -577.205_137_503_225_055_779_879_7,
                0.543_204_208_860_345_807_215_206_6,
                9.363_986_119_992_161_842_804_183,
                3.251_333_083_285_890_803_198_532,
            ],
        ),
        (
            2000.0,
            [
                4_334.833_056_033_052_683_774_093,
                16_480.081_152_844_756_776_245_16,
                5_115.471_910_640_226_679_427_269,
                -2.990_720_862_500_886_831_333_629,
                5.781_693_817_619_632_686_158_767,
                2.604_725_785_136_420_741_703_023,
            ],
        ),
        (
            -1500.0,
            [
                777.564_989_397_711_458_107_174_4,
                -10_520.576_317_682_008_692_339_58,
                -3_902.119_931_346_658_717_063_935,
                6.660_479_301_996_484_001_725_731,
                5.501_686_996_373_380_597_852_132,
                0.771_653_780_119_907_669_940_781_0,
            ],
        ),
    ];

    /// Curtis, *Orbital Mechanics for Engineering Students*, Example 3.7.
    #[test]
    fn matches_curtis_example_3_7() {
        let r = Vec3::new(7000.0, -12124.0, 0.0);
        let v = Vec3::new(2.6679, 4.6210, 0.0);
        let (rf, vf) = propagate(r, v, MU_EARTH_KM, 3600.0);
        let r_exp = Vec3::new(-3297.8, 7413.4, 0.0);
        let v_exp = Vec3::new(-8.2977, -0.96309, 0.0);
        let dr = (rf.x - r_exp.x).hypot(rf.y - r_exp.y);
        let dv = (vf.x - v_exp.x).hypot(vf.y - v_exp.y);
        assert!(dr < 1.0, "position off by {dr:.3} km: got {rf:?}");
        assert!(dv < 1e-3, "velocity off by {dv:.5} km/s: got {vf:?}");
    }

    /// Propagating forward then back by the same `dt` returns the original state.
    #[test]
    fn propagation_is_reversible() {
        let r = Vec3::new(6500.0, 900.0, 1200.0);
        let v = Vec3::new(-1.2, 7.0, 2.1);
        let (rf, vf) = propagate(r, v, MU_EARTH_KM, 5400.0);
        let (r0, v0) = propagate(rf, vf, MU_EARTH_KM, -5400.0);
        let dr = ((r0.x - r.x).powi(2) + (r0.y - r.y).powi(2) + (r0.z - r.z).powi(2)).sqrt();
        let dv = ((v0.x - v.x).powi(2) + (v0.y - v.y).powi(2) + (v0.z - v.z).powi(2)).sqrt();
        assert!(dr < 1e-6, "position drifted {dr:.3e} km on round trip");
        assert!(dv < 1e-9, "velocity drifted {dv:.3e} km/s on round trip");
    }

    /// The solver sits at the floor its own arithmetic sets, over a thousand revolutions.
    ///
    /// A round trip is the wrong oracle for this: forward and back retrace the same universal
    /// anomaly, so at whole revolution counts the errors cancel exactly and the metric reports zero.
    /// Exact periodicity is the right one --- two-body motion repeats, so propagating by `n + f`
    /// periods and by `f` periods must give the same state, and only the first call has a large
    /// anomaly. What remains is `ε·|dt|·|v|`: one unit in the last place of the span, carried at the
    /// orbital speed. The bound here is a small multiple of that, so a real regression in the solver
    /// fails it while the representation of `dt` does not.
    #[test]
    fn holds_the_ulp_floor_over_many_revolutions() {
        let a = 7500.0_f64;
        let ecc = 0.1;
        let rp = a * (1.0 - ecc);
        let vp = (MU_EARTH_KM * (1.0 + ecc) / rp).sqrt();
        let r = Vec3::new(rp, 0.0, 0.0);
        let v = Vec3::new(0.0, vp * 0.6, vp * 0.8);
        let period = core::f64::consts::TAU * (a * a * a / MU_EARTH_KM).sqrt();
        let (truth, _) = propagate(r, v, MU_EARTH_KM, 0.25 * period);

        for &revs in &[10.25_f64, 100.25, 1000.25] {
            let dt = revs * period;
            let (far, _) = propagate(r, v, MU_EARTH_KM, dt);
            let dr =
                ((far.x - truth.x).powi(2) + (far.y - truth.y).powi(2) + (far.z - truth.z).powi(2))
                    .sqrt();
            let floor = f64::EPSILON * dt * v.norm();
            assert!(
                dr < 5.0 * floor,
                "over {revs} revolutions the position is off by {dr:.3e} km, \
                 {:.1} times the {floor:.3e} km one ulp of the span is worth",
                dr / floor
            );
        }
    }

    /// Energy and angular momentum are two-body invariants across propagation.
    #[test]
    fn conserves_energy_and_angular_momentum() {
        let r = Vec3::new(8000.0, 0.0, 600.0);
        let v = Vec3::new(0.0, 6.5, 3.0);
        let energy = |r: Vec3, v: Vec3| v.dot(v) / 2.0 - MU_EARTH_KM / r.norm();
        let (rf, vf) = propagate(r, v, MU_EARTH_KM, 4000.0);
        assert!(
            (energy(rf, vf) - energy(r, v)).abs() < 1e-9,
            "energy drifted"
        );
        let h0 = r.cross(v).norm();
        let hf = rf.cross(vf).norm();
        assert!((hf - h0).abs() < 1e-6, "angular momentum drifted");
    }

    /// After one full period an elliptic orbit returns to its start.
    #[test]
    fn one_period_closes_the_orbit() {
        let r = Vec3::new(7500.0, 0.0, 0.0);
        let v = Vec3::new(0.0, 8.0, 0.0);
        let energy = v.dot(v) / 2.0 - MU_EARTH_KM / r.norm();
        let a = -MU_EARTH_KM / (2.0 * energy);
        let period = core::f64::consts::TAU * (a * a * a / MU_EARTH_KM).sqrt();
        let (rf, vf) = propagate(r, v, MU_EARTH_KM, period);
        let dr = ((rf.x - r.x).powi(2) + (rf.y - r.y).powi(2) + (rf.z - r.z).powi(2)).sqrt();
        assert!(
            dr < 1e-3,
            "orbit did not close after one period: {dr:.3e} km"
        );
        let _ = vf;
    }

    /// A hyperbolic state (v > escape) propagates through the branchless path.
    #[test]
    fn hyperbolic_state_propagates() {
        let r = Vec3::new(7000.0, 0.0, 0.0);
        let v_esc = (2.0 * MU_EARTH_KM / 7000.0).sqrt();
        let v = Vec3::new(0.5, v_esc * 1.2, 0.0); // clearly hyperbolic
        let (rf, vf) = propagate(r, v, MU_EARTH_KM, 1200.0);
        // Reversibility on the hyperbola is the check that the branch worked.
        let (r0, _) = propagate(rf, vf, MU_EARTH_KM, -1200.0);
        let dr = ((r0.x - r.x).powi(2) + (r0.y - r.y).powi(2) + (r0.z - r.z).powi(2)).sqrt();
        assert!(dr < 1e-5, "hyperbolic round trip drifted {dr:.3e} km");
    }

    /// `(r₀, v₀, dt, χ, r, scale)`.
    type UniversalRoot = ([f64; 3], [f64; 3], f64, f64, f64, f64);

    /// `(r₀, v₀, dt, χ, r, scale)`: the root of the universal Kepler equation at 100 digits
    /// (`derivations/kepler_universal.gp`), the radius `r(χ) = dF/dχ` there, and the floor of an
    /// `f64` evaluation of `F` in units of `ε`: the magnitudes of its four terms, plus `|∂F/∂α|` times
    /// the scale of `α = 2/r₀ − v₀²/μ`, which is formed from rounded operands. Dyadic states, so the reference is the root for the doubles
    /// passed: an ellipse, a hyperbola, `e = 0.999`, `e = 1.0001`, the double rounding of `e = 1`, and
    /// `e = 3`, each from `2⁻¹⁰ s` to `1e7 s` and backwards.
    #[rustfmt::skip]
    const UNIVERSAL_ROOTS: [UniversalRoot; 48] = [
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], 0.0009765625000000000000000000, 8.626671412751596656922688e-5, 7147.027026148617948136779, 1.233101131232619962950472),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], 1.000000000000000000000000, 0.08833918897778262947882931, 7146.691451167081170068834, 1262.725542126772325414406),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], 100.0000000000000000000000, 8.854922225656151117477969, 7112.514403961547803851848, 126916.5924906087385327499),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], 5000.000000000000000000000, 454.7277444256442693413818, 7329.510984438192093977364, 16734415.01035178248770720),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], 300000.0000000000000000000, 27127.57854523387745640009, 7171.068035549069479633387, 932768374.2225416921492787),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], 10000000.00000000000000000, 904012.1825428598419541423, 7149.321107589497311912519, 31069939775.81995956226850),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], -3000.000000000000000000000, -263.6241894040728851313824, 6816.672732546078815757148, 9470009.710704233539676877),
        ([7000.000000000000000000000, 1200.000000000000000000000, -800.0000000000000000000000], [-1.250000000000000000000000, 6.875000000000000000000000, 2.375000000000000000000000], -100.0000000000000000000000, -8.813445613093189782700152, 7179.562306930014267938005, 126614.7199162430453747993),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], 0.0009765625000000000000000000, 7.528992278188815270181306e-5, 8189.019458843442889429936, 1.233101102942838071496182),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], 1.000000000000000000000000, 0.07708520212841281927933300, 8191.501901254413275621009, 1262.695842219392866203214),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], 100.0000000000000000000000, 7.586791315996084371323109, 8464.863781727278702187205, 126570.0990689828606325571),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], 5000.000000000000000000000, 181.7361365963532723606230, 33146.70701497319023071497, 12168610.37554103566947375),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], 300000.0000000000000000000, 836.1039741218409163955085, 824809.7436850700176290413, 4283446642.439873453893693),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], 10000000.00000000000000000, 1765.493005468482416142056, 22182322.46903531161361056, 473481113423.7279687578777),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], -3000.000000000000000000000, -173.5089763656270440815267, 19277.03238223892159085851, 8526421.330928740888560248),
        ([8000.000000000000000000000, 1500.000000000000000000000, -900.0000000000000000000000], [1.125000000000000000000000, 9.500000000000000000000000, 3.250000000000000000000000], -100.0000000000000000000000, -7.819199091550129143978511, 7970.804355156854857227553, 128562.0363475210698922607),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], 0.0009765625000000000000000000, 8.807865021016564988987200e-5, 7000.000000003875045387605, 1.233101102943002057643069),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], 1.000000000000000000000000, 0.09019252036392695704157874, 7000.004063278019558648715, 1262.696018416420941619176),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], 100.0000000000000000000000, 9.001903004294252626307050, 7040.476572678168213982725, 126756.0205658956119710992),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], 5000.000000000000000000000, 214.9183734721436795727746, 30059.17471888388886543173, 15107742.48444641159189793),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], 300000.0000000000000000000, 1033.063241759541553703635, 533337.8160517703462763721, 6663478070.364351798780974),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], 10000000.00000000000000000, 3452.039927268549783130432, 5161397.597024077037782736, 2186773586137.464009739937),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], -3000.000000000000000000000, -164.5744009803299151358164, 20524.46272950296791719815, 7332480.318322608504153976),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.401434035971760749816895, 8.535245381295680999755859], -100.0000000000000000000000, -9.001903004294252626307050, 7040.476572678168213982725, 126756.0205658956119710992),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], 0.0009765625000000000000000000, 8.807865021016563199390282e-5, 7000.000000003879312203138, 1.233101102943002182914853),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], 1.000000000000000000000000, 0.09019252034471132765263941, 7000.004067752096891460412, 1262.696018550930175390199),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], 100.0000000000000000000000, 9.001883998300927194097651, 7040.521013356751774558714, 126756.1519140402739988922),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], 5000.000000000000000000000, 214.8379547368946448175282, 30081.24935845140830761645, 15104209.26713683978487178),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], 300000.0000000000000000000, 1029.845525921786503156877, 538013.8805234794033872318, 6628421202.784419235696783),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], 10000000.00000000000000000, 3345.265647850889794159182, 5677910.480166155531103163, 2048232452544.905408084493),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], -3000.000000000000000000000, -164.5269263101797210466115, 20536.34439474079940333954, 7331436.501214389502518161),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403195068240165710449219, 8.537593424320220947265625], -100.0000000000000000000000, -9.001883998300927194097651, 7040.521013356751774558714, 126756.1519140402739988922),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], 0.0009765625000000000000000000, 8.807865021016563362080641e-5, 7000.000000003878924311460, 1.233101102943002171526528),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], 1.000000000000000000000000, 0.09019252034645820015720079, 7000.004067345363262309274, 1262.696018538702083489492),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], 100.0000000000000000000000, 9.001885726106706263039238, 7040.516973313332411183534, 126756.1399733741536453692),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], 5000.000000000000000000000, 214.8452608316592636776927, 30079.24305125625578846347, 15104530.24387238633501524),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], 300000.0000000000000000000, 1030.136111617363323416661, 537590.2042987343749614708, 6631583360.514967163893709),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], 10000000.00000000000000000, 3354.289764990291075149734, 5632629.921070325919653500, 2059798303487.419229110792),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], -3000.000000000000000000000, -164.5312399300522567195211, 20535.26445663257270373352, 7331531.330729956313750648),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 6.403034994378685951232910, 8.537379993125796318054199], -100.0000000000000000000000, -9.001885726106706263039238, 7040.516973313332411183534, 126756.1399733741536453692),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], 0.0009765625000000000000000000, 8.807865021013309548579858e-5, 7000.000000011636772933510, 1.233101102943229938471583),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], 1.000000000000000000000000, 0.09019248540897687871118018, 7000.012202028998910348193, 1262.696263100787128080635),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], 100.0000000000000000000000, 8.967682075521427172184780, 7120.860133511548072011848, 126992.8009657241076553821),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], 5000.000000000000000000000, 146.5067044091679186446172, 59408.54004664683565333425, 12502801.64335099657182777),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], 300000.0000000000000000000, 379.8211348270083891745018, 3220505.116889182301636589, 1742780445.633309712313916),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], 10000000.00000000000000000, 586.8772035572617363399969, 106748470.5458341053224925, 90819735138.46058838875070),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], -3000.000000000000000000000, -119.8343090151786896633712, 36989.81455759134796835694, 6540791.818318731343889611),
        ([7000.000000000000000000000, 0.0, 0.0], [0.0, 9.055258929729461669921875, 12.07367857266217470169067], -100.0000000000000000000000, -8.967682075521427172184780, 7120.860133511548072011848, 126992.8009657241076553821),
    ];

    /// (G2) The solver's `χ` is the root of the universal Kepler equation, solved independently at
    /// 100 digits.
    ///
    /// The bound is the root's own conditioning: an `f64` evaluation of `F` is uncertain by a few ulps
    /// of `scale` — its terms, and its dependence on an `α` that cannot be formed exactly — and
    /// `dF/dχ = r` turns that into `χ`, so `|χ − χ*| ≤ 8ε·scale/r`. The near-parabolic and `e = 1` rows are where the solver once wandered
    /// six decades of `z` (see `universal_anomaly`), and the long hyperbolic spans are where the terms
    /// grow as `e^{√−α χ}`.
    #[test]
    fn the_universal_anomaly_is_the_root_of_kepler_s_equation() {
        let mut worst = 0.0_f64;
        for (r, v, dt, chi, radius, scale) in UNIVERSAL_ROOTS {
            let (got, _) = universal_anomaly(
                Vec3::new(r[0], r[1], r[2]),
                Vec3::new(v[0], v[1], v[2]),
                MU_EARTH_KM,
                dt,
            );
            let tol = 8.0 * f64::EPSILON * scale / radius;
            let d = (got - chi).abs();
            assert!(
                d <= tol,
                "r {r:?} v {v:?} dt {dt}: χ off by {d:.3e} (allowed {tol:.3e})"
            );
            worst = worst.max(d / tol);
        }
        println!("universal anomaly vs 100 digits: worst {worst:.3} of the conditioning bound");
    }

    /// (G4) The Lagrange coefficients `propagate` forms from the solver's `χ` keep the Wronskian
    /// `fġ − ḟg = 1` that `derivations/kepler_universal.mac` proves on every conic — the statement that
    /// the map `(r₀, v₀) ↦ (r, v)` preserves angular momentum.
    ///
    /// The coefficients are formed exactly as `propagate` forms them, with `r` the norm of the new
    /// position. The identity holds at the root, where `dt` equals the Kepler sum, and `g = dt −
    /// χ³S/√μ` carries whatever `F(χ)` the solver leaves, `fġ − ḟg − 1 = ḟ·F(χ)/√μ`. So the bound is
    /// 16 ulps of `|fġ| + |ḟ|(|g| + scale/√μ)`, with the row's `scale` the floor of `F` itself: over many
    /// revolutions `g` is a difference of two numbers the size of the span, the same `ε·|dt|` floor
    /// the module's `# Accuracy` section puts on the state.
    #[test]
    fn the_lagrange_coefficients_keep_their_wronskian() {
        let mut worst = 0.0_f64;
        for (r, v, dt, _, _, scale) in UNIVERSAL_ROOTS {
            let (r0v, v0v) = (Vec3::new(r[0], r[1], r[2]), Vec3::new(v[0], v[1], v[2]));
            let r0 = r0v.norm();
            let alpha = 2.0 / r0 - v0v.dot(v0v) / MU_EARTH_KM;
            let sqmu = MU_EARTH_KM.sqrt();
            let (chi, _) = universal_anomaly(r0v, v0v, MU_EARTH_KM, dt);
            let (c, s) = stumpff(alpha * chi * chi);
            let f = 1.0 - chi * chi / r0 * c;
            let g = dt - chi * chi * chi / sqmu * s;
            let rn = comb(f, r0v, g, v0v).norm();
            let fd = sqmu / (rn * r0) * (alpha * chi * chi * chi * s - chi);
            let gd = 1.0 - chi * chi / rn * c;
            let tol = 16.0 * f64::EPSILON * ((f * gd).abs() + fd.abs() * (g.abs() + scale / sqmu));
            let d = (f * gd - fd * g - 1.0).abs();
            assert!(
                d <= tol,
                "r {r:?} v {v:?} dt {dt}: fġ − ḟg − 1 = {d:.3e} (allowed {tol:.3e})"
            );
            worst = worst.max(d / tol);
        }
        println!("Lagrange Wronskian: worst {worst:.3} of the bound");
    }
}
