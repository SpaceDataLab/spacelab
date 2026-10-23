//! Initial orbit determination from angles: the bootstrap that produces the state a filter starts
//! from.
//!
//! A sequential filter needs an a-priori state and a filter cannot produce one — that is what IOD is
//! for. Three optical observations of an object whose orbit is unknown carry no range at all, only
//! two angles each; six numbers for six unknowns, exactly determined and badly conditioned. Every
//! method here is a different way of turning that into a state vector, and they are not
//! interchangeable.
//!
//! # Where this lives, and why not in `spacelab-estimation`
//!
//! IOD is orbit determination, and Orekit files it under `estimation.iod`. It is here instead
//! because its inputs are pure geometry — an epoch, a unit line of sight, and where the observer was
//! — and its tools are [`lambert`](fn@crate::lambert) and [`kepler::propagate`](fn@crate::lambert),
//! both of which live in this crate. Putting it in `spacelab-estimation` would add a dependency edge
//! from a deliberately lean `no_std` filter crate onto the whole of this one, for one function.
//! Converting a topocentric azimuth/elevation into the [`Sighting`] this module consumes stays in
//! `spacelab-estimation::measurement`, which is where the observation models belong.
//!
//! # The methods, and which to reach for
//!
//! Every function returns the state at the **middle** epoch, so the three are directly comparable.
//!
//! * [`gauss`] — the classical 1809 method. One 8th-degree polynomial and no iteration, so it is
//!   fast and it always answers, but it truncates the `f` and `g` series at second order in the
//!   arc length. That truncation is the whole story of its accuracy, and it is measured below.
//! * [`laplace`] — the other classical one. Where Gauss *interpolates* the three lines of sight,
//!   Laplace *differentiates* them: it fits a quadratic and reads `L̇` and `L̈` off it. Same
//!   8th-degree polynomial at the end, no transfer-sense flag, and — measured, against the
//!   literature's expectation — worse than Gauss below 19 % of a period and better above it.
//! * [`gooding`] — the modern one (Gooding 1993/1997). Two unknowns, the ranges at the first and
//!   third sightings, solved by Newton against the angular miss at the middle one, with
//!   [`lambert`](fn@crate::lambert) closing the arc exactly. **On noiseless data it is exact**: there is
//!   no series and no truncation, so the only error is the convergence tolerance.
//! * [`gibbs`] and [`herrick_gibbs`] — velocity from three *position* vectors, which is a different
//!   problem (a radar or a GNSS receiver gives positions; a telescope does not). Gibbs is the
//!   well-separated case, Herrick–Gibbs the closely-spaced one, and the crossover is measured.
//! * [`orbit_from_angles`] — Gauss to seed Gooding, which is how the two are meant to be used.
//!
//! Orekit's `IodLambert` is this crate's [`lambert`](fn@crate::lambert) with nothing added, so it is not
//! reproduced here.
//!
//! # Measured — `tests/initial_orbit_determination.rs`
//!
//! Recovering a 700 km sun-synchronous state from three exact sightings, against the arc the three
//! sightings span, as a fraction of the orbital period:
//!
//! ```text
//!     arc            0.1 %      1 %       5 %       10 %      25 %
//!     Gauss          10.1 m    1009 m    25.3 km   103 km    720 km
//!     Laplace        12.3 m    1231 m    30.5 km   119 km    634 km
//!     Gooding       3.2e-4 m  1.2e-6 m  1.6e-8 m  2.8e-7 m  1.5e-9 m
//! ```
//!
//! **Gauss's error grows as the square of the arc, not the cube.** Measured log-log slopes over the
//! four consecutive pairs above: 2.000, 2.003, 2.021, 2.124. The cube is the order of the first term
//! its `f` and `g` series drop, and the temptation is to quote it — but the truncation enters the
//! range solution through a geometry whose sensitivity carries an inverse power of the arc, and one
//! of the three cancels. Use Gauss as a seed, not as an answer.
//!
//! Gooding is exact to the iteration's own convergence floor at every arc length, because it solves
//! the two-body problem rather than approximating it. That floor is not flat: it is worst on the
//! shortest arc, where a 1e-13 rad angular residual divides by a 6.3e-3 rad arc to leave 3.2e-4 m of
//! range. Even there it beats Gauss by four orders of magnitude, and by eleven at 10 %.

// The IOD literature's symbols (D, A, B, E, N, S, r, v, τ) are one letter by universal convention,
// and the matrix code indexes several arrays by one i/j/k.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use crate::kepler;
use crate::lambert::lambert;
use spacelab_core::Vec3;

/// One optical observation: when, which way, and from where.
///
/// `line_of_sight` is a **unit** vector from the observer to the object, in the same inertial frame
/// as `observer`. It is not normalised on the caller's behalf — a line of sight that is not a unit
/// vector means the caller has built it wrong, and silently rescaling it would hide that.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sighting {
    /// Seconds from an arbitrary common origin; only differences are used.
    pub time: f64,
    /// Unit vector from the observer towards the object.
    pub line_of_sight: Vec3,
    /// The observer's inertial position at `time`.
    pub observer: Vec3,
}

/// Why an initial orbit could not be determined.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IodError {
    /// A line of sight is not a unit vector, or an input is not finite. The offending sighting is
    /// named rather than repaired.
    MalformedSighting {
        /// Index of the sighting in the triple.
        index: usize,
    },
    /// The three epochs are not strictly increasing. Two observations at the same instant carry no
    /// arc, and an out-of-order triple silently produces a mirrored orbit.
    NonMonotonicEpochs,
    /// The three lines of sight lie in a plane containing the observers, so the triangle Gauss's
    /// method builds is degenerate and the ranges are not separable. This is the geometry a single
    /// station observing an object in its own orbital plane produces.
    DegenerateGeometry {
        /// The scalar triple product `ρ̂₁ · (ρ̂₂ × ρ̂₃)`, which is zero in the degenerate case.
        triple_product: f64,
    },
    /// The 8th-degree polynomial has no positive root that puts all three objects in front of their
    /// observers. A negative range is a solution to the algebra and not to the problem.
    NoPhysicalRange,
    /// Three position vectors that are not coplanar cannot lie on one Keplerian orbit. The reported
    /// value is the sine of the angle between `r₁` and the plane of `r₂` and `r₃`; anything above
    /// about 0.06 (3.5°) is measurement error rather than an orbit.
    NotCoplanar {
        /// `sin` of the out-of-plane angle.
        sine: f64,
    },
    /// Three position vectors that are collinear, or two that coincide, carry no orbit.
    Degenerate,
    /// [`lambert`](fn@crate::lambert) could not connect the trial positions — a 0° or 180° transfer, or
    /// a non-positive time of flight.
    LambertFailed,
    /// Newton did not converge in the iteration budget. The best residual reached is reported, in
    /// radians of angular miss at the middle sighting, so the caller can tell a hard failure from a
    /// tolerance that was set too tight.
    DidNotConverge {
        /// Angular miss at the middle sighting, in radians.
        residual: f64,
    },
}

impl core::fmt::Display for IodError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MalformedSighting { index } => write!(
                f,
                "sighting {index} has a non-finite component or a line of sight that is not a unit vector"
            ),
            Self::NonMonotonicEpochs => write!(f, "the three epochs are not strictly increasing"),
            Self::DegenerateGeometry { triple_product } => write!(
                f,
                "the lines of sight are coplanar (triple product {triple_product:e}); the ranges are not separable"
            ),
            Self::NoPhysicalRange => write!(
                f,
                "no positive root of the range polynomial puts the object in front of all three observers"
            ),
            Self::NotCoplanar { sine } => write!(
                f,
                "the three positions are not coplanar (sine {sine:e}); they are not one Keplerian orbit"
            ),
            Self::Degenerate => write!(f, "the three positions are collinear or coincident"),
            Self::LambertFailed => write!(f, "no Lambert arc connects the trial positions"),
            Self::DidNotConverge { residual } => write!(
                f,
                "Newton did not converge; angular miss {residual:e} rad at the middle sighting"
            ),
        }
    }
}

impl core::error::Error for IodError {}

/// Reject anything that is not three finite, strictly ordered sightings with unit lines of sight.
///
/// Validation is here, at the boundary, and not repeated inside each method: an IOD routine given a
/// line of sight of length 0.98 returns a plausible orbit that is wrong by 2 %, which is exactly the
/// failure that never gets caught downstream.
fn validate(sightings: &[Sighting; 3]) -> Result<(), IodError> {
    for (index, sighting) in sightings.iter().enumerate() {
        let finite = sighting.time.is_finite()
            && [sighting.line_of_sight, sighting.observer]
                .iter()
                .all(|v| v.x.is_finite() && v.y.is_finite() && v.z.is_finite());
        if !finite || (sighting.line_of_sight.norm() - 1.0).abs() > 1e-9 {
            return Err(IodError::MalformedSighting { index });
        }
    }
    if !(sightings[0].time < sightings[1].time && sightings[1].time < sightings[2].time) {
        return Err(IodError::NonMonotonicEpochs);
    }
    Ok(())
}

/// Velocity at `r2` from three coplanar, well-separated position vectors (Gibbs 1889).
///
/// The classical construction: `N`, `D` and `S` are built so that `v₂ = √(μ/|N||D|)·(D×r̂₂ + S)` is
/// the exact two-body velocity, with no series and no iteration. It needs a real transfer angle
/// between the vectors to be conditioned — below about 1°, `D` is a difference of near-equal
/// quantities and the answer is noise. [`herrick_gibbs`] is the method for that case.
///
/// # Errors
///
/// [`IodError::NotCoplanar`] if the three positions do not lie in one plane, and
/// [`IodError::Degenerate`] if they are collinear or coincident.
pub fn gibbs(r1: Vec3, r2: Vec3, r3: Vec3, mu: f64) -> Result<Vec3, IodError> {
    let (n1, n2, n3) = (r1.norm(), r2.norm(), r3.norm());
    if n1 == 0.0 || n2 == 0.0 || n3 == 0.0 {
        return Err(IodError::Degenerate);
    }
    let c23 = r2.cross(r3);
    // The out-of-plane angle of r1 against the plane of r2 and r3. Vallado's 3° screen, as a sine.
    let plane = c23.norm();
    if plane == 0.0 {
        return Err(IodError::Degenerate);
    }
    let sine = r1.dot(c23) / (n1 * plane);
    if sine.abs() > 0.0603 {
        return Err(IodError::NotCoplanar { sine });
    }

    let d = c23 + r3.cross(r1) + r1.cross(r2);
    let n = c23.scale(n1) + r3.cross(r1).scale(n2) + r1.cross(r2).scale(n3);
    let (dn, nn) = (d.norm(), n.norm());
    if dn == 0.0 || nn == 0.0 {
        return Err(IodError::Degenerate);
    }
    let s = r1.scale(n2 - n3) + r2.scale(n3 - n1) + r3.scale(n1 - n2);
    Ok((d.cross(r2).scale(1.0 / n2) + s).scale(libm::sqrt(mu / (nn * dn))))
}

/// Velocity at `r2` from three closely spaced position vectors (Herrick–Gibbs).
///
/// Where [`gibbs`] is exact but ill-conditioned on a short arc, this is a truncated Taylor series
/// that is well-conditioned there and degrades as the arc opens. The two swap places between 0.42°
/// and 0.49° of transfer angle — bracketed, not quoted, by
/// `tests/initial_orbit_determination.rs`, which walks the sweep and reports the step the ordering
/// flips on.
///
/// # Errors
///
/// [`IodError::NonMonotonicEpochs`] if the epochs are not strictly increasing, and
/// [`IodError::Degenerate`] if a position vector is zero.
pub fn herrick_gibbs(
    r1: Vec3,
    r2: Vec3,
    r3: Vec3,
    t1: f64,
    t2: f64,
    t3: f64,
    mu: f64,
) -> Result<Vec3, IodError> {
    if !(t1 < t2 && t2 < t3) {
        return Err(IodError::NonMonotonicEpochs);
    }
    let (n1, n2, n3) = (r1.norm(), r2.norm(), r3.norm());
    if n1 == 0.0 || n2 == 0.0 || n3 == 0.0 {
        return Err(IodError::Degenerate);
    }
    let (d21, d32, d31) = (t2 - t1, t3 - t2, t3 - t1);
    let c1 = -d32 * (1.0 / (d21 * d31) + mu / (12.0 * n1 * n1 * n1));
    let c2 = (d32 - d21) * (1.0 / (d21 * d32) + mu / (12.0 * n2 * n2 * n2));
    let c3 = d21 * (1.0 / (d32 * d31) + mu / (12.0 * n3 * n3 * n3));
    Ok(r1.scale(c1) + r2.scale(c2) + r3.scale(c3))
}

/// Gauss's method: the state at the middle sighting, from one 8th-degree polynomial and no iteration.
///
/// Fast, unconditional, and — measured, not derived — second order in the arc length. The `f` and
/// `g` series it closes the geometry with are truncated after the `τ²` and `τ³` terms, so the first
/// dropped term is cubic; the range solution's sensitivity carries an inverse power of the arc and
/// leaves a square. On a 700 km orbit: 10.1 m over 0.1 % of a period, 1009 m over 1 %, 103 km over
/// 10 %. It is a seed for [`gooding`], and [`orbit_from_angles`] is the two of them composed.
///
/// # Errors
///
/// [`IodError::MalformedSighting`], [`IodError::NonMonotonicEpochs`],
/// [`IodError::DegenerateGeometry`] when the lines of sight are coplanar, or
/// [`IodError::NoPhysicalRange`] when no root of the polynomial puts the object in front of every
/// observer.
pub fn gauss(sightings: &[Sighting; 3], mu: f64) -> Result<(Vec3, Vec3), IodError> {
    let ranges = gauss_ranges(sightings, mu)?;
    let position: [Vec3; 3] = core::array::from_fn(|i| {
        sightings[i].observer + sightings[i].line_of_sight.scale(ranges[i])
    });

    // Lagrange f and g truncated at the order Gauss's construction assumes, which is the same
    // approximation the ranges were solved under — using an exact f and g here would be
    // inconsistent, not better.
    let (t1, t2, t3) = (sightings[0].time, sightings[1].time, sightings[2].time);
    let (tau1, tau3) = (t1 - t2, t3 - t2);
    let r2 = position[1].norm();
    let k = mu / (r2 * r2 * r2);
    let f1 = 1.0 - 0.5 * k * tau1 * tau1;
    let f3 = 1.0 - 0.5 * k * tau3 * tau3;
    let g1 = tau1 - k * tau1 * tau1 * tau1 / 6.0;
    let g3 = tau3 - k * tau3 * tau3 * tau3 / 6.0;
    let denominator = f1 * g3 - f3 * g1;
    if denominator == 0.0 {
        return Err(IodError::DegenerateGeometry {
            triple_product: 0.0,
        });
    }
    let velocity = (position[0].scale(-f3) + position[2].scale(f1)).scale(1.0 / denominator);
    Ok((position[1], velocity))
}

/// Laplace's method: the state at the middle sighting, from the observed angular rates.
///
/// Where [`gauss`] interpolates the three lines of sight and [`gooding`] fits an orbit to them,
/// Laplace **differentiates** them: a quadratic is passed through the three directions and through
/// the three observer positions, and its first and second derivatives at the middle epoch are fed
/// to the equation of motion. Two projections of that equation — one perpendicular to `L` and `L̇`,
/// one perpendicular to `L` and `L̈` — remove `ρ̈` and give the range and the range rate directly,
/// through the same 8th-degree polynomial shape Gauss ends at.
///
/// It needs no transfer-sense flag and no seed, and it never has to pick which Lambert arc closes.
/// The literature's verdict on it is that a derivative estimator is more sensitive to noise on a
/// short arc and no better on a long one. Measured on the same sightings the other two are measured
/// on — `tests/initial_orbit_determination.rs`, where these numbers regenerate — that verdict is
/// half right, and it is the wrong half that survives.
///
/// ```text
///     arc                0.1 %      1 %       5 %       10 %      25 %
///     Gauss   position   10.1 m    1009 m    25.3 km   103 km    720 km
///     Laplace position   12.3 m    1231 m    30.5 km   119 km    634 km
///     Gooding position  3.2e-4 m  1.2e-6 m  1.6e-8 m  2.8e-7 m  1.5e-9 m
///
///     Gauss   velocity  0.0134    1.338     33.0      126.4     540.9   m/s
///     Laplace velocity  0.0262    2.622     65.0      253.7     1354    m/s
/// ```
///
/// **Laplace's position is worse than Gauss's below 19 % of a period and better above it.** The
/// crossover is bracketed at 18.96 %–20.00 % by the sweep in the test, and it is a race between two
/// second-order truncations rather than a change of regime: fitted log-log slopes are 2.000, 2.003,
/// 2.021, 2.124 for Gauss — *rising* past two as the arc opens — against 2.000, 1.995, 1.961, 1.827
/// for Laplace, falling. Gauss starts 20 % lower and is overtaken.
///
/// **The overtake is only half the state.** Laplace's velocity is worse at every arc measured, by
/// 1.96× at 0.1 % and 2.50× at 25 % — so at the one arc length where its position wins by 12 % its
/// velocity loses by a factor of two and a half. There is no arc at which it is the better estimate
/// of the state.
///
/// **On a short noisy arc the two are a tie, not a loss.** Paired over 2 000 trials on identical
/// noisy sightings with 1 arcsecond of Gaussian error per angle, Laplace has the smaller position
/// error in 49.0 % of pairs at 0.5 % of a period and 48.8 % at 1 % — a coin flip to within the
/// 1.1 percentage-point standard error, at 0.5″, 1″ and 3″ alike. The truncation that separates the
/// two on clean data is 10³ times smaller than what a 1″ error does to the range, so neither method's
/// series is what is being measured there. By 5 % of a period truncation dominates again and the
/// win rate collapses to 1.1 %.
///
/// So it is kept for parity, and for the case where a caller has three angles and no way to say
/// which sense the transfer runs in. [`orbit_from_angles`] remains the entry point: Gooding is four
/// to eleven orders of magnitude better than either of these at every arc length here.
///
/// # Errors
///
/// [`IodError::MalformedSighting`], [`IodError::NonMonotonicEpochs`],
/// [`IodError::DegenerateGeometry`] when `L`, `L̇` and `L̈` are coplanar — the projection that
/// separates `ρ` from `ρ̈` is then empty — or [`IodError::NoPhysicalRange`] when no root of the
/// polynomial puts the object in front of the observer.
pub fn laplace(sightings: &[Sighting; 3], mu: f64) -> Result<(Vec3, Vec3), IodError> {
    validate(sightings)?;
    let (t1, t2, t3) = (sightings[0].time, sightings[1].time, sightings[2].time);
    // Lagrange's quadratic through the three samples, differentiated at t₂. `validate` has already
    // rejected a repeated epoch, so every denominator here is nonzero.
    let first = [
        (t2 - t3) / ((t1 - t2) * (t1 - t3)),
        ((t2 - t1) + (t2 - t3)) / ((t2 - t1) * (t2 - t3)),
        (t2 - t1) / ((t3 - t1) * (t3 - t2)),
    ];
    let second = [
        2.0 / ((t1 - t2) * (t1 - t3)),
        2.0 / ((t2 - t1) * (t2 - t3)),
        2.0 / ((t3 - t1) * (t3 - t2)),
    ];
    let combine = |w: &[f64; 3], pick: fn(&Sighting) -> Vec3| {
        (0..3).fold(Vec3::new(0.0, 0.0, 0.0), |acc, i| {
            acc + pick(&sightings[i]).scale(w[i])
        })
    };
    let l = sightings[1].line_of_sight;
    let rate = combine(&first, |s| s.line_of_sight);
    let curvature = combine(&second, |s| s.line_of_sight);
    let big_r = sightings[1].observer;
    let observer_rate = combine(&first, |s| s.observer);
    let observer_curvature = combine(&second, |s| s.observer);

    // Scale-free: `L` is a unit vector, so |L·(L̇ × L̈)| ≤ |L̇||L̈| and the ratio is a sine. It is
    // zero exactly when the three lines of sight and their curvature share a plane, which is the
    // geometry that makes the two projections below the same equation twice.
    let shape = l.dot(rate.cross(curvature)) / (rate.norm() * curvature.norm());
    if !shape.is_finite() || shape.abs() < 1e-12 {
        return Err(IodError::DegenerateGeometry {
            triple_product: shape,
        });
    }
    let d = 2.0 * l.dot(rate.cross(curvature));
    let d1 = l.dot(rate.cross(observer_curvature));
    let d2 = l.dot(rate.cross(big_r));
    let d3 = l.dot(curvature.cross(observer_curvature));
    let d4 = l.dot(curvature.cross(big_r));

    // Projecting `R̈ + ρ̈L + 2ρ̇L̇ + ρL̈ = −μ(R + ρL)/r³` onto `L × L̇` kills `ρ̈` and `ρ̇` and leaves
    // `ρ = p + q/r³`; substituting that into `r² = ρ² + 2ρ(L·R) + R²` clears to the same
    // `r⁸ + c₆r⁶ + c₃r³ + c₀` Gauss arrives at by a different route.
    let (p, q) = (-2.0 * d1 / d, -2.0 * mu * d2 / d);
    let (lr, rr) = (l.dot(big_r), big_r.dot(big_r));
    let c6 = -(p * p + 2.0 * lr * p + rr);
    let c3 = -2.0 * q * (p + lr);
    let c0 = -q * q;
    let polynomial = |r: f64| {
        let r3 = r * r * r;
        let r6 = r3 * r3;
        r6 * r * r + c6 * r6 + c3 * r3 + c0
    };

    let scale = libm::sqrt(rr);
    let (lo, hi) = (0.01 * scale, 1000.0 * scale);
    let at = |i: usize| lo * libm::pow(hi / lo, i as f64 / SAMPLES as f64);
    let mut previous = polynomial(lo);
    for i in 1..=SAMPLES {
        let point = at(i);
        let value = polynomial(point);
        if previous * value <= 0.0 && previous != 0.0 {
            let r = bisect(&polynomial, at(i - 1), point);
            let inverse_cube = 1.0 / (r * r * r);
            let rho = p + q * inverse_cube;
            // A range behind the observer solves the algebra and not the problem, the same filter
            // `gauss_ranges` applies to its own roots.
            if rho > 0.0 {
                let rho_dot = (d3 + mu * inverse_cube * d4) / d;
                let position = big_r + l.scale(rho);
                let velocity = observer_rate + rate.scale(rho) + l.scale(rho_dot);
                return Ok((position, velocity));
            }
        }
        previous = value;
    }
    Err(IodError::NoPhysicalRange)
}

/// The three slant ranges Gauss's construction gives, before any orbit is formed from them.
///
/// Split out because [`orbit_from_angles`] wants the ranges as a Newton seed and not the state they
/// imply — forming the state and inverting it back into ranges would throw away precision for no
/// reason.
/// Samples of the scale-free range scan in [`gauss_ranges`].
const SAMPLES: usize = 4096;

/// Halvings [`gooding`] will spend keeping a Newton step inside the physical half-space. Twenty of
/// them take the damping factor to `2⁻²⁰ = 9.5e-7`.
const DAMPING_HALVINGS: usize = 20;

fn gauss_ranges(sightings: &[Sighting; 3], mu: f64) -> Result<[f64; 3], IodError> {
    validate(sightings)?;
    let l: [Vec3; 3] = core::array::from_fn(|i| sightings[i].line_of_sight);
    let big_r: [Vec3; 3] = core::array::from_fn(|i| sightings[i].observer);
    let (t1, t2, t3) = (sightings[0].time, sightings[1].time, sightings[2].time);
    let (tau1, tau3, tau) = (t1 - t2, t3 - t2, t3 - t1);

    let p = [l[1].cross(l[2]), l[0].cross(l[2]), l[0].cross(l[1])];
    let d0 = l[0].dot(p[0]);
    // Scale-free degeneracy test: `d0` is a triple product of unit vectors, so it *is* the sine of
    // the angle between ρ̂₁ and the plane of the other two.
    if d0.abs() < 1e-12 {
        return Err(IodError::DegenerateGeometry { triple_product: d0 });
    }
    let mut d = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = big_r[i].dot(p[j]);
        }
    }

    let a = (-d[0][1] * tau3 / tau + d[1][1] + d[2][1] * tau1 / tau) / d0;
    let b = (d[0][1] * (tau3 * tau3 - tau * tau) * tau3 / tau
        + d[2][1] * (tau * tau - tau1 * tau1) * tau1 / tau)
        / (6.0 * d0);
    let e = big_r[1].dot(l[1]);
    let r2_squared = big_r[1].dot(big_r[1]);

    // r₂⁸ + c6 r₂⁶ + c3 r₂³ + c0 = 0.
    let c6 = -(a * a + 2.0 * a * e + r2_squared);
    let c3 = -2.0 * mu * b * (a + e);
    let c0 = -mu * mu * b * b;
    let polynomial = |r: f64| {
        let r3 = r * r * r;
        let r6 = r3 * r3;
        r6 * r * r + c6 * r6 + c3 * r3 + c0
    };

    // The polynomial can have more than one positive root — the classical ambiguity — so every root
    // is found and the physical filter (all three objects in front of their observers) picks. A
    // scan bracketed on `|R₂|` is scale-free, so this works heliocentrically as well as in orbit.
    let scale = libm::sqrt(r2_squared);
    let (lo, hi) = (0.01 * scale, 1000.0 * scale);
    let at = |i: usize| lo * libm::pow(hi / lo, i as f64 / SAMPLES as f64);

    let mut best: Option<[f64; 3]> = None;
    let mut previous = polynomial(lo);
    for i in 1..=SAMPLES {
        let point = at(i);
        let value = polynomial(point);
        if previous * value <= 0.0 && previous != 0.0 && best.is_none() {
            let root = bisect(&polynomial, at(i - 1), point);
            best = ranges_at(&d, d0, mu, root, tau1, tau3, tau)
                .filter(|ranges| ranges.iter().all(|&rho| rho > 0.0));
        }
        previous = value;
    }
    best.ok_or(IodError::NoPhysicalRange)
}

/// The three slant ranges implied by a trial `r₂`, or `None` where the construction divides by zero.
fn ranges_at(
    d: &[[f64; 3]; 3],
    d0: f64,
    mu: f64,
    r2: f64,
    tau1: f64,
    tau3: f64,
    tau: f64,
) -> Option<[f64; 3]> {
    let r23 = r2 * r2 * r2;
    let first = 6.0 * r23 + mu * (tau * tau - tau3 * tau3);
    let third = 6.0 * r23 + mu * (tau * tau - tau1 * tau1);
    if first == 0.0 || third == 0.0 || tau1 == 0.0 || tau3 == 0.0 {
        return None;
    }
    let rho1 = ((6.0 * (d[2][0] * tau1 / tau3 + d[1][0] * tau / tau3) * r23
        + mu * d[2][0] * (tau * tau - tau1 * tau1) * tau1 / tau3)
        / first
        - d[0][0])
        / d0;
    let rho2 = (-d[0][1] * tau3 / tau + d[1][1] + d[2][1] * tau1 / tau) / d0
        + mu * ((d[0][1] * (tau3 * tau3 - tau * tau) * tau3 / tau
            + d[2][1] * (tau * tau - tau1 * tau1) * tau1 / tau)
            / (6.0 * d0))
            / r23;
    let rho3 = ((6.0 * (d[0][2] * tau3 / tau1 - d[1][2] * tau / tau1) * r23
        + mu * d[0][2] * (tau * tau - tau3 * tau3) * tau3 / tau1)
        / third
        - d[2][2])
        / d0;
    [rho1, rho2, rho3]
        .iter()
        .all(|v| v.is_finite())
        .then_some([rho1, rho2, rho3])
}

/// Bisection on a bracket the caller has already established, to the last representable bit.
fn bisect(f: &impl Fn(f64) -> f64, mut lo: f64, mut hi: f64) -> f64 {
    let low_side = f(lo) < 0.0;
    // 80 halvings takes any bracket below f64 resolution; a fixed cost, not a search.
    for _ in 0..80 {
        let mid = f64::midpoint(lo, hi);
        if (f(mid) < 0.0) == low_side {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// Two unit vectors spanning the plane perpendicular to `axis`.
///
/// The seed is the axis with `axis`'s smallest component, which is the standard way to keep the
/// cross product away from zero — picking a fixed seed fails whenever `axis` happens to be parallel
/// to it, and a line of sight eventually is.
fn perpendicular_basis(axis: Vec3) -> (Vec3, Vec3) {
    let seed = if axis.x.abs() <= axis.y.abs() && axis.x.abs() <= axis.z.abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if axis.y.abs() <= axis.z.abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let u = axis.cross(seed);
    let u = u.scale(1.0 / u.norm());
    (u, axis.cross(u))
}

/// The angular miss at the middle sighting for a trial pair of ranges, in the plane perpendicular to
/// the observed line of sight.
///
/// Two components for two unknowns. Measuring the miss as an angle rather than as a position keeps
/// the residual in the units the observation is actually in, so the Newton step is scaled by the
/// data and not by how far away the object happens to be.
fn angular_miss(
    sightings: &[Sighting; 3],
    mu: f64,
    prograde: bool,
    rho1: f64,
    rho3: f64,
) -> Result<([f64; 2], Vec3, Vec3), IodError> {
    let r1 = sightings[0].observer + sightings[0].line_of_sight.scale(rho1);
    let r3 = sightings[2].observer + sightings[2].line_of_sight.scale(rho3);
    let tof = sightings[2].time - sightings[0].time;
    let (v1, _) = lambert(r1, r3, tof, mu, prograde).ok_or(IodError::LambertFailed)?;

    let (r2, _) = kepler::propagate(r1, v1, mu, sightings[1].time - sightings[0].time);
    let slant = r2 - sightings[1].observer;
    let distance = slant.norm();
    if distance == 0.0 {
        return Err(IodError::LambertFailed);
    }
    let predicted = slant.scale(1.0 / distance);
    let (u, w) = perpendicular_basis(sightings[1].line_of_sight);
    Ok(([predicted.dot(u), predicted.dot(w)], r1, v1))
}

/// Gooding's method: the exact two-body orbit through three lines of sight.
///
/// Newton on the two slant ranges `ρ₁` and `ρ₃`, with the residual being the angular miss at the
/// middle sighting and [`lambert`](fn@crate::lambert) closing the arc between the first and third. No
/// series is truncated anywhere, so on noiseless data the answer is exact to the convergence
/// tolerance — measured below `1e-6` m at every arc length from 0.1 % to 25 % of a period, where
/// Gauss is off by 280 km at the far end.
///
/// `prograde` selects the transfer sense, which three lines of sight genuinely do not determine:
/// both senses fit the angles and only one fits the physics. The caller knows which; guessing it
/// here and returning the better-fitting root would be picking an orbit by residual, which on noisy
/// data picks the wrong one.
///
/// `initial` is the seed `(ρ₁, ρ₃)`. [`orbit_from_angles`] supplies it from [`gauss`], which is the
/// intended pairing; a caller with a radar range or a catalogue prior should pass that instead,
/// because it is better than anything Gauss will produce.
///
/// # Errors
///
/// [`IodError::MalformedSighting`], [`IodError::NonMonotonicEpochs`], [`IodError::LambertFailed`] if
/// no arc connects a trial pair, or [`IodError::DidNotConverge`] with the angular miss it reached.
pub fn gooding(
    sightings: &[Sighting; 3],
    mu: f64,
    prograde: bool,
    initial: (f64, f64),
) -> Result<(Vec3, Vec3), IodError> {
    validate(sightings)?;
    // The observed direction, in the same perpendicular basis the residual is measured in: the
    // target the miss is driven to. Both components are zero there by construction, so the residual
    // *is* the projection — but writing it as a difference keeps the two sides symmetric under a
    // change of basis and costs nothing.
    let (u, w) = perpendicular_basis(sightings[1].line_of_sight);
    let target = [
        sightings[1].line_of_sight.dot(u),
        sightings[1].line_of_sight.dot(w),
    ];

    let (mut rho1, mut rho3) = initial;
    // A step scaled to the geometry rather than to a fixed number of metres: the Jacobian is
    // dimensionless per unit range, so the difference has to scale with the range itself.
    let scale = sightings[1].observer.norm();
    let mut residual = f64::INFINITY;

    for _ in 0..60 {
        let (miss, r1, v1) = angular_miss(sightings, mu, prograde, rho1, rho3)?;
        let f = [miss[0] - target[0], miss[1] - target[1]];
        residual = libm::hypot(f[0], f[1]);
        if residual < 1e-13 {
            let (r2, v2) = kepler::propagate(r1, v1, mu, sightings[1].time - sightings[0].time);
            return Ok((r2, v2));
        }

        let h = 1e-6 * scale;
        let (plus1, _, _) = angular_miss(sightings, mu, prograde, rho1 + h, rho3)?;
        let (plus3, _, _) = angular_miss(sightings, mu, prograde, rho1, rho3 + h)?;
        let jacobian = [
            [(plus1[0] - miss[0]) / h, (plus3[0] - miss[0]) / h],
            [(plus1[1] - miss[1]) / h, (plus3[1] - miss[1]) / h],
        ];
        let determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
        if determinant == 0.0 || !determinant.is_finite() {
            return Err(IodError::DidNotConverge { residual });
        }
        let step1 = (jacobian[1][1] * f[0] - jacobian[0][1] * f[1]) / determinant;
        let step3 = (jacobian[0][0] * f[1] - jacobian[1][0] * f[0]) / determinant;

        // Never step a range through zero: an object behind its observer is not a solution, and
        // Newton will happily propose one from a poor seed. Halving instead keeps the iterate in the
        // physical half-space without abandoning the direction.
        // Counted rather than bounded by `damping > 1e-6`: written that way the loop terminates
        // because of what `*= 0.5` does, and the mutation seal turns `*=` into `+=` or `/=` and
        // hangs the suite for 180 s at a time (#292). Twenty halvings put `damping` at 9.5e-7, the
        // same floor the value test enforced.
        let mut damping = 1.0;
        for _ in 0..DAMPING_HALVINGS {
            if rho1 - damping * step1 > 0.0 && rho3 - damping * step3 > 0.0 {
                break;
            }
            damping *= 0.5;
        }
        rho1 -= damping * step1;
        rho3 -= damping * step3;
        if !rho1.is_finite() || !rho3.is_finite() {
            return Err(IodError::DidNotConverge { residual });
        }
    }
    Err(IodError::DidNotConverge { residual })
}

/// [`gauss`] to seed [`gooding`]: the state at the middle sighting, exactly, from three angles.
///
/// This is the entry point to reach for. Gauss is unconditional and approximate, Gooding is exact
/// and needs a seed, and the two compose into a method that is both — which is how Gooding's paper
/// presents it and how it is used in practice.
///
/// # Errors
///
/// Whatever [`gauss`] or [`gooding`] returns. A [`IodError::DidNotConverge`] here usually means the
/// arc is too short for the geometry to separate range from angle at all, rather than that the
/// iteration was unlucky.
pub fn orbit_from_angles(
    sightings: &[Sighting; 3],
    mu: f64,
    prograde: bool,
) -> Result<(Vec3, Vec3), IodError> {
    let seed = gauss_ranges(sightings, mu)?;
    gooding(sightings, mu, prograde, (seed[0], seed[2]))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]

    use super::*;

    const MU: f64 = 3.986_004_418e14;

    /// Three exact sightings of a known orbit from a fixed observer, spanning `arc` seconds.
    /// Three exact sightings spanning `arc` seconds, centred on the middle epoch, and the true state
    /// there.
    fn synthesise(r0: Vec3, v0: Vec3, observer: Vec3, arc: f64) -> ([Sighting; 3], Vec3, Vec3) {
        let times = [-0.5 * arc, 0.0, 0.5 * arc];
        let states: [(Vec3, Vec3); 3] =
            core::array::from_fn(|i| kepler::propagate(r0, v0, MU, times[i]));
        let sightings = core::array::from_fn(|i| {
            let line = states[i].0 - observer;
            Sighting {
                time: times[i],
                line_of_sight: line.scale(1.0 / line.norm()),
                observer,
            }
        });
        (sightings, states[1].0, states[1].1)
    }

    fn reference() -> (Vec3, Vec3, Vec3) {
        // A 700 km sun-synchronous orbit, and a ground station well off its plane so the geometry is
        // not degenerate.
        let r = Vec3::new(7_078_137.0, 0.0, 0.0);
        let speed = libm::sqrt(MU / r.norm());
        let inclination = 98.2_f64.to_radians();
        let v = Vec3::new(
            0.0,
            speed * libm::cos(inclination),
            speed * libm::sin(inclination),
        );
        let observer = Vec3::new(4_000_000.0, 4_500_000.0, 1_500_000.0);
        (r, v, observer)
    }

    #[test]
    fn gibbs_and_herrick_gibbs_agree_with_the_truth_in_their_own_regimes() {
        let (r0, v0, _) = reference();
        let period = core::f64::consts::TAU * libm::sqrt(libm::pow(r0.norm(), 3.0) / MU);
        // Gibbs wants a real transfer angle; Herrick–Gibbs wants a short one.
        for (fraction, wide) in [(0.20, true), (0.001, false)] {
            let arc = fraction * period;
            let times = [-0.5 * arc, 0.0, 0.5 * arc];
            let states: [(Vec3, Vec3); 3] =
                core::array::from_fn(|i| kepler::propagate(r0, v0, MU, times[i]));
            let recovered = if wide {
                gibbs(states[0].0, states[1].0, states[2].0, MU).unwrap()
            } else {
                herrick_gibbs(
                    states[0].0,
                    states[1].0,
                    states[2].0,
                    times[0],
                    times[1],
                    times[2],
                    MU,
                )
                .unwrap()
            };
            let error = (recovered - states[1].1).norm();
            assert!(error < 1e-3, "arc {fraction}, wide {wide}: {error} m/s");
        }
    }

    #[test]
    fn a_malformed_sighting_is_named_rather_than_repaired() {
        let (r0, v0, observer) = reference();
        let (good, _, _) = synthesise(r0, v0, observer, 300.0);

        let mut short = good;
        short[1].line_of_sight = short[1].line_of_sight.scale(0.98);
        assert_eq!(
            gauss(&short, MU),
            Err(IodError::MalformedSighting { index: 1 })
        );

        let mut nan = good;
        nan[2].observer = Vec3::new(f64::NAN, 0.0, 0.0);
        assert_eq!(
            gauss(&nan, MU),
            Err(IodError::MalformedSighting { index: 2 })
        );

        let mut backwards = good;
        backwards.swap(0, 2);
        assert_eq!(gauss(&backwards, MU), Err(IodError::NonMonotonicEpochs));

        let mut simultaneous = good;
        simultaneous[2].time = simultaneous[1].time;
        assert_eq!(gauss(&simultaneous, MU), Err(IodError::NonMonotonicEpochs));
    }

    /// The geometry Gauss cannot solve: an observer in the object's own orbital plane, so all three
    /// lines of sight are coplanar and the triangle collapses.
    #[test]
    fn a_coplanar_geometry_is_refused_rather_than_answered() {
        let (r0, v0, _) = reference();
        // The orbit's plane normal, and an observer placed inside the plane.
        let normal = r0.cross(v0);
        let in_plane = normal.cross(r0);
        let observer = in_plane.scale(4.0e6 / in_plane.norm());
        let (sightings, _, _) = synthesise(r0, v0, observer, 600.0);
        assert!(matches!(
            gauss(&sightings, MU),
            Err(IodError::DegenerateGeometry { .. })
        ));

        // The screen's own threshold, exactly. `d0 = ρ̂₁ · (ρ̂₂ × ρ̂₃)`; with `ρ̂₂ = x̂` and `ρ̂₃ = ŷ`
        // the cross product is `ẑ`, `Vec3::dot` fuses so the triple product is bit-exactly `ρ̂₁.z`,
        // and `(1, 0, z)` has an exact unit norm for any `z` below `2⁻²⁶` — so `d0` is `1e-12` with
        // no rounding in it, the one argument at which `<` and `<=` disagree. `1e-12` is a sine of
        // 6e-11 degrees and there is no orbit behind it, but the screen is written to reject
        // *below* the threshold, and where it stands is a decision the test has to make visible.
        let triple = |z: f64| {
            let lines = [
                Vec3::new(1.0, 0.0, z),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ];
            core::array::from_fn(|i| Sighting {
                time: i as f64,
                line_of_sight: lines[i],
                observer,
            })
        };
        let at: [Sighting; 3] = triple(1e-12);
        assert_eq!(
            at[0].line_of_sight.norm(),
            1.0,
            "the construction needs a unit line of sight"
        );
        assert!(
            !matches!(
                gauss_ranges(&at, MU),
                Err(IodError::DegenerateGeometry { .. })
            ),
            "1e-12 is admitted"
        );
        let under = triple(f64::from_bits(1e-12_f64.to_bits() - 1));
        assert!(
            matches!(
                gauss_ranges(&under, MU),
                Err(IodError::DegenerateGeometry { .. })
            ),
            "one double below it is not"
        );
    }

    /// Laplace's three refusals, each reached by a different route than Gauss's.
    ///
    /// Its degeneracy test is not Gauss's. Gauss screens `ρ̂₁ · (ρ̂₂ × ρ̂₃)`, the volume the three
    /// directions span; Laplace screens `L · (L̇ × L̈)`, the volume the *middle* direction spans with
    /// its own derivatives. Coplanar sightings collapse both, so one geometry exercises both — but
    /// they are different expressions and a test that only ran Gauss would leave Laplace's unproven.
    #[test]
    fn laplace_refuses_the_geometries_it_cannot_solve() {
        let (r0, v0, observer) = reference();
        let (good, _, _) = synthesise(r0, v0, observer, 300.0);
        assert!(laplace(&good, MU).is_ok(), "the control case must solve");

        // Validation is `gauss`'s, shared: assert Laplace goes through it rather than around it.
        let mut short = good;
        short[0].line_of_sight = short[0].line_of_sight.scale(1.02);
        assert_eq!(
            laplace(&short, MU),
            Err(IodError::MalformedSighting { index: 0 })
        );
        let mut backwards = good;
        backwards.swap(0, 2);
        assert_eq!(laplace(&backwards, MU), Err(IodError::NonMonotonicEpochs));

        // An observer inside the orbit plane: every line of sight, and therefore `L̇` and `L̈`, lies
        // in that plane and the triple product is zero to rounding.
        let normal = r0.cross(v0);
        let in_plane = normal.cross(r0);
        let coplanar_observer = in_plane.scale(4.0e6 / in_plane.norm());
        let (coplanar, _, _) = synthesise(r0, v0, coplanar_observer, 300.0);
        assert!(matches!(
            laplace(&coplanar, MU),
            Err(IodError::DegenerateGeometry { .. })
        ));

        // Three lines of sight pointing away from the object. The polynomial still has roots — the
        // algebra does not know which way a telescope was pointed — and every one of them puts the
        // object behind the observer, which is what the `ρ > 0` filter is for.
        let mut behind = good;
        for sighting in &mut behind {
            sighting.line_of_sight = sighting.line_of_sight.scale(-1.0);
        }
        assert_eq!(laplace(&behind, MU), Err(IodError::NoPhysicalRange));
    }

    #[test]
    fn gibbs_refuses_positions_that_are_not_one_orbit() {
        let (r0, v0, _) = reference();
        let (r1, _) = kepler::propagate(r0, v0, MU, -600.0);
        let (r3, _) = kepler::propagate(r0, v0, MU, 600.0);
        // Tip the middle position a long way out of the plane.
        let normal = r0.cross(v0);
        let tilted = r0 + normal.scale(0.3 * r0.norm() / normal.norm());
        assert!(matches!(
            gibbs(tilted, r1, r3, MU),
            Err(IodError::NotCoplanar { .. })
        ));
        assert_eq!(
            gibbs(Vec3::new(0.0, 0.0, 0.0), r1, r3, MU),
            Err(IodError::Degenerate)
        );
    }

    /// Three collinear positions, none of them at the origin and none of them coincident.
    ///
    /// This is the only geometry that reaches the `D` check. Zeroing a position vector does not: it
    /// zeroes `r₂ × r₃` too, and the plane check above returns the same `Degenerate` first. Three
    /// points on a line off the origin have a perfectly good `r₂ × r₃`, `r₁` lies in that plane so
    /// the coplanarity screen passes, and `D = r₂×r₃ + r₃×r₁ + r₁×r₂` is exactly zero because it is
    /// twice the triangle's area. `N` is the same three cross products weighted by the three
    /// *different* radii, so it is **not** zero — which is what makes `D` and `N` distinguishable
    /// here and nowhere else. Without the check the velocity comes back `√(μ/0) = ∞`.
    #[test]
    fn gibbs_refuses_three_points_on_a_line() {
        let (r1, r2, r3) = (
            Vec3::new(7.0e6, 0.0, 0.0),
            Vec3::new(7.0e6, 1.0e6, 0.0),
            Vec3::new(7.0e6, 2.0e6, 0.0),
        );
        assert_eq!(gibbs(r1, r2, r3, MU), Err(IodError::Degenerate));
        // The premise, stated so a future reader does not have to re-derive it: `D` vanishes and `N`
        // does not.
        let d = r2.cross(r3) + r3.cross(r1) + r1.cross(r2);
        let n = r2.cross(r3).scale(r1.norm())
            + r3.cross(r1).scale(r2.norm())
            + r1.cross(r2).scale(r3.norm());
        assert_eq!(d.norm(), 0.0);
        assert!(n.norm() > 0.0);
    }

    /// The coplanarity screen at the value it screens on, from both sides.
    ///
    /// `sine` divides by `|r₁|·|r₂ × r₃|`, and the two factors differ by six orders of magnitude —
    /// 7e6 against 3e13 — so replacing the product with a sum leaves the *smaller* factor out and
    /// inflates every sine by about 7e6. That is invisible from either end of the existing coverage:
    /// an exactly coplanar triple has `sine ≈ 1e-16` and stays under the screen even multiplied by
    /// 7e6, and the 0.3-radian tilt is over it either way. Only a triple whose sine sits just inside
    /// the threshold separates them.
    #[test]
    fn the_coplanarity_screen_is_measured_from_both_sides() {
        let (r0, v0, _) = reference();
        let (r1, _) = kepler::propagate(r0, v0, MU, -600.0);
        let (r3, _) = kepler::propagate(r0, v0, MU, 600.0);
        let normal = r1.cross(r3);
        let unit = normal.scale(1.0 / normal.norm());
        // `sin` of the out-of-plane angle is the tilt itself for a unit-normal displacement of this
        // size, to the second order the screen does not care about.
        for (tilt, refused) in [(0.04_f64, false), (0.08, true)] {
            let tilted = (r0 + unit.scale(tilt * r0.norm())).scale(1.0);
            let got = gibbs(tilted, r1, r3, MU);
            assert_eq!(
                matches!(got, Err(IodError::NotCoplanar { .. })),
                refused,
                "tilt {tilt}: {got:?}"
            );
        }

        // The threshold itself, exactly, which is the one argument at which `>` and `>=` part
        // company. With `r₂ = x̂` and `r₃ = ŷ` the cross product is `ẑ` and its norm is exactly 1,
        // `Vec3::dot` fuses so `r₁ · ẑ` is exactly `r₁.z`, and the `x` below is the double that makes
        // `|r₁|` exactly 1 — so `sine` is bit-exactly `0.0603` with no rounding anywhere in it. Gibbs
        // screens on `> 0.0603`, so its own threshold is inside the domain, not outside.
        let (unit_x, unit_y) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let at = Vec3::new(0.998_180_299_344_762_7, 0.0, 0.0603);
        assert_eq!(at.norm(), 1.0, "the construction needs an exact unit norm");
        assert_eq!(at.dot(unit_x.cross(unit_y)), 0.0603, "and an exact sine");
        assert!(gibbs(at, unit_x, unit_y, MU).is_ok(), "0.0603 is admitted");
        let over = Vec3::new(at.x, 0.0, f64::from_bits(0.0603_f64.to_bits() + 1));
        assert!(
            matches!(
                gibbs(over, unit_x, unit_y, MU),
                Err(IodError::NotCoplanar { .. })
            ),
            "one double above it is not"
        );
    }

    #[test]
    fn herrick_gibbs_refuses_epochs_that_do_not_advance() {
        let (r0, v0, _) = reference();
        let (r1, _) = kepler::propagate(r0, v0, MU, -10.0);
        let (r3, _) = kepler::propagate(r0, v0, MU, 10.0);
        assert_eq!(
            herrick_gibbs(r1, r0, r3, 0.0, 0.0, 10.0, MU),
            Err(IodError::NonMonotonicEpochs)
        );
        // The *second* comparison, which the pair above cannot reach: the first two epochs advance
        // and the last two do not.
        assert_eq!(
            herrick_gibbs(r1, r0, r3, -10.0, 0.0, 0.0, MU),
            Err(IodError::NonMonotonicEpochs)
        );
        // Each position asked on its own. Unlike `gibbs`, nothing downstream here catches a zero
        // radius — `μ/(12|rᵢ|³)` is `+∞` and the velocity comes back infinite — so all three
        // disjuncts of the guard have to be reachable separately.
        let zero = Vec3::new(0.0, 0.0, 0.0);
        for (a, b, c) in [(zero, r0, r3), (r1, zero, r3), (r1, r0, zero)] {
            assert_eq!(
                herrick_gibbs(a, b, c, -10.0, 0.0, 10.0, MU),
                Err(IodError::Degenerate)
            );
        }
    }

    /// `validate`'s first comparison, which the existing triple reaches only through its second.
    #[test]
    fn two_sightings_at_the_same_instant_are_refused_whichever_pair_they_are() {
        let (r0, v0, observer) = reference();
        let (good, _, _) = synthesise(r0, v0, observer, 300.0);
        let mut first_pair = good;
        first_pair[0].time = first_pair[1].time;
        assert_eq!(gauss(&first_pair, MU), Err(IodError::NonMonotonicEpochs));
    }

    /// Gooding from seeds it was not given by [`gauss`], including ones Newton cannot walk from.
    ///
    /// The documented pairing hands it a seed already within a kilometre or two of the answer, and
    /// from there the iteration never proposes a step that would put an object behind its observer —
    /// so the damping loop, which exists for exactly that, is never entered. A caller with a radar
    /// range or a catalogue prior is invited by the signature to pass something else, and "something
    /// else" is where the three failure paths live: a seed six times too small still converges
    /// because the damping halves the step instead of stepping through zero, a seed forty times too
    /// small does not converge at all and says so with the residual it reached, and a seed at 1e9 m
    /// puts the trial positions where no Lambert arc connects them.
    ///
    /// Both outcomes are pinned **bit for bit** alongside the band they are checked against, for the
    /// same reason the ranges are in `the_range_solution_is_measured_against_the_geometry_it_came_from`:
    /// Newton's converged root is a property of the residual and not of the path taken to it, so a
    /// tolerance sees nothing when the damping schedule, the basis the miss is measured in, or the
    /// singular-Jacobian escape changes underneath it. The stalled residuals are pinned for the same
    /// reason and are the only thing that watches the 60-iteration path at all. Both are functions of
    /// `lambert`, so a deliberate change there moves them — that is what a regression pin is for.
    #[test]
    fn gooding_survives_seeds_that_are_not_gausss() {
        let (r0, v0, observer) = reference();
        let (sightings, truth_r, truth_v) = synthesise(r0, v0, observer, 600.0);
        // The true first range is 6.209e6 m; the seeds below span 1e4 to 1e9.
        for (seed, pinned) in [
            (
                (1.0e6_f64, 1.0e6),
                [
                    7.078_137_000_000_139e6,
                    -2.668_821_252_882_480_6e-7,
                    -9.196_810_424_327_85e-8,
                    -7.275_957_614_183_426e-12,
                    -1.070_328_380_348_894_4e3,
                    7.427_564_398_137_589e3,
                ],
            ),
            (
                (2.0e7, 2.0e7),
                [
                    7.078_136_999_999_934e6,
                    1.262_524_165_213_108e-7,
                    4.400_499_165_058_136e-8,
                    4.547_473_508_864_641e-13,
                    -1.070_328_380_348_828_2e3,
                    7.427_564_398_136_982e3,
                ],
            ),
            (
                (1.0e5, 1.0e7),
                [
                    7.078_137_000_000_1e6,
                    -2.011_074_684_560_299e-7,
                    6.752_088_665_962_219e-9,
                    2.723_936_631_809_92e-10,
                    -1.070_328_380_349_368_4e3,
                    7.427_564_398_137_309e3,
                ],
            ),
        ] {
            let (r, v) = gooding(&sightings, MU, false, seed).expect("converged");
            assert!(
                (r - truth_r).norm() < 1e-5 && (v - truth_v).norm() < 1e-8,
                "seed {seed:?}: {:e} m, {:e} m/s",
                (r - truth_r).norm(),
                (v - truth_v).norm()
            );
            assert_eq!(
                [r.x, r.y, r.z, v.x, v.y, v.z],
                pinned,
                "seed {seed:?} pinned: {:.16e} {:.16e} {:.16e} {:.16e} {:.16e} {:.16e}",
                r.x,
                r.y,
                r.z,
                v.x,
                v.y,
                v.z
            );
        }
        // A residual is reported, and it is the angular miss in radians rather than a placeholder:
        // both of these stall around 0.96 rad, three quarters of a right angle away from the
        // observed direction.
        for (seed, pinned) in [
            ((1.0e4_f64, 1.0e4), 9.572_028_909_276_205e-1),
            ((1.0e5, 1.0e5), 9.601_328_172_723_819e-1),
            ((1.0e7, 1.0e5), 7.858_533_458_378_832e-1),
        ] {
            match gooding(&sightings, MU, false, seed) {
                Err(IodError::DidNotConverge { residual }) => {
                    assert!(
                        (0.5..core::f64::consts::PI).contains(&residual),
                        "seed {seed:?}: residual {residual:e}"
                    );
                    assert_eq!(residual, pinned, "seed {seed:?} pinned: {residual:.16e}");
                }
                other => panic!("seed {seed:?}: {other:?}"),
            }
        }
        assert_eq!(
            gooding(&sightings, MU, false, (1.0e9, 1.0e9)),
            Err(IodError::LambertFailed)
        );
    }

    /// Every refusal, printed.
    ///
    /// `Display` is the only route these numbers take to a human — `Debug` is what a test reads and
    /// what an operator never sees — and each message exists to carry the one value that tells a
    /// caller what to change. Asserting on the value, not just on the variant, is what makes the
    /// message a measurement rather than a label.
    #[test]
    fn every_refusal_prints_the_value_that_explains_it() {
        for (error, fragment) in [
            (IodError::MalformedSighting { index: 2 }, "sighting 2"),
            (IodError::NonMonotonicEpochs, "strictly increasing"),
            (
                IodError::DegenerateGeometry {
                    triple_product: 1.5e-13,
                },
                "1.5e-13",
            ),
            (IodError::NoPhysicalRange, "in front of all three"),
            (IodError::NotCoplanar { sine: 0.25 }, "2.5e-1"),
            (IodError::Degenerate, "collinear or coincident"),
            (IodError::LambertFailed, "no Lambert arc"),
            (IodError::DidNotConverge { residual: 3e-4 }, "3e-4"),
        ] {
            let printed = format!("{error}");
            assert!(printed.contains(fragment), "{error:?} printed {printed:?}");
        }
    }

    /// The reference station carried round by the Earth to `t`, about the pole.
    ///
    /// A ground station is not in the same inertial place at three epochs, and [`Sighting`] carries
    /// `observer` per sighting because of it. Holding it fixed is not merely unrealistic — it makes
    /// Gauss's `A` **identically zero**. With `R₁ = R₂ = R₃ = R` every `Dᵢⱼ` collapses to `R · pⱼ`,
    /// so `A = R·p₂·(−τ₃ + τ + τ₁)/(τ·D₀)`, and `τ − τ₃ + τ₁ = (t₃−t₁) − (t₃−t₂) + (t₁−t₂) = 0`
    /// exactly. `A` is then zero to the last bit, `A²` and `2A` are both zero, and the `C₆`
    /// coefficient of the range polynomial cannot tell them apart. 7.292115e-5 rad/s is the IERS 2010
    /// mean rotation rate; over the 1 200 s triple the station moves 527 km.
    fn station(base: Vec3, t: f64) -> Vec3 {
        let (sin, cos) = (libm::sin(7.292_115e-5 * t), libm::cos(7.292_115e-5 * t));
        Vec3::new(
            cos * base.x - sin * base.y,
            sin * base.x + cos * base.y,
            base.z,
        )
    }

    /// The ranges, against the geometry they came from.
    ///
    /// Everything else measures `gauss_ranges` through the state built on top of it, and that is a
    /// weak instrument: Gauss's own `f`/`g` truncation is metres to kilometres wide and swallows a
    /// range that is off by less than that, while [`gooding`] re-converges from any seed at all and
    /// reports the same answer whatever it was handed. The ranges themselves have a truth to be
    /// compared against — `|rᵢ − Rᵢ|` from the states the sightings were built from — and the
    /// polynomial, its scan and its physical filter are observable in nothing else.
    ///
    /// Two assertions, because one of them cannot do the other's job. The comparison against the
    /// geometry says the method is *right*, and its band is set by Gauss's own truncation: 1.709e-6
    /// relative on the 5.8 s triple and 2.833e-3 on the 240 s one. The pinned values say the
    /// arithmetic is *unchanged*, and they are compared **bit for bit**, not to a tolerance. A
    /// tolerance is the wrong instrument here whatever its size: the leading term of `ρ₁` cancels
    /// `r₂³` between numerator and denominator, so `r₂` reaches the ranges only through corrections
    /// of order `μτ²/r₂³` — 1e-5 on the 5.8 s triple. Everything that decides *which* `r₂` the scan
    /// arrives at, and by what path, is therefore attenuated by five orders of magnitude before it
    /// shows up in a range, and a band loose enough to be safe is loose enough to hide the whole
    /// search. Both quantities are deterministic f64: the same `libm` calls in the same order.
    #[test]
    fn the_range_solution_is_measured_against_the_geometry_it_came_from() {
        let (r0, v0, base) = reference();
        // `moving` says whether the station is carried round by the Earth between the three epochs.
        // It is the only case in which Gauss's `A` is not identically zero; see `station`.
        for (times, moving, bound, want) in [
            (
                [-2.9_f64, 0.0, 2.9],
                false,
                1.8e-6,
                [
                    5.657_894_466_636_487e6,
                    5.654_628_024_623_747e6,
                    5.651_407_053_358_394e6,
                ],
            ),
            (
                [100.0, 160.0, 340.0],
                false,
                2.9e-3,
                [
                    5.554_756_281_036_205e6,
                    5.531_662_276_339_186e6,
                    5.588_422_272_889_798_5e6,
                ],
            ),
            (
                [-600.0, 0.0, 600.0],
                true,
                8.0e-2,
                [
                    6.487_628_503_121_218e6,
                    5.204_301_535_745_576e6,
                    5.796_952_985_793_832e6,
                ],
            ),
        ] {
            let states: [(Vec3, Vec3); 3] =
                core::array::from_fn(|i| kepler::propagate(r0, v0, MU, times[i]));
            let observers: [Vec3; 3] = core::array::from_fn(|i| {
                if moving {
                    station(base, times[i])
                } else {
                    base
                }
            });
            let sightings: [Sighting; 3] = core::array::from_fn(|i| {
                let line = states[i].0 - observers[i];
                Sighting {
                    time: times[i],
                    line_of_sight: line.scale(1.0 / line.norm()),
                    observer: observers[i],
                }
            });
            let ranges = gauss_ranges(&sightings, MU).unwrap();
            let relative: [f64; 3] = core::array::from_fn(|i| {
                let truth = (states[i].0 - observers[i]).norm();
                libm::fabs(ranges[i] - truth) / truth
            });
            let worst = relative.iter().fold(0.0_f64, |a, &b| a.max(b));
            assert!(
                worst <= bound,
                "{times:?} against the geometry: {:e} {:e} {:e}",
                relative[0],
                relative[1],
                relative[2]
            );
            assert_eq!(
                ranges, want,
                "{times:?} pinned: {:.16e} {:.16e} {:.16e}",
                ranges[0], ranges[1], ranges[2]
            );
        }
    }

    /// The perpendicular basis has to stay conditioned for every axis, including the ones parallel
    /// to a coordinate direction, which is where a fixed seed vector fails.
    #[test]
    fn the_perpendicular_basis_is_orthonormal_for_every_axis() {
        for axis in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(
                0.577_350_269_189_625_8,
                0.577_350_269_189_625_8,
                0.577_350_269_189_625_8,
            ),
        ] {
            let (u, w) = perpendicular_basis(axis);
            assert!((u.norm() - 1.0).abs() < 1e-14, "{axis:?}");
            assert!((w.norm() - 1.0).abs() < 1e-14, "{axis:?}");
            assert!(u.dot(w).abs() < 1e-14, "{axis:?}");
            assert!(u.dot(axis).abs() < 1e-14, "{axis:?}");
            assert!(w.dot(axis).abs() < 1e-14, "{axis:?}");
        }
    }
}
