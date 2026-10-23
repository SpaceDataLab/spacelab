//! Whether [`conjunction`](crate::conjunction)'s 2-D model applies to the conjunction it is about to
//! be handed — the four checks that decide, and the correction factor that comes out of them.
//!
//! The 2-D probability rests on three assumptions: rectilinear relative motion, a covariance frozen
//! at TCA, spherical hard bodies. Nothing in that module notices when the first two fail. Across
//! 1.79 M real conjunctions screened by NASA's CARA programme they fail on **5.3 %** of cases, and
//! about a tenth of those move `Pc` by more than 30 %; the rest are the price of a detector tuned
//! for **no missed detections**, which it achieves at a ~90 % false-alarm rate (Hall, Baars and
//! Casali, *A Multistep Probability of Collision Computation Algorithm*, AAS/AIAA Astrodynamics
//! Specialist Conference, Big Sky MT, August 2023, NASA NTRS 20230010175).
//!
//! # The check is a comparison of two curves, not a second computation of `Pc`
//!
//! Under the rectilinear model the mean relative position is `r̄(δt) = r̄ + v̄ δt` with a covariance
//! `A` that does not move, so the squared Mahalanobis distance of the collision point from the
//! relative-position distribution is an exact parabola in time:
//!
//! ```text
//! M′(δt) = r̄(δt)ᵀ A⁻¹ r̄(δt) = a δt² + b δt + c
//! a = v̄ᵀA⁻¹v̄     b = 2 r̄ᵀA⁻¹v̄     c = r̄ᵀA⁻¹r̄
//! ```
//!
//! Its minimum, its 1-σ width in time and the duration that holds all but `γ` of the probability are
//! closed forms of `(a, b, c)` and cost three quadratic forms to obtain. Under two-body motion the
//! same quantity is
//!
//! ```text
//! M(t) = r̄(t)ᵀ A(t)⁻¹ r̄(t) + ln(|A(t)| / |A(TCA)|)
//! ```
//!
//! — the log term accounting for a covariance that no longer has the volume it had at TCA. The two
//! curves are *identically equal* when the motion really is rectilinear and the covariance really is
//! frozen, which is what makes their difference a measure of exactly the assumptions in question and
//! nothing else. Where they differ, they differ in four specific ways, and each has its own check.
//!
//! # What each check is asking
//!
//! * **NPD** — is every one of the nine eigenvalues of the primary, secondary and joint position
//!   covariances positive? A symmetric matrix is positive definite exactly when its Cholesky
//!   factorisation exists, so that is what this module computes: it is the same factorisation the
//!   Mahalanobis distance needs anyway, and an eigen-solve would answer a question nobody asked (*how*
//!   singular) at the price of a less reliable answer to the one that was.
//! * **Offset** — TCA is where `|r̄|` is smallest; `T′` is where the *Mahalanobis* distance is
//!   smallest, and the 2-D projection is taken at TCA. `V_o = |T′ − TCA| / p_min` above
//!   [`OFFSET_LIMIT`] says the projection is being taken at the wrong instant. At a LEO period of
//!   ~100 min that is an offset above ~60 s.
//! * **Extended** — the covariance can only be treated as frozen over an encounter short against an
//!   orbital period. `V_e = Δτ′ / p_min` above [`EXTENDED_LIMIT`]; at LEO, a duration above ~120 s.
//! * **Inaccuracy** — the two curves themselves. In the small-hard-body limit of the 3-D
//!   expected-collision-rate integral, `Pc ∝ v̆ · w · exp(−M_min/2) / (2√|A|)`, so the ratio of the
//!   curvilinear estimate to the rectilinear one is a difference of three logarithms,
//!
//!   ```text
//!   ln F_c = ln(v̆_T / v̆_TCA) + ln(w / w′) − (M_min − M′_min) / 2
//!   ```
//!
//!   and `V_i = 1 − exp(−|ln F_c|)` above [`INACCURACY_LIMIT`] is a violation. `F_c < 1` means the
//!   rectilinear estimate is too large, `F_c > 1` that it is too small.
//!
//! # What this is not
//!
//! The paper's curvilinear covariance is the *effective* one from Hall's iterative re-centring of the
//! linearisation on the peak-overlap point (*Expected Collision Rates for Tracked Satellites*, J.
//! Spacecraft and Rockets **58**(3), 2021); this module asks the caller for the plainly propagated
//! relative state and covariance instead. The re-centring exists to make the 3-D *probability*
//! accurate, while the detector only needs the two curves to part company where the trajectory bends,
//! and the plainly propagated covariance is the one every propagator can already supply. What that
//! substitution does to `F_c` is not measured here; the paper calls `F_c` a rough correction in
//! either case, and the flags — not the factor — are what this module is for.
//!
//! The escalation ladder itself (2-D `Nc`, 3-D `Nc`, simplified-dynamics and brute-force Monte
//! Carlo) is not here. This is the step that decides whether any of it is needed.

// The paper's own symbols are one letter (a, b, c, w, T, M, F_c, V_i) and the 3×3 loops index by i/j.
#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

use crate::conjunction::ConjunctionGeometry;

/// The fraction of the collision probability the conjunction bounds are permitted to leave outside
/// themselves, `γ = 10⁻⁶`, which is what fixes [`DURATION_WIDTHS`].
pub const DURATION_TAIL_FRACTION: f64 = 1.0e-6;

/// The conjunction duration in 1-σ widths of the Mahalanobis curve: `2√2 · erfc⁻¹(γ)` at
/// `γ =` [`DURATION_TAIL_FRACTION`], i.e. `≈ 9.78 w′`.
///
/// Computed in Maxima at 40 digits — `erfc⁻¹(10⁻⁶) = 3.458910737279500022150927635957569519438`,
/// residual `|erfc(x) − γ| = 3.4e-42` — rather than transcribed from the paper's "≈ 9.8".
#[allow(clippy::excessive_precision)]
pub const DURATION_WIDTHS: f64 = 9.783_276_951_397_180_772_462_244_919_914_047_240_392;

/// The value of `V_o` above which the offset between TCA and the rectilinear Mahalanobis minimum is
/// an offset violation. The paper's threshold, selected so that no actual usage violation went
/// undetected across its 1.79 M test conjunctions.
pub const OFFSET_LIMIT: f64 = 0.01;

/// The value of `V_e` above which the rectilinear encounter duration is an extended-duration
/// violation. Twice [`OFFSET_LIMIT`], and from the same calibration.
pub const EXTENDED_LIMIT: f64 = 0.02;

/// The value of `V_i` above which the rectilinear and curvilinear Mahalanobis curves have parted
/// company far enough to be an inaccuracy violation.
pub const INACCURACY_LIMIT: f64 = 0.02;

/// Doublings of the search window [`CurvilinearEncounter::locate`] will try before giving up on
/// finding the curvilinear minimum. Eight of them reach 256 rectilinear durations either side of the
/// rectilinear minimum: past that the two models have nothing in common and the honest answer is a
/// refusal, not a number found by searching until something turned up.
pub const MAX_WINDOW_EXPANSIONS: usize = 8;

/// Doublings of the trial half-width [`CurvilinearEncounter::locate`] will try before giving up on
/// bracketing the curve's 1-σ rise.
///
/// Far more than [`MAX_WINDOW_EXPANSIONS`], and for a reason: the window is about *where* the minimum
/// is and the width about *how large* it is, and the second ratio is the one that gets extreme. A
/// crossing conjunction at 10 km/s has `w′` of tens of milliseconds while a co-orbiting pair's
/// curvilinear curve takes days to rise by one — six or seven decades apart, and the co-orbiting case
/// is precisely what this module exists to flag. Reaching 2²⁴ means the curve is flat over seventeen
/// million rectilinear widths, which is a degenerate covariance rather than a long encounter.
pub const MAX_WIDTH_EXPANSIONS: usize = 24;

/// Intervals in each coarse scan of the curvilinear curve. The scan only has to isolate the minimum
/// into one interval for the golden section that follows, and every point costs the caller a
/// propagation.
const COARSE_INTERVALS: usize = 32;

/// Golden-section steps refining the coarse bracket. `0.618⁴⁸ ≈ 4e-11` of the bracket, and the loop
/// leaves early once the bracket stops shrinking in `f64`.
const GOLDEN_STEPS: usize = 48;

/// Bisections locating the 1-σ rise. `f64` runs out of interval before this does; the count is here
/// so that a wrong comparison inside the loop is a wrong answer rather than a hang.
const RISE_BISECTIONS: usize = 64;

/// The relative state and joint position covariance at one instant of a curvilinear trajectory, in
/// the same inertial frame as the [`ConjunctionGeometry`] it is compared against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativeSample {
    /// Relative position `r₂ − r₁`, metres.
    pub position: [f64; 3],
    /// Relative velocity `v₂ − v₁`, m/s.
    pub velocity: [f64; 3],
    /// Joint relative-position covariance `C₁ + C₂`, m².
    pub covariance: [[f64; 3]; 3],
}

/// The rectilinear Mahalanobis curve `M′(δt) = a δt² + b δt + c` of a conjunction, and the three
/// closed forms — minimum, width, duration — that summarise it.
///
/// Existing at all is the statement that the joint covariance at TCA is positive definite: it is
/// built by the Cholesky factorisation that every one of those quantities is computed through, so no
/// method here can meet a singular covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectilinearEncounter {
    quadratic: f64,
    linear: f64,
    constant: f64,
    speed: f64,
    non_positive_definite: bool,
}

impl RectilinearEncounter {
    /// Build the curve from a conjunction geometry at TCA.
    ///
    /// The primary and secondary covariances are factorised too, but only to answer the NPD check:
    /// an object covariance that is not positive definite is a fact about the orbit determination
    /// that produced it, and it does not stop the joint covariance from being invertible.
    ///
    /// # Errors
    ///
    /// * [`DomainError::NonFiniteState`] if any component of the relative position or velocity is
    ///   not finite — reported for what it is rather than left to surface as a degenerate velocity
    ///   three lines later.
    /// * [`DomainError::NonPositiveDefiniteCovariance`] if the *joint* covariance `C₁ + C₂` is not
    ///   positive definite. Unlike the per-object case this one is fatal: every quantity below is a
    ///   quadratic form in its inverse.
    /// * [`DomainError::DegenerateEncounterVelocity`] if the relative velocity is zero, which is the
    ///   only way `a = v̄ᵀA⁻¹v̄` can fail to be positive once `A` is known positive definite.
    pub fn from_geometry(geometry: &ConjunctionGeometry) -> Result<Self, DomainError> {
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        for component in position.iter().chain(velocity.iter()) {
            if !component.is_finite() {
                return Err(DomainError::NonFiniteState);
            }
        }

        let mut joint = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                joint[i][j] =
                    geometry.primary_covariance[i][j] + geometry.secondary_covariance[i][j];
            }
        }
        let factor = Cholesky3::factor(&joint).ok_or(DomainError::NonPositiveDefiniteCovariance)?;
        let non_positive_definite = Cholesky3::factor(&geometry.primary_covariance).is_none()
            || Cholesky3::factor(&geometry.secondary_covariance).is_none();

        let quadratic = factor.bilinear_inverse(&velocity, &velocity);
        if !quadratic.is_finite() || quadratic <= 0.0 {
            return Err(DomainError::DegenerateEncounterVelocity);
        }
        Ok(Self {
            quadratic,
            linear: 2.0 * factor.bilinear_inverse(&position, &velocity),
            constant: factor.bilinear_inverse(&position, &position),
            speed: norm3(&velocity),
            non_positive_definite,
        })
    }

    /// `M′(δt)`, the squared Mahalanobis distance `δt` seconds after TCA under rectilinear motion.
    #[must_use]
    pub fn modified_mahalanobis(&self, offset: f64) -> f64 {
        (self.quadratic * offset + self.linear) * offset + self.constant
    }

    /// `T′ − TCA`: seconds from TCA to the minimum of the curve, signed.
    #[must_use]
    pub fn time_of_minimum(&self) -> f64 {
        -self.linear / (2.0 * self.quadratic)
    }

    /// `M′_min`, the curve's minimum value.
    #[must_use]
    pub fn minimum(&self) -> f64 {
        self.constant - self.linear * self.linear / (4.0 * self.quadratic)
    }

    /// `w′ = a^{−1/2}`, the 1-σ width of the curve in time: the offset from `T′` at which `M′` has
    /// risen by exactly one.
    #[must_use]
    pub fn width(&self) -> f64 {
        1.0 / libm::sqrt(self.quadratic)
    }

    /// `Δτ′`, the conjunction duration: the span about `T′` outside which less than
    /// [`DURATION_TAIL_FRACTION`] of the probability lies.
    #[must_use]
    pub fn duration(&self) -> f64 {
        DURATION_WIDTHS * self.width()
    }

    /// `|v̄|` at TCA, m/s.
    #[must_use]
    pub const fn speed(&self) -> f64 {
        self.speed
    }

    /// Whether either object's own position covariance failed to be positive definite — the NPD
    /// usage violation. The joint covariance is not among the three, because a conjunction whose
    /// joint covariance is singular is a [`DomainError`], not a flag.
    #[must_use]
    pub const fn non_positive_definite(&self) -> bool {
        self.non_positive_definite
    }
}

/// The minimum of the curvilinear Mahalanobis curve: when it happens, how deep it is, how wide it is
/// in time, and how fast the objects are closing there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvilinearEncounter {
    time_of_minimum: f64,
    minimum: f64,
    width: f64,
    speed: f64,
}

impl CurvilinearEncounter {
    /// Locate the minimum of `M(t) = r̄(t)ᵀA(t)⁻¹r̄(t) + ln(|A(t)|/|A(0)|)` from a caller-supplied
    /// trajectory, where `sample(t)` is the relative state and joint covariance `t` seconds after
    /// TCA. `A(0)` is taken from the sampler itself, so the curve is zero-referenced against its own
    /// model rather than against the geometry's — which is what makes it equal to the rectilinear
    /// curve, term for term, when the sampler is rectilinear.
    ///
    /// The search is a coarse scan over the rectilinear duration, doubled until the minimum is
    /// interior, then a golden section; the width is then found as a level set — the offset at which
    /// the curve has risen by exactly one, averaged over the two sides — rather than from a second
    /// difference, because a level set is the definition of `w` in `M ≈ M_min + [(t−T)/w]²` and needs
    /// no step size chosen for it. It costs a few hundred evaluations of `sample`, against the tens
    /// of thousands of quadrature points a single 2-D `Nc` estimate would.
    ///
    /// # Errors
    ///
    /// * [`DomainError::NonPositiveDefiniteSample`] if a sampled covariance is not positive definite,
    ///   and [`DomainError::NonFiniteSample`] if a sampled curve value is not finite. Both end the
    ///   search where the trajectory model stopped being one, rather than searching on through NaN.
    /// * [`DomainError::CurvilinearMinimumNotBracketed`] if [`MAX_WINDOW_EXPANSIONS`] doublings of
    ///   the window never put the minimum inside it.
    /// * [`DomainError::CurvilinearWidthNotBracketed`] if [`MAX_WIDTH_EXPANSIONS`] doublings never
    ///   see the curve rise by one on some side of its minimum.
    /// * [`DomainError::DegenerateEncounterVelocity`] if the relative speed at the minimum is zero
    ///   or not finite, which would make `ln(v̆_T/v̆_TCA)` meaningless.
    pub fn locate<F>(rectilinear: &RectilinearEncounter, mut sample: F) -> Result<Self, DomainError>
    where
        F: FnMut(f64) -> RelativeSample,
    {
        let reference = Cholesky3::factor(&sample(0.0).covariance)
            .ok_or(DomainError::NonPositiveDefiniteSample)?;
        let log_determinant_at_tca = reference.log_determinant();

        let mut curve = |t: f64| -> Result<f64, DomainError> {
            let state = sample(t);
            let factor = Cholesky3::factor(&state.covariance)
                .ok_or(DomainError::NonPositiveDefiniteSample)?;
            let value = factor.bilinear_inverse(&state.position, &state.position)
                + factor.log_determinant()
                - log_determinant_at_tca;
            if value.is_finite() {
                Ok(value)
            } else {
                Err(DomainError::NonFiniteSample)
            }
        };

        let bracket = bracket_minimum(
            &mut curve,
            rectilinear.time_of_minimum(),
            rectilinear.duration(),
        )?;
        let (time_of_minimum, minimum) = golden_minimum(&mut curve, bracket)?;
        let trial = rectilinear.width();
        let forward = unit_rise(&mut curve, time_of_minimum, minimum, trial, 1.0)?;
        let backward = unit_rise(&mut curve, time_of_minimum, minimum, trial, -1.0)?;

        let speed = norm3(&sample(time_of_minimum).velocity);
        if !speed.is_finite() || speed <= 0.0 {
            return Err(DomainError::DegenerateEncounterVelocity);
        }
        Ok(Self {
            time_of_minimum,
            minimum,
            width: f64::midpoint(forward, backward),
            speed,
        })
    }

    /// `T − TCA`: seconds from TCA to the curvilinear minimum, signed.
    #[must_use]
    pub const fn time_of_minimum(&self) -> f64 {
        self.time_of_minimum
    }

    /// `M_min`, the curve's minimum value.
    #[must_use]
    pub const fn minimum(&self) -> f64 {
        self.minimum
    }

    /// `w`, the 1-σ width of the curve in time about its minimum.
    #[must_use]
    pub const fn width(&self) -> f64 {
        self.width
    }

    /// `|v̄_T|`, the relative speed at the curvilinear minimum, m/s.
    #[must_use]
    pub const fn speed(&self) -> f64 {
        self.speed
    }
}

/// The four 2-D `Pc` usage violations, the indicator each was decided on, and the correction factor
/// that falls out of the fourth.
///
/// A flag is a statement that the 2-D estimate is *not known to be* accurate, not that it is wrong:
/// the thresholds are deliberately conservative, and on the paper's test set roughly nine flags in
/// ten belong to conjunctions whose 2-D estimate was fine.
///
/// Three of the four flags are the indicators compared against their published thresholds, so they
/// are derived on demand rather than stored beside the numbers they come from — a flag and an
/// indicator held separately are two records of one fact, and two records of one fact can disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UsageViolations {
    non_positive_definite: bool,
    offset_indicator: f64,
    extended_indicator: f64,
    inaccuracy_indicator: f64,
    correction_factor: f64,
}

impl UsageViolations {
    /// Compare the two curves against each other and against the orbital period.
    ///
    /// `minimum_period` is the shorter of the two objects' orbital periods, in seconds; it is what
    /// turns an offset and a duration into the dimensionless ratios the thresholds are stated on.
    ///
    /// # Errors
    ///
    /// [`DomainError::InvalidOrbitalPeriod`] if `minimum_period` is not finite and positive. There
    /// is no default: a period guessed on the caller's behalf would silently rescale both of the
    /// thresholds this function exists to apply.
    pub fn assess(
        rectilinear: &RectilinearEncounter,
        curvilinear: &CurvilinearEncounter,
        minimum_period: f64,
    ) -> Result<Self, DomainError> {
        check_period(minimum_period)?;

        // The paper writes V_o as δT′/p_min, but δT′ is a signed offset and a bare threshold test on
        // it would be blind to every conjunction whose Mahalanobis minimum precedes TCA.
        let offset_indicator = libm::fabs(rectilinear.time_of_minimum()) / minimum_period;
        let extended_indicator = rectilinear.duration() / minimum_period;

        let log_correction = libm::log(curvilinear.speed() / rectilinear.speed())
            + libm::log(curvilinear.width() / rectilinear.width())
            - 0.5 * (curvilinear.minimum() - rectilinear.minimum());
        let inaccuracy_indicator = 1.0 - libm::exp(-libm::fabs(log_correction));

        Ok(Self {
            non_positive_definite: rectilinear.non_positive_definite(),
            offset_indicator,
            extended_indicator,
            inaccuracy_indicator,
            correction_factor: libm::exp(log_correction),
        })
    }

    /// Whether any of the four checks flagged.
    #[must_use]
    pub fn any(&self) -> bool {
        self.non_positive_definite || self.offset() || self.extended() || self.inaccuracy()
    }

    /// The NPD violation: one of the two object covariances is not positive definite.
    #[must_use]
    pub const fn non_positive_definite(&self) -> bool {
        self.non_positive_definite
    }

    /// The offset violation: TCA and the rectilinear Mahalanobis minimum are too far apart.
    #[must_use]
    pub fn offset(&self) -> bool {
        exceeds(self.offset_indicator, OFFSET_LIMIT)
    }

    /// The extended-duration violation: the encounter is too long against an orbital period.
    #[must_use]
    pub fn extended(&self) -> bool {
        exceeds(self.extended_indicator, EXTENDED_LIMIT)
    }

    /// The inaccuracy violation: the rectilinear and curvilinear curves differ too much.
    #[must_use]
    pub fn inaccuracy(&self) -> bool {
        exceeds(self.inaccuracy_indicator, INACCURACY_LIMIT)
    }

    /// `V_o`, the offset from TCA to the rectilinear minimum in orbital periods.
    #[must_use]
    pub const fn offset_indicator(&self) -> f64 {
        self.offset_indicator
    }

    /// `V_e`, the conjunction duration in orbital periods.
    #[must_use]
    pub const fn extended_indicator(&self) -> f64 {
        self.extended_indicator
    }

    /// `V_i = 1 − exp(−|ln F_c|)`, the two curves' disagreement folded into `[0, 1)`.
    #[must_use]
    pub const fn inaccuracy_indicator(&self) -> f64 {
        self.inaccuracy_indicator
    }

    /// `F_c`, the rough multiplicative correction to the 2-D estimate: `Pc ≈ F_c · Pc_2D`. Below one
    /// the rectilinear estimate is too large, above one too small.
    #[must_use]
    pub const fn correction_factor(&self) -> f64 {
        self.correction_factor
    }
}

/// Run all four checks on a conjunction: build the rectilinear curve from `geometry`, locate the
/// curvilinear one through `sample`, and compare both against `minimum_period` seconds.
///
/// `sample(t)` is the relative state and joint covariance `t` seconds after TCA, under whatever
/// dynamics the caller propagates with — two-body is what the paper assumes, and anything richer
/// only sharpens the comparison.
///
/// # Errors
///
/// Every error of [`RectilinearEncounter::from_geometry`], [`CurvilinearEncounter::locate`] and
/// [`UsageViolations::assess`]. The period is validated first, before the search spends a few
/// hundred propagations on a request that cannot be answered.
pub fn usage_violations<F>(
    geometry: &ConjunctionGeometry,
    minimum_period: f64,
    sample: F,
) -> Result<UsageViolations, DomainError>
where
    F: FnMut(f64) -> RelativeSample,
{
    check_period(minimum_period)?;
    let rectilinear = RectilinearEncounter::from_geometry(geometry)?;
    let curvilinear = CurvilinearEncounter::locate(&rectilinear, sample)?;
    UsageViolations::assess(&rectilinear, &curvilinear, minimum_period)
}

/// The paper states all three graded checks the same way — a violation is declared when the
/// indicator *exceeds* its threshold, so the threshold itself is not one. Written once rather than
/// three times, because the boundary convention is the part that can drift silently and only one of
/// the three indicators can be driven onto its threshold exactly: `V_o` and `V_e` are ratios a test
/// can construct, while `V_i = 1 − exp(−|ln F_c|)` steps by about two ulps of its own value for each
/// ulp of the ratio behind it, and skips the one output that would land on the limit.
fn exceeds(indicator: f64, limit: f64) -> bool {
    indicator > limit
}

/// Why a conjunction could not be checked against the 2-D model's domain. Each variant names the
/// input that was wrong, because "no violation" is the answer a caller acts on by doing nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainError {
    /// A component of the relative position or velocity at TCA is not finite.
    NonFiniteState,
    /// The joint covariance `C₁ + C₂` at TCA is not positive definite, so no Mahalanobis distance is
    /// defined and none of the four checks has a meaning.
    NonPositiveDefiniteCovariance,
    /// The relative velocity at TCA, or at the curvilinear minimum, is zero or not finite.
    DegenerateEncounterVelocity,
    /// A covariance returned by the trajectory sampler is not positive definite.
    NonPositiveDefiniteSample,
    /// The trajectory sampler produced a curve value that is not finite.
    NonFiniteSample,
    /// The minimum orbital period is not finite and positive.
    InvalidOrbitalPeriod,
    /// [`MAX_WINDOW_EXPANSIONS`] doublings of the search window did not put the curvilinear minimum
    /// inside it.
    CurvilinearMinimumNotBracketed,
    /// [`MAX_WIDTH_EXPANSIONS`] doublings did not find the offset at which the curvilinear curve
    /// rises by one above its minimum.
    CurvilinearWidthNotBracketed,
}

impl core::fmt::Display for DomainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteState => {
                write!(f, "relative position or velocity at TCA is not finite")
            }
            Self::NonPositiveDefiniteCovariance => write!(
                f,
                "joint position covariance at TCA is not positive definite; no Mahalanobis distance is defined"
            ),
            Self::DegenerateEncounterVelocity => {
                write!(f, "relative velocity is zero or not finite")
            }
            Self::NonPositiveDefiniteSample => {
                write!(f, "sampled joint covariance is not positive definite")
            }
            Self::NonFiniteSample => {
                write!(
                    f,
                    "sampled trajectory gave a non-finite Mahalanobis distance"
                )
            }
            Self::InvalidOrbitalPeriod => {
                write!(f, "minimum orbital period is not finite and positive")
            }
            Self::CurvilinearMinimumNotBracketed => write!(
                f,
                "curvilinear Mahalanobis minimum lies outside every window the search opened"
            ),
            Self::CurvilinearWidthNotBracketed => write!(
                f,
                "curvilinear Mahalanobis curve does not rise by one within the widths searched"
            ),
        }
    }
}

impl core::error::Error for DomainError {}

/// Reject a period that cannot scale a threshold, in one place, so both entry points reject the same
/// set of values.
fn check_period(minimum_period: f64) -> Result<(), DomainError> {
    if !minimum_period.is_finite() || minimum_period <= 0.0 {
        return Err(DomainError::InvalidOrbitalPeriod);
    }
    Ok(())
}

/// Scan `[seed − half, seed + half]` for the coarse minimum, doubling `half` until that minimum is
/// interior, and return the two neighbouring nodes as a bracket.
fn bracket_minimum<F>(curve: &mut F, seed: f64, duration: f64) -> Result<(f64, f64), DomainError>
where
    F: FnMut(f64) -> Result<f64, DomainError>,
{
    let mut half = duration;
    for _ in 0..=MAX_WINDOW_EXPANSIONS {
        let step = 2.0 * half / index_as_f64(COARSE_INTERVALS);
        let node = |i: usize| seed - half + step * index_as_f64(i);
        let mut best = 0usize;
        let mut best_value = f64::INFINITY;
        for i in 0..=COARSE_INTERVALS {
            let value = curve(node(i))?;
            if value < best_value {
                best_value = value;
                best = i;
            }
        }
        if best > 0 && best < COARSE_INTERVALS {
            return Ok((node(best - 1), node(best + 1)));
        }
        half *= 2.0;
    }
    Err(DomainError::CurvilinearMinimumNotBracketed)
}

/// Golden-section minimisation on a bracket, returning the location and value of the minimum.
fn golden_minimum<F>(curve: &mut F, bracket: (f64, f64)) -> Result<(f64, f64), DomainError>
where
    F: FnMut(f64) -> Result<f64, DomainError>,
{
    let (mut lo, mut hi) = bracket;
    let span = hi - lo;
    let ratio = 0.5 * (libm::sqrt(5.0) - 1.0);
    let mut left = hi - ratio * span;
    let mut right = lo + ratio * span;
    let mut f_left = curve(left)?;
    let mut f_right = curve(right)?;
    for _ in 0..GOLDEN_STEPS {
        if f_left < f_right {
            hi = right;
            right = left;
            f_right = f_left;
            left = hi - ratio * (hi - lo);
            f_left = curve(left)?;
        } else {
            lo = left;
            left = right;
            f_left = f_right;
            right = lo + ratio * (hi - lo);
            f_right = curve(right)?;
        }
        if hi - lo <= f64::EPSILON * span {
            break;
        }
    }
    if f_left < f_right {
        Ok((left, f_left))
    } else {
        Ok((right, f_right))
    }
}

/// The offset from `centre` in the given `direction` at which the curve has risen exactly one above
/// `minimum` — the 1-σ width on that side. `seed` starts the expansion at the rectilinear width, the
/// only scale available before the curvilinear one is known.
fn unit_rise<F>(
    curve: &mut F,
    centre: f64,
    minimum: f64,
    seed: f64,
    direction: f64,
) -> Result<f64, DomainError>
where
    F: FnMut(f64) -> Result<f64, DomainError>,
{
    let mut hi = seed;
    for expansion in 0..=MAX_WIDTH_EXPANSIONS {
        if curve(centre + direction * hi)? - minimum >= 1.0 {
            break;
        }
        if expansion == MAX_WIDTH_EXPANSIONS {
            return Err(DomainError::CurvilinearWidthNotBracketed);
        }
        hi *= 2.0;
    }
    let mut lo = 0.0;
    for _ in 0..RISE_BISECTIONS {
        let mid = f64::midpoint(lo, hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if curve(centre + direction * mid)? - minimum < 1.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(f64::midpoint(lo, hi))
}

/// Lower Cholesky factor `L` of a symmetric positive-definite 3×3, or `None` when no such factor
/// exists — which is the same statement as "some eigenvalue is not positive", reached without an
/// eigen-solve.
#[derive(Debug, Clone, Copy)]
struct Cholesky3 {
    l: [[f64; 3]; 3],
}

impl Cholesky3 {
    /// Factorise, symmetrising the input first: any asymmetry in a covariance is rounding.
    fn factor(m: &[[f64; 3]; 3]) -> Option<Self> {
        let mut l = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..=i {
                let mut sum = f64::midpoint(m[i][j], m[j][i]);
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= 0.0 {
                        return None;
                    }
                    l[i][i] = libm::sqrt(sum);
                } else {
                    l[i][j] = sum / l[j][j];
                    if !l[i][j].is_finite() {
                        return None;
                    }
                }
            }
        }
        Some(Self { l })
    }

    /// `L⁻¹b` by forward substitution.
    fn forward(&self, b: &[f64; 3]) -> [f64; 3] {
        let mut x = [0.0_f64; 3];
        for i in 0..3 {
            let mut sum = b[i];
            for k in 0..i {
                sum -= self.l[i][k] * x[k];
            }
            x[i] = sum / self.l[i][i];
        }
        x
    }

    /// `uᵀA⁻¹v`, as `(L⁻¹u)·(L⁻¹v)` — one triangular solve each, and never the explicit inverse.
    fn bilinear_inverse(&self, u: &[f64; 3], v: &[f64; 3]) -> f64 {
        let a = self.forward(u);
        let b = self.forward(v);
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// `ln|A| = 2 Σ ln Lᵢᵢ`, in logarithms throughout: a position covariance in m² has a determinant
    /// near `10¹⁸`, and the curve only ever wants the log of a ratio of two of them.
    fn log_determinant(&self) -> f64 {
        2.0 * (libm::log(self.l[0][0]) + libm::log(self.l[1][1]) + libm::log(self.l[2][2]))
    }
}

/// Euclidean norm of a 3-vector.
fn norm3(v: &[f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// `usize → f64` for scan indices, which are small counts and exactly representable.
#[allow(clippy::cast_precision_loss)]
fn index_as_f64(n: usize) -> f64 {
    n as f64
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]
    use super::*;
    use proptest::prelude::*;

    /// `A = L Lᵀ` from the six entries of a lower-triangular `L` with positive diagonal — every such
    /// matrix is symmetric positive definite, and nothing else is needed to make one.
    fn spd(l: [f64; 6]) -> [[f64; 3]; 3] {
        let lower = [[l[0], 0.0, 0.0], [l[1], l[2], 0.0], [l[3], l[4], l[5]]];
        let mut a = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    a[i][j] += lower[i][k] * lower[j][k];
                }
            }
        }
        a
    }

    /// `A⁻¹` by the adjugate: a different route to the inverse from the triangular solves the module
    /// uses, so agreeing with it is evidence rather than a tautology.
    fn inverse3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let cofactor = |i: usize, j: usize| {
            let rows: [usize; 2] = [(i + 1) % 3, (i + 2) % 3];
            let cols: [usize; 2] = [(j + 1) % 3, (j + 2) % 3];
            m[rows[0]][cols[0]] * m[rows[1]][cols[1]] - m[rows[0]][cols[1]] * m[rows[1]][cols[0]]
        };
        let det = m[0][0] * cofactor(0, 0) + m[0][1] * cofactor(0, 1) + m[0][2] * cofactor(0, 2);
        let mut inverse = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                inverse[i][j] = cofactor(j, i) / det;
            }
        }
        inverse
    }

    fn quadratic_form(inverse: &[[f64; 3]; 3], u: &[f64; 3]) -> f64 {
        let mut acc = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                acc += u[i] * inverse[i][j] * u[j];
            }
        }
        acc
    }

    fn sum3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut c = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        c
    }

    fn drift(origin: [f64; 3], velocity: [f64; 3], t: f64) -> [f64; 3] {
        [
            origin[0] + velocity[0] * t,
            origin[1] + velocity[1] * t,
            origin[2] + velocity[2] * t,
        ]
    }

    /// A head-on LEO conjunction: a few hundred metres of miss, ~14 km/s of closing speed, two
    /// correlated position covariances of a few hundred metres.
    fn head_on() -> ConjunctionGeometry {
        ConjunctionGeometry::new(
            [120.0, -45.0, 30.0],
            [-13_900.0, 900.0, 250.0],
            spd([180.0, 40.0, 90.0, -25.0, 15.0, 60.0]),
            spd([260.0, -70.0, 140.0, 30.0, -20.0, 110.0]),
            3.0,
            1.5,
        )
    }

    /// The sampler that reproduces the rectilinear model exactly: constant velocity, frozen
    /// covariance. Everything the curvilinear machinery finds on it must equal the closed form.
    fn frozen(geometry: &ConjunctionGeometry) -> impl FnMut(f64) -> RelativeSample + use<> {
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        move |t| RelativeSample {
            position: drift(position, velocity, t),
            velocity,
            covariance,
        }
    }

    #[test]
    fn the_rectilinear_curve_is_the_mahalanobis_distance_it_claims_to_be() {
        let geometry = head_on();
        let curve = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let inverse = inverse3(&sum3(
            &geometry.primary_covariance,
            &geometry.secondary_covariance,
        ));
        let mut worst = 0.0_f64;
        for offset in [-60.0, -12.5, 0.0, 3.25, 40.0] {
            let direct = quadratic_form(
                &inverse,
                &drift(
                    geometry.relative_position,
                    geometry.relative_velocity,
                    offset,
                ),
            );
            let carried = curve.modified_mahalanobis(offset);
            worst = worst.max((carried - direct).abs() / direct);
        }
        assert!(
            worst < 1e-12,
            "the quadratic and the direct form disagree by {worst:e}"
        );

        // The speed is only ever read as a ratio against another speed, and a ratio hides whatever a
        // shared error does to both sides. Pin the value itself.
        let v = geometry.relative_velocity;
        let expected = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        assert!((curve.speed() - expected).abs() < 1e-9 * expected);
    }

    #[test]
    fn a_determinant_that_grows_shifts_the_curve_by_the_log_of_the_volume_ratio() {
        // The log term is identically zero whenever the covariance is frozen — it is normalised
        // against the sampler's own `t = 0` — so a constant scaling cannot see it and neither can any
        // of the rectilinear tests. A determinant that moves can. With a fixed relative position and
        // `A(t) = A·e^{t/τ}`, the curve is exactly
        //
        //     M(s) = c·e^{−s} + 3s,   s = t/τ,   c = M′(0)
        //
        // whose minimum is at `s* = ln(c/3)` with value `3 + 3 ln(c/3)`, and whose unit rise solves
        // `3(e^{−x} + x − 1) = 1` either side. Every one of those numbers carries the 3 that the
        // log-determinant's own factor of two produces, so a mutation of it moves all of them.
        let isotropic = spd([100.0, 0.0, 100.0, 0.0, 0.0, 100.0]);
        let geometry = ConjunctionGeometry::new(
            [400.0, 0.0, 0.0],
            [0.0, 7_500.0, 0.0],
            isotropic,
            isotropic,
            1.0,
            1.0,
        );
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        assert_eq!(rectilinear.time_of_minimum(), 0.0, "the miss is transverse");
        let base = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let c = rectilinear.modified_mahalanobis(0.0);
        let tau = rectilinear.width();

        let curvilinear = CurvilinearEncounter::locate(&rectilinear, |t| {
            let factor = libm::exp(t / tau);
            let mut inflated = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    inflated[i][j] = base[i][j] * factor;
                }
            }
            RelativeSample {
                position: [400.0, 0.0, 0.0],
                velocity: geometry.relative_velocity,
                covariance: inflated,
            }
        })
        .unwrap();

        let expected_minimum = 3.0 + 3.0 * libm::log(c / 3.0);
        assert!(
            (curvilinear.minimum() - expected_minimum).abs() < 1e-9 * libm::fabs(expected_minimum),
            "M_min = {} against {expected_minimum}",
            curvilinear.minimum()
        );
        let expected_time = tau * libm::log(c / 3.0);
        assert!(
            (curvilinear.time_of_minimum() - expected_time).abs() < 1e-6 * tau,
            "T = {} against {expected_time}",
            curvilinear.time_of_minimum()
        );

        // `3(e^{−x} + x − 1) = 1`, one root either side of zero, by Newton from ±1.
        let rise = |x: f64| 3.0 * (libm::exp(-x) + x - 1.0) - 1.0;
        let slope = |x: f64| 3.0 * (1.0 - libm::exp(-x));
        let root = |mut x: f64| {
            for _ in 0..60 {
                x -= rise(x) / slope(x);
            }
            x
        };
        let expected_width = tau * 0.5 * (root(1.0) - root(-1.0));
        assert!(
            (curvilinear.width() - expected_width).abs() < 1e-7 * expected_width,
            "w = {} against {expected_width}",
            curvilinear.width()
        );

        // With the reported speed identical either side, `F_c` is the width ratio against half the
        // gap between the two minima — and this is the only case in the file where that gap is both
        // non-zero and known in closed form, so it is the only one that can see the sign it enters
        // with.
        let violations = UsageViolations::assess(&rectilinear, &curvilinear, 5_400.0).unwrap();
        let log_correction =
            libm::log(expected_width / tau) - 0.5 * (expected_minimum - rectilinear.minimum());
        let expected_factor = libm::exp(log_correction);
        assert!(
            (violations.correction_factor() - expected_factor).abs() < 1e-7 * expected_factor,
            "F_c = {} against {expected_factor}",
            violations.correction_factor()
        );
        let expected_indicator = 1.0 - libm::exp(-libm::fabs(log_correction));
        assert!(
            (violations.inaccuracy_indicator() - expected_indicator).abs() < 1e-7,
            "V_i = {} against {expected_indicator}",
            violations.inaccuracy_indicator()
        );
        assert!(violations.inaccuracy());
        assert!(
            violations.any(),
            "an inaccuracy violation on its own is a usage violation"
        );
    }

    #[test]
    fn the_speed_ratio_enters_the_correction_factor_and_nothing_else_does() {
        // `ln(v̆_T / v̆_TCA)` is the one term of `F_c` that no physical sampler in this file moves on
        // its own: a curve that bends changes the widths and the minima too. Holding the curve exactly
        // rectilinear and reporting a different velocity at the minimum is inconsistent as a
        // trajectory and exact as a test of the term — every other contribution is then zero, so `F_c`
        // is the ratio and nothing else.
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        let ratio = 2.0;
        let curvilinear = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift(position, velocity, t),
            velocity: [
                velocity[0] * ratio,
                velocity[1] * ratio,
                velocity[2] * ratio,
            ],
            covariance,
        })
        .unwrap();
        let violations = UsageViolations::assess(&rectilinear, &curvilinear, 5_400.0).unwrap();
        assert!(
            (violations.correction_factor() - ratio).abs() < 1e-9 * ratio,
            "F_c = {} where only the speed ratio {ratio} differs",
            violations.correction_factor()
        );
        // `V_i = 1 − exp(−|ln F_c|) = 1 − 1/F_c` above one, so a doubling is exactly a half.
        assert!(
            (violations.inaccuracy_indicator() - 0.5).abs() < 1e-9,
            "V_i = {} where the closed form is 1 − 1/{ratio}",
            violations.inaccuracy_indicator()
        );
    }

    #[test]
    fn the_closed_forms_summarise_the_curve_they_claim_to_summarise() {
        let curve = RectilinearEncounter::from_geometry(&head_on()).unwrap();
        let (peak, minimum, width) = (curve.time_of_minimum(), curve.minimum(), curve.width());
        assert!((curve.modified_mahalanobis(peak) - minimum).abs() < 1e-9 * minimum.abs());
        for step in [-4.0, -1.5, -0.25, 0.25, 1.5, 4.0] {
            assert!(
                curve.modified_mahalanobis(peak + step * width) > minimum,
                "the claimed minimum is not one, {step} widths away"
            );
        }
        // w′ is defined as the offset at which the curve has risen by exactly one.
        for side in [-1.0, 1.0] {
            let rise = curve.modified_mahalanobis(peak + side * width) - minimum;
            assert!((rise - 1.0).abs() < 1e-9, "rise at one width is {rise}");
        }
    }

    #[test]
    fn the_duration_leaves_exactly_the_named_tail_outside_it() {
        let curve = RectilinearEncounter::from_geometry(&head_on()).unwrap();
        // Δτ′ = 2√2 erfc⁻¹(γ) w′ ⇔ erfc(Δτ′ / (2√2 w′)) = γ.
        let argument = curve.duration() / (2.0 * libm::sqrt(2.0) * curve.width());
        let tail = libm::erfc(argument);
        assert!(
            (tail - DURATION_TAIL_FRACTION).abs() < 1e-12 * DURATION_TAIL_FRACTION,
            "the duration leaves {tail:e} outside it, not {DURATION_TAIL_FRACTION:e}"
        );
    }

    #[test]
    fn the_leo_thresholds_are_the_sixty_and_hundred_twenty_seconds_the_paper_names() {
        let period = 100.0 * 60.0;
        assert!((OFFSET_LIMIT * period - 60.0).abs() < 1e-12);
        assert!((EXTENDED_LIMIT * period - 120.0).abs() < 1e-12);
    }

    #[test]
    fn a_frozen_rectilinear_sampler_reproduces_the_closed_form_curve() {
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let curvilinear = CurvilinearEncounter::locate(&rectilinear, frozen(&geometry)).unwrap();

        let width = rectilinear.width();
        assert!(
            (curvilinear.time_of_minimum() - rectilinear.time_of_minimum()).abs() < 1e-6 * width
        );
        assert!((curvilinear.minimum() - rectilinear.minimum()).abs() < 1e-9);
        assert!((curvilinear.width() - width).abs() < 1e-9 * width);
        assert!((curvilinear.speed() - rectilinear.speed()).abs() < 1e-12 * rectilinear.speed());

        let violations = UsageViolations::assess(&rectilinear, &curvilinear, 5_400.0).unwrap();
        assert!(
            violations.inaccuracy_indicator() < 1e-9,
            "V_i = {} on a trajectory that is rectilinear by construction",
            violations.inaccuracy_indicator()
        );
        assert!((violations.correction_factor() - 1.0).abs() < 1e-9);
        assert!(!violations.inaccuracy());
    }

    #[test]
    fn a_bending_trajectory_is_flagged_where_the_straight_one_through_it_is_not() {
        // A 1 m/s conjunction: 100 m of positional uncertainty over 1 m/s of closing speed makes the
        // encounter ~100 s wide, and a milli-g of relative acceleration moves it by hundreds of metres
        // in that time — the low-velocity case the 2-D method is documented to lose.
        let geometry = ConjunctionGeometry::new(
            [50.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            spd([70.0, 0.0, 70.0, 0.0, 0.0, 70.0]),
            spd([70.0, 0.0, 70.0, 0.0, 0.0, 70.0]),
            2.0,
            2.0,
        );
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let straight = CurvilinearEncounter::locate(&rectilinear, frozen(&geometry)).unwrap();
        let unflagged = UsageViolations::assess(&rectilinear, &straight, 5_400.0).unwrap();
        assert!(!unflagged.inaccuracy());

        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let bent = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: [50.0 + 0.5e-3 * t * t, t, 0.0],
            velocity: [1.0e-3 * t, 1.0, 0.0],
            covariance,
        })
        .unwrap();
        let flagged = UsageViolations::assess(&rectilinear, &bent, 5_400.0).unwrap();
        assert!(
            flagged.inaccuracy(),
            "V_i = {} on a trajectory bending away from the primary",
            flagged.inaccuracy_indicator()
        );
        assert!(
            flagged.correction_factor() < 1.0,
            "F_c = {} where the curved trajectory misses by more than the straight one",
            flagged.correction_factor()
        );
    }

    #[test]
    fn a_trajectory_bending_towards_the_primary_corrects_the_estimate_upwards() {
        let geometry = ConjunctionGeometry::new(
            [50.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            spd([70.0, 0.0, 70.0, 0.0, 0.0, 70.0]),
            spd([70.0, 0.0, 70.0, 0.0, 0.0, 70.0]),
            2.0,
            2.0,
        );
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let bent = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: [50.0 - 0.5e-3 * t * t, t, 0.0],
            velocity: [-1.0e-3 * t, 1.0, 0.0],
            covariance,
        })
        .unwrap();
        let violations = UsageViolations::assess(&rectilinear, &bent, 5_400.0).unwrap();
        assert!(
            violations.correction_factor() > 1.0,
            "F_c = {} where the curved trajectory passes closer than the straight one",
            violations.correction_factor()
        );
        assert!(violations.inaccuracy());
    }

    #[test]
    fn a_covariance_that_grows_through_the_encounter_is_flagged_on_its_own() {
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let scale = rectilinear.width();
        // The mean trajectory is still exactly straight; only the covariance moves.
        let growing = CurvilinearEncounter::locate(&rectilinear, |t| {
            let factor = 1.0 + (t / scale) * (t / scale);
            let mut inflated = [[0.0_f64; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    inflated[i][j] = covariance[i][j] * factor;
                }
            }
            RelativeSample {
                position: drift(position, velocity, t),
                velocity,
                covariance: inflated,
            }
        })
        .unwrap();
        let violations = UsageViolations::assess(&rectilinear, &growing, 5_400.0).unwrap();
        assert!(
            violations.inaccuracy(),
            "V_i = {} on a covariance that triples in volume within one width",
            violations.inaccuracy_indicator()
        );
    }

    #[test]
    fn a_minimum_off_the_other_end_of_every_window_is_refused_too() {
        // The mirror of the test above, and not a copy of it. The coarse scan reports its result as
        // an index, and the interior test guards the two ends with two different comparisons; a
        // refusal driven only to the right leaves the lower half of that guard unexercised, and the
        // lower half is the one whose failure subtracts from an index of zero.
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let velocity = geometry.relative_velocity;
        let far = -1.0e6;
        let outcome = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift([0.0, 0.0, 0.0], velocity, t - far),
            velocity,
            covariance,
        });
        assert_eq!(outcome, Err(DomainError::CurvilinearMinimumNotBracketed));
    }

    /// A 1 m/s conjunction with 99 m of joint uncertainty: `w′` is 99 s and the first search window
    /// 968 s wide, both large enough that an additive corruption of the expansion cannot pass for a
    /// multiplicative one.
    fn slow_pair() -> ConjunctionGeometry {
        ConjunctionGeometry::new(
            [50.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            spd([70.0, 0.0, 70.0, 0.0, 0.0, 70.0]),
            spd([70.0, 0.0, 70.0, 0.0, 0.0, 70.0]),
            2.0,
            2.0,
        )
    }

    #[test]
    fn a_minimum_the_first_window_misses_is_found_by_the_second() {
        // Every other test here puts the curvilinear minimum inside the first window, so the
        // expansion is only ever exercised on its way to a refusal — which measures that it stops,
        // not that it works. A minimum between one duration and two is found on the second pass, and
        // *where* it sits in that window is the second half of the statement: the coarse scan
        // returns the nodes either side of its best index, and a bracket that drops one of them
        // still holds a minimum that fell on the other side of the node.
        let geometry = slow_pair();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        let width = rectilinear.width();
        let duration = rectilinear.duration();
        for offset in [1_500.0_f64, 1_530.0, 1_560.0] {
            assert!(
                offset > duration && offset < 2.0 * duration,
                "{offset} is not between one window and two"
            );
            let shifted = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
                position: drift(position, velocity, t - offset),
                velocity,
                covariance,
            })
            .unwrap();
            assert!(
                (shifted.time_of_minimum() - offset).abs() < 1e-6 * width,
                "T = {} against a curve translated to {offset}",
                shifted.time_of_minimum()
            );
            assert!((shifted.minimum() - rectilinear.minimum()).abs() < 1e-9);
            assert!((shifted.width() - width).abs() < 1e-9 * width);
        }
    }

    #[test]
    fn a_width_the_seed_does_not_reach_is_found_by_doubling() {
        // The width search starts at the rectilinear width, which is the right guess for the
        // conjunctions the 2-D method handles and the wrong one for the case this module exists to
        // flag. A curve twenty times wider needs five doublings before it has risen by one, and
        // nothing else here reaches a width by expanding — the flat-curve test drives the expansion
        // to its cap, which measures the refusal rather than the search.
        const STRETCH: f64 = 20.0;
        let geometry = slow_pair();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        let stretched = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift(position, velocity, t / STRETCH),
            velocity,
            covariance,
        })
        .unwrap();
        let expected = STRETCH * rectilinear.width();
        assert!(
            (stretched.width() - expected).abs() < 1e-9 * expected,
            "w = {} against {expected}",
            stretched.width()
        );
    }

    #[test]
    fn every_refusal_names_the_input_it_refused() {
        let messages = [
            (
                DomainError::NonFiniteState,
                "relative position or velocity at TCA is not finite",
            ),
            (
                DomainError::NonPositiveDefiniteCovariance,
                "joint position covariance at TCA is not positive definite; no Mahalanobis distance is defined",
            ),
            (
                DomainError::DegenerateEncounterVelocity,
                "relative velocity is zero or not finite",
            ),
            (
                DomainError::NonPositiveDefiniteSample,
                "sampled joint covariance is not positive definite",
            ),
            (
                DomainError::NonFiniteSample,
                "sampled trajectory gave a non-finite Mahalanobis distance",
            ),
            (
                DomainError::InvalidOrbitalPeriod,
                "minimum orbital period is not finite and positive",
            ),
            (
                DomainError::CurvilinearMinimumNotBracketed,
                "curvilinear Mahalanobis minimum lies outside every window the search opened",
            ),
            (
                DomainError::CurvilinearWidthNotBracketed,
                "curvilinear Mahalanobis curve does not rise by one within the widths searched",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(format!("{error}"), expected);
            let dynamic: &dyn core::error::Error = &error;
            assert_eq!(format!("{dynamic}"), expected);
        }
    }

    #[test]
    fn the_offset_check_stands_on_both_sides_of_its_threshold() {
        // With an isotropic joint covariance the Mahalanobis minimum is the geometric one, so sliding
        // the miss vector along the velocity by `d` seconds puts T′ exactly `d` seconds from TCA.
        let isotropic = spd([100.0, 0.0, 100.0, 0.0, 0.0, 100.0]);
        let velocity = [0.0, 7_500.0, 0.0];
        let period = 6_000.0;
        let slide = |d: f64| {
            let geometry = ConjunctionGeometry::new(
                [200.0, velocity[1] * d, 0.0],
                velocity,
                isotropic,
                isotropic,
                1.0,
                1.0,
            );
            let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
            let curvilinear =
                CurvilinearEncounter::locate(&rectilinear, frozen(&geometry)).unwrap();
            UsageViolations::assess(&rectilinear, &curvilinear, period).unwrap()
        };

        let at_threshold = slide(OFFSET_LIMIT * period);
        assert!(
            (at_threshold.offset_indicator() - OFFSET_LIMIT).abs() < 1e-12,
            "V_o = {} where the construction says {OFFSET_LIMIT}",
            at_threshold.offset_indicator()
        );
        assert!(
            !at_threshold.offset(),
            "the threshold itself is not above it"
        );
        assert!(
            !at_threshold.any(),
            "at this geometry the offset check is the only one that can flag, and it does not"
        );
        assert!(
            slide(-OFFSET_LIMIT * period - 1.0).offset(),
            "a minimum one second before TCA-minus-sixty is a violation, sign notwithstanding"
        );
        let over = slide(OFFSET_LIMIT * period + 1.0);
        assert!(over.offset());
        assert!(
            over.any(),
            "an offset violation on its own is a usage violation"
        );
    }

    #[test]
    fn the_extended_check_stands_on_both_sides_of_its_threshold() {
        // A transverse miss puts the Mahalanobis minimum exactly at TCA, so `V_o` is identically zero
        // and the extended check is the only one left that can flag. That isolation is the point: it
        // pins the threshold and the disjunction in `any()` against this one term at the same time.
        let isotropic = spd([100.0, 0.0, 100.0, 0.0, 0.0, 100.0]);
        let geometry = ConjunctionGeometry::new(
            [200.0, 0.0, 0.0],
            [0.0, 7_500.0, 0.0],
            isotropic,
            isotropic,
            1.0,
            1.0,
        );
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        assert_eq!(rectilinear.time_of_minimum(), 0.0, "the miss is transverse");
        let curvilinear = CurvilinearEncounter::locate(&rectilinear, frozen(&geometry)).unwrap();
        let duration = rectilinear.duration();
        let at_threshold =
            UsageViolations::assess(&rectilinear, &curvilinear, duration / EXTENDED_LIMIT).unwrap();
        assert!((at_threshold.extended_indicator() - EXTENDED_LIMIT).abs() < 1e-12);
        assert!(!at_threshold.extended());
        assert!(!at_threshold.any(), "the threshold itself flags nothing");
        let over = UsageViolations::assess(
            &rectilinear,
            &curvilinear,
            duration / (EXTENDED_LIMIT + 1e-3),
        )
        .unwrap();
        assert!(over.extended());
        assert_eq!(over.offset_indicator(), 0.0);
        assert!(
            over.any(),
            "an extended-duration violation on its own is a usage violation"
        );
    }

    #[test]
    fn an_object_covariance_that_is_not_positive_definite_is_a_flag_and_not_a_refusal() {
        let singular = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]];
        let geometry = ConjunctionGeometry::new(
            [120.0, -45.0, 30.0],
            [-7_100.0, 900.0, 250.0],
            singular,
            spd([180.0, 40.0, 90.0, -25.0, 15.0, 60.0]),
            3.0,
            1.5,
        );
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        assert!(rectilinear.non_positive_definite());
        let curvilinear = CurvilinearEncounter::locate(&rectilinear, frozen(&geometry)).unwrap();
        let violations = UsageViolations::assess(&rectilinear, &curvilinear, 5_400.0).unwrap();
        assert!(violations.non_positive_definite());
        assert!(violations.any());
    }

    #[test]
    fn a_joint_covariance_that_is_not_positive_definite_is_a_refusal() {
        let flat = [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]];
        let geometry =
            ConjunctionGeometry::new([1.0, 2.0, 3.0], [-7_100.0, 0.0, 0.0], flat, flat, 1.0, 1.0);
        assert_eq!(
            RectilinearEncounter::from_geometry(&geometry),
            Err(DomainError::NonPositiveDefiniteCovariance)
        );
    }

    #[test]
    fn a_zero_relative_velocity_has_no_curve_and_says_so() {
        let covariance = spd([100.0, 0.0, 100.0, 0.0, 0.0, 100.0]);
        let geometry = ConjunctionGeometry::new(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            covariance,
            covariance,
            1.0,
            1.0,
        );
        assert_eq!(
            RectilinearEncounter::from_geometry(&geometry),
            Err(DomainError::DegenerateEncounterVelocity)
        );
    }

    #[test]
    fn a_non_finite_relative_state_is_named_for_what_it_is() {
        let covariance = spd([100.0, 0.0, 100.0, 0.0, 0.0, 100.0]);
        for state in [
            ([f64::NAN, 2.0, 3.0], [-7_100.0, 0.0, 0.0]),
            ([1.0, 2.0, 3.0], [-7_100.0, f64::INFINITY, 0.0]),
        ] {
            let geometry =
                ConjunctionGeometry::new(state.0, state.1, covariance, covariance, 1.0, 1.0);
            assert_eq!(
                RectilinearEncounter::from_geometry(&geometry),
                Err(DomainError::NonFiniteState)
            );
        }
    }

    #[test]
    fn a_period_that_is_not_a_period_is_refused_before_a_single_propagation() {
        let geometry = head_on();
        for period in [0.0, -5_400.0, f64::NAN, f64::INFINITY] {
            let mut calls = 0usize;
            let outcome = usage_violations(&geometry, period, |t| {
                calls += 1;
                frozen(&geometry)(t)
            });
            assert_eq!(outcome, Err(DomainError::InvalidOrbitalPeriod));
            assert_eq!(calls, 0, "the sampler ran for a period of {period}");
        }
    }

    #[test]
    fn a_minimum_outside_every_window_the_search_opens_is_refused() {
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let velocity = geometry.relative_velocity;
        let far = 1.0e6;
        let outcome = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift([0.0, 0.0, 0.0], velocity, t - far),
            velocity,
            covariance,
        });
        assert_eq!(outcome, Err(DomainError::CurvilinearMinimumNotBracketed));
    }

    #[test]
    fn a_curve_too_flat_to_have_a_width_is_refused() {
        // A drift that stops: the relative position moves for half a rectilinear width and then holds,
        // so the curve has a genuine interior minimum but never rises past a quarter. No number of
        // doublings finds a rise of one, and the refusal is the only honest answer.
        let isotropic = spd([300.0, 0.0, 300.0, 0.0, 0.0, 300.0]);
        let geometry = ConjunctionGeometry::new(
            [400.0, 0.0, 0.0],
            [0.0, 7_500.0, 0.0],
            isotropic,
            isotropic,
            1.0,
            1.0,
        );
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        assert_eq!(rectilinear.time_of_minimum(), 0.0, "the miss is transverse");
        let stop = 0.5 * rectilinear.width();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let position = geometry.relative_position;
        let velocity = geometry.relative_velocity;
        let outcome = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift(position, velocity, t.clamp(-stop, stop)),
            velocity,
            covariance,
        });
        assert_eq!(outcome, Err(DomainError::CurvilinearWidthNotBracketed));
    }

    #[test]
    fn a_sampler_that_stops_being_a_trajectory_ends_the_search() {
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let velocity = geometry.relative_velocity;

        let non_finite = CurvilinearEncounter::locate(&rectilinear, |_| RelativeSample {
            position: [f64::NAN, 0.0, 0.0],
            velocity,
            covariance,
        });
        assert_eq!(non_finite, Err(DomainError::NonFiniteSample));

        let singular = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift(geometry.relative_position, velocity, t),
            velocity,
            covariance: [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]],
        });
        assert_eq!(singular, Err(DomainError::NonPositiveDefiniteSample));
    }

    #[test]
    fn a_zero_relative_speed_at_the_curvilinear_minimum_is_refused() {
        let geometry = head_on();
        let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
        let covariance = sum3(&geometry.primary_covariance, &geometry.secondary_covariance);
        let velocity = geometry.relative_velocity;
        let outcome = CurvilinearEncounter::locate(&rectilinear, |t| RelativeSample {
            position: drift(geometry.relative_position, velocity, t),
            velocity: [0.0, 0.0, 0.0],
            covariance,
        });
        assert_eq!(outcome, Err(DomainError::DegenerateEncounterVelocity));
    }

    /// A conjunction geometry with a strictly positive-definite covariance on each object, spanning
    /// head-on to co-orbiting relative speeds and metres to kilometres of miss.
    fn any_geometry() -> impl Strategy<Value = ConjunctionGeometry> {
        (
            proptest::array::uniform3(-2_000.0_f64..2_000.0),
            proptest::array::uniform3(-9_000.0_f64..9_000.0),
            proptest::array::uniform6(-300.0_f64..300.0),
            proptest::array::uniform6(-300.0_f64..300.0),
            5.0_f64..900.0,
            5.0_f64..900.0,
        )
            .prop_map(
                |(position, velocity, mut first, mut second, floor, ceiling)| {
                    // Force the diagonal of each L positive and away from zero, which is exactly the
                    // condition for L Lᵀ to be positive definite.
                    for (entries, scale) in [(&mut first, floor), (&mut second, ceiling)] {
                        entries[0] = entries[0].abs() + scale;
                        entries[2] = entries[2].abs() + scale;
                        entries[5] = entries[5].abs() + scale;
                    }
                    let speed = velocity[0].abs() + velocity[1].abs() + velocity[2].abs();
                    let velocity = if speed < 1.0 {
                        [1.0, 0.0, 0.0]
                    } else {
                        velocity
                    };
                    ConjunctionGeometry::new(position, velocity, spd(first), spd(second), 3.0, 2.0)
                },
            )
    }

    proptest! {
        /// The curve the module carries and the direct quadratic form are the same function, over
        /// every geometry rather than the five offsets one case can name.
        #[test]
        fn the_carried_quadratic_is_the_direct_mahalanobis_distance(
            geometry in any_geometry(),
            offset in -400.0_f64..400.0,
        ) {
            let curve = RectilinearEncounter::from_geometry(&geometry).unwrap();
            let inverse = inverse3(&sum3(&geometry.primary_covariance, &geometry.secondary_covariance));
            let direct = quadratic_form(
                &inverse,
                &drift(geometry.relative_position, geometry.relative_velocity, offset),
            );
            let carried = curve.modified_mahalanobis(offset);
            prop_assert!(
                (carried - direct).abs() <= 1e-9 * direct.abs().max(1.0),
                "carried {carried} against direct {direct}"
            );
        }

        /// A rectilinear sampler is the rectilinear model, so the detector must find nothing — this is
        /// the property that makes the inaccuracy indicator a measure of curvature and not of the
        /// search's own convergence.
        #[test]
        fn a_rectilinear_sampler_never_registers_an_inaccuracy_violation(
            geometry in any_geometry(),
        ) {
            let rectilinear = RectilinearEncounter::from_geometry(&geometry).unwrap();
            let curvilinear = CurvilinearEncounter::locate(&rectilinear, frozen(&geometry)).unwrap();
            let violations = UsageViolations::assess(&rectilinear, &curvilinear, 5_400.0).unwrap();
            prop_assert!(
                violations.inaccuracy_indicator() < 1e-6,
                "V_i = {} with F_c = {}",
                violations.inaccuracy_indicator(),
                violations.correction_factor()
            );
            prop_assert!(!violations.inaccuracy());
        }
    }
}
