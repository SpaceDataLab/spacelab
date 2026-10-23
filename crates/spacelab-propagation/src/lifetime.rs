//! Orbital lifetime and re-entry prediction.
//!
//! "How long will it stay up, and does it meet the post-mission disposal rule" is the question a
//! mission-feasibility study is bought to answer, and every ingredient for it was already in this
//! crate: the DSST drag mean-element rates that make a decade of decay affordable to integrate
//! ([`crate::dsst::drag`]), the empirical atmospheres ([`crate::nrlmsise00`],
//! [`crate::jb2008`](mod@crate::jb2008),
//! [`crate::harris_priester`](mod@crate::harris_priester)), the space-weather drivers
//! ([`crate::space_weather`]) and the calibrated-density layer ([`crate::density_calibration`]).
//! Nothing joined them. This does.
//!
//! # A single lifetime number is not a prediction
//!
//! Over a decade the answer is dominated by how active the Sun turns out to be, and that is not
//! known. A tool that quotes one number is quoting the median of a distribution it did not compute,
//! and the customer cannot tell a confident answer from an unknowable one. So the primary result
//! here is [`LifetimeEnsemble`] — the decay epoch under each of a set of driver trajectories, from
//! which quantiles are read — and [`lifetime`] is the single member it is built out of.
//!
//! Two things follow from that, and both are visible in the API:
//!
//! * **The sample is censored.** A member that has not re-entered by the horizon has a lifetime
//!   longer than the horizon and no known value, so it contributes to the *count* and not to the
//!   order statistics. [`LifetimeEnsemble::quantile`] therefore returns `None` past the censoring
//!   fraction rather than the largest decay it happened to see. Quoting the last simulated decay as
//!   the 90th percentile, when a fifth of the ensemble never decayed at all, is the specific lie
//!   this shape exists to make impossible.
//! * **Sensitivity is reported separately from spread.** Solar activity and the ballistic
//!   coefficient are independent unknowns and averaging over one does not describe the other.
//!   [`ballistic_log_sensitivity`] answers the second: `∂ln T / ∂ln B`, which is exactly `−1` for a
//!   drag-only decay through a fixed atmosphere and departs from it by however much the rest of the
//!   model matters.
//!
//! # What is in the force model, and what is not
//!
//! The right-hand side is the **zonal geopotential plus atmospheric drag**, both as DSST mean-element
//! rates, plus the Keplerian mean-longitude rate. That is the pairing decay is made of: drag removes
//! the energy, and `J2` turns the apsides, which is what decides the density the perigee passes
//! through. Third-body attraction, solar radiation pressure and the tesseral resonance are left out —
//! they are a small correction to a LEO decay and a caller who needs them has
//! [`crate::dsst::propagator::DsstMeanPropagator`] one module away. Leaving them out is a modelling
//! choice, stated here so that a residual against a reference is attributed to it rather than to a
//! bug.
//!
//! Short-period corrections are not applied. The stopping test is on the **mean** perigee altitude,
//! which is the right variable: the osculating perigee oscillates by tens of kilometres over one
//! revolution and a threshold crossing on it is a crossing of the oscillation, not of the decay.

use alloc::vec::Vec;

use crate::driver::{Controls, IntegrationError, integrate_dop853};
use crate::dsst::auxiliary::DsstAuxiliary;
use crate::dsst::drag::{Atmosphere, DsstAtmosphericDrag};
use crate::dsst::zonal::DsstZonal;
use crate::{DynamicalSystem, Tolerance};

/// The conventional entry interface: 120 km altitude.
///
/// Not a physical boundary — an object at a 120 km perigee has hours left, not days, and the
/// mean-element model is past its own validity well before it. It is where operational practice
/// stops predicting and starts calling the object decayed, and where this module stops too.
pub const DEFAULT_REENTRY_ALTITUDE: f64 = 120_000.0;

/// The most chunks one scan will step through: a century at a one-day chunk is 36 525 of them, and
/// this is twenty-seven times that.
///
/// Low enough that a test can stand on its own boundary. A ceiling nothing can reach is a ceiling
/// nothing has checked the sense of.
pub const MAX_CHUNKS: f64 = 1.0e6;

/// The most halvings any `f64` bracket can need — `f64::MAX` to the smallest subnormal is 2 098
/// doublings. Counted rather than left to the two exit tests, because a termination condition one
/// wrong operator can disable is a hang, and a hang is the one failure a test cannot see.
const MAX_BISECTIONS: usize = 2_100;

/// The atmosphere a decay integration reads — density and co-rotation as functions of **time** as
/// well as position.
///
/// [`crate::dsst::drag::Atmosphere`] has no time argument, deliberately: it is validated bit-for-bit
/// against Orekit's fixed-epoch `getMeanElementRate`, and every adapter in this crate implements it
/// at a frozen epoch. A decade of decay cannot be frozen — the solar flux driving the density is the
/// whole uncertainty — so this trait adds the argument and a private adapter binds it back down for
/// each quadrature. An atmosphere that does not vary in time reaches it through [`Frozen`].
pub trait DecayAtmosphere {
    /// Local mass density `ρ` at inertial `position`, `t` seconds after the initial epoch (kg/m³).
    fn density(&self, t: f64, position: [f64; 3]) -> f64;

    /// Inertial velocity of the atmosphere at `position` (m/s) — `ω_body × r` for a co-rotating
    /// atmosphere. Time-invariant for a rigidly rotating body, so it takes no `t`.
    fn velocity(&self, position: [f64; 3]) -> [f64; 3];
}

/// Any fixed-epoch [`Atmosphere`] read as a [`DecayAtmosphere`] that ignores time.
///
/// The honest use is a test or a short span. Over a decade it asserts that the Sun does not change,
/// which is the assumption this module exists to stop a caller making silently — so it is a named
/// wrapper rather than a blanket implementation.
#[derive(Debug, Clone, Copy)]
pub struct Frozen<A>(pub A);

impl<A: Atmosphere> DecayAtmosphere for Frozen<A> {
    fn density(&self, _t: f64, position: [f64; 3]) -> f64 {
        self.0.density(position)
    }

    fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
        self.0.velocity(position)
    }
}

/// A [`DecayAtmosphere`] bound to one epoch, so the DSST quadrature can read it.
struct At<'a, A> {
    atmosphere: &'a A,
    t: f64,
}

impl<A: DecayAtmosphere> Atmosphere for At<'_, A> {
    fn density(&self, position: [f64; 3]) -> f64 {
        self.atmosphere.density(self.t, position)
    }

    fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
        self.atmosphere.velocity(position)
    }
}

/// The decaying spacecraft and the body it is decaying towards.
///
/// `zonal` is borrowed because it owns its coefficient tables and one field serves every member of
/// an ensemble.
#[derive(Clone, Copy)]
pub struct DecayModel<'a> {
    /// Zonal geopotential mean-element rates — `J2` upwards.
    pub zonal: &'a DsstZonal,
    /// Cannonball drag: `Cd`, `S/m` and the altitude above which the atmosphere is neglected.
    pub drag: DsstAtmosphericDrag,
    /// Central attraction `µ`, m³/s².
    pub mu: f64,
    /// Equatorial radius of the central body, m — the reference the perigee altitude is measured
    /// from. Spherical, matching the mean-element perigee radius `a(1 − e)` it is subtracted from;
    /// an oblate reference would be a geodetic altitude for a radius that is not one.
    pub body_radius: f64,
    /// Mean perigee altitude at which the object is declared re-entered, m. See
    /// [`DEFAULT_REENTRY_ALTITUDE`].
    pub reentry_altitude: f64,
}

/// What became of one object over one horizon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Lifetime {
    /// The mean perigee altitude reached [`DecayModel::reentry_altitude`].
    Reentry {
        /// Seconds from the initial epoch to the crossing, bracketed to the requested tolerance.
        seconds: f64,
        /// Mean perigee altitude at the reported epoch, m — at or below the threshold, by however
        /// far the last bisection step overshot it.
        perigee_altitude: f64,
    },
    /// The object was still up at the end of the horizon. Its lifetime is longer than `horizon` and
    /// this run does not know by how much — which is the whole reason it is a separate variant and
    /// not a very large number.
    Survives {
        /// The horizon it outlived, seconds.
        horizon: f64,
        /// Mean perigee altitude at the horizon, m.
        perigee_altitude: f64,
    },
    /// The propagation stopped being steppable while the perigee was still **above** the threshold.
    ///
    /// The third answer, and the one a decay tool is most tempted to hide. A mean-element
    /// integration that can no longer hold its tolerance has met either a genuine collapse — the
    /// last hours, where `da/dt` is kilometres per orbit and the averaging the model rests on has
    /// stopped being an averaging — or a force model that is wrong. Reporting either as a re-entry
    /// would put a decay epoch on a run that never saw one, so both come back here with the epoch
    /// and the altitude reached, and the caller decides. A `perigee_altitude` a few kilometres above
    /// the threshold is the first; one still at the initial altitude is the second.
    ///
    /// The first case is not an integrator that could be tuned into succeeding. An atmosphere whose
    /// density keeps rising as the orbit falls gives a decay with a **finite-time singularity**: for
    /// an exponential, `dε/dt = −A·exp(−ε/H)` integrates to `ε = H·ln(1 − A·t/H)`, and `a → −∞` at
    /// `t* = H/A`. No solution exists past `t*`, so nothing can be asked to produce one — measured on
    /// a 400 km circular start, `rtol` from 1e-14 to 1e-10 and a step floor from 1e-6 to 1e-3 all
    /// stop at the same epoch and the same altitude.
    Unresolved {
        /// Seconds from the initial epoch to where the propagation stopped, bracketed to the
        /// requested tolerance.
        seconds: f64,
        /// The last mean perigee altitude successfully propagated to, m.
        perigee_altitude: f64,
    },
}

/// Why a lifetime could not be predicted. Every variant names the argument that was wrong and its
/// value, so that a plausible-looking answer is never the report for a malformed request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LifetimeError {
    /// The horizon is not a positive, finite number of seconds.
    Horizon(f64),
    /// The chunk length is not a positive, finite number of seconds.
    Chunk(f64),
    /// The horizon needs more chunks than one scan will take. Both arguments are individually well
    /// formed and the *pair* is not, which is why it is neither of the two above.
    TooManyChunks {
        /// The requested chunk length, seconds.
        chunk: f64,
        /// `ceil(horizon / chunk)`, against a ceiling of [`MAX_CHUNKS`].
        requested: f64,
    },
    /// The bracketing tolerance is not a positive, finite number of seconds.
    Tolerance(f64),
    /// The initial mean equinoctial state is not one this model can start from: a non-positive or
    /// non-finite semi-major axis, or an eccentricity vector at or outside the unit circle. A
    /// hyperbolic or parabolic state has no perigee to decay towards and `√(1 − k² − h²)` is not
    /// real, so the failure is named here rather than propagated as a NaN.
    InitialState([f64; 6]),
    /// A quantity was asked of a run that produced no lifetime — [`Lifetime::Survives`] or
    /// [`Lifetime::Unresolved`]. Carries the epoch the run stopped at, seconds, which is what the
    /// caller has to change (a longer horizon) or investigate (a propagation that broke down).
    NoLifetime(f64),
}

impl core::fmt::Display for LifetimeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Horizon(horizon) => write!(f, "horizon {horizon} is not positive and finite"),
            Self::Chunk(chunk) => write!(f, "chunk length {chunk} is not positive and finite"),
            Self::TooManyChunks { chunk, requested } => write!(
                f,
                "a {chunk} s chunk over this horizon is {requested} chunks, past the {MAX_CHUNKS} a scan will take"
            ),
            Self::Tolerance(tolerance) => {
                write!(f, "tolerance {tolerance} is not positive and finite")
            }
            Self::InitialState(state) => write!(
                f,
                "initial mean state {state:?} is not elliptic with a positive semi-major axis"
            ),
            Self::NoLifetime(seconds) => {
                write!(
                    f,
                    "the run produced no decay epoch, stopping at {seconds} s"
                )
            }
        }
    }
}

impl core::error::Error for LifetimeError {}

/// Mean perigee altitude of an equinoctial mean state, or `None` if the state is not elliptic.
///
/// `a(1 − e) − R` with `e = √(k² + h²)`. `None` covers every way the state can stop being one a
/// perigee exists for at once — non-finite components, `a ≤ 0`, `e ≥ 1` — because each of them makes
/// the *next* [`DsstAuxiliary`] non-real and there is nothing to be gained by telling them apart.
fn perigee_altitude(state: &[f64; 6], body_radius: f64) -> Option<f64> {
    if !state.iter().all(|component| component.is_finite()) {
        return None;
    }
    let ecc = libm::sqrt(state[1] * state[1] + state[2] * state[2]);
    if state[0] <= 0.0 || ecc >= 1.0 {
        return None;
    }
    Some(state[0] * (1.0 - ecc) - body_radius)
}

/// The DSST mean-element right-hand side of a decay: zonal + drag + the Keplerian rate.
struct DecaySystem<'a, A> {
    model: &'a DecayModel<'a>,
    atmosphere: &'a A,
}

impl<A: DecayAtmosphere> DynamicalSystem<6> for DecaySystem<'_, A> {
    fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let aux = DsstAuxiliary::new(y[0], y[1], y[2], y[3], y[4], y[5], self.model.mu);
        let zonal = self.model.zonal.mean_element_rates(&aux);
        let at = At {
            atmosphere: self.atmosphere,
            t,
        };
        let drag = self.model.drag.mean_element_rates(&aux, &at);
        for i in 0..6 {
            dydt[i] = zonal[i] + drag[i];
        }
        dydt[5] += aux.n;
    }
}

/// The tolerance the mean-element propagation is held to, matching
/// [`crate::dsst::propagator::DsstMeanPropagator`] so a drag-free configuration reproduces it.
fn controls() -> Controls {
    Controls::new(Tolerance::new(1e-14, 1e-12), 1e-6)
}

/// The first step the mean-element propagation attempts. Handed over unclamped: the driver takes
/// `h₀.min(t₁ − t₀)` itself, so a span shorter than this is already covered.
const INITIAL_STEP: f64 = 60.0;

/// Propagate the mean state from `state` at `from` to `to`, or `None` if it could not be stepped.
fn advance<A: DecayAtmosphere>(
    model: &DecayModel<'_>,
    atmosphere: &A,
    state: [f64; 6],
    from: f64,
    to: f64,
) -> Result<[f64; 6], IntegrationError> {
    if to <= from {
        return Ok(state);
    }
    let system = DecaySystem { model, atmosphere };
    integrate_dop853(&system, from, &state, to, INITIAL_STEP, controls())
        .map(|trajectory| trajectory.final_state())
}

/// Predict when one object re-enters, under one atmosphere.
///
/// `initial` is the **mean** equinoctial state `[a, k, h, q, p, λ]` at `t = 0`; the answer is
/// seconds from that epoch. The scan advances in chunks of `chunk` seconds and brackets the first
/// one across which the mean perigee altitude falls to [`DecayModel::reentry_altitude`], then
/// bisects that chunk to `tolerance`.
///
/// Bisection rather than a secant step, for the same reason [`crate::dsst`]'s callers take it: the
/// bracket is the guarantee, and halving it a counted number of times costs
/// `ceil(log2(chunk/tolerance))` propagations however the decay behaves inside — 11 for a one-day
/// chunk resolved to a minute. A superlinear step would be fewer on a smooth decay and unbounded on
/// the last few orbits, which is exactly where a decay is not smooth.
///
/// # What a chunk can and cannot promise
///
/// The scan sees a crossing only if two chunk ends straddle it, and perigee altitude falls
/// monotonically under drag, so the first crossing is the only one — there is no second minimum for
/// a coarse chunk to skip past. What a long chunk *can* do is put the whole terminal plunge inside
/// it — measured on a 400 km circular orbit through a 60 km-scale-height exponential atmosphere, the
/// last 150 km of altitude go in under half a day. That costs accuracy in the bracket, not
/// correctness. What ends a scan early is a threshold set below where the decay stops being
/// integrable at all: the same atmosphere loses its solution outright at 110.8 km
/// ([`Lifetime::Unresolved`] states why), and the bracket then ends on a failed advance. That is not
/// silent either — the bisection narrows it anyway, and the answer comes back as `Unresolved` with
/// the altitude actually reached rather than as a re-entry that was never observed.
///
/// # Errors
///
/// [`LifetimeError`] for a horizon, chunk or tolerance that is not positive and finite, for a pair
/// of them needing more than [`MAX_CHUNKS`] chunks, and for an initial state that is not elliptic.
/// A propagation that breaks down is **not** an error — it is [`Lifetime::Unresolved`], because it
/// is an outcome of the physics as often as of the arguments and the two are told apart by the
/// altitude it stopped at, which only the caller can judge.
pub fn lifetime<A: DecayAtmosphere>(
    model: &DecayModel<'_>,
    atmosphere: &A,
    initial: [f64; 6],
    horizon: f64,
    chunk: f64,
    tolerance: f64,
) -> Result<Lifetime, LifetimeError> {
    // Written against `is_finite` rather than as a negated comparison so NaN is rejected by name:
    // `!(x > 0.0)` and `x <= 0.0` agree everywhere except on NaN, which is the value a caller
    // passes by accident.
    if !horizon.is_finite() || horizon <= 0.0 {
        return Err(LifetimeError::Horizon(horizon));
    }
    if !chunk.is_finite() || chunk <= 0.0 {
        return Err(LifetimeError::Chunk(chunk));
    }
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(LifetimeError::Tolerance(tolerance));
    }
    let Some(initial_altitude) = perigee_altitude(&initial, model.body_radius) else {
        return Err(LifetimeError::InitialState(initial));
    };
    if initial_altitude <= model.reentry_altitude {
        return Ok(Lifetime::Reentry {
            seconds: 0.0,
            perigee_altitude: initial_altitude,
        });
    }

    let requested = libm::ceil(horizon / chunk);
    if requested > MAX_CHUNKS {
        return Err(LifetimeError::TooManyChunks { chunk, requested });
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to [1, MAX_CHUNKS] here, and MAX_CHUNKS is 1e6"
    )]
    let chunks = requested.clamp(1.0, MAX_CHUNKS) as usize;

    let mut anchor_time = 0.0;
    let mut anchor_state = initial;
    let mut anchor_altitude = initial_altitude;

    for i in 1..=chunks {
        // Counted from the start rather than accumulated: `t += chunk` drifts by an ulp per step
        // over the 36 525 chunks a century takes, and `i * chunk` commits one rounding per sample.
        #[expect(
            clippy::cast_precision_loss,
            reason = "i <= MAX_CHUNKS = 1e6, exactly representable in f64"
        )]
        let t = (i as f64 * chunk).min(horizon);
        let reached = advance(model, atmosphere, anchor_state, anchor_time, t)
            .ok()
            .and_then(|state| {
                perigee_altitude(&state, model.body_radius).map(|altitude| (state, altitude))
            });
        match reached {
            Some((state, altitude)) if altitude > model.reentry_altitude => {
                anchor_time = t;
                anchor_state = state;
                anchor_altitude = altitude;
            }
            other => {
                return Ok(refine(
                    model,
                    atmosphere,
                    &Anchor {
                        time: anchor_time,
                        state: anchor_state,
                        altitude: anchor_altitude,
                    },
                    (t, other.map(|(_, altitude)| altitude)),
                    tolerance,
                ));
            }
        }
    }

    Ok(Lifetime::Survives {
        horizon,
        perigee_altitude: anchor_altitude,
    })
}

/// The last chunk end the propagation reached with the perigee still above the threshold.
#[derive(Clone, Copy)]
struct Anchor {
    time: f64,
    state: [f64; 6],
    altitude: f64,
}

/// Bisect `(anchor.time, end.0]`, where the perigee is above the threshold at the anchor and the
/// bracket's far end is either at or below it (`end.1` is `Some`) or could not be propagated to at
/// all (`None`), down to `tolerance`.
///
/// Which of those two the *narrowed* bracket ends on is what decides the answer, and it is tracked
/// rather than assumed: a bracket whose far end carries an altitude is a threshold crossing and a
/// bracket whose far end carries nothing is a propagation that stopped. Collapsing the two would
/// mean reporting a decay epoch for a run that never propagated to one.
fn refine<A: DecayAtmosphere>(
    model: &DecayModel<'_>,
    atmosphere: &A,
    anchor: &Anchor,
    end: (f64, Option<f64>),
    tolerance: f64,
) -> Lifetime {
    let (mut low, mut low_altitude) = (anchor.time, anchor.altitude);
    let (mut high, mut high_altitude) = end;
    for _ in 0..MAX_BISECTIONS {
        if high - low <= tolerance {
            break;
        }
        let mid = f64::midpoint(low, high);
        // A midpoint not strictly inside is the interval exhausting `f64`, not a failure to
        // converge. Stopping there is the only honest answer and it terminates.
        if mid <= low || mid >= high {
            break;
        }
        let reached = advance(model, atmosphere, anchor.state, anchor.time, mid)
            .ok()
            .and_then(|state| perigee_altitude(&state, model.body_radius));
        match reached {
            Some(altitude) if altitude > model.reentry_altitude => {
                low = mid;
                low_altitude = altitude;
            }
            other => {
                high = mid;
                high_altitude = other;
            }
        }
    }
    high_altitude.map_or(
        // Nothing was ever propagated to past `low`, so `low` is where this run stops knowing.
        Lifetime::Unresolved {
            seconds: low,
            perigee_altitude: low_altitude,
        },
        |perigee_altitude| Lifetime::Reentry {
            seconds: high,
            perigee_altitude,
        },
    )
}

/// The decay epochs of an ensemble of driver trajectories, and what became of the members that did
/// not produce one.
///
/// Built by [`ensemble_lifetime`]. The re-entry times are held sorted; the other two outcomes are
/// held as counts, because neither has a decay epoch to put in the order statistics — and both still
/// count towards the denominator, so a run that mostly failed cannot present the few decays it did
/// see as a well-determined distribution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LifetimeEnsemble {
    reentries: Vec<f64>,
    survivors: usize,
    unresolved: usize,
}

impl LifetimeEnsemble {
    /// Build from one lifetime per ensemble member, in any order.
    #[must_use]
    pub fn from_members(members: &[Lifetime]) -> Self {
        let mut reentries: Vec<f64> = Vec::new();
        let (mut survivors, mut unresolved) = (0, 0);
        for member in members {
            match member {
                Lifetime::Reentry { seconds, .. } => reentries.push(*seconds),
                Lifetime::Survives { .. } => survivors += 1,
                Lifetime::Unresolved { .. } => unresolved += 1,
            }
        }
        reentries.sort_by(f64::total_cmp);
        Self {
            reentries,
            survivors,
            unresolved,
        }
    }

    /// How many members the ensemble was run with.
    #[must_use]
    pub fn members(&self) -> usize {
        self.reentries.len() + self.survivors + self.unresolved
    }

    /// How many members were still up at the horizon.
    #[must_use]
    pub fn survivors(&self) -> usize {
        self.survivors
    }

    /// How many members stopped being propagable before reaching the threshold.
    ///
    /// A non-zero count here is a finding, not a statistic: it says the model or the horizon does
    /// not cover what those members did, and every quantile the ensemble reports is conditioned on
    /// that.
    #[must_use]
    pub fn unresolved(&self) -> usize {
        self.unresolved
    }

    /// The decay epochs that were observed, ascending, seconds.
    #[must_use]
    pub fn reentry_times(&self) -> &[f64] {
        &self.reentries
    }

    /// The `p`-quantile of the lifetime, seconds, or `None` when the sample does not determine it.
    ///
    /// The nearest-rank definition: the smallest observed decay epoch that at least a fraction `p`
    /// of the ensemble is at or before. Nearest-rank rather than an interpolation between two order
    /// statistics, because an interpolated value is an epoch no member decayed at, and the point of
    /// the ensemble is that each number in it came out of a propagation.
    ///
    /// `None` in three cases, and the third is the one that matters. `p` outside `[0, 1]` or not
    /// finite is a malformed ask. An empty ensemble determines nothing. And **`p` past the censoring
    /// fraction is not determined by this sample**: with a fifth of the members still up at the
    /// horizon or unresolved, every quantile above the 80th lies in the tail the run did not
    /// observe, and the largest decay it happened to see is not it. Returning that number would
    /// understate the answer by however long those members go on to last.
    #[must_use]
    pub fn quantile(&self, p: f64) -> Option<f64> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return None;
        }
        let members = self.members();
        if members == 0 {
            return None;
        }
        #[expect(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "`p` is in [0, 1] and `members >= 1`, so the ceiling is in [1, members]"
        )]
        let rank = libm::ceil(p * members as f64).max(1.0) as usize;
        // `rank` counts members; `reentries` holds only those that decayed. A rank past them is a
        // quantile inside the censored tail.
        self.reentries.get(rank - 1).copied()
    }
}

/// Run one decay per ensemble member and collect the distribution.
///
/// `member(i)` supplies the atmosphere of the `i`-th member — the same orbit and spacecraft under a
/// different solar-driver trajectory. Sequential and deterministic: the same members in the same
/// order give the same ensemble, which is what makes a calibration measurement repeatable.
///
/// # Errors
///
/// The first [`LifetimeError`] any member raises, with the member index. A malformed argument is
/// malformed for every member, so there is nothing to be gained by running the rest of them.
pub fn ensemble_lifetime<A, F>(
    model: &DecayModel<'_>,
    member: F,
    members: usize,
    initial: [f64; 6],
    horizon: f64,
    chunk: f64,
    tolerance: f64,
) -> Result<LifetimeEnsemble, (usize, LifetimeError)>
where
    A: DecayAtmosphere,
    F: Fn(usize) -> A,
{
    let mut outcomes = Vec::with_capacity(members);
    for index in 0..members {
        let atmosphere = member(index);
        let outcome = lifetime(model, &atmosphere, initial, horizon, chunk, tolerance)
            .map_err(|error| (index, error))?;
        outcomes.push(outcome);
    }
    Ok(LifetimeEnsemble::from_members(&outcomes))
}

/// The logarithmic sensitivity of the lifetime to the ballistic coefficient,
/// `∂ln T / ∂ln (Cd·S/m)`, by a central difference of relative size `fraction`.
///
/// The number to compare it against is **`−1` exactly**. Drag enters the mean-element rates as a
/// multiplier, so a drag-only decay through a time-invariant atmosphere obeys
/// `dy/dt = B·f(y)`, whose solution depends on `B` and `t` only through `B·t`: doubling the
/// ballistic coefficient halves every epoch on the trajectory, including the re-entry one. Whatever
/// this returns other than `−1` is the size of everything the argument leaves out — the zonal rates,
/// which do not scale with `B`, and any time dependence in the atmosphere, which breaks the
/// rescaling of `t`.
///
/// # Errors
///
/// [`LifetimeError`] as [`lifetime`] raises it, or [`LifetimeError::Tolerance`] for a `fraction`
/// that is not in `(0, 1)`. A run that produced no decay epoch — either end of the difference
/// surviving the horizon, or either end breaking down before the threshold — is
/// [`LifetimeError::NoLifetime`] carrying the epoch it stopped at, because a derivative of a
/// quantity neither run observed is not a number worth returning.
pub fn ballistic_log_sensitivity<A: DecayAtmosphere>(
    model: &DecayModel<'_>,
    atmosphere: &A,
    initial: [f64; 6],
    horizon: f64,
    chunk: f64,
    tolerance: f64,
    fraction: f64,
) -> Result<f64, LifetimeError> {
    if !fraction.is_finite() || fraction <= 0.0 || fraction >= 1.0 {
        return Err(LifetimeError::Tolerance(fraction));
    }
    let mut decays = [0.0f64; 2];
    for (slot, scale) in decays.iter_mut().zip([1.0 - fraction, 1.0 + fraction]) {
        let perturbed = DecayModel {
            drag: DsstAtmosphericDrag {
                area_over_mass: model.drag.area_over_mass * scale,
                ..model.drag
            },
            ..*model
        };
        *slot = match lifetime(&perturbed, atmosphere, initial, horizon, chunk, tolerance)? {
            Lifetime::Reentry { seconds, .. } => seconds,
            Lifetime::Survives { horizon, .. } => return Err(LifetimeError::NoLifetime(horizon)),
            Lifetime::Unresolved { seconds, .. } => {
                return Err(LifetimeError::NoLifetime(seconds));
            }
        };
    }
    // ∂ln T / ∂ln B, with ln B differing by ln((1+f)/(1−f)) between the two runs.
    Ok(libm::log(decays[1] / decays[0]) / libm::log((1.0 + fraction) / (1.0 - fraction)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;
    use alloc::{format, vec};

    /// Assert a call was refused with a given error.
    ///
    /// `assert_eq!` on the `Result` cannot express this: the malformed arguments under test include
    /// NaN, which is not equal to itself, so the case the check exists for is the one that would
    /// fail. The `Debug` rendering compares what is actually meant — the same variant carrying back
    /// the same value, NaN included.
    fn refused<T: core::fmt::Debug>(got: &Result<T, LifetimeError>, expected: &LifetimeError) {
        assert_eq!(format!("{got:?}"), format!("{:?}", Err::<&T, _>(expected)));
    }

    const MU: f64 = 3.986_004_415e14;
    const AE: f64 = 6_378_136.46;
    const C20: f64 = -1.082_626_892_167_358_6e-3;
    /// WGS84 Earth rotation rate, rad/s.
    const OMEGA: f64 = 7.292_115e-5;

    /// A zonal field with `J2` only, and one with no field at all — the second is what isolates
    /// drag so the analytic decay law applies.
    fn zonal(j2: bool) -> DsstZonal {
        DsstZonal::new(MU, AE, vec![0.0, 0.0, if j2 { C20 } else { 0.0 }])
    }

    /// `ρ = ρ₀·exp(−(r − r₀)/H)` about the spherical radius, co-rotating about `ẑ`.
    #[derive(Clone, Copy)]
    struct Exponential {
        rho0: f64,
        r0: f64,
        scale_height: f64,
    }

    impl Atmosphere for Exponential {
        fn density(&self, position: [f64; 3]) -> f64 {
            let r = libm::sqrt(
                position[0] * position[0] + position[1] * position[1] + position[2] * position[2],
            );
            self.rho0 * libm::exp(-(r - self.r0) / self.scale_height)
        }

        fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
            [-OMEGA * position[1], OMEGA * position[0], 0.0]
        }
    }

    /// 400 km circular, equatorial, and dense enough to decay inside a test.
    fn model(zonal: &DsstZonal, area_over_mass: f64) -> DecayModel<'_> {
        DecayModel {
            zonal,
            drag: DsstAtmosphericDrag::new(2.2, area_over_mass, MU),
            mu: MU,
            body_radius: AE,
            reentry_altitude: DEFAULT_REENTRY_ALTITUDE,
        }
    }

    fn circular(altitude: f64) -> [f64; 6] {
        [AE + altitude, 0.0, 0.0, 0.0, 0.0, 0.0]
    }

    fn air() -> Frozen<Exponential> {
        Frozen(Exponential {
            rho0: 2.0e-11,
            r0: AE + 400_000.0,
            scale_height: 60_000.0,
        })
    }

    /// The decay of a circular orbit is `da/dt = −ρ(a)·(Cd·S/m)·√(µa)`, and the DSST Gauss average
    /// has to reproduce it.
    ///
    /// This is the whole assembly against an oracle that shares none of its machinery. For a
    /// circular orbit the drag acceleration is antiparallel to the velocity, so the energy rate is
    /// `−½ρ(Cd·S/m)v³` with `v = √(µ/a)`; differentiating `E = −µ/2a` gives the law above. The
    /// reference is that scalar ODE integrated on its own by classical RK4 at a step small enough
    /// that its own truncation is far below the agreement asserted — no equinoctial elements, no
    /// quadrature, no `DsstAuxiliary`.
    ///
    /// The co-rotation is what the two do not share exactly: the reference is a two-body drag law
    /// with `v_rel = v`, and the model's atmosphere turns with the Earth, which at an equatorial
    /// 400 km takes ~6 % off `|v_rel|` and so ~11 % off `|v_rel|²`. The comparison is therefore run
    /// with the field switched off *and* the co-rotation removed, which is the configuration the
    /// closed form describes; `the_corotation_slows_the_decay` measures what putting it back does.
    #[test]
    fn a_circular_decay_follows_the_analytic_energy_law() {
        #[derive(Clone, Copy)]
        struct Static(Exponential);
        impl Atmosphere for Static {
            fn density(&self, position: [f64; 3]) -> f64 {
                self.0.density(position)
            }
            fn velocity(&self, _position: [f64; 3]) -> [f64; 3] {
                [0.0, 0.0, 0.0]
            }
        }

        let field = zonal(false);
        let atmosphere = Frozen(Static(air().0));
        let spacecraft = model(&field, 0.02);
        // Ten days takes the orbit from 400 km to ~330 km, which is inside the smooth part of the
        // decay: this orbit collapses at ~15 days, and the last 150 km go in under half a day, where
        // the step controller is fighting the plunge rather than measuring the law.
        let span = 10.0 * 86_400.0;

        let start = circular(400_000.0);
        let stepped = advance(&spacecraft, &atmosphere, start, 0.0, span).unwrap();

        // The reference: `da/dt = −ρ(a)·(Cd·S/m)·√(µa)`, RK4 at 600 s. One step moves `a` by ~0.1 m
        // out of 6.8e6, so the reference's own truncation is nowhere near what is being measured.
        let b = spacecraft.drag.cd * spacecraft.drag.area_over_mass;
        let rate = |a: f64| -atmosphere.0.0.density([a, 0.0, 0.0]) * b * libm::sqrt(MU * a);
        let mut a = start[0];
        let h = 600.0;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the span is ten days and the step 600 s, so the count is 1440"
        )]
        let steps = (span / h) as usize;
        for _ in 0..steps {
            let k1 = rate(a);
            let k2 = rate(a + 0.5 * h * k1);
            let k3 = rate(a + 0.5 * h * k2);
            let k4 = rate(a + h * k3);
            a += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        let decayed = start[0] - stepped[0];
        let reference = start[0] - a;
        let relative = (decayed - reference).abs() / reference;
        assert!(
            relative < 1.0e-9,
            "10-day decay {decayed:.1} m against the analytic {reference:.1} m ({relative:.2e})"
        );
        std::println!(
            "circular decay: DSST {decayed:.1} m, analytic {reference:.1} m, {relative:.2e} relative"
        );
    }

    /// A co-rotating atmosphere is overtaken more slowly than a static one, so the same orbit through
    /// the same density decays less. Measures the effect the analytic comparison above holds out.
    #[test]
    fn the_corotation_slows_the_decay() {
        #[derive(Clone, Copy)]
        struct Static(Exponential);
        impl Atmosphere for Static {
            fn density(&self, position: [f64; 3]) -> f64 {
                self.0.density(position)
            }
            fn velocity(&self, _position: [f64; 3]) -> [f64; 3] {
                [0.0, 0.0, 0.0]
            }
        }

        let field = zonal(false);
        let spacecraft = model(&field, 0.02);
        let start = circular(400_000.0);
        let span = 10.0 * 86_400.0;

        let still = advance(&spacecraft, &Frozen(Static(air().0)), start, 0.0, span).unwrap();
        let turning = advance(&spacecraft, &air(), start, 0.0, span).unwrap();
        let (still_decay, turning_decay) = (start[0] - still[0], start[0] - turning[0]);
        assert!(
            turning_decay < still_decay,
            "co-rotating decay {turning_decay:.1} m is not below the static {still_decay:.1} m"
        );
        // The instantaneous *rate* ratio is `(1 − ωr/v)²` — the energy loss goes as `|v_rel|·(v_rel·v)`
        // and an equatorial 400 km orbit overtakes the atmosphere at ~94 % of its inertial speed. The
        // ratio of the *decays* is not that number, because the decay is self-amplifying: whichever
        // orbit is lower sits in denser air. Integrating `dε/dt = −A·exp(−ε/H)` from `ε = 0` gives
        // `ε = H·ln(1 − A·t/H)`, so writing `x = 1 − exp(−D_still/H)` for the static case, the slower
        // one decays `H·|ln(1 − r·x)|` and the ratio is `ln(1 − r·x)/ln(1 − x)` — here 0.80, not 0.88.
        let r = {
            let u = OMEGA * start[0] / libm::sqrt(MU / start[0]);
            (1.0 - u) * (1.0 - u)
        };
        let x = 1.0 - libm::exp(-still_decay / air().0.scale_height);
        let expected = libm::log(1.0 - r * x) / libm::log(1.0 - x);
        let ratio = turning_decay / still_decay;
        assert!(
            (ratio - expected).abs() < 5.0e-3,
            "co-rotation ratio {ratio:.4} against the closed-form {expected:.4}"
        );
        std::println!(
            "co-rotation: {ratio:.4} of the static decay over 10 days, closed form {expected:.4}"
        );
    }

    /// Doubling the ballistic coefficient halves the lifetime, and the model has to know it exactly.
    ///
    /// With no field and a time-invariant atmosphere the mean-element rate is `B·f(y)`, so `y(t)`
    /// depends on `B` and `t` only through `B·t` and every epoch on the trajectory — the re-entry one
    /// included — scales as `1/B`. This is an identity of the equations, not an approximation, which
    /// is why it is asserted to the bracketing tolerance rather than to a percentage.
    #[test]
    fn the_lifetime_scales_exactly_as_the_inverse_ballistic_coefficient() {
        let field = zonal(false);
        let atmosphere = air();
        let start = circular(250_000.0);
        let (horizon, chunk, tolerance) = (400.0 * 86_400.0, 86_400.0, 60.0);

        let mut decays = [0.0f64; 2];
        for (slot, area_over_mass) in decays.iter_mut().zip([0.01, 0.02]) {
            let spacecraft = model(&field, area_over_mass);
            let Lifetime::Reentry { seconds, .. } =
                lifetime(&spacecraft, &atmosphere, start, horizon, chunk, tolerance).unwrap()
            else {
                unreachable!("a 250 km orbit at this density decays inside 400 days")
            };
            *slot = seconds;
        }
        // Both epochs are bracketed to `tolerance`, so the halving can only be off by that much.
        let expected = decays[0] / 2.0;
        assert!(
            (decays[1] - expected).abs() <= tolerance,
            "double the ballistic coefficient decayed at {:.1} s, not the {expected:.1} s that halves {:.1} s",
            decays[1],
            decays[0]
        );
        std::println!(
            "1/B scaling: {:.0} s at S/m=0.01, {:.0} s at 0.02",
            decays[0],
            decays[1]
        );
    }

    /// The reported sensitivity is `−1` for the case the derivation covers, and departs from it by
    /// the size of the zonal rates when the field is switched on.
    #[test]
    fn the_ballistic_sensitivity_is_minus_one_without_a_field() {
        let atmosphere = air();
        let start = circular(250_000.0);
        let (horizon, chunk, tolerance) = (400.0 * 86_400.0, 86_400.0, 60.0);

        let bare = zonal(false);
        let without = ballistic_log_sensitivity(
            &model(&bare, 0.015),
            &atmosphere,
            start,
            horizon,
            chunk,
            tolerance,
            0.1,
        )
        .unwrap();
        assert!(
            (without + 1.0).abs() < 1.0e-3,
            "drag-only sensitivity {without:.6}, not −1"
        );

        let with_j2 = zonal(true);
        let with = ballistic_log_sensitivity(
            &model(&with_j2, 0.015),
            &atmosphere,
            start,
            horizon,
            chunk,
            tolerance,
            0.1,
        )
        .unwrap();
        assert!(with.is_finite(), "sensitivity under J2 is {with}");
        std::println!("∂lnT/∂lnB: {without:.6} drag-only, {with:.6} with J2");
    }

    /// An orbit high enough not to decay is reported as surviving, not as a very large lifetime.
    #[test]
    fn an_orbit_that_does_not_decay_says_so() {
        let field = zonal(true);
        let spacecraft = model(&field, 0.02);
        let horizon = 30.0 * 86_400.0;
        let outcome = lifetime(
            &spacecraft,
            &air(),
            circular(1_200_000.0),
            horizon,
            86_400.0,
            60.0,
        )
        .unwrap();
        let Lifetime::Survives {
            horizon: reported,
            perigee_altitude,
        } = outcome
        else {
            unreachable!("a 1 200 km orbit does not decay in a month")
        };
        assert_eq!(reported, horizon);
        assert!(
            perigee_altitude > 1_100_000.0,
            "perigee fell to {perigee_altitude:.0} m in a month at 1 200 km"
        );
    }

    /// The third answer: a run that neither re-entered nor reached the horizon.
    ///
    /// The trigger here is the second case [`Lifetime::Unresolved`] names — a force model that does
    /// not evaluate — rather than the first. That is deliberate, for two reasons. Driving the
    /// *physical* case needs a threshold below where the decay stops being integrable, and there the
    /// bisection lands on a state whose perigee is already under the surface, which is a crossing
    /// and is honestly reported as one. And a force model that breaks at an *interior* epoch tests
    /// the step controller at a discontinuity rather than anything this module owns — it is what
    /// exposed the non-finite-error mapping in [`crate::driver`], and it is asserted there. Breaking
    /// from the first evaluation asserts what does belong here: the perigee reported is the initial
    /// one, which no re-entry could ever be.
    #[test]
    fn a_run_that_neither_reentered_nor_survived_says_so() {
        struct Unevaluable(Exponential);
        impl DecayAtmosphere for Unevaluable {
            fn density(&self, _t: f64, _position: [f64; 3]) -> f64 {
                f64::INFINITY
            }
            fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
                self.0.velocity(position)
            }
        }

        let field = zonal(true);
        let spacecraft = model(&field, 0.02);
        let outcome = lifetime(
            &spacecraft,
            &Unevaluable(air().0),
            circular(400_000.0),
            30.0 * 86_400.0,
            86_400.0,
            60.0,
        )
        .unwrap();
        let Lifetime::Unresolved {
            seconds,
            perigee_altitude,
        } = outcome
        else {
            unreachable!("an atmosphere of infinite density cannot be stepped through")
        };
        assert_eq!(
            seconds, 0.0,
            "no epoch was ever reached, so none can be reported"
        );
        assert!(
            (perigee_altitude - 400_000.0).abs() < 1.0,
            "reported {perigee_altitude:.1} m, not the 400 km the orbit never left"
        );
    }

    /// A state already at or below the threshold has decayed, and the answer is zero rather than an
    /// error or a scan that starts by looking for a crossing it is already past.
    #[test]
    fn a_state_already_below_the_threshold_has_a_lifetime_of_zero() {
        let field = zonal(true);
        let spacecraft = model(&field, 0.02);
        let outcome = lifetime(
            &spacecraft,
            &air(),
            circular(90_000.0),
            86_400.0,
            3_600.0,
            60.0,
        )
        .unwrap();
        assert_eq!(
            outcome,
            Lifetime::Reentry {
                seconds: 0.0,
                perigee_altitude: 90_000.0
            }
        );
    }

    /// Every malformed argument is refused by name, and none of them can produce a lifetime.
    #[test]
    fn every_malformed_argument_names_itself() {
        let field = zonal(true);
        let spacecraft = model(&field, 0.02);
        let start = circular(400_000.0);
        let run = |horizon, chunk, tolerance, initial| {
            lifetime(&spacecraft, &air(), initial, horizon, chunk, tolerance)
        };

        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            refused(
                &run(bad, 3_600.0, 60.0, start),
                &LifetimeError::Horizon(bad),
            );
            refused(&run(86_400.0, bad, 60.0, start), &LifetimeError::Chunk(bad));
            refused(
                &run(86_400.0, 3_600.0, bad, start),
                &LifetimeError::Tolerance(bad),
            );
        }

        // Not elliptic: a negative axis, an eccentricity vector on the unit circle, and a NaN.
        for bad in [
            [-7.0e6, 0.0, 0.0, 0.0, 0.0, 0.0],
            [7.0e6, 0.8, 0.6, 0.0, 0.0, 0.0],
            [7.0e6, f64::NAN, 0.0, 0.0, 0.0, 0.0],
        ] {
            refused(
                &run(86_400.0, 3_600.0, 60.0, bad),
                &LifetimeError::InitialState(bad),
            );
        }

        // The chunk count is the pair, not either argument: both are well formed here.
        let err = run(1.0e9, 1.0, 60.0, start).unwrap_err();
        assert_eq!(
            err,
            LifetimeError::TooManyChunks {
                chunk: 1.0,
                requested: 1.0e9
            }
        );
    }

    /// Exactly [`MAX_CHUNKS`] chunks is accepted and one more is refused — the boundary the ceiling
    /// is stated at, stood on from both sides.
    #[test]
    fn the_chunk_ceiling_is_reachable_from_both_sides() {
        let field = zonal(true);
        let spacecraft = model(&field, 0.0);
        let start = circular(1_200_000.0);
        // Zero area-to-mass: no drag, so the scan runs to the horizon and the cost is the chunk
        // count alone. A ceiling test must not also be a decay test.
        let at = lifetime(&spacecraft, &air(), start, MAX_CHUNKS, 1.0, 0.5).unwrap();
        assert!(matches!(at, Lifetime::Survives { .. }));
        assert_eq!(
            lifetime(&spacecraft, &air(), start, MAX_CHUNKS + 1.0, 1.0, 0.5),
            Err(LifetimeError::TooManyChunks {
                chunk: 1.0,
                requested: MAX_CHUNKS + 1.0
            })
        );
    }

    /// Every error renders, and no two render alike — the check that the `Display` arms are not
    /// interchangeable and that none of them is empty.
    #[test]
    fn every_error_renders_and_no_two_render_alike() {
        extern crate std;
        use std::string::ToString as _;
        let errors = [
            LifetimeError::Horizon(-1.0),
            LifetimeError::Chunk(0.0),
            LifetimeError::TooManyChunks {
                chunk: 1.0,
                requested: 2.0e6,
            },
            LifetimeError::Tolerance(f64::NAN),
            LifetimeError::InitialState([0.0; 6]),
            LifetimeError::NoLifetime(3.0e6),
        ];
        for (i, error) in errors.iter().enumerate() {
            let rendered = error.to_string();
            assert!(!rendered.is_empty(), "{error:?} renders empty");
            for other in &errors[i + 1..] {
                assert_ne!(rendered, other.to_string(), "{error:?} vs {other:?}");
            }
        }
    }

    /// The quantile is read off the order statistics, and stops where the sample stops determining
    /// it.
    #[test]
    fn a_censored_ensemble_refuses_the_quantiles_it_did_not_observe() {
        let horizon = 100.0;
        let members: Vec<Lifetime> = (1..=8)
            .map(|i| Lifetime::Reentry {
                seconds: f64::from(i) * 10.0,
                perigee_altitude: 120_000.0,
            })
            .chain([
                Lifetime::Survives {
                    horizon,
                    perigee_altitude: 300_000.0,
                },
                Lifetime::Survives {
                    horizon,
                    perigee_altitude: 310_000.0,
                },
            ])
            .collect();
        let ensemble = LifetimeEnsemble::from_members(&members);
        assert_eq!(ensemble.members(), 10);
        assert_eq!(ensemble.survivors(), 2);
        assert_eq!(ensemble.reentry_times().len(), 8);

        // Nearest rank: ceil(p·n) into the ascending decay epochs.
        assert_eq!(ensemble.quantile(0.1), Some(10.0));
        assert_eq!(ensemble.quantile(0.5), Some(50.0));
        assert_eq!(ensemble.quantile(0.8), Some(80.0));
        // Past the censoring fraction the sample says nothing, and neither does this.
        assert_eq!(ensemble.quantile(0.81), None);
        assert_eq!(ensemble.quantile(0.9), None);
        assert_eq!(ensemble.quantile(1.0), None);
        // Malformed asks and an empty ensemble.
        for p in [-0.1, 1.1, f64::NAN] {
            assert_eq!(ensemble.quantile(p), None);
        }
        assert_eq!(LifetimeEnsemble::default().quantile(0.5), None);
        // p = 0 is rank 1, the earliest decay — not an error and not the median.
        assert_eq!(ensemble.quantile(0.0), Some(10.0));
    }

    /// An ensemble is built by running the members, and a denser atmosphere decays sooner — so the
    /// spread across the members is the spread of the drivers and not an artefact of the collection.
    #[test]
    fn a_denser_driver_decays_sooner_and_the_ensemble_orders_them() {
        let field = zonal(false);
        let spacecraft = model(&field, 0.02);
        let start = circular(250_000.0);
        let member = |i: usize| {
            #[expect(clippy::cast_precision_loss, reason = "i < 5")]
            let scale = 1.0 + 0.25 * i as f64;
            Frozen(Exponential {
                rho0: 2.0e-11 * scale,
                r0: AE + 400_000.0,
                scale_height: 60_000.0,
            })
        };
        let ensemble = ensemble_lifetime(
            &spacecraft,
            member,
            5,
            start,
            400.0 * 86_400.0,
            86_400.0,
            60.0,
        )
        .unwrap();
        assert_eq!(ensemble.members(), 5);
        assert_eq!(ensemble.survivors(), 0);
        let times = ensemble.reentry_times();
        assert!(
            times.windows(2).all(|pair| pair[0] <= pair[1]),
            "decay epochs are not sorted: {times:?}"
        );
        // The densest member is 2× the sparsest, and density scales the rate exactly as `B` does,
        // so its lifetime is half — the same identity the ballistic scaling test rests on.
        let ratio = times[times.len() - 1] / times[0];
        assert!(
            (ratio - 2.0).abs() < 1.0e-3,
            "densest to sparsest lifetime ratio {ratio:.6}, not 2"
        );
        assert_eq!(ensemble.quantile(0.2), Some(times[0]));
        std::println!("ensemble: {:?} days", {
            let mut days = vec![];
            for t in times {
                days.push(t / 86_400.0);
            }
            days
        });
    }

    /// A member that raises names which member it was, so a malformed ensemble is not reported as a
    /// distribution over the members that happened to run.
    #[test]
    fn an_ensemble_reports_the_member_that_failed() {
        let field = zonal(true);
        let spacecraft = model(&field, 0.02);
        let error = ensemble_lifetime(
            &spacecraft,
            |_| air(),
            3,
            circular(400_000.0),
            -1.0,
            86_400.0,
            60.0,
        )
        .unwrap_err();
        assert_eq!(error, (0, LifetimeError::Horizon(-1.0)));
    }

    /// The sensitivity refuses a step fraction outside `(0, 1)`, and refuses to differentiate a
    /// lifetime it never observed.
    #[test]
    fn the_sensitivity_refuses_a_bad_step_and_a_horizon_it_did_not_reach() {
        let field = zonal(true);
        let spacecraft = model(&field, 0.02);
        let start = circular(400_000.0);
        for bad in [0.0, 1.0, -0.1, f64::NAN] {
            refused(
                &ballistic_log_sensitivity(
                    &spacecraft,
                    &air(),
                    start,
                    86_400.0,
                    3_600.0,
                    60.0,
                    bad,
                ),
                &LifetimeError::Tolerance(bad),
            );
        }
        // A month is nowhere near long enough for a 1200 km orbit at this density, so there is no
        // lifetime to differentiate and the horizon is what the caller has to change.
        let horizon = 30.0 * 86_400.0;
        assert_eq!(
            ballistic_log_sensitivity(
                &spacecraft,
                &air(),
                circular(1_200_000.0),
                horizon,
                86_400.0,
                60.0,
                0.1
            ),
            Err(LifetimeError::NoLifetime(horizon))
        );
    }

    /// A time-varying atmosphere is read at the epoch of each quadrature, not at the initial one.
    ///
    /// The whole reason [`DecayAtmosphere`] carries a `t` is that the density a decade out is not
    /// the density today. A member whose density switches on halfway through must decay later than
    /// one where it was on from the start, and sooner than one where it is never on — which is three
    /// distinct answers out of three atmospheres that agree at `t = 0`.
    #[test]
    fn the_atmosphere_is_read_at_the_epoch_of_each_quadrature() {
        struct Switched {
            on_at: f64,
            air: Exponential,
        }
        impl DecayAtmosphere for Switched {
            fn density(&self, t: f64, position: [f64; 3]) -> f64 {
                if t >= self.on_at {
                    self.air.density(position)
                } else {
                    0.0
                }
            }
            fn velocity(&self, position: [f64; 3]) -> [f64; 3] {
                self.air.velocity(position)
            }
        }

        let field = zonal(false);
        let spacecraft = model(&field, 0.02);
        let start = circular(250_000.0);
        let (horizon, chunk, tolerance) = (400.0 * 86_400.0, 86_400.0, 60.0);
        let half = 40.0 * 86_400.0;

        let decay = |on_at: f64| {
            let atmosphere = Switched {
                on_at,
                air: air().0,
            };
            match lifetime(&spacecraft, &atmosphere, start, horizon, chunk, tolerance).unwrap() {
                Lifetime::Reentry { seconds, .. } => seconds,
                Lifetime::Survives { .. } => f64::INFINITY,
                Lifetime::Unresolved { seconds, .. } => {
                    unreachable!("the run broke down at {seconds} s")
                }
            }
        };
        let always = decay(0.0);
        let later = decay(half);
        assert!(always.is_finite(), "an always-on atmosphere did not decay");
        assert!(
            later > always,
            "switching the atmosphere on at {half:.0} s decayed at {later:.0} s, no later than the \
             always-on {always:.0} s"
        );
        // Every second before the switch is a second the orbit did not decay in, so the delay is
        // the switch epoch itself — an identity, because the state at the switch is the initial one.
        assert!(
            (later - (always + half)).abs() <= 2.0 * tolerance,
            "delayed decay {later:.1} s is not the always-on {always:.1} s plus the {half:.1} s wait"
        );
        assert!(
            decay(horizon).is_infinite(),
            "a never-on atmosphere decayed"
        );
    }

    /// `a(1 − e) − R`, on numbers whose answer is arithmetic rather than a propagation.
    #[test]
    fn the_perigee_altitude_is_the_apsis_minus_the_body_radius() {
        // a = 7 000 km, e = 0.1: the perigee radius is 6 300 km and the altitude is what the
        // equatorial radius leaves of it, which here is negative. That is the case worth pinning —
        // a perigee below the surface is a state this function answers for, not one it refuses.
        let below = perigee_altitude(&[7.0e6, 0.1, 0.0, 0.0, 0.0, 0.0], AE).unwrap();
        assert!(
            (below - (6.3e6 - AE)).abs() < 1.0e-6,
            "{below:.6} m, not the −78 136.46 m a 6 300 km perigee leaves"
        );
        // The same eccentricity as the modulus of (k, h) rather than as either component.
        let split = perigee_altitude(&[7.0e6, 0.06, 0.08, 0.0, 0.0, 0.0], AE).unwrap();
        assert!(
            (split - below).abs() < 1.0e-6,
            "(k, h) = (0.06, 0.08) gave {split:.6} m against {below:.6} m for e = 0.1"
        );
        // Circular: the perigee is the axis, and the altitude is what stands above the radius.
        let circle = perigee_altitude(&circular(400_000.0), AE).unwrap();
        assert!(
            (circle - 400_000.0).abs() < 1.0e-6,
            "a circular 400 km orbit reported {circle:.6} m"
        );
    }

    /// The mean longitude rate is the perturbation *plus* the Keplerian mean motion.
    ///
    /// `λ` is the only element with a two-body rate, and the DSST mean-element rates do not carry
    /// it: they are the perturbation alone. Dropping the `+ n` leaves an orbit that does not go
    /// round, and with the field off there is nothing else in `dλ/dt` big enough for a test of the
    /// decay to notice.
    #[test]
    fn the_mean_longitude_rate_carries_the_keplerian_mean_motion() {
        let field = zonal(false);
        let spacecraft = model(&field, 0.02);
        let atmosphere = air();
        let system = DecaySystem {
            model: &spacecraft,
            atmosphere: &atmosphere,
        };
        let state = circular(400_000.0);
        let mut dydt = [0.0f64; 6];
        system.derivative(0.0, &state, &mut dydt);

        let n = libm::sqrt(MU / (state[0] * state[0] * state[0]));
        let relative = (dydt[5] - n).abs() / n;
        assert!(
            relative < 1.0e-6,
            "dλ/dt {:.9e} rad/s against n = {n:.9e} rad/s ({relative:.2e})",
            dydt[5]
        );
    }

    /// The members that produced no decay epoch are counted, and counted in the denominator.
    #[test]
    fn an_ensemble_counts_the_members_that_produced_no_epoch() {
        let ensemble = LifetimeEnsemble::from_members(&[
            Lifetime::Reentry {
                seconds: 30.0,
                perigee_altitude: 120_000.0,
            },
            Lifetime::Unresolved {
                seconds: 10.0,
                perigee_altitude: 150_000.0,
            },
            Lifetime::Survives {
                horizon: 100.0,
                perigee_altitude: 300_000.0,
            },
            Lifetime::Unresolved {
                seconds: 20.0,
                perigee_altitude: 140_000.0,
            },
            Lifetime::Reentry {
                seconds: 10.0,
                perigee_altitude: 118_000.0,
            },
        ]);
        assert_eq!(ensemble.members(), 5);
        assert_eq!(ensemble.survivors(), 1);
        assert_eq!(ensemble.unresolved(), 2);
        assert_eq!(ensemble.reentry_times(), &[10.0, 30.0]);
        // Three of five determined an epoch, so the 60th percentile onwards is not this sample's
        // to report — the two that broke down censor it exactly as the survivor does.
        assert_eq!(ensemble.quantile(0.4), Some(30.0));
        assert_eq!(ensemble.quantile(0.6), None);
    }

    /// A tolerance no finer than the bracket is honoured: the chunk end is the answer, unbisected.
    ///
    /// The bracket the scan hands over is `(anchor, chunk end]`, and its width is one chunk however
    /// far into the scan the crossing falls. A caller who asks for less resolution than that has
    /// asked for no bisection at all, and the epoch that comes back is the chunk end itself.
    #[test]
    fn a_tolerance_wider_than_the_bracket_stops_at_the_chunk_end() {
        let field = zonal(false);
        let atmosphere = air();
        let mut spacecraft = model(&field, 0.02);
        let start = circular(250_000.0);
        let horizon = 400.0 * 86_400.0;
        // A threshold at 200 km rather than the default 120: the far end of the bracket has to be
        // an altitude the propagation actually reached, and a third of a lifetime past a crossing
        // at 120 km is inside the plunge the atmosphere has no solution through.
        spacecraft.reentry_altitude = 200_000.0;

        let Lifetime::Reentry { seconds: decay, .. } =
            lifetime(&spacecraft, &atmosphere, start, horizon, 86_400.0, 60.0).unwrap()
        else {
            unreachable!("a 250 km orbit at this density falls to 200 km inside 400 days")
        };
        // Two thirds of the lifetime per chunk puts the crossing inside the second one, so the
        // anchor the bisection starts from is not zero.
        let chunk = decay / 1.5;
        let Lifetime::Reentry { seconds, .. } =
            lifetime(&spacecraft, &atmosphere, start, horizon, chunk, 1.5 * chunk).unwrap()
        else {
            unreachable!("the same orbit and atmosphere decayed at {decay} s a moment ago")
        };
        assert_eq!(
            seconds,
            2.0 * chunk,
            "a tolerance of {:.0} s over a bracket of {chunk:.0} s bisected anyway",
            1.5 * chunk
        );
    }

    /// A perigee exactly at the threshold has re-entered, at both places the comparison is made.
    ///
    /// The threshold is a level the perigee *falls to*, so equality belongs on the re-entered side.
    /// An exact tie is not reachable by choosing a round number — but it is reachable by asking the
    /// propagation what altitude it arrives at and making that the threshold, because the same
    /// inputs give the same bits. The scan compares at every chunk end and the bisection compares
    /// at every midpoint, and the two are separate comparisons in separate functions.
    #[test]
    fn a_perigee_exactly_at_the_threshold_has_reentered() {
        let field = zonal(false);
        let atmosphere = air();
        let start = circular(250_000.0);
        let horizon = 400.0 * 86_400.0;
        let mut spacecraft = model(&field, 0.02);

        let Lifetime::Reentry { seconds: decay, .. } =
            lifetime(&spacecraft, &atmosphere, start, horizon, 86_400.0, 60.0).unwrap()
        else {
            unreachable!("a 250 km orbit at this density decays inside 400 days")
        };
        let altitude_at = |spacecraft: &DecayModel<'_>, t: f64| {
            let state = advance(spacecraft, &atmosphere, start, 0.0, t).unwrap();
            perigee_altitude(&state, spacecraft.body_radius).unwrap()
        };

        // The scan's comparison, at the end of the first chunk.
        let chunk = 0.5 * decay;
        spacecraft.reentry_altitude = altitude_at(&spacecraft, chunk);
        let Lifetime::Reentry { seconds, .. } =
            lifetime(&spacecraft, &atmosphere, start, horizon, chunk, 60.0).unwrap()
        else {
            unreachable!("the threshold is an altitude this orbit reaches at {chunk} s")
        };
        assert_eq!(
            seconds, chunk,
            "the tie at the chunk end was read as still flying"
        );

        // The bisection's comparison, at the first midpoint of the first chunk. A tolerance of one
        // eighth of the chunk stops the halving while the two answers are still an eighth apart:
        // the tie sends the bracket down to the midpoint, and reading it the other way sends the
        // anchor up to it and the epoch out to five eighths.
        let chunk = 0.8 * decay;
        spacecraft.reentry_altitude = altitude_at(&spacecraft, 0.5 * chunk);
        let Lifetime::Reentry { seconds, .. } = lifetime(
            &spacecraft,
            &atmosphere,
            start,
            horizon,
            chunk,
            0.125 * chunk,
        )
        .unwrap() else {
            unreachable!(
                "the threshold is an altitude this orbit reaches at {} s",
                0.5 * chunk
            )
        };
        assert_eq!(
            seconds,
            0.5 * chunk,
            "the tie at the midpoint was read as still flying"
        );
    }
}
