//! A propagatable satellite, and the batch entry point over many of them.

use core::fmt;

use spacelab_core::{NoradId, UtcDateTime, Vec3};
use spacelab_frames::{StateVector, Teme};

/// Nanoseconds in a minute, as an exact `f64`. `6e10 < 2⁵³`.
const NS_PER_MINUTE: f64 = 60e9;

/// Metres in a kilometre. SGP4 speaks kilometres; nothing else here does.
const METRES_PER_KM: f64 = 1000.0;

/// Degrees to radians. A TLE prints angles in degrees; the model wants radians.
const DEG_TO_RAD: f64 = core::f64::consts::PI / 180.0;

/// Revolutions per day to radians per minute: `2π / 1440`.
const REV_PER_DAY_TO_RAD_PER_MIN: f64 = core::f64::consts::PI / 720.0;

/// Equatorial radius of the WGS-72 ellipsoid, metres.
///
/// WGS-72, not WGS-84: this is the constant the SGP4 model is defined against, and
/// the re-entry test must ask the same question the model asks.
const WGS72_EQUATORIAL_RADIUS_M: f64 = 6_378_135.0;

/// Why a satellite could not be built, or a state could not be produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropagationError {
    /// The two-line element set could not be read.
    MalformedElements,
    /// The element set is syntactically fine but its checksum is wrong.
    ///
    /// Separate from [`Self::MalformedElements`] because a caller ingesting ten
    /// thousand catalog entries may legitimately choose to trust the digits over
    /// the checksum, and should have to say so.
    BadChecksum,
    /// The element set is outside the range SGP4 admits.
    UnusableElements,
    /// SGP4 diverged: the mean elements left the model's domain.
    ///
    /// This is not a bug. A propagator asked to carry an object outside the region
    /// where its series converge has no answer, and should not invent one.
    Diverged,
    /// The object has re-entered: its predicted radius fell below the Earth's.
    ///
    /// The `sgp4` crate does not report this. Vallado's criterion is `r < Rₑ`, and
    /// without it the propagator cheerfully returns positions inside the planet —
    /// six of them, in the published verification set.
    Decayed,
}

impl fmt::Display for PropagationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedElements => write!(f, "the element set could not be parsed"),
            Self::BadChecksum => write!(f, "the element set's checksum is wrong"),
            Self::UnusableElements => write!(f, "the element set is outside SGP4's domain"),
            Self::Diverged => write!(f, "SGP4 diverged: the mean elements left its domain"),
            Self::Decayed => write!(f, "the object has re-entered"),
        }
    }
}

impl core::error::Error for PropagationError {}

/// Which constants and expressions SGP4 is evaluated with.
///
/// # This is not a style preference
///
/// A TLE is a *fit*, not a measurement. The elements are chosen so that a particular
/// propagator reproduces the observations. Feeding them to a different propagator is
/// a model mismatch, and it fails *plausibly*: the orbit still looks like an orbit.
///
/// Three separate things get called "AFSPC compatibility". Only two of them are choices
/// here, and the one the vendor crate spells that way is not the one the literature means.
///
/// **Gravity model and epoch expression.** WGS-72 with the AFSPC expressions is what
/// the elements were fitted under. WGS-84 displaces the ISS by 85 m over 48 hours.
///
/// **The Lyddane branch.** The lunisolar long-period periodics are recovered one way for
/// low-inclination orbits and another way above them. The original Spacetrack Report #3
/// takes that decision on the epoch inclination, the later GSFC code on the perturbed one.
/// This is not a switch in the vendor crate: the gate is the literal `inclination >= 0.2`
/// on the perturbed value in `sgp4`'s `deep_space.rs`, with no STR#3 arm to select. No
/// `Convention` here reaches it.
///
/// **The node reduction.** *Inside* that low-inclination arm, the node `p₂₂` enters the ω
/// expression once more, reduced modulo 2π — by `rem_euclid` on the crate's
/// "AFSPC compatibility" propagation path and by a truncating `%` otherwise. The two agree
/// unless the node is negative, so the switch is confined to deep-space orbits below
/// 0.2 rad of perturbed inclination and is invisible everywhere else. This one *is* a
/// choice: [`Convention::AfspcNodeReduction`].
///
/// Vallado settles the first two in *Revisiting Spacetrack Report #3* (AIAA 2006-6753,
/// rev. 3). On the constants: *"We use WGS-72 as the default value … it will cause a
/// difference in ephemeris results."* And on the published verification output, Appendix E:
///
/// > Note that this version includes the results of the Lyddane choice using the
/// > **perturbed inclination as indicated in the GSFC code**, and also uses the
/// > **WGS-72 constants**. … The only differences from the original paper is satellite
/// > **23599** … These test case results were run using the **'a', '72' options**.
///
/// So: WGS-72 from the first, GSFC from the second. That combination is
/// [`Convention::Reference`], and it is the default.
///
/// "AFSPC compatible" therefore does not mean "matches the reference implementation".
/// `propagate_afspc_compatibility_mode` moves satellite 23599 by 918 m in seven hours
/// and a kilometre a day — measured, in `tests/sgp4_verification.rs`. Vallado names the
/// same satellite as the only one whose published numbers moved.
///
/// Vallado writes that the operational code's *"modulo 2π reduction of the XNODE variable"*
/// is defective, and it is tempting to read that as the third item above — but `gpredict`
/// and `PreviSat` both reduce the node Euclidean-ly *before* the expression, and `gpredict`
/// agrees with the published output to 19 m. So the mechanism is not established here, only
/// the measurement. See `notes/10-previsat.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Convention {
    /// WGS-72 gravity, AFSPC epoch and sidereal expressions, GSFC Lyddane branch.
    ///
    /// Reproduces `tcppver.out`, `python-sgp4` and Skyfield. The published
    /// verification set holds 31 loadable element sets, 30 of which the reference
    /// answers for at least once and which are therefore comparable at all.
    #[default]
    Reference,
    /// As [`Self::Reference`], but with the vendor crate's "AFSPC compatibility"
    /// propagation path.
    ///
    /// The variant is named for what that path changes, which is one modulo: the node
    /// is reduced by `rem_euclid` here and by a truncating `%` in [`Self::Reference`].
    /// They differ only where that angle is negative, and only inside the
    /// low-inclination Lyddane arm of the deep-space correction. It does *not* select
    /// the Spacetrack Report #3 Lyddane arm — the crate has no such arm. See the
    /// discussion on [`Convention`] for the measurement.
    AfspcNodeReduction,
    /// WGS-84 gravity and the IAU sidereal-time expression.
    ///
    /// More modern, and inconsistent with every published element set. Available
    /// because someone will one day fit elements this way.
    Iau,
}

/// The deep-space boundary: a period of 225 minutes, expressed in rad/min.
///
/// SGP4 branches on the *period*, not on the altitude, and the period it branches on
/// is the one implied by the Brouwer mean motion `n₀"`, not the Kozai `n₀` a TLE
/// prints. The two differ by a few parts in `10⁵`, which is enough to move an object
/// across a boundary it sits close to.
const DEEP_SPACE_MEAN_MOTION: f64 = core::f64::consts::TAU / 225.0;

/// Lower edge of the 24-hour resonance window, rad/min. Strict.
const SYNCHRONOUS_LO: f64 = 0.003_490_658_5;

/// Upper edge of the 24-hour resonance window, rad/min. Strict.
const SYNCHRONOUS_HI: f64 = 0.005_235_987_7;

/// Lower edge of the 12-hour resonance window, rad/min. Inclusive.
const HALF_DAY_LO: f64 = 8.26e-3;

/// Upper edge of the 12-hour resonance window, rad/min. Inclusive.
const HALF_DAY_HI: f64 = 9.24e-3;

/// The 12-hour window also demands an eccentric orbit: a Molniya, not a circular
/// twice-a-day satellite.
const HALF_DAY_MIN_ECCENTRICITY: f64 = 0.5;

/// Which branch of the model an element set falls in.
///
/// SGP4 is two algorithms behind one name. Above a 225-minute period the near-Earth
/// series suffice; below it the lunisolar periodics and, for two narrow bands of mean
/// motion, a numerically integrated geopotential resonance are added. Which branch an
/// object takes determines its cost, its error growth, and whether the propagator
/// carries state at all — so it is worth being able to ask.
///
/// The boundaries are the published ones, applied to the Brouwer mean motion `n₀"`
/// recovered from the element set, and to the element set's eccentricity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    /// Period below \\(225\\) min: the near-Earth series, no lunisolar terms.
    NearEarth,
    /// Deep space, but outside both resonance windows: lunisolar periodics only.
    DeepSpaceNonResonant,
    /// Deep space, geosynchronous: `0.0034906585 < n₀" < 0.0052359877` rad/min.
    ///
    /// The resonance argument is \\(\lambda = M + \Omega + \omega - \theta\\).
    DeepSpaceSynchronous,
    /// Deep space, Molniya: `8.26e-3 ≤ n₀" ≤ 9.24e-3` rad/min **and** `e₀ ≥ 0.5`.
    ///
    /// The resonance argument is \\(\lambda = M + 2\Omega - 2\theta\\).
    DeepSpaceHalfDay,
}

impl Regime {
    /// Whether this regime runs the resonance integrator.
    ///
    /// True for exactly the two resonant deep-space regimes. These are the objects
    /// whose propagation cost grows with the distance from epoch, because the
    /// integrator takes fixed 720-minute steps from the epoch every time.
    #[must_use]
    pub const fn is_resonant(self) -> bool {
        matches!(self, Self::DeepSpaceSynchronous | Self::DeepSpaceHalfDay)
    }

    /// Whether this regime takes the deep-space branch at all.
    #[must_use]
    pub const fn is_deep_space(self) -> bool {
        !matches!(self, Self::NearEarth)
    }

    /// Classify from the Brouwer mean motion in rad/min and the eccentricity.
    fn classify(brouwer_mean_motion: f64, eccentricity: f64) -> Self {
        if brouwer_mean_motion > DEEP_SPACE_MEAN_MOTION {
            return Self::NearEarth;
        }
        // The two windows are disjoint — 0.00524 < 0.00826 — so the order of the two
        // tests cannot matter. The published code tests them in this order anyway.
        if (HALF_DAY_LO..=HALF_DAY_HI).contains(&brouwer_mean_motion)
            && eccentricity >= HALF_DAY_MIN_ECCENTRICITY
        {
            Self::DeepSpaceHalfDay
        } else if brouwer_mean_motion > SYNCHRONOUS_LO && brouwer_mean_motion < SYNCHRONOUS_HI {
            Self::DeepSpaceSynchronous
        } else {
            Self::DeepSpaceNonResonant
        }
    }
}

impl fmt::Display for Regime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NearEarth => write!(f, "near-earth"),
            Self::DeepSpaceNonResonant => write!(f, "deep-space non-resonant"),
            Self::DeepSpaceSynchronous => write!(f, "deep-space synchronous"),
            Self::DeepSpaceHalfDay => write!(f, "deep-space half-day"),
        }
    }
}

/// One satellite, with its per-object constants already precomputed.
///
/// SGP4 splits into a costly initialisation from the element set and a cheap
/// evaluation per epoch. Keeping them apart is what makes propagating one object
/// over 10⁵ epochs affordable, and it is why this is a struct and not a function.
#[derive(Debug, Clone)]
pub struct Satellite {
    norad_id: NoradId,
    /// The element-set epoch, on the naive UTC coordinate. See the crate docs.
    epoch_naive_ns: i128,
    constants: sgp4::Constants,
    convention: Convention,
    /// The Brouwer mean motion `n₀"`, rad/min — the un-Kozai'd value the model runs on.
    brouwer_mean_motion: f64,
    regime: Regime,
}

impl Satellite {
    /// Build from the two lines of a TLE.
    ///
    /// # Errors
    ///
    /// [`PropagationError::MalformedElements`] if the lines do not parse,
    /// [`PropagationError::UnusableElements`] if SGP4 rejects them.
    pub fn from_tle(line1: &str, line2: &str) -> Result<Self, PropagationError> {
        for line in [line1, line2] {
            if !checksum_is_valid(line) {
                return Err(PropagationError::BadChecksum);
            }
        }
        Self::from_tle_unchecked(line1, line2)
    }

    /// Build from a TLE without verifying its checksums.
    ///
    /// Two of the published verification element sets fail their checksum, and one
    /// of them is the historic Spacetrack Report #3 case. A catalog of ten thousand
    /// real objects will contain more. Skipping the check is a decision, so it has
    /// its own function.
    ///
    /// # Errors
    ///
    /// [`PropagationError::MalformedElements`] if the lines do not parse,
    /// [`PropagationError::UnusableElements`] if SGP4 rejects them.
    pub fn from_tle_unchecked(line1: &str, line2: &str) -> Result<Self, PropagationError> {
        // The underlying parser validates the checksum itself and has no way to be
        // told not to. Recompute the digit so it accepts the line: the caller has
        // already said it does not care, and the checksum feeds nothing downstream.
        let (r1, r2) = (repair_checksum(line1), repair_checksum(line2));
        let elements = sgp4::Elements::from_tle(None, r1.as_bytes(), r2.as_bytes())
            .map_err(|_| PropagationError::MalformedElements)?;
        Self::from_elements(&elements, Convention::default())
    }

    /// Build from a parsed element set, whether from a TLE or from OMM.
    ///
    /// # Errors
    ///
    /// [`PropagationError::UnusableElements`] if SGP4 rejects the elements, or the
    /// epoch cannot be expressed on the Gregorian calendar.
    pub fn from_elements(
        elements: &sgp4::Elements,
        convention: Convention,
    ) -> Result<Self, PropagationError> {
        use sgp4::chrono::{Datelike, Timelike};

        let dt = elements.datetime;
        let seconds = f64::from(dt.second()) + f64::from(dt.nanosecond()) * 1e-9;

        #[allow(clippy::cast_possible_truncation)] // chrono bounds every field
        let epoch = UtcDateTime::new(
            dt.year(),
            dt.month() as u8,
            dt.day() as u8,
            dt.hour() as u8,
            dt.minute() as u8,
            seconds,
        )
        .map_err(|_| PropagationError::UnusableElements)?;

        // The gravity model is part of the convention, and the un-Kozai transformation
        // reads J₂ and kₑ from it — so the regime must be classified under the same
        // model the constants are built with, not under a fixed one.
        let geopotential = match convention {
            Convention::Reference | Convention::AfspcNodeReduction => &sgp4::WGS72,
            Convention::Iau => &sgp4::WGS84,
        };

        let constants = match convention {
            Convention::Reference | Convention::AfspcNodeReduction => {
                sgp4::Constants::from_elements_afspc_compatibility_mode(elements)
            }
            Convention::Iau => sgp4::Constants::from_elements(elements),
        }
        .map_err(|_| PropagationError::UnusableElements)?;

        let orbit = sgp4::Orbit::from_kozai_elements(
            geopotential,
            elements.inclination * DEG_TO_RAD,
            elements.right_ascension * DEG_TO_RAD,
            elements.eccentricity,
            elements.argument_of_perigee * DEG_TO_RAD,
            elements.mean_anomaly * DEG_TO_RAD,
            elements.mean_motion * REV_PER_DAY_TO_RAD_PER_MIN,
        )
        .map_err(|_| PropagationError::UnusableElements)?;

        Ok(Self {
            norad_id: NoradId::new(elements.norad_id.try_into().unwrap_or(u32::MAX)),
            epoch_naive_ns: epoch.naive_ns_since_j2000(),
            constants,
            convention,
            brouwer_mean_motion: orbit.mean_motion,
            regime: Regime::classify(orbit.mean_motion, elements.eccentricity),
        })
    }

    /// The catalog number.
    #[must_use]
    pub const fn norad_id(&self) -> NoradId {
        self.norad_id
    }

    /// The convention this satellite is propagated under.
    #[must_use]
    pub const fn convention(&self) -> Convention {
        self.convention
    }

    /// Which branch of the model this element set takes.
    ///
    /// Determined once, at construction, from the Brouwer mean motion and the
    /// eccentricity. See [`Regime`].
    #[must_use]
    pub const fn regime(&self) -> Regime {
        self.regime
    }

    /// The Brouwer mean motion `n₀"`, radians per minute.
    ///
    /// Not the mean motion the TLE prints. That one is Kozai's, and the first thing
    /// SGP4 does is undo the Kozai transformation to recover this. Every branch
    /// boundary in the model is expressed against *this* quantity.
    #[must_use]
    pub const fn brouwer_mean_motion(&self) -> f64 {
        self.brouwer_mean_motion
    }

    /// The element-set epoch, as the UTC label the elements carry.
    ///
    /// The instant [`Self::propagate_minutes`] measures from, and the inverse of
    /// [`Self::minutes_since_epoch`]. Anything that has to place a propagation on a
    /// time scale needs it: an Earth-rotation angle is a function of UT1, and the only
    /// clock a TLE offers is this one.
    #[must_use]
    pub fn epoch(&self) -> UtcDateTime {
        // The naive coordinate has no leap-second labels in it by construction, so the
        // extra-second flag is false for every epoch that can round-trip.
        UtcDateTime::from_naive_ns_since_j2000(self.epoch_naive_ns, false)
    }

    /// Minutes from the element-set epoch to a UTC label, on the naive coordinate.
    ///
    /// The subtraction happens in exact integer nanoseconds and only then becomes a
    /// float, so no epoch-sized quantity ever enters the mantissa beside a
    /// microsecond-sized one.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // the difference is small; the epochs never are
    pub fn minutes_since_epoch(&self, utc: UtcDateTime) -> f64 {
        let delta_ns = utc.naive_ns_since_j2000() - self.epoch_naive_ns;
        (delta_ns as f64) / NS_PER_MINUTE
    }

    /// Propagate to one epoch.
    ///
    /// # Errors
    ///
    /// [`PropagationError::Diverged`] if SGP4 cannot carry the object there.
    pub fn propagate_at(&self, utc: UtcDateTime) -> Result<StateVector<Teme>, PropagationError> {
        self.propagate_minutes(self.minutes_since_epoch(utc))
    }

    /// Propagate by a signed offset in minutes from the element-set epoch.
    ///
    /// SGP4's native parameter, and the one the published verification cases are
    /// expressed in. Negative offsets propagate backwards, which the model permits
    /// and the verification set exercises.
    ///
    /// # Errors
    ///
    /// [`PropagationError::Diverged`] if SGP4 cannot carry the object there: the
    /// mean eccentricity left `[0, 1)`, the semi-latus rectum went negative, or the
    /// object decayed. The verification set contains cases designed to do each.
    pub fn propagate_minutes(&self, minutes: f64) -> Result<StateVector<Teme>, PropagationError> {
        let minutes = sgp4::MinutesSinceEpoch(minutes);
        let p = match self.convention {
            // Note: `propagate`, not `propagate_afspc_compatibility_mode`. The latter
            // changes how the node is reduced modulo 2π — not which Lyddane arm runs —
            // and no published verification output uses it.
            Convention::Reference | Convention::Iau => self.constants.propagate(minutes),
            Convention::AfspcNodeReduction => {
                self.constants.propagate_afspc_compatibility_mode(minutes)
            }
        }
        .map_err(|_| PropagationError::Diverged)?;

        let position = Vec3::new(p.position[0], p.position[1], p.position[2]).scale(METRES_PER_KM);

        // Vallado's re-entry criterion, which the crate omits. Without it a decayed
        // object is reported at a plausible-looking point inside the Earth.
        if position.norm() < WGS72_EQUATORIAL_RADIUS_M {
            return Err(PropagationError::Decayed);
        }

        Ok(StateVector::new(
            position,
            Vec3::new(p.velocity[0], p.velocity[1], p.velocity[2]).scale(METRES_PER_KM),
        ))
    }

    /// Propagate to many epochs, writing into a caller-provided slice.
    ///
    /// # Panics
    ///
    /// If `out` is shorter than `times`.
    pub fn propagate_into(
        &self,
        times: &[UtcDateTime],
        out: &mut [Result<StateVector<Teme>, PropagationError>],
    ) {
        assert!(
            out.len() >= times.len(),
            "output slice holds {} of {} epochs",
            out.len(),
            times.len()
        );
        for (t, slot) in times.iter().zip(out.iter_mut()) {
            *slot = self.propagate_at(*t);
        }
    }
}

/// Rewrite a line's checksum digit to whatever its contents imply.
fn repair_checksum(line: &str) -> String {
    if line.len() < 69 {
        return line.to_owned();
    }
    let mut bytes = line.as_bytes().to_vec();
    let sum: u32 = bytes[..68]
        .iter()
        .map(|&b| match b {
            b'0'..=b'9' => u32::from(b - b'0'),
            b'-' => 1,
            _ => 0,
        })
        .sum();
    #[allow(clippy::cast_possible_truncation)] // sum % 10 is a single digit
    {
        bytes[68] = b'0' + (sum % 10) as u8;
    }
    String::from_utf8(bytes).unwrap_or_else(|_| line.to_owned())
}

/// The TLE checksum: digits summed modulo ten, with a minus sign counting as one.
///
/// Published with the format itself. Column 69 holds the result; everything else in
/// the line contributes.
fn checksum_is_valid(line: &str) -> bool {
    let bytes = line.as_bytes();
    if bytes.len() < 69 {
        return false;
    }
    let sum: u32 = bytes[..68]
        .iter()
        .map(|&b| match b {
            b'0'..=b'9' => u32::from(b - b'0'),
            b'-' => 1,
            _ => 0,
        })
        .sum();
    bytes[68].is_ascii_digit() && sum % 10 == u32::from(bytes[68] - b'0')
}

/// Propagate every satellite to every epoch.
///
/// # Layout
///
/// `out[s * times.len() + t]` — **satellite-major**. Each satellite's time series
/// is contiguous, which is what the downstream loops want:
///
/// - a coverage engine walks one object's track looking for sign changes in `g(t)`,
///   and wants those samples adjacent;
/// - a parallel backend splits on the satellite axis, where there is no shared state
///   at all;
/// - a GPU block would take one satellite and stride through its epochs.
///
/// The transpose — epoch-major — would suit a frame table shared across satellites,
/// but the frame table is hoisted out of this loop entirely
/// (`plans/08-parallelism.md`), so there is nothing left to share.
///
/// # Failure is per-cell
///
/// A decayed object yields `Err` in its own slots and does not spoil the batch. A
/// catalog of ten thousand objects always contains a few that SGP4 cannot carry.
///
/// # Panics
///
/// If `out` is shorter than `satellites.len() * times.len()`.
pub fn propagate_many_into(
    satellites: &[Satellite],
    times: &[UtcDateTime],
    out: &mut [Result<StateVector<Teme>, PropagationError>],
) {
    let needed = satellites.len() * times.len();
    assert!(
        out.len() >= needed,
        "output slice holds {} of {needed} cells",
        out.len()
    );

    let t = times.len();
    // Each object is independent (its own output row), so partition the catalogue across scoped
    // threads — embarrassingly parallel, exact, order-preserving. Small batches stay sequential:
    // below ~2 objects per core the spawn overhead outweighs the work (`Satellite` is `Sync`, so the
    // shared `satellites`/`times` slices need no copy). Dep-free, mirroring `spacelab_coverage`.
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    if threads <= 1 || satellites.len() < 2 * threads || t == 0 {
        for (satellite, row) in satellites.iter().zip(out.chunks_mut(t.max(1))) {
            satellite.propagate_into(times, row);
        }
        return;
    }
    let per = satellites.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (sat_chunk, out_chunk) in satellites.chunks(per).zip(out.chunks_mut(per * t)) {
            scope.spawn(move || {
                for (satellite, row) in sat_chunk.iter().zip(out_chunk.chunks_mut(t)) {
                    satellite.propagate_into(times, row);
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{DEEP_SPACE_MEAN_MOTION, Regime, SYNCHRONOUS_HI, SYNCHRONOUS_LO};

    /// Which side of each published inequality the boundary itself falls on.
    ///
    /// `Regime` is otherwise reached only through real element sets, and no catalogue object
    /// lands on a boundary to the bit — so all three strict comparisons could be non-strict
    /// without a test noticing. SPACETRACK Report No. 3 puts a period of exactly 225 minutes on
    /// the deep-space side of the branch, and treats the synchronous window as open at both ends.
    #[test]
    fn every_regime_boundary_belongs_to_the_side_the_published_test_puts_it_on() {
        assert_eq!(
            Regime::classify(DEEP_SPACE_MEAN_MOTION, 0.0),
            Regime::DeepSpaceNonResonant,
            "a period of exactly 225 minutes takes the deep-space branch",
        );
        assert_eq!(
            Regime::classify(SYNCHRONOUS_LO, 0.0),
            Regime::DeepSpaceNonResonant,
            "the synchronous window is open at its lower end",
        );
        assert_eq!(
            Regime::classify(SYNCHRONOUS_HI, 0.0),
            Regime::DeepSpaceNonResonant,
            "the synchronous window is open at its upper end",
        );
        assert_eq!(
            Regime::classify(f64::midpoint(SYNCHRONOUS_LO, SYNCHRONOUS_HI), 0.0),
            Regime::DeepSpaceSynchronous,
            "the interior of the window is resonant",
        );
    }
}
