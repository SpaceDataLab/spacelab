//! Deep-space radiometric observables: DSN sequential range in range units, n-way averaged Doppler
//! and the measured frequency that complements it, ΔDOR against a quasar, and the solar-corona delay
//! all of them carry.
//!
//! [`measurement`](crate::measurement) and [`multistatic`](crate::multistatic) compute a geometry —
//! metres, metres per second, seconds of arrival difference. A DSN tracking station reports neither.
//! It reports **range units**, a phase count of the ranging code whose scale is set by the uplink
//! carrier and which wraps at a power of two, and a **Doppler count** averaged over a compression
//! interval, referred to a reference frequency the receiver was configured with. Turning one into
//! the other is not a unit conversion: both observables are integrals of the *transmitter* frequency
//! over an interval whose endpoints the orbit itself determines, and the transmitter is ramped.
//!
//! [`delta_dor`] is the one deep-space observable that carries plane-of-sky information rather than
//! line-of-sight: an arrival-time difference across an intercontinental baseline, differenced again
//! against a quasar so that the station clocks and the atmosphere above each antenna cancel.
//! [`solar_corona_delay`] is the correction none of them survives without near a superior
//! conjunction, where the plasma column adds tens of metres at X-band.
//!
//! # The formulation
//!
//! Moyer, *Formulation for Observed and Computed Values of Deep Space Network Data Types for
//! Navigation*, JPL Deep-Space Communications and Navigation Series Monograph 2 (2000), §13.3 —
//! eq. 13-52 and 13-53 for the interval endpoints, eq. 13-54 for both observables.
//!
//! Sequential range, with `d` the uplink band's denominator from table 13-1 and `C` the lowest
//! ranging component:
//!
//! ```text
//! RU = (221 / 2d) ∫ f_T(t) dt   (mod 2^(C+6)),   over [t_transmit, t_receive]
//! ```
//!
//! Averaged Doppler, with `M` the turnaround ratio and `T_c` the count interval:
//!
//! ```text
//! observed = M(reference uplink, downlink) f_reference − (M(uplink, downlink) / T_c) ∫ f_T(t) dt
//! ```
//!
//! taken between the two transmission epochs that correspond to the ends of the count interval.
//!
//! # What this module owns and what it does not
//!
//! It owns the frequency algebra: the band ratios, the exact integral of a ramped exciter, the
//! modulus, and the two observables. It does **not** solve light time and it does not convert
//! timescales. The endpoints arrive as [`Scalar`]s the caller has already solved
//! ([`light_time`](crate::light_time)), for the same reason [`multistatic`](crate::multistatic)
//! takes its epochs rather than deriving them — and because that is what makes the observable
//! differentiable end to end: the state enters only through those endpoints, and
//! `d/dt ∫f_T = f_T(t)` is carried by the dual arithmetic rather than by a hand-written partial.
//!
//! One consequence is worth stating rather than discovering. Moyer's integral is over **UTC**,
//! because that is the timescale the exciter's ramp table is written in, while the light-time
//! endpoints are natural in a dynamical scale. This module integrates in whatever scale the ramp
//! table and the endpoints share; keeping them the same scale is the caller's job, and a caller who
//! mixes them gets an error proportional to `f_T × ΔTDB−UTC`, which at 7 GHz and 32 s is 2.2e11
//! cycles rather than a rounding term.

use spacelab_autodiff::Scalar;

use crate::light_time::SPEED_OF_LIGHT;
use crate::multistatic::{TdoaGeometry, tdoa};

/// Why a radiometric observable could not be formed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RadiometricError {
    /// A ramp table with no segments. Every observable here is an integral over the table, and an
    /// empty table makes that integral silently zero rather than undefined.
    EmptyRampTable,
    /// A segment that ends no later than it starts, or that starts before its predecessor ends.
    /// Reported at construction, because an overlapping table would let one epoch resolve to two
    /// frequencies depending on which segment the search happened to land in.
    RampTableNotOrdered {
        /// The index of the offending segment.
        index: usize,
    },
    /// An epoch outside the span the table covers. The exciter frequency before the first ramp and
    /// after the last is not zero and is not the nearest ramp extrapolated — it is unknown, and a
    /// silent extrapolation of a ramp rate is the one error this class of table invites.
    EpochOutsideRampTable {
        /// The epoch asked for.
        epoch: f64,
        /// The first epoch the table covers.
        first: f64,
        /// The last epoch the table covers.
        last: f64,
    },
    /// An integration interval that runs backwards, or a count interval that is not positive.
    NonPositiveInterval {
        /// The interval asked for, seconds.
        seconds: f64,
    },
    /// A ranging component whose modulus `2^(C+6)` is not a finite positive number.
    UnrepresentableModulus {
        /// The lowest ranging component asked for.
        component: i32,
    },
    /// An electron-density series with no terms. A corona correction of exactly zero is what a
    /// caller gets by not applying the correction, so an empty series is a construction mistake
    /// rather than a way to spell "no plasma".
    EmptyCoronaSeries,
    /// A series exponent that is not positive and finite. `r^-k` with `k ≤ 0` is a density that does
    /// not fall off with distance, which the whole integral's convergence rests on.
    NonPositiveExponent {
        /// The exponent asked for.
        exponent: f64,
    },
    /// A carrier frequency that is not positive and finite. The plasma delay goes as `1/f²`, so a
    /// zero frequency is an infinite correction rather than a large one.
    NonPositiveFrequency {
        /// The frequency asked for, hertz.
        hertz: f64,
    },
    /// A line of sight whose perpendicular distance from the Sun is inside the photosphere.
    LineOfSightInsideTheSun {
        /// The perpendicular distance from the Sun's centre to the line, metres.
        impact_parameter: f64,
    },
}

impl core::fmt::Display for RadiometricError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyRampTable => write!(f, "a ramp table with no segments spans no epoch"),
            Self::RampTableNotOrdered { index } => {
                write!(f, "ramp segment {index} is not after the one before it")
            }
            Self::EpochOutsideRampTable { epoch, first, last } => write!(
                f,
                "epoch {epoch} is outside the ramp table's span [{first}, {last}]"
            ),
            Self::NonPositiveInterval { seconds } => {
                write!(f, "an interval of {seconds} s is not positive")
            }
            Self::UnrepresentableModulus { component } => write!(
                f,
                "lowest ranging component {component} gives a modulus 2^({component}+6) that is not finite and positive"
            ),
            Self::EmptyCoronaSeries => {
                write!(f, "an electron-density series with no terms is not a model")
            }
            Self::NonPositiveExponent { exponent } => write!(
                f,
                "electron-density exponent {exponent} is not positive and finite"
            ),
            Self::NonPositiveFrequency { hertz } => {
                write!(f, "carrier frequency {hertz} Hz is not positive and finite")
            }
            Self::LineOfSightInsideTheSun { impact_parameter } => write!(
                f,
                "the line of sight passes {impact_parameter} m from the Sun's centre, inside the photosphere"
            ),
        }
    }
}

impl core::error::Error for RadiometricError {}

/// A DSN frequency band.
///
/// The three the deep-space network transmits on. Ku appears in Tudat's enumeration but has no
/// turnaround ratio in Moyer's table, so it is not admitted here rather than admitted and rejected
/// at every use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// S-band.
    S,
    /// X-band.
    X,
    /// Ka-band.
    Ka,
}

impl Band {
    /// The band's denominator when it is the *uplink*, Moyer (2000) table 13-1.
    #[must_use]
    pub const fn uplink_denominator(self) -> f64 {
        match self {
            Self::S => 221.0,
            Self::X => 749.0,
            Self::Ka => 3599.0,
        }
    }

    /// The band's numerator when it is the *downlink*, Moyer (2000) table 13-1.
    #[must_use]
    pub const fn downlink_numerator(self) -> f64 {
        match self {
            Self::S => 240.0,
            Self::X => 880.0,
            Self::Ka => 3344.0,
        }
    }

    /// The transponder turnaround ratio `M` for this band uplinked and `downlink` returned.
    ///
    /// A ratio of two small integers, so it is formed as one division of exact values rather than
    /// stored as a rounded decimal per pair — nine pairs, one expression, and the four-digit
    /// integers are exact in `f64`.
    #[must_use]
    pub const fn turnaround_ratio(self, downlink: Self) -> f64 {
        downlink.downlink_numerator() / self.uplink_denominator()
    }

    /// The factor that turns the transmitter-frequency integral into range units.
    ///
    /// Moyer's three cases — `1/2` for S, `221/(2·749)` for X, `221/(2·3599)` for Ka — are one
    /// expression, `221 / 2d`, and writing it as three constants is what lets two of them drift
    /// apart from the third.
    #[must_use]
    pub const fn range_conversion_factor(self) -> f64 {
        Self::S.uplink_denominator() / (2.0 * self.uplink_denominator())
    }
}

/// The Cassini transponder's Ka-band turnaround ratio, which is not the table 13-1 value.
///
/// Moyer (2000) §13.2.2. Present because a spacecraft-specific ratio is exactly the kind of value a
/// caller would otherwise hard-code at a call site.
pub const CASSINI_KA_TURNAROUND_RATIO: f64 = 14.0 / 15.0;

/// One linear ramp of a station's exciter: a frequency and the rate it climbs at, over an interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RampSegment {
    /// The epoch the ramp begins, in the table's own timescale.
    pub start: f64,
    /// The epoch the ramp ends.
    pub end: f64,
    /// The transmitted frequency at [`start`](Self::start), hertz.
    pub start_frequency: f64,
    /// The rate the frequency climbs at, hertz per second.
    pub rate: f64,
}

impl RampSegment {
    /// The transmitted frequency at `epoch`, extrapolating the ramp — private because extrapolating
    /// one segment is only ever correct inside that segment, which [`RampTable`] is what enforces.
    fn frequency_at<S: Scalar>(&self, epoch: S) -> S {
        S::from_f64(self.start_frequency)
            + S::from_f64(self.rate) * (epoch - S::from_f64(self.start))
    }

    /// The exact integral of this ramp between two epochs inside it.
    ///
    /// Linear in `t`, so the trapezoid is not an approximation: `∫ (f₀ + r(t−t₀)) dt` over a span
    /// `Δ` beginning at `a` is `Δ (f(a) + rΔ/2)`.
    fn integral<S: Scalar>(&self, from: S, to: S) -> S {
        let span = to - from;
        let at_from = S::from_f64(self.start_frequency)
            + S::from_f64(self.rate) * (from - S::from_f64(self.start));
        span * (at_from + S::from_f64(self.rate) * span / S::from_f64(2.0))
    }
}

/// A station's ramp table, checked once so that every epoch inside it resolves to one frequency.
///
/// The segments are borrowed rather than owned: the caller holds the table it parsed out of an ODF
/// or a station schedule, and this crate is `no_std` without an allocator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RampTable<'a> {
    segments: &'a [RampSegment],
}

impl<'a> RampTable<'a> {
    /// Validate a ramp table.
    ///
    /// # Errors
    ///
    /// [`RadiometricError::EmptyRampTable`] if there are no segments, and
    /// [`RadiometricError::RampTableNotOrdered`] if any segment ends no later than it starts or
    /// begins before its predecessor ends. Gaps are permitted — a station that stopped transmitting
    /// has a gap, and an epoch inside one is reported when it is asked for rather than forbidden
    /// when the table is built.
    pub fn new(segments: &'a [RampSegment]) -> Result<Self, RadiometricError> {
        if segments.is_empty() {
            return Err(RadiometricError::EmptyRampTable);
        }
        for (index, segment) in segments.iter().enumerate() {
            if segment.end <= segment.start
                || !segment.start.is_finite()
                || !segment.end.is_finite()
            {
                return Err(RadiometricError::RampTableNotOrdered { index });
            }
            if index > 0 && segment.start < segments[index - 1].end {
                return Err(RadiometricError::RampTableNotOrdered { index });
            }
        }
        Ok(Self { segments })
    }

    /// The segments, in the order they were given.
    #[must_use]
    pub const fn segments(&self) -> &'a [RampSegment] {
        self.segments
    }

    /// The first epoch the table covers.
    #[must_use]
    pub fn first(&self) -> f64 {
        self.segments[0].start
    }

    /// The last epoch the table covers.
    #[must_use]
    pub fn last(&self) -> f64 {
        self.segments[self.segments.len() - 1].end
    }

    /// The segment containing `epoch`, or the error naming the span that does not.
    ///
    /// Half-open, `[start, end)`, with the final `end` admitted as the one closed endpoint. Two
    /// things depend on that and neither is a preference. A ramp table is *discontinuous* where one
    /// segment ends and the next begins — the fixture in this module's tests steps from 13 Hz to
    /// 10 Hz at `t = 9` — so an epoch on a shared boundary belongs to the segment that is starting,
    /// which is the frequency the exciter is actually transmitting from that instant. And the walk
    /// in [`integral`](Self::integral) advances its cursor to a segment's `end`; resolving that
    /// cursor back to the segment it just finished would return a zero-length overlap and never
    /// advance again.
    // The exact `==` is the point: the table's last instant is covered and the next representable
    // one is not. A tolerance here would extrapolate the final ramp past the epoch the station
    // stopped transmitting at, which is the one failure this whole lookup exists to refuse.
    #[allow(clippy::float_cmp)]
    fn segment_at(&self, epoch: f64) -> Result<&'a RampSegment, RadiometricError> {
        self.segments
            .iter()
            .find(|s| epoch >= s.start && epoch < s.end)
            .or_else(|| self.segments.last().filter(|s| epoch == s.end))
            .ok_or_else(|| self.outside(epoch))
    }

    /// The refusal for an epoch the table does not cover, naming the span that does.
    fn outside(&self, epoch: f64) -> RadiometricError {
        RadiometricError::EpochOutsideRampTable {
            epoch,
            first: self.first(),
            last: self.last(),
        }
    }

    /// The transmitted frequency at `epoch`, hertz.
    ///
    /// # Errors
    ///
    /// [`RadiometricError::EpochOutsideRampTable`] if no segment covers `epoch`.
    pub fn frequency_at<S: Scalar>(&self, epoch: S) -> Result<S, RadiometricError> {
        Ok(self.segment_at(epoch.value())?.frequency_at(epoch))
    }

    /// The integral of the transmitted frequency from `from` to `to`, cycles.
    ///
    /// Exact rather than quadratured: every segment is linear, so the sum of per-segment trapezoids
    /// is the integral itself. Generic over [`Scalar`] in the *endpoints* — the segment data is
    /// `f64` because a ramp table is a measured input, while the endpoints carry the state's
    /// derivatives, and `d/dt ∫ f_T = f_T(t)` then arrives without a hand-written partial.
    ///
    /// # Errors
    ///
    /// [`RadiometricError::NonPositiveInterval`] if `to` is not after `from`, and
    /// [`RadiometricError::EpochOutsideRampTable`] if either endpoint, or any gap between them,
    /// falls outside a segment.
    pub fn integral<S: Scalar>(&self, from: S, to: S) -> Result<S, RadiometricError> {
        let (a, b) = (from.value(), to.value());
        // Finiteness is checked alongside the ordering rather than left to the comparison: `b > a` is
        // false for a NaN endpoint, so the ordering test alone would report the interval as
        // non-positive and hide which of the two was not a number.
        if b <= a || !a.is_finite() || !b.is_finite() {
            return Err(RadiometricError::NonPositiveInterval { seconds: b - a });
        }
        let mut total = S::from_f64(0.0);
        let mut covered = from;
        // Driven by the segment list, not by a cursor of its own. Each segment the interval touches
        // contributes its exact trapezoid over the overlap, and `covered` records how far that has
        // got so a gap between two segments is reported rather than interpolated across. Making the
        // *iterator* the thing that advances is deliberate: a hand-rolled cursor whose stop
        // condition slips by one — `<` to `<=` — sets `stop` equal to itself and never moves again,
        // which is a hang rather than a wrong answer, and a hang is detected only by a clock.
        for segment in self.segments.iter().filter(|s| s.end > a && s.start < b) {
            if segment.start > covered.value() {
                return Err(self.outside(covered.value()));
            }
            // `to` rather than `S::from_f64(segment.end)` when the interval ends first, and the
            // difference is the derivative: the endpoint carries `d/dto`, the segment's own `end` is
            // a constant. They agree in value whenever both apply.
            let stop = if segment.end < b {
                S::from_f64(segment.end)
            } else {
                to
            };
            total = total + segment.integral(covered, stop);
            covered = stop;
        }
        if covered.value() < b {
            return Err(self.outside(covered.value()));
        }
        Ok(total)
    }
}

/// The modulus a sequential-range observable wraps at, `2^(C+6)` range units.
///
/// # Errors
///
/// [`RadiometricError::UnrepresentableModulus`] if `2^(C+6)` is not finite and positive.
pub fn range_modulus(lowest_ranging_component: i32) -> Result<f64, RadiometricError> {
    let modulus = libm::exp2(f64::from(lowest_ranging_component) + 6.0);
    if modulus.is_finite() && modulus > 0.0 {
        Ok(modulus)
    } else {
        Err(RadiometricError::UnrepresentableModulus {
            component: lowest_ranging_component,
        })
    }
}

/// DSN sequential range, in range units — Moyer (2000) eq. 13-54.
///
/// `transmit` and `receive` are the epochs the ranging code left the transmitting station and
/// arrived at the receiving one, in the ramp table's timescale; the round trip between them is
/// where the orbit enters. `uplink` sets the scale, because the ranging code is a subcarrier of the
/// uplink carrier and a range unit is defined against it.
///
/// The wrap is applied as `x − m⌊x/m⌋` with the floor taken on the *value*, so the observable's
/// derivative passes through the modulus unchanged — which is correct everywhere except exactly at
/// a wrap, where the observable is discontinuous and no derivative exists.
///
/// # Errors
///
/// Whatever [`RampTable::integral`] and [`range_modulus`] report.
pub fn sequential_range<S: Scalar>(
    table: &RampTable<'_>,
    uplink: Band,
    transmit: S,
    receive: S,
    lowest_ranging_component: i32,
) -> Result<S, RadiometricError> {
    let modulus = range_modulus(lowest_ranging_component)?;
    let cycles = table.integral(transmit, receive)?;
    let units = S::from_f64(uplink.range_conversion_factor()) * cycles;
    let wraps = libm::floor(units.value() / modulus);
    Ok(units - S::from_f64(modulus * wraps))
}

/// Which way the ramp integral enters the averaged-Doppler observable.
///
/// A sum type rather than a boolean, because the two are one choice of convention and a `bool`
/// parameter at the call site says nothing about which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DopplerSignature {
    /// The DSN convention and Tudat's default: the ramp integral is *subtracted* from the reference
    /// term, so the observable is the count an ODF records.
    Subtracted,
    /// The ramp integral is added.
    Added,
}

impl DopplerSignature {
    /// The factor the ramp term is multiplied by.
    const fn sign(self) -> f64 {
        match self {
            Self::Subtracted => -1.0,
            Self::Added => 1.0,
        }
    }
}

/// The reference the receiver was configured with: a frequency and the uplink band it belongs to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopplerReference {
    /// The reference frequency, hertz.
    pub frequency: f64,
    /// The uplink band the reference is stated against, which need not be the band actually
    /// transmitted on — that is the whole reason it is carried separately.
    pub band: Band,
}

/// Everything about a Doppler link that does not change within a pass: the two bands, the reference
/// the receiver was configured with, the convention, and the count interval.
///
/// A parameter object rather than five more arguments, and the bands are the reason: `uplink` and
/// `reference.band` are both uplink bands and both of type [`Band`], so as adjacent positional
/// arguments they are silently interchangeable and a caller who swaps them gets a wrong answer with
/// no diagnostic. Named fields make that swap visible at the call site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DopplerLink {
    /// The band actually transmitted on.
    pub uplink: Band,
    /// The band the transponder returns.
    pub downlink: Band,
    /// The reference the receiver was configured with.
    pub reference: DopplerReference,
    /// Which way the ramp integral enters.
    pub signature: DopplerSignature,
    /// The Doppler count interval `T_c`, seconds.
    pub count_interval: f64,
}

/// DSN n-way averaged Doppler — Moyer (2000) eq. 13-52, 13-53 and 13-54.
///
/// `transmit_start` and `transmit_end` are the two transmission epochs corresponding to the ends of
/// the count interval, each already reduced from its reception epoch by that end's own light time
/// (eq. 13-52 and 13-53). They are *not* separated by the count interval: the difference between
/// them is the count interval plus the change in round-trip light time across it, and that
/// difference is the Doppler signature itself.
///
/// # Errors
///
/// [`RadiometricError::NonPositiveInterval`] if the count interval is not positive and finite, and
/// whatever [`RampTable::integral`] reports for the interval between the two transmission epochs.
pub fn averaged_doppler<S: Scalar>(
    table: &RampTable<'_>,
    link: DopplerLink,
    transmit_start: S,
    transmit_end: S,
) -> Result<S, RadiometricError> {
    if link.count_interval <= 0.0 || !link.count_interval.is_finite() {
        return Err(RadiometricError::NonPositiveInterval {
            seconds: link.count_interval,
        });
    }
    let cycles = table.integral(transmit_start, transmit_end)?;
    let reference_term =
        S::from_f64(link.reference.band.turnaround_ratio(link.downlink) * link.reference.frequency);
    let ramp_term = S::from_f64(
        link.signature.sign() * link.uplink.turnaround_ratio(link.downlink) / link.count_interval,
    ) * cycles;
    Ok(reference_term + ramp_term)
}

/// The frequency the station receives on a coherent two-way link, hertz.
///
/// ```text
/// f_received = M(uplink, downlink) · f_T(t_transmit) · dt_T/dt_R
/// ```
///
/// The DSN's *measured frequency* data type, and the instantaneous counterpart of
/// [`averaged_doppler`]: that one integrates the exciter over a compression interval, this one
/// samples it at the transmission epoch and scales by how fast transmitter time runs against
/// receiver time. `transmit_rate` is that ratio, `dt_T/dt_R`, and it is the caller's because it
/// comes out of the light-time solution this module deliberately does not own — it is `1` for a
/// stationary link and `1 − ρ̇/c` to first order for a receding one.
///
/// The two observables are not independent, and the relation is exact rather than asymptotic: over
/// a flat exciter, with the receiver referred to the same band and frequency,
/// `averaged_doppler + measured_frequency = M · f_reference`. The count is what the carrier *lost*
/// against the reference; the measured frequency is what it kept.
///
/// # Errors
///
/// [`RadiometricError::EpochOutsideRampTable`] if the ramp table does not cover `transmit_epoch`.
pub fn measured_frequency<S: Scalar>(
    table: &RampTable<'_>,
    uplink: Band,
    downlink: Band,
    transmit_epoch: S,
    transmit_rate: S,
) -> Result<S, RadiometricError> {
    let transmitted = table.frequency_at(transmit_epoch)?;
    Ok(S::from_f64(uplink.turnaround_ratio(downlink)) * transmitted * transmit_rate)
}

/// A radio source in the ICRF, by its catalogue coordinates.
///
/// Right ascension and declination rather than a direction vector, because the only thing a ΔDOR
/// observable needs from the source is a *unit* vector and a caller handed a three-vector can pass
/// one that is not. The catalogue publishes angles; this takes what the catalogue publishes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quasar {
    /// Right ascension, radians.
    pub right_ascension: f64,
    /// Declination, radians.
    pub declination: f64,
}

impl Quasar {
    /// The unit vector towards the source, in the equatorial frame the states are expressed in.
    #[must_use]
    pub fn direction(self) -> [f64; 3] {
        let (sin_dec, cos_dec) = (libm::sin(self.declination), libm::cos(self.declination));
        let (sin_ra, cos_ra) = (
            libm::sin(self.right_ascension),
            libm::cos(self.right_ascension),
        );
        [cos_dec * cos_ra, cos_dec * sin_ra, sin_dec]
    }
}

/// Delta differential one-way range, seconds — the spacecraft's arrival-time difference across the
/// baseline minus the quasar's.
///
/// ```text
/// ΔDOR = (ρ₂ − ρ₁)/c + (B · ŝ)/c,   B = r₂ − r₁
/// ```
///
/// The first term is [`multistatic::tdoa`](crate::multistatic::tdoa), the arrival-time difference
/// for a signal from a source at finite range. The second is the same difference for a source at
/// infinity: a plane wave reaches the station displaced *towards* it earlier, by `(B · ŝ)/c`, so
/// subtracting the quasar's DOR adds it.
///
/// # Why the quasar is in the observable at all
///
/// Everything common to the two sources cancels in the difference — station clock offsets, the
/// tropospheric and ionospheric delay above each antenna, and the instrumental delay of each
/// receiving chain — because the two signals traverse the same atmosphere through nearly the same
/// line of sight, minutes apart. What is left is geometry. That cancellation is the entire reason
/// the technique exists, and it is why the observable is *differenced* rather than the spacecraft's
/// DOR being used alone.
///
/// # The far-field limit, which is the thing to test it against
///
/// As the spacecraft recedes, `(ρ₂ − ρ₁) → −B · ŝ_spacecraft`, so
///
/// ```text
/// ΔDOR → B · (ŝ_quasar − ŝ_spacecraft) / c
/// ```
///
/// which is proportional to the *angular* offset between the two sources projected on the baseline.
/// Two consequences follow and both are checkable without an external reference: a spacecraft in the
/// quasar's exact direction at infinite range gives exactly zero, and a small angular offset gives
/// `B_projected θ / c` to first order.
///
/// The states are in the same frame and at the same epoch convention as
/// [`TdoaGeometry`] itself; light time between the emission and
/// the two receptions is the geometry's business, not this function's.
#[must_use]
pub fn delta_dor<S: Scalar>(satellite: &[S; 6], geometry: &TdoaGeometry, quasar: Quasar) -> S {
    let source = quasar.direction();
    let first = geometry.first().station.position;
    let second = geometry.second().station.position;
    let projected = (second[0] - first[0]) * source[0]
        + (second[1] - first[1]) * source[1]
        + (second[2] - first[2]) * source[2];
    tdoa(satellite, geometry) + S::from_f64(projected / SPEED_OF_LIGHT)
}

/// The Sun's radius as the solar-corona model measures distance in, metres.
///
/// Tudat's default and the scale Aksim & Pavlov's coefficients are quoted against. It is not the
/// IAU nominal solar radius (6.957e8 m); the two differ by 0.13 %, which enters the delay as
/// `(R☉)^k` and so matters at `k = 6`. The model's constant travels with the model.
pub const SOLAR_RADIUS: f64 = 696e6;

/// `40.3 m³/s²` — the coefficient that turns a column of electrons into a range delay.
///
/// `1 / (2 ε₀ mₑ (2π)² / e²)`, so `delay = 40.3 · TEC / f²` metres with TEC in electrons per square
/// metre and `f` in hertz. The same constant the two-frequency ionospheric correction uses.
pub const PLASMA_DELAY_COEFFICIENT: f64 = 40.3;

/// The astronomical unit, metres — IAU 2012 Resolution B2, an exact defining constant.
///
/// Local because this crate depends on `libm` and `spacelab-autodiff` and nothing else; it exists
/// only to keep [`AKSIM_PAVLOV_2022`] written as the scaling its source states rather than as a
/// decimal nobody can trace.
const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;

/// One term of the inverse-power-series electron density, `c (r/R☉)^-k`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoronaTerm {
    /// `c` — the electron density this term contributes at one solar radius, per cubic metre.
    pub coefficient: f64,
    /// `k` — the positive exponent, with `r` measured in [`SOLAR_RADIUS`].
    pub exponent: f64,
}

/// The single-term solar-wind density of Aksim & Pavlov (2022), and Tudat's default.
///
/// `1.31 × 5.97e6 m⁻³` at one astronomical unit falling as `r^-2`, rescaled to one solar radius.
/// Written as that product rather than as its decimal value so both factors and the scaling stay
/// traceable; `1.31 × 5.97 ≈ 7.8` electrons per cubic centimetre at 1 AU is the familiar number.
pub const AKSIM_PAVLOV_2022: [CoronaTerm; 1] = [CoronaTerm {
    coefficient: 1.31
        * 5.97e6
        * (ASTRONOMICAL_UNIT / SOLAR_RADIUS)
        * (ASTRONOMICAL_UNIT / SOLAR_RADIUS),
    exponent: 2.0,
}];

/// Whether the observable rides the group velocity or the phase velocity.
///
/// Moyer (2000) §10.4.2: plasma retards the group and advances the phase, by the same magnitude.
/// A range observable is a group measurement and a carrier-phase or Doppler observable is a phase
/// measurement, so the same electron column enters the two with opposite sign — which is why an
/// observable's *type* has to be part of the correction and cannot be left to the caller's memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasmaSignature {
    /// A ranging observable — the signal arrives late.
    GroupDelay,
    /// A carrier-phase or Doppler observable — the phase arrives early.
    PhaseAdvance,
}

impl PlasmaSignature {
    const fn sign(self) -> f64 {
        match self {
            Self::GroupDelay => 1.0,
            Self::PhaseAdvance => -1.0,
        }
    }
}

/// A validated inverse-power-series electron-density model.
///
/// Validated once at construction for the same reason [`RampTable`] is: the series is a constant of
/// the tracking configuration, the geometry is not, and a model that can be rejected on every
/// evaluation is a model whose caller has to handle an error that never varies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoronaSeries<'a> {
    terms: &'a [CoronaTerm],
}

impl<'a> CoronaSeries<'a> {
    /// Validate a series.
    ///
    /// # Errors
    ///
    /// [`RadiometricError::EmptyCoronaSeries`] if there are no terms, and
    /// [`RadiometricError::NonPositiveExponent`] if any exponent is not positive and finite.
    pub fn new(terms: &'a [CoronaTerm]) -> Result<Self, RadiometricError> {
        if terms.is_empty() {
            return Err(RadiometricError::EmptyCoronaSeries);
        }
        for term in terms {
            if term.exponent <= 0.0 || !term.exponent.is_finite() {
                return Err(RadiometricError::NonPositiveExponent {
                    exponent: term.exponent,
                });
            }
        }
        Ok(Self { terms })
    }

    /// The terms, in the order they were given.
    #[must_use]
    pub const fn terms(&self) -> &'a [CoronaTerm] {
        self.terms
    }

    /// Electron density per cubic metre at `radius` metres from the Sun's centre.
    #[must_use]
    pub fn electron_density<S: Scalar>(&self, radius: S) -> S {
        let scaled = radius * S::from_f64(1.0 / SOLAR_RADIUS);
        let mut total = S::from_f64(0.0);
        for term in self.terms {
            total = total + S::from_f64(term.coefficient) * scaled.powf(-term.exponent);
        }
        total
    }
}

/// The solar-corona range correction along one leg, metres.
///
/// ```text
/// Δρ = ± (40.3 / f²) ∫ n_e ds
/// ```
///
/// with the sign from [`PlasmaSignature`] and the integral taken along the straight line from the
/// transmitter to the receiver. Both positions are **heliocentric**, at their own epochs; this is a
/// path integral through a static field and it needs no epoch of its own.
///
/// Tudat returns the same quantity divided by `c`, as a light-time correction in seconds. This one
/// is in metres, to compose with [`shapiro_delay`](crate::modifier::shapiro_delay) and the rest of
/// the [`modifier`](crate::modifier) layer, which are all range corrections.
///
/// # How the integral is taken
///
/// Substituting `r = p / cos θ` for `p` the perpendicular distance from the Sun to the line turns
/// each term into a closed form (Verma et al. 2013, eq. A.4):
///
/// ```text
/// ∫ c (r/R☉)^-k ds = c R☉^k p^(1−k) ∫ cos^(k−2) θ dθ,   θ ∈ [α − π/2, β + α − π/2]
/// ```
///
/// where `α` is the Sun–receiver–transmitter angle and `β` the receiver–Sun–transmitter angle. The
/// two bounds are strictly inside `(−π/2, π/2)`: the triangle's angles sum to `π`, so
/// `β + α = π − γ` and the upper bound is `π/2 − γ`. Nothing in the integrand is evaluated at the
/// singular endpoint, and that is a proof rather than a tolerance.
///
/// `∫cos^n` for integer `n ≥ 0` is the exact downward recursion. For a non-integer exponent — the
/// `r^-2.7` term of Verma et al. is the one in the literature — there is no elementary form, and
/// the integrand's derivative is singular where the bound approaches `±π/2`. Tudat falls back to
/// 50-point Gauss–Legendre along the *line*; this uses tanh–sinh in `θ`, which is the rule built
/// for an endpoint singularity of exactly that kind. The two paths agree at integer exponents,
/// which is what the tests use to check both at once.
///
/// # Errors
///
/// [`RadiometricError::NonPositiveFrequency`] if `frequency` is not positive and finite, and
/// [`RadiometricError::LineOfSightInsideTheSun`] if the line's perpendicular distance from the Sun
/// is under one [`SOLAR_RADIUS`]. That distance is the path's own closest approach when the
/// perpendicular foot lies between the two endpoints, and a lower bound on it otherwise; the second
/// case is refused too, and it is the geometry where the transmitter sits within a fraction of a
/// degree of the anti-Sun direction, where the correction is negligible and the closed form's
/// `p^(1−k)` factor is not.
pub fn solar_corona_delay<S: Scalar>(
    series: &CoronaSeries<'_>,
    transmitter_from_sun: &[S; 3],
    receiver_from_sun: &[S; 3],
    frequency: f64,
    signature: PlasmaSignature,
) -> Result<S, RadiometricError> {
    if frequency <= 0.0 || !frequency.is_finite() {
        return Err(RadiometricError::NonPositiveFrequency { hertz: frequency });
    }
    let to_transmitter = [
        transmitter_from_sun[0] - receiver_from_sun[0],
        transmitter_from_sun[1] - receiver_from_sun[1],
        transmitter_from_sun[2] - receiver_from_sun[2],
    ];
    let to_sun = [
        -receiver_from_sun[0],
        -receiver_from_sun[1],
        -receiver_from_sun[2],
    ];
    let alpha = angle_between(&to_sun, &to_transmitter);
    let beta = angle_between(receiver_from_sun, transmitter_from_sun);

    let impact = norm(receiver_from_sun) * alpha.sin();
    if impact.value() < SOLAR_RADIUS || !impact.value().is_finite() {
        return Err(RadiometricError::LineOfSightInsideTheSun {
            impact_parameter: impact.value(),
        });
    }

    let half_pi = S::from_f64(core::f64::consts::FRAC_PI_2);
    let lower = alpha - half_pi;
    let upper = beta + alpha - half_pi;

    let mut column = S::from_f64(0.0);
    for term in series.terms {
        let scale =
            S::from_f64(libm::pow(SOLAR_RADIUS, term.exponent)) * impact.powf(1.0 - term.exponent);
        column = column
            + S::from_f64(term.coefficient)
                * scale
                * cosine_power_integral(lower, upper, term.exponent - 2.0);
    }
    Ok(S::from_f64(signature.sign() * PLASMA_DELAY_COEFFICIENT / (frequency * frequency)) * column)
}

fn norm<S: Scalar>(v: &[S; 3]) -> S {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The angle between two vectors as `atan2(|a × b|, a · b)`.
///
/// Not `acos` of the normalised dot product: that loses half the significant digits as the angle
/// approaches zero, which is the regime a solar-conjunction geometry lives in.
fn angle_between<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> S {
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    norm(&cross).atan2(dot)
}

/// `∫ cos^p θ dθ` between two bounds, exactly for integer `p ≥ 0` and by quadrature otherwise.
fn cosine_power_integral<S: Scalar>(lower: S, upper: S, power: f64) -> S {
    match integer_power(power) {
        Some(n) => integer_cosine_power_integral(lower, upper, n),
        None => tanh_sinh_cosine_power_integral(lower, upper, power),
    }
}

/// `power` as a `u32` when it is a whole number the recursion can walk to, `None` otherwise.
///
/// The 64 ceiling is not a domain limit on the physics — no published corona model exceeds `k = 6`
/// — it bounds the recursion, since the exponent arrives from the caller and `k = 1e300` would
/// otherwise be a loop that does not end.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the guard immediately preceding admits only whole numbers in [0, 64]"
)]
// `trunc(x) == x` is an integrality test, not a comparison of two computed quantities. A margin
// here would route a genuinely fractional exponent into the exact recursion, which is the one
// mistake this predicate exists to prevent.
#[expect(
    clippy::float_cmp,
    reason = "exact integrality is the question being asked"
)]
fn integer_power(power: f64) -> Option<u32> {
    ((0.0..=64.0).contains(&power) && libm::trunc(power) == power).then_some(power as u32)
}

/// The exact downward recursion `∫cos^n = cos^(n−1) sin / n + (n−1)/n ∫cos^(n−2)`, walked upwards
/// from `∫cos^0 = Δθ` or `∫cos^1 = Δsin θ`.
fn integer_cosine_power_integral<S: Scalar>(lower: S, upper: S, power: u32) -> S {
    let parity = power % 2;
    let mut integral = if parity == 0 {
        upper - lower
    } else {
        upper.sin() - lower.sin()
    };
    // Stepped by the range rather than by an induction variable of its own, for the reason
    // [`RampTable::integral`](RampTable::integral) is driven by its segment list: a counter that a
    // mutation can stop incrementing does not give a wrong answer, it stops giving one at all.
    for n in (parity..=power).step_by(2).skip(1) {
        let order = f64::from(n);
        let boundary = |x: S| x.cos().powf(order - 1.0) * x.sin() * S::from_f64(1.0 / order);
        integral =
            boundary(upper) - boundary(lower) + S::from_f64((order - 1.0) / order) * integral;
    }
    integral
}

/// Step and half-width of the tanh–sinh rule: 105 nodes over `|t| ≤ 3.25`, where the outermost
/// weight is 2e-16 of the largest, so the rule is truncated where its own terms have left the
/// mantissa rather than at a round number.
///
/// The step is measured, not chosen: against a 40-digit reference for `∫cos⁵` over `[-1.55, 1.55]`
/// — the widest bounds this module can produce, reaching to within 0.02 rad of the singularity —
/// `h = 1/8` converges to 4.2e-12 relative and `h = 1/16` to 1.3e-26, which is past what `f64`
/// carries. The first was enough for the corona model and not enough to agree with the exact
/// recursion, and a rule that disagrees with the exact path is a rule that cannot check it.
const TANH_SINH_STEP: f64 = 0.0625;
const TANH_SINH_HALF_NODES: i32 = 52;

/// `∫ cos^p θ dθ` by the double-exponential rule.
///
/// The nodes and weights are functions of the rule alone, so they stay `f64` and only the integrand
/// is differentiated — the `Scalar` work is one `cos` and one `powf` per node.
fn tanh_sinh_cosine_power_integral<S: Scalar>(lower: S, upper: S, power: f64) -> S {
    let half = (upper - lower) * S::from_f64(0.5);
    let middle = (upper + lower) * S::from_f64(0.5);
    let mut total = S::from_f64(0.0);
    for k in -TANH_SINH_HALF_NODES..=TANH_SINH_HALF_NODES {
        let t = f64::from(k) * TANH_SINH_STEP;
        let u = core::f64::consts::FRAC_PI_2 * libm::sinh(t);
        let node = libm::tanh(u);
        let weight = core::f64::consts::FRAC_PI_2 * libm::cosh(t) / (libm::cosh(u) * libm::cosh(u));
        let theta = middle + half * S::from_f64(node);
        total = total + S::from_f64(weight) * theta.cos().powf(power);
    }
    half * total * S::from_f64(TANH_SINH_STEP)
}

#[cfg(test)]
mod tests {
    // The band ratios and the conversion factor are ratios of small exact integers, so `==` against
    // Moyer's published fractions is the assertion — a tolerance there would only hide a swapped
    // numerator. Where a comparison is genuinely floating point it is written as a residual instead.
    #![allow(clippy::float_cmp, clippy::unwrap_used)]

    use super::*;
    use crate::multistatic::Terminal;
    use spacelab_autodiff::Dual;

    /// The ramp table from Tudat's own `unitTestTransmittingFrequencies.cpp`, whose expected
    /// integrals that test writes out as hand-formed trapezoids rather than computing them the way
    /// the code under test does.
    ///
    /// It is a deliberately nasty table: the frequency **steps** at every joint — 14 Hz down to 5 at
    /// `t = 5`, 13 down to 10 at `t = 9`, 11 down to 2 at `t = 10` — so a lookup that resolves a
    /// shared boundary to the wrong side is off by a whole segment rather than by a rounding.
    const TUDAT_RAMPS: [RampSegment; 4] = [
        RampSegment {
            start: 1.0,
            end: 5.0,
            start_frequency: 10.0,
            rate: 1.0,
        },
        RampSegment {
            start: 5.0,
            end: 9.0,
            start_frequency: 5.0,
            rate: 2.0,
        },
        RampSegment {
            start: 9.0,
            end: 10.0,
            start_frequency: 10.0,
            rate: 1.0,
        },
        RampSegment {
            start: 10.0,
            end: 15.0,
            start_frequency: 2.0,
            rate: 3.0,
        },
    ];

    fn tudat_table() -> RampTable<'static> {
        RampTable::new(&TUDAT_RAMPS).unwrap()
    }

    #[test]
    fn the_ramp_integral_reproduces_tudats_own_trapezoids() {
        let table = tudat_table();
        // Every right-hand side is Tudat's expression verbatim, and every one of them is a trapezoid
        // formed from frequencies read off the ramps by hand — an oracle that shares no arithmetic
        // with the walk it is checking.
        for (from, to, expected) in [
            (2.0, 4.0, 2.0 * (11.0 + 13.0) / 2.0),
            (5.0, 7.0, 2.0 * (5.0 + 9.0) / 2.0),
            (7.0, 9.0, 2.0 * (9.0 + 13.0) / 2.0),
            (5.0, 9.0, 4.0 * (5.0 + 13.0) / 2.0),
            (
                2.0,
                13.0,
                (3.0 * (11.0 + 14.0)
                    + 4.0 * (5.0 + 13.0)
                    + 1.0 * (11.0 + 10.0)
                    + 3.0 * (2.0 + 11.0))
                    / 2.0,
            ),
        ] {
            let got: f64 = table.integral(from, to).unwrap();
            assert!(
                (got - expected).abs() <= 8.0 * f64::EPSILON * expected.abs(),
                "integral({from}, {to}) = {got}, Tudat's trapezoid is {expected}"
            );
        }
    }

    #[test]
    fn a_shared_boundary_belongs_to_the_ramp_that_is_starting() {
        let table = tudat_table();
        // Tudat's own comment on `t = 5` is "start frequency of the 1st ramp" — 5 Hz, not the 14 Hz
        // the segment that just ended would extrapolate to. The two differ by a factor of nearly
        // three here precisely so that a half-open lookup and a closed one cannot both pass.
        for (epoch, expected) in [
            (1.0, 10.0),
            (2.0, 11.0),
            (5.0, 5.0),
            (9.0, 10.0),
            (13.0, 11.0),
            (15.0, 17.0),
        ] {
            assert_eq!(
                table.frequency_at(epoch).unwrap(),
                expected,
                "frequency at {epoch}"
            );
        }
        // The span the table reports is the span its segments cover, and neither end is zero — a
        // ramp table for a real pass starts at the epoch the exciter was switched on.
        assert_eq!(table.first(), 1.0);
        assert_eq!(table.last(), 15.0);
        assert_eq!(table.segments().len(), 4);
    }

    #[test]
    fn every_turnaround_ratio_is_moyers_table() {
        // Moyer (2000) table 13-1: the downlink supplies the numerator, the uplink the denominator.
        // Written as the nine explicit fractions rather than as the two vectors the implementation
        // composes them from, so a swapped numerator and denominator is visible here.
        for (up, down, expected) in [
            (Band::S, Band::S, 240.0 / 221.0),
            (Band::S, Band::X, 880.0 / 221.0),
            (Band::S, Band::Ka, 3344.0 / 221.0),
            (Band::X, Band::S, 240.0 / 749.0),
            (Band::X, Band::X, 880.0 / 749.0),
            (Band::X, Band::Ka, 3344.0 / 749.0),
            (Band::Ka, Band::S, 240.0 / 3599.0),
            (Band::Ka, Band::X, 880.0 / 3599.0),
            (Band::Ka, Band::Ka, 3344.0 / 3599.0),
        ] {
            assert_eq!(
                up.turnaround_ratio(down),
                expected,
                "{up:?} up, {down:?} down"
            );
        }
        // The ratio is not symmetric, and the two bands are the same type, so a caller who swaps
        // them gets a number rather than a compile error. This is what `DopplerLink`'s named fields
        // exist to make visible.
        assert_ne!(
            Band::X.turnaround_ratio(Band::S),
            Band::S.turnaround_ratio(Band::X)
        );
        // Cassini's Ka/Ka transponder is not the table value.
        assert_ne!(
            CASSINI_KA_TURNAROUND_RATIO,
            Band::Ka.turnaround_ratio(Band::Ka)
        );
        assert_eq!(CASSINI_KA_TURNAROUND_RATIO, 14.0 / 15.0);
    }

    #[test]
    fn the_range_conversion_factor_is_moyers_three_cases() {
        // The three cases as Moyer states them, against the single `221 / 2d` the implementation
        // forms — the point of the test is that one expression reproduces three published values.
        assert_eq!(Band::S.range_conversion_factor(), 0.5);
        assert_eq!(Band::X.range_conversion_factor(), 221.0 / (749.0 * 2.0));
        assert_eq!(Band::Ka.range_conversion_factor(), 221.0 / (3599.0 * 2.0));
    }

    #[test]
    fn a_constant_exciter_makes_range_units_the_round_trip_times_the_scale() {
        // One ramp, rate zero: the integral is `f Δt` in closed form, so the observable is
        // `(221/2d) f Δt` with nothing left to a quadrature.
        let flat = [RampSegment {
            start: 0.0,
            end: 1_000.0,
            start_frequency: 7.2e9,
            rate: 0.0,
        }];
        let table = RampTable::new(&flat).unwrap();
        let (transmit, receive) = (10.0, 410.0);
        let expected = Band::X.range_conversion_factor() * 7.2e9 * (receive - transmit);
        // Large enough to wrap many times over, so the modulus has to be defeated to see the scale.
        let modulus = range_modulus(30).unwrap();
        let got: f64 = sequential_range(&table, Band::X, transmit, receive, 30).unwrap();
        let unwrapped = got + modulus * libm::floor(expected / modulus);
        assert!(
            (unwrapped - expected).abs() <= 1e-6 * expected,
            "range units {unwrapped} against the closed form {expected}"
        );
        assert!(
            got >= 0.0 && got < modulus,
            "the observable lies in one period"
        );
    }

    #[test]
    fn the_ranging_modulus_is_two_to_the_component_plus_six() {
        assert_eq!(range_modulus(0).unwrap(), 64.0);
        assert_eq!(range_modulus(-6).unwrap(), 1.0);
        assert_eq!(range_modulus(14).unwrap(), 1_048_576.0);
        // Both arms of the guard, and they fail differently. 2^1024 is not finite; 2^-1075 is
        // finite and is zero, because the smallest subnormal is 2^-1074 — a modulus of zero would
        // make the wrap divide by nothing rather than overflow, so `is_finite` alone misses it.
        assert_eq!(
            range_modulus(1_018),
            Err(RadiometricError::UnrepresentableModulus { component: 1_018 })
        );
        assert_eq!(range_modulus(-1_080).unwrap(), libm::exp2(-1074.0));
        assert_eq!(
            range_modulus(-1_081),
            Err(RadiometricError::UnrepresentableModulus { component: -1_081 })
        );
    }

    #[test]
    fn the_integrals_derivative_is_the_frequency_at_the_moving_endpoint() {
        // The whole reason the endpoints are `Scalar` rather than `f64`: `d/dt ∫_a^t f = f(t)`, so
        // the partial the filter needs arrives from the dual arithmetic and is never written down.
        // Checked on a *ramped* segment, where `f(t)` is not the same as `f` anywhere else in it.
        let table = tudat_table();
        let from = Dual::<1>::constant(5.5);
        let to = Dual::<1>::variable(7.5, 0);
        let integral = table.integral(from, to).unwrap();
        let expected = table.frequency_at(7.5).unwrap();
        assert!(
            (integral.grad()[0] - expected).abs() < 1e-12,
            "d/d(to) = {}, f_T(7.5) = {expected}",
            integral.grad()[0]
        );
        // And the lower endpoint enters with the opposite sign.
        let from = Dual::<1>::variable(5.5, 0);
        let to = Dual::<1>::constant(7.5);
        let integral = table.integral(from, to).unwrap();
        let expected = table.frequency_at(5.5).unwrap();
        assert!(
            (integral.grad()[0] + expected).abs() < 1e-12,
            "d/d(from) = {}, −f_T(5.5) = {}",
            integral.grad()[0],
            -expected
        );

        // The endpoint landing *exactly* on a segment boundary is the case the walk has to choose
        // in: `to` carries the derivative, the segment's own `end` is a constant, and they agree in
        // value. Taking the constant there returns the right number with a silently zero partial.
        // The answer is the ramp's frequency approached from the **left** — 5 + 2·(9−5) = 13 — even
        // though `frequency_at(9)` reports the *next* ramp's 10; the derivative of an integral at a
        // discontinuity is the one-sided limit on the side the integral came from.
        let from = Dual::<1>::constant(5.5);
        let to = Dual::<1>::variable(9.0, 0);
        let integral = table.integral(from, to).unwrap();
        assert!(
            (integral.grad()[0] - 13.0).abs() < 1e-12,
            "d/d(to) at a segment boundary = {}, the left limit is 13",
            integral.grad()[0]
        );
        assert_eq!(table.frequency_at(9.0).unwrap(), 10.0);

        // The *lower* endpoint on a boundary is the mirror case and it fails through a different
        // line. The walk admits a segment when `s.end > from`; admitting one that ends exactly
        // there adds a zero-width trapezoid — no change in value — and then replaces the endpoint's
        // dual with that segment's own constant `end`, so everything after it is differentiated
        // against nothing.
        let from = Dual::<1>::variable(5.0, 0);
        let to = Dual::<1>::constant(7.5);
        let integral = table.integral(from, to).unwrap();
        assert!(
            (integral.grad()[0] + 5.0).abs() < 1e-12,
            "d/d(from) at a segment boundary = {}, and −f_T(5) is −5",
            integral.grad()[0]
        );
    }

    #[test]
    fn a_doppler_count_over_a_flat_exciter_is_the_two_terms_by_hand() {
        let flat = [RampSegment {
            start: 0.0,
            end: 10_000.0,
            start_frequency: 7.15e9,
            rate: 0.0,
        }];
        let table = RampTable::new(&flat).unwrap();
        let link = DopplerLink {
            uplink: Band::X,
            downlink: Band::X,
            reference: DopplerReference {
                frequency: 7.15e9,
                band: Band::X,
            },
            signature: DopplerSignature::Subtracted,
            count_interval: 60.0,
        };
        // Transmission epochs exactly one count interval apart: the round-trip light time did not
        // change across the count, so there is no Doppler signature and the two terms cancel.
        let quiet: f64 = averaged_doppler(&table, link, 100.0, 160.0).unwrap();
        assert!(
            quiet.abs() < 1e-3,
            "a range rate of zero should give no count, got {quiet}"
        );
        // Stretch the transmission interval by 1 ms — the signature of a receding spacecraft — and
        // the count is the reference times that fractional stretch, with the DSN's sign.
        let stretched: f64 = averaged_doppler(&table, link, 100.0, 160.001).unwrap();
        let expected = -Band::X.turnaround_ratio(Band::X) * 7.15e9 * 0.001 / 60.0;
        assert!(
            (stretched - expected).abs() < 1e-6 * expected.abs(),
            "count {stretched} against the hand-formed {expected}"
        );
        // The convention is the only difference between the two signs, and it is not cosmetic.
        let added = DopplerLink {
            signature: DopplerSignature::Added,
            ..link
        };
        let other: f64 = averaged_doppler(&table, added, 100.0, 160.001).unwrap();
        let reference_term = Band::X.turnaround_ratio(Band::X) * 7.15e9;
        assert!(
            (other + stretched - 2.0 * reference_term).abs() < 1e-3,
            "the two conventions straddle the reference term"
        );
    }

    #[test]
    fn the_reference_band_is_not_the_transmitted_band() {
        // `reference.band` and `uplink` are both `Band`, and this is the case that separates them:
        // a receiver configured against an S-band reference while the station transmits on X.
        let flat = [RampSegment {
            start: 0.0,
            end: 10_000.0,
            start_frequency: 7.15e9,
            rate: 0.0,
        }];
        let table = RampTable::new(&flat).unwrap();
        let base = DopplerLink {
            uplink: Band::X,
            downlink: Band::X,
            reference: DopplerReference {
                frequency: 2.1e9,
                band: Band::S,
            },
            signature: DopplerSignature::Subtracted,
            count_interval: 60.0,
        };
        let swapped = DopplerLink {
            reference: DopplerReference {
                frequency: 2.1e9,
                band: Band::X,
            },
            ..base
        };
        let a: f64 = averaged_doppler(&table, base, 100.0, 160.0).unwrap();
        let b: f64 = averaged_doppler(&table, swapped, 100.0, 160.0).unwrap();
        // The reference term is `M(reference band, downlink) f_ref`, so the two differ by the
        // difference of the two turnaround ratios times the reference frequency.
        let expected =
            (Band::S.turnaround_ratio(Band::X) - Band::X.turnaround_ratio(Band::X)) * 2.1e9;
        assert!(
            (a - b - expected).abs() < 1e-3,
            "swapping the reference band moved the count by {}, expected {expected}",
            a - b
        );
    }

    #[test]
    fn a_gap_between_two_ramps_is_reported_rather_than_interpolated() {
        // The station stopped transmitting between 5 and 7. An integral that spans the gap must not
        // quietly extend either neighbour across it.
        let gapped = [
            RampSegment {
                start: 0.0,
                end: 5.0,
                start_frequency: 100.0,
                rate: 0.0,
            },
            RampSegment {
                start: 7.0,
                end: 12.0,
                start_frequency: 100.0,
                rate: 0.0,
            },
        ];
        let table = RampTable::new(&gapped).unwrap();
        assert_eq!(
            table.integral(1.0, 9.0),
            Err(RadiometricError::EpochOutsideRampTable {
                epoch: 5.0,
                first: 0.0,
                last: 12.0
            })
        );
        // Either side of the gap on its own is fine, and so is the gap's own interior as a lookup.
        assert!(table.integral(1.0, 4.0).is_ok());
        assert!(table.integral(7.5, 9.0).is_ok());
        assert!(matches!(
            table.frequency_at(6.0),
            Err(RadiometricError::EpochOutsideRampTable { .. })
        ));

        // Running off the *end* is a different refusal from running into a gap: no later segment
        // exists to notice the shortfall, so it is the walk's own coverage check that reports it,
        // and it names the epoch coverage stopped at rather than the one that was asked for.
        assert_eq!(
            tudat_table().integral(13.0, 20.0),
            Err(RadiometricError::EpochOutsideRampTable {
                epoch: 15.0,
                first: 1.0,
                last: 15.0
            })
        );
    }

    #[test]
    fn a_table_that_could_answer_twice_is_refused_at_construction() {
        assert_eq!(RampTable::new(&[]), Err(RadiometricError::EmptyRampTable));
        // A segment that ends before it starts.
        let backwards = [RampSegment {
            start: 5.0,
            end: 1.0,
            start_frequency: 1.0,
            rate: 0.0,
        }];
        assert_eq!(
            RampTable::new(&backwards),
            Err(RadiometricError::RampTableNotOrdered { index: 0 })
        );
        // A zero-length segment, which would contribute nothing and stall the walk.
        let degenerate = [RampSegment {
            start: 5.0,
            end: 5.0,
            start_frequency: 1.0,
            rate: 0.0,
        }];
        assert_eq!(
            RampTable::new(&degenerate),
            Err(RadiometricError::RampTableNotOrdered { index: 0 })
        );
        // Two segments that overlap: epoch 4 would resolve to whichever the search reached first.
        let overlapping = [
            RampSegment {
                start: 0.0,
                end: 5.0,
                start_frequency: 1.0,
                rate: 0.0,
            },
            RampSegment {
                start: 4.0,
                end: 9.0,
                start_frequency: 2.0,
                rate: 0.0,
            },
        ];
        assert_eq!(
            RampTable::new(&overlapping),
            Err(RadiometricError::RampTableNotOrdered { index: 1 })
        );
        // Touching at a point is not overlapping, and is the ordinary case.
        let touching = [
            RampSegment {
                start: 0.0,
                end: 5.0,
                start_frequency: 1.0,
                rate: 0.0,
            },
            RampSegment {
                start: 5.0,
                end: 9.0,
                start_frequency: 2.0,
                rate: 0.0,
            },
        ];
        assert!(RampTable::new(&touching).is_ok());
        // A NaN endpoint is refused rather than compared, because `end <= start` is false for it.
        let nan = [RampSegment {
            start: f64::NAN,
            end: 5.0,
            start_frequency: 1.0,
            rate: 0.0,
        }];
        assert_eq!(
            RampTable::new(&nan),
            Err(RadiometricError::RampTableNotOrdered { index: 0 })
        );
    }

    #[test]
    fn an_interval_that_does_not_run_forwards_is_refused() {
        let table = tudat_table();
        assert_eq!(
            table.integral(7.0, 7.0),
            Err(RadiometricError::NonPositiveInterval { seconds: 0.0 })
        );
        assert_eq!(
            table.integral(9.0, 7.0),
            Err(RadiometricError::NonPositiveInterval { seconds: -2.0 })
        );
        // A NaN endpoint reaches the same refusal rather than passing an ordering test it cannot
        // fail: `b > a` is false for NaN, so without the finiteness check this would be reported as
        // an ordinary backwards interval and the NaN would never be named.
        assert!(matches!(
            table.integral(f64::NAN, 7.0),
            Err(RadiometricError::NonPositiveInterval { .. })
        ));
        // And a count interval of zero would divide the ramp term by nothing.
        let link = DopplerLink {
            uplink: Band::X,
            downlink: Band::X,
            reference: DopplerReference {
                frequency: 7.15e9,
                band: Band::X,
            },
            signature: DopplerSignature::Subtracted,
            count_interval: 0.0,
        };
        assert_eq!(
            averaged_doppler(&table, link, 2.0, 4.0),
            Err(RadiometricError::NonPositiveInterval { seconds: 0.0 })
        );
    }

    /// A superior-conjunction geometry: an Earth station at one astronomical unit and a spacecraft
    /// beyond the Sun, offset so the line of sight clears the photosphere by `impact` metres.
    ///
    /// Built along two **generic** unit vectors rather than along the axes. The first version of
    /// this fixture put the station on `x` and the spacecraft in the `xy` plane, which is the same
    /// geometry up to a rotation and gives the same delay — and leaves two of the three components
    /// of every cross product identically zero, so most of `angle_between` could be mutated with
    /// nothing moving. A planar fixture tests a planar special case of a three-dimensional routine.
    fn conjunction(impact: f64) -> ([f64; 3], [f64; 3]) {
        // `along` is orthogonal to `across` — `2·3 + 3·6 + 6·(−4) = 0` — and neither has a zero
        // component, so every term of every cross product and dot product is live.
        const ALONG: [f64; 3] = [2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0];
        let across = [3.0, 6.0, -4.0].map(|c: f64| c / libm::sqrt(61.0));
        let depth = -1.5 * ASTRONOMICAL_UNIT;
        // With the spacecraft at `depth·along + offset·across` the impact parameter is
        // `AU·offset / hypot(AU − depth, offset)`; this is that inverted exactly.
        let base = ASTRONOMICAL_UNIT - depth;
        let offset =
            impact * base / libm::sqrt(ASTRONOMICAL_UNIT * ASTRONOMICAL_UNIT - impact * impact);
        let mut receiver = [0.0; 3];
        let mut transmitter = [0.0; 3];
        for axis in 0..3 {
            receiver[axis] = ASTRONOMICAL_UNIT * ALONG[axis];
            transmitter[axis] = depth * ALONG[axis] + offset * across[axis];
        }
        (receiver, transmitter)
    }

    /// The electron column along the segment by composite Simpson in arc length — a different
    /// variable, a different rule, and no trigonometry, so it shares nothing with the closed form
    /// beyond the density model itself.
    fn column_by_quadrature(series: &CoronaSeries<'_>, from: &[f64; 3], to: &[f64; 3]) -> f64 {
        const INTERVALS: u32 = 200_000;
        let count = f64::from(INTERVALS);
        let step = [
            (to[0] - from[0]) / count,
            (to[1] - from[1]) / count,
            (to[2] - from[2]) / count,
        ];
        let length = libm::sqrt(
            (to[0] - from[0]) * (to[0] - from[0])
                + (to[1] - from[1]) * (to[1] - from[1])
                + (to[2] - from[2]) * (to[2] - from[2]),
        );
        let at = |i: u32| {
            let f = f64::from(i);
            let point = [
                from[0] + step[0] * f,
                from[1] + step[1] * f,
                from[2] + step[2] * f,
            ];
            let r = libm::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
            series.electron_density(r)
        };
        let mut total = at(0) + at(INTERVALS);
        for i in 1..INTERVALS {
            total += at(i) * if i % 2 == 0 { 2.0 } else { 4.0 };
        }
        total * length / (3.0 * count)
    }

    #[test]
    fn the_corona_closed_form_matches_a_line_integral_taken_in_arc_length() {
        // The identity is Verma et al. (2013) eq. A.4. It was re-derived and checked in Maxima at
        // k = 2, 3, 4 and 6 against `quad_qags` in the arc-length variable before this test was
        // written; what the test adds is that *this* code evaluates it — the substitution
        // r = p/cos θ, the bounds from the two angles, and the recursion — at real magnitudes.
        let (receiver, transmitter) = conjunction(10.0 * SOLAR_RADIUS);
        for exponent in [2.0, 3.0, 4.0, 6.0, 2.7] {
            let terms = [CoronaTerm {
                coefficient: AKSIM_PAVLOV_2022[0].coefficient,
                exponent,
            }];
            let series = CoronaSeries::new(&terms).unwrap();
            let expected = column_by_quadrature(&series, &transmitter, &receiver);
            let delay: f64 = solar_corona_delay(
                &series,
                &transmitter,
                &receiver,
                8.4e9,
                PlasmaSignature::GroupDelay,
            )
            .unwrap();
            let column = delay * 8.4e9 * 8.4e9 / PLASMA_DELAY_COEFFICIENT;
            assert!(
                (column - expected).abs() < 1e-8 * expected,
                "k = {exponent}: closed form {column:.9e} against the quadrature {expected:.9e}"
            );
        }
    }

    #[test]
    fn the_column_does_not_depend_on_which_end_is_called_the_receiver() {
        // The closed form is written around the *receiver's* distance and the angle at the
        // receiver, so its symmetry under swapping the two ends is a property of the algebra rather
        // than of the expression, and worth asserting.
        let (receiver, transmitter) = conjunction(4.0 * SOLAR_RADIUS);
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).unwrap();
        let forward: f64 = solar_corona_delay(
            &series,
            &transmitter,
            &receiver,
            8.4e9,
            PlasmaSignature::GroupDelay,
        )
        .unwrap();
        let reversed: f64 = solar_corona_delay(
            &series,
            &receiver,
            &transmitter,
            8.4e9,
            PlasmaSignature::GroupDelay,
        )
        .unwrap();
        assert!(
            (forward - reversed).abs() < 1e-12 * forward,
            "{forward} one way, {reversed} the other"
        );
    }

    #[test]
    fn the_two_integration_paths_agree_where_both_apply() {
        // The exact recursion and the double-exponential rule are the module's own cross-check:
        // they share no code, and at an integer exponent both are supposed to be right. The bounds
        // reach to within 0.02 rad of the singular endpoint, which is where the quadrature is asked
        // to do the work it was chosen for.
        for (lower, upper) in [(-0.7, 0.4), (-1.55, 1.55), (0.2, 1.5), (-1.5, -0.1)] {
            for power in 0..=6u32 {
                let exact = integer_cosine_power_integral(lower, upper, power);
                let quadrature = tanh_sinh_cosine_power_integral(lower, upper, f64::from(power));
                let floor = if exact.abs() > 1e-12 {
                    exact.abs()
                } else {
                    1e-12
                };
                assert!(
                    (exact - quadrature).abs() < 1e-12 * floor,
                    "cos^{power} over [{lower}, {upper}]: {exact} exact, {quadrature} by quadrature"
                );
            }
        }
        // And the dispatcher routes on integrality rather than on magnitude.
        assert_eq!(integer_power(4.0), Some(4));
        assert_eq!(integer_power(0.0), Some(0));
        assert_eq!(integer_power(2.7), None);
        assert_eq!(integer_power(-1.0), None);
        assert_eq!(
            integer_power(65.0),
            None,
            "the recursion bound is not open-ended"
        );
        assert_eq!(integer_power(f64::INFINITY), None);
    }

    #[test]
    fn the_default_series_is_the_solar_winds_own_density_at_one_astronomical_unit() {
        // 1.31 × 5.97e6 m⁻³ at 1 AU is the number Aksim & Pavlov quote, and about 8 electrons per
        // cubic centimetre is what the solar wind is measured at there. Recovering it from a
        // coefficient stated at one solar radius is the check on the (AU/R☉)² rescaling's sign.
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).unwrap();
        let at_one_au: f64 = series.electron_density(ASTRONOMICAL_UNIT);
        assert!(
            (at_one_au - 1.31 * 5.97e6).abs() < 1e-6 * 1.31 * 5.97e6,
            "{at_one_au} m^-3 at 1 AU"
        );
        // And the falloff is the exponent it says it is.
        let at_two_au: f64 = series.electron_density(2.0 * ASTRONOMICAL_UNIT);
        assert!((at_one_au / at_two_au - 4.0).abs() < 1e-9);
    }

    #[test]
    fn the_delay_is_signed_by_the_observable_and_falls_as_one_over_frequency_squared() {
        let (receiver, transmitter) = conjunction(6.0 * SOLAR_RADIUS);
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).unwrap();
        let delay = |f, signature| -> f64 {
            solar_corona_delay(&series, &transmitter, &receiver, f, signature).unwrap()
        };
        let group = delay(8.4e9, PlasmaSignature::GroupDelay);
        let phase = delay(8.4e9, PlasmaSignature::PhaseAdvance);
        assert!(group > 0.0, "a ranging observable is retarded, got {group}");
        assert!(
            (group + phase).abs() < 1e-12 * group,
            "the two are equal and opposite"
        );
        // Two bands, one column: the ratio is the inverse square of the frequency ratio and nothing
        // else, which is what makes the dual-frequency cancellation work.
        let s_band = delay(2.3e9, PlasmaSignature::GroupDelay);
        let ratio = (8.4e9 / 2.3e9) * (8.4e9 / 2.3e9);
        assert!(
            (s_band / group - ratio).abs() < 1e-12 * ratio,
            "S/X delay ratio {} against {ratio}",
            s_band / group
        );
        // The size is the one the technique is known for: tens of metres at X-band a few solar
        // radii off the limb, which is why superior-conjunction arcs are calibrated or discarded.
        assert!(
            (10.0..300.0).contains(&group),
            "X-band delay at 6 R_sun is {group} m"
        );
    }

    #[test]
    fn a_line_of_sight_that_grazes_the_sun_is_refused_rather_than_extrapolated() {
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).unwrap();
        let (receiver, transmitter) = conjunction(0.5 * SOLAR_RADIUS);
        let refused = solar_corona_delay::<f64>(
            &series,
            &transmitter,
            &receiver,
            8.4e9,
            PlasmaSignature::GroupDelay,
        );
        assert!(matches!(
            refused,
            Err(RadiometricError::LineOfSightInsideTheSun { .. })
        ));
        // One solar radius is the boundary, and just outside it is admitted.
        let (receiver, transmitter) = conjunction(1.000_001 * SOLAR_RADIUS);
        assert!(
            solar_corona_delay::<f64>(
                &series,
                &transmitter,
                &receiver,
                8.4e9,
                PlasmaSignature::GroupDelay
            )
            .is_ok()
        );

        // *Exactly* one solar radius, which is a policy the guard states and a straddle cannot
        // check. Not a physical configuration — the station is at the photosphere — but the
        // arithmetic is exact rather than nearly so: with the station on `y` and the spacecraft
        // displaced along `x` from it, the dot product is `+0.0` and `atan2` returns `π/2` to the
        // last bit, `sin` of that is 1.0, and `sqrt(fl(r·r)) == r`, so the impact parameter is
        // `SOLAR_RADIUS` itself and not a neighbour of it.
        let grazing = [0.0, SOLAR_RADIUS, 0.0];
        let beyond = [1e11, SOLAR_RADIUS, 0.0];
        assert!(
            solar_corona_delay::<f64>(
                &series,
                &beyond,
                &grazing,
                8.4e9,
                PlasmaSignature::GroupDelay
            )
            .is_ok(),
            "the boundary itself is inside the domain"
        );
    }

    #[test]
    fn a_series_that_does_not_converge_and_a_frequency_that_does_not_exist_are_refused() {
        assert_eq!(
            CoronaSeries::new(&[]),
            Err(RadiometricError::EmptyCoronaSeries)
        );
        for exponent in [0.0, -2.0, f64::NAN] {
            let terms = [CoronaTerm {
                coefficient: 1.0,
                exponent,
            }];
            assert!(matches!(
                CoronaSeries::new(&terms),
                Err(RadiometricError::NonPositiveExponent { .. })
            ));
        }
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).unwrap();
        let (receiver, transmitter) = conjunction(6.0 * SOLAR_RADIUS);
        for frequency in [0.0, -8.4e9, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                solar_corona_delay::<f64>(
                    &series,
                    &transmitter,
                    &receiver,
                    frequency,
                    PlasmaSignature::GroupDelay
                ),
                Err(RadiometricError::NonPositiveFrequency { .. })
            ));
        }
        assert_eq!(series.terms().len(), 1);
    }

    #[test]
    fn the_measured_frequency_and_the_doppler_count_add_up_to_the_reference() {
        // Two observables of one link, and the identity between them is exact over a flat exciter:
        // the count is what the carrier lost against the reference, the measured frequency is what
        // it kept. Neither function knows about the other, so the sum landing on `M · f_reference`
        // checks the turnaround ratio, the sign convention and the count-interval scaling at once.
        let flat = [RampSegment {
            start: 0.0,
            end: 10_000.0,
            start_frequency: 7.15e9,
            rate: 0.0,
        }];
        let table = RampTable::new(&flat).unwrap();
        let link = DopplerLink {
            uplink: Band::X,
            downlink: Band::X,
            reference: DopplerReference {
                frequency: 7.15e9,
                band: Band::X,
            },
            signature: DopplerSignature::Subtracted,
            count_interval: 60.0,
        };
        let ratio = Band::X.turnaround_ratio(Band::X);
        for rate in [1.0, 1.0 - 1e-5, 1.0 + 3e-6] {
            let instantaneous: f64 =
                measured_frequency(&table, Band::X, Band::X, 100.0, rate).unwrap();
            let counted: f64 = averaged_doppler(&table, link, 100.0, 100.0 + 60.0 * rate).unwrap();
            let total = ratio * 7.15e9;
            assert!(
                (instantaneous + counted - total).abs() < 1e-3,
                "rate {rate}: {instantaneous:.6} + {counted:.6} against {total:.6}"
            );
        }

        // On a *ramped* exciter it is the frequency at the transmission epoch that enters, not the
        // segment's start frequency: the Tudat fixture's second ramp is 5 Hz at t = 5 and 2 Hz per
        // second after it, so t = 7 is 9 Hz and nothing else.
        let ramped = tudat_table();
        let got: f64 = measured_frequency(&ramped, Band::S, Band::X, 7.0, 1.0).unwrap();
        assert!(
            (got - Band::S.turnaround_ratio(Band::X) * 9.0).abs() < 1e-12,
            "{got}"
        );

        // The two bands are the same type and the ratio is not symmetric, so swapping them is a
        // silent change of answer rather than a compile error — asserted, not assumed.
        let swapped: f64 = measured_frequency(&ramped, Band::X, Band::S, 7.0, 1.0).unwrap();
        assert!(swapped < got, "{swapped} against {got}");

        // And the epoch is checked against the table rather than extrapolated off the end.
        assert!(matches!(
            measured_frequency::<f64>(&ramped, Band::X, Band::X, 20.0, 1.0),
            Err(RadiometricError::EpochOutsideRampTable { .. })
        ));

        // Differentiable in the transmission epoch, and the partial is the ramp rate times the
        // ratio — the term a filter needs when the light-time solution moves that epoch.
        let epoch = Dual::<1>::variable(7.0, 0);
        let derivative = measured_frequency(&ramped, Band::S, Band::X, epoch, Dual::constant(1.0))
            .unwrap()
            .grad()[0];
        assert!(
            (derivative - Band::S.turnaround_ratio(Band::X) * 2.0).abs() < 1e-12,
            "d/d(transmit) = {derivative}"
        );
    }

    /// DSS-14 (Goldstone) and DSS-43 (Canberra) in metres — the longest baseline the DSN observes a
    /// spacecraft on, and the one ΔDOR is normally formed across.
    const GOLDSTONE: [f64; 3] = [-2_353_621.4, -4_641_341.5, 3_677_052.3];
    const CANBERRA: [f64; 3] = [-4_460_894.9, 2_682_361.5, -3_674_748.5];

    fn baseline() -> TdoaGeometry {
        TdoaGeometry::new(
            0.0,
            Terminal::fixed(GOLDSTONE, 1.0),
            Terminal::fixed(CANBERRA, 1.0),
        )
        .unwrap()
    }

    #[test]
    fn the_quasar_direction_is_the_unit_vector_the_catalogue_angles_name() {
        for (ra, dec, expected) in [
            (0.0, 0.0, [1.0, 0.0, 0.0]),
            (core::f64::consts::FRAC_PI_2, 0.0, [0.0, 1.0, 0.0]),
            (0.0, core::f64::consts::FRAC_PI_2, [0.0, 0.0, 1.0]),
        ] {
            let got = Quasar {
                right_ascension: ra,
                declination: dec,
            }
            .direction();
            for (a, b) in got.iter().zip(&expected) {
                assert!((a - b).abs() < 1e-15, "{got:?} against {expected:?}");
            }
        }
        // Unit norm at an arbitrary pair, which is the invariant the type exists to hold.
        let d = Quasar {
            right_ascension: 2.71,
            declination: -0.83,
        }
        .direction();
        assert!((d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - 1.0).abs() < 1e-15);
    }

    #[test]
    fn delta_dor_vanishes_on_a_spacecraft_in_the_quasars_own_direction() {
        // Not "is small": what is left is the wavefront's curvature over the baseline, which falls
        // as 1/range. Asserting the *rate* is a statement about the limit; asserting a tolerance
        // would only be a statement about the range that happened to be picked.
        let geometry = baseline();
        let quasar = Quasar {
            right_ascension: 1.3,
            declination: 0.4,
        };
        let direction = quasar.direction();
        let at = |range: f64| -> f64 {
            let state = [
                range * direction[0],
                range * direction[1],
                range * direction[2],
                0.0,
                0.0,
                0.0,
            ];
            delta_dor(&state, &geometry, quasar)
        };
        let near = at(1e11);
        let far = at(4e11);
        assert!(
            ((near / far).abs() - 4.0).abs() < 0.01,
            "the residual should fall as 1/range: {near:.6e} at 1e11, {far:.6e} at 4e11"
        );
        // And what is left is small against the undifferenced arrival-time difference, which is
        // the quantity the quasar was observed in order to remove.
        let state = [
            1e11 * direction[0],
            1e11 * direction[1],
            1e11 * direction[2],
            0.0,
            0.0,
            0.0,
        ];
        let raw: f64 = tdoa(&state, &geometry);
        assert!(
            raw.abs() > 1e4 * near.abs(),
            "TDOA {raw:.6e} s against the differenced {near:.6e} s"
        );
    }

    #[test]
    fn the_far_field_limit_is_the_baseline_projected_on_the_angular_offset() {
        // ΔDOR → B·(ŝ_quasar − ŝ_spacecraft)/c. The code forms it from two finite ranges; this
        // forms it from two plane waves. They agree only if the quasar term carries the sign and
        // the baseline direction the derivation says.
        let geometry = baseline();
        let quasar = Quasar {
            right_ascension: 1.3,
            declination: 0.4,
        };
        let source = quasar.direction();
        // A direction one milliradian off the quasar.
        let offset = 1e-3;
        let mut spacecraft = [
            source[0] - offset * source[1],
            source[1] + offset * source[0],
            source[2],
        ];
        let norm = libm::sqrt(
            spacecraft[0] * spacecraft[0]
                + spacecraft[1] * spacecraft[1]
                + spacecraft[2] * spacecraft[2],
        );
        for component in &mut spacecraft {
            *component /= norm;
        }
        let range = 1e14;
        let state = [
            range * spacecraft[0],
            range * spacecraft[1],
            range * spacecraft[2],
            0.0,
            0.0,
            0.0,
        ];
        let got: f64 = delta_dor(&state, &geometry, quasar);
        let b = [
            CANBERRA[0] - GOLDSTONE[0],
            CANBERRA[1] - GOLDSTONE[1],
            CANBERRA[2] - GOLDSTONE[2],
        ];
        let expected = (b[0] * (source[0] - spacecraft[0])
            + b[1] * (source[1] - spacecraft[1])
            + b[2] * (source[2] - spacecraft[2]))
            / SPEED_OF_LIGHT;
        assert!(
            (got - expected).abs() < 1e-3 * expected.abs(),
            "{got:.9e} s against the plane-wave {expected:.9e} s"
        );
        // And the quasar term is doing work: without it the observable is the raw TDOA, which is
        // orders larger and carries the station clock the differencing exists to remove.
        let raw: f64 = tdoa(&state, &geometry);
        assert!(
            raw.abs() > 1e2 * got.abs(),
            "TDOA {raw:.6e} s should dwarf the differenced {got:.6e} s"
        );
    }
    #[test]
    fn the_corona_delay_and_delta_dor_carry_their_partials() {
        // Both are written over `Scalar` for one reason: the filter needs ∂observable/∂state, and a
        // correction evaluated at `f64` enters with a zero partial — which is worse than not
        // applying it, because it biases the residual without widening the covariance. Central
        // differences on the same code are not an independent oracle for the *value*, but they are
        // one for the *derivative*, which is the thing no other test here touches.
        let series = CoronaSeries::new(&AKSIM_PAVLOV_2022).unwrap();
        let (receiver, transmitter) = conjunction(8.0 * SOLAR_RADIUS);
        let corona = |r: &[f64; 3]| -> f64 {
            let lifted = [
                Dual::<1>::constant(r[0]),
                Dual::<1>::constant(r[1]),
                Dual::<1>::constant(r[2]),
            ];
            let emitter = transmitter.map(Dual::<1>::constant);
            solar_corona_delay(
                &series,
                &emitter,
                &lifted,
                8.4e9,
                PlasmaSignature::GroupDelay,
            )
            .unwrap()
            .value()
        };
        for axis in 0..3 {
            let mut lifted = receiver.map(Dual::<1>::constant);
            lifted[axis] = Dual::variable(receiver[axis], 0);
            let emitter = transmitter.map(Dual::<1>::constant);
            let analytic = solar_corona_delay(
                &series,
                &emitter,
                &lifted,
                8.4e9,
                PlasmaSignature::GroupDelay,
            )
            .unwrap()
            .grad()[0];
            let step = 1e4;
            let (mut up, mut down) = (receiver, receiver);
            up[axis] += step;
            down[axis] -= step;
            let numeric = (corona(&up) - corona(&down)) / (2.0 * step);
            assert!(
                (analytic - numeric).abs() < 1e-5 * numeric.abs(),
                "axis {axis}: AD {analytic:.9e} m/m against central difference {numeric:.9e}"
            );
        }

        // ΔDOR's partial has a closed form in the far field: ∂/∂r of −B·(r/|r|)/c, which is the
        // baseline projected off the line of sight. Compared here against the dual arithmetic.
        let geometry = baseline();
        let quasar = Quasar {
            right_ascension: 1.3,
            declination: 0.4,
        };
        let direction = quasar.direction();
        let range = 1e11;
        let position = [
            range * direction[0],
            range * direction[1],
            range * direction[2],
        ];
        let b = [
            CANBERRA[0] - GOLDSTONE[0],
            CANBERRA[1] - GOLDSTONE[1],
            CANBERRA[2] - GOLDSTONE[2],
        ];
        let projected = b[0] * direction[0] + b[1] * direction[1] + b[2] * direction[2];
        for axis in 0..3 {
            let mut state = [Dual::<1>::constant(0.0); 6];
            for (slot, x) in state.iter_mut().zip(&position) {
                *slot = Dual::constant(*x);
            }
            state[axis] = Dual::variable(position[axis], 0);
            let analytic = delta_dor(&state, &geometry, quasar).grad()[0];
            let expected = -(b[axis] - projected * direction[axis]) / (range * SPEED_OF_LIGHT);
            assert!(
                (analytic - expected).abs() < 1e-3 * expected.abs(),
                "axis {axis}: AD {analytic:.9e} s/m against the far-field {expected:.9e}"
            );
        }
    }

    #[test]
    fn every_refusal_names_the_input_it_refused() {
        use core::error::Error;
        let cases: [(RadiometricError, &str); 9] = [
            (RadiometricError::EmptyRampTable, "no segments"),
            (RadiometricError::RampTableNotOrdered { index: 3 }, "3"),
            (
                RadiometricError::EpochOutsideRampTable {
                    epoch: 6.0,
                    first: 0.0,
                    last: 12.0,
                },
                "6",
            ),
            (
                RadiometricError::NonPositiveInterval { seconds: -2.0 },
                "-2",
            ),
            (
                RadiometricError::UnrepresentableModulus { component: 1_018 },
                "1018",
            ),
            (RadiometricError::EmptyCoronaSeries, "no terms"),
            (
                RadiometricError::NonPositiveExponent { exponent: -2.5 },
                "-2.5",
            ),
            (RadiometricError::NonPositiveFrequency { hertz: 0.0 }, "0"),
            (
                RadiometricError::LineOfSightInsideTheSun {
                    impact_parameter: 3.5e8,
                },
                "350000000",
            ),
        ];
        for (error, needle) in cases {
            let rendered = format!("{error}");
            assert!(
                rendered.contains(needle),
                "{rendered:?} does not name {needle:?}"
            );
            // The `dyn Error` path is the one a caller upcasting through `?` actually takes.
            let dynamic: &dyn Error = &error;
            assert!(dynamic.source().is_none());
        }
    }
}
