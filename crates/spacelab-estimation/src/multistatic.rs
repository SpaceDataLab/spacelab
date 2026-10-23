//! Observables whose signal path touches more than one terminal: bistatic range and range rate,
//! TDOA, FDOA, and the inter-satellite pair.
//!
//! Everything in [`measurement`](crate::measurement) is one satellite, one station and the straight
//! line between them. The observables here are differences and sums of *two* such legs, and the two
//! legs do not share an epoch — a bistatic radar transmits at one instant, the pulse reflects at a
//! second and arrives at a third; a TDOA pair receives one emission at two different instants. That
//! is the whole content of the ticket sentence "a TDOA whose two legs share an epoch is not a
//! TDOA", and it is why [`Terminal`] carries an epoch alongside its position and why the geometry
//! types refuse a leg that runs backwards in time.
//!
//! # What is here, and what each is the observable of
//!
//! * [`bistatic_range`] / [`bistatic_range_rate`] — one transmitter, a separate receiver, the
//!   satellite reflecting between them. The observable of a bistatic radar SSA network and of a
//!   GNSS-reflectometry receiver.
//! * [`tdoa`] / [`fdoa`] — one emitter, two receivers. The observables of passive RF geolocation.
//! * [`inter_satellite_range`] / [`inter_satellite_range_rate`] — two satellites, no ground at all.
//!   The GRACE-type link and the formation-flying observable.
//!
//! # Conventions, stated because a signature cannot carry them
//!
//! * **Positions are at their own leg's epoch.** Each [`Terminal`] is where that terminal was when
//!   *its* leg touched it. Nothing here propagates a terminal from one epoch to another: the
//!   observables are algebra over positions the caller has already placed in time, exactly as the
//!   single-station observables are evaluated at an epoch [`light_time`](crate::light_time) settled.
//! * **The bistatic range is the sum of the two legs**, metres, not their half-sum. A system that
//!   publishes the half-sum is publishing a mean one-way range and the caller halves it.
//! * **TDOA is `t_second − t_first`**, seconds — positive when the signal reaches the second
//!   terminal later. **FDOA is `f_second − f_first`**, hertz, at the stated carrier.
//! * **FDOA is first order in `v/c`.** The transverse-Doppler term depends only on the emitter's
//!   speed and cancels in the difference; what is left out is of order `f·(v/c)²`, which is 1.4 Hz
//!   at 2.2 GHz and 7.5 km/s.
//! * **No light time, no aberration, no atmosphere inside these functions**, for the same reasons
//!   [`measurement`](crate::measurement) states. Corrections attach through
//!   [`modifier`](crate::modifier).
//!
//! # How this meets the decorator layer
//!
//! The single-satellite observables here — bistatic and TDOA/FDOA — implement
//! [`Observable`], so they wrap in the existing
//! [`ShapiroDelay`](crate::modifier::ShapiroDelay), [`Bias`](crate::modifier::Bias) and
//! [`EstimatedBias`](crate::modifier::EstimatedBias) without growing corrections of their own.
//!
//! The inter-satellite pair does not, and cannot: `Observable` is a function of **one** six-element
//! state, and an inter-satellite link is a function of two. It is given [`PairObservable`] and
//! [`pair_partials`] instead, which produce a Jacobian split between the two states — the thing a
//! joint formation-flying fit needs and the reason the pair is worth having at all. Where the remote
//! satellite's ephemeris is *held fixed*, the link is not a new observable: it is
//! [`Range`](crate::modifier::Range) over a [`Station`] carrying the remote's position and velocity,
//! bit for bit, which the unit tests assert rather than assume.

use spacelab_autodiff::{Dual, Scalar, jacobian};

use crate::light_time::SPEED_OF_LIGHT;
use crate::measurement::{Station, range, range_rate};
use crate::modifier::Observable;

/// Why a multi-terminal geometry could not be built.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultistaticError {
    /// A leg was declared to arrive no later than it was emitted. Reported at construction, so an
    /// assembled geometry is one light can traverse in the order the epochs claim.
    NonCausalLeg {
        /// The epoch the leg left its emitter.
        emitted: f64,
        /// The epoch the leg was said to reach its receiver.
        received: f64,
    },
    /// A carrier frequency that is not strictly positive. An FDOA at zero frequency is identically
    /// zero with identically zero partials, which reaches the filter as a silently rank-deficient
    /// row rather than as an error.
    NonPositiveFrequency {
        /// The frequency asked for, hertz.
        frequency: f64,
    },
}

impl core::fmt::Display for MultistaticError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonCausalLeg { emitted, received } => write!(
                f,
                "a leg emitted at {emitted} cannot be received at {received}"
            ),
            Self::NonPositiveFrequency { frequency } => {
                write!(f, "carrier frequency {frequency} Hz is not positive")
            }
        }
    }
}

impl core::error::Error for MultistaticError {}

/// One end of a signal leg: where the terminal is, how it moves, and the epoch its leg touches it.
///
/// The epoch is not decoration. Two legs of one observation happen at two instants, and an
/// equatorial terminal travels at 465 m/s, so a millisecond of arrival-time difference is 0.47 m of
/// terminal displacement.
/// Carrying the epoch is what lets [`BistaticGeometry::leg_residuals`] and
/// [`TdoaGeometry::leg_residuals`] state whether the positions supplied and the epochs supplied
/// describe the same signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Terminal {
    /// Position and velocity in the working frame, at [`epoch`](Self::epoch).
    pub station: Station,
    /// The epoch this terminal transmits or receives at, in the caller's own timescale.
    pub epoch: f64,
}

impl Terminal {
    /// A terminal from a station and the epoch its leg touches it.
    #[must_use]
    pub const fn new(station: Station, epoch: f64) -> Self {
        Self { station, epoch }
    }

    /// A terminal fixed in the working frame — velocity zero, as [`Station::fixed`].
    #[must_use]
    pub const fn fixed(position: [f64; 3], epoch: f64) -> Self {
        Self {
            station: Station::fixed(position),
            epoch,
        }
    }
}

/// Flight time of the straight leg between a terminal and a point, seconds.
fn leg_time(satellite: &[f64; 3], terminal: &Terminal) -> f64 {
    let p = terminal.station.position;
    let d = [
        satellite[0] - p[0],
        satellite[1] - p[1],
        satellite[2] - p[2],
    ];
    libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / SPEED_OF_LIGHT
}

/// Transmitter, reflection and receiver of one bistatic observation, with the three epochs.
///
/// Causality is checked once, here: the pulse must leave the transmitter before it reflects and
/// reflect before it arrives. The fields are private because that is an invariant, not a
/// preference — a geometry that exists is one whose epochs are ordered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BistaticGeometry {
    transmitter: Terminal,
    bounce: f64,
    receiver: Terminal,
}

impl BistaticGeometry {
    /// A bistatic geometry from its two terminals and the epoch of the reflection.
    ///
    /// The satellite state these observables are evaluated at is the state at `bounce`.
    ///
    /// # Errors
    ///
    /// [`MultistaticError::NonCausalLeg`] if the transmission is not strictly before the reflection
    /// or the reflection not strictly before the reception.
    pub fn new(
        transmitter: Terminal,
        bounce: f64,
        receiver: Terminal,
    ) -> Result<Self, MultistaticError> {
        causal(transmitter.epoch, bounce)?;
        causal(bounce, receiver.epoch)?;
        Ok(Self {
            transmitter,
            bounce,
            receiver,
        })
    }

    /// The transmitting terminal, at the epoch it transmitted.
    #[must_use]
    pub const fn transmitter(&self) -> Terminal {
        self.transmitter
    }

    /// The epoch of the reflection — the epoch the satellite state belongs to.
    #[must_use]
    pub const fn bounce(&self) -> f64 {
        self.bounce
    }

    /// The receiving terminal, at the epoch it received.
    #[must_use]
    pub const fn receiver(&self) -> Terminal {
        self.receiver
    }

    /// How far the carried epochs are from the flight times the geometry implies, seconds, up leg
    /// then down leg.
    ///
    /// Zero when the positions and the epochs describe one signal. A caller who solved the two legs
    /// with [`light_time::down_leg`](crate::light_time::down_leg) gets rounding; a caller who reused
    /// one epoch for both legs gets the whole flight time, which is the mistake this reports.
    #[must_use]
    pub fn leg_residuals(&self, satellite: &[f64; 3]) -> [f64; 2] {
        [
            (self.bounce - self.transmitter.epoch) - leg_time(satellite, &self.transmitter),
            (self.receiver.epoch - self.bounce) - leg_time(satellite, &self.receiver),
        ]
    }
}

/// One emission seen by two receivers, with the emission epoch and both reception epochs.
///
/// The two receptions are independent epochs: they coincide only where the satellite sits on the
/// perpendicular bisector of the baseline, which is the surface a zero TDOA measures and a
/// perfectly legal configuration. What is refused is a reception at or before the emission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TdoaGeometry {
    emission: f64,
    first: Terminal,
    second: Terminal,
}

impl TdoaGeometry {
    /// A TDOA geometry from the emission epoch and the two receiving terminals.
    ///
    /// The satellite state these observables are evaluated at is the state at `emission`.
    ///
    /// # Errors
    ///
    /// [`MultistaticError::NonCausalLeg`] if either reception is not strictly after the emission.
    pub fn new(emission: f64, first: Terminal, second: Terminal) -> Result<Self, MultistaticError> {
        causal(emission, first.epoch)?;
        causal(emission, second.epoch)?;
        Ok(Self {
            emission,
            first,
            second,
        })
    }

    /// The epoch of the emission — the epoch the satellite state belongs to.
    #[must_use]
    pub const fn emission(&self) -> f64 {
        self.emission
    }

    /// The terminal the difference is measured *from*.
    #[must_use]
    pub const fn first(&self) -> Terminal {
        self.first
    }

    /// The terminal the difference is measured *to*.
    #[must_use]
    pub const fn second(&self) -> Terminal {
        self.second
    }

    /// The TDOA the carried epochs assert, seconds: `t_second − t_first`.
    ///
    /// This is the *observation* — what two synchronised clocks read — against which
    /// [`tdoa`] is the model. Two geometries built from one epoch return exactly zero here for every
    /// satellite position, which is how that mistake is visible.
    #[must_use]
    pub fn observed(&self) -> f64 {
        self.second.epoch - self.first.epoch
    }

    /// How far the carried epochs are from the flight times the geometry implies, seconds, first
    /// terminal then second.
    #[must_use]
    pub fn leg_residuals(&self, satellite: &[f64; 3]) -> [f64; 2] {
        [
            (self.first.epoch - self.emission) - leg_time(satellite, &self.first),
            (self.second.epoch - self.emission) - leg_time(satellite, &self.second),
        ]
    }
}

fn causal(emitted: f64, received: f64) -> Result<(), MultistaticError> {
    if emitted < received {
        Ok(())
    } else {
        Err(MultistaticError::NonCausalLeg { emitted, received })
    }
}

/// Bistatic range `ρ_tx + ρ_rx`, metres — the path the pulse covered, transmitter to satellite to
/// receiver.
///
/// The satellite state is the state at [`BistaticGeometry::bounce`]. Against Maxima at 40 digits
/// (`derivations/multistatic_observables.mac`) this reproduces 3 163 500.191 334 347 76 m on the
/// fixture geometry to **0 ulp**.
#[must_use]
pub fn bistatic_range<S: Scalar>(satellite: &[S; 6], geometry: &BistaticGeometry) -> S {
    range(satellite, &geometry.transmitter.station) + range(satellite, &geometry.receiver.station)
}

/// Bistatic range rate, m/s — the time derivative of [`bistatic_range`], which is what a bistatic
/// Doppler measures once scaled by the carrier.
///
/// Both legs contribute their own closing rate against their own terminal's velocity;
/// `derivations/multistatic_observables.mac` confirms symbolically that `d|a−b|/dt = (Δr·Δv)/|Δr|`
/// with *both* endpoints moving (residual exactly 0), which the single-station range rate never
/// exercises because its station is fixed.
#[must_use]
pub fn bistatic_range_rate<S: Scalar>(satellite: &[S; 6], geometry: &BistaticGeometry) -> S {
    range_rate(satellite, &geometry.transmitter.station)
        + range_rate(satellite, &geometry.receiver.station)
}

/// Time difference of arrival `(ρ_second − ρ_first)/c`, seconds.
///
/// The level set is a hyperboloid of revolution with the two terminals as foci —
/// `derivations/multistatic_observables.mac` checks that independently, parametrising the sheet and
/// recovering the constant range difference to 4e-34 m at 40 digits. The gradient is
/// `(ρ̂_second − ρ̂_first)/c`, so it vanishes along the baseline: a single pair never resolves the
/// direction it points in, and geolocation needs a second pair.
#[must_use]
pub fn tdoa<S: Scalar>(satellite: &[S; 6], geometry: &TdoaGeometry) -> S {
    (range(satellite, &geometry.second.station) - range(satellite, &geometry.first.station))
        / S::from_f64(SPEED_OF_LIGHT)
}

/// Frequency difference of arrival `(f/c)(ρ̇_first − ρ̇_second)`, hertz — the frequency the second
/// terminal receives minus the frequency the first does, at carrier `frequency`.
///
/// First order in `v/c`, as [the module documentation](self) states. The emitter's transverse
/// Doppler is common to both legs and cancels here exactly; the terms left out are of order
/// `f·(v/c)²`.
#[must_use]
pub fn fdoa<S: Scalar>(satellite: &[S; 6], geometry: &TdoaGeometry, frequency: f64) -> S {
    S::from_f64(frequency)
        * (range_rate(satellite, &geometry.first.station)
            - range_rate(satellite, &geometry.second.station))
        / S::from_f64(SPEED_OF_LIGHT)
}

/// Separation of two satellites, metres — the inter-satellite link's one-way range.
///
/// Both states are variables, which is the difference from [`measurement::range`](crate::measurement::range):
/// a fit over an inter-satellite link solves for both orbits, and the partials with respect to the
/// remote state are not zero. [`pair_partials`] produces both halves.
#[must_use]
pub fn inter_satellite_range<S: Scalar>(local: &[S; 6], remote: &[S; 6]) -> S {
    let d = [
        local[0] - remote[0],
        local[1] - remote[1],
        local[2] - remote[2],
    ];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

/// Inter-satellite range rate, m/s — positive when the two are separating.
#[must_use]
pub fn inter_satellite_range_rate<S: Scalar>(local: &[S; 6], remote: &[S; 6]) -> S {
    let d = [
        local[0] - remote[0],
        local[1] - remote[1],
        local[2] - remote[2],
    ];
    let w = [
        local[3] - remote[3],
        local[4] - remote[4],
        local[5] - remote[5],
    ];
    let separation2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    (d[0] * w[0] + d[1] * w[1] + d[2] * w[2]) * separation2.sqrt().recip()
}

/// [`bistatic_range`] as an [`Observable`], so it composes with the modifier chain.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BistaticRange {
    /// The two terminals and the three epochs.
    pub geometry: BistaticGeometry,
}

impl<const K: usize> Observable<K> for BistaticRange {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        bistatic_range(satellite, &self.geometry)
    }
}

/// [`bistatic_range_rate`] as an [`Observable`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BistaticRangeRate {
    /// The two terminals and the three epochs.
    pub geometry: BistaticGeometry,
}

impl<const K: usize> Observable<K> for BistaticRangeRate {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        bistatic_range_rate(satellite, &self.geometry)
    }
}

/// [`tdoa`] as an [`Observable`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tdoa {
    /// The emission epoch and the two receiving terminals.
    pub geometry: TdoaGeometry,
}

impl<const K: usize> Observable<K> for Tdoa {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        tdoa(satellite, &self.geometry)
    }
}

/// [`fdoa`] as an [`Observable`], carrying the carrier frequency it is measured at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fdoa {
    geometry: TdoaGeometry,
    frequency: f64,
}

impl Fdoa {
    /// An FDOA observable on `geometry` at carrier `frequency`, hertz.
    ///
    /// # Errors
    ///
    /// [`MultistaticError::NonPositiveFrequency`] if `frequency` is not strictly positive.
    pub fn new(geometry: TdoaGeometry, frequency: f64) -> Result<Self, MultistaticError> {
        if frequency > 0.0 {
            Ok(Self {
                geometry,
                frequency,
            })
        } else {
            Err(MultistaticError::NonPositiveFrequency { frequency })
        }
    }

    /// The emission epoch and the two receiving terminals.
    #[must_use]
    pub const fn geometry(&self) -> TdoaGeometry {
        self.geometry
    }

    /// The carrier frequency, hertz.
    #[must_use]
    pub const fn frequency(&self) -> f64 {
        self.frequency
    }
}

impl<const K: usize> Observable<K> for Fdoa {
    fn value<S: Scalar>(&self, satellite: &[S; 6], _parameters: &[S; K]) -> S {
        fdoa(satellite, &self.geometry, self.frequency)
    }
}

/// An observable of **two** satellite states and the `K` estimated measurement parameters.
///
/// The companion of [`Observable`] for links that have no ground end.
/// It is a separate trait rather than an `Observable` over a twelve-element state because the
/// filter's state *is* six elements per satellite: a joint fit carries two blocks and needs the
/// Jacobian split between them, which [`pair_partials`] returns and a flattened twelve-vector would
/// have to be re-split at every call site.
pub trait PairObservable<const K: usize> {
    /// The observable's value at these two states and these parameters.
    fn value<S: Scalar>(&self, local: &[S; 6], remote: &[S; 6], parameters: &[S; K]) -> S;
}

/// [`inter_satellite_range`] as a [`PairObservable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterSatelliteRange;

impl<const K: usize> PairObservable<K> for InterSatelliteRange {
    fn value<S: Scalar>(&self, local: &[S; 6], remote: &[S; 6], _parameters: &[S; K]) -> S {
        inter_satellite_range(local, remote)
    }
}

/// [`inter_satellite_range_rate`] as a [`PairObservable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterSatelliteRangeRate;

impl<const K: usize> PairObservable<K> for InterSatelliteRangeRate {
    fn value<S: Scalar>(&self, local: &[S; 6], remote: &[S; 6], _parameters: &[S; K]) -> S {
        inter_satellite_range_rate(local, remote)
    }
}

/// One inter-satellite observation's value and its partials, from automatic differentiation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairPartials<const K: usize> {
    /// The observable's value.
    pub value: f64,
    /// `∂value/∂local(t)` — chain it through the *local* satellite's state transition matrix.
    pub local: [f64; 6],
    /// `∂value/∂remote(t)` — chain it through the *remote* satellite's state transition matrix. A
    /// fit that drops this block is fitting one orbit to a link that constrains two.
    pub remote: [f64; 6],
    /// `∂value/∂parameter` for each estimated measurement parameter. These do not go through either
    /// state transition matrix.
    pub parameters: [f64; K],
}

/// Evaluate a [`PairObservable`] and its partials at one observation.
///
/// Three automatic-differentiation passes — local block, remote block, parameters — for the same
/// reason [`modifier::partials`](crate::modifier::partials) takes two: the seeded dimension is a
/// const generic and `12 + K` cannot be spelled on stable Rust. All three read one source, so the
/// value they agree on is the value the partials belong to.
#[must_use]
pub fn pair_partials<const K: usize, O: PairObservable<K>>(
    observable: &O,
    local: &[f64; 6],
    remote: &[f64; 6],
    parameters: &[f64; K],
) -> PairPartials<K> {
    let mut lifted_parameters = [Dual::<6>::constant(0.0); K];
    for (slot, p) in lifted_parameters.iter_mut().zip(parameters) {
        *slot = Dual::constant(*p);
    }
    let frozen_remote = constants::<6>(remote);
    let (value, local_jacobian) = jacobian::<6, 1, _>(
        |s| [observable.value(s, &frozen_remote, &lifted_parameters)],
        local,
    );
    let frozen_local = constants::<6>(local);
    let (_, remote_jacobian) = jacobian::<6, 1, _>(
        |s| [observable.value(&frozen_local, s, &lifted_parameters)],
        remote,
    );

    let mut lifted_local = [Dual::<K>::constant(0.0); 6];
    let mut lifted_remote = [Dual::<K>::constant(0.0); 6];
    for i in 0..6 {
        lifted_local[i] = Dual::constant(local[i]);
        lifted_remote[i] = Dual::constant(remote[i]);
    }
    let (_, parameter_jacobian) = jacobian::<K, 1, _>(
        |p| [observable.value(&lifted_local, &lifted_remote, p)],
        parameters,
    );

    PairPartials {
        value: value[0],
        local: first_column(&local_jacobian),
        remote: first_column(&remote_jacobian),
        parameters: first_column(&parameter_jacobian),
    }
}

/// A six-element state lifted to duals of width `N` with zero derivative — the block that is *not*
/// being seeded in one pass.
fn constants<const N: usize>(state: &[f64; 6]) -> [Dual<N>; 6] {
    let mut lifted = [Dual::<N>::constant(0.0); 6];
    for (slot, x) in lifted.iter_mut().zip(state) {
        *slot = Dual::constant(*x);
    }
    lifted
}

/// The single output column of a scalar observable's Jacobian.
fn first_column<const N: usize>(jacobian: &[[f64; 1]; N]) -> [f64; N] {
    let mut column = [0.0; N];
    for (slot, row) in column.iter_mut().zip(jacobian) {
        *slot = row[0];
    }
    column
}

#[cfg(test)]
mod tests {
    // The Maxima references below are written at the precision they were computed to, not at the
    // shortest `f64` that round-trips: a truncated constant cannot be checked against the source.
    // Digit separators are left out for the same reason — the constant is compared by eye against
    // the printed output of `derivations/multistatic_observables.mac`.
    #![allow(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::needless_range_loop
    )]

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    /// The fixture of `derivations/multistatic_observables.mac`, in metres and m/s.
    const SATELLITE: [f64; 6] = [
        3_961_000.0,
        2_614_000.0,
        5_215_000.0,
        -3_140.0,
        6_190.0,
        -700.0,
    ];
    const REMOTE: [f64; 6] = [
        4_512_000.0,
        1_878_000.0,
        4_903_000.0,
        -2_050.0,
        5_780.0,
        2_400.0,
    ];
    const CARRIER: f64 = 2.2e9;

    fn first_terminal() -> Terminal {
        Terminal::new(
            Station {
                position: [4_075_530.0, 931_781.0, 4_801_618.0],
                velocity: [-67.9, 297.2, 0.0],
            },
            0.0,
        )
    }

    fn second_terminal() -> Terminal {
        Terminal::new(
            Station {
                position: [3_183_899.0, 1_421_478.0, 5_322_820.0],
                velocity: [-103.6, 232.2, 0.0],
            },
            1.0,
        )
    }

    /// The two terminals as a bistatic pair: transmit at 0, reflect at 1, receive at 2.
    fn bistatic() -> BistaticGeometry {
        let mut receiver = second_terminal();
        receiver.epoch = 2.0;
        BistaticGeometry::new(first_terminal(), 1.0, receiver).expect("an ordered geometry")
    }

    /// The two terminals as a TDOA pair, emitting at 0 and receiving at 1 and 2.
    fn tdoa_pair() -> TdoaGeometry {
        let mut second = second_terminal();
        second.epoch = 2.0;
        TdoaGeometry::new(
            0.0,
            {
                let mut first = first_terminal();
                first.epoch = 1.0;
                first
            },
            second,
        )
        .expect("an ordered geometry")
    }

    /// Every two-leg observable is the single-leg observables combined, and nothing else: no
    /// rearranged sum, no second `sqrt`, no fused multiply. Checked at zero ulps against the
    /// functions [`crate::measurement`] already verifies.
    #[test]
    fn the_two_leg_observables_are_the_single_leg_ones_combined() {
        let (b, t) = (bistatic(), tdoa_pair());
        let tx = b.transmitter().station;
        let rx = b.receiver().station;
        assert_within_ulps(
            bistatic_range(&SATELLITE, &b),
            range(&SATELLITE, &tx) + range(&SATELLITE, &rx),
            0,
            "the bistatic range",
        );
        assert_within_ulps(
            bistatic_range_rate(&SATELLITE, &b),
            range_rate(&SATELLITE, &tx) + range_rate(&SATELLITE, &rx),
            0,
            "the bistatic range rate",
        );
        let (one, two) = (t.first().station, t.second().station);
        assert_within_ulps(
            tdoa(&SATELLITE, &t),
            (range(&SATELLITE, &two) - range(&SATELLITE, &one)) / SPEED_OF_LIGHT,
            0,
            "the TDOA",
        );
        assert_within_ulps(
            fdoa(&SATELLITE, &t, CARRIER),
            CARRIER * (range_rate(&SATELLITE, &one) - range_rate(&SATELLITE, &two))
                / SPEED_OF_LIGHT,
            0,
            "the FDOA",
        );
    }

    /// Each observable against the closed form Maxima evaluated at 40 digits on this geometry
    /// (`derivations/multistatic_observables.mac`). Measured agreement: **0 ulp** on five of the
    /// six, and **1 ulp** on the TDOA — which is the only one that differences two nearly equal
    /// lengths, 1.427 Mm against 1.736 Mm, and so is the only one that loses a digit to
    /// cancellation before the division by `c`.
    #[test]
    fn every_observable_matches_the_maxima_closed_form() {
        let (b, t) = (bistatic(), tdoa_pair());
        assert_within_ulps(
            bistatic_range(&SATELLITE, &b),
            3.16350019133434775975857394752e6,
            0,
            "the bistatic range against Maxima",
        );
        assert_within_ulps(
            bistatic_range_rate(&SATELLITE, &b),
            9.12320439456152896377995923281e3,
            0,
            "the bistatic range rate against Maxima",
        );
        assert_within_ulps(
            tdoa(&SATELLITE, &t),
            -1.02936463346488750872129618964e-3,
            1,
            "the TDOA against Maxima",
        );
        assert_within_ulps(
            fdoa(&SATELLITE, &t, CARRIER),
            1.73842815162547150815909549816e4,
            0,
            "the FDOA against Maxima",
        );
        assert_within_ulps(
            inter_satellite_range(&SATELLITE, &REMOTE),
            9.70897007926175452790431266953e5,
            0,
            "the inter-satellite range against Maxima",
        );
        assert_within_ulps(
            inter_satellite_range_rate(&SATELLITE, &REMOTE),
            -6.67939024124905219627190708466e1,
            0,
            "the inter-satellite range rate against Maxima",
        );
    }

    /// The bistatic gradient is the sum of the two unit line-of-sight vectors, which Maxima
    /// establishes symbolically (residual exactly `[0, 0, 0]`) and evaluates at this geometry.
    /// Measured: the AD partials reproduce those three numbers to **1 ulp** each, and give exactly
    /// zero for the three velocity partials a range cannot have.
    #[test]
    fn the_bistatic_gradient_is_the_sum_of_the_two_unit_vectors() {
        let b = bistatic();
        let (_, jac) = jacobian::<6, 1, _>(|s| [bistatic_range(s, &b)], &SATELLITE);
        let maxima = [
            4.78425527005991124489541937078e-1,
            1.80441329126225152612351648908e0,
            1.62583515565325185868685719401e-1,
        ];
        for i in 0..3 {
            assert_within_ulps(jac[i][0], maxima[i], 1, "a bistatic position partial");
        }
        // A range does not depend on velocity.
        for i in 3..6 {
            assert_within_ulps(jac[i][0], 0.0, 0, "a bistatic velocity partial");
        }
    }

    /// The TDOA gradient is `(ρ̂_second − ρ̂_first)/c`, so it can never exceed `2/c` and it vanishes
    /// along the baseline — the reason one pair cannot geolocate. Checked against the closed form at
    /// the fixture, where it measures 3.94e-9 s/m against the 6.67e-9 ceiling.
    #[test]
    fn the_tdoa_gradient_is_the_difference_of_the_two_unit_vectors() {
        let t = tdoa_pair();
        let (_, jac) = jacobian::<6, 1, _>(|s| [tdoa(s, &t)], &SATELLITE);
        let unit = |terminal: &Terminal| {
            let p = terminal.station.position;
            let d = [
                SATELLITE[0] - p[0],
                SATELLITE[1] - p[1],
                SATELLITE[2] - p[2],
            ];
            let n = libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            [d[0] / n, d[1] / n, d[2] / n]
        };
        let (one, two) = (unit(&t.first()), unit(&t.second()));
        let mut norm = 0.0;
        for i in 0..3 {
            let want = (two[i] - one[i]) / SPEED_OF_LIGHT;
            assert!(
                (jac[i][0] - want).abs() <= 1e-14 * want.abs(),
                "∂TDOA/∂r[{i}] = {} against {want}",
                jac[i][0]
            );
            norm += jac[i][0] * jac[i][0];
        }
        let norm = libm::sqrt(norm);
        assert!(
            norm < 2.0 / SPEED_OF_LIGHT,
            "a TDOA gradient of {norm} exceeds the 2/c ceiling"
        );
    }

    /// A TDOA is zero exactly on the perpendicular bisector of the baseline, and that is a legal
    /// geometry rather than the degenerate one the epoch check exists to catch: both receptions
    /// then happen at the same epoch.
    #[test]
    fn the_tdoa_vanishes_on_the_perpendicular_bisector() {
        let first = Terminal::fixed([-1.0e6, 0.0, 0.0], 1.0);
        let second = Terminal::fixed([1.0e6, 0.0, 0.0], 1.0);
        let geometry = TdoaGeometry::new(0.0, first, second).expect("equal epochs are legal");
        let overhead = [0.0, 3.0e6, 4.0e6, 1.0, 2.0, 3.0];
        assert_within_ulps(
            tdoa(&overhead, &geometry),
            0.0,
            0,
            "the TDOA on the bisector",
        );
        assert_within_ulps(geometry.observed(), 0.0, 0, "the observed TDOA");
    }

    /// The FDOA is zero when both legs close at the same rate, and its sign says which terminal sees
    /// the higher frequency: the one the emitter is approaching faster.
    #[test]
    fn the_fdoa_sign_follows_the_closing_terminal() {
        let t = tdoa_pair();
        let (one, two) = (t.first().station, t.second().station);
        let approaching_second = fdoa(&SATELLITE, &t, CARRIER);
        assert!(
            range_rate(&SATELLITE, &one) > range_rate(&SATELLITE, &two),
            "the fixture must have the first leg opening faster",
        );
        assert!(
            approaching_second > 0.0,
            "the terminal the emitter recedes from slower must see the higher frequency, got {approaching_second}"
        );
    }

    /// A leg that arrives no later than it left is refused at construction, in every position it can
    /// appear in, so an assembled geometry is one light can traverse.
    #[test]
    fn a_non_causal_leg_is_refused_at_construction() {
        let a = Terminal::fixed([1.0e6, 0.0, 0.0], 0.0);
        let b = Terminal::fixed([0.0, 1.0e6, 0.0], 2.0);
        assert_eq!(
            BistaticGeometry::new(a, -1.0, b),
            Err(MultistaticError::NonCausalLeg {
                emitted: 0.0,
                received: -1.0,
            })
        );
        assert_eq!(
            BistaticGeometry::new(a, 3.0, b),
            Err(MultistaticError::NonCausalLeg {
                emitted: 3.0,
                received: 2.0,
            })
        );
        // Equal epochs are a zero-length leg, not a legal one.
        assert!(BistaticGeometry::new(a, 0.0, b).is_err());
        assert!(TdoaGeometry::new(2.0, a, b).is_err());
        assert!(TdoaGeometry::new(-1.0, a, b).is_ok());
    }

    /// The epochs and the positions have to describe one signal, and the residuals say whether they
    /// do. Built from the flight times themselves the residuals are rounding; built by reusing one
    /// epoch for both legs — the mistake the ticket names — the residual is the whole flight time,
    /// 5.79 ms and 4.76 ms on this geometry.
    #[test]
    fn the_leg_residuals_expose_epochs_that_do_not_match_the_geometry() {
        let position = [SATELLITE[0], SATELLITE[1], SATELLITE[2]];
        let mut transmitter = first_terminal();
        let mut receiver = second_terminal();
        transmitter.epoch = 0.0;
        let up = leg_time(&position, &transmitter);
        let down = leg_time(&position, &receiver);
        receiver.epoch = up + down;
        let solved = BistaticGeometry::new(transmitter, up, receiver).expect("ordered");
        for residual in solved.leg_residuals(&position) {
            assert!(
                libm::fabs(residual) < 1e-18,
                "a solved geometry left {residual} s of leg residual"
            );
        }

        // One epoch for both receptions: the residual is the flight time that was never accounted.
        let shared = TdoaGeometry::new(
            0.0,
            {
                let mut t = first_terminal();
                t.epoch = 1.0;
                t
            },
            {
                let mut t = second_terminal();
                t.epoch = 1.0;
                t
            },
        )
        .expect("ordered");
        let residuals = shared.leg_residuals(&position);
        assert!(
            libm::fabs(residuals[0] - (1.0 - up)) < 1e-15
                && libm::fabs(residuals[1] - (1.0 - down)) < 1e-15,
            "the shared-epoch residuals {residuals:?} do not name the missing flight times"
        );
        assert_within_ulps(shared.observed(), 0.0, 0, "a shared epoch reads zero TDOA");
    }

    /// The flight time of one leg, on its own. It reaches a caller only through the residuals, where
    /// it stands beside an epoch gap three orders of magnitude larger: one ulp of a 5.79 ms flight
    /// time is 1/63 of an ulp of the 0.494 s residual it is subtracted from, so a residual checked at
    /// zero ulps still cannot see the flight time's own last digits. Pinned here against Maxima at 40
    /// digits, where nothing is added to it, on both legs of the fixture.
    #[test]
    fn the_leg_flight_time_is_the_leg_range_over_c() {
        let position = [SATELLITE[0], SATELLITE[1], SATELLITE[2]];
        assert_within_ulps(
            leg_time(&position, &first_terminal()),
            5.79083271164055675258452927652e-3,
            0,
            "the first leg's flight time against Maxima",
        );
        assert_within_ulps(
            leg_time(&position, &second_terminal()),
            4.76146807817566924386323308688e-3,
            0,
            "the second leg's flight time against Maxima",
        );
    }

    /// A leg residual is an epoch gap minus a flight time, and that composition is the whole content
    /// of the type: the gap alone says nothing about the geometry, the flight time alone says nothing
    /// about the epochs. The epochs here are placed by hand and wrongly, as a caller who never solved
    /// the light time places them, and they are exact binary fractions, so the gaps are the same
    /// numbers in Maxima as in `f64` and the residual is the oracle's to 40 digits.
    #[test]
    fn the_leg_residuals_are_the_epoch_gaps_minus_the_flight_times() {
        let position = [SATELLITE[0], SATELLITE[1], SATELLITE[2]];
        let mut transmitter = first_terminal();
        transmitter.epoch = 0.25;
        let mut receiver = second_terminal();
        receiver.epoch = 1.5;
        let unsolved = BistaticGeometry::new(transmitter, 0.75, receiver).expect("ordered");
        let residuals = unsolved.leg_residuals(&position);
        assert_within_ulps(
            residuals[0],
            4.94209167288359443247415470723e-1,
            0,
            "the bistatic up-leg residual against Maxima",
        );
        assert_within_ulps(
            residuals[1],
            7.45238531921824330756136766913e-1,
            0,
            "the bistatic down-leg residual against Maxima",
        );

        let mut first = first_terminal();
        first.epoch = 1.25;
        let mut second = second_terminal();
        second.epoch = 1.75;
        let pair = TdoaGeometry::new(0.5, first, second).expect("ordered");
        let residuals = pair.leg_residuals(&position);
        assert_within_ulps(
            residuals[0],
            7.44209167288359443247415470723e-1,
            0,
            "the TDOA first-leg residual against Maxima",
        );
        assert_within_ulps(
            residuals[1],
            1.24523853192182433075613676691e0,
            0,
            "the TDOA second-leg residual against Maxima",
        );
    }

    /// A geometry hands back the epochs it was built from. Nothing inside the module reads them —
    /// the residuals take the private fields directly — so an accessor answering a constant would be
    /// invisible at every other assertion here, and the epoch of the reflection is what tells a
    /// caller which state to evaluate the observables at. None of these epochs is 0, 1 or −1 and the
    /// two receptions differ, so a constant answer is a visibly wrong one.
    #[test]
    fn a_geometry_reports_the_epochs_it_was_built_from() {
        let mut transmitter = first_terminal();
        transmitter.epoch = 0.25;
        let mut receiver = second_terminal();
        receiver.epoch = 1.5;
        let b = BistaticGeometry::new(transmitter, 0.75, receiver).expect("ordered");
        assert_within_ulps(b.transmitter().epoch, 0.25, 0, "the transmission epoch");
        assert_within_ulps(b.bounce(), 0.75, 0, "the reflection epoch");
        assert_within_ulps(b.receiver().epoch, 1.5, 0, "the reception epoch");

        let mut first = first_terminal();
        first.epoch = 1.25;
        let mut second = second_terminal();
        second.epoch = 1.75;
        let t = TdoaGeometry::new(0.5, first, second).expect("ordered");
        assert_within_ulps(t.emission(), 0.5, 0, "the emission epoch");
        assert_within_ulps(t.first().epoch, 1.25, 0, "the first reception epoch");
        assert_within_ulps(t.second().epoch, 1.75, 0, "the second reception epoch");
        // The observation the model is fitted against: t_second − t_first, positive when the signal
        // reaches the second terminal later.
        assert_within_ulps(t.observed(), 0.5, 0, "the observed TDOA");
    }

    /// Each refusal names the numbers it refused. A geometry is rejected at construction and the
    /// caller sees nothing else of it, so a message that omits the epochs or the frequency costs the
    /// caller the diagnosis.
    #[test]
    fn every_refusal_names_the_epochs_or_the_frequency_it_refused() {
        let messages = [
            (
                MultistaticError::NonCausalLeg {
                    emitted: 2.0,
                    received: 1.5,
                },
                "a leg emitted at 2 cannot be received at 1.5",
            ),
            (
                MultistaticError::NonPositiveFrequency { frequency: -3.25 },
                "carrier frequency -3.25 Hz is not positive",
            ),
        ];
        for (error, expected) in messages {
            assert_eq!(format!("{error}"), expected);
            let dynamic: &dyn core::error::Error = &error;
            assert_eq!(format!("{dynamic}"), expected);
        }
    }

    /// A carrier that is not a carrier is refused, because an FDOA at zero frequency is an
    /// identically zero row the filter cannot distinguish from a real one.
    #[test]
    fn a_non_positive_carrier_frequency_is_refused() {
        assert_eq!(
            Fdoa::new(tdoa_pair(), 0.0),
            Err(MultistaticError::NonPositiveFrequency { frequency: 0.0 })
        );
        assert!(Fdoa::new(tdoa_pair(), -1.0).is_err());
        let ok = Fdoa::new(tdoa_pair(), CARRIER).expect("a real carrier");
        assert_within_ulps(ok.frequency(), CARRIER, 0, "the carried frequency");
        assert_eq!(ok.geometry(), tdoa_pair());
    }

    /// Lifting into the decorator layer adds no arithmetic: each observable reproduces its free
    /// function to the bit.
    #[test]
    fn every_observable_reproduces_the_free_function_to_the_ulp() {
        let none: [f64; 0] = [];
        let (b, t) = (bistatic(), tdoa_pair());
        assert_within_ulps(
            BistaticRange { geometry: b }.value(&SATELLITE, &none),
            bistatic_range(&SATELLITE, &b),
            0,
            "the bistatic range",
        );
        assert_within_ulps(
            BistaticRangeRate { geometry: b }.value(&SATELLITE, &none),
            bistatic_range_rate(&SATELLITE, &b),
            0,
            "the bistatic range rate",
        );
        assert_within_ulps(
            Tdoa { geometry: t }.value(&SATELLITE, &none),
            tdoa(&SATELLITE, &t),
            0,
            "the TDOA",
        );
        assert_within_ulps(
            Fdoa::new(t, CARRIER)
                .expect("a real carrier")
                .value(&SATELLITE, &none),
            fdoa(&SATELLITE, &t, CARRIER),
            0,
            "the FDOA",
        );
    }

    /// With the remote ephemeris held fixed an inter-satellite link is not a new observable: it is
    /// the single-station range over a *moving* station, bit for bit, in value and in all six local
    /// partials. This is what makes the pair trait worth its own existence — it is needed only when
    /// the remote state is also being solved for.
    #[test]
    fn a_link_to_a_known_remote_is_the_moving_station_range() {
        let none: [f64; 0] = [];
        let station = Station {
            position: [REMOTE[0], REMOTE[1], REMOTE[2]],
            velocity: [REMOTE[3], REMOTE[4], REMOTE[5]],
        };
        let p = pair_partials(&InterSatelliteRange, &SATELLITE, &REMOTE, &none);
        assert_within_ulps(
            p.value,
            range(&SATELLITE, &station),
            0,
            "the link range against the moving-station range",
        );
        assert_within_ulps(
            pair_partials(&InterSatelliteRangeRate, &SATELLITE, &REMOTE, &none).value,
            range_rate(&SATELLITE, &station),
            0,
            "the link rate against the moving-station range rate",
        );
        let (_, jac) = jacobian::<6, 1, _>(|s| [range(s, &station)], &SATELLITE);
        for i in 0..6 {
            assert_within_ulps(p.local[i], jac[i][0], 0, "a local partial");
        }
    }

    /// The pair Jacobian is split between the two states, and for a separation it is antisymmetric:
    /// moving one satellite along the line of sight and moving the other the other way are the same
    /// change of range. Checked to the bit — the two blocks come from two independent AD passes, so
    /// this is a statement about the passes agreeing, not about one being copied.
    #[test]
    fn the_pair_jacobian_is_antisymmetric_for_a_separation() {
        let none: [f64; 0] = [];
        let p = pair_partials(&InterSatelliteRange, &SATELLITE, &REMOTE, &none);
        for i in 0..6 {
            assert_within_ulps(p.remote[i], -p.local[i], 0, "the mirrored partial");
        }
        let rate = pair_partials(&InterSatelliteRangeRate, &SATELLITE, &REMOTE, &none);
        for i in 0..6 {
            assert_within_ulps(
                rate.remote[i],
                -rate.local[i],
                0,
                "the mirrored rate partial",
            );
        }
        // The rate does depend on velocity, unlike the range: a link whose velocity partials
        // vanished would be a range in disguise.
        let speed_partial = libm::sqrt((3..6).map(|i| rate.local[i] * rate.local[i]).sum::<f64>());
        assert!(
            (speed_partial - 1.0).abs() < 1e-15,
            "the rate's velocity gradient is a unit line of sight, got {speed_partial}"
        );
    }

    /// An estimated parameter reaches a pair observable through the same third pass the single-state
    /// layer uses: a bias added to a link is a unit column, and the state blocks do not move.
    #[test]
    fn an_estimated_parameter_reaches_the_pair_partials() {
        struct BiasedLink;
        impl PairObservable<2> for BiasedLink {
            fn value<S: Scalar>(&self, local: &[S; 6], remote: &[S; 6], parameters: &[S; 2]) -> S {
                inter_satellite_range(local, remote) + parameters[1]
            }
        }
        let p = pair_partials(&BiasedLink, &SATELLITE, &REMOTE, &[3.0, -7.5]);
        let bare = pair_partials(&InterSatelliteRange, &SATELLITE, &REMOTE, &[0.0; 2]);
        assert_within_ulps(p.value, bare.value - 7.5, 0, "the biased value");
        assert_within_ulps(p.parameters[1], 1.0, 0, "∂link/∂bias");
        assert_within_ulps(p.parameters[0], 0.0, 0, "∂link/∂(unused parameter)");
        for i in 0..6 {
            assert_within_ulps(p.local[i], bare.local[i], 0, "a local partial");
            assert_within_ulps(p.remote[i], bare.remote[i], 0, "a remote partial");
        }
    }
}
