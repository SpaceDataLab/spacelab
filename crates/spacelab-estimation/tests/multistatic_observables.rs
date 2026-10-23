//! The multi-terminal observables end to end: AD partials against a central difference, and one
//! synthetic orbit determination per observable whose posterior passes the NEES consistency test.
//!
//! `orbit_determination.rs` closes the same loop for range and range rate from a fixed station.
//! What is new here is that each observable is built from two legs at two epochs, and that the
//! terminals *move* — the sites rotate with the Earth and each is taken at the epoch its own leg
//! touches it, solved from the flight time rather than assumed equal to the observation epoch.
//!
//! # Units
//!
//! The dynamics are canonical (`μ = 1`, length unit 7000 km), as in the sibling tests, because the
//! integrator's tolerance is then one number for the state and the state transition matrix alike.
//! The *measurements* are SI — metres, seconds, hertz — because a TDOA divided by a canonical speed
//! of light would not be a time. The boundary between the two is one diagonal scaling in the chain
//! rule, `∂m/∂x_canonical = ∂m/∂x_SI · diag(LU, LU, LU, LU/TU, LU/TU, LU/TU)`, and every error
//! reported below is in metres.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_autodiff::Scalar;
use spacelab_estimation::modifier::{Observable, Partials, partials};
use spacelab_estimation::multistatic::{
    BistaticGeometry, BistaticRange, BistaticRangeRate, Fdoa, InterSatelliteRange,
    InterSatelliteRangeRate, PairObservable, Tdoa, TdoaGeometry, Terminal, pair_partials,
};
use spacelab_estimation::{
    GaussNewton, ResidualModel, Rng, SPEED_OF_LIGHT, SquareRootKalman, Station,
    chi_square_mean_interval, is_consistent, normalized_squared, range,
};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};

/// IERS 2010 conventional GM, m³/s².
const EARTH_GM: f64 = 3.986_004_418e14;
/// Equatorial radius, m — where the ground terminals sit.
const EARTH_RADIUS: f64 = 6_378_137.0;
/// Earth rotation rate, rad/s: the terminals are not fixed in the working frame.
const EARTH_ROTATION: f64 = 7.292_115e-5;
/// The canonical length unit, metres.
const LENGTH_UNIT: f64 = 7.0e6;
/// S-band carrier the FDOA is measured at, Hz.
const CARRIER: f64 = 2.2e9;
/// One canonical time unit less than a full revolution of the truth orbit.
const SPAN: f64 = 8.0;

/// The canonical time unit, seconds: `√(LU³/GM)`, 927.6 s.
fn time_unit() -> f64 {
    libm::sqrt(LENGTH_UNIT * LENGTH_UNIT * LENGTH_UNIT / EARTH_GM)
}

/// `∂x_SI/∂x_canonical` — diagonal, so a vector.
fn si_scale() -> [f64; 6] {
    let (lu, tu) = (LENGTH_UNIT, time_unit());
    [lu, lu, lu, lu / tu, lu / tu, lu / tu]
}

/// A canonical state in metres and m/s.
fn to_si(x: &[f64; 6]) -> [f64; 6] {
    let s = si_scale();
    let mut out = [0.0; 6];
    for i in 0..6 {
        out[i] = x[i] * s[i];
    }
    out
}

/// Two-body in canonical units (`μ = 1`), as in `orbit_determination.rs`.
struct TwoBody;
impl ScalarSystem<6> for TwoBody {
    fn eval<S: Scalar>(&self, _t: S, y: &[S; 6]) -> [S; 6] {
        let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        let inv_r3 = r2.powf(-1.5);
        [
            y[3],
            y[4],
            y[5],
            -(y[0] * inv_r3),
            -(y[1] * inv_r3),
            -(y[2] * inv_r3),
        ]
    }
}

struct TwoBodyState;
impl DynamicalSystem<6> for TwoBodyState {
    fn derivative(&self, t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        *dydt = TwoBody.eval::<f64>(t, y);
    }
}

fn controls() -> Controls {
    Controls::new(Tolerance::new(1e-13, 1e-13), 1e-10)
}

/// A ground terminal's position on the ellipsoid's sphere at epoch zero, metres.
fn base_site(latitude_degrees: f64, longitude_degrees: f64) -> [f64; 3] {
    let (lat, lon) = (
        latitude_degrees.to_radians(),
        longitude_degrees.to_radians(),
    );
    [
        EARTH_RADIUS * lat.cos() * lon.cos(),
        EARTH_RADIUS * lat.cos() * lon.sin(),
        EARTH_RADIUS * lat.sin(),
    ]
}

/// Three sites spread in longitude, so a pair's baseline is not the same direction all day.
fn sites() -> [[f64; 3]; 3] {
    [
        base_site(5.0, 0.0),
        base_site(21.0, 55.0),
        base_site(-12.0, 110.0),
    ]
}

/// Where a site is, and how it is moving, at `seconds` — rotating with the Earth about the third
/// axis. This is what makes a terminal's epoch matter: 465 m/s at the equator, so 0.47 m of motion
/// per millisecond of arrival-time difference.
fn site(base: [f64; 3], seconds: f64) -> Station {
    let (s, c) = (
        libm::sin(EARTH_ROTATION * seconds),
        libm::cos(EARTH_ROTATION * seconds),
    );
    let position = [
        c * base[0] - s * base[1],
        s * base[0] + c * base[1],
        base[2],
    ];
    Station {
        position,
        velocity: [
            -EARTH_ROTATION * position[1],
            EARTH_ROTATION * position[0],
            0.0,
        ],
    }
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
}

/// Solve one leg's flight time against a *moving* terminal: `direction` is `-1` for a leg that
/// left the terminal before `epoch` and `+1` for one that reaches it after.
///
/// Four fixed-point passes; the map contracts at `v/c ≈ 1.6e-6` for a ground site, so this is
/// converged to well past `f64` after two.
fn solve_leg(base: [f64; 3], epoch: f64, satellite: [f64; 3], direction: f64) -> (Station, f64) {
    let mut delay = 0.0;
    let mut station = site(base, epoch);
    for _ in 0..4 {
        delay = distance(satellite, station.position) / SPEED_OF_LIGHT;
        station = site(base, epoch + direction * delay);
    }
    (station, delay)
}

/// The bistatic geometry of a pulse that left `transmitter`, reflected off the satellite at
/// `epoch`, and reached `receiver` — each terminal at its own leg's epoch.
fn bistatic(
    transmitter: [f64; 3],
    receiver: [f64; 3],
    epoch: f64,
    satellite: [f64; 3],
) -> BistaticGeometry {
    let (up_station, up) = solve_leg(transmitter, epoch, satellite, -1.0);
    let (down_station, down) = solve_leg(receiver, epoch, satellite, 1.0);
    BistaticGeometry::new(
        Terminal::new(up_station, epoch - up),
        epoch,
        Terminal::new(down_station, epoch + down),
    )
    .expect("a solved bistatic geometry is causal")
}

/// The TDOA geometry of one emission at `epoch` reaching two terminals at two epochs.
fn pair(first: [f64; 3], second: [f64; 3], epoch: f64, satellite: [f64; 3]) -> TdoaGeometry {
    let (one, delay_one) = solve_leg(first, epoch, satellite, 1.0);
    let (two, delay_two) = solve_leg(second, epoch, satellite, 1.0);
    TdoaGeometry::new(
        epoch,
        Terminal::new(one, epoch + delay_one),
        Terminal::new(two, epoch + delay_two),
    )
    .expect("a solved TDOA geometry is causal")
}

/// Which observable one record carries, with the geometry it was taken on.
#[derive(Clone, Copy)]
enum Model {
    BistaticRange(BistaticGeometry),
    BistaticRangeRate(BistaticGeometry),
    Tdoa(TdoaGeometry),
    Fdoa(TdoaGeometry),
}

impl Model {
    fn value(self, satellite: &[f64; 6]) -> f64 {
        let none: [f64; 0] = [];
        match self {
            Self::BistaticRange(geometry) => BistaticRange { geometry }.value(satellite, &none),
            Self::BistaticRangeRate(geometry) => {
                BistaticRangeRate { geometry }.value(satellite, &none)
            }
            Self::Tdoa(geometry) => Tdoa { geometry }.value(satellite, &none),
            Self::Fdoa(geometry) => Fdoa::new(geometry, CARRIER)
                .expect("a real carrier")
                .value(satellite, &none),
        }
    }

    fn linearise(self, satellite: &[f64; 6]) -> Partials<0> {
        let none: [f64; 0] = [];
        match self {
            Self::BistaticRange(geometry) => {
                partials(&BistaticRange { geometry }, satellite, &none)
            }
            Self::BistaticRangeRate(geometry) => {
                partials(&BistaticRangeRate { geometry }, satellite, &none)
            }
            Self::Tdoa(geometry) => partials(&Tdoa { geometry }, satellite, &none),
            Self::Fdoa(geometry) => partials(
                &Fdoa::new(geometry, CARRIER).expect("a real carrier"),
                satellite,
                &none,
            ),
        }
    }
}

/// Which of the four an orbit determination is being run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    BistaticRange,
    BistaticRangeRate,
    Tdoa,
    Fdoa,
}

const KINDS: [Kind; 4] = [
    Kind::BistaticRange,
    Kind::BistaticRangeRate,
    Kind::Tdoa,
    Kind::Fdoa,
];

impl Kind {
    /// The measurement noise, in the observable's own unit: 1 m of bistatic path, 1 cm/s of
    /// bistatic Doppler, 1 ns of arrival-time difference, 0.5 Hz at the S-band carrier.
    fn sigma(self) -> f64 {
        match self {
            Self::BistaticRange => 1.0,
            Self::BistaticRangeRate => 0.01,
            Self::Tdoa => 1.0e-9,
            Self::Fdoa => 0.5,
        }
    }

    fn model(self, a: [f64; 3], b: [f64; 3], epoch: f64, satellite: [f64; 3]) -> Model {
        match self {
            Self::BistaticRange => Model::BistaticRange(bistatic(a, b, epoch, satellite)),
            Self::BistaticRangeRate => Model::BistaticRangeRate(bistatic(a, b, epoch, satellite)),
            Self::Tdoa => Model::Tdoa(pair(a, b, epoch, satellite)),
            Self::Fdoa => Model::Fdoa(pair(a, b, epoch, satellite)),
        }
    }
}

#[derive(Clone, Copy)]
struct Obs {
    time: f64,
    model: Model,
    value: f64,
    sigma: f64,
}

/// The truth orbit, canonical: perigee at one length unit, out of the equatorial plane.
const TRUTH: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.4];

fn observation_times() -> Vec<f64> {
    (1..=12).map(|k| SPAN * f64::from(k) / 13.0).collect()
}

fn truth_trajectory(x: &[f64; 6]) -> impl Fn(f64) -> [f64; 6] {
    let traj = integrate(&Dop853, &TwoBodyState, 0.0, x, SPAN, 0.01, controls())
        .expect("the truth orbit integrates");
    move |t| traj.eval(t)
}

/// Synthesise one kind's observations from the truth orbit over the three sites' three pairs.
fn synthesise(kind: Kind, noise: Option<&mut Rng>) -> Vec<Obs> {
    let truth = truth_trajectory(&TRUTH);
    let s = sites();
    let tu = time_unit();
    let mut rng = noise;
    let mut obs = Vec::new();
    for t in observation_times() {
        let state = to_si(&truth(t));
        let position = [state[0], state[1], state[2]];
        for (a, b) in [(0, 1), (1, 2), (2, 0)] {
            let model = kind.model(s[a], s[b], t * tu, position);
            let clean = model.value(&state);
            let value = match rng {
                Some(ref mut r) => clean + kind.sigma() * r.standard_normal(),
                None => clean,
            };
            obs.push(Obs {
                time: t,
                model,
                value,
                sigma: kind.sigma(),
            });
        }
    }
    obs
}

/// Six canonical epoch states from observations taken in SI units.
struct Determination<'a> {
    obs: &'a [Obs],
}

impl ResidualModel<6> for Determination<'_> {
    fn residuals(&self, x: &[f64; 6], emit: &mut dyn FnMut(&[f64; 6], f64, f64)) {
        let var = Variational::<6, _>::new(&TwoBody);
        let aug0: [f64; 42] = augment::<6, 42>(x);
        let traj = integrate(&Dop853, &var, 0.0, &aug0, SPAN, 0.01, controls())
            .expect("propagated augmented system");
        let scale = si_scale();

        for o in self.obs {
            let aug = traj.eval(o.time);
            let (state, phi) = split::<6, 42>(&aug);
            let p = o.model.linearise(&to_si(&state));

            let mut h = [0.0; 6];
            for j in 0..6 {
                for n in 0..6 {
                    h[j] += p.state[n] * scale[n] * phi[n][j];
                }
            }
            emit(&h, p.value - o.value, 1.0 / (o.sigma * o.sigma));
        }
    }
}

/// Position error of a canonical estimate against the truth, in metres.
fn position_error(x: &[f64; 6]) -> f64 {
    libm::sqrt(
        (0..3)
            .map(|i| (x[i] - TRUTH[i]) * (x[i] - TRUTH[i]))
            .sum::<f64>(),
    ) * LENGTH_UNIT
}

/// A start 0.5 % off in every component — an initial-orbit error, not a nudge.
fn perturbed_start() -> [f64; 6] {
    let mut x0 = TRUTH;
    for i in 0..6 {
        x0[i] += 0.005 * (if i % 2 == 0 { 1.0 } else { -1.0 }) * TRUTH[i].abs().max(0.1);
    }
    x0
}

/// The AD partials against a central-difference Jacobian, over several geometries and all four
/// observables.
///
/// The central difference is the independent route, with its own `√ε` floor — it is the check, not
/// the reference. Measured worst relative disagreement over the 4 observables × 4 satellite states
/// × 3 site pairs = 48 geometries: reported below by the test itself, and asserted under 1e-6, the
/// floor the difference itself carries at these step sizes.
#[test]
fn the_ad_partials_match_a_central_difference_at_every_geometry() {
    let s = sites();
    let states: [[f64; 6]; 4] = [
        [7.0e6, 0.0, 0.0, 0.0, 7.546e3, 3.019e3],
        [3.961e6, 2.614e6, 5.215e6, -3.14e3, 6.19e3, -0.7e3],
        [-4.2e6, 5.1e6, 2.7e6, -5.0e3, -2.1e3, 4.4e3],
        [1.2e7, -8.0e6, 3.0e6, 1.9e3, 2.6e3, -1.1e3],
    ];
    let mut worst = [0.0f64; 4];
    let mut worst_where = ["", "", "", ""];

    for (k, kind) in KINDS.into_iter().enumerate() {
        for state in states {
            let position = [state[0], state[1], state[2]];
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                let model = kind.model(s[a], s[b], 1234.5, position);
                let p = model.linearise(&state);
                for i in 0..6 {
                    // One metre and one metre per second: the same number serves both halves
                    // because the position step is 1e-7 of the state and the observables are
                    // exactly linear in velocity, where a central difference has no truncation
                    // error at all.
                    const STEP: f64 = 1.0;
                    let (mut lo, mut hi) = (state, state);
                    lo[i] -= STEP;
                    hi[i] += STEP;
                    let central = (model.value(&hi) - model.value(&lo)) / (2.0 * STEP);
                    let scale = p.state[i].abs().max(central.abs());
                    let relative = if scale == 0.0 {
                        0.0
                    } else {
                        (p.state[i] - central).abs() / scale
                    };
                    if relative > worst[k] {
                        worst[k] = relative;
                        worst_where[k] = if i < 3 { "position" } else { "velocity" };
                    }
                }
            }
        }
        println!(
            "{:?}: worst AD vs central difference {:.3e} (on a {} partial)",
            kind, worst[k], worst_where[k]
        );
        assert!(
            worst[k] < 1e-6,
            "{:?} disagrees with the central difference by {:.3e}",
            kind,
            worst[k]
        );
    }
}

/// Each observable, on its own, recovers the injected epoch state from noiseless data.
///
/// Three ground sites, three pairs, twelve epochs over one revolution — no other observable and no
/// a-priori. Measured position errors against the injected truth, from a start 0.5 % off in every
/// component, are printed by the test.
#[test]
fn each_observable_recovers_the_injected_state() {
    for kind in KINDS {
        let obs = synthesise(kind, None);
        let (x, cov, report) = GaussNewton {
            max_iterations: 60,
            ..GaussNewton::default()
        }
        .solve(&Determination { obs: &obs }, &perturbed_start())
        .unwrap_or_else(|e| panic!("{kind:?} is not observable from this geometry: {e:?}"));

        assert!(report.converged, "{kind:?} did not converge: {report:?}");
        let error = position_error(&x);
        println!(
            "{:?}: recovered the epoch position to {:.3e} m in {} iterations, cost {:.3e}",
            kind, error, report.iterations, report.cost
        );
        assert!(
            error < 1.0e-3,
            "{kind:?} left {error} m of epoch position error (report {report:?})"
        );
        for i in 0..6 {
            assert!(
                cov[i][i].is_finite() && cov[i][i] > 0.0,
                "{kind:?} reported an unphysical variance {} for state {i}",
                cov[i][i]
            );
        }
    }
}

/// The covariance each observable's fit reports is realistic, by NEES over 40 independent noise
/// realisations per observable.
///
/// Each run draws its own Gaussian measurement noise at the observable's σ, solves from the truth,
/// and forms `(x_true − x̂)ᵀ P⁻¹ (x_true − x̂)` through the covariance's square root. For a
/// consistent fit the mean over the runs is the state dimension, 6; the acceptance band from
/// `chi_square_mean_interval(6, 40, 1.96)` is (4.95, 7.11). The measured means are printed.
#[test]
fn every_observable_supports_a_consistent_posterior() {
    let runs = 40;
    for kind in KINDS {
        let mut rng = Rng::new(0x5eed_0314 + kind as u64);
        let mut total = 0.0;
        for _ in 0..runs {
            let obs = synthesise(kind, Some(&mut rng));
            let (x, cov, report) = GaussNewton::default()
                .solve(&Determination { obs: &obs }, &TRUTH)
                .expect("solved");
            assert!(
                report.converged,
                "{kind:?} run did not converge: {report:?}"
            );

            let mut error = [0.0; 6];
            for i in 0..6 {
                error[i] = TRUTH[i] - x[i];
            }
            let root = SquareRootKalman::<6>::from_covariance([0.0; 6], &cov)
                .expect("the reported covariance is positive definite")
                .covariance_sqrt();
            total += normalized_squared(&root, &error);
        }
        let mean = total / runs as f64;
        let (lo, hi) = chi_square_mean_interval(6, runs, 1.96);
        println!("{kind:?}: mean NEES {mean:.3} in ({lo:.3}, {hi:.3})");
        assert!(
            is_consistent(mean, 6, runs, 1.96),
            "{kind:?} mean NEES {mean} is outside the 95 % band ({lo}, {hi})"
        );
    }
}

// ------------------------------------------------------------------------------------------------
// The inter-satellite pair: twelve states solved jointly, which is the case the six-state
// `Observable` shape cannot express and `pair_partials` exists for.
// ------------------------------------------------------------------------------------------------

/// The remote satellite's truth state, canonical — a 216 km separation from [`TRUTH`].
const REMOTE_TRUTH: [f64; 6] = [0.998_571, 0.028_571, 0.011_429, -0.029_15, 0.999_2, 0.398_9];

#[derive(Clone, Copy)]
enum Link {
    Range,
    RangeRate,
}

#[derive(Clone, Copy)]
struct LinkObs {
    time: f64,
    link: Link,
    value: f64,
    sigma: f64,
}

#[derive(Clone, Copy)]
struct AnchorObs {
    time: f64,
    station: Station,
    value: f64,
    sigma: f64,
}

/// σ of the inter-satellite range, metres, and of its rate, m/s — a microwave ranging instrument,
/// two decades coarser than GRACE-FO's laser.
const LINK_RANGE_SIGMA: f64 = 1.0e-2;
const LINK_RATE_SIGMA: f64 = 1.0e-5;
/// σ of the ground range that anchors the local satellite, metres.
const ANCHOR_SIGMA: f64 = 1.0;

/// The local satellite is anchored by ground ranges; the remote one is seen *only* through the
/// link, so recovering it is a statement about the link observables and nothing else.
struct FormationDetermination<'a> {
    link: &'a [LinkObs],
    anchor: &'a [AnchorObs],
}

impl ResidualModel<12> for FormationDetermination<'_> {
    fn residuals(&self, x: &[f64; 12], emit: &mut dyn FnMut(&[f64; 12], f64, f64)) {
        let var = Variational::<6, _>::new(&TwoBody);
        let local0: [f64; 6] = [x[0], x[1], x[2], x[3], x[4], x[5]];
        let remote0: [f64; 6] = [x[6], x[7], x[8], x[9], x[10], x[11]];
        let arc = |start: &[f64; 6]| {
            integrate(
                &Dop853,
                &var,
                0.0,
                &augment::<6, 42>(start),
                SPAN,
                0.01,
                controls(),
            )
            .expect("propagated augmented system")
        };
        let (local_arc, remote_arc) = (arc(&local0), arc(&remote0));
        let scale = si_scale();

        for o in self.link {
            let (local, local_phi) = split::<6, 42>(&local_arc.eval(o.time));
            let (remote, remote_phi) = split::<6, 42>(&remote_arc.eval(o.time));
            let (local_si, remote_si) = (to_si(&local), to_si(&remote));
            let none: [f64; 0] = [];
            let p = match o.link {
                Link::Range => pair_partials(&InterSatelliteRange, &local_si, &remote_si, &none),
                Link::RangeRate => {
                    pair_partials(&InterSatelliteRangeRate, &local_si, &remote_si, &none)
                }
            };

            let mut h = [0.0; 12];
            for j in 0..6 {
                for n in 0..6 {
                    h[j] += p.local[n] * scale[n] * local_phi[n][j];
                    h[6 + j] += p.remote[n] * scale[n] * remote_phi[n][j];
                }
            }
            emit(&h, p.value - o.value, 1.0 / (o.sigma * o.sigma));
        }

        for o in self.anchor {
            let (local, local_phi) = split::<6, 42>(&local_arc.eval(o.time));
            let none: [f64; 0] = [];
            let p = partials(
                &spacelab_estimation::modifier::Range { station: o.station },
                &to_si(&local),
                &none,
            );
            let mut h = [0.0; 12];
            for j in 0..6 {
                for n in 0..6 {
                    h[j] += p.state[n] * scale[n] * local_phi[n][j];
                }
            }
            emit(&h, p.value - o.value, 1.0 / (o.sigma * o.sigma));
        }
    }
}

type FormationData = (Vec<LinkObs>, Vec<AnchorObs>);

fn synthesise_formation(noise: Option<&mut Rng>) -> FormationData {
    let local = truth_trajectory(&TRUTH);
    let remote = truth_trajectory(&REMOTE_TRUTH);
    let s = sites();
    let tu = time_unit();
    let mut rng = noise;
    let (mut link, mut anchor) = (Vec::new(), Vec::new());

    for t in observation_times() {
        let (a, b) = (to_si(&local(t)), to_si(&remote(t)));
        let none: [f64; 0] = [];
        for (kind, sigma) in [
            (Link::Range, LINK_RANGE_SIGMA),
            (Link::RangeRate, LINK_RATE_SIGMA),
        ] {
            let clean = match kind {
                Link::Range => InterSatelliteRange.value(&a, &b, &none),
                Link::RangeRate => InterSatelliteRangeRate.value(&a, &b, &none),
            };
            let value = match rng {
                Some(ref mut r) => clean + sigma * r.standard_normal(),
                None => clean,
            };
            link.push(LinkObs {
                time: t,
                link: kind,
                value,
                sigma,
            });
        }
        for base in s {
            let station = site(base, t * tu);
            let clean = range(&a, &station);
            let value = match rng {
                Some(ref mut r) => clean + ANCHOR_SIGMA * r.standard_normal(),
                None => clean,
            };
            anchor.push(AnchorObs {
                time: t,
                station,
                value,
                sigma: ANCHOR_SIGMA,
            });
        }
    }
    (link, anchor)
}

/// The two truth states concatenated.
fn formation_truth() -> [f64; 12] {
    let mut x = [0.0; 12];
    x[..6].copy_from_slice(&TRUTH);
    x[6..].copy_from_slice(&REMOTE_TRUTH);
    x
}

/// A joint twelve-state fit recovers *both* orbits, with the remote one seen only through the link.
///
/// This is what the pair partials buy. The ground ranges observe the local satellite alone; every
/// piece of information about the remote one arrives through the inter-satellite range and range
/// rate, so a fit that dropped `PairPartials::remote` would leave the remote block unobservable and
/// the solve rank-deficient. Measured errors are printed.
#[test]
fn a_joint_formation_fit_recovers_both_orbits_through_the_link() {
    let (link, anchor) = synthesise_formation(None);
    let truth = formation_truth();
    let mut x0 = truth;
    for i in 0..12 {
        x0[i] += 0.002 * (if i % 2 == 0 { 1.0 } else { -1.0 }) * truth[i].abs().max(0.1);
    }

    let (x, cov, report) = GaussNewton {
        max_iterations: 60,
        ..GaussNewton::default()
    }
    .solve(
        &FormationDetermination {
            link: &link,
            anchor: &anchor,
        },
        &x0,
    )
    .expect("both orbits are observable from link plus anchor");

    assert!(
        report.converged,
        "the joint fit did not converge: {report:?}"
    );
    let local_error = libm::sqrt(
        (0..3)
            .map(|i| (x[i] - truth[i]) * (x[i] - truth[i]))
            .sum::<f64>(),
    ) * LENGTH_UNIT;
    let remote_error = libm::sqrt(
        (6..9)
            .map(|i| (x[i] - truth[i]) * (x[i] - truth[i]))
            .sum::<f64>(),
    ) * LENGTH_UNIT;
    println!(
        "formation: local epoch position to {local_error:.3e} m, remote to {remote_error:.3e} m, \
         cost {:.3e} in {} iterations",
        report.cost, report.iterations
    );
    assert!(
        local_error < 1.0e-3 && remote_error < 1.0e-3,
        "local {local_error} m, remote {remote_error} m"
    );
    for i in 0..12 {
        assert!(
            cov[i][i].is_finite() && cov[i][i] > 0.0,
            "unphysical variance {} for joint state {i}",
            cov[i][i]
        );
    }
}

/// The joint fit's twelve-state covariance passes NEES over 30 noise realisations.
///
/// The acceptance band from `chi_square_mean_interval(12, 30, 1.96)` is printed with the measured
/// mean.
#[test]
fn the_joint_formation_covariance_passes_the_nees_consistency_test() {
    let runs = 30;
    let truth = formation_truth();
    let mut rng = Rng::new(0x5eed_0f0a);
    let mut total = 0.0;

    for _ in 0..runs {
        let (link, anchor) = synthesise_formation(Some(&mut rng));
        let (x, cov, report) = GaussNewton::default()
            .solve(
                &FormationDetermination {
                    link: &link,
                    anchor: &anchor,
                },
                &truth,
            )
            .expect("solved");
        assert!(report.converged, "a run did not converge: {report:?}");

        let mut error = [0.0; 12];
        for i in 0..12 {
            error[i] = truth[i] - x[i];
        }
        let root = SquareRootKalman::<12>::from_covariance([0.0; 12], &cov)
            .expect("the reported covariance is positive definite")
            .covariance_sqrt();
        total += normalized_squared(&root, &error);
    }

    let mean = total / runs as f64;
    let (lo, hi) = chi_square_mean_interval(12, runs, 1.96);
    println!("formation: mean NEES {mean:.3} in ({lo:.3}, {hi:.3})");
    assert!(
        is_consistent(mean, 12, runs, 1.96),
        "mean NEES {mean} is outside the 95 % band ({lo}, {hi}) for 12 degrees of freedom"
    );
}
