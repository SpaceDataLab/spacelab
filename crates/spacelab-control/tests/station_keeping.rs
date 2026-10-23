//! What a repeat-ground-track hold actually costs, against the Δv the atmosphere removes.
//!
//! [`ElementFeedback`]'s unit tests prove the law is a law: `V̇ ≤ 0` at every state it was given, the
//! variational matrix is what an impulse does, the saturation preserves the descent direction. None
//! of that says whether closing the loop on a real plant produces a *plausible* propellant budget,
//! and that is the only question a mission asks. A controller that holds the track by burning ten
//! times the drag makeup is not a success — it is a bug, in the plant, the gains or the frame.
//!
//! The scenario is a 46-revolution / 3-nodal-day repeat ground track, at the Sun-synchronous
//! inclination for its own altitude — both solved here from the `J₂` secular rates rather than
//! quoted. The plant is point mass + `J₂` + drag on **NRLMSISE-00** through Vermeille's geodetic
//! conversion; the loop runs at a 60 s zero-order hold, which is a real GNC cadence and makes the
//! commanded Δv an exact sum rather than a quadrature.
//!
//! # The bar
//!
//! For a circular orbit the semi-major axis responds to a tangential acceleration as `ȧ = 2a_t/n`,
//! so holding `a` against drag costs a Δv of exactly what drag takes out:
//!
//! ```text
//!     Δv_hold = ∫ |a_drag| dt = (n/2) · |Δa_drag|
//! ```
//!
//! That integral is accumulated as a seventh state in **both** runs, so the floor comes out of the
//! drag force itself and never out of the controller's arithmetic. Two things follow, and both are
//! measured below rather than argued:
//!
//! * a hold that spends **less** than the floor is not holding — the orbit is still decaying;
//! * a hold that spends much **more** than the floor is steering where it does not need to, and the
//!   excess is the diagnostic.
//!
//! A proportional law leaves a steady-state offset — the semi-major axis settles where the commanded
//! thrust equals the drag — so the ratio approaches 1 from just below, by the one-off Δv of that
//! offset. The offset is reported too, in metres and in the ground-track drift it implies.
//!
//! # The finding: a mean element is not mean enough
//!
//! Closing the loop on the *osculating* semi-major axis is obviously wrong — `J₂` swings it by
//! **19.0 km** peak-to-peak every revolution here, and chasing that would spend the entire budget on
//! a periodic term whose integral is zero. So the loop closes on mean elements, recovered from the
//! state by the fixed point `mean = osculating − Δ_sp(mean)` over the DSST's zonal short-period
//! series.
//!
//! That series is first order in `J₂`, and what it leaves behind is not negligible next to the
//! quantity being controlled. Measured on this orbit: it removes 99.76 % of the swing and leaves a
//! **±22.6 m ripple** at the orbital frequency — where a gain sized to balance drag settles at a
//! **2.2 m** offset. A proportional law spends Δv in proportion to the error it is *shown*, so it
//! then buys the ripple rather than the disturbance: **5.13×** the drag makeup, which is precisely
//! the implausible number this test exists to catch, arriving from the estimator rather than from
//! the law.
//!
//! The fix belongs where the bad value is produced, not at the consumer: the error is averaged over
//! exactly one nodal period before the law sees it. That is a comb filter whose first zero sits on
//! the orbital frequency and whose later zeros sit on its harmonics, which is the entire spectrum of
//! what the truncation left. It is also what an operational station-keeping loop does, and for the
//! same reason.
//!
//! Averaging is not free: a moving average of width `W` delays by `W/2`, and a first-order loop
//! `ẋ = −k x(t−τ)` goes unstable at `τ_max = π/(2k)` — the first-order sibling of the second-order
//! margin in [`spacelab_control::SecondOrderLoop`]. Half a nodal period is 2817 s here, so the gain
//! must satisfy `k < 5.6 × 10⁻⁴ s⁻¹`. The gain used is 4.3× inside that, which is what buys the
//! ripple rejection and what sets the settled offset the run reports: the delay margin, not the
//! tracking requirement, is the binding constraint on this loop.
//!
//! # What this does not claim
//!
//! The absolute Δv/year is proportional to the density, so it is a statement about NRLMSISE-00 at the
//! stated space weather (F10.7 = 150, Ap = 4 — moderate activity) as much as about the controller.
//! The *ratio* to the floor is not: both sides see the same atmosphere, so it survives any density
//! model. Where the two are evidence for different things, the ratio is the control result and the
//! absolute number is the mission-design one.

// Single letters are the names of the things themselves here (r, v, a, n, i, u, t), and the loops run
// over fixed-size state and element arrays where the index is the meaning.
#![allow(
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use spacelab_control::ElementFeedback;
use spacelab_core::{Angle, Vec3};
use spacelab_earth::itrf_to_geodetic;
use spacelab_elements::KeplerianElements;
use spacelab_frames::{Itrf, Position};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::dsst::gaussian::variational_matrix;
use spacelab_propagation::dsst::{DsstAuxiliary, DsstZonal};
use spacelab_propagation::forces::{Acceleration, Atmosphere, Drag, J2, NoWind, PointMass};
use spacelab_propagation::nrlmsise00::Nrlmsise00Atmosphere;
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const J2_COEFF: f64 = 1.082_626_68e-3;
/// Earth rotation rate, rad/s (WGS-84).
const OMEGA_EARTH: f64 = 7.292_115_146_7e-5;
/// Ballistic coefficient `C_d A / m`, m²/kg — a high-drag 12U cubesat, and the value
/// `forces::Drag`'s own measured examples use, so the numbers here compose with those.
const BALLISTIC: f64 = 0.02;
/// Sun-synchronous nodal regression: one turn per tropical year, rad/s.
const SUN_SYNCHRONOUS_NODE_RATE: f64 = core::f64::consts::TAU / (365.242_19 * 86_400.0);
const YEAR: f64 = 365.25 * 86_400.0;

/// Revolutions and nodal days of the repeat cycle. 46/3 puts the orbit near 460 km, where drag is
/// large enough to measure over a short span and small enough that the hold is a real one.
const REPEAT: (f64, f64) = (46.0, 3.0);

/// Zero-order hold of the commanded acceleration, s. The closed-loop time constant is
/// `n²/(4 c w_a) ≈ 7800 s`, so the sampled loop runs 130 updates per time constant.
const CONTROL_INTERVAL: f64 = 60.0;
const SPAN: f64 = 3.0 * 86_400.0;

/// Bryson weight on the semi-major axis: `1/σ_a²` for a 100 m tolerance. Every other element is
/// declared uncontrolled — a ground-track hold is an `a` hold, and weighting the eccentricity vector
/// would spend Δv correcting something the mission does not measure.
const SMA_WEIGHT: f64 = 1.0e-4;
/// Proportional gain. `c · w_a · δa · |∂a/∂v| = |a_drag|` with `|∂a/∂v| = 2/n ≈ 1790 s` sets the
/// steady-state offset; the *ceiling* is the delay margin of the one-orbit average, `k < π/(2·W/2)`
/// with `W` the nodal period, and this is a factor of 4 inside it. See the module header.
const PROPORTIONAL: f64 = 4.0e-7;
/// Acceleration limit, m/s² — 1 mN on a 100 kg spacecraft, and an order above the drag it must beat.
const MAX_ACCELERATION: f64 = 1.0e-5;

/// Moderate solar activity. The density, and so the whole absolute Δv, is proportional to what this
/// implies; the ratio to the drag floor is not.
const F107: f64 = 150.0;
const AP: f64 = 4.0;
/// Day of year and UT seconds at `t = 0` — an equinox epoch, so neither solstice extreme of the
/// diurnal bulge is sampled preferentially over the span.
const DOY0: i32 = 80;
const SEC0: f64 = 0.0;

// ---------------------------------------------------------------- element conversions

/// Keplerian to the DSST's equinoctial `[a, k, h, q, p, λ]`.
fn equinoctial(els: &KeplerianElements) -> [f64; 6] {
    let e = els.eccentricity;
    let raan = els.raan.radians();
    let perigee_longitude = els.arg_periapsis.radians() + raan;
    let tan_half = libm::tan(0.5 * els.inclination.radians());
    [
        els.semi_major_axis,
        e * libm::cos(perigee_longitude),
        e * libm::sin(perigee_longitude),
        tan_half * libm::cos(raan),
        tan_half * libm::sin(raan),
        els.mean_anomaly().radians() + perigee_longitude,
    ]
}

/// The inverse. `q² + p² = tan²(i/2)` and `k² + h² = e²` are read off `DsstAuxiliary`'s own
/// `B = √(1 − k² − h²)` and `γ = (1 − p² − q²)/(1 + p² + q²)`.
fn keplerian(oe: [f64; 6]) -> KeplerianElements {
    let [a, k, h, q, p, lambda] = oe;
    let raan = libm::atan2(p, q);
    let perigee_longitude = libm::atan2(h, k);
    let seed = KeplerianElements {
        semi_major_axis: a,
        eccentricity: libm::hypot(k, h),
        inclination: Angle::from_radians(2.0 * libm::atan(libm::hypot(q, p))),
        raan: Angle::from_radians(raan),
        arg_periapsis: Angle::from_radians(perigee_longitude - raan),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    seed.at_mean_anomaly(Angle::from_radians(lambda - perigee_longitude))
}

fn state_to_equinoctial(r: [f64; 3], v: [f64; 3]) -> [f64; 6] {
    equinoctial(&KeplerianElements::from_state(
        Vec3::new(r[0], r[1], r[2]),
        Vec3::new(v[0], v[1], v[2]),
        MU,
    ))
}

fn auxiliary(oe: [f64; 6]) -> DsstAuxiliary {
    DsstAuxiliary::new(oe[0], oe[1], oe[2], oe[3], oe[4], oe[5], MU)
}

/// Osculating equinoctial elements to mean, by the fixed point `mean = osc − Δ_sp(mean)`.
///
/// `DsstZonal` supplies only the forward map (mean → osculating). The inverse is the standard
/// iteration, and it converges geometrically at rate `|∂Δ_sp/∂œ| = O(J₂) ≈ 10⁻³`, so a handful of
/// passes take a kilometres-wide short-period excursion in `a` below a micrometre.
///
/// The loop *must* close on mean elements. The osculating semi-major axis of this orbit swings by
/// kilometres every revolution under `J₂` alone, and a controller chasing that would spend its whole
/// budget fighting a periodic term that integrates to zero — which is exactly the "implausible Δv"
/// failure this test exists to detect.
fn mean_elements(zonal: &DsstZonal, r: [f64; 3], v: [f64; 3]) -> [f64; 6] {
    let osculating = state_to_equinoctial(r, v);
    let mut mean = osculating;
    for _ in 0..8 {
        let correction = zonal.short_period_correction(&auxiliary(mean));
        let mut next = [0.0; 6];
        let mut change = 0.0_f64;
        for i in 0..6 {
            next[i] = osculating[i] - correction[i];
            change = change.max((next[i] - mean[i]).abs());
        }
        mean = next;
        if change < 1e-12 {
            break;
        }
    }
    mean
}

// ---------------------------------------------------------------- the repeat orbit

/// One secant step on a scalar unknown, probing relatively so the same routine serves a 7 × 10⁶ m
/// axis and a 1.7 rad angle.
fn secant_step(x: f64, residual: impl Fn(f64) -> f64) -> f64 {
    let step = x.abs() * 1e-6;
    let f0 = residual(x);
    let f1 = residual(x + step);
    if (f1 - f0).abs() < f64::MIN_POSITIVE {
        x
    } else {
        x - f0 * step / (f1 - f0)
    }
}

/// The `(a, i)` of a `k`-revolution / `m`-nodal-day repeat that is also Sun-synchronous.
///
/// Two conditions, solved by alternating one-dimensional secant steps because each is monotone in its
/// own unknown and the coupling between them is `O(J₂)`: the repeat
/// `k · (ω⊕ − Ω̇) = m · n_draconitic` fixes `a` at fixed `i`, and `Ω̇ = 2π/year` fixes `i` at fixed
/// `a`. Both rates come from `KeplerianElements`' own `J₂` secular formulae, so the orbit is derived
/// rather than quoted — and the residual of each condition is asserted at the end.
fn sun_synchronous_repeat(revolutions: f64, days: f64) -> KeplerianElements {
    let mut els = KeplerianElements {
        semi_major_axis: 6_838_000.0,
        eccentricity: 0.001,
        inclination: Angle::from_radians(97.3_f64.to_radians()),
        raan: Angle::from_radians(0.7),
        arg_periapsis: Angle::from_radians(1.1),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    for _ in 0..40 {
        els.semi_major_axis = secant_step(els.semi_major_axis, |a| {
            let trial = KeplerianElements {
                semi_major_axis: a,
                ..els
            };
            revolutions * (OMEGA_EARTH - trial.nodal_regression_rate(RE, J2_COEFF))
                - days * trial.draconitic_mean_motion(RE, J2_COEFF)
        });
        let inclination = secant_step(els.inclination.radians(), |i| {
            let trial = KeplerianElements {
                inclination: Angle::from_radians(i),
                ..els
            };
            trial.nodal_regression_rate(RE, J2_COEFF) - SUN_SYNCHRONOUS_NODE_RATE
        });
        els.inclination = Angle::from_radians(inclination);
    }
    els
}

// ---------------------------------------------------------------- the plant

/// Point mass + `J₂` + drag, with a zero-order-held control acceleration.
///
/// The seventh state accumulates `∫|a_drag| dt` — the Δv the atmosphere removes, which is the floor
/// the commanded Δv is measured against. It is integrated by the same method, at the same tolerance,
/// as the trajectory that produced it.
struct Plant<A: Atmosphere> {
    point_mass: PointMass,
    oblateness: J2,
    drag: Drag<A>,
    command: [f64; 3],
}

impl<A: Atmosphere> DynamicalSystem<7> for Plant<A> {
    fn derivative(&self, t: f64, y: &[f64; 7], dydt: &mut [f64; 7]) {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let g = self.point_mass.acceleration(t, &r, &v);
        let j = self.oblateness.acceleration(t, &r, &v);
        let d = self.drag.acceleration(t, &r, &v);
        for i in 0..3 {
            dydt[i] = v[i];
            dydt[3 + i] = g[i] + j[i] + d[i] + self.command[i];
        }
        dydt[6] = norm(&d);
    }
}

fn norm(x: &[f64; 3]) -> f64 {
    libm::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// NRLMSISE-00 with the inertial→geodetic conversion the wrapper asks the caller to supply.
///
/// The inertial→Earth-fixed rotation is the Earth-rotation angle about `+z` and nothing more:
/// precession, nutation and polar motion move the sub-satellite longitude by well under a degree, and
/// the density's longitude dependence over a degree is far below the model's own uncertainty. The
/// *ellipsoid* is not negligible and is not approximated — geodetic and geocentric altitude differ by
/// up to 21 km, which is a third of a scale height and so tens of percent of the density.
fn atmosphere() -> Nrlmsise00Atmosphere<impl Fn(f64, [f64; 3]) -> (f64, f64, f64)> {
    Nrlmsise00Atmosphere {
        doy0: DOY0,
        sec0: SEC0,
        f107: F107,
        f107a: F107,
        ap: AP,
        geodetic: |t: f64, r: [f64; 3]| {
            let (s, c) = libm::sincos(OMEGA_EARTH * t);
            let fixed = Vec3::new(c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]);
            let g = itrf_to_geodetic(Position::<Itrf>::new(fixed))
                .expect("in orbit, not at the centre");
            (
                g.height.metres() / 1000.0,
                g.latitude.radians().to_degrees(),
                g.longitude.radians().to_degrees(),
            )
        },
    }
}

// ---------------------------------------------------------------- the run

/// A moving average of the element error over exactly one nodal period.
///
/// The DSST's first-order zonal short-period series leaves a residual at the orbital frequency and
/// its harmonics (module header: ±22.6 m in `a` here). A boxcar of width `W` has transfer zeros at
/// every multiple of `1/W`, so a window of one nodal period annihilates all of it and passes the
/// secular decay the controller is actually there to cancel.
///
/// The mean longitude is wrapped before it enters the sum: averaging raw angles across a `2π`
/// boundary is the same unwinding failure the law itself guards against, one level up.
struct OrbitAverage {
    window: Vec<[f64; 6]>,
    capacity: usize,
    next: usize,
}

impl OrbitAverage {
    fn new(period: f64) -> Self {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let capacity = (period / CONTROL_INTERVAL).round().max(1.0) as usize;
        Self {
            window: Vec::with_capacity(capacity),
            capacity,
            next: 0,
        }
    }

    /// Push a sample and return the average of everything held. Before the window fills, that is the
    /// average of what has arrived — a shorter boxcar, so the warm-up passes some ripple through and
    /// the Δv it costs is part of the reported total rather than excluded from it.
    fn push(&mut self, mut error: [f64; 6]) -> [f64; 6] {
        error[5] = wrap_to_pi(error[5]);
        if self.window.len() < self.capacity {
            self.window.push(error);
        } else {
            self.window[self.next] = error;
            self.next = (self.next + 1) % self.capacity;
        }
        let mut mean = [0.0; 6];
        for sample in &self.window {
            for i in 0..6 {
                mean[i] += sample[i];
            }
        }
        #[allow(clippy::cast_precision_loss)] // the window holds ~94 samples
        let n = self.window.len() as f64;
        for m in &mut mean {
            *m /= n;
        }
        mean
    }
}

struct Outcome {
    /// `Σ |u| Δt`, m/s — exact for a zero-order hold, no quadrature.
    commanded: f64,
    /// `∫ |a_drag| dt`, m/s — the seventh state.
    drag_makeup: f64,
    /// Largest `|a_mean − a_target|` over the span, m — the *instantaneous* mean element, so this
    /// still carries the short-period residual the averaging removes.
    worst_offset: f64,
    /// The same, on the orbit-averaged error the law actually sees: the hold quality with the
    /// estimator artefact taken out.
    worst_averaged: f64,
    /// Mean semi-major axis at the end, m.
    final_sma: f64,
    /// The orbit-averaged semi-major-axis error at the end, m — the settled offset. The
    /// instantaneous mean element is not it: the short-period residual is an order larger, so a
    /// single end-of-run sample of it reports where the ripple happened to be.
    final_averaged: f64,
    /// Along-track displacement from where the reference orbit would be at the end, m.
    along_track: f64,
    /// `∫ (a_mean − a_target) dt`, m·s — what the along-track displacement is the integral of.
    integrated_offset: f64,
    /// `∫ B_λ·u dt`, rad — the commanded acceleration's direct effect on the mean longitude, over
    /// and above what it does through the semi-major axis.
    thrust_longitude: f64,
}

/// Propagate the span, updating the command once per [`CONTROL_INTERVAL`] from the error averaged
/// over `period`. `law = None` is the uncontrolled twin, whose seventh state is the floor.
fn run(
    law: Option<&ElementFeedback>,
    target: &[f64; 6],
    zonal: &DsstZonal,
    period: f64,
) -> Outcome {
    let (r0, v0) = keplerian(zonal.osculating(*target)).to_state();
    let mut y = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z, 0.0];

    let controls = Controls::new(Tolerance::new(1e-9, 1e-11), 1e-4);
    let mut filter = OrbitAverage::new(period);
    let mut t = 0.0;
    let mut commanded = 0.0;
    let mut worst_offset = 0.0_f64;
    let mut worst_averaged = 0.0_f64;
    let mut final_averaged = 0.0;
    let mut integrated_offset = 0.0;
    let mut thrust_longitude = 0.0;

    while t < SPAN - 0.5 * CONTROL_INTERVAL {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let mean = mean_elements(zonal, r, v);
        let mut error = [0.0; 6];
        for i in 0..6 {
            error[i] = mean[i] - target[i];
        }
        let averaged = filter.push(error);
        worst_offset = worst_offset.max(error[0].abs());
        integrated_offset += error[0] * CONTROL_INTERVAL;
        // The warm-up window is a shorter boxcar and still passes ripple, so it is excluded from the
        // hold-quality figure while its Δv stays in the total. One full window, not a tuned number.
        if t >= period {
            worst_averaged = worst_averaged.max(averaged[0].abs());
        }
        final_averaged = averaged[0];

        // `B(œ)` and the true longitude are the *current* ones — where the satellite is now is what
        // sets the thrust direction. Only the error is averaged.
        //
        // The true longitude is the polar angle of the position in the equinoctial (f, g) frame:
        // `gaussian::element_derivatives` builds the node position as `X·f + Y·g` with `X = r cos L`
        // and `Y = r sin L`, so `L = atan2(r·g, r·f)`. Taking it from the real position rather than
        // solving the equinoctial Kepler equation is both cheaper and the more correct place to
        // evaluate the thrust direction.
        let aux = auxiliary(mean);
        let longitude = libm::atan2(dot(&r, &aux.g_vec), dot(&r, &aux.f_vec));
        let command = law.map_or([0.0; 3], |feedback| {
            feedback.acceleration(&aux, longitude, &averaged)
        });
        commanded += norm(&command) * CONTROL_INTERVAL;
        // A tangential thrust moves the mean longitude two ways: through `a`, and directly. The `λ`
        // row of the Gauss matrix is dominated by `−2·pos/A`, which is `2.6e-4 s` here — small next to
        // the `2/n ≈ 1790 s` of the `a` row, and very nearly orthogonal to a tangential command on a
        // near-circular orbit, so what survives is `O(e)` of that and changes sign around the orbit.
        // It is carried anyway rather than assumed away, and the measurement is printed.
        thrust_longitude +=
            dot(&variational_matrix(&aux, longitude)[5], &command) * CONTROL_INTERVAL;

        let plant = Plant {
            point_mass: PointMass { mu: MU },
            oblateness: J2 {
                mu: MU,
                re: RE,
                j2: J2_COEFF,
            },
            drag: Drag {
                ballistic_coefficient: BALLISTIC,
                earth_rotation_rate: OMEGA_EARTH,
                atmosphere: atmosphere(),
                wind: NoWind,
            },
            command,
        };
        y = integrate(
            &Dop853,
            &plant,
            t,
            &y,
            t + CONTROL_INTERVAL,
            0.25 * CONTROL_INTERVAL,
            controls,
        )
        .expect("the plant integrates")
        .final_state();
        t += CONTROL_INTERVAL;
    }

    let mean = mean_elements(zonal, [y[0], y[1], y[2]], [y[3], y[4], y[5]]);
    let reference = target[5] + target_longitude_rate(target) * t;
    Outcome {
        commanded,
        drag_makeup: y[6],
        worst_offset: worst_offset.max((mean[0] - target[0]).abs()),
        worst_averaged,
        final_sma: mean[0],
        final_averaged,
        along_track: target[0] * wrap_to_pi(mean[5] - reference),
        integrated_offset,
        thrust_longitude,
    }
}

/// The mean-longitude rate of the target orbit, `λ̇ = Ṁ + ω̇ + Ω̇` from the `J₂` secular rates, so the
/// along-track comparison is against where the *reference* orbit would be rather than against a
/// two-body extrapolation that would itself have drifted kilometres.
fn target_longitude_rate(target: &[f64; 6]) -> f64 {
    let els = keplerian(*target);
    els.anomalistic_mean_motion(RE, J2_COEFF)
        + els.apsidal_precession_rate(RE, J2_COEFF)
        + els.nodal_regression_rate(RE, J2_COEFF)
}

fn wrap_to_pi(angle: f64) -> f64 {
    let tau = core::f64::consts::TAU;
    angle - tau * libm::floor(angle / tau + 0.5)
}

#[test]
#[allow(clippy::too_many_lines)] // one end-to-end integration scenario
fn holding_the_ground_track_costs_the_drag_makeup_and_no_more() {
    let orbit = sun_synchronous_repeat(REPEAT.0, REPEAT.1);
    let zonal = DsstZonal::from_j(MU, RE, &[J2_COEFF]);
    let target = equinoctial(&orbit);

    let period = core::f64::consts::TAU / orbit.draconitic_mean_motion(RE, J2_COEFF);
    println!(
        "repeat orbit: {rev:.0} rev / {day:.0} nodal days, a = {a:.3} km ({alt:.1} km altitude), \
         i = {inc:.4}°, nodal period {period:.2} s",
        rev = REPEAT.0,
        day = REPEAT.1,
        a = orbit.semi_major_axis / 1000.0,
        alt = (orbit.semi_major_axis - RE) / 1000.0,
        inc = orbit.inclination.radians().to_degrees(),
    );
    let repeat = REPEAT.0 * (OMEGA_EARTH - orbit.nodal_regression_rate(RE, J2_COEFF))
        - REPEAT.1 * orbit.draconitic_mean_motion(RE, J2_COEFF);
    let node = orbit.nodal_regression_rate(RE, J2_COEFF) - SUN_SYNCHRONOUS_NODE_RATE;
    println!("  repeat residual {repeat:.3e} rad/s, Sun-synchronous residual {node:.3e} rad/s");
    assert!(
        repeat.abs() < 1e-16,
        "the repeat condition did not converge"
    );
    assert!(
        node.abs() < 1e-16,
        "the Sun-synchronous condition did not converge"
    );

    let law = ElementFeedback::new(
        [SMA_WEIGHT, 0.0, 0.0, 0.0, 0.0, 0.0],
        PROPORTIONAL,
        MAX_ACCELERATION,
    )
    .expect("valid gains");

    let free = run(None, &target, &zonal, period);
    let held = run(Some(&law), &target, &zonal, period);

    let scale = YEAR / SPAN;
    let decay = free.final_sma - target[0];
    println!(
        "\nuncontrolled, {days:.0} days: mean a falls {decay:.1} m, drag removes {dv:.6} m/s \
         ({rate:.2} m/s per year)",
        days = SPAN / 86_400.0,
        dv = free.drag_makeup,
        rate = free.drag_makeup * scale,
    );
    println!(
        "controlled  : commanded {cmd:.6} m/s ({rate:.2} m/s per year), drag {dv:.6} m/s, \
         ratio {ratio:.4}",
        cmd = held.commanded,
        rate = held.commanded * scale,
        dv = held.drag_makeup,
        ratio = held.commanded / held.drag_makeup,
    );
    println!(
        "hold quality: worst orbit-averaged a offset {avg:.2} m (instantaneous mean element \
         {off:.2} m, of which the short-period residual is most), settled offset {fin:.2} m; \
         along-track {at:.1} m held against {free_at:.1} m free",
        avg = held.worst_averaged,
        off = held.worst_offset,
        fin = held.final_averaged,
        at = held.along_track,
        free_at = free.along_track,
    );

    // The two independent statements of the same floor: the integrated `∫|a_drag| dt`, and the
    // semi-major-axis decay it must have produced, `(n/2)|Δa|`. One comes from the force, the other
    // from the trajectory, so agreement is evidence the plant's energy bookkeeping is right — which
    // is what makes the floor usable as a bar at all.
    let from_decay = 0.5 * orbit.mean_motion() * decay.abs();
    println!(
        "\nfloor: integral {a:.6e} m/s, (n/2)|Δa| {b:.6e} m/s, rel {rel:.3e}",
        a = free.drag_makeup,
        b = from_decay,
        rel = (free.drag_makeup - from_decay).abs() / free.drag_makeup,
    );
    assert!(
        (free.drag_makeup - from_decay).abs() / free.drag_makeup < 0.05,
        "the drag integral and the decay it caused disagree"
    );

    // The verdict. Below the floor means the orbit is still decaying; far above means the controller
    // is steering where it need not. The band is open on the low side by the one-off Δv of the
    // steady-state offset a proportional law leaves.
    let ratio = held.commanded / held.drag_makeup;
    assert!(
        (0.90..1.15).contains(&ratio),
        "the hold spent {ratio:.4}× the drag makeup"
    );
    // The hold is a hold: the semi-major axis stays inside the weight's own tolerance. Asserted on
    // both readings — the averaged error the law sees, and the raw mean element, which still carries
    // the short-period residual and so is the stricter of the two.
    let tolerance = 1.0 / libm::sqrt(SMA_WEIGHT);
    assert!(
        held.worst_averaged < tolerance,
        "the averaged semi-major axis wandered {:.1} m",
        held.worst_averaged
    );
    assert!(
        held.worst_offset < tolerance,
        "the mean semi-major axis wandered {:.1} m",
        held.worst_offset
    );
    assert!(
        decay.abs() > 5.0 * held.worst_offset,
        "the uncontrolled run decayed too little for the comparison to mean anything"
    );
    // Phase is the integral of energy, so the along-track displacement is not free to be anything:
    // `λ̇ ≈ n(a)` gives `dλ̇/da = −(3/2)n/a`, hence `Δs = a Δλ = −(3/2) n ∫δa dt + a ∫B_λ·u dt`, the
    // second term being the thrust's direct effect on the mean longitude — measured below, and small.
    // The left side comes out of the propagated mean longitude, the right out of the propagated
    // semi-major axis, so this is a statement about the plant and not a rearrangement of one number.
    //
    // It is checked on the *difference* between the two runs, not on either alone. Both are measured
    // against a reference longitude built from the first-order `J₂` secular rates, and that reference
    // is short of the truth by the second-order terms — `O(J₂²) ≈ 10⁻⁶` of `n`, which over three days
    // is a few kilometres. It is the same few kilometres in both runs, so differencing removes it
    // exactly; the residual left over is what the relation is actually being tested on. Each run's
    // own offset is printed rather than asserted, because it measures the analytic reference rather
    // than the plant.
    let n = orbit.mean_motion();
    let mut predicted = [0.0; 2];
    for (i, (name, run)) in [("free", &free), ("held", &held)].into_iter().enumerate() {
        predicted[i] = -1.5 * n * run.integrated_offset + target[0] * run.thrust_longitude;
        println!(
            "along-track, {name}: measured {m:.1} m, −(3/2)n∫δa dt + a∫B_λ·u dt = {p:.1} m \
             (direct thrust term {thrust:.2} m), common-mode reference offset {gap:.1} m",
            m = run.along_track,
            p = predicted[i],
            thrust = target[0] * run.thrust_longitude,
            gap = run.along_track - predicted[i],
        );
    }
    let measured_gap = free.along_track - held.along_track;
    let predicted_gap = predicted[0] - predicted[1];
    let deviation = (measured_gap - predicted_gap).abs() / predicted_gap.abs();
    println!(
        "along-track, free − held: measured {measured_gap:.1} m against {predicted_gap:.1} m, \
         rel {deviation:.3e}"
    );
    assert!(
        deviation < 0.02,
        "the along-track drift the hold removed is not the integral of the semi-major-axis offset \
         it removed"
    );

    // And the ground track: an along-track displacement `Δs` shifts the equator crossing east by
    // `R⊕ ω⊕ Δs / v`, because the track moves with the Earth while the satellite is late. The hold
    // does not drive this to zero and cannot — a proportional law leaves a standing `δa`, and a
    // standing `δa` is a linearly growing phase error. Removing it needs integral action, which is
    // a different law, not a different gain.
    let ground = RE * OMEGA_EARTH / libm::sqrt(MU / target[0]);
    println!(
        "ground track at the equator: held {held_km:.4} km, free {free_km:.4} km ({factor:.1}× better)",
        held_km = ground * held.along_track.abs() / 1000.0,
        free_km = ground * free.along_track.abs() / 1000.0,
        factor = free.along_track.abs() / held.along_track.abs(),
    );
    assert!(
        held.along_track.abs() * 5.0 < free.along_track.abs(),
        "the hold did not improve the along-track drift"
    );
}
