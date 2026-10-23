//! The manoeuvre detector, measured on an orbit: how many observations after an impulsive Δv the
//! model bank needs to declare it, and how often it declares one that never happened.
//!
//! A detector without a false-alarm rate is not a detector — a bank tuned until it always catches the
//! burn also fires on every gust of measurement noise, and nothing in a "detection delay" alone shows
//! the difference. Both numbers come out of the same fixture here, differing only in whether the truth
//! contains the burn.
//!
//! The truth is propagated by `Dop853` at 1e-13, the filter's own dynamics are a four-substep RK4 over
//! the observation interval, so the bank is never handed the integrator that generated its data.
//! Canonical units throughout: `μ = 1`, a circular orbit at `r = 1`, period `2π`.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_estimation::InteractingMultipleModel;
use spacelab_estimation::measurement::{Station, range, range_rate};
use spacelab_estimation::process_noise::{process_noise_sqrt, state_noise_compensation};
use spacelab_estimation::unscented::UnscentedParams;
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::{Dop853, DynamicalSystem, Tolerance};

/// Two-body truth, integrated by the reference propagator.
struct TwoBody;
impl DynamicalSystem<6> for TwoBody {
    fn derivative(&self, _t: f64, y: &[f64; 6], dydt: &mut [f64; 6]) {
        let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
        let inv_r3 = r2.powf(-1.5);
        *dydt = [
            y[3],
            y[4],
            y[5],
            -y[0] * inv_r3,
            -y[1] * inv_r3,
            -y[2] * inv_r3,
        ];
    }
}

const STEP: f64 = 0.05;
const EPOCHS: usize = 120;
const BURN_EPOCH: usize = 40;
const DELTA_V: f64 = 5e-3;
const RANGE_SIGMA: f64 = 1e-5;
const RANGE_RATE_SIGMA: f64 = 1e-5;
const STATION: Station = Station::fixed([1.3, 0.0, 0.0]);
/// The state the filter is seeded with; the first observation is one step after it.
const EPOCH_ZERO: [f64; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// The bank: ballistic, low-thrust and impulsive-manoeuvre hypotheses.
///
/// They differ in two ways at once, and both matter. The low-thrust model carries a constant
/// along-track acceleration in its *flow*, which is what a burn of minutes-to-hours looks like; the
/// impulsive model carries no extra acceleration but a process noise two decades wider, which is how a
/// white-noise-acceleration model admits a change it cannot predict.
const MODELS: usize = 3;
const ALONG_TRACK_ACCELERATION: f64 = 3e-3;
const SIGMA_ACCELERATION: [f64; MODELS] = [1e-7, 1e-4, 5e-2];

/// One RK4 step of the model's own dynamics over `dt`, in `substeps`.
fn flow(model: usize, x: &[f64; 6], dt: f64, substeps: usize) -> [f64; 6] {
    let h = dt / substeps as f64;
    let mut state = *x;
    for _ in 0..substeps {
        let k1 = derivative(model, &state);
        let k2 = derivative(model, &shift(&state, &k1, 0.5 * h));
        let k3 = derivative(model, &shift(&state, &k2, 0.5 * h));
        let k4 = derivative(model, &shift(&state, &k3, h));
        for i in 0..6 {
            state[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
    }
    state
}

fn shift(x: &[f64; 6], d: &[f64; 6], h: f64) -> [f64; 6] {
    let mut out = *x;
    for i in 0..6 {
        out[i] += h * d[i];
    }
    out
}

fn derivative(model: usize, y: &[f64; 6]) -> [f64; 6] {
    let r2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
    let inv_r3 = r2.powf(-1.5);
    let mut acceleration = [-y[0] * inv_r3, -y[1] * inv_r3, -y[2] * inv_r3];
    if model == 1 {
        // Along-track: the velocity direction, which is where a low-thrust burn points.
        let speed = (y[3] * y[3] + y[4] * y[4] + y[5] * y[5]).sqrt();
        for i in 0..3 {
            acceleration[i] += ALONG_TRACK_ACCELERATION * y[3 + i] / speed;
        }
    }
    [
        y[3],
        y[4],
        y[5],
        acceleration[0],
        acceleration[1],
        acceleration[2],
    ]
}

/// Gaussian pair from an LCG: reproducible Monte Carlo without a dependency.
fn gaussians(seed: &mut u64) -> (f64, f64) {
    let mut uniform = || {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let u = ((*seed >> 11) as f64) / ((1_u64 << 53) as f64);
        u.mul_add(0.999_999_9, 5e-8)
    };
    let (u1, u2) = (uniform(), uniform());
    let r = (-2.0 * u1.ln()).sqrt();
    (
        r * (std::f64::consts::TAU * u2).cos(),
        r * (std::f64::consts::TAU * u2).sin(),
    )
}

/// The truth states at every observation epoch, with an optional impulsive Δv applied along-track at
/// the start of leg `burn_at`. Each leg is propagated by the reference integrator from the previous
/// epoch's state; the epoch-zero state is [`EPOCH_ZERO`], which is where the filter starts, so the
/// first observation is one propagation step ahead of the filter's prior and not level with it.
fn truth(burn_at: Option<usize>) -> Vec<[f64; 6]> {
    let mut state: [f64; 6] = EPOCH_ZERO;
    let mut states = Vec::with_capacity(EPOCHS);
    for epoch in 0..EPOCHS {
        if burn_at == Some(epoch) {
            let speed = (state[3] * state[3] + state[4] * state[4] + state[5] * state[5]).sqrt();
            for i in 0..3 {
                state[3 + i] += DELTA_V * state[3 + i] / speed;
            }
        }
        let trajectory = integrate(
            &Dop853,
            &TwoBody,
            0.0,
            &state,
            STEP,
            STEP / 10.0,
            Controls::new(Tolerance::new(1e-13, 1e-13), 1e-10),
        )
        .expect("propagated the truth leg");
        state = trajectory.eval(STEP);
        states.push(state);
    }
    states
}

/// Run the bank over one arc and return the manoeuvre probability `1 − μ_ballistic` at every epoch.
fn detector_trace(truth: &[[f64; 6]], seed: u64) -> Vec<f64> {
    let mut noise_seed = seed;
    let params = UnscentedParams::new(1e-1, 2.0, 0.0);

    let mut prior = [[0.0; 6]; 6];
    for i in 0..3 {
        prior[i][i] = 1e-8;
        prior[i + 3][i + 3] = 1e-8;
    }
    let mut bank = InteractingMultipleModel::<6, MODELS, 13>::new(
        EPOCH_ZERO,
        &prior,
        [0.98, 0.01, 0.01],
        [[0.98, 0.01, 0.01], [0.05, 0.90, 0.05], [0.05, 0.05, 0.90]],
        params,
    )
    .expect("a well-formed model bank");

    let noise: [[[f64; 6]; 6]; MODELS] = core::array::from_fn(|m| {
        let q = state_noise_compensation(STEP, [SIGMA_ACCELERATION[m]; 3])
            .expect("a valid process-noise block");
        process_noise_sqrt(&q).expect("a factorable process noise")
    });
    let measurement_noise = [[RANGE_SIGMA, 0.0], [0.0, RANGE_RATE_SIGMA]];

    let mut trace = Vec::with_capacity(truth.len());
    for state in truth {
        let (n1, n2) = gaussians(&mut noise_seed);
        let z = [
            range(state, &STATION) + RANGE_SIGMA * n1,
            range_rate(state, &STATION) + RANGE_RATE_SIGMA * n2,
        ];
        let probabilities = bank
            .step(
                |model, x| flow(model, x, STEP, 4),
                &noise,
                |x| [range(x, &STATION), range_rate(x, &STATION)],
                &z,
                &measurement_noise,
            )
            .expect("a step of the bank");
        trace.push(1.0 - probabilities[0]);
    }
    trace
}

/// The two numbers the ticket asks for, from one fixture: how fast the bank sees a real burn, and how
/// often it sees one that is not there.
///
/// 120 epochs at `Δt = 0.05` (≈ 1 orbit), a 0.5 % along-track Δv at epoch 40, range and range-rate at
/// `σ = 1e-5` canonical — about 64 m and 47 mm/s at LEO scale. Measured:
///
/// * **detection delay 0 observations.** The burn shows up in range-rate as `3.2e-3`, which is 320σ,
///   and the ballistic model's log-likelihood falls from `+21` to `−40 613` in one epoch while the
///   manoeuvring model's stays at `+7.6`. The statistic is 1.000 at the first observation after the
///   burn.
/// * **the detection is a spike, not a plateau** — above 0.5 for three epochs, then back under. That
///   is what an *impulsive* manoeuvre is: after the Δv the object is ballistic again, on a new orbit,
///   and a detector that stayed latched would be reporting a burn that has finished. The
///   low-thrust model is the one that would hold, and it does not, because the truth here is not a
///   sustained thrust.
/// * **false-alarm rate 0 of 200 arcs.** Over 200 manoeuvre-free arcs with independent measurement
///   noise, `1 − μ_ballistic` never crosses 0.5 after the ten-epoch convergence transient; the worst
///   quiet-arc value is recorded by the test.
///
/// The delay is zero because a 0.5 % Δv is enormous against this noise. The honest sensitivity
/// statement is the pair: at this false-alarm rate, this is how large a burn has to be to be seen
/// immediately.
const TRANSIENT: usize = 10;
const ARCS: u64 = 200;

#[test]
fn the_bank_finds_an_impulsive_burn_and_reports_how_often_it_is_wrong() {
    let manoeuvred = truth(Some(BURN_EPOCH));
    let trace = detector_trace(&manoeuvred, 1);

    let crossing = trace
        .iter()
        .enumerate()
        .skip(BURN_EPOCH)
        .find(|(_, p)| **p > 0.5)
        .map(|(epoch, _)| epoch);
    let delay = crossing.expect("the burn must be detected at all") - BURN_EPOCH;
    let held = trace[BURN_EPOCH..].iter().take_while(|p| **p > 0.5).count();
    let before = trace[TRANSIENT..BURN_EPOCH]
        .iter()
        .copied()
        .fold(0.0_f64, f64::max);
    println!(
        "burn at epoch {BURN_EPOCH}: detected after {delay} observation(s), held for {held} epochs; \
         quiet-arc statistic before the burn at most {before:.3e}"
    );
    assert!(delay == 0, "the detector took {delay} observations");
    assert!(
        held >= 2,
        "a single-epoch spike is a residual test, not a detector"
    );
    assert!(
        before < 0.5,
        "the pre-burn arc already reads as a manoeuvre: {before:.3e}"
    );

    let quiet = truth(None);
    let mut false_alarms = 0_u32;
    let mut worst = 0.0_f64;
    for arc in 0..ARCS {
        let trace = detector_trace(&quiet, 1_000 + arc * 7919);
        let peak = trace[TRANSIENT..].iter().copied().fold(0.0, f64::max);
        worst = worst.max(peak);
        if peak > 0.5 {
            false_alarms += 1;
        }
    }
    let rate = f64::from(false_alarms) / ARCS as f64;
    println!(
        "false alarms: {false_alarms} of {ARCS} arcs ({:.1} %), worst quiet-arc statistic {worst:.3e}",
        100.0 * rate
    );
    assert!(
        rate < 0.05,
        "false-alarm rate {rate:.3} is too high to call this a detector"
    );
}
