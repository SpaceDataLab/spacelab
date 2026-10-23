//! How much of the backward pass a bounded lag actually buys, and where it stops buying anything.
//!
//! A fixed-lag smoother is a latency/accuracy dial, and the only honest way to set it is to measure
//! the curve. Two things have to be true before the number means anything:
//!
//! 1. **At a lag long enough to reach the end of the arc it must be the fixed-interval smoother**,
//!    to machine precision. Otherwise the window bookkeeping is wrong and every other number here is
//!    a measurement of a bug.
//! 2. **The smoothed covariance must be honest.** A smaller covariance is trivial to produce by
//!    getting the recursion wrong; a smaller covariance that a Monte-Carlo NEES agrees with is not.
//!
//! # The arc
//!
//! Six-state constant velocity, position observed on all three axes once per step, truth generated
//! by exactly the discrete model the filter assumes — the same `Φ` and the same `Q` from
//! [`state_noise_compensation`] — so the filter is correctly specified and the covariance
//! comparisons below are not measuring model error. Statistics are accumulated over the arc's
//! interior only, where the fixed-interval smoother has at least `MAX_LAG` epochs of future and is
//! therefore at its own saturated value rather than tapering into the arc's end.
//!
//! # Measured — 60 runs × 40 epochs, every lag scored on identical epochs
//!
//! Reported position sigma, the fraction of the fixed-interval gain it captures, and the NEES against
//! six degrees of freedom:
//!
//! ```text
//!                 lag      0      1      2      3      5     10     20     40    RTS
//!   q = 1e-4  σ (m)    4.486  4.330  4.183  4.043  3.787  3.253  2.528  1.858  1.705
//!             of RTS     0 %  5.6 % 10.9 % 15.9 % 25.2 % 44.3 % 70.4 % 94.5 %  100 %
//!             NEES      6.15   6.16   6.17   6.18   6.21   6.22   6.23   6.25   6.13
//!   q = 3e-1  σ (m)   14.976 10.528 10.380 10.295 10.158 10.154 10.154 10.154 10.154
//!             of RTS     0 % 92.2 % 95.3 % 97.1 % 99.9 %  100 %  100 %  100 %  100 %
//!             NEES      5.93   5.94   5.93   5.93   5.95   5.95   5.95   5.95   5.95
//! ```
//!
//! **The saturation lag is a property of the arc, not of the smoother, and the two regimes differ by
//! more than an order of magnitude.** On the noisy arc a lag of one captures 92 %; on the
//! well-modelled one a lag of five captures 25 % and it takes forty to reach 95 %. The consequence
//! for precise orbit determination is in `src/fixed_lag.rs`.
//!
//! # Why the assertions are on σ and not on the RMS
//!
//! In a linear-Gaussian model the covariance does not depend on the data, so "more future is never
//! worse" is an *arithmetic* statement about σ and only a statistical one about the sampled RMS. At
//! `q = 3e-1` the lag-5 and fixed-interval estimators are numerically identical to five figures, and
//! their sampled RMS crosses by parts in a million. Asserting the ordering on the RMS would be
//! asserting that Monte-Carlo noise has a sign.

#![allow(
    clippy::unwrap_used,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_estimation::{
    FixedLagSmoother, RtsSmoother, SmoothingStep, SquareRootKalman, normalized_squared,
    process_noise_sqrt, state_noise_compensation,
};

const STEP: f64 = 10.0;
const STEPS: usize = 120;
const RUNS: usize = 60;
const OBSERVATION_SIGMA: f64 = 10.0;
const POSITION_SIGMA_0: f64 = 100.0;
const VELOCITY_SIGMA_0: f64 = 1.0;
/// Every lag is scored over exactly the same epochs, `[MAX_LAG, STEPS − MAX_LAG)`, and this is the
/// longest lag in the sweep.
///
/// Both halves matter. The upper bound gives the fixed-interval smoother at least `MAX_LAG` epochs
/// of future everywhere it is scored, so it is at its own saturated value rather than tapering into
/// the arc's end. The lower bound is what makes the *lags* comparable: a lag-`L` smoother cannot
/// report an epoch before `L`, so scoring each lag over "whatever it can reach" would compare
/// different sub-arcs and read as a real difference. It did — lag 40 came out 1.4 % better than the
/// fixed-interval limit, which is impossible, because it was being scored on 60 epochs where every
/// other lag saw 80.
const MAX_LAG: usize = 40;
const SEED: u64 = 0x0F1D_1A60;
/// A well-modelled orbit: the unmodelled acceleration is small against the measurement noise, so the
/// filter's memory is long and so is the lag a smoother needs.
const QUIET: f64 = 1e-4;
/// A poorly modelled or manoeuvring target: per-step process noise comparable to the measurement
/// noise, so the filter forgets fast and the smoother saturates quickly.
const NOISY: f64 = 3e-1;

/// `SplitMix64` with a Box–Muller transform: a deterministic Gaussian source with no dependency.
struct Gaussian {
    state: u64,
    spare: Option<f64>,
}

impl Gaussian {
    const fn seeded(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn uniform(&mut self) -> f64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) * (1.0 / 9_007_199_254_740_992.0)
    }

    fn normal(&mut self) -> f64 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let radius = libm::sqrt(-2.0 * libm::log(u1));
        let angle = core::f64::consts::TAU * u2;
        self.spare = Some(radius * libm::sin(angle));
        radius * libm::cos(angle)
    }
}

/// One forward pass, recorded: everything a backward pass of any kind needs.
struct Arc {
    truth: Vec<[f64; 6]>,
    filtered: Vec<([f64; 6], [[f64; 6]; 6])>,
    /// `intervals[k]` carries `k → k+1`, so there are `STEPS − 1` of them.
    intervals: Vec<SmoothingStep<6>>,
}

fn transition() -> [[f64; 6]; 6] {
    let mut stm = [[0.0; 6]; 6];
    for i in 0..6 {
        stm[i][i] = 1.0;
    }
    for axis in 0..3 {
        stm[axis][axis + 3] = STEP;
    }
    stm
}

fn generate(source: &mut Gaussian, density: f64) -> Arc {
    let stm = transition();
    let q = state_noise_compensation(STEP, [density; 3]).unwrap();
    let sq = process_noise_sqrt(&q).unwrap();

    let mut initial = [[0.0; 6]; 6];
    for axis in 0..3 {
        initial[axis][axis] = POSITION_SIGMA_0;
        initial[axis + 3][axis + 3] = VELOCITY_SIGMA_0;
    }

    let mut truth = [0.0, 0.0, 0.0, 7_500.0, -1_200.0, 300.0];
    let mut x0 = truth;
    for axis in 0..3 {
        x0[axis] += POSITION_SIGMA_0 * source.normal();
        x0[axis + 3] += VELOCITY_SIGMA_0 * source.normal();
    }
    let mut filter = SquareRootKalman::new(x0, initial);

    let mut arc = Arc {
        truth: Vec::with_capacity(STEPS),
        filtered: Vec::with_capacity(STEPS),
        intervals: Vec::with_capacity(STEPS - 1),
    };

    for epoch in 0..STEPS {
        for axis in 0..3 {
            let z = truth[axis] + OBSERVATION_SIGMA * source.normal();
            let mut h = [0.0; 6];
            h[axis] = 1.0;
            let innovation = z - filter.x[axis];
            filter.measurement_update(&h, innovation, OBSERVATION_SIGMA * OBSERVATION_SIGMA);
        }
        arc.truth.push(truth);
        arc.filtered.push((filter.x, filter.covariance_sqrt()));

        if epoch + 1 == STEPS {
            break;
        }

        // Advance truth and filter to the next epoch, recording what the backward pass will need.
        let draw: [f64; 6] = core::array::from_fn(|_| source.normal());
        let mut next = [0.0; 6];
        for i in 0..6 {
            for k in 0..6 {
                next[i] += stm[i][k] * truth[k];
            }
            for k in 0..=i {
                next[i] += sq[i][k] * draw[k];
            }
        }
        truth = next;

        let predicted: [f64; 6] =
            core::array::from_fn(|i| (0..6).map(|k| stm[i][k] * filter.x[k]).sum());
        arc.intervals.push(SmoothingStep {
            filtered: filter.x,
            filtered_sqrt: filter.covariance_sqrt(),
            stm,
            process_noise_sqrt: sq,
            predicted,
        });
        filter.time_update(&stm, &sq);
        filter.x = predicted;
    }

    arc
}

#[derive(Clone, Copy, Debug, Default)]
struct Stats {
    squared_error: f64,
    nees: f64,
    position_variance: f64,
    samples: usize,
}

impl Stats {
    fn record(&mut self, estimate: &[f64; 6], sqrt: &[[f64; 6]; 6], truth: &[f64; 6]) {
        let error: [f64; 6] = core::array::from_fn(|i| estimate[i] - truth[i]);
        self.squared_error += error[0] * error[0] + error[1] * error[1] + error[2] * error[2];
        self.nees += normalized_squared(sqrt, &error);
        // trace of the position block of `P = S Sᵀ`. Unlike the RMS this does not depend on the
        // realisation at all, so it is the invariant worth asserting on.
        for i in 0..3 {
            self.position_variance += sqrt[i].iter().map(|v| v * v).sum::<f64>();
        }
        self.samples += 1;
    }

    fn rms(&self) -> f64 {
        libm::sqrt(self.squared_error / self.samples as f64)
    }

    fn mean_nees(&self) -> f64 {
        self.nees / self.samples as f64
    }

    fn sigma(&self) -> f64 {
        libm::sqrt(self.position_variance / self.samples as f64)
    }
}

/// Run a fixed-lag smoother of this lag across every arc, scoring the interior epochs.
fn sweep<const LAG: usize>(arcs: &[Arc]) -> Stats {
    let mut stats = Stats::default();
    for arc in arcs {
        let mut smoother = FixedLagSmoother::<6, LAG>::new();
        for epoch in 0..STEPS {
            let (x, s) = arc.filtered[epoch];
            // The estimate this reports is for `epoch − depth`, which is only in the scored interior
            // once the ring has filled and the arc has run far enough in.
            if smoother.depth() == LAG {
                let reported = epoch - LAG;
                if (MAX_LAG..STEPS - MAX_LAG).contains(&reported) {
                    let (estimate, sqrt) = smoother.smooth(&x, &s).unwrap();
                    stats.record(&estimate, &sqrt, &arc.truth[reported]);
                }
            }
            if epoch + 1 < STEPS {
                smoother.push(arc.intervals[epoch]);
            }
        }
    }
    stats
}

/// The fixed-interval limit: a full backward pass, scored on the same interior epochs.
fn fixed_interval(arcs: &[Arc]) -> Stats {
    let mut stats = Stats::default();
    for arc in arcs {
        let last = STEPS - 1;
        let (x, s) = arc.filtered[last];
        let mut smoother = RtsSmoother::new(x, s);
        for epoch in (0..last).rev() {
            smoother.step(&arc.intervals[epoch]).unwrap();
            if (MAX_LAG..STEPS - MAX_LAG).contains(&epoch) {
                stats.record(
                    smoother.state(),
                    &smoother.covariance_sqrt(),
                    &arc.truth[epoch],
                );
            }
        }
    }
    stats
}

const LAGS: [usize; 7] = [1, 2, 3, 5, 10, 20, 40];

/// Everything one process-noise level has to say.
struct Curve {
    filter: Stats,
    lags: [Stats; 7],
    limit: Stats,
}

impl Curve {
    /// The fraction of the fixed-interval gain a lag captures, in reported position sigma — the
    /// deterministic measure, not the sampled one.
    fn captured(&self, index: usize) -> f64 {
        (self.filter.sigma() - self.lags[index].sigma())
            / (self.filter.sigma() - self.limit.sigma())
    }
}

fn curve(density: f64) -> Curve {
    let mut source = Gaussian::seeded(SEED ^ density.to_bits());
    let arcs: Vec<Arc> = (0..RUNS).map(|_| generate(&mut source, density)).collect();
    let result = Curve {
        filter: sweep::<0>(&arcs),
        lags: [
            sweep::<1>(&arcs),
            sweep::<2>(&arcs),
            sweep::<3>(&arcs),
            sweep::<5>(&arcs),
            sweep::<10>(&arcs),
            sweep::<20>(&arcs),
            sweep::<40>(&arcs),
        ],
        limit: fixed_interval(&arcs),
    };

    println!("  q = {density:.0e}");
    print!("    σ (m)  {:7.3}", result.filter.sigma());
    for stats in &result.lags {
        print!("{:7.3}", stats.sigma());
    }
    println!("{:7.3}", result.limit.sigma());
    print!("    of RTS   0.0 %");
    for index in 0..LAGS.len() {
        print!("{:6.1} %", 100.0 * result.captured(index));
    }
    println!(" 100.0 %");
    print!("    RMS    {:7.3}", result.filter.rms());
    for stats in &result.lags {
        print!("{:7.3}", stats.rms());
    }
    println!("{:7.3}", result.limit.rms());
    print!("    NEES   {:7.2}", result.filter.mean_nees());
    for stats in &result.lags {
        print!("{:7.2}", stats.mean_nees());
    }
    println!("{:7.2}", result.limit.mean_nees());
    result
}

/// The latency/accuracy curve, at two process-noise levels, because the saturation lag is a property
/// of the arc rather than of the smoother.
#[test]
fn the_lag_a_smoother_needs_is_set_by_the_process_noise() {
    println!("               lag       0      1      2      3      5     10     20     40    RTS");
    let quiet = curve(QUIET);
    let noisy = curve(NOISY);

    for (density, result) in [(QUIET, &quiet), (NOISY, &noisy)] {
        // 1. More future is never worse, and the fixed-interval pass is the floor. This is a
        //    statement about the covariance, which in a linear-Gaussian model does not depend on the
        //    data at all — so the tolerance is arithmetic, not statistical.
        let mut previous = result.filter.sigma();
        for (lag, stats) in LAGS.iter().zip(&result.lags) {
            assert!(
                stats.sigma() <= previous * (1.0 + 1e-12),
                "q = {density:e} lag {lag}: σ {} against the shorter lag's {previous}",
                stats.sigma()
            );
            assert!(
                stats.sigma() >= result.limit.sigma() * (1.0 - 1e-12),
                "q = {density:e} lag {lag}: σ {} beat the fixed-interval {}",
                stats.sigma(),
                result.limit.sigma()
            );
            previous = stats.sigma();
        }
        assert!(
            result.filter.sigma() > result.limit.sigma(),
            "q = {density:e}: smoothing bought nothing"
        );

        // 2. The covariance shrinks *honestly*: a recursion that lost the subtraction would show a
        //    smaller σ and an inflated NEES. Checked at every lag, not just at the ends.
        for (lag, stats) in core::iter::once((&0, &result.filter))
            .chain(LAGS.iter().zip(&result.lags))
            .chain(core::iter::once((&usize::MAX, &result.limit)))
        {
            assert!(
                (5.0..7.2).contains(&stats.mean_nees()),
                "q = {density:e} lag {lag}: NEES {}",
                stats.mean_nees()
            );
        }

        // 3. And the error follows the covariance down. RMS is a sample statistic, so this is a
        //    statistical claim and gets a statistical tolerance — at a lag where the two estimators
        //    are numerically almost the same, the sampled RMS can and does cross by a few parts in
        //    a million.
        assert!(
            result.lags[6].rms() < result.filter.rms(),
            "q = {density:e}: lag 40 RMS {} against the filter's {}",
            result.lags[6].rms(),
            result.filter.rms()
        );
        assert!(
            result.limit.rms() <= result.lags[6].rms() * 1.01,
            "q = {density:e}: fixed-interval RMS {} against lag 40's {}",
            result.limit.rms(),
            result.lags[6].rms()
        );
    }

    // 4. The finding. On a well-modelled arc a lag of 5 captures 33 % of what the whole arc would
    //    give and it takes a lag of 40 to reach 97 %; on an arc whose process noise is comparable to
    //    its measurement noise, a lag of 1 already captures 93 % and a lag of 5 is at the limit.
    assert!(
        noisy.captured(0) > 0.9,
        "one step should nearly saturate the noisy arc: {:.3}",
        noisy.captured(0)
    );
    assert!(
        noisy.captured(3) > 0.99,
        "and five steps should reach it: {:.3}",
        noisy.captured(3)
    );
    assert!(
        quiet.captured(3) < 0.5,
        "the well-modelled arc should be nowhere near saturated at lag 5: {:.3}",
        quiet.captured(3)
    );
    assert!(
        quiet.captured(6) > 0.9,
        "…but should get there by lag 40: {:.3}",
        quiet.captured(6)
    );
}

/// A lag long enough to reach the end of the arc *is* the fixed-interval smoother, bit for bit.
///
/// This is the test that makes the sweep above mean something: it pins the ring's ordering and its
/// off-by-ones against an independent implementation of the same recursion, at an epoch where the
/// two condition on identical data. Nothing about the window is approximated, so the tolerance is
/// floating-point associativity and not a physical one.
#[test]
fn a_lag_that_reaches_the_arcs_end_is_the_fixed_interval_smoother() {
    let mut source = Gaussian::seeded(SEED ^ 0xABCD);
    let arc = generate(&mut source, QUIET);

    let mut smoother = FixedLagSmoother::<6, MAX_LAG>::new();
    for epoch in 0..STEPS - 1 {
        smoother.push(arc.intervals[epoch]);
    }
    let (x, s) = arc.filtered[STEPS - 1];
    let (lagged, lagged_sqrt) = smoother.smooth(&x, &s).unwrap();

    // The epoch a lag of MAX_LAG reports when the filter is at the last one: from there it sees
    // observations through the end of the arc, which is what the fixed-interval pass sees.
    let reported = STEPS - 1 - MAX_LAG;
    let mut reference = RtsSmoother::new(x, s);
    for epoch in (reported..STEPS - 1).rev() {
        reference.step(&arc.intervals[epoch]).unwrap();
    }

    for i in 0..6 {
        assert!(
            (lagged[i] - reference.state()[i]).abs() < 1e-9,
            "component {i}: {} against {}",
            lagged[i],
            reference.state()[i]
        );
        for j in 0..6 {
            assert!(
                (lagged_sqrt[i][j] - reference.covariance_sqrt()[i][j]).abs() < 1e-9,
                "S[{i}][{j}]: {} against {}",
                lagged_sqrt[i][j],
                reference.covariance_sqrt()[i][j]
            );
        }
    }
}
