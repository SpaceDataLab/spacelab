//! The delay a formation survives, measured against the published bound — and the two things that
//! bound does not say.
//!
//! # Leg 1: the bound, against an integrated DDE
//!
//! For a fixed connected undirected graph with uniform delay, [`Protocol::DelayedRelative`] reaches
//! consensus if and only if `τ < π/(2 λ_max(L))` (Olfati-Saber & Murray, *Consensus problems in
//! networks of agents with switching topology and time-delays*, IEEE TAC 49(9), 2004). A sharp
//! published result with an equality case is a better oracle than a tolerance, and it is checked the
//! way §4.13 checks the attitude margin: bisect the delay on the sign of a decay rate taken from an
//! integrated trajectory, over an ensemble of random connected graphs, and report the **worst**
//! relative gap rather than the mean — an ensemble average hides the one graph that disagrees.
//!
//! What is watched is the amplitude of the `λ_max` **mode**, not the aggregate disagreement norm.
//! That is not a refinement, it is the measurement: the crossing is a property of one mode, and the
//! norm is dominated by whichever mode is largest. Measured on the same ensemble, the aggregate
//! norm's slope at exactly `τ_max` is as far off as `−2.8e-4`, where the mode's is `−1.1e-14`; a
//! bisection on the norm converges **6.5 % above** the true crossing and reads as a refutation of
//! the published bound. It is not — it is the wrong observable.
//!
//! # Leg 2: the same links, the other protocol, no bound at all
//!
//! `derivations/consensus_delay.mac` eliminates the delay from [`Protocol::CurrentSelf`]'s
//! characteristic equation and finds no imaginary-axis crossing at any positive delay: an agent that
//! compares its neighbour's *old* state against its own *current* one is delay-independent. That is
//! a claim about every delay, so it is measured where the other protocol is long dead — at a hundred
//! times its bound, on the same graphs, with the losing protocol run alongside for contrast.
//!
//! # Leg 3: the bound governs convergence, not clearance
//!
//! A formation is not a scalar, and consensus is not collision avoidance. Running the law on the
//! offset error of a four-satellite square through a slot swap, and asking the *conjunction*
//! machinery — not a separation check invented here — what the closest approach is worth, the answer
//! turns out to be **almost independent of the delay**. Buying delay margin buys nothing here. That
//! is the useful direction of a negative result: the quantity the published bound governs is not the
//! quantity a flight-safety review asks about.

#![allow(
    clippy::expect_used,
    clippy::many_single_char_names,
    clippy::needless_range_loop
)]

use spacelab_control::delayed::{DelayedSystem, InitialValue, Past, integrate_delayed};
use spacelab_control::formation::{ConsensusNetwork, DelayBound, Graph, Protocol};
use spacelab_estimation::conjunction::ConjunctionGeometry;
use spacelab_estimation::sampling::Rng;
use spacelab_propagation::{Dop853, Tolerance, driver::Controls};

/// Agents in the ensemble graphs. Six is large enough for the spectrum to spread — `λ_max/λ₂` runs
/// past 9 on the sparser draws — and small enough that a bisection over 24 draws stays quick.
const AGENTS: usize = 6;
/// Ensemble size. Reported as a worst case over the draws, so this is a count of chances to fail.
const DRAWS: usize = 24;
/// Ensemble seed. Pinned, because a worst case over a random ensemble is itself a random variable
/// and a test that cannot be reproduced cannot be debugged.
const SEED: u64 = 0x_c0_5e_75_c0_5e_75;

/// Run length, in delays. Sixty is set by the measurement rather than by taste: the decay rate at
/// `0.99 τ_max` is `≈ −1.5e-3 /s` on these graphs, so the second half has to be long enough for that
/// to move the log-amplitude by something a difference can see. At forty the answer is biased.
const CYCLES: f64 = 60.0;

/// Delays over which `CurrentSelf` is given to settle at `BEYOND` times the other protocol's
/// bound. Its decay rate falls as `1/τ`, so a delay a hundred times larger needs a run a hundred
/// times longer before the ratio means anything.
const SETTLE: f64 = 500.0;

/// How far past its own bound `CurrentSelf` is pushed. A hundred times is not a margin, it is a
/// different regime: `DelayedRelative` at that delay has been unstable for two orders of magnitude.
const BEYOND: f64 = 100.0;

/// Integration tolerances. Tight, because the measurement is the *sign* of a decay rate near the
/// point where it vanishes, and a loose integration manufactures its own damping.
fn controls() -> Controls {
    Controls::new(Tolerance::new(1e-12, 1e-12), 1e-10)
}

/// A random connected graph on `AGENTS` nodes with gains drawn log-uniformly over a decade.
///
/// A spanning path first, then each remaining pair with probability one half: the path guarantees
/// connectivity by construction rather than by rejection sampling, which would bias the draw towards
/// dense graphs. The gains vary because a bound quoted on an unweighted graph proves nothing about a
/// weighted one, and these weights are control gains.
fn random_graph(rng: &mut Rng) -> Graph<AGENTS> {
    let mut a = [[0.0; AGENTS]; AGENTS];
    let gain = |rng: &mut Rng| 0.01 * libm::pow(10.0, rng.unit());
    for i in 1..AGENTS {
        let w = gain(rng);
        a[i - 1][i] = w;
        a[i][i - 1] = w;
    }
    for i in 0..AGENTS {
        for j in (i + 2)..AGENTS {
            if rng.unit() < 0.5 {
                let w = gain(rng);
                a[i][j] = w;
                a[j][i] = w;
            }
        }
    }
    Graph::new(a).expect("a path is connected, and adding edges keeps it so")
}

/// A starting disagreement with zero mean, so the consensus value is the origin.
fn scattered(rng: &mut Rng) -> [f64; AGENTS] {
    let mut x: [f64; AGENTS] = core::array::from_fn(|_| rng.standard_normal());
    #[allow(clippy::cast_precision_loss)]
    let mean = x.iter().sum::<f64>() / AGENTS as f64;
    for xi in &mut x {
        *xi -= mean;
    }
    x
}

fn project(mode: &[f64; AGENTS], state: &[f64; AGENTS]) -> f64 {
    mode.iter().zip(state).map(|(m, x)| m * x).sum()
}

/// The log-amplitude slope of the `λ_max` mode over the second half of a run: negative is consensus,
/// positive is not.
///
/// The amplitude is `hypot(y(t), y(t−τ))` rather than `|y(t)|`. At the crossing the mode is
/// oscillatory with `ωτ = π/2`, so `y(t−τ)` *is* the quadrature component and the pair traces a
/// circle — the same trick §4.13 uses on the attitude loop's phase plane, and it costs nothing here
/// because the delayed value is already in the solution. Taking `|y|` alone would put a `log` of
/// zero twice per cycle in the middle of a slope estimate.
///
/// The second half only, because the first is the transient the constant initial history injects,
/// and the slope of a transient is not the slope of a mode.
fn mode_decay_rate(
    graph: &Graph<AGENTS>,
    mode: &[f64; AGENTS],
    protocol: Protocol,
    delay: f64,
    start: [f64; AGENTS],
) -> f64 {
    let net = ConsensusNetwork::new(*graph, protocol, delay).expect("a positive delay");
    let span = CYCLES * delay;
    let solution = integrate_delayed(
        &Dop853,
        &net,
        InitialValue::at_rest(0.0, start),
        span,
        0.05 * delay,
        controls(),
    )
    .expect("the delayed consensus integrates");

    let amplitude = |t: f64| {
        let now = project(mode, &solution.eval(t));
        let then = project(mode, &solution.eval(t - delay));
        libm::log(libm::hypot(now, then))
    };
    let (a, b) = (0.5 * span, span);
    (amplitude(b) - amplitude(a)) / (b - a)
}

/// The critical delay, found by bisecting on the sign of [`mode_decay_rate`]. Shares no algebra with
/// `π/(2 λ_max)` — the answer comes out of a trajectory.
fn measured_critical_delay(
    graph: &Graph<AGENTS>,
    mode: &[f64; AGENTS],
    start: [f64; AGENTS],
    bound: f64,
) -> f64 {
    let rate = |tau: f64| mode_decay_rate(graph, mode, Protocol::DelayedRelative, tau, start);
    assert!(rate(0.9 * bound) < 0.0, "below the bound it must decay");
    assert!(rate(1.1 * bound) > 0.0, "above the bound it must grow");
    // Twenty halvings of a 20 % bracket is 2e-7 of it, below the gap being measured.
    let (mut lo, mut hi) = (0.9 * bound, 1.1 * bound);
    for _ in 0..20 {
        let mid = f64::midpoint(lo, hi);
        if rate(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// How far the disagreement falls over `cycles` delays, as `‖x(T)‖ / ‖x(0)‖`.
///
/// A *ratio* rather than a slope, because at a hundred times the bound the disagreement collapses
/// seven orders in the first tenth of the run and then wanders at `1e-5` while the fast modes'
/// cancellation unwinds — a slope taken across that reads positive without anything diverging. The
/// question leg 2 asks is whether consensus is reached, and a ratio answers it without needing the
/// asymptotic regime.
///
/// `cycles` is a parameter because the two protocols cannot be watched over the same window at this
/// delay: the one that survives needs hundreds of delays to settle, and the one that does not
/// overflows `f64` in about two hundred. Running the loser for the winner's span does not produce a
/// bigger number, it produces a step-size underflow.
fn reduction(
    graph: &Graph<AGENTS>,
    protocol: Protocol,
    delay: f64,
    start: [f64; AGENTS],
    cycles: f64,
) -> f64 {
    let net = ConsensusNetwork::new(*graph, protocol, delay).expect("a positive delay");
    let span = cycles * delay;
    let solution = integrate_delayed(
        &Dop853,
        &net,
        InitialValue::at_rest(0.0, start),
        span,
        0.05 * delay,
        controls(),
    )
    .expect("the delayed consensus integrates");
    ConsensusNetwork::disagreement(&solution.eval(span)) / ConsensusNetwork::disagreement(&start)
}

#[test]
fn the_critical_delay_is_the_published_bound_and_the_other_protocol_has_none() {
    let mut rng = Rng::new(SEED);
    let mut worst_gap = 0.0_f64;
    let mut worst_spread = 0.0_f64;
    let mut worst_reduction = 0.0_f64;
    let mut weakest_blowup = f64::INFINITY;

    for draw in 0..DRAWS {
        let graph = random_graph(&mut rng);
        let start = scattered(&mut rng);
        let (spectrum, modes) = graph.laplacian_eigen();
        let top = modes[AGENTS - 1];
        let net = ConsensusNetwork::new(graph, Protocol::DelayedRelative, 1.0).expect("positive");
        let DelayBound::Seconds(bound) = net.delay_bound() else {
            panic!("the delayed-relative protocol has a finite bound");
        };

        let measured = measured_critical_delay(&graph, &top, start, bound);
        let gap = (measured - bound).abs() / bound;
        worst_gap = worst_gap.max(gap);
        worst_spread = worst_spread.max(spectrum[AGENTS - 1] / spectrum[1]);

        // The same graph, the same links, the same start, a hundred times the delay one of them
        // survives.
        let survives = reduction(&graph, Protocol::CurrentSelf, BEYOND * bound, start, SETTLE);
        let dies = reduction(
            &graph,
            Protocol::DelayedRelative,
            BEYOND * bound,
            start,
            CYCLES,
        );
        worst_reduction = worst_reduction.max(survives);
        weakest_blowup = weakest_blowup.min(dies);

        println!(
            "draw {draw:2}: λ₂ {:.4e}, λ_max {:.4e} (spread {:5.2}), τ_max {bound:7.3} s, \
             measured {measured:7.3} s, rel {gap:.3e} | at {:7.1} s: CurrentSelf ×{survives:.2e} over {SETTLE:.0} delays, \
             DelayedRelative ×{dies:.2e} over {CYCLES:.0}",
            spectrum[1],
            spectrum[AGENTS - 1],
            spectrum[AGENTS - 1] / spectrum[1],
            BEYOND * bound,
        );

        assert!(
            survives < 1.0e-3,
            "draw {draw}: CurrentSelf did not converge at {BEYOND}x the DelayedRelative bound"
        );
        assert!(
            dies > 1.0e3,
            "draw {draw}: DelayedRelative should have diverged far past its own bound"
        );
    }

    println!(
        "\nensemble of {DRAWS} random connected graphs on {AGENTS} agents, \
         worst λ_max/λ₂ = {worst_spread:.2}\n  \
         worst |measured − π/(2λ_max)| / τ_max = {worst_gap:.3e}\n  \
         at {BEYOND}x that delay: CurrentSelf worst reduction ×{worst_reduction:.2e} over \
         {SETTLE:.0} delays, DelayedRelative weakest blow-up ×{weakest_blowup:.2e} over {CYCLES:.0}"
    );
    assert!(
        worst_gap < 2e-3,
        "the worst draw crosses {worst_gap:.3e} away from the published bound"
    );
}

// ---------------------------------------------------------------- leg 3: a real formation

/// Satellites in the square.
const SQUARE: usize = 4;
/// Half the side of the design square, metres, in the LVLH plane.
const HALF_SIDE: f64 = 50.0;
/// The out-of-plane bias between the two satellites that swap slots, metres. Without it the swap is
/// an exact collision and the miss distance is zero at every delay, which measures nothing.
const OUT_OF_PLANE: f64 = 10.0;
/// Equivalent-sphere radius of one satellite, metres — a 12U cubesat with its panels out.
const HARD_BODY: f64 = 0.5;
/// Per-satellite position uncertainty, metres, 1-σ isotropic: a relative-GNSS formation.
const SIGMA: f64 = 2.0;
/// Link gain in the ring, 1/s.
const RING_GAIN: f64 = 0.02;

/// The design shape: a 100 m square in the LVLH plane, about the formation's own centre.
fn design_offsets() -> [[f64; 3]; SQUARE] {
    [
        [HALF_SIDE, HALF_SIDE, 0.0],
        [-HALF_SIDE, HALF_SIDE, 0.0],
        [-HALF_SIDE, -HALF_SIDE, 0.0],
        [HALF_SIDE, -HALF_SIDE, 0.0],
    ]
}

/// One axis of a 3-D formation state, so the scalar consensus law applies to it unchanged.
struct Axis<'a, P> {
    past: &'a P,
    axis: usize,
}

impl<P: Past<{ 3 * SQUARE }>> Past<SQUARE> for Axis<'_, P> {
    fn at(&self, s: f64) -> [f64; SQUARE] {
        let full = self.past.at(s);
        core::array::from_fn(|i| full[3 * i + self.axis])
    }
}

/// The 3-D formation: the same scalar consensus, three times, because `L ⊗ I₃` has exactly the
/// spectrum of `L` with multiplicity three and the axes never mix. Measured, not assumed —
/// `the_three_dimensional_bound_is_the_scalar_one` bisects the 3-D critical delay.
struct Formation(ConsensusNetwork<SQUARE>);

impl DelayedSystem<{ 3 * SQUARE }> for Formation {
    fn delays(&self) -> &[f64] {
        self.0.delays()
    }

    fn derivative(
        &self,
        t: f64,
        y: &[f64; 3 * SQUARE],
        past: &impl Past<{ 3 * SQUARE }>,
        dydt: &mut [f64; 3 * SQUARE],
    ) {
        for axis in 0..3 {
            let plane: [f64; SQUARE] = core::array::from_fn(|i| y[3 * i + axis]);
            let mut rate = [0.0; SQUARE];
            self.0
                .derivative(t, &plane, &Axis { past, axis }, &mut rate);
            for (i, r) in rate.into_iter().enumerate() {
                dydt[3 * i + axis] = r;
            }
        }
    }
}

/// The ring the four satellites talk over: each to the two beside it in the square, which is the
/// topology a formation with directional cross-links actually has.
fn ring() -> Graph<SQUARE> {
    let mut a = [[0.0; SQUARE]; SQUARE];
    for i in 0..SQUARE {
        a[i][(i + 1) % SQUARE] = RING_GAIN;
        a[(i + 1) % SQUARE][i] = RING_GAIN;
    }
    Graph::new(a).expect("a connected ring")
}

/// The slot swap: satellites 0 and 2 start in each other's places, biased out of plane so the paths
/// miss rather than meet. Everything is in offset-error coordinates `zᵢ = xᵢ − δᵢ`.
fn swap_start() -> [f64; 3 * SQUARE] {
    let offsets = design_offsets();
    let mut start = [0.0; 3 * SQUARE];
    for axis in 0..3 {
        start[axis] = offsets[2][axis] - offsets[0][axis];
        start[6 + axis] = offsets[0][axis] - offsets[2][axis];
    }
    start[2] += OUT_OF_PLANE;
    start[8] -= OUT_OF_PLANE;
    start
}

/// The closest approach any pair makes while the formation reconfigures, and what the conjunction
/// machinery makes of it.
struct ClosestApproach {
    separation: f64,
    probability: f64,
    time: f64,
    pair: (usize, usize),
}

fn closest_approach(delay: f64) -> ClosestApproach {
    let net = ConsensusNetwork::new(ring(), Protocol::DelayedRelative, delay).expect("positive");
    let formation = Formation(net);
    let span = CYCLES * delay;
    let solution = integrate_delayed(
        &Dop853,
        &formation,
        InitialValue::at_rest(0.0, swap_start()),
        span,
        0.05 * delay,
        controls(),
    )
    .expect("the formation integrates");

    let offsets = design_offsets();
    let covariance = [
        [SIGMA * SIGMA, 0.0, 0.0],
        [0.0, SIGMA * SIGMA, 0.0],
        [0.0, 0.0, SIGMA * SIGMA],
    ];

    let mut worst = ClosestApproach {
        separation: f64::INFINITY,
        probability: 0.0,
        time: 0.0,
        pair: (0, 1),
    };
    let samples = 4_000_i32;
    for k in 0..=samples {
        let t = span * f64::from(k) / f64::from(samples);
        let z = solution.eval(t);
        let mut rate = [0.0; 3 * SQUARE];
        formation.derivative(t, &z, &solution, &mut rate);

        for i in 0..SQUARE {
            for j in (i + 1)..SQUARE {
                let position: [f64; 3] = core::array::from_fn(|axis| {
                    (z[3 * j + axis] + offsets[j][axis]) - (z[3 * i + axis] + offsets[i][axis])
                });
                let separation = libm::sqrt(position.iter().map(|c| c * c).sum::<f64>());
                if separation >= worst.separation {
                    continue;
                }
                let velocity: [f64; 3] =
                    core::array::from_fn(|axis| rate[3 * j + axis] - rate[3 * i + axis]);
                let probability = ConjunctionGeometry::new(
                    position, velocity, covariance, covariance, HARD_BODY, HARD_BODY,
                )
                .project()
                .map_or(0.0, |plane| plane.probability_max());
                worst = ClosestApproach {
                    separation,
                    probability,
                    time: t,
                    pair: (i, j),
                };
            }
        }
    }
    worst
}

/// `L ⊗ I₃` claims the 3-D formation has exactly the scalar bound. Bisected rather than asserted:
/// the delay at which the 3-D `λ_max` mode stops decaying, against `π/(2 λ_max(L))`.
///
/// The initial condition is the `λ_max` eigenvector, scaled differently on each axis, and not the
/// slot swap of the test below — because on a four-satellite ring the swap does not excite that mode
/// at all. `λ_max(C₄)` belongs to `(1, −1, 1, −1)/2`, and the swap's out-of-plane start
/// `(1, 0, −1, 0)` is orthogonal to it. Run from there, a delay *past* the bound produces no growth
/// for hundreds of delays, because the only thing growing starts at rounding. A bound test whose
/// initial condition misses the mode measures the initial condition.
#[test]
fn the_three_dimensional_bound_is_the_scalar_one() {
    let graph = ring();
    let (spectrum, modes) = graph.laplacian_eigen();
    let top = modes[SQUARE - 1];
    let bound = ConsensusNetwork::new(graph, Protocol::DelayedRelative, 1.0)
        .expect("positive")
        .delay_bound()
        .seconds();

    // Different amplitudes per axis, so a run that silently coupled them would not look symmetric.
    let start: [f64; 3 * SQUARE] = core::array::from_fn(|k| {
        #[allow(clippy::cast_precision_loss)]
        let scale = 1.0 + (k % 3) as f64;
        scale * top[k / 3]
    });

    // The worst of the three axes: the claim is that all three carry the same bound, so the one that
    // crosses first is the one that has to match.
    let grew = |delay: f64| {
        let formation = Formation(
            ConsensusNetwork::new(graph, Protocol::DelayedRelative, delay).expect("positive"),
        );
        let span = CYCLES * delay;
        let solution = integrate_delayed(
            &Dop853,
            &formation,
            InitialValue::at_rest(0.0, start),
            span,
            0.05 * delay,
            controls(),
        )
        .expect("the formation integrates");

        let amplitude = |t: f64, axis: usize| {
            let now: f64 = (0..SQUARE)
                .map(|i| top[i] * solution.eval(t)[3 * i + axis])
                .sum();
            let then: f64 = (0..SQUARE)
                .map(|i| top[i] * solution.eval(t - delay)[3 * i + axis])
                .sum();
            libm::log(libm::hypot(now, then))
        };
        (0..3)
            .map(|axis| (amplitude(span, axis) - amplitude(0.5 * span, axis)) / (0.5 * span))
            .fold(f64::NEG_INFINITY, f64::max)
            > 0.0
    };

    let (mut lo, mut hi) = (0.9 * bound, 1.1 * bound);
    assert!(!grew(lo), "below the bound the formation must converge");
    assert!(grew(hi), "above the bound it must not");
    for _ in 0..20 {
        let mid = f64::midpoint(lo, hi);
        if grew(mid) { hi = mid } else { lo = mid }
    }
    let measured = f64::midpoint(lo, hi);
    let gap = (measured - bound).abs() / bound;
    println!(
        "3-D ring of {SQUARE} at {RING_GAIN} /s: λ_max = {:.6e} /s, scalar τ_max = {bound:.6} s, \
         measured on the 12-state formation {measured:.6} s, rel {gap:.3e}",
        spectrum[SQUARE - 1],
    );
    assert!(
        gap < 2e-3,
        "the 3-D crossing sits {gap:.3e} from the scalar bound"
    );
}

#[test]
fn the_delay_bound_governs_convergence_and_not_clearance() {
    let bound = ConsensusNetwork::new(ring(), Protocol::DelayedRelative, 1.0)
        .expect("positive")
        .delay_bound()
        .seconds();

    println!(
        "ring of {SQUARE} at {RING_GAIN} /s, τ_max = {bound:.3} s. Satellites 0 and 2 swap slots \
         {:.0} m apart, biased {OUT_OF_PLANE:.0} m out of plane; hard body {:.0} m combined, \
         σ {SIGMA:.0} m each.\n",
        2.0 * libm::sqrt(2.0) * HALF_SIDE,
        2.0 * HARD_BODY,
    );

    let fractions = [0.10, 0.25, 0.50, 0.75, 0.90, 0.99];
    let mut approaches = Vec::new();
    for fraction in fractions {
        let worst = closest_approach(fraction * bound);
        println!(
            "τ = {:5.1} % of τ_max ({:6.3} s): closest approach {:7.3} m at t = {:7.2} s between \
             {} and {}, Pc = {:.4e}",
            100.0 * fraction,
            fraction * bound,
            worst.separation,
            worst.time,
            worst.pair.0,
            worst.pair.1,
            worst.probability,
        );
        approaches.push(worst);
    }

    let separations: Vec<f64> = approaches.iter().map(|a| a.separation).collect();
    let probabilities: Vec<f64> = approaches.iter().map(|a| a.probability).collect();
    let tightest = separations.iter().copied().fold(f64::INFINITY, f64::min);
    let loosest = separations.iter().copied().fold(0.0_f64, f64::max);
    let spread = (loosest - tightest) / tightest;
    let worst_pc = probabilities.iter().copied().fold(0.0_f64, f64::max);
    println!(
        "\nacross a factor of {:.0} in delay — 10 % to 99 % of the stability bound — the closest \
         approach moves {tightest:.3} m to {loosest:.3} m, a spread of {:.2} %, and Pc stays at \
         {worst_pc:.4e}.\nThe bound governs whether the formation converges. It does not govern \
         how close it passes on the way, and a consensus law has no collision avoidance to add.",
        fractions[fractions.len() - 1] / fractions[0],
        100.0 * spread,
    );

    // Every run converges — nothing here is outside the stability bound.
    for (fraction, worst) in fractions.iter().zip(&approaches) {
        assert!(
            worst.separation.is_finite() && worst.separation > 0.0,
            "τ = {fraction} τ_max produced a degenerate approach"
        );
    }
    // The claim is the *insensitivity*: a factor of ten in delay moves the miss distance by less
    // than one part in a hundred. If a future change makes the delay matter here, this fails and the
    // paragraph in docs/verification.md §4.15 has to be rewritten rather than quietly kept.
    assert!(
        spread < 0.01,
        "the closest approach varies by {:.2} % across the delay sweep, not the < 1 % measured",
        100.0 * spread
    );
    // And it is a real conjunction, not a comfortable one: a 10 m miss at σ = 2 m per satellite.
    assert!(
        worst_pc > 1e-6,
        "the swap should register as a conjunction worth reporting, not {worst_pc:.3e}"
    );
}
