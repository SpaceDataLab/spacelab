//! Gates of `long_run` (plan 18 Ф31.F): the integrals and the finite sums against the rows
//! `derivations/long_run_coverage.gp` computes at 60 digits on the doubles passed here (G2), a
//! polar orbit over an equatorial point in closed form (G3), the commensurate answer against its
//! bounds and its longitude average, and simulations along the trajectory within their sampling
//! error (S1). The `.gp` checks that every row below is what it prints.

#![expect(
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::type_complexity,
    reason = "the rows are copied verbatim from derivations/long_run_coverage.gp, which checks them \
              against its output; some are the doubles of 30, 45, 60 and 90 degrees"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "the pass, sample and batch counts converted are far below 2^52"
)]

use core::f64::consts::{FRAC_PI_2, PI, TAU};
use spacelab_constellation::{
    LongRun, LongRunError, LongRunGeometry, Rates, RepeatTrack, long_run_coverage,
};

fn geometry(inclination: f64, latitude: f64, footprint: f64) -> LongRunGeometry {
    LongRunGeometry {
        inclination,
        latitude,
        footprint,
    }
}

fn incommensurate(g: &LongRunGeometry, kappa: f64) -> (f64, f64) {
    match long_run_coverage(g, Rates::Incommensurate { kappa }) {
        Ok(LongRun::Incommensurate(s)) => (s.coverage, s.passes_per_revolution),
        other => panic!("{g:?} at kappa {kappa}: {other:?}"),
    }
}

fn commensurate(g: &LongRunGeometry, np: u32, nd: u32, ground_offset: f64) -> (u64, f64, f64) {
    let track = RepeatTrack::new(np, nd).unwrap_or_else(|e| panic!("{np}/{nd}: {e}"));
    match long_run_coverage(
        g,
        Rates::Commensurate {
            track,
            ground_offset,
        },
    ) {
        Ok(LongRun::Commensurate {
            passes_per_cycle,
            stats,
        }) => (
            passes_per_cycle,
            stats.coverage,
            stats.passes_per_revolution,
        ),
        other => panic!("{g:?} on {np}/{nd} at {ground_offset}: {other:?}"),
    }
}

/// (i, φ, λ, κ, coverage, passes per revolution): the M0 cases of plan 18 first, then a low orbit
/// whose band holds the ground point always, one with a band of `Δ = π`, a retrograde one, and two
/// hemisphere footprints.
#[rustfmt::skip]
const INCOMMENSURATE_ROWS: &[(f64, f64, f64, f64, f64, f64)] = &[
    (1.7069320084504542e0, 6.108652381980153e-1, 1.3962634015954636e-1, 6.60234717e-2, 3.8494980674265832e-3, 1.1139533211048018e-1),
    (9.005898940290741e-1, 6.108652381980153e-1, 3.490658503988659e-1, 6.47731e-2, 4.360280506673029e-2, 4.712828536298516e-1),
    (9.005898940290741e-1, 1.0471975511965979e0, 3.490658503988659e-1, 6.47731e-2, 2.7679427295461342e-2, 3.235619647204305e-1),
    (1.710422666954443e0, 1.4835298641951802e0, 2.0943951023931956e-1, 6.71e-2, 4.000920565527691e-2, 7.976805567388363e-1),
    (5.235987755982989e-1, 1.7453292519943295e-1, 2.6179938779914946e-1, 6.34199e-2, 2.460911645178593e-2, 3.6141516930273915e-1),
    (8.726646259971647e-2, 5.235987755982989e-2, 1.7453292519943295e-1, 7.02e-2, 4.8661115388309084e-2, 9.298e-1),
    (1.4835298641951802e0, 1.3962634015954636e0, 3.490658503988659e-1, 6.55e-2, 9.97226230461554e-2, 1e0),
    (2.443460952792061e0, -5.235987755982989e-1, 4.363323129985824e-1, 7.01e-2, 6.424570277341314e-2, 5.510652774206037e-1),
    (9.005898940290741e-1, 6.108652381980153e-1, 1.5707963267948966e0, 6.47731e-2, 4.9999999999999994e-1, 9.352269e-1),
    (1.7069320084504542e0, -1.2217304763960306e0, 1.5707963267948966e0, 6.60234717e-2, 5e-1, 1e0),
];

/// (λ, κ, coverage `I(λ)/π²` by the AGM, passes per revolution by the closed form): a polar orbit
/// over an equatorial point.
#[rustfmt::skip]
const POLAR_ROWS: &[(f64, f64, f64, f64)] = &[
    (8.726646259971647e-2, 6.60234717e-2, 1.212419355462978e-3, 5.567681626503857e-2),
    (3.490658503988659e-1, 6.47731e-2, 1.9493368287039584e-2, 2.2270767893897708e-1),
    (7.853981633974483e-1, 6.71e-2, 1.0104692052104725e-1, 5.014288952350844e-1),
    (1.0471975511965979e0, 7.02e-2, 1.8478152943239975e-1, 6.693639758440031e-1),
];

/// (Np, Nd, i, φ, λ, ground offset, passes per cycle, coverage), by root search along the
/// trajectory over one cycle.
#[rustfmt::skip]
const COMMENSURATE_ROWS: &[(u32, u32, f64, f64, f64, f64, u64, f64)] = &[
    (233, 16, 1.7139133254584316e0, 6.108652381980153e-1, 1.3962634015954636e-1, 2.6966460545835135e-3, 27, 3.832037266428593e-3),
    (233, 16, 1.7139133254584316e0, 6.108652381980153e-1, 1.3962634015954636e-1, 1.6719205538417783e-2, 26, 3.87011199994686e-3),
    (43, 3, 1.1344640137963142e0, 3.490658503988659e-1, 2.6179938779914946e-1, 5.406461775945225e-2, 8, 1.3246757491650296e-2),
    (15, 1, 9.005898940290741e-1, 1.0471975511965979e0, 3.490658503988659e-1, 2.0943951023931956e-1, 5, 2.8823316354851032e-2),
    (143, 10, 1.7212437083168077e0, 1.4835298641951802e0, 2.0943951023931956e-1, 2.196917939573282e-3, 106, 3.682969861573778e-2),
    (29, 2, 5.235987755982989e-1, 1.7453292519943295e-1, 2.6179938779914946e-1, 1.9499540608488372e-1, 10, 2.453758614731982e-2),
    (14, 1, 1.7069320084504542e0, 0.0, 5.235987755982989e-2, 1.121997376282069e-1, 0, 0.0),
];

#[test]
fn the_integrals_are_the_rows_of_the_pari_file() {
    let (mut worst_p, mut worst_nu) = (0.0_f64, 0.0_f64);
    for &(i, phi, lambda, kappa, p, nu) in INCOMMENSURATE_ROWS {
        let (got_p, got_nu) = incommensurate(&geometry(i, phi, lambda), kappa);
        worst_p = worst_p.max((got_p - p).abs() / p);
        worst_nu = worst_nu.max((got_nu - nu).abs() / nu);
        assert!(
            (got_p - p).abs() <= 1e-14 * p && (got_nu - nu).abs() <= 1e-14 * nu,
            "i {i} phi {phi} lambda {lambda} kappa {kappa}: P {got_p} vs {p}, nu {got_nu} vs {nu}"
        );
    }
    println!("largest relative error: P {worst_p:.1e}, nu {worst_nu:.1e}");
}

/// G3: `P = I(λ)/π²`, `I(λ) = 2∫₀^λ sin μ K(sin μ) dμ`, from the `.gp`; `ν` by its closed form,
/// evaluated here, `ν = (2/π)(κu* + acos(cos λ / cos u*))` with `cos² u*` the positive root of
/// `κ²y² + cos²λ (1 − κ²) y − cos²λ = 0`.
#[test]
fn a_polar_orbit_over_an_equatorial_point_is_the_closed_form() {
    for &(lambda, kappa, p, nu) in POLAR_ROWS {
        let c2 = libm::cos(lambda).powi(2);
        let k2 = kappa * kappa;
        let y = (-c2 * (1.0 - k2) + libm::sqrt(c2 * c2 * (1.0 - k2).powi(2) + 4.0 * k2 * c2))
            / (2.0 * k2);
        let turn = libm::acos(libm::sqrt(y));
        let closed_nu =
            (2.0 / PI) * (kappa * turn + libm::acos(libm::cos(lambda) / libm::cos(turn)));
        assert!((closed_nu - nu).abs() <= 1e-14 * nu, "{closed_nu} vs {nu}");
        let (got_p, got_nu) = incommensurate(&geometry(FRAC_PI_2, 0.0, lambda), kappa);
        assert!(
            (got_p - p).abs() <= 1e-14 * p && (got_nu - nu).abs() <= 1e-14 * nu,
            "lambda {lambda}: P {got_p} vs {p}, nu {got_nu} vs {nu}"
        );
    }
}

/// A hemisphere footprint covers a point half the time at every inclination: `Δ(−φ_s) = π − Δ(φ_s)`
/// and `φ_s(u + π) = −φ_s(u)`.
#[test]
fn a_hemisphere_footprint_covers_half_the_time() {
    for i_deg in [0.0, 28.5, 51.6, 63.4, 90.0, 97.8, 141.0, 180.0] {
        for phi_deg in [-89.0, -45.0, 0.0, 12.0, 60.0, 90.0] {
            // An equatorial orbit keeps a pole exactly 90 degrees away at every instant: the
            // symmetry holds, but the double below pi/2 never reaches the pole.
            if (i_deg < 1.0 || i_deg > 179.0) && phi_deg > 89.5 {
                continue;
            }
            let g = geometry(f64::to_radians(i_deg), f64::to_radians(phi_deg), FRAC_PI_2);
            let (p, _) = incommensurate(&g, 0.0647731);
            assert!((p - 0.5).abs() <= 1e-14, "i {i_deg} phi {phi_deg}: {p}");
        }
    }
}

#[test]
fn the_finite_sums_are_the_rows_of_the_trajectory() {
    let mut worst = 0.0_f64;
    for &(np, nd, i, phi, lambda, offset, passes, p) in COMMENSURATE_ROWS {
        let (got_passes, got_p, got_nu) = commensurate(&geometry(i, phi, lambda), np, nd, offset);
        assert_eq!(got_passes, passes, "{np}/{nd} at {offset}");
        assert!(
            (got_p - p).abs() <= 1e-14 * p.max(1e-3),
            "{np}/{nd} at {offset}: P {got_p} vs {p}"
        );
        assert!((got_nu * f64::from(np) - got_passes as f64).abs() <= 1e-12);
        worst = worst.max((got_p - p).abs() / p.max(1e-3));
    }
    println!("largest relative error of P: {worst:.1e}");
}

/// A track without passes has no mean pass and no mean gap; one with them has `P/ν` and
/// `(1 − P)/ν`.
#[test]
fn the_means_follow_the_passes() {
    let track = RepeatTrack::new(14, 1).unwrap_or_else(|e| panic!("{e}"));
    let g = geometry(1.7069320084504542e0, 0.0, 5.235987755982989e-2);
    let Ok(answer) = long_run_coverage(
        &g,
        Rates::Commensurate {
            track,
            ground_offset: 1.121997376282069e-1,
        },
    ) else {
        panic!("refused")
    };
    assert_eq!(answer.stats().mean_pass, None);
    assert_eq!(answer.stats().mean_gap, None);
    let Ok(answer) = long_run_coverage(
        &g,
        Rates::Incommensurate {
            kappa: 0.0660234717,
        },
    ) else {
        panic!("refused")
    };
    let s = answer.stats();
    assert_eq!(s.mean_pass, Some(s.coverage / s.passes_per_revolution));
    assert_eq!(
        s.mean_gap,
        Some((1.0 - s.coverage) / s.passes_per_revolution)
    );
}

/// The commensurate answer is within `1/Np` of the integral in `P` at every ground longitude, and
/// averaged over `K` equally spaced longitudes of a slot it is the integral within `1/(Np K)` in
/// `P` and `R/(2 Np K)` in `ν`, `R` the monotone runs of `h±` over a revolution (module docs).
#[test]
fn the_finite_sums_average_to_the_integrals_over_the_ground_longitude() {
    const K: u32 = 128;
    // R: at most 2 curves x 2 halves x 8 pieces x 2 runs a piece on these cases, a generous count.
    const RUNS: f64 = 64.0;
    for &(np, nd, i_deg, phi_deg, lambda_deg) in &[
        (233, 16, 98.2, 35.0, 8.0),
        (43, 3, 65.0, 20.0, 15.0),
        (15, 1, 51.6, 60.0, 20.0),
        (143, 10, 98.62, 85.0, 12.0),
    ] {
        let g = geometry(
            f64::to_radians(i_deg),
            f64::to_radians(phi_deg),
            f64::to_radians(lambda_deg),
        );
        let (p, nu) = incommensurate(&g, f64::from(nd) / f64::from(np));
        let slot = TAU / f64::from(np);
        let (mut sum_p, mut sum_nu, mut widest) = (0.0, 0.0, 0.0_f64);
        for j in 0..K {
            let offset = (f64::from(j) + 0.5) * slot / f64::from(K);
            let (_, p_rep, nu_rep) = commensurate(&g, np, nd, offset);
            widest = widest.max((p_rep - p).abs() * f64::from(np));
            sum_p += p_rep;
            sum_nu += nu_rep;
        }
        let scale = f64::from(np) * f64::from(K);
        let (avg_p, avg_nu) = (sum_p / f64::from(K), sum_nu / f64::from(K));
        assert!(widest <= 1.0, "{np}/{nd}: Np |P_rep - P| reaches {widest}");
        assert!(
            (avg_p - p).abs() * scale <= 1.0 + 1e-6,
            "{np}/{nd}: Np K |avg P - P| = {}",
            (avg_p - p).abs() * scale
        );
        assert!(
            (avg_nu - nu).abs() * 2.0 * scale <= RUNS,
            "{np}/{nd}: 2 Np K |avg nu - nu| = {}",
            (avg_nu - nu).abs() * 2.0 * scale
        );
        println!(
            "{np}/{nd}: Np max|P_rep - P| {widest:.3}, Np K |avg P - P| {:.3}, 2 Np K |avg nu - nu| {:.3}",
            (avg_p - p).abs() * scale,
            (avg_nu - nu).abs() * 2.0 * scale
        );
    }
}

/// S1: the time series along the trajectory, by incremental rotations from the visibility
/// condition `G · S ≥ cos λ` alone (no `Δ`, no `α`, no `h±`), `STEPS` samples a revolution, the
/// standard error by batch means.
fn simulate(i: f64, phi: f64, lambda: f64, kappa: f64) -> [(f64, f64); 2] {
    const REVOLUTIONS: u32 = 20_000;
    const BATCHES: u32 = 40;
    const STEPS: u32 = 720;
    let (si, ci) = libm::sincos(i);
    let (sp, cp) = libm::sincos(phi);
    let threshold = libm::cos(lambda);
    let du = TAU / f64::from(STEPS);
    let (s1, c1) = libm::sincos(du);
    let (sk, ck) = libm::sincos(-kappa * du);
    let x0 = 0.3;
    let per_batch = REVOLUTIONS / BATCHES;
    let mut covered_prev = false;
    let (mut ps, mut nus) = (Vec::new(), Vec::new());
    for batch in 0..BATCHES {
        let (mut hits, mut entries) = (0_u64, 0_u64);
        for rev in 0..per_batch {
            // Each revolution restarts from libm, so the rotations' round-off does not accumulate.
            let k = f64::from(batch * per_batch + rev);
            let (mut su, mut cu) = (0.0, 1.0);
            let (mut sx, mut cx) = libm::sincos(x0 - kappa * TAU * k);
            for _ in 0..STEPS {
                let dot = cp * (cx * cu - sx * ci * su) + sp * si * su;
                let covered = dot >= threshold;
                hits += u64::from(covered);
                entries += u64::from(covered && !covered_prev);
                covered_prev = covered;
                (su, cu) = (su * c1 + cu * s1, cu * c1 - su * s1);
                (sx, cx) = (sx * ck + cx * sk, cx * ck - sx * sk);
            }
        }
        ps.push(hits as f64 / f64::from(per_batch * STEPS));
        nus.push(entries as f64 / f64::from(per_batch));
    }
    [ps, nus].map(|xs| {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, libm::sqrt(var / n))
    })
}

#[test]
fn simulations_agree_within_their_sampling_error() {
    for &(i, phi, lambda, kappa, ..) in &INCOMMENSURATE_ROWS[..5] {
        let (p, nu) = incommensurate(&geometry(i, phi, lambda), kappa);
        let [(sim_p, se_p), (sim_nu, se_nu)] = simulate(i, phi, lambda, kappa);
        let (z_p, z_nu) = ((sim_p - p) / se_p, (sim_nu - nu) / se_nu);
        println!(
            "i {i:.4} phi {phi:.4}: P {p:.6} sim {sim_p:.6} z {z_p:+.2}; nu {nu:.6} sim {sim_nu:.6} z {z_nu:+.2}"
        );
        assert!(z_p.abs() < 4.0 && z_nu.abs() < 4.0, "z {z_p} {z_nu}");
    }
}

#[test]
fn refusals_are_typed() {
    let g = geometry(1.0, 0.5, 0.2);
    let rates = Rates::Incommensurate { kappa: 0.066 };
    let bad = |g: LongRunGeometry, rates| long_run_coverage(&g, rates).err();
    assert_eq!(
        bad(geometry(-0.1, 0.5, 0.2), rates),
        Some(LongRunError::BadInclination)
    );
    assert_eq!(
        bad(geometry(3.2, 0.5, 0.2), rates),
        Some(LongRunError::BadInclination)
    );
    assert_eq!(
        bad(geometry(f64::NAN, 0.5, 0.2), rates),
        Some(LongRunError::BadInclination)
    );
    assert_eq!(
        bad(geometry(1.0, 1.6, 0.2), rates),
        Some(LongRunError::BadLatitude)
    );
    assert_eq!(
        bad(geometry(1.0, f64::INFINITY, 0.2), rates),
        Some(LongRunError::BadLatitude)
    );
    assert_eq!(
        bad(geometry(1.0, 0.5, 0.0), rates),
        Some(LongRunError::BadFootprint)
    );
    assert_eq!(
        bad(geometry(1.0, 0.5, PI), rates),
        Some(LongRunError::BadFootprint)
    );
    assert_eq!(
        bad(geometry(1.0, 0.5, f64::NAN), rates),
        Some(LongRunError::BadFootprint)
    );
    for kappa in [0.0, -0.1, f64::NAN, f64::INFINITY] {
        assert_eq!(
            bad(g, Rates::Incommensurate { kappa }),
            Some(LongRunError::BadRateRatio)
        );
    }
    let track = RepeatTrack::new(15, 1).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(
        bad(
            g,
            Rates::Commensurate {
                track,
                ground_offset: f64::NAN
            }
        ),
        Some(LongRunError::BadOffset)
    );
}
