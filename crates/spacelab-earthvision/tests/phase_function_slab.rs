//! The slab against the transfer equation, and each phase function against the reflectance it
//! implies.
//!
//! `monte_carlo_slab` traces photons. `derivations/slab_doubling.jl` solves the azimuthally
//! decomposed transfer equation on a Gauss quadrature by doubling and adding. They share no
//! arithmetic, which is what makes the comparison a verification rather than a Monte Carlo agreeing
//! with itself. The solver is given the *analytic* phase function, integrated over azimuth by the
//! trapezoid rule, so no Legendre truncation stands between the two. DISORT (`pydisort`,
//! `tools/gen_disort_oracle.py`) solved the same seven cases first; its distance is a comparator row.
//!
//! The tabulated members — Mie and ice — are not in the reference table. They are covered
//! instead by two checks that do not need a second solver: energy conservation at `ω = 1`, which only
//! a correct sampler passes, and the thin-slab limit, where the reflectance is the phase function's
//! own backscatter fraction and nothing else.

// Test code: the tabulated fixtures compare exactly against the values they were built from, and the
// grid indices are small integers.
#![allow(
    clippy::float_cmp,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::items_after_statements,
    clippy::needless_range_loop,
    clippy::unwrap_used,
    clippy::expect_used
)]

use spacelab_earthvision::ice::{HexagonalCrystal, trace_on_grid};
use spacelab_earthvision::mie::{Complex, SizeDistribution};
use spacelab_earthvision::phase::{PhaseFunction, TabulatedPhase, tabulate_mie};
use spacelab_earthvision::radiative_transfer::{Atmosphere, monte_carlo_slab};

const PHOTONS: u32 = 1_000_000;

/// A quarter-degree grid in scattering angle, ascending in cosine — fine enough for a halo.
fn angle_grid() -> Vec<f64> {
    (0..721)
        .map(|i| {
            ((720 - i) as f64 * std::f64::consts::PI / 720.0)
                .cos()
                .clamp(-1.0, 1.0)
        })
        .collect()
}

/// `∫₋₁⁰ p dμ` — the fraction of a scattering event that goes backwards, by Simpson on a fine grid.
fn backscatter_fraction(phase: &PhaseFunction) -> f64 {
    const NODES: usize = 200_001;
    let h = 1.0 / (NODES - 1) as f64;
    let mut total = 0.0;
    for i in 0..NODES {
        let weight = if i == 0 || i == NODES - 1 {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += weight * phase.value(-1.0 + h * i as f64);
    }
    total * h / 3.0
}

/// The seven slab cases: isotropic to strongly forward-peaked, absorbing and conservative, normal
/// and slant incidence. Name, `τ`, `ω`, `μ₀`, phase function.
const CASES: [(&str, f64, f64, f64, PhaseFunction<'static>); 7] = [
    (
        "isotropic thin",
        0.0,
        1.0,
        1.0,
        PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
    ),
    (
        "isotropic",
        1.0,
        1.0,
        1.0,
        PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
    ),
    (
        "isotropic absorbing",
        1.0,
        0.9,
        0.6,
        PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
    ),
    (
        "hg g=0.5",
        2.0,
        1.0,
        0.8,
        PhaseFunction::HenyeyGreenstein { asymmetry: 0.5 },
    ),
    (
        "hg g=0.85",
        4.0,
        0.99,
        0.7,
        PhaseFunction::HenyeyGreenstein { asymmetry: 0.85 },
    ),
    (
        "hg g=-0.4",
        1.5,
        0.85,
        1.0,
        PhaseFunction::HenyeyGreenstein { asymmetry: -0.4 },
    ),
    (
        "two-lobe",
        3.0,
        0.95,
        0.75,
        PhaseFunction::DoubleHenyeyGreenstein {
            forward: 0.85,
            backward: -0.6,
            forward_weight: 0.6,
        },
    ),
];

const SEED: u64 = 20_240_813;

fn slab(case: &(&str, f64, f64, f64, PhaseFunction<'static>)) -> (f64, f64) {
    let (_, tau, omega, mu0, phase) = *case;
    let atmosphere = Atmosphere {
        optical_depth: tau,
        single_scattering_albedo: omega,
        phase,
    };
    let result = monte_carlo_slab(&atmosphere, mu0, PHOTONS, SEED);
    (result.reflectance, result.transmittance)
}

/// (G3) The slab against the transfer equation solved by doubling and adding, on the seven [`CASES`].
///
/// `derivations/slab_doubling.jl` solves the equation azimuthal order by order on 128 Gauss nodes
/// per hemisphere, doubling a single-scattering layer of `2⁻⁴⁰` of the depth forty times; 256
/// nodes move no value by more than 8.4e-12 and a `2⁻³⁴` start by 4.2e-9, and
/// `derivations/slab_doubling.mac` recomputes the method on a reduced input in 40-digit arithmetic.
/// A photon ends in exactly one of reflection, transmission and absorption, so each estimate is a
/// binomial proportion with `σ = √(p(1 − p)/N)` at the reference `p`. Every estimate is held to 4σ;
/// the worst of the twelve is 1.22σ. At `τ = 0` both are exact and `σ = 0`.
#[test]
fn the_photon_slab_meets_the_doubling_solution() {
    // reflectance, total transmittance
    const DOUBLING: [(f64, f64); 7] = [
        (0.0, 1.0),
        (0.341_328_759_811_201, 0.658_671_240_196_554),
        (0.360_929_443_633_221, 0.458_943_251_841_913),
        (0.388_621_809_357_278, 0.611_378_190_679_398),
        (0.282_283_476_174_921, 0.640_119_246_675_284),
        (0.366_877_511_127_119, 0.355_597_215_076_836),
        (0.473_042_276_030_450, 0.291_857_443_995_510),
    ];
    let mut worst: f64 = 0.0;
    for (case, (want_r, want_t)) in CASES.iter().zip(DOUBLING) {
        let (r, t) = slab(case);
        let mut sigmas = [0.0; 2];
        for (k, (got, want)) in [(r, want_r), (t, want_t)].into_iter().enumerate() {
            let sigma = (want * (1.0 - want) / f64::from(PHOTONS)).sqrt();
            if sigma == 0.0 {
                assert_eq!(got, want, "{}: a certain outcome", case.0);
            } else {
                sigmas[k] = (got - want).abs() / sigma;
                worst = worst.max(sigmas[k]);
            }
        }
        println!(
            "{:20} R {r:.6} vs {want_r:.6} ({:.2}σ)   T {t:.6} vs {want_t:.6} ({:.2}σ)",
            case.0, sigmas[0], sigmas[1]
        );
    }
    assert!(worst < 4.0, "worst residual {worst:.2}σ");
}

/// The distance from the photon slab to DISORT on the seven [`CASES`]: `tools/gen_disort_oracle.py`,
/// `pydisort` at 96 streams and 96 moments.
#[test]
#[ignore = "comparator: the distance to disort, not a verdict"]
fn slab_distance_to_disort() {
    // reflectance, total transmittance
    const DISORT: [(f64, f64); 7] = [
        (0.0, 1.000_000_001),
        (0.341_328_759, 0.658_671_240),
        (0.360_929_444, 0.458_943_252),
        (0.388_621_810, 0.611_378_191),
        (0.282_283_476, 0.640_119_247),
        (0.366_877_511, 0.355_597_215),
        (0.473_042_276, 0.291_857_444),
    ];
    let mut distances = Vec::new();
    for (case, (want_r, want_t)) in CASES.iter().zip(DISORT) {
        let (r, t) = slab(case);
        distances.push((r - want_r).abs());
        distances.push((t - want_t).abs());
    }
    let max = distances.iter().copied().fold(0.0, f64::max);
    let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
    println!(
        "comparison | disort | not recorded | slab R and T, Monte Carlo at 1e6 photons | 1 | {} | {max:.3e} | {rms:.3e}",
        distances.len()
    );
}

/// The thin-slab limit ties every member to its own phase function, with no second solver involved.
///
/// As `τ → 0` at normal incidence a photon scatters at most once, so the reflectance is `ω τ` times
/// the fraction of a single scattering that goes backwards. That fraction is an integral of `p` and
/// nothing else — so this is the check that reaches the tabulated members, whose Legendre expansions
/// are too long to hand to a discrete-ordinates code. Residuals are recorded per member.
#[test]
fn the_thin_slab_reflectance_is_each_phase_functions_backscatter_fraction() {
    let mu = angle_grid();

    let mut mie_value = vec![0.0; mu.len()];
    let mut mie_cumulative = vec![0.0; mu.len()];
    let droplets = SizeDistribution::Gamma {
        effective_radius: 6.0,
        effective_variance: 0.1,
    };
    let optics = tabulate_mie(
        &droplets,
        Complex::new(1.333, 0.0),
        0.55,
        &mu,
        &mut mie_value,
        &mut mie_cumulative,
    )
    .unwrap();
    let mie =
        PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &mie_value, &mie_cumulative).unwrap());

    let mut ice_value = vec![0.0; mu.len()];
    let mut ice_cumulative = vec![0.0; mu.len()];
    let crystal = HexagonalCrystal {
        aspect_ratio: 2.0,
        refractive_index: 1.3105,
        absorption: 0.0,
        size_parameter: 200.0,
        roughness: 0.0,
    };
    trace_on_grid(&crystal, &mu, &mut ice_value, 400_000, 7).unwrap();
    TabulatedPhase::prepare(&mu, &mut ice_value, &mut ice_cumulative).unwrap();
    let ice =
        PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &ice_value, &ice_cumulative).unwrap());

    println!(
        "Mie droplet g = {:.4}, omega = {:.6}",
        optics.asymmetry, optics.single_scattering_albedo
    );
    let members: [(&str, PhaseFunction); 5] = [
        (
            "isotropic",
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
        ),
        (
            "hg g=0.85",
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.85 },
        ),
        (
            "two-lobe",
            PhaseFunction::DoubleHenyeyGreenstein {
                forward: 0.85,
                backward: -0.6,
                forward_weight: 0.6,
            },
        ),
        ("mie droplet", mie),
        ("hexagonal ice", ice),
    ];

    const TAU: f64 = 1e-3;
    const OMEGA: f64 = 0.9;
    for (name, phase) in members {
        let want = OMEGA * TAU * backscatter_fraction(&phase);
        let atmosphere = Atmosphere {
            optical_depth: TAU,
            single_scattering_albedo: OMEGA,
            phase,
        };
        let got = monte_carlo_slab(&atmosphere, 1.0, 20_000_000, 4_242).reflectance;
        println!(
            "{name:16} R = {got:.3e} vs omega*tau*b = {want:.3e}   ratio {:.4}",
            got / want
        );
        // The limit is approached as O(tau ln tau), so at tau = 1e-3 a few per cent of bias is the
        // physics rather than an error; twenty million photons put the sampling noise an order below
        // that. A member whose sampler disagreed with its own density would miss by far more.
        assert!(
            (0.9..1.1).contains(&(got / want)),
            "{name}: {got} vs {want}"
        );
    }
}

/// A conservative slab loses nothing, whatever it scatters with.
///
/// `ω = 1` makes `R + T = 1` an identity, and it is one the sampler can break: a phase function that
/// returns a cosine outside `[−1, 1]`, or a tabulated CDF that does not reach 1, strands photons
/// inside the layer and they surface here as `unresolved`. This is the check that reaches the Mie and
/// ice members at finite optical depth.
#[test]
fn every_member_conserves_energy_in_a_conservative_slab() {
    let mu = angle_grid();

    let mut mie_value = vec![0.0; mu.len()];
    let mut mie_cumulative = vec![0.0; mu.len()];
    tabulate_mie(
        &SizeDistribution::Lognormal {
            median_radius: 0.3,
            log_sigma: 0.7,
        },
        Complex::new(1.53, 0.0),
        0.55,
        &mu,
        &mut mie_value,
        &mut mie_cumulative,
    )
    .unwrap();

    let mut ice_value = vec![0.0; mu.len()];
    let mut ice_cumulative = vec![0.0; mu.len()];
    trace_on_grid(
        &HexagonalCrystal {
            aspect_ratio: 2.0,
            refractive_index: 1.3105,
            absorption: 0.0,
            size_parameter: 200.0,
            roughness: 0.0,
        },
        &mu,
        &mut ice_value,
        200_000,
        11,
    )
    .unwrap();
    TabulatedPhase::prepare(&mu, &mut ice_value, &mut ice_cumulative).unwrap();

    let members: [(&str, PhaseFunction); 4] = [
        (
            "hg g=0.9",
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.9 },
        ),
        (
            "two-lobe",
            PhaseFunction::DoubleHenyeyGreenstein {
                forward: 0.9,
                backward: -0.7,
                forward_weight: 0.5,
            },
        ),
        (
            "mie aerosol",
            PhaseFunction::Tabulated(
                TabulatedPhase::new(&mu, &mie_value, &mie_cumulative).unwrap(),
            ),
        ),
        (
            "hexagonal ice",
            PhaseFunction::Tabulated(
                TabulatedPhase::new(&mu, &ice_value, &ice_cumulative).unwrap(),
            ),
        ),
    ];
    for (name, phase) in members {
        let result = monte_carlo_slab(
            &Atmosphere {
                optical_depth: 3.0,
                single_scattering_albedo: 1.0,
                phase,
            },
            0.8,
            200_000,
            5,
        );
        assert_eq!(
            result.absorptance, 0.0,
            "{name} absorbed in a conservative slab"
        );
        assert_eq!(result.unresolved, 0.0, "{name} stranded photons");
        assert!(
            (result.reflectance + result.transmittance - 1.0).abs() < 1e-12,
            "{name}: R + T = {}",
            result.reflectance + result.transmittance
        );
        assert!(
            result.reflectance > 0.01 && result.transmittance > 0.01,
            "{name} filled one channel only"
        );
    }
}

/// A finely tabulated Henyey–Greenstein produces the same slab as the analytic one. (S1)
///
/// This is what carries the doubling verification across to the tabulated members: the analytic path
/// is checked against the transfer equation above, and this says the tabulated path is the same
/// path. Both
/// runs use the identical photon stream, so the difference is the phase function and nothing else.
#[test]
fn a_tabulated_henyey_greenstein_gives_the_same_slab_as_the_analytic_one() {
    let mu = angle_grid();
    let analytic = PhaseFunction::HenyeyGreenstein { asymmetry: 0.7 };
    let mut value: Vec<f64> = mu.iter().map(|m| analytic.value(*m)).collect();
    let mut cumulative = vec![0.0; mu.len()];
    TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).unwrap();
    let tabulated =
        PhaseFunction::Tabulated(TabulatedPhase::new(&mu, &value, &cumulative).unwrap());

    for (tau, omega, mu0) in [(1.0, 1.0, 1.0), (3.0, 0.95, 0.7), (0.3, 0.8, 0.5)] {
        let one = monte_carlo_slab(
            &Atmosphere {
                optical_depth: tau,
                single_scattering_albedo: omega,
                phase: analytic,
            },
            mu0,
            PHOTONS,
            77,
        );
        let two = monte_carlo_slab(
            &Atmosphere {
                optical_depth: tau,
                single_scattering_albedo: omega,
                phase: tabulated,
            },
            mu0,
            PHOTONS,
            77,
        );
        println!(
            "tau={tau} omega={omega} mu0={mu0}: dR = {:.2e}, dT = {:.2e}",
            (one.reflectance - two.reflectance).abs(),
            (one.transmittance - two.transmittance).abs()
        );
        assert!((one.reflectance - two.reflectance).abs() < 3e-3);
        assert!((one.transmittance - two.transmittance).abs() < 3e-3);
    }
}
