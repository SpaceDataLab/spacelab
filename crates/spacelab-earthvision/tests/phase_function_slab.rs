//! The slab against DISORT, and each phase function against the reflectance it implies.
//!
//! `monte_carlo_slab` traces photons. DISORT (Stamnes et al.) solves the azimuthally decomposed
//! transfer equation on a Gauss quadrature by eigen-decomposition. They share no arithmetic, which is
//! what makes the comparison a verification rather than a Monte Carlo agreeing with itself. The
//! reference values below come from `tools/gen_disort_oracle.py`, run against `pydisort` — an
//! ephemeral oracle, never a dependency of this workspace.
//!
//! The oracle's own convergence is recorded with it: moving from 32 to 96 streams and moments shifts
//! the hardest case (`g = 0.85`) by `1.8e-7` in reflectance, four orders below the Monte Carlo's
//! sampling noise at the photon counts used here. So the residuals reported are the Monte Carlo's.
//!
//! Both codes are given the *analytic* phase function rather than a shared table: DISORT takes
//! Legendre moments, which for Henyey–Greenstein are `gˡ` exactly, and 96 of them put the truncated
//! expansion within `1e-6` of the closed form. Neither code has to imitate the other's representation
//! for the comparison to be of the same scatterer.
//!
//! The tabulated members — Mie and ice — are not in the DISORT table, because a halo needs hundreds
//! of Legendre moments and the truncation would become the thing being measured. They are covered
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

/// The slab against DISORT: seven cases spanning isotropic to strongly forward-peaked, absorbing and
/// conservative, normal and slant incidence.
#[test]
fn the_photon_slab_reproduces_disort() {
    // name, tau, omega, mu0, phase, DISORT reflectance, DISORT transmittance
    let cases: [(&str, f64, f64, f64, PhaseFunction, f64, f64); 7] = [
        (
            "isotropic thin",
            0.0,
            1.0,
            1.0,
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
            0.0,
            1.000_000_001,
        ),
        (
            "isotropic",
            1.0,
            1.0,
            1.0,
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
            0.341_328_759,
            0.658_671_240,
        ),
        (
            "isotropic absorbing",
            1.0,
            0.9,
            0.6,
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
            0.360_929_444,
            0.458_943_252,
        ),
        (
            "hg g=0.5",
            2.0,
            1.0,
            0.8,
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.5 },
            0.388_621_810,
            0.611_378_191,
        ),
        (
            "hg g=0.85",
            4.0,
            0.99,
            0.7,
            PhaseFunction::HenyeyGreenstein { asymmetry: 0.85 },
            0.282_283_476,
            0.640_119_247,
        ),
        (
            "hg g=-0.4",
            1.5,
            0.85,
            1.0,
            PhaseFunction::HenyeyGreenstein { asymmetry: -0.4 },
            0.366_877_511,
            0.355_597_215,
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
            0.473_042_276,
            0.291_857_444,
        ),
    ];

    let mut worst: f64 = 0.0;
    for (name, tau, omega, mu0, phase, want_r, want_t) in cases {
        let atmosphere = Atmosphere {
            optical_depth: tau,
            single_scattering_albedo: omega,
            phase,
        };
        let result = monte_carlo_slab(&atmosphere, mu0, PHOTONS, 20_240_813);
        let (dr, dt) = (
            (result.reflectance - want_r).abs(),
            (result.transmittance - want_t).abs(),
        );
        println!(
            "{name:20} R {:.6} vs {want_r:.6} ({dr:.2e})   T {:.6} vs {want_t:.6} ({dt:.2e})",
            result.reflectance, result.transmittance
        );
        worst = worst.max(dr).max(dt);
    }
    // Three standard errors of a binomial at a million photons is 1.5e-3; the assertion is set from
    // that rather than from the residuals, so a real bias has somewhere to show up.
    assert!(worst < 1.5e-3, "worst DISORT residual {worst}");
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
/// This is what carries the DISORT verification across to the tabulated members: the analytic path is
/// checked against a second solver above, and this says the tabulated path is the same path. Both
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
