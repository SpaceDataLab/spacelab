//! The 6S coupling on atmospheres AERONET measured, against DISORT solving the same ones.
//!
//! Every earlier check on [`coupling_from_slab`] tested it as a *solver*: against DISORT in
//! `phase_function_slab`, against libRadtran through pyspectral in `pyspectral_rayleigh`, on
//! optical depths somebody chose — `0.1`, `1e-4`, `0.7`, `1.2` in the module's own fixtures, and a
//! US Standard profile in `docs/verification.md` §5.1. None of those atmospheres was measured. This
//! one closes that: `tools/gen_aeronet_oracle.py` takes the optical depth, the single-scattering
//! albedo and the asymmetry parameter — which are exactly [`Atmosphere`]'s three fields — from
//! AERONET Version 3 **Level 2.0** almucantar inversions, all three from the same retrieval so the
//! triple is self-consistent, and `tools/gen_aeronet_disort_oracle.py` solves those same
//! atmospheres with DISORT.
//!
//! A hundred and twenty atmospheres: five aerosol regimes (savanna burning, Amazon smoke, Arabian
//! dust, Sahelian dust, industrial haze) × six retrievals spanning each site's own `AOD(440)` range
//! × the four inversion wavelengths. They cover **τ 0.079–3.778, ω 0.755–0.992, g 0.442–0.826** at
//! solar zenith angles the sun actually had, 53.3° to 75.8°. No hand-chosen fixture in this crate
//! ever reached `ω = 0.76`, and none of them absorbed at all.
//!
//! **The residuals, and what they are made of.** Every point is measured at [`SEEDS`] independent
//! streams, so each carries its own standard error, and the residual is reported against it rather
//! than against a number from somewhere else. At [`PHOTONS`] photons:
//!
//! | term | n | worst | RMS | signed mean | RMS sampling noise | ratio |
//! |---|---|---|---|---|---|---|
//! | `transmittance_down` | 120 | 5.92e-3 | 1.23e-3 | +1.46e-4 | 9.83e-4 | 1.25 |
//! | `transmittance_up`   | 120 | 2.58e-3 | 7.64e-4 | −3.66e-5 | 7.62e-4 | 1.00 |
//! | `spherical_albedo`   | 120 | 7.00e-3 | 2.29e-3 | −1.50e-4 | 2.35e-3 | 0.98 |
//! | `path_reflectance`   | 360 | 1.63e-2 | 4.33e-3 | +1.06e-4 | 4.25e-3 | 1.02 |
//!
//! **All four terms agree with DISORT at their own sampling noise**, and the signed means say they
//! do not lean: against standard errors of 9.0e-5, 7.0e-5, 2.1e-4 and 2.2e-4 the four means sit at
//! +1.6σ, −0.5σ, −0.7σ and +0.5σ. `T↓`'s ratio of 1.25 is the largest and it is one heavy row — the
//! 5.92e-3 worst case is 19 % of that term's total variance on its own.
//!
//! **How this differs from the reading #326 published, which is the point of #327.** That version
//! reported the path reflectance's worst case as about twice its noise and left open whether the
//! excess was a systematic in the next-event estimator at high `τ`. It was neither: it was two
//! statistics that could not be compared. The residual was a maximum over 180 points at three
//! azimuths; the noise column was a maximum deviation among *three seeds* at azimuth 0 alone, which
//! is both a much smaller order statistic and the azimuth where the path reflectance is largest and
//! its relative noise smallest — at 1.6 M photons the backscattering azimuth is **1.66×** noisier
//! than the forward one. Two convergence studies settled it. Ours, in photons
//! (`examples/aeronet_path_residual.rs`, 32 seeds on these atmospheres): RMS bias **1.53e-3 →
//! 7.06e-4 → 3.60e-4** at 100 k → 400 k → 1.6 M, ratios 2.16 and 1.96 against the 2.00 that `1/√N`
//! predicts, with no floor and the aggregate lean never above 1.8σ. Theirs, in streams
//! (`tools/gen_aeronet_disort_oracle.py`): the reference moves **1.67e-7 worst** between 96 and 256
//! streams. The residual is our sampling and nothing else.
//!
//! **The spherical albedo's dominant error is the quadrature, not the sampler.**
//! [`coupling_from_slab`] integrates `∫₀¹ R(μ) 2μ dμ` on a four-node Gauss–Legendre rule; the
//! reference is computed on *those same four nodes*, so the 7.00e-3 above is the Monte Carlo's. A
//! sixteen-node value is generated beside it, and the four-node rule differs from it by **1.30e-2
//! worst, 2.66e-3 RMS** — larger than our disagreement with DISORT on the shared nodes. It is
//! recorded rather than fixed: `s` enters the correction only through `1/(1 − s·ρ_s)`, and at the
//! `s ≈ 0.1` these atmospheres produce and a surface at `ρ_s = 0.15` a 1.3 % error in `s` moves the
//! retrieved reflectance by 2e-4 relative. The term is small enough that its own quadrature is not
//! worth a node.
//!
//! **What assuming an atmosphere costs, which is the number this whole leg exists to produce.**
//! Build the TOA reflectance of a surface at `ρ_s = 0.15` through each measured atmosphere, then
//! invert it with the coupling of the atmosphere the fixtures assume — `τ = 0.1`, `ω = 1`, `g = 0`.
//! The retrieved surface reflectance is wrong by **0.2933 at worst and 0.1308 RMS**, against a
//! truth of 0.15: a 196 % error on the worst column. The best case is 1e-4, on the thinnest and
//! least absorbing of the hundred and twenty — assuming is nearly free exactly where the assumption
//! happens to be true, and ruinous everywhere else. That is the argument for measuring the
//! atmosphere, stated as a number rather than as a preference.
//!
//! **Bounds of the claim.** AERONET promotes an almucantar SSA to Level 2.0 only where the load
//! constrains it — in practice `AOD(440) ≥ 0.4` — so every column here is a loaded one and none of
//! this speaks for a pristine maritime atmosphere. AERONET measures no ozone, and its optical depth
//! is aerosol-only with the molecular and gaseous terms already removed, so no ozone enters either
//! side and none of it hides in the residual; the molecular half has its own oracle in
//! `pyspectral_rayleigh`. And this is still a comparison of two solvers on a measured *atmosphere*,
//! not against a measured *radiance* — that rung is RadCalNet's, and `radcalnet_measured_radiance`
//! is where it is climbed.

#![allow(clippy::doc_markdown)] // RadCalNet and the other network names are proper nouns, not items
#![allow(clippy::expect_used)] // a malformed fixture is a broken generator, and should stop the run
#![allow(clippy::cast_precision_loss)] // the counts are 120 rows and 3 azimuths

use spacelab_earthvision::{
    Atmosphere, AtmosphericCoupling, PhaseFunction, coupling_from_slab, path_reflectance,
};

const PHOTONS: u32 = 150_000;
const SEED: u64 = 20_260_817;

/// Independent streams per atmosphere, per azimuth.
///
/// Three, and the *gate* below pools the standard errors they estimate over the whole fixture rather
/// than trusting any one of them: a per-row `t` on two degrees of freedom has tails far too heavy to
/// threshold — `P(|t₂| > 3) ≈ 0.1` — while an RMS over 120 or 360 of them is stable to a few per
/// cent. Two numbers of the same kind, compared; #327 is what happens when they are not.
const SEEDS: u64 = 3;

/// The view cosine the reference was generated at — a 36.9° view zenith.
const MU_VIEW: f64 = 0.8;
/// Azimuths from the forward-scattering half-plane, as `path_reflectance` measures them.
const AZIMUTHS_DEG: [f64; 3] = [0.0, 90.0, 180.0];

/// The atmosphere the module's own fixtures assume, and that `docs/verification.md` §5.1 feeds the
/// model: a chosen optical depth, no absorption, isotropic scattering.
const ASSUMED: Atmosphere<'static> = Atmosphere {
    optical_depth: 0.1,
    single_scattering_albedo: 1.0,
    phase: PhaseFunction::HenyeyGreenstein { asymmetry: 0.0 },
};

/// A vegetated-surface reflectance in the red — the retrieval this bias is reported on.
const SURFACE: f64 = 0.15;

struct Measured {
    key: String,
    regime: String,
    solar_zenith_deg: f64,
    optical_depth: f64,
    single_scattering_albedo: f64,
    asymmetry: f64,
}

struct Reference {
    key: String,
    transmittance_down: f64,
    transmittance_up: f64,
    spherical_albedo_g4: f64,
    spherical_albedo_g16: f64,
    path_reflectance: [f64; 3],
}

fn fields(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

fn data(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
}

fn measured() -> Vec<Measured> {
    let text = include_str!("fixtures/aeronet_atmosphere.txt");
    data(text)
        .map(|line| {
            let f = fields(line);
            let parse = |index: usize| f[index].parse::<f64>().expect("a fixture number");
            Measured {
                key: format!("{} {} {} {}", f[0], f[2], f[3], f[5]),
                regime: f[1].to_owned(),
                solar_zenith_deg: parse(4),
                optical_depth: parse(6),
                single_scattering_albedo: parse(7),
                asymmetry: parse(8),
            }
        })
        .collect()
}

fn reference() -> Vec<Reference> {
    let text = include_str!("fixtures/aeronet_coupling_disort.txt");
    data(text)
        .map(|line| {
            let f = fields(line);
            let parse = |index: usize| f[index].parse::<f64>().expect("a fixture number");
            Reference {
                key: format!("{} {} {} {}", f[0], f[1], f[2], f[3]),
                transmittance_down: parse(4),
                transmittance_up: parse(5),
                spherical_albedo_g4: parse(6),
                spherical_albedo_g16: parse(7),
                path_reflectance: [parse(8), parse(9), parse(10)],
            }
        })
        .collect()
}

fn slab(row: &Measured) -> Atmosphere<'static> {
    Atmosphere {
        optical_depth: row.optical_depth,
        single_scattering_albedo: row.single_scattering_albedo,
        phase: PhaseFunction::HenyeyGreenstein {
            asymmetry: row.asymmetry,
        },
    }
}

fn mu_sun(row: &Measured) -> f64 {
    row.solar_zenith_deg.to_radians().cos()
}

/// Root-mean-square of a relative-error sample, reported beside its worst member.
fn spread(errors: &[f64]) -> (f64, f64) {
    let worst = errors.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let rms = (errors.iter().map(|e| e * e).sum::<f64>() / errors.len() as f64).sqrt();
    (worst, rms)
}

/// A seed per atmosphere, per azimuth, per draw.
///
/// All three have to enter it. [`path_reflectance`] is called once per azimuth, so a seed that
/// ignored the row and the azimuth would hand the whole fixture one random walk, and the RMS below
/// would be one draw of a statistic rather than an average of 120 independent ones.
fn stream(row: usize, azimuth: usize, draw: u64) -> u64 {
    let point = (row * AZIMUTHS_DEG.len() + azimuth) as u64;
    SEED.wrapping_add(point.wrapping_mul(0xff51_afd7_ed55_8ccd))
        .wrapping_add(draw.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

/// One term's disagreement with DISORT and the sampling noise it was measured with — the two numbers
/// whose ratio is the only honest reading of a Monte Carlo residual.
#[derive(Default)]
struct Term {
    /// `mean(ours)/reference − 1`, one entry per comparison point.
    bias: Vec<f64>,
    /// The standard error of that mean, relative to the reference, at the same point.
    noise: Vec<f64>,
}

impl Term {
    /// Fold in one comparison point: `values` are the same quantity at [`SEEDS`] streams.
    fn push(&mut self, values: &[f64], reference: f64) {
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (count - 1.0);
        self.bias.push(mean / reference - 1.0);
        self.noise.push((variance / count).sqrt() / reference);
    }

    /// Print the term's worst and RMS bias beside the RMS noise, and return the three.
    ///
    /// The signed mean is there to be read against `noise/√n`: an RMS ratio near 1 says the residual
    /// is the right *size* to be sampling, and only a mean consistent with zero says it does not
    /// lean.
    fn report(&self, name: &str) -> (f64, f64, f64) {
        let (worst, rms) = spread(&self.bias);
        let (_, noise) = spread(&self.noise);
        let count = self.bias.len() as f64;
        let mean = self.bias.iter().sum::<f64>() / count;
        println!(
            "MEASURED {name:<10} n {:3}  bias worst {worst:.3e} rms {rms:.3e} mean {mean:+.3e}  \
             noise rms {noise:.3e} mean-se {:.3e}  ratio {:.2}",
            self.bias.len(),
            noise / count.sqrt(),
            rms / noise
        );
        (worst, rms, noise)
    }
}

#[test]
fn the_coupling_terms_reproduce_disort_on_measured_atmospheres() {
    let rows = measured();
    let references = reference();
    assert_eq!(rows.len(), references.len(), "the two fixtures pair up");

    let mut down = Term::default();
    let mut up = Term::default();
    let mut albedo = Term::default();
    let mut path = Term::default();
    let mut quadrature = Vec::new();

    for (index, (row, expected)) in rows.iter().zip(references.iter()).enumerate() {
        assert_eq!(row.key, expected.key, "the fixtures are in the same order");
        let atmosphere = slab(row);
        let couplings: Vec<AtmosphericCoupling> = (0..SEEDS)
            .map(|draw| {
                coupling_from_slab(
                    &atmosphere,
                    mu_sun(row),
                    MU_VIEW,
                    AZIMUTHS_DEG[0].to_radians(),
                    PHOTONS,
                    stream(index, 0, draw),
                )
            })
            .collect();
        let draws = |term: fn(&AtmosphericCoupling) -> f64| {
            couplings.iter().map(term).collect::<Vec<f64>>()
        };

        down.push(
            &draws(|c| c.transmittance_down),
            expected.transmittance_down,
        );
        up.push(&draws(|c| c.transmittance_up), expected.transmittance_up);
        albedo.push(&draws(|c| c.spherical_albedo), expected.spherical_albedo_g4);
        quadrature.push(expected.spherical_albedo_g4 / expected.spherical_albedo_g16 - 1.0);

        path.push(&draws(|c| c.path_reflectance), expected.path_reflectance[0]);
        for (azimuth_index, azimuth) in AZIMUTHS_DEG.iter().enumerate().skip(1) {
            let draws: Vec<f64> = (0..SEEDS)
                .map(|draw| {
                    path_reflectance(
                        &atmosphere,
                        mu_sun(row),
                        MU_VIEW,
                        azimuth.to_radians(),
                        PHOTONS,
                        stream(index, azimuth_index, draw),
                    )
                })
                .collect();
            path.push(&draws, expected.path_reflectance[azimuth_index]);
        }
    }

    let measurements = [
        ("T-down", down.report("T-down")),
        ("T-up", up.report("T-up")),
        ("albedo", albedo.report("albedo")),
        ("path", path.report("path")),
    ];
    let (quad_worst, quad_rms) = spread(&quadrature);
    println!("MEASURED quadrature worst {quad_worst:.3e} rms {quad_rms:.3e}");

    // The gate is the ratio and not the magnitude. A residual that is the estimator's own sampling
    // noise falls as `1/sqrt(photons)` and says nothing about either solver, so what has to hold is
    // that it is no larger than the noise it was measured with -- and both sides are then the same
    // statistic over the same points, which is exactly what #326's comparison of a
    // max-over-three-seeds against a max-over-180-rows was not (#327).
    // 1.6, against a measured maximum of 1.25 on `T-down`. The ratio itself carries about 8 % of
    // sampling error at 120 points -- 6.5 % from the RMS residual and 4.6 % from an RMS of variances
    // on two degrees of freedom apiece -- so the bound has to clear the measurement by more than
    // that or it is a coin toss rather than a gate.
    for (name, (worst, rms, noise)) in measurements {
        assert!(
            rms < 1.6 * noise,
            "{name}: RMS residual {rms:.3e} against RMS sampling noise {noise:.3e}"
        );
        // Both could grow together and the ratio would not notice; these are the magnitudes the
        // module docs publish, at about 1.5x the measured worst case.
        let bound = match name {
            "T-down" => 9.0e-3,
            "T-up" => 4.0e-3,
            "albedo" => 1.1e-2,
            _ => 2.5e-2,
        };
        assert!(worst < bound, "{name}: worst residual {worst:.3e}");
    }
    assert!(
        quad_worst < 2.0e-2,
        "the four-node rule against sixteen nodes, worst {quad_worst:.3e}"
    );
}

#[test]
fn the_fixture_spans_the_regimes_a_correction_has_to_survive() {
    let rows = measured();
    let regimes: std::collections::BTreeSet<&str> =
        rows.iter().map(|row| row.regime.as_str()).collect();
    let tau: Vec<f64> = rows.iter().map(|row| row.optical_depth).collect();
    let ssa: Vec<f64> = rows
        .iter()
        .map(|row| row.single_scattering_albedo)
        .collect();
    let g: Vec<f64> = rows.iter().map(|row| row.asymmetry).collect();
    let range = |v: &[f64]| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    println!(
        "MEASURED regimes {} tau {:?} ssa {:?} g {:?} sza {:?}",
        regimes.len(),
        range(&tau),
        range(&ssa),
        range(&g),
        range(&rows.iter().map(|r| r.solar_zenith_deg).collect::<Vec<_>>()),
    );
    assert_eq!(regimes.len(), 5, "five aerosol regimes");
    // Every column absorbs and every one is forward-scattering: neither is true of a single fixture
    // this crate had before, and a regenerated file that quietly narrowed to the easy end would
    // leave the residuals above looking better than they are.
    assert!(
        ssa.iter().all(|&w| w < 1.0),
        "every measured column absorbs"
    );
    assert!(g.iter().all(|&value| value > 0.4), "every column is hazy");
    assert!(
        range(&tau).0 < 0.1 && range(&tau).1 > 3.0,
        "two decades of tau"
    );
    assert!(range(&ssa).0 < 0.8, "the absorbing end is represented");
}

#[test]
fn correcting_with_an_assumed_atmosphere_biases_the_surface_reflectance() {
    let rows = measured();
    let mut errors = Vec::new();
    for row in &rows {
        let truth = coupling_from_slab(
            &slab(row),
            mu_sun(row),
            MU_VIEW,
            AZIMUTHS_DEG[0].to_radians(),
            PHOTONS,
            SEED,
        );
        let assumed: AtmosphericCoupling = coupling_from_slab(
            &ASSUMED,
            mu_sun(row),
            MU_VIEW,
            AZIMUTHS_DEG[0].to_radians(),
            PHOTONS,
            SEED,
        );
        let toa = truth.toa_from_surface(SURFACE);
        errors.push(assumed.surface_from_toa(toa) - SURFACE);
    }
    let (worst, rms) = spread(&errors);
    let best = errors.iter().fold(f64::INFINITY, |a, b| a.min(b.abs()));
    println!("MEASURED assumed-atmosphere bias worst {worst:.4} rms {rms:.4} best {best:.4}");
    // The headline of this leg: on a surface whose reflectance is 0.15, assuming the atmosphere
    // costs up to 0.2933 -- a 196 % error -- and 0.1308 RMS across the fixture. The floor is asserted
    // rather than the value, so a better sampler cannot fail the test, only a smaller error can.
    assert!(worst > 0.25, "worst assumed-atmosphere bias {worst:.4}");
    assert!(rms > 0.1, "RMS assumed-atmosphere bias {rms:.4}");
    assert!(best > 0.0, "assuming an atmosphere is never free");
}
