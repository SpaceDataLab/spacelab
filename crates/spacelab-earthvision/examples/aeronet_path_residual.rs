#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]
//! Is the `path_reflectance` residual against DISORT a bias, or the tail of the distribution?
//!
//! A single Monte Carlo draw carries the bias and the noise together, so no amount of staring at one
//! carries the answer. This separates them: it runs every atmosphere in the fixture at many seeds,
//! which measures the estimator's **expectation** with an error bar, and the bias is what survives
//! once the noise has been averaged away.
//!
//! The discriminator is what happens to each of the two as the photon count rises. The standard error
//! falls as `1/√N` by construction. If the disagreement with DISORT is sampling, it falls with it and
//! the `z = bias/SE` column stays flat and small; if it is a systematic, the bias holds still while
//! the error bar shrinks under it and `z` grows as `√N`. Run the ladder rather than one point:
//!
//! ```text
//! cargo run --release -p spacelab-earthvision --example aeronet_path_residual -- 100000  32
//! cargo run --release -p spacelab-earthvision --example aeronet_path_residual -- 400000  32
//! cargo run --release -p spacelab-earthvision --example aeronet_path_residual -- 1600000 32
//! ```
//!
//! The other half of the question is whether the *reference* is converged, and that one belongs to
//! `tools/gen_aeronet_disort_oracle.py`, whose header carries the stream-count sweep.
//!
//! Not a test: it is a measurement whose cost is counted in photon-seconds, and its answers live in
//! `docs/verification.md` §5.3 rather than in an assertion.

use spacelab_earthvision::{Atmosphere, PhaseFunction, path_reflectance};

/// The view cosine and azimuths `tools/gen_aeronet_disort_oracle.py` generated the reference at.
const MU_VIEW: f64 = 0.8;
const AZIMUTHS_DEG: [f64; 3] = [0.0, 90.0, 180.0];

/// The seed the committed test uses; the study's seeds are derived from it so that seed 0 of the
/// study is the number the test reports.
const BASE_SEED: u64 = 20_260_817;

const THREADS: usize = 8;

struct Case {
    tau: f64,
    ssa: f64,
    g: f64,
    solar_zenith_deg: f64,
    reference: [f64; 3],
}

fn data(text: &str) -> impl Iterator<Item = Vec<&str>> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| line.split_whitespace().collect())
}

fn cases() -> Vec<Case> {
    let measured = include_str!("../tests/fixtures/aeronet_atmosphere.txt");
    let reference = include_str!("../tests/fixtures/aeronet_coupling_disort.txt");
    data(measured)
        .zip(data(reference))
        .map(|(m, r)| {
            let number = |f: &Vec<&str>, i: usize| f[i].parse::<f64>().expect("a fixture number");
            assert_eq!(
                (m[0], m[2], m[3], m[5]),
                (r[0], r[1], r[2], r[3]),
                "the two fixtures are not row-aligned"
            );
            Case {
                tau: number(&m, 6),
                ssa: number(&m, 7),
                g: number(&m, 8),
                solar_zenith_deg: number(&m, 4),
                reference: [number(&r, 8), number(&r, 9), number(&r, 10)],
            }
        })
        .collect()
}

/// Mean and standard error over the seeds, at one atmosphere and one azimuth.
///
/// The seed mixes the row and the azimuth as well as the draw. It has to: `path_reflectance` is
/// called once per azimuth, so a seed that depended on the draw alone would give all three azimuths
/// of an atmosphere the identical random walk and every atmosphere in the fixture the identical
/// stream, and the aggregate significance below would be computed over 360 samples of which perhaps
/// 120 are independent.
fn sample(
    case: &Case,
    index: usize,
    azimuth_index: usize,
    photons: u32,
    seeds: usize,
) -> (f64, f64) {
    let atmosphere = Atmosphere {
        optical_depth: case.tau,
        single_scattering_albedo: case.ssa,
        phase: PhaseFunction::HenyeyGreenstein { asymmetry: case.g },
    };
    let mu_sun = case.solar_zenith_deg.to_radians().cos();
    let azimuth = AZIMUTHS_DEG[azimuth_index].to_radians();

    let stream = (index * AZIMUTHS_DEG.len() + azimuth_index) as u64;
    let mut values = Vec::with_capacity(seeds);
    for draw in 0..seeds {
        // SplitMix's increment: distinct draws give uncorrelated xoshiro states through `Rng::new`.
        let seed = BASE_SEED
            .wrapping_add(stream.wrapping_mul(0xff51_afd7_ed55_8ccd))
            .wrapping_add((draw as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        values.push(path_reflectance(
            &atmosphere,
            mu_sun,
            MU_VIEW,
            azimuth,
            photons,
            seed,
        ));
    }
    let mean = values.iter().sum::<f64>() / seeds as f64;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (seeds - 1) as f64;
    (mean, (variance / seeds as f64).sqrt())
}

struct Row {
    tau: f64,
    ssa: f64,
    g: f64,
    azimuth: usize,
    reference: f64,
    mean: f64,
    standard_error: f64,
}

impl Row {
    fn azimuth_deg(&self) -> f64 {
        AZIMUTHS_DEG[self.azimuth]
    }
    fn relative_bias(&self) -> f64 {
        self.mean / self.reference - 1.0
    }
    fn z(&self) -> f64 {
        (self.mean - self.reference) / self.standard_error
    }
}

fn main() {
    let mut argv = std::env::args().skip(1);
    let photons: u32 = argv
        .next()
        .map_or(250_000, |v| v.parse().expect("photons is a number"));
    let seeds: usize = argv
        .next()
        .map_or(24, |v| v.parse().expect("seeds is a number"));
    assert!(seeds >= 2, "a standard error needs at least two seeds");

    let cases = cases();
    let work: Vec<(usize, usize)> = (0..cases.len())
        .flat_map(|case| (0..AZIMUTHS_DEG.len()).map(move |azimuth| (case, azimuth)))
        .collect();

    let started = std::time::Instant::now();
    let mut rows: Vec<Row> = Vec::with_capacity(work.len());
    std::thread::scope(|scope| {
        let cases = &cases;
        let handles: Vec<_> = (0..THREADS)
            .map(|thread| {
                let slice: Vec<(usize, usize)> =
                    work.iter().skip(thread).step_by(THREADS).copied().collect();
                scope.spawn(move || {
                    slice
                        .into_iter()
                        .map(|(index, azimuth)| {
                            let case = &cases[index];
                            let (mean, standard_error) =
                                sample(case, index, azimuth, photons, seeds);
                            Row {
                                tau: case.tau,
                                ssa: case.ssa,
                                g: case.g,
                                azimuth,
                                reference: case.reference[azimuth],
                                mean,
                                standard_error,
                            }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for handle in handles {
            rows.extend(handle.join().expect("a worker thread panicked"));
        }
    });
    rows.sort_by(|a, b| a.tau.total_cmp(&b.tau).then(a.azimuth.cmp(&b.azimuth)));

    println!(
        "# photons {photons}, seeds {seeds}, {} samples, {:.1} s",
        rows.len(),
        started.elapsed().as_secs_f64()
    );
    println!("# tau      ssa     g       phi   reference    mean         rel_bias    z");
    for row in &rows {
        println!(
            "{:8.4} {:.4} {:.4} {:5.0} {:12.6} {:12.6} {:+11.3e} {:+8.2}",
            row.tau,
            row.ssa,
            row.g,
            row.azimuth_deg(),
            row.reference,
            row.mean,
            row.relative_bias(),
            row.z()
        );
    }

    summarise(&rows, photons, seeds);
}

/// Mean relative bias, its worst case and its RMS, over a subset of the comparison points.
fn band(rows: &[&Row], name: &str) {
    if rows.is_empty() {
        return;
    }
    let count = rows.len() as f64;
    let mean = rows.iter().map(|r| r.relative_bias()).sum::<f64>() / count;
    let rms = (rows.iter().map(|r| r.relative_bias().powi(2)).sum::<f64>() / count).sqrt();
    let worst = rows
        .iter()
        .map(|r| r.relative_bias().abs())
        .fold(0.0_f64, f64::max);
    let mean_z = rows.iter().map(|r| r.z()).sum::<f64>() / count;
    println!(
        "{name:<18} n {:3}  mean {mean:+.3e}  rms {rms:.3e}  worst {worst:.3e}  mean z {mean_z:+.2}",
        rows.len()
    );
}

fn summarise(rows: &[Row], photons: u32, seeds: usize) {
    let all: Vec<&Row> = rows.iter().collect();
    let count = rows.len() as f64;
    let worst_z = rows.iter().map(|r| r.z().abs()).fold(0.0_f64, f64::max);
    let beyond_three = rows.iter().filter(|r| r.z().abs() > 3.0).count();
    // Under a pure sampling residual each z is a standard normal, so their mean has standard
    // deviation 1/sqrt(n) and the product below is itself a z-score for "the whole set leans one
    // way". It is only meaningful because `sample` gives every point its own stream.
    let mean_z = rows.iter().map(Row::z).sum::<f64>() / count;

    println!("\n# summary at photons {photons}, seeds {seeds}");
    println!("worst |z|             {worst_z:.2}");
    println!("|z| > 3               {beyond_three} of {}", rows.len());
    println!(
        "mean z                {mean_z:+.3}  (aggregate {:+.2} sigma)",
        mean_z * count.sqrt()
    );
    band(&all, "all");

    for (name, lower, upper) in [
        ("tau < 0.5", 0.0, 0.5),
        ("0.5 <= tau < 1.5", 0.5, 1.5),
        ("tau >= 1.5", 1.5, f64::INFINITY),
    ] {
        band(
            &rows
                .iter()
                .filter(|r| r.tau >= lower && r.tau < upper)
                .collect::<Vec<_>>(),
            name,
        );
    }
    // Resolved by azimuth as well, because that is what #326's noise column was not: it was measured
    // at azimuth 0 alone, where the path reflectance is largest and its *relative* noise smallest,
    // while the residual it was compared against spanned all three.
    for (azimuth, degrees) in AZIMUTHS_DEG.iter().enumerate() {
        band(
            &rows
                .iter()
                .filter(|r| r.azimuth == azimuth)
                .collect::<Vec<_>>(),
            &format!("phi = {degrees:.0} deg"),
        );
    }
}
