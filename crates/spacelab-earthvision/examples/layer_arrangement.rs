#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]
//! Does the vertical *arrangement* of Rayleigh and aerosol matter, or only their sum? (#368)
//!
//! `radiative_transfer::Atmosphere` is one homogeneous layer: a single `optical_depth`, a single
//! `single_scattering_albedo` and a single `PhaseFunction`, with Rayleigh and aerosol mixed into an
//! effective medium. In the real column they are not co-located -- Rayleigh follows the ~8 km
//! pressure scale height, boundary-layer aerosol sits under ~1-2 km -- and no effective medium can
//! represent where a photon meets which scatterer. #357 closed because the geometry it blamed turned
//! out to displace the air mass by under a per cent; this arrangement is what the air-mass
//! coefficient is left pointing at, and unlike sphericity it is not bounded by a small displacement.
//!
//! The stacked walk this measurement needs now ships: `monte_carlo_stack`, `stack_path_reflectance`
//! and `coupling_from_stack`, with `monte_carlo_slab` and `coupling_from_slab` routed through the
//! same kernel on a one-slice stack, so the single-layer reference below is not a second opinion
//! about what ships -- it *is* what ships. That the two agree is asserted in the library rather than
//! argued here: a uniform medium cut twelve ways reproduces the undivided one bit for bit, which is
//! the check that a layer boundary is transparent when there is no contrast across it. What is
//! compared, at the fixture's own optical depths and geometries:
//!
//! * **mixed** -- what ships today,
//! * **Rayleigh over aerosol** -- the physical order, maximally separated, which bounds the effect,
//! * **aerosol over Rayleigh** -- the same separation upside down, so the *sign* is measured rather
//!   than assumed,
//! * **stratified** -- both components on exponential profiles, `H = 8 km` and `H = 1.5 km`, sliced
//!   into layers of equal geometric thickness. This is the arrangement a layered `Atmosphere` would
//!   actually carry, and the one whose difference from `mixed` decides the ticket.
//!
//! Every arrangement holds the total extinction optical depth `tau_r + tau_a` and the total
//! scattering optical depth `tau_r + omega_a tau_a` *exactly* fixed, so the only thing that varies
//! is where the two sit relative to each other.
//!
//! Against the transport's own noise, which is measured on the same rows at three seeds -- a
//! difference below it is a difference this comparison cannot see.
//!
//! ```text
//! cargo run --release -p spacelab-earthvision --example layer_arrangement
//! cargo run --release -p spacelab-earthvision --example layer_arrangement -- 160000
//! ```
//!
//! Not a test: a measurement whose answer belongs in `docs/verification.md` §5.4, and whose job is
//! to say whether a Type-1 change to a public type is justified.

use spacelab_earthvision::{
    Atmosphere, AtmosphericCoupling, Layer, PhaseFunction, TabulatedPhase, coupling_from_slab,
    coupling_from_stack,
};

/// The seed the committed test uses, so this speaks about the same draws the section reports.
const SEED: u64 = 20_260_819;

/// RadCalNet simulates nadir view only.
const MU_VIEW: f64 = 1.0;

/// Quoted from RadCalNet's MODTRAN deck, Card 3A2 `G: 0.500`.
const AEROSOL_ASYMMETRY: f64 = 0.5;

/// The fitted aerosol single-scattering albedo the suite runs at.
const AEROSOL_SSA: f64 = 0.88;

const PHASE_NODES: usize = 401;

/// The nine gas windows the suite compares on.
const WINDOW_NM: [f64; 9] = [
    410.0, 440.0, 490.0, 550.0, 600.0, 660.0, 680.0, 870.0, 1030.0,
];

/// Molecular scale height: the pressure scale height, because Rayleigh scattering is proportional to
/// air density and the atmosphere is hydrostatic.
const RAYLEIGH_SCALE_KM: f64 = 8.0;

/// Boundary-layer aerosol scale height. The 1-2 km the ticket names; the ends are reported too,
/// because a conclusion that holds only at one of them is a conclusion about the choice.
const AEROSOL_SCALE_KM: f64 = 1.5;

/// Where the stratified profile is truncated. At 30 km the Rayleigh profile has 2.3 per mil left and
/// the aerosol 1e-9, so the truncation is far below anything else measured here.
const TOP_KM: f64 = 30.0;

/// Slices of equal geometric thickness in the stratified arrangement.
const STRATIFIED_LAYERS: usize = 24;

// ---------------------------------------------------------------------------------------------
// The fixture.
// ---------------------------------------------------------------------------------------------

struct Row {
    site: String,
    solar_zenith_deg: f64,
    wavelength_nm: f64,
    boa: f64,
    /// RadCalNet's own published TOA reflectance -- the oracle the residual is measured against.
    toa: f64,
    aod550: f64,
    angstrom: f64,
    tau_rayleigh: f64,
    ozone_du: f64,
}

impl Row {
    fn mu_sun(&self) -> f64 {
        libm::cos(self.solar_zenith_deg.to_radians())
    }

    fn air_mass(&self) -> f64 {
        1.0 / self.mu_sun() + 1.0 / MU_VIEW
    }

    fn tau_aerosol(&self) -> f64 {
        self.aod550 * libm::pow(self.wavelength_nm / 550.0, -self.angstrom)
    }
}

fn rows() -> Vec<Row> {
    include_str!("../tests/fixtures/radcalnet_toa.txt")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let parse = |index: usize| f[index].parse::<f64>().expect("a fixture number");
            let wavelength_nm = parse(4);
            (WINDOW_NM.contains(&wavelength_nm) && f[20] == "ok").then(|| Row {
                site: f[0].to_owned(),
                solar_zenith_deg: parse(3),
                wavelength_nm,
                boa: parse(5),
                toa: parse(7),
                aod550: parse(9),
                angstrom: parse(10),
                tau_rayleigh: parse(11),
                ozone_du: parse(14),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The arrangements. All four hold `tau_r + tau_a` and `tau_r + omega_a tau_a` exactly fixed.
// ---------------------------------------------------------------------------------------------

/// The Rayleigh/aerosol mixture on the node grid, unnormalised -- weighted by *scattering* depth,
/// which is what makes the mixed layer's phase function the one the stacked layers average to.
fn phase_mixture(
    tau_rayleigh: f64,
    tau_aerosol: f64,
    aerosol_ssa: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let rayleigh = PhaseFunction::Rayleigh;
    let aerosol = PhaseFunction::HenyeyGreenstein {
        asymmetry: AEROSOL_ASYMMETRY,
    };
    let scale = 2.0 / (PHASE_NODES - 1) as f64;
    let mu: Vec<f64> = (0..PHASE_NODES).map(|i| -1.0 + scale * i as f64).collect();
    let mut value: Vec<f64> = mu
        .iter()
        .map(|&m| tau_rayleigh * rayleigh.value(m) + tau_aerosol * aerosol_ssa * aerosol.value(m))
        .collect();
    let mut cumulative = vec![0.0; mu.len()];
    TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).expect("a phase mixture");
    (mu, value, cumulative)
}

/// What ships today: one layer, both components mixed into an effective medium.
fn mixed_grids(tau_r: f64, tau_a: f64, aerosol_ssa: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    phase_mixture(tau_r, tau_a, aerosol_ssa)
}

fn mixed_atmosphere(
    tau_r: f64,
    tau_a: f64,
    aerosol_ssa: f64,
    grids: &(Vec<f64>, Vec<f64>, Vec<f64>),
) -> Atmosphere<'_> {
    let table = TabulatedPhase::new(&grids.0, &grids.1, &grids.2).expect("a phase mixture");
    Atmosphere {
        optical_depth: tau_r + tau_a,
        single_scattering_albedo: (tau_r + tau_a * aerosol_ssa) / (tau_r + tau_a),
        phase: PhaseFunction::Tabulated(table),
    }
}

/// The two components maximally separated, in the order given. Pure phase functions per layer, so
/// nothing is tabulated and the arrangement is the only difference from `mixed`.
fn separated<'a>(
    tau_top: f64,
    omega_top: f64,
    tau_bottom: f64,
    omega_bottom: f64,
    top_is_rayleigh: bool,
) -> Vec<Layer<'a>> {
    let rayleigh = PhaseFunction::Rayleigh;
    let aerosol = PhaseFunction::HenyeyGreenstein {
        asymmetry: AEROSOL_ASYMMETRY,
    };
    let (phase_top, phase_bottom) = if top_is_rayleigh {
        (rayleigh, aerosol)
    } else {
        (aerosol, rayleigh)
    };
    vec![
        Layer {
            thickness: tau_top,
            single_scattering_albedo: omega_top,
            phase: phase_top,
        },
        Layer {
            thickness: tau_bottom,
            single_scattering_albedo: omega_bottom,
            phase: phase_bottom,
        },
    ]
}

/// The two components on exponential profiles, sliced into layers of equal geometric thickness.
///
/// Slice `i` gets the fraction of each component's column that lies between its edges, so the sum
/// over slices is the column exactly and no normalisation constant is left to be trusted.
fn stratified_depths(tau_r: f64, tau_a: f64, aerosol_scale_km: f64) -> Vec<(f64, f64)> {
    let dz = TOP_KM / STRATIFIED_LAYERS as f64;
    let share = |scale: f64, z0: f64, z1: f64| {
        (libm::exp(-z0 / scale) - libm::exp(-z1 / scale)) / (1.0 - libm::exp(-TOP_KM / scale))
    };
    (0..STRATIFIED_LAYERS)
        .rev() // top slice first: the walk's tz runs downward
        .map(|i| {
            let (z0, z1) = (i as f64 * dz, (i + 1) as f64 * dz);
            (
                tau_r * share(RAYLEIGH_SCALE_KM, z0, z1),
                tau_a * share(aerosol_scale_km, z0, z1),
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The measurement. Paired: every arrangement runs on the *same* seed as the mixed reference it is
// differenced against, so the Monte-Carlo noise the two share cancels and what survives is the
// arrangement. The noise that remains is then measured directly -- how much the paired difference
// itself moves between seeds -- rather than substituted for by the noise of an unpaired run, which
// is the larger and wrong number.
// ---------------------------------------------------------------------------------------------

/// How many independent streams each paired difference is averaged over.
const DRAWS: u64 = 3;

#[derive(Clone, Copy)]
struct Sample {
    rho_toa: f64,
    path: f64,
    two_way: f64,
}

fn sample(coupling: &AtmosphericCoupling, boa: f64) -> Sample {
    Sample {
        rho_toa: coupling.toa_from_surface(boa),
        path: coupling.path_reflectance,
        two_way: coupling.transmittance_down * coupling.transmittance_up,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Arrangement {
    /// One layer, both components mixed into an effective medium -- what ships today.
    Mixed,
    /// The two maximally separated, Rayleigh on top or aerosol on top.
    Separated { rayleigh_on_top: bool },
    /// Both on exponential profiles, sliced by equal geometric thickness.
    Stratified { aerosol_scale_km: f64 },
}

/// The coupling this row's atmosphere produces under one arrangement, at one stream.
///
/// `Mixed` goes through `coupling_from_slab` rather than through a one-slice stack. They are the
/// same numbers -- `coupling_from_slab` delegates, and `a_uniform_stack_reproduces_the_undivided_coupling`
/// holds the twelve-slice case to the bit -- and calling the single-layer entry point is what makes
/// every difference below a difference from what actually ships.
fn couple(
    row: &Row,
    arrangement: Arrangement,
    aerosol_ssa: f64,
    photons: u32,
    seed: u64,
) -> AtmosphericCoupling {
    let (tau_r, tau_a) = (row.tau_rayleigh, row.tau_aerosol());
    match arrangement {
        Arrangement::Mixed => {
            let grids = mixed_grids(tau_r, tau_a, aerosol_ssa);
            let atmosphere = mixed_atmosphere(tau_r, tau_a, aerosol_ssa, &grids);
            coupling_from_slab(&atmosphere, row.mu_sun(), MU_VIEW, 0.0, photons, seed)
        }
        Arrangement::Separated { rayleigh_on_top } => {
            let layers = if rayleigh_on_top {
                separated(tau_r, 1.0, tau_a, aerosol_ssa, true)
            } else {
                separated(tau_a, aerosol_ssa, tau_r, 1.0, false)
            };
            coupling_from_stack(&layers, row.mu_sun(), MU_VIEW, 0.0, photons, seed)
        }
        Arrangement::Stratified { aerosol_scale_km } => {
            let depths = stratified_depths(tau_r, tau_a, aerosol_scale_km);
            let grids: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = depths
                .iter()
                .map(|&(r, a)| phase_mixture(r, a, aerosol_ssa))
                .collect();
            let layers: Vec<Layer> = depths
                .iter()
                .zip(grids.iter())
                .map(|(&(r, a), g)| Layer {
                    thickness: r + a,
                    single_scattering_albedo: (r + a * aerosol_ssa) / (r + a),
                    phase: PhaseFunction::Tabulated(
                        TabulatedPhase::new(&g.0, &g.1, &g.2).expect("a phase mixture"),
                    ),
                })
                .collect();
            coupling_from_stack(&layers, row.mu_sun(), MU_VIEW, 0.0, photons, seed)
        }
    }
}

/// One row's paired difference against `Mixed`, averaged over the streams, with the scatter across
/// those streams kept -- the estimate and its own error bar, per row.
#[derive(Clone, Copy)]
struct Paired {
    rho_toa: f64,
    rho_toa_error: f64,
    path: f64,
    two_way: f64,
    air_mass: f64,
}

fn paired(
    row: &Row,
    reference: &[Sample],
    arrangement: Arrangement,
    photons: u32,
    index: usize,
) -> Paired {
    let mut draws: Vec<Sample> = Vec::new();
    for (draw, base) in reference.iter().enumerate() {
        let seed = stream(index, draw as u64);
        let moved = sample(
            &couple(row, arrangement, AEROSOL_SSA, photons, seed),
            row.boa,
        );
        draws.push(Sample {
            rho_toa: moved.rho_toa - base.rho_toa,
            path: moved.path - base.path,
            two_way: moved.two_way - base.two_way,
        });
    }
    let n = DRAWS as f64;
    let mean = |f: fn(&Sample) -> f64| draws.iter().map(f).sum::<f64>() / n;
    let rho_toa = mean(|s| s.rho_toa);
    // Standard error of the mean across streams. With three draws it is a coarse estimate of a
    // coarse quantity, which is the honest size of the bar and the reason it is reported per row
    // and then pooled rather than quoted for any single row.
    let variance = draws
        .iter()
        .map(|s| (s.rho_toa - rho_toa) * (s.rho_toa - rho_toa))
        .sum::<f64>()
        / (n - 1.0);
    Paired {
        rho_toa,
        rho_toa_error: libm::sqrt(variance / n),
        path: mean(|s| s.path),
        two_way: mean(|s| s.two_way),
        air_mass: row.air_mass(),
    }
}

/// The pooled verdict for one arrangement.
struct Verdict {
    n: usize,
    rms: f64,
    worst: f64,
    mean_signed: f64,
    /// Pooled Monte-Carlo bar on the same quantity: the rms of the per-row standard errors. A
    /// signal below this is one the comparison cannot resolve, whatever its sign.
    noise: f64,
    slope: f64,
    slope_error: f64,
}

fn verdict(rows: &[Paired]) -> Verdict {
    let n = rows.len();
    let count = n as f64;
    let mean_signed = rows.iter().map(|p| p.rho_toa).sum::<f64>() / count;
    let rms = libm::sqrt(rows.iter().map(|p| p.rho_toa * p.rho_toa).sum::<f64>() / count);
    let worst = rows
        .iter()
        .fold(0.0_f64, |a, p| a.max(libm::fabs(p.rho_toa)));
    let noise = libm::sqrt(
        rows.iter()
            .map(|p| p.rho_toa_error * p.rho_toa_error)
            .sum::<f64>()
            / count,
    );
    let mean_m = rows.iter().map(|p| p.air_mass).sum::<f64>() / count;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for p in rows {
        sxy += (p.air_mass - mean_m) * (p.rho_toa - mean_signed);
        sxx += (p.air_mass - mean_m) * (p.air_mass - mean_m);
    }
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let residual = rows
        .iter()
        .map(|p| {
            let e = p.rho_toa - mean_signed - slope * (p.air_mass - mean_m);
            e * e
        })
        .sum::<f64>()
        / (count - 2.0).max(1.0);
    Verdict {
        n,
        rms,
        worst,
        mean_signed,
        noise,
        slope,
        slope_error: if sxx > 0.0 {
            libm::sqrt(residual / sxx)
        } else {
            0.0
        },
    }
}

fn name(arrangement: Arrangement) -> String {
    match arrangement {
        Arrangement::Mixed => "mixed".to_owned(),
        Arrangement::Separated {
            rayleigh_on_top: true,
        } => "Rayleigh over aerosol".to_owned(),
        Arrangement::Separated {
            rayleigh_on_top: false,
        } => "aerosol over Rayleigh".to_owned(),
        Arrangement::Stratified { aerosol_scale_km } => {
            format!("stratified, H_a = {aerosol_scale_km:.1} km")
        }
    }
}

fn t_ratio(slope: f64, error: f64) -> f64 {
    if error > 0.0 {
        libm::fabs(slope / error)
    } else {
        0.0
    }
}

/// What the arrangement does to TOA reflectance, against the noise on the same estimate.
fn report_reflectance(measured: &[(Arrangement, Vec<Paired>)]) {
    println!(
        "\n# what the arrangement does to TOA reflectance, differenced against mixed at the same seed"
    );
    println!(
        "\n{:>24} {:>6} {:>11} {:>11} {:>12} {:>11} {:>7}",
        "arrangement", "n", "rms", "worst", "mean signed", "MC noise", "signal"
    );
    for (a, rows) in measured {
        let v = verdict(rows);
        println!(
            "{:>24} {:>6} {:>11.4e} {:>11.4e} {:>+12.4e} {:>11.4e} {:>6.1}x",
            name(*a),
            v.n,
            v.rms,
            v.worst,
            v.mean_signed,
            v.noise,
            v.rms / v.noise
        );
    }
    println!(
        "# the last column is rms / MC noise -- under 1 is a difference this comparison cannot see"
    );
}

/// The same thing in the two terms the 6S coupling is built from, which move in opposite directions
/// and partly cancel in the total -- so a total that looks small can still be two large terms.
fn report_terms(measured: &[(Arrangement, Vec<Paired>)]) {
    println!("\n# and to the two terms the coupling is built from");
    println!(
        "\n{:>24} {:>13} {:>11} {:>13} {:>11}",
        "arrangement", "rms d_rho_atm", "worst", "rms d_T*T", "worst"
    );
    for (a, rows) in measured {
        let pooled = |f: fn(&Paired) -> f64| {
            let count = rows.len() as f64;
            (
                libm::sqrt(rows.iter().map(|p| f(p) * f(p)).sum::<f64>() / count),
                rows.iter().fold(0.0_f64, |m, p| m.max(libm::fabs(f(p)))),
            )
        };
        let (path_rms, path_worst) = pooled(|p| p.path);
        let (tt_rms, tt_worst) = pooled(|p| p.two_way);
        println!(
            "{:>24} {:>13.4e} {:>11.4e} {:>13.4e} {:>11.4e}",
            name(*a),
            path_rms,
            path_worst,
            tt_rms,
            tt_worst
        );
    }
}

/// The air-mass slope, which is the whole point of the ticket.
fn report_slopes(measured: &[(Arrangement, Vec<Paired>)]) {
    println!(
        "\n# the air-mass slope, which is the coefficient #368 exists to move. The section reports\n\
         # +0.00710 +/- 0.00073 of *relative* residual per unit; at mean rho_TOA 0.2306 that is\n\
         # 1.64e-3 of TOA reflectance per unit air mass, and an arrangement that explains it has to\n\
         # carry a slope of that size with the sign that cancels it"
    );
    println!(
        "\n{:>24} {:>14} {:>12} {:>7} {:>12}",
        "arrangement", "slope /air mass", "std err", "|t|", "of the 1.64e-3"
    );
    for (a, rows) in measured {
        let v = verdict(rows);
        println!(
            "{:>24} {:>+14.4e} {:>12.4e} {:>7.2} {:>11.2} %",
            name(*a),
            v.slope,
            v.slope_error,
            t_ratio(v.slope, v.slope_error),
            100.0 * v.slope / 1.64e-3
        );
    }
}

/// The arrangement a layered `Atmosphere` would actually carry, per site: the sites differ in
/// altitude and in how much aerosol they hold, so a real arrangement effect has to vary across them.
fn report_sites(rows: &[Row], shipped: &[Paired]) {
    println!(
        "\n# stratified, H_a = {AEROSOL_SCALE_KM} km, per site -- the sites differ in altitude and in\n\
         # how much aerosol they carry, so a real arrangement effect has to vary across them"
    );
    println!(
        "\n{:>10} {:>6} {:>11} {:>12} {:>11} {:>14} {:>7}",
        "site", "n", "rms", "mean signed", "MC noise", "slope /air mass", "|t|"
    );
    let mut sites: Vec<&str> = rows.iter().map(|r| r.site.as_str()).collect();
    sites.sort_unstable();
    sites.dedup();
    for site in sites {
        let subset: Vec<Paired> = rows
            .iter()
            .zip(shipped.iter())
            .filter(|(row, _)| row.site == site)
            .map(|(_, p)| *p)
            .collect();
        let v = verdict(&subset);
        println!(
            "{:>10} {:>6} {:>11.4e} {:>+12.4e} {:>11.4e} {:>+14.4e} {:>7.2}",
            site,
            v.n,
            v.rms,
            v.mean_signed,
            v.noise,
            v.slope,
            t_ratio(v.slope, v.slope_error)
        );
    }
}

/// Ozone cross-section per wavelength, cm^2 per molecule, at RadCalNet's own published grid.
///
/// The same fixture and the same column `radcalnet_measured_radiance.rs` reads, because a residual
/// against RadCalNet's TOA that used a different ozone would be measuring the ozone.
fn ozone_table() -> Vec<(f64, f64)> {
    include_str!("../tests/fixtures/ozone_cross_section.txt")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            (
                f[0].parse::<f64>().expect("a wavelength"),
                f[1].parse::<f64>().expect("a cross-section"),
            )
        })
        .collect()
}

/// Molecules per cm^2 in one Dobson unit.
const DOBSON_PER_CM2: f64 = 2.686_780_111e25 * 1.0e-5 * 1.0e-4;

/// Two-way ozone transmittance above the scattering layer. Ozone sits at ~25 km, above everything
/// the stack rearranges, so this is a multiplicative factor on every arrangement alike -- and it has
/// to be here anyway, because without it the residual against RadCalNet is dominated by the missing
/// Chappuis absorption rather than by the arrangement.
fn ozone_transmittance(row: &Row, table: &[(f64, f64)]) -> f64 {
    let sigma = table
        .iter()
        .find(|(w, _)| libm::fabs(w - row.wavelength_nm) < 0.5)
        .map(|&(_, sigma)| sigma)
        .expect("an ozone cross-section at every compared wavelength");
    libm::exp(-sigma * row.ozone_du * DOBSON_PER_CM2 * row.air_mass())
}

/// The aerosol single-scattering albedos the residual is scanned over.
///
/// 0.88 is the shipped value, fitted under the mixed layer. Handing a second arrangement another
/// arrangement's optimum and reporting that it does worse would be a rigged comparison, so each is
/// given its own best value on this grid and both curves are printed rather than their minima.
const SSA_SCAN: [f64; 5] = [0.84, 0.86, 0.88, 0.90, 0.92];

/// rms and mean signed forward residual against RadCalNet's published TOA, over the gated rows.
fn residual(rows: &[Row], arrangement: Arrangement, aerosol_ssa: f64, photons: u32) -> (f64, f64) {
    let ozone = ozone_table();
    let mut sum = 0.0;
    let mut square = 0.0;
    for (index, row) in rows.iter().enumerate() {
        let modelled: f64 = (0..DRAWS)
            .map(|draw| {
                ozone_transmittance(row, &ozone)
                    * couple(row, arrangement, aerosol_ssa, photons, stream(index, draw))
                        .toa_from_surface(row.boa)
            })
            .sum::<f64>()
            / DRAWS as f64;
        let error = modelled - row.toa;
        sum += error;
        square += error * error;
    }
    let n = rows.len() as f64;
    (libm::sqrt(square / n), sum / n)
}

/// Does the arrangement move the comparison *towards* RadCalNet, or only away from `mixed`?
///
/// Everything above is a paired difference from the mixed layer, and a paired difference has no
/// oracle in it: it says the arrangement is real and says nothing about which arrangement is right.
/// This is the question the ticket has to answer before a default changes -- the same rows, the same
/// ozone, against RadCalNet's own published TOA.
fn report_residual(rows: &[Row], photons: u32) {
    println!(
        "\n# forward residual against RadCalNet's published TOA, over the same {} rows",
        rows.len()
    );
    println!(
        "\n{:>24} {:>9} {:>12} {:>13}",
        "arrangement", "omega_a", "rms", "mean signed"
    );
    for arrangement in [
        Arrangement::Mixed,
        Arrangement::Stratified {
            aerosol_scale_km: AEROSOL_SCALE_KM,
        },
    ] {
        for ssa in SSA_SCAN {
            let (rms, bias) = residual(rows, arrangement, ssa, photons);
            println!(
                "{:>24} {ssa:>9.2} {rms:>12.4e} {bias:>+13.4e}",
                name(arrangement)
            );
        }
    }
    println!(
        "# a lower rms at the arrangement's own best omega_a is what would justify a default change"
    );
}

fn main() {
    let photons: u32 = std::env::args()
        .nth(1)
        .map_or(40_000, |a| a.parse().expect("a photon count"));

    let rows = rows();
    println!(
        "# {} gated comparisons, {} photons, {} paired draws, Rayleigh H = {} km, {} slices",
        rows.len(),
        photons,
        DRAWS,
        RAYLEIGH_SCALE_KM,
        STRATIFIED_LAYERS
    );

    let arrangements = [
        Arrangement::Separated {
            rayleigh_on_top: true,
        },
        Arrangement::Separated {
            rayleigh_on_top: false,
        },
        Arrangement::Stratified {
            aerosol_scale_km: 1.0,
        },
        Arrangement::Stratified {
            aerosol_scale_km: AEROSOL_SCALE_KM,
        },
        Arrangement::Stratified {
            aerosol_scale_km: 2.0,
        },
    ];

    // The mixed reference, once per row per stream. Every arrangement is differenced against it,
    // so computing it inside the arrangement loop would repeat the whole shipped transport five
    // times over for numbers that cannot differ.
    let reference: Vec<Vec<Sample>> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            (0..DRAWS)
                .map(|draw| {
                    sample(
                        &couple(
                            row,
                            Arrangement::Mixed,
                            AEROSOL_SSA,
                            photons,
                            stream(index, draw),
                        ),
                        row.boa,
                    )
                })
                .collect()
        })
        .collect();

    let measured: Vec<(Arrangement, Vec<Paired>)> = arrangements
        .iter()
        .map(|&a| {
            (
                a,
                rows.iter()
                    .enumerate()
                    .map(|(index, row)| paired(row, &reference[index], a, photons, index))
                    .collect(),
            )
        })
        .collect();

    report_reflectance(&measured);
    report_terms(&measured);
    report_slopes(&measured);

    let shipped = measured
        .iter()
        .find(|(a, _)| {
            *a == Arrangement::Stratified {
                aerosol_scale_km: AEROSOL_SCALE_KM,
            }
        })
        .expect("the stratified arrangement");
    report_sites(&rows, &shipped.1);
    report_residual(&rows, photons);
}

/// A seed per row, per draw -- the suite's own, so the reference draw is the draw it reports.
fn stream(row: usize, draw: u64) -> u64 {
    SEED.wrapping_add((row as u64).wrapping_mul(0xff51_afd7_ed55_8ccd))
        .wrapping_add(draw.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}
