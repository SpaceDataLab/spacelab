#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::needless_range_loop
)]
//! The one parameter RadCalNet does not publish, scanned in full, and the budget it lives inside.
//!
//! `tests/radcalnet_measured_radiance.rs` compares our Monte-Carlo slab against RadCalNet's MODTRAN
//! on 42 measured scenes. Every input to that comparison is measured except one: RadCalNet's deck
//! says `NSSALB: 0`, which hands the aerosol single-scattering albedo to MODTRAN's internal rural
//! model and publishes nothing about it. The test constant `AEROSOL_SSA` is therefore **fitted**, and
//! the test can only afford a three-point bracket -- enough to say the minimum is interior, not
//! enough to say how sharp it is, whether one value serves all six instruments, or how it trades
//! against the rest of the error budget. Those answers cost a hundred times the photons, so they
//! live here, off the suite's clock.
//!
//! Seven things are measured:
//!
//! 1. **The scan over the whole gated fixture.** Its depth -- how far the residual climbs between the
//!    minimum and the ends of a plausible albedo range -- is what says whether the data determine the
//!    parameter at all. A flat scan would mean the fit is unidentified and the fitted value is
//!    decoration; a deep one means it is measured.
//! 2. **The same scan per instrument.** One value serving six targets, from a 1 km playa to a 30 m
//!    gravel disk to a site 2824 m up, is a claim. Six per-site minima that agree is the measurement
//!    behind it. Six that disagree would mean the parameter is absorbing something that varies by
//!    site, and the fit would be laundering a systematic into a constant.
//! 3. **The same scan on the hold-out**, `radcalnet_holdout.txt` -- 120 days sampled by date through
//!    each site's archive, disjoint from the 42 the value was fitted on. The suite asks only whether
//!    the in-sample value predicts inside the bar there; this asks what refitting on those days would
//!    have bought, which is the part of the out-of-sample residual the albedo is responsible for.
//! 4. **What the residual depends on**, over every comparison in the workspace pooled: the relative
//!    residual regressed on aerosol optical depth, air mass, columnar water, surface reflectance and
//!    the path-reflectance ratio, standardised so the five compare directly. The section's per-site
//!    table is a grouping; this is a diagnosis, and it names which approximation to fix next.
//! 5. **Whether the reference drifts.** Railroad Valley publishes from 2013, and every residual in
//!    the section assumes RadCalNet is stationary across that record. Regressed per site on epoch,
//!    because a recalibration or a processing change moves the published reflectance with the
//!    atmosphere doing nothing -- and that would be a property of the oracle, not of our slab. Twice:
//!    on epoch alone, and with (4)'s drivers in the design, because a sampler that fixes the calendar
//!    does not fix the weather, and the univariate slope cannot tell the two apart.
//! 6. **What water would cost if the model had a channel for it.** RadCalNet publishes an
//!    uncertainty for six inputs; four of them the budget below charges us for, and water is not one,
//!    because a slab with no gaseous absorption responds to the water column by exactly zero. That
//!    zero describes the model rather than the scene, so the water slope from (4) is multiplied by
//!    the published uncertainty instead -- an upper bound on the missing term, not a budget entry.
//! 7. **The error budget at the fitted point, per input.** RadCalNet's published TOA uncertainty
//!    against this model's response to the published uncertainty of the four measured inputs it
//!    reads, itemised, so it is visible which measurement is worth improving. The test reports the
//!    quadrature sum of the same four terms; this prints them separately.
//!
//! ```text
//! cargo run --release -p spacelab-earthvision --example radcalnet_error_budget
//! cargo run --release -p spacelab-earthvision --example radcalnet_error_budget -- 160000 21 9
//! ```
//!
//! The arguments are photons, scan points and seeds. Photons buy the scan's resolution; seeds buy the
//! budget's, because the budget is a finite difference and its noise is what the averaging beats
//! down. At the default 40 000 / 11 / 3 this takes minutes; at 160 000 / 21 / 9, an hour.
//!
//! Not a test: it is a measurement whose cost is counted in photon-seconds, and its answers live in
//! `docs/verification.md` §5.4 rather than in an assertion.

use spacelab_earthvision::{
    Atmosphere, AtmosphericCoupling, PhaseFunction, TabulatedPhase, coupling_from_slab,
};

/// The seed the committed test uses, so the scan's own draw is the number the test reports.
const SEED: u64 = 20_260_819;

/// RadCalNet simulates nadir view only, so there is no view-angle term and no BRDF.
const MU_VIEW: f64 = 1.0;

/// Quoted from RadCalNet's MODTRAN deck, Card 3A2 `G: 0.500`. Measured; not fitted.
const AEROSOL_ASYMMETRY: f64 = 0.5;
const PHASE_NODES: usize = 401;
const DOBSON_PER_CM2: f64 = 2.686_780_111e25 * 1.0e-5 * 1.0e-4;

/// The nine gas windows the test compares on; the scan speaks about the same comparison it does.
const WINDOW_NM: [f64; 9] = [
    410.0, 440.0, 490.0, 550.0, 600.0, 660.0, 680.0, 870.0, 1030.0,
];

/// The scan's span. Wide enough to bracket every published desert-aerosol albedo -- pure scattering
/// at the top, strongly absorbing dust at the bottom -- so the minimum has room to be interior.
const SCAN_LOW: f64 = 0.80;
const SCAN_HIGH: f64 = 1.00;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Input {
    Aod550,
    Angstrom,
    OzoneColumn,
    Pressure,
}

const INPUTS: [(Input, &str); 4] = [
    (Input::Aod550, "aod550"),
    (Input::Angstrom, "angstrom"),
    (Input::OzoneColumn, "ozone"),
    (Input::Pressure, "pressure"),
];

#[derive(Clone)]
struct Row {
    site: String,
    date: String,
    solar_zenith_deg: f64,
    wavelength_nm: f64,
    boa: f64,
    toa: f64,
    toa_unc: f64,
    aod550: f64,
    angstrom: f64,
    tau_rayleigh: f64,
    pressure_hpa: f64,
    water_cm: f64,
    ozone_du: f64,
    aod550_unc: f64,
    angstrom_unc: f64,
    ozone_du_unc: f64,
    pressure_hpa_unc: f64,
    water_cm_unc: f64,
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

    /// Total scattering depth at this wavelength: the pressure-scaled Rayleigh part plus the aerosol.
    fn tau_scattering(&self) -> f64 {
        self.tau_rayleigh + self.tau_aerosol()
    }

    /// The path-reflectance term relative to the signal, `tau_scat / (4 mu_s rho_BOA)`.
    ///
    /// The same ratio the window classification had to bound. It is here as a *driver* rather than as
    /// a bound: over a dark target at a long slant path the light that never reached the ground is
    /// most of what the sensor sees, and if our residual climbs in anything, this is the candidate.
    fn path_ratio(&self) -> f64 {
        self.tau_scattering() / (4.0 * self.mu_sun() * self.boa)
    }

    /// The scene's day of year, which the annual harmonic in the drift regression is built on.
    fn day_of_year(&self) -> f64 {
        let (_, doy) = self.date.split_once('-').expect("a YYYY-DDD date");
        doy.parse::<f64>().expect("a day")
    }

    /// The scene's date as a decimal year, for the epoch regression.
    ///
    /// `YYYY-DDD`, RadCalNet's own file naming. The Julian-year divisor is deliberate: a day-of-year
    /// over 365.25 is an approximation worth about seven hours at the end of a leap year, which is
    /// four orders of magnitude below the multi-year spans this is regressed over.
    fn epoch_year(&self) -> f64 {
        let (year, _) = self.date.split_once('-').expect("a YYYY-DDD date");
        year.parse::<f64>().expect("a year") + self.day_of_year() / 365.25
    }

    /// This row with one measured input moved to the edge of its own published uncertainty. Pressure
    /// moves through the Rayleigh depth the generator already scaled by it, so the perturbation is
    /// the ratio -- recomputing Bodhaine here would measure a second Rayleigh implementation.
    fn perturbed(&self, input: Input) -> Self {
        let mut moved = self.clone();
        match input {
            Input::Aod550 => moved.aod550 += self.aod550_unc,
            Input::Angstrom => moved.angstrom += self.angstrom_unc,
            Input::OzoneColumn => moved.ozone_du += self.ozone_du_unc,
            Input::Pressure => {
                moved.pressure_hpa += self.pressure_hpa_unc;
                moved.tau_rayleigh *= moved.pressure_hpa / self.pressure_hpa;
            }
        }
        moved
    }
}

fn data(text: &str) -> impl Iterator<Item = &str> {
    text.lines()
        .filter(|line| !line.starts_with('#') && !line.trim().is_empty())
}

/// The gated rows only: a gas window, and an uncertainty RadCalNet itself considers in range.
fn rows() -> Vec<Row> {
    parse_rows(include_str!("../tests/fixtures/radcalnet_toa.txt"))
}

/// The hold-out, on the same gate: days sampled by date through each site's archive.
fn holdout() -> Vec<Row> {
    parse_rows(include_str!("../tests/fixtures/radcalnet_holdout.txt"))
}

fn parse_rows(text: &str) -> Vec<Row> {
    data(text)
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let parse = |index: usize| f[index].parse::<f64>().expect("a fixture number");
            let wavelength_nm = parse(4);
            // The generator drops a column whose published atmosphere carries a negative
            // quantity, because a negative ozone column becomes a transmittance above one. This
            // fixture is generated, so the check belongs here too: a regeneration that lost the
            // rule would otherwise reintroduce it silently.
            assert!(
                parse(9) >= 0.0 && parse(12) > 0.0 && parse(13) >= 0.0 && parse(14) > 0.0,
                "{} {}: a negative published atmospheric quantity reached the fixture",
                f[0],
                f[1]
            );
            (WINDOW_NM.contains(&wavelength_nm) && f[20] == "ok").then(|| Row {
                site: f[0].to_owned(),
                date: f[1].to_owned(),
                solar_zenith_deg: parse(3),
                wavelength_nm,
                boa: parse(5),
                toa: parse(7),
                toa_unc: parse(8),
                aod550: parse(9),
                angstrom: parse(10),
                tau_rayleigh: parse(11),
                pressure_hpa: parse(12),
                water_cm: parse(13),
                ozone_du: parse(14),
                aod550_unc: parse(15),
                angstrom_unc: parse(16),
                ozone_du_unc: parse(17),
                pressure_hpa_unc: parse(18),
                water_cm_unc: parse(19),
            })
        })
        .collect()
}

fn ozone_cross_section() -> Vec<(f64, f64)> {
    let text = include_str!("../tests/fixtures/ozone_cross_section.txt");
    data(text)
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            (
                f[0].parse::<f64>().expect("a wavelength"),
                f[1].parse::<f64>().expect("a cross-section"),
            )
        })
        .collect()
}

fn sigma_for(table: &[(f64, f64)], row: &Row) -> f64 {
    table
        .iter()
        .find(|(w, _)| (w - row.wavelength_nm).abs() < 0.5)
        .map(|(_, sigma)| *sigma)
        .expect("every fixture wavelength has a cross-section")
}

fn coupling(row: &Row, aerosol_ssa: f64, seed: u64, photons: u32) -> AtmosphericCoupling {
    let (tau_r, tau_a) = (row.tau_rayleigh, row.tau_aerosol());
    let rayleigh = PhaseFunction::Rayleigh;
    let aerosol = PhaseFunction::HenyeyGreenstein {
        asymmetry: AEROSOL_ASYMMETRY,
    };
    let scale = 2.0 / (PHASE_NODES - 1) as f64;
    let mu: Vec<f64> = (0..PHASE_NODES).map(|i| -1.0 + scale * i as f64).collect();
    let mut value: Vec<f64> = mu
        .iter()
        .map(|&m| tau_r * rayleigh.value(m) + tau_a * aerosol_ssa * aerosol.value(m))
        .collect();
    let mut cumulative = vec![0.0; mu.len()];
    TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).expect("a phase mixture");
    let table = TabulatedPhase::new(&mu, &value, &cumulative).expect("a phase mixture");
    let atmosphere = Atmosphere {
        optical_depth: tau_r + tau_a,
        single_scattering_albedo: (tau_r + tau_a * aerosol_ssa) / (tau_r + tau_a),
        phase: PhaseFunction::Tabulated(table),
    };
    coupling_from_slab(&atmosphere, row.mu_sun(), MU_VIEW, 0.0, photons, seed)
}

fn ozone_transmittance(row: &Row, sigma_cm2: f64) -> f64 {
    libm::exp(-sigma_cm2 * row.ozone_du * DOBSON_PER_CM2 * row.air_mass())
}

fn model_toa(row: &Row, sigma: f64, aerosol_ssa: f64, seed: u64, photons: u32) -> f64 {
    ozone_transmittance(row, sigma)
        * coupling(row, aerosol_ssa, seed, photons).toa_from_surface(row.boa)
}

/// A seed per row, per draw -- the test's stream, so scan point `AEROSOL_SSA` reproduces its number.
fn stream(row: usize, draw: u64) -> u64 {
    SEED.wrapping_add((row as u64).wrapping_mul(0xff51_afd7_ed55_8ccd))
        .wrapping_add(draw.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

fn rms(values: &[f64]) -> f64 {
    libm::sqrt(values.iter().map(|e| e * e).sum::<f64>() / values.len() as f64)
}

/// The RMS TOA residual over the rows `keep` admits, at one candidate albedo.
fn residual(
    rows: &[Row],
    table: &[(f64, f64)],
    candidate: f64,
    photons: u32,
    keep: &dyn Fn(&Row) -> bool,
) -> (usize, f64) {
    let errors: Vec<f64> = rows
        .iter()
        .enumerate()
        .filter(|(_, row)| keep(row))
        .map(|(index, row)| {
            let sigma = sigma_for(table, row);
            model_toa(row, sigma, candidate, stream(index, 0), photons) - row.toa
        })
        .collect();
    (errors.len(), rms(&errors))
}

/// The candidate that minimises a parabola through the scan's lowest interior point and its
/// neighbours. Returns the grid point itself where the three are not convex, which is the honest
/// answer for a scan whose minimum sits on an end.
fn vertex(scan: &[(f64, f64)]) -> f64 {
    let best = (1..scan.len() - 1)
        .min_by(|a, b| scan[*a].1.total_cmp(&scan[*b].1))
        .unwrap_or(1);
    let (left, mid, right) = (scan[best - 1], scan[best], scan[best + 1]);
    let curvature = right.1 - 2.0 * mid.1 + left.1;
    if curvature <= 0.0 {
        return mid.0;
    }
    mid.0 - 0.5 * (mid.0 - left.0) * (right.1 - left.1) / curvature
}

/// Ordinary least squares with the standard error of every coefficient.
///
/// The normal matrix is at most 14x14 here, so Gauss-Jordan on `[X'X | I]` costs nothing and produces
/// the inverse the standard errors need in the same pass. Two things are deliberate. The pivot is
/// checked rather than assumed, because a design with two columns that are the same variable would
/// otherwise return a coefficient built out of rounding error. And the caller is expected to have
/// standardised the columns, so the threshold below can be relative to a normal matrix whose
/// diagonal is `n`.
fn least_squares(design: &[Vec<f64>], observed: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (n, p) = (design.len(), design[0].len());
    assert!(n > p, "{n} points cannot determine {p} coefficients");
    let mut normal = vec![vec![0.0; p]; p];
    let mut projection = vec![0.0; p];
    for (row, y) in design.iter().zip(observed) {
        for i in 0..p {
            projection[i] += row[i] * y;
            for j in 0..p {
                normal[i][j] += row[i] * row[j];
            }
        }
    }
    let mut inverse = vec![vec![0.0; p]; p];
    for i in 0..p {
        inverse[i][i] = 1.0;
    }
    for col in 0..p {
        let pivot = (col..p)
            .max_by(|a, b| normal[*a][col].abs().total_cmp(&normal[*b][col].abs()))
            .expect("a column");
        assert!(
            normal[pivot][col].abs() > 1e-8 * n as f64,
            "the design is singular at column {col}: two drivers are the same variable"
        );
        normal.swap(col, pivot);
        inverse.swap(col, pivot);
        let scale = normal[col][col];
        for j in 0..p {
            normal[col][j] /= scale;
            inverse[col][j] /= scale;
        }
        for row in 0..p {
            if row == col {
                continue;
            }
            let factor = normal[row][col];
            for j in 0..p {
                normal[row][j] -= factor * normal[col][j];
                inverse[row][j] -= factor * inverse[col][j];
            }
        }
    }
    let beta: Vec<f64> = (0..p)
        .map(|i| (0..p).map(|j| inverse[i][j] * projection[j]).sum())
        .collect();
    let residual: f64 = design
        .iter()
        .zip(observed)
        .map(|(row, y)| {
            let predicted: f64 = row.iter().zip(&beta).map(|(x, c)| x * c).sum();
            (y - predicted) * (y - predicted)
        })
        .sum();
    let variance = residual / (n - p) as f64;
    let error = (0..p)
        .map(|i| libm::sqrt(variance * inverse[i][i]))
        .collect();
    (beta, error)
}

/// What moved with epoch besides the residual, per site, which is what the controlled column holds
/// fixed. Split into the pooled fixture and the sampled days alone, because only the second is the
/// sampler's to answer for.
struct Confound<'a> {
    site: &'a str,
    season: f64,
    season_t: f64,
    sampled_season: f64,
    sampled_season_t: f64,
    surface: f64,
    surface_t: f64,
}

/// Everything in [`Confound`] for one site: what moved with epoch besides the residual.
///
/// `design` is the site's own epoch design, so the pooled columns share a centring and a denominator
/// with the drift slope they explain. The sampled column has to build its own, because it is a
/// different set of rows and the midpoint of the whole site is not the midpoint of those.
fn confounds<'a>(site: &'a str, mine: &[&Point], design: &[Vec<f64>]) -> Confound<'a> {
    let slope_on_epoch = |values: Vec<f64>| {
        let (fit, err) = least_squares(design, &values);
        (fit[1], fit[1] / err[1])
    };
    let (season, season_t) = slope_on_epoch(mine.iter().map(|p| p.day_of_year).collect());
    let (surface, surface_t) = slope_on_epoch(mine.iter().map(|p| p.boa).collect());

    let held: Vec<&&Point> = mine.iter().filter(|p| p.held_out).collect();
    let years: Vec<f64> = held.iter().map(|p| p.epoch_year).collect();
    let mid = f64::midpoint(
        years.iter().copied().fold(f64::INFINITY, f64::min),
        years.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    let held_design: Vec<Vec<f64>> = years.iter().map(|y| vec![1.0, y - mid]).collect();
    let held_days: Vec<f64> = held.iter().map(|p| p.day_of_year).collect();
    let (fit, err) = least_squares(&held_design, &held_days);

    Confound {
        site,
        season,
        season_t,
        sampled_season: fit[1],
        sampled_season_t: fit[1] / err[1],
        surface,
        surface_t,
    }
}

/// One comparison reduced to its relative residual and the physical quantities it could depend on.
struct Point {
    site: String,
    /// `YYYY-DDD`, the grouping key for the per-day fit below. Comparing `epoch_year` would work
    /// today because it is parsed from this string, but it is a float and this is a key.
    date: String,
    epoch_year: f64,
    day_of_year: f64,
    relative: f64,
    boa: f64,
    drivers: [f64; 8],
    /// True for the days the sampler chose and nothing was fitted on. The drift regression pools
    /// both sets, so without this flag the season column below cannot say which of the two carries
    /// the calendar drift it reports.
    held_out: bool,
}

/// Every published quantity that could move the residual, which is the list the drift regression has
/// to control for before it may call a surviving slope a property of the reference.
///
/// The last three were added after the first drift verdict: ozone enters the answer directly through
/// the Chappuis transmittance, pressure scales the Rayleigh depth, and the Angstrom exponent carries
/// AOD to every wavelength off 550 nm. A secular change in any of the three at a site -- an instrument
/// swap, an ozone-product version change -- lands in an epoch slope that does not hold them fixed.
const DRIVERS: [&str; 8] = [
    "aod550",
    "air mass",
    "water cm",
    "rho_BOA",
    "path ratio",
    "ozone DU",
    "pressure",
    "angstrom",
];

/// Every gated comparison, once, at the fitted albedo -- the input both regressions below read.
fn points(
    rows: &[Row],
    fitted_rows: usize,
    table: &[(f64, f64)],
    fitted: f64,
    photons: u32,
) -> Vec<Point> {
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            let sigma = sigma_for(table, row);
            let modelled = model_toa(row, sigma, fitted, stream(index, 0), photons);
            Point {
                site: row.site.clone(),
                date: row.date.clone(),
                epoch_year: row.epoch_year(),
                day_of_year: row.day_of_year(),
                relative: (modelled - row.toa) / row.toa,
                boa: row.boa,
                held_out: index >= fitted_rows,
                drivers: [
                    row.aod550,
                    row.air_mass(),
                    row.water_cm,
                    row.boa,
                    row.path_ratio(),
                    row.ozone_du,
                    row.pressure_hpa,
                    row.angstrom,
                ],
            }
        })
        .collect()
}

/// Centre and scale a column, returning the standardised values and the original spread.
fn standardise(column: &[f64]) -> (Vec<f64>, f64, f64) {
    let n = column.len() as f64;
    let mean = column.iter().sum::<f64>() / n;
    let sd = libm::sqrt(column.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0));
    assert!(
        sd > 0.0,
        "a driver that never varies cannot be regressed on"
    );
    (column.iter().map(|v| (v - mean) / sd).collect(), mean, sd)
}

/// What the residual actually depends on, rather than which group it falls in.
///
/// Section 5.4 reports the residual per site and per wavelength. Both are groupings: "Baotou sand is
/// worst" invites a guess at why, and the guess in the prose -- that it is the darkest target, so
/// path reflectance is most of the signal -- was stated and never measured. This measures it, by
/// regressing the *relative* residual on the eight published quantities that could produce it.
///
/// The columns are standardised, so a coefficient is the change in relative residual per one standard
/// deviation of its driver and the eight are directly comparable. The drivers are **not** independent
/// -- `path ratio` is built from `rho_BOA` and the optical depth by construction -- so the pairwise
/// correlations are printed alongside. Collinearity does not bias the coefficients; it inflates their
/// standard errors, and the `t` column is where that shows.
fn report_drivers(points: &[Point]) -> Bounds {
    println!(
        "\n# what the residual depends on, over {} comparisons",
        points.len()
    );
    let columns: Vec<(Vec<f64>, f64, f64)> = (0..DRIVERS.len())
        .map(|d| standardise(&points.iter().map(|p| p.drivers[d]).collect::<Vec<f64>>()))
        .collect();
    let (pooled, pooled_error) = fit_drivers(points, &columns, &[]);
    let sites = site_indicators(points);
    let (within, within_error) = fit_drivers(points, &columns, &sites);

    println!(
        "{:>10} {:>10} {:>10} {:>7} {:>12} {:>10} {:>12} {:>10} {:>7} {:>9}",
        "driver", "coef", "std err", "t", "mean", "sd", "within site", "std err", "t", "within sd"
    );
    println!(
        "{:>10} {:10.5} {:10.5} {:7.2}",
        "intercept",
        pooled[0],
        pooled_error[0],
        pooled[0] / pooled_error[0]
    );
    for (d, name) in DRIVERS.iter().enumerate() {
        let (_, mean, sd) = columns[d];
        println!(
            "{name:>10} {:10.5} {:10.5} {:7.2} {mean:12.4} {sd:10.4} {:12.5} {:10.5} {:7.2} {:9.3}",
            pooled[d + 1],
            pooled_error[d + 1],
            pooled[d + 1] / pooled_error[d + 1],
            within[d + 1],
            within_error[d + 1],
            within[d + 1] / within_error[d + 1],
            within_spread(points, d)
        );
    }
    println!("# coefficients are per standard deviation of the driver, so they compare directly");
    println!(
        "# the `within site` column adds {} indicators against a reference site, so it is identified",
        sites.len()
    );
    println!(
        "# from variation inside a site only -- what collapses there was geography, not physics"
    );
    println!("# `within sd` is that variation as a fraction of the pooled spread the coefficients");
    println!("# are quoted per, so a small one means the within-site number is an extrapolation");

    report_driver_correlations(points, &columns);

    // Two drivers are wanted in their own unit rather than per standard deviation, because the
    // bounds below multiply them by a published uncertainty in cm and by a displacement in air
    // masses. The *larger* of the two designs is the one to carry each time: these feed bounds, and
    // a bound built from the smaller of two defensible estimates is not one.
    let per_unit = |name: &str| {
        let d = DRIVERS
            .iter()
            .position(|driver| *driver == name)
            .expect("a named driver");
        let sd = columns[d].2;
        let (slope, error, design) = if pooled[d + 1].abs() >= within[d + 1].abs() {
            (pooled[d + 1], pooled_error[d + 1], "pooled")
        } else {
            (within[d + 1], within_error[d + 1], "within site")
        };
        println!(
            "# the {name} bound below is taken from the {design} column, the larger of the two"
        );
        (slope / sd, error / sd)
    };
    let (water_per_cm, water_per_cm_err) = per_unit("water cm");
    let (per_air_mass, per_air_mass_err) = per_unit("air mass");
    Bounds {
        water_per_cm,
        water_per_cm_err,
        per_air_mass,
        per_air_mass_err,
    }
}

/// The two empirical slopes the bounds below are built from, each in the driver's own unit.
struct Bounds {
    water_per_cm: f64,
    water_per_cm_err: f64,
    per_air_mass: f64,
    per_air_mass_err: f64,
}

/// A driver's within-site spread as a fraction of its pooled one.
///
/// Both coefficient columns are quoted per *pooled* standard deviation, which is what makes them
/// comparable -- but a driver whose spread is almost entirely between sites is identified in the
/// second column by a sliver of its own range, and printing that coefficient without this number
/// invites reading an extrapolation as a measurement. Surface pressure is the case: it is very
/// nearly site elevation.
fn within_spread(points: &[Point], driver: usize) -> f64 {
    let mut sites: Vec<&str> = points.iter().map(|p| p.site.as_str()).collect();
    sites.sort_unstable();
    sites.dedup();
    let inside: f64 = sites
        .iter()
        .map(|site| {
            let mine: Vec<f64> = points
                .iter()
                .filter(|p| p.site == *site)
                .map(|p| p.drivers[driver])
                .collect();
            let mean = mine.iter().sum::<f64>() / mine.len() as f64;
            mine.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>()
        })
        .sum();
    let all: Vec<f64> = points.iter().map(|p| p.drivers[driver]).collect();
    let mean = all.iter().sum::<f64>() / all.len() as f64;
    let total: f64 = all.iter().map(|v| (v - mean) * (v - mean)).sum();
    libm::sqrt(inside / total)
}

/// One indicator column per site after the first, so the intercept stays identified.
///
/// The reference is whichever site sorts first, and which one it is does not matter: adding a
/// constant to every indicator moves only the intercept, and no driver coefficient with it.
fn site_indicators(points: &[Point]) -> Vec<Vec<f64>> {
    let mut sites: Vec<&str> = points.iter().map(|p| p.site.as_str()).collect();
    sites.sort_unstable();
    sites.dedup();
    sites[1..]
        .iter()
        .map(|site| {
            points
                .iter()
                .map(|p| f64::from(u8::from(p.site == *site)))
                .collect()
        })
        .collect()
}

/// The driver regression, with or without the site indicators appended.
///
/// The indicators go *after* the drivers so that index `d + 1` is driver `d` in both fits and the
/// two columns of the table are read off the same offset.
fn fit_drivers(
    points: &[Point],
    columns: &[(Vec<f64>, f64, f64)],
    sites: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>) {
    let design: Vec<Vec<f64>> = (0..points.len())
        .map(|i| {
            let mut row = vec![1.0];
            row.extend(columns.iter().map(|(values, _, _)| values[i]));
            row.extend(sites.iter().map(|column| column[i]));
            row
        })
        .collect();
    let observed: Vec<f64> = points.iter().map(|p| p.relative).collect();
    least_squares(&design, &observed)
}

/// How much of each driver is already in the others, which is why the `t` column is the one to read.
fn report_driver_correlations(points: &[Point], columns: &[(Vec<f64>, f64, f64)]) {
    println!("\n# the drivers are correlated by construction, which is why the t column matters");
    print!("{:>10}", "");
    for name in DRIVERS {
        print!(" {name:>10}");
    }
    println!();
    let n = points.len() as f64;
    for (i, name) in DRIVERS.iter().enumerate() {
        print!("{name:>10}");
        for j in 0..DRIVERS.len() {
            let r: f64 = (0..points.len())
                .map(|k| columns[i].0[k] * columns[j].0[k])
                .sum::<f64>()
                / (n - 1.0);
            print!(" {r:10.3}");
        }
        println!();
    }
}

/// What the water column would cost the budget if the model had a channel for it.
///
/// The budget's four terms are this model's own response to inputs it reads. Water is the input it
/// does not: `Atmosphere` carries no gaseous absorption, so moving the published water column to the
/// edge of its uncertainty moves the answer by exactly zero, and printing that zero would describe
/// the model rather than the scene. The driver regression measures the sensitivity the model lacks --
/// the residual does move with water -- so that slope times RadCalNet's own published water
/// uncertainty bounds what the missing term is worth. It is an upper bound and not a budget entry:
/// the slope is empirical, so it carries whatever else tracks the water column along with it.
fn report_water_bound(rows: &[Row], per_cm: f64, per_cm_err: f64) {
    let n = rows.len() as f64;
    let uncertainty = rows.iter().map(|row| row.water_cm_unc).sum::<f64>() / n;
    let toa = rows.iter().map(|row| row.toa).sum::<f64>() / n;
    println!("\n# the input the model has no channel for, bounded by the slope it does have");
    println!("#   sensitivity  {per_cm:9.5} +/- {per_cm_err:.5} of signal per cm of column");
    println!("#   RadCalNet's own water uncertainty averages {uncertainty:.4} cm over {n:.0} rows");
    println!(
        "#   so a water-aware model would carry {:.4e} of TOA reflectance, at mean rho_TOA {toa:.4}",
        per_cm * uncertainty * toa
    );
}

/// How much of the residual's air-mass dependence a pseudo-spherical slant path could remove.
///
/// [`crate::radiative_transfer::Atmosphere`] is plane-parallel: the slant path is `sec z`, which
/// diverges as the sun sets and is 4 % long at the 81.5 deg RadCalNet reaches. The standard repair --
/// 6S, MODTRAN and libRadtran all carry it -- replaces `sec z` by a real relative air mass for the
/// extinction while leaving the scattering plane-parallel. This bounds what that would buy *before*
/// paying for it, the same way the water term above is bounded rather than charged: the displacement
/// it would apply to each row's air mass, times the residual's measured response to air mass.
///
/// Two displacements, because the answer should not rest on one parameterisation. The first is exact
/// for the slab this transport models -- a homogeneous shell of thickness `H` on a sphere of radius
/// `R` has slant path `sqrt((R+H)^2 - R^2 sin^2 z) - R cos z`, and `H = 8 km` is the scale height of
/// the atmosphere whose column the Rayleigh depth is computed from. The second is Kasten & Young's
/// 1989 empirical fit to the US Standard Atmosphere, `1/(cos z + 0.50572 (96.07995 - z)^-1.6364)`,
/// which is the larger of the two and is therefore the one the bound is quoted on.
fn report_sphericity(rows: &[Row], per_air_mass: f64, per_air_mass_err: f64) {
    const EARTH_KM: f64 = 6371.0;
    const SCALE_HEIGHT_KM: f64 = 8.0;
    let shell = |z: f64| {
        let s = libm::sin(z.to_radians());
        (libm::sqrt(
            (EARTH_KM + SCALE_HEIGHT_KM) * (EARTH_KM + SCALE_HEIGHT_KM)
                - EARTH_KM * EARTH_KM * s * s,
        ) - EARTH_KM * libm::cos(z.to_radians()))
            / SCALE_HEIGHT_KM
    };
    let kasten_young =
        |z: f64| 1.0 / (libm::cos(z.to_radians()) + 0.50572 * libm::pow(96.07995 - z, -1.6364));

    let n = rows.len() as f64;
    let toa = rows.iter().map(|row| row.toa).sum::<f64>() / n;
    println!("\n# what a pseudo-spherical slant path would displace the air mass by, per row");
    println!(
        "{:>14} {:>12} {:>12} {:>12}",
        "displacement", "mean", "worst", "worst frac"
    );
    let mut widest = 0.0_f64;
    for (name, air_mass) in [
        ("shell 8 km", &shell as &dyn Fn(f64) -> f64),
        ("Kasten-Young", &kasten_young),
    ] {
        let displaced: Vec<f64> = rows
            .iter()
            .map(|row| 1.0 / row.mu_sun() - air_mass(row.solar_zenith_deg))
            .collect();
        let mean = displaced.iter().sum::<f64>() / n;
        let worst = displaced.iter().copied().fold(0.0_f64, f64::max);
        let fraction = rows
            .iter()
            .zip(&displaced)
            .map(|(row, d)| d * row.mu_sun())
            .fold(0.0_f64, f64::max);
        println!(
            "{name:>14} {mean:12.5} {worst:12.5} {:11.2} %",
            100.0 * fraction
        );
        widest = widest.max(mean);
    }
    println!(
        "#   sensitivity  {per_air_mass:9.5} +/- {per_air_mass_err:.5} of signal per unit of air mass"
    );
    println!(
        "#   so a pseudo-spherical slab would move the mean row by {:.4e} of TOA reflectance, at \
         mean rho_TOA {toa:.4}",
        widest * per_air_mass * toa
    );
    println!(
        "# only the solar leg carries this: RadCalNet publishes nadir, so the view leg is 1.0"
    );
}

/// Whether RadCalNet itself has moved over the years it publishes.
///
/// Every residual in Section 5.4 treats the reference as stationary. A site's radiometer gets
/// recalibrated, a processing version changes, an instrument is replaced -- and the published
/// top-of-atmosphere reflectance shifts with the atmosphere doing nothing. If our residual drifts
/// with epoch, that is a property of the oracle rather than of our slab, and it belongs in the bounds
/// of the claim before anyone quotes the bar. The file listing carries no reprocessing versions -- 0
/// multi-version site-days across all six sites -- so epoch is the only handle there is.
///
/// Epoch alone is not enough to answer it. The days are sampled on the calendar, not on the weather,
/// so nothing stops a site's later sample from being dustier or damper than its earlier one,
/// and the drivers above move the residual at `t` between 4 and 10. A slope on epoch alone cannot tell
/// a reference that moved from an atmosphere that was sampled differently. So both are printed: the
/// univariate slope, and the same slope with every published driver in the design, centred within the
/// site. The gap between them is the confounding, measured rather than assumed.
///
/// The controlled column asks the narrower question -- at the same measured atmosphere and the same
/// measured surface brightness, does the residual move with epoch -- and that is deliberate: a target
/// that genuinely changed shows up in `rho_BOA`, which enters both sides of the comparison, so holding
/// it fixed separates a drifting *reference* from a drifting *site*.
///
/// **The season is balanced in the hold-out and unbalanced in the fitted scenes.** Archive density is
/// not uniform through the year, so a sample whose early days are summer-heavy and whose later days
/// are not shows a residual trend that is the calendar rather than the reference. The generator now
/// picks the hold-out to satisfy the balance equation directly, and the second table's `held-out only`
/// column is that constraint reported back. The 42 fitted scenes were chosen for their content, so
/// theirs cannot be balanced after the fact -- which is what the annual harmonic in this design is
/// left holding, and why it stays in it.
///
/// Eleven controls of very different natural size make the normal matrix worth conditioning, so the
/// controls are standardised within the site. That cannot touch the epoch coefficient -- rescaling a
/// regressor scales only its own coefficient -- so the slope stays in units of per year.
/// One point per calendar day: that day's gated wavelengths collapsed into a single observation.
///
/// Nothing inside a day is independent. The rows share the day's aerosol, water, ozone, pressure and
/// solar geometry, and the wavelengths of one scene move together -- so a row-level fit divides by a
/// standard error built on ten to fourteen times the information the design holds. The six per-day
/// drivers survive the average unchanged; `rho_BOA` and the path ratio are per-wavelength, and their
/// day mean is the quantity a day-mean residual can depend on.
fn per_day(mine: &[&Point]) -> Vec<Point> {
    let mut dates: Vec<&str> = mine.iter().map(|p| p.date.as_str()).collect();
    dates.sort_unstable();
    dates.dedup();
    dates
        .into_iter()
        .map(|date| {
            let day: Vec<&&Point> = mine.iter().filter(|p| p.date == date).collect();
            let n = day.len() as f64;
            let mean =
                |value: &dyn Fn(&Point) -> f64| day.iter().map(|p| value(p)).sum::<f64>() / n;
            let mut drivers = [0.0; 8];
            for (d, averaged) in drivers.iter_mut().enumerate() {
                *averaged = mean(&|p| p.drivers[d]);
            }
            Point {
                site: day[0].site.clone(),
                date: date.to_string(),
                epoch_year: day[0].epoch_year,
                day_of_year: day[0].day_of_year,
                relative: mean(&|p| p.relative),
                boa: mean(&|p| p.boa),
                held_out: day[0].held_out,
                drivers,
            }
        })
        .collect()
}

/// One site's drift verdict: the epoch slope alone, and then the same slope holding every published
/// driver and the annual harmonic fixed.
struct Drift {
    n: usize,
    low: f64,
    high: f64,
    slope: f64,
    slope_err: f64,
    controlled: f64,
    controlled_err: f64,
}

/// Fit [`Drift`] for one site, returning the univariate design as well because [`confounds`] has to
/// share its centring to be explaining the slope this printed.
fn drift_fit(mine: &[&Point]) -> (Drift, Vec<Vec<f64>>) {
    let years: Vec<f64> = mine.iter().map(|p| p.epoch_year).collect();
    let (low, high) = (
        years.iter().copied().fold(f64::INFINITY, f64::min),
        years.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    );
    // Centred on the site's own midpoint so the slope and the intercept are uncorrelated and the
    // standard error below is the slope's alone.
    let mid = f64::midpoint(low, high);
    let design: Vec<Vec<f64>> = years.iter().map(|y| vec![1.0, y - mid]).collect();
    let observed: Vec<f64> = mine.iter().map(|p| p.relative).collect();
    let (beta, error) = least_squares(&design, &observed);

    let mut controls: Vec<Vec<f64>> = (0..DRIVERS.len())
        .map(|d| standardise(&mine.iter().map(|p| p.drivers[d]).collect::<Vec<f64>>()).0)
        .collect();
    for phase in [libm::sin, libm::cos] {
        let harmonic: Vec<f64> = mine
            .iter()
            .map(|p| phase(std::f64::consts::TAU * p.day_of_year / 365.25))
            .collect();
        controls.push(standardise(&harmonic).0);
    }
    let controlled_design: Vec<Vec<f64>> = years
        .iter()
        .enumerate()
        .map(|(i, y)| {
            let mut row = vec![1.0, y - mid];
            row.extend(controls.iter().map(|column| column[i]));
            row
        })
        .collect();
    let (controlled, controlled_error) = least_squares(&controlled_design, &observed);

    let fit = Drift {
        n: mine.len(),
        low,
        high,
        slope: beta[1],
        slope_err: error[1],
        controlled: controlled[1],
        controlled_err: controlled_error[1],
    };
    (fit, design)
}

fn drift_header() {
    println!(
        "{:>10} {:>5} {:>16} {:>12} {:>11} {:>6} {:>12} {:>11} {:>6}",
        "site", "n", "span", "slope /yr", "std err", "t", "controlled", "std err", "t"
    );
}

fn print_drift(site: &str, fit: &Drift) {
    println!(
        "{site:>10} {:5} {:7.1}-{:6.1} {:12.5} {:11.5} {:6.2} {:12.5} {:11.5} {:6.2}",
        fit.n,
        fit.low,
        fit.high,
        fit.slope,
        fit.slope_err,
        fit.slope / fit.slope_err,
        fit.controlled,
        fit.controlled_err,
        fit.controlled / fit.controlled_err
    );
}

fn report_drift(points: &[Point]) {
    println!("\n# does the reference drift? relative residual against epoch, per site");
    drift_header();
    let mut sites: Vec<&str> = points.iter().map(|p| p.site.as_str()).collect();
    sites.sort_unstable();
    sites.dedup();
    // What else moved with epoch, printed after the verdict it explains.
    let mut elsewhere: Vec<Confound<'_>> = Vec::new();
    // The same fit on the days those rows came from, printed as its own table.
    let mut daily: Vec<(&str, Drift)> = Vec::new();
    for site in sites {
        let mine: Vec<&Point> = points.iter().filter(|p| p.site == site).collect();
        let (fit, design) = drift_fit(&mine);
        print_drift(site, &fit);
        elsewhere.push(confounds(site, &mine, &design));

        let days = per_day(&mine);
        let averaged: Vec<&Point> = days.iter().collect();
        daily.push((site, drift_fit(&averaged).0));
    }
    println!("# |t| under 2 is a reference this comparison cannot distinguish from stationary");
    println!(
        "# the n above counts rows, and the rows of one day are not independent of each other"
    );

    println!("\n# the same fit on the days those rows came from, the wavelengths averaged first");
    drift_header();
    for (site, fit) in &daily {
        print_drift(site, fit);
    }
    println!("# this is the honest n: 20 sampled days plus 3 fitted scenes, against 12 columns.");
    println!("# quote this table. the row-level one above is what the clustering was worth, and");
    println!("# the gap between the two |t| columns is that number rather than an argument for it");

    println!(
        "\n# and what else moved with epoch, which is what the controls above are holding fixed"
    );
    println!(
        "{:>10} {:>14} {:>7} {:>14} {:>7} {:>14} {:>7}",
        "site", "season d/yr", "t", "held-out only", "t", "rho_BOA /yr", "t"
    );
    for c in elsewhere {
        println!(
            "{:>10} {:14.3} {:7.2} {:14.3} {:7.2} {:14.6} {:7.2}",
            c.site,
            c.season,
            c.season_t,
            c.sampled_season,
            c.sampled_season_t,
            c.surface,
            c.surface_t
        );
    }
    println!("# the held column is the sampler's own constraint reported back: it balances");
    println!("# day-of-year against epoch over the days it picks, and what is left there is a");
    println!("# finite archive's discrete floor times the wavelengths each day kept. the pooled");
    println!("# column also carries the 42 fitted scenes, chosen for content, which no sampling");
    println!("# rule can balance after the fact -- that residue is the harmonic's whole job.");
    println!("# a resolvable rho_BOA column is the target itself changing, not the reference");
}

/// What this model's answer moves by when each measured input is pushed to the edge of its own
/// published uncertainty, itemised, against RadCalNet's published uncertainty on the same rows.
fn report_budget(rows: &[Row], table: &[(f64, f64)], fitted: f64, photons: u32, seeds: u64) {
    println!(
        "\n# the error budget at omega_a = {fitted:.4}, per measured input, over {seeds} draws"
    );
    let mut budget = Vec::new();
    for (input, name) in INPUTS {
        // Common random numbers: the perturbed and unperturbed runs share a seed, so their photon
        // paths cancel and the difference is a derivative rather than the difference of two noisy
        // means. What survives is beaten down by averaging the *difference* over draws -- averaging
        // its magnitude would be biased upward and would report noise as sensitivity.
        let terms: Vec<f64> = rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let sigma = sigma_for(table, row);
                let moved = row.perturbed(input);
                let total: f64 = (0..seeds)
                    .map(|draw| {
                        let seed = stream(index, draw);
                        model_toa(&moved, sigma, fitted, seed, photons)
                            - model_toa(row, sigma, fitted, seed, photons)
                    })
                    .sum();
                total / seeds as f64
            })
            .collect();
        let value = rms(&terms);
        println!("{name:>10} {value:12.4e} of TOA reflectance");
        budget.push(value);
    }
    let ours = libm::sqrt(budget.iter().map(|v| v * v).sum::<f64>());
    let reference = rms(&rows.iter().map(|r| r.toa_unc).collect::<Vec<f64>>());
    println!(
        "{:>10} {ours:12.4e} against RadCalNet's {reference:.4e} -- {:.1} % of the quadrature sum",
        "ours",
        100.0 * ours / libm::sqrt(ours * ours + reference * reference)
    );
}

/// What the pooled fit costs on days it never saw, split into the two things it can be.
///
/// The suite's hold-out test asks only whether the fitted albedo predicts inside the bar. This asks
/// the harder question: of the residual it leaves out of sample, how much is the albedo being wrong
/// for these days, and how much is error the albedo cannot reach? Re-fitting on the hold-out answers
/// it -- the gap between the two curves at the same points is the mis-specification, and what is left
/// at the hold-out's own minimum is everything else.
fn report_holdout(
    held: &[Row],
    table: &[(f64, f64)],
    fitted: f64,
    candidates: &[f64],
    photons: u32,
) {
    println!("\n# the hold-out: days sampled by date, never used to fit anything");
    println!("{:>10} {:>5} {:>12}", "omega_a", "n", "rms");
    let scan: Vec<(f64, f64)> = candidates
        .iter()
        .map(|&candidate| {
            let (n, value) = residual(held, table, candidate, photons, &|_| true);
            println!("{candidate:10.3} {n:5} {value:12.4e}");
            (candidate, value)
        })
        .collect();
    // Both residuals are evaluated at an interpolated albedo, not at the nearest grid point. Taking
    // the scan's minimum for one and an interpolated value for the other would compare two different
    // things, and the difference between them is smaller than that mistake.
    let own = vertex(&scan);
    let (_, at_own) = residual(held, table, own, photons, &|_| true);
    let (n, at_fitted) = residual(held, table, fitted, photons, &|_| true);
    println!("# its own minimum is at omega_a = {own:.4}, against {fitted:.4} fitted in sample");
    println!(
        "# over {n} comparisons: {at_fitted:.4e} at the in-sample value, {at_own:.4e} at its own \
         -- refitting buys {:.1} %",
        100.0 * (at_fitted - at_own) / at_fitted
    );
}

fn main() {
    let mut argv = std::env::args().skip(1);
    let photons: u32 = argv
        .next()
        .map_or(40_000, |v| v.parse().expect("photons is a number"));
    let steps: usize = argv
        .next()
        .map_or(11, |v| v.parse().expect("steps is a number"));
    let seeds: u64 = argv
        .next()
        .map_or(3, |v| v.parse().expect("seeds is a number"));
    assert!(steps >= 3, "a parabolic minimum needs three points");
    assert!(seeds >= 1, "the budget needs at least one draw");

    let rows = rows();
    let table = ozone_cross_section();
    let mut sites: Vec<String> = rows.iter().map(|row| row.site.clone()).collect();
    sites.sort_unstable();
    sites.dedup();
    println!(
        "# {} gated comparisons, {} instruments, {photons} photons, {steps} points over \
         [{SCAN_LOW}, {SCAN_HIGH}]",
        rows.len(),
        sites.len()
    );

    let candidates: Vec<f64> = (0..steps)
        .map(|i| SCAN_LOW + (SCAN_HIGH - SCAN_LOW) * i as f64 / (steps - 1) as f64)
        .collect();

    println!("\n# the whole fixture");
    println!("{:>10} {:>5} {:>12}", "omega_a", "n", "rms");
    let whole: Vec<(f64, f64)> = candidates
        .iter()
        .map(|&candidate| {
            let (n, value) = residual(&rows, &table, candidate, photons, &|_| true);
            println!("{candidate:10.3} {n:5} {value:12.4e}");
            (candidate, value)
        })
        .collect();
    let fitted = vertex(&whole);
    let floor = whole.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
    let ceiling = whole
        .iter()
        .map(|(_, v)| *v)
        .fold(f64::NEG_INFINITY, f64::max);
    println!("# parabolic minimum at omega_a = {fitted:.4}");
    println!(
        "# the scan spans {:.4e} of TOA reflectance, {:.2}x the floor -- flat means unidentified",
        ceiling - floor,
        ceiling / floor
    );

    println!("\n# per instrument -- one fitted albedo has to serve all six targets");
    println!(
        "{:>10} {:>5} {:>10} {:>12} {:>12}",
        "site", "n", "omega_a", "rms there", "at fitted"
    );
    for site in &sites {
        let keep = |row: &Row| row.site == *site;
        let scan: Vec<(f64, f64)> = candidates
            .iter()
            .map(|&candidate| {
                (
                    candidate,
                    residual(&rows, &table, candidate, photons, &keep).1,
                )
            })
            .collect();
        let (n, at_fitted) = residual(&rows, &table, fitted, photons, &keep);
        let best = vertex(&scan);
        let at_best = scan.iter().map(|(_, v)| *v).fold(f64::INFINITY, f64::min);
        println!("{site:>10} {n:5} {best:10.4} {at_best:12.4e} {at_fitted:12.4e}");
    }

    let held = holdout();
    report_holdout(&held, &table, fitted, &candidates, photons);

    // Both regressions read every comparison in the workspace, fitted and held out together: five
    // correlated drivers cannot be separated on 42 scenes, and the drift regression wants the widest
    // epoch span there is. Pooling is legitimate here in a way it is not for the fit -- neither
    // regression estimates omega_a, and both are asking what is left over once it is fixed.
    let pooled: Vec<Row> = rows.iter().chain(&held).cloned().collect();
    let points = points(&pooled, rows.len(), &table, fitted, photons);
    let bounds = report_drivers(&points);
    report_water_bound(&pooled, bounds.water_per_cm, bounds.water_per_cm_err);
    report_sphericity(&pooled, bounds.per_air_mass, bounds.per_air_mass_err);
    report_drift(&points);

    report_budget(&rows, &table, fitted, photons, seeds);
}
