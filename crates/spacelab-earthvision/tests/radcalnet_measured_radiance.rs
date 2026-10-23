//! The 6S coupling on scenes RadCalNet measured, against the MODTRAN that RadCalNet publishes.
//!
//! `aeronet_measured_atmosphere` ends by naming what it is not: "this is still a comparison of two
//! solvers on a measured *atmosphere*, not against a measured *radiance*". This is that rung.
//! RadCalNet publishes, for instrumented desert and gravel targets, the surface reflectance measured
//! on the ground, the atmosphere measured over it, and the top-of-atmosphere reflectance MODTRAN
//! 5.3.3 produces from the two. `tools/gen_radcalnet_oracle.py` takes all three.
//!
//! So this is our Monte-Carlo slab against MODTRAN with DISORT at eight streams, **both driven by
//! the same measured surface and the same measured atmosphere**, on days nobody chose for us. A
//! third radiative-transfer lineage after DISORT and libRadtran, and the first attached to a real
//! target.
//!
//! **It is still not a satellite measurement.** RadCalNet's TOA reflectance is a model output; what
//! is measured is everything fed into it. The residual below is between two radiative-transfer codes
//! on one measured scene, and nothing here says what a sensor would have recorded.
//!
//! **What is quoted, what is measured, and the one thing that is fitted.** RadCalNet's MODTRAN deck
//! is published in the format specification, so most of the model is not a choice of ours: a
//! Lambertian surface (`SURREF: LAMBER`), rural aerosol (`IHAZE: 2`), and a **Henyey-Greenstein
//! aerosol phase function of asymmetry 0.5** (`IPH: 2`, `G: 0.500`) -- which is
//! [`PhaseFunction::HenyeyGreenstein`] at [`AEROSOL_ASYMMETRY`], quoted rather than chosen. The
//! optical depths, the ozone column, the pressure, the geometry and both reflectances are measured.
//! What is left is `NSSALB: 0`, which hands the aerosol single-scattering albedo to MODTRAN's
//! internal rural model and publishes nothing, so it cannot be looked up. It is **fitted**:
//! [`AEROSOL_SSA`] is an interior minimum of a scan over the same comparisons the residuals are
//! reported on, and `examples/radcalnet_error_budget.rs` runs the scan in full. One free parameter
//! against 313 comparisons -- which is why the *signed mean* below is near zero by construction and
//! carries no information, and the RMS and the ratio are what to read.
//!
//! **How the measurements become an [`Atmosphere`].** The type is one grey slab, so the column is
//! collapsed into one the only way it allows:
//!
//! ```text
//! tau   = tau_R(lambda, P) + AOT(550)(lambda/550)^-alpha       Rayleigh at the measured pressure
//! omega = (tau_R + tau_a w_a) / tau                            scattering-weighted
//! p     = (tau_R p_Rayleigh + tau_a w_a p_HG(0.5)) / (tau_R + tau_a w_a)
//! ```
//!
//! with `p` a [`PhaseFunction::Tabulated`] on [`PHASE_NODES`] nodes, because neither member alone is
//! the mixture: at 410 nm the Rayleigh depth is twice the aerosol's and at 870 nm the aerosol's is
//! three times the Rayleigh, so a comparison run on either alone would be measuring the wrong
//! atmosphere at one end of the band or the other.
//!
//! Ozone is applied from outside as `exp(-tau_3(1/mu_s + 1/mu_v))` over the whole coupling, which is
//! exact for an absorber entirely above the scattering -- the layer sits at 20-25 km. `tau_3` is
//! RadCalNet's **measured** column times a published cross-section
//! (`tests/fixtures/ozone_cross_section.txt`), where `docs/verification.md` Section 5.1 carries the
//! same absorber as one *fitted* scalar per wavelength. Nothing about the ozone is free here, and
//! `the_chappuis_band_needs_the_measured_ozone_column` measures that dropping it costs a factor of
//! 8.0 between 540 and 690 nm.
//!
//! **Both sides of the error budget, not one.** A residual is only meaningful against a bar, and a
//! bar built from the reference's uncertainty alone charges our side nothing for the inputs it was
//! handed. RadCalNet publishes an uncertainty for every atmospheric input it publishes -- on these
//! scenes 0.001-0.012 in AOT(550), 0.002-0.30 in the Angstrom exponent, 5-10 DU of ozone and
//! 0.6-4 hPa of pressure -- and `the_error_budget_charges_our_own_side_for_the_inputs_it_was_given`
//! propagates all four through this model by moving each to the edge of its own published
//! uncertainty and re-running. The perturbed runs share the base run's seed, so the photon paths
//! cancel in the difference and 8000 photons resolve a derivative that 40 000 would be needed to
//! resolve as two independent means. It comes to 5.3e-4 of TOA reflectance -- **6.2 %** of the
//! quadrature sum, so the reference's uncertainty still dominates, but not by so much that our side
//! was fairly counted as zero. The same test measures the term's own sampling noise from a
//! four-times-photons control and confirms it inflates the bar by 0.006 %.
//!
//! **Which wavelengths are comparable is measured too, across the whole published grid.** A slab with
//! no gaseous absorption may only be compared where no gas absorbs, and
//! `radcalnets_own_data_says_which_windows_are_windows` decides that from RadCalNet's own spectra
//! with no radiative-transfer model in the loop, by regressing `-ln(rho_TOA/rho_BOA)` on the two-way
//! water column at each of the 211 wavelengths from 400 to 2500 nm. It finds 0.5 % of water
//! absorption at 870 and 1030 nm -- and 2.7 %, 3.3 % and 5.1 % at 1240, 1550 and 2130 nm, the three
//! wavelengths conventionally called SWIR windows. Those three are held out of every gate
//! ([`WEAK_BAND_NM`]) and reported as their own class. Reading the spectrum for notches by eye put
//! them in the window list in the first place; the regression is what took them out. The full grid
//! then says there was no better choice available: the cleanest wavelength beyond 1100 nm is 1630 nm
//! at 1.2 %, still twice 1030 nm's.
//!
//! That regression has one neglected term and it is bounded rather than ignored -- see
//! [`absorption_spectrum`], where pooling all 42 scenes without the bound returns a physically
//! impossible **-2.5 %** of water absorption at 870 nm.
//!
//! **The residuals.** Every point is measured at [`SEEDS`] independent streams, so each carries its
//! own standard error, and each is compared against the quadrature sum of that, RadCalNet's own
//! published uncertainty, and this model's response to the published uncertainty of its inputs. At
//! [`PHOTONS`] photons, over [`WINDOW_NM`] on the rows RadCalNet does not flag:
//!
//! | term | n | worst | RMS | signed mean | RMS bar | ratio |
//! |---|---|---|---|---|---|---|
//! | forward `rho_TOA` | 313 | 7.84e-3 | 1.99e-3 | +3.3e-4 | 8.53e-3 | **0.372** |
//! | inverse `rho_s`   | 313 | 9.42e-3 | 2.37e-3 | -3.8e-4 | 9.34e-3 | **0.456** |
//!
//! Both sit inside the band, and the inverse is the wider of the two because the correction divides
//! by `T_down T_up` and RadCalNet's surface uncertainty is smaller than its top-of-atmosphere one.
//! No single instrument is an outlier -- the forward ratio runs from 0.17 at Gobabeb to 0.44 at the
//! Baotou sand target, and every one of the six is asserted separately, because six sites averaged
//! together can hide one that does not work.
//!
//! And the residual climbs the absorption ladder it should:
//! `the_residual_climbs_the_absorption_ladder_it_is_supposed_to` measures 0.90 % relative in the
//! windows, 3.5 % (3.8x) in the weak SWIR bands and 130 % (145x) inside the O2-A and water bands,
//! always overpredicting where an absorber acts, because the slab cannot absorb.
//!
//! **Bounds of the claim.** Nadir only: RadCalNet simulates no other view and its data policy says
//! the data are "only valid for the conditions stated i.e. at nadir and times indicated", so nothing
//! here speaks for an off-nadir view or for a BRDF. Solar zenith runs to 81.5 degrees and the slab is
//! plane-parallel, so the largest air masses carry a sphericity error neither side models the same
//! way -- `the_plane_parallel_slab_survives_a_days_air_mass_sweep` bounds it on the days where the
//! atmosphere holds still while the sun moves, which is the only way to see it separately from the
//! aerosol. The two 30 m targets (Gobabeb, La Crau) are small enough that an adjacency effect is part
//! of their own error budget. The aerosol single-scattering albedo is fitted rather than known, so
//! any error correlated with it across the whole fixture is absorbed into it and is invisible here.
//! And nothing is claimed about gaseous absorption at all: the windows exclude it by construction,
//! and carrying a gas layer in [`Atmosphere`] is a separate piece of work.

#![allow(clippy::doc_markdown)] // RadCalNet, MODTRAN and the site names are proper nouns
#![allow(clippy::expect_used)] // a malformed fixture is a broken generator, and should stop the run
#![allow(clippy::cast_precision_loss)] // the counts are a few hundred rows

use core::num::NonZeroUsize;

use spacelab_earthvision::{
    Atmosphere, AtmosphericCoupling, KDistribution, PhaseFunction, TabulatedPhase,
    correlated_k_coupling, coupling_from_slab, gas_optical_depths, gas_optical_depths_binned,
};

const PHOTONS: u32 = 40_000;
const SEED: u64 = 20_260_819;

/// Photons in a sensitivity run, which is a *difference* against a base run at the same seed.
///
/// Common random numbers: the perturbed and base runs share their photon stream, so everything the
/// perturbation did not touch cancels and what is left is the derivative. Five times fewer photons
/// resolve it than would be needed to resolve two independent means to the same precision, which is
/// what makes charging our own side of the budget affordable at all;
/// `the_error_budget_charges_our_own_side_for_the_inputs_it_was_given` measures that this count is
/// enough by recomputing at four times it.
const PHOTONS_SENSITIVITY: u32 = 8_000;

/// Independent streams per scene, for the same reason `aeronet_measured_atmosphere` takes three: a
/// per-row `t` on two degrees of freedom has tails far too heavy to threshold, while an RMS over a
/// couple of hundred of them is stable to a few per cent.
const SEEDS: u64 = 3;

/// Nadir. RadCalNet simulates no other view, so the relative azimuth is degenerate and passing
/// anything but zero would be decoration.
const MU_VIEW: f64 = 1.0;

/// MODTRAN Card 3A2 `G` from RadCalNet's site default parameters, with Card 3A1 `IPH: 2` selecting
/// the Henyey-Greenstein form. Quoted from their deck, not chosen here.
const AEROSOL_ASYMMETRY: f64 = 0.5;

/// The aerosol single-scattering albedo, and the one number in this file that is fitted.
///
/// RadCalNet's deck says `NSSALB: 0`, which hands the aerosol single-scattering albedo to MODTRAN's
/// internal rural model and publishes nothing. It cannot be looked up, so it is recovered:
/// `the_unpublished_aerosol_albedo_is_the_one_thing_this_comparison_fits` checks that this is an
/// interior minimum, and `examples/radcalnet_error_budget.rs` runs the whole scan and the per-site
/// breakdown that shows one value serves all six instruments.
const AEROSOL_SSA: f64 = 0.88;

/// The aerosol type RadCalNet publishes per scene, and the reason the single fitted [`AEROSOL_SSA`]
/// is allowed to stand for all 42 of them. `R` is the rural model, MODTRAN's `IHAZE: 2`; a scene
/// marked urban, maritime or desert would carry different microphysics, and fitting one albedo
/// across the mixture would be describing two models with one number.
const AEROSOL_TYPE: &str = "R";

/// Nodes in the tabulated Rayleigh/aerosol mixture, uniform in `mu`.
///
/// The table is sampled by inverting the CDF of its *interpolant*, so the node count is an accuracy
/// knob; `the_tabulated_mixture_is_the_analytic_one_where_the_mixture_is_trivial` measures what this
/// one costs by driving the mixture to a pure Rayleigh and comparing against the analytic member.
const PHASE_NODES: usize = 401;

/// Molecules per cm^2 in one Dobson unit: 0.01 mm of pure ozone at 273.15 K and 1013.25 hPa, so the
/// CODATA 2018 Loschmidt constant 2.686 780 111e25 m^-3 times 1e-5 m, converted from m^-2 to cm^-2.
const DOBSON_PER_CM2: f64 = 2.686_780_111e25 * 1.0e-5 * 1.0e-4;

/// Standard pressure, the reference the tabulated Rayleigh depths are given at.
const STANDARD_PRESSURE_HPA: f64 = 1013.25;

/// Where a slab with no gaseous absorption may be compared to MODTRAN at all.
///
/// Chosen by measurement, not by reputation:
/// `radcalnets_own_data_says_which_windows_are_windows` regresses RadCalNet's own
/// `-ln(rho_TOA/rho_BOA)` on the two-way water column at all 211 published wavelengths and finds
/// under 0.2 % of water absorption at 870 and 1030 nm. The visible members are outside that
/// regression's domain (see its own bound) and are the wavelengths ozone and Rayleigh dominate, which
/// is the regime `docs/verification.md` Section 5.1 already covers from the other side.
const WINDOW_NM: [f64; 9] = [
    410.0, 440.0, 490.0, 550.0, 600.0, 660.0, 680.0, 870.0, 1030.0,
];

/// The three wavelengths conventionally called SWIR windows that this fixture shows are not.
///
/// At 10 nm resolution they carry 1.1 % (1240 nm) to 4.7 % (2130 nm) of water absorption over the
/// range of columns here, plus a water-independent term. Held out of every gate and reported
/// separately, because a slab that cannot absorb has nothing to say about them.
const WEAK_BAND_NM: [f64; 3] = [1240.0, 1550.0, 2130.0];

/// The far end of the ladder, carried so the exclusion can be measured rather than asserted: 760 nm
/// is the O2-A band and 940 nm a water band.
const STRONG_BAND_NM: [f64; 2] = [760.0, 940.0];

/// Below this the two-way-attenuation form the absorption regression fits has no column for the path
/// reflectance, which there is a large positive term. Those wavelengths are classified by the
/// Rayleigh and ozone physics of Section 5.1 instead.
const FIT_FLOOR_NM: f64 = 760.0;

/// Scenes a wavelength needs before its regression is read: three coefficients, and a design whose
/// water column has to span enough to separate them.
const MIN_SCENES: usize = 10;

/// One row in this many gets the four-times-photons control that measures the sensitivity term's own
/// sampling noise. That control is the single most expensive thing in this file, and seventy-odd rows
/// are enough for the RMS it feeds.
const CONTROL_STRIDE: usize = 4;

/// How much water absorption a wavelength may carry and still be called a window: half a per cent of
/// the beam at the wettest scene in the fixture, which is 3.37 cm of water at an air mass of 2.8.
/// The comparison's own residual in the windows is 1.4 %, so this is well inside it.
const WINDOW_WATER_LOSS: f64 = 0.005;

/// RadCalNet's fill convention, carried into both reflectance tables of `radcalnet_absorption.txt`.
const FILL: f64 = 9000.0;

/// How large the neglected path term `tau_scat / (4 mu_s rho_BOA)` may be before a scene leaves the
/// absorption regression. See [`absorption_spectrum`] for what happens without it.
const PATH_LIMIT: f64 = 0.08;

/// A measured input of ours, and the axis its published uncertainty is propagated along.
///
/// The columnar water vapour is deliberately absent. [`Atmosphere`] has no water channel at all, so
/// its uncertainty has nothing to move -- and that is not a gap in the budget but the reason
/// [`WINDOW_NM`] exists: the comparison is restricted to where water does not act, so the model's
/// deafness to it costs nothing there. The one place it would matter, [`WEAK_BAND_NM`], is held out
/// of every gate for exactly that reason.
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
    key: String,
    site: String,
    date: String,
    solar_zenith_deg: f64,
    wavelength_nm: f64,
    boa: f64,
    boa_unc: f64,
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
    /// Published, parsed, and used by nothing here -- see [`Input`] for why that is the point.
    water_cm_unc: f64,
    /// RadCalNet's own decoded uncertainty flag: `ok`, or `lut` where an input sat outside their
    /// uncertainty look-up table, or `band` inside one of their absorption regions.
    flag: String,
    /// RadCalNet's aerosol type for the scene, which selects the MODTRAN `IHAZE` model and with it
    /// the aerosol microphysics [`AEROSOL_SSA`] and [`AEROSOL_ASYMMETRY`] stand in for.
    aerosol_type: String,
}

impl Row {
    fn mu_sun(&self) -> f64 {
        self.solar_zenith_deg.to_radians().cos()
    }

    /// Aerosol optical depth at this wavelength, from the measured 550 nm value and Angstrom exponent.
    fn tau_aerosol(&self) -> f64 {
        self.aod550 * libm::pow(self.wavelength_nm / 550.0, -self.angstrom)
    }

    fn window(&self) -> bool {
        WINDOW_NM.contains(&self.wavelength_nm)
    }

    fn weak_band(&self) -> bool {
        WEAK_BAND_NM.contains(&self.wavelength_nm)
    }

    fn strong_band(&self) -> bool {
        STRONG_BAND_NM.contains(&self.wavelength_nm)
    }

    /// Two-way air mass through a plane-parallel atmosphere.
    fn air_mass(&self) -> f64 {
        1.0 / self.mu_sun() + 1.0 / MU_VIEW
    }

    /// Instrument and date, which is the unit a day's air-mass sweep is taken over.
    fn day(&self) -> String {
        format!("{} {}", self.site, self.date)
    }

    /// A row the gates use: a gas window, and an uncertainty RadCalNet itself considers in range.
    fn gated(&self) -> bool {
        self.window() && self.flag == "ok"
    }

    /// This row with one measured input moved to the edge of its own published uncertainty.
    ///
    /// Pressure moves through the Rayleigh depth, which the generator already scaled by the measured
    /// pressure, so the perturbation is the ratio of the two -- recomputing Bodhaine here would put a
    /// second Rayleigh implementation in the difference and measure that instead.
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

/// The 42 scenes [`AEROSOL_SSA`] was fitted on, and every residual in this file except one.
fn rows() -> Vec<Row> {
    parse_rows(include_str!("fixtures/radcalnet_toa.txt"))
}

/// The hold-out: days chosen by index through each site's archive, disjoint from [`rows`].
///
/// Same columns, written by the same generator code, so nothing here is comparing two formatters.
fn holdout() -> Vec<Row> {
    parse_rows(include_str!("fixtures/radcalnet_holdout.txt"))
}

fn parse_rows(text: &str) -> Vec<Row> {
    data(text)
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let parse = |index: usize| f[index].parse::<f64>().expect("a fixture number");
            // The generator drops a column whose published atmosphere carries a negative quantity,
            // because a negative ozone column makes `ozone_transmittance` return more than 1 and the
            // model add light. This fixture is generated, so the check belongs here too: a
            // regeneration that lost the rule would otherwise reintroduce it silently.
            assert!(
                parse(9) >= 0.0 && parse(12) > 0.0 && parse(13) >= 0.0 && parse(14) > 0.0,
                "{} {}: a negative published atmospheric quantity reached the fixture",
                f[0],
                f[1]
            );
            Row {
                key: format!("{} {} {} {}", f[0], f[1], f[2], f[4]),
                site: f[0].to_owned(),
                date: f[1].to_owned(),
                solar_zenith_deg: parse(3),
                wavelength_nm: parse(4),
                boa: parse(5),
                boa_unc: parse(6),
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
                flag: f[20].to_owned(),
                aerosol_type: f[21].to_owned(),
            }
        })
        .collect()
}

/// One column of the absorption fixture: the atmosphere the whole 400-2500 nm spectrum was taken
/// through, which is what the regression's design matrix is built from.
struct Scene {
    site: String,
    air_mass: f64,
    pressure_hpa: f64,
    aod550: f64,
    angstrom: f64,
    water_cm: f64,
    ozone_du: f64,
}

impl Scene {
    /// Solar cosine, from the two-way air mass this fixture publishes at nadir view.
    fn mu_sun(&self) -> f64 {
        1.0 / (self.air_mass - 1.0 / MU_VIEW)
    }
}

/// One published wavelength across every scene: the 1013.25 hPa Rayleigh depth, and RadCalNet's own
/// surface and top-of-atmosphere reflectance -- their two numbers, not a ratio derived from them,
/// because the ratio alone cannot say where it is readable.
struct Spectral {
    wavelength_nm: f64,
    tau_rayleigh_1013: f64,
    boa: Vec<f64>,
    toa: Vec<f64>,
}

/// The scenes and the two reflectance matrices, with no model of ours anywhere in them.
fn absorption() -> (Vec<Scene>, Vec<Spectral>) {
    let text = include_str!("fixtures/radcalnet_absorption.txt");
    let mut scenes = Vec::new();
    let (mut boa, mut toa) = (Vec::new(), Vec::new());
    for line in data(text) {
        let f: Vec<&str> = line.split_whitespace().collect();
        let parse = |index: usize| f[index].parse::<f64>().expect("a fixture number");
        match f[0] {
            "scene" => scenes.push(Scene {
                site: f[1].to_owned(),
                air_mass: parse(5),
                pressure_hpa: parse(6),
                aod550: parse(7),
                angstrom: parse(8),
                water_cm: parse(9),
                ozone_du: parse(10),
            }),
            "band" => {}
            kind => {
                let values: Vec<f64> = f[3..]
                    .iter()
                    .map(|v| v.parse::<f64>().expect("a reflectance"))
                    .collect();
                let row = (parse(1), parse(2), values);
                if kind == "boa" { &mut boa } else { &mut toa }.push(row);
            }
        }
    }
    assert_eq!(boa.len(), toa.len(), "the two matrices differ in length");
    let spectrum: Vec<Spectral> = boa
        .into_iter()
        .zip(toa)
        .map(|(down, up)| {
            assert!(
                same_wavelength(down.0, up.0),
                "the two matrices are tabulated on different wavelengths"
            );
            assert!(
                down.2.len() == scenes.len() && up.2.len() == scenes.len(),
                "a reflectance row and the scene list disagree on how many scenes there are"
            );
            Spectral {
                wavelength_nm: down.0,
                tau_rayleigh_1013: down.1,
                boa: down.2,
                toa: up.2,
            }
        })
        .collect();
    (scenes, spectrum)
}

/// RadCalNet's own absorption-region mask, one flag per published wavelength.
///
/// Specification section 3.5 marks a "predetermined absorption region" by adding 9 to the left of
/// the decimal point of the uncertainty, so it is their opinion about the wavelength rather than
/// about the scene -- and the generator refuses to write this block unless that is true in every
/// scene it downloaded. Read in a second pass over the same fixture because it answers a different
/// question from the reflectance matrices and shares nothing with them but the file.
fn radcalnet_absorption_regions() -> Vec<(f64, bool)> {
    let text = include_str!("fixtures/radcalnet_absorption.txt");
    data(text)
        .map(|line| line.split_whitespace().collect::<Vec<&str>>())
        .filter(|f| f[0] == "band")
        .map(|f| (f[1].parse::<f64>().expect("a wavelength"), f[2] == "1"))
        .collect()
}

/// The 223 K ozone cross-section, by wavelength, from the third fixture.
fn ozone_cross_section() -> Vec<(f64, f64)> {
    let text = include_str!("fixtures/ozone_cross_section.txt");
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

/// Two-way ozone transmittance above the scattering layer, `exp(-tau_3 (1/mu_s + 1/mu_v))`.
fn ozone_transmittance(row: &Row, sigma_cm2: f64) -> f64 {
    libm::exp(-sigma_cm2 * row.ozone_du * DOBSON_PER_CM2 * row.air_mass())
}

/// One channel of `gas_transmittance.txt`: one gas's exponential-sum fit at one wavelength.
struct GasChannel {
    gas: String,
    wavelength_nm: f64,
    /// Largest slant column the fit covers. Past it the channel is opaque and has to be refused.
    u_max: f64,
    weights: Vec<f64>,
    /// `rates[node][term]`, in the node order of [`GasTable::nodes`].
    rates: Vec<Vec<f64>>,
}

/// The gas table, and the pressure ladder its rates are tabulated on.
struct GasTable {
    /// Surface pressures, hPa, descending, exactly as the fixture's column header names them.
    nodes: Vec<f64>,
    channels: Vec<GasChannel>,
}

/// Read the gas table, taking the term count and the pressure ladder from the column header.
///
/// Both are read rather than hard-coded, because the generator is free to change either -- and the
/// failure mode of a hard-coded width is not a parse error but a silent column shift, which reads as
/// a physically-plausible wrong absorber. Reading the header means the file is understood or it is
/// rejected, with nothing in between.
fn gas_table() -> GasTable {
    let text = include_str!("fixtures/gas_transmittance.txt");
    let header = text
        .lines()
        .find(|line| line.starts_with("# gas wavelength_nm u_max "))
        .expect("the gas table's column header");
    let names: Vec<&str> = header.split_whitespace().skip(1).collect();
    let terms = names
        .iter()
        .filter(|name| {
            name.strip_prefix('w')
                .is_some_and(|rest| rest.parse::<usize>().is_ok())
        })
        .count();
    let nodes: Vec<f64> = names
        .iter()
        .filter_map(|name| name.strip_prefix("k1@"))
        .map(|hpa| hpa.parse::<f64>().expect("a node pressure"))
        .collect();
    assert!(terms > 0 && nodes.len() > 1, "the column header is not one");
    assert!(
        nodes.windows(2).all(|pair| pair[0] > pair[1]),
        "the pressure ladder is not descending"
    );
    let channels = data(text)
        .map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(
                f.len(),
                3 + terms * (1 + nodes.len()),
                "a gas row and the column header disagree on how wide a row is"
            );
            let number = |index: usize| f[index].parse::<f64>().expect("a fixture number");
            GasChannel {
                gas: f[0].to_owned(),
                wavelength_nm: number(1),
                u_max: number(2),
                weights: (0..terms).map(|j| number(3 + j)).collect(),
                rates: (0..nodes.len())
                    .map(|node| {
                        (0..terms)
                            .map(|j| number(3 + terms * (1 + node) + j))
                            .collect()
                    })
                    .collect(),
            }
        })
        .collect();
    GasTable { nodes, channels }
}

impl GasTable {
    /// The `(weight, rate)` terms for one gas at one wavelength, rates interpolated to `pressure_hpa`.
    ///
    /// `None` where the table declines to answer, which is three separate refusals and not one: the
    /// wavelength is not tabulated, the surface pressure is off the ends of the ladder, or the slant
    /// column `column * air_mass` is past this channel's own `u_max`. All three are the fixture
    /// header's instructions taken literally -- it says to interpolate between nodes and refuse off
    /// the ends, and that a caller handed a column past `u_max` must refuse the channel rather than
    /// extrapolate the row.
    fn terms(
        &self,
        gas: &str,
        wavelength_nm: f64,
        pressure_hpa: f64,
        column: f64,
        air_mass: f64,
    ) -> Result<Vec<(f64, f64)>, Refusal> {
        let channel = self
            .channels
            .iter()
            .find(|channel| {
                channel.gas == gas && same_wavelength(channel.wavelength_nm, wavelength_nm)
            })
            .ok_or(Refusal::NotTabulated)?;
        if column * air_mass > channel.u_max {
            return Err(Refusal::ColumnPastFit);
        }
        let last = self.nodes.len() - 1;
        if pressure_hpa > self.nodes[0] || pressure_hpa < self.nodes[last] {
            return Err(Refusal::PressureOffLadder);
        }
        let upper = (0..last)
            .find(|&i| pressure_hpa >= self.nodes[i + 1])
            .unwrap_or(last - 1);
        let (hi, lo) = (self.nodes[upper], self.nodes[upper + 1]);
        let t = (pressure_hpa - lo) / (hi - lo);
        Ok(channel
            .weights
            .iter()
            .enumerate()
            .map(|(j, &weight)| {
                let (above, below) = (channel.rates[upper][j], channel.rates[upper + 1][j]);
                (weight, below + t * (above - below))
            })
            .collect())
    }
}

/// Why the gas table declined to answer. Three separate facts, kept separate: one is a missing
/// wavelength, one is a ladder that does not reach this site, and one is a channel that is opaque at
/// this scene's column. Only the last is a property of the atmosphere rather than of the table, and
/// collapsing them to a `None` would let the second hide inside the third's count.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Refusal {
    NotTabulated,
    PressureOffLadder,
    ColumnPastFit,
}

/// The product k-distribution acting in one channel of one scene.
///
/// The vertical columns are written once each and used twice -- for the domain check and for the
/// distribution itself -- because a column that disagreed between the two would refuse the right
/// channels and then model the wrong ones.
fn scene_gas_terms(
    table: &GasTable,
    scene: &Scene,
    wavelength_nm: f64,
    bins: Option<NonZeroUsize>,
) -> Result<Vec<(f64, f64)>, Refusal> {
    // Water's column is a measurement RadCalNet publishes; every other gas is well mixed, so one
    // pressure ratio is its column and the table's per-gas unit absorbs the mixing ratio.
    let air = scene.pressure_hpa / STANDARD_PRESSURE_HPA;
    let columns = [
        ("H2O", scene.water_cm),
        ("CO2", air),
        ("N2O", air),
        ("CH4", air),
        ("O2", air),
    ];
    let mut fitted = Vec::new();
    for (gas, column) in columns {
        let terms = table.terms(
            gas,
            wavelength_nm,
            scene.pressure_hpa,
            column,
            scene.air_mass,
        )?;
        fitted.push((terms, column));
    }
    let gases: Vec<KDistribution<'_>> = fitted
        .iter()
        .map(|(terms, column)| KDistribution::new(terms, *column).expect("a k-distribution"))
        .collect();
    Ok(match bins {
        Some(bins) => gas_optical_depths_binned(&gases, bins),
        None => gas_optical_depths(&gases),
    })
}

/// Model TOA reflectance for one scene at one published wavelength.
///
/// Every per-wavelength input comes from a fixture: the Rayleigh depth and both reflectances from
/// RadCalNet, the ozone cross-section from the MPI-Mainz atlas, the gas terms from HITRAN. Nothing
/// here is fitted to this comparison, and the gas table in particular was built without RadCalNet
/// entering it at any point, so every one of the 211 wavelengths is out of sample for it.
fn scene_toa(
    scene: &Scene,
    band: &Spectral,
    boa: f64,
    sigma_ozone: f64,
    gas: &[(f64, f64)],
    photons: u32,
    seed: u64,
) -> f64 {
    let tau_r = band.tau_rayleigh_1013 * scene.pressure_hpa / STANDARD_PRESSURE_HPA;
    let tau_a = scene.aod550 * libm::pow(band.wavelength_nm / 550.0, -scene.angstrom);
    let (mu, mut value) = phase_mixture(tau_r, tau_a, AEROSOL_SSA);
    let mut cumulative = vec![0.0; mu.len()];
    TabulatedPhase::prepare(&mu, &mut value, &mut cumulative).expect("a phase mixture");
    let table = TabulatedPhase::new(&mu, &value, &cumulative).expect("a phase mixture");
    let atmosphere = Atmosphere {
        optical_depth: tau_r + tau_a,
        single_scattering_albedo: (tau_r + tau_a * AEROSOL_SSA) / (tau_r + tau_a),
        phase: PhaseFunction::Tabulated(table),
    };
    let ozone = libm::exp(-sigma_ozone * scene.ozone_du * DOBSON_PER_CM2 * scene.air_mass);
    ozone
        * correlated_k_coupling(
            &atmosphere,
            gas,
            scene.mu_sun(),
            MU_VIEW,
            0.0,
            photons,
            seed,
        )
        .toa_from_surface(boa)
}

/// Mean gaseous optical depth of a product distribution, `sum_p w_p tau_p`.
///
/// This is the fixture header's own weakness measure, computed from the row rather than from a
/// window list: it needs no threshold, no line-by-line pass and no hand classification, and it is
/// the quantity that decides which rung of the ladder a channel sits on below.
fn mean_gas_depth(terms: &[(f64, f64)]) -> f64 {
    terms.iter().map(|&(weight, depth)| weight * depth).sum()
}

/// The Rayleigh/aerosol mixture on the node grid, unnormalised — `prepare` rescales it.
fn mixture(row: &Row, aerosol_ssa: f64) -> (Vec<f64>, Vec<f64>) {
    phase_mixture(row.tau_rayleigh, row.tau_aerosol(), aerosol_ssa)
}

/// The same mixture from the two depths directly, for callers that are not a [`Row`].
fn phase_mixture(tau_rayleigh: f64, tau_aerosol: f64, aerosol_ssa: f64) -> (Vec<f64>, Vec<f64>) {
    let rayleigh = PhaseFunction::Rayleigh;
    let aerosol = PhaseFunction::HenyeyGreenstein {
        asymmetry: AEROSOL_ASYMMETRY,
    };
    let scale = 2.0 / (PHASE_NODES - 1) as f64;
    let mu: Vec<f64> = (0..PHASE_NODES).map(|i| -1.0 + scale * i as f64).collect();
    let value = mu
        .iter()
        .map(|&m| tau_rayleigh * rayleigh.value(m) + tau_aerosol * aerosol_ssa * aerosol.value(m))
        .collect();
    (mu, value)
}

/// The coupling this row's measured atmosphere produces, at one Monte-Carlo stream.
fn coupling(row: &Row, aerosol_ssa: f64, seed: u64, photons: u32) -> AtmosphericCoupling {
    let (tau_r, tau_a) = (row.tau_rayleigh, row.tau_aerosol());
    let (mu, mut value) = mixture(row, aerosol_ssa);
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

/// A seed per row, per draw. Both have to enter it, or the whole fixture would share one walk and
/// the RMS below would be one draw of a statistic rather than an average of three hundred.
fn stream(row: usize, draw: u64) -> u64 {
    SEED.wrapping_add((row as u64).wrapping_mul(0xff51_afd7_ed55_8ccd))
        .wrapping_add(draw.wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

/// Model TOA reflectance for one row: the coupling, then the ozone above it.
fn model_toa(row: &Row, sigma: f64, aerosol_ssa: f64, seed: u64, photons: u32) -> f64 {
    ozone_transmittance(row, sigma)
        * coupling(row, aerosol_ssa, seed, photons).toa_from_surface(row.boa)
}

/// This model's response to the published uncertainty of each measured input it reads, per input.
///
/// Each input is moved to the edge of its own uncertainty and the model is re-run **at the same
/// seed**, so the two runs share their photon paths and the difference is a derivative rather than
/// the difference of two noisy means. That is the whole reason [`PHOTONS_SENSITIVITY`] can be a fifth
/// of [`PHOTONS`]. The remaining noise is beaten down by averaging the *difference* over the
/// [`SEEDS`] streams -- averaging the difference, not the magnitude, because the magnitude of a noisy
/// difference is biased upward and would inflate the bar with noise.
fn input_terms(
    row: &Row,
    sigma: f64,
    index: usize,
    photons: u32,
    surface: bool,
) -> [f64; INPUTS.len()] {
    let value = |row: &Row, seed: u64| {
        if surface {
            coupling(row, AEROSOL_SSA, seed, photons)
                .surface_from_toa(row.toa / ozone_transmittance(row, sigma))
        } else {
            model_toa(row, sigma, AEROSOL_SSA, seed, photons)
        }
    };
    let moved: Vec<Row> = INPUTS
        .iter()
        .map(|(input, _)| row.perturbed(*input))
        .collect();
    let mut total = [0.0_f64; INPUTS.len()];
    for draw in 0..SEEDS {
        let seed = stream(index, draw);
        let base = value(row, seed);
        for (slot, perturbed) in moved.iter().enumerate() {
            total[slot] += value(perturbed, seed) - base;
        }
    }
    total.map(|d| d / SEEDS as f64)
}

/// This model's own share of the error budget, over all four inputs in quadrature.
fn input_sensitivity(row: &Row, sigma: f64, index: usize, photons: u32, surface: bool) -> f64 {
    libm::sqrt(
        input_terms(row, sigma, index, photons, surface)
            .iter()
            .map(|d| d * d)
            .sum::<f64>(),
    )
}

/// One comparison's disagreement with RadCalNet, and the bar it is measured against.
///
/// The residual is absolute reflectance because RadCalNet's uncertainty is; the bar is the quadrature
/// sum of three terms -- RadCalNet's published uncertainty, our own sampling standard error, and this
/// model's response to the published uncertainty of the inputs it was given. The third is what stops
/// the bar being one-sided.
#[derive(Default)]
struct Residual {
    site: Vec<String>,
    wavelength: Vec<f64>,
    signed: Vec<f64>,
    bar: Vec<f64>,
}

impl Residual {
    /// Fold in one point: `values` are the model at [`SEEDS`] streams, against `published +- unc`,
    /// with `ours` this model's own response to its inputs' uncertainty.
    fn push(&mut self, row: &Row, values: &[f64], published: f64, unc: f64, ours: f64) {
        let count = values.len() as f64;
        let mean = values.iter().sum::<f64>() / count;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (count - 1.0);
        self.site.push(row.site.clone());
        self.wavelength.push(row.wavelength_nm);
        self.signed.push(mean - published);
        self.bar
            .push(libm::sqrt(unc * unc + variance / count + ours * ours));
    }

    /// Signed mean and RMS over a subset, as a fraction of the mean bar over the same subset.
    fn subset(&self, keep: impl Fn(usize) -> bool) -> (usize, f64, f64, f64) {
        let index: Vec<usize> = (0..self.signed.len()).filter(|i| keep(*i)).collect();
        let n = index.len() as f64;
        let mean = index.iter().map(|&i| self.signed[i]).sum::<f64>() / n;
        let rms = libm::sqrt(index.iter().map(|&i| self.signed[i].powi(2)).sum::<f64>() / n);
        let bar = libm::sqrt(index.iter().map(|&i| self.bar[i].powi(2)).sum::<f64>() / n);
        (index.len(), mean, rms, rms / bar)
    }

    /// Every distinct site in the accumulator, in a stable order.
    fn sites(&self) -> Vec<String> {
        let mut sites = self.site.clone();
        sites.sort_unstable();
        sites.dedup();
        sites
    }

    /// The residual split by wavelength and by site.
    ///
    /// A residual is only diagnosable once it is broken down: a bias flat across the band is an
    /// aerosol or geometry error, one concentrated at particular wavelengths is an absorber the slab
    /// does not carry, and one concentrated at one instrument is that instrument's.
    fn breakdown(&self, name: &str) {
        let mut wavelengths = self.wavelength.clone();
        wavelengths.sort_by(f64::total_cmp);
        wavelengths.dedup();
        for w in wavelengths {
            let (n, mean, rms, ratio) = self.subset(|i| same_wavelength(self.wavelength[i], w));
            println!(
                "MEASURED {name} {w:6.0} nm  n {n:3}  mean {mean:+.4e} rms {rms:.4e} ratio {ratio:.2}"
            );
        }
        for site in self.sites() {
            let (n, mean, rms, ratio) = self.subset(|i| self.site[i] == site);
            println!(
                "MEASURED {name} {site:>9}  n {n:3}  mean {mean:+.4e} rms {rms:.4e} ratio {ratio:.2}"
            );
        }
    }

    /// The worst per-site ratio, so one instrument cannot hide inside an average over six.
    fn worst_site(&self) -> (String, f64) {
        self.sites()
            .into_iter()
            .map(|site| {
                let (_, _, _, ratio) = self.subset(|i| self.site[i] == site);
                (site, ratio)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .expect("at least one site")
    }

    /// Worst, RMS, signed mean and the RMS of residual-over-bar, printed and returned.
    fn report(&self, name: &str) -> (f64, f64, f64) {
        let n = self.signed.len() as f64;
        let worst = self.signed.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
        let rms = libm::sqrt(self.signed.iter().map(|e| e * e).sum::<f64>() / n);
        let mean = self.signed.iter().sum::<f64>() / n;
        let ratio = libm::sqrt(
            self.signed
                .iter()
                .zip(&self.bar)
                .map(|(e, b)| (e / b) * (e / b))
                .sum::<f64>()
                / n,
        );
        let bar = libm::sqrt(self.bar.iter().map(|b| b * b).sum::<f64>() / n);
        println!(
            "MEASURED {name:<18} n {:3}  worst {worst:.4e} rms {rms:.4e} mean {mean:+.4e}  \
             bar rms {bar:.4e}  ratio {ratio:.3}",
            self.signed.len(),
        );
        (worst, rms, ratio)
    }
}

/// Whether two wavelengths are the same fixture row.
///
/// All three fixtures tabulate a 10 nm grid of integers written out in full, so the parsed values are
/// exact and a tolerance is only here to keep the comparison honest if that ever stops being true.
fn same_wavelength(a: f64, b: f64) -> bool {
    (a - b).abs() < 0.5
}

/// The cross-section for a row's wavelength, from the third fixture.
fn sigma_for(table: &[(f64, f64)], row: &Row) -> f64 {
    sigma_at(table, row.wavelength_nm)
}

/// The cross-section at one wavelength, for callers that are not a [`Row`].
fn sigma_at(table: &[(f64, f64)], wavelength_nm: f64) -> f64 {
    table
        .iter()
        .find(|(tabulated, _)| same_wavelength(*tabulated, wavelength_nm))
        .map(|(_, sigma)| *sigma)
        .expect("every fixture wavelength has a cross-section")
}

/// Least squares for a three-column design, by Gauss-Jordan on the normal equations.
///
/// Three parameters over a dozen scenes on a well-scaled design; the normal equations lose a few
/// digits to squaring the condition number and there are eleven left.
fn fit3(design: &[[f64; 3]], observed: &[f64]) -> [f64; 3] {
    let mut m = [[0.0_f64; 4]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = design.iter().map(|row| row[i] * row[j]).sum();
        }
        m[i][3] = design
            .iter()
            .zip(observed)
            .map(|(row, y)| row[i] * y)
            .sum::<f64>();
    }
    for column in 0..3 {
        let pivot = (column..3)
            .max_by(|a, b| m[*a][column].abs().total_cmp(&m[*b][column].abs()))
            .unwrap_or(column);
        m.swap(column, pivot);
        for row in 0..3 {
            if row != column {
                let factor = m[row][column] / m[column][column];
                let pivot_row = m[column];
                for (target, source) in m[row].iter_mut().zip(&pivot_row).skip(column) {
                    *target -= factor * source;
                }
            }
        }
    }
    [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
}

/// What one wavelength's two-way attenuation says about the gas acting there.
struct Absorber {
    wavelength_nm: f64,
    scenes: usize,
    /// Fraction of the beam the fitted water term removes at the wettest scene the fit saw.
    water_loss: f64,
    /// The water-independent part, which is where a well-mixed gas such as oxygen or CO2 lands.
    constant: f64,
}

/// Regress `-ln(rho_TOA/rho_BOA)` on scattering depth and water column at every published wavelength.
///
/// No radiative-transfer model of ours enters this, so nothing it finds can be an error of ours.
/// Over a bright surface and a thin atmosphere the beam is attenuated twice and the path reflectance
/// is small, so RadCalNet's own spectra satisfy
///
/// ```text
/// -ln(rho_TOA / rho_BOA) = a m tau_scat + b m W + c
/// ```
///
/// with `m` the two-way air mass and `W` the measured water column. A significant `b` is water
/// absorption at that wavelength -- and a wavelength with water absorption is not a window, whatever
/// it is conventionally called.
///
/// **The neglected term, and why it is bounded rather than ignored.** That form has no column for the
/// path reflectance, which adds to `rho_TOA` without passing through the surface. Its size relative
/// to the signal is `tau_scat / (4 mu_s rho_BOA)` -- large in the blue, large over a dark target,
/// large at high aerosol load -- and it runs from 0.004 to 21 across this fixture. Left unbounded it
/// does not average out: it correlates with the instrument, and the instrument correlates with the
/// water column, so it lands in `b` with the wrong sign. Pooling all 42 scenes at 870 nm returns
/// **-2.5 %** of water absorption, which is not a number any atmosphere can produce. Bounding it at
/// [`PATH_LIMIT`] returns +1.0 % from the twelve scenes that survive, and the SWIR ordering
/// (1240 < 1550 < 2130 nm) holds across every bound from 0.03 to 0.12. That bound is also what makes
/// [`FIT_FLOOR_NM`] redundant in the visible, where it excludes nearly everything on its own.
fn absorption_spectrum() -> Vec<Absorber> {
    let (scenes, spectrum) = absorption();
    let mut measured = Vec::new();
    for band in spectrum {
        if band.wavelength_nm < FIT_FLOOR_NM {
            continue;
        }
        let tau = |i: usize| {
            band.tau_rayleigh_1013 * scenes[i].pressure_hpa / STANDARD_PRESSURE_HPA
                + scenes[i].aod550 * libm::pow(band.wavelength_nm / 550.0, -scenes[i].angstrom)
        };
        let usable: Vec<usize> = (0..scenes.len())
            .filter(|&i| {
                let mu_sun = 1.0 / (scenes[i].air_mass - 1.0 / MU_VIEW);
                // The positivity guard is not decoration: RadCalNet publishes a surface reflectance
                // of exactly 0.0000 at some SWIR wavelengths, and both the path ratio and the log
                // below are undefined there.
                band.boa[i] > 0.0
                    && band.boa[i] < FILL
                    && band.toa[i] > 0.0
                    && band.toa[i] < FILL
                    && tau(i) / (4.0 * mu_sun * band.boa[i]) < PATH_LIMIT
            })
            .collect();
        if usable.len() < MIN_SCENES {
            continue;
        }
        let design: Vec<[f64; 3]> = usable
            .iter()
            .map(|&i| {
                [
                    scenes[i].air_mass * tau(i),
                    scenes[i].air_mass * scenes[i].water_cm,
                    1.0,
                ]
            })
            .collect();
        let observed: Vec<f64> = usable
            .iter()
            .map(|&i| -libm::log(band.toa[i] / band.boa[i]))
            .collect();
        let [_, water, constant] = fit3(&design, &observed);
        let wettest = usable
            .iter()
            .map(|&i| scenes[i].air_mass * scenes[i].water_cm)
            .fold(0.0_f64, f64::max);
        measured.push(Absorber {
            wavelength_nm: band.wavelength_nm,
            scenes: usable.len(),
            water_loss: 1.0 - libm::exp(-water * wettest),
            constant,
        });
    }
    measured
}

#[test]
fn the_two_fixtures_are_tabulated_on_the_same_wavelengths() {
    let table = ozone_cross_section();
    let mut wavelengths: Vec<f64> = rows().iter().map(|row| row.wavelength_nm).collect();
    wavelengths.sort_by(f64::total_cmp);
    wavelengths.dedup();
    for wavelength in &wavelengths {
        assert!(
            table.iter().any(|(w, _)| same_wavelength(*w, *wavelength)),
            "{wavelength} nm has RadCalNet data and no ozone cross-section"
        );
    }
    // And the absorption fixture has to reach every wavelength the comparison uses, or the window
    // classification would be speaking about a grid the residuals are not measured on.
    let spectrum = absorption().1;
    for wavelength in &wavelengths {
        assert!(
            spectrum
                .iter()
                .any(|band| same_wavelength(band.wavelength_nm, *wavelength)),
            "{wavelength} nm is compared but is not in the absorption spectrum"
        );
    }
    // The generators keep the grid in three places on purpose, so each is runnable alone. This is
    // what makes that safe: a grid edited in one and not the others stops the build. The ozone table
    // is tabulated on RadCalNet's whole published grid rather than on the 14 gated wavelengths,
    // because a comparison at 211 wavelengths needs every per-wavelength input at 211 -- and the
    // Chappuis band sits in the middle of the visible, so a 14-point ozone table would have excluded
    // most of the windows.
    assert_eq!(
        table.len(),
        spectrum.len(),
        "the ozone table and the absorption spectrum tabulate different wavelengths"
    );
    for (band, (wavelength, _)) in spectrum.iter().zip(&table) {
        assert!(
            same_wavelength(band.wavelength_nm, *wavelength),
            "{} nm in the absorption spectrum meets {wavelength} nm in the ozone table",
            band.wavelength_nm
        );
    }
}

#[test]
fn our_slab_reproduces_radcalnets_modtran_toa_on_measured_scenes() {
    let table = ozone_cross_section();
    let mut forward = Residual::default();
    for (index, row) in rows().iter().enumerate().filter(|(_, r)| r.gated()) {
        let sigma = sigma_for(&table, row);
        let draws: Vec<f64> = (0..SEEDS)
            .map(|draw| model_toa(row, sigma, AEROSOL_SSA, stream(index, draw), PHOTONS))
            .collect();
        let ours = input_sensitivity(row, sigma, index, PHOTONS_SENSITIVITY, false);
        forward.push(row, &draws, row.toa, row.toa_unc, ours);
    }
    forward.breakdown("forward");
    let (worst, rms, ratio) = forward.report("forward TOA");
    let (site, worst_site) = forward.worst_site();

    // The gate is the ratio, as everywhere else in this crate. A residual smaller than the
    // quadrature sum of RadCalNet's own uncertainty, our sampling error and our response to the
    // uncertainty of our inputs is the most agreement either side can claim; a larger one is a
    // finding to chase, not a tolerance to widen.
    assert!(ratio < 1.0, "forward TOA residual over its bar: {ratio:.3}");
    // And per instrument, because six sites averaged together can hide one that does not work: the
    // targets run from a 1 km playa to a 30 m gravel disk to a site 2824 m up.
    assert!(
        worst_site < 1.0,
        "{site} alone sits at {worst_site:.3} of its bar"
    );
    // The magnitudes the module docs publish, so that both growing together cannot pass unnoticed.
    assert!(worst < 1.2e-2, "forward TOA worst residual {worst:.4e}");
    assert!(rms < 3.0e-3, "forward TOA RMS residual {rms:.4e}");
}

/// The inverse over one fixture: RadCalNet's TOA reflectance in, the ground measurement out.
///
/// Shared by the fitted fixture and the hold-out so the two are the same measurement rather than two
/// copies of one, for the same reason the generator writes both files through a single
/// `comparison_rows` -- a hold-out scored by a second copy of this loop would be testing the copy.
fn inverse_residual(rows: &[Row], table: &[(f64, f64)], label: &str) -> Residual {
    let mut inverse = Residual::default();
    for (index, row) in rows.iter().enumerate().filter(|(_, r)| r.gated()) {
        let sigma = sigma_for(table, row);
        let draws: Vec<f64> = (0..SEEDS)
            .map(|draw| {
                coupling(row, AEROSOL_SSA, stream(index, draw), PHOTONS)
                    .surface_from_toa(row.toa / ozone_transmittance(row, sigma))
            })
            .collect();
        // The inverse reads the same atmosphere, so it carries the same input uncertainty; measured
        // on the surface side, where the coupling divides it by T_down T_up.
        let ours = input_sensitivity(row, sigma, index, PHOTONS_SENSITIVITY, true);
        inverse.push(row, &draws, row.boa, row.boa_unc, ours);
    }
    inverse.breakdown(label);
    inverse
}

#[test]
fn the_correction_recovers_the_surface_reflectance_that_was_measured_on_the_ground() {
    let inverse = inverse_residual(&rows(), &ozone_cross_section(), "inverse");
    let (worst, rms, ratio) = inverse.report("inverse surface");
    let (site, worst_site) = inverse.worst_site();

    // This is the number an Earth-observation user cares about: RadCalNet's top-of-atmosphere
    // reflectance in, the reflectance measured on the ground out.
    assert!(
        ratio < 1.0,
        "inverse surface residual over its bar: {ratio:.3}"
    );
    assert!(
        worst_site < 1.0,
        "{site} alone sits at {worst_site:.3} of its bar"
    );
    assert!(worst < 1.4e-2, "inverse surface worst residual {worst:.4e}");
    assert!(rms < 3.5e-3, "inverse surface RMS residual {rms:.4e}");
}

#[test]
fn the_correction_recovers_the_ground_on_days_nobody_chose() {
    // The test above is the number a user cares about, and it is measured entirely inside the 42
    // scenes AEROSOL_SSA was fitted on. The forward direction stopped being an in-sample statement
    // when the hold-out arrived; the inverse did not follow it until now, which left the direction
    // that matters most as the only one still reported where the free parameter was fitted.
    let inverse = inverse_residual(&holdout(), &ozone_cross_section(), "holdout inverse");
    let (worst, rms, ratio) = inverse.report("holdout inverse surface");
    let (site, worst_site) = inverse.worst_site();

    // The ratio and nothing else. The in-sample test above also gates `worst` and `rms` against
    // magnitudes measured on its own 313 comparisons; carrying those over to 1003 comparisons drawn
    // from thirteen years of weather would be a tolerance fitted to the sample it is meant to
    // judge. They are printed instead, and a ratio above 1.0 here is a finding about the fit -- the
    // failure this fixture exists to detect -- rather than a number to widen.
    assert!(
        ratio < 1.0,
        "out of sample the surface residual sits at {ratio:.3} of its bar, so the correction does \
         not generalise off the days the albedo was fitted on"
    );
    assert!(
        worst_site < 1.0,
        "{site} alone sits at {worst_site:.3} of its bar out of sample"
    );
    println!("MEASURED holdout inverse worst {worst:.4e} rms {rms:.4e}, reported and not gated");
}

#[test]
fn the_error_budget_charges_our_own_side_for_the_inputs_it_was_given() {
    // Two things have to be true for the third term of the bar to mean anything: it has to be a
    // derivative rather than sampling noise, and it has to be large enough that leaving it out was
    // an understatement of the budget rather than a rounding choice.
    let table = ozone_cross_section();
    let mut total = [0.0_f64; INPUTS.len()];
    let mut cheap: Vec<f64> = Vec::new();
    let mut dear: Vec<(usize, f64)> = Vec::new();
    let (mut published, mut count) = (0.0, 0.0);
    for (index, row) in rows().iter().enumerate().filter(|(_, r)| r.gated()) {
        let sigma = sigma_for(&table, row);
        let terms = input_terms(row, sigma, index, PHOTONS_SENSITIVITY, false);
        for (slot, term) in terms.iter().enumerate() {
            total[slot] += term * term;
        }
        cheap.push(libm::sqrt(terms.iter().map(|d| d * d).sum::<f64>()));
        // The four-times-photons control is what makes the noise measurable, and it is also four
        // times the cost of everything else in this file put together. Every CONTROL_STRIDE-th row
        // is enough for an RMS over seventy-odd of them, and the rows it lands on are a fixed
        // stride rather than a choice.
        if cheap.len() % CONTROL_STRIDE == 1 {
            dear.push((
                cheap.len() - 1,
                input_sensitivity(row, sigma, index, 4 * PHOTONS_SENSITIVITY, false),
            ));
        }
        published += row.toa_unc * row.toa_unc;
        count += 1.0;
    }
    for (slot, (_, name)) in INPUTS.iter().enumerate() {
        println!(
            "MEASURED budget {name:>9}  rms {:.4e} of TOA reflectance",
            libm::sqrt(total[slot] / count)
        );
    }
    let rms = |v: &[f64]| libm::sqrt(v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64);
    let (ours, reference) = (rms(&cheap), libm::sqrt(published / count));
    let drift = rms(&dear
        .iter()
        .map(|(index, value)| cheap[*index] - value)
        .collect::<Vec<f64>>());
    // What the drift says about the noise. The two estimates are independent draws at N and 4N
    // photons, so their standard errors are `s` and `s/2` and the RMS of their difference is
    // `s sqrt(1 + 1/4)`. Solving for `s` gives the sampling noise the cheap term carries, which
    // enters the bar in quadrature and so inflates it -- by an amount worth computing rather than
    // bounding through a proxy on the term itself.
    let noise = drift / libm::sqrt(1.25);
    let bar = libm::sqrt(reference * reference + ours * ours);
    let inflation = libm::sqrt(bar * bar + noise * noise) / bar - 1.0;
    println!(
        "MEASURED budget: ours {ours:.4e} against RadCalNet's {reference:.4e} \
         ({:.1} % of the quadrature sum); the {PHOTONS_SENSITIVITY}-photon term carries \
         {noise:.4e} of sampling noise, which inflates the bar by {:.4} %",
        100.0 * ours / bar,
        100.0 * inflation,
    );

    // Noise added to the bar weakens every gate in this file rather than describing it, so the
    // question is not whether the term is noisy -- at a fifth of the photons it is -- but whether
    // its noise moves the bar the gates are measured against.
    assert!(
        inflation < 5.0e-3,
        "the sensitivity term's own sampling noise {noise:.4e} inflates the residual bar by \
         {:.3} %, which is too much of the bar to pass as an accounting term",
        100.0 * inflation
    );
    // And it is not negligible: leaving it out was charging our side of the comparison nothing.
    assert!(
        ours > 0.05 * reference,
        "our own input sensitivity {ours:.4e} is below a twentieth of RadCalNet's {reference:.4e}, so \
         the one-sided budget this test exists to fix was not one-sided after all"
    );
}

/// The whole fitted spectrum as a ladder of contiguous runs rather than 175 lines, then the
/// wavelengths the gates name. A run breaks where the classification flips or where the usable grid
/// has a gap, so a printed interval is a real interval and not two of them joined across a
/// wavelength nobody could fit.
fn report_spectrum(measured: &[Absorber]) {
    let clean = |a: &Absorber| a.water_loss.abs() < WINDOW_WATER_LOSS;
    let mut start = 0;
    for i in 0..measured.len() {
        let last = i + 1 == measured.len();
        if !last
            && clean(&measured[i + 1]) == clean(&measured[start])
            && measured[i + 1].wavelength_nm - measured[i].wavelength_nm < 15.0
        {
            continue;
        }
        let worst = measured[start..=i]
            .iter()
            .map(|a| a.water_loss.abs())
            .fold(0.0_f64, f64::max);
        println!(
            "MEASURED spectrum {:6.0}-{:6.0} nm  {:>6}  worst water loss {:5.2} %",
            measured[start].wavelength_nm,
            measured[i].wavelength_nm,
            if clean(&measured[start]) {
                "window"
            } else {
                "band"
            },
            100.0 * worst
        );
        start = i + 1;
    }
    for w in WINDOW_NM
        .iter()
        .chain(&WEAK_BAND_NM)
        .chain(&STRONG_BAND_NM)
        .filter(|w| **w >= FIT_FLOOR_NM)
    {
        let a = measured
            .iter()
            .find(|a| same_wavelength(a.wavelength_nm, *w))
            .expect("a fitted wavelength");
        println!(
            "MEASURED absorption {w:6.0} nm  n {:2}  water loss at the wettest scene {:+.3} % \
             constant {:+.5}",
            a.scenes,
            100.0 * a.water_loss,
            a.constant
        );
    }
}

#[test]
fn radcalnets_own_data_says_which_windows_are_windows() {
    // **Bound.** Below FIT_FLOOR_NM the path reflectance is a large positive term the fitted form has
    // no column for, so the fit there is meaningless and is not run: those wavelengths are classified
    // by the Rayleigh and ozone physics Section 5.1 covers, not by this regression.
    let measured = absorption_spectrum();
    let at = |w: f64| {
        measured
            .iter()
            .find(|a| same_wavelength(a.wavelength_nm, w))
            .expect("a fitted wavelength")
    };
    report_spectrum(&measured);

    // 940 nm is a water band and must dominate everything.
    assert!(
        at(940.0).water_loss > 0.10,
        "940 nm: {:.3} %",
        100.0 * at(940.0).water_loss
    );
    // 760 nm is the O2-A band, and oxygen is well mixed, so its absorption has to land in the
    // *constant* rather than in the water term. It does, by two orders of magnitude over a real
    // window's -- which is the check that the two columns are separable at all.
    //
    // The water term at 760 nm is *not* asserted to be zero, and measures a few per cent. At 10 nm
    // resolution the O2-A band sits between the 720 and 820 nm water bands and the fit cannot
    // separate the shoulder from the oxygen; that is a reason 760 nm belongs in STRONG_BAND_NM and
    // not a defect in the regression.
    let window_constant = at(870.0).constant.abs().max(at(1030.0).constant.abs());
    assert!(
        at(760.0).constant > 0.20 && at(760.0).constant > 10.0 * window_constant,
        "760 nm constant {:.4} against {window_constant:.4} in a window: the O2-A band is not \
         separating from the windows",
        at(760.0).constant
    );
    // The two NIR members of WINDOW_NM earn their place here. The bar is looser than
    // WINDOW_WATER_LOSS because the path bound leaves 870 nm only twelve scenes -- the dark gravel
    // targets are exactly the ones it excludes -- and the fit's own resolution there is about a per
    // cent. What carries the argument is not this number but the gap to the SWIR members below.
    for w in [870.0, 1030.0] {
        assert!(
            at(w).water_loss.abs() < 0.015,
            "{w} nm water: {:.3} % over {} scenes",
            100.0 * at(w).water_loss,
            at(w).scenes
        );
    }
    // And the three conventional SWIR windows do not: each carries at least 2 % of water absorption
    // in absolute terms and at least twice the cleanest genuine window, which is why WEAK_BAND_NM is
    // held out of every gate.
    let clean = at(870.0).water_loss.abs().max(at(1030.0).water_loss.abs());
    for w in WEAK_BAND_NM {
        assert!(
            at(w).water_loss > 0.02 && at(w).water_loss > 2.0 * clean,
            "{w} nm carries {:.3} % of water absorption against {:.3} % in a real window, so it \
             does not belong in WEAK_BAND_NM",
            100.0 * at(w).water_loss,
            100.0 * clean
        );
    }
    // The full grid is here so the choice can be checked against every alternative rather than
    // against the handful somebody thought of: no wavelength beyond 1100 nm reaches a real window's
    // cleanliness, so there is no SWIR window to have picked instead.
    let best_swir = measured
        .iter()
        .filter(|a| a.wavelength_nm > 1100.0)
        .min_by(|a, b| a.water_loss.abs().total_cmp(&b.water_loss.abs()))
        .expect("the SWIR is fitted");
    println!(
        "MEASURED the cleanest wavelength beyond 1100 nm is {:.0} nm at {:.3} % of water \
         absorption, against {:.3} % at 1030 nm",
        best_swir.wavelength_nm,
        100.0 * best_swir.water_loss.abs(),
        100.0 * at(1030.0).water_loss.abs()
    );
    assert!(
        best_swir.water_loss.abs() > at(1030.0).water_loss.abs(),
        "{:.0} nm is cleaner than 1030 nm and WINDOW_NM does not use it",
        best_swir.wavelength_nm
    );

    // And the SWIR verdict has to come from more than one instrument, or it would be measuring one
    // radiometer's calibration rather than the atmosphere. Only three of the six sites publish
    // beyond 1 um, so this is the exposed end of the fixture and the one worth asserting.
    let (scenes, spectrum) = absorption();
    let instruments: std::collections::BTreeSet<&str> = spectrum
        .iter()
        .find(|band| same_wavelength(band.wavelength_nm, 1550.0))
        .map(|band| {
            (0..scenes.len())
                .filter(|&i| band.boa[i] < FILL && band.toa[i] < FILL)
                .map(|i| scenes[i].site.as_str())
                .collect()
        })
        .expect("1550 nm is published");
    println!("MEASURED the SWIR classification draws on {instruments:?}");
    assert!(
        instruments.len() >= 3,
        "the SWIR is fitted from {} instrument(s), which cannot separate an atmosphere from a \
         calibration",
        instruments.len()
    );
}

#[test]
fn radcalnets_own_absorption_flag_never_contradicts_the_bands_we_measure() {
    // Two criteria, not two opinions on one. Specification section 3.5 marks a "predetermined
    // absorption region" where RadCalNet's own *retrieved uncertainty* exceeds ten per cent, which
    // happens only where the band is close to opaque; the regression above marks a wavelength where
    // water absorption is measurable at all. Theirs therefore has to be the narrower set, and what is
    // checked is containment. Equality would be the wrong assertion and would fail for a good reason.
    let mask = radcalnet_absorption_regions();
    let measured = absorption_spectrum();
    let ours = |w: f64| {
        measured
            .iter()
            .find(|a| same_wavelength(a.wavelength_nm, w))
    };
    let flagged: Vec<f64> = mask.iter().filter(|(_, on)| *on).map(|(w, _)| *w).collect();

    // Three outcomes, and only the third is a finding. A flagged wavelength we could not fit is not a
    // disagreement: those are the band cores where RadCalNet's own reflectance is a fill value, so
    // there is nothing for the regression to read and the two methods never meet.
    let (mut agreed, mut unfittable, mut contradicted) = (Vec::new(), Vec::new(), Vec::new());
    for &w in &flagged {
        match ours(w) {
            None => unfittable.push(w),
            Some(a) if a.water_loss.abs() >= WINDOW_WATER_LOSS => agreed.push((w, a.water_loss)),
            Some(a) => contradicted.push((w, a.water_loss, a.scenes)),
        }
    }
    println!(
        "MEASURED RadCalNet flags {} of {} published wavelengths, {:.0}-{:.0} nm: {} we also \
         measure as bands, {} we cannot fit because their own reflectance is fill there, {} \
         contradicted",
        flagged.len(),
        mask.len(),
        flagged.first().copied().unwrap_or(f64::NAN),
        flagged.last().copied().unwrap_or(f64::NAN),
        agreed.len(),
        unfittable.len(),
        contradicted.len()
    );
    for (w, loss) in &agreed {
        println!(
            "MEASURED   agreed {w:6.0} nm  water loss {:+.2} %",
            100.0 * loss
        );
    }

    // Non-vacuity, first: a mask read as all-false would satisfy containment trivially. It is only
    // evidence about our classification where the two sets can actually meet.
    assert!(
        !agreed.is_empty(),
        "no flagged wavelength is fitted, so the containment below is vacuous and this test is \
         measuring the parser rather than the classification"
    );
    assert!(
        contradicted.is_empty(),
        "RadCalNet calls {contradicted:?} (nm, water loss, scenes) a predetermined absorption \
         region and the regression calls it a window: one of the two methods is wrong, and which \
         one is a question to answer rather than a tolerance to widen"
    );

    // And the gap, which is the reason the window list had to be measured rather than taken from
    // this flag. Every one of these removes water from the beam by our regression and none of them
    // is an absorption region to RadCalNet -- their threshold is on the retrieval, not on the gas.
    for w in WEAK_BAND_NM.iter().chain(&STRONG_BAND_NM) {
        let Some(a) = ours(*w) else { continue };
        let theirs = flagged.iter().any(|f| same_wavelength(*f, *w));
        println!(
            "MEASURED   gap    {w:6.0} nm  water loss {:+.2} %  RadCalNet absorption region: {}",
            100.0 * a.water_loss,
            if theirs { "yes" } else { "no" }
        );
        assert!(
            !theirs,
            "{w} nm is on both lists, so the two criteria are not the strict-subset pair this test \
             claims and the reported gap is not the gap"
        );
    }
}

#[test]
fn the_residual_climbs_the_absorption_ladder_it_is_supposed_to() {
    let table = ozone_cross_section();
    let (mut window, mut weak, mut strong) = (Vec::new(), Vec::new(), Vec::new());
    for (index, row) in rows().iter().enumerate() {
        if row.window() && row.flag != "ok" {
            continue;
        }
        let sigma = sigma_for(&table, row);
        let residual =
            (model_toa(row, sigma, AEROSOL_SSA, stream(index, 0), PHOTONS) - row.toa) / row.toa;
        if row.strong_band() {
            strong.push(residual);
        } else if row.weak_band() {
            weak.push(residual);
        } else {
            window.push(residual);
        }
    }
    let rms = |v: &[f64]| libm::sqrt(v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64);
    let (in_window, in_weak, in_strong) = (rms(&window), rms(&weak), rms(&strong));
    println!(
        "MEASURED ladder: window rms {in_window:.4} over {} points, weak band {in_weak:.4} over {} \
         points ({:.1}x), strong band {in_strong:.4} over {} points ({:.0}x)",
        window.len(),
        weak.len(),
        in_weak / in_window,
        strong.len(),
        in_strong / in_window,
    );
    // The exclusions are measured rather than asserted, and floors are asserted rather than values,
    // so a better sampler cannot fail this test and only a wrong classification can.
    assert!(
        in_weak > 1.5 * in_window && in_strong > 10.0 * in_weak,
        "the three classes do not separate: {in_window:.4} / {in_weak:.4} / {in_strong:.4}"
    );
    assert!(
        weak.iter().chain(&strong).all(|residual| *residual > 0.0),
        "a slab that cannot absorb must overpredict wherever an absorber acts"
    );
}

/// Photons per *product term* of the gas distribution.
///
/// Splitting a photon budget across the terms costs nothing in accuracy: the weighted sum of
/// independent means has variance `sum_p w_p^2 sigma^2 / n_p`, so sixteen terms at `n` photons each
/// carry the noise of one run at `16 n` when the weights are comparable. What it costs is time, one
/// transport per term, which is why the suite sweeps a stride through the scenes and
/// `examples/radcalnet_error_budget.rs` sweeps all of them.
const PHOTONS_SPECTRAL: u32 = 1_200;

/// Stride through the fixture's scene list for the suite's own 211-wavelength sweep.
const SPECTRAL_STRIDE: usize = 6;

/// How many terms one channel's five-gas product is resorted onto before the transport runs.
///
/// Not a tuning knob and not a guess. The exact product of the shipped table is a median of 18816
/// terms per channel and 368640 at its worst, so one sweep of the 211 published wavelengths is
/// 5575735 transport runs *per scene* -- a test that never finishes rather than a slow one.
///
/// What 256 bins costs is measured against the exact product, channel by channel, by
/// `the_resort_costs_less_than_the_photon_noise_it_replaces`: **3.9e-3** of channel transmittance at
/// worst, one-signed, in the strong water bands, for 211 terms per channel against the exact 26636.
/// The bar it is held against is this file's own window rung -- 1.52e-2 rms with no gas in the
/// transport at all, which is what the sweep can resolve -- so the resort sits at a third of the
/// noise it hides inside, and two orders below the 130 % strong-band residual the table exists to
/// remove. The measurement is the reason for the number; it is not chosen to make a tolerance pass.
const GAS_BINS: Option<NonZeroUsize> = NonZeroUsize::new(256);

/// Mean gaseous optical depth separating a window from a band, and a band from a strong one.
///
/// Taken from the table rather than from a hand-written wavelength list: the fixture header's own
/// `<tau> = sum_j w_j k_j u`, evaluated at the column the scene actually carries. That matters here
/// because the classification has to mean the same thing at all 211 wavelengths, where no hand list
/// exists, and because a threshold on a quantity the fixture computes cannot drift away from it.
const GAS_WINDOW_DEPTH: f64 = 1.0e-3;
const GAS_STRONG_DEPTH: f64 = 1.0e-1;

/// Every published wavelength, with the gas table inside the transport and without it.
///
/// This is the out-of-sample claim in its strongest form. The gas table was fitted to HITRAN line
/// parameters and nothing else -- no RadCalNet quantity enters its generator at any point -- so all
/// 211 wavelengths and all 42 scenes are out of sample for it, not merely the ones no parameter was
/// tuned on. `AEROSOL_SSA`, the one fitted number in this file, was fitted on 42 scenes at 14 window
/// wavelengths where by construction no gas absorbs.
///
/// The two runs share their seed and their total photon budget, so the difference between them is
/// the gas layer and not a different Monte-Carlo draw or a different amount of effort.
/// What one 211-wavelength sweep collected, so the loop that gathers it and the assertions that read
/// it are separate functions rather than one long test.
struct Sweep {
    /// Relative TOA residual per rung, with the gas layer in the transport and without it.
    with: [Vec<f64>; 3],
    without: [Vec<f64>; 3],
    /// `(wavelength, residual)` of the largest residual each rung saw, with the gas layer in.
    worst: [(f64, f64); 3],
    /// Counted by [`Refusal`], indexed by its discriminant.
    refused: [usize; 3],
    /// Channels skipped because RadCalNet published a fill value, and scenes actually visited.
    filled: usize,
    scenes: usize,
    /// Published wavelengths the sweep ran over, so the coverage accounting can be closed rather
    /// than compared against a number someone typed.
    channels: usize,
    /// What the table costs to use: transport runs summed over channels, and the worst single
    /// channel. Five absorbers multiply, so the product term count is the number of runs one channel
    /// takes. Nothing is capped -- a silent prune is indistinguishable in the output from a band that
    /// was never in the table -- so this is reported rather than left to be inferred from a clock.
    transports: usize,
    worst_terms: (usize, f64),
}

fn sweep_every_published_wavelength() -> Sweep {
    let table = gas_table();
    let ozone = ozone_cross_section();
    let (scenes, spectrum) = absorption();
    let mut sweep = Sweep {
        with: [Vec::new(), Vec::new(), Vec::new()],
        without: [Vec::new(), Vec::new(), Vec::new()],
        worst: [(0.0, 0.0); 3],
        refused: [0; 3],
        filled: 0,
        scenes: scenes.len().div_ceil(SPECTRAL_STRIDE),
        channels: spectrum.len(),
        transports: 0,
        worst_terms: (0, 0.0),
    };
    for (index, scene) in scenes.iter().enumerate().step_by(SPECTRAL_STRIDE) {
        for (band_index, band) in spectrum.iter().enumerate() {
            let (boa, toa) = (band.boa[index], band.toa[index]);
            if boa >= FILL || toa >= FILL || boa <= 0.0 || toa <= 0.0 {
                sweep.filled += 1;
                continue;
            }
            let gas = match scene_gas_terms(&table, scene, band.wavelength_nm, GAS_BINS) {
                Ok(gas) => gas,
                Err(why) => {
                    sweep.refused[why as usize] += 1;
                    continue;
                }
            };
            let sigma = sigma_at(&ozone, band.wavelength_nm);
            let seed = stream(index * spectrum.len() + band_index, 0);
            let terms = u32::try_from(gas.len()).expect("a term count");
            sweep.transports += gas.len();
            if gas.len() > sweep.worst_terms.0 {
                sweep.worst_terms = (gas.len(), band.wavelength_nm);
            }
            let residual = |model: f64| (model - toa) / toa;
            let depth = mean_gas_depth(&gas);
            let rung =
                usize::from(depth >= GAS_WINDOW_DEPTH) + usize::from(depth >= GAS_STRONG_DEPTH);
            let modelled = residual(scene_toa(
                scene,
                band,
                boa,
                sigma,
                &gas,
                PHOTONS_SPECTRAL,
                seed,
            ));
            if libm::fabs(modelled) > libm::fabs(sweep.worst[rung].1) {
                sweep.worst[rung] = (band.wavelength_nm, modelled);
            }
            sweep.with[rung].push(modelled);
            sweep.without[rung].push(residual(scene_toa(
                scene,
                band,
                boa,
                sigma,
                &[],
                PHOTONS_SPECTRAL * terms,
                seed,
            )));
        }
    }
    sweep
}

#[test]
fn the_gas_table_flattens_the_ladder_at_every_published_wavelength() {
    let Sweep {
        with,
        without,
        worst,
        refused,
        filled,
        scenes,
        channels,
        transports,
        worst_terms,
    } = sweep_every_published_wavelength();
    let names = ["window", "weak band", "strong band"];
    let rms = |v: &[f64]| libm::sqrt(v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64);
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let compared: usize = with.iter().map(Vec::len).sum();
    println!(
        "MEASURED 211-wavelength sweep: {compared} comparisons over {scenes} scenes, {filled} \
         RadCalNet fill values, refused {} not tabulated / {} pressure off the ladder / {} column \
         past the fit",
        refused[Refusal::NotTabulated as usize],
        refused[Refusal::PressureOffLadder as usize],
        refused[Refusal::ColumnPastFit as usize],
    );
    println!(
        "  product terms: {transports} transport runs over {compared} channels, mean {:.1}, worst \
         {} at {:.0} nm",
        transports as f64 / compared as f64,
        worst_terms.0,
        worst_terms.1,
    );
    for rung in 0..3 {
        println!(
            "  {:11} {:4} points   rms {:.4} -> {:.4}   mean {:+.4} -> {:+.4}   worst {:+.4} at \
             {:.0} nm",
            names[rung],
            with[rung].len(),
            rms(&without[rung]),
            rms(&with[rung]),
            mean(&without[rung]),
            mean(&with[rung]),
            worst[rung].1,
            worst[rung].0,
        );
        assert!(
            !with[rung].is_empty(),
            "the {} rung has no points at all",
            names[rung]
        );
    }

    // The ladder is what #353 exists to flatten, so it is asserted as a ratio of ratios: the bands
    // must move towards the windows, not merely improve. Comparing each rung against itself would
    // pass on a model that got uniformly better and left the ladder exactly as steep.
    let steepness = |v: &[Vec<f64>]| rms(&v[2]) / rms(&v[0]);
    let (before, after) = (steepness(&without), steepness(&with));
    println!("  strong/window steepness {before:.1}x -> {after:.1}x");
    assert!(
        after < before / 10.0,
        "the gas layer did not flatten the ladder: {before:.1}x -> {after:.1}x"
    );
    // And the sign has to go with it. A slab that cannot absorb overpredicts wherever an absorber
    // acts -- that is what the ladder test at 14 wavelengths asserts -- so the part of the bias that
    // belongs to the gas has to go. What is compared is the *excess* over the windows, because the
    // windows carry a bias of their own that no absorber explains and the gas layer has no business
    // removing: aerosol microphysics stood in for by one albedo and one asymmetry, and the
    // plane-parallel slab of #357. Charging the gas layer with that too would be asking it to fix
    // somebody else's error, and it would pass or fail on how big that error happened to be.
    let excess = |v: &[Vec<f64>], rung: usize| mean(&v[rung]) - mean(&v[0]);
    for rung in [1, 2] {
        assert!(
            excess(&without, rung) > 0.0,
            "a slab that cannot absorb must overpredict in the {} rung",
            names[rung]
        );
        println!(
            "  {:11} excess over the windows {:+.4} -> {:+.4}",
            names[rung],
            excess(&without, rung),
            excess(&with, rung),
        );
        assert!(
            libm::fabs(excess(&with, rung)) < excess(&without, rung) / 3.0,
            "the {} rung keeps its band-specific bias: {:+.4} against {:+.4}",
            names[rung],
            excess(&with, rung),
            excess(&without, rung)
        );
    }
    // Coverage is a result too, and the refusals are read rather than counted: a pressure off the
    // ladder is a table that does not reach one of RadCalNet's own sites, which is a defect, while a
    // column past the fit is a channel that is genuinely opaque at this scene's water, which is the
    // documented behaviour and the reason `u_max` is in the file.
    assert_eq!(
        (
            refused[Refusal::NotTabulated as usize],
            refused[Refusal::PressureOffLadder as usize]
        ),
        (0, 0),
        "the table does not span RadCalNet's own grid and sites"
    );
    // Coverage closes as an identity rather than clearing a floor someone typed: every scene-channel
    // the sweep visited was either compared, skipped for a RadCalNet fill value, or refused with a
    // reason. A number like `compared > 1000` cannot tell a table that lost a hundred channels from a
    // fixture that gained a hundred fill values; this can.
    let accounted: usize = compared + filled + refused.iter().sum::<usize>();
    assert_eq!(
        accounted,
        scenes * channels,
        "{scenes} scenes times {channels} published wavelengths is {}, but {compared} compared plus \
         {filled} filled plus {} refused accounts for {accounted}",
        scenes * channels,
        refused.iter().sum::<usize>()
    );
    assert!(
        channels == 211 && compared > scenes * channels / 2,
        "the sweep is no longer a sweep: {compared} comparisons over {channels} channels"
    );
}

#[test]
fn the_chappuis_band_needs_the_measured_ozone_column() {
    let table = ozone_cross_section();
    let (mut with, mut without) = (Vec::new(), Vec::new());
    for (index, row) in rows().iter().enumerate().filter(|(_, r)| r.gated()) {
        if !(540.0..=690.0).contains(&row.wavelength_nm) {
            continue;
        }
        let modelled =
            coupling(row, AEROSOL_SSA, stream(index, 0), PHOTONS).toa_from_surface(row.boa);
        let sigma = sigma_for(&table, row);
        with.push((modelled * ozone_transmittance(row, sigma) - row.toa) / row.toa);
        without.push((modelled - row.toa) / row.toa);
    }
    let rms = |v: &[f64]| libm::sqrt(v.iter().map(|e| e * e).sum::<f64>() / v.len() as f64);
    let (corrected, raw) = (rms(&with), rms(&without));
    println!(
        "MEASURED Chappuis 540-690 nm over {} points: with the measured ozone column {corrected:.4}, \
         without it {raw:.4}, a factor of {:.1}",
        with.len(),
        raw / corrected
    );
    // Section 5.1 measured the same absorber as a *fitted* scalar per wavelength and found the
    // molecular atmosphere 16.5 % low at 600 nm without it. Here the column is RadCalNet's own
    // measurement and the cross-section is published, so the improvement is a prediction rather than
    // a fit -- which is why the floor is asserted on the ratio and not on the corrected value.
    assert!(
        raw > 2.5 * corrected,
        "dropping the ozone costs only {:.1}x in the Chappuis band, where it should dominate",
        raw / corrected
    );
}

#[test]
fn the_unpublished_aerosol_albedo_is_the_one_thing_this_comparison_fits() {
    // Three points bracketing the constant the rest of the file runs at. That is all a test needs to
    // say the minimum is interior; the nine-point scan and the per-site breakdown that show *how*
    // interior cost a hundred times as many photons and live in
    // `examples/radcalnet_error_budget.rs`, which is not on the suite's clock.
    let table = ozone_cross_section();
    let rows = rows();

    // What the fit is allowed to assume, checked before it is used. One albedo may stand for the
    // whole fixture only because RadCalNet ran one aerosol model over the whole fixture; the moment a
    // scene arrives on a different one, the minimum below is an average of two microphysics and the
    // number it returns describes neither.
    let types: std::collections::BTreeSet<&str> =
        rows.iter().map(|row| row.aerosol_type.as_str()).collect();
    println!("MEASURED RadCalNet's aerosol type across the fixture: {types:?}");
    assert_eq!(
        types,
        std::collections::BTreeSet::from([AEROSOL_TYPE]),
        "the fixture mixes aerosol models, so one fitted albedo and the quoted IHAZE: 2 deck no \
         longer describe every scene in it"
    );

    let measured: Vec<(f64, f64)> = [AEROSOL_SSA - 0.02, AEROSOL_SSA, AEROSOL_SSA + 0.02]
        .iter()
        .map(|&candidate| {
            let mut residual = Residual::default();
            for (index, row) in rows.iter().enumerate().filter(|(_, r)| r.gated()) {
                let sigma = sigma_for(&table, row);
                let modelled = model_toa(row, sigma, candidate, stream(index, 0), PHOTONS);
                residual.push(row, &[modelled, modelled], row.toa, row.toa_unc, 0.0);
            }
            let (_, rms, _) = residual.report(&format!("omega_a {candidate:.2}"));
            (candidate, rms)
        })
        .collect();
    println!(
        "MEASURED aerosol albedo: {:.2} -> {:.4e}, {:.2} -> {:.4e}, {:.2} -> {:.4e}",
        measured[0].0, measured[0].1, measured[1].0, measured[1].1, measured[2].0, measured[2].1
    );

    // A minimum on the edge of the bracket is not a minimum, it is a direction; only an interior one
    // says the data determine the parameter rather than merely prefer more or less of it.
    assert!(
        measured[1].1 < measured[0].1 && measured[1].1 < measured[2].1,
        "the bracket does not contain a minimum, so {AEROSOL_SSA} is not a fitted value"
    );
}

#[test]
fn the_fitted_albedo_predicts_days_nobody_chose() {
    // The test above establishes that ω_a is *identified*. This one establishes that it is not
    // merely absorbing this fixture's own peculiarities: it is applied unchanged to days selected by
    // index through each site's archive, with nothing about their weather entering the choice. The
    // fitted fixture's own 42 scenes were picked after scanning, and a number fitted and reported on
    // those alone can only say how well the picking went.
    let fitted = rows();
    let held = holdout();

    // Disjoint, asserted here and not only in the generator, because the claim is made here.
    let day = |r: &Row| format!("{} {}", r.site, r.date);
    let seen: std::collections::BTreeSet<String> = fitted.iter().map(day).collect();
    let shared: std::collections::BTreeSet<String> = held
        .iter()
        .map(day)
        .filter(|key| seen.contains(key))
        .collect();
    assert!(
        shared.is_empty(),
        "{shared:?} is in both fixtures, so nothing below is out of sample"
    );

    let table = ozone_cross_section();
    let mut out = Residual::default();
    for (index, row) in held.iter().enumerate().filter(|(_, r)| r.gated()) {
        let sigma = sigma_for(&table, row);
        let draws: Vec<f64> = (0..SEEDS)
            .map(|draw| model_toa(row, sigma, AEROSOL_SSA, stream(index, draw), PHOTONS))
            .collect();
        let ours = input_sensitivity(row, sigma, index, PHOTONS_SENSITIVITY, false);
        out.push(row, &draws, row.toa, row.toa_unc, ours);
    }
    out.breakdown("holdout");
    let (_, _, ratio) = out.report("holdout TOA");
    let (site, worst_site) = out.worst_site();
    println!(
        "MEASURED the hold-out spans {} days over {} instruments, none of them chosen for weather",
        held.iter()
            .map(day)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        out.sites().len()
    );

    // The claim, and the same gate the in-sample residual is held to: the pooled fit predicts days
    // it never saw to inside the quadrature bar. A larger ratio here than in sample would say ω_a
    // had absorbed something specific to the 42 chosen scenes, which is the failure this fixture
    // exists to detect -- and it would be a finding about the fit, not a tolerance to widen.
    assert!(
        ratio < 1.0,
        "out of sample the residual sits at {ratio:.3} of its bar, so the fitted albedo does not \
         generalise off the days it was fitted on"
    );
    assert!(
        worst_site < 1.0,
        "{site} alone sits at {worst_site:.3} of its bar out of sample"
    );
}

#[test]
fn the_plane_parallel_slab_survives_a_days_air_mass_sweep() {
    // Both codes are plane-parallel, so a sphericity error is common to them and cancels; what does
    // not cancel is any difference in how each *handles* a long slant path. Separating that from an
    // aerosol error needs the aerosol to hold still while the geometry moves, which is what a sweep
    // day is: a cloud-free day whose AOT barely changes across the sun's whole arc. The generator
    // takes five times from one, and one time from every other day, so a day with more than one time
    // in the fixture is a sweep day -- no column has to say so.
    let table = ozone_cross_section();
    let rows = rows();
    let mut trend: Vec<(String, f64, f64, usize)> = Vec::new();
    let mut days: Vec<String> = rows.iter().map(Row::day).collect();
    days.sort_unstable();
    days.dedup();
    for day in days {
        let scenes: Vec<(usize, &Row)> = rows
            .iter()
            .enumerate()
            .filter(|(_, row)| row.day() == day && row.gated())
            .collect();
        let air_mass: Vec<f64> = scenes.iter().map(|(_, row)| row.air_mass()).collect();
        let span = air_mass.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - air_mass.iter().copied().fold(f64::INFINITY, f64::min);
        // RadCalNet publishes 09:00-15:00 local only, so a day's whole arc is half a unit of air
        // mass at most sites; below that a slope is not resolvable against the Monte-Carlo noise.
        if span < 0.5 {
            continue;
        }
        let residual: Vec<f64> = scenes
            .iter()
            .map(|(index, row)| {
                let sigma = sigma_for(&table, row);
                (model_toa(row, sigma, AEROSOL_SSA, stream(*index, 0), PHOTONS) - row.toa) / row.toa
            })
            .collect();
        // One slope per day, over that day's own air-mass range: the intercept absorbs whatever is
        // wrong with the day's aerosol and the slope is what is left for the geometry.
        let n = air_mass.len() as f64;
        let mean_m = air_mass.iter().sum::<f64>() / n;
        let mean_r = residual.iter().sum::<f64>() / n;
        let covariance: f64 = air_mass
            .iter()
            .zip(&residual)
            .map(|(m, r)| (m - mean_m) * (r - mean_r))
            .sum();
        let spread: f64 = air_mass.iter().map(|m| (m - mean_m) * (m - mean_m)).sum();
        let slope = covariance / spread;
        println!(
            "MEASURED sweep {day:>18}  n {:2}  air mass {:.2}-{:.2}  relative residual \
             {:+.4} at the mean, slope {slope:+.4} per air mass",
            scenes.len(),
            air_mass.iter().copied().fold(f64::INFINITY, f64::min),
            air_mass.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            mean_r,
        );
        trend.push((day, slope, span, scenes.len()));
    }

    assert!(
        trend.len() >= 5,
        "only {} days sweep more than one unit of air mass, which is not enough to bound the \
         plane-parallel term",
        trend.len()
    );
    // The bound: across a day's whole arc the relative residual may not drift by more than this much
    // of the signal. It is asserted on `slope * span`, the drift the fixture actually reaches, so a
    // day with a shorter arc is not held to a bar its geometry cannot test.
    let worst = trend
        .iter()
        .max_by(|a, b| (a.1 * a.2).abs().total_cmp(&(b.1 * b.2).abs()))
        .expect("a sweep day");
    println!(
        "MEASURED sweep worst: {} drifts {:+.4} of the signal across {:.2} of air mass",
        worst.0,
        worst.1 * worst.2,
        worst.2
    );
    assert!(
        (worst.1 * worst.2).abs() < 0.03,
        "{} drifts {:+.4} of the signal across its air-mass arc, so the plane-parallel slab does \
         not hold at a fixed atmosphere",
        worst.0,
        worst.1 * worst.2
    );
}

#[test]
fn the_tabulated_mixture_is_the_analytic_one_where_the_mixture_is_trivial() {
    // Drive a row's aerosol to nothing and the mixture is pure Rayleigh, where the analytic member
    // is exact. What is left in the difference is the tabulation: PHASE_NODES nodes, sampled by
    // inverting the CDF of the piecewise-linear interpolant rather than of the density.
    let mut row = rows().into_iter().next().expect("a fixture row");
    row.aod550 = 0.0;
    let analytic = coupling_from_slab(
        &Atmosphere {
            optical_depth: row.tau_rayleigh,
            single_scattering_albedo: 1.0,
            phase: PhaseFunction::Rayleigh,
        },
        row.mu_sun(),
        MU_VIEW,
        0.0,
        PHOTONS,
        SEED,
    );
    let tabulated = coupling(&row, AEROSOL_SSA, SEED, PHOTONS);
    let terms = [
        (
            "path",
            analytic.path_reflectance,
            tabulated.path_reflectance,
        ),
        (
            "T-down",
            analytic.transmittance_down,
            tabulated.transmittance_down,
        ),
        (
            "T-up",
            analytic.transmittance_up,
            tabulated.transmittance_up,
        ),
        (
            "albedo",
            analytic.spherical_albedo,
            tabulated.spherical_albedo,
        ),
    ];
    for (name, exact, table) in terms {
        let relative = table / exact - 1.0;
        println!(
            "MEASURED tabulation {name:<7} analytic {exact:.6e} tabulated {table:.6e} {relative:+.3e}"
        );
        assert!(
            relative.abs() < 2.0e-2,
            "{name}: the {PHASE_NODES}-node table differs from analytic Rayleigh by {relative:+.3e}"
        );
    }
}

#[test]
fn the_fixture_spans_the_regimes_a_correction_has_to_survive() {
    let rows = rows();
    let sites: std::collections::BTreeSet<&str> =
        rows.iter().map(|row| row.site.as_str()).collect();
    let scenes: std::collections::BTreeSet<&str> = rows
        .iter()
        .map(|row| row.key.rsplit_once(' ').expect("a key").0)
        .collect();
    let range = |v: Vec<f64>| {
        (
            v.iter().copied().fold(f64::INFINITY, f64::min),
            v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let zenith = range(rows.iter().map(|r| r.solar_zenith_deg).collect());
    let aod = range(rows.iter().map(|r| r.aod550).collect());
    let angstrom = range(rows.iter().map(|r| r.angstrom).collect());
    let surface = range(rows.iter().map(|r| r.boa).collect());
    let ozone = range(rows.iter().map(|r| r.ozone_du).collect());
    let water = range(rows.iter().map(|r| r.water_cm).collect());
    let pressure = range(rows.iter().map(|r| r.pressure_hpa).collect());
    let air_mass = range(rows.iter().map(Row::air_mass).collect());
    println!(
        "MEASURED {} rows, {} instruments, {} scenes; sza {zenith:?} air mass {air_mass:?} \
         aod550 {aod:?} angstrom {angstrom:?} boa {surface:?} ozone_du {ozone:?} \
         water_cm {water:?} pressure_hpa {pressure:?}",
        rows.len(),
        sites.len(),
        scenes.len(),
    );
    assert_eq!(sites.len(), 6, "six instrumented sites");
    assert_eq!(scenes.len(), 42, "forty-two scenes");
    // A regenerated fixture that quietly narrowed to the easy middle of every distribution would
    // leave the residuals above looking better than they are, so each axis has to keep its ends.
    assert!(zenith.0 < 10.0 && zenith.1 > 78.0, "sza {zenith:?}");
    assert!(air_mass.1 > 6.0, "air mass {air_mass:?}");
    assert!(aod.0 < 0.02 && aod.1 > 0.20, "aod550 {aod:?}");
    assert!(
        angstrom.0 < 0.15 && angstrom.1 > 1.7,
        "angstrom {angstrom:?}"
    );
    assert!(surface.0 < 0.10 && surface.1 > 0.40, "boa {surface:?}");
    assert!(ozone.1 - ozone.0 > 100.0, "ozone_du {ozone:?}");
    // Golmud at 2824 m against Railroad Valley at 1435 m and La Crau at sea level: the point of
    // carrying three altitudes is that the Rayleigh depth is a third smaller at one end.
    assert!(
        pressure.0 < 750.0 && pressure.1 > 1010.0,
        "pressure_hpa {pressure:?}"
    );
    assert!(water.0 < 0.3 && water.1 > 3.0, "water_cm {water:?}");
    assert!(
        rows.iter().any(Row::weak_band) && rows.iter().any(Row::strong_band),
        "the two absorbing classes survived regeneration"
    );
    assert!(
        rows.iter().any(|row| row.flag != "ok"),
        "RadCalNet's own uncertainty flags survived regeneration"
    );
    // Every published input uncertainty is a real number and not a placeholder, or the third term of
    // the error budget would be quietly zero.
    assert!(
        rows.iter().all(|row| row.aod550_unc > 0.0
            && row.angstrom_unc > 0.0
            && row.ozone_du_unc > 0.0
            && row.pressure_hpa_unc > 0.0
            && row.water_cm_unc > 0.0),
        "a published input uncertainty is missing, so the error budget is one-sided again"
    );
}

/// What the resort costs on the shipped table rather than on a stand-in, and the measurement that
/// sets [`GAS_BINS`].
///
/// Compared through the direct beam and not through the transport, deliberately. The bias enters by
/// exactly the same mechanism either way -- merging grey absorbers of unequal depth -- but the direct
/// beam is deterministic, where a Monte-Carlo comparison at `PHOTONS_SPECTRAL` photons carries about
/// 1e-2 of its own noise and could not see a 1e-4 bias through it. Measuring the approximation where
/// it is visible, and then using it where it is not, is the point.
#[test]
fn the_resort_costs_less_than_the_photon_noise_it_replaces() {
    let table = gas_table();
    let (scenes, spectrum) = absorption();
    let beam = |terms: &[(f64, f64)], air_mass: f64| -> f64 {
        terms
            .iter()
            .map(|&(weight, depth)| weight * libm::exp(-depth * air_mass))
            .sum()
    };
    let counts = [32_usize, 64, 128, 256];
    let mut worst = [(0.0_f64, 0.0_f64); 4];
    let (mut compared, mut exact_terms, mut cut_terms) = (0_usize, 0_usize, 0_usize);

    // A handful of scenes, not the whole fixture: the exact product this exists to avoid is 3.9
    // million terms per scene, so the sample is chosen to span the water column rather than to be
    // large. The bias depends on the columns and on nothing else about the scene.
    for scene in scenes.iter().step_by(scenes.len().div_ceil(4)) {
        for band in &spectrum {
            let Ok(exact) = scene_gas_terms(&table, scene, band.wavelength_nm, None) else {
                continue;
            };
            compared += 1;
            exact_terms += exact.len();
            let whole = beam(&exact, scene.air_mass);
            for (slot, &bins) in worst.iter_mut().zip(&counts) {
                let cut =
                    scene_gas_terms(&table, scene, band.wavelength_nm, NonZeroUsize::new(bins))
                        .expect("the exact product was not refused, so neither is the resort");
                if bins == 256 {
                    cut_terms += cut.len();
                }
                let part = beam(&cut, scene.air_mass);
                assert!(
                    part <= whole + 1.0e-12,
                    "{bins} bins transmitted more than the exact product at {} nm, which Jensen \
                     forbids",
                    band.wavelength_nm
                );
                if (whole - part) / whole > slot.0 {
                    *slot = ((whole - part) / whole, band.wavelength_nm);
                }
            }
        }
    }

    println!("resort bias over {compared} channel-scenes of the shipped table:");
    for (&bins, &(bias, nm)) in counts.iter().zip(&worst) {
        println!("  {bins:4} bins   worst {bias:.3e} at {nm:.0} nm");
    }
    println!(
        "  terms: exact {} per channel on average, {} bins {}",
        exact_terms / compared,
        counts[3],
        cut_terms / compared
    );

    // The bar is what the sweep can resolve, not a round number. Its window rung -- channels with no
    // gas worth the name -- reports 1.52e-2 rms against RadCalNet with the gas layer switched out
    // entirely, so that is the floor any gas modelling sits on. A deterministic one-signed bias a
    // third of it cannot change a verdict; one approaching it could.
    let operating = worst[3].0;
    assert!(
        operating < 5.0e-3,
        "the resort at {} bins now costs {operating:.3e} of channel transmittance at {:.0} nm, \
         which is no longer a third of the 1.52e-2 the sweep's own window rung can resolve",
        counts[3],
        worst[3].1
    );
    assert!(
        worst[0].0 > worst[3].0,
        "eight times the transport runs bought nothing: {:.3e} at {} bins against {:.3e} at {}",
        worst[0].0,
        counts[0],
        worst[3].0,
        counts[3]
    );

    // The bar above cannot say which binning measure is in force, and that is measured rather than
    // suspected: at 256 bins `w exp(-tau)` and the obvious `w + exp(-tau)` land at 4.10e-3 and
    // 3.75e-3 on this table, so the operating point is insensitive to the choice and a mutation of
    // it survives the whole suite. Where the choice matters is the cheap end of the ladder --
    // 32 bins costs 3.35e-2 on transmitted weight against 1.88e-1 on `w + exp(-tau)` and 1.0e-1 on
    // plain weight. So the rule buys its factor of five where bins are scarce and nothing where
    // they are not, which is the honest statement of what it is for, and the coarse rung is the
    // only rung that holds it.
    let coarse = worst[0].0;
    assert!(
        coarse < 8.0e-2,
        "the resort at {} bins now costs {coarse:.3e} at {:.0} nm, which is where binning on plain \
         weight or on `w + exp(-tau)` sits: the measure is no longer transmitted weight",
        counts[0],
        worst[0].1
    );
}

/// The photon budget follows the weight, and the redistribution can only help.
///
/// [`correlated_k_coupling`] divides its budget in proportion to term weight rather than equally.
/// For a weighted sum `sum_j w_j X_j` at fixed `sum_j n_j` the variance `sum_j w_j^2 sigma^2 / n_j`
/// is smallest at `n_j proportional to w_j`, and the factor between that and the equal split is
/// `m sum_j w_j^2` -- at least 1 by Cauchy-Schwarz, equal to 1 only where the weights are already
/// uniform.
///
/// **The bound is the point of the test, not the average.** A gain that is merely large in the mean
/// would still permit a channel somewhere in the table to come out noisier, and a redistribution
/// that helps the median band while hurting the 2350 nm one is not a change that can be applied
/// without a threshold. What is asserted is therefore the floor over every channel first, and the
/// scale of the win second.
///
/// The tail matters for the opposite reason. Proportional allocation sends a 1e-9-weight term below
/// one photon, and the floor keeps it at one rather than dropping it -- dropping it would break
/// `sum w = 1` silently. What that costs is bounded here against the resort's own 3.9e-3.
#[test]
fn the_photon_budget_follows_the_weight_where_the_weight_is_concentrated() {
    let table = gas_table();
    let (scenes, spectrum) = absorption();
    let bins = GAS_BINS;
    let (mut worst, mut total_gain, mut channels) = (f64::INFINITY, 0.0_f64, 0_usize);
    let (mut worst_gain_nm, mut worst_floored) = (0.0_f64, 0.0_f64);
    for scene in scenes.iter().step_by(scenes.len().div_ceil(4)) {
        for band in &spectrum {
            let Ok(cut) = scene_gas_terms(&table, scene, band.wavelength_nm, bins) else {
                continue;
            };
            let sum_sq: f64 = cut.iter().map(|&(w, _)| w * w).sum();
            #[expect(clippy::cast_precision_loss, reason = "term counts are far below 2^53")]
            let gain = cut.len() as f64 * sum_sq;
            let floored: f64 = cut
                .iter()
                .filter(|&&(w, _)| libm::round(f64::from(PHOTONS) * w) < 1.0)
                .map(|&(w, _)| w)
                .sum();
            assert!(
                gain >= 1.0 - 1.0e-12,
                "proportional allocation is noisier than the equal split at {} nm, gain {gain:.4}, \
                 which Cauchy-Schwarz forbids and which would make the redistribution unsafe to \
                 apply without a per-channel threshold",
                band.wavelength_nm
            );
            if gain < worst {
                worst = gain;
                worst_gain_nm = band.wavelength_nm;
            }
            worst_floored = worst_floored.max(floored);
            total_gain += gain;
            channels += 1;
        }
    }
    let mean =
        total_gain / f64::from(u32::try_from(channels).expect("far fewer than 4e9 channels"));
    println!(
        "  {channels} channels, mean variance gain {mean:.2}x, worst {worst:.2}x at {worst_gain_nm:.0} nm"
    );
    println!("  worst channel floors {worst_floored:.3e} of its weight to one photon");
    assert!(
        mean > 5.0,
        "the mean variance gain is {mean:.2}x, and below about 5x the redistribution stops paying \
         for the extra concept in `correlated_k_coupling`"
    );
    assert!(
        worst_floored < 1.0e-3,
        "one channel floors {worst_floored:.3e} of its weight to a single photon, which is no \
         longer small against the 3.9e-3 the resort itself costs"
    );
}
