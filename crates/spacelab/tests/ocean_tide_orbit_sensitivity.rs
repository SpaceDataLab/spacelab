//! What the choice of ocean-tide atlas does to an orbit, measured before an orbit is asked to rank
//! the atlases.
//!
//! The ocean-tide default is meant to be the atlas that fits LAGEOS and GNSS orbits best
//! (`docs/verification.md` §10.8). That criterion is only usable where the atlases move the orbit
//! by more than the fit's own floor. This file measures how far they move it. Every run starts from
//! the same state and uses the same force model: degree-20 geopotential, analytic Sun and Moon,
//! Schwarzschild, and an ocean tide to degree 6. Only the tide table differs between runs. The
//! difference between two runs is then fitted with the six epoch-state parameters, through the
//! reference run's state transition matrix. What is left is the part of an atlas change that an arc
//! fit cannot absorb, and so the most a residual could show.
//!
//! Each atlas difference is split in two. "Its wave set" is FES2014b restricted to the Doodson
//! numbers the other atlas carries, against all of FES2014b: what the missing waves cost.
//! "Its coefficients" is the other atlas against that restriction: what its numbers cost on the
//! waves both carry. 3D RMS after the fit, millimetres, measured 2026-10-24:
//!
//! | row | LAGEOS-1, 7 d | LAGEOS-2, 7 d | GPS G02, 1 d |
//! |---|---|---|---|
//! | integration noise (two tolerances) | 0.019 | 0.005 | 0.001 |
//! | no ocean tide | 323.5 | 375.3 | 9.67 |
//! | FES2014b, degree 6 → 20 | 4.09 | 5.60 | 0.009 |
//! | GOT5.5: its wave set | 22.6 | 11.4 | 1.03 |
//! | GOT5.5: its coefficients | 7.95 | 8.81 | 0.39 |
//! | EOT20: its wave set | 23.7 | 34.1 | 0.34 |
//! | EOT20: its coefficients | 7.52 | 5.29 | 0.42 |
//! | FES2004: its wave set | 27.0 | 31.1 | 0.19 |
//! | FES2004: its coefficients | 24.3 | 39.9 | 0.94 |
//!
//! Three things follow. On GPS no atlas moves the orbit by 1.2 mm, against the 94 mm that
//! `igs_orbit_fit.rs` reaches with DE440, so a GNSS fit cannot rank atlases at all. On LAGEOS the
//! effect is centimetres, and **coverage costs more than coefficients**. GOT has no long-period
//! waves and EOT20 has 16 of FES2014b's 35. Leaving those waves out moves a week of LAGEOS by
//! 11 to 34 mm; the two modern atlases' own coefficients move it by 5 to 9 mm. And FES2004's
//! coefficients are the outlier at 24 to 40 mm, which is the generation gap the coefficient
//! comparison in `spacelab-propagation/tests/ocean_tide_atlas.rs` sees too.
//!
//! The six-parameter fit is a floor on what an analysis absorbs, not the analysis itself. An ILRS
//! analysis centre also estimates along-track and once-per-revolution accelerations, which absorb
//! more. Ranking the coefficients of GOT, EOT20 and FES2014b therefore needs a LAGEOS orbit
//! determination from normal points with those parameters, at a residual floor of millimetres.
//! `slr_range_residuals.rs` reaches 1.6 cm, without ocean loading at the stations.

// `FES2014b`, `GOT5.5` and the other model names are the atlases' own spelling, not identifiers.
#![allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names
)]

use spacelab::{EopSource, earth_orientation_at};
use spacelab_autodiff::Scalar;
use spacelab_core::{Angle, Epoch, Ut1, UtcDateTime};
use spacelab_eop::EopSeries;
use spacelab_fa::fundamental::{fad03, faf03, fal03, falp03, faom03};
use spacelab_fa::{delta_at, gmst82};
use spacelab_frames::CioInterpolator;
use spacelab_gnss::{SatId, Sp3, TimeSystem};
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::ephemeris::{AnalyticMoon, AnalyticSun};
use spacelab_propagation::forces::{Acceleration, BodyOriented, Plain, ThirdBody};
use spacelab_propagation::geopotential_coeffs::{MODEL_COEFFS, MODEL_GM, MODEL_RE};
use spacelab_propagation::linalg::lu_solve;
use spacelab_propagation::ocean_tide::OceanTide;
use spacelab_propagation::ocean_tide_eot20::EOT20_WAVES;
use spacelab_propagation::ocean_tide_fes2004::{FES2004_WAVES, OceanTideWave, SCALE};
use spacelab_propagation::ocean_tide_fes2014::FES2014B_WAVES;
use spacelab_propagation::ocean_tide_got::GOT_WAVES;
use spacelab_propagation::variational::{ScalarSystem, Variational, augment, split};
use spacelab_propagation::{Dop853, GeoPotential, Schwarzschild, Tolerance};

const LAGEOS1_SP3: &str = include_str!("fixtures/ilrsa_lageos1_arc250111.sp3");
const LAGEOS2_SP3: &str = include_str!("fixtures/ilrsa_lageos2_arc250111.sp3");
const LAGEOS_EOP: &str = include_str!("fixtures/finals2000A_20250105_20250112.txt");
const GPS_SP3: &str = include_str!("fixtures/igs20_g02_2025001.sp3");
const GPS_EOP: &str = include_str!("fixtures/finals2000A_2025001.txt");

const GPS_MINUS_TAI: f64 = -19.0;
const MU_SUN: f64 = 1.327_124_400_18e20;
const MU_MOON: f64 = 4.902_800_066e12;
const GEOPOTENTIAL_DEGREE: usize = 20;
const TIDE_DEGREE: usize = 6;

/// An orbit's starting point and the Earth orientation over its span.
struct Arc {
    initial: [f64; 6],
    cio: CioInterpolator,
    epoch_tt_seconds: f64,
    epoch_ut1_ns: i128,
    span: f64,
    spacing: f64,
}

fn build_arc(sp3_text: &str, eop_text: &str, sat: SatId, span: f64) -> Arc {
    let sp3 = Sp3::parse(sp3_text);
    let series = EopSeries::parse_finals2000a(eop_text).expect("the EOP excerpt parses");
    let start = sp3.start_epoch().expect("the file carries epochs");
    let epochs = sp3.epochs().to_vec();
    let positions = sp3
        .positions(sat)
        .expect("the file carries the satellite")
        .to_vec();
    let approximate = UtcDateTime::from_naive_ns_since_j2000((start * 1e9).round() as i128, false);
    let offset_to_utc = match sp3.time_system() {
        TimeSystem::Utc => 0.0,
        TimeSystem::Gps => {
            f64::from(delta_at(approximate).expect("a tabulated leap-second count")) + GPS_MINUS_TAI
        }
        other => panic!("no leap-second handling for {other:?} in this test"),
    };
    let utc = UtcDateTime::from_naive_ns_since_j2000(
        ((start - offset_to_utc) * 1e9).round() as i128,
        false,
    );
    let e = earth_orientation_at(utc, EopSource::Series(&series)).expect("the epoch is in range");
    let cio = CioInterpolator::new(
        e.orientation.ut1,
        e.orientation.tt,
        span,
        e.orientation.xp,
        e.orientation.yp,
        Angle::from_radians(0.0),
        Angle::from_radians(0.0),
        e.orientation.lod_seconds,
    );

    // The first five samples in GCRF, and the epoch velocity from the fourth-order one-sided
    // difference. Any bound orbit near the satellite's own serves: what is measured is a difference
    // between two propagations from the same start.
    let spacing = epochs[1] - epochs[0];
    let mut p = [[0.0; 3]; 5];
    for k in 0..5 {
        let itrf = positions[k].expect("the first samples are present");
        let m = cio.rotation_rows(epochs[k] - epochs[0]);
        for i in 0..3 {
            p[k][i] = m[0][i] * itrf[0] + m[1][i] * itrf[1] + m[2][i] * itrf[2];
        }
    }
    let mut initial = [0.0; 6];
    for i in 0..3 {
        initial[i] = p[0][i];
        initial[3 + i] = (-25.0 * p[0][i] + 48.0 * p[1][i] - 36.0 * p[2][i] + 16.0 * p[3][i]
            - 3.0 * p[4][i])
            / (12.0 * spacing);
    }
    Arc {
        initial,
        cio,
        epoch_tt_seconds: e.orientation.tt.ns_since_j2000() as f64 / 1e9,
        epoch_ut1_ns: e.orientation.ut1.ns_since_j2000(),
        span,
        spacing,
    }
}

/// The force model every run shares; only `tide` differs between them.
struct Model<R: Fn(f64) -> [[f64; 3]; 3], T: Acceleration> {
    gravity: BodyOriented<GeoPotential, R>,
    sun: ThirdBody<AnalyticSun>,
    moon: ThirdBody<AnalyticMoon>,
    relativity: Schwarzschild,
    tide: T,
}

impl<R: Fn(f64) -> [[f64; 3]; 3], T: Acceleration> ScalarSystem<6> for Model<R, T> {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 6]) -> [S; 6] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let g = self.gravity.acceleration(t, &r, &v);
        let s = self.sun.acceleration(t, &r, &v);
        let m = self.moon.acceleration(t, &r, &v);
        let rel = self.relativity.acceleration(t, &r, &v);
        let o = self.tide.acceleration(t, &r, &v);
        [
            y[3],
            y[4],
            y[5],
            g[0] + s[0] + m[0] + rel[0] + o[0],
            g[1] + s[1] + m[1] + rel[1] + o[1],
            g[2] + s[2] + m[2] + rel[2] + o[2],
        ]
    }
}

/// A table cut to `degree`, so the per-step sum does not walk FES2014b's 175 033 rows to use 25.
fn truncated(waves: &'static [OceanTideWave], degree: u8) -> &'static [OceanTideWave] {
    let cut: Vec<OceanTideWave> = waves
        .iter()
        .map(|w| OceanTideWave {
            doodson: w.doodson,
            coeffs: Box::leak(
                w.coeffs
                    .iter()
                    .copied()
                    .filter(|c| c.0 <= degree)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
        })
        .collect();
    Box::leak(cut.into_boxed_slice())
}

#[expect(
    clippy::type_complexity,
    reason = "the closures are anonymous, and naming them would box every force evaluation"
)]
fn model<'a>(
    arc: &'a Arc,
    waves: &'static [OceanTideWave],
    degree: usize,
) -> Model<
    impl Fn(f64) -> [[f64; 3]; 3] + 'a,
    OceanTide<impl Fn(f64) -> [[f64; 3]; 3] + 'a, impl Fn(f64) -> (f64, [f64; 5]) + 'a>,
> {
    let coefficients: Vec<_> = MODEL_COEFFS
        .iter()
        .copied()
        .filter(|(n, _, _, _)| *n <= GEOPOTENTIAL_DEGREE)
        .collect();
    let geopotential =
        GeoPotential::from_normalized(MODEL_GM, MODEL_RE, GEOPOTENTIAL_DEGREE, &coefficients)
            .expect("a degree well inside the model");
    let epoch_tt = arc.epoch_tt_seconds;
    let epoch_ut1 = arc.epoch_ut1_ns;
    Model {
        gravity: BodyOriented {
            inner: geopotential,
            rotate: |t: f64| arc.cio.rotation_rows(t),
        },
        sun: ThirdBody {
            mu: MU_SUN,
            ephemeris: AnalyticSun {
                epoch_j2000_seconds: epoch_tt,
            },
        },
        moon: ThirdBody {
            mu: MU_MOON,
            ephemeris: AnalyticMoon {
                epoch_j2000_seconds: epoch_tt,
            },
        },
        relativity: Schwarzschild { mu: MODEL_GM },
        tide: OceanTide {
            mu_earth: MODEL_GM,
            re_earth: MODEL_RE,
            waves,
            scale: SCALE,
            max_degree: degree,
            admittance: &[],
            rotate: |t: f64| arc.cio.rotation_rows(t),
            delaunay_gmst: move |t: f64| {
                let ut1 = Epoch::<Ut1>::from_ns_since_j2000(epoch_ut1 + (t * 1e9).round() as i128);
                let c = (epoch_tt + t) / (86_400.0 * 36_525.0);
                (
                    gmst82(ut1).radians(),
                    [fal03(c), falp03(c), faf03(c), fad03(c), faom03(c)],
                )
            },
        },
    }
}

fn positions(
    arc: &Arc,
    waves: &'static [OceanTideWave],
    degree: usize,
    rtol: f64,
) -> Vec<[f64; 3]> {
    let traj = integrate(
        &Dop853,
        &Plain(model(arc, waves, degree)),
        0.0,
        &arc.initial,
        arc.span,
        60.0,
        Controls::new(Tolerance::new(1e-9, rtol), 1e-3),
    )
    .expect("the arc integrates");
    sample_times(arc)
        .map(|t| {
            let y = traj.eval(t);
            [y[0], y[1], y[2]]
        })
        .collect()
}

fn sample_times(arc: &Arc) -> impl Iterator<Item = f64> + '_ {
    let n = (arc.span / arc.spacing).round() as usize;
    (0..=n).map(move |k| k as f64 * arc.spacing)
}

/// The reference orbit's states and position partials `∂r(t)/∂x₀` at every sample.
fn reference(arc: &Arc, waves: &'static [OceanTideWave]) -> Vec<([f64; 6], [[f64; 6]; 3])> {
    let m = model(arc, waves, TIDE_DEGREE);
    let var = Variational::<6, _>::new(&m);
    let traj = integrate(
        &Dop853,
        &var,
        0.0,
        &augment::<6, 42>(&arc.initial),
        arc.span,
        60.0,
        Controls::new(Tolerance::new(1e-9, 1e-13), 1e-3),
    )
    .expect("the variational arc integrates");
    sample_times(arc)
        .map(|t| {
            let (state, phi) = split::<6, 42>(&traj.eval(t));
            (state, [phi[0], phi[1], phi[2]])
        })
        .collect()
}

/// What is left of `other − reference` once the six epoch-state parameters have absorbed what they
/// can: the part of an atlas change an arc fit cannot hide. Radial, along, cross and 3D RMS, and
/// the raw (unfitted) worst 3D difference, metres.
fn post_fit(
    arc: &Arc,
    reference: &[([f64; 6], [[f64; 6]; 3])],
    base: &[[f64; 3]],
    other: &[[f64; 3]],
) -> ([f64; 4], f64) {
    // Velocity columns scaled by the span, so the normal matrix is not 10¹¹ out of balance.
    let scale = |j: usize| if j < 3 { 1.0 } else { arc.span };
    let mut n = [[0.0; 6]; 6];
    let mut b = [0.0; 6];
    let mut raw: f64 = 0.0;
    for ((_, phi), (p, q)) in reference.iter().zip(base.iter().zip(other)) {
        let d = [q[0] - p[0], q[1] - p[1], q[2] - p[2]];
        raw = raw.max((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        for i in 0..3 {
            for j in 0..6 {
                b[j] += phi[i][j] * scale(j) * d[i];
                for k in 0..6 {
                    n[j][k] += phi[i][j] * scale(j) * phi[i][k] * scale(k);
                }
            }
        }
    }
    let x = lu_solve(n, b).expect("six parameters are observable over an arc");
    let mut sums = [0.0; 4];
    for ((state, phi), (p, q)) in reference.iter().zip(base.iter().zip(other)) {
        let mut e = [0.0; 3];
        for i in 0..3 {
            let fitted: f64 = (0..6).map(|j| phi[i][j] * scale(j) * x[j]).sum();
            e[i] = q[i] - p[i] - fitted;
        }
        let axes = rtn_axes(state);
        for (a, axis) in axes.iter().enumerate() {
            let c = e[0] * axis[0] + e[1] * axis[1] + e[2] * axis[2];
            sums[a] += c * c;
        }
        sums[3] += e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }
    let count = reference.len() as f64;
    (sums.map(|s| (s / count).sqrt()), raw)
}

fn rtn_axes(state: &[f64; 6]) -> [[f64; 3]; 3] {
    let unit = |a: [f64; 3]| {
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let r = [state[0], state[1], state[2]];
    let v = [state[3], state[4], state[5]];
    let radial = unit(r);
    let normal = unit(cross(r, v));
    [radial, cross(normal, radial), normal]
}

/// FES2014b's waves whose Doodson numbers `like` also carries: what FES2014b says about another
/// atlas's wave set, so a difference splits into the coefficients and the coverage.
fn restricted(
    waves: &'static [OceanTideWave],
    like: &'static [OceanTideWave],
) -> &'static [OceanTideWave] {
    let kept: Vec<OceanTideWave> = waves
        .iter()
        .filter(|w| like.iter().any(|l| l.doodson == w.doodson))
        .map(|w| OceanTideWave {
            doodson: w.doodson,
            coeffs: w.coeffs,
        })
        .collect();
    Box::leak(kept.into_boxed_slice())
}

/// One propagation's positions at the sample times.
type Run = Vec<[f64; 3]>;

/// One arc's table: the 3D post-fit difference of each row, millimetres.
struct Sensitivity(Vec<(String, f64)>);

impl Sensitivity {
    fn mm(&self, row: &str) -> f64 {
        self.0
            .iter()
            .find(|(name, _)| name == row)
            .map(|(_, mm)| *mm)
            .expect("a row the table carries")
    }
}

fn measure(label: &str, arc: &Arc) -> Sensitivity {
    let fes14 = truncated(FES2014B_WAVES, 20);
    let reference = reference(arc, fes14);
    let run = |waves: &'static [OceanTideWave], degree: usize| positions(arc, waves, degree, 1e-14);
    let base = run(fes14, TIDE_DEGREE);
    let mut rows: Vec<(String, Run, Run)> = vec![
        (
            "integration noise".into(),
            base.clone(),
            positions(arc, fes14, TIDE_DEGREE, 1e-13),
        ),
        ("no ocean tide".into(), base.clone(), run(&[], TIDE_DEGREE)),
        (
            "FES2014b, degree 6 -> 20".into(),
            base.clone(),
            run(fes14, 20),
        ),
    ];
    for (name, table) in [
        ("GOT5.5", GOT_WAVES),
        ("EOT20", EOT20_WAVES),
        ("FES2004", FES2004_WAVES),
    ] {
        let theirs = run(table, TIDE_DEGREE);
        let same_waves = run(restricted(fes14, table), TIDE_DEGREE);
        rows.push((format!("{name}: total"), base.clone(), theirs.clone()));
        rows.push((
            format!("{name}: its wave set"),
            base.clone(),
            same_waves.clone(),
        ));
        rows.push((format!("{name}: its coefficients"), same_waves, theirs));
    }
    eprintln!(
        "{label}: differences after a six-parameter fit (mm); base FES2014b to degree {TIDE_DEGREE}"
    );
    eprintln!(
        "  {:32} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "", "radial", "along", "cross", "3D", "raw max"
    );
    let mut table = Vec::new();
    for (name, from, to) in &rows {
        let (rms, raw) = post_fit(arc, &reference, from, to);
        eprintln!(
            "  {name:32} {:8.4} {:8.4} {:8.4} {:8.4} {:10.4}",
            rms[0] * 1e3,
            rms[1] * 1e3,
            rms[2] * 1e3,
            rms[3] * 1e3,
            raw * 1e3
        );
        table.push((name.clone(), rms[3] * 1e3));
    }
    Sensitivity(table)
}

/// The measurement, and the claims `docs/verification.md` §10.8 makes from it. The bounds are the
/// observed values of 2026-10-24, rounded outward.
#[test]
#[ignore = "about 2.5 min in release: cargo nextest run -p spacelab --release --run-ignored only \
            --test ocean_tide_orbit_sensitivity"]
fn the_atlas_choice_is_visible_on_lageos_and_invisible_on_gps() {
    let week = 7.0 * 86_400.0 - 3_600.0;
    let lageos = [
        measure(
            "LAGEOS-1, 2025-01-05, 7 d",
            &build_arc(
                LAGEOS1_SP3,
                LAGEOS_EOP,
                SatId {
                    system: 'L',
                    prn: 51,
                },
                week,
            ),
        ),
        measure(
            "LAGEOS-2, 2025-01-05, 7 d",
            &build_arc(
                LAGEOS2_SP3,
                LAGEOS_EOP,
                SatId {
                    system: 'L',
                    prn: 52,
                },
                week,
            ),
        ),
    ];
    let gps = measure(
        "GPS G02, 2025-01-01, 1 d",
        &build_arc(
            GPS_SP3,
            GPS_EOP,
            SatId {
                system: 'G',
                prn: 2,
            },
            86_400.0 - 900.0,
        ),
    );

    // Two tolerances on one model agree to 0.019 mm, so every row below is physics.
    for table in lageos.iter().chain([&gps]) {
        assert!(table.mm("integration noise") < 0.05);
    }
    for table in &lageos {
        // The whole effect: 323 and 375 mm.
        assert!(table.mm("no ocean tide") > 250.0);
        // Degree 6 to 20 on FES2014b: 4.1 and 5.6 mm.
        assert!(table.mm("FES2014b, degree 6 -> 20") > 3.0);
        // The two modern atlases disagree with FES2014b by 5.3 to 8.8 mm on the waves they share,
        // and the waves they do not carry cost more, 11.4 to 34.1 mm.
        for atlas in ["GOT5.5", "EOT20"] {
            let coefficients = table.mm(&format!("{atlas}: its coefficients"));
            let coverage = table.mm(&format!("{atlas}: its wave set"));
            assert!(
                (3.0..12.0).contains(&coefficients),
                "{atlas}: {coefficients:.2} mm"
            );
            assert!(
                coverage > coefficients,
                "{atlas}: {coverage:.2} against {coefficients:.2} mm"
            );
        }
        // FES2004's coefficients are the outlier, 24.3 and 39.9 mm.
        assert!(table.mm("FES2004: its coefficients") > 20.0);
    }
    // At GPS altitude nothing reaches 1.2 mm, against `igs_orbit_fit`'s 94 mm floor.
    for atlas in ["GOT5.5", "EOT20", "FES2004"] {
        assert!(gps.mm(&format!("{atlas}: total")) < 2.0);
    }
}
