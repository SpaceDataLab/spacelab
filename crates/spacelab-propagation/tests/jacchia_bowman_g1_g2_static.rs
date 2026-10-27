//! (G1) derived constants and an independent Maxima reference; (G2) defining equations.

#![allow(
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::unwrap_used,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::float_cmp
)]
// printed values digit for digit, the papers' notation, test-only unwraps
// Reference values are copied from the Maxima outputs at their printed precision.
#![allow(clippy::excessive_precision)]

use spacelab_propagation::jacchia_bowman::constants::*;
use spacelab_propagation::jacchia_bowman::profile::inflection_temperature;
use spacelab_propagation::jacchia_bowman::quadrature::boole;
use spacelab_propagation::jacchia_bowman::static_model::{
    gravity, log10_hydrogen_500, mean_molar_mass_mixing,
};
use spacelab_propagation::jacchia_bowman::variations::{
    diurnal_ratio, high_altitude_factor, wrap_pi,
};
use spacelab_propagation::jacchia_bowman::{
    NumberDensities, Profile, static_state, static_state_with_step,
};

/// Maxima reference (derivations/jacchia_bowman_static_reference.out): T_inf, z, then log10 of n(N2),
/// n(O2), n(O), n(Ar), n(He), n(H) in m^-3 (0 = none) and log10 rho in kg m^-3, computed
/// with adaptive quadrature (quad_qags, epsrel 1e-13).
const MAXIMA_REF: &[(f64, f64, [f64; 7])] = &[
    (
        600.0,
        95.0,
        [
            19.351406079641773,
            18.75111866195331,
            17.889301913943353,
            17.429046318066202,
            14.568952359189511,
            0.0,
            -5.85928713843183,
        ],
    ),
    (
        600.0,
        100.0,
        [
            18.952649007594072,
            18.317261185535145,
            17.818852355070387,
            17.0302892460185,
            14.170195287141809,
            0.0,
            -6.2580442104795315,
        ],
    ),
    (
        600.0,
        110.0,
        [
            18.19837010199022,
            17.478091935169534,
            17.391286637287124,
            16.140765908237668,
            13.706359462098263,
            0.0,
            -7.013780402473767,
        ],
    ),
    (
        600.0,
        150.0,
        [
            16.337554624279534,
            15.400445991789155,
            16.184107234890256,
            13.630634207417256,
            13.27983123572873,
            0.0,
            -8.808445796787787,
        ],
    ),
    (
        600.0,
        400.0,
        [
            10.58650778689702,
            8.84192603698111,
            12.86719048049824,
            5.461544346191026,
            12.422178860715613,
            0.0,
            -12.667429527771244,
        ],
    ),
    (
        600.0,
        800.0,
        [
            2.585788940380178,
            -0.29688549001739256,
            8.297255360409538,
            -5.947284327313432,
            11.278518051735865,
            11.911124354640046,
            -14.579838500751613,
        ],
    ),
    (
        600.0,
        2000.0,
        [
            -16.805437066612367,
            -22.446847574037857,
            -2.777752643518466,
            -33.59976681669103,
            8.507842519817979,
            11.213342924516093,
            -15.559606898263722,
        ],
    ),
    (
        1000.0,
        95.0,
        [
            19.350149705686096,
            18.74986228799763,
            17.888045539987672,
            17.427789944110526,
            14.567695985233833,
            0.0,
            -5.860543512387507,
        ],
    ),
    (
        1000.0,
        100.0,
        [
            18.948392548607565,
            18.313004726548638,
            17.81459589608388,
            17.026032787031994,
            14.165938828155301,
            0.0,
            -6.262300669466039,
        ],
    ),
    (
        1000.0,
        110.0,
        [
            18.197292836603694,
            17.479968804078894,
            17.381304125393154,
            16.148535034974312,
            13.693684894520876,
            0.0,
            -7.014879390125868,
        ],
    ),
    (
        1000.0,
        150.0,
        [
            16.503303274852726,
            15.60701186122373,
            16.226812259378082,
            13.918613294138146,
            13.243608525206378,
            0.0,
            -8.667447673567443,
        ],
    ),
    (
        1000.0,
        400.0,
        [
            12.794686944494254,
            11.393342507804249,
            14.040679749728932,
            8.697575171983642,
            12.638046505455305,
            0.0,
            -11.4881955491238,
        ],
    ),
    (
        1000.0,
        800.0,
        [
            7.9884005566559555,
            5.903613948112283,
            11.294631795335192,
            1.8446663763055386,
            11.950187009902953,
            10.307405892735936,
            -13.950790130029839,
        ],
    ),
    (
        1000.0,
        2000.0,
        [
            -3.64683916867564,
            -7.38692661853877,
            4.649301315653214,
            -14.747504502940645,
            10.287667656362924,
            9.888634024882574,
            -15.84818694358443,
        ],
    ),
    (
        1900.0,
        95.0,
        [
            19.349090099569725,
            18.748802681881262,
            17.8869859338713,
            17.426730337994154,
            14.566636379117464,
            0.0,
            -5.861603118503877,
        ],
    ),
    (
        1900.0,
        100.0,
        [
            18.94484482983714,
            18.309457007778214,
            17.811048177313456,
            17.022485068261567,
            14.162391109384878,
            0.0,
            -6.265848388236463,
        ],
    ),
    (
        1900.0,
        110.0,
        [
            18.196025424765345,
            17.480909143233745,
            17.37338144170447,
            16.15387891058602,
            13.683758179421496,
            0.0,
            -7.016151823152246,
        ],
    ),
    (
        1900.0,
        150.0,
        [
            16.591288207554957,
            15.719995512258468,
            16.239438482277837,
            14.081458894229137,
            13.212584928424635,
            0.0,
            -8.591253187804206,
        ],
    ),
    (
        1900.0,
        400.0,
        [
            14.158505787807606,
            12.988473178562213,
            14.707208410144819,
            10.754075837904628,
            12.706134284393002,
            0.0,
            -10.68209494630533,
        ],
    ),
    (
        1900.0,
        800.0,
        [
            11.618142559184566,
            10.08755010962381,
            13.253753878653484,
            7.133987244141907,
            12.340307040739178,
            9.011791640605097,
            -12.291558420083721,
        ],
    ),
    (
        1900.0,
        2000.0,
        [
            5.493490222972452,
            3.091601137303017,
            9.755627809869914,
            -1.599832762392549,
            11.465062141545182,
            8.791123927183289,
            -14.679524022385623,
        ],
    ),
];

fn logs(n: &NumberDensities, rho: f64) -> [f64; 7] {
    let l = |v: f64| if v > 0.0 { v.log10() } else { 0.0 };
    [
        l(n.n2),
        l(n.o2),
        l(n.o),
        l(n.ar),
        l(n.he),
        l(n.h),
        rho.log10(),
    ]
}

/// (G1) The quartic coefficients c3/G_x = -17/23275 and c4/G_x = -8/814625 used by the code
/// equal the Maxima solution of eq. (11) (derivations/jacchia_bowman_quartic.out, 30-digit values).
#[test]
fn quartic_coefficients_match_maxima() {
    assert!((QUARTIC_C3_OVER_G - (-7.30397422126745435016111707841e-4)).abs() < 1e-19);
    assert!((QUARTIC_C4_OVER_G - (-9.82046954119993862206536750038e-6)).abs() < 1e-21);
}

/// (G1) T_x(600 K) and T(100 km) at T_inf = 600 K equal the Maxima values
/// 349.624513689967861 K and 192.021735435108631 K (derivations/jacchia_bowman_quartic.out).
#[test]
fn profile_values_match_maxima() {
    assert!((inflection_temperature(600.0) - 349.624513689967861).abs() < 1e-10);
    let p = Profile::new(600.0).unwrap();
    assert!((p.gx - 9.045330743169683885).abs() < 1e-12);
    assert!((p.temperature(100.0) - 192.021735435108631).abs() < 1e-10);
}

/// (G1) Boole weights: the composite rule integrates polynomials of degree <= 5 exactly
/// (derivations/jacchia_bowman_diurnal.out: error 0 up to x^5, -128/21 for x^6 on [0, 4]).
#[test]
fn boole_rule_exactness() {
    for k in 0..=5 {
        let exact = 4f64.powi(k + 1) / (k + 1) as f64;
        let got = boole(|x| x.powi(k), 0.0, 4.0, 4.0);
        assert!(
            (got - exact).abs() < 1e-12 * exact,
            "x^{k}: {got} vs {exact}"
        );
    }
    let got = boole(|x| x.powi(6), 0.0, 4.0, 4.0);
    let exact = 4f64.powi(7) / 7.0;
    assert!(
        (exact - got - (-128.0 / 21.0)).abs() < 1e-9,
        "x^6 error {}",
        exact - got
    );
}

/// (G1) Eq. (17) as printed (product form) equals the single-bracket form in the code
/// (Maxima residual 0, derivations/jacchia_bowman_diurnal.out), checked on a grid of latitude,
/// declination and hour angle.
#[test]
fn diurnal_single_bracket_equals_eq17() {
    let (m, n, r) = (DIURNAL_M, DIURNAL_N, DIURNAL_R);
    for lat in [-80.0f64, -33.0, 0.0, 12.0, 61.0] {
        for dec in [-23.0f64, -5.0, 0.0, 17.0] {
            for h in [-170.0f64, -90.0, -10.0, 0.0, 45.0, 135.0, 179.0] {
                let (lat, dec, h) = (lat.to_radians(), dec.to_radians(), h.to_radians());
                let eta = 0.5 * (lat - dec).abs();
                let theta = 0.5 * (lat + dec).abs();
                let tau = wrap_pi(
                    h + DIURNAL_BETA_DEG.to_radians()
                        + DIURNAL_P_DEG.to_radians() * (h + DIURNAL_GAMMA_DEG.to_radians()).sin(),
                );
                let sm = theta.sin().powf(m);
                let cm = eta.cos().powf(m);
                let eq17 = (1.0 + r * sm)
                    * (1.0 + r * (cm - sm) / (1.0 + r * sm) * (tau / 2.0).cos().powf(n));
                let code = diurnal_ratio(lat, dec, h);
                assert!((eq17 - code).abs() < 1e-14, "{eq17} vs {code}");
            }
        }
    }
}

/// (G1) The Diurnal values at the Table 1 spot cells and the Sec. 15 example equal the Maxima
/// values (derivations/jacchia_bowman_diurnal.out): 1.15392495599141 (45 deg, -20 deg, 11h) and
/// 1.06847221486092 (-90 deg, 23 deg 44').
#[test]
fn diurnal_values_match_maxima() {
    let r = diurnal_ratio(
        45f64.to_radians(),
        (-20f64).to_radians(),
        (-15f64).to_radians(),
    );
    assert!((r - 1.15392495599141023).abs() < 1e-13);
    let dec = (23.0f64 + 44.0 / 60.0).to_radians();
    let r = diurnal_ratio((-90f64).to_radians(), dec, 0.0);
    assert!((r - 1.06847221486092444).abs() < 1e-13);
}

/// (G1) High-altitude cubic coefficients a2, a3 at F10bar = 70, 150, 230 equal the Maxima
/// values (derivations/jacchia_bowman_hermite.out), checked through the factor at H = 1/2.
#[test]
fn hermite_blend_matches_maxima() {
    for (fb, a2, a3) in [
        (70.0, 1.2492, -0.66575),
        (150.0, 0.094, 0.07625),
        (230.0, -1.0612, 0.81825),
    ] {
        let expect = 1.0 + a2 * 0.25 + a3 * 0.125;
        let got = high_altitude_factor(fb, 1250.0);
        assert!(
            (got - expect).abs() < 1e-14,
            "F10bar {fb}: {got} vs {expect}"
        );
    }
}

/// (G2) The 1000-1500 km blend meets its four end conditions: value 1 and slope 0 at 1000 km,
/// value and slope of B06 eq. (10) at 1500 km (one-sided differences, h = 1e-5 km).
#[test]
fn hermite_blend_end_conditions() {
    let h = 1e-5;
    for fb in [65.0, 120.0, 180.0, 260.0] {
        let f = |z: f64| high_altitude_factor(fb, z);
        let eq10 = |z: f64| {
            HIGH_ALT_C[0] + HIGH_ALT_C[1] * fb + HIGH_ALT_C[2] * z + HIGH_ALT_C[3] * z * fb
        };
        assert_eq!(f(1000.0), 1.0);
        assert!((f(1000.0 + h) - 1.0).abs() < 1e-6);
        assert!((f(1500.0 - 1e-9) - eq10(1500.0)).abs() < 1e-10);
        let slope_left = (f(1500.0 - h) - f(1500.0 - 2.0 * h)) / h;
        let slope_eq10 = HIGH_ALT_C[2] + HIGH_ALT_C[3] * fb;
        assert!(
            (slope_left - slope_eq10).abs() < 1e-9,
            "{slope_left} vs {slope_eq10}"
        );
    }
}

/// (G2) Temperature profile, SR313 eqs. (10)-(13): T(90) = 183 K, T'(90) = 0, T(125) = T_x,
/// T'(125) = G_x from both sides, T''(125) = 0 from both sides, and T -> T_inf.
#[test]
fn temperature_profile_conditions() {
    let h = 1e-4;
    for t_inf in [500.0, 600.0, 1000.0, 1500.0, 1900.0, 2500.0] {
        let p = Profile::new(t_inf).unwrap();
        assert!((p.temperature(Z0) - T0).abs() < 1e-10, "T(90)");
        assert!(p.gradient(Z0).abs() < 1e-12, "T'(90)");
        assert!((p.temperature(ZX) - p.tx).abs() < 1e-12, "T(125)");
        let below = (p.temperature(ZX) - p.temperature(ZX - h)) / h;
        let above = (p.temperature(ZX + h) - p.temperature(ZX)) / h;
        assert!((below - p.gx).abs() < 1e-5 * p.gx && (above - p.gx).abs() < 1e-5 * p.gx);
        let h2 = 1e-2;
        let curv_below = (p.temperature(ZX) - 2.0 * p.temperature(ZX - h2)
            + p.temperature(ZX - 2.0 * h2))
            / (h2 * h2);
        let curv_above = (p.temperature(ZX + 2.0 * h2) - 2.0 * p.temperature(ZX + h2)
            + p.temperature(ZX))
            / (h2 * h2);
        // one-sided second differences are O(h) accurate: |T'''| h ~ 1e-3 K/km^2 here
        assert!(
            curv_below.abs() < 5e-3 && curv_above.abs() < 5e-3,
            "{curv_below} {curv_above}"
        );
        assert!(
            (p.temperature(1.0e7) - t_inf).abs() < 1e-3 * t_inf,
            "T at 1e7 km"
        );
        // monotonic rise
        let mut prev = p.temperature(Z0);
        for k in 1..=2000 {
            let t = p.temperature(Z0 + k as f64);
            assert!(t > prev && t < t_inf);
            prev = t;
        }
        // analytic gradient against central differences
        for z in [95.0, 110.0, 124.0, 126.0, 200.0, 800.0] {
            let fd = (p.temperature(z + h) - p.temperature(z - h)) / (2.0 * h);
            assert!(
                (fd - p.gradient(z)).abs() < 1e-6 * (1.0 + p.gradient(z).abs()),
                "z {z}"
            );
        }
    }
}

/// (G2) The no-inflection condition of SR313 eq. (12), 4/3 < (z_x - z_0) G_x/(T_x - T_0) < 2:
/// the code's ratio is 1.90 for every profile.
#[test]
fn no_inflection_condition() {
    for t_inf in [400.0, 900.0, 2000.0] {
        let p = Profile::new(t_inf).unwrap();
        let q = (ZX - Z0) * p.gx / (p.tx - T0);
        assert!((q - 1.90).abs() < 1e-12 && q > 4.0 / 3.0 && q < 2.0);
    }
}

/// (G2) Densities above the homopause satisfy the diffusion equation (6) for every species:
/// d ln n_i/dz = -M_i g/(R* T) - (1 + alpha_i) d ln T/dz, by central differences (h = 0.01 km),
/// to a relative 1e-6.
#[test]
fn diffusion_equation_by_finite_differences() {
    let h = 0.01;
    for t_inf in [600.0, 1200.0, 2000.0] {
        let p = Profile::new(t_inf).unwrap();
        for z in [110.0, 130.0, 250.0, 480.0, 520.0, 900.0, 2000.0] {
            let lo = static_state(&p, z - h).n;
            let hi = static_state(&p, z + h).n;
            let t = p.temperature(z);
            let dlnt = p.gradient(z) / t;
            let g = gravity(z) * M_PER_KM;
            let pairs = [
                (lo.n2, hi.n2, M_N2, 0.0),
                (lo.o2, hi.o2, M_O2, 0.0),
                (lo.o, hi.o, M_O, 0.0),
                (lo.ar, hi.ar, M_AR, 0.0),
                (lo.he, hi.he, M_HE, ALPHA_HE),
                (lo.h, hi.h, M_H, 0.0),
            ];
            for (k, (a, b, m, alpha)) in pairs.iter().enumerate() {
                if *a == 0.0 {
                    continue; // hydrogen below 500 km
                }
                let fd = (b.ln() - a.ln()) / (2.0 * h);
                let eq6 = -m * g / (R_STAR * t) - (1.0 + alpha) * dlnt;
                assert!(
                    (fd - eq6).abs() < 1e-6 * eq6.abs(),
                    "T_inf {t_inf} z {z} species {k}: {fd} vs {eq6}"
                );
            }
        }
    }
}

/// (G2) Below the homopause the density satisfies the barometric equation (5):
/// d ln rho/dz = d ln(M/T)/dz - M g/(R* T), by central differences.
#[test]
fn barometric_equation_by_finite_differences() {
    let h = 0.01;
    let p = Profile::new(900.0).unwrap();
    for z in [91.0, 95.0, 100.0, 104.0] {
        let rho = |s: f64| static_state(&p, s).density;
        let fd = (rho(z + h).ln() - rho(z - h).ln()) / (2.0 * h);
        let m = |s: f64| mean_molar_mass_mixing(s);
        let dlnm = (m(z + h).ln() - m(z - h).ln()) / (2.0 * h);
        let t = p.temperature(z);
        let eq5 = dlnm - p.gradient(z) / t - m(z) * gravity(z) * M_PER_KM / (R_STAR * t);
        assert!((fd - eq5).abs() < 1e-7 * eq5.abs(), "z {z}: {fd} vs {eq5}");
    }
}

/// (G2) The species densities are continuous at the homopause (105 km) and the total
/// composition below it satisfies eqs. (2)-(4): sum n_i M_i / A = rho, and M = rho A / N.
#[test]
fn homopause_continuity_and_mixing_composition() {
    let p = Profile::new(1100.0).unwrap();
    let below = static_state(&p, Z_HOMOPAUSE).n;
    let above = static_state(&p, Z_HOMOPAUSE + 1e-9).n;
    for (a, b) in [
        (below.n2, above.n2),
        (below.o, above.o),
        (below.he, above.he),
    ] {
        assert!((a / b - 1.0).abs() < 1e-8);
    }
    for z in [90.0, 97.0, 105.0] {
        let s = static_state(&p, z);
        let mbar = s.density * AVOGADRO / s.n.total();
        // eqs. (3)-(4) conserve mass and particle number exactly only if q0 sums to 1 and
        // M_0 = sum q0 M_i. The printed q0 sum to 0.99999289 and give sum q0 M_i = 28.95978
        // against M_0 = 28.960: mass -7.5e-6, particles +2.9e-6, mean mass -1.03e-5 relative
        // at 90 km.
        assert!(
            (mbar / mean_molar_mass_mixing(z) - 1.0).abs() < 1.1e-5,
            "z {z}: {mbar}"
        );
        assert!((s.density / s.n.mass_density() - 1.0).abs() < 1e-15);
    }
}

/// (G2) Hydrogen at 500 km equals eq. (7) with the profile temperature at 500 km (SR332 p. 9).
#[test]
fn hydrogen_boundary_value() {
    for t_inf in [600.0, 1000.0, 1900.0] {
        let p = Profile::new(t_inf).unwrap();
        let s = static_state(&p, Z_HYDROGEN);
        let expect = log10_hydrogen_500(p.temperature(Z_HYDROGEN));
        assert!((s.n.h.log10() - expect).abs() < 1e-12);
        assert_eq!(static_state(&p, Z_HYDROGEN - 1e-6).n.h, 0.0);
    }
}

/// (G2) Quadrature convergence: halving the default largest panel (1 km -> 0.5 km) changes
/// log10 rho and every log10 n_i by less than 5e-12 for 90-2500 km and T_inf 500-2500 K
/// (sixth-order convergence: 4 km -> 2 km -> 1 km change ~3e-9, 5e-11, 1e-12).
#[test]
fn quadrature_converges() {
    let mut worst: f64 = 0.0;
    for t_inf in [500.0, 800.0, 1200.0, 1900.0, 2500.0] {
        let p = Profile::new(t_inf).unwrap();
        for z in [
            90.5, 99.0, 105.0, 117.0, 125.0, 133.0, 180.0, 350.0, 600.0, 1100.0, 2500.0,
        ] {
            let a = static_state_with_step(&p, z, 1.0);
            let b = static_state_with_step(&p, z, 0.5);
            let la = logs(&a.n, a.density);
            let lb = logs(&b.n, b.density);
            for k in 0..7 {
                worst = worst.max((la[k] - lb[k]).abs());
            }
        }
    }
    assert!(worst < 5e-12, "worst change {worst}");
}

/// (G1) The static model equals an independent Maxima evaluation of the same equations with
/// adaptive quadrature (derivations/jacchia_bowman_static_reference.mac) to 1e-9 in every log10.
#[test]
fn static_model_matches_maxima_reference() {
    for (t_inf, z, r) in MAXIMA_REF {
        let p = Profile::new(*t_inf).unwrap();
        let s = static_state(&p, *z);
        let l = logs(&s.n, s.density);
        for k in 0..7 {
            assert!(
                (l[k] - r[k]).abs() < 1e-9,
                "T_inf {t_inf} z {z} col {k}: {} vs {}",
                l[k],
                r[k]
            );
        }
    }
}

/// (G2) Gravity eq. (8): g(0) = 9.80665 m s^-2 and g scales as (1 + z/R_e)^-2.
#[test]
fn gravity_law() {
    assert_eq!(gravity(0.0), 9.80665);
    let z = 1000.0;
    assert!((gravity(z) * (1.0 + z / 6356.766f64).powi(2) - 9.80665).abs() < 1e-12);
}
