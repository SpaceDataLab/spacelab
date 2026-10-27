//! SR313 Table 5 (Jacchia 1970), transcribed from the page images (PDF pages 51-52, 59 and 61,
//! 78-79; rendered at 200-400 dpi and rotated, since the text layer is unusable). Units as
//! printed: T in K, log10 n in cm^-3, log10 rho in g cm^-3, mean molecular weight in g/mol.
//! A NaN marks a blank cell (species not tabulated at that height).

#![allow(
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::unwrap_used,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::float_cmp
)] // printed values digit for digit, the papers' notation, test-only unwraps

#[path = "jacchia_bowman_common/mod.rs"]
mod common;
use common::{log10_g_cm3, log10_per_cm3};
use spacelab_propagation::jacchia_bowman::{Profile, static_state};

const NA: f64 = f64::NAN;

/// (z km, T, log N2, log O2, log O, log Ar, log He, log H, mean mol. wt, log rho)
type Row = (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64);

/// T_inf = 600 K: SR313 Table 5, printed p. 52 and its continuation (PDF 51-52).
const T600: &[Row] = &[
    (
        90.0, 183.0, 13.7498, 13.1724, 11.6094, 11.8276, 8.9685, NA, 28.88, -8.461,
    ),
    (
        100.0, 192.0, 12.9527, 12.3173, 11.8187, 11.0304, 8.1714, NA, 28.15, -9.258,
    ),
    (
        110.0, 231.7, 12.1984, 11.4781, 11.3912, 10.1409, 7.7075, NA, 27.27, -10.014,
    ),
    (
        120.0, 305.2, 11.5552, 10.7604, 10.9726, 9.2746, 7.5586, NA, 26.30, -10.650,
    ),
    (
        125.0, 349.6, 11.2847, 10.4599, 10.7928, 8.9141, 7.4917, NA, 25.83, -10.914,
    ),
    (
        150.0, 503.2, 10.3376, 9.4005, 10.1840, 7.6308, 7.2810, NA, 23.63, -11.808,
    ),
    (
        200.0, 569.7, 9.0365, 7.9220, 9.4178, 5.7984, 7.0694, NA, 19.74, -12.904,
    ),
    (
        250.0, 587.4, 7.8691, 6.5904, 8.7453, 4.1393, 6.8962, NA, 17.28, -13.733,
    ),
    (
        300.0, 594.3, 6.7493, 5.3120, 8.1036, 2.5446, 6.7338, NA, 15.84, -14.434,
    ),
    (
        400.0, 598.5, 4.5865, 2.8420, 6.8671, NA, 6.4233, NA, 11.27, -15.662,
    ),
    (
        500.0, 599.4, 2.4965, 0.4547, 5.6731, NA, 6.1244, 6.1241, 4.54, -16.627,
    ),
    (
        600.0, 599.7, 0.4690, NA, 4.5150, NA, 5.8346, 6.0510, 2.39, -17.137,
    ),
    (
        800.0, 599.9, NA, NA, 2.2972, NA, 5.2797, 5.9111, 1.58, -17.579,
    ),
    (
        1000.0, NA, NA, NA, 0.2003, NA, 4.7551, 5.7790, 1.27, -17.859,
    ),
    (1500.0, NA, NA, NA, NA, NA, 3.5606, 5.4782, 1.04, -18.278),
    (2000.0, NA, NA, NA, NA, NA, NA, 5.2133, 1.01, -18.560),
    (2500.0, NA, NA, NA, NA, NA, 1.5762, 4.9784, 1.01, -18.797),
];

/// T_inf = 1000 K: SR313 Table 5 (PDF 59 and 61; PDF 60 is a microfiche target).
const T1000: &[Row] = &[
    (
        90.0, 183.0, 13.7498, 13.1724, 11.6094, 11.8276, 8.9685, NA, 28.88, -8.461,
    ),
    (
        100.0, 195.9, 12.9484, 12.3130, 11.8144, 11.0262, 8.1671, NA, 28.15, -9.262,
    ),
    (
        110.0, 252.9, 12.1973, 11.4800, 11.3812, 10.1487, 7.6948, NA, 27.30, -10.015,
    ),
    (
        120.0, 358.1, 11.5831, 10.8000, 10.9656, 9.3373, 7.5350, NA, 26.47, -10.624,
    ),
    (
        125.0, 421.8, 11.3344, 10.5260, 10.7931, 9.0129, 7.4655, NA, 26.09, -10.868,
    ),
    (
        150.0, 690.1, 10.5033, 9.6070, 10.2267, 7.9188, 7.2448, NA, 24.50, -11.667,
    ),
    (
        200.0, 889.2, 9.5513, 8.5352, 9.6358, 6.6081, 7.0562, NA, 21.85, -12.524,
    ),
    (
        250.0, 953.4, 8.7982, 7.6793, 9.1926, 5.5470, 6.9342, NA, 19.66, -13.135,
    ),
    (
        300.0, 978.9, 8.1057, 6.8899, 8.7922, 4.5643, 6.8298, NA, 18.07, -13.641,
    ),
    (
        400.0, 994.3, 6.7947, 5.3934, 8.0406, 2.6977, 6.6392, NA, 16.22, -14.488,
    ),
    (
        500.0, 997.9, 5.5371, 3.9571, 7.3217, 0.9050, 6.4588, 4.4359, 14.74, -15.227,
    ),
    (
        600.0, 999.0, 4.3192, 2.5660, 6.6259, NA, 6.2845, 4.3916, 12.23, -15.899,
    ),
    (
        800.0, 999.7, 1.9884, NA, 5.2946, NA, 5.9513, 4.3074, 6.08, -16.950,
    ),
    (
        1000.0, 999.9, NA, NA, 4.0362, NA, 5.6365, 4.2281, 4.18, -17.496,
    ),
    (
        1500.0, 1000.0, NA, NA, 1.1712, NA, 4.9198, 4.0475, 3.65, -18.243,
    ),
    (
        2000.0, 1000.0, NA, NA, NA, NA, 4.2888, 3.8886, 3.15, -18.847,
    ),
    (
        2500.0, 1000.0, NA, NA, NA, NA, 3.7291, 3.7477, 2.47, -19.347,
    ),
];

/// T_inf = 1900 K: SR313 Table 5 (PDF 78-79).
const T1900: &[Row] = &[
    (
        90.0, 183.0, 13.7498, 13.1724, 11.6094, 11.8276, 8.9685, NA, 28.88, -8.461,
    ),
    (
        100.0, 199.2, 12.9448, 12.3095, 11.8109, 11.0226, 8.1635, NA, 28.15, -9.266,
    ),
    (
        110.0, 270.7, 12.1960, 11.4809, 11.3733, 10.1540, 7.6849, NA, 27.32, -10.016,
    ),
    (
        120.0, 402.9, 11.6014, 10.8263, 10.9597, 9.3797, 7.5176, NA, 26.59, -10.607,
    ),
    (
        125.0, 482.9, 11.3663, 10.5689, 10.7916, 9.0779, 7.4465, NA, 26.26, -10.838,
    ),
    (
        150.0, 870.0, 10.5913, 9.7200, 10.2393, 8.0816, 7.2137, NA, 24.99, -11.591,
    ),
    (
        200.0, 1408.5, 9.7879, 8.8321, 9.6908, 7.0251, 6.9991, NA, 23.21, -12.344,
    ),
    (
        250.0, 1680.5, 9.2816, 8.2646, 9.3687, 6.3357, 6.8902, NA, 21.82, -12.793,
    ),
    (
        300.0, 1799.2, 8.8747, 7.8041, 9.1236, 5.7681, 6.8179, NA, 20.63, -13.133,
    ),
    (
        400.0, 1872.6, 8.1585, 6.9885, 8.7071, 4.7542, 6.7073, NA, 18.73, -13.682,
    ),
    (
        500.0, 1889.9, 7.4899, 6.2253, 8.3235, 3.8024, 6.6098, 3.0811, 17.41, -14.146,
    ),
    (
        600.0, 1895.4, 6.8465, 5.4906, 7.9555, 2.8855, 6.5173, 3.0567, 16.49, -14.559,
    ),
    (
        800.0, 1898.6, 5.6181, 4.0876, 7.2537, 1.1342, 6.3415, 3.0118, 14.97, -15.292,
    ),
    (
        1000.0, 1899.4, 4.4581, 2.7626, 6.5911, NA, 6.1756, 2.9699, 12.75, -15.940,
    ),
    (
        1500.0, 1899.9, 1.8177, NA, 5.0830, NA, 5.7983, 2.8748, 5.94, -17.131,
    ),
    (
        2000.0, 1900.0, NA, NA, 3.7555, NA, 5.4662, 2.7911, 4.23, -17.678,
    ),
    (
        2500.0, 1900.0, NA, NA, 2.5780, NA, 5.1716, 2.7169, 4.02, -18.001,
    ),
];

const NAMES: [&str; 9] = ["T", "N2", "O2", "O", "Ar", "He", "H", "Mbar", "rho"];

/// Model values in the table's columns and units.
fn model_row(t_inf: f64, z: f64) -> [f64; 9] {
    let p = Profile::new(t_inf).unwrap();
    let s = static_state(&p, z);
    let n = s.n;
    let lg = |v: f64| if v > 0.0 { log10_per_cm3(v) } else { NA };
    [
        s.temperature,
        lg(n.n2),
        lg(n.o2),
        lg(n.o),
        lg(n.ar),
        lg(n.he),
        lg(n.h),
        n.mean_molar_mass() * 1.0e3,
        log10_g_cm3(s.density),
    ]
}

fn printed(r: &Row) -> [f64; 9] {
    [r.1, r.2, r.3, r.4, r.5, r.6, r.7, r.8, r.9]
}

/// Differences model - printed for every printed cell.
fn diffs(t_inf: f64, rows: &[Row]) -> Vec<(f64, usize, f64, f64)> {
    let mut out = Vec::new();
    for r in rows {
        let m = model_row(t_inf, r.0);
        for (k, v) in printed(r).iter().enumerate() {
            if v.is_finite() {
                out.push((r.0, k, *v, m[k] - v));
            }
        }
    }
    out
}

/// Half a unit of the last printed digit of each column.
const HALF_UNIT: [f64; 9] = [0.05, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 5e-5, 0.005, 5e-4];

/// Columns whose printed values the static model must reproduce at the printed digits:
/// T, N2, O2, H. O, Ar and He carry boundary offsets (see the tests below), and the mean
/// molecular weight and the density carry them and the unlisted hydrogen below 500 km.
fn check_exact_columns(t_inf: f64, rows: &[Row]) {
    let mut bad = Vec::new();
    for (z, k, v, d) in diffs(t_inf, rows) {
        // One unit of the last digit: Jacchia's own rounding of quantities near a half unit.
        let tol = if [0usize, 1, 2, 6].contains(&k) {
            2.0 * HALF_UNIT[k]
        } else {
            continue;
        };
        if d.abs() > tol {
            bad.push(format!(
                "T_inf {t_inf} z {z} {}: printed {v}, model - printed {d:+.5}",
                NAMES[k]
            ));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

/// (G5) SR313 Table 5, T_inf = 600 K (p. 52): T, log n(N2), log n(O2) and log n(H) agree within
/// one unit of the last printed digit at 90-2500 km.
#[test]
fn sr313_table5_600k_t_n2_o2_h() {
    check_exact_columns(600.0, T600);
}

/// (G5) SR313 Table 5, T_inf = 1000 K: T, log n(N2), log n(O2), log n(H) within one unit of the
/// last printed digit.
#[test]
fn sr313_table5_1000k_t_n2_o2_h() {
    check_exact_columns(1000.0, T1000);
}

/// (G5) SR313 Table 5, T_inf = 1900 K: T, log n(N2), log n(O2), log n(H) within one unit of the
/// last printed digit.
#[test]
fn sr313_table5_1900k_t_n2_o2_h() {
    check_exact_columns(1900.0, T1900);
}

/// (G5) Report: prints every printed cell with the model difference (run with --nocapture).
#[test]
fn sr313_table5_report() {
    for (t_inf, rows) in [(600.0, T600), (1000.0, T1000), (1900.0, T1900)] {
        for (z, k, v, d) in diffs(t_inf, rows) {
            let units = d / (2.0 * HALF_UNIT[k]);
            println!(
                "{t_inf:6} {z:6} {:4} printed {v:>10} diff {d:+.5} ({units:+.1} last-digit units)",
                NAMES[k]
            );
        }
    }
}

fn column_diffs(k: usize) -> Vec<(f64, f64, f64)> {
    let mut out = Vec::new();
    for (t_inf, rows) in [(600.0, T600), (1000.0, T1000), (1900.0, T1900)] {
        for (z, kk, _, d) in diffs(t_inf, rows) {
            if kk == k {
                out.push((t_inf, z, d));
            }
        }
    }
    out
}

/// (G5) Printed He disagrees with eq. (3) + q0(He) = 0.00001289 (SR313 p. 3) by a constant
/// offset: model - printed = -0.00116 +- 0.0001 in log n(He) at every height and T_inf (90 to
/// 2500 km). A constant offset points at the 90 km boundary composition, not at the
/// integration: the table behaves as if q0(He) were 1.289e-5 x 10^0.00116 = 1.2925e-5.
/// Not tuned; reported as a possible erratum of the printed q0(He).
#[test]
fn sr313_table5_helium_constant_offset() {
    for (t_inf, z, d) in column_diffs(5) {
        assert!(
            (d + 0.00116).abs() < 1.0e-4,
            "T_inf {t_inf} z {z}: He diff {d}"
        );
    }
}

/// (G5) Printed Ar is above the model by 0.00011-0.00022 in log n(Ar) at every height.
/// With q0(Ar) = 0.0093432 (the SR332 p. 5 value) instead of SR313's 0.00934 the model agrees
/// within one unit of the last printed digit: the SR313 table appears computed with the
/// unrounded fraction. Not tuned (the census prescribes the SR313 value); reported.
#[test]
fn sr313_table5_argon_matches_sr332_fraction() {
    let shift = (0.0093432f64 / 0.00934).log10();
    for (t_inf, z, d) in column_diffs(4) {
        assert!(
            (-2.3e-4..=-1.0e-4).contains(&d),
            "T_inf {t_inf} z {z}: Ar diff {d}"
        );
        assert!(
            (d + shift).abs() <= 1.0e-4,
            "T_inf {t_inf} z {z}: Ar diff with SR332 q0 {}",
            d + shift
        );
    }
}

/// (G5) Atomic oxygen: within 2 units of the last printed digit (0.0002) above 90 km; at 90 km
/// the model is 0.0015 above the printed 11.6094 for every T_inf. Eqs. (2)-(4) with the printed
/// rho_1, M(90) from eq. (1) and M_0 = 28.960 give 11.6109; the difference corresponds to a
/// change of 1 - M/M_0 by 0.35 %, e.g. M_0 = 28.95968 instead of 28.960. Reported, not tuned.
#[test]
fn sr313_table5_atomic_oxygen() {
    for (t_inf, z, d) in column_diffs(3) {
        if z == 90.0 {
            assert!(
                (d - 0.0015).abs() < 1.0e-4,
                "T_inf {t_inf}: O diff at 90 km {d}"
            );
        } else {
            assert!(d.abs() <= 2.0e-4, "T_inf {t_inf} z {z}: O diff {d}");
        }
    }
}

/// (G5) log10 rho: within 0.0016 (1.6 units of the last printed digit) of the printed value at
/// all transcribed heights and T_inf, except T_inf = 600 K at 400 km (-0.0054), where the
/// printed total includes hydrogen below 500 km (see the next test). The residual is the He
/// offset where He carries the mass (high altitude, 1000-1900 K).
#[test]
fn sr313_table5_log_density() {
    for (t_inf, z, d) in column_diffs(8) {
        if t_inf == 600.0 && z == 400.0 {
            continue;
        }
        assert!(d.abs() <= 1.6e-3, "T_inf {t_inf} z {z}: log rho diff {d}");
    }
}

/// (G5) SR313 p. 7 says no hydrogen is computed below 500 km and the census keeps it out.
/// The printed T_inf = 600 K row at 400 km (rho 2.177e-16 g/cm3, mean mol. wt 11.27) is,
/// however, reproduced only if hydrogen is carried below 500 km by eq. (6) from the eq. (7)
/// value at 500 km: the table totals include unlisted hydrogen. This test computes that
/// hydrogen in the test only, to explain the printed row; the model does not include it.
#[test]
fn sr313_table5_600k_400km_includes_unlisted_hydrogen() {
    use spacelab_propagation::jacchia_bowman::constants::{AVOGADRO, M_H, M_PER_KM, R_STAR};
    use spacelab_propagation::jacchia_bowman::quadrature::boole;
    use spacelab_propagation::jacchia_bowman::static_model::gravity;
    let p = Profile::new(600.0).unwrap();
    let s500 = static_state(&p, 500.0);
    let s400 = static_state(&p, 400.0);
    let integral = boole(|z| gravity(z) / p.temperature(z), 400.0, 500.0, 0.5) * M_PER_KM;
    let n_h = s500.n.h * (s500.temperature / s400.temperature) * (M_H * integral / R_STAR).exp();
    let rho = s400.density + n_h * M_H / AVOGADRO;
    let mbar = rho * AVOGADRO / (s400.n.total() + n_h) * 1.0e3;
    let without = log10_g_cm3(s400.density) - (-15.662);
    let with = log10_g_cm3(rho) - (-15.662);
    println!(
        "600 K 400 km: log rho diff without H {without:+.5}, with H {with:+.5}, Mbar with H {mbar:.3}, log n(H) {:.4}",
        log10_per_cm3(n_h)
    );
    assert!(without < -5.0e-3, "without H: {without}");
    assert!(with.abs() < 1.0e-3, "with H: {with}");
    assert!(
        (mbar - 11.27).abs() < 0.01,
        "mean molecular weight with H: {mbar}"
    );
}
