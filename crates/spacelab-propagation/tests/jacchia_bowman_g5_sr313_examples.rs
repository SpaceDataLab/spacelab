//! Printed vectors of SR313 other than Table 5: the Sec. 15 worked example (p. 35-37),
//! Table 1 (p. 46, diurnal ratio) and the hours of extreme temperature (p. 19).

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
use spacelab_propagation::jacchia_bowman::constants::{SEASLAT_AMP, SEASLAT_DECAY, SEASLAT_Z_REF};
use spacelab_propagation::jacchia_bowman::variations::{diurnal_ratio, seasonal_latitudinal_log10};
use spacelab_propagation::jacchia_bowman::{Profile, static_state};

fn log_rho(t_inf: f64, z: f64) -> f64 {
    log10_g_cm3(static_state(&Profile::new(t_inf).unwrap(), z).density)
}

/// (G5) SR313 Sec. 15, p. 35: phi = +45 deg, 11h00m L.S.T., sun declination -20.0 deg,
/// R = 0.31: Table 1 gives T_l/T_c = 1.154. Eq. (17) gives 1.15392.
#[test]
fn sr313_example_diurnal_ratio() {
    let r = diurnal_ratio(
        45f64.to_radians(),
        (-20f64).to_radians(),
        (-15f64).to_radians(),
    );
    assert!((r - 1.154).abs() < 5.0e-4, "{r}");
}

/// (G5) SR313 Sec. 15, p. 36: Table 6 at T_inf = 1053 K gives log rho = -14.011 at 350 km and
/// -11.403 at 140 km; p. 37: -16.815 at 800 km (g cm^-3). The static model agrees within
/// one unit of the third decimal (Table 6 is itself interpolated in steps of 50 K).
#[test]
fn sr313_example_table6_densities() {
    for (z, printed) in [(350.0, -14.011), (140.0, -11.403), (800.0, -16.815)] {
        let d = log_rho(1053.0, z) - printed;
        assert!(d.abs() <= 1.0e-3, "z {z}: model - printed {d:+.5}");
    }
}

/// (G5) SR313 Sec. 15, p. 36: "From Table 5 we find, by interpolation, for T_inf = 1051",
/// at 800 km, log n(O) = 5.513 and log n_0(He) = 5.998 (cm^-3). The printed values belong to
/// T_inf = 1053 K, the temperature of the rest of the example: there the model gives O within
/// 0.001 and He 0.0012 low (the Table 5 helium offset, test
/// `sr313_table5_helium_constant_offset`); at 1051 K O is 0.008 low. "1051" is reported as a
/// misprint for 1053.
#[test]
fn sr313_example_800km_composition() {
    let at = |t: f64| static_state(&Profile::new(t).unwrap(), 800.0).n;
    let n = at(1053.0);
    let d_o = log10_per_cm3(n.o) - 5.513;
    let d_he = log10_per_cm3(n.he) - 5.998;
    assert!(d_o.abs() <= 1.0e-3, "O at 1053 K: {d_o:+.5}");
    assert!((d_he + 0.00116).abs() <= 1.0e-3, "He at 1053 K: {d_he:+.5}");
    let d_o_1051 = log10_per_cm3(at(1051.0).o) - 5.513;
    assert!(d_o_1051 < -5.0e-3, "O at 1051 K: {d_o_1051:+.5}");
}

/// (G5) SR313 Sec. 15, p. 36: at 140 km Table 4 gives S = 0.105 (eq. (24) amplitude
/// 0.02 (z - 90) exp(-0.045 (z - 90))).
#[test]
fn sr313_example_seasonal_amplitude_140km() {
    let dz = 140.0 - SEASLAT_Z_REF;
    let s = SEASLAT_AMP * dz * (-SEASLAT_DECAY * dz).exp();
    assert!((s - 0.105).abs() < 5.0e-4, "{s}");
}

/// (G5) SR313 Sec. 15, p. 36: on 1969 January 20.8 the J70 phase sin[(360/Y)(d + 100)] gives
/// P = +0.882 and Delta log rho = S P sin^2(phi) = +0.046 at 140 km, phi = 45 deg. The census
/// prescribes the SR332 phase sin(2 pi Phi + 1.72) (SR332 p. 46): it gives P = 0.8794 and the
/// same +0.046 at the printed three decimals. MJD of 1969 Jan 20.8 = 36204 + 4037.8 (11 years
/// from 1958 Jan 1 with 3 leap days, plus 19.8 days).
#[test]
fn sr313_example_seasonal_latitudinal_140km() {
    let mjd = 36204.0 + 11.0 * 365.0 + 3.0 + 19.8;
    let dl = seasonal_latitudinal_log10(140.0, 45f64.to_radians(), mjd);
    assert!((dl - 0.046).abs() < 5.0e-4, "{dl}");
}

/// SR313 Table 1 (p. 46), T_l/T_c x 1000, R = 0.31, hours 12..23 L.S.T., latitudes +90..-90
/// in steps of 15 deg. Read from a 300 dpi, deskewed render of PDF page 45. Hour 19 at -45 deg
/// in the second panel is illegible (0 = not used).
const TABLE1_PANEL1: [[u32; 12]; 13] = [
    [1198; 12],
    [
        1229, 1236, 1239, 1237, 1231, 1222, 1211, 1200, 1188, 1178, 1170, 1163,
    ],
    [
        1253, 1267, 1272, 1269, 1258, 1241, 1220, 1198, 1177, 1157, 1140, 1128,
    ],
    [
        1270, 1289, 1296, 1292, 1276, 1253, 1224, 1193, 1163, 1136, 1113, 1095,
    ],
    [
        1277, 1300, 1309, 1303, 1284, 1256, 1222, 1185, 1150, 1117, 1089, 1068,
    ],
    [
        1274, 1299, 1308, 1301, 1281, 1251, 1214, 1174, 1136, 1100, 1070, 1047,
    ],
    [
        1260, 1285, 1294, 1288, 1268, 1237, 1200, 1161, 1122, 1087, 1057, 1034,
    ],
    [
        1237, 1260, 1269, 1263, 1244, 1216, 1181, 1145, 1109, 1076, 1048, 1027,
    ],
    [
        1206, 1226, 1234, 1229, 1212, 1188, 1158, 1126, 1094, 1066, 1042, 1023,
    ],
    [
        1171, 1187, 1193, 1189, 1176, 1156, 1132, 1106, 1080, 1057, 1038, 1023,
    ],
    [
        1134, 1145, 1149, 1146, 1137, 1123, 1106, 1088, 1070, 1054, 1040, 1030,
    ],
    [
        1099, 1105, 1107, 1105, 1101, 1093, 1085, 1075, 1066, 1058, 1051, 1045,
    ],
    [1069; 12],
];
const TABLE1_PANEL2: [[u32; 12]; 13] = [
    [1188; 12],
    [
        1220, 1227, 1230, 1228, 1222, 1213, 1202, 1190, 1179, 1169, 1160, 1153,
    ],
    [
        1246, 1260, 1265, 1262, 1250, 1233, 1212, 1190, 1168, 1148, 1132, 1119,
    ],
    [
        1265, 1285, 1292, 1287, 1271, 1247, 1218, 1187, 1157, 1129, 1106, 1088,
    ],
    [
        1275, 1298, 1307, 1301, 1282, 1254, 1219, 1182, 1145, 1112, 1084, 1062,
    ],
    [
        1275, 1300, 1309, 1303, 1282, 1251, 1214, 1173, 1134, 1098, 1067, 1044,
    ],
    [
        1264, 1289, 1298, 1292, 1271, 1240, 1203, 1162, 1123, 1087, 1056, 1033,
    ],
    [
        1243, 1266, 1275, 1269, 1250, 1221, 1186, 1148, 1111, 1077, 1049, 1027,
    ],
    [
        1214, 1235, 1242, 1237, 1220, 1195, 1164, 1131, 1098, 1069, 1044, 1024,
    ],
    [
        1179, 1196, 1202, 1198, 1185, 1164, 1139, 0, 1086, 1062, 1041, 1026,
    ],
    [
        1143, 1155, 1159, 1156, 1147, 1132, 1114, 1095, 1077, 1060, 1046, 1035,
    ],
    [
        1108, 1114, 1116, 1115, 1110, 1102, 1093, 1083, 1074, 1065, 1058, 1052,
    ],
    [1077; 12],
];

/// Largest |model x 1000 - printed| over a panel, for the declination `dec_deg`.
fn table1_max_error(panel: &[[u32; 12]; 13], dec_deg: f64) -> f64 {
    let mut worst: f64 = 0.0;
    for (i, row) in panel.iter().enumerate() {
        let lat = (90.0 - 15.0 * i as f64).to_radians();
        for (j, &v) in row.iter().enumerate() {
            if v == 0 {
                continue;
            }
            let hour_angle = (15.0 * j as f64).to_radians(); // hours 12..23
            let r = diurnal_ratio(lat, dec_deg.to_radians(), hour_angle) * 1000.0;
            worst = worst.max((r - v as f64).abs());
        }
    }
    worst
}

/// (G5) SR313 Table 1, first panel, headed "delta = +23 deg 44'": all 13 x 12 cells agree with
/// eq. (17) to the rounding (|diff| <= 0.501; the worst cell, +15 deg at 15h, is 1301.500
/// printed 1301) when the declination is 23.44 deg (decimal), and not with 23 deg 44' =
/// 23.733 deg (worst cell off by more than 1; phi = +90: 1198.9 against printed 1198). The header
/// appears to write 23.44 deg as 23 deg 44'. Reported as a possible erratum.
#[test]
fn sr313_table1_panel1_declination_23_44_decimal() {
    let decimal = table1_max_error(&TABLE1_PANEL1, 23.44);
    let sexagesimal = table1_max_error(&TABLE1_PANEL1, 23.0 + 44.0 / 60.0);
    assert!(decimal <= 0.501, "23.44 deg: worst {decimal}");
    assert!(sexagesimal > 1.0, "23 deg 44': worst {sexagesimal}");
}

/// (G5) SR313 Table 1, second panel (delta = +20 deg): every legible cell agrees with eq. (17)
/// to the rounding (|diff| <= 0.501).
#[test]
fn sr313_table1_panel2_declination_20() {
    let worst = table1_max_error(&TABLE1_PANEL2, 20.0);
    assert!(worst <= 0.501, "worst {worst}");
}

/// (G5) SR313 p. 19: the hours of minimum and maximum are 2.87 h and 14.08 h L.S.T. These are
/// the zeros of tau = -pi and tau = 0 in eq. (16) (Maxima, derivations/jacchia_bowman_diurnal.mac:
/// 2.86569 h and 14.08173 h). Located here on the model's own diurnal ratio at the equator.
#[test]
fn sr313_hours_of_extreme_temperature() {
    let ratio_at = |h: f64| diurnal_ratio(0.0, 0.0, ((h - 12.0) * 15.0).to_radians());
    let n = 240_000;
    let (mut h_min, mut h_max, mut r_min, mut r_max) = (0.0, 0.0, f64::MAX, f64::MIN);
    for k in 0..n {
        let h = 24.0 * k as f64 / n as f64;
        let r = ratio_at(h);
        if r < r_min {
            (r_min, h_min) = (r, h);
        }
        if r > r_max {
            (r_max, h_max) = (r, h);
        }
    }
    assert!((h_min - 2.87).abs() < 0.005, "minimum at {h_min} h");
    assert!((h_max - 14.08).abs() < 0.005, "maximum at {h_max} h");
}

/// SR313 Table 4a (p. 51): S of eq. (24) by height (km, S), read from a 300 dpi render.
const TABLE4_S: [(f64, f64); 20] = [
    (90.0, 0.000),
    (95.0, 0.080),
    (100.0, 0.128),
    (105.0, 0.153),
    (110.0, 0.163),
    (115.0, 0.162),
    (120.0, 0.156),
    (130.0, 0.132),
    (140.0, 0.105),
    (150.0, 0.081),
    (160.0, 0.060),
    (170.0, 0.044),
    (180.0, 0.031),
    (190.0, 0.022),
    (200.0, 0.016),
    (220.0, 0.007),
    (240.0, 0.004),
    (260.0, 0.001),
    (280.0, 0.001),
    (300.0, 0.000),
];

/// (G5) SR313 Table 4a (p. 51): S = 0.02 (z - 90) exp(-0.045 (z - 90)) agrees with every
/// printed value at the printed 3 decimals except 260 km, printed 0.001 where the formula gives
/// 0.00162 (0.002). Other entries are rounded, not truncated (240 km: 0.00352 printed 0.004),
/// so 260 km is reported as a possible misprint. (The 140 km label is faint; 0.105 fits.)
#[test]
fn sr313_table4a_amplitude() {
    for (z, printed) in TABLE4_S {
        let dz = z - SEASLAT_Z_REF;
        let s = SEASLAT_AMP * dz * (-SEASLAT_DECAY * dz).exp();
        if z == 260.0 {
            assert!(
                (s - 0.00162).abs() < 5e-5 && (s - printed).abs() > 5e-4,
                "{s}"
            );
        } else {
            assert!(
                (s - printed).abs() <= 5e-4 + 1e-12,
                "z {z}: {s} vs {printed}"
            );
        }
    }
}

/// SR313 Table 4b (p. 51): P for the Northern Hemisphere (upper sign) at 10-day steps from
/// Jan 1; "May 31" is illegible (None). The "Oct. 20" row of SR332 is Oct 28 here too.
const TABLE4_P: [Option<f64>; 37] = [
    Some(0.989),
    Some(0.948),
    Some(0.880),
    Some(0.786),
    Some(0.668),
    Some(0.531),
    Some(0.378),
    Some(0.214),
    Some(0.043),
    Some(-0.129),
    Some(-0.297),
    Some(-0.456),
    Some(-0.602),
    Some(-0.730),
    Some(-0.836),
    None,
    Some(-0.972),
    Some(-0.998),
    Some(-0.994),
    Some(-0.961),
    Some(-0.900),
    Some(-0.812),
    Some(-0.699),
    Some(-0.567),
    Some(-0.417),
    Some(-0.255),
    Some(-0.086),
    Some(0.086),
    Some(0.255),
    Some(0.417),
    Some(0.567),
    Some(0.699),
    Some(0.812),
    Some(0.900),
    Some(0.961),
    Some(0.994),
    Some(0.998),
];

/// (G5) SR313 Table 4b (p. 51) prints the J70 phase sin[(360 deg/Y)(d + 100)], Y = 365 days:
/// that formula reproduces all 36 legible values at the printed 3 decimals (checks the
/// transcription). The census prescribes the SR332 phase sin(2 pi Phi + 1.72),
/// Phi = (MJD - 36204)/365.2422 (SR332 p. 44, 46). Evaluated for 1958 it stays within 0.005 of
/// the printed J70 values: 1.72 rad is 98.55 deg against J70's 98.63 deg, and the year lengths
/// differ (365.2422 against 365).
#[test]
fn sr313_table4b_phase() {
    let z = 110.0;
    let dz = z - SEASLAT_Z_REF;
    let s = SEASLAT_AMP * dz * (-SEASLAT_DECAY * dz).exp();
    let (mut worst, mut worst_j70): (f64, f64) = (0.0, 0.0);
    for (k, p) in TABLE4_P.iter().enumerate() {
        let Some(p) = p else { continue };
        let d = 10.0 * k as f64;
        let mjd = 36204.0 + d;
        let model = seasonal_latitudinal_log10(z, std::f64::consts::FRAC_PI_2, mjd) / s;
        let j70 = (360.0 / 365.0 * (d + 100.0)).to_radians().sin();
        worst = worst.max((model - p).abs());
        worst_j70 = worst_j70.max((j70 - p).abs());
    }
    println!("Table 4b: J70 formula worst {worst_j70:.5}, SR332 phase worst {worst:.4}");
    assert!(worst_j70 <= 5e-4 + 1e-12, "{worst_j70}");
    assert!(worst < 5e-3, "{worst}");
}
