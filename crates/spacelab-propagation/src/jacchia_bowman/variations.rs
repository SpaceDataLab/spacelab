//! Solar, diurnal, semiannual, seasonal-latitudinal and high-altitude terms.
//!
//! Gate: `derivations/jacchia_bowman_hermite.mac` and `derivations/jacchia_bowman_diurnal.mac`, and the
//! `(G2)` tests in `tests/jacchia_bowman_g2_model.rs`.

use super::constants::{
    DIURNAL_BETA_DEG, DIURNAL_GAMMA_DEG, DIURNAL_M, DIURNAL_N, DIURNAL_P_DEG, DIURNAL_R, DTC_B,
    DTC_C, DTC_F_OFFSET, DTC_F_SCALE, DTC_H7_SCALE, DTC_H8_SCALE, DTC_Z_HIGH, DTC_Z_LOW, DTC_Z_MID,
    FS_WEIGHT_EXP, FS_WEIGHT_REF, FSM_WEIGHTS, FSMJ_WEIGHTS, HIGH_ALT_C, HIGH_ALT_Z_HIGH,
    HIGH_ALT_Z_LOW, HOURS_PER_DAY, MJD_1958_JAN_1, OMEGA_DAY_OFFSET, OMEGA_YEAR_DAYS, SEASLAT_AMP,
    SEASLAT_DECAY, SEASLAT_PHASE, SEASLAT_Z_REF, SEMIANNUAL_B, SEMIANNUAL_C, SEMIANNUAL_Z_SCALE_KM,
    TC_COEFFS, TROPICAL_YEAR_DAYS,
};
use std::f64::consts::{PI, TAU};

/// Solar indices in sfu. Daily values carry the lags of B08 p. 5 (applied by the caller); the
/// bars are the 81-day centred averages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarIndices {
    /// Daily F10.7, lagged 1 day.
    pub f10: f64,
    /// 81-day centred average of F10.7.
    pub f10_bar: f64,
    /// Daily S10, lagged 1 day.
    pub s10: f64,
    /// 81-day centred average of S10.
    pub s10_bar: f64,
    /// Daily M10, lagged 2 days.
    pub m10: f64,
    /// 81-day centred average of M10.
    pub m10_bar: f64,
    /// Daily Y10, lagged 5 days.
    pub y10: f64,
    /// 81-day centred average of Y10.
    pub y10_bar: f64,
}

/// FSbar = F10bar W_T + S10bar (1 - W_T), W_T = (F10bar/240)^(1/4) (B08 p. 5, eq. (1)).
/// As printed: no clamp of W_T at 1 (census Sec. 5, item 1).
pub fn fs_bar(ix: &SolarIndices) -> f64 {
    let w = (ix.f10_bar / FS_WEIGHT_REF).powf(FS_WEIGHT_EXP);
    ix.f10_bar * w + ix.s10_bar * (1.0 - w)
}

/// Nighttime minimum global exospheric temperature Tc, K (B08 p. 5, eq. (2)).
pub fn tc(ix: &SolarIndices) -> f64 {
    let c = TC_COEFFS;
    c[0] + c[1] * fs_bar(ix)
        + c[2] * (ix.f10 - ix.f10_bar)
        + c[3] * (ix.s10 - ix.s10_bar)
        + c[4] * (ix.m10 - ix.m10_bar)
        + c[5] * (ix.y10 - ix.y10_bar)
}

/// Wraps an angle into [-pi, pi).
pub fn wrap_pi(x: f64) -> f64 {
    (x + PI).rem_euclid(TAU) - PI
}

/// Ratio T_l / Tc of the diurnal variation (SR313 p. 18, eq. (17), single-bracket form, equal
/// to eq. (17) by `derivations/jacchia_bowman_diurnal.mac`). `lat` and `dec` in rad; `hour_angle` = hour
/// angle of the sun at the point (local solar time counted from upper culmination), rad.
pub fn diurnal_ratio(lat: f64, dec: f64, hour_angle: f64) -> f64 {
    let eta = 0.5 * (lat - dec).abs();
    let theta = 0.5 * (lat + dec).abs();
    let h = hour_angle;
    let tau = wrap_pi(
        h + DIURNAL_BETA_DEG.to_radians()
            + DIURNAL_P_DEG.to_radians() * (h + DIURNAL_GAMMA_DEG.to_radians()).sin(),
    );
    let c = eta.cos().powf(DIURNAL_M);
    let s = theta.sin().powf(DIURNAL_M);
    1.0 + DIURNAL_R * (s + (c - s) * (0.5 * tau).cos().powf(DIURNAL_N))
}

/// Local solar time in hours, [0, 24), from the hour angle of the sun (rad).
pub fn local_solar_time_hours(hour_angle: f64) -> f64 {
    // Noon (hour angle 0) is half a day; a full turn of hour angle is one day.
    let lst = HOURS_PER_DAY / 2.0 + wrap_pi(hour_angle) * HOURS_PER_DAY / TAU;
    lst.rem_euclid(HOURS_PER_DAY)
}

/// B06 Delta Tc, K (B06 p. 784-786, eqs. (7)-(8), Table 4), applied only inside the printed
/// bands 200 <= z < 700 km; 0 outside (no blending, task rule and census Sec. 5 item 3).
/// `f10` daily F10 (B06 p. 784 prints F10 without a bar), `lst_hours` local solar time,
/// `lat` latitude (rad), `z` height (km).
pub fn delta_tc_b06(f10: f64, lst_hours: f64, lat: f64, z: f64) -> f64 {
    let f = (f10 - DTC_F_OFFSET) / DTC_F_SCALE;
    let t = lst_hours / HOURS_PER_DAY;
    let phi = lat.cos();
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    if (DTC_Z_MID..DTC_Z_HIGH).contains(&z) {
        // eq. (7)
        let b = DTC_B;
        let h = z / DTC_H7_SCALE;
        b[0] + f * (b[1] + b[2] * t + b[3] * t2 + b[4] * t3 + b[5] * t4 + b[6] * t5)
            + phi * (b[7] * t + b[8] * t2 + b[9] * t3 + b[10] * t4 + b[11] * t5)
            + phi * h * (b[12] + b[13] * t + b[14] * t2 + b[15] * t3 + b[16] * t4 + b[17] * t5)
            + b[18] * phi
    } else if (DTC_Z_LOW..DTC_Z_MID).contains(&z) {
        // eq. (8)
        let c = DTC_C;
        let h = (z - DTC_Z_LOW) / DTC_H8_SCALE;
        h * c[0]
            + h * f * (c[1] + c[2] * t + c[3] * t2 + c[4] * t3 + c[5] * t4 + c[6] * t5)
            + h * phi
                * (c[7] * t
                    + c[8] * t2
                    + c[9] * t3
                    + c[10] * t4
                    + c[11] * t5
                    + c[12]
                    + c[13] * f
                    + c[14] * f * t
                    + c[15] * f * t2)
            + c[16]
            + phi
                * (c[17] * t + c[18] * t2 + c[19] * t3 + c[20] * f + c[21] * f * t + c[22] * f * t2)
    } else {
        0.0
    }
}

/// Semiannual Delta log10 rho = F(z) G(t) (B08 p. 7-10, eqs. (3)-(7), Tables 1-2).
/// `doy` = day of year t (1.0 at 0h UT on January 1), omega = 2 pi (t - 1)/365 (B06 p. 780;
/// B08 does not define omega). No floor or cut-off of F(z).
pub fn semiannual_log10(ix: &SolarIndices, z: f64, doy: f64) -> f64 {
    let fsmj =
        FSMJ_WEIGHTS[0] * ix.f10_bar + FSMJ_WEIGHTS[1] * ix.s10_bar + FSMJ_WEIGHTS[2] * ix.m10_bar;
    let zz = z / SEMIANNUAL_Z_SCALE_KM;
    let b = SEMIANNUAL_B;
    let fz =
        b[0] + b[1] * fsmj + b[2] * zz * fsmj + b[3] * zz * zz * fsmj + b[4] * zz * fsmj * fsmj;
    let fsm =
        FSM_WEIGHTS[0] * ix.f10_bar + FSM_WEIGHTS[1] * ix.s10_bar + FSM_WEIGHTS[2] * ix.m10_bar;
    let w = TAU * (doy - OMEGA_DAY_OFFSET) / OMEGA_YEAR_DAYS;
    let (s1, c1) = w.sin_cos();
    let (s2, c2) = (2.0 * w).sin_cos();
    let c = SEMIANNUAL_C;
    let gt = c[0]
        + c[1] * s1
        + c[2] * c1
        + c[3] * s2
        + c[4] * c2
        + fsm * (c[5] + c[6] * s1 + c[7] * c1 + c[8] * s2 + c[9] * c2);
    fz * gt
}

/// Seasonal-latitudinal Delta log10 rho of the lower thermosphere: SR313 p. 26 eq. (24)
/// amplitude S = 0.02 (z - 90) exp(-0.045 (z - 90)), phase P = sin(2 pi Phi + 1.72) with
/// Phi = (MJD - 36204)/365.2422 (SR332 p. 44, 46, eqs. (23)-(24); census Sec. 1).
pub fn seasonal_latitudinal_log10(z: f64, lat: f64, mjd: f64) -> f64 {
    let dz = z - SEASLAT_Z_REF;
    let s = SEASLAT_AMP * dz * (-SEASLAT_DECAY * dz).exp();
    let phase = (mjd - MJD_1958_JAN_1) / TROPICAL_YEAR_DAYS;
    let p = (TAU * phase + SEASLAT_PHASE).sin();
    let sin_lat = lat.sin();
    // phi/|phi| sin^2(phi) = sin(phi) |sin(phi)| for |phi| <= pi/2, and 0 at the equator.
    s * p * sin_lat * sin_lat.abs()
}

/// High-altitude density factor (B06 p. 789-790, eqs. (10)-(11), Table 5). 1 at and below
/// 1000 km; eq. (10) at and above 1500 km; in between the Hermite cubic in H = (z - 1000)/500
/// with value 1 and slope 0 at 1000 km and the value and slope of eq. (10) at 1500 km
/// (derivations/jacchia_bowman_hermite.mac; the factor 500 enters once, B06 eq. (11) prints it twice).
pub fn high_altitude_factor(f10_bar: f64, z: f64) -> f64 {
    let c = HIGH_ALT_C;
    let eq10 = |zz: f64| c[0] + c[1] * f10_bar + c[2] * zz + c[3] * zz * f10_bar;
    if z <= HIGH_ALT_Z_LOW {
        1.0
    } else if z >= HIGH_ALT_Z_HIGH {
        eq10(z)
    } else {
        let width = HIGH_ALT_Z_HIGH - HIGH_ALT_Z_LOW;
        let h = (z - HIGH_ALT_Z_LOW) / width;
        let f1500 = eq10(HIGH_ALT_Z_HIGH);
        let slope_h = width * (c[2] + c[3] * f10_bar);
        let a2 = 3.0 * f1500 - slope_h - 3.0;
        let a3 = slope_h - 2.0 * f1500 + 2.0;
        1.0 + h * h * (a2 + a3 * h)
    }
}
