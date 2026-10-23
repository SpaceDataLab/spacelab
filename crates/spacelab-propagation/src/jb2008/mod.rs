//! JB2008 (Jacchia–Bowman 2008) empirical thermospheric density and temperature model.
//!
//! A faithful, numerically exact port of Bruce R. Bowman's canonical FORTRAN `JB2008.for`
//! (Space Environment Technologies; the model's source of record — sol.spacenvironment.net/JB2008).
//! JB2008 is a CIRA "integration form" Jacchia model: there are no tabulated densities, the
//! barometric and diffusion equations are integrated numerically (Newton–Cotes 5-point) from 90 km
//! to the input altitude. On top of the Jacchia-1971 temperature profile it layers the JB2008
//! empirical corrections — the `Tc` exospheric-temperature fit driven by the new solar indices
//! (F10/S10/M10/Y10), the Dst-based geomagnetic-storm term `DSTDTC`, the local-solar-time/latitude
//! `dTc` correction, the J70 seasonal-latitudinal variation and the new semiannual density model.
//!
//! The public surface is the pure [`jb2008`] function (explicit-angle interface identical to the
//! FORTRAN subroutine `JB2008(AMJD, SUN, SAT, …)`) plus [`Jb2008Input`] (the caller-owned
//! space-weather drivers) and [`Jb2008Output`]. The coefficient tables live in `tables` and index
//! 1:1 with the FORTRAN `DATA` statements. [`Jb2008Atmosphere`] wraps the model as a
//! [`crate::forces::Atmosphere`] for the drag path.
//!
//! The mathematical constants π, π/2, 2π and ln(10) are taken at full precision (`core::f64::consts`),
//! not from the FORTRAN's 7–10 digit `DATA` values — matching Orekit and the SatelliteToolbox Julia
//! port. Validated against the canonical FORTRAN's own output across 90–2300 km (see the tests).

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::suboptimal_flops,
    // The FORTRAN `IDINT` step counts truncate toward zero by design, and their arguments are
    // small in-range positives; the band comparisons and altitude gates are written as explicit
    // chains to mirror the source `IF (…GE….AND….LE.…)` structure for auditability.
    clippy::cast_possible_truncation,
    clippy::manual_range_contains
)]

mod atmosphere;
mod tables;

pub use atmosphere::Jb2008Atmosphere;
use core::f64::consts::{FRAC_PI_2, LN_10, PI, TAU};
use tables::{ALPHA, AMW, BDT, CDT, CHT, CXAMB, FRAC, FZM, GTM, WT};

/// Universal gas constant in mks units (joules / K / kmol) — FORTRAN `RSTAR`.
const RSTAR: f64 = 8314.32;
/// Avogadro's number in mks units (molecules / kmol) — FORTRAN `AVOGAD`.
const AVOGAD: f64 = 6.02257e26;
/// Mean sea-level molecular weight used in the number-density split — FORTRAN `28.960`.
const MB0: f64 = 28.960;
/// Reference density at 90 km (kg/m³) — FORTRAN `3.46D-6`.
const RHO_Z1: f64 = 3.46e-6;
/// Newton–Cotes step-size factors for the 90–105, 105–500 and >500 km regimes.
const R1: f64 = 0.010;
const R2: f64 = 0.025;
const R3: f64 = 0.075;

/// Caller-owned JB2008 space-weather drivers for a single evaluation.
///
/// Every field is a solar/geomagnetic index the model consumes verbatim; the port does **not** load
/// data files (SOLFSMY.TXT / DTCFILE.TXT) — the caller supplies the values, exactly as the FORTRAN
/// subroutine's argument list does. The tabular-time offsets in the field docs are the delays at
/// which each index is sampled relative to the evaluation epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jb2008Input {
    /// 10.7-cm solar flux, 1e-22 W/(m²·Hz), sampled 1 day earlier.
    pub f10: f64,
    /// 10.7-cm flux, 81-day centered average, sampled 1 day earlier.
    pub f10b: f64,
    /// EUV index (26–34 nm) scaled to F10, sampled 1 day earlier.
    pub s10: f64,
    /// EUV 81-day centered average index, sampled 1 day earlier.
    pub s10b: f64,
    /// MG-II index scaled to F10, sampled 2 days earlier.
    pub xm10: f64,
    /// MG-II 81-day centered average index, sampled 2 days earlier.
    pub xm10b: f64,
    /// Solar X-ray + Lyman-α index scaled to F10, sampled 5 days earlier.
    pub y10: f64,
    /// Solar X-ray + Lyman-α 81-day centered average index, sampled 5 days earlier.
    pub y10b: f64,
    /// Exospheric temperature change from the Dst geomagnetic index (K) — `DSTDTC`.
    pub dstdtc: f64,
}

/// JB2008 outputs at the evaluation point.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Jb2008Output {
    /// Total mass density (kg/m³) — FORTRAN `RHO`.
    pub rho: f64,
    /// Exospheric temperature above the point (K) — FORTRAN `TEMP(1)` = Tℓ + `DSTDTC`.
    pub t_exo: f64,
    /// Temperature at the point (K) — FORTRAN `TEMP(2)`.
    pub t_alt: f64,
}

/// Mean molecular mass at altitude `z` (km), 90–105 km regime — FORTRAN `XAMBAR`, Eq. (1).
fn xambar(z: f64) -> f64 {
    let dz = z - 100.0;
    let mut amb = CXAMB[6];
    let mut i = 5;
    loop {
        amb = dz * amb + CXAMB[i];
        if i == 0 {
            break;
        }
        i -= 1;
    }
    amb
}

/// Gravity (m/s²) at altitude `z` (km) — FORTRAN `XGRAV`, Eq. (8). Earth radius 6356.766 km.
fn xgrav(z: f64) -> f64 {
    let t = 1.0 + z / 6356.766;
    9.80665 / (t * t)
}

/// Local temperature (K) at altitude `z` (km) — FORTRAN `XLOCAL`, Eq. (10)/(13).
fn xlocal(z: f64, tc: &[f64; 4]) -> f64 {
    let dz = z - 125.0;
    if dz <= 0.0 {
        ((-9.8204695e-6 * dz - 7.3039742e-4) * dz * dz + 1.0) * dz * tc[1] + tc[0]
    } else {
        tc[0] + tc[2] * libm::atan(tc[3] * dz * (1.0 + 4.5e-6 * libm::pow(dz, 2.5)))
    }
}

/// Day-of-year (+ fraction) from days-since-1950 — FORTRAN `TMOUTD`.
fn day_of_year(d1950: f64) -> f64 {
    let mut iyday = d1950 as i32;
    let frac = d1950 - f64::from(iyday);
    iyday += 364;
    let mut itemp = iyday / 1461;
    iyday -= itemp * 1461;
    itemp = iyday / 365;
    if itemp >= 3 {
        itemp = 3;
    }
    iyday = iyday - 365 * itemp + 1;
    f64::from(iyday) + frac
}

/// Semiannual variation (Δ log₁₀ ρ) — FORTRAN `SEMIAN08`. Returns `(Fz, Gt, Δlog₁₀ρ)`; `Fz` is
/// clamped to ≥ 1e-6 as in the source.
fn semian08(day: f64, ht_km: f64, f10b: f64, s10b: f64, xm10b: f64) -> (f64, f64, f64) {
    // 81-day centered index for the amplitude Fz.
    let mut fsmb = 1.00 * f10b - 0.70 * s10b - 0.04 * xm10b;
    let htz = ht_km / 1000.0;
    let mut fzz = FZM[0]
        + FZM[1] * fsmb
        + FZM[2] * fsmb * htz
        + FZM[3] * fsmb * (htz * htz)
        + FZM[4] * (fsmb * fsmb) * htz;

    // 81-day centered index for the phase function Gt.
    fsmb = 1.00 * f10b - 0.75 * s10b - 0.37 * xm10b;
    let tau = (day - 1.0) / 365.0;
    let sin1p = libm::sin(TAU * tau);
    let cos1p = libm::cos(TAU * tau);
    let sin2p = libm::sin(2.0 * TAU * tau);
    let cos2p = libm::cos(2.0 * TAU * tau);
    let gtz = GTM[0]
        + GTM[1] * sin1p
        + GTM[2] * cos1p
        + GTM[3] * sin2p
        + GTM[4] * cos2p
        + GTM[5] * fsmb
        + GTM[6] * fsmb * sin1p
        + GTM[7] * fsmb * cos1p
        + GTM[8] * fsmb * sin2p
        + GTM[9] * fsmb * cos2p;

    if fzz < 1.0e-6 {
        fzz = 1.0e-6;
    }
    (fzz, gtz, fzz * gtz)
}

/// Local-solar-time / latitude correction to `Tc` (K) — FORTRAN `DTSUB`.
///
/// `xlst` is local solar time in hours (0–24), `xlat` geocentric latitude (rad), `zht` altitude (km).
/// The five altitude bands (120–200, 200–240, 240–300, 300–600, 600–800 km) blend smoothly; the model
/// contributes no dTc outside 120–800 km.
fn dtsub(f10: f64, xlst: f64, xlat: f64, zht: f64) -> f64 {
    let tx = xlst / 24.0;
    let tx2 = tx * tx;
    let tx3 = tx2 * tx;
    let tx4 = tx3 * tx;
    let tx5 = tx4 * tx;
    let ycs = libm::cos(xlat);
    let f = (f10 - 100.0) / 100.0;

    // DTC200 (the C(17..23) block) and its altitude derivative DTC200DZ (the full C block). Note the
    // FORTRAN uses B(2) for the second term of the derivative block; BDT[1] == CDT[1] there.
    let p_c200 = CDT[16]
        + CDT[17] * tx * ycs
        + CDT[18] * tx2 * ycs
        + CDT[19] * tx3 * ycs
        + CDT[20] * f * ycs
        + CDT[21] * tx * f * ycs
        + CDT[22] * tx2 * f * ycs;
    let p_cfull = CDT[0]
        + BDT[1] * f
        + CDT[2] * tx * f
        + CDT[3] * tx2 * f
        + CDT[4] * tx3 * f
        + CDT[5] * tx4 * f
        + CDT[6] * tx5 * f
        + CDT[7] * tx * ycs
        + CDT[8] * tx2 * ycs
        + CDT[9] * tx3 * ycs
        + CDT[10] * tx4 * ycs
        + CDT[11] * tx5 * ycs
        + CDT[12] * ycs
        + CDT[13] * f * ycs
        + CDT[14] * tx * f * ycs
        + CDT[15] * tx2 * f * ycs;

    // The B(1..19) block evaluated at a scaled height `h`, and its height derivative (the B(13..18)
    // ycs terms).
    let p_bfull = |h: f64| -> f64 {
        BDT[0]
            + BDT[1] * f
            + BDT[2] * tx * f
            + BDT[3] * tx2 * f
            + BDT[4] * tx3 * f
            + BDT[5] * tx4 * f
            + BDT[6] * tx5 * f
            + BDT[7] * tx * ycs
            + BDT[8] * tx2 * ycs
            + BDT[9] * tx3 * ycs
            + BDT[10] * tx4 * ycs
            + BDT[11] * tx5 * ycs
            + BDT[12] * h * ycs
            + BDT[13] * tx * h * ycs
            + BDT[14] * tx2 * h * ycs
            + BDT[15] * tx3 * h * ycs
            + BDT[16] * tx4 * h * ycs
            + BDT[17] * tx5 * h * ycs
            + BDT[18] * ycs
    };
    let p_bdz = BDT[12] * ycs
        + BDT[13] * tx * ycs
        + BDT[14] * tx2 * ycs
        + BDT[15] * tx3 * ycs
        + BDT[16] * tx4 * ycs
        + BDT[17] * tx5 * ycs;

    if zht >= 120.0 && zht <= 200.0 {
        let dtc200 = p_c200;
        let dtc200dz = p_cfull;
        let cc = 3.0 * dtc200 - dtc200dz;
        let dd = dtc200 - cc;
        let zp = (zht - 120.0) / 80.0;
        cc * zp * zp + dd * zp * zp * zp
    } else if zht > 200.0 && zht <= 240.0 {
        let h = (zht - 200.0) / 50.0;
        h * p_cfull + p_c200
    } else if zht > 240.0 && zht <= 300.0 {
        let aa = (40.0 / 50.0) * p_cfull + p_c200;
        let bb = p_cfull;
        let dtc300 = p_bfull(300.0 / 100.0);
        let dtc300dz = p_bdz;
        let cc = 3.0 * dtc300 - dtc300dz - 3.0 * aa - 2.0 * bb;
        let dd = dtc300 - aa - bb - cc;
        let zp = (zht - 240.0) / 60.0;
        aa + bb * zp + cc * zp * zp + dd * zp * zp * zp
    } else if zht > 300.0 && zht <= 600.0 {
        p_bfull(zht / 100.0)
    } else if zht > 600.0 && zht <= 800.0 {
        let aa = p_bfull(600.0 / 100.0);
        let bb = p_bdz;
        let cc = -(3.0 * aa + 4.0 * bb) / 4.0;
        let dd = (aa + bb) / 4.0;
        let zp = (zht - 600.0) / 100.0;
        aa + bb * zp + cc * zp * zp + dd * zp * zp * zp
    } else {
        0.0
    }
}

/// Number of Newton–Cotes outer steps for a log-height span: FORTRAN `IDINT(AL/R) + 1`.
fn n_steps(al: f64, r: f64) -> i32 {
    (al / r) as i32 + 1
}

/// Total mass density and temperatures at a point — the canonical explicit-angle JB2008 subroutine.
///
/// Mirrors `SUBROUTINE JB2008(AMJD, SUN, SAT, F10, …, TEMP, RHO)` argument-for-argument:
///
/// * `amjd` — date/time in Modified Julian Days and fraction (`MJD = JD − 2400000.5`).
/// * `sun_ra`, `sun_dec` — right ascension and declination of the Sun (rad).
/// * `sat_ra` — right ascension of the position (rad).
/// * `sat_lat` — geocentric latitude of the position (rad).
/// * `sat_alt_km` — height of the position (km). The barometric integration starts at 90 km, so a
///   height below that — or a non-finite one — is off the model's domain and **every** output field
///   is `NaN`.
/// * `input` — the space-weather drivers.
///
/// Only the hour angle `sat_ra − sun_ra` and `sun_dec` enter, so any consistent frame for the two
/// right ascensions is admissible. Returns density (kg/m³) and the two temperatures.
pub fn jb2008(
    amjd: f64,
    sun_ra: f64,
    sun_dec: f64,
    sat_ra: f64,
    sat_lat: f64,
    sat_alt_km: f64,
    input: &Jb2008Input,
) -> Jb2008Output {
    // Off-domain heights used to leave through three different doors. Below ~89.1045 km the
    // barometric integral takes the density to `NaN` while `t_alt` still reads a plausible 0 K;
    // between there and the documented 90 km floor the result is finite but outside the range the
    // model claims; and an infinite height saturates `(al / r) as i32` at `i32::MAX`, whose `+ 1`
    // panics under debug assertions and wraps to a negative step count without them — a panic that
    // crosses the PyO3 boundary as a process abort. One door now, and it is `NaN` in every field:
    // a `NaN` density beside a plausible temperature is the combination that misleads.
    if !sat_alt_km.is_finite() || sat_alt_km < 90.0 {
        return Jb2008Output {
            rho: f64::NAN,
            t_exo: f64::NAN,
            t_alt: f64::NAN,
        };
    }

    // Eq. (14): nighttime minimum global exospheric temperature Tc.
    let mut fnw = libm::pow(input.f10b / 240.0, 0.25);
    if fnw > 1.0 {
        fnw = 1.0;
    }
    let fsb = input.f10b * fnw + input.s10b * (1.0 - fnw);
    let tsubc = 392.4
        + 3.227 * fsb
        + 0.298 * (input.f10 - input.f10b)
        + 2.259 * (input.s10 - input.s10b)
        + 0.312 * (input.xm10 - input.xm10b)
        + 0.178 * (input.y10 - input.y10b);

    // Eq. (15)–(16).
    let eta = 0.5 * libm::fabs(sat_lat - sun_dec);
    let theta = 0.5 * libm::fabs(sat_lat + sun_dec);
    let h = sat_ra - sun_ra;
    let tau = h - 0.64577182 + 0.10471976 * libm::sin(h + 0.75049158);

    // Local solar time (hours) from the hour angle: (H + π)·(12/π), wrapped to [0, 24).
    //
    // The FORTRAN wraps with one conditional subtract and one conditional add, which is total only
    // for `H ∈ [−3π, 3π)` — outside that band it returns a local solar time still off by a whole
    // multiple of 24 h, and the caller's contract above places no such bound on the two right
    // ascensions. `rem_euclid` is the same map made total. It agrees with the two-step form
    // bit-for-bit at every one of 4 000 001 hour angles sampled across [−3π, 3π]; the single
    // exception is the endpoint `H = 3π`, where the two-step form returns exactly 24.0 — outside
    // its own stated range.
    let glsthr = ((h + PI) * (12.0 / PI)).rem_euclid(24.0);

    // Eq. (17): exospheric temperature under low geomagnetic activity.
    let c = libm::pow(libm::cos(eta), 2.5);
    let s = libm::pow(libm::sin(theta), 2.5);
    let cos_tau = libm::fabs(libm::cos(0.5 * tau));
    let df = s + (c - s) * cos_tau * cos_tau * cos_tau;
    let tsubl = tsubc * (1.0 + 0.31 * df);

    // Local-solar-time/latitude dTc, then the local exospheric temperature with the storm term.
    let dtclst = dtsub(input.f10, glsthr, sat_lat, sat_alt_km);
    let t_exo = tsubl + input.dstdtc;
    let tinf = tsubl + input.dstdtc + dtclst;

    // Eq. (9)/(11) and the Tc profile array.
    let tsubx = 444.3807 + 0.02385 * tinf - 392.8292 * libm::exp(-0.0021357 * tinf);
    let gsubx = 0.054285714 * (tsubx - 183.0);
    let tc3 = (tinf - tsubx) / FRAC_PI_2;
    let tc = [tsubx, gsubx, tc3, gsubx / tc3];

    // Eq. (5): integrate M̄·g/T from 90 km to min(alt, 105) by Newton–Cotes 5-point.
    let z1 = 90.0;
    let z2 = libm::fmin(sat_alt_km, 105.0);
    let mut al = libm::log(z2 / z1);
    let mut n = n_steps(al, R1);
    let mut zr = libm::exp(al / f64::from(n));
    let ambar1 = xambar(z1);
    let tloc1 = xlocal(z1, &tc);
    let mut zend = z1;
    let mut sum2 = 0.0;
    let mut ain = ambar1 * xgrav(z1) / tloc1;
    let mut ambar2 = 0.0;
    let mut tloc2 = 0.0;
    let mut z = 0.0;
    let mut gravl = 0.0;
    for _ in 0..n {
        z = zend;
        zend = zr * z;
        let dz = 0.25 * (zend - z);
        let mut sum1 = WT[0] * ain;
        for &wt in WT.iter().skip(1) {
            z += dz;
            ambar2 = xambar(z);
            tloc2 = xlocal(z, &tc);
            gravl = xgrav(z);
            ain = ambar2 * gravl / tloc2;
            sum1 += wt * ain;
        }
        sum2 += dz * sum1;
    }
    let fact1 = 1000.0 / RSTAR;
    let mut rho = RHO_Z1 * ambar2 * tloc1 * libm::exp(-fact1 * sum2) / ambar1 / tloc2;

    // Eq. (2)–(4): number-density logs of N2, O2, O, Ar, He from the total.
    let anm = AVOGAD * rho;
    let an = anm / ambar2;
    let fact_num = anm / MB0;
    let mut aln = [0.0f64; 6];
    aln[0] = libm::log(FRAC[0] * fact_num);
    aln[3] = libm::log(FRAC[2] * fact_num);
    aln[4] = libm::log(FRAC[3] * fact_num);
    aln[1] = libm::log(fact_num * (1.0 + FRAC[1]) - an);
    aln[2] = libm::log(2.0 * (an - fact_num));

    let t_alt;
    if sat_alt_km <= 105.0 {
        t_alt = tloc2;
        aln[5] = aln[4] - 25.0; // negligible hydrogen below 105 km
    } else {
        // Eq. (6): integrate g/T from 105 km to min(alt, 500).
        let z3 = libm::fmin(sat_alt_km, 500.0);
        al = libm::log(z3 / z);
        n = n_steps(al, R2);
        zr = libm::exp(al / f64::from(n));
        sum2 = 0.0;
        ain = gravl / tloc2;
        let mut tloc3 = 0.0;
        for _ in 0..n {
            z = zend;
            zend = zr * z;
            let dz = 0.25 * (zend - z);
            let mut sum1 = WT[0] * ain;
            for &wt in WT.iter().skip(1) {
                z += dz;
                tloc3 = xlocal(z, &tc);
                gravl = xgrav(z);
                ain = gravl / tloc3;
                sum1 += wt * ain;
            }
            sum2 += dz * sum1;
        }

        // Continue from min(alt, 500) to max(alt, 500) (up to 500 km for the hydrogen anchor).
        let z4 = libm::fmax(sat_alt_km, 500.0);
        al = libm::log(z4 / z);
        let r = if sat_alt_km > 500.0 { R3 } else { R2 };
        n = n_steps(al, r);
        zr = libm::exp(al / f64::from(n));
        let mut sum3 = 0.0;
        let mut tloc4 = 0.0;
        for _ in 0..n {
            z = zend;
            zend = zr * z;
            let dz = 0.25 * (zend - z);
            let mut sum1 = WT[0] * ain;
            for &wt in WT.iter().skip(1) {
                z += dz;
                tloc4 = xlocal(z, &tc);
                gravl = xgrav(z);
                ain = gravl / tloc4;
                sum1 += wt * ain;
            }
            sum3 += dz * sum1;
        }

        let altr;
        let fact2;
        let hsign;
        if sat_alt_km <= 500.0 {
            t_alt = tloc3;
            altr = libm::log(tloc3 / tloc2);
            fact2 = fact1 * sum2;
            hsign = 1.0;
        } else {
            t_alt = tloc4;
            altr = libm::log(tloc4 / tloc2);
            fact2 = fact1 * (sum2 + sum3);
            hsign = -1.0;
        }
        for i in 0..5 {
            aln[i] = aln[i] - (1.0 + ALPHA[i]) * altr - fact2 * AMW[i];
        }

        // Eq. (7): hydrogen number density anchored at 500 km (CIRA72 log10 fit, → 1/m³).
        let al10t5 = libm::log10(tinf);
        let alnh5 = (5.5 * al10t5 - 39.40) * al10t5 + 73.13;
        aln[5] = LN_10 * (alnh5 + 6.0) + hsign * (libm::log(tloc4 / tloc3) + fact1 * sum3 * AMW[5]);
    }

    // Eq. (24): J70 seasonal-latitudinal variation (Δ log₁₀ ρ).
    let trash = (amjd - 36204.0) / 365.2422;
    let capphi = trash - libm::trunc(trash);
    let sgn = if sat_lat >= 0.0 { 1.0 } else { -1.0 };
    let sin_lat = libm::sin(sat_lat);
    let hm90 = sat_alt_km - 90.0;
    let dlrsl = 0.02
        * hm90
        * libm::exp(-0.045 * hm90)
        * sgn
        * libm::sin(TAU * capphi + 1.72)
        * (sin_lat * sin_lat);

    // Eq. (23): new semiannual variation (Δ log₁₀ ρ), only where the integration stayed below 2000 km.
    let mut dlrsa = 0.0;
    if z < 2000.0 {
        let d1950 = amjd - 33281.0;
        let (fzz, _gtz, drlog) = semian08(
            day_of_year(d1950),
            sat_alt_km,
            input.f10b,
            input.s10b,
            input.xm10b,
        );
        dlrsa = drlog;
        if fzz < 0.0 {
            dlrsa = 0.0;
        }
    }

    // Apply the summed δ-log-ρ to every species (natural-log form: ln10 · Δlog₁₀ρ).
    let dlr = LN_10 * (dlrsl + dlrsa);
    let mut sumnm = 0.0;
    for i in 0..6 {
        aln[i] += dlr;
        sumnm += libm::exp(aln[i]) * AMW[i];
    }
    rho = sumnm / AVOGAD;

    // High-altitude (>1000 km) exospheric density correction factor.
    rho *= density_correction(sat_alt_km, input.f10b);

    Jb2008Output { rho, t_exo, t_alt }
}

/// High-altitude exospheric density correction factor `FEX` — FORTRAN `JB2008` tail.
///
/// Unity below 1000 km; a cubic blend over 1000–1500 km; and the linear high-altitude fit above
/// 1500 km. `alt_km` in km, `f10b` the 81-day centered F10.
fn density_correction(alt_km: f64, f10b: f64) -> f64 {
    if alt_km >= 1000.0 && alt_km < 1500.0 {
        let zeta = (alt_km - 1000.0) * 0.002;
        let zeta2 = zeta * zeta;
        let zeta3 = zeta * zeta2;
        let f15c = CHT[0] + CHT[1] * f10b + CHT[2] * 1500.0 + CHT[3] * f10b * 1500.0;
        let f15c_zeta = (CHT[2] + CHT[3] * f10b) * 500.0;
        let fex2 = 3.0 * f15c - f15c_zeta - 3.0;
        let fex3 = f15c_zeta - 2.0 * f15c + 2.0;
        1.0 + fex2 * zeta2 + fex3 * zeta3
    } else if alt_km >= 1500.0 {
        CHT[0] + CHT[1] * f10b + CHT[2] * alt_km + CHT[3] * f10b * alt_km
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forces::Atmosphere;
    use spacelab_autodiff::Dual;
    use spacelab_testkit::nan_max;

    /// `day_of_year` is a quadrennial-block trick, not a calendar: it folds the days-since-1950 count
    /// into 1461-day blocks, divides by 365 with the fourth year clamped, and adds one. Every golden
    /// case in this file shares one epoch, so the fold was never exercised — the clamp, the divide and
    /// the subtraction could each be replaced and no assertion moved. The oracle is CPython's
    /// proleptic-Gregorian calendar (`date(1949,12,31) + timedelta(days=d1950)`, `.tm_yday`), which
    /// shares no code with the folding, and the cases cover all four positions in the block, both
    /// sides of a leap day, and both sides of a year boundary.
    #[test]
    fn the_day_of_year_folds_the_1950_day_count_back_onto_the_calendar() {
        // (days since 1950, day of year) — block position in the comment.
        const CASES: [(f64, i32); 24] = [
            (1.0, 1),       // 1950-01-01, block-year 1
            (2.0, 2),       // 1950-01-02, block-year 1
            (365.0, 365),   // 1950-12-31, block-year 1
            (366.0, 1),     // 1951-01-01, block-year 2
            (367.0, 2),     // 1951-01-02, block-year 2
            (730.0, 365),   // 1951-12-31, block-year 2
            (731.0, 1),     // 1952-01-01, block-year 3 (leap)
            (760.0, 30),    // 1952-01-30, block-year 3
            (790.0, 60),    // 1952-02-29, block-year 3 — the leap day itself
            (791.0, 61),    // 1952-03-01, block-year 3
            (1095.0, 365),  // 1952-12-30, block-year 3
            (1096.0, 366),  // 1952-12-31, block-year 3 — the 366th day
            (1097.0, 1),    // 1953-01-01, block-year 0 — the block rolls over
            (1126.0, 30),   // 1953-01-30, block-year 0
            (1460.0, 364),  // 1953-12-30, block-year 0
            (1461.0, 365),  // 1953-12-31, block-year 0
            (1462.0, 1),    // 1954-01-01, block-year 1
            (18262.0, 365), // 1999-12-31, block-year 2
            (18321.0, 59),  // 2000-02-28, block-year 3
            (18322.0, 60),  // 2000-02-29, block-year 3
            (27028.0, 365), // 2023-12-31, block-year 2
            (27059.0, 31),  // 2024-01-31, block-year 3
            (27060.0, 32),  // 2024-02-01, block-year 3
            (27394.0, 366), // 2024-12-31, block-year 3
        ];
        for (d1950, want) in CASES {
            let got = day_of_year(d1950);
            assert!(
                (got - f64::from(want)).abs() < 1e-12,
                "d1950 = {d1950}: day-of-year {got} vs calendar {want}"
            );
        }
        // The fractional part of the day survives the integer folding untouched.
        for frac in [0.0_f64, 0.25, 0.5, 0.999] {
            let got = day_of_year(790.0 + frac);
            assert!(
                (got - (60.0 + frac)).abs() < 1e-12,
                "fraction {frac} lost: {got}"
            );
        }
    }

    /// Space-weather drivers shared by every golden case (the JB2008 authors' `InputParams[0]`
    /// values, also used in Orekit's `JB2008Test`).
    const DRIVERS: Jb2008Input = Jb2008Input {
        f10: 91.00,
        f10b: 137.10,
        s10: 108.80,
        s10b: 123.80,
        xm10: 116.70,
        xm10b: 128.50,
        y10: 168.00,
        y10b: 138.60,
        dstdtc: 43.0,
    };

    // Common epoch/angles for the golden sweep.
    const AMJD: f64 = 52951.003805740744;
    const SUN_RA: f64 = 3.046653643566772;
    const SUN_DEC: f64 = -0.285987757544287;
    const SAT_RA: f64 = 1.28211886851503;

    /// Reference output of the canonical FORTRAN `JB2008.for` (Bowman / Space Environment
    /// Technologies), produced by `gfortran --std=legacy JB2008.for driver.f90` calling the
    /// `JB2008` subroutine directly with these explicit-angle inputs. Columns:
    /// `(sat_lat_rad, altitude_km, T_exo_K, T_alt_K, rho_kg_m3)`. Case 1 (91 km) reproduces
    /// Orekit's own reference density 0.27945654e-05. The sweep exercises every density regime
    /// (≤105, 105–500, >500 km) and every `dTc` altitude band (120–200, …, 600–800 km).
    #[rustfmt::skip]
    const GOLDEN: [(f64, f64, f64, f64, f64); 15] = [
        (-1.4877186543999,   91.0, 9.55228511955559156e+02, 1.83065114058510488e+02, 2.79456652628281775e-06),
        (-1.20,             100.0, 9.20479122509624858e+02, 1.95362428657865166e+02, 4.56795943859597690e-07),
        (-0.90,             110.0, 8.87818096825455996e+02, 2.48404651734700849e+02, 8.21485136788679065e-08),
        (-0.60,             130.0, 8.61835706449436884e+02, 4.61517954279987123e+02, 7.55434050198758302e-09),
        (-0.30,             150.0, 8.44323385491735735e+02, 6.31412583172617929e+02, 1.96430628926120325e-09),
        ( 0.00,             180.0, 8.35391967116129535e+02, 7.39094227713098917e+02, 5.03853287990276035e-10),
        ( 0.30,             220.0, 8.32995701436382660e+02, 7.94957075790404247e+02, 1.24183853891096289e-10),
        ( 0.60,             270.0, 8.34343918966640217e+02, 8.21860974331738134e+02, 2.95163058731025377e-11),
        ( 0.90,             400.0, 8.42733051905366551e+02, 8.39640296255508247e+02, 1.54849289094396471e-12),
        ( 1.10,             550.0, 8.53293164498746251e+02, 8.47374074487552662e+02, 9.54180307090993054e-14),
        ( 1.20,             700.0, 8.60112705548120175e+02, 8.56211710903569269e+02, 1.22949956656878185e-14),
        ( 1.30,             900.0, 8.67925502193465604e+02, 8.67801957069549644e+02, 3.11369410538303611e-15),
        (-1.00,            1200.0, 8.98092988748583821e+02, 8.98049804517595248e+02, 1.06176955148455672e-15),
        ( 0.50,            1700.0, 8.33285646067644620e+02, 8.33276243908899005e+02, 2.13303298609386956e-16),
        (-0.50,            2300.0, 8.55013801663028858e+02, 8.55010558419578274e+02, 8.08731045471757742e-17),
    ];

    /// Orekit 13.1.7 `getDensity(...)` densities for the same 15 explicit-angle cases (independent
    /// Java lineage, same full-precision math constants as this port). The `matches_fortran_reference`
    /// test shows this port reproduces these to ~3e-15 relative — i.e. bit-close to a second
    /// implementation — while both differ from the FORTRAN by the same 8.2e-8, which is the FORTRAN's
    /// truncated π/π/2/ln(10) `DATA` values, not a porting error.
    #[rustfmt::skip]
    const OREKIT_RHO: [f64; 15] = [
        2.79456652601987750e-06, 4.56795943608897540e-07, 8.21485136382563300e-08,
        7.55434050084640500e-09, 1.96430629227979600e-09, 5.03853287600763200e-10,
        1.24183852893192240e-10, 2.95163053427021200e-11, 1.54849282760591280e-12,
        9.54180247623318900e-14, 1.22949949298983380e-14, 3.11369396841466530e-15,
        1.06176950369512470e-15, 2.13303285008241880e-16, 8.08730978905969400e-17,
    ];

    /// The pure model matches the canonical FORTRAN across 90–2300 km (density < 1e-6, temperature
    /// < 1e-8 relative), and matches Orekit — which shares this port's full-precision constants — to
    /// a few ULP (< 1e-12). The FORTRAN residual is that model's own 7–10 digit π/π/2/ln(10) `DATA`
    /// values, confirmed by Orekit differing from the FORTRAN by the identical amount.
    #[test]
    fn matches_fortran_reference() {
        let mut max_rho = 0.0_f64;
        let mut max_temp = 0.0_f64;
        let mut max_orekit = 0.0_f64;
        for (&(lat, alt, t_exo_ref, t_alt_ref, rho_ref), &orekit) in GOLDEN.iter().zip(&OREKIT_RHO)
        {
            let out = jb2008(AMJD, SUN_RA, SUN_DEC, SAT_RA, lat, alt, &DRIVERS);
            max_rho = nan_max(max_rho, ((out.rho - rho_ref) / rho_ref).abs());
            max_temp = nan_max(max_temp, ((out.t_exo - t_exo_ref) / t_exo_ref).abs());
            max_temp = nan_max(max_temp, ((out.t_alt - t_alt_ref) / t_alt_ref).abs());
            max_orekit = nan_max(max_orekit, ((out.rho - orekit) / orekit).abs());
        }
        assert!(
            max_rho < 1.0e-6,
            "max density rel err vs FORTRAN {max_rho:e} exceeds 1e-6"
        );
        assert!(
            max_temp < 1.0e-8,
            "max temperature rel err vs FORTRAN {max_temp:e} exceeds 1e-8"
        );
        assert!(
            max_orekit < 1.0e-12,
            "max density rel err vs Orekit {max_orekit:e} exceeds 1e-12"
        );
    }

    /// Local solar time is a function of the hour angle *modulo a full turn*, so the model must be
    /// too. Every golden case shares one epoch and one pair of right ascensions with `|H| < π`, which
    /// is the one regime where the wrap does nothing at all — so nothing here saw the wrap, and the
    /// documented contract that "any consistent frame for the two right ascensions is admissible"
    /// went unmeasured. It was also false: the FORTRAN's conditional-subtract/conditional-add pair
    /// is total only on `H ∈ [−3π, 3π)`, and at `H = −4 − 2π` it left the density 2.5 % wrong, at
    /// `H = −4 − 4π` 97 % wrong.
    ///
    /// 400 km is chosen because `dtsub` — the only consumer of the local solar time — contributes
    /// nothing outside 120–800 km, so a lower or higher sample could not see the wrap either.
    #[test]
    fn the_density_depends_on_the_hour_angle_only_through_a_full_turn() {
        let (lat, alt) = (0.3, 400.0);
        let rho = |sat_ra: f64| jb2008(AMJD, SUN_RA, SUN_DEC, sat_ra, lat, alt, &DRIVERS).rho;

        for &h in &[-4.0_f64, -PI, -1.0, 0.0, 1.0, PI, 4.0] {
            let base = rho(SUN_RA + h);
            for k in -3..=3 {
                let turned = rho(SUN_RA + h + f64::from(k) * TAU);
                assert!(
                    ((turned - base) / base).abs() < 1.0e-12,
                    "H = {h} turned by {k}: {turned:e} vs {base:e}"
                );
            }
        }

        // The wrapped interval is half-open at 0, not at 24: `H = −π` maps to local midnight, which
        // is the *right* limit. The model is genuinely discontinuous there — `dtsub` is a plain
        // quintic in `xlst/24`, not a periodic one — so which side the boundary belongs to is
        // observable, and it is the only thing that distinguishes 0.0 h from 24.0 h.
        let at = rho(SUN_RA - PI);
        let right = rho(SUN_RA - PI + 1.0e-9);
        let left = rho(SUN_RA - PI - 1.0e-9);
        assert!(
            ((at - right) / at).abs() < 1.0e-8,
            "H = −π should agree with the right limit: {at:e} vs {right:e}"
        );
        assert!(
            ((at - left) / at).abs() > 1.0e-3,
            "H = −π should differ from the left limit: {at:e} vs {left:e}"
        );
    }

    /// The height domain was prose, and off it the model answered in three different ways: at 89 km
    /// a `NaN` density beside `t_alt = 0 K` and a perfectly plausible `t_exo = 832.99 K`; between
    /// 89.1045 km — the real floor of the barometric integral, not the 90 km the doc comment claimed
    /// — and 90 km a finite answer the model does not stand behind; and at an infinite height a
    /// panic, because `(al / r) as i32` saturates at `i32::MAX` and the `+ 1` overflows.
    ///
    /// A caller in the drag loop sees none of that: a `NaN` density is caught by the first assertion
    /// downstream, but a plausible temperature next to it invites the reading that only the density
    /// leg failed. So the domain is enforced at the boundary and every field goes `NaN` together.
    #[test]
    fn every_output_is_nan_below_the_barometric_floor_and_at_a_non_finite_height() {
        let at = |alt: f64| jb2008(AMJD, SUN_RA, SUN_DEC, SUN_RA + 1.0, 0.3, alt, &DRIVERS);

        for &alt in &[
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -1.0,
            0.0,
            50.0,
            89.0,
            89.5,
            89.999_999,
        ] {
            let out = at(alt);
            assert!(
                out.rho.is_nan() && out.t_exo.is_nan() && out.t_alt.is_nan(),
                "alt {alt} km is off-domain: rho {}, t_exo {}, t_alt {}",
                out.rho,
                out.t_exo,
                out.t_alt
            );
        }

        // The floor is inclusive, and it is the only altitude that separates `< 90.0` from `<= 90.0`.
        for &alt in &[90.0, 90.000_001, 105.0, 400.0, 1500.0] {
            let out = at(alt);
            assert!(
                out.rho.is_finite()
                    && out.rho > 0.0
                    && out.t_exo.is_finite()
                    && out.t_alt.is_finite(),
                "alt {alt} km is in-domain: rho {}, t_exo {}, t_alt {}",
                out.rho,
                out.t_exo,
                out.t_alt
            );
        }
    }

    /// `dtsub` is five polynomial pieces glued at 120/200/240/300/600/800 km, and the gluing is the
    /// whole of its construction: the two cubics exist only to carry value and slope across their
    /// joins. Nothing measured that. The 600–800 km cubic in particular was sampled once, at 700 km,
    /// where `zp = (700−600)/100` is exactly 1 — and `dd · zp · zp · zp` is `dd` for *any*
    /// association of those three multiplications, so all three of them could be turned into
    /// divisions and every golden case still passed.
    ///
    /// The end conditions are what the construction fixes, so they are what is asserted: the
    /// correction vanishes at both ends of the supported range and is continuous at all five joins.
    /// Worst measured `|dTc(800 km)|` over the sweep below is 2.84e-14 K (1.98e-15 of the value at
    /// 600 km); the surviving mutants leave `−6·dd`, of order 10 K.
    #[test]
    fn the_dtc_bands_join_continuously_and_vanish_at_both_ends() {
        const JOINS: [f64; 6] = [120.0, 200.0, 240.0, 300.0, 600.0, 800.0];
        let mut worst_jump = 0.0_f64;
        let mut worst_end = 0.0_f64;
        for lat in [-1.53_f64, -0.7, 0.0, 0.7, 1.53] {
            for i in 0..=24 {
                let xlst = f64::from(i);
                for f10 in [70.0_f64, 100.0, 150.0, 250.0, 350.0] {
                    for z in JOINS {
                        let below = dtsub(f10, xlst, lat, z - 1.0e-9);
                        let above = dtsub(f10, xlst, lat, z + 1.0e-9);
                        worst_jump = nan_max(worst_jump, (above - below).abs());
                    }
                    // Both ends of the supported range: the model contributes no dTc below 120 km
                    // or above 800 km, so the cubics must reach zero there.
                    worst_end = nan_max(worst_end, dtsub(f10, xlst, lat, 120.0).abs());
                    worst_end = nan_max(worst_end, dtsub(f10, xlst, lat, 800.0).abs());
                }
            }
        }
        assert!(
            worst_jump < 1.0e-8,
            "dTc band joins are discontinuous by {worst_jump:e} K"
        );
        assert!(
            worst_end < 1.0e-12,
            "dTc does not vanish at 120/800 km: {worst_end:e} K"
        );
    }

    /// Eq. (14) blends F10B and S10B with a weight `min(1, (F10B/240)^¼)`, and the clamp is what
    /// makes the blend saturate: above F10B = 240 the weight is 1, S10B drops out of `fsb`, and the
    /// nighttime minimum exospheric temperature stops responding to it. Every fixture in this file
    /// runs at F10B = 137.1, where the clamp never fires, so the comparison could be replaced by an
    /// equality test — never true, clamp never applied — and nothing moved.
    ///
    /// The two driver sets below differ only in S10 and S10B, by the same amount, which holds the
    /// direct `2.259·(S10 − S10B)` term fixed and leaves `fsb` as S10B's only remaining path into
    /// the temperature. Under the clamp that path is closed and the two temperatures are bit-equal;
    /// without it the weight is 1.0574, `(1 − fnw)` goes negative, and `fsb` moves by 3.44 units —
    /// 11 K of exospheric temperature.
    #[test]
    fn above_the_f10b_ceiling_the_euv_index_stops_moving_the_temperature() {
        const SOLAR_MAX: Jb2008Input = Jb2008Input {
            f10: 320.0,
            f10b: 300.0,
            s10: 300.0,
            s10b: 280.0,
            xm10: 300.0,
            xm10b: 280.0,
            y10: 300.0,
            y10b: 280.0,
            dstdtc: 100.0,
        };
        const SHIFTED: Jb2008Input = Jb2008Input {
            s10: 360.0,
            s10b: 340.0,
            ..SOLAR_MAX
        };
        // The same shift below the ceiling, where the clamp does not fire.
        const SOLAR_MIN: Jb2008Input = Jb2008Input {
            f10: 90.0,
            f10b: 80.0,
            ..SOLAR_MAX
        };
        const SOLAR_MIN_SHIFTED: Jb2008Input = Jb2008Input {
            s10: 360.0,
            s10b: 340.0,
            ..SOLAR_MIN
        };
        let t_exo =
            |input: &Jb2008Input| jb2008(AMJD, SUN_RA, SUN_DEC, SAT_RA, 0.3, 400.0, input).t_exo;
        let (clamped, clamped_shifted) = (t_exo(&SOLAR_MAX), t_exo(&SHIFTED));
        assert!(
            clamped.to_bits() == clamped_shifted.to_bits(),
            "S10B still reaches the exospheric temperature above F10B = 240: \
             {clamped} vs {clamped_shifted}"
        );
        // And the clamp is not vacuous: below the ceiling the same shift does move it.
        let (a, b) = (t_exo(&SOLAR_MIN), t_exo(&SOLAR_MIN_SHIFTED));
        assert!(
            ((a - b) / a).abs() > 1.0e-3,
            "below the ceiling S10B should still move the temperature: {a} vs {b}"
        );
    }

    /// The [`Jb2008Atmosphere`] wrapper, evaluated in f64 at exactly the reference point, returns the
    /// bare model density (the local-exponential factor is `exp(0) = 1` there).
    #[test]
    fn atmosphere_wrapper_reduces_to_model_at_point() {
        let (lat, alt, _, _, rho_ref) = GOLDEN[8]; // 400 km
        // Place the "inertial" position on the equator so |r| is unambiguous; the geometry closure
        // returns the golden angles regardless, so the model sees the reference inputs.
        let r_mag = (6_356.766 + alt) * 1_000.0;
        let atmo = Jb2008Atmosphere {
            mjd0: AMJD,
            input: DRIVERS,
            geometry: |_t: f64, _r: [f64; 3]| (alt, lat, SAT_RA),
            sun: |_t: f64| (SUN_RA, SUN_DEC),
        };
        let rho = atmo.density(0.0, &[r_mag, 0.0, 0.0]);
        assert!(((rho - rho_ref) / rho_ref).abs() < 1.0e-6);
    }

    /// The wrapper's `Scalar`-generic density produces a finite, negative radial gradient in `Dual`
    /// (density falls with altitude), so it drops into the STM / dd-precision paths unchanged.
    #[test]
    fn atmosphere_wrapper_is_generic_over_scalar() {
        let atmo = Jb2008Atmosphere {
            mjd0: AMJD,
            input: DRIVERS,
            geometry: |_t: f64, r: [f64; 3]| {
                let rm = libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
                (rm / 1_000.0 - 6_356.766, 0.5, SAT_RA) // altitude_km from |r|, fixed lat/ra
            },
            sun: |_t: f64| (SUN_RA, SUN_DEC),
        };
        let r0 = (6_356.766 + 400.0) * 1_000.0;
        let r: [Dual<3>; 3] = [
            Dual::variable(r0, 0),
            Dual::variable(0.0, 1),
            Dual::variable(0.0, 2),
        ];
        let rho = atmo.density(0.0, &r);
        assert!(rho.value() > 0.0 && rho.value().is_finite());
        // ∂ρ/∂x along the radial (x) direction must be finite and negative.
        let dpdx = rho.grad()[0];
        assert!(dpdx.is_finite() && dpdx < 0.0, "dρ/dx = {dpdx:e}");
    }

    /// Unit direction of the scale-height fixture, off both the equator and the axes:
    /// `0.36² + 0.48² + 0.8² = 1` exactly. Evaluated along `+x`, the `y` and `z` terms of `|r|` and
    /// `|r₀|` are multiplied by a zero the fixture itself supplied and their mutants are unobservable.
    const DIR: [f64; 3] = [0.36, 0.48, 0.8];

    /// Earth radius the altitude fixture reduces against, km.
    const RE_KM: f64 = 6_356.766;

    /// Position at radius `r` along [`DIR`].
    fn along(r: f64) -> [f64; 3] {
        [r * DIR[0], r * DIR[1], r * DIR[2]]
    }

    /// Altitude tracks `|r|` (spherical) at a fixed latitude and right ascension, so a `Dual` variable
    /// is a pure change of height. Named so the oracle can ask it for the same altitude the wrapper
    /// computes rather than recovering it by arithmetic that rounds differently.
    fn fixture_geometry(_t: f64, r: [f64; 3]) -> (f64, f64, f64) {
        (
            libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) / 1_000.0 - RE_KM,
            0.5,
            SAT_RA,
        )
    }

    /// The wrapper carrying the fixture closures. Both capture nothing, so they are plain function
    /// pointers rather than two opaque `impl Fn` parameters.
    type Fixture = Jb2008Atmosphere<fn(f64, [f64; 3]) -> (f64, f64, f64), fn(f64) -> (f64, f64)>;

    fn fixture() -> Fixture {
        Jb2008Atmosphere {
            mjd0: AMJD,
            input: DRIVERS,
            geometry: fixture_geometry,
            sun: |_t: f64| (SUN_RA, SUN_DEC),
        }
    }

    /// In f64 the lift is `ρ₀·exp((|r₀|−|r|)/H)` evaluated at `|r| = |r₀|`, i.e. `exp(0) = 1` whatever
    /// `H` is, so `density_pair`, its 10 km offset and the `−10 000/ln(ρ_up/ρ₀)` line are dead — which
    /// is why they survived. `H` is observable only through the derivative, and the wrapper test above
    /// asserts only that the derivative is finite and negative.
    ///
    /// The oracle is the model's own log-slope over the same 10 km, from `jb2008` directly — the
    /// function the FORTRAN golden sweep pins — so the lift must reproduce `∂ρ/∂rᵢ = −ρ₀·r̂ᵢ/H`.
    #[test]
    fn the_local_scale_height_is_the_models_own_log_slope() {
        let atmo = fixture();
        // 450 km is deliberately absent: JB2008's scale height there is 50.3 km, within 1% of the
        // fallback constant, so a fixture at that altitude cannot tell the computed `H` from the
        // constant it falls back to.
        for alt_km in [200.0, 300.0, 800.0] {
            let r0 = (RE_KM + alt_km) * 1_000.0;
            let (alt, lat, ra) = fixture_geometry(0.0, along(r0));
            let rho0 = jb2008(AMJD, SUN_RA, SUN_DEC, ra, lat, alt, &DRIVERS).rho;
            let rho_up = jb2008(AMJD, SUN_RA, SUN_DEC, ra, lat, alt + 10.0, &DRIVERS).rho;
            let h = -10_000.0 / libm::log(rho_up / rho0);
            let slope = -rho0 / h; // ∂ρ/∂|r|; the chain rule gives ∂ρ/∂rᵢ = slope · rᵢ/|r|

            let p = along(r0);
            let r: [Dual<3>; 3] = [
                Dual::variable(p[0], 0),
                Dual::variable(p[1], 1),
                Dual::variable(p[2], 2),
            ];
            let got = atmo.density(0.0, &r).grad();
            let mut rel = 0.0_f64;
            for i in 0..3 {
                rel = nan_max(rel, ((got[i] - slope * DIR[i]) / (slope * DIR[i])).abs());
            }
            println!(
                "JB2008 {alt_km:.0} km: H = {h:.1} m, ∇ρ {got:?}, ∂ρ/∂|r| {slope:e} (rel {rel:e})"
            );
            assert!(
                rel < 1.0e-12,
                "at {alt_km:.0} km the lift's scale height is not the model's: {got:?} vs {slope:e}·DIR"
            );
            // Not the 50 km fallback: otherwise the assertion is measuring a constant.
            assert!(
                ((h - 50_000.0) / 50_000.0).abs() > 0.05,
                "the fixture landed on the fallback, so it proves nothing: H = {h:.1}"
            );
        }
    }

    /// The premise the wrapper's two guards rest on, swept rather than assumed.
    ///
    /// `density` falls back to a 50 km scale height when `!(ρ₀ > 0 && ρ_up > 0)` or when the computed
    /// `hh` is non-finite or `≤ 1000 m`. Five mutants of those guards survive — `>` weakened to `>=`
    /// on each of the three comparisons and `&&` widened to `||` on each pair — because every one of
    /// them changes the branch only where an operand sits exactly on the boundary or where the two
    /// operands disagree. This measures how far from either the model ever gets, over its whole valid
    /// range and both extremes of solar and geomagnetic activity, and that margin is what makes those
    /// five equivalent rather than merely unkilled.
    #[test]
    fn neither_lift_guard_comes_near_its_boundary_anywhere_in_the_models_range() {
        let mut worst_rho = f64::INFINITY;
        let mut worst_hh = f64::INFINITY;
        // JB2008 is defined from 90 km up; 2500 km is past every altitude the drag path is used at.
        for quiet in [true, false] {
            let drivers = if quiet {
                Jb2008Input {
                    f10: 70.0,
                    f10b: 70.0,
                    s10: 60.0,
                    s10b: 60.0,
                    xm10: 70.0,
                    xm10b: 70.0,
                    y10: 70.0,
                    y10b: 70.0,
                    dstdtc: 0.0,
                }
            } else {
                Jb2008Input {
                    f10: 300.0,
                    f10b: 250.0,
                    s10: 280.0,
                    s10b: 240.0,
                    xm10: 280.0,
                    xm10b: 240.0,
                    y10: 280.0,
                    y10b: 240.0,
                    dstdtc: 250.0,
                }
            };
            let mut alt = 90.0;
            while alt <= 2500.0 {
                for (lat, ra) in [(-1.5, 0.0), (0.0, SAT_RA), (1.5, 5.5)] {
                    let rho0 = jb2008(AMJD, SUN_RA, SUN_DEC, ra, lat, alt, &drivers).rho;
                    let rho_up = jb2008(AMJD, SUN_RA, SUN_DEC, ra, lat, alt + 10.0, &drivers).rho;
                    let hh = -10_000.0 / libm::log(rho_up / rho0);
                    assert!(
                        rho0.is_finite() && rho_up.is_finite() && hh.is_finite(),
                        "non-finite at {alt} km: ρ₀ {rho0:e}, ρ_up {rho_up:e}, hh {hh:e}"
                    );
                    worst_rho = nan_min(worst_rho, nan_min(rho0, rho_up));
                    worst_hh = nan_min(worst_hh, hh);
                }
                alt += 10.0;
            }
        }
        println!("JB2008 90-2500 km: min ρ = {worst_rho:e} kg/m³, min H = {worst_hh:.1} m");
        assert!(worst_rho > 1e-30, "density reached {worst_rho:e} kg/m³");
        assert!(worst_hh > 3_000.0, "H fell to {worst_hh:.1} m");
    }

    fn nan_min(a: f64, b: f64) -> f64 {
        if a < b { a } else { b }
    }

    /// `mjd = mjd0 + t/86 400` is the wrapper's only time dependence, and every test above evaluates
    /// at `t = 0`, where the term is zero: dropping it, negating it or rescaling it changes nothing
    /// they can see. JB2008 reads the epoch through its semiannual and seasonal-latitudinal terms, so
    /// two thirds of a year later the density is a different number and the model says which.
    #[test]
    fn the_epoch_advances_with_the_propagation_time() {
        let atmo = fixture();
        let p = along((RE_KM + 450.0) * 1_000.0);
        let (alt, lat, ra) = fixture_geometry(0.0, p);

        let t = 2.0 / 3.0 * 365.25 * 86_400.0;
        let want = jb2008(AMJD + t / 86_400.0, SUN_RA, SUN_DEC, ra, lat, alt, &DRIVERS).rho;
        // At |r| = |r₀| the lift is exp(0) = 1, so the f64 density is the model's own value.
        let got: f64 = atmo.density(t, &p);
        let at_epoch: f64 = atmo.density(0.0, &p);
        let season = (got / at_epoch - 1.0).abs();
        println!(
            "JB2008 MJD {AMJD:.3} -> {:.3}: ρ {at_epoch:e} -> {got:e} ({:.1}% seasonal)",
            AMJD + t / 86_400.0,
            100.0 * season
        );
        assert!(
            ((got - want) / want).abs() < 1.0e-15,
            "the epoch did not advance with t: {got:e} vs {want:e}"
        );
        // The season has to move, or the assertion above is satisfied by any wrong epoch too.
        assert!(
            season > 0.05,
            "the fixture's season barely moves ({:.3}%), so it proves nothing",
            100.0 * season
        );
    }

    /// Below 105 km the model takes the single-integration branch and hydrogen is negligible; the
    /// 91 km case must still reproduce the FORTRAN density to full working precision.
    #[test]
    fn low_altitude_branch() {
        let (lat, alt, _, _, rho_ref) = GOLDEN[0];
        let out = jb2008(AMJD, SUN_RA, SUN_DEC, SAT_RA, lat, alt, &DRIVERS);
        assert!(((out.rho - rho_ref) / rho_ref).abs() < 1.0e-6);
    }
}
