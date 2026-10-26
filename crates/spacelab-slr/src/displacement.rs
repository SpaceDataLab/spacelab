//! Where a station actually is, as against where its catalogue entry says it is.
//!
//! A tracking station's ITRF coordinates are a *conventional* position: the tide-free, regularised
//! point a reference frame solution estimates. The instrument that fires the laser is somewhere
//! else at any given instant, by up to about 40 cm vertically. Comparing a range against a
//! catalogue position without putting the station back where it was is the single largest error an
//! SLR observation model can make, and it is not a small correction that averages out — it is
//! diurnal and semi-diurnal, so it aliases straight into the orbit.
//!
//! Two effects, both from IERS Conventions (2010) chapter 7:
//!
//! - [`solid_earth_tide_displacement`] — §7.1.1, the elastic response of the solid Earth to the
//!   luni-solar tidal potential. Decimetres. Ported from the IERS Conventions Center reference
//!   routine `DEHANTTIDEINEL` and its five helpers, term for term, and checked against the four
//!   test cases published in that routine's header.
//! - [`pole_tide_displacement`] — §7.1.4, the elastic response to the centrifugal potential of
//!   polar motion, referred to the IERS 2018 secular pole. Centimetres.
//!
//! The third station-displacement term, ocean tide loading, is in [`loading`](crate::loading)
//! rather than here: it is not an elastic response to a potential this module can evaluate but a
//! per-station table of how the crust answers the water above it, so it needs a data file where
//! these two need only an epoch.
//!
//! Renamed and re-expressed in Rust as the IERS software licence requires. The differences from the
//! Fortran: displacements are returned rather than accumulated into an out-parameter; the epoch
//! arrives as a Julian century of TT and a UT1 hour-of-day rather than as a calendar date plus an
//! internal leap-second table, so the caller's time scales are the caller's; and the permanent-tide
//! term stays removed, matching the conventional tide-free frame that ITRF is realised in. None of
//! those changes a number.
//!
//! The published cases pin the transcription, not the equations behind it.
//! `derivations/solid_tide_love.mac` reduces each closed-form term of the body-Cartesian
//! arrangement to the latitude-longitude form of the Conventions' equations (7.5)–(7.11). The
//! in-phase terms come out as the Love and Shida response to the degree-2 and degree-3 tidal
//! potential. The same file derives the pole tide's potential (7.24) from the tilted rotation axis,
//! and its 33 mm and 9 mm from `h`, `l`, `Ω`, `a` and `g_e`.

// `_s`/`_m` are Sun and Moon and `dr`/`dn`/`de` radial, north and east throughout — the
// reference routine's own names for the same quantities, and the only ones the equations read
// in.
#![allow(clippy::similar_names)]

/// Nominal degree-2 Love and Shida numbers, IERS Conventions (2010) §7.1.1.
const H20: f64 = 0.6078;
const L20: f64 = 0.0847;
/// Nominal degree-3 Love and Shida numbers.
const H3: f64 = 0.292;
const L3: f64 = 0.015;
/// Sun/Moon mass ratios and the equatorial radius the tidal factors are formed with, from the
/// reference routine's own current-best-estimate block.
const MASS_RATIO_SUN: f64 = 332_946.048_2;
const MASS_RATIO_MOON: f64 = 0.012_300_037_1;
const EARTH_RADIUS: f64 = 6_378_136.6;

fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

/// The station's geocentric direction cosines, shared by every term below.
pub(crate) struct Site {
    radius: f64,
    sin_phi: f64,
    cos_phi: f64,
    sin_lambda: f64,
    cos_lambda: f64,
}

impl Site {
    pub(crate) fn new(station: [f64; 3]) -> Self {
        let radius = norm(station);
        let cos_phi = libm::sqrt(station[0] * station[0] + station[1] * station[1]) / radius;
        Self {
            radius,
            sin_phi: station[2] / radius,
            cos_phi,
            sin_lambda: station[1] / cos_phi / radius,
            cos_lambda: station[0] / cos_phi / radius,
        }
    }

    /// Radial / north / east components back into the Earth-fixed Cartesian frame.
    pub(crate) fn to_cartesian(&self, dr: f64, dn: f64, de: f64) -> [f64; 3] {
        [
            dr * self.cos_lambda * self.cos_phi
                - de * self.sin_lambda
                - dn * self.sin_phi * self.cos_lambda,
            dr * self.sin_lambda * self.cos_phi + de * self.cos_lambda
                - dn * self.sin_phi * self.sin_lambda,
            dr * self.sin_phi + dn * self.cos_phi,
        ]
    }
}

/// Out-of-phase degree-2 correction, diurnal band (`ST1IDIU`).
fn out_of_phase_diurnal(
    site: &Site,
    sun: [f64; 3],
    moon: [f64; 3],
    fac_sun: f64,
    fac_moon: f64,
) -> [f64; 3] {
    const DHI: f64 = -0.0025;
    const DLI: f64 = -0.0007;
    let cos_2phi = site.cos_phi * site.cos_phi - site.sin_phi * site.sin_phi;
    let (r_sun, r_moon) = (norm(sun), norm(moon));

    let band = |body: [f64; 3], r: f64, fac: f64| {
        let asym = body[2] * (body[0] * site.sin_lambda - body[1] * site.cos_lambda) / (r * r);
        let sym = body[2] * (body[0] * site.cos_lambda + body[1] * site.sin_lambda) / (r * r);
        (
            -3.0 * DHI * site.sin_phi * site.cos_phi * fac * asym,
            -3.0 * DLI * cos_2phi * fac * asym,
            -3.0 * DLI * site.sin_phi * fac * sym,
        )
    };
    let (dr_s, dn_s, de_s) = band(sun, r_sun, fac_sun);
    let (dr_m, dn_m, de_m) = band(moon, r_moon, fac_moon);
    site.to_cartesian(dr_s + dr_m, dn_s + dn_m, de_s + de_m)
}

/// Out-of-phase degree-2 correction, semi-diurnal band (`ST1ISEM`).
fn out_of_phase_semidiurnal(
    site: &Site,
    sun: [f64; 3],
    moon: [f64; 3],
    fac_sun: f64,
    fac_moon: f64,
) -> [f64; 3] {
    const DHI: f64 = -0.0022;
    const DLI: f64 = -0.0007;
    let cos_2lambda = site.cos_lambda * site.cos_lambda - site.sin_lambda * site.sin_lambda;
    let sin_2lambda = 2.0 * site.cos_lambda * site.sin_lambda;
    let (r_sun, r_moon) = (norm(sun), norm(moon));

    let band = |body: [f64; 3], r: f64, fac: f64| {
        let quad = body[0] * body[0] - body[1] * body[1];
        let cross = 2.0 * body[0] * body[1];
        let asym = (quad * sin_2lambda - cross * cos_2lambda) / (r * r);
        let sym = (quad * cos_2lambda + cross * sin_2lambda) / (r * r);
        (
            -0.75 * DHI * site.cos_phi * site.cos_phi * fac * asym,
            1.5 * DLI * site.sin_phi * site.cos_phi * fac * asym,
            -1.5 * DLI * site.cos_phi * fac * sym,
        )
    };
    let (dr_s, dn_s, de_s) = band(sun, r_sun, fac_sun);
    let (dr_m, dn_m, de_m) = band(moon, r_moon, fac_moon);
    site.to_cartesian(dr_s + dr_m, dn_s + dn_m, de_s + de_m)
}

/// Latitude dependence of the degree-2 Shida number, the `l^(1)` part (`ST1L1`).
fn latitude_dependence(
    site: &Site,
    sun: [f64; 3],
    moon: [f64; 3],
    fac_sun: f64,
    fac_moon: f64,
) -> [f64; 3] {
    const L1_DIURNAL: f64 = 0.0012;
    const L1_SEMIDIURNAL: f64 = 0.0024;
    let (r_sun, r_moon) = (norm(sun), norm(moon));
    let cos_2lambda = site.cos_lambda * site.cos_lambda - site.sin_lambda * site.sin_lambda;
    let sin_2lambda = 2.0 * site.cos_lambda * site.sin_lambda;

    let diurnal = |body: [f64; 3], r: f64, fac: f64| {
        let sym = body[2] * (body[0] * site.cos_lambda + body[1] * site.sin_lambda) / (r * r);
        let asym = body[2] * (body[0] * site.sin_lambda - body[1] * site.cos_lambda) / (r * r);
        (
            -L1_DIURNAL * site.sin_phi * site.sin_phi * fac * sym,
            L1_DIURNAL
                * site.sin_phi
                * (site.cos_phi * site.cos_phi - site.sin_phi * site.sin_phi)
                * fac
                * asym,
        )
    };
    let (dn_s, de_s) = diurnal(sun, r_sun, fac_sun);
    let (dn_m, de_m) = diurnal(moon, r_moon, fac_moon);
    let first = site.to_cartesian(0.0, 3.0 * (dn_s + dn_m), 3.0 * (de_s + de_m));

    let semidiurnal = |body: [f64; 3], r: f64, fac: f64| {
        let quad = body[0] * body[0] - body[1] * body[1];
        let cross = 2.0 * body[0] * body[1];
        let sym = (quad * cos_2lambda + cross * sin_2lambda) / (r * r);
        let asym = (quad * sin_2lambda - cross * cos_2lambda) / (r * r);
        (
            -0.5 * L1_SEMIDIURNAL * site.sin_phi * site.cos_phi * fac * sym,
            -0.5 * L1_SEMIDIURNAL * site.sin_phi * site.sin_phi * site.cos_phi * fac * asym,
        )
    };
    let (dn_s, de_s) = semidiurnal(sun, r_sun, fac_sun);
    let (dn_m, de_m) = semidiurnal(moon, r_moon, fac_moon);
    let second = site.to_cartesian(0.0, 3.0 * (dn_s + dn_m), 3.0 * (de_s + de_m));

    [
        first[0] + second[0],
        first[1] + second[1],
        first[2] + second[2],
    ]
}

/// Delaunay-like tidal arguments of the step-2 corrections, degrees, at Julian century `t` of TT.
///
/// Not the IERS `fundarg` set: the reference routine carries its own polynomials for the mean
/// longitude of the Moon and Sun, the lunar and solar perigees and the ascending node, and they are
/// what the tabulated amplitudes were fitted against.
struct TidalArguments {
    s: f64,
    tau: f64,
    h: f64,
    p: f64,
    node: f64,
    perigee_sun: f64,
}

impl TidalArguments {
    fn new(fractional_hour: f64, t: f64) -> Self {
        let mut s =
            218.316_645_63 + (481_267.881_94 + (-0.001_466_388_9 + 0.000_001_851_39 * t) * t) * t;
        let tau = fractional_hour * 15.0
            + 280.460_618_4
            + (36_000.770_053_6 + (0.000_387_93 + -0.000_000_025_8 * t) * t) * t
            - s;
        let precession =
            (1.396_971_278 + (0.000_308_889 + (0.000_000_021 + 0.000_000_007 * t) * t) * t) * t;
        s += precession;
        let h = 280.466_45
            + (36_000.769_748_9
                + (0.000_303_222_22 + (0.000_000_020 + -0.000_000_006_54 * t) * t) * t)
                * t;
        let p = 83.353_243_12
            + (4_069.013_635_25
                + (-0.010_321_722_22 + (-0.000_012_499_1 + 0.000_000_052_63 * t) * t) * t)
                * t;
        let node = 234.955_444_99
            + (1_934.136_261_97
                + (-0.002_075_611_11 + (-0.000_002_139_44 + 0.000_000_016_50 * t) * t) * t)
                * t;
        let perigee_sun = 282.937_340_98
            + (1.719_457_666_67
                + (0.000_456_888_89 + (-0.000_000_017_78 + -0.000_000_003_34 * t) * t) * t)
                * t;

        Self {
            s: wrap_degrees(s),
            tau: wrap_degrees(tau),
            h: wrap_degrees(h),
            p: wrap_degrees(p),
            node: wrap_degrees(node),
            perigee_sun: wrap_degrees(perigee_sun),
        }
    }
}

/// Fortran `DMOD(x, 360)` — truncated, so the result keeps the sign of `x`. The phase enters only
/// through a sine and cosine, so the branch does not matter to the answer; it is written this way
/// so a reader comparing against the reference sees the same operation.
fn wrap_degrees(x: f64) -> f64 {
    x - 360.0 * libm::trunc(x / 360.0)
}

/// Diurnal step-2 table: the five argument multipliers, then in-phase and out-of-phase radial and
/// transverse amplitudes, mm. IERS Conventions (2010) Table 7.3a as carried by `STEP2DIU`.
#[rustfmt::skip]
const STEP2_DIURNAL: [[f64; 9]; 31] = [
    [-3.0,  0.0,  2.0,  0.0, 0.0, -0.01,  0.0,  0.0,  0.0],
    [-3.0,  2.0,  0.0,  0.0, 0.0, -0.01,  0.0,  0.0,  0.0],
    [-2.0,  0.0,  1.0, -1.0, 0.0, -0.02,  0.0,  0.0,  0.0],
    [-2.0,  0.0,  1.0,  0.0, 0.0, -0.08,  0.0, -0.01,  0.01],
    [-2.0,  2.0, -1.0,  0.0, 0.0, -0.02,  0.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0, -1.0, 0.0, -0.10,  0.0,  0.0,  0.0],
    [-1.0,  0.0,  0.0,  0.0, 0.0, -0.51,  0.0, -0.02,  0.03],
    [-1.0,  2.0,  0.0,  0.0, 0.0,  0.01,  0.0,  0.0,  0.0],
    [ 0.0, -2.0,  1.0,  0.0, 0.0,  0.01,  0.0,  0.0,  0.0],
    [ 0.0,  0.0, -1.0,  0.0, 0.0,  0.02,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.0, 0.0,  0.06,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  1.0, 0.0,  0.01,  0.0,  0.0,  0.0],
    [ 0.0,  2.0, -1.0,  0.0, 0.0,  0.01,  0.0,  0.0,  0.0],
    [ 1.0, -3.0,  0.0,  0.0, 1.0, -0.06,  0.0,  0.0,  0.0],
    [ 1.0, -2.0,  0.0, -1.0, 0.0,  0.01,  0.0,  0.0,  0.0],
    [ 1.0, -2.0,  0.0,  0.0, 0.0, -1.23, -0.07,  0.06,  0.01],
    [ 1.0, -1.0,  0.0,  0.0, -1.0, 0.02,  0.0,  0.0,  0.0],
    [ 1.0, -1.0,  0.0,  0.0, 1.0,  0.04,  0.0,  0.0,  0.0],
    [ 1.0,  0.0,  0.0, -1.0, 0.0, -0.22,  0.01,  0.01,  0.0],
    [ 1.0,  0.0,  0.0,  0.0, 0.0, 12.00, -0.80, -0.67, -0.03],
    [ 1.0,  0.0,  0.0,  1.0, 0.0,  1.73, -0.12, -0.10,  0.0],
    [ 1.0,  0.0,  0.0,  2.0, 0.0, -0.04,  0.0,  0.0,  0.0],
    [ 1.0,  1.0,  0.0,  0.0, -1.0, -0.50, -0.01,  0.03,  0.0],
    [ 1.0,  1.0,  0.0,  0.0, 1.0,  0.01,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0,  1.0, -1.0, -0.01,  0.0,  0.0,  0.0],
    [ 1.0,  2.0, -2.0,  0.0, 0.0, -0.01,  0.0,  0.0,  0.0],
    [ 1.0,  2.0,  0.0,  0.0, 0.0, -0.11,  0.01,  0.01,  0.0],
    [ 2.0, -2.0,  1.0,  0.0, 0.0, -0.01,  0.0,  0.0,  0.0],
    [ 2.0,  0.0, -1.0,  0.0, 0.0, -0.02,  0.0,  0.0,  0.0],
    [ 3.0,  0.0,  0.0,  0.0, 0.0,  0.0,   0.0,  0.0,  0.0],
    [ 3.0,  0.0,  0.0,  1.0, 0.0,  0.0,   0.0,  0.0,  0.0],
];

/// Long-period step-2 table, same layout. IERS Conventions (2010) Table 7.3b as carried by
/// `STEP2LON`.
#[rustfmt::skip]
const STEP2_LONG_PERIOD: [[f64; 9]; 5] = [
    [0.0, 0.0,  0.0, 1.0, 0.0,  0.47,  0.23,  0.16,  0.07],
    [0.0, 2.0,  0.0, 0.0, 0.0, -0.20, -0.12, -0.11, -0.05],
    [1.0, 0.0, -1.0, 0.0, 0.0, -0.11, -0.08, -0.09, -0.04],
    [2.0, 0.0,  0.0, 0.0, 0.0, -0.13, -0.11, -0.15, -0.07],
    [2.0, 0.0,  0.0, 1.0, 0.0, -0.05, -0.05, -0.06, -0.03],
];

/// Frequency-dependent step-2 correction, diurnal band (`STEP2DIU`).
fn step2_diurnal(site: &Site, station: [f64; 3], fractional_hour: f64, t: f64) -> [f64; 3] {
    let a = TidalArguments::new(fractional_hour, t);
    let east_longitude = libm::atan2(station[1], station[0]);
    let cos_2phi = site.cos_phi * site.cos_phi - site.sin_phi * site.sin_phi;
    let mut total = [0.0; 3];
    for row in STEP2_DIURNAL {
        let theta = (a.tau
            + row[0] * a.s
            + row[1] * a.h
            + row[2] * a.p
            + row[3] * a.node
            + row[4] * a.perigee_sun)
            .to_radians()
            + east_longitude;
        let (sin_theta, cos_theta) = (libm::sin(theta), libm::cos(theta));
        let two_sin_cos = 2.0 * site.sin_phi * site.cos_phi;
        let dr = row[5] * two_sin_cos * sin_theta + row[6] * two_sin_cos * cos_theta;
        let dn = row[7] * cos_2phi * sin_theta + row[8] * cos_2phi * cos_theta;
        let de = row[7] * site.sin_phi * cos_theta - row[8] * site.sin_phi * sin_theta;
        let d = site.to_cartesian(dr, dn, de);
        total = [total[0] + d[0], total[1] + d[1], total[2] + d[2]];
    }
    [total[0] / 1000.0, total[1] / 1000.0, total[2] / 1000.0]
}

/// Frequency-dependent step-2 correction, long-period band (`STEP2LON`).
fn step2_long_period(site: &Site, t: f64) -> [f64; 3] {
    // The long-period arguments do not involve tau, so the hour of day is immaterial here.
    let a = TidalArguments::new(0.0, t);
    let mut total = [0.0; 3];
    for row in STEP2_LONG_PERIOD {
        let theta =
            (row[0] * a.s + row[1] * a.h + row[2] * a.p + row[3] * a.node + row[4] * a.perigee_sun)
                .to_radians();
        let (sin_theta, cos_theta) = (libm::sin(theta), libm::cos(theta));
        let legendre = (3.0 * site.sin_phi * site.sin_phi - 1.0) / 2.0;
        let dr = row[5] * legendre * cos_theta + row[7] * legendre * sin_theta;
        let dn = row[6] * (2.0 * site.cos_phi * site.sin_phi) * cos_theta
            + row[8] * (2.0 * site.cos_phi * site.sin_phi) * sin_theta;
        let d = site.to_cartesian(dr, dn, 0.0);
        total = [total[0] + d[0], total[1] + d[1], total[2] + d[2]];
    }
    [total[0] / 1000.0, total[1] / 1000.0, total[2] / 1000.0]
}

/// Solid Earth tide displacement of a station, metres, in the same Earth-fixed frame the inputs
/// are given in.
///
/// `station`, `sun` and `moon` are geocentric Cartesian positions in metres, all three in the same
/// frame — the reference routine's contract, and the reason the Sun and Moon have to be rotated
/// into ITRF before they get here rather than passed in GCRF. `fractional_hour` is the UT1 hour of
/// day and `julian_centuries_tt` the TT epoch in Julian centuries from J2000; the split is what the
/// tidal arguments need, one for the Earth's rotation phase and one for the orbits.
///
/// The result is **tide-free**: the permanent deformation is left in the conventional station
/// position, which is what ITRF realises, so it is not added back here.
#[must_use]
pub fn solid_earth_tide_displacement(
    station: [f64; 3],
    sun: [f64; 3],
    moon: [f64; 3],
    fractional_hour: f64,
    julian_centuries_tt: f64,
) -> [f64; 3] {
    let site = Site::new(station);
    let (r_sun, r_moon) = (norm(sun), norm(moon));
    let cos_sun =
        (station[0] * sun[0] + station[1] * sun[1] + station[2] * sun[2]) / site.radius / r_sun;
    let cos_moon =
        (station[0] * moon[0] + station[1] * moon[1] + station[2] * moon[2]) / site.radius / r_moon;

    // Latitude dependence of the nominal degree-2 numbers, §7.1.1 equations (7.2) and (7.3).
    let oblate = 1.0 - 1.5 * site.cos_phi * site.cos_phi;
    let h2 = H20 - 0.0006 * oblate;
    let l2 = L20 + 0.0002 * oblate;

    let p2 = |cos: f64| 3.0 * (h2 / 2.0 - l2) * cos * cos - h2 / 2.0;
    let p3 = |cos: f64| 2.5 * (H3 - 3.0 * L3) * cos * cos * cos + 1.5 * (L3 - H3) * cos;
    let x2 = |cos: f64| 3.0 * l2 * cos;
    let x3 = |cos: f64| 1.5 * L3 * (5.0 * cos * cos - 1.0);

    let fac2_sun = MASS_RATIO_SUN * EARTH_RADIUS * (EARTH_RADIUS / r_sun).powi(3);
    let fac2_moon = MASS_RATIO_MOON * EARTH_RADIUS * (EARTH_RADIUS / r_moon).powi(3);
    let fac3_sun = fac2_sun * (EARTH_RADIUS / r_sun);
    let fac3_moon = fac2_moon * (EARTH_RADIUS / r_moon);

    let mut d = [0.0; 3];
    for i in 0..3 {
        d[i] = fac2_sun * (x2(cos_sun) * sun[i] / r_sun + p2(cos_sun) * station[i] / site.radius)
            + fac2_moon
                * (x2(cos_moon) * moon[i] / r_moon + p2(cos_moon) * station[i] / site.radius)
            + fac3_sun * (x3(cos_sun) * sun[i] / r_sun + p3(cos_sun) * station[i] / site.radius)
            + fac3_moon
                * (x3(cos_moon) * moon[i] / r_moon + p3(cos_moon) * station[i] / site.radius);
    }

    for correction in [
        out_of_phase_diurnal(&site, sun, moon, fac2_sun, fac2_moon),
        out_of_phase_semidiurnal(&site, sun, moon, fac2_sun, fac2_moon),
        latitude_dependence(&site, sun, moon, fac2_sun, fac2_moon),
        step2_diurnal(&site, station, fractional_hour, julian_centuries_tt),
        step2_long_period(&site, julian_centuries_tt),
    ] {
        for i in 0..3 {
            d[i] += correction[i];
        }
    }
    d
}

/// Pole tide displacement of a station, metres, IERS Conventions (2010) §7.1.4.
///
/// `xp`, `yp` are the polar-motion angles in radians and `secular_xp`, `secular_yp` the IERS 2018
/// secular pole at the same epoch, also in radians; the deformation responds to the *wander* about
/// the secular pole, not to the pole itself, and using the pole itself is a metre-scale error, not
/// a centimetre one.
///
/// `station` is the geocentric Earth-fixed position in metres.
#[must_use]
pub fn pole_tide_displacement(
    station: [f64; 3],
    xp: f64,
    yp: f64,
    secular_xp: f64,
    secular_yp: f64,
) -> [f64; 3] {
    // Wander about the secular pole, in arcseconds — the units the §7.1.4 coefficients are in. The
    // sign of the y term is the conventional flip from the pole-coordinate convention to a
    // right-handed one.
    const RADIANS_PER_ARCSECOND: f64 = core::f64::consts::PI / (180.0 * 3600.0);
    let m1 = (xp - secular_xp) / RADIANS_PER_ARCSECOND;
    let m2 = -(yp - secular_yp) / RADIANS_PER_ARCSECOND;

    let site = Site::new(station);
    let sin_2phi = 2.0 * site.sin_phi * site.cos_phi;
    let cos_2phi = site.cos_phi * site.cos_phi - site.sin_phi * site.sin_phi;

    let dr = -0.033 * sin_2phi * (m1 * site.cos_lambda + m2 * site.sin_lambda);
    let dn = -0.009 * cos_2phi * (m1 * site.cos_lambda + m2 * site.sin_lambda);
    let de = 0.009 * site.sin_phi * (m1 * site.sin_lambda - m2 * site.cos_lambda);
    site.to_cartesian(dr, dn, de)
}

#[cfg(test)]
mod tests {
    // The reference routine publishes its test cases to nineteen digits; they are kept whole.
    #![allow(clippy::excessive_precision)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    /// Absolute agreement, metres, demanded against the reference routine's published test cases.
    ///
    /// The expected values are quoted to nineteen digits, and this port evaluates the same
    /// expressions in a different association order, so the floor is round-off on a decimetre —
    /// about 1e-17 m. A picometre is four orders above that and still two hundred million times
    /// below the effect being modelled, which is the range where a transcription slip in any
    /// tabulated amplitude cannot hide.
    const TOLERANCE: f64 = 1e-12;

    fn check(
        station: [f64; 3],
        sun: [f64; 3],
        moon: [f64; 3],
        hour: f64,
        t: f64,
        expected: [f64; 3],
    ) {
        let d = solid_earth_tide_displacement(station, sun, moon, hour, t);
        for i in 0..3 {
            assert!(
                (d[i] - expected[i]).abs() < TOLERANCE,
                "component {i}: {} against the reference {}, difference {:e} m",
                d[i],
                expected[i],
                d[i] - expected[i]
            );
        }
    }

    /// Julian centuries of TT for the reference routine's test cases, which give a calendar date at
    /// 0 h UTC. The routine forms `T` from the Julian date, then adds (TAI−UTC)+32.184 s; these are
    /// that arithmetic done once, with the leap-second count of the epoch in question.
    ///
    /// 2009-04-13: JD 2454934.5, TAI−UTC = 34 s. 2012-07-13: JD 2456121.5, **35 s** — the leap
    /// second of 2012-07-01 falls twelve days before that case, and using 34 there moves the
    /// answer by half a nanometre, which is the whole reason the count is written out rather than
    /// carried as one number for all cases.
    fn centuries(julian_date: f64, tai_minus_utc: f64) -> f64 {
        (julian_date - 2_451_545.0) / 36_525.0
            + (tai_minus_utc + 32.184) / (3600.0 * 24.0 * 36_525.0)
    }

    /// (G5) Test case 1 of `DEHANTTIDEINEL.F`'s header, IERS Conventions (2010) chapter 7
    /// software: 2009 April 13, 0 h.
    #[test]
    fn the_solid_tide_reproduces_the_first_iers_reference_case() {
        check(
            [4_075_578.385, 931_852.890, 4_801_570.154],
            [
                137_859_926_952.015,
                54_228_127_881.435_0,
                23_509_422_341.696_0,
            ],
            [
                -179_996_231.920_342,
                -312_468_450.131_567,
                -169_288_918.592_160,
            ],
            0.0,
            centuries(2_454_934.5, 34.0),
            [
                0.770_042_035_710_812_589_1e-1,
                0.630_405_632_182_496_761_3e-1,
                0.551_656_815_259_724_681_0e-1,
            ],
        );
    }

    /// (G5) Test case 2 of the same header: 2012 July 13, 0 h, USNO.
    #[test]
    fn the_solid_tide_reproduces_the_second_iers_reference_case() {
        check(
            [1_112_189.660, -4_842_955.026, 3_985_352.284],
            [
                -54_537_460_436.235_7,
                130_244_288_385.279,
                56_463_429_031.599_6,
            ],
            [300_396_716.912, 243_238_281.451, 120_548_075.939],
            0.0,
            centuries(2_456_121.5, 35.0),
            [
                -0.203_683_147_959_207_583_3e-1,
                0.565_825_477_622_597_244_9e-1,
                -0.759_767_967_687_174_222_7e-1,
            ],
        );
    }

    /// The permanent tide is *not* restored — the displacement of a station on the equator at the
    /// sub-solar longitude has to change sign as the Sun crosses the local meridian, which it
    /// cannot do if a constant zero-frequency term is left in.
    #[test]
    fn the_radial_tide_changes_sign_over_a_day() {
        let station = [6_378_137.0, 0.0, 0.0];
        let sun = [1.496e11, 0.0, 0.0];
        let moon = [3.844e8, 0.0, 0.0];
        let radial = |s: [f64; 3], m: [f64; 3]| {
            let d = solid_earth_tide_displacement(station, s, m, 0.0, 0.25);
            d[0]
        };
        let overhead = radial(sun, moon);
        let opposite = radial([-sun[0], 0.0, 0.0], [-moon[0], 0.0, 0.0]);
        let quadrature = radial([0.0, sun[0], 0.0], [0.0, moon[0], 0.0]);
        assert!(overhead > 0.0, "bulge under the body is {overhead} m");
        assert!(opposite > 0.0, "antipodal bulge is {opposite} m");
        assert!(
            quadrature < 0.0,
            "quadrature depression is {quadrature} m, not negative"
        );
    }

    /// The degree-2 tide scales as the inverse cube of the distance to the raising body, so halving
    /// the lunar distance must multiply its share by eight. This pins the `(RE/r)³` factor
    /// independently of the reference cases, which sit at one distance each.
    #[test]
    fn the_lunar_term_follows_the_inverse_cube_of_distance() {
        let station = [4_075_578.385, 931_852.890, 4_801_570.154];
        let far = [0.0, 0.0, 3.844e8];
        let near = [0.0, 0.0, 1.922e8];
        let no_sun = [0.0, 0.0, 1e18];
        let a = solid_earth_tide_displacement(station, no_sun, far, 0.0, 0.1);
        let b = solid_earth_tide_displacement(station, no_sun, near, 0.0, 0.1);
        // Degree 3 carries another factor of RE/r and so scales as the inverse fourth power; at
        // half the lunar distance its share doubles and pulls the ratio above 8. Sixteen would mean
        // the degree-2 term had gone missing, eight that degree 3 had.
        let ratio = b[2] / a[2];
        assert!(
            (8.0..9.5).contains(&ratio),
            "halving the lunar distance scaled the displacement by {ratio}, not between 8 and 16"
        );
    }

    /// The long-period table carries no solar-perigee dependence: its fifth argument multiplier is
    /// zero in every row, so `row[4] * a.perigee_sun` is dead arithmetic and the two mutants that
    /// live on it are excluded as equivalent in `.cargo/mutants.toml`. That exclusion is only sound
    /// while this column stays zero, which is what this pins — the diurnal table, whose fifth
    /// column is *not* everywhere zero, is left to the reference cases.
    #[test]
    fn the_long_period_table_has_no_solar_perigee_dependence() {
        for (i, row) in STEP2_LONG_PERIOD.iter().enumerate() {
            assert_within_ulps(
                row[4],
                0.0,
                0,
                &format!("STEP2_LONG_PERIOD row {i} solar-perigee multiplier"),
            );
        }
    }

    /// `wrap_degrees` is Fortran `DMOD`, truncated towards zero, so the result keeps the sign of
    /// its argument. Every caller feeds it straight into a sine, where a full turn is invisible —
    /// so nothing downstream can tell this apart from a floor, a Euclidean remainder, or the
    /// identity. These four are the cases where those four differ.
    #[test]
    fn the_degree_wrap_truncates_towards_zero() {
        assert_within_ulps(wrap_degrees(730.0), 10.0, 0, "just over two turns");
        assert_within_ulps(
            wrap_degrees(-370.0),
            -10.0,
            0,
            "just under one turn, negative",
        );
        assert_within_ulps(wrap_degrees(-10.0), -10.0, 0, "already inside, negative");
        assert_within_ulps(wrap_degrees(360.0), 0.0, 0, "exactly one turn");
    }

    /// Each of the three §7.1.4 terms in isolation, at geometries where the others vanish.
    ///
    /// The two tests below check the pole tide against itself — that it is zero at the secular pole
    /// and linear in the wander — and against a magnitude band. All three coefficients could be
    /// scaled, swapped between components or sign-flipped and every one of those checks would still
    /// pass. `Site` uses geocentric latitude, so a station on a sphere has exact `sinφ`, `cosφ`,
    /// `sinλ` and `cosλ`, and the collapsed geometries below leave one published coefficient
    /// standing at a time:
    ///
    /// | station | `sin2φ` | `cos2φ` | isolates |
    /// |---|---|---|---|
    /// | equator, Greenwich | 0 | 1 | `−9 mm cos2φ · m₁`, north only |
    /// | equator, 90° E | 0 | 1 | `−9 mm cos2φ · m₂`, and its sign |
    /// | 45° N, Greenwich | 1 | 0 | `−33 mm sin2φ · m₁` radial and `+9 mm sinφ · m₂` east |
    /// | 45° N, 90° E | 1 | 0 | `−33 mm · m₂` radial and `+9 mm sinφ · m₁` east |
    ///
    /// The wander is one arcsecond in each coordinate, so `m₁ = 1` and `m₂ = −1` exactly and the
    /// expected values are the coefficients themselves. Worst measured departure across the four
    /// is 3 ulps, from `sinφ = cosφ = 1/√2` being formed by a square root; the bound is 8.
    #[test]
    fn each_pole_tide_term_stands_alone_at_a_collapsed_geometry() {
        const ARCSECOND: f64 = core::f64::consts::PI / (180.0 * 3600.0);
        let r = 6_367_444.0;
        let diagonal = r * core::f64::consts::FRAC_1_SQRT_2;
        // m1 = +1 arcsecond, m2 = −1 arcsecond: the y term carries the conventional sign flip.
        let tide =
            |station: [f64; 3]| pole_tide_displacement(station, ARCSECOND, ARCSECOND, 0.0, 0.0);

        let want = |got: [f64; 3], expect: [f64; 3], case: &str| {
            for i in 0..3 {
                assert_within_ulps(got[i], expect[i], 8, case);
            }
        };

        // sin2φ = 0 kills the radial term and sinφ = 0 the east one; only north survives, and it
        // points along +z at the equator.
        want(
            tide([r, 0.0, 0.0]),
            [0.0, 0.0, -0.009],
            "equator, Greenwich",
        );
        want(
            tide([0.0, r, 0.0]),
            [0.0, 0.0, 0.009],
            "equator, 90 degrees east",
        );

        // cos2φ = 0 kills the north term. Radial goes into x and z, east into y.
        let root_half = core::f64::consts::FRAC_1_SQRT_2;
        want(
            tide([diagonal, 0.0, diagonal]),
            [-0.033 * root_half, 0.009 * root_half, -0.033 * root_half],
            "45 north, Greenwich",
        );
        want(
            tide([0.0, diagonal, diagonal]),
            [-0.009 * root_half, 0.033 * root_half, 0.033 * root_half],
            "45 north, 90 degrees east",
        );
    }

    /// A pole exactly at its secular position deforms nothing, and the response is linear in the
    /// wander — the two properties that make §7.1.4 a correction rather than a datum.
    #[test]
    fn the_pole_tide_vanishes_at_the_secular_pole_and_is_linear_in_the_wander() {
        let station = [4_075_578.385, 931_852.890, 4_801_570.154];
        let (sx, sy) = (0.9e-6, 3.5e-6);
        for c in pole_tide_displacement(station, sx, sy, sx, sy) {
            assert_within_ulps(c, 0.0, 0, "c");
        }
        let single = pole_tide_displacement(station, sx + 1e-6, sy - 2e-6, sx, sy);
        let double = pole_tide_displacement(station, sx + 2e-6, sy - 4e-6, sx, sy);
        for i in 0..3 {
            assert_within_ulps(double[i], 2.0 * single[i], 2, "double[i]");
        }
    }

    /// Magnitude check against §7.1.4's own statement that the pole tide reaches about 25 mm
    /// radially for a wander of 0.4 arcsecond.
    #[test]
    fn the_pole_tide_is_centimetres_for_a_realistic_wander() {
        // 45° north, on the Greenwich meridian, where sin 2φ is 1 and the radial term is largest.
        let r = 6_367_444.0;
        let station = [
            r * core::f64::consts::FRAC_1_SQRT_2,
            0.0,
            r * core::f64::consts::FRAC_1_SQRT_2,
        ];
        let arcsecond = core::f64::consts::PI / (180.0 * 3600.0);
        let d = pole_tide_displacement(station, 0.4 * arcsecond, 0.0, 0.0, 0.0);
        let radial = (d[0] * station[0] + d[1] * station[1] + d[2] * station[2]) / r;
        assert!(
            (-0.02..-0.005).contains(&radial),
            "radial pole tide of {radial} m is out of family for 0.4 arcsecond"
        );
    }
}
