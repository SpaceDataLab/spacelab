//! Low-precision analytic ephemerides of the perturbing bodies.
//!
//! A third-body or radiation-pressure force needs the Sun's (and Moon's) position as a function of
//! time, but only to the accuracy those small perturbations warrant — a solar position good to a few
//! hundredths of a degree changes a `~10⁻⁷ m/s²` acceleration in its last figures, not its first.
//! These closed-form series (Astronomical Almanac / Montenbruck–Gill low precision) need no data
//! files and no interpolation, which keeps the propagator `no_std` and self-contained. They are
//! **not** for pointing, eclipse timing to the second, or anything where the body position is the
//! answer rather than a perturbation — that is what a JPL/SPICE kernel is for, and where a future
//! `spacelab-ephemeris` over `anise` would go.

// Astronomy speaks in single letters — L (mean longitude), g (mean anomaly), λ (ecliptic
// longitude), ε (obliquity), r, x, y, z. These are the sources' names. The lunar-series amplitudes
// are a transcribed table, so their digit grouping is the source's, not a readability choice.
#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::inconsistent_digit_grouping
)]

use core::f64::consts::PI;

const DEG: f64 = PI / 180.0;
/// The astronomical unit, metres — IAU 2012 exact definition.
pub const ASTRONOMICAL_UNIT: f64 = 1.495_978_707e11;
/// Seconds per Julian century.
const SECONDS_PER_JULIAN_CENTURY: f64 = 86_400.0 * 36_525.0;
/// The IAU 1976 constant of annual aberration, radians. Divided by the Earth–Sun distance in AU it
/// is the longitude by which an apparent solar position trails the geometric one.
const CONSTANT_OF_ABERRATION: f64 = 20.495_52 * DEG / 3600.0;

/// The geocentric position of a perturbing body in the inertial (equatorial, J2000/GCRF-aligned)
/// frame, as a function of time past the propagation epoch.
///
/// Returns `f64`, not a [`Scalar`](spacelab_autodiff::Scalar): the body's position depends on time
/// alone, never on the satellite state being differentiated, so it enters a force's Jacobian as a
/// constant. The force lifts it with `Scalar::from_f64`.
///
/// An implementation with no position for a given `t` returns the frame origin, and every force in
/// this crate reads that through [`is_absent`] as *no perturbing body*. See its documentation for
/// why the sentinel is in-band.
pub trait Ephemeris {
    /// Geocentric inertial position of the body, metres, at `t` seconds past the epoch.
    ///
    /// `[0.0; 3]` means the implementation cannot answer for this `t` — see [`is_absent`].
    fn position(&self, t: f64) -> [f64; 3];
}

/// Whether an [`Ephemeris::position`] is the *absent-body* value: the frame origin.
///
/// The trait hands back a bare `[f64; 3]` and has no error channel. It is `no_std`, and its value
/// enters a force's Jacobian as a constant, so a `Result` would have to be dealt with inside every
/// right-hand side of every integrator step. The origin can stand in for "unknown" because it is not
/// a configuration a caller can mean: a perturbing body coincident with the frame origin makes the
/// indirect term `μ s/‖s‖³` of the tidal difference infinite, so every model built on it is
/// undefined there whatever was intended.
///
/// The forces have to honour it explicitly, because the arithmetic does not do it for them. Left to
/// run, `‖s‖ = 0` gives `ThirdBody` a `NaN` and gives the radiation forces something worse — a
/// *finite* acceleration scaled by `(AU/‖r‖)²`, eight orders too large, which no downstream check
/// on the state would flag as a failure.
///
/// The comparison is exact, and against both signed zeros. A body a metre off the origin is a
/// physical input, however unlikely, and is not this.
#[must_use]
pub fn is_absent(position: [f64; 3]) -> bool {
    // Not a tolerance: the sentinel is the literal origin, written by the implementation itself.
    #[allow(clippy::float_cmp)]
    {
        position[0] == 0.0 && position[1] == 0.0 && position[2] == 0.0
    }
}

/// Rotates a vector from the mean equator and equinox **of date** into J2000, by the IAU 2006
/// equatorial precession angles `ζ_A`, `z_A`, `θ_A` (Capitaine, Wallace & Chapront 2003,
/// arcseconds).
///
/// The `~17 mas` frame bias between J2000 and the GCRS is three orders below the series that
/// consumes this and is not applied; the matrix is otherwise `erfa.pmat06` to `0.017″`.
fn j2000_from_of_date(tc: f64, v: [f64; 3]) -> [f64; 3] {
    const ARCSEC: f64 = DEG / 3600.0;
    let zeta = ARCSEC
        * (2.650_545
            + tc * (2306.083_227
                + tc * (0.298_849_9
                    + tc * (0.018_018_28 + tc * (-0.000_005_971 + tc * -0.000_000_317_3)))));
    let z = ARCSEC
        * (-2.650_545
            + tc * (2306.077_181
                + tc * (1.092_734_8
                    + tc * (0.018_268_37 + tc * (-0.000_028_596 + tc * -0.000_000_290_4)))));
    let theta = ARCSEC
        * tc
        * (2004.191_903
            + tc * (-0.429_493_4
                + tc * (-0.041_822_64 + tc * (-0.000_007_089 + tc * -0.000_000_127_4))));

    let (sin_zeta, cos_zeta) = libm::sincos(zeta);
    let (sin_z, cos_z) = libm::sincos(z);
    let (sin_theta, cos_theta) = libm::sincos(theta);

    // `p` is `R3(−z)·R2(θ)·R3(−ζ)`, which maps J2000 → of date. A rotation's inverse is its
    // transpose, so the vector is contracted against the *columns*.
    let p = [
        [
            cos_zeta * cos_theta * cos_z - sin_zeta * sin_z,
            -sin_zeta * cos_theta * cos_z - cos_zeta * sin_z,
            -sin_theta * cos_z,
        ],
        [
            cos_zeta * cos_theta * sin_z + sin_zeta * cos_z,
            -sin_zeta * cos_theta * sin_z + cos_zeta * cos_z,
            -sin_theta * sin_z,
        ],
        [cos_zeta * sin_theta, -sin_zeta * sin_theta, cos_theta],
    ];
    [
        p[0][0] * v[0] + p[1][0] * v[1] + p[2][0] * v[2],
        p[0][1] * v[0] + p[1][1] * v[1] + p[2][1] * v[2],
        p[0][2] * v[0] + p[1][2] * v[1] + p[2][2] * v[2],
    ]
}

/// The Sun, by the Astronomical Almanac's low-precision series: geocentric ecliptic longitude from
/// the mean longitude and the equation of centre, then rotation into the equatorial frame by the
/// mean obliquity.
///
/// That series is referred to the mean equinox and equator **of date** — the obliquity's
/// `−0.0130042 T` is precession in the obliquity, and the mean longitude's `36000.771°/century` is
/// the tropical rate — so it is precessed back to J2000 here, which is the frame this trait
/// promises and the frame its consumers work in. Skipping that rotation costs `0.33°` at 2024 and
/// `1.4°` a century out, two orders above the series' own error.
///
/// The Almanac also prints the **apparent** longitude, so the series carries annual aberration and
/// every consumer here wants geometry. `κ/r` is added back, `κ = 20.49552″`; the classical constant
/// form is used rather than an explicit `v_earth/c`, because at this accuracy the two are the same
/// answer — `derivations/vallado_errata.py` measures 34.23″ worst and 11.06″ RMS against 34.23″ and
/// 11.05″ over ±25 y, and identical to four figures over ±100 y. Leaving the aberration in costs
/// 45.77″ worst and 18.06″ RMS, so this one term is 39 % of the RMS error (#306).
///
/// Note what the same erratum's other half would cost: the Almanac pairs its apparent longitude with
/// the **mean** obliquity, so the equator the series defines is already the mean one, and rotating
/// nutation out — as "TOD, not MOD" reads — moves the residual the wrong way, to 56.35″.
///
/// Measured against ERFA `epv00` over ±1 century of J2000, 512 epochs: `≤ 0.0132°` in angle and
/// `≤ 9.9e-5 AU` in range — ample for the third-body and radiation-pressure perturbations that
/// consume it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticSun {
    /// The propagation epoch, in seconds past J2000 (2000-01-01 12:00 TT). The caller converts its
    /// epoch to this at the boundary; the ephemeris itself carries no time-scale machinery.
    pub epoch_j2000_seconds: f64,
}

impl Ephemeris for AnalyticSun {
    fn position(&self, t: f64) -> [f64; 3] {
        // Julian centuries of (TT) time past J2000.
        let tc = (self.epoch_j2000_seconds + t) / SECONDS_PER_JULIAN_CENTURY;

        let mean_longitude = (280.460 + 36_000.771 * tc) * DEG;
        let mean_anomaly = (357.528 + 35_999.050 * tc) * DEG;

        // Earth–Sun distance, AU, from the low-order eccentricity terms. Read before the longitude
        // because the aberration term below is scaled by it.
        let r_au = 1.000_14
            - 0.016_71 * libm::cos(mean_anomaly)
            - 0.000_14 * libm::cos(2.0 * mean_anomaly);
        let r = r_au * ASTRONOMICAL_UNIT;

        // The Almanac prints the apparent longitude; every consumer wants the geometric one, so the
        // annual aberration it carries is added back.
        let ecliptic_longitude = mean_longitude
            + (1.915 * libm::sin(mean_anomaly) + 0.020 * libm::sin(2.0 * mean_anomaly)) * DEG
            + CONSTANT_OF_ABERRATION / r_au;
        let obliquity = (23.439_291 - 0.013_004_2 * tc) * DEG;

        let (sin_lambda, cos_lambda) =
            (libm::sin(ecliptic_longitude), libm::cos(ecliptic_longitude));
        let (sin_eps, cos_eps) = (libm::sin(obliquity), libm::cos(obliquity));
        j2000_from_of_date(
            tc,
            [
                r * cos_lambda,
                r * cos_eps * sin_lambda,
                r * sin_eps * sin_lambda,
            ],
        )
    }
}

/// The Moon, by Montenbruck & Gill's low-precision series ("Satellite Orbits" §3.3.2): the ecliptic
/// longitude, latitude and distance from a handful of Delaunay-argument terms, rotated to the
/// equatorial frame by the J2000 obliquity. That series is referred to J2000 already — unlike the
/// solar one — so no precession is applied.
///
/// Unlike the solar series it is also **geometric**, which is a measurement rather than an
/// assumption: applying the light-time shift the Moon's own 1.28 s range implies moves the series
/// 0.758″, and the residual against `moon98` the wrong way, 278.3″ → 279.0″
/// (`derivations/analytic_ephemeris.py`, `light_time_check`). So the aberration correction
/// [`AnalyticSun`] needs has no counterpart here.
///
/// Measured against ERFA `moon98` over ±1 century of J2000, 512 epochs: `≤ 278″` in angle and
/// `≤ 458 km` in range — the Moon is the largest third-body perturber, but a `~10⁻⁶ m/s²`
/// acceleration tolerates that.
///
/// The larger of the luni-solar perturbations; pair it with an [`AnalyticSun`] third body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnalyticMoon {
    /// The propagation epoch, in seconds past J2000 (2000-01-01 12:00 TT), as for [`AnalyticSun`].
    pub epoch_j2000_seconds: f64,
}

impl Ephemeris for AnalyticMoon {
    fn position(&self, t: f64) -> [f64; 3] {
        const ARCSEC: f64 = DEG / 3600.0;
        let tc = (self.epoch_j2000_seconds + t) / SECONDS_PER_JULIAN_CENTURY;
        let frac = |x: f64| x - libm::floor(x);

        // Mean arguments (fractions of a turn → radians): Moon's mean longitude, Moon's and Sun's
        // mean anomalies, the mean elongation, and the argument of latitude.
        let l0 = frac(0.606_433 + 1336.851_344 * tc);
        let l = PI * 2.0 * frac(0.374_897 + 1325.552_410 * tc);
        let lp = PI * 2.0 * frac(0.993_133 + 99.997_361 * tc);
        let d = PI * 2.0 * frac(0.827_361 + 1236.853_086 * tc);
        let f = PI * 2.0 * frac(0.259_086 + 1342.227_825 * tc);

        // Ecliptic longitude correction, arcseconds.
        let d_l = 22640.0 * libm::sin(l) - 4586.0 * libm::sin(l - 2.0 * d)
            + 2370.0 * libm::sin(2.0 * d)
            + 769.0 * libm::sin(2.0 * l)
            - 668.0 * libm::sin(lp)
            - 412.0 * libm::sin(2.0 * f)
            - 212.0 * libm::sin(2.0 * l - 2.0 * d)
            - 206.0 * libm::sin(l + lp - 2.0 * d)
            + 192.0 * libm::sin(l + 2.0 * d)
            - 165.0 * libm::sin(lp - 2.0 * d)
            - 125.0 * libm::sin(d)
            - 110.0 * libm::sin(l + lp)
            + 148.0 * libm::sin(l - lp)
            - 55.0 * libm::sin(2.0 * f - 2.0 * d);
        let longitude = PI * 2.0 * frac(l0 + d_l / 1_296_000.0);

        // Ecliptic latitude.
        let s = f + (d_l + 412.0 * libm::sin(2.0 * f) + 541.0 * libm::sin(lp)) * ARCSEC;
        let h = f - 2.0 * d;
        let n = -526.0 * libm::sin(h) + 44.0 * libm::sin(l + h)
            - 31.0 * libm::sin(-l + h)
            - 23.0 * libm::sin(lp + h)
            + 11.0 * libm::sin(-lp + h)
            - 25.0 * libm::sin(-2.0 * l + f)
            + 21.0 * libm::sin(-l + f);
        let latitude = (18520.0 * libm::sin(s) + n) * ARCSEC;

        // Geocentric distance, metres.
        let r = 385000e3
            - 20905e3 * libm::cos(l)
            - 3699e3 * libm::cos(2.0 * d - l)
            - 2956e3 * libm::cos(2.0 * d)
            - 570e3 * libm::cos(2.0 * l)
            + 246e3 * libm::cos(2.0 * l - 2.0 * d)
            - 205e3 * libm::cos(lp - 2.0 * d)
            - 171e3 * libm::cos(l + 2.0 * d)
            - 152e3 * libm::cos(l + lp - 2.0 * d);

        // Ecliptic → equatorial by a frame rotation about x through −ε (J2000 mean obliquity).
        let (sin_lat, cos_lat) = (libm::sin(latitude), libm::cos(latitude));
        let (sin_lon, cos_lon) = (libm::sin(longitude), libm::cos(longitude));
        let (x_ec, y_ec, z_ec) = (r * cos_lon * cos_lat, r * sin_lon * cos_lat, r * sin_lat);
        let (sin_eps, cos_eps) = libm::sincos(23.439_291_11 * DEG);
        [
            x_ec,
            cos_eps * y_ec - sin_eps * z_ec,
            sin_eps * y_ec + cos_eps * z_ec,
        ]
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]

    use super::*;
    use spacelab_testkit::nan_max;

    /// `(seconds past J2000, the series' position, ERFA's position)`, metres, from
    /// `derivations/analytic_ephemeris.py` at 50 decimal digits.
    ///
    /// Two columns because they answer two different questions. The **series** column is a
    /// transcription pin: the derivation writes the published arrangement — degrees and
    /// arcseconds, sums spelled out, the rotations at the end — against this module's
    /// radians-as-you-go Horner, so any single amplitude, argument or sign moves a digit here.
    /// The **ERFA** column is an accuracy oracle independent of the series in origin and in
    /// method (`epv00` for the Sun, the abridged ELP2000 `moon98` for the Moon), and it is what
    /// pins the *frame*: a transcription can be perfect and still be referred to the wrong
    /// equinox.
    type Golden = (f64, [f64; 3], [f64; 3]);

    const SUN: [Golden; 13] = [
        (
            -3.15576e9,
            [
                28167214741.63290835148,
                -132450499569.6860145069,
                -57459459572.14703589479,
            ],
            [
                28170396856.14079035946,
                -132446178961.8899991535,
                -57457476940.4035511461,
            ],
        ),
        (
            -1.5e9,
            [
                964832862.7099257047885,
                139475457346.0690153248,
                60487896645.34318882052,
            ],
            [
                962106319.0096082389991,
                139483586765.8201656694,
                60491652564.88166304806,
            ],
        ),
        (
            -7.84308258e8,
            [
                134120844712.8546511573,
                -57488695256.12007082724,
                -24927520071.87101081167,
            ],
            [
                134108786271.7815446001,
                -57492695550.91807288352,
                -24929835235.83183646567,
            ],
        ),
        (
            0.0,
            [
                26507662136.61559315173,
                -132752709207.3643176294,
                -57555343056.05256580437,
            ],
            [
                26499029719.14862760604,
                -132757417633.0395555619,
                -57556716961.19887535287,
            ],
        ),
        (
            8.64e7,
            [
                -149539728435.5121784151,
                -10023243602.41699522406,
                -4345443030.76398841894,
            ],
            [
                -149538016033.4322709736,
                -10023561367.59736998481,
                -4345651872.533140277234,
            ],
        ),
        (
            7.57339269184e8,
            [
                24822312238.47691160337,
                -133031282404.948326894,
                -57667652027.77624930643,
            ],
            [
                24813057981.6821171563,
                -133033125859.4346241374,
                -57667964264.91996867165,
            ],
        ),
        (
            1.5e9,
            [
                -55619282989.15422226305,
                129860653831.5653686855,
                56285458376.84220154416,
            ],
            [
                -55615047651.42481025186,
                129861110105.3145206153,
                56286221131.85145484482,
            ],
        ),
        (
            2.698761487e9,
            [
                -43225880879.06620477007,
                133799567147.1962361929,
                57979259628.74010544351,
            ],
            [
                -43207710532.46469292692,
                133790276668.877521761,
                57975440413.98861757422,
            ],
        ),
        (
            3.019895577e9,
            [
                -147448907803.9539196681,
                28214613873.855541376,
                12228531979.32650501963,
            ],
            [
                -147445775696.1285293855,
                28210189394.7600353037,
                12226249484.99575707038,
            ],
        ),
        (
            3.069300822e9,
            [
                144231034442.832146172,
                36628399290.19972918678,
                15867514260.85501836,
            ],
            [
                144242240519.0175900016,
                36598708248.93527928566,
                15854066093.2385393244,
            ],
        ),
        (
            3.15576e9,
            [
                24843208688.79265441104,
                -133038320794.846832528,
                -57643937671.95756632546,
            ],
            [
                24837806137.31792438277,
                -133043498495.67685623,
                -57645971187.8213753133,
            ],
        ),
        (
            -9.46728e9,
            [
                31470786476.8273681387,
                -131796627677.1325973102,
                -57245988825.89662595244,
            ],
            [
                31485054004.21063732549,
                -131773879434.6089218414,
                -57236488836.89407431702,
            ],
        ),
        (
            9.46728e9,
            [
                21500442602.51787138119,
                -133559421747.0977010895,
                -57799104154.66687922954,
            ],
            [
                21500894543.52257273114,
                -133574511782.6239542428,
                -57806116292.11318649401,
            ],
        ),
    ];

    const MOON: [Golden; 13] = [
        (
            -3.15576e9,
            [
                -21368234.87458305534443,
                -340402532.9456722445757,
                -144319777.1507238897981,
            ],
            [
                -21391401.42430451611118,
                -340496629.8213130613147,
                -144462776.7660692476242,
            ],
        ),
        (
            -1.5e9,
            [
                153129535.581166244392,
                319161587.0032233632897,
                175536815.071663627464,
            ],
            [
                153200762.4713019693789,
                319143855.5536886082757,
                175596006.7592621908043,
            ],
        ),
        (
            -7.84308258e8,
            [
                -199850876.2886467097804,
                284890752.0610037635742,
                94530875.69176748632204,
            ],
            [
                -199454550.0506111659666,
                285144855.1315032595891,
                94654403.23820460922499,
            ],
        ),
        (
            0.0,
            [
                -291492224.1600419237295,
                -266630266.9123104876023,
                -76098486.23612924421203,
            ],
            [
                -291605466.3790748250698,
                -266715233.2831513871592,
                -76099036.32740979337892,
            ],
        ),
        (
            8.64e7,
            [
                144249721.2888966671794,
                341598778.3845721997048,
                146715213.7042406616861,
            ],
            [
                144081054.5332234199268,
                341340854.235887187035,
                146579272.7186728862903,
            ],
        ),
        (
            7.57339269184e8,
            [
                -368023857.5828822338311,
                142542502.5717727254829,
                89256829.97011934228666,
            ],
            [
                -367984446.8822184779617,
                142719557.7767159284417,
                89312980.48207951671053,
            ],
        ),
        (
            1.5e9,
            [
                383386385.8694832701789,
                93266400.03211139968867,
                80793931.5841832280627,
            ],
            [
                383410142.7781092684525,
                93412997.58215474737172,
                80860624.27018081074537,
            ],
        ),
        (
            2.698761487e9,
            [
                235103362.928682364346,
                -258197564.7110772860935,
                -88683629.15761513318337,
            ],
            [
                235031516.566155143417,
                -258068607.4983842898952,
                -88570601.76263875371349,
            ],
        ),
        (
            3.019895577e9,
            [
                278027634.72473453821,
                -205701759.3326994193897,
                -121291981.1366689654601,
            ],
            [
                277586615.3443044905943,
                -205577891.8814867154296,
                -121131660.5272834780057,
            ],
        ),
        (
            3.069300822e9,
            [
                168326746.7535158537375,
                -317119623.0768393133642,
                -175680198.6037669867506,
            ],
            [
                168272844.7023613875117,
                -317359029.1115359315775,
                -175692525.3946109012565,
            ],
        ),
        (
            3.15576e9,
            [
                -355109676.1545417439696,
                98066972.62595107937489,
                45866094.09125699259836,
            ],
            [
                -354988786.597064229405,
                98256146.41166254613673,
                45961236.49165706967647,
            ],
        ),
        (
            -9.46728e9,
            [
                383529615.9667456047297,
                75747562.70271492963971,
                19353103.47426703702529,
            ],
            [
                383454604.6476667317226,
                75845176.68866854077868,
                19474032.26860967415789,
            ],
        ),
        (
            9.46728e9,
            [
                197646453.1899011536489,
                284825344.2749009289985,
                109603266.0788945640927,
            ],
            [
                197579188.8107239291919,
                285016658.5947043998889,
                109446409.3296933458826,
            ],
        ),
    ];

    /// The epoch of the last two goldens is three centuries out, where neither series is accurate
    /// and only the transcription is being pinned. Everything at or inside this bound carries an
    /// accuracy claim as well.
    const CLAIMED_SPAN: f64 = 3.15576e9;

    /// `2^24` seconds ≈ 194 days. A power of two, so `t − SPLIT` is exact for every golden epoch
    /// and so is the sum back — the goldens are evaluated at the argument they were computed at,
    /// with neither the epoch nor the elapsed time zero and the two unequal.
    const SPLIT: f64 = 16_777_216.0;

    fn norm(v: [f64; 3]) -> f64 {
        libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }

    fn relative_miss(got: [f64; 3], want: [f64; 3]) -> f64 {
        norm([got[0] - want[0], got[1] - want[1], got[2] - want[2]]) / norm(want)
    }

    fn separation_arcsec(a: [f64; 3], b: [f64; 3]) -> f64 {
        let cosine = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (norm(a) * norm(b));
        libm::acos(cosine.clamp(-1.0, 1.0)) / DEG * 3600.0
    }

    /// Each golden evaluated through the trait, at the `(epoch, elapsed)` split described above.
    fn evaluated(
        golden: &[Golden; 13],
        body: impl Fn(f64) -> [f64; 3],
    ) -> [([f64; 3], Golden); 13] {
        core::array::from_fn(|i| (body(golden[i].0), golden[i]))
    }

    fn sun() -> [([f64; 3], Golden); 13] {
        evaluated(&SUN, |t| {
            AnalyticSun {
                epoch_j2000_seconds: t - SPLIT,
            }
            .position(SPLIT)
        })
    }

    fn moon() -> [([f64; 3], Golden); 13] {
        evaluated(&MOON, |t| {
            AnalyticMoon {
                epoch_j2000_seconds: t - SPLIT,
            }
            .position(SPLIT)
        })
    }

    /// The bar is the f64 evaluation noise with room to spare, measured: worst `3.27e-13` for the
    /// Sun and `1.48e-12` for the Moon, both at three centuries out where the argument reduction of
    /// a `1336 revolutions/century` rate has the least left. `libm` is pure Rust and bit-identical
    /// across platforms, so that floor is a constant, not a sample.
    ///
    /// It has to be this tight. `frac(x) = x − ⌊x⌋` written as `x + ⌊x⌋` shifts every mean argument
    /// by an even number of turns, which no sine or cosine can see *mathematically* — only the lost
    /// argument-reduction digits give it away, and they move the Moon to `8.68e-12`. A looser bar
    /// makes that mutation invisible; the smallest thing the bar must still see is the quintic
    /// precession term, `3.0e-10` at three centuries.
    fn assert_is_the_series(label: &str, evaluated: &[([f64; 3], Golden); 13]) {
        let mut worst: f64 = 0.0;
        for &(got, (t, series, _)) in evaluated {
            let miss = relative_miss(got, series);
            assert!(
                miss < 5e-12,
                "{label} at t={t:e}: {miss:.3e} from the 50-digit series"
            );
            worst = nan_max(worst, miss);
        }
        assert!(
            worst > 0.0,
            "the goldens must exercise the series, not a constant"
        );
    }

    fn assert_reaches_erfa(
        label: &str,
        evaluated: &[([f64; 3], Golden); 13],
        angle_bar: f64,
        range_bar: f64,
    ) {
        for &(got, (t, _, erfa)) in evaluated {
            if libm::fabs(t) > CLAIMED_SPAN {
                continue;
            }
            let angle = separation_arcsec(got, erfa);
            let range = libm::fabs(norm(got) - norm(erfa));
            assert!(
                angle < angle_bar,
                "{label} at t={t:e}: {angle:.1}\" from ERFA"
            );
            assert!(
                range < range_bar,
                "{label} at t={t:e}: {range:.0} m from ERFA in range"
            );
        }
    }

    #[test]
    fn the_solar_series_is_the_almanac_series_precessed_to_j2000() {
        assert_is_the_series("Sun", &sun());
    }

    #[test]
    fn the_lunar_series_is_the_brown_series() {
        assert_is_the_series("Moon", &moon());
    }

    /// The frame claim, made assertable. Dropping the precession leaves the Sun 5054″ (1.4°) from
    /// ERFA at the ends of this span — 105× the bound below — while every accuracy test that
    /// existed before, all of them at or near J2000 where precession is zero, still passes.
    ///
    /// The solar bar is 48″ rather than 56″ because the aberration term (#306) moved the sweep's
    /// worst from 55.09″ to 47.42″, and the golden at `t = 3.069300822e9` is the epoch that worst
    /// occurs at, so the bar is witnessed rather than merely asserted.
    ///
    /// This test is **not** what pins the aberration term, and it is worth saying which one is.
    /// Removing `κ/r` and re-running leaves this test green: the term is a constant offset in
    /// longitude, so against the several other error sources in a low-precision series it cancels at
    /// some epochs and adds at others — over these eleven it takes the worst *down*, to 26.93″. What
    /// it buys is the RMS and the worst over a dense sweep, neither of which eleven hand-picked
    /// extremes can measure. The gate that does see the term is
    /// [`the_solar_series_is_the_almanac_series_precessed_to_j2000`]: the derivation carries the same
    /// `κ/r`, and 5e-12 leaves no room for a 20.5″ disagreement.
    #[test]
    fn both_bodies_reach_erfa_at_the_documented_accuracy() {
        assert_reaches_erfa("Sun", &sun(), 48.0, 1.0e-4 * ASTRONOMICAL_UNIT);
        assert_reaches_erfa("Moon", &moon(), 280.0, 460e3);
    }

    /// `ζ_A` and `z_A` are exact negatives at `t = 0` and `θ_A` is zero there, so the rotation
    /// collapses to `R3(0)`: a vector referred to the mean equinox of J2000 *is* a J2000 vector.
    #[test]
    fn the_precession_is_the_identity_at_its_own_epoch() {
        let v = [1.234e11, -5.678e10, 9.012e10];
        let miss = relative_miss(j2000_from_of_date(0.0, v), v);
        assert!(
            miss < 1e-15,
            "precession at J2000 moved a vector by {miss:.3e}"
        );
    }

    /// A frame change preserves lengths and angles; nothing in the goldens would separate a
    /// rotation from a general linear map that happens to agree along the ecliptic.
    #[test]
    fn the_precession_is_an_isometry() {
        let (u, v) = ([1.234e11, -5.678e10, 9.012e10], [-3.3e10, 7.7e10, -1.1e11]);
        for tc in [-3.0, -1.0, -0.13, 0.5, 1.0, 3.0] {
            let (ru, rv) = (j2000_from_of_date(tc, u), j2000_from_of_date(tc, v));
            for (before, after) in [(norm(u), norm(ru)), (norm(v), norm(rv))] {
                let drift = libm::fabs(after - before) / before;
                assert!(drift < 1e-15, "length moved {drift:.3e} at tc={tc}");
            }
            let opened = libm::fabs(separation_arcsec(ru, rv) - separation_arcsec(u, v));
            assert!(
                opened < 1e-6,
                "the angle between two vectors opened by {opened:.3e}\""
            );
        }
    }
}
