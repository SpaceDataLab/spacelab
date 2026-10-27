//! Receiver-to-satellite ray geometry.
//!
//! Ported from Orekit's `Ray`. The ray-perigee frame turns the slant path into an arc
//! parameterised by distance `s` from the perigee, and `Ray::point_at` maps a distance
//! back to the geodetic point where the electron density is evaluated.
//!
//! Gate: `derivations/nequick_ray.mac` shows that the point of Eq. 178 to 187 at distance `s` is
//! `rp·P + s·A`, a straight line through the perigee `P` along the azimuth `A`, that Eq. 155 and
//! 156 are the zenith angle and the line's distance from the centre, and, at 60 digits on four
//! geometries, that Eq. 153 to 177 put `s1` on the receiver and `s2` on the satellite. The module
//! tests `the_ray_perigee_is_the_one_the_icd_equations_define` (G2) and
//! `the_ray_is_a_straight_line_through_both_end_points` (G1) hold the code to them.

use std::f64::consts::{FRAC_PI_2, PI};

use crate::types::{GeodeticPoint, SinCos};

/// Mean Earth radius, m (reference Table 2.5.2).
pub(crate) const RE: f64 = 6371200.0;

/// Threshold guarding the ray-perigee degeneracies (pole / zenith).
const THRESHOLD: f64 = 1.0e-10;

/// Below this ray-perigee radius (m) the path is treated as a pure zenith segment.
const ZENITH_RADIUS: f64 = 1.0;

/// Ray-perigee parameters for a slant path. By convention point 1 is the lower point.
pub(crate) struct Ray {
    /// Receiver altitude, m.
    rec_h: f64,
    /// Satellite altitude, m.
    sat_h: f64,
    /// Distance of the satellite from the ray perigee, m.
    s2: f64,
    /// Ray-perigee radius, m.
    rp: f64,
    /// Ray-perigee latitude, rad.
    lat_p: f64,
    /// Ray-perigee longitude, rad.
    lon_p: f64,
    /// Sine/cosine of the ray-perigee latitude.
    sc_lat_p: SinCos,
    /// Sine of the satellite azimuth seen from the ray perigee.
    sin_az_p: f64,
    /// Cosine of the satellite azimuth seen from the ray perigee.
    cos_az_p: f64,
}

impl Ray {
    /// Build the ray-perigee frame from the receiver and satellite geodetic points.
    pub(crate) fn new(rec: GeodeticPoint, sat: GeodeticPoint) -> Self {
        // Integration limits in m (Eq. 140, 141).
        let rec_h = rec.altitude;
        let sat_h = sat.altitude;
        let r1 = RE + rec_h;
        let r2 = RE + sat_h;

        let lat1 = rec.latitude;
        let lat2 = sat.latitude;
        let lon1 = rec.longitude;
        let lon2 = sat.longitude;
        let sc_lat_sat = SinCos::of(lat2);
        let sc_lat_rec = SinCos::of(lat1);
        let sc_lon21 = SinCos::of(lon2 - lon1);

        // Zenith angle (Eq. 153 to 155, with the stabilised forms Orekit uses).
        let cos_d =
            (sc_lat_rec.s * sc_lat_sat.s + sc_lat_rec.c * sc_lat_sat.c * sc_lon21.c).min(1.0);
        let sin_d_sin_m = sc_lat_sat.c * sc_lon21.s;
        let sin_d_cos_m = sc_lat_sat.s * sc_lat_rec.c - sc_lat_sat.c * sc_lat_rec.s * sc_lon21.c;
        let sin_d = libm::sqrt(sin_d_sin_m * sin_d_sin_m + sin_d_cos_m * sin_d_cos_m);
        let u = r2 * sin_d;
        let v = r2 * cos_d - r1;
        let inv = 1.0 / libm::sqrt(u * u + v * v);
        let sin_z = u * inv;
        let cos_z = v * inv;

        // Ray-perigee radius (Eq. 156).
        let rp = r1 * sin_z;

        // Ray-perigee latitude and longitude.
        let (lat_p, lon_p) = if libm::fabs(libm::fabs(lat1) - FRAC_PI_2) < THRESHOLD {
            // Receiver almost at a pole (Eq. 157, 164).
            (libm::copysign(libm::atan2(u, v), lat1), lon2 + PI)
        } else if libm::fabs(sin_z) < THRESHOLD {
            // Satellite almost at the receiver zenith.
            (lat1, lon1)
        } else {
            // Eq. 158 to 167.
            let sin_az = sc_lon21.s * sc_lat_sat.c / sin_d;
            let cos_az = (sc_lat_sat.s - cos_d * sc_lat_rec.s) / (sin_d * sc_lat_rec.c);
            let sin_lat_p = sc_lat_rec.s * sin_z - sc_lat_rec.c * cos_z * cos_az;
            let cos_lat_p = libm::sqrt(1.0 - sin_lat_p * sin_lat_p);
            let lat_p = libm::atan2(sin_lat_p, cos_lat_p);
            let lon_p = if cos_lat_p < THRESHOLD {
                0.0
            } else {
                let sin_lon_p = -sin_az * cos_z / cos_lat_p;
                let cos_lon_p = (sin_z - sc_lat_rec.s * sin_lat_p) / (sc_lat_rec.c * cos_lat_p);
                libm::atan2(sin_lon_p, cos_lon_p) + lon1
            };
            (lat_p, lon_p)
        };

        let sc_lat_p = SinCos::of(lat_p);

        // Azimuth of the satellite seen from the ray perigee.
        let (sin_az_p, cos_az_p) = if libm::fabs(libm::fabs(lat_p) - FRAC_PI_2) < THRESHOLD
            || libm::fabs(sin_z) < THRESHOLD
        {
            // Eq. 172, 173.
            (0.0, -libm::copysign(1.0, lat_p))
        } else {
            let sc_lon = SinCos::of(lon2 - lon_p);
            let psi = great_circle_angle(lat_p, sc_lat_p, sc_lat_sat, sc_lon);
            let sc_psi = SinCos::of(psi);
            // Eq. 174, 175.
            (
                sc_lat_sat.c * sc_lon.s / sc_psi.s,
                (sc_lat_sat.s - sc_lat_p.s * sc_psi.c) / (sc_lat_p.c * sc_psi.s),
            )
        };

        // Integration end point at the satellite (Eq. 177); `distance_at` gives Eq. 176's.
        let s2 = libm::sqrt(r2 * r2 - rp * rp);

        Self {
            rec_h,
            sat_h,
            s2,
            rp,
            lat_p,
            lon_p,
            sc_lat_p,
            sin_az_p,
            cos_az_p,
        }
    }

    /// Receiver altitude, m.
    pub(crate) fn rec_h(&self) -> f64 {
        self.rec_h
    }

    /// Satellite altitude, m.
    pub(crate) fn sat_h(&self) -> f64 {
        self.sat_h
    }

    /// Distance of the satellite from the ray perigee, m.
    pub(crate) fn s2(&self) -> f64 {
        self.s2
    }

    /// Distance from the ray perigee, m, of the point on the ray at `height` (Eq. 176, 177).
    pub(crate) fn distance_at(&self, height: f64) -> f64 {
        let r = RE + height;
        libm::sqrt(r * r - self.rp * self.rp)
    }

    /// Geodetic point at distance `s` (m) from the ray perigee (Eq. 178 to 187).
    pub(crate) fn point_at(&self, s: f64) -> GeodeticPoint {
        // Height (Eq. 178).
        let rp = self.rp;
        let height = libm::sqrt(s * s + rp * rp) - RE;

        if rp < ZENITH_RADIUS {
            return GeodeticPoint::new(self.lat_p, self.lon_p, height);
        }

        // Great-circle parameters (Eq. 179 to 181).
        let tan_ds = s / rp;
        let cos_ds = 1.0 / libm::sqrt(1.0 + tan_ds * tan_ds);
        let sin_ds = tan_ds * cos_ds;

        // Latitude (Eq. 182, 183).
        let sin_lat_s = self.sc_lat_p.s * cos_ds + self.sc_lat_p.c * sin_ds * self.cos_az_p;
        let cos_lat_s = libm::sqrt(1.0 - sin_lat_s * sin_lat_s);

        // Longitude (Eq. 184 to 187).
        let sin_lon_s = sin_ds * self.sin_az_p * self.sc_lat_p.c;
        let cos_lon_s = cos_ds - self.sc_lat_p.s * sin_lat_s;

        GeodeticPoint::new(
            libm::atan2(sin_lat_s, cos_lat_s),
            libm::atan2(sin_lon_s, cos_lon_s) + self.lon_p,
            height,
        )
    }
}

/// Great-circle angle from the ray perigee to the satellite (Orekit `greatCircleAngle`,
/// Eq. 168 to 171).
fn great_circle_angle(lat_p: f64, sc_lat_p: SinCos, sc_lat: SinCos, sc_lon: SinCos) -> f64 {
    if libm::fabs(libm::fabs(lat_p) - FRAC_PI_2) < THRESHOLD {
        libm::fabs(libm::asin(sc_lat.s) - lat_p)
    } else {
        let cos_phi = sc_lat_p.s * sc_lat.s + sc_lat_p.c * sc_lat.c * sc_lon.c;
        let sin_phi = libm::sqrt(1.0 - cos_phi * cos_phi);
        libm::atan2(sin_phi, cos_phi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The four geometries of `derivations/nequick_ray.mac`, (lat°, lon°, h m) receiver then
    /// satellite: low latitude, high latitude, across the anti-meridian, the far south looking east.
    const GEOMETRIES: [[f64; 6]; 4] = [
        [-3.0, 40.19, 0.0, 25.0, 97.5, 20_281_546.18],
        [82.49, -62.34, 78.1, 54.45, -118.47, 20_370_730.85],
        [-31.8, -179.99, 12.78, -14.31, 177.43, 20_100_697.9],
        [-77.84, 166.67, 98.0, -23.45, -99.29, 20_089_134.77],
    ];

    fn ray_of(g: &[f64; 6]) -> Ray {
        let at = |i: usize| GeodeticPoint::new(g[i].to_radians(), g[i + 1].to_radians(), g[i + 2]);
        Ray::new(at(0), at(3))
    }

    fn cartesian(p: GeodeticPoint) -> [f64; 3] {
        let r = RE + p.altitude;
        let (sf, cf) = libm::sincos(p.latitude);
        let (sl, cl) = libm::sincos(p.longitude);
        [r * cf * cl, r * cf * sl, r * sf]
    }

    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn norm(a: [f64; 3]) -> f64 {
        libm::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    }

    /// (G2) The ray perigee is the one ICD Eq. 153 to 177 define.
    ///
    /// The references are those equations evaluated as printed at 60 digits on the exact doubles
    /// below, by `derivations/nequick_ray.mac`, which also shows there that the perigee, `s1` and
    /// `s2` put the line on both end points to 1e-40 of the radius. The code takes the stabilised
    /// forms (the zenith angle from `atan2(r2 sin d, r2 cos d − r1)`, the perigee latitude from
    /// `sin ζ` and `cos ζ` in place of `cos δp` and `sin δp`), so agreement here is agreement of two
    /// routes to the same quantities, not a transcription read back.
    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the derivation's 40-digit values, kept whole"
    )]
    fn the_ray_perigee_is_the_one_the_icd_equations_define() {
        // (rp, phip, lamp, sin sigp, cos sigp, s1, s2) per geometry.
        const WANT: [[f64; 7]; 4] = [
            [
                6.170_064_480_660_431_341_273_279_198_868_267_716_331e6,
                -1.776_212_153_869_023_903_615_156_507_798_022_484_75e-1,
                4.812_441_411_175_583_263_558_011_282_486_940_561_571e-1,
                8.750_234_445_582_013_906_571_182_512_983_017_243_45e-1,
                4.840_805_423_413_548_617_443_974_799_335_434_210_842e-1,
                1.588_236_048_102_586_569_365_938_016_768_208_795_779e6,
                2.592_873_277_350_817_810_677_840_923_698_505_972_304e7,
            ],
            [
                4.168_811_189_316_068_485_381_703_874_123_476_920_564e6,
                7.599_673_258_894_928_461_694_520_204_634_992_577_32e-1,
                7.902_595_373_813_304_420_944_903_345_045_741_581_532e-1,
                -1.647_244_008_921_167_725_340_685_083_876_019_390_136e-1,
                9.863_396_330_629_390_251_972_894_787_167_731_273_07e-1,
                4.818_111_444_889_247_951_706_142_151_901_037_617_51e6,
                2.641_499_344_792_678_438_644_601_033_878_165_259_202e7,
            ],
            [
                2.495_294_437_846_357_470_276_921_707_567_480_027_508e6,
                -1.371_699_135_963_349_488_190_297_762_181_823_380_491e0,
                -7.330_830_925_639_697_478_314_111_652_153_778_184_054e-1,
                -6.182_146_267_313_907_551_514_665_696_585_694_357_621e-1,
                -7.860_093_353_741_844_707_311_336_116_107_819_433_174e-1,
                5.862_240_011_846_014_417_300_953_360_040_120_921_524e6,
                2.635_402_975_061_835_513_908_279_969_567_595_887_665e7,
            ],
            [
                6.305_141_427_352_208_834_102_246_823_161_159_427_586e6,
                -1.296_479_651_090_909_944_726_492_008_927_239_693_616e0,
                2.357_539_091_211_262_488_436_096_190_424_053_823_874e0,
                7.677_527_574_186_963_343_892_458_806_122_555_564_268e-1,
                -6.407_462_083_196_344_757_581_868_588_240_585_824_778e-1,
                9.157_673_208_249_738_855_545_060_443_674_152_568_315e5,
                2.569_814_210_641_613_445_375_530_581_557_136_333_161e7,
            ],
        ];
        for (g, w) in GEOMETRIES.iter().zip(&WANT) {
            let ray = ray_of(g);
            let got = [
                ray.rp,
                ray.lat_p,
                ray.lon_p,
                ray.sin_az_p,
                ray.cos_az_p,
                ray.distance_at(ray.rec_h()),
                ray.s2(),
            ];
            for (k, (&x, &y)) in got.iter().zip(w).enumerate() {
                // Lengths relative to themselves, angles and their sines absolutely.
                let scale = if matches!(k, 0 | 5 | 6) { y.abs() } else { 1.0 };
                assert!(
                    (x - y).abs() < 1.0e-12 * scale,
                    "geometry {g:?}, quantity {k}: {x} vs {y}"
                );
            }
        }
    }

    /// (G1) `point_at` traces a straight line, through the receiver at `s1` and the satellite at
    /// `s2`, at arc length equal to the change in `s`.
    ///
    /// `derivations/nequick_ray.mac` shows symbolically that Eq. 178 to 187 map `s` to
    /// `rp·P + s·A`, with P the perigee and A the unit direction there. This checks the shipped map
    /// against that consequence in Cartesian coordinates, which the code never forms.
    #[test]
    fn the_ray_is_a_straight_line_through_both_end_points() {
        for g in &GEOMETRIES {
            let ray = ray_of(g);
            let rec = GeodeticPoint::new(g[0].to_radians(), g[1].to_radians(), g[2]);
            let sat = GeodeticPoint::new(g[3].to_radians(), g[4].to_radians(), g[5]);
            let s1 = ray.distance_at(ray.rec_h());
            let s2 = ray.s2();
            let x1 = cartesian(ray.point_at(s1));
            let x2 = cartesian(ray.point_at(s2));
            assert!(norm(sub(x1, cartesian(rec))) < 1.0e-6, "receiver, {g:?}");
            assert!(norm(sub(x2, cartesian(sat))) < 1.0e-6, "satellite, {g:?}");
            let chord = sub(x2, x1);
            for f in [0.1, 0.35, 0.5, 0.8] {
                let s = s1 + f * (s2 - s1);
                let d = sub(cartesian(ray.point_at(s)), x1);
                // Distance along the line equals the change in s ...
                assert!(
                    (norm(d) - (s - s1)).abs() < 1.0e-6,
                    "arc length at {f}, {g:?}: {} vs {}",
                    norm(d),
                    s - s1
                );
                // ... and the point is on the chord: d = f·chord.
                let off = sub(d, [f * chord[0], f * chord[1], f * chord[2]]);
                assert!(
                    norm(off) < 1.0e-6,
                    "off the line by {} m at {f}, {g:?}",
                    norm(off)
                );
            }
        }
    }

    /// Great-circle angle from `(lat_p, 0)` to `(lat, lon)` by the haversine formula — a different
    /// route to the same arc than the law of cosines the production code takes.
    fn haversine(lat_p: f64, lat: f64, lon: f64) -> f64 {
        let dlat = libm::sin(0.5 * (lat - lat_p));
        let dlon = libm::sin(0.5 * lon);
        let a = dlat * dlat + libm::cos(lat_p) * libm::cos(lat) * dlon * dlon;
        2.0 * libm::asin(libm::sqrt(a).min(1.0))
    }

    #[test]
    fn the_polar_branch_of_the_great_circle_angle_is_the_co_latitude() {
        // Within `THRESHOLD` of a pole the law of cosines has nothing left to say: its longitude
        // term carries a factor `cos(lat_p)` that has already vanished, and the arc is just the
        // satellite's co-latitude. Every STEC fixture in the suite puts the ray perigee well away
        // from either pole, so this branch — and the subtraction inside it — never runs at all.
        let lat_p = FRAC_PI_2 - 1.0e-11;
        for lat in [-0.9_f64, 0.0, 0.4, 1.5] {
            for lon in [0.0_f64, 1.0, PI] {
                let got =
                    great_circle_angle(lat_p, SinCos::of(lat_p), SinCos::of(lat), SinCos::of(lon));
                assert!(
                    (got - (lat_p - lat)).abs() < 1e-13,
                    "polar arc at lat {lat}, lon {lon}: {got} vs {}",
                    lat_p - lat
                );
                // The stabilised branch is not merely self-consistent: an independent spherical
                // formula puts the arc in the same place, to the 1e-11 the vanished term is worth.
                assert!(
                    (got - haversine(lat_p, lat, lon)).abs() < 1e-9,
                    "haversine disagrees at lat {lat}, lon {lon}: {got} vs {}",
                    haversine(lat_p, lat, lon)
                );
            }
        }
    }
}
