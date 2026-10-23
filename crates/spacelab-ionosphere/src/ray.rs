//! Receiver-to-satellite ray geometry.
//!
//! Ported from Orekit's `Ray`. The ray-perigee frame turns the slant path into an arc
//! parameterised by distance `s` from the perigee, and `Ray::point_at` maps a distance
//! back to the geodetic point where the electron density is evaluated.

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

        // Integration end points (Eq. 176, 177).
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
