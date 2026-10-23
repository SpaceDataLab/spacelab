//! **Centred-dipole magnetic coordinates** — the first degree of [`crate::igrf`], and the magnetic
//! latitude / magnetic local time that upper-atmosphere models are organized in.
//!
//! The degree-1 part of the geomagnetic potential is a point dipole at Earth's centre with moment
//! direction `m̂ ∝ (g₁¹, h₁¹, g₁⁰)` in Earth-fixed Cartesian components (`x` toward the Greenwich
//! meridian on the equator, `z` toward the geographic north pole). Because `g₁⁰` is negative, `m̂`
//! points *south*: the north geomagnetic pole is the direction `−m̂`, which is what `Self::axis`
//! returns and what every magnetic latitude here is measured from.
//!
//! Why it earns a module: the aurora, the ion-drag wind jets and the storm-time response of the
//! thermosphere are organized about the magnetic pole, which sits some 11° from the geographic one,
//! and in magnetic local time rather than solar local time. A model expanded only in geographic
//! coordinates cannot represent a feature that is fixed in magnetic ones — it can only smear it
//! around a circle of latitude.
//!
//! **This is the centred dipole, not quasi-dipole or apex coordinates.** The centred dipole ignores
//! the non-dipole field, which displaces the true magnetic equator by up to a few degrees (most of
//! it over the South Atlantic). Quasi-dipole coordinates, which HWM14's disturbance model uses, trace
//! real field lines and need the full field plus an apex solver. The error is quantified in this
//! module's own tests rather than asserted away.

// The geomagnetism symbols (g, h, m, n̂, θ, φ) are one/two letters by universal convention.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use libm::{asin, atan2, cos, floor, sin, sqrt};

use crate::igrf14_coeffs::{G_2025, G_SV, H_2025, H_SV};

/// Reference epoch of the IGRF-14 coefficients this crate carries.
pub const REFERENCE_EPOCH: f64 = 2025.0;

/// The centred dipole at one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CentredDipole {
    /// Unit vector toward the north geomagnetic pole, Earth-fixed Cartesian.
    axis: [f64; 3],
    /// Dipole moment magnitude `√(g₁⁰² + g₁¹² + h₁¹²)`, nanotesla.
    moment_nt: f64,
}

impl CentredDipole {
    /// The dipole at decimal `year`, from the IGRF-14 epoch-2025.0 coefficients and their
    /// 2025–2030 secular variation.
    ///
    /// **Outside 2025–2030 this extrapolates a straight line.** The pole moves of order 0.05° a
    /// year, so a decade either side costs about half a degree of pole position — acceptable where
    /// the feature being located is an auroral oval several degrees wide, and not acceptable if what
    /// is wanted is the field vector, which is what [`crate::igrf`] is for and which degrades much
    /// faster.
    #[must_use]
    pub fn at_year(year: f64) -> Self {
        let dt = year - REFERENCE_EPOCH;
        let g10 = G_2025[1][0] + dt * G_SV[1][0];
        let g11 = G_2025[1][1] + dt * G_SV[1][1];
        let h11 = H_2025[1][1] + dt * H_SV[1][1];
        let moment_nt = sqrt(g10 * g10 + g11 * g11 + h11 * h11);
        // −m̂: the geomagnetic north pole is where the moment vector points *away* from.
        let axis = [-g11 / moment_nt, -h11 / moment_nt, -g10 / moment_nt];
        Self { axis, moment_nt }
    }

    /// Unit vector toward the north geomagnetic pole, Earth-fixed Cartesian.
    #[must_use]
    pub const fn axis(&self) -> [f64; 3] {
        self.axis
    }

    /// Dipole moment magnitude, nanotesla.
    #[must_use]
    pub const fn moment_nt(&self) -> f64 {
        self.moment_nt
    }

    /// Geographic `(latitude, longitude)` of the north geomagnetic pole, radians.
    #[must_use]
    pub fn pole(&self) -> (f64, f64) {
        (asin(self.axis[2]), atan2(self.axis[1], self.axis[0]))
    }

    /// Magnetic latitude of a geographic `(latitude, longitude)` in radians: the angle between the
    /// position and the dipole equator, positive toward the north geomagnetic pole.
    #[must_use]
    pub fn magnetic_latitude(&self, latitude: f64, longitude: f64) -> f64 {
        let r = unit(latitude, longitude);
        asin(dot(&r, &self.axis).clamp(-1.0, 1.0))
    }

    /// Magnetic longitude of a geographic `(latitude, longitude)`, radians in `(−π, π]`.
    ///
    /// The prime meridian is the one containing the geographic poles — the conventional centred-dipole
    /// choice. Only differences of this quantity have physical meaning here, and
    /// [`Self::magnetic_local_time`] takes one.
    #[must_use]
    pub fn magnetic_longitude(&self, latitude: f64, longitude: f64) -> f64 {
        let (x, y) = self.frame();
        let r = unit(latitude, longitude);
        atan2(dot(&r, &y), dot(&r, &x))
    }

    /// Magnetic local time, hours in `[0, 24)`, of a geographic `(latitude, longitude)` given the
    /// geographic `(latitude, longitude)` of the sub-solar point.
    ///
    /// Noon is the magnetic meridian of the Sun, midnight its antipode — the same construction as
    /// solar local time with the magnetic pole in place of the geographic one. A wind or an aurora
    /// driven by the magnetosphere is organized in this coordinate, not in solar local time, and at
    /// high latitude the two differ by up to the 11° pole offset, which is nearly an hour.
    #[must_use]
    pub fn magnetic_local_time(
        &self,
        latitude: f64,
        longitude: f64,
        sub_solar_latitude: f64,
        sub_solar_longitude: f64,
    ) -> f64 {
        let delta = self.magnetic_longitude(latitude, longitude)
            - self.magnetic_longitude(sub_solar_latitude, sub_solar_longitude);
        let hours = 12.0 + delta * (12.0 / core::f64::consts::PI);
        hours - 24.0 * floor(hours / 24.0)
    }

    /// `(cos, sin)` of the angle from geographic north to magnetic north at a geographic point,
    /// measured positive toward geographic east.
    ///
    /// This is the rotation that carries a horizontal vector's magnetic components to geographic
    /// ones — a field expanded in magnetic coordinates produces `(u_m, v_m)` along magnetic east
    /// and north, and
    ///
    /// ```text
    /// u =  c·u_m + s·v_m
    /// v = −s·u_m + c·v_m
    /// ```
    ///
    /// puts them back on the geographic axes. Both frames are right-handed about the same local
    /// vertical, so one angle describes the whole rotation.
    ///
    /// Returns `(1, 0)` — the identity — where either frame is undefined: on the geographic axis,
    /// where east does not exist, and on the magnetic axis, where magnetic east does not.
    #[must_use]
    pub fn meridian_rotation(&self, latitude: f64, longitude: f64) -> (f64, f64) {
        let r = unit(latitude, longitude);
        let (Some(east), Some(east_m)) = (
            normalized(cross(&[0.0, 0.0, 1.0], &r)),
            normalized(cross(&self.axis, &r)),
        ) else {
            return (1.0, 0.0);
        };
        // r̂ × ê is already a unit vector: r̂ ⟂ ê and both are unit.
        let north = cross(&r, &east);
        let north_m = cross(&r, &east_m);
        (dot(&north_m, &north), dot(&north_m, &east))
    }

    /// The `(x̂, ŷ)` of the dipole frame whose `ẑ` is [`Self::axis`], with `ŷ = ẑ_geo × axis`
    /// normalized so the geographic poles sit on the prime meridian.
    fn frame(&self) -> ([f64; 3], [f64; 3]) {
        // ẑ_geo × axis = (−a_y, a_x, 0); its norm is the sine of the dipole tilt, ~0.16, so it is
        // never near zero for any realistic Earth field. A dipole exactly along the rotation axis
        // has no distinguished meridian at all, and that is the case this returns the geographic
        // frame for rather than dividing by zero.
        let Some(y) = normalized(cross(&[0.0, 0.0, 1.0], &self.axis)) else {
            return ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        };
        (cross(&y, &self.axis), y)
    }
}

fn unit(latitude: f64, longitude: f64) -> [f64; 3] {
    let (c, s) = (cos(latitude), sin(latitude));
    [c * cos(longitude), c * sin(longitude), s]
}

fn normalized(v: [f64; 3]) -> Option<[f64; 3]> {
    let n = sqrt(dot(&v, &v));
    (n > 0.0).then(|| [v[0] / n, v[1] / n, v[2] / n])
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;
    use crate::igrf::field_ecef;

    const DEG: f64 = core::f64::consts::PI / 180.0;

    /// Rodrigues rotation of a unit vector about a unit axis, right-handed.
    fn rotate(v: &[f64; 3], axis: &[f64; 3], theta: f64) -> [f64; 3] {
        let (c, s) = (cos(theta), sin(theta));
        let perpendicular = cross(axis, v);
        let along = dot(axis, v) * (1.0 - c);
        [
            v[0] * c + perpendicular[0] * s + axis[0] * along,
            v[1] * c + perpendicular[1] * s + axis[1] * along,
            v[2] * c + perpendicular[2] * s + axis[2] * along,
        ]
    }

    fn geographic(v: &[f64; 3]) -> (f64, f64) {
        (asin(v[2].clamp(-1.0, 1.0)), atan2(v[1], v[0]))
    }

    #[test]
    fn the_pole_is_where_the_igrf_field_dips_into_the_ground() {
        // The strongest available check that the axis is built with the right signs: at the pole the
        // real IGRF field — all thirteen degrees, not the dipole this module extracts — must point
        // into the ground, and out of it at the antipode. A flipped sign in any of the three
        // coefficients moves the pole to the other hemisphere or the other side of the globe, and
        // the radial sign goes the wrong way.
        //
        // How vertical it is, is the interesting number, and it is wildly asymmetric. Measured here
        // and cross-checked against `ppigrf` at the identical points: 5.8% of the field is
        // horizontal at the north geomagnetic pole, 24.6% at the south — inclination −86.7° against
        // +75.7°. That is not an implementation defect but the southern field being far less
        // dipolar: the south *dip* pole sits some 16° from the antipode of the north one, while the
        // north dip pole is within a few degrees. Bounds below are the measured values with slack.
        let d = CentredDipole::at_year(2025.0);
        let (lat, lon) = d.pole();
        for (sign, name, bound) in [(1.0, "north", 0.10), (-1.0, "south", 0.30)] {
            let (la, lo) = (
                sign * lat,
                lon + if sign > 0.0 {
                    0.0
                } else {
                    core::f64::consts::PI
                },
            );
            let r = unit(la, lo);
            let pos = [r[0] * 6.4e6, r[1] * 6.4e6, r[2] * 6.4e6];
            let b = field_ecef(pos, 2025.0);
            let radial = dot(&b, &r);
            let mag = sqrt(dot(&b, &b));
            let horizontal = sqrt((mag * mag - radial * radial).max(0.0)) / mag;
            // Down is −r̂ in the northern magnetic hemisphere.
            assert!(
                sign * radial < 0.0,
                "{name} geomagnetic pole: field is not into the ground, B·r̂ = {radial:e}"
            );
            assert!(
                horizontal < bound,
                "{name} geomagnetic pole: {:.1}% of the field is horizontal",
                100.0 * horizontal
            );
        }
    }

    #[test]
    fn the_pole_sits_where_the_published_geomagnetic_pole_sits() {
        let (lat, lon) = CentredDipole::at_year(2025.0).pole();
        // IGRF-14 epoch 2025.0: the geomagnetic north pole is in the Canadian Arctic, some 9° from
        // the geographic pole and near 73° west. This is a bound, not a restatement of the inputs:
        // it fails for any of the six sign/axis permutations of (g₁¹, h₁¹, g₁⁰).
        assert!(
            (80.0 * DEG..81.5 * DEG).contains(&lat),
            "pole latitude {:.3}°",
            lat / DEG
        );
        assert!(
            (-74.0 * DEG..-71.0 * DEG).contains(&lon),
            "pole longitude {:.3}°",
            lon / DEG
        );
        // The tilt is the 11° every textbook quotes, and the moment is ~30 µT.
        let tilt = core::f64::consts::FRAC_PI_2 - lat;
        assert!(
            (8.0 * DEG..11.0 * DEG).contains(&tilt),
            "tilt {:.2}°",
            tilt / DEG
        );
        assert!(
            (29_000.0..30_500.0).contains(&CentredDipole::at_year(2025.0).moment_nt()),
            "moment"
        );
    }

    #[test]
    fn magnetic_latitude_is_ninety_at_the_pole_and_zero_on_the_dipole_equator() {
        let d = CentredDipole::at_year(2025.0);
        let (lat, lon) = d.pole();
        assert!((d.magnetic_latitude(lat, lon) - core::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!(
            (d.magnetic_latitude(-lat, lon + core::f64::consts::PI) + core::f64::consts::FRAC_PI_2)
                .abs()
                < 1e-12
        );
        // Points perpendicular to the axis are on the magnetic equator, whatever their geographic
        // latitude — which is the whole point of the coordinate.
        for k in 0..12 {
            let lambda = f64::from(k) * 30.0 * DEG;
            // Solve for the geographic latitude at this longitude where r̂ ⟂ axis.
            let a = d.axis();
            let horizontal = a[0] * cos(lambda) + a[1] * sin(lambda);
            let phi = atan2(-horizontal, a[2]);
            let m = d.magnetic_latitude(phi, lambda);
            assert!(m.abs() < 1e-12, "magnetic latitude {m:e} at λ={lambda}");
        }
    }

    #[test]
    fn the_centred_dipole_equator_is_typically_within_a_few_degrees_of_the_real_one() {
        // The honest size of the approximation, measured rather than asserted away: walk the dipole
        // equator and record how far the real IGRF field's inclination is from horizontal.
        //
        // Measured, and cross-checked point-for-point against `ppigrf`: the median is 3.1° and the
        // worst is 39.3°, at 345°E — the equatorial Atlantic, the edge of the South Atlantic
        // Anomaly, which is where the non-dipole field is largest. So "within a few degrees" is
        // true of the typical meridian and false of the worst one by an order of magnitude, and a
        // model organized in these coordinates carries that error in the Atlantic sector. Quasi-
        // dipole coordinates are the fix, and cost a field-line trace per evaluation.
        let d = CentredDipole::at_year(2025.0);
        let a = d.axis();
        let mut inclinations = Vec::new();
        for k in 0..72 {
            let lambda = f64::from(k) * 5.0 * DEG;
            let horizontal = a[0] * cos(lambda) + a[1] * sin(lambda);
            let phi = atan2(-horizontal, a[2]);
            let r = unit(phi, lambda);
            let pos = [r[0] * 6.371e6, r[1] * 6.371e6, r[2] * 6.371e6];
            let b = field_ecef(pos, 2025.0);
            let inclination = asin((dot(&b, &r) / sqrt(dot(&b, &b))).clamp(-1.0, 1.0)).abs();
            inclinations.push((inclination, lambda / DEG));
        }
        let (worst, worst_lon) = inclinations
            .iter()
            .copied()
            .fold(
                (0.0_f64, 0.0_f64),
                |acc, x| if x.0 > acc.0 { x } else { acc },
            );
        let mut sorted: Vec<f64> = inclinations.iter().map(|x| x.0).collect();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        println!(
            "|inclination| on the centred-dipole equator: median {:.2}°, worst {:.2}° at {worst_lon:.0}°E",
            median / DEG,
            worst / DEG
        );
        assert!(
            median < 6.0 * DEG,
            "the typical meridian should be near-horizontal, median {:.2}°",
            median / DEG
        );
        assert!(
            worst < 45.0 * DEG,
            "worst |inclination| {:.2}° at {worst_lon:.0}°E",
            worst / DEG
        );
        assert!(
            (330.0..=360.0).contains(&worst_lon),
            "the worst meridian should be the Atlantic one, got {worst_lon:.0}°E"
        );
    }

    #[test]
    fn the_meridian_rotation_is_the_rotation_it_claims_to_be() {
        // Two independent checks over the globe. First, `(c, s)` is a rotation: c² + s² = 1 wherever
        // both frames exist. Second — the one that catches a transposed or negated angle — rotating
        // the magnetic unit vectors by it must reproduce their geographic components, computed here
        // from the vectors directly instead of from the angle.
        let d = CentredDipole::at_year(2025.0);
        let mut worst: f64 = 0.0;
        for i in 1..36 {
            for k in 0..36 {
                let (lat, lon) = ((f64::from(i) * 5.0 - 90.0) * DEG, f64::from(k) * 10.0 * DEG);
                let (c, s) = d.meridian_rotation(lat, lon);
                assert!(
                    (c * c + s * s - 1.0).abs() < 1e-12,
                    "not a rotation at {i} {k}"
                );
                let r = unit(lat, lon);
                let east = normalized(cross(&[0.0, 0.0, 1.0], &r)).unwrap();
                let north = cross(&r, &east);
                let east_m = normalized(cross(&d.axis(), &r)).unwrap();
                let north_m = cross(&r, &east_m);
                // A unit magnetic wind (u_m, v_m) and the two ways of resolving it geographically.
                for (um, vm) in [(1.0, 0.0), (0.0, 1.0), (0.6, -0.8)] {
                    let w = [
                        um * east_m[0] + vm * north_m[0],
                        um * east_m[1] + vm * north_m[1],
                        um * east_m[2] + vm * north_m[2],
                    ];
                    worst = worst
                        .max((dot(&w, &east) - (c * um + s * vm)).abs())
                        .max((dot(&w, &north) - (c * vm - s * um)).abs());
                }
            }
        }
        assert!(worst < 1e-12, "worst rotation mismatch {worst:e}");
    }

    #[test]
    fn the_meridian_rotation_is_the_identity_only_on_the_magnetic_prime_meridian() {
        // The magnetic prime meridian is the one through the geographic poles, so magnetic north and
        // geographic north are parallel there and nowhere else on that latitude circle. This pins the
        // rotation's zero, which the round-trip test above cannot: it is satisfied by any consistent
        // pair of frames.
        let d = CentredDipole::at_year(2025.0);
        let (_, pole_lon) = d.pole();
        let on = d.meridian_rotation(20.0 * DEG, pole_lon);
        assert!(
            on.1.abs() < 1e-12,
            "sin {:e} on the magnetic meridian",
            on.1
        );
        let off = d.meridian_rotation(20.0 * DEG, pole_lon + core::f64::consts::FRAC_PI_2);
        assert!(
            off.1.abs() > 0.05,
            "the rotation should be a few degrees a quarter-turn away, sin {:e}",
            off.1
        );
    }

    #[test]
    fn magnetic_local_time_is_noon_under_the_sun_and_midnight_opposite() {
        let d = CentredDipole::at_year(2025.0);
        let (sun_lat, sun_lon) = (0.2, -0.7);
        assert!((d.magnetic_local_time(sun_lat, sun_lon, sun_lat, sun_lon) - 12.0).abs() < 1e-9);
        let anti =
            d.magnetic_local_time(-sun_lat, sun_lon + core::f64::consts::PI, sun_lat, sun_lon);
        assert!(
            anti < 1e-9 || (anti - 24.0).abs() < 1e-9,
            "antipode MLT {anti}"
        );
        // It advances an hour per 15° of magnetic longitude and stays inside the day.
        for k in 0..24 {
            let lon = -core::f64::consts::PI + f64::from(k) * 15.0 * DEG;
            let mlt = d.magnetic_local_time(0.1, lon, sun_lat, sun_lon);
            assert!((0.0..24.0).contains(&mlt), "MLT {mlt} out of range");
        }
    }

    #[test]
    fn magnetic_local_time_differs_from_solar_local_time_by_up_to_an_hour() {
        // The number that justifies the module. If MLT were within minutes of solar local time
        // everywhere, a geographic expansion would already capture the auroral pattern.
        let d = CentredDipole::at_year(2025.0);
        let (sun_lat, sun_lon) = (0.0, 0.0);
        let mut worst: f64 = 0.0;
        for j in 0..36 {
            for k in 0..18 {
                let lon = -core::f64::consts::PI + f64::from(j) * 10.0 * DEG;
                let lat = -80.0 * DEG + f64::from(k) * 10.0 * DEG;
                let mlt = d.magnetic_local_time(lat, lon, sun_lat, sun_lon);
                let slt = 12.0 + (lon - sun_lon) * (12.0 / core::f64::consts::PI);
                let slt = slt - 24.0 * floor(slt / 24.0);
                let mut diff = (mlt - slt).abs();
                if diff > 12.0 {
                    diff = 24.0 - diff;
                }
                worst = worst.max(diff);
            }
        }
        println!("worst |MLT − solar LT| over the globe: {worst:.2} h");
        assert!(
            worst > 0.5,
            "the two clocks should part company by at least half an hour somewhere, got {worst:.2} h"
        );
    }

    #[test]
    fn the_secular_variation_reproduces_the_published_model_away_from_the_epoch() {
        // At the reference epoch the drift term is multiplied by zero, so every other test in this
        // module is blind to *how* it is applied — a sign flip, a division and an addition all agree
        // at 2025.0. This pins it three and a half years out against `ppigrf`, which carries its own
        // copy of the IGRF-14 tables and interpolates linearly between their 2025.0 and 2030.0
        // columns rather than multiplying a secular-variation column the way this module does.
        //
        // Both are exact linear functions of the same published constants, so the tolerances are set
        // at floating-point rounding, not at model error. For scale: the pole moves 0.207° over the
        // five-year span, so any mutation of this arithmetic misses by a degree or more.
        let d = CentredDipole::at_year(2028.5);
        let (lat, lon) = d.pole();
        assert!(
            (lat - 1.412_537_568_442_043).abs() < 1e-9,
            "latitude {lat:.15} rad"
        );
        assert!(
            (lon + 1.272_330_406_576_405_6).abs() < 1e-9,
            "longitude {lon:.15} rad"
        );
        assert!(
            (d.moment_nt() - 29_676.764_647_827).abs() < 1e-6,
            "moment {:.9} nT",
            d.moment_nt()
        );
    }

    #[test]
    fn the_magnetic_prime_meridian_runs_through_the_geographic_poles() {
        // ŷ is ẑ_geo × axis, so both axes lie in the x̂–ẑ plane and the geographic poles sit on the
        // prime meridian: the north one beyond the magnetic pole at ±π, the south one at zero. This
        // is what fixes the *origin* of magnetic longitude, which the rotation test below cannot —
        // that one is satisfied by any origin at all.
        let d = CentredDipole::at_year(2025.0);
        let north = d.magnetic_longitude(core::f64::consts::FRAC_PI_2, 0.0);
        let south = d.magnetic_longitude(-core::f64::consts::FRAC_PI_2, 0.0);
        assert!(
            (north.abs() - core::f64::consts::PI).abs() < 1e-9,
            "geographic north pole at magnetic longitude {north}"
        );
        assert!(
            south.abs() < 1e-9,
            "geographic south pole at magnetic longitude {south}"
        );
    }

    #[test]
    fn magnetic_longitude_advances_one_for_one_around_the_dipole_axis() {
        // Rotating a point about the magnetic axis by θ must advance its magnetic longitude by
        // exactly θ, positive eastward. That is the entire content of the frame being orthonormal
        // and right-handed, and it is what a frame that is merely *some* pair of vectors fails.
        let d = CentredDipole::at_year(2025.0);
        let a = d.axis();
        let seed = normalized(cross(&a, &[0.0, 0.0, 1.0])).unwrap();
        // 30° off the axis: far from the degenerate point where longitude is undefined.
        let base = [
            cos(30.0 * DEG) * a[0] + sin(30.0 * DEG) * seed[0],
            cos(30.0 * DEG) * a[1] + sin(30.0 * DEG) * seed[1],
            cos(30.0 * DEG) * a[2] + sin(30.0 * DEG) * seed[2],
        ];
        let mut previous = f64::NAN;
        for k in 0..12 {
            let (lat, lon) = geographic(&rotate(&base, &a, f64::from(k) * 30.0 * DEG));
            let current = d.magnetic_longitude(lat, lon);
            if k > 0 {
                let raw = current - previous;
                let step = raw
                    - core::f64::consts::TAU
                        * floor((raw + core::f64::consts::PI) / core::f64::consts::TAU);
                assert!(
                    (step - 30.0 * DEG).abs() < 1e-9,
                    "step {:.6}° at k={k}",
                    step / DEG
                );
            }
            previous = current;
        }
    }

    #[test]
    fn an_axial_dipole_falls_back_to_the_geographic_frame() {
        // Unreachable through `at_year` — the real Earth is tilted 9° — but this is the branch that
        // keeps a degenerate dipole from dividing by zero, and it should degrade to the geographic
        // frame rather than to NaN. With no tilt there is no distinguished meridian, so magnetic
        // longitude is geographic longitude.
        let d = CentredDipole {
            axis: [0.0, 0.0, 1.0],
            moment_nt: 30_000.0,
        };
        for k in 0..8 {
            let lon = (f64::from(k) - 3.5) * 45.0 * DEG;
            let magnetic = d.magnetic_longitude(0.3, lon);
            assert!(
                (magnetic - lon).abs() < 1e-12,
                "magnetic longitude {magnetic} at geographic {lon}"
            );
        }
    }

    #[test]
    fn normalizing_the_zero_vector_is_none_rather_than_nan() {
        // The frame fallback and the meridian-rotation identity both hang off this returning `None`
        // instead of a vector of NaN, and neither reaches it with real inputs.
        assert_eq!(normalized([0.0, 0.0, 0.0]), None);
        let v = normalized([3.0, 0.0, 4.0]).unwrap();
        assert!(
            (v[0] - 0.6).abs() < 1e-15 && (v[2] - 0.8).abs() < 1e-15,
            "{v:?}"
        );
    }

    #[test]
    fn magnetic_local_time_runs_eastward_from_the_sun() {
        // Dusk is east of the sub-solar meridian, so a point a quarter turn east of the Sun's
        // magnetic meridian reads 18 h and not 6 h. Nothing else pins this sign: under the sun the
        // hour angle is zero, and at the antipode 0 h and 24 h are the same instant, so a flipped
        // hour angle survives both of those checks.
        let d = CentredDipole::at_year(2025.0);
        let (sun_lat, sun_lon) = (0.2, -0.7);
        let sun = unit(sun_lat, sun_lon);
        for (turn, expected) in [(0.25, 18.0), (-0.25, 6.0), (0.5, 0.0)] {
            let (lat, lon) = geographic(&rotate(&sun, &d.axis(), turn * core::f64::consts::TAU));
            let mlt = d.magnetic_local_time(lat, lon, sun_lat, sun_lon);
            let mut diff = (mlt - expected).abs();
            if diff > 12.0 {
                diff = 24.0 - diff;
            }
            assert!(
                diff < 1e-9,
                "expected {expected} h a turn of {turn}, got {mlt} h"
            );
        }
    }

    #[test]
    fn the_pole_drifts_with_the_secular_variation() {
        let (lat_2025, lon_2025) = CentredDipole::at_year(2025.0).pole();
        let (lat_2030, lon_2030) = CentredDipole::at_year(2030.0).pole();
        let moved = sqrt(
            (lat_2030 - lat_2025) * (lat_2030 - lat_2025)
                + (lon_2030 - lon_2025) * (lon_2030 - lon_2025) * cos(lat_2025) * cos(lat_2025),
        ) / DEG;
        println!("geomagnetic pole moves {moved:.3}° from 2025 to 2030");
        assert!(
            (0.01..1.0).contains(&moved),
            "the pole should drift a fraction of a degree over five years, got {moved:.3}°"
        );
    }
}
