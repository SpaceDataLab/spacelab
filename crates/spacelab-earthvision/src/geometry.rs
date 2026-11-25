//! Earth-observation viewing geometry — the sun/sensor angles atmospheric correction and BRDF need.
//!
//! Inputs are unit direction vectors in one consistent local frame: `to_sun` and `to_sensor` point
//! from the ground point toward the sun and the satellite, `zenith` is the local outward normal (up)
//! and `north` a horizontal reference. The caller builds these from an ephemeris; keeping the
//! geometry frame-agnostic keeps it dependency-light and exactly testable.
//!
//! Gate: `derivations/observation_geometry.mac` shows that the code's dot products and `atan2` read
//! back the zenith and azimuth of a direction in the right-handed ENU triad `E = N × U`, that the
//! scattering cosine is the spherical law of cosines in the azimuth difference, and prints the
//! Kasten–Young air mass at 40 digits; `derivations/observation_geometry.gp` recomputes those at 60.
//! The module tests `the_angles_are_the_ones_the_enu_directions_were_built_from` (G1) and
//! `the_air_mass_is_kasten_young_s_formula` (G2) hold the shipped code to them.

use spacelab_core::{Angle, Vec3};

const PI: f64 = core::f64::consts::PI;
const TAU: f64 = core::f64::consts::TAU;

/// The sun/sensor geometry at a ground point. All angles radians.
#[derive(Clone, Copy, Debug)]
pub struct ObservationGeometry {
    /// Solar zenith angle `θs` — angle of the sun from the local vertical.
    pub solar_zenith: Angle,
    /// View (sensor) zenith angle `θv`.
    pub view_zenith: Angle,
    /// Relative azimuth `φv − φs`, wrapped to `[−π, π)`.
    pub relative_azimuth: Angle,
    /// Scattering angle between the incident (sun → ground) and scattered (ground → sensor) rays:
    /// `0` is forward scatter, `π` is exact backscatter.
    pub scattering_angle: Angle,
    /// Relative optical air mass at the solar zenith (Kasten–Young).
    pub air_mass: f64,
}

/// Wrap an angle to `[−π, π)` — the half-open interval `fmod` produces, so the map is a bijection of
/// `ℝ/2πℤ` and `wrap_pi(π) = −π`.
fn wrap_pi(a: f64) -> f64 {
    let x = libm::fmod(a + PI, TAU);
    let x = if x < 0.0 { x + TAU } else { x };
    x - PI
}

/// Relative optical air mass at zenith `angle` (Kasten & Young 1989): `1 / (cos θ + 0.50572
/// (96.07995° − θ)^−1.6364)`. Finite even at the horizon, unlike the plane-parallel `1/cos θ`, and
/// accurate to a fraction of a percent through `θ = 90°`.
#[must_use]
pub fn air_mass(angle: Angle) -> f64 {
    let z = angle.radians();
    let z_deg = z * (180.0 / PI);
    1.0 / (libm::cos(z) + 0.50572 * libm::pow(96.079_95 - z_deg, -1.6364))
}

/// The angle between `a` and `b`, `atan2(|a × b|, a·b)`. `acos` of the dot product loses `√(2u)` of
/// angle next to `0` and `π`, where the cosine is flat: the nadir view, the sun overhead, forward
/// scatter and the backscatter hot spot (`derivations/angle_from_vectors.mac`).
fn angle_between(a: Vec3, b: Vec3) -> f64 {
    libm::atan2(a.cross(b).norm(), a.dot(b))
}

/// Compute the [`ObservationGeometry`] from the sun and sensor directions at a ground point.
#[must_use]
pub fn observation_geometry(
    to_sun: Vec3,
    to_sensor: Vec3,
    zenith: Vec3,
    north: Vec3,
) -> ObservationGeometry {
    let east = north.cross(zenith); // right-handed ENU: E = N × U
    let solar_zenith = angle_between(to_sun, zenith);
    let view_zenith = angle_between(to_sensor, zenith);
    // Azimuths measured from north toward east; the vertical component cancels in the ratio.
    let sun_az = libm::atan2(to_sun.dot(east), to_sun.dot(north));
    let view_az = libm::atan2(to_sensor.dot(east), to_sensor.dot(north));
    // The scattering angle is between the photon's incident propagation (−to_sun) and its scattered
    // propagation (to_sensor).
    let scattering = angle_between(-to_sun, to_sensor);
    ObservationGeometry {
        solar_zenith: Angle::from_radians(solar_zenith),
        view_zenith: Angle::from_radians(view_zenith),
        relative_azimuth: Angle::from_radians(wrap_pi(view_az - sun_az)),
        scattering_angle: Angle::from_radians(scattering),
        air_mass: air_mass(Angle::from_radians(solar_zenith)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,
        clippy::excessive_precision,
        clippy::unreadable_literal
    )]
    use super::*;

    const DEG: f64 = PI / 180.0;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn sun_and_sensor_at_zenith_give_zero_angles() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        let g = observation_geometry(up, up, up, north);
        assert!(approx(g.solar_zenith.radians(), 0.0, 1e-12));
        assert!(approx(g.view_zenith.radians(), 0.0, 1e-12));
        assert!(approx(g.air_mass, 1.0, 1e-3)); // straight up ⇒ one air mass
        // Sun and sensor coincident ⇒ exact backscatter.
        assert!(approx(g.scattering_angle.radians(), PI, 1e-9));
    }

    #[test]
    fn scattering_angle_spans_forward_to_back() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        // Sun at 45° in the north, sensor at 45° in the south ⇒ forward scatter (rays aligned).
        let s2 = 2.0_f64.sqrt() / 2.0;
        let to_sun = Vec3::new(s2, 0.0, s2);
        let to_sensor = Vec3::new(-s2, 0.0, s2);
        let g = observation_geometry(to_sun, to_sensor, up, north);
        // Incident (−to_sun) and scattered (to_sensor): (−s2,0,−s2)·(−s2,0,s2)=s2²−s2²=0 ⇒ 90°.
        assert!(approx(g.scattering_angle.radians(), PI / 2.0, 1e-9));
        // Relative azimuth is 180° (sun in the north, sensor in the south).
        assert!(approx(g.relative_azimuth.radians().abs(), PI, 1e-9));
        assert!(approx(g.solar_zenith.radians(), 45.0 * DEG, 1e-9));
    }

    #[test]
    fn wrap_pi_lands_in_its_half_open_range_and_moves_only_by_whole_turns() {
        // `fmod(π + π, 2π) = 0`, so +π is the excluded end and maps to −π.
        assert_eq!(wrap_pi(PI), -PI);
        assert_eq!(wrap_pi(-PI), -PI);
        assert_eq!(wrap_pi(0.0), 0.0);
        // −4.5 is below −π, the one regime where `fmod` returns a negative value and the branch that
        // lifts it back into `[0, 2π)` actually runs.
        for a in [-4.5_f64, -3.0, -1.7, -0.4, 0.0, 0.9, 2.6, 3.0] {
            for k in -3..=3 {
                let x = a + f64::from(k) * TAU;
                let w = wrap_pi(x);
                assert!((-PI..PI).contains(&w), "wrap_pi({x}) = {w} outside [−π, π)");
                let turns = (x - w) / TAU;
                assert!(
                    (turns - libm::round(turns)).abs() < 1e-12,
                    "wrap_pi({x}) = {w} moved by {turns} turns"
                );
            }
        }
    }

    #[test]
    fn the_geometry_reports_every_angle_its_directions_encode() {
        // Sun 35° from the vertical at azimuth 40°, sensor 20° at 250°: every angle distinct and
        // non-zero, so no part of the answer can hide behind a coordinate that happens to vanish.
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        // E = N × U = x̂ × ẑ = −ŷ.
        let dir = |zen: f64, az: f64| {
            let (sz, cz) = (libm::sin(zen), libm::cos(zen));
            Vec3::new(sz * libm::cos(az), -sz * libm::sin(az), cz)
        };
        let (zs, azs) = (35.0 * DEG, 40.0 * DEG);
        let (zv, azv) = (20.0 * DEG, 250.0 * DEG);
        let g = observation_geometry(dir(zs, azs), dir(zv, azv), up, north);

        assert!(approx(g.solar_zenith.radians(), zs, 1e-12));
        assert!(
            approx(g.view_zenith.radians(), zv, 1e-12),
            "view zenith {}",
            g.view_zenith.radians()
        );
        // 250° − 40° = 210°, one turn above the +π end, so −150°.
        assert!(
            approx(g.relative_azimuth.radians(), -150.0 * DEG, 1e-12),
            "relative azimuth {}",
            g.relative_azimuth.radians()
        );
        // Spherical law of cosines for the angle between the incident and the scattered ray.
        let cos_scat =
            -(libm::cos(zs) * libm::cos(zv) + libm::sin(zs) * libm::sin(zv) * libm::cos(azv - azs));
        assert!(approx(
            g.scattering_angle.radians(),
            libm::acos(cos_scat),
            1e-12
        ));
        // Kasten–Young at the solar zenith, written out rather than read back from `air_mass`.
        let ky = 1.0 / (libm::cos(zs) + 0.50572 * libm::pow(96.079_95 - 35.0, -1.6364));
        assert!(approx(g.air_mass, ky, 1e-15), "air mass {}", g.air_mass);
    }

    /// (G1) Every angle is the one the direction was built from: for `d(z, a) = sin z cos a N +
    /// sin z sin a E + cos z U` the code returns `z`, the wrapped `a_v − a_s`, and the scattering
    /// cosine `−(cos z_s cos z_v + sin z_s sin z_v cos(a_v − a_s))` — the identities
    /// `derivations/observation_geometry.mac` proves, on a grid that crosses every quadrant and the
    /// `±π` seam of the relative azimuth. Cosines are compared rather than angles where `acos` would
    /// amplify an ulp near `0` or `π`; the bounds are a few ulps of the quantities summed.
    #[test]
    fn the_angles_are_the_ones_the_enu_directions_were_built_from() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let north = Vec3::new(1.0, 0.0, 0.0);
        // E = N × U = x̂ × ẑ = −ŷ.
        let dir = |zen: f64, az: f64| {
            let (sz, cz) = (libm::sin(zen), libm::cos(zen));
            Vec3::new(sz * libm::cos(az), -sz * libm::sin(az), cz)
        };
        let zeniths = [0.05, 0.4, 0.9, 1.3, 1.55];
        let azimuths = [-3.0, -1.7, -0.2, 0.6, 1.9, 3.1];
        for &zs in &zeniths {
            for &azs in &azimuths {
                for &zv in &zeniths {
                    for &azv in &azimuths {
                        let g = observation_geometry(dir(zs, azs), dir(zv, azv), up, north);
                        let tol = 16.0 * f64::EPSILON;
                        assert!(
                            (libm::cos(g.solar_zenith.radians()) - libm::cos(zs)).abs() <= tol,
                            "solar zenith {} vs {zs}",
                            g.solar_zenith.radians()
                        );
                        assert!(
                            (libm::cos(g.view_zenith.radians()) - libm::cos(zv)).abs() <= tol,
                            "view zenith {} vs {zv}",
                            g.view_zenith.radians()
                        );
                        // The relative azimuth is a difference of two atan2's, each good to an ulp
                        // of π over the horizontal component sin z.
                        let dphi = g.relative_azimuth.radians() - (azv - azs);
                        let turns = dphi / TAU;
                        let tol_az =
                            8.0 * f64::EPSILON * PI * (1.0 / libm::sin(zs) + 1.0 / libm::sin(zv));
                        assert!(
                            (turns - libm::round(turns)).abs() * TAU <= tol_az,
                            "relative azimuth {} vs {}",
                            g.relative_azimuth.radians(),
                            azv - azs
                        );
                        assert!((-PI..PI).contains(&g.relative_azimuth.radians()));
                        let cos_scat = -(libm::cos(zs) * libm::cos(zv)
                            + libm::sin(zs) * libm::sin(zv) * libm::cos(azv - azs));
                        assert!(
                            (libm::cos(g.scattering_angle.radians()) - cos_scat).abs() <= tol,
                            "scattering cosine {} vs {cos_scat}",
                            libm::cos(g.scattering_angle.radians())
                        );
                    }
                }
            }
        }
    }

    /// (S1) The three angles keep their digits next to `0` and `π` (Ф28.N, C3 row 15). `A` is the
    /// double nearest `u = (2, −3, 6)/7` and each `b` the doubles nearest `cos t·u + sin t·w`,
    /// `w = (3, 6, 2)/7`, at `t = 10⁻ᵏ` and `π − 10⁻ᵏ`, k = 1..12; `derivations/angle_from_vectors.gp`
    /// gives the angle between `A` and `b` at 100 digits. `to_sun = b`, `to_sensor = −A`,
    /// `zenith = A` puts it in the solar zenith and the scattering angle, `to_sun = −A`,
    /// `to_sensor = b` in the view zenith and again the scattering angle. The bound is 2u of
    /// absolute error, the cross product's rounding, which `atan2` passes on unamplified
    /// (`derivations/angle_from_vectors.mac` 4), plus one ulp of the angle for the two roundings
    /// of the result and of the reference. Measured: at most 0.31 u next to 0, and exact or one ulp
    /// of π next to π. `acos` of the dot product was 4.9 u off at 1e-1, 930 u at 1e-3, 4.4e7 u
    /// (4.9e-9 rad) at 1e-8 and 1.9e8 u (2.1e-8 rad) at 1e-11; next to π, 16 u at `π − 10⁻¹` and
    /// 1.3e8 u (1.5e-8 rad) at `π − 10⁻¹²`.
    #[test]
    fn the_angles_keep_their_digits_next_to_zero_and_pi() {
        const A: [f64; 3] = [
            0.2857142857142856984253854,
            -0.4285714285714285476380780,
            0.8571428571428570952761561,
        ];
        #[rustfmt::skip]
        const ROWS: [([f64; 3], f64); 24] = [
        ([0.3270726543566480071234537, -0.3408588565647297508043323, 0.8813845464231158111090281], 0.1000000000000000135927475),
        ([0.2899856429765472198845089, -0.4199787144635708235895777, 0.8599570955954749518213021], 0.009999999999999996030332443),
        ([0.2861427142142975998950760, -0.4277140715714464191954391, 0.8574281428095594792537781], 0.0009999999999999788124737479),
        ([0.2857571414285000055421904, -0.4284857121430000193029741, 0.8571714242856666166048285], 9.999999999997153007459013e-5),
        ([0.2857185714142856380881597, -0.4285628571214287196866621, 0.8571457142428571263081949], 1.000000000000220366168395e-5),
        ([0.2857147142855714361253661, -0.4285705714283571388101279, 0.8571431428567143351315849], 1.000000000020992123939867e-6),
        ([0.2857143285714271452313540, -0.4285713428571407135336813, 0.8571428857142814328184954], 1.000000000028757327186489e-7),
        ([0.2857142899999999818838603, -0.4285714199999999807211282, 0.8571428599999999509151394], 9.999999994736441743814509e-9),
        ([0.2857142861428571545268085, -0.4285714277142856909463831, 0.8571428574285714363512056], 1.000000027229219767499213e-9),
        ([0.2857142857571428606888730, -0.4285714284857142786222539, 0.8571428571714285737925820], 1.000000082740370999087040e-10),
        ([0.2857142857185714368561946, -0.4285714285628571262876108, 0.8571428571457142542300289], 1.000000875758074940703406e-11),
        ([0.2857142857147143000240419, -0.4285714285705714554630674, 0.8571428571431428666826946], 9.999858085701168821672939e-13),
        ([-0.2415011543736524379788477, 0.5120018565307208335823930, -0.8243368797677854686867249], 3.041592653589793280895418),
        ([-0.2814143572615472321807317, 0.4371212858935708545082832, -0.8542429051188082933521173], 3.131592653589793210951140),
        ([-0.2852855715000118808077900, 0.4294283570000178573700111, -0.8568567143333690738771224], 3.140592653589793243853290),
        ([-0.2856714271429285911807483, 0.4286571407141428480258583, -0.8571142814286190070305338], 3.141492653589793243142076),
        ([-0.2857099999857143357573364, 0.4285799999785712688371575, -0.8571399999571428507394444], 3.141582653589793228327945),
        ([-0.2857138571427143003411686, 0.4285722857140714103785228, -0.8571425714281428742680191], 3.141591653589793265051474),
        ([-0.2857142428571414205507040, 0.4285715142857121628949812, -0.8571428285714243200388296], 3.141592553589793227656749),
        ([-0.2857142814285714149669104, 0.4285714371428571145550279, -0.8571428542857142396371728], 3.141592643589793243726202),
        ([-0.2857142852857142978351135, 0.4285714294285714043297730, -0.8571428568571428652234090], 3.141592652589793266744575),
        ([-0.2857142856714285916730489, 0.4285714286571428721650534, -0.8571428571142857277820326], 3.141592653489793238118769),
        ([-0.2857142857100000155057273, 0.4285714285800000244996966, -0.8571428571400000473445857], 3.141592653579793237635240),
        ([-0.2857142857138571523378801, 0.4285714285722856953242399, -0.8571428571425714348919200], 3.141592653588793260584364),
        ];
        let a = Vec3::new(A[0], A[1], A[2]);
        let north = Vec3::new(1.0, 0.0, 0.0);
        let u = f64::EPSILON / 2.0;
        for (b, want) in ROWS {
            let b = Vec3::new(b[0], b[1], b[2]);
            let lit = observation_geometry(b, -a, a, north);
            let seen = observation_geometry(-a, b, a, north);
            let bound = 2.0 * u + (want.next_up() - want);
            for (name, got) in [
                ("solar zenith", lit.solar_zenith.radians()),
                ("scattering angle", lit.scattering_angle.radians()),
                ("view zenith", seen.view_zenith.radians()),
                ("scattering angle", seen.scattering_angle.radians()),
            ] {
                assert!(
                    (got - want).abs() <= bound,
                    "{name} at {want:e}: {got:.17e}, {:.3} u off",
                    (got - want).abs() / u
                );
            }
        }
    }

    /// `(zenith, m)`: Kasten & Young (1989) at 40 digits from `derivations/observation_geometry.mac`,
    /// re-checked at 60 by `derivations/observation_geometry.gp`, for dyadic zeniths and the double
    /// nearest `π/2`.
    const KASTEN_YOUNG: [(f64, f64); 7] = [
        (0.0, 9.997119918558381315393438355106013890384e-1),
        (0.5, 1.138826624854773829714342718122949757378),
        (1.0, 1.846471199582342699878539262176237955795),
        (1.375, 5.017246978829234931212123394087344332953),
        (1.53125, 1.810269069650286168852866930954648170836e1),
        (1.5625, 3.163415279610748979292860300654418652789e1),
        (
            core::f64::consts::FRAC_PI_2,
            3.791960837783627306248202055583037736895e1,
        ),
    ];

    /// (G2) `air_mass` is Kasten & Young's formula `1/(cos z + 0.50572 (96.07995° − z)^−1.6364)` with
    /// the constants as printed, evaluated in Maxima and PARI/GP. The bound is relative and carries the
    /// power's conditioning `c·z/(b − z)`, which grows toward the horizon where `b − z` is 6°.
    #[test]
    fn the_air_mass_is_kasten_young_s_formula() {
        for (z, want) in KASTEN_YOUNG {
            let got = air_mass(Angle::from_radians(z));
            let z_deg = z * (180.0 / PI);
            let tol = 64.0 * f64::EPSILON * (1.0 + 1.6364 * z_deg / (96.079_95 - z_deg));
            let rel = (got / want - 1.0).abs();
            assert!(
                rel <= tol,
                "air mass at {z}: {got} vs {want} (rel {rel:.3e}, allowed {tol:.3e})"
            );
        }
    }

    #[test]
    fn air_mass_matches_known_values_and_is_finite_at_the_horizon() {
        assert!(approx(air_mass(Angle::from_radians(0.0)), 1.0, 2e-3));
        // Kasten–Young sits just below the plane-parallel 1/cos θ (= 2.0 at 60°): ~1.994.
        let m60 = air_mass(Angle::from_radians(60.0 * DEG));
        assert!((1.98..2.0).contains(&m60), "air mass at 60° = {m60}");
        // Finite at the horizon (~38), where 1/cos θ diverges.
        let horizon = air_mass(Angle::from_radians(90.0 * DEG));
        assert!(
            (30.0..45.0).contains(&horizon),
            "horizon air mass {horizon}"
        );
    }
}
