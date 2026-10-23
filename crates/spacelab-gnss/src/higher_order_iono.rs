//! **Second-order ionospheric delay** — the higher-order term Orekit's ionosphere models omit.
//!
//! The first-order ionospheric delay (`±40.3·STEC/f²`) is removed by any dual-frequency combination,
//! but the *second-order* term — the geomagnetic-field-dependent `1/f³` correction from the Faraday
//! effect — survives an ionosphere-free combination and asymmetrically biases the two frequencies. It
//! is ~0.1 % of the first-order delay (mm–cm at zenith), the dominant residual ionospheric error in
//! mm-level PPP and precise orbit determination. Orekit models only the first order, so this is a
//! beyond-Orekit accuracy term.
//!
//! The group delay follows Bassiri & Hajj (1993), in the form given by ESA Navipedia
//! (<https://gssc.esa.int/navipedia/index.php/Ionospheric_Delay>):
//!
//! ```text
//! I₂,phase = -(7527·c) / (2·f³) · ∫ Nₑ·B·cos θ dl        I₂,group = -2·I₂,phase
//!         ⇒ I₂,group = (7527·c / f³) · ∫ Nₑ·B·cos θ dl ≈ (7527·c / f³) · (B⃗·k̂) · STEC
//! ```
//!
//! with `Nₑ` in e⁻/m³, `B` in Tesla, `f` in Hz, the delay in metres, and `θ` the angle between the
//! field `B⃗` and the propagation direction `k̂` (receiver→satellite), so `B·cos θ = B⃗·k̂`. Following
//! the standard approximation the slowly varying `B⃗·k̂` is evaluated at the ionospheric pierce point
//! and pulled out of the integral, leaving the slant TEC.
//!
//! `B⃗` defaults to the IGRF-14 (epoch 2025.0) centred dipole, the field the correction is
//! conventionally written with. That default is **not** uniformly adequate: the argument for it — the
//! second-order term is small, so an error inside it is smaller still — bounds the residual by the
//! term, not below it. Over the South Atlantic Anomaly, where the non-dipole field is largest, the
//! dipole and the full model give `B⃗·k̂` of *opposite sign* at the zenith (measured: a factor
//! −2.04 at 0° N, 20° W through a 400 km shell), so the dipole reverses the term rather than
//! approximating it. [`second_order_group_delay_igrf`] evaluates the full IGRF-14 field
//! ([`spacelab_geomagnetism::igrf`]) instead, and is what mm-level work should use.
//!
//! `derivations/second_order_iono.mac` derives both constants of that formula from Appleton–Hartree
//! rather than quoting them: the `1/f²` coefficient comes out at **40.308**, which is the 40.3
//! everybody knows, and the `1/f³` group constant at `q³/(8π³ε₀m²)` = `7527.3854·c`, of which the
//! published four digits are a rounding worth 5.12×10⁻⁵. It also shows the group/phase factor of
//! −2 is `f·d/df` on an `f⁻³` term and nothing else.

// The pierce-point/receiver geometry uses paired latitudes/longitudes and single-letter ENU axes that
// are the conventional notation.
#![allow(clippy::similar_names, clippy::many_single_char_names)]

use libm::{asin, cos, sin, sqrt};

/// IGRF-14 first-degree Gauss coefficients, epoch 2025.0 (nanotesla), from the IAGA V-MOD
/// `igrf14coeffs.txt`. These three define the centred geomagnetic dipole.
const G10_NT: f64 = -29350.0;
const G11_NT: f64 = -1410.3;
const H11_NT: f64 = 4545.5;

/// IGRF geomagnetic reference radius (m) — 6371.2 km, the sphere the Gauss coefficients refer to.
const IGRF_RADIUS: f64 = 6_371_200.0;

/// Elementary charge (C), electron mass (kg) and vacuum permittivity (F/m), CODATA 2018 — the
/// charge and the speed of light exact by the 2019 SI redefinition.
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
const ELECTRON_MASS: f64 = 9.109_383_701_5e-31;
const VACUUM_PERMITTIVITY: f64 = 8.854_187_812_8e-12;

/// Bassiri–Hajj second-order group-delay constant. With `B⃗·k̂` in Tesla, STEC in e⁻/m² and `f` in Hz,
/// `CONST·(B⃗·k̂)·STEC / f³` is the group delay in metres.
///
/// Navipedia publishes the grouping as `7527·c`, and `derivations/second_order_iono.mac` shows where
/// that four-digit number comes from: expanding Appleton–Hartree to second order gives the phase
/// term `−q³·Nₑ·B·cosθ / (16π³ε₀m²f³)`, so the group constant is `q³/(8π³ε₀m²)` and the published
/// figure is `7527.3854` rounded. Carrying the derivation rather than the rounding is worth
/// **5.12×10⁻⁵ relative** — invisible beside this model's own approximations (a dipole or a
/// thin shell, and `B⃗·k̂` pulled out of the integral, both percent-level), but free, and it keeps
/// the constant a consequence of CODATA rather than a literal nobody can re-derive. The same
/// expansion one order earlier gives 40.308, the first-order constant everyone knows as 40.3, which
/// is what says the expansion is the right one.
const SECOND_ORDER_GROUP_CONST: f64 = {
    let pi3 = core::f64::consts::PI * core::f64::consts::PI * core::f64::consts::PI;
    let q3 = ELEMENTARY_CHARGE * ELEMENTARY_CHARGE * ELEMENTARY_CHARGE;
    q3 / (8.0 * pi3 * VACUUM_PERMITTIVITY * ELECTRON_MASS * ELECTRON_MASS)
};

/// Mean Earth radius (m) for the thin-shell pierce-point geometry.
const EARTH_MEAN_RADIUS: f64 = 6_371_000.0;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Earth's magnetic field vector `B⃗` (Tesla) at ECEF position `r` (m), as the IGRF-14 (2025.0) centred
/// dipole: `B⃗ = (a³ B₀ / r³)·[3(ĝ·r̂)r̂ − ĝ]`, with `ĝ = (g₁¹, h₁¹, g₁⁰)/B₀` the dipole-moment direction
/// (points to the *southern* geomagnetic pole, since `g₁⁰ < 0`) and `B₀ = √(g₁₀²+g₁₁²+h₁₁²)`.
///
/// This is the degree-1 IGRF scalar potential `V = a(a/r)²·(g₁¹cosφ + h₁¹sinφ)sinθ + g₁⁰cosθ)` written
/// vectorially and differentiated (`B⃗ = −∇V`); at the north geomagnetic pole it points down and at the
/// magnetic equator its magnitude is `B₀`, as it must.
#[must_use]
pub fn geomagnetic_dipole_field(r: [f64; 3]) -> [f64; 3] {
    let b0_nt = sqrt(G10_NT * G10_NT + G11_NT * G11_NT + H11_NT * H11_NT);
    let g_hat = [G11_NT / b0_nt, H11_NT / b0_nt, G10_NT / b0_nt];
    let b0 = b0_nt * 1e-9; // nanotesla → tesla

    let rn = sqrt(dot(r, r));
    let r_hat = [r[0] / rn, r[1] / rn, r[2] / rn];
    let g_dot_r = dot(g_hat, r_hat);
    let ratio = IGRF_RADIUS / rn;
    let scale = b0 * ratio * ratio * ratio;

    [
        scale * (3.0 * g_dot_r * r_hat[0] - g_hat[0]),
        scale * (3.0 * g_dot_r * r_hat[1] - g_hat[1]),
        scale * (3.0 * g_dot_r * r_hat[2] - g_hat[2]),
    ]
}

/// Second-order ionospheric **group** delay (m) for the pierce point `pierce` (ECEF, m), line-of-sight
/// unit vector `los_unit` (ECEF, receiver→satellite), slant TEC `stec` (e⁻/m²; 1 TECU = 1e16) and
/// carrier `frequency` (Hz). Signed: the delay is positive or negative depending on the field/ray
/// geometry, which is exactly the L1/L2 asymmetry an ionosphere-free combination fails to cancel.
#[must_use]
pub fn second_order_group_delay(
    pierce: [f64; 3],
    los_unit: [f64; 3],
    stec: f64,
    frequency: f64,
) -> f64 {
    let b = geomagnetic_dipole_field(pierce);
    let b_cos_theta = dot(b, los_unit); // B·cos θ = B⃗·k̂
    SECOND_ORDER_GROUP_CONST * b_cos_theta * stec / (frequency * frequency * frequency)
}

/// As [`second_order_group_delay`] but with the field from the full IGRF-14 model (degree/order 13,
/// [`spacelab_geomagnetism::igrf::field_ecef`]) at decimal `year`, rather than the centred dipole. This is the
/// accurate form: the dipole's departure from the real field is bounded by the second-order term
/// itself rather than by anything smaller, and over the South Atlantic Anomaly the two disagree in
/// sign (see the module documentation), so prefer this wherever the term matters at all.
#[must_use]
pub fn second_order_group_delay_igrf(
    pierce: [f64; 3],
    los_unit: [f64; 3],
    stec: f64,
    frequency: f64,
    year: f64,
) -> f64 {
    let b = spacelab_geomagnetism::igrf::field_ecef(pierce, year);
    let b_cos_theta = dot(b, los_unit);
    SECOND_ORDER_GROUP_CONST * b_cos_theta * stec / (frequency * frequency * frequency)
}

/// Second-order ionospheric group delay (m) from station-relative geometry: receiver geocentric
/// `latitude`/`longitude` (rad), signal `elevation`/`azimuth` (rad), slant TEC `stec` (e⁻/m²) and
/// `frequency` (Hz), with the ionospheric shell at `pierce_height` (m, typically ~350–450 km).
///
/// Builds the thin-shell pierce point and the ECEF line of sight, then defers to
/// [`second_order_group_delay`]. The pierce point is placed on a sphere of radius
/// `EARTH_MEAN_RADIUS + pierce_height`; a kilometre-level pierce error is negligible against the
/// slowly varying dipole field.
#[must_use]
pub fn second_order_group_delay_geodetic(
    latitude: f64,
    longitude: f64,
    elevation: f64,
    azimuth: f64,
    pierce_height: f64,
    stec: f64,
    frequency: f64,
) -> f64 {
    // Earth-central angle from the station to the pierce point (thin-shell single-layer geometry).
    let central = core::f64::consts::FRAC_PI_2
        - elevation
        - asin(EARTH_MEAN_RADIUS / (EARTH_MEAN_RADIUS + pierce_height) * cos(elevation));

    let (sin_lat, cos_lat) = (sin(latitude), cos(latitude));
    let pierce_lat = asin(sin_lat * cos(central) + cos_lat * sin(central) * cos(azimuth));
    let pierce_lon = longitude + asin(sin(central) * sin(azimuth) / cos(pierce_lat));

    let radius = EARTH_MEAN_RADIUS + pierce_height;
    let (sin_pl, cos_pl) = (sin(pierce_lat), cos(pierce_lat));
    let (sin_plon, cos_plon) = (sin(pierce_lon), cos(pierce_lon));
    let pierce = [
        radius * cos_pl * cos_plon,
        radius * cos_pl * sin_plon,
        radius * sin_pl,
    ];

    // Line-of-sight unit vector: local ENU (east, north, up) at the station rotated into ECEF.
    let (sin_lon, cos_lon) = (sin(longitude), cos(longitude));
    let (e, n, u) = (
        cos(elevation) * sin(azimuth),
        cos(elevation) * cos(azimuth),
        sin(elevation),
    );
    let los = [
        -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u,
        cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u,
        cos_lat * n + sin_lat * u,
    ];

    second_order_group_delay(pierce, los, stec, frequency)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b0_tesla() -> f64 {
        sqrt(G10_NT * G10_NT + G11_NT * G11_NT + H11_NT * H11_NT) * 1e-9
    }

    fn norm(v: [f64; 3]) -> f64 {
        sqrt(dot(v, v))
    }

    #[test]
    fn the_second_order_constant_is_the_published_one_and_says_by_how_much_it_is_not() {
        // Navipedia's grouping is `7527·c`; the constant here is `q³/(8π³ε₀m²)`, which
        // `derivations/second_order_iono.mac` derives from Appleton–Hartree. The two agree to four
        // digits by construction, and the point of the test is the fifth: the published figure is
        // low by 5.12e-5 relative, so a transcription slip in any of the three CODATA values would
        // move this far more than the rounding does.
        const SPEED_OF_LIGHT: f64 = 299_792_458.0;
        let published = 7527.0 * SPEED_OF_LIGHT;
        let relative = (published - SECOND_ORDER_GROUP_CONST) / SECOND_ORDER_GROUP_CONST;
        assert!(
            (-5.2e-5..-5.0e-5).contains(&relative),
            "derived {SECOND_ORDER_GROUP_CONST:e} against published {published:e} is {relative:e}"
        );

        // The same expansion one order earlier is the first-order constant everybody knows as 40.3,
        // and reproducing it is what says the expansion — and so the constant above — is the right
        // one. `q²/(8π²ε₀m)` to six digits is 40.3082.
        let pi2 = core::f64::consts::PI * core::f64::consts::PI;
        let first_order = ELEMENTARY_CHARGE * ELEMENTARY_CHARGE
            / (8.0 * pi2 * VACUUM_PERMITTIVITY * ELECTRON_MASS);
        assert!(
            (first_order - 40.308_193).abs() < 1e-6,
            "first-order constant {first_order}"
        );
    }

    #[test]
    fn reduced_dipole_moment_matches_the_known_igrf_value() {
        // IGRF-14 2025.0 surface equatorial field ≈ 2.973e-5 T (the "reduced" dipole moment).
        assert!((b0_tesla() - 2.9733e-5).abs() < 1e-8, "B0 = {}", b0_tesla());
    }

    #[test]
    fn dipole_magnitude_is_b0_at_the_magnetic_equator_and_2b0_at_the_pole() {
        let b0_nt = sqrt(G10_NT * G10_NT + G11_NT * G11_NT + H11_NT * H11_NT);
        let g_hat = [G11_NT / b0_nt, H11_NT / b0_nt, G10_NT / b0_nt];

        // Geomagnetic north pole: r̂ = -ĝ (ĝ points south). Field magnitude there is 2·B₀.
        let pole = [
            -g_hat[0] * IGRF_RADIUS,
            -g_hat[1] * IGRF_RADIUS,
            -g_hat[2] * IGRF_RADIUS,
        ];
        assert!((norm(geomagnetic_dipole_field(pole)) - 2.0 * b0_tesla()).abs() < 1e-9);

        // A point on the geomagnetic equator: any r̂ ⟂ ĝ. Field magnitude there is B₀.
        // Build one via ĝ × ẑ (⟂ ĝ), normalized, at radius = IGRF_RADIUS.
        let z = [0.0, 0.0, 1.0];
        let perp = [
            g_hat[1] * z[2] - g_hat[2] * z[1],
            g_hat[2] * z[0] - g_hat[0] * z[2],
            g_hat[0] * z[1] - g_hat[1] * z[0],
        ];
        let pn = norm(perp);
        let equator = [
            perp[0] / pn * IGRF_RADIUS,
            perp[1] / pn * IGRF_RADIUS,
            perp[2] / pn * IGRF_RADIUS,
        ];
        let b = geomagnetic_dipole_field(equator);
        assert!((norm(b) - b0_tesla()).abs() < 1e-9, "|B|={}", norm(b));
        // At the magnetic equator the field is perpendicular to the radial (horizontal).
        assert!(dot(b, equator).abs() / (norm(b) * norm(equator)) < 1e-9);
    }

    #[test]
    fn second_order_delay_has_the_expected_millimetre_magnitude() {
        // 50 TECU, GPS L1, field antiparallel/parallel to the line of sight (worst case B·cos θ = ±B₀).
        let stec = 50.0e16; // e⁻/m²
        let f_l1 = 1.575_42e9; // Hz
        let b0_nt = sqrt(G10_NT * G10_NT + G11_NT * G11_NT + H11_NT * H11_NT);
        let g_hat = [G11_NT / b0_nt, H11_NT / b0_nt, G10_NT / b0_nt];

        // Pierce point at the geomagnetic north pole (radius IGRF_RADIUS), LOS along the field (down).
        let pierce = [
            -g_hat[0] * IGRF_RADIUS,
            -g_hat[1] * IGRF_RADIUS,
            -g_hat[2] * IGRF_RADIUS,
        ];
        let b = geomagnetic_dipole_field(pierce);
        let los = [b[0] / norm(b), b[1] / norm(b), b[2] / norm(b)]; // k̂ ∥ B⃗ → cos θ = 1
        let delay = second_order_group_delay(pierce, los, stec, f_l1);

        // Closed form: (7527·c/f³)·|B|·STEC with |B| = 2·B₀.
        let expected = SECOND_ORDER_GROUP_CONST * (2.0 * b0_tesla()) * stec / (f_l1 * f_l1 * f_l1);
        assert!((delay - expected).abs() < 1e-12);
        // Physically a couple of centimetres at the pole with full alignment and 50 TECU.
        assert!((0.005..0.05).contains(&delay.abs()), "delay {delay} m");
    }

    #[test]
    fn geodetic_wrapper_reduces_toward_zero_at_the_magnetic_equator_zenith() {
        // On the geographic equator looking straight up, the line of sight is horizontal-radial while
        // the (near-equatorial) field is largely horizontal-north, so B·k̂ — and the delay — is small.
        let f_l1 = 1.575_42e9;
        let zenith_equator = second_order_group_delay_geodetic(
            0.0,
            0.0,                          // lat, lon on the equator
            core::f64::consts::FRAC_PI_2, // elevation 90° (zenith)
            0.0,
            350e3,
            50.0e16,
            f_l1,
        );
        // A low-elevation pass toward the magnetic north gives a much larger second-order term.
        let low_north = second_order_group_delay_geodetic(
            core::f64::consts::FRAC_PI_4,
            0.0,
            0.174_53, // ~10° elevation
            0.0,      // due north
            350e3,
            50.0e16,
            f_l1,
        );
        assert!(zenith_equator.abs() < low_north.abs());
        // Both stay in the physical mm–cm band.
        assert!(low_north.abs() < 0.05);
    }

    /// Local east/north/up at geocentric `lat`/`lon`, with north taken as `up × east` rather than
    /// written out — a different route to the same basis than the rotation under test.
    fn enu(lat: f64, lon: f64) -> ([f64; 3], [f64; 3], [f64; 3]) {
        let up = [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)];
        let east = [-sin(lon), cos(lon), 0.0];
        let north = [
            up[1] * east[2] - up[2] * east[1],
            up[2] * east[0] - up[0] * east[2],
            up[0] * east[1] - up[1] * east[0],
        ];
        (east, north, up)
    }

    /// Where the ray leaving `station` along `los` meets the sphere of radius `shell` — the pierce
    /// point by vector geometry, which shares no arithmetic with the module's spherical trigonometry.
    fn pierce_by_ray(station: [f64; 3], los: [f64; 3], shell: f64) -> [f64; 3] {
        let b = 2.0 * dot(station, los);
        let c = dot(station, station) - shell * shell;
        let s = 0.5 * (-b + sqrt(b * b - 4.0 * c));
        [
            station[0] + s * los[0],
            station[1] + s * los[1],
            station[2] + s * los[2],
        ]
    }

    #[test]
    fn the_dipole_falls_off_as_the_inverse_cube_and_points_down_at_its_own_pole() {
        // Both facts the fixtures above cannot see. They evaluate the field only at `IGRF_RADIUS`,
        // where `(a/r)³` is the identity and any rearrangement of it is invisible, and they check only
        // |B⃗|, which cannot see what happens inside a single component.
        let b0_nt = sqrt(G10_NT * G10_NT + G11_NT * G11_NT + H11_NT * H11_NT);
        let g_hat = [G11_NT / b0_nt, H11_NT / b0_nt, G10_NT / b0_nt];

        // The axis is anchored externally: these three coefficients place the north geomagnetic pole
        // at 80.79° N, 72.76° W, the IGRF-14 2025.0 dipole pole published alongside them.
        let pole_lat = asin(-g_hat[2]).to_degrees();
        let pole_lon = libm::atan2(-g_hat[1], -g_hat[0]).to_degrees();
        assert!(
            (pole_lat - 80.789).abs() < 0.01 && (pole_lon + 72.763).abs() < 0.01,
            "dipole pole at {pole_lat} N, {pole_lon} E"
        );

        // A dipole is exactly inverse-cube: B⃗(λr⃗) = B⃗(r⃗)/λ³, componentwise.
        for r in [[-4.0e6, 3.0e6, 5.0e6], [1.0e7, -2.0e6, 3.0e6]] {
            let b = geomagnetic_dipole_field(r);
            for lambda in [2.0_f64, 3.7] {
                let scaled =
                    geomagnetic_dipole_field([r[0] * lambda, r[1] * lambda, r[2] * lambda]);
                for axis in 0..3 {
                    let want = b[axis] / lambda.powi(3);
                    assert!(
                        (scaled[axis] - want).abs() < 1e-12 * norm(b),
                        "axis {axis} at λ = {lambda}: {} vs {want}",
                        scaled[axis]
                    );
                }
            }
        }

        // Over the north geomagnetic pole at twice the reference radius the field is radial and
        // downward, of magnitude 2·B₀/2³ = B₀/4 — i.e. exactly (B₀/4)·ĝ, since r̂ = −ĝ there.
        let r = [
            -g_hat[0] * 2.0 * IGRF_RADIUS,
            -g_hat[1] * 2.0 * IGRF_RADIUS,
            -g_hat[2] * 2.0 * IGRF_RADIUS,
        ];
        let b = geomagnetic_dipole_field(r);
        for axis in 0..3 {
            let want = b0_tesla() / 4.0 * g_hat[axis];
            assert!(
                (b[axis] - want).abs() < 1e-12 * b0_tesla(),
                "axis {axis}: {} vs {want}",
                b[axis]
            );
        }
    }

    #[test]
    fn the_delay_is_linear_in_tec_and_inverse_cubic_in_frequency() {
        // Every existing assertion on the delay is a band or an inequality, so the shape of
        // `CONST·(B⃗·k̂)·STEC/f³` is unconstrained: dividing by the TEC instead of multiplying keeps
        // the answer in the millimetre band it is checked against.
        let pierce = [-4.0e6, 3.0e6, 5.0e6];
        let n = norm(pierce);
        let los = [pierce[0] / n, pierce[1] / n, pierce[2] / n];
        let (stec, f_l1) = (50.0e16, 1.575_42e9);

        for label in ["dipole", "igrf"] {
            let delay = |s: f64, f: f64| {
                if label == "dipole" {
                    second_order_group_delay(pierce, los, s, f)
                } else {
                    second_order_group_delay_igrf(pierce, los, s, f, 2025.0)
                }
            };
            let base = delay(stec, f_l1);
            assert!(base.abs() > 1e-6, "{label}: delay {base} m is not physical");
            assert!(
                (delay(2.0 * stec, f_l1) - 2.0 * base).abs() < 1e-12 * base.abs(),
                "{label}: not linear in TEC"
            );
            assert!(
                (delay(stec, f_l1 / 2.0) - 8.0 * base).abs() < 1e-11 * base.abs(),
                "{label}: not inverse-cubic in frequency"
            );
        }

        // The full IGRF-14 field differs from the centred dipole only by its higher-degree terms, so
        // the two delays must agree to well within a factor of two — a zero from either is not "small".
        let dipole = second_order_group_delay(pierce, los, stec, f_l1);
        let full = second_order_group_delay_igrf(pierce, los, stec, f_l1, 2025.0);
        assert!(
            (full - dipole).abs() < 0.25 * dipole.abs(),
            "dipole {dipole} vs full {full}"
        );
    }

    #[test]
    fn the_geodetic_wrapper_pierces_the_shell_where_the_ray_meets_it() {
        // The wrapper reaches the pierce point by spherical trigonometry and the line of sight by a
        // written-out ENU rotation. Here both are rebuilt by vector geometry — a ray/sphere
        // intersection and a cross product — and the two routes must land on the same delay. The
        // existing fixture compares two geometries against each other with `<`, which no rearrangement
        // of this arithmetic is obliged to break.
        let (stec, f_l1) = (50.0e16, 1.575_42e9);
        for (lat_deg, lon_deg, el_deg, az_deg, h) in [
            (0.0_f64, 0.0_f64, 90.0_f64, 0.0_f64, 350.0e3_f64),
            (45.0, 10.0, 10.0, 0.0, 350.0e3),
            (-33.0, 151.0, 25.0, 137.0, 450.0e3),
            (60.0, -70.0, 5.0, 300.0, 400.0e3),
            (-70.0, -170.0, 55.0, 220.0, 400.0e3),
        ] {
            let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
            let (el, az) = (el_deg.to_radians(), az_deg.to_radians());
            let (e, n, u) = enu(lat, lon);
            let mut los = [0.0; 3];
            for axis in 0..3 {
                los[axis] =
                    cos(el) * sin(az) * e[axis] + cos(el) * cos(az) * n[axis] + sin(el) * u[axis];
            }
            assert!(
                (norm(los) - 1.0).abs() < 1e-15,
                "the reference line of sight is a unit vector"
            );
            assert!(
                (dot(los, u) - sin(el)).abs() < 1e-15,
                "the reference line of sight is at elevation {el_deg}"
            );

            let station = [
                EARTH_MEAN_RADIUS * u[0],
                EARTH_MEAN_RADIUS * u[1],
                EARTH_MEAN_RADIUS * u[2],
            ];
            let pierce = pierce_by_ray(station, los, EARTH_MEAN_RADIUS + h);
            let want = second_order_group_delay(pierce, los, stec, f_l1);
            let got = second_order_group_delay_geodetic(lat, lon, el, az, h, stec, f_l1);
            assert!(
                (got - want).abs() < 1e-9 * want.abs(),
                "lat {lat_deg}, el {el_deg}, az {az_deg}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn igrf_variant_stays_in_band_and_differs_from_the_dipole() {
        // Same geometry through the dipole and the full IGRF-14 field: both physical (mm–cm), and the
        // full field genuinely differs from the dipole by its higher-degree terms (~10 % of |B|).
        let pierce = [-4.0e6, 3.0e6, 5.0e6]; // ~400 km, mid-latitude
        let n = norm(pierce);
        let los = [pierce[0] / n, pierce[1] / n, pierce[2] / n]; // radial line of sight
        let (stec, f_l1) = (50.0e16, 1.575_42e9);
        let dipole = second_order_group_delay(pierce, los, stec, f_l1);
        let full = second_order_group_delay_igrf(pierce, los, stec, f_l1, 2025.0);
        assert!(full.abs() < 0.05, "full {full}");
        assert!(
            (full - dipole).abs() > 1e-7,
            "dipole {dipole} vs full {full}"
        );
    }

    #[test]
    fn over_the_south_atlantic_the_dipole_gets_the_sign_of_the_term_wrong() {
        // (0° N, 20° W) is inside the South Atlantic Anomaly, where the non-dipole part of the
        // geomagnetic field is largest. Looking straight up through a 400 km shell there, the two
        // fields give `B⃗·k̂` of opposite sign, so the centred dipole does not approximate the
        // second-order term at this geometry — it reverses it.
        let (stec, f_l1) = (50.0e16, 1.575_42e9);
        let (lat, lon) = (0.0, -20.0_f64.to_radians());
        let (_, _, up) = enu(lat, lon);
        let shell = EARTH_MEAN_RADIUS + 400e3;
        let pierce = [up[0] * shell, up[1] * shell, up[2] * shell];

        let dipole = second_order_group_delay(pierce, up, stec, f_l1);
        let full = second_order_group_delay_igrf(pierce, up, stec, f_l1, 2025.0);
        // Both delays are linear in the slant TEC, so the ratio is geometry alone: measured
        // −2.0364, i.e. the dipole is wrong by 3.0× its own value. This is the claim the module
        // documentation makes, and it is the reason the documentation no longer calls the dipole
        // adequate at the mm level.
        assert!(dipole * full < 0.0, "dipole {dipole} m, full {full} m");
        assert!(
            (full / dipole + 2.0364).abs() < 1e-3,
            "ratio {}",
            full / dipole
        );
    }
}
