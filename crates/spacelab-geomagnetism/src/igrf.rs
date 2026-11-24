//! **IGRF-14 main geomagnetic field** — the full degree/order-13 International Geomagnetic Reference
//! Field, the standard model of Earth's core field. Where `crate::higher_order_iono` needs only the
//! dipole, this is the complete spherical-harmonic field for any application that wants the real
//! geomagnetic vector: higher-order ionospheric corrections, charged-particle environment, attitude
//! magnetometer references.
//!
//! The synthesis is the standard one (as in the IAGA reference code and the `ppigrf` port): Schmidt
//! semi-normalised associated Legendre functions `Pⁿₘ(cos θ)` (recursion from Wertz, *Spacecraft
//! Attitude Determination and Control*) evaluated against the Gauss coefficients `gⁿₘ`, `hⁿₘ`, giving
//! the geocentric field
//!
//! ```text
//! B_r = Σ (a/r)ⁿ⁺² (n+1) Pⁿₘ (gⁿₘ cos mφ + hⁿₘ sin mφ)
//! B_θ = − Σ (a/r)ⁿ⁺² dPⁿₘ/dθ (gⁿₘ cos mφ + hⁿₘ sin mφ)
//! B_φ = Σ (a/r)ⁿ⁺² (m / sin θ) Pⁿₘ (gⁿₘ sin mφ − hⁿₘ cos mφ)
//! ```
//!
//! with `a = 6371.2 km` the IGRF reference radius, `θ` geocentric colatitude, `φ` east longitude, `r`
//! geocentric radius. The coefficients are the machine-generated epoch-2025.0 values plus the
//! 2025-2030 secular variation (`crate::igrf14_coeffs`); `gⁿₘ(t) = gⁿₘ(2025) + (t − 2025)·ġⁿₘ`, so
//! the model is valid roughly 2025-2030. Earlier epochs would need the full DGRF time series.
//!
//! Gate: `derivations/igrf_synthesis.mac` proves the recursion is the Schmidt quasi-normalised
//! definition and its `θ`-derivative for every degree and order to 13, that each term solves
//! Laplace's equation, that the three components are `−∇V` and that the ECEF rotation is the
//! spherical triad; it evaluates `−∇V` at 40 digits from the IAGA file's exact decimals, re-derived
//! from PARI's Legendre polynomials by `derivations/igrf_synthesis.gp`. The module test
//! `the_field_is_minus_the_gradient_of_the_schmidt_potential` (G2) ties the shipped synthesis to it;
//! `field_matches_ppigrf_geocentric_oracle` is a comparator.

// The geomagnetism symbols (n, m, g, h, θ, φ, P, S, B_r/θ/φ) are one/two letters by universal
// convention; IGRF/IAGA/Schmidt/DGRF are proper nouns.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]

use libm::{atan2, cos, hypot, sin, sqrt};

use crate::igrf14_coeffs::{G_2025, G_SV, H_2025, H_SV};

/// IGRF reference radius (km) — the sphere the Gauss coefficients refer to.
const REFERENCE_RADIUS_KM: f64 = 6371.2;
/// Coefficient epoch (decimal year); secular variation is applied relative to this.
const REFERENCE_EPOCH: f64 = 2025.0;
/// Maximum spherical-harmonic degree/order of IGRF-14.
const MAX_DEGREE: usize = 13;
/// `|sin θ|` at and below which `B_φ` takes its on-axis limit instead of dividing by `sin θ`.
///
/// The limit drops the m ≥ 2 terms, which are O(θ): about 6e3 nT per radian of colatitude, so a
/// threshold of 1e-12 put 6e-9 nT of truncation where the round-off is 1e-11 nT. Above this value
/// `Pⁿ₁` and `Pⁿ₂`, of order sin θ and sin² θ, are normal doubles and the quotient keeps its
/// relative accuracy; below it the dropped terms are under 1e-150 nT.
const AXIS_SIN: f64 = 1.491_668_146_240_041_3e-154;

/// Schmidt semi-normalised associated Legendre functions `Pⁿₘ(cos θ)` and their `θ`-derivatives, for
/// geocentric colatitude `colatitude` (rad). Returned as `[n][m]` tables (`n, m ≤ 13`); the recursion
/// is Wertz's (via `ppigrf`) with the Schmidt normalisation applied last.
fn schmidt_legendre(colatitude: f64) -> ([[f64; 14]; 14], [[f64; 14]; 14]) {
    let (sinth, costh) = (sin(colatitude), cos(colatitude));
    let mut p = [[0.0f64; 14]; 14];
    let mut dp = [[0.0f64; 14]; 14];
    let mut s = [[0.0f64; 14]; 14];
    p[0][0] = 1.0;
    s[0][0] = 1.0;

    for n in 1..=MAX_DEGREE {
        for m in 0..=n {
            if n == m {
                p[n][n] = sinth * p[n - 1][m - 1];
                dp[n][n] = sinth * dp[n - 1][m - 1] + costh * p[n - 1][n - 1];
            } else if n == 1 {
                p[n][m] = costh * p[n - 1][m];
                dp[n][m] = costh * dp[n - 1][m] - sinth * p[n - 1][m];
            } else {
                let k = (((n - 1) * (n - 1)) as f64 - (m * m) as f64)
                    / (((2 * n - 1) * (2 * n - 3)) as f64);
                p[n][m] = costh * p[n - 1][m] - k * p[n - 2][m];
                dp[n][m] = costh * dp[n - 1][m] - sinth * p[n - 1][m] - k * dp[n - 2][m];
            }
            // Accumulate the Schmidt normalisation factor.
            if m == 0 {
                s[n][0] = s[n - 1][0] * (2.0 * n as f64 - 1.0) / n as f64;
            } else {
                let delta = if m == 1 { 2.0 } else { 1.0 };
                s[n][m] = s[n][m - 1] * sqrt((n - m + 1) as f64 * delta / (n + m) as f64);
            }
        }
    }
    for n in 1..=MAX_DEGREE {
        for m in 0..=n {
            p[n][m] *= s[n][m];
            dp[n][m] *= s[n][m];
        }
    }
    (p, dp)
}

/// The geocentric field components `[B_r, B_θ, B_φ]` in **nanotesla** at geocentric colatitude
/// `colatitude` (rad), east `longitude` (rad), geocentric `radius_km` (km) and decimal `year`.
///
/// On the polar axis (`|sin θ| ≤ 1.5e-154`, the square root of the smallest normal double) `B_φ`
/// takes the limit of `(m/sin θ)·Pⁿₘ`:
/// `cos θ·dPⁿ₁/dθ` for m = 1 and 0 for m ≥ 2 (`derivations/igrf_synthesis.mac` §6).
#[must_use]
pub fn field_geocentric(colatitude: f64, longitude: f64, radius_km: f64, year: f64) -> [f64; 3] {
    let dt = year - REFERENCE_EPOCH;
    let (p, dp) = schmidt_legendre(colatitude);
    let (sinth, costh) = (sin(colatitude), cos(colatitude));
    let ratio = REFERENCE_RADIUS_KM / radius_km;

    let (mut b_r, mut b_theta, mut b_phi) = (0.0, 0.0, 0.0);
    let mut ratio_pow = ratio * ratio * ratio; // (a/r)^{n+2}, starting at n = 1
    for n in 1..=MAX_DEGREE {
        for m in 0..=n {
            let g = G_2025[n][m] + dt * G_SV[n][m];
            let h = H_2025[n][m] + dt * H_SV[n][m];
            let m_phi = m as f64 * longitude;
            let (cos_mphi, sin_mphi) = (cos(m_phi), sin(m_phi));
            let gh_cos = g * cos_mphi + h * sin_mphi;

            b_r += ratio_pow * (n as f64 + 1.0) * p[n][m] * gh_cos;
            b_theta -= ratio_pow * dp[n][m] * gh_cos;
            if sinth.abs() > AXIS_SIN {
                b_phi += ratio_pow * (m as f64 / sinth) * p[n][m] * (g * sin_mphi - h * cos_mphi);
            } else if m == 1 {
                // On the axis `Pⁿ₁/sin θ` is 0/0, and its limit is `cos θ·dPⁿ₁/dθ` (l'Hôpital, with
                // cos θ = ±1 there); for m ≥ 2 the quotient carries sin^(m−1) θ and vanishes.
                b_phi += ratio_pow * costh * dp[n][1] * (g * sin_mphi - h * cos_mphi);
            }
        }
        ratio_pow *= ratio;
    }
    [b_r, b_theta, b_phi]
}

/// Earth's magnetic field vector `B⃗` in **tesla**, in the ECEF frame, at ECEF position `position` (m)
/// and decimal `year` — the full IGRF-14 field (degree/order 13). Rotates the geocentric
/// `[B_r, B_θ, B_φ]` of [`field_geocentric`] into ECEF.
#[must_use]
pub fn field_ecef(position: [f64; 3], year: f64) -> [f64; 3] {
    let r = sqrt(position[0] * position[0] + position[1] * position[1] + position[2] * position[2]);
    if r == 0.0 {
        return [0.0; 3];
    }
    // Not `acos(z/r)`: it returns exactly 0 within 1.5e-8 rad of the axis, where `z/r` rounds to 1.
    let colatitude = atan2(hypot(position[0], position[1]), position[2]);
    let longitude = atan2(position[1], position[0]);
    let [b_r, b_theta, b_phi] = field_geocentric(colatitude, longitude, r / 1000.0, year);

    let (sinth, costh) = (sin(colatitude), cos(colatitude));
    let (sinl, cosl) = (sin(longitude), cos(longitude));
    // r̂, θ̂, φ̂ in ECEF; nT → tesla.
    [
        (b_r * sinth * cosl + b_theta * costh * cosl - b_phi * sinl) * 1e-9,
        (b_r * sinth * sinl + b_theta * costh * sinl + b_phi * cosl) * 1e-9,
        (b_r * costh - b_theta * sinth) * 1e-9,
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal)] // ppigrf oracle field values, quoted as printed

    use super::*;

    #[test]
    fn schmidt_legendre_low_order_closed_forms() {
        // Schmidt: P₁⁰ = cos θ, P₁¹ = sin θ, P₂⁰ = (3cos²θ − 1)/2.
        let theta = 0.7_f64;
        let (p, dp) = schmidt_legendre(theta);
        assert!((p[1][0] - cos(theta)).abs() < 1e-12);
        assert!((p[1][1] - sin(theta)).abs() < 1e-12);
        assert!((p[2][0] - (3.0 * cos(theta) * cos(theta) - 1.0) / 2.0).abs() < 1e-12);
        // dP₁⁰/dθ = −sin θ, dP₁¹/dθ = cos θ.
        assert!((dp[1][0] + sin(theta)).abs() < 1e-12);
        assert!((dp[1][1] - cos(theta)).abs() < 1e-12);
    }

    #[test]
    fn field_matches_ppigrf_geocentric_oracle() {
        // Independent oracle: ppigrf.igrf_gc(r, colat°, lon°, <date>), the full degree-13
        // synthesis, in nT. Two points spanning both hemispheres and a range of altitude/longitude.
        //
        // The 2030 rows are what make the secular variation observable at all: at the coefficient
        // epoch `dt` is zero, so `g(2025) ± 0·ġ` is the same number whichever sign the code carries
        // and the whole `G_SV`/`H_SV` half of the model goes unchecked.
        let cases: [(f64, f64, f64, f64, [f64; 3]); 4] = [
            // (radius_km, colat_deg, lon_deg, year, [B_r, B_theta, B_phi])
            (
                6771.2,
                60.0,
                45.0,
                2025.0,
                [-27002.739425, -25615.913726, 1644.019857],
            ),
            (
                6571.2,
                100.0,
                200.0,
                2025.0,
                [10774.367007, -29092.710256, 5732.318771],
            ),
            (
                6771.2,
                60.0,
                45.0,
                2030.0,
                [-27157.781183, -25674.456676, 1668.477242],
            ),
            (
                6571.2,
                100.0,
                200.0,
                2030.0,
                [10661.028616, -28967.657836, 5742.434133],
            ),
        ];
        for (r, colat_deg, lon_deg, year, want) in cases {
            let b = field_geocentric(colat_deg.to_radians(), lon_deg.to_radians(), r, year);
            for i in 0..3 {
                assert!(
                    (b[i] - want[i]).abs() < 1e-3,
                    "r={r} year={year} B[{i}] {} vs {}",
                    b[i],
                    want[i]
                );
            }
        }
    }

    /// (G3) On the polar axis every component is a closed-form sum over the sealed table.
    ///
    /// At `θ = 0` and `θ = π` only two orders survive. `Pⁿ₀(±1) = (±1)ⁿ`, so at the reference
    /// radius `B_r = Σ (n+1)(±1)ⁿ gⁿ₀(t)`. Every `dPⁿₘ/dθ` but `m = 1` vanishes, and `dPⁿ₁/dθ` is
    /// `√(n(n+1)/2)` north, `(−1)ⁿ√(n(n+1)/2)` south. `B_φ`'s `Pⁿ₁/sin θ` is `0/0` there; its limit
    /// is `cos θ·dPⁿ₁/dθ = cⁿ⁺¹√(n(n+1)/2)` with `c = cos θ = ±1`
    /// (`derivations/igrf_synthesis.mac` §6). So
    ///
    /// ```text
    /// B_θ = −Σ c·L_n (gⁿ₁ cos φ + hⁿ₁ sin φ),   B_φ = Σ L_n (gⁿ₁ sin φ − hⁿ₁ cos φ),   L_n = cⁿ⁺¹√(n(n+1)/2).
    /// ```
    ///
    /// The test sums these by hand, with no recursion and no division. The east component used to
    /// be dropped on the axis: 103.8 nT short north and 7 224 nT south at 6771 km. Both 2025 and
    /// 2030 are checked, so the secular-variation arithmetic is in the sums too. Measured: 8.8e-16
    /// relative at worst.
    #[test]
    fn on_the_polar_axis_every_component_is_its_closed_form_limit() {
        let longitude = 0.7_f64;
        for year in [2025.0, 2030.0] {
            let dt = year - REFERENCE_EPOCH;
            for (colatitude, c) in [(0.0, 1.0_f64), (core::f64::consts::PI, -1.0)] {
                let (mut b_r, mut b_theta, mut b_phi) = (0.0, 0.0, 0.0);
                for n in 1..=MAX_DEGREE {
                    let sign = if n % 2 == 0 { 1.0 } else { c };
                    let g0 = G_2025[n][0] + dt * G_SV[n][0];
                    let g1 = G_2025[n][1] + dt * G_SV[n][1];
                    let h1 = H_2025[n][1] + dt * H_SV[n][1];
                    let limit = sign * c * sqrt((n * (n + 1)) as f64 / 2.0);
                    b_r += (n as f64 + 1.0) * sign * g0;
                    b_theta -= c * limit * (g1 * cos(longitude) + h1 * sin(longitude));
                    b_phi += limit * (g1 * sin(longitude) - h1 * cos(longitude));
                }
                let got = field_geocentric(colatitude, longitude, REFERENCE_RADIUS_KM, year);
                for (k, want) in [b_r, b_theta, b_phi].into_iter().enumerate() {
                    assert!(
                        (got[k] - want).abs() < 1e-14 * want.abs(),
                        "{year}, colatitude {colatitude}: component {k} {} against {want}",
                        got[k],
                    );
                }
            }
        }
    }

    /// (G3) `field_ecef` on the polar axis and within a metre of it, against
    /// `derivations/igrf_synthesis.mac` §6 at 60 digits (nT, epoch 2025.0, 6771.2 km).
    ///
    /// On the axis the horizontal field is the closed form `−Σ (a/r)ⁿ⁺² √(n(n+1)/2) (gⁿ₁, hⁿ₁)`,
    /// with the factor `(−1)ⁿ⁺¹` on the south axis. Off it, by `2⁻ᵏ (1, 2)` m for k = 21, 14, 7, 5
    /// and 0, the reference is `−∇V` at the exact position. `acos(z/r)` returned a colatitude of
    /// exactly 0 within 0.1 m of the axis, and the east component was dropped there: the north axis
    /// read (−1151.5, 0.0) nT for (−1151.5, 103.8), and 2⁻²¹ m off it, at longitude 1.107 rad, the
    /// horizontal field was about 1000 nT wrong. A position built from a latitude of ±90° lies
    /// 4e-10 m off the axis, inside that band; its field is the axis value to 1e-12 nT. Measured:
    /// 2.2e-11 nT at worst, six ulps of the 4.8e4 nT vertical component.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn the_ecef_field_on_and_beside_the_polar_axis_is_its_closed_form() {
        const R: f64 = 6_771_200.0;
        const NORTH: [f64; 3] = [
            -1.151_547_607_136_706_617_179_656_736_269_532e3,
            1.037_578_257_798_808_756_172_236_696_900_194e2,
            -4.792_238_942_664_172_145_396_781_761_202_037e4,
        ];
        const SOUTH: [f64; 3] = [
            1.075_430_758_645_771_036_249_069_251_678_241e4,
            -7.224_120_924_918_786_541_950_305_843_499_910e3,
            -4.285_276_937_750_965_403_984_691_623_266_732e4,
        ];
        #[rustfmt::skip]
        let near: [(i32, f64, [f64; 3]); 10] = [
            (21, R, [-1.151_547_607_141_951_637_685_994_510_937_858e3, 1.037_578_257_716_629_978_880_850_863_120_616e2, -4.792_238_942_664_178_131_935_418_683_985_324e4]),
            (21, -R, [1.075_430_758_646_286_532_447_646_370_659_367e4, -7.224_120_924_909_817_364_481_726_718_741_380e3, -4.285_276_937_750_942_991_820_667_858_303_115e4]),
            (14, R, [-1.151_547_607_808_069_241_988_955_307_256_515e3, 1.037_578_247_279_925_262_845_821_190_772_312e2, -4.792_238_942_664_938_422_338_335_960_883_229e4]),
            (14, -R, [1.075_430_758_711_754_549_665_515_765_314_365e4, -7.224_120_923_770_731_825_972_288_438_211_614e3, -4.285_276_937_748_096_646_984_156_081_282_087e4]),
            (7, R, [-1.151_547_693_071_122_561_040_838_732_729_017e3, 1.037_576_911_381_721_134_783_490_494_041_828e2, -4.792_238_942_762_255_528_839_838_054_670_268e4]),
            (7, -R, [1.075_430_767_091_660_730_211_676_728_109_483e4, -7.224_120_777_967_782_898_575_558_102_455_113e3, -4.285_276_937_383_764_417_908_563_472_207_377e4]),
            (5, R, [-1.151_547_950_874_370_008_877_900_551_467_896e3, 1.037_572_872_130_452_518_387_626_772_055_803e2, -4.792_238_943_056_504_892_109_528_320_675_909e4]),
            (5, -R, [1.075_430_792_429_329_529_783_340_486_987_234e4, -7.224_120_337_114_771_990_360_374_147_422_021e3, -4.285_276_936_282_160_371_084_586_323_354_482e4]),
            (0, R, [-1.151_558_606_741_427_580_987_813_666_769_948e3, 1.037_405_916_403_800_881_892_124_641_280_231e2, -4.792_238_955_217_779_022_814_655_202_586_337e4]),
            (0, -R, [1.075_431_839_719_269_418_814_742_572_055_020e4, -7.224_102_115_190_349_875_099_929_262_315_368e3, -4.285_276_890_747_764_502_318_964_104_789_088e4]),
        ];
        let check = |position: [f64; 3], want: [f64; 3], what: &str| {
            let got = field_ecef(position, 2025.0);
            for k in 0..3 {
                let got_nt = got[k] * 1e9;
                assert!(
                    (got_nt - want[k]).abs() < 5e-11,
                    "{what}: component {k} {got_nt} nT against {} nT",
                    want[k]
                );
            }
        };
        check([0.0, 0.0, R], NORTH, "north axis");
        check([0.0, 0.0, -R], SOUTH, "south axis");
        for (k, z, want) in near {
            let off = libm::ldexp(1.0, -k);
            check(
                [off, 2.0 * off, z],
                want,
                &format!("2^-{k} (1, 2) m from z = {z}"),
            );
        }
        for (latitude, want) in [(90.0_f64, NORTH), (-90.0, SOUTH)] {
            let phi = latitude.to_radians();
            check(
                [R * cos(phi), 0.0, R * sin(phi)],
                want,
                &format!("latitude {latitude}"),
            );
        }
        // Off the axis but inside the limit's band: sin θ = 4.4e-164, where the field is the axis
        // value to 1e-150 nT. The band is where sin² θ leaves the normal doubles.
        assert_eq!(AXIS_SIN.to_bits(), sqrt(f64::MIN_POSITIVE).to_bits());
        let off = libm::ldexp(1.0, -520);
        check(
            [off, 2.0 * off, R],
            NORTH,
            "2^-520 (1, 2) m from the north axis",
        );
        check(
            [off, 2.0 * off, -R],
            SOUTH,
            "2^-520 (1, 2) m from the south axis",
        );
    }

    /// The geocentric → ECEF rotation, by projection onto an independently built triad.
    ///
    /// The only assertion this rotation had was a magnitude band, and a rotation preserves
    /// magnitude — so every one of the nine operators in the three rows could be changed without
    /// moving the asserted number. Here the spherical unit vectors are written out from their
    /// definitions in the test, and each component of the returned vector is recovered by a dot
    /// product: `B⃗·r̂ = B_r`, `B⃗·θ̂ = B_θ`, `B⃗·φ̂ = B_φ`, with the geocentric components taken from
    /// [`field_geocentric`], which the oracle test above pins independently.
    #[test]
    fn the_ecef_rotation_projects_back_onto_the_spherical_triad() {
        let pos = [4.0e6, 2.0e6, 5.0e6];
        let year = 2027.5;
        let r = sqrt(pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]);
        let (colat, lon) = (libm::acos(pos[2] / r), atan2(pos[1], pos[0]));
        let (st, ct) = (sin(colat), cos(colat));
        let (sl, cl) = (sin(lon), cos(lon));
        let triad = [
            [st * cl, st * sl, ct],  // r̂
            [ct * cl, ct * sl, -st], // θ̂
            [-sl, cl, 0.0],          // φ̂
        ];

        let b_ecef = field_ecef(pos, year);
        let want = field_geocentric(colat, lon, r / 1000.0, year);
        for (k, (unit, want_nt)) in triad.iter().zip(want).enumerate() {
            let got_nt = (b_ecef[0] * unit[0] + b_ecef[1] * unit[1] + b_ecef[2] * unit[2]) * 1e9;
            assert!(
                (got_nt - want_nt).abs() < 1e-6 * want_nt.abs(),
                "component {k}: {got_nt} nT projected, {want_nt} nT geocentric",
            );
        }
        // A rotation is norm-preserving, which no combination of the three rows need be.
        let n_ecef = sqrt(b_ecef[0] * b_ecef[0] + b_ecef[1] * b_ecef[1] + b_ecef[2] * b_ecef[2]);
        let n_sph = sqrt(want[0] * want[0] + want[1] * want[1] + want[2] * want[2]) * 1e-9;
        assert!(
            (n_ecef - n_sph).abs() < 1e-12 * n_sph,
            "{n_ecef} vs {n_sph}"
        );
    }

    #[test]
    fn ecef_field_magnitude_is_physical_and_secular_variation_moves_it() {
        // ~400 km over the mid-latitude northern hemisphere: |B| in the ~20-50 µT band.
        let pos = [4.0e6, 2.0e6, 5.0e6];
        let b2025 = field_ecef(pos, 2025.0);
        let mag = sqrt(b2025[0] * b2025[0] + b2025[1] * b2025[1] + b2025[2] * b2025[2]);
        assert!((15e-6..60e-6).contains(&mag), "|B| {mag} T");
        // Secular variation shifts the field a few years out.
        let b2030 = field_ecef(pos, 2030.0);
        let d = sqrt(
            (b2030[0] - b2025[0]).powi(2)
                + (b2030[1] - b2025[1]).powi(2)
                + (b2030[2] - b2025[2]).powi(2),
        );
        assert!(d > 1e-9, "secular variation should move B: {d} T");
    }

    /// (G2) The field is minus the gradient of the Schmidt potential. `derivations/igrf_synthesis.mac`
    /// proves that the Wertz recursion times its accumulated factors is the Schmidt quasi-normalised
    /// `Pⁿₘ` and its `θ`-derivative for every `n, m ≤ 13`, and that the three coded components are
    /// `−∂V/∂r`, `−(1/r)∂V/∂θ` and `−(1/(r sin θ))∂V/∂φ`; it then evaluates `−∇V` from the definition at
    /// 40 digits, with the coefficients read as exact decimals from the IAGA file, at these three
    /// points — one at the coefficient epoch, one at the end of the secular-variation window, one
    /// between, in both hemispheres. `derivations/igrf_synthesis.gp` re-evaluates them from PARI's own
    /// Legendre polynomials and agrees to 6e-36 nT. The inputs are doubles (radians, kilometres), so
    /// the reference is the field at exactly what the test passes.
    ///
    /// The bound is the sum's rounding: about a hundred terms of up to 3e4 nT, each a few roundings
    /// deep, against an ulp of 3.6e-12 nT at that size; measured 1.5e-11 nT, four ulps, at worst.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn the_field_is_minus_the_gradient_of_the_schmidt_potential() {
        let cases: [(f64, f64, f64, f64, [f64; 3]); 3] = [
            (
                1.0,
                0.5,
                6771.25,
                2025.0,
                [
                    -2.745691301632609806507634325022553533148e4,
                    -2.437751823657892777948550313259657607389e4,
                    1.810361034812790661533836797608695878954e3,
                ],
            ),
            (
                1.75,
                3.5,
                6571.5,
                2030.0,
                [
                    1.085958398417876417948311373642412659937e4,
                    -2.889611203608967979272386856787747651347e4,
                    5.757108496833010019237629378447388139413e3,
                ],
            ),
            (
                2.625,
                5.25,
                7000.0,
                2027.5,
                [
                    2.221308232454103436252749632446625474901e4,
                    -1.386195549581576320968095115259524339195e4,
                    1.912815784310201997253132608934219810885e3,
                ],
            ),
        ];
        for (colatitude, longitude, radius, year, want) in cases {
            let got = field_geocentric(colatitude, longitude, radius, year);
            for k in 0..3 {
                assert!(
                    (got[k] - want[k]).abs() < 1e-10,
                    "colatitude {colatitude}, component {k}: {} against {}",
                    got[k],
                    want[k]
                );
            }
        }
    }

    /// The IGRF-14 Gauss coefficients, sealed.
    ///
    /// The field oracles above compare against `ppigrf` to a relative tolerance the low-degree
    /// terms dominate; a sign flipped at degree 12 moves the field by well under it, and
    /// `cargo mutants` finds 38 such sign deletions in the table that no test here can see. The
    /// constant comes from `derivations/table_seals.py`, which reads the generated file with
    /// Python's float parser — a different parse of the same text than rustc's.
    #[test]
    fn the_igrf14_coefficients_are_the_ones_the_generator_wrote() {
        let flat: Vec<f64> = [&G_2025, &H_2025, &G_SV, &H_SV]
            .iter()
            .copied()
            .flatten()
            .flatten()
            .copied()
            .collect();
        assert_eq!(flat.len(), 4 * 14 * 14, "IGRF-14 table shape moved");
        let seal = spacelab_testkit::fnv1a_f64(flat);
        assert_eq!(
            seal, 0xab08_48c8_9526_2035,
            "IGRF-14 (epoch 2025.0) coefficients moved"
        );
    }
}
