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

// The geomagnetism symbols (n, m, g, h, θ, φ, P, S, B_r/θ/φ) are one/two letters by universal
// convention; IGRF/IAGA/Schmidt/DGRF are proper nouns.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::cast_precision_loss
)]

use libm::{acos, atan2, cos, sin, sqrt};

use crate::igrf14_coeffs::{G_2025, G_SV, H_2025, H_SV};

/// IGRF reference radius (km) — the sphere the Gauss coefficients refer to.
const REFERENCE_RADIUS_KM: f64 = 6371.2;
/// Coefficient epoch (decimal year); secular variation is applied relative to this.
const REFERENCE_EPOCH: f64 = 2025.0;
/// Maximum spherical-harmonic degree/order of IGRF-14.
const MAX_DEGREE: usize = 13;

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
#[must_use]
pub fn field_geocentric(colatitude: f64, longitude: f64, radius_km: f64, year: f64) -> [f64; 3] {
    let dt = year - REFERENCE_EPOCH;
    let (p, dp) = schmidt_legendre(colatitude);
    let sinth = sin(colatitude);
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
            if sinth.abs() > 1e-12 {
                b_phi += ratio_pow * (m as f64 / sinth) * p[n][m] * (g * sin_mphi - h * cos_mphi);
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
    let colatitude = acos(position[2] / r);
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

    /// On the polar axis the east component is not computed at all, and `B_r` collapses to a sum
    /// over the zonal coefficients that the test can do by hand.
    ///
    /// At `θ = 0` every `Pⁿₘ` with `m ≥ 1` vanishes and `Pⁿ₀(1) = 1`, so at the reference radius
    /// `B_r = Σ (n+1)·gⁿ₀(t)` — a check on the secular-variation arithmetic that needs no external
    /// model, only the sealed table. The `sin θ` guard is exercised on the one input that can
    /// distinguish `>` from `>=`: `sin(1e-12)` rounds to exactly `1e-12` (the cubic correction is
    /// 25 orders of magnitude below the ulp), so a `>=` there admits the division and writes a
    /// non-zero `B_φ` where the contract says the term is dropped.
    #[test]
    fn the_polar_axis_drops_the_east_component_and_leaves_a_zonal_sum() {
        for year in [2025.0, 2030.0] {
            let dt = year - REFERENCE_EPOCH;
            let want_b_r: f64 = (1..=MAX_DEGREE)
                .map(|n| (n as f64 + 1.0) * (G_2025[n][0] + dt * G_SV[n][0]))
                .sum();
            let b = field_geocentric(0.0, 0.7, REFERENCE_RADIUS_KM, year);
            assert!(
                (b[0] - want_b_r).abs() < 1e-9 * want_b_r.abs(),
                "{year}: B_r {} vs the zonal sum {want_b_r}",
                b[0],
            );
            assert_eq!(b[2].to_bits(), 0.0_f64.to_bits(), "{year}: B_φ is not +0");
        }
        // The threshold itself: `sin θ` is exactly the guard value here, and the guard excludes it.
        let b = field_geocentric(1e-12, 0.7, REFERENCE_RADIUS_KM, 2025.0);
        assert_eq!(sin(1e-12).to_bits(), 1e-12_f64.to_bits());
        assert_eq!(b[2].to_bits(), 0.0_f64.to_bits(), "B_φ at sin θ = 1e-12");
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
        let (colat, lon) = (acos(pos[2] / r), atan2(pos[1], pos[0]));
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
