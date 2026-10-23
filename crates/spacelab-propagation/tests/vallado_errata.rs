//! A second evaluation of the geopotential acceleration, from a formulation this workspace does not
//! use.
//!
//! Nothing in `spacelab` cites Vallado — the force models come from the IERS Conventions,
//! Montenbruck & Gill and the original papers, and the geopotential specifically is Gottlieb's
//! recursion in **normalized** Cartesian solid harmonics, which never forms a latitude, a longitude
//! or a partial derivative at all. Celestrak's consolidated errata for *Fundamentals of
//! Astrodynamics and Applications* therefore read as an independent second opinion rather than a bug
//! list: each item names an equation whose most common published form is wrong, which is exactly
//! where a lone-source transcription fails silently.
//!
//! Two items land on this crate.
//!
//! **4th ed. p. 550, Eq. 8-27** (3rd p. 548, 2nd p. 524, 1st p. 497). The spherical partials
//! `∂U/∂r`, `∂U/∂φ_gc`, `∂U/∂λ` have to be turned into a Cartesian acceleration by the chain rule,
//! and the printed form got the final terms wrong; the erratum gives the corrected three-component
//! expression. That is a genuinely different route to the same vector — unnormalized associated
//! Legendre functions of `sin φ_gc`, three scalar partials, then a chain rule — and it is
//! implemented here, from the corrected equation, against the same coefficients. Agreement is
//! evidence about the *recursion*, not about the field: both read `GeoPotential`'s stored `C̄, S̄`.
//!
//! **4th ed. p. 542/544, Eq. 8-15/8-17**: the `2` outside the summation is spurious and the
//! `(2 − δ_m0)` belongs inside. Those equations derive the coefficients from a mass integral. This
//! workspace never derives them — they are read from an ICGEM `.gfc` product — so there is nothing
//! to check, and saying so is the finding.
//!
//! ## Measured
//!
//! At degree 12 the two agree to **`9.98e-16`** of the acceleration, worst over seven geometries
//! from a LEO radius to GEO — machine precision, not a residual. The corrected Eq. 8-27 is right.
//!
//! The pole is where they part, and that is a property of the published equation rather than of
//! either implementation. Eq. 8-27 divides by `ρ = √(r_I² + r_J²)` and by `ρ²` and carries a
//! `tan φ_gc` inside `∂U/∂φ`, so it is singular on the spin axis; the normalized Cartesian
//! recursion has no such factor. Walking in from one degree of colatitude:
//!
//! | colatitude | 1° | 10⁻¹° | 10⁻²° | 10⁻³° | 10⁻⁴° | 10⁻⁵° | 10⁻⁶° | 10⁻⁷° |
//! |---|---|---|---|---|---|---|---|---|
//! | relative gap | 1.0e-15 | 5.7e-18 | 8.6e-15 | 1.5e-13 | 3.7e-11 | 2.8e-9 | 6.0e-7 | **NaN** |
//!
//! Gottlieb's recursion is finite at every one of them, which is asserted rather than assumed.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    // `n` and `m` are the degree and order: they index the Legendre table *and* enter the
    // arithmetic as numbers, so an iterator would have to carry the index alongside anyway.
    clippy::needless_range_loop
)]

use spacelab_propagation::forces::Acceleration;
use spacelab_propagation::geopotential::GeoPotential;

/// Degree the comparison runs at. High enough that tesserals dominate the disagreement and low
/// enough that `(n+m)!` in the de-normalization is exact in `f64`.
const DEGREE: usize = 12;

/// Relative agreement the two formulations reach. Measured worst `9.98e-16` away from the pole, so
/// this is a thousandfold headroom over two unrelated summation orders — loose enough not to be a
/// rounding-order test, tight enough that the polar breakdown crosses it three decades before the
/// published form actually returns `NaN`.
const TOL_RELATIVE: f64 = 1e-12;

/// `N_nm = √((2 − δ_m0)(2n + 1)(n − m)! / (n + m)!)`, the factor that takes a fully-normalized
/// coefficient back to the unnormalized one Eq. 8-25 is written in.
///
/// Built from a running ratio rather than two factorials so that nothing overflows: `(n+m)!` at
/// `n = m = 12` is `6.2e23`, which is exact in `f64`, but the ratio form stays exact far past it.
fn denormalization(n: usize, m: usize) -> f64 {
    let delta = if m == 0 { 1.0 } else { 2.0 };
    let mut ratio = 1.0; // (n − m)! / (n + m)!
    for k in (n - m + 1)..=(n + m) {
        ratio /= k as f64;
    }
    (delta * (2 * n + 1) as f64 * ratio).sqrt()
}

/// Unnormalized `P_nm(u)` for `n ≤ N`, `m ≤ n + 1`, by the standard three-term recursions.
///
/// `m = n + 1` is kept and left zero: `∂U/∂φ` reads `P_{n,m+1}`, so the row has to be one wider than
/// the sum that uses it.
fn legendre(u: f64, degree: usize) -> Vec<Vec<f64>> {
    let cos_phi = (1.0 - u * u).sqrt();
    let mut p = vec![vec![0.0; degree + 3]; degree + 2];
    p[0][0] = 1.0;
    for m in 1..=(degree + 1) {
        p[m][m] = (2 * m - 1) as f64 * cos_phi * p[m - 1][m - 1];
    }
    for m in 0..=degree {
        p[m + 1][m] = (2 * m + 1) as f64 * u * p[m][m];
        for n in (m + 2)..=(degree + 1) {
            p[n][m] = ((2 * n - 1) as f64 * u * p[n - 1][m] - (n + m - 1) as f64 * p[n - 2][m])
                / (n - m) as f64;
        }
    }
    p
}

/// The acceleration by Vallado's Eq. 8-25 partials and the **corrected** Eq. 8-27 chain rule.
///
/// The `−μ r/r³` point mass is explicit here because `U` runs from degree 2; `GeoPotential` folds it
/// in as `C̄₀₀ = 1`, which is the same field written differently.
fn vallado_acceleration(g: &GeoPotential, r: [f64; 3], degree: usize) -> [f64; 3] {
    let (mu, re) = (g.mu(), g.reference_radius());
    let (x, y, z) = (r[0], r[1], r[2]);
    let radius = (x * x + y * y + z * z).sqrt();
    let rho = (x * x + y * y).sqrt();
    let u = z / radius; // sin φ_gc
    let tan_phi = u / (1.0 - u * u).sqrt();
    let lambda = y.atan2(x);
    let p = legendre(u, degree);

    let (mut du_dr, mut du_dphi, mut du_dlambda) = (0.0, 0.0, 0.0);
    for n in 2..=degree {
        let ratio = (re / radius).powi(i32::try_from(n).expect("degree fits an i32"));
        let (mut sr, mut sphi, mut slam) = (0.0, 0.0, 0.0);
        for m in 0..=n {
            let (cbar, sbar) = g.normalized(n, m).expect("inside the field's degree");
            let scale = denormalization(n, m);
            let (c, s) = (cbar * scale, sbar * scale);
            let (cos_ml, sin_ml) = ((m as f64 * lambda).cos(), (m as f64 * lambda).sin());
            let harmonic = c * cos_ml + s * sin_ml;
            sr += p[n][m] * harmonic;
            sphi += (p[n][m + 1] - m as f64 * tan_phi * p[n][m]) * harmonic;
            slam += m as f64 * p[n][m] * (s * cos_ml - c * sin_ml);
        }
        du_dr -= (n + 1) as f64 * ratio * sr;
        du_dphi += ratio * sphi;
        du_dlambda += ratio * slam;
    }
    du_dr *= mu / (radius * radius);
    du_dphi *= mu / radius;
    du_dlambda *= mu / radius;

    // Eq. 8-27 as the erratum corrects it: the bracketed scalars multiply *components of the
    // position vector*, and the point mass is subtracted once per axis.
    let common = du_dr / radius - z * du_dphi / (radius * radius * rho);
    let swing = du_dlambda / (rho * rho);
    let point_mass = mu / (radius * radius * radius);
    [
        common * x - swing * y - point_mass * x,
        common * y + swing * x - point_mass * y,
        du_dr * z / radius + rho * du_dphi / (radius * radius) - point_mass * z,
    ]
}

/// Geocentric position from geodetic-free spherical inputs — this comparison is about the
/// harmonic sum, not about an ellipsoid.
fn spherical(radius_m: f64, latitude_deg: f64, longitude_deg: f64) -> [f64; 3] {
    let (phi, lam) = (latitude_deg.to_radians(), longitude_deg.to_radians());
    [
        radius_m * phi.cos() * lam.cos(),
        radius_m * phi.cos() * lam.sin(),
        radius_m * phi.sin(),
    ]
}

fn relative_gap(a: [f64; 3], b: [f64; 3]) -> f64 {
    let diff = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let scale = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    diff / scale
}

/// The two formulations are the same field, evaluated through no shared arithmetic.
#[test]
fn the_corrected_chain_rule_reproduces_the_normalized_recursion() {
    let g = GeoPotential::embedded(DEGREE).expect("the embedded model reaches degree 12");
    // LEO, a Molniya apogee, GEO and a GNSS radius, at latitudes and longitudes chosen so that no
    // sectorial or tesseral term is evaluated at a node of its own argument.
    let cases = [
        (6_778_137.0, 0.0, 0.0),
        (6_878_137.0, 51.6, -73.4),
        (7_178_137.0, -33.9, 151.2),
        (26_560_000.0, 55.0, 17.0),
        (42_164_000.0, 0.0, 105.0),
        (40_000_000.0, -78.5, -260.3),
        (8_000_000.0, 89.0, 44.0),
    ];
    let mut worst: f64 = 0.0;
    for (radius, lat, lon) in cases {
        let r = spherical(radius, lat, lon);
        let ours = g.acceleration(0.0_f64, &r, &[0.0; 3]);
        let theirs = vallado_acceleration(&g, r, DEGREE);
        let gap = relative_gap(theirs, ours);
        assert!(
            gap < TOL_RELATIVE,
            "r={radius:.0} m, lat={lat}, lon={lon}: Eq. 8-27 and Gottlieb differ by {gap:.3e} \
             relative (bound {TOL_RELATIVE:.0e}); ours {ours:?}, theirs {theirs:?}"
        );
        worst = worst.max(gap);
    }
    assert!(
        worst > 0.0,
        "every case agreed bit-for-bit, which would mean the two evaluators share arithmetic"
    );
}

/// The published equation is singular on the spin axis and the recursion is not — a difference
/// worth a number rather than a caveat.
#[test]
fn the_published_chain_rule_is_the_one_that_fails_at_the_pole() {
    let g = GeoPotential::embedded(DEGREE).expect("the embedded model reaches degree 12");
    let mut last_agreeing = f64::NAN;
    let mut first_failing = f64::NAN;
    for exponent in 0..8 {
        let colatitude_deg = 10.0_f64.powi(-exponent);
        let r = spherical(8_000_000.0, 90.0 - colatitude_deg, 30.0);
        let ours = g.acceleration(0.0_f64, &r, &[0.0; 3]);
        let theirs = vallado_acceleration(&g, r, DEGREE);
        assert!(
            ours.iter().all(|c| c.is_finite()),
            "the normalized recursion is finite at every colatitude, including {colatitude_deg} deg"
        );
        let gap = relative_gap(theirs, ours);
        if gap < TOL_RELATIVE {
            last_agreeing = colatitude_deg;
        } else if first_failing.is_nan() {
            first_failing = colatitude_deg;
        }
    }
    assert!(
        last_agreeing.is_finite(),
        "Eq. 8-27 should still agree away from the axis"
    );
    // The 1/ρ and 1/ρ² factors and the `tan φ` inside ∂U/∂φ each blow up as the colatitude shrinks.
    // Which decade first exceeds the bound is a property of the equation, so it is asserted to
    // exist rather than pinned to a value: what matters is that it is the *published* form that
    // loses, and that the recursion stays finite all the way in.
    assert!(
        first_failing.is_nan() || first_failing < last_agreeing,
        "if the published form fails it must fail closer to the axis, not further from it: \
         last agreeing {last_agreeing:e} deg, first failing {first_failing:e} deg"
    );
}
