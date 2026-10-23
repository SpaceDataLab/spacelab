//! The air-mass and refraction fits against a ray integrated through a real atmosphere.
//!
//! `observability.rs` carries four published fits to two physical quantities, and until this file
//! existed they were verified against *each other*: Kasten & Young (1989) and Young (1994) are
//! independent fits in different arguments, joined through Sæmundsson refraction, and they agree to
//! 6e-3 relative at the horizon. That is a real cross-check and it has one blind spot — it cannot
//! see an error the fits share, and all four were fitted to the same atmosphere.
//!
//! `tools/gen_airmass_oracle.py` computes both quantities instead: the U.S. Standard Atmosphere
//! 1976's hydrostatic layers for the density, Gladstone–Dale for the refractive index, and the
//! spherical ray invariant `n r sin ψ = const` for the path. Two quadratures, no fit anywhere. Its
//! header records the two choices it cannot make for itself — the Earth radius, worth 0.2 % at the
//! horizon across the mean/equatorial/geopotential spread, and where the atmosphere is truncated,
//! worth 4.1e-6 of the column.
//!
//! The verdict splits. **The air-mass fits are confirmed**: better than 0.2 % everywhere above 0.5°
//! of apparent altitude and 0.63 % at the horizon itself, against a computation that shares no
//! coefficient with them. **The refraction fits are not**: both run high at every altitude, by up to
//! 4.8 % (Sæmundsson) and 3.3 % (Bennett), and the third test here shows that is theirs rather than
//! the integration's.

#![allow(clippy::expect_used, clippy::unwrap_used)] // a malformed fixture is a broken generator

use spacelab_coverage::observability::{
    ApparentAltitude, Refraction, TrueAltitude, airmass_kasten_young_1989, airmass_young_1994,
};

/// The conditions the oracle integrates at — U.S. Standard Atmosphere sea level, not the 1010 hPa /
/// 283 K the two refraction fits are published at. Comparing them at their own conditions against a
/// ray traced through a different atmosphere would put a 1.5 % density difference into the residual
/// and call it a modelling error.
const USSA: Refraction = Refraction {
    pressure_hpa: 1013.25,
    temperature_k: 288.15,
};

/// `(apparent altitude °, air mass, refraction ′, true altitude °)`.
fn oracle() -> Vec<(f64, f64, f64, f64)> {
    let path = "tests/fixtures/airmass_integrated.txt";
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("{path} — see tools/gen_airmass_oracle.py"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l
                .split_whitespace()
                .map(|f| f.parse().expect("a number"))
                .collect();
            (v[0], v[1], v[2], v[3])
        })
        .collect()
}

#[test]
fn the_air_mass_fits_agree_with_a_numerically_integrated_ray() {
    let rows = oracle();
    assert_eq!(rows.len(), 19, "fixture has {} rows", rows.len());

    let mut worst_horizon = 0.0_f64;
    let mut worst_above = 0.0_f64;
    let mut worst_young = 0.0_f64;
    for (apparent_deg, air_mass, _, true_deg) in rows {
        let kasten = airmass_kasten_young_1989(ApparentAltitude(apparent_deg.to_radians()));
        let error = ((kasten - air_mass) / air_mass).abs();
        if apparent_deg < 0.5 {
            worst_horizon = worst_horizon.max(error);
        } else {
            worst_above = worst_above.max(error);
        }

        // Young 1994 is indexed by the *true* zenith angle of the same ray, which is where the
        // oracle's refraction earns its place: without it there is no way to ask the second fit
        // about the geometry the first one was asked about.
        let young = airmass_young_1994(TrueAltitude(true_deg.to_radians()));
        if young.is_finite() {
            worst_young = worst_young.max(((young - air_mass) / air_mass).abs());
        }
    }

    // Measured: 0.63 % at the apparent horizon, 0.21 % at 0.25°, and 0.16 % everywhere above 0.5°.
    // Kasten & Young's fit is quoted as good to about 0.1 % over its range and this is that claim
    // confirmed from outside it.
    assert!(
        worst_horizon < 7e-3,
        "Kasten & Young is off by {worst_horizon:.4e} below 0.5° apparent altitude"
    );
    assert!(
        worst_above < 2e-3,
        "Kasten & Young is off by {worst_above:.4e} above 0.5° apparent altitude"
    );
    // Young 1994 is the better of the two by an order of magnitude wherever it is defined — 0.05 %
    // at worst, and 8e-6 at 80° — which is not visible in a comparison of the two fits against each
    // other, because there they differ by the sum of their errors and neither is the reference.
    assert!(
        worst_young < 6e-4,
        "Young 1994 is off by {worst_young:.4e} where it is defined"
    );
}

/// The half degree of refracted visibility Young 1994 discards, priced.
///
/// `airmass_young_1994` returns `INFINITY` below the *true* horizon, and its doc comment says that
/// costs "38 air masses". That was an estimate from the other fit. Here it is the integrated value:
/// at the apparent horizon the true altitude is −0.548° and the ray carries 38.16 air masses, and a
/// quarter degree higher it is still −0.252° and 34.49.
#[test]
fn young_1994_refuses_a_ray_that_carries_thirty_eight_air_masses() {
    let rows = oracle();
    let (apparent_deg, air_mass, _, true_deg) = rows[0];
    // The equality is the point: the generator writes the apparent horizon as exactly 0.0, and a
    // row that is merely near it is a fixture whose grid has moved out from under this test.
    #[allow(clippy::float_cmp)]
    let at_the_horizon = apparent_deg == 0.0;
    assert!(at_the_horizon, "the first row should be the horizon");
    assert!(
        true_deg < 0.0,
        "a ray leaving horizontally comes from below the true horizon: {true_deg}°"
    );
    assert!(
        (air_mass - 38.159).abs() < 0.01,
        "integrated horizon air mass {air_mass}, against the 38 the doc comment quotes"
    );
    assert!(
        airmass_young_1994(TrueAltitude(true_deg.to_radians())).is_infinite(),
        "Young 1994 must refuse it rather than continue toward its pole at z = 93.6°"
    );
    assert!(
        airmass_kasten_young_1989(ApparentAltitude(apparent_deg.to_radians())).is_finite(),
        "Kasten & Young is in its published range at the apparent horizon"
    );
}

/// Both refraction fits run high, and the small-angle limit says it is theirs.
///
/// Above about 20° of altitude refraction is `(n₀ − 1) tan z` and nothing else — one constant, no
/// atmosphere model, no lapse rate. The oracle reproduces that to 0.13 % against Ciddor's (1996)
/// dry-air refractivity at 550 nm, which is the control: the ray geometry and the density profile
/// are not what is producing the gap. Against it Sæmundsson runs +4.8 % and Bennett +2.9 % at 50°
/// altitude — 2.3″ and 1.4″ — and both are still high at the horizon, +2.4 % and +3.3 %, which is
/// 47″ and 65″.
///
/// Neither fit is implemented wrongly; both are horizon-optimised, tightest at 2–3° altitude where
/// they agree with the integration to 0.4 % and 0.6 %, and neither was ever meant for the zenith —
/// at 90° both return a *negative* refraction. What this pins is that the module's fits carry a
/// systematic of a few percent that no comparison between them can reveal, because they share it.
#[test]
fn both_refraction_fits_run_high_against_a_first_principles_integration() {
    const REFRACTIVITY_550NM: f64 = 2.778_7e-4;
    let rows = oracle();

    // The control. At 85° altitude the integrated refraction divided by `tan z` must be the
    // sea-level refractivity, or the oracle is wrong before any fit is compared to it.
    let (_, _, refraction_arcmin, true_deg) = rows[17];
    assert!(
        (true_deg - 84.998).abs() < 0.01,
        "row 17 is 85°: {true_deg}"
    );
    let radians = (refraction_arcmin / 60.0).to_radians();
    let implied = radians / libm::tan((90.0 - true_deg).to_radians());
    assert!(
        ((implied - REFRACTIVITY_550NM) / REFRACTIVITY_550NM).abs() < 3e-3,
        "the small-angle limit implies a refractivity of {implied:.6e} against Ciddor's \
         {REFRACTIVITY_550NM:.6e}; the oracle's own geometry is what is wrong, not the fits"
    );

    let mut worst_saemundsson = 0.0_f64;
    let mut worst_bennett = 0.0_f64;
    let mut best_saemundsson = f64::INFINITY;
    for (apparent_deg, _, refraction_arcmin, true_deg) in rows {
        if refraction_arcmin <= 0.0 {
            continue; // the zenith row, where both fits are negative and the truth is zero
        }
        let saemundsson = USSA
            .apparent(TrueAltitude(true_deg.to_radians()))
            .map(|a| (a.0 - true_deg.to_radians()).to_degrees() * 60.0);
        let bennett = USSA
            .geometric(ApparentAltitude(apparent_deg.to_radians()))
            .map(|t| (apparent_deg.to_radians() - t.0).to_degrees() * 60.0);

        if let Some(value) = saemundsson {
            let excess = (value - refraction_arcmin) / refraction_arcmin;
            assert!(
                excess > 0.0,
                "Sæmundsson is below the integration at {apparent_deg}°, which no row does"
            );
            worst_saemundsson = worst_saemundsson.max(excess);
            best_saemundsson = best_saemundsson.min(excess);
        }
        if let Some(value) = bennett {
            let excess = (value - refraction_arcmin) / refraction_arcmin;
            assert!(
                excess > 0.0,
                "Bennett is below the integration at {apparent_deg}°, which no row does"
            );
            worst_bennett = worst_bennett.max(excess);
        }
    }

    // Measured: Sæmundsson 0.43 %–4.79 %, Bennett 0.58 %–3.27 %.
    assert!(
        (0.040..0.060).contains(&worst_saemundsson),
        "Sæmundsson's worst excess is {worst_saemundsson:.4}, outside the measured 4.79 %"
    );
    assert!(
        (0.025..0.040).contains(&worst_bennett),
        "Bennett's worst excess is {worst_bennett:.4}, outside the measured 3.27 %"
    );
    // The minimum is what says "horizon-optimised" rather than "uniformly biased": the fits are at
    // their tightest a few degrees up, not at the zenith where the physics is simplest.
    assert!(
        best_saemundsson < 0.006,
        "Sæmundsson's best agreement is {best_saemundsson:.4}; it should reach 0.43 % somewhere \
         in the 2°–3° band the fit was tuned on"
    );
}
