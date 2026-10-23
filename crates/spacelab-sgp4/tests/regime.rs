//! Which branch of the model an element set takes, and why it is worth asking.
//!
//! [`Regime`] reimplements the published branch conditions so a caller can ask them
//! without propagating. Reimplementing a condition is exactly the sort of thing that
//! silently drifts, so it is checked two ways: against the propagator's own internal
//! selector for the resonant cases, and against the deep-space period test for the
//! rest.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::{Convention, Regime, Satellite};

const CASES: &str = include_str!("fixtures/sgp4_ver_tle.txt");

/// The 33 published element sets as `(line1, line2)`.
fn tle_lines() -> Vec<(&'static str, &'static str)> {
    CASES
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|line| {
            let f: Vec<&str> = line.split('|').collect();
            (f[0], f[1])
        })
        .collect()
}

/// The propagator decides internally whether an object needs the resonance
/// integrator, and exposes that decision as `Constants::initial_state()`, which is
/// `Some` for exactly the deep-space *resonant* orbits.
///
/// Our [`Regime`] answers the same question from the published inequalities. If the
/// two ever disagree, one of them is reading a different mean motion — which is the
/// specific way this goes wrong, because the branch is expressed against the Brouwer
/// mean motion and a TLE prints the Kozai one.
#[test]
fn our_resonance_classification_matches_the_propagator_its_own_selector() {
    let lines = tle_lines();
    let mut checked = 0;
    let mut unparsed = 0;
    let mut synchronous = 0;
    let mut half_day = 0;

    for (l1, l2) in &lines {
        // The vendor parser validates the checksum and cannot be told not to, so it
        // refuses the four published sets that are deliberately corrupt or malformed.
        // Those are exactly the sets `Satellite::from_tle_unchecked` exists to load,
        // and they cannot take part in a cross-check against a propagator that will
        // not build them.
        let Ok(elements) = sgp4::Elements::from_tle(None, l1.as_bytes(), l2.as_bytes()) else {
            unparsed += 1;
            continue;
        };
        let Ok(constants) = sgp4::Constants::from_elements_afspc_compatibility_mode(&elements)
        else {
            continue;
        };
        let satellite = Satellite::from_tle_unchecked(l1, l2).expect("the elements just parsed");

        let theirs = constants.initial_state().is_some();
        let ours = satellite.regime().is_resonant();
        assert_eq!(
            ours,
            theirs,
            "catalog {}: we classify it {} (resonant = {ours}) but the propagator {} \
             allocate a resonance state",
            satellite.norad_id().get(),
            satellite.regime(),
            if theirs { "does" } else { "does not" },
        );

        checked += 1;
        match satellite.regime() {
            Regime::DeepSpaceSynchronous => synchronous += 1,
            Regime::DeepSpaceHalfDay => half_day += 1,
            _ => {}
        }
    }

    println!(
        "regime: {checked} element sets cross-checked ({synchronous} synchronous, \
         {half_day} half-day), {unparsed} refused by the vendor parser"
    );
    assert_eq!(
        checked + unparsed,
        lines.len(),
        "every published element set must be either cross-checked or accounted for",
    );
    // The published set was chosen to exercise both resonance windows. If either
    // count reaches zero the cross-check has stopped testing the branch it exists for.
    assert!(synchronous > 0, "no geosynchronous resonant case survived");
    assert!(half_day > 0, "no Molniya resonant case survived");
}

/// The deep-space branch is a period test, and the period is the one implied by the
/// Brouwer mean motion the accessor returns — not by the mean motion the TLE prints.
#[test]
fn the_deep_space_boundary_is_a_period_of_225_minutes() {
    for (l1, l2) in tle_lines() {
        let Ok(satellite) = Satellite::from_tle_unchecked(l1, l2) else {
            continue;
        };
        let period_min = core::f64::consts::TAU / satellite.brouwer_mean_motion();
        assert_eq!(
            satellite.regime().is_deep_space(),
            period_min >= 225.0,
            "catalog {} has a Brouwer period of {period_min:.3} min but is classified {}",
            satellite.norad_id().get(),
            satellite.regime(),
        );
    }
}

/// The 12-hour window has two conditions, and the second one is load-bearing.
///
/// Catalog 28129 (a GPS satellite) sits squarely inside the Molniya *mean-motion*
/// window and is still not resonant, because its orbit is very nearly circular. A
/// classifier that tested only the mean motion would put it on the resonance
/// integrator and get a different orbit.
#[test]
fn the_half_day_window_needs_eccentricity_as_well_as_mean_motion() {
    let (l1, l2) = tle_lines()
        .into_iter()
        .find(|(_, l2)| l2[2..7].trim() == "28129")
        .expect("catalog 28129 is in the published set");
    let satellite = Satellite::from_tle_unchecked(l1, l2).expect("28129 parses");

    let n = satellite.brouwer_mean_motion();
    assert!(
        (8.26e-3..=9.24e-3).contains(&n),
        "28129 should sit inside the 12-hour mean-motion window; n₀\" = {n:.6e} rad/min",
    );
    assert_eq!(
        satellite.regime(),
        Regime::DeepSpaceNonResonant,
        "28129 is inside the window but its eccentricity is far below 0.5",
    );
}

/// Every resonant orbit is a deep-space orbit; no near-Earth orbit is resonant.
#[test]
fn the_regimes_nest_the_way_the_model_does() {
    for regime in [
        Regime::NearEarth,
        Regime::DeepSpaceNonResonant,
        Regime::DeepSpaceSynchronous,
        Regime::DeepSpaceHalfDay,
    ] {
        assert!(!regime.is_resonant() || regime.is_deep_space());
        assert_eq!(regime.is_deep_space(), regime != Regime::NearEarth);
        // Every variant has a distinct, non-empty rendering.
        assert_ne!(format!("{regime}"), "");
    }
}

/// The Brouwer mean motion itself, against the published un-Kozai recursion.
///
/// Everything above hangs off this one number and checks it only by band: a 225-minute period
/// either side of the deep-space branch, catalog 28129 somewhere inside the 12-hour window. A
/// band admits a great many wrong numbers, and the conversion reads the *inclination*, so the
/// degrees-to-radians factor and the recursion itself are both unpinned by it. Transcribed from
/// SPACETRACK Report No. 3 (Hoots & Roehrich 1980) §appendix, the `a₁ → δ₁ → a₀ → δ₀ → n₀″`
/// sequence, with the WGS-72 constants as published there rather than as read back out of the
/// propagator.
#[test]
fn the_brouwer_mean_motion_is_the_published_un_kozai_recursion() {
    // kₑ from its WGS-72 definition rather than from the nine printed digits of the report:
    // `0.743669161e-1` is short by 4.5e-10 relative, which moves n₀" by 3e-13 — above the
    // tolerance a mutation test needs.
    const MU_WGS72: f64 = 398_600.8; // km³/s²
    const AE_WGS72: f64 = 6378.135; // km
    const J2: f64 = 1.082_616e-3;
    const DEG: f64 = core::f64::consts::PI / 180.0;
    const REV_PER_DAY: f64 = core::f64::consts::PI / 720.0;

    let mut checked = 0;
    for (l1, l2) in tle_lines() {
        let Ok(elements) = sgp4::Elements::from_tle(None, l1.as_bytes(), l2.as_bytes()) else {
            continue;
        };
        let Ok(satellite) = Satellite::from_elements(&elements, Convention::Reference) else {
            continue;
        };

        let (i, e) = (elements.inclination * DEG, elements.eccentricity);
        let n0 = elements.mean_motion * REV_PER_DAY;
        let ke = 60.0 * (MU_WGS72 / AE_WGS72.powi(3)).sqrt(); // earth-radii^{3/2} per minute
        let a1 = (ke / n0).powf(2.0 / 3.0);
        let p0 = 0.75 * J2 * (3.0 * i.cos().powi(2) - 1.0) / (1.0 - e * e).powf(1.5);
        let d1 = p0 / (a1 * a1);
        let a0 = a1 * (1.0 - d1 / 3.0 - d1 * d1 - (134.0 / 81.0) * d1 * d1 * d1);
        let want = n0 / (1.0 + p0 / (a0 * a0));

        let got = satellite.brouwer_mean_motion();
        assert!(
            (got - want).abs() <= 1e-14 * want,
            "catalog {}: n₀\" = {got:.17e} rad/min, un-Kozai says {want:.17e}",
            satellite.norad_id().get(),
        );
        checked += 1;
    }
    assert!(
        checked >= 29,
        "only {checked} element sets reached the recursion"
    );
}
