//! Accessor + error surface of the analytical mean-element propagators (Brouwer–Lyddane,
//! Eckstein–Hechler). The golden tests drive `propagate`; these pin the query methods
//! (`mean_elements`, `osculating_at`, `reference_radius`, `c20`) and the Eckstein–Hechler rejection /
//! `Display` paths that the happy path never touches. The oracle is internal self-consistency and the
//! documented model contracts, verified on a real LEO orbit (a = 7200 km, e = 1e-3, i = 98°).

#![allow(clippy::float_cmp, clippy::unwrap_used)]

use spacelab_core::Angle;
use spacelab_elements::{BrouwerLyddane, EcksteinHechler, EcksteinHechlerError, KeplerianElements};

const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
// Un-normalized zonal coefficients Cn0 (verbatim from the Orekit oracle runs of both propagators).
const C20: f64 = -1.082_635_902_887_277e-3;
const C30: f64 = 2.532_440_651_276_641e-6;
const C40: f64 = 1.619_933_508_937_329_4e-6;
const C50: f64 = 2.276_461_426_648_441e-7;
const C60: f64 = -5.407_738_861_511_067e-7;

fn leo() -> KeplerianElements {
    KeplerianElements {
        semi_major_axis: 7_200_000.0,
        eccentricity: 1.0e-3,
        inclination: Angle::from_radians(98.0_f64.to_radians()),
        raan: Angle::from_radians(30.0_f64.to_radians()),
        arg_periapsis: Angle::from_radians(40.0_f64.to_radians()),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    }
    .at_mean_anomaly(Angle::from_radians(50.0_f64.to_radians()))
}

#[test]
fn brouwer_lyddane_accessors_are_self_consistent() {
    let bl = BrouwerLyddane::from_osculating(&leo(), RE, C20, C30, C40, C50).unwrap();

    // The build parameters round-trip through the accessors exactly.
    assert_eq!(bl.reference_radius(), RE);
    assert_eq!(bl.c20(), C20);

    // The secular mean semi-major axis differs from the osculating one only by the J2 short-period `a`
    // amplitude (~9.2 km here), which the model strips out — so it stays within one such band.
    let mean = bl.mean_elements();
    assert!((mean.semi_major_axis - 7_200_000.0).abs() < 12_000.0);

    // `propagate` is by construction the Cartesian conversion of `osculating_at` — they must agree to
    // the bit, at t = 0 and after a partial orbit.
    for dt in [0.0, 900.0] {
        let (r, v) = bl.propagate(dt);
        let (r2, v2) = bl.osculating_at(dt).to_state();
        assert_eq!(r, r2, "position mismatch at dt = {dt}");
        assert_eq!(v, v2, "velocity mismatch at dt = {dt}");
    }
    // osculating_at(0) sits on the initial osculating sphere (~7200 km radius).
    let r0 = bl.osculating_at(0.0).to_state().0.norm();
    assert!((r0 - 7_200_000.0).abs() < 20_000.0);
}

#[test]
fn eckstein_hechler_accessors_track_the_propagated_state() {
    let eh = EcksteinHechler::from_osculating(&leo(), RE, C20, C30, C40, C50, C60).unwrap();

    assert_eq!(eh.reference_radius(), RE);

    // The osculating elements at dt convert to the same *position* the propagator reports (the velocity
    // differs: `propagate` carries the short-period time derivative, the Keplerian state does not).
    for dt in [0.0, 900.0, 3600.0] {
        let (r_prop, _) = eh.propagate(dt);
        let (r_osc, _) = eh.osculating_at(dt).to_state();
        assert!(
            (r_prop - r_osc).norm() < 1.0e-3,
            "EH osculating_at vs propagate position at dt = {dt}"
        );
    }
    // Sanity: a near-circular 7200 km orbit stays near that radius.
    let a0 = eh.osculating_at(0.0).semi_major_axis;
    assert!((a0 - 7_200_000.0).abs() < 20_000.0);
}

#[test]
fn eckstein_hechler_rejects_an_equatorial_orbit() {
    // sin²i → 0 makes the Eckstein–Hechler zonal expansion singular; the constructor must reject it.
    let equatorial = KeplerianElements {
        inclination: Angle::from_radians(1.0e-6),
        ..leo()
    };
    let result = EcksteinHechler::from_osculating(&equatorial, RE, C20, C30, C40, C50, C60);
    assert!(matches!(
        result,
        Err(EcksteinHechlerError::AlmostEquatorial { .. })
    ));
}

#[test]
fn eckstein_hechler_error_display_names_the_failure() {
    let equatorial = EcksteinHechlerError::AlmostEquatorial {
        inclination_deg: 0.0,
    };
    let critical = EcksteinHechlerError::AlmostCriticallyInclined {
        inclination_deg: 63.4,
    };
    let eccentric = EcksteinHechlerError::TooLargeEccentricity { eccentricity: 0.2 };
    assert!(format!("{equatorial}").contains("equatorial"));
    assert!(format!("{critical}").contains("critically inclined"));
    assert!(format!("{eccentric}").contains("eccentricity"));
}
