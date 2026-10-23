//! Constructor + atmosphere-geometry surface of the DSST Gaussian forces (drag, SRP). The Orekit-golden
//! inline tests drive `with_rbar` and a single full-circle orbit; these pin the default `new`
//! constructor, the custom-reference SRP constructor, and the two drag integration-limit regimes the
//! golden orbit never enters — a trajectory entirely above the atmosphere, and one straddling `rbar`.
//! The oracle is the analytic geometry of the atmosphere arc, verified on real LEO equinoctial states.

#![allow(clippy::float_cmp)] // the above-atmosphere guard returns a literal exact [0.0, 0.0]

use spacelab_propagation::dsst::DsstAtmosphericDrag;
use spacelab_propagation::{DsstAuxiliary, DsstSolarRadiationPressure};

const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_137.0;
// DsstAtmosphericDrag::new uses rbar = 1000 km + WGS84 equatorial radius.
const DEFAULT_RBAR: f64 = 7_378_137.0;

/// Equinoctial auxiliary state from classical-ish inputs: `a`, eccentricity vector `(ex, ey)`, node
/// vector left at zero (planar), and true longitude `lm`.
fn aux(a: f64, ex: f64, ey: f64) -> DsstAuxiliary {
    DsstAuxiliary::new(a, ex, ey, 0.0, 0.0, 0.5, MU)
}

#[test]
fn drag_limits_vanish_above_the_atmosphere() {
    // perigee = a(1−e). With a = 8000 km and e ≈ 0.007, perigee ≈ 7943 km > rbar (7378 km): the whole
    // orbit is above the atmosphere, so there is no arc to integrate — both limits collapse to zero.
    let drag = DsstAtmosphericDrag::new(2.2, 10.0 / 1000.0, MU);
    let ll = drag.l_limits(&aux(8_000_000.0, 0.005, 0.005));
    assert_eq!(ll, [0.0, 0.0]);
}

#[test]
fn drag_limits_are_a_partial_arc_when_the_orbit_straddles_rbar() {
    // a = rbar, e = 0.05 ⇒ perigee ≈ 7009 km < rbar < apogee ≈ 7747 km: the orbit dips into and climbs
    // out of the atmosphere, so the limits bound a *proper* arc — a non-empty span strictly under a full
    // revolution (unlike the golden full-circle orbit whose span is exactly 2π).
    let drag = DsstAtmosphericDrag::new(2.2, 10.0 / 1000.0, MU);
    let ll = drag.l_limits(&aux(DEFAULT_RBAR, 0.03, 0.04));
    let span = ll[1] - ll[0];
    assert!(span > 0.0, "entry must precede exit: {ll:?}");
    assert!(
        span < 2.0 * core::f64::consts::PI,
        "a straddling orbit spans less than a full revolution: {span}"
    );
}

#[test]
fn srp_with_reference_builds_a_working_force() {
    // The custom-reference constructor (explicit d_ref / p_ref) must yield a usable force: finite
    // mean-element rates for a lit LEO with the Sun ~1 AU up the x-axis.
    const AU: f64 = 1.495_978_707e11;
    const P_REF: f64 = 4.56e-6;
    let srp = DsstSolarRadiationPressure::with_reference(1.5, 10.0 / 1000.0, MU, RE, AU, P_REF);
    let rates = srp.mean_element_rates(&aux(7_000_000.0, 0.01, 0.0), [AU, 0.0, 0.0]);
    assert!(rates.iter().all(|r| r.is_finite()), "rates: {rates:?}");
    // SRP perturbs a LEO's semi-major axis at the sub-µm/s level, not zero and not divergent.
    assert!(rates[0].abs() < 1.0e-3);
}
