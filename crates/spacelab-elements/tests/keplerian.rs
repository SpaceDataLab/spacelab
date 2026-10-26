//! Keplerian elements, checked by the identities they must satisfy — not a reference table.
//!
//! * **Round-trip** `state → elements → state` reproduces the state, including the circular,
//!   equatorial and circular-equatorial singular cases where a naïve conversion returns `NaN`.
//! * **Invariance**: the osculating elements of a two-body orbit are constant except the anomaly.
//! * **Cross-check**: advancing the anomaly in closed form (`propagate`) agrees with numerically
//!   integrating the two-body ODE — two independent implementations of the same flow.
//!
//! The gates are `kepler_s_equation_is_solved_to_its_conditioning` (G2) and
//! `the_conversions_match_the_element_map_at_60_digits` (G1), against
//! `derivations/keplerian_elements.gp`; `derivations/keplerian_elements.mac` proves `from_state`
//! inverts `to_state` identically, branch by branch.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
// The 60-digit reference rows are pasted as the derivation prints them.
#![allow(
    clippy::unreadable_literal,
    clippy::excessive_precision,
    clippy::type_complexity
)]
#![allow(clippy::many_single_char_names)]

use spacelab_core::{Angle, Vec3};
use spacelab_elements::KeplerianElements;
use spacelab_propagation::driver::{Controls, integrate};
use spacelab_propagation::forces::{Newtonian, Plain, PointMass};
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;

fn deg(d: f64) -> Angle {
    Angle::from_degrees(d)
}

fn elements(a: f64, e: f64, i: f64, raan: f64, argp: f64, nu: f64) -> KeplerianElements {
    KeplerianElements {
        semi_major_axis: a,
        eccentricity: e,
        inclination: deg(i),
        raan: deg(raan),
        arg_periapsis: deg(argp),
        true_anomaly: deg(nu),
        mu: MU,
    }
}

/// Relative difference of two vectors, scaled by the reference length.
fn rel(a: Vec3, b: Vec3) -> f64 {
    (a - b).norm() / b.norm()
}

/// A spread of orbit types: near-circular LEO, SSO, Molniya, near-GEO, retrograde, polar.
fn cases() -> [KeplerianElements; 6] {
    [
        elements(7_000e3, 0.001, 51.6, 40.0, 30.0, 10.0),
        elements(7_200e3, 0.1, 98.0, 120.0, 200.0, 45.0),
        elements(26_600e3, 0.74, 63.4, 10.0, 270.0, 100.0),
        elements(42_164e3, 0.0005, 0.05, 80.0, 60.0, 300.0),
        elements(8_000e3, 0.3, 145.0, 200.0, 15.0, 250.0),
        elements(7_500e3, 0.02, 89.9, 5.0, 350.0, 123.0),
    ]
}

#[test]
fn state_to_elements_and_back_reproduces_the_state() {
    for els in cases() {
        let (r, v) = els.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        let (r2, v2) = back.to_state();
        assert!(
            rel(r2, r) < 1e-14,
            "position round-trip {:.2e} for {els:?}",
            rel(r2, r)
        );
        assert!(
            rel(v2, v) < 1e-14,
            "velocity round-trip {:.2e} for {els:?}",
            rel(v2, v)
        );

        // for the non-singular cases the elements themselves round-trip too
        if els.eccentricity > 1e-3 && els.inclination.degrees() > 1.0 {
            assert!((back.semi_major_axis - els.semi_major_axis).abs() < 1e-3);
            assert!((back.eccentricity - els.eccentricity).abs() < 1e-12);
            assert!((back.inclination.radians() - els.inclination.radians()).abs() < 1e-12);
            assert!((back.raan.radians() - els.raan.radians()).abs() < 1e-12);
            assert!((back.arg_periapsis.radians() - els.arg_periapsis.radians()).abs() < 1e-11);
            assert!((back.true_anomaly.radians() - els.true_anomaly.radians()).abs() < 1e-11);
        }
    }
}

#[test]
fn the_singular_cases_round_trip_without_nan() {
    // ω undefined (circular inclined), Ω undefined (equatorial elliptical), both (circular
    // equatorial): the geometry is carried in the anomaly, and the state must still reconstruct.
    //
    // Each branch resolves its quadrant from the sign of one component — r_z, e_y, r·v, r_y — and a
    // single sample per branch exercises only one side of that test, leaving the correction that
    // handles the other side free to be anything. The anomaly and the periapsis longitude are swept
    // over the full turn so every sign is visited.
    let mut singular = Vec::new();
    for j in 0..16 {
        let nu = 360.0 * f64::from(j) / 16.0;
        singular.push(elements(7_000e3, 0.0, 45.0, 30.0, 0.0, nu)); // circular inclined
        singular.push(elements(7_000e3, 0.0, 0.0, 0.0, 0.0, nu)); // circular equatorial
        // Retrograde equatorial. `i = 180°` takes the same no-node branch as `i = 0`, but the
        // substitute longitude is measured the other way round; sweeping only the prograde side
        // left that reversal free to be wrong, and it was — the state came back reflected in y.
        singular.push(elements(7_000e3, 0.0, 180.0, 0.0, 0.0, nu));
        for k in 0..8 {
            // elliptical equatorial: ω is the longitude of periapsis, and its sign matters too
            singular.push(elements(7_000e3, 0.2, 0.0, 0.0, 45.0 * f64::from(k), nu));
            singular.push(elements(7_000e3, 0.2, 180.0, 0.0, 45.0 * f64::from(k), nu));
        }
    }
    for els in singular {
        let (r, v) = els.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        assert!(back.semi_major_axis.is_finite() && back.eccentricity.is_finite());
        let (r2, v2) = back.to_state();
        assert!(
            rel(r2, r) < 1e-14,
            "singular position round-trip {:.2e} for {els:?}",
            rel(r2, r)
        );
        assert!(
            rel(v2, v) < 1e-14,
            "singular velocity round-trip {:.2e} for {els:?}",
            rel(v2, v)
        );
    }
}

#[test]
fn mean_and_true_anomaly_are_mutual_inverses() {
    // ν → M → ν over a range of eccentricities and anomalies (Kepler's equation solved back).
    for &e in &[0.0, 0.01, 0.3, 0.74, 0.95] {
        for nu_deg in [1.0, 47.0, 123.0, 200.0, 340.0] {
            let els = elements(1e7, e, 30.0, 0.0, 0.0, nu_deg);
            let m = els.mean_anomaly();
            let recovered = els.at_mean_anomaly(m).true_anomaly.radians();
            let want = els.true_anomaly.radians();
            let d = (recovered - want).rem_euclid(Angle::TAU);
            let d = d.min(Angle::TAU - d);
            assert!(d < 1e-12, "ν→M→ν off by {d:.2e} at e={e}, ν={nu_deg}°");
        }
    }
}

#[test]
fn period_matches_the_closed_form_for_a_known_orbit() {
    // 700 km circular: T = 2π√(a³/μ) ≈ 5926 s.
    let els = elements(RE + 700e3, 0.0, 51.6, 0.0, 0.0, 0.0);
    let a = RE + 700e3;
    let want = Angle::TAU * (a * a * a / MU).sqrt();
    assert!(
        (els.period() - want).abs() < 1e-6,
        "period {} vs {want}",
        els.period()
    );
    assert!(
        (5900.0..5950.0).contains(&els.period()),
        "period {} s out of range",
        els.period()
    );
}

#[test]
fn osculating_elements_are_invariant_along_a_two_body_orbit() {
    // Under pure two-body motion a, e, i, Ω, ω are constants of the motion; only ν advances.
    let els0 = elements(9_000e3, 0.15, 55.0, 70.0, 130.0, 20.0);
    let (r0, v0) = els0.to_state();
    let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];

    let traj = integrate(
        &Dop853,
        &Plain(Newtonian(PointMass { mu: MU })),
        0.0,
        &y0,
        3.0 * els0.period(),
        60.0,
        Controls::new(Tolerance::new(1e-13, 1e-13), 1e-3),
    )
    .expect("propagated");

    let yf = traj.final_state();
    let els = KeplerianElements::from_state(
        Vec3::new(yf[0], yf[1], yf[2]),
        Vec3::new(yf[3], yf[4], yf[5]),
        MU,
    );
    assert!(
        (els.semi_major_axis - els0.semi_major_axis).abs() < 1e-3,
        "a drifted"
    );
    assert!(
        (els.eccentricity - els0.eccentricity).abs() < 1e-10,
        "e drifted"
    );
    assert!(
        (els.inclination.radians() - els0.inclination.radians()).abs() < 1e-10,
        "i drifted"
    );
    assert!(
        (els.raan.radians() - els0.raan.radians()).abs() < 1e-10,
        "Ω drifted"
    );
    assert!(
        (els.arg_periapsis.radians() - els0.arg_periapsis.radians()).abs() < 1e-9,
        "ω drifted"
    );
}

#[test]
fn closed_form_propagation_matches_the_numerical_two_body_flow() {
    // The analytic Kepler flow (advance M, solve back to ν) and numerically integrating ẍ = −μr/r³
    // are two implementations of the same trajectory; they must agree to the integrator tolerance.
    for els0 in cases() {
        let (r0, v0) = els0.to_state();
        let y0 = [r0.x, r0.y, r0.z, v0.x, v0.y, v0.z];
        let dt = 0.37 * els0.period(); // an arbitrary fraction of an orbit

        let (r_a, v_a) = els0.propagate(dt).to_state();

        let traj = integrate(
            &Dop853,
            &Plain(Newtonian(PointMass { mu: MU })),
            0.0,
            &y0,
            dt,
            60.0,
            Controls::new(Tolerance::new(1e-13, 1e-13), 1e-3),
        )
        .expect("propagated");
        let yf = traj.final_state();
        let r_n = Vec3::new(yf[0], yf[1], yf[2]);
        let v_n = Vec3::new(yf[3], yf[4], yf[5]);

        assert!(
            rel(r_a, r_n) < 1e-9,
            "position mismatch {:.2e} for {els0:?}",
            rel(r_a, r_n)
        );
        assert!(
            rel(v_a, v_n) < 1e-9,
            "velocity mismatch {:.2e} for {els0:?}",
            rel(v_a, v_n)
        );
    }
}

// Ω is compared with `==`: below the threshold the convention is that it is set to the literal
// zero, and "approximately folded away" is not a thing the convention can mean.
#[allow(clippy::float_cmp)]
#[test]
fn the_equatorial_classification_is_scale_free() {
    // The equatorial test is `|n| ≤ ε·|h|`, not `|n| ≤ ε`. Since `|n| = |h| sin i`, dividing by `|h|`
    // is exactly what makes the classification a statement about the inclination instead of about
    // the orbit's size. No round-trip can see the difference: Ω and the argument of latitude are
    // both measured from the same node vector, so any node direction rebuilds the same state. The
    // reported Ω can — below the threshold the convention is Ω = 0, and it has to hold at every
    // orbit radius, from LEO to geostationary.
    for &a in &[7_000e3, 42_164e3] {
        // sin i = 1e-13 is two decades under the 1e-11 threshold and eight above the node vector's
        // own round-off (|n| ≈ 5e-3 m²/s against ~1e-5 of cancellation error inside h).
        let flat = elements(a, 0.2, 1e-13_f64.to_degrees(), 30.0, 40.0, 70.0);
        let (r, v) = flat.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        assert_eq!(
            back.raan.radians(),
            0.0,
            "a={a}: Ω was not folded into the longitude of periapsis"
        );
        assert!(
            (back.arg_periapsis.degrees() - 70.0).abs() < 1e-6,
            "a={a}: ϖ = {} should be Ω+ω",
            back.arg_periapsis.degrees()
        );

        // Two decades above the threshold the node is real and must be reported.
        let tilted = elements(a, 0.2, 1e-9_f64.to_degrees(), 30.0, 40.0, 70.0);
        let (r, v) = tilted.to_state();
        let back = KeplerianElements::from_state(r, v, MU);
        assert!(
            (back.raan.degrees() - 30.0).abs() < 1e-6,
            "a={a}: Ω = {} should be 30°",
            back.raan.degrees()
        );
    }
}

/// `(e, M, ν)`: Kepler's equation `M = E − e sin E` solved at 60 digits by bracketed root finding
/// and mapped to the true anomaly, from `derivations/keplerian_elements.gp`. Dyadic eccentricities
/// up to `1 − 2⁻¹⁶`, the double nearest `0.8` where the solver's starting guess switches, and mean
/// anomalies from `2⁻²⁰` to just below a turn.
#[rustfmt::skip]
const KEPLER: [(f64, f64, f64); 144] = [
    (0.0, 0.0, 0.0),
    (0.0, 0.5000000000000000000000000, 0.5000000000000000000000000),
    (0.0, 1.000000000000000000000000, 1.000000000000000000000000),
    (0.0, 1.500000000000000000000000, 1.500000000000000000000000),
    (0.0, 2.000000000000000000000000, 2.000000000000000000000000),
    (0.0, 2.500000000000000000000000, 2.500000000000000000000000),
    (0.0, 3.000000000000000000000000, 3.000000000000000000000000),
    (0.0, 3.500000000000000000000000, 3.500000000000000000000000),
    (0.0, 4.000000000000000000000000, 4.000000000000000000000000),
    (0.0, 4.500000000000000000000000, 4.500000000000000000000000),
    (0.0, 5.000000000000000000000000, 5.000000000000000000000000),
    (0.0, 5.500000000000000000000000, 5.500000000000000000000000),
    (0.0, 6.000000000000000000000000, 6.000000000000000000000000),
    (0.0, 9.536743164062500000000000e-7, 9.536743164062500000000000e-7),
    (0.0, 0.0009765625000000000000000000, 0.0009765625000000000000000000),
    (0.0, 6.281250000000000000000000, 6.281250000000000000000000),
    (0.06250000000000000000000000, 0.0, 0.0),
    (0.06250000000000000000000000, 0.5000000000000000000000000, 0.5642804969883523423279559),
    (0.06250000000000000000000000, 1.000000000000000000000000, 1.109589958787130051016435),
    (0.06250000000000000000000000, 1.500000000000000000000000, 1.625052685064115637439763),
    (0.06250000000000000000000000, 2.000000000000000000000000, 2.109858482369625276786066),
    (0.06250000000000000000000000, 2.500000000000000000000000, 2.570335557222411502784580),
    (0.06250000000000000000000000, 3.000000000000000000000000, 3.016369648141963867981439),
    (0.06250000000000000000000000, 3.500000000000000000000000, 3.459159900270235690370195),
    (0.06250000000000000000000000, 4.000000000000000000000000, 3.910124432924005968934002),
    (0.06250000000000000000000000, 4.500000000000000000000000, 4.380077133634711173414988),
    (0.06250000000000000000000000, 5.000000000000000000000000, 4.877726645293801644078986),
    (0.06250000000000000000000000, 5.500000000000000000000000, 5.406787602735119464174285),
    (0.06250000000000000000000000, 6.000000000000000000000000, 5.962259795963604519908192),
    (0.06250000000000000000000000, 9.536743164062500000000000e-7, 1.082948094527923230555587e-6),
    (0.06250000000000000000000000, 0.0009765625000000000000000000, 0.001108938822057162472666374),
    (0.06250000000000000000000000, 6.281250000000000000000000, 6.280987662776323210470839),
    (0.3125000000000000000000000, 0.0, 0.0),
    (0.3125000000000000000000000, 0.5000000000000000000000000, 0.9364190127934519037824647),
    (0.3125000000000000000000000, 1.000000000000000000000000, 1.621179432976294818829452),
    (0.3125000000000000000000000, 1.500000000000000000000000, 2.102510794190252363765401),
    (0.3125000000000000000000000, 2.000000000000000000000000, 2.471066841399434986355563),
    (0.3125000000000000000000000, 2.500000000000000000000000, 2.780920175734597733407639),
    (0.3125000000000000000000000, 3.000000000000000000000000, 3.063442551679978813106773),
    (0.3125000000000000000000000, 3.500000000000000000000000, 3.340406682539196325539618),
    (0.3125000000000000000000000, 4.000000000000000000000000, 3.631760935460470668823223),
    (0.3125000000000000000000000, 4.500000000000000000000000, 3.962272031446383958395243),
    (0.3125000000000000000000000, 5.000000000000000000000000, 4.371104784414813944562326),
    (0.3125000000000000000000000, 5.500000000000000000000000, 4.927670172023661045622124),
    (0.3125000000000000000000000, 6.000000000000000000000000, 5.727885408583038356481882),
    (0.3125000000000000000000000, 9.536743164062500000000000e-7, 1.916640624188665276882949e-6),
    (0.3125000000000000000000000, 0.0009765625000000000000000000, 0.001962639399169988259341354),
    (0.3125000000000000000000000, 6.281250000000000000000000, 6.279295841097313842245634),
    (0.8000000000000000444089210, 0.0, 0.0),
    (0.8000000000000000444089210, 0.5000000000000000000000000, 2.285737410055886544768196),
    (0.8000000000000000444089210, 1.000000000000000000000000, 2.615308495307799660177934),
    (0.8000000000000000444089210, 1.500000000000000000000000, 2.790482668487561227955639),
    (0.8000000000000000444089210, 2.000000000000000000000000, 2.916229170786917861676650),
    (0.8000000000000000444089210, 2.500000000000000000000000, 3.020472570854204638126191),
    (0.8000000000000000444089210, 3.000000000000000000000000, 3.115347719258455422985816),
    (0.8000000000000000444089210, 3.500000000000000000000000, 3.208357958387560204953666),
    (0.8000000000000000444089210, 4.000000000000000000000000, 3.306223120171490227219745),
    (0.8000000000000000444089210, 4.500000000000000000000000, 3.417795681220379140979714),
    (0.8000000000000000444089210, 5.000000000000000000000000, 3.559719898159626681577348),
    (0.8000000000000000444089210, 5.500000000000000000000000, 3.777733087582390948892454),
    (0.8000000000000000444089210, 6.000000000000000000000000, 4.326807183568686243559553),
    (0.8000000000000000444089210, 9.536743164062500000000000e-7, 1.430511474566007407205745e-5),
    (0.8000000000000000444089210, 0.0009765625000000000000000000, 0.01464797186813134985343213),
    (0.8000000000000000444089210, 6.281250000000000000000000, 6.254159322849944164798646),
    (0.8125000000000000000000000, 0.0, 0.0),
    (0.8125000000000000000000000, 0.5000000000000000000000000, 2.324443190056628866190023),
    (0.8125000000000000000000000, 1.000000000000000000000000, 2.638144920378090655407745),
    (0.8125000000000000000000000, 1.500000000000000000000000, 2.805417226556306568709207),
    (0.8125000000000000000000000, 2.000000000000000000000000, 2.925713807179051778075424),
    (0.8125000000000000000000000, 2.500000000000000000000000, 3.025541550403409743734693),
    (0.8125000000000000000000000, 3.000000000000000000000000, 3.116443650587666570917264),
    (0.8125000000000000000000000, 3.500000000000000000000000, 3.205568302682191414894888),
    (0.8125000000000000000000000, 4.000000000000000000000000, 3.299319730578131520825888),
    (0.8125000000000000000000000, 4.500000000000000000000000, 3.406126820195799524593994),
    (0.8125000000000000000000000, 5.000000000000000000000000, 3.541808622584309124472040),
    (0.8125000000000000000000000, 5.500000000000000000000000, 3.749737551210174736149164),
    (0.8125000000000000000000000, 6.000000000000000000000000, 4.271962752738027069467413),
    (0.8125000000000000000000000, 9.536743164062500000000000e-7, 1.581383438574942805707140e-5),
    (0.8125000000000000000000000, 0.0009765625000000000000000000, 0.01619273195524844403975355),
    (0.8125000000000000000000000, 6.281250000000000000000000, 6.251098966462210992641627),
    (0.9492187500000000000000000, 0.0, 0.0),
    (0.9492187500000000000000000, 0.5000000000000000000000000, 2.778060220314020300867909),
    (0.9492187500000000000000000, 1.000000000000000000000000, 2.911438645333870006854804),
    (0.9492187500000000000000000, 1.500000000000000000000000, 2.986214154956962466824954),
    (0.9492187500000000000000000, 2.000000000000000000000000, 3.041260873499978762961379),
    (0.9492187500000000000000000, 2.500000000000000000000000, 3.087502812038723940112179),
    (0.9492187500000000000000000, 3.000000000000000000000000, 3.129857902111114950341117),
    (0.9492187500000000000000000, 3.500000000000000000000000, 3.171435223314386705497048),
    (0.9492187500000000000000000, 4.000000000000000000000000, 3.215034471167348912720049),
    (0.9492187500000000000000000, 4.500000000000000000000000, 3.264294919832415179651902),
    (0.9492187500000000000000000, 5.000000000000000000000000, 3.325870588448338330609621),
    (0.9492187500000000000000000, 5.500000000000000000000000, 3.417290559514763085493451),
    (0.9492187500000000000000000, 6.000000000000000000000000, 3.630130805158084404779911),
    (0.9492187500000000000000000, 9.536743164062500000000000e-7, 0.0001163523864772944090919755),
    (0.9492187500000000000000000, 0.0009765625000000000000000000, 0.1188715330619081256486773),
    (0.9492187500000000000000000, 6.281250000000000000000000, 6.049169182156763143944453),
    (0.9892578125000000000000000, 0.0, 0.0),
    (0.9892578125000000000000000, 0.5000000000000000000000000, 2.981887730984650832035369),
    (0.9892578125000000000000000, 1.000000000000000000000000, 3.039563924115248852401729),
    (0.9892578125000000000000000, 1.500000000000000000000000, 3.072476945396299429896424),
    (0.9892578125000000000000000, 2.000000000000000000000000, 3.096887425679905050989619),
    (0.9892578125000000000000000, 2.500000000000000000000000, 3.117470810608450299222677),
    (0.9892578125000000000000000, 3.000000000000000000000000, 3.136357667082005262831063),
    (0.9892578125000000000000000, 3.500000000000000000000000, 3.154904496785621589238747),
    (0.9892578125000000000000000, 4.000000000000000000000000, 3.174334893746285839711618),
    (0.9892578125000000000000000, 4.500000000000000000000000, 3.196232747053467693042567),
    (0.9892578125000000000000000, 5.000000000000000000000000, 3.223467672583362801972076),
    (0.9892578125000000000000000, 5.500000000000000000000000, 3.263484106963895679156500),
    (0.9892578125000000000000000, 6.000000000000000000000000, 3.353768645549584081555160),
    (0.9892578125000000000000000, 9.536743164062500000000000e-7, 0.001208109849454522393266429),
    (0.9892578125000000000000000, 0.0009765625000000000000000000, 1.021864419357387685105723),
    (0.9892578125000000000000000, 6.281250000000000000000000, 4.766589776956174902867842),
    (0.9990234375000000000000000, 0.0, 0.0),
    (0.9990234375000000000000000, 0.5000000000000000000000000, 3.093972080833340704265157),
    (0.9990234375000000000000000, 1.000000000000000000000000, 3.111102169247478080925082),
    (0.9990234375000000000000000, 1.500000000000000000000000, 3.120920704683830519424855),
    (0.9990234375000000000000000, 2.000000000000000000000000, 3.128216155106073522278801),
    (0.9990234375000000000000000, 2.500000000000000000000000, 3.134373500381786864803853),
    (0.9990234375000000000000000, 3.000000000000000000000000, 3.140025804284620768177657),
    (0.9990234375000000000000000, 3.500000000000000000000000, 3.145576844662670109457162),
    (0.9990234375000000000000000, 4.000000000000000000000000, 3.151390992193021541703381),
    (0.9990234375000000000000000, 4.500000000000000000000000, 3.157939416828760235374899),
    (0.9990234375000000000000000, 5.000000000000000000000000, 3.166073804397373660508826),
    (0.9990234375000000000000000, 5.500000000000000000000000, 3.177995018703441523041818),
    (0.9990234375000000000000000, 6.000000000000000000000000, 3.204679519381349503993738),
    (0.9990234375000000000000000, 9.536743164062500000000000e-7, 0.04416902345287737798519458),
    (0.9990234375000000000000000, 0.0009765625000000000000000000, 2.632792208494564797945325),
    (0.9990234375000000000000000, 6.281250000000000000000000, 3.540064787944180798474640),
    (0.9999847412109375000000000, 0.0, 0.0),
    (0.9999847412109375000000000, 0.5000000000000000000000000, 3.135646580866509004297307),
    (0.9999847412109375000000000, 1.000000000000000000000000, 3.137784670425886095711702),
    (0.9999847412109375000000000, 1.500000000000000000000000, 3.139010703500960705939022),
    (0.9999847412109375000000000, 2.000000000000000000000000, 3.139921845637256785507545),
    (0.9999847412109375000000000, 2.500000000000000000000000, 3.140690917670429354648916),
    (0.9999847412109375000000000, 3.000000000000000000000000, 3.141396938697410560801087),
    (0.9999847412109375000000000, 3.500000000000000000000000, 3.142090317156000189808636),
    (0.9999847412109375000000000, 4.000000000000000000000000, 3.142816543717474764919356),
    (0.9999847412109375000000000, 4.500000000000000000000000, 3.143634436578882448197043),
    (0.9999847412109375000000000, 5.000000000000000000000000, 3.144650291231891091185002),
    (0.9999847412109375000000000, 5.500000000000000000000000, 3.146138682395996158942063),
    (0.9999847412109375000000000, 6.000000000000000000000000, 3.149467697659854793052239),
    (0.9999847412109375000000000, 9.536743164062500000000000e-7, 2.483788576876902465998664),
    (0.9999847412109375000000000, 0.0009765625000000000000000000, 3.080468289271376302487425),
    (0.9999847412109375000000000, 6.281250000000000000000000, 3.190152473293516734839835),
];

/// The angular distance between two angles, folded into `[0, π]`.
fn turn_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(Angle::TAU);
    d.min(Angle::TAU - d)
}

/// (G2) `at_mean_anomaly` lands on the true anomaly of the root of Kepler's equation, solved
/// independently at 60 digits, to within what the problem's own conditioning allows.
///
/// The bound is 4 ulps of `max(M, 1)` carried through `dν/dM = (1 + e cos ν)²/(1 − e²)^{3/2}` —
/// the factor that turns one rounding of the mean anomaly into true anomaly, and is 2.4e7 at
/// perigee for `e = 1 − 2⁻¹⁶` — plus 4 ulps of `2π` for the final `atan2` and wrap. The bound is the
/// problem's, not the solver's, so it is the same for every method; the worst row measures 0.094
/// of it.
#[test]
fn kepler_s_equation_is_solved_to_its_conditioning() {
    let mut worst = 0.0_f64;
    for (e, m, nu) in KEPLER {
        let els = KeplerianElements {
            semi_major_axis: 1e7,
            eccentricity: e,
            inclination: deg(30.0),
            raan: deg(0.0),
            arg_periapsis: deg(0.0),
            true_anomaly: deg(0.0),
            mu: MU,
        };
        let got = els
            .at_mean_anomaly(Angle::from_radians(m))
            .true_anomaly
            .radians();
        let cond = (1.0 + e * nu.cos()).powi(2) / (1.0 - e * e).powf(1.5);
        let tol = 4.0 * f64::EPSILON * (m.max(1.0) * cond + Angle::TAU);
        let d = turn_distance(got, nu);
        assert!(
            d <= tol,
            "e={e}, M={m}: ν off by {d:.3e} rad (allowed {tol:.3e})"
        );
        worst = worst.max(d / tol);
    }
    println!("Kepler's equation vs 60 digits: worst {worst:.3} of the conditioning bound");
}

/// `(a, e, i, Ω, ω, ν, r, v)`: dyadic elements and the state `to_state`'s formulas give at 60 digits,
/// `μ = 3.986004418e14`. LEO near-SSO, GTO, Molniya, near-GEO, retrograde, `e = 15/16` near
/// perigee, near-polar.
#[rustfmt::skip]
const STATES: [(f64, f64, f64, f64, f64, f64, [f64; 3], [f64; 3]); 7] = [
    (7000000.000000000000000000, 0.0009765625000000000000000000, 1.718750000000000000000000, 0.5000000000000000000000000, 1.250000000000000000000000, 2.000000000000000000000000, [-6163041.099515602210198794, -3239612.359025670170389614, -749395.4073998694800622621], [180.3451072973939092080877, 1358.262363294825433069575, -7417.502636633387682520064]),
    (24500000.00000000000000000, 0.7304687500000000000000000, 0.1250000000000000000000000, 3.000000000000000000000000, 4.000000000000000000000000, 0.2500000000000000000000000, [3793570.844805611515940561, 5461243.047218783931452091, -746635.0795461070561643373], [-7701.564838332034381834639, 6564.564495063125901051651, -680.0487030153644824594418]),
    (26600000.00000000000000000, 0.7187500000000000000000000, 1.106445312500000000000000, 5.000000000000000000000000, 4.718750000000000000000000, 3.000000000000000000000000, [20674231.25148380364352719, -151826.6671416042757106190, 39494433.51528679382717514], [-96.50434503224183699069849, 1551.517976617455877990497, 693.9125580053763841003618]),
    (42164000.00000000000000000, 0.0002441406250000000000000000, 0.0009765625000000000000000000, 1.000000000000000000000000, 2.000000000000000000000000, 5.500000000000000000000000, [-25378821.56380514244446802, 33661574.46040287471614544, 38616.19223148693556178021], [-2455.186981015746500014673, -1851.729168325836386752056, 1.040503326190074068467922]),
    (7500000.000000000000000000, 0.1250000000000000000000000, 2.750000000000000000000000, 0.7500000000000000000000000, 0.5000000000000000000000000, 1.500000000000000000000000, [1964208.873531231040296046, -6576204.523663466180195386, 2539699.635768279128718726], [-6629.539093572262103315116, -3331.569007000427089900672, -859.3972801492453457833157]),
    (10000000.00000000000000000, 0.9375000000000000000000000, 0.5000000000000000000000000, 2.500000000000000000000000, 6.000000000000000000000000, 0.03125000000000000000000000, [-403176.2761413909170393650, 471886.2984077876203660869, -74711.74112621029689337989], [-25236.30074893278940430884, -18284.33676201045234589418, 16253.37462085117711689635]),
    (8000000.000000000000000000, 0.2500000000000000000000000, 1.570312500000000000000000, 4.500000000000000000000000, 3.250000000000000000000000, 4.250000000000000000000000, [-613062.1115637569473958343, -2861146.362675975680432590, 7918032.852456570443313096], [1400.232784946283122181630, 6491.702850414713472936178, 715.1877148504463527993491]),
];

/// (G1) Both conversions against the element map evaluated at 60 digits.
///
/// `to_state` to 4 ulps of `|r|` and `|v|` per component. `from_state`, handed the reference state
/// rounded to doubles, returns the dyadic elements it was built from to 64 ulps times each
/// element's own conditioning: `a` relative (`a = −μ/2ℰ` divides by an energy formed as a
/// difference); `e` absolute, as the length of a vector whose components are differences of order
/// one; the angles absolute in units of `2π`, plus `1/e` for the two measured from the eccentricity
/// vector and `1/sin i` for the two measured from the node — a direction known to `ε` absolute is
/// known to `ε/e` in angle. The identities behind this are `derivations/keplerian_elements.mac`'s 31.
#[test]
fn the_conversions_match_the_element_map_at_60_digits() {
    let (mut worst_state, mut worst_elements) = (0.0_f64, 0.0_f64);
    for (a, e, i, raan, argp, nu, r, v) in STATES {
        let els = KeplerianElements {
            semi_major_axis: a,
            eccentricity: e,
            inclination: Angle::from_radians(i),
            raan: Angle::from_radians(raan),
            arg_periapsis: Angle::from_radians(argp),
            true_anomaly: Angle::from_radians(nu),
            mu: MU,
        };
        let (r_want, v_want) = (Vec3::new(r[0], r[1], r[2]), Vec3::new(v[0], v[1], v[2]));
        let (r_got, v_got) = els.to_state();
        for (got, want) in [(r_got, r_want), (v_got, v_want)] {
            let tol = 4.0 * f64::EPSILON * want.norm();
            for d in [got.x - want.x, got.y - want.y, got.z - want.z] {
                assert!(
                    d.abs() <= tol,
                    "a={a}, e={e}: state off by {d:.3e} (allowed {tol:.3e})"
                );
                worst_state = worst_state.max(d.abs() / tol);
            }
        }

        let back = KeplerianElements::from_state(r_want, v_want, MU);
        let ulps = 64.0 * f64::EPSILON;
        let (by_e, by_node) = (1.0 / e, 1.0 / i.sin());
        let checks = [
            ((back.semi_major_axis - a).abs() / a, ulps),
            ((back.eccentricity - e).abs(), ulps),
            (
                turn_distance(back.inclination.radians(), i),
                ulps * Angle::TAU,
            ),
            (
                turn_distance(back.raan.radians(), raan),
                ulps * (Angle::TAU + by_node),
            ),
            (
                turn_distance(back.arg_periapsis.radians(), argp),
                ulps * (Angle::TAU + by_e + by_node),
            ),
            (
                turn_distance(back.true_anomaly.radians(), nu),
                ulps * (Angle::TAU + by_e),
            ),
        ];
        for (k, (d, tol)) in checks.iter().enumerate() {
            assert!(
                d <= tol,
                "a={a}, e={e}: element {k} off by {d:.3e} (allowed {tol:.3e})"
            );
            worst_elements = worst_elements.max(d / tol);
        }
    }
    println!(
        "element map vs 60 digits: to_state worst {worst_state:.3}, from_state worst {worst_elements:.3} of the bound"
    );
}
