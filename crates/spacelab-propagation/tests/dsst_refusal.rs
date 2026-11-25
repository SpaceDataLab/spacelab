//! Every public constructor and entry point of `dsst`, on arbitrary finite and non-finite input,
//! answers or refuses: it never panics (finding F110).
//!
//! Five sites used to panic. Three `expect`s on the DOP853 driver (`DsstZonal::propagate_mean`,
//! `DsstMeanPropagator::propagate`, reached by every osculating propagation), which a negative span
//! or a state the driver cannot step through tripped; two equinoctial Kepler solvers that panicked
//! when 50 Halley steps did not converge, which `e ≥ 1` or a NaN mean longitude guarantees; and the
//! Gauss–Legendre table lookup, on any order outside its seven. A panic is no answer to a caller:
//! across the C boundary it is `SL_ERR_PANIC` and in Python a `PanicException` that `except
//! Exception` does not catch.
//!
//! The contract each test holds is the refusal order the entry points document: a mean state that
//! is not a finite ellipse is [`DsstError::MeanState`] before anything else is looked at, a span
//! that is not finite and forward is [`DsstError::Span`] next, a mean longitude whose eccentric
//! longitude does not exist is [`DsstError::Kepler`], and past those any outcome but a panic is
//! accepted — this file decides refusal, not accuracy.
//!
//! Spans are drawn within a day: past that the mean-element integration of a valid state runs
//! hundreds of steps and the unoptimised test build spends seconds on a single case, with nothing
//! new to say. The one span beyond the step budget is the deterministic test at the end.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use proptest::prelude::*;
use spacelab_propagation::DynamicalSystem;
use spacelab_propagation::driver::IntegrationError;
use spacelab_propagation::dsst::drag::Atmosphere;
use spacelab_propagation::dsst::gaussian::{self, GAUSS_ORDER, NodeState};
use spacelab_propagation::dsst::third_body::ThirdBodyTruncation;
use spacelab_propagation::dsst::{
    DsstAtmosphericDrag, DsstAuxiliary, DsstError, DsstMeanPropagator, DsstSolarRadiationPressure,
    DsstTesseral, DsstThirdBody, DsstZonal,
};

const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
const J: [f64; 4] = [
    1.082_626_68e-3,
    -2.532_656_49e-6,
    -1.619_621_59e-6,
    -2.272_960_83e-7,
];
const SUN_MU: f64 = 1.327_124_400_18e20;
const MOON_MU: f64 = 4.902_800_066e12;

/// Mostly a plausible value, then any double at all — wide finite values, subnormals, ±∞, NaN — and
/// the edges by name.
fn scalar(plausible: core::ops::Range<f64>) -> impl Strategy<Value = f64> {
    prop_oneof![
        4 => plausible,
        1 => any::<f64>(),
        1 => prop::sample::select(vec![
            0.0,
            -0.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            f64::MAX,
            1.0,
            1.0 - 1e-12,
        ]),
    ]
}

/// `[a, k, h, q, p, λ]`: `k` and `h` each in `(−1, 1)`, so a fifth of the plausible draws are
/// already past `e = 1`.
fn state() -> impl Strategy<Value = [f64; 6]> {
    (
        scalar(6.6e6..5.0e7),
        scalar(-1.0..1.0),
        scalar(-1.0..1.0),
        scalar(-2.0..2.0),
        scalar(-2.0..2.0),
        scalar(-10.0..10.0),
    )
        .prop_map(|(a, k, h, q, p, l)| [a, k, h, q, p, l])
}

fn span() -> impl Strategy<Value = f64> {
    prop_oneof![
        3 => -86_400.0..86_400.0,
        1 => prop::sample::select(vec![
            0.0,
            -0.0,
            -1e-300,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ]),
    ]
}

fn position(scale: f64) -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3(scalar(-scale..scale))
}

fn aux(s: [f64; 6]) -> DsstAuxiliary {
    DsstAuxiliary::new(s[0], s[1], s[2], s[3], s[4], s[5], MU)
}

fn is_ellipse(s: &[f64; 6]) -> bool {
    s.iter().all(|x| x.is_finite()) && s[0] > 0.0 && s[1] * s[1] + s[2] * s[2] < 1.0
}

/// A state-taking entry point: `MeanState` for anything but a finite ellipse, then `Span` for a
/// span that is not finite and forward, then anything but a panic.
fn holds_the_order(
    got: &Result<[f64; 6], DsstError>,
    s: [f64; 6],
    dt: Option<f64>,
) -> Result<(), TestCaseError> {
    if !is_ellipse(&s) {
        prop_assert!(
            matches!(got, Err(DsstError::MeanState { .. })),
            "{s:?}: {got:?}"
        );
    } else if let Some(dt) = dt.filter(|dt| !(dt.is_finite() && *dt >= 0.0)) {
        prop_assert!(matches!(got, Err(DsstError::Span { .. })), "{dt}: {got:?}");
    }
    Ok(())
}

/// An auxiliary-taking entry point that needs the eccentric longitude: `Kepler` unless `e < 1`.
fn needs_an_ellipse<T: core::fmt::Debug>(
    got: &Result<T, DsstError>,
    a: &DsstAuxiliary,
) -> Result<(), TestCaseError> {
    let elliptic = a.k * a.k + a.h * a.h < 1.0;
    if !elliptic {
        prop_assert!(
            matches!(got, Err(DsstError::Kepler { .. })),
            "k = {}, h = {}: {got:?}",
            a.k,
            a.h
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct Air {
    rho: f64,
    omega: f64,
}

impl Atmosphere for Air {
    fn density(&self, _: [f64; 3]) -> f64 {
        self.rho
    }
    fn velocity(&self, p: [f64; 3]) -> [f64; 3] {
        [-self.omega * p[1], self.omega * p[0], 0.0]
    }
}

/// Rows `0..rows`, each of a drawn length, so the shapes run from empty to complete.
fn table() -> impl Strategy<Value = Vec<Vec<f64>>> {
    prop::collection::vec(prop::collection::vec(scalar(-1e-5..1e-5), 0..7), 0..7)
}

fn tesseral_holds(cnm: &[Vec<f64>], snm: &[Vec<f64>]) -> bool {
    cnm.len() == snm.len()
        && cnm.len() >= 3
        && (2..cnm.len()).all(|n| cnm[n].len() > n && snm[n].len() > n)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn the_constructors_build_or_refuse(
        mu in scalar(1e13..1e15),
        re in scalar(6e6..7e6),
        c_n0 in prop::collection::vec(scalar(-1e-3..1e-3), 0..10),
        j in prop::collection::vec(scalar(-1e-3..1e-3), 0..8),
        (cnm, snm) in (table(), table()),
        (ar3, ecc) in (0_usize..40, 0_usize..40),
        values in prop::array::uniform6(scalar(-1e7..1e7)),
        s in state(),
    ) {
        let len = c_n0.len();
        prop_assert_eq!(DsstZonal::new(mu, re, c_n0).is_ok(), len >= 3);
        let j_len = j.len();
        prop_assert_eq!(DsstZonal::from_j(mu, re, &j).is_ok(), j_len >= 1);

        let holds = tesseral_holds(&cnm, &snm);
        let tesseral = DsstTesseral::new(mu, re, values[0], cnm, snm);
        prop_assert_eq!(tesseral.is_ok(), holds, "{:?}", tesseral.err());
        if let Ok(t) = tesseral {
            let a = aux(s);
            let _ = t.mean_element_rates(&a, values[1]);
            let _ = t.short_period_correction(&a, values[1]);
            holds_the_order(&t.osculating(s, values[1]), s, None)?;
        }

        let truncation = ThirdBodyTruncation::new(ar3, ecc);
        prop_assert_eq!(truncation.is_ok(), ar3 < 22 && ecc <= ar3);
        if let Ok(t) = truncation {
            prop_assert_eq!((t.max_ar3_pow(), t.max_ecc_pow()), (ar3, ecc));
        }

        let _ = DsstThirdBody::new(mu, values[2]);
        let _ = DsstSolarRadiationPressure::new(values[0], values[1], mu, re);
        let _ = DsstSolarRadiationPressure::with_reference(
            values[0], values[1], mu, re, values[2], values[3],
        );
        let _ = DsstAtmosphericDrag::new(values[0], values[1], mu);
        let _ = DsstAtmosphericDrag::with_rbar(values[0], values[1], mu, values[4]);
        let _ = DsstAuxiliary::new(values[0], values[1], values[2], values[3], values[4], values[5], mu);
    }

    #[test]
    fn the_zonal_field_answers_or_refuses(
        j in prop::collection::vec(scalar(-1e-3..1e-3), 1..6),
        s in state(),
        dt in span(),
    ) {
        let z = DsstZonal::from_j(MU, RE, &j).unwrap();
        let a = aux(s);
        let _ = z.max_degree();
        let _ = z.max_ecc_pow(&a);
        let _ = z.mean_element_rates(&a);
        needs_an_ellipse(&z.short_period_correction(&a), &a)?;
        holds_the_order(&z.osculating(s), s, None)?;
        holds_the_order(&z.propagate_mean(s, dt), s, Some(dt))?;
        holds_the_order(&z.propagate_osculating(s, dt), s, Some(dt))?;
    }

    #[test]
    fn a_third_body_answers_or_refuses(
        mu3 in scalar(1e12..2e20),
        s in state(),
        body in position(2e11),
        r3 in scalar(1e8..2e11),
        (ar3, ecc) in (0_usize..22, 0_usize..22),
    ) {
        let body_tb = DsstThirdBody::new(MU, mu3);
        let a = aux(s);
        let _ = body_tb.truncation(&a, r3);
        let _ = body_tb.mean_element_rates(&a, body);
        if let Ok(t) = ThirdBodyTruncation::new(ar3, ecc.min(ar3)) {
            let _ = body_tb.mean_element_rates_with_trunc(&a, body, t);
        }
        needs_an_ellipse(&body_tb.short_period_correction(&a, body), &a)?;
        holds_the_order(&body_tb.osculating(s, body), s, None)?;
    }

    #[test]
    fn radiation_pressure_answers_or_refuses(
        cr in scalar(0.5..2.0),
        aom in scalar(0.0..0.1),
        s in state(),
        sun in position(2e11),
    ) {
        let srp = DsstSolarRadiationPressure::new(cr, aom, MU, RE);
        let a = aux(s);
        needs_an_ellipse(&srp.mean_element_rates(&a, sun), &a)?;
        holds_the_order(&srp.osculating(s, sun), s, None)?;
    }

    #[test]
    fn drag_answers_or_refuses(
        cd in scalar(1.0..3.0),
        aom in scalar(0.0..0.1),
        rbar in scalar(6.4e6..8e6),
        air in (scalar(0.0..1e-10), scalar(-1e-4..1e-4)).prop_map(|(rho, omega)| Air { rho, omega }),
        s in state(),
    ) {
        let drag = DsstAtmosphericDrag::with_rbar(cd, aom, MU, rbar);
        let a = aux(s);
        // Only the full-circle arc is centred on the true longitude; the other two need none.
        let _ = drag.l_limits(&a);
        let _ = drag.mean_element_rates(&a, &air);
        if let Ok(order) = drag.gauss_order(&a, &air) {
            prop_assert!(GAUSS_ORDER.contains(&order), "Gauss order {order}");
        }
        holds_the_order(&drag.osculating(s, &air), s, None)?;
    }

    #[test]
    fn the_gaussian_kernels_answer_or_refuse(
        s in state(),
        order in 0_usize..64,
        (low, high, x) in (scalar(-7.0..7.0), scalar(-7.0..7.0), scalar(-7.0..7.0)),
        force in prop::array::uniform3(scalar(-1e-5..1e-5)),
        v in prop::array::uniform6(scalar(-1e7..1e7)),
        threshold in scalar(0.0..1e-6),
    ) {
        let a = aux(s);
        let accel = |node: &NodeState| [force[0] * node.pos[0], force[1], force[2] * node.vel[2]];
        prop_assert_eq!(
            gaussian::integrate(order, &a, low, high, &accel).is_ok(),
            GAUSS_ORDER.contains(&order)
        );
        let _ = gaussian::mean_element_rates(&a, low, high, accel);
        let _ = gaussian::select_order(&a, low, high, accel, threshold);
        let _ = gaussian::value(&a, x, &accel);
        let _ = gaussian::variational_matrix(&a, x);
        let (p, q) = ([v[0], v[1], v[2]], [v[3], v[4], v[5]]);
        let _ = (gaussian::norm(p), gaussian::dot(p, q), gaussian::cross(p, q), gaussian::dot_ext(p, q));
        let _ = gaussian::linear_combination2(v[0], v[1], v[2], v[3]);
        let _ = gaussian::linear_combination3(v[0], v[1], v[2], v[3], v[4], v[5]);
    }
}

fn ephemeris(t: f64) -> ([f64; 3], [f64; 3]) {
    let (ws, wm) = (1.99e-7, 2.66e-6);
    (
        [
            1.496e11 * libm::cos(ws * t),
            1.496e11 * libm::sin(ws * t),
            0.0,
        ],
        [3.84e8 * libm::cos(wm * t), 3.84e8 * libm::sin(wm * t), 0.0],
    )
}

fn gmst(t: f64) -> f64 {
    7.292_115e-5 * t
}

struct Forces {
    zonal: DsstZonal,
    sun: DsstThirdBody,
    moon: DsstThirdBody,
    srp: DsstSolarRadiationPressure,
    tesseral: DsstTesseral,
}

fn forces() -> Forces {
    let mut cnm = vec![vec![0.0; 3]; 3];
    let mut snm = vec![vec![0.0; 3]; 3];
    cnm[2][2] = 1.57e-6;
    snm[2][2] = -9.0e-7;
    Forces {
        zonal: DsstZonal::from_j(MU, RE, &J).unwrap(),
        sun: DsstThirdBody::new(MU, SUN_MU),
        moon: DsstThirdBody::new(MU, MOON_MU),
        srp: DsstSolarRadiationPressure::new(1.5, 0.02, MU, RE),
        tesseral: DsstTesseral::new(MU, RE, 7.292_115e-5, cnm, snm).unwrap(),
    }
}

type Propagator<'a> = DsstMeanPropagator<'a, fn(f64) -> ([f64; 3], [f64; 3]), fn(f64) -> f64>;

fn propagator(f: &Forces) -> Propagator<'_> {
    DsstMeanPropagator {
        zonal: &f.zonal,
        sun: &f.sun,
        moon: &f.moon,
        srp: &f.srp,
        tesseral: &f.tesseral,
        ephemeris,
        gmst,
        mu: MU,
    }
}

proptest! {
    // Each case integrates up to a day under five forces in the unoptimised build.
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn the_combined_propagator_answers_or_refuses(
        s in state(),
        dt in span(),
        t in scalar(-1e9..1e9),
    ) {
        let f = forces();
        let prop = propagator(&f);
        holds_the_order(&prop.propagate(s, dt), s, Some(dt))?;
        holds_the_order(&prop.propagate_osculating(s, dt), s, Some(dt))?;
        holds_the_order(&prop.osculating_at(s, t), s, None)?;
        holds_the_order(&prop.mean_from_osculating(s, t), s, None)?;
        let mut rates = [0.0; 6];
        prop.derivative(t, &s, &mut rates);
    }
}

/// The LEO mean state of the zonal fixture.
const LEO: [f64; 6] = [
    7.2e6,
    3.090_169_943_749_474_7e-3,
    9.510_565_162_951_536e-3,
    0.854_890_329_218_965,
    0.769_746_709_859_868,
    1.431_169_986_635_350_2,
];

/// `[NaN; 6]` and `e = 1 − 1e-12`, the two states the ticket names, through both propagators: the
/// first is no ellipse; the second is one, and its mean rates carry `1/B` with `B = √(1 − e²)` at
/// 1.4e-6, so the step collapses and the driver says so.
#[test]
fn a_nan_state_and_a_grazing_ellipse_are_errors() {
    let f = forces();
    let prop = propagator(&f);
    let nan = [f64::NAN; 6];
    let mut grazing = LEO;
    grazing[1] = 1.0 - 1e-12;
    grazing[2] = 0.0;
    for got in [
        f.zonal.propagate_mean(nan, 3600.0),
        f.zonal.propagate_osculating(nan, 3600.0),
        prop.propagate(nan, 3600.0),
        prop.propagate_osculating(nan, 3600.0),
    ] {
        assert!(matches!(got, Err(DsstError::MeanState { .. })), "{got:?}");
    }
    for got in [
        f.zonal.propagate_mean(grazing, 3600.0),
        f.zonal.propagate_osculating(grazing, 3600.0),
        prop.propagate(grazing, 3600.0),
        prop.propagate_osculating(grazing, 3600.0),
    ] {
        assert!(
            matches!(
                got,
                Err(DsstError::Integration(
                    IntegrationError::StepUnderflow { .. }
                ))
            ),
            "{got:?}"
        );
    }
}

/// An orbit wholly inside the atmosphere integrates over the full circle centred on its true
/// longitude, so with `e ≥ 1` there is no centre and the limits are refused; the partial arc is
/// centred on perigee and answers.
#[test]
fn the_full_circle_drag_arc_needs_an_eccentric_longitude() {
    let drag = DsstAtmosphericDrag::new(2.2, 0.01, MU);
    let inside = aux([1.0e6, 1.5, 0.0, 0.1, 0.1, 0.3]);
    for got in [
        drag.l_limits(&inside),
        drag.mean_element_rates(
            &inside,
            &Air {
                rho: 1e-12,
                omega: 7.292_115e-5,
            },
        )
        .map(|r| [r[0], r[1]]),
    ] {
        assert!(matches!(got, Err(DsstError::Kepler { .. })), "{got:?}");
    }
    let straddling = aux([4.2e7, 1.5, 0.0, 0.1, 0.1, 0.3]);
    assert!(
        drag.l_limits(&straddling).is_ok(),
        "the partial arc is centred on perigee, not on a true longitude"
    );
}

/// A span past the driver's step budget ends as `StepLimit`, where it used to panic: the one span
/// the property tests above do not draw, because reaching the limit costs seconds per case.
#[test]
fn a_span_past_the_step_budget_is_an_integration_error() {
    let f = forces();
    let got = f.zonal.propagate_mean(LEO, 1e12);
    assert!(
        matches!(
            got,
            Err(DsstError::Integration(IntegrationError::StepLimit { .. }))
        ),
        "{got:?}"
    );
}
