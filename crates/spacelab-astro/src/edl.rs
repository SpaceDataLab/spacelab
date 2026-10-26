//! Entry, descent and landing — planar atmospheric-entry dynamics.
//!
//! The hypersonic entry phase, in the planar flight-path-angle form (Vinh; Allen–Eggers) over a
//! spherical planet with an exponential atmosphere. State is speed `V`, flight-path angle `γ`
//! (positive above the local horizontal, so an entry is negative), radius `r`, and downrange `s`:
//!
//! ```text
//! V̇ = −D/m − g sin γ
//! γ̇ = [L/m − (g − V²/r) cos γ] / V
//! ṙ = V sin γ
//! ṡ = V cos γ · R/r
//! ```
//!
//! with drag `D/m = ½ ρ V² / β` (`β = m/(C_D A)` the ballistic coefficient), lift `L/m = (L/D)·D/m`,
//! `g = μ/r²`, and `ρ = ρ₀ exp(−(r−R)/H)`. SI units. Guidance (bank-angle steering, parachute and
//! powered phases) is a later extension; this is the ballistic/lifting hypersonic arc.
//!
//! Gate: `derivations/edl_allen_eggers.mac` reduces these equations, without lift or gravity on a
//! straight path, to Allen and Eggers' closed form `V(h) = V_e exp(−(ρ(h) − ρ_e) H/(2β sin|γ|))`
//! and derives its peak deceleration as the exact maximum, at `ρ = β sin|γ|/H`;
//! `derivations/edl_allen_eggers.gp` evaluates the peak at 100 digits for the test's entry. The
//! module test `a_vertical_ballistic_entry_peaks_where_allen_and_eggers_put_it` (G3) flies that entry
//! through [`propagate_entry`].

#![allow(clippy::many_single_char_names, clippy::cast_precision_loss)]

/// An entry vehicle, by its aerodynamic ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryVehicle {
    /// Ballistic coefficient `β = m/(C_D·A)`, kg/m².
    pub ballistic_coefficient: f64,
    /// Lift-to-drag ratio (0 for a ballistic capsule).
    pub lift_to_drag: f64,
}

/// A spherical planet with an exponential atmosphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atmosphere {
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Surface radius, m.
    pub radius: f64,
    /// Surface density, kg/m³.
    pub surface_density: f64,
    /// Density scale height, m.
    pub scale_height: f64,
}

impl Atmosphere {
    /// Mars: DE-grade `μ`, mean radius, and a standard exponential fit (`ρ₀ ≈ 0.020 kg/m³`,
    /// `H ≈ 11.1 km`).
    pub const MARS: Self = Self {
        mu: 4.282_837e13,
        radius: 3.389_5e6,
        surface_density: 0.020,
        scale_height: 11_100.0,
    };
    /// Earth: standard `μ`, radius, sea-level density and a `~7.2 km` scale height.
    pub const EARTH: Self = Self {
        mu: 3.986_004_418e14,
        radius: 6.378_137e6,
        surface_density: 1.225,
        scale_height: 7_200.0,
    };
}

/// The planar entry state `[V, γ, r, s]`.
pub type EntryState = [f64; 4];

/// The entry equations of motion `[V̇, γ̇, ṙ, ṡ]`.
#[must_use]
pub fn entry_derivative(s: EntryState, vehicle: EntryVehicle, atm: Atmosphere) -> EntryState {
    let (speed, gamma, radius) = (s[0], s[1], s[2]);
    let g = atm.mu / (radius * radius);
    let rho = atm.surface_density * libm::exp(-(radius - atm.radius) / atm.scale_height);
    let d_over_m = 0.5 * rho * speed * speed / vehicle.ballistic_coefficient;
    let l_over_m = vehicle.lift_to_drag * d_over_m;
    let (sg, cg) = (libm::sin(gamma), libm::cos(gamma));
    [
        -d_over_m - g * sg,
        (l_over_m - (g - speed * speed / radius) * cg) / speed,
        speed * sg,
        speed * cg * atm.radius / radius,
    ]
}

/// The outcome of an entry: the peak deceleration (in Earth-g) and the state where the integration
/// stopped (touchdown at the surface, or the terminal condition).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntryOutcome {
    /// Peak sensed deceleration along the trajectory, in Earth-g (`9.80665 m/s²`).
    pub peak_deceleration_g: f64,
    /// Altitude at peak deceleration, m.
    pub peak_altitude: f64,
    /// The final state `[V, γ, r, s]`.
    pub final_state: EntryState,
}

/// Propagate an entry with RK4 (step `dt`, up to `max_steps`) until the vehicle reaches the surface
/// or slows below `stop_speed`, tracking the peak deceleration (drag, in Earth-g).
#[must_use]
pub fn propagate_entry(
    initial: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    dt: f64,
    stop_speed: f64,
    max_steps: usize,
) -> EntryOutcome {
    const G_EARTH: f64 = 9.806_65;
    let add = |a: EntryState, b: EntryState, h: f64| core::array::from_fn(|i| a[i] + b[i] * h);
    let drag_g = |s: EntryState| {
        let rho = atm.surface_density * libm::exp(-(s[2] - atm.radius) / atm.scale_height);
        0.5 * rho * s[0] * s[0] / vehicle.ballistic_coefficient / G_EARTH
    };
    let mut s = initial;
    let (mut peak, mut peak_alt) = (0.0_f64, initial[2] - atm.radius);
    for _ in 0..max_steps {
        let d = drag_g(s);
        if d > peak {
            peak = d;
            peak_alt = s[2] - atm.radius;
        }
        // Written as the flight condition and negated, not as `s[0] < stop_speed || s[2] <=
        // atm.radius`. Off NaN the two are the same test — `>=` and `>` are the exact complements
        // the strictness in `the_stop_rule_is_strict_in_speed_and_inclusive_at_the_surface` pins.
        // On NaN they differ, and that is the point: a NaN state satisfies neither `<` nor `<=`,
        // so the original form flew all `max_steps` of it, while every comparison here is false and
        // the run ends on the pass that produced it.
        if !(s[0] >= stop_speed && s[2] > atm.radius) {
            break;
        }
        let k1 = entry_derivative(s, vehicle, atm);
        let k2 = entry_derivative(add(s, k1, dt / 2.0), vehicle, atm);
        let k3 = entry_derivative(add(s, k2, dt / 2.0), vehicle, atm);
        let k4 = entry_derivative(add(s, k3, dt), vehicle, atm);
        s = core::array::from_fn(|i| s[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    }
    EntryOutcome {
        peak_deceleration_g: peak,
        peak_altitude: peak_alt,
        final_state: s,
    }
}

/// The entry EOM with the lift vertically modulated by `cos_bank = cos σ` (bank-angle guidance): the
/// vertical lift component is `(L/D)·cos σ·D/m`. `cos_bank = 1` is full lift up, `−1` full lift down.
#[must_use]
pub fn entry_derivative_banked(
    s: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    cos_bank: f64,
) -> EntryState {
    let (speed, gamma, radius) = (s[0], s[1], s[2]);
    let g = atm.mu / (radius * radius);
    let rho = atm.surface_density * libm::exp(-(radius - atm.radius) / atm.scale_height);
    let d_over_m = 0.5 * rho * speed * speed / vehicle.ballistic_coefficient;
    let l_over_m = vehicle.lift_to_drag * d_over_m * cos_bank;
    let (sg, cg) = (libm::sin(gamma), libm::cos(gamma));
    [
        -d_over_m - g * sg,
        (l_over_m - (g - speed * speed / radius) * cg) / speed,
        speed * sg,
        speed * cg * atm.radius / radius,
    ]
}

/// The number of prediction steps one call may spend, shared across everything it in turn calls.
///
/// Each loop below carries its own cap and each is reasonable alone, but they *multiply*: a run of
/// the closed loop is `2000` guidance passes × `40` bisections × `40 000` prediction steps, so a
/// trajectory that never meets its terminal condition costs 3.2e9 steps — 1.3e10 right-hand sides —
/// before the outermost cap ends it. A shared budget is what makes the cost of a call the sum of
/// its parts rather than their product.
///
/// The reference Mars entry (`fnpeg_guided_entry_hits_the_target_range`, the most expensive fixture
/// in this module) spends 197 506 steps end to end, and a single prediction 372 of its 40 000. So
/// [`GUIDANCE_STEP_BUDGET`] is ~101× the whole measured flight and ~161× below the composed cap it
/// replaces: an entry two orders of magnitude longer, or steered two orders finer, still fits.
const GUIDANCE_STEP_BUDGET: u32 = 20_000_000;

/// Integrate an entry under a constant bank to the terminal condition (speed `≤ terminal_speed` or
/// the surface) and return the downrange flown.
#[must_use]
pub fn predicted_range(
    initial: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    cos_bank: f64,
    terminal_speed: f64,
) -> f64 {
    predicted_range_within(initial, vehicle, atm, cos_bank, terminal_speed, &mut 40_000)
}

/// [`predicted_range`], drawing its steps from `steps_left`. The public entry point hands it this
/// loop's own cap, so on its own the budget never binds and the trajectory flown is unchanged.
fn predicted_range_within(
    initial: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    cos_bank: f64,
    terminal_speed: f64,
    steps_left: &mut u32,
) -> f64 {
    let add = |a: EntryState, b: EntryState, h: f64| core::array::from_fn(|i| a[i] + b[i] * h);
    let dt = 0.5;
    let mut s = initial;
    for _ in 0..40_000 {
        // Negated flight condition, so a NaN state stops here rather than at the cap — see the stop
        // test in `propagate_entry`.
        if *steps_left == 0 || !(s[0] > terminal_speed && s[2] > atm.radius) {
            break;
        }
        *steps_left -= 1;
        let k1 = entry_derivative_banked(s, vehicle, atm, cos_bank);
        let k2 = entry_derivative_banked(add(s, k1, dt / 2.0), vehicle, atm, cos_bank);
        let k3 = entry_derivative_banked(add(s, k2, dt / 2.0), vehicle, atm, cos_bank);
        let k4 = entry_derivative_banked(add(s, k3, dt), vehicle, atm, cos_bank);
        s = core::array::from_fn(|i| s[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]));
    }
    s[3] - initial[3]
}

/// FNPEG corrector: the bank cosine `cos σ ∈ [−1, 1]` whose predicted range matches `range_to_go`.
/// Range increases monotonically with `cos σ` (more lift up flies farther), so a bisection is robust.
#[must_use]
pub fn fnpeg_bank_command(
    initial: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    range_to_go: f64,
    terminal_speed: f64,
) -> f64 {
    let mut steps_left = 40 * 40_000;
    fnpeg_bank_command_within(
        initial,
        vehicle,
        atm,
        range_to_go,
        terminal_speed,
        &mut steps_left,
    )
}

/// [`fnpeg_bank_command`], drawing its steps from `steps_left`. As above, the public entry point
/// hands it its own composed cap, so alone the budget never binds.
fn fnpeg_bank_command_within(
    initial: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    range_to_go: f64,
    terminal_speed: f64,
    steps_left: &mut u32,
) -> f64 {
    let (mut lo, mut hi) = (-1.0, 1.0);
    for _ in 0..40 {
        let mid = f64::midpoint(lo, hi);
        let flown = predicted_range_within(initial, vehicle, atm, mid, terminal_speed, steps_left);
        if flown < range_to_go {
            lo = mid; // need more range → more lift up
        } else {
            hi = mid;
        }
    }
    f64::midpoint(lo, hi)
}

/// The outcome of a closed-loop guided entry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuidedEntry {
    /// Downrange error at the terminal condition, m (target minus flown).
    pub range_error: f64,
    /// The terminal state `[V, γ, r, s]`.
    pub final_state: EntryState,
}

/// A closed-loop FNPEG-guided entry to a target downrange: every `guidance_dt` seconds the bank is
/// recomputed by the predictor-corrector to null the remaining range error, then flown forward.
///
/// The whole run draws its predictions from one `GUIDANCE_STEP_BUDGET`; a trajectory that exhausts
/// it stops where it had got to, and the range error it reports is then the honest miss.
#[must_use]
pub fn guided_entry(
    initial: EntryState,
    vehicle: EntryVehicle,
    atm: Atmosphere,
    target_range: f64,
    terminal_speed: f64,
    guidance_dt: f64,
) -> GuidedEntry {
    let add = |a: EntryState, b: EntryState, h: f64| core::array::from_fn(|i| a[i] + b[i] * h);
    let mut s = initial;
    let mut steps_left = GUIDANCE_STEP_BUDGET;
    for _ in 0..2_000 {
        // Negated flight condition — see the stop test in `propagate_entry`.
        if steps_left == 0 || !(s[0] > terminal_speed && s[2] > atm.radius) {
            break;
        }
        let flown = s[3] - initial[3];
        let cos_bank = fnpeg_bank_command_within(
            s,
            vehicle,
            atm,
            target_range - flown,
            terminal_speed,
            &mut steps_left,
        );
        // Fly one guidance cycle at this bank.
        let inner = 20;
        let dt = guidance_dt / f64::from(inner);
        for _ in 0..inner {
            if !(s[0] > terminal_speed && s[2] > atm.radius) {
                break;
            }
            let k1 = entry_derivative_banked(s, vehicle, atm, cos_bank);
            let k2 = entry_derivative_banked(add(s, k1, dt / 2.0), vehicle, atm, cos_bank);
            let k3 = entry_derivative_banked(add(s, k2, dt / 2.0), vehicle, atm, cos_bank);
            let k4 = entry_derivative_banked(add(s, k3, dt), vehicle, atm, cos_bank);
            s = core::array::from_fn(|i| {
                s[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
            });
        }
    }
    GuidedEntry {
        range_error: target_range - (s[3] - initial[3]),
        final_state: s,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]
    // The `==` here are the assertions: a state handed back untouched, a term that must be exactly
    // zero, a lift component that must not move at all.
    #![allow(clippy::float_cmp, clippy::manual_midpoint)]
    use super::*;
    use spacelab_testkit::assert_within_ulps;

    /// A state and a vehicle with nothing at zero and nothing symmetric: hypersonic, well inside the
    /// atmosphere, descending at a shallow-but-real angle, lifting, and already downrange.
    const VEHICLE: EntryVehicle = EntryVehicle {
        ballistic_coefficient: 115.0,
        lift_to_drag: 0.24,
    };
    const STATE: EntryState = [5321.0, -0.2715, 3.389_5e6 + 62_500.0, 137_000.0];
    const COS_BANK: f64 = 0.37;

    /// `[V̇, γ̇, ṙ, ṡ]` at `STATE`, evaluated at 60 decimal digits from the equations of the module
    /// header, unbanked and at `COS_BANK`.
    const DERIVATIVE: [f64; 4] = [
        -7.865_661_531_129_359_860_074_466,
        0.001_232_498_226_779_378_667_319_796,
        -1_426.968_712_458_566_615_013_526,
        5_033.280_103_253_884_379_204_645,
    ];
    const DERIVATIVE_BANKED: [f64; 4] = [
        -7.865_661_531_129_359_860_074_466,
        0.000_981_601_267_927_477_500_676_915_9,
        -1_426.968_712_458_566_615_013_526,
        5_033.280_103_253_884_379_204_645,
    ];

    /// One classical Runge–Kutta step of the unbanked equations from `STATE` over 0.75 s, and the
    /// sensed drag at `STATE` in Earth-g — both at 60 decimal digits.
    const RK4_STEP: EntryState = [
        5_314.778_635_321_540_586_110_520,
        -0.270_561_777_235_870_283_209_517_4,
        3_450_932.181_468_133_764_376_732,
        140_773.865_513_298_650_193_820_3,
    ];
    const DRAG_G: f64 = 0.900_359_993_578_327_859_538_598_6;

    /// The equations of motion against a 60-digit evaluation of the closed form in the module
    /// header, at a state with no zero and no symmetry in it.
    #[test]
    fn the_entry_equations_are_their_closed_form() {
        let got = entry_derivative(STATE, VEHICLE, Atmosphere::MARS);
        for (i, name) in ["dV/dt", "dgamma/dt", "dr/dt", "ds/dt"].iter().enumerate() {
            assert_within_ulps(got[i], DERIVATIVE[i], 8, name);
        }
    }

    /// Banking scales the lift and nothing else: at `cos σ = 1` the banked equations are the
    /// unbanked ones bit for bit, and at any other bank only `γ̇` moves.
    #[test]
    fn banking_scales_the_lift_term_and_leaves_the_rest_untouched() {
        let plain = entry_derivative(STATE, VEHICLE, Atmosphere::MARS);
        let full_up = entry_derivative_banked(STATE, VEHICLE, Atmosphere::MARS, 1.0);
        assert_eq!(plain, full_up, "full lift up is not the unbanked case");
        let banked = entry_derivative_banked(STATE, VEHICLE, Atmosphere::MARS, COS_BANK);
        for (i, name) in ["dV/dt", "dgamma/dt", "dr/dt", "ds/dt"].iter().enumerate() {
            assert_within_ulps(banked[i], DERIVATIVE_BANKED[i], 8, name);
            if i != 1 {
                assert_eq!(banked[i], plain[i], "banking moved {name}");
            }
        }
        // Lift down is lift up reflected about the ballistic case, on the one axis it touches.
        let full_down = entry_derivative_banked(STATE, VEHICLE, Atmosphere::MARS, -1.0);
        let ballistic = EntryVehicle {
            lift_to_drag: 0.0,
            ..VEHICLE
        };
        let none = entry_derivative(STATE, ballistic, Atmosphere::MARS);
        let bound = 8.0 * f64::EPSILON * (full_up[1].abs() + full_down[1].abs());
        assert!(
            ((full_up[1] + full_down[1]) / 2.0 - none[1]).abs() <= bound,
            "lift up and lift down do not straddle the ballistic case"
        );
    }

    /// One step of the propagator is one classical Runge–Kutta step of those equations, and the
    /// peak it reports on that step is the sensed drag at the state it started from.
    #[test]
    fn one_step_of_the_propagator_is_one_runge_kutta_step() {
        let out = propagate_entry(STATE, VEHICLE, Atmosphere::MARS, 0.75, 0.0, 1);
        for (i, name) in ["V", "gamma", "r", "s"].iter().enumerate() {
            assert_within_ulps(out.final_state[i], RK4_STEP[i], 8, name);
        }
        assert_within_ulps(out.peak_deceleration_g, DRAG_G, 4, "peak deceleration");
        assert_eq!(
            out.peak_altitude,
            STATE[2] - Atmosphere::MARS.radius,
            "the peak altitude did not start at the entry state"
        );
    }

    /// Asked for no steps at all, the propagator reports the state it was handed and a peak of
    /// zero at the altitude it started from — the only path on which the peak tracker's initial
    /// value is what comes back.
    #[test]
    fn no_steps_reports_the_state_it_was_handed() {
        let out = propagate_entry(STATE, VEHICLE, Atmosphere::MARS, 0.75, 0.0, 0);
        assert_eq!(out.final_state, STATE);
        assert_eq!(out.peak_deceleration_g, 0.0);
        assert_eq!(out.peak_altitude, STATE[2] - Atmosphere::MARS.radius);
    }

    /// Far enough out the exponential atmosphere underflows to nothing and the vehicle senses no
    /// drag at all. The reported peak is then zero at the altitude the run began at, not wherever
    /// the vehicle happened to reach — a peak of zero is not a peak.
    #[test]
    fn a_vehicle_that_never_feels_drag_reports_no_peak() {
        let atm = Atmosphere::MARS;
        let start = [5_321.0, -0.2715, atm.radius + 1.0e7, 0.0];
        let out = propagate_entry(start, VEHICLE, atm, 10.0, 0.0, 50);
        assert_eq!(
            out.peak_deceleration_g, 0.0,
            "phantom drag above the atmosphere"
        );
        assert_eq!(
            out.peak_altitude,
            start[2] - atm.radius,
            "the peak altitude drifted with a vehicle that never decelerated"
        );
        assert!(
            out.final_state[2] < start[2],
            "the vehicle did not descend at all"
        );
    }

    /// The two halves of the stop rule are not interchangeable: the speed test is strict, so a
    /// vehicle exactly at the terminal speed still flies, while the radius test is inclusive, so a
    /// vehicle exactly at the surface has landed. And either one alone ends the run.
    #[test]
    fn the_stop_rule_is_strict_in_speed_and_inclusive_at_the_surface() {
        let atm = Atmosphere::MARS;
        let at_the_limit = propagate_entry(STATE, VEHICLE, atm, 0.75, STATE[0], 5);
        assert_ne!(
            at_the_limit.final_state, STATE,
            "a vehicle exactly at the terminal speed was stopped early"
        );
        let just_above = propagate_entry(STATE, VEHICLE, atm, 0.75, STATE[0] + 1.0, 5);
        assert_eq!(
            just_above.final_state, STATE,
            "a vehicle below the terminal speed kept flying"
        );
        let on_the_surface = [STATE[0], STATE[1], atm.radius, STATE[3]];
        let landed = propagate_entry(on_the_surface, VEHICLE, atm, 0.75, 0.0, 5);
        assert_eq!(
            landed.final_state, on_the_surface,
            "a vehicle at the surface kept flying"
        );
    }

    /// A NaN state ends the run on the pass that produced it. Written the natural way round — stop
    /// once the speed is *below* the limit or the radius *at or below* the surface — neither half
    /// is ever true of a NaN, so the run would fly its whole cap: 40 000 steps for a single range
    /// prediction, and 2000·40·40 000 of them under closed-loop guidance. Nothing is integrated
    /// here, so the untouched components come back bit for bit.
    #[test]
    fn a_nan_state_stops_the_run_rather_than_flying_the_whole_cap() {
        let atm = Atmosphere::MARS;
        let nan_speed = [f64::NAN, STATE[1], STATE[2], STATE[3]];

        let out = propagate_entry(nan_speed, VEHICLE, atm, 0.75, 0.0, 40_000);
        assert!(out.final_state[0].is_nan(), "the NaN speed was integrated");
        assert_eq!(
            out.final_state[1..],
            nan_speed[1..],
            "a NaN state was flown instead of stopping the run"
        );

        assert_eq!(
            predicted_range(nan_speed, VEHICLE, atm, COS_BANK, 0.0),
            0.0,
            "the range predictor flew a NaN state"
        );

        let guided = guided_entry(nan_speed, VEHICLE, atm, 1.0e6, 0.0, 10.0);
        assert_eq!(
            guided.range_error, 1.0e6,
            "the guidance loop flew a NaN state"
        );
        assert_eq!(
            guided.final_state[1..],
            nan_speed[1..],
            "the guidance loop advanced a NaN state"
        );
    }

    /// The prediction budget is spent one step per step and stops the flight when it runs out —
    /// which is what stops the three caps multiplying. Flown against `propagate_entry` at the same
    /// bank, step and terminal speed, a budget of `n` must buy exactly `n` steps and no more, and
    /// the trajectory it buys must be the same one: a budget that is not decremented, or is tested
    /// the wrong way round, cannot reproduce both halves of that.
    #[test]
    fn the_prediction_budget_buys_exactly_the_steps_it_is_given() {
        let atm = Atmosphere::MARS;
        let initial = entry_interface(atm);
        for n in [0_u32, 1, 7, 300] {
            let mut steps_left = n;
            let flown = predicted_range_within(initial, VEHICLE, atm, 1.0, 0.0, &mut steps_left);
            let reference = propagate_entry(initial, VEHICLE, atm, 0.5, 0.0, n as usize);
            assert_eq!(steps_left, 0, "budget {n} was not spent");
            assert_eq!(
                flown,
                reference.final_state[3] - initial[3],
                "budget {n} bought a different trajectory"
            );
        }
        // And the full flight is far inside the budget the closed loop is given, so the budget
        // never clips a trajectory that would have terminated on its own.
        let mut steps_left = GUIDANCE_STEP_BUDGET;
        let full = predicted_range_within(initial, VEHICLE, atm, 1.0, 900.0, &mut steps_left);
        let spent = GUIDANCE_STEP_BUDGET - steps_left;
        assert!(spent < 40_000, "a single prediction spent {spent} steps");
        assert_eq!(
            full,
            predicted_range(initial, VEHICLE, atm, 1.0, 900.0),
            "the budgeted prediction is not the public one"
        );
    }

    /// Both flight conditions are strict, and strictness is observable only at the tie: a state that
    /// arrives exactly at the terminal speed, or exactly at the surface, has already finished, and
    /// the loop must not buy it another step. No trajectory in this file reaches either boundary
    /// exactly — the note on `the_range_predictor_is_the_propagator_flying_full_lift_up` says so —
    /// which is why a `>=` in either place is invisible to every other test here. So the tie is
    /// constructed rather than flown to.
    #[test]
    fn a_state_that_starts_exactly_on_a_flight_boundary_does_not_advance() {
        let atm = Atmosphere::MARS;
        let interface = entry_interface(atm);
        let terminal_speed = 900.0;
        let at_speed = [terminal_speed, interface[1], interface[2], interface[3]];
        let at_surface = [interface[0], interface[1], atm.radius, interface[3]];

        for (boundary, s) in [("terminal speed", at_speed), ("surface", at_surface)] {
            let mut steps_left = GUIDANCE_STEP_BUDGET;
            let flown =
                predicted_range_within(s, VEHICLE, atm, 1.0, terminal_speed, &mut steps_left);
            assert_eq!(flown, 0.0, "the predictor flew past the {boundary}");
            assert_eq!(
                steps_left, GUIDANCE_STEP_BUDGET,
                "the predictor spent a step at the {boundary}"
            );

            let guided = guided_entry(s, VEHICLE, atm, 0.0, terminal_speed, 10.0);
            assert_eq!(
                guided.final_state, s,
                "the guidance loop advanced past the {boundary}"
            );
        }
    }

    /// The *inner* guard's strictness, which the outer one hides. A tie at the top of a cycle is
    /// intercepted by the outer guard, so the only way to separate `>` from `>=` at
    /// `guided_entry`'s sub-step test is a Runge-Kutta step that lands exactly on the terminal
    /// speed — and no flown fixture does, because the terminal speed is what the corrector is
    /// pointed at. The circularity breaks on a *ballistic* vehicle: at `lift_to_drag = 0` the
    /// vertical lift term is `0.0 · D/m · cos σ`, which is `±0.0` for every bank, and the `γ̇`
    /// numerator `±0.0 − (g − V²/r)·cos γ` is bit-identical either way while `(g − V²/r)·cos γ ≠ 0`.
    /// So the trajectory does not depend on the bank at all, `propagate_entry` at the sub-step size
    /// reproduces it exactly, and the tie can be *read off* rather than guessed.
    #[test]
    fn the_guidance_sub_step_stops_at_the_terminal_speed_and_not_one_step_past_it() {
        let atm = Atmosphere::MARS;
        let ballistic = EntryVehicle {
            ballistic_coefficient: 65.0,
            lift_to_drag: 0.0,
        };
        // 40 km up, where drag already dominates gravity, so the vehicle is decelerating and the
        // outer guard passes on the way in.
        let initial = [5_800.0, -15.5_f64.to_radians(), atm.radius + 40_000.0, 0.0];
        let guidance_dt = 10.0;
        let sub_dt = guidance_dt / 20.0; // exactly 0.5 — the inner loop's own step
        let one = propagate_entry(initial, ballistic, atm, sub_dt, 0.0, 1).final_state;
        assert!(
            one[0] < initial[0] && one[2] > atm.radius,
            "fixture does not decelerate above the surface: {one:?}"
        );

        let guided = guided_entry(initial, ballistic, atm, 0.0, one[0], guidance_dt);
        assert_eq!(
            guided.final_state, one,
            "the guidance loop flew a sub-step past the terminal speed"
        );
    }

    /// The same for the inner guard's *radius* half, with the circularity broken the other way: at
    /// `surface_density = 0` the density is `0.0` whatever the surface radius is, and `g = μ/r²`
    /// never reads it, so `[V, γ, r]` is independent of `atm.radius` and only the downrange rate
    /// `V cos γ · R/r` sees it. That lets the surface be moved onto a step's endpoint after the
    /// fact.
    #[test]
    fn the_guidance_sub_step_stops_at_the_surface_and_not_one_step_below_it() {
        let vacuum = Atmosphere {
            surface_density: 0.0,
            ..Atmosphere::MARS
        };
        let initial = [
            5_800.0,
            -15.5_f64.to_radians(),
            vacuum.radius + 40_000.0,
            0.0,
        ];
        let guidance_dt = 10.0;
        let one = propagate_entry(initial, VEHICLE, vacuum, guidance_dt / 20.0, 0.0, 1).final_state;
        let atm = Atmosphere {
            radius: one[2],
            ..vacuum
        };
        assert!(
            initial[2] > atm.radius,
            "fixture does not descend: {initial:?} -> {one:?}"
        );

        let guided = guided_entry(initial, VEHICLE, atm, 0.0, 0.0, guidance_dt);
        assert_eq!(
            guided.final_state[2], atm.radius,
            "the guidance loop flew a sub-step below the surface"
        );
    }

    /// The corrector's budget is a *product*: forty bisections, each of which may fly a whole
    /// prediction. Written as a sum it is 40 040 — barely one prediction's worth — and the later
    /// bisections then run out of steps and bisect on ranges that were cut short, so the bank comes
    /// back from an interval that never converged. Nothing noticed, because no test measured what a
    /// corrector costs, only that it answers.
    ///
    /// Picking the fixture took measuring, because *exceeding* the smaller budget is not enough —
    /// the answer has to move. The reference −15.5° entry is steep: one prediction costs 467 steps
    /// and the whole corrector 11 059, inside the sum, where the two forms cannot differ. At −10° the
    /// corrector spends 53 110, past the sum — and still returns the same bank, because a clipped
    /// prediction comes back short, which sends the bisection the same way the full one would.
    /// At −9° it spends 264 578 and the bank does move: −0.091 016 838 227 915 on the product, which
    /// is also what an unbounded budget gives, against −0.0625 on the sum. Shallower still (−8° and
    /// below) the entry skips, every prediction saturates its own 40 000-step cap, and the corrector
    /// exhausts even the product — so −9° is the window where the product is both necessary and
    /// sufficient.
    #[test]
    fn the_correctors_budget_is_the_product_of_its_two_caps() {
        let atm = Atmosphere::MARS;
        let steep = entry_interface(atm);
        let initial = [steep[0], -9.0_f64.to_radians(), steep[2], steep[3]];
        let terminal_speed = 900.0;
        let range_to_go = predicted_range(initial, VEHICLE, atm, TARGET_BANK, terminal_speed);

        // A budget large enough that it cannot bind, to say what the corrector's answer *is*.
        let mut generous = GUIDANCE_STEP_BUDGET;
        let unbudgeted = fnpeg_bank_command_within(
            initial,
            VEHICLE,
            atm,
            range_to_go,
            terminal_speed,
            &mut generous,
        );
        let spent = GUIDANCE_STEP_BUDGET - generous;
        assert!(
            spent > 40 + 40_000,
            "the corrector spent {spent} steps, which the sum of its two caps would have covered — \
             this fixture cannot tell the product from the sum"
        );

        // The public entry point's own budget must not bind. A clipped bisection still answers, and
        // answers plausibly, so the comparison is against the unbudgeted bank exactly rather than
        // against anything near it.
        assert_eq!(
            fnpeg_bank_command(initial, VEHICLE, atm, range_to_go, terminal_speed),
            unbudgeted,
            "the public corrector's budget bound and cut the bisection short"
        );
    }

    /// Runge–Kutta is fourth order, so over a fixed span each halving of the step must cut the
    /// difference between successive solutions by sixteen. Any stage weight that is wrong drops the
    /// order, which no accuracy bound on a single step size would notice.
    #[test]
    fn the_propagator_converges_at_fourth_order() {
        let span = 20.0;
        let solutions: Vec<EntryState> = [10usize, 20, 40, 80, 160]
            .iter()
            .map(|&n| {
                #[expect(clippy::cast_precision_loss, reason = "n is at most 160")]
                let dt = span / n as f64;
                propagate_entry(STATE, VEHICLE, Atmosphere::MARS, dt, 0.0, n).final_state
            })
            .collect();
        let gap = |a: &EntryState, b: &EntryState| (a[0] - b[0]).abs();
        for (k, w) in solutions.windows(3).enumerate() {
            let ratio = gap(&w[0], &w[1]) / gap(&w[1], &w[2]);
            assert!(
                (13.0..19.0).contains(&ratio),
                "halving {k} converged at a ratio of {ratio:.2}, not 16"
            );
        }
    }

    /// The predictor is monotone in the bank cosine — which is what makes the corrector's bisection
    /// valid — and the corrector inverts it: the bank it returns flies the range it was asked for.
    #[test]
    fn the_range_predictor_is_monotone_and_the_corrector_inverts_it() {
        let atm = Atmosphere::MARS;
        let initial = [5_800.0, -15.5_f64.to_radians(), atm.radius + 125_000.0, 0.0];
        let v_term = 900.0;
        let banks = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0];
        let ranges: Vec<f64> = banks
            .iter()
            .map(|&b| predicted_range(initial, VEHICLE, atm, b, v_term))
            .collect();
        for (k, w) in ranges.windows(2).enumerate() {
            assert!(
                w[1] > w[0] + 1_000.0,
                "range is not increasing in the bank cosine at step {k}: {w:?}"
            );
        }
        for &target in &[ranges[1], f64::midpoint(ranges[2], ranges[3]), ranges[4]] {
            let bank = fnpeg_bank_command(initial, VEHICLE, atm, target, v_term);
            let flown = predicted_range(initial, VEHICLE, atm, bank, v_term);
            assert!(
                (flown - target).abs() < 100.0,
                "the commanded bank {bank:.6} flies {flown:.0} m, not the {target:.0} m asked for"
            );
        }
    }

    /// (G3) Allen and Eggers' entry, flown by the propagator. With `µ = 0` and a vertical path the
    /// module's equations are exactly the Allen–Eggers model: `γ̇ = V cos γ / r` is 6e-17 of `V/r`
    /// for the double nearest `−π/2`, and `sin γ` rounds to `−1`. `derivations/edl_allen_eggers.mac`
    /// derives the speed `V(h)` in closed form and its peak deceleration as the exact maximum, at
    /// `h* = H ln(ρ₀H/β)`; `derivations/edl_allen_eggers.gp` evaluates it at 100 digits for this
    /// entry — Mars atmosphere, `β = 100 kg/m²`, 6 km/s from 120 km.
    ///
    /// The propagator samples the drag at the start of each step, so the reported peak is the
    /// largest sample, within one step of the true maximum: within `V*·dt` of `h*` in altitude,
    /// and, as `d² ln D/dh² = −1/H²` there, below `D*` by at most `(V*·dt/H)²/2` relative. The RK4
    /// error at `dt = 1 ms` on a 2-second time scale is far below either. The speed at touchdown
    /// is held to the closed form `V(h)` at the altitude where the run stopped, 1e-12 relative.
    #[test]
    fn a_vertical_ballistic_entry_peaks_where_allen_and_eggers_put_it() {
        const PEAK_ALTITUDE: f64 = 8_852.329_874_314_487_887_370_117;
        const PEAK_SPEED: f64 = 3_639.265_476_811_942_643_139_040;
        const PEAK_G: f64 = 60.835_043_818_699_680_002_959_86;
        let atm = Atmosphere {
            mu: 0.0,
            ..Atmosphere::MARS
        };
        let vehicle = EntryVehicle {
            ballistic_coefficient: 100.0,
            lift_to_drag: 0.0,
        };
        let entry_altitude = 120_000.0;
        let (entry_speed, dt) = (6_000.0, 1e-3);
        let initial = [
            entry_speed,
            -core::f64::consts::FRAC_PI_2,
            atm.radius + entry_altitude,
            0.0,
        ];
        let out = propagate_entry(initial, vehicle, atm, dt, 0.0, 200_000);

        let step = PEAK_SPEED * dt;
        assert!(
            (out.peak_altitude - PEAK_ALTITUDE).abs() <= step,
            "peak at {:.3} m, Allen–Eggers {PEAK_ALTITUDE:.3} m, one step {step:.3} m",
            out.peak_altitude
        );
        let deficit = 0.5 * (step / atm.scale_height).powi(2);
        let rel = out.peak_deceleration_g / PEAK_G - 1.0;
        assert!(
            rel <= 1e-12 && rel >= -deficit - 1e-12,
            "peak {:.15} g, Allen–Eggers {PEAK_G:.15} g: {rel:.3e} relative, allowed down to {:.3e}",
            out.peak_deceleration_g,
            -deficit
        );

        let [speed, _, radius, _] = out.final_state;
        println!(
            "Allen–Eggers: peak {:.3} m off h*, {rel:.3e} relative off D*, touchdown at {speed:.6} m/s",
            out.peak_altitude - PEAK_ALTITUDE
        );
        assert!(
            radius <= atm.radius,
            "the run stopped above the surface at {radius} m"
        );
        let density = |h: f64| atm.surface_density * (-h / atm.scale_height).exp();
        let closed = entry_speed
            * (-(density(radius - atm.radius) - density(entry_altitude)) * atm.scale_height
                / (2.0 * vehicle.ballistic_coefficient))
                .exp();
        assert!(
            (speed / closed - 1.0).abs() <= 1e-12,
            "touchdown speed {speed:.12} m/s, closed form {closed:.12} m/s"
        );
    }

    /// A ballistic Mars entry decelerates through a peak and slows from hypersonic to subsonic, with
    /// the altitude falling monotonically — the Allen–Eggers profile.
    #[test]
    fn ballistic_mars_entry_decelerates_through_a_peak() {
        let vehicle = EntryVehicle {
            ballistic_coefficient: 65.0,
            lift_to_drag: 0.0,
        };
        // Entry interface: 125 km altitude, 5.8 km/s, −15.5° flight path.
        let r0 = Atmosphere::MARS.radius + 125_000.0;
        let initial = [5_800.0, -15.5_f64.to_radians(), r0, 0.0];
        let out = propagate_entry(initial, vehicle, Atmosphere::MARS, 0.1, 400.0, 40_000);

        println!(
            "Mars entry: peak {:.1} g at {:.0} km, final V {:.0} m/s",
            out.peak_deceleration_g,
            out.peak_altitude / 1000.0,
            out.final_state[0]
        );
        // A steep Mars entry peaks at several to a few tens of Earth-g.
        assert!(
            (2.0..40.0).contains(&out.peak_deceleration_g),
            "peak decel {:.1} g implausible",
            out.peak_deceleration_g
        );
        // The peak occurs in the atmosphere, well above the surface but below the entry interface.
        assert!(
            (5_000.0..80_000.0).contains(&out.peak_altitude),
            "peak at {:.0} m",
            out.peak_altitude
        );
        // It slowed from hypersonic (5.8 km/s) to the terminal condition.
        assert!(
            out.final_state[0] <= 400.0,
            "did not decelerate to terminal: {:.0} m/s",
            out.final_state[0]
        );
    }

    /// Lift raises the trajectory: a lifting entry (L/D > 0) sheds less peak deceleration than the
    /// ballistic case from the same interface, because it stays higher (thinner air) longer.
    #[test]
    fn lift_reduces_peak_deceleration() {
        let r0 = Atmosphere::MARS.radius + 125_000.0;
        let initial = [5_800.0, -15.5_f64.to_radians(), r0, 0.0];
        let ballistic = propagate_entry(
            initial,
            EntryVehicle {
                ballistic_coefficient: 65.0,
                lift_to_drag: 0.0,
            },
            Atmosphere::MARS,
            0.1,
            400.0,
            40_000,
        );
        let lifting = propagate_entry(
            initial,
            EntryVehicle {
                ballistic_coefficient: 65.0,
                lift_to_drag: 0.3,
            },
            Atmosphere::MARS,
            0.1,
            400.0,
            40_000,
        );
        assert!(
            lifting.peak_deceleration_g < ballistic.peak_deceleration_g,
            "lift did not soften the peak"
        );
    }

    /// The FNPEG closed-loop guided entry hits a commanded downrange far better than any fixed bank —
    /// the predictor-corrector steers the bank angle to null the range error.
    #[test]
    fn fnpeg_guided_entry_hits_the_target_range() {
        let vehicle = EntryVehicle {
            ballistic_coefficient: 115.0,
            lift_to_drag: 0.24,
        };
        let atm = Atmosphere::MARS;
        let initial = [5_800.0, -15.5_f64.to_radians(), atm.radius + 125_000.0, 0.0];
        let v_term = 900.0;
        let full_up = predicted_range(initial, vehicle, atm, 1.0, v_term);
        let full_down = predicted_range(initial, vehicle, atm, -1.0, v_term);
        let target = f64::midpoint(full_up, full_down); // an achievable mid-range
        let guided = guided_entry(initial, vehicle, atm, target, v_term, 5.0);
        println!(
            "FNPEG: full-up {:.0} km, full-down {:.0} km, target {:.0} km, guided error {:.2} km",
            full_up / 1000.0,
            full_down / 1000.0,
            target / 1000.0,
            guided.range_error / 1000.0
        );
        assert!(
            guided.range_error.abs() < 5_000.0,
            "guided missed by {:.1} km",
            guided.range_error / 1000.0
        );
        // The range-control authority (full-up vs target) is large — a fixed bank could not do this.
        assert!(
            (full_up - target).abs() > 20_000.0,
            "range-control authority too small to test"
        );
    }

    /// The FNPEG scenario, flown with the vehicle already downrange of the interface so that a law
    /// which confuses the flown range with the absolute one cannot pass.
    const DOWNRANGE_AT_INTERFACE: f64 = 173_000.0;
    /// The bank cosine whose predicted range the guided flight is asked to reproduce. Dyadic, so the
    /// corrector's bisection lands on it exactly rather than converging to it.
    const TARGET_BANK: f64 = 0.5;

    fn entry_interface(atm: Atmosphere) -> EntryState {
        [
            5_800.0,
            -15.5_f64.to_radians(),
            atm.radius + 125_000.0,
            DOWNRANGE_AT_INTERFACE,
        ]
    }

    /// `predicted_range` is `propagate_entry` with the bank held: same Runge-Kutta, same step, same
    /// terminal condition, and `entry_derivative_banked` at `cos σ = 1` is `entry_derivative`. So at
    /// full lift up the two must agree bit for bit — the predictor is not an independent integrator
    /// that happens to land nearby. (The stop rules differ in strictness at exactly `stop_speed`, a
    /// value this trajectory does not take.)
    #[test]
    fn the_range_predictor_is_the_propagator_flying_full_lift_up() {
        let atm = Atmosphere::MARS;
        let initial = entry_interface(atm);
        let terminal_speed = 900.0;
        let predicted = predicted_range(initial, VEHICLE, atm, 1.0, terminal_speed);
        let propagated = propagate_entry(initial, VEHICLE, atm, 0.5, terminal_speed, 40_000);
        assert_eq!(
            predicted,
            propagated.final_state[3] - initial[3],
            "the predictor and the propagator flew different trajectories under the same lift"
        );
    }

    /// The predictor integrates at 0.5 s, so a 10 s guidance cycle cut into 20 inner steps flies
    /// exactly the trajectory the predictor predicted. FNPEG is then at a fixed point: the bank
    /// solved for at the interface stays the solution at every later cycle, and the flight lands on
    /// the requested range to the bisection's own resolution rather than to guidance tolerance. Any
    /// drift between the two integrations shows up here as a range error of metres — a 20 s cycle,
    /// which halves the inner step count, misses by 444 m.
    #[test]
    fn a_guidance_cycle_matched_to_the_predictor_flies_exactly_what_it_predicted() {
        let atm = Atmosphere::MARS;
        let initial = entry_interface(atm);
        let terminal_speed = 900.0;
        let target = predicted_range(initial, VEHICLE, atm, TARGET_BANK, terminal_speed);
        let guided = guided_entry(initial, VEHICLE, atm, target, terminal_speed, 10.0);
        assert!(
            guided.range_error.abs() < 1e-6,
            "the guided flight missed its own prediction by {:.3e} m",
            guided.range_error
        );
        assert!(
            guided.final_state[0] <= terminal_speed
                && guided.final_state[0] > terminal_speed - 10.0,
            "stopped at {:.3} m/s, not within one step of the terminal speed",
            guided.final_state[0]
        );
        assert!(
            guided.final_state[2] > atm.radius + 1_000.0,
            "stopped {:.0} m above the surface — it flew into the ground instead of slowing to the \
             terminal speed",
            guided.final_state[2] - atm.radius
        );
    }
}
