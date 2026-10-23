//! Atmospheric drag, checked by the decay physics it must reproduce — no reference trajectory.
//!
//! Drag is the first dissipative force, so the invariants are one-sided, not conserved:
//!
//! * **The density profile** is `ρ₀ exp(−(h−h₀)/H)` — asserted directly, because every other check
//!   here takes its expected value from the same model and so cannot see a wrong one.
//! * **The drag law** is `−½ (C_d A/m) ρ ‖v_rel‖ v_rel` in every component at one state, with the
//!   co-rotation term formed from the general cross product rather than its simplification.
//! * **Energy falls monotonically** — drag can only remove orbital energy, never add it.
//! * **The circular decay rate** matches King-Hele's leading term `Δa = −2π (C_d A/m) ρ a²` per
//!   revolution, with the atmosphere held inertial (`ω⊕ = 0`) so the closed form applies.
//! * **Co-rotation reduces drag** — a prograde orbit sees `v_rel = v − ω⊕×r < v`, so switching the
//!   Earth's rotation on slows the decay. This is the term a naïve model drops.
//! * **`NoWind` is bit-exact** — the default wind reproduces the windless drag law to the last bit,
//!   so admitting a wind term changes no existing answer.
//! * **A solid-body wind is a change of co-rotation rate** — `w = Ω×r` and `ω⊕ + Ω` are the same
//!   force, which is what makes the wind plumbing testable without a wind model.
//! * **The decay paradox** — as `a` falls the orbit *speeds up*, `v = √(μ/a)`, though drag opposes
//!   the motion at every instant.
//! * **The STM contracts phase volume** — `det Φ < 1` by the integral of `tr(∂a/∂v) < 0`, the mark
//!   of a dissipative flow (a conservative force keeps `det Φ = 1`). This exercises the drag
//!   Jacobian from automatic differentiation inside the variational equations.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::driver::Controls;
use spacelab_propagation::driver::integrate;
use spacelab_propagation::forces::{
    Acceleration, Atmosphere, Drag, ExponentialAtmosphere, Newtonian, NoWind, Plain, PointMass,
    Wind,
};
use spacelab_propagation::variational::{Variational, augment, split};
use spacelab_propagation::{Dop853, Tolerance};

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;
const OMEGA_EARTH: f64 = 7.292_115_9e-5;
const BALLISTIC: f64 = 0.02; // C_d A / m, m²/kg

fn atmosphere() -> ExponentialAtmosphere {
    ExponentialAtmosphere {
        reference_density: 1e-11,
        reference_altitude: 350e3,
        scale_height: 50e3,
        earth_radius: RE,
    }
}

fn drag(omega: f64) -> Drag<ExponentialAtmosphere> {
    Drag {
        ballistic_coefficient: BALLISTIC,
        earth_rotation_rate: omega,
        atmosphere: atmosphere(),
        wind: NoWind,
    }
}

/// A circular orbit at altitude `alt`, inclination `inc`, ascending at `x̂`.
fn leo(alt: f64, inc: f64) -> ([f64; 6], f64) {
    let a = RE + alt;
    let v = (MU / a).sqrt();
    let period = std::f64::consts::TAU * (a * a * a / MU).sqrt();
    ([a, 0.0, 0.0, 0.0, v * inc.cos(), v * inc.sin()], period)
}

/// Osculating semi-major axis from the vis-viva energy: `a = 1 / (2/r − v²/μ)`. Needs no elements
/// crate, and is exactly the instantaneous two-body `a`.
fn sma(y: &[f64; 6]) -> f64 {
    let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
    let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    1.0 / (2.0 / r - v2 / MU)
}

fn energy(y: &[f64; 6]) -> f64 {
    let r = (y[0] * y[0] + y[1] * y[1] + y[2] * y[2]).sqrt();
    let v2 = y[3] * y[3] + y[4] * y[4] + y[5] * y[5];
    0.5 * v2 - MU / r
}

fn speed(y: &[f64; 6]) -> f64 {
    (y[3] * y[3] + y[4] * y[4] + y[5] * y[5]).sqrt()
}

/// Determinant of a 6×6 by Gaussian elimination with partial pivoting.
fn det6(m: &[[f64; 6]; 6]) -> f64 {
    let mut a = *m;
    let mut det = 1.0;
    for col in 0..6 {
        let mut pivot = col;
        for row in col + 1..6 {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            a.swap(col, pivot);
            det = -det;
        }
        det *= a[col][col];
        for row in col + 1..6 {
            let factor = a[row][col] / a[col][col];
            for c in col..6 {
                a[row][c] -= factor * a[col][c];
            }
        }
    }
    det
}

/// Propagate `revs` revolutions of a drag-perturbed circular orbit and return the final state.
fn decay(alt: f64, inc: f64, omega: f64, revs: f64) -> [f64; 6] {
    let (y0, period) = leo(alt, inc);
    let force = Newtonian((PointMass { mu: MU }, drag(omega)));
    integrate(
        &Dop853,
        &Plain(force),
        0.0,
        &y0,
        revs * period,
        30.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-4),
    )
    .expect("propagated")
    .final_state()
}

#[test]
fn drag_removes_energy_monotonically() {
    let (y0, period) = leo(350e3, 51.6_f64.to_radians());
    let force = Newtonian((PointMass { mu: MU }, drag(OMEGA_EARTH)));

    let mut prev = energy(&y0);
    // sample once per half-revolution; each later energy must be strictly below the last
    for k in 1..=20 {
        let yk = integrate(
            &Dop853,
            &Plain(force),
            0.0,
            &y0,
            f64::from(k) * 0.5 * period,
            30.0,
            Controls::new(Tolerance::new(1e-12, 1e-12), 1e-4),
        )
        .expect("propagated")
        .final_state();
        let e = energy(&yk);
        assert!(e < prev, "energy rose at sample {k}: {e:.6e} ≥ {prev:.6e}");
        prev = e;
    }
}

/// `ρ₀ exp(−(h−h₀)/H)` written out, so nothing downstream has to take the model's word for it.
fn closed_form_density(r: [f64; 3]) -> f64 {
    let altitude = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt() - RE;
    1e-11 * (-(altitude - 350e3) / 50e3).exp()
}

#[test]
fn the_exponential_atmosphere_is_its_closed_form() {
    // Nothing else in this file asserts the density *profile*. The King-Hele comparison below takes
    // its predicted decay rate from `density` itself, so a wrong profile moves the measurement and
    // the prediction together and the ratio survives — even a reversed one, where the air thickens
    // with altitude, passes every decay test here.
    let air = atmosphere();
    // Off every axis for the value sweep: at `[RE + h, 0, 0]` the `y` and `z` terms of the radius
    // are exact zeros, and the sum that forms `‖r‖` is then two thirds unasserted.
    let dir = [2.0_f64, -3.0, 1.5];
    let unit = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let general = |h: f64| dir.map(|d| d * (RE + h) / unit);
    for &h in &[0.0, 200e3, 350e3, 700e3, 1200e3] {
        let r = general(h);
        let (got, want) = (air.density::<f64>(0.0, &r), closed_form_density(r));
        let rel = (got - want).abs() / want;
        assert!(rel < 1e-14, "ρ({h:.0} m) is {got:.6e}, want {want:.6e}");
    }
    // The exact statements stay on the axis, where `‖r‖` is `RE + h` to the last bit.
    let at = |h: f64| air.density::<f64>(0.0, &[RE + h, 0.0, 0.0]);
    // The scale height and the direction again, this time without the exponential: one `H` up is a
    // factor `e` down, and the reference altitude is exactly where `ρ = ρ₀`.
    assert!(
        at(300e3) > at(350e3) && at(350e3) > at(400e3),
        "density must fall with altitude, not rise"
    );
    let folding = at(350e3) / at(400e3);
    assert!(
        (folding - std::f64::consts::E).abs() < 1e-12,
        "one scale height must be a factor e, got {folding}"
    );
    assert!(
        (at(350e3) / 1e-11 - 1.0).abs() < 1e-15,
        "ρ at the reference altitude must be ρ₀, got {:.6e}",
        at(350e3)
    );
}

#[test]
fn drag_opposes_the_co_rotating_velocity_exactly() {
    // `a = −½ (C_d A/m) ρ ‖v_rel‖ v_rel`, `v_rel = v − ω⊕×r`. The cross product is written here from
    // the general definition, not from the `(−ωy, ωx, 0)` simplification the force uses, so a wrong
    // simplification is a disagreement rather than a shared assumption. The decay tests cannot make
    // this statement: they compare orbit-averaged `a` over ten revolutions to 5%, and the
    // co-rotation check asserts only an inequality between two whole propagations.
    let r = [5.1e6, 3.7e6, 2.4e6];
    let v = [-2.9e3, 5.4e3, 3.1e3];
    let w = [0.0, 0.0, OMEGA_EARTH];
    let cross = [
        w[1] * r[2] - w[2] * r[1],
        w[2] * r[0] - w[0] * r[2],
        w[0] * r[1] - w[1] * r[0],
    ];
    let v_rel = [v[0] - cross[0], v[1] - cross[1], v[2] - cross[2]];
    let vr = (v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2]).sqrt();
    let k = -0.5 * BALLISTIC * closed_form_density(r) * vr;

    let got = drag(OMEGA_EARTH).acceleration::<f64>(0.0, &r, &v);
    let scale = k.abs() * vr;
    for i in 0..3 {
        let rel = (got[i] - k * v_rel[i]).abs() / scale;
        assert!(
            rel < 1e-14,
            "drag axis {i}: {:.6e} vs {:.6e}",
            got[i],
            k * v_rel[i]
        );
    }
}

#[test]
fn circular_decay_matches_the_king_hele_rate() {
    // With the atmosphere inertial (ω⊕ = 0), Δa per revolution is −2π (C_d A/m) ρ a² to leading
    // order. Measure it over 10 revolutions and compare to the closed form.
    let alt = 350e3;
    let (y0, _) = leo(alt, 51.6_f64.to_radians());
    let a0 = sma(&y0);
    let revs = 10.0;

    let yf = decay(alt, 51.6_f64.to_radians(), 0.0, revs);
    let measured = sma(&yf) - a0;

    let rho = atmosphere().density::<f64>(0.0, &[a0, 0.0, 0.0]);
    let predicted = -revs * std::f64::consts::TAU * BALLISTIC * rho * a0 * a0;

    let rel = (measured - predicted).abs() / predicted.abs();
    println!(
        "decay: measured {measured:.3} m, King-Hele {predicted:.3} m over {revs} revs, {:.2}%",
        100.0 * rel
    );
    assert!(measured < 0.0, "the orbit must decay, not climb");
    assert!(
        rel < 0.05,
        "decay rate {measured:.3} m vs King-Hele {predicted:.3} m ({:.1}%)",
        100.0 * rel
    );
}

#[test]
fn co_rotation_reduces_drag_on_a_prograde_orbit() {
    // An equatorial prograde orbit runs with the atmosphere; v_rel = v − ω⊕×r < v, so the real
    // (co-rotating) model must decay less than one that treats the atmosphere as inertial.
    let revs = 10.0;
    let a_inertial = sma(&decay(350e3, 0.0, 0.0, revs));
    let a_corotating = sma(&decay(350e3, 0.0, OMEGA_EARTH, revs));

    assert!(
        a_corotating > a_inertial,
        "co-rotation should slow the decay: co-rotating a={a_corotating:.1} ≤ inertial a={a_inertial:.1}"
    );
}

#[test]
fn the_orbit_speeds_up_as_it_decays() {
    // The decay paradox: drag decelerates at every instant, yet the secular effect of losing
    // altitude raises the orbital speed, v = √(μ/a).
    let (y0, _) = leo(350e3, 51.6_f64.to_radians());
    let yf = decay(350e3, 51.6_f64.to_radians(), OMEGA_EARTH, 15.0);
    assert!(
        speed(&yf) > speed(&y0),
        "speed should rise as the orbit decays: {:.3} ≤ {:.3}",
        speed(&yf),
        speed(&y0)
    );
}

#[test]
fn the_dissipative_stm_contracts_phase_volume() {
    // det Φ = exp(∫ tr A dt) with tr A = tr(∂a_drag/∂v) = −2 (C_d A/m) ρ |v_rel| < 0, so a
    // drag-perturbed flow contracts phase-space volume: det Φ < 1, strictly. A conservative force
    // would hold det Φ = 1. This runs the drag Jacobian (from AD) through the variational equations.
    let (y0, period) = leo(350e3, 45.0_f64.to_radians());
    let force = Newtonian((PointMass { mu: MU }, drag(OMEGA_EARTH)));
    let var = Variational::<6, _>::new(&force);
    let aug0: [f64; 42] = augment::<6, 42>(&y0);

    let traj = integrate(
        &Dop853,
        &var,
        0.0,
        &aug0,
        8.0 * period,
        30.0,
        Controls::new(Tolerance::new(1e-12, 1e-12), 1e-4),
    )
    .expect("propagated with STM");
    let (_, phi) = split::<6, 42>(&traj.final_state());

    let d = det6(&phi);
    println!("det Φ = {d:.9} (dissipative: < 1)");
    assert!(d > 0.0, "det Φ must stay positive (no reflection): {d}");
    assert!(
        d < 1.0 - 1e-6,
        "drag STM must contract phase volume, det Φ = {d:.9} not < 1"
    );
    // and it must not have collapsed — the contraction over a few revolutions is small
    assert!(
        d > 0.99,
        "det Φ = {d:.9} contracted implausibly far for {BALLISTIC} m²/kg over 8 revs"
    );
}

/// The drag law as it stood before a wind term existed, transplanted verbatim.
#[derive(Clone, Copy)]
struct WindlessDrag {
    ballistic_coefficient: f64,
    earth_rotation_rate: f64,
    atmosphere: ExponentialAtmosphere,
}

impl Acceleration for WindlessDrag {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let omega = S::from_f64(self.earth_rotation_rate);
        let v_rel = [v[0] + omega * r[1], v[1] - omega * r[0], v[2]];
        let speed = (v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2]).sqrt();
        let rho = self.atmosphere.density(t.value(), r);
        let k = S::from_f64(-0.5 * self.ballistic_coefficient) * rho * speed;
        [k * v_rel[0], k * v_rel[1], k * v_rel[2]]
    }
}

/// A wind that rotates rigidly about the pole at `rate`: `w = Ω ẑ × r`.
#[derive(Clone, Copy)]
struct SolidBodyWind {
    rate: f64,
}

impl Wind for SolidBodyWind {
    fn velocity<S: Scalar>(&self, _t: f64, r: &[S; 3]) -> [S; 3] {
        let rate = S::from_f64(self.rate);
        [
            S::from_f64(0.0) - rate * r[1],
            rate * r[0],
            S::from_f64(0.0),
        ]
    }
}

/// States chosen so no component is incidentally zero and both hemispheres are covered.
fn wind_states() -> [([f64; 3], [f64; 3]); 4] {
    [
        ([5.1e6, 3.7e6, 2.4e6], [-2.9e3, 5.4e3, 3.1e3]),
        ([-4.2e6, 5.3e6, -1.9e6], [4.1e3, 2.6e3, -5.2e3]),
        ([6.6e6, -1.1e6, 8.0e5], [7.0e2, 7.4e3, 1.2e3]),
        ([1.3e6, -2.2e6, 6.4e6], [-6.1e3, -3.3e3, 1.7e3]),
    ]
}

#[test]
fn no_wind_reproduces_the_windless_drag_law_bit_for_bit() {
    // A default that shifted every existing trajectory in the last place would be a silent change of
    // answer for every caller who never asked for wind, and no tolerance-based test would see it.
    // Subtracting an exact +0.0 is the identity on every f64, so the claim is bit equality, not 1e-15.
    let windless = WindlessDrag {
        ballistic_coefficient: BALLISTIC,
        earth_rotation_rate: OMEGA_EARTH,
        atmosphere: atmosphere(),
    };
    let with_default = drag(OMEGA_EARTH);
    for (r, v) in wind_states() {
        let want = windless.acceleration::<f64>(0.0, &r, &v);
        let got = with_default.acceleration::<f64>(0.0, &r, &v);
        for k in 0..3 {
            assert_eq!(
                got[k].to_bits(),
                want[k].to_bits(),
                "NoWind perturbed axis {k} at r={r:?}: {:.17e} vs {:.17e}",
                got[k],
                want[k]
            );
        }
    }
}

#[test]
fn a_solid_body_wind_is_a_change_of_co_rotation_rate() {
    // w = Ω ẑ × r enters v_rel exactly where ω⊕ ẑ × r does, so the two must be the same force. This
    // is the statement that the wind reaches the drag law with the right sign and the right frame —
    // a sign error here would read as an atmosphere counter-rotating at 150 m/s and pass every
    // magnitude check in this file. It is also how the 22.6 km/week sensitivity was measured.
    let extra = 150.0 / (RE + 400e3); // a 150 m/s zonal wind at 400 km, as a rotation rate
    let winded = Drag {
        ballistic_coefficient: BALLISTIC,
        earth_rotation_rate: OMEGA_EARTH,
        atmosphere: atmosphere(),
        wind: SolidBodyWind { rate: extra },
    };
    let faster = drag(OMEGA_EARTH + extra);
    for (r, v) in wind_states() {
        let want = faster.acceleration::<f64>(0.0, &r, &v);
        let got = winded.acceleration::<f64>(0.0, &r, &v);
        let scale = want.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for k in 0..3 {
            let rel = (got[k] - want[k]).abs() / scale;
            assert!(
                rel < 1e-14,
                "wind axis {k} at r={r:?}: {:.6e} vs co-rotation {:.6e} ({rel:.2e})",
                got[k],
                want[k]
            );
        }
    }
}

/// A hand check that the tuple form and the direct sum agree, so the composition itself is not the
/// thing under test in the physics above.
#[test]
fn composed_acceleration_is_the_sum_of_its_terms() {
    let r = [7.0e6, 1.0e6, 0.5e6];
    let v = [1.0e3, 7.0e3, 0.5e3];
    let pm = PointMass { mu: MU };
    let dg = drag(OMEGA_EARTH);

    let sum = (pm, dg).acceleration::<f64>(0.0, &r, &v);
    let a_pm = pm.acceleration::<f64>(0.0, &r, &v);
    let a_dg = dg.acceleration::<f64>(0.0, &r, &v);
    for k in 0..3 {
        assert!(
            (sum[k] - (a_pm[k] + a_dg[k])).abs() < 1e-18,
            "term {k} not summed"
        );
    }
}
