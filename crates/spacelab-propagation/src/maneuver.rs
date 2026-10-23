//! Manoeuvres and empirical accelerations — finite (continuous) thrust with mass depletion,
//! instantaneous impulses, and the polynomial / argument-of-latitude-harmonic empirical terms orbit
//! determination uses to absorb un-modelled residuals. Ports of Orekit's
//! `ConstantThrustManeuver`, `ImpulseManeuver`, and `ParametricAcceleration` /
//! `PolynomialAccelerationModel`, validated against them (`tools/orekit-oracle`,
//! `orekit_maneuver.csv`).
//!
//! # Frames
//!
//! Every applied vector here — a thrust direction, an impulse Δv, an empirical direction — is quoted
//! in one of two frames, named once by [`Frame`]: the inertial propagation frame, or the local-orbital
//! **RTN** frame (radial / transverse / normal, a.k.a. RSW / QSW / LVLH). The RTN basis is built from
//! the instantaneous `(r, v)` — `R̂ = r̂`, `Ŵ = (r×v)/‖r×v‖`, `T̂ = Ŵ × R̂` (Orekit `LOFType.QSW`,
//! against which the basis is validated) — so an RTN direction rotates into the inertial frame the
//! propagator integrates. Making the frame a two-variant sum rather than two booleans keeps the illegal
//! "inertial-and-RTN" state unrepresentable, and the one rotation is written over any [`Scalar`] so it
//! threads the state-transition matrix like every other force.
//!
//! # The mass channel (finite thrust)
//!
//! A [`crate::forces::Newtonian`] force evolves the six-element `[r, v]` state; that is enough for a
//! conservative or dissipative acceleration, but **not** for thrust. Thrust's acceleration is
//! `a = (F/m) d̂`, and `m` is not constant: the same burn that accelerates the spacecraft depletes its
//! propellant at `ṁ = −F/(I_sp g₀)`. The acceleration therefore depends on a quantity the six-element
//! state does not carry. Orekit resolves this by making mass a first-class field of its
//! `SpacecraftState` that its `ConstantThrustManeuver` reads and depletes; [`ConstantThrust`] mirrors
//! that exactly, as a **seven-element** `[r, v, m]` [`ScalarSystem<7>`] — the mass rides in `y[6]`, the
//! thrust reads it, and `ṁ` is the seventh derivative. Written over [`Scalar`], the `7×7` Jacobian (and
//! hence a `7²+7` variational STM, including `∂/∂m`) still comes from automatic differentiation.
//!
//! This is a deliberate departure from the `Acceleration` tuple composition of [`crate::forces`]: an
//! [`crate::forces::Acceleration`] cannot express thrust, because its signature has no mass to read or
//! write. So [`ConstantThrust`] instead *owns* the background force — any composed
//! [`crate::forces::Acceleration`] (gravity, drag, SRP, …), evaluated on the `[r, v]` sub-state — and
//! augments it with the thrust term and the mass equation. The trade-off: the thrust model is the
//! outermost wrapper (`Plain(ConstantThrust { background: (PointMass, J2, …), … })`) rather than one
//! more term in a `Newtonian((…))` tuple. The alternative — a `mass: impl Fn(f64) -> f64` closure the
//! caller supplies to a six-element `Acceleration` — was rejected: it cannot deplete mass
//! self-consistently (the closure and the integrator would drift), and it re-introduces the
//! coordinate-vs-derivative coupling the mass channel makes explicit.
//!
//! # Impulses
//!
//! An [`ImpulsiveManeuver`] is an instantaneous Δv: a pure state transform `[r, v] ↦ [r, v + Δv]` (with
//! `Δv` rotated in from its [`Frame`]), position continuous, velocity discontinuous. Orekit wires the
//! same physics as a propagator **event** — an `ImpulseManeuver` is an `EventDetector` whose handler
//! resets the state when a trigger (a date, an apside, …) fires. This crate's integrator has no
//! in-the-loop reset hook: events are located *after* a step over the dense [`crate::driver::Trajectory`]
//! by [`spacelab_coverage::find_events`], not resolved mid-step. Adding a reset-on-event mechanism to
//! the driver is real surgery and out of scope here; the idiomatic composition instead is to integrate
//! up to the trigger time, apply [`ImpulsiveManeuver::apply`], and integrate onward from the reset
//! state — the state transform is the reusable unit, and the caller (or a future event-aware driver)
//! sequences it. The transform is validated against Orekit's `ImpulseManeuver` output directly.

// Manoeuvre and orbital-frame algebra speaks in single letters — (r, v) for the state, (R, T, W) for
// the RTN axes, u for the argument of latitude, F/m and I_sp/g₀ for the rocket equation. These are the
// domain's names, not terse locals.
#![allow(clippy::many_single_char_names, clippy::similar_names)]

use spacelab_autodiff::Scalar;

use crate::forces::Acceleration;
use crate::spacecraft::dot3;
use crate::variational::ScalarSystem;

/// Standard gravity `g₀`, m/s² — the constant that turns specific impulse into an effective exhaust
/// velocity `I_sp g₀` in the rocket equation. Orekit `Constants.G0_STANDARD_GRAVITY`; the CGPM-defined
/// value, **not** any local `g`.
pub const G0_STANDARD_GRAVITY: f64 = 9.806_65;

/// The frame a manoeuvre or empirical vector is quoted in: the inertial propagation frame, or the
/// local-orbital **RTN** frame built from `(r, v)`.
///
/// `Rtn` is the radial / transverse / normal frame — `R̂ = r̂`, `Ŵ = (r×v)/‖r×v‖`, `T̂ = Ŵ × R̂` — a.k.a.
/// RSW / QSW / LVLH, matching Orekit `LOFType.QSW` (validated against it). Naming the frame as a sum
/// type makes the two mutually-exclusive choices one argument instead of two booleans with a runtime
/// check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    /// The inertial propagation frame — the vector is used as given.
    Inertial,
    /// The RTN / RSW local-orbital frame — the vector's `(R, T, W)` components are rotated to inertial
    /// with the basis from the instantaneous `(r, v)`.
    Rtn,
}

impl Frame {
    /// Rotate a vector expressed in this frame into the inertial frame, given the inertial state
    /// `(r, v)`. `Inertial` is the identity; `Rtn` maps `(R, T, W)` components onto the RTN basis.
    #[must_use]
    pub fn rotate_to_inertial<S: Scalar>(self, vec: &[S; 3], r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        match self {
            Frame::Inertial => *vec,
            Frame::Rtn => {
                let [rr, tt, ww] = rtn_basis(r, v);
                [
                    vec[0] * rr[0] + vec[1] * tt[0] + vec[2] * ww[0],
                    vec[0] * rr[1] + vec[1] * tt[1] + vec[2] * ww[1],
                    vec[0] * rr[2] + vec[1] * tt[2] + vec[2] * ww[2],
                ]
            }
        }
    }
}

/// The RTN / RSW orthonormal basis `[R̂, T̂, Ŵ]`, each a unit vector in the inertial frame, from the
/// inertial state `(r, v)`: `R̂ = r̂` (radial), `Ŵ = (r×v)/‖r×v‖` (orbit normal, cross-track),
/// `T̂ = Ŵ × R̂` (in-plane transverse, along the motion for a prograde orbit). This is Orekit's
/// `LOFType.QSW` triad (Q, S, W), against which it is validated.
///
/// `T̂` is a cross product of two orthogonal unit vectors, so it is unit without a further norm — the
/// only normalisation is `r̂` and `Ŵ`.
#[must_use]
pub fn rtn_basis<S: Scalar>(r: &[S; 3], v: &[S; 3]) -> [[S; 3]; 3] {
    let inv_r = dot3(r, r).sqrt().recip();
    let rhat = [r[0] * inv_r, r[1] * inv_r, r[2] * inv_r];
    let h = cross(r, v);
    let inv_h = dot3(&h, &h).sqrt().recip();
    let what = [h[0] * inv_h, h[1] * inv_h, h[2] * inv_h];
    let that = cross(&what, &rhat);
    [rhat, that, what]
}

// -------------------------------------------------------------------------------------------------
// Finite (continuous) constant-thrust manoeuvre
// -------------------------------------------------------------------------------------------------

/// A finite-duration **constant-thrust** manoeuvre with mass depletion, as a seven-element
/// `[r, v, m]` system — a port of Orekit's `ConstantThrustManeuver`.
///
/// Over the burn window `[t_start, t_end)` it adds the thrust acceleration `a = (F/m) d̂` to the
/// background force and depletes mass at the constant flow rate `ṁ = −F/(I_sp g₀)`; outside the
/// window the thrust and the mass rate are zero. `F` is the thrust magnitude (N), `I_sp` the specific
/// impulse (s), `d̂` the (unit) [`direction`](Self::direction) in its [`frame`](Self::frame). The mass
/// is state `y[6]`, so the acceleration reads the *current* depleted mass — the coupling that makes a
/// long burn's late acceleration larger than its early acceleration.
///
/// The window test is on the `f64` time `t` (a discrete on/off that the STM is not defined across, like
/// the shadow gate of [`crate::forces::SolarRadiationPressure`]); everything else is written over
/// [`Scalar`], so `F/m` and an RTN direction carry their derivatives into the `7×7` Jacobian.
///
/// `background` is any composed [`Acceleration`] (gravity, drag, SRP, third body, …), evaluated on the
/// `[r, v]` sub-state; see the module docs for why thrust is the outer seven-element wrapper rather than
/// a term in a six-element [`crate::forces::Newtonian`] tuple. Wrap in [`crate::forces::Plain`] for
/// state-only propagation, or [`crate::variational::Variational`] (`N = 7`, `M = 56`) for the STM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantThrust<A> {
    /// The background acceleration (gravity, drag, …), evaluated on the `[r, v]` sub-state.
    pub background: A,
    /// Thrust magnitude `F`, newtons.
    pub thrust: f64,
    /// Specific impulse `I_sp`, seconds.
    pub isp: f64,
    /// Standard gravity `g₀`, m/s² (see [`G0_STANDARD_GRAVITY`]).
    pub g0: f64,
    /// Unit thrust direction in [`frame`](Self::frame).
    pub direction: [f64; 3],
    /// The frame [`direction`](Self::direction) is quoted in.
    pub frame: Frame,
    /// Burn start, seconds from the propagation epoch (inclusive).
    pub t_start: f64,
    /// Burn end, seconds from the propagation epoch (exclusive).
    pub t_end: f64,
}

impl<A> ConstantThrust<A> {
    /// The propellant mass flow rate `F/(I_sp g₀)`, kg/s (positive; mass decreases by this). Orekit
    /// `ConstantThrustManeuver.getFlowRate()` returns its negation.
    #[must_use]
    pub fn flow_rate(&self) -> f64 {
        self.thrust / (self.isp * self.g0)
    }

    /// Whether the burn is active at time `t` (seconds from epoch): `t ∈ [t_start, t_end)`.
    #[must_use]
    pub fn is_firing(&self, t: f64) -> bool {
        t >= self.t_start && t < self.t_end
    }
}

impl<A: Acceleration> ScalarSystem<7> for ConstantThrust<A> {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 7]) -> [S; 7] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let m = y[6];
        let a = self.background.acceleration(t, &r, &v);

        if self.is_firing(t.value()) {
            let d = [
                S::from_f64(self.direction[0]),
                S::from_f64(self.direction[1]),
                S::from_f64(self.direction[2]),
            ];
            let dir = self.frame.rotate_to_inertial(&d, &r, &v);
            let f_over_m = S::from_f64(self.thrust) / m;
            let mdot = S::from_f64(-self.flow_rate());
            [
                v[0],
                v[1],
                v[2],
                a[0] + f_over_m * dir[0],
                a[1] + f_over_m * dir[1],
                a[2] + f_over_m * dir[2],
                mdot,
            ]
        } else {
            [v[0], v[1], v[2], a[0], a[1], a[2], S::from_f64(0.0)]
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Impulsive manoeuvre
// -------------------------------------------------------------------------------------------------

/// An instantaneous velocity change — a port of Orekit's `ImpulseManeuver` as a pure state transform.
///
/// [`apply`](Self::apply) maps `[r, v] ↦ [r, v + Δv]`, with `Δv` rotated in from its [`Frame`]:
/// position is continuous, only velocity jumps. See the module docs for why this is a transform rather
/// than an integrator event (Orekit models it as an event-detector reset; this crate locates events
/// after the step, so the caller sequences integrate → apply → integrate).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImpulsiveManeuver {
    /// The velocity increment `Δv`, m/s, in [`frame`](Self::frame).
    pub delta_v: [f64; 3],
    /// The frame [`delta_v`](Self::delta_v) is quoted in.
    pub frame: Frame,
}

impl ImpulsiveManeuver {
    /// Apply the impulse to a six-element `[r, v]` state, returning `[r, v + Δv_inertial]`.
    #[must_use]
    pub fn apply(&self, state: &[f64; 6]) -> [f64; 6] {
        let r = [state[0], state[1], state[2]];
        let v = [state[3], state[4], state[5]];
        let dv = self.frame.rotate_to_inertial(&self.delta_v, &r, &v);
        [r[0], r[1], r[2], v[0] + dv[0], v[1] + dv[1], v[2] + dv[2]]
    }

    /// Apply the impulse to a seven-element `[r, v, m]` state (the [`ConstantThrust`] layout), leaving
    /// the mass unchanged — an impulse is instantaneous, so it burns no modelled propellant here.
    #[must_use]
    pub fn apply_with_mass(&self, state: &[f64; 7]) -> [f64; 7] {
        let six = self.apply(&[state[0], state[1], state[2], state[3], state[4], state[5]]);
        [six[0], six[1], six[2], six[3], six[4], six[5], state[6]]
    }
}

// -------------------------------------------------------------------------------------------------
// Empirical accelerations
// -------------------------------------------------------------------------------------------------

/// A scalar acceleration magnitude as a function of time and state — the Strategy an
/// [`EmpiricalAcceleration`] multiplies by its direction, mirroring Orekit's `AccelerationModel`.
/// Written over [`Scalar`] so the empirical term carries its Jacobian.
pub trait AccelerationSignal {
    /// The signed magnitude at time `t` (seconds from epoch) and inertial state `(r, v)`.
    fn signal<S: Scalar>(&self, t: f64, r: &[S; 3], v: &[S; 3]) -> S;
}

/// A polynomial in elapsed time: `Σ c_k (t − t_ref)^k`, `τ = t − t_ref` in seconds — the textbook
/// empirical polynomial, modelling Orekit's `PolynomialAccelerationModel`. Orekit evaluates its
/// coefficients in the shifted variable `1 + (t − t_ref)` (its `signedAmplitude` Horner is
/// `amp·(1 + dt) + c_k`), so this clean form reproduces an Orekit model exactly with `t_ref` moved one
/// second earlier — the bridge the validation test documents. `K` is the number of coefficients
/// (degree + 1); const-generic to keep the array on the stack, matching [`crate::forces::Zonal`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolynomialSignal<const K: usize> {
    /// Reference time `t_ref`, seconds from the propagation epoch.
    pub reference_time: f64,
    /// Coefficients `c_0 … c_{K−1}`, low order first: magnitude `= c₀ + c₁τ + c₂τ² + …`, `τ = t − t_ref`.
    pub coefficients: [f64; K],
}

impl<const K: usize> AccelerationSignal for PolynomialSignal<K> {
    fn signal<S: Scalar>(&self, t: f64, _r: &[S; 3], _v: &[S; 3]) -> S {
        // Horner in τ = t − t_ref (an f64: the signal depends on time alone, not the state), lifted to
        // S at the end so an inertial empirical term is a pure constant of the AD pass.
        let tau = t - self.reference_time;
        let mut acc = 0.0_f64;
        for &c in self.coefficients.iter().rev() {
            acc = acc * tau + c;
        }
        S::from_f64(acc)
    }
}

/// A harmonic in the **argument of latitude** `u`: `c cos(h u) + s sin(h u)` — the classic
/// once-per-revolution (`h = 1`) empirical term orbit determination uses to absorb periodic residuals,
/// resolved per RTN component. `u` is the angle from the ascending node to the satellite in the orbital
/// plane, computed geometrically from `(r, v)` (validated against Orekit's `CircularOrbit.getAlphaV`).
///
/// This is **not** Orekit's `HarmonicAccelerationModel`, which is harmonic in *time* (a fixed period and
/// phase); the argument-of-latitude form is the OD-standard empirical harmonic (GEODYN / GNSS CODE
/// model). `u` is undefined for an exactly equatorial orbit (the node is undefined); like the
/// argument of latitude itself, this term is for inclined orbits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicSignal {
    /// The harmonic number `h` (multiple of the argument of latitude; `1` = once per revolution).
    pub harmonic: f64,
    /// Cosine amplitude `c`.
    pub cos_amplitude: f64,
    /// Sine amplitude `s`.
    pub sin_amplitude: f64,
}

impl AccelerationSignal for HarmonicSignal {
    fn signal<S: Scalar>(&self, _t: f64, r: &[S; 3], v: &[S; 3]) -> S {
        let (sin_u, cos_u) = argument_of_latitude_sincos(r, v);
        // h·u via atan2 so any integer harmonic works; the branch cut at ±π is invisible to the
        // periodic cos/sin for integer h. For h = 1 this equals (cos_u, sin_u) directly.
        let u = sin_u.atan2(cos_u);
        let hu = u * S::from_f64(self.harmonic);
        hu.cos() * S::from_f64(self.cos_amplitude) + hu.sin() * S::from_f64(self.sin_amplitude)
    }
}

/// An empirical acceleration — a signed scalar [`AccelerationSignal`] along a fixed direction in a
/// [`Frame`] — a port of Orekit's `ParametricAcceleration` (`acceleration = direction · signal`).
/// Compose the RTN radial / transverse / normal components as three of these summed through the tuple
/// [`Acceleration`] impls, exactly as Orekit adds three `ParametricAcceleration`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalAcceleration<Sig> {
    /// The (unit) direction in [`frame`](Self::frame). Its magnitude folds into the signal, so a unit
    /// direction keeps the amplitude the signal's.
    pub direction: [f64; 3],
    /// The frame [`direction`](Self::direction) is quoted in.
    pub frame: Frame,
    /// The scalar magnitude model.
    pub signal: Sig,
}

impl<Sig: AccelerationSignal> Acceleration for EmpiricalAcceleration<Sig> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let d = [
            S::from_f64(self.direction[0]),
            S::from_f64(self.direction[1]),
            S::from_f64(self.direction[2]),
        ];
        let dir = self.frame.rotate_to_inertial(&d, r, v);
        let s = self.signal.signal(t.value(), r, v);
        [dir[0] * s, dir[1] * s, dir[2] * s]
    }
}

// -------------------------------------------------------------------------------------------------
// Small vector helpers
// -------------------------------------------------------------------------------------------------

/// Vector cross product over any [`Scalar`].
fn cross<S: Scalar>(a: &[S; 3], b: &[S; 3]) -> [S; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// `(sin u, cos u)` for the argument of latitude `u` (angle from the ascending node to the satellite,
/// measured in the orbital plane in the direction of motion), from the inertial `(r, v)`.
///
/// With `ĥ = (r×v)/‖r×v‖` the orbit normal, `n̂ = (ẑ×ĥ)/‖ẑ×ĥ‖` the ascending-node direction, and
/// `T̂ = ĥ × n̂` the in-plane unit 90° ahead of the node: `cos u = r̂·n̂`, `sin u = r̂·T̂`. Both come out
/// as a unit pair because `r̂` lies in the plane spanned by the orthonormal `(n̂, T̂)`. Undefined for an
/// equatorial orbit (`ẑ×ĥ = 0`).
fn argument_of_latitude_sincos<S: Scalar>(r: &[S; 3], v: &[S; 3]) -> (S, S) {
    let inv_r = dot3(r, r).sqrt().recip();
    let rhat = [r[0] * inv_r, r[1] * inv_r, r[2] * inv_r];
    let h = cross(r, v);
    let inv_h = dot3(&h, &h).sqrt().recip();
    let hhat = [h[0] * inv_h, h[1] * inv_h, h[2] * inv_h];
    // node = ẑ × ĥ = (−ĥ_y, ĥ_x, 0), the ascending-node direction (before normalising).
    let node = [-hhat[1], hhat[0], S::from_f64(0.0)];
    let inv_n = dot3(&node, &node).sqrt().recip();
    let nhat = [node[0] * inv_n, node[1] * inv_n, node[2] * inv_n];
    let that = cross(&hhat, &nhat);
    let cos_u = dot3(&rhat, &nhat);
    let sin_u = dot3(&rhat, &that);
    (sin_u, cos_u)
}

#[cfg(test)]
mod tests {
    // Verbatim Orekit `%.17e` golden vectors (tools/orekit-oracle, orekit_maneuver.csv) — full precision.
    #![allow(
        clippy::float_cmp,
        clippy::unreadable_literal,
        clippy::excessive_precision
    )]
    use super::*;
    use crate::forces::{Plain, PointMass};
    use crate::rk4::rk4_integrate;
    use spacelab_testkit::nan_max;

    const MU: f64 = 3.98600441500000000e+14;

    /// Max per-component relative error (absolute where the reference is zero), reporting the actual.
    fn max_rel(label: &str, got: &[f64], want: &[f64]) -> f64 {
        let mut m = 0.0_f64;
        for i in 0..got.len() {
            let rel = if want[i] == 0.0 {
                got[i].abs()
            } else {
                (got[i] - want[i]).abs() / want[i].abs()
            };
            m = m.max(rel);
        }
        std::println!("{label}: max rel err {m:.3e}");
        m
    }

    // ---- 1. Constant-thrust short arc, matched fixed-step RK4 vs Orekit ConstantThrustManeuver ----
    // A LEO arc under point-mass gravity plus an always-firing constant thrust, both integrated with
    // the identical classical RK4 tableau at a fixed 30 s step (20 steps over 600 s). Because the
    // Butcher tableau, step, force formula and thrust direction match, the only divergence is ~1 ulp
    // per gravity evaluation (op-order of `−μr/|r|³`), so the match is near machine precision.
    const CT_THRUST: f64 = 4.00000000000000000e+02;
    const CT_ISP: f64 = 3.00000000000000000e+02;
    const CT_FLOW: f64 = 1.35962161730390440e-01;
    const CT_DIR: [f64; 3] = [
        5.99999925000014100e-01,
        4.99999937500011760e-01,
        -6.24499921937514700e-01,
    ];
    const CT_MASS0: f64 = 1200.0;
    const CT_SPAN: f64 = 600.0;
    const CT_NSTEPS: u32 = 20;
    const CT_R0: [f64; 3] = [
        2.22700285905359270e+06,
        5.12915631219342350e+06,
        4.19949438742875100e+06,
    ];
    const CT_V0: [f64; 3] = [
        -6.51943386513380700e+03,
        -2.82313073482967400e+02,
        3.80426923262362830e+03,
    ];
    const CT_RF: [f64; 3] = [
        -1.83219886138269260e+06,
        3.96164298050954570e+06,
        5.43753786260741200e+06,
    ];
    const CT_VF: [f64; 3] = [
        -6.53216024504010900e+03,
        -3.47475695170036850e+03,
        1.71043429657034180e+02,
    ];
    const CT_MF: f64 = 1.11842270296176600e+03;
    const CT_ACCEL0: [f64; 3] = [
        1.99999975000004700e-01,
        1.66666645833337230e-01,
        -2.08166640645838200e-01,
    ];

    fn constant_thrust() -> ConstantThrust<PointMass> {
        ConstantThrust {
            background: PointMass { mu: MU },
            thrust: CT_THRUST,
            isp: CT_ISP,
            g0: G0_STANDARD_GRAVITY,
            direction: CT_DIR,
            frame: Frame::Inertial,
            // Fires the whole arc — Orekit's window [−100, 700] s brackets [0, 600] with no interior
            // trigger, so both RK4 sequences see a smooth always-on thrust.
            t_start: -100.0,
            t_end: 700.0,
        }
    }

    /// The burn window is the half-open interval `[t_start, t_end)` the doc comment states.
    ///
    /// Every fixture in this file fires across its whole arc — Orekit's window `[−100, 700]` brackets
    /// the integrated `[0, 600]` deliberately, so that both RK4 sequences see a smooth always-on
    /// thrust — which means `is_firing` returns `true` at every sampled time and three mutants of it
    /// survive: a body replaced by `true`, `&&` weakened to `||`, and `t < t_end` widened to `<=`.
    /// None of them is visible without sampling from outside the window and exactly on both edges.
    #[test]
    fn the_burn_window_is_the_half_open_interval_it_documents() {
        let ct = ConstantThrust {
            t_start: 10.0,
            t_end: 20.0,
            ..constant_thrust()
        };
        for (t, want) in [
            (-1.0e3, false),
            (9.999_999_999, false),
            (10.0, true),
            (15.0, true),
            (19.999_999_999, true),
            (20.0, false),
            (1.0e3, false),
        ] {
            assert_eq!(ct.is_firing(t), want, "is_firing({t}) should be {want}");
        }
    }

    /// The seven-element `[r, v, m]` state integrated with the matched RK4 reproduces Orekit's final
    /// position, velocity and mass; the initial thrust acceleration matches `F/m₀·d̂`; the flow rate is
    /// the rocket-equation value.
    #[test]
    fn constant_thrust_matches_orekit() {
        let ct = constant_thrust();
        assert!((ct.flow_rate() - CT_FLOW).abs() < 1e-15 * CT_FLOW);

        // Thrust acceleration at the initial firing state: eval minus the background gravity.
        let y0 = [
            CT_R0[0], CT_R0[1], CT_R0[2], CT_V0[0], CT_V0[1], CT_V0[2], CT_MASS0,
        ];
        let d0 = ct.eval::<f64>(0.0, &y0);
        let grav = PointMass { mu: MU }.acceleration::<f64>(0.0, &CT_R0, &CT_V0);
        let thrust_accel = [d0[3] - grav[0], d0[4] - grav[1], d0[5] - grav[2]];
        // Measured residuals are ~1e-15 (near bit-exact); the gates keep ~3 orders of libm headroom.
        assert!(max_rel("ct accel0", &thrust_accel, &CT_ACCEL0) < 1e-12);
        assert!(
            (d0[6] + CT_FLOW).abs() < 1e-15 * CT_FLOW,
            "mdot != -flow_rate"
        );

        // Matched fixed-step RK4 propagation of [r, v, m].
        let yf = rk4_integrate::<7>(&Plain(ct), 0.0, &y0, CT_SPAN, CT_NSTEPS);
        let rf = max_rel("ct rf", &yf[0..3], &CT_RF);
        let vf = max_rel("ct vf", &yf[3..6], &CT_VF);
        let mf = ((yf[6] - CT_MF) / CT_MF).abs();
        std::println!("ct mf: rel err {mf:.3e}");
        assert!(rf < 1e-12, "final position vs Orekit: {rf:.3e}");
        assert!(vf < 1e-12, "final velocity vs Orekit: {vf:.3e}");
        assert!(mf < 1e-13, "final mass vs Orekit: {mf:.3e}");
    }

    // ---- 2. Impulsive maneuver vs Orekit ImpulseManeuver (inertial and RTN/QSW) ----
    const IMP_R: [f64; 3] = [
        6.26383676685378050e+06,
        2.73755933437816470e+06,
        2.29806441317037600e+06,
    ];
    const IMP_VPRE: [f64; 3] = [
        -3.18398692573610330e+03,
        6.11664197152401300e+03,
        3.00502609735479800e+03,
    ];

    /// The state transform `v ↦ v + Δv` reproduces Orekit's `ImpulseManeuver` post-impulse velocity,
    /// for an inertial Δv and an RTN (QSW) Δv — the latter validating the RTN→inertial rotation against
    /// Orekit's `LofOffset(QSW)`.
    #[test]
    fn impulse_matches_orekit() {
        let pre = [
            IMP_R[0],
            IMP_R[1],
            IMP_R[2],
            IMP_VPRE[0],
            IMP_VPRE[1],
            IMP_VPRE[2],
        ];

        let inertial = ImpulsiveManeuver {
            delta_v: [50.0, -30.0, 20.0],
            frame: Frame::Inertial,
        };
        let post_in = inertial.apply(&pre);
        let want_in = [
            -3.13398692573610330e+03,
            6.08664197152401300e+03,
            3.02502609735479800e+03,
        ];
        assert!(max_rel("impulse inertial", &post_in[3..6], &want_in) < 1e-12);

        // The seven-element form agrees on `[r, v]` and leaves the mass untouched.
        let post7 =
            inertial.apply_with_mass(&[pre[0], pre[1], pre[2], pre[3], pre[4], pre[5], 950.0]);
        assert_eq!(&post7[0..6], &post_in[0..6]);
        assert_eq!(post7[6], 950.0);

        let rtn = ImpulsiveManeuver {
            delta_v: [15.0, 40.0, -10.0],
            frame: Frame::Rtn,
        };
        let post_rtn = rtn.apply(&pre);
        let want_rtn = [
            -3.18923192231559730e+03,
            6.15872853844793700e+03,
            3.01626045961703130e+03,
        ];
        assert!(max_rel("impulse rtn", &post_rtn[3..6], &want_rtn) < 1e-12);
    }

    // ---- 3. RTN (QSW) basis and argument of latitude vs Orekit ----
    /// The RTN basis matches Orekit's `LOFType.QSW` axes, and the argument of latitude matches Orekit's
    /// `CircularOrbit.getAlphaV`, at a fixed LEO state.
    #[test]
    fn rtn_frame_matches_orekit() {
        let r = [6.80e6, 9.0e5, 2.1e6];
        let v = [-1.2e3, 6.4e3, 3.2e3];
        let [rr, tt, ww] = rtn_basis(&r, &v);
        let want_r = [
            9.47925097196537200e-01,
            1.25460674628953440e-01,
            2.92741574134224700e-01,
        ];
        let want_t = [
            -2.44920597063388650e-01,
            8.74713346478164900e-01,
            4.18199070571759150e-01,
        ];
        let want_w = [
            -2.03597424441099240e-01,
            -4.68119835741466800e-01,
            8.59890637317521500e-01,
        ];
        assert!(max_rel("rtn R", &rr, &want_r) < 1e-13);
        assert!(max_rel("rtn T", &tt, &want_t) < 1e-13);
        assert!(max_rel("rtn W", &ww, &want_w) < 1e-13);

        let alpha_v = 6.10729534713573300e-01_f64;
        let (sin_u, cos_u) = argument_of_latitude_sincos(&r, &v);
        assert!((sin_u - alpha_v.sin()).abs() < 1e-13, "sin u: {sin_u}");
        assert!((cos_u - alpha_v.cos()).abs() < 1e-13, "cos u: {cos_u}");
    }

    // ---- 4. Polynomial empirical acceleration vs Orekit ParametricAcceleration ----
    const POLY_DIR: [f64; 3] = [
        3.03045763365663200e-01,
        -5.05076272276105300e-01,
        8.08122035641768500e-01,
    ];

    /// The polynomial-in-time empirical acceleration matches Orekit's `ParametricAcceleration` +
    /// `PolynomialAccelerationModel` at four epochs. Orekit evaluates its polynomial in `1 + (t − t_ref)`
    /// (from the class bytecode: `amp = amp·(1 + dt) + cₖ`), so this crate's clean `Σ cₖ(t − t_ref)ᵏ`
    /// with `reference_time = 49 s` reproduces Orekit's `t_ref = 50 s` exactly — the documented
    /// one-second bridge between the two conventions.
    #[test]
    fn polynomial_empirical_matches_orekit() {
        let emp = EmpiricalAcceleration {
            direction: POLY_DIR,
            frame: Frame::Inertial,
            signal: PolynomialSignal {
                reference_time: 49.0, // Orekit t_ref = 50 s, evaluated in 1 + (t − t_ref)
                coefficients: [2.0e-7, 3.0e-9, -1.0e-11],
            },
        };
        // Inertial direction and time-only signal → position/velocity are unused; pass a dummy state.
        let r = [7.0e6, 0.0, 0.0];
        let v = [0.0, 7.5e3, 0.0];
        let cases: [(f64, [f64; 3]); 4] = [
            (
                0.0,
                [
                    8.78529667997057800e-09,
                    -1.46421611332842980e-08,
                    2.34274578132548770e-08,
                ],
            ),
            (
                100.0,
                [
                    9.90929341629382100e-08,
                    -1.65154890271563660e-07,
                    2.64247824434501900e-07,
                ],
            ),
            (
                250.0,
                [
                    1.20912229125265980e-07,
                    -2.01520381875443280e-07,
                    3.22432611000709300e-07,
                ],
            ),
            (
                500.0,
                [
                    -1.45768042636517600e-07,
                    2.42946737727529370e-07,
                    -3.88714780364047000e-07,
                ],
            ),
        ];
        let mut worst = 0.0_f64;
        for (t, want) in cases {
            let a = emp.acceleration::<f64>(t, &r, &v);
            worst = nan_max(worst, max_rel(&std::format!("poly t={t}"), &a, &want));
        }
        assert!(worst < 1e-13, "polynomial empirical vs Orekit: {worst:.3e}");
    }

    // ---- 5. Harmonic empirical (argument of latitude) — analytic anchors ----
    /// The argument-of-latitude harmonic evaluates to its cosine amplitude at the ascending node
    /// (`u = 0`) and to its sine amplitude a quarter revolution later (`u = π/2`), for a polar orbit
    /// whose node is along `+x`. This is Orekit's `HarmonicAccelerationModel` sibling but in `u`, not
    /// time, so it is anchored analytically rather than against Orekit.
    #[test]
    fn harmonic_empirical_is_analytic() {
        let sig = HarmonicSignal {
            harmonic: 1.0,
            cos_amplitude: 3.0e-9,
            sin_amplitude: -5.0e-9,
        };
        // Ascending node: r along +x in the equatorial plane, velocity northward (+z) → u = 0.
        let node_r = [7.0e6, 0.0, 0.0];
        let node_v = [0.0, 0.0, 7.5e3];
        let s0 = sig.signal::<f64>(0.0, &node_r, &node_v);
        assert!(
            (s0 - sig.cos_amplitude).abs() < 1e-18,
            "u=0 → cos amp: {s0}"
        );

        // Quarter revolution later: r at +z, velocity −x (continuing the polar orbit) → u = π/2.
        let quarter_r = [0.0, 0.0, 7.0e6];
        let quarter_v = [-7.5e3, 0.0, 0.0];
        let s1 = sig.signal::<f64>(0.0, &quarter_r, &quarter_v);
        assert!(
            (s1 - sig.sin_amplitude).abs() < 1e-18,
            "u=π/2 → sin amp: {s1}"
        );

        // Second harmonic at u = π/2: cos(2u) = −1, sin(2u) = 0 → −cos amplitude.
        let sig2 = HarmonicSignal {
            harmonic: 2.0,
            cos_amplitude: 3.0e-9,
            sin_amplitude: -5.0e-9,
        };
        let s2 = sig2.signal::<f64>(0.0, &quarter_r, &quarter_v);
        assert!(
            (s2 + sig2.cos_amplitude).abs() < 1e-18,
            "2u=π → −cos amp: {s2}"
        );
    }

    // ---- 6. Automatic differentiation: the STM path is real ----
    /// The seven-element constant-thrust system monomorphises under [`spacelab_autodiff::Dual`] and its
    /// AD Jacobian matches central finite differences — including the novel `∂v̇/∂m = −F/m²·d̂` coupling
    /// the mass channel introduces. The same AD gives the `7×7` STM.
    #[test]
    fn constant_thrust_ad_jacobian_matches_finite_differences() {
        use spacelab_autodiff::Dual;
        let ct = constant_thrust();
        let y0 = [
            CT_R0[0], CT_R0[1], CT_R0[2], CT_V0[0], CT_V0[1], CT_V0[2], CT_MASS0,
        ];
        let yd: [Dual<7>; 7] = core::array::from_fn(|i| Dual::<7>::variable(y0[i], i));
        let d = ct.eval::<Dual<7>>(Dual::<7>::constant(0.0), &yd);
        let d_f64 = ct.eval::<f64>(0.0, &y0);
        for i in 0..7 {
            assert!(
                (d[i].value() - d_f64[i]).abs() < 1e-18,
                "value drift on {i}"
            );
        }
        // ∂v̇ₓ/∂m at column 6, rows 3..6, against a mass finite difference.
        let h = 1e-3;
        let mut yp = y0;
        let mut ym = y0;
        yp[6] += h;
        ym[6] -= h;
        let ap = ct.eval::<f64>(0.0, &yp);
        let am = ct.eval::<f64>(0.0, &ym);
        for row in 3..6 {
            let fd = (ap[row] - am[row]) / (2.0 * h);
            let ad = d[row].grad()[6];
            assert!(
                (fd - ad).abs() <= 1e-6 * ad.abs() + 1e-18,
                "∂d[{row}]/∂m: AD {ad:.6e} vs FD {fd:.6e}"
            );
        }
    }

    /// A harmonic RTN empirical acceleration monomorphises under [`spacelab_autodiff::Dual`] and its AD
    /// Jacobian `∂a/∂[r, v]` matches finite differences — the RTN frame and the argument-of-latitude
    /// geometry both differentiate cleanly.
    #[test]
    fn harmonic_rtn_empirical_ad_jacobian_matches_finite_differences() {
        use spacelab_autodiff::Dual;
        let emp = EmpiricalAcceleration {
            direction: [0.0, 1.0, 0.0], // transverse
            frame: Frame::Rtn,
            signal: HarmonicSignal {
                harmonic: 1.0,
                cos_amplitude: 2.0e-8,
                sin_amplitude: 4.0e-8,
            },
        };
        let r = [6.8e6, 9.0e5, 2.1e6];
        let v = [-1.2e3, 6.4e3, 3.2e3];
        let rd: [Dual<6>; 3] = core::array::from_fn(|i| Dual::<6>::variable(r[i], i));
        let vd: [Dual<6>; 3] = core::array::from_fn(|i| Dual::<6>::variable(v[i], i + 3));
        let a = emp.acceleration::<Dual<6>>(Dual::<6>::constant(0.0), &rd, &vd);
        let a_f64 = emp.acceleration::<f64>(0.0, &r, &v);
        for i in 0..3 {
            assert!((a[i].value() - a_f64[i]).abs() < 1e-22, "value drift {i}");
        }
        let state = [r[0], r[1], r[2], v[0], v[1], v[2]];
        let h = 1.0;
        for j in 0..6 {
            let mut sp = state;
            let mut sm = state;
            sp[j] += h;
            sm[j] -= h;
            let ap = emp.acceleration::<f64>(0.0, &[sp[0], sp[1], sp[2]], &[sp[3], sp[4], sp[5]]);
            let am = emp.acceleration::<f64>(0.0, &[sm[0], sm[1], sm[2]], &[sm[3], sm[4], sm[5]]);
            for i in 0..3 {
                let fd = (ap[i] - am[i]) / (2.0 * h);
                let ad = a[i].grad()[j];
                assert!(
                    (fd - ad).abs() <= 1e-5 * ad.abs() + 1e-18,
                    "∂a[{i}]/∂s[{j}]: AD {ad:.6e} vs FD {fd:.6e}"
                );
            }
        }
    }
}
