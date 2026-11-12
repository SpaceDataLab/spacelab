//! Manoeuvres and empirical accelerations — finite (continuous) thrust with mass depletion,
//! instantaneous impulses, and the polynomial / argument-of-latitude-harmonic empirical terms orbit
//! determination uses to absorb un-modelled residuals. Ports of Orekit's
//! `ConstantThrustManeuver`, `ImpulseManeuver`, and `ParametricAcceleration` /
//! `PolynomialAccelerationModel`. `derivations/maneuver.mac` derives the rocket equation from the mass
//! channel, the RTN basis and the argument of latitude from their definitions and the polynomial from
//! its Horner loop, and evaluates the test goldens at 40 digits; Orekit is a comparator
//! (`docs/comparisons.md`).
//!
//! # Frames
//!
//! Every applied vector here — a thrust direction, an impulse Δv, an empirical direction — is quoted
//! in one of two frames, named once by [`Frame`]: the inertial propagation frame, or the local-orbital
//! **RTN** frame (radial / transverse / normal, a.k.a. RSW / QSW / LVLH). The RTN basis is built from
//! the instantaneous `(r, v)` — `R̂ = r̂`, `Ŵ = (r×v)/‖r×v‖`, `T̂ = Ŵ × R̂` (Orekit `LOFType.QSW`) —
//! so an RTN direction rotates into the inertial frame the propagator integrates. Making the frame a
//! two-variant sum rather than two booleans keeps the illegal "inertial-and-RTN" state
//! unrepresentable, and the one rotation is written over any [`Scalar`] so it threads the
//! state-transition matrix like every other force.
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
//! sequences it. Orekit's `ImpulseManeuver` output is the transform's comparator.

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
/// RSW / QSW / LVLH, Orekit's `LOFType.QSW`. Naming the frame as a sum type makes the two
/// mutually-exclusive choices one argument instead of two booleans with a runtime check.
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
/// `LOFType.QSW` triad (Q, S, W).
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
/// plane, computed geometrically from `(r, v)` (Orekit's `CircularOrbit.getAlphaV`).
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
    // Verbatim Orekit `%.17e` vectors (tools/orekit-oracle, orekit_maneuver.csv) and the 40-digit
    // goldens of derivations/maneuver.mac.
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

    /// `|got − want| / |want|`, the norm of the difference relative to the norm of the reference.
    fn relative_distance(got: &[f64], want: &[f64]) -> f64 {
        let diff: f64 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
        let norm: f64 = want.iter().map(|w| w * w).sum();
        libm::sqrt(diff / norm)
    }

    // ---- 1. Constant thrust: a LEO arc under point-mass gravity and an always-firing thrust, the
    // classical RK4 tableau at a fixed 30 s step (20 steps over 600 s). CT_FLOW, CT_ACCEL0, CT_RF, CT_VF
    // and CT_MF are Orekit's `ConstantThrustManeuver` under the same tableau and step (comparator). ----
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

    /// `derivations/maneuver.mac` (4) at 40 digits: `F/(I_sp g₀)`, `F/m₀·d̂`, and `[r, v, m]` after the
    /// 20 RK4 steps.
    const CT_FLOW_MAXIMA: f64 = 1.3596216173039043234e-1;
    const CT_ACCEL0_MAXIMA: [f64; 3] = [
        1.999999750000047e-1,
        1.6666664583333725333e-1,
        -2.0816664064583823333e-1,
    ];
    const CT_FINAL_MAXIMA: [f64; 7] = [
        -1.8321988613826936014e6,
        3.9616429805095439805e6,
        5.4375378626074098227e6,
        -6.5321602450401098252e3,
        -3.4747569517003689538e3,
        1.7104342965703234992e2,
        1.1184227029617657406e3,
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

    /// The burn window is the half-open interval `[t_start, t_end)` the doc comment states. (S2)
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

    fn ct_y0() -> [f64; 7] {
        [
            CT_R0[0], CT_R0[1], CT_R0[2], CT_V0[0], CT_V0[1], CT_V0[2], CT_MASS0,
        ]
    }

    /// The thrust acceleration at the initial firing state: `eval` less the background gravity.
    fn ct_thrust_acceleration0() -> [f64; 3] {
        let d0 = constant_thrust().eval::<f64>(0.0, &ct_y0());
        let grav = PointMass { mu: MU }.acceleration::<f64>(0.0, &CT_R0, &CT_V0);
        [d0[3] - grav[0], d0[4] - grav[1], d0[5] - grav[2]]
    }

    fn ct_final() -> [f64; 7] {
        rk4_integrate::<7>(&Plain(constant_thrust()), 0.0, &ct_y0(), CT_SPAN, CT_NSTEPS)
    }

    /// The distance to Orekit's `ConstantThrustManeuver` under the identical RK4 tableau and step.
    /// [`constant_thrust_is_the_rocket_equation_under_rk4`] decides.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn constant_thrust_distance_to_orekit() {
        let yf = ct_final();
        for (quantity, got, want) in [
            (
                "flow rate",
                &[constant_thrust().flow_rate()][..],
                &[CT_FLOW][..],
            ),
            (
                "initial thrust acceleration",
                &ct_thrust_acceleration0()[..],
                &CT_ACCEL0[..],
            ),
            ("position after RK4 30 s x 20", &yf[0..3], &CT_RF[..]),
            ("velocity after RK4 30 s x 20", &yf[3..6], &CT_VF[..]),
            ("mass after RK4 30 s x 20", &yf[6..7], &[CT_MF][..]),
        ] {
            crate::dsst::comparison_row(
                &std::format!("constant thrust, {quantity}"),
                "relative",
                &[relative_distance(got, want)],
            );
        }
    }

    /// The flow rate is `F/(I_sp g₀)`, `ṁ` its negative, the thrust acceleration `F/m₀·d̂`, and the
    /// seven-element state after 20 steps is the classical RK4 tableau run on `[r, v, m]` at 40 digits
    /// by `derivations/maneuver.mac` (4), whose (1) derives the rocket equation from the mass channel.
    /// Measured: flow exact, acceleration 7.2e-16, the final state 3.8e-16 at worst. (G3)
    #[test]
    fn constant_thrust_is_the_rocket_equation_under_rk4() {
        let ct = constant_thrust();
        let flow = (ct.flow_rate() - CT_FLOW_MAXIMA).abs() / CT_FLOW_MAXIMA;
        assert_eq!(ct.eval::<f64>(0.0, &ct_y0())[6], -ct.flow_rate());
        let accel0 = relative_distance(&ct_thrust_acceleration0(), &CT_ACCEL0_MAXIMA);
        let yf = ct_final();
        let rf = relative_distance(&yf[0..3], &CT_FINAL_MAXIMA[0..3]);
        let vf = relative_distance(&yf[3..6], &CT_FINAL_MAXIMA[3..6]);
        let mf = relative_distance(&yf[6..7], &CT_FINAL_MAXIMA[6..7]);
        println!(
            "constant thrust against 40 digits: flow {flow:.3e}, accel0 {accel0:.3e}, r {rf:.3e}, v {vf:.3e}, m {mf:.3e}"
        );
        assert!(flow < 5e-16, "flow rate {flow:.3e}");
        assert!(accel0 < 3e-15, "initial thrust acceleration {accel0:.3e}");
        assert!(rf < 1.5e-15, "final position {rf:.3e}");
        assert!(vf < 1e-15, "final velocity {vf:.3e}");
        assert!(mf < 1e-15, "final mass {mf:.3e}");
    }

    // ---- 2. Impulsive manoeuvre, inertial and RTN (QSW) ----
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
    const IMP_DV_INERTIAL: [f64; 3] = [50.0, -30.0, 20.0];
    const IMP_DV_RTN: [f64; 3] = [15.0, 40.0, -10.0];
    /// Orekit's `ImpulseManeuver` post-impulse velocities (comparator).
    const OREKIT_IMPULSE_INERTIAL: [f64; 3] = [
        -3.13398692573610330e+03,
        6.08664197152401300e+03,
        3.02502609735479800e+03,
    ];
    const OREKIT_IMPULSE_RTN: [f64; 3] = [
        -3.18923192231559730e+03,
        6.15872853844793700e+03,
        3.01626045961703130e+03,
    ];
    /// `derivations/maneuver.mac` (4) at 40 digits: `v + Δv`, and `v + Δv_R R̂ + Δv_T T̂ + Δv_W Ŵ`.
    const IMPULSE_INERTIAL_MAXIMA: [f64; 3] = [
        -3.1339869257361033e3,
        6.086641971524013e3,
        3.025026097354798e3,
    ];
    const IMPULSE_RTN_MAXIMA: [f64; 3] = [
        -3.1892319223155975326e3,
        6.1587285384479377663e3,
        3.0162604596170314747e3,
    ];

    /// The post-impulse state for each frame.
    fn impulses() -> [[f64; 6]; 2] {
        let pre = [
            IMP_R[0],
            IMP_R[1],
            IMP_R[2],
            IMP_VPRE[0],
            IMP_VPRE[1],
            IMP_VPRE[2],
        ];
        [(IMP_DV_INERTIAL, Frame::Inertial), (IMP_DV_RTN, Frame::Rtn)]
            .map(|(delta_v, frame)| ImpulsiveManeuver { delta_v, frame }.apply(&pre))
    }

    /// The distance to Orekit's `ImpulseManeuver`, inertial and through `LofOffset(QSW)`.
    /// [`the_impulse_adds_the_rotated_delta_v`] decides.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn impulse_distance_to_orekit() {
        let [inertial, rtn] = impulses();
        for (quantity, got, want) in [
            ("inertial", &inertial[3..6], OREKIT_IMPULSE_INERTIAL),
            ("RTN (QSW)", &rtn[3..6], OREKIT_IMPULSE_RTN),
        ] {
            crate::dsst::comparison_row(
                &std::format!("impulsive manoeuvre, post-impulse velocity, {quantity} delta-v"),
                "relative",
                &[relative_distance(got, &want)],
            );
        }
    }

    /// The impulse keeps the position and adds `Δv`, rotated from RTN with the basis of the pre-impulse
    /// state, as `derivations/maneuver.mac` (4) evaluates at 40 digits; the seven-element form leaves
    /// the mass untouched. Measured: inertial exact, RTN 6.0e-17. (G3)
    #[test]
    fn the_impulse_adds_the_rotated_delta_v() {
        let [inertial, rtn] = impulses();
        assert_eq!(&inertial[0..3], &IMP_R[..]);
        assert_eq!(&rtn[0..3], &IMP_R[..]);
        let a = relative_distance(&inertial[3..6], &IMPULSE_INERTIAL_MAXIMA);
        let b = relative_distance(&rtn[3..6], &IMPULSE_RTN_MAXIMA);
        println!("impulse against 40 digits: inertial {a:.3e}, RTN {b:.3e}");
        assert!(a < 3e-16, "inertial {a:.3e}");
        assert!(b < 3e-16, "RTN {b:.3e}");

        let mass = 950.0;
        let post7 = ImpulsiveManeuver {
            delta_v: IMP_DV_INERTIAL,
            frame: Frame::Inertial,
        }
        .apply_with_mass(&[
            IMP_R[0],
            IMP_R[1],
            IMP_R[2],
            IMP_VPRE[0],
            IMP_VPRE[1],
            IMP_VPRE[2],
            mass,
        ]);
        assert_eq!(&post7[0..6], &inertial[..]);
        assert_eq!(post7[6], mass);
    }

    // ---- 3. RTN (QSW) basis and argument of latitude ----
    const RTN_R: [f64; 3] = [6.80e6, 9.0e5, 2.1e6];
    const RTN_V: [f64; 3] = [-1.2e3, 6.4e3, 3.2e3];
    /// Orekit's `LOFType.QSW` axes Q, S, W and `CircularOrbit.getAlphaV` (comparator).
    const OREKIT_RTN: [[f64; 3]; 3] = [
        [
            9.47925097196537200e-01,
            1.25460674628953440e-01,
            2.92741574134224700e-01,
        ],
        [
            -2.44920597063388650e-01,
            8.74713346478164900e-01,
            4.18199070571759150e-01,
        ],
        [
            -2.03597424441099240e-01,
            -4.68119835741466800e-01,
            8.59890637317521500e-01,
        ],
    ];
    const OREKIT_ALPHA_V: f64 = 6.10729534713573300e-01;
    /// `derivations/maneuver.mac` (4) at 40 digits: `R̂`, `T̂`, `Ŵ`, then `(sin u, cos u)`.
    const RTN_MAXIMA: [[f64; 3]; 3] = [
        [
            9.4792509719653700467e-1,
            1.2546067462895342709e-1,
            2.9274157413422466321e-1,
        ],
        [
            -2.4492059706338861479e-1,
            8.7471334647816483425e-1,
            4.1819907057175915703e-1,
        ],
        [
            -2.035974244410992267e-1,
            -4.681198357414667826e-1,
            8.5989063731752135518e-1,
        ],
    ];
    const SIN_COS_U_MAXIMA: [f64; 2] = [5.7346526928335946811e-1, 8.1922987306723868742e-1];

    /// The distance to Orekit's QSW axes and `getAlphaV`.
    /// [`the_rtn_basis_and_the_argument_of_latitude_are_their_definitions`] decides.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn rtn_frame_distance_to_orekit() {
        let basis = rtn_basis(&RTN_R, &RTN_V);
        for (axis, got, want) in [
            ("R", basis[0], OREKIT_RTN[0]),
            ("T", basis[1], OREKIT_RTN[1]),
            ("W", basis[2], OREKIT_RTN[2]),
        ] {
            crate::dsst::comparison_row(
                &std::format!("RTN (QSW) basis, {axis} axis"),
                "relative",
                &[relative_distance(&got, &want)],
            );
        }
        let (sin_u, cos_u) = argument_of_latitude_sincos(&RTN_R, &RTN_V);
        crate::dsst::comparison_row(
            "argument of latitude, (sin u, cos u)",
            "relative",
            &[relative_distance(
                &[sin_u, cos_u],
                &[OREKIT_ALPHA_V.sin(), OREKIT_ALPHA_V.cos()],
            )],
        );
    }

    /// The RTN basis and `(sin u, cos u)` at a LEO state are their definitions, evaluated at 40
    /// digits by `derivations/maneuver.mac` (4), whose (2) proves the construction orthonormal and the
    /// latitude the orbit's `u` for every node and inclination. Measured 1.6e-16. (G3)
    #[test]
    fn the_rtn_basis_and_the_argument_of_latitude_are_their_definitions() {
        let basis = rtn_basis(&RTN_R, &RTN_V);
        let (sin_u, cos_u) = argument_of_latitude_sincos(&RTN_R, &RTN_V);
        let mut worst = relative_distance(&[sin_u, cos_u], &SIN_COS_U_MAXIMA);
        for (got, want) in basis.iter().zip(RTN_MAXIMA) {
            worst = nan_max(worst, relative_distance(got, &want));
        }
        println!("RTN basis and argument of latitude against 40 digits: {worst:.3e}");
        assert!(
            worst < 6e-16,
            "RTN basis or argument of latitude {worst:.3e}"
        );
    }

    // ---- 4. Polynomial empirical acceleration ----
    const POLY_DIR: [f64; 3] = [
        3.03045763365663200e-01,
        -5.05076272276105300e-01,
        8.08122035641768500e-01,
    ];
    const POLY_COEFFICIENTS: [f64; 3] = [2.0e-7, 3.0e-9, -1.0e-11];
    /// The shipped reference time. Orekit's `PolynomialAccelerationModel` evaluates
    /// `amp += amp·dt + cₖ`, the polynomial in `1 + (t − t_ref)` (`derivations/maneuver.mac` (3)), so its
    /// reference date of 50 s is this crate's 49 s.
    const POLY_REFERENCE_TIME: f64 = 49.0;
    const POLY_TIMES: [f64; 4] = [0.0, 100.0, 250.0, 500.0];
    /// Orekit's `ParametricAcceleration` + `PolynomialAccelerationModel` at the four epochs
    /// (comparator).
    const OREKIT_POLY: [[f64; 3]; 4] = [
        [
            8.78529667997057800e-09,
            -1.46421611332842980e-08,
            2.34274578132548770e-08,
        ],
        [
            9.90929341629382100e-08,
            -1.65154890271563660e-07,
            2.64247824434501900e-07,
        ],
        [
            1.20912229125265980e-07,
            -2.01520381875443280e-07,
            3.22432611000709300e-07,
        ],
        [
            -1.45768042636517600e-07,
            2.42946737727529370e-07,
            -3.88714780364047000e-07,
        ],
    ];
    /// `derivations/maneuver.mac` (4) at 40 digits: `Σ cₖ (t − t_ref)ᵏ d̂` at the four epochs.
    const POLY_MAXIMA: [[f64; 3]; 4] = [
        [
            8.785296679970576168e-9,
            -1.4642161133284292647e-8,
            2.3427457813254868815e-8,
        ],
        [
            9.9092934162938209768e-8,
            -1.65154890271563672047e-7,
            2.64247824434501881815e-7,
        ],
        [
            1.20912229125265960168e-7,
            -2.01520381875443253647e-7,
            3.22432611000709213815e-7,
        ],
        [
            -1.45768042636517655832e-7,
            2.42946737727529410353e-7,
            -3.88714780364047066185e-7,
        ],
    ];

    /// The polynomial empirical acceleration at the four epochs. Inertial and time-only, so the state
    /// is not read.
    fn polynomial_accelerations() -> [[f64; 3]; 4] {
        let emp = EmpiricalAcceleration {
            direction: POLY_DIR,
            frame: Frame::Inertial,
            signal: PolynomialSignal {
                reference_time: POLY_REFERENCE_TIME,
                coefficients: POLY_COEFFICIENTS,
            },
        };
        POLY_TIMES.map(|t| emp.acceleration::<f64>(t, &[7.0e6, 0.0, 0.0], &[0.0, 7.5e3, 0.0]))
    }

    /// The distance to Orekit's `ParametricAcceleration` + `PolynomialAccelerationModel`.
    /// [`the_polynomial_empirical_acceleration_is_its_polynomial`] decides.
    #[test]
    #[ignore = "comparator: the distance to Orekit, not a verdict"]
    fn polynomial_empirical_distance_to_orekit() {
        let distances: std::vec::Vec<f64> = polynomial_accelerations()
            .iter()
            .zip(OREKIT_POLY)
            .map(|(got, want)| relative_distance(got, &want))
            .collect();
        crate::dsst::comparison_row(
            "polynomial empirical acceleration, Orekit reference date one second later",
            "relative",
            &distances,
        );
    }

    /// The polynomial empirical acceleration is `Σ cₖ (t − t_ref)ᵏ d̂` at four epochs, as
    /// `derivations/maneuver.mac` (4) evaluates at 40 digits. Measured 4.3e-16. (G3)
    #[test]
    fn the_polynomial_empirical_acceleration_is_its_polynomial() {
        let mut worst = 0.0_f64;
        for (got, want) in polynomial_accelerations().iter().zip(POLY_MAXIMA) {
            worst = nan_max(worst, relative_distance(got, &want));
        }
        println!("polynomial empirical acceleration against 40 digits: {worst:.3e}");
        assert!(
            worst < 1.5e-15,
            "polynomial empirical acceleration {worst:.3e}"
        );
    }

    // ---- 5. Harmonic empirical (argument of latitude) — analytic anchors ----
    /// The argument-of-latitude harmonic evaluates to its cosine amplitude at the ascending node
    /// (`u = 0`) and to its sine amplitude a quarter revolution later (`u = π/2`), for a polar orbit
    /// whose node is along `+x`. This is Orekit's `HarmonicAccelerationModel` sibling but in `u`, not
    /// time, so it is anchored analytically rather than against Orekit. (G3)
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
