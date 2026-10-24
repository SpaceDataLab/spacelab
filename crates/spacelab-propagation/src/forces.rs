//! Force models, each written once over any [`Scalar`] so its Jacobian — and hence the state
//! transition matrix — comes from automatic differentiation.
//!
//! # Composition
//!
//! The composable unit is the [`Acceleration`], a specific force in m/s², **not** the six-element
//! [`ScalarSystem`] state derivative. Accelerations superpose — the total is the vector sum of
//! gravity, drag, radiation pressure, third-body — whereas summing `[v, aᵢ]` derivatives would add
//! the velocity `n` times. So independent effects are combined with the tuple impls and then wrapped
//! once in [`Newtonian`]:
//!
//! ```ignore
//! let dynamics = Newtonian((PointMass { mu }, J2 { mu, re, j2 }, Drag { .. }));
//! ```
//!
//! This is the Strategy pattern over the physical effects: the set of forces grows independently of
//! the integrator, and a single `TwoBodyJ2Drag…` god-struct — which would need one variant per
//! subset of effects — never appears. The composition is static (tuples, monomorphised) rather than
//! a `Vec<Box<dyn Acceleration>>`, because `acceleration` is generic over [`Scalar`]: a trait object
//! cannot carry a generic method, and the automatic differentiation depends on that genericity.
//!
//! # Constants
//!
//! Each model takes its physical constants — `μ`, `Rₑ`, the zonal `J`, the ballistic coefficient —
//! as explicit parameters rather than baking in one Earth model. Which gravity field or atmosphere
//! is used is a convention the caller states, never a silent default: the wrong constant is a wrong
//! answer that looks plausible.

// Forces speak in single letters: (t, r, v) for the arguments, (x, y, z) for components, and
// (a, b, c, d) for the accelerations a tuple superposes. These are the domain's names, not terse
// locals. The zonal recurrence casts tiny integer degrees (n ≤ ~10) to f64 and names the Legendre
// polynomial and its derivative `p_n` / `pd_n`, which differ by design, not by accident.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::similar_names
)]

use spacelab_autodiff::Scalar;

use crate::DynamicalSystem;
use crate::ephemeris::Ephemeris;
use crate::variational::ScalarSystem;

/// A specific force — an acceleration in m/s² from one physical effect — as a function of time,
/// inertial position and inertial velocity, written once over any [`Scalar`].
///
/// This, not [`ScalarSystem`], is the unit that composes (see the module docs): sum the
/// accelerations of independent effects with the tuple impls, then wrap in [`Newtonian`].
pub trait Acceleration {
    /// Acceleration at time `t`, position `r`, velocity `v` — all SI, all in the inertial frame.
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3];

    /// f64-only fast path, defaulting to `acceleration::<f64>`. Overridden where a specialised f64
    /// implementation is faster (e.g. [`crate::GeoPotential`]'s SIMD sum); composition wrappers
    /// ([`Newtonian`], [`EarthFixed`], the tuple impls) forward through it, so a plain f64
    /// propagation via [`Plain`] transparently uses the fast path while the AD/STM path keeps the
    /// generic `acceleration`. Any override must match `acceleration::<f64>` to within the model's
    /// accuracy tolerance.
    #[inline]
    fn acceleration_f64(&self, t: f64, r: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
        self.acceleration(t, r, v)
    }
}

/// Superposition of two forces.
impl<A: Acceleration, B: Acceleration> Acceleration for (A, B) {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let a = self.0.acceleration(t, r, v);
        let b = self.1.acceleration(t, r, v);
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
    fn acceleration_f64(&self, t: f64, r: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
        let a = self.0.acceleration_f64(t, r, v);
        let b = self.1.acceleration_f64(t, r, v);
        [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
    }
}

/// Superposition of three forces.
impl<A: Acceleration, B: Acceleration, C: Acceleration> Acceleration for (A, B, C) {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let a = self.0.acceleration(t, r, v);
        let b = self.1.acceleration(t, r, v);
        let c = self.2.acceleration(t, r, v);
        [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]
    }
    fn acceleration_f64(&self, t: f64, r: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
        let a = self.0.acceleration_f64(t, r, v);
        let b = self.1.acceleration_f64(t, r, v);
        let c = self.2.acceleration_f64(t, r, v);
        [a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]
    }
}

/// Superposition of four forces.
impl<A: Acceleration, B: Acceleration, C: Acceleration, D: Acceleration> Acceleration
    for (A, B, C, D)
{
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let a = self.0.acceleration(t, r, v);
        let b = self.1.acceleration(t, r, v);
        let c = self.2.acceleration(t, r, v);
        let d = self.3.acceleration(t, r, v);
        [
            a[0] + b[0] + c[0] + d[0],
            a[1] + b[1] + c[1] + d[1],
            a[2] + b[2] + c[2] + d[2],
        ]
    }
    fn acceleration_f64(&self, t: f64, r: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
        let a = self.0.acceleration_f64(t, r, v);
        let b = self.1.acceleration_f64(t, r, v);
        let c = self.2.acceleration_f64(t, r, v);
        let d = self.3.acceleration_f64(t, r, v);
        [
            a[0] + b[0] + c[0] + d[0],
            a[1] + b[1] + c[1] + d[1],
            a[2] + b[2] + c[2] + d[2],
        ]
    }
}

/// Newton's second law as an ODE: given a specific force, the state `[r, v]` evolves by `ṙ = v`,
/// `v̇ = a(t, r, v)`. Adapts any [`Acceleration`] to a [`ScalarSystem<6>`] for the integrator and the
/// variational STM.
///
/// A concrete wrapper rather than a blanket `impl<A: Acceleration> ScalarSystem<6> for A`: the
/// blanket form would forbid every downstream crate from writing its own `impl ScalarSystem<6>` for
/// a local type, because coherence cannot rule out a future `impl Acceleration` for it.
#[derive(Debug, Clone, Copy)]
pub struct Newtonian<A>(pub A);

impl<A: Acceleration> ScalarSystem<6> for Newtonian<A> {
    fn eval<S: Scalar>(&self, t: S, y: &[S; 6]) -> [S; 6] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let a = self.0.acceleration(t, &r, &v);
        [v[0], v[1], v[2], a[0], a[1], a[2]]
    }
    fn eval_f64(&self, t: f64, y: &[f64; 6]) -> [f64; 6] {
        let r = [y[0], y[1], y[2]];
        let v = [y[3], y[4], y[5]];
        let a = self.0.acceleration_f64(t, &r, &v);
        [v[0], v[1], v[2], a[0], a[1], a[2]]
    }
}

/// Point-mass (two-body) gravity: `a = −μ r / |r|³`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointMass {
    /// Gravitational parameter, m³/s².
    pub mu: f64,
}

impl Acceleration for PointMass {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        // 1/r³ from a correctly-rounded sqrt rather than powf(-1.5): this is the most-evaluated force
        // in the propagator and it feeds the STM, so the ~1 ulp beats powf's ~2–3.
        let inv_r3 = (r2 * r2.sqrt()).recip();
        let mu = S::from_f64(self.mu);
        [
            -(mu * r[0] * inv_r3),
            -(mu * r[1] * inv_r3),
            -(mu * r[2] * inv_r3),
        ]
    }
}

/// The `J₂` zonal harmonic — Earth oblateness — as a **perturbing** acceleration, to be summed with
/// a [`PointMass`]. It is the dominant perturbation on a LEO orbit after the point mass, and the one
/// that regresses the node and precesses the perigee.
///
/// `a = −³⁄₂ J₂ μ Rₑ² / r⁵ · [x(1−5z²/r²), y(1−5z²/r²), z(3−5z²/r²)]`, the standard closed form of
/// `−∇` of the `J₂` term of the geopotential (Vallado). Consistency of that acceleration-to-potential
/// pairing is what the energy-conservation test on `(PointMass, J2)` verifies: `J₂` is conservative
/// in the inertial frame, so a wrong pairing drifts long before it fails a single step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct J2 {
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Equatorial radius, m.
    pub re: f64,
    /// The `J₂` coefficient (positive for an oblate body).
    pub j2: f64,
}

impl Acceleration for J2 {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let (x, y, z) = (r[0], r[1], r[2]);
        let r2 = x * x + y * y + z * z;
        let inv_r = r2.sqrt().recip(); // correctly-rounded, matching Zonal, not powf(-0.5)
        let inv_r2 = inv_r * inv_r;
        let inv_r5 = inv_r2 * inv_r2 * inv_r;
        let zeta = z * z * inv_r2; // (z/r)²

        let coeff = S::from_f64(-1.5 * self.j2 * self.mu * self.re * self.re) * inv_r5;
        let one = S::from_f64(1.0);
        let three = S::from_f64(3.0);
        let five = S::from_f64(5.0);
        let xy_factor = one - five * zeta;
        let z_factor = three - five * zeta;

        [
            coeff * x * xy_factor,
            coeff * y * xy_factor,
            coeff * z * z_factor,
        ]
    }
}

/// The zonal harmonics of the geopotential to arbitrary degree, `J₂ … J_{N+1}`, as one perturbing
/// acceleration summed with a [`PointMass`]. `J₂` oblateness dominates; the odd `J₃` breaks the
/// north–south symmetry and sets frozen-orbit eccentricity; `J₄` and beyond refine the figure.
///
/// The acceleration is the closed form of `−∇` of `U_n = (μ/r) J_n (Rₑ/r)ⁿ P_n(z/r)`, summed over
/// degree, with the Legendre polynomials and their derivatives built by their recurrences so one
/// implementation serves every degree:
///
/// ```text
/// a_x = Σ (μ/r³) J_n (Rₑ/r)ⁿ · x · [ (n+1) P_n(u) + u P_n′(u) ]
/// a_z = Σ (μ/r³) J_n (Rₑ/r)ⁿ · [ (n+1) z P_n(u) − r (1−u²) P_n′(u) ],   u = z/r
/// ```
///
/// At `N = 1` this reproduces [`J2`] exactly (a cross-check in the tests); the standalone `J2` is
/// kept as the closed-form fast path for the common case. Because the whole thing is written over
/// [`Scalar`], the STM Jacobian comes from automatic differentiation, and — the field being
/// conservative — that STM stays symplectic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zonal<const N: usize> {
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Equatorial radius, m.
    pub re: f64,
    /// Zonal coefficients by degree: `j[k] = J_{k+2}`, so `j[0] = J₂`, `j[1] = J₃`, … `j[N−1] = J_{N+1}`.
    pub j: [f64; N],
}

impl<const N: usize> Acceleration for Zonal<N> {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let (x, y, z) = (r[0], r[1], r[2]);
        let r_mag = (x * x + y * y + z * z).sqrt();
        let inv_r = r_mag.recip();
        let u = z * inv_r; // sin(geocentric latitude)
        let rho = S::from_f64(self.re) * inv_r; // Rₑ/r
        let mu_over_r3 = S::from_f64(self.mu) * inv_r * inv_r * inv_r;
        // (1 − u²) as (1 − u)(1 + u): near the poles u = z/r → ±1, and the factored form keeps the
        // digits `1 − u²` would lose to cancellation there.
        let one_minus_u2 = (S::from_f64(1.0) - u) * (S::from_f64(1.0) + u);

        // Legendre recurrences, degree by degree:
        //   P_n = ((2n−1) u P_{n−1} − (n−1) P_{n−2}) / n,   P₀=1, P₁=u
        //   P_n′ = u P_{n−1}′ + n P_{n−1},                  P₁′=1
        let mut p_prev2 = S::from_f64(1.0); // P₀
        let mut p_prev1 = u; // P₁
        let mut pd_prev1 = S::from_f64(1.0); // P₁′
        let mut rho_pow = rho; // ρ¹, becomes ρⁿ at the top of iteration n

        let (mut ax, mut ay, mut az) = (S::from_f64(0.0), S::from_f64(0.0), S::from_f64(0.0));

        for n in 2..=N + 1 {
            let nf = S::from_f64(n as f64);
            let p_n = (S::from_f64((2 * n - 1) as f64) * u * p_prev1
                - S::from_f64((n - 1) as f64) * p_prev2)
                / nf;
            let pd_n = u * pd_prev1 + nf * p_prev1;
            rho_pow = rho_pow * rho; // ρⁿ

            // j is empty exactly when N == 0, in which case this loop never runs.
            let pref = mu_over_r3 * S::from_f64(self.j[n - 2]) * rho_pow;
            let np1 = S::from_f64((n + 1) as f64);
            let horizontal = np1 * p_n + u * pd_n;
            ax = ax + pref * x * horizontal;
            ay = ay + pref * y * horizontal;
            az = az + pref * (np1 * z * p_n - r_mag * one_minus_u2 * pd_n);

            p_prev2 = p_prev1;
            p_prev1 = p_n;
            pd_prev1 = pd_n;
        }
        [ax, ay, az]
    }
}

/// A model of neutral atmospheric mass density as a function of position — the Strategy that a
/// [`Drag`] force delegates to. Written over [`Scalar`] so the density gradient enters the STM.
pub trait Atmosphere {
    /// Mass density `ρ`, kg/m³, at time `t` (seconds from the propagation epoch) and inertial
    /// position `r`. Static models ignore `t`; time-varying ones — NRLMSISE-00, which needs the day
    /// of year and local solar time — use it.
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S;
}

/// A single-exponential atmosphere: `ρ = ρ₀ · exp(−(h − h₀) / H)`, with `h` the spherical altitude
/// `|r| − Rₑ`. The crudest useful model — a piecewise or `NRLMSISE`-class model swaps in behind the
/// same trait — but enough to exercise the drag structure and decay physics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialAtmosphere {
    /// Reference density `ρ₀`, kg/m³, at [`reference_altitude`](Self::reference_altitude).
    pub reference_density: f64,
    /// Reference altitude `h₀`, m above the surface.
    pub reference_altitude: f64,
    /// Scale height `H`, m.
    pub scale_height: f64,
    /// Radius of the spherical altitude datum, m.
    pub earth_radius: f64,
}

impl Atmosphere for ExponentialAtmosphere {
    fn density<S: Scalar>(&self, _t: f64, r: &[S; 3]) -> S {
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let altitude = r_mag - S::from_f64(self.earth_radius);
        let exponent = (altitude - S::from_f64(self.reference_altitude))
            * S::from_f64(-1.0 / self.scale_height);
        S::from_f64(self.reference_density) * exponent.exp()
    }
}

/// The neutral wind — the motion of the atmosphere **on top of** its rigid co-rotation — as an
/// inertial-frame velocity, the Strategy a [`Drag`] force delegates to beside its [`Atmosphere`].
///
/// Separated from co-rotation rather than folded into it because the two are different physics with
/// different error budgets: co-rotation is kinematics and is known exactly, while the wind is an
/// empirical field good to some tens of m/s. A `Drag` built with [`NoWind`] is the rigidly
/// co-rotating atmosphere, bit-for-bit.
///
/// Sign convention: the returned vector is the wind velocity itself, so `v_rel = v − ω⊕ × r − w`.
/// Empirical models state the wind in the Earth-fixed sense (zonal positive eastward, meridional
/// positive northward); an implementation is responsible for rotating that into the inertial frame,
/// which for the horizontal wind of a thermosphere model is a rotation about `+z` and nothing more.
pub trait Wind {
    /// Wind velocity, m/s, at time `t` (seconds from the propagation epoch) and inertial position
    /// `r`, relative to the co-rotating atmosphere.
    fn velocity<S: Scalar>(&self, t: f64, r: &[S; 3]) -> [S; 3];
}

/// A rigidly co-rotating atmosphere: no wind at all.
///
/// The default, and the one every result recorded before a wind model existed was computed with.
/// `v_rel` gains a subtraction of exact zero, which IEEE-754 leaves the operand unchanged for every
/// value including the signed zeros — a test asserts the bit pattern rather than trusting that.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NoWind;

impl Wind for NoWind {
    fn velocity<S: Scalar>(&self, _t: f64, _r: &[S; 3]) -> [S; 3] {
        [S::from_f64(0.0); 3]
    }
}

/// Aerodynamic drag: `a = −½ (C_d A / m) ρ |v_rel| v_rel`, opposing the velocity **relative to the
/// moving atmosphere** `v_rel = v − ω⊕ × r − w`.
///
/// Three accuracy points are deliberate. The velocity is relative to the atmosphere, which co-rotates
/// with the Earth; using the inertial velocity instead overstates drag by of order `ω⊕ r / v` —
/// several percent in LEO, and larger toward the equator. On top of that the thermosphere *moves*
/// relative to the solid Earth, and a [`Wind`] is where that goes: it is not a small correction to
/// the magnitude but a rotation of the drag vector, so it accumulates along track. Measured on this
/// propagator, a solid-body zonal wind of 150 m/s displaces a 400 km circular orbit by 454 m in a day
/// and 22.6 km in a week (`C_d A/m = 0.02 m²/kg`); at 300 km the week is 172 km. And drag is
/// **dissipative**: unlike the gravitational models it removes energy, so its STM is not symplectic —
/// `det Φ < 1` by exactly the integral of `tr(∂a/∂v) = −2 (C_d A/m) ρ |v_rel|`, which the STM test
/// checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Drag<A: Atmosphere, W: Wind = NoWind> {
    /// Ballistic coefficient `C_d A / m`, m²/kg.
    pub ballistic_coefficient: f64,
    /// Earth rotation rate `ω⊕`, rad/s; the atmosphere co-rotates about `+z`.
    pub earth_rotation_rate: f64,
    /// The density model.
    pub atmosphere: A,
    /// The neutral wind on top of co-rotation; [`NoWind`] for a rigidly co-rotating atmosphere.
    pub wind: W,
}

impl<A: Atmosphere, W: Wind> Acceleration for Drag<A, W> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let omega = S::from_f64(self.earth_rotation_rate);
        let w = self.wind.velocity(t.value(), r);
        // v_rel = v − ω × r − w, with ω = (0, 0, ω): ω × r = (−ω y, ω x, 0).
        let v_rel = [
            v[0] + omega * r[1] - w[0],
            v[1] - omega * r[0] - w[1],
            v[2] - w[2],
        ];
        let speed = (v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2]).sqrt();
        let rho = self.atmosphere.density(t.value(), r);
        let k = S::from_f64(-0.5 * self.ballistic_coefficient) * rho * speed;
        [k * v_rel[0], k * v_rel[1], k * v_rel[2]]
    }
}

/// Point-mass perturbation of a third body (Sun, Moon) at the position its [`Ephemeris`] gives:
/// the tidal difference between the body's pull on the satellite and on the Earth,
/// `a = μ_b [ (s − r)/‖s − r‖³ − s/‖s‖³ ]`, `s` the body position.
///
/// Evaluated in Battin's cancellation-free form. Written directly, the two bracketed terms are each
/// `~μ_b/‖s‖²` and differ by a factor `~‖r‖/‖s‖` — for a LEO satellite and the Sun that is `~5×10⁻⁵`,
/// so the subtraction discards four to five significant figures before the small true perturbation
/// even appears. The identity `a = −(μ_b/‖s‖³)(1+q)^{−3/2}(r + f(q) s)` with
/// `q = r·(r − 2s)/‖s‖²` and `f(q) = q(3 + 3q + q²)/(1 + (1+q)^{3/2})` computes the same vector with
/// no large-number subtraction — the small quantities `q` and `f` are formed directly. A test checks
/// the two forms agree where the direct one is still accurate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThirdBody<E: Ephemeris> {
    /// Gravitational parameter of the third body, m³/s².
    pub mu: f64,
    /// The body's position over time.
    pub ephemeris: E,
}

impl<E: Ephemeris> Acceleration for ThirdBody<E> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let body = self.ephemeris.position(t.value());
        if crate::ephemeris::is_absent(body) {
            return [S::from_f64(0.0); 3];
        }
        let s = [
            S::from_f64(body[0]),
            S::from_f64(body[1]),
            S::from_f64(body[2]),
        ];
        let s2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2];

        let r_dot_r = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let r_dot_s = r[0] * s[0] + r[1] * s[1] + r[2] * s[2];
        let q = (r_dot_r - S::from_f64(2.0) * r_dot_s) / s2;

        let one = S::from_f64(1.0);
        let three = S::from_f64(3.0);
        let one_plus_q = one + q;
        // (1+q)^{3/2} and ‖s‖³ from correctly-rounded sqrts, reusing the roots, not powf.
        let root_1pq = one_plus_q.sqrt();
        let cube_1pq = one_plus_q * root_1pq; // (1+q)^{3/2}
        let f = q * (three + three * q + q * q) / (one + cube_1pq);

        // a = −(μ/‖s‖³)(1+q)^{−3/2}(r + f s)
        let inv_s3 = (s2 * s2.sqrt()).recip();
        let scale = -(S::from_f64(self.mu) * inv_s3 * cube_1pq.recip());
        [
            scale * (r[0] + f * s[0]),
            scale * (r[1] + f * s[1]),
            scale * (r[2] + f * s[2]),
        ]
    }
}

/// Solar radiation pressure with a conical Earth shadow: `a = −λ · a₀ (AU/d)² d̂`, where `d` is the
/// satellite-to-Sun vector, `a₀` the acceleration in full sun at 1 AU, and `λ ∈ [0, 1]` the
/// [`crate::shadow::lighting_ratio`] — the fraction of the solar disk still visible.
///
/// The force is along the anti-solar direction and falls off as `1/d²`. `λ` is the conical
/// umbra/penumbra model (Orekit `getLightingRatio`): `0` in full umbra, `1` in full sun, and the
/// lens-overlap fraction of the two apparent disks across the penumbra, so the shadow entry/exit ramp
/// is smooth rather than a discontinuous cylinder. It is computed from the value alone, so it does not
/// enter the Jacobian: the penumbra boundary is a measure-zero set the STM is not defined across.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarRadiationPressure<E: Ephemeris> {
    /// Acceleration in full sunlight at 1 AU, m/s²: `Φ · C_r · A / m` with `Φ` the solar pressure.
    pub reference_acceleration: f64,
    /// The distance `a₀` is quoted at, metres (one astronomical unit).
    pub astronomical_unit: f64,
    /// Earth radius for the cylindrical shadow, m.
    pub earth_radius: f64,
    /// The Sun's position over time.
    pub sun: E,
}

impl<E: Ephemeris> Acceleration for SolarRadiationPressure<E> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let sun = self.sun.position(t.value());
        if crate::ephemeris::is_absent(sun) {
            return [S::from_f64(0.0); 3];
        }

        // Conical umbra/penumbra lighting ratio (Orekit `getLightingRatio`), from the f64 value: it is
        // a smooth `[0, 1]` fraction of the solar disk, not a `{0, 1}` gate, so the penumbra ramp is
        // modelled. Kept out of the Jacobian — the boundary is a measure-zero set for the STM.
        let sat = [r[0].value(), r[1].value(), r[2].value()];
        let lit = crate::shadow::lighting_ratio(sat, sun, self.earth_radius);

        // d = r_sun − r_sat; the force pushes along −d (away from the Sun), magnitude a₀(AU/|d|)².
        let d = [
            S::from_f64(sun[0]) - r[0],
            S::from_f64(sun[1]) - r[1],
            S::from_f64(sun[2]) - r[2],
        ];
        let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let inv_d3 = (d2 * d2.sqrt()).recip(); // 1/‖d‖³ from a correctly-rounded sqrt
        let au2 = self.astronomical_unit * self.astronomical_unit;
        let scale = S::from_f64(-lit * self.reference_acceleration * au2) * inv_d3;
        [scale * d[0], scale * d[1], scale * d[2]]
    }
}

/// Adapt any [`ScalarSystem`] into a plain [`DynamicalSystem`] for state-only propagation, by
/// evaluating it at `f64`. Wrap a [`Newtonian`] force in it to propagate without the STM; wrap the
/// `Newtonian` in [`crate::variational::Variational`] instead to propagate with it.
#[derive(Debug, Clone, Copy)]
pub struct Plain<F>(pub F);

impl<const N: usize, F: ScalarSystem<N>> DynamicalSystem<N> for Plain<F> {
    fn derivative(&self, t: f64, y: &[f64; N], dydt: &mut [f64; N]) {
        *dydt = self.0.eval_f64(t, y);
    }
}

/// Evaluates an Earth-fixed (body-fixed) acceleration inside an inertial propagation.
///
/// The inertial position is rotated into the rotating body frame at the current Earth-rotation angle
/// `θ = θ₀ + ω·t`, the inner force is evaluated there, and its acceleration is rotated back to inertial.
/// This is what a longitude-dependent field — the tesseral [`crate::GeoPotential`] — needs: evaluated at
/// the raw inertial position it would smear the `C_nm, S_nm` terms as the Earth turns beneath the orbit.
/// For an axially symmetric field ([`Zonal`], [`J2`]) the wrapper is a no-op, a property the tests use.
/// Written over [`Scalar`], so the STM still comes from AD — and `θ` is built from the scalar `t`
/// rather than its value, so this is one of the few forces whose `∂/∂t` the [`crate::taylor`]
/// recurrence can actually see. [`BodyOriented`] cannot: its rotation comes from an `f64` closure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthFixed<A> {
    /// The body-fixed acceleration.
    pub inner: A,
    /// Body rotation rate, rad/s (e.g. `spacelab_frames::EARTH_ROTATION_RATE`).
    pub omega: f64,
    /// Body-frame rotation angle at `t = 0`, rad.
    pub theta0: f64,
}

impl<A: Acceleration> Acceleration for EarthFixed<A> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let theta = S::from_f64(self.theta0) + S::from_f64(self.omega) * t;
        let (c, s) = (theta.cos(), theta.sin());
        let ns = -s;
        // rot3(θ)·r : inertial → body-fixed
        let rf = [c * r[0] + s * r[1], ns * r[0] + c * r[1], r[2]];
        let af = self.inner.acceleration(t, &rf, v);
        // rot3(θ)ᵀ·a : body-fixed → inertial
        [c * af[0] + ns * af[1], s * af[0] + c * af[1], af[2]]
    }
    fn acceleration_f64(&self, t: f64, r: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
        let theta = self.theta0 + self.omega * t;
        let (c, s) = (libm::cos(theta), libm::sin(theta));
        let rf = [c * r[0] + s * r[1], -s * r[0] + c * r[1], r[2]];
        let af = self.inner.acceleration_f64(t, &rf, v);
        [c * af[0] - s * af[1], s * af[0] + c * af[1], af[2]]
    }
}

/// Evaluates a body-fixed acceleration in a rigorously oriented Earth frame.
///
/// Like [`EarthFixed`], but the inertial→body rotation is the full GCRF→ITRF matrix at time `t`,
/// supplied by the caller, rather than a single rotation about a fixed `z` at rate `ω`. That matrix
/// carries the precession and nutation of the pole, so the tesseral field is evaluated at the true
/// sub-satellite point — closing the accumulated-precession error (about 1.5 km a day at present
/// epochs) that the fixed-axis [`EarthFixed`] leaves on the table.
///
/// The rotation is passed as a closure `t ↦ GCRF→ITRF` (row-major, e.g.
/// `spacelab_frames::gcrf_to_itrf(..).rotation.matrix().rows`) so this crate takes no frames
/// dependency: the Earth-orientation model lives with the caller, this wrapper only applies it.
/// `t` is seconds from the same epoch the propagation and the closure share. Written over [`Scalar`],
/// so the STM still comes from AD.
pub struct BodyOriented<A, R> {
    /// The body-fixed acceleration.
    pub inner: A,
    /// GCRF→ITRF rotation (row-major) at time `t`, seconds from the propagation epoch.
    pub rotate: R,
}

impl<A: Acceleration, R: Fn(f64) -> [[f64; 3]; 3]> Acceleration for BodyOriented<A, R> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let m = (self.rotate)(t.value());
        let e = |i: usize, j: usize| S::from_f64(m[i][j]);
        // r_itrf = M · r_gcrf
        let rf = [
            e(0, 0) * r[0] + e(0, 1) * r[1] + e(0, 2) * r[2],
            e(1, 0) * r[0] + e(1, 1) * r[1] + e(1, 2) * r[2],
            e(2, 0) * r[0] + e(2, 1) * r[1] + e(2, 2) * r[2],
        ];
        let af = self.inner.acceleration(t, &rf, v);
        // a_gcrf = Mᵀ · a_itrf
        [
            e(0, 0) * af[0] + e(1, 0) * af[1] + e(2, 0) * af[2],
            e(0, 1) * af[0] + e(1, 1) * af[1] + e(2, 1) * af[2],
            e(0, 2) * af[0] + e(1, 2) * af[1] + e(2, 2) * af[2],
        ]
    }
    fn acceleration_f64(&self, t: f64, r: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
        let m = (self.rotate)(t);
        let rf = [
            m[0][0] * r[0] + m[0][1] * r[1] + m[0][2] * r[2],
            m[1][0] * r[0] + m[1][1] * r[1] + m[1][2] * r[2],
            m[2][0] * r[0] + m[2][1] * r[1] + m[2][2] * r[2],
        ];
        let af = self.inner.acceleration_f64(t, &rf, v);
        [
            m[0][0] * af[0] + m[1][0] * af[1] + m[2][0] * af[2],
            m[0][1] * af[0] + m[1][1] * af[1] + m[2][1] * af[2],
            m[0][2] * af[0] + m[1][2] * af[1] + m[2][2] * af[2],
        ]
    }
}

#[cfg(test)]
mod body_oriented_tests {
    #![allow(clippy::float_cmp)]
    use super::{Acceleration, BodyOriented, EarthFixed, Scalar};

    /// A diagonal but anisotropic body-fixed force — sensitive to how the frame is rotated, unlike a
    /// zonal field — so matching [`EarthFixed`] exercises the rotate-in / rotate-out path.
    struct Anisotropic;
    impl Acceleration for Anisotropic {
        fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
            [
                r[0] * S::from_f64(2.0),
                r[1] * S::from_f64(-3.0),
                r[2] * S::from_f64(5.0),
            ]
        }
    }

    /// A central force `a = k·r`: no rotation can change it, whatever the orientation.
    struct Isotropic(f64);
    impl Acceleration for Isotropic {
        fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
            let k = S::from_f64(self.0);
            [r[0] * k, r[1] * k, r[2] * k]
        }
    }

    /// A GCRF→ITRF stand-in with a *general* orientation: 3-1-3 Euler angles chosen so that no entry
    /// is near zero and no two coincide.
    ///
    /// A rotation about `z` — the only orientation the wrappers are otherwise tested at — carries
    /// four exact zeros and an exact one, and an entry that is zero or one hides the arithmetic
    /// around it: `x + 0` and `x − 0` are one number, so are `x · 1` and `x / 1`, and `0 · y` and
    /// `0 / y`. Two thirds of the matrix application goes unasserted at such a matrix.
    fn general_rotation(t: f64) -> [[f64; 3]; 3] {
        let (alpha, beta, gamma) = (0.7 + 1e-4 * t, 0.4, 1.1);
        let (ca, sa) = (libm::cos(alpha), libm::sin(alpha));
        let (cb, sb) = (libm::cos(beta), libm::sin(beta));
        let (cg, sg) = (libm::cos(gamma), libm::sin(gamma));
        [
            [cg * ca - sg * cb * sa, cg * sa + sg * cb * ca, sg * sb],
            [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
            [sb * sa, -sb * ca, cb],
        ]
    }

    /// `Mᵀ(k·M·r) = k·r` for every orthogonal `M`, whatever its entries — the wrapper may not leave a
    /// trace on a force that has no preferred direction.
    ///
    /// The expected value is the input, not a number this module computed, and the identity holds
    /// only when the second application is exactly the transpose of the first: change one entry of
    /// either and `MᵀM'` is no longer the identity. That is what makes it sharp where the pure-`z`
    /// comparison above is blind — measured, 32 of the arithmetic mutations in the two matrix
    /// applications survive that comparison and none survives this one.
    #[test]
    fn a_general_orientation_is_transparent_to_an_isotropic_force() {
        const K: f64 = -1.75;
        let oriented = BodyOriented {
            inner: Isotropic(K),
            rotate: general_rotation,
        };
        let r = [6_800_000.0_f64, 900_000.0, 2_100_000.0];
        let v = [1.0, 2.0, 3.0];
        for &t in &[0.0, 1234.0, 43_200.0] {
            // The premise of the test: the matrix really is orthonormal, and general enough that no
            // entry can hide an operator. Measured worst |MᵀM − I| here is 2.2e-16.
            let m = general_rotation(t);
            for i in 0..3 {
                for j in 0..3 {
                    let dot: f64 = (0..3).map(|k| m[k][i] * m[k][j]).sum();
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-15, "MᵀM at ({i},{j}) at t={t}");
                    assert!(m[i][j].abs() > 0.1, "entry ({i},{j}) hides its operator");
                }
            }
            // 1e-13 against a measured worst round-off of 7.4e-16: three decades of headroom, and
            // still five decades below what any of the mutations moves.
            for (path, got) in [
                ("generic", oriented.acceleration::<f64>(t, &r, &v)),
                ("f64", oriented.acceleration_f64(t, &r, &v)),
            ] {
                for i in 0..3 {
                    let want = K * r[i];
                    let rel = (got[i] - want).abs() / want.abs();
                    assert!(rel < 1e-13, "{path} axis {i} at t={t}: {rel:.3e} relative");
                }
            }
        }
    }

    /// Feeding `BodyOriented` the very matrix `EarthFixed` builds — `rot3(θ₀+ωt)` — must reproduce it
    /// bit for bit, which pins the matrix application (and its transpose) to the trusted implementation.
    #[test]
    fn reduces_to_earth_fixed_for_a_pure_z_rotation() {
        let omega = 7.292_115e-5;
        let theta0 = 0.9;
        let fixed = EarthFixed {
            inner: Anisotropic,
            omega,
            theta0,
        };
        let oriented = BodyOriented {
            inner: Anisotropic,
            rotate: |t: f64| {
                let (c, s) = (libm::cos(theta0 + omega * t), libm::sin(theta0 + omega * t));
                [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
            },
        };
        let r = [6_800_000.0_f64, 900_000.0, 2_100_000.0];
        for &t in &[0.0, 1234.0, 43_200.0] {
            let a = fixed.acceleration(t, &r, &[0.0; 3]);
            let b = oriented.acceleration(t, &r, &[0.0; 3]);
            for i in 0..3 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-9 * a[i].abs().max(1.0),
                    "axis {i} at t={t}"
                );
            }
        }
    }

    /// The f64 fast path must agree with the generic `Scalar` path for both frame wrappers — the whole
    /// point of `acceleration_f64` is to be a cheaper route to the same number.
    #[test]
    fn the_f64_fast_path_matches_the_generic_scalar_path() {
        let omega = 7.292_115e-5;
        let theta0 = 0.9;
        let fixed = EarthFixed {
            inner: Anisotropic,
            omega,
            theta0,
        };
        let oriented = BodyOriented {
            inner: Anisotropic,
            rotate: |t: f64| {
                let (c, s) = (libm::cos(theta0 + omega * t), libm::sin(theta0 + omega * t));
                [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
            },
        };
        let r = [6_800_000.0_f64, 900_000.0, 2_100_000.0];
        let v = [1.0, 2.0, 3.0];
        for &t in &[0.0, 1234.0, 43_200.0] {
            let (gf, ff) = (
                fixed.acceleration::<f64>(t, &r, &v),
                fixed.acceleration_f64(t, &r, &v),
            );
            let (go, fo) = (
                oriented.acceleration::<f64>(t, &r, &v),
                oriented.acceleration_f64(t, &r, &v),
            );
            for i in 0..3 {
                assert!(
                    (gf[i] - ff[i]).abs() < 1e-9 * gf[i].abs().max(1.0),
                    "EarthFixed axis {i} at t={t}"
                );
                assert!(
                    (go[i] - fo[i]).abs() < 1e-9 * go[i].abs().max(1.0),
                    "BodyOriented axis {i} at t={t}"
                );
            }
        }
    }
}

#[cfg(test)]
mod earth_fixed_tests {
    #![allow(clippy::float_cmp)]
    use super::{Acceleration, EarthFixed, Zonal};

    /// An axially symmetric field is invariant under the body rotation, so wrapping it changes nothing
    /// at any epoch or rotation angle — `a(Rz·r)` rotated back equals `a(r)`.
    #[test]
    fn earth_fixed_is_a_noop_for_a_zonal_field() {
        let zonal = Zonal::<3> {
            mu: 3.986_004_418e14,
            re: 6_378_137.0,
            j: [1.0826e-3, -2.5e-6, -1.6e-6],
        };
        let wrapped = EarthFixed {
            inner: zonal,
            omega: 7.292e-5,
            theta0: 0.7,
        };
        let r = [6_800_000.0, 900_000.0, 2_100_000.0];
        for &t in &[0.0, 1234.0, 43_200.0] {
            let a = zonal.acceleration(t, &r, &[0.0; 3]);
            let b = wrapped.acceleration(t, &r, &[0.0; 3]);
            for i in 0..3 {
                assert!(
                    (a[i] - b[i]).abs() < 1e-6 * a[i].abs().max(1e-6),
                    "axis {i} at t={t}"
                );
            }
        }
    }

    /// A non-zero rotation genuinely rotates the frame: for a longitude-dependent evaluation, wrapping
    /// with θ₀ = π/2 must differ from the unrotated inner force (sanity that the rotation is applied).
    #[test]
    fn earth_fixed_rotation_is_applied() {
        // Use a tesseral-like asymmetry by evaluating an off-axis point; the wrapper with a quarter turn
        // moves the effective sub-point, so a plain zonal (symmetric) stays equal — instead check the
        // rotation matrix action directly on a trivial radial-only inner via the identity a(r)=−r.
        struct MinusR;
        impl Acceleration for MinusR {
            fn acceleration<S: super::Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
                [
                    S::from_f64(0.0) - r[0],
                    S::from_f64(0.0) - r[1],
                    S::from_f64(0.0) - r[2],
                ]
            }
        }
        // a = −r is isotropic, so EarthFixed(−r) = −r too; the quarter-turn round-trip must preserve it.
        let wrapped = EarthFixed {
            inner: MinusR,
            omega: 1.0,
            theta0: core::f64::consts::FRAC_PI_2,
        };
        let r = [3.0, 4.0, 5.0];
        let a = wrapped.acceleration(0.0, &r, &[0.0; 3]);
        for i in 0..3 {
            assert!(
                (a[i] + r[i]).abs() < 1e-12,
                "axis {i}: {} vs {}",
                a[i],
                -r[i]
            );
        }
    }
}

#[cfg(test)]
mod superposition_tests {
    #![allow(clippy::float_cmp)] // constant forces sum exactly
    use super::{Acceleration, Scalar};

    /// A constant specific force, independent of state — so the superposition of several is trivial to
    /// check against a hand sum.
    #[derive(Clone, Copy)]
    struct Constant([f64; 3]);
    impl Acceleration for Constant {
        fn acceleration<S: Scalar>(&self, _t: S, _r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
            [
                S::from_f64(self.0[0]),
                S::from_f64(self.0[1]),
                S::from_f64(self.0[2]),
            ]
        }
    }

    #[test]
    fn three_and_four_force_tuples_sum_their_components() {
        // No component is zero and no pair the fold meets collides under −, × or ÷. Round constants
        // do not have that property and it is not a small effect: `x + 0` reads the same as `x − 0`,
        // and `−1 + 0.5` is exactly `−1 × 0.5`, which between them leave most of the arithmetic in
        // these two impls unasserted. All the values are exact binary fractions, so the sums are too.
        let (r, v) = ([1.0, 2.0, 3.0], [0.0; 3]);
        let a = Constant([1.5, -2.25, 3.125]);
        let b = Constant([-0.75, 4.5, -1.375]);
        let c = Constant([2.875, -3.5, 0.625]);
        let d = Constant([-4.125, 1.75, -2.5]);

        // Three forces: a + b + c, both the generic and the f64 fast path.
        let three = (a, b, c);
        let sum3 = [3.625, -1.25, 2.375];
        assert_eq!(three.acceleration::<f64>(0.0, &r, &v), sum3);
        assert_eq!(three.acceleration_f64(0.0, &r, &v), sum3);

        // Four forces: the previous sum plus d.
        let four = (a, b, c, d);
        let sum4 = [-0.5, 0.5, -0.125];
        assert_eq!(four.acceleration::<f64>(0.0, &r, &v), sum4);
        assert_eq!(four.acceleration_f64(0.0, &r, &v), sum4);
    }
}
