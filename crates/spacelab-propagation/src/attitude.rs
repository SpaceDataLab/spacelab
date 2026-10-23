//! Attitude on `SO(3)` by a Lie-group integrator (Runge–Kutta–Munthe-Kaas), for the free rigid body.
//!
//! # Why not just integrate the matrix
//!
//! A rigid body's orientation is a rotation `R ∈ SO(3)`, evolving by `Ṙ = R [ω]×`. Integrating the
//! nine entries of `R` (or the four of a quaternion) with an ordinary Runge–Kutta walks the state
//! *off* the manifold: `RᵀR` drifts from the identity and the quaternion from unit norm, and the
//! usual fix — renormalise every step — is a projection that quietly destroys the method's order and
//! its conservation properties. A **Lie-group integrator** advances the state *on* the group by
//! construction: it takes the step in the Lie algebra `so(3)` and maps it back with the exponential,
//! so `RᵀR = I` and `det R = 1` hold to machine precision for all time, no renormalisation.
//!
//! # The method
//!
//! RKMK4 wraps the classical order-4 Runge–Kutta around the exponential. Each step solves, in the
//! algebra, `σ̇ = dexp⁻¹_σ(A)` with `σ(0)=0` and `A` the body angular velocity, then sets
//! `R_{n+1} = Rₙ exp(σ(h))`. The two Lie-group pieces are:
//!
//! * the **exponential** `exp: so(3) → SO(3)`, Rodrigues' formula — the exact exponential of a skew
//!   matrix (`derivations/lie_so3.mac` checks it is orthogonal with unit determinant);
//! * **`dexp⁻¹`**, the inverse of the exponential's derivative, as its Bernoulli series
//!   `v − ½[σ,v] + 1/12 [σ,[σ,v]] − …`; for order 4 the two-bracket truncation suffices, and in
//!   `so(3)` every bracket `[a,b]` is the cross product `a × b` (also verified in the derivation).
//!
//! The free rigid body's angular velocity obeys Euler's equations `I ω̇ = (Iω) × ω`, integrated by
//! the classical RK4 alongside; the two invariants — the angular-momentum magnitude `|Iω|` and the
//! kinetic energy `½ ω·Iω` — are conserved to the method's order, and `R` stays orthogonal exactly.

#![allow(clippy::needless_range_loop, clippy::many_single_char_names)]

/// A 3-vector.
pub type Vec3 = [f64; 3];
/// A 3×3 matrix, row-major.
pub type Mat3 = [[f64; 3]; 3];

pub(crate) fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn matmul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    c
}

/// The exponential `exp([ω]×) ∈ SO(3)` by Rodrigues' formula, with a small-angle Taylor fallback so
/// the `sin θ/θ` and `(1−cos θ)/θ²` coefficients stay accurate as `θ → 0`.
#[must_use]
pub fn exp_so3(w: Vec3) -> Mat3 {
    let t2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    let theta = libm::sqrt(t2);
    let (a, b) = if theta < 1e-8 {
        // sinθ/θ = 1 − θ²/6 + …,  (1−cosθ)/θ² = 1/2 − θ²/24 + …
        (1.0 - t2 / 6.0, 0.5 - t2 / 24.0)
    } else {
        // `b` uses the half-angle form `2sin²(θ/2)/θ²`. Written literally as `(1−cosθ)/θ²` it
        // subtracts two numbers that agree to θ²/2, so it loses a digit per decade below θ ≈ 0.1
        // and is worthless by the threshold: 28% relative error at θ = 1.86e-8, the first f64
        // above it, against 2.9e-17 for the half-angle form at the same point. The error rides on
        // `b·K²`, so it stays under an ulp on the diagonal but reaches 2.9e-9 relative on the
        // off-diagonals. Raising the threshold instead was the alternative and is worse: the
        // two-term Taylor series is already 2.8e-15 off by θ = 1e-3 and would need more terms,
        // while this form is exact at every θ and leaves the branch guarding only θ = 0.
        // `a = sinθ/θ` needs no such care — `sin θ ≈ θ` is a smooth ratio, not a cancellation.
        let half = 0.5 * theta;
        let sinc_half = libm::sin(half) / half;
        (libm::sin(theta) / theta, 0.5 * sinc_half * sinc_half)
    };
    // R = I + a·K + b·K², K = [ω]×.
    let k = [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]];
    let k2 = matmul(&k, &k);
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (if i == j { 1.0 } else { 0.0 }) + a * k[i][j] + b * k2[i][j];
        }
    }
    r
}

/// `dexp⁻¹₋σ(v)` in `so(3)`, the coefficient RKMK needs here, truncated for order 4.
///
/// The kinematics are right-trivialized — `Ṙ = R·[ω]×` with ω in the body frame — so writing
/// `R(t) = Rₙ·exp(σ)` gives `Ṙ = R·dexp₋σ(σ̇)` and hence `σ̇ = dexp⁻¹₋σ(A)`, at **−σ**, not `+σ`.
/// Substituting into the Bernoulli series `dexp⁻¹ᵤ(v) = v − ½[u,v] + 1/12[u,[u,v]] − …` flips the
/// odd bracket and leaves the even one: `v + ½(σ×v) + 1/12 σ×(σ×v)`. Brackets are cross products.
///
/// Taking the ½ term at `+σ` costs three orders: the method then converges at 2, not 4.
fn dexpinv_neg(sigma: Vec3, v: Vec3) -> Vec3 {
    let sv = cross(sigma, v);
    let ssv = cross(sigma, sv);
    [
        v[0] + 0.5 * sv[0] + ssv[0] / 12.0,
        v[1] + 0.5 * sv[1] + ssv[1] / 12.0,
        v[2] + 0.5 * sv[2] + ssv[2] / 12.0,
    ]
}

/// Euler's equation for the body angular acceleration: `ω̇ = I⁻¹((Iω) × ω)`, diagonal inertia `I`.
fn euler(w: Vec3, inertia: Vec3) -> Vec3 {
    let iw = [inertia[0] * w[0], inertia[1] * w[1], inertia[2] * w[2]];
    let m = cross(iw, w);
    [m[0] / inertia[0], m[1] / inertia[1], m[2] / inertia[2]]
}

fn axpy(w: Vec3, a: f64, d: Vec3) -> Vec3 {
    [w[0] + a * d[0], w[1] + a * d[1], w[2] + a * d[2]]
}

/// One RKMK4 step of size `h`: advance the angular velocity by RK4 on Euler's equations and the
/// attitude by RKMK4 (dexp⁻¹-corrected RK4 composed with the exponential). Returns `(R_next, ω_next)`.
fn rkmk4_step(r: &Mat3, w: Vec3, inertia: Vec3, h: f64) -> (Mat3, Vec3) {
    // Classical RK4 for ω, recording the stage angular velocities (the algebra elements A_i).
    let d1 = euler(w, inertia);
    let ws2 = axpy(w, 0.5 * h, d1);
    let d2 = euler(ws2, inertia);
    let ws3 = axpy(w, 0.5 * h, d2);
    let d3 = euler(ws3, inertia);
    let ws4 = axpy(w, h, d3);
    let d4 = euler(ws4, inertia);
    let w_next = [
        w[0] + h / 6.0 * (d1[0] + 2.0 * d2[0] + 2.0 * d3[0] + d4[0]),
        w[1] + h / 6.0 * (d1[1] + 2.0 * d2[1] + 2.0 * d3[1] + d4[1]),
        w[2] + h / 6.0 * (d1[2] + 2.0 * d2[2] + 2.0 * d3[2] + d4[2]),
    ];

    // RKMK4 for R: k1 = A(Rₙ) = ω; the later stages apply dexp⁻¹ at the accumulated algebra point.
    let ka = w;
    let kb = dexpinv_neg([0.5 * h * ka[0], 0.5 * h * ka[1], 0.5 * h * ka[2]], ws2);
    let kc = dexpinv_neg([0.5 * h * kb[0], 0.5 * h * kb[1], 0.5 * h * kb[2]], ws3);
    let kd = dexpinv_neg([h * kc[0], h * kc[1], h * kc[2]], ws4);
    let sigma = [
        h / 6.0 * (ka[0] + 2.0 * kb[0] + 2.0 * kc[0] + kd[0]),
        h / 6.0 * (ka[1] + 2.0 * kb[1] + 2.0 * kc[1] + kd[1]),
        h / 6.0 * (ka[2] + 2.0 * kb[2] + 2.0 * kc[2] + kd[2]),
    ];
    (matmul(r, &exp_so3(sigma)), w_next)
}

/// Integrate the free rigid body from `(R0, ω0)` for `steps` fixed steps of `dt`, with principal
/// moments of inertia `inertia = (I₁, I₂, I₃)`. Returns the final attitude and body angular velocity.
#[must_use]
pub fn integrate_attitude(
    r0: &Mat3,
    w0: Vec3,
    inertia: Vec3,
    dt: f64,
    steps: usize,
) -> (Mat3, Vec3) {
    let mut r = *r0;
    let mut w = w0;
    for _ in 0..steps {
        let (rn, wn) = rkmk4_step(&r, w, inertia, dt);
        r = rn;
        w = wn;
    }
    (r, w)
}

// ---------------------------------------------------------------------------------------------------
// Mass properties
// ---------------------------------------------------------------------------------------------------

/// A rigid body's inertia tensor about its centre of mass, in the body frame, kg·m², carrying the
/// inverse the dynamics need.
///
/// The constructor is where the invariant lives: a matrix that is not symmetric positive definite is
/// not an inertia tensor, and `ω̇ = I⁻¹(…)` is not defined for one. Rejecting it once here is why
/// [`integrate_attitude_torqued`] never has to ask, and why the inverse is formed once instead of at
/// every stage of every step. What is deliberately *not* checked is the triangle inequality
/// `λ₁ + λ₂ ≥ λ₃` on the eigenvalues: every real mass distribution satisfies it, but violating it
/// leaves the body unphysical rather than the equations undefined, and testing it needs an
/// eigendecomposition this crate has no other use for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Inertia {
    tensor: Mat3,
    inverse: Mat3,
}

impl Inertia {
    /// Accept a tensor, or reject it. `None` unless `tensor` is symmetric — to a relative `1e-12` of
    /// its trace, the tolerance an assembled tensor's rounding needs — and positive definite by
    /// Sylvester's criterion on the three leading principal minors. A non-finite entry fails one of
    /// the minors and is rejected with the rest.
    #[must_use]
    pub fn from_tensor(tensor: Mat3) -> Option<Self> {
        let trace = tensor[0][0] + tensor[1][1] + tensor[2][2];
        if !trace.is_finite() || trace <= 0.0 {
            return None;
        }
        let tol = 1e-12 * trace;
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            if libm::fabs(tensor[i][j] - tensor[j][i]) > tol {
                return None;
            }
        }
        let mut sym = tensor;
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let mean = f64::midpoint(tensor[i][j], tensor[j][i]);
            sym[i][j] = mean;
            sym[j][i] = mean;
        }

        let minor1 = sym[0][0];
        let minor2 = sym[0][0] * sym[1][1] - sym[0][1] * sym[0][1];
        let det = sym[0][0] * (sym[1][1] * sym[2][2] - sym[1][2] * sym[1][2])
            - sym[0][1] * (sym[0][1] * sym[2][2] - sym[1][2] * sym[0][2])
            + sym[0][2] * (sym[0][1] * sym[1][2] - sym[1][1] * sym[0][2]);
        if !(minor1 > 0.0 && minor2 > 0.0 && det > 0.0) {
            return None;
        }

        // Column k of the inverse is the solution of I x = ê_k.
        let mut inverse = [[0.0; 3]; 3];
        for k in 0..3 {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let column = crate::linalg::lu_solve(sym, e)?;
            for i in 0..3 {
                inverse[i][k] = column[i];
            }
        }
        Some(Self {
            tensor: sym,
            inverse,
        })
    }

    /// A body whose body axes are its principal axes, with moments `(I₁, I₂, I₃)`.
    #[must_use]
    pub fn principal(moments: Vec3) -> Option<Self> {
        Self::from_tensor([
            [moments[0], 0.0, 0.0],
            [0.0, moments[1], 0.0],
            [0.0, 0.0, moments[2]],
        ])
    }

    /// The tensor itself.
    #[must_use]
    pub fn tensor(&self) -> &Mat3 {
        &self.tensor
    }

    /// Its inverse.
    #[must_use]
    pub fn inverse(&self) -> &Mat3 {
        &self.inverse
    }

    /// `I ω` — the angular momentum of a body spinning at `w`, body frame.
    #[must_use]
    pub fn apply(&self, w: Vec3) -> Vec3 {
        matvec(&self.tensor, w)
    }

    /// `I⁻¹ m` — the angular acceleration a body moment `m` alone would produce.
    #[must_use]
    pub fn solve(&self, m: Vec3) -> Vec3 {
        matvec(&self.inverse, m)
    }
}

fn matvec(m: &Mat3, v: Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// The tensor of a homogeneous solid cuboid about its centre, body axes along its edges:
/// `diag(y²+z², x²+z², x²+y²) · m/12`.
#[must_use]
pub fn solid_box_inertia(mass: f64, lengths: Vec3) -> Mat3 {
    let [x, y, z] = lengths;
    let c = mass / 12.0;
    [
        [c * (y * y + z * z), 0.0, 0.0],
        [0.0, c * (x * x + z * z), 0.0],
        [0.0, 0.0, c * (x * x + y * y)],
    ]
}

/// The tensor of a homogeneous flat rectangular panel about its centroid, with side `sides[0]` along
/// body **x̂**, `sides[1]` along **ŷ** and its normal along **ẑ**: the cuboid of zero thickness,
/// `diag(b², a², a²+b²) · m/12`. Put it where it belongs with [`rotate_inertia`] and
/// [`translate_inertia`] — this is deliberately the panel's own frame, so that the placement is the
/// caller's explicit rotation rather than a normal vector this function would have to complete into
/// a frame by an arbitrary convention.
#[must_use]
pub fn thin_plate_inertia(mass: f64, sides: [f64; 2]) -> Mat3 {
    solid_box_inertia(mass, [sides[0], sides[1], 0.0])
}

/// `R I Rᵀ` — the same body's tensor after the body is rotated by `R`, or equivalently its tensor
/// expressed in a frame `R` maps the part's own axes onto.
#[must_use]
pub fn rotate_inertia(r: &Mat3, i: &Mat3) -> Mat3 {
    let ri = matmul(r, i);
    let mut out = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            out[a][b] = ri[a][0] * r[b][0] + ri[a][1] * r[b][1] + ri[a][2] * r[b][2];
        }
    }
    out
}

/// The parallel-axis theorem: a part of mass `mass` whose centroid sits at `offset` from the
/// assembly's centre of mass contributes `I + m(|d|²E − d dᵀ)` about that centre.
#[must_use]
pub fn translate_inertia(i: &Mat3, mass: f64, offset: Vec3) -> Mat3 {
    let d2 = offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2];
    let mut out = *i;
    for a in 0..3 {
        for b in 0..3 {
            let kronecker = if a == b { d2 } else { 0.0 };
            out[a][b] += mass * (kronecker - offset[a] * offset[b]);
        }
    }
    out
}

/// Sum of part tensors — inertia about a *common* point is additive, which is what makes
/// [`translate_inertia`] the step that has to come first.
#[must_use]
pub fn sum_inertia(parts: &[Mat3]) -> Mat3 {
    let mut out = [[0.0; 3]; 3];
    for part in parts {
        for a in 0..3 {
            for b in 0..3 {
                out[a][b] += part[a][b];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------------
// The torqued rigid body
// ---------------------------------------------------------------------------------------------------

/// Euler's equation with an applied body moment: `ω̇ = I⁻¹(M + (Iω) × ω)`.
///
/// The sign follows the free-body [`euler`] above, which is the same statement at `M = 0`:
/// `I ω̇ + ω × (Iω) = M`.
fn euler_torqued(w: Vec3, inertia: &Inertia, moment: Vec3) -> Vec3 {
    let gyroscopic = cross(inertia.apply(w), w);
    inertia.solve([
        moment[0] + gyroscopic[0],
        moment[1] + gyroscopic[1],
        moment[2] + gyroscopic[2],
    ])
}

/// One RKMK4 step of a body under an applied moment.
///
/// The moment generally depends on where the body is pointing — a solar panel's arm turns with it —
/// so unlike the free-body [`rkmk4_step`] this one has to *form* the stage attitudes `Rₙ exp(σᵢ)`
/// rather than only the algebra elements. They are exactly the points `dexp⁻¹` is already evaluated
/// at, so the extra cost is three exponentials and three matrix products per step, and the order is
/// the same 4.
fn rkmk4_step_torqued(
    r: &Mat3,
    w: Vec3,
    t: f64,
    inertia: &Inertia,
    h: f64,
    moment: &mut impl FnMut(&Mat3, Vec3, f64) -> Vec3,
) -> (Mat3, Vec3) {
    let half = 0.5 * h;

    let d1 = euler_torqued(w, inertia, moment(r, w, t));
    let ws2 = axpy(w, half, d1);

    let ka = w;
    let sigma_b = [half * ka[0], half * ka[1], half * ka[2]];
    let rb = matmul(r, &exp_so3(sigma_b));
    let d2 = euler_torqued(ws2, inertia, moment(&rb, ws2, t + half));
    let ws3 = axpy(w, half, d2);
    let kb = dexpinv_neg(sigma_b, ws2);

    let sigma_c = [half * kb[0], half * kb[1], half * kb[2]];
    let rc = matmul(r, &exp_so3(sigma_c));
    let d3 = euler_torqued(ws3, inertia, moment(&rc, ws3, t + half));
    let ws4 = axpy(w, h, d3);
    let kc = dexpinv_neg(sigma_c, ws3);

    let sigma_d = [h * kc[0], h * kc[1], h * kc[2]];
    let rd = matmul(r, &exp_so3(sigma_d));
    let d4 = euler_torqued(ws4, inertia, moment(&rd, ws4, t + h));
    let kd = dexpinv_neg(sigma_d, ws4);

    let w_next = [
        w[0] + h / 6.0 * (d1[0] + 2.0 * d2[0] + 2.0 * d3[0] + d4[0]),
        w[1] + h / 6.0 * (d1[1] + 2.0 * d2[1] + 2.0 * d3[1] + d4[1]),
        w[2] + h / 6.0 * (d1[2] + 2.0 * d2[2] + 2.0 * d3[2] + d4[2]),
    ];
    let sigma = [
        h / 6.0 * (ka[0] + 2.0 * kb[0] + 2.0 * kc[0] + kd[0]),
        h / 6.0 * (ka[1] + 2.0 * kb[1] + 2.0 * kc[1] + kd[1]),
        h / 6.0 * (ka[2] + 2.0 * kb[2] + 2.0 * kc[2] + kd[2]),
    ];
    (matmul(r, &exp_so3(sigma)), w_next)
}

/// Integrate a rigid body under an applied moment from `(R0, ω0)` at `t0`, for `steps` fixed steps
/// of `dt`. Returns the final attitude and body angular velocity.
///
/// `moment(R, ω, t)` returns the total external moment about the centre of mass **in the body
/// frame**, N·m — the sum of whatever of [`crate::torque`]'s contributions apply. It is handed the
/// stage attitude so that a body-frame moment which depends on an inertial direction (the Sun, the
/// geomagnetic field, nadir) can rotate that direction itself; it is called four times per step.
///
/// At `moment ≡ 0` this reproduces [`integrate_attitude`] to rounding but not bit-for-bit: the free
/// body divides by each principal moment where this one multiplies by a full inverse.
#[must_use]
pub fn integrate_attitude_torqued(
    r0: &Mat3,
    w0: Vec3,
    t0: f64,
    inertia: &Inertia,
    dt: f64,
    steps: usize,
    mut moment: impl FnMut(&Mat3, Vec3, f64) -> Vec3,
) -> (Mat3, Vec3) {
    let mut r = *r0;
    let mut w = w0;
    for step in 0..steps {
        #[allow(clippy::cast_precision_loss)]
        let t = dt.mul_add(step as f64, t0);
        let (rn, wn) = rkmk4_step_torqued(&r, w, t, inertia, dt, &mut moment);
        r = rn;
        w = wn;
    }
    (r, w)
}
