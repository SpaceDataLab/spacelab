//! Periodic orbits in the CR3BP by differential correction.
//!
//! Libration orbits (planar Lyapunov, and the halo family that bifurcates from it) are found by
//! single-shooting: start perpendicular to the x-z plane, integrate the state together with its
//! state-transition matrix to the next plane crossing, and Newton-correct the initial condition so
//! the crossing is perpendicular again — a periodic orbit. The STM comes from integrating the
//! variational equations `Φ̇ = A(t)·Φ` alongside the state, `A` being the CR3BP Jacobian. See Koon,
//! Lo, Marsden & Ross, *Dynamical Systems, the Three-Body Problem and Space Mission Design*, §4;
//! Howell (1984). Verified here by orbit closure — a corrected orbit, propagated a full period,
//! returns to its start.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::manual_midpoint,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::cr3bp::Cr3bp;

/// A synodic state `[x, y, z, ẋ, ẏ, ż]`.
type State = [f64; 6];
/// A 6×6 state-transition matrix, row-major.
type Stm = [[f64; 6]; 6];

/// The CR3BP variational Jacobian `A = ∂ṡ/∂s`: identity coupling of position to velocity, the
/// pseudo-potential Hessian `U`, and the Coriolis block.
fn jacobian(mu: f64, s: State) -> Stm {
    let (x, y, z) = (s[0], s[1], s[2]);
    let (m1, m2) = (1.0 - mu, mu);
    let (dx1, dx2) = (x + mu, x - 1.0 + mu);
    let r1 = libm::sqrt(dx1 * dx1 + y * y + z * z);
    let r2 = libm::sqrt(dx2 * dx2 + y * y + z * z);
    let r13 = r1 * r1 * r1;
    let r15 = r13 * r1 * r1;
    let r23 = r2 * r2 * r2;
    let r25 = r23 * r2 * r2;

    let uxx = 1.0 - m1 / r13 - m2 / r23 + 3.0 * m1 * dx1 * dx1 / r15 + 3.0 * m2 * dx2 * dx2 / r25;
    let uyy = 1.0 - m1 / r13 - m2 / r23 + 3.0 * m1 * y * y / r15 + 3.0 * m2 * y * y / r25;
    let uzz = -m1 / r13 - m2 / r23 + 3.0 * m1 * z * z / r15 + 3.0 * m2 * z * z / r25;
    let uxy = 3.0 * m1 * dx1 * y / r15 + 3.0 * m2 * dx2 * y / r25;
    let uxz = 3.0 * m1 * dx1 * z / r15 + 3.0 * m2 * dx2 * z / r25;
    let uyz = 3.0 * m1 * y * z / r15 + 3.0 * m2 * y * z / r25;

    let mut a = [[0.0; 6]; 6];
    a[0][3] = 1.0;
    a[1][4] = 1.0;
    a[2][5] = 1.0;
    a[3][0] = uxx;
    a[3][1] = uxy;
    a[3][2] = uxz;
    a[4][0] = uxy;
    a[4][1] = uyy;
    a[4][2] = uyz;
    a[5][0] = uxz;
    a[5][1] = uyz;
    a[5][2] = uzz;
    a[3][4] = 2.0;
    a[4][3] = -2.0;
    a
}

/// State derivative `ṡ = [ẋ, ẏ, ż, ẍ, ÿ, z̈]`.
fn state_dot(sys: Cr3bp, s: State) -> State {
    let p = spacelab_core::Vec3::new(s[0], s[1], s[2]);
    let v = spacelab_core::Vec3::new(s[3], s[4], s[5]);
    let a = sys.acceleration(p, v);
    [s[3], s[4], s[5], a.x, a.y, a.z]
}

fn matmul(a: &Stm, b: &Stm) -> Stm {
    let mut c = [[0.0; 6]; 6];
    for i in 0..6 {
        for k in 0..6 {
            let aik = a[i][k];
            for j in 0..6 {
                c[i][j] += aik * b[k][j];
            }
        }
    }
    c
}

fn axpy(s: State, k: State, h: f64) -> State {
    core::array::from_fn(|i| s[i] + h * k[i])
}
fn axpy_m(m: &Stm, k: &Stm, h: f64) -> Stm {
    core::array::from_fn(|i| core::array::from_fn(|j| m[i][j] + h * k[i][j]))
}

/// One RK4 step of the coupled `(state, STM)` system.
fn rk4(sys: Cr3bp, s: State, phi: &Stm, h: f64) -> (State, Stm) {
    let d = |s: State, phi: &Stm| (state_dot(sys, s), matmul(&jacobian(sys.mu, s), phi));
    let (k1s, k1p) = d(s, phi);
    let (k2s, k2p) = d(axpy(s, k1s, h / 2.0), &axpy_m(phi, &k1p, h / 2.0));
    let (k3s, k3p) = d(axpy(s, k2s, h / 2.0), &axpy_m(phi, &k2p, h / 2.0));
    let (k4s, k4p) = d(axpy(s, k3s, h), &axpy_m(phi, &k3p, h));
    let sn =
        core::array::from_fn(|i| s[i] + h / 6.0 * (k1s[i] + 2.0 * k2s[i] + 2.0 * k3s[i] + k4s[i]));
    let pn = core::array::from_fn(|i| {
        core::array::from_fn(|j| {
            phi[i][j] + h / 6.0 * (k1p[i][j] + 2.0 * k2p[i][j] + 2.0 * k3p[i][j] + k4p[i][j])
        })
    });
    (sn, pn)
}

/// Integrate `(state, STM)` from a perpendicular x-z-plane start to the next `y = 0` crossing,
/// returning `(half_period, crossing_state, crossing_STM)`. The crossing time is refined by Newton
/// on `y`. Returns `None` if no crossing is found within a bounded time.
fn to_crossing(sys: Cr3bp, s0: State) -> Option<(f64, State, Stm)> {
    let h = 2.5e-4;
    let mut s = s0;
    let mut phi = identity();
    let mut t = 0.0;
    for _ in 0..200_000 {
        let (sn, pn) = rk4(sys, s, &phi, h);
        // Detect `y` returning through zero in either direction (the linear seed may leave the plane
        // toward −y). `s` is the pre-crossing state; the initial `y = 0` cannot false-trigger.
        if t > 2.0 * h && s[1] * sn[1] < 0.0 {
            let (mut cs, mut cp, mut ct) = (s, phi, t);
            for _ in 0..40 {
                let dt = -cs[1] / cs[4]; // Newton on y: y / ẏ
                if !dt.is_finite() {
                    return None;
                }
                let (ns, np) = rk4(sys, cs, &cp, dt);
                cs = ns;
                cp = np;
                ct += dt;
                if cs[1].abs() < 1e-13 {
                    break;
                }
            }
            return Some((ct, cs, cp));
        }
        s = sn;
        phi = pn;
        t += h;
    }
    None
}

fn identity() -> Stm {
    let mut m = [[0.0; 6]; 6];
    for i in 0..6 {
        m[i][i] = 1.0;
    }
    m
}

/// A corrected periodic orbit: its initial synodic state and full period.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicOrbit {
    /// Initial state on the x-z plane, crossing perpendicularly.
    pub state: State,
    /// Full period (twice the plane-to-plane time).
    pub period: f64,
}

/// A planar Lyapunov orbit about a collinear libration point, by single-shooting.
///
/// `li` selects the point (0 → L1, 1 → L2). `x_amplitude` is the displacement of the orbit's x-axis
/// crossing from the libration point. The in-plane velocity `ẏ₀` is seeded from the linear
/// (Lindstedt) frequency, then corrected so `ẋ = 0` at the half-period crossing.
///
/// The seed is first-order, so the amplitude has a working range and it is not large: measured at
/// Earth–Moon L1 and L2, the correction converges in three to eight steps up to `x_amplitude ≈
/// 0.016`, and past that the seed leaves Newton's basin — `0.018` at L2 returns `None`, `0.022` at
/// L1 needs 36 steps, and the periods it does return stop tracking the family.
///
/// Returns `None` if `li` is not `0` or `1`, or if the shooting does not converge.
#[must_use]
pub fn lyapunov_orbit(sys: Cr3bp, li: usize, x_amplitude: f64) -> Option<PeriodicOrbit> {
    lyapunov_shoot(sys, li, x_amplitude).map(|(orbit, _)| orbit)
}

/// [`lyapunov_orbit`] with the number of Newton corrections it took.
///
/// The count is not diagnostics. Newton here is self-correcting: it targets `ẋ = 0` at the crossing
/// and reaches the same orbit from any seed inside the basin, so the linear seed below --- `γ`, `c₂`,
/// the frequency `ω`, the Lindstedt `ẏ₀` --- does not appear in the answer at all. Twenty mutations
/// of those five lines survived every closure and periodicity assertion in this file for that reason.
/// What a corrupted seed changes is how far Newton has to walk, and the count is the only place it
/// shows: `the_linear_seed_lands_inside_newtons_quadratic_basin` is what asserts it.
pub(crate) fn lyapunov_shoot(
    sys: Cr3bp,
    li: usize,
    x_amplitude: f64,
) -> Option<(PeriodicOrbit, u32)> {
    let lx = sys.collinear_points().get(li)?.x;
    // c2 = Uxx-related coefficient at the collinear point; ω is the in-plane linear frequency.
    let mu = sys.mu;
    let gamma1 = (lx + mu).abs();
    let gamma2 = (lx - 1.0 + mu).abs();
    let c2 = (1.0 - mu) / (gamma1 * gamma1 * gamma1) + mu / (gamma2 * gamma2 * gamma2);
    let omega = libm::sqrt((2.0 - c2 + libm::sqrt(9.0 * c2 * c2 - 8.0 * c2)) / 2.0);
    let uxx = 1.0 + 2.0 * c2;
    // Linear seed: x(t) = lx + Ax cos ωt, ẏ₀ = −(ω² + Uxx)/(2ω) · Ax · ω.
    let mut vy0 = -(omega * omega + uxx) / (2.0 * omega) * x_amplitude * omega;
    let x0 = lx + x_amplitude;

    for steps in 0..50 {
        let s0 = [x0, 0.0, 0.0, 0.0, vy0, 0.0];
        let (_, cs, phi) = to_crossing(sys, s0)?;
        let vx1 = cs[3];
        if vx1.abs() < 1e-12 {
            let (thalf, _, _) = to_crossing(sys, s0)?;
            return Some((
                PeriodicOrbit {
                    state: s0,
                    period: 2.0 * thalf,
                },
                steps,
            ));
        }
        // dẋ₁/dẏ₀ with the free final time (y₁ = 0): Φ[3][4] − (ẍ₁/ẏ₁)·Φ[1][4].
        let acc = state_dot(sys, cs);
        let denom = phi[3][4] - (acc[3] / cs[4]) * phi[1][4];
        vy0 -= vx1 / denom;
    }
    None
}

/// Correct a perpendicular-crossing guess `[x0, 0, z0, 0, ẏ0, 0]` to a three-dimensional halo orbit:
/// hold `z0` (the out-of-plane amplitude) and vary `[x0, ẏ0]` by a 2×2 Newton step so that *both*
/// `ẋ` and `ż` vanish at the half-period plane crossing — the perpendicular crossing that, with the
/// x-z-plane symmetry, closes a periodic 3D orbit. The correction uses the STM with a free final time.
///
/// Returns `None` if the crossing cannot be found or the Newton iteration does not converge; a poor
/// guess (halos exist only above a minimum amplitude, and bifurcate from the planar Lyapunov family)
/// is the usual cause.
#[must_use]
pub fn halo_orbit(sys: Cr3bp, guess: State) -> Option<PeriodicOrbit> {
    let mut x0 = guess[0];
    let z0 = guess[2];
    let mut vy0 = guess[4];
    for _ in 0..80 {
        let s0 = [x0, 0.0, z0, 0.0, vy0, 0.0];
        let (thalf, cs, phi) = to_crossing(sys, s0)?;
        let (vx1, vz1) = (cs[3], cs[5]);
        if vx1.abs() < 1e-11 && vz1.abs() < 1e-11 {
            return Some(PeriodicOrbit {
                state: s0,
                period: 2.0 * thalf,
            });
        }
        // Target [ẋ₁, ż₁] = 0 by varying [x0, ẏ0], with the final time free (y₁ = 0):
        // M = [[Φ40, Φ45],[Φ60, Φ65]] − (1/ẏ₁)·[[ẍ₁],[z̈₁]]·[Φ20, Φ25].
        let acc = state_dot(sys, cs);
        let (ax1, az1, vy1) = (acc[3], acc[5], cs[4]);
        let m00 = phi[3][0] - ax1 / vy1 * phi[1][0];
        let m01 = phi[3][4] - ax1 / vy1 * phi[1][4];
        let m10 = phi[5][0] - az1 / vy1 * phi[1][0];
        let m11 = phi[5][4] - az1 / vy1 * phi[1][4];
        let det = m00 * m11 - m01 * m10;
        if det.abs() < 1e-14 {
            return None;
        }
        x0 -= (m11 * vx1 - m01 * vz1) / det;
        vy0 -= (m00 * vz1 - m10 * vx1) / det;
    }
    None
}

/// Generate a halo orbit directly from its out-of-plane amplitude: build Richardson's third-order
/// analytic seed for collinear point `li` (0 → L1, 1 → L2), class `northern` (`true` northern), and
/// converge it with [`halo_orbit`]. `az` is Richardson's γ-normalized amplitude (synodic
/// `z0 ≈ γ·az`). This is the ergonomic entry point — no hand-tuned guess — and the robust way to start
/// a [`halo_family`] sweep. Returns `None` if the point index or amplitude is invalid, or the
/// corrector fails to converge.
#[must_use]
pub fn richardson_halo(sys: Cr3bp, li: usize, az: f64, northern: bool) -> Option<PeriodicOrbit> {
    let guess = crate::richardson::richardson_halo_guess(sys, li, az, northern)?;
    halo_orbit(sys, guess)
}

/// Trace a halo family by natural-parameter continuation in the out-of-plane amplitude `z0`, calling
/// `emit` for each converged member. Starting from `seed` (a converged halo, e.g. from
/// [`halo_orbit`]), each step increases `z0` by `z_step` and re-corrects using the previous member as
/// the predictor — a robust family generator that needs no analytic (Richardson) initial guess. Stops
/// early if a step fails to converge (the family end, or a step too large).
pub fn halo_family<F: FnMut(PeriodicOrbit)>(
    sys: Cr3bp,
    seed: PeriodicOrbit,
    z_step: f64,
    count: usize,
    mut emit: F,
) {
    let mut prev = seed;
    let mut prev2: Option<PeriodicOrbit> = None;
    emit(prev);
    for _ in 1..count {
        let z_next = prev.state[2] + z_step;
        // Secant predictor: extrapolate [x0, ẏ0] linearly from the last two members (uniform z step);
        // reuse the previous member for the first step, where there is no second-to-last.
        let (gx, gvy) = prev2.map_or((prev.state[0], prev.state[4]), |p2| {
            (
                2.0 * prev.state[0] - p2.state[0],
                2.0 * prev.state[4] - p2.state[4],
            )
        });
        match halo_orbit(sys, [gx, 0.0, z_next, 0.0, gvy, 0.0]) {
            Some(o) => {
                emit(o);
                prev2 = Some(prev);
                prev = o;
            }
            None => break,
        }
    }
}

/// The monodromy matrix: the state-transition matrix integrated over one full period of `orbit`.
/// Its eigenvalues (which come in reciprocal pairs for this Hamiltonian system) classify the orbit's
/// stability; the eigenvectors of the off-unit-circle pair span the stable and unstable manifolds.
/// `forward = false` integrates in reverse time, giving `M⁻¹` — whose dominant eigenvector is the
/// *stable* direction of `M`.
#[must_use]
pub fn monodromy(sys: Cr3bp, orbit: &PeriodicOrbit, forward: bool) -> Stm {
    let h = if forward { 2.5e-4 } else { -2.5e-4 };
    let steps = libm::round(orbit.period / 2.5e-4) as i64;
    let mut s = orbit.state;
    let mut phi = identity();
    for _ in 0..steps {
        let (ns, np) = rk4(sys, s, &phi, h);
        s = ns;
        phi = np;
    }
    phi
}

fn matvec(m: &Stm, v: &[f64; 6]) -> [f64; 6] {
    core::array::from_fn(|i| (0..6).map(|j| m[i][j] * v[j]).sum())
}
fn norm6(v: &[f64; 6]) -> f64 {
    libm::sqrt(v.iter().map(|x| x * x).sum())
}

/// Dominant (largest-magnitude) eigenpair `(λ, v̂)` of `m` by power iteration; `v̂` is unit length and
/// `λ` is the signed Rayleigh quotient.
fn dominant_eigen(m: &Stm) -> (f64, [f64; 6]) {
    let mut v = [1.0, 0.3, -0.2, 0.1, -0.4, 0.2];
    let inv0 = 1.0 / norm6(&v);
    v = core::array::from_fn(|i| v[i] * inv0);
    for _ in 0..3000 {
        let mv = matvec(m, &v);
        let inv = 1.0 / norm6(&mv);
        v = core::array::from_fn(|i| mv[i] * inv);
    }
    let mv = matvec(m, &v);
    let lambda = (0..6).map(|i| v[i] * mv[i]).sum(); // Rayleigh quotient, v is unit
    (lambda, v)
}

/// A manifold trajectory's seed: a state on `orbit` displaced by `epsilon` along the stable or
/// unstable eigenvector, and the eigenvalue (`|λ| > 1` unstable, `< 1` stable) that governs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldSeed {
    /// The perturbed state; propagate it forward (unstable) or backward (stable) to trace the manifold.
    pub state: State,
    /// The eigenvalue of the monodromy along this direction.
    pub eigenvalue: f64,
}

/// Seed the unstable manifold of `orbit`: the initial state displaced by `epsilon` along the
/// unstable eigenvector. Propagated forward, it departs the orbit exponentially — the ballistic
/// escape route a low-energy transfer rides.
#[must_use]
pub fn unstable_manifold_seed(sys: Cr3bp, orbit: &PeriodicOrbit, epsilon: f64) -> ManifoldSeed {
    let (lambda, v) = dominant_eigen(&monodromy(sys, orbit, true));
    ManifoldSeed {
        state: core::array::from_fn(|i| orbit.state[i] + epsilon * v[i]),
        eigenvalue: lambda,
    }
}

/// Seed the stable manifold of `orbit` (the dominant direction of `M⁻¹`, i.e. the reverse-time
/// monodromy). Propagated *backward*, it approaches the orbit — the capture route into a libration orbit.
#[must_use]
pub fn stable_manifold_seed(sys: Cr3bp, orbit: &PeriodicOrbit, epsilon: f64) -> ManifoldSeed {
    // The reverse-time monodromy's dominant eigenvalue is λ_u; the stable multiplier under M is its
    // reciprocal, 1/λ_u (< 1).
    let (lambda, v) = dominant_eigen(&monodromy(sys, orbit, false));
    ManifoldSeed {
        state: core::array::from_fn(|i| orbit.state[i] + epsilon * v[i]),
        eigenvalue: 1.0 / lambda,
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::expect_used,
        clippy::unwrap_used,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    use super::*;

    /// The half-period state of `orbit`, integrated at a step that divides the half period exactly so
    /// that the arrival is not itself an interpolation.
    fn half_period_state(sys: Cr3bp, orbit: &PeriodicOrbit) -> State {
        let steps = libm::round(orbit.period / 2.0 / 2.5e-4);
        flow(sys, orbit.state, orbit.period / 2.0 / steps, steps as i64)
    }

    /// A corrected orbit crosses the x-z plane perpendicularly at half its period.
    ///
    /// This is the condition the correction solves for, and asserting the *returned* orbit against it
    /// is what makes the corrector's own exit tests observable. Two of them were not: `to_crossing`
    /// refines the crossing by Newton on `y` until `|y| < 1e-13`, and inverting that comparison
    /// leaves one Newton step and `|y| ≈ 1e-9` — a half period off by the same order, which closure
    /// at 1e-5 cannot see. `halo_orbit` accepts when `|ẋ| < 1e-11` *and* `|ż| < 1e-11`, and relaxing
    /// the `&&` to `||` accepts an orbit one iteration short in whichever component is still large.
    fn assert_perpendicular_crossing(sys: Cr3bp, orbit: &PeriodicOrbit, what: &str) {
        let cs = half_period_state(sys, orbit);
        assert!(
            cs[1].abs() < 1e-12,
            "{what}: the half-period state is off the x-z plane, y = {:.3e}",
            cs[1]
        );
        assert!(
            cs[3].abs() < 1e-11,
            "{what}: the half-period crossing is not perpendicular, x-dot = {:.3e}",
            cs[3]
        );
        assert!(
            cs[5].abs() < 1e-11,
            "{what}: the half-period crossing is not perpendicular, z-dot = {:.3e}",
            cs[5]
        );
    }

    /// Every halo the corrector returns crosses perpendicularly, not just the one the closure test
    /// happens to pick.
    ///
    /// `halo_orbit` accepts when `|ẋ₁| < 1e-11` *and* `|ż₁| < 1e-11`. Relaxing that `&&` to `||`
    /// accepts one Newton step earlier, whenever either component gets there first, and the orbit it
    /// returns is correspondingly less converged. On the guess the closure test uses, the two
    /// components reach the tolerance on the same step and the mutation is invisible; over 336
    /// guesses spanning `x₀ ∈ [0.80, 0.87]`, `z₀ ∈ [0.02, 0.14]` and `ẏ₀ ∈ [0.10, 0.225]`, 175
    /// converged and 18 of those came back different — by up to 5.7e-10 in `x₀`.
    ///
    /// The four below are from that set. Their half-period `ẋ` is 5e-15 to 5e-14 as corrected, and
    /// 1.1e-11 to 1.4e-10 one step short, so the `1e-11` this asserts sits between the two with two
    /// orders of margin on the converged side.
    #[test]
    fn every_corrected_halo_crosses_perpendicularly() {
        let sys = Cr3bp::EARTH_MOON;
        for &(x0, z0, vy0) in &[
            (0.82, 0.04, 0.10),
            (0.82, 0.02, 0.175),
            (0.83, 0.02, 0.20),
            (0.84, 0.06, 0.175),
        ] {
            let orbit = halo_orbit(sys, [x0, 0.0, z0, 0.0, vy0, 0.0])
                .unwrap_or_else(|| panic!("halo from ({x0}, {z0}, {vy0}) did not converge"));
            assert_perpendicular_crossing(sys, &orbit, &format!("halo from ({x0}, {z0}, {vy0})"));
        }
    }

    /// A planar Lyapunov orbit about Earth–Moon L1, corrected and then propagated a full period,
    /// closes on itself — the definition of periodicity, and an end-to-end check of the STM,
    /// crossing detection and Newton correction.
    #[test]
    fn lyapunov_orbit_about_l1_closes() {
        let sys = Cr3bp::EARTH_MOON;
        let orbit = lyapunov_orbit(sys, 0, 0.01).expect("converges");
        // Propagate the corrected state one full period with the same RK4 and check closure.
        let mut s = orbit.state;
        let mut phi = identity();
        let h = 2.5e-4;
        let steps = (orbit.period / h).round() as i64;
        for _ in 0..steps {
            let (ns, np) = rk4(sys, s, &phi, h);
            s = ns;
            phi = np;
        }
        let dr = ((s[0] - orbit.state[0]).powi(2) + (s[1] - orbit.state[1]).powi(2)).sqrt();
        let dv = ((s[3] - orbit.state[3]).powi(2) + (s[4] - orbit.state[4]).powi(2)).sqrt();
        assert!(dr < 1e-5, "orbit did not close: position gap {dr:.3e}");
        assert_perpendicular_crossing(sys, &orbit, "L1 Lyapunov");
        assert!(dv < 1e-5, "orbit did not close: velocity gap {dv:.3e}");
        // Planar: z and ż stay zero.
        assert!(
            s[2].abs() < 1e-12 && s[5].abs() < 1e-12,
            "orbit left the plane"
        );
        // A sensible period for an Earth-Moon L1 Lyapunov: a few nondimensional time units.
        assert!(
            orbit.period > 1.0 && orbit.period < 6.0,
            "implausible period {}",
            orbit.period
        );
    }

    /// The linear seed lands inside Newton's quadratic basin, and the family it traces is monotone.
    ///
    /// Nothing else in this file can see the seed. The correction targets `ẋ = 0` at the crossing and
    /// reaches the same orbit from anywhere in the basin, so `γ`, `c₂`, `ω` and the Lindstedt `ẏ₀`
    /// cancel out of every closure and periodicity assertion — twenty mutations of those five lines
    /// survived all of them. What a corrupted seed changes is the distance Newton has to walk, and
    /// the count is where that shows.
    ///
    /// The budget is measured and two-sided. Over both collinear points and eight amplitudes the
    /// worst call takes 8 corrections and the mean is 5.0, against the 50 the loop allows; a
    /// one-sided budget would pass on a counter that stopped counting. Three is the floor: the seed
    /// is a first-order approximation, so it is never already the answer.
    ///
    /// The amplitude band is measured too. Beyond about `0.016` at Earth–Moon L1 and L2 the linear
    /// seed leaves the basin: `0.018` at L2 fails to converge outright, `0.022` at L1 needs 36
    /// corrections, and where it does converge the period jumps off the family's smooth trend. That
    /// is the practical meaning of the "small ⇒ near-linear" in the docstring, so the sweep stops
    /// where the method does, and the monotone-period assertion is what would notice if it moved.
    #[test]
    fn the_linear_seed_lands_inside_newtons_quadratic_basin() {
        let sys = Cr3bp::EARTH_MOON;
        let (mut worst, mut least, mut total, mut calls) = (0, u32::MAX, 0u32, 0u32);
        for li in 0..2 {
            let mut previous = 0.0;
            for k in 1..=8 {
                let a = 0.002 * f64::from(k);
                let (orbit, iters) = lyapunov_shoot(sys, li, a)
                    .unwrap_or_else(|| panic!("L{} a={a} diverged", li + 1));
                assert!(
                    orbit.period > previous,
                    "L{} a={a}: period {} did not grow past {previous}",
                    li + 1,
                    orbit.period
                );
                previous = orbit.period;
                worst = worst.max(iters);
                least = least.min(iters);
                total += iters;
                calls += 1;
            }
        }
        let mean = f64::from(total) / f64::from(calls);
        println!("LYAPUNOV least {least} worst {worst} mean {mean:.2}");
        assert!(least >= 3, "a call converged in {least} corrections");
        assert!(worst <= 8, "worst call took {worst} corrections");
        assert!(mean < 5.5, "mean {mean:.2} corrections per call");
    }

    /// The analytic Jacobian matches a finite-difference of the equations of motion.
    #[test]
    fn jacobian_matches_finite_difference() {
        let sys = Cr3bp::EARTH_MOON;
        let s = [0.85, 0.05, 0.03, 0.01, 0.12, -0.02];
        let a = jacobian(sys.mu, s);
        let eps = 1e-6;
        for j in 0..6 {
            let mut sp = s;
            let mut sm = s;
            sp[j] += eps;
            sm[j] -= eps;
            let (fp, fm) = (state_dot(sys, sp), state_dot(sys, sm));
            for i in 0..6 {
                let fd = (fp[i] - fm[i]) / (2.0 * eps);
                assert!(
                    (a[i][j] - fd).abs() < 1e-5,
                    "A[{i}][{j}] {} vs FD {fd}",
                    a[i][j]
                );
            }
        }
    }

    /// Propagate `s0` for `steps` of `h` under the Earth–Moon dynamics, discarding the STM.
    fn flow(sys: Cr3bp, s0: State, h: f64, steps: i64) -> State {
        let mut s = s0;
        let mut phi = identity();
        for _ in 0..steps {
            let (ns, np) = rk4(sys, s, &phi, h);
            s = ns;
            phi = np;
        }
        s
    }

    /// `m · v`, written here rather than taken from `matvec`, which is the function under test.
    fn apply(m: &Stm, v: &[f64; 6]) -> [f64; 6] {
        let mut out = [0.0; 6];
        for (i, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for j in 0..6 {
                acc += m[i][j] * v[j];
            }
            *o = acc;
        }
        out
    }

    fn separation(a: State, b: State) -> f64 {
        (0..6).map(|i| (a[i] - b[i]).powi(2)).sum::<f64>().sqrt()
    }

    /// The unstable multiplier is the factor by which the manifold actually departs, over one period.
    ///
    /// This used to assert `|λ| > 1` and an eigenvector residual computed with `matvec` — the very
    /// function it was checking. Both are satisfied by the identity matrix: the Rayleigh quotient of
    /// a normalised vector rounds to `1.0000000000000002`, which clears `> 1.0`, and `I·v − λv` is
    /// then zero to the ulp. So was the growth test, because a generic displacement of a hyperbolic
    /// orbit departs whatever the reported eigenpair says. Replacing `matvec` with a constant, or
    /// collapsing the monodromy's step count to zero, passed all three.
    ///
    /// The multiplier has a meaning outside the eigensolver: it is the growth of a small displacement
    /// along the eigenvector under the *nonlinear* flow, over one period. Measured at Earth–Moon L1
    /// with `Ax = 0.01` and `ε = 1e-8`, `‖φ_T(x₀ + εv) − φ_T(x₀)‖/ε` is 2561.363 against a reported
    /// λ of 2561.276 — 3.4e-5 relative, the nonlinear remainder. Nothing that gets the eigenpair
    /// wrong can match that, and the flow used to check it shares no code with `dominant_eigen`.
    #[test]
    fn the_unstable_multiplier_is_the_manifolds_actual_growth() {
        let sys = Cr3bp::EARTH_MOON;
        let h = 2.5e-4;
        for li in 0..2 {
            let orbit = lyapunov_orbit(sys, li, 0.01).expect("converges");
            let m = monodromy(sys, &orbit, true);
            let (lambda, v) = dominant_eigen(&m);
            let steps = libm::round(orbit.period / h) as i64;

            // A saddle, not a rounding error: the Earth–Moon collinear Lyapunov multipliers are in
            // the thousands, so `> 1` is the wrong bar by three orders of magnitude.
            assert!(
                lambda.abs() > 1e3,
                "L{}: |unstable multiplier| {lambda:e} is not hyperbolic",
                li + 1
            );
            assert!(
                (norm6(&v) - 1.0).abs() < 1e-12,
                "L{}: eigenvector is not a unit vector",
                li + 1
            );
            let mv = apply(&m, &v);
            let resid = norm6(&core::array::from_fn(|i| mv[i] - lambda * v[i])) / norm6(&mv);
            assert!(
                resid < 1e-12,
                "L{}: eigenvector residual {resid:.3e}",
                li + 1
            );

            let eps = 1e-8;
            let base = flow(sys, orbit.state, h, steps);
            let seed = unstable_manifold_seed(sys, &orbit, eps);
            assert!(
                (separation(seed.state, orbit.state) - eps).abs() < 1e-16,
                "L{}: the seed is not `epsilon` from the orbit",
                li + 1
            );
            let grew = separation(flow(sys, seed.state, h, steps), base) / eps;
            let miss = (grew - lambda.abs()).abs() / lambda.abs();
            println!(
                "L{} unstable: lambda {lambda:.6e}, growth {grew:.6e}, miss {miss:.3e}",
                li + 1
            );
            assert!(
                miss < 1e-3,
                "L{}: growth {grew:.6e} does not match the multiplier {lambda:.6e}",
                li + 1
            );
        }
    }

    /// Stable and unstable multipliers are a reciprocal pair — the Hamiltonian (symplectic) symmetry.
    ///
    /// `λ_u·λ_s ≈ 1` alone proves nothing here: `stable_manifold_seed` *returns* `1/λ` of the
    /// reverse-time monodromy, so the product is one whenever the two calls happen to agree, which
    /// they do under any mutation that corrupts both the same way. What makes it a physical claim is
    /// the second half — the stable seed, propagated *backwards* for one period, must contract by
    /// exactly `|λ_s|`, and the reverse-time flow is not what either eigensolve looked at.
    #[test]
    fn stable_and_unstable_multipliers_are_reciprocal() {
        let sys = Cr3bp::EARTH_MOON;
        let h = 2.5e-4;
        let orbit = lyapunov_orbit(sys, 0, 0.01).expect("converges");
        let steps = libm::round(orbit.period / h) as i64;
        let eps = 1e-8;
        let u = unstable_manifold_seed(sys, &orbit, eps).eigenvalue;
        let stable = stable_manifold_seed(sys, &orbit, eps);
        assert!(
            u.abs() > 1e3 && stable.eigenvalue.abs() < 1e-3,
            "|u| {u:e} should be far above 1 and |s| {:e} far below",
            stable.eigenvalue
        );
        assert!(
            (u * stable.eigenvalue - 1.0).abs() < 1e-2,
            "λ_u·λ_s = {} is not 1 (reciprocal pair)",
            u * stable.eigenvalue
        );
        // Backwards for one period: the stable direction contracts by |λ_s|, which is the reciprocal
        // of the *reverse*-time growth the eigensolve reported.
        let base = flow(sys, orbit.state, -h, steps);
        let shrank = separation(flow(sys, stable.state, -h, steps), base) / eps;
        let miss = (shrank - 1.0 / stable.eigenvalue.abs()).abs() * stable.eigenvalue.abs();
        println!(
            "stable: lambda {:.6e}, reverse growth {shrank:.6e}",
            stable.eigenvalue
        );
        assert!(
            miss < 1e-3,
            "reverse-time growth {shrank:.6e} does not match 1/|λ_s| = {:.6e}",
            1.0 / stable.eigenvalue.abs()
        );
    }

    /// A genuinely three-dimensional halo orbit about Earth–Moon L1 converges under the 2×2
    /// differential correction and closes over a full period.
    #[test]
    fn a_three_dimensional_halo_orbit_closes() {
        let sys = Cr3bp::EARTH_MOON;
        // Search a few guesses near L1; halos exist above a minimum out-of-plane amplitude.
        let guesses = [
            (0.823, 0.08, 0.15),
            (0.85, 0.10, 0.18),
            (0.82, 0.06, 0.13),
            (0.84, 0.12, 0.20),
            (0.855, 0.04, 0.17),
            (0.80, 0.10, 0.10),
            (0.87, 0.08, 0.22),
        ];
        let mut found = None;
        for &(x0, z0, vy0) in &guesses {
            if let Some(orbit) = halo_orbit(sys, [x0, 0.0, z0, 0.0, vy0, 0.0])
                && orbit.state[2].abs() > 0.02
            {
                found = Some(orbit);
                break;
            }
        }
        let orbit = found.expect("a halo converged");
        let mut s = orbit.state;
        let mut phi = identity();
        let h = 2.5e-4;
        let steps = libm::round(orbit.period / h) as i64;
        for _ in 0..steps {
            let (ns, np) = rk4(sys, s, &phi, h);
            s = ns;
            phi = np;
        }
        let dr = ((s[0] - orbit.state[0]).powi(2)
            + (s[1] - orbit.state[1]).powi(2)
            + (s[2] - orbit.state[2]).powi(2))
        .sqrt();
        println!(
            "Earth-Moon L1 halo: z-amplitude {:.3}, period {:.3}, closure {dr:.2e}",
            orbit.state[2], orbit.period
        );
        assert!(dr < 1e-4, "halo did not close: {dr:.3e}");
        assert_perpendicular_crossing(sys, &orbit, "L1 halo");
        assert!(
            orbit.state[2].abs() > 0.02,
            "orbit is not 3D (z0 = {})",
            orbit.state[2]
        );
    }

    /// Continuation traces a halo family of increasing out-of-plane amplitude, each member closing.
    #[test]
    fn halo_family_continues_by_amplitude() {
        let sys = Cr3bp::EARTH_MOON;
        let seed = halo_orbit(sys, [0.823, 0.0, 0.08, 0.0, 0.15, 0.0]).expect("seed halo");
        let mut family = std::vec::Vec::new();
        halo_family(sys, seed, 0.002, 10, |o| family.push(o));
        assert!(family.len() >= 3, "family too short: {}", family.len());
        for w in family.windows(2) {
            assert!(
                w[1].state[2] > w[0].state[2],
                "amplitude did not grow along the family"
            );
        }
        let last = *family.last().unwrap();
        let mut s = last.state;
        let mut phi = identity();
        let h = 2.5e-4;
        let steps = libm::round(last.period / h) as i64;
        for _ in 0..steps {
            let (ns, np) = rk4(sys, s, &phi, h);
            s = ns;
            phi = np;
        }
        let dr = ((s[0] - last.state[0]).powi(2)
            + (s[1] - last.state[1]).powi(2)
            + (s[2] - last.state[2]).powi(2))
        .sqrt();
        println!(
            "halo family: {} members, z {:.3} → {:.3}",
            family.len(),
            family[0].state[2],
            last.state[2]
        );
        // RK4-limited over a full period, and amplified by the larger halo's instability.
        assert!(dr < 5e-4, "last family member did not close: {dr:.3e}");
        assert_perpendicular_crossing(sys, &last, "last halo family member");
    }
}
