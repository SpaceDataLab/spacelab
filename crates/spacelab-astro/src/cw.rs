//! Clohessy–Wiltshire (Hills) relative motion — the linearised dynamics of a deputy satellite about a
//! chief on a circular reference orbit, in the chief's rotating LVLH frame.
//!
//! The LVLH frame here is radial `x̂ = r̂`, cross-track `ẑ = ĥ` (along the angular momentum) and
//! along-track `ŷ = ẑ × x̂`. In it the relative state `[x, y, z, ẋ, ẏ, ż]` evolves by the closed-form
//! Clohessy–Wiltshire state-transition matrix at the chief mean motion `n = √(μ/a³)`. This is the
//! workhorse of formation flying and close proximity operations; it is exact only in the linear,
//! circular-chief limit, so it is used for design and short-horizon prediction, with a full nonlinear
//! propagation for truth.

// The LVLH axes (x, y, z), the state (r, v), the angular momentum (h) and the mean motion (n) are
// single letters here because that is exactly the notation of the equations.
#![allow(clippy::many_single_char_names)]

use spacelab_core::Vec3;

fn scale(v: Vec3, s: f64) -> Vec3 {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

/// The chief's LVLH basis `[x̂, ŷ, ẑ]` from its inertial state: radial, along-track, cross-track.
#[must_use]
pub fn lvlh_basis(r: Vec3, v: Vec3) -> [Vec3; 3] {
    let x = scale(r, 1.0 / r.norm());
    let h = r.cross(v);
    let z = scale(h, 1.0 / h.norm());
    let y = z.cross(x);
    [x, y, z]
}

/// A deputy's inertial relative state `(rel_r, rel_v) = (r_dep − r_chief, v_dep − v_chief)` expressed
/// as a Clohessy–Wiltshire state `[x, y, z, ẋ, ẏ, ż]` in the chief's rotating LVLH frame.
///
/// Position projects onto the basis; the LVLH-frame relative velocity subtracts the frame rotation,
/// `v_lvlh = Rᵀ v_rel − ω × r_lvlh` with `ω = n ẑ`.
#[must_use]
pub fn relative_lvlh_state(
    chief_r: Vec3,
    chief_v: Vec3,
    rel_r: Vec3,
    rel_v: Vec3,
    n: f64,
) -> [f64; 6] {
    let [x, y, z] = lvlh_basis(chief_r, chief_v);
    let rl = Vec3::new(rel_r.dot(x), rel_r.dot(y), rel_r.dot(z));
    let vr = Vec3::new(rel_v.dot(x), rel_v.dot(y), rel_v.dot(z));
    // ω × r_lvlh with ω = (0, 0, n).
    let wl = Vec3::new(-n * rl.y, n * rl.x, 0.0);
    [rl.x, rl.y, rl.z, vr.x - wl.x, vr.y - wl.y, vr.z - wl.z]
}

/// The same relative state in **curvilinear** coordinates: radial separation, and arc lengths along
/// and across the chief's orbit rather than straight-line offsets.
///
/// `chief_radius` is `‖r_chief‖`. Writing the deputy's absolute position in the rotating frame as
/// `P = (r_c + x, y, z)`, the three coordinates are
///
/// ```text
/// δr = ‖P‖ − r_c            radial, unchanged in kind
/// s  = r_c · atan2(P_y, P_x)   along-track arc length
/// w  = r_c · asin(P_z / ‖P‖)   cross-track arc length
/// ```
///
/// and the velocities are the time derivatives of those, taken in the rotating frame (so the input's
/// velocity block is already the LVLH-relative one [`relative_lvlh_state`] produces).
///
/// # Why this exists
///
/// The rectilinear along-track axis is a straight line; the chief actually travels an arc. Rectilinear
/// coordinates therefore carry a curvature error `≈ y²/(2 r_c)` before any dynamics are applied at
/// all — 78 m at 1000 km of along-track separation, which is the whole reason a Clohessy–Wiltshire
/// prediction degrades with separation faster than its `O(ρ²/a)` linearisation alone explains. The
/// curvilinear coordinates remove that term by construction. Vallado & Alfano (2014), *Curvilinear
/// Coordinate Transformation for Relative Motion*, Celestial Mechanics and Dynamical Astronomy
/// 118(3):253–271, with the errata in (2016) 125:263–264.
///
/// [`cw_propagate_curvilinear`] is the composition worth reaching for; this and its inverse are
/// public because the coordinates are useful on their own, for stating a separation or a keep-out
/// box in a frame that does not lie about distance.
///
/// # Domain
///
/// Defined wherever the deputy is neither at the geocentre (`‖P‖ = 0`, so no direction exists) nor on
/// the chief's angular-momentum axis through it (`P_x = P_y = 0`, so `s` is arbitrary). Both need a
/// separation of order `r_c`, which is far outside any regime where relative motion is the right
/// model, so they are documented rather than guarded — the same convention [`cw_propagate`] takes
/// with `n = 0`.
#[must_use]
pub fn curvilinear_from_lvlh(state: [f64; 6], chief_radius: f64) -> [f64; 6] {
    let [x, y, z, vx, vy, vz] = state;
    let p = Vec3::new(chief_radius + x, y, z);
    let pdot = Vec3::new(vx, vy, vz);

    let r = p.norm();
    let in_plane = p.x * p.x + p.y * p.y;
    let lambda = libm::atan2(p.y, p.x);
    let phi = libm::asin(p.z / r);

    let r_dot = p.dot(pdot) / r;
    let lambda_dot = (p.x * pdot.y - p.y * pdot.x) / in_plane;
    let phi_dot = (pdot.z * r - p.z * r_dot) / (r * r * libm::cos(phi));

    [
        r - chief_radius,
        chief_radius * lambda,
        chief_radius * phi,
        r_dot,
        chief_radius * lambda_dot,
        chief_radius * phi_dot,
    ]
}

/// The inverse of [`curvilinear_from_lvlh`]: curvilinear back to rectilinear LVLH.
///
/// Exact, not an approximation of the forward map — the round trip is the identity to rounding at
/// every separation, which is what the tests assert.
#[must_use]
pub fn lvlh_from_curvilinear(state: [f64; 6], chief_radius: f64) -> [f64; 6] {
    let [dr, s, w, dr_dot, s_dot, w_dot] = state;
    let (lambda, phi) = (s / chief_radius, w / chief_radius);
    let (lambda_dot, phi_dot) = (s_dot / chief_radius, w_dot / chief_radius);
    let r = chief_radius + dr;

    let (sin_lambda, cos_lambda) = libm::sincos(lambda);
    let (sin_phi, cos_phi) = libm::sincos(phi);

    let p = Vec3::new(
        r * cos_phi * cos_lambda,
        r * cos_phi * sin_lambda,
        r * sin_phi,
    );
    let pdot = Vec3::new(
        dr_dot * cos_phi * cos_lambda
            - r * sin_phi * phi_dot * cos_lambda
            - r * cos_phi * sin_lambda * lambda_dot,
        dr_dot * cos_phi * sin_lambda - r * sin_phi * phi_dot * sin_lambda
            + r * cos_phi * cos_lambda * lambda_dot,
        dr_dot * sin_phi + r * cos_phi * phi_dot,
    );

    [p.x - chief_radius, p.y, p.z, pdot.x, pdot.y, pdot.z]
}

/// [`cw_propagate`], but with the curvature of the chief's orbit taken out of the coordinates first
/// and put back afterwards.
///
/// Takes and returns a **rectilinear** LVLH state, so it is a drop-in for [`cw_propagate`] with one
/// extra argument: the chief's radius, which the rectilinear form never needed because it does not
/// know the along-track axis is curved.
///
/// # Measured
///
/// Against a nonlinear two-body oracle on a circular 7000 km chief, quarter-period propagations, the
/// two forms cross over between 1 and 3 km of initial along-track separation and diverge from there.
/// The full sweep and the crossover are in `tests/relative_motion_curvature.rs`; at 100 km of
/// separation the curvilinear form is two orders better. It removes the *coordinate* curvature and
/// nothing else: an eccentric chief breaks the Clohessy–Wiltshire dynamics themselves, and no change
/// of coordinates repairs that.
#[must_use]
pub fn cw_propagate_curvilinear(state: [f64; 6], chief_radius: f64, n: f64, dt: f64) -> [f64; 6] {
    let curvilinear = curvilinear_from_lvlh(state, chief_radius);
    lvlh_from_curvilinear(cw_propagate(curvilinear, n, dt), chief_radius)
}

/// Propagate a Clohessy–Wiltshire relative state by `dt` seconds under chief mean motion `n`, via the
/// closed-form state-transition matrix.
///
/// The along-track axis is treated as a straight line. [`cw_propagate_curvilinear`] is the same
/// dynamics with the chief's actual curvature removed from the coordinates, and is the better choice
/// past a few kilometres of along-track separation.
#[must_use]
pub fn cw_propagate(state: [f64; 6], n: f64, dt: f64) -> [f64; 6] {
    let [x0, y0, z0, vx0, vy0, vz0] = state;
    let (s, c) = libm::sincos(n * dt);
    let nt = n * dt;

    let x = (4.0 - 3.0 * c) * x0 + (s / n) * vx0 + (2.0 / n) * (1.0 - c) * vy0;
    let y = 6.0 * (s - nt) * x0
        + y0
        + (2.0 / n) * (c - 1.0) * vx0
        + (1.0 / n) * (4.0 * s - 3.0 * nt) * vy0;
    let z = c * z0 + (s / n) * vz0;

    let vx = 3.0 * n * s * x0 + c * vx0 + 2.0 * s * vy0;
    let vy = 6.0 * n * (c - 1.0) * x0 - 2.0 * s * vx0 + (4.0 * c - 3.0) * vy0;
    let vz = -n * s * z0 + c * vz0;

    [x, y, z, vx, vy, vz]
}

#[cfg(test)]
mod tests {
    // The references below are carried at the precision they were computed to, so that
    // regenerating one is a copy rather than a rounding decision.
    #![allow(clippy::excessive_precision, clippy::float_cmp)]
    use spacelab_testkit::assert_within_ulps;

    use super::*;
    use crate::kepler::propagate;

    const MU: f64 = 3.986_004_418e14;
    const TAU: f64 = core::f64::consts::TAU;

    /// A mean motion and a span whose product is not a multiple of a quarter turn, so neither `sin`
    /// nor `cos` is 0 or ±1 and no coefficient of the transition matrix drops out. `n·dt = 1.3579`.
    const N: f64 = 0.001_1;
    const DT: f64 = 1234.5;
    /// Six distinct nonzero components, none a multiple of another.
    const S0: [f64; 6] = [1000.0, -2000.0, 500.0, 1.0, -0.5, 0.3];
    const NAMES: [&str; 6] = ["x", "y", "z", "vx", "vy", "vz"];

    /// A chief on no axis and in no plane, so the three basis vectors are distinct and every term
    /// of the two cross products matters.
    const CHIEF_R: Vec3 = Vec3::new(3000e3, 4000e3, 5000e3);
    const CHIEF_V: Vec3 = Vec3::new(-4200.0, 1300.0, 1500.0);

    /// Every entry of the transition matrix, against the published closed form.
    ///
    /// The three tests that were here compare the propagator against a nonlinear two-body oracle
    /// and are the right check on the *model* — but the linearisation error is the signal there, so
    /// they can only assert to a percent, and a percent leaves every coefficient free. Twenty-three
    /// mutations of this function survived them. The reference below is the Clohessy–Wiltshire
    /// matrix as published (Vallado, *Fundamentals of Astrodynamics and Applications*, §6.7;
    /// Alfriend et al., *Spacecraft Formation Flying*, §5.2), evaluated at 60 digits.
    #[test]
    fn every_entry_of_the_transition_matrix_is_the_published_closed_form() {
        let got = cw_propagate(S0, N, DT);
        let want = [
            3_537.795_509_123_616_011_503_969,
            -5_642.604_277_551_699_946_432_585,
            372.194_226_011_710_110_246_981_5,
            2.459_340_147_346_565_387_335_485,
            -6.083_150_120_071_955_561_650_064,
            -0.474_215_630_625_444_814_548_305_6,
        ];
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            // Eight operations deep with a `1/n` of order a thousand in the middle of it; measured,
            // the worst of the six is one ulp.
            assert_within_ulps(g, w, 4, NAMES[i]);
        }
    }

    /// The propagated trajectory solves `ẍ − 2nẏ − 3n²x = 0`, `ÿ + 2nẋ = 0`, `z̈ + n²z = 0`.
    ///
    /// This is the one check on the transition matrix that needs no reference at all: it asks
    /// whether the closed form is a solution of the equations it claims to solve, by differentiating
    /// the propagated state numerically and substituting. It ties the position block to the velocity
    /// block, which no per-entry comparison does — an error that appears identically in both would
    /// pass every value assertion and fail this one.
    ///
    /// A fourth-order central difference at `h = 1 s` sits near the balance point between truncation
    /// (`h⁴f⁽⁵⁾/30`, and every derivative here costs a factor `n = 1.1e-3`) and cancellation
    /// (`ε·f/h`). The bounds are the measured residuals rounded up, not a guess.
    #[test]
    fn the_propagated_trajectory_solves_the_clohessy_wiltshire_equations() {
        let h = 1.0;
        let ddt = |i: usize, t: f64| {
            let at = |k: f64| cw_propagate(S0, N, t + k * h)[i];
            (at(-2.0) - 8.0 * at(-1.0) + 8.0 * at(1.0) - at(2.0)) / (12.0 * h)
        };
        for &t in &[0.0, 300.0, DT, 4321.0] {
            let s = cw_propagate(S0, N, t);
            // The velocity block is the derivative of the position block.
            for i in 0..3 {
                let r = ddt(i, t) - s[i + 3];
                assert!(r.abs() < 1e-11, "t {t}: d(r{i})/dt - v{i} = {r:.3e}");
            }
            // And the derivative of the velocity block is the right-hand side of the equations.
            let rhs = [
                2.0 * N * s[4] + 3.0 * N * N * s[0],
                -2.0 * N * s[3],
                -N * N * s[2],
            ];
            for (i, &want) in rhs.iter().enumerate() {
                let r = ddt(i + 3, t) - want;
                assert!(r.abs() < 1e-14, "t {t}: a{i} residual {r:.3e}");
            }
        }
    }

    /// `Φ(t₂)Φ(t₁) = Φ(t₁+t₂)`: propagating in two steps is propagating in one. A transition matrix
    /// that is not the exponential of a constant system matrix fails this for almost any pair of
    /// spans, and it holds for every state at once rather than for one chosen sample.
    #[test]
    fn propagating_in_two_steps_is_propagating_in_one() {
        for &(t1, t2) in &[
            (300.0, 900.0),
            (DT, -DT / 3.0),
            (-700.0, -1500.0),
            (0.0, DT),
        ] {
            let two = cw_propagate(cw_propagate(S0, N, t1), N, t2);
            let one = cw_propagate(S0, N, t1 + t2);
            for i in 0..6 {
                // The position entries reach 1e4 m here, so a few ulps of them is 1e-11.
                let d = (two[i] - one[i]).abs();
                assert!(d < 1e-11, "({t1}, {t2}) component {i}: {d:.3e}");
            }
        }
    }

    /// The LVLH triad is orthonormal and right-handed, with `x̂` along the radius and `ẑ` along the
    /// angular momentum. Those four statements fix the basis uniquely, and none of them restates
    /// the construction — a swapped cross-product argument or a dropped normalisation breaks one.
    #[test]
    fn the_lvlh_triad_is_orthonormal_right_handed_and_carried_by_the_orbit() {
        let [x, y, z] = lvlh_basis(CHIEF_R, CHIEF_V);
        for (n, v) in [("x", x), ("y", y), ("z", z)] {
            assert!((v.norm() - 1.0).abs() < 1e-15, "|{n}| = {}", v.norm());
        }
        for (n, a, b) in [("xy", x, y), ("yz", y, z), ("zx", z, x)] {
            assert!(a.dot(b).abs() < 1e-15, "{n} dot = {}", a.dot(b));
        }
        assert!((x.cross(y) - z).norm() < 1e-15, "not right-handed");
        // Parallel and pointing the same way, stated as a cross product and a sign rather than as
        // the quotient that built them.
        let h = CHIEF_R.cross(CHIEF_V);
        assert!(
            x.cross(CHIEF_R).norm() < 1e-8 && x.dot(CHIEF_R) > 0.0,
            "x off the radius"
        );
        assert!(
            z.cross(h).norm() < 1e-5 && z.dot(h) > 0.0,
            "z off the angular momentum"
        );
    }

    /// A deputy rigidly attached to the rotating frame has no velocity in it. Its inertial relative
    /// velocity is `ω × ρ` exactly, so the `ω × r_lvlh` that `relative_lvlh_state` subtracts has to
    /// cancel it to nothing — which pins that term's sign, its axis and its factor `n` at once. The
    /// `ω` here is built from the chief state directly, not from `lvlh_basis`, so a wrong triad
    /// cannot arrange for the test to agree with itself.
    #[test]
    fn a_deputy_frozen_in_the_rotating_frame_has_no_lvlh_velocity() {
        let h = CHIEF_R.cross(CHIEF_V);
        let omega = h.scale(N / h.norm());
        let rho = Vec3::new(1000.0, -2000.0, 500.0);
        let state = relative_lvlh_state(CHIEF_R, CHIEF_V, rho, omega.cross(rho), N);
        for (i, c) in state.iter().enumerate().skip(3) {
            assert!(c.abs() < 1e-15, "component {i} is {c:.3e}");
        }
        // The position half is unaffected by the rotation and must still be the full separation.
        let carried = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
        assert!(
            (carried - rho.norm()).abs() < 1e-9,
            "{carried} vs {}",
            rho.norm()
        );
    }

    /// The relative state is the projection onto the triad, less the frame rotation, at 60 digits.
    #[test]
    fn the_relative_state_is_the_projection_onto_the_triad_less_the_frame_rotation() {
        let got = relative_lvlh_state(
            CHIEF_R,
            CHIEF_V,
            Vec3::new(1000.0, -2000.0, 500.0),
            Vec3::new(1.7, -0.9, 2.3),
            N,
        );
        let want = [
            -353.553_390_593_273_762_200_422_2,
            -1_301.285_520_896_677_504_480_514,
            1_852.472_939_911_582_809_357_446,
            0.407_063_558_098_678_065_272_829_0,
            -0.662_883_008_163_968_966_452_654_9,
            2.122_200_306_347_353_030_709_691,
        ];
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            assert_within_ulps(g, w, 8, NAMES[i]);
        }
    }

    #[test]
    fn cw_matches_nonlinear_relative_motion_for_small_separations() {
        // Oracle: propagate a chief (circular) and a deputy (chief + a small relative state) with the
        // full two-body flow, transform the true relative state to LVLH, and compare to the CW STM.
        // They agree to the linearisation error O(ρ²/a), far below the ~km signal.
        let r0 = 7_000e3;
        let vc = (MU / r0).sqrt();
        let n = vc / r0;
        let chief_r = Vec3::new(r0, 0.0, 0.0);
        let chief_v = Vec3::new(0.0, vc, 0.0);
        // At t=0 the LVLH axes coincide with the inertial axes.
        let s0 = [1_000.0, -2_000.0, 500.0, 1.0, -0.5, 0.3]; // m, m/s
        let rel_r = Vec3::new(s0[0], s0[1], s0[2]);
        // Inertial relative velocity adds the frame rotation ω × r: v_inertial = v_lvlh + ω × r.
        let rel_v = Vec3::new(s0[3] - n * s0[1], s0[4] + n * s0[0], s0[5]);
        let dep_r = chief_r + rel_r;
        let dep_v = chief_v + rel_v;

        let period = TAU / n;
        for frac in [0.1, 0.25, 0.5] {
            let dt = frac * period;
            let (cr, cv) = propagate(chief_r, chief_v, MU, dt);
            let (dr, dv) = propagate(dep_r, dep_v, MU, dt);
            let truth = relative_lvlh_state(cr, cv, dr - cr, dv - cv, n);
            let cw = cw_propagate(s0, n, dt);
            let perr = ((truth[0] - cw[0]).powi(2)
                + (truth[1] - cw[1]).powi(2)
                + (truth[2] - cw[2]).powi(2))
            .sqrt();
            let mag = (truth[0].powi(2) + truth[1].powi(2) + truth[2].powi(2)).sqrt();
            // CW is a linear model, so its accuracy is relative: the error is a small fraction of the
            // (multi-km) relative displacement — here well under 1%. That 1% is a statement about
            // *this* separation, not about the model: `tests/relative_motion_curvature.rs` measures
            // where it stops holding (30.75 km over a quarter period, 3.71 km over a full one) and
            // what `cw_propagate_curvilinear` buys past it.
            assert!(
                perr / mag < 0.01,
                "frac {frac}: CW vs truth relative position error {:.4} ({perr:.1} m of {mag:.0} m)",
                perr / mag
            );
        }
    }

    #[test]
    fn linearisation_error_shrinks_with_separation() {
        // Halving the separation should roughly quarter the CW-vs-truth error (second-order model).
        let r0 = 7_000e3;
        let vc = (MU / r0).sqrt();
        let n = vc / r0;
        let chief_r = Vec3::new(r0, 0.0, 0.0);
        let chief_v = Vec3::new(0.0, vc, 0.0);
        let dt = 0.25 * TAU / n;

        let err = |scale_factor: f64| {
            let s0 = [
                1_000.0 * scale_factor,
                -2_000.0 * scale_factor,
                500.0 * scale_factor,
                1.0 * scale_factor,
                -0.5 * scale_factor,
                0.3 * scale_factor,
            ];
            let rel_r = Vec3::new(s0[0], s0[1], s0[2]);
            let rel_v = Vec3::new(s0[3] - n * s0[1], s0[4] + n * s0[0], s0[5]);
            let (cr, cv) = propagate(chief_r, chief_v, MU, dt);
            let (dr, dv) = propagate(chief_r + rel_r, chief_v + rel_v, MU, dt);
            let truth = relative_lvlh_state(cr, cv, dr - cr, dv - cv, n);
            let cw = cw_propagate(s0, n, dt);
            ((truth[0] - cw[0]).powi(2) + (truth[1] - cw[1]).powi(2) + (truth[2] - cw[2]).powi(2))
                .sqrt()
        };
        let big = err(1.0);
        let small = err(0.5);
        // Second order ⟹ ratio ≈ 4; allow a generous band around it.
        assert!(big / small > 3.0, "error ratio {} not ~4", big / small);
    }

    #[test]
    fn drift_free_relative_orbit_is_periodic() {
        // The Clohessy–Wiltshire bounded orbit condition ẏ0 = −2 n x0 cancels the secular along-track
        // drift, so the relative state returns to itself after one chief period.
        let n = 0.001_1;
        let x0 = 500.0;
        let s0 = [x0, 0.0, 300.0, 0.4, -2.0 * n * x0, 0.2];
        let period = TAU / n;
        let back = cw_propagate(s0, n, period);
        for i in 0..6 {
            assert!(
                (back[i] - s0[i]).abs() < 1e-6,
                "component {i}: {} vs {}",
                back[i],
                s0[i]
            );
        }
    }
}
