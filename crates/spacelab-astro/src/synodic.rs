//! The CR3BP synodic (rotating) frame and its transformation to an inertial frame.
//!
//! CR3BP orbits are computed in the co-rotating synodic frame, where the primaries are fixed; to
//! propagate them in an ephemeris model, plot a ground track, or patch them to a conic, they must be
//! expressed in an inertial frame. The synodic frame rotates about `ẑ` at the (nondimensional) mean
//! motion 1, so at rotation angle `θ` the map is a rotation `Rz(θ)` on position and, on velocity, the
//! same rotation applied to the synodic velocity plus the frame's `ω × r` term (`ω = ẑ`). Units are
//! nondimensional CR3BP; scale to physical units with the system's length and time.

#![allow(clippy::many_single_char_names, clippy::similar_names)]

/// A synodic or inertial state `[x, y, z, ẋ, ẏ, ż]`.
pub type State = [f64; 6];

fn rot_z(theta: f64, x: f64, y: f64) -> (f64, f64) {
    let (s, c) = (libm::sin(theta), libm::cos(theta));
    (c * x - s * y, s * x + c * y)
}

/// Synodic → inertial at rotation angle `θ`. The inertial velocity carries the `ω × r` term of the
/// rotating frame (`ω = ẑ`), so `v_I = Rz(θ)·(v_S + [−y, x, 0])`.
#[must_use]
pub fn synodic_to_inertial(s: State, theta: f64) -> State {
    let (rx, ry) = rot_z(theta, s[0], s[1]);
    // velocity in the rotating frame plus the frame rotation ω × r, then rotate into inertial.
    let (vx_r, vy_r) = (s[3] - s[1], s[4] + s[0]);
    let (vx, vy) = rot_z(theta, vx_r, vy_r);
    [rx, ry, s[2], vx, vy, s[5]]
}

/// Inertial → synodic at rotation angle `θ`, the inverse of [`synodic_to_inertial`].
#[must_use]
pub fn inertial_to_synodic(s: State, theta: f64) -> State {
    let (rx, ry) = rot_z(-theta, s[0], s[1]);
    let (vx_i, vy_i) = rot_z(-theta, s[3], s[4]);
    // remove the frame rotation ω × r (r is now the synodic position).
    [rx, ry, s[2], vx_i + ry, vy_i - rx, s[5]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_the_identity() {
        let s: State = [0.85, 0.05, 0.03, 0.01, 0.12, -0.02];
        for &theta in &[0.0, 0.7, 2.4, -1.3, 6.0] {
            let back = inertial_to_synodic(synodic_to_inertial(s, theta), theta);
            for i in 0..6 {
                assert!(
                    (back[i] - s[i]).abs() < 1e-13,
                    "component {i} drifted at θ={theta}"
                );
            }
        }
    }

    /// A libration point (fixed in the synodic frame) revolves on a circle in the inertial frame at
    /// the mean motion 1 — `|v_I| = |r_I|` and `v_I ⟂ r_I`.
    #[test]
    fn a_synodic_equilibrium_is_circular_inertial_motion() {
        let l1_x = 0.836_915; // Earth-Moon L1, on the x-axis, zero synodic velocity
        let fixed: State = [l1_x, 0.0, 0.0, 0.0, 0.0, 0.0];
        for &theta in &[0.0, 1.1, 3.0] {
            let s = synodic_to_inertial(fixed, theta);
            let r = (s[0] * s[0] + s[1] * s[1]).sqrt();
            let v = (s[3] * s[3] + s[4] * s[4]).sqrt();
            assert!(
                (v - r).abs() < 1e-12,
                "|v_I| {v} != |r_I| {r} (mean motion 1)"
            );
            let dot = s[0] * s[3] + s[1] * s[4];
            assert!(
                dot.abs() < 1e-12,
                "v_I not perpendicular to r_I (dot {dot})"
            );
        }
    }
}
