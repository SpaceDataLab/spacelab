//! Attitude *laws* (pointing providers) — the analogue of Orekit's `AttitudeProvider`.
//!
//! Where [`crate::attitude`] *integrates* a rigid body's orientation on `SO(3)`, this module *pre-
//! scribes* it: given a spacecraft's inertial position/velocity (and, where a law needs it, the
//! satellite→Sun direction and time), each provider returns the **inertial→body rotation matrix**
//! `M` with `v_body = M · v_inertial`. That is exactly Orekit `attitude.getRotation().getMatrix()`,
//! and exactly the `inertial_to_body` matrix [`crate::spacecraft::BoxWing::drag_acceleration_inertial`]
//! and `srp_acceleration_inertial` already consume — so a provider plugs an attitude-dependent force
//! straight in.
//!
//! `M`'s rows are the body axes expressed in the inertial frame: `(M v)_i = ê_i · v`, so
//! `M = [x̂; ŷ; ẑ]` row-wise. Every law below builds an orthonormal right-handed `{x̂, ŷ, ẑ}` and
//! returns those three rows.
//!
//! The provided laws mirror the Orekit set:
//! * [`LocalOrbitalFrame`] — the CCSDS / Vallado local orbital frames ([`LofType`]): `Qsw` (RSW/RTN),
//!   `Tnw`, `Vnc`, and `Lvlh` (the nadir-pointing local-vertical/local-horizontal frame — Orekit
//!   `NadirPointing` for a spherical Earth);
//! * [`YawSteering`] — nadir `ẑ` with a yaw that drives body `+x̂` onto the Sun's local-horizontal
//!   projection (Orekit `YawSteering`), keeping the array axis `ŷ` ⟂ Sun;
//! * [`SunPointing`] — body `+ẑ` on the Sun with the velocity resolving roll (Orekit
//!   `CelestialBodyPointed`);
//! * [`SpinStabilized`] — a base orientation spun at a constant rate about a fixed inertial axis
//!   (Orekit `SpinStabilized`).
//!
//! Gate: `derivations/attitude_law_frames.mac` proves that each of the three row shapes the laws use
//! — `(x, z×x, z)`, `(x, y, x×y)`, `(y×z, y, z)` — is orthonormal with determinant `+1` exactly when
//! its two input vectors are an orthonormal pair, that every law feeds it one, and that Rodrigues'
//! exponential is a one-parameter group fixing its axis; `derivations/attitude_law_frames.gp` builds
//! every law at 100 digits on 200 states. The module tests
//! `every_law_is_an_orthonormal_right_handed_frame_to_rounding` (G1) and
//! `the_spin_law_composes_as_a_one_parameter_group` (G4) hold the code to them.

// The single letters (r, v, h, x, y, z, s) are the notation of orbital-frame geometry.
#![allow(clippy::many_single_char_names)]

use crate::attitude::exp_so3;

/// A 3-vector.
pub type Vec3 = [f64; 3];
/// A 3×3 matrix, row-major.
pub type Mat3 = [[f64; 3]; 3];

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm(a: Vec3) -> f64 {
    libm::sqrt(dot(a, a))
}

/// Unit vector; returns the input unchanged if it is (near) zero — callers guard degeneracy first.
fn normalize(a: Vec3) -> Vec3 {
    let n = norm(a);
    if n < 1e-300 { a } else { scale(a, 1.0 / n) }
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

/// The inertial→body matrix whose rows are the body axes `{x̂, ŷ, ẑ}` in inertial coordinates.
fn frame(x: Vec3, y: Vec3, z: Vec3) -> Mat3 {
    [x, y, z]
}

/// An attitude law: the inertial→body rotation matrix as a function of the spacecraft state.
///
/// `t` is seconds from the caller's reference epoch; `position`/`velocity` are inertial; `sun` is the
/// (not necessarily unit) satellite→Sun direction, inertial. Laws that don't need a given input
/// ignore it (a local orbital frame ignores `t` and `sun`; a spin law ignores `position`/`velocity`).
pub trait AttitudeProvider {
    /// The inertial→body matrix `M` (`v_body = M · v_inertial`).
    fn inertial_to_body(&self, t: f64, position: Vec3, velocity: Vec3, sun: Vec3) -> Mat3;
}

/// Which local orbital frame [`LocalOrbitalFrame`] realises. Axes are given as the body basis in
/// inertial coordinates; `r̂` is radial-out, `v̂` the velocity direction, `ĥ = (r×v)/|r×v|` the orbit
/// normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LofType {
    /// RSW / RTN (CCSDS "LVLH"): `x̂ = r̂`, `ẑ = ĥ`, `ŷ = ẑ×x̂` (along-track for a circular orbit).
    Qsw,
    /// `x̂ = v̂`, `ẑ = ĥ`, `ŷ = ẑ×x̂`.
    Tnw,
    /// `x̂ = v̂`, `ŷ = ĥ`, `ẑ = x̂×ŷ`.
    Vnc,
    /// Nadir local-vertical/local-horizontal: `ẑ = −r̂` (nadir), `ŷ = −ĥ`, `x̂ = ŷ×ẑ` (≈ `v̂`). This is
    /// the spherical-Earth `NadirPointing` frame.
    Lvlh,
}

/// A local-orbital-frame attitude law (Orekit `LofOffset` with zero offset / `NadirPointing`).
#[derive(Debug, Clone, Copy)]
pub struct LocalOrbitalFrame {
    /// Which local frame to align the body with.
    pub lof: LofType,
}

impl LocalOrbitalFrame {
    /// A law aligning the body with the given local orbital frame.
    #[must_use]
    pub const fn new(lof: LofType) -> Self {
        Self { lof }
    }

    /// The nadir-pointing local frame ([`LofType::Lvlh`]) — spherical-Earth `NadirPointing`.
    #[must_use]
    pub const fn nadir() -> Self {
        Self { lof: LofType::Lvlh }
    }

    /// The inertial→body matrix from position/velocity alone (independent of time and Sun).
    #[must_use]
    pub fn matrix(&self, position: Vec3, velocity: Vec3) -> Mat3 {
        let rhat = normalize(position);
        let hhat = normalize(cross(position, velocity));
        let vhat = normalize(velocity);
        match self.lof {
            LofType::Qsw => {
                let (x, z) = (rhat, hhat);
                frame(x, cross(z, x), z)
            }
            LofType::Tnw => {
                let (x, z) = (vhat, hhat);
                frame(x, cross(z, x), z)
            }
            LofType::Vnc => {
                let (x, y) = (vhat, hhat);
                frame(x, y, cross(x, y))
            }
            LofType::Lvlh => {
                let z = scale(rhat, -1.0); // nadir
                let y = scale(hhat, -1.0);
                frame(cross(y, z), y, z)
            }
        }
    }
}

impl AttitudeProvider for LocalOrbitalFrame {
    fn inertial_to_body(&self, _t: f64, position: Vec3, velocity: Vec3, _sun: Vec3) -> Mat3 {
        self.matrix(position, velocity)
    }
}

/// Nadir `ẑ` with the yaw that points body `+x̂` at the Sun's projection into the local horizontal
/// plane — Orekit `YawSteering`. The array axis `ŷ = ẑ×x̂` is then ⟂ the Sun (`ŷ · sun = 0`), maximising
/// illumination of a `±ŷ`-mounted solar array. If the Sun is (near) parallel to nadir the roll is
/// undefined, and the law falls back to the [`LofType::Lvlh`] frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct YawSteering;

impl AttitudeProvider for YawSteering {
    fn inertial_to_body(&self, _t: f64, position: Vec3, velocity: Vec3, sun: Vec3) -> Mat3 {
        let z = scale(normalize(position), -1.0); // nadir
        let s = normalize(sun);
        let s_perp = sub(s, scale(z, dot(s, z))); // Sun in the local horizontal plane
        if norm(s_perp) < 1e-9 {
            return LocalOrbitalFrame::nadir().matrix(position, velocity);
        }
        // `s_perp` is a cancellation whose rounding is `ε/‖s_perp‖` of its length, and that much of
        // it lies along `z`: kept as the axis, it would leave the frame `ε/‖s_perp‖` off
        // orthogonal. `z × ŝ⊥` is near unit length, so the cross products complete an exact frame.
        let y = normalize(cross(z, normalize(s_perp)));
        frame(cross(y, z), y, z)
    }
}

/// Body `+ẑ` pointed at the Sun, with the velocity resolving roll — Orekit `CelestialBodyPointed`
/// (primary axis on the target, secondary axis kept near a phasing vector). `ŷ = ẑ×v̂` normalised,
/// `x̂ = ŷ×ẑ`. If the Sun is (near) parallel to the velocity the roll falls back to the position
/// vector as the phasing reference.
#[derive(Debug, Clone, Copy, Default)]
pub struct SunPointing;

impl AttitudeProvider for SunPointing {
    fn inertial_to_body(&self, _t: f64, position: Vec3, velocity: Vec3, sun: Vec3) -> Mat3 {
        let z = normalize(sun);
        let mut y = cross(z, normalize(velocity));
        if norm(y) < 1e-9 {
            y = cross(z, normalize(position));
        }
        // A short `ẑ × v̂` carries `ε/‖ẑ × v̂‖` of its length along `ẑ`; taking `x̂` from it and `ŷ`
        // back from `ẑ × x̂` makes both cross products of near-orthogonal unit vectors.
        let x = normalize(cross(normalize(y), z));
        frame(x, cross(z, x), z)
    }
}

/// A base orientation spun at a constant rate about a fixed inertial axis — Orekit `SpinStabilized`.
///
/// `M(t) = base · exp([−â]× · rate·(t − t0))`: at `t = t0` the attitude is `base`; the inertial spin
/// axis `â` keeps a constant body representation `base·â`; a general inertial vector's body
/// coordinates rotate at `−rate` about `base·â`, returning to `base` every `2π/rate`.
#[derive(Debug, Clone, Copy)]
pub struct SpinStabilized {
    base: Mat3,
    axis: Vec3,
    t0: f64,
    rate: f64,
}

impl SpinStabilized {
    /// A spin law with inertial→body orientation `base` at `t0`, spinning at `rate` (rad/s) about the
    /// inertial axis `axis` (normalised here).
    #[must_use]
    pub fn new(base: Mat3, t0: f64, axis: Vec3, rate: f64) -> Self {
        Self {
            base,
            axis: normalize(axis),
            t0,
            rate,
        }
    }
}

impl AttitudeProvider for SpinStabilized {
    fn inertial_to_body(&self, t: f64, _position: Vec3, _velocity: Vec3, _sun: Vec3) -> Mat3 {
        let angle = self.rate * (t - self.t0);
        matmul(&self.base, &exp_so3(scale(self.axis, -angle)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PI: f64 = core::f64::consts::PI;

    /// `‖MᵀM − I‖∞` and `det M` — an orthonormal right-handed frame has both ≈ (0, +1).
    fn orthonormality(m: &Mat3) -> (f64, f64) {
        let mut max_off = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                let dotij = m[i][0] * m[j][0] + m[i][1] * m[j][1] + m[i][2] * m[j][2];
                let expect = if i == j { 1.0 } else { 0.0 };
                max_off = max_off.max((dotij - expect).abs());
            }
        }
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        (max_off, det)
    }

    fn apply(m: &Mat3, v: Vec3) -> Vec3 {
        [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
    }

    /// `‖A − B‖∞`.
    fn max_abs_diff(a: &Mat3, b: &Mat3) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..3 {
            for j in 0..3 {
                worst = worst.max((a[i][j] - b[i][j]).abs());
            }
        }
        worst
    }

    /// A generic (non-circular, out-of-plane) state so no two axes coincide by accident.
    fn sample_state() -> (Vec3, Vec3) {
        (
            [7_000_000.0, 1_000_000.0, -500_000.0],
            [500.0, 7_200.0, 1_100.0],
        )
    }

    #[test]
    fn every_local_frame_is_orthonormal_and_right_handed() {
        let (r, v) = sample_state();
        for lof in [LofType::Qsw, LofType::Tnw, LofType::Vnc, LofType::Lvlh] {
            let m = LocalOrbitalFrame::new(lof).inertial_to_body(0.0, r, v, [0.0; 3]);
            let (off, det) = orthonormality(&m);
            assert!(off < 1e-12, "{lof:?} not orthonormal: {off:e}");
            assert!((det - 1.0).abs() < 1e-12, "{lof:?} det {det}");
        }
    }

    #[test]
    fn qsw_axes_are_radial_alongtrack_normal_on_a_circular_orbit() {
        let a = 7_000_000.0;
        let vc = 7_546.0;
        let (r, v) = ([a, 0.0, 0.0], [0.0, vc, 0.0]); // prograde circular in the xy-plane
        let m = LocalOrbitalFrame::new(LofType::Qsw).inertial_to_body(0.0, r, v, [0.0; 3]);
        // Rows: x̂ = r̂ = +x, ŷ = along-track = +y, ẑ = ĥ = +z.
        assert!((apply(&m, [1.0, 0.0, 0.0])[0] - 1.0).abs() < 1e-12); // r̂ → body x
        assert!((apply(&m, [0.0, 1.0, 0.0])[1] - 1.0).abs() < 1e-12); // v̂ → body y
        assert!((apply(&m, [0.0, 0.0, 1.0])[2] - 1.0).abs() < 1e-12); // ĥ → body z
    }

    #[test]
    fn lvlh_body_z_points_to_nadir() {
        let (r, v) = sample_state();
        let m = LocalOrbitalFrame::nadir().inertial_to_body(0.0, r, v, [0.0; 3]);
        // The radial vector, in body coordinates, lies along −ẑ: (M r)_z = −|r|.
        let body_r = apply(&m, r);
        let rmag = norm(r);
        assert!(
            (body_r[2] + rmag).abs() < 1e-6 * rmag,
            "z {} vs -{rmag}",
            body_r[2]
        );
        // and its radial/along-track components vanish (pure nadir).
        assert!(body_r[0].abs() < 1e-6 * rmag && body_r[1].abs() < 1e-6 * rmag);
    }

    #[test]
    fn yaw_steering_keeps_nadir_and_puts_the_sun_in_the_x_z_plane() {
        let (r, v) = sample_state();
        let sun = [1.0e11, -3.0e10, 4.0e10];
        let m = YawSteering.inertial_to_body(0.0, r, v, sun);
        let (off, det) = orthonormality(&m);
        assert!(off < 1e-12 && (det - 1.0).abs() < 1e-12);
        // Nadir preserved: body ẑ maps the radial vector to −|r|.
        let body_r = apply(&m, r);
        assert!((body_r[2] + norm(r)).abs() < 1e-6 * norm(r));
        // Sun has no body-y component and a positive body-x component (points at the Sun projection).
        let body_sun = apply(&m, sun);
        assert!(body_sun[1].abs() < 1e-3, "sun·ŷ = {}", body_sun[1]);
        assert!(body_sun[0] > 0.0, "sun·x̂ = {}", body_sun[0]);
    }

    #[test]
    fn sun_pointing_puts_body_z_on_the_sun() {
        let (r, v) = sample_state();
        let sun = [1.0e11, -3.0e10, 4.0e10];
        let m = SunPointing.inertial_to_body(0.0, r, v, sun);
        let (off, det) = orthonormality(&m);
        assert!(off < 1e-12 && (det - 1.0).abs() < 1e-12);
        // The Sun, in body coordinates, lies along +ẑ.
        let body_sun = normalize(apply(&m, sun));
        assert!(
            (body_sun[2] - 1.0).abs() < 1e-12,
            "sun in body: {body_sun:?}"
        );
    }

    #[test]
    fn spin_fixes_the_inertial_axis_and_is_periodic() {
        let base = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let axis = [0.0, 0.0, 1.0];
        let rate = 0.5; // rad/s
        let law = SpinStabilized::new(base, 0.0, axis, rate);
        let ignore = [0.0; 3];

        // At t0 the attitude is the base.
        let m0 = law.inertial_to_body(0.0, ignore, ignore, ignore);
        assert_eq!(m0, base);

        // The inertial spin axis keeps a constant body representation.
        let t = 1.234;
        let mt = law.inertial_to_body(t, ignore, ignore, ignore);
        let axis_body = apply(&mt, axis);
        assert!(
            (axis_body[2] - 1.0).abs() < 1e-12,
            "spin axis moved: {axis_body:?}"
        );

        // A quarter turn rotates body coordinates of x̂ by −rate·t about +z.
        let quarter = (PI / 2.0) / rate;
        let mq = law.inertial_to_body(quarter, ignore, ignore, ignore);
        let x_body = apply(&mq, [1.0, 0.0, 0.0]);
        assert!(
            (x_body[0] - 0.0).abs() < 1e-9 && (x_body[1] + 1.0).abs() < 1e-9,
            "{x_body:?}"
        );

        // One full period returns to the base.
        let period = 2.0 * PI / rate;
        let mp = law.inertial_to_body(period, ignore, ignore, ignore);
        let (off, _) = orthonormality(&mp);
        assert!(off < 1e-12);
        for i in 0..3 {
            for j in 0..3 {
                assert!((mp[i][j] - base[i][j]).abs() < 1e-9);
            }
        }
    }

    /// (G1) Every law's matrix is orthonormal with determinant `+1` to rounding, on 500 states
    /// spread over every direction — the identity `derivations/attitude_law_frames.mac` proves for
    /// each row shape. A unit vector from [`normalize`] is within an ulp of unit length and a cross
    /// product of two adds a few more, so `16 ε` bounds `‖MMᵀ − I‖∞` and `|det M − 1|`; a frame with a
    /// wrong cross-product order or a missed normalisation is off at order one.
    #[test]
    fn every_law_is_an_orthonormal_right_handed_frame_to_rounding() {
        let mut seed = 0x2026_1026_u64;
        let mut next = move || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)] // 53 of the 64 bits are all a uniform draw needs
            let u = (seed >> 11) as f64 / (1_u64 << 53) as f64;
            u - 0.5
        };
        let bound = 16.0 * f64::EPSILON;
        let mut worst = 0.0_f64;
        for _ in 0..500 {
            let r = [1.4e7 * next(), 1.4e7 * next(), 1.4e7 * next()];
            let v = [1.6e4 * next(), 1.6e4 * next(), 1.6e4 * next()];
            let sun = [3e11 * next(), 3e11 * next(), 3e11 * next()];
            let laws: [(&str, Mat3); 6] = [
                ("qsw", LocalOrbitalFrame::new(LofType::Qsw).matrix(r, v)),
                ("tnw", LocalOrbitalFrame::new(LofType::Tnw).matrix(r, v)),
                ("vnc", LocalOrbitalFrame::new(LofType::Vnc).matrix(r, v)),
                ("lvlh", LocalOrbitalFrame::nadir().matrix(r, v)),
                ("yaw", YawSteering.inertial_to_body(0.0, r, v, sun)),
                ("sun", SunPointing.inertial_to_body(0.0, r, v, sun)),
            ];
            for (name, m) in laws {
                let (off, det) = orthonormality(&m);
                let miss = off.max((det - 1.0).abs());
                assert!(miss <= bound, "{name} at r {r:?}, v {v:?}: {miss:e}");
                worst = worst.max(miss / f64::EPSILON);
            }
        }
        println!("attitude frames: worst {worst:.2} ε");
    }

    /// (G1) The same bound where the Sun-referenced laws are worst conditioned: the Sun a few
    /// nanoradians off nadir for [`YawSteering`] and off the velocity for [`SunPointing`], just above
    /// each fallback threshold. There the reference direction is a near-cancellation (`s − (s·z)z`, or
    /// `ẑ × v̂`), whose rounding is `ε/sin` of the angle relative to its own length, so a frame that
    /// keeps it as an axis is off orthogonality by the same `ε/sin`. Both laws did, and this test
    /// measured them at `4.3e-9` (yaw, 1e-8 rad) and `5.9e-10` (Sun pointing, 3e-9 rad); completed
    /// by cross products of the two well-conditioned axes, as they now are, both stay under `2 ε`.
    #[test]
    fn the_sun_referenced_laws_stay_orthonormal_next_to_their_degeneracy() {
        let position = [4.1e6, -3.3e6, 5.2e6];
        let rn = norm(position);
        let nadir = scale(position, -1.0 / rn);
        let side = normalize(cross(nadir, [0.0, 0.0, 1.0]));
        let velocity = [-5.0e3, 1.0e3, 6.0e3];
        let vhat = normalize(velocity);
        let vside = normalize(cross(vhat, [1.0, 0.0, 0.0]));
        let bound = 16.0 * f64::EPSILON;
        let mut worst = 0.0_f64;
        for angle in [3e-9, 1e-8, 1e-6, 1e-3] {
            let sun_near_nadir = [
                1.5e11 * (nadir[0] + angle * side[0]),
                1.5e11 * (nadir[1] + angle * side[1]),
                1.5e11 * (nadir[2] + angle * side[2]),
            ];
            let yaw = YawSteering.inertial_to_body(0.0, position, velocity, sun_near_nadir);
            let sun_near_velocity = [
                1.5e11 * (vhat[0] + angle * vside[0]),
                1.5e11 * (vhat[1] + angle * vside[1]),
                1.5e11 * (vhat[2] + angle * vside[2]),
            ];
            let pointing = SunPointing.inertial_to_body(0.0, position, velocity, sun_near_velocity);
            for (name, m) in [("yaw", yaw), ("sun", pointing)] {
                let (off, det) = orthonormality(&m);
                let miss = off.max((det - 1.0).abs());
                println!("{name} at {angle:e} rad: {miss:.3e}");
                worst = worst.max(miss);
            }
        }
        assert!(
            worst <= bound,
            "orthonormality lost next to the degeneracy: {worst:e}"
        );
    }

    /// (G4) The spin law is a one-parameter group over its own epoch:
    /// `M(t₀ + t₁ + t₂) = M(t₀ + t₁)·baseᵀ·M(t₀ + t₂)`, and the inertial spin axis has the constant
    /// body representation `base·â` — `derivations/attitude_law_frames.mac` (`spin_composition`,
    /// `spin_fixes_the_inertial_axis`) and `.gp`. A non-trivial base and axis, so that neither
    /// identity holds by a coincidence of coordinates; the bound is a few ulps of the products.
    #[test]
    fn the_spin_law_composes_as_a_one_parameter_group() {
        let base = exp_so3([0.3, -1.1, 0.7]);
        let axis = [1.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0];
        let (t0, rate) = (3.0, 0.5);
        let law = SpinStabilized::new(base, t0, axis, rate);
        let at = |t: f64| law.inertial_to_body(t, [0.0; 3], [0.0; 3], [0.0; 3]);
        let transpose = |m: &Mat3| {
            let mut t = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    t[i][j] = m[j][i];
                }
            }
            t
        };
        for (t1, t2) in [(1.25, 4.5), (-2.0, 0.375), (10.0, -7.5)] {
            let whole = at(t0 + t1 + t2);
            let composed = matmul(&matmul(&at(t0 + t1), &transpose(&base)), &at(t0 + t2));
            let miss = max_abs_diff(&whole, &composed);
            assert!(miss < 64.0 * f64::EPSILON, "t1 {t1}, t2 {t2}: {miss:e}");
            let fixed = apply(&at(t0 + t1), axis);
            let want = apply(&base, axis);
            for k in 0..3 {
                assert!(
                    (fixed[k] - want[k]).abs() < 16.0 * f64::EPSILON,
                    "axis moved at t1 {t1}"
                );
            }
        }
    }

    /// [`normalize`]'s degenerate contract: a vector whose norm underflows comes back unchanged.
    ///
    /// Every caller hands it a position, velocity or Sun vector of order `1e6`, so the guard is never
    /// taken and weakening `n < 1e-300` to `n == 1e-300` is invisible — yet under that mutant a zero
    /// vector is scaled by `1/0 = ∞`, every component becomes NaN, and the NaN then travels silently
    /// into the frame. `to_bits` rather than `==`: unchanged here means bit-for-bit.
    #[test]
    fn normalize_leaves_a_vector_whose_norm_underflows_alone() {
        // `norm` squares before it adds, and `x²` underflows to zero below |x| ≈ 2.2e-162, so both of
        // these have `norm(a) == 0.0` exactly.
        for v in [[0.0; 3], [1e-200, -2e-200, 3e-200]] {
            let got = normalize(v);
            for k in 0..3 {
                assert_eq!(
                    got[k].to_bits(),
                    v[k].to_bits(),
                    "normalize({v:?}) changed component {k}: {got:?}"
                );
            }
        }
    }

    /// The LVLH cross-track axis is the *negative* orbit normal, as [`LofType::Lvlh`] documents.
    ///
    /// `every_local_frame_is_orthonormal_and_right_handed` cannot see the sign: flipping `ŷ` flips
    /// `x̂ = ŷ×ẑ` with it and the frame stays orthonormal and right-handed, while
    /// `lvlh_body_z_points_to_nadir` constrains only `ẑ`. So deleting the `-` in `ŷ = −ĥ` survives
    /// both. The oracle is `ĥ` for [`sample_state`] evaluated outside this crate, at full precision.
    #[test]
    fn the_lvlh_cross_track_axis_is_the_negative_orbit_normal() {
        const HHAT: Vec3 = [
            0.092_615_512_269_831_07,
            -0.156_658_153_733_012_15,
            0.983_300_864_311_610_8,
        ];
        let (r, v) = sample_state();
        let m = LocalOrbitalFrame::nadir().inertial_to_body(0.0, r, v, [0.0; 3]);
        for k in 0..3 {
            assert!(
                (m[1][k] + HHAT[k]).abs() < 1e-15,
                "LVLH ŷ = {:?} should be −ĥ = {:?}",
                m[1],
                scale(HHAT, -1.0)
            );
        }
        // The along-track axis follows from it: x̂ = ŷ×ẑ ≈ +v̂ on a near-circular orbit.
        let along = dot(m[0], normalize(v));
        assert!(along > 0.9, "x̂·v̂ = {along}");
    }

    /// [`YawSteering`]'s Sun-parallel fallback fires strictly below its threshold, not at it.
    ///
    /// The threshold is on `‖s⊥‖`, the Sun's projection into the local horizontal plane, and no other
    /// fixture puts the Sun anywhere near nadir, so `< 1e-9` weakened to `== 1e-9` or widened to
    /// `<= 1e-9` is unobservable. Two fixtures separate all three: an exactly degenerate Sun
    /// (`‖s⊥‖ = 0`), which must fall back, and one sitting exactly *on* the threshold, which must not.
    ///
    /// The boundary fixture is exact in `f64`, not approximate — measured, not assumed: `R = 2²³`
    /// makes `‖r‖` and `r̂` exact, `1e-18 + 1.0` rounds to `1.0` so `normalize(sun)` returns the Sun
    /// unchanged, and `sqrt(1e-9 · 1e-9) == 1e-9` bit-for-bit.
    #[test]
    fn yaw_steering_falls_back_only_strictly_below_its_threshold() {
        const R: f64 = 8_388_608.0; // 2²³ m
        let position = [0.0, 0.0, R];
        let velocity = [8_192.0, 0.0, 0.0];
        let lvlh = LocalOrbitalFrame::nadir().matrix(position, velocity);

        // Sun exactly along nadir: ‖s⊥‖ = 0, the roll is undefined, the law must be LVLH.
        let degenerate = YawSteering.inertial_to_body(0.0, position, velocity, [0.0, 0.0, 1.0]);
        assert!(
            max_abs_diff(&degenerate, &lvlh) < 1e-15,
            "a Sun along nadir must fall back to LVLH: {degenerate:?}"
        );

        // ‖s⊥‖ exactly the threshold: the fallback must not fire, so body x̂ is the Sun's horizontal
        // projection (+ŷ inertial) rather than LVLH's x̂ ≈ v̂ (+x̂ inertial).
        let on_threshold = YawSteering.inertial_to_body(0.0, position, velocity, [0.0, 1e-9, 1.0]);
        let want = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        assert!(
            max_abs_diff(&on_threshold, &want) < 1e-12,
            "on the threshold the law must steer, not fall back: {on_threshold:?}"
        );
    }

    /// [`SunPointing`]'s phasing fallback fires strictly below its threshold, not at it.
    ///
    /// Same shape as [`yaw_steering_falls_back_only_strictly_below_its_threshold`], on `‖ẑ×v̂‖`: three
    /// mutants of `< 1e-9` survived — `==`, `>` and `<=` — because every other fixture has the Sun
    /// well away from the velocity, where the guard is false by a factor of `1e9`. `SPEED = 2¹³` keeps
    /// `normalize(velocity)` exact, so the boundary case lands on `1e-9` bit-for-bit.
    #[test]
    fn sun_pointing_uses_the_phasing_fallback_only_strictly_below_its_threshold() {
        const SPEED: f64 = 8_192.0; // 2¹³ m/s
        let position = [0.0, 8_388_608.0, 0.0];
        let sun = [0.0, 0.0, 1.0];
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let phased = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];

        // ẑ ⟂ v̂: ‖ẑ×v̂‖ = 1, nowhere near the threshold, so the velocity resolves the roll.
        let ordinary = SunPointing.inertial_to_body(0.0, position, [SPEED, 0.0, 0.0], sun);
        assert!(
            max_abs_diff(&ordinary, &identity) < 1e-12,
            "the velocity should resolve the roll here: {ordinary:?}"
        );

        // Velocity exactly along the Sun: ‖ẑ×v̂‖ = 0, the roll is undefined, position takes over.
        let degenerate = SunPointing.inertial_to_body(0.0, position, [0.0, 0.0, SPEED], sun);
        let (off, det) = orthonormality(&degenerate);
        assert!(
            off < 1e-12 && (det - 1.0).abs() < 1e-12,
            "the fallback frame is degenerate: off {off:e}, det {det}"
        );
        assert!(
            max_abs_diff(&degenerate, &phased) < 1e-12,
            "the phasing reference should be the position here: {degenerate:?}"
        );

        // ‖ẑ×v̂‖ exactly the threshold: the fallback must not fire.
        let on_threshold =
            SunPointing.inertial_to_body(0.0, position, [1e-9 * SPEED, 0.0, SPEED], sun);
        assert!(
            max_abs_diff(&on_threshold, &identity) < 1e-12,
            "on the threshold the velocity still resolves the roll: {on_threshold:?}"
        );
    }

    /// The spin angle is measured from the law's own epoch, not from zero.
    ///
    /// `spin_fixes_the_inertial_axis_and_is_periodic` builds its law with `t0 = 0`, where `t − t0` and
    /// `t + t0` are the same number, so the sign of the epoch offset is unobservable. A non-zero epoch
    /// separates them: at `t = t0 = 3 s` and `rate = 0.5 rad/s` the mutant is 3 rad from the base.
    #[test]
    fn the_spin_angle_is_measured_from_the_laws_own_epoch() {
        let base = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let rate = 0.5;
        let t0 = 3.0;
        let law = SpinStabilized::new(base, t0, [0.0, 0.0, 1.0], rate);
        let ignore = [0.0; 3];

        let m0 = law.inertial_to_body(t0, ignore, ignore, ignore);
        assert!(
            max_abs_diff(&m0, &base) < 1e-15,
            "at its own epoch the attitude must be the base: {m0:?}"
        );

        // A quarter turn *after the epoch*, not after zero.
        let mq = law.inertial_to_body(t0 + (PI / 2.0) / rate, ignore, ignore, ignore);
        let x_body = apply(&mq, [1.0, 0.0, 0.0]);
        assert!(
            x_body[0].abs() < 1e-9 && (x_body[1] + 1.0).abs() < 1e-9,
            "{x_body:?}"
        );
    }
}
