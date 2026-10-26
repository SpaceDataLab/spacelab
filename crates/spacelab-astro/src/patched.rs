//! Patched-conic building blocks: sphere of influence, planetocentric hyperbolic arcs, and B-plane
//! targeting.
//!
//! The patched-conic method stitches a heliocentric transfer (the
//! [`crate::transfer`](mod@crate::transfer) / Lambert leg) to planetocentric hyperbolae at each
//! planet's sphere of influence. This module supplies the
//! pieces: the SOI radius that bounds the patch, the hyperbola an excess velocity implies at a
//! planet, and the B-plane parameters that aim a flyby or arrival. See Curtis, *Orbital Mechanics
//! for Engineering Students*, §8, §11; Vallado, §12.
//!
//! Gate: `derivations/patched_conics.mac` derives the sphere of influence from Laplace's criterion,
//! the four hyperbolic-arc closed forms from vis-viva and the asymptote, the impact parameter
//! `|a|√(e²−1) = h/v∞`, and the coded `B` as the limit of the aiming offset along the incoming
//! asymptote. `derivations/patched_conics.gp` re-evaluates every reference value of the tests at 100
//! digits by other routes (energy and angular momentum for the arcs, the conic followed out to
//! `1e40` periapsis radii for the B-plane). The module tests
//! `the_sphere_of_influence_solves_laplace_s_criterion` (G2), `the_hyperbolic_arc_is_its_closed_form`
//! (G1), `the_impact_parameter_is_the_angular_momentum_over_the_excess_speed` (G1) and
//! `the_b_plane_is_the_aiming_plane_of_the_incoming_asymptote` (G3) hold the code to them.

#![allow(clippy::many_single_char_names, clippy::similar_names)]

use spacelab_core::Vec3;

/// The Laplace sphere-of-influence radius, `r = a·(m_planet/m_sun)^(2/5)`: inside it the planet's
/// gravity dominates and the heliocentric conic is patched to a planetocentric one. `a` is the
/// planet's orbital semi-major axis, in the same length unit as the result.
#[must_use]
pub fn sphere_of_influence(planet_semi_major_axis: f64, planet_to_sun_mass_ratio: f64) -> f64 {
    planet_semi_major_axis * libm::pow(planet_to_sun_mass_ratio, 0.4)
}

/// A planetocentric hyperbolic arc — a departure injection or an arrival flyby — from its
/// hyperbolic excess speed `v∞` and periapsis radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperbolicArc {
    /// Semi-major axis (negative for a hyperbola), length.
    pub semi_major_axis: f64,
    /// Eccentricity (`> 1`).
    pub eccentricity: f64,
    /// Speed at periapsis, length/time.
    pub periapsis_speed: f64,
    /// Asymptote turn angle `δ = 2·asin(1/e)`, radians.
    pub turn_angle: f64,
}

/// The hyperbola an excess speed `v_inf` implies at a planet of gravitational parameter `mu`, given
/// the periapsis radius `r_periapsis`.
#[must_use]
pub fn hyperbolic_arc(v_inf: f64, mu: f64, r_periapsis: f64) -> HyperbolicArc {
    let v2 = v_inf * v_inf;
    HyperbolicArc {
        semi_major_axis: -mu / v2,
        eccentricity: 1.0 + r_periapsis * v2 / mu,
        periapsis_speed: libm::sqrt(v2 + 2.0 * mu / r_periapsis),
        turn_angle: 2.0 * libm::asin(mu / (mu + r_periapsis * v2)), // = 2·asin(1/e)
    }
}

/// B-plane (aiming-plane) parameters of a hyperbolic arrival.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BPlane {
    /// `B·T̂` — the component along the B-plane's T axis (in the reference plane).
    pub b_dot_t: f64,
    /// `B·R̂` — the component along the B-plane's R axis (completing the right-handed frame).
    pub b_dot_r: f64,
    /// The impact parameter `|B| = |a|·√(e²−1)`.
    pub b_magnitude: f64,
}

/// The B-plane parameters of a planetocentric hyperbolic state `(r, v)` under `mu`, resolved against
/// a reference direction `reference_north` (e.g. the pole `ẑ`). Returns `None` if the state is not
/// hyperbolic (`e ≤ 1`) or the asymptote is parallel to the reference.
#[must_use]
pub fn b_plane(r: Vec3, v: Vec3, mu: f64, reference_north: Vec3) -> Option<BPlane> {
    let rn = r.norm();
    let v2 = v.dot(v);
    let h = r.cross(v);
    let h_hat = h.scale(1.0 / h.norm());
    let rv = r.dot(v);
    let c = v2 - mu / rn;
    let e_vec = Vec3::new(
        (c * r.x - rv * v.x) / mu,
        (c * r.y - rv * v.y) / mu,
        (c * r.z - rv * v.z) / mu,
    );
    let e = e_vec.norm();
    if e <= 1.0 {
        return None; // not a hyperbola — no asymptote, no B-plane
    }
    let e_hat = e_vec.scale(1.0 / e);
    let a = 1.0 / (2.0 / rn - v2 / mu); // negative
    let sqe = libm::sqrt(e * e - 1.0);

    // Incoming asymptote unit vector Ŝ = (ê + √(e²−1)·(ĥ×ê))/e — the direction of travel far before
    // periapsis, the limit of v̂ at the true anomaly `−arccos(−1/e)`, so `Ŝ·ê = +1/e`. Negating the
    // `ê` term gives the *outgoing* asymptote, `Ŝ·ê = −1/e`, and with it the departure aiming plane.
    let hxe = h_hat.cross(e_hat);
    let s = Vec3::new(
        (e_hat.x + sqe * hxe.x) / e,
        (e_hat.y + sqe * hxe.y) / e,
        (e_hat.z + sqe * hxe.z) / e,
    );
    // B = |a|·√(e²−1)·(Ŝ × ĥ) — perpendicular to Ŝ, in the orbit plane, of length the impact parameter.
    let b_mag = a.abs() * sqe;
    let b = s.cross(h_hat).scale(b_mag);
    // B-plane axes: T̂ = (Ŝ × N̂)/|·|, R̂ = Ŝ × T̂.
    let sxn = s.cross(reference_north);
    let sxn_norm = sxn.norm();
    if sxn_norm < 1e-12 {
        return None; // asymptote parallel to the reference; T is undefined
    }
    let t_hat = sxn.scale(1.0 / sxn_norm);
    let r_hat = s.cross(t_hat);
    Some(BPlane {
        b_dot_t: b.dot(t_hat),
        b_dot_r: b.dot(r_hat),
        b_magnitude: b_mag,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision, clippy::expect_used)]
    use super::*;

    /// (G2) The radius returned is the root of Laplace's criterion `M² r⁵ = m² a⁵`, the equation
    /// `derivations/patched_conics.mac` solves: `(r/a)⁵ = (m/M)²`. `pow` is correctly rounded to
    /// within an ulp or so and the fifth power multiplies its error by five, so 16 ulps.
    #[test]
    fn the_sphere_of_influence_solves_laplace_s_criterion() {
        let au = 1.495_978_707e8;
        for (a, ratio) in [
            (au, 3.003_5e-6),
            (1.524 * au, 3.227e-7),
            (5.2 * au, 9.547_9e-4),
            (1.0, 1e-12),
        ] {
            let r = sphere_of_influence(a, ratio);
            let lhs = (r / a).powi(5);
            let rhs = ratio * ratio;
            assert!(
                (lhs / rhs - 1.0).abs() <= 16.0 * f64::EPSILON,
                "a {a}, m/M {ratio}: (r/a)^5 = {lhs:e}, (m/M)^2 = {rhs:e}"
            );
        }
    }

    /// (G1) The impact parameter the B-plane reports is the angular momentum over the excess speed,
    /// `|B| = |a|√(e²−1) = |r × v|/v∞` with `v∞² = v² − 2µ/r` (`derivations/patched_conics.mac`).
    /// Neither side cancels on these states, so 64 ulps.
    #[test]
    fn the_impact_parameter_is_the_angular_momentum_over_the_excess_speed() {
        for &(r, v, mu, north, ..) in &AIMING {
            let bp = b_plane(r, v, mu, north).expect("hyperbolic");
            let v_inf = (v.dot(v) - 2.0 * mu / r.norm()).sqrt();
            let want = r.cross(v).norm() / v_inf;
            assert!(
                (bp.b_magnitude - want).abs() <= 64.0 * f64::EPSILON * want,
                "|B| {:.17e}, h/v_inf {want:.17e}",
                bp.b_magnitude
            );
        }
    }

    /// The Laplace SOI reproduces the published Earth and Mars values.
    #[test]
    fn sphere_of_influence_matches_earth_and_mars() {
        let au = 1.495_978_707e8; // km
        let earth = sphere_of_influence(au, 3.003_5e-6);
        let mars = sphere_of_influence(1.524 * au, 3.227e-7);
        assert!(
            (earth - 924_000.0).abs() < 15_000.0,
            "Earth SOI {earth:.0} km"
        );
        assert!((mars - 577_000.0).abs() < 15_000.0, "Mars SOI {mars:.0} km");
    }

    /// `(v_inf, mu, r_periapsis, a, e, v_periapsis, turn angle)` at 60 decimal digits, from the
    /// defining relations rather than the shipped expressions: `a = −μ/v_∞²`, `e = 1 + r_p v_∞²/μ`,
    /// `v_p = √(v_∞² + 2μ/r_p)` (vis-viva at periapsis) and `δ = 2 asin(1/e)`, cross-checked against
    /// the `2 asin(μ/(μ + r_p v_∞²))` form. Earth departure, Mars capture, a Jupiter flyby.
    const ARCS: [(f64, f64, f64, f64, f64, f64, f64); 3] = [
        (
            3.0,
            398_600.0,
            6578.0,
            -44_288.888_888_888_888_888_888_89,
            1.148_524_836_929_252_383_341_696,
            11.410_164_399_632_208_425_049_17,
            2.113_174_817_088_339_117_015_914,
        ),
        (
            1.35,
            42_828.37,
            3689.5,
            -23_499.791_495_198_902_606_310_01,
            1.157_001_393_001_881_696_641_735,
            5.003_883_943_456_582_533_791_297,
            2.087_526_679_168_519_459_675_233,
        ),
        (
            5.72,
            126_686_534.0,
            78_240.0,
            -3_872_027.177_368_086_458_995_550,
            1.020_206_469_742_080_085_638_778,
            57.193_764_811_676_222_905_306_97,
            2.742_874_867_959_808_662_240_267,
        ),
    ];

    /// Two planetocentric hyperbolic states and the aiming plane each implies, at 60 decimal digits.
    /// The B-plane was not evaluated from the shipped closed form but from its definition: propagate
    /// the hyperbola back onto the incoming asymptote, take `Ŝ = v̂` and `B = r − (r·Ŝ)Ŝ` there, and
    /// resolve against `T̂ = (Ŝ × N̂)/|·|`, `R̂ = Ŝ × T̂`. `derivations/bplane_asymptote.py` runs it and
    /// asserts that the closed form and the propagated limit agree to 1e-12 relative on both cases.
    const AIMING: [(Vec3, Vec3, f64, Vec3, f64, f64, f64); 2] = [
        (
            Vec3 {
                x: 10_000.0,
                y: 2000.0,
                z: 500.0,
            },
            Vec3 {
                x: 1.0,
                y: 11.0,
                z: 0.5,
            },
            398_600.0,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            16_278.021_635_272_253_192_433_82,
            -46.745_618_375_512_104_123_958_08,
            16_278.088_754_863_971_098_252_41,
        ),
        (
            Vec3 {
                x: -8200.0,
                y: 14_300.0,
                z: -3100.0,
            },
            Vec3 {
                x: 5.7,
                y: -1.9,
                z: 4.2,
            },
            42_828.37,
            Vec3 {
                x: 0.4,
                y: -0.6,
                z: 1.86,
            },
            -12_469.139_078_476_552_364_986_09,
            -137.460_250_834_991_876_988_805_2,
            12_469.896_738_904_885_751_362_11,
        ),
    ];

    /// The incoming asymptote of the first `AIMING` case, from the same derivation. Used as a
    /// reference direction it leaves `T̂` undefined, which is the one input `b_plane` must refuse.
    const ASYMPTOTE: Vec3 = Vec3 {
        x: 0.670_583_187_526_118_178_729_1,
        y: 0.739_504_022_388_586_807_071_0,
        z: 0.058_753_633_746_446_041_075_0,
    };

    /// (G1) Every field of the arc is a closed form in `(v_∞, μ, r_p)`, and every one of them was
    /// free: the sign and range checks below hold for a wide family of wrong expressions. The four
    /// closed forms are derived in `derivations/patched_conics.mac`; `derivations/patched_conics.gp`
    /// recomputes `ARCS` at 100 digits from the energy and angular momentum instead and agrees to the
    /// 25 digits written.
    #[test]
    fn the_hyperbolic_arc_is_its_closed_form() {
        for &(v_inf, mu, r_p, a, e, v_p, turn) in &ARCS {
            let arc = hyperbolic_arc(v_inf, mu, r_p);
            for (name, got, want) in [
                ("semi-major axis", arc.semi_major_axis, a),
                ("eccentricity", arc.eccentricity, e),
                ("periapsis speed", arc.periapsis_speed, v_p),
                ("turn angle", arc.turn_angle, turn),
            ] {
                assert!(
                    (got - want).abs() <= 1e-14 * want.abs(),
                    "v_inf {v_inf}, mu {mu}: {name} {got:.17e}, want {want:.17e}"
                );
            }
        }
    }

    /// `|B|² = (B·T)² + (B·R)²` holds for *any* unit vector in the orbit plane taken as Ŝ, because
    /// `B` is built as `|B|(Ŝ × ĥ)` and `T̂`, `R̂` span the plane perpendicular to Ŝ by construction.
    /// So the identity cannot tell the arrival aiming plane from the departure one, and the values
    /// have to be pinned against the asymptote itself.
    ///
    /// (G3) `derivations/patched_conics.mac` shows the coded `B` is the limit of the aiming offset
    /// `r − (r·Ŝ)Ŝ` as the conic runs out along its incoming asymptote, and
    /// `derivations/patched_conics.gp` takes that limit at 100 digits on these states, matching `AIMING`
    /// to the 25 digits written; rounding the decimal inputs to doubles moves it by 3e-17 of `|B|`.
    #[test]
    fn the_b_plane_is_the_aiming_plane_of_the_incoming_asymptote() {
        for &(r, v, mu, north, b_dot_t, b_dot_r, b_magnitude) in &AIMING {
            let bp = b_plane(r, v, mu, north).expect("a hyperbolic state has an aiming plane");
            for (name, got, want) in [
                ("B·T", bp.b_dot_t, b_dot_t),
                ("B·R", bp.b_dot_r, b_dot_r),
                ("|B|", bp.b_magnitude, b_magnitude),
            ] {
                assert!(
                    (got - want).abs() <= 1e-12 * b_magnitude,
                    "{name} {got:.17e}, want {want:.17e}"
                );
            }
        }
    }

    /// A reference direction along the asymptote leaves `T̂ = (Ŝ × N̂)/|·|` a division by zero. The
    /// function must refuse it rather than return infinities.
    #[test]
    fn a_reference_along_the_asymptote_has_no_aiming_plane() {
        let (r, v, mu, ..) = AIMING[0];
        assert!(
            b_plane(r, v, mu, ASYMPTOTE).is_none(),
            "an undefined T axis was answered instead of refused"
        );
        assert!(
            b_plane(r, v, mu, ASYMPTOTE.scale(-2.5)).is_none(),
            "an antiparallel reference was answered instead of refused"
        );
    }

    /// A hyperbolic departure from LEO: eccentricity > 1, periapsis speed exceeds escape, turn angle
    /// in (0, π).
    #[test]
    fn hyperbolic_arc_is_physical() {
        let mu = 398_600.0; // km³/s²
        let r_p = 6_578.0; // 200 km altitude
        let arc = hyperbolic_arc(3.0, mu, r_p);
        assert!(arc.eccentricity > 1.0, "e = {}", arc.eccentricity);
        assert!(arc.semi_major_axis < 0.0, "a should be negative");
        assert!(
            arc.periapsis_speed > (2.0 * mu / r_p).sqrt(),
            "v_p below escape"
        );
        assert!(arc.turn_angle > 0.0 && arc.turn_angle < core::f64::consts::PI);
    }

    /// The B-plane is geometrically consistent: `B ⟂ Ŝ`, so `|B|² = (B·T)² + (B·R)²`.
    #[test]
    fn b_plane_is_self_consistent() {
        let mu = 398_600.0;
        // A clearly hyperbolic planetocentric state (v well above escape at r).
        let r = Vec3::new(10_000.0, 2_000.0, 500.0);
        let v = Vec3::new(1.0, 11.0, 0.5);
        let bp = b_plane(r, v, mu, Vec3::new(0.0, 0.0, 1.0)).expect("hyperbolic");
        let recovered = (bp.b_dot_t * bp.b_dot_t + bp.b_dot_r * bp.b_dot_r).sqrt();
        assert!(
            (recovered - bp.b_magnitude).abs() / bp.b_magnitude < 1e-9,
            "B not in the T-R plane"
        );
        // An elliptic state has no B-plane.
        assert!(b_plane(r, Vec3::new(0.0, 5.0, 0.0), mu, Vec3::new(0.0, 0.0, 1.0)).is_none());
    }
}
