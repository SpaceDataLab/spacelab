//! Conical umbra/penumbra shadow geometry for radiation forces — the smooth lighting ratio
//! `λ ∈ [0, 1]` (fraction of the solar disk visible) past a spherical occulting body.
//!
//! The numerical (Cowell) [`crate::forces::SolarRadiationPressure`] previously gated illumination with
//! a crude binary *cylindrical* Earth shadow (`ν ∈ {0, 1}`, no penumbra). This is the Orekit
//! `SolarRadiationPressure.getLightingRatio` / `OccultationEngine` model reduced to a sphere: exact in
//! the umbra (`0`) and full sun (`1`), and the lens-overlap area fraction of the two apparent disks in
//! the penumbra. It is `no_std` so both the numerical force and the `std`-only DSST mean-element path
//! can share one definition. `λ` is used as an f64 value gate — the penumbra boundary is a
//! measure-zero set the STM is not defined across, so it stays out of the Jacobian.
//!
//! `derivations/shadow_lens.mac` shows that Orekit's lens arrangement is the textbook
//! circle-intersection area for every pair of radii and separation, not only at the test's five
//! points. It also shows that the penumbra meets both plateaus continuously, exactly 0 and exactly 1.

// The penumbra area formula names paired Sun/Earth quantities (`a_s2`/`a_e2`, `alpha1`/`alpha2`)
// that clippy flags as too similar — the pairing is the whole point, so keep the Orekit names.
#![allow(clippy::similar_names)]

use libm::{acos, asin, sqrt};

/// Sun radius (m) — Orekit `Constants.SUN_RADIUS`.
pub const SUN_RADIUS: f64 = 6.957e8;

/// Angular margin for the umbra/penumbra transition — Orekit `SolarRadiationPressure.ANGULAR_MARGIN`.
const ANGULAR_MARGIN: f64 = 1.0e-10;

fn norm(v: [f64; 3]) -> f64 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Angle between two vectors, Orekit `Vector3D.angle` (robust near 0 and π via the cross product).
fn angle(u: [f64; 3], v: [f64; 3]) -> f64 {
    use core::f64::consts::PI;
    let norm_product = norm(u) * norm(v);
    let d = dot(u, v);
    let threshold = norm_product * 0.9999;
    if d < -threshold || d > threshold {
        let cross_norm = norm(cross(u, v));
        if d >= 0.0 {
            asin(cross_norm / norm_product)
        } else {
            PI - asin(cross_norm / norm_product)
        }
    } else {
        acos(d / norm_product)
    }
}

/// Fraction of the occulted (Sun) disk still visible, given the apparent angular `separation` of the
/// two disk centres, the occulting body's apparent (`alpha_central`) and the Sun's apparent
/// (`alpha_sun`) radii — Orekit `SolarRadiationPressure.maskingRatio`. `0` in full umbra, `1` in full
/// sun, the lens-overlap fraction in penumbra.
#[must_use]
pub fn masking_ratio(separation: f64, alpha_central: f64, alpha_sun: f64) -> f64 {
    use core::f64::consts::PI;
    if separation - alpha_central + alpha_sun <= ANGULAR_MARGIN {
        0.0
    } else if separation - alpha_central - alpha_sun < -ANGULAR_MARGIN {
        // Penumbra: overlap area of the two apparent disks.
        let s_ea2 = separation * separation;
        let oo2s_ea = 1.0 / (2.0 * separation);
        let a_s2 = alpha_sun * alpha_sun;
        let a_e2 = alpha_central * alpha_central;
        let a_e2m_a_s2 = a_e2 - a_s2;

        let alpha1 = (s_ea2 - a_e2m_a_s2) * oo2s_ea;
        let alpha2 = (s_ea2 + a_e2m_a_s2) * oo2s_ea;

        // Boundary protection (Orekit Precision.SAFE_MIN / nextDown(1)).
        let almost0 = f64::MIN_POSITIVE;
        let almost1 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF);
        let a1o_a_s = (alpha1 / alpha_sun).clamp(-almost1, almost1);
        let a_s2m_a12 = (a_s2 - alpha1 * alpha1).max(almost0);
        let a2o_a_e = (alpha2 / alpha_central).clamp(-almost1, almost1);
        let a_e2m_a22 = (a_e2 - alpha2 * alpha2).max(almost0);

        let p1 = a_s2 * acos(a1o_a_s) - alpha1 * sqrt(a_s2m_a12);
        let p2 = a_e2 * acos(a2o_a_e) - alpha2 * sqrt(a_e2m_a22);

        1.0 - (p1 + p2) / (PI * a_s2)
    } else {
        1.0
    }
}

/// Conical lighting ratio `λ ∈ [0, 1]` (Sun visibility) at geocentric position `pos` (m), for the Sun
/// at `sun_pos` (m) and a single spherical occulting body of radius `occulting_radius` (m, e.g. the
/// Earth). Orekit `SolarRadiationPressure.getLightingRatio` reduced to one sphere.
#[must_use]
pub fn lighting_ratio(pos: [f64; 3], sun_pos: [f64; 3], occulting_radius: f64) -> f64 {
    if norm(sun_pos) < 2.0 * SUN_RADIUS {
        return 1.0; // degenerate (Sun-centred frame); kept for faithfulness.
    }
    let sun_sat = [
        pos[0] - sun_pos[0],
        pos[1] - sun_pos[1],
        pos[2] - sun_pos[2],
    ];
    let separation = angle(sun_sat, pos);
    let occulted_apparent_radius = asin(SUN_RADIUS / norm(sun_sat));
    let limb_radius = asin(occulting_radius / norm(pos));
    masking_ratio(separation, limb_radius, occulted_apparent_radius)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;

    #[test]
    fn masking_ratio_umbra_penumbra_sun() {
        // Earth apparent radius 0.1 rad, Sun 0.01 rad.
        assert!(masking_ratio(0.0, 0.1, 0.01) < 1e-12); // Sun fully behind Earth
        assert!((masking_ratio(0.2, 0.1, 0.01) - 1.0).abs() < 1e-12); // well clear → full sun
        let p = masking_ratio(0.1, 0.1, 0.01); // Sun centre on Earth's limb → partial
        assert!(p > 0.0 && p < 1.0, "penumbra {p}");
        // Monotone: wider separation → more Sun visible.
        assert!(masking_ratio(0.105, 0.1, 0.01) > p);
    }

    /// Independent oracle: the textbook area of intersection of two circles (radii `small_r`, `big_r`,
    /// centre distance `d`), as the fraction of the occulted (Sun) disk *not* covered. This is a
    /// different algebraic arrangement of the same lens overlap than [`masking_ratio`]'s Orekit form,
    /// so agreement proves the port carries no transcription error.
    fn masking_ratio_reference(d: f64, big_r: f64, small_r: f64) -> f64 {
        use core::f64::consts::PI;
        if d >= big_r + small_r {
            return 1.0; // disjoint disks → fully lit
        }
        if d <= big_r - small_r {
            return 0.0; // Sun disk fully behind Earth disk → umbra
        }
        let t1 = small_r
            * small_r
            * acos((d * d + small_r * small_r - big_r * big_r) / (2.0 * d * small_r));
        let t2 =
            big_r * big_r * acos((d * d + big_r * big_r - small_r * small_r) / (2.0 * d * big_r));
        let t3 = 0.5
            * sqrt(
                (-d + small_r + big_r)
                    * (d + small_r - big_r)
                    * (d - small_r + big_r)
                    * (d + small_r + big_r),
            );
        1.0 - (t1 + t2 - t3) / (PI * small_r * small_r)
    }

    #[test]
    fn masking_ratio_matches_circle_intersection_oracle() {
        let (big_r, small_r) = (0.1, 0.01);
        for &d in &[0.091, 0.095, 0.10, 0.105, 0.109] {
            let got = masking_ratio(d, big_r, small_r);
            let want = masking_ratio_reference(d, big_r, small_r);
            assert!((got - want).abs() < 1e-12, "d={d}: got {got}, want {want}");
        }
    }

    #[test]
    fn lighting_ratio_full_sun_and_umbra() {
        let sun = [1.496e11, 0.0, 0.0];
        let ae = 6.378_137e6;
        // Sun-side satellite → fully lit.
        assert!((lighting_ratio([7.0e6, 0.0, 0.0], sun, ae) - 1.0).abs() < 1e-12);
        // Directly behind the Earth, just above the surface → deep umbra.
        assert!(lighting_ratio([-6.6e6, 0.0, 0.0], sun, ae) < 1e-9);
        // Far off the shadow axis → fully lit again.
        assert!((lighting_ratio([-6.6e6, 1.0e8, 0.0], sun, ae) - 1.0).abs() < 1e-12);
    }

    const EARTH_RADIUS: f64 = 6.378_137e6;

    /// An orthogonal pair with no zero component, both exact in binary64: `|p| = |q| = 7` and
    /// `p·q = 12 + 6 − 18 = 0`. Everything below is built in their plane, so no component of any
    /// input vanishes and no cross-product row is dead.
    const P: [f64; 3] = [2.0, 3.0, 6.0];
    const Q: [f64; 3] = [6.0, 2.0, -3.0];

    /// A vector of length `len` at exactly `theta` from [`P`], in the `P`–`Q` plane.
    fn at_angle(theta: f64, len: f64) -> [f64; 3] {
        let (c, s) = (libm::cos(theta), libm::sin(theta));
        let k = len / 7.0;
        [
            (c * P[0] + s * Q[0]) * k,
            (c * P[1] + s * Q[1]) * k,
            (c * P[2] + s * Q[2]) * k,
        ]
    }

    /// `norm`, `dot` and `cross` against their definitions on integers, where the expected value is
    /// checkable by inspection. Every call site in this module feeds them geometry symmetric about
    /// the shadow axis, and a zero component makes most of their arithmetic unobservable.
    #[test]
    fn the_vector_helpers_match_their_definitions() {
        assert_eq!(norm([2.0, 3.0, 6.0]), 7.0); // 4 + 9 + 36 = 49
        assert_eq!(dot([1.0, 2.0, 3.0], [4.0, 5.0, 7.0]), 35.0); // 4 + 10 + 21
        assert_eq!(dot([1.0, -2.0, 3.0], [4.0, 5.0, -7.0]), -27.0); // 4 − 10 − 21
        assert_eq!(cross([1.0, 2.0, 3.0], [4.0, 5.0, 7.0]), [-1.0, 5.0, -3.0]);
    }

    /// `angle` against the angle its inputs were *built* from, on both sides of the `0.9999`
    /// threshold that selects between the `asin` and `acos` arms. The module's own geometry is
    /// collinear, which is the one family where the two arms cannot be told apart; away from it
    /// each is accurate only in its own half, so a wrong threshold shows up as lost digits rather
    /// than as a wrong branch. `|u| = 7` and `|v| = 3` keep `norm_product` off 1.
    #[test]
    fn angle_recovers_the_angle_its_inputs_were_built_from() {
        use core::f64::consts::{FRAC_PI_2, PI};
        for &theta in &[
            1.0e-8,
            1.0e-4,
            0.3,
            1.0,
            FRAC_PI_2 - 1.0e-9,
            FRAC_PI_2,
            FRAC_PI_2 + 1.0e-9,
            2.0,
            PI - 1.0e-4,
            PI - 1.0e-8,
        ] {
            let got = angle(P, at_angle(theta, 3.0));
            assert!((got - theta).abs() < 1e-15, "theta {theta}: got {got}");
        }
    }

    /// Both plateaus are exact, not approximate. The lens formula is continuous into them but not
    /// exact there — at external tangency it returns `1 − 4.8e-7`, because `acos` is evaluated at a
    /// clamped argument — so `0` and `1` have to be asserted as equalities or the branch boundaries
    /// carry no information.
    #[test]
    fn masking_ratio_is_exact_on_both_plateaus() {
        assert_eq!(masking_ratio(0.05, 0.1, 0.01), 0.0); // Sun disk well inside the Earth disk
        assert_eq!(masking_ratio(0.089, 0.1, 0.01), 0.0); // just inside internal tangency
        assert_eq!(masking_ratio(0.11, 0.1, 0.01), 1.0); // external tangency
        assert_eq!(masking_ratio(0.2, 0.1, 0.01), 1.0);
    }

    /// The one input family where `< -ANGULAR_MARGIN` and `<= -ANGULAR_MARGIN` part company: the
    /// difference `separation − α_c − α_s` can only land *exactly* on `−1e-10` when the radii are
    /// themselves of that order, and then both subtractions are exact. Here the occulting disk has
    /// zero radius and blocks nothing, so full sun is the geometrically right answer; the inclusive
    /// form takes the lens branch instead and returns `1 − 4.7e-9`.
    #[test]
    fn a_zero_radius_occulter_at_the_angular_margin_blocks_nothing() {
        assert_eq!((1.0e-10 - 0.0) - 2.0e-10, -ANGULAR_MARGIN);
        assert_eq!(masking_ratio(1.0e-10, 0.0, 2.0e-10), 1.0);
    }

    /// The degenerate-Sun guard, whose threshold no other case reaches. Below it the geometry is
    /// meaningless and the answer is full sun even for a satellite the geometry would put in deep
    /// umbra; at the threshold itself, and above, the geometry is evaluated.
    #[test]
    fn lighting_ratio_short_circuits_a_sun_centred_frame() {
        let pos = [-6.6e6, 0.0, 0.0];
        assert_eq!(lighting_ratio(pos, [1.0e9, 0.0, 0.0], EARTH_RADIUS), 1.0);
        let at_threshold = [2.0 * SUN_RADIUS, 0.0, 0.0];
        assert!(lighting_ratio(pos, at_threshold, EARTH_RADIUS) < 1e-12);
        assert!(lighting_ratio(pos, [1.6e9, 0.0, 0.0], EARTH_RADIUS) < 1e-12);
    }

    /// A penumbra crossing with nothing on an axis, against the circle-intersection oracle at the
    /// separation and apparent radii the geometry was *constructed* from. Every case above lies on
    /// the shadow axis, where `λ` is pinned to a plateau and the entire vector path — the
    /// `pos − sun_pos` difference, `angle`, `cross` — can be wrong without moving it.
    ///
    /// The bound is 3e-9 rather than the 1e-12 of the oracle test above because `α_c/α_s ≈ 246`
    /// here instead of 10: a 60-digit evaluation of the same lens area puts *both* f64 arrangements
    /// about 5e-10 off the truth at the worst sample, in opposite directions. That is the
    /// cancellation in `s² ∓ (α_c² − α_s²)`, not a defect in either.
    #[test]
    fn lighting_ratio_crosses_the_penumbra_off_axis() {
        let d_sun = 1.496e11;
        let pos = at_angle(0.0, 7.0e6); // [2, 3, 6]·10⁶ m, |pos| = 7000 km exactly
        let alpha_c = asin(EARTH_RADIUS / 7.0e6);
        let alpha_s = asin(SUN_RADIUS / d_sun);
        for k in -4..=4 {
            let separation = alpha_c + f64::from(k) * 0.4 * alpha_s;
            let sun_sat = at_angle(separation, d_sun);
            let sun_pos = [
                pos[0] - sun_sat[0],
                pos[1] - sun_sat[1],
                pos[2] - sun_sat[2],
            ];
            let got = lighting_ratio(pos, sun_pos, EARTH_RADIUS);
            let want = masking_ratio_reference(separation, alpha_c, alpha_s);
            assert!((got - want).abs() < 3e-9, "k={k}: got {got}, want {want}");
        }
    }
}
