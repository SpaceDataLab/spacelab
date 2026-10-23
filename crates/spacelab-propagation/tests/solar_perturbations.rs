//! Third-body and solar-radiation-pressure forces, on a fixed test Sun so the geometry is exact.
//!
//! * **Battin ≡ direct**: the cancellation-free third-body form equals the textbook
//!   difference-of-two-terms, evaluated in double-double so that the comparison is limited by f64
//!   round-off rather than by the cancellation that motivates the rewrite in the first place.
//! * **Tidal limit**: far from the body the third-body acceleration is the tidal field
//!   `(μ/d³)(3(r·ŝ)ŝ − r)`.
//! * **The SRP vector**: `−a₀(AU/‖d‖)²d̂` in every component, at a body off all three axes.
//! * **SRP direction / shadow / inverse-square**: the force points anti-sunward, vanishes in the
//!   cylindrical shadow, and falls as `1/d²`.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::many_single_char_names
)]

use spacelab_autodiff::{DoubleDouble, Scalar};
use spacelab_propagation::ephemeris::{ASTRONOMICAL_UNIT, AnalyticMoon, Ephemeris};
use spacelab_propagation::forces::{Acceleration, SolarRadiationPressure, ThirdBody};

const MU_SUN: f64 = 1.327_124_400_18e20;
const MU_MOON: f64 = 4.902_800_066e12;
const RE: f64 = 6_378_137.0;

/// A body pinned in space, so the geometry under test is exactly what we set.
struct Fixed([f64; 3]);
impl Ephemeris for Fixed {
    fn position(&self, _t: f64) -> [f64; 3] {
        self.0
    }
}

fn norm(a: [f64; 3]) -> f64 {
    (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// `a = μ[(s−r)/‖s−r‖³ − s/‖s‖³]`, the textbook difference, evaluated in double-double.
///
/// The difference is why [`ThirdBody`] does not use this form: the two terms agree to `r/s ≈ 5e-5`,
/// so an f64 evaluation gives up four or five digits to cancellation and can only confirm the
/// cancellation-free rewrite to about `1e-8`. That is not sharp enough to see the `q²` term in
/// Battin's series, which moves the answer by `8e-9`. In double-double the same subtraction has
/// twenty-seven digits left afterwards, which makes this an exact oracle for an f64 result.
fn direct_third_body(mu: f64, s: [f64; 3], r: [f64; 3]) -> [f64; 3] {
    let lift = |v: [f64; 3]| v.map(DoubleDouble::from_f64);
    let (s, r) = (lift(s), lift(r));
    let sr = [s[0] - r[0], s[1] - r[1], s[2] - r[2]];
    let inv_cube = |v: [DoubleDouble; 3]| {
        let n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        (n2 * n2.sqrt()).recip()
    };
    let (to_body, to_origin) = (inv_cube(sr), inv_cube(s));
    let mu = DoubleDouble::from_f64(mu);
    [0, 1, 2].map(|i| (mu * (sr[i] * to_body - s[i] * to_origin)).value())
}

#[test]
fn the_battin_form_matches_the_direct_difference() {
    // The body is off every axis and off the equator. On the `+x` axis — where a test Sun naturally
    // lands — `s[1]` and `s[2]` are exact zeros, and every product and sum that touches them is a
    // term that cannot be wrong: two thirds of the two dot products and two thirds of the final
    // `r + f·s` go unasserted. Measured, that hides ten of this function's arithmetic mutations.
    let dir = [2.0, -3.0, 1.5];
    let scale = ASTRONOMICAL_UNIT / norm(dir);
    let s = [dir[0] * scale, dir[1] * scale, dir[2] * scale];
    let r = [6.8e6, 0.9e6, 2.1e6];

    let direct = direct_third_body(MU_SUN, s, r);
    let battin = ThirdBody {
        mu: MU_SUN,
        ephemeris: Fixed(s),
    }
    .acceleration::<f64>(0.0, &r, &[0.0; 3]);
    let rel = norm([
        battin[0] - direct[0],
        battin[1] - direct[1],
        battin[2] - direct[2],
    ]) / norm(direct);
    println!("Battin vs direct (double-double): {rel:.2e} relative");
    assert!(
        rel < 1e-14,
        "Battin third-body form disagrees with the direct one: {rel:.2e}"
    );
}

#[test]
fn third_body_reduces_to_the_tidal_field() {
    // Far from the body, a ≈ (μ/d³)(3(r·ŝ)ŝ − r), ŝ the unit body direction.
    let s = [ASTRONOMICAL_UNIT, 0.0, 0.0];
    let r = [7.0e6, 1.0e6, 0.5e6];
    let a = ThirdBody {
        mu: MU_SUN,
        ephemeris: Fixed(s),
    }
    .acceleration::<f64>(0.0, &r, &[0.0; 3]);

    let d = norm(s);
    let s_hat = [s[0] / d, s[1] / d, s[2] / d];
    let r_dot_s = r[0] * s_hat[0] + r[1] * s_hat[1] + r[2] * s_hat[2];
    let coeff = MU_SUN / d.powi(3);
    let tidal = [
        coeff * (3.0 * r_dot_s * s_hat[0] - r[0]),
        coeff * (3.0 * r_dot_s * s_hat[1] - r[1]),
        coeff * (3.0 * r_dot_s * s_hat[2] - r[2]),
    ];

    let rel = norm([a[0] - tidal[0], a[1] - tidal[1], a[2] - tidal[2]]) / norm(tidal);
    println!(
        "third-body vs tidal: {rel:.2e} relative; |a| = {:.3e} m/s²",
        norm(a)
    );
    assert!(
        rel < 1e-4,
        "third-body is not the tidal field to leading order: {rel:.2e}"
    );
    // physical scale: the Sun on a LEO satellite is a few ×10⁻⁷ m/s².
    assert!(
        (1e-7..1e-6).contains(&norm(a)),
        "third-body magnitude {:.3e} off scale",
        norm(a)
    );
}

fn srp(sun: [f64; 3]) -> SolarRadiationPressure<Fixed> {
    SolarRadiationPressure {
        reference_acceleration: 1.368e-7, // Φ·C_r·A/m for C_r=1.5, A/m=0.02
        astronomical_unit: ASTRONOMICAL_UNIT,
        earth_radius: RE,
        sun: Fixed(sun),
    }
}

#[test]
fn the_moon_third_body_has_the_right_magnitude_on_a_leo_satellite() {
    // With the real lunar ephemeris the Moon's tidal pull on a LEO satellite is ~1.1×10⁻⁶ m/s² —
    // about twice the Sun's, because it is so much closer despite its far smaller mass.
    let moon = ThirdBody {
        mu: MU_MOON,
        ephemeris: AnalyticMoon {
            epoch_j2000_seconds: 0.0,
        },
    };
    let a = moon.acceleration::<f64>(0.0, &[7.0e6, 0.0, 0.0], &[0.0; 3]);
    let mag = norm(a);
    println!("Moon third-body on LEO: {mag:.3e} m/s²");
    assert!(
        (5e-7..2e-6).contains(&mag),
        "Moon third-body magnitude {mag:.3e} off the ~10⁻⁶ scale"
    );
}

#[test]
fn the_srp_vector_is_the_anti_solar_inverse_square_law_at_a_general_geometry() {
    // The specification, arranged the other way round: magnitude `a₀(AU/‖d‖)²` along `−d̂`, against
    // the code's `a₀·AU²·‖d‖⁻³·d`. With the Sun on `+x` — the geometry of every other test here —
    // `d[1]` and `d[2]` are the satellite's own coordinates negated, five orders below `d[0]`, so
    // the whole transverse part of the calculation sits below any tolerance these tests assert;
    // and `sun[i] + r[i]` differs from `sun[i] − r[i]` by only `1e-4` relative, which the sign,
    // shadow and ratio checks are far too loose to notice. Measured: eight mutations survive them.
    let dir = [2.0, -3.0, 1.5];
    let scale = ASTRONOMICAL_UNIT / norm(dir);
    let sun = [dir[0] * scale, dir[1] * scale, dir[2] * scale];
    let r = [6.8e6, 0.9e6, 2.1e6]; // sunward of the Earth, so the lighting ratio is exactly 1

    let d = [sun[0] - r[0], sun[1] - r[1], sun[2] - r[2]];
    let dn = norm(d);
    let ratio = ASTRONOMICAL_UNIT / dn;
    let magnitude = 1.368e-7 * ratio * ratio;
    let want = [
        -magnitude * d[0] / dn,
        -magnitude * d[1] / dn,
        -magnitude * d[2] / dn,
    ];

    let got = srp(sun).acceleration::<f64>(0.0, &r, &[0.0; 3]);
    let rel = norm([got[0] - want[0], got[1] - want[1], got[2] - want[2]]) / magnitude;
    println!("SRP vs the definition: {rel:.2e} relative");
    assert!(rel < 1e-14, "SRP is not −a₀(AU/‖d‖)²d̂: {rel:.2e}");
}

#[test]
fn srp_pushes_anti_sunward_in_full_sun() {
    // Sun on +x, satellite on the day side: the force is along −x, away from the Sun.
    let a =
        srp([ASTRONOMICAL_UNIT, 0.0, 0.0]).acceleration::<f64>(0.0, &[7.0e6, 0.0, 0.0], &[0.0; 3]);
    assert!(a[0] < 0.0, "SRP should push away from the Sun (−x): {a:?}");
    assert!(
        a[1].abs() < 1e-20 && a[2].abs() < 1e-20,
        "no transverse SRP for an on-axis satellite: {a:?}"
    );
}

#[test]
fn srp_vanishes_in_the_cylindrical_shadow() {
    let sun = [ASTRONOMICAL_UNIT, 0.0, 0.0];
    // directly behind Earth on the Sun line, within one Earth radius → umbra → no force
    let dark = srp(sun).acceleration::<f64>(0.0, &[-7.0e6, 0.0, 0.0], &[0.0; 3]);
    assert!(norm(dark) == 0.0, "no SRP in shadow, got {dark:?}");

    // behind Earth but outside the shadow cylinder (perp > Rₑ) → still lit
    let grazing = srp(sun).acceleration::<f64>(0.0, &[-7.0e6, 8.0e6, 0.0], &[0.0; 3]);
    assert!(
        norm(grazing) > 0.0,
        "outside the shadow cylinder should be lit"
    );

    // day side → lit
    let lit = srp(sun).acceleration::<f64>(0.0, &[7.0e6, 0.0, 0.0], &[0.0; 3]);
    assert!(norm(lit) > 0.0, "day side should be lit");
}

#[test]
fn srp_follows_the_inverse_square_law() {
    // Same sub-satellite geometry, Sun at 1 AU then 2 AU: the force must quarter.
    let sat = [1000.0, 0.0, 0.0];
    let near = norm(srp([ASTRONOMICAL_UNIT, 0.0, 0.0]).acceleration::<f64>(0.0, &sat, &[0.0; 3]));
    let far =
        norm(srp([2.0 * ASTRONOMICAL_UNIT, 0.0, 0.0]).acceleration::<f64>(0.0, &sat, &[0.0; 3]));

    // near ≈ reference (satellite ~at 1 AU from the Sun), far ≈ reference/4
    assert!(
        (near - 1.368e-7).abs() < 1e-10,
        "SRP at 1 AU should be the reference: {near:.4e}"
    );
    assert!(
        (near / far - 4.0).abs() < 1e-3,
        "SRP must obey 1/d²: ratio {:.4}",
        near / far
    );
}

/// A body at the frame origin means *no body*, and both forces have to say so themselves.
///
/// [`Ephemeris::position`] returns a bare `[f64; 3]`. It cannot report "no data", so an
/// implementation that has none — `spacelab-ephemeris`'s `PerturbingBody` outside its kernel's span
/// is the one in the workspace — returns the origin, and
/// [`spacelab_propagation::ephemeris::is_absent`] is the whole of that contract.
///
/// Nothing in the arithmetic honours it. `ThirdBody` divides by `‖s‖³` and returns `NaN`, which at
/// least poisons the state visibly. `SolarRadiationPressure` is worse: with the Sun at the origin
/// the satellite-to-Sun vector is just `−r`, so the force comes out **finite** and scaled by
/// `(AU/‖r‖)² = 4.3e8` — 58.8 m/s² on the satellite below, six times surface gravity, from a force
/// whose reference value is 1.368e-7 m/s². A propagation would run to completion and report a
/// trajectory.
#[test]
// Exact, not near: the contract is that the force is never entered, so anything it could return
// is a bug however small. A tolerance here would pass on the NaN this test exists to close.
#[allow(clippy::float_cmp)]
fn a_body_at_the_origin_contributes_nothing() {
    let sat = [7_000e3, 1_500e3, -900e3];
    let absent = [0.0; 3];

    let third = ThirdBody {
        mu: MU_SUN,
        ephemeris: Fixed(absent),
    }
    .acceleration::<f64>(0.0, &sat, &[0.0; 3]);
    assert_eq!(third, [0.0; 3], "third body at the origin: {third:?}");

    let radiation = srp(absent).acceleration::<f64>(0.0, &sat, &[0.0; 3]);
    assert_eq!(radiation, [0.0; 3], "Sun at the origin: {radiation:?}");

    // And the sentinel is exactly the origin, not a neighbourhood of it: a body one metre out is a
    // physical input and still gets a force, however extreme.
    let near = srp([1.0, 0.0, 0.0]).acceleration::<f64>(0.0, &sat, &[0.0; 3]);
    assert!(
        norm(near) > 0.0 && norm(near).is_finite(),
        "a body one metre off the origin is not the sentinel: {near:?}"
    );
}
