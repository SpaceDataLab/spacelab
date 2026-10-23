//! Orbit design: solving for the elements that meet a mission constraint, rather than describing an
//! orbit already flown. The inverse direction to the rest of the crate — and the reason the verified
//! `J₂` secular rates ([`KeplerianElements::nodal_regression_rate`]) matter beyond analysis.
//!
//! [`KeplerianElements::nodal_regression_rate`]: crate::KeplerianElements::nodal_regression_rate

// The design formulae are single letters — n (mean motion), a, e, i, p, q — as every reference
// writes them. The repeat solver's eight arguments are each a distinct, named physical quantity (the
// resonance, the orbit shape, the gravity field μ/Rₑ/J₂, Earth rotation), not a grab-bag of flags; a
// `GeopotentialField` parameter object is the refactor if this surface grows.
#![allow(clippy::many_single_char_names, clippy::too_many_arguments)]

use spacelab_core::Angle;

/// The nodal precession rate that holds an orbit plane fixed relative to the mean Sun: one turn per
/// tropical year, `2π / (365.2422 · 86400 s)`. An orbit whose node regresses at exactly this rate
/// keeps a constant mean local solar time at the ascending node — the defining property of a
/// Sun-synchronous orbit, and why nearly every Earth-observation satellite flies one.
///
/// Computed from the tropical year, not written as a truncated decimal.
pub const SUN_SYNCHRONOUS_NODE_RATE: f64 = core::f64::consts::TAU / (365.2422 * 86_400.0);

/// The (prograde) critical inclination `i = acos(1/√5) ≈ 63.43°`, where the first-order apsidal
/// precession `ω̇ ∝ 5cos²i − 1` vanishes and the argument of perigee is frozen. Molniya and Tundra
/// orbits fly it to hold apogee over one hemisphere; its retrograde complement is `180° − i`.
///
/// A closed form, `acos(1/√5)`, not a tabled angle.
#[must_use]
pub fn critical_inclination() -> Angle {
    Angle::from_radians(libm::acos(1.0 / libm::sqrt(5.0)))
}

/// The inclination whose first-order nodal regression equals `node_rate` — Sun-synchronous when
/// `node_rate` is [`SUN_SYNCHRONOUS_NODE_RATE`].
///
/// Inverts `Ω̇ = −³⁄₂ n J₂ (Rₑ/p)² cos i` for `i`, with `n = √(μ/a³)` and `p = a(1 − e²)`. Returns
/// `None` when the required `cos i` exceeds unity in magnitude: the orbit is then too high — its mean
/// motion too slow — for `J₂` to precess the node fast enough, and no inclination is Sun-synchronous.
/// For the eastward solar rate the solution is retrograde, `i > 90°` (≈ 98° in low orbit).
#[must_use]
pub fn sun_synchronous_inclination(
    semi_major_axis: f64,
    eccentricity: f64,
    mu: f64,
    re: f64,
    j2: f64,
    node_rate: f64,
) -> Option<Angle> {
    let n = libm::sqrt(mu / (semi_major_axis * semi_major_axis * semi_major_axis));
    let p = semi_major_axis * (1.0 - eccentricity * eccentricity);
    let factor = 1.5 * n * j2 * (re / p) * (re / p);
    if factor == 0.0 {
        return None;
    }
    let cos_i = -node_rate / factor;
    if cos_i.abs() > 1.0 {
        return None;
    }
    Some(Angle::from_radians(libm::acos(cos_i)))
}

/// The semi-major axis whose ground track repeats after `revs` orbital revolutions in `days` nodal
/// days — the resonance every imaging mission (Landsat, Sentinel) is designed around.
///
/// The track closes when the draconitic mean motion `Ṁ + ω̇` and the Earth's rotation relative to the
/// precessing node `ω⊕ − Ω̇` are in the ratio `revs/days`. All three `J₂` secular rates scale with the
/// mean motion `n ∝ a^{−3/2}`, so the condition `Ṁ + ω̇ = (revs/days)(ω⊕ − Ω̇)` closes to
/// `n = (revs/days) ω⊕ / (C + (revs/days) D)`, with `C, D` the (weakly `a`-dependent) `J₂` factors —
/// a fixed point in `a` that converges in a handful of iterations from the Keplerian guess.
///
/// `earth_rotation_rate` is `ω⊕` (sidereal), `revs`/`days` the repeat parameters (e.g. 233 and 16 for
/// Landsat-8).
#[must_use]
pub fn repeat_ground_track_sma(
    revs: u32,
    days: u32,
    eccentricity: f64,
    inclination: Angle,
    mu: f64,
    re: f64,
    j2: f64,
    earth_rotation_rate: f64,
) -> f64 {
    let q = f64::from(revs) / f64::from(days);
    let cos_i = libm::cos(inclination.radians());
    let root_1_e2 = libm::sqrt(1.0 - eccentricity * eccentricity);

    // Keplerian starting guess: n ≈ q ω⊕, a = (μ/n²)^{1/3}.
    let mut a = libm::cbrt(mu / (q * earth_rotation_rate * q * earth_rotation_rate));
    for _ in 0..60 {
        let p = a * (1.0 - eccentricity * eccentricity);
        let scale = j2 * (re / p) * (re / p);
        // n_nodal = n·C, Ω̇ = n·D
        let c = 1.0
            + 0.75
                * scale
                * (root_1_e2 * (3.0 * cos_i * cos_i - 1.0) + (5.0 * cos_i * cos_i - 1.0));
        let d = -1.5 * scale * cos_i;
        let n = q * earth_rotation_rate / (c + q * d);
        a = libm::cbrt(mu / (n * n));
    }
    a
}

/// The frozen-orbit eccentricity that balances the `J₂`/`J₃` secular drift of perigee, taken at
/// argument of perigee `ω = 90°`: `e_frozen ≈ −(J₃ / 2J₂)(Rₑ/a) sin i`.
///
/// Holding this `(e, ω = 90°)` keeps the perigee altitude nearly constant turn after turn — the
/// "frozen orbit" Earth-observation missions fly so the altitude over each latitude repeats. `j3` is
/// the (negative) third zonal harmonic; the result is small (`~10⁻³` in low orbit) and positive for a
/// prograde-or-retrograde orbit with the usual `J₃ < 0`.
#[must_use]
pub fn frozen_eccentricity(
    semi_major_axis: f64,
    inclination: Angle,
    re: f64,
    j2: f64,
    j3: f64,
) -> f64 {
    -(j3 / (2.0 * j2)) * (re / semi_major_axis) * libm::sin(inclination.radians())
}

#[cfg(test)]
mod tests {
    // Goldens are kept at the precision the CAS produced them; they are not f64 literals typed
    // by hand, and truncating them would be discarding the oracle's digits.
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::excessive_precision)]
    use super::*;
    use crate::keplerian::KeplerianElements;

    // WGS-84 / EGM zonal harmonics and geometry (primary-source values).
    const RE: f64 = 6_378_137.0;
    const MU: f64 = 3.986_004_418e14;
    const J2: f64 = 1.082_626_68e-3;
    const J3: f64 = -2.532_653_65e-6;
    const EARTH_RATE: f64 = 7.292_115_146_7e-5;

    #[test]
    fn sun_synchronous_low_orbit_is_near_ninety_eight_degrees() {
        let a = RE + 700_000.0;
        let i = sun_synchronous_inclination(a, 0.0, MU, RE, J2, SUN_SYNCHRONOUS_NODE_RATE)
            .expect("a Sun-synchronous inclination exists in low orbit");
        let deg = i.radians().to_degrees();
        assert!((97.5..98.7).contains(&deg), "i = {deg}°");
    }

    #[test]
    fn sun_synchronous_returns_none_when_the_orbit_is_too_high() {
        // A geosynchronous-radius orbit is far too slow for J2 to precess the node a full turn a year.
        assert!(
            sun_synchronous_inclination(42_164_000.0, 0.0, MU, RE, J2, SUN_SYNCHRONOUS_NODE_RATE)
                .is_none()
        );
    }

    #[test]
    fn repeat_ground_track_closes_the_resonance_it_solves() {
        // Landsat-8: 233 revolutions in 16 days, Sun-synchronous.
        let (revs, days) = (233u32, 16u32);
        let a = repeat_ground_track_sma(
            revs,
            days,
            0.0,
            Angle::from_radians(98.2_f64.to_radians()),
            MU,
            RE,
            J2,
            EARTH_RATE,
        );
        assert!((7.0e6..7.2e6).contains(&a), "a = {a}"); // ~705 km altitude
        // The solved a must satisfy Ṁ+ω̇ = (revs/days)(ω⊕ − Ω̇).
        let kep = KeplerianElements {
            semi_major_axis: a,
            eccentricity: 0.0,
            inclination: Angle::from_radians(98.2_f64.to_radians()),
            raan: Angle::from_radians(0.0),
            arg_periapsis: Angle::from_radians(0.0),
            true_anomaly: Angle::from_radians(0.0),
            mu: MU,
        };
        let lhs = kep.draconitic_mean_motion(RE, J2);
        let rhs =
            (f64::from(revs) / f64::from(days)) * (EARTH_RATE - kep.nodal_regression_rate(RE, J2));
        assert!(
            (lhs - rhs).abs() < 1e-11 * rhs,
            "resonance residual {}",
            lhs - rhs
        );
    }

    #[test]
    fn frozen_eccentricity_is_the_small_positive_j3_over_j2_value() {
        let a = RE + 700_000.0;
        let e = frozen_eccentricity(a, Angle::from_radians(98.0_f64.to_radians()), RE, J2, J3);
        assert!((0.8e-3..1.3e-3).contains(&e), "e_frozen = {e}");
    }

    fn at(a: f64, e: f64, i: Angle) -> KeplerianElements {
        KeplerianElements {
            semi_major_axis: a,
            eccentricity: e,
            inclination: i,
            raan: Angle::from_radians(0.0),
            arg_periapsis: Angle::from_radians(0.0),
            true_anomaly: Angle::from_radians(0.0),
            mu: MU,
        }
    }

    /// Every Sun-synchronous case above runs at `e = 0`, where the semi-latus rectum `p = a(1 − e²)`
    /// collapses to `a` and can be computed any way at all. At a real eccentricity it cannot: invert
    /// the node rate and read it back through `nodal_regression_rate`, whose closed form is derived
    /// independently in `derivations/j2_secular_rates.mac`.
    #[test]
    fn the_sun_synchronous_inclination_reproduces_the_node_rate_it_was_given() {
        for &e in &[0.0, 0.01, 0.2] {
            for &alt in &[600_000.0, 900_000.0] {
                let a = RE + alt;
                let i = sun_synchronous_inclination(a, e, MU, RE, J2, SUN_SYNCHRONOUS_NODE_RATE)
                    .expect("a Sun-synchronous inclination exists in low orbit");
                let got = at(a, e, i).nodal_regression_rate(RE, J2);
                let rel = (got - SUN_SYNCHRONOUS_NODE_RATE).abs() / SUN_SYNCHRONOUS_NODE_RATE;
                assert!(rel < 1e-14, "e={e}, alt={alt}: node rate off by {rel:.3e}");
            }
        }
    }

    /// The domain test is `|cos i| > 1`, not `≥`: the node rate an equatorial orbit delivers is the
    /// fastest J₂ can produce, and it is attainable — it must be answered, not refused. Unlike the
    /// float ties elsewhere in this crate the boundary is constructible, since `x/x` is exactly one
    /// for every finite non-zero `x`, so this is the edge of the stated domain and not a
    /// measure-zero coincidence.
    #[test]
    fn the_fastest_attainable_node_rate_is_inside_the_domain() {
        let (a, e) = (RE + 700_000.0, 0.1);
        let n = libm::sqrt(MU / (a * a * a));
        let p = a * (1.0 - e * e);
        let fastest = 1.5 * n * J2 * (RE / p) * (RE / p);

        let i = sun_synchronous_inclination(a, e, MU, RE, J2, -fastest)
            .expect("the equatorial orbit delivers exactly this node rate");
        assert!(
            i.radians() == 0.0,
            "the boundary solution is equatorial, not {}",
            i.degrees()
        );
        assert!(
            sun_synchronous_inclination(a, e, MU, RE, J2, -fastest * 1.000_000_1).is_none(),
            "no orbit regresses its node faster than the equatorial one"
        );
    }

    /// As above for the repeat-ground-track fixed point: at `e = 0` the `√(1−e²)` that weights the
    /// `Ṁ` correction against the `ω̇` one is exactly 1, and the resonance closes whatever it is.
    #[test]
    fn the_repeat_ground_track_closes_the_resonance_at_a_real_eccentricity() {
        let (revs, days) = (233u32, 16u32);
        let inc = Angle::from_radians(98.2_f64.to_radians());
        for &e in &[0.001, 0.05, 0.1] {
            let a = repeat_ground_track_sma(revs, days, e, inc, MU, RE, J2, EARTH_RATE);
            let kep = at(a, e, inc);
            let lhs = kep.draconitic_mean_motion(RE, J2);
            let rhs = (f64::from(revs) / f64::from(days))
                * (EARTH_RATE - kep.nodal_regression_rate(RE, J2));
            let rel = (lhs - rhs).abs() / rhs;
            assert!(rel < 1e-14, "e={e}: resonance residual {rel:.3e}");
        }
    }

    // e_frozen from derivations/j2_secular_rates.mac, which solves ω̇ = 0 at ω = 90° for the summed
    // J₂ + J₃ disturbing function; 40 significant digits. Columns: a (m), i (deg), e_frozen.
    const GOLDEN_FROZEN: [(f64, f64, f64); 4] = [
        (
            7_078_137.0,
            98.0,
            0.001_043_745_787_785_998_169_407_129_919_476_945_602_359,
        ),
        (
            7_078_137.0,
            30.0,
            0.000_527_001_637_603_233_626_893_510_678_912_176_742_879_7,
        ),
        (
            7_078_137.0,
            145.0,
            0.000_604_551_442_495_256_386_624_177_126_816_138_396_838_4,
        ),
        (
            26_560_000.0,
            63.4,
            0.000_251_157_003_396_418_363_328_321_684_767_835_552_591_5,
        ),
    ];

    /// The frozen eccentricity is asserted above only to lie in a band two hundred times wider than
    /// the value's own precision, which admits a 23 % error in the `Rₑ/a` factor and a 2 % one in
    /// `sin i`. The CAS derivation pins it instead.
    #[test]
    fn the_frozen_eccentricity_matches_the_solved_j2_j3_balance() {
        for (a, i_deg, want) in GOLDEN_FROZEN {
            let got = frozen_eccentricity(a, Angle::from_degrees(i_deg), RE, J2, J3);
            let rel = (got - want).abs() / want;
            assert!(rel < 5e-15, "a={a}, i={i_deg}°: e_frozen off by {rel:.3e}");
        }
    }
}
