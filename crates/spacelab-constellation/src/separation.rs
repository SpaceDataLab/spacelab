//! The minimum distance between two satellites of a Walker or circular lattice shell, over all time,
//! in closed form (plan 18 Ф27.J): a shell is safe from its own satellites iff that minimum is above
//! a keep-out distance.
//!
//! **Two satellites.** On circular orbits of one radius `r` and one inclination `i`, with node
//! difference `δΩ` and argument-of-latitude difference `δu` (b minus a), the angle `ψ` between the
//! position vectors obeys `cos ψ(t) = A + B cos(u_a(t) + u_b(t))`, with `A` and `B` constant and
//! `B = −sin²i sin²(δΩ/2) ≤ 0`. The sum of the latitudes turns at twice the mean motion, so the
//! largest cosine, `A − B`, is reached twice per orbit, and
//!
//! ```text
//! sin(ψ_min/2) = |sin(δu/2) cos(δΩ/2) + cos i cos(δu/2) sin(δΩ/2)| = ρ |sin(δu/2 + φ)|,
//! ρ cos φ = cos(δΩ/2),   ρ sin φ = cos i sin(δΩ/2),   ρ² = 1 − sin²i sin²(δΩ/2),
//! ```
//!
//! the chord being `2r sin(ψ_min/2)`. The minimum depends only on `δΩ` and `δu`, so it holds as well
//! under any motion that turns every node and every argument of latitude at common rates. Derived
//! from the two circular motions in `derivations/walker_min_separation.mac` (27 identities): the
//! form of `cos ψ(t)`, the sign of `B`, the stationary instants, the half-angle, the symmetries, the
//! Walker classes and the special cases below.
//!
//! **The pair classes.** A Walker Delta `i: T/P/F` (`T = P·S`, `F` taken mod `P` as the generator
//! does) is a subgroup of the node and latitude torus: plane `P` is plane 0 with its slots moved on
//! by `F`. So every difference between two of its satellites is an element of the group, and the
//! minimum depends only on that difference. The differences with plane offset `Δp` have
//! `δΩ = 2πΔp/P` and `δu = 2π(jP + ΔpF)/T` for every integer `j`, so `δu/2 + φ` runs over
//! `(π/S)(j + ΔpF/P + Sφ/π)`, one coset of a lattice of step `π/S`. `|sin|` has period `π`, which is
//! a lattice step times `S`, and grows with the distance from its zeros, so its smallest value on the
//! coset is at the point nearest a multiple of `π`:
//!
//! ```text
//! sin(ψ_min(Δp)/2) = ρ sin((π/S)·‖ΔpF/P + Sφ/π‖),   ‖x‖ the distance from x to the nearest integer,
//! ```
//!
//! and for `Δp = 0`, where the zero difference is the satellite itself, `sin(π/S)`. The class of `Δp`
//! is the negative of the class of `P − Δp`, and swapping the two satellites leaves the minimum
//! unchanged, so a Delta needs the `⌊P/2⌋ + 1` classes `Δp = 0 … ⌊P/2⌋` and no other. A Star's nodes
//! span `π` and do not wrap: its plane offsets `Δp = 0 … P − 1` are `P` classes, with
//! `δΩ = πΔp/P` and the same coset. A circular 2-D lattice `(No, Nso, Nc)` is the Walker Delta
//! `No·Nso/No/Nc` satellite for satellite ([`LatticeFlower`]); its common `Ω₀` and `ω` drop out.
//! `derivations/walker_min_separation.gp` checks the class form against the minimum over every pair
//! on 4 074 shells (`P, S ≤ 7`, `F ≤ P + 1`, Delta and Star, seven inclinations from 0 to π), and the
//! per-pair form against a minimum over time found without it on 200 random pairs, at 60 digits.
//!
//! **Degenerate shells.** One satellite, or a zero dimension, has no pair: refused. Two satellites
//! that meet — counter-rotating polar planes (`i = 90°`, `δΩ = π`, so `ρ = 0`), a crossing both reach
//! at once, or an equatorial shell whose longitudes repeat (`i = 0`, 4/2/0) — have a minimum of 0,
//! and 0 is reported (its enclosure's lower end), not refused: a shell that runs into itself is in
//! the model's domain, and it is the answer a collision screen needs most. A total `T` not divisible
//! by `P` cannot be stated: [`Walker`] takes `P` and `S`, and [`LatticeFlower`] `No` and `Nso`.
//!
//! **Round-off.** The minimum is returned as an enclosure **under H-libm, not outward-rounded**
//! ([`ShellSeparation::LABEL`]): `sin`, `cos`, `hypot`, `atan2` within their H-libm constants and
//! each operation one rounding, summed to first order over the operations above, put the computed
//! `sin(ψ_min/2)` within 50 `u` of the exact one (`u = 2⁻⁵³`), and within `ρ + 12u` when
//! `ρ < 10u`, where `atan2` has no first-order bound; the allowance [`Q_ALLOWANCE`] is `64u`. The
//! angle and the chord are taken at the ends of that interval, each widened by one ulp for `asin`
//! and the product. The angle's enclosure widens to about `2√(2·Q_ALLOWANCE)` next to `π`, where
//! `asin` is steep; the chord's does not.

use crate::lattice::LatticeFlower;
use crate::walker::{Walker, WalkerPattern};

const PI: f64 = core::f64::consts::PI;

/// The allowance on the computed `sin(ψ_min/2)`, `64u = 2⁻⁴⁷`: the first-order round-off of its
/// evaluation is at most `50u` under H-libm (module docs).
pub const Q_ALLOWANCE: f64 = 32.0 * f64::EPSILON;

/// Why a minimum separation was not computed: input outside the model is refused, never answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeparationError {
    /// Fewer than two satellites: a dimension is zero, or the shell is one satellite.
    NoPair,
    /// The radius or `mu` is not finite and positive, or an angle is not finite. With `mu = 0`
    /// nothing moves, and the minimum over time is not this formula's.
    BadOrbit,
    /// A lattice's eccentricity is not exactly 0: the closed form is for circular orbits.
    NotCircular,
    /// The keep-out distance is not finite and non-negative.
    BadThreshold,
}

impl core::fmt::Display for SeparationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoPair => write!(f, "the shell has fewer than two satellites"),
            Self::BadOrbit => write!(
                f,
                "the radius or mu is not finite and positive, or an angle is not finite"
            ),
            Self::NotCircular => write!(f, "the lattice is not circular"),
            Self::BadThreshold => write!(f, "the threshold is not finite and non-negative"),
        }
    }
}

impl core::error::Error for SeparationError {}

/// The minimum over all time and all pairs of the distance between two satellites of a shell, as
/// an enclosure, [`ShellSeparation::LABEL`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShellSeparation {
    /// Lower end of the minimum angle between two position vectors, radians.
    pub angle_lower: f64,
    /// Upper end of the minimum angle, radians.
    pub angle_upper: f64,
    /// Lower end of the minimum chord at the shell's radius, metres.
    pub chord_lower: f64,
    /// Upper end of the minimum chord, metres.
    pub chord_upper: f64,
    /// A plane offset whose pair class reaches the minimum: 0 is two satellites of one plane. A
    /// Delta's offsets are reported up to `⌊P/2⌋`, since `Δp` and `P − Δp` are one class.
    pub plane_offset: u32,
}

/// Whether a shell keeps its satellites farther apart than a keep-out distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeparationVerdict {
    /// Proven: no two satellites come within the threshold, ever.
    Safe,
    /// Proven: two satellites come within the threshold, or exactly to it.
    Unsafe,
    /// The threshold lies inside the enclosure, within its round-off: neither is proven.
    Undecided,
}

impl ShellSeparation {
    /// What the enclosure rests on: the `libm` elementary functions within their measured error
    /// (H-libm), and round-to-nearest arithmetic with a first-order allowance rather than outward
    /// rounding.
    pub const LABEL: &'static str = "under H-libm, not outward-rounded";

    /// Whether the shell is safe for the keep-out distance `threshold`, metres: safe iff the minimum
    /// chord is above it. `Safe` is decided on the enclosure's lower end and `Unsafe` on its upper
    /// end, so neither is said on a rounded value.
    ///
    /// # Errors
    /// [`SeparationError::BadThreshold`] for a threshold that is not finite and non-negative.
    pub fn verdict(&self, threshold: f64) -> Result<SeparationVerdict, SeparationError> {
        if !(threshold.is_finite() && threshold >= 0.0) {
            return Err(SeparationError::BadThreshold);
        }
        Ok(if self.chord_lower > threshold {
            SeparationVerdict::Safe
        } else if self.chord_upper <= threshold {
            SeparationVerdict::Unsafe
        } else {
            SeparationVerdict::Undecided
        })
    }
}

/// The minimum distance between two satellites of the Walker `w` over all time (module docs).
///
/// # Errors
/// [`SeparationError::NoPair`] for fewer than two satellites, and [`SeparationError::BadOrbit`]
/// for a semi-major axis or `mu` that is not finite and positive, or an inclination that is not
/// finite.
pub fn walker_min_separation(w: &Walker) -> Result<ShellSeparation, SeparationError> {
    check_orbit(w.semi_major_axis, w.mu, &[w.inclination.radians()])?;
    let star = w.pattern == WalkerPattern::Star;
    shell(
        w.planes,
        w.per_plane,
        w.phasing,
        star,
        w.inclination.radians(),
        w.semi_major_axis,
    )
}

/// The minimum distance between two satellites of the circular 2-D lattice `l` over all time: the
/// Walker Delta `No·Nso/No/Nc` of [`walker_min_separation`] (module docs).
///
/// # Errors
/// [`SeparationError::NotCircular`] for an eccentricity that is not exactly 0,
/// [`SeparationError::NoPair`] for fewer than two satellites, and [`SeparationError::BadOrbit`]
/// for a semi-major axis or `mu` that is not finite and positive, or an angle that is not finite.
pub fn lattice_min_separation(l: &LatticeFlower) -> Result<ShellSeparation, SeparationError> {
    // the model is exactly circular; any other eccentricity, NaN included, is outside it
    if l.eccentricity != 0.0 {
        return Err(SeparationError::NotCircular);
    }
    let angles = [
        l.inclination.radians(),
        l.arg_periapsis.radians(),
        l.raan0.radians(),
    ];
    check_orbit(l.semi_major_axis, l.mu, &angles)?;
    shell(
        l.orbits,
        l.sats_per_orbit,
        l.config,
        false,
        l.inclination.radians(),
        l.semi_major_axis,
    )
}

fn check_orbit(radius: f64, mu: f64, angles: &[f64]) -> Result<(), SeparationError> {
    let finite_positive = |x: f64| x.is_finite() && x > 0.0;
    if finite_positive(radius) && finite_positive(mu) && angles.iter().all(|a| a.is_finite()) {
        Ok(())
    } else {
        Err(SeparationError::BadOrbit)
    }
}

/// The enclosure of the class minimum of a Walker of `planes` planes of `per_plane` slots, phasing
/// `phasing` (used mod `planes`), Star when `star`, inclination `inc`, at `radius`.
fn shell(
    planes: u32,
    per_plane: u32,
    phasing: u32,
    star: bool,
    inc: f64,
    radius: f64,
) -> Result<ShellSeparation, SeparationError> {
    let (best, plane_offset) =
        class_minimum(planes, per_plane, phasing, star, inc).ok_or(SeparationError::NoPair)?;
    let q_lower = (best - Q_ALLOWANCE).max(0.0);
    let q_upper = (best + Q_ALLOWANCE).min(1.0);
    Ok(ShellSeparation {
        angle_lower: (2.0 * libm::asin(q_lower)).next_down().max(0.0),
        angle_upper: (2.0 * libm::asin(q_upper)).next_up(),
        chord_lower: (2.0 * (radius * q_lower)).next_down().max(0.0),
        chord_upper: (2.0 * (radius * q_upper)).next_up(),
        plane_offset,
    })
}

/// The computed `sin(ψ_min/2)` over the pair classes and a plane offset that reaches it, or `None`
/// for fewer than two satellites.
fn class_minimum(
    planes: u32,
    per_plane: u32,
    phasing: u32,
    star: bool,
    inc: f64,
) -> Option<(f64, u32)> {
    if u64::from(planes) * u64::from(per_plane) < 2 {
        return None;
    }
    let cos_i = libm::cos(inc);
    let slots = f64::from(per_plane);
    let node_span = if star {
        2.0 * f64::from(planes)
    } else {
        f64::from(planes)
    };
    let last = if star { planes - 1 } else { planes / 2 };
    let mut best = f64::INFINITY;
    let mut plane_offset = 0;
    for dp in 0..=last {
        let q = if dp == 0 {
            if per_plane < 2 {
                continue;
            }
            libm::sin(PI / slots)
        } else {
            // b = δΩ/2; ρ and φ as in the module docs
            let b = PI * f64::from(dp) / node_span;
            let (sin_b, cos_b) = (libm::sin(b), libm::cos(b));
            let y_b = cos_i * sin_b;
            let rho = libm::hypot(cos_b, y_b);
            let phi = libm::atan2(y_b, cos_b);
            let x = f64::from(phase_residue(dp, phasing, planes)) / f64::from(planes)
                + slots * phi / PI;
            rho * libm::sin(PI * (x - libm::round(x)).abs() / slots)
        };
        if q < best {
            best = q;
            plane_offset = dp;
        }
    }
    Some((best, plane_offset))
}

/// `Δp·F mod P`, the numerator of the coset's offset `ΔpF/P` (`F` used mod `P`).
fn phase_residue(dp: u32, phasing: u32, planes: u32) -> u32 {
    let p = u64::from(planes);
    let r = (u64::from(dp) * (u64::from(phasing) % p)) % p;
    // r < planes, so it fits
    u32::try_from(r).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::approx_constant,
        clippy::too_many_lines,
        reason = "the 60-digit values are kept as the derivation prints them, pi and its fractions \
                  among them; a test unwraps what it has just checked, and its tables are long"
    )]
    use super::*;
    use spacelab_core::{Angle, Vec3};
    use spacelab_elements::KeplerianElements;

    const MU: f64 = 3.986_004_418e14;
    const U: f64 = f64::EPSILON / 2.0;

    fn walker(
        pattern: WalkerPattern,
        inc: f64,
        planes: u32,
        per_plane: u32,
        phasing: u32,
        a: f64,
    ) -> Walker {
        Walker {
            pattern,
            inclination: Angle::from_radians(inc),
            planes,
            per_plane,
            phasing,
            semi_major_axis: a,
            mu: MU,
        }
    }

    fn delta(inc: f64, planes: u32, per_plane: u32, phasing: u32) -> Walker {
        walker(
            WalkerPattern::Delta,
            inc,
            planes,
            per_plane,
            phasing,
            7_000_000.0,
        )
    }

    /// `sin(ψ_min/2)` of one pair from its node and latitude differences, the per-pair form.
    fn pair_q(cos_i: f64, d_node: f64, d_lat: f64) -> f64 {
        (libm::sin(0.5 * d_lat) * libm::cos(0.5 * d_node)
            + cos_i * libm::cos(0.5 * d_lat) * libm::sin(0.5 * d_node))
        .abs()
    }

    /// (G3) The f64 code against the mathematics: the minimum angle and chord of 15 shells, each
    /// the minimum over every pair at 60 digits in `derivations/walker_min_separation.gp`, which
    /// recomputes these literals and checks them to 1e-21. Each enclosure contains its value, and
    /// the computed `sin(ψ_min/2)` is within [`Q_ALLOWANCE`] of the exact one; the worst is printed.
    /// The shells are Galileo, Globalstar, Iridium (Star, and with `F = 8 ≥ P`), a 1584-satellite
    /// shell of 72 planes, and the degenerate cases: `F = 0`, one plane, `i = 0`, `i = 90°`,
    /// `i = 120°`, two planes half a turn apart, and the three that meet.
    #[test]
    fn the_f64_code_holds_the_60_digit_values() {
        use WalkerPattern::{Delta, Star};
        let deg = f64::to_radians;
        let cases = [
            (
                Delta,
                deg(56.0),
                3,
                8,
                1,
                29_600_318.0,
                0.1598113707890583549856,
                4725435.060682667288297,
            ),
            (
                Delta,
                deg(52.0),
                8,
                6,
                1,
                7_792_137.0,
                0.1748572907451338051791,
                1360776.840645342741334,
            ),
            (
                Star,
                deg(86.4),
                6,
                11,
                2,
                7_158_137.0,
                0.07156334527594328243304,
                512150.9264667877266972,
            ),
            (
                Star,
                deg(86.4),
                6,
                11,
                8,
                7_158_137.0,
                0.07156334527594328243304,
                512150.9264667877266972,
            ),
            (
                Delta,
                deg(50.0),
                3,
                4,
                0,
                7_000_000.0,
                0.08016551146340497945592,
                561008.3301977001547150,
            ),
            (
                Delta,
                deg(30.0),
                1,
                6,
                0,
                7_000_000.0,
                1.047197551196597746154,
                7000000.0,
            ),
            (
                Delta,
                deg(40.0),
                1,
                2,
                0,
                7_000_000.0,
                3.141592653589793238463,
                14000000.0,
            ),
            (
                Delta,
                0.0,
                2,
                2,
                1,
                7_000_000.0,
                1.570796326794896619231,
                9899494.936611665341612,
            ),
            (Delta, 0.0, 4, 3, 1, 7_000_000.0, 0.0, 0.0),
            (Delta, 0.0, 2, 2, 0, 7_000_000.0, 0.0, 0.0),
            (
                Delta,
                deg(90.0),
                5,
                3,
                1,
                7_000_000.0,
                0.2520436791177251363824,
                1759639.489236153713356,
            ),
            (Delta, deg(90.0), 4, 3, 1, 7_000_000.0, 0.0, 0.0),
            (
                Delta,
                deg(120.0),
                3,
                4,
                1,
                7_000_000.0,
                0.4364690287349192555862,
                3031088.913245535263673,
            ),
            (
                Delta,
                deg(30.0),
                2,
                1,
                0,
                7_000_000.0,
                2.094395102393195492308,
                12124355.65298214105469,
            ),
            (
                Delta,
                deg(53.0),
                72,
                22,
                17,
                6_928_137.0,
                0.002911555714658274537849,
                20171.64974935719710328,
            ),
        ];
        let mut worst = 0.0_f64;
        for (pattern, inc, planes, per_plane, phasing, a, psi, chord) in cases {
            let w = walker(pattern, inc, planes, per_plane, phasing, a);
            let s = walker_min_separation(&w).unwrap();
            assert!(
                s.angle_lower <= psi && psi <= s.angle_upper,
                "{w:?}: angle [{}, {}] against {psi}",
                s.angle_lower,
                s.angle_upper
            );
            assert!(
                s.chord_lower <= chord && chord <= s.chord_upper,
                "{w:?}: chord [{}, {}] against {chord}",
                s.chord_lower,
                s.chord_upper
            );
            let (q_got, _) =
                class_minimum(planes, per_plane, phasing, pattern == Star, inc).unwrap();
            worst = worst.max((q_got - libm::sin(0.5 * psi)).abs());
        }
        println!(
            "worst |q - q_exact| = {worst:.3e} = {:.2} u, allowance {:.0} u",
            worst / U,
            Q_ALLOWANCE / U
        );
        assert!(worst <= Q_ALLOWANCE);
    }

    /// (S1) The class form is the minimum over every pair of the generator's own satellites: for
    /// every Walker Delta and Star with `P, S ≤ 6`, every `F ≤ P + 1` and six inclinations, the
    /// per-pair form evaluated on each pair of emitted elements (node and true-anomaly differences)
    /// has its smallest value within the allowance, plus `16u` for its own rounding, of the class
    /// form's. The worst is printed.
    #[test]
    fn the_class_minimum_is_the_minimum_over_every_pair_the_generator_emits() {
        let mut worst = 0.0_f64;
        let mut shells = 0;
        for planes in 1..=6 {
            for per_plane in 1..=6 {
                if planes * per_plane < 2 {
                    continue;
                }
                for phasing in 0..=planes + 1 {
                    for pattern in [WalkerPattern::Delta, WalkerPattern::Star] {
                        for inc in [0.0, 0.3, 0.98, 1.5, PI / 2.0, 2.2] {
                            let w = walker(pattern, inc, planes, per_plane, phasing, 7e6);
                            let mut sats = Vec::new();
                            w.generate(|e| sats.push(e));
                            let cos_i = libm::cos(inc);
                            let mut brute = f64::INFINITY;
                            for (k, a) in sats.iter().enumerate() {
                                for b in &sats[k + 1..] {
                                    let d_node = b.raan.radians() - a.raan.radians();
                                    let d_lat = b.true_anomaly.radians() - a.true_anomaly.radians();
                                    brute = brute.min(pair_q(cos_i, d_node, d_lat));
                                }
                            }
                            let star = pattern == WalkerPattern::Star;
                            let (q, _) =
                                class_minimum(planes, per_plane, phasing, star, inc).unwrap();
                            worst = worst.max((brute - q).abs());
                            // 16u for the pair form's own rounding
                            assert!(
                                (brute - q).abs() <= Q_ALLOWANCE + 16.0 * U,
                                "{w:?}: every pair {brute}, class form {q}"
                            );
                            shells += 1;
                        }
                    }
                }
            }
        }
        println!(
            "{shells} shells: class form against every pair, worst {worst:.3e} = {:.2} u",
            worst / U
        );
    }

    /// The smallest angle and chord between two satellites over `steps` instants of one period,
    /// from the elements propagated and turned into position vectors: no closed form is used. The
    /// instants start 0.37 step after `t = 0`, so that none lands on a symmetric instant of the
    /// pattern where a minimum sits.
    fn scan(sats: &[KeplerianElements], steps: u32) -> (f64, f64) {
        let period = sats[0].period();
        let (mut angle, mut chord) = (f64::INFINITY, f64::INFINITY);
        let mut at: Vec<Vec3> = Vec::with_capacity(sats.len());
        for k in 0..steps {
            let t = period * (f64::from(k) + 0.37) / f64::from(steps);
            at.clear();
            at.extend(sats.iter().map(|e| e.propagate(t).to_state().0));
            for (m, a) in at.iter().enumerate() {
                for b in &at[m + 1..] {
                    angle = angle.min(libm::atan2(a.cross(*b).norm(), a.dot(*b)));
                    chord = chord.min((*a - *b).norm());
                }
            }
        }
        (angle, chord)
    }

    /// (S1) A dense time scan never falls below the closed form and stays within the bound of what
    /// a scan can miss. Each satellite's direction turns at the mean motion `n`, so the angle between
    /// two is `2n`-Lipschitz in time (the triangle inequality on the sphere; also
    /// `rate_bound_is_a_convex_combination` in the derivation), and the instant of the minimum is
    /// within half a step `h/2` of a sample: the scan's minimum is at most `n·h` above the true one,
    /// `r·n·h` in chord. The scan propagates the generator's elements and measures the vectors, with
    /// `N = 2048` steps per period (`n·h = 2π/2048 = 3.1e-3`). Its own rounding is allowed `1e-13`
    /// relative. Printed: the gap and its bound.
    #[test]
    fn a_dense_time_scan_never_falls_below_the_closed_form() {
        const STEPS: u32 = 2048;
        let deg = f64::to_radians;
        let shells = [
            walker(WalkerPattern::Delta, deg(56.0), 3, 8, 1, 29_600_318.0),
            walker(WalkerPattern::Star, deg(86.4), 6, 11, 2, 7_158_137.0),
            walker(WalkerPattern::Delta, deg(52.0), 8, 6, 1, 7_792_137.0),
            walker(WalkerPattern::Delta, deg(50.0), 3, 4, 0, 7e6),
            walker(WalkerPattern::Delta, deg(90.0), 5, 3, 1, 7e6),
            walker(WalkerPattern::Delta, deg(120.0), 3, 4, 1, 7e6),
            walker(WalkerPattern::Delta, deg(90.0), 4, 3, 1, 7e6),
        ];
        let miss = core::f64::consts::TAU / f64::from(STEPS);
        for w in shells {
            let mut sats = Vec::new();
            w.generate(|e| sats.push(e));
            let (angle, chord) = scan(&sats, STEPS);
            let s = walker_min_separation(&w).unwrap();
            let a = w.semi_major_axis;
            assert!(
                angle >= s.angle_lower - 1e-13 && angle <= s.angle_upper + miss,
                "{w:?}: scan {angle} against [{}, {}] + {miss}",
                s.angle_lower,
                s.angle_upper
            );
            assert!(
                chord >= s.chord_lower - 1e-13 * a && chord <= s.chord_upper + a * miss,
                "{w:?}: scan {chord} m against [{}, {}] + {}",
                s.chord_lower,
                s.chord_upper,
                a * miss
            );
            let star = w.pattern == WalkerPattern::Star;
            let inc = w.inclination.radians();
            let (q, _) = class_minimum(w.planes, w.per_plane, w.phasing, star, inc).unwrap();
            println!(
                "{}/{}/{} at {:.1}: scan - closed form = {:.3e} rad of {miss:.3e}",
                w.total(),
                w.planes,
                w.phasing,
                inc.to_degrees(),
                angle - 2.0 * libm::asin(q),
            );
        }
    }

    /// (G3) The degenerate cases have the values the derivation gives them: one plane is `2π/S`;
    /// `F = 0` is a shell like any other; `i = 0` is the longitude spacing; two planes half a turn
    /// apart are `π − 2i` (2/2/0); a pair that meets is 0, reported and not refused — counter-rotating
    /// polar planes (12/4/1 at 90°, planes 0 and 2), a co-located equatorial pair (4/2/0) — and is
    /// unsafe for any keep-out above the round-off.
    #[test]
    fn the_degenerate_shells_have_their_exact_values() {
        let deg = f64::to_radians;
        let within = |s: ShellSeparation, psi: f64| s.angle_lower <= psi && psi <= s.angle_upper;
        let one_plane = walker_min_separation(&delta(deg(30.0), 1, 6, 0)).unwrap();
        assert!(within(one_plane, PI / 3.0) && one_plane.plane_offset == 0);
        let two_planes = walker_min_separation(&delta(deg(30.0), 2, 1, 0)).unwrap();
        assert!(within(two_planes, PI - 2.0 * deg(30.0)) && two_planes.plane_offset == 1);
        let equatorial = walker_min_separation(&delta(0.0, 2, 2, 1)).unwrap();
        assert!(within(equatorial, PI / 2.0));
        for (w, offset) in [(delta(deg(90.0), 4, 3, 1), 2), (delta(0.0, 2, 2, 0), 1)] {
            let s = walker_min_separation(&w).unwrap();
            assert_eq!((s.angle_lower, s.chord_lower), (0.0, 0.0), "{w:?}");
            assert!(s.chord_upper <= 2.0 * w.semi_major_axis * Q_ALLOWANCE * 1.01);
            assert_eq!(s.plane_offset, offset, "{w:?}");
            assert_eq!(s.verdict(1e-3), Ok(SeparationVerdict::Unsafe));
        }
        // 15/5/1 at 90°: odd P, no counter-rotating pair, so the shell is clear of itself
        let polar = walker_min_separation(&delta(deg(90.0), 5, 3, 1)).unwrap();
        assert!(polar.chord_lower > 1e6);
    }

    /// (S2) Every input outside the model is a typed refusal: no pair, a bad orbit, a lattice that is
    /// not circular, a threshold that is not a distance.
    #[test]
    fn input_outside_the_model_is_refused() {
        for (planes, per_plane) in [(1, 1), (0, 5), (5, 0), (0, 0)] {
            assert_eq!(
                walker_min_separation(&delta(1.0, planes, per_plane, 0)),
                Err(SeparationError::NoPair)
            );
        }
        let good = delta(1.0, 3, 4, 1);
        for bad in [
            Walker {
                semi_major_axis: 0.0,
                ..good
            },
            Walker {
                semi_major_axis: f64::INFINITY,
                ..good
            },
            Walker {
                semi_major_axis: f64::NAN,
                ..good
            },
            Walker { mu: 0.0, ..good },
            Walker { mu: -MU, ..good },
            Walker {
                mu: f64::NAN,
                ..good
            },
            Walker {
                inclination: Angle::from_radians(f64::NAN),
                ..good
            },
            Walker {
                inclination: Angle::from_radians(f64::INFINITY),
                ..good
            },
        ] {
            assert_eq!(
                walker_min_separation(&bad),
                Err(SeparationError::BadOrbit),
                "{bad:?}"
            );
        }
        let lattice = circular_lattice(3, 4, 1);
        for e in [1e-12, 0.1, f64::NAN, -1e-300] {
            let l = LatticeFlower {
                eccentricity: e,
                ..lattice
            };
            assert_eq!(
                lattice_min_separation(&l),
                Err(SeparationError::NotCircular)
            );
        }
        for bad in [
            LatticeFlower {
                raan0: Angle::from_radians(f64::NAN),
                ..lattice
            },
            LatticeFlower {
                arg_periapsis: Angle::from_radians(f64::INFINITY),
                ..lattice
            },
            LatticeFlower { mu: 0.0, ..lattice },
        ] {
            assert_eq!(lattice_min_separation(&bad), Err(SeparationError::BadOrbit));
        }
        let s = walker_min_separation(&good).unwrap();
        for t in [f64::NAN, -1.0, f64::INFINITY] {
            assert_eq!(s.verdict(t), Err(SeparationError::BadThreshold));
        }
    }

    /// (S2) The verdict is `Safe` only below the enclosure's lower end, `Unsafe` only at or above its
    /// upper end, and `Undecided` between them.
    #[test]
    fn the_verdict_says_safe_only_below_the_enclosure() {
        let s = walker_min_separation(&delta(0.9, 6, 4, 1)).unwrap();
        assert!(s.chord_lower < s.chord_upper);
        assert_eq!(s.verdict(0.0), Ok(SeparationVerdict::Safe));
        assert_eq!(
            s.verdict(s.chord_lower.next_down()),
            Ok(SeparationVerdict::Safe)
        );
        assert_eq!(s.verdict(s.chord_lower), Ok(SeparationVerdict::Undecided));
        assert_eq!(
            s.verdict(s.chord_upper.next_down()),
            Ok(SeparationVerdict::Undecided)
        );
        assert_eq!(s.verdict(s.chord_upper), Ok(SeparationVerdict::Unsafe));
        assert_eq!(s.verdict(1e7), Ok(SeparationVerdict::Unsafe));
    }

    fn circular_lattice(orbits: u32, sats_per_orbit: u32, config: u32) -> LatticeFlower {
        LatticeFlower {
            orbits,
            sats_per_orbit,
            config,
            semi_major_axis: 7e6,
            eccentricity: 0.0,
            inclination: Angle::from_radians(0.97),
            arg_periapsis: Angle::from_radians(1.3),
            raan0: Angle::from_radians(0.4),
            mu: MU,
        }
    }

    /// (S1) A circular lattice is its Walker Delta: the same enclosure bit for bit for every
    /// configuration, `Nc ≥ No` included, whatever its `Ω₀` and `ω`; and a dense scan of the
    /// lattice's own emitted satellites stays within the scan bound of it.
    #[test]
    fn a_circular_lattice_is_its_walker_delta() {
        for (no, nso, nc) in [(3, 4, 1), (5, 2, 7), (6, 6, 0), (4, 5, 3)] {
            let l = circular_lattice(no, nso, nc);
            let w = walker(WalkerPattern::Delta, 0.97, no, nso, nc, 7e6);
            assert_eq!(lattice_min_separation(&l), walker_min_separation(&w));
        }
        let l = circular_lattice(5, 2, 7);
        let mut sats = Vec::new();
        l.generate(|e| sats.push(e));
        let (angle, _) = scan(&sats, 2048);
        let s = lattice_min_separation(&l).unwrap();
        let miss = core::f64::consts::TAU / 2048.0;
        assert!(angle >= s.angle_lower - 1e-13 && angle <= s.angle_upper + miss);
    }

    #[test]
    fn each_error_says_something_different() {
        let messages: Vec<String> = [
            SeparationError::NoPair,
            SeparationError::BadOrbit,
            SeparationError::NotCircular,
            SeparationError::BadThreshold,
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        for (k, m) in messages.iter().enumerate() {
            assert_ne!(m, "");
            assert!(!messages[k + 1..].contains(m), "{m}");
        }
    }
}
