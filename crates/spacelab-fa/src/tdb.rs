//! Barycentric Dynamical Time: the scale a solar-system ephemeris is a function of.
//!
//! TT runs on the geoid; TDB runs at the solar-system barycentre. General relativity
//! puts a periodic difference between them, dominated by the annual term of the
//! Earth's eccentric orbit and reaching \u{b1}1.7 ms. The Earth's own gravity field and
//! its motion about the barycentre are what generate it, so TDB\u{2212}TT is not a
//! definitional offset like `TT = TAI + 32.184 s` but a computed one, and the
//! function that computes it is a 787-term periodic series.
//!
//! It is small, and it is not negligible: 1.7 ms of a planetary ephemeris argument
//! is metres of planetary position, and any propagator that reads a JPL ephemeris is
//! reading it at a TDB argument whether or not it says so.
//!
//! # Provenance
//!
//! The series is Fairhead & Bretagnon (1990, A&A 229, 240) with the fundamental
//! arguments of Simon et al. (1994, A&A 282, 663), adopted for IAU 2006 resolution
//! B3 and described in IERS Conventions 2010 (TN36) section 10.1. The coefficients
//! (`crate::tdb_tables`) are machine-extracted from SOFA's `dtdb.c` by
//! `derivations/tdb_tables.py`, which checks the extraction against the SOFA `t_dtdb`
//! self-test vector to `< 1e-15` s; no document in the repository prints them.
//! `derivations/tdb_series.mac` sums the series in 40-digit arithmetic and writes the
//! values the tests hold [`tdb_minus_tt`] to: it decides the evaluation, not the
//! transcription. `pyerfa` is a comparator.
//!
//! # What is not here
//!
//! The full model carries ten diurnal terms that depend on where the observer
//! stands: a clock on the rotating Earth sees a slightly different TDB from one at
//! the geocentre. At the equator they reach 2.14 µs. Everything in this workspace
//! that wants TDB wants it as the argument of an ephemeris — a geocentric or
//! barycentric quantity, for which those terms are zero by construction — so they
//! are not implemented rather than implemented and silently passed zeros.

use libm::sin;
use spacelab_core::time::{Epoch, Tdb, Tt};

use crate::tdb_tables::{FAIRHEAD, GROUP_END, JPL_MASS, JPL_MASS_T2};

/// Days in a Julian millennium. The series argument is millennia, not centuries.
const DAYS_PER_JULIAN_MILLENNIUM: f64 = 365_250.0;

/// Nanoseconds in one second, as the `f64` the series result is scaled by.
const NS_PER_SECOND: f64 = 1e9;

/// `TDB − TT` in seconds, at the geocentre.
///
/// Peaks near \u{b1}1.7 ms, dominated by the annual term. Evaluating the series at TT
/// rather than TDB is what every implementation of this model does and what the model
/// is defined for: the two arguments differ by at most 1.7 ms, and the series changes
/// by `< 1e-13` s over that interval.
#[must_use]
pub fn tdb_minus_tt(tt: Epoch<Tt>) -> f64 {
    let (day, frac) = tt.day_and_fraction();
    #[allow(clippy::cast_precision_loss)] // |day| < 2^53 for any epoch this crate accepts
    let t = ((day as f64) + frac) / DAYS_PER_JULIAN_MILLENNIUM;

    // Horner over the five power-of-t groups, summed within each group first: the
    // amplitudes are sorted descending, so accumulating from the small end keeps the
    // partial sums from swallowing the tail.
    let mut group = [0.0_f64; 5];
    let mut lo = 0;
    for (acc, &hi) in group.iter_mut().zip(&GROUP_END) {
        for term in FAIRHEAD[lo..hi].iter().rev() {
            *acc += term[0] * sin(term[1] * t + term[2]);
        }
        lo = hi;
    }
    let mut w = 0.0;
    for &g in group.iter().rev() {
        w = w * t + g;
    }

    // The series was fitted with the IAU planetary masses; this carries it onto the
    // JPL ones the ephemerides are actually built with.
    let mut adjustment = 0.0;
    for term in &JPL_MASS {
        adjustment += term[0] * sin(term[1] * t + term[2]);
    }
    w + adjustment + JPL_MASS_T2 * t * t
}

/// `TT → TDB`.
#[must_use]
pub fn tt_to_tdb(tt: Epoch<Tt>) -> Epoch<Tdb> {
    Epoch::from_ns_since_j2000(tt.ns_since_j2000() + offset_ns(tdb_minus_tt(tt)))
}

/// `TDB → TT`.
///
/// The series takes TDB where [`tt_to_tdb`] gives it TT. That is not an approximation
/// resolved by iterating: the model is a function of the barycentric argument to
/// within its own resolution, and the round trip through both directions is exact to
/// the nanosecond an [`Epoch`] stores.
#[must_use]
pub fn tdb_to_tt(tdb: Epoch<Tdb>) -> Epoch<Tt> {
    let at_tdb = Epoch::<Tt>::from_ns_since_j2000(tdb.ns_since_j2000());
    Epoch::from_ns_since_j2000(tdb.ns_since_j2000() - offset_ns(tdb_minus_tt(at_tdb)))
}

/// Seconds to the exact nanosecond count an [`Epoch`] is built from.
#[allow(clippy::cast_possible_truncation)] // |seconds| < 2e-3, so the product is < 2e6
fn offset_ns(seconds: f64) -> i128 {
    libm::round(seconds * NS_PER_SECOND) as i128
}

#[cfg(test)]
mod tests {
    // Exact values to 25 digits as the derivation prints them.
    #![allow(clippy::excessive_precision, clippy::unreadable_literal)]

    use super::*;

    /// TT days since J2000: the SOFA self-test epoch (TT JD 2400000.5 + 53736.0), two centuries,
    /// one and a half either side of J2000, and a three-quarter-day fraction.
    const TDB_DAYS: [f64; 8] = [
        2191.5, -73_050.0, -36_525.0, -18_262.5, 1234.75, 18_262.5, 36_525.0, 73_050.0,
    ];

    /// `TDB − TT` at [`TDB_DAYS`]: the series summed in 40-digit arithmetic at the `t` this module
    /// divides each day count to, to 25 digits (`derivations/tdb_series.mac` reads this table back).
    const TDB_EXACT: [f64; 8] = [
        -5.541284160984533953499400e-5,
        -1.292788238000218095574254e-5,
        -3.332174010175927790157063e-5,
        -7.069829559474016350723833e-5,
        1.148843571247779995129053e-3,
        -8.018829477932007950283626e-5,
        -7.561525671871068648326857e-5,
        -1.588113502821555415920919e-4,
    ];

    /// The worst miss measured is 2.673e-16 s, two centuries out: there the annual term's argument
    /// is 1263 rad, whose half-ulp of 1.1e-13 rad times the term's 1.66 ms is 1.9e-16 s. The bound
    /// is 3.7 times that.
    const TDB_BOUND: f64 = 1.0e-15;

    /// `TDB − TT` is its series summed exactly. (G3)
    ///
    /// The series' blocks multiply `t⁰ … t⁴`, `t` in millennia: a tenth of a millennium near J2000
    /// cannot see the `t⁴` block, two centuries out it carries its own weight.
    #[test]
    fn the_offset_is_its_series_summed_exactly() {
        let mut worst = (0.0_f64, 0.0_f64);
        for (days, want) in TDB_DAYS.into_iter().zip(TDB_EXACT) {
            let miss = (tdb_minus_tt(Epoch::<Tt>::from_days_since_j2000(days)) - want).abs();
            if miss > worst.0 || miss.is_nan() {
                worst = (miss, days);
            }
        }
        assert!(worst.0 < TDB_BOUND, "{:.3e} s at day {}", worst.0, worst.1);
    }

    /// The annual term is the whole story, and it has to have the right size.
    #[test]
    fn the_offset_peaks_near_one_point_seven_milliseconds() {
        let mut worst = 0.0_f64;
        for d in 0..366 {
            let tt = Epoch::<Tt>::from_days_since_j2000(f64::from(d));
            worst = worst.max(tdb_minus_tt(tt).abs());
        }
        assert!(
            (1.6e-3..1.8e-3).contains(&worst),
            "peak |TDB-TT| over a year is {worst:.4e} s, not the expected ~1.7 ms"
        );
    }

    #[test]
    fn tt_and_tdb_round_trip_to_the_nanosecond() {
        for d in [-36_525.0, -1_000.5, 0.0, 1_234.75, 18_262.25, 36_525.0] {
            let tt = Epoch::<Tt>::from_days_since_j2000(d);
            let back = tdb_to_tt(tt_to_tdb(tt));
            assert_eq!(back, tt, "round trip drifted at day {d}");
        }
    }

    /// The scales are distinct types, so the only way to compare them is through a
    /// conversion — which is the property that makes the tag worth carrying.
    #[test]
    fn the_conversion_actually_moves_the_epoch() {
        let tt = Epoch::<Tt>::from_days_since_j2000(100.0);
        let tdb = tt_to_tdb(tt);
        let moved = tdb.ns_since_j2000() - tt.ns_since_j2000();
        assert!(moved != 0, "TDB and TT cannot be the same instant here");
        assert!(
            moved.unsigned_abs() < 2_000_000,
            "{moved} ns is not a TDB offset"
        );
    }

    /// The epoch moves by the offset the series computed, in the unit the epoch is stored in.
    ///
    /// Neither the round trip nor the bounds above pin this: a conversion that always
    /// displaced the epoch by a single nanosecond would round-trip exactly, would move the
    /// epoch, and would be well under two milliseconds. What it would not be is `TDB − TT`.
    /// The two public functions report the same quantity in different units, and the whole
    /// content of the assertion is that they agree.
    #[test]
    fn the_epoch_moves_by_the_offset_the_series_computed() {
        let mut worst_ns = 0_i128;
        for d in [-36_525.0, -1_000.5, 0.0, 45.25, 91.75, 1_234.75, 36_525.0] {
            let tt = Epoch::<Tt>::from_days_since_j2000(d);
            let moved = tt_to_tdb(tt).ns_since_j2000() - tt.ns_since_j2000();
            #[allow(clippy::cast_precision_loss)] // |moved| < 2e6, exact in f64
            let seconds = (moved as f64) * 1e-9;
            let want = tdb_minus_tt(tt);
            assert!(
                (seconds - want).abs() < 0.5e-9,
                "day {d}: epoch moved {seconds:.12} s, series says {want:.12} s"
            );
            worst_ns = worst_ns.max(moved.abs());
        }
        // And the size is milliseconds of general relativity, not the nanosecond an
        // always-one-nanosecond conversion would give.
        assert!(worst_ns > 1_000_000, "worst displacement was {worst_ns} ns");
    }
}
