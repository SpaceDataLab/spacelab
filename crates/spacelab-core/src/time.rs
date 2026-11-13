//! Epochs, tagged with their time scale.
//!
//! An epoch is an exact integer count of nanoseconds from the J2000 epoch instant
//! *on its own scale*, i.e. from Julian Date 2451545.0 measured on that scale.
//!
//! Fixed point, not a floating Julian Date. Near JD 2460000 the representable
//! `f64` values sit 40.2 µs apart — `2^-31` days — so no rounding does better
//! than half of that, which is why SOFA threads a two-part `(d1, d2)` split
//! through every one of its routines. An `i128` of nanoseconds resolves exactly,
//! spans far more than the age of the universe, and lets the API never mention
//! the problem.

use core::marker::PhantomData;

/// Nanoseconds in one day. Exact.
pub const NS_PER_DAY: i128 = 86_400 * 1_000_000_000;

/// Nanoseconds in one day, as `f64`.
///
/// `8.64e13 < 2⁵³`, so this is exactly representable and every `i128` nanosecond
/// offset within a day converts to `f64` without loss.
pub const NS_PER_DAY_F64: f64 = 86_400e9;

/// Days in a Julian century. Exact, by definition.
pub const DAYS_PER_JULIAN_CENTURY: f64 = 36_525.0;

/// Seconds in one day. Exact.
pub const SECONDS_PER_DAY: f64 = 86_400.0;

/// A **uniform** scale on which time is measured.
///
/// Every scale here ticks without interruption, so an integer count of
/// nanoseconds from a fixed instant names each of its moments exactly once.
///
/// UTC is deliberately absent. It is not uniform — a day with a leap second holds
/// 86401 of them — and so it is a calendar, [`UtcDateTime`](crate::UtcDateTime),
/// not an [`Epoch`].
///
/// Sealed: adding a scale means adding its conversions, and the exact ones live here.
/// A conversion that needs a measured table or a fitted series does not — `UTC ↔ TAI`
/// needs the leap-second table and `TT ↔ TDB` a 787-term series, so both live in
/// `spacelab-fa`, where their provenance can be stated and tested against an oracle.
pub trait TimeScale: sealed::Sealed + Copy + core::fmt::Debug + 'static {
    /// Short name, for diagnostics.
    const NAME: &'static str;
}

mod sealed {
    pub trait Sealed {}
}

macro_rules! time_scale {
    ($(#[$m:meta])* $name:ident, $s:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $name;
        impl sealed::Sealed for $name {}
        impl TimeScale for $name {
            const NAME: &'static str = $s;
        }
    };
}

time_scale!(
    /// International Atomic Time.
    Tai, "TAI"
);
time_scale!(
    /// Terrestrial Time. `TT = TAI + 32.184 s`, exactly.
    Tt, "TT"
);
time_scale!(
    /// Universal Time, tied to the rotation of the Earth. Obtaining it requires
    /// Earth-orientation data; it is not a function of any other scale alone.
    Ut1, "UT1"
);
time_scale!(
    /// Barycentric Dynamical Time, the argument of a solar-system ephemeris.
    ///
    /// Differs from TT by a relativistic periodic term reaching ±1.7 ms, so the
    /// conversion is a fitted series rather than an offset: `spacelab_fa::tt_to_tdb`.
    Tdb, "TDB"
);

/// An instant, on a known time scale.
///
/// The scale is a type parameter, erased at monomorphization. Passing an
/// `Epoch<Utc>` where an `Epoch<Ut1>` is expected does not compile — and the gap
/// between them reaches 0.9 seconds of Earth rotation, which is 400 metres at the
/// equator, while leap seconds hold it there. From 2027-05-20 none do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Epoch<S: TimeScale> {
    ns_since_j2000: i128,
    // `fn(S) -> S` keeps the parameter invariant and keeps `Epoch` unconditionally
    // `Send + Sync` regardless of the tag type.
    _scale: PhantomData<fn(S) -> S>,
}

impl<S: TimeScale> Epoch<S> {
    /// The J2000 epoch on this scale: Julian Date 2451545.0.
    pub const J2000: Self = Self::from_ns_since_j2000(0);

    /// Construct from an exact nanosecond count since J2000.
    #[must_use]
    pub const fn from_ns_since_j2000(ns: i128) -> Self {
        Self {
            ns_since_j2000: ns,
            _scale: PhantomData,
        }
    }

    /// The exact nanosecond count since J2000.
    #[must_use]
    pub const fn ns_since_j2000(self) -> i128 {
        self.ns_since_j2000
    }

    /// Construct from a floating count of days since J2000.
    ///
    /// Lossy by construction; prefer [`Self::from_ns_since_j2000`]. Provided for
    /// ingesting external data that never had more than `f64` days to give.
    ///
    /// `None` unless `|days| < 2⁸⁰`, 3.3 × 10²¹ years, which also refuses NaN and both
    /// infinities. Inside that range the count fits an `i128` (Kani proof
    /// `from_days_since_j2000_is_total`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // `whole` is integer-valued; `frac * NS_PER_DAY_F64` < 2^53
    pub fn from_days_since_j2000(days: f64) -> Option<Self> {
        // 2⁸⁰ days are 1.04 × 10³⁸ ns, below `i128::MAX` = 1.70 × 10³⁸.
        #[expect(
            clippy::cast_precision_loss,
            reason = "2⁸⁰ is a power of two, exact in f64"
        )]
        const LIMIT_DAYS: f64 = (1_u128 << 80) as f64;
        if !days.is_finite() || days.abs() >= LIMIT_DAYS {
            return None;
        }
        // Split before scaling, so the integer part never enters a product.
        let whole = libm::trunc(days);
        let frac = days - whole;
        let ns = (whole as i128) * NS_PER_DAY + (frac * NS_PER_DAY_F64) as i128;
        Some(Self::from_ns_since_j2000(ns))
    }

    /// Whole days since J2000, and the fraction of a day remaining, in `[0, 1)`.
    ///
    /// Euclidean, not truncating: one nanosecond before J2000 is day `-1` with a
    /// fraction just under one, never day `0` with a negative fraction.
    ///
    /// Splitting here rather than forming a single `days: f64` is what preserves
    /// precision — the large integer part never enters the mantissa alongside the
    /// small fractional one.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // `rem` < 2^53, so the conversion is exact
    pub fn day_and_fraction(self) -> (i128, f64) {
        let day = self.ns_since_j2000.div_euclid(NS_PER_DAY);
        let rem = self.ns_since_j2000.rem_euclid(NS_PER_DAY);
        let frac = (rem as f64) / NS_PER_DAY_F64;
        (day, frac)
    }

    /// Julian centuries since J2000 on this scale.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn julian_centuries(self) -> f64 {
        let (day, frac) = self.day_and_fraction();
        ((day as f64) + frac) / DAYS_PER_JULIAN_CENTURY
    }
}

/// Offset between TAI and TT: 32.184 s, exactly, by definition.
const TT_MINUS_TAI_NS: i128 = 32_184_000_000;

/// `TT = TAI + 32.184 s`. Exact, hence infallible.
///
/// Compare the signature of a UTC-to-UT1 conversion, which must take Earth
/// orientation data and return a `Result`. Conversions that are definitionally
/// exact return a value; conversions that need measured data cannot pretend to.
/// That asymmetry is visible in the types, which is the point.
///
/// Exact for every epoch up to `i128::MAX − 32 184 000 000` ns, 5.4 × 10²¹ years after
/// J2000; only past it does the `i128` addition overflow (Kani proof
/// `tai_tt_round_trip_is_exact`).
#[must_use]
pub const fn tai_to_tt(t: Epoch<Tai>) -> Epoch<Tt> {
    Epoch::from_ns_since_j2000(t.ns_since_j2000() + TT_MINUS_TAI_NS)
}

/// `TAI = TT - 32.184 s`. Exact, hence infallible.
///
/// Exact for every epoch down to `i128::MIN + 32 184 000 000` ns, as for [`tai_to_tt`].
#[must_use]
pub const fn tt_to_tai(t: Epoch<Tt>) -> Epoch<Tai> {
    Epoch::from_ns_since_j2000(t.ns_since_j2000() - TT_MINUS_TAI_NS)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // a panicking test is a failing test
    use super::*;

    #[test]
    fn j2000_is_zero_on_every_scale() {
        assert_eq!(Epoch::<Tai>::J2000.ns_since_j2000(), 0);
        assert_eq!(Epoch::<Ut1>::J2000.ns_since_j2000(), 0);
    }

    #[test]
    fn day_and_fraction_splits_at_midday() {
        let (day, frac) = Epoch::<Ut1>::from_ns_since_j2000(NS_PER_DAY / 2).day_and_fraction();
        assert_eq!(day, 0);
        assert!((frac - 0.5).abs() < 1e-15);
    }

    #[test]
    fn day_and_fraction_is_euclidean_before_j2000() {
        let (day, frac) = Epoch::<Ut1>::from_ns_since_j2000(-1).day_and_fraction();
        assert_eq!(day, -1);
        assert!(frac > 0.999_999_999 && frac < 1.0);
    }

    #[test]
    fn tai_tt_round_trips_exactly() {
        let tai = Epoch::<Tai>::from_ns_since_j2000(1_234_567_891_011);
        assert_eq!(tt_to_tai(tai_to_tt(tai)), tai);
    }

    #[test]
    fn a_day_count_without_a_nanosecond_count_is_refused() {
        // NaN once became J2000 through the saturating casts, and an infinity overflowed.
        let limit = 1_208_925_819_614_629_174_706_176.0_f64;
        for days in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, limit, -limit] {
            assert_eq!(Epoch::<Tt>::from_days_since_j2000(days), None, "{days}");
        }
        let below = f64::from_bits(limit.to_bits() - 1);
        let t = Epoch::<Tt>::from_days_since_j2000(-below).unwrap();
        #[allow(clippy::cast_possible_truncation)] // `below` is integer-valued and below 2^80
        let whole = below as i128;
        assert_eq!(t.ns_since_j2000(), -whole * NS_PER_DAY);
    }

    #[test]
    fn one_julian_century_is_one() {
        let t = Epoch::<Tt>::from_ns_since_j2000(36_525 * NS_PER_DAY);
        assert!((t.julian_centuries() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn julian_centuries_is_linear_through_the_origin() {
        // One point is one equation, and the point above is the one where the answer happens to
        // be 1 -- so a `julian_centuries` that ignored its epoch entirely would satisfy it. These
        // pin the slope and the origin, on both sides of J2000, where every IAU series is
        // evaluated in `t` and a wrong sign there is a century of nutation in the wrong direction.
        for (ns, expected) in [
            (0_i128, 0.0_f64),
            (36_525 * NS_PER_DAY, 1.0),
            (-36_525 * NS_PER_DAY, -1.0),
            (18_262 * NS_PER_DAY, 18_262.0 / 36_525.0),
            (365_250 * NS_PER_DAY, 10.0),
            // Whole days leave the day fraction at zero, where adding it and subtracting it are
            // the same operation. These two carry a fraction, one of them on the Euclidean side
            // of the origin where the day number is negative and the fraction is not.
            (36_525 * NS_PER_DAY + NS_PER_DAY / 4, 36_525.25 / 36_525.0),
            (-NS_PER_DAY / 4, -0.25 / 36_525.0),
        ] {
            let t = Epoch::<Tt>::from_ns_since_j2000(ns);
            assert!(
                (t.julian_centuries() - expected).abs() < 1e-15,
                "{ns} ns gave {} centuries, expected {expected}",
                t.julian_centuries()
            );
        }
    }

    #[test]
    fn day_and_fraction_reconstructs_the_epoch() {
        for ns in [
            0_i128,
            1,
            -1,
            NS_PER_DAY - 1,
            -NS_PER_DAY - 1,
            987_654_321_000_000_000,
        ] {
            let t = Epoch::<Ut1>::from_ns_since_j2000(ns);
            let (day, frac) = t.day_and_fraction();
            #[allow(clippy::cast_possible_truncation)]
            let back = day * NS_PER_DAY + (frac * NS_PER_DAY_F64).round() as i128;
            assert_eq!(back, ns, "round trip failed for {ns}");
        }
    }
}

// Bounded proofs, `just kani`. Each states its domain; overflow and NaN checks are Kani's own.
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Every `f64`, NaN and both infinities included.
    #[kani::proof]
    fn from_days_since_j2000_is_total() {
        let _ = Epoch::<Tt>::from_days_since_j2000(kani::any());
    }

    /// Every `i128`; the fraction lies in `[0, 1)`.
    #[kani::proof]
    fn day_and_fraction_is_total() {
        let (_, frac) = Epoch::<Tai>::from_ns_since_j2000(kani::any()).day_and_fraction();
        assert!((0.0..1.0).contains(&frac));
    }

    /// Every `i128` at least 32.184 s inside the end it moves towards.
    #[kani::proof]
    fn tai_tt_round_trip_is_exact() {
        let tai = Epoch::<Tai>::from_ns_since_j2000(kani::any());
        if tai.ns_since_j2000() <= i128::MAX - TT_MINUS_TAI_NS {
            assert_eq!(tt_to_tai(tai_to_tt(tai)), tai);
        }
        let tt = Epoch::<Tt>::from_ns_since_j2000(kani::any());
        if tt.ns_since_j2000() >= i128::MIN + TT_MINUS_TAI_NS {
            assert_eq!(tai_to_tt(tt_to_tai(tt)), tt);
        }
    }
}
