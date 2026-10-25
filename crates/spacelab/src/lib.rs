//! Space data computation, assembled.
//!
//! The other crates know nothing of one another beyond what they must. This one
//! joins them, and its whole job is the single function that was missing:
//! **turning a UTC label and a table of Earth-orientation measurements into the
//! transform that carries a satellite's state onto the ground.**
//!
//! Until it existed, `teme_to_itrf` took `xp`, `yp` and a UT1 epoch that nothing
//! supplied, and the example propagated the ISS with `xp = yp = 0` and `UT1 = UTC`.
//! That is worth up to 0.9 seconds of Earth rotation — four hundred metres at the
//! equator — and more once leap seconds end on 2027-05-20; a comment saying "real
//! work loads EOP" is not real work.
//!
//! ```no_run
//! # use spacelab::{teme_to_itrf_at, EarthOrientationError, EopSource};
//! # use spacelab_core::UtcDateTime;
//! # use spacelab_eop::EopSeries;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let series = EopSeries::parse_finals2000a(&std::fs::read_to_string("finals2000A.all")?)?;
//! let utc = UtcDateTime::new(2026, 7, 10, 12, 0, 0.0)?;
//!
//! let (transform, quality) = teme_to_itrf_at(utc, EopSource::Series(&series))?;
//! # let _ = (transform, quality);
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]

use core::fmt;

use spacelab_core::{Angle, Epoch, UtcDateTime, time::tai_to_tt};
use spacelab_eop::{EopError, EopQuality, EopSeries};
use spacelab_fa::{LeapError, utc_to_tai};
use spacelab_frames::{Itrf, Orientation, Teme, Transform, teme_to_itrf};

pub use spacelab_core as core_types;
pub use spacelab_earth as earth;
pub use spacelab_eop as eop;
pub use spacelab_fa as fa;
pub use spacelab_frames as frames;
pub use spacelab_sgp4 as sgp4;

/// Nanoseconds in a second, as an exact `f64`.
const NS_PER_SECOND: f64 = 1e9;

/// TT − TAI, in nanoseconds. Exact, by definition.
const TT_MINUS_TAI_NS: i128 = 32_184_000_000;

/// Why the Earth's orientation at an instant could not be established.
#[derive(Debug, Clone, PartialEq)]
pub enum EarthOrientationError {
    /// The leap-second table could not answer for this date.
    Leap(LeapError),
    /// The Earth-orientation series could not answer for this date.
    Eop(EopError),
}

impl fmt::Display for EarthOrientationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Leap(e) => write!(f, "leap seconds: {e}"),
            Self::Eop(e) => write!(f, "earth orientation: {e}"),
        }
    }
}

impl std::error::Error for EarthOrientationError {}

impl From<LeapError> for EarthOrientationError {
    fn from(e: LeapError) -> Self {
        Self::Leap(e)
    }
}

impl From<EopError> for EarthOrientationError {
    fn from(e: EopError) -> Self {
        Self::Eop(e)
    }
}

/// Everything the frame chain needs at one instant, drawn from measurement.
///
/// # The three coordinates, and why they are computed this way
///
/// `UtcDateTime::naive_ns_since_j2000` counts every UTC day as exactly 86400
/// seconds. It is not an instant on any physical scale, but it is the coordinate in
/// which the offsets are *additive*:
///
/// ```text
/// TAI = naive + ΔAT              ΔAT from the leap-second table
/// TT  = TAI + 32.184 s           exact, by definition
/// UT1 = naive + ΔUT1             ΔUT1 measured, interpolated from the IERS series
/// ```
///
/// A `23:59:60` label lands 86400 s into its day, so a leap second needs no special
/// case in TAI. It needs one in UT1: the label's MJD falls in the next day's row,
/// whose `ΔUT1` is one second larger, so UT1 is taken from that row's naive
/// coordinate, one second before the label's.
#[must_use = "the quality of the answer is part of the answer"]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthOrientation {
    /// What [`teme_to_itrf`] consumes.
    pub orientation: Orientation,
    /// Where the `xp`, `yp` and `ΔUT1` above came from, and how well they are known.
    pub quality: EarthOrientationQuality,
}

/// Where the orientation in an [`EarthOrientation`] came from.
///
/// The two arms are not two grades of the same thing. [`Self::Tabulated`] carries the IERS's own
/// verdict on its numbers; [`Self::Absent`] means there were no numbers, so there is no
/// uncertainty to report and none is offered — the field does not exist on that arm rather than
/// being set to zero, because a zero σ on a value nobody measured is the more dangerous answer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EarthOrientationQuality {
    /// Interpolated from a tabulated series.
    Tabulated {
        /// Whether the bracketing days were observed or predicted by the IERS.
        quality: EopQuality,
        /// The IERS uncertainty on `ΔUT1`, in seconds. One second is 465 m at the equator.
        dut1_sigma_seconds: f64,
    },
    /// [`EopSource::NoEop`] was chosen: `UT1 = UTC`, `xp = yp = 0`, `LOD = 0`.
    ///
    /// The error is then `ΔUT1` itself: under 0.9 s while leap seconds held it there, a few tenths
    /// of a second in practice, and growing without them from 2027-05-20 — hundreds of metres of
    /// longitude, and tens of milliseconds on the edge of a low-Earth-orbit pass.
    Absent,
}

impl EarthOrientationQuality {
    /// The IERS verdict on the tabulated days, or `None` if no series was supplied.
    #[must_use]
    pub const fn tabulated(self) -> Option<EopQuality> {
        match self {
            Self::Tabulated { quality, .. } => Some(quality),
            Self::Absent => None,
        }
    }

    /// The IERS uncertainty on `ΔUT1` in seconds, or `None` if no series was supplied.
    #[must_use]
    pub const fn dut1_sigma_seconds(self) -> Option<f64> {
        match self {
            Self::Tabulated {
                dut1_sigma_seconds, ..
            } => Some(dut1_sigma_seconds),
            Self::Absent => None,
        }
    }
}

/// Where [`earth_orientation_at`] should get the Earth's orientation.
///
/// This exists so that computing without Earth orientation is something a caller *says*. Passing
/// zeros for `xp`, `yp` and `ΔUT1` is a legitimate choice — it is what a TLE-only coverage estimate
/// deserves, since the element set's own along-track error dwarfs the frame difference — but it was
/// previously indistinguishable at a call site from having forgotten. `EopSource::NoEop` is a
/// declaration; five literal zeros are not.
#[derive(Debug, Clone, Copy)]
pub enum EopSource<'a> {
    /// A tabulated IERS series, interpolated to the instant.
    Series(&'a EopSeries),
    /// No Earth-orientation data at all: `UT1 = UTC`, `xp = yp = 0`, `LOD = 0`.
    NoEop,
}

/// Establish the Earth's orientation at a UTC instant, from the `source` the caller names.
///
/// The leap-second lookup happens on both arms: `TT` is needed for the TIO locator whether or not
/// there is a pole to locate, and a date outside the table's era is still a date this cannot
/// answer for.
///
/// # Errors
///
/// [`EarthOrientationError::Leap`] before the leap-second table's era, 1972-01-01, or on a
/// `23:59:60` label of a day that had no leap second;
/// [`EarthOrientationError::Eop`] outside the tabulated series, or inside a hole in it.
/// We do not extrapolate the rotation of the Earth. [`EopSource::NoEop`] cannot raise the second.
pub fn earth_orientation_at(
    utc: UtcDateTime,
    source: EopSource<'_>,
) -> Result<EarthOrientation, EarthOrientationError> {
    let naive = utc.naive_ns_since_j2000();

    let tai = utc_to_tai(utc)?;
    let tt = tai_to_tt(tai);

    let Some(eop) = (match source {
        EopSource::Series(series) => Some(series.at_mjd(utc.modified_julian_date())?),
        EopSource::NoEop => None,
    }) else {
        return Ok(EarthOrientation {
            orientation: Orientation {
                ut1: Epoch::from_ns_since_j2000(naive),
                tt,
                xp: Angle::from_radians(0.0),
                yp: Angle::from_radians(0.0),
                lod_seconds: 0.0,
            },
            quality: EarthOrientationQuality::Absent,
        });
    };

    let row_naive = if utc.is_leap_second_label() {
        naive - 1_000_000_000
    } else {
        naive
    };
    #[allow(clippy::cast_possible_truncation)] // |ΔUT1| ≤ 3600 s, so the product is far inside i128
    let ut1_ns = row_naive + (eop.dut1_seconds * NS_PER_SECOND).round() as i128;

    Ok(EarthOrientation {
        orientation: Orientation {
            ut1: Epoch::from_ns_since_j2000(ut1_ns),
            tt,
            xp: eop.xp,
            yp: eop.yp,
            lod_seconds: eop.lod_seconds,
        },
        quality: EarthOrientationQuality::Tabulated {
            quality: eop.quality,
            dut1_sigma_seconds: eop.dut1_sigma_seconds,
        },
    })
}

/// The TEME-to-ITRF transform at a UTC instant, built from the Earth orientation `source` names.
///
/// The returned [`EarthOrientationQuality`] is not decoration. A `ΔUT1` predicted a month ahead
/// carries millisecond error, which is metres of longitude, and an access window computed from it
/// is a forecast rather than a measurement; with [`EopSource::NoEop`] there is no `ΔUT1` at all.
///
/// # Errors
///
/// As [`earth_orientation_at`].
pub fn teme_to_itrf_at(
    utc: UtcDateTime,
    source: EopSource<'_>,
) -> Result<(Transform<Teme, Itrf>, EarthOrientationQuality), EarthOrientationError> {
    let e = earth_orientation_at(utc, source)?;
    let o = e.orientation;
    Ok((
        teme_to_itrf(o.ut1, o.tt, o.xp, o.yp, o.lod_seconds),
        e.quality,
    ))
}

/// `TT − TAI`, in nanoseconds. Exposed so a caller can check the chain by hand.
#[must_use]
pub const fn tt_minus_tai_ns() -> i128 {
    TT_MINUS_TAI_NS
}
