//! The terrestrial reference frame a product states its coordinates are in.
//!
//! An SP3 orbit names its frame on line 1, and an IERS EOP 20 C04 series names the ITRF its pole
//! coordinates are consistent with on line 2. `finals2000A` and a SINEX file name none. Two products
//! may be combined only when both name the same ITRF solution, and [`ReferenceFrame::agrees_with`]
//! is that check, written once here so an orbit's frame and an Earth-orientation series' frame are
//! values of one type.

/// The reference frame a product states its coordinates are in.
///
/// The same orbit format carries IGS orbits in an ITRS realization, ILRS orbits in SLRF2020 and IAC
/// GLONASS orbits in **PZ-90**, which is a different datum. PZ-90.11 agrees with ITRF2008 at the
/// centimetre, but the PZ-90 and PZ-90.02 that older products use differ from ITRF by decimetres to
/// metres, and a station whose coordinates are ITRF differenced against a satellite whose orbit is
/// PZ-90 is wrong by that much with nothing to show for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceFrame {
    /// The product does not say. Not a default frame: an absence.
    #[default]
    Unstated,
    /// A realization of the International Terrestrial Reference System.
    Itrs(Itrs),
    /// PZ-90, the GLONASS datum. Its own realizations differ from ITRF by decimetres before
    /// PZ-90.11, which the five-character SP3 label does not distinguish.
    Pz90,
    /// WGS 84, the GPS broadcast datum. Its later realizations agree with ITRF at the centimetre.
    Wgs84,
    /// A label the reader does not know. Deliberately not treated as ITRS: an unrecognised datum is
    /// not a compatible one.
    Unrecognised,
}

/// A named realization of the ITRS, as an SP3 or SINEX label designates it.
///
/// The analysis-centre labels are realizations *of* an ITRF solution, and successive letters are
/// updates rather than new solutions: `IGS14` and `IGb14` both realize ITRF2014, and `IGS20`,
/// `IGb20` and `IGc20` all realize ITRF2020. [`Itrs::itrf`] is what two labels have to agree on
/// before their coordinates can be differenced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Itrs {
    /// ITRF2008, the IERS solution itself.
    Itrf2008,
    /// The IGS realization of ITRF2008, and its `IGb08` update.
    Igs08,
    /// ITRF2014.
    Itrf2014,
    /// ITRF2020.
    Itrf2020,
    /// The IGS realization of ITRF2014, and its `IGb14` update.
    Igs14,
    /// The IGS realization of ITRF2020, and its `IGb20` / `IGc20` updates.
    Igs20,
    /// The ILRS realization of ITRF2020. A superset: it carries stations that joined the network
    /// after ITRF2020's data cut-off and therefore have no ITRF2020 solution at all.
    Slrf2020,
}

impl Itrs {
    /// The IERS solution this label realizes, which is what two labels must share before their
    /// coordinates may be differenced.
    #[must_use]
    pub const fn itrf(self) -> Self {
        match self {
            Self::Itrf2008 | Self::Igs08 => Self::Itrf2008,
            Self::Itrf2014 | Self::Igs14 => Self::Itrf2014,
            Self::Itrf2020 | Self::Igs20 | Self::Slrf2020 => Self::Itrf2020,
        }
    }
}

impl ReferenceFrame {
    /// Whether coordinates in `self` and `other` may be differenced without a datum transformation.
    ///
    /// [`ReferenceFrame::Unstated`] agrees with nothing, including itself: a product that does not
    /// say what frame it is in has not said it agrees with anything, and answering `true` there
    /// would be the silent default this type exists to remove.
    #[must_use]
    pub const fn agrees_with(self, other: Self) -> bool {
        match (self, other) {
            (Self::Itrs(a), Self::Itrs(b)) => matches!(
                (a.itrf(), b.itrf()),
                (Itrs::Itrf2008, Itrs::Itrf2008)
                    | (Itrs::Itrf2014, Itrs::Itrf2014)
                    | (Itrs::Itrf2020, Itrs::Itrf2020)
            ),
            (Self::Pz90, Self::Pz90) | (Self::Wgs84, Self::Wgs84) => true,
            _ => false,
        }
    }

    /// The ITRS realization, or `None` for a frame that is not one.
    #[must_use]
    pub const fn itrs(self) -> Option<Itrs> {
        match self {
            Self::Itrs(r) => Some(r),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Itrs, ReferenceFrame};

    /// The whole point: a GLONASS orbit in PZ-90 must not be differenced against ITRF coordinates,
    /// and an unstated frame must not be differenced against anything at all.
    #[test]
    fn frames_agree_only_within_one_itrf_solution() {
        let igs20 = ReferenceFrame::Itrs(Itrs::Igs20);
        let itrf2020 = ReferenceFrame::Itrs(Itrs::Itrf2020);
        let slr20 = ReferenceFrame::Itrs(Itrs::Slrf2020);
        let igs14 = ReferenceFrame::Itrs(Itrs::Igs14);

        // Successive IGS updates and the IERS solution they realize are one frame.
        for other in [igs20, itrf2020, slr20] {
            assert!(igs20.agrees_with(other), "{igs20:?} vs {other:?}");
            assert!(other.agrees_with(igs20));
        }
        // A different ITRF solution is a different frame, centimetres apart.
        assert!(!igs20.agrees_with(igs14));
        assert!(!igs14.agrees_with(igs20));
        // A different datum entirely.
        assert!(!igs20.agrees_with(ReferenceFrame::Pz90));
        assert!(!ReferenceFrame::Pz90.agrees_with(igs20));
        assert!(ReferenceFrame::Pz90.agrees_with(ReferenceFrame::Pz90));
        // An absence is not a match, not even with itself.
        assert!(!ReferenceFrame::Unstated.agrees_with(ReferenceFrame::Unstated));
        assert!(!ReferenceFrame::Unstated.agrees_with(igs20));
        assert!(!ReferenceFrame::Unrecognised.agrees_with(ReferenceFrame::Unrecognised));

        assert_eq!(igs20.itrs().map(Itrs::itrf), Some(Itrs::Itrf2020));
        assert_eq!(ReferenceFrame::Pz90.itrs(), None);
    }
}
