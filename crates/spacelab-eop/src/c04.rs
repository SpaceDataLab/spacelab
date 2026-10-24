//! Parsing of the IERS EOP 20 C04 series, `eopc04.1962-now`.
//!
//! # Format
//!
//! `#` header lines, then one fixed-width record of 218 columns per day at 0h UTC. The column
//! ranges below are the byte-by-byte description the IERS publishes beside the file (its `ReadMe`).
//! Unlike `finals2000A`, the series has no predictions and no rapid service: every row is the final
//! combined solution, and its units are seconds of arc and seconds throughout.
//!
//! Two header lines carry what a data row cannot:
//!
//! * the second names the ITRF the pole coordinates are consistent with --
//!   `# EOP (IERS) 20 C04 TIME SERIES  consistent with ITRF 2020 - sampled at 0h UTC` -- which is
//!   the realization of every row, back to 1962, because the IERS reprocessed the whole series;
//! * the fourth names the precession-nutation model the celestial-pole offsets refer to. The IERS
//!   publishes the same series against IAU 1980 too, where columns 63-86 are `dψ, dε` rather than
//!   `dX, dY`. Read as `dX, dY` they are wrong by their whole size, so the line is required.
//!
//! `tests/c04_parse.rs` reads a committed excerpt with its header verbatim, against the published
//! byte ranges and against Bulletin B's pole on the days both hold.

use spacelab_core::{Angle, Itrs, ReferenceFrame};

use crate::finals::{Eop, EopError, EopQuality, EopSample, EopSeries, number};

/// Zero-based, half-open column ranges of the 20 C04 record.
mod col {
    use core::ops::Range;

    pub const MJD: Range<usize> = 16..26;
    pub const XP: Range<usize> = 26..38;
    pub const YP: Range<usize> = 38..50;
    pub const DUT1: Range<usize> = 50..62;
    pub const DX: Range<usize> = 62..74;
    pub const DY: Range<usize> = 74..86;
    pub const LOD: Range<usize> = 110..122;
    pub const XP_SIGMA: Range<usize> = 122..134;
    pub const YP_SIGMA: Range<usize> = 134..146;
    pub const DUT1_SIGMA: Range<usize> = 146..158;
}

/// The header line a file must carry before its first record: the model `dX, dY` refer to.
const MODEL_LINE: &str = "Reference Precession-Nutation Model: IAU 2000";

/// The ITRF a `consistent with ITRF NNNN` header names. An unknown year is not a default.
fn stated_frame(line: &str) -> Option<ReferenceFrame> {
    let (_, rest) = line.split_once("consistent with ITRF")?;
    let year: String = rest
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    Some(match year.as_str() {
        "2008" => ReferenceFrame::Itrs(Itrs::Itrf2008),
        "2014" => ReferenceFrame::Itrs(Itrs::Itrf2014),
        "2020" => ReferenceFrame::Itrs(Itrs::Itrf2020),
        _ => ReferenceFrame::Unrecognised,
    })
}

impl EopSeries {
    /// Parse the contents of an IERS EOP 20 C04 file (`eopc04.1962-now`).
    ///
    /// Every row is final. The frame stated on the header's ITRF line becomes
    /// [`EopSeries::frame`]; a file without that line reads as [`ReferenceFrame::Unstated`].
    ///
    /// # Errors
    ///
    /// [`EopError::Header`] if no precession-nutation model line precedes the first record, or the
    /// line names a model other than IAU 2000; [`EopError::Field`] on a malformed number;
    /// [`EopError::NotMonotonic`] if the days are out of order; [`EopError::Empty`] if no record
    /// remains.
    pub fn parse_c04(text: &str) -> Result<Self, EopError> {
        let mut samples: Vec<EopSample> = Vec::new();
        let mut frame = ReferenceFrame::Unstated;
        let mut model_checked = false;

        for (index, line) in text.lines().enumerate() {
            let line_no = index + 1;
            if let Some(header) = line.strip_prefix('#') {
                if let Some(stated) = stated_frame(header) {
                    frame = stated;
                }
                if header.contains("Precession-Nutation Model") {
                    if header.trim() != MODEL_LINE {
                        return Err(EopError::Header {
                            line: line_no,
                            expected: MODEL_LINE,
                        });
                    }
                    model_checked = true;
                }
                continue;
            }
            if line.trim().is_empty() {
                continue;
            }
            if !model_checked {
                return Err(EopError::Header {
                    line: line_no,
                    expected: MODEL_LINE,
                });
            }

            let mjd = number(line, col::MJD, line_no)?;
            let values = Eop {
                xp: Angle::from_arcseconds(number(line, col::XP, line_no)?),
                yp: Angle::from_arcseconds(number(line, col::YP, line_no)?),
                dut1_seconds: number(line, col::DUT1, line_no)?,
                lod_seconds: number(line, col::LOD, line_no)?,
                dx: Angle::from_arcseconds(number(line, col::DX, line_no)?),
                dy: Angle::from_arcseconds(number(line, col::DY, line_no)?),
                quality: EopQuality::Final,
                xp_sigma: Angle::from_arcseconds(number(line, col::XP_SIGMA, line_no)?),
                yp_sigma: Angle::from_arcseconds(number(line, col::YP_SIGMA, line_no)?),
                dut1_sigma_seconds: number(line, col::DUT1_SIGMA, line_no)?,
            };

            if samples.last().is_some_and(|prev| mjd <= prev.mjd) {
                return Err(EopError::NotMonotonic { line: line_no });
            }
            // C04 has no rapid service: the row is the combined solution, which `finals2000A`
            // calls Bulletin B, and `bulletin_a` repeats it because that field is never empty.
            samples.push(EopSample {
                mjd,
                bulletin_a: values,
                bulletin_b: Some(values),
            });
        }

        if samples.is_empty() {
            return Err(EopError::Empty);
        }
        Ok(Self { samples, frame })
    }
}

#[cfg(test)]
mod tests {
    use super::stated_frame;
    use spacelab_core::{Itrs, ReferenceFrame};

    #[test]
    fn the_itrf_line_names_its_solution_and_nothing_else_is_guessed() {
        let line = " EOP (IERS) 20 C04 TIME SERIES  consistent with ITRF 2020 - sampled at 0h UTC";
        assert_eq!(
            stated_frame(line),
            Some(ReferenceFrame::Itrs(Itrs::Itrf2020))
        );
        assert_eq!(
            stated_frame(" EOP (IERS) 14 C04 TIME SERIES  consistent with ITRF 2014"),
            Some(ReferenceFrame::Itrs(Itrs::Itrf2014))
        );
        assert_eq!(
            stated_frame(" consistent with ITRF 2008"),
            Some(ReferenceFrame::Itrs(Itrs::Itrf2008))
        );
        assert_eq!(
            stated_frame(" consistent with ITRF 2030"),
            Some(ReferenceFrame::Unrecognised)
        );
        assert_eq!(
            stated_frame(" consistent with ITRF"),
            Some(ReferenceFrame::Unrecognised)
        );
        assert_eq!(stated_frame(" Contact: christian.bizouard@obspm.fr"), None);
    }
}
