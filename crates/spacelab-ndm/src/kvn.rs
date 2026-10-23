//! The keyword-value notation the whole Navigation Data Message family shares.
//!
//! Every NDM in KVN is the same three line shapes — `KEYWORD = VALUE [UNITS]`, a `COMMENT`, and a
//! bare structural marker such as `META_START` — so the scanner is written once here and each
//! message type supplies only its own grammar and its own keyword table.
//!
//! Two decisions live at this level because getting them wrong is invisible downstream.
//!
//! **Units are stripped, never read.** CCSDS 502.0-B-3 §7.4.5 and 508.0-B-1 §3.2.4 make the unit
//! annotation optional and *informative*: a producer may omit it, and where it is present the
//! standard already fixes what the unit must be. Reading it would mean honouring a file that
//! disagrees with the standard, which is how a kilometre becomes a metre. So the annotation is
//! discarded and the standard's unit is applied.
//!
//! **An error names the keyword and the line, never the text.** These parsers read files that
//! arrive over a network from a third party, and an error message is a place untrusted bytes end up
//! in a log. The keyword is a `&'static str` from our own table; the line number is an integer.

use core::fmt;
use core::str::FromStr;

use spacelab_core::{UtcDateTime, calendar};

/// Metres in a kilometre. CCSDS states OPM and OEM positions in km; this workspace is SI throughout.
pub(crate) const M_PER_KM: f64 = 1000.0;

/// Square metres in a square kilometre.
///
/// Every entry of an OPM or OEM covariance takes the same factor, not just the position block:
/// `km²`, `km²/s` and `km²/s²` are each a product of two kilometre-valued factors, so all 21 scale
/// by `1e6`. A CDM covariance is already in metres and takes none.
pub(crate) const M2_PER_KM2: f64 = 1.0e6;

/// The number of entries in the lower triangle of a 6×6 covariance.
pub const COVARIANCE_ENTRIES: usize = 21;

/// Why a Navigation Data Message could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NdmError {
    /// The file does not open with the version keyword this message type requires.
    ///
    /// Checked first and separately from a missing keyword, because handing an OEM to the OPM
    /// reader otherwise fails as "no `EPOCH`" — an error that describes the file rather than the
    /// mistake.
    NotThisMessage {
        /// The version keyword the reader was looking for, e.g. `CCSDS_OPM_VERS`.
        expected: &'static str,
    },
    /// A keyword the standard makes mandatory is absent from the section it belongs to.
    MissingKeyword {
        /// The CCSDS keyword.
        keyword: &'static str,
    },
    /// A keyword is present but its value does not parse as the type CCSDS gives it.
    MalformedValue {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// The CCSDS keyword.
        keyword: &'static str,
    },
    /// A value in a CCSDS ASCII time code does not name an instant on the UTC calendar.
    ///
    /// Separate from [`Self::MalformedValue`] because the failure is semantic rather than
    /// syntactic: `2019-02-29T00:00:00` parses as six integers and is still not a date.
    MalformedEpoch {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// The CCSDS keyword.
        keyword: &'static str,
    },
    /// An OEM ephemeris line does not carry 7 or 10 whitespace-separated fields.
    ///
    /// Seven is epoch plus position and velocity; ten adds the optional acceleration. Anything else
    /// is a truncated or a doubled record, and reading it shifted would land as a plausible orbit.
    MalformedEphemerisRecord {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// Whitespace-separated fields the line actually carried.
        fields: usize,
    },
    /// A block marker closed a block that was not open, or opened one already open.
    UnbalancedBlock {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// The structural keyword at fault, e.g. `META_STOP`.
        keyword: &'static str,
    },
    /// A covariance block ended without its 21 lower-triangle entries.
    ShortCovariance {
        /// Line number the block ended on.
        line: usize,
        /// Entries the block actually carried.
        entries: usize,
    },
    /// An OEM carries no ephemeris segment.
    ///
    /// An empty file is not an empty ephemeris: a caller that reads "no states" acts by doing
    /// nothing, which is the same thing it would do with a valid ephemeris of a stationary object.
    NoSegments,
    /// A CDM does not carry exactly the two objects the standard requires.
    ObjectCount {
        /// `OBJECT = OBJECTn` sections the file actually carried.
        found: usize,
    },
}

impl fmt::Display for NdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NotThisMessage { expected } => {
                write!(f, "the message does not open with {expected}")
            }
            Self::MissingKeyword { keyword } => write!(f, "no {keyword}"),
            Self::MalformedValue { line, keyword } => {
                write!(f, "line {line} has an unreadable {keyword}")
            }
            Self::MalformedEpoch { line, keyword } => {
                write!(
                    f,
                    "line {line} has a {keyword} that is not a UTC calendar instant"
                )
            }
            Self::MalformedEphemerisRecord { line, fields } => write!(
                f,
                "line {line} has {fields} fields where an ephemeris record has 7 or 10"
            ),
            Self::UnbalancedBlock { line, keyword } => {
                write!(
                    f,
                    "line {line} has a {keyword} the grammar does not admit there"
                )
            }
            Self::ShortCovariance { line, entries } => write!(
                f,
                "the covariance block ending at line {line} has {entries} of {COVARIANCE_ENTRIES} entries"
            ),
            Self::NoSegments => write!(f, "the ephemeris message carries no segment"),
            Self::ObjectCount { found } => {
                write!(
                    f,
                    "the conjunction message describes {found} objects rather than 2"
                )
            }
        }
    }
}

impl core::error::Error for NdmError {}

/// One significant line of a KVN message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Item<'a> {
    /// `KEYWORD = VALUE`, with the unit annotation already stripped from the value.
    Pair {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// The keyword, trimmed.
        key: &'a str,
        /// The value, trimmed, without its unit annotation.
        value: &'a str,
    },
    /// A `COMMENT` line, with the keyword removed and the remainder trimmed.
    Comment {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// Everything after `COMMENT`.
        text: &'a str,
    },
    /// A line carrying no `=` — a block marker, or an OEM ephemeris record.
    Bare {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// The line, trimmed.
        text: &'a str,
    },
}

/// Scan a KVN message into its significant lines, dropping blanks.
///
/// A comment is *yielded* rather than dropped, against the instinct to skip it: NASA CARA writes the
/// hard-body radius of a conjunction as `COMMENT HBR = 15.0`, because CDM 1.0 has no keyword for it
/// (508.0-B-1 §3.2, table 3-3). A reader that drops comments cannot compute a collision probability
/// from the file that carries one.
pub(crate) fn scan(text: &str) -> impl Iterator<Item = Item<'_>> {
    text.lines().enumerate().filter_map(|(index, raw)| {
        let line = index + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return None;
        }
        if let Some(rest) = comment_body(trimmed) {
            return Some(Item::Comment { line, text: rest });
        }
        match trimmed.split_once('=') {
            Some((key, value)) => Some(Item::Pair {
                line,
                key: key.trim(),
                value: strip_units(value),
            }),
            None => Some(Item::Bare {
                line,
                text: trimmed,
            }),
        }
    })
}

/// The text of a `COMMENT` line, or `None` if this is not one.
///
/// `COMMENT` on its own is a comment with no body; `COMMENTARY = 3` is a keyword that merely starts
/// with the same letters, so the prefix has to be followed by whitespace or by nothing at all.
fn comment_body(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("COMMENT")?;
    if rest.is_empty() {
        return Some(rest);
    }
    rest.starts_with(char::is_whitespace).then(|| rest.trim())
}

/// Drop a trailing `[UNITS]` annotation from a value.
///
/// Anchored at the *end* rather than at the first bracket, so that a free-text value which happens
/// to contain one — an `OBJECT_NAME` is whatever the catalogue holds — keeps it.
fn strip_units(value: &str) -> &str {
    let value = value.trim();
    match value.rfind('[') {
        Some(open) if value.ends_with(']') => value[..open].trim_end(),
        _ => value,
    }
}

/// Parse a value, naming the keyword and line if it does not.
pub(crate) fn value<T: FromStr>(
    line: usize,
    keyword: &'static str,
    text: &str,
) -> Result<T, NdmError> {
    text.parse()
        .map_err(|_| NdmError::MalformedValue { line, keyword })
}

/// Parse a CCSDS ASCII time code into a UTC calendar instant.
///
/// Both codes CCSDS 502.0-B-3 §7.5.10 admits are accepted: `YYYY-MM-DDThh:mm:ss[.d…]` and the
/// day-of-year form `YYYY-DDDThh:mm:ss[.d…]`, each with an optional trailing `Z`.
///
/// A `:60` second label is *accepted* here, unlike the OMM reader in `spacelab-sgp4`, and the
/// difference is deliberate. That reader hands its epoch to SGP4, which propagates on a coordinate
/// where every UTC day is 86 400 seconds long and a leap second therefore has no representation. An
/// NDM epoch is a label on the UTC calendar and [`UtcDateTime`] represents leap-second labels
/// exactly, so refusing one here would discard an instant the type can hold.
pub(crate) fn epoch(
    line: usize,
    keyword: &'static str,
    text: &str,
) -> Result<UtcDateTime, NdmError> {
    parse_epoch(text).ok_or(NdmError::MalformedEpoch { line, keyword })
}

fn parse_epoch(text: &str) -> Option<UtcDateTime> {
    let text = text.trim().trim_end_matches(['Z', 'z']);
    let (date_text, time_text) = text.split_once('T')?;

    let mut date_parts = date_text.split('-');
    let year: i32 = date_parts.next()?.trim().parse().ok()?;
    let second_field: u32 = date_parts.next()?.trim().parse().ok()?;
    let (month, day) = match date_parts.next() {
        Some(day) => (u8::try_from(second_field).ok()?, day.trim().parse().ok()?),
        None => day_of_year(year, second_field)?,
    };
    if date_parts.next().is_some() {
        return None;
    }

    let mut time_parts = time_text.split(':');
    let hour: u8 = time_parts.next()?.trim().parse().ok()?;
    let minute: u8 = time_parts.next()?.trim().parse().ok()?;
    let second: f64 = time_parts.next()?.trim().parse().ok()?;
    if time_parts.next().is_some() || !second.is_finite() {
        return None;
    }

    UtcDateTime::new(year, month, day, hour, minute, second).ok()
}

/// Turn an ordinal day into a month and a day of that month.
///
/// The roll check at the end is the whole of the domain test, and deliberately the only one. Day 0
/// lands on 31 December of the previous year and day 366 of a common year on 1 January of the next,
/// so both leave the stated year and both fall out of the same comparison — which is what stops
/// `2019-366` reading as 1 January 2020. A separate `1 ≤ ordinal ≤ length` guard in front of it
/// would be a second statement of one rule, with its own leap-year arithmetic to get wrong and
/// nothing able to tell the two apart.
fn day_of_year(year: i32, ordinal: u32) -> Option<(u8, u8)> {
    let days = calendar::days_from_civil(year, 1, 1) + i64::from(ordinal) - 1;
    let (rolled_year, month, day) = calendar::civil_from_days(days);
    (rolled_year == year).then_some((month, day))
}

/// Write a UTC calendar instant in the CCSDS calendar time code.
///
/// The seconds field is emitted at whatever precision the value needs and no more, so that a value
/// read from a file and written back reproduces its own decimal text: `f64`'s `Display` is the
/// shortest decimal that round-trips, and the only thing added here is the leading zero the code
/// requires below ten seconds.
pub(crate) fn write_epoch(out: &mut String, epoch: UtcDateTime) {
    use core::fmt::Write as _;
    let second = epoch.second();
    let pad = if second < 10.0 { "0" } else { "" };
    // Writing into a String is infallible; `fmt::Write` returns Result only because the trait must
    // serve sinks that can fail.
    let _ = write!(
        out,
        "{:04}-{:02}-{:02}T{:02}:{:02}:{pad}{second}",
        epoch.year(),
        epoch.month(),
        epoch.day(),
        epoch.hour(),
        epoch.minute(),
    );
}

/// Collect the lower-triangle entries of a 6×6 covariance from a keyword table.
///
/// The 21 keywords are in the order CCSDS lists them, which is also the order of the lower triangle
/// read row by row, so an index into this table is an index into the triangle.
pub(crate) const OPM_COVARIANCE_KEYWORDS: [&str; COVARIANCE_ENTRIES] = [
    "CX_X",
    "CY_X",
    "CY_Y",
    "CZ_X",
    "CZ_Y",
    "CZ_Z",
    "CX_DOT_X",
    "CX_DOT_Y",
    "CX_DOT_Z",
    "CX_DOT_X_DOT",
    "CY_DOT_X",
    "CY_DOT_Y",
    "CY_DOT_Z",
    "CY_DOT_X_DOT",
    "CY_DOT_Y_DOT",
    "CZ_DOT_X",
    "CZ_DOT_Y",
    "CZ_DOT_Z",
    "CZ_DOT_X_DOT",
    "CZ_DOT_Y_DOT",
    "CZ_DOT_Z_DOT",
];

/// The CDM's own names for the same triangle, in the RTN frame (508.0-B-1 table 3-5).
pub(crate) const CDM_COVARIANCE_KEYWORDS: [&str; COVARIANCE_ENTRIES] = [
    "CR_R",
    "CT_R",
    "CT_T",
    "CN_R",
    "CN_T",
    "CN_N",
    "CRDOT_R",
    "CRDOT_T",
    "CRDOT_N",
    "CRDOT_RDOT",
    "CTDOT_R",
    "CTDOT_T",
    "CTDOT_N",
    "CTDOT_RDOT",
    "CTDOT_TDOT",
    "CNDOT_R",
    "CNDOT_T",
    "CNDOT_N",
    "CNDOT_RDOT",
    "CNDOT_TDOT",
    "CNDOT_NDOT",
];

/// Expand a lower triangle, read row by row, into the full symmetric 6×6 matrix.
#[must_use]
pub fn symmetric_from_lower(lower: &[f64; COVARIANCE_ENTRIES]) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for (row, entries) in out.iter_mut().enumerate() {
        for (column, entry) in entries.iter_mut().enumerate() {
            // Row-major lower triangle: entry (i, j) with i >= j sits at i(i+1)/2 + j. Sorting the
            // pair first is what makes the result symmetric rather than half a matrix.
            let (i, j) = (row.max(column), row.min(column));
            *entry = lower[i * (i + 1) / 2 + j];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    #[test]
    fn a_unit_annotation_is_stripped_and_a_bracket_inside_a_name_is_not() {
        assert_eq!(strip_units("  6655.9942        [km] "), "6655.9942");
        assert_eq!(strip_units("715"), "715");
        // An object name is catalogue free text, and the catalogue does hold brackets.
        assert_eq!(
            strip_units(" COSMOS 2251 [DEB] TRAILER"),
            "COSMOS 2251 [DEB] TRAILER"
        );
        // A value that is nothing but its own units is empty, not the units.
        assert_eq!(strip_units("[km]"), "");
    }

    #[test]
    fn a_keyword_that_merely_starts_with_comment_is_not_one() {
        assert_eq!(comment_body("COMMENT"), Some(""));
        assert_eq!(comment_body("COMMENT HBR = 15.0"), Some("HBR = 15.0"));
        assert_eq!(comment_body("COMMENTARY = 3"), None);
    }

    #[test]
    fn a_comment_carrying_an_equals_sign_stays_a_comment() {
        let items: Vec<_> = scan("COMMENT HBR = 15.0\nX = 1.0\n").collect();
        assert_eq!(
            items,
            vec![
                Item::Comment {
                    line: 1,
                    text: "HBR = 15.0"
                },
                Item::Pair {
                    line: 2,
                    key: "X",
                    value: "1.0"
                },
            ]
        );
    }

    #[test]
    fn the_line_number_counts_the_blank_lines_it_skips() {
        let items: Vec<_> = scan("\n\n\nX = 1.0\n").collect();
        assert_eq!(
            items,
            vec![Item::Pair {
                line: 4,
                key: "X",
                value: "1.0"
            }]
        );
    }

    #[test]
    fn both_ccsds_time_codes_name_the_same_instant() {
        let calendar = parse_epoch("2019-12-18T12:00:00.331").unwrap();
        let ordinal = parse_epoch("2019-352T12:00:00.331Z").unwrap();
        assert_eq!(calendar, ordinal);
        assert_eq!(calendar.month(), 12);
        assert_eq!(calendar.day(), 18);
        assert_eq!(calendar.second(), 0.331);
    }

    #[test]
    fn an_ordinal_day_outside_the_year_does_not_roll_into_the_next_one() {
        assert!(parse_epoch("2019-366T00:00:00").is_none());
        assert!(parse_epoch("2020-366T00:00:00").is_some()); // 2020 is a leap year
        assert!(parse_epoch("2019-000T00:00:00").is_none());
        assert!(parse_epoch("2019-365T00:00:00").is_some());
        // Both ends of the year, and the first day especially: it is the one ordinal a domain test
        // written with the wrong comparison rejects while still refusing 0 and 366.
        let first = parse_epoch("2019-001T00:00:00").unwrap();
        assert_eq!((first.month(), first.day()), (1, 1));
        let last = parse_epoch("2019-365T00:00:00").unwrap();
        assert_eq!((last.month(), last.day()), (12, 31));
    }

    /// A time code is three fields and a finite second, and either failure alone is a refusal.
    ///
    /// The two conditions are joined by `||` precisely so that neither can carry the other: with
    /// `&&` a fourth colon-separated field would be read and discarded, which is how `12:00:00:00`
    /// becomes noon rather than an error.
    #[test]
    fn a_time_code_with_a_fourth_field_or_an_unreadable_second_is_refused() {
        assert!(parse_epoch("2019-12-18T12:00:00:00").is_none());
        assert!(parse_epoch("2019-12-18T12:00:NaN").is_none());
        assert!(parse_epoch("2019-12-18T12:00:inf").is_none());
        assert!(parse_epoch("2019-12-18T12:00:00").is_some());
    }

    /// Each error says something, and says something different.
    ///
    /// These messages are the only account a caller gets of a file it cannot read, and they carry
    /// the keyword and the line rather than the offending text — so an empty or shared rendering is
    /// not a cosmetic failure but the loss of the whole diagnosis.
    #[test]
    fn every_error_renders_and_no_two_render_alike() {
        let rendered = [
            NdmError::NotThisMessage { expected: "X" }.to_string(),
            NdmError::MissingKeyword { keyword: "X" }.to_string(),
            NdmError::MalformedValue {
                line: 7,
                keyword: "X",
            }
            .to_string(),
            NdmError::MalformedEpoch {
                line: 7,
                keyword: "X",
            }
            .to_string(),
            NdmError::MalformedEphemerisRecord { line: 7, fields: 3 }.to_string(),
            NdmError::UnbalancedBlock {
                line: 7,
                keyword: "X",
            }
            .to_string(),
            NdmError::ShortCovariance {
                line: 7,
                entries: 3,
            }
            .to_string(),
            NdmError::NoSegments.to_string(),
            NdmError::ObjectCount { found: 3 }.to_string(),
        ];
        for (index, text) in rendered.iter().enumerate() {
            assert!(
                !text.is_empty(),
                "error {index} renders as the empty string"
            );
            for other in &rendered[index + 1..] {
                assert_ne!(text, other, "two errors render identically: {text:?}");
            }
        }
    }

    #[test]
    fn a_date_the_calendar_does_not_hold_is_refused() {
        assert!(parse_epoch("2019-02-29T00:00:00").is_none());
        assert!(parse_epoch("2020-02-29T00:00:00").is_some());
        assert!(parse_epoch("2019-13-01T00:00:00").is_none());
        assert!(parse_epoch("2019-01-01T24:00:00").is_none());
    }

    /// The one place this reader and the OMM reader in `spacelab-sgp4` deliberately disagree.
    #[test]
    fn a_leap_second_label_is_a_representable_instant_here() {
        let leap = parse_epoch("2016-12-31T23:59:60.0").unwrap();
        assert_eq!(leap.second(), 60.0);
        assert!(leap.is_leap_second_label());
    }

    #[test]
    fn an_epoch_written_back_reproduces_its_own_text() {
        for text in [
            "2019-12-18T12:00:00.331",
            "2021-06-03T00:00:00",
            "2010-03-13T22:37:52.618",
            "2016-12-31T23:59:60",
            "2022-12-18T14:28:15.1172",
            // Both sides of the leading-zero threshold, which is `< 10` and not `<= 10`: a second
            // of exactly ten needs no pad, and padding it writes a four-character seconds field.
            "2019-12-18T12:00:10",
            "2019-12-18T12:00:09.5",
        ] {
            let mut out = String::new();
            write_epoch(&mut out, parse_epoch(text).unwrap());
            assert_eq!(out, text);
        }
    }

    #[test]
    fn the_triangle_expands_into_the_matrix_it_was_read_from() {
        let mut lower = [0.0; COVARIANCE_ENTRIES];
        for (entry, ordinal) in lower.iter_mut().zip(1_u8..) {
            *entry = f64::from(ordinal);
        }
        let full = symmetric_from_lower(&lower);
        // Row-by-row lower triangle: index 0 is (0,0), 1 is (1,0), 2 is (1,1), 3 is (2,0)...
        assert_eq!(full[0][0], 1.0);
        assert_eq!(full[1][0], 2.0);
        assert_eq!(full[1][1], 3.0);
        assert_eq!(full[2][0], 4.0);
        assert_eq!(full[5][5], 21.0);
        for (row, entries) in full.iter().enumerate() {
            for (column, entry) in entries.iter().enumerate() {
                assert_eq!(*entry, full[column][row]);
            }
        }
    }
}
