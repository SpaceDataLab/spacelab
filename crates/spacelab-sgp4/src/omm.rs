//! CCSDS Orbit Mean-Elements Message ingest, in the two encodings that need no parser dependency.
//!
//! # Why this exists when [`Satellite::from_tle`] already works
//!
//! The two-line element set is a punched-card format with 69 columns per line, and the catalogue
//! outgrew it. Three consequences, in the order they bite:
//!
//! * **The catalog number field is five digits.** Objects past 99 999 cannot be named in a TLE at
//!   all. The workaround in circulation is Alpha-5, which re-reads the leading digit as a letter and
//!   buys 240 000 more slots at the price of a format that is no longer numeric; OMM simply carries
//!   `NORAD_CAT_ID` as an integer.
//! * **`BSTAR`, `MEAN_MOTION_DOT` and `MEAN_MOTION_DDOT` are printed into fixed-width exponential
//!   fields** that drop digits the fit produced. OMM prints them at source precision.
//! * **`MEAN_ELEMENT_THEORY` has nowhere to live in 138 columns.** A TLE is silent about which
//!   propagator its elements were fitted for, so a caller cannot tell SGP4 elements from SGP4-XP
//!   elements, and both look like orbits. OMM states it, and [`parse_omm_kvn`] refuses anything this
//!   crate does not implement rather than propagating it plausibly.
//!
//! # The two encodings, and the two that are missing
//!
//! Celestrak serves the same query as `TLE`, `3LE`, `CSV`, `KVN`, `XML`, `JSON` and `JSON-PRETTY`.
//! CSV and KVN are line-oriented and parse with the same machinery as the RINEX, SP3, BLQ and ICGEM
//! readers already in this workspace — no dependency, which is why they are the two implemented here.
//!
//! XML (per CCSDS 505.0-B-3) would want an XML reader; nothing in the workspace has one, so it is a
//! new dependency and a separate decision. JSON is subtler: `serde_json` is already compiled, pulled
//! in transitively by the `sgp4` crate's default features, so the marginal build cost is zero — but
//! a direct `Cargo.toml` entry is still a dependency decision rather than a free ride, and `Elements`
//! already carries the `serde` renames a caller would need to do it themselves in three lines.
//!
//! The size argument runs the other way and is worth knowing before choosing: a 3LE record is ~168
//! characters, the same record in XML is ~1 200, and JSON is roughly three times CSV.
//!
//! [`Satellite::from_tle`]: crate::Satellite::from_tle

use core::fmt;
use core::str::FromStr;

use sgp4::chrono::{NaiveDate, NaiveDateTime};
use sgp4::{Classification, Elements};

/// Why an OMM record could not be read.
///
/// Every variant names the record it came from — records are numbered from 1 in file order — and the
/// CCSDS keyword at fault. It deliberately does not echo the offending text: this parser's input is
/// a network download, and an error message is a place untrusted bytes end up in a log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmmError {
    /// The CSV had no header row to name its columns.
    MissingHeader,
    /// A record is missing a keyword SGP4 cannot be evaluated without.
    MissingField {
        /// Record number, counting from 1 in file order.
        record: usize,
        /// The CCSDS keyword.
        field: &'static str,
    },
    /// A keyword is present but its value does not parse as the type CCSDS gives it.
    MalformedField {
        /// Record number, counting from 1 in file order.
        record: usize,
        /// The CCSDS keyword.
        field: &'static str,
    },
    /// A CSV row has a different number of fields than the header has columns.
    ///
    /// Separate from [`Self::MalformedField`] because it is a statement about the file rather than
    /// about a value: every subsequent column of that row is shifted, so nothing in it can be
    /// trusted, including the fields that happen to still parse.
    RowWidthMismatch {
        /// Record number, counting from 1 in file order.
        record: usize,
        /// Columns the header declared.
        expected: usize,
        /// Fields the row actually carried.
        found: usize,
    },
    /// The record declares mean elements this crate does not propagate.
    ///
    /// `MEAN_ELEMENT_THEORY` is the field a TLE cannot carry, and honouring it is the reason to
    /// prefer OMM. SGP4-XP in particular reads its drag parameter from `BTERM` rather than `BSTAR`
    /// and is a different model; feeding its elements to SGP4 produces an orbit-shaped wrong answer.
    UnsupportedTheory {
        /// Record number, counting from 1 in file order.
        record: usize,
    },
}

impl fmt::Display for OmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingHeader => write!(f, "the OMM CSV has no header row"),
            Self::MissingField { record, field } => {
                write!(f, "record {record} has no {field}")
            }
            Self::MalformedField { record, field } => {
                write!(f, "record {record} has an unreadable {field}")
            }
            Self::RowWidthMismatch {
                record,
                expected,
                found,
            } => write!(
                f,
                "record {record} has {found} fields where the header declares {expected}"
            ),
            Self::UnsupportedTheory { record } => write!(
                f,
                "record {record} declares a MEAN_ELEMENT_THEORY this crate does not propagate"
            ),
        }
    }
}

impl core::error::Error for OmmError {}

/// Read a CCSDS OMM in Celestrak's CSV encoding.
///
/// The first line names the columns and the rest are records, in the RFC 4180 sense: a field may be
/// double-quoted, and a quoted field may contain commas and doubled quotes. Column *order* is not
/// assumed — the header is what binds a name to a position, so a file that reorders or omits columns
/// this crate does not read still parses.
///
/// # Errors
///
/// [`OmmError::MissingHeader`] for an empty input, and otherwise the first record-level failure:
/// see [`OmmError`].
pub fn parse_omm_csv(text: &str) -> Result<Vec<Elements>, OmmError> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = split_csv(lines.next().ok_or(OmmError::MissingHeader)?)
        .into_iter()
        .map(str::trim)
        .collect();

    let mut out = Vec::new();
    for (index, line) in lines.enumerate() {
        let record = index + 1;
        let values = split_csv(line);
        if values.len() != header.len() {
            return Err(OmmError::RowWidthMismatch {
                record,
                expected: header.len(),
                found: values.len(),
            });
        }
        let mut fields = Fields::default();
        for (name, value) in header.iter().zip(values) {
            fields.set(name, value);
        }
        out.push(fields.finish(record)?);
    }
    Ok(out)
}

/// Read a CCSDS OMM in KVN (keyword-value notation).
///
/// A KVN file is a concatenation of whole messages, each opening with `CCSDS_OMM_VERS`, which is
/// what delimits records here. Blank lines and `COMMENT` lines are skipped, and a keyword with an
/// empty value counts as absent — Celestrak emits `CREATION_DATE  =` with nothing after it.
///
/// # Errors
///
/// The first record-level failure; see [`OmmError`].
pub fn parse_omm_kvn(text: &str) -> Result<Vec<Elements>, OmmError> {
    let mut out = Vec::new();
    let mut fields = Fields::default();
    let mut open = false;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("COMMENT") {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let (key, value) = (key.trim(), value.trim());
        if key == "CCSDS_OMM_VERS" {
            if open {
                out.push(core::mem::take(&mut fields).finish(out.len() + 1)?);
            }
            open = true;
            continue;
        }
        if open {
            fields.set(key, value);
        }
    }
    if open {
        out.push(fields.finish(out.len() + 1)?);
    }
    Ok(out)
}

/// Split one CSV line into fields, honouring RFC 4180 double-quoting.
///
/// Quoting matters for exactly one column in practice and it is enough: `OBJECT_NAME` is free text
/// from the satellite catalogue, and a name containing a comma would otherwise shift every field
/// after it — a failure that lands as a plausible orbit rather than as a parse error.
///
/// The returned slices have their surrounding quotes removed; a doubled `""` *inside* a quoted field
/// is left as-is here and unescaped only where the value becomes an owned string, because no numeric
/// column can contain one.
fn split_csv(line: &str) -> Vec<&str> {
    let bytes = line.as_bytes();
    let mut fields = Vec::new();
    let mut start = 0;
    let mut quoted = false;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'"' => quoted = !quoted,
            b',' if !quoted => {
                fields.push(unquote(&line[start..i]));
                start = i + 1;
            }
            _ => {}
        }
    }
    fields.push(unquote(&line[start..]));
    fields
}

/// Strip one layer of surrounding double quotes, if the field is quoted at all.
fn unquote(field: &str) -> &str {
    let trimmed = field.trim();
    trimmed
        .strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(trimmed)
}

/// One record's keywords, as raw text, before any of them is given a type.
///
/// Both encodings converge here so that the map from CCSDS keyword to [`Elements`] field exists in
/// one place: adding a keyword to the CSV reader and forgetting it in the KVN reader is the failure
/// this shape makes unrepresentable.
#[derive(Default)]
struct Fields<'a> {
    object_name: Option<&'a str>,
    object_id: Option<&'a str>,
    norad_cat_id: Option<&'a str>,
    classification_type: Option<&'a str>,
    epoch: Option<&'a str>,
    mean_motion: Option<&'a str>,
    eccentricity: Option<&'a str>,
    inclination: Option<&'a str>,
    ra_of_asc_node: Option<&'a str>,
    arg_of_pericenter: Option<&'a str>,
    mean_anomaly: Option<&'a str>,
    ephemeris_type: Option<&'a str>,
    element_set_no: Option<&'a str>,
    rev_at_epoch: Option<&'a str>,
    bstar: Option<&'a str>,
    mean_motion_dot: Option<&'a str>,
    mean_motion_ddot: Option<&'a str>,
    mean_element_theory: Option<&'a str>,
}

impl<'a> Fields<'a> {
    /// Record one keyword, ignoring anything this crate does not read.
    ///
    /// Trimming and the empty-value rule live here rather than in each encoding's reader, because
    /// the two disagree otherwise: KVN writes a keyword it has no value for as `CREATION_DATE  =`,
    /// while CSV writes the same absence as a zero-width field between two commas. Deciding once
    /// that both mean *absent* is what makes a missing `BSTAR` the same error either way.
    fn set(&mut self, key: &str, value: &'a str) {
        let value = value.trim();
        if value.is_empty() {
            return;
        }
        let slot = match key {
            "OBJECT_NAME" => &mut self.object_name,
            "OBJECT_ID" => &mut self.object_id,
            "NORAD_CAT_ID" => &mut self.norad_cat_id,
            "CLASSIFICATION_TYPE" => &mut self.classification_type,
            "EPOCH" => &mut self.epoch,
            "MEAN_MOTION" => &mut self.mean_motion,
            "ECCENTRICITY" => &mut self.eccentricity,
            "INCLINATION" => &mut self.inclination,
            "RA_OF_ASC_NODE" => &mut self.ra_of_asc_node,
            "ARG_OF_PERICENTER" => &mut self.arg_of_pericenter,
            "MEAN_ANOMALY" => &mut self.mean_anomaly,
            "EPHEMERIS_TYPE" => &mut self.ephemeris_type,
            "ELEMENT_SET_NO" => &mut self.element_set_no,
            "REV_AT_EPOCH" => &mut self.rev_at_epoch,
            "BSTAR" => &mut self.bstar,
            "MEAN_MOTION_DOT" => &mut self.mean_motion_dot,
            "MEAN_MOTION_DDOT" => &mut self.mean_motion_ddot,
            "MEAN_ELEMENT_THEORY" => &mut self.mean_element_theory,
            _ => return,
        };
        *slot = Some(value);
    }

    fn finish(self, record: usize) -> Result<Elements, OmmError> {
        if let Some(theory) = self.mean_element_theory {
            // Celestrak writes `SGP/SGP4`; the CCSDS registry's own name is `SGP4`. Anything else —
            // SGP4-XP, DSST, USM — is a different propagator and this crate has only the one.
            let theory = theory.trim();
            if !theory.eq_ignore_ascii_case("SGP4") && !theory.eq_ignore_ascii_case("SGP/SGP4") {
                return Err(OmmError::UnsupportedTheory { record });
            }
        }

        let epoch = self
            .epoch
            .ok_or(OmmError::MissingField {
                record,
                field: "EPOCH",
            })
            .and_then(|t| {
                parse_epoch(t).ok_or(OmmError::MalformedField {
                    record,
                    field: "EPOCH",
                })
            })?;

        // Optional in CCSDS and unused by the model: they travel with the record so that a caller
        // re-emitting it does not silently invent values, and default to what a TLE would print
        // when the producer left them out.
        let classification = match self.classification_type.map(str::trim) {
            None | Some("U") => Classification::Unclassified,
            Some("C") => Classification::Classified,
            Some("S") => Classification::Secret,
            Some(_) => {
                return Err(OmmError::MalformedField {
                    record,
                    field: "CLASSIFICATION_TYPE",
                });
            }
        };

        Ok(Elements {
            object_name: self.object_name.map(unescape_quotes),
            international_designator: self.object_id.map(unescape_quotes),
            norad_id: required(self.norad_cat_id, "NORAD_CAT_ID", record)?,
            classification,
            datetime: epoch,
            // Fitted quantities, all three of them: a missing BSTAR defaulted to zero is a satellite
            // with no drag, which is not a parse failure anywhere downstream and is wrong by metres
            // per revolution in LEO. Require them rather than invent them.
            mean_motion_dot: required(self.mean_motion_dot, "MEAN_MOTION_DOT", record)?,
            mean_motion_ddot: required(self.mean_motion_ddot, "MEAN_MOTION_DDOT", record)?,
            drag_term: required(self.bstar, "BSTAR", record)?,
            element_set_number: optional(self.element_set_no, "ELEMENT_SET_NO", record)?,
            inclination: required(self.inclination, "INCLINATION", record)?,
            right_ascension: required(self.ra_of_asc_node, "RA_OF_ASC_NODE", record)?,
            eccentricity: required(self.eccentricity, "ECCENTRICITY", record)?,
            argument_of_perigee: required(self.arg_of_pericenter, "ARG_OF_PERICENTER", record)?,
            mean_anomaly: required(self.mean_anomaly, "MEAN_ANOMALY", record)?,
            mean_motion: required(self.mean_motion, "MEAN_MOTION", record)?,
            revolution_number: optional(self.rev_at_epoch, "REV_AT_EPOCH", record)?,
            ephemeris_type: optional(self.ephemeris_type, "EPHEMERIS_TYPE", record)?,
        })
    }
}

/// Undo RFC 4180's `""` escape, which survives [`split_csv`] because no numeric column can hold one.
fn unescape_quotes(field: &str) -> String {
    field.replace("\"\"", "\"")
}

fn required<T: FromStr>(
    value: Option<&str>,
    field: &'static str,
    record: usize,
) -> Result<T, OmmError> {
    value
        .ok_or(OmmError::MissingField { record, field })?
        .trim()
        .parse()
        .map_err(|_| OmmError::MalformedField { record, field })
}

fn optional<T: FromStr + Default>(
    value: Option<&str>,
    field: &'static str,
    record: usize,
) -> Result<T, OmmError> {
    match value.map(str::trim).filter(|v| !v.is_empty()) {
        None => Ok(T::default()),
        Some(v) => v
            .parse()
            .map_err(|_| OmmError::MalformedField { record, field }),
    }
}

/// Parse a CCSDS ASCII time code into a naive UTC label.
///
/// Both codes the standard admits are accepted: `YYYY-MM-DDThh:mm:ss[.d…]` and the day-of-year form
/// `YYYY-DDDThh:mm:ss[.d…]`, each with an optional trailing `Z`. Fractional seconds past the ninth
/// digit are dropped, which is the resolution the return type holds.
///
/// A `:60` second label is rejected. This crate propagates on the naive coordinate — every UTC day
/// exactly 86 400 seconds, matching the convention the elements were fitted under — so an epoch
/// inside a leap second has no representation here, and refusing it beats rounding it into one.
fn parse_epoch(text: &str) -> Option<NaiveDateTime> {
    let text = text.trim().trim_end_matches(['Z', 'z']);
    let (date_text, time_text) = text.split_once('T')?;

    let mut date_parts = date_text.split('-');
    let year: i32 = date_parts.next()?.parse().ok()?;
    let second_part: u32 = date_parts.next()?.parse().ok()?;
    let date = match date_parts.next() {
        Some(day) => NaiveDate::from_ymd_opt(year, second_part, day.parse().ok()?)?,
        None => NaiveDate::from_yo_opt(year, second_part)?,
    };
    if date_parts.next().is_some() {
        return None;
    }

    let mut time_parts = time_text.split(':');
    let hour: u32 = time_parts.next()?.parse().ok()?;
    let minute: u32 = time_parts.next()?.parse().ok()?;
    let seconds = time_parts.next()?;
    if time_parts.next().is_some() {
        return None;
    }

    let (whole, fraction) = match seconds.split_once('.') {
        Some((whole, fraction)) => {
            if fraction.is_empty() || !fraction.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let mut nanoseconds = 0_u32;
            for i in 0..9 {
                let digit = fraction
                    .as_bytes()
                    .get(i)
                    .map_or(0, |b| u32::from(b - b'0'));
                nanoseconds = nanoseconds * 10 + digit;
            }
            (whole, nanoseconds)
        }
        None => (seconds, 0),
    };
    date.and_hms_nano_opt(hour, minute, whole.parse().ok()?, fraction)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    const CSV: &str = "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,\
RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,\
ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT\n\
ISS (ZARYA),1998-067A,2026-08-14T19:49:00.316704,15.49444701,.00075677,51.6328,9.8801,46.9314,\
313.2307,0,U,25544,999,58084,.10032188E-3,.5167E-4,0\n";

    #[test]
    fn a_csv_record_carries_every_field_the_model_reads() {
        let elements = parse_omm_csv(CSV).unwrap();
        assert_eq!(elements.len(), 1);
        let e = &elements[0];
        assert_eq!(e.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(e.international_designator.as_deref(), Some("1998-067A"));
        assert_eq!(e.norad_id, 25544);
        assert_eq!(e.classification, Classification::Unclassified);
        assert!((e.mean_motion - 15.494_447_01).abs() < 1e-12);
        assert!((e.eccentricity - 0.000_756_77).abs() < 1e-12);
        assert!((e.drag_term - 0.000_100_321_88).abs() < 1e-16);
        assert!((e.mean_motion_dot - 5.167e-5).abs() < 1e-16);
        assert_eq!(e.revolution_number, 58084);
        assert_eq!(e.element_set_number, 999);
    }

    #[test]
    fn the_two_encodings_of_one_record_parse_identically() {
        let kvn = "CCSDS_OMM_VERS = 2.0\n\
CREATION_DATE  = \n\
ORIGINATOR     = \n\
\n\
OBJECT_NAME    = ISS (ZARYA)\n\
OBJECT_ID      = 1998-067A\n\
CENTER_NAME    = EARTH\n\
REF_FRAME      = TEME\n\
TIME_SYSTEM    = UTC\n\
MEAN_ELEMENT_THEORY = SGP/SGP4\n\
\n\
EPOCH          = 2026-08-14T19:49:00.316704\n\
MEAN_MOTION    = 15.49444701\n\
ECCENTRICITY   = .00075677\n\
INCLINATION    = 51.6328\n\
RA_OF_ASC_NODE = 9.8801\n\
ARG_OF_PERICENTER = 46.9314\n\
MEAN_ANOMALY   = 313.2307\n\
\n\
EPHEMERIS_TYPE = 0\n\
CLASSIFICATION_TYPE = U\n\
NORAD_CAT_ID   = 25544\n\
ELEMENT_SET_NO = 999\n\
REV_AT_EPOCH   = 58084\n\
BSTAR          = .10032188E-3\n\
MEAN_MOTION_DOT = .5167E-4\n\
MEAN_MOTION_DDOT = 0\n";
        assert_eq!(parse_omm_kvn(kvn).unwrap(), parse_omm_csv(CSV).unwrap());
    }

    /// The whole point of the format: six digits, which no TLE column can hold.
    #[test]
    fn a_catalog_number_above_the_tle_ceiling_is_read_as_itself() {
        let csv = CSV.replace("25544", "270123");
        let elements = parse_omm_csv(&csv).unwrap();
        assert_eq!(elements[0].norad_id, 270_123);
    }

    #[test]
    fn a_quoted_object_name_does_not_shift_the_columns_after_it() {
        let csv = CSV.replace("ISS (ZARYA)", "\"COSMOS 2251, DEB\"");
        let elements = parse_omm_csv(&csv).unwrap();
        assert_eq!(elements[0].object_name.as_deref(), Some("COSMOS 2251, DEB"));
        assert_eq!(elements[0].norad_id, 25544);
    }

    #[test]
    fn a_theory_this_crate_does_not_implement_is_refused() {
        let kvn = "CCSDS_OMM_VERS = 2.0\n\
MEAN_ELEMENT_THEORY = SGP4-XP\n\
EPOCH = 2026-08-14T19:49:00\n\
MEAN_MOTION = 15.49444701\n\
ECCENTRICITY = .00075677\n\
INCLINATION = 51.6328\n\
RA_OF_ASC_NODE = 9.8801\n\
ARG_OF_PERICENTER = 46.9314\n\
MEAN_ANOMALY = 313.2307\n\
NORAD_CAT_ID = 25544\n\
BSTAR = .10032188E-3\n\
MEAN_MOTION_DOT = .5167E-4\n\
MEAN_MOTION_DDOT = 0\n";
        assert_eq!(
            parse_omm_kvn(kvn),
            Err(OmmError::UnsupportedTheory { record: 1 })
        );
    }

    #[test]
    fn a_fitted_field_left_out_is_an_error_rather_than_a_zero() {
        let csv = CSV.replace(",.10032188E-3,", ",,");
        assert_eq!(
            parse_omm_csv(&csv),
            Err(OmmError::MissingField {
                record: 1,
                field: "BSTAR"
            })
        );
    }

    #[test]
    fn a_short_row_is_refused_rather_than_read_shifted() {
        let csv = format!("{}\nA,B,C\n", CSV.trim_end());
        assert_eq!(
            parse_omm_csv(&csv),
            Err(OmmError::RowWidthMismatch {
                record: 2,
                expected: 17,
                found: 3
            })
        );
    }

    #[test]
    fn both_ccsds_time_codes_name_the_same_instant() {
        let calendar = parse_epoch("2026-08-14T19:49:00.316704").unwrap();
        let ordinal = parse_epoch("2026-226T19:49:00.316704Z").unwrap();
        assert_eq!(calendar, ordinal);
    }

    #[test]
    fn a_leap_second_label_is_refused_rather_than_rounded() {
        assert!(parse_epoch("2016-12-31T23:59:60.000").is_none());
        assert!(parse_epoch("2016-12-31T23:59:59.000").is_some());
    }

    #[test]
    fn fractional_seconds_are_read_to_the_nanosecond() {
        use sgp4::chrono::Timelike;
        assert_eq!(
            parse_epoch("2026-08-14T19:49:00.123456789")
                .unwrap()
                .nanosecond(),
            123_456_789
        );
        // Fewer digits are right-padded, not left-padded: `.1` is a tenth of a second.
        assert_eq!(
            parse_epoch("2026-08-14T19:49:00.1").unwrap().nanosecond(),
            100_000_000
        );
    }

    #[test]
    fn an_empty_input_is_a_missing_header_rather_than_an_empty_catalogue() {
        assert_eq!(parse_omm_csv(""), Err(OmmError::MissingHeader));
        assert_eq!(parse_omm_kvn("").unwrap(), Vec::new());
    }
}
