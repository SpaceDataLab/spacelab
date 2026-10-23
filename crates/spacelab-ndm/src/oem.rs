//! Orbit Ephemeris Message — a segmented time series of Cartesian states (CCSDS 502.0-B-3 §5).
//!
//! An OEM is how an operator hands over a trajectory. It differs from every other message in the
//! family by being *segmented*: `META_START`/`META_STOP` opens a new block of metadata, and a new
//! block exists precisely where the old one stopped describing the data — annex G's own example
//! splits at a trajectory-correction manoeuvre, because the interpolation either side of a burn is
//! not the same interpolation. Flattening the segments away would discard that, so a segment is a
//! first-class object here and a caller that only wants states walks them itself.
//!
//! The reader does not interpolate. `INTERPOLATION` and `INTERPOLATION_DEGREE` are carried through
//! as the producer's *instruction*, and honouring it — Lagrange or Hermite at the stated degree — is
//! the caller's, because the workspace already holds those kernels and a second copy here would be
//! a second thing to keep correct.

use spacelab_core::UtcDateTime;

use crate::kvn::{
    COVARIANCE_ENTRIES, Item, M_PER_KM, M2_PER_KM2, NdmError, epoch, scan, value, write_epoch,
};

/// The version keyword an OEM must open with.
const VERSION_KEYWORD: &str = "CCSDS_OEM_VERS";

/// Fields in an ephemeris line without the optional acceleration: epoch, position, velocity.
const FIELDS_WITHOUT_ACCELERATION: usize = 7;
/// Fields in an ephemeris line that carries the optional acceleration too.
const FIELDS_WITH_ACCELERATION: usize = 10;

/// One state on the ephemeris grid, in SI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EphemerisRecord {
    /// The instant this state is valid at.
    pub epoch: UtcDateTime,
    /// Position \[m].
    pub position: [f64; 3],
    /// Velocity \[m/s].
    pub velocity: [f64; 3],
    /// Acceleration \[m/s²], when the producer states it.
    pub acceleration: Option<[f64; 3]>,
}

/// One covariance on the (independent) covariance grid, in SI.
///
/// Its epochs need not be the ephemeris epochs — annex G's example states two covariances against a
/// grid of many states — which is why they are a separate series rather than a field on
/// [`EphemerisRecord`].
#[derive(Debug, Clone, PartialEq)]
pub struct OemCovariance {
    /// The instant this covariance is valid at.
    pub epoch: UtcDateTime,
    /// `COV_REF_FRAME`, when stated; otherwise the segment's own `REF_FRAME`.
    pub ref_frame: Option<String>,
    /// Lower triangle, row by row, in SI: m², m²/s, m²/s².
    pub lower: [f64; COVARIANCE_ENTRIES],
}

/// One segment's metadata block.
#[derive(Debug, Clone, PartialEq)]
pub struct OemMetadata {
    /// `OBJECT_NAME`.
    pub object_name: String,
    /// `OBJECT_ID`, the international designator.
    pub object_id: String,
    /// `CENTER_NAME`, the body the states are relative to.
    pub center_name: String,
    /// `REF_FRAME`.
    pub ref_frame: String,
    /// `REF_FRAME_EPOCH`, for a frame whose realisation is epoch-dependent.
    pub ref_frame_epoch: Option<UtcDateTime>,
    /// `TIME_SYSTEM`.
    pub time_system: String,
    /// `START_TIME`, the first epoch of the segment.
    pub start_time: UtcDateTime,
    /// `USEABLE_START_TIME`, inside which the producer vouches for the ephemeris.
    pub useable_start_time: Option<UtcDateTime>,
    /// `USEABLE_STOP_TIME`.
    pub useable_stop_time: Option<UtcDateTime>,
    /// `STOP_TIME`, the last epoch of the segment.
    pub stop_time: UtcDateTime,
    /// `INTERPOLATION`, the scheme the producer intends, e.g. `HERMITE`.
    pub interpolation: Option<String>,
    /// `INTERPOLATION_DEGREE`.
    pub interpolation_degree: Option<u32>,
}

/// One `META_START`…`META_STOP` block and the data that follows it.
#[derive(Debug, Clone, PartialEq)]
pub struct OemSegment {
    /// The block's metadata.
    pub metadata: OemMetadata,
    /// States, in file order.
    pub states: Vec<EphemerisRecord>,
    /// Covariances, in file order. Empty unless the segment carries a `COVARIANCE_START` block.
    pub covariances: Vec<OemCovariance>,
}

/// A parsed Orbit Ephemeris Message, in SI throughout.
#[derive(Debug, Clone, PartialEq)]
pub struct Oem {
    /// `CREATION_DATE`.
    pub creation_date: UtcDateTime,
    /// `ORIGINATOR`.
    pub originator: String,
    /// `MESSAGE_ID`, optional since version 3.0.
    pub message_id: Option<String>,
    /// Segments, in file order. Never empty — see [`NdmError::NoSegments`].
    pub segments: Vec<OemSegment>,
}

/// Where in the grammar the reader currently stands.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Before the first `META_START`.
    Header,
    /// Between `META_START` and `META_STOP`.
    Metadata,
    /// After `META_STOP`, reading ephemeris lines.
    Data,
    /// Between `COVARIANCE_START` and `COVARIANCE_STOP`.
    Covariance,
}

/// Read a CCSDS Orbit Ephemeris Message in KVN.
///
/// # Errors
///
/// [`NdmError::NotThisMessage`] if the file does not open with `CCSDS_OEM_VERS`,
/// [`NdmError::NoSegments`] if it carries no `META_START` block, and otherwise the first structural
/// failure; see [`NdmError`].
#[allow(clippy::too_many_lines)]
pub fn parse_oem(text: &str) -> Result<Oem, NdmError> {
    let mut opened = false;
    let mut state = State::Header;

    let mut creation_date = None;
    let mut originator = None;
    let mut message_id = None;
    let mut segments: Vec<OemSegment> = Vec::new();
    let mut metadata = MetadataBuilder::default();
    let mut pending_covariance: Option<CovarianceBuilder> = None;

    for item in scan(text) {
        match item {
            Item::Comment { .. } => {}
            Item::Bare { line, text } => match state {
                State::Header | State::Data if text == "META_START" => {
                    state = State::Metadata;
                    metadata = MetadataBuilder::default();
                }
                State::Data if text == "COVARIANCE_START" => state = State::Covariance,
                State::Data => {
                    let record = ephemeris_record(line, text)?;
                    let segment = segments.last_mut().ok_or(NdmError::UnbalancedBlock {
                        line,
                        keyword: "META_STOP",
                    })?;
                    segment.states.push(record);
                }
                State::Metadata if text == "META_STOP" => {
                    segments.push(OemSegment {
                        metadata: core::mem::take(&mut metadata).finish()?,
                        states: Vec::new(),
                        covariances: Vec::new(),
                    });
                    state = State::Data;
                }
                State::Covariance if text == "COVARIANCE_STOP" => {
                    flush_covariance(&mut pending_covariance, &mut segments, line)?;
                    state = State::Data;
                }
                State::Covariance => {
                    let builder = pending_covariance
                        .as_mut()
                        .ok_or(NdmError::MissingKeyword { keyword: "EPOCH" })?;
                    for token in text.split_whitespace() {
                        if builder.entries.len() == COVARIANCE_ENTRIES {
                            return Err(NdmError::MalformedValue {
                                line,
                                keyword: "COVARIANCE_START",
                            });
                        }
                        let entry: f64 = value(line, "COVARIANCE_START", token)?;
                        builder.entries.push(entry * M2_PER_KM2);
                    }
                }
                State::Header | State::Metadata => {
                    return Err(NdmError::UnbalancedBlock {
                        line,
                        keyword: "META_START",
                    });
                }
            },
            Item::Pair {
                line,
                key,
                value: text,
            } => {
                if key == VERSION_KEYWORD {
                    opened = true;
                    continue;
                }
                if !opened {
                    return Err(NdmError::NotThisMessage {
                        expected: VERSION_KEYWORD,
                    });
                }
                match state {
                    State::Header => match key {
                        "CREATION_DATE" => {
                            creation_date = Some(epoch(line, "CREATION_DATE", text)?);
                        }
                        "ORIGINATOR" => originator = Some(text.to_owned()),
                        "MESSAGE_ID" => message_id = Some(text.to_owned()),
                        _ => {}
                    },
                    State::Metadata => metadata.set(line, key, text)?,
                    State::Covariance => match key {
                        "EPOCH" => {
                            flush_covariance(&mut pending_covariance, &mut segments, line)?;
                            pending_covariance = Some(CovarianceBuilder {
                                epoch: epoch(line, "EPOCH", text)?,
                                ref_frame: None,
                                entries: Vec::with_capacity(COVARIANCE_ENTRIES),
                            });
                        }
                        "COV_REF_FRAME" => {
                            let builder = pending_covariance
                                .as_mut()
                                .ok_or(NdmError::MissingKeyword { keyword: "EPOCH" })?;
                            builder.ref_frame = Some(text.to_owned());
                        }
                        _ => {}
                    },
                    State::Data => {
                        return Err(NdmError::UnbalancedBlock {
                            line,
                            keyword: key_marker(key),
                        });
                    }
                }
            }
        }
    }

    if !opened {
        return Err(NdmError::NotThisMessage {
            expected: VERSION_KEYWORD,
        });
    }
    if state == State::Metadata || state == State::Covariance {
        return Err(NdmError::UnbalancedBlock {
            line: text.lines().count(),
            keyword: if state == State::Metadata {
                "META_STOP"
            } else {
                "COVARIANCE_STOP"
            },
        });
    }
    if segments.is_empty() {
        return Err(NdmError::NoSegments);
    }

    Ok(Oem {
        creation_date: creation_date.ok_or(NdmError::MissingKeyword {
            keyword: "CREATION_DATE",
        })?,
        originator: originator.ok_or(NdmError::MissingKeyword {
            keyword: "ORIGINATOR",
        })?,
        message_id,
        segments,
    })
}

/// A `KEYWORD = VALUE` line where the grammar expects an ephemeris record names itself in the error.
///
/// Only the structural keywords are worth naming; anything else is reported as the generic marker,
/// because a producer's private keyword is not one this crate has a `&'static str` for and echoing
/// it would put file text into a log.
fn key_marker(key: &str) -> &'static str {
    match key {
        "EPOCH" => "EPOCH",
        "COV_REF_FRAME" => "COV_REF_FRAME",
        _ => "META_START",
    }
}

struct CovarianceBuilder {
    epoch: UtcDateTime,
    ref_frame: Option<String>,
    entries: Vec<f64>,
}

fn flush_covariance(
    pending: &mut Option<CovarianceBuilder>,
    segments: &mut [OemSegment],
    line: usize,
) -> Result<(), NdmError> {
    let Some(builder) = pending.take() else {
        return Ok(());
    };
    let entries = builder.entries.len();
    if entries != COVARIANCE_ENTRIES {
        return Err(NdmError::ShortCovariance { line, entries });
    }
    let mut lower = [0.0; COVARIANCE_ENTRIES];
    lower.copy_from_slice(&builder.entries);
    let segment = segments.last_mut().ok_or(NdmError::UnbalancedBlock {
        line,
        keyword: "META_STOP",
    })?;
    segment.covariances.push(OemCovariance {
        epoch: builder.epoch,
        ref_frame: builder.ref_frame,
        lower,
    });
    Ok(())
}

fn ephemeris_record(line: usize, text: &str) -> Result<EphemerisRecord, NdmError> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    if fields.len() != FIELDS_WITHOUT_ACCELERATION && fields.len() != FIELDS_WITH_ACCELERATION {
        return Err(NdmError::MalformedEphemerisRecord {
            line,
            fields: fields.len(),
        });
    }
    let mut components = [0.0; 9];
    for (slot, token) in components.iter_mut().zip(&fields[1..]) {
        *slot = value::<f64>(line, "EPHEMERIS", token)? * M_PER_KM;
    }
    Ok(EphemerisRecord {
        epoch: epoch(line, "EPHEMERIS", fields[0])?,
        position: [components[0], components[1], components[2]],
        velocity: [components[3], components[4], components[5]],
        acceleration: (fields.len() == FIELDS_WITH_ACCELERATION).then_some([
            components[6],
            components[7],
            components[8],
        ]),
    })
}

#[derive(Default)]
struct MetadataBuilder {
    object_name: Option<String>,
    object_id: Option<String>,
    center_name: Option<String>,
    ref_frame: Option<String>,
    ref_frame_epoch: Option<UtcDateTime>,
    time_system: Option<String>,
    start_time: Option<UtcDateTime>,
    useable_start_time: Option<UtcDateTime>,
    useable_stop_time: Option<UtcDateTime>,
    stop_time: Option<UtcDateTime>,
    interpolation: Option<String>,
    interpolation_degree: Option<u32>,
}

impl MetadataBuilder {
    fn set(&mut self, line: usize, key: &str, text: &str) -> Result<(), NdmError> {
        match key {
            "OBJECT_NAME" => self.object_name = Some(text.to_owned()),
            "OBJECT_ID" => self.object_id = Some(text.to_owned()),
            "CENTER_NAME" => self.center_name = Some(text.to_owned()),
            "REF_FRAME" => self.ref_frame = Some(text.to_owned()),
            "REF_FRAME_EPOCH" => self.ref_frame_epoch = Some(epoch(line, "REF_FRAME_EPOCH", text)?),
            "TIME_SYSTEM" => self.time_system = Some(text.to_owned()),
            "START_TIME" => self.start_time = Some(epoch(line, "START_TIME", text)?),
            "USEABLE_START_TIME" => {
                self.useable_start_time = Some(epoch(line, "USEABLE_START_TIME", text)?);
            }
            "USEABLE_STOP_TIME" => {
                self.useable_stop_time = Some(epoch(line, "USEABLE_STOP_TIME", text)?);
            }
            "STOP_TIME" => self.stop_time = Some(epoch(line, "STOP_TIME", text)?),
            "INTERPOLATION" => self.interpolation = Some(text.to_owned()),
            "INTERPOLATION_DEGREE" => {
                self.interpolation_degree = Some(value(line, "INTERPOLATION_DEGREE", text)?);
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(self) -> Result<OemMetadata, NdmError> {
        let missing = |keyword| NdmError::MissingKeyword { keyword };
        Ok(OemMetadata {
            object_name: self.object_name.ok_or_else(|| missing("OBJECT_NAME"))?,
            object_id: self.object_id.ok_or_else(|| missing("OBJECT_ID"))?,
            center_name: self.center_name.ok_or_else(|| missing("CENTER_NAME"))?,
            ref_frame: self.ref_frame.ok_or_else(|| missing("REF_FRAME"))?,
            ref_frame_epoch: self.ref_frame_epoch,
            time_system: self.time_system.ok_or_else(|| missing("TIME_SYSTEM"))?,
            start_time: self.start_time.ok_or_else(|| missing("START_TIME"))?,
            useable_start_time: self.useable_start_time,
            useable_stop_time: self.useable_stop_time,
            stop_time: self.stop_time.ok_or_else(|| missing("STOP_TIME"))?,
            interpolation: self.interpolation,
            interpolation_degree: self.interpolation_degree,
        })
    }
}

/// Write a CCSDS Orbit Ephemeris Message in KVN, at version 3.0.
#[must_use]
pub fn write_oem(oem: &Oem) -> String {
    use core::fmt::Write as _;
    let mut out = String::new();

    let _ = writeln!(out, "{VERSION_KEYWORD} = 3.0");
    let mut date = String::new();
    write_epoch(&mut date, oem.creation_date);
    let _ = writeln!(out, "CREATION_DATE = {date}");
    let _ = writeln!(out, "ORIGINATOR = {}", oem.originator);
    if let Some(id) = &oem.message_id {
        let _ = writeln!(out, "MESSAGE_ID = {id}");
    }

    for segment in &oem.segments {
        let meta = &segment.metadata;
        let _ = writeln!(out, "META_START");
        let _ = writeln!(out, "OBJECT_NAME = {}", meta.object_name);
        let _ = writeln!(out, "OBJECT_ID = {}", meta.object_id);
        let _ = writeln!(out, "CENTER_NAME = {}", meta.center_name);
        let _ = writeln!(out, "REF_FRAME = {}", meta.ref_frame);
        for (key, instant) in [
            ("REF_FRAME_EPOCH", meta.ref_frame_epoch),
            ("START_TIME", Some(meta.start_time)),
            ("USEABLE_START_TIME", meta.useable_start_time),
            ("USEABLE_STOP_TIME", meta.useable_stop_time),
            ("STOP_TIME", Some(meta.stop_time)),
        ] {
            // TIME_SYSTEM has to precede every epoch keyword the standard orders after it, and
            // REF_FRAME_EPOCH precedes TIME_SYSTEM, so the one is emitted inside this loop.
            if key == "START_TIME" {
                let _ = writeln!(out, "TIME_SYSTEM = {}", meta.time_system);
            }
            if let Some(instant) = instant {
                let mut text = String::new();
                write_epoch(&mut text, instant);
                let _ = writeln!(out, "{key} = {text}");
            }
        }
        if let Some(scheme) = &meta.interpolation {
            let _ = writeln!(out, "INTERPOLATION = {scheme}");
        }
        if let Some(degree) = meta.interpolation_degree {
            let _ = writeln!(out, "INTERPOLATION_DEGREE = {degree}");
        }
        let _ = writeln!(out, "META_STOP");

        for record in &segment.states {
            let mut text = String::new();
            write_epoch(&mut text, record.epoch);
            for metres in record.position.iter().chain(&record.velocity) {
                let _ = write!(text, " {}", metres / M_PER_KM);
            }
            if let Some(acceleration) = record.acceleration {
                for metres in acceleration {
                    let _ = write!(text, " {}", metres / M_PER_KM);
                }
            }
            let _ = writeln!(out, "{text}");
        }

        if !segment.covariances.is_empty() {
            let _ = writeln!(out, "COVARIANCE_START");
            for covariance in &segment.covariances {
                let mut text = String::new();
                write_epoch(&mut text, covariance.epoch);
                let _ = writeln!(out, "EPOCH = {text}");
                if let Some(frame) = &covariance.ref_frame {
                    let _ = writeln!(out, "COV_REF_FRAME = {frame}");
                }
                let mut index = 0;
                for row in 0..6 {
                    let mut line = String::new();
                    for _ in 0..=row {
                        let _ = write!(line, " {}", covariance.lower[index] / M2_PER_KM2);
                        index += 1;
                    }
                    let _ = writeln!(out, "{}", line.trim_start());
                }
            }
            let _ = writeln!(out, "COVARIANCE_STOP");
        }
    }

    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    /// CCSDS 502.0-B-3 annex G, figure G-11: two segments, no acceleration, no covariance.
    const TWO_SEGMENTS: &str = include_str!("../tests/fixtures/ccsds_502_g11.oem");
    /// Figure G-13: one segment with two covariance matrices.
    const WITH_COVARIANCE: &str = include_str!("../tests/fixtures/ccsds_502_g13.oem");

    #[test]
    fn the_two_segments_of_the_annex_example_stay_two_segments() {
        let oem = parse_oem(TWO_SEGMENTS).unwrap();
        assert_eq!(oem.originator, "NASA/JPL");
        assert_eq!(oem.segments.len(), 2);
        // The split is at the trajectory-correction manoeuvre: the first segment stops before the
        // second one starts, which is exactly the information flattening them would lose.
        assert_eq!(oem.segments[0].metadata.stop_time.day(), 28);
        assert_eq!(oem.segments[1].metadata.start_time.day(), 28);
        assert_eq!(
            oem.segments[0].metadata.interpolation.as_deref(),
            Some("HERMITE")
        );
        assert_eq!(oem.segments[0].metadata.interpolation_degree, Some(7));
        assert_eq!(oem.segments[0].states.len(), 4);
        assert_eq!(oem.segments[1].states.len(), 4);
    }

    #[test]
    fn an_ephemeris_record_reads_in_metres_and_metres_per_second() {
        let oem = parse_oem(TWO_SEGMENTS).unwrap();
        let first = oem.segments[0].states[0];
        assert_eq!(first.position, [2_789_619.0, -280_045.0, -1_746_755.0]);
        for (component, expected) in first.velocity.iter().zip([4733.72, -2495.86, -1041.95]) {
            assert!(
                (component - expected).abs() < 1e-9,
                "{component} vs {expected}"
            );
        }
        assert!(first.acceleration.is_none());
    }

    #[test]
    fn the_optional_acceleration_is_read_when_the_record_carries_ten_fields() {
        let oem = parse_oem(include_str!("../tests/fixtures/ccsds_502_g12.oem")).unwrap();
        let first = oem.segments[0].states[0];
        assert_eq!(first.acceleration, Some([8.0, 1.0, -159.0]));
        // The last record of the same segment carries a different one, so this is not a constant.
        let last = *oem.segments[0].states.last().unwrap();
        assert_eq!(last.acceleration, Some([-3.0, 0.0, 0.0]));
    }

    /// Two ways a line can fail to be a record, and the reason they are two errors rather than one.
    ///
    /// The annex prints `< intervening data records omitted here >` where it elides rows. That is
    /// *seven* whitespace-separated tokens — exactly the count a record without acceleration has —
    /// so the arity check passes it and the value parse is what refuses it. A truncated record has
    /// the wrong arity instead, and reading it shifted would land as a plausible state. Both must
    /// be refused, and a reader that only checked one of them would admit the other.
    #[test]
    fn a_line_that_is_not_a_record_is_refused_rather_than_read_shifted() {
        const RECORD: &str =
            "2019-12-18T12:02:00.331   2776.033 -336.859 -2008.682    5.63678 -2.33951 -1.94687";

        let elision = TWO_SEGMENTS.replace(RECORD, "   < intervening data records omitted here >");
        assert_eq!(
            parse_oem(&elision),
            Err(NdmError::MalformedValue {
                line: 23,
                keyword: "EPHEMERIS"
            })
        );

        let truncated = TWO_SEGMENTS.replace(RECORD, "2019-12-18T12:02:00.331   2776.033 -336.859");
        assert_eq!(
            parse_oem(&truncated),
            Err(NdmError::MalformedEphemerisRecord {
                line: 23,
                fields: 3
            })
        );

        // Eight fields is neither 7 nor 10: a record with a stray token is not an accelerated one.
        let doubled = TWO_SEGMENTS.replace(RECORD, &format!("{RECORD} 0.008"));
        assert_eq!(
            parse_oem(&doubled),
            Err(NdmError::MalformedEphemerisRecord {
                line: 23,
                fields: 8
            })
        );
    }

    #[test]
    fn two_covariances_on_their_own_grid_are_two_covariances() {
        let oem = parse_oem(WITH_COVARIANCE).unwrap();
        assert_eq!(oem.segments.len(), 1);
        let covariances = &oem.segments[0].covariances;
        assert_eq!(covariances.len(), 2);
        // Neither covariance epoch is an ephemeris epoch, which is why they are a separate series.
        assert_eq!(covariances[1].epoch.day(), 29);
        assert!(
            oem.segments[0]
                .states
                .iter()
                .all(|state| state.epoch != covariances[1].epoch)
        );
        assert_eq!(covariances[0].ref_frame.as_deref(), Some("EME2000"));
        assert!((covariances[0].lower[0] - 3.331_349_4e2).abs() < 1e-6);
        assert!((covariances[1].lower[0] - 3.442_450_5e2).abs() < 1e-6);
    }

    #[test]
    fn a_covariance_block_that_stops_short_is_refused() {
        let text = WITH_COVARIANCE.replace(
            "-3.0413460e-07 -4.9894969e-07 3.5403109e-07       1.8692631e-10   1.0088625e-10   6.2244443e-10\n\nEPOCH = 2019-12-29T21:00:00",
            "EPOCH = 2019-12-29T21:00:00",
        );
        assert!(matches!(
            parse_oem(&text),
            Err(NdmError::ShortCovariance { entries: 15, .. })
        ));
    }

    #[test]
    fn a_message_carrying_no_segment_is_refused_rather_than_read_as_an_empty_ephemeris() {
        assert_eq!(
            parse_oem(
                "CCSDS_OEM_VERS = 3.0\nCREATION_DATE = 2019-11-04T17:22:31\nORIGINATOR = NASA/JPL\n"
            ),
            Err(NdmError::NoSegments)
        );
    }

    #[test]
    fn a_metadata_block_left_open_is_refused() {
        let text = TWO_SEGMENTS.replacen("META_STOP", "", 1);
        assert!(matches!(
            parse_oem(&text),
            Err(NdmError::UnbalancedBlock { .. })
        ));
    }

    /// The useable span is not the ephemeris span, and dropping it silently widens the second.
    ///
    /// Figure G-11 states both, and they differ from `START_TIME` / `STOP_TIME` by ten minutes at the
    /// front and five at the back — the interpolator's warm-up and cool-down. A reader that ignored
    /// the keywords would hand a consumer an ephemeris that looks valid over an interval the producer
    /// said it is not.
    #[test]
    fn the_useable_span_is_read_and_is_not_the_ephemeris_span() {
        let oem = parse_oem(TWO_SEGMENTS).unwrap();
        let meta = &oem.segments[0].metadata;
        assert_eq!(
            meta.useable_start_time,
            Some(UtcDateTime::new(2019, 12, 18, 12, 10, 0.331).unwrap())
        );
        assert_eq!(
            meta.useable_stop_time,
            Some(UtcDateTime::new(2019, 12, 28, 21, 23, 0.331).unwrap())
        );
        assert_eq!(
            meta.start_time,
            UtcDateTime::new(2019, 12, 18, 12, 0, 0.331).unwrap()
        );
        assert_eq!(
            meta.stop_time,
            UtcDateTime::new(2019, 12, 28, 21, 28, 0.331).unwrap()
        );
    }

    #[test]
    fn the_message_identifier_is_read_and_written_back() {
        let oem = parse_oem(WITH_COVARIANCE).unwrap();
        assert_eq!(oem.message_id.as_deref(), Some("OEM 201113719185"));
        assert!(write_oem(&oem).contains("MESSAGE_ID = OEM 201113719185\n"));
        // The other annex figure states no identifier, so this is a read and not a constant.
        assert!(parse_oem(TWO_SEGMENTS).unwrap().message_id.is_none());
    }

    /// `REF_FRAME_EPOCH` reads, survives a round trip, and stays ahead of `TIME_SYSTEM` in the output.
    ///
    /// No annex figure states it, so the message is the G-11 one with the keyword inserted. The
    /// ordering half matters because 502.0-B-3 fixes the metadata keyword order and `REF_FRAME_EPOCH`
    /// is the one epoch that precedes `TIME_SYSTEM`: emitting the time system before it, or once per
    /// epoch keyword, produces a file that no longer round-trips through a strict reader.
    #[test]
    fn a_frame_epoch_is_read_and_written_before_the_time_system() {
        let text = TWO_SEGMENTS.replacen(
            "REF_FRAME           = EME2000\n",
            "REF_FRAME           = EME2000\nREF_FRAME_EPOCH     = 2019-12-18T00:00:00\n",
            1,
        );
        let oem = parse_oem(&text).unwrap();
        assert_eq!(
            oem.segments[0].metadata.ref_frame_epoch,
            Some(UtcDateTime::new(2019, 12, 18, 0, 0, 0.0).unwrap())
        );
        assert!(oem.segments[1].metadata.ref_frame_epoch.is_none());

        let written = write_oem(&oem);
        let keywords: Vec<&str> = written
            .lines()
            .filter_map(|line| line.split(" = ").next())
            .filter(|key| {
                matches!(
                    *key,
                    "REF_FRAME" | "REF_FRAME_EPOCH" | "TIME_SYSTEM" | "START_TIME" | "STOP_TIME"
                )
            })
            .collect();
        assert_eq!(
            keywords,
            [
                "REF_FRAME",
                "REF_FRAME_EPOCH",
                "TIME_SYSTEM",
                "START_TIME",
                "STOP_TIME",
                "REF_FRAME",
                "TIME_SYSTEM",
                "START_TIME",
                "STOP_TIME",
            ]
        );
        assert_eq!(parse_oem(&written).unwrap(), oem);
    }

    /// Both blocks that can be left open, and the keyword each one is missing.
    ///
    /// The two are one `if` in the reader and they have to be told apart: a file whose metadata never
    /// closes and a file whose covariance never closes fail at the same place, and reporting the
    /// wrong one of the two sends the reader of the log to the wrong half of the message.
    #[test]
    fn a_block_left_open_at_the_end_of_the_message_names_the_keyword_it_is_missing() {
        let metadata_only = TWO_SEGMENTS.split("META_STOP").next().unwrap();
        assert_eq!(
            parse_oem(metadata_only),
            Err(NdmError::UnbalancedBlock {
                line: 16,
                keyword: "META_STOP"
            })
        );

        let covariance_open = WITH_COVARIANCE.replace("COVARIANCE_STOP", "");
        assert_eq!(
            parse_oem(&covariance_open),
            Err(NdmError::UnbalancedBlock {
                line: 46,
                keyword: "COVARIANCE_STOP"
            })
        );
    }

    /// A `KEYWORD = VALUE` line where a record belongs names the block it should have opened.
    ///
    /// Three keywords, three markers: the two that belong to a covariance block are named as
    /// themselves, and everything else is reported as the metadata marker rather than echoed, because
    /// a producer's private keyword is file text and file text does not go into an error.
    #[test]
    fn a_keyword_where_a_record_belongs_names_the_block_it_should_have_opened() {
        for (line, marker) in [
            ("EPOCH = 2019-12-18T12:00:00", "EPOCH"),
            ("COV_REF_FRAME = EME2000", "COV_REF_FRAME"),
            ("OBJECT_NAME = MARS GLOBAL SURVEYOR", "META_START"),
        ] {
            let text = TWO_SEGMENTS.replacen("META_STOP\n", &format!("META_STOP\n{line}\n"), 1);
            assert_eq!(
                parse_oem(&text),
                Err(NdmError::UnbalancedBlock {
                    line: 18,
                    keyword: marker
                }),
                "{line}"
            );
        }
    }

    #[test]
    fn a_message_that_is_not_an_oem_is_named_as_such() {
        assert_eq!(
            parse_oem("CCSDS_OPM_VERS = 3.0\nORIGINATOR = JAXA\n"),
            Err(NdmError::NotThisMessage {
                expected: "CCSDS_OEM_VERS"
            })
        );
    }
}
