//! A RINEX 3 **observation** reader — the measurement file [`crate::rinex`] has no counterpart for.
//!
//! [`crate::rinex`] reads the navigation message (where the satellites are); this reads what the
//! receiver measured (pseudorange, carrier phase, Doppler, signal strength). Everything downstream —
//! the linear combinations in [`crate::combinations`], the cycle-slip detectors in
//! [`crate::cycle_slip`], any ambiguity resolution — consumes these records.
//!
//! Layout, per RINEX 3.05 §A2 (fixed columns throughout, labels in columns 61–80):
//!
//! * `SYS / # / OBS TYPES` — `A1,2X,I3,13(1X,A3)`: one system letter, the number of observation
//!   descriptors, then the descriptors themselves, continued on further lines whose first six columns
//!   are blank. The descriptor order fixes the order of the values on every observation record for
//!   that system, so a miscounted list shifts every value into the wrong slot.
//! * `> yyyy mm dd hh mm ss.sssssss  f nn      rrr.rrrrrrrrrrrr` — the epoch record: `A1,1X,I4,
//!   4(1X,I2.2),F11.7,2X,I1,I3,6X,F15.12`. Flag `f` is 0 (fine) or 1 (a power failure preceded this
//!   epoch); 2–6 introduce `nn` *event* lines carrying no observations, which are stepped over.
//! * `Gnn` then, per declared descriptor, `F14.3,I1,I1` — value, loss-of-lock indicator, signal
//!   strength. A blank value is an absent observation, not a zero one, and a record may stop early
//!   rather than pad with blanks.
//!
//! The input is untrusted: every field is read by column and validated here, and the errors carry
//! line numbers and counts only — never a slice of the file, which would put attacker-chosen text
//! into a log (the rule [`crate::vmf_grid`] and the CCSDS readers already follow).
//!
//! Carrier phase is reported in **cycles**, pseudorange in **metres** — the units RINEX defines.
//! [`crate::combinations`] takes metres for both, so phase needs its wavelength applied first.

// The doc allow is for format proper nouns (RINEX, GLONASS, Doppler), not identifiers.
#![allow(clippy::doc_markdown)]

use std::collections::BTreeMap;
use std::fmt;

use crate::sp3::{JDN_SECONDS_AT_J2000, SatId, TimeSystem, jdn, parse_sat_id, parse_time_system};

/// Columns one observation occupies on a data record: `F14.3` value, `I1` LLI, `I1` signal strength.
const OBSERVATION_WIDTH: usize = 16;

/// First column of the first observation on a data record (after the `A3` satellite identifier).
const OBSERVATION_START: usize = 3;

/// Upper bound on the descriptors one `SYS / # / OBS TYPES` list may declare.
///
/// The format's own limit is the `I3` count field; this is far above any real receiver's list and
/// exists so a corrupt count cannot drive an unbounded read.
const MAX_OBS_TYPES: usize = 100;

/// The header label whose records this reader must get exactly right, named once because it is both
/// matched on and reported in errors.
const OBS_TYPES_LABEL: &str = "SYS / # / OBS TYPES";

/// What a RINEX 3 observation descriptor measures — the first character of a code such as `C1C`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObsKind {
    /// `C` — pseudorange, metres.
    Pseudorange,
    /// `L` — carrier phase, **cycles** (not metres).
    CarrierPhase,
    /// `D` — Doppler, Hz, positive for an approaching satellite.
    Doppler,
    /// `S` — raw signal strength, receiver-dependent units.
    SignalStrength,
    /// `I` — ionosphere phase delay, cycles on the first frequency of the pair.
    IonospherePhaseDelay,
    /// `X` — receiver channel number.
    ReceiverChannel,
}

impl ObsKind {
    /// The descriptor letter this kind is written as.
    #[must_use]
    pub const fn letter(self) -> char {
        match self {
            Self::Pseudorange => 'C',
            Self::CarrierPhase => 'L',
            Self::Doppler => 'D',
            Self::SignalStrength => 'S',
            Self::IonospherePhaseDelay => 'I',
            Self::ReceiverChannel => 'X',
        }
    }

    fn from_letter(c: char) -> Option<Self> {
        Some(match c {
            'C' => Self::Pseudorange,
            'L' => Self::CarrierPhase,
            'D' => Self::Doppler,
            'S' => Self::SignalStrength,
            'I' => Self::IonospherePhaseDelay,
            'X' => Self::ReceiverChannel,
            _ => return None,
        })
    }
}

/// One RINEX 3 observation descriptor, e.g. `C1C` — what is measured, on which band, from which
/// tracking channel.
///
/// The band character is the one [`crate::combinations::carrier_frequency`] takes: it names the
/// frequency only together with the constellation letter, which is why it is not resolved here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ObsCode {
    /// What the descriptor measures.
    pub kind: ObsKind,
    /// Band / frequency character (`1`, `2`, `5`, `6`, `7`, `8`, `9`).
    pub band: char,
    /// Tracking-channel attribute (`C`, `W`, `Q`, …). `None` for the `I` and `X` descriptors, which
    /// the format writes with a blank third column.
    pub attribute: Option<char>,
}

impl fmt::Display for ObsCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.kind.letter(), self.band)?;
        match self.attribute {
            Some(a) => write!(f, "{a}"),
            None => write!(f, " "),
        }
    }
}

impl ObsCode {
    /// Parse a three-character descriptor. `None` if the letter or the band is not one the format
    /// defines.
    fn parse(code: &str) -> Option<Self> {
        let mut chars = code.chars();
        let kind = ObsKind::from_letter(chars.next()?)?;
        let band = chars.next()?;
        if !band.is_ascii_digit() {
            return None;
        }
        let attribute = match chars.next() {
            None | Some(' ') => None,
            Some(a) if a.is_ascii_alphanumeric() => Some(a),
            Some(_) => return None,
        };
        Some(Self {
            kind,
            band,
            attribute,
        })
    }
}

/// One measured value with the two quality digits that follow it on the record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// The value, in the unit of its [`ObsCode::kind`].
    pub value: f64,
    /// Loss-of-lock indicator. Bit 0 set means the receiver lost lock — a cycle slip it declares
    /// itself, independent of anything [`crate::cycle_slip`] detects. `None` where the column is
    /// blank, which the format defines as "no information", not as "no slip".
    pub lli: Option<u8>,
    /// Signal-strength projected onto 1–9 (1 worst, 9 best), or `None` where blank.
    pub signal_strength: Option<u8>,
}

impl Observation {
    /// Whether the receiver flagged a loss of lock on this observation (LLI bit 0).
    #[must_use]
    pub fn lost_lock(&self) -> bool {
        self.lli.is_some_and(|f| f & 0x01 != 0)
    }
}

/// One satellite's observations at one epoch, positionally aligned with that constellation's
/// descriptor list in [`ObsHeader::obs_types`]. `None` is an absent observation.
#[derive(Debug, Clone, PartialEq)]
pub struct SatObservation {
    /// Which satellite.
    pub sat: SatId,
    /// One slot per declared descriptor, in the header's order.
    pub values: Vec<Option<Observation>>,
}

/// Every satellite observed at one instant.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsEpoch {
    /// The instant, **seconds since J2000** (2000-01-01 12:00) in [`ObsHeader::time_system`] — the
    /// same convention, and the same deliberate refusal to normalise the scale, as
    /// [`crate::sp3::Sp3::start_epoch`].
    pub seconds_j2000: f64,
    /// Epoch flag 1: a power failure occurred between the previous epoch and this one. The
    /// observations are still observations; the receiver clock may have jumped.
    pub power_failure: bool,
    /// Receiver clock offset, seconds, where the record carries the optional field.
    pub receiver_clock_offset: Option<f64>,
    /// The satellites observed, in file order.
    pub satellites: Vec<SatObservation>,
}

/// The header fields this reader interprets. Records it does not interpret are skipped, not rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct ObsHeader {
    /// Format version from `RINEX VERSION / TYPE`.
    pub version: f64,
    /// Satellite-system character of the file (`G`, `R`, `E`, `C`, `J`, `I`, `S`, or `M` for mixed).
    pub file_system: char,
    /// `MARKER NAME`, empty where the record is absent.
    pub marker_name: String,
    /// `APPROX POSITION XYZ`, metres in the file's terrestrial frame.
    pub approx_position: Option<[f64; 3]>,
    /// `ANTENNA: DELTA H/E/N`, metres: height above the marker, then east and north eccentricity.
    pub antenna_delta_hen: Option<[f64; 3]>,
    /// `INTERVAL`, seconds.
    pub interval: Option<f64>,
    /// Time system named by `TIME OF FIRST OBS`. [`TimeSystem::Gps`] when the record is absent or
    /// does not name one, which is what the format prescribes.
    pub time_system: TimeSystem,
    /// Per constellation letter, the observation descriptors in the order every data record for that
    /// constellation reports them.
    pub obs_types: BTreeMap<char, Vec<ObsCode>>,
}

/// A parsed RINEX 3 observation file.
#[derive(Debug, Clone, PartialEq)]
pub struct RinexObs {
    /// The interpreted header.
    pub header: ObsHeader,
    /// Observation epochs, in file order.
    pub epochs: Vec<ObsEpoch>,
}

/// A failure while reading a RINEX 3 observation file.
///
/// Every variant carries positions and counts only. Echoing the offending text would copy
/// attacker-controlled bytes into whatever consumes the error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RinexObsError {
    /// No `RINEX VERSION / TYPE` record, or its file-type column is not `O`.
    NotObservationData,
    /// The version is outside the 3.x this reader implements.
    UnsupportedVersion {
        /// Major version the file declares.
        major: u8,
    },
    /// The header never terminates with `END OF HEADER`.
    NoEndOfHeader,
    /// The header declares no `SYS / # / OBS TYPES` list, so no data record can be interpreted.
    MissingObsTypes,
    /// A header record's fixed columns do not hold what its label promises.
    MalformedHeaderRecord {
        /// Line number, counting from 1 in file order.
        line: usize,
        /// The RINEX label of the record at fault.
        label: &'static str,
    },
    /// A `SYS / # / OBS TYPES` list carries a different number of descriptors than it declares.
    ///
    /// The count is what aligns values with descriptors on every data record for that system, so a
    /// list read short or long puts every subsequent value in the wrong slot.
    ObsTypeCountMismatch {
        /// Line the list's first record is on.
        line: usize,
        /// Descriptors the header declares.
        expected: usize,
        /// Descriptors actually read.
        found: usize,
    },
    /// An epoch record's calendar, flag or satellite-count fields are unreadable.
    MalformedEpochRecord {
        /// Line number, counting from 1 in file order.
        line: usize,
    },
    /// The file ends, or the next epoch begins, before an epoch's declared records are all present.
    TruncatedEpoch {
        /// Line the epoch record is on.
        line: usize,
        /// Satellite records the epoch declares.
        expected: usize,
        /// Satellite records actually present.
        found: usize,
    },
    /// A data record names a constellation the header declared no descriptor list for.
    UndeclaredSystem {
        /// Line number, counting from 1 in file order.
        line: usize,
    },
    /// A data record's satellite identifier or one of its observation fields is unreadable.
    MalformedObservationRecord {
        /// Line number, counting from 1 in file order.
        line: usize,
    },
}

impl fmt::Display for RinexObsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NotObservationData => write!(f, "not a RINEX observation file"),
            Self::UnsupportedVersion { major } => {
                write!(f, "RINEX version {major} is not the version 3 read here")
            }
            Self::NoEndOfHeader => write!(f, "the header has no END OF HEADER"),
            Self::MissingObsTypes => write!(f, "the header declares no SYS / # / OBS TYPES list"),
            Self::MalformedHeaderRecord { line, label } => {
                write!(f, "line {line} does not hold a readable {label}")
            }
            Self::ObsTypeCountMismatch {
                line,
                expected,
                found,
            } => write!(
                f,
                "the observation-type list at line {line} declares {expected} descriptors and carries {found}"
            ),
            Self::MalformedEpochRecord { line } => {
                write!(f, "line {line} does not hold a readable epoch record")
            }
            Self::TruncatedEpoch {
                line,
                expected,
                found,
            } => write!(
                f,
                "the epoch at line {line} declares {expected} satellite records and carries {found}"
            ),
            Self::UndeclaredSystem { line } => write!(
                f,
                "line {line} observes a constellation the header declares no descriptors for"
            ),
            Self::MalformedObservationRecord { line } => {
                write!(f, "line {line} does not hold a readable observation record")
            }
        }
    }
}

impl std::error::Error for RinexObsError {}

/// The trimmed contents of columns `start..end`, clamped to the line's length.
///
/// `None` only where the requested columns split a multi-byte character, which no RINEX file
/// contains — an input that reaches it is rejected rather than realigned.
fn column(line: &str, start: usize, end: usize) -> Option<&str> {
    let end = end.min(line.len());
    if start >= end {
        return Some("");
    }
    line.get(start..end).map(str::trim)
}

/// The RINEX label, columns 61–80.
fn label_of(line: &str) -> &str {
    column(line, 60, 80).unwrap_or("")
}

/// A fixed-column integer field. An empty field is `None`, an unparsable one is an error.
fn integer_field(line: &str, start: usize, end: usize) -> Result<Option<u64>, ()> {
    let raw = column(line, start, end).ok_or(())?;
    if raw.is_empty() {
        return Ok(None);
    }
    raw.parse().map(Some).map_err(|_| ())
}

/// A fixed-column float field. An empty field is `None`, an unparsable one is an error.
fn float_field(line: &str, start: usize, end: usize) -> Result<Option<f64>, ()> {
    let raw = column(line, start, end).ok_or(())?;
    if raw.is_empty() {
        return Ok(None);
    }
    let value: f64 = raw.parse().map_err(|_| ())?;
    if value.is_finite() {
        Ok(Some(value))
    } else {
        Err(())
    }
}

/// Three consecutive `F14.4` fields, as `APPROX POSITION XYZ` and `ANTENNA: DELTA H/E/N` carry them.
fn triple_field(line: &str) -> Result<[f64; 3], ()> {
    let mut out = [0.0; 3];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = float_field(line, k * 14, (k + 1) * 14)?.ok_or(())?;
    }
    Ok(out)
}

/// The descriptors on one `SYS / # / OBS TYPES` record: 13 slots of `1X,A3` after column 6.
fn obs_types_on_line(line: &str) -> Result<Vec<ObsCode>, ()> {
    let mut out = Vec::new();
    for slot in 0..13 {
        let start = 7 + 4 * slot;
        let raw = column(line, start, start + 3).ok_or(())?;
        if raw.is_empty() {
            continue;
        }
        out.push(ObsCode::parse(raw).ok_or(())?);
    }
    Ok(out)
}

/// The header state a `SYS / # / OBS TYPES` list is assembled in across its continuation lines.
struct ObsTypeList {
    line: usize,
    declared: usize,
    codes: Vec<ObsCode>,
}

/// Absorb one `SYS / # / OBS TYPES` record: a system letter in column 1 opens a new list, a blank
/// one continues the list opened last.
fn absorb_obs_types(
    line: &str,
    lineno: usize,
    lists: &mut BTreeMap<char, ObsTypeList>,
    open: &mut Option<char>,
) -> Result<(), RinexObsError> {
    let malformed = RinexObsError::MalformedHeaderRecord {
        line: lineno,
        label: OBS_TYPES_LABEL,
    };
    let system = column(line, 0, 1).ok_or(malformed)?.chars().next();
    let codes = obs_types_on_line(line).map_err(|()| malformed)?;
    let Some(sys) = system else {
        let list = open.and_then(|s| lists.get_mut(&s)).ok_or(malformed)?;
        list.codes.extend(codes);
        return Ok(());
    };
    let declared = integer_field(line, 3, 6)
        .map_err(|()| malformed)?
        .ok_or(malformed)?;
    let declared = usize::try_from(declared).map_err(|_| malformed)?;
    if declared > MAX_OBS_TYPES {
        return Err(malformed);
    }
    lists.insert(
        sys,
        ObsTypeList {
            line: lineno,
            declared,
            codes,
        },
    );
    *open = Some(sys);
    Ok(())
}

/// Close every descriptor list at `END OF HEADER`, checking each against the count it declared.
fn seal_obs_types(
    lists: BTreeMap<char, ObsTypeList>,
) -> Result<BTreeMap<char, Vec<ObsCode>>, RinexObsError> {
    let mut out = BTreeMap::new();
    for (system, list) in lists {
        if list.codes.len() != list.declared {
            return Err(RinexObsError::ObsTypeCountMismatch {
                line: list.line,
                expected: list.declared,
                found: list.codes.len(),
            });
        }
        out.insert(system, list.codes);
    }
    if out.is_empty() {
        return Err(RinexObsError::MissingObsTypes);
    }
    Ok(out)
}

/// The `RINEX VERSION / TYPE` record: the version, the mandatory `O` file type and the system letter.
fn parse_version_record(line: &str) -> Result<(f64, char), RinexObsError> {
    if label_of(line) != "RINEX VERSION / TYPE" {
        return Err(RinexObsError::NotObservationData);
    }
    let version = float_field(line, 0, 9)
        .ok()
        .flatten()
        .ok_or(RinexObsError::NotObservationData)?;
    if column(line, 20, 21) != Some("O") {
        return Err(RinexObsError::NotObservationData);
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let major = version.trunc().clamp(0.0, 255.0) as u8;
    if major != 3 {
        return Err(RinexObsError::UnsupportedVersion { major });
    }
    let file_system = column(line, 40, 41)
        .and_then(|s| s.chars().next())
        .ok_or(RinexObsError::NotObservationData)?;
    Ok((version, file_system))
}

/// Read the header, returning it and the index of the first line after `END OF HEADER`.
fn parse_header(lines: &[&str]) -> Result<(ObsHeader, usize), RinexObsError> {
    let first = lines.first().ok_or(RinexObsError::NotObservationData)?;
    let (version, file_system) = parse_version_record(first)?;

    let mut header = ObsHeader {
        version,
        file_system,
        marker_name: String::new(),
        approx_position: None,
        antenna_delta_hen: None,
        interval: None,
        time_system: TimeSystem::Gps,
        obs_types: BTreeMap::new(),
    };
    let mut lists: BTreeMap<char, ObsTypeList> = BTreeMap::new();
    let mut open: Option<char> = None;

    for (index, line) in lines.iter().enumerate().skip(1) {
        let lineno = index + 1;
        let label = label_of(line);
        let malformed = |label: &'static str| RinexObsError::MalformedHeaderRecord {
            line: lineno,
            label,
        };
        match label {
            "END OF HEADER" => {
                header.obs_types = seal_obs_types(lists)?;
                return Ok((header, index + 1));
            }
            "MARKER NAME" => {
                header.marker_name = column(line, 0, 60)
                    .ok_or_else(|| malformed("MARKER NAME"))?
                    .into();
            }
            "APPROX POSITION XYZ" => {
                header.approx_position =
                    Some(triple_field(line).map_err(|()| malformed("APPROX POSITION XYZ"))?);
            }
            "ANTENNA: DELTA H/E/N" => {
                header.antenna_delta_hen =
                    Some(triple_field(line).map_err(|()| malformed("ANTENNA: DELTA H/E/N"))?);
            }
            "INTERVAL" => {
                header.interval = float_field(line, 0, 10).map_err(|()| malformed("INTERVAL"))?;
            }
            "TIME OF FIRST OBS" => {
                // The scale sits in columns 49-51; an absent one leaves the format's GPS default.
                if let Some(system) = column(line, 48, 51).filter(|s| !s.is_empty()) {
                    header.time_system =
                        parse_time_system(system).ok_or_else(|| malformed("TIME OF FIRST OBS"))?;
                }
            }
            OBS_TYPES_LABEL => absorb_obs_types(line, lineno, &mut lists, &mut open)?,
            _ => {}
        }
    }
    Err(RinexObsError::NoEndOfHeader)
}

/// The epoch record's `A1,1X,I4,4(1X,I2.2),F11.7` calendar, as seconds since J2000.
fn epoch_seconds(line: &str) -> Result<f64, ()> {
    let year = integer_field(line, 2, 6)?.ok_or(())?;
    let month = integer_field(line, 7, 9)?.ok_or(())?;
    let day = integer_field(line, 10, 12)?.ok_or(())?;
    let hour = integer_field(line, 13, 15)?.ok_or(())?;
    let minute = integer_field(line, 16, 18)?.ok_or(())?;
    let second = float_field(line, 18, 29)?.ok_or(())?;
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(());
    }
    if hour > 23 || minute > 59 || !(0.0..61.0).contains(&second) {
        return Err(());
    }
    let (y, m, d) = (
        i64::try_from(year).map_err(|_| ())?,
        i64::try_from(month).map_err(|_| ())?,
        i64::try_from(day).map_err(|_| ())?,
    );
    #[allow(clippy::cast_precision_loss)]
    let days = jdn(y, m, d) as f64;
    let hms = u32::try_from(hour * 3600 + minute * 60).map_err(|_| ())?;
    Ok(days.mul_add(86_400.0, second) + f64::from(hms) - JDN_SECONDS_AT_J2000)
}

/// One satellite's record: `A3` identifier then one `F14.3,I1,I1` group per declared descriptor.
fn parse_observation_record(
    line: &str,
    lineno: usize,
    obs_types: &BTreeMap<char, Vec<ObsCode>>,
) -> Result<SatObservation, RinexObsError> {
    let malformed = RinexObsError::MalformedObservationRecord { line: lineno };
    let id = column(line, 0, OBSERVATION_START).ok_or(malformed)?;
    let sat = parse_sat_id(id).ok_or(malformed)?;
    let codes = obs_types
        .get(&sat.system)
        .ok_or(RinexObsError::UndeclaredSystem { line: lineno })?;

    let mut values = Vec::with_capacity(codes.len());
    for slot in 0..codes.len() {
        let start = OBSERVATION_START + OBSERVATION_WIDTH * slot;
        let Some(value) = float_field(line, start, start + 14).map_err(|()| malformed)? else {
            values.push(None);
            continue;
        };
        let digit = |from: usize| -> Result<Option<u8>, RinexObsError> {
            match integer_field(line, from, from + 1).map_err(|()| malformed)? {
                Some(d) => u8::try_from(d).map(Some).map_err(|_| malformed),
                None => Ok(None),
            }
        };
        values.push(Some(Observation {
            value,
            lli: digit(start + 14)?,
            signal_strength: digit(start + 15)?,
        }));
    }
    Ok(SatObservation { sat, values })
}

impl RinexObs {
    /// Parse a RINEX 3 observation file from its text.
    ///
    /// # Errors
    /// Returns [`RinexObsError`] if the file is not RINEX 3 observation data, if the header is
    /// missing its terminator or its observation-type lists, or if any epoch or observation record
    /// does not hold what its fixed columns promise. Header records this reader does not interpret
    /// are skipped rather than rejected, so an unfamiliar but well-formed file still reads.
    pub fn parse(text: &str) -> Result<Self, RinexObsError> {
        let lines: Vec<&str> = text.lines().collect();
        let (header, start) = parse_header(&lines)?;

        // The walk is driven by the iterator and never by an index the body advances. Every path
        // through the body consumes at least the record it read, and an iterator is the only form
        // of that statement which no arithmetic can undo.
        let mut records = lines.iter().enumerate().skip(start);
        let mut epochs = Vec::new();
        while let Some((position, line)) = records.next() {
            if !line.starts_with('>') {
                // Only epoch records may appear here; anything else means the file lost alignment.
                continue;
            }
            let lineno = position + 1;
            let malformed = RinexObsError::MalformedEpochRecord { line: lineno };
            let flag = integer_field(line, 31, 32)
                .map_err(|()| malformed)?
                .ok_or(malformed)?;
            // The count is the format's `I3` field and is read as three columns, so it cannot
            // exceed 999 however corrupt the file is; the records it claims are then bounded by
            // the lines that follow, which `TruncatedEpoch` reports.
            let count = integer_field(line, 32, 35)
                .map_err(|()| malformed)?
                .ok_or(malformed)?;
            let count = usize::try_from(count).map_err(|_| malformed)?;

            // Flags 2-6 introduce `count` event lines that carry no observations. Stepping over
            // them by their own count is what keeps the following epoch aligned, exactly as the
            // navigation reader steps over a constellation's block length.
            if flag >= 2 {
                records.by_ref().take(count).for_each(drop);
                continue;
            }

            let seconds_j2000 = epoch_seconds(line).map_err(|()| malformed)?;
            let receiver_clock_offset = float_field(line, 41, 56).map_err(|()| malformed)?;

            let mut satellites = Vec::new();
            while satellites.len() < count {
                let Some((at, record)) = records.next().filter(|(_, l)| !l.starts_with('>')) else {
                    return Err(RinexObsError::TruncatedEpoch {
                        line: lineno,
                        expected: count,
                        found: satellites.len(),
                    });
                };
                satellites.push(parse_observation_record(record, at + 1, &header.obs_types)?);
            }

            epochs.push(ObsEpoch {
                seconds_j2000,
                power_failure: flag == 1,
                receiver_clock_offset,
                satellites,
            });
        }

        Ok(Self { header, epochs })
    }

    /// The observation slot `code` occupies for constellation `system`, or `None` where the header
    /// does not declare it.
    ///
    /// Values on a record are positional, so this is how a caller reaches a named observation
    /// without assuming an order the file is free to choose.
    #[must_use]
    pub fn slot_of(&self, system: char, code: ObsCode) -> Option<usize> {
        self.header
            .obs_types
            .get(&system)?
            .iter()
            .position(|c| *c == code)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// One header record: the body in the fixed columns, the RINEX label in columns 61-80.
    fn pad(body: &str, label: &str) -> String {
        format!("{body:<60}{label}")
    }

    /// A minimal well-formed file: one mixed header, one epoch, two satellites.
    fn fixture() -> String {
        [
            pad(
                "     3.05           OBSERVATION DATA    M: MIXED",
                "RINEX VERSION / TYPE",
            ),
            pad("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            pad("R    1 C1C", "SYS / # / OBS TYPES"),
            pad("END", "END OF HEADER"),
            "> 2024 01 09 12 00  0.0000000  0  2".into(),
            "G07  23629347.915 6 124144623.325 7".into(),
            "R05  21000000.000".into(),
        ]
        .join("\n")
    }

    #[test]
    fn a_minimal_file_parses() {
        let obs = RinexObs::parse(&fixture()).unwrap();
        assert_eq!(obs.epochs.len(), 1);
        assert_eq!(obs.epochs[0].satellites.len(), 2);
        assert_eq!(obs.header.obs_types[&'G'].len(), 2);
        assert_eq!(obs.header.obs_types[&'R'].len(), 1);
    }

    /// A descriptor list whose count disagrees with its descriptors shifts every value it indexes,
    /// so it is refused rather than truncated to whichever of the two is shorter.
    #[test]
    fn a_miscounted_observation_type_list_is_refused() {
        let text = fixture().replace("G    2 C1C L1C", "G    3 C1C L1C");
        assert_eq!(
            RinexObs::parse(&text),
            Err(RinexObsError::ObsTypeCountMismatch {
                line: 2,
                expected: 3,
                found: 2,
            })
        );
    }

    /// The file type column, the version and the terminator each reject on their own.
    #[test]
    fn the_header_gates_are_each_reachable() {
        let text = fixture();
        assert_eq!(
            RinexObs::parse(&text.replace("OBSERVATION DATA", "NAVIGATION DATA ")),
            Err(RinexObsError::NotObservationData)
        );
        assert_eq!(
            RinexObs::parse(&text.replace("     3.05", "     2.11")),
            Err(RinexObsError::UnsupportedVersion { major: 2 })
        );
        assert_eq!(
            RinexObs::parse(&text.replace("END OF HEADER", "COMMENT      ")),
            Err(RinexObsError::NoEndOfHeader)
        );
        assert_eq!(RinexObs::parse(""), Err(RinexObsError::NotObservationData));
    }

    /// The header ends where it says it ends: [`parse_header`] returns the index of the first line
    /// *after* `END OF HEADER`, and the record loop reads the data section from there.
    ///
    /// A start index that names any other line is invisible to a well-formed file, because the
    /// record loop steps over whatever does not open with `>` — so this asserts the index itself
    /// rather than a parse that would forgive it.
    #[test]
    fn the_header_ends_at_the_line_after_its_terminator() {
        let text = fixture();
        let lines: Vec<&str> = text.lines().collect();
        let (_, index) = parse_header(&lines).unwrap();
        assert_eq!(index, 4);
        assert_eq!(label_of(lines[index - 1]), "END OF HEADER");
        assert!(
            lines[index].starts_with('>'),
            "the data section starts at the first epoch record"
        );
    }

    /// A header with no descriptor list cannot interpret a single data record, and says so at the
    /// terminator rather than at the first record it fails on.
    #[test]
    fn a_header_without_observation_types_is_refused() {
        let text: String = fixture()
            .lines()
            .filter(|l| !l.contains("SYS / # / OBS TYPES"))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(RinexObs::parse(&text), Err(RinexObsError::MissingObsTypes));
    }

    /// More than 13 descriptors need a continuation line, and the continuation's descriptors have to
    /// land at the *end* of the same system's list — the slot index they take is the column every
    /// observation record reports them in.
    #[test]
    fn a_descriptor_list_continues_onto_a_second_line() {
        // 14 descriptors: 13 on the first line, one on the continuation.
        let first: Vec<String> = (0..13).map(|k| format!("C{}C", k % 10)).collect();
        let text = [
            pad(
                "     3.05           OBSERVATION DATA    G: GPS",
                "RINEX VERSION / TYPE",
            ),
            pad(&format!("G   14 {}", first.join(" ")), OBS_TYPES_LABEL),
            pad("       L1C", OBS_TYPES_LABEL),
            pad("END", "END OF HEADER"),
        ]
        .join("\n");

        let obs = RinexObs::parse(&text).unwrap();
        let codes = &obs.header.obs_types[&'G'];
        assert_eq!(codes.len(), 14);
        assert_eq!(
            codes[13],
            ObsCode {
                kind: ObsKind::CarrierPhase,
                band: '1',
                attribute: Some('C'),
            }
        );
        assert_eq!(obs.slot_of('G', codes[13]), Some(13));
    }

    /// The declared-descriptor cap stops a corrupt `I3` count from driving an unbounded read, and
    /// it binds *above* itself: a list of exactly [`MAX_OBS_TYPES`] descriptors is a legal list,
    /// and one that declares more is refused where it is declared rather than where the count is
    /// later found not to match its descriptors.
    #[test]
    fn the_descriptor_count_cap_binds_above_itself() {
        // MAX_OBS_TYPES distinct descriptors, 13 to a record and the rest on continuation lines.
        let codes: Vec<String> = (0..MAX_OBS_TYPES)
            .map(|k| {
                let attribute = char::from(b'A' + u8::try_from(k / 10).unwrap());
                format!("C{}{attribute}", k % 10)
            })
            .collect();
        let header = |declared: usize| {
            let mut records = vec![pad(
                "     3.05           OBSERVATION DATA    G: GPS",
                "RINEX VERSION / TYPE",
            )];
            for (record, chunk) in codes.chunks(13).enumerate() {
                let body = if record == 0 {
                    format!("G  {declared:>3} {}", chunk.join(" "))
                } else {
                    format!("       {}", chunk.join(" "))
                };
                records.push(pad(&body, OBS_TYPES_LABEL));
            }
            records.push(pad("END", "END OF HEADER"));
            records.join("\n")
        };

        let obs = RinexObs::parse(&header(MAX_OBS_TYPES)).unwrap();
        assert_eq!(obs.header.obs_types[&'G'].len(), MAX_OBS_TYPES);
        assert_eq!(
            RinexObs::parse(&header(MAX_OBS_TYPES + 1)),
            Err(RinexObsError::MalformedHeaderRecord {
                line: 2,
                label: OBS_TYPES_LABEL,
            })
        );
    }

    /// The time scale is a header field. The GPS default belongs to the file that names no scale —
    /// substituting it for one the file does name would date every epoch in the wrong system.
    #[test]
    fn the_time_of_first_obs_record_names_the_time_scale() {
        let file = |body: &str| {
            [
                pad(
                    "     3.05           OBSERVATION DATA    R: GLONASS",
                    "RINEX VERSION / TYPE",
                ),
                pad("R    1 C1C", OBS_TYPES_LABEL),
                pad(body, "TIME OF FIRST OBS"),
                pad("END", "END OF HEADER"),
            ]
            .join("\n")
        };
        let calendar = "  2024     1     9    12     0    0.0000000";

        let named = RinexObs::parse(&file(&format!("{calendar}     GLO"))).unwrap();
        assert_eq!(named.header.time_system, TimeSystem::Glo);

        // Columns 49-51 blank: the format's own default stands and the file still reads.
        let blank = RinexObs::parse(&file(calendar)).unwrap();
        assert_eq!(blank.header.time_system, TimeSystem::Gps);

        assert_eq!(
            RinexObs::parse(&file(&format!("{calendar}     XXX"))),
            Err(RinexObsError::MalformedHeaderRecord {
                line: 3,
                label: "TIME OF FIRST OBS",
            })
        );
    }

    /// A continuation line with no list open is a descriptor list without a system, which would put
    /// its descriptors on whichever constellation happened to be read last.
    #[test]
    fn an_orphaned_continuation_line_is_refused() {
        let text = fixture().replace("G    2 C1C L1C", "       C1C L1C");
        assert_eq!(
            RinexObs::parse(&text),
            Err(RinexObsError::MalformedHeaderRecord {
                line: 2,
                label: "SYS / # / OBS TYPES",
            })
        );
    }

    /// An epoch that promises more records than it carries is a truncation, not an empty epoch.
    #[test]
    fn a_truncated_epoch_is_refused() {
        let text = fixture().replace("  0  2", "  0  3");
        assert_eq!(
            RinexObs::parse(&text),
            Err(RinexObsError::TruncatedEpoch {
                line: 5,
                expected: 3,
                found: 2,
            })
        );
    }

    /// The epoch record's calendar is a range check on every field, and the fields are checked
    /// independently: an impossible month is refused whatever the day is, and the other way round.
    ///
    /// RINEX epochs are UTC, so a second of exactly 60 is a leap second and has to be **accepted**;
    /// the field's range therefore ends at 61, not at 60. A reader that refused it would reject the
    /// one epoch of the year the observation was hardest to make.
    #[test]
    fn an_out_of_range_epoch_calendar_is_refused() {
        // `> yyyy mm dd hh mm sss.sssssss  f nn` at the columns the format fixes.
        let epoch = |month: u32, day: u32, hour: u32, minute: u32, second: f64| {
            format!("> 2024 {month:02} {day:02} {hour:02} {minute:02}{second:11.7}  0  0")
        };

        for (month, day, hour, minute, second) in [
            (1, 1, 0, 0, 0.0),
            (12, 31, 23, 59, 59.999_999_9),
            (12, 31, 23, 59, 60.0),
        ] {
            assert!(
                epoch_seconds(&epoch(month, day, hour, minute, second)).is_ok(),
                "{month:02}-{day:02} {hour:02}:{minute:02}:{second} is a legal epoch"
            );
        }

        for (month, day, hour, minute, second, why) in [
            (0, 1, 0, 0, 0.0, "month 0"),
            (13, 1, 0, 0, 0.0, "month 13"),
            (1, 0, 0, 0, 0.0, "day 0"),
            (1, 32, 0, 0, 0.0, "day 32"),
            (1, 1, 24, 0, 0.0, "hour 24"),
            (1, 1, 0, 60, 0.0, "minute 60"),
            (1, 1, 0, 0, 61.0, "second 61"),
            (1, 1, 0, 0, -0.001, "a negative second"),
        ] {
            assert!(
                epoch_seconds(&epoch(month, day, hour, minute, second)).is_err(),
                "{why} is not a legal epoch"
            );
        }

        // The leap second is a second of its own minute, not a relabelled zero of the next.
        let last = epoch_seconds(&epoch(12, 31, 23, 59, 59.0)).unwrap();
        let leap = epoch_seconds(&epoch(12, 31, 23, 59, 60.0)).unwrap();
        assert!((leap - last - 1.0).abs() < 1e-9, "{leap} vs {last}");

        // And the refusal is what the reader reports, at the line the record is on.
        assert_eq!(
            RinexObs::parse(&fixture().replace("> 2024 01 09", "> 2024 01 32")),
            Err(RinexObsError::MalformedEpochRecord { line: 5 })
        );
    }

    /// A constellation with no declared descriptors has no interpretation for its values.
    #[test]
    fn an_undeclared_constellation_is_refused() {
        let text = fixture().replace("R05  21000000.000", "E05  21000000.000");
        assert_eq!(
            RinexObs::parse(&text),
            Err(RinexObsError::UndeclaredSystem { line: 7 })
        );
    }

    /// Every error prints a distinct, non-empty message, and none of them quotes the file.
    #[test]
    fn every_error_prints_a_distinct_message() {
        let errors = [
            RinexObsError::NotObservationData,
            RinexObsError::UnsupportedVersion { major: 2 },
            RinexObsError::NoEndOfHeader,
            RinexObsError::MissingObsTypes,
            RinexObsError::MalformedHeaderRecord {
                line: 3,
                label: "INTERVAL",
            },
            RinexObsError::ObsTypeCountMismatch {
                line: 4,
                expected: 3,
                found: 2,
            },
            RinexObsError::MalformedEpochRecord { line: 5 },
            RinexObsError::TruncatedEpoch {
                line: 6,
                expected: 2,
                found: 1,
            },
            RinexObsError::UndeclaredSystem { line: 7 },
            RinexObsError::MalformedObservationRecord { line: 8 },
        ];
        let messages: Vec<String> = errors.iter().map(ToString::to_string).collect();
        assert!(messages.iter().all(|m| !m.is_empty()));
        let distinct: std::collections::HashSet<&String> = messages.iter().collect();
        assert_eq!(distinct.len(), errors.len(), "duplicate: {messages:?}");
    }

    /// Event records (flags 2-6) carry no observations, and their count is a *line* count. Stepping
    /// over it by anything else desynchronises every epoch that follows — which is why the epoch
    /// after the event has to come back with its own two satellites.
    ///
    /// The event declares more lines than precede it in the file, so a step that goes the wrong way
    /// leaves the file at its front instead of landing on the earlier epoch and reading it again.
    #[test]
    fn event_records_are_stepped_over_by_their_own_count() {
        let mut text = fixture();
        text.push_str("\n> 2024 01 09 12 00 30.0000000  4  9");
        text.push_str(&"\nAN EVENT LINE, CARRYING NO OBSERVATION".repeat(9));
        text.push_str(
            "\n> 2024 01 09 12 01  0.0000000  0  2\n\
             G07  23629350.000 6 124144630.000 7\n\
             R05  21000010.000",
        );
        let obs = RinexObs::parse(&text).unwrap();
        assert_eq!(obs.epochs.len(), 2, "the event was read as an epoch");
        assert_eq!(obs.epochs[1].satellites.len(), 2);
        assert!((obs.epochs[1].seconds_j2000 - obs.epochs[0].seconds_j2000 - 60.0).abs() < 1e-9);
    }

    /// A line in the data section that opens no epoch is stepped over, and the step has to go
    /// *forward*: the record loop's termination is that advance and its only bound is the file's
    /// own length.
    ///
    /// The stray line sits before the first epoch record, where a step in the wrong direction runs
    /// off the front of the file rather than finding an earlier epoch to read a second time.
    #[test]
    fn a_line_that_opens_no_epoch_is_stepped_over() {
        let text = fixture().replace("> 2024 01 09", "THIS LINE OPENS NO EPOCH\n> 2024 01 09");
        let obs = RinexObs::parse(&text).unwrap();
        assert_eq!(obs.epochs.len(), 1);
        assert_eq!(obs.epochs[0].satellites.len(), 2);
    }

    /// A descriptor is a letter, a digit and an optional attribute — and the blank-attribute forms
    /// the format uses for `I`/`X` round-trip through [`ObsCode::parse`] and `Display`.
    #[test]
    fn observation_codes_parse_and_print() {
        for code in ["C1C", "L2W", "D5Q", "S1X", "I1 ", "X1 "] {
            let parsed = ObsCode::parse(code).unwrap();
            assert_eq!(parsed.to_string(), code, "{code}");
        }
        assert!(ObsCode::parse("Z1C").is_none(), "unknown kind letter");
        assert!(ObsCode::parse("CXC").is_none(), "non-digit band");
        assert!(
            ObsCode::parse("C1-").is_none(),
            "non-alphanumeric attribute"
        );
    }

    /// The LLI is a bit field, not a boolean: bit 0 is loss of lock, bit 1 is a half-cycle
    /// ambiguity, and only bit 0 means the arc broke.
    #[test]
    fn loss_of_lock_reads_bit_zero_only() {
        let with = |lli: Option<u8>| Observation {
            value: 1.0,
            lli,
            signal_strength: None,
        };
        assert!(!with(None).lost_lock());
        assert!(!with(Some(0)).lost_lock());
        assert!(with(Some(1)).lost_lock());
        assert!(
            !with(Some(2)).lost_lock(),
            "half-cycle ambiguity is not a slip"
        );
        assert!(with(Some(3)).lost_lock());
    }
}
