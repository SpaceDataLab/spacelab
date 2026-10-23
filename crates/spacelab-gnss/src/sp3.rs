//! SP3 precise-ephemeris ingestion and polynomial interpolation.
//!
//! Broadcast ephemerides ([`crate::gps`], [`crate::glonass`]) are good to a few metres; the IGS
//! publishes *precise* orbit products in the SP3 format — Earth-fixed satellite positions on a coarse
//! grid (typically 15 minutes) accurate to centimetres. This module parses those position records and
//! interpolates them with Neville's algorithm (the standard ≈10th-order Lagrange interpolation), so a
//! caller gets a centimetre-level position at any instant — more accurate than any broadcast model.
//!
//! Position (`P`) records carry the ECEF position (km → m) and satellite clock bias (µs → s); velocity
//! (`V`) records carry the ECEF velocity (dm/s → m/s). Each is interpolated independently. Header,
//! correlation (`EP`/`EV`) and comment lines are skipped; the `999999.999999` clock sentinel is read as
//! "no clock".

// The single letters (x, y, z, t, p) are position/time notation; the doc allow is for SP3/IGS names;
// the calendar fields are parsed as f64 then narrowed to integer date components.
#![allow(
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]

use std::collections::BTreeMap;

/// A satellite identifier — the SP3 system letter and PRN, e.g. `G01`, `R05`, `E12`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SatId {
    /// Constellation letter (`G`/`R`/`E`/`C`/`J`/`I`/`S`).
    pub system: char,
    /// Satellite PRN / slot.
    pub prn: u8,
}

/// Integer Julian Day Number for a proleptic-Gregorian date (Fliegel–Van Flandern).
pub(crate) fn jdn(y: i64, m: i64, d: i64) -> i64 {
    let a = (14 - m) / 12;
    let yy = y + 4800 - a;
    let mm = m + 12 * a - 3;
    d + (153 * mm + 2) / 5 + 365 * yy + yy / 4 - yy / 100 + yy / 400 - 32045
}

pub(crate) fn parse_sat_id(tok: &str) -> Option<SatId> {
    let mut chars = tok.chars();
    let system = chars.next()?;
    let prn = tok.get(1..)?.trim().parse().ok()?;
    Some(SatId { system, prn })
}

/// The time system the epoch records are stamped in, from the SP3 `%c` header (SP3-c/d field 4).
///
/// It matters because an SP3 epoch is not a time until the scale is named: IGS products are GPS
/// time, IGLOS products are UTC, and the two differ by the whole leap-second count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeSystem {
    /// GPS time. The default the format assigns when the field is absent or blank.
    #[default]
    Gps,
    /// GLONASS time, which is UTC(SU) plus three hours and carries leap seconds.
    Glo,
    /// Galileo system time.
    Gal,
    /// International Atomic Time.
    Tai,
    /// Coordinated Universal Time.
    Utc,
    /// QZSS time.
    Qzs,
    /// BeiDou time.
    Bdt,
    /// NavIC (IRNSS) time.
    Irn,
}

impl TimeSystem {
    /// Seconds to add to an epoch in this system to reach TAI, where that offset is a *constant*.
    ///
    /// `None` for [`TimeSystem::Utc`] and [`TimeSystem::Glo`], whose offset is the leap-second count
    /// at the epoch — a property of the date, not of the scale, so it is the caller's leap-second
    /// table to apply and not this type's to guess.
    #[must_use]
    pub const fn tai_offset_seconds(self) -> Option<f64> {
        match self {
            // GNSS system times are steered to a fixed offset from TAI by their control segments.
            Self::Gps | Self::Gal | Self::Qzs | Self::Irn => Some(19.0),
            Self::Bdt => Some(33.0), // BDT started 14 s behind GPS time, at 2006-01-01 UTC
            Self::Tai => Some(0.0),
            Self::Glo | Self::Utc => None,
        }
    }
}

/// The reference frame an SP3 file states its coordinates are in — line 1, columns 47-51 of the
/// SP3-c/d header.
///
/// The reader used to discard this field, and discarding it is not a small omission: the same
/// format carries IGS orbits in an ITRS realization, ILRS orbits in SLRF2020 and IAC GLONASS orbits
/// in **PZ-90**, which is a different datum. PZ-90.11 agrees with ITRF2008 at the centimetre, but
/// the PZ-90 and PZ-90.02 that older products use differ from ITRF by decimetres to metres, and a
/// station whose coordinates are ITRF differenced against a satellite whose orbit is PZ-90 is wrong
/// by that much with nothing to show for it. The eight SP3 fixtures in this workspace carry three
/// distinct labels between them -- `IGS20`, `IGc20` and `SLR20` -- so the field is populated in
/// practice and not a theoretical concern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReferenceFrame {
    /// No header line, or the field is blank. Not a default frame: an absence.
    #[default]
    Unstated,
    /// A realization of the International Terrestrial Reference System.
    Itrs(Itrs),
    /// PZ-90, the GLONASS datum. Its own realizations differ from ITRF by decimetres before
    /// PZ-90.11, which the five-character label does not distinguish.
    Pz90,
    /// WGS 84, the GPS broadcast datum. Its later realizations agree with ITRF at the centimetre.
    Wgs84,
    /// A five-character label this reader does not know. Deliberately not treated as ITRS: an
    /// unrecognised datum is not a compatible one.
    Unrecognised,
}

/// The format version, from line 1 column 2.
///
/// Reported rather than acted on: this reader's record handling is the same for every version,
/// because the versions differ in what the *header* may declare and not in how a `P`, `V` or `*`
/// line is laid out. What the version does tell a caller is which limits apply — SP3-c caps a file
/// at 85 satellites and a 60-character comment, SP3-d raises those to 999 and 80 — and whether a
/// missing `%c` record means "GPS time by default" (SP3-a, which has no such record) or "the
/// producer omitted it".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Version {
    /// No `#` header line. An absence, not a default.
    #[default]
    Unstated,
    /// SP3-a.
    A,
    /// SP3-b.
    B,
    /// SP3-c.
    C,
    /// SP3-d: at most 999 satellites, unlimited comment records, 80-character comments.
    D,
    /// A version character this reader has never heard of. Kept as broadcast rather than mapped to
    /// the newest known version, because a file from the future is not a file this reader has
    /// verified it can read.
    Unrecognised(char),
}

impl Version {
    /// Lower case only. Every version of the format spells its own character in lower case in
    /// every example it prints, so `#D` is a producer's mistake and not a dialect — reporting it as
    /// [`Version::Unrecognised`] surfaces that, where quietly accepting it would hide it. The
    /// character is kept, so a caller that wants to be lenient still can be.
    fn from_char(c: char) -> Self {
        match c {
            'a' => Self::A,
            'b' => Self::B,
            'c' => Self::C,
            'd' => Self::D,
            other => Self::Unrecognised(other),
        }
    }
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
    /// [`ReferenceFrame::Unstated`] agrees with nothing, including itself: a file that does not say
    /// what frame it is in has not said it agrees with anything, and answering `true` there would
    /// be the silent default this type exists to remove.
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

/// The SP3 five-character coordinate-system label, trimmed and upper-cased by the caller.
pub(crate) fn parse_reference_frame(tok: &str) -> ReferenceFrame {
    match tok.trim() {
        "" | "ccccc" => ReferenceFrame::Unstated,
        "ITR08" | "ITRF08" | "ITRF2008" => ReferenceFrame::Itrs(Itrs::Itrf2008),
        "ITR14" | "ITRF14" | "ITRF2014" => ReferenceFrame::Itrs(Itrs::Itrf2014),
        "ITR20" | "ITRF20" | "ITRF2020" => ReferenceFrame::Itrs(Itrs::Itrf2020),
        // The IGS letters are updates of one realization, not separate solutions.
        "IGS08" | "IGb08" | "IGB08" => ReferenceFrame::Itrs(Itrs::Igs08),
        "IGS14" | "IGb14" | "IGB14" => ReferenceFrame::Itrs(Itrs::Igs14),
        "IGS20" | "IGb20" | "IGB20" | "IGc20" | "IGC20" => ReferenceFrame::Itrs(Itrs::Igs20),
        "SLR20" | "SLRF20" | "SLRF2020" => ReferenceFrame::Itrs(Itrs::Slrf2020),
        "PZ90" | "PZ-90" | "PZ9011" => ReferenceFrame::Pz90,
        "WGS84" | "WGS-84" => ReferenceFrame::Wgs84,
        _ => ReferenceFrame::Unrecognised,
    }
}

pub(crate) fn parse_time_system(tok: &str) -> Option<TimeSystem> {
    Some(match tok {
        "GPS" | "ccc" => TimeSystem::Gps, // "ccc" is the format's blank filler, and GPS is its default
        "GLO" => TimeSystem::Glo,
        "GAL" => TimeSystem::Gal,
        "TAI" => TimeSystem::Tai,
        "UTC" => TimeSystem::Utc,
        "QZS" => TimeSystem::Qzs,
        "BDT" => TimeSystem::Bdt,
        "IRN" => TimeSystem::Irn,
        _ => return None,
    })
}

/// Seconds from JDN 0 at 00:00 to the J2000 epoch, JD 2451545.0 = 2000-01-01 12:00. `jdn` counts
/// whole days from midnight, JD from noon, so the half-day is not optional.
pub(crate) const JDN_SECONDS_AT_J2000: f64 = 2_451_545.0 * 86_400.0 + 43_200.0;

/// A parsed SP3 file: the epoch grid (seconds from the first epoch) and, per satellite, the ECEF
/// positions (m), clock biases (s) and — when the file carries `V` records — velocities (m/s) at each
/// epoch (`None` where a satellite/quantity is absent from an epoch).
#[derive(Debug, Clone, Default)]
pub struct Sp3 {
    /// The first epoch, seconds since J2000 in `time_system`. Absent for a file with no epochs.
    start_epoch: Option<f64>,
    /// The scale `start_epoch` and the epoch grid are stamped in.
    time_system: TimeSystem,
    /// The frame the positions are in, from line 1 of the header.
    reference_frame: ReferenceFrame,
    /// The format version, from line 1 column 2.
    version: Version,
    epochs: Vec<f64>,
    positions: BTreeMap<SatId, Vec<Option<[f64; 3]>>>,
    clocks: BTreeMap<SatId, Vec<Option<f64>>>,
    velocities: BTreeMap<SatId, Vec<Option<[f64; 3]>>>,
}

/// The SP3 sentinel for a missing/bad clock, in microseconds (`999999.999999`).
const BAD_CLOCK_US: f64 = 999_999.0;

/// Set `map[id][ei] = val`, back-filling the series to `n_epochs` with `None` on first sight.
fn set_sample<T: Copy>(
    map: &mut BTreeMap<SatId, Vec<Option<T>>>,
    id: SatId,
    ei: usize,
    n_epochs: usize,
    val: T,
) {
    let series = map.entry(id).or_insert_with(|| vec![None; n_epochs]);
    if series.len() < n_epochs {
        series.resize(n_epochs, None);
    }
    series[ei] = Some(val);
}

/// Neville's algorithm: the interpolating polynomial through `(xs[i], ys[i])` evaluated at `t`.
fn neville(xs: &[f64], ys: &[f64], t: f64) -> f64 {
    let n = xs.len();
    let mut c = ys.to_vec();
    for k in 1..n {
        for i in (k..n).rev() {
            c[i] = ((t - xs[i - k]) * c[i] - (t - xs[i]) * c[i - 1]) / (xs[i] - xs[i - k]);
        }
    }
    c[n - 1]
}

/// The `(start, count)` window of `order + 1` samples centred on the sample epoch nearest `t`.
fn window(epochs: &[f64], t: f64, order: usize) -> Option<(usize, usize)> {
    if epochs.len() < 2 {
        return None;
    }
    let want = (order + 1).min(epochs.len());
    let nearest = epochs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| libm::fabs(**a - t).total_cmp(&libm::fabs(**b - t)))
        .map(|(i, _)| i)?;
    let start = nearest.saturating_sub(want / 2).min(epochs.len() - want);
    Some((start, want))
}

impl Sp3 {
    /// The satellites present in the file, in ascending order.
    pub fn satellites(&self) -> impl Iterator<Item = SatId> + '_ {
        self.positions.keys().copied()
    }

    /// The number of epochs.
    #[must_use]
    pub fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    /// The epoch grid, seconds from the first epoch — the times [`Sp3::positions`] is sampled on.
    #[must_use]
    pub fn epochs(&self) -> &[f64] {
        &self.epochs
    }

    /// `sat`'s ECEF positions (m) on the [`Sp3::epochs`] grid, `None` at an epoch the file carries no
    /// record for. `None` for a satellite the file does not contain.
    ///
    /// These are the measurements. An orbit fit weighs *these*, not [`Sp3::interpolate`] — the
    /// interpolant is a smooth curve drawn through them, so fitting to it would credit the polynomial
    /// with information the product does not carry.
    #[must_use]
    pub fn positions(&self, sat: SatId) -> Option<&[Option<[f64; 3]>]> {
        self.positions.get(&sat).map(Vec::as_slice)
    }

    /// The first epoch, **seconds since J2000** (2000-01-01 12:00) in [`Sp3::time_system`]. `None`
    /// for a file with no epoch records.
    ///
    /// The epoch grid the interpolators take is relative to this instant, which is enough to
    /// interpolate the file against itself and not enough for anything else: a force model needs the
    /// absolute epoch to reach the EOP series and the planetary ephemeris. Reaching TT from here is
    /// `start_epoch + time_system.tai_offset_seconds() + 32.184`, and the offset is deliberately not
    /// applied for you — the scale is a property of the file, and silently normalising it would hide
    /// a GLONASS product's leap seconds behind a number that looks right.
    #[must_use]
    pub fn start_epoch(&self) -> Option<f64> {
        self.start_epoch
    }

    /// The format version, from line 1 column 2.
    ///
    /// [`Version::Unstated`] for a file with no `#` header line. See [`Version`] for what the
    /// version does and does not change about a file.
    #[must_use]
    pub const fn version(&self) -> Version {
        self.version
    }

    /// The reference frame the positions are in, from line 1 of the header.
    ///
    /// [`ReferenceFrame::Unstated`] when the file has no `#` line or the field is blank — which is
    /// an absence and not a default, because the only safe thing to do with an unstated frame is
    /// refuse to difference it against anything.
    #[must_use]
    pub const fn reference_frame(&self) -> ReferenceFrame {
        self.reference_frame
    }

    /// The time system the epochs are stamped in, from the `%c` header. [`TimeSystem::Gps`] when the
    /// file does not say, which is what the format prescribes.
    #[must_use]
    pub fn time_system(&self) -> TimeSystem {
        self.time_system
    }

    /// Parse an SP3 file. Epoch lines start with `*`, position lines with `P`, velocity lines with `V`.
    /// Line 1 supplies the [`ReferenceFrame`] and the `%c` header record the [`TimeSystem`]; other
    /// header, correlation and comment lines are skipped, and a satellite first seen mid-file is
    /// back-filled with `None`.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut epochs: Vec<f64> = Vec::new();
        let mut positions: BTreeMap<SatId, Vec<Option<[f64; 3]>>> = BTreeMap::new();
        let mut clocks: BTreeMap<SatId, Vec<Option<f64>>> = BTreeMap::new();
        let mut velocities: BTreeMap<SatId, Vec<Option<[f64; 3]>>> = BTreeMap::new();
        let mut t0: Option<f64> = None;
        let mut time_system: Option<TimeSystem> = None;
        let mut reference_frame = ReferenceFrame::Unstated;
        let mut version = Version::Unstated;

        for line in text.lines() {
            if line.starts_with('#') && !line.starts_with("##") {
                // Line 1, columns 47-51 by the SP3-c/d spec, read by index because the two A5
                // fields around it may be blank. Only the *first* `#` line is the header record;
                // `##` is the GPS-week line and any later `#` is a comment in some producers.
                // No length guard: `get(46..51)` already returns `None` for a line too short to
                // hold the field, and a guard that rejects nothing its successor does not is dead
                // code a mutation block cannot tell from live code.
                if reference_frame == ReferenceFrame::Unstated {
                    reference_frame = line
                        .get(46..51)
                        .map_or(ReferenceFrame::Unstated, parse_reference_frame);
                }
                if version == Version::Unstated {
                    version = line
                        .chars()
                        .nth(1)
                        .map_or(Version::Unstated, Version::from_char);
                }
            } else if line.starts_with("%c") {
                // `%c G  cc GPS ccc cccc …` — columns 10-12 by the SP3-c/d spec, read by index
                // rather than by token because the two fields before it may both be blank. Only the
                // *first* `%c` record carries the time system; the second is filler that would
                // otherwise read as `ccc` and silently reset a GLONASS file to GPS.
                if time_system.is_none() {
                    time_system = line.get(9..12).and_then(parse_time_system);
                }
            } else if let Some(rest) = line.strip_prefix('*') {
                let f: Vec<f64> = rest
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                if f.len() < 6 {
                    continue;
                }
                let (y, m, d) = (f[0] as i64, f[1] as i64, f[2] as i64);
                let secs = (jdn(y, m, d) as f64) * 86_400.0 + f[3] * 3600.0 + f[4] * 60.0 + f[5];
                let t = *t0.get_or_insert(secs);
                epochs.push(secs - t);
                // Extend every series by one absent slot (a record below fills it if present).
                for s in positions.values_mut() {
                    s.push(None);
                }
                for s in clocks.values_mut() {
                    s.push(None);
                }
                for s in velocities.values_mut() {
                    s.push(None);
                }
            } else if line.starts_with('P') || line.starts_with('V') {
                let idx = epochs.len().checked_sub(1);
                let (Some(ei), Some(id)) = (idx, parse_sat_id(line.get(1..4).unwrap_or("").trim()))
                else {
                    continue;
                };
                let vals: Vec<f64> = line[4..]
                    .split_whitespace()
                    .filter_map(|t| t.parse().ok())
                    .collect();
                if vals.len() < 3 {
                    continue;
                }
                let n = epochs.len();
                if line.starts_with('P') {
                    set_sample(
                        &mut positions,
                        id,
                        ei,
                        n,
                        [vals[0] * 1000.0, vals[1] * 1000.0, vals[2] * 1000.0],
                    );
                    // Clock bias (µs → s); the 999999.999999 sentinel means "no clock".
                    if let Some(&clk_us) = vals.get(3)
                        && libm::fabs(clk_us) < BAD_CLOCK_US
                    {
                        set_sample(&mut clocks, id, ei, n, clk_us * 1.0e-6);
                    }
                } else {
                    // Velocity dm/s → m/s.
                    set_sample(
                        &mut velocities,
                        id,
                        ei,
                        n,
                        [vals[0] * 0.1, vals[1] * 0.1, vals[2] * 0.1],
                    );
                }
            }
        }
        Sp3 {
            start_epoch: t0.map(|t| t - JDN_SECONDS_AT_J2000),
            time_system: time_system.unwrap_or_default(),
            reference_frame,
            version,
            epochs,
            positions,
            clocks,
            velocities,
        }
    }

    /// Interpolate satellite `sat`'s ECEF position (metres) at `t` seconds from the first epoch, using
    /// the `order + 1` samples nearest `t` (Neville). `None` if the satellite is unknown or has too few
    /// samples. `order` is clamped to the available sample count.
    #[must_use]
    pub fn interpolate(&self, sat: SatId, t: f64, order: usize) -> Option<[f64; 3]> {
        self.interp_vec3(&self.positions, sat, t, order)
    }

    /// Interpolate satellite `sat`'s ECEF velocity (m/s) at `t` from the file's `V` records; `None` if
    /// it carries none for this satellite.
    #[must_use]
    pub fn interpolate_velocity(&self, sat: SatId, t: f64, order: usize) -> Option<[f64; 3]> {
        self.interp_vec3(&self.velocities, sat, t, order)
    }

    /// Interpolate satellite `sat`'s clock bias (seconds) at `t`; `None` if it has too few clock
    /// samples.
    #[must_use]
    pub fn interpolate_clock(&self, sat: SatId, t: f64, order: usize) -> Option<f64> {
        let series = self.clocks.get(&sat)?;
        let pts: Vec<(f64, f64)> = self
            .epochs
            .iter()
            .zip(series)
            .filter_map(|(&e, c)| c.map(|v| (e, v)))
            .collect();
        let epochs: Vec<f64> = pts.iter().map(|(e, _)| *e).collect();
        let (start, want) = window(&epochs, t, order)?;
        let ys: Vec<f64> = pts[start..start + want].iter().map(|(_, v)| *v).collect();
        Some(neville(&epochs[start..start + want], &ys, t))
    }

    /// Neville interpolation of a per-satellite 3-vector series (position or velocity) at `t`.
    fn interp_vec3(
        &self,
        map: &BTreeMap<SatId, Vec<Option<[f64; 3]>>>,
        sat: SatId,
        t: f64,
        order: usize,
    ) -> Option<[f64; 3]> {
        let series = map.get(&sat)?;
        let pts: Vec<(f64, [f64; 3])> = self
            .epochs
            .iter()
            .zip(series)
            .filter_map(|(&e, p)| p.map(|xyz| (e, xyz)))
            .collect();
        let epochs: Vec<f64> = pts.iter().map(|(e, _)| *e).collect();
        let (start, want) = window(&epochs, t, order)?;
        let xs = &epochs[start..start + want];
        let mut out = [0.0; 3];
        for axis in 0..3 {
            let ys: Vec<f64> = pts[start..start + want]
                .iter()
                .map(|(_, p)| p[axis])
                .collect();
            out[axis] = neville(xs, &ys, t);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::format_push_string)]
    use super::*;

    /// A synthetic SP3 for one satellite whose position is an exact cubic in time — an interpolator of
    /// order ≥ 3 must reproduce it to rounding. Units: km.
    fn cubic(t: f64) -> [f64; 3] {
        [
            1000.0 + 2.0 * t - 0.001 * t * t + 1.0e-7 * t * t * t,
            -500.0 + 1.5 * t + 0.002 * t * t,
            300.0 - 0.5 * t + 5.0e-8 * t * t * t,
        ]
    }

    /// The exact time derivative of [`cubic`], km/s.
    fn cubic_velocity(t: f64) -> [f64; 3] {
        [
            2.0 - 0.002 * t + 3.0e-7 * t * t,
            1.5 + 0.004 * t,
            -0.5 + 1.5e-7 * t * t,
        ]
    }

    /// A linear synthetic clock bias, microseconds.
    fn clock_us(t: f64) -> f64 {
        1.0 + 2.0e-3 * t
    }

    /// The header line of each SP3 product this workspace actually reads, verbatim.
    ///
    /// Transcribed from `crates/spacelab/tests/fixtures/`: the IGS final and rapid GPS combinations
    /// and the ILRS LAGEOS combination. The GLONASS line is IAC's own format for the same field and
    /// is the case the type exists for.
    #[test]
    fn the_header_names_the_frame_and_each_real_product_names_a_different_one() {
        for (line, want) in [
            (
                "#cP2025  1  1  0  0  0.00000000      96 ORBIT IGS20 HLM  IGS",
                ReferenceFrame::Itrs(Itrs::Igs20),
            ),
            (
                "#cP2026  8 12  0  0  0.00000000      96 ORBIT IGc20 HLM  IGS",
                ReferenceFrame::Itrs(Itrs::Igs20),
            ),
            (
                "#cV2025  1 12  0  0  0.00000000    5040   SLR SLR20 FIT COMB",
                ReferenceFrame::Itrs(Itrs::Slrf2020),
            ),
            (
                "#cP2025  1  1  0  0  0.00000000      96 ORBIT PZ90  HLM  IAC",
                ReferenceFrame::Pz90,
            ),
        ] {
            assert_eq!(Sp3::parse(line).reference_frame(), want, "{line}");
        }
    }

    /// Every label the table claims to recognise, and what each reduces to.
    ///
    /// The function *is* a lookup table, so the test is the table: a mutation block over the first
    /// version deleted seven arms with nothing moving, because the fixture-driven test above only
    /// ever asks for the four labels this repository happens to hold. An arm no test names is an arm
    /// that can be removed, and a producer writing `ITRF14` would then have read as `Unrecognised`.
    #[test]
    fn every_recognised_label_reduces_to_its_own_solution() {
        use Itrs::{Igs08, Igs14, Igs20, Itrf2008, Itrf2014, Itrf2020, Slrf2020};

        for (labels, want, itrf) in [
            (&["ITR08", "ITRF08", "ITRF2008"][..], Itrf2008, Itrf2008),
            (&["ITR14", "ITRF14", "ITRF2014"][..], Itrf2014, Itrf2014),
            (&["ITR20", "ITRF20", "ITRF2020"][..], Itrf2020, Itrf2020),
            (&["IGS08", "IGb08", "IGB08"][..], Igs08, Itrf2008),
            (&["IGS14", "IGb14", "IGB14"][..], Igs14, Itrf2014),
            (
                &["IGS20", "IGb20", "IGB20", "IGc20", "IGC20"][..],
                Igs20,
                Itrf2020,
            ),
            (&["SLR20", "SLRF20", "SLRF2020"][..], Slrf2020, Itrf2020),
        ] {
            for label in labels {
                let got = parse_reference_frame(label);
                assert_eq!(got, ReferenceFrame::Itrs(want), "{label}");
                assert_eq!(got.itrs().map(Itrs::itrf), Some(itrf), "{label}");
                // Padding is what the fixed-width field actually delivers.
                assert_eq!(parse_reference_frame(&format!("{label} ")), got, "{label} ");
            }
        }

        for label in ["PZ90", "PZ-90", "PZ9011"] {
            assert_eq!(
                parse_reference_frame(label),
                ReferenceFrame::Pz90,
                "{label}"
            );
        }
        for label in ["WGS84", "WGS-84"] {
            assert_eq!(
                parse_reference_frame(label),
                ReferenceFrame::Wgs84,
                "{label}"
            );
        }
        // The format's blank filler and an actually blank field are both an absence, not a default.
        for label in ["", "ccccc", "     "] {
            assert_eq!(
                parse_reference_frame(label),
                ReferenceFrame::Unstated,
                "{label:?}"
            );
        }
        // Anything else is refused rather than guessed at, including labels that merely look close.
        for label in ["ITRF", "IGS", "IGS21", "GTRF", "PZ", "cccc"] {
            assert_eq!(
                parse_reference_frame(label),
                ReferenceFrame::Unrecognised,
                "{label}"
            );
        }
    }

    /// The whole point: a GLONASS orbit in PZ-90 must not be differenced against ITRF coordinates,
    /// and an unstated frame must not be differenced against anything at all.
    #[test]
    fn frames_agree_only_within_one_itrf_solution() {
        let igs20 = ReferenceFrame::Itrs(Itrs::Igs20);
        let update = parse_reference_frame("IGc20");
        let itrf2020 = ReferenceFrame::Itrs(Itrs::Itrf2020);
        let slr20 = ReferenceFrame::Itrs(Itrs::Slrf2020);
        let igs14 = ReferenceFrame::Itrs(Itrs::Igs14);

        // Successive IGS updates and the IERS solution they realize are one frame.
        for other in [update, itrf2020, slr20] {
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

    /// A file with no header record says nothing about its frame, and nothing is what it reports.
    #[test]
    fn a_file_without_a_header_line_leaves_the_frame_unstated() {
        let no_header = "*  2024  1  9  0  0  0.00000000\nPG01     1.0     2.0     3.0     4.0\n";
        assert_eq!(
            Sp3::parse(no_header).reference_frame(),
            ReferenceFrame::Unstated
        );
        // A `##` line is the GPS-week record, not the header, and must not be read as one.
        assert_eq!(
            Sp3::parse("## 2431 259200.00000000   900.00000000 61264 0.0000000000000\n")
                .reference_frame(),
            ReferenceFrame::Unstated
        );
        // A short header line cannot be sliced at column 47 and must not panic.
        assert_eq!(
            Sp3::parse("#cP2025\n").reference_frame(),
            ReferenceFrame::Unstated
        );
    }

    fn synthetic_sp3() -> String {
        // Five epochs 900 s apart on 2024-01-09, one satellite G01, with P (position km + clock µs) and
        // V (velocity dm/s) records.
        let mut s = String::from("#dV2024  1  9  0  0  0.00000000       5 ORBIT IGS20 HLM  IGS\n");
        for k in 0..5 {
            let sec = f64::from(k * 900);
            let (mm, ss) = ((k * 15) % 60, 0);
            let hh = (k * 15) / 60;
            s.push_str(&format!("*  2024  1  9 {hh:2}  {mm}  {ss}.00000000\n"));
            let p = cubic(sec);
            s.push_str(&format!(
                "PG01 {:13.6} {:13.6} {:13.6} {:13.6}\n",
                p[0],
                p[1],
                p[2],
                clock_us(sec)
            ));
            let v = cubic_velocity(sec); // km/s → dm/s is ×1e4
            s.push_str(&format!(
                "VG01 {:13.6} {:13.6} {:13.6} {:13.6}\n",
                v[0] * 1.0e4,
                v[1] * 1.0e4,
                v[2] * 1.0e4,
                0.0
            ));
        }
        s.push_str("EOF\n");
        s
    }

    #[test]
    fn parses_epochs_and_satellites() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        assert_eq!(sp3.epoch_count(), 5);
        let sats: Vec<SatId> = sp3.satellites().collect();
        assert_eq!(
            sats,
            vec![SatId {
                system: 'G',
                prn: 1
            }]
        );
    }

    /// The two accessors an orbit fit reads, against the grid and the samples the file was written
    /// from. `epoch_count` above says how many there are; nothing else said *what* they are, so a
    /// parser that returned one epoch, or the right count of the wrong positions, read as covered.
    #[test]
    fn the_accessors_hand_back_the_grid_and_the_samples_the_file_carries() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        assert_eq!(
            sp3.epochs(),
            [0.0, 900.0, 1800.0, 2700.0, 3600.0].as_slice()
        );

        let g01 = SatId {
            system: 'G',
            prn: 1,
        };
        let pos = sp3.positions(g01).unwrap();
        assert_eq!(pos.len(), sp3.epochs().len());
        for (k, sample) in pos.iter().enumerate() {
            let want = cubic(sp3.epochs()[k]);
            let got = sample.unwrap();
            for axis in 0..3 {
                // The record carries six decimals of a kilometre, so a millimetre is its resolution.
                assert!(
                    (got[axis] - want[axis] * 1000.0).abs() < 1e-3,
                    "epoch {k} axis {axis}: {}",
                    got[axis]
                );
            }
        }

        // A satellite the file does not carry is absent, which is not the same as carried with no
        // samples — the caller that cannot tell those apart fits an empty arc and reports success.
        assert!(
            sp3.positions(SatId {
                system: 'R',
                prn: 7
            })
            .is_none()
        );
    }

    #[test]
    fn interpolation_reproduces_a_cubic_exactly() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        let g01 = SatId {
            system: 'G',
            prn: 1,
        };
        // Off-grid time (t = 1350 s, between samples 2 and 3): cubic recovered to sub-mm.
        let got = sp3.interpolate(g01, 1350.0, 3).unwrap();
        let want = cubic(1350.0);
        for axis in 0..3 {
            let want_m = want[axis] * 1000.0;
            assert!(
                (got[axis] - want_m).abs() < 1e-3,
                "axis {axis}: {} vs {want_m}",
                got[axis]
            );
        }
    }

    /// The absolute epoch, against a date arithmetic that shares nothing with the parser's.
    ///
    /// `jdn` counts whole days from midnight and JD is measured from noon, so the half-day between
    /// them is the one place this conversion can be wrong by a number that still looks like a date.
    /// The two references are computed by calendar difference — 8 774 days from 2000-01-01 to
    /// 2024-01-09 and 9 132 to 2025-01-01, each less the half day — and one of them is the real IGS
    /// final product's first epoch, so the header shape is a real one and not only the synthetic.
    #[test]
    fn the_start_epoch_is_seconds_since_j2000_in_the_files_own_scale() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        assert_eq!(sp3.start_epoch(), Some(758_030_400.0));
        // No `%c` record in the synthetic file, and the format's default is GPS.
        assert_eq!(sp3.time_system(), TimeSystem::Gps);

        // The first two header lines and first epoch of IGS0OPSFIN_20250010000_01D_15M_ORB.SP3.
        let igs = "#cP2025  1  1  0  0  0.00000000      96 ORBIT IGS20 HLM  IGS\n\
                   ## 2347 259200.00000000   900.00000000 60676 0.0000000000000\n\
                   %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
                   %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
                   *  2025  1  1  0  0  0.00000000\n\
                   PG01  15931.689402   2160.462672  21149.136216 999999.999999\n";
        let sp3 = Sp3::parse(igs);
        assert_eq!(sp3.start_epoch(), Some(788_961_600.0));
        assert_eq!(sp3.time_system(), TimeSystem::Gps);
        assert_eq!(sp3.time_system().tai_offset_seconds(), Some(19.0));
    }

    /// The time system is read from the *first* `%c` record only.
    ///
    /// The second one is all filler, and its columns 10-12 hold `ccc`, which the parser maps to the
    /// format's GPS default. Reading both would therefore turn every GLONASS product into a GPS one
    /// — silently, and with the epochs off by the whole leap-second count.
    #[test]
    fn the_time_system_is_taken_from_the_first_c_record_and_the_filler_cannot_undo_it() {
        let glo = "#cP2025  1  1  0  0  0.00000000      96 ORBIT PZ90  HLM  IAC\n\
                   %c R  cc GLO ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
                   %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
                   *  2025  1  1  0  0  0.00000000\n\
                   PR01  15931.689402   2160.462672  21149.136216 999999.999999\n";
        let sp3 = Sp3::parse(glo);
        assert_eq!(sp3.time_system(), TimeSystem::Glo);
        // UTC-bearing scales have no constant offset to TAI, and saying so is the point.
        assert_eq!(sp3.time_system().tai_offset_seconds(), None);

        // A file with no epoch records has no epoch to report.
        assert_eq!(Sp3::parse("#cP2025  1  1  0  0  0.0\n").start_epoch(), None);
    }

    /// Every token the `%c` column can carry, read at the parser rather than through a whole file.
    ///
    /// Through `Sp3::parse` the GPS arm is invisible: an unrecognised token leaves `time_system`
    /// `None` and `unwrap_or_default()` answers `Gps` regardless, so deleting the arm that
    /// *recognises* GPS changes nothing a parsed file can show. Recognised-as-GPS against
    /// fell-back-to-the-default is a distinction only the function has, and the six scales below it
    /// are the ones no fixture in this file ever names — a product stamped in BDT or IRN would have
    /// been read as GPS, and the epochs off by the 14 s between them.
    #[test]
    fn every_time_system_token_maps_to_its_own_scale() {
        for (tok, want, tai) in [
            ("GPS", TimeSystem::Gps, Some(19.0)),
            ("ccc", TimeSystem::Gps, Some(19.0)),
            ("GLO", TimeSystem::Glo, None),
            ("GAL", TimeSystem::Gal, Some(19.0)),
            ("TAI", TimeSystem::Tai, Some(0.0)),
            ("UTC", TimeSystem::Utc, None),
            ("QZS", TimeSystem::Qzs, Some(19.0)),
            ("BDT", TimeSystem::Bdt, Some(33.0)),
            ("IRN", TimeSystem::Irn, Some(19.0)),
        ] {
            assert_eq!(parse_time_system(tok), Some(want), "token {tok}");
            assert_eq!(want.tai_offset_seconds(), tai, "offset for {tok}");
        }

        // `GST` is Galileo System Time under its own name and `UT1` is a scale, but neither is a
        // token this format defines; the caller's fallback to GPS is a different statement from
        // having read one.
        assert_eq!(parse_time_system("GST"), None);
        assert_eq!(parse_time_system("UT1"), None);
        assert_eq!(parse_time_system(""), None);
    }

    #[test]
    fn interpolation_passes_through_the_sample_points() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        let g01 = SatId {
            system: 'G',
            prn: 1,
        };
        let got = sp3.interpolate(g01, 900.0, 3).unwrap(); // exactly at epoch index 1
        let want = cubic(900.0);
        for axis in 0..3 {
            assert!((got[axis] - want[axis] * 1000.0).abs() < 1e-3);
        }
    }

    #[test]
    fn interpolation_recovers_the_clock_and_velocity() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        let g01 = SatId {
            system: 'G',
            prn: 1,
        };
        // Clock is linear in time → recovered exactly (order ≥ 1), reported in seconds.
        let clk = sp3.interpolate_clock(g01, 1350.0, 3).unwrap();
        assert!(
            (clk - clock_us(1350.0) * 1.0e-6).abs() < 1e-15,
            "clock {clk}"
        );
        // Velocity is the cubic's exact derivative → recovered to sub-mm/s, reported in m/s.
        let vel = sp3.interpolate_velocity(g01, 1350.0, 3).unwrap();
        let want = cubic_velocity(1350.0);
        for axis in 0..3 {
            let want_ms = want[axis] * 1000.0;
            assert!(
                (vel[axis] - want_ms).abs() < 1e-6,
                "axis {axis}: {} vs {want_ms}",
                vel[axis]
            );
        }
    }

    #[test]
    fn a_position_only_file_has_no_velocity_and_the_clock_sentinel_is_skipped() {
        // Two epochs, position only, every clock the 999999.999999 sentinel.
        let text = "\
            #dP2024  1  9  0  0  0.00000000       2 ORBIT IGS20 HLM  IGS\n\
            *  2024  1  9  0  0  0.00000000\n\
            PG01  16000.000000  -5000.000000  20000.000000 999999.999999\n\
            *  2024  1  9  0 15  0.00000000\n\
            PG01  16100.000000  -5100.000000  20050.000000 999999.999999\n\
            EOF\n";
        let sp3 = Sp3::parse(text);
        let g01 = SatId {
            system: 'G',
            prn: 1,
        };
        assert!(sp3.interpolate(g01, 100.0, 1).is_some()); // position present
        assert!(sp3.interpolate_velocity(g01, 100.0, 1).is_none()); // no V records
        assert!(sp3.interpolate_clock(g01, 100.0, 1).is_none()); // all-sentinel clock skipped
    }

    // Reference Julian Day Numbers from CPython's proleptic-Gregorian calendar,
    // `date(y, m, d).toordinal() + 1721425` — an independent implementation, not Fliegel–Van
    // Flandern again. Two anchors fix the offset: 2000-01-01 is JDN 2451545 (the J2000.0 day) and
    // 1858-11-17 is JDN 2400001 (MJD 0).
    const JDN_CASES: [(i64, i64, i64, i64); 18] = [
        (1, 1, 1, 1_721_426),
        (1582, 10, 15, 2_299_161),
        (1858, 11, 17, 2_400_001),
        (1900, 2, 28, 2_415_079),
        (1900, 3, 1, 2_415_080), // 1900 is not a leap year: the yy/100 term
        (1999, 12, 31, 2_451_544),
        (2000, 1, 1, 2_451_545),
        (2000, 2, 29, 2_451_604),
        (2000, 3, 1, 2_451_605), // 2000 is a leap year: the yy/400 term
        (2024, 1, 9, 2_460_319),
        (2024, 1, 10, 2_460_320),
        (2024, 2, 28, 2_460_369),
        (2024, 2, 29, 2_460_370),
        (2024, 3, 1, 2_460_371), // the (14 - m)/12 branch flips between February and March
        (2024, 12, 31, 2_460_676),
        (2025, 1, 1, 2_460_677),
        (2100, 2, 28, 2_488_128),
        (2100, 3, 1, 2_488_129),
    ];

    #[test]
    fn jdn_matches_the_gregorian_calendar_at_every_boundary_that_matters() {
        // `parse` only ever uses `jdn` inside `secs - t0`, so a file whose epochs share one calendar
        // date cannot see it at all — every constant cancels, including a `jdn` that returns 0. This
        // pins the function itself against an external calendar.
        for (y, m, d, want) in JDN_CASES {
            assert_eq!(jdn(y, m, d), want, "jdn({y}, {m}, {d})");
        }
        // Consecutive days differ by exactly one, across the century and leap-day boundaries above.
        for w in JDN_CASES.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b.3 == a.3 + 1 {
                assert_eq!(jdn(b.0, b.1, b.2) - jdn(a.0, a.1, a.2), 1);
            }
        }
    }

    #[test]
    fn the_epoch_grid_spans_a_day_boundary_and_carries_hours_minutes_and_seconds() {
        // Three epochs across midnight, each with a different hour, minute and second, so no term of
        // `jdn·86400 + h·3600 + m·60 + s` is constant across the file and none can cancel:
        //   e1 = 86400 − (23·3600 + 45·60 + 30) = 86400 − 85530 = 870 s
        //   e2 = e1 + (1·3600 + 2·60 + 3.5)     = 870 + 3723.5   = 4593.5 s
        let text = "\
            #dP2024 12 31 23 45 30.00000000       3 ORBIT IGS20 HLM  IGS\n\
            *  2024 12 31 23 45 30.00000000\n\
            PG01  16000.000000  -5000.000000  20000.000000\n\
            *  2025  1  1  0  0  0.00000000\n\
            PG01  16100.000000  -5100.000000  20050.000000\n\
            *  2025  1  1  1  2  3.50000000\n\
            PG01  16200.000000  -5200.000000  20100.000000\n\
            EOF\n";
        let sp3 = Sp3::parse(text);
        assert_eq!(sp3.epochs, vec![0.0, 870.0, 4593.5], "epoch grid");
    }

    #[test]
    fn the_interpolation_window_slides_with_the_query_time() {
        // Twelve samples 100 s apart and `order = 3` leave eight possible window starts, so the
        // centring and the end clamp are both reachable. With a five-epoch file and `order = 3` there
        // are only two, and with two epochs there is exactly one — which is why every existing
        // fixture leaves this arithmetic unobservable.
        let epochs: Vec<f64> = (0..12).map(|i| f64::from(i) * 100.0).collect();
        // `want = order + 1 = 4`, `start = nearest − want/2`, clamped to `len − want = 8`. Ties in the
        // nearest-sample search go to the earlier sample (`min_by` keeps the first minimum).
        for (t, want_start) in [
            (0.0, 0),    // first sample: saturating_sub keeps the start at zero
            (250.0, 0),  // tie between samples 2 and 3 -> 2, minus 2 -> 0
            (450.0, 2),  // tie between samples 4 and 5 -> 4, minus 2 -> 2
            (650.0, 4),  // nearest 6, centred -> 4
            (1050.0, 8), // nearest 10, centred -> 8, at the clamp
            (1100.0, 8), // last sample: nearest 11, centred -> 9, clamped to 8
        ] {
            assert_eq!(window(&epochs, t, 3), Some((want_start, 4)), "t = {t}");
        }
        // An order past the sample count takes the whole grid and pins `want` to the length.
        assert_eq!(window(&epochs, 650.0, 50), Some((0, 12)));
    }

    #[test]
    fn short_records_are_skipped_and_a_clock_at_the_sentinel_is_rejected() {
        // Three guards none of the fixtures above can reach: an epoch line with fewer than six
        // numeric fields, a position record with no clock column at all, and a clock at exactly the
        // 999999 µs sentinel — which is bad, while 999998 µs is a real (if absurd) bias.
        let text = "\
            #dP2025  1  1  0  0  0.00000000       2 ORBIT IGS20 HLM  IGS\n\
            *  2025  1  1  0  0  0.00000000\n\
            PG01  16000.000000  -5000.000000  20000.000000\n\
            PG02  16000.000000  -5000.000000  20000.000000 999999.000000\n\
            PG03  16000.000000  -5000.000000  20000.000000 999998.000000\n\
            *  2025  1  1  0 15\n\
            *  2025  1  1  0 15  0.00000000\n\
            PG01  16100.000000  -5100.000000  20050.000000\n\
            PG02  16100.000000  -5100.000000  20050.000000 999999.000000\n\
            PG03  16100.000000  -5100.000000  20050.000000 999998.000000\n\
            EOF\n";
        let sp3 = Sp3::parse(text);
        let sat = |prn| SatId { system: 'G', prn };
        assert_eq!(
            sp3.epochs,
            vec![0.0, 900.0],
            "the five-field epoch line is not an epoch"
        );
        // A three-column position record is a valid record with no clock.
        assert!(sp3.interpolate(sat(1), 450.0, 1).is_some());
        assert!(sp3.interpolate_clock(sat(1), 450.0, 1).is_none());
        assert!(
            sp3.interpolate_clock(sat(2), 450.0, 1).is_none(),
            "999999 µs is the sentinel"
        );
        let clk = sp3.interpolate_clock(sat(3), 450.0, 1).unwrap();
        assert!(
            (clk - 999_998.0e-6).abs() < 1e-15,
            "999998 µs is a bias, not a sentinel: {clk}"
        );
    }

    #[test]
    fn unknown_satellite_has_no_interpolant() {
        let sp3 = Sp3::parse(&synthetic_sp3());
        assert!(
            sp3.interpolate(
                SatId {
                    system: 'R',
                    prn: 7
                },
                500.0,
                3
            )
            .is_none()
        );
    }
}
