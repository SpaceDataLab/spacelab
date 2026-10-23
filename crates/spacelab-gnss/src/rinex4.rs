//! RINEX 4.02 navigation files — every record type the format defines, read from its own columns.
//!
//! RINEX 4 changes one thing structurally and everything follows from it. A navigation record no
//! longer begins with a satellite identifier; it begins with a **Data Record Header Line**
//!
//! ```text
//! > EPH G01 LNAV
//! > STO R   FDMA
//! > ION I   L1NV KLOB
//! ```
//!
//! naming the record type, the transmitting satellite (or just its constellation) and the navigation
//! *message* the data came out of. Two consequences: a file can carry things that are not
//! ephemerides — system time offsets, Earth orientation, ionosphere models, broadcast by the
//! satellites and previously discarded — and one constellation can carry several different
//! ephemeris messages at once, with different fields and different lengths. GPS alone broadcasts
//! LNAV (8 lines), CNAV (9) and CNAV-2 (10), and they are not the same orbit representation:
//! LNAV gives `√A` and `Δn`, the CNAV family gives `√A₀`, `Ȧ`, `Δn₀` and `Δn̈₀`.
//!
//! So the block length is a function of the *message type*, not of the constellation, and a reader
//! that guesses it desynchronises the whole file. The table in [`block_len`] is read from the
//! specification, and it was checked a second way before any of this was written: the 148 789-line
//! DLR `BRD400DLR` product for 2026-07-19 was walked with the blocks defined as "everything up to
//! the next `>`", and the 36 (record, constellation, message) combinations it contains produced
//! exactly the lengths the specification gives. Two independent readings, one of them of real data.
//!
//! What this module does *not* do is decide what the numbers mean. The legacy Keplerian messages
//! (LNAV, FNAV, INAV, D1, D2) decode into the same [`BroadcastEphemeris`] the RINEX 3 reader
//! produces, field for field, because the first eight lines of those records are unchanged from
//! RINEX 3 — and `tests/rinex4_nav.rs` asserts exactly that against the two products for the same
//! day, which is the check that this reader and that one cannot have drifted apart.
//!
//! ## Columns
//!
//! Every value line in the format is `4X,4E19.12` — four spaces then four nineteen-column fields at
//! 0-based offsets 4, 23, 42 and 61 — and every epoch line puts its three values at 23, 42 and 61.
//! `STO`, `EOP` and `ION` records use the same columns as `EPH`, which is why
//! `rinex::orbit_fields` is shared rather than reimplemented here. A blank field is `0.0`;
//! the format uses blanks for spares and for values the satellite did not transmit. A field that is
//! not a number refuses its record.

// Fixed-column RINEX fields are cast between float and small integer types; the doc allow is for
// spec proper nouns (RINEX, BeiDou, GLONASS, NavIC, Klobuchar, NeQuick).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::module_name_repetitions
)]

use crate::glonass::GlonassEphemeris;
use crate::gps::{BroadcastEphemeris, GnssSystem};
use crate::rinex::{orbit_fields, parse_field, parse_glonass_record, parse_record};

/// Kilometres → metres, for the GLONASS and SBAS records' km / km·s⁻¹ / km·s⁻² fields.
const KM_TO_M: f64 = 1000.0;

/// What a navigation data record carries (specification Table 19).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordKind {
    /// Ephemerides: orbit, clock, biases, accuracy and status.
    Eph,
    /// System time and UTC proxy offset parameters.
    Sto,
    /// Earth orientation parameters.
    Eop,
    /// Global or regional ionospheric model parameters.
    Ion,
}

impl RecordKind {
    /// The three-character code, as it appears in the record header line.
    fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "EPH" => Self::Eph,
            "STO" => Self::Sto,
            "EOP" => Self::Eop,
            "ION" => Self::Ion,
            _ => return None,
        })
    }
}

/// The navigation message a record was decoded from (specification Tables 20 and 21).
///
/// One enum rather than two, although the specification tabulates ephemeris messages and
/// constellation messages separately: `LNAV`, `FDMA` and `L1NV` appear in both lists with the same
/// meaning, and the [`RecordKind`] beside it already says which list applies. The reduced-granularity
/// codes — `IFNV`, `D1D2`, `CNVX`, `LXOC` — stand for *groups* of messages that broadcast identical
/// constellation-wide data, and exist so a file need not repeat it once per signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NavMessage {
    /// GPS / QZSS / NavIC legacy navigation message.
    Lnav,
    /// GLONASS legacy FDMA message.
    Fdma,
    /// Galileo free navigation message (E5a).
    Fnav,
    /// Galileo integrity navigation message (E1, E5b).
    Inav,
    /// BeiDou-2/3 MEO/IGSO navigation message.
    D1,
    /// BeiDou-2/3 GEO navigation message.
    D2,
    /// SBAS navigation message.
    Sbas,
    /// GPS / QZSS CNAV navigation message (L2C, L5).
    Cnav,
    /// BeiDou-3 CNAV-1 navigation message (B1C).
    Cnv1,
    /// GPS / QZSS CNAV-2 (L1C) or BeiDou-3 CNAV-2 (B2a).
    Cnv2,
    /// BeiDou-3 CNAV-3 navigation message (B2b).
    Cnv3,
    /// NavIC L1 navigation message.
    L1nv,
    /// GLONASS L1 CDMA navigation message.
    L1oc,
    /// GLONASS L3 CDMA navigation message.
    L3oc,
    /// Galileo I/NAV and F/NAV taken together, for constellation-wide records.
    Ifnv,
    /// BeiDou D1 and D2 taken together, for constellation-wide records.
    D1d2,
    /// The CNAV family taken together, for constellation-wide records.
    Cnvx,
    /// GLONASS L1OC and L3OC taken together, for constellation-wide records.
    Lxoc,
}

impl NavMessage {
    /// The four-character code, as it appears in the record header line.
    fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "LNAV" => Self::Lnav,
            "FDMA" => Self::Fdma,
            "FNAV" => Self::Fnav,
            "INAV" => Self::Inav,
            "D1" => Self::D1,
            "D2" => Self::D2,
            "SBAS" => Self::Sbas,
            "CNAV" => Self::Cnav,
            "CNV1" => Self::Cnv1,
            "CNV2" => Self::Cnv2,
            "CNV3" => Self::Cnv3,
            "L1NV" => Self::L1nv,
            "L1OC" => Self::L1oc,
            "L3OC" => Self::L3oc,
            "IFNV" => Self::Ifnv,
            "D1D2" => Self::D1d2,
            "CNVX" => Self::Cnvx,
            "LXOC" => Self::Lxoc,
            _ => return None,
        })
    }
}

/// The message subtype, present only where Table 25 defines one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Subtype {
    /// QZSS wide-area ionosphere coefficients.
    Wide,
    /// QZSS Japan-area ionosphere coefficients.
    Japn,
    /// NavIC L1 Klobuchar ionosphere model.
    Klob,
    /// NavIC L1 NeQuick-N ionosphere model.
    Neqn,
}

impl Subtype {
    fn from_code(code: &str) -> Option<Self> {
        Some(match code {
            "WIDE" => Self::Wide,
            "JAPN" => Self::Japn,
            "KLOB" => Self::Klob,
            "NEQN" => Self::Neqn,
            _ => return None,
        })
    }
}

/// A record header line, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Header {
    kind: RecordKind,
    /// Constellation letter, as transmitted (`G`, `R`, `E`, `C`, `J`, `I`, `S`).
    constellation: char,
    /// Transmitting satellite, where the line names one. Constellation-wide records leave it blank.
    prn: Option<u8>,
    message: NavMessage,
    subtype: Option<Subtype>,
}

/// Decode a record header line: `A1, 1X,A3, 1X,A3, 1X,A4, 1X,A4`.
///
/// Read from the fixed columns rather than by splitting on whitespace, because the satellite field
/// is `A3` and is *blank-padded* on a constellation-wide record (`> STO R   FDMA`) — splitting would
/// put `FDMA` where the PRN belongs and then fail to find a message type at all.
fn parse_header(line: &str) -> Option<Header> {
    if !line.starts_with('>') {
        return None;
    }
    let kind = RecordKind::from_code(line.get(2..5)?.trim())?;
    let sat = line.get(6..9)?;
    let constellation = sat.chars().next()?;
    if !constellation.is_ascii_uppercase() {
        return None;
    }
    let prn = match sat.get(1..3)?.trim() {
        "" => None,
        digits => Some(digits.parse().ok()?),
    };
    let message = NavMessage::from_code(line.get(10..14).unwrap_or("").trim())?;
    let subtype = line
        .get(15..19)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .and_then(Subtype::from_code);
    Some(Header {
        kind,
        constellation,
        prn,
        message,
        subtype,
    })
}

/// How many lines follow a record header line, for each (record kind, message) pair.
///
/// `None` means the pair is not defined by the format, and the walk skips the header line alone
/// rather than guessing a length — guessing is how a reader loses the rest of the file.
///
/// The ephemeris lengths are the reason this function exists: within one constellation they differ
/// by message. GPS is 8 lines of LNAV, 9 of CNAV and 10 of CNAV-2; BeiDou is 8 of D1/D2, 10 of
/// CNAV-1 and CNAV-2, 9 of CNAV-3.
#[must_use]
pub fn block_len(kind: RecordKind, message: NavMessage) -> Option<usize> {
    use NavMessage as M;
    Some(match kind {
        RecordKind::Eph => match message {
            M::Lnav | M::Fnav | M::Inav | M::D1 | M::D2 => 8,
            M::Cnav | M::Cnv3 | M::L1nv | M::L1oc | M::L3oc => 9,
            M::Cnv1 | M::Cnv2 => 10,
            M::Fdma => 5,
            M::Sbas => 4,
            M::Ifnv | M::D1d2 | M::Cnvx | M::Lxoc => return None,
        },
        RecordKind::Sto => 2,
        RecordKind::Eop => 3,
        RecordKind::Ion => match message {
            // Klobuchar (8 coefficients) and BDGIM (9) both need three lines; NeQuick-G's three
            // `ai` and its disturbance flags need two; the GLONASS CDMA model's three fit on one.
            M::Lnav | M::D1d2 | M::Cnvx => 3,
            M::Ifnv => 2,
            M::Lxoc => 1,
            // NavIC's `L1NV` is deliberately absent: its two ionosphere models have different
            // lengths and only the subtype tells them apart, so `ion_block_len` answers for it.
            _ => return None,
        },
    })
}

/// NavIC's two `L1NV` ionosphere messages carry different models and different lengths, so the
/// subtype is part of the block length for them and for nothing else.
fn ion_block_len(message: NavMessage, subtype: Option<Subtype>) -> Option<usize> {
    match (message, subtype) {
        (NavMessage::L1nv, Some(Subtype::Klob)) => Some(4),
        (NavMessage::L1nv, Some(Subtype::Neqn)) => Some(7),
        (m, _) => block_len(RecordKind::Ion, m),
    }
}

/// A calendar epoch as the format writes it, with no time scale attached.
///
/// The scale is a property of the constellation and of the record — a GLONASS `Toc` is UTC(SU), a
/// GPS one is GPS time, an `STO` reference epoch is the system time of the transmitting
/// constellation — so attaching one here would be inventing information the columns do not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Epoch {
    /// Four-digit year.
    pub year: i32,
    /// Month, 1–12.
    pub month: u32,
    /// Day of month, 1–31.
    pub day: u32,
    /// Hour, 0–23.
    pub hour: u32,
    /// Minute, 0–59.
    pub minute: u32,
    /// Second, 0–59. The format writes it as `I2.2`, so it carries no fraction.
    pub second: u32,
}

/// Read the calendar from an epoch line whose year begins at column 4.
///
/// Both layouts in the format put the year there: `EPH` writes `A1,I2.2,1X,I4` (the satellite
/// identifier, then the year at 4) and `STO`/`EOP`/`ION` write `4X,I4`.
fn parse_epoch(line: &str) -> Option<Epoch> {
    Some(Epoch {
        year: line.get(4..8)?.trim().parse().ok()?,
        month: line.get(9..11)?.trim().parse().ok()?,
        day: line.get(12..14)?.trim().parse().ok()?,
        hour: line.get(15..17)?.trim().parse().unwrap_or(0),
        minute: line.get(18..20)?.trim().parse().unwrap_or(0),
        second: line.get(21..23)?.trim().parse().unwrap_or(0),
    })
}

/// A quasi-Keplerian ephemeris from the CNAV message family (CNAV, CNAV-1/2/3, NavIC L1NV).
///
/// The modern civil messages do not parameterise the orbit the way the legacy ones do, and the
/// difference is not cosmetic. LNAV broadcasts `√A` and a constant mean-motion correction `Δn`;
/// this family broadcasts a reference `√A₀` **with a rate** `Ȧ`, and a mean-motion correction that
/// is itself linear in time, `Δn₀ + ½·Δn̈₀·tₖ`. Over a fit interval that is what lets the message be
/// valid for longer at the same accuracy, and it is why these parameters cannot be poured into a
/// [`BroadcastEphemeris`] and propagated with the legacy equations: two of the terms would be
/// silently dropped.
///
/// `toe` is not a separate field. IS-GPS-200 §30.3.4.4 requires `Toe = Toc` for these messages, and
/// the format therefore transmits only `Toc`; the third orbit line carries `t_op`, the *prediction*
/// epoch, which is a different quantity and is kept as its own field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CnavEphemeris {
    /// Which constellation these parameters belong to.
    pub system: GnssSystem,
    /// Satellite PRN.
    pub prn: u8,
    /// The message these parameters were broadcast in.
    pub message: NavMessage,
    /// Time of clock, seconds of system week — and, for this family, the time of ephemeris too.
    pub toc: f64,
    /// Reference semi-major-axis square root `√A₀`, √m.
    pub sqrt_a0: f64,
    /// Semi-major-axis rate `Ȧ`, m/s.
    pub a_dot: f64,
    /// Mean-motion correction at the reference epoch `Δn₀`, rad/s.
    pub delta_n0: f64,
    /// Rate of the mean-motion correction `Δn̈₀`, rad/s².
    pub delta_n0_dot: f64,
    /// Mean anomaly at the reference epoch, rad.
    pub m0: f64,
    /// Eccentricity.
    pub e: f64,
    /// Argument of perigee, rad.
    pub omega: f64,
    /// Longitude of ascending node at weekly epoch, rad.
    pub omega0: f64,
    /// Rate of right ascension, rad/s.
    pub omega_dot: f64,
    /// Inclination at the reference epoch, rad.
    pub i0: f64,
    /// Rate of inclination, rad/s.
    pub idot: f64,
    /// Argument-of-latitude cosine correction, rad.
    pub cuc: f64,
    /// Argument-of-latitude sine correction, rad.
    pub cus: f64,
    /// Radius cosine correction, m.
    pub crc: f64,
    /// Radius sine correction, m.
    pub crs: f64,
    /// Inclination cosine correction, rad.
    pub cic: f64,
    /// Inclination sine correction, rad.
    pub cis: f64,
    /// Prediction epoch `t_op`, seconds of system week. `None` on NavIC L1NV, whose third orbit
    /// line carries `IODEC` in that column instead.
    pub t_op: Option<f64>,
    /// SV clock bias, s.
    pub af0: f64,
    /// SV clock drift, s/s.
    pub af1: f64,
    /// SV clock drift rate, s/s².
    pub af2: f64,
    /// SV health as broadcast, a per-signal bit field whose meaning is message-specific.
    pub health: u8,
    /// Group delay `TGD`, s.
    pub tgd: f64,
    /// Continuous week number `wn_op`, where the message carries one.
    pub week: Option<u16>,
}

impl CnavEphemeris {
    /// Semi-major axis at `tk` seconds from the reference epoch: `A(tₖ) = A₀ + Ȧ·tₖ`.
    ///
    /// The legacy message has no `Ȧ` and this reduces to `A₀`. Over a two-hour arc `Ȧ·tₖ` is
    /// centimetres to decimetres; over CNAV's longer fit interval it is the term that lets the
    /// message stay valid for it.
    #[must_use]
    pub fn semi_major_axis(&self, tk: f64) -> f64 {
        self.sqrt_a0 * self.sqrt_a0 + self.a_dot * tk
    }

    /// Corrected mean motion `n_A = n₀ + Δn₀ + ½·Δn̈₀·tₖ`, rad/s, with `n₀ = √(µ/A₀³)`.
    ///
    /// `n₀` is computed from `A₀` and not from `A(tₖ)` — IS-GPS-200 Table 30-II sheet 1 defines it
    /// that way, and the difference is not a rounding choice: `Ȧ` already carries the secular drift,
    /// so recomputing `n₀` from the drifting axis would count it twice.
    #[must_use]
    pub fn mean_motion(&self, tk: f64) -> f64 {
        let a0 = self.sqrt_a0 * self.sqrt_a0;
        let n0 = libm::sqrt(self.system.gravitational_parameter() / (a0 * a0 * a0));
        n0 + self.delta_n0 + 0.5 * self.delta_n0_dot * tk
    }

    /// ECEF position and velocity (`[m]`, `[m/s]`) at `tk` seconds from the reference epoch.
    ///
    /// IS-GPS-200 Table 30-II, the CNAV/CNAV-2/CNAV-3 user equations. Two terms distinguish it from
    /// the legacy algorithm in [`crate::gps::BroadcastEphemeris::position_velocity_ecef`]: the axis
    /// drifts as `A₀ + Ȧ·tₖ`, and the corrected radius rate gains `Ȧ(1 − e·cos Eₖ)`. Everything
    /// else — the harmonic corrections, the node rotation, the velocity chain — is identical, which
    /// is why `Ȧ = Δn̈₀ = 0` reproduces the legacy result exactly.
    ///
    /// **One term the specification's velocity drops, stated rather than hidden.** `Mₖ = M₀ + n_A·tₖ`
    /// with `n_A` itself linear in `tₖ` makes `dMₖ/dt = n_A + ½·Δn̈₀·tₖ`, but Table 30-II sheet 3
    /// defines `Ėₖ = n_A/(1 − e·cos Eₖ)` without that half-term. This follows the specification,
    /// because the specification is what every receiver computes and interoperability is the point
    /// of a broadcast message. The size of what is dropped, measured on the fixture's GPS records:
    /// `½·Δn̈₀·tₖ ≈ 5e-11 rad/s` two hours out against `n_A ≈ 1.46e-4`, so under 2 mm/s.
    ///
    /// `tk` is signed and is the caller's to unwrap across a week boundary, exactly as for the
    /// legacy message.
    // The single letters are the specification's own symbols; renaming them would make the code
    // harder to check against Table 30-II, which is the only thing that makes it reviewable.
    #[allow(clippy::many_single_char_names)]
    #[must_use]
    pub fn position_velocity_ecef(&self, tk: f64) -> ([f64; 3], [f64; 3]) {
        let a = self.semi_major_axis(tk);
        let e = self.e;
        let n = self.mean_motion(tk);

        let ek = crate::gps::kepler_solve(self.m0 + n * tk, e);
        let (sin_ek, cos_ek) = libm::sincos(ek);
        let one_minus_ecos = 1.0 - e * cos_ek;

        let sqrt_1me2 = libm::sqrt(1.0 - e * e);
        let nu = libm::atan2(sqrt_1me2 * sin_ek, cos_ek - e);
        let phi = nu + self.omega;
        let (sin2phi, cos2phi) = libm::sincos(2.0 * phi);

        // Second-harmonic corrections, accumulated exactly as `BroadcastEphemeris` accumulates
        // them: floating-point addition does not associate, and the equality this shares with the
        // legacy propagator when the rates are zero is worth having exact rather than approximate.
        let du = self.cus * sin2phi + self.cuc * cos2phi;
        let dr = self.crs * sin2phi + self.crc * cos2phi;
        let di = self.cis * sin2phi + self.cic * cos2phi;

        let u = phi + du;
        let r = a * one_minus_ecos + dr;
        let i = self.i0 + self.idot * tk + di;

        let (sin_u, cos_u) = libm::sincos(u);
        let xp = r * cos_u;
        let yp = r * sin_u;

        let omega_e = self.system.rotation_rate();
        let omega_k = self.omega0 + (self.omega_dot - omega_e) * tk - omega_e * self.toc;
        let (sin_o, cos_o) = libm::sincos(omega_k);
        let (sin_i, cos_i) = libm::sincos(i);

        let position = [
            xp * cos_o - yp * cos_i * sin_o,
            xp * sin_o + yp * cos_i * cos_o,
            yp * sin_i,
        ];

        let ek_dot = n / one_minus_ecos;
        let nu_dot = ek_dot * sqrt_1me2 / one_minus_ecos;
        let u_dot = nu_dot * (1.0 + 2.0 * (self.cus * cos2phi - self.cuc * sin2phi));
        // The CNAV radius rate, Table 30-II sheet 3: the legacy two terms plus `Ȧ(1 − e·cos Eₖ)`.
        let r_dot = self.a_dot * one_minus_ecos
            + a * e * sin_ek * ek_dot
            + 2.0 * nu_dot * (self.crs * cos2phi - self.crc * sin2phi);
        let i_dot = self.idot + 2.0 * nu_dot * (self.cis * cos2phi - self.cic * sin2phi);

        let xp_dot = r_dot * cos_u - yp * u_dot;
        let yp_dot = r_dot * sin_u + xp * u_dot;
        let omega_k_dot = self.omega_dot - omega_e;

        let velocity = [
            xp_dot * cos_o - yp_dot * cos_i * sin_o + yp * sin_i * sin_o * i_dot
                - position[1] * omega_k_dot,
            xp_dot * sin_o + yp_dot * cos_i * cos_o - yp * sin_i * cos_o * i_dot
                + position[0] * omega_k_dot,
            yp_dot * sin_i + yp * cos_i * i_dot,
        ];

        (position, velocity)
    }

    /// ECEF position (metres) at `tk` seconds from the reference epoch.
    #[must_use]
    pub fn position_ecef(&self, tk: f64) -> [f64; 3] {
        self.position_velocity_ecef(tk).0
    }
}

/// A legacy Keplerian ephemeris together with the message it was broadcast in.
///
/// The message type is not decoration, and this is the one place in the format where dropping it
/// loses information that cannot be recovered. Galileo transmits its ephemeris in both I/NAV and
/// F/NAV, and the *orbit* is the same in the two — but the clock is not: F/NAV carries the E5a−E1
/// ionosphere-free calibration and I/NAV the E5b−E1 one, so `af0` differs between them by the
/// broadcast group delay. Measured on DLR's product for 2024-01-01: **2 659 of 2 660 Galileo
/// (satellite, `toe`) pairs carry two records**, bit-identical in every one of the fifteen orbit
/// elements and in `toe`, `toc` and the week — and differing in `af0` on 2 584 of them (worst
/// 1.9e-9 s, 0.58 m of range), in the health word on 214, and in `BGD E5a/E1` on 39.
///
/// A `Vec<BroadcastEphemeris>` cannot express that. A caller looking up "E08 at `toe` 86 400" would
/// get whichever of the two happened to be later in the file, and be a third of a metre out half the
/// time with nothing to indicate it. RINEX 3 has the same ambiguity and no way at all to resolve it:
/// its records carry the frequency pair in bits 8 and 9 of the data-source field, which
/// [`BroadcastEphemeris`] does not keep. RINEX 4 states it in the record header, so this reader can.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KeplerianRecord {
    /// The message the record was broadcast in — `Lnav`, `Fnav`, `Inav`, `D1` or `D2`.
    pub message: NavMessage,
    /// The ephemeris, decoded exactly as the RINEX 3 reader decodes the same eight lines.
    pub ephemeris: BroadcastEphemeris,
}

/// An SBAS ephemeris: an ECEF state vector with a constant acceleration, not a Keplerian set.
///
/// The geostationary augmentation satellites broadcast their own position directly, in the same
/// shape as GLONASS but in an Earth-fixed frame and valid over minutes rather than half an hour.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SbasEphemeris {
    /// PRN as transmitted (the format writes PRN − 100).
    pub prn: u8,
    /// Time of clock, seconds of day.
    pub toc: f64,
    /// Clock bias `aGf0`, s.
    pub af0: f64,
    /// Relative frequency bias `aGf1`, s/s.
    pub af1: f64,
    /// Transmission time of message, seconds of GPS week.
    pub transmission_time: f64,
    /// ECEF position, m.
    pub position: [f64; 3],
    /// ECEF velocity, m/s.
    pub velocity: [f64; 3],
    /// ECEF acceleration, m/s².
    pub acceleration: [f64; 3],
    /// Health word.
    pub health: u8,
    /// User range accuracy code, as broadcast.
    ///
    /// `u16` rather than `u8` because the column is not a URA *index*: measured over one day of
    /// BRD400DLR, it takes 2, 4096 and 32767, the last being the "not available" sentinel that
    /// 8 123 of the day's 9 410 SBAS records carry. Truncating to a byte makes 4096 and 32767
    /// both read as 255.
    pub accuracy_code: u16,
    /// Issue of data, navigation.
    pub iodn: u16,
}

/// A GLONASS CDMA ephemeris (L1OC or L3OC): the PZ-90 state, with the CDMA message's extra terms.
///
/// Structurally the FDMA record with five more lines. The fields kept here are the ones that change
/// an orbit or a clock; the attitude, yaw-rate and phase-centre blocks the message also carries are
/// not, because nothing in this repository consumes them and a field nobody reads is a field nobody
/// notices is wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlonassCdmaEphemeris {
    /// Orbital slot.
    pub slot: u8,
    /// The message these parameters were broadcast in — `L1oc` or `L3oc`.
    pub message: NavMessage,
    /// Reference epoch, seconds of day, UTC(SU).
    pub toe: f64,
    /// PZ-90 position, m.
    pub position: [f64; 3],
    /// PZ-90 velocity, m/s.
    pub velocity: [f64; 3],
    /// PZ-90 luni-solar acceleration, m/s².
    pub luni_solar: [f64; 3],
    /// Clock bias `−τₙ`, s.
    pub tau: f64,
    /// Relative frequency offset `+γₙ`.
    pub gamma: f64,
    /// Clock drift rate `β`, s/s².
    pub beta: f64,
    /// Signal health (0 = healthy).
    pub health: u8,
    /// Data validity (0 = valid).
    pub data_invalid: u8,
    /// `TGD_L2OCp` group delay, s.
    pub tgd_l2ocp: f64,
    /// Age of ephemeris data, days.
    pub age_of_ephemeris: f64,
    /// Age of clock data, days.
    pub age_of_clock: f64,
}

/// A system time offset record: one constellation's broadcast of its offset to another time scale.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemTimeOffset {
    /// Constellation that broadcast it.
    pub constellation: char,
    /// Transmitting satellite, where the record names one.
    pub prn: Option<u8>,
    /// The message it was decoded from.
    pub message: NavMessage,
    /// Reference epoch for the offset.
    pub epoch: Epoch,
    /// The four-character correction type: two two-letter system codes, e.g. `GPUT` for GPS−UTC or
    /// `GLGP` for GLONASS−GPS.
    pub correction: String,
    /// SBAS provider identifier, present only on an `SBUT` record.
    pub sbas_id: String,
    /// UTC laboratory identifier, present only where the offset is to a UTC realisation.
    pub utc_id: String,
    /// Transmission time of message, seconds from the start of the reference epoch's week. The
    /// format permits it to be negative or to exceed 604 800.
    pub transmission_time: f64,
    /// Constant term `A0`, s.
    pub a0: f64,
    /// Rate `A1`, s/s.
    pub a1: f64,
    /// Acceleration `A2`, s/s².
    pub a2: f64,
}

/// Earth orientation parameters as a constellation broadcasts them.
///
/// The pole coordinates come with a rate and an acceleration, in arc-seconds per day and per day
/// squared, which is the navigation message's own parameterisation and not the IERS one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthOrientation {
    /// Constellation that broadcast it.
    pub constellation: char,
    /// Transmitting satellite, where the record names one.
    pub prn: Option<u8>,
    /// The message it was decoded from.
    pub message: NavMessage,
    /// Reference epoch of the parameters.
    pub epoch: Epoch,
    /// `Xp`, `dXp/dt`, `d²Xp/dt²` — arc-seconds, arc-seconds/day, arc-seconds/day².
    pub x_pole: [f64; 3],
    /// `Yp`, `dYp/dt`, `d²Yp/dt²` — same units.
    pub y_pole: [f64; 3],
    /// Transmission time of message, seconds from the start of the reference epoch's week.
    pub transmission_time: f64,
    /// `ΔUT1`, `dΔUT1/dt`, `d²ΔUT1/dt²` — seconds, seconds/day, seconds/day².
    ///
    /// The specification warns that this is UT1−UTC for some constellations and ICD versions and
    /// UT1−GPST for others, distinguishable only by magnitude, so the caller has to decide.
    pub delta_ut1: [f64; 3],
}

/// The ionosphere model a record carries, and its coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IonosphereModel {
    /// The eight-coefficient Klobuchar model (GPS, BeiDou D1/D2, QZSS, NavIC).
    Klobuchar {
        /// `α₀…α₃`, s and s/semicircleⁿ.
        alpha: [f64; 4],
        /// `β₀…β₃`, s and s/semicircleⁿ.
        beta: [f64; 4],
        /// The region the coefficients apply in, present only on NavIC L1 (Table A38). Every
        /// other constellation broadcasts a global Klobuchar and leaves this `None`.
        region: Option<Region>,
    },
    /// Galileo's NeQuick-G effective ionisation level.
    NequickG {
        /// `a₀`, `a₁`, `a₂` — sfu, sfu/deg, sfu/deg².
        ai: [f64; 3],
        /// The five-bit disturbance flag field, as broadcast.
        disturbance_flags: u8,
    },
    /// NavIC's regional NeQuick-N model (Table A39): three regions, each with its own effective
    /// ionisation level and the longitude/MODIP box it applies in.
    ///
    /// A different quantity from [`IonosphereModel::NequickG`] despite the shared name --
    /// Galileo broadcasts one global `aᵢ` triple, NavIC broadcasts three regional ones.
    NequickN {
        /// The three regions, in the order the message transmits them.
        regions: [NequickRegion; 3],
    },
    /// BeiDou's nine-coefficient global ionospheric delay correction model.
    Bdgim {
        /// `α₁…α₉`, TECu.
        alpha: [f64; 9],
    },
    /// The GLONASS CDMA model's three coefficients.
    GlonassCdma {
        /// `c_A`, `c_F10.7`, `c_Ap`.
        c: [f64; 3],
    },
}

/// The longitude/latitude box a regional ionosphere message applies in, degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    /// Minimum and maximum longitude.
    pub longitude: [f64; 2],
    /// Minimum and maximum latitude, or modified dip latitude on a NeQuick-N region.
    pub latitude: [f64; 2],
}

/// One region of NavIC's NeQuick-N message (Table A39).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NequickRegion {
    /// `a₀`, `a₁`, `a₂` -- sfu, sfu/deg, sfu/deg².
    pub ai: [f64; 3],
    /// Ionospheric disturbance flag, as broadcast.
    pub disturbance_flag: u8,
    /// The longitude and MODIP bounds the coefficients apply in.
    pub bounds: Region,
}

/// One ionosphere record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ionosphere {
    /// Constellation that broadcast it.
    pub constellation: char,
    /// Transmitting satellite, where the record names one.
    pub prn: Option<u8>,
    /// The message it was decoded from.
    pub message: NavMessage,
    /// Regional subtype, where the message defines one — `WIDE` or `JAPN` for QZSS, `KLOB` or
    /// `NEQN` for NavIC L1.
    pub subtype: Option<Subtype>,
    /// Transmit time of the ionosphere data.
    pub epoch: Epoch,
    /// Issue of data, on the two NavIC L1 messages that carry one (`IODK`, `IODN`). No other
    /// ionosphere message has a field for it.
    pub issue_of_data: Option<f64>,
    /// The model and its coefficients.
    pub model: IonosphereModel,
}

/// Everything a RINEX 4 navigation file holds, sorted by record type rather than by file order.
///
/// Separate vectors rather than one `Vec<NavRecord>` because the four record types answer four
/// unrelated questions and every caller wants one of them. Within each vector the order is the
/// file's.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rinex4Nav {
    /// Format version from the header, e.g. `4.02`.
    pub version: f64,
    /// Current number of leap seconds since 1980-01-06, where the header states one.
    pub leap_seconds: Option<i32>,
    /// Legacy Keplerian ephemerides — LNAV, FNAV, INAV, D1, D2 — decoded exactly as RINEX 3, each
    /// tagged with the message it came from. See [`KeplerianRecord`] for why the tag is load-bearing.
    pub keplerian: Vec<KeplerianRecord>,
    /// CNAV-family ephemerides, which carry `Ȧ` and `Δn̈₀` and are not the same parameterisation.
    pub cnav: Vec<CnavEphemeris>,
    /// GLONASS FDMA ephemerides.
    pub glonass: Vec<GlonassEphemeris>,
    /// GLONASS CDMA (L1OC, L3OC) ephemerides.
    pub glonass_cdma: Vec<GlonassCdmaEphemeris>,
    /// SBAS ephemerides.
    pub sbas: Vec<SbasEphemeris>,
    /// System time offset records.
    pub time_offsets: Vec<SystemTimeOffset>,
    /// Earth orientation records.
    pub earth_orientation: Vec<EarthOrientation>,
    /// Ionosphere model records.
    pub ionosphere: Vec<Ionosphere>,
    /// Records whose header line was well formed but whose body could not be decoded, by line
    /// number. An empty vector is the assertion worth making about a real file: nothing was lost.
    pub rejected: Vec<usize>,
}

/// Why a file could not be read at all.
///
/// Only whole-file failures are errors. A single unreadable record is recorded in
/// [`Rinex4Nav::rejected`] and the walk continues, because a navigation file is a concatenation of
/// independent messages and one corrupt satellite is not a reason to discard the other ninety.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rinex4Error {
    /// No `RINEX VERSION / TYPE` record, or its file-type column is not `N`.
    NotNavigationData,
    /// The version is outside the 4.x this reader implements.
    UnsupportedVersion {
        /// Major version the file declares.
        major: u8,
    },
    /// The header never terminates with `END OF HEADER`.
    NoEndOfHeader,
}

impl core::fmt::Display for Rinex4Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotNavigationData => f.write_str(
                "not a RINEX navigation file: no `RINEX VERSION / TYPE` record of type N",
            ),
            Self::UnsupportedVersion { major } => {
                write!(f, "RINEX major version {major} is not 4")
            }
            Self::NoEndOfHeader => f.write_str("the header has no `END OF HEADER` record"),
        }
    }
}

impl core::error::Error for Rinex4Error {}

/// The header label a line carries, from columns 61-80.
///
/// Read from the columns rather than by searching the whole line, because a `COMMENT` is free text
/// and a comment mentioning `END OF HEADER` would otherwise end the header. Trailing padding to
/// column 80 is optional in practice -- the specification's own examples in Table A8 stop at the end
/// of the label -- so the slice is open-ended and trimmed.
/// The longest record body the format defines -- CNAV-1 and CNAV-2, ten lines including the epoch
/// line. Only a capacity hint; nothing depends on it being right.
const MAX_BLOCK_LEN: usize = 10;

fn label(line: &str) -> &str {
    line.get(60..).unwrap_or("").trim_end()
}

/// The constellation letter, as a [`GnssSystem`] where the Keplerian model applies to it.
fn keplerian_system(letter: char) -> Option<GnssSystem> {
    Some(match letter {
        'G' => GnssSystem::Gps,
        'E' => GnssSystem::Galileo,
        'C' => GnssSystem::Beidou,
        'J' => GnssSystem::Qzss,
        'I' => GnssSystem::Navic,
        _ => return None,
    })
}

/// Parse a RINEX 4 navigation file.
///
/// # Errors
///
/// [`Rinex4Error`] if the header does not identify a version-4 navigation file, or never ends.
/// Individual records that fail to decode are counted in [`Rinex4Nav::rejected`] rather than
/// failing the file.
pub fn parse(text: &str) -> Result<Rinex4Nav, Rinex4Error> {
    let lines: Vec<&str> = text.lines().collect();

    let version_line = lines
        .iter()
        .find(|l| label(l) == "RINEX VERSION / TYPE")
        .ok_or(Rinex4Error::NotNavigationData)?;
    let version: f64 = version_line
        .get(0..9)
        .and_then(|s| s.trim().parse().ok())
        .ok_or(Rinex4Error::NotNavigationData)?;
    let major = version as u8;
    if major != 4 {
        return Err(Rinex4Error::UnsupportedVersion { major });
    }
    if version_line.get(20..21).map(str::trim) != Some("N") {
        return Err(Rinex4Error::NotNavigationData);
    }

    let end = lines
        .iter()
        .position(|l| label(l) == "END OF HEADER")
        .ok_or(Rinex4Error::NoEndOfHeader)?;

    let mut out = Rinex4Nav {
        version,
        leap_seconds: lines[..end]
            .iter()
            .find(|l| label(l) == "LEAP SECONDS")
            .and_then(|l| l.get(0..6)?.trim().parse().ok()),
        ..Rinex4Nav::default()
    };

    // The walk consumes the lines rather than indexing them. A hand-written cursor is the shape
    // that `reseal-gnss-obs` had to retire from `RinexObs::parse`: `idx += 1` frozen to `idx *= 1`
    // stops a skip skipping, and once a test makes the skip branch reachable the mutant is a hang
    // rather than a wrong answer. Consuming from an iterator leaves no counter to freeze -- the
    // header line comes off `records.next()` and the body off `records.by_ref().take(len)`, so the
    // walk cannot fail to advance.
    let mut records = lines.iter().enumerate().skip(end + 1);
    let mut body: Vec<&str> = Vec::with_capacity(MAX_BLOCK_LEN);
    while let Some((idx, line)) = records.next() {
        let Some(header) = parse_header(line) else {
            continue;
        };
        let len = match header.kind {
            RecordKind::Ion => ion_block_len(header.message, header.subtype),
            kind => block_len(kind, header.message),
        };
        let Some(len) = len else {
            continue;
        };
        body.clear();
        body.extend(records.by_ref().take(len).map(|(_, l)| *l));
        // A block that runs off the end of the file is reported by the line its header sits on;
        // decoding a partial one would invent the columns the file does not have.
        if body.len() < len || !decode(&header, &body, &mut out) {
            out.rejected.push(idx + 1);
        }
    }

    Ok(out)
}

/// Decode one record body into `out`. `false` means the columns did not hold what the header
/// promised, and the caller records the line rather than inventing a value.
fn decode(header: &Header, body: &[&str], out: &mut Rinex4Nav) -> bool {
    match header.kind {
        RecordKind::Eph => decode_eph(header, body, out),
        RecordKind::Sto => decode_sto(header, body, out),
        RecordKind::Eop => decode_eop(header, body, out),
        RecordKind::Ion => decode_ion(header, body, out),
    }
}

fn decode_eph(header: &Header, body: &[&str], out: &mut Rinex4Nav) -> bool {
    use NavMessage as M;
    match header.message {
        // The legacy Keplerian messages are, line for line, the RINEX 3 record. Reusing that
        // decoder is the point rather than a convenience: it is what makes "the same orbit as the
        // RINEX 3 twin" a property of one piece of code instead of a coincidence between two.
        M::Lnav | M::Fnav | M::Inav | M::D1 | M::D2 => {
            let Some(system) = keplerian_system(header.constellation) else {
                return false;
            };
            match parse_record(body, system) {
                Some(ephemeris) => {
                    out.keplerian.push(KeplerianRecord {
                        message: header.message,
                        ephemeris,
                    });
                    true
                }
                None => false,
            }
        }
        M::Cnav | M::Cnv1 | M::Cnv2 | M::Cnv3 | M::L1nv => match decode_cnav(header, body) {
            Some(eph) => {
                out.cnav.push(eph);
                true
            }
            None => false,
        },
        M::Fdma => match parse_glonass_record(&body[..4]) {
            Some(eph) => {
                out.glonass.push(eph);
                true
            }
            None => false,
        },
        M::L1oc | M::L3oc => match decode_glonass_cdma(header, body) {
            Some(eph) => {
                out.glonass_cdma.push(eph);
                true
            }
            None => false,
        },
        M::Sbas => match decode_sbas(body) {
            Some(eph) => {
                out.sbas.push(eph);
                true
            }
            None => false,
        },
        M::Ifnv | M::D1d2 | M::Cnvx | M::Lxoc => false,
    }
}

fn decode_cnav(header: &Header, body: &[&str]) -> Option<CnavEphemeris> {
    let system = keplerian_system(header.constellation)?;
    let epoch = body[0];
    let prn: u8 = epoch.get(1..3)?.trim().parse().ok()?;
    let cal = parse_epoch(epoch)?;
    let toc = f64::from(
        crate::rinex::day_of_week(cal.year, cal.month, cal.day) * 86_400
            + cal.hour * 3600
            + cal.minute * 60
            + cal.second,
    );

    let af0 = parse_field(epoch, 23, 19)?;
    let af1 = parse_field(epoch, 42, 19)?;
    let af2 = parse_field(epoch, 61, 19)?;

    let o1 = orbit_fields(body[1])?; // A DOT, Crs, Δn0, M0
    let o2 = orbit_fields(body[2])?; // Cuc, e, Cus, √A0
    let o3 = orbit_fields(body[3])?; // t_op (or IODEC on L1NV), Cic, Ω0, Cis
    let o4 = orbit_fields(body[4])?; // i0, Crc, ω, Ω̇
    let o5 = orbit_fields(body[5])?; // IDOT, Δn0-dot, then message-specific
    let o6 = orbit_fields(body[6])?; // URAI_ED / URAI, health, TGD, URAI_NED2

    // The third orbit line's first column is `t_op` on every message in this family except NavIC
    // L1NV, where Table A31 puts `IODEC` there and the message carries no prediction epoch at all.
    let t_op = (header.message != NavMessage::L1nv).then_some(o3[0]);
    // `wn_op` sits on the last line of CNAV and CNAV-2; CNAV-3 and L1NV do not carry it.
    let week = match header.message {
        NavMessage::Cnav | NavMessage::Cnv2 => orbit_fields(body[body.len() - 1])
            .map(|last| last[1])
            .filter(|w| *w > 0.0)
            .map(|w| w as u16),
        _ => None,
    };

    Some(CnavEphemeris {
        system,
        prn,
        message: header.message,
        toc,
        sqrt_a0: o2[3],
        a_dot: o1[0],
        delta_n0: o1[2],
        delta_n0_dot: o5[1],
        m0: o1[3],
        e: o2[1],
        omega: o4[2],
        omega0: o3[2],
        omega_dot: o4[3],
        i0: o4[0],
        idot: o5[0],
        cuc: o2[0],
        cus: o2[2],
        crc: o4[1],
        crs: o1[1],
        cic: o3[1],
        cis: o3[3],
        t_op,
        af0,
        af1,
        af2,
        health: o6[1] as u8,
        tgd: o6[2],
        week,
    })
}

fn decode_glonass_cdma(header: &Header, body: &[&str]) -> Option<GlonassCdmaEphemeris> {
    let epoch = body[0];
    let slot: u8 = epoch.get(1..3)?.trim().parse().ok()?;
    let cal = parse_epoch(epoch)?;
    let toe = f64::from(cal.hour * 3600 + cal.minute * 60 + cal.second);

    let tau = parse_field(epoch, 23, 19)?;
    let gamma = parse_field(epoch, 42, 19)?;
    let beta = parse_field(epoch, 61, 19)?;

    let l1 = orbit_fields(body[1])?; // X,  X',  X'',  signal health
    let l2 = orbit_fields(body[2])?; // Y,  Y',  Y'',  data validity
    let l3 = orbit_fields(body[3])?; // Z,  Z',  Z'',  TGD_L2OCp
    let l4 = orbit_fields(body[4])?; // satellite type, source flags, AODE, AODC

    Some(GlonassCdmaEphemeris {
        slot,
        message: header.message,
        toe,
        position: [l1[0] * KM_TO_M, l2[0] * KM_TO_M, l3[0] * KM_TO_M],
        velocity: [l1[1] * KM_TO_M, l2[1] * KM_TO_M, l3[1] * KM_TO_M],
        luni_solar: [l1[2] * KM_TO_M, l2[2] * KM_TO_M, l3[2] * KM_TO_M],
        tau,
        gamma,
        beta,
        health: l1[3] as u8,
        data_invalid: l2[3] as u8,
        tgd_l2ocp: l3[3],
        age_of_ephemeris: l4[2],
        age_of_clock: l4[3],
    })
}

fn decode_sbas(body: &[&str]) -> Option<SbasEphemeris> {
    let epoch = body[0];
    let prn: u8 = epoch.get(1..3)?.trim().parse().ok()?;
    let cal = parse_epoch(epoch)?;
    let toc = f64::from(cal.hour * 3600 + cal.minute * 60 + cal.second);

    let af0 = parse_field(epoch, 23, 19)?;
    let af1 = parse_field(epoch, 42, 19)?;
    let transmission_time = parse_field(epoch, 61, 19)?;

    let l1 = orbit_fields(body[1])?; // X,  X',  X'',  health
    let l2 = orbit_fields(body[2])?; // Y,  Y',  Y'',  accuracy code
    let l3 = orbit_fields(body[3])?; // Z,  Z',  Z'',  IODN

    Some(SbasEphemeris {
        prn,
        toc,
        af0,
        af1,
        transmission_time,
        position: [l1[0] * KM_TO_M, l2[0] * KM_TO_M, l3[0] * KM_TO_M],
        velocity: [l1[1] * KM_TO_M, l2[1] * KM_TO_M, l3[1] * KM_TO_M],
        acceleration: [l1[2] * KM_TO_M, l2[2] * KM_TO_M, l3[2] * KM_TO_M],
        health: l1[3] as u8,
        accuracy_code: l2[3] as u16,
        iodn: l3[3] as u16,
    })
}

/// One of the three left-justified `A18` text fields on an `STO` record's epoch line.
fn text_field(line: &str, start: usize) -> String {
    line.get(start..start + 18).unwrap_or("").trim().to_string()
}

fn decode_sto(header: &Header, body: &[&str], out: &mut Rinex4Nav) -> bool {
    let Some(epoch) = parse_epoch(body[0]) else {
        return false;
    };
    let Some(values) = orbit_fields(body[1]) else {
        return false;
    };
    out.time_offsets.push(SystemTimeOffset {
        constellation: header.constellation,
        prn: header.prn,
        message: header.message,
        epoch,
        correction: text_field(body[0], 24),
        sbas_id: text_field(body[0], 43),
        utc_id: text_field(body[0], 62),
        transmission_time: values[0],
        a0: values[1],
        a1: values[2],
        a2: values[3],
    });
    true
}

fn decode_eop(header: &Header, body: &[&str], out: &mut Rinex4Nav) -> bool {
    let Some(epoch) = parse_epoch(body[0]) else {
        return false;
    };
    // Line 0 is `4X,I4,5(1X,I2.2),3E19.12` and line 1 is `4X,A19,3E19.12` — the spare in the first
    // column of line 1 is why both are read at 23/42/61 rather than through `orbit_fields`.
    let Some(x) = triple(body[0]) else {
        return false;
    };
    let Some(y) = triple(body[1]) else {
        return false;
    };
    let Some(t) = orbit_fields(body[2]) else {
        return false;
    };
    out.earth_orientation.push(EarthOrientation {
        constellation: header.constellation,
        prn: header.prn,
        message: header.message,
        epoch,
        x_pole: x,
        y_pole: y,
        transmission_time: t[0],
        delta_ut1: [t[1], t[2], t[3]],
    });
    true
}

/// The three `E19.12` fields at columns 23, 42 and 61 — the value part of an epoch line, and of any
/// line whose first column is a spare.
fn triple(line: &str) -> Option<[f64; 3]> {
    Some([
        parse_field(line, 23, 19)?,
        parse_field(line, 42, 19)?,
        parse_field(line, 61, 19)?,
    ])
}

fn decode_ion(header: &Header, body: &[&str], out: &mut Rinex4Nav) -> bool {
    let Some(epoch) = parse_epoch(body[0]) else {
        return false;
    };
    let Some(first) = triple(body[0]) else {
        return false;
    };
    let mut issue_of_data = None;
    let model = match (header.message, header.subtype) {
        // Galileo's NeQuick-G: three `ai` on the epoch line, the disturbance flags on the next.
        (NavMessage::Ifnv, _) => {
            let Some(flags) = parse_field(body[1], 4, 19) else {
                return false;
            };
            IonosphereModel::NequickG {
                ai: first,
                disturbance_flags: flags as u8,
            }
        }
        // BeiDou's BDGIM, nine coefficients over three lines.
        (NavMessage::Cnvx, _) if header.constellation == 'C' => {
            let (Some(second), Some(third)) = (orbit_fields(body[1]), orbit_fields(body[2])) else {
                return false;
            };
            IonosphereModel::Bdgim {
                alpha: [
                    first[0], first[1], first[2], second[0], second[1], second[2], second[3],
                    third[0], third[1],
                ],
            }
        }
        (NavMessage::Lxoc, _) => IonosphereModel::GlonassCdma { c: first },
        // NavIC L1 broadcasts the same eight Klobuchar coefficients on a different set of lines:
        // Table A38 puts `IODK` alone on the epoch line, all four `α` on line 1, all four `β` on
        // line 2, and the region they apply in on line 3. Reading it with the layout below would
        // return `IODK` as `α₀`.
        (NavMessage::L1nv, Some(Subtype::Klob)) => {
            let (Some(a), Some(b), Some(box_)) = (
                orbit_fields(body[1]),
                orbit_fields(body[2]),
                orbit_fields(body[3]),
            ) else {
                return false;
            };
            issue_of_data = Some(first[0]);
            IonosphereModel::Klobuchar {
                alpha: a,
                beta: b,
                region: Some(Region {
                    longitude: [box_[0], box_[1]],
                    latitude: [box_[2], box_[3]],
                }),
            }
        }
        // NavIC's NeQuick-N, Table A39: three regions, each a coefficient line followed by the
        // longitude/MODIP box it covers.
        (NavMessage::L1nv, Some(Subtype::Neqn)) => {
            let mut regions = [NequickRegion {
                ai: [0.0; 3],
                disturbance_flag: 0,
                bounds: Region {
                    longitude: [0.0; 2],
                    latitude: [0.0; 2],
                },
            }; 3];
            let (pairs, _) = body[1..].as_chunks::<2>();
            for (r, lines) in regions.iter_mut().zip(pairs) {
                let (Some(a), Some(box_)) = (orbit_fields(lines[0]), orbit_fields(lines[1])) else {
                    return false;
                };
                r.ai = [a[0], a[1], a[2]];
                r.disturbance_flag = a[3] as u8;
                r.bounds = Region {
                    longitude: [box_[0], box_[1]],
                    latitude: [box_[2], box_[3]],
                };
            }
            issue_of_data = Some(first[0]);
            IonosphereModel::NequickN { regions }
        }
        // Everything else is Klobuchar: `α₀…α₂` on the epoch line, `α₃` and `β₀…β₂` on the next,
        // `β₃` on the last.
        _ => {
            let (Some(second), Some(third)) = (orbit_fields(body[1]), orbit_fields(body[2])) else {
                return false;
            };
            IonosphereModel::Klobuchar {
                alpha: [first[0], first[1], first[2], second[0]],
                beta: [second[1], second[2], second[3], third[0]],
                region: None,
            }
        }
    };
    out.ionosphere.push(Ionosphere {
        constellation: header.constellation,
        prn: header.prn,
        message: header.message,
        subtype: header.subtype,
        epoch,
        issue_of_data,
        model,
    });
    true
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::float_cmp)]
    use super::*;

    /// The specification's own example records (Table A12, A32, A41), verbatim.
    ///
    /// Written by the format's authors rather than by this reader, which is the property that
    /// matters: a fixture generated here would agree with whatever this reader believes about the
    /// columns. The header is Table A8's.
    const SPEC_EXAMPLES: &str = concat!(
        "     4.02           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n",
        "    18    18 1929      7                                    LEAP SECONDS\n",
        "                                                            END OF HEADER\n",
        "> EPH G04 LNAV\n",
        "G04 2019 03 14 04 00 00 1.330170780420e-04 7.275957614183e-12 0.000000000000e+00\n",
        "     9.800000000000e+01-1.718750000000e+00 4.639836124941e-09 2.148941747752e+00\n",
        "    -1.881271600723e-07 3.355251392350e-04 8.245930075645e-06 5.153800453186e+03\n",
        "     3.600000000000e+05-1.676380634308e-08 5.171400020311e-01 1.490116119385e-08\n",
        "     9.601921900531e-01 2.187187500000e+02-1.736906885738e+00-8.044977962767e-09\n",
        "    -2.932264997750e-10 1.000000000000e+00 2.044000000000e+03 0.000000000000e+00\n",
        "     4.000000000000e+00 6.300000000000e+01-8.847564458847e-09 8.660000000000e+02\n",
        "     3.553500000000e+05 4.000000000000e+00\n",
        "> EPH G04 CNAV\n",
        "G04 2019 03 14 03 30 00 1.330042141490e-04 7.226219622680e-12 0.000000000000e+00\n",
        "     2.001762390137e-03 6.914062500000e-01 4.625906973308e-09 1.887277537485e+00\n",
        "     1.024454832077e-08 3.348654136062e-04 8.376315236092e-06 5.153800325291e+03\n",
        "     2.412000000000e+05-4.656612873077e-09 5.171544951605e-01 2.328306436539e-08\n",
        "     9.601927657114e-01 2.174140625000e+02-1.737767543851e+00-8.034028170143e-09\n",
        "    -2.950122884460e-10-1.312310522376e-14-2.000000000000e+00 2.000000000000e+00\n",
        "     0.000000000000e+00 7.000000000000e+00-8.789356797934e-09 5.000000000000e+00\n",
        "    -5.820766091347e-10-6.606569513679e-09-1.178705133498e-08-1.178705133498e-08\n",
        "     3.558540000000e+05 2.044000000000e+03\n",
        "> EPH G04 CNV2\n",
        "G04 2019 03 14 03 30 00 1.330042141490e-04 7.226219622680e-12 0.000000000000e+00\n",
        "     2.001762390137e-03 6.914062500000e-01 4.625906973308e-09 1.887277537485e+00\n",
        "     1.024454832077e-08 3.348654136062e-04 8.376315236092e-06 5.153800325291e+03\n",
        "     2.412000000000e+05-4.656612873077e-09 5.171544951605e-01 2.328306436539e-08\n",
        "     9.601927657114e-01 2.174140625000e+02-1.737767543851e+00-8.034028170143e-09\n",
        "    -2.950122884460e-10-1.312310522376e-14-2.000000000000e+00 2.000000000000e+00\n",
        "     0.000000000000e+00 1.000000000000e+00-8.789356797934e-09 5.000000000000e+00\n",
        "    -5.820766091347e-10-6.606569513679e-09-1.178705133498e-08-1.178705133498e-08\n",
        "    -3.492459654808e-10-7.858034223318e-10\n",
        "     3.550860000000e+05 2.044000000000e+03\n",
        "> EPH I02 LNAV\n",
        "I02 2020 09 15 02 05 36 6.225099787116e-04 1.773514668457e-11 0.000000000000e+00\n",
        "     1.690000000000e+02-5.793750000000e+02 4.834487090078e-09-4.281979621524e-01\n",
        "    -1.904368400574e-05 2.015684265643e-03-3.430992364883e-06 6.493289550781e+03\n",
        "     1.803360000000e+05 2.495944499969e-07-1.337499015334e+00 7.450580596924e-08\n",
        "     5.022043764738e-01 1.946250000000e+02-2.970970345572e+00-4.461614415577e-09\n",
        "    -9.578970431139e-10                    2.123000000000e+03\n",
        "     2.000000000000e+00 0.000000000000e+00-1.862645149231e-09\n",
        "     1.804920000000e+05\n",
        "> EPH I10 L1NV\n",
        "I10 2023 06 24 00 05 00 1.527369022369e-07 1.364242052659e-12 0.000000000000e+00\n",
        "     0.000000000000e+00-2.593125000000e+02 7.028864208979e-09 2.300305834983e+00\n",
        "    -8.691102266312e-06 4.531537415460e-04-3.855675458908e-06 6.493495117188e+03\n",
        "     7.000000000000e+00 1.341104507446e-07 1.359342162629e-01-7.078051567078e-08\n",
        "     8.594830530333e-02 1.214375000000e+02-5.136403830694e-02-5.858815471753e-09\n",
        "    -4.846630453041e-10 0.000000000000e+00                    1.000000000000e+00\n",
        "     1.500000000000e+01 0.000000000000e+00                   -3.608874976635e-09\n",
        "                                           6.984919309616e-09 5.995389074087e-09\n",
        "     5.191380000000e+05\n",
        // Tables A18 and A41. No product broadcasts these five shapes -- the four BRD400DLR days
        // in `tests/rinex4_nav.rs` carry none of L1OC, L3OC, LXOC or either NavIC L1 ionosphere
        // subtype -- so the specification's own bytes are the only data they can be read against.
        "> EPH R26 L1OC\n",
        "R26 2024 02 03 00 15 00-1.605716170161e-05 1.652011860642e-12-2.081668171172e-17\n",
        "     1.812154053020e+04-2.071979139000e+00 5.729816621169e-10 1.000000000000e+00\n",
        "    -2.325615360260e+03 1.285475494340e+00-1.047737896442e-09 1.000000000000e+00\n",
        "     1.781341854668e+04 2.280306640081e+00-9.458744898438e-10 0.000000000000e+00\n",
        "     2.000000000000e+00 1.000000000000e+01 7.500000000000e-01 7.500000000000e-01\n",
        "     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n",
        "     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n",
        "     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n",
        "     1.500000000000e+01 5.000000000000e+00                    5.184000000000e+05\n",
        "> EPH R26 L3OC\n",
        "R26 2024 02 03 00 15 00-1.605716170161e-05 1.652011860642e-12-2.081668171172e-17\n",
        "     1.812154053020e+04-2.071979139000e+00 5.729816621169e-10 1.000000000000e+00\n",
        "    -2.325615360260e+03 1.285475494340e+00-1.047737896442e-09 1.000000000000e+00\n",
        "     1.781341854668e+04 2.280306640081e+00-9.458744898438e-10 0.000000000000e+00\n",
        "     2.000000000000e+00 1.000000000000e+01 7.500000000000e-01 7.500000000000e-01\n",
        "     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n",
        "     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n",
        "     0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00\n",
        "     1.500000000000e+01 5.000000000000e+00                    5.184000000000e+05\n",
        "> ION C03 D1D2\n",
        "    2021 07 05 00 09 00 7.450580596924e-09 4.470348358154e-08-4.172325134277e-07\n",
        "     5.960464477539e-07 1.187840000000e+05 1.802240000000e+05-6.553600000000e+05\n",
        "     5.242880000000e+05\n",
        "> ION I10 L1NV KLOB\n",
        "    2023 06 24 00 07 30 1.000000000000e+00\n",
        "     5.867332220078e-08 2.533197402954e-07-1.430511474609e-06-7.510185241699e-06\n",
        "     1.495040000000e+05-5.406720000000e+05 2.883584000000e+06 8.323072000000e+06\n",
        "     5.000000000000e+01 1.100000000000e+02 0.000000000000e+00 5.000000000000e+01\n",
        "> ION I10 L1NV NEQN\n",
        "    2023 06 24 00 17 24 0.000000000000e+00\n",
        "     1.982500000000e+02 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00\n",
        "     3.000000000000e+01 1.300000000000e+02-3.000000000000e+01-1.000000000000e+01\n",
        "     2.085000000000e+02-3.085937500000e-01-3.417968750000e-03 1.000000000000e+00\n",
        "     3.000000000000e+01 1.300000000000e+02-5.000000000000e+00 3.000000000000e+01\n",
        "     1.982500000000e+02 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00\n",
        "     3.000000000000e+01 1.300000000000e+02 3.500000000000e+01 5.000000000000e+01\n",
        "> ION R22 LXOC\n",
        "    2024 02 03 00 01 09 1.000000000000e+00 1.410000000000e+02 5.000000000000e+00\n",
    );

    fn parsed() -> Rinex4Nav {
        parse(SPEC_EXAMPLES).expect("the specification's own examples parse")
    }

    /// The header line's fields are read from columns, and a blank satellite field is not a PRN.
    #[test]
    fn a_record_header_line_is_read_from_its_columns() {
        let eph = parse_header("> EPH G01 LNAV").expect("a well-formed EPH header");
        assert_eq!(eph.kind, RecordKind::Eph);
        assert_eq!(eph.constellation, 'G');
        assert_eq!(eph.prn, Some(1));
        assert_eq!(eph.message, NavMessage::Lnav);
        assert_eq!(eph.subtype, None);

        // `> STO R   FDMA` is the case that makes column reading necessary rather than tidy:
        // splitting on whitespace puts `FDMA` in the PRN field and then finds no message type.
        let sto = parse_header("> STO R   FDMA").expect("a constellation-wide STO header");
        assert_eq!(sto.kind, RecordKind::Sto);
        assert_eq!(sto.constellation, 'R');
        assert_eq!(sto.prn, None);
        assert_eq!(sto.message, NavMessage::Fdma);

        let ion = parse_header("> ION I   L1NV KLOB").expect("a subtyped ION header");
        assert_eq!(ion.subtype, Some(Subtype::Klob));

        // Not a record line, an unknown record type, and an unknown message.
        assert_eq!(parse_header("G01 2024 01 01 00 00 00"), None);
        assert_eq!(parse_header("> XXX G01 LNAV"), None);
        assert_eq!(parse_header("> EPH G01 XXXX"), None);
    }

    /// GPS broadcasts three ephemeris messages at once and they are three different lengths.
    ///
    /// This is the whole reason the walk cannot key on the constellation: a reader that assumed
    /// eight lines for `G` would take the first two lines of the next record as CNAV's last two,
    /// and then be one record out for the rest of the file.
    #[test]
    fn the_block_length_follows_the_message_and_not_the_constellation() {
        use NavMessage as M;
        for (message, expected) in [
            (M::Lnav, 8),
            (M::Cnav, 9),
            (M::Cnv2, 10),
            (M::Fnav, 8),
            (M::Inav, 8),
            (M::D1, 8),
            (M::D2, 8),
            (M::Cnv1, 10),
            (M::Cnv3, 9),
            (M::L1nv, 9),
            (M::L1oc, 9),
            (M::L3oc, 9),
            (M::Fdma, 5),
            (M::Sbas, 4),
        ] {
            assert_eq!(
                block_len(RecordKind::Eph, message),
                Some(expected),
                "EPH {message:?}"
            );
        }
        // A constellation-wide code is not an ephemeris message and has no ephemeris length.
        assert_eq!(block_len(RecordKind::Eph, M::Cnvx), None);
        // STO and EOP are fixed; ION depends on the model, and on NavIC also on the subtype.
        assert_eq!(block_len(RecordKind::Sto, M::Cnvx), Some(2));
        assert_eq!(block_len(RecordKind::Eop, M::Cnvx), Some(3));
        assert_eq!(ion_block_len(M::Cnvx, None), Some(3));
        assert_eq!(ion_block_len(M::Ifnv, None), Some(2));
        assert_eq!(ion_block_len(M::Lxoc, None), Some(1));
        assert_eq!(ion_block_len(M::L1nv, Some(Subtype::Klob)), Some(4));
        assert_eq!(ion_block_len(M::L1nv, Some(Subtype::Neqn)), Some(7));
        assert_eq!(ion_block_len(M::L1nv, None), None);
    }

    /// The specification's own GPS example: LNAV into the legacy struct, CNAV and CNAV-2 into the
    /// quasi-Keplerian one, with nothing rejected.
    #[test]
    fn the_specifications_own_examples_decode_to_the_values_printed_in_them() {
        let nav = parsed();
        assert_eq!(nav.version, 4.02);
        assert_eq!(nav.leap_seconds, Some(18));
        assert!(nav.rejected.is_empty(), "{:?}", nav.rejected);

        // Two LNAV records: GPS G04 and NavIC I02.
        assert_eq!(nav.keplerian.len(), 2);
        let g04 = &nav.keplerian[0].ephemeris;
        assert_eq!(nav.keplerian[0].message, NavMessage::Lnav);
        assert_eq!(g04.system, GnssSystem::Gps);
        assert_eq!(g04.prn, 4);
        assert_eq!(g04.sqrt_a, 5.153_800_453_186e+03);
        assert_eq!(g04.e, 3.355_251_392_350e-04);
        assert_eq!(g04.toe, 3.600_000_000_000e+05);
        assert_eq!(g04.week, 2044);
        assert_eq!(g04.tgd, -8.847_564_458_847e-09);

        // NavIC's LNAV line 5 has a blank spare where GPS has "codes on L2", and the week still
        // reads out of the third column.
        let i02 = &nav.keplerian[1].ephemeris;
        assert_eq!(i02.system, GnssSystem::Navic);
        assert_eq!(i02.prn, 2);
        assert_eq!(i02.week, 2123);
        assert_eq!(i02.sqrt_a, 6.493_289_550_781e+03);

        // Three CNAV-family records: G04 CNAV, G04 CNV2, I10 L1NV.
        assert_eq!(nav.cnav.len(), 3);
        let cnav = &nav.cnav[0];
        assert_eq!(cnav.message, NavMessage::Cnav);
        assert_eq!(cnav.prn, 4);
        assert_eq!(cnav.sqrt_a0, 5.153_800_325_291e+03);
        assert_eq!(cnav.a_dot, 2.001_762_390_137e-03);
        assert_eq!(cnav.delta_n0, 4.625_906_973_308e-09);
        assert_eq!(cnav.delta_n0_dot, -1.312_310_522_376e-14);
        assert_eq!(cnav.t_op, Some(2.412_000_000_000e+05));
        assert_eq!(cnav.week, Some(2044));
        assert_eq!(cnav.health, 7);

        // CNAV-2 shares every orbit parameter with CNAV in this example and differs in health --
        // which is the check that the extra `ISC_L1Cd/L1Cp` line was stepped over and not read as
        // one of them.
        let cnv2 = &nav.cnav[1];
        assert_eq!(cnv2.message, NavMessage::Cnv2);
        assert_eq!(cnv2.sqrt_a0, cnav.sqrt_a0);
        assert_eq!(cnv2.delta_n0_dot, cnav.delta_n0_dot);
        assert_eq!(cnv2.health, 1);
        assert_eq!(cnv2.week, Some(2044));

        // NavIC L1NV puts `IODEC` where the others put `t_op`, so there is no prediction epoch to
        // report -- `Some(7.0)` would be a plausible-looking lie.
        let l1nv = &nav.cnav[2];
        assert_eq!(l1nv.message, NavMessage::L1nv);
        assert_eq!(l1nv.system, GnssSystem::Navic);
        assert_eq!(l1nv.t_op, None);
        assert_eq!(l1nv.week, None);
        assert_eq!(l1nv.sqrt_a0, 6.493_495_117_188e+03);
        assert_eq!(l1nv.a_dot, 0.0);
        assert_eq!(l1nv.delta_n0_dot, 0.0);
    }

    /// A header that is not version 4, not navigation data, or never ends.
    #[test]
    fn a_file_that_is_not_a_version_four_navigation_file_is_refused() {
        let v3 = SPEC_EXAMPLES.replace("     4.02  ", "     3.05  ");
        assert_eq!(
            parse(&v3),
            Err(Rinex4Error::UnsupportedVersion { major: 3 })
        );
        let obs = SPEC_EXAMPLES.replace("NAVIGATION DATA     M", "OBSERVATION DATA    M");
        assert_eq!(parse(&obs), Err(Rinex4Error::NotNavigationData));
        let headless = SPEC_EXAMPLES.replace("END OF HEADER", "COMMENT      ");
        assert_eq!(parse(&headless), Err(Rinex4Error::NoEndOfHeader));
        assert_eq!(
            parse("not a RINEX file"),
            Err(Rinex4Error::NotNavigationData)
        );
    }

    /// A record whose columns do not hold numbers is counted, not silently dropped and not
    /// zero-filled -- and the walk stays in step over the records after it.
    #[test]
    fn an_unreadable_record_is_counted_and_the_walk_stays_in_step() {
        let broken = SPEC_EXAMPLES.replace("5.153800453186e+03", "5.15380045X186e+03");
        assert_ne!(broken, SPEC_EXAMPLES);
        let nav = parse(&broken).expect("the file still parses as a file");
        // The line number is the record's own `>` header, 1-based -- `rejected` is what a caller
        // greps the file with, so a count alone is not the answer.
        let header = SPEC_EXAMPLES
            .lines()
            .position(|l| l == "> EPH G04 LNAV")
            .expect("the record the replacement broke");
        assert_eq!(nav.rejected, vec![header + 1], "{:?}", nav.rejected);
        // The LNAV record is gone; every record after it is still there and still itself.
        assert_eq!(nav.keplerian.len(), 1);
        assert_eq!(nav.keplerian[0].ephemeris.system, GnssSystem::Navic);
        assert_eq!(nav.cnav.len(), 3);
    }

    /// GLONASS's CDMA messages carry a PZ-90 state, and the reader has to put it in metres.
    ///
    /// Both messages carry the same nine-line body, so the only difference the reader may report
    /// is the message name -- and the state itself is checked against the orbit it describes,
    /// which is what a unit conversion can actually be wrong about.
    #[test]
    fn the_glonass_cdma_messages_decode_to_a_pz90_state_in_metres() {
        let nav = parsed();
        assert_eq!(nav.glonass_cdma.len(), 2);
        let [l1oc, l3oc] = [&nav.glonass_cdma[0], &nav.glonass_cdma[1]];
        assert_eq!(l1oc.message, NavMessage::L1oc);
        assert_eq!(l3oc.message, NavMessage::L3oc);

        assert_eq!(l1oc.slot, 26);
        // Toc is 00:15:00 UTC; GLONASS's time of ephemeris is seconds into the day.
        assert_eq!(l1oc.toe, 900.0);
        assert_eq!(l1oc.tau, -1.605_716_170_161e-05);
        assert_eq!(l1oc.gamma, 1.652_011_860_642e-12);
        assert_eq!(l1oc.beta, -2.081_668_171_172e-17);

        // Every component, not only the vector norm: a norm is degenerate under a single-component
        // error, and 25 517 km is still 25 517 km with one axis divided by a thousand instead of
        // multiplied.
        assert_eq!(
            l1oc.position,
            [
                1.812_154_053_020e+04 * 1000.0,
                -2.325_615_360_260e+03 * 1000.0,
                1.781_341_854_668e+04 * 1000.0
            ]
        );
        assert_eq!(
            l1oc.velocity,
            [
                -2.071_979_139_000e+00 * 1000.0,
                1.285_475_494_340e+00 * 1000.0,
                2.280_306_640_081e+00 * 1000.0
            ]
        );
        assert_eq!(
            l1oc.luni_solar,
            [
                5.729_816_621_169e-10 * 1000.0,
                -1.047_737_896_442e-09 * 1000.0,
                -9.458_744_898_438e-10 * 1000.0
            ]
        );

        // The three columns are kilometres, km/s and km/s²; a state read in the file's own units,
        // or with the scale applied the wrong way, misses the orbit by orders of magnitude.
        let norm = |v: [f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let radius = norm(l1oc.position);
        assert!(
            (25_400e3..25_600e3).contains(&radius),
            "GLONASS radius {radius} m"
        );
        let speed = norm(l1oc.velocity);
        assert!((3_300.0..3_500.0).contains(&speed), "speed {speed} m/s");
        // Luni-solar acceleration is a fraction of a µm/s²; anything larger is a scale error.
        let accel = norm(l1oc.luni_solar);
        assert!((1e-7..1e-5).contains(&accel), "acceleration {accel} m/s^2");

        assert_eq!(l1oc.health, 1);
        assert_eq!(l1oc.data_invalid, 1);
        assert_eq!(l1oc.tgd_l2ocp, 0.0);
        // Table A16 orbit line 4: satellite type, source flags, then AODE and AODC in days.
        assert_eq!(l1oc.age_of_ephemeris, 0.75);
        assert_eq!(l1oc.age_of_clock, 0.75);

        assert_eq!(l3oc.position, l1oc.position);
        assert_eq!(l3oc.slot, l1oc.slot);
    }

    /// NavIC's two L1 ionosphere messages do not share the legacy Klobuchar line layout.
    ///
    /// Table A38 puts `IODK` alone on the epoch line and all eight coefficients on lines of their
    /// own; Table A39's NeQuick-N is regional and carries three coefficient triples with the box
    /// each applies in. Read with the layout every other constellation uses, the `KLOB` record
    /// returns `IODK` as `α₀` and shifts everything after it by one column -- which is what this
    /// test exists to refuse, since no product broadcasts either message.
    #[test]
    fn the_navic_l1_ionosphere_messages_use_their_own_line_layout() {
        let nav = parsed();
        let by_message = |m: NavMessage, sub: Option<Subtype>| {
            nav.ionosphere
                .iter()
                .find(|i| i.message == m && i.subtype == sub)
                .expect("the record is in the specification's examples")
        };

        let klob = by_message(NavMessage::L1nv, Some(Subtype::Klob));
        assert_eq!(klob.constellation, 'I');
        assert_eq!(klob.prn, Some(10));
        assert_eq!(klob.issue_of_data, Some(1.0));
        let IonosphereModel::Klobuchar {
            alpha,
            beta,
            region,
        } = klob.model
        else {
            panic!("NavIC L1 KLOB is a Klobuchar record: {:?}", klob.model);
        };
        assert_eq!(alpha[0], 5.867_332_220_078e-08);
        assert_eq!(alpha[3], -7.510_185_241_699e-06);
        assert_eq!(beta[0], 1.495_040_000_000e+05);
        assert_eq!(beta[3], 8.323_072_000_000e+06);
        // 50°E..110°E, 0°N..50°N -- NavIC's service area, which is the check that line 3 was read
        // as a region and not as more coefficients.
        assert_eq!(
            region,
            Some(Region {
                longitude: [50.0, 110.0],
                latitude: [0.0, 50.0],
            })
        );

        let neqn = by_message(NavMessage::L1nv, Some(Subtype::Neqn));
        assert_eq!(neqn.issue_of_data, Some(0.0));
        let IonosphereModel::NequickN { regions } = neqn.model else {
            panic!("NavIC L1 NEQN is a NeQuick-N record: {:?}", neqn.model);
        };
        assert_eq!(regions[0].ai, [198.25, 0.0, 0.0]);
        assert_eq!(regions[0].disturbance_flag, 1);
        assert_eq!(regions[0].bounds.longitude, [30.0, 130.0]);
        assert_eq!(regions[0].bounds.latitude, [-30.0, -10.0]);
        assert_eq!(regions[1].ai[1], -3.085_937_500_000e-01);
        assert_eq!(regions[1].bounds.latitude, [-5.0, 30.0]);
        assert_eq!(regions[2].bounds.latitude, [35.0, 50.0]);
        // The three regions tile MODIP without overlapping, in the order transmitted.
        for r in regions.windows(2) {
            assert!(r[0].bounds.latitude[1] <= r[1].bounds.latitude[0]);
        }

        // GLONASS's CDMA model is one line and three coefficients, and it is not a Klobuchar.
        let lxoc = by_message(NavMessage::Lxoc, None);
        assert_eq!(lxoc.constellation, 'R');
        assert_eq!(lxoc.issue_of_data, None);
        assert_eq!(
            lxoc.model,
            IonosphereModel::GlonassCdma {
                c: [1.0, 141.0, 5.0]
            }
        );

        // A legacy Klobuchar carries no region, so a caller cannot mistake a global model for a
        // regional one.
        let legacy = by_message(NavMessage::D1d2, None);
        assert!(matches!(
            legacy.model,
            IonosphereModel::Klobuchar { region: None, .. }
        ));
    }

    /// A record is stamped with seconds of week, computed from the calendar date it prints.
    ///
    /// Only `Toc` is transmitted on a CNAV-family record -- IS-GPS-200 §30.3.4.4 requires
    /// `Toe = Toc`, so there is no second column to check the first against. The file's own
    /// cross-check is the LNAV record, which prints both: its `Toe` column has to come out equal
    /// to the `Toc` this reader computes from the same epoch line.
    #[test]
    fn the_seconds_of_week_a_record_is_stamped_with_is_computed_from_its_calendar_date() {
        let nav = parsed();
        // 2019-03-14 is a Thursday. LNAV G04 is stamped 04:00:00, so 4·86400 + 4·3600 = 360 000 s
        // -- and 3.600000000000e+05 is what the record's `Toe` column says.
        let g04 = &nav.keplerian[0].ephemeris;
        assert_eq!(g04.toc, 360_000.0);
        assert_eq!(g04.toc, g04.toe);

        // CNAV G04 is the same Thursday at 03:30:00: 4·86400 + 3·3600 + 30·60 = 358 200 s.
        assert_eq!(nav.cnav[0].toc, 358_200.0);
        assert_eq!(nav.cnav[1].toc, 358_200.0);

        // NavIC L1NV I10, 2023-06-24, a Saturday, at 00:05:00: 6·86400 + 5·60 = 518 700 s.
        let l1nv = nav
            .cnav
            .iter()
            .find(|e| e.message == NavMessage::L1nv)
            .expect("the NavIC L1NV example");
        assert_eq!(l1nv.toc, 518_700.0);

        // GLONASS is stamped in seconds of day, not of week -- 00:15:00 is 900 s either way only
        // because the example falls on a Saturday, so the day-of-week term is checked above and
        // not here.
        assert_eq!(nav.glonass_cdma[0].toe, 900.0);
    }

    /// A CNAV record that does not transmit a prediction week says so.
    ///
    /// Table A13 note 6* has a message zero the fields it does not carry, so a `wn_op` of zero is
    /// "absent", not "week zero" -- GPS week 0 was August 1980 and no live record can mean it.
    #[test]
    fn a_cnav_record_whose_prediction_week_is_absent_reports_none() {
        let present = parsed();
        assert_eq!(present.cnav[0].week, Some(2044));

        let zeroed = SPEC_EXAMPLES.replace(
            "     3.558540000000e+05 2.044000000000e+03\n",
            "     3.558540000000e+05 0.000000000000e+00\n",
        );
        assert_ne!(zeroed, SPEC_EXAMPLES);
        let nav = parse(&zeroed).expect("the file still parses");
        assert!(nav.rejected.is_empty(), "{:?}", nav.rejected);
        assert_eq!(nav.cnav[0].message, NavMessage::Cnav);
        assert_eq!(nav.cnav[0].week, None);
        // Only that record changed; CNAV-2's own week line is untouched.
        assert_eq!(nav.cnav[1].week, Some(2044));
    }

    /// Lines the walk cannot frame do not shift the records after them.
    ///
    /// Two ways to be unframeable: a line that is not a record header at all, and a header whose
    /// message type this reader does not know a block length for. Neither may consume the record
    /// that follows, and neither is a rejected record -- nothing was promised and nothing failed.
    #[test]
    fn a_line_the_walk_cannot_frame_leaves_the_records_after_it_alone() {
        let baseline = parsed();
        let padded = SPEC_EXAMPLES.replace(
            "> EPH G04 CNAV\n",
            "a line that is not a record header\n> EPH G04 ZZZZ\n> EPH G04 CNAV\n",
        );
        assert_ne!(padded, SPEC_EXAMPLES);
        let nav = parse(&padded).expect("the file still parses");
        assert!(nav.rejected.is_empty(), "{:?}", nav.rejected);
        assert_eq!(nav.keplerian.len(), baseline.keplerian.len());
        assert_eq!(nav.cnav.len(), baseline.cnav.len());
        assert_eq!(nav.ionosphere.len(), baseline.ionosphere.len());
        assert_eq!(nav.glonass_cdma.len(), baseline.glonass_cdma.len());
        assert_eq!(nav.cnav[0].sqrt_a0, baseline.cnav[0].sqrt_a0);
    }

    /// The analytic velocity is the derivative of the analytic position, checked numerically.
    ///
    /// Table 30-II gives the velocity as a chain of closed forms rather than as a derivative, so
    /// every term in it is a transcription that nothing in the position can contradict. A
    /// five-point central difference of the position is an independent statement of the same
    /// quantity: its truncation is `O(h⁴·p⁽⁵⁾/30)` ≈ 4e-11 m/s at `h = 5 s` on a GPS orbit and its
    /// cancellation `≈ ulp(|p|)/h` ≈ 7e-10 m/s, so agreement at 1e-6 m/s leaves four orders of
    /// margin over both.
    ///
    /// `Δn̈₀` is zeroed and `Ȧ` is not, deliberately. With `Δn̈₀ = 0` the mean motion is constant,
    /// `Ėₖ = n/(1 − e·cos Eₖ)` is exact rather than the specification's approximation, and the
    /// comparison is between two exact statements. Leaving `Δn̈₀` in would floor the agreement at
    /// the ~2 mm/s the specification drops — which is documented on
    /// [`CnavEphemeris::position_velocity_ecef`] and is not what this test is about. `Ȧ` stays,
    /// because `Ȧ(1 − e·cos Eₖ)` in the radius rate is the one term the rates-zeroed identity
    /// against the legacy propagator cannot see at all.
    #[test]
    fn the_analytic_velocity_is_the_derivative_of_the_analytic_position() {
        let nav = parsed();
        let mut eph = nav.cnav[0];
        assert_ne!(eph.a_dot, 0.0);
        eph.delta_n0_dot = 0.0;

        let h = 5.0;
        for tk in [-7200.0, -900.0, 0.0, 900.0, 7200.0] {
            let (_, analytic) = eph.position_velocity_ecef(tk);
            let p = |d: f64| eph.position_ecef(tk + d);
            let (m2, m1, p1, p2) = (p(-2.0 * h), p(-h), p(h), p(2.0 * h));
            for k in 0..3 {
                let numeric = (m2[k] - 8.0 * m1[k] + 8.0 * p1[k] - p2[k]) / (12.0 * h);
                assert!(
                    (analytic[k] - numeric).abs() < 1e-6,
                    "tk = {tk}, axis {k}: analytic {} against numeric {numeric}",
                    analytic[k]
                );
            }
        }
    }

    /// Every field of the epoch stamp reaches the seconds of week, including the seconds.
    ///
    /// The specification's example records are all stamped on a whole minute, and three of the four
    /// terms of `dow·86400 + h·3600 + m·60 + s` are zero in at least one of them -- so an example
    /// alone cannot tell `+ s` from `- s`, or `h · 3600` from `h / 3600`. The epochs here are the
    /// examples' with the stamp moved, which is the smallest edit that makes each term observable.
    #[test]
    fn every_field_of_the_epoch_stamp_reaches_the_seconds_of_week() {
        // GPS CNAV, 2019-03-14 (a Thursday) 03:30:00 -> 05:30:42.
        let moved = SPEC_EXAMPLES.replace(
            "G04 2019 03 14 03 30 00 1.330042141490e-04",
            "G04 2019 03 14 05 30 42 1.330042141490e-04",
        );
        assert_ne!(moved, SPEC_EXAMPLES);
        let nav = parse(&moved).expect("the file still parses");
        // 4·86400 + 5·3600 + 30·60 + 42 = 365 442 s.
        assert_eq!(nav.cnav[0].toc, 365_442.0);

        // GLONASS L1OC, 00:15:00 -> 07:15:23; seconds of day, so no day-of-week term.
        let moved = SPEC_EXAMPLES.replace(
            "R26 2024 02 03 00 15 00-1.605716170161e-05",
            "R26 2024 02 03 07 15 23-1.605716170161e-05",
        );
        assert_ne!(moved, SPEC_EXAMPLES);
        let nav = parse(&moved).expect("the file still parses");
        // 7·3600 + 15·60 + 23 = 26 123 s. Both L1OC and L3OC carry the same epoch line.
        assert_eq!(nav.glonass_cdma[0].toe, 26_123.0);
        assert_eq!(nav.glonass_cdma[1].toe, 26_123.0);
    }

    /// The walk starts after `END OF HEADER` and not before it.
    ///
    /// Nothing stops a `COMMENT` from beginning with `>`: the header's structure is the label in
    /// columns 61-80, and columns 1-60 are free text. A reader that started one line early, or at
    /// the label line itself, would decode a comment as a record.
    #[test]
    fn a_comment_that_looks_like_a_record_is_still_a_comment() {
        let baseline = parsed();
        // The bait is in two places, because the walk can start one line early *or* on the label
        // line itself: the last `COMMENT` before `END OF HEADER`, and columns 1-60 of the
        // `END OF HEADER` record, which the format leaves free.
        let smuggled = SPEC_EXAMPLES.replace(
            "                                                            END OF HEADER\n",
            concat!(
                "> EPH G04 LNAV                                              COMMENT\n",
                "> EPH G04 LNAV                                              END OF HEADER\n",
            ),
        );
        assert_ne!(smuggled, SPEC_EXAMPLES);
        let nav = parse(&smuggled).expect("the file still parses");
        assert!(nav.rejected.is_empty(), "{:?}", nav.rejected);
        assert_eq!(nav.keplerian.len(), baseline.keplerian.len());
        assert_eq!(
            nav.keplerian[0].ephemeris.af0,
            baseline.keplerian[0].ephemeris.af0
        );
    }

    /// A record whose block runs off the end of the file is reported by the line it starts on.
    ///
    /// Truncation is the one case where the reader cannot tell a short record from a corrupt one,
    /// so it says which line it gave up at rather than decoding a partial block.
    #[test]
    fn a_truncated_final_record_is_reported_by_the_line_it_starts_on() {
        let full: Vec<&str> = SPEC_EXAMPLES.lines().collect();
        let start = full
            .iter()
            .position(|l| *l == "> ION R22 LXOC")
            .expect("the LXOC record is last");
        // Keep the header line, drop its one body line.
        let truncated = full[..=start].join("\n");
        let nav = parse(&truncated).expect("the file still parses");
        // `rejected` holds 1-based line numbers of the record header lines, which is what a
        // caller greps the file with.
        assert_eq!(nav.rejected, vec![start + 1]);
        assert!(nav.ionosphere.iter().all(|i| i.message != NavMessage::Lxoc));
    }

    /// With no axis rate and no mean-motion rate, the CNAV algorithm *is* the legacy one.
    ///
    /// The two differ in exactly two terms — `A(tₖ) = A₀ + Ȧ·tₖ` and the `Ȧ(1 − e·cos Eₖ)` in the
    /// radius rate — so setting both rates to zero has to reproduce
    /// [`crate::gps::BroadcastEphemeris::position_velocity_ecef`] bit for bit. That makes the
    /// legacy propagator, which is verified against Orekit and against IGS precise orbits, the
    /// oracle for the new one: any structural error in the transcription of Table 30-II shows up
    /// here as a difference, and equality is exact rather than tolerance-bounded.
    #[test]
    fn with_the_rates_zeroed_the_cnav_algorithm_reproduces_the_legacy_one_exactly() {
        let nav = parsed();
        let cnav = &nav.cnav[0];
        let mut flat = *cnav;
        flat.a_dot = 0.0;
        flat.delta_n0_dot = 0.0;

        let legacy = crate::gps::BroadcastEphemeris {
            system: cnav.system,
            prn: cnav.prn,
            toe: cnav.toc,
            toc: cnav.toc,
            sqrt_a: cnav.sqrt_a0,
            e: cnav.e,
            i0: cnav.i0,
            omega0: cnav.omega0,
            omega: cnav.omega,
            m0: cnav.m0,
            delta_n: cnav.delta_n0,
            idot: cnav.idot,
            omega_dot: cnav.omega_dot,
            cuc: cnav.cuc,
            cus: cnav.cus,
            crc: cnav.crc,
            crs: cnav.crs,
            cic: cnav.cic,
            cis: cnav.cis,
            ..crate::gps::BroadcastEphemeris::default()
        };

        for tk in [-7200.0, -1800.0, 0.0, 1800.0, 7200.0] {
            assert_eq!(
                flat.position_velocity_ecef(tk),
                legacy.position_velocity_ecef(tk),
                "tk = {tk}"
            );
        }
    }

    /// The two rate terms are the whole difference, and each moves the orbit the way it should.
    ///
    /// `Ȧ` is a metres-per-second drift of the axis, so at two hours it is `Ȧ·tₖ` of radius and
    /// nothing else; `Δn̈₀` accelerates the mean anomaly, so its effect grows as `tₖ²` and is
    /// along-track. Asserted as signed, closed-form quantities rather than as "different", because
    /// a sign error in either term is exactly what this test exists to catch.
    #[test]
    fn each_rate_term_moves_the_orbit_by_the_amount_its_definition_says() {
        let nav = parsed();
        let cnav = &nav.cnav[0];
        assert_ne!(cnav.a_dot, 0.0, "the specification's example carries A-dot");

        // The axis rate, read straight back out.
        for tk in [3600.0, -3600.0] {
            let a0 = cnav.sqrt_a0 * cnav.sqrt_a0;
            let drift = cnav.semi_major_axis(tk) - a0;
            // `a0` is 2.7e7 m and the drift is metres, so the difference cancels eleven digits --
            // relative to the drift, not to the axis.
            assert!(
                (drift - cnav.a_dot * tk).abs() < 1e-12 * (cnav.a_dot * tk).abs(),
                "{drift} m against A-dot*tk {} m",
                cnav.a_dot * tk
            );
        }
        assert_eq!(cnav.semi_major_axis(0.0), cnav.sqrt_a0 * cnav.sqrt_a0);

        // The mean-motion rate enters as half of itself, per Table 30-II sheet 1.
        let n_at = |tk| cnav.mean_motion(tk);
        let measured = n_at(7200.0) - n_at(0.0);
        let expected = 0.5 * cnav.delta_n0_dot * 7200.0;
        // `n` is 1.46e-4 and the difference is 4.7e-11, so seven digits cancel; the tolerance is
        // relative to what is left, not to `n`.
        assert!(
            (measured - expected).abs() < 1e-8 * expected.abs(),
            "{measured} vs {expected}"
        );

        // With only `Ȧ` zeroed the radius at `tₖ` must fall by `Ȧ·tₖ·(1 − e·cos Eₖ)`, which is
        // within a part in 10³ of `Ȧ·tₖ` for a near-circular orbit.
        let mut no_a_dot = *cnav;
        no_a_dot.a_dot = 0.0;
        let tk = 7200.0;
        let norm = |p: [f64; 3]| p.iter().map(|x| x * x).sum::<f64>().sqrt();
        let drift = norm(cnav.position_ecef(tk)) - norm(no_a_dot.position_ecef(tk));
        let expected = cnav.a_dot * tk;
        assert!(
            (drift - expected).abs() < 1e-3 * expected.abs(),
            "radius drift {drift} m against A-dot*tk {expected} m"
        );
    }

    /// Every `Rinex4Error` prints something that names the problem.
    #[test]
    fn every_error_prints_a_distinct_message() {
        let messages: Vec<String> = [
            Rinex4Error::NotNavigationData,
            Rinex4Error::UnsupportedVersion { major: 3 },
            Rinex4Error::NoEndOfHeader,
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        assert_eq!(messages.len(), 3);
        for (i, a) in messages.iter().enumerate() {
            assert_ne!(a.len(), 0);
            for b in &messages[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
