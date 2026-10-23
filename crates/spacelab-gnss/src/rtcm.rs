//! RTCM 3 transport and the messages a real-time GNSS client actually needs: station coordinates,
//! broadcast ephemerides, and the SSR orbit and clock corrections that turn a broadcast orbit into a
//! precise one.
//!
//! Everything else in this crate reads a *file* — RINEX, SP3, an almanac — which is the same
//! information after somebody has already framed, checked and written it down. A real-time client
//! gets a byte stream instead, and the framing is where the problems are: a stream resynchronises
//! after packet loss, carries messages the reader does not implement, and contains bytes that look
//! like a preamble but are not. So the decoder here is a **scanner**, not a parser: it is handed
//! bytes, it finds frames, and it refuses everything whose CRC does not check.
//!
//! Written from the message and data-field definitions of RTCM Standard 10403 — the `DF` numbers
//! below are its — and from the IGS SSR format description, not from any decoder's source. The
//! standard is a paid document whose text belongs to RTCM; the field layouts are facts.
//!
//! # The frame
//!
//! ```text
//! 0xD3 | 6 reserved bits | 10-bit payload length | payload | 24-bit CRC
//! ```
//!
//! The CRC is **CRC-24Q**, the polynomial `0x1864CFB` that GPS itself uses for navigation-message
//! parity, computed over the header and the payload together with a **zero initial value** and no
//! final inversion. The zero init is the detail to get right: the same polynomial with `OpenPGP`'s
//! `0xB704CE` init is a different check, and it is the one whose `0x21CF02` check value is easier to
//! find. RTCM's variant is the reveng catalogue's `CRC-24/LTE-A`, whose published check value over
//! `123456789` is `0xCDE703`, and that is what the tests assert.
//!
//! It is a genuine check rather than a formality: the byte `0xD3` occurs in payload data
//! about once every 256 bytes, so a scanner that trusted the preamble alone would frame garbage
//! constantly, and the length field it then read would send it off by up to a kilobyte.
//! [`RtcmDecoder`] therefore validates before it yields, and on a failure advances by **one byte**
//! rather than by the length a corrupt header claimed.
//!
//! # What is decoded, and what is deliberately not
//!
//! * **1005 / 1006** — the stationary reference station's ECEF antenna reference point, which is the
//!   coordinate every differential correction is relative to. 1006 adds the antenna height.
//! * **1019** — the GPS broadcast ephemeris, decoded straight into the [`BroadcastEphemeris`] the
//!   rest of this crate propagates. The scale factors are IS-GPS-200's, which is why a RINEX
//!   navigation record and a 1019 frame describing the same satellite must yield the same orbit.
//! * **1057 / 1058 / 1060** — SSR orbit, clock, and combined corrections for GPS, with
//!   [`OrbitCorrection::at`] and [`ClockCorrection::at`] evaluating the polynomials in time and
//!   [`apply_orbit`] rotating the radial/along-track/cross-track triple into ECEF.
//! * **MSM observation messages (1074–1077 and the other constellations') are not decoded.** They
//!   are the largest part of a real stream and the least of its mathematics: an MSM7 block is a
//!   packing exercise over pseudorange, phase, Doppler, CNR and lock time, and the observables it
//!   yields are the ones [`crate::rinex_obs`] already models. Adding them is a day of bit tables,
//!   not a day of thinking, and it is recorded here as absent rather than hidden.
//! * **RTCM 2.x is not supported and will not be.** Different transport, different check,
//!   superseded since 2004.
//!
//! # The sign convention, which is the thing to get wrong
//!
//! RTCM SSR orbit corrections are **subtracted** from the broadcast orbit:
//! `X_precise = X_broadcast − δX`. The clock correction is **added** to the broadcast clock
//! *offset*: `dt_precise = dt_broadcast + δC/c`. The two go opposite ways, which is not a typo in
//! the standard and is the commonest integration error against an SSR stream. Both directions are
//! asserted below against a hand-computed case.
//!
//! # Time reference
//!
//! A correction's polynomial is evaluated about `t₀ = epoch + Δ/2`, where `Δ` is the update interval
//! named by DF391: the message's epoch stamps the *beginning* of its validity interval and the
//! polynomial is referenced to the middle of it. Using the epoch itself is wrong by half an
//! interval, which on the 60 s interval a typical stream uses is a 30 s lever arm on the rate terms.

// Bit-field extraction indexes by one letter and narrows through widths the reader has already
// bounds-checked, so the conversions below cannot lose information the field carried.
#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use crate::gps::{BroadcastEphemeris, GnssSystem};

/// Semicircles to radians: the broadcast angular quantities are all in half-turns.
const SEMICIRCLE: f64 = core::f64::consts::PI;

/// Speed of light in vacuum, m/s (IS-GPS-200, and the SI definition).
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// The RTCM 3 frame preamble.
pub const PREAMBLE: u8 = 0xD3;

/// SSR update intervals named by DF391, seconds.
pub const SSR_UPDATE_INTERVALS: [f64; 16] = [
    1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 240.0, 300.0, 600.0, 900.0, 1800.0, 3600.0,
    7200.0, 10800.0,
];

/// The largest satellite count DF387, which is six bits, can name.
pub const MAX_SSR_SATELLITES: usize = 63;

/// CRC-24Q (`x²⁴+x²³+x¹⁸+x¹⁷+x¹⁴+x¹¹+x¹⁰+x⁷+x⁶+x⁵+x⁴+x³+x+1`, generator `0x0186_4CFB`).
///
/// Zero initial value, most-significant bit first, no final inversion — the parameters GPS and RTCM
/// use, and therefore the reveng catalogue's CRC-24/LTE-A rather than its CRC-24/OPENPGP, which is
/// the same polynomial seeded with `0xB704CE`.
#[must_use]
pub fn crc24q(bytes: &[u8]) -> u32 {
    const POLYNOMIAL: u32 = 0x0186_4CFB;
    let mut crc = 0_u32;
    for byte in bytes {
        crc ^= u32::from(*byte) << 16;
        for _ in 0..8 {
            crc <<= 1;
            if crc & 0x0100_0000 != 0 {
                crc ^= POLYNOMIAL;
            }
        }
    }
    crc & 0x00FF_FFFF
}

/// A most-significant-bit-first reader over a payload.
///
/// Every read is bounds-checked and returns `None` past the end, so a message shorter than its own
/// definition — which happens when a producer truncates — ends the decode instead of reading
/// whatever follows it in the buffer.
#[derive(Debug, Clone)]
struct Bits<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Bits<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    const fn consumed(&self) -> usize {
        self.offset
    }

    /// How many bits the payload holds in total.
    const fn len(&self) -> usize {
        self.bytes.len() * 8
    }

    /// `width` bits as an unsigned integer, `width ≤ 64`.
    fn u64(&mut self, width: usize) -> Option<u64> {
        if width > 64 || self.offset + width > self.bytes.len() * 8 {
            return None;
        }
        let mut value = 0_u64;
        for i in 0..width {
            let bit = self.offset + i;
            let byte = self.bytes[bit / 8];
            value = (value << 1) | u64::from((byte >> (7 - bit % 8)) & 1);
        }
        self.offset += width;
        Some(value)
    }

    fn u32(&mut self, width: usize) -> Option<u32> {
        self.u64(width).map(|v| v as u32)
    }

    fn u8(&mut self, width: usize) -> Option<u8> {
        self.u64(width).map(|v| v as u8)
    }

    /// `width` bits as a two's-complement signed integer.
    fn i64(&mut self, width: usize) -> Option<i64> {
        let raw = self.u64(width)?;
        if width == 64 {
            return Some(raw.cast_signed());
        }
        let sign = 1_u64 << (width - 1);
        Some(if raw & sign == 0 {
            raw.cast_signed()
        } else {
            raw.cast_signed() - (1_i64 << width)
        })
    }

    /// A signed field scaled by a power of two, as the standard's tables specify them.
    ///
    /// The scale goes through `ldexp` rather than a multiply by a computed `2^p`, so the result is
    /// the integer's exact value with its exponent shifted: no rounding is introduced by the scale
    /// itself, only whatever the producer's own quantisation already cost.
    #[allow(clippy::cast_precision_loss)] // every width here is ≤ 32, so the integer is exact in f64
    fn scaled(&mut self, width: usize, power: i32) -> Option<f64> {
        self.i64(width).map(|v| libm::ldexp(v as f64, power))
    }

    /// An unsigned field scaled by a power of two.
    #[allow(clippy::cast_precision_loss)]
    fn scaled_unsigned(&mut self, width: usize, power: i32) -> Option<f64> {
        self.u64(width).map(|v| libm::ldexp(v as f64, power))
    }
}

/// A stationary reference station's antenna reference point (message 1005 or 1006).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StationReferencePoint {
    /// Reference station ID (DF003).
    pub station_id: u16,
    /// ITRF realisation year (DF021); `0` when the producer does not say.
    pub itrf_year: u8,
    /// ECEF antenna reference point, metres.
    pub ecef: [f64; 3],
    /// Antenna height above the marker, metres — `None` for 1005, which does not carry one.
    pub antenna_height: Option<f64>,
    /// Whether the producer claims GPS, GLONASS and Galileo service (DF022/DF023/DF024).
    pub services: [bool; 3],
}

/// An SSR orbit correction for one satellite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCorrection {
    /// Satellite PRN.
    pub prn: u8,
    /// The broadcast ephemeris issue this correction belongs to. Applying it to a different IODE is
    /// applying it to a different orbit.
    pub iode: u8,
    /// Reference epoch of the polynomial, s of GPS week — the message epoch plus half its update
    /// interval.
    pub reference: f64,
    /// Radial, along-track and cross-track offsets at the reference epoch, metres.
    pub delta: [f64; 3],
    /// Their rates, m/s.
    pub delta_rate: [f64; 3],
}

impl OrbitCorrection {
    /// The correction at `tow` (s of GPS week), as `δ₀ + δ̇·(t − t₀)`.
    #[must_use]
    pub fn at(&self, tow: f64) -> [f64; 3] {
        let dt = tow - self.reference;
        core::array::from_fn(|i| self.delta_rate[i].mul_add(dt, self.delta[i]))
    }
}

/// An SSR clock correction for one satellite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockCorrection {
    /// Satellite PRN.
    pub prn: u8,
    /// Reference epoch of the polynomial, s of GPS week.
    pub reference: f64,
    /// Polynomial coefficients `[C0, C1, C2]`, in metres, m/s and m/s².
    pub coefficients: [f64; 3],
}

impl ClockCorrection {
    /// The correction at `tow` (s of GPS week), in **metres** of range.
    #[must_use]
    pub fn at(&self, tow: f64) -> f64 {
        let dt = tow - self.reference;
        self.coefficients[2]
            .mul_add(dt, self.coefficients[1])
            .mul_add(dt, self.coefficients[0])
    }

    /// The correction at `tow` as a clock offset in **seconds**, ready to be added to the broadcast
    /// offset from [`BroadcastEphemeris::clock_correction`].
    #[must_use]
    pub fn seconds_at(&self, tow: f64) -> f64 {
        self.at(tow) / SPEED_OF_LIGHT
    }
}

/// Correct a broadcast satellite position with an SSR orbit correction.
///
/// The radial/along-track/cross-track basis is built from the satellite's own broadcast state —
/// radial along `r̂`, cross-track along `r̂ × v̂`, along-track completing the right-handed set — and
/// the correction is **subtracted**, which is RTCM's convention and the opposite of what the word
/// "correction" suggests.
///
/// Returns `None` for a degenerate state — zero radius, or velocity parallel to position — where the
/// basis does not exist and putting the along-track correction on an arbitrary axis would be worse
/// than declining.
#[must_use]
pub fn apply_orbit(
    position: [f64; 3],
    velocity: [f64; 3],
    correction: [f64; 3],
) -> Option<[f64; 3]> {
    let radial = normalise(position)?;
    let cross = normalise(cross_product(position, velocity))?;
    let along = cross_product(cross, radial);
    Some(core::array::from_fn(|i| {
        position[i]
            - (radial[i] * correction[0] + along[i] * correction[1] + cross[i] * correction[2])
    }))
}

fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalise(v: [f64; 3]) -> Option<[f64; 3]> {
    let norm = libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if norm <= 0.0 || !norm.is_finite() {
        return None;
    }
    Some([v[0] / norm, v[1] / norm, v[2] / norm])
}

/// The satellites in one SSR orbit message.
#[derive(Debug, Clone, PartialEq)]
pub struct SsrOrbitBlock {
    satellites: [OrbitCorrection; MAX_SSR_SATELLITES],
    count: usize,
    /// SSR issue of data (DF413): corrections from different issues must not be mixed.
    pub iod_ssr: u8,
}

/// The satellites in one SSR clock message.
#[derive(Debug, Clone, PartialEq)]
pub struct SsrClockBlock {
    satellites: [ClockCorrection; MAX_SSR_SATELLITES],
    count: usize,
    /// SSR issue of data (DF413).
    pub iod_ssr: u8,
}

impl SsrOrbitBlock {
    /// The corrections the message carried, in the order it listed them.
    #[must_use]
    pub fn satellites(&self) -> &[OrbitCorrection] {
        &self.satellites[..self.count]
    }
}

impl SsrClockBlock {
    /// The corrections the message carried, in the order it listed them.
    #[must_use]
    pub fn satellites(&self) -> &[ClockCorrection] {
        &self.satellites[..self.count]
    }
}

/// A decoded RTCM message.
///
/// [`Message::Unsupported`] is not an error: a real stream is mostly messages any given client does
/// not implement, and a client that treated those as failures would report a broken feed.
///
/// The SSR variants are large — each block is a fixed array sized by DF387's own six-bit satellite
/// count — and boxing them is not available here: this module is `no_std` and the crate does not
/// take an `alloc` dependency to carry one enum. A caller that moves many of these should match by
/// reference.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// 1005 or 1006.
    StationReferencePoint(StationReferencePoint),
    /// 1019, a GPS broadcast ephemeris.
    GpsEphemeris(BroadcastEphemeris),
    /// 1057.
    SsrOrbit(SsrOrbitBlock),
    /// 1058.
    SsrClock(SsrClockBlock),
    /// 1060, which carries both for the same satellites and the same issue of data.
    SsrOrbitAndClock(SsrOrbitBlock, SsrClockBlock),
    /// A well-framed message this decoder does not model, carrying its number.
    Unsupported(u16),
}

/// Why a well-framed payload could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    /// The payload ended before the message's own definition did.
    Truncated {
        /// The message number, if the first twelve bits were readable.
        message: u16,
    },
    /// The payload holds more than the message plus its padding to a byte boundary.
    WrongLength {
        /// The message number.
        message: u16,
        /// Bits the payload actually holds.
        payload_bits: usize,
        /// Bits the message occupies.
        expected: usize,
    },
    /// DF387 named more satellites than the message can hold.
    TooManySatellites {
        /// The message number.
        message: u16,
        /// The count that was named.
        count: usize,
    },
}

/// Decode one verified payload — the bytes between the length field and the CRC.
///
/// # Errors
///
/// [`DecodeError`] when the payload does not match the definition its own message number claims.
pub fn decode(payload: &[u8]) -> Result<Message, DecodeError> {
    let mut bits = Bits::new(payload);
    let Some(number) = bits.u32(12) else {
        return Err(DecodeError::Truncated { message: 0 });
    };
    let number = number as u16;
    match number {
        1005 | 1006 => decode_station(&mut bits, number).map(Message::StationReferencePoint),
        1019 => decode_gps_ephemeris(&mut bits).map(Message::GpsEphemeris),
        1057 => decode_ssr_orbit(&mut bits, number).map(Message::SsrOrbit),
        1058 => decode_ssr_clock(&mut bits, number).map(Message::SsrClock),
        1060 => decode_ssr_combined(&mut bits, number),
        _ => Ok(Message::Unsupported(number)),
    }
}

fn decode_station(bits: &mut Bits, number: u16) -> Result<StationReferencePoint, DecodeError> {
    let truncated = || DecodeError::Truncated { message: number };
    let station_id = bits.u32(12).ok_or_else(truncated)? as u16;
    let itrf_year = bits.u8(6).ok_or_else(truncated)?;
    let gps = bits.u8(1).ok_or_else(truncated)? == 1;
    let glonass = bits.u8(1).ok_or_else(truncated)? == 1;
    let galileo = bits.u8(1).ok_or_else(truncated)? == 1;
    let _reference_station = bits.u8(1).ok_or_else(truncated)?;
    let x = bits.i64(38).ok_or_else(truncated)?;
    let _oscillator = bits.u8(1).ok_or_else(truncated)?;
    let _reserved = bits.u8(1).ok_or_else(truncated)?;
    let y = bits.i64(38).ok_or_else(truncated)?;
    let _quarter_cycle = bits.u8(2).ok_or_else(truncated)?;
    let z = bits.i64(38).ok_or_else(truncated)?;
    let antenna_height = if number == 1006 {
        Some(f64::from(bits.u32(16).ok_or_else(truncated)?) * 1e-4)
    } else {
        None
    };

    let expected = if number == 1006 { 168 } else { 152 };
    check_length(number, bits.len(), expected)?;

    #[allow(clippy::cast_precision_loss)] // 38-bit integers are exact in f64
    Ok(StationReferencePoint {
        station_id,
        itrf_year,
        ecef: [x as f64 * 1e-4, y as f64 * 1e-4, z as f64 * 1e-4],
        antenna_height,
        services: [gps, glonass, galileo],
    })
}

fn decode_gps_ephemeris(bits: &mut Bits) -> Result<BroadcastEphemeris, DecodeError> {
    let truncated = || DecodeError::Truncated { message: 1019 };
    let prn = bits.u8(6).ok_or_else(truncated)?;
    let week = bits.u32(10).ok_or_else(truncated)? as u16;
    let _accuracy = bits.u8(4).ok_or_else(truncated)?;
    let _code_on_l2 = bits.u8(2).ok_or_else(truncated)?;
    let idot = bits.scaled(14, -43).ok_or_else(truncated)? * SEMICIRCLE;
    let _iode = bits.u8(8).ok_or_else(truncated)?;
    let toc = bits.scaled_unsigned(16, 4).ok_or_else(truncated)?;
    let af2 = bits.scaled(8, -55).ok_or_else(truncated)?;
    let af1 = bits.scaled(16, -43).ok_or_else(truncated)?;
    let af0 = bits.scaled(22, -31).ok_or_else(truncated)?;
    let _iodc = bits.u32(10).ok_or_else(truncated)?;
    let crs = bits.scaled(16, -5).ok_or_else(truncated)?;
    let delta_n = bits.scaled(16, -43).ok_or_else(truncated)? * SEMICIRCLE;
    let m0 = bits.scaled(32, -31).ok_or_else(truncated)? * SEMICIRCLE;
    let cuc = bits.scaled(16, -29).ok_or_else(truncated)?;
    let e = bits.scaled_unsigned(32, -33).ok_or_else(truncated)?;
    let cus = bits.scaled(16, -29).ok_or_else(truncated)?;
    let sqrt_a = bits.scaled_unsigned(32, -19).ok_or_else(truncated)?;
    let toe = bits.scaled_unsigned(16, 4).ok_or_else(truncated)?;
    let cic = bits.scaled(16, -29).ok_or_else(truncated)?;
    let omega0 = bits.scaled(32, -31).ok_or_else(truncated)? * SEMICIRCLE;
    let cis = bits.scaled(16, -29).ok_or_else(truncated)?;
    let i0 = bits.scaled(32, -31).ok_or_else(truncated)? * SEMICIRCLE;
    let crc_harmonic = bits.scaled(16, -5).ok_or_else(truncated)?;
    let omega = bits.scaled(32, -31).ok_or_else(truncated)? * SEMICIRCLE;
    let omega_dot = bits.scaled(24, -43).ok_or_else(truncated)? * SEMICIRCLE;
    let tgd = bits.scaled(8, -31).ok_or_else(truncated)?;
    let health = bits.u8(6).ok_or_else(truncated)?;
    let _l2p_data = bits.u8(1).ok_or_else(truncated)?;
    let _fit_interval = bits.u8(1).ok_or_else(truncated)?;

    check_length(1019, bits.len(), 488)?;

    Ok(BroadcastEphemeris {
        system: GnssSystem::Gps,
        prn,
        week,
        toe,
        toc,
        sqrt_a,
        e,
        i0,
        omega0,
        omega,
        m0,
        delta_n,
        idot,
        omega_dot,
        cuc,
        cus,
        crc: crc_harmonic,
        crs,
        cic,
        cis,
        af0,
        af1,
        af2,
        tgd,
        health,
    })
}

/// The header the SSR messages share, returning the polynomial reference epoch, the SSR issue of
/// data, and the satellite count.
///
/// `has_datum` is the one-bit satellite-reference-datum field, which the orbit-bearing messages
/// (1057, 1060) carry and the clock-only message (1058) does not. Getting it wrong shifts every
/// subsequent field by one bit, which is why each message's satellite payload is checked by round
/// trip against physical values rather than by inspection.
fn decode_ssr_header(
    bits: &mut Bits,
    number: u16,
    has_datum: bool,
) -> Result<(f64, u8, usize), DecodeError> {
    let truncated = || DecodeError::Truncated { message: number };
    let epoch = f64::from(bits.u32(20).ok_or_else(truncated)?);
    let interval = bits.u8(4).ok_or_else(truncated)? as usize;
    let _multiple_message = bits.u8(1).ok_or_else(truncated)?;
    if has_datum {
        let _datum = bits.u8(1).ok_or_else(truncated)?;
    }
    let iod_ssr = bits.u8(4).ok_or_else(truncated)?;
    let _provider = bits.u32(16).ok_or_else(truncated)?;
    let _solution = bits.u8(4).ok_or_else(truncated)?;
    let count = bits.u8(6).ok_or_else(truncated)? as usize;
    if count > MAX_SSR_SATELLITES {
        return Err(DecodeError::TooManySatellites {
            message: number,
            count,
        });
    }
    // The epoch stamps the start of the validity interval; the polynomial is about its midpoint.
    let reference = epoch + SSR_UPDATE_INTERVALS[interval] / 2.0;
    Ok((reference, iod_ssr, count))
}

#[allow(clippy::cast_precision_loss)] // the widest field here is 27 bits
fn read_orbit(
    bits: &mut Bits,
    number: u16,
    reference: f64,
) -> Result<OrbitCorrection, DecodeError> {
    let truncated = || DecodeError::Truncated { message: number };
    Ok(OrbitCorrection {
        prn: bits.u8(6).ok_or_else(truncated)?,
        iode: bits.u8(8).ok_or_else(truncated)?,
        reference,
        delta: [
            bits.i64(22).ok_or_else(truncated)? as f64 * 1e-4,
            bits.i64(20).ok_or_else(truncated)? as f64 * 4e-4,
            bits.i64(20).ok_or_else(truncated)? as f64 * 4e-4,
        ],
        delta_rate: [
            bits.i64(21).ok_or_else(truncated)? as f64 * 1e-6,
            bits.i64(19).ok_or_else(truncated)? as f64 * 4e-6,
            bits.i64(19).ok_or_else(truncated)? as f64 * 4e-6,
        ],
    })
}

/// In 1060 the satellite is named once and the clock fields follow the orbit fields, so `prn` is
/// supplied by the caller there rather than read again.
#[allow(clippy::cast_precision_loss)]
fn read_clock(
    bits: &mut Bits,
    number: u16,
    reference: f64,
    prn: Option<u8>,
) -> Result<ClockCorrection, DecodeError> {
    let truncated = || DecodeError::Truncated { message: number };
    let prn = match prn {
        Some(prn) => prn,
        None => bits.u8(6).ok_or_else(truncated)?,
    };
    Ok(ClockCorrection {
        prn,
        reference,
        coefficients: [
            bits.i64(22).ok_or_else(truncated)? as f64 * 1e-4,
            bits.i64(21).ok_or_else(truncated)? as f64 * 1e-6,
            bits.i64(27).ok_or_else(truncated)? as f64 * 2e-8,
        ],
    })
}

const EMPTY_ORBIT: OrbitCorrection = OrbitCorrection {
    prn: 0,
    iode: 0,
    reference: 0.0,
    delta: [0.0; 3],
    delta_rate: [0.0; 3],
};

const EMPTY_CLOCK: ClockCorrection = ClockCorrection {
    prn: 0,
    reference: 0.0,
    coefficients: [0.0; 3],
};

fn decode_ssr_orbit(bits: &mut Bits, number: u16) -> Result<SsrOrbitBlock, DecodeError> {
    let (reference, iod_ssr, count) = decode_ssr_header(bits, number, true)?;
    let mut satellites = [EMPTY_ORBIT; MAX_SSR_SATELLITES];
    for satellite in satellites.iter_mut().take(count) {
        *satellite = read_orbit(bits, number, reference)?;
    }
    check_length(number, bits.len(), bits.consumed())?;
    Ok(SsrOrbitBlock {
        satellites,
        count,
        iod_ssr,
    })
}

fn decode_ssr_clock(bits: &mut Bits, number: u16) -> Result<SsrClockBlock, DecodeError> {
    let (reference, iod_ssr, count) = decode_ssr_header(bits, number, false)?;
    let mut satellites = [EMPTY_CLOCK; MAX_SSR_SATELLITES];
    for satellite in satellites.iter_mut().take(count) {
        *satellite = read_clock(bits, number, reference, None)?;
    }
    check_length(number, bits.len(), bits.consumed())?;
    Ok(SsrClockBlock {
        satellites,
        count,
        iod_ssr,
    })
}

fn decode_ssr_combined(bits: &mut Bits, number: u16) -> Result<Message, DecodeError> {
    let (reference, iod_ssr, count) = decode_ssr_header(bits, number, true)?;
    let mut orbits = [EMPTY_ORBIT; MAX_SSR_SATELLITES];
    let mut clocks = [EMPTY_CLOCK; MAX_SSR_SATELLITES];
    for i in 0..count {
        let orbit = read_orbit(bits, number, reference)?;
        clocks[i] = read_clock(bits, number, reference, Some(orbit.prn))?;
        orbits[i] = orbit;
    }
    check_length(number, bits.len(), bits.consumed())?;
    Ok(Message::SsrOrbitAndClock(
        SsrOrbitBlock {
            satellites: orbits,
            count,
            iod_ssr,
        },
        SsrClockBlock {
            satellites: clocks,
            count,
            iod_ssr,
        },
    ))
}

/// A payload must hold the message and nothing beyond its padding to a byte boundary.
///
/// A payload *shorter* than the message never reaches here — the reads run out first and report
/// [`DecodeError::Truncated`]. What this catches is the opposite: a producer whose message is longer
/// than the one its own number names, whose surplus bits are a misalignment somewhere earlier that
/// every field after it inherited.
///
/// For the variable-length SSR messages the same call bounds the satellite blocks, and it is worth
/// saying what that does and does not prove. Under-reading the block by `k` bits leaves `k` more
/// trailing bits, so it is caught for `k ≥ 2`; over-reading either exhausts the payload or, for
/// `k ≥ 7`, fails here. A single-bit width error inside a block survives it, which is why the SSR
/// fields are also checked by round trip against physical values.
fn check_length(message: u16, payload_bits: usize, expected: usize) -> Result<(), DecodeError> {
    if payload_bits >= expected && payload_bits - expected < 8 {
        Ok(())
    } else {
        Err(DecodeError::WrongLength {
            message,
            payload_bits,
            expected,
        })
    }
}

/// A frame found in a byte stream: where it started, and its payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame<'a> {
    /// Offset of the preamble within the buffer the frame was found in.
    pub offset: usize,
    /// The payload, between the length field and the CRC.
    pub payload: &'a [u8],
}

/// A resynchronising scanner over an RTCM 3 byte stream.
///
/// Holds no buffer of its own: the caller owns the bytes and the decoder walks them, which keeps it
/// `no_std` and lets a client hand it a ring buffer, a file, or a socket read directly. When it
/// stops, [`RtcmDecoder::remaining`] says how many trailing bytes belong to a frame that has not
/// fully arrived and must be carried into the next read.
#[derive(Debug, Clone)]
pub struct RtcmDecoder<'a> {
    bytes: &'a [u8],
    cursor: usize,
    rejected: usize,
}

impl<'a> RtcmDecoder<'a> {
    /// Scan `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            cursor: 0,
            rejected: 0,
        }
    }

    /// How many candidate preambles were rejected — a false preamble in payload data, or a genuine
    /// frame corrupted in transit. A client watching this climb is watching its link degrade, which
    /// is why it is counted rather than discarded.
    #[must_use]
    pub const fn rejected(&self) -> usize {
        self.rejected
    }

    /// How many bytes remain unscanned.
    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.bytes.len() - self.cursor
    }
}

impl<'a> Iterator for RtcmDecoder<'a> {
    type Item = Frame<'a>;

    fn next(&mut self) -> Option<Frame<'a>> {
        while self.cursor + 3 <= self.bytes.len() {
            if self.bytes[self.cursor] != PREAMBLE {
                self.cursor += 1;
                continue;
            }
            let header = &self.bytes[self.cursor..self.cursor + 3];
            // The six reserved bits must be zero; a non-zero pattern is a byte that merely looks
            // like a preamble, and rejecting it here saves a CRC over up to a kilobyte of noise.
            if header[1] & 0xFC != 0 {
                self.cursor += 1;
                self.rejected += 1;
                continue;
            }
            let length = (usize::from(header[1] & 0x03) << 8) | usize::from(header[2]);
            let total = 3 + length + 3;
            if self.cursor + total > self.bytes.len() {
                // Not a rejection: the rest of the frame may simply not have arrived yet.
                return None;
            }
            let frame = &self.bytes[self.cursor..self.cursor + total];
            let stated = (u32::from(frame[total - 3]) << 16)
                | (u32::from(frame[total - 2]) << 8)
                | u32::from(frame[total - 1]);
            if crc24q(&frame[..total - 3]) == stated {
                let payload = &self.bytes[self.cursor + 3..self.cursor + 3 + length];
                let offset = self.cursor;
                self.cursor += total;
                return Some(Frame { offset, payload });
            }
            // Advance one byte, never by the length a failed header claimed: that length is exactly
            // the field a corruption is most likely to have moved.
            self.cursor += 1;
            self.rejected += 1;
        }
        None
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    /// A most-significant-bit-first writer, so a message can be built from physical values and read
    /// back. It is the decoder's inverse and shares no code with it; where the two disagree about a
    /// field width the fixed-length messages fail [`check_length`] and the variable-length ones
    /// fail the round trip.
    #[derive(Default)]
    struct Writer {
        bytes: Vec<u8>,
        bits: usize,
    }

    impl Writer {
        fn put(&mut self, value: u64, width: usize) {
            for i in (0..width).rev() {
                if self.bits.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                if (value >> i) & 1 == 1 {
                    let last = self.bytes.len() - 1;
                    self.bytes[last] |= 1 << (7 - self.bits % 8);
                }
                self.bits += 1;
            }
        }

        fn put_signed(&mut self, value: i64, width: usize) {
            let mask = if width == 64 {
                u64::MAX
            } else {
                (1 << width) - 1
            };
            self.put((value as u64) & mask, width);
        }

        /// Quantise a physical value by a power-of-two scale, as an IS-GPS-200 producer does.
        fn put_scaled(&mut self, value: f64, width: usize, power: i32) {
            self.put_signed(libm::round(libm::ldexp(value, -power)) as i64, width);
        }

        fn put_scaled_unsigned(&mut self, value: f64, width: usize, power: i32) {
            self.put(libm::round(libm::ldexp(value, -power)) as u64, width);
        }

        /// The SSR fields are scaled by decimal metric steps, not powers of two.
        fn put_metric(&mut self, value: f64, width: usize, step: f64) {
            self.put_signed(libm::round(value / step) as i64, width);
        }

        fn frame(mut self) -> Vec<u8> {
            // Pad to a byte boundary, as a producer must.
            let pad = (8 - self.bits % 8) % 8;
            self.put(0, pad);
            let length = self.bytes.len();
            let mut out = vec![PREAMBLE, (length >> 8) as u8 & 0x03, length as u8];
            out.extend_from_slice(&self.bytes);
            let crc = crc24q(&out);
            out.push((crc >> 16) as u8);
            out.push((crc >> 8) as u8);
            out.push(crc as u8);
            out
        }
    }

    /// A realistic GPS broadcast ephemeris, in physical units: a mid-plane MEO satellite whose `√a`
    /// puts the semi-major axis at 26 560 km, which is what makes the decoded orbit checkable
    /// against physics rather than against the encoder.
    const SQRT_A: f64 = 5153.65;
    const ECCENTRICITY: f64 = 0.008_6;
    const INCLINATION: f64 = 0.961_7;
    const TOE: f64 = 302_400.0;

    fn semicircles(radians: f64) -> f64 {
        radians / SEMICIRCLE
    }

    fn gps_ephemeris_frame() -> Vec<u8> {
        let mut w = Writer::default();
        w.put(1019, 12);
        w.put(7, 6); // PRN
        w.put(252, 10); // week, modulo 1024
        w.put(0, 4); // URA index
        w.put(1, 2); // code on L2
        w.put_scaled(semicircles(1.0e-10), 14, -43); // IDOT
        w.put(41, 8); // IODE
        w.put_scaled_unsigned(TOE, 16, 4); // toc
        w.put_scaled(5.0e-17, 8, -55); // af2
        w.put_scaled(1.0e-12, 16, -43); // af1
        w.put_scaled(1.0e-4, 22, -31); // af0
        w.put(41, 10); // IODC
        w.put_scaled(20.0, 16, -5); // Crs
        w.put_scaled(semicircles(4.6e-9), 16, -43); // dn
        w.put_scaled(semicircles(0.5), 32, -31); // M0
        w.put_scaled(1.0e-6, 16, -29); // Cuc
        w.put_scaled_unsigned(ECCENTRICITY, 32, -33);
        w.put_scaled(5.0e-6, 16, -29); // Cus
        w.put_scaled_unsigned(SQRT_A, 32, -19);
        w.put_scaled_unsigned(TOE, 16, 4);
        w.put_scaled(-1.0e-7, 16, -29); // Cic
        w.put_scaled(semicircles(-1.0), 32, -31); // Omega0
        w.put_scaled(1.0e-7, 16, -29); // Cis
        w.put_scaled(semicircles(INCLINATION), 32, -31); // i0
        w.put_scaled(250.0, 16, -5); // Crc
        w.put_scaled(semicircles(0.9), 32, -31); // omega
        w.put_scaled(semicircles(-8.0e-9), 24, -43); // Omega dot
        w.put_scaled(-5.0e-9, 8, -31); // TGD
        w.put(0, 6); // health
        w.put(0, 1); // L2 P data flag
        w.put(0, 1); // fit interval
        w.frame()
    }

    /// An SSR message for two satellites; 1060 additionally carries the clock terms per satellite.
    fn ssr_frame(number: u16) -> Vec<u8> {
        let mut w = Writer::default();
        w.put(u64::from(number), 12);
        w.put(302_400, 20); // epoch, s of GPS week
        w.put(6, 4); // update interval index -> 60 s
        w.put(0, 1); // multiple message indicator
        if number != 1058 {
            w.put(0, 1); // satellite reference datum
        }
        w.put(5, 4); // IOD SSR
        w.put(0, 16); // provider
        w.put(0, 4); // solution
        w.put(2, 6); // satellite count
        for (prn, iode, radial) in [(7_u64, 41_u64, 0.312_5), (9, 17, -1.406_25)] {
            w.put(prn, 6);
            if number == 1058 {
                w.put_metric(3.5, 22, 1e-4);
                w.put_metric(-0.01, 21, 1e-6);
                w.put_metric(2.0e-6, 27, 2e-8);
                continue;
            }
            w.put(iode, 8);
            w.put_metric(radial, 22, 1e-4);
            w.put_metric(0.2, 20, 4e-4);
            w.put_metric(-0.4, 20, 4e-4);
            w.put_metric(1.0e-3, 21, 1e-6);
            w.put_metric(-2.0e-3, 19, 4e-6);
            w.put_metric(4.0e-3, 19, 4e-6);
            if number == 1060 {
                w.put_metric(3.5, 22, 1e-4);
                w.put_metric(-0.01, 21, 1e-6);
                w.put_metric(2.0e-6, 27, 2e-8);
            }
        }
        w.frame()
    }

    fn decode_first(stream: &[u8]) -> Message {
        let frame = RtcmDecoder::new(stream).next().expect("one framed message");
        decode(frame.payload).expect("a decodable payload")
    }

    #[test]
    fn the_crc_reproduces_the_published_check_value() {
        // The reveng catalogue's check value for CRC-24/LTE-A, which is RTCM's CRC-24Q: the same
        // polynomial as CRC-24/OPENPGP but seeded with zero rather than 0xB704CE. Asserting the
        // OpenPGP check value here — 0x21CF02, the one a search finds first — would be asserting a
        // different algorithm.
        assert_eq!(crc24q(b"123456789"), 0x00CD_E703);
        // The degenerate cases that separate a zero initial value from a non-zero one: an empty
        // message and a zero byte both have zero remainder, a one byte does not.
        assert_eq!(crc24q(b""), 0);
        assert_eq!(crc24q(&[0x00]), 0);
        assert_ne!(crc24q(&[0x01]), 0);
    }

    #[test]
    fn a_flipped_bit_anywhere_in_a_frame_is_refused() {
        // The property the CRC is carried for: every single-bit flip in the frame must fail the
        // check, so none can reach a decoder. The fixture is a real 1057, 49 bytes framed, so this
        // is 392 corruptions and not a spot check.
        let good = ssr_frame(1057);
        assert_eq!(RtcmDecoder::new(&good).count(), 1);
        // Pinned so the sweep below cannot silently shrink if the fixture changes.
        assert_eq!(good.len(), 49, "the 1057 fixture is 3 + 43 + 3 bytes");
        let payload = RtcmDecoder::new(&good)
            .next()
            .expect("a frame")
            .payload
            .to_vec();
        for byte in 0..good.len() {
            for bit in 0..8 {
                let mut corrupted = good.clone();
                corrupted[byte] ^= 1 << bit;
                let found = RtcmDecoder::new(&corrupted).next();
                assert!(
                    found.is_none_or(|f| f.payload != payload.as_slice()),
                    "byte {byte} bit {bit} passed the CRC unchanged"
                );
            }
        }
    }

    #[test]
    fn the_scanner_resynchronises_past_noise_and_false_preambles() {
        let first = gps_ephemeris_frame();
        let second = ssr_frame(1058);
        let mut stream = vec![0xD3, 0xFF, 0x01, 0x99]; // reserved bits set: not a frame
        stream.extend_from_slice(&first);
        stream.extend_from_slice(&[0x01, 0x02, 0xD3, 0x00, 0x04, 0xAA, 0xBB]); // a truncated liar
        stream.extend_from_slice(&second);

        let mut decoder = RtcmDecoder::new(&stream);
        let a = decoder.next().expect("the first frame");
        let b = decoder.next().expect("the second frame");
        assert_eq!(a.offset, 4);
        assert!(matches!(decode(a.payload), Ok(Message::GpsEphemeris(_))));
        assert!(matches!(decode(b.payload), Ok(Message::SsrClock(_))));
        assert!(decoder.next().is_none());
        assert!(decoder.rejected() >= 2, "rejected {}", decoder.rejected());
    }

    #[test]
    fn a_frame_that_has_not_fully_arrived_is_not_a_rejection() {
        // The distinction a socket client depends on: a partial frame at the end of a read must
        // leave the scanner silent rather than counting a fault, or a healthy link would look
        // broken at every buffer boundary.
        let complete = gps_ephemeris_frame();
        for cut in 1..complete.len() {
            let mut decoder = RtcmDecoder::new(&complete[..cut]);
            assert!(decoder.next().is_none(), "cut {cut} yielded a frame");
            assert_eq!(decoder.rejected(), 0, "cut {cut} counted a rejection");
            assert_eq!(
                decoder.remaining(),
                cut,
                "cut {cut} consumed bytes it should have kept"
            );
        }
    }

    #[test]
    fn the_ephemeris_decodes_to_an_orbit_of_the_right_size_and_period() {
        // The check that matters: not that the round trip agrees with the encoder, but that the
        // decoded numbers are a GPS orbit. A transposed scale factor anywhere in the 488 bits moves
        // the semi-major axis or the mean motion off the constellation's, and both are pinned here
        // against IS-GPS-200 physics rather than against the fixture.
        let Message::GpsEphemeris(eph) = decode_first(&gps_ephemeris_frame()) else {
            panic!("1019 must decode as an ephemeris");
        };
        let a = eph.semi_major_axis();
        assert!(
            (a - SQRT_A * SQRT_A).abs() < 1.0,
            "semi-major axis {a} m, expected {} m",
            SQRT_A * SQRT_A
        );
        // A GPS satellite has a half-sidereal-day period, 43 082 s. The corrected mean motion this
        // fixture carries puts it at 43 076.7 s -- the 5.3 s is `delta_n` plus the fixture's own
        // round `sqrt_a`, and a scale-factor error anywhere would be orders of magnitude larger.
        let period = 2.0 * core::f64::consts::PI / eph.mean_motion();
        assert!((period - 43_082.0).abs() < 20.0, "period {period} s");

        // And the radius at the reference epoch is on the orbit `a` and `e` describe, which no
        // single field could produce on its own.
        let p = eph.position_ecef(TOE);
        let radius = libm::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
        assert!(
            radius > a * (1.0 - ECCENTRICITY) - 500.0 && radius < a * (1.0 + ECCENTRICITY) + 500.0,
            "radius {radius} m outside [{}, {}]",
            a * (1.0 - ECCENTRICITY),
            a * (1.0 + ECCENTRICITY)
        );

        // Then the fields whose quantisation is fine enough to name exactly.
        assert_eq!(eph.prn, 7);
        assert_eq!(eph.week, 252);
        assert_eq!(eph.system, GnssSystem::Gps);
        assert!((eph.toe - TOE).abs() < 1e-9);
        assert!((eph.toc - TOE).abs() < 1e-9);
        // Each tolerance below is that field's own quantisation step, which is what a round trip can
        // pin and no more: 2^-33 on `e`, 2^-31 semicircles (1.5e-9 rad) on the four angles, 2^-5 m
        // on the harmonic radii, 2^-31 s on TGD and 2^-43 semicircles on the node rate. A wrong
        // scale factor is orders of magnitude coarser than any of them.
        assert!((eph.e - ECCENTRICITY).abs() < 1.2e-10);
        assert!((eph.i0 - INCLINATION).abs() < 1.5e-9);
        assert!((eph.omega0 + 1.0).abs() < 1.5e-9);
        assert!((eph.omega - 0.9).abs() < 1.5e-9);
        assert!((eph.m0 - 0.5).abs() < 1.5e-9);
        assert!((eph.crc - 250.0).abs() < 0.032);
        assert!((eph.crs - 20.0).abs() < 0.032);
        assert!((eph.tgd + 5.0e-9).abs() < 4.7e-10);
        assert!((eph.omega_dot + 8.0e-9).abs() < 3.6e-13);

        // The clock polynomial and the harmonic corrections, on the same rule. Neither the orbit's
        // size nor its period depends on any of them, so everything above this line would pass with
        // a transposed scale factor on all eight. `af2` in particular has to carry a value: a field
        // the fixture leaves at zero is a field whose scale factor no round trip can observe.
        assert!((eph.af0 - 1.0e-4).abs() < 4.7e-10);
        assert!((eph.af1 - 1.0e-12).abs() < 1.2e-13);
        assert!((eph.af2 - 5.0e-17).abs() < 2.8e-17);
        assert!((eph.cuc - 1.0e-6).abs() < 1.9e-9);
        assert!((eph.cus - 5.0e-6).abs() < 1.9e-9);
        assert!((eph.cic + 1.0e-7).abs() < 1.9e-9);
        assert!((eph.cis - 1.0e-7).abs() < 1.9e-9);
        // The two rate fields the standard publishes in semicircles: the factor of pi is part of
        // the decode, and dropping or inverting it is a mean-motion error of a factor of ten.
        assert!((eph.idot - 1.0e-10).abs() < 3.6e-13);
        assert!((eph.delta_n - 4.6e-9).abs() < 3.6e-13);
    }

    #[test]
    fn the_ssr_orbit_message_round_trips_and_is_referenced_to_the_interval_midpoint() {
        let Message::SsrOrbit(block) = decode_first(&ssr_frame(1057)) else {
            panic!("1057 must decode as an orbit block");
        };
        assert_eq!(block.iod_ssr, 5);
        let satellites = block.satellites();
        assert_eq!(satellites.len(), 2);
        assert_eq!(satellites[0].prn, 7);
        assert_eq!(satellites[0].iode, 41);
        assert_eq!(satellites[1].prn, 9);
        assert_eq!(satellites[1].iode, 17);
        assert!((satellites[0].delta[0] - 0.312_5).abs() < 1e-4);
        assert!((satellites[1].delta[0] + 1.406_25).abs() < 1e-4);
        assert!((satellites[0].delta[1] - 0.2).abs() < 4e-4);
        assert!((satellites[0].delta[2] + 0.4).abs() < 4e-4);
        assert!((satellites[0].delta_rate[0] - 1.0e-3).abs() < 1e-6);
        assert!((satellites[0].delta_rate[1] + 2.0e-3).abs() < 4e-6);
        assert!((satellites[0].delta_rate[2] - 4.0e-3).abs() < 4e-6);

        // Epoch 302 400 with a 60 s interval: the polynomial is about 302 430.
        assert!((satellites[0].reference - 302_430.0).abs() < 1e-9);
        // Which is what makes the correction *at the epoch itself* not the constant term.
        let at_epoch = satellites[0].at(302_400.0);
        assert!(
            (at_epoch[0] - (0.312_5 - 30.0e-3)).abs() < 1e-4,
            "{at_epoch:?}"
        );
    }

    #[test]
    fn the_clock_only_message_has_no_datum_bit_and_still_aligns() {
        // 1058's header is one bit shorter than 1057's. Reading it with the orbit header's width
        // would shift every field after it, so the values are checked rather than the length.
        let Message::SsrClock(block) = decode_first(&ssr_frame(1058)) else {
            panic!("1058 must decode as a clock block");
        };
        assert_eq!(block.iod_ssr, 5);
        assert_eq!(block.satellites().len(), 2);
        for (satellite, prn) in block.satellites().iter().zip([7, 9]) {
            assert_eq!(satellite.prn, prn);
            assert!((satellite.coefficients[0] - 3.5).abs() < 1e-4);
            assert!((satellite.coefficients[1] + 0.01).abs() < 1e-6);
            assert!((satellite.coefficients[2] - 2.0e-6).abs() < 2e-8);
            assert!((satellite.reference - 302_430.0).abs() < 1e-9);
        }
    }

    #[test]
    fn the_combined_message_names_each_satellite_once_for_both_corrections() {
        // 1060 interleaves the orbit and clock fields per satellite and does not repeat the PRN. A
        // decoder that re-read it would be six bits out from the second field onwards.
        let Message::SsrOrbitAndClock(orbit, clock) = decode_first(&ssr_frame(1060)) else {
            panic!("1060 must decode as both blocks");
        };
        assert_eq!(orbit.satellites().len(), 2);
        assert_eq!(clock.satellites().len(), 2);
        assert_eq!(orbit.iod_ssr, 5);
        assert_eq!(clock.iod_ssr, 5);
        for (o, c) in orbit.satellites().iter().zip(clock.satellites()) {
            assert_eq!(o.prn, c.prn);
            assert!((o.reference - c.reference).abs() < 1e-9);
            assert!((c.coefficients[0] - 3.5).abs() < 1e-4);
            assert!((c.coefficients[1] + 0.01).abs() < 1e-6);
            assert!((c.coefficients[2] - 2.0e-6).abs() < 2e-8);
        }
        assert_eq!(orbit.satellites()[0].iode, 41);
        assert_eq!(orbit.satellites()[1].iode, 17);
    }

    #[test]
    fn the_station_message_round_trips_and_only_1006_carries_a_height() {
        for number in [1005_u16, 1006] {
            let mut w = Writer::default();
            w.put(u64::from(number), 12);
            w.put(1234, 12); // station ID
            w.put(20, 6); // ITRF year
            w.put(1, 1); // GPS
            w.put(0, 1); // GLONASS
            w.put(1, 1); // Galileo
            w.put(0, 1); // reference-station indicator
            w.put_signed(38_378_286_400, 38); // x = 3 837 828.64 m
            w.put(0, 1); // oscillator indicator
            w.put(0, 1); // reserved
            w.put_signed(45_534_215_000, 38); // y = 4 553 421.50 m
            w.put(0, 2); // quarter-cycle indicator
            w.put_signed(-6_186_112_000, 38); // z = -618 611.20 m
            if number == 1006 {
                w.put(21_340, 16); // antenna height 2.134 m
            }
            let Message::StationReferencePoint(station) = decode_first(&w.frame()) else {
                panic!("{number} must decode as a station reference point");
            };
            assert_eq!(station.station_id, 1234);
            assert_eq!(station.itrf_year, 20);
            assert_eq!(station.services, [true, false, true]);
            assert!(
                (station.ecef[0] - 3_837_828.64).abs() < 1e-4,
                "{:?}",
                station.ecef
            );
            assert!(
                (station.ecef[1] - 4_553_421.50).abs() < 1e-4,
                "{:?}",
                station.ecef
            );
            assert!(
                (station.ecef[2] + 618_611.20).abs() < 1e-4,
                "{:?}",
                station.ecef
            );
            match (number, station.antenna_height) {
                (1006, Some(h)) => assert!((h - 2.134).abs() < 1e-4, "{h}"),
                (1005, None) => {}
                other => panic!("{other:?} is the wrong height for {number}"),
            }
        }
    }

    #[test]
    fn a_message_that_is_not_its_own_declared_length_is_rejected() {
        // 1019 is 488 bits. A producer that emits fewer must not silently yield an ephemeris with
        // zeros where the tail should be, because those zeros are a valid-looking orbit; one that
        // emits more has misaligned somewhere earlier and its fields cannot be trusted either.
        let full = gps_ephemeris_frame();
        let payload = RtcmDecoder::new(&full)
            .next()
            .expect("a frame")
            .payload
            .to_vec();
        assert_eq!(
            decode(&payload[..40]),
            Err(DecodeError::Truncated { message: 1019 })
        );
        let mut padded = payload;
        padded.push(0);
        assert_eq!(
            decode(&padded),
            Err(DecodeError::WrongLength {
                message: 1019,
                payload_bits: 496,
                expected: 488
            })
        );
    }

    #[test]
    fn an_unmodelled_message_is_reported_rather_than_failed() {
        // 1074 is an MSM4 observation block, deliberately not decoded. It must come back named, so
        // a client can count what it is dropping.
        let mut w = Writer::default();
        w.put(1074, 12);
        w.put(0, 20);
        assert_eq!(decode_first(&w.frame()), Message::Unsupported(1074));
        // And a payload too short even to name itself is an error, not an unsupported message.
        assert_eq!(decode(&[0x43]), Err(DecodeError::Truncated { message: 0 }));
    }

    #[test]
    fn a_satellite_count_larger_than_the_field_can_hold_is_refused() {
        // DF387 is six bits, so 63 is the largest legal count and a message claiming it must not be
        // truncated into a shorter list. Here it claims 63 and carries none.
        let mut w = Writer::default();
        w.put(1057, 12);
        w.put(302_400, 20);
        w.put(6, 4);
        w.put(0, 1);
        w.put(0, 1);
        w.put(5, 4);
        w.put(0, 16);
        w.put(0, 4);
        w.put(63, 6);
        assert_eq!(
            decode(
                RtcmDecoder::new(&w.frame())
                    .next()
                    .expect("a frame")
                    .payload
            ),
            Err(DecodeError::Truncated { message: 1057 })
        );
    }

    #[test]
    fn the_orbit_correction_is_subtracted_and_the_clock_correction_added() {
        // The sign convention, on a case computed by hand. The satellite is on the +x axis moving
        // along +y, so radial is +x, cross-track is +z and along-track is +y — and a +1 m radial
        // correction must move the position to 25 999 999 m, not 26 000 001.
        let corrected = apply_orbit(
            [26_000_000.0, 0.0, 0.0],
            [0.0, 3_000.0, 0.0],
            [1.0, 2.0, 3.0],
        )
        .expect("a non-degenerate state");
        assert!((corrected[0] - 25_999_999.0).abs() < 1e-9, "{corrected:?}");
        assert!((corrected[1] + 2.0).abs() < 1e-9, "{corrected:?}");
        assert!((corrected[2] + 3.0).abs() < 1e-9, "{corrected:?}");

        // A zero correction must leave the position exactly where it was, which is what separates a
        // sign error from a basis error.
        let unchanged = apply_orbit([1.0e7, 2.0e7, 3.0e6], [-1.0e3, 2.0e3, 5.0e2], [0.0; 3])
            .expect("a non-degenerate state");
        assert!((unchanged[0] - 1.0e7).abs() < 1e-6, "{unchanged:?}");
        assert!((unchanged[1] - 2.0e7).abs() < 1e-6, "{unchanged:?}");
        assert!((unchanged[2] - 3.0e6).abs() < 1e-6, "{unchanged:?}");

        let clock = ClockCorrection {
            prn: 1,
            reference: 100.0,
            coefficients: [SPEED_OF_LIGHT * 1e-6, 0.0, 0.0],
        };
        assert!((clock.seconds_at(100.0) - 1e-6).abs() < 1e-18);
    }

    #[test]
    fn a_degenerate_state_has_no_orbital_basis_and_says_so() {
        // Velocity parallel to position: the cross-track axis does not exist. A silent fallback
        // would put the along-track correction on an arbitrary axis.
        assert!(apply_orbit([7e6, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).is_none());
        assert!(apply_orbit([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]).is_none());
        assert!(apply_orbit([f64::NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]).is_none());
    }

    #[test]
    fn the_correction_polynomials_are_the_ones_the_standard_defines() {
        let orbit = OrbitCorrection {
            prn: 5,
            iode: 42,
            reference: 60.0,
            delta: [1.0, 2.0, 3.0],
            delta_rate: [0.1, 0.2, 0.3],
        };
        let at = orbit.at(70.0);
        assert!((at[0] - 2.0).abs() < 1e-12, "{at:?}");
        assert!((at[1] - 4.0).abs() < 1e-12, "{at:?}");
        assert!((at[2] - 6.0).abs() < 1e-12, "{at:?}");

        // 1 + 0.5·10 + 0.25·100 — quadratic, not linear, which is the term an implementation drops.
        let clock = ClockCorrection {
            prn: 5,
            reference: 60.0,
            coefficients: [1.0, 0.5, 0.25],
        };
        assert!((clock.at(70.0) - 31.0).abs() < 1e-12, "{}", clock.at(70.0));
        // And backwards in time, where a dropped sign on `dt` would look identical at the midpoint.
        assert!((clock.at(50.0) - 21.0).abs() < 1e-12, "{}", clock.at(50.0));
    }

    #[test]
    fn the_update_interval_table_is_the_one_df391_names() {
        // A transposed entry shifts every SSR reference epoch by seconds, which shows up as a slow
        // drift rather than a failure. Both ends and the two values a real stream uses are pinned.
        assert!((SSR_UPDATE_INTERVALS[0] - 1.0).abs() < 1e-12);
        assert!((SSR_UPDATE_INTERVALS[6] - 60.0).abs() < 1e-12);
        assert!((SSR_UPDATE_INTERVALS[7] - 120.0).abs() < 1e-12);
        assert!((SSR_UPDATE_INTERVALS[15] - 10_800.0).abs() < 1e-12);
        assert!(
            SSR_UPDATE_INTERVALS.windows(2).all(|w| w[1] > w[0]),
            "the table must be strictly increasing"
        );
    }

    #[test]
    fn the_bit_reader_sign_extends_and_refuses_to_read_past_the_end() {
        let bytes = [0b1000_0000_u8, 0b0000_0001];
        let mut bits = Bits::new(&bytes);
        // 0b1000_0000_0000_0001 read as 16 signed bits is -32 767.
        assert_eq!(bits.i64(16), Some(-32_767));
        assert_eq!(bits.u64(1), None);

        let mut bits = Bits::new(&bytes);
        assert_eq!(bits.u64(16), Some(0x8001));
        assert_eq!(bits.consumed(), 16);

        // A one-bit signed field is -1 or 0, which is where an off-by-one in the sign mask shows.
        let mut bits = Bits::new(&[0b1000_0000]);
        assert_eq!(bits.i64(1), Some(-1));
        assert_eq!(bits.i64(1), Some(0));

        // A width the reader cannot represent is refused rather than wrapped.
        assert_eq!(Bits::new(&bytes).u64(65), None);

        // And the power-of-two scaling is exact: 2^-31 of 2^31 is one, with no rounding of its own.
        let mut bits = Bits::new(&[0x40, 0x00, 0x00, 0x00]);
        assert_eq!(bits.scaled(32, -31), Some(0.5));
    }

    #[test]
    fn the_width_guard_is_separate_from_the_buffer_guard() {
        // `u64` refuses for two independent reasons and the test above only reached one of them:
        // 65 bits out of two bytes fails on the buffer long before the width is looked at. With a
        // buffer that could serve the read, only the width guard is left standing.
        let wide = [0xFF_u8; 9];
        assert_eq!(Bits::new(&wide).u64(65), None);
        // And 64 is the widest field the reader accepts, not the first one it refuses.
        assert_eq!(Bits::new(&wide).u64(64), Some(u64::MAX));
    }

    #[test]
    fn the_radial_along_cross_basis_is_generic_and_not_a_permutation_of_the_axes() {
        // The hand-computed case above is axis-aligned, which is exactly the geometry on which the
        // basis degenerates into a permutation of the axes: a transposed term in the cross product
        // or a dropped factor in the normalisation moves nothing. This state is generic, and the
        // expected position is Maxima's at 40 digits rather than a second copy of the same algebra.
        let corrected = apply_orbit(
            [7.0e6, 2.0e6, 3.0e6],
            [-1.0e3, 6.0e3, 2.0e3],
            [1.0, 2.0, 3.0],
        )
        .expect("a non-degenerate state");
        let expected = [
            7_000_000.682_143_165,
            1_999_998.975_730_011_2,
            2_999_996.466_509_984,
        ];
        for (got, want) in corrected.iter().zip(expected) {
            // A micron, against corrections of 1 to 3 m: every mutation of the basis is metres.
            assert!((got - want).abs() < 1e-6, "{corrected:?} vs {expected:?}");
        }
    }

    #[test]
    fn the_scanner_reports_how_much_it_has_not_scanned() {
        let stream = ssr_frame(1057);
        let mut decoder = RtcmDecoder::new(&stream);
        assert_eq!(decoder.remaining(), stream.len());
        assert!(decoder.next().is_some());
        assert_eq!(decoder.remaining(), 0);
    }

    #[test]
    fn a_frame_past_255_bytes_needs_both_halves_of_the_length_field() {
        // DF002 is ten bits split across two header bytes, and every MSM7 observation frame -- the
        // bulk of a real stream's volume -- is longer than the low byte alone can express. A
        // fixture that never crosses 255 bytes never exercises the high two bits at all.
        let mut w = Writer::default();
        w.put(1077, 12); // MSM7, GPS
        for i in 0..300_u64 {
            w.put(i & 0xFF, 8);
        }
        let frame = w.frame();
        assert_eq!(frame.len(), 308, "3 header + 302 payload + 3 CRC");
        let framed: Vec<Frame> = RtcmDecoder::new(&frame).collect();
        assert_eq!(framed.len(), 1);
        assert_eq!(framed[0].payload.len(), 302);
        assert!(matches!(
            decode(framed[0].payload),
            Ok(Message::Unsupported(1077))
        ));
    }
}
