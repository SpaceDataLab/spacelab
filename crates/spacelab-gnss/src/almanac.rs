//! GPS almanac readers for the two formats the constellation is distributed in — SEM and Yuma — the
//! analogue of Orekit's `SEMParser` and `YUMAParser`.
//!
//! The almanac is not a smaller navigation message. It is a *different product*: a reduced Keplerian
//! set with no harmonic corrections, no `Δn`, no `i̇` and no group delay, valid for weeks rather than
//! hours, and it is what a receiver acquires with and what every visibility or geometry study takes
//! as input. [`crate::rinex`] reads the precise, short-validity broadcast record; this reads the
//! coarse, long-validity one. Planning a week of GNSS geometry from a navigation message is using the
//! wrong product, not merely the wrong file.
//!
//! Both formats carry the same nine numbers per satellite and disagree only on units and layout:
//! **SEM** is positional, in **semicircles**, and gives inclination as an *offset* from the 0.30
//! semicircle reference; **Yuma** is `key: value`, in **radians**, and gives inclination outright.
//! Reading a matched pair therefore checks each against the other to f64 rounding — see
//! `tests/gps_almanac.rs`.
//!
//! The reduced model is expressed as a [`BroadcastEphemeris`] with the terms the almanac does not
//! carry set to zero, so [`BroadcastEphemeris::position_velocity_ecef`] evaluates it unchanged. That
//! is exact rather than convenient: the IS-GPS-200 algorithm with `Cuc = Cus = Crc = Crs = Cic =
//! Cis = Δn = i̇ = 0` *is* the almanac algorithm of IS-GPS-200 §20.3.3.5.2.1.

// The GPS symbols are single letters by convention, and the format names are proper nouns.
#![allow(clippy::doc_markdown, clippy::many_single_char_names)]

use crate::gps::{BroadcastEphemeris, GnssSystem};

/// Semicircles → radians. SEM stores every angle in semicircles; Yuma stores them in radians.
const SEMICIRCLE: f64 = core::f64::consts::PI;

/// The reference inclination the SEM record's `δi` is an offset from, semicircles (IS-GPS-200: the
/// almanac transmits `δi` relative to 0.30 semicircles, never the inclination itself).
const REFERENCE_INCLINATION_SEMICIRCLES: f64 = 0.30;

/// The modulus of the almanac's week counter. Both formats truncate, so a file cannot say which
/// 1024-week era it belongs to — see [`resolve_week`].
const WEEK_MODULUS: i32 = 1024;

/// One satellite's almanac entry: the reduced orbit, plus the two identifiers only SEM carries.
#[derive(Debug, Clone, PartialEq)]
pub struct AlmanacEntry {
    /// The reduced Keplerian set, ready for [`BroadcastEphemeris::position_velocity_ecef`]. `toe`
    /// holds the almanac's time of applicability and `week` the resolved full week.
    pub ephemeris: BroadcastEphemeris,
    /// Space-vehicle number — the physical spacecraft, as opposed to the PRN it currently
    /// broadcasts. SEM only; `None` from a Yuma file, which does not carry it.
    pub svn: Option<u16>,
    /// SV configuration code (IS-GPS-200 Table 20-VII: block type and anti-spoof state). SEM only.
    pub config: Option<u8>,
}

/// Lift a truncated almanac week onto the continuous GPS week scale, choosing the era whose week is
/// nearest `hint`.
///
/// Neither format can resolve this itself: both store the week modulo 1024, so a file is consistent
/// with eras 19.6 years apart and the answer has to come from outside it. `hint` is any full GPS week
/// within 512 of the truth — the week the file was retrieved in will do. A `truncated` value exactly
/// 512 weeks from `hint` is genuinely ambiguous and resolves forward.
#[must_use]
pub fn resolve_week(truncated: u16, hint: u16) -> u16 {
    let offset = (i32::from(truncated) - i32::from(hint)).rem_euclid(WEEK_MODULUS);
    let signed = if offset > WEEK_MODULUS / 2 {
        offset - WEEK_MODULUS
    } else {
        offset
    };
    // Step back into range by whole cycles, never by clamping: the answer's only defining property
    // is that it is congruent to `truncated`, and `max(0)` would quietly return a week that is not.
    let mut week = i32::from(hint) + signed;
    if week < 0 {
        week += WEEK_MODULUS;
    }
    while week > i32::from(u16::MAX) {
        week -= WEEK_MODULUS;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    {
        week as u16
    }
}

/// The almanac's reduced model as a [`BroadcastEphemeris`]: every term the almanac does not transmit
/// is zero, which is what makes the broadcast algorithm reduce to the almanac algorithm.
#[allow(clippy::too_many_arguments)]
fn reduced(
    prn: u8,
    week: u16,
    toa: f64,
    e: f64,
    i0: f64,
    omega0: f64,
    omega: f64,
    m0: f64,
    omega_dot: f64,
    sqrt_a: f64,
    af0: f64,
    af1: f64,
    health: u8,
) -> BroadcastEphemeris {
    BroadcastEphemeris {
        system: GnssSystem::Gps,
        prn,
        week,
        toe: toa,
        // The almanac has one reference time, so the clock polynomial is referenced to it too.
        toc: toa,
        sqrt_a,
        e,
        i0,
        omega0,
        omega,
        m0,
        delta_n: 0.0,
        idot: 0.0,
        omega_dot,
        cuc: 0.0,
        cus: 0.0,
        crc: 0.0,
        crs: 0.0,
        cic: 0.0,
        cis: 0.0,
        af0,
        af1,
        // Not "zero acceleration measured": the almanac clock polynomial is first order and has no
        // third term at all. Same for the group delay, which is a navigation-message quantity.
        af2: 0.0,
        tgd: 0.0,
        health,
    }
}

/// The whitespace-separated numeric fields of one line, `None` if any of the first `n` will not parse.
fn numbers<const N: usize>(line: &str) -> Option<[f64; N]> {
    let mut out = [0.0; N];
    let mut fields = line.split_whitespace();
    for slot in &mut out {
        *slot = fields.next()?.parse().ok()?;
    }
    Some(out)
}

/// Parse a SEM almanac (`almanac.sem.week****.******.txt`).
///
/// Layout: a two-line header (`count title`, then `week toa`), then eight-line records separated by
/// blank lines — PRN, SVN, URA, three triples of orbit terms, health, configuration. `hint` resolves
/// the truncated week, as [`resolve_week`] describes. Records that will not parse are skipped rather
/// than failing the file: an almanac with one malformed satellite is still an almanac for the other
/// thirty-one, and a receiver treats it that way.
#[must_use]
pub fn parse_sem(text: &str, hint: u16) -> Vec<AlmanacEntry> {
    let lines: Vec<&str> = text.lines().map(str::trim_end).collect();
    let Some(header) = lines.iter().find(|l| !l.trim().is_empty()) else {
        return Vec::new();
    };
    let header_index = lines.iter().position(|l| l == header).unwrap_or(0);
    let Some([week, toa]) = lines.get(header_index + 1).and_then(|l| numbers::<2>(l)) else {
        return Vec::new();
    };
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let week = resolve_week(
        (week as i64).rem_euclid(i64::from(WEEK_MODULUS)) as u16,
        hint,
    );

    // Records are the non-blank runs after the header; taking them that way rather than by a fixed
    // stride means a stray blank line shifts nothing.
    let mut out = Vec::new();
    let mut idx = header_index + 2;
    while idx < lines.len() {
        if lines[idx].trim().is_empty() {
            idx += 1;
            continue;
        }
        let end = lines[idx..]
            .iter()
            .position(|l| l.trim().is_empty())
            .map_or(lines.len(), |n| idx + n);
        if end - idx >= 8
            && let Some(entry) = sem_record(&lines[idx..end], week, toa)
        {
            out.push(entry);
        }
        idx = end;
    }
    out
}

/// One eight-line SEM record. Angles arrive in semicircles and inclination as an offset from
/// [`REFERENCE_INCLINATION_SEMICIRCLES`], which is the difference from Yuma that a reader gets wrong
/// once and only once — the resulting orbit is inclined 54° instead of 55°, which looks plausible.
fn sem_record(block: &[&str], week: u16, toa: f64) -> Option<AlmanacEntry> {
    let prn: u8 = block[0].trim().parse().ok()?;
    let svn: u16 = block[1].trim().parse().ok()?;
    let [e, delta_i, omega_dot] = numbers::<3>(block[3])?;
    let [sqrt_a, omega0, omega] = numbers::<3>(block[4])?;
    let [m0, af0, af1] = numbers::<3>(block[5])?;
    let health: u8 = block[6].trim().parse().ok()?;
    let config: u8 = block[7].trim().parse().ok()?;

    Some(AlmanacEntry {
        ephemeris: reduced(
            prn,
            week,
            toa,
            e,
            (REFERENCE_INCLINATION_SEMICIRCLES + delta_i) * SEMICIRCLE,
            omega0 * SEMICIRCLE,
            omega * SEMICIRCLE,
            m0 * SEMICIRCLE,
            omega_dot * SEMICIRCLE,
            sqrt_a,
            af0,
            af1,
            health,
        ),
        svn: Some(svn),
        config: Some(config),
    })
}

/// The value after the first `:` on a line, parsed as a float.
fn after_colon(line: &str) -> Option<f64> {
    line.split_once(':')?.1.trim().parse().ok()
}

/// Parse a Yuma almanac (`almanac.yuma.week****.******.txt`).
///
/// Layout: `******** Week n almanac for PRN-nn ********` then thirteen `key: value` lines, all angles
/// already in radians. Each record carries its own week and time of applicability, so unlike SEM
/// there is no file-level header — a file whose records disagree is read as written rather than
/// reconciled. `hint` resolves each record's truncated week, as [`resolve_week`] describes.
#[must_use]
pub fn parse_yuma(text: &str, hint: u16) -> Vec<AlmanacEntry> {
    let mut out = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in text.lines().map(str::trim_end) {
        if line.trim_start().starts_with("****") {
            if let Some(entry) = yuma_record(&current, hint) {
                out.push(entry);
            }
            current.clear();
        } else if !line.trim().is_empty() {
            current.push(line);
        }
    }
    if let Some(entry) = yuma_record(&current, hint) {
        out.push(entry);
    }
    out
}

/// One Yuma record's `key: value` lines. Fields are found by their leading key rather than by
/// position: the labels carry unit annotations (`SQRT(A)  (m 1/2)`, `Rate of Right Ascen(r/s)`) that
/// have moved between producers, while the key prefixes have not.
fn yuma_record(block: &[&str], hint: u16) -> Option<AlmanacEntry> {
    let field = |key: &str| {
        block
            .iter()
            .find(|l| l.trim_start().starts_with(key))
            .and_then(|l| after_colon(l))
    };

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let prn = field("ID")? as u8;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let health = field("Health")? as u8;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let week = resolve_week(field("week")? as u16, hint);

    Some(AlmanacEntry {
        ephemeris: reduced(
            prn,
            week,
            field("Time of Applicability")?,
            field("Eccentricity")?,
            field("Orbital Inclination")?,
            field("Right Ascen at Week")?,
            field("Argument of Perigee")?,
            field("Mean Anom")?,
            field("Rate of Right Ascen")?,
            field("SQRT(A)")?,
            field("Af0")?,
            field("Af1")?,
            health,
        ),
        svn: None,
        config: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matched SEM/Yuma pair for GPS week 384 (truncated), time of applicability 147 456 s,
    /// retrieved from CelesTrak 2026-08-15. The same underlying scaled integers in both formats, so
    /// the pair pins the unit conversion in both directions.
    const SEM: &str = include_str!("../tests/fixtures/almanac.sem.week0384.147456.txt");
    const YUMA: &str = include_str!("../tests/fixtures/almanac.yuma.week0384.147456.txt");

    /// GPS week 2432 is the era the fixture was retrieved in; 2432 mod 1024 = 384.
    const HINT: u16 = 2432;

    #[test]
    fn a_truncated_week_resolves_to_the_era_nearest_the_hint() {
        assert_eq!(resolve_week(384, 2432), 2432);
        // Either side of the hint, within half a cycle.
        assert_eq!(resolve_week(383, 2432), 2431);
        assert_eq!(resolve_week(385, 2432), 2433);
        // Wrapping: a file from just before a rollover, read just after it.
        assert_eq!(resolve_week(0, 1023), 1024);
        // And the other way, where the nearest congruent week is negative and so cannot be the
        // answer — the era above it is. Every result must stay congruent to the truncated value.
        assert_eq!(resolve_week(1023, 2), 1023);
        for (truncated, hint) in [(1023_u16, 2_u16), (0, 1023), (384, 2432), (700, 100)] {
            let week = resolve_week(truncated, hint);
            assert_eq!(
                i32::from(week).rem_euclid(WEEK_MODULUS),
                i32::from(truncated),
                "resolve_week({truncated}, {hint}) = {week} is not congruent"
            );
        }
        // Exactly half a cycle away is ambiguous and documented to resolve forward.
        assert_eq!(resolve_week(384 + 512, 2432), 2432 + 512);
    }

    #[test]
    fn the_sem_header_gives_the_week_and_time_of_applicability_to_every_record() {
        let sem = parse_sem(SEM, HINT);
        assert_eq!(sem.len(), 32, "SEM record count");
        for entry in &sem {
            assert_eq!(entry.ephemeris.week, HINT);
            assert!((entry.ephemeris.toe - 147_456.0).abs() < 1e-9);
            assert_eq!(entry.ephemeris.system, GnssSystem::Gps);
        }
        // SEM carries the identifiers Yuma does not; PRN 1 was SVN 80 at this epoch.
        assert_eq!(sem[0].ephemeris.prn, 1);
        assert_eq!(sem[0].svn, Some(80));
        assert_eq!(sem[0].config, Some(12));
    }

    #[test]
    fn yuma_takes_its_week_from_each_record_and_carries_no_identifiers() {
        let yuma = parse_yuma(YUMA, HINT);
        assert_eq!(yuma.len(), 32, "Yuma record count");
        assert_eq!(yuma[0].ephemeris.prn, 1);
        assert_eq!(yuma[0].svn, None);
        assert_eq!(yuma[0].config, None);
        for entry in &yuma {
            assert_eq!(entry.ephemeris.week, HINT);
        }
    }

    /// The reduced model must actually *be* reduced: a non-zero harmonic term here would mean the
    /// almanac was read into the broadcast algorithm's correction slots, which no assertion on
    /// position would catch at the almanac's own accuracy.
    #[test]
    fn the_terms_the_almanac_does_not_carry_are_zero() {
        for entry in parse_sem(SEM, HINT) {
            let e = &entry.ephemeris;
            for (name, term) in [
                ("Cuc", e.cuc),
                ("Cus", e.cus),
                ("Crc", e.crc),
                ("Crs", e.crs),
                ("Cic", e.cic),
                ("Cis", e.cis),
                ("delta_n", e.delta_n),
                ("idot", e.idot),
                ("af2", e.af2),
                ("tgd", e.tgd),
            ] {
                assert!(term == 0.0, "PRN {}: {name} is {term:e}", e.prn);
            }
        }
    }

    /// The inclination reference is the trap the two formats disagree on: SEM transmits `δi` from
    /// 0.30 semicircles, Yuma the inclination itself. Reading `δi` as the inclination gives 0.83° —
    /// an equatorial orbit — and reading it without the reference gives 54.0° against the true 55°,
    /// which is close enough to look right. Pin the absolute value.
    #[test]
    fn the_sem_inclination_offset_is_lifted_onto_the_reference() {
        let sem = parse_sem(SEM, HINT);
        let i0 = sem[0].ephemeris.i0;
        assert!(
            (i0 - 0.957_014_660_9).abs() < 5e-10,
            "PRN 1 inclination {i0} rad"
        );
        // GPS is a 55° constellation; every healthy plane must land near it.
        for entry in &sem {
            let degrees = entry.ephemeris.i0.to_degrees();
            assert!(
                (52.0..58.0).contains(&degrees),
                "PRN {}: inclination {degrees}°",
                entry.ephemeris.prn
            );
        }
    }

    /// A file that is empty, headerless or truncated mid-record yields no entries rather than
    /// panicking or fabricating one — these are network products, and a partial download is the
    /// normal failure.
    #[test]
    fn a_malformed_file_yields_nothing_rather_than_a_wrong_orbit() {
        assert_eq!(parse_sem("", HINT).len(), 0);
        assert_eq!(parse_sem("32 CURRENT.ALM\n", HINT).len(), 0);
        assert_eq!(
            parse_sem("32 CURRENT.ALM\nnot a week\n\n1\n", HINT).len(),
            0
        );
        // A record cut short after the second orbit line.
        let truncated = "32 X\n 384 147456\n\n1\n80\n0\n 1e-3 4e-3 -2e-9\n 5.1e3 4e-2 5e-2\n";
        assert_eq!(parse_sem(truncated, HINT).len(), 0);
        assert_eq!(parse_yuma("", HINT).len(), 0);
        assert_eq!(
            parse_yuma(
                "******** Week 384 almanac for PRN-01 ********\nID: 01\n",
                HINT
            )
            .len(),
            0
        );
    }
}
