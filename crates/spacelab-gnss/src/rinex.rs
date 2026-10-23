//! A minimal RINEX 3 navigation-message parser for GPS records — the analogue of Orekit's
//! `RinexNavigationParser` on its GPS path.
//!
//! A RINEX 3 GPS navigation record is eight lines: an epoch/clock line (`Gnn yyyy mm dd hh mm ss` then
//! three clock terms) followed by seven "broadcast orbit" lines of four `D19.12` fields each, laid out
//! at fixed columns (§A5 of the format spec). This reads those columns, converts the Fortran `D`
//! exponent to `E`, and returns the [`BroadcastEphemeris`] parameters. Records for other
//! constellations are skipped by their own block length (8 lines for GPS/Galileo/BeiDou/QZSS/NavIC, 4
//! for GLONASS/SBAS).

// Fixed-column RINEX fields are cast between float and small integer types; the letters match the
// broadcast-orbit notation; the doc allow is for spec proper nouns (RINEX, BeiDou, GLONASS).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names
)]

use crate::glonass::GlonassEphemeris;
use crate::gps::{BroadcastEphemeris, GnssSystem};

/// Kilometres → metres, for the GLONASS record's km / km·s⁻¹ / km·s⁻² fields.
const KM_TO_M: f64 = 1000.0;

/// Day of week (0 = Sunday) for a proleptic-Gregorian date, by Sakamoto's method — used to place the
/// epoch/clock line's calendar time into GPS seconds of week (`toc`), consistent with `toe`.
pub(crate) fn day_of_week(y: i32, m: u32, d: u32) -> u32 {
    const T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    (((y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d as i32) % 7 + 7) % 7) as u32
}

/// Parse one fixed-width RINEX float field (`E19.12`, Fortran `D`/`d` exponent accepted).
///
/// A field that is blank, or a line too short to reach it, is `0.0` — both are how the format writes
/// "not present", and every table in the spec has spare and conditional columns. A field that holds
/// something which is *not* a number is `None`, and the record carrying it is refused. That
/// distinction is not pedantry: this function used to answer `0.0` to both, and a column offset that
/// made every field straddle its neighbour was invisible for it — the malformed reads returned zero
/// and the rest returned a number ten times too small. See the module header.
pub(crate) fn parse_field(line: &str, start: usize, width: usize) -> Option<f64> {
    let Some(raw) = line.get(start..start + width).map(str::trim) else {
        return Some(0.0);
    };
    if raw.is_empty() {
        return Some(0.0);
    }
    raw.replace(['D', 'd'], "E").parse().ok()
}

/// The four `E19.12` fields of a broadcast-orbit line: `4X` then 4×19 columns, per Table A9 of the
/// format ("BROADCAST ORBIT - n ... 4X,4E19.12"). `None` if any of the four is not a number.
pub(crate) fn orbit_fields(line: &str) -> Option<[f64; 4]> {
    Some([
        parse_field(line, 4, 19)?,
        parse_field(line, 23, 19)?,
        parse_field(line, 42, 19)?,
        parse_field(line, 61, 19)?,
    ])
}

/// The number of lines a navigation record of the given constellation occupies.
fn block_len(system: char) -> Option<usize> {
    match system {
        'G' | 'E' | 'C' | 'J' | 'I' => Some(8),
        'R' | 'S' => Some(4),
        _ => None,
    }
}

/// The [`GnssSystem`] this parser supports for a RINEX system character (GPS/Galileo/BeiDou share the
/// 8-line Keplerian layout at identical columns).
fn system_of(c: char) -> Option<GnssSystem> {
    match c {
        'G' => Some(GnssSystem::Gps),
        'E' => Some(GnssSystem::Galileo),
        'C' => Some(GnssSystem::Beidou),
        'J' => Some(GnssSystem::Qzss),
        'I' => Some(GnssSystem::Navic),
        _ => None,
    }
}

/// Parse one 8-line Keplerian record (GPS/Galileo/BeiDou). Returns `None` if the PRN cannot be read.
pub(crate) fn parse_record(block: &[&str], system: GnssSystem) -> Option<BroadcastEphemeris> {
    let epoch = block[0];
    let prn: u8 = epoch.get(1..3)?.trim().parse().ok()?;

    let year: i32 = epoch.get(4..8)?.trim().parse().ok()?;
    let month: u32 = epoch.get(9..11)?.trim().parse().ok()?;
    let day: u32 = epoch.get(12..14)?.trim().parse().ok()?;
    let hour: u32 = epoch.get(15..17)?.trim().parse().unwrap_or(0);
    let minute: u32 = epoch.get(18..20)?.trim().parse().unwrap_or(0);
    let second: u32 = epoch.get(21..23)?.trim().parse().unwrap_or(0);
    let toc = f64::from(day_of_week(year, month, day) * 86400 + hour * 3600 + minute * 60 + second);

    let af0 = parse_field(epoch, 23, 19)?;
    let af1 = parse_field(epoch, 42, 19)?;
    let af2 = parse_field(epoch, 61, 19)?;

    let o1 = orbit_fields(block[1])?; // IODE, Crs, Δn, M0
    let o2 = orbit_fields(block[2])?; // Cuc, e, Cus, √A
    let o3 = orbit_fields(block[3])?; // Toe, Cic, Ω0, Cis
    let o4 = orbit_fields(block[4])?; // i0, Crc, ω, Ω̇
    let o5 = orbit_fields(block[5])?; // IDOT, codesL2, week, L2Pflag
    let o6 = orbit_fields(block[6])?; // SVacc, SVhealth, TGD, IODC

    Some(BroadcastEphemeris {
        system,
        prn,
        week: o5[2] as u16,
        toe: o3[0],
        toc,
        sqrt_a: o2[3],
        e: o2[1],
        i0: o4[0],
        omega0: o3[2],
        omega: o4[2],
        m0: o1[3],
        delta_n: o1[2],
        idot: o5[0],
        omega_dot: o4[3],
        cuc: o2[0],
        cus: o2[2],
        crc: o4[1],
        crs: o1[1],
        cic: o3[1],
        cis: o3[3],
        af0,
        af1,
        af2,
        tgd: o6[2],
        health: o6[1] as u8,
    })
}

/// Parse every GPS/Galileo/BeiDou navigation record from RINEX 3 `text`, in file order. Header lines
/// through `END OF HEADER` are skipped; records of other constellations are stepped over by their
/// block length (so a GLONASS block never desynchronises the GPS/Galileo/BeiDou records after it).
#[must_use]
pub fn parse_nav(text: &str) -> Vec<BroadcastEphemeris> {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains("END OF HEADER"))
        .map_or(0, |i| i + 1);

    let mut out = Vec::new();
    let mut idx = start;
    while idx < lines.len() {
        let Some(sys_char) = lines[idx].chars().next() else {
            idx += 1;
            continue;
        };
        let Some(len) = block_len(sys_char) else {
            idx += 1;
            continue;
        };
        if idx + len > lines.len() {
            break;
        }
        if let Some(system) = system_of(sys_char)
            && let Some(eph) = parse_record(&lines[idx..idx + len], system)
        {
            out.push(eph);
        }
        idx += len;
    }
    out
}

/// Parse only the GPS records from RINEX 3 `text` — [`parse_nav`] filtered to [`GnssSystem::Gps`].
#[must_use]
pub fn parse_gps_nav(text: &str) -> Vec<BroadcastEphemeris> {
    let mut out = parse_nav(text);
    out.retain(|e| e.system == GnssSystem::Gps);
    out
}

/// Parse one 4-line GLONASS ('R') record: the PZ-90 state in km / km·s⁻¹ / km·s⁻² is scaled to SI, the
/// reference epoch taken as seconds of day from the epoch line. Returns `None` if the slot is unreadable.
pub(crate) fn parse_glonass_record(block: &[&str]) -> Option<GlonassEphemeris> {
    let epoch = block[0];
    let slot: u8 = epoch.get(1..3)?.trim().parse().ok()?;
    let hour: u32 = epoch.get(15..17)?.trim().parse().unwrap_or(0);
    let minute: u32 = epoch.get(18..20)?.trim().parse().unwrap_or(0);
    let second: u32 = epoch.get(21..23)?.trim().parse().unwrap_or(0);
    let toe = f64::from(hour * 3600 + minute * 60 + second);

    let tau = parse_field(epoch, 23, 19)?; // −TauN, as stored
    let gamma = parse_field(epoch, 42, 19)?;

    let l1 = orbit_fields(block[1])?; // X,  X',  X'',  health
    let l2 = orbit_fields(block[2])?; // Y,  Y',  Y'',  frequency number
    let l3 = orbit_fields(block[3])?; // Z,  Z',  Z'',  age of information

    Some(GlonassEphemeris {
        slot,
        toe,
        position: [l1[0] * KM_TO_M, l2[0] * KM_TO_M, l3[0] * KM_TO_M],
        velocity: [l1[1] * KM_TO_M, l2[1] * KM_TO_M, l3[1] * KM_TO_M],
        luni_solar: [l1[2] * KM_TO_M, l2[2] * KM_TO_M, l3[2] * KM_TO_M],
        tau,
        gamma,
        freq_channel: l2[3] as i8,
        health: l1[3] as u8,
    })
}

/// Parse every GLONASS record from RINEX 3 `text`. Header lines through `END OF HEADER` are skipped;
/// records of other constellations are stepped over by their block length.
#[must_use]
pub fn parse_glonass_nav(text: &str) -> Vec<GlonassEphemeris> {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines
        .iter()
        .position(|l| l.contains("END OF HEADER"))
        .map_or(0, |i| i + 1);

    let mut out = Vec::new();
    let mut idx = start;
    while idx < lines.len() {
        let Some(sys_char) = lines[idx].chars().next() else {
            idx += 1;
            continue;
        };
        let Some(len) = block_len(sys_char) else {
            idx += 1;
            continue;
        };
        if idx + len > lines.len() {
            break;
        }
        if sys_char == 'R'
            && let Some(eph) = parse_glonass_record(&lines[idx..idx + len])
        {
            out.push(eph);
        }
        idx += len;
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// One `D19.12` field: sign/space, one leading digit, 12 fractional, `D`, signed 2-digit exponent.
    fn d19(x: f64) -> String {
        let s = format!("{x:.12E}");
        let (mantissa, exp) = s.split_once('E').unwrap();
        let exp: i32 = exp.parse().unwrap();
        let sign = if x < 0.0 { '-' } else { ' ' };
        let esign = if exp < 0 { '-' } else { '+' };
        format!(
            "{sign}{}D{esign}{:02}",
            mantissa.trim_start_matches('-'),
            exp.abs()
        )
    }

    /// One broadcast-orbit line. The indent is **four** spaces, which is `4X` in the format's own
    /// notation and is the whole reason this helper is worth a comment: it used to be three, matching
    /// what the parser read rather than what the format writes, and a fixture that agrees with the
    /// parser about the columns cannot ever disagree with it about a value.
    fn orbit_line(a: f64, b: f64, c: f64, d: f64) -> String {
        format!("    {}{}{}{}", d19(a), d19(b), d19(c), d19(d))
    }

    /// The eight lines of one Keplerian record, tagged `sys` (`G`/`E`/`C`/`J`/`I`) at PRN 7, with
    /// every field distinct so no two can be confused for one another.
    fn keplerian_record(sys: char) -> String {
        let epoch = format!(
            "{sys}07 2024 03 05 12 34 56{}{}{}",
            d19(1.5e-4),
            d19(3.4e-12),
            d19(0.0)
        );
        format!(
            "{epoch}\n{o1}\n{o2}\n{o3}\n{o4}\n{o5}\n{o6}\n{o7}",
            o1 = orbit_line(1.0, 35.0, 4.5e-9, 0.321),
            o2 = orbit_line(1.2e-6, 0.006_2, 8.3e-6, 5153.65),
            o3 = orbit_line(216_000.0, -1.1e-7, -1.234, 9.0e-8),
            o4 = orbit_line(0.959_9, 210.0, 0.567, -8.1e-9),
            o5 = orbit_line(-2.1e-10, 1.0, 2300.0, 0.0),
            o6 = orbit_line(0.0, 0.0, -1.0e-8, 40.0),
            o7 = orbit_line(216_000.0, 4.0, 0.0, 0.0),
        )
    }

    /// The four lines of one GLONASS record at slot 5.
    fn glonass_record() -> String {
        let epoch = format!(
            "R05 2024 01 09 03 15 42{}{}{}",
            d19(-1.5e-4),
            d19(0.0),
            d19(11_742.0)
        );
        format!(
            "{epoch}\n{l1}\n{l2}\n{l3}",
            l1 = orbit_line(25_510.0, 0.1, 1.0e-9, 0.0),
            l2 = orbit_line(-1_200.0, 2.9, -2.0e-9, -1.0),
            l3 = orbit_line(800.0, 1.3, 3.0e-9, 0.0),
        )
    }

    fn with_header(body: &str) -> String {
        format!(
            "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n\
             {}END OF HEADER\n{body}\n",
            " ".repeat(60),
        )
    }

    /// Build a RINEX 3 GPS record from a parameter set, then confirm the parser recovers it.
    ///
    /// A round trip through one writer and one reader **cannot** find a column error, and did not:
    /// this test was green while every field of a real file was read one column to the left. It
    /// checks that the reader is the writer's inverse, which is a real property and a weaker one
    /// than being the format's inverse. The file in `tests/rinex3_nav_igs_brdc.rs` is the second
    /// half of that, and only the two together say the reader can read RINEX.
    #[test]
    fn round_trips_a_spec_column_gps_record() {
        let eph = BroadcastEphemeris {
            system: GnssSystem::Gps,
            prn: 7,
            week: 2300,
            toe: 216_000.0,
            toc: 0.0, // recomputed from the epoch line
            sqrt_a: 5153.65,
            e: 0.006_2,
            i0: 0.959_9,
            omega0: -1.234,
            omega: 0.567,
            m0: 0.321,
            delta_n: 4.5e-9,
            idot: -2.1e-10,
            omega_dot: -8.1e-9,
            cuc: 1.2e-6,
            cus: 8.3e-6,
            crc: 210.0,
            crs: 35.0,
            cic: -1.1e-7,
            cis: 9.0e-8,
            af0: 1.5e-4,
            af1: 3.4e-12,
            af2: 0.0,
            tgd: -1.0e-8,
            health: 0,
        };

        // Epoch/clock line: "G07 " + calendar + three clock terms at columns 23/42/61. The minute
        // and second are non-zero so that neither can be dropped from `toc` without notice.
        let epoch = format!(
            "G07 2024 01 09 12 34 56{}{}{}",
            d19(eph.af0),
            d19(eph.af1),
            d19(eph.af2)
        );
        let text = format!(
            "     3.04           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE\n\
             {header}END OF HEADER\n{epoch}\n{o1}\n{o2}\n{o3}\n{o4}\n{o5}\n{o6}\n{o7}\n",
            header = " ".repeat(60),
            epoch = epoch,
            o1 = orbit_line(1.0, eph.crs, eph.delta_n, eph.m0),
            o2 = orbit_line(eph.cuc, eph.e, eph.cus, eph.sqrt_a),
            o3 = orbit_line(eph.toe, eph.cic, eph.omega0, eph.cis),
            o4 = orbit_line(eph.i0, eph.crc, eph.omega, eph.omega_dot),
            o5 = orbit_line(eph.idot, 1.0, f64::from(eph.week), 0.0),
            o6 = orbit_line(0.0, f64::from(eph.health), eph.tgd, 40.0),
            o7 = orbit_line(216_000.0, 4.0, 0.0, 0.0),
        );

        let parsed = parse_gps_nav(&text);
        assert_eq!(parsed.len(), 1, "one GPS record expected");
        let p = parsed[0];

        assert_eq!(p.system, GnssSystem::Gps);
        assert_eq!(p.prn, 7);
        assert_eq!(p.week, 2300);
        assert_eq!(p.health, 0);

        // The same record with a Galileo system tag parses as Galileo — the shared 8-line layout —
        // and is filtered out of the GPS-only view.
        let galileo_text = text.replacen("G07", "E07", 1);
        let gal = parse_nav(&galileo_text);
        assert_eq!(gal.len(), 1);
        assert_eq!(gal[0].system, GnssSystem::Galileo);
        assert_eq!(parse_gps_nav(&galileo_text).len(), 0);
        // 2024-01-09 is a Tuesday (dow 2) → 2·86400 + 12·3600 + 34·60 + 56 = 218096 s of week.
        assert!((p.toc - 218_096.0).abs() < 1e-6, "toc {}", p.toc);

        // Every float field survives the D19.12 round-trip.
        for (got, want, name) in [
            (p.toe, eph.toe, "toe"),
            (p.sqrt_a, eph.sqrt_a, "sqrt_a"),
            (p.e, eph.e, "e"),
            (p.i0, eph.i0, "i0"),
            (p.omega0, eph.omega0, "omega0"),
            (p.omega, eph.omega, "omega"),
            (p.m0, eph.m0, "m0"),
            (p.delta_n, eph.delta_n, "delta_n"),
            (p.idot, eph.idot, "idot"),
            (p.omega_dot, eph.omega_dot, "omega_dot"),
            (p.cuc, eph.cuc, "cuc"),
            (p.cus, eph.cus, "cus"),
            (p.crc, eph.crc, "crc"),
            (p.crs, eph.crs, "crs"),
            (p.cic, eph.cic, "cic"),
            (p.cis, eph.cis, "cis"),
            (p.af0, eph.af0, "af0"),
            (p.af1, eph.af1, "af1"),
            (p.tgd, eph.tgd, "tgd"),
        ] {
            let tol = want.abs().mul_add(1e-9, 1e-15);
            assert!((got - want).abs() < tol, "{name}: {got} vs {want}");
        }

        // End-to-end: the parsed ephemeris yields a medium-Earth-orbit radius.
        let pos = p.position_ecef(0.0);
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!((26.0e6..27.2e6).contains(&r), "radius {r}");
    }

    /// Sakamoto's method on dates that reach both of its month-dependent pieces.
    ///
    /// Every date here used to be in January, which is the one month that hides two thirds of the
    /// function: `T[0]` is `0`, so the month offset could be added, subtracted or dropped without
    /// changing anything, and `m < 3` is true for January whether the comparison is strict or not.
    /// March is the month that separates `<` from `<=` — it is the first month of Sakamoto's
    /// shifted year — and it has a non-zero offset, so one date settles both.
    #[test]
    fn day_of_week_is_correct_on_known_dates() {
        assert_eq!(day_of_week(2024, 1, 7), 0); // Sunday
        assert_eq!(day_of_week(2024, 1, 9), 2); // Tuesday
        assert_eq!(day_of_week(2000, 1, 1), 6); // Saturday
        assert_eq!(day_of_week(2024, 2, 29), 4); // Thursday — the last day of the shifted year
        assert_eq!(day_of_week(2024, 3, 5), 2); // Tuesday — T[2] = 2, and `m < 3` is false
        assert_eq!(day_of_week(2024, 10, 1), 2); // Tuesday — T[9] = 6, the largest offset
        assert_eq!(day_of_week(2024, 12, 31), 2); // Tuesday
        assert_eq!(day_of_week(2100, 3, 1), 1); // Monday — a century that is not a leap year
    }

    #[test]
    fn skips_non_gps_records_and_the_header() {
        // A GLONASS (4-line) record before a GPS (8-line) one must not be miscounted.
        let text = "\
                    x END OF HEADER\n\
                    R01 2024 01 09 12 00 00 0.000000000000D+00 0.000000000000D+00 0.000000000000D+00\n   \
                    a\n   b\n   c\n\
                    G07 2024 01 09 12 00 00 0.000000000000D+00 0.000000000000D+00 0.000000000000D+00\n\
                    ";
        // The GLONASS block eats lines R01+3; the G07 line is incomplete (no 7 orbit lines) → 0 GPS.
        let parsed = parse_gps_nav(text);
        assert_eq!(parsed.len(), 0);
    }

    #[test]
    fn round_trips_a_glonass_record_and_propagates() {
        // A near-circular PZ-90 state in km / km·s⁻¹, with a small luni-solar term in km·s⁻².
        let (x, y, z) = (25_510.0_f64, -1_200.0, 800.0); // km
        let (vx, vy, vz) = (0.1_f64, 2.9, 1.3); // km/s (~3.1 km/s total)
        let (ax, ay, az) = (1.0e-9_f64, -2.0e-9, 3.0e-9); // km/s²
        let epoch = format!(
            "R05 2024 01 09 03 15 42{}{}{}",
            d19(-1.5e-4),  // −TauN
            d19(0.0),      // +GammaN
            d19(11_742.0)  // message frame time
        );
        let text = format!(
            "     3.04           N: GNSS NAV DATA    R: GLONASS          RINEX VERSION / TYPE\n\
             {header}END OF HEADER\n{epoch}\n{l1}\n{l2}\n{l3}\n",
            header = " ".repeat(60),
            epoch = epoch,
            l1 = orbit_line(x, vx, ax, 0.0),  // health 0
            l2 = orbit_line(y, vy, ay, -1.0), // frequency channel −1
            l3 = orbit_line(z, vz, az, 0.0),
        );

        let parsed = parse_glonass_nav(&text);
        assert_eq!(parsed.len(), 1);
        let g = parsed[0];
        assert_eq!(g.slot, 5);
        assert_eq!(g.freq_channel, -1);
        assert_eq!(g.health, 0);
        // 03:15:42 → 3·3600 + 15·60 + 42 = 11742 s of day.
        assert!((g.toe - 11_742.0).abs() < 1e-6);
        // Every component of every vector is scaled km → m — not just the first one that happens
        // to be checked. Each of the nine has its own `* KM_TO_M`, and each is a separate operator.
        for (got, want, name) in [
            (g.position[0], 25_510_000.0_f64, "x"),
            (g.position[1], -1_200_000.0, "y"),
            (g.position[2], 800_000.0, "z"),
            (g.velocity[0], 100.0, "vx"),
            (g.velocity[1], 2_900.0, "vy"),
            (g.velocity[2], 1_300.0, "vz"),
            (g.luni_solar[0], 1.0e-6, "ax"),
            (g.luni_solar[1], -2.0e-6, "ay"),
            (g.luni_solar[2], 3.0e-6, "az"),
        ] {
            let tol: f64 = want.abs().mul_add(1e-12, 1e-15);
            assert!((got - want).abs() < tol, "{name}: {got} vs {want}");
        }
        // And it propagates to a plausible GLONASS radius.
        let (p, _) = g.position_velocity_pz90(300.0);
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        assert!((24.5e6..26.5e6).contains(&r), "r={r}");
    }

    /// Every constellation the 8-line Keplerian layout covers is tagged, not just the two that had
    /// a fixture. Deleting the `C`, `J` or `I` arm silently drops those records instead.
    #[test]
    fn every_keplerian_constellation_character_is_recognised() {
        for (sys, want) in [
            ('G', GnssSystem::Gps),
            ('E', GnssSystem::Galileo),
            ('C', GnssSystem::Beidou),
            ('J', GnssSystem::Qzss),
            ('I', GnssSystem::Navic),
        ] {
            let parsed = parse_nav(&with_header(&keplerian_record(sys)));
            assert_eq!(parsed.len(), 1, "{sys}: no record");
            assert_eq!(parsed[0].system, want, "{sys}");
            assert_eq!(parsed[0].prn, 7, "{sys}");
        }
        // 'S' has a block length but no Keplerian layout, so it is stepped over, not parsed.
        assert_eq!(parse_nav(&with_header(&keplerian_record('S'))).len(), 0);
    }

    /// The record walk: two records in a row, the header terminator itself, and lines it cannot use.
    ///
    /// Every previous fixture held exactly one record, ending at the last line of the file — so the
    /// short-file guard `idx + len > lines.len()` was always false at its only evaluation, the
    /// advance `idx += len` was never asked to reach a second record, and the two `idx += 1` arms
    /// for unusable lines were never reached at all (the one fixture that had junk lines fed them to
    /// a preceding block instead). A file that begins at line zero with an unusable line, carries
    /// two records and puts the header terminator on a line that itself looks like a record
    /// exercises all four in one pass.
    #[test]
    fn the_record_walk_advances_past_each_block_and_over_what_it_cannot_use() {
        // Two records, both returned, in file order.
        let two = with_header(&format!(
            "{}\n{}",
            keplerian_record('G'),
            keplerian_record('E')
        ));
        let parsed = parse_nav(&two);
        assert_eq!(parsed.len(), 2, "both records expected: {parsed:?}");
        assert_eq!(parsed[0].system, GnssSystem::Gps);
        assert_eq!(parsed[1].system, GnssSystem::Galileo);

        // Same for the GLONASS walk, which is a second copy of the same loop.
        let two_r = with_header(&format!("{}\n{}", glonass_record(), glonass_record()));
        assert_eq!(parse_glonass_nav(&two_r).len(), 2);

        // The header terminator is skipped even when the line it sits on would otherwise start a
        // record: parsing must resume *after* it, not on it. Here that line is a well-formed
        // GLONASS epoch line, so starting on it yields a record built from the wrong four lines.
        let sneaky = |body: &str| {
            format!(
                "     3.04           N: GNSS NAV DATA    M: MIXED            RINEX VERSION / TYPE\n\
                 R01 2024 01 09 12 00 00                              END OF HEADER\n{body}\n",
            )
        };
        assert_eq!(parse_nav(&sneaky(&keplerian_record('G'))).len(), 1);
        let r = parse_glonass_nav(&sneaky(&glonass_record()));
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].slot, 5, "the terminator line was parsed as a record");

        // No header at all, and the first line is one the walk cannot use — once blank, once an
        // unknown system. Both `idx += 1` arms then run with `idx` at zero, which is where failing
        // to advance loses the record that follows.
        for lead in ["\n", "X not a system\n"] {
            let gps = format!("{lead}{}\n", keplerian_record('G'));
            assert_eq!(parse_nav(&gps).len(), 1, "lead {lead:?}");
            let glo = format!("{lead}{}\n", glonass_record());
            assert_eq!(parse_glonass_nav(&glo).len(), 1, "lead {lead:?}");
        }
    }
}
