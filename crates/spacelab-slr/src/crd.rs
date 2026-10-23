//! The ILRS Consolidated Laser Ranging Data format — reading a published normal point.
//!
//! A CRD file is a stream of records keyed by a leading token, grouped into *sessions* (one pass of
//! one satellite over one station) between an `h4` header and an `h8` terminator. What matters to
//! an observation model is in four of them: `h2` names the station, `h3` the target, `h4` says
//! **which corrections the publisher already applied** and what instant the time tag refers to,
//! `c0` gives the laser wavelength the refraction model needs, `20` carries the surface
//! meteorology, and `11` is the normal point itself.
//!
//! The `h4` flags are the part that cannot be assumed. An ILRS normal point is published with the
//! station's own system delay already removed but with **no** tropospheric correction and **no**
//! centre-of-mass correction, and a model that applies neither, or both twice, is out by metres and
//! by 25 cm respectively. They are parsed rather than hard-coded, so a file that breaks the habit
//! says so.
//!
//! Both revisions of the format are read. Version 1 and version 2 differ, for these records, only
//! in trailing fields — version 2 adds a network name to `h2`, a target class to `h3` and a
//! signal-to-noise column to `11` — so the parser reads by leading position and tolerates the extra
//! columns rather than demanding an exact width.
//!
//! Reference: Ricklefs & Moore, *Consolidated Laser Ranging Data Format (CRD)*, versions 1.01 and
//! 2.01, ILRS Data Formats and Procedures Working Group.

use std::string::{String, ToString};
use std::vec::Vec;

/// What instant a normal point's time tag refers to — `h4`'s epoch-event field, per record `11`.
///
/// This is the difference between reflecting at the tagged epoch and transmitting at it, which for
/// LAGEOS is 40 ms of satellite motion: about 240 m along track. It is read from the file, never
/// assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpochEvent {
    /// 0 — the pulse arrived back at the station at this epoch.
    GroundReceive,
    /// 1 — the pulse reflected off the spacecraft at this epoch.
    SpacecraftBounce,
    /// 2 — the pulse left the station at this epoch. What the ILRS network publishes.
    GroundTransmit,
    /// Anything else the format defines or a station writes; the caller decides whether it can use
    /// it rather than being handed a silent default.
    Other(u8),
}

impl EpochEvent {
    fn from_code(code: u8) -> Self {
        match code {
            0 => Self::GroundReceive,
            1 => Self::SpacecraftBounce,
            2 => Self::GroundTransmit,
            other => Self::Other(other),
        }
    }
}

/// One normal point — a compressed bin of raw returns, record `11`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalPoint {
    /// Seconds of day of the epoch event, in the station's time scale (UTC for the ILRS network).
    pub seconds_of_day: f64,
    /// Two-way time of flight, seconds, corrected for the station's own system delay.
    pub time_of_flight: f64,
    /// Length of the bin the point was formed over, seconds.
    pub window_length: f64,
    /// Raw ranges that went into the bin.
    pub raw_ranges: u32,
    /// Scatter of those raw ranges about the bin mean, picoseconds. 40 ps is 6 mm of one-way range,
    /// so this is the measurement's own precision statement.
    pub rms_picoseconds: f64,
}

/// Surface meteorology at the station, record `20`. The refraction model consumes all three.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meteorology {
    /// Seconds of day of the reading.
    pub seconds_of_day: f64,
    /// Surface pressure, hPa.
    pub pressure: f64,
    /// Surface temperature, K.
    pub temperature: f64,
    /// Relative humidity, percent.
    pub relative_humidity: f64,
}

/// The `h4` record's contiguous group of "correction already applied" indicators.
///
/// A range with a correction already removed must not have it removed again, so these decide what
/// a modelled range is allowed to contain. The ILRS network publishes ranges with the station's own
/// system delay taken out and nothing else, but the format permits any combination and a consumer
/// that assumes the common case will be metres wrong against a publisher that does not follow it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorrectionsApplied {
    /// Tropospheric refraction. The ILRS network does not apply it.
    pub troposphere: bool,
    /// The satellite centre-of-mass offset. The ILRS network does not apply it.
    pub centre_of_mass: bool,
    /// The station's own system delay. The ILRS network does apply it.
    pub system_delay: bool,
}

/// One pass of one satellite over one station, `h4` to `h8`.
#[derive(Clone, Debug)]
pub struct Session {
    /// Station four-character code from `h2`, e.g. `YARL`.
    pub station: String,
    /// Station pad identifier — the CDP number the reference frame catalogues the site under.
    pub pad_id: u32,
    /// Target name from `h3`, e.g. `lageos1`.
    pub target: String,
    /// ILRS satellite identifier, e.g. `7603901`.
    pub target_id: String,
    /// Calendar date of the start of the pass, UTC.
    pub year: i32,
    /// Month of the start of the pass, 1–12.
    pub month: u32,
    /// Day of the start of the pass, 1–31.
    pub day: u32,
    /// Transmitted wavelength, nanometres, from `c0`. 532 for most of the network.
    pub wavelength_nm: f64,
    /// What the normal-point time tags refer to.
    pub epoch_event: EpochEvent,
    /// Which corrections the publisher has already removed from the ranges.
    pub applied: CorrectionsApplied,
    /// Whether the measurement is two-way. A one-way (transponder) session needs a different model.
    pub two_way: bool,
    /// The publisher's own data-quality alert. Non-zero means the station flagged the pass.
    pub quality_alert: bool,
    /// Normal points in the pass, in file order.
    pub normal_points: Vec<NormalPoint>,
    /// Meteorological readings in the pass, in file order.
    pub meteorology: Vec<Meteorology>,
}

impl Session {
    /// The meteorological reading nearest in time to `seconds_of_day`.
    ///
    /// Stations log met records at their own cadence — sometimes one per normal point, sometimes
    /// one per pass — so this interpolates by nearest neighbour rather than assuming a pairing.
    /// `None` only when the pass carries no meteorology at all, in which case a refraction model
    /// has nothing to work from and the caller has to say so.
    #[must_use]
    pub fn meteorology_at(&self, seconds_of_day: f64) -> Option<Meteorology> {
        self.meteorology.iter().copied().reduce(|best, m| {
            if libm::fabs(m.seconds_of_day - seconds_of_day)
                < libm::fabs(best.seconds_of_day - seconds_of_day)
            {
                m
            } else {
                best
            }
        })
    }
}

fn number(field: &str) -> Option<f64> {
    field.parse::<f64>().ok()
}

/// Parse every session in a CRD file, of either format version.
///
/// Malformed records are skipped rather than rejected: an ILRS monthly file is the concatenation of
/// what forty stations submitted, and one station's bad line is not a reason to lose the other
/// thirty-nine passes. A session with no normal points is not returned at all.
#[must_use]
pub fn parse_crd(text: &str) -> Vec<Session> {
    let mut sessions = Vec::new();
    let mut station = String::new();
    let mut pad_id = 0_u32;
    let mut target = String::new();
    let mut target_id = String::new();
    let mut wavelength_nm = 532.0;
    let mut current: Option<Session> = None;

    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let Some(key) = f.first() else { continue };
        match *key {
            "h2" | "H2" if f.len() >= 3 => {
                station = f[1].to_string();
                pad_id = f[2].parse().unwrap_or(0);
            }
            "h3" | "H3" if f.len() >= 3 => {
                target = f[1].to_string();
                target_id = f[2].to_string();
            }
            "c0" | "C0" if f.len() >= 3 => {
                if let Some(w) = number(f[2]) {
                    wavelength_nm = w;
                }
            }
            "h4" | "H4" if f.len() >= 22 => {
                // A new session header closes the previous one whether or not an `h8` arrived. The
                // ILRS monthly files are concatenations of forty stations' submissions and about an
                // eighth of the sessions in them are unterminated; overwriting rather than flushing
                // here loses those passes without a word.
                push_if_populated(&mut sessions, current.take());
                // The wavelength is a per-session configuration record that may or may not precede
                // the header, so it is carried forward and re-read below if it follows.
                let flag = |i: usize| f[i] != "0";
                current = Some(Session {
                    station: station.clone(),
                    pad_id,
                    target: target.clone(),
                    target_id: target_id.clone(),
                    year: f[2].parse().unwrap_or(0),
                    month: f[3].parse().unwrap_or(0),
                    day: f[4].parse().unwrap_or(0),
                    wavelength_nm,
                    epoch_event: EpochEvent::GroundTransmit,
                    applied: CorrectionsApplied {
                        troposphere: flag(15),
                        centre_of_mass: flag(16),
                        system_delay: flag(18),
                    },
                    two_way: f[20] == "2",
                    quality_alert: flag(21),
                    normal_points: Vec::new(),
                    meteorology: Vec::new(),
                });
            }
            "11" if f.len() >= 13 => {
                let Some(s) = current.as_mut() else { continue };
                let (Some(sod), Some(tof), Some(window), Some(rms)) =
                    (number(f[1]), number(f[2]), number(f[5]), number(f[7]))
                else {
                    continue;
                };
                if let Ok(code) = f[4].parse::<u8>() {
                    s.epoch_event = EpochEvent::from_code(code);
                }
                s.wavelength_nm = wavelength_nm;
                s.normal_points.push(NormalPoint {
                    seconds_of_day: sod,
                    time_of_flight: tof,
                    window_length: window,
                    raw_ranges: f[6].parse().unwrap_or(0),
                    rms_picoseconds: rms,
                });
            }
            "20" if f.len() >= 5 => {
                let Some(s) = current.as_mut() else { continue };
                let (Some(sod), Some(pressure), Some(temperature), Some(humidity)) =
                    (number(f[1]), number(f[2]), number(f[3]), number(f[4]))
                else {
                    continue;
                };
                s.meteorology.push(Meteorology {
                    seconds_of_day: sod,
                    pressure,
                    temperature,
                    relative_humidity: humidity,
                });
            }
            "h8" | "H8" => push_if_populated(&mut sessions, current.take()),
            _ => {}
        }
    }
    push_if_populated(&mut sessions, current);
    sessions
}

/// A session with no normal points is a header and nothing else — a station reporting that it
/// tracked and got no returns. It is not an observation, so it does not become one.
fn push_if_populated(sessions: &mut Vec<Session>, session: Option<Session>) {
    if let Some(s) = session
        && !s.normal_points.is_empty()
    {
        sessions.push(s);
    }
}

#[cfg(test)]
mod tests {
    // Fixture parsing in tests: a malformed fixture should abort the test, not be handled.
    #![allow(clippy::expect_used)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    /// A version-2 session as the ILRS publishes it, trimmed to the records the parser reads.
    const V2: &str = "\
h1 CRD  2 2025  1  1  3
h2 HA4T       7119 14  2 3       ILRS
h3 lageos1     7603901 1155     8820 0 1 1
h4  1 2025  1  1  3  5 29 2025  1  1  3 10 21  0 0 0 0 1 0 2 0
c0 0  532.000 new la1 mcp ti1 swm met    cac
20 11144.007  710.80 281.50  38. 0
11 11144.006706800001     0.052749914678 new 2  120.0     16   39.0  -0.001  -1.218 na          2.67 0 na
20 11225.607  710.90 281.40  38. 0
11 11225.606845099999     0.051266264096 new 2  120.0     71   42.0  -0.212  -0.321 na         11.83 0 na
h8
";

    /// A record too short for the fields its key requires is dropped, and everything around it
    /// still lands. Without the length guard the very next statement indexes past the end of the
    /// split, so this is the difference between skipping a truncated line and aborting the parse.
    ///
    /// Real monthly files carry them: a submission interrupted mid-write leaves a partial record,
    /// and the forty stations concatenated into one file mean one bad line must not cost the rest.
    #[test]
    fn a_truncated_record_is_dropped_rather_than_indexed_past() {
        let ragged = concat!(
            "h2 HA4T\n",
            "h2 HA4T       7119 14  2 3       ILRS\n",
            "h3 lageos1\n",
            "h3 lageos1     7603901 1155     8820 0 1 1\n",
            "c0 0\n",
            "c0 0  423.000 new la1 mcp ti1 swm met    cac\n",
            "h4  1 2025  1  1  3  5 29 2025\n",
            "h4  1 2025  1  1  3  5 29 2025  1  1  3 10 21  0 0 0 0 1 0 2 0\n",
            "20 11144.007  710.80\n",
            "20 11144.007  710.80 281.50  38. 0\n",
            "11 11144.006706800001     0.052749914678 new 2\n",
            "11 11144.006706800001     0.052749914678 new 2  120.0     16   39.0  -0.001  -1.218 na          2.67 0 na\n",
            "h8\n",
        );
        let sessions = parse_crd(ragged);
        assert_eq!(sessions.len(), 1, "the well-formed records did not survive");
        let s = &sessions[0];
        assert_eq!(s.station, "HA4T");
        assert_eq!(s.target, "lageos1");
        assert_eq!(s.normal_points.len(), 1);
        assert_eq!(s.meteorology.len(), 1);
        // Not the 532 nm default: a c0 record that is never read leaves the default standing, and
        // asserting the default cannot tell that apart from having read the record.
        assert_within_ulps(s.wavelength_nm, 423.0, 0, "the c0 wavelength was not read");
    }

    /// `h8` closes the session, and records after it belong to nothing.
    ///
    /// Every fixture in this file ends at `h8` with nothing after, where the trailing flush after
    /// the loop produces the same answer whether the terminator was honoured or ignored. About an
    /// eighth of the sessions in a monthly file are unterminated, so the two paths both exist and
    /// have to be told apart: a stray record after a terminated session is a file-format error, not
    /// data to be attributed to the pass that just ended.
    #[test]
    fn a_record_after_the_terminator_joins_no_session() {
        let with_stray = concat!(
            "h2 HA4T       7119 14  2 3       ILRS\n",
            "h3 lageos1     7603901 1155     8820 0 1 1\n",
            "c0 0  532.000 new la1 mcp ti1 swm met    cac\n",
            "h4  1 2025  1  1  3  5 29 2025  1  1  3 10 21  0 0 0 0 1 0 2 0\n",
            "11 11144.006706800001     0.052749914678 new 2  120.0     16   39.0  -0.001  -1.218 na          2.67 0 na\n",
            "h8\n",
            "11 11225.606845099999     0.051266264096 new 2  120.0     71   42.0  -0.212  -0.321 na         11.83 0 na\n",
            "20 11225.607  710.90 281.40  38. 0\n",
        );
        let sessions = parse_crd(with_stray);
        assert_eq!(
            sessions.len(),
            1,
            "the stray records opened a second session"
        );
        assert_eq!(
            sessions[0].normal_points.len(),
            1,
            "a normal point after h8 was attributed to the closed session"
        );
        assert_eq!(sessions[0].meteorology.len(), 0);
    }

    /// The nearest meteorological record wins, and a tie goes to the earlier one.
    ///
    /// The published passes log met at a cadence that never lands equidistant between two readings,
    /// so the comparison in the fold is free to be `<=` and nothing sees it. It is not free: on a
    /// tie the two readings differ, and picking the later one silently changes the refraction.
    #[test]
    fn the_nearest_meteorology_breaks_a_tie_towards_the_earlier_reading() {
        let tied = concat!(
            "h2 HA4T       7119 14  2 3       ILRS\n",
            "h3 lageos1     7603901 1155     8820 0 1 1\n",
            "h4  1 2025  1  1  3  5 29 2025  1  1  3 10 21  0 0 0 0 1 0 2 0\n",
            "20 11100.000  700.00 280.00  30. 0\n",
            "20 11300.000  900.00 290.00  50. 0\n",
            // A pass with no normal points is dropped, so the session needs one to exist at all.
            "11 11200.000000000000     0.052749914678 new 2  120.0     16   39.0  -0.001  -1.218 na          2.67 0 na\n",
            "h8\n",
        );
        let sessions = parse_crd(tied);
        let s = &sessions[0];
        let exact = s.meteorology_at(11_100.0).expect("a reading is present");
        assert_within_ulps(
            exact.pressure,
            700.0,
            0,
            "an exact hit takes its own reading",
        );
        let midpoint = s.meteorology_at(11_200.0).expect("a reading is present");
        assert_within_ulps(
            midpoint.pressure,
            700.0,
            0,
            "the tie went to the later reading",
        );
        let later = s.meteorology_at(11_250.0).expect("a reading is present");
        assert_within_ulps(later.pressure, 900.0, 0, "nearest is the later reading");
    }

    /// The same pass in version 1, which is three columns narrower on `11` and one on `h2`.
    const V1: &str = "\
h1 CRD  1 2025  1  4  3
h2 BADL       1890 09 01  4
h3 lageos1     7603901 1155     8820 0 1
h4  1 2025  1  2  9 33 32 2025  1  2  9 44 49  0 0 0 0 1 0 2 0
c0 0 532.000 std las1
20 34412.345621879998 924.68 253.35 76 0
11 34430.839885349997 0.048243584495 std 2 120.0 176 88.1 0.560 0.165 -1.0 -1.0 0
h8
";

    #[test]
    fn a_version_2_session_parses_with_its_flags_and_meteorology() {
        let s = parse_crd(V2);
        assert_eq!(s.len(), 1);
        let s = &s[0];
        assert_eq!(s.station, "HA4T");
        assert_eq!(s.pad_id, 7119);
        assert_eq!(s.target, "lageos1");
        assert_eq!(s.target_id, "7603901");
        assert_eq!((s.year, s.month, s.day), (2025, 1, 1));
        assert_within_ulps(s.wavelength_nm, 532.0, 0, "s.wavelength_nm");
        assert_eq!(s.epoch_event, EpochEvent::GroundTransmit);
        assert!(!s.applied.troposphere, "the ILRS does not apply refraction");
        assert!(!s.applied.centre_of_mass, "nor the centre-of-mass offset");
        assert!(
            s.applied.system_delay,
            "but it does remove the system delay"
        );
        assert!(s.two_way);
        assert!(!s.quality_alert);
        assert_eq!(s.normal_points.len(), 2);
        assert_eq!(s.meteorology.len(), 2);
        assert_within_ulps(
            s.normal_points[0].time_of_flight,
            0.052_749_914_678,
            0,
            "s.normal_points[0].time_of_flight",
        );
        assert_eq!(s.normal_points[0].raw_ranges, 16);
        assert_within_ulps(
            s.normal_points[0].rms_picoseconds,
            39.0,
            0,
            "s.normal_points[0].rms_picoseconds",
        );
        assert_within_ulps(
            s.normal_points[0].window_length,
            120.0,
            0,
            "s.normal_points[0].window_length",
        );
    }

    /// The narrower version-1 record has to yield the same fields, which is the whole reason the
    /// parser reads by leading position rather than by exact width.
    #[test]
    fn a_version_1_session_parses_the_same_fields() {
        let s = parse_crd(V1);
        assert_eq!(s.len(), 1);
        let s = &s[0];
        assert_eq!(s.station, "BADL");
        assert_eq!(s.pad_id, 1890);
        assert_eq!(s.epoch_event, EpochEvent::GroundTransmit);
        assert_eq!(s.normal_points.len(), 1);
        assert_within_ulps(
            s.normal_points[0].time_of_flight,
            0.048_243_584_495,
            0,
            "s.normal_points[0].time_of_flight",
        );
        assert_within_ulps(
            s.meteorology[0].pressure,
            924.68,
            0,
            "s.meteorology[0].pressure",
        );
        assert_within_ulps(
            s.meteorology[0].temperature,
            253.35,
            0,
            "s.meteorology[0].temperature",
        );
        assert_within_ulps(
            s.meteorology[0].relative_humidity,
            76.0,
            0,
            "s.meteorology[0].relative_humidity",
        );
    }

    /// Two concatenated sessions from different stations must not bleed into each other — the
    /// failure mode of a parser that carries state across `h8`.
    #[test]
    fn concatenated_sessions_keep_their_own_stations() {
        let both = std::format!("{V2}{V1}");
        let s = parse_crd(&both);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].station, "HA4T");
        assert_eq!(s[1].station, "BADL");
        assert_eq!(s[0].normal_points.len(), 2);
        assert_eq!(s[1].normal_points.len(), 1);
    }

    /// A malformed normal point costs its own line and nothing else.
    #[test]
    fn a_bad_record_does_not_lose_the_pass() {
        let broken = V2.replace("0.051266264096", "not-a-number");
        let s = parse_crd(&broken);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].normal_points.len(), 1);
    }

    /// About an eighth of the sessions in an ILRS monthly file have no `h8`. A new `h4` has to
    /// close the one before it, or those passes vanish — which is a silent halving of the data set,
    /// not a parse error anyone would see.
    #[test]
    fn a_session_without_a_terminator_is_closed_by_the_next_header() {
        let unterminated = V2.replace("h8\n", "");
        let both = std::format!("{unterminated}{V1}");
        let s = parse_crd(&both);
        assert_eq!(s.len(), 2, "the unterminated pass was lost");
        assert_eq!(s[0].station, "HA4T");
        assert_eq!(s[0].normal_points.len(), 2);
        assert_eq!(s[1].station, "BADL");
    }

    /// A session that ends without normal points is not a session.
    #[test]
    fn an_empty_pass_is_dropped() {
        let empty = "h2 XXXX 1234 09 01 4\nh4  1 2025  1  2  9 33 32 2025  1  2  9 44 49  0 0 0 0 1 0 2 0\nh8\n";
        assert!(parse_crd(empty).is_empty());
    }

    /// The nearest met record is chosen by time, not by file order — a station logging one reading
    /// per pass and a station logging one per point have to work the same way.
    #[test]
    fn the_meteorology_is_taken_from_the_nearest_reading() {
        let s = &parse_crd(V2)[0];
        let nearest = |t: f64| s.meteorology_at(t).expect("a reading exists").pressure;
        assert_within_ulps(
            nearest(11_150.0),
            710.80,
            0,
            "pressure just after the first",
        );
        assert_within_ulps(
            nearest(11_300.0),
            710.90,
            0,
            "pressure just after the second",
        );
    }

    /// A flag column the publisher sets must be read, not assumed: a file that says the troposphere
    /// is already removed has to come back saying so, or the model would remove it twice.
    #[test]
    fn the_correction_flags_are_read_from_the_file() {
        let applied = V2.replace(" 0 0 0 0 1 0 2 0", " 0 1 1 0 1 0 2 1");
        let s = &parse_crd(&applied)[0];
        assert!(s.applied.troposphere);
        assert!(s.applied.centre_of_mass);
        assert!(s.quality_alert);
    }
}
