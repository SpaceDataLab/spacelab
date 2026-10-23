//! SINEX station coordinates — where the reference frame says a tracking station is.
//!
//! A terrestrial reference frame solution is published as a SINEX file: a `SOLUTION/ESTIMATE` block
//! of `STAX`/`STAY`/`STAZ` positions and `VELX`/`VELY`/`VELZ` velocities, each stamped with the
//! epoch the position refers to, and a `SOLUTION/EPOCHS` block saying over what interval each
//! solution is valid.
//!
//! The multi-solution structure is the part that is easy to get wrong. A station that was rebuilt,
//! or that sat on a fault that ruptured, appears several times with different `SOLN` numbers and
//! different coordinates, and picking the first match rather than the one covering the epoch is a
//! decimetre-to-metre error with no other symptom. [`StationCatalogue::position_at`] resolves the
//! interval, and falls back to the last solution — with velocity propagation — for an epoch beyond
//! the end of the frame's data span, which is the normal case for a frame used operationally.
//!
//! `SITE/ECCENTRICITY` is read from the same reader, because the ILRS publishes it in a separate
//! file and a position without it is not the point the instrument measures from. At three of the
//! twelve stations that tracked LAGEOS on 2025-01-06 the offset exceeds two and a half metres.
//!
//! Only the blocks needed to place a station are read. The covariance that follows them is most of
//! the file by volume and none of it by use here.
//!
//! Reference: *SINEX — Solution (Software/technique) INdependent EXchange Format*, version 2.02,
//! IERS/IGS.

// The format's own expansion capitalises the letters that make up the acronym.
#![allow(clippy::doc_markdown)]

use std::collections::BTreeMap;
use std::string::{String, ToString};
use std::vec::Vec;

/// One solution interval for one station: a position, a velocity and the span they describe.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StationSolution {
    /// Solution number within the station — SINEX `SOLN`.
    pub solution: u32,
    /// Position at [`reference_epoch`](Self::reference_epoch), metres, in the frame's own
    /// realisation.
    pub position: [f64; 3],
    /// Linear velocity, metres per year.
    pub velocity: [f64; 3],
    /// Decimal year the position refers to.
    pub reference_epoch: f64,
    /// Decimal year the solution's data span starts.
    pub start: f64,
    /// Decimal year the solution's data span ends.
    pub end: f64,
}

/// Which axes a `SITE/ECCENTRICITY` offset is expressed in — the block's `AXE` column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EccentricityAxis {
    /// Local up, north, east, metres. What the ILRS `ecc_une.snx` database uses.
    Une,
    /// Earth-fixed Cartesian dx, dy, dz, metres — the same offsets in `ecc_xyz.snx`.
    Xyz,
}

/// The offset from the point the reference frame solves for to the point the instrument measures
/// from, over the interval it applies — SINEX `SITE/ECCENTRICITY`.
///
/// For SLR this is the vector from the site's survey monument to the telescope's invariant point,
/// and at three of the twelve stations tracking LAGEOS on a given day it is over three metres —
/// vertical, almost entirely. A frame position used without it is not slightly wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eccentricity {
    /// Solution number within the site — SINEX `SBIN`.
    pub solution: u32,
    /// Decimal year the entry starts applying.
    pub start: f64,
    /// Decimal year it stops.
    pub end: f64,
    /// Which axes [`offset`](Self::offset) is in. The two ILRS databases publish the same
    /// eccentricities in either, so a consumer that assumes one silently mis-rotates the other.
    pub axis: EccentricityAxis,
    /// The offset itself, metres, in the order the block writes it: up/north/east for
    /// [`EccentricityAxis::Une`], dx/dy/dz for [`EccentricityAxis::Xyz`].
    pub offset: [f64; 3],
}

/// Every station in a SINEX file, keyed by the site code the file uses — for an SLR frame, the
/// four-digit CDP pad number as a string.
#[derive(Clone, Debug, Default)]
pub struct StationCatalogue {
    stations: BTreeMap<String, Vec<StationSolution>>,
    eccentricities: BTreeMap<String, Vec<Eccentricity>>,
}

impl StationCatalogue {
    /// Solutions held for a site code, in the order the file lists them.
    #[must_use]
    pub fn solutions(&self, code: &str) -> Option<&[StationSolution]> {
        self.stations.get(code).map(Vec::as_slice)
    }

    /// Number of distinct site codes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stations.len()
    }

    /// Whether the catalogue holds no stations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stations.is_empty()
    }

    /// Position of a station at a decimal year, metres, with the velocity applied from the
    /// solution's own reference epoch.
    ///
    /// The solution whose data span contains the epoch wins. Failing that — which is what happens
    /// whenever a frame is used after the data it was built from — the solution that ends latest is
    /// extrapolated forward, because that is the one still standing. A frame is not extrapolated
    /// *backwards* past its first solution: an epoch before any span returns the earliest solution
    /// propagated back, which is the same linear model and equally the caller's risk to take.
    #[must_use]
    pub fn position_at(&self, code: &str, decimal_year: f64) -> Option<[f64; 3]> {
        let solutions = self.stations.get(code)?;
        let covering = solutions
            .iter()
            .find(|s| decimal_year >= s.start && decimal_year <= s.end);
        let chosen = match covering {
            Some(s) => s,
            None => solutions
                .iter()
                .reduce(|a, b| if b.end > a.end { b } else { a })?,
        };
        let dt = decimal_year - chosen.reference_epoch;
        Some([
            chosen.position[0] + chosen.velocity[0] * dt,
            chosen.position[1] + chosen.velocity[1] * dt,
            chosen.position[2] + chosen.velocity[2] * dt,
        ])
    }

    /// Eccentricity in force at a decimal year, by the same interval rule as
    /// [`position_at`](Self::position_at).
    ///
    /// `None` means the file carries no entry for the site, which for the ILRS databases is the
    /// statement that the frame's point and the instrument's are the same one — a zero offset, not
    /// an unknown one.
    #[must_use]
    pub fn eccentricity_at(&self, code: &str, decimal_year: f64) -> Option<Eccentricity> {
        let entries = self.eccentricities.get(code)?;
        entries
            .iter()
            .find(|e| decimal_year >= e.start && decimal_year <= e.end)
            .or_else(|| {
                entries
                    .iter()
                    .reduce(|a, b| if b.end > a.end { b } else { a })
            })
            .copied()
    }
}

/// SINEX two-digit-year epochs, `YY:DDD:SSSSS`, as a decimal year.
///
/// The pivot is the one SINEX mandates: 00–49 is the twenty-first century, 50–99 the twentieth. The
/// all-zero epoch `00:000:00000` is the format's "unbounded" marker and becomes an infinity, so an
/// open-ended solution covers every epoch rather than none.
fn parse_epoch(field: &str) -> Option<f64> {
    let mut parts = field.split(':');
    let yy: i32 = parts.next()?.trim().parse().ok()?;
    let doy: f64 = parts.next()?.trim().parse().ok()?;
    let seconds: f64 = parts.next()?.trim().parse().ok()?;
    if yy == 0 && doy == 0.0 && seconds == 0.0 {
        return Some(f64::INFINITY);
    }
    let year = if yy < 50 { 2000 + yy } else { 1900 + yy };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = if leap { 366.0 } else { 365.0 };
    Some(f64::from(year) + (doy - 1.0 + seconds / 86_400.0) / days)
}

/// One whitespace-split `SITE/ECCENTRICITY` record: `CODE PT SBIN T START END AXE dU dN dE`.
///
/// `None` for a record whose axis keyword, solution number, epochs or offsets do not parse — a
/// malformed line is dropped rather than allowed to place a station wrongly.
fn parse_eccentricity(f: &[&str]) -> Option<Eccentricity> {
    let axis = match f[6] {
        "UNE" => EccentricityAxis::Une,
        "XYZ" => EccentricityAxis::Xyz,
        _ => return None,
    };
    Some(Eccentricity {
        solution: f[2].parse().ok()?,
        start: parse_epoch(f[4])?,
        end: parse_epoch(f[5])?,
        axis,
        offset: [f[7].parse().ok()?, f[8].parse().ok()?, f[9].parse().ok()?],
    })
}

/// Read the station positions, velocities and eccentricities out of a SINEX file.
///
/// Everything outside `SOLUTION/ESTIMATE`, `SOLUTION/EPOCHS` and `SITE/ECCENTRICITY` is ignored,
/// including the covariance blocks that make up most of the file. A station that has estimates but
/// no epoch record gets an unbounded span, which is what a single-solution site means.
#[must_use]
pub fn parse_sinex(text: &str) -> StationCatalogue {
    /// Position, velocity and reference epoch accumulated for one (site, solution) pair.
    #[derive(Clone, Copy, Default)]
    struct Partial {
        position: [f64; 3],
        velocity: [f64; 3],
        reference_epoch: f64,
        seen: u8,
    }

    let mut estimates: BTreeMap<(String, u32), Partial> = BTreeMap::new();
    let mut spans: BTreeMap<(String, u32), (f64, f64)> = BTreeMap::new();
    let mut eccentricities: BTreeMap<String, Vec<Eccentricity>> = BTreeMap::new();
    let mut block = "";

    for line in text.lines() {
        if let Some(name) = line.strip_prefix('+') {
            block = name.trim();
            continue;
        }
        if line.starts_with('-') {
            block = "";
            continue;
        }
        if line.starts_with('*') || line.starts_with('%') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();

        match block {
            "SOLUTION/ESTIMATE" if f.len() >= 9 => {
                let component = match f[1] {
                    "STAX" => 0,
                    "STAY" => 1,
                    "STAZ" => 2,
                    "VELX" => 3,
                    "VELY" => 4,
                    "VELZ" => 5,
                    _ => continue,
                };
                let (Ok(solution), Some(epoch), Ok(value)) =
                    (f[4].parse::<u32>(), parse_epoch(f[5]), f[8].parse::<f64>())
                else {
                    continue;
                };
                let entry = estimates
                    .entry((f[2].to_string(), solution))
                    .or_insert_with(|| Partial {
                        reference_epoch: epoch,
                        ..Partial::default()
                    });
                if component < 3 {
                    entry.position[component] = value;
                } else {
                    entry.velocity[component - 3] = value;
                }
                entry.seen |= 1 << component;
            }
            "SOLUTION/EPOCHS" if f.len() >= 6 => {
                let (Ok(solution), Some(start), Some(end)) =
                    (f[2].parse::<u32>(), parse_epoch(f[4]), parse_epoch(f[5]))
                else {
                    continue;
                };
                spans.insert((f[0].to_string(), solution), (start, end));
            }
            "SITE/ECCENTRICITY" if f.len() >= 10 => {
                if let Some(e) = parse_eccentricity(&f) {
                    eccentricities.entry(f[0].to_string()).or_default().push(e);
                }
            }
            _ => {}
        }
    }

    let mut stations: BTreeMap<String, Vec<StationSolution>> = BTreeMap::new();
    for ((code, solution), p) in estimates {
        // A site with fewer than three position components is a truncated record, not a station.
        if p.seen & 0b111 != 0b111 {
            continue;
        }
        let (start, end) = spans
            .get(&(code.clone(), solution))
            .copied()
            .unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
        stations.entry(code).or_default().push(StationSolution {
            solution,
            position: p.position,
            velocity: p.velocity,
            reference_epoch: p.reference_epoch,
            start,
            end,
        });
    }
    StationCatalogue {
        stations,
        eccentricities,
    }
}

#[cfg(test)]
mod tests {
    // Fixture parsing in tests: a malformed fixture should abort the test, not be handled.
    #![allow(clippy::expect_used)]

    use spacelab_testkit::assert_within_ulps;

    use super::*;

    /// Two stations from `ITRF2020-ILRS-TRF.SNX`, verbatim, one of them with the two solutions a
    /// rebuilt site carries.
    const SINEX: &str = "\
%=SNX 2.02 IGN 22:080:00000 IGN 79:329:00000 21:003:00000 C 01122 2 S
+SOLUTION/EPOCHS
*Code PT SOLN T Data_start__ Data_end____ Mean_epoch__
 7090  A    1 C 83:011:70773 21:002:61827 02:007:23100
 7110  A    1 C 82:362:51341 99:289:56376 91:143:53858
 7110  A    4 C 19:189:73912 20:366:14843 20:095:44378
-SOLUTION/EPOCHS
*-------------------------------------------------------------------------------
+SOLUTION/ESTIMATE
*INDEX TYPE__ CODE PT SOLN _REF_EPOCH__ UNIT S __ESTIMATED VALUE____ _STD_DEV___
     1 STAX   7090  A    1 15:001:00000 m    2 -.245009898814000E+07 0.11095E+00
     2 STAY   7090  A    1 15:001:00000 m    2 0.501601837771000E+07 0.19261E+00
     3 STAZ   7090  A    1 15:001:00000 m    2 -.307895670000000E+07 0.15348E+00
     4 VELX   7090  A    1 15:001:00000 m/y  2 -.470000000000000E-01 0.38655E-02
     5 VELY   7090  A    1 15:001:00000 m/y  2 0.100000000000000E-01 0.65747E-02
     6 VELZ   7090  A    1 15:001:00000 m/y  2 0.500000000000000E-01 0.65747E-02
     7 STAX   7110  A    1 15:001:00000 m    2 -.245000000000000E+07 0.11095E+00
     8 STAY   7110  A    1 15:001:00000 m    2 -.451000000000000E+07 0.19261E+00
     9 STAZ   7110  A    1 15:001:00000 m    2 0.345000000000000E+07 0.15348E+00
    10 STAX   7110  A    4 15:001:00000 m    2 -.245000100000000E+07 0.11095E+00
    11 STAY   7110  A    4 15:001:00000 m    2 -.451000200000000E+07 0.19261E+00
    12 STAZ   7110  A    4 15:001:00000 m    2 0.345000300000000E+07 0.15348E+00
-SOLUTION/ESTIMATE
";

    /// `parse_epoch` decides which century a two-digit year is in, whether that year is a leap
    /// year, and where in it the day-of-year and seconds fall. Every caller downstream compares the
    /// result against another epoch from the same function, so a shared error in any of those three
    /// cancels and nothing notices.
    #[test]
    fn the_two_digit_epoch_pivots_at_fifty_and_counts_leap_years() {
        let year = |field: &str| parse_epoch(field).expect("well-formed epoch");

        // The pivot is on `< 50`, so 49 is this century and 50 is the last one.
        assert_within_ulps(year("49:001:00000"), 2049.0, 0, "49 is 2049");
        assert_within_ulps(year("50:001:00000"), 1950.0, 0, "50 is 1950");

        // 2000 is divisible by 400 and so is a leap year, which is the only way the century rule is
        // reachable at all: a two-digit year can never land on 1900.
        assert_within_ulps(
            year("00:366:00000"),
            2000.0 + 365.0 / 366.0,
            0,
            "2000 has 366 days",
        );
        assert_within_ulps(
            year("99:366:00000"),
            1999.0 + 365.0 / 365.0,
            0,
            "1999 has 365",
        );
        assert_within_ulps(
            year("24:366:00000"),
            2024.0 + 365.0 / 366.0,
            0,
            "2024 has 366",
        );
        assert_within_ulps(
            year("23:366:00000"),
            2023.0 + 365.0 / 365.0,
            0,
            "2023 has 365",
        );

        // Day one at midnight is the start of the year exactly, and the seconds are a fraction of a
        // day within it — added, not subtracted, and to the year rather than away from it.
        assert_within_ulps(
            year("21:001:00000"),
            2021.0,
            0,
            "day one is the year itself",
        );
        assert_within_ulps(
            year("21:002:43200"),
            2021.0 + 1.5 / 365.0,
            0,
            "noon on day two is a day and a half in",
        );

        // The all-zero marker is unbounded; anything that differs in any one of its three fields is
        // an ordinary epoch.
        assert!(parse_epoch("00:000:00000").expect("marker").is_infinite());
        for near_miss in ["00:000:00001", "00:001:00000", "01:000:00000"] {
            assert!(
                parse_epoch(near_miss).expect("epoch").is_finite(),
                "{near_miss} is not the unbounded marker"
            );
        }
    }

    /// A catalogue with nothing in it says so, and one with something in it does not.
    #[test]
    fn an_empty_catalogue_reports_itself_empty() {
        assert!(parse_sinex("").is_empty());
        assert!(!parse_sinex(SINEX).is_empty());
    }

    /// Comment lines are ignored on the strength of their first character alone, even when what
    /// follows would parse. Real ITRF files retire a superseded solution by commenting it out, so a
    /// parser that reads through the marker adopts stations that were deliberately withdrawn.
    #[test]
    fn a_commented_record_is_not_read_as_a_record() {
        let with_comments = concat!(
            "+SOLUTION/EPOCHS\n",
            "*7500  A    1 C 90:001:00000 91:001:00000 90:180:00000\n",
            "-SOLUTION/EPOCHS\n",
            "+SOLUTION/ESTIMATE\n",
            "*1 STAX   7500  A    1 15:001:00000 m    2 0.100000000000000E+07 0.1E+00\n",
            "*2 STAY   7500  A    1 15:001:00000 m    2 0.200000000000000E+07 0.1E+00\n",
            "*3 STAZ   7500  A    1 15:001:00000 m    2 0.300000000000000E+07 0.1E+00\n",
            "-SOLUTION/ESTIMATE\n",
        );
        let c = parse_sinex(with_comments);
        assert!(c.is_empty(), "a commented-out solution was adopted");
    }

    /// A line too short for the fields its block requires is dropped, and the well-formed records
    /// around it still land. Without the length guard the very next statement indexes past the end.
    #[test]
    fn a_truncated_record_is_dropped_rather_than_indexed_past() {
        let ragged = concat!(
            "+SOLUTION/EPOCHS\n",
            " 7090  A    1 C 83:011:70773\n",
            " 7090  A    1 C 83:011:70773 21:002:61827 02:007:23100\n",
            "-SOLUTION/EPOCHS\n",
            "+SOLUTION/ESTIMATE\n",
            "     1 STAX   7090  A    1 15:001:00000 m\n",
            "     1 STAX   7090  A    1 15:001:00000 m    2 -.245009898814000E+07 0.11095E+00\n",
            "     2 STAY   7090  A    1 15:001:00000 m    2 0.501601837771000E+07 0.19261E+00\n",
            "     3 STAZ   7090  A    1 15:001:00000 m    2 -.307895670000000E+07 0.15348E+00\n",
            "-SOLUTION/ESTIMATE\n",
            "+SITE/ECCENTRICITY\n",
            " 7090  A    1 C 83:011:70773 21:002:61827 UNE\n",
            " 7090  A    1 C 83:011:70773 21:002:61827 UNE   0.0000   0.0000   0.0000\n",
            "-SITE/ECCENTRICITY\n",
        );
        let c = parse_sinex(ragged);
        assert_eq!(c.len(), 1, "the well-formed records did not survive");
        assert!(c.eccentricity_at("7090", 2000.0).is_some());
    }

    /// When no solution's span covers the epoch the catalogue falls back on the one that ends
    /// latest, and a tie between two goes to the earlier of them. Every fixture in this file has
    /// solutions whose spans end at different epochs, which leaves the comparison free to be `>=`.
    #[test]
    fn a_tie_on_the_latest_span_keeps_the_first_solution() {
        let tied = concat!(
            "+SOLUTION/EPOCHS\n",
            " 7500  A    1 C 90:001:00000 91:001:00000 90:180:00000\n",
            " 7500  A    2 C 92:001:00000 91:001:00000 92:180:00000\n",
            "-SOLUTION/EPOCHS\n",
            "+SOLUTION/ESTIMATE\n",
            "     1 STAX   7500  A    1 91:001:00000 m    2 0.100000000000000E+07 0.1E+00\n",
            "     2 STAY   7500  A    1 91:001:00000 m    2 0.000000000000000E+00 0.1E+00\n",
            "     3 STAZ   7500  A    1 91:001:00000 m    2 0.000000000000000E+00 0.1E+00\n",
            "     4 STAX   7500  A    2 91:001:00000 m    2 0.900000000000000E+07 0.1E+00\n",
            "     5 STAY   7500  A    2 91:001:00000 m    2 0.000000000000000E+00 0.1E+00\n",
            "     6 STAZ   7500  A    2 91:001:00000 m    2 0.000000000000000E+00 0.1E+00\n",
            "-SOLUTION/ESTIMATE\n",
            "+SITE/ECCENTRICITY\n",
            " 7500  A    1 C 90:001:00000 91:001:00000 UNE   1.0000   0.0000   0.0000\n",
            " 7500  A    2 C 92:001:00000 91:001:00000 UNE   9.0000   0.0000   0.0000\n",
            "-SITE/ECCENTRICITY\n",
        );
        let c = parse_sinex(tied);
        // 2020 is past both spans, so neither covers it and the tie-break decides.
        let p = c.position_at("7500", 2020.0).expect("the site is present");
        assert_within_ulps(p[0], 1.0e6, 0, "the tie went to the later solution");
        let e = c
            .eccentricity_at("7500", 2020.0)
            .expect("an entry is present");
        assert_within_ulps(
            e.offset[0],
            1.0,
            0,
            "the eccentricity tie went the other way",
        );
    }

    /// The eccentricity span is closed at both ends, and an epoch outside it does not match.
    ///
    /// The interval test is `start <= t && t <= end`, three separable claims: that the lower bound
    /// is inclusive, that the upper one is, and that both have to hold. The fixture below has two
    /// adjacent spans with different offsets, so each boundary picks a different answer.
    #[test]
    fn the_eccentricity_span_is_closed_at_both_ends() {
        let two_spans = concat!(
            "+SITE/ECCENTRICITY\n",
            " 7600  A    1 C 90:001:00000 95:001:00000 UNE   1.0000   0.0000   0.0000\n",
            " 7600  A    2 C 95:002:00000 99:001:00000 UNE   2.0000   0.0000   0.0000\n",
            "-SITE/ECCENTRICITY\n",
        );
        let c = parse_sinex(two_spans);
        let at = |t: f64| c.eccentricity_at("7600", t).expect("an entry").offset[0];

        let (first_start, first_end) = (1990.0, 1995.0 + 0.0 / 365.0);
        assert_within_ulps(at(first_start), 1.0, 0, "the lower bound is inclusive");
        assert_within_ulps(at(first_end), 1.0, 0, "the upper bound is inclusive");
        // Between the two spans nothing covers, so the fallback takes the later-ending entry.
        assert_within_ulps(at(1996.0), 2.0, 0, "inside the second span");
        // Before every span, the fallback still answers rather than returning nothing.
        assert_within_ulps(
            at(1980.0),
            2.0,
            0,
            "before every span the latest-ending wins",
        );
    }

    #[test]
    fn the_estimate_block_yields_positions_and_velocities() {
        let c = parse_sinex(SINEX);
        assert_eq!(c.len(), 2);
        let s = c.solutions("7090").expect("the site is present");
        assert_eq!(s.len(), 1);
        assert_within_ulps(s[0].position[0], -2_450_098.988_14, 0, "s[0].position[0]");
        assert_within_ulps(s[0].position[1], 5_016_018.377_71, 0, "s[0].position[1]");
        assert_within_ulps(s[0].position[2], -3_078_956.7, 0, "s[0].position[2]");
        assert_within_ulps(s[0].velocity[0], -0.047, 0, "s[0].velocity[0]");
        // 15:001:00000 is 2015 day 1 at 0 h, which is exactly the start of the year.
        assert_within_ulps(s[0].reference_epoch, 2015.0, 0, "s[0].reference_epoch");
    }

    /// The velocity is applied from the solution's own reference epoch, not from the span start.
    #[test]
    fn the_position_is_propagated_by_the_velocity() {
        let c = parse_sinex(SINEX);
        let p = c.position_at("7090", 2025.0).expect("the site is present");
        assert_within_ulps(p[0], -2_450_098.988_14 - 0.47, 2, "p[0]");
        assert_within_ulps(p[1], 5_016_018.377_71 + 0.10, 2, "p[1]");
        assert_within_ulps(p[2], -3_078_956.7 + 0.50, 2, "p[2]");
    }

    /// A site with a discontinuity must resolve to the solution covering the epoch. Taking the
    /// first would be metres wrong at a rebuilt station and would look like a range bias.
    #[test]
    fn a_multi_solution_site_resolves_by_epoch() {
        let c = parse_sinex(SINEX);
        let early = c
            .position_at("7110", 1990.0)
            .expect("covered by solution 1");
        let late = c
            .position_at("7110", 2020.0)
            .expect("covered by solution 4");
        assert_within_ulps(early[0], -2_450_000.0, 0, "early[0]");
        assert_within_ulps(late[0], -2_450_001.0, 0, "late[0]");
        assert_within_ulps(late[2], 3_450_003.0, 0, "late[2]");
    }

    /// An epoch past every span — the normal case for a frame in operational use — extrapolates
    /// from the solution that ends latest, not from the first in the file.
    #[test]
    fn an_epoch_beyond_the_frame_uses_the_last_standing_solution() {
        let c = parse_sinex(SINEX);
        let p = c.position_at("7110", 2025.0).expect("extrapolates");
        // Solution 4 has no velocity in this excerpt, so the position is its own.
        assert_within_ulps(p[0], -2_450_001.0, 0, "p[0]");
    }

    /// The eccentricity block, with the two axis conventions the ILRS publishes the same offsets in.
    ///
    /// Reading a `XYZ` entry as if it were `UNE` puts a three-metre vertical offset into the
    /// horizontal, which is why the axis is carried rather than assumed.
    #[test]
    fn the_eccentricity_block_carries_its_axis_convention() {
        const ECC: &str = "\
+SITE/ECCENTRICITY
*CODE PT SBIN T _DATA START_ __DATA_END__ AXE ARP->BENCHMARK(M)_________
 7090  A   13 L 92:203:00000 00:000:00000 UNE   3.1827  -0.0064   0.0194
 7110  A    1 L 82:362:51341 99:289:56376 XYZ   1.0000   2.0000   3.0000
-SITE/ECCENTRICITY
";
        let c = parse_sinex(ECC);
        let e = c.eccentricity_at("7090", 2025.0).expect("open-ended entry");
        assert_eq!(e.axis, EccentricityAxis::Une);
        assert_eq!(e.solution, 13);
        assert_within_ulps(e.offset[0], 3.1827, 0, "e.offset[0]");
        assert_within_ulps(e.offset[2], 0.0194, 0, "e.offset[2]");
        assert_eq!(
            c.eccentricity_at("7110", 1990.0).expect("covered").axis,
            EccentricityAxis::Xyz
        );
        // A site the file says nothing about is absent, not zero-by-default from another site.
        assert!(c.eccentricity_at("7119", 2025.0).is_none());
    }

    /// An unknown site is `None`, not a silent origin.
    #[test]
    fn an_unknown_site_is_absent() {
        let c = parse_sinex(SINEX);
        assert!(c.position_at("9999", 2025.0).is_none());
        assert!(c.solutions("9999").is_none());
    }

    /// The two-digit years pivot at 50, and the all-zero epoch means unbounded.
    #[test]
    fn the_two_digit_epochs_pivot_and_the_zero_epoch_is_unbounded() {
        assert_within_ulps(
            parse_epoch("15:001:00000").expect("parses"),
            2015.0,
            0,
            "parse_epoch('15:001:00000').expect('parses')",
        );
        let ninety_nine = parse_epoch("99:001:00000").expect("parses");
        assert_within_ulps(ninety_nine, 1999.0, 0, "ninety_nine");
        assert!(parse_epoch("00:000:00000").expect("parses").is_infinite());
        assert!(parse_epoch("not an epoch").is_none());
    }

    /// A truncated site — position components missing — is dropped rather than placed at a partly
    /// zero coordinate, which would put it inside the Earth and pass every plausibility check that
    /// only looks at one axis.
    #[test]
    fn a_site_missing_a_component_is_dropped() {
        let truncated = SINEX.replace("     9 STAZ   7110  A    1", "     9 STAQ   7110  A    1");
        let c = parse_sinex(&truncated);
        let s = c.solutions("7110").expect("solution 4 survives");
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].solution, 4);
    }
}
