//! Bias-SINEX: the differential and observable-specific code biases that turn a pseudorange into a
//! range, and the last ingredient a precise-point-positioning client needs that this crate did not
//! carry.
//!
//! A GNSS satellite's clock is defined on one *ionosphere-free pair* of signals — for GPS, the
//! P1/P2 combination — and every other observable is offset from it. The receiver that tracks C1C
//! rather than C1W, or L5 rather than L2, is measuring against a different zero, and the difference
//! is metres. Bias-SINEX is how the analysis centres publish those offsets:
//!
//! * **DSB**, a differential signal bias: the offset *between* two observables, `OBS1 − OBS2`.
//! * **OSB**, an observable-specific bias: the offset of one observable from the clock's own
//!   reference, which is what modern uncombined PPP consumes.
//! * **ISB**, an inter-system bias, between constellations at a receiver.
//!
//! # Fixed columns, not whitespace
//!
//! The `BIAS/SOLUTION` block is column-defined, and it has to be parsed that way: the station field
//! is blank for a satellite bias and the second observable is blank for an OSB, so splitting on
//! whitespace shifts every later field on exactly the records that matter most. The column offsets
//! below come from the Bias-SINEX 1.00 format description and are asserted against a record with
//! both fields empty.
//!
//! # Units are kept, not assumed
//!
//! A bias is published in nanoseconds or in cycles. Nanoseconds convert to metres with the speed of
//! light and nothing else; cycles need the carrier frequency, which depends on the observable and,
//! for GLONASS, on the satellite's channel. [`Bias::metres`] therefore returns `Option` and declines
//! the cycle case rather than guessing a frequency — a silent wrong conversion here is a
//! decimetre-level error that looks like a real one.

use crate::rtcm::SPEED_OF_LIGHT;

/// Which kind of bias a record carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasKind {
    /// Differential signal bias: `OBS1 − OBS2`.
    Dsb,
    /// Observable-specific bias, relative to the clock's own reference.
    Osb,
    /// Inter-system bias.
    Isb,
}

/// The unit a bias was published in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiasUnit {
    /// Nanoseconds.
    Nanoseconds,
    /// Cycles of the observable's own carrier.
    Cycles,
}

/// An epoch in Bias-SINEX's `YYYY:DDD:SSSSS` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SinexEpoch {
    /// Four-digit year.
    pub year: u16,
    /// Day of year, 1-366.
    pub day_of_year: u16,
    /// Seconds of day.
    pub second_of_day: u32,
}

impl SinexEpoch {
    /// Parse `YYYY:DDD:SSSSS`.
    ///
    /// The all-zero field is the format's "open interval" marker and yields `None`, which is how a
    /// bias valid until further notice is written; the caller must treat that as unbounded rather
    /// than as year zero.
    #[must_use]
    pub fn parse(field: &str) -> Option<Self> {
        let mut parts = field.trim().split(':');
        let year: u16 = parts.next()?.trim().parse().ok()?;
        let day_of_year: u16 = parts.next()?.trim().parse().ok()?;
        let second_of_day: u32 = parts.next()?.trim().parse().ok()?;
        if year == 0 && day_of_year == 0 && second_of_day == 0 {
            return None;
        }
        Some(Self {
            year,
            day_of_year,
            second_of_day,
        })
    }
}

/// One `BIAS/SOLUTION` record.
#[derive(Debug, Clone, PartialEq)]
pub struct Bias {
    /// DSB, OSB or ISB.
    pub kind: BiasKind,
    /// Satellite identifier, e.g. `G01` — empty for a receiver-only bias.
    pub prn: String,
    /// Station identifier — empty for a satellite bias.
    pub station: String,
    /// First observable, in RINEX 3 notation, e.g. `C1C`.
    pub observable: String,
    /// Second observable; empty for an OSB, which has only one.
    pub second_observable: String,
    /// Start of validity, `None` when the record leaves it open.
    pub start: Option<SinexEpoch>,
    /// End of validity, `None` when the record leaves it open.
    pub end: Option<SinexEpoch>,
    /// The published value, in [`Bias::unit`].
    pub value: f64,
    /// Its standard deviation, in the same unit; `None` when the record omits one.
    pub sigma: Option<f64>,
    /// The unit the value is in.
    pub unit: BiasUnit,
}

impl Bias {
    /// The bias in metres of range.
    ///
    /// `None` for a cycle-valued bias, whose conversion needs the carrier frequency this record does
    /// not carry.
    #[must_use]
    pub fn metres(&self) -> Option<f64> {
        match self.unit {
            BiasUnit::Nanoseconds => Some(self.value * 1e-9 * SPEED_OF_LIGHT),
            BiasUnit::Cycles => None,
        }
    }
}

/// A parsed Bias-SINEX file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BiasSinex {
    /// The records, in file order.
    pub biases: Vec<Bias>,
}

/// Column spans of the `BIAS/SOLUTION` record, from the Bias-SINEX 1.00 description. They are
/// one-based in the document and zero-based here.
mod columns {
    pub const KIND: (usize, usize) = (1, 5);
    pub const PRN: (usize, usize) = (11, 14);
    pub const STATION: (usize, usize) = (15, 24);
    pub const OBS1: (usize, usize) = (25, 28);
    pub const OBS2: (usize, usize) = (30, 33);
    pub const START: (usize, usize) = (35, 49);
    pub const END: (usize, usize) = (50, 64);
    pub const UNIT: (usize, usize) = (65, 69);
    pub const VALUE: (usize, usize) = (70, 91);
    pub const SIGMA: (usize, usize) = (91, 103);
}

/// A record that is too short to hold the value column is not a record.
const MINIMUM_RECORD: usize = columns::VALUE.1;

fn field(line: &str, (start, end): (usize, usize)) -> &str {
    let bytes = line.as_bytes();
    let end = end.min(bytes.len());
    if start >= end {
        return "";
    }
    // Bias-SINEX is ASCII by definition; slicing by byte is slicing by column.
    line[start..end].trim()
}

impl BiasSinex {
    /// Parse a Bias-SINEX file.
    ///
    /// Records outside `+BIAS/SOLUTION` … `-BIAS/SOLUTION` are ignored, as are comment lines (`*`)
    /// and records whose kind or value this reader does not recognise. A file with one unreadable
    /// record among four thousand is still four thousand-odd usable biases; refusing the file would
    /// be the wrong trade, and [`BiasSinex::biases`] having fewer entries than the file has lines is
    /// visible to a caller that cares.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut biases = Vec::new();
        let mut inside = false;
        for line in text.lines() {
            let trimmed = line.trim_end_matches('\r');
            if trimmed.starts_with("+BIAS/SOLUTION") {
                inside = true;
                continue;
            }
            if trimmed.starts_with("-BIAS/SOLUTION") {
                inside = false;
                continue;
            }
            if !inside || trimmed.starts_with('*') || trimmed.len() < MINIMUM_RECORD {
                continue;
            }
            if let Some(bias) = parse_record(trimmed) {
                biases.push(bias);
            }
        }
        Self { biases }
    }

    /// The observable-specific bias for `prn` and `observable`, if the file carries one.
    #[must_use]
    pub fn osb(&self, prn: &str, observable: &str) -> Option<&Bias> {
        self.biases.iter().find(|b| {
            b.kind == BiasKind::Osb
                && b.prn == prn
                && b.observable == observable
                && b.station.is_empty()
        })
    }

    /// The differential bias for `prn` between two observables.
    ///
    /// A DSB is antisymmetric by construction, so a file that publishes `C1C−C2W` also answers a
    /// request for `C2W−C1C`: the value is returned negated rather than reported absent, because a
    /// caller that had to know the publication order would get it wrong half the time.
    #[must_use]
    pub fn dsb(&self, prn: &str, from: &str, to: &str) -> Option<f64> {
        self.biases.iter().find_map(|b| {
            if b.kind != BiasKind::Dsb || b.prn != prn || !b.station.is_empty() {
                return None;
            }
            if b.observable == from && b.second_observable == to {
                Some(b.value)
            } else if b.observable == to && b.second_observable == from {
                Some(-b.value)
            } else {
                None
            }
        })
    }
}

fn parse_record(line: &str) -> Option<Bias> {
    let kind = match field(line, columns::KIND) {
        "DSB" => BiasKind::Dsb,
        "OSB" => BiasKind::Osb,
        "ISB" => BiasKind::Isb,
        _ => return None,
    };
    let unit = match field(line, columns::UNIT) {
        "ns" => BiasUnit::Nanoseconds,
        "cyc" => BiasUnit::Cycles,
        _ => return None,
    };
    let value = field(line, columns::VALUE).parse().ok()?;
    Some(Bias {
        kind,
        prn: field(line, columns::PRN).to_owned(),
        station: field(line, columns::STATION).to_owned(),
        observable: field(line, columns::OBS1).to_owned(),
        second_observable: field(line, columns::OBS2).to_owned(),
        start: SinexEpoch::parse(field(line, columns::START)),
        end: SinexEpoch::parse(field(line, columns::END)),
        value,
        sigma: field(line, columns::SIGMA).parse().ok(),
        unit,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    /// A Bias-SINEX excerpt in the published column layout, with the two records that break a
    /// whitespace parse: an OSB whose second observable is blank, and satellite records whose
    /// station field is blank.
    const FILE: &str = "\
%=BIA 1.00 IGS 2021:032:00000 IGS 2021:001:00000 2021:002:00000 R 2021:001:00000 2021:002:00000 A 00000000
*-------------------------------------------------------------------------------
+FILE/REFERENCE
 DESCRIPTION       An excerpt
-FILE/REFERENCE
 DSB  G001 G01           C1C  C2W  2021:001:00000 2021:002:00000 ns                 -5.6789      0.0123
+BIAS/SOLUTION
*BIAS SVN_ PRN STATION__ OBS1 OBS2 BIAS_START____ BIAS_END______ UNIT __ESTIMATED_VALUE____ _STD_DEV___
 DSB  G001 G01           C1C  C2W  2021:001:00000 2021:002:00000 ns                 -5.6789      0.0123
 OSB  G001 G01           C1C       2021:001:00000 2021:002:00000 ns                  1.2345      0.0056
 OSB  G001 G01           C2W       2021:001:00000 2021:002:00000 ns                 -4.4444      0.0056
 OSB  R701 R01           C1C       0000:000:00000 0000:000:00000 cyc                 0.2500      0.0100
 DSB       G01 ALGO      C1C  C1W  2021:001:00000 2021:002:00000 ns                  0.9876      0.0200
 ISB           ALGO      C1C  C1C  2021:001:00000 2021:002:00000 ns                  3.3333      0.0300
 XXX  G001 G02           C1C  C2W  2021:001:00000 2021:002:00000 ns                  1.0000      0.0100
*DSB  G001 G03           C1C  C2W  2021:001:00000 2021:002:00000 ns                  9.9999      0.0100
-BIAS/SOLUTION
";

    fn parsed() -> BiasSinex {
        BiasSinex::parse(FILE)
    }

    #[test]
    fn only_the_records_inside_the_solution_block_are_read() {
        // The identical DSB line above the `+BIAS/SOLUTION` marker must not be counted, and the
        // unknown `XXX` kind must not either.
        let file = parsed();
        assert_eq!(file.biases.len(), 6, "{:?}", file.biases);

        // Nor the commented-out G03 record. A `*` in the first column is how a Bias-SINEX producer
        // withdraws a record without deleting it, and it leaves every other column in place -- so a
        // reader that only looked at the column layout would take it for a live 9.9999 ns bias.
        assert!(file.dsb("G03", "C1C", "C2W").is_none(), "{:?}", file.biases);

        // The third kind the format defines: an inter-system bias, carried by a station rather than
        // a satellite, which is why its PRN column is blank.
        let isb: Vec<_> = file
            .biases
            .iter()
            .filter(|b| b.kind == BiasKind::Isb)
            .collect();
        assert_eq!(isb.len(), 1, "{isb:?}");
        assert_eq!(isb[0].station, "ALGO");
        assert_eq!(isb[0].prn, "");
        assert!((isb[0].value - 3.3333).abs() < 1e-12, "{}", isb[0].value);
    }

    #[test]
    fn a_blank_station_and_a_blank_second_observable_do_not_shift_the_later_columns() {
        // The reason this parser is column-based. On a whitespace split, the OSB record's value
        // would come from the sigma column and its epochs would come from the wrong fields.
        let file = parsed();
        let osb = file.osb("G01", "C1C").expect("the C1C OSB");
        assert_eq!(osb.kind, BiasKind::Osb);
        assert_eq!(osb.station, "");
        assert_eq!(osb.second_observable, "");
        assert_eq!(osb.unit, BiasUnit::Nanoseconds);
        assert!((osb.value - 1.2345).abs() < 1e-12, "{}", osb.value);
        assert_eq!(osb.sigma, Some(0.0056));
        assert_eq!(
            osb.start,
            Some(SinexEpoch {
                year: 2021,
                day_of_year: 1,
                second_of_day: 0
            })
        );
        assert_eq!(
            osb.end,
            Some(SinexEpoch {
                year: 2021,
                day_of_year: 2,
                second_of_day: 0
            })
        );
    }

    #[test]
    fn a_station_bias_is_not_returned_for_a_satellite_query() {
        // The ALGO record is a receiver DSB for the same satellite and the same first observable. A
        // lookup that ignored the station field would return a receiver bias where the caller asked
        // for a satellite one, which is a metre-level error with no symptom.
        let file = parsed();
        assert!(file.dsb("G01", "C1C", "C1W").is_none());
        let station_records: Vec<_> = file
            .biases
            .iter()
            .filter(|b| !b.station.is_empty() && b.kind == BiasKind::Dsb)
            .collect();
        assert_eq!(station_records.len(), 1, "{station_records:?}");
        assert_eq!(station_records[0].station, "ALGO");
        assert_eq!(station_records[0].prn, "G01");
    }

    #[test]
    fn a_differential_bias_answers_in_both_directions_with_the_sign_reversed() {
        let file = parsed();
        let forward = file.dsb("G01", "C1C", "C2W").expect("the published order");
        let reverse = file.dsb("G01", "C2W", "C1C").expect("the reversed order");
        assert!((forward + 5.6789).abs() < 1e-12, "{forward}");
        assert!((reverse - 5.6789).abs() < 1e-12, "{reverse}");
        assert!((forward + reverse).abs() < 1e-12);
        assert!(file.dsb("G02", "C1C", "C2W").is_none());

        // Half a match is not a match. `C2W -> C5Q` shares one observable with the published
        // `C1C -> C2W` record and nothing else; answering it with the negated value would be a
        // bias between the wrong pair of signals, which no caller could detect downstream.
        assert!(file.dsb("G01", "C2W", "C5Q").is_none());
        assert!(file.dsb("G01", "C5Q", "C1C").is_none());
    }

    #[test]
    fn nanoseconds_convert_to_metres_and_cycles_decline_to() {
        // 1 ns is 0.299 792 458 m, and the C1C OSB's 1.2345 ns is 0.37 m — a real PPP-scale bias.
        let file = parsed();
        let osb = file.osb("G01", "C1C").expect("the C1C OSB");
        let metres = osb.metres().expect("nanoseconds convert");
        assert!(
            (metres - 1.2345e-9 * SPEED_OF_LIGHT).abs() < 1e-12,
            "{metres}"
        );
        assert!((metres - 0.370_093_789).abs() < 1e-9, "{metres}");

        // The GLONASS record is in cycles, whose metre value depends on a channel-dependent
        // frequency this file does not carry. Declining is the answer; guessing 1575.42 MHz would be
        // a 19 cm error that looks like a bias.
        let glonass = file.osb("R01", "C1C").expect("the GLONASS OSB");
        assert_eq!(glonass.unit, BiasUnit::Cycles);
        assert_eq!(glonass.metres(), None);
    }

    #[test]
    fn an_all_zero_epoch_is_an_open_interval_rather_than_the_year_zero() {
        let file = parsed();
        let glonass = file.osb("R01", "C1C").expect("the GLONASS OSB");
        assert_eq!(glonass.start, None);
        assert_eq!(glonass.end, None);

        assert_eq!(SinexEpoch::parse("0000:000:00000"), None);
        assert_eq!(
            SinexEpoch::parse("2021:365:86399"),
            Some(SinexEpoch {
                year: 2021,
                day_of_year: 365,
                second_of_day: 86_399
            })
        );
        assert_eq!(SinexEpoch::parse("not:an:epoch"), None);
        assert_eq!(SinexEpoch::parse("2021:001"), None);
        assert_eq!(SinexEpoch::parse(""), None);

        // The marker is all three fields zero together. Any one of them alone is an ordinary
        // value, and reading a partly-zero epoch as "open" would silently extend a bias's validity
        // past the interval its producer published it for.
        assert!(SinexEpoch::parse("2021:000:00000").is_some());
        assert!(SinexEpoch::parse("0000:001:00000").is_some());
        assert!(SinexEpoch::parse("0000:000:00001").is_some());
    }

    #[test]
    fn a_record_that_stops_at_the_value_column_is_still_a_record() {
        // The sigma column is optional, so the shortest complete record ends where the value does,
        // at column 91. That is the exact length the guard is written against: a guard that also
        // rejected the boundary would drop every file that publishes no standard deviations.
        const RECORD: &str = " OSB  G001 G02           C1C       2021:001:00000 2021:002:00000 ns                  7.7777";
        assert_eq!(
            RECORD.len(),
            91,
            "the shortest complete record is 91 columns"
        );
        let file = BiasSinex::parse(&format!("+BIAS/SOLUTION\n{RECORD}\n-BIAS/SOLUTION\n"));
        let osb = file
            .osb("G02", "C1C")
            .expect("a record with no sigma is still a record");
        assert!((osb.value - 7.7777).abs() < 1e-12, "{}", osb.value);
        assert_eq!(osb.sigma, None);

        // One column shorter and the guard is the only thing standing between the caller and a
        // wrong answer: the truncated value column still parses, as 7.777 rather than 7.7777.
        let short = BiasSinex::parse(&format!(
            "+BIAS/SOLUTION\n{}\n-BIAS/SOLUTION\n",
            &RECORD[..90]
        ));
        assert!(short.biases.is_empty(), "{:?}", short.biases);
    }

    #[test]
    fn a_truncated_or_empty_file_yields_no_records_rather_than_panicking() {
        // Column slicing past the end of a short line is the obvious way to panic here, and a
        // truncated download is the obvious way to get one.
        assert_eq!(BiasSinex::parse("").biases.len(), 0);
        assert_eq!(BiasSinex::parse("+BIAS/SOLUTION\n DSB\n").biases.len(), 0);
        for cut in 0..FILE.len() {
            let _ = BiasSinex::parse(&FILE[..FILE.floor_char_boundary(cut)]);
        }
    }
}
