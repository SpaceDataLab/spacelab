//! The ILRS Data Handling File: what the network says to do with a station's ranges.
//!
//! A normal point is a measurement, and the ILRS publishes, per station and per target, the
//! systematic errors its own analysis has found in that measurement. The file is a SINEX product
//! with three blocks this module reads:
//!
//! - `MODEL/RANGE_BIAS`, model code `R` — a range bias in millimetres that the network has
//!   determined and that an analysis is told to *apply*, not to solve for.
//! - `SOLUTION/DATA_HANDLING`, model code `E` — a station whose range bias is to be *estimated*.
//!   The file publishes no value for these: the record is the statement that a bias is there, and
//!   an analysis that cannot solve for one has no number to reach for.
//! - `SOLUTION/DATA_HANDLING`, model code `X` — observations to be deleted outright.
//!
//! The file's own words on the sign, which is the one thing that cannot be inferred from the
//! numbers: values are "to be subtracted from the 1-way observations".
//!
//! The pressure, humidity and timing records in the same block are deliberately not read. They
//! correct a different measurement than the range this crate models, and a parser that returned
//! them without a consumer that verifies them would be a claim rather than a capability.
//!
//! Source: `ILRS_Data_Handling_File_<date>.snx`, published by the ILRS Analysis Standing Committee.

/// ILRS target number, the `PT` column: 51 and 52 are LAGEOS-1 and LAGEOS-2.
///
/// Named rather than passed as a bare integer because the same column also carries `--`, which
/// means every target, and a caller that confuses the two applies a LAGEOS bias to an Etalon range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target(pub u8);

impl Target {
    /// LAGEOS-1.
    pub const LAGEOS_1: Self = Self(51);
    /// LAGEOS-2.
    pub const LAGEOS_2: Self = Self(52);
}

/// What the file says about one station's range bias.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RangeBias {
    /// Subtract this many millimetres from the one-way observation. The file gives the value and
    /// says not to estimate it.
    Apply(f64),
    /// The station carries a range bias that is to be solved for. The file gives no value, so
    /// there is nothing an analysis that does not estimate can do with this beyond reporting it.
    Estimate,
}

/// One record, reduced to the fields this module matches on.
#[derive(Clone, Copy, Debug)]
struct Record {
    pad: u32,
    /// `None` is the file's `--`: every target.
    target: Option<u8>,
    /// `None` is the file's `----`: every wavelength.
    wavelength: Option<u32>,
    start: f64,
    end: f64,
    bias: RangeBias,
}

/// The range-bias and exclusion recommendations, indexed by nothing: the file holds a few hundred
/// records and a linear scan over them costs less than the structure that would avoid it.
#[derive(Clone, Debug, Default)]
pub struct DataHandling {
    bias: Vec<Record>,
    excluded: Vec<Record>,
}

impl DataHandling {
    /// Read the `MODEL/RANGE_BIAS` and `SOLUTION/DATA_HANDLING` blocks.
    ///
    /// A record whose station, epochs or value do not parse is dropped. That is the safe direction:
    /// a missed correction leaves a known bias in the residual, where it is visible, while a
    /// misparsed one moves the answer by a number nobody wrote.
    #[must_use]
    pub fn parse(text: &str) -> Self {
        let mut out = Self::default();
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
            if block.is_empty() || line.starts_with('*') || line.trim().is_empty() {
                continue;
            }
            if !matches!(block, "MODEL/RANGE_BIAS" | "SOLUTION/DATA_HANDLING") {
                continue;
            }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 7 {
                continue;
            }
            let Some(record) = record(&f) else { continue };
            match f[6] {
                "R" | "E" => out.bias.push(record),
                "X" => out.excluded.push(record),
                _ => {}
            }
        }
        out
    }

    /// The range-bias recommendation in force for one station, target, wavelength and epoch.
    ///
    /// A published value wins over an instruction to estimate: the two only ever coincide when the
    /// network has since measured what an earlier release could only flag, and the value is the
    /// later statement.
    #[must_use]
    pub fn range_bias(
        &self,
        pad: u32,
        target: Target,
        wavelength_nm: f64,
        decimal_year: f64,
    ) -> Option<RangeBias> {
        let mut found = None;
        for r in self
            .bias
            .iter()
            .filter(|r| r.matches(pad, target, wavelength_nm, decimal_year))
        {
            match r.bias {
                RangeBias::Apply(_) => return Some(r.bias),
                RangeBias::Estimate => found = Some(RangeBias::Estimate),
            }
        }
        found
    }

    /// Whether the file marks these observations for deletion.
    #[must_use]
    pub fn is_excluded(
        &self,
        pad: u32,
        target: Target,
        wavelength_nm: f64,
        decimal_year: f64,
    ) -> bool {
        self.excluded
            .iter()
            .any(|r| r.matches(pad, target, wavelength_nm, decimal_year))
    }
}

impl Record {
    fn matches(&self, pad: u32, target: Target, wavelength_nm: f64, decimal_year: f64) -> bool {
        self.pad == pad
            && self.target.is_none_or(|t| t == target.0)
            && self
                .wavelength
                .is_none_or(|w| w == wavelength_code(wavelength_nm))
            && decimal_year >= self.start
            && decimal_year <= self.end
    }
}

/// `CODE PT SOLN T START END M [E-VALUE [STD_DEV [E-RATE]] UNIT [CMNTS]]`, whitespace-split.
fn record(f: &[&str]) -> Option<Record> {
    let value = if f[6] == "R" {
        // A range bias is only ever published in millimetres, and a record that says otherwise is
        // a format this parser has not seen rather than a number to reinterpret.
        if !f.iter().skip(7).any(|t| *t == "mm") {
            return None;
        }
        RangeBias::Apply(f.get(7)?.parse().ok()?)
    } else {
        RangeBias::Estimate
    };
    Some(Record {
        pad: f[0].parse().ok()?,
        target: (f[1] != "--").then(|| f[1].parse().ok()).flatten(),
        wavelength: (f[2] != "----").then(|| f[2].parse().ok()).flatten(),
        start: epoch(f[4], f64::NEG_INFINITY)?,
        end: epoch(f[5], f64::INFINITY)?,
        bias: value,
    })
}

/// A SINEX `YY:DDD:SSSSS` epoch as a decimal year, with the all-zero marker resolved in the
/// direction the field is read in.
///
/// This is not [`crate::sinex`]'s reader, and the difference is the marker. There it can only ever
/// close an interval, so mapping it to `+inf` is right; here it opens one just as often — a station
/// whose bias has always been present is written `00:000:00000 00:000:00000` — and mapping a start
/// to `+inf` would silently drop every such record.
fn epoch(field: &str, unbounded: f64) -> Option<f64> {
    let mut parts = field.split(':');
    let yy: i32 = parts.next()?.trim().parse().ok()?;
    // The column is `YY`, two digits. A wider number still parses, but it is not a year the pivot
    // below can place: `-100` comes out as 1900, a century year the two-digit form cannot express,
    // and the leap rule would then be evaluated on a date no SINEX file can carry.
    if !(0..=99).contains(&yy) {
        return None;
    }
    let doy: f64 = parts.next()?.trim().parse().ok()?;
    let seconds: f64 = parts.next()?.trim().parse().ok()?;
    if yy == 0 && doy == 0.0 && seconds == 0.0 {
        return Some(unbounded);
    }
    let year = if yy < 50 { 2000 + yy } else { 1900 + yy };
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = if leap { 366.0 } else { 365.0 };
    Some(f64::from(year) + (doy - 1.0 + seconds / 86_400.0) / days)
}

/// The `SOLN` column's wavelength code for a transmitted wavelength in nanometres.
///
/// The file discriminates a multi-colour station's records by wavelength and writes the
/// discriminator as the wavelength's hundreds, then `01`. The rule is not stated in the file; it is
/// read off the five codes the file actually carries against the systems that carry them —
/// Zimmerwald and Concepción at 423 nm are `401`, the network's green systems at 532 nm are `501`,
/// Matera SAO-1's ruby at 694 nm is `601`, Concepción's 847 nm is `801`, and Wettzell WLRS at
/// 1064 nm is `1001`.
#[must_use]
pub fn wavelength_code(nanometres: f64) -> u32 {
    #[expect(
        clippy::cast_sign_loss,
        clippy::cast_possible_truncation,
        reason = "a transmitted wavelength is positive and under 2000 nm"
    )]
    let hundreds = (nanometres / 100.0) as u32;
    hundreds * 100 + 1
}

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)] // Maxima references, kept at the width they were computed
    #![allow(clippy::expect_used)]

    use super::*;
    use spacelab_testkit::assert_within_ulps;

    /// Records taken verbatim from `ILRS_Data_Handling_File_20250513.snx`, one of each shape the
    /// two blocks produce: a published bias, a bias to be estimated, an every-target deletion, and
    /// a pressure record this module must not read as a range bias.
    const EXCERPT: &str = "\
+MODEL/RANGE_BIAS
*CODE PT SOLN T START_DATE__ END_DATE____ M __E-VALUE___ STD_DEV _E-RATE__ UNIT   CMNTS
 1890 51  501 A 12:001:00000 00:000:00000 R         11.4     1.1             mm
 8834 51 1001 A 19:153:00000 00:000:00000 R          2.4     0.3             mm
 7887 51  501 A 11:296:00000 15:249:00000 R         21.4     1.7             mm
-MODEL/RANGE_BIAS
+SOLUTION/DATA_HANDLING
 7249 --  501 A 00:000:00000 00:000:00000 E                                  mm
 1863 -- ---- A 00:000:00000 94:001:00000 X
 7080 -- ---- A 95:065:00000 96:025:86400 P      -2.10                       mB   BRE
-SOLUTION/DATA_HANDLING
";

    #[test]
    fn a_published_bias_is_read_with_its_target_wavelength_and_span() {
        let dhf = DataHandling::parse(EXCERPT);
        assert_eq!(
            dhf.range_bias(1890, Target::LAGEOS_1, 532.0, 2025.02),
            Some(RangeBias::Apply(11.4))
        );
        // The same pad, the other sphere: the file has no LAGEOS-2 record in this excerpt.
        assert_eq!(dhf.range_bias(1890, Target::LAGEOS_2, 532.0, 2025.02), None);
        // Before the record opens.
        assert_eq!(dhf.range_bias(1890, Target::LAGEOS_1, 532.0, 2011.0), None);
    }

    /// Wettzell WLRS ranges in green and in the infrared, and its published bias is for the
    /// infrared only. Matching a station without matching its wavelength applies it to both.
    #[test]
    fn a_wavelength_specific_bias_does_not_reach_the_other_colour() {
        let dhf = DataHandling::parse(EXCERPT);
        assert_eq!(
            dhf.range_bias(8834, Target::LAGEOS_1, 1064.0, 2025.02),
            Some(RangeBias::Apply(2.4))
        );
        assert_eq!(dhf.range_bias(8834, Target::LAGEOS_1, 532.0, 2025.02), None);
    }

    /// The `E` records carry no value, and a reader that filled one in with zero would report a
    /// station as unbiased that the network has flagged as biased.
    #[test]
    fn a_station_to_be_estimated_yields_no_number() {
        let dhf = DataHandling::parse(EXCERPT);
        assert_eq!(
            dhf.range_bias(7249, Target::LAGEOS_1, 532.0, 2025.02),
            Some(RangeBias::Estimate)
        );
        assert_eq!(
            dhf.range_bias(7249, Target::LAGEOS_2, 532.0, 2025.02),
            Some(RangeBias::Estimate),
            "the -- target column covers every satellite"
        );
    }

    /// An all-zero start opens the interval rather than closing it at infinity, which is where a
    /// reader borrowed from the station catalogue would drop the record.
    #[test]
    fn an_unbounded_start_reaches_back_rather_than_forward() {
        let dhf = DataHandling::parse(EXCERPT);
        assert_eq!(
            dhf.range_bias(7249, Target::LAGEOS_1, 532.0, 1995.0),
            Some(RangeBias::Estimate)
        );
    }

    #[test]
    fn deletions_are_kept_apart_from_biases() {
        let dhf = DataHandling::parse(EXCERPT);
        assert!(dhf.is_excluded(1863, Target::LAGEOS_1, 532.0, 1990.0));
        assert!(!dhf.is_excluded(1863, Target::LAGEOS_1, 532.0, 2025.0));
        assert_eq!(dhf.range_bias(1863, Target::LAGEOS_1, 532.0, 1990.0), None);
    }

    /// The pressure record in the same block has a number in the value column and millibars in the
    /// unit column. Reading it as a range bias would move a station by 2.1 mm of nothing.
    #[test]
    fn a_pressure_record_is_not_a_range_bias() {
        let dhf = DataHandling::parse(EXCERPT);
        assert_eq!(dhf.range_bias(7080, Target::LAGEOS_1, 532.0, 1995.3), None);
        assert!(!dhf.is_excluded(7080, Target::LAGEOS_1, 532.0, 1995.3));
    }

    /// A synthetic record: the file publishes range biases in millimetres only, so the unit column
    /// is what tells this parser that it is reading the format it was written for. Accepting a
    /// centimetre value as millimetres moves a station by ten times the bias, in the wrong
    /// direction of safe -- a misparsed correction is invisible in the residual, a missed one is not.
    #[test]
    fn a_range_bias_in_any_other_unit_is_a_format_this_parser_has_not_seen() {
        let text = "\
+MODEL/RANGE_BIAS
 7237 51  501 A 20:001:00000 00:000:00000 R          1.5     0.2             cm
-MODEL/RANGE_BIAS
";
        assert_eq!(
            DataHandling::parse(text).range_bias(7237, Target::LAGEOS_1, 532.0, 2025.0),
            None
        );
    }

    /// The decimal year the epoch column reduces to, against `derivations`-grade references
    /// computed in Maxima at 40 digits: `year + (doy - 1 + s/86400)/days`, with `days` from the
    /// Gregorian rule. Every term is pinned here rather than through `range_bias`, because a
    /// containment test only reports which side of an endpoint a date fell on and the leap rule
    /// moves an endpoint by a day in a year -- inside the interval either way.
    #[test]
    fn the_epoch_column_reduces_to_a_decimal_year_by_the_gregorian_rule() {
        let at = |f: &str| epoch(f, f64::NEG_INFINITY).expect("a well-formed epoch");

        // 2000 is the century year that *is* a leap year, and the only one a two-digit column can
        // reach: it separates the `% 400` clause from the `% 100` one.
        assert_within_ulps(
            at("00:100:43200"),
            2_000.271_857_923_497_267_759_6,
            0,
            "2000, day 100, noon",
        );
        // A common year, where the whole leap clause must come out false.
        assert_within_ulps(
            at("01:200:00000"),
            2_001.545_205_479_452_054_794_5,
            0,
            "2001, day 200",
        );
        // Day 366 exists only in a leap year, and reading it against 365 days lands on 1 January.
        assert_within_ulps(
            at("04:366:00000"),
            2_004.997_267_759_562_841_530_1,
            0,
            "2004, day 366",
        );
        // The Y2K pivot: 50 is the first two-digit year that means the twentieth century.
        assert_within_ulps(at("50:001:00000"), 1_950.0, 0, "the pivot, 1950 not 2050");
        assert_within_ulps(at("49:001:00000"), 2_049.0, 0, "the pivot, 2049 not 1949");

        // Year zero is a year -- the first assertion above reads one -- and only the whole field
        // being zero is the marker that opens the interval.
        assert_eq!(
            epoch("00:000:00000", f64::NEG_INFINITY),
            Some(f64::NEG_INFINITY)
        );

        // A `YY` wider than the column cannot be placed by the pivot, so it is not a date.
        assert_eq!(epoch("-100:001:00000", f64::NEG_INFINITY), None);
        assert_eq!(epoch("100:001:00000", f64::NEG_INFINITY), None);
    }

    /// The five codes the file carries, against the systems that carry them.
    #[test]
    fn the_wavelength_code_reproduces_every_discriminator_in_the_file() {
        assert_eq!(
            wavelength_code(423.0),
            401,
            "Zimmerwald and Concepcion blue"
        );
        assert_eq!(wavelength_code(532.0), 501, "the network's green");
        assert_eq!(wavelength_code(532.08), 501, "Herstmonceux, off by 80 pm");
        assert_eq!(wavelength_code(694.0), 601, "Matera SAO-1 ruby");
        assert_eq!(wavelength_code(847.0), 801, "Concepcion TIGO");
        assert_eq!(wavelength_code(1064.0), 1001, "Wettzell WLRS infrared");
    }
}
