//! The `BLQ` ocean loading table, as the loading providers write it.
//!
//! A `BLQ` file is a concatenation of per-station blocks. Each block is a station name on its own
//! line, then six lines of eleven numbers: radial, west and south amplitudes in metres, then the
//! three matching phase lags in degrees. Lines beginning `$$` are commentary — provenance, the
//! ocean model used, the station's coordinates — and carry nothing the model reads.
//!
//! The format is column-oriented in its Fortran definition (`1X, 11F7.5`), but every generator in
//! circulation separates the fields with spaces, and reading by whitespace accepts both that and
//! the columns. A record that does not produce eleven numbers is dropped rather than padded: a
//! half-read loading table is a station displaced by the wrong number of centimetres, which is
//! worse than a station with no table at all.

use crate::loading::{OceanLoading, SPECIFIED_CONSTITUENTS};

/// One station's block: the name as written, and the coefficients under it.
#[derive(Clone, Debug)]
pub struct LoadingSite {
    /// Station name, verbatim and with surrounding whitespace removed. Providers write the
    /// four-character IGS or ILRS identifier here, but nothing in the format requires it.
    pub name: String,
    /// The eleven-constituent response.
    pub loading: OceanLoading,
}

/// Parse every station block in a `BLQ` file.
///
/// Blocks that do not carry six complete rows of eleven numbers are skipped. A file whose header
/// mentions stations it does not tabulate — the providers emit one when a request fails partway —
/// therefore returns the stations it does carry rather than an error for the file as a whole.
#[must_use]
pub fn parse_blq(text: &str) -> Vec<LoadingSite> {
    let mut sites = Vec::new();
    let mut name: Option<&str> = None;
    let mut rows: Vec<[f64; SPECIFIED_CONSTITUENTS]> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("$$") {
            continue;
        }
        if let Some(values) = row(trimmed) {
            // A row before any name belongs to no station; dropping it is better than guessing
            // an owner for it.
            if name.is_some() {
                rows.push(values);
            }
        } else {
            name = Some(trimmed);
            rows.clear();
        }
        if rows.len() == 6 {
            if let Some(station) = name.take() {
                sites.push(LoadingSite {
                    name: station.to_string(),
                    loading: OceanLoading {
                        amplitude: [rows[0], rows[1], rows[2]],
                        phase: [rows[3], rows[4], rows[5]],
                    },
                });
            }
            rows.clear();
        }
    }
    sites
}

/// Eleven numbers, or nothing — a station name parses as no numbers and is how the two line kinds
/// are told apart.
fn row(line: &str) -> Option<[f64; SPECIFIED_CONSTITUENTS]> {
    let mut values = [0.0; SPECIFIED_CONSTITUENTS];
    let mut fields = line.split_whitespace();
    for value in &mut values {
        *value = fields.next()?.parse().ok()?;
    }
    if fields.next().is_some() {
        return None;
    }
    Some(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use spacelab_testkit::assert_within_ulps;

    /// The two stations from `HARDISP`'s own header, in the format the loading service emits:
    /// a header block of `$$` commentary, the station name, more commentary, then the six rows.
    const TWO_STATIONS: &str = "\
$$ Ocean loading displacement
$$ Green's function: gutenberg-bullen
  ONSALA
$$ CSR4.0_f_PP ID: 2009-06-25 17:43:24
$$ Onsala,                              lon/lat:   11.9264   57.3958    0.00
  .00352 .00123 .00080 .00032 .00187 .00112 .00063 .00003 .00082 .00044 .00037
  .00144 .00035 .00035 .00008 .00053 .00049 .00018 .00009 .00012 .00005 .00006
  .00086 .00023 .00023 .00006 .00029 .00028 .00010 .00007 .00004 .00002 .00001
   -64.7  -52.0  -96.2  -55.2  -58.8 -151.4  -65.6 -138.1    8.4    5.2    2.1
    85.5  114.5   56.5  113.6   99.4   19.1   94.1  -10.4 -167.4 -170.0 -177.7
   109.5  147.0   92.7  148.8   50.5  -55.1   36.4 -170.4  -15.0    2.3    5.2
  REYKJAVIK
$$ CSR4.0_f_PP ID: 2009-06-25 20:02:03
  .02359 .01481 .00448 .00419 .00273 .00033 .00088 .00005 .00081 .00034 .00034
  .00514 .00280 .00089 .00078 .00106 .00074 .00035 .00018 .00004 .00001 .00003
  .00209 .00077 .00051 .00021 .00151 .00066 .00047 .00019 .00014 .00008 .00006
    78.5  102.3   76.3  104.1  -52.0 -160.4  -52.6 -128.0 -174.8 -175.7 -178.1
    54.2   93.8   38.9   96.9    2.3  -12.5    3.1  -31.6 -144.4 -122.6 -173.5
   156.2 -167.1  141.9 -164.9  155.9  178.7  155.6 -168.5  177.7  178.7  179.7
$$ END TABLE
";

    #[test]
    fn both_stations_are_read_with_their_rows_in_the_published_order() {
        let sites = parse_blq(TWO_STATIONS);
        assert_eq!(sites.len(), 2, "two station blocks");
        assert_eq!(sites[0].name, "ONSALA");
        assert_eq!(sites[1].name, "REYKJAVIK");

        // First and last entry of every row, so a transposed or rotated block cannot pass.
        let onsala = &sites[0].loading;
        assert_within_ulps(
            onsala.amplitude[0][0],
            0.003_52,
            0,
            "Onsala radial M2 amplitude",
        );
        assert_within_ulps(
            onsala.amplitude[0][10],
            0.000_37,
            0,
            "Onsala radial Ssa amplitude",
        );
        assert_within_ulps(
            onsala.amplitude[2][0],
            0.000_86,
            0,
            "Onsala south M2 amplitude",
        );
        assert_within_ulps(onsala.phase[0][0], -64.7, 0, "Onsala radial M2 phase");
        assert_within_ulps(onsala.phase[2][10], 5.2, 0, "Onsala south Ssa phase");

        let reykjavik = &sites[1].loading;
        assert_within_ulps(
            reykjavik.amplitude[0][0],
            0.023_59,
            0,
            "Reykjavik radial M2 amplitude",
        );
        assert_within_ulps(reykjavik.phase[1][7], -31.6, 0, "Reykjavik west Q1 phase");
    }

    /// A block that stops short is not a station with nine zeroes in it.
    #[test]
    fn a_truncated_block_is_dropped_rather_than_zero_filled() {
        let truncated = "\
  ONSALA
  .00352 .00123 .00080 .00032 .00187 .00112 .00063 .00003 .00082 .00044 .00037
  .00144 .00035 .00035 .00008 .00053 .00049 .00018 .00009 .00012 .00005 .00006
";
        assert!(
            parse_blq(truncated).is_empty(),
            "three of six rows is not a station"
        );
    }

    /// A row of ten or twelve numbers is a different format, not this one.
    #[test]
    fn a_row_of_the_wrong_width_is_not_a_row() {
        let ten = "  .00352 .00123 .00080 .00032 .00187 .00112 .00063 .00003 .00082 .00044";
        let twelve = format!("{ten} .00037 .00040");
        assert!(row(ten).is_none(), "ten fields");
        assert!(row(&twelve).is_none(), "twelve fields");
        assert!(row(&format!("{ten} .00037")).is_some(), "eleven fields");
    }

    /// Rows arriving before any station name belong to nobody.
    #[test]
    fn rows_without_a_station_name_are_not_attributed_to_the_next_one() {
        let orphaned = "\
  .00352 .00123 .00080 .00032 .00187 .00112 .00063 .00003 .00082 .00044 .00037
  .00144 .00035 .00035 .00008 .00053 .00049 .00018 .00009 .00012 .00005 .00006
";
        assert!(
            parse_blq(orphaned).is_empty(),
            "rows with no name above them"
        );
    }

    /// The `$$` commentary carries the station's own coordinates, which parse as numbers; treating
    /// one as a data row would shift the whole block by a line.
    #[test]
    fn commentary_between_the_name_and_the_rows_is_not_read_as_a_row() {
        let sites = parse_blq(TWO_STATIONS);
        // Onsala's commentary line holds eleven-ish fields of longitude and latitude; if it had
        // been read, the radial amplitudes would start at 11.9264 rather than 0.00352.
        assert_within_ulps(
            sites[0].loading.amplitude[0][0],
            0.003_52,
            0,
            "radial M2 amplitude",
        );
    }
}
