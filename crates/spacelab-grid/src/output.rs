//! Serialising a coverage map to portable text: CSV for a spreadsheet or `pandas`, GeoJSON for a
//! web map. Both are hand-written — the fields are an H3 index and a handful of numbers, so no
//! serialisation dependency earns its place.

// GeoJSON, MapLibre, deck.gl, pandas are product and format names in prose, not code identifiers.
#![allow(clippy::doc_markdown)]

use core::fmt::Write;

use h3o::CellIndex;

use crate::cell_center;

/// One cell's coverage over a scan: where it is, and how it was seen.
#[derive(Debug, Clone, PartialEq)]
pub struct CellCoverage {
    /// The canonical H3 index string (15 hex digits).
    pub h3_index: String,
    /// Cell-centre latitude, degrees.
    pub latitude_deg: f64,
    /// Cell-centre longitude, degrees.
    pub longitude_deg: f64,
    /// Number of access passes over the scan.
    pub passes: u32,
    /// Total time covered, seconds.
    pub coverage_seconds: f64,
    /// Worst-case revisit gap, seconds.
    pub max_gap_seconds: f64,
}

impl CellCoverage {
    /// Build a record for `cell`, filling its centre coordinates from the H3 index.
    #[must_use]
    pub fn new(cell: CellIndex, passes: u32, coverage_seconds: f64, max_gap_seconds: f64) -> Self {
        let g = cell_center(cell);
        Self {
            h3_index: cell.to_string(),
            latitude_deg: g.latitude.degrees(),
            longitude_deg: g.longitude.degrees(),
            passes,
            coverage_seconds,
            max_gap_seconds,
        }
    }
}

/// A coverage map as CSV, one row per cell, with a header.
#[must_use]
pub fn to_csv(records: &[CellCoverage]) -> String {
    let mut out = String::from(
        "h3_index,latitude_deg,longitude_deg,passes,coverage_seconds,max_gap_seconds\n",
    );
    for r in records {
        // The H3 index is hex digits only, so it never needs CSV quoting.
        let _ = writeln!(
            out,
            "{},{},{},{},{},{}",
            r.h3_index,
            r.latitude_deg,
            r.longitude_deg,
            r.passes,
            r.coverage_seconds,
            r.max_gap_seconds
        );
    }
    out
}

/// A coverage map as a GeoJSON `FeatureCollection` of cell-centre points, each carrying its coverage
/// as feature properties — ready to drop onto a `deck.gl` or MapLibre layer.
#[must_use]
pub fn to_geojson(records: &[CellCoverage]) -> String {
    let mut out = String::from("{\"type\":\"FeatureCollection\",\"features\":[");
    for (i, r) in records.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        // Coordinates are [lon, lat] by the GeoJSON spec. All property values are numbers or the
        // hex H3 index, so none needs JSON string escaping.
        let _ = write!(
            out,
            "{{\"type\":\"Feature\",\"geometry\":{{\"type\":\"Point\",\"coordinates\":[{},{}]}},\
             \"properties\":{{\"h3_index\":\"{}\",\"passes\":{},\"coverage_seconds\":{},\"max_gap_seconds\":{}}}}}",
            r.longitude_deg,
            r.latitude_deg,
            r.h3_index,
            r.passes,
            r.coverage_seconds,
            r.max_gap_seconds
        );
    }
    out.push_str("]}");
    out
}

/// A coverage map as a self-contained SVG heatmap: one disc per cell, placed by an equirectangular
/// projection of its longitude/latitude and coloured by revisit — green where the worst-case gap is
/// short, red where it is long. Vector, so it opens in any browser; no image-encoding dependency.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn to_svg(records: &[CellCoverage]) -> String {
    const W: f64 = 800.0;
    const H: f64 = 400.0;
    const M: f64 = 20.0;
    if records.is_empty() {
        return format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\"/>");
    }

    let (mut min_lon, mut max_lon) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_lat, mut max_lat) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_gap, mut max_gap) = (f64::INFINITY, f64::NEG_INFINITY);
    for r in records {
        min_lon = min_lon.min(r.longitude_deg);
        max_lon = max_lon.max(r.longitude_deg);
        min_lat = min_lat.min(r.latitude_deg);
        max_lat = max_lat.max(r.latitude_deg);
        min_gap = min_gap.min(r.max_gap_seconds);
        max_gap = max_gap.max(r.max_gap_seconds);
    }
    // Guard degenerate ranges (a single cell, or uniform coverage).
    let span_lon = (max_lon - min_lon).max(1e-6);
    let span_lat = (max_lat - min_lat).max(1e-6);
    let span_gap = (max_gap - min_gap).max(1e-6);

    let mut svg = format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{W}\" height=\"{H}\" \
         viewBox=\"0 0 {W} {H}\"><rect width=\"{W}\" height=\"{H}\" fill=\"#111\"/>"
    );
    for r in records {
        let x = M + (r.longitude_deg - min_lon) / span_lon * (W - 2.0 * M);
        let y = M + (max_lat - r.latitude_deg) / span_lat * (H - 2.0 * M); // north up
        let t = ((r.max_gap_seconds - min_gap) / span_gap).clamp(0.0, 1.0); // 0 best (green) → 1 worst (red)
        let (red, green) = ((255.0 * t) as u8, (255.0 * (1.0 - t)) as u8);
        let _ = write!(
            svg,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"6\" fill=\"rgb({red},{green},40)\"><title>{} passes, gap {:.0} s</title></circle>",
            r.passes, r.max_gap_seconds
        );
    }
    svg.push_str("</svg>");
    svg
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]

    use super::{CellCoverage, to_csv, to_geojson, to_svg};
    use crate::disk_cells;
    use h3o::Resolution;
    use spacelab_core::{Angle, Length};
    use spacelab_earth::Geodetic;

    fn sample() -> Vec<CellCoverage> {
        let center = Geodetic {
            latitude: Angle::from_degrees(35.0),
            longitude: Angle::from_degrees(33.0),
            height: Length::from_metres(0.0),
        };
        disk_cells(center, 1, Resolution::Five)
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(i, cell)| CellCoverage::new(cell, i as u32, 300.0 * i as f64, 3600.0))
            .collect()
    }

    #[test]
    fn csv_has_a_header_and_a_row_per_cell() {
        let records = sample();
        let csv = to_csv(&records);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "h3_index,latitude_deg,longitude_deg,passes,coverage_seconds,max_gap_seconds"
        );
        assert_eq!(
            lines.len(),
            records.len() + 1,
            "one header + one row per cell"
        );
        // the first data row starts with the canonical H3 index of the first cell
        assert!(
            lines[1].starts_with(&records[0].h3_index),
            "row must lead with the H3 index"
        );
    }

    #[test]
    fn svg_is_well_formed_with_a_disc_per_cell() {
        let records = sample();
        let svg = to_svg(&records);
        assert!(svg.starts_with("<svg xmlns=\"http://www.w3.org/2000/svg\""));
        assert!(svg.ends_with("</svg>"));
        assert_eq!(
            svg.matches("<circle").count(),
            records.len(),
            "one disc per cell"
        );
        // an empty map is still valid SVG
        assert!(to_svg(&[]).starts_with("<svg"));
    }

    #[test]
    fn geojson_is_a_feature_collection_with_lon_lat_order() {
        let records = sample();
        let json = to_geojson(&records);
        assert!(json.starts_with("{\"type\":\"FeatureCollection\",\"features\":["));
        assert!(json.ends_with("]}"));
        assert_eq!(
            json.matches("\"type\":\"Feature\"").count(),
            records.len(),
            "one feature per cell"
        );
        // coordinates are [lon, lat] by the GeoJSON spec — the record's own pair, in that order
        let pair = format!("[{},{}]", records[0].longitude_deg, records[0].latitude_deg);
        assert!(
            json.contains(&pair),
            "coordinates must be [lon, lat]: expected {pair}"
        );

        // The separator sits *between* features and nowhere else. Counting features cannot see
        // this: a comma before the first or after the last leaves the count unchanged and the
        // collection unparseable, and `starts_with` on the prefix passes either way because the
        // prefix ends at the bracket.
        assert!(
            json.starts_with("{\"type\":\"FeatureCollection\",\"features\":[{\"type\":\"Feature\""),
            "a comma before the first feature: {}",
            &json[..70.min(json.len())]
        );
        assert_eq!(
            json.matches("},{\"type\":\"Feature\"").count(),
            records.len() - 1,
            "one separator between each pair of features and no more"
        );
    }

    /// `svg_is_well_formed_with_a_disc_per_cell` counts discs and checks the envelope; it never
    /// looks at where a disc *is* or what colour it is. That leaves the whole projection free —
    /// 40 of this module's survivors are the two coordinate expressions and the colour ramp.
    /// `sample()` could not have caught the ramp in any case: every cell there carries the same
    /// 3600 s gap, so `span_gap` collapses to its 1e-6 floor and `t` is 0 everywhere.
    ///
    /// Three records pinned to the corners of their own bounding box make the expected output a
    /// geometric statement rather than a re-derivation: the north-west cell belongs at the top-left
    /// of the inset drawing area, the south-east at the bottom-right, and a cell at the centre of
    /// the box at the centre of the canvas. Every value below is exact in binary floating point.
    #[test]
    fn svg_places_each_cell_by_its_position_and_colours_it_by_its_gap() {
        let cell = |longitude_deg: f64, latitude_deg: f64, max_gap_seconds: f64| CellCoverage {
            h3_index: "8a1fb46622dffff".to_owned(),
            latitude_deg,
            longitude_deg,
            passes: 1,
            coverage_seconds: 0.0,
            max_gap_seconds,
        };
        // Bounding box 60° wide and 120° tall; gaps spanning 100..900 s.
        let records = vec![
            cell(-30.0, 60.0, 100.0),
            cell(30.0, -60.0, 900.0),
            cell(0.0, 0.0, 500.0),
        ];
        let svg = to_svg(&records);
        println!("{svg}");

        // W = 800, H = 400, M = 20, so the drawable area is 760 x 360 inset by 20.
        for (label, want) in [
            (
                "north-west, best gap",
                "<circle cx=\"20.0\" cy=\"20.0\" r=\"6\" fill=\"rgb(0,255,40)\">",
            ),
            (
                "south-east, worst gap",
                "<circle cx=\"780.0\" cy=\"380.0\" r=\"6\" fill=\"rgb(255,0,40)\">",
            ),
            (
                "centre, median gap",
                "<circle cx=\"400.0\" cy=\"200.0\" r=\"6\" fill=\"rgb(127,127,40)\">",
            ),
        ] {
            assert!(svg.contains(want), "{label}: expected {want}");
        }
    }
}
