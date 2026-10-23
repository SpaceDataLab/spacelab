//! Operational **VMF grid** reader — the 6-hourly gridded `a_h`/`a_w` (plus zenith delays) the TU Wien
//! VMF Data Server derives from ECMWF numerical weather, and the data source that feeds
//! [`crate::vmf1_mapping`] / [`crate::vmf3_mapping`]. Parse one epoch's grid, interpolate to a site,
//! hand the coefficients to the mapping function; blend two consecutive epochs in time with
//! [`VmfCoefficients::lerp`].
//!
//! File format (TU Wien GRID product): a `!`-comment header carrying `Data_types:` (VMF1 or VMF3) and
//! `Range/resolution: latmin latmax lonmin lonmax dlat dlon` (degrees), then `lat lon ah aw zhd zwd`
//! rows on a regular grid with latitudes descending. Bilinear in latitude/longitude, longitude
//! normalized to `[0, 2π)`, matching Orekit's `ViennaModelCoefficientsLoader`. The grids themselves
//! are operational data (refreshed every 6 h) and are a caller input, not embedded here.

// usize <-> f64 grid-index arithmetic (indices are small and exact in f64); the grid-geometry names
// (dlat/dlon, latmin/lonmin) and math indices (i/j/t/u) are standard; oracle literals stay byte-exact.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::unreadable_literal
)]

use core::f64::consts::PI;
use core::fmt;

/// The Vienna model a grid carries, read from its `! Data_types:` header line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VmfModel {
    /// VMF1 (2.0°×2.5° grids).
    Vmf1,
    /// VMF3 (1°×1° or 5°×5° grids).
    Vmf3,
}

/// Vienna coefficients at a point: the mapping `a` coefficients and the zenith delays shipped with
/// them. `ah`/`aw` feed [`crate::vmf1_mapping`] / [`crate::vmf3_mapping`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VmfCoefficients {
    /// Hydrostatic mapping coefficient `a_h`.
    pub ah: f64,
    /// Wet mapping coefficient `a_w`.
    pub aw: f64,
    /// Zenith hydrostatic delay, metres.
    pub zhd: f64,
    /// Zenith wet delay, metres.
    pub zwd: f64,
}

impl VmfCoefficients {
    /// Linear blend toward `other` by `frac` (clamped to `[0, 1]`) — temporal interpolation between two
    /// consecutive 6-hourly grids bracketing an epoch.
    #[must_use]
    pub fn lerp(self, other: Self, frac: f64) -> Self {
        let f = frac.clamp(0.0, 1.0);
        let mix = |a: f64, b: f64| a + (b - a) * f;
        Self {
            ah: mix(self.ah, other.ah),
            aw: mix(self.aw, other.aw),
            zhd: mix(self.zhd, other.zhd),
            zwd: mix(self.zwd, other.zwd),
        }
    }
}

/// A failure while parsing a VMF grid file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VmfGridError {
    /// No `! Data_types:` line naming VMF1 or VMF3.
    MissingDataType,
    /// No `! Range/resolution:` line, or it lacks the six range values.
    MalformedHeader,
    /// A data row (1-based line number) had fewer than six columns or an unparsable number.
    MalformedRow(usize),
    /// The number of data rows does not match the header's grid dimensions.
    RowCountMismatch {
        /// Rows implied by the header (`nlat * nlon`).
        expected: usize,
        /// Rows actually read.
        found: usize,
    },
}

impl fmt::Display for VmfGridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDataType => write!(f, "vmf grid: no VMF1/VMF3 Data_types header"),
            Self::MalformedHeader => {
                write!(f, "vmf grid: missing or malformed Range/resolution header")
            }
            Self::MalformedRow(line) => write!(f, "vmf grid: malformed data row at line {line}"),
            Self::RowCountMismatch { expected, found } => {
                write!(
                    f,
                    "vmf grid: expected {expected} rows from the header, found {found}"
                )
            }
        }
    }
}

impl std::error::Error for VmfGridError {}

/// A parsed operational VMF grid for one epoch, ready for spatial interpolation.
#[derive(Debug, Clone)]
pub struct VmfGrid {
    model: VmfModel,
    lat0: f64,
    lon0: f64,
    dlat: f64,
    dlon: f64,
    nlat: usize,
    nlon: usize,
    ah: Vec<f64>,
    aw: Vec<f64>,
    zhd: Vec<f64>,
    zwd: Vec<f64>,
}

impl VmfGrid {
    /// The Vienna model this grid carries.
    #[must_use]
    pub fn model(&self) -> VmfModel {
        self.model
    }

    /// Parse one TU Wien VMF GRID file from its text.
    ///
    /// # Errors
    /// Returns [`VmfGridError`] if the `Data_types:`/`Range/resolution:` headers are missing or
    /// malformed, a data row is malformed, or the row count disagrees with the header dimensions.
    pub fn parse(text: &str) -> Result<Self, VmfGridError> {
        let mut model: Option<VmfModel> = None;
        let mut range: Option<[f64; 6]> = None;
        let mut ah = Vec::new();
        let mut aw = Vec::new();
        let mut zhd = Vec::new();
        let mut zwd = Vec::new();

        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('!') {
                let rest = rest.trim_start();
                if let Some(payload) = rest.strip_prefix("Data_types:") {
                    if payload.contains("VMF3") {
                        model = Some(VmfModel::Vmf3);
                    } else if payload.contains("VMF1") {
                        model = Some(VmfModel::Vmf1);
                    }
                } else if let Some(payload) = rest.strip_prefix("Range/resolution:") {
                    let vals: Vec<f64> = payload
                        .split_whitespace()
                        .filter_map(|t| t.parse::<f64>().ok())
                        .collect();
                    if vals.len() < 6 {
                        return Err(VmfGridError::MalformedHeader);
                    }
                    range = Some([vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]]);
                }
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 6 {
                return Err(VmfGridError::MalformedRow(lineno + 1));
            }
            let num = |s: &str| {
                s.parse::<f64>()
                    .map_err(|_| VmfGridError::MalformedRow(lineno + 1))
            };
            ah.push(num(cols[2])?);
            aw.push(num(cols[3])?);
            zhd.push(num(cols[4])?);
            zwd.push(num(cols[5])?);
        }

        let model = model.ok_or(VmfGridError::MissingDataType)?;
        let [latmin, latmax, lonmin, lonmax, dlat_deg, dlon_deg] =
            range.ok_or(VmfGridError::MalformedHeader)?;
        if dlat_deg <= 0.0 || dlon_deg <= 0.0 {
            return Err(VmfGridError::MalformedHeader);
        }

        let nlat = ((latmax - latmin) / dlat_deg).round() as usize + 1;
        let mut nlon = ((lonmax - lonmin) / dlon_deg).round() as usize + 1;
        if model == VmfModel::Vmf1 {
            // TU Wien VMF1 grids run to 357.5°; a header ending at 360° carries the 0° column twice,
            // so drop that duplicate to keep the axis single-valued (Orekit does the same).
            let last = lonmin + (nlon as f64 - 1.0) * dlon_deg;
            if last > 357.5 + dlon_deg / 2.0 {
                nlon -= 1;
            }
        }
        if nlat < 2 || nlon < 2 {
            return Err(VmfGridError::MalformedHeader);
        }
        let expected = nlat * nlon;
        if ah.len() != expected {
            return Err(VmfGridError::RowCountMismatch {
                expected,
                found: ah.len(),
            });
        }

        // Reshape the file order (latitudes descending) into an ascending-latitude grid indexed
        // `lat_asc * nlon + lon`, so bilinear interpolation runs over strictly increasing axes.
        let mut g_ah = vec![0.0; expected];
        let mut g_aw = vec![0.0; expected];
        let mut g_zhd = vec![0.0; expected];
        let mut g_zwd = vec![0.0; expected];
        for m in 0..expected {
            let lat_desc = m / nlon;
            let lon_idx = m % nlon;
            let dst = (nlat - 1 - lat_desc) * nlon + lon_idx;
            g_ah[dst] = ah[m];
            g_aw[dst] = aw[m];
            g_zhd[dst] = zhd[m];
            g_zwd[dst] = zwd[m];
        }

        Ok(Self {
            model,
            lat0: latmin.to_radians(),
            lon0: lonmin.to_radians(),
            dlat: dlat_deg.to_radians(),
            dlon: dlon_deg.to_radians(),
            nlat,
            nlon,
            ah: g_ah,
            aw: g_aw,
            zhd: g_zhd,
            zwd: g_zwd,
        })
    }

    /// Bilinearly interpolate the coefficients at geodetic `latitude`/`longitude` (radians). Longitude
    /// is normalized to `[0, 2π)`; a point outside the grid clamps to its nearest edge cell (no
    /// wrap-around across the anti-meridian).
    #[must_use]
    pub fn interpolate(&self, latitude: f64, longitude: f64) -> VmfCoefficients {
        let (i, t) = locate(latitude, self.lat0, self.dlat, self.nlat);
        let (j, u) = locate(
            normalize_longitude(longitude),
            self.lon0,
            self.dlon,
            self.nlon,
        );
        let base = i * self.nlon + j;
        let bilinear = |g: &[f64]| {
            let v00 = g[base];
            let v01 = g[base + 1];
            let v10 = g[base + self.nlon];
            let v11 = g[base + self.nlon + 1];
            (1.0 - t) * (1.0 - u) * v00 + (1.0 - t) * u * v01 + t * (1.0 - u) * v10 + t * u * v11
        };
        VmfCoefficients {
            ah: bilinear(&self.ah),
            aw: bilinear(&self.aw),
            zhd: bilinear(&self.zhd),
            zwd: bilinear(&self.zwd),
        }
    }
}

/// Locate the cell index `i ∈ [0, n-2]` and fractional offset `t ∈ [0, 1]` of `x` on a regular axis
/// starting at `x0` with step `dx`; clamps outside the axis to the nearest edge cell.
fn locate(x: f64, x0: f64, dx: f64, n: usize) -> (usize, f64) {
    let f = (x - x0) / dx;
    let i = (f.floor() as isize).clamp(0, n as isize - 2) as usize;
    let t = (f - i as f64).clamp(0.0, 1.0);
    (i, t)
}

/// Normalize an angle to `[0, 2π)`, matching Orekit's `MathUtils.normalizeAngle(lon, π)`.
fn normalize_longitude(a: f64) -> f64 {
    let two_pi = 2.0 * PI;
    let r = a % two_pi;
    if r < 0.0 { r + two_pi } else { r }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)]

    use super::*;

    // A 2×2-node VMF3 grid (latitudes 0°/10°, longitudes 0°/10°), rows in file order (lat descending).
    const MINI: &str = "\
! Data_types:         VMF3 (lat lon ah aw zhd zwd)
! Range/resolution:   0 10 0 10 10 10
 10.0  0.0  0.0010 0.0004 2.0 0.10
 10.0 10.0  0.0020 0.0004 2.0 0.10
  0.0  0.0  0.0030 0.0004 2.0 0.10
  0.0 10.0  0.0040 0.0004 2.0 0.10
";

    #[test]
    fn parses_real_vmf3_5x5_and_matches_orekit_at_a_node() {
        let grid = VmfGrid::parse(include_str!("../tests/fixtures/vmf3_5x5_20181125.H00")).unwrap();
        assert_eq!(grid.model(), VmfModel::Vmf3);
        // Orekit ViennaModelCoefficientsLoaderTest, VMF3 5×5, node (77.5°, 2.5°).
        let c = grid.interpolate(77.5_f64.to_radians(), 2.5_f64.to_radians());
        assert!((c.ah - 0.00117002).abs() < 1e-8, "ah {}", c.ah);
        assert!((c.aw - 0.00045484).abs() < 1e-8, "aw {}", c.aw);
        assert!((c.zhd - 2.3203).abs() < 1e-4, "zhd {}", c.zhd);
        assert!((c.zwd - 0.0191).abs() < 1e-4, "zwd {}", c.zwd);
    }

    #[test]
    fn bilinear_at_the_cell_centre_is_the_mean_of_the_corners() {
        let grid = VmfGrid::parse(MINI).unwrap();
        let c = grid.interpolate(5.0_f64.to_radians(), 5.0_f64.to_radians());
        assert!((c.ah - 0.0025).abs() < 1e-12, "ah {}", c.ah); // (0.001+0.002+0.003+0.004)/4
    }

    #[test]
    fn interpolation_at_a_node_returns_that_node() {
        let grid = VmfGrid::parse(MINI).unwrap();
        assert!((grid.interpolate(0.0, 0.0).ah - 0.0030).abs() < 1e-12);
        let corner = grid.interpolate(10.0_f64.to_radians(), 10.0_f64.to_radians());
        assert!((corner.ah - 0.0020).abs() < 1e-12);
    }

    #[test]
    fn lerp_blends_two_epochs_at_the_midpoint() {
        let a = VmfCoefficients {
            ah: 0.001,
            aw: 0.0004,
            zhd: 2.0,
            zwd: 0.1,
        };
        let b = VmfCoefficients {
            ah: 0.003,
            aw: 0.0006,
            zhd: 2.4,
            zwd: 0.3,
        };
        let m = a.lerp(b, 0.5);
        assert!((m.ah - 0.002).abs() < 1e-12 && (m.zhd - 2.2).abs() < 1e-12);
    }

    #[test]
    fn feeds_the_vmf3_mapping_function() {
        // The operational chain: grid → coefficients → slant mapping factor.
        let grid = VmfGrid::parse(include_str!("../tests/fixtures/vmf3_5x5_20181125.H00")).unwrap();
        let c = grid.interpolate(77.5_f64.to_radians(), 2.5_f64.to_radians());
        let (mh, mw) = crate::vmf3_mapping(
            c.ah,
            c.aw,
            77.5_f64.to_radians(),
            2.5_f64.to_radians(),
            0.0,
            329.0,
            20.0_f64.to_radians(),
        );
        assert!(mh > 1.0 && mw > 1.0, "mapping factors {mh} {mw}");
    }

    #[test]
    fn rejects_a_row_count_that_disagrees_with_the_header() {
        let bad =
            "! Data_types: VMF3\n! Range/resolution: 0 10 0 10 10 10\n 10 0 0.001 0.0004 2.0 0.1\n";
        assert!(matches!(
            VmfGrid::parse(bad),
            Err(VmfGridError::RowCountMismatch { .. })
        ));
    }

    /// A malformed row names the line it is on, and the number is the 1-based line of the file.
    ///
    /// The only test that reached [`VmfGridError::MalformedRow`] matched it with a wildcard payload,
    /// so the line number could be off by one, or zero, and nothing would notice. It is the only
    /// thing in the error a person reading it acts on.
    #[test]
    fn a_malformed_row_reports_the_line_it_is_on() {
        // Line 3 has five columns, one short of the six the format requires.
        let short = "! Data_types: VMF3\n\
                     ! Range/resolution: 0 10 0 10 10 10\n\
                     10 0 0.001 0.0004 2.0\n";
        assert_eq!(
            VmfGrid::parse(short).unwrap_err(),
            VmfGridError::MalformedRow(3)
        );

        // Line 3 has six columns but the fourth is not a number. A blank line before it confirms
        // that skipped lines still count towards the number reported.
        let unparsable = "! Data_types: VMF3\n\
                          \n\
                          10 0 0.001 nan-ish 2.0 0.1\n";
        assert_eq!(
            VmfGrid::parse(unparsable).unwrap_err(),
            VmfGridError::MalformedRow(3)
        );
    }

    /// Each half of the two header guards refuses on its own.
    ///
    /// Both are `a || b` over conditions that no fixture ever separated — every rejected header
    /// failed both halves at once, so `&&` would have refused exactly the same files. A grid with
    /// one usable step and one unusable one, and a grid degenerate along one axis only, separate
    /// them.
    #[test]
    fn a_single_unusable_axis_is_enough_to_refuse_the_header() {
        let with_range = |range: &str| {
            format!(
                "! Data_types: VMF3\n! Range/resolution: {range}\n\
                 10 0 0.001 0.0004 2.0 0.1\n 0 0 0.002 0.0004 2.0 0.1\n"
            )
        };
        // A negative latitude step over a descending latitude range: `dlon` is fine, `dlat` is not.
        assert_eq!(
            VmfGrid::parse(&with_range("90 -90 0 10 -2 5")).unwrap_err(),
            VmfGridError::MalformedHeader
        );
        // The mirror image: `dlat` is fine, `dlon` is negative over a descending longitude range.
        assert_eq!(
            VmfGrid::parse(&with_range("0 10 360 0 5 -10")).unwrap_err(),
            VmfGridError::MalformedHeader
        );
        // One latitude row: bilinear interpolation has no cell to work in, though the longitudes do.
        assert_eq!(
            VmfGrid::parse(&with_range("0 0 0 10 10 10")).unwrap_err(),
            VmfGridError::MalformedHeader
        );
        // And one longitude column, with two latitude rows.
        assert_eq!(
            VmfGrid::parse(&with_range("0 10 0 0 10 10")).unwrap_err(),
            VmfGridError::MalformedHeader
        );
    }

    /// A two-row VMF1 grid on a 2.5° longitude axis, spanning `lonmin`…`lonmax`.
    fn vmf1_grid(lonmin: f64, lonmax: f64, columns: usize) -> String {
        use std::fmt::Write as _;
        let mut s = format!(
            "! Data_types:         VMF1 (lat lon ah aw zhd zwd)\n\
             ! Range/resolution:   0 2 {lonmin} {lonmax} 2 2.5\n"
        );
        for lat in [2.0_f64, 0.0] {
            for k in 0..columns {
                let lon = lonmin + 2.5 * k as f64;
                let _ = writeln!(s, " {lat} {lon} 0.0012 0.0004 2.0 0.10");
            }
        }
        s
    }

    /// The VMF1 duplicate-longitude trim, which no fixture in the crate reached.
    ///
    /// TU Wien VMF1 grids stop at 357.5°, but a header may declare the axis all the way to 360°,
    /// which repeats the 0° column. The parser drops that column, and everything about how it
    /// decides — the last node's longitude, the half-step tolerance, the direction of the
    /// comparison and the decrement — was unexercised, because the only grids in the crate are
    /// VMF3. The three headers below sit on either side of the decision and exactly on it.
    #[test]
    fn a_vmf1_axis_that_reaches_360_degrees_drops_its_duplicate_column() {
        // 0°…360° at 2.5° is 145 declared nodes, the last of which repeats the first: 144 rows.
        let wrapped = VmfGrid::parse(&vmf1_grid(0.0, 360.0, 144)).unwrap();
        assert_eq!(wrapped.model(), VmfModel::Vmf1);
        assert_eq!(wrapped.nlon, 144);
        assert_eq!(wrapped.nlat, 2);

        // 0°…357.5° already stops one step short, so nothing is dropped.
        let exact = VmfGrid::parse(&vmf1_grid(0.0, 357.5, 144)).unwrap();
        assert_eq!(exact.nlon, 144);

        // The boundary itself: an axis offset by half a step ends at 358.75°, which is exactly
        // `357.5 + dlon/2` — inside the tolerance, so still no duplicate to drop.
        let boundary = VmfGrid::parse(&vmf1_grid(1.25, 358.75, 144)).unwrap();
        assert_eq!(boundary.nlon, 144);
    }

    // A 3×3 VMF3 grid (latitudes 0°/10°/20°, longitudes 0°/10°/20°), rows in file order.
    const MINI3: &str = "\
! Data_types:         VMF3 (lat lon ah aw zhd zwd)
! Range/resolution:   0 20 0 20 10 10
 20.0  0.0  0.0070 0.0004 2.0 0.10
 20.0 10.0  0.0080 0.0004 2.0 0.10
 20.0 20.0  0.0090 0.0004 2.0 0.10
 10.0  0.0  0.0040 0.0004 2.0 0.10
 10.0 10.0  0.0050 0.0004 2.0 0.10
 10.0 20.0  0.0060 0.0004 2.0 0.10
  0.0  0.0  0.0010 0.0004 2.0 0.10
  0.0 10.0  0.0020 0.0004 2.0 0.10
  0.0 20.0  0.0030 0.0004 2.0 0.10
";

    /// The longitude index reaches the flat array, on a grid with more than one longitude cell.
    ///
    /// Every other fixture is two columns wide, so `locate` can only ever return `j = 0` and the
    /// `+ j` in `base = i·nlon + j` adds nothing. Here the cell is the interior one, where the row
    /// and column offsets are different numbers.
    #[test]
    fn the_longitude_index_offsets_the_row_it_selects() {
        let grid = VmfGrid::parse(MINI3).unwrap();
        // The interior node (10°, 10°) is its own value.
        assert!(
            (grid
                .interpolate(10.0_f64.to_radians(), 10.0_f64.to_radians())
                .ah
                - 0.0050)
                .abs()
                < 1e-12
        );
        // The centre of the north-east cell is the mean of 0.005, 0.006, 0.008 and 0.009.
        let c = grid.interpolate(15.0_f64.to_radians(), 15.0_f64.to_radians());
        assert!((c.ah - 0.0070).abs() < 1e-12, "ah {}", c.ah);
    }

    /// Longitudes outside `[0, 2π)` are folded into it — the only part of `normalize_longitude`
    /// that does anything, and the only part no caller in the test suite ever supplied.
    #[test]
    fn longitude_normalisation_folds_both_ways() {
        let two_pi = 2.0 * PI;
        assert!((normalize_longitude(-0.1) - (two_pi - 0.1)).abs() < 1e-15);
        assert!((normalize_longitude(7.0) - (7.0 - two_pi)).abs() < 1e-15);
        assert!((normalize_longitude(-two_pi - 0.25) - (two_pi - 0.25)).abs() < 1e-14);
        // Already inside the interval, including well past `2 + π` where a wrong period would bite.
        for a in [0.0, 1.0, 5.5, two_pi - 1e-9] {
            assert!((normalize_longitude(a) - a).abs() < 1e-15, "{a}");
        }
    }
}
