//! Orthorectification — reprojecting a satellite image onto a map grid over terrain, using an
//! [`crate::rpc::RationalPolynomialCamera`] and a digital elevation model.
//!
//! Two operations, inverse to each other:
//!
//! - **Indirect orthorectification** ([`orthorectify_point`], [`orthorectify`]) — the practical form:
//!   for each output map cell `(lat, lon)`, read the terrain height from the DEM, project it into the
//!   source image with the RPC, and sample there. Iterating over the *output* grid (hence "indirect")
//!   guarantees every output pixel is filled.
//! - **Ray–DEM intersection** ([`ray_dem_intersection`]) — the forward form: an image pixel's line of
//!   sight is the locus `RPC⁻¹(row, col, h)` as the height `h` varies; where that ray pierces the DEM
//!   surface is the ground point the pixel sees. Found by a bracketed bisection on
//!   `g(h) = h − DEM(RPC⁻¹(row, col, h))`.
//!
//! The two compose to the identity — a map point projected to the image and ray-traced back through the
//! DEM returns to itself — which is exactly how the module is verified (on the real Pléiades RPC).

// Grid/image indices are small; the usize↔f64 casts in the bilinear interpolators stay in range.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use crate::rpc::RationalPolynomialCamera;
use alloc::vec;
use alloc::vec::Vec;

/// A regular latitude/longitude grid of terrain heights (metres), row-major with row `r`, column `c`
/// at `(min_lat + r·d_lat, min_lon + c·d_lon)`.
#[derive(Clone, Debug)]
pub struct DemGrid {
    /// Latitude of row 0 (degrees).
    pub min_lat: f64,
    /// Longitude of column 0 (degrees).
    pub min_lon: f64,
    /// Latitude step per row (degrees, > 0).
    pub d_lat: f64,
    /// Longitude step per column (degrees, > 0).
    pub d_lon: f64,
    /// Number of rows (latitude samples).
    pub rows: usize,
    /// Number of columns (longitude samples).
    pub cols: usize,
    /// Heights (metres), `rows·cols` row-major.
    pub heights: Vec<f64>,
}

impl DemGrid {
    /// Bilinearly interpolate the height (metres) at `(lat, lon)`, clamping to the grid edges outside
    /// its extent.
    #[must_use]
    pub fn height_at(&self, lat: f64, lon: f64) -> f64 {
        let fr = ((lat - self.min_lat) / self.d_lat).clamp(0.0, (self.rows - 1) as f64);
        let fc = ((lon - self.min_lon) / self.d_lon).clamp(0.0, (self.cols - 1) as f64);
        let r0 = libm::floor(fr) as usize;
        let c0 = libm::floor(fc) as usize;
        let r1 = (r0 + 1).min(self.rows - 1);
        let c1 = (c0 + 1).min(self.cols - 1);
        let (wr, wc) = (fr - r0 as f64, fc - c0 as f64);
        let h = |r: usize, c: usize| self.heights[r * self.cols + c];
        let top = h(r0, c0) * (1.0 - wc) + h(r0, c1) * wc;
        let bot = h(r1, c0) * (1.0 - wc) + h(r1, c1) * wc;
        top * (1.0 - wr) + bot * wr
    }
}

/// The source-image coordinate `(row, col)` a map point `(lat, lon)` maps to — indirect
/// orthorectification of a single point. The DEM supplies the height the RPC needs.
#[must_use]
pub fn orthorectify_point(
    rpc: &RationalPolynomialCamera,
    dem: &DemGrid,
    lat: f64,
    lon: f64,
) -> (f64, f64) {
    rpc.forward(lat, lon, dem.height_at(lat, lon))
}

/// Bilinearly sample a row-major image at fractional `(row, col)`, or `None` if outside `[0, rows−1] ×
/// [0, cols−1]`.
#[must_use]
pub fn sample_bilinear(image: &[f64], rows: usize, cols: usize, row: f64, col: f64) -> Option<f64> {
    if row < 0.0 || col < 0.0 || row > (rows - 1) as f64 || col > (cols - 1) as f64 {
        return None;
    }
    let r0 = libm::floor(row) as usize;
    let c0 = libm::floor(col) as usize;
    let r1 = (r0 + 1).min(rows - 1);
    let c1 = (c0 + 1).min(cols - 1);
    let (wr, wc) = (row - r0 as f64, col - c0 as f64);
    let p = |r: usize, c: usize| image[r * cols + c];
    let top = p(r0, c0) * (1.0 - wc) + p(r0, c1) * wc;
    let bot = p(r1, c0) * (1.0 - wc) + p(r1, c1) * wc;
    Some(top * (1.0 - wr) + bot * wr)
}

/// The output grid of an orthorectification: a regular lat/lon raster to resample the source image onto.
#[derive(Clone, Copy, Debug)]
pub struct MapGrid {
    /// Latitude of output row 0 (degrees).
    pub min_lat: f64,
    /// Longitude of output column 0 (degrees).
    pub min_lon: f64,
    /// Latitude step (degrees).
    pub d_lat: f64,
    /// Longitude step (degrees).
    pub d_lon: f64,
    /// Output rows.
    pub rows: usize,
    /// Output columns.
    pub cols: usize,
}

/// A source image tile: a row-major `rows × cols` buffer whose element `[0]` is the absolute image
/// pixel `origin = [row, col]` (real ortho runs on tiles, not the whole scene).
#[derive(Clone, Copy, Debug)]
pub struct SourceTile<'a> {
    /// Pixel values, `rows·cols` row-major.
    pub data: &'a [f64],
    /// Tile height (pixels).
    pub rows: usize,
    /// Tile width (pixels).
    pub cols: usize,
    /// Absolute image coordinate `[row, col]` of `data[0]`.
    pub origin: [f64; 2],
}

impl SourceTile<'_> {
    /// Bilinearly sample the tile at absolute image `(row, col)`, or `None` if outside it.
    #[must_use]
    pub fn sample(&self, row: f64, col: f64) -> Option<f64> {
        sample_bilinear(
            self.data,
            self.rows,
            self.cols,
            row - self.origin[0],
            col - self.origin[1],
        )
    }
}

/// Orthorectify a `source` image tile onto the `out` map raster via `rpc` and `dem`. Each output cell is
/// filled by projecting its ground point into the source and bilinearly sampling; cells whose source
/// coordinate falls outside the tile get `fill`.
#[must_use]
pub fn orthorectify(
    rpc: &RationalPolynomialCamera,
    dem: &DemGrid,
    source: &SourceTile,
    out: MapGrid,
    fill: f64,
) -> Vec<f64> {
    let mut ortho = vec![fill; out.rows * out.cols];
    for r in 0..out.rows {
        let lat = out.min_lat + r as f64 * out.d_lat;
        for c in 0..out.cols {
            let lon = out.min_lon + c as f64 * out.d_lon;
            let (row, col) = orthorectify_point(rpc, dem, lat, lon);
            if let Some(v) = source.sample(row, col) {
                ortho[r * out.cols + c] = v;
            }
        }
    }
    ortho
}

/// The ground point `(lat, lon, height)` where the line of sight through image pixel `(row, col)`
/// pierces the DEM surface, searched over heights `[h_min, h_max]` by bisection. `None` if the ray does
/// not cross the surface within that height bracket (the pixel looks off the DEM, or the bracket is too
/// narrow). Increasing `h` must place the ray *above* the surface (`h_max` high, `h_min` low).
#[must_use]
pub fn ray_dem_intersection(
    rpc: &RationalPolynomialCamera,
    dem: &DemGrid,
    row: f64,
    col: f64,
    h_min: f64,
    h_max: f64,
) -> Option<(f64, f64, f64)> {
    // g(h) = h − DEM(ground point of the ray at height h). The ray is above the terrain where g > 0,
    // below where g < 0; the surface is the sign change.
    let g = |h: f64| {
        let gp = rpc.inverse(row, col, h).ok()?;
        Some((
            h - dem.height_at(gp.latitude, gp.longitude),
            gp.latitude,
            gp.longitude,
        ))
    };
    let (mut lo, mut hi) = (h_min, h_max);
    let (glo, _, _) = g(lo)?;
    let (ghi, _, _) = g(hi)?;
    if glo > 0.0 || ghi < 0.0 {
        return None; // no bracketed sign change (g(low) should be < 0, g(high) > 0)
    }
    let mut result = (0.0, 0.0, 0.0);
    for _ in 0..80 {
        let mid = f64::midpoint(lo, hi);
        let (gm, lat, lon) = g(mid)?;
        result = (lat, lon, mid);
        if libm::fabs(gm) < 1e-6 || (hi - lo) < 1e-9 {
            break;
        }
        if gm > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
    use super::*;

    const RPB: &str = include_str!("../tests/fixtures/pleiades_reunion.rpb");

    /// A tilted-plane DEM over the RPC's ground footprint (Réunion, ~lat −21.32, lon 55.75).
    fn tilted_dem() -> DemGrid {
        let (rows, cols) = (41usize, 41usize);
        let (min_lat, min_lon) = (-21.37, 55.70);
        let (d_lat, d_lon) = (0.0025, 0.0025);
        let mut heights = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let lat = min_lat + r as f64 * d_lat;
                let lon = min_lon + c as f64 * d_lon;
                // A ramp plus a gentle bump, ~900–1700 m.
                heights[r * cols + c] = 1300.0
                    + 4000.0 * (lat + 21.32)
                    + 3000.0 * (lon - 55.75)
                    + 120.0 * ((lat + 21.32) * (lon - 55.75) * 400.0);
            }
        }
        DemGrid {
            min_lat,
            min_lon,
            d_lat,
            d_lon,
            rows,
            cols,
            heights,
        }
    }

    /// A DEM of a single sample is flat by construction: `height_at` clamps both indices to zero, so
    /// it returns that one height *exactly* for every coordinate. It reduces `ray_dem_intersection`
    /// to the root find `g(h) = h − H`, whose answer is known in closed form and whose bracket
    /// endpoints can be placed exactly on the surface.
    fn flat_dem(height: f64) -> DemGrid {
        DemGrid {
            min_lat: 0.0,
            min_lon: 0.0,
            d_lat: 1.0,
            d_lon: 1.0,
            rows: 1,
            cols: 1,
            heights: vec![height],
        }
    }

    #[test]
    fn height_at_is_bilinear_and_clamps() {
        let dem = DemGrid {
            min_lat: 0.0,
            min_lon: 0.0,
            d_lat: 1.0,
            d_lon: 1.0,
            rows: 2,
            cols: 2,
            heights: vec![0.0, 10.0, 20.0, 30.0],
        };
        assert!((dem.height_at(0.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((dem.height_at(0.0, 0.5) - 5.0).abs() < 1e-12);
        assert!((dem.height_at(0.5, 0.5) - 15.0).abs() < 1e-12);
        // Outside the grid clamps to the edge: lat→row 0 gives the top edge (0,10)→5; lon→col 1 gives
        // the right edge (10,30) at lat 0.5 → 20.
        assert!((dem.height_at(-3.0, 0.5) - 5.0).abs() < 1e-12);
        assert!((dem.height_at(0.5, 9.0) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn height_at_reads_a_grid_whose_origin_and_step_are_not_the_identity() {
        // The fixture above sits at the origin with unit steps and is 2×2, which hides three things at
        // once: subtracting `min_lat` is the identity, dividing by `d_lat` is the identity, and
        // `rows − 1 = 1` coincides with the only other row. Offset the origin, give the axes different
        // non-unit steps, and grow the grid so interior and edge are distinct samples. Heights are
        // linear in the index — `h(r, c) = 28r + 7c` — and bilinear interpolation of a linear field is
        // exact, so every expected value below is that closed form and not a recorded output.
        let dem = DemGrid {
            min_lat: 10.0,
            min_lon: -30.0,
            d_lat: 0.25,
            d_lon: 4.0,
            rows: 3,
            cols: 4,
            heights: (0..12).map(|k| 7.0 * f64::from(k)).collect(),
        };
        let lat = |fr: f64| 10.0 + fr * 0.25;
        let lon = |fc: f64| -30.0 + fc * 4.0;
        assert!((dem.height_at(lat(1.5), lon(2.25)) - 57.75).abs() < 1e-12);
        // Past an edge the index saturates at that edge: rows at 2, columns at 3.
        assert!((dem.height_at(-1e3, lon(2.25)) - 15.75).abs() < 1e-12);
        assert!((dem.height_at(1e3, lon(2.25)) - 71.75).abs() < 1e-12);
        assert!((dem.height_at(lat(1.5), -1e3) - 42.0).abs() < 1e-12);
        assert!((dem.height_at(lat(1.5), 1e3) - 63.0).abs() < 1e-12);
    }

    #[test]
    fn sample_bilinear_includes_its_far_edge_and_nothing_past_it() {
        // The domain is the *closed* box [0, rows−1] × [0, cols−1]. Nothing sampled the last row or
        // the last column, which is where both the `>` bounds test and the `.min(rows − 1)` index
        // clamp live — the clamp only ever engages on the final row or column.
        let (rows, cols) = (3usize, 4usize);
        let img: Vec<f64> = (0..12).map(|k| 3.0 * f64::from(k)).collect();
        let at = |r: f64, c: f64| sample_bilinear(&img, rows, cols, r, c);
        // p(r, c) = 12r + 3c, again exact under bilinear interpolation.
        assert!((at(2.0, 3.0).unwrap() - 33.0).abs() < 1e-12);
        assert!((at(2.0, 1.5).unwrap() - 28.5).abs() < 1e-12);
        assert!((at(0.5, 3.0).unwrap() - 15.0).abs() < 1e-12);
        assert!(at(2.0 + 1e-12, 1.5).is_none());
        assert!(at(1.5, 3.0 + 1e-12).is_none());
        assert!(at(-1e-12, 1.5).is_none());
        assert!(at(1.5, -1e-12).is_none());
    }

    #[test]
    fn the_ray_search_needs_a_bracket_that_straddles_the_surface() {
        // Below the ellipsoid, so that a bracket symmetric about zero is expressible while staying
        // inside the RPC's height normalisation (offset 1295 m, scale 1315 m ⇒ valid from −20 m).
        const H: f64 = -10.0;
        let rpc = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        let dem = flat_dem(H);
        let (row, col) = rpc.forward(-21.32, 55.75, H);
        let solve =
            |h_min: f64, h_max: f64| ray_dem_intersection(&rpc, &dem, row, col, h_min, h_max);
        // An endpoint *on* the surface is a sign change, not the absence of one: g = 0 there.
        assert!(
            (solve(H, 20.0).unwrap().2 - H).abs() < 1e-6,
            "h_min on the surface"
        );
        assert!(
            (solve(-20.0, H).unwrap().2 - H).abs() < 1e-6,
            "h_max on the surface"
        );
        // A bracket wholly above the surface, or wholly below it, is no bracket at all.
        assert!(solve(0.0, 20.0).is_none(), "bracket above the surface");
        assert!(solve(-20.0, -15.0).is_none(), "bracket below the surface");
        // Symmetric about zero: `hi + lo` and `hi / lo` are both under 1e-9 at the first midpoint
        // while `hi − lo` is 40 m, so a search that stops on the wrong quantity answers 0 m. The
        // bisection reaches the root exactly here — mid is −10 m on the second step.
        let (lat, lon, h) = solve(-20.0, 20.0).unwrap();
        assert!((h - H).abs() < 1e-9, "symmetric bracket gave {h}");
        assert!((h - dem.height_at(lat, lon)).abs() < 1e-9);
    }

    #[test]
    fn ortho_round_trips_through_the_dem_on_the_real_rpc() {
        let rpc = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        let dem = tilted_dem();
        // Several ground points inside the footprint: forward to the image over terrain, then ray-trace
        // the pixel back through the DEM — it must return to the original ground point.
        for (lat, lon) in [
            (-21.32, 55.75),
            (-21.34, 55.73),
            (-21.30, 55.77),
            (-21.33, 55.78),
        ] {
            let h = dem.height_at(lat, lon);
            let (row, col) = orthorectify_point(&rpc, &dem, lat, lon);
            let (lat2, lon2, h2) =
                ray_dem_intersection(&rpc, &dem, row, col, 800.0, 1800.0).unwrap();
            assert!(
                (lat2 - lat).abs() < 1e-7 && (lon2 - lon).abs() < 1e-7 && (h2 - h).abs() < 1e-2,
                "round-trip ({lat},{lon},{h}) -> ({row},{col}) -> ({lat2},{lon2},{h2})"
            );
            // The recovered point lies on the DEM surface.
            assert!((h2 - dem.height_at(lat2, lon2)).abs() < 1e-2);
        }
    }

    #[test]
    fn indirect_ortho_resamples_a_synthetic_source() {
        let rpc = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
        let dem = tilted_dem();
        let out = MapGrid {
            min_lat: -21.331,
            min_lon: 55.744,
            d_lat: 0.0004,
            d_lon: 0.0004,
            rows: 4,
            cols: 4,
        };
        // Size a source tile to cover exactly where this output grid projects (the RPC's perspective
        // sends it far from image centre), so no cell falls outside.
        let mut min_row = f64::INFINITY;
        let mut min_col = f64::INFINITY;
        let mut max_row = f64::NEG_INFINITY;
        let mut max_col = f64::NEG_INFINITY;
        for r in 0..out.rows {
            for c in 0..out.cols {
                let (row, col) = orthorectify_point(
                    &rpc,
                    &dem,
                    out.min_lat + r as f64 * out.d_lat,
                    out.min_lon + c as f64 * out.d_lon,
                );
                min_row = min_row.min(row);
                min_col = min_col.min(col);
                max_row = max_row.max(row);
                max_col = max_col.max(col);
            }
        }
        let src_origin = [min_row.floor() - 2.0, min_col.floor() - 2.0];
        let src_rows = (max_row.ceil() - src_origin[0]) as usize + 3;
        let src_cols = (max_col.ceil() - src_origin[1]) as usize + 3;
        // A synthetic source tile linear in the absolute pixel (row, col); bilinear resampling of a
        // linear field is exact, so each output cell must equal f at the projected source coordinate.
        let f = |row: f64, col: f64| 0.3 * row + 0.2 * col + 5.0;
        let mut source = vec![0.0; src_rows * src_cols];
        for r in 0..src_rows {
            for c in 0..src_cols {
                source[r * src_cols + c] = f(src_origin[0] + r as f64, src_origin[1] + c as f64);
            }
        }
        let tile = SourceTile {
            data: &source,
            rows: src_rows,
            cols: src_cols,
            origin: src_origin,
        };
        let ortho = orthorectify(&rpc, &dem, &tile, out, f64::NAN);
        for r in 0..out.rows {
            for c in 0..out.cols {
                let lat = out.min_lat + r as f64 * out.d_lat;
                let lon = out.min_lon + c as f64 * out.d_lon;
                let (row, col) = orthorectify_point(&rpc, &dem, lat, lon);
                let got = ortho[r * out.cols + c];
                assert!(
                    !got.is_nan(),
                    "cell ({r},{c}) -> ({row},{col}) fell outside the tile"
                );
                let expect = f(row, col);
                assert!((got - expect).abs() < 1e-6, "ortho {got} vs {expect}");
            }
        }
    }
}
