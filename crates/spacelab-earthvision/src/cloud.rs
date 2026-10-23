//! Cloud masking and swath-footprint filtering — the onboard Earth-observation decision layer, behind
//! the opt-in `onnx` feature.
//!
//! [`CloudMasker`] runs a segmentation CNN (loaded from ONNX through spacelab-vision's `tract`
//! runtime) over a multispectral tile and thresholds it into a per-pixel cloud mask;
//! [`usable_footprints`] then projects the cloud-free across-track columns onto the ground through
//! [`crate::swath`]. Together they let a spacecraft decide *what is worth downlinking or tasking*
//! before spending the bandwidth — the payoff of onboard inference.

// usize -> f64/f32 ratios for clear-fraction arithmetic; the magnitudes are tiny (pixel counts).
#![allow(clippy::cast_precision_loss)]

use core::f64::consts::FRAC_PI_2;

use spacelab_vision::{OnnxModel, threshold_mask};

pub use spacelab_vision::OnnxError;

use crate::swath::{ground_point_at, swath_geometry};

/// A cloud-segmentation model plus the probability threshold above which a pixel counts as cloudy.
pub struct CloudMasker {
    model: OnnxModel,
    threshold: f32,
}

impl CloudMasker {
    /// Load a cloud-segmentation model from ONNX bytes. `threshold` is the cloud-probability cut (e.g.
    /// `0.5`) at or above which a pixel is flagged cloudy.
    ///
    /// # Errors
    /// Returns [`OnnxError`] if the bytes are not a model `tract` can load and optimize.
    pub fn new(onnx_bytes: &[u8], threshold: f32) -> Result<Self, OnnxError> {
        Ok(Self {
            model: OnnxModel::load(onnx_bytes)?,
            threshold,
        })
    }

    /// The per-pixel cloud mask (`true` = cloudy) of a `channels x height x width` tile, row-major over
    /// the model's output grid.
    ///
    /// # Errors
    /// Returns [`OnnxError`] if the tile length is not `channels * height * width` or inference fails.
    pub fn mask(
        &self,
        tile: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<Vec<bool>, OnnxError> {
        let probabilities = self.model.run_chw(tile, channels, height, width)?;
        Ok(threshold_mask(&probabilities, self.threshold))
    }

    /// The cloud-free fraction of a tile in `[0, 1]` — a scene-usability score; an empty output is
    /// reported as `0.0`.
    ///
    /// # Errors
    /// Returns [`OnnxError`] on the same conditions as [`Self::mask`].
    pub fn clear_fraction(
        &self,
        tile: &[f32],
        channels: usize,
        height: usize,
        width: usize,
    ) -> Result<f32, OnnxError> {
        let mask = self.mask(tile, channels, height, width)?;
        if mask.is_empty() {
            return Ok(0.0);
        }
        let clear = mask.iter().filter(|&&cloudy| !cloudy).count();
        Ok(clear as f32 / mask.len() as f32)
    }
}

/// One across-track ground footprint kept by [`usable_footprints`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    /// Geodetic latitude of the footprint, radians.
    pub lat: f64,
    /// Geodetic longitude of the footprint, radians.
    pub lon: f64,
    /// Elevation of the satellite above the footprint's local horizon, radians.
    pub elevation: f64,
}

/// The geometry of one across-track sensor line for [`usable_footprints`].
#[derive(Debug, Clone, Copy)]
pub struct SwathLine<'a> {
    /// Earth radius, metres.
    pub re: f64,
    /// Satellite altitude above the surface, metres.
    pub altitude: f64,
    /// Sub-satellite latitude, radians.
    pub sub_lat: f64,
    /// Sub-satellite longitude, radians.
    pub sub_lon: f64,
    /// Ground-track azimuth (radians, clockwise from north); across-track columns lie ±90° of it.
    pub ground_track_azimuth: f64,
    /// Signed off-nadir look angle of each across-track column (radians); the sign selects the side.
    pub look_angles: &'a [f64],
}

/// Project the cloud-free across-track columns of a sensor line onto the ground.
///
/// `cloudy` is the per-pixel cloud mask (row-major `rows x cols`, e.g. from [`CloudMasker::mask`]); a
/// column is usable when its cloud-free fraction is at least `min_clear`. Each usable, on-Earth column
/// (one per [`SwathLine::look_angles`] entry) becomes a ground [`Footprint`] via [`swath_geometry`] +
/// [`ground_point_at`]; columns whose line of sight misses the Earth are dropped. Returns an empty
/// vector on any shape mismatch (`cloudy.len() != rows*cols` or `look_angles.len() != cols`).
#[must_use]
pub fn usable_footprints(
    scan: &SwathLine<'_>,
    cloudy: &[bool],
    rows: usize,
    cols: usize,
    min_clear: f64,
) -> Vec<Footprint> {
    let mut footprints = Vec::new();
    if rows == 0 || cols == 0 || cloudy.len() != rows * cols || scan.look_angles.len() != cols {
        return footprints;
    }
    for (col, &look_angle) in scan.look_angles.iter().enumerate() {
        let clear = (0..rows).filter(|&row| !cloudy[row * cols + col]).count();
        if (clear as f64) < min_clear * rows as f64 {
            continue;
        }
        let Some(geometry) = swath_geometry(scan.re, scan.altitude, libm::fabs(look_angle)) else {
            continue;
        };
        let azimuth = scan.ground_track_azimuth + libm::copysign(FRAC_PI_2, look_angle);
        let (lat, lon) = ground_point_at(
            scan.sub_lat,
            scan.sub_lon,
            geometry.earth_central_angle,
            azimuth,
        );
        footprints.push(Footprint {
            lat,
            lon,
            elevation: geometry.elevation,
        });
    }
    footprints
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
    use super::*;

    const RE: f64 = 6_378_137.0;

    fn line(look_angles: &[f64]) -> SwathLine<'_> {
        SwathLine {
            re: RE,
            altitude: 700_000.0,
            sub_lat: 0.0,
            sub_lon: 0.0,
            ground_track_azimuth: 0.0,
            look_angles,
        }
    }

    #[test]
    fn usable_footprints_drops_cloudy_columns_and_places_the_rest_across_track() {
        // 2 rows x 3 across-track columns; the nadir column is fully cloudy, the two off-nadir clear.
        let cloudy = vec![false, true, false, false, true, false];
        let look = [-20.0_f64.to_radians(), 0.0, 20.0_f64.to_radians()];

        let fps = usable_footprints(&line(&look), &cloudy, 2, 3, 0.5);

        assert_eq!(fps.len(), 2); // the cloudy nadir column is dropped
        for f in &fps {
            assert!(f.elevation < FRAC_PI_2); // off-nadir => below zenith
        }
        // Signed look angles land on opposite sides of the northbound ground track (east vs west).
        assert!(fps[0].lon < 0.0 && fps[1].lon > 0.0, "{fps:?}");
    }

    #[test]
    fn usable_footprints_rejects_a_shape_mismatch() {
        assert_eq!(
            usable_footprints(&line(&[0.0]), &[false, false], 2, 3, 0.5).len(),
            0
        );
    }

    /// Each of the four shape guards on its own, and then the shapes that satisfy all four.
    ///
    /// `usable_footprints_rejects_a_shape_mismatch` trips two clauses at once, so no single one is
    /// observable: eight mutants of that line survived, including turning either `||` into `&&` and
    /// reading the pixel count as `rows + cols`. Here every fixture violates exactly one clause, and
    /// the last call violates none — without it the guard could fire unconditionally unnoticed.
    #[test]
    fn usable_footprints_rejects_each_shape_mismatch_on_its_own() {
        let look3 = [-0.1, 0.0, 0.1];
        // No rows to judge a column by. The other two clauses hold: 0 == 0 * 3, and 3 == 3.
        assert!(
            usable_footprints(&line(&look3), &[], 0, 3, 0.5).is_empty(),
            "rows"
        );
        // Nothing across track. Again the other clauses hold: 0 == 2 * 0, and 0 == 0.
        assert!(
            usable_footprints(&line(&[]), &[], 2, 0, 0.5).is_empty(),
            "cols"
        );
        // A mask that is not `rows * cols` — and 5 is neither 2 + 3 nor 2 / 3.
        assert!(
            usable_footprints(&line(&look3), &[false; 5], 2, 3, 0.5).is_empty(),
            "mask length"
        );
        // One look angle short of a full across-track line.
        assert!(
            usable_footprints(&line(&[0.0, 0.1]), &[false; 6], 2, 3, 0.5).is_empty(),
            "look angles"
        );
        // The same shapes, consistent: three clear columns, three footprints.
        assert_eq!(
            usable_footprints(&line(&look3), &[false; 6], 2, 3, 0.5).len(),
            3
        );
    }

    /// A column is usable when its clear count *reaches* `min_clear * rows`, counting down its own
    /// column of the mask.
    ///
    /// The original fixture is 2x3 with columns that are either wholly clear or wholly cloudy, so
    /// the threshold is never approached from either side and the row stride is never exercised
    /// against a different one. A 4x4 staircase gives the four columns clear counts 4, 2, 1 and 0
    /// against a cut of exactly 2: one comfortably above, one exactly on it, one below, one empty.
    /// Ten mutants of the two lines each move at least one column across the line.
    #[test]
    fn a_column_is_usable_when_its_clear_count_reaches_the_cut() {
        // rows 0..3 down, columns 0..3 across; `true` is cloudy.
        let cloudy = [
            false, true, true, true, //
            false, true, true, true, //
            false, false, true, true, //
            false, false, false, true,
        ];
        // Column 0 is clear in all 4 rows, column 1 in 2, column 2 in 1, column 3 in none; the cut
        // is 0.5 * 4 = 2, so columns 0 and 1 survive and column 1 sits exactly on it.
        let look = [-0.2, -0.1, 0.1, 0.2];
        let fps = usable_footprints(&line(&look), &cloudy, 4, 4, 0.5);
        assert_eq!(fps.len(), 2, "{fps:?}");
        // West of the ground track for both, and the further look angle lands further west.
        assert!(fps[0].lon < fps[1].lon && fps[1].lon < 0.0, "{fps:?}");
    }

    /// Each footprint sits at the look angle's ground range, ninety degrees off the ground track.
    ///
    /// The original fixture flies due north, so the across-track azimuth is `0 ± 90°` and reading
    /// it as `0 * (±90°)` gives the same nadir column back; the assertion is only that one longitude
    /// is negative and the other positive. Fly at a bearing instead and state the geometry as the
    /// triangle it is: for a satellite at radius `rs` looking `θ` off nadir at a target on a sphere
    /// of radius `Re`, the three interior angles are `θ` at the satellite, the earth-central angle
    /// `λ`, and `90° + ε` at the target, so
    ///
    ///     λ + θ + ε = 90°        (angle sum)
    ///     rs sin θ  = Re cos ε   (law of sines)
    ///
    /// with `λ` read back off the returned position as a great-circle distance. Neither identity
    /// restates the implementation; together they fix both `λ` and `ε` from `θ` alone.
    #[test]
    fn each_footprint_sits_at_its_look_angle_ninety_degrees_off_the_ground_track() {
        const ALT: f64 = 700_000.0;
        let (sub_lat, sub_lon, track) = (0.7, 0.17, 0.55);
        let look = [-0.4363, 0.4363];
        let scan = SwathLine {
            re: RE,
            altitude: ALT,
            sub_lat,
            sub_lon,
            ground_track_azimuth: track,
            look_angles: &look,
        };

        let fps = usable_footprints(&scan, &[false; 2], 1, 2, 0.0);
        assert_eq!(fps.len(), 2);
        for (f, &theta) in fps.iter().zip(&look) {
            let (dlon, sin_lat, cos_lat) =
                (f.lon - sub_lon, libm::sin(sub_lat), libm::cos(sub_lat));
            let (sin_f, cos_f) = (libm::sin(f.lat), libm::cos(f.lat));
            let lambda =
                libm::acos((sin_lat * sin_f + cos_lat * cos_f * libm::cos(dlon)).clamp(-1.0, 1.0));
            let bearing = libm::atan2(
                libm::sin(dlon) * cos_f,
                cos_lat * sin_f - sin_lat * cos_f * libm::cos(dlon),
            );

            let sum = lambda + libm::fabs(theta) + f.elevation;
            assert!((sum - FRAC_PI_2).abs() < 1e-12, "θ={theta}: λ+θ+ε = {sum}");
            let sines = (RE + ALT) * libm::sin(libm::fabs(theta)) - RE * libm::cos(f.elevation);
            assert!(
                libm::fabs(sines) < 1e-6,
                "θ={theta}: rs sinθ − Re cosε = {sines} m"
            );
            let want = track + libm::copysign(FRAC_PI_2, theta);
            assert!(
                (bearing - want).abs() < 1e-12,
                "θ={theta}: bearing {bearing} vs {want}"
            );
        }
    }

    /// The mask is the model's own probabilities against the cut, and the clear fraction counts what
    /// the mask left clear.
    ///
    /// Nine mutants survived because nothing in this crate ever ran the model: three constant
    /// returns from `mask`, three from `clear_fraction`, the `!` that separates clear from cloudy,
    /// and both arithmetic replacements of the ratio. The oracle is onnxruntime's output for the
    /// same fixture (`spacelab-vision/tests/fixtures`, from `gen_cloud_fixture.py`), so the
    /// expected mask is derived outside this code path, not from `mask` itself.
    #[test]
    fn the_cloud_mask_and_clear_fraction_follow_the_model_s_probabilities() {
        fn read_f32(bytes: &[u8]) -> Vec<f32> {
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        }
        const ONNX: &[u8] = include_bytes!("../../spacelab-vision/tests/fixtures/cloud_tiny.onnx");
        let input = read_f32(include_bytes!(
            "../../spacelab-vision/tests/fixtures/cloud_tiny_input.f32"
        ));
        let reference = read_f32(include_bytes!(
            "../../spacelab-vision/tests/fixtures/cloud_tiny_output.f32"
        ));
        // The fixture is a 3x8x8 tile through a net that halves the grid: 16 sigmoid outputs.
        assert_eq!(reference.len(), 16);
        let midpoint = 0.5;
        assert!(
            reference.iter().any(|&p| p >= midpoint) && reference.iter().any(|&p| p < midpoint),
            "the fixture must straddle the cut to say anything about a threshold"
        );

        for cut in [0.0, midpoint, 1.5] {
            let masker = CloudMasker::new(ONNX, cut).expect("fixture loads");
            let mask = masker.mask(&input, 3, 8, 8).expect("fixture runs");
            let want: Vec<bool> = reference.iter().map(|&p| p >= cut).collect();
            assert_eq!(mask, want, "cut {cut}");
            let clear = want.iter().filter(|&&cloudy| !cloudy).count();
            let fraction = masker
                .clear_fraction(&input, 3, 8, 8)
                .expect("fixture runs");
            #[allow(clippy::cast_precision_loss)]
            let expected = clear as f32 / 16.0;
            assert!((fraction - expected).abs() < 1e-7, "cut {cut}: {fraction}");
        }
        // Below every probability nothing is clear; above every probability everything is.
        let all_cloudy = CloudMasker::new(ONNX, 0.0).unwrap();
        assert_eq!(all_cloudy.clear_fraction(&input, 3, 8, 8).unwrap(), 0.0);
        let all_clear = CloudMasker::new(ONNX, 1.5).unwrap();
        assert_eq!(all_clear.clear_fraction(&input, 3, 8, 8).unwrap(), 1.0);
        // A tile the model cannot accept is an error, not a silent empty mask.
        assert!(all_clear.mask(&input, 3, 8, 9).is_err());
    }
}
