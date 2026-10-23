//! The empty / degenerate-input guards of the photogrammetry routines: each must return a neutral
//! result (None / zero / an offset-only fit) rather than index out of bounds, divide by zero, or invert
//! a singular matrix. The oracle here is the documented contract of each guard, not an external tool.

#![allow(clippy::float_cmp)] // the guarded returns are exact zeros / exact fallbacks
#![allow(clippy::unwrap_used)] // parsing the vendored fixture cannot fail

use spacelab_earthvision::{
    AdjustmentModel, DemGrid, GroundControlPoint, RationalPolynomialCamera,
    adjustment_residual_rms, brovey_pan_sharpen, depth_from_disparity, estimate_adjustment,
    gram_schmidt_pan_sharpen, ray_dem_intersection, sample_bilinear, triangulate,
};

const RPB: &str = include_str!("fixtures/pleiades_reunion.rpb");

#[test]
fn triangulation_of_fewer_than_two_views_is_none() {
    // A single ray (or none) does not fix a 3-D point — the DLT needs ≥2 views.
    assert!(triangulate(&[]).is_none());
}

#[test]
fn depth_from_a_non_positive_disparity_is_zero() {
    // Z = f·B/d only for d > 0; a zero/negative disparity means no valid intersection.
    assert_eq!(depth_from_disparity(0.0, 1000.0, 0.5), 0.0);
    assert_eq!(depth_from_disparity(-3.0, 1000.0, 0.5), 0.0);
    // Sanity: a positive disparity gives the finite depth.
    assert_eq!(depth_from_disparity(2.0, 1000.0, 0.5), 250.0);
}

#[test]
fn bilinear_sampling_outside_the_image_is_none() {
    let img = [1.0, 2.0, 3.0, 4.0]; // 2×2
    assert!(sample_bilinear(&img, 2, 2, -0.5, 0.0).is_none()); // row < 0
    assert!(sample_bilinear(&img, 2, 2, 0.0, -0.5).is_none()); // col < 0
    assert!(sample_bilinear(&img, 2, 2, 1.5, 0.0).is_none()); // row > rows−1
    assert!(sample_bilinear(&img, 2, 2, 0.0, 1.5).is_none()); // col > cols−1
    // In-bounds still samples (the corner value).
    assert_eq!(sample_bilinear(&img, 2, 2, 0.0, 0.0), Some(1.0));
}

#[test]
fn brovey_leaves_a_zero_sum_pixel_at_zero() {
    // Pixel 0 is zero in every band (Σ = 0) → left at zero; pixel 1 is scaled by PAN/ΣMS.
    let b0 = [0.0, 2.0];
    let b1 = [0.0, 3.0];
    let pan = [10.0, 10.0];
    let out = brovey_pan_sharpen(&[&b0, &b1], &pan);
    assert_eq!(out[0][0], 0.0);
    assert_eq!(out[1][0], 0.0);
    // Pixel 1: ΣMS = 5, scale = 10/5 = 2 → bands doubled.
    assert_eq!(out[0][1], 4.0);
    assert_eq!(out[1][1], 6.0);
}

#[test]
fn gram_schmidt_with_no_bands_is_empty() {
    let sharpened = gram_schmidt_pan_sharpen(&[], &[1.0, 2.0, 3.0]);
    assert_eq!(sharpened, [] as [Vec<f64>; 0]);
}

#[test]
fn a_single_gcp_affine_fit_falls_back_to_an_offset() {
    // One GCP under-determines the six-parameter affine: its normal matrix is rank-1 (singular), so
    // fit_axis must fall back to the mean residual — a pure shift that still zeroes that GCP's residual.
    let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
    let (lat, lon, h) = (cam.lat_offset, cam.lon_offset, cam.height_offset);
    let (r0, c0) = cam.forward(lat, lon, h);
    let gcp = GroundControlPoint {
        latitude: lat,
        longitude: lon,
        height: h,
        row: r0 + 12.5,
        col: c0 - 8.0,
    };
    let est = estimate_adjustment(&cam, &[gcp], AdjustmentModel::Affine);
    // The fallback keeps only the offset term; the linear (col, row) slopes are exactly zero.
    assert_eq!(est.row[1], 0.0);
    assert_eq!(est.row[2], 0.0);
    assert_eq!(est.col[1], 0.0);
    assert_eq!(est.col[2], 0.0);
    assert!((est.row[0] - 12.5).abs() < 1e-9);
    assert!((est.col[0] + 8.0).abs() < 1e-9);
    // And the offset absorbs the whole residual of its own GCP.
    assert!(adjustment_residual_rms(&cam, &[gcp], &est) < 1e-6);
}

#[test]
fn a_ray_that_never_reaches_the_dem_returns_none() {
    // A DEM held far above the search bracket has g(h) = h − DEM < 0 for every h in [h_min, h_max]: no
    // sign change, so the bisection cannot bracket a surface crossing.
    let cam = RationalPolynomialCamera::parse_rpb(RPB).unwrap();
    let dem = DemGrid {
        min_lat: cam.lat_offset - cam.lat_scale,
        min_lon: cam.lon_offset - cam.lon_scale,
        d_lat: cam.lat_scale,
        d_lon: cam.lon_scale,
        rows: 2,
        cols: 2,
        heights: vec![100_000.0; 4], // 100 km — above any real look-ray height
    };
    let hit = ray_dem_intersection(&cam, &dem, cam.line_offset, cam.samp_offset, 0.0, 3000.0);
    assert!(hit.is_none());
}
