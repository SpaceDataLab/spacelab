//! The WAM-IPE reader on a real product file, against an independent reader of the same bytes.
//!
//! The unit tests in `wam_ipe.rs` and `netcdf_classic.rs` run on files this crate's own test writer
//! made, so they cannot catch a misreading of the format that the writer shares. This one reads a
//! file NOAA wrote — `v1.2/wfs.20240502/12/…20240502_120000.nc`, pinned in `tools/wam-ipe.sha256`
//! as `34d872a4…` — and compares with SciPy 1.16.2's `scipy.io.netcdf_file`, a NetCDF classic
//! reader that shares no code with this one. The node values are SciPy's `den[k, j, i]` printed in
//! full; the off-grid values are the same trilinear interpolation in `ln ρ`, written separately in
//! Python over SciPy's array.

#![allow(clippy::expect_used, clippy::doc_markdown)]

use std::path::PathBuf;

use spacelab_propagation::wam_ipe::WamIpeField;

const KEY: &str = "v1.2/wfs.20240502/12/wam_fixed_height.wfs.t12z.wam10.20240502_120000.nc";

fn field() -> WamIpeField {
    let dir = std::env::var_os("WAM_IPE_DIR").map_or_else(
        || PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/spacelab/wam-ipe"),
        PathBuf::from,
    );
    let bytes = std::fs::read(dir.join(KEY)).expect("tools/fetch-wam-ipe.sh has run");
    WamIpeField::parse(&bytes).expect("a WAM-IPE fixed-height file")
}

/// The file's epoch is the hour its name says, and its heights are the fixed-height product's
/// 100–1000 km grid. (G5: the product's own naming and grid, not another reader.)
#[test]
#[ignore = "needs tools/fetch-wam-ipe.sh"]
fn the_noaa_file_is_the_hour_and_the_heights_its_product_names() {
    let f = field();
    // `time` is 19845.5 days since 1970-01-01: 2024-05-02 12:00 UTC, the hour in the file's name.
    assert_eq!(f.unix_seconds(), 19_845.5 * 86_400.0);
    assert_eq!(f.height_range(), (100.0, 1000.0));
}

/// The reader's density on NOAA's bytes, against SciPy's `netcdf_file` on the same bytes at grid
/// nodes, and against the trilinear interpolation in `ln ρ` written in Python over SciPy's array off
/// the grid.
#[test]
#[ignore = "comparator: the distance to scipy, not a verdict; needs tools/fetch-wam-ipe.sh"]
fn the_noaa_file_distance_to_scipy() {
    let f = field();
    // (height km, latitude °, longitude °, SciPy's den) at grid nodes, corners and edges included.
    let nodes = [
        (100.0, -88.0, 0.0, 2.730_179_744_503_402_6e-7),
        (400.0, 0.0, 0.0, 6.987_265_800_673_104e-12),
        (500.0, 0.0, 180.0, 8.418_623_328_836_083e-13),
        (490.0, 88.0, 356.0, 1.616_787_324_680_568_6e-12),
        (1000.0, -88.0, 68.0, 8.847_622_511_205_539e-15),
        (570.0, 44.0, 92.0, 5.969_006_213_780_992e-13),
    ];
    // Off the grid: inside a cell, poleward of the last full ring, and across the meridian.
    let points = [
        (493.7, 12.3, 201.9, 9.190_292_287_352_067e-13),
        (455.2, -71.6, 358.7, 2.929_362_219_772_517_4e-12),
        (512.0, 88.9, 3.0, 1.220_104_841_146_045e-12),
        (430.0, -45.0, -10.0, 4.514_371_925_448_156e-12),
    ];
    for (quantity, cases) in [
        ("WAM-IPE density at grid nodes", &nodes[..]),
        ("WAM-IPE density off the grid", &points[..]),
    ] {
        let distances: Vec<f64> = cases
            .iter()
            .map(|&(h, lat, lon, scipy)| {
                (f.density(h, lat, lon).expect("inside the product") / scipy - 1.0).abs()
            })
            .collect();
        let max = distances.iter().copied().fold(0.0, f64::max);
        #[expect(clippy::cast_precision_loss, reason = "a few distances")]
        let rms = (distances.iter().map(|d| d * d).sum::<f64>() / distances.len() as f64).sqrt();
        println!(
            "comparison | scipy | 1.16.2 | {quantity} | relative | {} | {max:.3e} | {rms:.3e}",
            distances.len()
        );
    }
}
