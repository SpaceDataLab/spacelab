//! H3 held to its definition, the grid constructed in Maxima from the constants that fix it.
//!
//! `derivations/h3_definition.mac` proves the constants' geometry (a regular icosahedron, axes on
//! the vertices, the resolution-0 unit `(3 - sqrt 5)/2`, the aperture-7 turn `atan(sqrt 3/5)`), then
//! constructs each sample point's cell centre at 40 digits: the closest face, the gnomonic plane, the
//! nearest lattice point, back to the sphere. Every entry below is that value rounded to the digits
//! shown, checked by the derivation. The points lie 1 to 7 degrees from a face centre, on every
//! resolution from 0 to 15, at least 2e-3 lattice units inside their cells.

// The literals are Maxima's 17 significant digits, read back by the derivation as written; the
// resolution column holds the whole numbers 0 to 15.
#![allow(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::expect_used,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use spacelab_core::{Angle, Length};
use spacelab_earth::Geodetic;
use spacelab_grid::{Resolution, cell_center, disk_cells};

/// Latitude and longitude in degrees, resolution, then the cell centre's latitude and longitude.
#[rustfmt::skip]
const H3_MAXIMA: [[f64; 5]; 24] = [
    [47.0418730813, 71.5373789992, 0.0, 4.6041894318837706e1, 7.1527903299099242e1],
    [24.3034786537, -106.7338093017, 1.0, 2.4486526988568927e1, -1.0822463430197651e2],
    [-26.1713203923, 71.2942664391, 2.0, -2.6946628939786627e1, 6.9639153041821503e1],
    [75.5660499423, 137.4500197543, 3.0, 7.5182971734892882e1, 1.3747842029024803e2],
    [-2.3472049119, -40.8433055437, 4.0, -2.359022542923487e0, -4.0810874969264015e1],
    [-36.0048146475, 157.0550873808, 5.0, -3.601902371560651e1, 1.5715633915315664e2],
    [57.9869034103, -81.3018698544, 6.0, 5.7963112876881077e1, -8.1254673068891654e1],
    [-10.4982180622, 26.6434289765, 7.0, -1.0491436979072679e1, 2.6650824596153089e1],
    [-25.9314554924, -153.3018330977, 8.0, -2.5930192367496193e1, -1.5330249668942055e2],
    [30.4969512411, -23.5668325662, 9.0, 3.0497512226758564e1, -2.3565222382145137e1],
    [2.9392650356, 133.5454597287, 10.0, 2.939198310500677e0, 1.3354547415205101e2],
    [-41.0846301321, -109.9411077306, 11.0, -4.1084688667602949e1, -1.0994135610435717e2],
    [28.5686221201, 29.1991621138, 12.0, 2.856869173545746e1, 2.9199234270583427e1],
    [7.4247550462, -151.1197849745, 13.0, 7.4247666013012148e0, -1.5111978548689525e2],
    [-72.9239013895, -55.9102577200, 14.0, -7.2923896936940203e1, -5.591028269903254e1],
    [16.2076244638, 97.9550364089, 15.0, 1.6207625928988791e1, 9.7955034740161698e1],
    [-12.4628861105, -77.2972478189, 0.0, -9.8975781915205022e0, -8.3849266070406424e1],
    [-61.5505909522, 101.4648492813, 1.0, -6.0432795263055544e1, 1.0279294251439185e2],
    [35.6905453961, 167.6016006559, 2.0, 3.5779335226174033e1, 1.6565042124233232e2],
    [-33.2301139863, -9.0862413381, 3.0, -3.3179169782249696e1, -8.5914428395278381e0],
    [44.1731542602, 73.6620535764, 4.0, 4.4266626417285588e1, 7.3446760075949987e1],
    [22.9719425566, -110.7582205014, 5.0, 2.2993341965149375e1, -1.1070154939334137e2],
    [-21.6485672520, 70.3146325236, 6.0, -2.1626165855151489e1, 7.0316754380201919e1],
    [75.6676248879, 158.9100824130, 7.0, 7.5672725728160477e1, 1.5888596506340851e2],
];

/// Degrees, on latitude and on longitude times cos(latitude): h3o's doubles through the projection and
/// back, 4.3e-14 at worst on these rows. The nearest wrong cell is 4e-6 degrees away, at resolution 15.
const TOL: f64 = 1e-13;

#[test]
fn h3_cell_centres_are_the_grid_constructed_by_its_definition() {
    let mut worst = 0.0_f64;
    for row in H3_MAXIMA {
        let [lat, lon, res, centre_lat, centre_lon] = row;
        let point = Geodetic {
            latitude: Angle::from_degrees(lat),
            longitude: Angle::from_degrees(lon),
            height: Length::from_metres(0.0),
        };
        let resolution = Resolution::try_from(res as u8).expect("a resolution 0..=15");
        let cells = disk_cells(point, 0, resolution).expect("a valid point");
        assert_eq!(cells.len(), 1);
        let centre = cell_center(cells[0]);
        let d_lat = (centre.latitude.degrees() - centre_lat).abs();
        let d_lon = (centre.longitude.degrees() - centre_lon).abs() * centre_lat.to_radians().cos();
        worst = worst.max(d_lat).max(d_lon);
        assert!(
            d_lat <= TOL && d_lon <= TOL,
            "{row:?}: h3o's centre {} {} is {d_lat:e} {d_lon:e} degrees off",
            centre.latitude.degrees(),
            centre.longitude.degrees()
        );
    }
    println!("largest difference {worst:e} degrees");
}

/// `n(r) = 2 + 120 7^r`, the count `h3_definition.mac` derives from resolution 0's 122 cells and the
/// 7 children of a hexagon, 6 of a pentagon.
#[test]
fn h3_cell_counts_are_two_plus_120_times_seven_to_the_resolution() {
    for r in 0..=15_u8 {
        let resolution = Resolution::try_from(r).expect("a resolution 0..=15");
        assert_eq!(
            resolution.cell_count(),
            2 + 120 * 7_u64.pow(u32::from(r)),
            "resolution {r}"
        );
    }
}
