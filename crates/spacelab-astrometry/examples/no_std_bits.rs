//! This crate's fused multiply-add kernels on a fixed sweep, printed as bit patterns.
//!
//! `just no-std` runs it with `std` and again without, and the two outputs must be equal. With `std`
//! the kernels call `f64::mul_add`, without it `libm::fma`; both are IEEE 754 fusedMultiplyAdd. The
//! route taken goes to stderr, so the recipe can check that the two runs took different ones.

use spacelab_astrometry::wcs::{Pixel, TanSip, deproject};

const SAMPLES: u32 = 20_000;

/// `SplitMix64`, as a fixed stream of uniform doubles in `[−1, 1)`.
fn stream(mut state: u64) -> impl FnMut() -> f64 {
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        2.0 * (f64::from_bits(0x3FF0_0000_0000_0000 | (z >> 12)) - 1.0) - 1.0
    }
}

fn bits(values: &[f64]) -> String {
    let words: Vec<String> = values
        .iter()
        .map(|v| format!(" {:016x}", v.to_bits()))
        .collect();
    words.concat()
}

fn main() {
    eprintln!(
        "route: {}",
        if cfg!(feature = "std") { "std" } else { "libm" }
    );
    let mut next = stream(0xA5_7120);
    for i in 0..SAMPLES {
        // A plate of about 2″ per pixel, rotated and sheared, with a second-order distortion.
        let (ra0, dec0) = (3.2 * (next() + 1.0), 1.5 * next());
        let scale = 1e-5 * (1.0 + 0.2 * next());
        let (rotation, shear) = (3.2 * next(), 0.05 * next());
        let (c, s) = (libm::cos(rotation), libm::sin(rotation));
        let distortion = [1e-10 * next(), 1e-10 * next(), 1e-10 * next()];
        let plate = TanSip {
            ra0,
            dec0,
            crpix: Pixel::new(1024.0 + 50.0 * next(), 1024.0 + 50.0 * next()),
            order: 2,
            xi: [
                &[1e-4 * next(), scale * c, scale * (shear - s)][..],
                &distortion[..],
            ]
            .concat(),
            eta: [
                &[1e-4 * next(), scale * s, scale * c][..],
                &[distortion[2], distortion[0], distortion[1]][..],
            ]
            .concat(),
        };
        let (xi, eta) = (0.02 * next(), 0.02 * next());
        let (ra, dec) = deproject(xi, eta, ra0, dec0);
        let pixel = Pixel::new(1024.0 + 900.0 * next(), 1024.0 + 900.0 * next());
        let (sky_ra, sky_dec) = plate.pixel_to_sky(pixel);
        let back = plate
            .sky_to_pixel(sky_ra, sky_dec)
            .map_or([f64::NAN; 2], |p| [p.x, p.y]);
        println!(
            "astrometry {i}{}{}{}",
            bits(&[ra, dec, sky_ra, sky_dec]),
            bits(&back),
            bits(&[plate.plate_scale()]),
        );
    }
}
