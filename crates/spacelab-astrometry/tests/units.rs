//! The arithmetic each unit does, asserted directly.
//!
//! The end-to-end tests in `plate_solve.rs` and `reduction.rs` say the chain produces the right
//! answer. They do not say that every piece of it is doing what its documentation claims, and a
//! mutation run over the crate found where the difference lies: an indexing helper nothing calls, a
//! second moment nothing reads, a median whose even branch is invisible on a sixteen-thousand-pixel
//! frame, a wrap no test ever pushes past zero, a detection cut whose `+` and `×` agree to a count
//! when the noise happens to be near unity. Each test below is one of those, pinned against a value
//! computed by hand or by an expression written independently of the one under test.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    // Every exact comparison here is against an integer-valued or literal quantity written out in
    // this file — pixel indices, a median of four small integers, a frame's own row stride.
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

use spacelab_astrometry::CatalogueStar;
use spacelab_astrometry::quads::{Quad, QuadMatch, build_quads, quad_code, star_correspondences};
use spacelab_astrometry::sources::{Frame, Source, background, detect};
use spacelab_astrometry::wcs::{
    Correspondence, Pixel, PlateSolution, TanSip, deproject, term_count,
};

const ARCSEC: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// `1.4826`, the factor that makes a median absolute deviation agree with a Gaussian sigma:
/// `1/Φ⁻¹(3/4)` as `derivations/source_moments.mac` prints it. Written out rather than imported so
/// that a change to the constant in the crate shows up here as a failure instead of propagating
/// silently into every expectation below.
const MAD_TO_SIGMA: f64 = 1.482_602_218_505_601_860_547_076_529_360_423_431_326_703_2;

/// xorshift64, so the jitter is the same on every machine and every run.
struct Xorshift(u64);

impl Xorshift {
    /// Uniform on `[-half_width, half_width)`.
    fn signed(&mut self, half_width: f64) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 11) as f64 / 9_007_199_254_740_992.0;
        half_width.mul_add(2.0 * unit, -half_width)
    }
}

// ---------------------------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------------------------

#[test]
fn the_frame_index_is_row_major_and_bounded_on_both_axes() {
    // `Frame::at` is the crate's only public random access to a frame, and nothing called it:
    // `detect` walks the buffer by index. Both terms of `y·width + x` and both bounds are
    // separately observable here.
    let pixels: Vec<f64> = (0..12).map(f64::from).collect();
    let frame = Frame {
        pixels: &pixels,
        width: 4,
        height: 3,
    };
    assert!(frame.is_consistent());

    assert_eq!(frame.at(0, 0), Some(0.0));
    assert_eq!(
        frame.at(3, 0),
        Some(3.0),
        "the last column of the first row"
    );
    assert_eq!(frame.at(0, 1), Some(4.0), "one row down is one stride on");
    assert_eq!(frame.at(2, 2), Some(10.0), "y·width + x, both terms");
    assert_eq!(frame.at(3, 2), Some(11.0), "the last pixel");

    // Both bounds are exclusive, and both are checked.
    assert_eq!(frame.at(4, 0), None, "x = width is outside");
    assert_eq!(frame.at(0, 3), None, "y = height is outside");
    assert_eq!(frame.at(4, 3), None);
    assert_eq!(frame.at(usize::MAX, 0), None);

    // And a frame whose stated shape does not match its buffer is rejected rather than read.
    assert!(
        !Frame {
            pixels: &pixels,
            width: 5,
            height: 3
        }
        .is_consistent()
    );
}

#[test]
fn a_sources_signal_to_noise_is_its_width_over_its_position_error() {
    // `sigma = width/Q` inverted. It is reported rather than recomputed by the caller because both
    // factors are measured from the blob and neither is recoverable from the other fields.
    let source = Source {
        centroid: Pixel::new(3.0, 4.0),
        flux: 1000.0,
        peak: 250.0,
        pixel_count: 9,
        width: 2.0,
        position_sigma: 0.5,
    };
    assert_eq!(source.signal_to_noise(), 4.0);

    // A perfectly measured position is an infinite signal-to-noise, and so is a nonsensical one:
    // the guard covers the whole non-positive half-line, not zero alone.
    for sigma in [0.0, -0.0, -1.0] {
        let degenerate = Source {
            position_sigma: sigma,
            ..source
        };
        assert_eq!(
            degenerate.signal_to_noise(),
            f64::INFINITY,
            "sigma {sigma} should not have divided"
        );
    }
}

#[test]
fn the_background_median_takes_the_pair_that_straddles_the_middle() {
    assert_eq!(background(&[]), None, "an empty frame has no background");

    // Odd: the middle element itself. Deviations |10−2| = 8, |1−2| = 1, |2−2| = 0, median 1.
    let (level, noise) = background(&[10.0, 1.0, 2.0]).unwrap();
    assert_eq!(level, 2.0);
    assert!((noise - MAD_TO_SIGMA).abs() < 1e-15, "{noise}");

    // Even: the mean of the two elements that *straddle* the middle, which on a small set is
    // nothing like either neighbour. Sorted, this is 1, 2, 10, 100, so the median is (2 + 10)/2 = 6
    // — taking the pair one place up gives 55 and taking the same element twice gives 10. On the
    // sixteen-thousand-pixel frames the detection tests use, all three agree to four decimals; that
    // is why those tests cannot see the difference and this one can.
    let (level, noise) = background(&[1.0, 2.0, 10.0, 100.0]).unwrap();
    assert_eq!(level, 6.0);
    // Deviations 5, 4, 4, 94; sorted 4, 4, 5, 94; median (4 + 5)/2 = 4.5.
    assert!((noise - MAD_TO_SIGMA * 4.5).abs() < 1e-14, "{noise}");

    // A frame with no spread has no scale to threshold against, and says so rather than inventing
    // one — `detect` turns that into `None` instead of a cut everything clears.
    assert_eq!(background(&[7.0; 16]), Some((7.0, 0.0)));
    let flat = [7.0; 16];
    assert!(
        detect(
            &Frame {
                pixels: &flat,
                width: 4,
                height: 4
            },
            5.0,
            1
        )
        .is_none()
    );
}

/// A frame with an exactly known median and MAD: two fifths of the pixels at `level`, the rest split
/// between `level ± amplitude`.
///
/// The plateaux either side of the middle index are a hundred pixels deep, so overwriting a handful
/// of them with a source moves neither statistic — which is what lets the tests below state the
/// detection cut in closed form instead of asserting a bound on it.
fn mottled(width: usize, height: usize, level: f64, amplitude: f64) -> Vec<f64> {
    (0..width * height)
        .map(|i| match i % 10 {
            0..=3 => level,
            4..=6 => level + amplitude,
            _ => level - amplitude,
        })
        .collect()
}

#[test]
fn the_detection_cut_is_the_background_plus_a_multiple_of_its_noise() {
    // The cut is `level + threshold·sigma`. On the frames the end-to-end tests use — sky 1000,
    // sigma near 1, threshold 5 — the product `5σ` and the *sum* `5 + σ` differ by one count in a
    // thousand and detect exactly the same stars. Here sigma is 14.826, so the readings of that one
    // expression sit at 44.5, 17.8 and 0.2 counts above the sky, and a block placed between them
    // separates them.
    let (width, height) = (32, 32);
    let (level, amplitude) = (500.0, 10.0);
    let mut pixels = mottled(width, height, level, amplitude);

    let noise = MAD_TO_SIGMA * amplitude;
    let cut = 3.0f64.mul_add(noise, level);
    assert!((cut - 544.478_066_555_168_1).abs() < 1e-9, "{cut}");

    // Well above the cut: 570 to 590, whatever the pixel's background value.
    for row in 15..18 {
        for column in 15..18 {
            pixels[row * width + column] += 80.0;
        }
    }
    // Between the cut and the cut a *sum* would give: 520 to 540, against 544.478 and 517.826.
    for row in 25..28 {
        for column in 5..8 {
            pixels[row * width + column] += 30.0;
        }
    }

    let frame = Frame {
        pixels: &pixels,
        width,
        height,
    };
    let (measured_level, measured_noise) = background(&pixels).unwrap();
    assert_eq!(measured_level, level, "the mottled median is the level");
    assert!((measured_noise - noise).abs() < 1e-12, "{measured_noise}");

    let sources = detect(&frame, 3.0, 4).unwrap();
    assert_eq!(
        sources.len(),
        1,
        "only the 80-count block clears level + 3 sigma: {sources:#?}"
    );
    assert!((sources[0].centroid.x - 16.0).abs() < 0.2);
    assert!((sources[0].centroid.y - 16.0).abs() < 0.2);

    // Dropping the threshold to 1 puts the cut at 514.8 — below the fainter block and still above
    // every background pixel — and both are found. That is what says the first result was the cut
    // and not a failure to see the second block at all.
    let both = detect(&frame, 1.0, 4).unwrap();
    assert_eq!(
        both.len(),
        2,
        "at level + 1 sigma = 514.8 both blocks clear"
    );
    assert!(both[0].flux > both[1].flux, "brightest first");
    assert!((both[1].centroid.x - 6.0).abs() < 0.2);
    assert!((both[1].centroid.y - 26.0).abs() < 0.2);
}

#[test]
fn a_pixel_exactly_at_the_cut_is_below_it() {
    // The neighbour test is strict, so a pixel sitting exactly on the cut is background. It is the
    // one place in the flood fill where `>` and `>=` differ, and what they differ by is a pixel of
    // flux and a shifted centroid.
    let (width, height) = (16, 16);
    let (level, amplitude) = (100.0, 4.0);
    let mut pixels = mottled(width, height, level, amplitude);
    let cut = 5.0f64.mul_add(MAD_TO_SIGMA * amplitude, level);

    for (x, y) in [(8, 8), (9, 8), (8, 9), (9, 9)] {
        pixels[y * width + x] = cut + 50.0;
    }
    pixels[8 * width + 10] = cut;

    let frame = Frame {
        pixels: &pixels,
        width,
        height,
    };
    assert_eq!(background(&pixels), Some((level, MAD_TO_SIGMA * amplitude)));

    let sources = detect(&frame, 5.0, 4).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(
        sources[0].pixel_count, 4,
        "the pixel exactly at the cut was counted in"
    );
    // The centroid of the 2×2 block. A fifth pixel at (10, 8) would drag it to 8.63.
    assert_eq!(sources[0].centroid, Pixel::new(8.5, 8.5));
}

#[test]
fn a_blob_touching_the_frame_edge_is_found_without_walking_off_it() {
    // The flood fill clamps its neighbourhood to `width − 1` and `height − 1`. A blob in the middle
    // of a frame never reaches either clamp; one in a corner reaches both, and an off-by-one there
    // reads past the buffer.
    let (width, height) = (16, 16);
    let (level, amplitude) = (100.0, 4.0);
    let mut pixels = mottled(width, height, level, amplitude);
    for (x, y) in [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (14, 14),
        (15, 14),
        (14, 15),
        (15, 15),
    ] {
        pixels[y * width + x] = 1000.0;
    }
    let frame = Frame {
        pixels: &pixels,
        width,
        height,
    };
    let sources = detect(&frame, 5.0, 4).unwrap();
    assert_eq!(sources.len(), 2, "both corner blobs: {sources:#?}");
    for source in &sources {
        assert_eq!(source.pixel_count, 4);
    }
    let corners: Vec<Pixel> = sources.iter().map(|source| source.centroid).collect();
    assert!(corners.contains(&Pixel::new(0.5, 0.5)), "{corners:?}");
    assert!(corners.contains(&Pixel::new(14.5, 14.5)), "{corners:?}");
}

#[test]
fn the_reported_width_and_position_error_are_the_blobs_own_moments() {
    // The second moment, the width derived from it and the `width/Q` uncertainty are computed by
    // the centroider and read by nothing else in the crate. Here all three are asserted against the
    // same quantities worked out from the frame.
    //
    // A 2×4 block of *equal* pixels: columns 9 to 12 of rows 4 and 5, so the centroid is
    // (10.5, 4.5) and the second moment about it is
    // `2·[(±1.5)² + (±0.5)²] + 8·(0.5)²  =  10 + 2 = 12` per unit of weight.
    //
    // Neither square nor centred on the diagonal, and that is the whole point. The reported width
    // comes from the *trace* of the moment matrix, which is invariant under `x <-> y`; a blob that
    // is symmetric about the diagonal with `cx == cy` therefore gives the same trace whichever way
    // the row-major index is decoded, and reading the column from `index / width` and the row from
    // `index % width` is invisible. Here the transposed decode gives 588 rather than 12.
    let (width, height) = (16, 16);
    let (level, amplitude) = (100.0, 4.0);
    let excess = 900.0;
    let build = |scale: f64| -> Vec<f64> {
        let mut pixels = mottled(width, height, level, amplitude);
        for row in 4..6 {
            for column in 9..13 {
                pixels[row * width + column] = scale.mul_add(excess, level);
            }
        }
        pixels
    };

    let pixels = build(1.0);
    let frame = Frame {
        pixels: &pixels,
        width,
        height,
    };
    let noise = MAD_TO_SIGMA * amplitude;
    assert_eq!(background(&pixels), Some((level, noise)));

    let sources = detect(&frame, 5.0, 4).unwrap();
    assert_eq!(sources.len(), 1);
    let source = &sources[0];

    assert_eq!(source.pixel_count, 8);
    assert_eq!(source.centroid, Pixel::new(10.5, 4.5));
    assert_eq!(source.flux, 8.0 * excess, "eight pixels of {excess} each");
    assert_eq!(source.peak, excess, "the brightest single pixel");

    // Sum v·r² = 12·excess, and `width = sqrt(0.5·Sum vr²/Sum v)` = sqrt(12/16).
    let want_width = 0.75_f64.sqrt();
    assert!(
        (source.width - want_width).abs() < 1e-14,
        "width {} is not sqrt(3/4) = {want_width}",
        source.width
    );

    // Q = flux/(sigma·sqrt(n)), n the pixels the flux was summed over.
    let want_q = 8.0 * excess / (noise * 8.0_f64.sqrt());
    assert!(
        (source.signal_to_noise() - want_q).abs() < 1e-9 * want_q,
        "Q is {}, not {want_q}",
        source.signal_to_noise()
    );
    assert!(
        (source.position_sigma - want_width / want_q).abs() < 1e-15,
        "sigma is {}, not width/Q = {}",
        source.position_sigma,
        want_width / want_q
    );

    // And the reason it is reported at all: four times the flux at the same shape is a quarter of
    // the position error.
    let brighter_pixels = build(4.0);
    let brighter = Frame {
        pixels: &brighter_pixels,
        width,
        height,
    };
    let bright = &detect(&brighter, 5.0, 4).unwrap()[0];
    assert_eq!(bright.width, source.width, "the shape did not change");
    assert!(
        (bright.position_sigma * 4.0 / source.position_sigma - 1.0).abs() < 1e-12,
        "{} against {}",
        bright.position_sigma,
        source.position_sigma
    );
}

#[test]
fn a_source_whose_signal_underflows_reports_an_infinite_position_error_not_a_nan() {
    // `flux/(sigma·sqrt(n))` is positive for every source the detector can build — the flux is
    // guarded positive and the noise is guarded positive and finite — *except* where the quotient
    // underflows, and the guard that catches that is strict for a reason. A `>=` there would divide
    // by an exact zero and, for a single-pixel blob whose width is also zero, hand back a NaN
    // position error. A NaN sigma does not fail loudly: it propagates into a plate fit's weights and
    // poisons every star in the frame.
    //
    // The frame is deliberately extreme, because that is the only place the two forms differ. Nine
    // pixels at −1e300, nine at +1e300 and eight at zero put the median at 0 and the MAD at 1e300;
    // the single denormal in the middle is then a source at `threshold = 0`, with a flux of 5e-324
    // against a noise of 1.5e300, whose ratio is below the smallest representable double.
    let (width, height) = (3, 9);
    let huge = 1e300;
    let mut pixels = vec![0.0; width * height];
    for i in 0..9 {
        pixels[i] = -huge;
        pixels[width * height - 1 - i] = huge;
    }
    pixels[4 * width + 1] = f64::MIN_POSITIVE * f64::EPSILON; // the smallest denormal, 4.94e-324
    let frame = Frame {
        pixels: &pixels,
        width,
        height,
    };

    let (level, noise) = background(&pixels).unwrap();
    assert_eq!(level, 0.0, "the median is the zero band");
    assert_eq!(noise, MAD_TO_SIGMA * huge);

    let sources = detect(&frame, 0.0, 1).unwrap();
    assert_eq!(sources.len(), 2, "the block of nine, and the denormal");
    let faint = &sources[1];
    assert_eq!(faint.pixel_count, 1);
    assert_eq!(faint.centroid, Pixel::new(1.0, 4.0));
    assert_eq!(faint.width, 0.0, "one pixel has no second moment");
    assert!(
        faint.position_sigma.is_infinite(),
        "sigma is {}, and a NaN here would silently poison a plate fit",
        faint.position_sigma
    );
}

// ---------------------------------------------------------------------------------------------
// wcs
// ---------------------------------------------------------------------------------------------

/// A one-arcsecond, unrotated plate about `(ra0, dec0)` with its reference pixel at `crpix`, fitted
/// to `count` stars on a golden-angle spiral whose measured pixels carry `jitter` pixels of
/// deterministic scatter.
fn spiral_plate(
    ra0: f64,
    dec0: f64,
    crpix: Pixel,
    count: usize,
    jitter: f64,
) -> (Vec<Correspondence>, Option<PlateSolution>) {
    let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let mut noise = Xorshift(0x9E37_79B9_7F4A_7C15);
    let correspondences: Vec<Correspondence> = (0..count)
        .map(|i| {
            let radius = 500.0 * ((i as f64 + 0.5) / count as f64).sqrt();
            let theta = i as f64 * golden;
            let (u, v) = (radius * theta.cos(), radius * theta.sin());
            let (ra, dec) = deproject(ARCSEC * u, ARCSEC * v, ra0, dec0);
            Correspondence {
                pixel: Pixel::new(
                    crpix.x + u + noise.signed(jitter),
                    crpix.y + v + noise.signed(jitter),
                ),
                right_ascension: ra,
                declination: dec,
            }
        })
        .collect();
    let solution = PlateSolution::fit(&correspondences, ra0, dec0, crpix, 1);
    (correspondences, solution)
}

#[test]
fn a_pixels_direction_is_the_unit_vector_of_its_sky_position() {
    // `pixel_to_direction` is what an orbit determination consumes, and nothing called it. Every
    // product in `(cos d cos a, cos d sin a, sin d)` is pinned, at a tangent point where no factor
    // is 1 or 0 and a wrong operator therefore shows.
    let wcs = TanSip {
        ra0: 1.0,
        dec0: 0.5,
        crpix: Pixel::new(0.0, 0.0),
        order: 1,
        xi: vec![0.0, ARCSEC, 0.0],
        eta: vec![0.0, 0.0, ARCSEC],
    };
    let at_reference = wcs.pixel_to_direction(Pixel::new(0.0, 0.0));
    let (cos_dec, sin_dec) = (0.5_f64.cos(), 0.5_f64.sin());
    for (component, want) in [
        (at_reference.x, cos_dec * 1.0_f64.cos()),
        (at_reference.y, cos_dec * 1.0_f64.sin()),
        (at_reference.z, sin_dec),
    ] {
        assert!(
            (component - want).abs() < 1e-14,
            "{component} against {want}"
        );
    }
    assert!((at_reference.norm() - 1.0).abs() < 1e-14);

    // Away from the reference pixel it is still the direction of `pixel_to_sky`, and it has moved.
    let probe = Pixel::new(1000.0, -700.0);
    let (ra, dec) = wcs.pixel_to_sky(probe);
    let direction = wcs.pixel_to_direction(probe);
    assert!((direction.x - dec.cos() * ra.cos()).abs() < 1e-14);
    assert!((direction.y - dec.cos() * ra.sin()).abs() < 1e-14);
    assert!((direction.z - dec.sin()).abs() < 1e-14);
    assert!(
        direction.dot(at_reference) < 1.0 - 1e-12,
        "the probe pixel did not move off the tangent point"
    );
}

#[test]
fn right_ascension_comes_back_wrapped_into_one_turn() {
    // A plate whose tangent point is at alpha = 0 puts half its frame at a *negative* right
    // ascension before wrapping, and nothing else in this crate's tests crosses that line — so
    // neither the branch nor the turn it adds was ever exercised.
    let wcs = TanSip {
        ra0: 0.0,
        dec0: 0.0,
        crpix: Pixel::new(0.0, 0.0),
        order: 1,
        xi: vec![0.0, ARCSEC, 0.0],
        eta: vec![0.0, 0.0, ARCSEC],
    };
    let turn = core::f64::consts::TAU;

    // Exactly at the tangent point the answer is zero, not a whole turn.
    let (at_zero, _) = wcs.pixel_to_sky(Pixel::new(0.0, 0.0));
    assert_eq!(at_zero, 0.0, "alpha = 0 came back as {at_zero}");

    // A hundred arcseconds to the negative side wraps to just under a turn. The tolerance is the
    // gnomonic xi³/3, 3.8e-11 rad at this offset — the projection's own curvature, not the wrap's.
    let (wrapped, _) = wcs.pixel_to_sky(Pixel::new(-100.0, 0.0));
    assert!(
        (wrapped - (turn - 100.0 * ARCSEC)).abs() < 1e-9,
        "alpha = {wrapped}, not one turn less 100 arcsec"
    );
    assert!((0.0..turn).contains(&wrapped));

    // And the positive side is left alone.
    let (positive, _) = wcs.pixel_to_sky(Pixel::new(100.0, 0.0));
    assert!((positive - 100.0 * ARCSEC).abs() < 1e-9, "{positive}");
}

#[test]
fn a_plate_whose_linear_part_is_not_invertible_has_no_inverse_map() {
    // `sky_to_pixel` seeds its Newton iteration by inverting the linear part, and refuses when that
    // determinant is zero *or* non-finite. The two halves of that guard reject different plates and
    // both have to be there:
    //
    //  - a singular linear part (the second row a multiple of the first) divides by zero, and the
    //    iteration then walks on NaN and never meets its convergence test;
    //  - a determinant that *overflows* does not. Dividing by an infinity gives an exact zero, so
    //    the very first step reports a correction of zero, passes the convergence test, and returns
    //    the reference pixel for every position on the sky.
    let singular = TanSip {
        ra0: 1.0,
        dec0: 0.3,
        crpix: Pixel::new(100.0, 200.0),
        order: 1,
        // Powers of two, so that `a1·b2 − a2·b1` is exactly zero rather than a rounding away from
        // it: the guard under test is `== 0.0`, and a plate scale in radians per pixel would leave
        // the fused multiply-add a fraction of an ulp short of the subtraction it is compared to.
        xi: vec![0.0, 1.0, 2.0],
        eta: vec![0.0, 2.0, 4.0],
    };
    assert!(singular.sky_to_pixel(1.0, 0.3).is_none());

    let overflowing = TanSip {
        xi: vec![0.0, 1e300, 0.0],
        eta: vec![0.0, 0.0, 1e300],
        ..singular.clone()
    };
    assert!(
        !overflowing.plate_scale().is_finite(),
        "the determinant has to overflow for this to be the case under test"
    );
    assert!(
        overflowing.sky_to_pixel(1.0, 0.3).is_none(),
        "an overflowing determinant returned a pixel"
    );

    // And the far hemisphere is refused before any of that: the gnomonic projection has no image
    // there, so there is nothing to invert.
    assert!(
        singular
            .sky_to_pixel(1.0 + core::f64::consts::PI, -0.3)
            .is_none()
    );
}

#[test]
fn the_inverse_map_converges_on_both_axes_before_it_reports_a_pixel() {
    // The Newton loop stops when *both* corrections are under a hundredth of a pixel, and the
    // conjunction is load-bearing: the two axes need not converge together, and stopping when either
    // one has leaves the other wherever it happened to be.
    //
    // This plate makes the asymmetry total. `eta` is exactly linear in `v`, so the very first solve
    // lands `v` on its answer and every `dv` after it is exactly zero; `xi` carries a `u²` term
    // worth ten pixels at the frame edge, so `du` starts at ten pixels and contracts by about a
    // factor of a hundred per iteration. Waiting for both gives 9.3e-6 px; stopping when either is
    // small returns at the first iteration, 9.8e-2 px out — four orders worse, and still inside the
    // "hundredth of a pixel" the function's documentation promises, which is why a round-trip test
    // with a loose tolerance cannot see it.
    let scale = ARCSEC;
    let quadratic = 10.0 * scale / (2048.0 * 2048.0);
    let wcs = TanSip {
        ra0: 1.0,
        dec0: 0.3,
        crpix: Pixel::new(0.0, 0.0),
        order: 2,
        // [1, u, v, u², uv, v²]
        xi: vec![0.0, scale, 0.0, quadratic, 0.0, 0.0],
        eta: vec![0.0, 0.0, scale, 0.0, 0.0, 0.0],
    };

    for pixel in [
        Pixel::new(2048.0, 1000.0),
        Pixel::new(-2048.0, 1000.0),
        Pixel::new(1500.0, -1200.0),
    ] {
        let (ra, dec) = wcs.pixel_to_sky(pixel);
        let back = wcs.sky_to_pixel(ra, dec).expect("the plate is invertible");
        assert!(
            (back.x - pixel.x).abs() < 1e-4 && (back.y - pixel.y).abs() < 1e-4,
            "{pixel:?} came back as {back:?}"
        );
    }
}

#[test]
fn a_star_direction_is_the_unit_vector_of_its_catalogue_position() {
    // The same three products on the catalogue side, at a right ascension whose cosine is neither 1
    // nor 0: at alpha = 0 a wrong operator on `cos d · cos a` is invisible.
    let star = CatalogueStar {
        right_ascension: 1.0,
        declination: 0.5,
        magnitude: 7.5,
    };
    let direction = star.direction();
    let (cos_dec, sin_dec) = (0.5_f64.cos(), 0.5_f64.sin());
    assert!((direction.x - cos_dec * 1.0_f64.cos()).abs() < 1e-14);
    assert!((direction.y - cos_dec * 1.0_f64.sin()).abs() < 1e-14);
    assert!((direction.z - sin_dec).abs() < 1e-14);
    assert!((direction.norm() - 1.0).abs() < 1e-14);
}

#[test]
fn a_plate_needs_at_least_as_many_stars_as_it_has_coefficients() {
    // Fewer stars than coefficients is refused; exactly as many is accepted, and *interpolates*.
    //
    // That second case is worth stating rather than leaving to be discovered. It spends every
    // degree of freedom, so its residuals are zero by construction and the scatter it reports is a
    // property of the fit rather than a measurement; the `max(1)` on the degrees of freedom is what
    // keeps that from being `0/0`, and a caller reading `position_covariance` off such a plate gets
    // zero for the same reason. One star more than coefficients is the first fit whose residual
    // means anything.
    let (ra0, dec0) = (1.0_f64, 0.4_f64);
    let crpix = Pixel::new(0.0, 0.0);
    let terms = term_count(1);
    assert_eq!(terms, 3);

    let (stars, _) = spiral_plate(ra0, dec0, crpix, terms, 0.0);
    assert!(
        PlateSolution::fit(&stars[..terms - 1], ra0, dec0, crpix, 1).is_none(),
        "an under-determined plate must fail"
    );

    let exact = spiral_plate(ra0, dec0, crpix, terms, 0.0)
        .1
        .expect("an exactly determined plate interpolates");
    assert_eq!(exact.residuals.len(), terms);
    assert!(
        exact.rms < 1e-15,
        "interpolation leaves no rms: {}",
        exact.rms
    );
    assert!(exact.variance_xi.is_finite() && exact.variance_eta.is_finite());

    let over = spiral_plate(ra0, dec0, crpix, terms + 1, 0.2)
        .1
        .expect("one degree of freedom is enough");
    assert!(over.variance_xi > 0.0 && over.variance_xi.is_finite());
}

#[test]
fn the_reported_scatter_is_the_scatter_of_the_reported_residuals() {
    // `rms`, `variance_xi` and `variance_eta` are three summaries of the per-star residuals the
    // solution also publishes, so they can be recomputed from them. Doing that is what pins the
    // `2n` in the root-mean-square and the `n − m` in the variance — on a noiseless fit all three
    // are zero and any denominator passes, which is why the plate here carries jitter.
    let count = 40;
    let solution = spiral_plate(2.3, -0.2, Pixel::new(600.0, -250.0), count, 0.3)
        .1
        .unwrap();

    assert_eq!(solution.residuals.len(), count);
    assert!(
        solution.rms > 0.01 * ARCSEC,
        "the jitter has to show: rms {} arcsec",
        solution.rms / ARCSEC
    );

    let sum_xi: f64 = solution.residuals.iter().map(|(x, _)| x * x).sum();
    let sum_eta: f64 = solution.residuals.iter().map(|(_, y)| y * y).sum();
    let freedom = (count - term_count(1)) as f64;
    assert_eq!(freedom, 37.0);

    let want_rms = ((sum_xi + sum_eta) / (2.0 * count as f64)).sqrt();
    assert!(
        (solution.rms - want_rms).abs() < 1e-14 * want_rms,
        "rms {} against {want_rms}",
        solution.rms
    );
    assert!(
        (solution.variance_xi - sum_xi / freedom).abs() < 1e-14 * solution.variance_xi,
        "variance_xi {} against {}",
        solution.variance_xi,
        sum_xi / freedom
    );
    assert!(
        (solution.variance_eta - sum_eta / freedom).abs() < 1e-14 * solution.variance_eta,
        "variance_eta {} against {}",
        solution.variance_eta,
        sum_eta / freedom
    );
}

#[test]
fn the_position_covariance_is_the_quadratic_form_at_the_offset_from_crpix() {
    // `m^T (A^T A)^-1 m · sigma²`, recomputed here from the published `normal_inverse` and a
    // monomial row written out by hand — `[1, u, v]` for an order-1 plate, since `monomials` is
    // private. The reference pixel is deliberately away from the origin and the probe far from the
    // reference pixel, so `pixel − crpix` is a large number that a sign or a swapped axis changes
    // outright.
    let crpix = Pixel::new(700.0, -300.0);
    let solution = spiral_plate(1.1, 0.35, crpix, 40, 0.3).1.unwrap();
    let probe = Pixel::new(1500.0, 900.0);
    let (u, v) = (probe.x - crpix.x, probe.y - crpix.y);
    assert_eq!((u, v), (800.0, 1200.0));

    let row = [1.0, u, v];
    let mut quadratic = 0.0;
    for (i, ri) in row.iter().enumerate() {
        for (j, rj) in row.iter().enumerate() {
            quadratic += ri * solution.normal_inverse[i * 3 + j] * rj;
        }
    }
    assert!(quadratic > 0.0, "the quadratic form is not positive");

    let covariance = solution.position_covariance(probe);
    let want_xi = quadratic * solution.variance_xi;
    let want_eta = quadratic * solution.variance_eta;
    assert!(
        (covariance[0][0] - want_xi).abs() < 1e-12 * want_xi,
        "variance of xi {} against {want_xi}",
        covariance[0][0]
    );
    assert!(
        (covariance[1][1] - want_eta).abs() < 1e-12 * want_eta,
        "variance of eta {} against {want_eta}",
        covariance[1][1]
    );
    assert_eq!(covariance[0][1], 0.0);
    assert_eq!(covariance[1][0], 0.0);

    // At the reference pixel the monomial row is `[1, 0, 0]` and the form collapses to one element,
    // which is the cheapest independent check there is that the offset is `pixel − crpix` and not
    // the pixel itself.
    let at_reference = solution.position_covariance(crpix);
    assert!(
        (at_reference[0][0] - solution.normal_inverse[0] * solution.variance_xi).abs()
            < 1e-15 * at_reference[0][0],
        "{}",
        at_reference[0][0]
    );
    // And extrapolating away from the stars that were fitted costs accuracy, which the covariance
    // is there to say.
    assert!(
        covariance[0][0] > 4.0 * at_reference[0][0],
        "1440 px out ({}) is not dearer than at crpix ({})",
        covariance[0][0],
        at_reference[0][0]
    );
}

// ---------------------------------------------------------------------------------------------
// quads
// ---------------------------------------------------------------------------------------------

#[test]
fn the_code_of_a_known_asterism_is_the_one_worked_out_by_hand() {
    // The invariance and permutation tests compare one code against another, so a transform wrong
    // in the *same* way for both still passes them. This is the arithmetic itself, on an asterism
    // whose AB-frame coordinates are exact halves.
    //
    // A = (0,0) and B = (4,4) are the widest pair (separation² 32 against 16). In the AB frame
    // `along = (r·d)/|d|²` and `across = (r_y d_x − r_x d_y)/|d|²`, and the code is
    // `(along − across, along + across)`:
    //
    //   (4,0): along = 16/32 = 1/2, across = (0·4 − 4·4)/32 = −1/2 → (1, 0)
    //   (0,4): along = 16/32 = 1/2, across = (4·4 − 0·4)/32 = +1/2 → (0, 1)
    //
    // Reading the first coordinate as `along + across` too — the mutation the sum survives — gives
    // (0,0) and (1,1) instead, which is inside the circle and passes every other test in the crate.
    let points = [(0.0, 0.0), (4.0, 4.0), (4.0, 0.0), (0.0, 4.0)];
    let quad = quad_code(&points, [0, 1, 2, 3]).expect("a square is an asterism");
    assert_eq!(
        quad.code,
        [0.0, 1.0, 1.0, 0.0],
        "the AB-frame coordinates are not (0,1) and (1,0)"
    );
    // Ordered so that x_C <= x_D, which put (0,4) first.
    assert_eq!(quad.stars, [0, 1, 3, 2]);

    // Both inner stars sit exactly on the circle with AB as diameter, and the boundary is
    // inclusive: a strict test would throw this asterism away.
    for (x, y) in [(quad.code[0], quad.code[1]), (quad.code[2], quad.code[3])] {
        assert_eq!(
            (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5),
            0.5,
            "({x}, {y}) is not on the circle"
        );
    }
}

#[test]
fn a_tie_for_the_widest_pair_is_broken_the_same_way_every_time() {
    // A 1×3 rectangle: its two diagonals are exactly equal, so which pair becomes A and B is
    // settled by the strictness of the search rather than by the geometry. The first maximal pair
    // wins. Taking the last instead picks the other diagonal and produces this code's coordinates
    // in the other order — which no invariance test would notice, because it is a perfectly valid
    // code for a perfectly valid asterism.
    //
    // Separations²: (0,3) and (1,2) both 10, the rest 1 or 9. With A = (0,0) and B = (1,3):
    //   (1,0): along = 1/10, across = −3/10 → (0.4, −0.2)
    //   (0,3): along = 9/10, across = +3/10 → (0.6,  1.2)
    let points = [(0.0, 0.0), (1.0, 0.0), (0.0, 3.0), (1.0, 3.0)];
    let quad = quad_code(&points, [0, 1, 2, 3]).expect("a rectangle is an asterism");
    assert_eq!(
        quad.stars,
        [0, 3, 1, 2],
        "the first of the two equal diagonals should have won"
    );
    let want = [0.4, -0.2, 0.6, 1.2];
    for (got, expected) in quad.code.iter().zip(want) {
        assert!(
            (got - expected).abs() < 1e-15,
            "code {:?} against {want:?}",
            quad.code
        );
    }
}

#[test]
fn the_widest_pair_is_the_one_farthest_apart_not_the_one_with_the_largest_product() {
    // The separation is `dx² + dy²`. Read as `dx²·dy²` it is zero for every axis-aligned pair and
    // largest for the most diagonal one, so on any asterism whose widest pair happens to be its
    // most diagonal — a square, a rectangle — the two agree and nothing shows.
    //
    // Here they disagree outright: `A = (0,0)`, `B = (10,0)` are 100 apart with a product of zero,
    // while `(0,0)–(7,3)` is only 58 apart and has a product of 441. The sum picks `A, B`; the
    // product picks `A` and `(7,3)`, and the code that follows is a different one.
    let points = [(0.0, 0.0), (10.0, 0.0), (3.0, 3.0), (7.0, 3.0)];
    let quad = quad_code(&points, [0, 1, 2, 3]).expect("a trapezium is an asterism");
    assert_eq!(quad.stars, [0, 1, 2, 3], "the widest pair is (0,0)-(10,0)");
    // In the AB frame: (3,3) has along = 0.3, across = 0.3 -> (0.0, 0.6); (7,3) has along = 0.7,
    // across = 0.3 -> (0.4, 1.0).
    let want = [0.0, 0.6, 0.4, 1.0];
    for (got, expected) in quad.code.iter().zip(want) {
        assert!(
            (got - expected).abs() < 1e-15,
            "code {:?} against {want:?}",
            quad.code
        );
    }
}

#[test]
fn two_inner_stars_that_tie_on_the_first_coordinate_are_still_ordered() {
    // `C` and `D` are put in order of their first coordinate, and the comparison is strict, so a tie
    // leaves them as the widest-pair search found them. It is not a hypothetical tie: the first
    // coordinate is `(rx − ry)/|d|`, which is equal for any two stars on the same anti-diagonal of
    // the `AB` frame, and every coordinate below is exact in binary because `|AB|² = 64`.
    //
    //   (3,1): along = 3/8, across = 1/8 -> (0.25, 0.5)
    //   (5,3): along = 5/8, across = 3/8 -> (0.25, 1.0)
    let points = [(0.0, 0.0), (8.0, 0.0), (3.0, 1.0), (5.0, 3.0)];
    let quad = quad_code(&points, [0, 1, 2, 3]).expect("an asterism");
    assert_eq!(quad.code, [0.25, 0.5, 0.25, 1.0]);
    assert_eq!(quad.stars, [0, 1, 2, 3], "a tie must not swap C and D");
}

#[test]
fn the_ordering_after_the_ab_swap_is_the_same_comparison_as_before_it() {
    // When `x_C + x_D > 1` the pair `A, B` is exchanged, which maps every coordinate `t -> 1 − t`
    // and therefore *reverses* the order of `C` and `D` — so the swap is followed by a second
    // ordering pass, and that pass has its own comparison to get right. It is reached only on this
    // side of the boundary, and it differs from a non-strict one only on a tie, which is what the
    // two stars below are placed to produce.
    //
    // Before the swap: (6,1) -> (0.625, 0.875), (7,2) -> (0.625, 1.125), summing to 1.25 > 1.
    // After it, with A = (8,0) and B = (0,0): (0.375, 0.125) and (0.375, −0.125).
    let points = [(0.0, 0.0), (8.0, 0.0), (6.0, 1.0), (7.0, 2.0)];
    let quad = quad_code(&points, [0, 1, 2, 3]).expect("an asterism");
    assert_eq!(
        quad.stars,
        [1, 0, 2, 3],
        "A and B should have been exchanged, and C and D left alone"
    );
    assert_eq!(quad.code, [0.375, 0.125, 0.375, -0.125]);
}

#[test]
fn every_quad_a_point_set_yields_is_a_distinct_four_star_subset() {
    // `build_quads` walks strictly increasing index quadruples. A bound that lets an index repeat
    // makes a degenerate subset, which the code builder rejects and nothing notices; one that lets
    // an index go *backwards* makes the same subset twice, which doubles every vote that subset
    // casts downstream. The count against an independently written enumeration sees both.
    let mut noise = Xorshift(0x1234_5678_9ABC_DEF0);
    let points: Vec<(f64, f64)> = (0..9)
        .map(|_| (noise.signed(100.0), noise.signed(100.0)))
        .collect();

    let quads = build_quads(&points, 9);
    assert!(!quads.is_empty(), "no quad survived a nine-star field");

    let mut expected = 0usize;
    for i in 0..9 {
        for j in (i + 1)..9 {
            for k in (j + 1)..9 {
                for l in (k + 1)..9 {
                    if quad_code(&points, [i, j, k, l]).is_some() {
                        expected += 1;
                    }
                }
            }
        }
    }
    assert_eq!(
        quads.len(),
        expected,
        "the quad list is not the subset list"
    );

    let mut seen: Vec<[usize; 4]> = Vec::new();
    for quad in &quads {
        let mut sorted = quad.stars;
        sorted.sort_unstable();
        assert!(
            sorted.windows(2).all(|pair| pair[0] < pair[1]),
            "a quad repeats a star: {:?}",
            quad.stars
        );
        assert!(
            !seen.contains(&sorted),
            "the subset {sorted:?} appears twice"
        );
        seen.push(sorted);
    }

    // And the cap is honoured: four stars admit one subset, three admit none.
    assert!(build_quads(&points, 4).len() <= 1);
    assert_eq!(
        build_quads(&points, 3),
        [] as [spacelab_astrometry::quads::Quad; 0]
    );
}

#[test]
fn a_correspondence_survives_only_by_being_the_best_in_both_directions() {
    // The vote is hand-built, because the geometry that produces a *contested* vote in a real field
    // is exactly the geometry a test cannot arrange. Image star 0 is matched to reference star 0
    // three times and to reference star 1 twice; reference star 0 is also matched to image star 2
    // twice. Both losing pairs clear the two-vote floor, so the only thing that can reject them is
    // the mutual test — one direction of it each.
    let quad = |stars: [usize; 4]| Quad {
        code: [0.0; 4],
        stars,
    };
    let image = vec![
        quad([0, 5, 6, 7]),
        quad([0, 5, 6, 7]),
        quad([0, 5, 6, 7]),
        quad([0, 0, 8, 8]),
        quad([2, 2, 9, 9]),
    ];
    let reference = vec![
        quad([0, 15, 16, 17]),
        quad([0, 15, 16, 17]),
        quad([0, 15, 16, 17]),
        quad([1, 1, 18, 18]),
        quad([0, 0, 19, 19]),
    ];
    let matches: Vec<QuadMatch> = (0..5)
        .map(|i| QuadMatch {
            image: i,
            reference: i,
            distance: 0.0,
        })
        .collect();

    let pairs = star_correspondences(&matches, &image, &reference, 2);
    assert!(
        pairs.contains(&(0, 0)),
        "the three-vote pair should survive: {pairs:?}"
    );
    assert!(
        !pairs.contains(&(0, 1)),
        "image star 0 is better matched by reference star 0: {pairs:?}"
    );
    assert!(
        !pairs.contains(&(2, 0)),
        "reference star 0 is better matched by image star 0: {pairs:?}"
    );
    // The uncontested pairs are untouched — the mutual test rejects rivals, not everything.
    for pair in [(5, 15), (6, 16), (7, 17)] {
        assert!(pairs.contains(&pair), "{pair:?} missing from {pairs:?}");
    }

    // Raising the floor above the winning count leaves nothing, which is what says the floor is a
    // floor and not a formality.
    assert_eq!(
        star_correspondences(&matches, &image, &reference, 4),
        [] as [(usize, usize); 0]
    );
    assert_eq!(
        star_correspondences(&[], &image, &reference, 1),
        [] as [(usize, usize); 0]
    );
}

#[test]
fn a_vote_is_counted_against_the_pair_it_names_and_not_against_half_of_it() {
    // Votes accumulate into a tally keyed by the *pair*, so a vote for `(0, 1)` finds the entry
    // whose image star is 0 **and** whose reference star is 1. Matching on either alone merges two
    // different correspondences into one, and the merged tally then reports the wrong one.
    //
    // Here `(0, 0)` is voted for once and `(0, 1)` three times, with `(0, 0)` seen first. Keyed by
    // the pair, `(0, 1)` wins on three votes and `(0, 0)` falls to the floor. Keyed by either half,
    // all four votes land on `(0, 0)` — the entry that happened to be created first — and `(0, 1)`
    // never exists at all, so the answer is the exact opposite.
    let quad = |stars: [usize; 4]| Quad {
        code: [0.0; 4],
        stars,
    };
    let image = vec![quad([0, 4, 5, 6]), quad([0, 0, 0, 7])];
    let reference = vec![quad([0, 14, 15, 16]), quad([1, 1, 1, 17])];
    let matches: Vec<QuadMatch> = (0..2)
        .map(|i| QuadMatch {
            image: i,
            reference: i,
            distance: 0.0,
        })
        .collect();

    let pairs = star_correspondences(&matches, &image, &reference, 2);
    assert!(
        pairs.contains(&(0, 1)),
        "the three-vote pair is missing: {pairs:?}"
    );
    assert!(
        !pairs.contains(&(0, 0)),
        "a one-vote pair is below the floor: {pairs:?}"
    );
}
