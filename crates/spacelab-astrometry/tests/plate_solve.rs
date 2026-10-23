//! The plate-solving chain, each link pinned against something it cannot have inherited from the
//! implementation: a synthetic sky whose true plate is known by construction, a distortion put in by
//! hand and asked for back, and a blind solve that is given no pointing at all.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    // The exact comparisons below are exact by construction: pixel indices, epoch nanoseconds and
    // catalogue magnitudes that were written as literals in this file.
    clippy::float_cmp,
    // Loop counters and pixel indices into f64: the frames here are at most 128 square.
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

use spacelab_astrometry::quads::{
    build_quads, code_distance, match_quads, quad_code, star_correspondences,
};
use spacelab_astrometry::sources::{Frame, background, detect};
use spacelab_astrometry::wcs::{
    Correspondence, Pixel, PlateSolution, TanSip, deproject, project, term_count,
};
use spacelab_astrometry::{CatalogueStar, InMemoryCatalogue, StarCatalogue};
use spacelab_core::Vec3;

const ARCSEC: f64 = core::f64::consts::PI / (180.0 * 3600.0);

/// A deterministic stream, so that a failure is reproducible and a passing test stays passing.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform on `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Irwin–Hall with twelve summands: mean zero, variance exactly one, and bounded — which is what
    /// makes the median-absolute-deviation estimate below comparable with a Gaussian σ.
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }
}

/// A synthetic plate: tangent point, scale, rotation. Its inverse is exact, so the correspondences
/// it generates carry no error for the fit to absorb.
struct SyntheticPlate {
    ra0: f64,
    dec0: f64,
    crpix: Pixel,
    scale: f64,
    rotation: f64,
}

impl SyntheticPlate {
    /// Pixel a star lands on, with an optional cubic radial distortion of `distortion` pixels at
    /// 1000 pixels from the reference.
    fn pixel_of(&self, star: &CatalogueStar, distortion: f64) -> Option<Pixel> {
        let (xi, eta) = project(star.right_ascension, star.declination, self.ra0, self.dec0)?;
        let (cos, sin) = (self.rotation.cos(), self.rotation.sin());
        // Invert [ξ; η] = s·R·[u; v]; R is orthogonal so the inverse is its transpose over s.
        let u = (cos * xi + sin * eta) / self.scale;
        let v = (-sin * xi + cos * eta) / self.scale;
        let radius = (u * u + v * v).sqrt();
        let stretch = 1.0 + distortion * (radius / 1000.0).powi(2) / 1000.0;
        Some(Pixel::new(
            self.crpix.x + u * stretch,
            self.crpix.y + v * stretch,
        ))
    }
}

/// A field of stars on a Fibonacci-like spiral about the tangent point, deterministic and never
/// collinear — collinearity is a separate test, not an accident this one should suffer.
fn synthetic_field(ra0: f64, dec0: f64, count: usize, radius: f64) -> Vec<CatalogueStar> {
    let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..count)
        .map(|i| {
            let fraction = (i as f64 + 0.5) / count as f64;
            let r = radius * fraction.sqrt();
            let theta = i as f64 * golden;
            // Small-angle offsets about the tangent point are good enough: the projection is exact
            // and the fit sees whatever this produces.
            let dec = dec0 + r * theta.sin();
            let ra = ra0 + r * theta.cos() / dec.cos();
            CatalogueStar {
                right_ascension: ra,
                declination: dec,
                magnitude: 6.0 + 4.0 * fraction,
            }
        })
        .collect()
}

#[test]
fn a_plate_fitted_to_its_own_projection_reproduces_it_to_machine_precision() {
    let plate = SyntheticPlate {
        ra0: 1.2,
        dec0: 0.6,
        crpix: Pixel::new(2048.0, 2048.0),
        scale: 1.0 * ARCSEC,
        rotation: 0.37,
    };
    let stars = synthetic_field(plate.ra0, plate.dec0, 40, 0.008);
    let correspondences: Vec<Correspondence> = stars
        .iter()
        .map(|star| Correspondence {
            pixel: plate.pixel_of(star, 0.0).unwrap(),
            right_ascension: star.right_ascension,
            declination: star.declination,
        })
        .collect();

    let solution = PlateSolution::fit(&correspondences, plate.ra0, plate.dec0, plate.crpix, 1)
        .expect("a 40-star field determines six plate constants");

    // A noiseless fit of a linear model to linear data: the residual is rounding, not misfit.
    assert!(
        solution.rms < 1.0e-12 * ARCSEC,
        "noiseless rms {} arcsec",
        solution.rms / ARCSEC
    );
    assert!(
        (solution.wcs.plate_scale() / plate.scale - 1.0).abs() < 1.0e-12,
        "recovered scale {} vs imposed {}",
        solution.wcs.plate_scale() / ARCSEC,
        plate.scale / ARCSEC
    );

    // The fitted rotation comes out of the same Jacobian the scale does.
    let recovered = solution.wcs.eta[1].atan2(solution.wcs.xi[1]);
    assert!(
        (recovered - plate.rotation).abs() < 1.0e-12,
        "recovered rotation {recovered} vs imposed {}",
        plate.rotation
    );
}

#[test]
fn the_fit_absorbs_a_distortion_only_when_it_has_the_order_to_represent_it() {
    // Built in the direction the fit models — pixel first, sky second. A distortion imposed the
    // other way round (place the star, then displace its pixel) is *not* representable by a
    // polynomial in the pixel, because the inverse of a cubic is not a cubic; measured, an order-3
    // fit then leaves 0.19 arcsec against the 5.1 arcsec of a linear one, a factor of 27 rather
    // than the exactness below. Which direction the distortion was generated in is the whole
    // difference, and it is the sort of thing a test can quietly get backwards.
    let (ra0, dec0) = (0.5_f64, -0.3_f64);
    let crpix = Pixel::new(1024.0, 1024.0);
    let (scale, rotation) = (0.5 * ARCSEC, -0.2_f64);
    let (cos, sin) = (rotation.cos(), rotation.sin());
    // 10 pixels of radial distortion at 1000 pixels from the reference.
    let cubic_coefficient = scale * 1.0e-8;

    let golden = core::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let correspondences: Vec<Correspondence> = (0..120)
        .map(|i| {
            let fraction = (i as f64 + 0.5) / 120.0;
            let radius = 1000.0 * fraction.sqrt();
            let theta = i as f64 * golden;
            let (u, v) = (radius * theta.cos(), radius * theta.sin());
            let r2 = u * u + v * v;
            let xi = scale * (cos * u - sin * v) + cubic_coefficient * u * r2;
            let eta = scale * (sin * u + cos * v) + cubic_coefficient * v * r2;
            let (ra, dec) = deproject(xi, eta, ra0, dec0);
            Correspondence {
                pixel: Pixel::new(crpix.x + u, crpix.y + v),
                right_ascension: ra,
                declination: dec,
            }
        })
        .collect();

    let linear = PlateSolution::fit(&correspondences, ra0, dec0, crpix, 1).unwrap();
    let cubic = PlateSolution::fit(&correspondences, ra0, dec0, crpix, 3).unwrap();

    // `u(u² + v²)` is `u³ + uv²`, both order-3 monomials, so the cubic model contains the truth
    // exactly and its residual is rounding. The linear model does not, and shows the distortion.
    // The distortion reaches 10 pixels — 5 arcsec — at the frame edge, and the linear fit is left
    // with 0.589 arcsec rms of it. The rest went into the plate *scale*: a least-squares line
    // through a radial cubic absorbs most of it as a magnification, which is exactly why an
    // uncorrected distortion presents as a wrong plate scale rather than as a bad fit.
    assert!(
        linear.rms > 0.5 * ARCSEC,
        "the linear fit should see the distortion, rms {} arcsec",
        linear.rms / ARCSEC
    );
    assert!(
        linear.wcs.plate_scale() > scale * 1.005,
        "the linear fit should have inflated the scale, {} against {}",
        linear.wcs.plate_scale() / ARCSEC,
        scale / ARCSEC
    );
    assert!(
        cubic.rms < 1.0e-8 * ARCSEC,
        "cubic rms {} arcsec is a residual distortion, not rounding",
        cubic.rms / ARCSEC
    );
    assert_eq!(cubic.wcs.xi.len(), term_count(3));

    // And the recovered coefficient is the one that was put in. Index 6 is `u³` and index 8 is
    // `uv²` in the graded ordering, and both carry the same coefficient here.
    for index in [6, 8] {
        assert!(
            (cubic.wcs.xi[index] / cubic_coefficient - 1.0).abs() < 1.0e-6,
            "xi[{index}] = {} against {cubic_coefficient} imposed",
            cubic.wcs.xi[index]
        );
    }
}

#[test]
fn sky_to_pixel_inverts_pixel_to_sky_across_the_frame() {
    let plate = SyntheticPlate {
        ra0: 4.1,
        dec0: 0.9,
        crpix: Pixel::new(512.0, 512.0),
        scale: 2.0 * ARCSEC,
        rotation: 1.1,
    };
    let stars = synthetic_field(plate.ra0, plate.dec0, 60, 0.006);
    let correspondences: Vec<Correspondence> = stars
        .iter()
        .map(|star| Correspondence {
            pixel: plate.pixel_of(star, 5.0).unwrap(),
            right_ascension: star.right_ascension,
            declination: star.declination,
        })
        .collect();
    let solution =
        PlateSolution::fit(&correspondences, plate.ra0, plate.dec0, plate.crpix, 3).unwrap();

    for &probe in &[
        Pixel::new(512.0, 512.0),
        Pixel::new(120.0, 900.0),
        Pixel::new(880.0, 130.0),
        Pixel::new(1000.0, 1000.0),
    ] {
        let (ra, dec) = solution.wcs.pixel_to_sky(probe);
        let back = solution
            .wcs
            .sky_to_pixel(ra, dec)
            .expect("a pixel inside the frame inverts");
        assert!(
            (back.x - probe.x).abs() < 1.0e-2 && (back.y - probe.y).abs() < 1.0e-2,
            "round trip {probe:?} -> {back:?}"
        );
    }
}

#[test]
fn the_covariance_falls_as_the_star_count_and_grows_as_the_noise() {
    let plate = SyntheticPlate {
        ra0: 2.0,
        dec0: 0.1,
        crpix: Pixel::new(1000.0, 1000.0),
        scale: 1.0 * ARCSEC,
        rotation: 0.0,
    };
    let sigma_pixels = 0.2;

    let measure = |count: usize, noise_scale: f64| -> f64 {
        let stars = synthetic_field(plate.ra0, plate.dec0, count, 0.005);
        let mut rng = SplitMix(0xA57E_1234_5678_9ABC);
        let correspondences: Vec<Correspondence> = stars
            .iter()
            .map(|star| {
                let exact = plate.pixel_of(star, 0.0).unwrap();
                Correspondence {
                    pixel: Pixel::new(
                        exact.x + noise_scale * sigma_pixels * rng.normal(),
                        exact.y + noise_scale * sigma_pixels * rng.normal(),
                    ),
                    right_ascension: star.right_ascension,
                    declination: star.declination,
                }
            })
            .collect();
        let solution =
            PlateSolution::fit(&correspondences, plate.ra0, plate.dec0, plate.crpix, 1).unwrap();
        solution.position_covariance(plate.crpix)[0][0]
    };

    // Four times the stars, a quarter of the variance — the 1/n of a least-squares mean. The
    // realisations differ, so this is a factor test with a wide band, not an equality.
    let few = measure(25, 1.0);
    let many = measure(100, 1.0);
    let ratio = few / many;
    assert!(
        (2.0..8.0).contains(&ratio),
        "variance ratio {ratio} for a fourfold star count"
    );

    // Three times the noise, nine times the variance, at fixed geometry and with the same random
    // draws scaled. Not exact — measured 8.9969 — and the 0.03 % is not rounding: the design matrix
    // is built from the *measured* pixels, so the noise enters `(AᵀA)⁻¹` as well as the residuals.
    // A test asserting exactness here would be asserting that the plate constants are fitted
    // against noiseless positions, which they are not.
    let loud = measure(25, 3.0);
    assert!(
        (loud / few / 9.0 - 1.0).abs() < 1.0e-2,
        "variance scaled by {} for a threefold noise",
        loud / few
    );
}

#[test]
fn collinear_stars_do_not_determine_a_plate() {
    let (ra0, dec0) = (1.0_f64, 0.2_f64);
    // Every star on one great-circle arc through the tangent point: the normal matrix is singular.
    let correspondences: Vec<Correspondence> = (0..20)
        .map(|i| {
            let t = (i as f64 - 10.0) * 1.0e-4;
            Correspondence {
                pixel: Pixel::new(1000.0 + t * 1.0e6, 1000.0),
                right_ascension: ra0 + t / dec0.cos(),
                declination: dec0,
            }
        })
        .collect();
    assert!(
        PlateSolution::fit(&correspondences, ra0, dec0, Pixel::new(1000.0, 1000.0), 1).is_none(),
        "a degenerate geometry must fail rather than return a plate nobody can use"
    );
}

#[test]
fn fewer_stars_than_coefficients_is_rejected() {
    let (ra0, dec0) = (0.3, 0.4);
    let stars = synthetic_field(ra0, dec0, 5, 0.003);
    let correspondences: Vec<Correspondence> = stars
        .iter()
        .map(|star| Correspondence {
            pixel: Pixel::new(500.0 + 1.0e5 * star.right_ascension, 500.0),
            right_ascension: star.right_ascension,
            declination: star.declination,
        })
        .collect();
    // Order 2 needs six terms per coordinate and there are five stars.
    assert!(PlateSolution::fit(&correspondences, ra0, dec0, Pixel::new(500.0, 500.0), 2).is_none());
}

// ---------------------------------------------------------------------------------------------
// Source detection
// ---------------------------------------------------------------------------------------------

/// Render a frame: a flat sky, Gaussian noise of unit sigma, and Gaussian sources.
fn render(width: usize, height: usize, sources: &[(f64, f64, f64)], psf: f64) -> Vec<f64> {
    let mut rng = SplitMix(0x1234_5678_9ABC_DEF0);
    let mut pixels = vec![0.0; width * height];
    for value in &mut pixels {
        *value = 1000.0 + rng.normal();
    }
    for &(x, y, peak) in sources {
        for row in 0..height {
            for column in 0..width {
                let (dx, dy) = (column as f64 - x, row as f64 - y);
                let r2 = dx * dx + dy * dy;
                if r2 < 100.0 * psf * psf {
                    pixels[row * width + column] += peak * (-0.5 * r2 / (psf * psf)).exp();
                }
            }
        }
    }
    pixels
}

#[test]
fn the_background_estimator_recovers_the_sky_and_its_noise_through_the_sources() {
    let pixels = render(
        128,
        128,
        &[(40.0, 40.0, 5000.0), (90.0, 70.0, 20000.0)],
        2.0,
    );
    let (level, noise) = background(&pixels).expect("a non-empty frame has a background");
    // The median is the sky even though the mean is not: the two sources carry 3.9e5 counts, which
    // is 24 counts per pixel of mean and about nothing of median.
    assert!(
        (level - 1000.0).abs() < 0.05,
        "median {level} against a sky of 1000"
    );
    assert!(
        (noise - 1.0).abs() < 0.05,
        "MAD-derived sigma {noise} against a unit-variance noise"
    );
}

#[test]
fn detection_finds_the_sources_it_was_given_and_centroids_them_sub_pixel() {
    let truth = [
        (40.25, 40.75, 5000.0),
        (90.5, 70.5, 20000.0),
        (20.0, 100.0, 1500.0),
    ];
    let pixels = render(128, 128, &truth, 2.0);
    let frame = Frame {
        pixels: &pixels,
        width: 128,
        height: 128,
    };
    let sources = detect(&frame, 5.0, 4).expect("a well-formed frame detects");

    assert_eq!(sources.len(), truth.len(), "{sources:#?}");
    // Brightest first is a contract other modules rely on.
    for pair in sources.windows(2) {
        assert!(pair[0].flux >= pair[1].flux);
    }
    for source in &sources {
        let nearest = truth
            .iter()
            .min_by(|a, b| {
                let d =
                    |t: &(f64, f64, f64)| (t.0 - source.centroid.x).hypot(t.1 - source.centroid.y);
                d(a).partial_cmp(&d(b)).unwrap()
            })
            .unwrap();
        let error = (nearest.0 - source.centroid.x).hypot(nearest.1 - source.centroid.y);
        assert!(
            error < 0.05,
            "centroid {:?} is {error} px from the source that made it",
            source.centroid
        );
        // The reported uncertainty has to bracket the error it is a claim about.
        assert!(
            source.position_sigma > 0.0 && source.position_sigma < 0.5,
            "position_sigma {} px",
            source.position_sigma
        );
    }

    // A brighter source is measured better: the uncertainty is width/Q and Q goes as the flux.
    let brightest = &sources[0];
    let faintest = &sources[sources.len() - 1];
    assert!(
        brightest.position_sigma < faintest.position_sigma,
        "{} vs {}",
        brightest.position_sigma,
        faintest.position_sigma
    );
}

#[test]
fn a_single_hot_pixel_is_not_a_source() {
    let mut pixels = render(64, 64, &[], 2.0);
    pixels[32 * 64 + 32] = 1.0e6;
    let frame = Frame {
        pixels: &pixels,
        width: 64,
        height: 64,
    };
    assert_eq!(
        detect(&frame, 5.0, 4).unwrap(),
        [] as [spacelab_astrometry::sources::Source; 0]
    );
    // With the cut lowered to one pixel it is found, which proves the rejection was the size cut
    // and not a failure to see it at all.
    assert_eq!(detect(&frame, 5.0, 1).unwrap().len(), 1);
}

#[test]
fn a_frame_whose_shape_contradicts_its_pixels_is_rejected() {
    let pixels = vec![0.0; 100];
    let frame = Frame {
        pixels: &pixels,
        width: 11,
        height: 10,
    };
    assert!(!frame.is_consistent());
    assert!(detect(&frame, 5.0, 1).is_none());
    // And a noiseless frame has no scale to threshold against.
    let flat = vec![7.0; 100];
    let flat_frame = Frame {
        pixels: &flat,
        width: 10,
        height: 10,
    };
    assert!(detect(&flat_frame, 5.0, 1).is_none());
}

// ---------------------------------------------------------------------------------------------
// Blind identification
// ---------------------------------------------------------------------------------------------

/// Rotate, scale and translate a point set — the three things the quad code must not see.
fn transform_points(
    points: &[(f64, f64)],
    angle: f64,
    scale: f64,
    dx: f64,
    dy: f64,
) -> Vec<(f64, f64)> {
    let (cos, sin) = (angle.cos(), angle.sin());
    points
        .iter()
        .map(|&(x, y)| {
            (
                scale * (cos * x - sin * y) + dx,
                scale * (sin * x + cos * y) + dy,
            )
        })
        .collect()
}

#[test]
fn the_quad_code_is_blind_to_rotation_scale_and_translation() {
    let points = [(0.0, 0.0), (100.0, 20.0), (30.0, 45.0), (60.0, 15.0)];
    let original = quad_code(&points, [0, 1, 2, 3]).expect("a generic quad has a code");
    for &(angle, scale, dx, dy) in &[
        (0.0, 1.0, 0.0, 0.0),
        (1.3, 3.7, -400.0, 900.0),
        (-2.9, 0.02, 1.0e4, -1.0e4),
    ] {
        let moved = transform_points(&points, angle, scale, dx, dy);
        let code = quad_code(&moved, [0, 1, 2, 3]).expect("a similarity preserves the asterism");
        assert!(
            code_distance(&original.code, &code.code) < 1.0e-12,
            "code {:?} against {:?} after ({angle}, {scale})",
            code.code,
            original.code
        );
    }
}

#[test]
fn the_canonical_ordering_makes_the_code_independent_of_how_the_stars_were_listed() {
    let points = [(0.0, 0.0), (100.0, 20.0), (30.0, 45.0), (60.0, 15.0)];
    let reference = quad_code(&points, [0, 1, 2, 3]).unwrap();
    for indices in [
        [1, 0, 2, 3],
        [0, 1, 3, 2],
        [3, 2, 1, 0],
        [2, 3, 0, 1],
        [1, 3, 0, 2],
    ] {
        let code = quad_code(&points, indices).unwrap();
        assert!(
            code_distance(&reference.code, &code.code) < 1.0e-12,
            "permutation {indices:?} gave {:?}",
            code.code
        );
        // And the roles come out the same, which is what makes the vote in
        // `star_correspondences` meaningful.
        assert_eq!(code.stars, reference.stars, "permutation {indices:?}");
    }
    // The canonical form is the one the conventions promise.
    assert!(reference.code[0] <= reference.code[2]);
    assert!(reference.code[0] + reference.code[2] <= 1.0);
}

#[test]
fn a_quad_whose_inner_stars_fall_outside_the_ab_circle_is_refused() {
    // C sits far off the AB segment, outside the circle on AB as diameter.
    let points = [(0.0, 0.0), (100.0, 0.0), (50.0, 400.0), (60.0, 5.0)];
    assert!(quad_code(&points, [0, 1, 2, 3]).is_none());
    // Degenerate: four coincident points have no widest pair.
    let coincident = [(1.0, 1.0); 4];
    assert!(quad_code(&coincident, [0, 1, 2, 3]).is_none());
    // A repeated index is not an asterism.
    assert!(quad_code(&points, [0, 1, 1, 3]).is_none());
}

#[test]
fn a_blind_solve_recovers_the_pointing_it_was_never_told() {
    let plate = SyntheticPlate {
        ra0: 3.4,
        dec0: 0.55,
        crpix: Pixel::new(1024.0, 1024.0),
        scale: 1.5 * ARCSEC,
        rotation: 2.31,
    };
    let catalogue = InMemoryCatalogue::new(synthetic_field(plate.ra0, plate.dec0, 30, 0.004));

    // What the camera saw: the same stars, through the plate, with a centroid error.
    let mut rng = SplitMix(0xDEAD_BEEF_CAFE_0001);
    let measured: Vec<(f64, f64)> = catalogue
        .stars()
        .iter()
        .map(|star| {
            let exact = plate.pixel_of(star, 0.0).unwrap();
            (exact.x + 0.15 * rng.normal(), exact.y + 0.15 * rng.normal())
        })
        .collect();

    // What the solver has: the catalogue, projected onto *its own* tangent plane about a guessed
    // centre. The guess is 4 arcminutes off and the scale and rotation are unknown to it.
    let guess_ra = plate.ra0 + 200.0 * ARCSEC / plate.dec0.cos();
    let guess_dec = plate.dec0 - 150.0 * ARCSEC;
    let reference: Vec<(f64, f64)> = catalogue
        .query(
            Vec3::new(
                guess_dec.cos() * guess_ra.cos(),
                guess_dec.cos() * guess_ra.sin(),
                guess_dec.sin(),
            ),
            0.02,
            12.0,
        )
        .iter()
        .map(|star| project(star.right_ascension, star.declination, guess_ra, guess_dec).unwrap())
        .collect();
    assert_eq!(reference.len(), catalogue.stars().len());

    let image_quads = build_quads(&measured, 10);
    let reference_quads = build_quads(&reference, 10);
    assert!(!image_quads.is_empty() && !reference_quads.is_empty());

    let matches = match_quads(&image_quads, &reference_quads, 0.01);
    let pairs = star_correspondences(&matches, &image_quads, &reference_quads, 2);
    assert!(
        pairs.len() >= 6,
        "{} correspondences from {} quad matches",
        pairs.len(),
        matches.len()
    );

    // Every correspondence must be the identity, because the two lists are the same stars in the
    // same order — a wrong pairing is the failure mode this whole module exists to avoid.
    for &(image_star, reference_star) in &pairs {
        assert_eq!(
            image_star, reference_star,
            "star {image_star} was matched to {reference_star}"
        );
    }

    // And the correspondences are enough to solve the plate about the *guessed* centre.
    let correspondences: Vec<Correspondence> = pairs
        .iter()
        .map(|&(image_star, reference_star)| {
            let star = catalogue.stars()[reference_star];
            Correspondence {
                pixel: Pixel::new(measured[image_star].0, measured[image_star].1),
                right_ascension: star.right_ascension,
                declination: star.declination,
            }
        })
        .collect();
    let solution = PlateSolution::fit(
        &correspondences,
        guess_ra,
        guess_dec,
        Pixel::new(1024.0, 1024.0),
        1,
    )
    .expect("six or more matched stars determine a linear plate");

    // The solved plate points where the true one did, to well inside a pixel, despite the tangent
    // point being 4 arcminutes away from the truth.
    let (ra, dec) = solution.wcs.pixel_to_sky(plate.crpix);
    let separation = ((ra - plate.ra0) * plate.dec0.cos()).hypot(dec - plate.dec0) / ARCSEC;
    assert!(
        separation < 1.0,
        "the solved plate points {separation} arcsec from the true tangent point"
    );
    assert!(
        (solution.wcs.plate_scale() / plate.scale - 1.0).abs() < 1.0e-3,
        "recovered scale {} arcsec/px against {} imposed",
        solution.wcs.plate_scale() / ARCSEC,
        plate.scale / ARCSEC
    );
}

#[test]
fn the_catalogue_query_honours_its_radius_magnitude_and_ordering_contract() {
    let catalogue = InMemoryCatalogue::new(vec![
        CatalogueStar {
            right_ascension: 0.0,
            declination: 0.0,
            magnitude: 9.0,
        },
        CatalogueStar {
            right_ascension: 0.001,
            declination: 0.0,
            magnitude: 4.0,
        },
        CatalogueStar {
            right_ascension: 0.0,
            declination: 1.0,
            magnitude: 2.0,
        },
    ]);
    assert_eq!(
        catalogue.stars()[0].magnitude,
        2.0,
        "sorted on construction"
    );

    let centre = Vec3::new(1.0, 0.0, 0.0);
    let near = catalogue.query(centre, 0.01, 20.0);
    assert_eq!(
        near.len(),
        2,
        "the star one radian away is outside 0.01 rad"
    );
    assert!(near[0].magnitude < near[1].magnitude, "brightest first");

    let bright = catalogue.query(centre, 0.01, 5.0);
    assert_eq!(bright.len(), 1);
    assert_eq!(bright[0].magnitude, 4.0);

    // The direction a star reports is the one the query filters on.
    let direction = catalogue.stars()[0].direction();
    assert!((direction.norm() - 1.0).abs() < 1.0e-15);
    assert!((direction.z - 1.0_f64.sin()).abs() < 1.0e-15);
}

#[test]
fn a_tan_plate_built_by_hand_agrees_with_the_projection_it_was_built_from() {
    // TanSip is usable without a fit, and this pins the coefficient convention: index 0 is the
    // constant, 1 is ∂/∂u and 2 is ∂/∂v. Getting those two transposed swaps the axes, which no
    // round-trip test through the fit itself would ever notice.
    let scale = 1.0 * ARCSEC;
    let wcs = TanSip {
        ra0: 1.0,
        dec0: 0.5,
        crpix: Pixel::new(0.0, 0.0),
        order: 1,
        xi: vec![0.0, scale, 0.0],
        eta: vec![0.0, 0.0, scale],
    };
    let (xi, eta) = wcs.intermediate(Pixel::new(100.0, 7.0));
    assert!((xi - 100.0 * scale).abs() < 1.0e-18, "xi {xi}");
    assert!((eta - 7.0 * scale).abs() < 1.0e-18, "eta {eta}");
    assert!((wcs.plate_scale() - scale).abs() < 1.0e-18);
}
