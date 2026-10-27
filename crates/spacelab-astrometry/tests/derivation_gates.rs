//! The plate-solving chain against the mathematics it codes, with every reference value printed by
//! a derivation and recomputed by a second engine:
//!
//! - `wcs` — `derivations/gnomonic_tan_plate.mac` (Paper II's TAN, the round trip, SIP in the coded
//!   basis, the least-squares plate on exact data) and `derivations/gnomonic_tan_plate.gp`;
//! - `quads` — `derivations/quad_hash.mac` (the code as a complex similarity) and
//!   `derivations/quad_hash.gp`;
//! - `sources` — `derivations/source_moments.mac` (the MAD factor, the frame's background and blob
//!   moments) and `derivations/source_moments.gp`;
//! - `reduce` — `derivations/relativistic_aberration.mac` (the coded boost is the aberration law
//!   and its own inverse) and `derivations/relativistic_aberration.gp`.

#![allow(
    clippy::expect_used,
    // Equalities below are exact by construction: binary-fraction inputs whose arithmetic is exact.
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::unreadable_literal,
    clippy::cast_precision_loss
)]

use spacelab_astrometry::quads::quad_code;
use spacelab_astrometry::reduce::{Aberration, SPEED_OF_LIGHT};
use spacelab_astrometry::sources::{Frame, background, detect};
use spacelab_astrometry::wcs::{Correspondence, Pixel, PlateSolution, TanSip, deproject, project};
use spacelab_core::Vec3;

fn relative(got: f64, want: f64) -> f64 {
    ((got - want) / want).abs()
}

// ---------------------------------------------------------------------------------------------
// wcs
// ---------------------------------------------------------------------------------------------

const RA0: f64 = 1.25;
const DEC0: f64 = 0.5;

/// `(ra, dec, xi, eta)` from `derivations/gnomonic_tan_plate.mac`: the stars at `(161/128,
/// 127/256)`, `(79/64, 33/64)` and `(41/32, 61/128)`, projected by Paper II's route — native
/// coordinates by its eq. (2) with `φ_p = 180°`, then `R_θ = cot θ`, `x = R sin φ`, `y = −R cos φ`.
const TAN: [(f64, f64, f64, f64); 3] = [
    (
        1.2578125,
        0.49609375,
        0.006870836682979272888933354842053015111573,
        -0.003893494409455924647699405689189852725542,
    ),
    (
        1.234375,
        0.515625,
        -0.01359588481762888980947666798792613229059,
        0.01567865290752769947818496722356329734449,
    ),
    (
        1.28125,
        0.4765625,
        0.02778172082060422935033400578894062350972,
        -0.02324259286176584323607780627493715352756,
    ),
];

/// `project` is Calabretta & Greisen's TAN as printed, and `deproject` inverts it. (G2)
///
/// `η` is a difference of two products near 0.4 that agree to its own size, so it carries their
/// rounding, a few `1e-17`; `2e-16` is a handful of those.
#[test]
fn the_projection_is_paper_ii_tan_and_deprojection_inverts_it() {
    for (ra, dec, xi, eta) in TAN {
        let (got_xi, got_eta) = project(ra, dec, RA0, DEC0).expect("near the tangent point");
        assert!((got_xi - xi).abs() < 2e-16, "xi {got_xi} vs {xi}");
        assert!((got_eta - eta).abs() < 2e-16, "eta {got_eta} vs {eta}");
        let (back_ra, back_dec) = deproject(got_xi, got_eta, RA0, DEC0);
        assert!((back_ra - ra).abs() < 1e-15, "ra {back_ra} vs {ra}");
        assert!((back_dec - dec).abs() < 1e-15, "dec {back_dec} vs {dec}");
    }
}

/// A SIP plate `CD·(u + f, v + g)` is the coded polynomial with the coefficients
/// `derivations/gnomonic_tan_plate.mac` maps it to, in the coded monomial order
/// `1, u, v, u², uv, v²`. (G1)
///
/// Every value is a binary fraction whose products and sums fit in a significand, so the
/// comparison is exact: a transposed monomial or coefficient would move it by whole terms.
#[test]
fn a_sip_plate_is_the_coded_polynomial_in_the_coded_monomial_order() {
    let plate = TanSip {
        ra0: RA0,
        dec0: DEC0,
        crpix: Pixel::new(512.0, 384.0),
        order: 2,
        xi: vec![
            0.0,
            0.00002288818359375,
            -0.0000019073486328125,
            0.0000000014551915228366851806640625,
            -0.000000001280568540096282958984375,
            0.00000000061118043959140777587890625,
        ],
        eta: vec![
            0.0,
            0.000003814697265625,
            0.000019073486328125,
            -0.000000000349245965480804443359375,
            0.0000000021536834537982940673828125,
            0.0000000009895302355289459228515625,
        ],
    };
    let (xi, eta) = plate.intermediate(Pixel::new(37.0, -21.0));
    assert_eq!(xi, -340_751_845.0 / 34_359_738_368.0);
    assert_eq!(eta, -77_643_725.0 / 8_589_934_592.0);
}

/// The fitted plate is the least-squares solution of its normal equations on exact data, with the
/// stated covariance, variances and position variance. (G2)
///
/// Seven stars, order 1, targets that are binary fractions with a scatter of a few `2⁻²⁴` rad.
/// The targets reach the fit through `deproject` and back through `project`, which moves each by
/// about `1e-16` rad against residuals of `1e-7`: that bounds the coefficients to `1e-10` relative
/// and the variances, which square the residuals, to `1e-7`. `(AᵀA)⁻¹` sees only the integer
/// pixels and is held to `1e-13`.
#[test]
fn the_plate_fit_is_the_least_squares_solution_on_exact_data() {
    let crpix = Pixel::new(512.0, 384.0);
    let offsets = [
        (-40.0, -30.0),
        (50.0, -20.0),
        (10.0, 45.0),
        (-25.0, 35.0),
        (60.0, 55.0),
        (-55.0, -50.0),
        (5.0, -5.0),
    ];
    let xis = [
        -0.000099003314971923828125,
        0.00089251995086669921875,
        0.000186979770660400390625,
        -0.0001752376556396484375,
        0.000720679759979248046875,
        -0.00019443035125732421875,
        0.000320255756378173828125,
    ];
    let etas = [
        -0.000503599643707275390625,
        -0.00008380413055419921875,
        0.0002591609954833984375,
        0.000049769878387451171875,
        0.000526487827301025390625,
        -0.00071346759796142578125,
        -0.000141084194183349609375,
    ];
    let correspondences: Vec<Correspondence> = (0..7)
        .map(|i| {
            let (ra, dec) = deproject(xis[i], etas[i], RA0, DEC0);
            Correspondence {
                pixel: Pixel::new(crpix.x + offsets[i].0, crpix.y + offsets[i].1),
                right_ascension: ra,
                declination: dec,
            }
        })
        .collect();
    let solution =
        PlateSolution::fit(&correspondences, RA0, DEC0, crpix, 1).expect("a well-posed plate");

    let xi_coefficients = [
        0.0002441385390936867081698731728288907996561,
        0.00001143965513604178820678864342879820093922,
        -0.000003813471110679696607435180898207553409617,
    ];
    let eta_coefficients = [
        -0.0001220698700350244532481549154485525938664,
        0.000003815888464917349595121756288996185814758,
        0.000007629092756206980825382520889829574266376,
    ];
    for (got, want) in solution.wcs.xi.iter().zip(xi_coefficients) {
        assert!(
            relative(*got, want) < 1e-10,
            "xi coefficient {got} vs {want}"
        );
    }
    for (got, want) in solution.wcs.eta.iter().zip(eta_coefficients) {
        assert!(
            relative(*got, want) < 1e-10,
            "eta coefficient {got} vs {want}"
        );
    }
    let normal_inverse = [
        0.1450845514474061335626253940957294353683,
        0.0002178274577242762969332186873029521352823,
        -0.0005560332473488105474347950702206936084838,
        0.0002178274577242762969332186873029521352823,
        0.0001231121987785788301695438410829640408316,
        -0.00007134510659876094097934166721784950503781,
        -0.0005560332473488105474347950702206936084838,
        -0.00007134510659876094097934166721784950503781,
        0.0001416319421478492845646757942544700928192,
    ];
    for (got, want) in solution.normal_inverse.iter().zip(normal_inverse) {
        assert!(relative(*got, want) < 1e-13, "(AᵀA)⁻¹ {got} vs {want}");
    }
    let variance_xi = 0.0000000000000159152596445256146302404712137307697475;
    let variance_eta = 0.00000000000002871666448176488020695573630691334208681;
    assert!(relative(solution.variance_xi, variance_xi) < 1e-7);
    assert!(relative(solution.variance_eta, variance_eta) < 1e-7);
    assert!(
        relative(
            solution.rms,
            0.0000001129246577227369073029386277288105622296
        ) < 1e-7
    );
    let covariance = solution.position_covariance(Pixel::new(crpix.x + 30.0, crpix.y - 12.0));
    assert!(
        relative(
            covariance[0][0],
            0.000000000000005635011771896418279524154348434269163574
        ) < 1e-7
    );
    assert!(
        relative(
            covariance[1][1],
            0.00000000000001016752136117400937127312542763445400683
        ) < 1e-7
    );
}

// ---------------------------------------------------------------------------------------------
// quads
// ---------------------------------------------------------------------------------------------

/// The quad of `derivations/quad_hash.mac`, A at the origin.
const QUAD: [(f64, f64); 4] = [(0.0, 0.0), (37.0, 11.0), (14.0, 19.0), (21.0, -4.0)];

/// The code is the defining normalisation `(1 + i)(z − a)/(b − a)`, exactly
/// `(89, 638, 556, 177)/745` for this quad, with the stars in their canonical roles. (G2)
#[test]
fn the_quad_code_is_the_similarity_that_puts_a_and_b_at_the_unit_corners() {
    let quad = quad_code(&QUAD, [0, 1, 2, 3]).expect("a valid quad");
    let want = [89.0 / 745.0, 638.0 / 745.0, 556.0 / 745.0, 177.0 / 745.0];
    for (got, want) in quad.code.iter().zip(want) {
        assert!((got - want).abs() < 2e-16, "{:?} vs {want:?}", quad.code);
    }
    assert_eq!(quad.stars, [0, 1, 2, 3]);
    // Listed with B first, the A-B swap restores A at the origin: the same code, bit for bit.
    let swapped = [QUAD[1], QUAD[0], QUAD[2], QUAD[3]];
    let again = quad_code(&swapped, [0, 1, 2, 3]).expect("a valid quad");
    assert_eq!(again.code, quad.code);
    assert_eq!(again.stars, [1, 0, 2, 3]);
}

/// A similarity of the field leaves the code unchanged — exactly, not to a tolerance. (G4)
///
/// Under `z → (3 + 4i)z + (17 − 5i)` every coordinate stays an integer and every dot and cross
/// product scales by `|3 + 4i|² = 25`, so each quotient the code forms is the same rational and
/// rounds to the same `f64`. A quarter turn, `z → iz`, is the same argument with factor 1.
#[test]
fn a_similarity_of_the_field_leaves_the_code_bit_for_bit() {
    let original = quad_code(&QUAD, [0, 1, 2, 3]).expect("a valid quad");
    let moved: Vec<(f64, f64)> = QUAD
        .iter()
        .map(|&(x, y)| (3.0 * x - 4.0 * y + 17.0, 4.0 * x + 3.0 * y - 5.0))
        .collect();
    assert_eq!(moved[1], (84.0, 176.0));
    let turned: Vec<(f64, f64)> = QUAD.iter().map(|&(x, y)| (-y, x)).collect();
    for field in [moved, turned] {
        let quad = quad_code(&field, [0, 1, 2, 3]).expect("a valid quad");
        assert_eq!(quad.code, original.code);
        assert_eq!(quad.stars, original.stars);
    }
}

// ---------------------------------------------------------------------------------------------
// sources
// ---------------------------------------------------------------------------------------------

/// `1/Φ⁻¹(3/4)`, the MAD of a unit Gaussian inverted, as `derivations/source_moments.mac` prints it.
const MAD_TO_SIGMA: f64 = 1.4826022185056018605470765293604234313267032;

/// The noise factor is `1/Φ⁻¹(3/4)` rounded to the nearest `f64`. (G3)
///
/// On `[0, 1, 2]` the median is 1 and the MAD is 1, so the reported noise is the factor itself.
#[test]
fn the_mad_to_sigma_factor_is_the_inverse_gaussian_upper_quartile() {
    assert_eq!(background(&[0.0, 1.0, 2.0]), Some((1.0, MAD_TO_SIGMA)));
}

/// The frame of `derivations/source_moments.mac`: background `10 + ((x + 2y) mod 3) − 1`, a 3×3
/// asymmetric blob at `x 3..5, y 4..6` and a 4×4 symmetric one at `x 10..13, y 9..12`.
fn moment_frame() -> Vec<f64> {
    let a = [[20.0, 40.0, 10.0], [60.0, 100.0, 30.0], [20.0, 50.0, 10.0]];
    let profile = [1.0, 3.0, 3.0, 1.0];
    (0..256)
        .map(|i| {
            let (x, y) = (i % 16, i / 16);
            if (3..=5).contains(&x) && (4..=6).contains(&y) {
                10.0 + a[y - 4][x - 3]
            } else if (10..=13).contains(&x) && (9..=12).contains(&y) {
                10.0 + 10.0 * profile[y - 9] * profile[x - 10]
            } else {
                9.0 + ((x + 2 * y) % 3) as f64
            }
        })
        .collect()
}

/// Each blob's flux, centroid, width and position error are the moments of its
/// background-subtracted pixels, and a profile symmetric about a point centroids on it exactly.
/// (G2, G3)
#[test]
fn the_blob_moments_are_their_definitions_and_a_symmetric_blob_centroids_exactly() {
    let pixels = moment_frame();
    assert_eq!(background(&pixels), Some((10.0, MAD_TO_SIGMA)));
    let frame = Frame {
        pixels: &pixels,
        width: 16,
        height: 16,
    };
    let sources = detect(&frame, 3.0, 3).expect("a consistent frame");
    assert_eq!(sources.len(), 2, "{sources:?}");

    // The symmetric blob is the brighter, and its centroid is its centre, exactly.
    let b = sources[0];
    assert_eq!((b.flux, b.peak, b.pixel_count), (640.0, 90.0, 16));
    assert_eq!((b.centroid.x, b.centroid.y), (11.5, 10.5));
    assert!(relative(b.width, 0.866025403784438646763723170752936183471402627) < 1e-15);
    assert!(
        relative(
            b.position_sigma,
            0.00802481990583136491882037428747667086261233998
        ) < 1e-15
    );

    let a = sources[1];
    assert_eq!((a.flux, a.peak, a.pixel_count), (340.0, 100.0, 9));
    assert!(relative(a.centroid.x, 131.0 / 34.0) < 1e-15, "{a:?}");
    assert!(relative(a.centroid.y, 171.0 / 34.0) < 1e-15, "{a:?}");
    assert!(relative(a.width, 0.655691082635528043256204619557255867220608971) < 1e-15);
    assert!(
        relative(
            a.position_sigma,
            0.00857760929796859229874263641580077520121276613
        ) < 1e-15
    );
    assert!(
        relative(
            a.signal_to_noise(),
            76.4421716888892642295857283146814810105158338
        ) < 1e-14
    );
}

// ---------------------------------------------------------------------------------------------
// reduce
// ---------------------------------------------------------------------------------------------

/// The coded boost is the relativistic aberration law, `cos θ' = (cos θ + β)/(1 + β cos θ)` in the
/// plane of the direction and the motion, at `|β| = 3/5` and at an Earth-like `β ≈ 9.3e-5`. (G2)
#[test]
fn the_boost_is_the_relativistic_aberration_law() {
    let directions = [
        Vec3::new(1.0, 2.0, 2.0),
        Vec3::new(-0.6, 0.0, 0.8),
        Vec3::new(2.0, -3.0, 6.0),
    ];
    let fast = [
        [
            0.4962962962962962962962962962962962962963,
            0.7851851851851851851851851851851851851852,
            0.3703703703703703703703703703703703703704,
        ],
        [
            -0.208163265306122448979591836734693877551,
            0.5387755102040816326530612244897959183673,
            0.816326530612244897959183673469387755102,
        ],
        [
            0.6331210191082802547770700636942675159236,
            0.1222929936305732484076433121019108280255,
            0.7643312101910828025477707006369426751593,
        ],
    ];
    let earth = [
        [
            0.333385220772806913559202147607236864455,
            0.6666069922311377980383385435637642765582,
            0.6667003918394098146555189730784553921878,
        ],
        [
            -0.5999695761909179721722899270901547019765,
            -0.00008339169140025920609533872348400794112357,
            0.8000228126066881598930190268833874486706,
        ],
        [
            0.2857383825540365113087487091298228592199,
            -0.4286309226668546437503102937878463758485,
            0.8571050745790529719295063283790830245722,
        ],
    ];
    let cases = [
        (
            Vec3::new(0.36 * SPEED_OF_LIGHT, 0.48 * SPEED_OF_LIGHT, 0.0),
            fast,
        ),
        (Vec3::new(12_000.0, -25_000.0, 3_000.0), earth),
    ];
    for (velocity, want) in cases {
        let aberration = Aberration::new(velocity).expect("subluminal");
        for (direction, want) in directions.iter().zip(want) {
            let got = aberration.apply(*direction).expect("a direction");
            for (g, w) in [got.x, got.y, got.z].into_iter().zip(want) {
                assert!((g - w).abs() < 1e-15, "{got:?} vs {want:?}");
            }
            let back = aberration.remove(got).expect("a direction");
            let unit = direction.scale(1.0 / direction.norm());
            assert!((back - unit).norm() < 1e-15, "{back:?} vs {unit:?}");
        }
    }
}
