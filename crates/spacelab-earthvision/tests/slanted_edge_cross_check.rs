//! `image_quality` against an independent ISO 12233 measurement of the same pixels.
//!
//! The fixture is written by `tools/gen_edge_oracle.py`, which measures a synthetic Gaussian edge in
//! numpy — numpy's least squares, numpy's binning, its own centred derivative — and stores both that
//! measurement and the closed form for the same point-spread function. The pixels travel in the
//! fixture at 17 significant digits, so the two codes measure bit-identical doubles rather than two
//! evaluations of `erf` in two libm implementations.
//!
//! iquaflow was the oracle the work was scoped against and could not be run: its current release
//! pins `opencv-python==4.4.0.44` and pulls `GPyOpt`, `sagemaker` and `torch`, none of which resolve
//! on a 2026 Python. The script's docstring records that; this test is what stands in its place.
//!
//! **The oracle is deliberately not bit-comparable, and that is the point.** It differentiates the
//! edge-spread function with the centred `[-1, 0, 1]/2h` filter and reports the transform raw; this
//! crate differences adjacent bins and divides out the two box filters the measurement itself
//! applies. So the closed form is the tighter check of the two, and the oracle's job is to show that
//! the gap between the codes is the gap those two conventions predict — a few per cent that grows
//! with frequency — rather than an error in either.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use spacelab_earthvision::features::Image;
use spacelab_earthvision::image_quality::{EdgeConfig, analyse_slanted_edge};

const FIXTURE: &str = include_str!("fixtures/slanted_edge.txt");

struct Fixture {
    h: usize,
    w: usize,
    pixels: Vec<f64>,
    oracle_rer: f64,
    exact_rer: f64,
    oracle_fwhm: f64,
    exact_fwhm: f64,
    /// `(frequency, oracle, exact)` per sampled frequency.
    mtf: Vec<(f64, f64, f64)>,
}

fn parse() -> Fixture {
    let mut out = Fixture {
        h: 0,
        w: 0,
        pixels: Vec::new(),
        oracle_rer: f64::NAN,
        exact_rer: f64::NAN,
        oracle_fwhm: f64::NAN,
        exact_fwhm: f64::NAN,
        mtf: Vec::new(),
    };
    let mut oracle_mtf: Vec<(f64, f64)> = Vec::new();
    let mut exact_mtf: Vec<(f64, f64)> = Vec::new();
    for line in FIXTURE.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut field = line.split_whitespace();
        let key = field.next().unwrap();
        let rest: Vec<&str> = field.collect();
        let number = |i: usize| rest[i].parse::<f64>().unwrap();
        match key {
            "geometry" => {
                out.h = rest[0].parse().unwrap();
                out.w = rest[1].parse().unwrap();
            }
            "oracle_rer" => out.oracle_rer = number(0),
            "exact_rer" => out.exact_rer = number(0),
            "oracle_fwhm" => out.oracle_fwhm = number(0),
            "exact_fwhm" => out.exact_fwhm = number(0),
            "oracle_mtf" => oracle_mtf.push((number(0), number(1))),
            "exact_mtf" => exact_mtf.push((number(0), number(1))),
            "pixels" => out
                .pixels
                .extend(rest.iter().map(|v| v.parse::<f64>().unwrap())),
            other => panic!("unknown key `{other}` in the fixture"),
        }
    }
    out.mtf = oracle_mtf
        .iter()
        .zip(&exact_mtf)
        .map(|(&(f, oracle), &(g, exact))| {
            assert_eq!(
                f.to_bits(),
                g.to_bits(),
                "the fixture's two MTF lists must be in step"
            );
            (f, oracle, exact)
        })
        .collect();
    assert_eq!(out.pixels.len(), out.h * out.w, "the fixture is truncated");
    out
}

/// The two measurements of the same edge, and the closed form both are approximating.
#[test]
fn the_slanted_edge_metrics_match_an_independent_implementation() {
    let want = parse();
    let image = Image {
        data: &want.pixels,
        h: want.h,
        w: want.w,
    };
    let edge = analyse_slanted_edge(&image, &EdgeConfig::default()).unwrap();

    // Against the closed form: this crate corrects for its own binning and differencing, so the
    // agreement is limited by the discretisation, not by a known bias.
    let rer = edge.rer();
    assert!(
        (rer / want.exact_rer - 1.0).abs() < 0.01,
        "RER {rer} against the exact {}",
        want.exact_rer
    );
    let fwhm = edge.fwhm().expect("the edge has a peak");
    assert!(
        (fwhm / want.exact_fwhm - 1.0).abs() < 0.02,
        "FWHM {fwhm} against the exact {}",
        want.exact_fwhm
    );

    // Against the oracle: the edge location and the profile are the same measurement in both, so
    // the two agree on RER and FWHM to well inside their common discretisation error.
    assert!(
        (rer / want.oracle_rer - 1.0).abs() < 0.01,
        "RER {rer} against the oracle's {}",
        want.oracle_rer
    );
    assert!(
        (fwhm / want.oracle_fwhm - 1.0).abs() < 0.02,
        "FWHM {fwhm} against the oracle's {}",
        want.oracle_fwhm
    );

    for &(frequency, oracle, exact) in &want.mtf {
        let got = edge.mtf(frequency);
        println!("MTF({frequency}): crate {got:.6}, oracle {oracle:.6}, exact {exact:.6}");
        // Relative alone is the wrong bound at the top of the band: by Nyquist this edge's response
        // is four per cent of the peak, where the discretisation error of a quarter-pixel bin is a
        // large fraction of what is left. The absolute floor is what makes the two ends comparable.
        assert!(
            (got - exact).abs() < 0.03 * exact + 0.002,
            "MTF({frequency}) {got} against the exact {exact}"
        );
        // The oracle reports the uncorrected transform, so it must sit *below* this crate's number
        // by roughly the response of its own binning and centred difference. Checking the sign of
        // the gap is the part that would catch a correction applied the wrong way round.
        assert!(
            oracle <= got * (1.0 + 1e-9),
            "MTF({frequency}): the uncorrected oracle {oracle} exceeded the corrected {got}"
        );
        let predicted = sinc(frequency * edge.bin_width) * sinc(2.0 * frequency * edge.bin_width);
        // Both filters are exact transfer functions, so the model would be exact if the bins were
        // uniformly populated. They are not — the projected pixels fall where the slant puts them —
        // and the resulting departure from a true box grows with frequency, reaching a few per cent
        // in each code by Nyquist. Hence one bound below 0.4 cycles/pixel and a looser one at the
        // top of the band, rather than one number that is either dishonest or useless.
        let tolerance = if frequency < 0.4 { 0.01 } else { 0.08 };
        assert!(
            (oracle / (got * predicted) - 1.0).abs() < tolerance,
            "MTF({frequency}): the gap to the oracle, {}, is not the {predicted} its filters predict",
            oracle / got
        );
    }
}

/// The normalised cardinal sine, `sin(πx)/(πx)` — the transfer function of a box of unit width.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let arg = core::f64::consts::PI * x;
    arg.sin() / arg
}
