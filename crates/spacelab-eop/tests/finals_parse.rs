//! Tests against a committed excerpt of `finals2000A.all`.
//!
//! The full file is 3.7 MB and changes daily; it is fetched, not vendored. See
//! `tools/fetch-eop.sh`. The excerpt is chosen to exercise every branch: 1973 rows
//! where the two bulletins genuinely disagree, the 2017 leap second, the boundary
//! where measurement gives way to prediction, and the reserved-but-empty rows at
//! the tail.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_eop::{EopError, EopQuality, EopSeries};

const EXCERPT: &str = include_str!("fixtures/finals2000A_excerpt.txt");

fn series() -> EopSeries {
    EopSeries::parse_finals2000a(EXCERPT).expect("the excerpt parses")
}

/// The load-bearing test of the whole parser.
///
/// Where the IERS has published both bulletins, they describe the same pole. After
/// 2000 they agree to under a milliarcsecond. If any column range were off by a
/// single character, the two would disagree by order 0.1 arcsec — the digits would
/// simply be different numbers. No amount of plausible output can survive this.
#[test]
fn the_two_bulletins_describe_the_same_pole() {
    let mut compared = 0;

    for sample in series().samples() {
        let Some(b) = sample.bulletin_b else { continue };
        if sample.mjd < 51_544.0 {
            continue; // before 2000 the rapid and final solutions really do differ
        }
        let a = sample.bulletin_a;

        let dxp = (a.xp.arcseconds() - b.xp.arcseconds()).abs();
        let dyp = (a.yp.arcseconds() - b.yp.arcseconds()).abs();
        let ddut1 = (a.dut1_seconds - b.dut1_seconds).abs();

        assert!(
            dxp < 1e-3,
            "MJD {}: xp differs by {dxp:.2e} arcsec",
            sample.mjd
        );
        assert!(
            dyp < 1e-3,
            "MJD {}: yp differs by {dyp:.2e} arcsec",
            sample.mjd
        );
        assert!(
            ddut1 < 1e-3,
            "MJD {}: dut1 differs by {ddut1:.2e} s",
            sample.mjd
        );
        compared += 1;
    }

    assert!(
        compared >= 10,
        "only {compared} rows carried both bulletins"
    );
}

#[test]
fn reserved_but_empty_days_are_skipped_not_guessed() {
    // The excerpt ends with four rows the IERS has reserved and not yet filled.
    let s = series();
    let text_rows = EXCERPT.lines().filter(|l| !l.trim().is_empty()).count();
    assert!(
        s.samples().len() < text_rows,
        "every line produced a sample; the empty tail rows were not skipped"
    );
    assert!(s.samples().iter().all(|x| x.mjd.is_finite()));
}

#[test]
fn measurement_gives_way_to_prediction_and_never_returns() {
    let s = series();
    let mut seen_prediction = false;
    for sample in s.samples() {
        match sample.bulletin_a.quality {
            EopQuality::Predicted => seen_prediction = true,
            EopQuality::Final => assert!(
                !seen_prediction,
                "MJD {}: a measured day follows a predicted one",
                sample.mjd
            ),
        }
    }
    assert!(seen_prediction, "the excerpt should contain predicted rows");
}

#[test]
fn interpolating_onto_a_tabulated_day_returns_that_day() {
    let s = series();
    for sample in s.samples() {
        let got = s.at_mjd(sample.mjd).expect("tabulated day is in range");
        assert_eq!(got, sample.best(), "MJD {}", sample.mjd);
    }
}

/// `UT1 - UTC` steps by exactly one second at a leap second, and the step lands on
/// a tabulated midnight. Interpolating naively across it smears a whole second over
/// a day: half a second of error at noon, which is 230 metres at the equator.
#[test]
fn a_leap_second_does_not_smear_across_the_day() {
    let s = series();
    // 2016-12-31 is MJD 57753; the leap second lands at the start of 57754.
    let before = s.at_mjd(57_753.0).expect("in range");
    let after = s.at_mjd(57_754.0).expect("in range");

    let step = after.dut1_seconds - before.dut1_seconds;
    assert!(
        (step.abs() - 1.0).abs() < 0.02,
        "expected a ~1 s step across the leap second, saw {step:.4} s"
    );

    // Midday on the 31st must stay on the pre-leap branch, not halfway up the step.
    let noon = s.at_mjd(57_753.5).expect("in range");
    let drift = noon.dut1_seconds - before.dut1_seconds;
    assert!(
        drift.abs() < 0.01,
        "leap second smeared into the day: dut1 drifted {drift:.4} s by noon"
    );
}

#[test]
fn interpolation_hits_samples_and_stays_bounded() {
    // Four-point Lagrange (IERS interp). At a tabulated day it returns that day exactly; at the
    // midpoint it stays within the bracketing pair (no wild cubic overshoot), and departs from the
    // pure two-point linear only by the small cubic correction.
    let s = series();
    let (lo, hi) = (57_752.0, 57_753.0); // interior, consecutive, no leap between
    let a = s.at_mjd(lo).unwrap();
    let b = s.at_mjd(hi).unwrap();

    assert!((s.at_mjd(lo).unwrap().xp.arcseconds() - a.xp.arcseconds()).abs() < 1e-15);

    let mid = s.at_mjd(f64::midpoint(lo, hi)).unwrap().xp.arcseconds();
    let (min, max) = (
        a.xp.arcseconds().min(b.xp.arcseconds()),
        a.xp.arcseconds().max(b.xp.arcseconds()),
    );
    let span = (max - min).max(1e-6);
    assert!(
        mid >= min - 0.5 * span && mid <= max + 0.5 * span,
        "midpoint {mid} overshoots [{min}, {max}]"
    );
    assert!(
        (mid - f64::midpoint(a.xp.arcseconds(), b.xp.arcseconds())).abs() < 0.05,
        "cubic correction implausibly large"
    );
}

#[test]
fn a_hole_in_the_series_is_refused_not_interpolated() {
    // The excerpt splices distant epochs together. Asking inside the splice must
    // fail: linear interpolation across a decade would invent an Earth orientation
    // for days nobody observed.
    let s = series();
    let err = s.at_mjd(45_000.0).unwrap_err();
    assert!(
        matches!(err, EopError::Gap { .. }),
        "expected a Gap, got {err:?}"
    );
}

#[test]
fn outside_the_span_we_do_not_extrapolate_the_earth() {
    let s = series();
    let (first, last) = s.span();
    assert!(matches!(
        s.at_mjd(first - 1.0).unwrap_err(),
        EopError::OutOfRange { .. }
    ));
    assert!(matches!(
        s.at_mjd(last + 1.0).unwrap_err(),
        EopError::OutOfRange { .. }
    ));
}

#[test]
fn quality_degrades_across_a_boundary() {
    let s = series();
    let boundary = s
        .samples()
        .windows(2)
        .find(|w| {
            w[0].bulletin_a.quality == EopQuality::Final
                && w[1].bulletin_a.quality == EopQuality::Predicted
                && (w[1].mjd - w[0].mjd) < 1.5
        })
        .expect("the excerpt straddles the final/predicted boundary");

    let mid = s
        .at_mjd(f64::midpoint(boundary[0].mjd, boundary[1].mjd))
        .unwrap();
    assert_eq!(
        mid.quality,
        EopQuality::Predicted,
        "an interval touching a prediction must not be reported as measured"
    );
}

/// The excerpt proves the parser handles every *shape* of row. This proves it
/// handles the real file: twenty thousand rows, half a century, both bulletins.
///
/// Ignored by default. CI has no network and vendors no 3.7 MB data file; run
/// `tools/fetch-eop.sh` then `cargo test -p spacelab-eop -- --ignored`.
#[test]
#[ignore = "needs a snapshot from tools/fetch-eop.sh"]
fn the_whole_published_series_parses() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/eop");
    let snapshot = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("run tools/fetch-eop.sh first ({}): {e}", dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "txt"))
        .expect("a .txt snapshot");

    let text = std::fs::read_to_string(&snapshot).expect("readable");
    let s = EopSeries::parse_finals2000a(&text).expect("the published file parses");

    let (first, last) = s.span();
    let finals = s
        .samples()
        .iter()
        .filter(|x| x.bulletin_a.quality == EopQuality::Final)
        .count();
    let with_b = s
        .samples()
        .iter()
        .filter(|x| x.bulletin_b.is_some())
        .count();

    println!(
        "{} rows, MJD {first}..{last}; {finals} measured, {} predicted, {with_b} with Bulletin B",
        s.samples().len(),
        s.samples().len() - finals
    );

    // Every day is present; the series is a grid, not a scatter.
    for w in s.samples().windows(2) {
        let gap = w[1].mjd - w[0].mjd;
        assert!(
            (gap - 1.0).abs() < 1e-9,
            "gap of {gap} days after MJD {}",
            w[0].mjd
        );
    }

    // The structural check, at full scale: post-2000 the bulletins agree.
    let worst = s
        .samples()
        .iter()
        .filter(|x| x.mjd >= 51_544.0)
        .filter_map(|x| {
            x.bulletin_b
                .map(|b| (x.bulletin_a.xp.arcseconds() - b.xp.arcseconds()).abs())
        })
        .fold(0.0_f64, f64::max);
    println!("worst |A-B| in xp after 2000: {worst:.2e} arcsec");
    assert!(
        worst < 1e-3,
        "columns are misaligned somewhere: {worst:.2e} arcsec"
    );
}

/// The four-point window is not always available, and every fixture above queries where it is.
///
/// `at_mjd` falls back to two-point linear interpolation in three situations: the first interval
/// (there is no sample below `index - 1`), the last (none above `index + 2`), and an interval whose
/// *neighbour* is across a hole — the bracketing pair is one day apart, so the interval itself is
/// sound, but reaching out to a fourth node would step over days nobody observed. That last case is
/// the one that matters operationally: `a_hole_in_the_series_is_refused_not_interpolated` asks
/// *inside* the splice and gets a `Gap`, which is a different branch entirely.
///
/// Every existing query lands either on a tabulated day, where the two branches agree by
/// construction, or in the deep interior, where only the Lagrange branch runs. So the guards
/// `index >= 1`, `index + 2 < n` and the two neighbour-gap tests are free: 26 of this module's
/// survivors sit on those four expressions.
///
/// The oracle is the two-point linear value computed here from the samples, which is what the
/// fallback is *defined* to produce — not a tolerance, and not a number the interpolator supplied.
#[test]
fn the_lagrange_window_is_refused_at_the_edges_and_beside_a_hole() {
    let parsed = series();
    let samples = parsed.samples();
    let n = samples.len();

    let linear = |lo: usize, hi: usize, mjd: f64| {
        let (a, b) = (samples[lo].best(), samples[hi].best());
        let t = (mjd - samples[lo].mjd) / (samples[hi].mjd - samples[lo].mjd);
        a.xp.arcseconds() + (b.xp.arcseconds() - a.xp.arcseconds()) * t
    };

    // The first interval, and the last. Both must be daily for the query to be legal at all.
    let mut cases: Vec<(&str, usize)> = Vec::new();
    assert!(
        samples[1].mjd - samples[0].mjd <= 1.5,
        "excerpt starts with a hole"
    );
    cases.push(("first interval", 0));
    assert!(
        samples[n - 1].mjd - samples[n - 2].mjd <= 1.5,
        "excerpt ends with a hole"
    );
    cases.push(("last interval", n - 2));

    // An interval sitting immediately before a hole: [i, i+1] is daily, [i+1, i+2] is not.
    let before_hole = (1..n - 2)
        .find(|&i| {
            samples[i + 1].mjd - samples[i].mjd <= 1.5
                && samples[i + 2].mjd - samples[i + 1].mjd > 1.5
        })
        .expect("the excerpt splices distant epochs, so some interval precedes the splice");
    cases.push(("interval before a hole", before_hole));

    // And one immediately after: [i, i+1] is daily, [i-1, i] is not.
    let after_hole = (1..n - 2)
        .find(|&i| {
            samples[i + 1].mjd - samples[i].mjd <= 1.5 && samples[i].mjd - samples[i - 1].mjd > 1.5
        })
        .expect("the excerpt splices distant epochs, so some interval follows the splice");
    cases.push(("interval after a hole", after_hole));

    for (label, i) in cases {
        let mjd = f64::midpoint(samples[i].mjd, samples[i + 1].mjd);
        let got = parsed
            .at_mjd(mjd)
            .expect("a daily interval is interpolable")
            .xp
            .arcseconds();
        let want = linear(i, i + 1, mjd);
        println!("EOP {label} at MJD {mjd}: xp = {got:.9}\", linear = {want:.9}\"");
        assert!(
            (got - want).abs() < 1e-15,
            "{label}: {got:.12} is not the linear value {want:.12} -- the four-point window was used"
        );
    }
}

/// The linear fallback carries its own leap-second de-step, and nothing reaches it.
///
/// `a_leap_second_does_not_smear_across_the_day` queries MJD 57753.5, deep in the interior, so the
/// four-point branch and *its* `destep_dut1` handle the jump. The two-point branch has a separate
/// copy of the same idea, and it only runs at an edge or beside a hole — where the published series
/// happens to have no leap second. Its true arm has therefore never executed: with `step.abs()`
/// never above 0.5, `b.dut1_seconds - step.signum()` is dead code, and seven survivors sit on it.
///
/// The fixture is real published rows with one deleted. Dropping 57752 puts a two-day hole
/// immediately below the interval that straddles the 2017-01-01 leap second, which is exactly the
/// condition that refuses the window: `s[1].mjd - s[0].mjd = 2 > 1.5`.
///
/// The oracle is arithmetic on the tabulated values, not a tolerance. `ΔUT1` steps by very nearly
/// +1 s across midnight; the de-stepped midpoint must sit between the two *continuous* values, and
/// the un-de-stepped one would sit half a second — 230 m of Earth rotation — away from it.
#[test]
fn the_two_point_branch_removes_a_leap_second_too() {
    let keep = [57_751.0, 57_753.0, 57_754.0, 57_755.0];
    let text: String = EXCERPT
        .lines()
        .filter(|l| {
            l.get(7..15)
                .and_then(|f| f.trim().parse::<f64>().ok())
                .is_some_and(|mjd| keep.contains(&mjd))
        })
        .fold(String::new(), |mut acc, l| {
            acc.push_str(l);
            acc.push('\n');
            acc
        });
    let s = EopSeries::parse_finals2000a(&text).expect("four real rows parse");
    assert_eq!(s.samples().len(), 4, "expected exactly the four kept rows");

    let (a, b) = (s.samples()[1].best(), s.samples()[2].best());
    let raw_step = b.dut1_seconds - a.dut1_seconds;
    assert!(
        raw_step.abs() > 0.5,
        "57753 -> 57754 should straddle the leap second, got {raw_step} s"
    );

    let got = s
        .at_mjd(57_753.5)
        .expect("a daily interval is interpolable")
        .dut1_seconds;
    let continuous_b = b.dut1_seconds - raw_step.signum();
    let want = f64::midpoint(a.dut1_seconds, continuous_b);
    let smeared = f64::midpoint(a.dut1_seconds, b.dut1_seconds);
    println!(
        "EOP two-point leap: ΔUT1 = {got:.9} s, de-stepped linear = {want:.9} s, smeared = {smeared:.9} s"
    );
    assert!(
        (got - want).abs() < 1e-12,
        "ΔUT1 {got:.9} s is not the de-stepped linear value {want:.9} s"
    );
    assert!(
        (got - smeared).abs() > 0.4,
        "the de-step made no difference -- the fixture does not straddle a leap"
    );
}

/// Every field of both bulletins, against the row it was cut from — not against each other.
///
/// The cross-bulletin test above asserts that A and B *agree*, which a mutant that makes B inherit
/// A's value satisfies perfectly. This reads MJD 41685 out of the excerpt and names each number as
/// the file prints it, including the two published-in-milliarcseconds conversions and the LOD in
/// milliseconds. 1973 is deliberate: the rapid and final solutions were still far apart then, so
/// every B field differs from its A counterpart by orders of magnitude more than the tolerance.
#[test]
fn every_field_of_both_bulletins_reads_the_row_it_came_from() {
    // 73 1 3 41685.00 I  0.118980 0.011039  0.135656 0.013616  I 0.8056163 0.0002710  3.5563 ...
    //   ...  -0.751 ... -0.701 ...   .141000   .134000   .8044000   -18.636    -3.571
    let s = series();
    let sample = s
        .samples()
        .iter()
        .find(|p| (p.mjd - 41_685.0).abs() < 1e-9)
        .expect("MJD 41685 is in the excerpt");

    let a = sample.bulletin_a;
    let close = |got: f64, want: f64, what: &str| {
        assert!((got - want).abs() < 1e-12, "{what}: {got} vs {want}");
    };
    close(a.xp.arcseconds(), 0.118_980, "A xp");
    close(a.yp.arcseconds(), 0.135_656, "A yp");
    close(a.dut1_seconds, 0.805_616_3, "A ΔUT1");
    close(a.xp_sigma.arcseconds(), 0.011_039, "A σ(xp)");
    close(a.yp_sigma.arcseconds(), 0.013_616, "A σ(yp)");
    close(a.dut1_sigma_seconds, 0.000_271_0, "A σ(ΔUT1)");
    close(a.lod_seconds, 3.556_3e-3, "A LOD, published in ms");
    close(a.dx.arcseconds(), -0.751e-3, "A dX, published in mas");
    close(a.dy.arcseconds(), -0.701e-3, "A dY, published in mas");
    assert_eq!(a.quality, EopQuality::Final);

    let b = sample.bulletin_b.expect("MJD 41685 carries bulletin B");
    close(b.xp.arcseconds(), 0.141_000, "B xp");
    close(b.yp.arcseconds(), 0.134_000, "B yp");
    close(b.dut1_seconds, 0.804_400_0, "B ΔUT1");
    close(b.dx.arcseconds(), -18.636e-3, "B dX, published in mas");
    close(b.dy.arcseconds(), -3.571e-3, "B dY, published in mas");
    assert_eq!(b.quality, EopQuality::Final);
    // Bulletin B does not republish the excess length of day or the uncertainties.
    close(b.lod_seconds, a.lod_seconds, "B LOD falls back to A");
    close(b.xp_sigma.arcseconds(), a.xp_sigma.arcseconds(), "B σ(xp)");
}

/// Bulletin B is final by construction, whatever the rapid solution's flag says.
///
/// The excerpt cannot show this — the IERS publishes B only for days it has finalised, so every row
/// that carries one is flagged `I`. Flipping the pole flag on a real row isolates the claim the
/// parser makes in its own comment, and separates it from simply inheriting the rapid quality.
#[test]
fn bulletin_b_is_final_even_when_the_rapid_flag_says_predicted() {
    let predicted = EXCERPT
        .lines()
        .find(|l| l.contains("41685.00"))
        .expect("MJD 41685 is in the excerpt")
        .replacen(" I  0.118980", " P  0.118980", 1);

    let s = EopSeries::parse_finals2000a(&predicted).expect("one row is a series");
    let sample = s.samples()[0];
    assert_eq!(sample.bulletin_a.quality, EopQuality::Predicted);
    assert_eq!(
        sample.bulletin_b.expect("bulletin B is present").quality,
        EopQuality::Final
    );
}

/// The weaker-of-two rule, on all four pairs, and through an interpolation between two final days.
///
/// `quality_degrades_across_a_boundary` only asserts the `Predicted` outcome, which a rule that
/// answers `Predicted` more often than it should still satisfies. The other three pairs are what
/// pin it: two measured days must interpolate to a measured answer.
#[test]
fn the_weaker_quality_wins_and_two_measured_days_stay_measured() {
    use EopQuality::{Final, Predicted};
    assert_eq!(Final.worst(Final), Final);
    assert_eq!(Final.worst(Predicted), Predicted);
    assert_eq!(Predicted.worst(Final), Predicted);
    assert_eq!(Predicted.worst(Predicted), Predicted);

    let s = series();
    let pair = s
        .samples()
        .windows(2)
        .find(|w| {
            w[0].bulletin_a.quality == Final
                && w[1].bulletin_a.quality == Final
                && (w[1].mjd - w[0].mjd) < 1.5
        })
        .expect("the excerpt has consecutive measured days");
    let mid = s.at_mjd(f64::midpoint(pair[0].mjd, pair[1].mjd)).unwrap();
    assert_eq!(
        mid.quality, Final,
        "two measured days bracket a measurement"
    );
}

/// One row of `finals2000A` in its fixed columns, carrying only the mandatory fields.
///
/// Synthetic rows are the only way to control the sample spacing and the shape of the series: the
/// real file is daily and smooth, which is exactly what hides an interpolation that never
/// normalises its parameter or picks the wrong window.
fn row(mjd: f64, xp: f64, dut1: f64) -> String {
    // Columns, in order: MJD 7..15, the pole flag at 16, xp and σ(xp) 18..36, yp and σ(yp) 37..55,
    // the ΔUT1 flag at 57, ΔUT1 and σ(ΔUT1) 58..78. Everything past that is optional.
    let sigma = 1e-4;
    let dut1_sigma = 1e-6;
    format!(
        "73 1 3 {mjd:8.2} I {xp:9.6}{sigma:9.6} {xp:9.6}{sigma:9.6}  I{dut1:10.7}{dut1_sigma:10.7}"
    )
}

/// Where a full window exists the interpolation is cubic, and it is the *right* four samples.
///
/// A degree-3 polynomial in MJD is reproduced exactly by four-point Lagrange and not at all by the
/// linear fallback, so recovering it pins both that the window was built and that it spans
/// `[index−1 ..= index+2]`. The query sits in the fourth interval of six samples, where `index + 2`
/// and `index · 2` disagree — the first still admits the window, the second does not — and a
/// spacing check that adds its two dates instead of subtracting them rejects every window there is.
/// The existing edge test only asserts that the refused cases fall back, which the fallback firing
/// everywhere also satisfies.
#[test]
fn a_full_window_interpolates_the_cubic_that_the_linear_fallback_cannot() {
    let cubic = |d: f64| 0.1 + 0.02 * d - 0.003 * d * d + 0.0004 * d * d * d;
    let mut text = String::new();
    for k in 0..6 {
        let d = f64::from(k);
        text.push_str(&row(60_000.0 + d, cubic(d), 0.25));
        text.push('\n');
    }
    let s = EopSeries::parse_finals2000a(&text).expect("the synthetic series parses");

    let query = 60_003.5;
    let want = cubic(3.5);
    let got = s.at_mjd(query).unwrap().xp.arcseconds();
    assert!((got - want).abs() < 1e-12, "xp {got} vs the cubic {want}");

    // Non-vacuity: the linear answer over the same interval is far outside that tolerance, so the
    // assertion above is about the interpolation order and not about the fixture being flat.
    let linear = f64::midpoint(cubic(3.0), cubic(4.0));
    assert!(
        (linear - want).abs() > 1e-6,
        "the fixture has no curvature to distinguish the branches"
    );
}

/// The interpolation parameter is a fraction of the interval, not of a day.
///
/// Every real row is one whole day after the last, so `mjd − lo` and `(mjd − lo)/(hi − lo)` are the
/// same number and the division is invisible. At half-day spacing they are not: a query at the
/// midpoint must return the midpoint, which a parameter left unnormalised — or taken modulo the
/// interval — puts at a quarter of the way across instead.
#[test]
fn the_interpolation_parameter_is_normalised_by_the_interval() {
    let text = format!(
        "{}\n{}\n",
        row(60_000.00, 0.100_000, 0.200_000_0),
        row(60_000.50, 0.200_000, 0.400_000_0)
    );
    let s = EopSeries::parse_finals2000a(&text).expect("the synthetic series parses");

    let mid = s.at_mjd(60_000.25).unwrap();
    assert!(
        (mid.xp.arcseconds() - 0.150_000).abs() < 1e-12,
        "xp {} at the midpoint",
        mid.xp.arcseconds()
    );
    assert!(
        (mid.dut1_seconds - 0.300_000_0).abs() < 1e-12,
        "ΔUT1 {}",
        mid.dut1_seconds
    );
}
