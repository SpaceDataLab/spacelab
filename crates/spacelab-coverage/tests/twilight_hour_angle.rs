//! `diurnal_arc` against astropy, at five sites and four solar declinations.
//!
//! #279 verified the polar branches and the exactly-π/2 equatorial equinox arc, and nothing pinned
//! an hour angle against an independent implementation. `tools/gen_twilight_oracle.py` supplies
//! one: astropy finds the instant the Sun's geometric altitude crosses each threshold and reads the
//! hour angle and declination off its own `HADec` frame at that instant, so nothing in the fixture
//! comes from the spherical triangle under test.
//!
//! # There is no residual, and the reason is worth more than a residual would be
//!
//! The expectation going in was a few arcseconds of solar parallax: astropy answers for an observer
//! on the ellipsoid and `diurnal_arc` is written for one at the geocentre, and the Sun's diurnal
//! parallax is 8.79″. Measured, the worst implied altitude difference over 74 crossings is
//! **2.4e-7″**, which is the root-finder's own tolerance and not a physical quantity.
//!
//! That is not the test failing to bite; it is the decomposition being clean. `HADec` → `AltAz` is
//! one rotation by `90° − φ` about the east axis, and the spherical triangle `diurnal_arc` solves is
//! that rotation, inverted. astropy's hour angle and declination are already topocentric, so the
//! parallax has been spent *before* the rotation and cannot reappear inside it. Fed the right `δ`,
//! this formula is exact — to the last bit, at every latitude from −24.6° to 69.6° and at all four
//! thresholds.
//!
//! What follows is where a real rise-time error would have to come from, and it is not here: the
//! solar ephemeris, the conversion from `δ` to a clock time, and the choice of `h₀`. None of those
//! is `spacelab-coverage`'s, which is the point of it not owning a time system.
//!
//! The second test is the one that does need the ephemeris. astropy scans a whole day at Tromsø and
//! finds the Sun never crosses; `diurnal_arc` says so from the declination alone. Six such days are
//! in the fixture, five polar-day and one polar-night.

#![allow(clippy::expect_used, clippy::unwrap_used)] // a malformed fixture is a broken generator

use spacelab_coverage::observability::{DiurnalArc, diurnal_arc};

struct Row {
    site: String,
    date: String,
    event: String,
    latitude: f64,
    declination: f64,
    threshold: f64,
    hour_angle: f64,
    outcome: String,
}

fn oracle() -> Vec<Row> {
    let path = "tests/fixtures/twilight_hour_angle.txt";
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("{path} — see tools/gen_twilight_oracle.py"));
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let number = |i: usize| f[i].parse::<f64>().expect("a number");
            Row {
                site: f[0].to_owned(),
                date: f[1].to_owned(),
                event: f[2].to_owned(),
                latitude: number(3).to_radians(),
                declination: number(4),
                threshold: number(5).to_radians(),
                hour_angle: number(6),
                outcome: f[7].to_owned(),
            }
        })
        .collect()
}

#[test]
fn the_diurnal_arc_inverts_astropys_rotation_exactly_at_seventy_four_crossings() {
    let rows = oracle();
    assert!(rows.len() >= 60, "fixture has only {} rows", rows.len());
    let mut crossings = 0;
    let mut worst_altitude_arcsec = 0.0_f64;
    let mut worst_at = String::new();

    for row in &rows {
        let arc = diurnal_arc(row.latitude, row.declination, row.threshold);
        let label = format!("{} {} {}", row.site, row.date, row.event);
        match (arc, row.outcome.as_str()) {
            (DiurnalArc::Rises { hour_angle }, "rises") => {
                crossings += 1;
                let difference = hour_angle - row.hour_angle.abs();
                // Back to the quantity the parallax bounds: `∂h/∂H = −cos φ cos δ sin H`.
                let sensitivity =
                    libm::cos(row.latitude) * libm::cos(row.declination) * libm::sin(hour_angle);
                let altitude_arcsec = (difference * sensitivity).to_degrees() * 3600.0;
                if altitude_arcsec.abs() > worst_altitude_arcsec {
                    worst_altitude_arcsec = altitude_arcsec.abs();
                    worst_at = label.clone();
                }
            }
            (DiurnalArc::AlwaysAbove, "always_above")
            | (DiurnalArc::AlwaysBelow, "always_below") => {}
            (ours, theirs) => panic!("{label}: we say {ours:?}, astropy says {theirs}"),
        }
    }

    assert!(crossings >= 50, "only {crossings} crossings in the fixture");
    // 1e-5″ rather than the 8.794″ of solar parallax the first draft of this test allowed for. The
    // parallax is already inside astropy's topocentric `(H, δ)`, so the rotation from them to an
    // altitude carries none of it, and what is left is the generator's `xtol = 1e-12` day —
    // 8.6e-8 s of rotation. A gate at arcseconds would have passed a formula with a wrong term in
    // it; this one would not.
    assert!(
        worst_altitude_arcsec < 1e-5,
        "worst implied altitude difference {worst_altitude_arcsec:.4e}″ at {worst_at}. This is an \
         identity, not an approximation: `diurnal_arc` inverts the same rotation ERFA's `hd2ae` \
         applies, so anything above the root-finder's own tolerance is a term that differs"
    );
    println!("worst implied altitude difference {worst_altitude_arcsec:.6e}″ at {worst_at}");
}

/// The polar branches, against an ephemeris rather than against an expectation.
///
/// #279 asserted them at Tromsø from the geometry alone. Here astropy independently finds no
/// crossing on the same days, and the two agree on *which* side the Sun stayed — which is the half
/// of the claim a `|cos H| > 1` test cannot make on its own.
#[test]
fn the_polar_branches_are_the_days_astropy_finds_no_crossing_on() {
    let rows = oracle();
    let mut above = 0;
    let mut below = 0;
    for row in &rows {
        match row.outcome.as_str() {
            "always_above" => above += 1,
            "always_below" => below += 1,
            _ => continue,
        }
        assert!(
            row.hour_angle.is_nan(),
            "a non-crossing row should carry no hour angle: {} {}",
            row.site,
            row.date
        );
        let arc = diurnal_arc(row.latitude, row.declination, row.threshold);
        let expected_above = row.outcome == "always_above";
        assert_eq!(
            matches!(arc, DiurnalArc::AlwaysAbove),
            expected_above,
            "{} {} {}: {arc:?} against astropy's {}",
            row.site,
            row.date,
            row.event,
            row.outcome
        );
        // And the duration is the branch's own answer, which nothing else here reaches.
        let duration = arc.duration(7.292_115e-5);
        assert!(
            if expected_above {
                duration.is_infinite()
            } else {
                duration == 0.0
            },
            "{} {}: duration {duration} for {arc:?}",
            row.site,
            row.date
        );
    }
    assert!(
        above > 0 && below > 0,
        "the fixture should exercise both polar branches, got {above} above and {below} below"
    );
}
