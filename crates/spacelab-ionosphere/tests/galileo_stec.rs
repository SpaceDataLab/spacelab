//! STEC oracle: reproduce the JRC reference implementation of NeQuick-G.
//!
//! `fixtures/nequick_g_jrc_stec.tsv` carries the 108 Annex E cases of the Galileo ionospheric
//! correction algorithm ICD (issue 1.2), each repeated for all twelve months, plus an
//! anti-meridian and a zenith geometry, with the slant TEC the JRC reference implementation
//! computes for each at 17 significant digits. The header of the fixture gives the provenance.
//! Positions there are geodetic degrees and metres; only the calendar month and the universal
//! time of day feed the model, so each epoch reduces to `(month, seconds_in_utc_day)`.

// Broadcast coefficients are transcribed byte-identical to the ICD, and the alpha round-trip
// compares verbatim-stored values (no arithmetic).
#![allow(clippy::unreadable_literal)]
#![allow(clippy::float_cmp)]
// A malformed fixture row is a broken test, and panicking on it is the report.
#![allow(clippy::unwrap_used)]

use spacelab_ionosphere::{DateTimeComponents, GeodeticPoint, NeQuickGalileo};

const FIXTURE: &str = include_str!("fixtures/nequick_g_jrc_stec.tsv");

/// One fixture row: the case label, our STEC, the reference STEC, and the ICD table value.
struct Case {
    label: String,
    month: u32,
    stec: f64,
    jrc: f64,
    icd_table: Option<f64>,
}

fn cases() -> Vec<Case> {
    let mut rows = FIXTURE.lines().filter(|l| !l.starts_with('#'));
    assert!(rows.next().is_some_and(|h| h.starts_with("case\t")));
    rows.map(|line| {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 14, "malformed row: {line}");
        let num = |i: usize| f[i].parse::<f64>().unwrap();
        let month: u32 = f[4].parse().unwrap();
        let at = |lon: usize| {
            GeodeticPoint::new(
                num(lon + 1).to_radians(),
                num(lon).to_radians(),
                num(lon + 2),
            )
        };
        let stec = NeQuickGalileo::new([num(1), num(2), num(3)]).stec(
            DateTimeComponents::new(month, num(5) * 3600.0),
            at(6),
            at(9),
        );
        Case {
            label: f[0].to_owned(),
            month,
            stec,
            jrc: num(12),
            icd_table: (f[13] != "-").then(|| num(13)),
        }
    })
    .collect()
}

/// Every case, every month, agrees with the reference implementation to rounding.
///
/// Measured worst case 1.7e-13 relative: the same Kronrod G7-K15 quadrature on the same CCIR and
/// MODIP data, with the two codes differing only in the order of floating-point operations.
#[test]
fn the_stec_matches_the_jrc_reference_in_every_month() {
    let cases = cases();
    assert_eq!(cases.len(), 108 * 12 + 2);
    for c in &cases {
        assert!(
            (c.stec - c.jrc).abs() <= 1.0e-10 * c.jrc,
            "{} month {}: stec = {}, JRC {}",
            c.label,
            c.month,
            c.stec,
            c.jrc
        );
    }
}

/// The ICD's printed Annex E values agree to the table's own departure from the reference. (G5)
///
/// Annex E prints two decimals, but its values are not the reference implementation's rounded:
/// 39 of the 108 differ from it by more than the 0.005 TECU rounding, up to 0.038 TECU (medium
/// solar activity, receiver at 40.19 E, 3.00 S). This is a check on the fixture's April rows
/// against the publication, not a tighter claim than the JRC oracle above.
#[test]
fn the_april_rows_reproduce_the_published_annex_e_table() {
    let april: Vec<Case> = cases()
        .into_iter()
        .filter(|c| c.icd_table.is_some())
        .collect();
    assert_eq!(april.len(), 108);
    for c in &april {
        let table = c.icd_table.unwrap();
        assert!(
            (c.stec - table).abs() < 0.04,
            "{}: stec = {}, Annex E {table}",
            c.label,
            c.stec
        );
    }
}

#[test]
fn alpha_round_trips() {
    let alpha = [236.831641, -0.39362878, 0.00402826613];
    let model = NeQuickGalileo::new(alpha);
    assert_eq!(model.alpha(), alpha);
}
