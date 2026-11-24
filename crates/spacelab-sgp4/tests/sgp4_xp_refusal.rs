//! An SGP4-XP element set is refused on every path into the propagator (F107, Ф20.Q).
//!
//! Column 63 of line 1, the ephemeris type, is `0` for an element set fitted for SGP4 and `4` for
//! one fitted for SGP4-XP. In an XP set the `n̈/6` field is a solar-radiation-pressure coefficient,
//! the `B*` field a ballistic coefficient, and the mean motion Brouwer's rather than Kozai's; fed to
//! SGP4 such a set gives "extremely poor propagation results" (Payne, Hoots, Butkus, Slatton and
//! Nguyen, "Improvements to the SGP4 propagator (SGP4-XP)", AMOS 2022, §4 and Table 3). SGP4 reads
//! every one of those fields as its own, so the answer is an orbit, and wrong.
//!
//! The refusal is an S2 gate: type 4 is refused by each constructor, and types 0, 2 and blank —
//! the published verification set carries 0 and blank — propagate bit for bit alike.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_sgp4::{Convention, Elements, PropagationError, Satellite, TleBuilder};

/// ISS, epoch 2008-264: the `sgp4` crate's documented element set, ephemeris type 0.
const LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
const LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

/// The ISS set re-printed with `ephemeris_type` in column 63 and its checksum recomputed.
fn iss_with_type(ephemeris_type: u8) -> (String, String) {
    let elements = Elements {
        ephemeris_type,
        ..Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).unwrap()
    };
    let tle = TleBuilder::from_elements(&elements).build().unwrap();
    assert_eq!(tle.line1().as_bytes()[62], b'0' + ephemeris_type);
    (tle.line1().to_owned(), tle.line2().to_owned())
}

#[test]
fn a_type_4_tle_is_refused_by_both_tle_constructors() {
    let (line1, line2) = iss_with_type(4);
    assert_eq!(
        Satellite::from_tle(&line1, &line2).unwrap_err(),
        PropagationError::Sgp4XpElements
    );
    assert_eq!(
        Satellite::from_tle_unchecked(&line1, &line2).unwrap_err(),
        PropagationError::Sgp4XpElements
    );
}

/// The gate sits on the element set, not on the text, so an SGP4-XP set is refused whatever it was
/// read from — a TLE, an OMM, the vendor crate's JSON — and under every convention.
#[test]
fn a_type_4_element_set_is_refused_under_every_convention() {
    let elements = Elements {
        ephemeris_type: 4,
        ..Elements::from_tle(None, LINE1.as_bytes(), LINE2.as_bytes()).unwrap()
    };
    for convention in [
        Convention::Reference,
        Convention::AfspcNodeReduction,
        Convention::Iau,
    ] {
        assert_eq!(
            Satellite::from_elements(&elements, convention).unwrap_err(),
            PropagationError::Sgp4XpElements,
            "{convention:?}"
        );
    }
}

/// Type 0 marks an SGP4 set (Payne et al., §4), Spacetrack Report No. 3 numbers SGP4 2 among its
/// ephemeris programs (§12), and a blank column is read as 0 (`twoline2rv`); none of them may move
/// a returned bit. The column feeds nothing in the model, so any difference here would be the gate
/// leaking into the arithmetic.
#[test]
fn types_0_2_and_blank_propagate_bit_for_bit_alike() {
    let mut blank = LINE1.to_owned();
    blank.replace_range(62..63, " ");
    let (type2_line1, type2_line2) = iss_with_type(2);
    let reference = Satellite::from_tle(LINE1, LINE2).unwrap();
    let others = [
        Satellite::from_tle(&blank, LINE2).unwrap(),
        Satellite::from_tle(&type2_line1, &type2_line2).unwrap(),
    ];
    let bits = |s: &Satellite, minutes: f64| {
        let sv = s.propagate_minutes(minutes).unwrap();
        let (r, v) = (sv.position.vector, sv.velocity.vector);
        [r.x, r.y, r.z, v.x, v.y, v.z].map(f64::to_bits)
    };
    for minutes in [-1440.0, 0.0, 97.5, 1440.0, 10_080.0] {
        let want = bits(&reference, minutes);
        for other in &others {
            assert_eq!(bits(other, minutes), want, "at {minutes} min");
        }
    }
}

#[test]
fn the_refusal_names_the_theory_and_the_column() {
    let message = PropagationError::Sgp4XpElements.to_string();
    assert!(message.contains("SGP4-XP"), "{message}");
    assert!(message.contains("ephemeris type 4"), "{message}");
}
