//! The scan catalogue's closed forms against their exact values.
//!
//! `derivations/scan_catalogue.mac` proves the sample normalisation runs from exactly -1 to +1,
//! the swath is antisymmetric, the ramps meet their limits and the VIIRS rows are `+-E`;
//! `derivations/scan_catalogue.gp` prints the edge angles below at 120 digits.

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]

use spacelab_sensors::Instrument;

/// `(pixels, first_roll_rad, last_roll_rad)` in the order of [`INSTRUMENTS`], printed verbatim
/// by `derivations/scan_catalogue.gp`.
const EDGES: [(u32, f64, f64); 11] = [
    (
        2048,
        9.663888068292602867426476e-1,
        -9.663888068292602867426476e-1,
    ),
    (
        409,
        9.630841064639428358353694e-1,
        -9.630841064639428358353694e-1,
    ),
    (
        6400,
        9.822713030224086858926532e-1,
        -9.822713030224086858926532e-1,
    ),
    (
        30,
        8.429940287132611856541426e-1,
        -8.429940287132611856541426e-1,
    ),
    (
        90,
        8.629605953560763160141497e-1,
        -8.629605953560763160141497e-1,
    ),
    (
        56,
        8.639379797371931405772269e-1,
        -8.639379797371931405772269e-1,
    ),
    (
        96,
        9.197885158010116870387850e-1,
        -9.197885158010116870387850e-1,
    ),
    (
        98,
        9.311331559389748292887890e-1,
        -9.311331559389748292887890e-1,
    ),
    (
        4000,
        8.115781021773632532695162e-1,
        -3.857177646907468365001357e-1,
    ),
    (
        3000,
        8.115781021773632532695162e-1,
        -3.857177646907468365001357e-1,
    ),
    (
        42,
        9.250245035569946757695561e-1,
        -9.250245035569946757695561e-1,
    ),
];

const INSTRUMENTS: [Instrument; 11] = [
    Instrument::Avhrr,
    Instrument::AvhrrGac,
    Instrument::Viirs,
    Instrument::AmsuA,
    Instrument::Mhs,
    Instrument::Hirs4,
    Instrument::Atms,
    Instrument::Mwhs2,
    Instrument::Olci,
    Instrument::SlstrNadir,
    Instrument::Ascat,
];

/// The cross-track members, whose swath is symmetric about nadir.
const CROSS_TRACK: usize = 8;

const ASCAT_INNER_RAD: f64 = 4.363323129985823942309227e-1;
const VIIRS_PITCH_EXTENT_RAD: f64 = 7.202545352330395662529647e-3;

fn roll(instrument: Instrument, pixel: u32) -> f64 {
    instrument
        .sample(0, 0, pixel)
        .expect("on the detector")
        .roll
}

/// (G3) Every instrument's first and last sample look at the 120-digit swath edges within 4 ulp
/// (a decimal degree, its conversion to radians and one product); ASCAT's two fans end at
/// `+-25 deg` at samples 20 and 21; the GAC centre sample 204 of 409 looks at nadir exactly; the
/// VIIRS detector rows 0 and 31 sit at `+-atan2(11.87/2, 824)` within 4 ulp.
#[test]
fn the_swath_edges_are_the_catalogued_angles() {
    let close = |got: f64, want: f64| (got - want).abs() <= 4.0 * f64::EPSILON * want.abs();
    for (instrument, (pixels, first, last)) in INSTRUMENTS.into_iter().zip(EDGES) {
        assert_eq!(instrument.pixels_per_line(), pixels, "{instrument:?}");
        let (a, b) = (roll(instrument, 0), roll(instrument, pixels - 1));
        assert!(
            close(a, first),
            "{instrument:?} first sample {a:.17e} against {first:.17e}"
        );
        assert!(
            close(b, last),
            "{instrument:?} last sample {b:.17e} against {last:.17e}"
        );
    }
    assert!(close(roll(Instrument::Ascat, 20), ASCAT_INNER_RAD));
    assert!(close(roll(Instrument::Ascat, 21), -ASCAT_INNER_RAD));
    assert_eq!(roll(Instrument::AvhrrGac, 204), 0.0);
    for (line, sign) in [(0, 1.0), (31, -1.0)] {
        let pitch = Instrument::Viirs
            .sample(0, line, 0)
            .expect("on the detector")
            .pitch;
        assert!(
            close(pitch, sign * VIIRS_PITCH_EXTENT_RAD),
            "VIIRS row {line}: {pitch:.17e}"
        );
    }
}

/// (G4) The cross-track swath is antisymmetric, `roll(n - 1 - p) = -roll(p)`, for every sample
/// of every cross-track member, within 4 ulp of the half-swath.
#[test]
fn every_cross_track_swath_is_antisymmetric_sample_by_sample() {
    for (instrument, (pixels, first, _)) in INSTRUMENTS.into_iter().zip(EDGES).take(CROSS_TRACK) {
        for p in 0..pixels {
            let (a, b) = (roll(instrument, p), roll(instrument, pixels - 1 - p));
            assert!(
                (a + b).abs() <= 4.0 * f64::EPSILON * first,
                "{instrument:?} samples {p} and {}: {a:.17e} and {b:.17e}",
                pixels - 1 - p
            );
        }
    }
}
