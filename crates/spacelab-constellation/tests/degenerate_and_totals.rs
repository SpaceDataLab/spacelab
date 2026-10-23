//! The satellite-count accessors and the degenerate-input guards of the generators.
//!
//! The coverage test drives the full generators; this pins `total()` on three families and confirms
//! that a generator with a zero dimension emits nothing rather than dividing by zero.

use spacelab_constellation::{
    DraimTetrahedral, FlowerConstellation, LatticeFlower, Walker, WalkerPattern,
};
use spacelab_core::Angle;

const MU: f64 = 3.986_004_418e14;
const RE: f64 = 6_378_137.0;

fn flower(petals: u32, satellites: u32) -> FlowerConstellation {
    FlowerConstellation {
        petals,
        days: 1,
        satellites,
        eccentricity: 0.05,
        inclination: Angle::from_radians(1.1),
        arg_periapsis: Angle::from_radians(0.0),
        raan: Angle::from_radians(0.0),
        mean_anomaly: Angle::from_radians(0.0),
        frame_rate: 7.292e-5,
        mu: MU,
    }
}

fn lattice(orbits: u32, sats_per_orbit: u32) -> LatticeFlower {
    LatticeFlower {
        orbits,
        sats_per_orbit,
        config: 1,
        semi_major_axis: RE + 1_000e3,
        eccentricity: 0.0,
        inclination: Angle::from_radians(1.0),
        arg_periapsis: Angle::from_radians(0.0),
        raan0: Angle::from_radians(0.0),
        mu: MU,
    }
}

#[test]
fn degenerate_generators_emit_nothing() {
    let walker = Walker {
        pattern: WalkerPattern::Star,
        inclination: Angle::from_radians(1.0),
        planes: 0, // zero planes
        per_plane: 4,
        phasing: 1,
        semi_major_axis: RE + 1_000e3,
        mu: MU,
    };
    let mut count = 0;
    walker.generate(|_| count += 1);
    assert_eq!(count, 0, "a zero-plane Walker emits nothing");

    let mut fcount = 0;
    flower(0, 8).generate(|_| fcount += 1); // zero petals
    assert_eq!(fcount, 0, "a zero-petal Flower emits nothing");

    let mut lcount = 0;
    lattice(0, 3).generate(|_| lcount += 1); // zero orbits
    assert_eq!(lcount, 0, "a zero-orbit lattice emits nothing");
}

#[test]
fn satellite_counts_match_the_configuration() {
    assert_eq!(DraimTetrahedral::nominal(MU).total(), 4);
    assert_eq!(flower(14, 8).total(), 8);
    assert_eq!(lattice(6, 4).total(), 24);
}
