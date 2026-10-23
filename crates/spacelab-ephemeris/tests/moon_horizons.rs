//! Validate the ANISE-backed DE440 ephemeris against a JPL Horizons vector.
//!
//! Gated on the kernel: `tools/fetch-ephemeris.sh` downloads `de440s.bsp` (not committed), and this
//! test skips cleanly when it is absent, so CI stays network-free. When present, the Moon's
//! geocentric ICRF position at J2000 must match the same Horizons vector the analytic series in
//! `spacelab-propagation` is checked against — but to sub-kilometre, since DE440 *is* the Horizons
//! source, not the arcminute the low-precision series manages.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;
use std::str::FromStr;

use spacelab_ephemeris::frames::{EARTH_J2000, MOON_J2000};
use spacelab_ephemeris::{Ephemeris, Epoch};

const KERNEL: &str = "../../data/ephemeris/de440s.bsp";

#[test]
fn moon_at_j2000_matches_jpl_horizons_to_sub_km() {
    if !Path::new(KERNEL).exists() {
        eprintln!("skipping: {KERNEL} not present — run tools/fetch-ephemeris.sh");
        return;
    }

    let eph = Ephemeris::load(KERNEL).expect("de440s.bsp loads");
    let epoch = Epoch::from_str("2000-01-01T12:00:00 TDB").expect("J2000 TDB epoch");
    let moon = eph
        .state(MOON_J2000, EARTH_J2000, epoch)
        .expect("Moon w.r.t. Earth");

    // JPL Horizons, geocentric ICRF, J2000 (the vector spacelab-propagation's analytic Moon uses).
    let reference = [-291_608.38, -266_716.83, -76_102.49]; // km
    let err = ((moon.position.x - reference[0]).powi(2)
        + (moon.position.y - reference[1]).powi(2)
        + (moon.position.z - reference[2]).powi(2))
    .sqrt();
    println!("ANISE DE440 Moon@J2000: |error| vs Horizons = {err:.3} km");
    // The reference is printed to ~10 m; DE440 reproduces Horizons exactly, so 1 km is generous.
    assert!(err < 1.0, "DE440 Moon@J2000 off Horizons by {err:.3} km");

    // Sanity: the Moon was near apogee, ~404 000 km.
    let dist = (moon.position.x.powi(2) + moon.position.y.powi(2) + moon.position.z.powi(2)).sqrt();
    assert!(
        (400_000.0..410_000.0).contains(&dist),
        "implausible lunar distance {dist:.0} km"
    );
}
