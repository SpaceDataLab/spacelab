//! End-to-end: a real Mars launch-window `C3` from DE440 states and a Lambert transfer.
//!
//! This exercises the whole deep-space stack together — ANISE DE440 ephemeris (Stage A) feeding
//! `spacelab_astro`'s Lambert solver and transfer metrics (Stage B). A small porkchop grid across
//! the late-2026 Earth→Mars window yields a minimum launch energy in the physically expected range.
//! Gated on the kernel, so it skips without `de440s.bsp`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::path::Path;
use std::str::FromStr;

use spacelab_astro::transfer;
use spacelab_ephemeris::frames::{EARTH_J2000, MARS_BARYCENTER_J2000, SUN_J2000};
use spacelab_ephemeris::{Ephemeris, Epoch};

const KERNEL: &str = "../../data/ephemeris/de440s.bsp";
const MU_SUN: f64 = 1.327_124_400_18e11; // km³/s²

#[test]
fn a_2026_mars_launch_window_has_physical_energy() {
    if !Path::new(KERNEL).exists() {
        eprintln!("skipping: {KERNEL} not present — run tools/fetch-ephemeris.sh");
        return;
    }
    let eph = Ephemeris::load(KERNEL).expect("de440s.bsp loads");

    // A coarse porkchop grid over the late-2026 window: departure dates × arrival dates.
    let pairs = [
        ("2026-11-01", "2027-07-01"),
        ("2026-11-20", "2027-08-01"),
        ("2026-11-20", "2027-09-01"),
        ("2026-12-10", "2027-08-15"),
        ("2026-12-10", "2027-09-20"),
        ("2027-01-05", "2027-10-01"),
    ];

    let mut best_c3 = f64::INFINITY;
    let mut best_miss = f64::INFINITY;
    for (dep, arr) in pairs {
        let t_dep = Epoch::from_str(&format!("{dep}T00:00:00 TDB")).unwrap();
        let t_arr = Epoch::from_str(&format!("{arr}T00:00:00 TDB")).unwrap();
        let tof = (t_arr - t_dep).to_seconds();
        let e = eph
            .state(EARTH_J2000, SUN_J2000, t_dep)
            .expect("Earth heliocentric");
        let m = eph
            .state(MARS_BARYCENTER_J2000, SUN_J2000, t_arr)
            .expect("Mars heliocentric");
        let Some(tr) = transfer(
            e.position, e.velocity, m.position, m.velocity, tof, MU_SUN, true,
        ) else {
            continue;
        };
        if tr.c3 < best_c3 {
            best_c3 = tr.c3;
            // Self-consistency of the best leg: the departure velocity reaches Mars.
            let (rf, _) = spacelab_astro::propagate(e.position, tr.v_departure, MU_SUN, tof);
            best_miss = ((rf.x - m.position.x).powi(2)
                + (rf.y - m.position.y).powi(2)
                + (rf.z - m.position.z).powi(2))
            .sqrt();
        }
    }

    println!(
        "2026 Mars window: min C3 = {best_c3:.2} km²/s²  (best-leg arc miss {best_miss:.3e} km)"
    );
    // A real Earth→Mars opportunity lands in the tens of km²/s²; this coarse grid brackets it.
    assert!(
        (5.0..80.0).contains(&best_c3),
        "min C3 {best_c3:.2} km²/s² is not a plausible Mars window"
    );
    assert!(
        best_miss < 1.0,
        "best transfer leg missed Mars by {best_miss:.3e} km"
    );
}
