#![allow(
    clippy::expect_used,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::unreadable_literal,
    clippy::excessive_precision
)]
//! Where does the f64 predicate actually place the footprint edge?
//!
//! The exact boundary is `cos(lam*) = (s·u + 1 - s²)/L`, `u = sqrt(L² - 1 + s²)`, proved in
//! `derivations/footprint3.mac` and confirmed in FriCAS. `derivations/edge_precision.gp` evaluates
//! it in PARI/GP at 40 significant digits. Here we bisect the *shipped* predicate on `lam` until
//! the f64 grid can no longer be halved, and report the gap in nanometres on the ground.
//!
//! This is the engine's own error. It is not the answer's error: that is dominated by the TLE.

use spacelab_core::Vec3;
use spacelab_sensors::{AccessGeometry, Constraint, ElevationMask};

const R: f64 = 6_378_137.0;

/// Reference values, PARI/GP, 40 digits: (alt_km, mask_deg, lam_exact).
const REFERENCE: &[(f64, f64, f64)] = &[
    (
        500.0,
        0.0,
        0.383_646_515_892_748_353_542_961_418_564_359_009_720_3,
    ),
    (
        500.0,
        5.0,
        0.305_699_893_490_672_418_190_311_639_063_031_088_965_2,
    ),
    (
        500.0,
        10.0,
        0.245_150_363_862_473_710_830_524_363_349_691_487_016_9,
    ),
    (
        500.0,
        25.0,
        0.136_397_850_944_727_070_638_360_160_813_425_746_698_4,
    ),
    (
        35786.0,
        5.0,
        1.332_260_000_810_660_042_688_024_456_791_372_885_186,
    ),
    (
        35786.0,
        10.0,
        1.246_735_699_519_634_203_079_351_617_392_727_975_519,
    ),
    (
        35786.0,
        25.0,
        0.996_934_384_218_321_689_067_182_869_567_025_756_126_3,
    ),
];

fn covered_at(mask: ElevationMask, l: f64, lam: f64) -> bool {
    let geom = AccessGeometry {
        satellite: Vec3::new(l * R * lam.sin(), 0.0, l * R * lam.cos()),
        cell: Vec3::new(0.0, 0.0, R),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    };
    mask.covered(&geom)
}

fn g_at(mask: ElevationMask, l: f64, lam: f64) -> f64 {
    let geom = AccessGeometry {
        satellite: Vec3::new(l * R * lam.sin(), 0.0, l * R * lam.cos()),
        cell: Vec3::new(0.0, 0.0, R),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    };
    mask.g(&geom)
}

fn main() {
    println!("engine error at the footprint edge (reference: PARI/GP, 40 digits)\n");
    println!(
        "{:>6} {:>6} {:>16} {:>16} {:>14}",
        "alt km", "mask°", "flip - exact (rad)", "on ground (nm)", "|g| at exact"
    );
    println!("{}", "-".repeat(64));

    let mut worst_nm: f64 = 0.0;
    for &(alt_km, mask_deg, lam_exact) in REFERENCE {
        let l = (R + alt_km * 1e3) / R;
        let mask = ElevationMask::new(mask_deg.to_radians()).expect("mask in range");

        // bisect on the f64 grid: covered() is true below lam*, false above
        let (mut lo, mut hi) = (lam_exact - 1e-9, lam_exact + 1e-9);
        assert!(covered_at(mask, l, lo), "below the edge must be covered");
        assert!(!covered_at(mask, l, hi), "above the edge must not be");
        while f64::from_bits(hi.to_bits() - lo.to_bits()) > 0.0 && hi.to_bits() - lo.to_bits() > 1 {
            let mid = f64::from_bits(lo.to_bits() + (hi.to_bits() - lo.to_bits()) / 2);
            if covered_at(mask, l, mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let flip = f64::midpoint(lo, hi);
        let d_rad = flip - lam_exact;
        let d_nm = d_rad.abs() * R * 1e9;
        worst_nm = worst_nm.max(d_nm);

        println!(
            "{alt_km:>6.0} {mask_deg:>6.1} {d_rad:>16.3e} {d_nm:>16.2} {:>14.3e}",
            g_at(mask, l, lam_exact).abs()
        );
    }

    println!("\n  worst engine displacement of the footprint edge: {worst_nm:.2} nm");
    println!(
        "  one ulp of lambda at this scale moves the edge by {:.2} nm",
        f64::EPSILON * 0.4 * R * 1e9
    );
    println!("\n  for scale:");
    println!("    TLE along-track drift, 1 day        : 1_000_000_000_000 nm  (1 km)");
    println!("    ignoring EOP                        :   236_700_000_000 nm  (237 m)");
    println!("    predicted EOP row used as final     :    11_500_000_000 nm  (11.5 m)");
    println!("    1 nrad of elevation error           :         5_000_000 nm  (5 mm)");
}
