//! How often the universal-variable [`lambert`] refuses an arc, and how far its answers miss.
//!
//! Sweeps 359 transfer angles × 5 radii × 10 times of flight × both senses and keeps the cases
//! [`lambert_multirev`] solves to within 1e-9 of the radius, so the population is arcs that
//! demonstrably have a transfer. `kepler::propagate` is the oracle for both solvers: fly the
//! returned `v1` for `tof` and measure the distance to `r2`. Long-way geometries — retrograde at
//! small Δν, prograde near 360° — are where the universal-variable root crowds `z = 4π²`, which is
//! what this counts.
//!
//! Reported by `Arc::y` and `CLOSURE_TOLERANCE`; run it before changing either.

use spacelab_astro::kepler::propagate;
use spacelab_astro::lambert::{lambert, lambert_multirev};
use spacelab_core::Vec3;

const MU: f64 = 398_600.441_8;

fn main() {
    let radii = [7000.0_f64, 7050.0, 9000.0, 12000.0, 42164.0];
    let tofs = [1.0_f64, 10.0, 60.0, 600.0, 3600.0, 2e4, 6e4, 2e5, 6e5, 6e6];
    let r1 = Vec3::new(7000.0, 0.0, 0.0);

    let (mut considered, mut refused, mut inaccurate) = (0usize, 0usize, 0usize);
    let mut worst_accepted = 0.0_f64;
    let mut worst_case = String::new();

    for dnu_deg in 1..=359 {
        let dnu = f64::from(dnu_deg).to_radians();
        for &r2n in &radii {
            let r2 = Vec3::new(r2n * dnu.cos(), r2n * dnu.sin(), 0.0);
            for &tof in &tofs {
                for prograde in [true, false] {
                    // Izzo decides which configurations are physically solvable at all.
                    let Some((iv1, _)) = lambert_multirev(r1, r2, tof, MU, prograde, 0, false)
                    else {
                        continue;
                    };
                    let izzo_miss = (propagate(r1, iv1, MU, tof).0 - r2).norm();
                    if !(izzo_miss.is_finite() && izzo_miss <= 1e-9 * r2n) {
                        continue;
                    }
                    considered += 1;
                    match lambert(r1, r2, tof, MU, prograde) {
                        None => refused += 1,
                        Some((v1, _)) => {
                            let miss = (propagate(r1, v1, MU, tof).0 - r2).norm();
                            let rel = miss / r2n;
                            if !(rel.is_finite() && rel <= 1e-6) {
                                inaccurate += 1;
                            } else if rel > worst_accepted {
                                worst_accepted = rel;
                                worst_case =
                                    format!("dnu={dnu_deg} r2={r2n} tof={tof} prograde={prograde}");
                            }
                        }
                    }
                }
            }
        }
    }

    println!("considered  {considered}");
    println!("refused     {refused}");
    println!("inaccurate  {inaccurate}");
    println!("worst accepted relative miss {worst_accepted:.3e}  [{worst_case}]");
}
