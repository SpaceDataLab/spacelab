//! A fingerprint of [`lambert`]'s behaviour over the whole of its argument domain, answers
//! included — the artifact the equivalence proofs in `.cargo/mutants.toml` are measured against.
//!
//! `lambert_longway_sweep.rs` counts refusals near one revolution and nothing else. Several of the
//! solver's guards are unreachable from that population: the time-of-flight screen, the `sin Δν`
//! screen, the `y ≤ 0` screen and the two ends of the bracket ladder are all about arguments the
//! sweep never presents or paths it never takes. Each of those guards is also *redundant* — the
//! closure check at the end of [`lambert`] refuses what they refuse — so removing one changes no
//! test outcome, and only a differential measurement can say whether it changes an answer.
//!
//! So this sweeps the pathological times of flight, the degenerate transfer angles, and hyperbolic
//! through multi-period elliptical arcs, and prints not just the counts but a hash of every velocity
//! it accepted. Two builds agreeing on that hash agree on every bit of every answer over 13 620
//! calls; two builds disagreeing say exactly where to look.

use spacelab_astro::kepler::propagate;
use spacelab_astro::lambert::lambert;
use spacelab_core::Vec3;

const MU: f64 = 398_600.441_8;

/// FNV-1a over the bit patterns of the accepted velocities: order-dependent, and sensitive to a
/// single ulp anywhere in the output.
struct Fingerprint(u64);

impl Fingerprint {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn absorb(&mut self, x: f64) {
        for byte in x.to_bits().to_le_bytes() {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn absorb_vec(&mut self, v: Vec3) {
        self.absorb(v.x);
        self.absorb(v.y);
        self.absorb(v.z);
    }
}

fn main() {
    let radii1 = [7000.0_f64, 8000.0, 42164.0];
    let radii2 = [7000.0_f64, 9000.0, 42164.0];
    // Hyperbolic (seconds), through one period, to several periods.
    let tofs = [
        1.0_f64, 10.0, 100.0, 1000.0, 6000.0, 60_000.0, 600_000.0, 6e6,
    ];
    // Arguments no transfer has: the screens exist for these, and the closure check also refuses
    // them, so they are counted separately from the physical population.
    let pathological = [
        0.0_f64,
        -1.0,
        -1e6,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];

    let mut fingerprint = Fingerprint::new();
    let (mut solved, mut refused) = (0usize, 0usize);
    let (mut path_solved, mut path_refused) = (0usize, 0usize);
    let (mut degenerate_solved, mut degenerate_refused) = (0usize, 0usize);
    let mut worst = 0.0_f64;
    let mut worst_case = String::new();

    for &r1n in &radii1 {
        let r1 = Vec3::new(r1n, 0.0, 0.0);
        for dnu_deg in (1..360).step_by(7) {
            let dnu = f64::from(dnu_deg).to_radians();
            for &r2n in &radii2 {
                let r2 = Vec3::new(r2n * dnu.cos(), r2n * dnu.sin(), 0.0);
                for prograde in [true, false] {
                    for &tof in &tofs {
                        match lambert(r1, r2, tof, MU, prograde) {
                            None => refused += 1,
                            Some((v1, v2)) => {
                                solved += 1;
                                fingerprint.absorb_vec(v1);
                                fingerprint.absorb_vec(v2);
                                let rel = (propagate(r1, v1, MU, tof).0 - r2).norm() / r2n;
                                if rel > worst {
                                    worst = rel;
                                    worst_case = format!(
                                        "r1={r1n} dnu={dnu_deg} r2={r2n} tof={tof} prograde={prograde}"
                                    );
                                }
                            }
                        }
                    }
                    for &tof in &pathological {
                        match lambert(r1, r2, tof, MU, prograde) {
                            None => path_refused += 1,
                            Some((v1, v2)) => {
                                path_solved += 1;
                                fingerprint.absorb_vec(v1);
                                fingerprint.absorb_vec(v2);
                            }
                        }
                    }
                }
            }
            // The transfer angles at which `A` vanishes and the plane is undefined, presented
            // exactly rather than approached: 0, π and 2π, at every radius pair.
            if dnu_deg == 1 {
                for &r2n in &radii2 {
                    for (dx, dy) in [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)] {
                        let r2 = Vec3::new(r2n * dx, r2n * dy, 0.0);
                        for prograde in [true, false] {
                            for &tof in &tofs {
                                match lambert(r1, r2, tof, MU, prograde) {
                                    None => degenerate_refused += 1,
                                    Some((v1, v2)) => {
                                        degenerate_solved += 1;
                                        fingerprint.absorb_vec(v1);
                                        fingerprint.absorb_vec(v2);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("physical    solved {solved:5}  refused {refused:5}");
    println!("pathologic  solved {path_solved:5}  refused {path_refused:5}");
    println!("degenerate  solved {degenerate_solved:5}  refused {degenerate_refused:5}");
    println!("worst relative miss {worst:.6e}  [{worst_case}]");
    println!("fingerprint 0x{:016x}", fingerprint.0);
}
