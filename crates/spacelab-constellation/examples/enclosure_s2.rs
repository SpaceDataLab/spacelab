//! The inputs and enclosures of the S2 containment check of plan 18 Ф28.V, as PARI/GP calls for
//! `derivations/enclosure_s2.gp`, which recomputes each value at 60 digits and counts the misses.
//!
//! ```text
//! cargo run -q --release -p spacelab-constellation --features std --example enclosure_s2 -- \
//!     kfold 1000000 1 | cat derivations/enclosure_s2.gp - | gp -q
//! ```
//!
//! `kfold N SEED` and `alpha N SEED` emit the degenerate families first, then `N` random sets drawn
//! from `SEED`; `kfold-planted` and `alpha-planted` emit the same with every enclosure collapsed to
//! one point, the areas rounded to nearest and α's midpoint, which the checker must refute. Each
//! double is the 16 hex digits of its bits. Nothing here decides anything.

use std::io::{self, BufWriter, Write};

use spacelab_constellation::{SphericalCap, covering_radius, kfold_areas};
use spacelab_core::Vec3;

const PI: f64 = core::f64::consts::PI;

/// `SplitMix64`: a fixed stream, so a failing set is reproducible from its seed and index.
struct SplitMix(u64);

impl SplitMix {
    fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next(&mut self) -> f64 {
        #[expect(clippy::cast_precision_loss, reason = "53 bits are exact in an f64")]
        let unit = (self.word() >> 11) as f64 / 9_007_199_254_740_992.0;
        unit
    }

    fn below(&mut self, n: u64) -> u64 {
        self.word() % n
    }

    /// A direction uniform on the sphere, of a length between 1/8 and 8.
    fn direction(&mut self) -> Vec3 {
        let z = 2.0 * self.next() - 1.0;
        let phi = 2.0 * PI * self.next();
        let r = libm::sqrt(1.0 - z * z);
        let len = libm::exp2(6.0 * self.next() - 3.0);
        Vec3::new(r * libm::cos(phi), r * libm::sin(phi), z).scale(len)
    }

    /// `10⁻ᵐ` for `m` in `1..=16`.
    fn decade(&mut self) -> f64 {
        let m = 1 + i32::try_from(self.below(16)).unwrap_or(15);
        10_f64.powi(-m)
    }
}

fn hex(x: f64) -> String {
    format!("0x{:016x}", x.to_bits())
}

fn hexes(xs: &[f64]) -> String {
    xs.iter().map(|&x| hex(x)).collect::<Vec<_>>().join(",")
}

fn unit(v: Vec3) -> Vec3 {
    v.scale(1.0 / v.norm())
}

/// A random rotation as three orthonormal columns, applied in doubles.
fn rotate(rng: &mut SplitMix) -> impl Fn(Vec3) -> Vec3 + use<> {
    let ex = unit(rng.direction());
    let ez = unit(ex.cross(rng.direction()));
    let ey = ez.cross(ex);
    move |c: Vec3| ex.scale(c.x) + ey.scale(c.y) + ez.scale(c.z)
}

fn cap(c: Vec3, half_angle: f64) -> SphericalCap {
    SphericalCap {
        centre: c,
        half_angle,
    }
}

/// Two caps whose centres are `d` apart: the first about `z`.
fn pair(d: f64, r1: f64, r2: f64) -> [SphericalCap; 2] {
    [
        cap(Vec3::new(0.0, 0.0, 1.0), r1),
        cap(Vec3::new(libm::sin(d), 0.0, libm::cos(d)), r2),
    ]
}

/// The degenerate families: coincident, `10⁻ᵐ` apart, tangent either side, three circles through
/// one point, caps beyond a hemisphere, and a hole an ulp short of π, each in random frames.
fn kfold_degenerate(rng: &mut SplitMix) -> Vec<Vec<SphericalCap>> {
    let mut sets = Vec::new();
    for frame in 0..8 {
        let rot = rotate(rng);
        let mut push = |caps: &[SphericalCap]| {
            sets.push(
                caps.iter()
                    .map(|c| {
                        cap(
                            if frame == 0 { c.centre } else { rot(c.centre) },
                            c.half_angle,
                        )
                    })
                    .collect::<Vec<_>>(),
            );
        };
        let a = cap(Vec3::new(0.3, -0.2, 0.9), 0.5);
        let complement = cap(-a.centre, PI - 0.5);
        let crossing = cap(Vec3::new(0.9, 0.2, 0.4), 0.6);
        push(&[a, a]);
        push(&[a, a, a]);
        push(&[a, complement]);
        push(&[a, complement, a, a]);
        push(&[a, crossing, a]);
        push(&[a, complement, crossing]);
        for m in 1..=16 {
            let delta = 10_f64.powi(-m);
            for rho in [0.6, 1.5, 2.9] {
                let z = Vec3::new(0.0, 0.0, 1.0);
                push(&[cap(z, rho), cap(z, rho + delta)]);
                push(&pair(delta, rho, rho));
                let chain: Vec<SphericalCap> = (0..3)
                    .map(|k| {
                        let t = f64::from(k) * delta;
                        cap(Vec3::new(libm::sin(t), 0.0, libm::cos(t)), rho)
                    })
                    .collect();
                push(&chain);
                // Tangent outside, inside, and complements touching, `δ` either side.
                for side in [-1.0, 1.0] {
                    let (r1, r2) = (rho, 0.4);
                    push(&pair(r1 + r2 + side * delta, r1, r2));
                    push(&pair((r1 - r2).abs() + side * delta, r1, r2));
                    let (c1, c2) = (2.5, 3.0);
                    push(&pair(2.0 * PI - c1 - c2 + side * delta, c1, c2));
                }
            }
        }
        // Three circles through one point; a fourth through it as well.
        let through = |r: f64, az: f64| {
            cap(
                Vec3::new(
                    libm::sin(r) * libm::cos(az),
                    libm::sin(r) * libm::sin(az),
                    libm::cos(r),
                ),
                r,
            )
        };
        push(&[
            through(0.5, 0.0),
            through(11.0 / 16.0, 2.0),
            through(10.0 / 16.0, 4.0),
        ]);
        push(&[
            through(0.5, 0.0),
            through(11.0 / 16.0, 2.0),
            through(10.0 / 16.0, 4.0),
            through(2.0, 5.0),
        ]);
        // Beyond a hemisphere.
        push(&[
            cap(Vec3::new(1.0, 2.0, 3.0), 2.0),
            cap(Vec3::new(-1.0, 1.0, 0.5), 2.6),
            cap(Vec3::new(0.0, -1.0, 1.0), 1.7),
        ]);
        // A circle through the hole of a cap an ulp short of π.
        for k in 1..=8_u64 {
            let rho_j = f64::from_bits(PI.to_bits() - k);
            let rho_i = 0.05 + 3.0 * rng.next();
            let t = rho_i + (rng.next() - 0.5) * 4e-15;
            push(&[
                cap(Vec3::new(libm::sin(t), 0.0, libm::cos(t)), rho_i),
                cap(Vec3::new(0.0, 0.0, -1.0), rho_j),
            ]);
        }
    }
    sets
}

/// A random set of one to six caps; one in eight has a degenerate partner added.
fn kfold_random(rng: &mut SplitMix) -> Vec<SphericalCap> {
    let m = 1 + rng.below(6);
    let mut caps: Vec<SphericalCap> = (0..m)
        .map(|_| {
            let rho = match rng.below(8) {
                0 => rng.decade(),
                1 => PI - rng.decade(),
                2 => 0.5 * PI + (rng.next() - 0.5) * rng.decade(),
                _ => PI * rng.next(),
            };
            cap(
                rng.direction(),
                rho.clamp(1e-300, f64::from_bits(PI.to_bits() - 1)),
            )
        })
        .collect();
    if rng.below(8) == 0 {
        let base = caps[0];
        let d = rng.decade();
        let near = match rng.below(4) {
            0 => base,
            1 => cap(-base.centre, PI - base.half_angle),
            2 => cap(base.centre + Vec3::new(d, -d, d), base.half_angle),
            _ => cap(base.centre, (base.half_angle + d).min(3.1)),
        };
        caps.push(near);
    }
    caps
}

fn emit_kfold(out: &mut impl Write, caps: &[SphericalCap], planted: bool) -> io::Result<()> {
    let Ok(r) = kfold_areas(caps) else {
        return Ok(());
    };
    let (lower, upper) = if planted {
        (r.at_least.clone(), r.at_least.clone())
    } else {
        (r.lower, r.upper)
    };
    let caps: Vec<String> = caps
        .iter()
        .map(|c| {
            format!(
                "[{}]",
                hexes(&[c.centre.x, c.centre.y, c.centre.z, c.half_angle])
            )
        })
        .collect();
    writeln!(
        out,
        "K([{}],[{}],[{}]);",
        caps.join(","),
        hexes(&lower),
        hexes(&upper)
    )
}

fn polyhedra() -> Vec<Vec<Vec3>> {
    let s = 1.0 / libm::sqrt(3.0);
    let g = 0.5 * (1.0 + libm::sqrt(5.0));
    let mut icosa = Vec::new();
    for a in [-1.0, 1.0] {
        for b in [-g, g] {
            icosa.push(Vec3::new(0.0, a, b));
            icosa.push(Vec3::new(a, b, 0.0));
            icosa.push(Vec3::new(b, 0.0, a));
        }
    }
    let mut cube = Vec::new();
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                cube.push(Vec3::new(x, y, z));
            }
        }
    }
    vec![
        vec![
            Vec3::new(s, s, s),
            Vec3::new(s, -s, -s),
            Vec3::new(-s, s, -s),
            Vec3::new(-s, -s, s),
        ],
        vec![
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ],
        cube,
        icosa,
    ]
}

/// The polyhedra, two sites `π − 10⁻ᵐ` apart with company, coincident and antipodal sites, and
/// sites `10⁻ᵐ` apart, in random frames.
fn alpha_degenerate(rng: &mut SplitMix) -> Vec<Vec<Vec3>> {
    let mut sets = Vec::new();
    for frame in 0..8 {
        let rot = rotate(rng);
        let mut push = |sites: &[Vec3]| {
            sets.push(
                sites
                    .iter()
                    .map(|&s| if frame == 0 { s } else { rot(s) })
                    .collect::<Vec<_>>(),
            );
        };
        for p in polyhedra() {
            push(&p);
        }
        let z = Vec3::new(0.0, 0.0, 1.0);
        for m in 1..=16 {
            let d = 10_f64.powi(-m);
            let near_anti = Vec3::new(libm::sin(d), 0.0, -libm::cos(d));
            push(&[z, near_anti]);
            push(&[z, near_anti, Vec3::new(1.0, 0.0, 0.0)]);
            push(&[
                z,
                near_anti,
                Vec3::new(0.3, 0.9, 0.1),
                Vec3::new(-0.5, -0.2, 0.4),
            ]);
            let near = Vec3::new(libm::sin(d), 0.0, libm::cos(d));
            push(&[
                z,
                near,
                Vec3::new(1.0, 1.0, -1.0),
                Vec3::new(-1.0, 0.2, -0.3),
            ]);
            push(&[
                z,
                near,
                Vec3::new(0.0, libm::sin(d), libm::cos(d)),
                -z,
                Vec3::new(1.0, 0.0, 0.0),
            ]);
        }
        push(&[z, z, -z, Vec3::new(1.0, 0.0, 0.0)]);
        push(&[z, -z]);
        push(&[z, -z, Vec3::new(1.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)]);
        push(&[
            z,
            z,
            z,
            Vec3::new(0.2, 0.3, -0.9),
            Vec3::new(-0.7, 0.1, 0.2),
        ]);
    }
    sets
}

fn alpha_random(rng: &mut SplitMix) -> Vec<Vec3> {
    let n = 2 + rng.below(7);
    let mut sites: Vec<Vec3> = (0..n).map(|_| rng.direction()).collect();
    if rng.below(8) == 0 {
        let base = sites[0];
        let d = rng.decade();
        sites.push(match rng.below(4) {
            0 => base,
            1 => -base,
            2 => base + Vec3::new(d, -d, d),
            _ => -base + Vec3::new(d, d, -d),
        });
    }
    sites
}

fn emit_alpha(out: &mut impl Write, sites: &[Vec3], fold: u32, planted: bool) -> io::Result<()> {
    let Ok(e) = covering_radius(sites, fold) else {
        return Ok(());
    };
    let (lower, upper) = if planted {
        let mid = e.lower.midpoint(e.upper);
        (mid, mid)
    } else {
        (e.lower, e.upper)
    };
    let sites: Vec<String> = sites
        .iter()
        .map(|s| format!("[{}]", hexes(&[s.x, s.y, s.z])))
        .collect();
    writeln!(
        out,
        "A({fold},[{}],{},{});",
        sites.join(","),
        hex(lower),
        hex(upper)
    )
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: enclosure_s2 (kfold|alpha)[-planted] N SEED";
    let (mode, n, seed) = match args.as_slice() {
        [mode, n, seed] => (
            mode.as_str(),
            n.parse::<u64>().map_err(|e| format!("{usage}: {e}"))?,
            seed.parse::<u64>().map_err(|e| format!("{usage}: {e}"))?,
        ),
        _ => return Err(usage.into()),
    };
    let planted = mode.ends_with("-planted");
    let mut rng = SplitMix(seed);
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let io = |e: io::Error| e.to_string();
    match mode.trim_end_matches("-planted") {
        "kfold" => {
            for caps in kfold_degenerate(&mut rng) {
                emit_kfold(&mut out, &caps, planted).map_err(io)?;
            }
            for _ in 0..n {
                let caps = kfold_random(&mut rng);
                emit_kfold(&mut out, &caps, planted).map_err(io)?;
            }
        }
        "alpha" => {
            for sites in alpha_degenerate(&mut rng) {
                for fold in 1..=u32::try_from(sites.len()).map_err(|e| e.to_string())? {
                    emit_alpha(&mut out, &sites, fold, planted).map_err(io)?;
                }
            }
            for _ in 0..n {
                let sites = alpha_random(&mut rng);
                let len = u64::try_from(sites.len()).map_err(|e| e.to_string())?;
                let fold = u32::try_from(1 + rng.below(len)).map_err(|e| e.to_string())?;
                emit_alpha(&mut out, &sites, fold, planted).map_err(io)?;
            }
        }
        _ => return Err(usage.into()),
    }
    out.flush().map_err(io)
}
