//! The batch kernels against the defining inequalities, decided exactly on an integer lattice.
//!
//! On small integers every product the kernels form is exact in `f64`, and a mask with
//! `sin(eps_min) = +-1/2` or a cone with `cos(alpha) = 3/4` squares exactly, so the kernels' verdict
//! is not "agrees within an ulp" but the mathematics itself, boundary cells included.
//! `derivations/kernel_lattice.g` counts the same lattice in GAP; the squarings it decides with are
//! `derivations/elevation_footprint.mac` (6) and `derivations/cone_footprint.mac` (3)-(5).

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::naive_bytecount,
    clippy::cast_precision_loss,
    clippy::many_single_char_names
)]

use spacelab_core::Vec3;
use spacelab_sensors::{
    AccessGeometry, Boresight, CellGrid, Cone, Constraint, ElevationMask,
    kernel::{cone_covered_into, elevation_accumulate_into, elevation_covered_into},
};

const AXES: [[i64; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];
const SATELLITES: [[i64; 3]; 3] = [[3, -2, 9], [-7, 5, 4], [8, 8, -8]];

/// `derivations/kernel_lattice.g`'s counts: per satellite, `[horizon, mask +30, mask -30,
/// nadir cone, fixed cone]`, of the 2197 cells.
const GAP_COUNTS: [[usize; 5]; 3] = [
    [1154, 523, 1669, 1781, 1481],
    [1155, 584, 1614, 1765, 270],
    [1099, 644, 1556, 2188, 0],
];

/// Every integer point of `[-6, 6]^3`, `x` outermost, with the zenith axis `(x + y + z) mod 6`.
fn lattice() -> Vec<([i64; 3], [i64; 3])> {
    let mut cells = Vec::new();
    for x in -6..=6_i64 {
        for y in -6..=6_i64 {
            for z in -6..=6_i64 {
                cells.push(([x, y, z], AXES[(x + y + z).rem_euclid(6) as usize]));
            }
        }
    }
    cells
}

fn dot(a: [i64; 3], b: [i64; 3]) -> i64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [i64; 3], b: [i64; 3]) -> [i64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3(a: [i64; 3]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}

/// The five defining inequalities, in exact integer arithmetic. See `kernel_lattice.g`.
fn exact(s: [i64; 3], cell: ([i64; 3], [i64; 3])) -> [bool; 5] {
    let (c, z) = cell;
    let d = sub(s, c);
    let (t, q) = (dot(d, z), dot(d, d));
    let n2 = dot(s, s);
    let ut = n2 - dot(c, s);
    let u = dot(sub(c, s), [0, 0, -1]);
    [
        t >= 0,
        t >= 0 && 4 * t * t >= q,
        t >= 0 || 4 * t * t <= q,
        ut >= 0 && 16 * ut * ut >= 9 * q * n2,
        u >= 0 && 16 * u * u >= 9 * q,
    ]
}

/// The double nearest `start` whose image under `f` is exactly `want`, within 16 ulp.
fn exact_preimage(start: f64, want: f64, f: impl Fn(f64) -> Option<f64>) -> f64 {
    let (mut down, mut up) = (start, start);
    for _ in 0..16 {
        for x in [down, up] {
            if f(x) == Some(want) {
                return x;
            }
        }
        down = down.next_down();
        up = up.next_up();
    }
    panic!("no double within 16 ulp of {start} maps to exactly {want}");
}

/// (G2) Every kernel output, and every scalar `covered`, is the defining inequality decided
/// exactly -- `eps >= eps_min` for masks with `sin(eps_min)` in `{0, 1/2, -1/2}`, `theta <= alpha`
/// for a cone with `cos(alpha) = 3/4` about the geocentric nadir and about a fixed axis -- on all 2197 lattice
/// cells for three satellites, horizon-grazing cells (`t = 0`) included; the per-satellite counts
/// are GAP's, and `elevation_accumulate_into` sums to GAP's totals cell by cell.
#[test]
fn the_kernels_decide_the_defining_inequalities_exactly_on_an_integer_lattice() {
    let thirty = exact_preimage(30.0_f64.to_radians(), 0.5, |e| {
        ElevationMask::new(e).ok().map(ElevationMask::sin_min)
    });
    let cone_angle = exact_preimage(0.75_f64.acos(), 0.75, |a| {
        Cone::new(a).ok().map(Cone::cos_half)
    });
    let masks = [
        ElevationMask::HORIZON,
        ElevationMask::new(thirty).expect("mask in range"),
        ElevationMask::new(-thirty).expect("mask in range"),
    ];
    assert_eq!(masks[2].sin_min(), -0.5, "sin is odd in libm");
    let cones = [
        Cone::new(cone_angle).expect("half-angle in range"),
        Cone::pointed(cone_angle, Boresight::Fixed(Vec3::new(0.0, 0.0, -1.0)))
            .expect("half-angle in range"),
    ];

    let cells = lattice();
    let n = cells.len();
    let col = |k: usize, zenith: bool| -> Vec<f64> {
        cells
            .iter()
            .map(|&(c, z)| if zenith { z[k] } else { c[k] } as f64)
            .collect()
    };
    let (px, py, pz) = (col(0, false), col(1, false), col(2, false));
    let (zx, zy, zz) = (col(0, true), col(1, true), col(2, true));
    let grid = CellGrid::new(&px, &py, &pz, &zx, &zy, &zz).expect("rectangular");

    let mut counters = [vec![0_u32; n], vec![0_u32; n], vec![0_u32; n]];
    for (si, &s) in SATELLITES.iter().enumerate() {
        let sat = vec3(s);
        let mut outs = vec![vec![0_u8; n]; 5];
        for (k, mask) in masks.iter().enumerate() {
            elevation_covered_into(*mask, sat, &grid, &mut outs[k]).expect("lengths match");
            elevation_accumulate_into(*mask, sat, &grid, &mut counters[k]).expect("lengths match");
        }
        for (k, cone) in cones.iter().enumerate() {
            cone_covered_into(*cone, sat, &grid, &mut outs[3 + k]).expect("lengths match");
        }
        for (i, &cell) in cells.iter().enumerate() {
            let want = exact(s, cell);
            let geom = AccessGeometry {
                satellite: sat,
                cell: vec3(cell.0),
                zenith: vec3(cell.1),
            };
            let scalar = [
                masks[0].covered(&geom),
                masks[1].covered(&geom),
                masks[2].covered(&geom),
                cones[0].covered(&geom),
                cones[1].covered(&geom),
            ];
            for k in 0..5 {
                assert_eq!(
                    outs[k][i] == 1,
                    want[k],
                    "satellite {s:?}, cell {cell:?}, predicate {k}: kernel"
                );
                assert_eq!(
                    scalar[k], want[k],
                    "satellite {s:?}, cell {cell:?}, predicate {k}"
                );
            }
        }
        for k in 0..5 {
            let count = outs[k].iter().filter(|&&b| b == 1).count();
            assert_eq!(count, GAP_COUNTS[si][k], "satellite {s:?}, predicate {k}");
        }
    }
    for (k, counter) in counters.iter().enumerate() {
        let gap_total: usize = GAP_COUNTS.iter().map(|row| row[k]).sum();
        assert_eq!(
            counter.iter().map(|&c| c as usize).sum::<usize>(),
            gap_total
        );
        for (i, &cell) in cells.iter().enumerate() {
            let want = SATELLITES.iter().filter(|&&s| exact(s, cell)[k]).count();
            assert_eq!(counter[i] as usize, want, "cell {cell:?}, mask {k}");
        }
    }
}
