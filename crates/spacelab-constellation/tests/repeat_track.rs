//! Gates of `repeat_track` (plan 18 Ф27.D): the gap sets against the sorted crossings, the
//! sub-cycles against their definition, and the swath rule against the rows
//! `derivations/repeat_track_gaps.gp` decides in exact arithmetic on the doubles passed here. The
//! `.gp` checks that every row below is what it prints.

#![expect(
    clippy::unreadable_literal,
    clippy::expect_used,
    clippy::type_complexity,
    reason = "the rows are copied verbatim from derivations/repeat_track_gaps.gp, which checks them \
              against its output, and a test fails loudly on a refused input"
)]

use spacelab_constellation::{Nodes, RepeatTrack, RepeatTrackError, SubCycle, Swath};

const RE: f64 = 6_378_137.0;

/// The crossings by their definition, in half-slots: the ascending node of revolution `k` at
/// `−2k·Nd`, the `j`-th crossing of either kind at `j·(Np − Nd)`, modulo `2Np`.
fn crossings(np: u32, nd: u32, n: u32, nodes: Nodes) -> Vec<i64> {
    let (np, nd, circle) = (i64::from(np), i64::from(nd), 2 * i64::from(np));
    let mut v: Vec<i64> = match nodes {
        Nodes::Ascending => (0..i64::from(n))
            .map(|k| (-2 * k * nd).rem_euclid(circle))
            .collect(),
        Nodes::Both => (0..2 * i64::from(n))
            .map(|j| (j * (np - nd)).rem_euclid(circle))
            .collect(),
    };
    v.sort_unstable();
    v.dedup();
    v
}

/// The gap multiset of sorted points on a circle of `circle` half-slots, as `(length, count)`.
fn sorted_gaps(points: &[i64], circle: i64) -> Vec<(u64, u64)> {
    let m = points.len();
    let mut gaps: Vec<i64> = (0..m)
        .map(|k| {
            if k + 1 < m {
                points[k + 1] - points[k]
            } else {
                circle - points[m - 1] + points[0]
            }
        })
        .collect();
    gaps.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for g in gaps {
        let g = u64::try_from(g).expect("a gap is positive");
        match out.last_mut() {
            Some(last) if last.0 == g => last.1 += 1,
            _ => out.push((g, 1)),
        }
    }
    out
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// (G3) For every `Np ≤ 60`, every `Nd < 2Np` prime to it, every number of revolutions up to one past
/// the cycle and both kinds of node, the gap set is the multiset of the sorted crossings; it has at
/// most three lengths, and with three the largest is the sum of the other two.
#[test]
fn every_gap_set_is_the_sorted_crossings() {
    let mut sets = 0_u64;
    for np in 1..=60_u32 {
        for nd in (1..2 * np).filter(|&nd| gcd(np, nd) == 1) {
            let track = RepeatTrack::new(np, nd).expect("a coprime pair");
            for n in 1..=np + 1 {
                for nodes in [Nodes::Ascending, Nodes::Both] {
                    let points = crossings(np, nd, n, nodes);
                    let want = sorted_gaps(&points, 2 * i64::from(np));
                    let got = track.gaps(n, nodes).expect("n >= 1");
                    let got_pairs: Vec<(u64, u64)> =
                        got.gaps().iter().map(|g| (g.length, g.count)).collect();
                    assert_eq!(
                        got_pairs, want,
                        "{np}/{nd} after {n} revolutions, {nodes:?}"
                    );
                    assert_eq!(got.crossings(), points.len() as u64);
                    assert_eq!(got.largest(), want.last().expect("one gap at least").0);
                    assert!(want.len() <= 3, "{np}/{nd} n = {n}: {want:?}");
                    if want.len() == 3 {
                        assert_eq!(want[2].0, want[0].0 + want[1].0, "{np}/{nd} n = {n}");
                    }
                    sets += 1;
                }
            }
        }
    }
    eprintln!("gap sets equal to the sorted crossings: {sets}");
}

/// (G3) The sub-cycles are the record returns of the ascending node, by their definition, for every
/// `Np ≤ 150` and `Nd < 2Np` prime to it; the last is the whole cycle at offset 0.
#[test]
fn sub_cycles_are_the_record_returns() {
    for np in 1..=150_u32 {
        for nd in (1..2 * np).filter(|&nd| gcd(np, nd) == 1) {
            let (p, d) = (i64::from(np), i64::from(nd));
            let mut want = Vec::new();
            let mut best = p;
            for k in 1..=p {
                let r = (-k * d).rem_euclid(p);
                let offset = if 2 * r <= p { r } else { r - p };
                if offset.abs() < best {
                    best = offset.abs();
                    want.push(SubCycle {
                        revolutions: u32::try_from(k).expect("k <= Np"),
                        offset,
                    });
                }
            }
            let got: Vec<SubCycle> = RepeatTrack::new(np, nd)
                .expect("coprime")
                .sub_cycles()
                .collect();
            assert_eq!(got, want, "{np}/{nd}");
            assert_eq!(got.last().map(|c| (c.revolutions, c.offset)), Some((np, 0)));
        }
    }
}

/// `derivations/repeat_track_gaps.gp`: `(Np, Nd, [(revolution, offset slots east)])`.
#[rustfmt::skip]
const SUB_CYCLE_ROWS: &[(u32, u32, &[(u32, i64)])] = &[
    (233, 16, &[(1, -16), (14, 9), (15, -7), (29, 2), (102, -1), (233, 0)]),
    (143, 10, &[(1, -10), (14, 3), (43, -1), (143, 0)]),
    (175, 12, &[(1, -12), (14, 7), (15, -5), (29, 2), (73, -1), (175, 0)]),
    (385, 27, &[(1, -27), (14, 7), (43, -6), (57, 1), (385, 0)]),
    (15, 1, &[(1, -1), (15, 0)]),
    (29, 2, &[(1, -2), (14, 1), (29, 0)]),
];

/// `derivations/repeat_track_gaps.gp`, by sorting: `(Np, Nd, revolutions, nodes, [(length in
/// half-slots, count)])`. 143/10 after 14 revolutions, both kinds, is M0-10's `{7, 13}`.
#[rustfmt::skip]
const GAP_ROWS: &[(u32, u32, u32, Nodes, &[(u64, u64)])] = &[
    (143, 10, 14, Nodes::Ascending, &[(20, 13), (26, 1)]),
    (143, 10, 14, Nodes::Both, &[(7, 13), (13, 15)]),
    (143, 10, 43, Nodes::Ascending, &[(6, 29), (8, 14)]),
    (143, 10, 43, Nodes::Both, &[(1, 43), (5, 15), (6, 28)]),
    (233, 16, 14, Nodes::Ascending, &[(32, 13), (50, 1)]),
    (233, 16, 14, Nodes::Both, &[(7, 13), (25, 15)]),
    (233, 16, 15, Nodes::Ascending, &[(18, 1), (32, 14)]),
    (233, 16, 15, Nodes::Both, &[(7, 15), (18, 2), (25, 13)]),
    (233, 16, 29, Nodes::Ascending, &[(14, 14), (18, 15)]),
    (233, 16, 29, Nodes::Both, &[(7, 43), (11, 15)]),
    (233, 16, 102, Nodes::Ascending, &[(4, 73), (6, 29)]),
    (233, 16, 102, Nodes::Both, &[(1, 73), (3, 131)]),
    (233, 16, 233, Nodes::Ascending, &[(2, 233)]),
    (233, 16, 233, Nodes::Both, &[(1, 466)]),
    (175, 12, 100, Nodes::Ascending, &[(2, 27), (4, 71), (6, 2)]),
    (175, 12, 100, Nodes::Both, &[(1, 127), (3, 69), (4, 4)]),
    (385, 27, 200, Nodes::Ascending, &[(2, 143), (8, 43), (10, 14)]),
    (385, 27, 200, Nodes::Both, &[(2, 385)]),
    (15, 1, 7, Nodes::Ascending, &[(2, 6), (18, 1)]),
    (15, 1, 7, Nodes::Both, &[(2, 13), (4, 1)]),
    (15, 1, 15, Nodes::Ascending, &[(2, 15)]),
    (15, 1, 15, Nodes::Both, &[(2, 15)]),
    (43, 3, 20, Nodes::Ascending, &[(2, 6), (4, 5), (6, 9)]),
    (43, 3, 20, Nodes::Both, &[(2, 37), (4, 3)]),
    (1, 1, 1, Nodes::Ascending, &[(2, 1)]),
    (1, 1, 1, Nodes::Both, &[(2, 1)]),
    (1, 2, 1, Nodes::Ascending, &[(2, 1)]),
    (1, 2, 1, Nodes::Both, &[(1, 2)]),
    (2, 1, 2, Nodes::Ascending, &[(2, 2)]),
    (2, 1, 2, Nodes::Both, &[(1, 4)]),
];

/// (G3) The sub-cycles and gap sets of the `.gp`'s rows: Landsat's 233/16 has its classic sub-cycle
/// at revolution 102 (7.004 days, one slot west).
#[test]
fn rows_of_the_pari_file() {
    for &(np, nd, want) in SUB_CYCLE_ROWS {
        let got: Vec<(u32, i64)> = RepeatTrack::new(np, nd)
            .expect("coprime")
            .sub_cycles()
            .map(|c| (c.revolutions, c.offset))
            .collect();
        assert_eq!(got, want, "{np}/{nd}");
    }
    for &(np, nd, n, nodes, want) in GAP_ROWS {
        let got = RepeatTrack::new(np, nd)
            .expect("coprime")
            .gaps(n, nodes)
            .expect("n >= 1");
        let got: Vec<(u64, u64)> = got.gaps().iter().map(|g| (g.length, g.count)).collect();
        assert_eq!(got, want, "{np}/{nd} after {n}, {nodes:?}");
    }
}

/// `derivations/repeat_track_gaps.gp`, decided in exact arithmetic with `κ = Nd/Np`: `(Np, Nd,
/// nodes, i, R, W, the first revolution whose swaths cover the equator, 0 for none)`. Each is at
/// least 3.6e-2 from its boundary, relative, at that revolution and the one before.
#[rustfmt::skip]
const SWATH_ROWS: &[(u32, u32, Nodes, f64, f64, f64, u32)] = &[
    (233, 16, Nodes::Ascending, 1.7139133254584316e0, 6.378137e6, 1.85e5, 233),
    (233, 16, Nodes::Both, 1.7139133254584316e0, 6.378137e6, 1.85e5, 168),
    (143, 10, Nodes::Ascending, 1.7212437083168077e0, 6.378137e6, 2.9e5, 143),
    (143, 10, Nodes::Both, 1.7212437083168077e0, 6.378137e6, 2.9e5, 122),
    (175, 12, Nodes::Ascending, 1.7135642596080327e0, 6.378137e6, 2.5e5, 175),
    (175, 12, Nodes::Both, 1.7135642596080327e0, 6.378137e6, 2.5e5, 139),
    (385, 27, Nodes::Ascending, 1.7217673070924062e0, 6.378137e6, 1.27e6, 57),
    (385, 27, Nodes::Both, 1.7217673070924062e0, 6.378137e6, 1.27e6, 22),
    (15, 1, Nodes::Ascending, 9.005898940290741e-1, 6.378137e6, 2e6, 0),
    (15, 1, Nodes::Both, 9.005898940290741e-1, 6.378137e6, 2e6, 0),
    (43, 3, Nodes::Ascending, 1.1344640137963142e0, 6.378137e6, 9e5, 43),
    (43, 3, Nodes::Both, 1.1344640137963142e0, 6.378137e6, 9e5, 22),
    (233, 16, Nodes::Ascending, 1.7139133254584316e0, 6.378137e6, 2e4, 0),
    (233, 16, Nodes::Both, 1.7139133254584316e0, 6.378137e6, 2e4, 0),
];

/// `derivations/repeat_track_gaps.gp`: `(Np, Nd, nodes, i, R, n, W below, W nearest, W above)`,
/// where `W* = 2R·asin(sin(x_g)·sin i′)` at revolution `n`; below and above are `W*(1 ∓ 1e-12)`.
#[rustfmt::skip]
const BOUNDARY_ROWS: &[(u32, u32, Nodes, f64, f64, u32, f64, f64, f64)] = &[
    (233, 16, Nodes::Ascending, 1.7139133254584316e0, 6.378137e6, 102, 5.04610705829261e5, 5.046107058297656e5, 5.046107058302702e5),
    (233, 16, Nodes::Both, 1.7139133254584316e0, 6.378137e6, 233, 8.410275682088312e4, 8.410275682096723e4, 8.410275682105133e4),
    (143, 10, Nodes::Both, 1.7212437083168077e0, 6.378137e6, 43, 8.206689013756815e5, 8.206689013765022e5, 8.206689013773229e5),
    (15, 1, Nodes::Ascending, 9.005898940290741e-1, 6.378137e6, 15, 2.1756017515106895e6, 2.175601751512865e6, 2.1756017515150406e6),
];

/// `derivations/repeat_track_gaps.gp`: `(Np, Nd, i, R, W, 2R·asin(sin(W/2R)/sin i′))` at 60 digits,
/// the nearest double; `R = 1e12` is the planar limit.
#[rustfmt::skip]
const WIDTH_ROWS: &[(u32, u32, f64, f64, f64, f64)] = &[
    (233, 16, 1.7139133254584316e0, 6.378137e6, 1.85e5, 1.8916894517984774e5),
    (143, 10, 1.7212437083168077e0, 6.378137e6, 2.9e5, 2.970817453043482e5),
    (15, 1, 9.005898940290741e-1, 6.378137e6, 2e6, 2.45507361620678e6),
    (43, 3, 1.1344640137963142e0, 6.378137e6, 9e5, 9.659253766485509e5),
    (233, 16, 1.7139133254584316e0, 1e12, 1.85e5, 1.891686429406979e5),
    (29, 2, 2.0943951023931957e0, 6.378137e6, 3e6, 3.6042905306545133e6),
];

/// (G1, rows from the `.gp`) The first covering revolution matches the exact decision on every row,
/// and agrees with [`RepeatTrack::equator_covered`] there and one revolution before.
#[test]
fn the_swath_rule_decides_the_rows() {
    for &(np, nd, nodes, inclination, radius, width, want) in SWATH_ROWS {
        let track = RepeatTrack::new(np, nd).expect("coprime");
        let swath = Swath {
            width,
            radius,
            inclination,
        };
        let got = track
            .revolutions_to_cover_equator(nodes, &swath)
            .expect("in domain");
        assert_eq!(got.unwrap_or(0), want, "{np}/{nd} {nodes:?} W = {width}");
        if want > 0 {
            assert!(
                track
                    .equator_covered(want, nodes, &swath)
                    .expect("in domain")
            );
            if want > 1 {
                assert!(
                    !track
                        .equator_covered(want - 1, nodes, &swath)
                        .expect("in domain")
                );
            }
        }
    }
}

/// (rule 13) At the boundary the decision errs towards "open": below `W*` and at the double nearest
/// it the gap is reported open, and `1e-12` above it covered.
#[test]
fn the_swath_rule_errs_towards_open_at_its_boundary() {
    for &(np, nd, nodes, inclination, radius, n, below, nearest, above) in BOUNDARY_ROWS {
        let track = RepeatTrack::new(np, nd).expect("coprime");
        let at = |width| {
            track
                .equator_covered(
                    n,
                    nodes,
                    &Swath {
                        width,
                        radius,
                        inclination,
                    },
                )
                .expect("in domain")
        };
        assert!(!at(below), "{np}/{nd} {nodes:?} below the boundary");
        assert!(
            !at(nearest),
            "{np}/{nd} {nodes:?} at the boundary: the safe side is open"
        );
        assert!(at(above), "{np}/{nd} {nodes:?} above the boundary");
    }
}

/// (G1, G3) The equator one crossing's swath covers equals `2R·asin(sin(W/2R)/sin i′)` at 60 digits,
/// and at `R = 1e12` it is the planar limit `W/sin i′`.
#[test]
fn the_covered_equator_is_the_rule_and_its_planar_limit() {
    for &(np, nd, inclination, radius, width, want) in WIDTH_ROWS {
        let track = RepeatTrack::new(np, nd).expect("coprime");
        let got = track
            .covered_equator(&Swath {
                width,
                radius,
                inclination,
            })
            .expect("in domain");
        assert!(
            (got - want).abs() <= 8.0 * f64::EPSILON * want,
            "{np}/{nd}: {got} vs {want}"
        );
        if radius > 1e11 {
            let kappa = f64::from(nd) / f64::from(np);
            let sin_ip = inclination.sin() / inclination.sin().hypot(inclination.cos() - kappa);
            assert!((got - width / sin_ip).abs() < 1e-13 * got, "planar limit");
        }
    }
}

/// (S2) The first covering revolution from the incremental scan equals the first `n` at which
/// [`RepeatTrack::equator_covered`] holds, over a sweep of tracks, inclinations and swaths.
#[test]
fn the_incremental_scan_agrees_with_the_gap_sets() {
    for np in [7_u32, 15, 29, 43, 101, 143, 233] {
        for nd in (1..np).filter(|&nd| gcd(np, nd) == 1).step_by(3) {
            let track = RepeatTrack::new(np, nd).expect("coprime");
            for deg in [30.0_f64, 63.4, 97.8, 120.0] {
                for frac in [0.3, 0.7, 1.1, 2.5] {
                    let width = frac * std::f64::consts::TAU * RE / f64::from(np);
                    let swath = Swath {
                        width,
                        radius: RE,
                        inclination: deg.to_radians(),
                    };
                    for nodes in [Nodes::Ascending, Nodes::Both] {
                        let Ok(scan) = track.revolutions_to_cover_equator(nodes, &swath) else {
                            continue;
                        };
                        let direct = (1..=np)
                            .find(|&n| track.equator_covered(n, nodes, &swath).expect("in domain"));
                        assert_eq!(scan, direct, "{np}/{nd} i = {deg} W = {width} {nodes:?}");
                    }
                }
            }
        }
    }
}

/// (S2, rule 23) Input outside the domain is a typed refusal.
#[test]
fn refusals_are_typed() {
    assert_eq!(
        RepeatTrack::new(143, 11),
        Err(RepeatTrackError::NotCoprime { gcd: 11 })
    );
    assert_eq!(RepeatTrack::new(0, 1), Err(RepeatTrackError::ZeroCount));
    assert_eq!(RepeatTrack::new(14, 0), Err(RepeatTrackError::ZeroCount));
    let track = RepeatTrack::new(233, 16).expect("coprime");
    assert_eq!(
        track.gaps(0, Nodes::Both),
        Err(RepeatTrackError::NoRevolutions)
    );
    assert_eq!(
        track.equator_covered(
            0,
            Nodes::Both,
            &Swath {
                width: 1e5,
                radius: RE,
                inclination: 1.7
            }
        ),
        Err(RepeatTrackError::NoRevolutions)
    );
    for inclination in [
        0.0,
        std::f64::consts::PI,
        -0.1,
        3.5,
        f64::NAN,
        f64::INFINITY,
    ] {
        let swath = Swath {
            width: 1e5,
            radius: RE,
            inclination,
        };
        assert_eq!(
            track.covered_equator(&swath),
            Err(RepeatTrackError::BadInclination)
        );
    }
    for (width, radius) in [
        (0.0, RE),
        (-1.0, RE),
        (f64::NAN, RE),
        (1e5, 0.0),
        (1e5, f64::INFINITY),
    ] {
        let swath = Swath {
            width,
            radius,
            inclination: 1.7,
        };
        assert_eq!(
            track.revolutions_to_cover_equator(Nodes::Ascending, &swath),
            Err(RepeatTrackError::BadSwath)
        );
    }
    // 3000 km of swath across a track at 5 deg reaches past the tangent circle's 5 deg; so does a
    // half-swath of a quarter turn or more.
    for (width, inclination) in [
        (3.0e6, 5_f64.to_radians()),
        (std::f64::consts::PI * RE, 1.7),
    ] {
        let swath = Swath {
            width,
            radius: RE,
            inclination,
        };
        assert_eq!(
            track.covered_equator(&swath),
            Err(RepeatTrackError::SwathBeyondTrack)
        );
    }
}
