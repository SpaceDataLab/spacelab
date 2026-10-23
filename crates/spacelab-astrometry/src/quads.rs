//! Blind identification: recognising a field with no prior pointing, from the *shape* of its
//! brightest asterisms.
//!
//! # The hash
//!
//! Take four stars. Call the most widely separated pair `A` and `B`, and put them at `(0,0)` and
//! `(1,1)` by the unique similarity transform that does so. The other two land somewhere, and their
//! four coordinates `(x_C, y_C, x_D, y_D)` are a **geometric hash**: unchanged by where the field is
//! on the sky, how the camera is rotated, and what the plate scale is — the three things a blind
//! solve does not know. Two quads match when their codes are close.
//!
//! This is Lang, Hogg, Mierle, Blanton & Roweis (2010), AJ **139**, 1782 — the algorithm behind
//! astrometry.net, whose implementation is BSD-3-Clause and therefore usable as an *oracle* to check
//! this one against. The method is described in the paper; the code here is written from it.
//!
//! Three conventions make the hash unique rather than merely invariant, and all three are symmetry
//! breaking rather than approximation:
//!
//! - `C` and `D` are ordered so that `x_C ≤ x_D`, since nothing distinguishes them;
//! - `A` and `B` are swapped, which maps every coordinate `t → 1 − t`, unless `x_C + x_D ≤ 1`;
//! - both `C` and `D` must lie inside the circle with `AB` as diameter, i.e. within `1/√2` of
//!   `(½, ½)`. A quad that fails this is discarded rather than fixed: outside that circle a small
//!   measurement error can reorder which pair is the widest, and the code then jumps.
//!
//! # Scaling
//!
//! Quads are built from all four-element subsets of the brightest `n` sources, so the count is
//! `C(n,4)` — 70 at `n = 8`, 210 at 10, 4845 at 20. The caller chooses `n`, and the cost of choosing
//! it too large is quadratic in the matching step that follows. Eight to twelve is the usual range;
//! astrometry.net's index files are built the same way and for the same reason.

use alloc::vec::Vec;

/// A four-star asterism and its scale- and rotation-invariant code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quad {
    /// `(x_C, y_C, x_D, y_D)` in the `AB` frame.
    pub code: [f64; 4],
    /// Indices into the point list, in the order `A, B, C, D`.
    pub stars: [usize; 4],
}

/// Distance between two codes, the largest absolute coordinate difference.
#[must_use]
pub fn code_distance(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| libm::fabs(p - q))
        .fold(0.0_f64, f64::max)
}

/// Build the hash for one four-element subset, or `None` if it does not satisfy the conventions
/// above.
#[must_use]
pub fn quad_code(points: &[(f64, f64)], indices: [usize; 4]) -> Option<Quad> {
    // The widest pair is A and B.
    let mut best = (f64::NEG_INFINITY, 0usize, 1usize);
    for i in 0..4 {
        for j in (i + 1)..4 {
            let (p, q) = (points[indices[i]], points[indices[j]]);
            let separation = (p.0 - q.0) * (p.0 - q.0) + (p.1 - q.1) * (p.1 - q.1);
            if separation > best.0 {
                best = (separation, i, j);
            }
        }
    }
    // `best.0` is never NaN and needs no separate test for one: it starts at −∞ and is only ever
    // assigned through `separation > best.0`, which is false for a NaN separation. So a field with a
    // non-finite coordinate leaves the accumulator at −∞, and four coincident stars leave it at 0;
    // one comparison rejects both.
    if best.0 <= 0.0 {
        return None;
    }
    let (mut a, mut b) = (indices[best.1], indices[best.2]);
    let others: Vec<usize> = indices
        .iter()
        .copied()
        .filter(|&index| index != a && index != b)
        .collect();
    if others.len() != 2 {
        return None; // a repeated index is not an asterism
    }
    let (mut c, mut d) = (others[0], others[1]);

    // Coordinates in the AB frame: (p,q) along and across AB, then the 45° similarity that puts B
    // at (1,1).
    let transform = |a: usize, b: usize, point: usize| -> (f64, f64) {
        let (ax, ay) = points[a];
        let (dx, dy) = (points[b].0 - ax, points[b].1 - ay);
        let squared = dx * dx + dy * dy;
        let (rx, ry) = (points[point].0 - ax, points[point].1 - ay);
        let along = (rx * dx + ry * dy) / squared;
        let across = (ry * dx - rx * dy) / squared;
        (along - across, along + across)
    };

    let mut code_c = transform(a, b, c);
    let mut code_d = transform(a, b, d);
    if code_c.0 > code_d.0 {
        core::mem::swap(&mut c, &mut d);
        core::mem::swap(&mut code_c, &mut code_d);
    }
    if code_c.0 + code_d.0 > 1.0 {
        core::mem::swap(&mut a, &mut b);
        code_c = transform(a, b, c);
        code_d = transform(a, b, d);
        if code_c.0 > code_d.0 {
            core::mem::swap(&mut c, &mut d);
            core::mem::swap(&mut code_c, &mut code_d);
        }
    }

    // Both must lie inside the circle on AB as diameter: |(x,y) − (½,½)| ≤ 1/√2.
    let inside = |(x, y): (f64, f64)| (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) <= 0.5;
    if !inside(code_c) || !inside(code_d) {
        return None;
    }

    Some(Quad {
        code: [code_c.0, code_c.1, code_d.0, code_d.1],
        stars: [a, b, c, d],
    })
}

/// Every valid quad among the first `brightest` points, which are assumed already sorted brightest
/// first — [`crate::sources::detect`] and [`crate::StarCatalogue::query`] both promise that order.
#[must_use]
pub fn build_quads(points: &[(f64, f64)], brightest: usize) -> Vec<Quad> {
    let n = brightest.min(points.len());
    let mut quads = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                for l in (k + 1)..n {
                    if let Some(quad) = quad_code(points, [i, j, k, l]) {
                        quads.push(quad);
                    }
                }
            }
        }
    }
    quads
}

/// A pair of quads whose codes agree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadMatch {
    /// Index into the image quads.
    pub image: usize,
    /// Index into the reference quads.
    pub reference: usize,
    /// Code distance between them.
    pub distance: f64,
}

/// Every pair of quads whose codes agree to `tolerance`.
///
/// The tolerance is in code units, which are fractions of the `AB` separation: a centroid error of
/// `σ` pixels on a quad whose stars are `L` pixels apart perturbs the code by roughly `σ/L`, so
/// `0.01` is the right order for a well-measured field and anything above `0.05` matches noise.
#[must_use]
pub fn match_quads(image: &[Quad], reference: &[Quad], tolerance: f64) -> Vec<QuadMatch> {
    let mut matches = Vec::new();
    for (i, left) in image.iter().enumerate() {
        for (j, right) in reference.iter().enumerate() {
            let distance = code_distance(&left.code, &right.code);
            if distance <= tolerance {
                matches.push(QuadMatch {
                    image: i,
                    reference: j,
                    distance,
                });
            }
        }
    }
    matches
}

/// Vote matching quads into star-to-star correspondences.
///
/// Each matching quad casts four votes, one per star in the order `A, B, C, D` — which is why the
/// canonical ordering above matters: two quads with the same code have their stars in the same
/// roles. A correspondence is kept when it is the most-voted partner *in both directions* and has at
/// least `min_votes`; that mutual test is what rejects the pair-up a single accidental code
/// collision would otherwise produce, since a spurious match rarely agrees with anything else.
#[must_use]
pub fn star_correspondences(
    matches: &[QuadMatch],
    image: &[Quad],
    reference: &[Quad],
    min_votes: usize,
) -> Vec<(usize, usize)> {
    let mut votes: Vec<(usize, usize, usize)> = Vec::new();
    for quad_match in matches {
        for role in 0..4 {
            let pair = (
                image[quad_match.image].stars[role],
                reference[quad_match.reference].stars[role],
            );
            if let Some(entry) = votes
                .iter_mut()
                .find(|(i, r, _)| *i == pair.0 && *r == pair.1)
            {
                entry.2 += 1;
            } else {
                votes.push((pair.0, pair.1, 1));
            }
        }
    }

    let best_for = |select: fn(&(usize, usize, usize)) -> usize, key: usize| -> Option<usize> {
        votes
            .iter()
            .filter(|entry| select(entry) == key)
            .max_by_key(|entry| entry.2)
            .map(|entry| entry.2)
    };

    votes
        .iter()
        .filter(|(image_star, reference_star, count)| {
            *count >= min_votes
                && best_for(|entry| entry.0, *image_star) == Some(*count)
                && best_for(|entry| entry.1, *reference_star) == Some(*count)
        })
        .map(|(image_star, reference_star, _)| (*image_star, *reference_star))
        .collect()
}
