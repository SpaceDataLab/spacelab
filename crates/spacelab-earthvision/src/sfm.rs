//! Structure-from-motion **feature tracks** — the multi-view correspondence graph that turns pairwise
//! matches into the point tracks bundle adjustment optimises.
//!
//! A track is one 3-D point seen in several views. [`build_tracks`] forms them by **union-find** over
//! the `(view, corner)` nodes: every pairwise match unions its two endpoints, and each connected
//! component becomes a track collecting one `(view, pixel)` observation per view. Components that put
//! two corners of the *same* view into one track are inconsistent (a point cannot appear twice in one
//! image) and are dropped. [`triangulate_tracks`] then turns each track into a 3-D point with
//! [`crate::camera::triangulate`]. `no_std` + `alloc`.

// Short names (a, b, c, v) index views/corners; the union-find is index arithmetic.
#![allow(clippy::many_single_char_names, clippy::doc_markdown)]

use crate::camera::{PinholeCamera, triangulate};
use crate::features::Corner;
use alloc::vec;
use alloc::vec::Vec;
use spacelab_core::Vec3;

/// One image observation in a track: the view it came from and its pixel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation2d {
    /// Index of the view.
    pub view: usize,
    /// Pixel column.
    pub x: f64,
    /// Pixel row.
    pub y: f64,
}

/// A feature track — one 3-D point's observations across views (at most one per view).
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    /// The `(view, pixel)` observations, in ascending view order.
    pub observations: Vec<Observation2d>,
}

/// Union-find with path compression and union by size.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        let (big, small) = if self.size[ra] >= self.size[rb] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }
}

/// Build feature tracks from per-view corner lists and pairwise `matches`, each a
/// `(view_a, corner_a, view_b, corner_b)` tuple. Only tracks with ≥2 consistent observations (at most
/// one per view) are returned. The contract is the set of tracks, not the order they come back in —
/// that follows the union-find representatives and so the balancing heuristic.
///
/// # Panics
///
/// If a match names a view or corner index outside `corners_per_view`.
#[must_use]
pub fn build_tracks(
    corners_per_view: &[&[Corner]],
    matches: &[(usize, usize, usize, usize)],
) -> Vec<Track> {
    // Global node ids: node(view, corner) = offset[view] + corner.
    let mut offset = vec![0usize; corners_per_view.len() + 1];
    for (v, cs) in corners_per_view.iter().enumerate() {
        offset[v + 1] = offset[v] + cs.len();
    }
    let total = *offset.last().unwrap_or(&0);
    let mut uf = UnionFind::new(total);
    for &(va, ca, vb, cb) in matches {
        uf.union(offset[va] + ca, offset[vb] + cb);
    }

    // Group nodes by root, then materialise each group's observations.
    let mut by_root: Vec<Vec<usize>> = vec![Vec::new(); total];
    for node in 0..total {
        let r = uf.find(node);
        by_root[r].push(node);
    }
    let node_view = |node: usize| offset.partition_point(|&o| o <= node) - 1;

    let mut tracks = Vec::new();
    for group in by_root {
        if group.len() < 2 {
            continue;
        }
        let mut obs = Vec::with_capacity(group.len());
        let mut seen_views = Vec::new();
        let mut consistent = true;
        for &node in &group {
            let v = node_view(node);
            if seen_views.contains(&v) {
                consistent = false; // two corners of the same view — drop the track
                break;
            }
            seen_views.push(v);
            let c = &corners_per_view[v][node - offset[v]];
            obs.push(Observation2d {
                view: v,
                x: c.x,
                y: c.y,
            });
        }
        if consistent && obs.len() >= 2 {
            obs.sort_by_key(|o| o.view);
            tracks.push(Track { observations: obs });
        }
    }
    tracks
}

/// Triangulate each track to a 3-D point using its views' `cameras` (indexed by `Observation2d::view`).
/// `None` where the track's rays are degenerate.
#[must_use]
pub fn triangulate_tracks(tracks: &[Track], cameras: &[PinholeCamera]) -> Vec<Option<Vec3>> {
    tracks
        .iter()
        .map(|t| {
            let views: Vec<(&PinholeCamera, [f64; 2])> = t
                .observations
                .iter()
                .map(|o| (&cameras[o.view], [o.x, o.y]))
                .collect();
            triangulate(&views)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
    use super::*;
    use spacelab_core::Mat3;

    fn cam(rot: Mat3, center: Vec3) -> PinholeCamera {
        PinholeCamera {
            rotation: rot,
            center,
            focal: 1000.0,
            principal: [320.0, 240.0],
        }
    }

    #[test]
    fn tracks_recover_a_multi_view_scene() {
        // Four cameras all seeing eight 3-D points. Each view's corners are the projected points;
        // consecutive views are matched corner-to-corner. Tracks must group all four views per point
        // and triangulate back to the truth.
        let cameras = [
            cam(Mat3::IDENTITY, Vec3::new(0.0, 0.0, 0.0)),
            cam(Mat3::rot2(0.2), Vec3::new(5.0, 0.0, 0.0)),
            cam(Mat3::rot2(-0.2), Vec3::new(-4.0, 1.0, 0.0)),
            cam(Mat3::rot2(0.1) * Mat3::rot1(0.1), Vec3::new(2.0, -3.0, 0.0)),
        ];
        let truth: Vec<Vec3> = (0..8)
            .map(|k| {
                Vec3::new(
                    f64::from(k) - 3.5,
                    0.5 * f64::from(k % 3),
                    12.0 + f64::from(k % 4),
                )
            })
            .collect();
        // Per-view corner lists = projected points (stable ordering k).
        let corner_lists: Vec<Vec<Corner>> = cameras
            .iter()
            .map(|c| {
                truth
                    .iter()
                    .map(|p| {
                        let px = c.project(*p);
                        Corner {
                            x: px[0],
                            y: px[1],
                            response: 1.0,
                        }
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<&[Corner]> = corner_lists.iter().map(Vec::as_slice).collect();
        // Match consecutive views corner-to-corner (the true correspondence).
        let mut matches = Vec::new();
        for v in 0..cameras.len() - 1 {
            for k in 0..truth.len() {
                matches.push((v, k, v + 1, k));
            }
        }

        let tracks = build_tracks(&refs, &matches);
        assert_eq!(tracks.len(), truth.len());
        assert!(tracks.iter().all(|t| t.observations.len() == cameras.len()));

        let points = triangulate_tracks(&tracks, &cameras);
        // One entry per track, and every well-conditioned track resolves — `flatten()` below would
        // otherwise be happy with an empty answer.
        assert_eq!(points.len(), tracks.len());
        assert!(points.iter().all(Option::is_some), "{points:?}");
        // Each recovered point matches one truth point tightly.
        for p in points.iter().flatten() {
            let best = truth
                .iter()
                .map(|t| (*t - *p).norm())
                .fold(f64::MAX, f64::min);
            assert!(best < 1e-6, "point {p:?} not near any truth");
        }
    }

    #[test]
    fn same_view_conflict_drops_the_track() {
        // One view, two corners, joined into a track — inconsistent (a point can't appear twice in one
        // image), so no track survives; and a clean 2-view match yields exactly one track.
        let a = [
            Corner {
                x: 1.0,
                y: 2.0,
                response: 1.0,
            },
            Corner {
                x: 3.0,
                y: 4.0,
                response: 1.0,
            },
        ];
        let b = [Corner {
            x: 5.0,
            y: 6.0,
            response: 1.0,
        }];
        let refs: Vec<&[Corner]> = vec![&a, &b];
        // corner 0 and corner 1 of view 0 unioned via view 1's single corner → same-view conflict.
        let conflict = build_tracks(&refs, &[(0, 0, 1, 0), (0, 1, 1, 0)]);
        assert_eq!(conflict.len(), 0);
        let clean = build_tracks(&refs, &[(0, 0, 1, 0)]);
        assert_eq!(clean.len(), 1);
        assert_eq!(clean[0].observations.len(), 2);
    }

    #[test]
    fn a_repeated_view_found_late_still_drops_the_whole_track() {
        // Nodes {v0c0, v1c0, v1c1}. Scanned in node order, two observations are already in hand when
        // the repeated view turns up, so a rule that let the observation count alone rescue the
        // group would emit a track built from a match it had just judged inconsistent. The conflict
        // test above cannot see that: there the repeat is the second node, so only one observation
        // survives the break.
        let a = [Corner {
            x: 1.0,
            y: 2.0,
            response: 1.0,
        }];
        let b = [
            Corner {
                x: 5.0,
                y: 6.0,
                response: 1.0,
            },
            Corner {
                x: 7.0,
                y: 8.0,
                response: 1.0,
            },
        ];
        let refs: Vec<&[Corner]> = vec![&a, &b];
        let tracks = build_tracks(&refs, &[(0, 0, 1, 0), (1, 0, 1, 1)]);
        assert!(tracks.is_empty(), "{tracks:?}");
    }
}
