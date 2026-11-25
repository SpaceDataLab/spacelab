//! Zones on the WGS-84 ellipsoid: closed regions bounded by geodesic edges (Ф31.C).
//!
//! # The model
//!
//! A zone is a list of vertices, geodetic latitude and longitude on WGS 84. Its edges are the
//! shortest geodesics between consecutive vertices, the last back to the first
//! ([`spacelab_earth::geodesic_inverse`], Ф20.U). The interior lies on the **left** of each
//! directed edge: the vertices run counter-clockwise seen from outside the ellipsoid, as in
//! `GeoJSON` (RFC 7946 §3.1.6) and S2. Reversing the order gives the complement, so a zone larger
//! than a hemisphere is as legal as a small one.
//!
//! The zone is closed. A vertex, and a point within [`ON_EDGE_M`] of an edge measured across it,
//! belong to the zone, and so to its complement too: the boundary is shared.
//!
//! A latitude of `±FRAC_PI_2` is read as the geodesic code reads it, the point `6e-17` rad from the
//! pole on its own meridian, 0.4 nm away. Nothing here treats a pole as special: a vertex there,
//! or an edge over one, is a point or a geodesic that passes within nanometres of it, and which
//! side of the pole that is decides nothing farther away.
//!
//! # Refusals
//!
//! [`Zone::new`] refuses fewer than three vertices, a coordinate that is not finite or a latitude
//! outside `[−π/2, π/2]`, an edge whose geodesic is not unique, and edges that meet anywhere but at
//! the vertex two consecutive edges share. An edge's geodesic is not unique when its endpoints
//! coincide, or when they lie on each other's cut locus. On an oblate ellipsoid the cut locus of a
//! point is an arc of the opposite parallel through its antipode, of half-length at most `fπ` of
//! the auxiliary sphere (Karney 2013, §5: `λ₁₂ ≥ π − fπ A₃ cos β₁`). An edge is refused when its
//! endpoints come within [`ANTIPODAL_MARGIN_RAD`] `= 2fπ` of antipodal on the auxiliary sphere,
//! twice that reach: inside it the shortest geodesic is not unique, or a nanometre of an endpoint
//! swings its azimuth.
//!
//! # Point in zone
//!
//! The test counts crossings of the meridian through the point, north of it
//! (`derivations/zone_meridian_crossing.mac`). Along a geodesic that is not a meridian the
//! longitude moves monotonically, because Clairaut's `N cos φ sin α` keeps the sign of `sin α`;
//! the shortest geodesic sweeps the reduced longitude difference of its ends, at most π. So an
//! edge crosses a meridian at most once, at a latitude found by Brent's method on the longitude.
//! A vertex on the meridian is taken as lying an infinitesimal step west of it, which makes the
//! count consistent through vertices. An edge heading east has the interior to its north, one
//! heading west to its south; the highest crossing of a reference meridian through no vertex
//! therefore says whether the north pole is inside, and the parity of the crossings above a point
//! says whether the point is on the pole's side.
//!
//! Two shortest geodesics meet at most once unless they overlap: if they met twice, the two arcs
//! between the meetings would both be shortest, and an interior sub-arc of a shortest geodesic is
//! its only shortest path. So two edges over a common range of longitude meet exactly when the
//! difference of their latitudes changes sign or vanishes across it, which is how [`Zone::new`]
//! finds self-intersections, exactly: a meeting the rounding of a latitude hides is not found.

use core::f64::consts::{FRAC_PI_2, PI, TAU};
use core::fmt;
use std::collections::BinaryHeap;

use spacelab_core::{Angle, Length};
use spacelab_coverage::brent;
use spacelab_earth::{
    EQUATORIAL_RADIUS, FLATTENING, GeodesicLine, POLAR_RADIUS, SurfacePoint, geodesic_inverse,
};

/// How close to antipodal, on the auxiliary sphere, an edge's endpoints may come: `2fπ` rad,
/// twice the half-length of the cut locus (the module documentation). About 134 km on the surface.
pub const ANTIPODAL_MARGIN_RAD: f64 = 2.0 * PI * FLATTENING;

/// The distance across an edge, metres, within which a point is on it, and so in the zone. A
/// thousand times what the geodesic code agrees with itself to, and far below any geodetic
/// significance.
pub const ON_EDGE_M: f64 = 1e-6;

/// `1 − f`.
const F1: f64 = 1.0 - FLATTENING;

/// `e² = f(2 − f)`.
const E2: f64 = FLATTENING * (2.0 - FLATTENING);

/// The smallest radius of curvature on WGS 84, `b²/a`, the meridian's at the equator, m.
const RHO_MIN: f64 = POLAR_RADIUS * POLAR_RADIUS / EQUATORIAL_RADIUS;

/// What an edge's latitude range is widened by, rad: twice [`ON_EDGE_M`] over the smallest radius
/// of curvature, so that every point within [`ON_EDGE_M`] of an edge falls inside its range.
const LAT_PAD: f64 = 2.0 * ON_EDGE_M / RHO_MIN;

/// The tolerance of a crossing's root, metres along the edge. A latitude moves by at most this
/// over the smallest radius of curvature, `2e-16` rad.
const CROSSING_TOL_M: f64 = 1e-9;

/// Why a zone, or a point asked about it, was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneError {
    /// Fewer than three vertices.
    TooFewVertices,
    /// A vertex's coordinate is not finite, or its latitude is outside `[−π/2, π/2]`.
    Vertex {
        /// The vertex, by its index.
        index: usize,
    },
    /// The geodesic of an edge is not unique: its endpoints coincide, or they come within
    /// [`ANTIPODAL_MARGIN_RAD`] of antipodal on the auxiliary sphere.
    Edge {
        /// The edge, by the index of its first vertex.
        index: usize,
    },
    /// Two edges meet other than consecutive edges at the vertex they share.
    SelfIntersecting {
        /// The first edge, by the index of its first vertex.
        first: usize,
        /// The second edge, by the index of its first vertex.
        second: usize,
    },
    /// A point asked about has a coordinate that is not finite, or a latitude outside
    /// `[−π/2, π/2]`.
    Point,
}

impl fmt::Display for ZoneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewVertices => write!(f, "a zone needs three or more vertices"),
            Self::Vertex { index } => write!(
                f,
                "vertex {index}: a coordinate is not finite or the latitude is outside \
                 [-pi/2, pi/2]"
            ),
            Self::Edge { index } => write!(
                f,
                "edge {index}: its endpoints coincide or are nearly antipodal, so its geodesic is \
                 not unique"
            ),
            Self::SelfIntersecting { first, second } => {
                write!(f, "edges {first} and {second} intersect")
            }
            Self::Point => write!(
                f,
                "a point's coordinate is not finite or its latitude is outside [-pi/2, pi/2]"
            ),
        }
    }
}

impl std::error::Error for ZoneError {}

/// `x` reduced to `(−π, π]`. `remainder` is exact, so a value already in the range comes back
/// unchanged, bit for bit.
pub(crate) fn reduce(x: f64) -> f64 {
    let r = libm::remainder(x, TAU);
    if r <= -PI { r + TAU } else { r }
}

/// Whether a latitude and longitude are a point on the ellipsoid.
fn valid(latitude: f64, longitude: f64) -> bool {
    (-FRAC_PI_2..=FRAC_PI_2).contains(&latitude) && longitude.is_finite()
}

/// The unit vector of geodetic `(lat, lon)` on the auxiliary sphere, at the reduced latitude
/// `tan β = (1 − f) tan φ`.
fn auxiliary(lat: f64, lon: f64) -> [f64; 3] {
    let (s, c) = libm::sincos(lat);
    let h = libm::hypot(F1 * s, c);
    let (sb, cb) = (F1 * s / h, c / h);
    let (sl, cl) = libm::sincos(lon);
    [cb * cl, cb * sl, sb]
}

/// The point `(lat, lon)` of the ellipsoid, Earth-fixed, m.
pub(crate) fn surface(lat: f64, lon: f64) -> [f64; 3] {
    let (s, c) = libm::sincos(lat);
    let (sl, cl) = libm::sincos(lon);
    let n = EQUATORIAL_RADIUS / libm::sqrt(1.0 - E2 * s * s);
    [n * c * cl, n * c * sl, n * (1.0 - E2) * s]
}

fn distance(p: [f64; 3], q: [f64; 3]) -> f64 {
    libm::hypot(libm::hypot(p[0] - q[0], p[1] - q[1]), p[2] - q[2])
}

/// How far `p` lies across the line through `(lat, lon)` with forward azimuth `az`, in the tangent
/// plane there, m: the component of `p − c` along the left of the direction of travel,
/// `n × t = sin α e_N − cos α e_E` (`derivations/zone_meridian_crossing.mac`). The point and its
/// azimuth come from one evaluation of the line, so near a pole, where a crossing's longitude is
/// ill-conditioned, they still describe one direction.
fn across(p: [f64; 3], lat: f64, lon: f64, az: f64) -> f64 {
    let c = surface(lat, lon);
    let (s, cp) = libm::sincos(lat);
    let (sl, cl) = libm::sincos(lon);
    let (sa, ca) = libm::sincos(az);
    let left = [-sa * s * cl + ca * sl, -sa * s * sl - ca * cl, sa * cp];
    ((p[0] - c[0]) * left[0] + (p[1] - c[1]) * left[1] + (p[2] - c[2]) * left[2]).abs()
}

/// A point on a geodesic line: latitude, longitude and forward azimuth, rad.
fn point_at(line: &GeodesicLine, s: f64) -> (f64, f64, f64) {
    line.position(Length::from_metres(s))
        .map_or((f64::NAN, f64::NAN, f64::NAN), |p| {
            (
                p.point.latitude.radians(),
                p.point.longitude.radians(),
                p.azimuth.radians(),
            )
        })
}

/// One edge: the shortest geodesic from vertex `a` to vertex `b`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Edge {
    /// The geodesic, from `a`.
    pub(crate) line: GeodesicLine,
    /// Its length, m.
    pub(crate) length: f64,
    lat_a: f64,
    lon_a: f64,
    lat_b: f64,
    lon_b: f64,
    /// The forward azimuths at `a` and at `b`, rad.
    az_a: f64,
    az_b: f64,
    /// The longitude it sweeps from `a` to `b`, signed, `|sweep| ≤ π`; zero on a meridian.
    sweep: f64,
    /// Its range of latitude, widened by [`LAT_PAD`].
    lat_lo: f64,
    lat_hi: f64,
}

impl Edge {
    /// The edge from `(lat_a, lon_a)` to `(lat_b, lon_b)`, longitudes already reduced, or
    /// [`ZoneError::Edge`] with `index` when its geodesic is not unique.
    #[expect(
        clippy::float_cmp,
        reason = "a reduced difference of exactly pi is the case the shortest geodesic's own side \
                  of the pole decides"
    )]
    fn new(index: usize, from: (f64, f64), to: (f64, f64)) -> Result<Self, ZoneError> {
        let point = |(lat, lon): (f64, f64)| SurfacePoint {
            latitude: Angle::from_radians(lat),
            longitude: Angle::from_radians(lon),
        };
        let (ua, ub) = (auxiliary(from.0, from.1), auxiliary(to.0, to.1));
        let cos_sep = ua[0] * ub[0] + ua[1] * ub[1] + ua[2] * ub[2];
        if cos_sep <= -libm::cos(ANTIPODAL_MARGIN_RAD) {
            return Err(ZoneError::Edge { index });
        }
        let inverse =
            geodesic_inverse(point(from), point(to)).map_err(|_| ZoneError::Edge { index })?;
        let length = inverse.distance.metres();
        if length <= 0.0 {
            return Err(ZoneError::Edge { index });
        }
        let az_a = inverse.azimuth1.radians();
        let az_b = inverse.azimuth2.radians();
        let line = GeodesicLine::new(point(from), inverse.azimuth1)
            .map_err(|_| ZoneError::Edge { index })?;
        let dlon = reduce(to.1 - from.1);
        // The shortest geodesic sweeps the reduced difference. At exactly π the side it passes
        // the pole on is the sign of sin α, which Clairaut keeps along it.
        let sweep = if dlon.abs() == PI {
            if libm::sin(az_a) < 0.0 { -PI } else { PI }
        } else {
            dlon
        };
        let (mut lo, mut hi) = (from.0.min(to.0), from.0.max(to.0));
        // The geodesic's vertex, its highest or lowest latitude, where cos α changes sign.
        let (cos_az_a, cos_az_b) = (libm::cos(az_a), libm::cos(az_b));
        if (cos_az_a > 0.0 && cos_az_b < 0.0) || (cos_az_a < 0.0 && cos_az_b > 0.0) {
            let (sin_lat, cos_lat) = libm::sincos(from.0);
            let hyp = libm::hypot(F1 * sin_lat, cos_lat);
            let (sin_beta, cos_beta) = (F1 * sin_lat / hyp, cos_lat / hyp);
            let sin_az = libm::sin(az_a);
            // Clairaut on the auxiliary sphere: cos β_vertex = |sin α₀| = |sin α₁ cos β₁|.
            let sin_alpha0 = sin_az * cos_beta;
            let cos_alpha0 = libm::hypot(cos_az_a, sin_az * sin_beta);
            let top = libm::atan2(cos_alpha0, F1 * sin_alpha0.abs());
            if cos_az_a > 0.0 {
                hi = hi.max(top);
            } else {
                lo = lo.min(-top);
            }
        }
        Ok(Self {
            line,
            length,
            lat_a: from.0,
            lon_a: from.1,
            lat_b: to.0,
            lon_b: to.1,
            az_a,
            az_b,
            sweep,
            lat_lo: lo - LAT_PAD,
            lat_hi: hi + LAT_PAD,
        })
    }

    /// Whether the edge lies on one meridian.
    fn meridian(&self) -> bool {
        self.sweep == 0.0
    }

    /// The longitude of `lon` relative to `a`, as an offset along the sweep, if `lon` is in the
    /// edge's closed range of longitude.
    fn offset(&self, lon: f64) -> Option<f64> {
        if self.meridian() {
            return None;
        }
        let mut o = reduce(lon - self.lon_a);
        if self.sweep > 0.0 && o < 0.0 {
            o += TAU;
        } else if self.sweep < 0.0 && o > 0.0 {
            o -= TAU;
        }
        (o.abs() <= self.sweep.abs()).then_some(o)
    }

    /// The point where the edge crosses longitude `lon`, which is in its closed range, as latitude,
    /// longitude and forward azimuth: a vertex's own at its longitude, else the line's at Brent's
    /// root of the longitude along it.
    #[expect(
        clippy::float_cmp,
        reason = "a vertex's longitude is copied, not computed: equality is identity"
    )]
    fn crossing(&self, lon: f64) -> Option<(f64, f64, f64)> {
        if lon == self.lon_a {
            return Some((self.lat_a, self.lon_a, self.az_a));
        }
        if lon == self.lon_b {
            return Some((self.lat_b, self.lon_b, self.az_b));
        }
        let target = self.offset(lon)?;
        let dir = self.sweep.signum();
        let along = |s: f64| {
            let mut o = reduce(point_at(&self.line, s).1 - self.lon_a);
            // A wrap shows as a jump of 2π; the true offset lies between 0 and the sweep.
            if dir > 0.0 && o < -FRAC_PI_2 {
                o += TAU;
            } else if dir < 0.0 && o > FRAC_PI_2 {
                o -= TAU;
            }
            dir * (o - target)
        };
        let s = match brent(along, 0.0, self.length, CROSSING_TOL_M) {
            Ok(bracket) => bracket.estimate(),
            // Within rounding of an end: the end it is nearer to.
            Err(_) => {
                if target.abs() < 0.5 * self.sweep.abs() {
                    0.0
                } else {
                    self.length
                }
            }
        };
        Some(point_at(&self.line, s))
    }

    /// The crossing of the meridian `lon` that counts for a point on it: the edge passes from
    /// one side of the meridian to the other, a vertex on it counting as lying an infinitesimal
    /// step west. The latitude of the crossing, or `None`.
    fn counted_crossing(&self, lon: f64) -> Option<f64> {
        if self.meridian() {
            return None;
        }
        let x_a = reduce(self.lon_a - lon);
        let mut x_b = reduce(self.lon_b - lon);
        // x_b continues x_a along the sweep; reduced, it may have wrapped by 2π.
        let ahead = x_a + self.sweep;
        if x_b - ahead > PI {
            x_b -= TAU;
        } else if ahead - x_b > PI {
            x_b += TAU;
        }
        if (x_a <= 0.0) == (x_b <= 0.0) {
            return None;
        }
        self.crossing(lon).map(|(lat, _, _)| lat)
    }

    /// Whether the point `(lat, lon)`, Earth-fixed `p`, lies within [`ON_EDGE_M`] of the edge: of
    /// a vertex, or across the edge where it crosses the point's meridian (where it passes the
    /// point's latitude, for an edge on a meridian).
    fn holds(&self, p: [f64; 3], lat: f64, lon: f64) -> bool {
        if lat < self.lat_lo || lat > self.lat_hi {
            return false;
        }
        if distance(p, surface(self.lat_a, self.lon_a)) <= ON_EDGE_M
            || distance(p, surface(self.lat_b, self.lon_b)) <= ON_EDGE_M
        {
            return true;
        }
        if self.meridian() {
            let (lo, hi) = (self.lat_a.min(self.lat_b), self.lat_a.max(self.lat_b));
            // The meridian's plane holds the opposite half-meridian too, half a turn away: a point
            // there is as near the plane as one on the edge, and is not on it.
            return (lo..=hi).contains(&lat)
                && reduce(lon - self.lon_a).abs() < FRAC_PI_2
                && across(p, lat, self.lon_a, self.az_a) <= ON_EDGE_M;
        }
        self.crossing(lon)
            .is_some_and(|(lat_c, lon_c, az)| across(p, lat_c, lon_c, az) <= ON_EDGE_M)
    }
}

/// Where two edges meet, as far as a self-intersection test needs it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Meet {
    /// At this latitude and longitude.
    At(f64, f64),
    /// At a point strictly inside both, or along a stretch.
    Elsewhere,
}

/// The meetings of two edges: every point at which their latitudes agree at an end of a common
/// range of longitude, and [`Meet::Elsewhere`] for a change of sign of their difference across
/// one, or for a meridian edge overlapping another's stretch.
#[expect(
    clippy::float_cmp,
    reason = "the ends of a common range are vertex longitudes, copied, and two edges meet there \
              exactly when their latitudes are the same double"
)]
fn meetings(e: &Edge, f: &Edge) -> Vec<Meet> {
    let mut out = Vec::new();
    if e.lat_hi < f.lat_lo || f.lat_hi < e.lat_lo {
        return out;
    }
    match (e.meridian(), f.meridian()) {
        (true, true) => {
            if e.lon_a == f.lon_a {
                let lo = e.lat_a.min(e.lat_b).max(f.lat_a.min(f.lat_b));
                let hi = e.lat_a.max(e.lat_b).min(f.lat_a.max(f.lat_b));
                if lo == hi {
                    out.push(Meet::At(lo, e.lon_a));
                } else if lo < hi {
                    out.push(Meet::Elsewhere);
                }
            }
        }
        (true, false) | (false, true) => {
            let (m, g) = if e.meridian() { (e, f) } else { (f, e) };
            if let Some((lat, _, _)) = g.offset(m.lon_a).and(g.crossing(m.lon_a)) {
                let (lo, hi) = (m.lat_a.min(m.lat_b), m.lat_a.max(m.lat_b));
                if (lo..=hi).contains(&lat) {
                    out.push(Meet::At(lat, m.lon_a));
                }
            }
        }
        (false, false) => {
            for (start, end) in common_ranges(e, f) {
                let d = |lon: f64| -> Option<(f64, f64)> {
                    Some((e.crossing(lon)?.0, f.crossing(lon)?.0))
                };
                let (Some((e0, f0)), Some((e1, f1))) = (d(start), d(end)) else {
                    continue;
                };
                let (d0, d1) = (e0 - f0, e1 - f1);
                if d0 == 0.0 {
                    out.push(Meet::At(e0, start));
                }
                if d1 == 0.0 && end != start {
                    out.push(Meet::At(e1, end));
                }
                if d0 != 0.0 && d1 != 0.0 && (d0 < 0.0) != (d1 < 0.0) {
                    out.push(Meet::Elsewhere);
                }
            }
        }
    }
    out
}

/// The ranges of longitude two non-meridian edges share, each as its two ends, which are always
/// longitudes of their vertices.
fn common_ranges(e: &Edge, f: &Edge) -> Vec<(f64, f64)> {
    // Each edge as the closed range [west, east] it covers going east from `west`.
    let span = |g: &Edge| {
        if g.sweep > 0.0 {
            (g.lon_a, g.lon_b, g.sweep)
        } else {
            (g.lon_b, g.lon_a, -g.sweep)
        }
    };
    let ((w1, e1, s1), (w2, e2, s2)) = (span(e), span(f));
    let east = |o: f64| {
        let mut o = reduce(o);
        if o < 0.0 {
            o += TAU;
        }
        o
    };
    let o = east(w2 - w1);
    let mut out = Vec::new();
    // The second range starts inside the first.
    if o <= s1 {
        out.push((w2, if s1 <= o + s2 { e1 } else { e2 }));
    }
    // The second range wraps round to cover the first's start.
    if o + s2 >= TAU && o > 0.0 {
        out.push((w1, if s1 <= o + s2 - TAU { e1 } else { e2 }));
    }
    out
}

/// A zone on WGS 84: a closed region bounded by geodesics. The module documentation states the
/// model.
#[derive(Debug, Clone)]
pub struct Zone {
    /// The vertices, `(latitude, reduced longitude)`, rad.
    vertices: Vec<(f64, f64)>,
    edges: Vec<Edge>,
    /// Whether the north pole's side of every meridian is inside.
    north_inside: bool,
}

impl Zone {
    /// The zone bounded by the shortest geodesics through `vertices`, counter-clockwise seen from
    /// outside the ellipsoid; the last vertex joins the first.
    ///
    /// # Errors
    ///
    /// [`ZoneError::TooFewVertices`], [`ZoneError::Vertex`], [`ZoneError::Edge`] or
    /// [`ZoneError::SelfIntersecting`], as the module documentation states.
    pub fn new(vertices: &[SurfacePoint]) -> Result<Self, ZoneError> {
        if vertices.len() < 3 {
            return Err(ZoneError::TooFewVertices);
        }
        let mut points = Vec::with_capacity(vertices.len());
        for (index, v) in vertices.iter().enumerate() {
            let (lat, lon) = (v.latitude.radians(), v.longitude.radians());
            if !valid(lat, lon) {
                return Err(ZoneError::Vertex { index });
            }
            points.push((lat, reduce(lon)));
        }
        let n = points.len();
        let edges = (0..n)
            .map(|i| Edge::new(i, points[i], points[(i + 1) % n]))
            .collect::<Result<Vec<_>, _>>()?;
        for i in 0..n {
            for j in i + 1..n {
                // Consecutive edges share one vertex, the only point they may meet at.
                let shared = if j == i + 1 {
                    Some(points[j])
                } else if i == 0 && j == n - 1 {
                    Some(points[0])
                } else {
                    None
                };
                let allowed = |m: &Meet| shared.is_some_and(|(lat, lon)| *m == Meet::At(lat, lon));
                if meetings(&edges[i], &edges[j]).iter().any(|m| !allowed(m)) {
                    return Err(ZoneError::SelfIntersecting {
                        first: i,
                        second: j,
                    });
                }
            }
        }
        let north_inside = north_inside(&points, &edges);
        Ok(Self {
            vertices: points,
            edges,
            north_inside,
        })
    }

    /// The vertices, `(latitude, longitude)` with the longitude reduced to `(−π, π]`, rad.
    #[must_use]
    pub fn vertices(&self) -> &[(f64, f64)] {
        &self.vertices
    }

    /// Whether `point` lies in the zone. A vertex, and a point within [`ON_EDGE_M`] of an edge
    /// across it, do: the zone is closed.
    ///
    /// # Errors
    ///
    /// [`ZoneError::Point`] if a coordinate is not finite or the latitude is outside
    /// `[−π/2, π/2]`.
    pub fn contains(&self, point: SurfacePoint) -> Result<bool, ZoneError> {
        let (lat, lon) = (point.latitude.radians(), point.longitude.radians());
        if !valid(lat, lon) {
            return Err(ZoneError::Point);
        }
        Ok(self.holds(lat, reduce(lon)))
    }

    /// [`Zone::contains`] for a valid point with its longitude reduced.
    pub(crate) fn holds(&self, lat: f64, lon: f64) -> bool {
        let p = surface(lat, lon);
        if self.vertices.contains(&(lat, lon)) || self.edges.iter().any(|e| e.holds(p, lat, lon)) {
            return true;
        }
        let above = self
            .edges
            .iter()
            .filter_map(|e| e.counted_crossing(lon))
            .filter(|&c| c > lat)
            .count();
        self.north_inside != (above % 2 == 1)
    }
}

// ── The elevation of a satellite over a zone ──────────────────────────────────────────────────

/// A bound on `|n''|`, the second derivative of the outward normal along a unit-speed geodesic of
/// WGS 84, m⁻²: its components' bounds give 2.4917e-14 (`derivations/access_zone_elevation.mac`,
/// claims 28–30).
pub(crate) const NORMAL_SECOND_DERIVATIVE: f64 = 2.5e-14;

/// The largest normal curvature of WGS 84, `a/b²`, 1/m: the meridian's at the equator (claim 31).
pub(crate) const KAPPA_MAX: f64 = 1.0 / RHO_MIN;

/// A bound on `|u''|` for the squared chord `u = |γ(s) − q|²` from a point `q` of the ellipsoid to
/// a unit-speed geodesic `γ`: `u'' = 2 − 2κ (γ − q)·n` with `|γ − q| ≤ 2a`, so `2 + 4a²/b²`.
pub(crate) const CHORD_CURVATURE: f64 = 2.0 + 4.0 * EQUATORIAL_RADIUS * KAPPA_MAX;

/// The line of sight `d = S − p` from the ellipsoid's point `(lat, lon)` to `sat`, and the outward
/// normal there, which is the zenith every elevation here is measured from.
pub(crate) fn sight(sat: [f64; 3], lat: f64, lon: f64) -> ([f64; 3], [f64; 3]) {
    let p = surface(lat, lon);
    let (s, c) = libm::sincos(lat);
    let (sl, cl) = libm::sincos(lon);
    (
        [sat[0] - p[0], sat[1] - p[1], sat[2] - p[2]],
        [c * cl, c * sl, s],
    )
}

/// The far foot of the normal through `sat`, `(lat, lon)`: the point of the ellipsoid other than
/// the nadir whose normal line passes through `sat`, from which `sat` stands at −90°. The feet are
/// `p = A S / (A + t)` (`derivations/access_zone_elevation.mac`, claim 5); with `x = −(a² + t)` the
/// far one solves `a²r²/x² + b²z²/(x + c)² = 1`, `c = a² − b²`, whose left side falls through 1 on
/// `[√(a²r² + b²z²) − c, √(a²r² + b²z²)]`, and lies at the longitude opposite the satellite's and
/// the latitude `atan2(−z x, r (x + c))`.
pub(crate) fn far_foot(sat: [f64; 3]) -> (f64, f64) {
    let gap = EQUATORIAL_RADIUS * EQUATORIAL_RADIUS - POLAR_RADIUS * POLAR_RADIUS;
    let r = libm::hypot(sat[0], sat[1]);
    let z = sat[2];
    let (ar, bz) = (EQUATORIAL_RADIUS * r, POLAR_RADIUS * z);
    let hi = libm::hypot(ar, bz);
    let lo = hi - gap;
    let residual = |x: f64| {
        let (across, along) = (ar / x, bz / (x + gap));
        across * across + along * along - 1.0
    };
    // Over a pole or the equator the root is an end of the bracket, where the residual's rounding
    // may keep it from changing sign; the latitude there does not depend on x.
    let x = brent(residual, lo, hi, 4.0 * f64::EPSILON * hi).map_or_else(
        |_| {
            if residual(lo).abs() < residual(hi).abs() {
                lo
            } else {
                hi
            }
        },
        |bracket| bracket.estimate(),
    );
    (
        libm::atan2(-z * x, r * (x + gap)),
        reduce(libm::atan2(sat[1], sat[0]) + PI),
    )
}

/// One stretch of an edge a boundary search has not ruled out: its ends along the edge, m, the
/// values and the satellite's distances there, and a bound on the value anywhere on it.
#[derive(Debug, Clone, Copy)]
struct Stretch {
    edge: usize,
    s0: f64,
    s1: f64,
    f0: (f64, f64),
    f1: (f64, f64),
    bound: f64,
}

impl Stretch {
    /// The stretch from `s0` to `s1` of `edge`, with its bound: along a unit-speed geodesic with
    /// `|f''| ≤ K`, `f` lies under its chord plus `K (s − s0)(s1 − s)/2`, so under
    /// `max(f₀, f₁) + K L²/8`. Every point of the stretch is within `L` of each end, so its
    /// distance from the satellite is at least the larger end's less `L`, which `curvature` takes.
    fn new(
        edge: usize,
        (s0, s1): (f64, f64),
        f0: (f64, f64),
        f1: (f64, f64),
        curvature: &impl Fn(f64) -> f64,
    ) -> Self {
        let length = s1 - s0;
        let k = curvature(f0.1.max(f1.1) - length);
        Self {
            edge,
            s0,
            s1,
            f0,
            f1,
            bound: f0.0.max(f1.0) + k * length * length / 8.0,
        }
    }
}

impl PartialEq for Stretch {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == core::cmp::Ordering::Equal
    }
}

impl Eq for Stretch {}

impl PartialOrd for Stretch {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Stretch {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.bound.total_cmp(&other.bound)
    }
}

/// What a search of a zone's boundary found: the largest value it reached, the point that reached
/// it, and a bound no point of the boundary exceeds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Extreme {
    pub(crate) best: f64,
    pub(crate) at: (f64, f64),
    pub(crate) bound: f64,
}

impl Zone {
    /// The largest value of `f` over the boundary, by branch and bound. `f(lat, lon)` returns the
    /// value and the satellite's distance from the point, and `curvature(d)` bounds `|f''|` along
    /// an edge, at unit speed, wherever that distance is at least `d`. The stretch with the
    /// largest bound is halved first; the search stops once `done(best, bound)`, or when no
    /// stretch can be halved in floating point.
    pub(crate) fn boundary_max(
        &self,
        f: impl Fn(f64, f64) -> (f64, f64),
        curvature: impl Fn(f64) -> f64,
        done: impl Fn(f64, f64) -> bool,
    ) -> Extreme {
        let mut found = Extreme {
            best: f64::NEG_INFINITY,
            at: (f64::NAN, f64::NAN),
            bound: f64::NEG_INFINITY,
        };
        let reach = |found: &mut Extreme, value: f64, at: (f64, f64)| {
            if value > found.best {
                found.best = value;
                found.at = at;
            }
        };
        let mut open = BinaryHeap::with_capacity(4 * self.edges.len());
        for (edge, e) in self.edges.iter().enumerate() {
            let (f0, f1) = (f(e.lat_a, e.lon_a), f(e.lat_b, e.lon_b));
            reach(&mut found, f0.0, (e.lat_a, e.lon_a));
            reach(&mut found, f1.0, (e.lat_b, e.lon_b));
            open.push(Stretch::new(edge, (0.0, e.length), f0, f1, &curvature));
        }
        // The largest bound of a stretch set aside: ruled out, or too short to halve.
        let mut aside = f64::NEG_INFINITY;
        loop {
            let top = open.peek().map_or(f64::NEG_INFINITY, |s| s.bound);
            let bound = top.max(aside).max(found.best);
            let Some(s) = open.pop().filter(|_| !done(found.best, bound)) else {
                found.bound = bound;
                return found;
            };
            let mid = 0.5 * (s.s0 + s.s1);
            if !(s.s0 < mid && mid < s.s1) {
                aside = aside.max(s.bound);
                continue;
            }
            let (lat, lon, _) = point_at(&self.edges[s.edge].line, mid);
            let fm = f(lat, lon);
            reach(&mut found, fm.0, (lat, lon));
            for half in [
                Stretch::new(s.edge, (s.s0, mid), s.f0, fm, &curvature),
                Stretch::new(s.edge, (mid, s.s1), fm, s.f1, &curvature),
            ] {
                if half.bound > found.best {
                    open.push(half);
                } else {
                    aside = aside.max(half.bound);
                }
            }
        }
    }
}

/// Whether the north pole's side is inside: the direction of the highest crossing of a meridian
/// through no vertex, which an edge with a sweep crosses at mid-sweep.
#[expect(
    clippy::float_cmp,
    reason = "a meridian through a vertex is one whose longitude is the vertex's, to the bit"
)]
fn north_inside(points: &[(f64, f64)], edges: &[Edge]) -> bool {
    let mut widest: Vec<&Edge> = edges.iter().filter(|e| !e.meridian()).collect();
    widest.sort_by(|a, b| b.sweep.abs().total_cmp(&a.sweep.abs()));
    for e in widest {
        for frac in [0.5, 0.25, 0.75, 0.375, 0.625] {
            let lon = reduce(e.lon_a + frac * e.sweep);
            if points.iter().any(|&(_, l)| l == lon) {
                continue;
            }
            // No vertex lies on this meridian, so every crossing is strictly inside one edge.
            let top = edges
                .iter()
                .filter_map(|g| g.counted_crossing(lon).map(|lat| (lat, g.sweep)))
                .max_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((_, sweep)) = top {
                // Heading east, the interior is on the left: north of the crossing.
                return sweep > 0.0;
            }
        }
    }
    // Unreachable for a zone that passed the intersection test: three or more edges that do not
    // overlap cannot all lie on meridians.
    false
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::unwrap_used,
        reason = "a test fails loudly on a case that does not hold"
    )]

    use super::*;

    fn deg(lat: f64, lon: f64) -> SurfacePoint {
        SurfacePoint {
            latitude: Angle::from_radians(lat.to_radians()),
            longitude: Angle::from_radians(lon.to_radians()),
        }
    }

    #[test]
    fn reduce_keeps_the_range_and_wraps_the_rest() {
        assert_eq!(reduce(PI).to_bits(), PI.to_bits());
        assert_eq!(reduce(-PI).to_bits(), PI.to_bits());
        assert_eq!(reduce(0.3).to_bits(), 0.3_f64.to_bits());
        assert!((reduce(3.0 * PI) - PI).abs() < 1e-15);
        assert!((reduce(-4.0) - (-4.0 + TAU)).abs() < 1e-15);
    }

    #[test]
    fn a_small_square_holds_its_centre_and_not_its_antipode() {
        let z = Zone::new(&[deg(0.0, 0.0), deg(0.0, 1.0), deg(1.0, 1.0), deg(1.0, 0.0)]).unwrap();
        assert!(z.contains(deg(0.5, 0.5)).unwrap());
        assert!(!z.contains(deg(-0.5, -179.5)).unwrap());
        assert!(!z.contains(deg(89.0, 0.5)).unwrap());
        let r = Zone::new(&[deg(1.0, 0.0), deg(1.0, 1.0), deg(0.0, 1.0), deg(0.0, 0.0)]).unwrap();
        assert!(!r.contains(deg(0.5, 0.5)).unwrap());
        assert!(r.contains(deg(-0.5, -179.5)).unwrap());
    }

    /// `(lat1, lon1, lat2, lon2, lon, lat)`, rad: the shortest geodesic from the first point to
    /// the second crosses longitude `lon` at latitude `lat`, from the defining integrals at 50
    /// digits by `derivations/zone_meridian_crossing.gp`; `derivations/verify.sh` regenerates the
    /// rows and compares them byte for byte.
    #[rustfmt::skip]
    #[expect(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::approx_constant,
        reason = "the rows are pasted as the derivation prints them; some angles are pi/4 or pi/2 \
                  to the last digit"
    )]
    const CROSSINGS: &[[f64; 6]] = &[
// BEGIN zone_meridian_crossing.gp
    [0.17453292519943295, 0.34906585039886590, 0.52359877559829893, 0.87266462599716477, 0.61086523819801530, 0.3720737296506536729633990],
    [0.69813170079773179, 2.9670597283903604, 0.87266462599716477, -2.7925268031909272, -3.0543261909900767, 0.8104455513559492414964552],
    [-0.78539816339744828, 0.17453292519943295, 1.0471975511965979, 0.69813170079773179, 0.47123889803846897, 0.5133933474937525408336715],
    [0.95993108859688125, -0.52359877559829893, 0.34906585039886590, -1.2217304763960306, -0.71558499331767511, 0.8700957262096392810382580],
    [0.0087266462599716477, 0.0, -0.0087266462599716477, 0.052359877559829890, 0.021816615649929118, 0.001454639692904370042033971],
    [-0.52359877559829893, 1.7453292519943295, 0.52359877559829893, 1.8151424220741028, 1.7627825445142729, -0.2811352222769722545323357],
    [1.3962634015954636, 0.0, 1.3962634015954636, 2.9670597283903604, 1.5707963267948966, 1.555372525155595968223976],
    [-1.2217304763960306, -2.0943951023931957, -1.3962634015954636, 2.6179938779914944, -2.8797932657906440, -1.404385259593438103313622],
    [0.78539816339744828, -0.17453292519943295, 0.78539816339744828, 0.17453292519943295, 0.052359877559829890, 0.7923900319771376433667067],
    [-0.34906585039886590, -1.0471975511965979, 0.87266462599716477, 0.52359877559829893, -0.17453292519943295, 0.5976361900092164787452104],
// END zone_meridian_crossing.gp
    ];

    /// G2: every reference edge crosses its meridian where the defining integrals put it, to
    /// `1e-15` rad (6.4 nm): the geodesic code's own agreement with them (`tests/geodesic.rs` in
    /// spacelab-earth, worst 3.7 nm) with Brent's 1 nm along the edge on top.
    #[test]
    fn edges_cross_meridians_where_the_defining_integrals_put_them() {
        let mut worst = 0.0f64;
        for r in CROSSINGS {
            let e = Edge::new(0, (r[0], r[1]), (r[2], r[3])).unwrap();
            let (lat, lon, _) = e.crossing(r[4]).unwrap();
            assert!((reduce(lon - r[4])).abs() < 1e-15, "{r:?}: lon {lon}");
            worst = worst.max((lat - r[5]).abs());
        }
        eprintln!(
            "crossings vs 50 digits, {} rows: worst {worst:.2e} rad",
            CROSSINGS.len()
        );
        assert!(worst <= 1e-15, "{worst:e}");
    }

    /// The satellite at geodetic `(lat, lon)` in degrees, `h` metres up, Earth-fixed.
    fn above(lat: f64, lon: f64, h: f64) -> [f64; 3] {
        let (lat, lon) = (lat.to_radians(), lon.to_radians());
        let p = surface(lat, lon);
        let (s, c) = libm::sincos(lat);
        let (sl, cl) = libm::sincos(lon);
        [p[0] + h * c * cl, p[1] + h * c * sl, p[2] + h * s]
    }

    fn sine(sat: [f64; 3], lat: f64, lon: f64) -> (f64, f64) {
        let (d, n) = sight(sat, lat, lon);
        let range = libm::hypot(libm::hypot(d[0], d[1]), d[2]);
        ((d[0] * n[0] + d[1] * n[1] + d[2] * n[2]) / range, range)
    }

    /// From the far foot the satellite stands at −90°: the line of sight is the inward normal
    /// there, to rounding, over the poles, the equator and between, from 200 km to 40 000 km up.
    #[test]
    fn the_far_foot_sees_the_satellite_at_minus_90_degrees() {
        let mut worst = 0.0_f64;
        for &(lat, lon, h) in &[
            (0.0, 0.0, 400e3),
            (90.0, 0.0, 700e3),
            (-90.0, 30.0, 700e3),
            (45.0, 100.0, 200e3),
            (-33.0, -70.0, 20_200e3),
            (61.0, 179.0, 40_000e3),
            (1e-9, -45.0, 35_786e3),
        ] {
            let (flat, flon) = far_foot(above(lat, lon, h));
            worst = worst.max(sine(above(lat, lon, h), flat, flon).0 + 1.0);
        }
        eprintln!("1 + sin ε at the far foot, worst {worst:.2e}");
        assert!(worst < 1e-15, "{worst:e}");
    }

    /// The boundary search brackets the extreme a dense sample of the boundary finds: every sample
    /// lies under its bound, the search reaches the sample's best to its tolerance, and the point
    /// it names has the value it reports, for satellites over the zone, beside it and far from it.
    #[test]
    fn the_boundary_search_brackets_a_dense_sample_of_the_boundary() {
        let zones = [
            Zone::new(&[
                deg(30.0, 30.0),
                deg(30.0, 36.0),
                deg(36.0, 36.0),
                deg(36.0, 30.0),
            ])
            .unwrap(),
            Zone::new(&[
                deg(75.0, 0.0),
                deg(75.0, 90.0),
                deg(75.0, 180.0),
                deg(75.0, -90.0),
            ])
            .unwrap(),
        ];
        let satellites = [
            (above(33.0, 33.0, 500e3), 500e3),
            (above(37.0, 37.0, 700e3), 700e3),
            (above(-10.0, 60.0, 800e3), 800e3),
            (above(70.0, 45.0, 400e3), 400e3),
            (above(0.0, -100.0, 35_786e3), 35_786e3),
        ];
        let mut cases = 0;
        for zone in &zones {
            for &(sat, h) in &satellites {
                // Every point of the ellipsoid is at least the height away.
                let curvature = |d: f64| {
                    let d = d.max(h);
                    NORMAL_SECOND_DERIVATIVE + 4.0 * KAPPA_MAX / d + 3.0 / (d * d)
                };
                let e = zone.boundary_max(
                    |lat, lon| sine(sat, lat, lon),
                    curvature,
                    |best, bound| bound - best <= 1e-12,
                );
                let mut dense = f64::NEG_INFINITY;
                for edge in &zone.edges {
                    for k in 0..=4000 {
                        let s = edge.length * f64::from(k) / 4000.0;
                        let (lat, lon, _) = point_at(&edge.line, s);
                        let v = sine(sat, lat, lon).0;
                        assert!(v <= e.bound + 1e-15, "{v} over the bound {}", e.bound);
                        dense = dense.max(v);
                    }
                }
                assert!(e.best >= dense - 2e-12, "{e:?} against {dense}");
                assert_eq!(sine(sat, e.at.0, e.at.1).0.to_bits(), e.best.to_bits());
                assert!(e.bound - e.best <= 1e-12, "{e:?}");
                cases += 1;
            }
        }
        assert_eq!(cases, 10);
    }
}
