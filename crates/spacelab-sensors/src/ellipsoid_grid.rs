//! A cell grid that stores positions only, and derives the zenith.
//!
//! [`crate::CellGrid`] stores six `f64` streams — position and zenith — which is 48 bytes per
//! cell. Measurement said that is the wrong number: at 0.47 Gcell/s the elevation kernel moves
//! 22.5 GB/s, close to this machine's DRAM ceiling, while the three-stream cone kernel runs
//! 2.7× faster per cell. The loop is starved, not slow. Wider vectors buy nothing.
//!
//! The zenith does not need storing. On the ellipsoid `x²/a² + y²/a² + z²/b² = 1` the outward
//! normal is `grad f ∝ (x/a², y/a², z/b²) ∝ (x, y, γz)` with `γ = a²/b²`
//! (`derivations/ellipsoid_normal.mac`). One multiply replaces three streams.
//!
//! The normal is not a unit vector, so the predicate carries its squared length:
//!
//! ```text
//! t = (d · n) / m,   m² = x² + y² + γ²z²
//! t >= s·sqrt(q)   <=>   (d·n) >= 0  &&  (d·n)² >= s²·m²·q
//! ```
//!
//! Still no square root, no division and no trigonometry, and now 24 bytes per cell.

use spacelab_core::Vec3;

use crate::{ElevationMask, SensorError};

/// `a²/b²` for WGS-84, the factor that turns a surface position into a zenith direction.
///
/// From the two defining constants, `a = 6378137.0` m and `1/f = 298.257223563`:
///
/// ```text
/// a²/b² = 1/(1-f)² = 1/(1-e²) = 1.006739496742276434954782158956759376656…
/// ```
///
/// Computed to 40 digits in PARI/GP two independent ways — from `f` and from `e²` — and rounded
/// once. `tests/ellipsoid_grid.rs` recomputes it from `f` and asserts the bits match, so it
/// cannot drift.
pub const WGS84_GAMMA: f64 = 1.006_739_496_742_276_5;

/// Cell centres on a biaxial ellipsoid, three streams, zenith derived.
///
/// # Invariant
///
/// The three slices have equal length, checked once in [`EllipsoidGrid::new`]. `gamma` is
/// `a²/b²` of the ellipsoid the cells lie on; using the wrong one tilts every zenith.
///
/// # Not bit-identical to [`crate::CellGrid`], and how far that was checked
///
/// A stored, normalised zenith rounds once at build time; a derived one rounds differently. Both
/// are one-ulp perturbations of the same exact quantity, so the *decision* can only differ where
/// `|g|` is of the order of one ulp of the dot products — about `1e-8` m, which is `5e-16` rad of
/// elevation and below what an `f64` elevation resolves at all.
///
/// `tests/ellipsoid_grid.rs` observes **zero disagreements in 983 040 decisions** across three
/// masks and forty satellite positions. That is worth exactly what it is worth: random geometry
/// never lands within an ulp of a boundary, so the test confirms the bulk and says nothing about
/// the boundary. The boundary claim above rests on the error argument, not on that count.
///
/// The derived zenith is checked directly against the exact ellipsoid normal over `10^5` random
/// surface points: agreement to better than `1e-14` rad, which moves a footprint edge by well
/// under a micrometre.
#[derive(Debug, Clone, Copy)]
pub struct EllipsoidGrid<'a> {
    px: &'a [f64],
    py: &'a [f64],
    pz: &'a [f64],
    gamma: f64,
}

impl<'a> EllipsoidGrid<'a> {
    /// Build from three equal-length position streams and the ellipsoid's `a²/b²`.
    ///
    /// # Errors
    ///
    /// [`SensorError::RaggedGrid`] if the slices differ in length.
    pub fn new(
        px: &'a [f64],
        py: &'a [f64],
        pz: &'a [f64],
        gamma: f64,
    ) -> Result<Self, SensorError> {
        let lens = [px.len(), py.len(), pz.len()];
        let shortest = lens.iter().copied().min().unwrap_or(0);
        let longest = lens.iter().copied().max().unwrap_or(0);
        if shortest != longest {
            return Err(SensorError::RaggedGrid { shortest, longest });
        }
        Ok(Self { px, py, pz, gamma })
    }

    /// Number of cells.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.px.len()
    }

    /// Whether the grid holds no cells.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.px.is_empty()
    }

    /// The three position streams.
    #[must_use]
    pub const fn columns(&self) -> (&[f64], &[f64], &[f64]) {
        (self.px, self.py, self.pz)
    }

    /// The stored `a²/b²`.
    #[must_use]
    pub const fn gamma(&self) -> f64 {
        self.gamma
    }

    /// The unit zenith of cell `i`, for callers that need it. Costs a square root.
    ///
    /// # Panics
    ///
    /// If `i` is out of bounds.
    #[must_use]
    pub fn zenith(&self, i: usize) -> Vec3 {
        let n = Vec3::new(self.px[i], self.py[i], self.gamma * self.pz[i]);
        n.scale(1.0 / n.norm())
    }
}

/// Coverage over an [`EllipsoidGrid`], writing `1` or `0` per cell.
///
/// Three streams in, one byte out. This is the kernel the engine should use.
///
/// # Errors
///
/// [`SensorError::OutputLengthMismatch`] if `out` is not exactly one byte per cell.
#[allow(clippy::needless_range_loop)]
pub fn elevation_covered_into(
    mask: ElevationMask,
    satellite: Vec3,
    cells: &EllipsoidGrid<'_>,
    out: &mut [u8],
) -> Result<(), SensorError> {
    let n = cells.len();
    if n != out.len() {
        return Err(SensorError::OutputLengthMismatch {
            cells: n,
            buffer: out.len(),
        });
    }
    let (px, py, pz) = cells.columns();
    let (px, py, pz) = (&px[..n], &py[..n], &pz[..n]);
    let out = &mut out[..n];

    let g = cells.gamma();
    let s = mask.sin_min();
    let s2 = s * s;
    let (rx, ry, rz) = (satellite.x, satellite.y, satellite.z);

    if s >= 0.0 {
        for i in 0..n {
            let (cx, cy, cz) = (px[i], py[i], pz[i]);
            let nz = g * cz;
            let (dx, dy, dz) = (rx - cx, ry - cy, rz - cz);
            let dn = dx * cx + dy * cy + dz * nz;
            let m2 = cx * cx + cy * cy + nz * nz;
            let q = dx * dx + dy * dy + dz * dz;
            out[i] = u8::from(dn >= 0.0 && dn * dn >= s2 * m2 * q);
        }
    } else {
        for i in 0..n {
            let (cx, cy, cz) = (px[i], py[i], pz[i]);
            let nz = g * cz;
            let (dx, dy, dz) = (rx - cx, ry - cy, rz - cz);
            let dn = dx * cx + dy * cy + dz * nz;
            let m2 = cx * cx + cy * cy + nz * nz;
            let q = dx * dx + dy * dy + dz * dz;
            out[i] = u8::from(dn >= 0.0 || dn * dn <= s2 * m2 * q);
        }
    }
    Ok(())
}

/// Increment `counters[i]` for every covered cell. See [`elevation_covered_into`].
///
/// # Errors
///
/// [`SensorError::OutputLengthMismatch`] if `counters` is not exactly one per cell.
#[allow(clippy::needless_range_loop)]
pub fn elevation_accumulate_into(
    mask: ElevationMask,
    satellite: Vec3,
    cells: &EllipsoidGrid<'_>,
    counters: &mut [u32],
) -> Result<(), SensorError> {
    let n = cells.len();
    if n != counters.len() {
        return Err(SensorError::OutputLengthMismatch {
            cells: n,
            buffer: counters.len(),
        });
    }
    let (px, py, pz) = cells.columns();
    let (px, py, pz) = (&px[..n], &py[..n], &pz[..n]);
    let counters = &mut counters[..n];

    let g = cells.gamma();
    let s = mask.sin_min();
    let s2 = s * s;
    let below = s < 0.0;
    let (rx, ry, rz) = (satellite.x, satellite.y, satellite.z);

    for i in 0..n {
        let (cx, cy, cz) = (px[i], py[i], pz[i]);
        let nz = g * cz;
        let (dx, dy, dz) = (rx - cx, ry - cy, rz - cz);
        let dn = dx * cx + dy * cy + dz * nz;
        let m2 = cx * cx + cy * cy + nz * nz;
        let q = dx * dx + dy * dy + dz * dz;
        let up = dn >= 0.0;
        let sq = dn * dn >= s2 * m2 * q;
        counters[i] += u32::from(if below { up || !sq } else { up && sq });
    }
    Ok(())
}
