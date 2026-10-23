//! Structure-of-arrays view over the cells of a coverage grid.

use crate::SensorError;

/// The six component slices of a [`CellGrid`], in the order `px, py, pz, zx, zy, zz`.
pub type Columns<'a> = (
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
    &'a [f64],
);

/// Cell centres and zeniths, split by component so the hot loop reads six contiguous streams.
///
/// # Why structure-of-arrays
///
/// The kernel touches `px[i], py[i], pz[i], zx[i], zy[i], zz[i]` and nothing else. In array-of-
/// structures each vector lane would need a gather; here each lane is a contiguous load, which
/// is what lets the autovectorizer emit `vfmadd` over `ymm` (or `zmm`) without any intrinsics.
///
/// The grid borrows rather than owns, so an Arrow/Parquet column can be handed straight in with
/// no copy, and the type stays `no_std`.
///
/// # Invariant
///
/// All six slices have the same length. That is checked once, in [`CellGrid::new`], and cannot be
/// violated afterwards: the fields are private and there is no other constructor.
#[derive(Debug, Clone, Copy)]
pub struct CellGrid<'a> {
    px: &'a [f64],
    py: &'a [f64],
    pz: &'a [f64],
    zx: &'a [f64],
    zy: &'a [f64],
    zz: &'a [f64],
}

impl<'a> CellGrid<'a> {
    /// Build from six equal-length component slices.
    ///
    /// `p*` are cell centres on the reference ellipsoid, metres. `z*` are the outward unit
    /// normals at those centres; they are not normalised here, and a non-unit `z` silently
    /// changes the meaning of the elevation predicate.
    ///
    /// # Errors
    ///
    /// [`SensorError::RaggedGrid`] if the slices differ in length.
    pub fn new(
        px: &'a [f64],
        py: &'a [f64],
        pz: &'a [f64],
        zx: &'a [f64],
        zy: &'a [f64],
        zz: &'a [f64],
    ) -> Result<Self, SensorError> {
        let lens = [px.len(), py.len(), pz.len(), zx.len(), zy.len(), zz.len()];
        let shortest = lens.iter().copied().min().unwrap_or(0);
        let longest = lens.iter().copied().max().unwrap_or(0);
        if shortest != longest {
            return Err(SensorError::RaggedGrid { shortest, longest });
        }
        Ok(Self {
            px,
            py,
            pz,
            zx,
            zy,
            zz,
        })
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

    /// The six component slices, all of length [`CellGrid::len`].
    #[must_use]
    pub const fn columns(&self) -> Columns<'a> {
        (self.px, self.py, self.pz, self.zx, self.zy, self.zz)
    }
}
