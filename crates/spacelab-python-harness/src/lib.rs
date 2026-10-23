//! `spacelab-python` loaded into CPython as `spacedatalab`, for this repository's Python tests, figures
//! and experiments. The wheel on PyPI is built by
//! [SpaceDataLab/spacedatalab](https://github.com/SpaceDataLab/spacedatalab) from the same
//! `register`, so what these tests import is what users import from spacelab.

use pyo3::prelude::*;

#[pymodule]
fn spacedatalab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    spacelab_python::register(m)
}
