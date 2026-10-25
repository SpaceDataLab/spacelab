//! NOAA's WAM-IPE neutral mass density — the public physics nowcast — as a field the calibration
//! layer can be fitted to.
//!
//! The operational WAM-IPE forecast system publishes, every ten minutes, a `wam_fixed_height` NetCDF
//! classic file: `den(time, hlevs, lat, lon)` in kg m⁻³ on 91 heights from 100 to 1000 km, 91
//! latitudes from −90° to 90° and 90 longitudes from 0° to 356°. WAM's own top is at 400–600 km
//! depending on solar activity; the levels above it are NOAA's offline extension. `tools/fetch-wam-ipe.sh`
//! says which files are nowcasts and fetches them; this module reads one.
//!
//! Three facts about the files are this module's to handle rather than the caller's:
//!
//! * **The pole rows hold zeros.** At ±90° the values are `0.0`, at every height and longitude. A
//!   pole is one point that the grid writes 90 times, and the product leaves it empty — except in a
//!   storm, when the south pole row also gains a few `NaN`, `±inf` and isolated positive values
//!   (2024-05-11 03 UT: 34 of its 8 190). So a pole row is used only if every cell of it holds a
//!   density, and none in the pinned set does. Within the last 2° of a pole the 88° ring is used,
//!   so the field is defined wherever a satellite can be.
//! * **A storm can leave holes.** Of the 1 390 nowcasts `tools/wam-ipe.sha256` pins, nine hold
//!   `0`, `±inf` or `NaN` inside the grid: all of them on 2024-05-11/12, the Gannon storm, at 120
//!   and 130 km, all south of 45° S (the most cells in one file: 1 180 of 728 910). A hole is a
//!   cell, not a file: [`WamIpeField::density`] has no value where the interpolation would touch
//!   one, and the field elsewhere — the satellites' shells among it — stays.
//! * **The height is the product's own coordinate.** NOAA does not document whether it is measured
//!   from an ellipsoid or a sphere, and the difference reaches 21 km between equator and pole —
//!   a third of a scale height at 500 km. [`WamIpeField::density`] therefore takes the product's
//!   height and says nothing about it; `docs/verification.md` §10.18 measures which reading the
//!   accelerometers agree with.
//!
//! Attribution: NOAA Space Weather Prediction Center, WAM-IPE, accessed from
//! <https://registry.opendata.aws/noaa-nws-wam-ipe>. NOAA does not endorse this library.

// Grid indices and counts are below a hundred, so they are exact in `f64`; `NetCDF` and `WAM-IPE`
// are names, not identifiers.
#![allow(clippy::cast_precision_loss, clippy::doc_markdown)]

use alloc::string::String;
use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::density_calibration::DensitySample;
use crate::netcdf_classic::{Dataset, NetcdfError};

/// Why a file could not be read as a WAM-IPE fixed-height density field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WamIpeError {
    /// The bytes are not a readable NetCDF classic file.
    Netcdf(NetcdfError),
    /// A variable the product always carries is absent.
    MissingVariable(&'static str),
    /// A variable's `units` attribute is not the one the product writes.
    Units {
        /// The variable.
        variable: &'static str,
        /// Its `units`, or empty when it has none.
        found: String,
    },
    /// The density variable's shape does not match its coordinate variables.
    Shape,
    /// A coordinate is not uniformly spaced, or the longitudes do not close the circle.
    Axis(&'static str),
    /// Fewer than two latitude rings hold values, so no cell can be interpolated in.
    Rings,
}

impl core::fmt::Display for WamIpeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Netcdf(e) => write!(f, "{e}"),
            Self::MissingVariable(name) => write!(f, "variable {name} is absent"),
            Self::Units { variable, found } => {
                write!(f, "variable {variable} has units {found:?}")
            }
            Self::Shape => write!(f, "den's shape does not match its coordinates"),
            Self::Axis(name) => write!(f, "coordinate {name} is not a uniform grid"),
            Self::Rings => write!(f, "fewer than two latitude rings hold values"),
        }
    }
}

impl core::error::Error for WamIpeError {}

impl From<NetcdfError> for WamIpeError {
    fn from(e: NetcdfError) -> Self {
        Self::Netcdf(e)
    }
}

/// A uniformly spaced coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Axis {
    start: f64,
    step: f64,
    count: usize,
}

impl Axis {
    fn from_values(values: &[f64], name: &'static str) -> Result<Self, WamIpeError> {
        let [first, second, ..] = values else {
            return Err(WamIpeError::Axis(name));
        };
        let step = second - first;
        // The coordinates are float32 multiples of 2, 4 and 10, which float32 holds exactly, so a
        // grid that is uniform at all is uniform to the bit.
        #[expect(clippy::float_cmp, reason = "exact by construction, see above")]
        let uniform = step > 0.0
            && values
                .iter()
                .enumerate()
                .all(|(i, &v)| v == first + step * i as f64);
        if uniform {
            Ok(Self {
                start: *first,
                step,
                count: values.len(),
            })
        } else {
            Err(WamIpeError::Axis(name))
        }
    }

    fn value(self, i: usize) -> f64 {
        self.start + self.step * i as f64
    }

    /// The cell between nodes `lo < hi` that `x` falls in, by its lower node, and the fractional
    /// position in it. `x` beyond either node takes that node, and `hi` itself is the upper end of
    /// the last cell, so both nodes of the cell are always in `lo..=hi`.
    fn locate(self, x: f64, lo: usize, hi: usize) -> (usize, f64) {
        let s = ((x - self.start) / self.step).clamp(lo as f64, hi as f64);
        let i = libm::floor(s).min((hi - 1) as f64);
        // `i` is a whole number in `lo..hi` by the clamp and the `min`.
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a clamped non-negative whole number below the axis length"
        )]
        let index = i as usize;
        (index, s - i)
    }
}

/// One WAM-IPE fixed-height density file: `ln ρ` on its height × latitude × longitude grid.
#[derive(Clone, Debug, PartialEq)]
pub struct WamIpeField {
    unix_seconds: f64,
    heights: Axis,
    latitudes: Axis,
    longitudes: Axis,
    /// The first and last latitude rows that hold values.
    rows: (usize, usize),
    /// `ln ρ` with ρ in kg m⁻³, height-major then latitude then longitude, as the file stores it;
    /// `NaN` where the file holds no positive, finite density.
    log_density: Vec<f64>,
    /// How many cells between the first and last full rings are holes.
    holes: usize,
}

fn units(d: &Dataset<'_>, name: &'static str, expected: &str) -> Result<(), WamIpeError> {
    let v = d.variable(name).ok_or(WamIpeError::MissingVariable(name))?;
    match v.text_attribute("units") {
        Some(u) if u == expected => Ok(()),
        found => Err(WamIpeError::Units {
            variable: name,
            found: String::from(found.unwrap_or("")),
        }),
    }
}

impl WamIpeField {
    /// Read a `wam_fixed_height` file.
    ///
    /// # Errors
    /// [`WamIpeError`] names the first way the bytes are not the product: not NetCDF classic, a
    /// variable or unit that is not the product's, a non-uniform axis, or fewer than two latitude
    /// rings with values.
    pub fn parse(bytes: &[u8]) -> Result<Self, WamIpeError> {
        let d = Dataset::parse(bytes)?;
        units(&d, "lon", "degrees_east")?;
        units(&d, "lat", "degrees_north")?;
        units(&d, "hlevs", "km")?;
        units(&d, "time", "days since 1970-01-01")?;
        units(&d, "den", "kg m^-3")?;
        let values = |name: &'static str| -> Result<Vec<f64>, WamIpeError> {
            let v = d.variable(name).ok_or(WamIpeError::MissingVariable(name))?;
            Ok(d.values(v)?)
        };
        let longitudes = Axis::from_values(&values("lon")?, "lon")?;
        let latitudes = Axis::from_values(&values("lat")?, "lat")?;
        let heights = Axis::from_values(&values("hlevs")?, "hlevs")?;
        #[expect(clippy::float_cmp, reason = "a whole number of 4° steps is exact")]
        let closes = longitudes.start == 0.0 && longitudes.step * longitudes.count as f64 == 360.0;
        if !closes {
            return Err(WamIpeError::Axis("lon"));
        }
        let [days] = values("time")?[..] else {
            return Err(WamIpeError::Shape);
        };

        let den = d
            .variable("den")
            .ok_or(WamIpeError::MissingVariable("den"))?;
        if d.shape(den) != [1, heights.count, latitudes.count, longitudes.count] {
            return Err(WamIpeError::Shape);
        }
        let density = d.values(den)?;
        let (nlat, nlon) = (latitudes.count, longitudes.count);
        let row_complete = |j: usize| {
            (0..heights.count).all(|k| {
                density[(k * nlat + j) * nlon..(k * nlat + j + 1) * nlon]
                    .iter()
                    .all(|&v| v > 0.0 && v.is_finite())
            })
        };
        let first = usize::from(!row_complete(0));
        let last = nlat - 1 - usize::from(!row_complete(nlat - 1));
        if last <= first {
            return Err(WamIpeError::Rings);
        }
        let mut log_density = Vec::with_capacity(density.len());
        let mut holes = 0;
        for (n, &v) in density.iter().enumerate() {
            let j = n / nlon % nlat;
            if (first..=last).contains(&j) && v > 0.0 && v.is_finite() {
                log_density.push(libm::log(v));
            } else {
                holes += usize::from((first..=last).contains(&j));
                log_density.push(f64::NAN);
            }
        }
        Ok(Self {
            unix_seconds: days * 86_400.0,
            heights,
            latitudes,
            longitudes,
            rows: (first, last),
            log_density,
            holes,
        })
    }

    /// The instant the field is valid at, seconds since 1970-01-01 00:00 UTC.
    #[must_use]
    pub const fn unix_seconds(&self) -> f64 {
        self.unix_seconds
    }

    /// How many grid cells off the pole rows hold no positive, finite density.
    #[must_use]
    pub const fn holes(&self) -> usize {
        self.holes
    }

    /// The lowest and highest height the field covers, km.
    #[must_use]
    pub fn height_range(&self) -> (f64, f64) {
        (
            self.heights.start,
            self.heights.value(self.heights.count - 1),
        )
    }

    fn at(&self, k: usize, j: usize, i: usize) -> f64 {
        self.log_density[(k * self.latitudes.count + j) * self.longitudes.count + i]
    }

    /// Density in kg m⁻³ at the product's height `height_km` and at `latitude_deg`,
    /// `longitude_deg`, interpolated linearly in `ln ρ` along all three axes. `None` outside the
    /// product's heights, and where a corner the interpolation weighs is a hole. Latitudes poleward
    /// of the last ring with values take that ring's.
    #[must_use]
    pub fn density(&self, height_km: f64, latitude_deg: f64, longitude_deg: f64) -> Option<f64> {
        let (low, high) = self.height_range();
        if !(low..=high).contains(&height_km) {
            return None;
        }
        let (k, fk) = self.heights.locate(height_km, 0, self.heights.count - 1);
        let (j, fj) = self
            .latitudes
            .locate(latitude_deg, self.rows.0, self.rows.1);
        let lon = longitude_deg - 360.0 * libm::floor(longitude_deg / 360.0);
        let s = lon / self.longitudes.step;
        let i0f = libm::floor(s);
        let fi = s - i0f;
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "a wrapped longitude over a positive step is a small non-negative whole number"
        )]
        let i0 = (i0f as usize) % self.longitudes.count;
        let i1 = (i0 + 1) % self.longitudes.count;
        // A corner of weight zero is skipped rather than multiplied, so that a hole next to a node
        // does not take the node's own value with it.
        let mut log_rho = 0.0;
        for (kk, wk) in [(k, 1.0 - fk), (k + 1, fk)] {
            for (jj, wj) in [(j, 1.0 - fj), (j + 1, fj)] {
                for (ii, wi) in [(i0, 1.0 - fi), (i1, fi)] {
                    let w = wk * wj * wi;
                    if w != 0.0 {
                        log_rho += w * self.at(kk, jj, ii);
                    }
                }
            }
        }
        log_rho.is_finite().then(|| libm::exp(log_rho))
    }

    /// The field at one height on its own grid, as samples of `ln(ρ_WAM-IPE / ρ_base)` for
    /// [`crate::density_calibration::DensityCorrection::fit_ridge`].
    ///
    /// `base(latitude_deg, longitude_deg)` is the base model at the same instant and the same
    /// physical point. Every `stride`-th latitude ring and longitude is taken, each weighted by
    /// `cos(latitude)` so that the fit is over the sphere rather than over the grid's crowding at
    /// the poles. Local solar time is mean solar time, `UT + longitude/15`. A point where `base`
    /// gives no positive, finite density is left out.
    #[must_use]
    pub fn calibration_samples(
        &self,
        height_km: f64,
        stride: NonZeroUsize,
        mut base: impl FnMut(f64, f64) -> f64,
    ) -> Vec<DensitySample> {
        let ut_hours = self.unix_seconds / 3_600.0;
        let mut samples = Vec::new();
        for j in (self.rows.0..=self.rows.1).step_by(stride.get()) {
            let lat = self.latitudes.value(j);
            for i in (0..self.longitudes.count).step_by(stride.get()) {
                let lon = self.longitudes.value(i);
                let (Some(rho), rho_base) = (self.density(height_km, lat, lon), base(lat, lon))
                else {
                    continue;
                };
                if !(rho_base > 0.0 && rho_base.is_finite()) {
                    continue;
                }
                let mut lst = ut_hours + lon / 15.0;
                lst -= 24.0 * libm::floor(lst / 24.0);
                samples.push(DensitySample {
                    latitude: lat.to_radians(),
                    local_solar_time: lst,
                    log_ratio: libm::log(rho / rho_base),
                    weight: libm::cos(lat.to_radians()),
                });
            }
        }
        samples
    }
}

#[cfg(test)]
mod tests {
    // The synthetic file stores float32, as the product does.
    #![allow(
        clippy::unwrap_used,
        clippy::float_cmp,
        clippy::cast_possible_truncation
    )]

    use super::*;
    use crate::density_calibration::DensityCorrection;
    use crate::netcdf_classic::writer::{self, Values, Var};

    const HEIGHTS: [f32; 3] = [100.0, 110.0, 120.0];
    const LATITUDES: [f32; 7] = [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0];
    /// A per-column offset, so that a longitude mix-up cannot pass.
    const COLUMN: [f64; 4] = [0.0, 0.3, -0.2, 0.5];
    /// 2024-05-10 06:00 UTC.
    const DAYS: f64 = 19_853.25;

    /// `ln ρ` of the synthetic field: linear in height and latitude, a per-column offset in
    /// longitude.
    fn log_rho(h: f64, lat: f64, column: usize) -> f64 {
        -20.0 - (h - 100.0) / 50.0 + 0.01 * lat + COLUMN[column % COLUMN.len()]
    }

    /// The product's layout with zero pole rows, one record, and the given changes.
    fn file(den_units: &str, longitudes: &[f32], hole: Option<(usize, f32)>) -> Vec<u8> {
        file_on(&LATITUDES, den_units, longitudes, hole)
    }

    fn file_on(
        latitudes: &[f32],
        den_units: &str,
        longitudes: &[f32],
        hole: Option<(usize, f32)>,
    ) -> Vec<u8> {
        let mut den = Vec::new();
        for &h in &HEIGHTS {
            for &lat in latitudes {
                for column in 0..longitudes.len() {
                    den.push(if lat.abs() == 90.0 {
                        0.0
                    } else {
                        libm::exp(log_rho(f64::from(h), f64::from(lat), column)) as f32
                    });
                }
            }
        }
        if let Some((n, v)) = hole {
            den[n] = v;
        }
        writer::write(
            false,
            1,
            &[
                ("lon", u32::try_from(longitudes.len()).unwrap()),
                ("lat", u32::try_from(latitudes.len()).unwrap()),
                ("hlevs", 3),
                ("time", 0),
            ],
            &[
                Var {
                    name: "lon",
                    dims: &[0],
                    units: Some("degrees_east"),
                    values: Values::Float(longitudes),
                },
                Var {
                    name: "lat",
                    dims: &[1],
                    units: Some("degrees_north"),
                    values: Values::Float(latitudes),
                },
                Var {
                    name: "hlevs",
                    dims: &[2],
                    units: Some("km"),
                    values: Values::Float(&HEIGHTS),
                },
                Var {
                    name: "time",
                    dims: &[3],
                    units: Some("days since 1970-01-01"),
                    values: Values::Double(&[DAYS]),
                },
                Var {
                    name: "den",
                    dims: &[3, 2, 1, 0],
                    units: Some(den_units),
                    values: Values::Float(&den),
                },
            ],
        )
    }

    fn field() -> WamIpeField {
        WamIpeField::parse(&file("kg m^-3", &[0.0, 90.0, 180.0, 270.0], None)).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        // The file stores float32, so a node is good to its rounding.
        (a / b - 1.0).abs() < 1e-6
    }

    #[test]
    fn nodes_read_back_and_cells_interpolate_linearly_in_log_density() {
        let f = field();
        assert_eq!(f.unix_seconds(), DAYS * 86_400.0);
        assert_eq!(f.height_range(), (100.0, 120.0));
        let node = f.density(110.0, 30.0, 90.0).unwrap();
        assert!(close(node, libm::exp(log_rho(110.0, 30.0, 1))));
        // Halfway along every axis: the mean of the eight corners' logarithms.
        let mid = f.density(105.0, 15.0, 45.0).unwrap();
        let expected = -20.0 - 0.1 + 0.15 + (COLUMN[0] + COLUMN[1]) / 2.0;
        assert!(
            close(mid, libm::exp(expected)),
            "{mid} against {}",
            libm::exp(expected)
        );
    }

    #[test]
    fn longitude_wraps_through_the_meridian() {
        let f = field();
        let east = f.density(100.0, 0.0, 315.0).unwrap();
        assert_eq!(f.density(100.0, 0.0, -45.0).unwrap(), east);
        assert_eq!(f.density(100.0, 0.0, 675.0).unwrap(), east);
        assert!(close(
            east,
            libm::exp(-20.0 + (COLUMN[3] + COLUMN[0]) / 2.0)
        ));
    }

    #[test]
    fn the_empty_pole_rows_give_way_to_the_last_full_ring() {
        let f = field();
        for lon in [0.0, 135.0] {
            assert_eq!(f.density(100.0, 89.0, lon), f.density(100.0, 60.0, lon));
            assert_eq!(f.density(100.0, -75.0, lon), f.density(100.0, -60.0, lon));
            assert!(f.density(100.0, 90.0, lon).unwrap().is_finite());
        }
    }

    #[test]
    fn a_storm_pole_row_with_stray_values_is_still_not_a_ring() {
        // Height 0, the south pole row, longitude 1: as in the storm files, a NaN, an infinity and a
        // positive value among the zeros.
        for stray in [f32::NAN, f32::INFINITY, 1e-9] {
            let f = WamIpeField::parse(&file(
                "kg m^-3",
                &[0.0, 90.0, 180.0, 270.0],
                Some((1, stray)),
            ))
            .unwrap();
            assert_eq!(f.holes(), 0, "{stray}");
            assert_eq!(
                f.density(100.0, -89.0, 90.0),
                field().density(100.0, -60.0, 90.0)
            );
        }
    }

    #[test]
    fn one_bad_value_anywhere_in_an_edge_row_takes_that_row_out() {
        // No pole rows: the edge rings at ±60° hold values, so each is a ring until one of its
        // values, at any height and longitude, is not a positive finite density.
        const RINGS: [f32; 5] = [-60.0, -30.0, 0.0, 30.0, 60.0];
        let lons = [0.0, 90.0, 180.0, 270.0];
        let whole = WamIpeField::parse(&file_on(&RINGS, "kg m^-3", &lons, None)).unwrap();
        assert_eq!((whole.rows, whole.holes()), ((0, 4), 0));
        for k in 0..HEIGHTS.len() {
            for i in 0..lons.len() {
                for (j, rows, holes) in [(0, (1, 4), 0), (4, (0, 3), 0), (1, (0, 4), 1)] {
                    let n = (k * RINGS.len() + j) * lons.len() + i;
                    let f = WamIpeField::parse(&file_on(&RINGS, "kg m^-3", &lons, Some((n, 0.0))))
                        .unwrap();
                    assert_eq!(
                        (f.rows, f.holes()),
                        (rows, holes),
                        "height {k}, ring {j}, column {i}"
                    );
                }
            }
        }
    }

    #[test]
    fn heights_outside_the_product_have_no_density() {
        let f = field();
        assert!(f.density(99.9, 0.0, 0.0).is_none());
        assert!(f.density(120.1, 0.0, 0.0).is_none());
        assert!(close(
            f.density(120.0, 0.0, 0.0).unwrap(),
            libm::exp(log_rho(120.0, 0.0, 0))
        ));
    }

    #[test]
    fn a_file_that_is_not_the_product_is_named() {
        let lons = [0.0, 90.0, 180.0, 270.0];
        assert_eq!(
            WamIpeField::parse(&file("g cm^-3", &lons, None)).unwrap_err(),
            WamIpeError::Units {
                variable: "den",
                found: String::from("g cm^-3")
            }
        );
        assert_eq!(
            WamIpeField::parse(&file("kg m^-3", &[0.0, 80.0, 160.0, 240.0], None)).unwrap_err(),
            WamIpeError::Axis("lon")
        );
        // Poles only, or one ring between them: no cell has two rows with values.
        for latitudes in [&[-90.0, 90.0][..], &[-90.0, 0.0, 90.0]] {
            assert_eq!(
                WamIpeField::parse(&file_on(latitudes, "kg m^-3", &lons, None)).unwrap_err(),
                WamIpeError::Rings
            );
        }
    }

    #[test]
    fn an_axis_is_uniform_and_increasing_or_refused() {
        let axis = Axis::from_values(&[0.0, 2.0, 4.0], "x").unwrap();
        assert_eq!((axis.start, axis.step, axis.count), (0.0, 2.0, 3));
        for values in [
            &[5.0, 5.0, 5.0][..],
            &[4.0, 2.0, 0.0],
            &[0.0, 1.0, 3.0],
            &[0.0],
        ] {
            assert_eq!(
                Axis::from_values(values, "x").unwrap_err(),
                WamIpeError::Axis("x"),
                "{values:?}"
            );
        }
    }

    #[test]
    fn a_point_is_located_in_a_cell_that_has_both_its_nodes() {
        let axis = Axis {
            start: 0.0,
            step: 1.0,
            count: 5,
        };
        assert_eq!(axis.locate(0.0, 0, 4), (0, 0.0));
        assert_eq!(axis.locate(3.5, 0, 4), (3, 0.5));
        // The last node is the upper end of the last cell, not a cell of its own.
        assert_eq!(axis.locate(4.0, 0, 4), (3, 1.0));
        // Inside narrower bounds, beyond them, and with the bounds one cell apart.
        assert_eq!(axis.locate(1.5, 1, 3), (1, 0.5));
        assert_eq!(axis.locate(0.2, 1, 3), (1, 0.0));
        assert_eq!(axis.locate(3.8, 1, 3), (2, 1.0));
        assert_eq!(axis.locate(-7.0, 2, 3), (2, 0.0));
        assert_eq!(axis.locate(9.0, 2, 3), (2, 1.0));
    }

    #[test]
    fn each_error_names_what_went_wrong() {
        let cases = [
            (
                WamIpeError::Netcdf(NetcdfError::Truncated(8)),
                "file ends inside the structure at byte 8",
            ),
            (
                WamIpeError::MissingVariable("den"),
                "variable den is absent",
            ),
            (
                WamIpeError::Units {
                    variable: "den",
                    found: String::from("g cm^-3"),
                },
                "variable den has units \"g cm^-3\"",
            ),
            (
                WamIpeError::Shape,
                "den's shape does not match its coordinates",
            ),
            (
                WamIpeError::Axis("lat"),
                "coordinate lat is not a uniform grid",
            ),
            (
                WamIpeError::Rings,
                "fewer than two latitude rings hold values",
            ),
        ];
        for (e, text) in cases {
            assert_eq!(alloc::format!("{e}"), text);
        }
    }

    #[test]
    fn a_hole_takes_out_the_cells_that_touch_it_and_nothing_else() {
        // Height 1 (110 km), latitude 3 (the equator), longitude 2 (180°): off the pole rows.
        let f = WamIpeField::parse(&file(
            "kg m^-3",
            &[0.0, 90.0, 180.0, 270.0],
            Some(((7 + 3) * 4 + 2, 0.0)),
        ))
        .unwrap();
        assert_eq!(f.holes(), 1);
        assert_eq!(field().holes(), 0);
        assert!(f.density(110.0, 0.0, 180.0).is_none());
        assert!(f.density(110.0, 15.0, 135.0).is_none());
        assert!(f.density(105.0, -10.0, 200.0).is_none());
        // A neighbouring node weighs the hole at zero, and a cell that does not touch it is whole.
        assert_eq!(
            f.density(110.0, 0.0, 90.0),
            field().density(110.0, 0.0, 90.0)
        );
        assert_eq!(
            f.density(120.0, 0.0, 180.0),
            field().density(120.0, 0.0, 180.0)
        );
        assert_eq!(
            f.density(105.0, 45.0, 45.0),
            field().density(105.0, 45.0, 45.0)
        );
    }

    #[test]
    fn calibration_samples_carry_the_ratio_to_the_base_on_the_sphere() {
        let f = field();
        let half = |lat: f64, lon: f64| f.density(110.0, lat, lon).unwrap() / 2.0;
        let all = f.calibration_samples(110.0, NonZeroUsize::new(1).unwrap(), half);
        assert_eq!(all.len(), 5 * 4, "five full rings of four longitudes");
        for s in &all {
            assert!((s.log_ratio - core::f64::consts::LN_2).abs() < 1e-12);
            assert_eq!(s.weight, libm::cos(s.latitude));
        }
        // 06:00 UTC: mean solar time at 0°, 90°, 180° and 270° east, wrapped into a day.
        for ring in all.chunks(4) {
            let lst: Vec<f64> = ring.iter().map(|s| s.local_solar_time).collect();
            assert_eq!(lst, [6.0, 12.0, 18.0, 0.0]);
        }
        let every_other = f.calibration_samples(110.0, NonZeroUsize::new(2).unwrap(), half);
        assert_eq!(every_other.len(), 3 * 2);
        let scale = DensityCorrection::<1>::fit(&all).unwrap();
        assert!((scale.factor(0.3, 7.0) - 2.0).abs() < 1e-12);
        // A base model with nothing to say at a point drops the point, not the fit.
        let partial = f.calibration_samples(110.0, NonZeroUsize::new(1).unwrap(), |lat, lon| {
            if lat > 50.0 { f64::NAN } else { half(lat, lon) }
        });
        assert_eq!(partial.len(), 4 * 4);
        // Nor does a base density that is zero, negative or infinite.
        for bad in [0.0, -1e-12, f64::INFINITY] {
            let partial =
                f.calibration_samples(110.0, NonZeroUsize::new(1).unwrap(), |lat, lon| {
                    if lat > 50.0 { bad } else { half(lat, lon) }
                });
            assert_eq!(partial.len(), 4 * 4, "{bad}");
        }
    }

    #[test]
    fn local_solar_time_runs_an_hour_per_fifteen_degrees_east() {
        // Columns 72° apart: at a multiple of 45°, `lon / 15` and `lon * 15` hours agree modulo a
        // day, so a 90° grid cannot tell the two apart.
        let lons = [0.0, 72.0, 144.0, 216.0, 288.0];
        let f = WamIpeField::parse(&file("kg m^-3", &lons, None)).unwrap();
        let samples = f.calibration_samples(110.0, NonZeroUsize::new(1).unwrap(), |_, _| 1e-9);
        assert_eq!(samples.len(), 5 * lons.len());
        for (s, expected) in samples.iter().zip([6.0, 10.8, 15.6, 20.4, 1.2]) {
            assert!(
                (s.local_solar_time - expected).abs() < 1e-9,
                "{} against {expected}",
                s.local_solar_time
            );
        }
    }

    #[test]
    fn between_the_edge_ring_and_its_neighbour_the_field_interpolates() {
        // Row 1 (-60°) is the first ring, row 5 (60°) the last; the field is linear in latitude.
        let f = field();
        for lat in [-55.0, -45.0, 45.0, 55.0] {
            assert!(close(
                f.density(100.0, lat, 0.0).unwrap(),
                libm::exp(log_rho(100.0, lat, 0))
            ));
        }
    }
}
