//! Modified-dip-latitude (MODIP) grid and its third-order interpolation.
//!
//! Ported from Orekit's `ModipGrid`. The interpolator reads any grid that already carries its
//! anti-meridian and past-pole wrap rows and columns ([`ModipGrid::from_wrapped`]). The
//! Galileo-specific `modipNeQG_wrapped.asc` is such a grid: a 5°×10° sampling whose wrap cells are
//! in the file, giving a 39×39 table read verbatim.
//!
//! Every grid samples latitude and longitude with an equal cell count per axis, so a single
//! `nb_cells` drives the index arithmetic of `compute`.

use std::f64::consts::{FRAC_PI_2, PI};

/// Galileo grid cell count per axis (36 in latitude and longitude).
const GALILEO_NB_CELLS: usize = 36;

/// Side length of the pre-wrapped Galileo grid (`GALILEO_NB_CELLS + 3`).
const GALILEO_GRID_SIDE: usize = GALILEO_NB_CELLS + 3;

/// Embedded Galileo MODIP grid, in degrees.
const GALILEO_MODIP_GRID: &str = include_str!("../assets/modipNeQG_wrapped.asc");

/// The bilinear-in-longitude / third-order-in-latitude MODIP interpolator.
pub(crate) struct ModipGrid {
    /// Cell count per axis (latitude and longitude counts are equal in both variants).
    nb_cells: usize,
    /// Cell size in latitude, radians (`π / nb_cells`).
    size_lat: f64,
    /// Cell size in longitude, radians (`2π / nb_cells`).
    size_lon: f64,
    /// The `(nb_cells + 3) × (nb_cells + 3)` grid, row = latitude, column = longitude, degrees.
    pub(crate) grid: Vec<Vec<f64>>,
}

impl ModipGrid {
    /// Parse the embedded Galileo MODIP grid (wrap rows/columns already in the file).
    pub(crate) fn galileo() -> Self {
        let values: Vec<f64> = GALILEO_MODIP_GRID
            .split_whitespace()
            .filter_map(|token| token.parse::<f64>().ok())
            .collect();
        let grid: Vec<Vec<f64>> = values
            .chunks(GALILEO_GRID_SIDE)
            .map(<[f64]>::to_vec)
            .collect();
        Self::from_wrapped(GALILEO_NB_CELLS, grid)
    }

    /// A grid of `nb_cells` cells per axis whose `(nb_cells + 3)²` samples already include the
    /// wrap rows and columns, row = latitude, column = longitude, degrees.
    pub(crate) fn from_wrapped(nb_cells: usize, grid: Vec<Vec<f64>>) -> Self {
        debug_assert!(grid.len() == nb_cells + 3 && grid.iter().all(|r| r.len() == nb_cells + 3));
        Self {
            nb_cells,
            size_lat: PI / nb_cells as f64,
            size_lon: 2.0 * PI / nb_cells as f64,
            grid,
        }
    }

    /// Compute the MODIP at a location, in degrees (Orekit `ModipGrid.computeMODIP`).
    pub(crate) fn compute(&self, latitude: f64, longitude: f64) -> f64 {
        let nb = self.nb_cells;

        // The coordinates arrive from a `GeodeticPoint`, whose constructor validates nothing, and
        // NaN compares false against every bound. It would slip past both saturating guards below
        // and reach `libm::floor(NaN) as usize`, which saturates to 0, indexing the grid at
        // `[0 - 1]`. Infinities are already answered by those guards, being ordered.
        if latitude.is_nan() {
            return f64::NAN;
        }

        // Latitude index: two points before and two after for the cubic (Eq. 12/13).
        let x = (latitude + FRAC_PI_2) / self.size_lat + 1.0;
        if x < 1.0 {
            return -90.0;
        }
        if x >= (nb + 1) as f64 {
            return 90.0;
        }
        let i = libm::floor(x) as usize;
        let delta_x = x - i as f64;

        // Longitude index, wrapped around the anti-meridian. The wrap is a modular reduction and
        // has to be written as one. Added or subtracted a turn at a time its cost is proportional
        // to the input, and the input is unreduced: `Ray` accumulates the longitude as
        // `atan2(..) + lon_p` from an unvalidated `GeodeticPoint`, so a receiver at 1e10 rad costs
        // 1.6e9 passes and one at 1e300 never returns. `fmod` is exact, and the branch leaves the
        // in-range case -- every longitude the fixtures produce -- on the arithmetic it had.
        let mut y = (longitude + PI) / self.size_lon + 1.0;
        if !(1.0..(nb + 1) as f64).contains(&y) {
            // NaN and both infinities land here, and none of them names a meridian; so does a
            // longitude near `f64::MAX`, which overflows when divided by the cell width.
            if !y.is_finite() {
                return f64::NAN;
            }
            y = libm::fmod(y - 1.0, nb as f64);
            if y < 0.0 {
                y += nb as f64;
            }
            y += 1.0;
            // A tiny negative remainder rounds up to a whole turn when the two are added, which
            // would index one column past the wrap. The fold is one cell wide and lands on its twin.
            if y >= (nb + 1) as f64 {
                y = 1.0;
            }
        }
        let j = libm::floor(y) as usize;
        let delta_y = y - j as f64;

        let g = &self.grid;
        let z1 = interpolate(
            g[i - 1][j - 1],
            g[i][j - 1],
            g[i + 1][j - 1],
            g[i + 2][j - 1],
            delta_x,
        );
        let z2 = interpolate(g[i - 1][j], g[i][j], g[i + 1][j], g[i + 2][j], delta_x);
        let z3 = interpolate(
            g[i - 1][j + 1],
            g[i][j + 1],
            g[i + 1][j + 1],
            g[i + 2][j + 1],
            delta_x,
        );
        let z4 = interpolate(
            g[i - 1][j + 2],
            g[i][j + 2],
            g[i + 1][j + 2],
            g[i + 2][j + 2],
            delta_x,
        );

        interpolate(z1, z2, z3, z4, delta_y)
    }
}

/// Third-order interpolation across four samples (Orekit `ModipGrid.interpolate`,
/// reference Eq. 128–138).
fn interpolate(z1: f64, z2: f64, z3: f64, z4: f64, x: f64) -> f64 {
    if x < 5e-11 {
        return z2;
    }
    let delta = 2.0 * x - 1.0;
    let g1 = z3 + z2;
    let g2 = z3 - z2;
    let g3 = z4 + z1;
    let g4 = (z4 - z1) / 3.0;
    let a0 = 9.0 * g1 - g3;
    let a1 = 9.0 * g2 - g4;
    let a2 = g3 - g1;
    let a3 = g4 - g2;
    0.0625 * (a0 + delta * (a1 + delta * (a2 + delta * a3)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// `interpolate` is the cubic through its own four samples.
    ///
    /// The samples sit at `x = −1, 0, 1, 2` and the argument runs over `[0, 1]`, so a cubic
    /// sampled at those nodes has to come back. Nothing in the sTEC fixtures pins this: the grid
    /// values are what they are, and any smooth surface through them clears an end-to-end
    /// comparison at 1e-3 TECU.
    #[test]
    #[allow(clippy::float_cmp)] // an interpolant at its own node, where exactness is the claim
    fn interpolate_is_the_cubic_through_its_four_samples() {
        let p = |t: f64| 0.7 + 1.3 * t - 0.4 * t * t + 0.11 * t * t * t;
        let (z1, z2, z3, z4) = (p(-1.0), p(0.0), p(1.0), p(2.0));
        for k in 0..=1000 {
            let x = f64::from(k) / 1000.0;
            let got = interpolate(z1, z2, z3, z4, x);
            assert!((got - p(x)).abs() < 1e-14, "at x = {x}: {got} vs {}", p(x));
        }

        // The short circuit stops just short of 5e-11, where the cubic still has to be evaluated:
        // it is 6.5e-11 away from `z2` there, four orders above the agreement above.
        let boundary = interpolate(z1, z2, z3, z4, 5e-11);
        assert!(
            (boundary - p(5e-11)).abs() < 1e-15,
            "at the guard boundary: {boundary} vs {}",
            p(5e-11)
        );

        // At the node itself the sample is returned, not recomputed. The cubic form agrees only
        // to rounding, so this is a statement about the short circuit.
        let g = ModipGrid::galileo();
        for row in g.grid.windows(4) {
            for (j, &centre) in row[1].iter().enumerate() {
                assert_eq!(
                    interpolate(row[0][j], centre, row[2][j], row[3][j], 0.0),
                    centre
                );
            }
        }
    }

    /// Every grid node comes back as itself, and the poles saturate.
    ///
    /// Latitude `−90 + 5a` and longitude `−180 + 10b` land on integer indices, so both
    /// interpolations short-circuit and the tabulated sample is the answer. The exception is the
    /// north pole, where the index reaches `nb + 1` and the saturation guard returns exactly 90°
    /// rather than the table's own value.
    #[test]
    #[allow(clippy::float_cmp)] // the saturation constants are exact by construction
    fn every_grid_node_is_reproduced_and_the_poles_saturate() {
        let galileo = ModipGrid::galileo();
        for a in 0..36_usize {
            for b in 0..=36_usize {
                let lat = ((a as f64) * 5.0 - 90.0).to_radians();
                let lon = ((b as f64) * 10.0 - 180.0).to_radians();
                let want = galileo.grid[a + 1][b + 1];
                let got = galileo.compute(lat, lon);
                assert!(
                    (got - want).abs() < 1e-12,
                    "galileo node ({a}, {b}): {got} vs {want}"
                );
            }
        }

        assert_saturates_beyond_the_poles(&galileo);
    }

    /// Beyond either pole is a caller error, and saturating is what keeps the stencil inside the
    /// table rather than indexing before its first row.
    #[allow(clippy::float_cmp)] // the saturation constants are exact by construction
    pub(crate) fn assert_saturates_beyond_the_poles(grid: &ModipGrid) {
        assert_eq!(grid.compute(-2.0, 0.0), -90.0);
        assert_eq!(grid.compute(2.0, 0.0), 90.0);
    }

    /// A grid sampled from a bicubic surface is interpolated back to that surface.
    ///
    /// `compute` is a tensor product — four cubics down the latitude stencil, then one across
    /// them in longitude — so it reproduces any polynomial of degree three in each index, and it
    /// does so only if all sixteen samples it reads are the ones the stencil calls for. Node
    /// reproduction cannot see a wrong row offset, because at a node the interpolation returns
    /// the centre sample and never looks at the other three. The test points are chosen by index
    /// and mapped back to a latitude and longitude, so the forward index arithmetic is under test
    /// rather than repeated.
    #[test]
    fn a_bicubic_sampled_on_the_grid_comes_back_from_between_the_nodes() {
        const NB: usize = 12;
        const SIDE: usize = NB + 3;
        // Degree three in each index, with terms of every mixed order so no offset cancels.
        let p = |u: f64, v: f64| {
            let (u2, u3) = (u * u, u * u * u);
            let (v2, v3) = (v * v, v * v * v);
            7.0 - 3.1 * u + 0.45 * u2 - 0.021 * u3 + 2.7 * v - 0.31 * v2 + 0.017 * v3 - 0.13 * u * v
                + 0.008 * u2 * v
                - 0.004 * u * v2
                + 0.0011 * u3 * v3
        };
        let grid: Vec<Vec<f64>> = (0..SIDE)
            .map(|i| (0..SIDE).map(|j| p(i as f64, j as f64)).collect())
            .collect();
        let size_lat = PI / NB as f64;
        let size_lon = 2.0 * PI / NB as f64;
        let g = ModipGrid {
            nb_cells: NB,
            size_lat,
            size_lon,
            grid,
        };

        for a in 0..(NB * 4) {
            // Indices strictly inside `[1, NB + 1)`, off the nodes so the cubic actually runs.
            let x = 1.0 + (a as f64 + 0.5) / 4.0;
            for b in 0..(NB * 4) {
                let y = 1.0 + (b as f64 + 0.5) / 4.0;
                let lat = (x - 1.0) * size_lat - FRAC_PI_2;
                let lon = (y - 1.0) * size_lon - PI;
                let got = g.compute(lat, lon);
                let want = p(x, y);
                assert!(
                    (got - want).abs() < 1e-9,
                    "at index ({x}, {y}): {got} vs {want}"
                );
            }
        }
    }

    /// The reference points Orekit's own `ModipGridTest` checks.
    ///
    /// A third-party implementation of the same interpolation over the same tables, at points
    /// chosen by its authors rather than by ours: five on grid nodes and nine strictly between them
    /// per grid. The interpolated nine are the ones that matter here — they are the
    /// only external statement about what the surface does away from a sample, and the row and
    /// column offsets of the stencil are visible nowhere else. Orekit's own tolerance is 1e-6°,
    /// which is what is used; the measured worst residual over the twenty-eight is 4.6e-7°.
    #[test]
    fn the_orekit_reference_points_are_reproduced_on_the_galileo_grid() {
        // (latitude °, longitude °, MODIP °) — Orekit `ModipGridTest.testGalileoGrid`.
        const GALILEO: [(f64, f64, f64); 14] = [
            (-85.0, -190.0, -77.31),
            (-85.0, -120.0, -76.81),
            (-85.0, 110.0, -77.19),
            (-5.0, -110.0, 3.20),
            (60.0, 20.0, 60.85),
            (55.1, 13.4, 57.954_196),
            (60.1, 13.4, 60.849_386),
            (65.1, 13.4, 63.771_290),
            (55.1, 23.4, 58.088_051),
            (60.1, 23.4, 60.946_660),
            (65.1, 23.4, 63.837_296),
            (55.1, 33.4, 58.259_195),
            (60.1, 33.4, 61.087_074),
            (65.1, 33.4, 63.954_376),
        ];
        assert_reference_points("galileo", &ModipGrid::galileo(), &GALILEO, -90.0);
    }

    /// Orekit's reference points on one grid, its wrap at the anti-meridian, and its poles.
    /// `at_pole` is the grid's own south-pole sample, which the stencil returns a hair inside it.
    pub(crate) fn assert_reference_points(
        name: &str,
        grid: &ModipGrid,
        cases: &[(f64, f64, f64)],
        at_pole: f64,
    ) {
        {
            for &(lat_deg, lon_deg, want) in cases {
                let got = grid.compute(lat_deg.to_radians(), lon_deg.to_radians());
                assert!(
                    (got - want).abs() < 1e-6,
                    "{name} at ({lat_deg}, {lon_deg}): {got} vs {want}"
                );
            }

            // Longitude runs off both ends of the table and has to come back around: the first
            // and last three columns are the same meridians as the ones a turn away.
            let mut lat_deg = -85.0_f64;
            while lat_deg <= 90.0 {
                for (a, b) in [(-190.0, 170.0), (-180.0, 180.0), (-170.0, 190.0)] {
                    let x = grid.compute(lat_deg.to_radians(), f64::to_radians(a));
                    let y = grid.compute(lat_deg.to_radians(), f64::to_radians(b));
                    assert!(
                        (x - y).abs() < 1e-6,
                        "{name} at {lat_deg}°: {a}° gives {x}, {b}° gives {y}"
                    );
                }
                lat_deg += 5.0;
            }

            // A degree past either pole, and a tenth of a nanodegree short of one.
            let mut lon_deg = -1000.0_f64;
            while lon_deg <= 1000.0 {
                let lon = lon_deg.to_radians();
                assert!((grid.compute(-91.0_f64.to_radians(), lon) + 90.0).abs() < 1e-6);
                assert!((grid.compute(91.0_f64.to_radians(), lon) - 90.0).abs() < 1e-6);
                assert!(
                    (grid.compute((-90.0_f64 + 1.0e-10).to_radians(), lon) - at_pole).abs() < 1e-6
                );
                assert!(
                    (grid.compute((90.0_f64 - 1.0e-10).to_radians(), lon) + at_pole).abs() < 1e-6
                );
                lon_deg += 0.125;
            }
        }
    }

    /// The longitude wrap terminates whatever the caller hands it, and is periodic.
    ///
    /// `GeodeticPoint::new` validates nothing and `Ray` carries the longitude additively as
    /// `atan2(..) + lon_p`, never reducing it, so the value reaching `compute` is whatever the
    /// caller supplied. Walked one turn at a time the wrap costs Θ(|longitude|): 1.6e9 passes at
    /// 1e10 rad and no return at all at 1e300 — a hang inside a `pub` STEC call. Every fixture
    /// stays within a turn of the prime meridian, so nothing could see it.
    #[test]
    fn the_longitude_wrap_terminates_for_any_finite_input() {
        let g = ModipGrid::galileo();
        let lat = 0.6;
        let base = g.compute(lat, 0.3);

        // A whole turn is the wrap's period, so the answer has to come back to itself. The bound is
        // set by the rounding in the argument, not by the wrap: at seven turns `0.3 + 14π` keeps
        // only 15 of its 16 digits.
        for k in [1.0_f64, -1.0, 5.0, -7.0] {
            let got = g.compute(lat, 0.3 + k * 2.0 * PI);
            assert!((got - base).abs() < 1e-9, "{k} turns away: {got} vs {base}");
        }

        // Far past anything a loop could walk. The answer is meaningless at these magnitudes —
        // every bit of the longitude is long gone — but it must be a MODIP, and it must arrive.
        for lon in [1e10_f64, -1e10, 1e300, -1e300] {
            let got = g.compute(lat, lon);
            assert!(
                (-90.0..=90.0).contains(&got),
                "longitude {lon:e} gave {got}"
            );
        }

        // The one input that fires the fold below the wrap. One ulp under −π leaves a remainder of
        // −2.55e-15 cells, and `r + nb` rounds *up* to a whole turn because `|r|` is under half an
        // ulp of 36; `+ 1` then lands on 37.0, the duplicated wrap column, which is one past the
        // last index the stencil can read from. Nothing else in the crate reaches it, so without
        // this the fold is untested code that the wrap's own correctness rests on.
        let under = f64::from_bits((-PI).to_bits() + 1);
        let folded = g.compute(lat, under);
        assert!(
            (folded - g.compute(lat, -PI)).abs() < 1e-9,
            "one ulp under −π indexed off the wrap column: {folded} vs {}",
            g.compute(lat, -PI)
        );
    }

    /// A coordinate outside the domain must not reach the grid index.
    ///
    /// NaN compares false against every bound, so it slips past the saturating latitude guards and
    /// past the wrap, and `libm::floor(NaN) as usize` is 0 — the lookup then indexes `[0 - 1]`. A
    /// longitude near `f64::MAX` gets there differently: divided by the cell width it overflows.
    #[test]
    #[allow(clippy::float_cmp)] // the poles are saturated to a literal, so exactness is the claim
    fn a_coordinate_outside_the_domain_does_not_index_the_grid() {
        assert_outside_the_domain_is_not_indexed(&ModipGrid::galileo());
    }

    #[allow(clippy::float_cmp)] // the poles are saturated to a literal, so exactness is the claim
    pub(crate) fn assert_outside_the_domain_is_not_indexed(g: &ModipGrid) {
        {
            assert!(g.compute(f64::NAN, 0.3).is_nan());
            assert!(g.compute(0.6, f64::NAN).is_nan());
            assert!(g.compute(0.6, f64::INFINITY).is_nan());
            assert!(g.compute(0.6, f64::NEG_INFINITY).is_nan());
            assert!(g.compute(0.6, f64::MAX).is_nan());
            assert!(g.compute(0.6, f64::MIN).is_nan());
            // Latitude keeps the saturation it already had: the guards are ordered comparisons and
            // an infinity answers them.
            assert_eq!(g.compute(f64::INFINITY, 0.3), 90.0);
            assert_eq!(g.compute(f64::NEG_INFINITY, 0.3), -90.0);
        }
    }
}
