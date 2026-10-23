//! RPC bias compensation by block adjustment (Grodecki & Dial 2003) — correcting the systematic error
//! of a vendor RPC with a small image-space affine, estimated from ground control.
//!
//! A vendor [`RationalPolynomialCamera`] carries a bias of a few metres to tens of metres. Rather than
//! re-fit its 80+ coefficients, Grodecki & Dial add a low-order **affine correction in image space** on
//! top of the RPC forward projection:
//!
//! ```text
//! row_adj = row_rpc + (a0 + a1·col_rpc + a2·row_rpc)
//! col_adj = col_rpc + (b0 + b1·col_rpc + b2·row_rpc)
//! ```
//!
//! The shift terms `(a0, b0)` absorb most of the bias; the full six-parameter affine also takes out a
//! linear drift across the scene. The parameters are estimated from ground control points by least
//! squares — and because the row and column corrections are independent linear models sharing the
//! design `[1, col_rpc, row_rpc]`, the fit is two small normal-equation solves.
//!
//! [`estimate_adjustment`] fits the correction; [`forward_adjusted`] applies it. `no_std` + `alloc`.

// Short affine names (a0/a1/a2, r0/c0) mirror the Grodecki–Dial equations; casts are in range.
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::similar_names
)]

use crate::rpc::RationalPolynomialCamera;
use alloc::vec::Vec;

/// A ground control point: a known ground position and its measured image location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundControlPoint {
    /// Geodetic latitude (degrees).
    pub latitude: f64,
    /// Geodetic longitude (degrees).
    pub longitude: f64,
    /// Height above the ellipsoid (metres).
    pub height: f64,
    /// Measured image row (line, pixels).
    pub row: f64,
    /// Measured image column (sample, pixels).
    pub col: f64,
}

/// Which image-space affine to fit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjustmentModel {
    /// Offset only: the two shift terms `a0, b0` (one GCP suffices).
    Shift,
    /// Full six-parameter affine `a0..a2, b0..b2` (needs ≥3 non-collinear GCPs).
    Affine,
}

/// An image-space affine bias correction: `Δrow = a0 + a1·col + a2·row`, `Δcol = b0 + b1·col + b2·row`,
/// with `row = [a0, a1, a2]` and `col = [b0, b1, b2]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RpcAdjustment {
    /// Row-correction affine coefficients `[a0, a1, a2]`.
    pub row: [f64; 3],
    /// Column-correction affine coefficients `[b0, b1, b2]`.
    pub col: [f64; 3],
}

impl RpcAdjustment {
    /// The zero adjustment (leaves the RPC unchanged).
    pub const IDENTITY: Self = Self {
        row: [0.0; 3],
        col: [0.0; 3],
    };

    /// Apply the affine correction to a raw RPC image point `(row_rpc, col_rpc)`.
    #[must_use]
    pub fn correct(&self, row_rpc: f64, col_rpc: f64) -> (f64, f64) {
        (
            row_rpc + self.row[0] + self.row[1] * col_rpc + self.row[2] * row_rpc,
            col_rpc + self.col[0] + self.col[1] * col_rpc + self.col[2] * row_rpc,
        )
    }
}

/// The bias-corrected forward projection: RPC forward, then the [`RpcAdjustment`] affine.
#[must_use]
pub fn forward_adjusted(
    rpc: &RationalPolynomialCamera,
    latitude: f64,
    longitude: f64,
    height: f64,
    adj: &RpcAdjustment,
) -> (f64, f64) {
    let (row, col) = rpc.forward(latitude, longitude, height);
    adj.correct(row, col)
}

/// Solve the symmetric 3×3 system `A x = b` (row-major `A`) by Gaussian elimination with partial
/// pivoting; `None` if singular.
fn solve_sym3(mut a: [f64; 9], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in (col + 1)..3 {
            if libm::fabs(a[r * 3 + col]) > libm::fabs(a[piv * 3 + col]) {
                piv = r;
            }
        }
        if libm::fabs(a[piv * 3 + col]) < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..3 {
                a.swap(col * 3 + c, piv * 3 + c);
            }
            b.swap(col, piv);
        }
        for r in 0..3 {
            if r != col {
                let f = a[r * 3 + col] / a[col * 3 + col];
                for c in col..3 {
                    a[r * 3 + c] -= f * a[col * 3 + c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    Some([b[0] / a[0], b[1] / a[4], b[2] / a[8]])
}

/// Least-squares fit of one affine axis: predictors `x_i = [1, col_i, row_i]`, target `t_i`. Returns the
/// three coefficients (falling back to an offset-only mean if the normal matrix is singular, e.g. one
/// GCP).
fn fit_axis(design: &[[f64; 3]], targets: &[f64]) -> [f64; 3] {
    let mut ata = [0.0f64; 9];
    let mut atb = [0.0f64; 3];
    for (x, &t) in design.iter().zip(targets) {
        for i in 0..3 {
            for j in 0..3 {
                ata[i * 3 + j] += x[i] * x[j];
            }
            atb[i] += x[i] * t;
        }
    }
    solve_sym3(ata, atb).unwrap_or_else(|| {
        let mean = targets.iter().sum::<f64>() / targets.len().max(1) as f64;
        [mean, 0.0, 0.0]
    })
}

/// Estimate the [`RpcAdjustment`] that best fits `gcps` under `model`, by least squares on the row and
/// column residuals of the RPC forward projection. `Affine` fits all six parameters; `Shift` fits only
/// the two offsets (the mean residual).
///
/// # Panics
///
/// If `gcps` is empty.
#[must_use]
pub fn estimate_adjustment(
    rpc: &RationalPolynomialCamera,
    gcps: &[GroundControlPoint],
    model: AdjustmentModel,
) -> RpcAdjustment {
    assert!(!gcps.is_empty(), "need at least one GCP");
    let mut design = Vec::with_capacity(gcps.len());
    let mut tr = Vec::with_capacity(gcps.len());
    let mut tc = Vec::with_capacity(gcps.len());
    for g in gcps {
        let (r0, c0) = rpc.forward(g.latitude, g.longitude, g.height);
        design.push([1.0, c0, r0]);
        tr.push(g.row - r0);
        tc.push(g.col - c0);
    }
    match model {
        AdjustmentModel::Affine => RpcAdjustment {
            row: fit_axis(&design, &tr),
            col: fit_axis(&design, &tc),
        },
        AdjustmentModel::Shift => {
            let n = gcps.len() as f64;
            RpcAdjustment {
                row: [tr.iter().sum::<f64>() / n, 0.0, 0.0],
                col: [tc.iter().sum::<f64>() / n, 0.0, 0.0],
            }
        }
    }
}

/// Root-mean-square image residual of the GCPs under an adjustment (pixels) — the fit quality.
#[must_use]
pub fn adjustment_residual_rms(
    rpc: &RationalPolynomialCamera,
    gcps: &[GroundControlPoint],
    adj: &RpcAdjustment,
) -> f64 {
    let mut s = 0.0;
    for g in gcps {
        let (r, c) = forward_adjusted(rpc, g.latitude, g.longitude, g.height, adj);
        s += (r - g.row) * (r - g.row) + (c - g.col) * (c - g.col);
    }
    libm::sqrt(s / gcps.len() as f64)
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unreadable_literal,
        clippy::float_cmp,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::too_many_lines
    )]
    use super::*;

    const RPB: &str = include_str!("../tests/fixtures/pleiades_reunion.rpb");

    fn rpc() -> RationalPolynomialCamera {
        RationalPolynomialCamera::parse_rpb(RPB).unwrap()
    }

    // A lat/lon/height grid inside the RPC's valid box.
    fn grid(cam: &RationalPolynomialCamera) -> Vec<(f64, f64, f64)> {
        let mut g = Vec::new();
        for fl in [-0.6, 0.0, 0.6] {
            for fo in [-0.6, 0.0, 0.6] {
                for h in [cam.height_offset, cam.height_offset + 400.0] {
                    g.push((
                        cam.lat_offset + fl * cam.lat_scale,
                        cam.lon_offset + fo * cam.lon_scale,
                        h,
                    ));
                }
            }
        }
        g
    }

    #[test]
    fn recovers_an_injected_affine_bias_exactly() {
        // Generate observations self-consistently: RPC forward + a known affine. estimate_adjustment
        // must recover that affine to machine precision (a noiseless, exactly-determined fit).
        let cam = rpc();
        let truth = RpcAdjustment {
            row: [12.5, 0.003, -0.002],
            col: [-8.0, -0.0015, 0.004],
        };
        let gcps: Vec<GroundControlPoint> = grid(&cam)
            .into_iter()
            .map(|(lat, lon, h)| {
                let (row, col) = forward_adjusted(&cam, lat, lon, h, &truth);
                GroundControlPoint {
                    latitude: lat,
                    longitude: lon,
                    height: h,
                    row,
                    col,
                }
            })
            .collect();
        let est = estimate_adjustment(&cam, &gcps, AdjustmentModel::Affine);
        for k in 0..3 {
            assert!(
                (est.row[k] - truth.row[k]).abs() < 1e-9,
                "row[{k}] {}",
                est.row[k]
            );
            assert!(
                (est.col[k] - truth.col[k]).abs() < 1e-9,
                "col[{k}] {}",
                est.col[k]
            );
        }
        // The corrected projection reproduces the observations, and the residual is ~0.
        assert!(adjustment_residual_rms(&cam, &gcps, &est) < 1e-6);
    }

    #[test]
    fn matches_the_numpy_lstsq_oracle_on_noisy_control() {
        // The same grid, but observations carry an injected affine (a=[12.5,0.003,-0.002],
        // b=[-8,-0.0015,0.004]) plus σ=0.3 px Gaussian noise (seed 20240721). The affine below is what
        // numpy.linalg.lstsq recovers on design [1, col_rpc, row_rpc]; our normal-equations solver must
        // match it. (Table + oracle generated by /tmp/rpc_ba_oracle.py.)
        let cam = rpc();
        let obs = [
            (
                -21.28631647997442,
                55.6528486828995,
                1295.0,
                12400.837275468053,
                1021.5213409542763,
            ),
            (
                -21.28631647997442,
                55.6528486828995,
                1695.0,
                12518.941943525564,
                1056.90264266156,
            ),
            (
                -21.28631647997442,
                55.7119698801,
                1295.0,
                12313.645344964667,
                13099.472313626344,
            ),
            (
                -21.28631647997442,
                55.7119698801,
                1695.0,
                12431.49524018634,
                13141.25038081396,
            ),
            (
                -21.28631647997442,
                55.7710910773005,
                1295.0,
                12229.619691124359,
                25165.091783917323,
            ),
            (
                -21.28631647997442,
                55.7710910773005,
                1695.0,
                12347.049597406654,
                25214.96613661118,
            ),
            (
                -21.2316081288,
                55.6528486828995,
                1295.0,
                437.54229728091536,
                946.8474396051475,
            ),
            (
                -21.2316081288,
                55.6528486828995,
                1695.0,
                555.4961913712183,
                981.0341141698536,
            ),
            (
                -21.2316081288,
                55.7119698801,
                1295.0,
                364.8777315254122,
                13031.856605223658,
            ),
            (
                -21.2316081288,
                55.7119698801,
                1695.0,
                482.2502692899697,
                13073.224979687784,
            ),
            (
                -21.2316081288,
                55.7710910773005,
                1295.0,
                294.29834317242813,
                25106.317513907892,
            ),
            (
                -21.2316081288,
                55.7710910773005,
                1695.0,
                412.3257963079194,
                25154.769046845948,
            ),
            (
                -21.176899777625582,
                55.6528486828995,
                1295.0,
                -11534.186442938746,
                881.1176084672879,
            ),
            (
                -21.176899777625582,
                55.6528486828995,
                1695.0,
                -11415.856485029313,
                913.0404716067032,
            ),
            (
                -21.176899777625582,
                55.7119698801,
                1295.0,
                -11593.917864070076,
                12973.789096881485,
            ),
            (
                -21.176899777625582,
                55.7119698801,
                1695.0,
                -11475.986939252602,
                13013.103273964638,
            ),
            (
                -21.176899777625582,
                55.7710910773005,
                1295.0,
                -11649.790091655716,
                25055.49946251908,
            ),
            (
                -21.176899777625582,
                55.7710910773005,
                1695.0,
                -11532.448437330917,
                25103.52131935273,
            ),
        ];
        let gcps: Vec<GroundControlPoint> = obs
            .iter()
            .map(
                |&(latitude, longitude, height, row, col)| GroundControlPoint {
                    latitude,
                    longitude,
                    height,
                    row,
                    col,
                },
            )
            .collect();
        let est = estimate_adjustment(&cam, &gcps, AdjustmentModel::Affine);
        let row_oracle = [
            12.468335756690482,
            0.0030012060651458863,
            -0.0019994112889335316,
        ];
        let col_oracle = [
            -8.097501643939532,
            -0.0015020246786141061,
            0.003999865799708609,
        ];
        for k in 0..3 {
            assert!(
                (est.row[k] - row_oracle[k]).abs() < 1e-6,
                "row[{k}] {} vs numpy",
                est.row[k]
            );
            assert!(
                (est.col[k] - col_oracle[k]).abs() < 1e-6,
                "col[{k}] {} vs numpy",
                est.col[k]
            );
        }
    }

    #[test]
    fn shift_model_takes_out_a_pure_offset_and_cuts_the_residual() {
        let cam = rpc();
        let truth = RpcAdjustment {
            row: [37.0, 0.0, 0.0],
            col: [-21.0, 0.0, 0.0],
        };
        let gcps: Vec<GroundControlPoint> = grid(&cam)
            .into_iter()
            .map(|(lat, lon, h)| {
                let (row, col) = forward_adjusted(&cam, lat, lon, h, &truth);
                GroundControlPoint {
                    latitude: lat,
                    longitude: lon,
                    height: h,
                    row,
                    col,
                }
            })
            .collect();
        let before = adjustment_residual_rms(&cam, &gcps, &RpcAdjustment::IDENTITY);
        let est = estimate_adjustment(&cam, &gcps, AdjustmentModel::Shift);
        assert!((est.row[0] - 37.0).abs() < 1e-9 && (est.col[0] + 21.0).abs() < 1e-9);
        let after = adjustment_residual_rms(&cam, &gcps, &est);
        assert!(
            before > 40.0 && after < 1e-6,
            "residual {before} -> {after}"
        );
    }

    /// Solve `A x = b` by Cramer's rule — an independent method, not a second elimination.
    fn cramer3(a: [f64; 9], b: [f64; 3]) -> [f64; 3] {
        let det = |m: [f64; 9]| {
            m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
                + m[2] * (m[3] * m[7] - m[4] * m[6])
        };
        let d = det(a);
        let mut x = [0.0; 3];
        for (c, xc) in x.iter_mut().enumerate() {
            let mut m = a;
            for r in 0..3 {
                m[r * 3 + c] = b[r];
            }
            *xc = det(m) / d;
        }
        x
    }

    #[test]
    fn solve_sym3_matches_cramers_rule_and_pivots_where_it_has_to() {
        // Only reached through `estimate_adjustment`, whose normal matrices are well conditioned and
        // already ordered — so the pivot search, the singularity test and the row swap were all
        // exercised only on inputs where doing nothing would have worked just as well.
        //
        // A zero on the diagonal: the solver must find the pivot below it rather than refuse.
        let a = [0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0, 5.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve_sym3(a, b).expect("non-singular");
        let want = cramer3(a, b);
        for k in 0..3 {
            assert!(
                (x[k] - want[k]).abs() < 1e-12,
                "x[{k}] {} vs {}",
                x[k],
                want[k]
            );
        }
        // A first column whose largest entry is not the one the diagonal would offer, with the
        // diagonal itself zero — the search has to compare against the running best, not a fixed cell.
        let a = [0.0, 100.0, 2.0, 100.0, 4.0, 1.0, 2.0, 1.0, 7.0];
        let b = [5.0, 1.0, 2.0];
        let x = solve_sym3(a, b).expect("non-singular");
        let want = cramer3(a, b);
        for k in 0..3 {
            assert!(
                (x[k] - want[k]).abs() < 1e-9,
                "x[{k}] {} vs {}",
                x[k],
                want[k]
            );
        }
        // Why the pivot is chosen by modulus and not by convenience. Eliminating on the 1e-16 in the
        // corner subtracts 1e16 times row 0 from row 1, which annihilates every digit of both, and
        // the answer for x0 comes back 0 instead of 1. Partial pivoting is the whole difference.
        let a = [1e-16, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b = [1.0, 2.0, 3.0];
        let x = solve_sym3(a, b).expect("non-singular");
        let want = cramer3(a, b);
        for k in 0..3 {
            assert!(
                (x[k] - want[k]).abs() < 1e-9,
                "x[{k}] {} vs {}",
                x[k],
                want[k]
            );
        }
        // The same point again, but decided by the *running* best rather than by the diagonal: the
        // first column is (0, 1, 1e-16), so a search that keeps comparing against the zero it started
        // from walks past the good pivot and lands on the 1e-16 below it.
        let a = [0.0, 1.0, 1e-16, 1.0, 1.0, 0.0, 1e-16, 0.0, 1.0];
        let x = solve_sym3(a, b).expect("non-singular");
        let want = cramer3(a, b);
        for k in 0..3 {
            assert!(
                (x[k] - want[k]).abs() < 1e-9,
                "x[{k}] {} vs {}",
                x[k],
                want[k]
            );
        }
        // Rank one: every row is a multiple of the first, so the second column has no pivot at all.
        assert!(solve_sym3([2.0, 4.0, 6.0, 4.0, 8.0, 12.0, 6.0, 12.0, 18.0], b).is_none());
        // Rank two: the failure is in the *last* column, after two eliminations have already run.
        assert!(solve_sym3([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0], b).is_none());
    }

    #[test]
    fn fit_axis_falls_back_to_the_target_mean_when_the_normal_matrix_is_singular() {
        // The documented fallback. Two identical design rows make `AᵀA` rank one, so `solve_sym3`
        // refuses and the fit degenerates to an offset-only model at the mean of the targets.
        let design = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]];
        let axis = fit_axis(&design, &[4.0, 10.0]);
        assert!((axis[0] - 7.0).abs() < 1e-12, "{axis:?}");
        assert!(axis[1] == 0.0 && axis[2] == 0.0, "{axis:?}");
    }

    #[test]
    fn the_residual_rms_is_the_root_mean_square_of_the_two_pixel_offsets() {
        // Every existing caller measures this at ~0 or against a loose `> 40`, so neither the sum of
        // the two squared components nor the division by the point count is observable. Displace two
        // GCPs from their own projections by offsets chosen so that the sum and the product of the
        // squares differ, and so does the mean from the total.
        let cam = rpc();
        let offsets = [(3.0, 4.0), (5.0, 12.0)];
        let gcps: Vec<GroundControlPoint> = grid(&cam)
            .into_iter()
            .take(2)
            .zip(offsets)
            .map(|((lat, lon, h), (dr, dc))| {
                let (row, col) = cam.forward(lat, lon, h);
                GroundControlPoint {
                    latitude: lat,
                    longitude: lon,
                    height: h,
                    row: row + dr,
                    col: col + dc,
                }
            })
            .collect();
        // √((3² + 4²) + (5² + 12²)) / 2) = √(25 + 169)/2) = √97.
        let got = adjustment_residual_rms(&cam, &gcps, &RpcAdjustment::IDENTITY);
        assert!((got - libm::sqrt(97.0)).abs() < 1e-9, "{got}");
    }
}
