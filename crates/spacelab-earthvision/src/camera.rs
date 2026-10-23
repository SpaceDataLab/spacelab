//! Frame-camera photogrammetry — the multi-view-geometry backbone: pinhole projection, space
//! resection, space intersection, and sparse bundle adjustment with the Schur-complement reduced
//! camera system.
//!
//! A [`PinholeCamera`] projects a world point `X` to a pixel by the collinearity equations: camera
//! coordinates `x_c = R (X − C)` (rotation `R` world→camera, centre `C` in world), then
//! `u = f·x_c/z_c + c_x`, `v = f·y_c/z_c + c_y`. This is the same `x = K[R | −R C]X` model OpenCV and
//! photogrammetry texts use (with `t = −R C`).
//!
//! - [`resection`] — recover a camera's pose (exterior orientation) from ground control points by
//!   Levenberg–Marquardt on the reprojection residuals, with analytic Jacobians (`R ← exp(δω)·R`,
//!   `C ← C + δC`).
//! - [`triangulate`] — recover a 3-D point from ≥2 rays: a linear DLT seed refined by Gauss–Newton.
//! - [`bundle_adjust`] — jointly refine all poses and all points. The normal equations have the
//!   photogrammetric arrowhead structure; the point block is block-diagonal, so a **Schur complement**
//!   eliminates it into the **reduced camera system** `S Δc = b` (dense over the cameras only), solved,
//!   then back-substituted for the point updates. LM damping on the diagonal.
//!
//! The analytic pose/point Jacobians are checked against finite differences, and the whole stack is
//! verified end-to-end by recovering a known synthetic bundle and cross-checked against OpenCV.

// Index loops read most clearly for these small fixed-size blocks; single/short names mirror the
// photogrammetric algebra (U, V, W, ε, δc, δp).
#![allow(
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]

use alloc::vec;
use alloc::vec::Vec;
use spacelab_core::{Mat3, Vec3};

/// Square, `no_std`-friendly (`f64::powi` is std-only).
#[inline]
fn sq(x: f64) -> f64 {
    x * x
}

// ---------------------------------------------------------------------------------------------------
// Small dense linear algebra (fixed 3×3 plus a general Gauss–Jordan solver for the reduced system).
// ---------------------------------------------------------------------------------------------------

/// Skew-symmetric `[v]×` such that `[v]× w = v × w`.
fn hat(v: Vec3) -> Mat3 {
    Mat3::from_rows([[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]])
}

fn m_add(a: Mat3, b: Mat3) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a.rows[i][j] + b.rows[i][j];
        }
    }
    Mat3::from_rows(r)
}

fn m_scale(a: Mat3, k: f64) -> Mat3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a.rows[i][j] * k;
        }
    }
    Mat3::from_rows(r)
}

/// The `SO(3)` exponential (Rodrigues' formula): axis-angle 3-vector → rotation matrix.
fn so3_exp(w: Vec3) -> Mat3 {
    let theta = w.norm();
    if theta < 1e-12 {
        return m_add(Mat3::IDENTITY, hat(w)); // first-order; the quadratic term is O(θ²)
    }
    let k = w.scale(1.0 / theta);
    let kx = hat(k);
    let kx2 = kx * kx;
    let (s, c) = (libm::sin(theta), libm::cos(theta));
    m_add(Mat3::IDENTITY, m_add(m_scale(kx, s), m_scale(kx2, 1.0 - c)))
}

/// Inverse of a flat row-major 3×3, or `None` if singular.
fn inv3(m: &[f64; 9]) -> Option<[f64; 9]> {
    let c00 = m[4] * m[8] - m[5] * m[7];
    let c01 = m[5] * m[6] - m[3] * m[8];
    let c02 = m[3] * m[7] - m[4] * m[6];
    let det = m[0] * c00 + m[1] * c01 + m[2] * c02;
    if libm::fabs(det) < 1e-300 {
        return None;
    }
    let id = 1.0 / det;
    Some([
        c00 * id,
        (m[2] * m[7] - m[1] * m[8]) * id,
        (m[1] * m[5] - m[2] * m[4]) * id,
        c01 * id,
        (m[0] * m[8] - m[2] * m[6]) * id,
        (m[2] * m[3] - m[0] * m[5]) * id,
        c02 * id,
        (m[1] * m[6] - m[0] * m[7]) * id,
        (m[0] * m[4] - m[1] * m[3]) * id,
    ])
}

/// Solve `A x = b` for an `n×n` row-major `A` by Gauss–Jordan with partial pivoting. `None` if `A`
/// is singular, or holds a value that is not finite.
///
/// Singular means singular *relative to the size of `A`*. An absolute pivot floor is not a rank
/// test, and the margin is not academic: with the 1e-14 floor this used to carry, the exactly
/// rank-deficient 3×3 in `solve_dense_returns_the_solution_it_was_given_and_refuses_a_singular_system`
/// is refused at its own scale but *answered* from a largest entry of 144 upward — the rounding
/// left in the vanished pivot outgrows the floor, and every scale a resection or a Schur step
/// actually works at is far above that. The same floor refuses a perfectly conditioned system once
/// its units are metres rather than millimetres. The tolerance below is `n · ε · max|aᵢⱼ|`, which
/// is scale-free.
fn solve_dense(n: usize, a: &[f64], b: &[f64]) -> Option<Vec<f64>> {
    let mut scale = 0.0_f64;
    for v in a {
        if !v.is_finite() {
            return None;
        }
        scale = scale.max(libm::fabs(*v));
    }
    if scale == 0.0 {
        return None;
    }
    let tol = n as f64 * f64::EPSILON * scale;
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..n {
        let mut piv = col;
        let mut best = libm::fabs(m[col * n + col]);
        for r in (col + 1)..n {
            let v = libm::fabs(m[r * n + col]);
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < tol {
            return None;
        }
        if piv != col {
            for c in 0..n {
                m.swap(col * n + c, piv * n + c);
            }
            x.swap(col, piv);
        }
        let d = m[col * n + col];
        for r in 0..n {
            if r != col {
                let f = m[r * n + col] / d;
                if f != 0.0 {
                    for c in col..n {
                        m[r * n + c] -= f * m[col * n + c];
                    }
                    x[r] -= f * x[col];
                }
            }
        }
    }
    for i in 0..n {
        x[i] /= m[i * n + i];
    }
    Some(x)
}

// ---------------------------------------------------------------------------------------------------
// Pinhole camera and projection Jacobians.
// ---------------------------------------------------------------------------------------------------

/// A calibrated pinhole (frame) camera: rotation `R` (world→camera), centre `C` (world), focal length
/// and principal point (pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PinholeCamera {
    /// Rotation world→camera.
    pub rotation: Mat3,
    /// Camera centre in world coordinates.
    pub center: Vec3,
    /// Focal length (pixels).
    pub focal: f64,
    /// Principal point `[c_x, c_y]` (pixels).
    pub principal: [f64; 2],
}

impl PinholeCamera {
    /// Camera-frame coordinates of a world point, `x_c = R (X − C)`.
    #[must_use]
    pub fn camera_coordinates(&self, world: Vec3) -> Vec3 {
        self.rotation * (world - self.center)
    }

    /// Project a world point to a pixel `(u, v)` — the collinearity equations.
    #[must_use]
    pub fn project(&self, world: Vec3) -> [f64; 2] {
        let xc = self.camera_coordinates(world);
        [
            self.focal * xc.x / xc.z + self.principal[0],
            self.focal * xc.y / xc.z + self.principal[1],
        ]
    }
}

/// One image observation of a point by a camera, referenced by index into the bundle's arrays.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    /// Index of the observing camera.
    pub camera: usize,
    /// Index of the observed point.
    pub point: usize,
    /// Measured pixel `[u, v]`.
    pub pixel: [f64; 2],
}

/// `∂(u, v)/∂x_c` (2×3) at camera coordinates `xc` — the perspective-division Jacobian.
fn d_pixel_d_cam(xc: Vec3, focal: f64) -> [[f64; 3]; 2] {
    let iz = 1.0 / xc.z;
    [
        [focal * iz, 0.0, -focal * xc.x * iz * iz],
        [0.0, focal * iz, -focal * xc.y * iz * iz],
    ]
}

/// Multiply a 2×3 by a `Mat3`, giving 2×3.
fn mul_2x3_mat3(a: &[[f64; 3]; 2], m: Mat3) -> [[f64; 3]; 2] {
    let mut r = [[0.0; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            r[i][j] = a[i][0] * m.rows[0][j] + a[i][1] * m.rows[1][j] + a[i][2] * m.rows[2][j];
        }
    }
    r
}

/// Analytic Jacobians of the reprojection at `world`: `a` (2×6, w.r.t. pose `[δω, δC]` with
/// `R ← exp(δω)R`, `C ← C + δC`), `b` (2×3, w.r.t. the world point), and the predicted pixel.
fn reprojection_jacobians(
    cam: &PinholeCamera,
    world: Vec3,
) -> ([[f64; 6]; 2], [[f64; 3]; 2], [f64; 2]) {
    let xc = cam.camera_coordinates(world);
    let jxc = d_pixel_d_cam(xc, cam.focal);
    let predicted = [
        cam.focal * xc.x / xc.z + cam.principal[0],
        cam.focal * xc.y / xc.z + cam.principal[1],
    ];
    // Pose: ∂x_c/∂δω = −[x_c]×  (left perturbation), ∂x_c/∂δC = −R.
    let a_rot = mul_2x3_mat3(&jxc, m_scale(hat(xc), -1.0));
    let a_cen = mul_2x3_mat3(&jxc, m_scale(cam.rotation, -1.0));
    let mut a = [[0.0; 6]; 2];
    for i in 0..2 {
        for j in 0..3 {
            a[i][j] = a_rot[i][j];
            a[i][j + 3] = a_cen[i][j];
        }
    }
    // Point: ∂x_c/∂X = R.
    let b = mul_2x3_mat3(&jxc, cam.rotation);
    (a, b, predicted)
}

fn apply_pose_step(cam: &PinholeCamera, delta: &[f64]) -> PinholeCamera {
    let mut c = *cam;
    c.rotation = so3_exp(Vec3::new(delta[0], delta[1], delta[2])) * cam.rotation;
    c.center = cam.center + Vec3::new(delta[3], delta[4], delta[5]);
    c
}

// ---------------------------------------------------------------------------------------------------
// Space resection: pose from ground control points.
// ---------------------------------------------------------------------------------------------------

/// Sum of squared reprojection residuals of `cam` over `(world, pixel)` control points.
fn resection_cost(cam: &PinholeCamera, controls: &[(Vec3, [f64; 2])]) -> f64 {
    let mut s = 0.0;
    for (world, pixel) in controls {
        let p = cam.project(*world);
        s += sq(p[0] - pixel[0]) + sq(p[1] - pixel[1]);
    }
    s
}

/// Recover a camera's exterior orientation (rotation + centre) from ground control points
/// `(world, pixel)` by Levenberg–Marquardt, starting from `initial` (intrinsics held fixed). Needs
/// ≥3 non-collinear controls.
#[must_use]
pub fn resection(
    controls: &[(Vec3, [f64; 2])],
    initial: PinholeCamera,
    max_iters: u32,
) -> PinholeCamera {
    let mut cam = initial;
    let mut cost = resection_cost(&cam, controls);
    let mut lambda = 1e-3;
    for _ in 0..max_iters {
        let mut h = [0.0f64; 36];
        let mut g = [0.0f64; 6];
        for (world, pixel) in controls {
            let (a, _b, pred) = reprojection_jacobians(&cam, *world);
            let r = [pred[0] - pixel[0], pred[1] - pixel[1]];
            for i in 0..6 {
                for k in 0..6 {
                    h[i * 6 + k] += a[0][i] * a[0][k] + a[1][i] * a[1][k];
                }
                g[i] += a[0][i] * r[0] + a[1][i] * r[1];
            }
        }
        // One LM trial (with a few damping escalations before giving up on the step).
        let mut improved = false;
        for _try in 0..8 {
            let mut hd = h;
            for i in 0..6 {
                hd[i * 6 + i] *= 1.0 + lambda;
            }
            let rhs: [f64; 6] = core::array::from_fn(|i| -g[i]);
            let Some(delta) = solve_dense(6, &hd, &rhs) else {
                lambda *= 10.0;
                continue;
            };
            let trial = apply_pose_step(&cam, &delta);
            let trial_cost = resection_cost(&trial, controls);
            if trial_cost < cost {
                cam = trial;
                cost = trial_cost;
                lambda = (lambda * 0.5).max(1e-12);
                improved = true;
                break;
            }
            lambda *= 10.0;
        }
        if !improved {
            break; // converged or stuck
        }
    }
    cam
}

/// Draw up to `k` distinct indices in `0..n` from a small LCG stream.
fn sample_distinct(seed: &mut u64, n: usize, k: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(k);
    let mut guard = 0;
    while out.len() < k && guard < 1000 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let idx = ((*seed >> 33) as usize) % n;
        if !out.contains(&idx) {
            out.push(idx);
        }
        guard += 1;
    }
    out
}

/// Reprojection error (pixels) of a control point under `cam`.
fn control_error(cam: &PinholeCamera, world: Vec3, pixel: [f64; 2]) -> f64 {
    let p = cam.project(world);
    libm::sqrt(sq(p[0] - pixel[0]) + sq(p[1] - pixel[1]))
}

/// RANSAC pose estimation — the outlier-rejecting front-end for [`resection`]. Over `iters` rounds it
/// resects a random `sample_size`-point subset (from `initial`), scores it by how many controls
/// reproject within `threshold` pixels, keeps the best consensus set, and refits the pose on it.
/// Returns the pose and the inlier indices. `seed` makes the sampling reproducible.
#[must_use]
pub fn ransac_resection(
    controls: &[(Vec3, [f64; 2])],
    initial: PinholeCamera,
    sample_size: usize,
    iters: u32,
    threshold: f64,
    seed: u64,
) -> (PinholeCamera, Vec<usize>) {
    let n = controls.len();
    let mut rng = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut best_cam = initial;
    let mut best_inliers: Vec<usize> = Vec::new();
    for _ in 0..iters {
        let sample = sample_distinct(&mut rng, n, sample_size.max(3));
        if sample.len() < 3 {
            continue;
        }
        let subset: Vec<(Vec3, [f64; 2])> = sample.iter().map(|&i| controls[i]).collect();
        let cam = resection(&subset, initial, 50);
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| control_error(&cam, controls[i].0, controls[i].1) < threshold)
            .collect();
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_cam = cam;
        }
    }
    // Refit the pose on the full consensus set, then recompute the inliers with the refined pose.
    if best_inliers.len() >= 3 {
        let subset: Vec<(Vec3, [f64; 2])> = best_inliers.iter().map(|&i| controls[i]).collect();
        best_cam = resection(&subset, best_cam, 100);
        best_inliers = (0..n)
            .filter(|&i| control_error(&best_cam, controls[i].0, controls[i].1) < threshold)
            .collect();
    }
    (best_cam, best_inliers)
}

// ---------------------------------------------------------------------------------------------------
// Space intersection: a 3-D point from multiple rays.
// ---------------------------------------------------------------------------------------------------

/// Triangulate a world point from ≥2 views `(camera, pixel)` — a linear DLT seed refined by
/// Gauss–Newton on the reprojection error. `None` if the linear system is degenerate (all rays
/// parallel / a single view).
#[must_use]
pub fn triangulate(views: &[(&PinholeCamera, [f64; 2])]) -> Option<Vec3> {
    if views.len() < 2 {
        return None;
    }
    let mut point = dlt_seed(views)?;

    // Gauss–Newton refinement minimising reprojection.
    for _ in 0..10 {
        let mut ntn = [0.0f64; 9];
        let mut ntr = [0.0f64; 3];
        for (cam, pixel) in views {
            let (_a, jb, pred) = reprojection_jacobians(cam, point);
            let r = [pred[0] - pixel[0], pred[1] - pixel[1]];
            for i in 0..3 {
                for k in 0..3 {
                    ntn[i * 3 + k] += jb[0][i] * jb[0][k] + jb[1][i] * jb[1][k];
                }
                ntr[i] += jb[0][i] * r[0] + jb[1][i] * r[1];
            }
        }
        let rhs = [-ntr[0], -ntr[1], -ntr[2]];
        let Some(step) = solve_dense(3, &ntn, &rhs) else {
            break;
        };
        point = point + Vec3::new(step[0], step[1], step[2]);
        if step[0].abs() + step[1].abs() + step[2].abs() < 1e-12 {
            break;
        }
    }
    Some(point)
}

/// The linear (DLT) triangulation of Hartley & Zisserman §12.2, and [`triangulate`]'s starting point.
///
/// Each view contributes two ray constraints, `(r₀ − a r₂)·X = (r₀ − a r₂)·C` and
/// `(r₁ − b r₂)·X = (r₁ − b r₂)·C`, with the normalised image coordinates `a = (u − cₓ)/f` and
/// `b = (v − c_y)/f`; the `2n × 3` system is solved through its 3×3 normal equations. For exact
/// correspondences the rays meet and the system is consistent, so this *is* the point; otherwise it
/// minimises the algebraic error `Σ z²‖â − a‖²` rather than the reprojection error, which is what
/// the Gauss–Newton pass above then corrects.
fn dlt_seed(views: &[(&PinholeCamera, [f64; 2])]) -> Option<Vec3> {
    let mut ata = [0.0f64; 9];
    let mut atb = [0.0f64; 3];
    for (cam, pixel) in views {
        let rows = cam.rotation.rows;
        let a = (pixel[0] - cam.principal[0]) / cam.focal;
        let b = (pixel[1] - cam.principal[1]) / cam.focal;
        for (coeff, row) in [(a, 0usize), (b, 1usize)] {
            let n = [
                rows[row][0] - coeff * rows[2][0],
                rows[row][1] - coeff * rows[2][1],
                rows[row][2] - coeff * rows[2][2],
            ];
            let rhs = n[0] * cam.center.x + n[1] * cam.center.y + n[2] * cam.center.z;
            for i in 0..3 {
                for j in 0..3 {
                    ata[i * 3 + j] += n[i] * n[j];
                }
                atb[i] += n[i] * rhs;
            }
        }
    }
    let seed = solve_dense(3, &ata, &atb)?;
    Some(Vec3::new(seed[0], seed[1], seed[2]))
}

// ---------------------------------------------------------------------------------------------------
// Bundle adjustment: sparse LM with the Schur-complement reduced camera system.
// ---------------------------------------------------------------------------------------------------

/// The residual scale `c` of the Tukey biweight, in pixels: beyond it an observation is an outlier
/// and carries no weight at all.
///
/// A newtype rather than a bare `f64` because this value decides *which observations exist*, not a
/// magnitude. `c ≤ 0` makes `e >= c` true for every residual, so every weight is zero, the normal
/// equations are the zero matrix and the adjustment returns whatever it started with — a fit that
/// used no data, reported as an ordinary RMS. `c = NaN` is worse in the other direction: `e >= NaN`
/// is false, so every observation takes the branch that divides by it and the whole solve is NaN.
/// Neither state is constructible here, and `c > 0.0` is already false for `NaN`.
///
/// There is no upper bound: a scale far above every residual is a legitimate request for plain
/// least squares, which is what the weights then reduce to.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OutlierScale(f64);

impl OutlierScale {
    /// `None` unless the scale is finite and strictly positive — the exact condition under which
    /// the biweight's two branches both mean what they say.
    #[must_use]
    pub fn new(pixels: f64) -> Option<Self> {
        (pixels > 0.0 && pixels.is_finite()).then_some(Self(pixels))
    }

    /// The scale in pixels.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// The Tukey biweight `w(e) = (1 − (e/c)²)²` for `e ≤ c`, else `0` — the IRLS reweighting of the
/// redescending Tukey M-estimator. Unlike Huber's `δ/e` tail, an observation beyond the scale `c` gets
/// **zero** weight, so a gross outlier exerts no pull and cannot capture a point (given a near-inlier
/// start, where the true correspondences sit well inside `c`).
fn tukey_weight(e: f64, c: f64) -> f64 {
    if e >= c {
        0.0
    } else {
        let t = 1.0 - (e / c) * (e / c);
        t * t
    }
}

/// Total sum-of-squares reprojection cost over all observations.
fn sq_cost(cameras: &[PinholeCamera], points: &[Vec3], observations: &[Observation]) -> f64 {
    let mut s = 0.0;
    for o in observations {
        let p = cameras[o.camera].project(points[o.point]);
        s += sq(p[0] - o.pixel[0]) + sq(p[1] - o.pixel[1]);
    }
    s
}

/// The IRLS majorizer `Σ wᵢ‖rᵢ‖²` at fixed per-observation weights — the surrogate whose decrease
/// guarantees the robust (Tukey) loss decreases (majorization-minimization), so it, not the robust
/// loss itself, is the line-search objective for a robust step.
fn weighted_sq_cost(
    cameras: &[PinholeCamera],
    points: &[Vec3],
    observations: &[Observation],
    weights: &[f64],
) -> f64 {
    let mut s = 0.0;
    for (o, &w) in observations.iter().zip(weights) {
        let p = cameras[o.camera].project(points[o.point]);
        s += w * (sq(p[0] - o.pixel[0]) + sq(p[1] - o.pixel[1]));
    }
    s
}

/// Per-observation Tukey biweights at the current estimate.
fn robust_weights(
    cameras: &[PinholeCamera],
    points: &[Vec3],
    observations: &[Observation],
    scale: f64,
) -> Vec<f64> {
    observations
        .iter()
        .map(|o| {
            let p = cameras[o.camera].project(points[o.point]);
            tukey_weight(
                libm::sqrt(sq(p[0] - o.pixel[0]) + sq(p[1] - o.pixel[1])),
                scale,
            )
        })
        .collect()
}

/// A camera (6) × point (3) cross block `W = Aᵀ B` for one observation, flat row-major.
type WBlock = (usize, usize, [f64; 18]);

/// Assemble the normal-equation blocks for the current linearisation point.
struct Blocks {
    u: Vec<[f64; 36]>,
    v: Vec<[f64; 9]>,
    ec: Vec<[f64; 6]>,
    ep: Vec<[f64; 3]>,
    w: Vec<WBlock>,
}

fn assemble(
    cameras: &[PinholeCamera],
    points: &[Vec3],
    observations: &[Observation],
    weights: Option<&[f64]>,
) -> Blocks {
    let (n, m) = (cameras.len(), points.len());
    let mut b = Blocks {
        u: vec![[0.0; 36]; n],
        v: vec![[0.0; 9]; m],
        ec: vec![[0.0; 6]; n],
        ep: vec![[0.0; 3]; m],
        w: Vec::with_capacity(observations.len()),
    };
    for (oi, o) in observations.iter().enumerate() {
        let (a, jb, pred) = reprojection_jacobians(&cameras[o.camera], points[o.point]);
        let r = [pred[0] - o.pixel[0], pred[1] - o.pixel[1]];
        // IRLS: each observation enters the weighted normal equations AᵀWA / AᵀWr with its fixed weight.
        let wt = weights.map_or(1.0, |w| w[oi]);
        let uj = &mut b.u[o.camera];
        let ecj = &mut b.ec[o.camera];
        for i in 0..6 {
            for k in 0..6 {
                uj[i * 6 + k] += wt * (a[0][i] * a[0][k] + a[1][i] * a[1][k]);
            }
            ecj[i] += wt * (a[0][i] * r[0] + a[1][i] * r[1]);
        }
        let vi = &mut b.v[o.point];
        let epi = &mut b.ep[o.point];
        for i in 0..3 {
            for k in 0..3 {
                vi[i * 3 + k] += wt * (jb[0][i] * jb[0][k] + jb[1][i] * jb[1][k]);
            }
            epi[i] += wt * (jb[0][i] * r[0] + jb[1][i] * r[1]);
        }
        let mut wij = [0.0; 18];
        for i in 0..6 {
            for k in 0..3 {
                wij[i * 3 + k] = wt * (a[0][i] * jb[0][k] + a[1][i] * jb[1][k]);
            }
        }
        b.w.push((o.camera, o.point, wij));
    }
    b
}

/// `(W · Vinv)` for a 6×3 `W` and 3×3 `Vinv`, giving 6×3 flat.
fn w_vinv(w: &[f64; 18], vinv: &[f64; 9]) -> [f64; 18] {
    let mut r = [0.0; 18];
    for i in 0..6 {
        for k in 0..3 {
            r[i * 3 + k] =
                w[i * 3] * vinv[k] + w[i * 3 + 1] * vinv[3 + k] + w[i * 3 + 2] * vinv[6 + k];
        }
    }
    r
}

/// One damped LM step by the Schur complement; returns updated cameras/points, or `None` if singular.
/// `reduced[j]` is the free camera's slot in the reduced system, or `None` if camera `j` is held fixed
/// (its pose is not updated, but its observations still constrain the points).
fn schur_step(
    cameras: &[PinholeCamera],
    points: &[Vec3],
    blocks: &Blocks,
    reduced: &[Option<usize>],
    n_free: usize,
    lambda: f64,
) -> Option<(Vec<PinholeCamera>, Vec<Vec3>)> {
    let (n, m) = (cameras.len(), points.len());
    // Damped point inverses V_i⁻¹ and, per point, its observing *free* cameras' W blocks (by slot).
    let mut vinv = vec![[0.0f64; 9]; m];
    let mut per_point: Vec<Vec<(usize, [f64; 18])>> = vec![Vec::new(); m];
    for (cam, pt, wij) in &blocks.w {
        if let Some(slot) = reduced[*cam] {
            per_point[*pt].push((slot, *wij));
        }
    }
    for i in 0..m {
        let mut vd = blocks.v[i];
        for d in 0..3 {
            vd[d * 3 + d] *= 1.0 + lambda;
        }
        vinv[i] = inv3(&vd)?;
    }
    // Reduced camera system S (6·n_free square) and RHS, over the free cameras only.
    let dim = 6 * n_free;
    let mut s = vec![0.0f64; dim * dim];
    let mut rhs = vec![0.0f64; dim];
    for j in 0..n {
        let Some(rj) = reduced[j] else { continue };
        let ud = blocks.u[j];
        for i in 0..6 {
            for k in 0..6 {
                let mut val = ud[i * 6 + k];
                if i == k {
                    val *= 1.0 + lambda;
                }
                s[(6 * rj + i) * dim + (6 * rj + k)] = val;
            }
            rhs[6 * rj + i] = -blocks.ec[j][i];
        }
    }
    // Subtract W Vinv Wᵀ and add W Vinv ε_p, per point over its observing free cameras.
    for i in 0..m {
        let vi = &vinv[i];
        let epi = &blocks.ep[i];
        for (rj, wj) in &per_point[i] {
            let wv = w_vinv(wj, vi); // 6×3
            for a in 0..6 {
                rhs[6 * rj + a] +=
                    wv[a * 3] * epi[0] + wv[a * 3 + 1] * epi[1] + wv[a * 3 + 2] * epi[2];
            }
            for (rk, wk) in &per_point[i] {
                for a in 0..6 {
                    for b in 0..6 {
                        let dot = wv[a * 3] * wk[b * 3]
                            + wv[a * 3 + 1] * wk[b * 3 + 1]
                            + wv[a * 3 + 2] * wk[b * 3 + 2];
                        s[(6 * rj + a) * dim + (6 * rk + b)] -= dot;
                    }
                }
            }
        }
    }
    let dc = solve_dense(dim, &s, &rhs)?;

    // Back-substitute point updates: δp_i = Vinv_i (−ε_p_i − Σ_j Wᵀ_ji δc_j), over free j only.
    let mut new_points = points.to_vec();
    for i in 0..m {
        let mut t = [-blocks.ep[i][0], -blocks.ep[i][1], -blocks.ep[i][2]];
        for (rj, wj) in &per_point[i] {
            for c in 0..3 {
                let mut acc = 0.0;
                for a in 0..6 {
                    acc += wj[a * 3 + c] * dc[6 * rj + a];
                }
                t[c] -= acc;
            }
        }
        let vi = &vinv[i];
        let dp = [
            vi[0] * t[0] + vi[1] * t[1] + vi[2] * t[2],
            vi[3] * t[0] + vi[4] * t[1] + vi[5] * t[2],
            vi[6] * t[0] + vi[7] * t[1] + vi[8] * t[2],
        ];
        new_points[i] = new_points[i] + Vec3::new(dp[0], dp[1], dp[2]);
    }

    let mut new_cameras = cameras.to_vec();
    for j in 0..n {
        if let Some(rj) = reduced[j] {
            new_cameras[j] = apply_pose_step(&cameras[j], &dc[6 * rj..6 * rj + 6]);
        }
    }
    Some((new_cameras, new_points))
}

/// Jointly refine camera poses and 3-D points to minimise total reprojection error, in place, by
/// sparse Levenberg–Marquardt with the Schur-complement reduced camera system. Intrinsics are held
/// fixed. `fixed_cameras` are held at their input pose to pin the gauge (a bundle has a free global
/// similarity — hold ≥2 cameras, or 1 camera and the scale, else the absolute solution is undetermined
/// even as the reprojection error goes to zero). Returns the final RMS reprojection error (pixels).
pub fn bundle_adjust(
    cameras: &mut [PinholeCamera],
    points: &mut [Vec3],
    observations: &[Observation],
    fixed_cameras: &[usize],
    max_iters: u32,
) -> f64 {
    bundle_adjust_inner(
        cameras,
        points,
        observations,
        fixed_cameras,
        max_iters,
        None,
    )
}

/// Outlier-robust bundle adjustment: the same sparse-Schur solve, but each observation enters the
/// normal equations under the **redescending Tukey M-estimator** by iteratively-reweighted least
/// squares — a residual beyond `outlier_scale` (pixels) is given **zero** weight, so a gross mismatch
/// (a wrong tie point) is dropped from the fit entirely rather than merely down-weighted. Each
/// iteration freezes the weights and minimises the majorizer `Σ wᵢ‖rᵢ‖²` (majorization-minimization),
/// which descends the robust loss. Because Tukey is non-convex, this **refines a near-inlier
/// initialisation** (e.g. one from [`ransac_resection`]) — the true correspondences must start well
/// inside `outlier_scale`; it is not a basin finder. Returns the final plain RMS reprojection error
/// (pixels).
pub fn bundle_adjust_robust(
    cameras: &mut [PinholeCamera],
    points: &mut [Vec3],
    observations: &[Observation],
    fixed_cameras: &[usize],
    max_iters: u32,
    outlier_scale: OutlierScale,
) -> f64 {
    bundle_adjust_inner(
        cameras,
        points,
        observations,
        fixed_cameras,
        max_iters,
        Some(outlier_scale.get()),
    )
}

fn bundle_adjust_inner(
    cameras: &mut [PinholeCamera],
    points: &mut [Vec3],
    observations: &[Observation],
    fixed_cameras: &[usize],
    max_iters: u32,
    robust_scale: Option<f64>,
) -> f64 {
    let n = cameras.len();
    let mut reduced = vec![None; n];
    let mut n_free = 0;
    for (j, slot) in reduced.iter_mut().enumerate() {
        if !fixed_cameras.contains(&j) {
            *slot = Some(n_free);
            n_free += 1;
        }
    }
    let mut lambda = 1e-3;
    for _ in 0..max_iters {
        // Robust: freeze the Tukey weights at the current estimate; the step then minimises the IRLS
        // majorizer Σwᵢ‖rᵢ‖² (whose descent guarantees the robust loss descends — MM). Plain L² is the
        // same loop with unit weights.
        let weights = robust_scale.map(|c| robust_weights(cameras, points, observations, c));
        let wref = weights.as_deref();
        let cost = match wref {
            Some(w) => weighted_sq_cost(cameras, points, observations, w),
            None => sq_cost(cameras, points, observations),
        };
        let blocks = assemble(cameras, points, observations, wref);
        let mut improved = false;
        for _try in 0..8 {
            if let Some((nc, np)) = schur_step(cameras, points, &blocks, &reduced, n_free, lambda) {
                let trial = match wref {
                    Some(w) => weighted_sq_cost(&nc, &np, observations, w),
                    None => sq_cost(&nc, &np, observations),
                };
                if trial < cost {
                    cameras.copy_from_slice(&nc);
                    points.copy_from_slice(&np);
                    lambda = (lambda * 0.5).max(1e-12);
                    improved = true;
                    break;
                }
            }
            lambda *= 10.0;
        }
        if !improved {
            break;
        }
    }
    libm::sqrt(sq_cost(cameras, points, observations) / (observations.len() as f64))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::cast_precision_loss,
        clippy::unreadable_literal
    )]
    use super::*;

    fn cam(rot: Mat3, center: Vec3) -> PinholeCamera {
        PinholeCamera {
            rotation: rot,
            center,
            focal: 1000.0,
            principal: [640.0, 480.0],
        }
    }

    /// Three views whose rotations do not share an axis.
    ///
    /// A triple built from `Mat3::rot2` alone has `rows[2][1] == 0` in every camera, which makes
    /// the DLT ray normal's middle term `rows[row][1] - a * 0` and the sign in front of the
    /// coefficient unobservable -- two mutants of the normal and the right-hand side lived in that
    /// zero. Composing all three axes and offsetting every centre in all three coordinates leaves
    /// no structural zero for a mutation to hide behind.
    fn skew_views() -> [PinholeCamera; 3] {
        [
            cam(
                Mat3::rot1(0.08) * Mat3::rot3(-0.11),
                Vec3::new(0.0, 0.0, 0.0),
            ),
            cam(
                Mat3::rot3(0.17) * Mat3::rot2(0.3) * Mat3::rot1(0.12),
                Vec3::new(5.0, 0.75, -0.5),
            ),
            cam(
                Mat3::rot2(-0.2) * Mat3::rot1(-0.25) * Mat3::rot3(0.09),
                Vec3::new(-4.0, 1.0, 0.6),
            ),
        ]
    }

    // A deterministic pseudo-random sequence (no std rng in no_std tests).
    fn lcg(seed: &mut u64) -> f64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f64) / ((1u64 << 31) as f64) - 1.0 // ~[-1, 1)
    }

    #[test]
    fn analytic_jacobians_match_finite_differences() {
        let c = cam(
            Mat3::rot3(0.2) * Mat3::rot1(-0.3),
            Vec3::new(1.0, -2.0, 10.0),
        );
        let x = Vec3::new(0.5, 0.7, 1.5);
        let (a, b, _p) = reprojection_jacobians(&c, x);
        let eps = 1e-6;
        // Point Jacobian B vs central differences.
        for j in 0..3 {
            let mut xp = x;
            let mut xm = x;
            match j {
                0 => {
                    xp.x += eps;
                    xm.x -= eps;
                }
                1 => {
                    xp.y += eps;
                    xm.y -= eps;
                }
                _ => {
                    xp.z += eps;
                    xm.z -= eps;
                }
            }
            let pp = c.project(xp);
            let pm = c.project(xm);
            for i in 0..2 {
                let fd = (pp[i] - pm[i]) / (2.0 * eps);
                assert!(
                    (fd - b[i][j]).abs() < 1e-3,
                    "B[{i}][{j}] {} vs {fd}",
                    b[i][j]
                );
            }
        }
        // Pose Jacobian A vs central differences on the 6-vector step.
        for j in 0..6 {
            let mut dp = [0.0; 6];
            let mut dm = [0.0; 6];
            dp[j] = eps;
            dm[j] = -eps;
            let pp = apply_pose_step(&c, &dp).project(x);
            let pm = apply_pose_step(&c, &dm).project(x);
            for i in 0..2 {
                let fd = (pp[i] - pm[i]) / (2.0 * eps);
                assert!(
                    (fd - a[i][j]).abs() < 1e-2,
                    "A[{i}][{j}] {} vs {fd}",
                    a[i][j]
                );
            }
        }
    }

    /// The two reprojection scores are the distances they claim to be.
    ///
    /// Six mutants of `resection_cost` and `control_error` -- the sum of the two axes, and the
    /// residual in each -- survived every pose fixture, because both scores are only ever read
    /// through a minimiser or a threshold: Levenberg--Marquardt derives its direction from the
    /// analytic Jacobian and reads the cost only to accept or reject, and any monotone
    /// mis-measurement of a residual that is heading to zero still accepts the steps that shrink it.
    /// Read directly they are two closed forms over three offsets with no zero and no coincidence
    /// in them.
    #[test]
    fn the_reprojection_scores_are_the_distances_they_claim_to_be() {
        let c = cam(Mat3::rot2(0.2), Vec3::new(1.0, -2.0, 0.5));
        // 3-4-5 and 5-12-13 right triangles, plus a sub-pixel offset: every offset is exact in
        // binary, neither axis is zero anywhere, and no two offsets share a magnitude, so a swapped
        // or fused pair of terms lands on a different number rather than the same one.
        let worlds = [
            Vec3::new(0.5, 1.5, 9.0),
            Vec3::new(-2.0, 0.75, 14.0),
            Vec3::new(3.25, -1.25, 7.5),
        ];
        let offsets = [[3.0, 4.0], [-5.0, 12.0], [0.5, -1.5]];
        let controls: Vec<(Vec3, [f64; 2])> = worlds
            .iter()
            .zip(offsets)
            .map(|(&w, d)| {
                let p = c.project(w);
                (w, [p[0] + d[0], p[1] + d[1]])
            })
            .collect();
        for ((world, pixel), d) in controls.iter().zip(offsets) {
            let want = libm::sqrt(d[0] * d[0] + d[1] * d[1]);
            let got = control_error(&c, *world, *pixel);
            assert!((got - want).abs() < 1e-9, "{got} vs {want}");
        }
        // 25 + 169 + 2.5.
        let want = 196.5;
        let got = resection_cost(&c, &controls);
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }

    /// The RANSAC sampler draws exactly `k` distinct indices, and gives up after a thousand tries.
    ///
    /// Four mutants of it survived -- the sample-size test, the attempt budget and the attempt
    /// counter -- because RANSAC swallows all of them: an extra index only makes a subset resection
    /// better determined, and an attempt budget is invisible unless it binds, which it does only
    /// when `k` distinct indices do not exist. The budget is checkable without restating the
    /// generator: a call that cannot succeed must draw exactly as many times as a thousand calls
    /// that succeed on their first draw, so the two leave the same seed behind.
    #[test]
    fn the_sampler_draws_k_distinct_indices_and_gives_up_after_a_thousand_tries() {
        let mut seed = 7u64;
        for k in [1usize, 3, 5] {
            let s = sample_distinct(&mut seed, 20, k);
            assert_eq!(s.len(), k, "{s:?}");
            for (i, a) in s.iter().enumerate() {
                assert!(*a < 20);
                assert!(!s[..i].contains(a), "repeat in {s:?}");
            }
        }
        // Ten distinct indices do not exist in `0..3`, so the loop runs its whole budget and
        // returns what it has.
        let mut exhausted = 12_345u64;
        let all = sample_distinct(&mut exhausted, 3, 10);
        assert_eq!(all.len(), 3, "{all:?}");
        let mut counted = 12_345u64;
        for _ in 0..1000 {
            assert_eq!(sample_distinct(&mut counted, 1_000_000, 1).len(), 1);
        }
        assert_eq!(
            exhausted, counted,
            "the budget is 1000 draws, not {counted:x} worth"
        );
    }

    /// The three bundle scores are what they claim to be, and the report a run returns is the RMS
    /// of the residuals it ends on.
    ///
    /// Ten mutants across `sq_cost`, `weighted_sq_cost`, `robust_weights` and the returned figure
    /// survived, for the reason every cost in this file survives: Levenberg--Marquardt takes its
    /// direction from the analytic Jacobian and reads a cost only to accept or reject a trial, so
    /// any mis-measurement that still shrinks as the residuals shrink accepts the same steps and
    /// converges to the same block. Read directly against hand-set offsets there is nothing to hide
    /// behind. `max_iters = 0` is what makes the report checkable: with no step taken the answer is
    /// the RMS of the offsets that went in.
    // Exactly zero is the claim about the residual past the cut, not approximately zero.
    #[allow(clippy::float_cmp)]
    #[test]
    fn the_bundle_scores_are_the_residuals_they_claim_to_be() {
        let mut cameras = [
            cam(Mat3::IDENTITY, Vec3::new(0.0, 0.0, 0.0)),
            cam(Mat3::rot3(0.2) * Mat3::rot1(0.1), Vec3::new(4.0, -1.0, 0.5)),
        ];
        let mut points = [Vec3::new(0.5, 1.5, 9.0), Vec3::new(-2.0, 0.75, 14.0)];
        // Four right-triangle offsets, all distinct, none on an axis: 5, 13, 25 and 2.5 in norm.
        let offsets = [[3.0, 4.0], [-5.0, 12.0], [7.0, -24.0], [1.5, -2.0]];
        let observations: Vec<Observation> = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .into_iter()
            .zip(offsets)
            .map(|((c, pt), d)| {
                let p = cameras[c].project(points[pt]);
                Observation {
                    camera: c,
                    point: pt,
                    pixel: [p[0] + d[0], p[1] + d[1]],
                }
            })
            .collect();
        let sq: f64 = offsets.iter().map(|d| d[0] * d[0] + d[1] * d[1]).sum();
        let got = sq_cost(&cameras, &points, &observations);
        assert!((got - sq).abs() < 1e-9, "{got} vs {sq}");

        // Weights chosen distinct, none 0 or 1, so a dropped or transposed factor moves the answer.
        let weights = [0.25, 2.5, 0.125, 4.0];
        let want: f64 = offsets
            .iter()
            .zip(weights)
            .map(|(d, w)| w * (d[0] * d[0] + d[1] * d[1]))
            .sum();
        let got = weighted_sq_cost(&cameras, &points, &observations, &weights);
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");

        // A scale between the third and fourth offset norms, so the biweight is asked for two live
        // values, one exact zero, and one just inside the cut.
        let scale = 20.0;
        let w = robust_weights(&cameras, &points, &observations, scale);
        for (got, d) in w.iter().zip(offsets) {
            let want = tukey_weight(libm::sqrt(d[0] * d[0] + d[1] * d[1]), scale);
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        assert_eq!(w[2], 0.0, "the 25-pixel residual is past the 20-pixel cut");
        assert!(w[1] > 0.0 && w[1] < 1.0, "{}", w[1]);

        // No iterations, so the report can only be the RMS of what it was handed.
        let rms = bundle_adjust(&mut cameras, &mut points, &observations, &[0], 0);
        let want = libm::sqrt(sq / 4.0);
        assert!((rms - want).abs() < 1e-9, "{rms} vs {want}");
    }

    /// A 3x3 with no inverse is refused rather than answered in infinities.
    ///
    /// The singularity test's `==` form survived because no fixture ever handed `inv3` a singular
    /// matrix: with `libm::fabs(det) == 1e-300` only a determinant of exactly that magnitude is
    /// refused, and a rank-2 matrix walks through into `1.0 / 0.0` and comes back a table of
    /// infinities and NaNs that the caller cannot tell from an answer.
    #[test]
    fn a_singular_three_by_three_has_no_inverse_to_return() {
        // Row 2 = row 0 + row 1: rank 2, determinant exactly zero.
        assert!(inv3(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 9.0]).is_none());
        // A zero matrix and a rank-1 matrix are the same refusal.
        assert!(inv3(&[0.0; 9]).is_none());
        assert!(inv3(&[1.0, 2.0, 4.0, 2.0, 4.0, 8.0, 3.0, 6.0, 12.0]).is_none());
        // A full-rank neighbour is inverted, and its inverse multiplies back to the identity.
        let m = [2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0];
        let inv = inv3(&m).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let e: f64 = (0..3).map(|k| m[i * 3 + k] * inv[k * 3 + j]).sum();
                let want = f64::from(u8::from(i == j));
                assert!((e - want).abs() < 1e-12, "({i},{j}) {e}");
            }
        }
    }

    // --- The normal-equation blocks of `assemble`, against the cost they are defined by. ---
    //
    // Every mutant of `assemble` that survived did so for the reason a wrong `ntn` survives in
    // `triangulate`: Levenberg--Marquardt reads the blocks only to propose a step, escalates the
    // damping until some step lowers the cost, and converges on the same block from almost any
    // invertible surrogate. So the blocks are checked against what they are defined to be. For the
    // weighted cost `C = sum w_i ||project - pixel||^2`, `ec` and `ep` are exactly `C/2`'s gradient
    // in the pose chart and in the point, for any residual; `u`, `v` and `w` are the Gauss--Newton
    // surrogate `J^T W J`, which equals `C/2`'s Hessian *only where the residual vanishes*, so
    // those are checked on exact correspondences, where the identity is not an approximation.
    //
    // The chart is the one `apply_pose_step` defines -- the same left perturbation the analytic
    // Jacobians are verified in above -- so the difference quotients are taken through it.

    /// Two cameras, two points, four camera-point pairs, and weights that are all distinct and
    /// none of them 0 or 1.
    type BlockFixture = ([PinholeCamera; 2], [Vec3; 2], [f64; 4], [(usize, usize); 4]);

    fn block_fixture() -> BlockFixture {
        let views = skew_views();
        (
            [views[0], views[1]],
            [Vec3::new(0.5, 1.5, 9.0), Vec3::new(-2.0, 0.75, 14.0)],
            [0.25, 2.5, 0.125, 4.0],
            [(0, 0), (0, 1), (1, 0), (1, 1)],
        )
    }

    fn observe(
        cameras: &[PinholeCamera; 2],
        points: &[Vec3; 2],
        pairs: &[(usize, usize); 4],
        offsets: &[[f64; 2]; 4],
    ) -> Vec<Observation> {
        pairs
            .iter()
            .zip(offsets)
            .map(|(&(c, pt), d)| {
                let p = cameras[c].project(points[pt]);
                Observation {
                    camera: c,
                    point: pt,
                    pixel: [p[0] + d[0], p[1] + d[1]],
                }
            })
            .collect()
    }

    fn bump_pose(cameras: &[PinholeCamera; 2], j: usize, i: usize, h: f64) -> [PinholeCamera; 2] {
        let mut out = *cameras;
        let mut d = [0.0; 6];
        d[i] = h;
        out[j] = apply_pose_step(&cameras[j], &d);
        out
    }

    fn move_point(points: &[Vec3; 2], i: usize, k: usize, h: f64) -> [Vec3; 2] {
        let mut out = *points;
        out[i] = out[i]
            + [
                Vec3::new(h, 0.0, 0.0),
                Vec3::new(0.0, h, 0.0),
                Vec3::new(0.0, 0.0, h),
            ][k];
        out
    }

    /// `ec` and `ep` are half the weighted cost's gradient, measured where the residuals are not
    /// zero so that a gradient exists to get wrong.
    #[test]
    fn the_normal_blocks_carry_the_gradient_of_the_weighted_cost() {
        let (cameras, points, weights, pairs) = block_fixture();
        let offsets = [[3.0, 4.0], [-5.0, 12.0], [7.0, -24.0], [1.5, -2.0]];
        let obs = observe(&cameras, &points, &pairs, &offsets);
        let cost = |c: &[PinholeCamera; 2], p: &[Vec3; 2]| weighted_sq_cost(c, p, &obs, &weights);
        let b = assemble(&cameras, &points, &obs, Some(&weights));
        let h = 1e-6;
        for j in 0..2 {
            for i in 0..6 {
                let fd = (cost(&bump_pose(&cameras, j, i, h), &points)
                    - cost(&bump_pose(&cameras, j, i, -h), &points))
                    / (4.0 * h);
                assert!(
                    (fd - b.ec[j][i]).abs() < 1e-4 * (1.0 + fd.abs()),
                    "ec[{j}][{i}] {} vs {fd}",
                    b.ec[j][i]
                );
            }
        }
        for i in 0..2 {
            for k in 0..3 {
                let fd = (cost(&cameras, &move_point(&points, i, k, h))
                    - cost(&cameras, &move_point(&points, i, k, -h)))
                    / (4.0 * h);
                assert!(
                    (fd - b.ep[i][k]).abs() < 1e-4 * (1.0 + fd.abs()),
                    "ep[{i}][{k}] {} vs {fd}",
                    b.ep[i][k]
                );
            }
        }
    }

    /// `u` and `v` are half the weighted cost's Hessian at exact correspondences.
    #[test]
    fn the_normal_blocks_carry_the_hessian_of_the_weighted_cost() {
        let (cameras, points, weights, pairs) = block_fixture();
        let obs = observe(&cameras, &points, &pairs, &[[0.0; 2]; 4]);
        let cost = |c: &[PinholeCamera; 2], p: &[Vec3; 2]| weighted_sq_cost(c, p, &obs, &weights);
        let b = assemble(&cameras, &points, &obs, Some(&weights));
        let h = 1e-4;
        let at_rest = cost(&cameras, &points);
        for j in 0..2 {
            for i in 0..6 {
                for k in 0..6 {
                    let f = |si: f64, sk: f64| {
                        let c = bump_pose(&cameras, j, i, si * h);
                        cost(&bump_pose(&c, j, k, sk * h), &points)
                    };
                    // The two bumps commute only to first order in the chart, so a diagonal entry
                    // is taken as a second difference rather than as a mixed one.
                    let fd = if i == k {
                        (f(1.0, 0.0) - 2.0 * at_rest + f(-1.0, 0.0)) / (2.0 * h * h)
                    } else {
                        (f(1.0, 1.0) - f(1.0, -1.0) - f(-1.0, 1.0) + f(-1.0, -1.0)) / (8.0 * h * h)
                    };
                    let want = b.u[j][i * 6 + k];
                    assert!(
                        (fd - want).abs() < 1e-3 * (1.0 + want.abs()),
                        "u[{j}][{i}][{k}] {want} vs {fd}"
                    );
                }
            }
        }
        for i in 0..2 {
            for a in 0..3 {
                for k in 0..3 {
                    let f = |sa: f64, sk: f64| {
                        let p = move_point(&points, i, a, sa * h);
                        cost(&cameras, &move_point(&p, i, k, sk * h))
                    };
                    let fd = if a == k {
                        (f(1.0, 0.0) - 2.0 * at_rest + f(-1.0, 0.0)) / (2.0 * h * h)
                    } else {
                        (f(1.0, 1.0) - f(1.0, -1.0) - f(-1.0, 1.0) + f(-1.0, -1.0)) / (8.0 * h * h)
                    };
                    let want = b.v[i][a * 3 + k];
                    assert!(
                        (fd - want).abs() < 1e-3 * (1.0 + want.abs()),
                        "v[{i}][{a}][{k}] {want} vs {fd}"
                    );
                }
            }
        }
    }

    /// `w` is the pose-point cross derivative of the single observation it belongs to, and it
    /// carries which camera and which point those are.
    #[test]
    fn the_cross_block_carries_the_mixed_derivative_of_its_own_observation() {
        let (cameras, points, weights, pairs) = block_fixture();
        let obs = observe(&cameras, &points, &pairs, &[[0.0; 2]; 4]);
        let b = assemble(&cameras, &points, &obs, Some(&weights));
        let h = 1e-4;
        assert_eq!(b.w.len(), pairs.len());
        for (oi, &(cj, pi)) in pairs.iter().enumerate() {
            let (got_cam, got_point, wij) = b.w[oi];
            assert_eq!((got_cam, got_point), (cj, pi));
            let one = [obs[oi]];
            let cost = |c: &[PinholeCamera; 2], p: &[Vec3; 2]| {
                weighted_sq_cost(c, p, &one, &weights[oi..=oi])
            };
            for i in 0..6 {
                for k in 0..3 {
                    let f = |si: f64, sk: f64| {
                        cost(
                            &bump_pose(&cameras, cj, i, si * h),
                            &move_point(&points, pi, k, sk * h),
                        )
                    };
                    let fd =
                        (f(1.0, 1.0) - f(1.0, -1.0) - f(-1.0, 1.0) + f(-1.0, -1.0)) / (8.0 * h * h);
                    let want = wij[i * 3 + k];
                    assert!(
                        (fd - want).abs() < 1e-3 * (1.0 + want.abs()),
                        "w[{oi}][{i}][{k}] {want} vs {fd}"
                    );
                }
            }
        }
    }

    /// Levenberg--Marquardt's damping schedule is a convergence *rate*, and a rate is only
    /// observable against a budget.
    ///
    /// Nine mutants of `resection` -- the damping applied to the diagonal, both escalations of
    /// `lambda` and the accept test -- survived because the fixture above asks for 100 iterations
    /// and accepts 1e-6, and every schedule that converges at all converges inside that. The oracle
    /// stays the known pose; what is added is the price. From the same perturbed start the shipped
    /// schedule walks a quadratic ladder -- 4.5e-2, 5.0e-3, 2.9e-4, 8.9e-6, 1.4e-7 metres of centre
    /// error at iterations 4 through 8 -- and reaches the double-precision floor at 12. A schedule
    /// that damps the wrong entries or escalates `lambda` the wrong way still gets there, but not
    /// on this budget.
    #[test]
    fn resection_pays_a_measured_price_for_its_damping_schedule() {
        let truth = cam(
            Mat3::rot3(0.15) * Mat3::rot2(0.1) * Mat3::rot1(-0.2),
            Vec3::new(2.0, -1.0, 12.0),
        );
        let mut seed = 42u64;
        let controls: Vec<(Vec3, [f64; 2])> = (0..12)
            .map(|_| {
                let w = Vec3::new(
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 2.0,
                );
                (w, truth.project(w))
            })
            .collect();
        let ladder = |controls: &[(Vec3, [f64; 2])], init, rungs: &[(u32, f64, f64)]| {
            for &(iters, center, rotation) in rungs {
                let est = resection(controls, init, iters);
                let dc = (est.center - truth.center).norm();
                let dr = m_add(est.rotation, m_scale(truth.rotation, -1.0));
                let rr: f64 = dr.rows.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
                println!("resection {iters:2} iterations: centre {dc:.4e}, rotation {rr:.4e}");
                assert!(dc < center, "{iters} iterations: centre {dc:.4e}");
                assert!(rr < rotation, "{iters} iterations: rotation {rr:.4e}");
            }
        };
        ladder(
            &controls,
            cam(Mat3::rot3(0.0), Vec3::new(0.0, 0.0, 8.0)),
            &[
                (4, 5.0e-2, 5.0e-3),
                (6, 3.5e-4, 3.5e-5),
                (8, 2.0e-7, 2.0e-8),
                (12, 1.0e-13, 1.0e-14),
            ],
        );

        // A start far enough out that Gauss--Newton alone overshoots: from here the damping does
        // escalate -- seven trials are rejected before an accepted step over the run -- and the
        // ladder is the only place in this file where the escalation is on the path to the answer.
        let mut seed = 42u64;
        let six: Vec<(Vec3, [f64; 2])> = (0..6)
            .map(|_| {
                let w = Vec3::new(
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 2.0,
                );
                (w, truth.project(w))
            })
            .collect();
        ladder(
            &six,
            cam(Mat3::rot1(0.5) * Mat3::rot2(0.4), Vec3::new(4.0, 3.0, 5.0)),
            &[
                (6, 1.1e0, 1.2e-1),
                (10, 1.5e-5, 1.6e-6),
                (14, 1.0e-13, 1.0e-14),
            ],
        );
    }

    /// Controls that do not determine a pose drive the normal matrix singular, and the damping is
    /// what gets the solver out of it.
    ///
    /// `resection` documents that it needs three non-collinear controls and has no way to say so
    /// when it does not get them; it returns a pose regardless. Three points on a line leave a
    /// one-parameter family and make `solve_dense` fail outright, which is the only path to the
    /// `lambda *= 10.0` inside the `else` -- unreached by every other fixture in this file, because
    /// on well-posed controls the undamped Gauss--Newton step is accepted on the first trial every
    /// time. Escalating there is what eventually makes `(1 + lambda) * diag` dominant enough to
    /// solve; the opposite move leaves the system singular for all eight trials and the run ends
    /// where it started.
    #[test]
    fn collinear_controls_leave_the_solver_leaning_on_its_damping() {
        let truth = cam(
            Mat3::rot3(0.15) * Mat3::rot2(0.1) * Mat3::rot1(-0.2),
            Vec3::new(2.0, -1.0, 12.0),
        );
        let collinear: Vec<(Vec3, [f64; 2])> = [-1.0, 0.0, 1.0, 2.0]
            .into_iter()
            .map(|t: f64| {
                let w = Vec3::new(0.4 * t, 1.2 * t, 0.5 * t);
                (w, truth.project(w))
            })
            .collect();
        let init = cam(Mat3::rot3(0.1), Vec3::new(0.0, 0.0, 8.0));
        let est = resection(&collinear, init, 20);
        // It terminates, it stays finite, and it does move: a step was taken, so the escalation
        // reached a solvable system.
        assert!(est.center.norm().is_finite(), "{:?}", est.center);
        assert!(
            est.rotation.rows.iter().flatten().all(|x| x.is_finite()),
            "{:?}",
            est.rotation
        );
        assert!(
            (est.center - init.center).norm() > 1e-6,
            "the damping never escalated far enough to take a step"
        );
        // And it is not a pose recovery: the controls do not determine one, so the answer is
        // whatever the line allowed, not the truth. Recorded so a later reader does not mistake the
        // assertion above for a convergence claim.
        assert!(
            (est.center - truth.center).norm() > 1.0,
            "{:?} vs {:?}",
            est.center,
            truth.center
        );
    }

    #[test]
    fn resection_recovers_a_known_pose() {
        let truth = cam(
            Mat3::rot3(0.15) * Mat3::rot2(0.1) * Mat3::rot1(-0.2),
            Vec3::new(2.0, -1.0, 12.0),
        );
        let mut seed = 42u64;
        let controls: Vec<(Vec3, [f64; 2])> = (0..12)
            .map(|_| {
                let w = Vec3::new(
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 2.0,
                );
                (w, truth.project(w))
            })
            .collect();
        // Start from a perturbed pose.
        let init = cam(Mat3::rot3(0.0), Vec3::new(0.0, 0.0, 8.0));
        let est = resection(&controls, init, 100);
        assert!(
            (est.center - truth.center).norm() < 1e-6,
            "center {:?}",
            est.center
        );
        let dr = m_add(est.rotation, m_scale(truth.rotation, -1.0));
        let rerr: f64 = dr.rows.iter().flatten().map(|x| x * x).sum();
        assert!(rerr.sqrt() < 1e-6, "rotation error {}", rerr.sqrt());
    }

    #[test]
    fn triangulate_recovers_a_known_point() {
        let [c1, c2, c3] = skew_views();
        let truth = Vec3::new(1.0, -0.5, 10.0);
        let views = [
            (&c1, c1.project(truth)),
            (&c2, c2.project(truth)),
            (&c3, c3.project(truth)),
        ];
        let est = triangulate(&views).unwrap();
        assert!((est - truth).norm() < 1e-8, "point {est:?} vs {truth:?}");
    }

    /// Three cameras and one point, with the DLT seed held to the geometry instead of to whatever
    /// Gauss–Newton could rescue.
    ///
    /// Every triangulation test above passes with the seed arbitrarily wrong: the refinement
    /// re-linearises from wherever it is handed and reaches the same reprojection minimum from a
    /// start metres away, so 35 mutants of the DLT assembly — the normalised coordinates, the ray
    /// normal, the right-hand side, the outer-product accumulation — outlived all of them. The seed
    /// has its own definition and it is checkable on its own: for exact correspondences the rays
    /// meet, the `2n × 3` system is consistent, and its least-squares solution is the point itself.
    /// The Gauss--Newton pass must land on a stationary point of the *reprojection* cost, which the
    /// seed it starts from is not.
    ///
    /// Twenty-one mutants of the refinement -- the normal-matrix accumulation, the gradient, the
    /// sign of the right-hand side, the update and the break test -- outlived every triangulation
    /// fixture, because every one of them measured rays that meet exactly. There the seed is
    /// already the answer, the first step is at roundoff, and whatever the loop computes it
    /// computes about nothing. Where the rays miss, the seed and the answer are 4.7e-4 apart and
    /// separately checkable, and the cost that separates them is computable from the forward map
    /// alone: the DLT minimises the depth-weighted algebraic error, the refinement minimises
    /// `sum ||project(X) - pixel||^2` in pixels, and the two have different minimisers.
    ///
    /// The oracle is stationarity, not an answer: a central-difference gradient of that cost is
    /// 4.1e1 at the seed and 2.8e-7 at the refined point, so anything that steps in the wrong
    /// direction, weights the normal equations wrongly or stops before the iteration has converged
    /// lands somewhere the gradient still points away from.
    #[test]
    fn the_refinement_reaches_the_reprojection_minimum_the_seed_misses() {
        let [c1, c2, c3] = skew_views();
        let truth = Vec3::new(1.0, -0.5, 10.0);
        let nudge = [[0.31, -0.22], [-0.17, 0.29], [0.24, 0.33]];
        let nudged = |p: [f64; 2], d: [f64; 2]| [p[0] + d[0], p[1] + d[1]];
        let views = [
            (&c1, nudged(c1.project(truth), nudge[0])),
            (&c2, nudged(c2.project(truth), nudge[1])),
            (&c3, nudged(c3.project(truth), nudge[2])),
        ];
        let reproj = |x: Vec3| -> f64 {
            views
                .iter()
                .map(|(camera, pixel)| {
                    let p = camera.project(x);
                    let (du, dv) = (p[0] - pixel[0], p[1] - pixel[1]);
                    du * du + dv * dv
                })
                .sum()
        };
        // Central differences at 1e-5: large enough that the 1e-1-scale cost keeps its significant
        // digits through the subtraction, small enough that the truncation error is below 1e-7.
        let grad_inf = |x: Vec3| -> f64 {
            [
                Vec3::new(1e-5, 0.0, 0.0),
                Vec3::new(0.0, 1e-5, 0.0),
                Vec3::new(0.0, 0.0, 1e-5),
            ]
            .into_iter()
            .map(|e| ((reproj(x + e) - reproj(x - e)) / 2e-5).abs())
            .fold(0.0, f64::max)
        };

        let seed = dlt_seed(&views).unwrap();
        let refined = triangulate(&views).unwrap();
        println!(
            "seed {seed:?} cost {:.9e} |grad| {:.3e}",
            reproj(seed),
            grad_inf(seed)
        );
        println!(
            "refined {refined:?} cost {:.9e} |grad| {:.3e}",
            reproj(refined),
            grad_inf(refined)
        );
        // The fixture is non-degenerate: the seed is genuinely not the reprojection minimiser, so
        // the loop has real work to do and a mutated loop has somewhere wrong to land.
        assert!(grad_inf(seed) > 1.0, "{}", grad_inf(seed));
        assert!((refined - seed).norm() > 1e-6);
        assert!(reproj(refined) < reproj(seed) - 4e-3);
        assert!(grad_inf(refined) < 1e-6, "{}", grad_inf(refined));
        // Stationary and a minimum, not a saddle: below all 26 neighbours of a cube at four scales.
        let at_min = reproj(refined);
        for step in [1e-6, 1e-4, 1e-2, 1.0] {
            for dx in [-1.0, 0.0, 1.0] {
                for dy in [-1.0, 0.0, 1.0] {
                    for dz in [-1.0, 0.0, 1.0] {
                        if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                            continue;
                        }
                        let n = refined + Vec3::new(dx * step, dy * step, dz * step);
                        assert!(
                            reproj(n) >= at_min,
                            "neighbour {n:?} at {step:.0e} beats the refined point: {:.12e} < {at_min:.12e}",
                            reproj(n)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_dlt_seed_is_exact_where_the_rays_meet() {
        let [c1, c2, c3] = skew_views();
        // Points off every axis and off the baseline, so no coordinate of the ray normal is zero and
        // no mutated term can cancel against a zero it did not earn.
        for truth in [
            Vec3::new(1.0, -0.5, 10.0),
            Vec3::new(-3.25, 2.75, 22.5),
            Vec3::new(0.125, 7.5, 6.0),
        ] {
            let views = [
                (&c1, c1.project(truth)),
                (&c2, c2.project(truth)),
                (&c3, c3.project(truth)),
            ];
            let seed = dlt_seed(&views).unwrap();
            let err = (seed - truth).norm();
            println!("dlt seed {seed:?} vs {truth:?}: {err:.3e}");
            assert!(err < 1e-12 * truth.norm(), "seed {seed:?} vs {truth:?}");
        }
        // Two views of the same camera give one ray, not two: the normal matrix is rank 2 and there
        // is no seed to return.
        let degenerate = [(&c1, c1.project(Vec3::new(1.0, -0.5, 10.0))); 2];
        assert!(dlt_seed(&degenerate).is_none());
        assert!(triangulate(&degenerate).is_none());
    }

    /// Where the rays miss, the seed is the minimiser of the cost that defines it.
    ///
    /// With inconsistent pixels there is no point to compare against, so the oracle is the defining
    /// property rather than an answer: the DLT minimises the *algebraic* error `Σ z²‖â − a‖²`, and
    /// that cost is computable from the forward map alone — `z` is the depth and `â − a` the
    /// normalised reprojection difference — without touching the normal equations under test. A
    /// wrongly assembled system solves some other problem, and its solution is then beaten by a
    /// neighbour of the true one.
    #[test]
    fn the_dlt_seed_minimises_the_algebraic_cost_that_defines_it() {
        let [c1, c2, c3] = skew_views();
        let truth = Vec3::new(1.0, -0.5, 10.0);
        // Pixel errors of a third of a pixel, different in every view and both axes, so the three
        // rays are mutually skew and the least-squares solution is not any of them.
        let nudge = [[0.31, -0.22], [-0.17, 0.29], [0.24, 0.33]];
        let nudged = |p: [f64; 2], d: [f64; 2]| [p[0] + d[0], p[1] + d[1]];
        let views = [
            (&c1, nudged(c1.project(truth), nudge[0])),
            (&c2, nudged(c2.project(truth), nudge[1])),
            (&c3, nudged(c3.project(truth), nudge[2])),
        ];
        let cost = |x: Vec3| -> f64 {
            views
                .iter()
                .map(|(camera, pixel)| {
                    let d = x - camera.center;
                    let r2 = camera.rotation.rows[2];
                    let depth = r2[0] * d.x + r2[1] * d.y + r2[2] * d.z;
                    let p = camera.project(x);
                    let du = depth * (p[0] - pixel[0]) / camera.focal;
                    let dv = depth * (p[1] - pixel[1]) / camera.focal;
                    du * du + dv * dv
                })
                .sum()
        };
        let seed = dlt_seed(&views).unwrap();
        let at_seed = cost(seed);
        println!("dlt seed {seed:?}, algebraic cost {at_seed:.6e}");
        // The 26 neighbours of a cube around the seed, at four scales spanning the geometry: a
        // minimiser is below all of them, and a seed solving the wrong system is not.
        for step in [1e-6, 1e-4, 1e-2, 1.0] {
            for dx in [-1.0, 0.0, 1.0] {
                for dy in [-1.0, 0.0, 1.0] {
                    for dz in [-1.0, 0.0, 1.0] {
                        if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                            continue;
                        }
                        let probe = seed + Vec3::new(dx * step, dy * step, dz * step);
                        assert!(
                            cost(probe) > at_seed,
                            "cost {:.17e} at {probe:?} is below the seed's {at_seed:.17e}",
                            cost(probe)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bundle_adjust_recovers_a_synthetic_block() {
        // Three cameras looking at a cloud of points; perturb everything but camera 0 and the scale,
        // then BA must drive the reprojection RMSE to ~0 and recover the structure.
        let truth_cams = [
            cam(Mat3::IDENTITY, Vec3::new(0.0, 0.0, 0.0)),
            cam(Mat3::rot2(0.25), Vec3::new(6.0, 0.0, 0.0)),
            cam(
                Mat3::rot2(-0.25) * Mat3::rot1(0.1),
                Vec3::new(-5.0, 2.0, 0.0),
            ),
        ];
        let mut seed = 7u64;
        let truth_pts: Vec<Vec3> = (0..24)
            .map(|_| {
                Vec3::new(
                    lcg(&mut seed) * 4.0,
                    lcg(&mut seed) * 4.0,
                    12.0 + lcg(&mut seed) * 3.0,
                )
            })
            .collect();
        let mut obs = Vec::new();
        for (ci, c) in truth_cams.iter().enumerate() {
            for (pi, p) in truth_pts.iter().enumerate() {
                obs.push(Observation {
                    camera: ci,
                    point: pi,
                    pixel: c.project(*p),
                });
            }
        }
        // Cameras 0 and 1 held fixed at truth pin the gauge (pose + scale); camera 2 and all points are
        // perturbed. BA must drive the reprojection RMSE to ~0 and recover camera 2 and the structure
        // in absolute coordinates.
        let mut cams = truth_cams;
        cams[2].center = cams[2].center + Vec3::new(-0.4, 0.3, -0.2);
        cams[2].rotation = so3_exp(Vec3::new(-0.03, 0.02, -0.02)) * cams[2].rotation;
        let mut pts: Vec<Vec3> = truth_pts
            .iter()
            .map(|p| {
                *p + Vec3::new(
                    lcg(&mut seed) * 0.3,
                    lcg(&mut seed) * 0.3,
                    lcg(&mut seed) * 0.5,
                )
            })
            .collect();

        let rms0 = sq_cost(&cams, &pts, &obs).sqrt();
        let rms = bundle_adjust(&mut cams, &mut pts, &obs, &[0, 1], 80);
        assert!(rms < 1e-6, "final RMS {rms} (from {rms0})");
        // Absolute structure recovered — the two fixed cameras pin pose and scale.
        for (est, truth) in pts.iter().zip(&truth_pts) {
            assert!((*est - *truth).norm() < 1e-4, "point {est:?} vs {truth:?}");
        }
        assert!((cams[2].center - truth_cams[2].center).norm() < 1e-4);
    }

    #[test]
    fn resection_matches_opencv_solve_pnp() {
        // 8 world points and their pixels; the pose below is what cv2.solvePnP (SOLVEPNP_ITERATIVE)
        // recovers from exactly these on the same intrinsics (f = 1200, principal [640, 512]).
        let obj = [
            Vec3::new(0.5, 0.7, 1.2),
            Vec3::new(-1.0, 0.3, 0.8),
            Vec3::new(2.0, -1.5, 1.9),
            Vec3::new(0.0, 2.0, -0.5),
            Vec3::new(-2.0, -1.0, 2.5),
            Vec3::new(1.2, 1.8, 0.2),
            Vec3::new(-0.5, -2.2, 1.0),
            Vec3::new(2.5, 0.5, -1.0),
        ];
        let img = [
            [503.0579620244217, -24.273715017137988],
            [669.1694386496628, 72.58072261042463],
            [274.5979569687516, 232.03644655520145],
            [548.2438276846577, -113.31716686728589],
            [837.38451037408, 233.39753574363277],
            [416.7369698754444, -147.3298956245286],
            [582.8587789028542, 364.3845071208846],
            [250.9230065616827, 21.103951447271584],
        ];
        let controls: Vec<(Vec3, [f64; 2])> = obj.iter().copied().zip(img).collect();
        let init = PinholeCamera {
            rotation: Mat3::IDENTITY,
            center: Vec3::new(0.0, 0.0, 8.0),
            focal: 1200.0,
            principal: [640.0, 512.0],
        };
        let est = resection(&controls, init, 100);

        let r_cv = Mat3::from_rows([
            [
                0.9640360719393384,
                -0.11702620226653215,
                -0.23861961357533779,
            ],
            [
                0.07982213875550284,
                0.9838782391452207,
                -0.16003761027020286,
            ],
            [0.2535012389797495, 0.13523490126285004, 0.9578353946875001],
        ]);
        let c_cv = Vec3::new(1.4999999999999987, -1.9999999999999982, 11.000000000000002);
        let dc = (est.center - c_cv).norm();
        assert!(dc < 1e-6, "resection centre vs cv2 solvePnP: {dc}");
        let rerr: f64 = m_add(est.rotation, m_scale(r_cv, -1.0))
            .rows
            .iter()
            .flatten()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        assert!(rerr < 1e-6, "resection rotation vs cv2 solvePnP: {rerr}");
    }

    #[test]
    fn triangulate_matches_opencv() {
        let intr = |rotation, center| PinholeCamera {
            rotation,
            center,
            focal: 1200.0,
            principal: [640.0, 512.0],
        };
        let c1 = intr(Mat3::IDENTITY, Vec3::new(0.0, 0.0, 0.0));
        let r2 = Mat3::from_rows([
            [0.9393727128473789, 0.0, 0.34289780745545134],
            [0.0, 1.0, 0.0],
            [-0.34289780745545134, 0.0, 0.9393727128473789],
        ]);
        let c2 = intr(r2, Vec3::new(6.0, 0.5, 0.0));
        let views = [
            (&c1, [770.0, 442.0]),
            (&c2, [612.032668958666, 400.23426810775993]),
        ];
        let est = triangulate(&views).unwrap();
        // cv2.triangulatePoints on the same two views.
        let x_cv = Vec3::new(1.300000000000001, -0.6999999999999995, 11.999999999999996);
        let d = (est - x_cv).norm();
        assert!(
            d < 1e-6,
            "triangulate vs cv2: {est:?} vs {x_cv:?} (|Δ| {d})"
        );
    }

    #[test]
    fn ransac_resection_rejects_outlier_controls() {
        // 24 exact controls plus 8 with garbage pixels. Plain resection is dragged off the pose; RANSAC
        // finds the inlier consensus, recovers the true pose, and flags exactly the good controls.
        let truth = cam(
            Mat3::rot3(0.1) * Mat3::rot1(-0.15),
            Vec3::new(1.5, -1.0, 11.0),
        );
        let mut seed = 5u64;
        let mut controls: Vec<(Vec3, [f64; 2])> = (0..24)
            .map(|_| {
                let w = Vec3::new(
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 2.0,
                );
                (w, truth.project(w))
            })
            .collect();
        for _ in 0..8 {
            let w = Vec3::new(
                lcg(&mut seed) * 3.0,
                lcg(&mut seed) * 3.0,
                lcg(&mut seed) * 2.0,
            );
            // A grossly wrong pixel (hundreds of px off the true projection).
            controls.push((
                w,
                [
                    lcg(&mut seed) * 400.0 + 640.0,
                    lcg(&mut seed) * 400.0 + 480.0,
                ],
            ));
        }
        let init = cam(Mat3::rot3(0.0), Vec3::new(0.0, 0.0, 8.0));

        let plain = resection(&controls, init, 100);
        let plain_err = (plain.center - truth.center).norm();
        assert!(
            plain_err > 0.1,
            "plain resection unexpectedly clean: {plain_err}"
        );

        let (rcam, inliers) = ransac_resection(&controls, init, 6, 300, 2.0, 12345);
        assert!(
            (rcam.center - truth.center).norm() < 1e-3,
            "ransac centre {:?} vs {:?}",
            rcam.center,
            truth.center
        );
        assert!(
            (24..=26).contains(&inliers.len()),
            "inlier count {} (expected the 24 good controls)",
            inliers.len()
        );
    }

    #[test]
    fn robust_bundle_adjust_downweights_bad_observations() {
        // The 3-camera block, but a handful of observations carry a gross pixel error. Plain L² BA is
        // pulled off those points; the redescending Tukey robust BA gives them zero weight and recovers
        // the structure exactly.
        let truth_cams = [
            cam(Mat3::IDENTITY, Vec3::new(0.0, 0.0, 0.0)),
            cam(Mat3::rot2(0.25), Vec3::new(6.0, 0.0, 0.0)),
            cam(
                Mat3::rot2(-0.25) * Mat3::rot1(0.1),
                Vec3::new(-5.0, 2.0, 0.0),
            ),
        ];
        let mut seed = 7u64;
        let truth_pts: Vec<Vec3> = (0..24)
            .map(|_| {
                Vec3::new(
                    lcg(&mut seed) * 4.0,
                    lcg(&mut seed) * 4.0,
                    12.0 + lcg(&mut seed) * 3.0,
                )
            })
            .collect();
        let mut obs = Vec::new();
        for (ci, c) in truth_cams.iter().enumerate() {
            for (pi, p) in truth_pts.iter().enumerate() {
                obs.push(Observation {
                    camera: ci,
                    point: pi,
                    pixel: c.project(*p),
                });
            }
        }
        // Corrupt five observations (each of a distinct point, so two clean views remain).
        for &k in &[3usize, 17, 33, 48, 61] {
            obs[k].pixel[0] += 90.0;
            obs[k].pixel[1] -= 70.0;
        }
        // Robust BA runs from a near-truth initialisation (e.g. after RANSAC), where only the true
        // outliers have large residuals — a small perturbation keeps the inliers well inside the Tukey scale.
        let perturbed: Vec<Vec3> = truth_pts
            .iter()
            .map(|p| {
                *p + Vec3::new(
                    lcg(&mut seed) * 0.02,
                    lcg(&mut seed) * 0.02,
                    lcg(&mut seed) * 0.03,
                )
            })
            .collect();

        let mean_err = |pts: &[Vec3]| -> f64 {
            pts.iter()
                .zip(&truth_pts)
                .map(|(a, b)| (*a - *b).norm())
                .sum::<f64>()
                / truth_pts.len() as f64
        };

        let mut cams_l2 = truth_cams;
        let mut pts_l2 = perturbed.clone();
        bundle_adjust(&mut cams_l2, &mut pts_l2, &obs, &[0, 1], 80);
        let err_l2 = mean_err(&pts_l2);

        let mut cams_r = truth_cams;
        let mut pts_r = perturbed;
        let scale = OutlierScale::new(6.0).unwrap();
        bundle_adjust_robust(&mut cams_r, &mut pts_r, &obs, &[0, 1], 100, scale);
        let err_robust = mean_err(&pts_r);

        // Tukey BA rejects the bad observations (zero weight) and recovers the structure; L² is dragged off it.
        assert!(
            err_robust < 1e-4 && err_l2 > 0.05,
            "robust {err_robust} vs L2 {err_l2}"
        );
    }

    /// The scale that would delete every observation, and the one that would poison every weight,
    /// are both refused at construction rather than run.
    ///
    /// `c <= 0` makes `e >= c` true for every residual, so `assemble` builds the zero matrix and the
    /// adjustment reports the RMS it started with as though it had fitted something. `c = NaN` sends
    /// the other branch, and every weight comes back NaN. Both are one keystroke away from a
    /// legitimate call.
    // Exactly zero is the claim about the deleted observations, not approximately zero.
    #[allow(clippy::float_cmp)]
    #[test]
    fn an_outlier_scale_that_deletes_or_poisons_every_observation_is_not_constructible() {
        for bad in [0.0, -6.0, f64::NAN, f64::INFINITY, -0.0] {
            assert!(OutlierScale::new(bad).is_none(), "{bad} was accepted");
        }
        for good in [f64::MIN_POSITIVE, 1e-9, 6.0, 1e30] {
            assert_eq!(OutlierScale::new(good).map(OutlierScale::get), Some(good));
        }
        // What the refused values would have done, measured on the biweight itself rather than
        // asserted: a non-positive scale zeroes a residual that is plainly an inlier, and a NaN one
        // takes the other branch and returns NaN.
        assert_eq!(tukey_weight(0.5, -6.0), 0.0);
        assert_eq!(tukey_weight(0.5, 0.0), 0.0);
        assert!(tukey_weight(0.5, f64::NAN).is_nan());
        // (1 − (1/12)²)² = (143/144)² = 20449/20736. The two-step evaluation lands one ulp above it.
        assert!((tukey_weight(0.5, 6.0) - 20449.0 / 20736.0).abs() <= f64::EPSILON / 2.0);
    }

    /// `solve_dense` is the only linear solver in this module and everything above rests on it, yet
    /// nothing held it to an answer: the bundle and the triangulation both re-solve until their own
    /// residual is small, so a solver that returns a *different* correct-looking step still converges
    /// and hides the mutation. Held here to a solution that is known before the solver runs.
    ///
    /// The three cases are the three code paths. The first needs a pivot swap on the very first
    /// column — `a[0][0]` is zero — which is the only way the swap loop and its `v > best` test are
    /// reached at all. The second is well conditioned and diagonally dominant, so no swap happens and
    /// the elimination is exercised alone. The third is exactly singular and must come back `None`.
    #[test]
    fn solve_dense_returns_the_solution_it_was_given_and_refuses_a_singular_system() {
        // Row-major A, and an x chosen first; b = A·x is then exact in f64 for these small integers,
        // so the recovered x may be compared at 1e-12 without a fitted tolerance.
        let needs_a_swap = [
            0.0, 2.0, 1.0, //
            3.0, -1.0, 4.0, //
            1.0, 5.0, -2.0,
        ];
        let diagonally_dominant = [
            9.0, 1.0, -2.0, //
            2.0, -8.0, 1.0, //
            -1.0, 3.0, 7.0,
        ];
        let x = [1.5, -2.25, 0.75];
        // |a[r][0]| ties at 4.0 between rows 0 and 2, which is the only input on which the pivot
        // search's `v > best` and its `>=` mutant part company. They pick different rows and both
        // are legal partial pivots, so the answer is the same to a rounding: measured here, x[2]
        // comes back one ulp either side of 0.75 (7.4999999999999988898e-1 against
        // 7.5000000000000011102e-1). See the `mutants.toml` entry.
        let ties_on_the_first_column = [
            4.0, 1.0, -2.0, //
            1.0, 6.0, 1.0, //
            -4.0, 2.0, 7.0,
        ];
        for (label, a) in [
            ("pivot swap", needs_a_swap),
            ("no swap", diagonally_dominant),
            ("pivot tie", ties_on_the_first_column),
        ] {
            let b: Vec<f64> = (0..3)
                .map(|r| a[r * 3] * x[0] + a[r * 3 + 1] * x[1] + a[r * 3 + 2] * x[2])
                .collect();
            let got = solve_dense(3, &a, &b).unwrap();
            for i in 0..3 {
                assert!(
                    libm::fabs(got[i] - x[i]) < 1e-12,
                    "{label}: x[{i}] = {}, expected {}",
                    got[i],
                    x[i]
                );
            }
        }
        // Row 2 = row 0 + row 1 exactly, so the third pivot is zero however the rows are ordered.
        let singular = [
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            5.0, 7.0, 9.0,
        ];
        assert!(solve_dense(3, &singular, &[1.0, 2.0, 3.0]).is_none());

        // Rank is a property of the matrix, not of the units it is written in, so whether a system
        // is refused cannot depend on a common factor. The rank-deficient one stays refused and the
        // full-rank one stays solved across twenty-four orders of magnitude, and the solution is
        // the same to the last bit -- an absolute pivot floor has neither property, and at 1e6,
        // which is where the normal equations of a resection in pixels sit, it accepts `singular`.
        let full_rank = [
            0.5, 0.25, 0.125, //
            0.25, 2.0, 0.0625, //
            0.125, 0.0625, 4.0,
        ];
        let reference = solve_dense(3, &full_rank, &[1.0, 2.0, 4.0]).unwrap();
        for exponent in [-12_i32, -6, 0, 6, 12] {
            let k = libm::pow(2.0, f64::from(exponent)); // exact, so scaling introduces no rounding
            let scaled: Vec<f64> = full_rank.iter().map(|v| v * k).collect();
            let rhs: Vec<f64> = [1.0, 2.0, 4.0].iter().map(|v| v * k).collect();
            assert_eq!(
                solve_dense(3, &scaled, &rhs).unwrap(),
                reference,
                "full rank at 2^{exponent}"
            );
            let scaled: Vec<f64> = singular.iter().map(|v| v * k).collect();
            assert!(
                solve_dense(3, &scaled, &rhs).is_none(),
                "rank deficient at 2^{exponent} was solved"
            );
        }

        // An all-zero matrix has no pivot at any scale, and one carrying a NaN has no rank to test.
        assert!(solve_dense(3, &[0.0; 9], &[1.0, 2.0, 3.0]).is_none());
        let mut with_a_nan = full_rank;
        with_a_nan[4] = f64::NAN;
        assert!(solve_dense(3, &with_a_nan, &[1.0, 2.0, 3.0]).is_none());

        // The floor `n · ε · max|aᵢⱼ|` is exclusive, and its boundary is exactly reachable: with
        // n = 2 and max|aᵢⱼ| = 1 it is 2⁻⁵¹, so a second pivot of exactly 2⁻⁵¹ is answered and
        // anything below it is refused. Rounding the comparison the other way would refuse a system
        // whose solution is exactly (1, 1).
        let floor = 2.0 * f64::EPSILON;
        let on_the_floor = solve_dense(2, &[1.0, 0.0, 0.0, floor], &[1.0, floor]).unwrap();
        assert!(
            (on_the_floor[0] - 1.0).abs() < 1e-12 && (on_the_floor[1] - 1.0).abs() < 1e-12,
            "{on_the_floor:?}"
        );
        let below = floor * (1.0 - f64::EPSILON);
        assert!(below < floor);
        assert!(solve_dense(2, &[1.0, 0.0, 0.0, below], &[1.0, floor]).is_none());
    }

    /// The Tukey biweight against its closed form, at points chosen so no two of them agree.
    ///
    /// `robust_weights` feeds this into an IRLS loop and the loop converges to much the same answer
    /// for any weight function that falls off, which is why every arithmetic mutant in these three
    /// lines survived. The values are hand-derived from `w = (1 − (e/c)²)²`, not read off a run:
    /// at `e = c/2` it is `(3/4)² = 9/16`, and at `e = c/√2` exactly `1/4`.
    #[test]
    fn the_tukey_biweight_is_its_closed_form() {
        let c = 4.0;
        for (e, want) in [
            (0.0, 1.0),
            (c / 2.0, 9.0 / 16.0),
            (c / core::f64::consts::SQRT_2, 0.25),
            (c, 0.0),
            (1.5 * c, 0.0),
        ] {
            let got = tukey_weight(e, c);
            assert!(
                libm::fabs(got - want) < 1e-15,
                "w({e}; {c}) = {got}, expected {want}"
            );
        }
        // The scale is a scale: the weight depends on `e/c` alone, so doubling both is a no-op. This
        // is what separates `(e/c)*(e/c)` from the `+`, `-` and `/` mutants of it, each of which
        // gives the right answer at some single `e` but not at the same `e/c` under two scales.
        for ratio in [0.1, 0.3, 0.6, 0.9] {
            let a = tukey_weight(ratio * 2.0, 2.0);
            let b = tukey_weight(ratio * 50.0, 50.0);
            assert!(
                libm::fabs(a - b) < 1e-15,
                "w is not a function of e/c: {a} vs {b}"
            );
        }
    }

    /// The damped normal equations of `blocks`, assembled whole rather than reduced.
    ///
    /// Returns the dimension, the row-major `(6·n_free + 3m)` matrix and its right-hand side.
    /// The same system `schur_step` reduces -- but assembled without ever inverting `V` or
    /// forming `W V⁻¹ Wᵀ`, so a mutant of either is free to show itself.
    fn full_normal_system(
        blocks: &Blocks,
        n_free: usize,
        m: usize,
        lambda: f64,
    ) -> (usize, Vec<f64>, Vec<f64>) {
        let dim = 6 * n_free + 3 * m;
        let mut a = vec![0.0f64; dim * dim];
        let mut rhs = vec![0.0f64; dim];
        for j in 0..n_free {
            for i in 0..6 {
                for k in 0..6 {
                    let mut v = blocks.u[j][i * 6 + k];
                    if i == k {
                        v *= 1.0 + lambda;
                    }
                    a[(6 * j + i) * dim + 6 * j + k] = v;
                }
                rhs[6 * j + i] = -blocks.ec[j][i];
            }
        }
        for i in 0..m {
            for r in 0..3 {
                for k in 0..3 {
                    let mut v = blocks.v[i][r * 3 + k];
                    if r == k {
                        v *= 1.0 + lambda;
                    }
                    a[(6 * n_free + 3 * i + r) * dim + 6 * n_free + 3 * i + k] = v;
                }
                rhs[6 * n_free + 3 * i + r] = -blocks.ep[i][r];
            }
        }
        for (cam, pt, wij) in &blocks.w {
            for i in 0..6 {
                for k in 0..3 {
                    let (row, col) = (6 * cam + i, 6 * n_free + 3 * pt + k);
                    a[row * dim + col] += wij[i * 3 + k];
                    a[col * dim + row] += wij[i * 3 + k];
                }
            }
        }
        (dim, a, rhs)
    }

    /// The Schur complement against a dense factorisation of the same normal equations.
    ///
    /// `schur_step` reduces `[[U, W], [Wᵀ, V]] [δc; δp] = [−ε_c; −ε_p]` to a `6·n_free` system by
    /// eliminating the points, then back-substitutes them. That is exact algebra, so forming the
    /// whole `(6n + 3m)` matrix and handing it to `solve_dense` has to produce the same step — and
    /// it is an independent route, not a re-derivation, because it never computes `V⁻¹`, never forms
    /// `W V⁻¹ Wᵀ`, and eliminates in a different order.
    ///
    /// Nothing above could see any of this: `bundle_adjust` re-linearises and re-damps until its own
    /// cost stops falling, so a step that is merely *a* descent direction still converges. Twenty-four
    /// of this module's survivors are in `schur_step` and five more in `w_vinv` for that reason.
    #[test]
    fn the_schur_step_produces_the_same_update_as_a_dense_solve_of_the_full_system() {
        let cameras = [
            cam(Mat3::rot2(0.15), Vec3::new(-3.0, 0.5, 0.0)),
            cam(
                Mat3::rot2(-0.2) * Mat3::rot1(0.08),
                Vec3::new(4.0, -1.0, 0.5),
            ),
        ];
        let mut seed = 21u64;
        let points: Vec<Vec3> = (0..4)
            .map(|_| {
                Vec3::new(
                    lcg(&mut seed) * 3.0,
                    lcg(&mut seed) * 3.0,
                    14.0 + lcg(&mut seed) * 2.0,
                )
            })
            .collect();
        // Observations of a *displaced* truth, so the residuals -- and therefore `ec`, `ep` and the
        // whole right-hand side -- are non-zero. A step computed from a zero residual is zero, and
        // zero equals zero however the assembly is mutated.
        let observations: Vec<Observation> = cameras
            .iter()
            .enumerate()
            .flat_map(|(ci, c)| {
                points.iter().enumerate().map(move |(pi, p)| Observation {
                    camera: ci,
                    point: pi,
                    pixel: c.project(*p + Vec3::new(0.05, -0.04, 0.07)),
                })
            })
            .collect();

        let blocks = assemble(&cameras, &points, &observations, None);
        let reduced = [Some(0usize), Some(1usize)];
        let (n_free, m) = (2usize, points.len());
        let lambda = 1e-3;
        let (schur_cams, schur_pts) =
            schur_step(&cameras, &points, &blocks, &reduced, n_free, lambda).unwrap();

        let (dim, a, rhs) = full_normal_system(&blocks, n_free, m, lambda);
        let dense = solve_dense(dim, &a, &rhs).unwrap();

        for j in 0..n_free {
            let want = apply_pose_step(&cameras[j], &dense[6 * j..6 * j + 6]);
            assert!(
                (schur_cams[j].center - want.center).norm() < 1e-9,
                "camera {j} centre: {:?} vs {:?}",
                schur_cams[j].center,
                want.center
            );
            for r in 0..3 {
                for c in 0..3 {
                    assert!(
                        libm::fabs(schur_cams[j].rotation.rows[r][c] - want.rotation.rows[r][c])
                            < 1e-11,
                        "camera {j} rotation[{r}][{c}]"
                    );
                }
            }
        }
        for i in 0..m {
            let want = points[i]
                + Vec3::new(
                    dense[6 * n_free + 3 * i],
                    dense[6 * n_free + 3 * i + 1],
                    dense[6 * n_free + 3 * i + 2],
                );
            assert!(
                (schur_pts[i] - want).norm() < 1e-9,
                "point {i}: {:?} vs {:?}",
                schur_pts[i],
                want
            );
        }
    }

    /// The pose and the fourteen world points the RANSAC tests below perturb four different ways.
    fn ransac_fixture() -> (PinholeCamera, Vec<Vec3>) {
        let truth = cam(
            Mat3::rot3(0.1) * Mat3::rot1(-0.15),
            Vec3::new(1.5, -1.0, 11.0),
        );
        let mut s = 5u64;
        let worlds = (0..14)
            .map(|_| Vec3::new(lcg(&mut s) * 3.0, lcg(&mut s) * 3.0, lcg(&mut s) * 2.0))
            .collect();
        (truth, worlds)
    }

    /// `worlds` seen by `truth`, each pixel displaced by up to `amp` px from a fixed LCG stream.
    fn jittered(
        truth: &PinholeCamera,
        worlds: &[Vec3],
        amp: f64,
        mut s: u64,
    ) -> Vec<(Vec3, [f64; 2])> {
        worlds
            .iter()
            .map(|&w| {
                let p = truth.project(w);
                (w, [p[0] + lcg(&mut s) * amp, p[1] + lcg(&mut s) * amp])
            })
            .collect()
    }

    #[test]
    fn distinct_seeds_draw_distinct_samples_because_the_mixer_is_a_bijection() {
        // `seed ^ K` is a bijection on u64, so no two seeds can share a sampling stream. `|` and `&`
        // are not: `0 | K == K | K` and `0 & K == !K & K`, so under either mutation two of the three
        // seeds below collapse onto the same stream and return the same pose. One round each, on
        // controls too noisy for one three-point fit to stand in for another, leaves the sampled
        // triple as the only thing deciding the answer -- and a round skipped outright (the guard
        // read as `== 3` or `<= 3`, which a three-point sample always satisfies) returns `init`.
        const K: u64 = 0x9e37_79b9_7f4a_7c15;
        let (truth, worlds) = ransac_fixture();
        let controls = jittered(&truth, &worlds, 2.0, 11);
        let init = cam(Mat3::rot3(0.0), Vec3::new(0.0, 0.0, 8.0));

        let poses: Vec<Vec3> = [0u64, K, !K]
            .iter()
            .map(|&s| ransac_resection(&controls, init, 3, 1, 0.5, s).0.center)
            .collect();
        for (i, p) in poses.iter().enumerate() {
            assert!(
                (*p - init.center).norm() > 1.0,
                "seed {i} left the pose at the initial guess: {p:?}"
            );
            for q in &poses[i + 1..] {
                assert!(
                    (*p - *q).norm() > 1e-3,
                    "two seeds sampled the same triple: {p:?} vs {q:?}"
                );
            }
        }
    }

    #[test]
    fn a_control_exactly_at_the_threshold_is_not_an_inlier() {
        // The consensus test is `< threshold`, so a control sitting exactly on it is out -- and the
        // fixture makes that distance exact rather than approximate. Six controls are exact
        // projections of `truth`; the seventh is displaced by (3, 4) px, and the threshold is the
        // distance `control_error` itself reports for it, so the comparison is a true tie. Seed 4
        // draws exactly the six exact controls in its single round, so the fitted pose is `truth` to
        // the bit and the tie is reached. Admitting it at either the vote or the refit turns the
        // answer into a seven-control least-squares compromise 0.15 units off the true centre.
        let (truth, worlds) = ransac_fixture();
        let mut controls: Vec<(Vec3, [f64; 2])> =
            worlds[..6].iter().map(|&w| (w, truth.project(w))).collect();
        let odd = worlds[6];
        let p = truth.project(odd);
        controls.push((odd, [p[0] + 3.0, p[1] + 4.0]));
        let threshold = control_error(&truth, odd, controls[6].1);

        let (fitted, inliers) = ransac_resection(&controls, truth, 6, 1, threshold, 4);
        assert_eq!(inliers, vec![0, 1, 2, 3, 4, 5], "threshold {threshold}");
        for (a, b) in [
            (fitted.center.x, truth.center.x),
            (fitted.center.y, truth.center.y),
            (fitted.center.z, truth.center.z),
        ] {
            assert_eq!(a.to_bits(), b.to_bits(), "refit moved the pose: {a} vs {b}");
        }
    }

    #[test]
    fn a_zero_threshold_admits_nothing_and_hands_back_the_pose_it_was_given() {
        // A threshold of zero is a real threshold, not a disabled one. No noisy control reprojects
        // exactly, so every round scores zero inliers, the strict `>` never lets a round claim the
        // lead, the refit's `>= 3` is not met, and the initial pose comes back bit for bit. Read as
        // `>=`, the last round's fit would be adopted instead; read as `<=`, the three controls each
        // round fitted exactly -- residual exactly zero -- would be admitted and the pose would move.
        let (truth, worlds) = ransac_fixture();
        let controls = jittered(&truth, &worlds, 2.0, 11);
        let init = cam(Mat3::rot3(0.0), Vec3::new(0.0, 0.0, 8.0));

        let (kept, inliers) = ransac_resection(&controls, init, 3, 5, 0.0, 9);
        assert!(inliers.is_empty(), "{inliers:?}");
        for (a, b) in [
            (kept.center.x, init.center.x),
            (kept.center.y, init.center.y),
            (kept.center.z, init.center.z),
        ] {
            assert_eq!(a.to_bits(), b.to_bits(), "pose moved: {a} vs {b}");
        }
    }

    #[test]
    fn the_pose_it_returns_minimises_the_cost_over_the_consensus_it_reports() {
        // The consensus refit is what makes the answer a least-squares pose rather than the
        // three-point fit that happened to win the vote. Its defining property is stationarity: no
        // nudge of the returned pose, in any of the six pose directions at any of three scales, may
        // lower the reprojection cost over the inliers it reports. Skipping the refit leaves a pose
        // that a 1e-2 nudge improves by 0.46 -- three times the whole cost measured here.
        let (truth, worlds) = ransac_fixture();
        let controls = jittered(&truth, &worlds, 0.3, 21);
        let init = cam(Mat3::rot3(0.0), Vec3::new(0.0, 0.0, 8.0));

        let (fitted, inliers) = ransac_resection(&controls, init, 3, 40, 1.0, 3);
        assert_eq!(inliers.len(), 14);
        let cost = |c: &PinholeCamera| -> f64 {
            inliers
                .iter()
                .map(|&i| sq(control_error(c, controls[i].0, controls[i].1)))
                .sum()
        };
        let here = cost(&fitted);
        assert!(here > 0.0, "the fixture has no residual to minimise");
        for h in [1e-4, 1e-3, 1e-2] {
            for k in 0..3 {
                for sign in [1.0, -1.0] {
                    let mut d = Vec3::new(0.0, 0.0, 0.0);
                    match k {
                        0 => d.x = sign * h,
                        1 => d.y = sign * h,
                        _ => d.z = sign * h,
                    }
                    let mut shifted = fitted;
                    shifted.center = fitted.center + d;
                    let mut turned = fitted;
                    turned.rotation = so3_exp(d) * fitted.rotation;
                    for probe in [shifted, turned] {
                        assert!(
                            cost(&probe) >= here,
                            "a nudge of {h} improved the cost by {:e}",
                            here - cost(&probe)
                        );
                    }
                }
            }
        }
    }

    /// Three cameras and ten points, the first camera exact and held fixed, the other two poses and
    /// every point displaced far enough that the fit must escalate its damping to get back.
    fn hard_bundle() -> (Vec<PinholeCamera>, Vec<Vec3>, Vec<Observation>) {
        let truth_cams = [
            cam(
                Mat3::rot3(0.05) * Mat3::rot1(-0.1),
                Vec3::new(0.0, 0.0, 9.0),
            ),
            cam(
                Mat3::rot3(-0.2) * Mat3::rot2(0.25),
                Vec3::new(4.0, 1.0, 8.0),
            ),
            cam(
                Mat3::rot1(0.18) * Mat3::rot2(-0.3),
                Vec3::new(-3.5, -1.2, 9.5),
            ),
        ];
        let mut s = 77u64;
        let truth_pts: Vec<Vec3> = (0..10)
            .map(|_| Vec3::new(lcg(&mut s) * 2.5, lcg(&mut s) * 2.5, lcg(&mut s) * 1.5))
            .collect();
        let mut observations = Vec::new();
        for (camera, c) in truth_cams.iter().enumerate() {
            for (point, &p) in truth_pts.iter().enumerate() {
                observations.push(Observation {
                    camera,
                    point,
                    pixel: c.project(p),
                });
            }
        }
        let cameras = vec![
            truth_cams[0],
            cam(Mat3::rot3(-0.1) * Mat3::rot2(0.1), Vec3::new(3.0, 0.4, 8.6)),
            cam(
                Mat3::rot1(0.05) * Mat3::rot2(-0.15),
                Vec3::new(-2.6, -0.5, 9.9),
            ),
        ];
        let mut t = 99u64;
        let points = truth_pts
            .iter()
            .map(|&p| p + Vec3::new(lcg(&mut t) * 0.4, lcg(&mut t) * 0.4, lcg(&mut t) * 0.4))
            .collect();
        (cameras, points, observations)
    }

    #[test]
    fn bundle_adjustment_pays_a_measured_price_for_its_damping_schedule() {
        // Levenberg-Marquardt's damping schedule is not free, and only its price shows it: a fit that
        // escalates lambda additively (1e-3 + 10) or backwards (1e-3 / 10) still converges to *a*
        // pose, just far slower or not at all. Measured, the additive ladder is still at 0.28 px
        // after forty iterations and the inverted one stalls at 0.44 px from the fifth on, against
        // 7.6e-14 px here. The budget below is what the tenfold ladder buys.
        let mut previous = f64::INFINITY;
        for (iters, budget) in [
            (1u32, f64::INFINITY),
            (2, 1.0),
            (8, 0.40),
            (20, 0.15),
            (40, 1e-9),
        ] {
            let (mut cameras, mut points, observations) = hard_bundle();
            let rms = bundle_adjust(&mut cameras, &mut points, &observations, &[0], iters);
            assert!(rms < budget, "{iters} iterations left {rms:e} px");
            assert!(rms <= previous, "{iters} iterations undid earlier progress");
            previous = rms;
        }
        // Non-vacuous: the fixture really does start far enough away to need the schedule.
        let (mut cameras, mut points, observations) = hard_bundle();
        let one = bundle_adjust(&mut cameras, &mut points, &observations, &[0], 1);
        assert!(one > 1.0, "the fixture starts already converged: {one:e}");
    }

    #[test]
    fn with_no_controls_at_all_the_solver_refuses_every_step_and_returns_the_initial_pose() {
        // The one input that reaches `resection`'s `solve_dense` refusal arm: an empty control set
        // makes the normal matrix identically zero, so every one of the eight damping trials refuses
        // and the pose is handed back untouched. The arm is reachable but blind to lambda -- the
        // Marquardt damping is multiplicative (`hd[i][i] *= 1.0 + lambda`), so it cannot lift a zero
        // pivot however the schedule moves it.
        let start = cam(Mat3::rot3(0.3) * Mat3::rot1(0.2), Vec3::new(1.0, -2.0, 7.0));
        let out = resection(&[], start, 5);
        for (a, b) in [
            (out.center.x, start.center.x),
            (out.center.y, start.center.y),
            (out.center.z, start.center.z),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for r in 0..3 {
            for k in 0..3 {
                assert_eq!(
                    out.rotation.rows[r][k].to_bits(),
                    start.rotation.rows[r][k].to_bits()
                );
            }
        }
    }
}
