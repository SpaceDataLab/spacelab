//! Host-side GPU acceleration for spacelab's embarrassingly-parallel batch operations.
//!
//! The `no_std` numerical core cannot host GPU kernels, so this is a separate std-only crate that
//! offloads *large* batches to a `wgpu` compute pipeline (Vulkan / Metal / DX12, so it runs on any GPU)
//! and **falls back to the CPU below a break-even batch size**, where the launch + PCIe transfer would
//! dwarf the work. The CPU path is always the correctness oracle: every kernel reproduces it.
//!
//! WGSL compute is `f32`, so the GPU is used where single precision is *appropriate* — screening and
//! coarse maps — not for precision propagation. The flagship case is a conjunction all-pairs screen:
//! the O(N²) closest-approach filter runs on the GPU in `f32` (fine against a kilometre-scale gate),
//! and the CPU computes the precise collision probability only for the few flagged pairs.

#![allow(
    clippy::cast_possible_truncation, // f32 is the GPU's — and this screen's — working precision
    clippy::doc_markdown,             // domain acronyms: WGSL, PCIe, DX12, GPU, …
    clippy::many_single_char_names    // short-lived vector-math locals (r, v, a, b, …)
)]

pub mod batch_sgp4;

use bytemuck::{Pod, Zeroable};
use spacelab_astro::lambert as astro_lambert;
use spacelab_core::Vec3;
use wgpu::util::DeviceExt;

/// A screening state — position (m) and velocity (m/s) at the screen epoch — padded so each `vec3<f32>`
/// meets WGSL's 16-byte storage alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ScreenState {
    /// Position, metres.
    pub r: [f32; 3],
    _pad0: f32,
    /// Velocity, metres per second.
    pub v: [f32; 3],
    _pad1: f32,
}

impl ScreenState {
    /// A state from a position (m) and velocity (m/s).
    #[must_use]
    pub const fn new(r: [f32; 3], v: [f32; 3]) -> Self {
        Self {
            r,
            _pad0: 0.0,
            v,
            _pad1: 0.0,
        }
    }
}

/// Below this object count the O(N²) CPU screen beats a GPU dispatch (launch + upload + readback).
/// Chosen conservatively; the crossover is measured in `examples/conjunction_bench`.
pub const GPU_MIN_OBJECTS: usize = 4096;

/// Per-object minimum closest-approach distance (m) to any *other* object, over a linear relative-motion
/// window `[0, window_s]` — the standard conjunction pre-screen. Below this each object's distance to
/// every flagged neighbour is refined on the CPU with the full collision-probability model.
///
/// Runs on the GPU for large catalogues (`len ≥ GPU_MIN_OBJECTS` and a device is available), else on the
/// CPU. Both paths compute the same closed-form closest approach `min_t |Δr₀ + Δv₀·t|`; the GPU does it
/// in `f32` (screening precision, ~metre), the CPU in `f64`.
#[must_use]
pub fn pairwise_min_approach(states: &[ScreenState], window_s: f32) -> Vec<f32> {
    if states.len() >= GPU_MIN_OBJECTS
        && let Some(gpu) = Gpu::new()
    {
        return gpu.pairwise_min_approach(states, window_s);
    }
    pairwise_min_approach_cpu(states, window_s)
}

/// The CPU reference for [`pairwise_min_approach`] (also the below-threshold fallback): an exact `f64`
/// O(N²) closest-approach screen. Returned as `f32` — the screening precision — so the GPU path is a
/// drop-in.
#[must_use]
pub fn pairwise_min_approach_cpu(states: &[ScreenState], window_s: f32) -> Vec<f32> {
    let window = f64::from(window_s);
    let n = states.len();
    let mut out = vec![0.0_f32; n];
    parallel_fill(&mut out, |i| {
        let (ri, vi) = (f64x3(states[i].r), f64x3(states[i].v));
        let mut best = f64::INFINITY;
        for (j, sj) in states.iter().enumerate() {
            if j == i {
                continue;
            }
            let dr = sub(ri, f64x3(sj.r));
            let dv = sub(vi, f64x3(sj.v));
            let vv = dot(dv, dv);
            // closest approach of two points in uniform relative motion, clamped to the window
            let t = if vv > 0.0 {
                (-dot(dr, dv) / vv).clamp(0.0, window)
            } else {
                0.0
            };
            let d = [dr[0] + dv[0] * t, dr[1] + dv[1] * t, dr[2] + dv[2] * t];
            best = best.min(dot(d, d).sqrt());
        }
        best as f32
    });
    out
}

fn f64x3(a: [f32; 3]) -> [f64; 3] {
    [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    if n > 0.0 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        a
    }
}

/// Fill `out[i] = f(i)` across scoped threads — the CPU-path parallelism these batch kernels get for
/// free because every output cell is independent. Order-preserving and exact: the result is identical
/// to the serial map, so these stay valid GPU oracles. Small batches (below ~2 cells per core) run
/// sequentially, where the spawn overhead would dominate. Dependency-free, mirroring `spacelab_sgp4`.
fn parallel_fill<T: Send, F: Fn(usize) -> T + Sync>(out: &mut [T], f: F) {
    let n = out.len();
    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    if threads <= 1 || n < 2 * threads {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = f(i);
        }
        return;
    }
    let per = n.div_ceil(threads);
    let f = &f;
    std::thread::scope(|scope| {
        for (c, chunk) in out.chunks_mut(per).enumerate() {
            let base = c * per;
            scope.spawn(move || {
                for (k, slot) in chunk.iter_mut().enumerate() {
                    *slot = f(base + k);
                }
            });
        }
    });
}

/// GPU/CPU work threshold for the coverage count: below this `ground × satellites` product the CPU
/// O(G·S) sweep beats a GPU dispatch. Measured in `examples/coverage_bench`.
pub const GPU_MIN_COVERAGE_WORK: usize = 1 << 20;

/// Per-ground-point count of visible satellites — elevation ≥ `min_el_rad` above the local (spherical)
/// horizon — i.e. the instantaneous coverage of a constellation over a grid of Earth-fixed points.
/// Positions are ECEF/ITRF (m). Runs on the GPU for large grids (`ground × satellites ≥
/// GPU_MIN_COVERAGE_WORK` and a device is available), else on the CPU. The GPU works in `f32`
/// (metre-scale — well inside a coverage map's needs), the CPU in `f64`.
#[must_use]
pub fn coverage_counts(satellites: &[Ecef], ground: &[Ecef], min_el_rad: f32) -> Vec<u32> {
    if satellites.len().saturating_mul(ground.len()) >= GPU_MIN_COVERAGE_WORK
        && let Some(gpu) = Gpu::new()
    {
        return gpu.coverage_counts(satellites, ground, min_el_rad);
    }
    coverage_counts_cpu(satellites, ground, min_el_rad)
}

/// The `f64` CPU reference for [`coverage_counts`] (also the below-threshold fallback): for each ground
/// point, count the satellites whose elevation above the local spherical horizon is at least
/// `min_el_rad`.
#[must_use]
pub fn coverage_counts_cpu(satellites: &[Ecef], ground: &[Ecef], min_el_rad: f32) -> Vec<u32> {
    let sin_min = f64::from(min_el_rad).sin();
    let mut out = vec![0_u32; ground.len()];
    parallel_fill(&mut out, |i| {
        let gp = f64x3(ground[i].xyz);
        let up = normalize(gp);
        let mut count = 0u32;
        for s in satellites {
            let d = sub(f64x3(s.xyz), gp); // line of sight, ground → satellite
            // elevation ≥ mask  ⇔  dot(d, up) ≥ sin(mask)·|d|
            if dot(d, up) >= sin_min * dot(d, d).sqrt() {
                count += 1;
            }
        }
        count
    });
    out
}

/// GPU/CPU work threshold for the porkchop grid: below this `departures × arrivals` cell count the CPU
/// beats a GPU dispatch. Measured in `examples/porkchop_bench`.
pub const GPU_MIN_PORKCHOP_CELLS: usize = 1 << 14;

/// A porkchop-plot grid of departure C3 (`|v₁ − v_dep|²`, m²/s²) — one Lambert transfer per (departure
/// `i`, arrival `j`) cell, row-major (`out[i*n_arr + j]`). `dep[i]`/`arr[j]` are the departure/arrival
/// body states (position m, velocity m/s) at times `dep_t[i]`/`arr_t[j]` (s); the time of flight is
/// `arr_t[j] − dep_t[i]`. Cells with no single-rev solution get [`PORKCHOP_NO_SOLUTION`]. Runs on the
/// GPU for large grids (`≥ GPU_MIN_PORKCHOP_CELLS` and a device is available), else on the CPU. The GPU
/// solves in `f32` (fine for a C3 contour map), the CPU in `f64`.
#[must_use]
pub fn porkchop_c3(
    dep: &[ScreenState],
    arr: &[ScreenState],
    dep_t: &[f32],
    arr_t: &[f32],
    mu: f32,
    prograde: bool,
) -> Vec<f32> {
    if dep.len().saturating_mul(arr.len()) >= GPU_MIN_PORKCHOP_CELLS
        && let Some(gpu) = Gpu::new()
    {
        return gpu.porkchop_c3(dep, arr, dep_t, arr_t, mu, prograde);
    }
    porkchop_c3_cpu(dep, arr, dep_t, arr_t, mu, prograde)
}

/// The `f64` CPU reference for [`porkchop_c3`] (also the below-threshold fallback), solving each cell
/// with the independent [`spacelab_astro::lambert`](fn@spacelab_astro::lambert).
#[must_use]
pub fn porkchop_c3_cpu(
    dep: &[ScreenState],
    arr: &[ScreenState],
    dep_t: &[f32],
    arr_t: &[f32],
    mu: f32,
    prograde: bool,
) -> Vec<f32> {
    let mu = f64::from(mu);
    let mut out = Vec::with_capacity(dep.len() * arr.len());
    for (i, di) in dep.iter().enumerate() {
        let r1 = vec3(di.r);
        let vdep = vec3(di.v);
        for (j, aj) in arr.iter().enumerate() {
            let tof = f64::from(arr_t[j] - dep_t[i]);
            let c3 = if tof > 0.0 {
                astro_lambert(r1, vec3(aj.r), tof, mu, prograde).map_or(
                    PORKCHOP_NO_SOLUTION,
                    |(v1, _)| {
                        let dv = Vec3::new(v1.x - vdep.x, v1.y - vdep.y, v1.z - vdep.z);
                        dv.dot(dv) as f32
                    },
                )
            } else {
                PORKCHOP_NO_SOLUTION
            };
            out.push(c3);
        }
    }
    out
}

fn vec3(a: [f32; 3]) -> Vec3 {
    Vec3::new(f64::from(a[0]), f64::from(a[1]), f64::from(a[2]))
}

/// GPU/CPU work threshold for batch SGP4: below this `satellites × epochs` product the CPU wins.
pub const GPU_MIN_SGP4_CELLS: usize = 1 << 14;

/// Batch near-Earth SGP4: TEME state (position m, velocity m/s) of every satellite at every epoch,
/// row-major (`out[i*times.len() + j]` = satellite `i` at `times[j]` minutes since its epoch). A
/// decayed/diverged cell is zero. Runs on the GPU for large batches (`≥ GPU_MIN_SGP4_CELLS` and a
/// device is available), else on the CPU. The GPU works in `f32` (~tens of metres over a day — well
/// inside SGP4's own ~km accuracy), the CPU in `f64`. Positions/velocities are **metres** (SGP4's
/// native km are scaled by 1000), so the output drops straight into [`pairwise_min_approach`].
#[must_use]
pub fn sgp4_propagate(sats: &[batch_sgp4::Sgp4Sat], times: &[f32]) -> Vec<ScreenState> {
    if sats.len().saturating_mul(times.len()) >= GPU_MIN_SGP4_CELLS
        && let Some(gpu) = Gpu::new()
    {
        let init: Vec<batch_sgp4::Sgp4Init> = sats.iter().map(|s| s.to_gpu()).collect();
        return km_states_to_m(gpu.sgp4_propagate(&init, times));
    }
    sgp4_propagate_cpu(sats, times)
}

/// The `f64` CPU reference for [`sgp4_propagate`] (also the below-threshold fallback).
#[must_use]
pub fn sgp4_propagate_cpu(sats: &[batch_sgp4::Sgp4Sat], times: &[f32]) -> Vec<ScreenState> {
    let t = times.len();
    if t == 0 {
        return Vec::new();
    }
    // Row-major (`out[i*t + j]` = satellite `i` at time `j`); every cell is an independent propagate.
    let mut out = vec![ScreenState::new([0.0; 3], [0.0; 3]); sats.len() * t];
    parallel_fill(&mut out, |cell| {
        let s = &sats[cell / t];
        let time = times[cell % t];
        s.propagate(f64::from(time)).map_or_else(
            || ScreenState::new([0.0; 3], [0.0; 3]),
            |(r, v)| {
                ScreenState::new(
                    [
                        (r[0] * 1000.0) as f32,
                        (r[1] * 1000.0) as f32,
                        (r[2] * 1000.0) as f32,
                    ],
                    [
                        (v[0] * 1000.0) as f32,
                        (v[1] * 1000.0) as f32,
                        (v[2] * 1000.0) as f32,
                    ],
                )
            },
        )
    });
    out
}

/// The GPU kernel emits SGP4's native kilometres; scale the batch to metres (leaving zeroed decayed
/// cells at the origin).
fn km_states_to_m(mut states: Vec<ScreenState>) -> Vec<ScreenState> {
    for s in &mut states {
        s.r = [s.r[0] * 1000.0, s.r[1] * 1000.0, s.r[2] * 1000.0];
        s.v = [s.v[0] * 1000.0, s.v[1] * 1000.0, s.v[2] * 1000.0];
    }
    states
}

const SCREEN_SHADER: &str = r"
struct State { r: vec3<f32>, v: vec3<f32> };
struct Params { n: u32, window: f32 };

@group(0) @binding(0) var<storage, read>       states:  array<State>;
@group(0) @binding(1) var<storage, read_write> out_min: array<f32>;
@group(0) @binding(2) var<uniform>             params:  Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n) { return; }
    let si = states[i];
    var best: f32 = 3.4e38;
    for (var j: u32 = 0u; j < params.n; j = j + 1u) {
        if (j == i) { continue; }
        let sj = states[j];
        let dr = si.r - sj.r;
        let dv = si.v - sj.v;
        let vv = dot(dv, dv);
        var t: f32 = 0.0;
        if (vv > 0.0) { t = clamp(-dot(dr, dv) / vv, 0.0, params.window); }
        let d = dr + dv * t;
        best = min(best, sqrt(dot(d, d)));
    }
    out_min[i] = best;
}
";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenParams {
    n: u32,
    window: f32,
    _pad: [u32; 2],
}

/// A padded Earth-fixed position (m). WGSL storage arrays give a `vec3<f32>` a 16-byte stride, so the
/// GPU reads `array<vec4<f32>>` and this matches it byte-for-byte.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Ecef {
    /// ECEF / ITRF position, metres.
    pub xyz: [f32; 3],
    _pad: f32,
}

impl Ecef {
    /// An Earth-fixed position from its metres.
    #[must_use]
    pub const fn new(xyz: [f32; 3]) -> Self {
        Self { xyz, _pad: 0.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoverageParams {
    n_sat: u32,
    n_ground: u32,
    sin_min_el: f32,
    _pad: u32,
}

const COVERAGE_SHADER: &str = r"
struct Params { n_sat: u32, n_ground: u32, sin_min_el: f32 };
@group(0) @binding(0) var<storage, read>       sats:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       ground: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<u32>;
@group(0) @binding(3) var<uniform>             params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;
    if (g >= params.n_ground) { return; }
    let gp = ground[g].xyz;
    let up = normalize(gp);
    var count: u32 = 0u;
    for (var s: u32 = 0u; s < params.n_sat; s = s + 1u) {
        let d = sats[s].xyz - gp;                // line of sight, ground -> satellite
        // elevation >= mask  <=>  dot(d, up) >= sin(mask) * |d|
        if (dot(d, up) >= params.sin_min_el * length(d)) { count = count + 1u; }
    }
    counts[g] = count;
}
";

/// Sentinel (m²/s²) written when a Lambert cell has no single-revolution solution — including a narrow
/// band around the 0°/180° transfer singularity, where the geometry is too ill-conditioned for `f32`
/// (the useless high-energy ridge; refine on the CPU if ever needed).
pub const PORKCHOP_NO_SOLUTION: f32 = 3.0e30;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PorkchopParams {
    n_dep: u32,
    n_arr: u32,
    mu: f32,
    prograde: u32,
}

// A faithful f32 port of `spacelab_astro::lambert` (universal-variable / Stumpff, Newton on z with an
// upward bracket) — one thread per (departure, arrival) cell, output = departure C3 = |v₁ − v_dep|².
const LAMBERT_SHADER: &str = r"
struct Body { r: vec3<f32>, v: vec3<f32> };
struct Params { n_dep: u32, n_arr: u32, mu: f32, prograde: u32 };

@group(0) @binding(0) var<storage, read>       dep:    array<Body>;
@group(0) @binding(1) var<storage, read>       arr:    array<Body>;
@group(0) @binding(2) var<storage, read>       times:  array<f32>;  // dep_t (n_dep) then arr_t (n_arr)
@group(0) @binding(3) var<storage, read_write> out_c3: array<f32>;
@group(0) @binding(4) var<uniform>             params: Params;

const TAU: f32 = 6.28318530717959;
const SENTINEL: f32 = 3.0e30;

fn stumpff_c(z: f32) -> f32 {
    if (z > 1e-6) { let s = sqrt(z); return (1.0 - cos(s)) / z; }
    if (z < -1e-6) { let s = sqrt(-z); return (cosh(s) - 1.0) / (-z); }
    return 0.5 - z / 24.0 + z * z / 720.0;
}
fn stumpff_s(z: f32) -> f32 {
    if (z > 1e-6) { let s = sqrt(z); return (s - sin(s)) / (s * z); }
    if (z < -1e-6) { let s = sqrt(-z); return (sinh(s) - s) / (s * (-z)); }
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0;
}
fn yfun(z: f32, a: f32, r1n: f32, r2n: f32) -> f32 {
    return r1n + r2n + a * (z * stumpff_s(z) - 1.0) / sqrt(stumpff_c(z));
}
fn tof_err(z: f32, a: f32, r1n: f32, r2n: f32, sqmu: f32, tof: f32) -> f32 {
    let c = stumpff_c(z);
    let s = stumpff_s(z);
    let yz = yfun(z, a, r1n, r2n);
    return pow(yz / c, 1.5) * s + a * sqrt(yz) - sqmu * tof;
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_dep * params.n_arr) { return; }
    let i = idx / params.n_arr;
    let j = idx % params.n_arr;
    let r1 = dep[i].r;
    let vdep = dep[i].v;
    let r2 = arr[j].r;
    let tof = times[params.n_dep + j] - times[i];
    if (tof <= 0.0) { out_c3[idx] = SENTINEL; return; }

    let r1n = length(r1);
    let r2n = length(r2);
    let cross_z = r1.x * r2.y - r1.y * r2.x;
    let cos_dnu = clamp(dot(r1, r2) / (r1n * r2n), -1.0, 1.0);
    let acos_v = acos(cos_dnu);
    var dnu: f32 = TAU - acos_v;
    if (params.prograde == 1u) {
        if (cross_z >= 0.0) { dnu = acos_v; }
    } else {
        if (cross_z < 0.0) { dnu = acos_v; }
    }
    let sin_dnu = sin(dnu);
    // Mask a narrow band around the 0°/180° transfer singularity (a → 0, plane undefined): there the
    // problem is so ill-conditioned that f32 cannot track f64 — the ridge is not a usable transfer and
    // is refined on the CPU if ever needed.
    if (abs(sin_dnu) < 1e-2) { out_c3[idx] = SENTINEL; return; }
    let a = sin_dnu * sqrt(r1n * r2n / (1.0 - cos_dnu));
    let sqmu = sqrt(params.mu);

    // Single-revolution root by bisection on the monotone tof(z), z in [-50, 39) — below the 4π²
    // multi-rev singularity. Robust in f32, where Newton diverges to a spurious root on the
    // ill-conditioned cells (near-singular geometry) that a porkchop grid is full of. A cell whose
    // required time of flight lies outside [tof(-50), tof(39)) has no single-rev solution → sentinel.
    var zlo: f32 = -50.0;
    var zhi: f32 = 39.0;
    // Long-way transfers (a < 0) drive y(z) negative at large z, where sqrt(y) is NaN; keep zlo the
    // largest z with BOTH y > 0 and tof_err < 0, so the bracket stays inside the valid region and
    // converges to the root from below. A cell with no such root is masked by the final gate.
    if (!(tof_err(zlo, a, r1n, r2n, sqmu, tof) < 0.0)) { out_c3[idx] = SENTINEL; return; }
    for (var it: i32 = 0; it < 70; it = it + 1) {
        let zm = 0.5 * (zlo + zhi);
        if (yfun(zm, a, r1n, r2n) > 0.0 && tof_err(zm, a, r1n, r2n, sqmu, tof) < 0.0) {
            zlo = zm;
        } else {
            zhi = zm;
        }
    }
    let z = zlo;
    let yz = yfun(z, a, r1n, r2n);
    if (!(yz > 0.0) || abs(tof_err(z, a, r1n, r2n, sqmu, tof)) > 1e-3 * sqmu * tof) {
        out_c3[idx] = SENTINEL; return;
    }
    let lg = a * sqrt(yz / params.mu);
    let lf = 1.0 - yz / r1n;
    let v1 = (r2 - lf * r1) / lg;
    let dv = v1 - vdep;
    out_c3[idx] = dot(dv, dv);
}
";

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Sgp4Params {
    n_sat: u32,
    n_time: u32,
    _pad: [u32; 2],
}

// A faithful f32 port of `batch_sgp4::Sgp4Sat::propagate` (near-Earth SGP4), one thread per
// (satellite, epoch) cell → TEME (position, velocity). A diverged/decayed cell writes zeros.
const SGP4_SHADER: &str = r"
struct Sgp4 {
    inclination: f32, right_ascension_0: f32, eccentricity_0: f32, argument_of_perigee_0: f32,
    mean_anomaly_0: f32, mean_motion_0: f32, right_ascension_dot: f32, argument_of_perigee_dot: f32,
    mean_anomaly_dot: f32, c1: f32, c4: f32, k0: f32, k1: f32, a0: f32,
    k2: f32, k3: f32, k4: f32, k5: f32, k6: f32, eta: f32, c5: f32, d2: f32, d3: f32, d4: f32,
    k7: f32, k8: f32, k9: f32, k10: f32, k11: f32, k12: f32, k13: f32,
    high_altitude: u32, elliptic: u32, pad0: u32, pad1: u32, pad2: u32,
};
struct State { r: vec3<f32>, v: vec3<f32> };
struct Params { n_sat: u32, n_time: u32 };

@group(0) @binding(0) var<storage, read>       sats:  array<Sgp4>;
@group(0) @binding(1) var<storage, read>       times: array<f32>;
@group(0) @binding(2) var<storage, read_write> out:   array<State>;
@group(0) @binding(3) var<uniform>             params: Params;

const KE: f32 = 0.07436685316871385;
const AE: f32 = 6378.137;
const J2: f32 = 0.00108262998905;
const TAU: f32 = 6.28318530717959;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_sat * params.n_time) { return; }
    let s = sats[idx / params.n_time];
    let t = times[idx % params.n_time];
    let zero = State(vec3<f32>(0.0), vec3<f32>(0.0));

    let p22 = s.right_ascension_0 + s.right_ascension_dot * t + s.k0 * t * t;
    let p23 = s.argument_of_perigee_0 + s.argument_of_perigee_dot * t;
    let p24 = s.mean_anomaly_0 + s.mean_anomaly_dot * t;
    var argument_of_perigee: f32;
    var mean_anomaly: f32;
    var a: f32;
    var p27: f32;
    if (s.high_altitude == 1u) {
        var aop = p23;
        var p26 = p24;
        if (s.elliptic == 1u) {
            let p25 = s.k13 * (pow(1.0 + s.eta * cos(p24), 3.0) - s.k11) + s.k12 * t;
            aop = p23 - p25;
            p26 = p24 + p25;
        }
        argument_of_perigee = aop;
        mean_anomaly = p26 + s.mean_motion_0
            * (s.k1 * t * t + s.k8 * t * t * t + t * t * t * t * (s.k9 + t * s.k10));
        let ad = 1.0 - s.c1 * t - s.d2 * t * t - s.d3 * t * t * t - s.d4 * t * t * t * t;
        a = s.a0 * ad * ad;
        p27 = s.eccentricity_0 - (s.c4 * t + s.c5 * (sin(p26) - s.k7));
    } else {
        argument_of_perigee = p23;
        mean_anomaly = p24 + s.mean_motion_0 * s.k1 * t * t;
        let ad = 1.0 - s.c1 * t;
        a = s.a0 * ad * ad;
        p27 = s.eccentricity_0 - s.c4 * t;
    }
    if (p27 < -0.001 || p27 >= 1.0) { out[idx] = zero; return; }
    let ecc = max(p27, 1e-6);
    let mean_motion = KE / pow(a, 1.5);
    let inc = s.inclination;

    let axn = ecc * cos(argument_of_perigee);
    let p37 = 1.0 / (a * (1.0 - ecc * ecc));
    let ayn = ecc * sin(argument_of_perigee) + p37 * s.k2;
    let p38 = (mean_anomaly + argument_of_perigee + p37 * s.k5 * axn) % TAU;
    var ew = p38;
    for (var it: i32 = 0; it < 10; it = it + 1) {
        let delta = (p38 - ayn * cos(ew) + axn * sin(ew) - ew)
            / (1.0 - cos(ew) * axn - sin(ew) * ayn);
        ew = ew + clamp(delta, -0.95, 0.95);
    }
    let p39 = axn * axn + ayn * ayn;
    let pl = a * (1.0 - p39);
    if (pl < 0.0) { out[idx] = zero; return; }
    let p40 = axn * sin(ew) - ayn * cos(ew);
    let r = a * (1.0 - (axn * cos(ew) + ayn * sin(ew)));
    let r_dot = sqrt(a) * p40 / r;
    let b = sqrt(1.0 - p39);
    let p41 = p40 / (1.0 + b);
    let p42 = a / r * (sin(ew) - ayn - axn * p41);
    let p43 = a / r * (cos(ew) - axn + ayn * p41);
    let u = atan2(p42, p43);
    let p44 = 2.0 * p43 * p42;
    let p45 = 1.0 - 2.0 * p42 * p42;
    let p46 = 0.5 * J2 / pl / pl;
    let rk = r * (1.0 - 1.5 * p46 * b * s.k6) + 0.5 * (0.5 * J2 / pl) * s.k3 * p45;
    let uk = u - 0.25 * p46 * s.k4 * p44;
    let inc_k = inc + 1.5 * p46 * cos(inc) * sin(inc) * p45;
    let raan_k = p22 + 1.5 * p46 * cos(inc) * p44;
    let rk_dot = r_dot - mean_motion * (0.5 * J2 / pl) * s.k3 * p44 / KE;
    let rfk_dot = sqrt(pl) / r + mean_motion * (0.5 * J2 / pl) * (s.k3 * p45 + 1.5 * s.k6) / KE;
    let u0 = -sin(raan_k) * cos(inc_k) * sin(uk) + cos(raan_k) * cos(uk);
    let u1 = cos(raan_k) * cos(inc_k) * sin(uk) + sin(raan_k) * cos(uk);
    let u2 = sin(inc_k) * sin(uk);
    let f = AE * KE / 60.0;
    out[idx] = State(
        vec3<f32>(rk * u0 * AE, rk * u1 * AE, rk * u2 * AE),
        vec3<f32>(
            (rk_dot * u0 + rfk_dot * (-sin(raan_k) * cos(inc_k) * cos(uk) - cos(raan_k) * sin(uk))) * f,
            (rk_dot * u1 + rfk_dot * (cos(raan_k) * cos(inc_k) * cos(uk) - sin(raan_k) * sin(uk))) * f,
            (rk_dot * u2 + rfk_dot * (sin(inc_k) * cos(uk))) * f,
        ),
    );
}
";

/// A live GPU compute context — device, queue, and one pipeline per kernel. Construction picks a
/// high-performance adapter and compiles the kernels; `None` if no GPU/driver is reachable, so the
/// caller falls back to the CPU.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    screen: wgpu::ComputePipeline,
    coverage: wgpu::ComputePipeline,
    lambert: wgpu::ComputePipeline,
    sgp4: wgpu::ComputePipeline,
}

impl Gpu {
    /// Acquire a GPU and compile the kernels, or `None` if none is available.
    #[must_use]
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("spacelab-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::default(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        let screen = build_pipeline(&device, SCREEN_SHADER, "conjunction-screen");
        let coverage = build_pipeline(&device, COVERAGE_SHADER, "coverage-count");
        let lambert = build_pipeline(&device, LAMBERT_SHADER, "porkchop-lambert");
        let sgp4 = build_pipeline(&device, SGP4_SHADER, "batch-sgp4");
        Some(Self {
            device,
            queue,
            screen,
            coverage,
            lambert,
            sgp4,
        })
    }

    /// GPU closest-approach screen — see [`pairwise_min_approach`].
    ///
    /// # Panics
    ///
    /// If `states` is empty.
    #[must_use]
    pub fn pairwise_min_approach(&self, states: &[ScreenState], window_s: f32) -> Vec<f32> {
        let n = states.len();
        assert!(n > 0, "empty catalogue");
        let out_bytes = (n * std::mem::size_of::<f32>()) as u64;
        let state_buf = self.storage(bytemuck::cast_slice(states), "states");
        let out_buf = self.output(out_bytes, "out_min");
        let param_buf = self.uniform(
            bytemuck::bytes_of(&ScreenParams {
                n: n as u32,
                window: window_s,
                _pad: [0, 0],
            }),
            "params",
        );
        self.run(
            &self.screen,
            &[bind(0, &state_buf), bind(1, &out_buf), bind(2, &param_buf)],
            &out_buf,
            out_bytes,
            n.div_ceil(64) as u32,
            "screen",
        )
    }

    /// GPU coverage count — see [`coverage_counts`].
    ///
    /// # Panics
    ///
    /// If `satellites` or `ground` is empty.
    #[must_use]
    pub fn coverage_counts(
        &self,
        satellites: &[Ecef],
        ground: &[Ecef],
        min_el_rad: f32,
    ) -> Vec<u32> {
        let (ns, ng) = (satellites.len(), ground.len());
        assert!(ns > 0 && ng > 0, "empty satellites or ground grid");
        let out_bytes = (ng * std::mem::size_of::<u32>()) as u64;
        let sat_buf = self.storage(bytemuck::cast_slice(satellites), "sats");
        let ground_buf = self.storage(bytemuck::cast_slice(ground), "ground");
        let out_buf = self.output(out_bytes, "counts");
        let param_buf = self.uniform(
            bytemuck::bytes_of(&CoverageParams {
                n_sat: ns as u32,
                n_ground: ng as u32,
                sin_min_el: min_el_rad.sin(),
                _pad: 0,
            }),
            "params",
        );
        self.run(
            &self.coverage,
            &[
                bind(0, &sat_buf),
                bind(1, &ground_buf),
                bind(2, &out_buf),
                bind(3, &param_buf),
            ],
            &out_buf,
            out_bytes,
            ng.div_ceil(64) as u32,
            "coverage",
        )
    }

    /// GPU porkchop grid — see [`porkchop_c3`].
    ///
    /// # Panics
    ///
    /// If a body/time axis length mismatches, or an axis is empty.
    #[must_use]
    pub fn porkchop_c3(
        &self,
        dep: &[ScreenState],
        arr: &[ScreenState],
        dep_t: &[f32],
        arr_t: &[f32],
        mu: f32,
        prograde: bool,
    ) -> Vec<f32> {
        assert!(
            dep.len() == dep_t.len()
                && arr.len() == arr_t.len()
                && !dep.is_empty()
                && !arr.is_empty(),
            "mismatched or empty porkchop grid axes"
        );
        let cells = dep.len() * arr.len();
        let out_bytes = (cells * std::mem::size_of::<f32>()) as u64;
        let dep_buf = self.storage(bytemuck::cast_slice(dep), "dep");
        let arr_buf = self.storage(bytemuck::cast_slice(arr), "arr");
        // One `times` buffer (dep_t then arr_t) keeps the kernel within the downlevel 4-storage-buffer
        // limit, so it runs on any GPU.
        let times: Vec<f32> = dep_t.iter().chain(arr_t).copied().collect();
        let times_buf = self.storage(bytemuck::cast_slice(&times), "times");
        let out_buf = self.output(out_bytes, "out_c3");
        let param_buf = self.uniform(
            bytemuck::bytes_of(&PorkchopParams {
                n_dep: dep.len() as u32,
                n_arr: arr.len() as u32,
                mu,
                prograde: u32::from(prograde),
            }),
            "params",
        );
        self.run(
            &self.lambert,
            &[
                bind(0, &dep_buf),
                bind(1, &arr_buf),
                bind(2, &times_buf),
                bind(3, &out_buf),
                bind(4, &param_buf),
            ],
            &out_buf,
            out_bytes,
            cells.div_ceil(64) as u32,
            "porkchop",
        )
    }

    /// GPU batch SGP4 — see [`sgp4_propagate`].
    ///
    /// # Panics
    ///
    /// If `sats` or `times` is empty.
    #[must_use]
    pub fn sgp4_propagate(&self, sats: &[batch_sgp4::Sgp4Init], times: &[f32]) -> Vec<ScreenState> {
        let (ns, nt) = (sats.len(), times.len());
        assert!(ns > 0 && nt > 0, "empty satellites or epochs");
        let cells = ns * nt;
        let out_bytes = (cells * std::mem::size_of::<ScreenState>()) as u64;
        let sat_buf = self.storage(bytemuck::cast_slice(sats), "sgp4_sats");
        let time_buf = self.storage(bytemuck::cast_slice(times), "sgp4_times");
        let out_buf = self.output(out_bytes, "sgp4_out");
        let param_buf = self.uniform(
            bytemuck::bytes_of(&Sgp4Params {
                n_sat: ns as u32,
                n_time: nt as u32,
                _pad: [0, 0],
            }),
            "params",
        );
        self.run(
            &self.sgp4,
            &[
                bind(0, &sat_buf),
                bind(1, &time_buf),
                bind(2, &out_buf),
                bind(3, &param_buf),
            ],
            &out_buf,
            out_bytes,
            cells.div_ceil(64) as u32,
            "sgp4",
        )
    }

    fn storage(&self, contents: &[u8], label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn uniform(&self, contents: &[u8], label: &str) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    fn output(&self, bytes: u64, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Bind the buffers, dispatch `groups` workgroups, and read the `out` buffer back as `Vec<T>`.
    // `get_mapped_range` cannot fail here: the buffer is mapped and `poll(Wait)` has completed.
    #[allow(clippy::expect_used)]
    fn run<T: Pod + Zeroable>(
        &self,
        pipeline: &wgpu::ComputePipeline,
        entries: &[wgpu::BindGroupEntry<'_>],
        out: &wgpu::Buffer,
        out_bytes: u64,
        groups: u32,
        label: &str,
    ) -> Vec<T> {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(out, 0, &staging, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice
            .get_mapped_range()
            .expect("staging buffer is mapped after poll(Wait)");
        let out_vec: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out_vec
    }
}

fn build_pipeline(device: &wgpu::Device, shader: &str, label: &str) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    })
}

fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::cast_precision_loss)]
    use super::*;
    use spacelab_testkit::nan_max;

    /// A deterministic pseudo-random catalogue (LEO-ish shell), no external RNG.
    fn catalogue(n: usize) -> Vec<ScreenState> {
        let mut s = 0x2545_f491_4f6c_dd1d_u64; // splitmix64 state
        let mut next = || {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32
        };
        (0..n)
            .map(|_| {
                let radius = 6.9e6 + next() * 3.0e5;
                let (a, b) = (
                    next() * core::f32::consts::TAU,
                    next() * core::f32::consts::TAU,
                );
                let r = [
                    radius * a.cos() * b.sin(),
                    radius * a.sin() * b.sin(),
                    radius * b.cos(),
                ];
                let speed = 7.5e3 + next() * 3.0e2;
                let v = [
                    speed * (next() - 0.5),
                    speed * (next() - 0.5),
                    speed * (next() - 0.5),
                ];
                ScreenState::new(r, v)
            })
            .collect()
    }

    #[test]
    fn cpu_screen_finds_a_planted_close_pair() {
        // Two objects on a near-collision course plus distant filler.
        let mut cat = catalogue(64);
        cat[0] = ScreenState::new([7.0e6, 0.0, 0.0], [0.0, 100.0, 0.0]);
        cat[1] = ScreenState::new([7.0e6 + 5.0e4, 0.0, 0.0], [-500.0, 100.0, 0.0]);
        let mins = pairwise_min_approach_cpu(&cat, 200.0);
        // Object 0 and 1 close from 50 km toward each other at 500 m/s over 200 s → ~50 km − 100 km,
        // clamped: they pass within a few km.
        assert!(mins[0] < 6.0e4, "planted pair min = {} m", mins[0]);
        assert_eq!(mins.len(), 64);
    }

    /// A deterministic set of points on a sphere of `radius` m (satellites or ground grid).
    fn sphere_points(n: usize, radius: f32, seed: u64) -> Vec<Ecef> {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32
        };
        (0..n)
            .map(|_| {
                let lon = next() * core::f32::consts::TAU;
                let colat = (2.0 * next() - 1.0).acos(); // uniform on the sphere
                Ecef::new([
                    radius * colat.sin() * lon.cos(),
                    radius * colat.sin() * lon.sin(),
                    radius * colat.cos(),
                ])
            })
            .collect()
    }

    #[test]
    fn parallel_cpu_paths_are_bit_identical_to_serial() {
        // Batch sizes above the sequential cutoff take the multi-threaded path; the result must equal
        // a serial recomputation exactly — the parallelism must not change the numbers.
        let cat = catalogue(1200);
        let window = 300.0_f32;
        let par = pairwise_min_approach_cpu(&cat, window);
        let w = f64::from(window);
        let serial: Vec<f32> = (0..cat.len())
            .map(|i| {
                let (ri, vi) = (f64x3(cat[i].r), f64x3(cat[i].v));
                let mut best = f64::INFINITY;
                for (j, sj) in cat.iter().enumerate() {
                    if j == i {
                        continue;
                    }
                    let dr = sub(ri, f64x3(sj.r));
                    let dv = sub(vi, f64x3(sj.v));
                    let vv = dot(dv, dv);
                    let t = if vv > 0.0 {
                        (-dot(dr, dv) / vv).clamp(0.0, w)
                    } else {
                        0.0
                    };
                    let d = [dr[0] + dv[0] * t, dr[1] + dv[1] * t, dr[2] + dv[2] * t];
                    best = best.min(dot(d, d).sqrt());
                }
                best as f32
            })
            .collect();
        assert_eq!(par, serial, "parallel screen differs from serial");

        let sats = sphere_points(300, 7.0e6, 0x11);
        let grid = sphere_points(2000, 6.371e6, 0x22);
        let mask = 5.0_f32.to_radians();
        let cov = coverage_counts_cpu(&sats, &grid, mask);
        let sin_min = f64::from(mask).sin();
        let serial_cov: Vec<u32> = grid
            .iter()
            .map(|g| {
                let (gp, up) = (f64x3(g.xyz), normalize(f64x3(g.xyz)));
                let mut c = 0u32;
                for s in &sats {
                    let d = sub(f64x3(s.xyz), gp);
                    if dot(d, up) >= sin_min * dot(d, d).sqrt() {
                        c += 1;
                    }
                }
                c
            })
            .collect();
        assert_eq!(cov, serial_cov, "parallel coverage differs from serial");
    }

    /// The closest-approach screen against the closed form, in all four regimes.
    ///
    /// `parallel_cpu_paths_are_bit_identical_to_serial` recomputes the screen with the very
    /// `sub`, `dot` and clamp it is checking, so a mutation lands on both sides and the two agree:
    /// it proves the threading, not the arithmetic. Here the answer is known in advance. One object
    /// starts `L` behind the crossing point moving at `u`; the other sits `D` off the track. The
    /// approach is exactly `D` at exactly `t = L/u`, so a window shorter than that must instead give
    /// `√((L − u·w)² + D²)`, a receding pair must give `√(L² + D²)` at `t = 0`, and a pair with no
    /// relative motion must give its present separation.
    #[test]
    fn the_cpu_screen_reproduces_the_closed_form_closest_approach() {
        const L: f32 = 6.0e5;
        const D: f32 = 4.0e4;
        const U: f32 = 3.0e3;
        let pair = |v0: [f32; 3]| {
            vec![
                ScreenState::new([-L, 0.0, 0.0], v0),
                ScreenState::new([0.0, D, 0.0], [0.0; 3]),
            ]
        };
        let t_star = f64::from(L) / f64::from(U);

        // The window reaches the crossing: the miss distance is the offset alone.
        let got = pairwise_min_approach_cpu(&pair([U, 0.0, 0.0]), (t_star * 2.0) as f32);
        assert!(
            (f64::from(got[0]) - f64::from(D)).abs() < 1.0,
            "crossing pair: {} m against {D} m",
            got[0]
        );

        // Half the window: the clamp stops the approach part way.
        let w = (t_star / 2.0) as f32;
        let got = pairwise_min_approach_cpu(&pair([U, 0.0, 0.0]), w);
        let want =
            ((f64::from(L) - f64::from(U) * f64::from(w)).powi(2) + f64::from(D).powi(2)).sqrt();
        assert!(
            (f64::from(got[0]) - want).abs() < 1.0,
            "clamped pair: {} m against {want} m",
            got[0]
        );

        // Receding, and stationary: both must report the present separation.
        let apart = (f64::from(L).powi(2) + f64::from(D).powi(2)).sqrt();
        for v0 in [[-U, 0.0, 0.0], [0.0; 3]] {
            let got = pairwise_min_approach_cpu(&pair(v0), (t_star * 2.0) as f32);
            assert!(
                (f64::from(got[0]) - apart).abs() < 1.0,
                "v0 {v0:?}: {} m against {apart} m",
                got[0]
            );
        }
    }

    /// Coverage counts a satellite exactly when it clears the elevation mask.
    ///
    /// The serial cross-check shares `normalize`, `sub` and `dot` with the code, so it cannot see a
    /// wrong local vertical or a wrong inner product. Placing satellites at a known elevation above
    /// a known ground point can: `d·up = |d| sin ε` by construction, so the mask is the only thing
    /// that decides the count.
    #[test]
    fn cpu_coverage_follows_the_elevation_mask() {
        let ground = Ecef::new([6.371e6, 0.0, 0.0]);
        let mask = 5.0_f32.to_radians();
        let at_elevation = |deg: f32| {
            let (s, c) = deg.to_radians().sin_cos();
            Ecef::new([6.371e6 + 1.0e6 * s, 1.0e6 * c, 0.0])
        };
        for (deg, want) in [
            (-30.0_f32, 0),
            (0.0, 0),
            (4.9, 0),
            (5.1, 1),
            (60.0, 1),
            (90.0, 1),
        ] {
            let counts = coverage_counts_cpu(&[at_elevation(deg)], &[ground], mask);
            assert_eq!(counts, vec![want], "a satellite at {deg} deg elevation");
        }
        // And the count is a count: three above the mask, two below.
        let sats: Vec<Ecef> = [7.0_f32, 20.0, 80.0, 1.0, -10.0]
            .into_iter()
            .map(at_elevation)
            .collect();
        assert_eq!(coverage_counts_cpu(&sats, &[ground], mask), vec![3]);
    }

    #[test]
    fn cpu_coverage_counts_only_visible_satellites() {
        // Ground point on +x at the surface (local up = +x). One satellite straight up (visible), one
        // antipodal (below the horizon).
        let g = Ecef::new([6.371e6, 0.0, 0.0]);
        let sats = [Ecef::new([7.0e6, 0.0, 0.0]), Ecef::new([-7.0e6, 0.0, 0.0])];
        let counts = coverage_counts_cpu(&sats, &[g], 5.0_f32.to_radians());
        assert_eq!(counts, vec![1]); // overhead visible, antipodal not
    }

    #[test]
    fn gpu_matches_cpu_coverage() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let sats = sphere_points(400, 7.0e6, 0x1234);
        let grid = sphere_points(3000, 6.371e6, 0x9abc);
        let mask = 5.0_f32.to_radians();
        let g = gpu.coverage_counts(&sats, &grid, mask);
        let c = coverage_counts_cpu(&sats, &grid, mask);
        assert_eq!(g.len(), c.len());
        let mut flips = 0u64;
        for (gi, ci) in g.iter().zip(&c) {
            let diff = i64::from(*gi) - i64::from(*ci);
            assert!(
                diff.abs() <= 1,
                "coverage count off by {diff} ({gi} vs {ci})"
            );
            flips += diff.unsigned_abs();
        }
        // Only satellites within f32-ε of the mask elevation can flip; nearly all points agree exactly.
        assert!(
            flips <= (g.len() as u64) / 50,
            "too many borderline flips: {flips} of {}",
            g.len()
        );
    }

    #[test]
    fn gpu_matches_cpu_screen() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let cat = catalogue(5000);
        let g = gpu.pairwise_min_approach(&cat, 300.0);
        let c = pairwise_min_approach_cpu(&cat, 300.0);
        assert_eq!(g.len(), c.len());
        let mut worst_rel = 0.0f64;
        for (gi, ci) in g.iter().zip(&c) {
            let (gi, ci) = (f64::from(*gi), f64::from(*ci));
            let rel = (gi - ci).abs() / ci.max(1.0);
            worst_rel = nan_max(worst_rel, rel);
        }
        // f32 screen vs f64 oracle: agreement to single precision over the O(N²) reduction.
        assert!(
            worst_rel < 1e-3,
            "GPU screen worst relative error {worst_rel:.2e} exceeds 1e-3 vs the f64 CPU oracle"
        );
    }

    const MU_SUN: f64 = 1.327_124_400_18e20;
    const AU: f64 = 1.495_978_707e11;
    const DAY: f64 = 86_400.0;

    /// A heliocentric circular-orbit state at true anomaly `theta` in the ecliptic — synthetic porkchop
    /// bodies without pulling in an ephemeris.
    fn circular(radius: f64, theta: f64) -> ScreenState {
        let speed = (MU_SUN / radius).sqrt();
        ScreenState::new(
            [
                (radius * theta.cos()) as f32,
                (radius * theta.sin()) as f32,
                0.0,
            ],
            [
                (-speed * theta.sin()) as f32,
                (speed * theta.cos()) as f32,
                0.0,
            ],
        )
    }

    #[test]
    fn cpu_porkchop_c3_is_physical() {
        // A near-Hohmann Earth→Mars transfer: Earth departs at 0°, Mars is ~166° around at arrival
        // (avoiding the 180° Lambert singularity), 250-day time of flight. Departure C3 is tens of
        // km²/s² — a sane launch energy, not a degenerate or diverged solve.
        let earth = circular(AU, 0.0);
        let mars = circular(1.524 * AU, 2.9);
        let c3 = porkchop_c3_cpu(
            &[earth],
            &[mars],
            &[0.0],
            &[(250.0 * DAY) as f32],
            MU_SUN as f32,
            true,
        );
        assert_eq!(c3.len(), 1);
        assert!(
            c3[0] > 1.0e6 && c3[0] < 2.0e8,
            "Earth→Mars C3 = {} m²/s² is not a sane launch energy",
            c3[0]
        );
    }

    #[test]
    fn gpu_matches_cpu_porkchop() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        let dep: Vec<ScreenState> = (0..64).map(|k| circular(AU, f64::from(k) * 0.09)).collect();
        let dep_t: Vec<f32> = (0..64)
            .map(|k| (f64::from(k) * 10.0 * DAY) as f32)
            .collect();
        let arr: Vec<ScreenState> = (0..64)
            .map(|k| circular(1.524 * AU, 0.5 + f64::from(k) * 0.04))
            .collect();
        let arr_t: Vec<f32> = (0..64)
            .map(|k| ((160.0 + f64::from(k) * 5.0) * DAY) as f32)
            .collect();
        let mu = MU_SUN as f32;
        let g = gpu.porkchop_c3(&dep, &arr, &dep_t, &arr_t, mu, true);
        let c = porkchop_c3_cpu(&dep, &arr, &dep_t, &arr_t, mu, true);
        assert_eq!(g.len(), c.len());
        let (mut compared, mut worst) = (0u32, 0.0f64);
        for (gi, ci) in g.iter().zip(&c) {
            // Compare only cells both solvers resolved to a meaningful launch energy.
            if *gi < 1e29 && *ci < 1e29 && *ci > 1.0e6 {
                worst = nan_max(
                    worst,
                    (f64::from(*gi) - f64::from(*ci)).abs() / f64::from(*ci),
                );
                compared += 1;
            }
        }
        assert!(
            compared > 200,
            "too few converged cells to compare: {compared}"
        );
        // f32 Lambert vs the f64 oracle: C3 is a difference of ~30 km/s velocities, so allow a few %.
        assert!(
            worst < 0.05,
            "GPU porkchop worst relative C3 error {worst:.3e} exceeds 5% ({compared} cells)"
        );
    }

    /// The near-Earth satellites of the official verification set, as batch elements.
    fn near_earth_sats() -> Vec<batch_sgp4::Sgp4Sat> {
        const CASES: &str = include_str!("../../spacelab-sgp4/tests/fixtures/sgp4_ver_tle.txt");
        CASES
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|line| {
                let f: Vec<&str> = line.split('|').collect();
                let el = sgp4::Elements::from_tle(None, f[0].as_bytes(), f[1].as_bytes()).ok()?;
                batch_sgp4::Sgp4Sat::from_elements(&el)
            })
            .collect()
    }

    /// The batch propagator answers in SI metres, laid out row-major.
    ///
    /// The units are pinned by physics rather than by restating the scale factor: a near-Earth
    /// orbit sits 4e6..5e7 m from the centre and moves at 1e3..1.5e4 m/s, and the same numbers in
    /// kilometres are three orders outside both windows, so no change to the conversion lands
    /// inside them. The lower radius bound is below the Earth's surface on purpose: the
    /// verification set carries decaying objects, and both this port and the `sgp4` crate return
    /// their sub-surface states rather than refusing them (29141 reaches 5250.9 km at t = 745). The layout is pinned by asking for one cell on its own and requiring the batch
    /// to agree with it bit for bit -- a wrong index divides or wraps differently at other shapes.
    #[allow(clippy::float_cmp)] // both calls run the same code on the same inputs
    #[test]
    fn the_cpu_batch_is_metres_in_row_major_order() {
        let sats = near_earth_sats();
        assert!(sats.len() >= 8, "fixture gave {} satellites", sats.len());
        let times = [0.0_f32, 17.5, 96.0, 745.0];
        let out = sgp4_propagate_cpu(&sats, &times);
        assert_eq!(out.len(), sats.len() * times.len());

        let mut checked = 0u32;
        for s in &out {
            let norm = |a: [f32; 3]| dot(f64x3(a), f64x3(a)).sqrt();
            let (r, v) = (norm(s.r), norm(s.v));
            if r == 0.0 {
                continue; // a decayed or diverged cell is zeroed by contract
            }
            assert!(
                (4.0e6..5.0e7).contains(&r),
                "|r| = {r} m is not a near-Earth radius"
            );
            assert!(
                (1.0e3..1.5e4).contains(&v),
                "|v| = {v} m/s is not a near-Earth speed"
            );
            checked += 1;
        }
        assert!(checked >= 24, "only {checked} cells were live");

        for i in 0..sats.len() {
            for j in 0..times.len() {
                let alone = sgp4_propagate_cpu(&sats[i..=i], &times[j..=j]);
                assert_eq!(alone.len(), 1);
                let cell = out[i * times.len() + j];
                assert_eq!(
                    (alone[0].r, alone[0].v),
                    (cell.r, cell.v),
                    "cell ({i}, {j})"
                );
            }
        }
    }

    /// The GPU path's kilometre-to-metre scaling lands where the CPU path's does.
    ///
    /// The kernel emits SGP4's native kilometres in `f32` and [`km_states_to_m`] scales them; the
    /// CPU scales in `f64` and rounds afterwards. The two orders of operations differ by at most an
    /// `f32` ulp, and nothing larger: a batch that came back in the wrong unit would be off by
    /// three orders, not by 1e-7.
    #[test]
    fn the_gpu_scaling_lands_where_the_cpu_path_does() {
        let sats = near_earth_sats();
        let times = [0.0_f32, 33.0, 210.0];
        let cpu = sgp4_propagate_cpu(&sats, &times);
        let kilometres: Vec<ScreenState> = sats
            .iter()
            .flat_map(|s| {
                times.iter().map(move |&t| {
                    s.propagate(f64::from(t)).map_or_else(
                        || ScreenState::new([0.0; 3], [0.0; 3]),
                        |(r, v)| {
                            ScreenState::new(
                                [r[0] as f32, r[1] as f32, r[2] as f32],
                                [v[0] as f32, v[1] as f32, v[2] as f32],
                            )
                        },
                    )
                })
            })
            .collect();
        let scaled = km_states_to_m(kilometres);
        assert_eq!(scaled.len(), cpu.len());
        let mut worst = 0.0_f64;
        for (a, b) in scaled.iter().zip(&cpu) {
            for k in 0..3 {
                for (x, y) in [(a.r[k], b.r[k]), (a.v[k], b.v[k])] {
                    if y != 0.0 {
                        worst = nan_max(worst, (f64::from(x) / f64::from(y) - 1.0).abs());
                    }
                }
            }
        }
        assert!(
            worst < 1e-6,
            "the two scalings differ by {worst:e} relative"
        );
    }

    /// Every public entry point falls back to its CPU reference below the dispatch threshold.
    ///
    /// Exact equality, not approximate: below the threshold the wrapper must not reach the GPU at
    /// all, and the `f32` kernel does not reproduce the `f64` reference bit for bit.
    #[allow(clippy::float_cmp)] // the point of the test is that it is the same code path
    #[test]
    fn the_public_entry_points_fall_back_to_the_cpu_below_the_threshold() {
        let cat = catalogue(64);
        assert!(cat.len() < GPU_MIN_OBJECTS);
        assert_eq!(
            pairwise_min_approach(&cat, 60.0),
            pairwise_min_approach_cpu(&cat, 60.0)
        );

        let sats = sphere_points(8, 7.0e6, 0x4e1f);
        let grid = sphere_points(16, 6.371e6, 0x9c07);
        let mask = 5.0_f32.to_radians();
        assert!(sats.len() * grid.len() < GPU_MIN_COVERAGE_WORK);
        assert_eq!(
            coverage_counts(&sats, &grid, mask),
            coverage_counts_cpu(&sats, &grid, mask)
        );

        let dep = [circular(AU, 0.0)];
        let arr = [circular(1.524 * AU, 2.9)];
        let (dep_t, arr_t) = ([0.0_f32], [(250.0 * DAY) as f32]);
        assert!(dep.len() * arr.len() < GPU_MIN_PORKCHOP_CELLS);
        assert_eq!(
            porkchop_c3(&dep, &arr, &dep_t, &arr_t, MU_SUN as f32, true),
            porkchop_c3_cpu(&dep, &arr, &dep_t, &arr_t, MU_SUN as f32, true)
        );

        let batch = near_earth_sats();
        let times = [0.0_f32, 60.0];
        assert!(batch.len() * times.len() < GPU_MIN_SGP4_CELLS);
        let (a, b) = (
            sgp4_propagate(&batch, &times),
            sgp4_propagate_cpu(&batch, &times),
        );
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(&b) {
            assert_eq!((x.r, x.v), (y.r, y.v));
        }
    }

    /// Closest approach and coverage are Euclidean, so a rigid rotation of the whole scene changes
    /// neither. Applied with three unequal Euler angles, so every component of every difference,
    /// dot product and normalisation carries a distinct value -- an axis-aligned scene lets a
    /// wrong component pass unnoticed.
    #[test]
    fn the_cpu_paths_are_invariant_under_a_rigid_rotation() {
        let (a, b, c) = (0.7_f64, -1.3_f64, 2.1_f64);
        let rot = |p: [f32; 3]| {
            let [x, y, z] = f64x3(p);
            let (x, y) = (a.cos() * x - a.sin() * y, a.sin() * x + a.cos() * y);
            let (y, z) = (b.cos() * y - b.sin() * z, b.sin() * y + b.cos() * z);
            let (z, x) = (c.cos() * z - c.sin() * x, c.sin() * z + c.cos() * x);
            [x as f32, y as f32, z as f32]
        };

        let cat = catalogue(200);
        let turned: Vec<ScreenState> = cat
            .iter()
            .map(|s| ScreenState::new(rot(s.r), rot(s.v)))
            .collect();
        let (before, after) = (
            pairwise_min_approach_cpu(&cat, 300.0),
            pairwise_min_approach_cpu(&turned, 300.0),
        );
        let mut worst = 0.0_f64;
        for (x, y) in before.iter().zip(&after) {
            worst = nan_max(worst, (f64::from(*x) / f64::from(*y) - 1.0).abs());
        }
        assert!(
            worst < 1e-4,
            "the screen turned with the scene: {worst:e} relative"
        );

        // The satellites sit far above and far below a 20-degree mask, so no count is borderline
        // and the two runs must agree exactly.
        let sats = sphere_points(40, 7.4e6, 0x2b6d);
        let grid = sphere_points(60, 6.371e6, 0x81ae);
        let mask = 20.0_f32.to_radians();
        let turn = |p: &[Ecef]| -> Vec<Ecef> { p.iter().map(|g| Ecef::new(rot(g.xyz))).collect() };
        assert_eq!(
            coverage_counts_cpu(&sats, &grid, mask),
            coverage_counts_cpu(&turn(&sats), &turn(&grid), mask),
            "the coverage count turned with the scene"
        );

        // A ground point at the centre of the Earth has no local vertical. It degrades to a zero
        // up-vector -- every line of sight is exactly on its horizon -- rather than to a NaN one,
        // which would silently answer "nothing is visible" for every mask.
        let centre = [Ecef::new([0.0; 3])];
        assert_eq!(
            coverage_counts_cpu(&sats, &centre, 0.0),
            vec![sats.len() as u32]
        );
        assert_eq!(coverage_counts_cpu(&sats, &centre, mask), vec![0]);
    }

    /// A transfer the departure orbit already flies costs nothing to enter.
    ///
    /// Departure and arrival sit on the same circular orbit, separated by exactly the arc it sweeps
    /// in the time of flight, so the Lambert velocity *is* the circular velocity and the departure
    /// C3 is zero. Departure is offset from epoch zero, so a time of flight taken as a sum rather
    /// than a difference asks for a different transfer entirely.
    #[test]
    fn porkchop_c3_vanishes_for_a_transfer_the_departure_orbit_already_flies() {
        let r = AU;
        let tof = 120.0 * DAY;
        let mean_motion = (MU_SUN / (r * r * r)).sqrt();
        let dep_t = 40.0 * DAY;
        // Tilted out of the ecliptic: `circular` builds a planar state, and in the plane the third
        // component of every difference is zero on both sides, so a wrong one cannot show.
        let tilt = |s: ScreenState| {
            let (c, sn) = (0.6_f64.cos(), 0.6_f64.sin());
            let turn = |p: [f32; 3]| {
                let [x, y, z] = f64x3(p);
                [x as f32, (c * y - sn * z) as f32, (sn * y + c * z) as f32]
            };
            ScreenState::new(turn(s.r), turn(s.v))
        };
        let c3 = porkchop_c3_cpu(
            &[tilt(circular(r, 0.0))],
            &[tilt(circular(r, mean_motion * tof))],
            &[dep_t as f32],
            &[(dep_t + tof) as f32],
            MU_SUN as f32,
            true,
        );
        assert_eq!(c3.len(), 1);
        // Against the (29.8 km/s)^2 = 8.9e8 m^2/s^2 scale of the orbit itself.
        assert!(
            f64::from(c3[0]) < 0.5,
            "C3 to re-enter the orbit already flown is {} m^2/s^2",
            c3[0]
        );
    }

    /// A porkchop grid whose departure and arrival axes overlap asks for transfers that arrive
    /// before they leave, and for one that takes no time at all. Every one of those cells is a
    /// non-solution and has to say so: a `tof ≤ 0` cell that came back with a number would be read
    /// as the cheapest transfer on the plot, since a degenerate arc's C3 is not large, it is
    /// arbitrary.
    #[test]
    #[allow(clippy::float_cmp)] // an exact sentinel, and being exactly it is the assertion
    fn a_porkchop_cell_that_arrives_before_it_departs_has_no_solution() {
        let r = AU;
        let mean_motion = (MU_SUN / (r * r * r)).sqrt();
        let times: Vec<f32> = (0..5).map(|k| (f64::from(k) * 30.0 * DAY) as f32).collect();
        let states: Vec<ScreenState> = times
            .iter()
            .map(|t| circular(r, mean_motion * f64::from(*t)))
            .collect();
        let c3 = porkchop_c3_cpu(&states, &states, &times, &times, MU_SUN as f32, true);
        for (i, ti) in times.iter().enumerate() {
            for (j, tj) in times.iter().enumerate() {
                if tj <= ti {
                    assert_eq!(
                        c3[i * times.len() + j],
                        PORKCHOP_NO_SOLUTION,
                        "departing at {ti} for an arrival at {tj} was solved"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_sgp4_matches_the_crate() {
        let Some(gpu) = Gpu::new() else {
            eprintln!("no GPU available, skipping");
            return;
        };
        // Near-Earth NORAD element sets (ISS + LEO objects).
        let tles = [
            (
                "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992",
                "2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008",
            ),
            (
                "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
                "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
            ),
            (
                "1 06251U 62025E   06176.82412014  .00008885  00000-0  12808-3 0  3985",
                "2 06251  58.0579  54.0425 0030035 139.1568 221.1854 15.56387291  6774",
            ),
        ];
        let mut sats = Vec::new();
        let mut refs = Vec::new();
        for (l1, l2) in tles {
            let el = sgp4::Elements::from_tle(None, l1.as_bytes(), l2.as_bytes()).unwrap();
            if let Some(s) = batch_sgp4::Sgp4Sat::from_elements(&el) {
                sats.push(s.to_gpu());
                refs.push(sgp4::Constants::from_elements(&el).unwrap());
            }
        }
        let times: Vec<f32> = (0..96).map(|k| f64::from(k) as f32 * 15.0).collect(); // 0..~24 h
        let g = gpu.sgp4_propagate(&sats, &times); // km
        let mut worst = 0.0f64;
        for (i, reference) in refs.iter().enumerate() {
            for (j, &t) in times.iter().enumerate() {
                let p = reference
                    .propagate(sgp4::MinutesSinceEpoch(f64::from(t)))
                    .unwrap();
                let st = g[i * times.len() + j];
                let dr = ((f64::from(st.r[0]) - p.position[0]).powi(2)
                    + (f64::from(st.r[1]) - p.position[1]).powi(2)
                    + (f64::from(st.r[2]) - p.position[2]).powi(2))
                .sqrt();
                worst = nan_max(worst, dr);
            }
        }
        // f32 SGP4 over ~24 h vs the f64 crate: agreement to well within SGP4's own ~km accuracy.
        assert!(
            worst < 2.0,
            "GPU SGP4 worst position error {worst:.3} km over 24 h"
        );
    }
}
