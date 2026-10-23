//! The coarse cull: a conservative spherical cap around the satellite direction.
//!
//! Before the exact per-cell predicate runs, most cells are nowhere near the footprint and can be
//! rejected by a single dot product. The cull's one hard requirement is that it **never drops a
//! covered cell** — a false negative here is a wrong answer, not a slow one. So the cap is derived
//! to be provably larger than any footprint, and the margin is a bound, not a guess.
//!
//! # The cap, derived
//!
//! The footprint half-angle on a sphere of radius `R`, satellite at `r_s = L·R`, mask `s = sin a`:
//!
//! ```text
//! cos(λ_max) = (s·u + 1 − s²) / L,   u = √(L² − 1 + s²)
//! ```
//!
//! proven equal to `SaVi`'s closed form in `derivations/footprint3.mac`. Two facts make a
//! conservative cap out of it:
//!
//! * **λ_max increases with L.** `derivations/cull_bound.mac` shows `d(cos λ_max)/dL < 0`. Every
//!   point on the WGS-84 ellipsoid has geocentric radius ≥ `R_polar`, so using `R_polar` gives the
//!   largest `L`, hence the largest λ_max, hence a cap no smaller than the true footprint. This
//!   handles the radius variation of the ellipsoid.
//! * **The geodetic zenith is not radial.** Elevation is measured against the geodetic horizon,
//!   which tilts from the geocentric radial by the vertical deflection, at most
//!   `δ_max = atan(e²/(2√(1−e²))) = 0.19242°` (at 45° latitude). Adding `δ_max` to the cap absorbs
//!   it. `derivations/cull_margin.m` falsifies the whole construction over 3.2×10⁶ ellipsoid cells:
//!   the worst covered cell sits 0.107° beyond the spherical λ_max — inside the 0.19° margin.
//!
//! So the cap half-angle is `λ_cap = λ_max(r_s / R_polar, sensor) + δ_max`.

#![allow(clippy::doc_markdown)] // λ_max, δ_max etc. are mathematics, not identifiers

use spacelab_core::Vec3;

use crate::{And, Cone, ElevationMask, Not, Or, SensorError, ellipsoid_grid::EllipsoidGrid};

/// Ellipsoid facts the cull needs, kept here so `spacelab-sensors` stays body-agnostic (the same
/// engine will one day cull the Moon). Construct with `CullBody::wgs84` for Earth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CullBody {
    /// Smallest geocentric radius of the body's surface, metres. `R_polar` for an oblate ellipsoid.
    pub min_radius_m: f64,
    /// Upper bound on the vertical deflection (geodetic normal vs geocentric radial), radians.
    pub deflection_margin_rad: f64,
}

impl CullBody {
    /// WGS-84 Earth.
    ///
    /// `min_radius` is the polar radius. `deflection_margin` is `0.0034` rad, which exceeds the
    /// exact maximum vertical deflection `atan(e²/(2√(1−e²))) = 0.00335844` rad — verified in
    /// `tests/cull.rs` to be ≥ the value recomputed from the WGS-84 flattening. The ~10 m of slack
    /// over the exact bound costs a fractionally larger candidate set and buys certainty.
    pub const WGS84: Self = Self {
        min_radius_m: 6_356_752.314_245_179,
        deflection_margin_rad: 0.0034,
    };
}

/// Conservative cap half-angle for one nadir constraint at a given satellite radius.
///
/// Every constraint's cap is an *upper bound* on the Earth-central angle from the sub-satellite
/// point to any cell it can cover. The default is unusable (`π`, no cull); each concrete constraint
/// overrides it. Composition follows set logic: AND shrinks to the smaller cap, OR grows to the
/// larger, NOT gives up (`π`).
pub trait Reach {
    /// The cap half-angle in radians, clamped to `[0, π]`. `π` means "cull nothing".
    fn reach(&self, sat_radius_m: f64, body: CullBody) -> f64;
}

/// Footprint half-angle for the mask on a sphere at radius ratio `l`, plus nothing. `s = sin a`.
fn elevation_lambda(l: f64, s: f64) -> f64 {
    if l <= 1.0 {
        return core::f64::consts::PI;
    }
    let u = libm::sqrt(l * l - 1.0 + s * s);
    let coslam = (s * u + 1.0 - s * s) / l;
    libm::acos(coslam.clamp(-1.0, 1.0))
}

impl Reach for ElevationMask {
    fn reach(&self, sat_radius_m: f64, body: CullBody) -> f64 {
        // Largest L -> largest footprint -> conservative (derivations/cull_bound.mac).
        let l = sat_radius_m / body.min_radius_m;
        (elevation_lambda(l, self.sin_min()) + body.deflection_margin_rad)
            .min(core::f64::consts::PI)
    }
}

impl Reach for Cone {
    /// The **near-side forward** footprint cap. The bare [`Cone`] predicate also matches cells
    /// behind the limb (it "sees through the Earth" — `cone.rs`), which no physical sensor does; a
    /// cone is only ever used as `And(Cone, ElevationMask::HORIZON)` or with another visibility
    /// term. This cap is conservative for that composed, near-side use, and `And`'s `min` makes the
    /// composite cap the tighter of the two. Culling a bare `Cone` alone is not a supported query.
    fn reach(&self, sat_radius_m: f64, body: CullBody) -> f64 {
        let l = sat_radius_m / body.min_radius_m;
        if l <= 1.0 {
            return core::f64::consts::PI;
        }
        let ca = self.cos_half();
        let sa = libm::sqrt((1.0 - ca * ca).max(0.0));
        // `SaVi` cone footprint radius rho = sin(lambda_max), horizon-limited past L = 1/sin a.
        let rho = if sa > 0.0 && l >= 1.0 / sa {
            libm::sqrt(l * l - 1.0) / l
        } else {
            sa * (l * ca - libm::sqrt((1.0 - l * l * sa * sa).max(0.0)))
        };
        (libm::asin(rho.clamp(-1.0, 1.0)) + body.deflection_margin_rad).min(core::f64::consts::PI)
    }
}

impl<A: Reach, B: Reach> Reach for And<A, B> {
    fn reach(&self, sat_radius_m: f64, body: CullBody) -> f64 {
        // The intersection lies inside the smaller footprint.
        self.0
            .reach(sat_radius_m, body)
            .min(self.1.reach(sat_radius_m, body))
    }
}

impl<A: Reach, B: Reach> Reach for Or<A, B> {
    fn reach(&self, sat_radius_m: f64, body: CullBody) -> f64 {
        // A cell covered by either operand lies inside at least one footprint.
        self.0
            .reach(sat_radius_m, body)
            .max(self.1.reach(sat_radius_m, body))
    }
}

impl<A: Reach> Reach for Not<A> {
    fn reach(&self, _sat_radius_m: f64, _body: CullBody) -> f64 {
        // The complement of a cap is not a cap. Cull nothing.
        core::f64::consts::PI
    }
}

/// The cosine threshold for the cull kernel, given a cap half-angle. Precomputed once per
/// satellite, not per cell.
#[must_use]
pub fn cap_cosine(reach_rad: f64) -> f64 {
    libm::cos(reach_rad)
}

/// The geocentric-latitude half-band a *circular* orbit's footprint can ever reach: cells with
/// `|latitude| >` this are never in view over any horizon and can skip the access-window search
/// entirely. Radians.
///
/// The cull is the instantaneous cap [`cull_into`] applied to the whole orbit at once. For a circular
/// orbit the sub-satellite geocentric latitude satisfies `|φ_sub| ≤ min(i, π−i)`, and a covered cell
/// lies within the footprint cap [`Reach::reach`] of the sub-satellite point, so
/// `|φ_cell| ≤ min(i, π−i) + λ_cap`. Conservative by the same margins as `cull_into` — it never
/// excludes a coverable cell — and clamped to `π/2` (a band reaching the pole means "cull nothing").
#[must_use]
pub fn reachable_band(
    inclination_rad: f64,
    sat_radius_m: f64,
    sensor: &impl Reach,
    body: CullBody,
) -> f64 {
    use core::f64::consts::{FRAC_PI_2, PI};
    let inc = inclination_rad.abs();
    let max_sublat = inc.min(PI - inc);
    (max_sublat + sensor.reach(sat_radius_m, body)).min(FRAC_PI_2)
}

/// Mark every cell inside the cap with `1`, the rest with `0`.
///
/// A cell is inside the cap when the geocentric angle between it and the satellite is `≤ λ_cap`,
/// i.e. `cos∠ ≥ cos_cap`. With `cos∠ = (r_sat·r_cell)/(|r_sat||r_cell|)` this is, for `cos_cap ≥ 0`
/// and requiring the same hemisphere, the square-root-free test
///
/// ```text
/// (r_sat·r_cell) ≥ 0   ∧   (r_sat·r_cell)² ≥ cos_cap² · |r_sat|² · |r_cell|²
/// ```
///
/// When `cos_cap ≤ 0` the cap spans at least a hemisphere and the squaring is unsafe, so every
/// cell is kept — correct, and the cull simply saves nothing there (a very low satellite or a
/// `Not`).
///
/// # Errors
///
/// [`SensorError::OutputLengthMismatch`] if `out` is not one byte per cell.
#[allow(clippy::needless_range_loop)]
pub fn cull_into(
    satellite: Vec3,
    cos_cap: f64,
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

    let (rx, ry, rz) = (satellite.x, satellite.y, satellite.z);
    let n2_sat = rx * rx + ry * ry + rz * rz; // hoisted

    if cos_cap <= 0.0 {
        out.fill(1);
        return Ok(());
    }
    let c2 = cos_cap * cos_cap;

    for i in 0..n {
        let (cx, cy, cz) = (px[i], py[i], pz[i]);
        let dot = rx * cx + ry * cy + rz * cz;
        let n2_cell = cx * cx + cy * cy + cz * cz;
        out[i] = u8::from(dot >= 0.0 && dot * dot >= c2 * n2_sat * n2_cell);
    }
    Ok(())
}
