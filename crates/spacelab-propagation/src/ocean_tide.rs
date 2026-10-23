//! IERS 2010 ocean-tide geopotential perturbation (IERS Conventions 2010, §6.3, Eq. 6.15).
//!
//! Ocean mass redistributed by the tides perturbs the geopotential. Each tidal constituent (one
//! Doodson wave) contributes a time-periodic change to the normalized Stokes coefficients,
//!
//! ```text
//! ΔC̄nm += (C⁺nm + C⁻nm)·cos θ + (S⁺nm + S⁻nm)·sin θ
//! ΔS̄nm += (S⁺nm − S⁻nm)·cos θ − (C⁺nm − C⁻nm)·sin θ
//! ```
//!
//! summed over all waves, where `θ = c_γ·γ + c_l·l + c_l′·l′ + c_F·F + c_D·D + c_Ω·Ω`, `γ = GMST + π`,
//! and `(l, l′, F, D, Ω)` are the five Delaunay arguments. The per-wave multipliers `c_·` come from
//! decoding the Doodson number into six Doodson arguments and mapping those to Delaunay space (see
//! `OceanTide::delta_cs`) — the exact recipe Orekit's `OceanTidesWave` uses. `C±/S±` are the
//! prograde(+)/retrograde(−) coefficients from the tide table, in units of the table's
//! [`crate::ocean_tide_fes2004::SCALE`].
//!
//! Like [`crate::solid_tide::SolidEarthTide`], this is a peer [`Acceleration`] superposed on the
//! static [`GeoPotential`]: the ΔC̄/ΔS̄ track sidereal time and so change every step. The gradient is
//! taken by *reusing* the tested [`GeoPotential`] recursion — a field to the caller's chosen degree
//! (`max_degree`) carrying only the ΔC̄/ΔS̄ (plus the forced monopole) is built each step and the
//! monopole −μr/r³ subtracted, leaving the tidal acceleration. The satellite is rotated GCRF↔ITRF
//! here (the tidal field is body-fixed) via the caller's `rotate` closure; the caller also owns the
//! time scales and supplies GMST and the Delaunay arguments through `delaunay_gmst`.
//!
//! Unlike the solid tide there is **no** permanent-tide subtraction — the ocean tide has no
//! zero-frequency permanent part folded into the background field. This module models the
//! ocean-tide *waves* only; the ocean **pole** tide (IERS §6.4) is a separate, out-of-scope effect.
//!
//! # Minor-wave admittance
//!
//! The explicit file gives only ~18 *major* waves. The many *minor* constituents are recovered by
//! **admittance inference** (IERS §6.3.2; CNES aviso-fes): the admittance `Z(f) = coef / H_eq(f)`
//! (H_eq = Cartwright equilibrium amplitude) is smooth in tidal frequency within each species, so a
//! minor's coefficient is interpolated from a few reference majors and rescaled by its own `H_eq`.
//! Every interpolation is *real*-weighted, so per minor `M` it reduces to `coef(M) = Σ Wₖ·coef(majorₖ)`
//! with real `Wₖ` — identical for each channel `(C⁺,S⁺,C⁻,S⁻)` at every `(n,m)`, the complex
//! geopotential coefficient playing the role of the complex elevation amplitude. The weights are the
//! generated [`crate::ocean_tide_admittance`] tables ([`ADMITTANCE_SPLINE`], the FES default, or
//! [`ADMITTANCE_LINEAR`], IERS/GOT-style); each inferred wave adds an Eq. 6.15 term under its own
//! Doodson argument. Setting `admittance = &[]` (the default) keeps the Orekit-exact explicit-only
//! path. Weights are validated against `pyfes` (see `experiments/ocean_tide_admittance_verify.py`).
//!
//! [`ADMITTANCE_SPLINE`]: crate::ocean_tide_admittance::ADMITTANCE_SPLINE
//! [`ADMITTANCE_LINEAR`]: crate::ocean_tide_admittance::ADMITTANCE_LINEAR
//!
//! # Table-agnostic
//!
//! The wave table is data: [`OceanTide`] takes any `&'static [OceanTideWave]` plus its `scale` and a
//! runtime `max_degree`. The bundled [`crate::ocean_tide_fes2004`] (FES2004, degree 6) is the
//! reproducible Orekit gate; [`crate::ocean_tide_fes2014`] (FES2014b) is the modern production table,
//! full to degree/order 100 — the same Doodson-keyed prograde/retrograde shape, evaluated to whatever
//! `max_degree` the caller sets (0..=`GRID_MAX`) with no recompile. Another model (GOT5.5, …) drops in
//! by machine-generating a sibling table (same generator shape) and passing it here. Modern models are
//! the intended production default; FES2004 at degree 6 stays as the Orekit gate.
//!
//! # Provenance
//!
//! Algorithm: IERS Conventions 2010 Eq. 6.15, ported verbatim from Orekit's
//! `org.orekit.forces.gravity.potential.OceanTidesWave.addContribution` (Doodson decode, Doodson→
//! Delaunay map, and the ΔC̄/ΔS̄ recurrence). Coefficients: `derivations/ocean_tide_fes2004.py` from
//! the FES2004 `fes2004_Cnm-Snm.dat` (degree-6 gate), and `derivations/ocean_tide_fes2014.py` from the
//! FES2014b `fes2014b_Cnm-Snm.txt` (full degree/order 100). Validated against Orekit's `OceanTides` in
//! the tests.

// Dense IERS/geodesy notation (ΔC̄_nm, C⁺nm, θ) and the domain's Doodson/Delaunay argument names
// (cS/cPs, cL/cLPrime, cNPrime) — kept verbatim from Orekit, so they read alike by construction.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::doc_markdown,
    // The ΔC̄/ΔS̄ grid, admittance channels, and flattened coefficients are fixed GRID_MAX-sized
    // stack arrays by design (no_std, no alloc); the runtime degree bounds the work, not the size.
    clippy::large_stack_arrays
)]

use spacelab_autodiff::Scalar;

use crate::GeoPotential;
use crate::forces::Acceleration;
use crate::ocean_tide_admittance::InferredWave;
use crate::ocean_tide_fes2004::OceanTideWave;

/// Highest degree/order the fixed-size grids can hold, sizing every stack array here. Matches the
/// geopotential's [`GeoPotential::MAX_DEGREE`] so any table can be evaluated to the field's ceiling
/// without heap allocation; each [`OceanTide`] picks its actual evaluation degree at runtime via
/// [`OceanTide::max_degree`] (`0..=GRID_MAX`), so the FES2004 gate stays degree 6 while FES2014b can
/// run to 100+. The grids grow to `(GRID_MAX+1)²`; that is ~0.23 MB for `[[[f64; 2]; …]; …]`, on the
/// stack, no alloc — the same fixed-array-by-degree trade the [`GeoPotential`] itself makes.
const GRID_MAX: usize = GeoPotential::MAX_DEGREE;
/// Number of `(n, m)` cells with `2 ≤ n ≤ GRID_MAX`, `0 ≤ m ≤ n` — the flattened-coefficient bound.
const N_COEFFS: usize = count_cells(GRID_MAX);

/// Count the `(n, m)` cells with `2 ≤ n ≤ max_degree`, `0 ≤ m ≤ n` (`Σ_{n=2}^{max} (n+1)`). `const`
/// so it sizes the fixed coefficient array; matches the runtime flatten loop in [`OceanTide`].
const fn count_cells(max_degree: usize) -> usize {
    let mut count = 0;
    let mut n = 2;
    while n <= max_degree {
        count += n + 1;
        n += 1;
    }
    count
}

/// IERS 2010 ocean-tide geopotential perturbation as a superposed acceleration.
///
/// `mu_earth`/`re_earth` must equal the static [`GeoPotential`]'s `μ`/`Rₑ`. `waves`/`scale` are the
/// tide table (e.g. [`crate::ocean_tide_fes2004::FES2004_WAVES`] and
/// [`crate::ocean_tide_fes2004::SCALE`]). `max_degree` is the highest degree/order to evaluate
/// (`0..=GRID_MAX`): the FES2004 gate uses 6 (Orekit-exact), a modern table like
/// [`crate::ocean_tide_fes2014::FES2014B_WAVES`] uses up to 100. `rotate` gives the GCRF→ITRF matrix
/// (row-major) at time `t` (seconds from epoch); `delaunay_gmst` returns `(GMST, [l, l′, F, D, Ω])` in
/// radians at `t` — the caller owns the UT1/TT time scales, exactly as for
/// [`crate::solid_tide::SolidEarthTide`].
pub struct OceanTide<R, F> {
    /// Earth gravitational parameter, m³/s² (the geopotential's `μ`).
    pub mu_earth: f64,
    /// Earth reference radius, m (the geopotential's `Rₑ`).
    pub re_earth: f64,
    /// Ocean-tide waves (Doodson-keyed prograde/retrograde coefficient grids).
    pub waves: &'static [OceanTideWave],
    /// Coefficient scale for `waves` (FES2004: 10⁻¹¹, [`crate::ocean_tide_fes2004::SCALE`]).
    pub scale: f64,
    /// Highest degree/order to evaluate, `0..=GRID_MAX`. Rows in `waves`/`admittance` with degree
    /// above this are skipped and the field is built only to here — so one table can be run at any
    /// resolution the caller needs (FES2004 gate at 6, FES2014b at 100) without a recompile. Values
    /// above `GRID_MAX` are a programming error (the fixed grids are sized to `GRID_MAX`).
    pub max_degree: usize,
    /// Admittance-inferred minor waves ([`crate::ocean_tide_admittance`]). `&[]` disables inference
    /// (explicit-wave-only, the Orekit-exact default); `ADMITTANCE_SPLINE` (FES) or `ADMITTANCE_LINEAR`
    /// (IERS/GOT-style) add minor constituents whose coefficients are inferred from `waves`.
    pub admittance: &'static [InferredWave],
    /// GCRF→ITRF rotation (row-major) at time `t`, seconds from the propagation epoch.
    pub rotate: R,
    /// `(GMST, [l, l′, F, D, Ω])` in radians at time `t`, seconds from epoch.
    pub delaunay_gmst: F,
}

/// Tidal argument θ of one Doodson wave: decode the six Doodson multipliers, map to Delaunay space,
/// and dot with `(γ, l, l′, F, D, Ω)`. Verbatim from Orekit's `OceanTidesWave` (Doodson decode +
/// Doodson→Delaunay map), shared by the explicit and admittance-inferred waves.
#[must_use]
fn doodson_theta(doodson: u32, gamma: f64, delaunay: [f64; 5]) -> f64 {
    let [l, l_prime, f, d, omega] = delaunay;
    let doodson = i64::from(doodson);
    // Doodson number → six Doodson arguments (Orekit OceanTidesWave constructor).
    let c_ps = (doodson % 10) as i32 - 5;
    let c_n_prime = ((doodson / 10) % 10) as i32 - 5;
    let c_p = ((doodson / 100) % 10) as i32 - 5;
    let c_h = ((doodson / 1000) % 10) as i32 - 5;
    let c_s = ((doodson / 10000) % 10) as i32 - 5;
    let c_tau = ((doodson / 100_000) % 10) as i32;
    // Doodson arguments → Delaunay multipliers.
    let c_gamma = c_tau;
    let c_l = -c_p;
    let c_l_prime = -c_ps;
    let c_f = -c_tau + c_s + c_h + c_p + c_ps;
    let c_d = -c_h - c_ps;
    let c_omega = -c_tau + c_s + c_h + c_p - c_n_prime + c_ps;
    f64::from(c_gamma) * gamma
        + f64::from(c_l) * l
        + f64::from(c_l_prime) * l_prime
        + f64::from(c_f) * f
        + f64::from(c_d) * d
        + f64::from(c_omega) * omega
}

impl<R, F> OceanTide<R, F> {
    /// Ocean-tide ΔC̄nm, ΔS̄nm (Eq. 6.15) summed over all waves, as `grid[n][m] = [ΔC̄nm, ΔS̄nm]` for
    /// `2 ≤ n ≤ self.max_degree`. The grid is sized to `GRID_MAX` (fixed, no alloc) but only cells up
    /// to `self.max_degree` are filled — higher-degree rows in `waves` are skipped, so the same table
    /// evaluates at whatever resolution the caller set. Depends only on time through `gmst` and the
    /// Delaunay arguments `[l, l′, F, D, Ω]` (all radians); the satellite position enters only later,
    /// in the gradient. When `self.admittance` is non-empty, each inferred minor wave adds its own
    /// Eq. 6.15 term, its prograde/retrograde coefficients the real-weighted sum of its majors' (below).
    #[must_use]
    fn delta_cs(&self, gmst: f64, delaunay: [f64; 5]) -> [[[f64; 2]; GRID_MAX + 1]; GRID_MAX + 1] {
        let gamma = gmst + core::f64::consts::PI;
        let mut grid = [[[0.0_f64; 2]; GRID_MAX + 1]; GRID_MAX + 1];
        for wave in self.waves {
            let theta = doodson_theta(wave.doodson, gamma, delaunay);
            let (sin, cos) = libm::sincos(theta);
            for &(n, m, c_plus, s_plus, c_minus, s_minus) in wave.coeffs {
                let n = n as usize;
                if n > self.max_degree {
                    continue;
                }
                let m = m as usize;
                let c_plus = c_plus * self.scale;
                let s_plus = s_plus * self.scale;
                let c_minus = c_minus * self.scale;
                let s_minus = s_minus * self.scale;
                let cell = &mut grid[n][m];
                cell[0] += (c_plus + c_minus) * cos + (s_plus + s_minus) * sin;
                cell[1] += (s_plus - s_minus) * cos - (c_plus - c_minus) * sin;
            }
        }
        self.add_admittance(gamma, delaunay, &mut grid);
        grid
    }

    /// Add the admittance-inferred minor waves' Eq. 6.15 terms into `grid`. For each inferred wave the
    /// four coefficient channels `(C⁺,S⁺,C⁻,S⁻)ₙₘ` are assembled as `Σ weight·(reference major)ₙₘ`
    /// (real weights from [`crate::ocean_tide_admittance`]), then folded in under the minor's own
    /// Doodson argument. A reference major absent from `self.waves` contributes nothing.
    fn add_admittance(
        &self,
        gamma: f64,
        delaunay: [f64; 5],
        grid: &mut [[[f64; 2]; GRID_MAX + 1]; GRID_MAX + 1],
    ) {
        for inferred in self.admittance {
            // Inferred (C⁺, S⁺, C⁻, S⁻) per (n, m), in the table's coefficient units (scale applied
            // below, exactly as for the explicit waves).
            let mut chan = [[[0.0_f64; 4]; GRID_MAX + 1]; GRID_MAX + 1];
            for &(ref_doodson, weight) in inferred.refs {
                let Some(major) = self.waves.iter().find(|w| w.doodson == ref_doodson) else {
                    continue;
                };
                for &(n, m, c_plus, s_plus, c_minus, s_minus) in major.coeffs {
                    let n = n as usize;
                    if n > self.max_degree {
                        continue;
                    }
                    let cell = &mut chan[n][m as usize];
                    cell[0] += weight * c_plus;
                    cell[1] += weight * s_plus;
                    cell[2] += weight * c_minus;
                    cell[3] += weight * s_minus;
                }
            }
            let theta = doodson_theta(inferred.doodson, gamma, delaunay);
            let (sin, cos) = libm::sincos(theta);
            #[allow(clippy::needless_range_loop)]
            for n in 2..=self.max_degree {
                for m in 0..=n {
                    let [c_plus, s_plus, c_minus, s_minus] = chan[n][m];
                    let c_plus = c_plus * self.scale;
                    let s_plus = s_plus * self.scale;
                    let c_minus = c_minus * self.scale;
                    let s_minus = s_minus * self.scale;
                    let cell = &mut grid[n][m];
                    cell[0] += (c_plus + c_minus) * cos + (s_plus + s_minus) * sin;
                    cell[1] += (s_plus - s_minus) * cos - (c_plus - c_minus) * sin;
                }
            }
        }
    }
}

impl<R: Fn(f64) -> [[f64; 3]; 3], F: Fn(f64) -> (f64, [f64; 5])> Acceleration for OceanTide<R, F> {
    fn acceleration<S: Scalar>(&self, t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let m = (self.rotate)(t.value());
        let e = |i: usize, j: usize| S::from_f64(m[i][j]);
        // r_itrf = M · r_gcrf (the tidal field is body-fixed).
        let rf = [
            e(0, 0) * r[0] + e(0, 1) * r[1] + e(0, 2) * r[2],
            e(1, 0) * r[0] + e(1, 1) * r[1] + e(1, 2) * r[2],
            e(2, 0) * r[0] + e(2, 1) * r[1] + e(2, 2) * r[2],
        ];

        let (gmst, delaunay) = (self.delaunay_gmst)(t.value());
        let grid = self.delta_cs(gmst, delaunay);
        // Flatten the used triangle (2 ≤ n ≤ self.max_degree, 0 ≤ m ≤ n) into GeoPotential input;
        // `n`/`m` are the coefficients' degree/order (they go into each tuple), not merely grid
        // indices. `n_used` counts the cells actually written, so only that prefix is passed on.
        let mut coeffs = [(0usize, 0usize, 0.0_f64, 0.0_f64); N_COEFFS];
        let mut n_used = 0;
        #[allow(clippy::needless_range_loop)]
        for n in 2..=self.max_degree {
            for m in 0..=n {
                coeffs[n_used] = (n, m, grid[n][m][0], grid[n][m][1]);
                n_used += 1;
            }
        }

        let zero = [S::from_f64(0.0); 3];
        // The tidal field carries a forced monopole (C̄₀₀ = 1); the tidal acceleration is the field's
        // acceleration minus the monopole −μr/r³, i.e. plus μr/r³ (same as SolidEarthTide).
        let a_itrf = match GeoPotential::from_normalized(
            self.mu_earth,
            self.re_earth,
            self.max_degree,
            &coeffs[..n_used],
        ) {
            Some(geo) => {
                let full = geo.acceleration::<S>(S::from_f64(0.0), &rf, &zero);
                let r2 = rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2];
                let mu_inv_r3 = S::from_f64(self.mu_earth) * (r2 * r2.sqrt()).recip();
                [
                    full[0] + rf[0] * mu_inv_r3,
                    full[1] + rf[1] * mu_inv_r3,
                    full[2] + rf[2] * mu_inv_r3,
                ]
            }
            None => zero,
        };
        // a_gcrf = Mᵀ · a_itrf
        [
            e(0, 0) * a_itrf[0] + e(1, 0) * a_itrf[1] + e(2, 0) * a_itrf[2],
            e(0, 1) * a_itrf[0] + e(1, 1) * a_itrf[1] + e(2, 1) * a_itrf[2],
            e(0, 2) * a_itrf[0] + e(1, 2) * a_itrf[1] + e(2, 2) * a_itrf[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::type_complexity,
        // The FES2014b spot-check compares `delta_cs` against a from-scratch recompute that runs the
        // identical Eq. 6.15 operations in the identical wave order — so the match is exact by design.
        clippy::float_cmp
    )]
    use super::*;
    use crate::ocean_tide_fes2004::{FES2004_WAVES, SCALE};

    // Orekit `OceanTides` (FES2004, IERS 2010, degree = order = 6, no pole tide, no interpolation) at
    // 2024-01-01T00:00:00 UTC — emitted by `tools/orekit-oracle` into `orekit_ocean_tide.csv`. The
    // GMST and five Delaunay arguments are Orekit's own (BodiesElements), so the Rust side drives the
    // identical θ; the ΔC̄/ΔS̄ and acceleration are then a pure port comparison.
    const GMST: f64 = 1.588_276_262_128_853_2e2;
    const DELAUNAY: [f64; 5] = [
        2.001_127_675_839_614_3e3,
        1.570_239_420_028_980_5e2,
        2.025_544_516_208_826e3,
        1.870_222_765_770_187_2e3,
        -5.918_788_376_630_828,
    ];
    // Orekit's ΔC̄nm/ΔS̄nm as `(n, m, ΔC̄, ΔS̄)`, degree 2..6 (the 25-cell FES2004 gate).
    const OREKIT_DCS: [(usize, usize, f64, f64); count_cells(6)] = [
        (
            2,
            0,
            8.666_130_906_763_387e-11,
            -5.779_563_670_386_048_6e-11,
        ),
        (
            2,
            1,
            -3.589_491_203_179_547_5e-11,
            3.215_670_192_256_671_6e-11,
        ),
        (
            2,
            2,
            -2.368_089_474_826_122_5e-10,
            -4.075_251_486_829_836_5e-10,
        ),
        (
            3,
            0,
            1.632_358_869_790_170_5e-10,
            1.214_024_859_748_792_6e-10,
        ),
        (3, 1, -5.767_054_465_199_161e-10, -8.382_194_967_377_437e-11),
        (
            3,
            2,
            4.449_670_626_697_701e-11,
            -2.560_708_601_621_238_2e-11,
        ),
        (
            3,
            3,
            -6.186_012_283_990_516e-10,
            1.357_097_903_718_008_3e-11,
        ),
        (4, 0, -7.747_568_215_109_409e-11, -7.088_010_683_345_709e-11),
        (4, 1, -5.648_333_470_154_019e-10, -3.904_614_322_735_742e-11),
        (
            4,
            2,
            1.632_797_775_008_947_7e-10,
            3.461_413_884_569_123_5e-10,
        ),
        (
            4,
            3,
            2.426_844_841_552_219_5e-10,
            2.863_805_627_712_946_3e-10,
        ),
        (
            4,
            4,
            9.354_054_238_621_524e-11,
            -3.658_893_671_278_213_4e-10,
        ),
        (5, 0, -9.088_577_019_074_235e-11, -6.655_543_710_381_15e-11),
        (5, 1, 5.698_287_296_374_181e-10, -1.194_454_936_120_526e-10),
        (5, 2, 7.244_131_102_107_734e-11, 2.988_001_194_306_748_6e-10),
        (
            5,
            3,
            -6.148_255_630_432_63e-11,
            -2.361_714_485_267_578_5e-10,
        ),
        (
            5,
            4,
            -2.742_890_154_002_806_6e-10,
            2.723_151_016_092_099_7e-10,
        ),
        (
            5,
            5,
            1.829_211_793_851_293_4e-10,
            -2.226_201_502_903_791_2e-10,
        ),
        (
            6,
            0,
            -3.924_682_435_825_514_4e-11,
            1.068_919_385_873_082_8e-10,
        ),
        (6, 1, 8.291_366_205_754_389e-11, -8.612_704_894_225_254e-12),
        (
            6,
            2,
            -1.482_149_168_650_079e-10,
            -1.128_717_055_905_799_1e-10,
        ),
        (
            6,
            3,
            -1.353_871_895_556_656e-10,
            4.470_373_900_058_295_4e-11,
        ),
        (
            6,
            4,
            -3.065_361_224_949_125e-11,
            5.504_941_397_032_704_5e-11,
        ),
        (
            6,
            5,
            -3.052_425_497_913_347e-11,
            -1.827_269_139_941_464_4e-10,
        ),
        (6, 6, -3.573_981_005_964_112e-10, 2.646_396_224_467_273e-10),
    ];
    const SAT_ITRF: [f64; 3] = [
        860_234.487_436_545,
        -6_739_356.482_322_041,
        3_015_140.267_186_367,
    ];
    const A_TIDE_ITRF: [f64; 3] = [
        2.498_773_707_099_302e-9,
        2.713_406_085_950_795e-9,
        -2.849_702_724_339_571_4e-8,
    ];
    const MU_EARTH: f64 = 3.986_004_415e14;
    const RE_EARTH: f64 = 6.378_136_46e6;

    fn fes2004_tide() -> OceanTide<impl Fn(f64) -> [[f64; 3]; 3], impl Fn(f64) -> (f64, [f64; 5])> {
        OceanTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            waves: FES2004_WAVES,
            scale: SCALE,
            max_degree: 6, // the Orekit gate: degree = order = 6
            admittance: &[],
            rotate: |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            delaunay_gmst: |_t: f64| (GMST, DELAUNAY),
        }
    }

    /// The ocean-tide ΔC̄nm/ΔS̄nm (Eq. 6.15) match Orekit's `OceanTidesWave.addContribution` given
    /// Orekit's own GMST + Delaunay arguments. This is a pure deterministic sum with no frame/gradient
    /// path, so it is near-bit-exact; the residual is only libm-vs-Hipparchus `sincos` and the ±π
    /// reconstruction of γ (the emitted GMST is `getGamma() − π`).
    #[test]
    fn delta_cs_matches_orekit() {
        let tide = fes2004_tide();
        let grid = tide.delta_cs(GMST, DELAUNAY);
        let mut max_rel = 0.0_f64;
        for &(n, m, want_c, want_s) in &OREKIT_DCS {
            for (got, want) in [(grid[n][m][0], want_c), (grid[n][m][1], want_s)] {
                let rel = (got - want).abs() / want.abs();
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 1e-9,
                    "ΔC̄/ΔS̄ ({n},{m}): got {got:e}, want {want:e}, rel {rel:.3e}"
                );
            }
        }
        std::eprintln!("ocean tide ΔC̄/ΔS̄ vs Orekit: max rel {max_rel:.3e}");
    }

    /// The ocean-tide acceleration matches Orekit's `OceanTides` (FES2004, no pole tide, no
    /// interpolation) at the emitted ITRF geometry (identity rotation: the state is already in the
    /// ITRF). ΔC̄/ΔS̄ are bit-exact (above), so the ~5e-8 relative residual is NOT the coefficient sum
    /// and NOT a gradient-model disagreement — it is the catastrophic cancellation of recovering the
    /// tidal acceleration (~2.87e-8 m/s²) by subtracting the *forced* monopole from the field: this
    /// port reuses the tested `GeoPotential` recursion, which sets C̄₀₀ = 1, then subtracts −μr/r³.
    /// The monopole is ~7.21 m/s², so its ULP (~1.6e-15) sets an absolute floor of ~1.5e-15 m/s²,
    /// i.e. ULP(μ/r²)/|a_tide| ≈ 5.6e-8 relative — matched by the observed 5.2e-8 (abs ≈ 1.5e-15
    /// m/s², physically negligible). Orekit's ocean field has no monopole (C̄₀₀ = 0) and so avoids the
    /// cancellation; this is the documented cost of the shared-evaluator technique, as in
    /// [`crate::solid_tide`]. The bound is tight to what the cancellation floor allows.
    #[test]
    fn acceleration_matches_orekit() {
        let tide = fes2004_tide();
        let a = tide.acceleration::<f64>(0.0, &SAT_ITRF, &[0.0; 3]);
        let mag = libm::sqrt(A_TIDE_ITRF.iter().map(|v| v * v).sum());
        let diff = libm::sqrt((0..3).map(|i| (a[i] - A_TIDE_ITRF[i]).powi(2)).sum());
        let rel = diff / mag;
        std::eprintln!("ocean tide acceleration vs Orekit: rel diff {rel:.3e}");
        assert!(
            rel < 1e-7,
            "spacelab ocean tide should match Orekit to the monopole-cancellation floor, got {rel:.3e}"
        );
    }

    // ── Admittance inference ─────────────────────────────────────────────────────────────────────
    use crate::ocean_tide_admittance::{ADMITTANCE_LINEAR, ADMITTANCE_SPLINE, InferredWave};

    // Reference-major complex amplitudes driving the pyfes cross-check (keyed by Doodson). Q1..S2
    // reuse aviso-fes `tests/library/inference.cpp`; K2/2N2/Node/Mm/Mf are arbitrary distinct values.
    // Kept identical to `experiments/ocean_tide_admittance_verify.py`'s `MAJOR_TIDE`.
    const MAJOR_Z: [(u32, f64, f64); 11] = [
        (135655, 2.044_581_413_269_043, -2.377_691_745_758_056_6), // Q1
        (145555, 7.722_073_554_992_676, 1.225_713_372_230_529_8),  // O1
        (165555, -7.948_722_839_355_469, 5.252_667_903_900_146_5), // K1
        (245655, -18.638_496_398_925_781, 3.005_345_582_962_036_1), // N2
        (255555, -90.521_110_534_667_969, -20.603_012_084_960_938), // M2
        (273555, -24.581_066_131_591_797, -25.664_165_496_826_172), // S2
        (275555, -6.0, -7.0),                                      // K2
        (235755, -2.5, 1.3),                                       // 2N2
        (55565, 1.2, 0.0),                                         // Node
        (65455, 3.1, -1.1),                                        // Mm
        (75555, 5.7, 2.2),                                         // Mf
    ];

    // pyfes-recovered inferred minor amplitudes `(Doodson, Re, Im)`, produced by
    // `experiments/ocean_tide_admittance_verify.py` (least-squares recovery from
    // `evaluate_tide_from_constituents`; max |algo − pyfes| ≈ 7e-15 SPLINE / 3e-14 LINEAR).
    const SPLINE_GOLDEN: [(u32, f64, f64); 18] = [
        (125755, 3.431_286_581_0e-1, -6.562_209_061_1e-1), // 2Q1
        (127555, 4.033_779_378_9e-1, -7.385_332_815_2e-1), // Sigma1
        (137455, 3.723_773_048_4e-1, -3.840_580_221_2e-1), // Rho1
        (155455, 2.782_692_909_2e-2, 7.021_193_304_1e-2),  // M11
        (155655, 7.623_467_721_9e-2, 1.958_054_850_7e-1),  // M12
        (157455, 1.728_933_715_8e-3, 3.936_057_300_6e-2),  // Chi1
        (162556, -1.127_569_398_9e-1, 9.349_776_127_3e-2), // Pi1
        (167555, -1.324_036_974_9e-1, 7.800_198_208_1e-2), // Phi1
        (173655, -1.733_986_821_2e-1, 7.344_921_559_1e-2), // Theta1
        (175455, -9.649_018_906_6e-1, 3.914_427_865_9e-1), // J1
        (185555, -8.200_780_802_7e-1, 2.691_602_961_7e-1), // OO1
        (227655, -7.163_090_789_8e-1, 5.934_083_819_4e-1), // Eps2
        (237555, -2.779_570_232_2e0, 1.523_878_589_6e0),   // Mu2
        (247455, -3.498_890_654_8e0, 3.750_200_295_7e-1),  // Nu2
        (263655, -5.528_285_911_5e-1, -3.152_625_249_6e-1), // Lambda2
        (265455, -2.022_899_677_9e0, -1.269_405_076_0e0),  // L2
        (272556, -1.462_403_656_0e0, -1.495_426_584_4e0),  // T2
        (285455, -1.828_792_214_6e-1, -5.102_388_802_9e-1), // Eta2
    ];
    const LINEAR_GOLDEN: [(u32, f64, f64); 32] = [
        (58555, 9.595_077_308_7e-2, -1.262_221_406_0e-2), // Sta
        (63655, 5.519_804_183_6e-1, -1.820_805_024_2e-1), // MSm
        (73555, 5.017_028_121_2e-1, 1.371_189_538_3e-1),  // MSf
        (83655, 2.020_445_220_1e-1, 2.126_155_433_2e-1),  // MStm
        (95355, 1.359_691_821_8e-1, 2.690_106_688_7e-1),  // Mqm
        (125755, 3.453_742_358_8e-1, -6.598_289_706_0e-1), // 2Q1
        (127555, 4.053_148_748_1e-1, -7.416_511_434_7e-1), // Sigma1
        (137455, 3.742_646_489_8e-1, -3.855_647_396_1e-1), // Rho1
        (147555, 8.825_341_418_4e-2, 1.829_762_211_0e-2), // Tau1
        (153655, 1.564_181_219_0e-2, 1.643_286_710_7e-2), // Beta1
        (155655, 8.914_731_553_0e-2, 1.881_039_508_4e-1), // M1
        (157455, 3.979_727_811_7e-3, 3.814_888_296_1e-2), // Chi1
        (162556, -1.066_409_408_1e-1, 8.998_468_419_3e-2), // Pi1
        (166554, -4.333_084_488_6e-2, 2.687_637_975_4e-2), // Psi1
        (167555, -1.208_508_385_5e-1, 7.081_455_707_1e-2), // Phi1
        (173655, -1.647_085_403_5e-1, 6.799_761_112_0e-2), // Theta1
        (175455, -9.246_445_082_0e-1, 3.657_085_963_5e-1), // J1
        (183555, -2.264_010_061_2e-1, 7.359_198_485_9e-2), // SO1
        (185555, -7.903_187_537_5e-1, 2.506_649_520_0e-1), // OO1
        (195455, -2.041_971_727_2e-1, 5.729_311_567_0e-2), // Ups1
        (227655, -8.131_288_500_7e-1, 6.164_405_542_4e-1), // Eps2
        (237555, -3.155_665_368_7e0, 1.440_757_122_5e0),  // Mu2
        (247455, -3.506_890_078_7e0, 3.941_399_576_9e-1), // Nu2
        (253755, -2.751_975_347_1e-1, -4.739_046_469_8e-2), // Gamma2
        (254556, -3.137_600_479_4e-1, -6.157_231_950_4e-2), // Alpha2
        (256554, -2.705_426_650_5e-1, -6.687_735_049_4e-2), // Beta2
        (257555, -9.883_409_268_9e-2, -2.643_555_821_5e-2), // Delta2
        (263655, -5.387_547_730_1e-1, -2.703_239_626_7e-1), // Lambda2
        (265455, -1.987_788_666_1e0, -1.105_767_707_0e0), // L2
        (272556, -1.477_731_224_1e0, -1.461_620_796_9e0), // T2
        (274554, -1.999_776_397_5e-1, -2.204_335_687_8e-1), // R2
        (285455, -2.091_546_390_6e-1, -5.405_590_805_5e-1), // Eta2
    ];

    fn major_z(doodson: u32) -> (f64, f64) {
        MAJOR_Z
            .iter()
            .find(|&&(d, _, _)| d == doodson)
            .map(|&(_, re, im)| (re, im))
            .expect("every admittance reference is one of the FES2004 majors above")
    }

    /// Apply the table's real weights to the reference majors' complex amplitudes (`coef(minor) =
    /// Σ weight·coef(ref)`) and check every inferred minor against the pyfes-recovered golden. This
    /// is the standalone-admittance validation: the weights the runtime ships equal the FES oracle's.
    fn check_against_pyfes(table: &[InferredWave], golden: &[(u32, f64, f64)]) -> f64 {
        assert_eq!(
            table.len(),
            golden.len(),
            "golden must cover every inferred wave (no silent skips)"
        );
        let mut max_err = 0.0_f64;
        for inferred in table {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for &(ref_doodson, weight) in inferred.refs {
                let (r, i) = major_z(ref_doodson);
                re += weight * r;
                im += weight * i;
            }
            let &(_, gre, gim) = golden
                .iter()
                .find(|&&(d, _, _)| d == inferred.doodson)
                .expect("golden entry for each inferred wave");
            let err = libm::hypot(re - gre, im - gim);
            max_err = max_err.max(err);
        }
        max_err
    }

    /// The shipped admittance weights reproduce the pyfes-inferred minor amplitudes. Tolerance is set
    /// by the 10-significant-figure golden, not the method (weights match pyfes to ~1e-14 at full
    /// precision — see the experiment script).
    #[test]
    fn admittance_weights_match_pyfes() {
        let spline = check_against_pyfes(ADMITTANCE_SPLINE, &SPLINE_GOLDEN);
        let linear = check_against_pyfes(ADMITTANCE_LINEAR, &LINEAR_GOLDEN);
        std::eprintln!("admittance vs pyfes: SPLINE max {spline:.2e}, LINEAR max {linear:.2e}");
        assert!(spline < 1e-6, "SPLINE admittance vs pyfes: {spline:.3e}");
        assert!(linear < 1e-6, "LINEAR admittance vs pyfes: {linear:.3e}");
    }

    fn wave_coeffs(doodson: u32) -> &'static [(u8, u8, f64, f64, f64, f64)] {
        FES2004_WAVES
            .iter()
            .find(|w| w.doodson == doodson)
            .map(|w| w.coeffs)
            .expect("FES2004 wave present")
    }

    /// The FES admittance predicts an *explicit* wave from its neighbours in the geopotential domain
    /// (the whole point of admittance) — with the correct sign and phase per channel and (n,m). 2N2
    /// is tabulated explicitly *and* would be inferred (0.264·N2 − 0.0253·M2 by the SPLINE scheme);
    /// predicting it from the explicit N2/M2 grids recovers the explicit 2N2 to ~35% (the admittance
    /// approximation error), confirming the per-channel real-weighted combination is phase-correct —
    /// a wrong phase (e.g. a 90° Doodson-Warburg slip) would leave the residual O(1) or larger.
    #[test]
    fn admittance_predicts_explicit_2n2_in_geopotential_domain() {
        let (n2, m2, tn2) = (
            wave_coeffs(245655),
            wave_coeffs(255555),
            wave_coeffs(235755),
        );
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for &(n, m, cp, sp, cm, sm) in tn2 {
            let explicit = [cp, sp, cm, sm];
            let find = |grid: &'static [(u8, u8, f64, f64, f64, f64)]| {
                grid.iter()
                    .find(|&&(gn, gm, ..)| gn == n && gm == m)
                    .map_or([0.0; 4], |&(_, _, a, b, c, d)| [a, b, c, d])
            };
            let (n2c, m2c) = (find(n2), find(m2));
            for ch in 0..4 {
                let predicted = 0.264 * n2c[ch] - 0.0253 * m2c[ch];
                num += (predicted - explicit[ch]).powi(2);
                den += explicit[ch].powi(2);
            }
        }
        let rel = (num / den).sqrt();
        std::eprintln!("admittance-predicted 2N2 vs explicit: rel {rel:.3}");
        assert!(
            rel < 0.40,
            "SPLINE-predicted 2N2 should match explicit 2N2 to the admittance floor, got {rel:.3}"
        );
    }

    /// Enabling admittance adds the minor waves to Eq. 6.15 (the wiring fires and is deterministic),
    /// while leaving the explicit-wave sum — the Orekit-exact default — untouched when `admittance`
    /// is empty (guaranteed by [`delta_cs_matches_orekit`], which runs with `admittance: &[]`).
    #[test]
    fn admittance_augments_delta_cs() {
        let base = fes2004_tide().delta_cs(GMST, DELAUNAY);
        for table in [ADMITTANCE_SPLINE, ADMITTANCE_LINEAR] {
            let mut tide = fes2004_tide();
            tide.admittance = table;
            let augmented = tide.delta_cs(GMST, DELAUNAY);
            let mut diff = 0.0_f64;
            for n in 2..=tide.max_degree {
                for m in 0..=n {
                    diff += (augmented[n][m][0] - base[n][m][0]).abs();
                    diff += (augmented[n][m][1] - base[n][m][1]).abs();
                }
            }
            assert!(diff > 0.0, "admittance must change ΔC̄/ΔS̄");
            // Determinism: recomputing yields the identical grid.
            assert_eq!(augmented, tide.delta_cs(GMST, DELAUNAY));
        }
    }

    fn tide_with(
        waves: &'static [OceanTideWave],
        max_degree: usize,
        admittance: &'static [InferredWave],
    ) -> OceanTide<impl Fn(f64) -> [[f64; 3]; 3], impl Fn(f64) -> (f64, [f64; 5])> {
        OceanTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            waves,
            scale: SCALE,
            max_degree,
            admittance,
            rotate: |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            delaunay_gmst: |_t: f64| (GMST, DELAUNAY),
        }
    }

    /// An admittance wave that infers a major from *itself* at unit weight contributes exactly what
    /// that major already contributes, so the grid doubles cell for cell — to the last bit.
    ///
    /// `add_admittance` restates IERS Eq. 6.15 for the inferred waves, and nothing asserted the
    /// restatement: `admittance_augments_delta_cs` only requires the grid to *move*, and to move
    /// reproducibly. 47 of that function's 51 mutants pass a test of that shape — every operator of
    /// the fold, both sides of each weighted accumulation, the reference lookup and the degree cut.
    /// Pointing a wave at itself at weight 1 makes the explicit-wave loop the oracle for the inferred
    /// one: same coefficients, same θ, so the two folds must land on the same double and their sum on
    /// exactly twice it. That loop is what `delta_cs_matches_orekit` pins against Orekit.
    ///
    /// The second case is the same identity at `max_degree = GRID_MAX`, where the scratch channel
    /// grid is read at its last row and column — the only reach that separates `GRID_MAX + 1` from
    /// `GRID_MAX` or `GRID_MAX - 1`.
    #[test]
    fn a_self_referencing_admittance_wave_doubles_its_major() {
        const M2: u32 = 255_555;
        let top = u8::try_from(GRID_MAX).expect("GRID_MAX fits a coefficient degree");
        let self_ref: &'static [InferredWave] = Box::leak(Box::new([InferredWave {
            doodson: M2,
            refs: Box::leak(Box::new([(M2, 1.0)])),
        }]));
        let cases: [(&'static [OceanTideWave], usize); 2] = [
            (
                Box::leak(Box::new([OceanTideWave {
                    doodson: M2,
                    coeffs: wave_coeffs(M2),
                }])),
                6,
            ),
            // One coefficient, at the top corner of the grid, with no zero among the four channels.
            (
                Box::leak(Box::new([OceanTideWave {
                    doodson: M2,
                    coeffs: Box::leak(Box::new([(top, top, 3.5, -1.25, 0.75, -2.125)])),
                }])),
                GRID_MAX,
            ),
        ];

        for (waves, max_degree) in cases {
            let alone = tide_with(waves, max_degree, &[]).delta_cs(GMST, DELAUNAY);
            let doubled = tide_with(waves, max_degree, self_ref).delta_cs(GMST, DELAUNAY);
            let mut reached = 0_usize;
            for n in 2..=max_degree {
                for m in 0..=n {
                    for ch in 0..2 {
                        assert_eq!(
                            doubled[n][m][ch],
                            2.0 * alone[n][m][ch],
                            "degree {max_degree}, cell ({n},{m}) channel {ch}"
                        );
                        reached += usize::from(alone[n][m][ch] != 0.0);
                    }
                }
            }
            assert!(reached > 0, "degree {max_degree}: nothing to double");
        }
    }

    fn tide_with_rotation<R: Fn(f64) -> [[f64; 3]; 3]>(
        rotate: R,
    ) -> OceanTide<R, impl Fn(f64) -> (f64, [f64; 5])> {
        OceanTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            waves: FES2004_WAVES,
            scale: SCALE,
            max_degree: 6,
            admittance: &[],
            rotate,
            delaunay_gmst: |_t: f64| (GMST, DELAUNAY),
        }
    }

    /// The Earth orientation must be transparent: the tidal field is body-fixed, so rotating the
    /// satellite in and the acceleration out is the same computation whichever side does it.
    ///
    /// `acceleration_matches_orekit` passes the identity — the oracle's state is already in the ITRF
    /// — where every off-diagonal entry is an exact zero and every diagonal one an exact one, and
    /// `x · 1 ≡ x / 1`, `x + 0 ≡ x − 0`. Both frame transforms are then unassertable: 24 mutants.
    #[test]
    fn a_general_earth_orientation_is_transparent() {
        // A ZXZ rotation with no entry near 0 or ±1, so each product and sum carries its operator.
        let m = {
            let (alpha, beta, gamma) = (0.7_f64, 0.4_f64, 1.1_f64);
            let (ca, sa) = (libm::cos(alpha), libm::sin(alpha));
            let (cb, sb) = (libm::cos(beta), libm::sin(beta));
            let (cg, sg) = (libm::cos(gamma), libm::sin(gamma));
            [
                [cg * ca - sg * cb * sa, cg * sa + sg * cb * ca, sg * sb],
                [-sg * ca - cg * cb * sa, -sg * sa + cg * cb * ca, cg * sb],
                [sb * sa, -sb * ca, cb],
            ]
        };
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| m[k][i] * m[k][j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-15, "MᵀM at ({i},{j}) is {dot}");
                assert!(m[i][j].abs() > 0.1, "entry ({i},{j}) hides its operator");
            }
        }
        let apply = |v: [f64; 3]| [0, 1, 2].map(|i| (0..3).map(|j| m[i][j] * v[j]).sum::<f64>());

        // The satellite is in the GCRF and the force rotates it; or it is already in the ITRF and
        // the force does nothing. The two answers are each other's Mᵀ.
        let gcrf = [4_100_000.0_f64, -5_300_000.0, 2_600_000.0];
        let got = tide_with_rotation(move |_t: f64| m).acceleration::<f64>(0.0, &gcrf, &[0.0; 3]);
        let in_frame =
            tide_with_rotation(|_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
                .acceleration::<f64>(0.0, &apply(gcrf), &[0.0; 3]);
        let want = [0, 1, 2].map(|i| (0..3).map(|j| m[j][i] * in_frame[j]).sum::<f64>());

        let norm = |v: [f64; 3]| libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        let rel = norm([0, 1, 2].map(|i| got[i] - want[i])) / norm(want);
        std::eprintln!("ocean tide, rotated vs pre-rotated: {rel:.3e} relative");
        // Loose by the standard of this file for the reason `acceleration_matches_orekit` records:
        // subtracting the forced monopole cancels eight digits, so merely reassociating the same
        // arithmetic already costs ~5e-8. Every mutation here moves the answer by a fraction of it.
        assert!(
            rel < 1e-7,
            "the Earth orientation is not transparent: {rel:.3e}"
        );
    }

    // ── FES2014b full-resolution table ───────────────────────────────────────────────────────────
    use crate::ocean_tide_fes2014::{FES2014B_WAVES, SCALE as FES2014B_SCALE};

    // Raw FES2014b rows lifted verbatim from `fes2014b_Cnm-Snm.txt` for M2 (Doodson 255555): a low
    // cell, a mid-degree cell, and the top (degree-100) cell. If the generated table carries these
    // byte-for-byte, the parse is faithful to the file and the high degrees were not truncated.
    const M2_RAW: [(u8, u8, f64, f64, f64, f64); 3] = [
        (2, 2, -39.01577, 47.08311, 9.78995, 5.43588),
        (60, 4, 0.04328, -0.04936, 0.06141, 0.04615),
        (100, 1, -0.01134, 0.00125, -0.00829, -0.01883),
    ];

    fn fes2014b_tide(
        max_degree: usize,
    ) -> OceanTide<impl Fn(f64) -> [[f64; 3]; 3], impl Fn(f64) -> (f64, [f64; 5])> {
        OceanTide {
            mu_earth: MU_EARTH,
            re_earth: RE_EARTH,
            waves: FES2014B_WAVES,
            scale: FES2014B_SCALE,
            max_degree,
            admittance: &[],
            rotate: |_t: f64| [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            delaunay_gmst: |_t: f64| (GMST, DELAUNAY),
        }
    }

    /// Independent ΔC̄nm/ΔS̄nm at one cell: sum Eq. 6.15 over every wave in `table` carrying `(n, m)`,
    /// reusing the shared `doodson_theta` but doing the accumulation here — a from-scratch check of
    /// `delta_cs`'s indexing and recurrence in the same wave order, not a call back into it.
    fn recompute_cell(
        table: &[OceanTideWave],
        scale: f64,
        gmst: f64,
        delaunay: [f64; 5],
        n: usize,
        m: usize,
    ) -> [f64; 2] {
        let gamma = gmst + core::f64::consts::PI;
        let (mut dc, mut ds) = (0.0_f64, 0.0_f64);
        for wave in table {
            let theta = doodson_theta(wave.doodson, gamma, delaunay);
            let (sin, cos) = libm::sincos(theta);
            for &(wn, wm, cp, sp, cm, sm) in wave.coeffs {
                if usize::from(wn) != n || usize::from(wm) != m {
                    continue;
                }
                let (cp, sp, cm, sm) = (cp * scale, sp * scale, cm * scale, sm * scale);
                dc += (cp + cm) * cos + (sp + sm) * sin;
                ds += (sp - sm) * cos - (cp - cm) * sin;
            }
        }
        [dc, ds]
    }

    /// FES2014b is loaded at full resolution and evaluated correctly: 35 waves to degree/order 100
    /// (no cap), verbatim-faithful to the source file, and `delta_cs` reproduces a from-scratch Eq.
    /// 6.15 recompute at a low cell and the top (degree-100) cell — the degree the FES2004 gate cannot
    /// reach. Run at `max_degree = GRID_MAX` (the geopotential ceiling), so the full grid, flattened
    /// coefficients, and degree-120 field build are all exercised end to end.
    #[test]
    fn fes2014b_full_resolution_and_spot_check() {
        // (a) shape: 35 waves, max degree 100.
        assert_eq!(FES2014B_WAVES.len(), 35, "FES2014b wave count");
        let max_deg = FES2014B_WAVES
            .iter()
            .flat_map(|w| w.coeffs)
            .map(|&(n, ..)| n)
            .max()
            .unwrap();
        assert_eq!(max_deg, 100, "FES2014b max degree");

        // (b) parse fidelity: the table carries the raw M2 rows verbatim, including the degree-100 one.
        let m2 = FES2014B_WAVES
            .iter()
            .find(|w| w.doodson == 255_555)
            .expect("M2 present");
        for &(n, m, cp, sp, cm, sm) in &M2_RAW {
            let got = m2
                .coeffs
                .iter()
                .find(|&&(gn, gm, ..)| gn == n && gm == m)
                .copied()
                .unwrap_or_else(|| panic!("M2 ({n},{m}) missing from table"));
            assert_eq!(got, (n, m, cp, sp, cm, sm), "M2 ({n},{m}) not verbatim");
        }

        // (c) delta_cs matches an independent Eq. 6.15 recompute at two cells, one at the top degree
        // (100) — proving the runtime degree reaches past the FES2004 gate's 6, bit-for-bit.
        let tide = fes2014b_tide(GRID_MAX);
        let grid = tide.delta_cs(GMST, DELAUNAY);
        for (n, m) in [(2_usize, 2_usize), (100, 1)] {
            let want = recompute_cell(FES2014B_WAVES, FES2014B_SCALE, GMST, DELAUNAY, n, m);
            assert_eq!(
                grid[n][m], want,
                "ΔC̄/ΔS̄ ({n},{m}): delta_cs vs from-scratch recompute"
            );
        }

        // End-to-end: the full-resolution acceleration is finite and nonzero (a degree-120 field is
        // built each step, exercising the GRID_MAX grids and the flattened-coefficient buffer).
        let a = tide.acceleration::<f64>(0.0, &SAT_ITRF, &[0.0; 3]);
        assert!(
            a.iter().all(|c| c.is_finite()),
            "FES2014b acceleration non-finite: {a:?}"
        );
        assert!(a.iter().any(|&c| c != 0.0), "FES2014b acceleration is zero");
    }
}
