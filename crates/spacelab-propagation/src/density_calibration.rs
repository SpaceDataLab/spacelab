//! Open, HASDM-style thermospheric **density calibration** — a data-driven multiplicative correction
//! to *any* base density model (JB2008 / DTM2020 / NRLMSISE-00), fit to observed density/model ratios.
//!
//! The operational High Accuracy Satellite Drag Model (Storz et al. 2005) corrects a base
//! thermosphere by solving, in near-real time, for a low-degree spherical-harmonic field of
//! corrections against the drag of a fleet of calibration satellites. The *classified* part is the
//! tracking feed; the *mathematics* is open. This is that mathematics, driven instead by **openly
//! published accelerometer-derived densities** (CHAMP / GRACE / GRACE-FO / Swarm): fit a low-degree
//! real spherical-harmonic multiplicative correction to `ln(ρ_observed / ρ_base)` in the sun-fixed
//! (latitude × local-solar-time) frame — the frame the diurnal density bulge lives in — and apply it
//! on top of the free base model. Degree 0 recovers the classic scalar density scale factor estimated
//! in orbit determination; higher degrees resolve the diurnal/latitudinal structure HASDM captures.
//!
//! This is our own model — no NRLMSIS 2.x code, data, or patented method is used. The correction is a
//! plain weighted least-squares regression on an orthogonal-ish spherical-harmonic basis, so it is
//! exact and verifiable (a known coefficient set round-trips through [`DensityCorrection::fit`]).

// Spherical-harmonic index maths uses short symbols (n, m, x, s) and index-parallel loops over the
// coefficient/basis vectors; the WLS normal equations reuse the crate's const-generic LU solver, so
// `K = (degree + 1)^2` is a const generic like `linalg::Lu<K>`. Degree indices are tiny in f64.
#![allow(
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::cast_precision_loss
)]

use spacelab_autodiff::Scalar;

use crate::forces::Atmosphere;
use crate::linalg::lu_solve;

/// One calibration observation: a density ratio at a sun-fixed location.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensitySample {
    /// Geocentric latitude of the observation, radians.
    pub latitude: f64,
    /// Local solar time, hours in `[0, 24)`.
    pub local_solar_time: f64,
    /// `ln(ρ_observed / ρ_base)` — the log of the observed-to-model density ratio.
    pub log_ratio: f64,
    /// Observation weight (e.g. `1/σ²`); non-positive weights drop the sample.
    pub weight: f64,
}

/// A real spherical-harmonic multiplicative log-density correction in the sun-fixed
/// (latitude × local-solar-time) frame, with `K = (degree + 1)²` coefficients. `DensityCorrection<1>`
/// is a pure scale factor (degree 0); `DensityCorrection<25>` is degree 4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityCorrection<const K: usize> {
    coeffs: [f64; K],
}

const fn isqrt(k: usize) -> usize {
    let mut i = 1;
    while (i + 1) * (i + 1) <= k {
        i += 1;
    }
    i
}

impl<const K: usize> DensityCorrection<K> {
    /// The identity correction (all coefficients zero → factor `1` everywhere).
    #[must_use]
    pub const fn identity() -> Self {
        Self { coeffs: [0.0; K] }
    }

    /// Wrap a known coefficient vector (packed in this module's basis order).
    #[must_use]
    pub const fn from_coefficients(coeffs: [f64; K]) -> Self {
        Self { coeffs }
    }

    /// The packed coefficient vector.
    #[must_use]
    pub const fn coefficients(&self) -> &[f64; K] {
        &self.coeffs
    }

    /// The spherical-harmonic degree, `√K − 1`.
    #[must_use]
    pub const fn degree() -> usize {
        isqrt(K) - 1
    }

    /// The `K` real spherical-harmonic basis functions at a sun-fixed location, packed as
    /// `[Y₀₀, Y₁₀, Y₁₁ᶜ, Y₁₁ˢ, Y₂₀, …]` (per degree `n`: the zonal term, then `(cos, sin)` for each
    /// order `m = 1..n`).
    #[must_use]
    pub fn basis(latitude: f64, local_solar_time: f64) -> [f64; K] {
        debug_assert!(
            isqrt(K) * isqrt(K) == K,
            "K must be a perfect square (degree+1)^2"
        );
        let deg = Self::degree();
        let x = libm::sin(latitude); // cos(colatitude)
        let s = libm::cos(latitude); // sin(colatitude)
        let lambda = local_solar_time * (core::f64::consts::TAU / 24.0);

        // Fully-normalised associated Legendre P̄_nm(x), lower triangle, degree ≤ `deg`.
        let mut p = [[0.0_f64; K]; K]; // p[n][m]; only n,m ≤ deg used
        p[0][0] = 1.0;
        if deg >= 1 {
            p[1][1] = libm::sqrt(3.0) * s;
            p[1][0] = libm::sqrt(3.0) * x;
        }
        for n in 2..=deg {
            let nf = n as f64;
            // sectorial P̄_nn
            p[n][n] = libm::sqrt((2.0 * nf + 1.0) / (2.0 * nf)) * s * p[n - 1][n - 1];
            for m in 0..n {
                let mf = m as f64;
                let a = libm::sqrt((2.0 * nf - 1.0) * (2.0 * nf + 1.0) / ((nf - mf) * (nf + mf)));
                let term2 = if n >= 2 && m <= n - 2 {
                    let b = libm::sqrt(
                        (2.0 * nf + 1.0) * (nf - mf - 1.0) * (nf + mf - 1.0)
                            / ((2.0 * nf - 3.0) * (nf - mf) * (nf + mf)),
                    );
                    b * p[n - 2][m]
                } else {
                    0.0
                };
                p[n][m] = a * x * p[n - 1][m] - term2;
            }
        }

        let sqrt2 = core::f64::consts::SQRT_2;
        let mut basis = [0.0_f64; K];
        let mut k = 0;
        for n in 0..=deg {
            basis[k] = p[n][0];
            k += 1;
            for m in 1..=n {
                let (sin_ml, cos_ml) = libm::sincos(m as f64 * lambda);
                basis[k] = sqrt2 * p[n][m] * cos_ml;
                k += 1;
                basis[k] = sqrt2 * p[n][m] * sin_ml;
                k += 1;
            }
        }
        basis
    }

    /// The log-density correction `ln(ρ_corrected / ρ_base)` at a sun-fixed location.
    #[must_use]
    pub fn log_correction(&self, latitude: f64, local_solar_time: f64) -> f64 {
        let b = Self::basis(latitude, local_solar_time);
        let mut acc = 0.0;
        for i in 0..K {
            acc += self.coeffs[i] * b[i];
        }
        acc
    }

    /// The multiplicative correction factor `ρ_corrected / ρ_base = exp(log_correction)`.
    #[must_use]
    pub fn factor(&self, latitude: f64, local_solar_time: f64) -> f64 {
        libm::exp(self.log_correction(latitude, local_solar_time))
    }

    /// Fit the correction by ordinary weighted least squares to `samples`. Returns `None` if the normal
    /// equations are singular (too few or degenerate samples for the degree).
    #[must_use]
    pub fn fit(samples: &[DensitySample]) -> Option<Self> {
        Self::fit_ridge(samples, 0.0)
    }

    /// Fit with Tikhonov (ridge) regularization: solves `(BᵀW B + ridge·I) c = BᵀW y`. A small positive
    /// `ridge` stabilizes the coefficients when the samples do not span enough local-solar-time /
    /// latitude to determine the full degree — e.g. a single satellite-month, whose sun-fixed coverage
    /// is narrow, otherwise yields huge cancelling coefficients. It trades a little training fit for
    /// physically-bounded, better-generalizing coefficients. `ridge = 0` reproduces [`Self::fit`].
    ///
    /// # Errors
    /// Returns `None` if the (regularized) normal equations are singular.
    #[must_use]
    pub fn fit_ridge(samples: &[DensitySample], ridge: f64) -> Option<Self> {
        let mut ata = [[0.0_f64; K]; K];
        let mut atb = [0.0_f64; K];
        for sample in samples {
            if sample.weight <= 0.0 {
                continue;
            }
            let b = Self::basis(sample.latitude, sample.local_solar_time);
            let wy = sample.weight * sample.log_ratio;
            for i in 0..K {
                atb[i] += wy * b[i];
                let wbi = sample.weight * b[i];
                for j in 0..K {
                    ata[i][j] += wbi * b[j];
                }
            }
        }
        for i in 0..K {
            ata[i][i] += ridge;
        }
        let coeffs = lu_solve::<K>(ata, atb)?;
        Some(Self { coeffs })
    }
}

/// A base [`Atmosphere`] wrapped with a [`DensityCorrection`] — the correction slotted straight into
/// the drag force: `ρ_corrected = ρ_base · factor(latitude, local_solar_time)`.
///
/// `geodetic` maps `(t, inertial position in metres) → (altitude_km, geodetic_latitude_deg,
/// geodetic_longitude_deg)` — the same provider [`crate::nrlmsise00`]'s model takes — and `sec0` is
/// the UT seconds-of-day at the propagation epoch, so local solar time is `sec/3600 + longitude/15`
/// hours. The correction is a slowly-varying multiplier: it is evaluated at the point in `f64` and
/// lifted into `S`, so the base model carries the position derivative (exactly as the empirical
/// density models already treat their table lookups).
pub struct CalibratedAtmosphere<A, G, const K: usize> {
    base: A,
    correction: DensityCorrection<K>,
    geodetic: G,
    sec0: f64,
}

impl<A, G, const K: usize> CalibratedAtmosphere<A, G, K> {
    /// Wrap `base` with `correction`, using `geodetic` for the sun-fixed location and `sec0` as the
    /// epoch UT seconds-of-day.
    pub fn new(base: A, correction: DensityCorrection<K>, geodetic: G, sec0: f64) -> Self {
        Self {
            base,
            correction,
            geodetic,
            sec0,
        }
    }
}

impl<A, G, const K: usize> Atmosphere for CalibratedAtmosphere<A, G, K>
where
    A: Atmosphere,
    G: Fn(f64, [f64; 3]) -> (f64, f64, f64),
{
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let (_alt, lat_deg, lon_deg) = (self.geodetic)(t, rf);
        let total_sec = self.sec0 + t;
        let sec = total_sec - 86_400.0 * libm::floor(total_sec / 86_400.0);
        let mut lst = sec / 3_600.0 + lon_deg / 15.0;
        lst -= 24.0 * libm::floor(lst / 24.0); // wrap to [0, 24) hours
        let factor = self.correction.factor(lat_deg.to_radians(), lst);
        self.base.density::<S>(t, r) * S::from_f64(factor)
    }
}

#[cfg(test)]
mod tests {
    // The oracle table below is machine-generated at full f64 precision; digit grouping it
    // would mean hand-editing 110 numbers.
    #![allow(
        clippy::unwrap_used,
        clippy::float_cmp,
        clippy::cast_lossless,
        clippy::unreadable_literal
    )]

    use super::*;

    /// The basis against fully-normalised real spherical harmonics built from Rodrigues' formula in
    /// `tools/gen_shbasis_oracle.mac`, which shares no arithmetic with the recursion here — the
    /// oracle differentiates `(x²−1)ⁿ` and applies `N_nm` directly, the code walks a three-term
    /// recursion. Every other test reaches `basis` only through `fit`, which asks whether a fitted
    /// model reproduces the samples it was fitted to; the least-squares coefficients absorb a wrong
    /// basis exactly, so that question cannot detect one.
    ///
    /// Degrees 1, 4 and 5: the loop that never runs, the smallest degree that exercises both sides
    /// of the `m <= n - 2` guard, and the first degree at which that bound and `n / 2` disagree.
    #[test]
    #[allow(clippy::too_many_lines)] // the length is the reference table, not logic
    fn the_basis_is_the_fully_normalised_real_harmonic_at_that_sun_fixed_point() {
        fn check<const K: usize>(lat_deg: f64, lst: f64, want: &[f64; K]) {
            let got = DensityCorrection::<K>::basis(lat_deg.to_radians(), lst);
            for (k, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert!(
                    (g - w).abs() <= 2e-14 * w.abs().max(1.0),
                    "lat={lat_deg} lst={lst}: basis[{k}] = {g}, want {w}"
                );
            }
        }
        check::<25>(
            21.2,
            3.7,
            &[
                1.0,
                0.6263521286474361,
                1.2935118852351393,
                1.8820714717122429,
                -0.6794102606154813,
                1.0459558566668798,
                1.5218752150431316,
                -0.8530870245246814,
                2.2223676788214206,
                -1.122355847340425,
                -0.4188168741573118,
                -0.6093823332592597,
                -0.816206946727551,
                2.1262917914470596,
                -2.3309655584122666,
                0.5596153187221067,
                -0.121732804949783,
                -1.3352125606209904,
                -1.942746330900395,
                0.06249739977234496,
                -0.16281129271873274,
                -2.528803253811,
                0.6071119471327047,
                -1.7616548815723512,
                -1.5862011800247773,
            ],
        );
        check::<25>(
            -58.4,
            11.2,
            &[
                1.0,
                -1.4752343241106296,
                -1.255450589053552,
                0.2668542605558269,
                1.3151616172570686,
                2.3910299059384603,
                -0.5082290956564753,
                0.6869102634886088,
                -0.3058321535379735,
                -0.7066773924395401,
                -3.08528842893821,
                0.6557983002154102,
                -1.5479231899614616,
                0.6891798068251371,
                -0.3442868917991628,
                0.2501390687268719,
                -0.1290051664378563,
                3.042713810441845,
                -0.6467487856934099,
                2.425970198835851,
                -1.0801115222253739,
                0.8797152564532204,
                -0.6391505463484071,
                0.15825943296278375,
                -0.1757649068382393,
            ],
        );
        check::<25>(
            74.9,
            22.9,
            &[
                1.0,
                1.67224765000001,
                0.6118258661289423,
                -0.18123107811425582,
                2.0084499006814363,
                1.3208478779610937,
                -0.3912529662441029,
                0.15586640631774326,
                -0.10122082778715091,
                2.12102984032762,
                2.095050316091267,
                -0.6205821762508051,
                0.39814522294958676,
                -0.25855853097877735,
                0.033962071745825995,
                -0.039764476967883294,
                2.042506001696352,
                2.851168290971669,
                -0.8445545241937793,
                0.7457829222833359,
                -0.48431709260782724,
                0.09836835227565487,
                -0.11517454258115592,
                0.005876984758721302,
                -0.01319992388776264,
            ],
        );
        check::<4>(
            21.2,
            3.7,
            &[
                1.0,
                0.6263521286474361,
                1.2935118852351393,
                1.8820714717122429,
            ],
        );
        check::<36>(
            21.2,
            3.7,
            &[
                1.0,
                0.6263521286474361,
                1.2935118852351393,
                1.8820714717122429,
                -0.6794102606154813,
                1.0459558566668798,
                1.5218752150431316,
                -0.8530870245246814,
                2.2223676788214206,
                -1.122355847340425,
                -0.4188168741573118,
                -0.6093823332592597,
                -0.816206946727551,
                2.1262917914470596,
                -2.3309655584122666,
                0.5596153187221067,
                -0.121732804949783,
                -1.3352125606209904,
                -1.942746330900395,
                0.06249739977234496,
                -0.16281129271873274,
                -2.528803253811,
                0.6071119471327047,
                -1.7616548815723512,
                -1.5862011800247773,
                1.0379561856301962,
                -0.5656035948667538,
                -0.8229583371807183,
                0.8225148138806773,
                -2.1427243471892097,
                -0.3419992460069364,
                0.08210675458768063,
                -2.1128813248262883,
                -1.902446889994968,
                0.30255766246228016,
                -2.298153608984373,
            ],
        );
    }

    #[test]
    fn degree_zero_is_a_scalar_scale_factor() {
        assert_eq!(DensityCorrection::<1>::degree(), 0);
        assert_eq!(DensityCorrection::<25>::degree(), 4);
        // At degree 0 the correction is uniform: a single scale factor exp(c0).
        let c = DensityCorrection::<1>::from_coefficients([0.5]);
        let f0 = c.factor(0.3, 5.0);
        let f1 = c.factor(-1.1, 19.0);
        assert!((f0 - f1).abs() < 1e-12 && (f0 - libm::exp(0.5)).abs() < 1e-12);
    }

    #[test]
    fn identity_is_unit_factor_everywhere() {
        let c = DensityCorrection::<25>::identity();
        assert!((c.factor(0.4, 3.0) - 1.0).abs() < 1e-15);
        assert!((c.factor(-0.9, 21.5) - 1.0).abs() < 1e-15);
    }

    // A well-spread sun-fixed sampling: 7 latitudes × 12 local-solar-times = 84 points > K=25.
    fn grid() -> alloc::vec::Vec<(f64, f64)> {
        let mut g = alloc::vec::Vec::new();
        for i in 0..7 {
            let lat = (-75.0 + 25.0 * i as f64).to_radians();
            for j in 0..12 {
                g.push((lat, 2.0 * j as f64));
            }
        }
        g
    }

    #[test]
    fn weighted_least_squares_round_trips_a_known_field() {
        // A known degree-4 coefficient set (deterministic, no RNG).
        let mut truth = [0.0_f64; 25];
        for (k, c) in truth.iter_mut().enumerate() {
            *c = 0.15 * libm::sin(0.7 * k as f64 + 1.0) / (1.0 + k as f64);
        }
        let field = DensityCorrection::<25>::from_coefficients(truth);

        let samples: alloc::vec::Vec<DensitySample> = grid()
            .into_iter()
            .map(|(lat, lst)| DensitySample {
                latitude: lat,
                local_solar_time: lst,
                log_ratio: field.log_correction(lat, lst),
                weight: 1.0,
            })
            .collect();

        let fitted = DensityCorrection::<25>::fit(&samples).unwrap();
        for k in 0..25 {
            assert!(
                (fitted.coefficients()[k] - truth[k]).abs() < 1e-9,
                "coeff {k}: {} vs {}",
                fitted.coefficients()[k],
                truth[k]
            );
        }
    }

    #[test]
    fn calibration_reduces_the_density_ratio_residual() {
        let mut truth = [0.0_f64; 25];
        for (k, c) in truth.iter_mut().enumerate() {
            *c = 0.2 * libm::cos(1.3 * k as f64);
        }
        let field = DensityCorrection::<25>::from_coefficients(truth);

        // Observations = structured correction + deterministic pseudo-noise.
        let samples: alloc::vec::Vec<DensitySample> = grid()
            .into_iter()
            .enumerate()
            .map(|(i, (lat, lst))| {
                let noise = 0.02 * libm::sin(11.0 * i as f64);
                DensitySample {
                    latitude: lat,
                    local_solar_time: lst,
                    log_ratio: field.log_correction(lat, lst) + noise,
                    weight: 1.0,
                }
            })
            .collect();

        let fitted = DensityCorrection::<25>::fit(&samples).unwrap();
        let mut raw = 0.0;
        let mut corrected = 0.0;
        for s in &samples {
            raw += s.log_ratio * s.log_ratio;
            let r = s.log_ratio - fitted.log_correction(s.latitude, s.local_solar_time);
            corrected += r * r;
        }
        // The fit absorbs the structured part; only the ~0.02 noise floor remains.
        assert!(corrected < 0.1 * raw, "corrected {corrected} vs raw {raw}");
    }

    #[test]
    fn fit_is_none_when_underdetermined() {
        let samples = [DensitySample {
            latitude: 0.0,
            local_solar_time: 12.0,
            log_ratio: 0.1,
            weight: 1.0,
        }];
        // One sample cannot determine 25 coefficients.
        assert!(DensityCorrection::<25>::fit(&samples).is_none());
    }

    #[test]
    fn ridge_shrinks_the_coefficients_and_zero_reproduces_ols() {
        let samples: alloc::vec::Vec<DensitySample> = grid()
            .into_iter()
            .map(|(lat, lst)| DensitySample {
                latitude: lat,
                local_solar_time: lst,
                log_ratio: 0.3 * libm::sin(lat)
                    + 0.1 * libm::cos(lst * core::f64::consts::TAU / 24.0),
                weight: 1.0,
            })
            .collect();
        let ols = DensityCorrection::<25>::fit(&samples).unwrap();
        let ridged = DensityCorrection::<25>::fit_ridge(&samples, 10.0).unwrap();
        let sq_norm =
            |c: &DensityCorrection<25>| c.coefficients().iter().map(|x| x * x).sum::<f64>();
        assert!(
            sq_norm(&ridged) < sq_norm(&ols),
            "ridge did not shrink coefficients"
        );
        // ridge = 0 is exactly the ordinary least-squares fit.
        assert_eq!(DensityCorrection::<25>::fit_ridge(&samples, 0.0), Some(ols));
    }

    fn base_atmosphere() -> crate::forces::ExponentialAtmosphere {
        crate::forces::ExponentialAtmosphere {
            reference_density: 1e-12,
            reference_altitude: 400_000.0,
            scale_height: 60_000.0,
            earth_radius: 6_378_137.0,
        }
    }

    #[test]
    fn identity_calibration_is_exactly_the_base_model() {
        let base = base_atmosphere();
        let cal = CalibratedAtmosphere::new(
            base,
            DensityCorrection::<25>::identity(),
            |_t, _rf| (400.0, 0.0, 0.0),
            0.0,
        );
        let r = [7.0e6, 0.0, 0.0];
        assert!((cal.density::<f64>(0.0, &r) - base.density::<f64>(0.0, &r)).abs() < 1e-30);
    }

    #[test]
    fn calibrated_atmosphere_multiplies_by_the_scale_factor() {
        let base = base_atmosphere();
        // Degree 0 = a uniform scale factor exp(0.25), independent of the geodetic details.
        let corr = DensityCorrection::<1>::from_coefficients([0.25]);
        let geodetic = |_t: f64, rf: [f64; 3]| {
            let alt =
                (libm::sqrt(rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2]) - 6_378_137.0) / 1e3;
            (alt, 12.0, 34.0)
        };
        let cal = CalibratedAtmosphere::new(base, corr, geodetic, 0.0);
        let r = [7.0e6, 1.0e6, 2.0e6];
        let ratio = cal.density::<f64>(0.0, &r) / base.density::<f64>(0.0, &r);
        assert!((ratio - libm::exp(0.25)).abs() < 1e-9, "ratio {ratio}");
    }

    #[test]
    fn local_solar_time_maps_from_epoch_seconds_and_longitude() {
        // Fixed geodetic (lat 30°, lon 45°), epoch UT = 6 h (21600 s): lst = 6 + 45/15 = 9 h.
        let base = base_atmosphere();
        let corr = DensityCorrection::<9>::from_coefficients([
            0.1, 0.02, -0.03, 0.04, 0.01, -0.02, 0.03, -0.01, 0.02,
        ]);
        let cal = CalibratedAtmosphere::new(base, corr, |_t, _rf| (400.0, 30.0, 45.0), 21_600.0);
        let r = [7.0e6, 0.0, 0.0];
        let ratio = cal.density::<f64>(0.0, &r) / base.density::<f64>(0.0, &r);
        let expected = corr.factor(30.0_f64.to_radians(), 9.0);
        assert!(
            (ratio - expected).abs() < 1e-9,
            "ratio {ratio} vs {expected}"
        );
    }

    #[test]
    fn local_solar_time_crosses_the_day_boundary_and_wraps_a_negative_longitude() {
        // The fixture above evaluates at `t = 0` with a longitude that is a whole multiple of 15°,
        // so both `floor`s return zero and the elapsed time never enters: the epoch seconds, the
        // day wrap and the hour wrap are all identities there. Here 70 000 s of epoch plus 25 000 s
        // of propagation crosses midnight, and −37.5° of longitude is −2.5 h, which drives the local
        // solar time below zero so the `[0, 24)` wrap has to lift it back.
        let base = base_atmosphere();
        let corr = DensityCorrection::<9>::from_coefficients([
            0.1, 0.02, -0.03, 0.04, 0.01, -0.02, 0.03, -0.01, 0.02,
        ]);
        let (lat_deg, lon_deg) = (-12.25, -37.5);
        let cal = CalibratedAtmosphere::new(
            base,
            corr,
            move |_t, _rf| (400.0, lat_deg, lon_deg),
            70_000.0,
        );
        let r = [7.0e6, 0.0, 0.0];
        let ratio = cal.density::<f64>(25_000.0, &r) / base.density::<f64>(25_000.0, &r);
        // 70 000 + 25 000 = 95 000 s is 8 600 s into the following day; 8 600/3600 h − 2.5 h is
        // negative, so one turn of 24 h is added back.
        let expected_lst = 8_600.0 / 3_600.0 - 2.5 + 24.0;
        let expected = corr.factor(lat_deg.to_radians(), expected_lst);
        assert!(
            (ratio - expected).abs() < 1e-9 * expected.abs(),
            "ratio {ratio} vs {expected} at lst {expected_lst}"
        );
    }
}
