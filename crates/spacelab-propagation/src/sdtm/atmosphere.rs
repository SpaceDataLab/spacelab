//! SDTM as a [`crate::forces::Atmosphere`] for the [`crate::forces::Drag`] force.
//!
//! [`super::Thermosphere`] needs the geodetic altitude and latitude, the local solar time and the day
//! of year — none of which the bare inertial position carries. This wrapper supplies them from the
//! **same closure [`crate::nrlmsise00::Nrlmsise00Atmosphere`] takes**, `(t, r) → (altitude_km,
//! geodetic_latitude_deg, geodetic_longitude_deg)`, so one geodetic provider serves both models and a
//! caller can swap the atmosphere under a drag force without rebuilding its frame plumbing. The local
//! solar time and the day of year are then this wrapper's arithmetic, not the caller's, from the epoch
//! UT and the longitude — again exactly as the NRLMSISE-00 wrapper does it.
//!
//! The drivers are held constant over the span, as operational orbit determination does with a single
//! space-weather record; the day of year advances with the propagation time. [`super::SdtmDrivers::
//! storm`] is a state run over an index *history*, so it is computed once per epoch by
//! [`crate::space_weather::sdtm_drivers`] and held here — the history belongs to whoever fetched it.
//!
//! # Why this wrapper does not evaluate the model twice
//!
//! The [`crate::forces::Atmosphere`] density is written over [`Scalar`] so its gradient enters the
//! state-transition matrix, and the *ported* atmospheres here are f64-only: each is transcribed Fortran
//! over fixed tables, so its wrapper reads a scale height from a second evaluation 10 km higher and
//! lifts the value through `ρ = ρ₀·exp(−(|r| − |r₀|)/H)`.
//!
//! SDTM has a closed form, so `H` is exact and free: [`super::Thermosphere::density_gradient`] returns
//! `dρ/dz` analytically beside `ρ` from one evaluation, and `1/H = −(dρ/dz)/ρ`. The lift is therefore
//! the same expression as the other three wrappers' with the *exact* local logarithmic gradient in it
//! instead of a 10 km finite difference of one, at half the model evaluations. What stays approximate
//! is only what is approximate for all four: the horizontal gradient is dropped, and `d z/d|r|` is
//! taken as one, which is exact at the pole and the equator and low by 5.4e-6 at 45° latitude, where
//! the ellipsoid normal is 11.5′ off the radius.
//!
//! Below [`super::BOUNDARY_ALTITUDE`] the model returns the boundary density with a zero gradient, so
//! the lift degenerates to that constant rather than to an infinity — `1/H`, not `H`, is what the
//! expression carries, and it is zero there.
//!
//! What checks it: eight of the tests below, whose residuals `docs/verification.md` §10.16 publishes —
//! the value bit-equal to the field at the named point, the dual gradient against the analytic
//! `dρ/dz·r̂` to 1e-14 relative, the boundary degeneracy, and a 6015-point positivity sweep.
//!
//! Gate: `derivations/atmosphere_lift.mac` shows that with `1/H = −ρ'/ρ` the lift's gradient at `r₀`
//! is `ρ'·r̂` and that on a spherical geodetic map the radial chain rule gives the same vector. The
//! module tests `the_gradient_the_stm_reads_is_the_fields_own_analytic_one` (G1) and
//! `the_lift_gradient_is_the_radial_derivative_of_the_density` (G2) hold the shipped wrapper to it.

use spacelab_autodiff::Scalar;

use crate::forces::Atmosphere;

use super::{SdtmDrivers, SdtmPoint, Thermosphere};

/// SDTM wrapped as a drag [`Atmosphere`]. See the module docs for the geodetic closure and the lift.
pub struct SdtmAtmosphere<G, const K: usize> {
    /// The fitted field — [`super::coefficients::FIELD`] for the shipped one.
    pub field: Thermosphere<K>,
    /// The indices, held constant over the span. `storm` is [`super::StormResponse::response`] over
    /// the caller's index history, so it must have been run through **this field's** response
    /// ([`Thermosphere::storm`]); a response built with another is another model's driver.
    pub drivers: SdtmDrivers,
    /// Day of year at propagation time `t = 0`, `1.0` at the start of 1 January.
    pub doy0: f64,
    /// UT seconds into the day at `t = 0`.
    pub sec0: f64,
    /// Maps `(t, inertial position in metres)` to `(altitude_km, geodetic_latitude_deg,
    /// geodetic_longitude_deg)`; the caller supplies the Earth-rotation and ellipsoid model.
    pub geodetic: G,
}

impl<G: Fn(f64, [f64; 3]) -> (f64, f64, f64), const K: usize> SdtmAtmosphere<G, K> {
    /// The model's own evaluation point at `t`, from the injected geodetic provider.
    fn point(&self, t: f64, rf: [f64; 3]) -> SdtmPoint {
        let (alt_km, lat_deg, lon_deg) = (self.geodetic)(t, rf);
        let total_sec = self.sec0 + t;
        let sec = total_sec - 86_400.0 * libm::floor(total_sec / 86_400.0);
        let mut lst = sec / 3_600.0 + lon_deg / 15.0;
        lst -= 24.0 * libm::floor(lst / 24.0); // wrap to [0, 24) hours
        SdtmPoint {
            altitude: alt_km * 1_000.0,
            latitude: lat_deg.to_radians(),
            local_solar_time: lst,
            day_of_year: self.doy0 + total_sec / 86_400.0,
        }
    }
}

impl<G: Fn(f64, [f64; 3]) -> (f64, f64, f64), const K: usize> Atmosphere for SdtmAtmosphere<G, K> {
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let (rho, grad) = self
            .field
            .density_gradient(&self.point(t, rf), &self.drivers);
        // 1/H = −(dρ/dz)/ρ, exact and unguarded: the profile is a sum of positive exponentials of a
        // *bounded* geopotential height, so it has no zero to divide by anywhere it is defined —
        // `the_profile_never_underflows_so_the_reciprocal_scale_height_needs_no_guard` measures that.
        let inv_h = -grad / rho;
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let r0 = S::from_f64(libm::sqrt(rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2]));
        S::from_f64(rho) * ((r0 - r_mag) * S::from_f64(inv_h)).exp()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // several assertions here are exact by construction

    use super::*;
    use crate::sdtm::{BOUNDARY_ALTITUDE, boundary_density, coefficients};

    /// About 1130 km up, off every axis so no component is degenerate.
    const R_REF: [f64; 3] = [4.1e6, 3.3e6, 5.2e6];
    const DOY0: f64 = 172.0;
    const SEC0: f64 = 29_000.0;

    fn drivers() -> SdtmDrivers {
        SdtmDrivers {
            f107: 128.0,
            f107a: 141.0,
            storm: 22.0,
        }
    }

    fn at(
        alt_km: f64,
        lat_deg: f64,
        lon_deg: f64,
    ) -> SdtmAtmosphere<impl Fn(f64, [f64; 3]) -> (f64, f64, f64), 9> {
        SdtmAtmosphere {
            field: coefficients::FIELD,
            drivers: drivers(),
            doy0: DOY0,
            sec0: SEC0,
            // Fixed geodetic output: these tests are about what the wrapper does with the triple, not
            // about the ellipsoid conversion, which the caller owns.
            geodetic: move |_t, _r| (alt_km, lat_deg, lon_deg),
        }
    }

    /// The field's own `(ρ, dρ/dz)` at the point the wrapper should have assembled, built here from
    /// the physical definitions rather than from the wrapper: apparent local solar time is UT plus the
    /// longitude at 15°/h, and the day of year carries the fraction of the day.
    fn reference(doy: f64, sec: f64, alt_km: f64, lat_deg: f64, lon_deg: f64) -> (f64, f64) {
        let mut lst = sec / 3_600.0 + lon_deg / 15.0;
        lst -= 24.0 * libm::floor(lst / 24.0);
        coefficients::FIELD.density_gradient(
            &SdtmPoint {
                altitude: alt_km * 1_000.0,
                latitude: lat_deg.to_radians(),
                local_solar_time: lst,
                day_of_year: doy + sec / 86_400.0,
            },
            &drivers(),
        )
    }

    /// At the point the geodetic closure names, the exponential lift is `exp(0)`, so the wrapper's
    /// answer is the field's — exactly, not approximately. That makes this one assertion cover the
    /// whole point assembly: the `km → m` altitude factor, the degrees-to-radians latitude, the local
    /// solar time from UT and longitude, and the fractional day of year.
    #[test]
    fn the_density_at_the_named_point_is_the_fields_own() {
        for (alt_km, lat_deg, lon_deg) in [
            (400.0, 12.0, -47.0),
            (180.0, -63.0, 121.0),
            (900.0, 78.0, 5.0),
        ] {
            let got: f64 = at(alt_km, lat_deg, lon_deg).density(0.0, &R_REF);
            let (want, _) = reference(DOY0, SEC0, alt_km, lat_deg, lon_deg);
            assert_eq!(got, want, "alt {alt_km} km, lat {lat_deg}, lon {lon_deg}");
        }
    }

    /// The longitude can put apparent local solar time past 24 h or below 0 h, and the model's
    /// harmonic basis is periodic in it — so the wrap has to happen, and it has to happen the way the
    /// fit's own data did it (the density product reports local solar time in `[0, 24)`).
    #[test]
    fn the_local_solar_time_wraps_into_the_day_the_basis_is_written_over() {
        // SEC0 is 08:03 UT: +170° is past midnight, −170° is before the previous one.
        for lon_deg in [170.0, -170.0, 0.0, 179.999, -179.999] {
            let atm = at(430.0, 17.0, lon_deg);
            let lst = atm.point(0.0, R_REF).local_solar_time;
            assert!((0.0..24.0).contains(&lst), "lon {lon_deg} gave lst {lst}");
            let (want, _) = reference(DOY0, SEC0, 430.0, 17.0, lon_deg);
            assert_eq!(atm.density(0.0, &R_REF), want, "lon {lon_deg}");
        }
    }

    /// `sec0 + t` is a running UT that has to reach both the time of day and the day of year. Every
    /// other test here sits at `t = 0`, where the split is trivial; this one steps across midnight in
    /// both directions and whole days beyond, which is the only way the `86 400`, the `floor` and the
    /// day advance are separable from each other.
    #[test]
    fn the_running_seconds_reach_both_the_time_of_day_and_the_day_of_year() {
        let (alt_km, lat_deg, lon_deg) = (350.0, -22.0, 96.0);
        let atm = at(alt_km, lat_deg, lon_deg);
        for (label, t, doy, sec) in [
            (
                "one second before midnight",
                86_400.0 - SEC0 - 1.0,
                DOY0,
                86_399.0,
            ),
            (
                "one second after midnight",
                86_400.0 - SEC0 + 1.0,
                DOY0 + 1.0,
                1.0,
            ),
            ("a full day later", 86_400.0, DOY0 + 1.0, SEC0),
            (
                "three days later",
                3.0 * 86_400.0 + 600.0,
                DOY0 + 3.0,
                SEC0 + 600.0,
            ),
        ] {
            let got: f64 = atm.density(t, &R_REF);
            let (want, _) = reference(doy, sec, alt_km, lat_deg, lon_deg);
            assert_eq!(got, want, "{label}");
        }
    }

    /// In `f64` the lift is the identity: `r₀` is read back from the very position that was passed in,
    /// so the exponent is `exp(0)` for every `r` and the answer is the field's value at whatever point
    /// the geodetic closure named. That is the design — the exponential exists to give the STM a
    /// gradient, not to move the value — and it is worth pinning, because a test written on the
    /// opposite assumption would read the exponent's arithmetic as untested when it is unreachable.
    #[test]
    fn the_f64_value_does_not_depend_on_the_radius_at_a_fixed_geodetic_point() {
        let atm = at(420.0, 33.0, -118.0);
        let (want, _) = reference(DOY0, SEC0, 420.0, 33.0, -118.0);
        for scale in [0.5, 0.999, 1.0, 1.001, 3.0] {
            let r = R_REF.map(|c| c * scale);
            assert_eq!(atm.density(0.0, &r), want, "radius scaled by {scale}");
        }
    }

    /// (G1) The gradient the state-transition matrix reads is the field's **own** `dρ/dz`, projected on
    /// `r̂` — not a 10 km finite difference of it, which is what the other three wrappers in this crate
    /// have to use. With `1/H = −(dρ/dz)/ρ` the lift's derivative at `r = r₀` is `ρ·(−1/H) = dρ/dz`
    /// identically, so this pins the sign of the exponent, the direction of `r₀ − ‖r‖` and the
    /// reciprocal together against a number the wrapper did not compute. The identity is
    /// `derivations/atmosphere_lift.mac`'s `analytic_lift_gradient_1..3`.
    #[test]
    fn the_gradient_the_stm_reads_is_the_fields_own_analytic_one() {
        use spacelab_autodiff::Dual;

        for (alt_km, lat_deg, lon_deg) in [
            (420.0, 33.0, -118.0),
            (150.0, -71.0, 44.0),
            (1_100.0, 8.0, 150.0),
        ] {
            let (rho, dz) = reference(DOY0, SEC0, alt_km, lat_deg, lon_deg);
            let rd = [
                Dual::<3>::variable(R_REF[0], 0),
                Dual::<3>::variable(R_REF[1], 1),
                Dual::<3>::variable(R_REF[2], 2),
            ];
            let dual = at(alt_km, lat_deg, lon_deg).density(0.0, &rd);
            assert_eq!(dual.value(), rho, "alt {alt_km}: the value moved");

            let r0 = libm::sqrt(R_REF.iter().map(|c| c * c).sum::<f64>());
            let grad = dual.grad();
            for i in 0..3 {
                let want = dz * (R_REF[i] / r0);
                let miss = libm::fabs(grad[i] - want) / libm::fabs(want);
                assert!(miss < 1e-14, "alt {alt_km}, axis {i}: {miss:.3e}");
            }
        }
    }

    /// (G2) The gradient the STM reads is the derivative of the density the wrapper returns, measured
    /// on the wrapper alone.
    ///
    /// With a spherical geodetic map `z = |r| − R` the f64 value does move with the radius, so the
    /// derivative it defines can be taken by a central difference of the wrapper's own output and
    /// compared with its dual gradient — no call to `density_gradient` in the reference.
    /// `derivations/atmosphere_lift.mac` (`spherical_chain_rule_1..3`) is the chain rule this rests
    /// on: `∂ρ(|r| − R)/∂xᵢ = ρ'·xᵢ/|r|`. A 1 m step leaves a truncation of `(δ/H)²/6 < 3e-9` against
    /// the tightest 8 km layer and a rounding of `ε·H/δ < 2e-12`, so 1e-7 relative is a bound on
    /// the wrong-derivative case, which would be off at order one.
    #[test]
    fn the_lift_gradient_is_the_radial_derivative_of_the_density() {
        use spacelab_autodiff::Dual;

        const EARTH: f64 = 6_371_000.0;
        let spherical = |_t: f64, r: [f64; 3]| {
            let alt_km = (libm::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) - EARTH) / 1_000.0;
            (alt_km, 33.0, -118.0)
        };
        let atm = SdtmAtmosphere {
            field: coefficients::FIELD,
            drivers: drivers(),
            doy0: DOY0,
            sec0: SEC0,
            geodetic: spherical,
        };
        let dir = [0.6, -0.48, 0.64];
        for alt_km in [125.0, 250.0, 420.0, 800.0, 1_500.0] {
            let radius = EARTH + alt_km * 1_000.0;
            let at = |dr: f64| dir.map(|c| c * (radius + dr));
            let step = 1.0;
            let slope = (atm.density::<f64>(0.0, &at(step)) - atm.density::<f64>(0.0, &at(-step)))
                / (2.0 * step);
            let r = at(0.0);
            let rd = [
                Dual::<3>::variable(r[0], 0),
                Dual::<3>::variable(r[1], 1),
                Dual::<3>::variable(r[2], 2),
            ];
            let grad = atm.density(0.0, &rd).grad();
            for i in 0..3 {
                let want = slope * dir[i];
                let miss = libm::fabs(grad[i] - want) / libm::fabs(want);
                assert!(miss < 1e-7, "alt {alt_km} km, axis {i}: {miss:.3e}");
            }
        }
    }

    /// Below the boundary the field returns its constant with a zero gradient, so `1/H` is zero and
    /// the lift is that constant for every radius — the degenerate case that would be an infinity if
    /// the wrapper carried `H` instead of its reciprocal.
    #[test]
    fn below_the_boundary_the_lift_is_the_constant_the_field_returns() {
        use spacelab_autodiff::Dual;

        let want = coefficients::FIELD.boundary_scale() * boundary_density();
        for alt_km in [0.0, 60.0, BOUNDARY_ALTITUDE / 1_000.0] {
            let atm = at(alt_km, 40.0, 20.0);
            let rd = [
                Dual::<3>::variable(R_REF[0], 0),
                Dual::<3>::variable(R_REF[1], 1),
                Dual::<3>::variable(R_REF[2], 2),
            ];
            let dual = atm.density(0.0, &rd);
            assert_eq!(dual.value(), want, "alt {alt_km} km");
            assert_eq!(dual.grad(), [0.0; 3], "alt {alt_km} km has a gradient");
        }
    }

    /// `1/H = −(dρ/dz)/ρ` is written without a guard on `ρ`, and that is a claim about the profile
    /// rather than about the code: the geopotential height it is a function of is **bounded** by
    /// `R + z₀ ≈ 6477 km` however high the geometric altitude goes, so the species exponentials cannot
    /// run away and the sum has no zero to divide by. Measured over 6015 points of the admissible
    /// envelope — out to 652 000 km, a hundred Earth radii and far past any orbit, at quiet and at
    /// `ap` 400: the smallest density is **9.0e-33 kg/m³**, 290 orders above f64's floor, and the
    /// tightest layer **8.25 km**, at the boundary where the Bates gradient is steepest.
    #[test]
    fn the_profile_never_underflows_so_the_reciprocal_scale_height_needs_no_guard() {
        let mut smallest = f64::INFINITY;
        let mut tightest = f64::INFINITY;
        let mut points = 0u32;
        for step in 0..=400 {
            // 120.001 km to 6.4e5 km, log-spaced: a hundred Earth radii, far past any orbit.
            let alt_km = 120.001 * libm::exp(f64::from(step) * 0.0215);
            for lat_deg in [-89.0_f64, -45.0, 0.0, 45.0, 89.0] {
                for (f107, storm) in [(65.0, 0.0), (150.0, 22.0), (300.0, 400.0)] {
                    let (rho, dz) = coefficients::FIELD.density_gradient(
                        &SdtmPoint {
                            altitude: alt_km * 1_000.0,
                            latitude: lat_deg.to_radians(),
                            local_solar_time: 14.0,
                            day_of_year: 200.0,
                        },
                        &SdtmDrivers {
                            f107,
                            f107a: f107,
                            storm,
                        },
                    );
                    points += 1;
                    assert!(rho > 0.0, "ρ = {rho:e} at {alt_km} km, lat {lat_deg}");
                    let inv_h = -dz / rho;
                    assert!(
                        inv_h.is_finite() && inv_h > 0.0,
                        "1/H = {inv_h:e} at {alt_km} km, lat {lat_deg}"
                    );
                    smallest = smallest.min(rho);
                    tightest = tightest.min(1.0 / inv_h);
                }
            }
        }
        assert_eq!(points, 6_015);
        // Bands, not equalities: a refit moves these, and what is being pinned is that the smallest
        // density stays orders above f64's floor and the tightest layer orders above zero.
        assert!(
            smallest > 1e-60,
            "the profile reached {smallest:e} kg/m³, on its way to f64's floor"
        );
        assert!(
            (1.0e3..1.0e6).contains(&tightest),
            "the tightest layer is {tightest:e} m"
        );
    }

    /// The ship requirement of `plans/16` Ф15.I: the calibration decorator of
    /// [`crate::density_calibration`] takes this wrapper as its base, **and takes the same geodetic
    /// closure**, so a caller assembles one provider and gets both. Pinned as the product, since a
    /// decorator that silently dropped its multiplier would still compile and still be an
    /// [`Atmosphere`].
    #[test]
    fn the_calibration_decorator_wraps_it_with_the_same_geodetic_closure() {
        use crate::density_calibration::{CalibratedAtmosphere, DensityCorrection};

        let (alt_km, lat_deg, lon_deg) = (420.0, 33.0, -118.0);
        let geodetic = move |_t: f64, _r: [f64; 3]| (alt_km, lat_deg, lon_deg);
        let correction = DensityCorrection::<9>::from_coefficients([
            0.11, -0.02, 0.05, 0.01, -0.03, 0.0, 0.0, 0.0, 0.0,
        ]);
        let calibrated = CalibratedAtmosphere::new(
            SdtmAtmosphere {
                field: coefficients::FIELD,
                drivers: drivers(),
                doy0: DOY0,
                sec0: SEC0,
                geodetic,
            },
            correction,
            geodetic,
            SEC0,
        );
        let (base, _) = reference(DOY0, SEC0, alt_km, lat_deg, lon_deg);
        let mut lst = SEC0 / 3_600.0 + lon_deg / 15.0;
        lst -= 24.0 * libm::floor(lst / 24.0);
        let want = base * correction.factor(lat_deg.to_radians(), lst);
        assert_eq!(calibrated.density::<f64>(0.0, &R_REF), want);
    }
}
