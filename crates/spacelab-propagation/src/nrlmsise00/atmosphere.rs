//! NRLMSISE-00 as a [`crate::forces::Atmosphere`] for the [`crate::forces::Drag`] force.
//!
//! The [`super`] model is f64 and needs geodetic latitude/longitude/altitude, the day of year, UT and
//! local solar time — none of which the bare inertial position carries. This wrapper supplies them:
//! the epoch (day of year and UT seconds at `t = 0`) and constant space weather (F10.7, Ap) are held
//! as fields; the inertial→geodetic conversion (which needs the Earth-rotation frame) is injected as a
//! closure, so this crate stays independent of the frames/Earth crates. `gtd7d` — the effective total
//! mass density for drag, including anomalous oxygen — supplies the value.
//!
//! The [`crate::forces::Atmosphere`] density is written over [`Scalar`] so its gradient can enter the
//! state-transition matrix, but the empirical model is defined only in f64. The value is therefore
//! evaluated in f64 and lifted to `S` through a **local exponential** `ρ = ρ₀·exp(−(|r|−|r₀|)/H)`,
//! with the scale height `H` read from a second model evaluation 10 km higher. The STM then sees the
//! correct local `dρ/d|r|` without differentiating the whole empirical model — the same device
//! operational orbit determination uses for tabulated density.
//!
//! Gate: `derivations/atmosphere_lift.mac` shows the lift is `ρ₀` at `r₀` with gradient
//! `−(ρ₀/H)·r̂`, that the 10 km quotient is the layer's own `H` on an exponential layer (and the
//! harmonic mean of the local `H` on any other), and that the day split and hour wrap land where the
//! tests below step. The module tests `the_density_at_the_named_point_is_the_model_converted_to_si`
//! (G3) and `the_gradient_the_stm_reads_is_minus_rho_over_the_scale_height` (G1) hold the shipped
//! wrapper to it; the model's own structure is gated in `super`.

#![allow(clippy::cast_possible_truncation)] // day-of-year advance is a small, in-range integer

use spacelab_autodiff::Scalar;

use crate::forces::Atmosphere;

use super::{Nrlmsise00Flags, Nrlmsise00Input, gtd7d};

/// NRLMSISE-00 wrapped as a drag [`Atmosphere`]. See the module docs for the local-exponential lift.
pub struct Nrlmsise00Atmosphere<G> {
    /// Day of year (1..=366) at propagation time `t = 0`.
    pub doy0: i32,
    /// UT seconds into the day at `t = 0`.
    pub sec0: f64,
    /// Daily F10.7 flux (previous day), in solar-flux units.
    pub f107: f64,
    /// 81-day average F10.7 flux centred on the day.
    pub f107a: f64,
    /// Daily magnetic index Ap.
    pub ap: f64,
    /// Maps `(t, inertial position in metres)` to `(altitude_km, geodetic_latitude_deg,
    /// geodetic_longitude_deg)`; the caller supplies the Earth-rotation and ellipsoid model.
    pub geodetic: G,
}

impl<G: Fn(f64, [f64; 3]) -> (f64, f64, f64)> Nrlmsise00Atmosphere<G> {
    /// Total mass density (kg/m³) at the point, and at 10 km higher (for the local scale height).
    fn density_pair(&self, t: f64, rf: [f64; 3]) -> (f64, f64) {
        let (alt, lat, lon) = (self.geodetic)(t, rf);
        let total_sec = self.sec0 + t;
        let sec = total_sec - 86_400.0 * libm::floor(total_sec / 86_400.0);
        let doy = self.doy0 + libm::floor(total_sec / 86_400.0) as i32;
        let lst = sec / 3_600.0 + lon / 15.0;
        let base = Nrlmsise00Input {
            year: 0,
            doy,
            sec,
            alt,
            g_lat: lat,
            g_long: lon,
            lst,
            f107a: self.f107a,
            f107: self.f107,
            ap: self.ap,
            ap_array: None,
        };
        let flags = Nrlmsise00Flags::default();
        let mut up = base;
        up.alt = alt + 10.0;
        // d[5] is g/cm³ from gtd7d (includes anomalous oxygen); ×1000 → kg/m³.
        (
            gtd7d(&base, &flags).d[5] * 1000.0,
            gtd7d(&up, &flags).d[5] * 1000.0,
        )
    }
}

impl<G: Fn(f64, [f64; 3]) -> (f64, f64, f64)> Atmosphere for Nrlmsise00Atmosphere<G> {
    fn density<S: Scalar>(&self, t: f64, r: &[S; 3]) -> S {
        let rf = [r[0].value(), r[1].value(), r[2].value()];
        let (rho0, rho_up) = self.density_pair(t, rf);
        // Local scale height H = −Δh / ln(ρ_up/ρ₀), Δh = 10 km; fall back to 50 km if degenerate.
        // A guard on the two densities being positive would be redundant: a zero or negative one
        // sends the logarithm to an infinity or a NaN, and both quotients then fail the finiteness
        // and floor tests below — `-10 km / ±∞` is `∓0.0`, which is not above `1 km`.
        let hh = -10_000.0 / libm::log(rho_up / rho0);
        let h_scale = if hh.is_finite() && hh > 1_000.0 {
            hh
        } else {
            50_000.0
        };
        let r_mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let r0 = S::from_f64(libm::sqrt(rf[0] * rf[0] + rf[1] * rf[1] + rf[2] * rf[2]));
        S::from_f64(rho0) * ((r0 - r_mag) * S::from_f64(1.0 / h_scale)).exp()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)] // several assertions here are exact by construction

    use super::*;
    use crate::nrlmsise00::{Nrlmsise00Input, gtd7d};

    /// About 1130 km up, off every axis so no component is degenerate.
    const R_REF: [f64; 3] = [4.1e6, 3.3e6, 5.2e6];
    const DOY0: i32 = 172;
    const SEC0: f64 = 29_000.0;

    fn at(
        alt: f64,
        lat: f64,
        lon: f64,
        ap: f64,
        f107: f64,
    ) -> Nrlmsise00Atmosphere<impl Fn(f64, [f64; 3]) -> (f64, f64, f64)> {
        Nrlmsise00Atmosphere {
            doy0: DOY0,
            sec0: SEC0,
            f107,
            f107a: f107,
            ap,
            // Fixed geodetic output: this test is about what the wrapper does with the triple, not
            // about the ellipsoid conversion, which the caller owns.
            geodetic: move |_t, _r| (alt, lat, lon),
        }
    }

    /// The model's own total mass density in SI, built from the physical definitions rather than
    /// from the wrapper: apparent local solar time is UT plus the longitude at 15°/h.
    fn reference_si(doy: i32, sec: f64, alt: f64, lat: f64, lon: f64, ap: f64, f107: f64) -> f64 {
        let input = Nrlmsise00Input {
            year: 0,
            doy,
            sec,
            alt,
            g_lat: lat,
            g_long: lon,
            lst: sec / 3_600.0 + lon / 15.0,
            f107a: f107,
            f107,
            ap,
            ap_array: None,
        };
        gtd7d(&input, &Nrlmsise00Flags::default()).d[5] * 1_000.0
    }

    /// (G3) At the point the geodetic closure names, the exponential lift is `exp(0)`
    /// (`derivations/atmosphere_lift.mac`, `lift_value_at_r0`), so the wrapper's answer is the
    /// model's — exactly, not approximately. That makes this one assertion cover the
    /// whole input assembly (day, UT seconds, local solar time, the three space-weather fields) and
    /// the `g/cm³ → kg/m³` factor of 1000, none of which the crate tested anywhere before.
    #[test]
    fn the_density_at_the_named_point_is_the_model_converted_to_si() {
        for (alt, lat, lon, ap, f107) in [
            (400.0, 12.0, -47.0, 4.0, 150.0),
            (85.0, -63.0, 121.0, 40.0, 70.0),
            (900.0, 78.0, 5.0, 15.0, 220.0),
        ] {
            let got: f64 = at(alt, lat, lon, ap, f107).density(0.0, &R_REF);
            let want = reference_si(DOY0, SEC0, alt, lat, lon, ap, f107);
            assert_eq!(got, want, "alt {alt} km, lat {lat}, lon {lon}");
        }
    }

    /// `sec0 + t` is a running UT that has to be split back into a day count and a time of day. Every
    /// other test here sits at `t = 0`, where the split is trivial; this one steps across midnight in
    /// both directions and one whole day beyond, which is the only way the `floor`, the `86 400` and
    /// the day increment are separable from each other.
    #[test]
    fn the_running_seconds_split_into_a_day_and_a_time_of_day() {
        let (alt, lat, lon, ap, f107) = (350.0, -22.0, 96.0, 12.0, 130.0);
        let atm = at(alt, lat, lon, ap, f107);
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
                DOY0 + 1,
                1.0,
            ),
            ("a full day later", 86_400.0, DOY0 + 1, SEC0),
            (
                "three days later",
                3.0 * 86_400.0 + 600.0,
                DOY0 + 3,
                SEC0 + 600.0,
            ),
        ] {
            let got: f64 = atm.density(t, &R_REF);
            assert_eq!(
                got,
                reference_si(doy, sec, alt, lat, lon, ap, f107),
                "{label}"
            );
        }
    }

    /// In `f64` the lift is the identity: `r₀` is read back from the very position that was passed
    /// in, so the exponent is `exp(0)` for every `r` and the answer is the model's value at whatever
    /// point the geodetic closure named. That is the design — the exponential exists to give the STM
    /// a gradient, not to move the value — and it is worth pinning, because a test written on the
    /// opposite assumption would read the exponent's arithmetic as untested when it is unreachable.
    #[test]
    fn the_f64_value_does_not_depend_on_the_radius_at_a_fixed_geodetic_point() {
        let (alt, lat, lon, ap, f107) = (420.0, 33.0, -118.0, 9.0, 160.0);
        let atm = at(alt, lat, lon, ap, f107);
        let want = reference_si(DOY0, SEC0, alt, lat, lon, ap, f107);
        for scale in [0.5, 0.999, 1.0, 1.001, 3.0] {
            let r = R_REF.map(|c| c * scale);
            assert_eq!(atm.density(0.0, &r), want, "radius scaled by {scale}");
        }
    }

    /// (G1) The whole point of the exponential lift is that the state-transition matrix sees a
    /// gradient the empirical model cannot supply. `∂ρ/∂r = −(ρ/H)·r̂` in closed form
    /// (`derivations/atmosphere_lift.mac`, `lift_gradient_1..3`), with `H` rebuilt here
    /// from two direct model calls 10 km apart — so the exponent's sign, its `10 km` base and the
    /// direction of `r₀ − ‖r‖` are pinned against something the wrapper did not compute.
    ///
    /// The second fixture is the fallback arm. Above the auroral zone in a severe storm the total
    /// mass density is **not** monotone across a 10 km step — Joule heating lifts the layer — so
    /// `ln(ρ_up/ρ₀)` turns positive and the measured scale height comes out negative. Swept against
    /// the reference C over `0..1500 km` at three geometries, that happens between `102.5` and
    /// `117.5 km` at `85°N` with `ap = 300`, `F10.7 = 250`, and nowhere else; at `110 km` it is
    /// `−3674 m`. The fallback is a reachable branch on physical input, not defensive padding, and it
    /// has to use exactly `50 km`.
    #[test]
    fn the_gradient_the_stm_reads_is_minus_rho_over_the_scale_height() {
        use spacelab_autodiff::Dual;

        for (label, alt, lat, lon, ap, f107) in [
            ("a monotone layer", 420.0, 33.0, -118.0, 9.0, 160.0),
            ("a storm-lifted layer", 110.0, 85.0, 0.0, 300.0, 250.0),
        ] {
            let rho0 = reference_si(DOY0, SEC0, alt, lat, lon, ap, f107);
            let rho_up = reference_si(DOY0, SEC0, alt + 10.0, lat, lon, ap, f107);
            let measured = -10_000.0 / libm::log(rho_up / rho0);
            let h_scale = if measured.is_finite() && measured > 1_000.0 {
                measured
            } else {
                50_000.0
            };
            // The two fixtures must land on different arms, or one of them proves nothing.
            assert_eq!(
                h_scale == 50_000.0,
                rho_up >= rho0,
                "{label}: measured scale height {measured:e}"
            );

            let rd = [
                Dual::<3>::variable(R_REF[0], 0),
                Dual::<3>::variable(R_REF[1], 1),
                Dual::<3>::variable(R_REF[2], 2),
            ];
            let dual = at(alt, lat, lon, ap, f107).density(0.0, &rd);
            assert_eq!(
                dual.value(),
                rho0,
                "{label}: the value moved with the derivatives"
            );

            let r0 = libm::sqrt(dot(&R_REF, &R_REF));
            let grad = dual.grad();
            for i in 0..3 {
                let want = -rho0 / h_scale * (R_REF[i] / r0);
                let miss = libm::fabs(grad[i] - want) / libm::fabs(want);
                assert!(miss < 1e-13, "{label}, axis {i}: {miss:.3e}");
            }
        }
    }
    /// The `1 km` floor under the measured scale height never fires on an atmosphere this model
    /// describes, and that is a property of the model's coefficients rather than of the code.
    ///
    /// It matters because the floor is what decides between the measured height and the `50 km`
    /// fallback, and a mutation of its comparison is observable only at exactly `1000 m`. Swept over
    /// `0..1500 km`, every 15° of latitude and three levels of solar and magnetic activity —
    /// `11 739` points — the tightest layer the model produces is `3.13 km`, three times the floor,
    /// and the only points that reach the fallback are the `9` where the layer is not monotone at
    /// all. Pinned as a band so that a coefficient change which brings a real layer near `1 km`
    /// fails here instead of silently changing which arm the wrapper takes.
    #[test]
    fn the_scale_height_floor_is_far_below_any_layer_the_model_produces() {
        let (mut tightest, mut fallbacks, mut points) = (f64::INFINITY, 0u32, 0u32);
        for alt_step in 0..=300 {
            let alt = f64::from(alt_step) * 5.0;
            for lat_step in -6..=6 {
                let lat = f64::from(lat_step) * 15.0;
                for (ap, f107) in [(0.0, 65.0), (4.0, 150.0), (300.0, 250.0)] {
                    let rho0 = reference_si(DOY0, SEC0, alt, lat, 20.0, ap, f107);
                    let rho_up = reference_si(DOY0, SEC0, alt + 10.0, lat, 20.0, ap, f107);
                    let hh = -10_000.0 / libm::log(rho_up / rho0);
                    points += 1;
                    // No NaN sink here: the branch itself excludes the non-finite values, so the
                    // bare comparison cannot silently drop the sample the assertion is about.
                    if hh.is_finite() && hh > 0.0 {
                        if hh < tightest {
                            tightest = hh;
                        }
                    } else {
                        fallbacks += 1;
                    }
                }
            }
        }
        assert_eq!(points, 11_739);
        assert_eq!(fallbacks, 9, "the non-monotone corner moved");
        assert!(
            tightest > 3.0e3,
            "a real layer reached {tightest:e} m, near the 1 km floor"
        );
    }

    fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
}
