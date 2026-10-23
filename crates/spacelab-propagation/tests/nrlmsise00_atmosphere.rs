//! The NRLMSISE-00 drag-atmosphere wrapper, on a real LEO point.
//!
//! `atmosphere.rs` adapts the verified NRLMSISE-00 model (`gtd7d`) into the `Atmosphere` the
//! `Drag` force consumes: it assembles the model input from the epoch and a geodetic closure,
//! converts g/cm³ to kg/m³, and — so the density can enter the state-transition matrix — lifts the
//! f64 value to any `Scalar` through a local exponential whose scale height comes from a second
//! evaluation 10 km higher. This exercises all three, which had no coverage: the value equals the
//! underlying model, the epoch (day-of-year / UT) advance is applied, and the `Scalar` gradient is
//! exactly the local `-ρ₀/H`.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_possible_truncation,
    clippy::doc_markdown
)]

use spacelab_autodiff::Dual;
use spacelab_propagation::forces::Atmosphere;
use spacelab_propagation::{Nrlmsise00Atmosphere, Nrlmsise00Flags, Nrlmsise00Input, gtd7d};

const RE_M: f64 = 6_378_137.0;
const DOY0: i32 = 100;
const SEC0: f64 = 43_200.0; // noon UT
const F107: f64 = 150.0;
const F107A: f64 = 150.0;
const AP: f64 = 4.0;
const G_LAT: f64 = 30.0;
const G_LON: f64 = 45.0;

/// The wrapper with a geodetic closure that reads altitude off the inertial radius and holds
/// lat/lon fixed — enough to drive it deterministically. (The real `Drag` force injects the
/// Earth-frame conversion here.)
fn atmosphere() -> Nrlmsise00Atmosphere<impl Fn(f64, [f64; 3]) -> (f64, f64, f64)> {
    Nrlmsise00Atmosphere {
        doy0: DOY0,
        sec0: SEC0,
        f107: F107,
        f107a: F107A,
        ap: AP,
        geodetic: |_t: f64, r: [f64; 3]| {
            let mag = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            ((mag - RE_M) / 1000.0, G_LAT, G_LON)
        },
    }
}

/// Reproduce the wrapper's model input at time `t` and altitude, then read (ρ₀, ρ_up) in kg/m³
/// straight from `gtd7d`. This is the oracle the wrapper must reproduce.
fn expected_pair(t: f64, alt_km: f64) -> (f64, f64) {
    let total = SEC0 + t;
    let sec = total - 86_400.0 * (total / 86_400.0).floor();
    let doy = DOY0 + (total / 86_400.0).floor() as i32;
    let lst = sec / 3_600.0 + G_LON / 15.0;
    let base = Nrlmsise00Input {
        year: 0,
        doy,
        sec,
        alt: alt_km,
        g_lat: G_LAT,
        g_long: G_LON,
        lst,
        f107a: F107A,
        f107: F107,
        ap: AP,
        ap_array: None,
    };
    let flags = Nrlmsise00Flags::default();
    let mut up = base;
    up.alt = alt_km + 10.0;
    (
        gtd7d(&base, &flags).d[5] * 1000.0,
        gtd7d(&up, &flags).d[5] * 1000.0,
    )
}

#[test]
fn density_matches_the_underlying_model_including_the_epoch_advance() {
    let atmo = atmosphere();
    let r = [RE_M + 400_000.0, 0.0, 0.0]; // 400 km altitude on the equator
    // t = 0, then times that cross day boundaries so the day-of-year / UT advance is exercised.
    for &t in &[0.0, 90_000.0, 200_000.0] {
        let (rho0, _) = expected_pair(t, 400.0);
        let got = atmo.density::<f64>(t, &r);
        assert!(rho0 > 0.0, "a 400 km density must be positive");
        assert!(
            (got - rho0).abs() <= 1e-12 * rho0,
            "t={t}: wrapper {got:e} != gtd7d {rho0:e} kg/m³",
        );
    }
}

#[test]
fn the_scalar_gradient_is_the_local_minus_rho_over_h() {
    let atmo = atmosphere();
    let rf = [RE_M + 400_000.0, 0.0, 0.0];
    let (rho0, rho_up) = expected_pair(0.0, 400.0);
    let h_scale = -10_000.0 / (rho_up / rho0).ln();

    // Seed all three position components; grad() is then ∇ρ in kg/m³ per metre.
    let r = [
        Dual::<3>::variable(rf[0], 0),
        Dual::<3>::variable(rf[1], 1),
        Dual::<3>::variable(rf[2], 2),
    ];
    let rho = atmo.density::<Dual<3>>(0.0, &r);
    let grad = rho.grad();

    // The value still equals the underlying model.
    assert!((rho.value() - rho0).abs() <= 1e-12 * rho0);
    // ∇ρ = -ρ₀/H · r̂. Here r̂ = (1, 0, 0), so ∂ρ/∂x = -ρ₀/H and the tangential parts vanish.
    let expected_dx = -rho0 / h_scale;
    assert!(
        (grad[0] - expected_dx).abs() <= 1e-6 * expected_dx.abs(),
        "∂ρ/∂x {:e} != -ρ₀/H {expected_dx:e}",
        grad[0],
    );
    assert!(
        grad[1].abs() <= 1e-18 && grad[2].abs() <= 1e-18,
        "an off-radial gradient leaked: {grad:?}",
    );
    // Physical sanity: density falls with altitude, so the radial derivative is negative, and the
    // thermospheric scale height at 400 km is tens of kilometres.
    assert!(grad[0] < 0.0, "density must decrease with altitude");
    assert!(
        h_scale > 10_000.0 && h_scale < 200_000.0,
        "unphysical scale height {h_scale} m",
    );
}
