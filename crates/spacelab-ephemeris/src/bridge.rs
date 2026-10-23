//! Bridge a DE440 body into the propagator's [`Ephemeris`](spacelab_propagation::ephemeris::Ephemeris)
//! trait, so `ThirdBody` and `SolarRadiationPressure` fly on real ephemerides rather than the
//! low-precision analytic series.
//!
//! The trait wants a body's position **relative to the centre being propagated about**, in J2000
//! axes and in **metres**, at `t` seconds past a J2000-TT propagation epoch. ANISE returns km, so the
//! adapter scales by 1000 and offsets the epoch. The epoch is built with `Epoch::from_et_seconds`: ET
//! is seconds past J2000, agreeing with the trait's TT-seconds to under 2 ms — a sub-metre shift in a
//! `~10⁻⁷ m/s²` third-body term, far below what it warrants.
//!
//! The centre is the caller's: `EARTH_J2000` for a geocentric orbit, `MOON_J2000` for a
//! selenocentric one, where the Earth is then itself the dominant third body.
//!
//! Note: each `position` call is one ANISE translation (a Chebyshev evaluation). For a long
//! integration this is called once per body per RHS evaluation; correct but not free. A throughput
//! path would pre-sample the ephemeris onto a spline over the arc.

use spacelab_propagation::ephemeris::Ephemeris as EphemerisTrait;

use crate::{Ephemeris, Epoch, Frame};

/// A single perturbing body drawn from a loaded [`Ephemeris`], as the propagator's trait sees it.
#[derive(Clone, Copy)]
pub struct PerturbingBody<'a> {
    ephemeris: &'a Ephemeris,
    body: Frame,
    center: Frame,
    epoch_j2000_seconds: f64,
}

impl Ephemeris {
    /// A perturbing body for a force model: `body` (e.g. `frames::MOON_J2000`, `frames::SUN_J2000`)
    /// relative to `center`, with the propagation epoch as seconds past J2000.
    ///
    /// `center` is the body the integration is about, and must match the frame the state vector is
    /// expressed in — a third-body term computed about the wrong origin is not a small error but a
    /// different force.
    #[must_use]
    pub fn perturbing_body(
        &self,
        body: Frame,
        center: Frame,
        epoch_j2000_seconds: f64,
    ) -> PerturbingBody<'_> {
        PerturbingBody {
            ephemeris: self,
            body,
            center,
            epoch_j2000_seconds,
        }
    }
}

impl EphemerisTrait for PerturbingBody<'_> {
    fn position(&self, t: f64) -> [f64; 3] {
        let epoch = Epoch::from_et_seconds(self.epoch_j2000_seconds + t);
        // A query outside the kernel's span cannot yield a meaningful perturbation, so return the
        // trait's absent-body value rather than fabricate one -- see
        // `spacelab_propagation::ephemeris::is_absent`, which is what the forces read it through.
        // Keep t within the kernel's coverage all the same: no perturbation is a wrong answer too,
        // just a bounded one.
        self.ephemeris
            .state(self.body, self.center, epoch)
            .map_or([0.0; 3], |s| {
                [
                    s.position.x * 1000.0,
                    s.position.y * 1000.0,
                    s.position.z * 1000.0,
                ]
            })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use std::path::Path;

    use spacelab_propagation::ephemeris::Ephemeris as EphemerisTrait;
    use spacelab_propagation::forces::{Acceleration, ThirdBody};

    use anise::constants::frames::EARTH_J2000;

    use crate::frames::MOON_J2000;
    use crate::{Ephemeris, Epoch};

    const KERNEL: &str = "../../data/ephemeris/de440s.bsp";
    const MU_MOON: f64 = 4.902_800e12; // m³/s²

    #[test]
    fn perturbing_body_matches_horizons_and_drives_third_body() {
        if !Path::new(KERNEL).exists() {
            eprintln!("skipping: {KERNEL} not present — run tools/fetch-ephemeris.sh");
            return;
        }
        let eph = Ephemeris::load(KERNEL).expect("de440s.bsp loads");
        let moon = eph.perturbing_body(MOON_J2000, EARTH_J2000, 0.0); // epoch = J2000

        // The adapter reproduces the Horizons Moon vector, in metres.
        let p = moon.position(0.0);
        let reference = [-291_608.38e3, -266_716.83e3, -76_102.49e3]; // metres
        let err = ((p[0] - reference[0]).powi(2)
            + (p[1] - reference[1]).powi(2)
            + (p[2] - reference[2]).powi(2))
        .sqrt();
        assert!(err < 1_000.0, "adapter off Horizons by {err:.1} m");

        // Fed into a ThirdBody force, it produces a sensible lunar perturbation on a LEO satellite
        // (order 10⁻⁶ m/s²) — the whole point of the bridge.
        let force = ThirdBody {
            mu: MU_MOON,
            ephemeris: moon,
        };
        let r = [7_000e3, 0.0, 0.0];
        let v = [0.0, 7_500.0, 0.0];
        let a = force.acceleration::<f64>(0.0, &r, &v);
        let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        println!("DE440 lunar third-body accel on LEO: {mag:.3e} m/s²");
        assert!(
            (1e-8..1e-4).contains(&mag),
            "lunar third-body accel {mag:.3e} m/s² is out of range"
        );
    }

    /// `t` is an offset from the construction epoch, and the offset is an addition.
    ///
    /// The test above builds the body at J2000 and asks for `t = 0`, where `E + t`, `E − t` and
    /// `E · t` are the same number — so the epoch arithmetic, the one line of logic the adapter
    /// owns, went unchecked. Both the epoch and the offset are non-zero here, and the reference is
    /// the same kernel queried directly at the epoch the adapter should have formed.
    #[test]
    fn the_offset_is_added_to_the_construction_epoch() {
        if !Path::new(KERNEL).exists() {
            eprintln!("skipping: {KERNEL} not present — run tools/fetch-ephemeris.sh");
            return;
        }
        let eph = Ephemeris::load(KERNEL).expect("de440s.bsp loads");
        let base = 3.155_76e8; // ~2010-01-01, ten Julian years past J2000
        let offset = 7.0 * 86_400.0; // one week, over which the Moon moves ~10⁵ km

        let want = eph
            .state(
                MOON_J2000,
                EARTH_J2000,
                Epoch::from_et_seconds(base + offset),
            )
            .expect("inside the de440s span");
        let got = eph
            .perturbing_body(MOON_J2000, EARTH_J2000, base)
            .position(offset);
        for (k, (g, w)) in got
            .iter()
            .zip([want.position.x, want.position.y, want.position.z])
            .enumerate()
        {
            assert!(
                (g - w * 1000.0).abs() < 1e-6,
                "component {k}: {g} m, kernel says {} m",
                w * 1000.0,
            );
        }

        // And it is a *different* place from the construction epoch, so the assertion above is
        // not satisfied by an adapter that ignores `t` altogether.
        let at_base = eph
            .perturbing_body(MOON_J2000, EARTH_J2000, base)
            .position(0.0);
        let moved = ((got[0] - at_base[0]).powi(2)
            + (got[1] - at_base[1]).powi(2)
            + (got[2] - at_base[2]).powi(2))
        .sqrt();
        assert!(moved > 1e8, "the Moon moved only {moved:.3e} m in a week");
    }
}
