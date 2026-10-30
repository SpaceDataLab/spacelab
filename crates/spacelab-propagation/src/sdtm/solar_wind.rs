//! The solar-wind field's hourly storm drive (`docs/verification.md` §10.31, §10.32).
//!
//! [`super::solar_wind_coefficients::FIELD`] is SDTM-1 fitted again with its storm state run hourly
//! on `(1 − s)·ap60 + s·N`: GFZ's hourly `ap60`, and `N` Newell et al. (2007)'s solar-wind coupling
//! put on `ap60`'s scale by a quantile table, [`CouplingMap`]. Held out by mission family it lifts
//! the largest storms' peaks from 0.64 of the measured to 0.93, which no change to the storm state
//! itself did (§10.30, §10.32). It is a second field rather than the default because the solar wind
//! is measured less than an hour before it reaches the Earth and the north–south field the coupling
//! turns on is not forecast operationally: the 3-hourly [`super::coefficients::FIELD`] stays the
//! forecast path, and this one is for reconstruction and nowcasting.
//!
//! The field was fitted on OMNI's hourly averages, time-shifted from L1 to the bow-shock
//! nose. A caller with real-time L1 data shifts it the same way first; no real-time solar wind was
//! in the fit, and `docs/limitations.md` says so.
//!
//! The coupling, its half-angle form and its values at four winds are
//! `derivations/sdtm_solar_wind.mac`, with `sdtm_solar_wind.gp` the second engine.

use super::StormResponse;

/// Cadence of the solar-wind field's storm drive, seconds: `ap60`'s hour, and OMNI's.
pub const HOURLY_CADENCE: f64 = 3_600.0;

/// The solar wind over one hour at the bow-shock nose, as OMNI's hourly averages give it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarWind {
    /// Bulk flow speed, km/s.
    pub speed: f64,
    /// Interplanetary magnetic field `By`, GSM, nT.
    pub by: f64,
    /// Interplanetary magnetic field `Bz`, GSM, nT.
    pub bz: f64,
}

impl SolarWind {
    /// Newell et al. (2007)'s coupling `V^{4/3}·B_T^{2/3}·|sin(θ/2)|^{8/3}`, in
    /// (km/s)^{4/3}·nT^{2/3}, with `B_T = √(By² + Bz²)` the field transverse to the Sun–Earth line
    /// and `θ = atan2(By, Bz)` its clock angle.
    ///
    /// The rate at which the solar wind opens magnetic flux at the magnetopause: zero for a field
    /// due north, largest for one due south. Of the three couplings `tools/fit-sdtm-corpus.py`
    /// tried, it tracks `ap60` best hour by hour.
    #[must_use]
    pub fn coupling(&self) -> f64 {
        let transverse = libm::hypot(self.by, self.bz);
        let half_clock = libm::atan2(self.by, self.bz) / 2.0;
        libm::pow(self.speed, 4.0 / 3.0)
            * libm::pow(transverse, 2.0 / 3.0)
            * libm::pow(libm::fabs(libm::sin(half_clock)), 8.0 / 3.0)
    }
}

/// One hour of the solar-wind field's input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StormHour {
    /// GFZ's `ap60` for the hour.
    pub ap60: f64,
    /// The solar wind over the same hour, `None` where it was not measured.
    pub solar_wind: Option<SolarWind>,
}

impl StormHour {
    /// The hour from four values as a caller's arrays carry them, the convention both bridges
    /// take: a NaN speed, `By` or `Bz` marks an hour without the solar wind.
    ///
    /// `None` unless `ap60` is finite and not negative — GFZ writes a missing hour as −1 — and a
    /// solar wind that is there is one: a speed in `[0, 5000)` km/s, and `|By|` and `|Bz|` below
    /// 500 nT. The bounds are far outside anything measured at L1 and below every fill value the
    /// feeds write — OMNI's 9999 km/s and 999.9 nT, ACE's −9999.9 — so a fill value passed for a
    /// missing hour is refused rather than read as the largest storm on record.
    #[must_use]
    pub fn from_values(ap60: f64, speed: f64, by: f64, bz: f64) -> Option<Self> {
        if !(ap60.is_finite() && ap60 >= 0.0) {
            return None;
        }
        if speed.is_nan() || by.is_nan() || bz.is_nan() {
            return Some(Self {
                ap60,
                solar_wind: None,
            });
        }
        let field = |b: f64| b.abs() < 500.0;
        ((0.0..5_000.0).contains(&speed) && field(by) && field(bz)).then_some(Self {
            ap60,
            solar_wind: Some(SolarWind { speed, by, bz }),
        })
    }
}

/// The map from [`SolarWind::coupling`] onto `ap60`'s scale: linear between `M` knots, and the end
/// knots' values outside them.
///
/// The generator puts one knot per `ap60` class, over the hours both series have since 1999: the
/// class's value at the coupling's quantile at the class's mid-probability. So the mapped coupling
/// has `ap60`'s distribution, to the table's interpolation, and differs from `ap60` in which hours
/// carry which value — which is what the solar wind adds. Above the last knot it holds that knot's
/// `ap60`, the largest class an hour with the solar wind has had.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CouplingMap<const M: usize> {
    coupling: [f64; M],
    ap60: [f64; M],
}

impl<const M: usize> CouplingMap<M> {
    /// The map through the knots `(coupling[i], ap60[i])`.
    ///
    /// `None` unless there are two knots or more, every one finite, the couplings strictly
    /// increasing and the `ap60` values not decreasing: the interpolation divides by the
    /// couplings' differences, and a decreasing table would read a stronger wind as a quieter hour.
    #[must_use]
    pub fn new(coupling: [f64; M], ap60: [f64; M]) -> Option<Self> {
        let finite = coupling.iter().chain(&ap60).all(|x| x.is_finite());
        let increasing = coupling.windows(2).all(|w| w[0] < w[1]);
        let ordered = ap60.windows(2).all(|w| w[0] <= w[1]);
        (M >= 2 && finite && increasing && ordered).then_some(Self { coupling, ap60 })
    }

    /// The same without the checks — for the generated table, and nothing else, as
    /// [`super::Thermosphere`]'s `from_fitted` is.
    pub(crate) const fn from_fitted(coupling: [f64; M], ap60: [f64; M]) -> Self {
        Self { coupling, ap60 }
    }

    /// The knots: their couplings and their `ap60` values.
    #[must_use]
    pub const fn knots(&self) -> (&[f64; M], &[f64; M]) {
        (&self.coupling, &self.ap60)
    }

    /// `ap60` on the table for `coupling`, which is NaN for a NaN.
    ///
    /// The generator's `np.interp`, operation for operation: the knot below and its slope to the
    /// next, and the end values outside the knots.
    #[must_use]
    pub fn ap60(&self, coupling: f64) -> f64 {
        if coupling.is_nan() {
            return coupling;
        }
        let above = self.coupling.partition_point(|&knot| knot <= coupling);
        if above == 0 {
            return self.ap60[0];
        }
        if above == M {
            return self.ap60[M - 1];
        }
        let j = above - 1;
        let slope = (self.ap60[j + 1] - self.ap60[j]) / (self.coupling[j + 1] - self.coupling[j]);
        slope * (coupling - self.coupling[j]) + self.ap60[j]
    }
}

/// How the solar-wind field drives its storm state: hour by hour, `(1 − s)·ap60 + s·N` with `N` the
/// mapped coupling, and `ap60` alone in an hour without the solar wind.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarWindDrive<const M: usize> {
    share: f64,
    map: CouplingMap<M>,
}

impl<const M: usize> SolarWindDrive<M> {
    /// The drive with the solar wind's share `s`. `None` unless `share` is in `[0, 1]`: the drive is
    /// a weighted mean of two series on `ap60`'s scale.
    #[must_use]
    pub fn new(share: f64, map: CouplingMap<M>) -> Option<Self> {
        (0.0..=1.0).contains(&share).then_some(Self { share, map })
    }

    /// The same without the checks — for the generated table, and nothing else.
    pub(crate) const fn from_fitted(share: f64, map: CouplingMap<M>) -> Self {
        Self { share, map }
    }

    /// The solar wind's share `s`.
    #[must_use]
    pub const fn share(&self) -> f64 {
        self.share
    }

    /// The coupling's map onto `ap60`'s scale.
    #[must_use]
    pub const fn map(&self) -> &CouplingMap<M> {
        &self.map
    }

    /// The drive of one hour.
    #[must_use]
    pub fn drive(&self, hour: &StormHour) -> f64 {
        match hour.solar_wind {
            Some(wind) => {
                (1.0 - self.share) * hour.ap60 + self.share * self.map.ap60(wind.coupling())
            }
            None => hour.ap60,
        }
    }

    /// The storm driver at the end of `hours`: `response` run at [`HOURLY_CADENCE`] over each
    /// hour's [`Self::drive`], oldest first, with [`StormResponse::response`]'s conventions. The
    /// last hour is the one before the hour the evaluation epoch falls in, because that hour's
    /// `ap60` is not published at the epoch — the causality the field was fitted with.
    ///
    /// `response` is the field's own: [`super::Thermosphere::storm`] of
    /// [`super::solar_wind_coefficients::FIELD`] for the shipped one, which is fitted with this
    /// drive and no other. `hours` is [`super::solar_wind_coefficients::HISTORY`] long: nitric oxide
    /// remembers a storm for days, and a shorter history reads as a different storm.
    #[must_use]
    pub fn storm(&self, response: &StormResponse, hours: &[StormHour]) -> f64 {
        response.run(hours.iter().map(|hour| self.drive(hour)), HOURLY_CADENCE)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::float_cmp)] // several assertions here are exact by construction
    #![allow(clippy::excessive_precision)] // the derivation's 30-digit values, kept deliberately

    use super::*;
    use crate::sdtm::solar_wind_coefficients::{DRIVE, FIELD, HISTORY};

    fn wind(speed: f64, by: f64, bz: f64) -> SolarWind {
        SolarWind { speed, by, bz }
    }

    /// `derivations/sdtm_solar_wind.mac` §7, 30 digits, which `sdtm_solar_wind.gp` reproduces at 60.
    /// Seven `libm` calls of an ulp or so each, two of them raised to `8/3` and `4/3`, is a few parts
    /// in 1e16; a wrong exponent or clock angle is a part in a thousand.
    #[test]
    fn the_coupling_is_the_one_the_derivation_prints() {
        for (w, reference) in [
            (
                wind(400.0, 3.0, -4.0),
                7_488.301_762_986_854_812_308_204_567_69,
            ),
            (
                wind(650.0, -2.0, -12.0),
                29_513.375_985_248_402_334_841_277_081_5,
            ),
            (
                wind(350.0, 4.0, 3.0),
                843.549_792_461_330_495_144_929_383_662,
            ),
            (
                wind(800.0, 0.0, -20.0),
                54_719.230_293_654_303_659_299_483_921_4,
            ),
        ] {
            let got = w.coupling();
            println!("{w:?}: {:.2e} relative", (got / reference - 1.0).abs());
            assert!(
                (got / reference - 1.0).abs() < 2.0e-15,
                "{w:?}: {got} against {reference}"
            );
        }
    }

    /// The half-angle form `sin²(θ/2) = (1 − Bz/B_T)/2` the derivation proves, as a second route
    /// over winds in every quadrant: a branch of `atan2` taken wrongly would be a factor, not an ulp.
    ///
    /// The second route cancels near due north, where `1 − Bz/B_T` is small, so its own error is
    /// about `ε/half`: the difference is weighted by `half` to measure the shipped arithmetic and
    /// not the route's cancellation (2.6e-12 unweighted at 0.01 rad from north).
    #[test]
    fn the_coupling_agrees_with_its_half_angle_form_in_every_quadrant() {
        let mut worst = 0.0_f64;
        for i in 0..72 {
            let clock = f64::from(i) * core::f64::consts::TAU / 72.0 + 0.01;
            for (speed, transverse) in [(300.0, 2.0), (550.0, 9.0), (900.0, 40.0)] {
                let (by, bz) = (transverse * libm::sin(clock), transverse * libm::cos(clock));
                let half = (1.0 - bz / libm::hypot(by, bz)) / 2.0;
                let route = libm::pow(speed, 4.0 / 3.0)
                    * libm::pow(libm::hypot(by, bz), 2.0 / 3.0)
                    * libm::pow(half, 4.0 / 3.0);
                let got = wind(speed, by, bz).coupling();
                worst = worst.max(half * (got - route).abs() / route);
            }
        }
        assert!(worst < 1.0e-14, "worst weighted difference {worst:e}");
    }

    #[test]
    fn the_coupling_vanishes_due_north_and_without_a_field() {
        assert_eq!(wind(500.0, 0.0, 7.0).coupling(), 0.0);
        assert_eq!(wind(500.0, 0.0, 0.0).coupling(), 0.0);
        let east = wind(500.0, 4.0, -3.0).coupling();
        assert_eq!(east, wind(500.0, -4.0, -3.0).coupling(), "even in By");
        assert!(
            east > wind(500.0, 4.0, 3.0).coupling(),
            "south couples more"
        );
    }

    fn map() -> CouplingMap<4> {
        CouplingMap::new([10.0, 200.0, 1_000.0, 5_000.0], [0.0, 3.0, 7.0, 39.0])
            .expect("increasing knots")
    }

    #[test]
    fn the_map_passes_through_its_knots_and_holds_its_ends() {
        let m = map();
        let (coupling, ap60) = m.knots();
        for (x, y) in coupling.iter().zip(ap60) {
            assert_eq!(m.ap60(*x), *y);
        }
        assert_eq!(m.ap60(0.0), 0.0);
        assert_eq!(m.ap60(-5.0), 0.0);
        assert_eq!(m.ap60(1.0e9), 39.0);
        assert_eq!(m.ap60(f64::INFINITY), 39.0);
        assert_eq!(m.ap60(600.0), 5.0, "linear between knots");
        assert!(m.ap60(f64::NAN).is_nan(), "a NaN is not a quiet hour");
    }

    /// No step between neighbouring couplings rises by more than the steepest segment allows, which is
    /// what continuity is on a grid, and none falls.
    #[test]
    fn the_map_is_continuous_and_monotone() {
        let m = map();
        let (coupling, ap60) = m.knots();
        let steepest = (0..3)
            .map(|j| (ap60[j + 1] - ap60[j]) / (coupling[j + 1] - coupling[j]))
            .fold(0.0, f64::max);
        let mut last = m.ap60(0.0);
        for i in 1..=60_000 {
            let x = f64::from(i) * 0.1;
            let y = m.ap60(x);
            assert!(y >= last, "falls at {x}: {last} -> {y}");
            assert!(
                y - last <= steepest * 0.1 * (1.0 + 1.0e-9),
                "jumps at {x}: {last} -> {y}"
            );
            last = y;
        }
    }

    #[test]
    fn the_map_refuses_a_table_that_is_not_one() {
        assert!(
            CouplingMap::new([1.0, 1.0, 2.0], [0.0, 1.0, 2.0]).is_none(),
            "a repeated knot"
        );
        assert!(
            CouplingMap::new([2.0, 1.0, 3.0], [0.0, 1.0, 2.0]).is_none(),
            "unsorted"
        );
        assert!(
            CouplingMap::new([1.0, 2.0, 3.0], [0.0, 2.0, 1.0]).is_none(),
            "decreasing"
        );
        assert!(CouplingMap::new([1.0, f64::NAN, 3.0], [0.0, 1.0, 2.0]).is_none());
        assert!(CouplingMap::new([1.0, 2.0, 3.0], [0.0, 1.0, f64::INFINITY]).is_none());
        assert!(
            CouplingMap::new([1.0], [0.0]).is_none(),
            "one knot is not a line"
        );
        assert!(
            CouplingMap::new([1.0, 2.0, 3.0], [0.0, 0.0, 2.0]).is_some(),
            "a flat run is"
        );
    }

    #[test]
    fn the_drive_is_ap60_at_share_zero_and_without_the_wind() {
        let storm = wind(700.0, 2.0, -15.0);
        let none = SolarWindDrive::new(0.0, map()).expect("a share in [0, 1]");
        let all = SolarWindDrive::new(1.0, map()).expect("a share in [0, 1]");
        let half = SolarWindDrive::new(0.5, map()).expect("a share in [0, 1]");
        let hour = StormHour {
            ap60: 27.0,
            solar_wind: Some(storm),
        };
        assert_eq!(none.drive(&hour), 27.0);
        assert_eq!(all.drive(&hour), map().ap60(storm.coupling()));
        assert_eq!(
            half.drive(&hour),
            0.5 * 27.0 + 0.5 * map().ap60(storm.coupling())
        );
        let gap = StormHour {
            ap60: 27.0,
            solar_wind: None,
        };
        for drive in [none, all, half] {
            assert_eq!(
                drive.drive(&gap),
                27.0,
                "an hour without the wind is its ap60"
            );
        }
    }

    #[test]
    fn an_hour_from_values_takes_nan_as_a_gap_and_refuses_a_fill_value() {
        let gap = StormHour {
            ap60: 12.0,
            solar_wind: None,
        };
        assert_eq!(StormHour::from_values(12.0, f64::NAN, 1.0, -2.0), Some(gap));
        assert_eq!(
            StormHour::from_values(12.0, 450.0, f64::NAN, -2.0),
            Some(gap)
        );
        assert_eq!(
            StormHour::from_values(12.0, 450.0, 1.0, f64::NAN),
            Some(gap)
        );
        assert_eq!(
            StormHour::from_values(12.0, 450.0, 1.0, -2.0),
            Some(StormHour {
                ap60: 12.0,
                solar_wind: Some(wind(450.0, 1.0, -2.0)),
            })
        );
        for (ap60, speed, by, bz) in [
            (-1.0, 450.0, 1.0, -2.0),
            (f64::NAN, 450.0, 1.0, -2.0),
            (f64::INFINITY, 450.0, 1.0, -2.0),
            (12.0, 9_999.0, 1.0, -2.0),
            (12.0, -9_999.9, 1.0, -2.0),
            (12.0, 450.0, 999.9, -2.0),
            (12.0, 450.0, 1.0, -999.9),
            (12.0, 450.0, f64::INFINITY, -2.0),
            (12.0, 5_000.0, 1.0, -2.0),
            (12.0, 450.0, 500.0, -2.0),
            (12.0, 450.0, 1.0, -500.0),
        ] {
            assert_eq!(
                StormHour::from_values(ap60, speed, by, bz),
                None,
                "{ap60} {speed} {by} {bz}"
            );
        }
        for (speed, by, bz) in [
            (0.0, 1.0, -2.0),
            (4_999.9, 1.0, -2.0),
            (450.0, 499.9, -499.9),
        ] {
            assert_eq!(
                StormHour::from_values(12.0, speed, by, bz).and_then(|h| h.solar_wind),
                Some(wind(speed, by, bz)),
                "{speed} {by} {bz} is inside the bounds"
            );
        }
    }

    #[test]
    fn the_drive_refuses_a_share_outside_the_unit_interval() {
        for share in [-0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert!(SolarWindDrive::new(share, map()).is_none(), "{share}");
        }
    }

    /// The hourly path runs the field's own state over the drives, bit for bit what the slice
    /// response gives on the same series: the iterator is the buffer's arithmetic, not an
    /// approximation of it.
    #[test]
    fn the_storm_is_the_response_over_the_hourly_drives() {
        let hour = |i: usize| f64::from(u32::try_from(i).expect("an hour count fits in u32"));
        let peak = hour(HISTORY) - 60.0;
        let hours: Vec<StormHour> = (0..HISTORY)
            .map(|i| {
                let t = hour(i);
                let storm = libm::exp(-0.5 * ((t - peak) / 9.0).powi(2));
                StormHour {
                    ap60: (3.0 + 250.0 * storm).round(),
                    solar_wind: (i % 37 != 0).then_some(wind(
                        380.0 + 400.0 * storm,
                        5.0 * libm::sin(t / 7.0),
                        -3.0 - 25.0 * storm,
                    )),
                }
            })
            .collect();
        let drives: Vec<f64> = hours.iter().map(|h| DRIVE.drive(h)).collect();
        assert_eq!(
            DRIVE.storm(FIELD.storm(), &hours),
            FIELD.storm().response(&drives, HOURLY_CADENCE)
        );
        assert_eq!(DRIVE.storm(FIELD.storm(), &[]), 0.0);
    }

    /// A steady hour holds the state at its steady value: the cooling's `D = H/(1 + H/H_s)` at the
    /// drive, whatever the share, which is the start every window takes.
    #[test]
    fn a_steady_hour_holds_the_storm_at_its_steady_state() {
        let StormResponse::Cooling(cooling) = FIELD.storm() else {
            panic!("the shipped solar-wind field is the cooling state");
        };
        let hour = StormHour {
            ap60: 32.0,
            solar_wind: Some(wind(520.0, -3.0, -6.0)),
        };
        let steady = cooling.steady_state(DRIVE.drive(&hour));
        let got = DRIVE.storm(FIELD.storm(), &vec![hour; HISTORY]);
        assert!(
            (got / steady - 1.0).abs() < 1.0e-12,
            "{got} against {steady}"
        );
    }
}
