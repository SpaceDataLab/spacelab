//! The ITRF→GCRF chain against NEPTUNE's, on one epoch and one state.
//!
//! `orekit_oracle.rs` measures the chain against Orekit; this measures it against a fourth lineage
//! (Space-Systems/neptune, TU Braunschweig ILR/IRAS, Apache-2.0), whose reduction lives in libslam
//! (`src/astro/slam_reduction.f90`) and, like ours, runs the IAU 2006/2000A CIO route with the CIP
//! coordinates taken from an interpolation table.
//!
//! The comparison is arranged so that only the algorithm differs:
//!
//!   * The state is one Orbit State Vector lifted verbatim from ESA's Sentinel-1B precise orbit
//!     product, whose header declares `EARTH_FIXED` and `UTC`. It is what NEPTUNE was fed, as the
//!     `input type 4` state of `tools/neptune-oracle/decks/sentinel1b-matched/neptune.inp`.
//!   * The Earth orientation parameters are **NEPTUNE's own**, read out of the row for MJD 58659 of
//!     the `eop19620101.txt` it ships (IERS 20 C04). Feeding it ours would measure the two EOP
//!     series against each other and call the result a frame difference.
//!
//! What remains uncontrolled, and is therefore what the residual is made of: NEPTUNE interpolates
//! its EOP table to the epoch where this hard-codes the 0h row (2 s away, so sub-micro-arcsecond),
//! and it reads the CIP from `txys_interpolation.dat` where `spacelab_fa` evaluates the series.
//! Neither side applies a Helmert step — libslam has none — so the deck's "ITRF-2000" label names
//! the input product's realization and not a datum transformation either code performs.
//!
//! Measured: 1.3e-5 m and 1.6e-7 m/s, which is about 0.4 µas of pole at this radius.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_core::calendar::UtcDateTime;
use spacelab_core::time::{Epoch, Tt, Ut1};
use spacelab_core::{Angle, Vec3};
use spacelab_frames::{Itrf, StateVector, gcrf_to_itrf};

/// TT − TAI, seconds, exact by definition.
const TT_MINUS_TAI: f64 = 32.184;

/// `S1B_OPER_AUX_POEORB_OPOD_20210330T193037_V20190624T225942_20190626T005942.EOF`, the OSV at
/// `UTC=2019-06-25T00:00:02.000000`, metres and metres per second, Earth-fixed.
const R_ITRF: Vec3 = Vec3::new(-999_068.229_447, -776_132.523_648, 6_951_328.124_086);
const V_ITRF: Vec3 = Vec3::new(-183.002_918, 7_536.746_464, 813.463_694);

/// The GCRF state NEPTUNE derived from that ITRF state, as `init_neptune` logs it, converted from
/// the kilometres it prints.
/// Written at the shortest decimal that round-trips to the same `f64` as the printed value: the
/// digits beyond it do not name a different number, and clippy reads them as precision that is not
/// there.
const R_GCRF_NEPTUNE: Vec3 = Vec3::new(
    -807_817.645_389_187_1,
    962_600.149_554_167_6,
    6_952_863.677_215_289,
);
const V_GCRF_NEPTUNE: Vec3 = Vec3::new(
    7_451.921_920_356_678,
    465.739_557_476_013_43,
    799.590_602_065_284_8,
);

/// Row `2019 06 25 58659` of NEPTUNE's own `work/data/eop19620101.txt`, arcseconds and seconds.
const XP_ARCSEC: f64 = 0.150_506;
const YP_ARCSEC: f64 = 0.424_491;
const DUT1_SECONDS: f64 = -0.174_412_4;
const LOD_SECONDS: f64 = 0.000_071_8;
const DX_ARCSEC: f64 = 0.000_024;
const DY_ARCSEC: f64 = -0.000_162;
/// TAI − UTC at that epoch, seconds — the last column of the same row.
const DAT_SECONDS: f64 = 37.0;

/// Our GCRF state for the epoch and EOP NEPTUNE was given.
#[allow(clippy::cast_possible_truncation)] // |offset| < 70, so offset*1e9 < 7e10 << i128::MAX
fn our_gcrf() -> StateVector<spacelab_frames::Gcrf> {
    let utc = UtcDateTime::new(2019, 6, 25, 0, 0, 2.0).expect("valid utc");
    let ns = |offset: f64| utc.naive_ns_since_j2000() + (offset * 1e9).round() as i128;
    let transform = gcrf_to_itrf(
        Epoch::<Ut1>::from_ns_since_j2000(ns(DUT1_SECONDS)),
        Epoch::<Tt>::from_ns_since_j2000(ns(DAT_SECONDS + TT_MINUS_TAI)),
        Angle::from_arcseconds(XP_ARCSEC),
        Angle::from_arcseconds(YP_ARCSEC),
        Angle::from_arcseconds(DX_ARCSEC),
        Angle::from_arcseconds(DY_ARCSEC),
        LOD_SECONDS,
    );
    transform
        .inverse()
        .state(StateVector::<Itrf>::new(R_ITRF, V_ITRF))
}

/// The same terrestrial state lifted into the celestial frame by two independent implementations.
///
/// The bound is two orders above the measurement, which is enough room for the last digits of a
/// series evaluated in a different order and no room at all for anything structural: a dropped CIO
/// locator, polar motion with the wrong sign, ERA against GMST, or the missing `ω × r` term in the
/// velocity each land four or more orders above it.
#[test]
fn the_gcrf_state_matches_neptunes_own_reduction() {
    let ours = our_gcrf();
    let position = (ours.position.vector - R_GCRF_NEPTUNE).norm();
    let velocity = (ours.velocity.vector - V_GCRF_NEPTUNE).norm();
    println!("ITRF→GCRF vs NEPTUNE: |Δr| = {position:.3e} m, |Δv| = {velocity:.3e} m/s");
    assert!(
        position < 1e-3,
        "position differs by {position:.3e} m — beyond the CIP-interpolation difference the module \
         documents"
    );
    assert!(
        velocity < 1e-5,
        "velocity differs by {velocity:.3e} m/s — beyond what the position difference and the LOD \
         term can account for"
    );
}
