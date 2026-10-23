//! The other reality check: model a published laser range and report what is left over.
//!
//! An SLR normal point is the most accurate range measurement made of an artificial satellite —
//! a few millimetres of formal precision on a passive sphere of solid brass and glass, published by
//! stations that have no interest in agreeing with this workspace. Nothing here is fitted. The
//! satellite comes from the ILRS combined orbit, the station from ITRF2020, the Earth orientation
//! from the IERS, and the residual is the difference between a measured flight time and a modelled
//! one — in millimetres of real range, not in units of agreement with ourselves.
//!
//! Every term below is needed to reach centimetres, and each is worth a different amount:
//!
//! | term | size on a LAGEOS pass | what it is |
//! |---|---|---|
//! | station motion during the flight | 37 m | the up and down legs are solved separately |
//! | satellite motion during the flight | 300 km | the reflection is at `t + τ_up`, not at `t` |
//! | centre-of-mass offset | 251 mm | the array reflects in front of the centre of mass |
//! | tropospheric refraction | 2–8 m | Mendes–Pavlis, one way, at the laser wavelength |
//! | solid Earth tide | up to 0.4 m | IERS §7.1.1, the station is not where its catalogue says |
//! | pole tide | up to 25 mm | IERS §7.1.4, about the 2018 secular pole |
//! | Shapiro delay | 7 mm one way | IERS §11 |
//! | station eccentricity | up to 3.18 m | monument to telescope, at three of the twelve pads |
//! | published range bias | up to 12 mm | ILRS Data Handling File, at four of the twelve pads |
//! | ocean tide loading | up to ~30 mm | modelled in `slr::loading`, **not applied here** — see below |
//!
//! **Measured on 2025-01-06**, 211 normal points from twelve stations on both spheres:
//! **bias +0.59 cm, RMS 1.63 cm**, worst single residual 7.02 cm, every station under 4 cm. The
//! per-station table is printed by the test.
//!
//! Two things in that table are findings rather than noise. The first is what the Data Handling
//! File does and does not say. It publishes a range bias for four of these twelve pads — BADL,
//! HA4T, HERL and WETL, none larger than 12 mm — and those are applied here; three of the four
//! moved towards zero (HA4T +1.39 → +0.36 cm, HERL −0.17 → −0.04, WETL +0.26 → +0.14) and BADL
//! crossed it (+0.54 → −0.65), which is inside the day-to-day scatter of 24 points. For the other
//! seven pads it carries a record with **no value**: model code `E`, "estimate a range bias here".
//! Among those seven are the three worst stations — KUN2 3.62, CHAL 3.06 and BEIL −3.18 cm, all
//! with |bias| ≈ RMS, the signature of an offset rather than noise — so the network agrees they are
//! biased and does not say by how much. Nothing an analysis that fits nothing can apply. But four
//! of the seven sit under a centimetre, so `E` marks where a bias is to be solved for, not which
//! stations are bad.
//!
//! The second is that YARL, HA4T and THTL are the three pads whose ITRF2020 entry is a survey
//! monument rather than the telescope: before the eccentricity was applied they came out at 1.95,
//! 1.21 and 3.08 m. Solving the 69 YARL residuals for a station offset returned up +3.173 m,
//! north −0.009 m, east +0.018 m against the site log's published +3.1827, −0.0064, +0.0194 — the
//! model recovers a surveyed quantity it was not given.
//!
//! **What is not applied here.** Ocean tide loading is the leading omitted term — millimetres
//! inland, a few centimetres at a coastal site — and it is no longer unmodelled: `slr::loading`
//! carries the full `HARDISP` admittance expansion. What is missing is the input, a per-station
//! `BLQ` table for these twelve pads, which the loading providers issue per request and this
//! workspace does not ship. Then the seven `E`-flagged range biases, which cannot be applied
//! without estimating them; atmospheric pressure loading (~5 mm); and the station-specific
//! centre-of-mass correction (the ILRS single value of 251 mm stands in for values that range over
//! about 6 mm between detector types). Together they set the floor this test can reach, and it is
//! why the bound is centimetres rather than the millimetres the measurement itself is capable of.
//!
//! # The same model over a week
//!
//! One day says whether the model is right. It cannot say whether a station's bias repeats, whether
//! anything moves when the orbit product changes, or whether what is left has a frequency. So the
//! model runs unchanged over **2025-01-05 .. 2025-01-12** — 2158 published normal points, 26 pads,
//! both spheres, and the two weekly ILRS combined arcs.
//!
//! **The arcs do not overlap.** One ends at 23:58 and the next begins two minutes later, so there is
//! no common span to difference; what a change of orbit product shows up in is the step across the
//! seam, measured against the step the same station takes between two consecutive days *inside* one
//! arc. Across the seam, 7 stations, RMS **0.95 cm**. Within one arc, 44 consecutive-day steps, RMS
//! **0.88 cm**. Changing the arc moves a station by what an ordinary day moves it: the weekly
//! products are consistent across their boundary.
//!
//! **108 of the 2158 points are declined, by reason, and the total is asserted.** A single day of the
//! ILRS network is uniform and a week is not: IRKL and MDVS publish nine sessions with the system
//! delay still in the range, and TKBL 7306 and YEBL 7817 have no ITRF2020 solution at all — the
//! orbit is SLRF2020, which carries stations that joined after ITRF2020's data cut-off.
//!
//! **The scatter is a function of the publisher's own return count**, which is not a property of the
//! residual: under 30 raw ranges the residuals scatter at 3.99 cm and above it at 1.4 to 1.9 cm,
//! while the bias is the same in every bin. All 2050 modelled points give bias +0.37 cm, RMS
//! 2.47 cm; the 1516 formed from thirty returns or more give **bias +0.44 cm, RMS 1.63 cm, worst
//! 7.02 cm** — the same 1.63 cm the single day reached, over eight days and twenty stations. The
//! three worst residuals in the campaign, near −50 cm, are one POT3 pass whose points were formed
//! from 23, 29 and 26 returns; on that count alone POT3 goes from −6.08 cm bias / 17.02 cm RMS to
//! −0.22 / 0.74.
//!
//! **By frequency, nothing can be attributed.** Seven stations carry a semi-diurnal amplitude above
//! three formal sigma, 0.53 to 3.32 cm — the ocean-loading scale. It is not evidence of ocean
//! loading: BADL is a thousand kilometres from any coast and shows 0.63 cm, and a station observes
//! the same satellite at the same points of its own sidereal cycle, so any elevation-dependent error
//! lands in the same band. Two limits are structural and reported rather than worked around: over
//! eight days the Rayleigh limit is 0.125 cycles per day, so M2/S2 (0.068 apart) and K1/O1 (0.073)
//! do not separate, and a station that only ranges at one local hour cannot constrain a diurnal term
//! at all. Separating them needs the per-station `BLQ` table named above.
//!
//! Sources, all primary and all fetched rather than transcribed — `tools/fetch-slr-data.sh`:
//! `lageos{1,2}_202501.np2` (ILRS CRD v2 normal points, EDC/DGFI-TUM),
//! `ilrsa.orb.lageos{1,2}.*.v80.sp3` (the ILRS combined orbit, SLRF2020),
//! `ITRF2020-ILRS-TRF.SNX` (IGN), the ILRS site logs (eccentricities),
//! `ILRS_Data_Handling_File_*.snx` (ILRS Analysis Standing Committee),
//! `finals2000A.all` (IERS Rapid Service).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    // `m` is the frame rotation, `d` a displacement, `r` a range: the names the equations use.
    clippy::many_single_char_names
)]

use std::collections::BTreeMap;

use spacelab::{EopSource, earth_orientation_at};
use spacelab_core::{Angle, Epoch, Tt, Ut1, UtcDateTime, Vec3};
use spacelab_earth::itrf_to_geodetic;
use spacelab_eop::{EopSeries, subdaily_correction};
use spacelab_estimation::Srif;
use spacelab_frames::{Itrf, Position, gcrf_to_itrf};
use spacelab_gnss::{Itrs, ReferenceFrame, SatId, Sp3, TimeSystem};
use spacelab_propagation::ephemeris::{AnalyticMoon, AnalyticSun, Ephemeris};
use spacelab_slr::crd::{EpochEvent, Session, parse_crd};
use spacelab_slr::dhf::{DataHandling, RangeBias, Target};
use spacelab_slr::range::SPEED_OF_LIGHT;
use spacelab_slr::sinex::{EccentricityAxis, StationCatalogue, parse_sinex};
use spacelab_slr::troposphere::refraction_delay;
use spacelab_slr::{
    pole_tide_displacement, shapiro_delay, solid_earth_tide_displacement, two_way_light_time,
};

/// LAGEOS-1 normal points for 2025-01-06, extracted from the ILRS January 2025 monthly file.
const CRD_LAGEOS1: &str = include_str!("fixtures/slr_lageos1_20250106.npt");
/// LAGEOS-2 for the same day, same file.
const CRD_LAGEOS2: &str = include_str!("fixtures/slr_lageos2_20250106.npt");
/// The ILRS combined orbit for LAGEOS-1, 2-minute grid, trimmed to the day plus interpolation margin.
const SP3_LAGEOS1: &str = include_str!("fixtures/ilrsa_lageos1_20250106.sp3");
/// The same for LAGEOS-2.
const SP3_LAGEOS2: &str = include_str!("fixtures/ilrsa_lageos2_20250106.sp3");
/// The twelve observing pads' `SITE/ID`, `SOLUTION/EPOCHS` and `SOLUTION/ESTIMATE` records out of
/// `ITRF2020-ILRS-TRF.SNX`; the covariance blocks that make up the rest of that file are dropped.
const SINEX: &str = include_str!("fixtures/itrf2020_ilrs_slr_stations.snx");
/// The frame those coordinates are in, declared here because the file does not say.
///
/// Checked on the full `ITRF2020-ILRS-TRF.SNX` and not merely on this excerpt: it carries one
/// `+FILE/COMMENT` block naming the software that wrote it, the string `ITRF` appears nowhere else,
/// and the `%=SNX` header line has no frame field. The realization of an ITRF SINEX lives in its
/// *filename*, so declaring it beside the path that chose the file is as close to checked as the
/// format allows -- and it is what the orbit product's own stated frame is then asserted against.
const STATION_FRAME: ReferenceFrame = ReferenceFrame::Itrs(Itrs::Itrf2020);
/// MJD 60673-60683 of the IERS `finals2000A` series — the same excerpt the IGS orbit fit reads.
const EOP: &str = include_str!("fixtures/finals2000A_2025001.txt");
/// Monument-to-telescope offsets for the three pads that have one, from the ILRS site logs. Without
/// them those three stations are out by 2.6 to 3.2 metres and the other nine are unaffected.
const ECCENTRICITY: &str = include_str!("fixtures/ilrs_eccentricity.snx");
/// The ILRS Data Handling File, cut to these twelve pads: the range biases the network has measured
/// and the stations it says to solve one for.
const DATA_HANDLING: &str = include_str!("fixtures/ilrs_data_handling.snx");

/// The same products over 2025-01-05..12, for the arc campaign. Eight days, twenty-six pads and two
/// weekly orbit arcs, which is what it takes to see a station's bias repeat, to cross an arc
/// boundary, and to ask whether anything is left at a tidal frequency.
const WEEK_CRD_LAGEOS1: &str = include_str!("fixtures/slr_lageos1_20250105_20250112.npt");
/// LAGEOS-2 over the same span.
const WEEK_CRD_LAGEOS2: &str = include_str!("fixtures/slr_lageos2_20250105_20250112.npt");
/// The ILRS combined orbit for the week ending 2025-01-11, trimmed to the observed passes.
const WEEK_SP3_LAGEOS1_A: &str = include_str!("fixtures/ilrsa_lageos1_arc250111.sp3");
/// The next arc, which the eighth day is scored against.
const WEEK_SP3_LAGEOS1_B: &str = include_str!("fixtures/ilrsa_lageos1_arc250118.sp3");
/// The same two arcs for LAGEOS-2.
const WEEK_SP3_LAGEOS2_A: &str = include_str!("fixtures/ilrsa_lageos2_arc250111.sp3");
/// The second LAGEOS-2 arc.
const WEEK_SP3_LAGEOS2_B: &str = include_str!("fixtures/ilrsa_lageos2_arc250118.sp3");
/// ITRF2020 for the twenty-six pads that tracked over the span.
const WEEK_SINEX: &str = include_str!("fixtures/itrf2020_ilrs_slr_stations_20250105_20250112.snx");
/// MJD 60672-60695 of `finals2000A`: the span plus eight days at each end.
const WEEK_EOP: &str = include_str!("fixtures/finals2000A_20250105_20250112.txt");
/// Eccentricities for those pads that have one.
const WEEK_ECCENTRICITY: &str = include_str!("fixtures/ilrs_eccentricity_20250105_20250112.snx");
/// The Data Handling File, cut to the same pads.
const WEEK_DATA_HANDLING: &str = include_str!("fixtures/ilrs_data_handling_20250105_20250112.snx");

/// SP3 identifiers of the two spheres.
const LAGEOS1: SatId = SatId {
    system: 'L',
    prn: 51,
};
/// LAGEOS-2.
const LAGEOS2: SatId = SatId {
    system: 'L',
    prn: 52,
};

/// The ILRS single-value centre-of-mass correction for both LAGEOS satellites, metres.
///
/// The retroreflectors sit on the sphere's surface and the effective reflection point is this far in
/// front of the centre of mass, so the measured range is *shorter* than the range to the centre the
/// orbit product quotes. Station-dependent values derived from the array's response range over about
/// 6 mm; the network's conventional single value is used, and that spread is part of the floor.
const CENTRE_OF_MASS: f64 = 0.251;

/// Earth's gravitational parameter, m³/s² — for the Shapiro delay only, which needs two digits.
const EARTH_GM: f64 = 3.986_004_418e14;

/// Neville degree for the 2-minute orbit grid. Eleven samples span 20 minutes, 8.9% of a LAGEOS
/// revolution; the interpolation error there is far below the product's own centimetres.
const INTERPOLATION_ORDER: usize = 10;

/// Light-time convergence, seconds. Both legs contract by `v/c ≈ 2e-5` per iteration, so this is
/// reached in four and the cap only bounds a pathological ephemeris.
const LIGHT_TIME_TOLERANCE: f64 = 1e-14;
/// Iteration cap for the same.
const LIGHT_TIME_ITERATIONS: usize = 20;

/// Below this the Mendes–Pavlis mapping function is outside the elevation band it was fitted over,
/// and no ILRS station tracks there anyway.
const MINIMUM_ELEVATION_DEGREES: f64 = 10.0;

/// One station, resolved once per pass: where it is and what the refraction model needs to know.
struct Station {
    /// Earth-fixed position from the catalogue with the solution's velocity applied, metres. Tides
    /// are added per epoch, not here.
    itrf: [f64; 3],
    /// Geodetic latitude, radians.
    latitude: f64,
    /// Ellipsoidal height, metres.
    height: f64,
    /// Local vertical in Earth-fixed axes, for the elevation angle.
    up: [f64; 3],
    /// Geodetic longitude, radians — only for reporting a residual against local solar time.
    longitude: f64,
}

/// One weekly ILRS orbit product: the parsed file and the span its own epoch grid covers, in the
/// seconds-since-J2000 scale the normal-point epochs are built in.
///
/// The weekly products do **not** overlap — one ends at 23:58 and the next begins two minutes
/// later — so which arc a measurement belongs to is decided by containment, and a measurement
/// inside none of them is a fixture that does not reach the day rather than a bad epoch.
struct OrbitArc {
    sp3: Sp3,
    start: f64,
    end: f64,
}

impl OrbitArc {
    fn parse(text: &str) -> Self {
        let sp3 = Sp3::parse(text);
        assert_eq!(
            sp3.time_system(),
            TimeSystem::Utc,
            "an ILRS orbit product is stamped in UTC"
        );
        // The frame the products claim, checked rather than asserted in prose. `spacelab_gnss` now
        // surfaces line 1's coordinate-system field, and an SP3 in another datum -- an IAC GLONASS
        // product is PZ-90, decimetres from ITRF -- would otherwise be differenced against ITRF
        // coordinates with nothing to show for it.
        // The orbit is SLRF2020 and the stations are `STATION_FRAME`. Those realize the same IERS
        // solution, which is what makes differencing them legitimate; SLRF2020 is a superset
        // carrying pads that joined after ITRF2020's cut-off, and those are refused separately
        // below by having no station coordinates at all.
        assert!(
            sp3.reference_frame().agrees_with(STATION_FRAME),
            "the orbit's {:?} does not realize the stations' {STATION_FRAME:?}",
            sp3.reference_frame()
        );
        let start = sp3.start_epoch().expect("the file carries epochs");
        let end = start
            + sp3
                .epochs()
                .last()
                .copied()
                .expect("the file carries epochs");
        Self { sp3, start, end }
    }

    fn covers(&self, t: f64) -> bool {
        t >= self.start && t <= self.end
    }
}

/// One modelled measurement.
struct Residual {
    /// `NAME pad`, the label the table is grouped by.
    label: String,
    /// Elevation of the reflection point seen from the station, degrees.
    elevation: f64,
    /// Observed minus modelled one-way range, metres, with any published range bias already
    /// subtracted from the observation.
    value: f64,
    /// What the Data Handling File says about this station, target and wavelength.
    handling: Option<RangeBias>,
    /// Transmit epoch, seconds since J2000 on the naive UTC scale — the day the point belongs to
    /// and its phase within the day, which is what a campaign bins by.
    epoch: f64,
    /// The station's geodetic longitude, radians, so a residual can be placed in local solar time.
    longitude: f64,
    /// Raw ranges the publisher formed this point from — its own statement of how much measurement
    /// is behind the number, and the only quality indicator here that does not come from the
    /// residual itself.
    returns: u32,
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

/// `GCRF = Mᵀ · ITRF`, where `m` is the rows of the GCRF→ITRF rotation.
fn to_gcrf(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

/// `ITRF = M · GCRF`.
fn to_itrf(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// The IERS 2018 secular pole, radians — Conventions (2010) chapter 7 as amended 2018/02/01,
/// Eq. (21): `xs = 55.0 + 1.677 (t − 2000)`, `ys = 320.5 + 3.460 (t − 2000)` milliarcseconds.
///
/// The pole tide responds to the wander *about* this line. Feeding it the pole itself instead is a
/// metre-scale error at mid-latitudes, not a refinement.
fn secular_pole(decimal_year: f64) -> (f64, f64) {
    const MAS: f64 = std::f64::consts::PI / (180.0 * 3600.0 * 1000.0);
    let dt = decimal_year - 2000.0;
    ((55.0 + 1.677 * dt) * MAS, (320.5 + 3.460 * dt) * MAS)
}

/// Resolve a pad: the frame's position for the survey monument, plus the eccentricity that takes it
/// to the telescope's invariant point. `None` if the frame does not carry the pad at all.
///
/// The eccentricity is the difference between a centimetre-level residual and a three-metre one at
/// the three pads that have one, and it is applied in the *local* frame of the monument — the
/// rotation is built from the monument's own geodetic coordinates, which a three-metre radial shift
/// changes by a hundredth of an arcsecond.
fn station_at(
    catalogue: &StationCatalogue,
    eccentricities: &StationCatalogue,
    pad_id: u32,
    decimal_year: f64,
) -> Option<Station> {
    let code = format!("{pad_id:04}");
    let monument = catalogue.position_at(&code, decimal_year)?;
    let g = itrf_to_geodetic(Position::<Itrf>::new(Vec3::new(
        monument[0],
        monument[1],
        monument[2],
    )))
    .expect("an SLR pad is on the Earth's surface");
    let (phi, lambda) = (g.latitude.radians(), g.longitude.radians());
    let up = [
        phi.cos() * lambda.cos(),
        phi.cos() * lambda.sin(),
        phi.sin(),
    ];
    let north = [
        -phi.sin() * lambda.cos(),
        -phi.sin() * lambda.sin(),
        phi.cos(),
    ];
    let east = [-lambda.sin(), lambda.cos(), 0.0];

    let mut itrf = monument;
    if let Some(e) = eccentricities.eccentricity_at(&code, decimal_year) {
        let d = match e.axis {
            EccentricityAxis::Xyz => e.offset,
            EccentricityAxis::Une => [
                e.offset[0] * up[0] + e.offset[1] * north[0] + e.offset[2] * east[0],
                e.offset[0] * up[1] + e.offset[1] * north[1] + e.offset[2] * east[1],
                e.offset[0] * up[2] + e.offset[1] * north[2] + e.offset[2] * east[2],
            ],
        };
        for i in 0..3 {
            itrf[i] += d[i];
        }
    }

    Some(Station {
        itrf,
        latitude: phi,
        height: g.height.metres(),
        up,
        longitude: lambda,
    })
}

/// Earth orientation at one instant, with the sub-daily corrections already folded in.
///
/// They are separated from [`earth_orientation_at`]'s daily values because the daily series is
/// smoothed over the diurnal and semi-diurnal tidal terms, which at an SLR station are tens of
/// microarcseconds of polar motion and tens of microseconds of UT1 — millimetres of range, right at
/// the level this test resolves.
struct Orientation {
    ut1_ns: i128,
    tt_ns: i128,
    xp: Angle,
    yp: Angle,
    lod: f64,
}

/// `None` for an epoch outside the EOP series, which is a missing input rather than a bad one.
fn orientation_at(utc: UtcDateTime, series: &EopSeries) -> Option<Orientation> {
    let e = earth_orientation_at(utc, EopSource::Series(series)).ok()?;
    let sd = subdaily_correction(utc.modified_julian_date());
    Some(Orientation {
        ut1_ns: e.orientation.ut1.ns_since_j2000() + (sd.ddut1_seconds * 1e9).round() as i128,
        tt_ns: e.orientation.tt.ns_since_j2000(),
        xp: Angle::from_radians(e.orientation.xp.radians() + sd.dxp.radians()),
        yp: Angle::from_radians(e.orientation.yp.radians() + sd.dyp.radians()),
        lod: e.orientation.lod_seconds + sd.dlod_seconds,
    })
}

/// Where the station actually is at this instant: the catalogue entry plus the two deformations the
/// Conventions make mandatory at this accuracy.
///
/// `m0` rotates GCRF into ITRF at the same instant — it is passed rather than recomputed because
/// the caller already has it, and the Sun and Moon have to be in the station's frame.
fn displaced_station(
    itrf: [f64; 3],
    o: &Orientation,
    m0: [[f64; 3]; 3],
    ut1_hours: f64,
) -> [f64; 3] {
    let tt_seconds = o.tt_ns as f64 / 1e9;
    let sun = to_itrf(
        m0,
        AnalyticSun {
            epoch_j2000_seconds: tt_seconds,
        }
        .position(0.0),
    );
    let moon = to_itrf(
        m0,
        AnalyticMoon {
            epoch_j2000_seconds: tt_seconds,
        }
        .position(0.0),
    );
    let (sxp, syp) = secular_pole(2000.0 + tt_seconds / (365.25 * 86_400.0));
    let solid = solid_earth_tide_displacement(
        itrf,
        sun,
        moon,
        ut1_hours,
        Epoch::<Tt>::from_ns_since_j2000(o.tt_ns).julian_centuries(),
    );
    let pole = pole_tide_displacement(itrf, o.xp.radians(), o.yp.radians(), sxp, syp);
    let mut here = itrf;
    for i in 0..3 {
        here[i] += solid[i] + pole[i];
    }
    here
}

/// Model every normal point of one pass and append its residuals.
///
/// `None` is returned for a pass this model declines to use rather than silently mismodel: a
/// one-way transponder session, a session the station flagged, a session whose time tags are not
/// transmit epochs, or one whose publisher already applied a correction that is applied here.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)] // one measurement chain, split only where a term is reusable
fn model_pass(
    session: &Session,
    station: &Station,
    arcs: &[OrbitArc],
    sat: SatId,
    series: &EopSeries,
    handling: Option<RangeBias>,
    out: &mut Vec<Residual>,
    declined: &mut Declined,
) {
    let day = UtcDateTime::from_date(session.year, session.month as u8, session.day as u8)
        .expect("the session header carries a calendar date");
    let day_seconds = day.naive_ns_since_j2000() as f64 / 1e9;
    let label = format!("{} {}", session.station, session.pad_id);
    let wavelength_um = session.wavelength_nm / 1000.0;
    // "estimated values (when provided) to be subtracted from the 1-way observations" -- the Data
    // Handling File's own sentence, and the only place the sign of these numbers is written down.
    // A station the file only marks for estimation publishes no value, so nothing is subtracted.
    let bias = match handling {
        Some(RangeBias::Apply(millimetres)) => millimetres / 1000.0,
        Some(RangeBias::Estimate) | None => 0.0,
    };

    for np in &session.normal_points {
        let Some(met) = session.meteorology_at(np.seconds_of_day) else {
            declined.record(&session.station, "no meteorology at this epoch", 1);
            continue;
        };
        let transmit = day_seconds + np.seconds_of_day;
        let utc = UtcDateTime::from_naive_ns_since_j2000((transmit * 1e9).round() as i128, false);
        let Some(o) = orientation_at(utc, series) else {
            declined.record(&session.station, "outside the Earth-orientation series", 1);
            continue;
        };
        let zero = Angle::from_radians(0.0);

        // The rotation at `dt` seconds past the transmit epoch. It is recomputed rather than
        // interpolated: the whole round trip is 0.08 s, and the terms that matter over it — the
        // Earth rotation angle above all — are exactly the ones an interpolator would have to be
        // trusted on.
        let rotation = |dt: f64| {
            let dn = (dt * 1e9).round() as i128;
            gcrf_to_itrf(
                Epoch::<Ut1>::from_ns_since_j2000(o.ut1_ns + dn),
                Epoch::<Tt>::from_ns_since_j2000(o.tt_ns + dn),
                o.xp,
                o.yp,
                zero,
                zero,
                o.lod,
            )
            .rotation
            .matrix()
            .rows
        };

        let here = displaced_station(station.itrf, &o, rotation(0.0), np.seconds_of_day / 3600.0);
        // Which weekly product this measurement is scored against. The arcs abut rather than
        // overlap, so a point belongs to exactly one; a point inside none of them means the
        // fixtures do not cover the day, which is a broken excerpt, not a skippable measurement.
        let arc = arcs
            .iter()
            .find(|a| a.covers(transmit))
            .expect("every normal point falls inside one of the orbit arcs");
        let sp3 = &arc.sp3;
        let sp3_t = transmit - arc.start;
        let ground = |dt: f64| to_gcrf(rotation(dt), here);
        let space = |dt: f64| {
            sp3.interpolate(sat, sp3_t + dt, INTERPOLATION_ORDER)
                .map(|p| to_gcrf(rotation(dt), p))
        };
        let Some(r) = two_way_light_time(
            0.0,
            ground,
            space,
            LIGHT_TIME_TOLERANCE,
            LIGHT_TIME_ITERATIONS,
        ) else {
            declined.record(&session.station, "the light time did not converge", 1);
            continue;
        };

        // Elevation of the reflection point, taken Earth-fixed at the epoch of each end: that is
        // the geometry the beam actually traversed, and it differs from the transmit-epoch geometry
        // by 0.018° — a centimetre of two-way refraction at low elevation.
        let Some(bounce) = sp3.interpolate(sat, sp3_t + r.up_leg, INTERPOLATION_ORDER) else {
            declined.record(
                &session.station,
                "the orbit does not reach the reflection",
                1,
            );
            continue;
        };
        let los = [
            bounce[0] - here[0],
            bounce[1] - here[1],
            bounce[2] - here[2],
        ];
        let elevation = (dot(station.up, los) / norm(los)).asin();
        if elevation.to_degrees() < MINIMUM_ELEVATION_DEGREES {
            declined.record(&session.station, "below the elevation cut", 1);
            continue;
        }

        let refraction = refraction_delay(
            station.latitude,
            station.height,
            met.pressure,
            met.temperature,
            met.relative_humidity,
            wavelength_um,
            elevation,
        );

        // Shapiro on each leg separately — the two endpoints differ between them — averaged down to
        // the one-way range the normal point is expressed as.
        let transmitter = ground(0.0);
        let reflector = to_gcrf(rotation(r.up_leg), bounce);
        let receiver = ground(r.time_of_flight());
        let shapiro = f64::midpoint(
            shapiro_delay(
                norm(transmitter),
                norm(reflector),
                SPEED_OF_LIGHT * r.up_leg,
                EARTH_GM,
            ),
            shapiro_delay(
                norm(reflector),
                norm(receiver),
                SPEED_OF_LIGHT * r.down_leg,
                EARTH_GM,
            ),
        );

        let observed = 0.5 * SPEED_OF_LIGHT * np.time_of_flight - bias;
        let modelled = r.range - CENTRE_OF_MASS + refraction + shapiro;
        out.push(Residual {
            label: label.clone(),
            elevation: elevation.to_degrees(),
            value: observed - modelled,
            handling,
            epoch: transmit,
            longitude: station.longitude,
            returns: np.raw_ranges,
        });
    }
}

/// The sessions the model declined, and what each refusal cost.
///
/// Counted rather than asserted away. A single day of the ILRS network happens to be uniform, but
/// over a week it is not: some publishers submit ranges with their own system delay still in, and
/// that range cannot be modelled without their calibration record. Dropping such a session silently
/// would report a smaller campaign with no note of what was left out, and asserting on it would
/// abort a measurement over a data-handling case the format explicitly permits.
#[derive(Default)]
struct Declined {
    /// `"STATION: reason"` to the sessions and the normal points it cost.
    by_reason: BTreeMap<String, (usize, usize)>,
}

impl Declined {
    fn record(&mut self, station: &str, reason: &str, points: usize) {
        let e = self
            .by_reason
            .entry(format!("{station}: {reason}"))
            .or_insert((0, 0));
        e.0 += 1;
        e.1 += points;
    }

    fn points(&self) -> usize {
        self.by_reason.values().map(|(_, n)| n).sum()
    }
}

/// Read one satellite's span: normal points, orbit, stations, and the residual of every point that
/// passes the model's own admissibility checks, with the declined sessions alongside.
fn residuals(
    crd_text: &str,
    arcs: &[OrbitArc],
    sat: SatId,
    catalogue: &StationCatalogue,
    eccentricities: &StationCatalogue,
    series: &EopSeries,
    handling: &DataHandling,
) -> (Vec<Residual>, Declined) {
    let mut out = Vec::new();
    let mut declined = Declined::default();
    for session in parse_crd(crd_text) {
        let points = session.normal_points.len();
        if !session.two_way {
            declined.record(&session.station, "one-way session", points);
            continue;
        }
        if !session.applied.system_delay {
            // The published time of flight still contains the station's internal delay, and taking
            // it out needs that station's calibration record rather than anything modelled here.
            declined.record(&session.station, "system delay not applied", points);
            continue;
        }
        if session.applied.troposphere || session.applied.centre_of_mass {
            declined.record(
                &session.station,
                "publisher already applied a correction this model applies",
                points,
            );
            continue;
        }
        if session.quality_alert {
            declined.record(&session.station, "quality alert", points);
            continue;
        }
        if session.epoch_event != EpochEvent::GroundTransmit {
            declined.record(
                &session.station,
                "time tags are not transmit epochs",
                points,
            );
            continue;
        }
        let decimal_year = f64::from(session.year) + f64::from(session.day - 1) / 365.0;
        let Some(station) = station_at(catalogue, eccentricities, session.pad_id, decimal_year)
        else {
            // The orbit is SLRF2020 and the stations are ITRF2020. For a pad in both those are the
            // same coordinates by construction, but SLRF2020 also carries stations that joined the
            // network after ITRF2020's data cut-off, and for those there is no position to model
            // from. Naming the pad matters: this is a missing input, not a bad measurement.
            declined.record(
                &session.station,
                "no ITRF2020 solution for this pad",
                points,
            );
            continue;
        };
        let target = Target(sat.prn);
        if handling.is_excluded(session.pad_id, target, session.wavelength_nm, decimal_year) {
            declined.record(
                &session.station,
                "the Data Handling File deletes this station's data",
                points,
            );
            continue;
        }
        let advice =
            handling.range_bias(session.pad_id, target, session.wavelength_nm, decimal_year);
        model_pass(
            &session,
            &station,
            arcs,
            sat,
            series,
            advice,
            &mut out,
            &mut declined,
        );
    }
    (out, declined)
}

fn rms(v: &[f64]) -> f64 {
    (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// The measurement. Every number printed and asserted here was observed, then rounded outward.
#[test]
fn ilrs_normal_points_on_lageos_are_modelled_to_centimetres() {
    let catalogue = parse_sinex(SINEX);
    assert_eq!(
        catalogue.len(),
        12,
        "the twelve pads that observed that day"
    );
    let eccentricities = parse_sinex(ECCENTRICITY);
    let series = EopSeries::parse_finals2000a(EOP).expect("the EOP excerpt parses");
    let handling = DataHandling::parse(DATA_HANDLING);

    let mut all: Vec<Residual> = Vec::new();
    let (one, declined_one) = residuals(
        CRD_LAGEOS1,
        &[OrbitArc::parse(SP3_LAGEOS1)],
        LAGEOS1,
        &catalogue,
        &eccentricities,
        &series,
        &handling,
    );
    let (two, declined_two) = residuals(
        CRD_LAGEOS2,
        &[OrbitArc::parse(SP3_LAGEOS2)],
        LAGEOS2,
        &catalogue,
        &eccentricities,
        &series,
        &handling,
    );
    all.extend(one);
    all.extend(two);
    // On this one day every publisher followed the ILRS common case, which is what makes the count
    // below exact. The campaign over the week does not have that luxury and reports what it dropped.
    assert!(
        declined_one.by_reason.is_empty() && declined_two.by_reason.is_empty(),
        "no session on 2025-01-06 was declined: {:?} {:?}",
        declined_one.by_reason,
        declined_two.by_reason
    );

    let mut by_station: BTreeMap<&str, Vec<&Residual>> = BTreeMap::new();
    for r in &all {
        by_station.entry(r.label.as_str()).or_default().push(r);
    }

    println!("station        n    bias cm    rms cm   min el   max el  DHF");
    let mut worst_station = 0.0_f64;
    for (label, group) in &by_station {
        let v: Vec<f64> = group.iter().map(|r| r.value).collect();
        let el: Vec<f64> = group.iter().map(|r| r.elevation).collect();
        let station_rms = rms(&v);
        // One line per station, so a station carrying two verdicts across the two spheres shows
        // the one that moved its number rather than a summary of both.
        let advice = match group.iter().find_map(|r| r.handling) {
            Some(RangeBias::Apply(mm)) => format!("applied {mm:+.1} mm"),
            Some(RangeBias::Estimate) => "estimate".to_string(),
            None => String::new(),
        };
        println!(
            "{label:<12} {:4}  {:8.2}  {:8.2}  {:7.1}  {:7.1}  {advice}",
            v.len(),
            mean(&v) * 100.0,
            station_rms * 100.0,
            el.iter().copied().fold(f64::INFINITY, f64::min),
            el.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
        worst_station = worst_station.max(station_rms);
    }
    let v: Vec<f64> = all.iter().map(|r| r.value).collect();
    println!(
        "all {} normal points: bias {:.2} cm, RMS {:.2} cm, worst {:.2} cm",
        v.len(),
        mean(&v) * 100.0,
        rms(&v) * 100.0,
        v.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs())) * 100.0,
    );

    // The fixtures are committed, so the count is exact. It is asserted because the two ways this
    // chain has silently lost data — a case-sensitive record key and a session with no `h8` — both
    // showed up as fewer normal points and nothing else.
    assert_eq!(v.len(), 211, "89 LAGEOS-1 and 122 LAGEOS-2 normal points");
    // Measured: bias +0.59 cm, RMS 1.63 cm, worst single residual 7.02 cm, worst station 3.81 cm
    // (KUN2, one of the seven the Data Handling File flags without a value). The bounds sit above those, because what they guard against is not a slow drift but a
    // frame, time-scale, sign or missing-correction error, and every one of those is decimetres at
    // least — the centre-of-mass offset alone is 25 cm and the eccentricities are metres.
    assert!(rms(&v) < 0.03, "overall RMS {:.4} m", rms(&v));
    assert!(mean(&v).abs() < 0.02, "overall bias {:.4} m", mean(&v));
    assert!(
        worst_station < 0.05,
        "worst station RMS {worst_station:.4} m"
    );
    let worst = v.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
    assert!(worst < 0.12, "worst single residual {worst:.4} m");
}

/// The UTC calendar day an epoch belongs to, as `(year, month, day)`.
fn calendar_day(epoch: f64) -> (i32, u8, u8) {
    let d = UtcDateTime::from_naive_ns_since_j2000((epoch * 1e9).round() as i128, false);
    (d.year(), d.month(), d.day())
}

fn stdev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// A mean plus a diurnal and a semi-diurnal harmonic, least-squares fitted to one station's
/// residuals, with the formal error of each amplitude.
///
/// The two frequencies are the solar ones — one and two cycles per day — and they stand for
/// **bands**, not constituents. Over eight days the Rayleigh limit is `1/8 = 0.125` cycles per day,
/// while M2 and S2 are 0.068 apart and K1 and O1 are 0.073 apart: neither pair separates. What is
/// fitted is the diurnal and the semi-diurnal energy taken together, and a station observed only in
/// its own night contributes a badly conditioned design, which shows up as a large `sigma` rather
/// than as a wrong amplitude.
struct Harmonic {
    diurnal: f64,
    diurnal_sigma: f64,
    semidiurnal: f64,
    semidiurnal_sigma: f64,
    /// Smallest over largest diagonal of the information square root: a lower bound on how
    /// well-posed the fit is, and the number that separates a measured amplitude from one the
    /// solver invented to pass through a two-hour window of local time.
    conditioning: f64,
}

/// `points` are `(days since the span began, residual in metres)`.
///
/// `None` when the design is singular or there are no degrees of freedom left — an unobserved
/// direction is reported as absent rather than as an invented amplitude.
fn harmonic_fit(points: &[(f64, f64)]) -> Option<Harmonic> {
    const UNKNOWNS: usize = 5;
    let mut srif = Srif::<UNKNOWNS>::new();
    for &(t, value) in points {
        let a = core::f64::consts::TAU * t;
        srif.add_row(
            &[1.0, a.cos(), a.sin(), (2.0 * a).cos(), (2.0 * a).sin()],
            value,
            1.0,
        );
    }
    let x = srif.solve().ok()?;
    let p = srif.covariance().ok()?;
    let (residual_ss, rows) = srif.residual_stats();
    let dof = (rows as usize).checked_sub(UNKNOWNS)?;
    if dof == 0 {
        return None;
    }
    // Unit weights went in, so the covariance is `(HᵀH)⁻¹` and has to be scaled by the post-fit
    // variance before it means anything in metres.
    let scale = residual_ss / dof as f64;

    // `A = hypot(c, s)`, and its variance by linear propagation through that — the full 2x2 block,
    // because the cosine and sine coefficients of one frequency are correlated whenever the
    // sampling is not uniform in phase, which for a station that only tracks at night it never is.
    let amplitude = |c: usize, s: usize| {
        let (cc, ss) = (x[c], x[s]);
        let a = cc.hypot(ss);
        if a == 0.0 {
            return (0.0, 0.0);
        }
        let var =
            scale * (cc * cc * p[c][c] + ss * ss * p[s][s] + 2.0 * cc * ss * p[c][s]) / (a * a);
        (a, var.max(0.0).sqrt())
    };
    let (diurnal, diurnal_sigma) = amplitude(1, 2);
    let (semidiurnal, semidiurnal_sigma) = amplitude(3, 4);
    let r = srif.information_sqrt();
    let (mut smallest, mut largest) = (f64::INFINITY, 0.0_f64);
    for (i, row) in r.iter().enumerate() {
        smallest = smallest.min(row[i].abs());
        largest = largest.max(row[i].abs());
    }
    Some(Harmonic {
        diurnal,
        diurnal_sigma,
        semidiurnal,
        semidiurnal_sigma,
        conditioning: smallest / largest,
    })
}

/// How tightly a station's observations cluster in local solar time, as the resultant length of the
/// unit vectors at one cycle per day: `0` is uniform round the clock, `1` is every point at the same
/// hour.
///
/// This is the diagnostic for the diurnal fit rather than an input to it. A station that only ranges
/// in its own night samples a fraction of the phase circle, and a fit over that fraction cannot tell
/// a diurnal term from a constant — which is a large formal sigma, not a wrong amplitude, but it is
/// worth being able to see which one is which.
fn local_time_concentration(group: &[&Residual]) -> f64 {
    let (mut c, mut s) = (0.0, 0.0);
    for r in group {
        let d = UtcDateTime::from_naive_ns_since_j2000((r.epoch * 1e9).round() as i128, false);
        let utc = f64::from(d.hour()) + f64::from(d.minute()) / 60.0 + d.second() / 3600.0;
        let hours = (utc + r.longitude.to_degrees() / 15.0).rem_euclid(24.0);
        let a = core::f64::consts::TAU * hours / 24.0;
        c += a.cos();
        s += a.sin();
    }
    let n = group.len() as f64;
    (c / n).hypot(s / n)
}

/// A station's residuals grouped by the day they were measured on, in the order the days ran.
fn daily_means(group: &[&Residual]) -> Vec<((i32, u8, u8), f64, usize)> {
    let mut by_day: BTreeMap<(i32, u8, u8), Vec<f64>> = BTreeMap::new();
    for r in group {
        by_day
            .entry(calendar_day(r.epoch))
            .or_default()
            .push(r.value);
    }
    by_day
        .into_iter()
        .map(|(d, v)| (d, mean(&v), v.len()))
        .collect()
}

/// The trimmed orbit files are cut around the passes, not to whole days, so this checks the cut.
///
/// [`Sp3::interpolate`] takes the `order + 1` samples nearest by **index**. On a file with gaps
/// that is only the right stencil if the samples around the epoch are contiguous on the product's
/// own 120 s grid; across a gap it would draw a degree-10 polynomial through hours of arc and
/// return a plausible, wrong position. The generator leaves twenty minutes either side of every
/// normal point for this reason, and what follows is the check that it did.
#[test]
fn the_trimmed_orbit_arcs_carry_an_unbroken_grid_around_every_pass() {
    /// The ILRS combined orbit's sampling interval, seconds.
    const ORBIT_STEP: f64 = 120.0;

    let series = EopSeries::parse_finals2000a(WEEK_EOP).expect("the EOP excerpt parses");
    let mut checked = 0_usize;
    for (crd, texts) in [
        (WEEK_CRD_LAGEOS1, [WEEK_SP3_LAGEOS1_A, WEEK_SP3_LAGEOS1_B]),
        (WEEK_CRD_LAGEOS2, [WEEK_SP3_LAGEOS2_A, WEEK_SP3_LAGEOS2_B]),
    ] {
        let arcs = [OrbitArc::parse(texts[0]), OrbitArc::parse(texts[1])];
        assert!(
            arcs[0].end < arcs[1].start,
            "the weekly arcs abut rather than overlap: {} .. {} then {} .. {}",
            arcs[0].start,
            arcs[0].end,
            arcs[1].start,
            arcs[1].end
        );
        for session in parse_crd(crd) {
            let day = UtcDateTime::from_date(session.year, session.month as u8, session.day as u8)
                .expect("the session header carries a calendar date");
            let day_seconds = day.naive_ns_since_j2000() as f64 / 1e9;
            for np in &session.normal_points {
                let transmit = day_seconds + np.seconds_of_day;
                let Some(arc) = arcs.iter().find(|a| a.covers(transmit)) else {
                    panic!("no arc covers a normal point at {transmit}");
                };
                let t = transmit - arc.start;
                // The samples `Sp3::interpolate` will take: `order + 1` of them, centred on the one
                // nearest `t` and clamped to the ends of the file. A one-sided window at the edge of
                // an arc is correct — the product stops there — but a window that steps over a gap
                // is not, so what is asserted is that these run consecutively on the 120 s grid.
                let epochs = arc.sp3.epochs();
                let want = (INTERPOLATION_ORDER + 1).min(epochs.len());
                let nearest = epochs
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| (*a - t).abs().total_cmp(&(*b - t).abs()))
                    .map(|(i, _)| i)
                    .expect("the arc carries epochs");
                let first = nearest.saturating_sub(want / 2).min(epochs.len() - want);
                for pair in epochs[first..first + want].windows(2) {
                    assert!(
                        (pair[1] - pair[0] - ORBIT_STEP).abs() < 1e-6,
                        "the stencil for a normal point {:.0} s into arc {:.0} steps over a \
                         {:.0} s gap",
                        t,
                        arc.start,
                        pair[1] - pair[0],
                    );
                }
                assert_eq!(
                    want,
                    INTERPOLATION_ORDER + 1,
                    "an arc with fewer samples than the interpolation order"
                );
                checked += 1;
            }
        }
    }
    // Sanity that the loop above actually ran over the campaign's data rather than an empty file.
    assert_eq!(
        checked, 2158,
        "942 LAGEOS-1 and 1216 LAGEOS-2 normal points"
    );
    // The series is read here only so a fixture that does not span the campaign fails in this
    // cheap test rather than as silently dropped points in the expensive one.
    let (first, last) = series.span();
    assert!(
        first <= 60_680.0 && last >= 60_688.0,
        "the EOP excerpt has to reach past both ends of the span, MJD {first} .. {last}"
    );
}

/// The campaign. Eight days, two spheres, twenty-six pads and two weekly orbit arcs, asked three
/// questions a single day cannot answer: does a station's bias repeat, does anything move when the
/// orbit product changes, and is there anything left at a tidal frequency.
#[test]
#[allow(clippy::too_many_lines)] // one end-to-end integration scenario
fn a_week_of_normal_points_holds_its_bias_across_days_and_the_arc_seam() {
    // The editing rule, stated before it is applied and measured from a variable that is not the
    // residual: the number of raw ranges the publisher formed each normal point from. The table
    // below shows where the scatter changes — under thirty returns the residuals scatter at 4.0 cm
    // and above it at 1.5 cm, while the *bias* is the same in every bin. That is noise leaving, not
    // signal, and it is the publisher's own statement about the measurement rather than this
    // model's opinion of it.
    const MINIMUM_RETURNS: u32 = 30;
    // Chosen from the measurement below, where the well-sampled stations sit at 0.05 to 0.09 and
    // the ones that only ever range at the same local hour fall to 1e-3 and below. A fit under
    // this is not a small amplitude, it is no amplitude at all.
    const WELL_POSED: f64 = 0.02;
    // The first day of the second arc: the day-to-day step that lands here is the only one in the
    // campaign that also changes orbit product.
    const SEAM: (i32, u8, u8) = (2025, 1, 12);

    let catalogue = parse_sinex(WEEK_SINEX);
    let eccentricities = parse_sinex(WEEK_ECCENTRICITY);
    let series = EopSeries::parse_finals2000a(WEEK_EOP).expect("the EOP excerpt parses");
    let handling = DataHandling::parse(WEEK_DATA_HANDLING);

    let mut all: Vec<Residual> = Vec::new();
    let (one, declined_one) = residuals(
        WEEK_CRD_LAGEOS1,
        &[
            OrbitArc::parse(WEEK_SP3_LAGEOS1_A),
            OrbitArc::parse(WEEK_SP3_LAGEOS1_B),
        ],
        LAGEOS1,
        &catalogue,
        &eccentricities,
        &series,
        &handling,
    );
    let (two, declined_two) = residuals(
        WEEK_CRD_LAGEOS2,
        &[
            OrbitArc::parse(WEEK_SP3_LAGEOS2_A),
            OrbitArc::parse(WEEK_SP3_LAGEOS2_B),
        ],
        LAGEOS2,
        &catalogue,
        &eccentricities,
        &series,
        &handling,
    );
    all.extend(one);
    all.extend(two);

    println!("declined sessions        sessions  normal points");
    for (reason, (sessions, points)) in declined_one
        .by_reason
        .iter()
        .chain(declined_two.by_reason.iter())
    {
        println!("  {reason:<60} {sessions:3}  {points:4}");
    }
    let declined_points = declined_one.points() + declined_two.points();
    println!("  {declined_points} normal points declined of 2158 published");
    println!();

    let v: Vec<f64> = all.iter().map(|r| r.value).collect();
    let strong: Vec<&Residual> = all
        .iter()
        .filter(|r| r.returns >= MINIMUM_RETURNS)
        .collect();
    let strong_values: Vec<f64> = strong.iter().map(|r| r.value).collect();
    let span_start = all.iter().map(|r| r.epoch).fold(f64::INFINITY, f64::min);

    // --- day by day -----------------------------------------------------------------------
    let mut by_day: BTreeMap<(i32, u8, u8), Vec<f64>> = BTreeMap::new();
    for r in &strong {
        by_day
            .entry(calendar_day(r.epoch))
            .or_default()
            .push(r.value);
    }
    println!("day             n    bias cm    rms cm");
    for ((y, m, d), values) in &by_day {
        println!(
            "{y:04}-{m:02}-{d:02}  {:5}  {:8.2}  {:8.2}",
            values.len(),
            mean(values) * 100.0,
            rms(values) * 100.0,
        );
    }

    // --- station by station ---------------------------------------------------------------
    let mut by_station: BTreeMap<&str, Vec<&Residual>> = BTreeMap::new();
    for r in &strong {
        by_station.entry(r.label.as_str()).or_default().push(r);
    }
    println!();
    println!(
        "station        n  days   bias cm    rms cm  day sd  lst R   cond    diurnal cm  \
         semidiurnal cm  DHF"
    );
    let mut worst_station = 0.0_f64;
    let mut significant = Vec::new();
    for (label, group) in &by_station {
        let values: Vec<f64> = group.iter().map(|r| r.value).collect();
        let station_rms = rms(&values);
        let days = daily_means(group);
        let day_means: Vec<f64> = days.iter().map(|(_, m, _)| *m).collect();
        let fit = harmonic_fit(
            &group
                .iter()
                .map(|r| ((r.epoch - span_start) / 86_400.0, r.value))
                .collect::<Vec<_>>(),
        );
        let advice = match group.iter().find_map(|r| r.handling) {
            Some(RangeBias::Apply(mm)) => format!("applied {mm:+.1} mm"),
            Some(RangeBias::Estimate) => "estimate".to_string(),
            None => String::new(),
        };
        let harmonics = match &fit {
            Some(h) => {
                // Three sigma is the line between an amplitude the data support and one the fit
                // could have produced from noise on this sampling.
                if h.conditioning <= WELL_POSED {
                    // The amplitude is printed as absent rather than as a number. A fit through two
                    // hours of local time returns metres of "diurnal" signal with a formal sigma to
                    // match, and writing that in a table invites someone to read it as a
                    // measurement.
                    format!("{:6.4}      --              --", h.conditioning)
                } else {
                    if h.semidiurnal > 3.0 * h.semidiurnal_sigma {
                        significant.push((*label, h.semidiurnal));
                    }
                    format!(
                        "{:6.4}  {:6.2} +-{:5.2}  {:6.2} +-{:5.2}",
                        h.conditioning,
                        h.diurnal * 100.0,
                        h.diurnal_sigma * 100.0,
                        h.semidiurnal * 100.0,
                        h.semidiurnal_sigma * 100.0,
                    )
                }
            }
            None => "    --      --              --".to_string(),
        };
        println!(
            "{label:<12} {:4}  {:4}  {:8.2}  {:8.2}  {:6.2}  {:5.2}  {harmonics}  {advice}",
            values.len(),
            days.len(),
            mean(&values) * 100.0,
            station_rms * 100.0,
            stdev(&day_means) * 100.0,
            local_time_concentration(group),
        );
        worst_station = worst_station.max(station_rms);
    }

    // --- what the tail is made of ---------------------------------------------------------
    //
    // A campaign statistic hides where its worst points came from, and "the RMS is 2 cm" reads very
    // differently depending on whether that is every station or one of them. So the twelve largest
    // are named, with the day and the elevation, which is what tells a low-elevation refraction
    // shortfall apart from a station that is simply wrong.
    let mut tail: Vec<&Residual> = all.iter().collect();
    tail.sort_by(|a, b| b.value.abs().total_cmp(&a.value.abs()));
    println!();
    println!("the twelve largest residuals");
    for r in tail.iter().take(12) {
        let (y, m, d) = calendar_day(r.epoch);
        println!(
            "  {:<12} {y:04}-{m:02}-{d:02}  {:+8.2} cm at {:5.1} deg, from {:5} returns",
            r.label,
            r.value * 100.0,
            r.elevation,
            r.returns,
        );
    }

    // Residual scatter against the publisher's own return count. The bins are the measurement's
    // property, not the residual's, so this is a check on the data rather than a way of choosing
    // which residuals to believe.
    println!();
    println!("returns in the bin      n    bias cm    rms cm   worst cm");
    for (lo, hi, label) in [
        (0, 30, "under 30"),
        (30, 50, "30 to 50"),
        (50, 100, "50 to 100"),
        (100, 1000, "100 to 1000"),
        (1000, u32::MAX, "1000 and over"),
    ] {
        let bin: Vec<f64> = all
            .iter()
            .filter(|r| r.returns >= lo && r.returns < hi)
            .map(|r| r.value)
            .collect();
        if bin.is_empty() {
            continue;
        }
        println!(
            "  {label:<18} {:5}  {:8.2}  {:8.2}  {:9.2}",
            bin.len(),
            mean(&bin) * 100.0,
            rms(&bin) * 100.0,
            bin.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs())) * 100.0,
        );
    }

    // --- the arc seam ---------------------------------------------------------------------
    //
    // There is no overlap to compare on: the weekly products abut at a two-minute seam, so what a
    // change of orbit product can be seen in is the step across it. The null is the step the same
    // station takes between two consecutive days *inside* one arc, which carries the same
    // day-to-day geometry and weather but no change of product.
    let mut within: Vec<f64> = Vec::new();
    let mut across: Vec<(&str, f64)> = Vec::new();
    for (label, group) in &by_station {
        let days = daily_means(group);
        for pair in days.windows(2) {
            // A station that skipped a day contributes nothing: two means a week apart are not a
            // day-to-day difference and would inflate the null.
            let (a, b) = (&pair[0], &pair[1]);
            let consecutive = b.0.2 == a.0.2 + 1 && b.0.1 == a.0.1;
            if !consecutive {
                continue;
            }
            if b.0 == SEAM {
                across.push((*label, b.1 - a.1));
            } else {
                within.push(b.1 - a.1);
            }
        }
    }
    let across_values: Vec<f64> = across.iter().map(|(_, d)| *d).collect();
    println!();
    println!("arc seam, 2025-01-11 (arc 250111) to 2025-01-12 (arc 250118):");
    for (label, d) in &across {
        println!("  {label:<12} {:+8.2} cm", d * 100.0);
    }
    println!(
        "  across the seam: {} stations, mean {:+.2} cm, RMS {:.2} cm",
        across_values.len(),
        mean(&across_values) * 100.0,
        rms(&across_values) * 100.0,
    );
    println!(
        "  within one arc:  {} consecutive-day steps, mean {:+.2} cm, RMS {:.2} cm",
        within.len(),
        mean(&within) * 100.0,
        rms(&within) * 100.0,
    );

    println!();
    println!(
        "all {} modelled normal points over {} days: bias {:+.2} cm, RMS {:.2} cm, worst {:.2} cm",
        v.len(),
        by_day.len(),
        mean(&v) * 100.0,
        rms(&v) * 100.0,
        v.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs())) * 100.0,
    );
    println!(
        "the {} of them from {MINIMUM_RETURNS} returns or more: bias {:+.2} cm, RMS {:.2} cm, \
         worst {:.2} cm",
        strong_values.len(),
        mean(&strong_values) * 100.0,
        rms(&strong_values) * 100.0,
        strong_values
            .iter()
            .copied()
            .fold(0.0_f64, |a, b| a.max(b.abs()))
            * 100.0,
    );
    println!(
        "stations whose semi-diurnal amplitude exceeds three formal sigma: {}",
        if significant.is_empty() {
            "none".to_string()
        } else {
            significant
                .iter()
                .map(|(l, a)| format!("{l} {:.2} cm", a * 100.0))
                .collect::<Vec<_>>()
                .join(", ")
        }
    );

    assert_eq!(
        v.len() + declined_points,
        2158,
        "942 LAGEOS-1 and 1216 LAGEOS-2 normal points, every one modelled or accounted for"
    );
    assert_eq!(by_day.len(), 8, "2025-01-05 through 2025-01-12");
    // The bias is asserted over everything, because a sign or frame error would move every point
    // whatever its return count; the scatter is asserted over the edited set, because the whole
    // point of the table above is that the weak points carry noise the model cannot be blamed for.
    assert!(mean(&v).abs() < 0.02, "overall bias {:.4} m", mean(&v));
    assert!(
        rms(&strong_values) < 0.03,
        "RMS over the well-sampled points {:.4} m",
        rms(&strong_values)
    );
    assert!(
        worst_station < 0.05,
        "worst station RMS {worst_station:.4} m"
    );
    // The ticket's question. A change of orbit product that moved the range would show up here as a
    // seam step larger than the ordinary day-to-day step, and the bound is on the ratio rather than
    // on the step itself, so it stays meaningful if the network's noise changes.
    assert!(
        rms(&across_values) < 2.0 * rms(&within),
        "the arc seam moves stations by {:.2} cm against a day-to-day {:.2} cm",
        rms(&across_values) * 100.0,
        rms(&within) * 100.0,
    );
}
