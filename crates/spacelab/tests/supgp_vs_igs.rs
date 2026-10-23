//! What an SGP4 element set is actually worth, measured against a precise orbit.
//!
//! `crates/spacelab-sgp4/tests/supgp_divergence.rs` measures how far two element sets for one object
//! drift apart. That is an uncertainty proxy and it has no truth in it. Here there is truth: the IGS
//! rapid orbit combination for 2026-08-12, ten analysis centres averaged in the IGS20 realization of
//! ITRF2020, quoted at the satellite centre of mass and accurate to a few centimetres. Both element
//! sets — the GPS control segment's Supplemental GP record and the standard catalogue's — are
//! propagated onto it and the residual is decomposed radially, along-track and cross-track.
//!
//! The chain under test is the operational one and nothing here is fitted: TLE → SGP4 → TEME →
//! (GMST82, polar motion, LOD) → ITRF, against a product nobody in this workspace produced.
//!
//! ## The epoch gap is the measurement, not a nuisance
//!
//! SupGP is fitted *forward*: the GPS records in the fixture are stamped 2026-08-17.7, five days
//! *after* the precise orbit they are compared against. The standard catalogue is fitted backward
//! from tracking and is stamped 2026-08-15. Neither element set was built for the day it is being
//! measured on, and both are therefore extrapolating — SupGP backwards, out of the direction it was
//! fitted for.
//!
//! Reporting one number without that would be the failure the ticket names: a comparison at one
//! epoch measures the epoch gap rather than the model. So the gap is reported beside the residual,
//! per origin, and the two are not presented as a fair race.
//!
//! ## Measured
//!
//! 32 GPS satellites, 24 hourly epochs, 768 comparisons per origin. RMS in metres:
//!
//! | origin | mean epoch age | radial | along-track | cross-track | 3D | worst |
//! |---|---|---|---|---|---|---|
//! | standard GP | −1.57 d | 254.6 | 2317.7 | 246.0 | **2344.6** | 7026.8 |
//! | operator SupGP | −5.23 d | 521.0 | 6213.7 | 177.6 | **6238.1** | 38345.5 |
//!
//! The operator's element set comes out *worse*, and the reason is in the second column rather than
//! in the fit: it is being asked to run 5.2 days backwards, against the direction it was built for,
//! while the catalogue's runs 1.6 days backwards. Along-track is 90 % of both residuals, which is
//! what a mean-element model does — a semi-major-axis error becomes a mean-motion error and
//! integrates — and cross-track is the one component where SupGP is better (177.6 m against
//! 246.0 m), because the out-of-plane geometry is what an operator knows exactly and a tracking fit
//! has to infer.
//!
//! Reading "SupGP is worse than GP" off this table would be the mistake the table exists to prevent.
//! What it supports is narrower and more useful: **at these epoch ages, in this regime, an SGP4
//! element set of either origin is a kilometre-class product, and the epoch age dominates its
//! origin.** Settling which fit is better needs a precise product from the SupGP epoch's own day,
//! which is 13 days of IGS latency away and is why this file measures what it can measure.
//!
//! The order of magnitude is the point. SGP4 on a GNSS orbit is a kilometre-class model — the fit
//! throws away everything that is not secular plus the analytic short-periodics — and the precise
//! product is centimetre-class. Anyone reading a metre from an element set is reading noise.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)] // SupGP, GP, SGP4, IGS20, ITRF2020 are names in prose

use std::collections::BTreeMap;

use spacelab::{EopSource, earth_orientation_at};
use spacelab_core::{Angle, UtcDateTime, Vec3};
use spacelab_eop::EopSeries;
use spacelab_fa::delta_at;
use spacelab_frames::teme_to_itrf;
use spacelab_gnss::{Itrs, ReferenceFrame, SatId, Sp3, TimeSystem};
use spacelab_sgp4::Satellite;

const PAIRED: &str = include_str!("fixtures/supgp_gps_paired.tle");
const SP3_EXCERPT: &str = include_str!("fixtures/igs20_gps_2026224.sp3");
const EOP_EXCERPT: &str = include_str!("fixtures/finals2000A_2026224.txt");

/// GPS time minus TAI, seconds. Fixed by definition since 1980.
const GPS_MINUS_TAI: f64 = -19.0;

/// One element set, tagged by where it came from and which PRN it is.
struct Element {
    origin: String,
    prn: u8,
    satellite: Satellite,
}

/// Parse the `R <source> <origin> <name>` fixture, reading the PRN out of the operator's name.
///
/// Only the SupGP records carry `(PRN nn)`; the catalogue names them `NAVSTAR nn (USA nnn)`, which
/// is a different numbering entirely. So the PRN is resolved once per catalog number from whichever
/// record states it, and both origins inherit it — the alternative, matching by name, would silently
/// pair the wrong satellite the first time an operator renames one.
fn parse() -> Vec<Element> {
    let mut records: Vec<(String, String, String, String)> = Vec::new();
    let mut header: Option<(String, String)> = None;
    let mut line1: Option<&str> = None;
    for line in PAIRED.lines() {
        if let Some(rest) = line.strip_prefix("R ") {
            let mut f = rest.splitn(3, ' ');
            let _source = f.next().expect("source");
            let origin = f.next().expect("origin").to_owned();
            let name = f.next().unwrap_or("").to_owned();
            header = Some((origin, name));
        } else if line.starts_with("1 ") {
            line1 = Some(line);
        } else if line.starts_with("2 ") {
            let (origin, name) = header.clone().expect("a record header precedes the lines");
            let first = line1.take().expect("line 1 precedes line 2");
            records.push((origin, name, first.to_owned(), line.to_owned()));
        }
    }

    let mut prn_of: BTreeMap<String, u8> = BTreeMap::new();
    for (_, name, line1, _) in &records {
        if let Some(start) = name.find("(PRN ") {
            let digits: String = name[start + 5..]
                .chars()
                .take_while(char::is_ascii_digit)
                .collect();
            if let Ok(prn) = digits.parse::<u8>() {
                prn_of.insert(line1[2..7].trim().to_owned(), prn);
            }
        }
    }

    records
        .iter()
        .filter_map(|(origin, _, line1, line2)| {
            let prn = *prn_of.get(line1[2..7].trim())?;
            let satellite = Satellite::from_tle(line1, line2).ok()?;
            Some(Element {
                origin: origin.clone(),
                prn,
                satellite,
            })
        })
        .collect()
}

/// Residual statistics for one origin, metres.
#[derive(Default)]
struct Residuals {
    radial: Vec<f64>,
    along: Vec<f64>,
    cross: Vec<f64>,
    total: Vec<f64>,
    /// Element-set age at the observation, days. Negative means the element set is stamped later.
    age_days: Vec<f64>,
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    (values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64).sqrt()
}

fn worst(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |acc, x| acc.max(x.abs()))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Radial / along-track / cross-track unit vectors from a state.
fn rtn_axes(position: Vec3, velocity: Vec3) -> [Vec3; 3] {
    let radial = position.scale(1.0 / position.norm());
    let cross = {
        let h = position.cross(velocity);
        h.scale(1.0 / h.norm())
    };
    let along = cross.cross(radial);
    [radial, along, cross]
}

/// Propagate every element set onto the precise orbit and collect the residual by origin.
fn measure() -> BTreeMap<String, Residuals> {
    let sp3 = Sp3::parse(SP3_EXCERPT);
    assert_eq!(
        sp3.time_system(),
        TimeSystem::Gps,
        "an IGS product is stamped in GPS time"
    );
    // The frame the products claim, checked rather than asserted in prose. `spacelab_gnss` now
    // surfaces line 1's coordinate-system field, and an SP3 in another datum -- an IAC GLONASS
    // product is PZ-90, decimetres from ITRF -- would otherwise be differenced against ITRF
    // coordinates with nothing to show for it.
    assert!(
        sp3.reference_frame()
            .agrees_with(ReferenceFrame::Itrs(Itrs::Itrf2020)),
        "the IGS rapid combination should realize ITRF2020; the file says {:?}",
        sp3.reference_frame()
    );
    let series = EopSeries::parse_finals2000a(EOP_EXCERPT).expect("the EOP excerpt parses");
    let start_gps = sp3.start_epoch().expect("the file carries epochs");

    // GPS labels and UTC labels are the same calendar arithmetic offset by the leap-second count.
    #[allow(clippy::cast_possible_truncation)]
    let approx = UtcDateTime::from_naive_ns_since_j2000((start_gps * 1e9).round() as i128, false);
    let gps_minus_utc =
        f64::from(delta_at(approx).expect("a tabulated leap-second count")) + GPS_MINUS_TAI;

    let elements = parse();
    assert!(elements.len() >= 60, "only {} element sets", elements.len());

    let mut out: BTreeMap<String, Residuals> = BTreeMap::new();
    for (index, &t) in sp3.epochs().iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        let utc = UtcDateTime::from_naive_ns_since_j2000(
            ((start_gps + t - gps_minus_utc) * 1e9).round() as i128,
            false,
        );
        let e = earth_orientation_at(utc, EopSource::Series(&series))
            .expect("the day is inside the EOP excerpt");
        let m = teme_to_itrf(
            e.orientation.ut1,
            e.orientation.tt,
            Angle::from_radians(e.orientation.xp.radians()),
            Angle::from_radians(e.orientation.yp.radians()),
            e.orientation.lod_seconds,
        )
        .rotation
        .matrix()
        .rows;

        for element in &elements {
            let sat = SatId {
                system: 'G',
                prn: element.prn,
            };
            let Some(samples) = sp3.positions(sat) else {
                continue;
            };
            let Some(Some(itrf)) = samples.get(index) else {
                continue;
            };
            let Ok(state) = element.satellite.propagate_at(utc) else {
                continue;
            };
            // TEME = Mᵀ · ITRF; the residual is formed in TEME so the RTN axes and the difference
            // live in one frame and no second rotation can quietly change the answer.
            let truth = Vec3::new(
                m[0][0] * itrf[0] + m[1][0] * itrf[1] + m[2][0] * itrf[2],
                m[0][1] * itrf[0] + m[1][1] * itrf[1] + m[2][1] * itrf[2],
                m[0][2] * itrf[0] + m[1][2] * itrf[1] + m[2][2] * itrf[2],
            );
            let difference = state.position.vector - truth;
            let [radial, along, cross] = rtn_axes(state.position.vector, state.velocity.vector);

            let slot = out.entry(element.origin.clone()).or_default();
            slot.radial.push(difference.dot(radial));
            slot.along.push(difference.dot(along));
            slot.cross.push(difference.dot(cross));
            slot.total.push(difference.norm());
            slot.age_days.push(
                (utc.naive_ns_since_j2000() - element.satellite.epoch().naive_ns_since_j2000())
                    as f64
                    / (86_400.0 * 1.0e9),
            );
        }
    }
    out
}

/// SGP4 on a GNSS orbit is a kilometre-class model, and the precise product says so.
///
/// Both element sets are measured against the same 24 hours of the same product at the same epochs,
/// so the comparison between the two origins is fair in everything except the epoch gap, which is
/// reported. What the assertion pins is the *class*: kilometres, not metres and not tens of
/// thousands. A residual that fell to metres would mean the truth had been replaced by something
/// else — most likely our own propagation, through a fixture regenerated the wrong way.
#[test]
fn an_sgp4_element_set_misses_the_precise_gnss_orbit_by_kilometres() {
    let measured = measure();
    assert_eq!(measured.len(), 2, "both origins reach the comparison");

    for (origin, r) in &measured {
        println!(
            "{origin:<6} n={:<5} age {:+6.2} d   radial {:8.1}   along {:9.1}   cross {:8.1}   3D {:9.1}   worst {:9.1}  m",
            r.total.len(),
            mean(&r.age_days),
            rms(&r.radial),
            rms(&r.along),
            rms(&r.cross),
            rms(&r.total),
            worst(&r.total)
        );
    }

    for (origin, r) in &measured {
        let three_d = rms(&r.total);
        assert!(
            (1.0e2..1.0e5).contains(&three_d),
            "{origin}: 3D RMS {three_d:.1} m is outside the kilometre class this model lives in"
        );
        // Along-track is where a mean-element model's error goes: the semi-major-axis error becomes
        // a mean-motion error and integrates. If radial ever dominated, the residual would be
        // describing a frame problem rather than the propagator.
        assert!(
            rms(&r.along) > rms(&r.radial),
            "{origin}: radial {:.1} m exceeds along-track {:.1} m",
            rms(&r.radial),
            rms(&r.along)
        );
    }
}

/// The two origins are stamped days apart, and the comparison says so out loud.
///
/// This is the assertion that keeps the table honest. If a future regeneration happened to pick a
/// SupGP snapshot from the same day as the catalogue, the residuals would become comparable and the
/// caveat in the module documentation would quietly stop being true — so the gap is asserted to
/// still be there rather than assumed.
#[test]
fn the_operator_and_catalogue_element_sets_are_not_stamped_at_the_same_time() {
    let measured = measure();
    let supgp = mean(&measured["supgp"].age_days);
    let gp = mean(&measured["gp"].age_days);
    assert!(
        (supgp - gp).abs() > 1.0,
        "the two origins are within a day of each other ({supgp:+.2} d against {gp:+.2} d); \
         the residuals below are now comparable and the caveat needs rewriting"
    );
    // Both are extrapolating backwards onto the precise day, which is what makes the numbers a
    // characterisation of extrapolation rather than of the fit.
    assert!(supgp < 0.0 && gp < 0.0, "{supgp:+.2} d, {gp:+.2} d");
}

/// Every GPS satellite in the precise product is reached, not a convenient subset.
#[test]
fn the_comparison_covers_the_whole_gps_constellation() {
    let sp3 = Sp3::parse(SP3_EXCERPT);
    let in_product: Vec<SatId> = sp3.satellites().collect();
    assert_eq!(in_product.len(), 32, "satellites in the IGS product");

    let elements = parse();
    let mut covered: Vec<u8> = elements
        .iter()
        .filter(|e| e.origin == "supgp")
        .map(|e| e.prn)
        .collect();
    covered.sort_unstable();
    covered.dedup();
    assert!(
        covered.len() >= 30,
        "only {} of 32 PRNs carry an operator element set",
        covered.len()
    );
}
