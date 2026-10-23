//! Two fits of one object, from disjoint information: the operator's and the catalogue's.
//!
//! Celestrak's Supplemental GP (SupGP) element sets are contributed by the operators — SpaceX,
//! OneWeb, Iridium, the GPS control segment, the ILRS analysis centres for the laser targets — and
//! are fitted *forward* from an ephemeris the operator already has. The standard GP catalogue is
//! fitted *backward* from tracking observations. For one object at one time there are therefore two
//! element sets built from information that does not overlap, which is a case the catalogue almost
//! never presents.
//!
//! Neither is truth, and this file does not pretend otherwise. What the divergence between them
//! measures is how much an element set *commits to* — the natural input to a conjunction covariance,
//! which today comes from the caller and nothing produces it. `crates/spacelab/tests/supgp_vs_igs.rs`
//! is the other half: there a genuinely precise product exists and both element sets are measured
//! against it.
//!
//! The fixture is cut by `tools/gen_supgp_fixtures.py` from snapshots taken 2026-08-15 (SupGP
//! sources within minutes of each other; the standard catalogue about an hour earlier).
//!
//! ## Measured
//!
//! Separation between the operator's element set and the catalogue's, propagated to a common time
//! and read at 1, 7 and 30 days past the later of the two epochs. Median over the objects of each
//! source, in kilometres:
//!
//! | source | objects | +1 day | +7 days | +30 days |
//! |---|---|---|---|---|
//! | starlink (≈550 km) | 60 | **7.548** | **530.265** | **9210.155** |
//! | oneweb (≈1200 km) | 40 | 1.097 | 31.927 | 769.699 |
//! | iridium (≈780 km) | 80 | 0.400 | 32.838 | 672.371 |
//! | gps (MEO) | 32 | 1.108 | 2.377 | 12.205 |
//! | cpf (laser targets) | 111 | 1.616 | 2.541 | 11.763 |
//!
//! The ordering is the physics and not the providers: the two drag-free regimes stay inside 13 km
//! after a month, and Starlink — 550 km, high area-to-mass, actively manoeuvring — is three orders
//! of magnitude worse over the same span. A conjunction screening that treats a 30-day-old element
//! set as if it carried the same information in both regimes is wrong by that factor.
//!
//! And on the laser targets, where several ILRS analysis centres fit the *same* CPF prediction:
//! **45 objects with more than one provider, median spread 896.0 m, worst 3020.9 m**, at the latest
//! epoch among the providers. That is a disagreement number that needs no truth at all — two
//! independent fits of one published prediction, differing by most of a kilometre.
//!
//! ## Two things the shape of the data forces
//!
//! **An object may carry several SupGP records.** 45 of the 111 laser targets in the fixture do,
//! because several ILRS analysis centres each fit the same CPF prediction independently. Reducing
//! that to "the first one" would discard the only place in the catalogue where two providers state a
//! disagreement about one object at one time — so it is measured instead.
//!
//! **The two epochs are never the same.** Comparing at "the epoch" would measure the epoch gap, not
//! the fit. Every comparison here is at an absolute UTC instant, reached from each element set's own
//! epoch, so the gap enters as propagation and not as an offset.

#![allow(clippy::expect_used, clippy::unwrap_used)]
#![allow(clippy::doc_markdown)] // SupGP, GP, SpaceX, OneWeb, ILRS are names in prose

use std::collections::BTreeMap;

use spacelab_core::UtcDateTime;
use spacelab_sgp4::Satellite;

const FIXTURE: &str = include_str!("fixtures/supgp_paired.tle");

/// Days past the later epoch at which the two fits are compared.
const HORIZONS_DAYS: [f64; 3] = [1.0, 7.0, 30.0];

/// One element set from the fixture, with the tag that says where it came from.
struct Record {
    source: String,
    origin: String,
    key: String,
    satellite: Satellite,
}

fn parse() -> Vec<Record> {
    let mut out = Vec::new();
    let mut pending: Option<(String, String)> = None;
    let mut line1: Option<&str> = None;
    for line in FIXTURE.lines() {
        if let Some(rest) = line.strip_prefix("R ") {
            let mut f = rest.splitn(3, ' ');
            let source = f.next().expect("source").to_owned();
            let origin = f.next().expect("origin").to_owned();
            pending = Some((source, origin));
        } else if line.starts_with("1 ") {
            line1 = Some(line);
        } else if line.starts_with("2 ") {
            let (source, origin) = pending.clone().expect("a record header precedes the lines");
            let first = line1.take().expect("line 1 precedes line 2");
            let satellite = Satellite::from_tle(first, line)
                .unwrap_or_else(|e| panic!("{source}/{origin} {}: {e}", &first[2..7]));
            out.push(Record {
                source,
                origin,
                key: first[2..7].trim().to_owned(),
                satellite,
            });
        }
    }
    out
}

/// Group the fixture by `(source, catalog key)`, keeping *every* record.
fn by_object() -> BTreeMap<(String, String), Vec<Record>> {
    let mut grouped: BTreeMap<(String, String), Vec<Record>> = BTreeMap::new();
    for record in parse() {
        grouped
            .entry((record.source.clone(), record.key.clone()))
            .or_default()
            .push(record);
    }
    grouped
}

/// The instant `days` after `from`, as a UTC label.
fn plus_days(from: UtcDateTime, days: f64) -> UtcDateTime {
    #[allow(clippy::cast_possible_truncation)]
    let ns = from.naive_ns_since_j2000() + (days * 86_400.0 * 1.0e9) as i128;
    UtcDateTime::from_naive_ns_since_j2000(ns, false)
}

/// Separation of two element sets at one absolute instant, in metres. `None` if either refuses.
fn separation(a: &Satellite, b: &Satellite, at: UtcDateTime) -> Option<f64> {
    let (Ok(x), Ok(y)) = (a.propagate_at(at), b.propagate_at(at)) else {
        return None;
    };
    Some((x.position.vector - y.position.vector).norm())
}

/// The later of two epochs — the instant from which both element sets are propagating forward.
fn later(a: UtcDateTime, b: UtcDateTime) -> UtcDateTime {
    if a.naive_ns_since_j2000() >= b.naive_ns_since_j2000() {
        a
    } else {
        b
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if values.is_empty() {
        return f64::NAN;
    }
    values[values.len() / 2]
}

/// The operator's fit and the catalogue's are not the same numbers relabelled.
///
/// The premise of every measurement below is that the two element sets carry independent
/// information. If some source were simply republishing the catalogue, its divergence would be zero
/// and the "uncertainty proxy" would be a proxy for nothing. Checked at the later of the two epochs,
/// where the propagation has had no time to hide an agreement.
#[test]
fn the_operator_fit_and_the_catalogue_fit_are_genuinely_different_element_sets() {
    let mut checked = 0_usize;
    for ((source, key), records) in by_object() {
        let gp = records
            .iter()
            .find(|r| r.origin == "gp")
            .expect("a gp record");
        for supgp in records.iter().filter(|r| r.origin == "supgp") {
            let at = later(gp.satellite.epoch(), supgp.satellite.epoch());
            let Some(gap) = separation(&gp.satellite, &supgp.satellite, at) else {
                continue;
            };
            assert!(
                gap > 1.0,
                "{source} {key}: the two fits agree to {gap:.3e} m — one is a copy of the other"
            );
            checked += 1;
        }
    }
    assert!(checked > 300, "only {checked} pairs compared");
}

/// How far apart the two fits drift, per source, at 1, 7 and 30 days.
///
/// This is the uncertainty proxy, and its units are the ones a conjunction screening cares about.
/// The assertion is deliberately a wide band rather than a tight tolerance: the number depends on
/// the solar activity of the week the snapshot was taken, and pinning it would make the test a
/// record of one Tuesday. What is pinned is the *ordering* the physics requires — the drag-driven
/// LEO constellations must diverge faster than the drag-free GNSS and laser targets — and that the
/// divergence grows with the horizon rather than staying flat.
#[test]
fn the_two_fits_diverge_faster_where_drag_dominates() {
    let grouped = by_object();
    let mut per_source: BTreeMap<String, Vec<Vec<f64>>> = BTreeMap::new();

    for ((source, _), records) in &grouped {
        let gp = records
            .iter()
            .find(|r| r.origin == "gp")
            .expect("a gp record");
        // One object contributes once, through its first operator record: the several-provider case
        // is measured on its own terms in the test below, and letting it in here would weight the
        // laser targets by how many centres happened to fit them.
        let Some(supgp) = records.iter().find(|r| r.origin == "supgp") else {
            continue;
        };
        let start = later(gp.satellite.epoch(), supgp.satellite.epoch());
        let slot = per_source
            .entry(source.clone())
            .or_insert_with(|| vec![Vec::new(); HORIZONS_DAYS.len()]);
        for (index, days) in HORIZONS_DAYS.iter().enumerate() {
            if let Some(gap) = separation(&gp.satellite, &supgp.satellite, plus_days(start, *days))
            {
                slot[index].push(gap);
            }
        }
    }

    let mut medians: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for (source, horizons) in &mut per_source {
        let row: Vec<f64> = horizons.iter_mut().map(|v| median(v)).collect();
        println!(
            "{source:<9} n={:<4} 1d {:8.3} km   7d {:9.3} km   30d {:10.3} km",
            horizons[0].len(),
            row[0] / 1000.0,
            row[1] / 1000.0,
            row[2] / 1000.0
        );
        medians.insert(source.clone(), row);
    }

    for (source, row) in &medians {
        for pair in row.windows(2) {
            assert!(
                pair[1] > pair[0],
                "{source}: the divergence does not grow with the horizon ({pair:?})"
            );
        }
    }

    let starlink = medians["starlink"][2];
    let oneweb = medians["oneweb"][2];
    let gps = medians["gps"][2];
    let cpf = medians["cpf"][2];
    assert!(
        starlink > gps && starlink > cpf,
        "the drag-dominated LEO constellation does not diverge fastest: \
         starlink {starlink:.0} m, gps {gps:.0} m, cpf {cpf:.0} m"
    );
    assert!(
        oneweb > gps,
        "OneWeb at 1200 km does not out-diverge GNSS: {oneweb:.0} m against {gps:.0} m"
    );
}

/// Several analysis centres fit the same laser target, and they do not agree.
///
/// This is the one case in the whole catalogue where two providers state, on the same day, two
/// element sets for one object from the same underlying prediction. The spread between *them* is a
/// provider-disagreement number that needs no truth at all — and it is the reason the fixture keeps
/// every record for an object rather than the first.
#[test]
fn several_analysis_centres_fit_one_laser_target_and_disagree() {
    let grouped = by_object();
    let mut spreads = Vec::new();
    let mut objects = 0_usize;
    for ((source, _), records) in &grouped {
        if source != "cpf" {
            continue;
        }
        let operator: Vec<&Record> = records.iter().filter(|r| r.origin == "supgp").collect();
        if operator.len() < 2 {
            continue;
        }
        objects += 1;
        // Compared at the latest epoch among the providers, so no fit is asked to extrapolate
        // backwards past the data it was built from.
        let at = operator
            .iter()
            .map(|r| r.satellite.epoch())
            .reduce(later)
            .expect("at least two records");
        let mut worst = 0.0_f64;
        for (index, first) in operator.iter().enumerate() {
            for second in &operator[index + 1..] {
                if let Some(gap) = separation(&first.satellite, &second.satellite, at) {
                    worst = worst.max(gap);
                }
            }
        }
        if worst > 0.0 {
            spreads.push(worst);
        }
    }

    assert!(objects >= 40, "only {objects} multi-provider objects");
    let count = spreads.len();
    let worst = spreads.iter().copied().fold(0.0_f64, f64::max);
    let middle = median(&mut spreads);
    println!(
        "cpf multi-provider: {objects} objects, {count} with a spread, median {middle:.1} m, worst {worst:.1} m"
    );
    assert!(
        middle > 1.0,
        "the analysis centres agree to {middle:.3e} m — they are republishing one fit"
    );
}

/// The fixture keeps every record an object has, and the reader does not silently take the first.
///
/// A structural check rather than a numerical one, because the failure it guards against is silent:
/// a `HashMap<catalog, record>` built anywhere in this path would drop 84 of the 195 laser-target
/// records and every measurement above would still produce a number.
#[test]
fn no_object_with_several_operator_records_is_reduced_to_one() {
    let grouped = by_object();
    let multi: Vec<_> = grouped
        .iter()
        .filter(|(_, records)| records.iter().filter(|r| r.origin == "supgp").count() > 1)
        .collect();
    assert!(
        multi.len() >= 40,
        "only {} objects carry more than one operator record",
        multi.len()
    );
    let total: usize = grouped
        .values()
        .map(|records| records.iter().filter(|r| r.origin == "supgp").count())
        .sum();
    assert!(
        total > grouped.len(),
        "{total} operator records for {} objects",
        grouped.len()
    );

    // And every object has exactly one catalogue record to compare against — several would mean the
    // generator paired an object with someone else's element set.
    for ((source, key), records) in &grouped {
        let catalogue = records.iter().filter(|r| r.origin == "gp").count();
        assert_eq!(
            catalogue, 1,
            "{source} {key}: {catalogue} catalogue records"
        );
    }
}
