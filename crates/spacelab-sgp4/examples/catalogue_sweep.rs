//! Compare this crate's SGP4 against satellite.js over an entire GP catalogue.
//!
//! ```text
//! node tools/sgp4-catalogue-oracle/oracle.cjs data/oracles/satellite-js/satellite.js CAT.tle \
//!   | cargo run --release -p spacelab-sgp4 --example catalogue_sweep -- CAT.tle FIXTURE
//! ```
//!
//! `tools/sgp4-catalogue-oracle/run.sh` is the supported entry point; this is the half that reads
//! the oracle stream.
//!
//! The published verification set, which `tests/sgp4_verification.rs` runs, holds 31 element sets
//! chosen to exercise the branches. That is the right first bar and it is a small one: it says
//! nothing about the 16,000 element sets an operator actually propagates, where the eccentricities,
//! the drag terms and the epoch ages are whatever the catalogue happens to contain. This example
//! raises the bar to the catalogue.
//!
//! What it can and cannot show is the same limit stated on the crate itself: satellite.js is a
//! second transcription of the same Vallado listing, so agreement establishes that both transcribed
//! it correctly and nothing about whether the model is right.
//!
//! The comparison is streamed rather than collected. Nine million samples is not a data structure.
//! What survives the pass is a decade histogram, the worst sample per object, and a bounded set of
//! cases -- the worst overall plus the worst of each [`Regime`], so that the committed fixture
//! cannot end up all near-Earth by accident.

#![allow(clippy::expect_used, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::io::{BufRead, BufWriter, Write};

use spacelab_core::linalg::Vec3;
use spacelab_sgp4::{Convention, Regime, Satellite};

/// Offsets kept in the committed fixture, in minutes from epoch.
///
/// Seven points rather than the whole sweep: the fixture exists so an offline test can re-check the
/// hardest objects without the oracle, not to archive the sweep. Both endpoints and the epoch are
/// included because that is where a resonance integrator's error is largest and smallest.
const FIXTURE_OFFSETS: [i64; 7] = [-1440, -720, -360, 0, 360, 720, 1440];

/// How many worst-residual objects the fixture carries, over and above the per-regime worst.
const WORST_CASES: usize = 48;

/// The catalogue, keyed by columns 3-7 of line 1 verbatim.
///
/// Verbatim rather than decoded, because the key only has to match what the oracle emits. Alpha-5
/// decoding happens once, inside [`Satellite::from_tle`], where it belongs.
type Catalogue = HashMap<String, (String, String)>;

/// One object's contribution to the fixture.
#[derive(Clone)]
struct Case {
    key: String,
    norad_id: u32,
    regime: Regime,
    worst_position_m: f64,
    worst_velocity_m_s: f64,
    line1: String,
    line2: String,
    /// `(minutes, position_km, velocity_km_s)` at the offsets in [`FIXTURE_OFFSETS`].
    samples: Vec<(i64, [f64; 3], [f64; 3])>,
}

/// Counts that describe the sweep itself, as opposed to the residual it measured.
#[derive(Default)]
struct Tally {
    objects: usize,
    our_parse_failed: usize,
    oracle_init_failed: usize,
    init_agreed: usize,
    compared: usize,
    both_failed: usize,
    only_ours_failed: usize,
    only_oracle_failed: usize,
}

/// Decade histogram of the position residual, in metres.
///
/// Percentiles over nine million samples would need either the samples or an approximation with its
/// own error bar. Counting decades needs neither and answers the question that matters -- how much
/// of the catalogue agrees to floating-point noise, and how long the tail is.
#[derive(Default)]
struct Decades {
    /// Bucket `i` counts residuals in `[10^(i-13), 10^(i-12))` metres; bucket 0 is exact zero.
    bins: [u64; 20],
}

impl Decades {
    fn add(&mut self, metres: f64) {
        if metres <= 0.0 {
            self.bins[0] += 1;
            return;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bin = (metres.log10().floor() as i64 + 13).clamp(1, 19) as usize;
        self.bins[bin] += 1;
    }

    /// Every non-empty bucket as `(lower decade exponent, count)`; `None` is the exact-zero bucket.
    fn occupied(&self) -> impl Iterator<Item = (Option<i64>, u64)> {
        self.bins
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(bin, count)| {
                let exponent = i64::try_from(bin).expect("bucket index fits an i64") - 13;
                (if bin == 0 { None } else { Some(exponent) }, *count)
            })
    }
}

/// Everything the streaming pass keeps.
#[derive(Default)]
struct Sweep {
    tally: Tally,
    decades: Decades,
    worst_velocity_m_s: f64,
    /// The worst [`WORST_CASES`] objects overall, most-different first.
    worst: Vec<Case>,
    /// The worst object of each regime, so no branch of the model can drop out of the fixture.
    per_regime: HashMap<&'static str, Case>,
    /// The `#` lines the oracle wrote about itself.
    provenance: Vec<String>,
}

impl Sweep {
    fn record(&mut self, case: Case) {
        let name = regime_name(case.regime);
        match self.per_regime.get(name) {
            Some(held) if held.worst_position_m >= case.worst_position_m => {}
            _ => {
                self.per_regime.insert(name, case.clone());
            }
        }
        self.worst.push(case);
        self.worst.sort_by(|a, b| {
            b.worst_position_m
                .partial_cmp(&a.worst_position_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.worst.truncate(WORST_CASES);
    }

    /// The worst objects plus any regime not already represented among them.
    fn cases(&self) -> Vec<Case> {
        let mut cases = self.worst.clone();
        for case in self.per_regime.values() {
            if !cases.iter().any(|held| held.key == case.key) {
                cases.push(case.clone());
            }
        }
        cases.sort_by(|a, b| {
            b.worst_position_m
                .partial_cmp(&a.worst_position_m)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        cases
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let catalogue_path = args
        .next()
        .expect("usage: catalogue_sweep CATALOGUE_TLE FIXTURE");
    let fixture_path = args
        .next()
        .expect("usage: catalogue_sweep CATALOGUE_TLE FIXTURE");

    let catalogue = read_catalogue(&catalogue_path);
    let sweep = compare(&catalogue);
    let cases = sweep.cases();
    report(&sweep, catalogue.len(), &cases);
    write_fixture(&fixture_path, &sweep, catalogue.len(), &cases);
    println!("{} cases -> {fixture_path}", cases.len());

    // Convention is not a free parameter here: the oracle uses WGS-72 and the AFSPC sidereal
    // expressions, which is what Convention::Reference selects. Naming it keeps the default from
    // silently becoming part of the measurement.
    debug_assert_eq!(Convention::default(), Convention::Reference);
}

/// Read a 2LE or 3LE catalogue into element-set pairs.
fn read_catalogue(path: &str) -> Catalogue {
    let text = std::fs::read_to_string(path).expect("catalogue readable");
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .collect();
    let mut catalogue = Catalogue::new();
    let mut index = 0;
    while index < lines.len() {
        if lines[index].starts_with("1 ")
            && index + 1 < lines.len()
            && lines[index + 1].starts_with("2 ")
        {
            let key = lines[index][2..7].trim().to_owned();
            catalogue.insert(key, (lines[index].to_owned(), lines[index + 1].to_owned()));
            index += 2;
        } else {
            index += 1;
        }
    }
    catalogue
}

/// The object currently being compared, and what its samples have shown so far.
struct Open {
    key: String,
    satellite: Satellite,
    worst_position_m: f64,
    worst_velocity_m_s: f64,
    samples: Vec<(i64, [f64; 3], [f64; 3])>,
}

/// Stream the oracle's answers from stdin and compare each against our own.
fn compare(catalogue: &Catalogue) -> Sweep {
    let mut sweep = Sweep::default();
    let mut open: Option<Open> = None;

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        let read = reader.read_line(&mut line).expect("oracle stream readable");
        let done = read == 0;
        let record = line.trim_end();

        // An object ends at the next object header or at end of stream; both close it the same way.
        let closes = done || record.starts_with('O') || record.starts_with('I');
        if closes && let Some(object) = open.take() {
            sweep.close(object, catalogue);
        }
        if done {
            break;
        }

        if let Some(rest) = record.strip_prefix("# ") {
            sweep.provenance.push(rest.to_owned());
            continue;
        }

        let mut field = record.split_ascii_whitespace();
        match field.next() {
            Some(tag @ ("O" | "I")) => {
                let key = field
                    .next()
                    .expect("object header carries a key")
                    .to_owned();
                open = sweep.begin(tag, key, catalogue);
            }
            Some("E") => {
                if let Some(object) = open.as_ref() {
                    let minutes: f64 = field
                        .next()
                        .expect("offset")
                        .parse()
                        .expect("offset is a number");
                    if object.satellite.propagate_minutes(minutes).is_err() {
                        sweep.tally.both_failed += 1;
                    } else {
                        sweep.tally.only_oracle_failed += 1;
                    }
                }
            }
            Some("S") => {
                if let Some(object) = open.as_mut() {
                    sweep.sample(object, &mut field);
                }
            }
            _ => {}
        }
    }

    sweep
}

impl Sweep {
    /// Start an object, or count why it will not be compared.
    fn begin(&mut self, tag: &str, key: String, catalogue: &Catalogue) -> Option<Open> {
        self.tally.objects += 1;
        let Some((line1, line2)) = catalogue.get(&key) else {
            panic!("oracle emitted object {key}, which the catalogue does not carry");
        };
        if tag == "I" {
            self.tally.oracle_init_failed += 1;
        }
        match Satellite::from_tle(line1, line2) {
            Err(_) => {
                self.tally.our_parse_failed += 1;
                None
            }
            Ok(_) if tag == "I" => None,
            Ok(satellite) => {
                self.tally.init_agreed += 1;
                Some(Open {
                    key,
                    satellite,
                    worst_position_m: 0.0,
                    worst_velocity_m_s: 0.0,
                    samples: Vec::new(),
                })
            }
        }
    }

    /// Compare one sample, and keep it if it sits on one of the offsets the fixture carries.
    fn sample<'a>(&mut self, object: &mut Open, field: &mut impl Iterator<Item = &'a str>) {
        let mut number = || -> f64 {
            field
                .next()
                .expect("sample field")
                .parse()
                .expect("sample field is a number")
        };
        let minutes = number();
        let oracle_position = [number(), number(), number()];
        let oracle_velocity = [number(), number(), number()];

        let Ok(state) = object.satellite.propagate_minutes(minutes) else {
            self.tally.only_ours_failed += 1;
            return;
        };
        self.tally.compared += 1;

        let position_gap = (state.position.vector - km(oracle_position)).norm();
        let velocity_gap = (state.velocity.vector - km(oracle_velocity)).norm();
        self.decades.add(position_gap);
        object.worst_position_m = object.worst_position_m.max(position_gap);
        object.worst_velocity_m_s = object.worst_velocity_m_s.max(velocity_gap);
        self.worst_velocity_m_s = self.worst_velocity_m_s.max(velocity_gap);

        #[allow(clippy::cast_possible_truncation)]
        let whole = minutes as i64;
        if FIXTURE_OFFSETS.contains(&whole) {
            object
                .samples
                .push((whole, oracle_position, oracle_velocity));
        }
    }

    /// Finish an object and offer it to the fixture.
    fn close(&mut self, object: Open, catalogue: &Catalogue) {
        let (line1, line2) = &catalogue[&object.key];
        self.record(Case {
            norad_id: object.satellite.norad_id().get(),
            regime: object.satellite.regime(),
            worst_position_m: object.worst_position_m,
            worst_velocity_m_s: object.worst_velocity_m_s,
            line1: line1.clone(),
            line2: line2.clone(),
            samples: object.samples,
            key: object.key,
        });
    }
}

fn report(sweep: &Sweep, objects: usize, cases: &[Case]) {
    let tally = &sweep.tally;
    let worst_position = cases.first().map_or(0.0, |c| c.worst_position_m);
    println!("objects in catalogue      {objects}");
    println!("objects in oracle stream  {}", tally.objects);
    println!("both accepted the element set {}", tally.init_agreed);
    println!("oracle rejected           {}", tally.oracle_init_failed);
    println!("we rejected               {}", tally.our_parse_failed);
    println!("samples compared          {}", tally.compared);
    println!("both refused to propagate {}", tally.both_failed);
    println!("only we refused           {}", tally.only_ours_failed);
    println!("only the oracle refused   {}", tally.only_oracle_failed);
    println!("worst position            {worst_position:.4e} m");
    println!(
        "worst velocity            {:.4e} m/s",
        sweep.worst_velocity_m_s
    );
    println!("position residual, by decade:");
    for (exponent, count) in sweep.decades.occupied() {
        let share = 100.0 * count as f64 / tally.compared.max(1) as f64;
        match exponent {
            None => println!("  exactly 0        {count:>10}  {share:6.2}%"),
            Some(e) => println!("  1e{:<3} .. 1e{:<3} {count:>10}  {share:6.2}%", e, e + 1),
        }
    }
}

fn write_fixture(path: &str, sweep: &Sweep, objects: usize, cases: &[Case]) {
    let tally = &sweep.tally;
    let worst_position = cases.first().map_or(0.0, |c| c.worst_position_m);
    let file = std::fs::File::create(path).expect("fixture writable");
    let mut out = BufWriter::new(file);
    let mut put = |text: String| writeln!(out, "{text}").expect("fixture written");

    put("# SGP4 across a whole GP catalogue: this crate against satellite.js.".to_owned());
    put(
        "# Written by crates/spacelab-sgp4/examples/catalogue_sweep.rs; do not hand-edit."
            .to_owned(),
    );
    put("# Positions are km and velocities km/s in TEME, as satellite.js returns them.".to_owned());
    for entry in &sweep.provenance {
        put(format!("#: {entry}"));
    }
    put(format!(
        "#: catalogue objects={objects} compared_objects={}",
        tally.init_agreed
    ));
    put(format!(
        "#: samples compared={} both_refused={} only_ours_refused={} only_oracle_refused={}",
        tally.compared, tally.both_failed, tally.only_ours_failed, tally.only_oracle_failed
    ));
    put(format!(
        "#: init oracle_rejected={} we_rejected={}",
        tally.oracle_init_failed, tally.our_parse_failed
    ));
    put(format!(
        "#: worst position_m={worst_position:.6e} velocity_m_s={:.6e}",
        sweep.worst_velocity_m_s
    ));
    for (exponent, count) in sweep.decades.occupied() {
        match exponent {
            None => put(format!("#: decade exactly_zero count={count}")),
            Some(e) => put(format!("#: decade exponent={e} count={count}")),
        }
    }
    for case in cases {
        put(format!(
            "T {} {} {:.6e} {:.6e}",
            case.norad_id,
            regime_name(case.regime),
            case.worst_position_m,
            case.worst_velocity_m_s
        ));
        put(case.line1.clone());
        put(case.line2.clone());
        for (minutes, position, velocity) in &case.samples {
            put(format!(
                "S {minutes} {} {} {} {} {} {}",
                position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]
            ));
        }
    }
    out.flush().expect("fixture flushed");
}

/// satellite.js answers in kilometres; every state in this workspace is in metres.
fn km(v: [f64; 3]) -> Vec3 {
    Vec3::new(v[0] * 1.0e3, v[1] * 1.0e3, v[2] * 1.0e3)
}

fn regime_name(regime: Regime) -> &'static str {
    match regime {
        Regime::NearEarth => "near-earth",
        Regime::DeepSpaceNonResonant => "deep-space",
        Regime::DeepSpaceSynchronous => "synchronous",
        Regime::DeepSpaceHalfDay => "half-day",
    }
}
