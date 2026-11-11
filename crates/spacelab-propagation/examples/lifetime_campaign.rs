//! One worker of `tools/lifetime-campaign.py`: fits a ballistic coefficient to an observed decay, then
//! runs each driver scenario of an object through [`spacelab_propagation::lifetime::lifetime`].
//!
//! Usage: `lifetime_campaign SW-All.csv [tolerance] < jobs`. A job is one tab-separated line:
//!
//! ```text
//! id  model  t_h  a e i raan argp M  t0  a e i raan argp M  horizon_s  members
//! ```
//!
//! Elements are mean, metres and radians; times are Unix seconds. The first state starts the fit, the
//! second the prediction; `t_h` written `B=value` gives the coefficient instead, and the first state is
//! not read. `model` is `sdtm` or `msis`. `members` is `;`-separated: `O` observed, `P` persistence of
//! the epoch, `Q:days` persistence of that long before it, `A:shift_days:ratio:tau_s`, or `T:step_s` for a trace of `a` under the measured indices.
//! The output is `id member B seconds`, with `seconds` empty for a survivor and `nan` for a run that
//! broke down; a trace prints `id T:seconds B a`.
//!
//! `lifetime_campaign --conformal alpha < scores` prints the conformal radius of one score per line, or
//! `inf` when the calibration set is too small for that `alpha`.

#![allow(clippy::expect_used)]

use std::io::BufRead;

use spacelab_propagation::Tolerance;
use spacelab_propagation::dsst::drag::DsstAtmosphericDrag;
use spacelab_propagation::dsst::zonal::DsstZonal;
use spacelab_propagation::lifetime::{
    DEFAULT_REENTRY_ALTITUDE, DecayModel, Lifetime, lifetime, propagate_decay,
};
use spacelab_propagation::lifetime_drivers::{
    DriverHistory, SCENARIO_TOLERANCE, Scenario, ScenarioAtmosphere, conformal_radius,
    decay_envelope,
};
use spacelab_propagation::sdtm::Envelope;
use spacelab_propagation::sdtm::coefficients::{ENVELOPE, FIELD, HARMONICS, HISTORY};
use spacelab_propagation::space_weather::parse_space_weather;

/// EGM2008, tide-free: `earth_gravity_constant`, `radius` and the fully normalised `C̄ₙ₀` of degrees
/// 2–6 from its `egm2008.dat` header and table, as printed.
const MU: f64 = 3.986_004_415e14;
const RE: f64 = 6_378_136.3;
const C_BAR: [f64; 5] = [
    -0.484_165_143_790_815e-3,
    0.957_161_207_093_473e-6,
    0.539_965_866_638_991e-6,
    0.686_702_913_736_681e-7,
    -0.149_953_927_978_527e-6,
];
const BODY_RADIUS: f64 = 6_378_137.0;
/// The drivers' own grid: a scan stops there anyway.
const CHUNK: f64 = 10_800.0;
/// The fit's stop: the miss in the mean `a`, m.
const FIT_MISS: f64 = 0.1;
/// The fit's other stop: the width of its bracket in `ln B`.
const FIT_WIDTH: f64 = 7.0e-5;
/// The most propagations one fit may take; halving alone would reach `FIT_WIDTH` in 18.
const FIT_PROPAGATIONS: usize = 24;
/// Ten minutes: a GCAT decay time with no hour in it is good to half a day.
const TOLERANCE: f64 = 600.0;
static DECAY_ENVELOPE: Envelope = decay_envelope(&ENVELOPE);

fn equinoctial(elements: &[f64]) -> [f64; 6] {
    let [a, e, i, raan, argp, mean_anomaly] = elements else {
        panic!("six elements");
    };
    let longitude = raan + argp;
    let half = (i / 2.0).tan();
    [
        *a,
        e * longitude.cos(),
        e * longitude.sin(),
        half * raan.cos(),
        half * raan.sin(),
        mean_anomaly + longitude,
    ]
}

struct Runner<'a> {
    zonal: &'a DsstZonal,
    history: &'a DriverHistory,
    use_field: bool,
    tolerance: Tolerance,
}

impl Runner<'_> {
    fn model(&self, b: f64) -> DecayModel<'_> {
        DecayModel {
            zonal: self.zonal,
            drag: DsstAtmosphericDrag::new(1.0, b, MU),
            mu: MU,
            body_radius: BODY_RADIUS,
            reentry_altitude: DEFAULT_REENTRY_ALTITUDE,
        }
    }

    fn atmosphere(&self, scenario: Scenario, epoch: f64) -> ScenarioAtmosphere<'_, HARMONICS> {
        ScenarioAtmosphere {
            history: self.history,
            scenario,
            epoch,
            field: self.use_field.then_some((&FIELD, &DECAY_ENVELOPE)),
            tolerance: self.tolerance,
        }
    }

    /// How far `b` takes the mean `a` down from `start` within `span` under the measured indices.
    fn fall(&self, b: f64, start: [f64; 6], epoch: f64, span: f64) -> Option<f64> {
        let atmosphere = self.atmosphere(Scenario::Observed, epoch);
        propagate_decay(&self.model(b), &atmosphere, start, 0.0, span)
            .ok()
            .map(|end| start[0] - end[0])
    }

    /// Prints the mean `a` every `step` under the measured indices, as `id T:seconds B a`.
    fn trace(&self, cells: &[&str], b: f64, start: [f64; 6], epoch: f64, span: f64, step: f64) {
        let model = self.model(b);
        let atmosphere = self.atmosphere(Scenario::Observed, epoch);
        let (mut t, mut y) = (0.0, start);
        println!("{}\tT:{t}\t{b:e}\t{}", cells[0], y[0]);
        while t < span {
            let next = (t + step).min(span);
            let Ok(end) = propagate_decay(&model, &atmosphere, y, t, next) else {
                return;
            };
            (t, y) = (next, end);
            println!("{}\tT:{t}\t{b:e}\t{}", cells[0], y[0]);
        }
    }

    /// The `B` in `[1e-4, 1]` that takes `start` to the observed `target` semi-major axis.
    ///
    /// The fall is nearly proportional to `B`, so a secant in `(ln B, ln fall)` needs a handful of
    /// propagations, each as long as the fit window; bisection took twenty. It is kept inside a
    /// bracket, and a step leaving the bracket halves it instead. `FIT_MISS` is far inside the noise
    /// of a semi-major axis read from element sets.
    fn fit(&self, start: [f64; 6], epoch: f64, span: f64, target: f64) -> Option<f64> {
        let wanted = start[0] - target;
        if wanted <= 0.0 {
            return None;
        }
        let (bottom, top) = (libm::log(1.0e-4), 0.0);
        let (mut lo, mut hi) = (bottom, top);
        let mut x = libm::log(0.01);
        let mut last: Option<(f64, f64)> = None;
        for _ in 0..FIT_PROPAGATIONS {
            let Some(fall) = self.fall(libm::exp(x), start, epoch, span) else {
                hi = x;
                x = f64::midpoint(lo, hi);
                last = None;
                continue;
            };
            if (fall - wanted).abs() < FIT_MISS {
                return Some(libm::exp(x));
            }
            if fall > wanted {
                hi = x;
            } else {
                lo = x;
            }
            if hi - lo < FIT_WIDTH {
                break;
            }
            let y = libm::log(fall.max(FIT_MISS));
            let slope = last.map_or(1.0, |(x0, y0)| (y - y0) / (x - x0));
            let next = x + (libm::log(wanted) - y) / slope;
            last = Some((x, y));
            x = if next > lo && next < hi {
                next
            } else {
                f64::midpoint(lo, hi)
            };
        }
        // A bracket still on an end of the range never held the answer.
        (lo > bottom && hi < top && hi - lo < FIT_WIDTH).then(|| libm::exp(f64::midpoint(lo, hi)))
    }
}

fn scenario(spec: &str, runner: &Runner<'_>, epoch: f64) -> Option<Scenario> {
    match spec.split(':').collect::<Vec<_>>().as_slice() {
        ["O"] => Some(Scenario::Observed),
        ["P"] => Some(Scenario::Persistence(runner.history.state(epoch))),
        ["Q", days] => {
            let days: f64 = days.parse().ok()?;
            Some(Scenario::Persistence(
                runner.history.state(epoch - days * 86_400.0),
            ))
        }
        ["A", shift, ratio, tau] => Some(Scenario::Analog {
            shift_days: shift.parse().ok()?,
            ratio: ratio.parse().ok()?,
            tau: tau.parse().ok()?,
        }),
        _ => None,
    }
}

fn conformal(alpha: f64) {
    let scores: Vec<f64> = std::io::stdin()
        .lock()
        .lines()
        .map(|line| line.expect("a score").parse().expect("a number"))
        .collect();
    println!(
        "{}",
        conformal_radius(&scores, alpha).unwrap_or(f64::INFINITY)
    );
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: lifetime_campaign SW-All.csv");
    if path == "--conformal" {
        let alpha = std::env::args().nth(2).expect("an alpha");
        conformal(alpha.parse().expect("a number"));
        return;
    }
    // A second argument tightens the tolerance, for the convergence check.
    let tolerance = std::env::args().nth(2).map_or(SCENARIO_TOLERANCE, |text| {
        let value = text.parse().expect("a tolerance");
        Tolerance::new(value, value)
    });
    let text = std::fs::read_to_string(path).expect("the space-weather file");
    let days = parse_space_weather(&text);
    let history =
        DriverHistory::from_days(&days, FIELD.storm(), HISTORY).expect("a contiguous feed");
    let j: Vec<f64> = (2..)
        .zip(C_BAR)
        .map(|(n, c): (u32, f64)| -f64::from(2 * n + 1).sqrt() * c)
        .collect();
    let zonal = DsstZonal::from_j(MU, RE, &j);
    for line in std::io::stdin().lock().lines() {
        let line = line.expect("a job");
        let cells: Vec<&str> = line.split('\t').collect();
        let number = |k: usize| cells[k].parse::<f64>().expect("a number");
        let elements = |k: usize| (k..k + 6).map(number).collect::<Vec<f64>>();
        let runner = Runner {
            zonal: &zonal,
            history: &history,
            use_field: cells[1] == "sdtm",
            tolerance,
        };
        let t_0 = number(9);
        let initial = equinoctial(&elements(10));
        let horizon = number(16);
        let fitted = cells[2].strip_prefix("B=").map_or_else(
            || {
                let t_h = number(2);
                runner.fit(equinoctial(&elements(3)), t_h, t_0 - t_h, initial[0])
            },
            |given| given.parse().ok(),
        );
        let Some(b) = fitted else {
            println!("{}\tfit\t\tnan", cells[0]);
            continue;
        };
        for (index, spec) in cells[17].split(';').enumerate() {
            if let Some(step) = spec.strip_prefix("T:") {
                runner.trace(
                    &cells,
                    b,
                    initial,
                    t_0,
                    horizon,
                    step.parse().expect("a step"),
                );
                continue;
            }
            let scenario = scenario(spec, &runner, t_0).expect("a member");
            let atmosphere = runner.atmosphere(scenario, t_0);
            let seconds = match lifetime(
                &runner.model(b),
                &atmosphere,
                initial,
                horizon,
                CHUNK,
                TOLERANCE,
            ) {
                Ok(Lifetime::Reentry { seconds, .. }) => seconds.to_string(),
                Ok(Lifetime::Survives { .. }) => String::new(),
                Ok(Lifetime::Unresolved { .. }) | Err(_) => "nan".to_string(),
            };
            println!("{}\t{index}\t{b:e}\t{seconds}", cells[0]);
        }
    }
}
