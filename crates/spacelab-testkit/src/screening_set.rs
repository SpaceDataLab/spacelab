//! A synthetic all-against-all screening set whose answer key is exact, and the score of a screener
//! on it (plan 18 Ф21.O, decision D-47).
//!
//! A conjunction screen is judged by what it misses, and a miss can only be counted against a key
//! that is known to be complete. A catalogue of real objects has no such key: its "truth" is another
//! code's output. This set is built so that its key is mathematics. Every object is a circular
//! two-body orbit, and every pair whose radii are within the threshold is one of three geometries
//! whose closest approaches have a closed form:
//!
//! * **coplanar**, two radii: the minima are the alignments, at distance `|r_b − r_a|`, at
//!   `t_k = (2πk − Δu₀)/(n_b − n_a)`;
//! * **crossing**, one radius and two planes meeting at `γ`: two minima per revolution, at the
//!   mutual node, at distance `2r cos(γ/2) |sin(δ/2)|`, where `δ` is the phase between the two
//!   objects measured from that node;
//! * **comoving**, one plane and one radius: the separation is constant and has no minimum.
//!
//! Every other pair has radii further apart than the threshold, so the triangle inequality keeps it
//! apart by at least that much at every epoch. The key is therefore the whole truth: every local
//! minimum of the separation below the threshold, over every pair, inside the window.
//!
//! # The set
//!
//! 100 objects, a one-day window `[0, 86 400]` s, a 10 km threshold, `μ = 3.986004418e14` m³/s²:
//!
//! * a Walker 40/5/0 shell at 6 928 km and 55°, whose 780 pairs never come within 26 km of each
//!   other — a background that the radial filter cannot discard and that never conjuncts;
//! * eleven crossing pairs planted at about 10 m, 100 m, 1 km, 5 km, 9 999 m and 9 999.99 m, and,
//!   as hard negatives, at 10 000.01 m, 10 001 m, 10.1 km, 12 km and 50 km, with relative planes
//!   from 1° to 174°;
//! * six coplanar pairs at radius gaps of 300 m, 1 km, 5 km and 9 999 m, and 10 001 m and 15 km
//!   beyond the threshold, each aligning once inside the window;
//! * 26 background orbits, each at a radius of its own.
//!
//! That is 179 keyed events, 145 hard-negative minima and 797 keyed pairs. The inputs are exact
//! doubles — radii in whole metres, angles in whole multiples of `2⁻³²` rad — so the key is exact for
//! the very numbers a test reads, not for their decimal approximations.
//!
//! # Gate
//!
//! `derivations/screening_set.gp` writes `tests/fixtures/screening_set.txt` from the design at 80
//! digits, checks every listed minimum as a negative-to-positive root of `Δr·Δv` on the Cartesian
//! motion, and finds exactly the listed roots by a 60 s scan polished with `solve()` on every planted
//! pair and every fifth shell pair that is not comoving.
//! `derivations/screening_set.mac` derives the three closed forms and the radial bound
//! symbolically, then re-derives every key from the file's integers at 50 digits, brackets every
//! listed minimum on the Cartesian motion, checks that no event pair is within the threshold at
//! either end of the window and that every pair it does not list is radially separated, and
//! catches four keys planted wrong (G3). The tests below hold this module's `f64` motion to those
//! keys and its scorer to planted defects; `spacelab-estimation/tests/screening_set.rs` scores the
//! workspace's screener on the set.

use alloc::vec::Vec;

/// The committed set, as `derivations/screening_set.gp` wrote it.
pub const SCREENING_SET: &str = include_str!("../tests/fixtures/screening_set.txt");

/// Which part of the set an object belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// The Walker shell: one radius, five planes, never within the threshold of itself.
    Shell,
    /// One object of a planted crossing pair.
    Crossing,
    /// One object of a planted coplanar pair.
    Coplanar,
    /// An orbit alone at its radius.
    Background,
}

/// The geometry of a keyed pair, which decides the closed form of its minima.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairKind {
    /// One plane, two radii.
    Coplanar,
    /// One radius, two planes.
    Crossing,
    /// One plane, one radius: the separation is constant.
    Comoving,
}

/// One object: a circular orbit, its angles in radians.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Object {
    /// The part of the set it was built for.
    pub group: Group,
    /// Orbit radius, m.
    pub radius: f64,
    /// Inclination, rad.
    pub inclination: f64,
    /// Right ascension of the ascending node, rad.
    pub raan: f64,
    /// Argument of latitude at `t = 0`, rad.
    pub u0: f64,
}

/// An object's motion, with its frame evaluated once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CircularOrbit {
    radius: f64,
    mean_motion: f64,
    p: [f64; 3],
    q: [f64; 3],
    u0: f64,
}

impl Object {
    /// The two-body motion of this object about a point mass `mu`, m³/s².
    #[must_use]
    pub fn orbit(&self, mu: f64) -> CircularOrbit {
        let (si, ci) = (libm::sin(self.inclination), libm::cos(self.inclination));
        let (so, co) = (libm::sin(self.raan), libm::cos(self.raan));
        CircularOrbit {
            radius: self.radius,
            mean_motion: libm::sqrt(mu / (self.radius * self.radius * self.radius)),
            p: [co, so, 0.0],
            q: [-so * ci, co * ci, si],
            u0: self.u0,
        }
    }
}

impl CircularOrbit {
    /// Position (m) and velocity (m/s) at `t` seconds: `r (p cos u + q sin u)` and
    /// `r n (−p sin u + q cos u)` with `u = u₀ + n t`.
    #[must_use]
    pub fn state(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        let u = self.u0 + self.mean_motion * t;
        let (sin_u, cos_u) = (libm::sin(u), libm::cos(u));
        let speed = self.radius * self.mean_motion;
        let (p, q) = (self.p, self.q);
        (
            [
                self.radius * (p[0] * cos_u + q[0] * sin_u),
                self.radius * (p[1] * cos_u + q[1] * sin_u),
                self.radius * (p[2] * cos_u + q[2] * sin_u),
            ],
            [
                speed * (q[0] * cos_u - p[0] * sin_u),
                speed * (q[1] * cos_u - p[1] * sin_u),
                speed * (q[2] * cos_u - p[2] * sin_u),
            ],
        )
    }

    /// The mean motion `√(μ/r³)`, rad/s.
    #[must_use]
    pub const fn mean_motion(&self) -> f64 {
        self.mean_motion
    }
}

/// A pair whose radii are within the threshold, or a planted pair, with the least distance it ever
/// reaches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeyedPair {
    /// The lower object index.
    pub a: usize,
    /// The higher object index.
    pub b: usize,
    /// Its geometry.
    pub kind: PairKind,
    /// The least separation over all time, m.
    pub least_distance: f64,
}

/// A local minimum of one pair's separation: a keyed event or hard negative, or an approach a
/// screener reports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Minimum {
    /// One object index.
    pub a: usize,
    /// The other object index.
    pub b: usize,
    /// Time of closest approach, s.
    pub tca: f64,
    /// Separation at `tca`, m.
    pub distance: f64,
}

/// The set: objects, window, threshold and the answer key.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreeningSet {
    /// Gravitational parameter, m³/s².
    pub mu: f64,
    /// Start of the window, s.
    pub start: f64,
    /// End of the window, s.
    pub end: f64,
    /// The screening threshold, m: an event is a minimum below it.
    pub threshold: f64,
    /// Object `k` is `objects[k]`.
    pub objects: Vec<Object>,
    /// Every pair the radial bound cannot separate, and every planted pair.
    pub pairs: Vec<KeyedPair>,
    /// Every local minimum in the open window below the threshold, over all pairs.
    pub events: Vec<Minimum>,
    /// The planted pairs' other minima in the window, at or beyond the threshold.
    pub negatives: Vec<Minimum>,
}

/// A line of the set's text that does not parse, by its 1-based number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// The line's first word names no kind of line.
    UnknownLine(usize),
    /// The line has the wrong number of fields for its kind.
    FieldCount(usize),
    /// A field is not the number, group or kind its position calls for.
    Field(usize),
    /// An object's index is not the next one.
    ObjectOrder(usize),
    /// A pair or minimum names an object that does not exist, or names one object twice.
    ObjectIndex(usize),
    /// A header line is missing or repeated.
    Header,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownLine(n) => write!(f, "line {n}: no such kind of line"),
            Self::FieldCount(n) => write!(f, "line {n}: wrong number of fields"),
            Self::Field(n) => write!(f, "line {n}: a field does not parse"),
            Self::ObjectOrder(n) => write!(f, "line {n}: object out of order"),
            Self::ObjectIndex(n) => write!(f, "line {n}: no such object, or one object twice"),
            Self::Header => write!(f, "mu, window, threshold and angle_exponent, once each"),
        }
    }
}

impl core::error::Error for ParseError {}

/// How close a reported approach must come to a keyed event to recall it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Largest time difference, s.
    pub tca: f64,
    /// Largest distance difference, m.
    pub distance: f64,
}

/// A screener's score on the set.
#[derive(Clone, Debug, PartialEq)]
pub struct Score {
    /// The number of keyed events.
    pub keyed: usize,
    /// Indices into [`ScreeningSet::events`] that no report recalled.
    pub misses: Vec<usize>,
    /// Indices into the reports that recalled no event.
    pub false_alarms: Vec<usize>,
    /// The largest time difference between a recalled event and its report, s.
    pub worst_tca: f64,
    /// The largest distance difference between a recalled event and its report, m.
    pub worst_distance: f64,
}

impl Score {
    /// The number of keyed events a report recalled.
    #[must_use]
    pub fn recalled(&self) -> usize {
        self.keyed - self.misses.len()
    }
}

/// The committed set.
///
/// # Errors
///
/// [`ParseError`] if the committed file does not parse, which its own tests rule out.
pub fn screening_set() -> Result<ScreeningSet, ParseError> {
    ScreeningSet::parse(SCREENING_SET)
}

fn number(field: &str, line: usize) -> Result<f64, ParseError> {
    field.parse::<f64>().map_err(|_| ParseError::Field(line))
}

fn index(field: &str, line: usize) -> Result<usize, ParseError> {
    field.parse::<usize>().map_err(|_| ParseError::Field(line))
}

/// An angle stored as an integer number of `2^exponent` rad. The integer is below `2⁵³` and the
/// scale a power of two, so the product is exact.
fn angle(field: &str, exponent: i32, line: usize) -> Result<f64, ParseError> {
    let units = field.parse::<i64>().map_err(|_| ParseError::Field(line))?;
    if units.unsigned_abs() >= 1 << 53 {
        return Err(ParseError::Field(line));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "below 2^53 on the line above, so exact"
    )]
    let value = units as f64;
    Ok(value * libm::ldexp(1.0, exponent))
}

/// Fill a header slot, refusing a second value.
fn once<T>(slot: &mut Option<T>, value: T) -> Result<(), ParseError> {
    if slot.replace(value).is_some() {
        Err(ParseError::Header)
    } else {
        Ok(())
    }
}

impl ScreeningSet {
    /// Read a set in the format `derivations/screening_set.gp` writes.
    ///
    /// # Errors
    ///
    /// [`ParseError`] naming the first line that does not parse, or [`ParseError::Header`] when a
    /// header value is missing or repeated.
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let (mut mu, mut window, mut threshold, mut exponent) = (None, None, None, None);
        let mut set = Self {
            mu: 0.0,
            start: 0.0,
            end: 0.0,
            threshold: 0.0,
            objects: Vec::new(),
            pairs: Vec::new(),
            events: Vec::new(),
            negatives: Vec::new(),
        };
        for (i, line) in text.lines().enumerate() {
            let n = i + 1;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let w: Vec<&str> = line.split(' ').collect();
            let count = match w[0] {
                "mu" | "threshold" | "angle_exponent" => 2,
                "window" => 3,
                "pair" | "event" | "negative" => 5,
                "object" => 7,
                _ => return Err(ParseError::UnknownLine(n)),
            };
            if w.len() != count {
                return Err(ParseError::FieldCount(n));
            }
            match w[0] {
                "mu" => once(&mut mu, number(w[1], n)?)?,
                "threshold" => once(&mut threshold, number(w[1], n)?)?,
                "window" => once(&mut window, (number(w[1], n)?, number(w[2], n)?))?,
                "angle_exponent" => once(
                    &mut exponent,
                    w[1].parse::<i32>().map_err(|_| ParseError::Field(n))?,
                )?,
                "object" => set.push_object(&w, exponent.ok_or(ParseError::Header)?, n)?,
                "pair" => set.push_pair(&w, n)?,
                _ => set.push_minimum(&w, n)?,
            }
        }
        let (Some(mu), Some((start, end)), Some(threshold), Some(_)) =
            (mu, window, threshold, exponent)
        else {
            return Err(ParseError::Header);
        };
        set.mu = mu;
        set.start = start;
        set.end = end;
        set.threshold = threshold;
        Ok(set)
    }

    fn pair_of(&self, a: &str, b: &str, line: usize) -> Result<(usize, usize), ParseError> {
        let (a, b) = (index(a, line)?, index(b, line)?);
        if a >= b || b >= self.objects.len() {
            return Err(ParseError::ObjectIndex(line));
        }
        Ok((a, b))
    }

    fn push_object(&mut self, w: &[&str], exponent: i32, n: usize) -> Result<(), ParseError> {
        if index(w[1], n)? != self.objects.len() {
            return Err(ParseError::ObjectOrder(n));
        }
        let group = match w[2] {
            "shell" => Group::Shell,
            "crossing" => Group::Crossing,
            "coplanar" => Group::Coplanar,
            "background" => Group::Background,
            _ => return Err(ParseError::Field(n)),
        };
        self.objects.push(Object {
            group,
            radius: number(w[3], n)?,
            inclination: angle(w[4], exponent, n)?,
            raan: angle(w[5], exponent, n)?,
            u0: angle(w[6], exponent, n)?,
        });
        Ok(())
    }

    fn push_pair(&mut self, w: &[&str], n: usize) -> Result<(), ParseError> {
        let (a, b) = self.pair_of(w[1], w[2], n)?;
        let kind = match w[3] {
            "coplanar" => PairKind::Coplanar,
            "crossing" => PairKind::Crossing,
            "comoving" => PairKind::Comoving,
            _ => return Err(ParseError::Field(n)),
        };
        self.pairs.push(KeyedPair {
            a,
            b,
            kind,
            least_distance: number(w[4], n)?,
        });
        Ok(())
    }

    fn push_minimum(&mut self, w: &[&str], n: usize) -> Result<(), ParseError> {
        let (a, b) = self.pair_of(w[1], w[2], n)?;
        let minimum = Minimum {
            a,
            b,
            tca: number(w[3], n)?,
            distance: number(w[4], n)?,
        };
        if w[0] == "event" {
            self.events.push(minimum);
        } else {
            self.negatives.push(minimum);
        }
        Ok(())
    }

    /// Score a screener's reports against the key.
    ///
    /// Each report is an approach the screener claims is below the threshold, objects in either
    /// order. A keyed event is recalled by a report of the same pair within `tolerance` of its time
    /// and of its distance, and each report recalls at most one event, the nearest in time. A
    /// report that recalls none is a false alarm — whether it names a hard negative, a minimum at
    /// the wrong distance or at no minimum at all, or carries a NaN.
    #[must_use]
    pub fn score(&self, reports: &[Minimum], tolerance: Tolerance) -> Score {
        let mut recalled = alloc::vec![false; self.events.len()];
        let mut score = Score {
            keyed: self.events.len(),
            misses: Vec::new(),
            false_alarms: Vec::new(),
            worst_tca: 0.0,
            worst_distance: 0.0,
        };
        for (j, report) in reports.iter().enumerate() {
            let pair = (report.a.min(report.b), report.a.max(report.b));
            let mut best: Option<(usize, f64)> = None;
            for (i, event) in self.events.iter().enumerate() {
                let dt = libm::fabs(event.tca - report.tca);
                let dd = libm::fabs(event.distance - report.distance);
                if !recalled[i]
                    && (event.a, event.b) == pair
                    && dt <= tolerance.tca
                    && dd <= tolerance.distance
                    && best.is_none_or(|(_, best_dt)| dt < best_dt)
                {
                    best = Some((i, dt));
                }
            }
            if let Some((i, dt)) = best {
                recalled[i] = true;
                score.worst_tca = score.worst_tca.max(dt);
                score.worst_distance = score
                    .worst_distance
                    .max(libm::fabs(self.events[i].distance - report.distance));
            } else {
                score.false_alarms.push(j);
            }
        }
        score.misses = (0..self.events.len()).filter(|&i| !recalled[i]).collect();
        score
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::expect_used,
        reason = "a test of the committed file fails loudly"
    )]

    use super::*;
    use alloc::format;
    use alloc::string::String;

    fn set() -> ScreeningSet {
        screening_set().expect("the committed set parses")
    }

    const TIGHT: Tolerance = Tolerance {
        tca: 1e-3,
        distance: 1e-3,
    };

    /// What is planted, the reports, the misses and the false alarms it must score.
    type Case<'a> = (&'a str, &'a [Minimum], &'a [usize], &'a [usize]);

    /// The counts and the header the module documentation states.
    #[test]
    fn the_committed_set_is_the_one_documented() {
        let s = set();
        assert_eq!(
            (
                s.objects.len(),
                s.pairs.len(),
                s.events.len(),
                s.negatives.len()
            ),
            (100, 797, 179, 145)
        );
        assert_eq!(
            (s.mu, s.start, s.end, s.threshold),
            (3.986_004_418e14, 0.0, 86_400.0, 1e4)
        );
        let count = |g| s.objects.iter().filter(|o| o.group == g).count();
        assert_eq!(
            [
                Group::Shell,
                Group::Crossing,
                Group::Coplanar,
                Group::Background
            ]
            .map(count),
            [40, 22, 12, 26]
        );
        assert!(s.events.iter().all(|e| e.distance < s.threshold));
        assert!(s.negatives.iter().all(|e| e.distance >= s.threshold));
    }

    /// The completeness the key rests on, restated in `f64` over all 4 950 pairs: a pair the key
    /// does not list has radii further apart than the threshold, and the radii are whole metres, so
    /// the comparison is exact.
    #[test]
    fn every_pair_the_key_does_not_list_is_radially_separated() {
        let s = set();
        let n = s.objects.len();
        let mut unlisted = 0;
        for a in 0..n {
            for b in a + 1..n {
                if !s.pairs.iter().any(|p| (p.a, p.b) == (a, b)) {
                    unlisted += 1;
                    let gap = libm::fabs(s.objects[a].radius - s.objects[b].radius);
                    assert!(
                        gap > s.threshold,
                        "objects {a} and {b} are {gap} m apart in radius"
                    );
                }
            }
        }
        assert_eq!(unlisted, n * (n - 1) / 2 - s.pairs.len());
    }

    /// The `f64` motion at each keyed minimum (G3): the separation is the key's distance, and the
    /// range rate changes sign from negative to positive across the key's time.
    ///
    /// The motion's own rounding — `u₀ + n t` at `t ~ 10⁵` s carries about `10⁻¹⁴` rad, which at
    /// 7 000 km is `10⁻⁷` m — is what the 1e-6 m allowance is for; the root is bracketed at
    /// ±1 ms, which a 10 m pass at 7 km/s crosses in 7 m of range rate.
    #[test]
    fn the_f64_motion_meets_every_keyed_minimum() {
        let s = set();
        let orbits: Vec<CircularOrbit> = s.objects.iter().map(|o| o.orbit(s.mu)).collect();
        let rel = |m: &Minimum, t: f64| {
            let ((ra, va), (rb, vb)) = (orbits[m.a].state(t), orbits[m.b].state(t));
            let dr = [rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2]];
            let dv = [vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]];
            (
                libm::sqrt(dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]),
                dr[0] * dv[0] + dr[1] * dv[1] + dr[2] * dv[2],
            )
        };
        let mut worst: f64 = 0.0;
        for m in s.events.iter().chain(&s.negatives) {
            let (distance, _) = rel(m, m.tca);
            worst = worst.max(libm::fabs(distance - m.distance));
            assert!(
                rel(m, m.tca - 1e-3).1 < 0.0 && rel(m, m.tca + 1e-3).1 > 0.0,
                "{m:?}: the range rate does not rise through zero at the key"
            );
        }
        assert!(worst < 1e-6, "the f64 separation is {worst} m off a key");
    }

    /// The motion is the stated circle: `|r| = r`, `|v| = r n`, `r · v = 0`, and the velocity is the
    /// derivative of the position (S2).
    #[test]
    fn every_orbit_is_the_circle_it_states() {
        let s = set();
        for o in &s.objects {
            let orbit = o.orbit(s.mu);
            for t in [0.0, 1_234.5, 86_400.0] {
                let (r, v) = orbit.state(t);
                let norm = |x: [f64; 3]| libm::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
                assert!(libm::fabs(norm(r) - o.radius) < 1e-8 * o.radius);
                assert!(libm::fabs(norm(v) - o.radius * orbit.mean_motion()) < 1e-8 * norm(v));
                assert!(
                    libm::fabs(r[0] * v[0] + r[1] * v[1] + r[2] * v[2]) < 1e-8 * norm(r) * norm(v)
                );
                let h = 1e-3;
                let (ahead, behind) = (orbit.state(t + h).0, orbit.state(t - h).0);
                for k in 0..3 {
                    let slope = (ahead[k] - behind[k]) / (2.0 * h);
                    assert!(libm::fabs(slope - v[k]) < 1e-6 * norm(v), "{o:?} at {t} s");
                }
            }
        }
    }

    /// The key scored as its own reports is perfect, in either object order.
    #[test]
    fn the_key_reported_back_scores_every_event_and_no_false_alarm() {
        let s = set();
        let swapped: Vec<Minimum> = s
            .events
            .iter()
            .map(|e| Minimum {
                a: e.b,
                b: e.a,
                ..*e
            })
            .collect();
        for reports in [&s.events, &swapped] {
            let score = s.score(reports, TIGHT);
            assert_eq!((score.recalled(), score.keyed), (179, 179));
            assert!(
                score.misses.is_empty() && score.false_alarms.is_empty(),
                "{score:?}"
            );
            assert_eq!((score.worst_tca, score.worst_distance), (0.0, 0.0));
        }
    }

    /// Rule 16: the scorer fails on planted defects, each as itself.
    ///
    /// A report moved past the tolerance in time or in distance is one miss and one false alarm; a
    /// dropped report is one miss and nothing else; a hard negative or a NaN reported is one false
    /// alarm and nothing else; the same event reported twice recalls it once.
    #[test]
    fn a_planted_wrong_report_is_scored_as_what_it_is() {
        let s = set();
        let base = s.events.clone();
        let mut moved = base.clone();
        moved[17].tca += 2.0 * TIGHT.tca;
        let mut far = base.clone();
        far[17].distance += 2.0 * TIGHT.distance;
        let mut dropped = base.clone();
        dropped.remove(17);
        let mut negative = base.clone();
        negative.push(s.negatives[3]);
        let mut nan = base.clone();
        nan.push(Minimum {
            tca: f64::NAN,
            ..base[5]
        });
        let mut twice = base.clone();
        twice.push(base[5]);
        let cases: [Case; 6] = [
            ("moved in time", &moved, &[17], &[17]),
            ("moved in distance", &far, &[17], &[17]),
            ("dropped", &dropped, &[17], &[]),
            ("a hard negative", &negative, &[], &[179]),
            ("a NaN", &nan, &[], &[179]),
            ("twice", &twice, &[], &[179]),
        ];
        for (what, reports, misses, alarms) in cases {
            let score = s.score(reports, TIGHT);
            assert_eq!(
                (score.misses.as_slice(), score.false_alarms.as_slice()),
                (misses, alarms),
                "{what}"
            );
        }
    }

    /// Rule 23: each malformed line is refused by name, with its line number.
    #[test]
    fn a_malformed_set_is_refused_by_line() {
        let head = "mu 1\nwindow 0 1\nthreshold 1\nangle_exponent -32\n";
        let obj = "object 0 shell 7 0 0 0\nobject 1 shell 7 0 0 1\n";
        let cases = [
            (format!("{head}{obj}bogus 1\n"), ParseError::UnknownLine(7)),
            (
                format!("{head}{obj}pair 0 1 crossing\n"),
                ParseError::FieldCount(7),
            ),
            (
                format!("{head}{obj}pair 0 1 skew 1\n"),
                ParseError::Field(7),
            ),
            (format!("{head}{obj}event 0 x 1 1\n"), ParseError::Field(7)),
            (
                format!("{head}object 1 shell 7 0 0 0\n"),
                ParseError::ObjectOrder(5),
            ),
            (
                format!("{head}object 0 comet 7 0 0 0\n"),
                ParseError::Field(5),
            ),
            (
                format!("{head}{obj}event 1 0 1 1\n"),
                ParseError::ObjectIndex(7),
            ),
            (
                format!("{head}{obj}event 0 2 1 1\n"),
                ParseError::ObjectIndex(7),
            ),
            (
                format!("{head}{obj}event 1 1 1 1\n"),
                ParseError::ObjectIndex(7),
            ),
            (
                format!("{head}object 0 shell 7 9007199254740992 0 0\n"),
                ParseError::Field(5),
            ),
            (format!("{head}mu 2\n"), ParseError::Header),
            (
                String::from("window 0 1\nthreshold 1\nangle_exponent -32\n"),
                ParseError::Header,
            ),
            (String::from("object 0 shell 7 0 0 0\n"), ParseError::Header),
        ];
        for (text, expected) in cases {
            assert_eq!(ScreeningSet::parse(&text), Err(expected), "{text}");
        }
        assert_eq!(
            format!("{}", ParseError::ObjectIndex(7)),
            "line 7: no such object, or one object twice"
        );
    }
}
