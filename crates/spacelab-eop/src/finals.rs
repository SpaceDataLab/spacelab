//! Parsing of the IERS `finals2000A` series.
//!
//! # Format
//!
//! Fixed-width, 187 columns, one row per day. The column offsets below were not
//! taken from memory: they were verified against the data itself. Where both
//! bulletins are present, their pole positions must agree — after 2000 they do, to
//! `5.6e-4` arcsec. A single-character misalignment would put that disagreement at
//! order `0.1`. That check runs as a test.
//!
//! Rows near the end of the file carry only a date and an MJD: the IERS has
//! reserved the day but has nothing to say about it yet. They are skipped.

use core::fmt;
use core::ops::Range;

use spacelab_core::{Angle, ReferenceFrame};

/// Zero-based, half-open column ranges of the `finals2000A` record.
mod col {
    use core::ops::Range;

    pub const MJD: Range<usize> = 7..15;

    pub const FLAG_POLE: Range<usize> = 16..17;
    pub const XP: Range<usize> = 18..27;
    pub const XP_SIGMA: Range<usize> = 27..36;
    pub const YP: Range<usize> = 37..46;
    pub const YP_SIGMA: Range<usize> = 46..55;

    pub const DUT1: Range<usize> = 58..68;
    pub const DUT1_SIGMA: Range<usize> = 68..78;

    /// Excess length of day, milliseconds. Bulletin A only — the combined solution
    /// does not republish it.
    pub const LOD: Range<usize> = 79..86;

    pub const DX: Range<usize> = 97..106;
    pub const DY: Range<usize> = 116..125;

    pub const B_XP: Range<usize> = 134..144;
    pub const B_YP: Range<usize> = 144..154;
    pub const B_DUT1: Range<usize> = 154..165;
    pub const B_DX: Range<usize> = 165..175;
    pub const B_DY: Range<usize> = 175..185;
}

/// Whether a value was measured or predicted.
///
/// Predicted `ΔUT1` a month ahead carries millisecond error. That is metres of
/// longitude, so this cannot be a footnote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EopQuality {
    /// Determined from observation.
    Final,
    /// Extrapolated by the IERS. Error grows with the horizon.
    Predicted,
}

impl EopQuality {
    /// The weaker of two qualities. Interpolating across a boundary yields a
    /// prediction, not a measurement.
    #[must_use]
    pub fn worst(self, other: Self) -> Self {
        if self == Self::Final && other == Self::Final {
            Self::Final
        } else {
            Self::Predicted
        }
    }
}

/// Earth orientation at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eop {
    /// Position of the Celestial Intermediate Pole along the IERS meridian.
    pub xp: Angle,
    /// Position of the pole, ninety degrees west.
    pub yp: Angle,
    /// `UT1 - UTC`, in seconds. Kept below 0.9 s by leap seconds.
    pub dut1_seconds: f64,
    /// Excess length of day, in seconds: how much the day exceeded 86400 SI seconds.
    /// Bulletin A only. The instantaneous Earth-rotation rate is `ω₀·(1 − LOD/86400)`.
    pub lod_seconds: f64,
    /// Celestial pole offset from the IAU 2000A model, `dX`.
    pub dx: Angle,
    /// Celestial pole offset from the IAU 2000A model, `dY`.
    pub dy: Angle,
    /// Whether these values were observed or predicted.
    pub quality: EopQuality,
    /// Reported uncertainty on `xp`.
    pub xp_sigma: Angle,
    /// Reported uncertainty on `yp`.
    pub yp_sigma: Angle,
    /// Reported uncertainty on `dut1_seconds`, in seconds.
    pub dut1_sigma_seconds: f64,
}

/// One tabulated day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EopSample {
    /// Modified Julian Date, at 0h UTC.
    pub mjd: f64,
    /// The rapid-service (Bulletin A) values, always present. A C04 series has no rapid service,
    /// and there this repeats `bulletin_b`.
    pub bulletin_a: Eop,
    /// The final combined (Bulletin B) solution, where the IERS has published one. Every row of a
    /// C04 series is that solution.
    pub bulletin_b: Option<Eop>,
}

impl EopSample {
    /// Bulletin B if it exists, otherwise Bulletin A.
    ///
    /// B is the final combined solution. A is the rapid service. Reading only A —
    /// which is what a naive column-slice does — silently prefers the provisional
    /// answer to the definitive one.
    #[must_use]
    pub fn best(&self) -> Eop {
        self.bulletin_b.unwrap_or(self.bulletin_a)
    }
}

/// Why an EOP record could not be read, or a query answered.
#[derive(Debug, Clone, PartialEq)]
pub enum EopError {
    /// A field did not parse as a number.
    Field {
        /// One-based line number in the source.
        line: usize,
        /// The column range at fault.
        columns: Range<usize>,
    },
    /// The series held no usable rows.
    Empty,
    /// A C04 header line the records depend on is missing, or names what this reader cannot use.
    Header {
        /// One-based line number: the offending header line, or the first record when the line
        /// never appeared.
        line: usize,
        /// The header line the reader needed.
        expected: &'static str,
    },
    /// Rows were not in ascending order of MJD.
    NotMonotonic {
        /// One-based line number where the order broke.
        line: usize,
    },
    /// The requested date falls in a hole in the series.
    ///
    /// `finals2000A` is tabulated every day. A gap means the file was truncated or
    /// spliced, and interpolating across it would quietly invent an Earth
    /// orientation for days nobody observed.
    Gap {
        /// Last tabulated day before the hole.
        after: f64,
        /// First tabulated day after the hole.
        before: f64,
    },
    /// The requested date lies outside the tabulated span.
    OutOfRange {
        /// What was asked for.
        mjd: f64,
        /// The first tabulated day.
        first: f64,
        /// The last tabulated day.
        last: f64,
    },
}

impl fmt::Display for EopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Field { line, columns } => {
                write!(
                    f,
                    "line {line}: columns {}..{} are not a number",
                    columns.start, columns.end
                )
            }
            Self::Empty => write!(f, "no usable rows"),
            Self::Header { line, expected } => write!(f, "line {line}: expected `{expected}`"),
            Self::Gap { after, before } => {
                write!(f, "no tabulated day between MJD {after} and {before}")
            }
            Self::NotMonotonic { line } => write!(f, "line {line}: MJD went backwards"),
            Self::OutOfRange { mjd, first, last } => {
                write!(
                    f,
                    "MJD {mjd} lies outside the tabulated span [{first}, {last}]"
                )
            }
        }
    }
}

impl core::error::Error for EopError {}

/// A parsed `finals2000A` or C04 series.
#[derive(Debug, Clone, PartialEq)]
pub struct EopSeries {
    pub(crate) samples: Vec<EopSample>,
    pub(crate) frame: ReferenceFrame,
}

/// Read a fixed-width field, returning `None` when it is blank.
fn field(line: &str, columns: Range<usize>) -> Option<&str> {
    let text = line.get(columns)?.trim();
    (!text.is_empty()).then_some(text)
}

/// Read a required numeric field.
pub(crate) fn number(line: &str, columns: Range<usize>, line_no: usize) -> Result<f64, EopError> {
    field(line, columns.clone())
        .and_then(|t| t.parse().ok())
        .ok_or(EopError::Field {
            line: line_no,
            columns,
        })
}

/// Read an optional numeric field. A blank is absence, not an error.
fn optional(line: &str, columns: Range<usize>) -> Option<f64> {
    field(line, columns).and_then(|t| t.parse().ok())
}

/// Degree-3 Lagrange interpolation of `ys` at `x`, over the four nodes `xs`.
fn lagrange4(xs: [f64; 4], ys: [f64; 4], x: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..4 {
        let mut term = ys[i];
        for j in 0..4 {
            if i != j {
                term *= (x - xs[j]) / (xs[i] - xs[j]);
            }
        }
        acc += term;
    }
    acc
}

/// Remove one-second leap jumps from four consecutive `ΔUT1` values, anchored so the value at
/// `base` is unchanged — expressing all four in `base`'s `ΔAT`, the convention that holds across
/// the query interval. `ΔUT1 − ΔAT` is the quantity that is continuous at a leap second.
fn destep_dut1(d: &mut [f64; 4], base: usize) {
    for k in (base + 1)..4 {
        let step = d[k] - d[k - 1];
        if step.abs() > 0.5 {
            let s = step.signum();
            d[k..].iter_mut().for_each(|v| *v -= s);
        }
    }
    for k in (0..base).rev() {
        let step = d[k] - d[k + 1];
        if step.abs() > 0.5 {
            let s = step.signum();
            d[..=k].iter_mut().for_each(|v| *v -= s);
        }
    }
}

impl EopSeries {
    /// Parse the contents of a `finals2000A.all` file.
    ///
    /// Rows carrying only a date and an MJD — days the IERS has reserved but not
    /// yet filled — are skipped rather than reported. Everything else must parse.
    ///
    /// # Errors
    ///
    /// [`EopError::Field`] on a malformed number, [`EopError::NotMonotonic`] if the
    /// days are out of order, [`EopError::Empty`] if nothing usable remains.
    pub fn parse_finals2000a(text: &str) -> Result<Self, EopError> {
        let mut samples: Vec<EopSample> = Vec::new();

        for (index, line) in text.lines().enumerate() {
            let line_no = index + 1;

            // A reserved-but-empty day. The pole flag is the sentinel: without it
            // there is no data, whatever else the row may contain.
            let Some(flag) = field(line, col::FLAG_POLE) else {
                continue;
            };
            let quality = match flag {
                "I" => EopQuality::Final,
                _ => EopQuality::Predicted,
            };

            let mjd = number(line, col::MJD, line_no)?;

            let bulletin_a = Eop {
                xp: Angle::from_arcseconds(number(line, col::XP, line_no)?),
                yp: Angle::from_arcseconds(number(line, col::YP, line_no)?),
                dut1_seconds: number(line, col::DUT1, line_no)?,
                // LOD is published in milliseconds; a blank early-series LOD is zero.
                lod_seconds: optional(line, col::LOD).unwrap_or(0.0) / 1000.0,
                dx: Angle::from_arcseconds(optional(line, col::DX).unwrap_or(0.0) / 1000.0),
                dy: Angle::from_arcseconds(optional(line, col::DY).unwrap_or(0.0) / 1000.0),
                quality,
                xp_sigma: Angle::from_arcseconds(number(line, col::XP_SIGMA, line_no)?),
                yp_sigma: Angle::from_arcseconds(number(line, col::YP_SIGMA, line_no)?),
                dut1_sigma_seconds: number(line, col::DUT1_SIGMA, line_no)?,
            };

            // Bulletin B is all-or-nothing: the IERS publishes the combined
            // solution as a unit, so a partial row would mean we misread a column.
            let bulletin_b = match (
                optional(line, col::B_XP),
                optional(line, col::B_YP),
                optional(line, col::B_DUT1),
            ) {
                (Some(xp), Some(yp), Some(dut1)) => Some(Eop {
                    xp: Angle::from_arcseconds(xp),
                    yp: Angle::from_arcseconds(yp),
                    dut1_seconds: dut1,
                    dx: Angle::from_arcseconds(optional(line, col::B_DX).unwrap_or(0.0) / 1000.0),
                    dy: Angle::from_arcseconds(optional(line, col::B_DY).unwrap_or(0.0) / 1000.0),
                    // Bulletin B exists only for days the IERS has finalised.
                    quality: EopQuality::Final,
                    ..bulletin_a
                }),
                _ => None,
            };

            if samples.last().is_some_and(|prev| mjd <= prev.mjd) {
                return Err(EopError::NotMonotonic { line: line_no });
            }
            samples.push(EopSample {
                mjd,
                bulletin_a,
                bulletin_b,
            });
        }

        if samples.is_empty() {
            return Err(EopError::Empty);
        }
        Ok(Self {
            samples,
            frame: ReferenceFrame::Unstated,
        })
    }

    /// The terrestrial frame the series states its pole coordinates are in.
    ///
    /// [`ReferenceFrame::Unstated`] for `finals2000A`, which has no header to state one. A C04
    /// series names its ITRF on its second line, and that ITRF holds for every row: the IERS
    /// reprocessed the whole series, so it is a property of the file's vintage, not of a row's
    /// epoch.
    #[must_use]
    pub const fn frame(&self) -> ReferenceFrame {
        self.frame
    }

    /// Every tabulated day, in ascending order.
    #[must_use]
    pub fn samples(&self) -> &[EopSample] {
        &self.samples
    }

    /// The first and last tabulated MJD.
    #[must_use]
    pub fn span(&self) -> (f64, f64) {
        let first = self.samples.first().map_or(f64::NAN, |s| s.mjd);
        let last = self.samples.last().map_or(f64::NAN, |s| s.mjd);
        (first, last)
    }

    /// Earth orientation at an arbitrary UTC instant, by four-point Lagrange interpolation
    /// (the IERS `interp` scheme), falling back to two-point linear at the series edges.
    ///
    /// The result's [`EopQuality`] is the weaker of the two bracketing days: an interval that
    /// touches a prediction yields a prediction. This is the base series only; the diurnal and
    /// semidiurnal terms live in [`crate::subdaily`] and are added by the caller when they matter.
    ///
    /// # Errors
    ///
    /// [`EopError::OutOfRange`] outside the tabulated span. We do not extrapolate
    /// the rotation of the Earth.
    pub fn at_mjd(&self, mjd: f64) -> Result<Eop, EopError> {
        let (first, last) = self.span();
        if !(first..=last).contains(&mjd) {
            return Err(EopError::OutOfRange { mjd, first, last });
        }

        // Days are consecutive, so the index is arithmetic rather than a search —
        // but do not assume it: verify, and fall back to a binary search.
        let index = match self
            .samples
            .binary_search_by(|s| s.mjd.partial_cmp(&mjd).unwrap_or(core::cmp::Ordering::Less))
        {
            Ok(exact) => return Ok(self.samples[exact].best()),
            Err(insert) => insert.saturating_sub(1),
        };

        let lo = &self.samples[index];
        let hi = &self.samples[index + 1];

        // The series is daily. A wider interval is a hole, and interpolation across a hole is
        // an invention.
        if hi.mjd - lo.mjd > 1.5 {
            return Err(EopError::Gap {
                after: lo.mjd,
                before: hi.mjd,
            });
        }

        let t = (mjd - lo.mjd) / (hi.mjd - lo.mjd);
        let lerp = |x: f64, y: f64| x + (y - x) * t;

        // Four-point Lagrange when a full, gap-free window [index-1 ..= index+2] exists; the
        // smooth pole, rotation and pole offsets gain from the higher order. `ΔUT1` is
        // discontinuous at a leap second (a one-second step on a tabulated midnight), so its four
        // values are de-stepped into the query interval's `ΔAT` first — interpolating across the
        // raw step would smear a whole second (~230 m). The reported uncertainties stay linear:
        // an interpolated σ must not overshoot below zero.
        let n = self.samples.len();
        let window = (index >= 1 && index + 2 < n).then(|| {
            let s = [&self.samples[index - 1], lo, hi, &self.samples[index + 2]];
            (s[1].mjd - s[0].mjd <= 1.5 && s[3].mjd - s[2].mjd <= 1.5).then_some(s)
        });

        if let Some(Some(s)) = window {
            let xs = [s[0].mjd, s[1].mjd, s[2].mjd, s[3].mjd];
            let e = [s[0].best(), s[1].best(), s[2].best(), s[3].best()];
            let lag =
                |g: fn(&Eop) -> f64| lagrange4(xs, [g(&e[0]), g(&e[1]), g(&e[2]), g(&e[3])], mjd);
            let mut d = [
                e[0].dut1_seconds,
                e[1].dut1_seconds,
                e[2].dut1_seconds,
                e[3].dut1_seconds,
            ];
            destep_dut1(&mut d, 1);
            Ok(Eop {
                xp: Angle::from_radians(lag(|p| p.xp.radians())),
                yp: Angle::from_radians(lag(|p| p.yp.radians())),
                dut1_seconds: lagrange4(xs, d, mjd),
                lod_seconds: lag(|p| p.lod_seconds),
                dx: Angle::from_radians(lag(|p| p.dx.radians())),
                dy: Angle::from_radians(lag(|p| p.dy.radians())),
                quality: e[1].quality.worst(e[2].quality),
                xp_sigma: Angle::from_radians(lerp(
                    e[1].xp_sigma.radians(),
                    e[2].xp_sigma.radians(),
                )),
                yp_sigma: Angle::from_radians(lerp(
                    e[1].yp_sigma.radians(),
                    e[2].yp_sigma.radians(),
                )),
                dut1_sigma_seconds: lerp(e[1].dut1_sigma_seconds, e[2].dut1_sigma_seconds),
            })
        } else {
            let (a, b) = (lo.best(), hi.best());
            let hi_dut1 = {
                let step = b.dut1_seconds - a.dut1_seconds;
                if step.abs() > 0.5 {
                    b.dut1_seconds - step.signum()
                } else {
                    b.dut1_seconds
                }
            };
            Ok(Eop {
                xp: Angle::from_radians(lerp(a.xp.radians(), b.xp.radians())),
                yp: Angle::from_radians(lerp(a.yp.radians(), b.yp.radians())),
                dut1_seconds: lerp(a.dut1_seconds, hi_dut1),
                lod_seconds: lerp(a.lod_seconds, b.lod_seconds),
                dx: Angle::from_radians(lerp(a.dx.radians(), b.dx.radians())),
                dy: Angle::from_radians(lerp(a.dy.radians(), b.dy.radians())),
                quality: a.quality.worst(b.quality),
                xp_sigma: Angle::from_radians(lerp(a.xp_sigma.radians(), b.xp_sigma.radians())),
                yp_sigma: Angle::from_radians(lerp(a.yp_sigma.radians(), b.yp_sigma.radians())),
                dut1_sigma_seconds: lerp(a.dut1_sigma_seconds, b.dut1_sigma_seconds),
            })
        }
    }
}

#[cfg(test)]
mod interp_tests {
    use super::{destep_dut1, lagrange4};

    #[test]
    fn lagrange4_reproduces_a_line_and_a_cubic_exactly() {
        let xs = [10.0, 11.0, 12.0, 13.0];
        let line = xs.map(|x| 3.0 * x + 1.0);
        assert!((lagrange4(xs, line, 11.5) - (3.0 * 11.5 + 1.0)).abs() < 1e-9);
        // Degree-3 interpolation is exact for a cubic.
        let cubic = xs.map(|x: f64| 2.0 * x.powi(3) - 5.0 * x * x + x - 7.0);
        let want = 2.0 * 11.5_f64.powi(3) - 5.0 * 11.5 * 11.5 + 11.5 - 7.0;
        assert!((lagrange4(xs, cubic, 11.5) - want).abs() < 1e-6);
        // At a node it returns the node value.
        assert!((lagrange4(xs, line, 12.0) - line[2]).abs() < 1e-9);
    }

    #[test]
    fn destep_dut1_removes_a_leap_jump_anchored_at_the_base() {
        // A +1 s jump between indices 1 and 2; anchored at base 1, [0,1] stay, [2,3] are undone.
        let mut d = [-0.3, -0.4, 0.6, 0.5];
        destep_dut1(&mut d, 1);
        assert!((d[0] - -0.3).abs() < 1e-12 && (d[1] - -0.4).abs() < 1e-12);
        assert!((d[2] - -0.4).abs() < 1e-12 && (d[3] - -0.5).abs() < 1e-12);
    }

    #[test]
    fn destep_dut1_undoes_a_jump_below_the_base_by_moving_the_earlier_days() {
        // A +1 s jump between indices 0 and 1 — the leap fell on the window's first midnight, so
        // it is the backward sweep that has to undo it, and only day 0 may move. The test above
        // puts its jump above the base, which leaves this whole half of the function unexercised:
        // reading the wrong neighbour, or starting the forward sweep one day early, both keep the
        // sequence continuous there while shifting the anchored day by a whole second.
        let mut d = [0.4, -0.6, -0.61, -0.62];
        destep_dut1(&mut d, 1);
        assert!((d[0] - -0.6).abs() < 1e-12, "day 0 = {}", d[0]);
        assert!(
            (d[1] - -0.6).abs() < 1e-12,
            "the base must not move: {}",
            d[1]
        );
        assert!((d[2] - -0.61).abs() < 1e-12 && (d[3] - -0.62).abs() < 1e-12);
    }
}
