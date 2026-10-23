//! Time-variable gravity — the secular drift of the low-degree zonal geopotential coefficients (IERS
//! Conventions 2010, §6.2, Table 6.2).
//!
//! A static gravity field (the embedded XGM2019e in [`crate::geopotential`]) is a snapshot: Earth's
//! oblateness actually drifts, dominated by post-glacial rebound, so `C̄₂₀` (hence `J₂`) changes
//! measurably over a mission. The IERS conventions capture this with a linear secular rate on the
//! degree-2, -3 and -4 zonals:
//!
//! ```text
//! C̄ₙ₀(t) = C̄ₙ₀(t₀) + (dC̄ₙ₀/dt)·(t − t₀),   t₀ = J2000.0
//! ```
//!
//! with the conventional zero-tide values and rates of Table 6.2 (`C̄₂₀` after Cheng et al. 2010, rate
//! after Nerem et al. 1993; `C̄₃₀`/`C̄₄₀` from EGM2008, rates from Cheng et al. 1997). Add
//! [`secular_zonal_correction`] to the mean `C̄₂₀`, `C̄₃₀`, `C̄₄₀` at the propagation epoch; the rate is
//! tide-system independent, so the correction applies whatever tide convention the base field uses.
//!
//! Those constants are the *conventional* IERS field. The **full** GRACE/GRACE-FO time-variable field
//! — every degree/order carrying a (piecewise-linear) trend and annual/semiannual terms — is handled
//! by [`TimeVariableField`] and [`parse_icgem_time_variable`], which read the ICGEM `gfc` format
//! (`gfct`/`trnd`/`acos`/`asin`) that models such as EIGEN-6S and ITSG-Grace are distributed in. Those
//! models express the low-degree zonals as a *sequence* of annual `gfct`/`trnd` segments (the
//! coefficient bends over the mission, it does not follow one global line) plus global periodic terms;
//! [`TimeVariableCoefficient`] carries both. The full model file is a large external download (ICGEM,
//! `icgem.gfz-potsdam.de`), not vendored; only a small real EIGEN-6S2 excerpt is kept as a fixture.

// The Stokes-coefficient symbols (C̄ₙ₀) and the IERS primary-source values are quoted as printed; the
// ICGEM parser casts small in-range integers from the YYYYMMDD epoch.
#![allow(
    clippy::unreadable_literal,
    clippy::doc_markdown,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names
)]

use alloc::vec::Vec;

/// Reference epoch of the secular model — J2000.0, as a decimal year. The rates are per Julian year.
pub const REFERENCE_YEAR: f64 = 2000.0;

/// IERS 2010 conventional secular rates `dC̄ₙ₀/dt` (per year) for `n = 2, 3, 4` — Table 6.2.
const D_C20_DT: f64 = 11.6e-12;
const D_C30_DT: f64 = 4.9e-12;
const D_C40_DT: f64 = 4.7e-12;

/// IERS 2010 conventional mean coefficients `C̄ₙ₀(J2000.0)` — Table 6.2 (`C̄₂₀` is the zero-tide value).
const C20_J2000: f64 = -0.48416948e-3;
const C30_J2000: f64 = 0.9571612e-6;
const C40_J2000: f64 = 0.5399659e-6;

/// The additive secular corrections `[ΔC̄₂₀, ΔC̄₃₀, ΔC̄₄₀]` to the mean zonal (normalised) coefficients
/// at `years_since_j2000` decimal years after J2000.0: `ΔC̄ₙ₀ = (dC̄ₙ₀/dt)·(t − t₀)`. Add these to the
/// base field's `C̄₂₀`, `C̄₃₀`, `C̄₄₀` to make it time-variable.
#[must_use]
pub fn secular_zonal_correction(years_since_j2000: f64) -> [f64; 3] {
    [
        D_C20_DT * years_since_j2000,
        D_C30_DT * years_since_j2000,
        D_C40_DT * years_since_j2000,
    ]
}

/// The full IERS conventional zonal coefficients `[C̄₂₀(t), C̄₃₀(t), C̄₄₀(t)]` at `years_since_j2000`,
/// i.e. the Table 6.2 mean values plus the secular drift. `C̄₂₀` is zero-tide.
#[must_use]
pub fn conventional_zonal_coefficients(years_since_j2000: f64) -> [f64; 3] {
    let d = secular_zonal_correction(years_since_j2000);
    [C20_J2000 + d[0], C30_J2000 + d[1], C40_J2000 + d[2]]
}

/// One piecewise-linear segment of a time-variable coefficient, valid on `[t0_year, t1_year)`
/// (decimal years): `C̄(t) = c_ref + trend_c·(t − t0_year)` (and `S̄`). ICGEM temporal models express
/// the secular part as a *sequence* of these — `gfct` sets the reference `C̄(t₀)`, `trnd` the slope —
/// so a coefficient bends over time (the annual GRACE segments) instead of following one global line.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeSegment {
    /// Segment start `t₀` (decimal year) — also the trend's reference epoch.
    pub t0_year: f64,
    /// Segment end `t₁` (decimal year); `+∞` if the file gave no upper bound.
    pub t1_year: f64,
    /// Reference `C̄ₙₘ(t₀)`.
    pub c_ref: f64,
    /// Reference `S̄ₙₘ(t₀)`.
    pub s_ref: f64,
    /// Secular trend `dC̄ₙₘ/dt` (per year).
    pub trend_c: f64,
    /// Secular trend `dS̄ₙₘ/dt` (per year).
    pub trend_s: f64,
}

/// A periodic (annual `T = 1 yr`, semiannual `T = 0.5 yr`, …) variation of one Stokes coefficient,
/// valid on `[t0_year, t1_year]` with the phase referenced to `t0_year`:
/// `cos_c·cos(2π(t − t₀)/T) + sin_c·sin(2π(t − t₀)/T)` (and `S̄`). ICGEM gives the cosine amplitude on
/// an `acos` line and the sine amplitude on an `asin` line sharing the same period and epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicTerm {
    /// Phase reference `t₀` (decimal year).
    pub t0_year: f64,
    /// End of validity `t₁` (decimal year); `+∞` if unbounded.
    pub t1_year: f64,
    /// Period `T` (years).
    pub period_years: f64,
    /// Cosine amplitude of `C̄ₙₘ`.
    pub cos_c: f64,
    /// Cosine amplitude of `S̄ₙₘ`.
    pub cos_s: f64,
    /// Sine amplitude of `C̄ₙₘ`.
    pub sin_c: f64,
    /// Sine amplitude of `S̄ₙₘ`.
    pub sin_s: f64,
}

/// One time-variable normalised Stokes coefficient `(n, m)`: a piecewise-linear secular part
/// (`segments`, from `gfct`/`trnd`) plus periodic terms (`periodic`, from `acos`/`asin`). `C̄ₙₘ(t)` is
/// the active segment's reference-plus-trend, with every periodic term active at `t` added on top.
#[derive(Clone, Debug, PartialEq)]
pub struct TimeVariableCoefficient {
    /// Spherical-harmonic degree `n`.
    pub degree: usize,
    /// Spherical-harmonic order `m`.
    pub order: usize,
    /// Piecewise-linear secular segments, in file (chronological) order.
    pub segments: Vec<TimeSegment>,
    /// Periodic terms.
    pub periodic: Vec<PeriodicTerm>,
}

/// Epoch-matching tolerance (years) for attaching a `trnd`/`acos`/`asin` to the `gfct` it shares an
/// epoch with — the shared epoch strings are byte-identical, so any sub-day tolerance suffices.
const EPOCH_TOL: f64 = 1e-6;

impl TimeVariableCoefficient {
    /// The active segment at `year`: the one whose `[t₀, t₁)` brackets it; failing that (outside the
    /// model span, or a gap) the latest segment with `t₀ ≤ year`, or the first if `year` precedes them
    /// all — i.e. the end segments extrapolate the model beyond its span.
    fn segment_at(&self, year: f64) -> Option<&TimeSegment> {
        if let Some(s) = self
            .segments
            .iter()
            .find(|s| year >= s.t0_year && year < s.t1_year)
        {
            return Some(s);
        }
        if let Some(first) = self.segments.first()
            && year < first.t0_year
        {
            return Some(first);
        }
        self.segments.iter().rfind(|s| s.t0_year <= year)
    }

    /// Evaluate `(C̄ₙₘ(t), S̄ₙₘ(t))` at decimal `year`.
    #[must_use]
    pub fn evaluate(&self, year: f64) -> (f64, f64) {
        let (mut c, mut s) = self.segment_at(year).map_or((0.0, 0.0), |seg| {
            let dt = year - seg.t0_year;
            (seg.c_ref + seg.trend_c * dt, seg.s_ref + seg.trend_s * dt)
        });
        for term in &self.periodic {
            if year >= term.t0_year && year <= term.t1_year {
                let arg = core::f64::consts::TAU * (year - term.t0_year) / term.period_years;
                let (co, si) = (libm::cos(arg), libm::sin(arg));
                c += term.cos_c * co + term.sin_c * si;
                s += term.cos_s * co + term.sin_s * si;
            }
        }
        (c, s)
    }
}

/// A full time-variable gravity field: the Stokes coefficients that carry a trend and/or periodic
/// variation. GRACE/GRACE-FO products (EIGEN-6S, ITSG-Grace) express these in the ICGEM `gfc` format;
/// [`parse_icgem_time_variable`] reads them. The static base field is separate
/// ([`crate::geopotential`]); this is only the time-variable part, added to the mean coefficients at
/// the propagation epoch.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TimeVariableField {
    /// The time-variable coefficients, in file order.
    pub coefficients: Vec<TimeVariableCoefficient>,
}

impl TimeVariableField {
    /// The time-variable `(C̄ₙₘ(t), S̄ₙₘ(t))` at decimal `year`, or `None` if `(n, m)` has no
    /// time-variable entry (its coefficient is then purely the static base field).
    #[must_use]
    pub fn coefficient(&self, degree: usize, order: usize, year: f64) -> Option<(f64, f64)> {
        self.coefficients
            .iter()
            .find(|c| c.degree == degree && c.order == order)
            .map(|c| c.evaluate(year))
    }
}

/// `YYYYMMDD[.frac]` → decimal year, `year + (days_elapsed + sub_day)/days_in_year`, using the real
/// calendar (leap-aware day-of-year and 365/366-day years) so it matches the standard ICGEM epoch
/// convention. The `.frac` is a fraction of a day (e.g. `19850109.1751`).
fn decimal_year(yyyymmdd: f64) -> f64 {
    const CUM: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let whole = yyyymmdd as i64;
    let frac_day = yyyymmdd - whole as f64;
    let (year, month, day) = (whole / 10000, (whole / 100) % 100, whole % 100);
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let mut days_elapsed = CUM[(month - 1).clamp(0, 11) as usize] + (day - 1).max(0);
    if leap && month > 2 {
        days_elapsed += 1;
    }
    let days_in_year = if leap { 366.0 } else { 365.0 };
    year as f64 + (days_elapsed as f64 + frac_day) / days_in_year
}

/// Index of the `(degree, order)` coefficient, creating an empty one at the end if absent.
fn coeff_index(field: &mut TimeVariableField, degree: usize, order: usize) -> usize {
    if let Some(i) = field
        .coefficients
        .iter()
        .position(|c| c.degree == degree && c.order == order)
    {
        i
    } else {
        field.coefficients.push(TimeVariableCoefficient {
            degree,
            order,
            segments: Vec::new(),
            periodic: Vec::new(),
        });
        field.coefficients.len() - 1
    }
}

/// Parse the time-variable part of an ICGEM `gfc` gravity-field file: the `gfct` (piecewise reference
/// `C̄(t₀)`), `trnd`/`dot` (secular slope), and `acos`/`asin` (periodic cosine/sine amplitude) lines,
/// grouped by degree/order. The columns are `key L M C S σC σS t0 t1 [period]` (epochs `yyyymmdd`,
/// period in years on the periodic lines). A `trnd`/`acos`/`asin` binds to the `gfct` segment it shares
/// an epoch with (byte-identical `t0`), so file ordering does not matter. Static `gfc` and header lines
/// are ignored — this returns only the time-variable part. Models such as EIGEN-6S and ITSG-Grace use
/// exactly this layout, including the piecewise annual `gfct`/`trnd` segments of the low-degree zonals.
#[must_use]
pub fn parse_icgem_time_variable(text: &str) -> TimeVariableField {
    let mut field = TimeVariableField::default();
    let num = |s: &str| s.replace(['D', 'd'], "E").parse::<f64>().ok();
    let date = |s: &str| s.parse::<f64>().ok().map(decimal_year);
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 5 {
            continue;
        }
        let (Ok(n), Ok(m)) = (f[1].parse::<usize>(), f[2].parse::<usize>()) else {
            continue;
        };
        let kind = f[0];
        if !matches!(kind, "gfct" | "trnd" | "dot" | "acos" | "asin") {
            continue;
        }
        let t0 = f.get(7).copied().and_then(date);
        let t1 = f.get(8).copied().and_then(date);
        let ci = coeff_index(&mut field, n, m);
        let coeff = &mut field.coefficients[ci];
        match kind {
            "gfct" => coeff.segments.push(TimeSegment {
                t0_year: t0.unwrap_or(REFERENCE_YEAR),
                t1_year: t1.unwrap_or(f64::INFINITY),
                c_ref: num(f[3]).unwrap_or(0.0),
                s_ref: num(f[4]).unwrap_or(0.0),
                trend_c: 0.0,
                trend_s: 0.0,
            }),
            "trnd" | "dot" => {
                let (tc, ts) = (num(f[3]).unwrap_or(0.0), num(f[4]).unwrap_or(0.0));
                let idx = t0
                    .and_then(|e| {
                        coeff
                            .segments
                            .iter()
                            .position(|s| (s.t0_year - e).abs() < EPOCH_TOL)
                    })
                    .or_else(|| coeff.segments.len().checked_sub(1));
                if let Some(i) = idx {
                    coeff.segments[i].trend_c = tc;
                    coeff.segments[i].trend_s = ts;
                }
            }
            _ => {
                // acos / asin: cosine or sine amplitude of a periodic term keyed by (period, epoch).
                let period = f.last().and_then(|s| num(s)).unwrap_or(0.0);
                let (amp_c, amp_s) = (num(f[3]).unwrap_or(0.0), num(f[4]).unwrap_or(0.0));
                let (pt0, pt1) = (t0.unwrap_or(REFERENCE_YEAR), t1.unwrap_or(f64::INFINITY));
                let idx = coeff
                    .periodic
                    .iter()
                    .position(|t| {
                        (t.period_years - period).abs() < 1e-9
                            && (t.t0_year - pt0).abs() < EPOCH_TOL
                    })
                    .unwrap_or_else(|| {
                        coeff.periodic.push(PeriodicTerm {
                            t0_year: pt0,
                            t1_year: pt1,
                            period_years: period,
                            cos_c: 0.0,
                            cos_s: 0.0,
                            sin_c: 0.0,
                            sin_s: 0.0,
                        });
                        coeff.periodic.len() - 1
                    });
                if kind == "acos" {
                    coeff.periodic[idx].cos_c = amp_c;
                    coeff.periodic[idx].cos_s = amp_s;
                } else {
                    coeff.periodic[idx].sin_c = amp_c;
                    coeff.periodic[idx].sin_s = amp_s;
                }
            }
        }
    }
    field
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unreadable_literal, clippy::float_cmp, clippy::unwrap_used)]

    use super::*;

    // An ICGEM time-variable snippet in the real column layout `key L M C S σC σS t0 t1 [period]`.
    // (2,0) is piecewise: two annual `gfct`/`trnd` segments plus a global annual and semiannual term,
    // exactly the structure the low-degree zonals of EIGEN-6S / ITSG-Grace use. Values are chosen for
    // a clean hand-check, not physical. The static `gfc` line is the base field and is ignored.
    const ICGEM_FIXTURE: &str = "\
gfc    2    0  -0.484000000000e-03   0.0
gfct   2    0  -0.484000000000e-03   0.0   1.0e-12  0.0   20000101.0000   20100101.0000
trnd   2    0   1.160000000000e-11   0.0   1.0e-13  0.0   20000101.0000   20100101.0000
gfct   2    0  -0.483900000000e-03   0.0   1.0e-12  0.0   20100101.0000   20200101.0000
trnd   2    0   5.000000000000e-11   0.0   1.0e-13  0.0   20100101.0000   20200101.0000
acos   2    0   1.000000000000e-11   0.0   1.0e-13  0.0   20000101.0000   20500101.0000   1.0
asin   2    0   2.000000000000e-11   0.0   1.0e-13  0.0   20000101.0000   20500101.0000   1.0
acos   2    0   3.000000000000e-11   0.0   1.0e-13  0.0   20000101.0000   20500101.0000   0.5
gfct   2    1  -0.200000000000e-09   0.140000000000e-08  1.0e-12  1.0e-12  20000101.0000  20500101.0000
asin   2    1   5.000000000000e-11   7.000000000000e-11  1.0e-13  1.0e-13  20000101.0000  20500101.0000  1.0
";

    #[test]
    fn icgem_piecewise_parse_and_evaluate_matches_the_hand_built_series() {
        let field = parse_icgem_time_variable(ICGEM_FIXTURE);
        assert_eq!(field.coefficients.len(), 2); // (2,0) and (2,1); the static `gfc` line is ignored
        let c20_coeff = &field.coefficients[0];
        assert_eq!(c20_coeff.segments.len(), 2); // two piecewise-linear secular segments
        assert_eq!(c20_coeff.periodic.len(), 2); // annual + semiannual

        // C̄₂₀ at 2000.25 → first segment (dt 0.25 yr); annual arg π/2 (cos 0, sin 1), semiannual arg
        // π (cos −1). The periodic phase is referenced to the term's own t₀ = 2000.0.
        let (c20a, s20a) = field.coefficient(2, 0, 2000.25).unwrap();
        let expect_a = -0.484e-3 + 1.16e-11 * 0.25 + 2.0e-11 - 3.0e-11;
        assert!(
            (c20a - expect_a).abs() < 1e-18,
            "C20(2000.25) {c20a} vs {expect_a}"
        );
        assert!(s20a.abs() < 1e-18);

        // C̄₂₀ at 2015.0 → the *second* segment (dt 5 yr from its t₀ = 2010.0), proving piecewise
        // selection; both periodics complete whole cycles (cos 1, sin 0) at 15 yr from 2000.0.
        let (c20b, _) = field.coefficient(2, 0, 2015.0).unwrap();
        let expect_b = -0.4839e-3 + 5.0e-11 * 5.0 + 1.0e-11 + 3.0e-11;
        assert!(
            (c20b - expect_b).abs() < 1e-18,
            "C20(2015) {c20b} vs {expect_b}"
        );

        // (2,1) carries only an annual sine term; at its epoch (dt 0, sin 0) it is the reference.
        let (c21, s21) = field.coefficient(2, 1, 2000.0).unwrap();
        assert!((c21 - (-0.2e-9)).abs() < 1e-18 && (s21 - 1.4e-9).abs() < 1e-18);

        // A coefficient with no time-variable entry returns None (it is purely the static field).
        assert!(field.coefficient(4, 4, 2010.0).is_none());
    }

    #[test]
    fn secular_only_reduces_to_the_linear_trend() {
        // A single gfct + trnd segment with no periodic terms is exactly the IERS-style linear
        // secular formula. 2000 is a leap year, so 20000101 → 2000.0 exactly.
        let field = parse_icgem_time_variable(
            "gfct 3 0 0.9571612e-06 0.0 0.0 0.0 20000101.0 20500101.0\n\
             trnd 3 0 4.9e-12 0.0 0.0 0.0 20000101.0 20500101.0\n",
        );
        let (c30, _) = field.coefficient(3, 0, 2025.0).unwrap(); // dt = 25 yr from 2000.0
        assert!((c30 - (0.9571612e-06 + 4.9e-12 * 25.0)).abs() < 1e-18);
    }

    fn flat_segment(t0_year: f64, t1_year: f64, c_ref: f64, s_ref: f64) -> TimeSegment {
        TimeSegment {
            t0_year,
            t1_year,
            c_ref,
            s_ref,
            trend_c: 0.0,
            trend_s: 0.0,
        }
    }

    #[test]
    fn a_segment_boundary_year_belongs_to_the_later_segment() {
        // `[t₀, t₁)` is half-open: at exactly 2010.0 the second segment has taken over, so its own
        // reference applies with zero elapsed time — not the first segment run out to its end. Both
        // periodic terms complete whole cycles in the 10 yr since their epoch (cos 1, sin 0).
        let field = parse_icgem_time_variable(ICGEM_FIXTURE);
        let (c20, _) = field.coefficient(2, 0, 2010.0).unwrap();
        let expect = -0.4839e-3 + 1.0e-11 + 3.0e-11;
        assert!((c20 - expect).abs() < 1e-20, "C20(2010) {c20} vs {expect}");
    }

    #[test]
    fn the_end_segments_extrapolate_outside_the_model_span() {
        let field = parse_icgem_time_variable(ICGEM_FIXTURE);
        // Five years before the model starts: the first segment's line, run backwards. The periodic
        // terms are valid from 2000.0 on and contribute nothing here.
        let (before, _) = field.coefficient(2, 0, 1995.0).unwrap();
        let expect_before = -0.484e-3 - 1.16e-11 * 5.0;
        assert!(
            (before - expect_before).abs() < 1e-20,
            "C20(1995) {before} vs {expect_before}"
        );
        // Five years after it ends: the last segment's line, run forwards from its own t₀ = 2010.0,
        // and now both periodic terms are in — their window runs to 2050 — at whole cycles again.
        let (after, _) = field.coefficient(2, 0, 2025.0).unwrap();
        let expect_after = -0.4839e-3 + 5.0e-11 * 15.0 + 1.0e-11 + 3.0e-11;
        assert!(
            (after - expect_after).abs() < 1e-20,
            "C20(2025) {after} vs {expect_after}"
        );
    }

    #[test]
    fn the_bracketing_segment_wins_over_a_later_listed_one_that_has_expired() {
        // A short segment nested inside a long one. Only the long one is valid at 2015, even though
        // the nested one starts before 2015 and comes later in the list.
        let coeff = TimeVariableCoefficient {
            degree: 2,
            order: 0,
            segments: vec![
                flat_segment(2000.0, 2020.0, 1.0e-9, 0.0),
                flat_segment(2010.0, 2012.0, 2.0e-9, 0.0),
            ],
            periodic: Vec::new(),
        };
        assert_eq!(coeff.evaluate(2015.0), (1.0e-9, 0.0));
    }

    #[test]
    fn an_empty_segment_is_never_selected() {
        // `[t₀, t₁)` with `t₁ == t₀` holds no year at all, so even 2000.0 falls through it to the
        // fallback — the last listed segment that has started.
        let coeff = TimeVariableCoefficient {
            degree: 2,
            order: 0,
            segments: vec![
                flat_segment(2000.0, 2000.0, 1.0e-9, 0.0),
                flat_segment(1990.0, 1995.0, 2.0e-9, 0.0),
            ],
            periodic: Vec::new(),
        };
        assert_eq!(coeff.evaluate(2000.0), (2.0e-9, 0.0));
    }

    #[test]
    fn a_periodic_term_carries_its_own_phase_into_both_channels() {
        // One term sampled 1 yr after its epoch, at a phase of exactly 60° where cos = 1/2 and
        // sin = √3/2, so the four amplitudes enter with four different weights. The 6 yr period is
        // deliberate: with the annual 1 yr and an integer epoch, `year − t₀` and `year + t₀` differ
        // by a whole number of cycles and the phase reference cannot be seen at all.
        const COS_60: f64 = 0.5;
        const SIN_60: f64 = 0.8660254037844386; // √3/2
        let coeff = TimeVariableCoefficient {
            degree: 3,
            order: 1,
            segments: vec![flat_segment(2000.0, 2050.0, 1.0e-9, 2.0e-9)],
            periodic: vec![PeriodicTerm {
                t0_year: 2000.0,
                t1_year: 2050.0,
                period_years: 6.0,
                cos_c: 1.0e-10,
                cos_s: 3.0e-10,
                sin_c: 5.0e-10,
                sin_s: 7.0e-10,
            }],
        };
        let (c, s) = coeff.evaluate(2001.0);
        let expect_c = 1.0e-9 + 1.0e-10 * COS_60 + 5.0e-10 * SIN_60;
        let expect_s = 2.0e-9 + 3.0e-10 * COS_60 + 7.0e-10 * SIN_60;
        assert!((c - expect_c).abs() < 1e-22, "C {c} vs {expect_c}");
        assert!((s - expect_s).abs() < 1e-22, "S {s} vs {expect_s}");
    }

    #[test]
    fn a_five_column_line_parses_with_default_epochs() {
        // `key L M C S` is the shortest line the format allows — no sigmas, no epochs. The segment
        // then starts at J2000.0 and never ends, and the trend, having no epoch to match on, binds
        // to the segment already there.
        let field = parse_icgem_time_variable("gfct 3 1 1.0e-9 2.0e-9\ntrnd 3 1 3.0e-11 4.0e-11\n");
        let seg = field.coefficients[0].segments[0];
        assert_eq!((seg.t0_year, seg.t1_year), (REFERENCE_YEAR, f64::INFINITY));
        let (c, s) = field.coefficient(3, 1, 2010.0).unwrap(); // dt = 10 yr from J2000.0
        assert!((c - 1.3e-9).abs() < 1e-21, "C {c}");
        assert!((s - 2.4e-9).abs() < 1e-21, "S {s}");
    }

    #[test]
    fn a_trend_binds_to_the_segment_sharing_its_epoch_whatever_the_line_order() {
        // Both segments are declared before either trend, and the trends then arrive newest first.
        // Each must still land on the segment whose epoch it repeats, not on the last one pushed.
        let field = parse_icgem_time_variable(
            "gfct 4 0 1.0e-9 0.0 0.0 0.0 20000101.0 20100101.0\n\
             gfct 4 0 2.0e-9 0.0 0.0 0.0 20100101.0 20200101.0\n\
             trnd 4 0 7.0e-11 0.0 0.0 0.0 20100101.0 20200101.0\n\
             trnd 4 0 3.0e-11 0.0 0.0 0.0 20000101.0 20100101.0\n",
        );
        let (early, _) = field.coefficient(4, 0, 2005.0).unwrap();
        let (late, _) = field.coefficient(4, 0, 2015.0).unwrap();
        assert!((early - (1.0e-9 + 3.0e-11 * 5.0)).abs() < 1e-21, "{early}");
        assert!((late - (2.0e-9 + 7.0e-11 * 5.0)).abs() < 1e-21, "{late}");
    }

    #[test]
    fn decimal_year_follows_the_gregorian_calendar() {
        // `(yyyymmdd, year, days elapsed before the date, days in that year)`, the day counts taken
        // off the calendar: 2000 is a leap year by the 400 rule, 1900 is not one by the 100 rule,
        // 2004 is one by the 4 rule and 2003 is not.
        let cases: [(f64, f64, f64, f64); 7] = [
            (20000215.0, 2000.0, 45.0, 366.0), // 31 + 14; February is before the leap day
            (20000301.0, 2000.0, 60.0, 366.0), // 31 + 29; March is after it
            (19000301.0, 1900.0, 59.0, 365.0), // 31 + 28
            (20040301.0, 2004.0, 60.0, 366.0),
            (20030301.0, 2003.0, 59.0, 365.0),
            (20030704.0, 2003.0, 184.0, 365.0), // 181 through June, then 3
            (20001231.0, 2000.0, 365.0, 366.0), // the 366th day of a leap year
        ];
        for (yyyymmdd, year, elapsed, length) in cases {
            let expect = year + elapsed / length;
            let got = decimal_year(yyyymmdd);
            assert!((got - expect).abs() < 1e-12, "{yyyymmdd} {got} vs {expect}");
        }
        // The fraction is a fraction of a day, as in the models' `19850109.1751`. Packing it into
        // the same double as the date costs it precision — the spacing at 1.985e7 is 3.7e-9 of a
        // day — so this one is good to ~5e-12 yr, not to the last bit.
        let got = decimal_year(19850109.1751);
        let expect = 1985.0 + (8.0 + 0.1751) / 365.0;
        assert!((got - expect).abs() < 1e-9, "{got} vs {expect}");
    }

    // A verbatim excerpt of the real EIGEN-6S2 model (ICGEM) — the (2,0) and (3,0) time-variable
    // lines only; the full degree-260 model is the external ICGEM download.
    const EIGEN6S2_EXCERPT: &str =
        include_str!("../tests/fixtures/eigen6s2_lowdeg_timevariable.gfc");

    #[test]
    fn parses_real_eigen6s2_excerpt_and_matches_independent_reference() {
        let field = parse_icgem_time_variable(EIGEN6S2_EXCERPT);
        assert_eq!(field.coefficients.len(), 2); // (2,0) and (3,0)
        let c20 = field
            .coefficients
            .iter()
            .find(|c| c.degree == 2 && c.order == 0)
            .unwrap();
        assert_eq!(c20.segments.len(), 29); // the real piecewise annual segments
        assert_eq!(c20.periodic.len(), 2); // annual + semiannual
        assert!(
            c20.periodic
                .iter()
                .any(|p| (p.period_years - 1.0).abs() < 1e-9)
        );
        assert!(
            c20.periodic
                .iter()
                .any(|p| (p.period_years - 0.5).abs() < 1e-9)
        );

        // C̄₂₀ at three epochs falling in three *different* piecewise segments (a purely secular model
        // would miss two of these), plus C̄₃₀. The expected values were cross-checked this commit
        // against an independent Python reference of the ICGEM temporal formula — using pyshtools'
        // calendar epoch convention and reading the same real model — with exact (0.0) agreement.
        let cases: [(usize, usize, f64, f64); 4] = [
            (2, 0, 1995.4520547945206, -0.0004841651998743561),
            (2, 0, 2005.495890410959, -0.0004841652646102638),
            (2, 0, 2012.6666666666667, -0.00048416547843533117),
            (3, 0, 2008.2021857923498, 9.572638769560007e-07),
        ];
        for (n, m, year, expect) in cases {
            let (c, _) = field.coefficient(n, m, year).unwrap();
            assert!(
                (c - expect).abs() < 1e-16,
                "C({n},{m},{year}) {c} vs {expect}"
            );
        }
    }

    #[test]
    fn secular_correction_is_the_rate_times_elapsed_years() {
        let d = secular_zonal_correction(25.0); // ~2025
        assert!((d[0] - 11.6e-12 * 25.0).abs() < 1e-24);
        assert!((d[1] - 4.9e-12 * 25.0).abs() < 1e-24);
        assert!((d[2] - 4.7e-12 * 25.0).abs() < 1e-24);
        // Zero drift at the reference epoch.
        assert_eq!(secular_zonal_correction(0.0), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn conventional_coefficients_match_table_6_2_at_j2000() {
        let c = conventional_zonal_coefficients(0.0);
        assert!((c[0] - (-0.48416948e-3)).abs() < 1e-18, "C20 {}", c[0]);
        assert!((c[1] - 0.9571612e-6).abs() < 1e-18, "C30 {}", c[1]);
        assert!((c[2] - 0.5399659e-6).abs() < 1e-18, "C40 {}", c[2]);
    }

    #[test]
    fn c20_drifts_toward_zero_reducing_oblateness() {
        // C̄₂₀ is negative and the rate is positive, so |C̄₂₀| (and J₂) shrinks over time — the
        // post-glacial-rebound signature the conventional model encodes.
        let c2000 = conventional_zonal_coefficients(0.0)[0];
        let c2025 = conventional_zonal_coefficients(25.0)[0];
        assert!(c2025 > c2000 && c2025.abs() < c2000.abs());
        // The drift itself (taken directly, without the cancelling C̄₂₀ subtraction) is ~2.9e-10 / 25 yr.
        assert!((secular_zonal_correction(25.0)[0] - 2.9e-10).abs() < 1e-24);
    }

    #[test]
    fn conventional_coefficients_carry_the_drift_on_every_zonal() {
        // Table 6.2 mean plus 25 yr of rate, summed by hand: rates 11.6e-12, 4.9e-12 and 4.7e-12
        // per year give 2.900e-10, 1.225e-10 and 1.175e-10 on top of -0.48416948e-3, 0.9571612e-6
        // and 0.5399659e-6.
        let c = conventional_zonal_coefficients(25.0);
        assert!((c[0] - (-4.8416919e-4)).abs() < 1e-20, "C20 {}", c[0]);
        assert!((c[1] - 9.572837e-7).abs() < 1e-20, "C30 {}", c[1]);
        assert!((c[2] - 5.400834e-7).abs() < 1e-20, "C40 {}", c[2]);
    }
}
