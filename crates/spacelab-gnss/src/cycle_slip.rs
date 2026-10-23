//! **Cycle-slip detection** on a single satellite arc.
//!
//! A carrier-phase observable is a range plus an unknown integer number of wavelengths. That integer
//! is constant only while the receiver keeps lock; when lock breaks it jumps, and every phase
//! measurement after the jump is offset by a whole number of cycles. Nothing downstream —
//! ambiguity resolution, PPP, a phase-connected arc — survives an undetected slip, so the arc has to
//! be cut where the integer changed.
//!
//! Both detectors here work on a *test series* the caller forms with [`crate::combinations`], and
//! neither knows or cares which one:
//!
//! * the **geometry-free** combination `L1 − L2` removes geometry, clocks and troposphere, leaving a
//!   slowly varying ionospheric term plus the ambiguity difference. Over seconds the ionosphere
//!   moves by millimetres, so a step of a cycle stands out;
//! * the **Melbourne–Wübbena** combination leaves the wide-lane ambiguity alone, so it is flat over
//!   an arc and steps by whole wide-lane cycles.
//!
//! [`GeometryFreeDetector`] tests the first difference: cheap, and blind to a slip that happens to
//! sit on the extrapolation of the trend. [`PolynomialFitDetector`] fits a low-order polynomial over
//! a window and tests the prediction residual, which follows a curving ionosphere the first
//! difference charges as noise — at the cost of needing a window before it says anything.
//!
//! Neither replaces the receiver's own loss-of-lock indicator
//! ([`crate::rinex_obs::Observation::lost_lock`]); they catch the slips it does not report.

use std::fmt;

/// What a detector concluded about one sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SlipVerdict {
    /// No test was made, and the arc starts (or restarts) at this sample.
    ///
    /// Either the detector has too little history yet, or the gap since the previous sample exceeded
    /// the configured maximum — over a long gap the test series moves for reasons that are not
    /// slips, so continuity is not testable and claiming it would be a guess.
    ArcStart,
    /// Tested and continuous, with the test statistic that was compared to the threshold.
    Continuous {
        /// The statistic, in the units of the test series.
        statistic: f64,
    },
    /// Tested and rejected: a slip sits between the previous sample and this one, and the arc
    /// restarts here.
    Slipped {
        /// The statistic, in the units of the test series.
        statistic: f64,
    },
}

impl SlipVerdict {
    /// Whether this sample begins a new arc — [`Self::ArcStart`] or [`Self::Slipped`].
    ///
    /// The two differ in *why* the arc broke and agree on the consequence, which is what a caller
    /// maintaining an ambiguity per arc acts on.
    #[must_use]
    pub const fn breaks_arc(self) -> bool {
        matches!(self, Self::ArcStart | Self::Slipped { .. })
    }
}

/// A refusal to build a detector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorError {
    /// The threshold was not finite and positive. A non-positive threshold rejects every sample.
    NonPositiveThreshold,
    /// The maximum gap was not finite and positive.
    NonPositiveGap,
    /// The window holds too few samples to determine a polynomial of that order.
    ///
    /// A fit needs `order + 1` samples to be determined and `order + 2` for the residual to carry
    /// any information at all — with exactly `order + 1` the polynomial passes through every sample
    /// and the residual is identically zero, so the detector would never fire.
    WindowTooShort {
        /// Polynomial order requested.
        order: usize,
        /// Window length requested.
        window: usize,
    },
}

impl fmt::Display for DetectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NonPositiveThreshold => write!(f, "the threshold is not finite and positive"),
            Self::NonPositiveGap => write!(f, "the maximum gap is not finite and positive"),
            Self::WindowTooShort { order, window } => write!(
                f,
                "a window of {window} cannot test a polynomial of order {order}"
            ),
        }
    }
}

impl std::error::Error for DetectorError {}

fn check_limits(threshold: f64, max_gap: f64) -> Result<(), DetectorError> {
    if !(threshold.is_finite() && threshold > 0.0) {
        return Err(DetectorError::NonPositiveThreshold);
    }
    if !(max_gap.is_finite() && max_gap > 0.0) {
        return Err(DetectorError::NonPositiveGap);
    }
    Ok(())
}

/// First-difference detector on a test series: a slip where `|x(t) − x(t_prev)|` exceeds the
/// threshold.
///
/// The threshold is on the raw difference, not on a rate, so it has to be chosen for the sampling
/// interval the arc is observed at — on the geometry-free combination at 30 s, real ionospheric
/// change is at the centimetre level and a threshold of a few centimetres separates it from the
/// 0.05–0.11 m a one-cycle L1 or L2 slip moves it by.
///
/// What it cannot see: a slip whose size is under the threshold, and the pair of L1/L2 slips
/// `(ΔN1, ΔN2)` with `ΔN1·λ1 = ΔN2·λ2`. On GPS L1/L2 the smallest such pair is `(77, 60)`, which no
/// receiver produces by accident — this is the standard blind spot of the geometry-free test and
/// the reason a Melbourne–Wübbena series is usually run beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct GeometryFreeDetector {
    threshold: f64,
    max_gap: f64,
    previous: Option<(f64, f64)>,
}

impl GeometryFreeDetector {
    /// A detector rejecting a first difference above `threshold` (units of the test series), and
    /// restarting the arc after a gap longer than `max_gap` seconds.
    ///
    /// # Errors
    /// [`DetectorError::NonPositiveThreshold`] or [`DetectorError::NonPositiveGap`] if either limit
    /// is not finite and positive.
    pub fn new(threshold: f64, max_gap: f64) -> Result<Self, DetectorError> {
        check_limits(threshold, max_gap)?;
        Ok(Self {
            threshold,
            max_gap,
            previous: None,
        })
    }

    /// The rejection threshold this detector was built with.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Forget the arc, so the next sample is an [`SlipVerdict::ArcStart`].
    pub fn reset(&mut self) {
        self.previous = None;
    }

    /// Test one sample: `time` in seconds, `value` the test series.
    ///
    /// A non-finite value, or a time not strictly after the previous one, restarts the arc rather
    /// than testing against it — out-of-order samples are a caller error the detector cannot repair,
    /// and continuing would test a difference over a negative interval.
    pub fn update(&mut self, time: f64, value: f64) -> SlipVerdict {
        if !(time.is_finite() && value.is_finite()) {
            self.previous = None;
            return SlipVerdict::ArcStart;
        }
        let Some((t_prev, x_prev)) = self.previous else {
            self.previous = Some((time, value));
            return SlipVerdict::ArcStart;
        };
        self.previous = Some((time, value));
        let gap = time - t_prev;
        if gap <= 0.0 || gap > self.max_gap {
            return SlipVerdict::ArcStart;
        }
        let statistic = (value - x_prev).abs();
        if statistic > self.threshold {
            SlipVerdict::Slipped { statistic }
        } else {
            SlipVerdict::Continuous { statistic }
        }
    }
}

/// Polynomial-fit detector: fit the window, predict the next sample, test the residual.
///
/// A least-squares polynomial of the configured order is fitted to the last `window` samples and
/// extrapolated to the new epoch; the statistic is `|value − prediction|`. Against
/// [`GeometryFreeDetector`] this buys sensitivity where the series curves — a second-order fit
/// follows the ionospheric trend a first difference has to leave inside its threshold — and costs a
/// window of samples before the first verdict, plus a rejected sample's worth of latency after every
/// restart.
///
/// The fit is done in a time variable shifted to the window's last epoch and scaled to its span, so
/// the normal matrix stays conditioned whether the caller times samples in seconds since the arc
/// began or in seconds since J2000.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialFitDetector {
    order: usize,
    window: usize,
    threshold: f64,
    max_gap: f64,
    samples: Vec<(f64, f64)>,
}

impl PolynomialFitDetector {
    /// A detector fitting `order`-degree polynomials over the last `window` samples and rejecting a
    /// prediction residual above `threshold`, restarting after a gap longer than `max_gap` seconds.
    ///
    /// # Errors
    /// [`DetectorError::WindowTooShort`] if `window < order + 2`, and the limit errors of
    /// [`GeometryFreeDetector::new`].
    pub fn new(
        order: usize,
        window: usize,
        threshold: f64,
        max_gap: f64,
    ) -> Result<Self, DetectorError> {
        check_limits(threshold, max_gap)?;
        if window < order + 2 {
            return Err(DetectorError::WindowTooShort { order, window });
        }
        Ok(Self {
            order,
            window,
            threshold,
            max_gap,
            samples: Vec::new(),
        })
    }

    /// The rejection threshold this detector was built with.
    #[must_use]
    pub const fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Samples currently held — the fit begins once this reaches the configured window.
    #[must_use]
    pub fn buffered(&self) -> usize {
        self.samples.len()
    }

    /// Forget the arc, so the next `window` samples are [`SlipVerdict::ArcStart`].
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Test one sample: `time` in seconds, `value` the test series.
    ///
    /// Behaves as [`GeometryFreeDetector::update`] on non-finite input, out-of-order times and long
    /// gaps, and additionally returns [`SlipVerdict::ArcStart`] while the window is still filling.
    pub fn update(&mut self, time: f64, value: f64) -> SlipVerdict {
        if !(time.is_finite() && value.is_finite()) {
            self.samples.clear();
            return SlipVerdict::ArcStart;
        }
        if let Some(&(t_prev, _)) = self.samples.last() {
            let gap = time - t_prev;
            if gap <= 0.0 || gap > self.max_gap {
                self.samples.clear();
            }
        }
        if self.samples.len() < self.window {
            self.samples.push((time, value));
            return SlipVerdict::ArcStart;
        }

        let verdict = match predict(&self.samples, self.order, time) {
            // A window of distinct epochs always determines the fit; a singular normal matrix means
            // the caller's times are degenerate, and there is nothing to test against.
            None => {
                self.samples.clear();
                self.samples.push((time, value));
                return SlipVerdict::ArcStart;
            }
            Some(prediction) => {
                let statistic = (value - prediction).abs();
                if statistic > self.threshold {
                    SlipVerdict::Slipped { statistic }
                } else {
                    SlipVerdict::Continuous { statistic }
                }
            }
        };

        if matches!(verdict, SlipVerdict::Slipped { .. }) {
            self.samples.clear();
        } else {
            self.samples.remove(0);
        }
        self.samples.push((time, value));
        verdict
    }
}

/// Least-squares polynomial of degree `order` through `samples`, evaluated at `at`.
///
/// `None` where the normal matrix is singular, which for distinct sample epochs it is not.
fn predict(samples: &[(f64, f64)], order: usize, at: f64) -> Option<f64> {
    let n = order + 1;
    let (t_ref, _) = *samples.last()?;
    let span = samples
        .iter()
        .map(|&(t, _)| (t - t_ref).abs())
        .fold(0.0_f64, f64::max);
    let scale = if span > 0.0 { span } else { 1.0 };

    // Normal equations for the shifted, scaled Vandermonde system.
    let mut normal = vec![0.0_f64; n * n];
    let mut rhs = vec![0.0_f64; n];
    for &(t, y) in samples {
        let tau = (t - t_ref) / scale;
        let mut powers = vec![1.0_f64; n];
        for k in 1..n {
            powers[k] = powers[k - 1] * tau;
        }
        for i in 0..n {
            rhs[i] += powers[i] * y;
            for j in 0..n {
                normal[i * n + j] += powers[i] * powers[j];
            }
        }
    }

    let coefficients = solve(&mut normal, &mut rhs, n)?;
    let tau = (at - t_ref) / scale;
    Some(
        coefficients
            .iter()
            .rev()
            .fold(0.0, |acc, &c| acc.mul_add(tau, c)),
    )
}

/// Gaussian elimination with partial pivoting, in place. `None` on a singular matrix.
fn solve(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for k in 0..n {
        let pivot = (k..n).max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))?;
        let head = a[pivot * n + k];
        if !head.is_finite() || head == 0.0 {
            return None;
        }
        if pivot != k {
            for c in 0..n {
                a.swap(k * n + c, pivot * n + c);
            }
            b.swap(k, pivot);
        }
        for i in k + 1..n {
            let factor = a[i * n + k] / a[k * n + k];
            if factor == 0.0 {
                continue;
            }
            for c in k..n {
                a[i * n + c] -= factor * a[k * n + c];
            }
            b[i] -= factor * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in i + 1..n {
            sum -= a[i * n + j] * x[j];
        }
        let value = sum / a[i * n + i];
        if !value.is_finite() {
            return None;
        }
        x[i] = value;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    // The boundary tests below stand exactly on a tie and the solver fixtures reduce exactly, so
    // equality is the assertion and a tolerance would be the weaker statement.
    #![allow(clippy::float_cmp, clippy::unwrap_used)]
    use super::*;

    /// A clean 30 s geometry-free arc: a linear ionospheric trend, no slip anywhere.
    fn clean_arc(epochs: usize) -> Vec<(f64, f64)> {
        (0..epochs)
            .map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let t = k as f64 * 30.0;
                (t, 0.10 + 2.0e-4 * t)
            })
            .collect()
    }

    /// The first difference fires on a one-cycle L1 slip and stays quiet on the trend that carries
    /// it. Measured on the arc above: the clean first difference is 6.0e-3 m per 30 s epoch, and a
    /// one-cycle L1 slip adds 0.190 m — a factor of 32, which is why the threshold is not delicate.
    #[test]
    fn the_geometry_free_detector_finds_a_one_cycle_slip() {
        let mut detector = GeometryFreeDetector::new(0.05, 60.0).unwrap();
        let arc = clean_arc(10);
        let slip_at = 5;
        let lambda1 = 0.190_293_672_798_365_f64;

        let mut worst_clean = 0.0_f64;
        let mut fired = Vec::new();
        for (k, &(t, x)) in arc.iter().enumerate() {
            let value = if k >= slip_at { x + lambda1 } else { x };
            match detector.update(t, value) {
                SlipVerdict::ArcStart => assert_eq!(k, 0, "unexpected restart at {k}"),
                SlipVerdict::Continuous { statistic } => worst_clean = worst_clean.max(statistic),
                SlipVerdict::Slipped { statistic } => fired.push((k, statistic)),
            }
        }
        assert_eq!(fired.len(), 1, "fired at {fired:?}");
        assert_eq!(fired[0].0, slip_at);
        assert!(
            (fired[0].1 - (lambda1 + 6.0e-3)).abs() < 1e-9,
            "statistic {}",
            fired[0].1
        );
        assert!((worst_clean - 6.0e-3).abs() < 1e-9, "clean {worst_clean}");
    }

    /// A gap longer than the configured maximum restarts the arc instead of testing across it — the
    /// series really has moved, and calling that a slip would be as wrong as calling it continuous.
    #[test]
    fn a_long_gap_restarts_rather_than_fires() {
        let mut detector = GeometryFreeDetector::new(0.05, 60.0).unwrap();
        assert_eq!(detector.update(0.0, 0.10), SlipVerdict::ArcStart);
        assert!(matches!(
            detector.update(30.0, 0.106),
            SlipVerdict::Continuous { .. }
        ));
        // 300 s later the ionosphere has moved by more than the threshold, and it is not a slip.
        assert_eq!(detector.update(330.0, 0.166), SlipVerdict::ArcStart);
        // The arc resumes from there.
        assert!(matches!(
            detector.update(360.0, 0.172),
            SlipVerdict::Continuous { .. }
        ));
    }

    /// Out-of-order and non-finite samples restart the arc rather than testing a negative interval.
    #[test]
    fn degenerate_samples_restart_the_arc() {
        let mut gf = GeometryFreeDetector::new(0.05, 60.0).unwrap();
        assert_eq!(gf.update(0.0, 0.10), SlipVerdict::ArcStart);
        assert_eq!(gf.update(0.0, 0.10), SlipVerdict::ArcStart, "same epoch");
        assert_eq!(gf.update(-30.0, 0.10), SlipVerdict::ArcStart, "backwards");
        assert_eq!(gf.update(30.0, f64::NAN), SlipVerdict::ArcStart, "NaN");
        assert_eq!(gf.update(60.0, 0.10), SlipVerdict::ArcStart, "after NaN");

        let mut pf = PolynomialFitDetector::new(2, 6, 0.05, 60.0).unwrap();
        for (t, x) in clean_arc(6) {
            assert_eq!(pf.update(t, x), SlipVerdict::ArcStart);
        }
        assert_eq!(pf.buffered(), 6);
        assert_eq!(pf.update(180.0, f64::INFINITY), SlipVerdict::ArcStart);
        assert_eq!(pf.buffered(), 0, "the window survived a non-finite sample");
    }

    /// The polynomial fit is exact on a polynomial: a second-order detector fed a quadratic
    /// predicts it to the last bits, so the whole residual is the slip and nothing else.
    #[test]
    fn the_polynomial_detector_predicts_a_quadratic_exactly() {
        let mut detector = PolynomialFitDetector::new(2, 6, 0.05, 60.0).unwrap();
        let quadratic = |t: f64| 0.10 + 1.0e-3 * t + 2.0e-6 * t * t;

        let mut worst_clean = 0.0_f64;
        let (mut fired, mut restarted) = (Vec::new(), Vec::new());
        for k in 0..20 {
            let t = f64::from(k) * 30.0;
            // A slip is a step, not a spike: every epoch after it carries the same offset.
            let value = if k >= 12 {
                quadratic(t) + 0.19
            } else {
                quadratic(t)
            };
            match detector.update(t, value) {
                SlipVerdict::ArcStart => restarted.push(k),
                SlipVerdict::Continuous { statistic } => worst_clean = worst_clean.max(statistic),
                SlipVerdict::Slipped { statistic } => fired.push((k, statistic)),
            }
        }
        assert_eq!(fired.len(), 1, "fired at {fired:?}");
        assert_eq!(fired[0].0, 12);
        assert!((fired[0].1 - 0.19).abs() < 1e-9, "statistic {}", fired[0].1);
        // The window fills once at the start and once after the slip discards it — and nowhere else,
        // which is what says the arc was cut at 12 and only at 12.
        assert_eq!(restarted, vec![0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17]);
        // The prediction residual on the clean quadratic, which is the detector's own noise floor.
        assert!(worst_clean < 1e-9, "clean residual {worst_clean:.3e}");
    }

    /// The curvature the first difference charges as noise and the polynomial fit follows.
    ///
    /// On a geometry-free series curving at 2e-6 m/s² the first difference grows to 0.075 m over a
    /// 30-epoch arc — past any threshold that would still catch a 0.02 m half-cycle slip — while the
    /// second-order fit predicts it exactly. Measured below: first difference 7.5e-2 m at the last
    /// epoch, polynomial residual 3.5e-16 m.
    #[test]
    fn the_polynomial_fit_follows_curvature_the_first_difference_cannot() {
        let curve = |t: f64| 0.10 + 2.0e-6 * t * t;
        let mut gf = GeometryFreeDetector::new(0.05, 60.0).unwrap();
        let mut pf = PolynomialFitDetector::new(2, 6, 0.05, 60.0).unwrap();

        let (mut worst_difference, mut worst_residual) = (0.0_f64, 0.0_f64);
        for k in 0..30 {
            let t = f64::from(k) * 30.0;
            if let SlipVerdict::Continuous { statistic } | SlipVerdict::Slipped { statistic } =
                gf.update(t, curve(t))
            {
                worst_difference = worst_difference.max(statistic);
            }
            if let SlipVerdict::Continuous { statistic } | SlipVerdict::Slipped { statistic } =
                pf.update(t, curve(t))
            {
                worst_residual = worst_residual.max(statistic);
            }
        }
        assert!(
            worst_difference > 0.05,
            "the first difference stayed inside the threshold: {worst_difference:.3e}"
        );
        assert!(
            worst_residual < 1e-9,
            "the polynomial fit did not follow the curve: {worst_residual:.3e}"
        );
    }

    #[test]
    fn a_detector_refuses_limits_it_cannot_use() {
        assert_eq!(
            GeometryFreeDetector::new(0.0, 60.0),
            Err(DetectorError::NonPositiveThreshold)
        );
        assert_eq!(
            GeometryFreeDetector::new(f64::NAN, 60.0),
            Err(DetectorError::NonPositiveThreshold)
        );
        assert_eq!(
            GeometryFreeDetector::new(0.05, -1.0),
            Err(DetectorError::NonPositiveGap)
        );
        // Zero is the boundary the sign test is about: a maximum gap of zero admits no pair of
        // samples at all, so it is refused rather than accepted as a degenerate limit.
        assert_eq!(
            GeometryFreeDetector::new(0.05, 0.0),
            Err(DetectorError::NonPositiveGap)
        );
        assert_eq!(
            PolynomialFitDetector::new(2, 3, 0.05, 60.0),
            Err(DetectorError::WindowTooShort {
                order: 2,
                window: 3
            })
        );
        // order + 2 is admitted; order + 1 is not.
        assert!(PolynomialFitDetector::new(2, 4, 0.05, 60.0).is_ok());

        let messages: Vec<String> = [
            DetectorError::NonPositiveThreshold,
            DetectorError::NonPositiveGap,
            DetectorError::WindowTooShort {
                order: 2,
                window: 3,
            },
        ]
        .iter()
        .map(ToString::to_string)
        .collect();
        let distinct: std::collections::HashSet<&String> = messages.iter().collect();
        assert_eq!(distinct.len(), 3, "duplicate: {messages:?}");
    }

    /// Both verdicts that break an arc say so, and the one that does not, does not.
    #[test]
    fn the_arc_breaking_verdicts_are_the_two_that_break_an_arc() {
        assert!(SlipVerdict::ArcStart.breaks_arc());
        assert!(SlipVerdict::Slipped { statistic: 1.0 }.breaks_arc());
        assert!(!SlipVerdict::Continuous { statistic: 1.0 }.breaks_arc());
    }

    /// Each detector hands back the threshold it was built with. Neither value is 0, 1 or −1, so
    /// the assertion separates the accessor from a constant rather than agreeing with one.
    #[test]
    fn a_detector_reports_the_threshold_it_was_built_with() {
        assert_eq!(
            GeometryFreeDetector::new(0.037, 60.0).unwrap().threshold(),
            0.037
        );
        assert_eq!(
            PolynomialFitDetector::new(2, 6, 0.052, 60.0)
                .unwrap()
                .threshold(),
            0.052
        );
    }

    /// A reset arc is a new arc: the next sample is compared against nothing, not against the
    /// history that stood before the reset.
    #[test]
    fn a_reset_detector_starts_a_new_arc() {
        let mut gf = GeometryFreeDetector::new(0.05, 60.0).unwrap();
        assert_eq!(gf.update(0.0, 0.10), SlipVerdict::ArcStart);
        assert!(matches!(
            gf.update(30.0, 0.106),
            SlipVerdict::Continuous { .. }
        ));
        gf.reset();
        // Continuous with the previous sample, and an arc start without it.
        assert_eq!(gf.update(60.0, 0.112), SlipVerdict::ArcStart);

        let mut pf = PolynomialFitDetector::new(2, 6, 0.05, 60.0).unwrap();
        for (t, x) in clean_arc(6) {
            assert_eq!(pf.update(t, x), SlipVerdict::ArcStart);
        }
        assert_eq!(pf.buffered(), 6);
        pf.reset();
        assert_eq!(pf.buffered(), 0);
        assert_eq!(pf.update(180.0, 0.136), SlipVerdict::ArcStart);
    }

    /// A gap of exactly the configured maximum is the last one the arc survives. The tie is
    /// reachable because the sampling and the limit are both whole seconds, so `time - t_prev` is
    /// `max_gap` bit for bit rather than approaching it.
    #[test]
    fn a_gap_of_exactly_the_maximum_stays_on_the_same_arc() {
        let mut gf = GeometryFreeDetector::new(0.05, 60.0).unwrap();
        assert_eq!(gf.update(0.0, 0.10), SlipVerdict::ArcStart);
        assert!(matches!(
            gf.update(60.0, 0.112),
            SlipVerdict::Continuous { .. }
        ));

        let mut pf = PolynomialFitDetector::new(2, 6, 0.05, 60.0).unwrap();
        for (t, x) in clean_arc(6) {
            assert_eq!(pf.update(t, x), SlipVerdict::ArcStart);
        }
        assert!(matches!(
            pf.update(210.0, 0.142),
            SlipVerdict::Continuous { .. }
        ));
        assert_eq!(pf.buffered(), 6, "the window survived a gap of the maximum");
    }

    /// A gap past the maximum throws the window away instead of fitting across it, so the sample
    /// that carries the gap starts an arc and is the only thing left in the buffer.
    #[test]
    fn a_gap_past_the_maximum_discards_the_polynomial_window() {
        let mut pf = PolynomialFitDetector::new(2, 6, 0.05, 60.0).unwrap();
        for (t, x) in clean_arc(6) {
            assert_eq!(pf.update(t, x), SlipVerdict::ArcStart);
        }
        assert_eq!(pf.buffered(), 6);
        // 300 s after the window's last epoch, and the series has moved with it: fitted across,
        // the prediction would land on the sample and the arc would read as continuous.
        assert_eq!(pf.update(450.0, 0.19), SlipVerdict::ArcStart);
        assert_eq!(pf.buffered(), 1);
    }

    /// A statistic of exactly the threshold is the largest one that is not a slip. Both thresholds
    /// are read off the computation rather than chosen, so the comparison stands on its own tie.
    #[test]
    fn a_statistic_of_exactly_the_threshold_is_not_a_slip() {
        let (previous, value) = (0.10_f64, 0.15_f64);
        let tie = (value - previous).abs();
        let mut gf = GeometryFreeDetector::new(tie, 60.0).unwrap();
        assert_eq!(gf.update(0.0, previous), SlipVerdict::ArcStart);
        assert_eq!(
            gf.update(30.0, value),
            SlipVerdict::Continuous { statistic: tie }
        );

        let quadratic = |t: f64| 0.10 + 1.0e-3 * t + 2.0e-6 * t * t;
        let arc: Vec<(f64, f64)> = (0..13)
            .map(|k| {
                let t = f64::from(k) * 30.0;
                (
                    t,
                    if k >= 12 {
                        quadratic(t) + 0.19
                    } else {
                        quadratic(t)
                    },
                )
            })
            .collect();

        // What the fit arrives at on the last sample, taken from a run whose threshold cannot
        // fire. Every earlier verdict is continuous in both runs, so the two see the same window.
        let mut wide = PolynomialFitDetector::new(2, 6, 1.0, 60.0).unwrap();
        let mut residual = 0.0_f64;
        for &(t, x) in &arc {
            if let SlipVerdict::Continuous { statistic } | SlipVerdict::Slipped { statistic } =
                wide.update(t, x)
            {
                residual = statistic;
            }
        }
        assert!(residual > 0.0, "the step left no residual to stand on");

        let mut pf = PolynomialFitDetector::new(2, 6, residual, 60.0).unwrap();
        let mut verdict = SlipVerdict::ArcStart;
        for &(t, x) in &arc {
            verdict = pf.update(t, x);
        }
        assert_eq!(
            verdict,
            SlipVerdict::Continuous {
                statistic: residual
            }
        );
    }

    /// A window whose samples all share one epoch has no span to divide by, and the fallback scale
    /// of one is what keeps the fit variable a number: dividing by the span itself would send the
    /// fit variable at the prediction epoch to an infinity, which the Horner fold turns into a NaN
    /// and an answer the constant fit still has into no answer at all. That answer is the mean —
    /// the least-squares constant through samples at a single epoch, at every prediction epoch.
    #[test]
    fn a_window_at_one_epoch_is_fitted_by_its_mean() {
        let samples = [(90.0, 0.10), (90.0, 0.20), (90.0, 0.30)];
        let prediction = predict(&samples, 0, 120.0).unwrap();
        assert_eq!(prediction, (0.10 + 0.20 + 0.30) / 3.0);
    }

    /// The scale is the window's span, and a span is a difference: a window that ends at the time
    /// origin has `t_ref == 0`, where a ratio would divide by zero, send the span to infinity and
    /// collapse the normal matrix to rank one. Measured residual against the quadratic the samples
    /// were drawn from: 4.4e-16 m.
    #[test]
    fn a_window_ending_at_the_time_origin_is_fitted_from_its_span() {
        let quadratic = |t: f64| 0.10 + 1.0e-3 * t + 2.0e-6 * t * t;
        let samples: Vec<(f64, f64)> = (0..6)
            .map(|k| {
                let t = f64::from(k) * 30.0 - 150.0;
                (t, quadratic(t))
            })
            .collect();
        assert_eq!(samples[5].0, 0.0, "the last epoch is the time origin");

        let prediction = predict(&samples, 2, 30.0).unwrap();
        assert!(
            (prediction - quadratic(30.0)).abs() < 1.0e-15,
            "residual {:.3e}",
            (prediction - quadratic(30.0)).abs()
        );
    }

    /// `solve` is reached only through `predict`, which accumulates
    /// `normal[i * n + j] += powers[i] * powers[j]` — a matrix that is symmetric, so every index
    /// expression equals its own transpose, and diagonally dominant once the fit variable is
    /// scaled to the span, so the partial-pivot search never moves a row. Nothing arriving that
    /// way can testify about either.
    ///
    /// These two systems are asymmetric and carry an exact zero where a mis-indexed pivot search
    /// lands, so a wrong index reads the zero and the elimination refuses instead of answering.
    /// The entries are small integers and every elimination factor is a dyadic rational, so the
    /// reduction is exact in `f64` and the solution is asserted by equality.
    #[test]
    fn the_solver_pivots_where_the_normal_equations_never_ask_it_to() {
        for (label, mut a, mut b) in [
            (
                // The leading entry is exactly zero, so the row swap is not an optimisation:
                // eliminating in place would divide by it.
                "a zero leading entry",
                vec![0.0, 1.0, 0.0, -2.0, 0.0, -2.0, 0.0, 8.0, -2.0],
                vec![2.0, -8.0, 10.0],
            ),
            (
                // The pivot is the leading entry and the rest of its column is zero, so a search
                // that picks any other row picks a zero.
                "a zero column under the pivot",
                vec![1.0, -1.0, -4.0, -1.0, 0.0, 0.0, 0.0, 0.0, 4.0],
                vec![-13.0, -1.0, 12.0],
            ),
        ] {
            let x = solve(&mut a, &mut b, 3).unwrap_or_else(|| panic!("{label}: refused"));
            assert_eq!(x, vec![1.0, 2.0, 3.0], "{label}");
        }
    }

    /// The two shapes `solve` has to refuse. A singular system is what `predict` reports as a
    /// degenerate window; a non-finite head is what it hands over when a sample's powers overflow,
    /// and the refusal is what stops the elimination dividing by an infinity and returning a
    /// finite, wrong answer.
    #[test]
    fn the_solver_refuses_a_system_it_cannot_reduce() {
        // Row 1 is twice row 0.
        let mut singular = vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 1.0, 1.0];
        let mut rhs = vec![14.0, 28.0, 6.0];
        assert_eq!(solve(&mut singular, &mut rhs, 3), None);

        let mut infinite = vec![f64::INFINITY, 3.0, 1.0, 8.0, 4.0, 2.0, 4.0, 1.0, 5.0];
        let mut rhs = vec![11.0, 22.0, 21.0];
        assert_eq!(solve(&mut infinite, &mut rhs, 3), None);
    }
}
