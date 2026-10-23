//! Error tolerance and the scaled error norm that drives step acceptance.

/// Absolute and relative tolerance for the scaled error norm.
///
/// The scaled error of a step is `‖ y_err_i / (atol + rtol · max(|y_i|, |y_next_i|)) ‖` in the RMS
/// norm; a step is accepted when it is `≤ 1`. Absolute tolerance dominates where the state passes
/// through zero, relative tolerance where it is large — the standard Hairer construction.
///
/// A single scalar pair is the starting point. Per-quantity tolerances (position vs velocity vs
/// mass vs attitude) are a documented tier-1 companion and slot in by making [`Tolerance::scale`]
/// index-aware; nothing else changes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerance {
    /// Absolute tolerance.
    pub atol: f64,
    /// Relative tolerance.
    pub rtol: f64,
}

impl Tolerance {
    /// Build a tolerance. Both must be non-negative and not both zero.
    #[must_use]
    pub const fn new(atol: f64, rtol: f64) -> Self {
        Self { atol, rtol }
    }

    /// The per-component denominator `atol + rtol · max(|y|, |y_next|)`.
    #[inline]
    #[must_use]
    pub fn scale(&self, y: f64, y_next: f64) -> f64 {
        self.atol + self.rtol * libm::fabs(y).max(libm::fabs(y_next))
    }

    /// The RMS scaled error norm over a step. Acceptance is `norm ≤ 1`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // N is a small state dimension, exactly representable
    pub fn error_norm<const N: usize>(
        &self,
        y: &[f64; N],
        y_next: &[f64; N],
        y_err: &[f64; N],
    ) -> f64 {
        let mut sum = 0.0;
        for ((&yi, &yn), &err) in y.iter().zip(y_next).zip(y_err) {
            let r = err / self.scale(yi, yn);
            sum += r * r;
        }
        libm::sqrt(sum / N as f64)
    }
}

impl Default for Tolerance {
    /// `atol = 1e-9`, `rtol = 1e-9` — a tight default suited to metre-level orbit work in SI units.
    fn default() -> Self {
        Self::new(1e-9, 1e-9)
    }
}
