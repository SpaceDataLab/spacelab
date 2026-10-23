//! Shared analytical helpers from the `NeQuick` reference (section F.2.12).

/// Clipped exponential (reference section F.2.12.2).
///
/// Saturates the argument outside `[-80, 80]` to the reference's fixed bounds, so the
/// exponential never overflows or underflows the profile arithmetic.
pub(crate) fn clip_exp(power: f64) -> f64 {
    if power > 80.0 {
        5.5406e34
    } else if power < -80.0 {
        1.8049e-35
    } else {
        libm::exp(power)
    }
}

/// Smoothly join two functions with a continuous first derivative (section F.2.12.1).
///
/// `f1` dominates where `a * x` is large and positive, `f2` where it is large and
/// negative, with a transition of width set by `a` around `x = 0`.
pub(crate) fn join(f1: f64, f2: f64, a: f64, x: f64) -> f64 {
    let ee = clip_exp(a * x);
    (f1 * ee + f2) / (ee + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The clamp is not the exponential it replaces, and the bounds belong to the exponential.
    ///
    /// `exp(80)` is `5.540_622_38e34` against a clamp of `5.5406e34` — a difference in the fifth
    /// significant figure, so which side of `±80` the boundary falls on is observable in the
    /// electron density, not a rounding detail. Section F.2.12.2 puts both bounds inside the
    /// exponential branch, and the sTEC fixtures never drive an argument anywhere near them.
    #[test]
    #[allow(clippy::float_cmp)] // the reference's exact constants, and being exactly them is the point
    fn the_exponential_is_clipped_to_the_published_constants_beyond_eighty() {
        assert_eq!(
            clip_exp(80.0),
            libm::exp(80.0),
            "80 belongs to the exponential"
        );
        assert_eq!(
            clip_exp(-80.0),
            libm::exp(-80.0),
            "-80 belongs to the exponential"
        );
        assert_eq!(clip_exp(80.5), 5.5406e34);
        assert_eq!(clip_exp(1e300), 5.5406e34);
        assert_eq!(clip_exp(-80.5), 1.8049e-35);
        assert_eq!(clip_exp(-1e300), 1.8049e-35);
    }
}
