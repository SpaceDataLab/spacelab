//! Composite Boole rule (closed five-point Newton-Cotes).
//!
//! Weights 14/45, 64/45, 24/45, 64/45, 14/45 times the subinterval width, derived in
//! `derivations/jacchia_bowman_diurnal.mac` (exact for polynomials of degree <= 5).

const W_END: f64 = 14.0 / 45.0;
const W_ODD: f64 = 64.0 / 45.0;
const W_MID: f64 = 24.0 / 45.0;

/// Integral of `f` over [a, b] with panels of at most `max_step` (each panel has four
/// subintervals). Returns 0 when a == b.
pub fn boole<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, max_step: f64) -> f64 {
    let len = b - a;
    if len == 0.0 {
        return 0.0;
    }
    let panels = (len.abs() / max_step).ceil().max(1.0) as usize;
    let w = len / panels as f64;
    let h = w / 4.0;
    let mut sum = 0.0;
    let mut f_left = f(a);
    for k in 0..panels {
        let x0 = a + w * k as f64;
        let f_right = f(if k + 1 == panels { b } else { x0 + w });
        sum += W_END * (f_left + f_right)
            + W_ODD * (f(x0 + h) + f(x0 + 3.0 * h))
            + W_MID * f(x0 + 2.0 * h);
        f_left = f_right;
    }
    sum * h
}
