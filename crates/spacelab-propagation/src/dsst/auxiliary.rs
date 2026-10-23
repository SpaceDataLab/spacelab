//! Equinoctial auxiliary elements for the DSST (Danielson §2.1), the internal representation every
//! DSST force is written in terms of.
//!
//! Ported from Orekit's `AuxiliaryElements` together with the derived factors of
//! `DSSTGravityContext` / `DSSTZonalContext`. The zonal contribution is built with a body-fixed
//! frame equal to the orbit frame (Orekit's `new DSSTZonal(provider)`, a null central-body frame),
//! so the transform to the pole is the identity and the direction cosines (α, β, γ) reduce to the
//! z-components of the equinoctial f/g/w vectors — pure functions of `(p, q)`. Field names mirror
//! Orekit for auditability.

use libm::sqrt;

/// Retrograde factor I. Orekit keeps it in the formulas for theoretical consistency but always uses
/// the direct (prograde) representation I = +1, since it does not implement retrograde equinoctial
/// elements.
pub(crate) const I: f64 = 1.0;

/// Equinoctial auxiliary elements and the derived DSST gravity-context factors, all `f64`.
///
/// Built from the equinoctial elements `(a, k, h, q, p, λ)` and `µ`, where in Danielson/Orekit DSST
/// naming `k = eₓ`, `h = e_y`, `q = hₓ`, `p = h_y` (i.e. `k, h` are the eccentricity-vector
/// components and `q, p` the inclination-vector components).
#[derive(Debug, Clone, Copy)]
pub struct DsstAuxiliary {
    /// Semi-major axis `a`.
    pub sma: f64,
    /// x-component of the eccentricity vector, `k = eₓ`.
    pub k: f64,
    /// y-component of the eccentricity vector, `h = e_y`.
    pub h: f64,
    /// x-component of the inclination vector, `q = hₓ`.
    pub q: f64,
    /// y-component of the inclination vector, `p = h_y`.
    pub p: f64,
    /// Mean longitude `λ` (does not enter the axisymmetric zonal mean rates, kept for completeness).
    pub lambda: f64,
    /// Gravitational parameter `µ`.
    pub mu: f64,

    /// Eccentricity `e = √(k² + h²)`.
    pub ecc: f64,
    /// Keplerian mean motion `n = √(µ / |a|³)`.
    pub n: f64,
    /// `A = √(µ · |a|)`.
    pub big_a: f64,
    /// `B = √(1 − k² − h²)`.
    pub big_b: f64,
    /// `C = 1 + p² + q²`.
    pub big_c: f64,
    /// `χ = 1 / B`.
    pub chi: f64,
    /// `χ²`.
    pub chi2: f64,
    /// `χ³`.
    pub chi3: f64,
    /// Direction cosine α = z-component of the equinoctial f vector.
    pub alpha: f64,
    /// Direction cosine β = z-component of the equinoctial g vector.
    pub beta: f64,
    /// Direction cosine γ = z-component of the equinoctial w vector (= cos i).
    pub gamma: f64,
    /// Equinoctial reference-frame f vector [Danielson 2.1.4-(1)], a pure function of `(p, q)`.
    /// Needed by the third-body force, whose direction cosines are a general body direction dotted
    /// with f/g/w (the zonal α/β/γ above are just the z-components of these three).
    pub f_vec: [f64; 3],
    /// Equinoctial reference-frame g vector [Danielson 2.1.4-(1)].
    pub g_vec: [f64; 3],
    /// Equinoctial reference-frame w vector [Danielson 2.1.4-(1)] (orbit-plane normal).
    pub w_vec: [f64; 3],
    /// `2a / A`.
    pub ax2oa: f64,
    /// `1 / (A · B)`.
    pub ooab: f64,
    /// `B / A`.
    pub boa: f64,
    /// `B / (A · (1 + B))`.
    pub boabpo: f64,
    /// `C / (2 · A · B)`.
    pub co2ab: f64,
    /// `µ / a`.
    pub muoa: f64,
}

impl DsstAuxiliary {
    /// Build the auxiliary elements from equinoctial elements `(a, k, h, q, p, λ)` and `µ`.
    ///
    /// Mirrors `AuxiliaryElements` (A, B, C and the f/g/w z-components) and the `DSSTGravityContext`
    /// factors (χ, `2a/A`, `1/AB`, `B/A`, `B/A(1+B)`, `C/2AB`, `µ/a`) plus `DSSTZonalContext.chi3`.
    #[must_use]
    pub fn new(sma: f64, k: f64, h: f64, q: f64, p: f64, lambda: f64, mu: f64) -> Self {
        let k2 = k * k;
        let h2 = h * h;
        let q2 = q * q;
        let p2 = p * p;

        // A, B, C parameters [Danielson 2.1.6-(1)].
        let big_b = sqrt(1.0 - k2 - h2);
        let big_c = 1.0 + q2 + p2;

        // Direction cosines: z-components of the equinoctial reference-frame vectors [2.1.4-(1)],
        // evaluated for a pole along the orbit-frame +Z (null body-fixed frame ⇒ identity transform).
        let oo_c = 1.0 / big_c;
        let px2 = 2.0 * p;
        let qx2 = 2.0 * q;
        let pq2 = px2 * q;
        let alpha = -px2 * I * oo_c;
        let beta = qx2 * oo_c;
        let gamma = (1.0 - p2 - q2) * I * oo_c;
        // Full equinoctial reference frame [Danielson 2.1.4-(1)]; the z-components are α/β/γ above.
        let f_vec = [oo_c * (1.0 - p2 + q2), oo_c * pq2, alpha];
        let g_vec = [oo_c * pq2 * I, oo_c * (1.0 + p2 - q2) * I, beta];
        let w_vec = [oo_c * px2, -oo_c * qx2, gamma];

        // Keplerian mean motion and A = √(µ|a|) [DSSTGravityContext].
        let abs_a = libm::fabs(sma);
        let n = sqrt(mu / abs_a) / abs_a;
        let big_a = sqrt(mu * abs_a);

        let chi = 1.0 / big_b;
        let chi2 = chi * chi;
        let chi3 = chi * chi2;

        let ax2oa = 2.0 * sma / big_a;
        let boa = big_b / big_a;
        let ooab = 1.0 / (big_a * big_b);
        let co2ab = big_c * ooab / 2.0;
        let boabpo = boa / (1.0 + big_b);
        let muoa = mu / sma;

        Self {
            sma,
            k,
            h,
            q,
            p,
            lambda,
            mu,
            ecc: sqrt(k2 + h2),
            n,
            big_a,
            big_b,
            big_c,
            chi,
            chi2,
            chi3,
            alpha,
            beta,
            gamma,
            f_vec,
            g_vec,
            w_vec,
            ax2oa,
            ooab,
            boa,
            boabpo,
            co2ab,
            muoa,
        }
    }
}
