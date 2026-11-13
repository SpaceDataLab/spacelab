//! GCRF to ITRF: the rigorous IAU 2006/2000A, CIO-based transform.
//!
//! ```text
//! GCRF ──precession-nutation (X,Y,s)──▶ CIRS ──ERA──▶ TIRS ──polar motion──▶ ITRF
//! ```
//!
//! This is the IERS-recommended Celestial Intermediate Origin chain, the pole tracked
//! rigorously through precession and nutation rather than approximated by a fixed
//! axis. It is what the geopotential must be evaluated in for the pole-dependent
//! (tesseral) terms to be correct — the difference from a fixed-axis rotation is the
//! full accumulated precession, about 1.5 km a day at present epochs.
//!
//! The three angles come from [`spacelab_fa`] ([`cip_xy`], [`cio_locator_s`],
//! [`era`]); this module only assembles rotations. `derivations/gcrf_itrf_chain.mac` composes
//! the IERS (2010) chain in 40-digit arithmetic at six epochs and the module test
//! `the_transform_is_the_iers_chain_at_six_epochs` (G3) holds [`gcrf_to_itrf`] to it; `pyerfa`
//! is a comparator in `tests/cio_oracle.rs`.
//!
//! Gate: `derivations/cio_rotation.mac` proves the coded GCRS→CIRS product is the IERS 2010
//! Eq. 5.10 closed form, that the CIP is its CIRS `z`-axis and that the polar-motion product is
//! `Wᵀ`, and prints four reference matrices that `derivations/cio_rotation.gp` rebuilds from the
//! rotation product. The module tests `the_cio_rotation_is_the_iers_closed_form` (G1) and
//! `the_cirs_pole_is_the_cip_the_angles_name` (G1) tie the code to it.

use spacelab_core::{Angle, Epoch, Mat3, Tt, Ut1, Vec3};
use spacelab_fa::{cio_locator_s, cip_xy, era};

use crate::polar_motion::tio_locator;
use crate::rotation::{Cirs, Gcrf, Itrf, Rotation, Tirs, Transform};
use crate::teme::{EARTH_ROTATION_RATE, Orientation};

/// GCRS to CIRS: the precession-nutation rotation `Q(t)` from the CIP coordinates
/// `X, Y` and the CIO locator `s`.
///
/// IERS Conventions 2010 Eq. 5.10:
///
/// ```text
/// Q(t) = R₃(−(E+s)) · R₂(d) · R₃(E),   E = atan2(Y, X),  d = atan √(r²/(1−r²)),  r² = X²+Y²
/// ```
#[allow(clippy::many_single_char_names)] // IERS notation: X, Y, s, E, d
#[must_use]
pub fn gcrf_to_cirs(x: f64, y: f64, s: f64) -> Rotation<Gcrf, Cirs> {
    let r2 = x * x + y * y;
    let e = if r2 > 0.0 { libm::atan2(y, x) } else { 0.0 };
    let d = libm::atan(libm::sqrt(r2 / (1.0 - r2)));
    Rotation::from_matrix(Mat3::rot3(-(e + s)) * Mat3::rot2(d) * Mat3::rot3(e))
}

/// CIRS to TIRS: Earth rotation about the Celestial Intermediate Pole, by the Earth
/// Rotation Angle. A single rotation about the pole, `R₃(ERA)`.
#[must_use]
pub fn cirs_to_tirs(era: Angle) -> Rotation<Cirs, Tirs> {
    Rotation::from_matrix(Mat3::rot3(era.radians()))
}

/// TIRS to ITRF: polar motion, `W(t)ᵀ = R₁(−y_p)·R₂(−x_p)·R₃(s′)`.
///
/// The matrix is identical to [`crate::pef_to_itrf`] — TIRS and PEF differ only by the
/// TIO locator `s′`, which is already this function's third argument — but it carries
/// the `Tirs` source frame in the type, so it composes only into the CIO chain.
#[must_use]
pub fn tirs_to_itrf(xp: Angle, yp: Angle, sp: Angle) -> Rotation<Tirs, Itrf> {
    Rotation::from_matrix(
        Mat3::rot1(-yp.radians()) * Mat3::rot2(-xp.radians()) * Mat3::rot3(sp.radians()),
    )
}

/// The full GCRF to ITRF transform, IAU 2006/2000A, rotation and angular velocity.
///
/// UT1 drives Earth rotation (ERA); TT drives the slowly-varying geometry (CIP
/// coordinates, CIO locator, TIO locator). Both are supplied by the caller — one is
/// never derived from the other inside the transform. The `dx`/`dy` celestial-pole
/// offsets (IERS EOP) correct the CIP position from the model; the CIO locator `s`
/// stays the model value. Pass zero for the pure IAU 2006/2000A pole.
///
/// The angular velocity is the Earth's spin about the Celestial Intermediate Pole.
/// The CIP is the CIRS `z`-axis by construction, so the rate vector is `(0,0,ω)` in
/// CIRS and is rotated back into the GCRF (the `From` frame) here — unlike TEME,
/// whose `z` already approximates the pole, GCRF's does not, and using it would tilt
/// `ω` by the precession-nutation angle. The rate is the instantaneous
/// `ω₀·(1 − LOD/86400)`; pass `0` for the nominal rate.
#[must_use]
pub fn gcrf_to_itrf(
    ut1: Epoch<Ut1>,
    tt: Epoch<Tt>,
    xp: Angle,
    yp: Angle,
    dx: Angle,
    dy: Angle,
    lod_seconds: f64,
) -> Transform<Gcrf, Itrf> {
    let (x, y) = cip_xy(tt);
    let s = cio_locator_s(tt, x, y);
    // dx/dy are the IERS celestial-pole offsets: correct the CIP position, keep the model s.
    let q = gcrf_to_cirs(x + dx.radians(), y + dy.radians(), s);
    let rotation: Rotation<Gcrf, Itrf> =
        tirs_to_itrf(xp, yp, tio_locator(tt)) * cirs_to_tirs(era(ut1)) * q;
    let rate = EARTH_ROTATION_RATE * (1.0 - lod_seconds / 86_400.0);
    let omega = q.inverse().matrix() * Vec3::new(0.0, 0.0, rate);
    Transform::new(rotation, omega)
}

/// Build a GCRF→ITRF transform for each epoch, writing into a caller-provided slice.
///
/// The CIO chain is a function of the epoch alone; hoisting it out of any per-object
/// loop is the same four-orders-of-magnitude win described on [`Orientation`] and
/// [`crate::teme_to_itrf_into`].
///
/// # Panics
///
/// If `out` is shorter than `epochs`.
pub fn gcrf_to_itrf_into(epochs: &[Orientation], out: &mut [Transform<Gcrf, Itrf>]) {
    assert!(
        out.len() >= epochs.len(),
        "output slice holds {} of {} epochs",
        out.len(),
        epochs.len()
    );
    // The coverage batch uses the pure model pole; sub-milliarcsecond CIP offsets are negligible there.
    let zero = Angle::from_radians(0.0);
    for (o, slot) in epochs.iter().zip(out.iter_mut()) {
        *slot = gcrf_to_itrf(o.ut1, o.tt, o.xp, o.yp, zero, zero, o.lod_seconds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TT days since J2000 of the chain gate: `t = 0.06`, a century and half a century either side
    /// of J2000, and two centuries out — the epochs `spacelab_fa` holds `X, Y, s` at.
    const CHAIN_TT_DAYS: [f64; 6] = [2191.5, -36_525.0, -18_262.5, 18_262.5, 36_525.0, 73_050.0];

    /// UT1 days since J2000 at the same instants, nine hours behind TT: the transform must take
    /// the ERA from UT1 and everything else from TT, and a nine-hour gap tells the two apart.
    const CHAIN_UT1_DAYS: [f64; 6] = [
        2191.125,
        -36_525.375,
        -18_262.875,
        18_262.125,
        36_524.625,
        73_049.625,
    ];

    /// `(x_p, y_p, dX, dY)` in arcseconds at each epoch: polar motion of the observed size with
    /// both signs, and celestial-pole offsets of a few tenths of a milliarcsecond, none at the first.
    const CHAIN_EOP: [[f64; 4]; 6] = [
        [0.0526, 0.3838, 0.0, 0.0],
        [0.161, 0.301, 0.0003, -0.00016],
        [-0.12, 0.45, -0.0005, 0.00025],
        [0.25, 0.05, 0.00014, 0.0004],
        [0.03, -0.52, 0.0006, -0.0007],
        [-0.2, 0.2, -0.0002, 0.00012],
    ];

    /// The GCRF→ITRF matrix at [`CHAIN_TT_DAYS`], [`CHAIN_UT1_DAYS`] and [`CHAIN_EOP`]: IERS (2010)
    /// eqs (5.1), (5.3), (5.10), (5.13) and (5.15) composed in 40-digit arithmetic on the exact
    /// `X, Y, s` and ERA, to 25 digits (`derivations/gcrf_itrf_chain.mac` (8) reads this table back).
    #[expect(
        clippy::unreadable_literal,
        clippy::excessive_precision,
        reason = "25-digit literals written by the derivation"
    )]
    const CHAIN_EXACT: [[[f64; 3]; 3]; 6] = [
        [
            [
                8.197555488544985134875432e-1,
                -5.727133980466696832819871e-1,
                -4.514643914984849613971973e-4,
            ],
            [
                5.727132823121961242806741e-1,
                8.197556721024495640926932e-1,
                -3.664956369937373245638696e-4,
            ],
            [
                5.799874573151259997975767e-4,
                4.187717855440453653073546e-5,
                9.999998309304113444658045e-1,
            ],
        ],
        [
            [
                -8.250968785492402412386973e-1,
                5.649357232831171040046117e-1,
                -7.922724713230814580104394e-3,
            ],
            [
                -5.649082721438347785971490e-1,
                -8.251349065604192202091491e-1,
                -5.570461291073301037644937e-3,
            ],
            [
                -9.684269294448733986975493e-3,
                -1.205574949214731020220287e-4,
                9.999530990972141667764149e-1,
            ],
        ],
        [
            [
                8.225970136770599053645334e-1,
                -5.686106559014516482008454e-1,
                4.009374627528451661368239e-3,
            ],
            [
                5.686039884714725170949851e-1,
                8.226067844248355911296629e-1,
                2.753636244243962625766807e-3,
            ],
            [
                -4.863885680859272028955995e-3,
                1.461345322100335156123535e-5,
                9.999881711313041912353139e-1,
            ],
        ],
        [
            [
                8.174912494756424668095025e-1,
                -5.759269589518412775871757e-1,
                -4.024299098852162783725610e-3,
            ],
            [
                5.759202958137590985428480e-1,
                8.175011530231109814421774e-1,
                -2.770861901339942621246141e-3,
            ],
            [
                4.885683221935739563968598e-3,
                -5.252016960313770176736547e-5,
                9.999880635993045094526638e-1,
            ],
        ],
        [
            [
                -8.148851903351813783388271e-1,
                5.795677288787839701782037e-1,
                7.960792340186670966193252e-3,
            ],
            [
                -5.795408559447884783630241e-1,
                -8.149240721521785674131868e-1,
                5.581479883757473504410391e-3,
            ],
            [
                9.722286931434731191387357e-3,
                -6.533910940280420183060166e-5,
                9.999527353168366806789502e-1,
            ],
        ],
        [
            [
                -8.095640373733299350909152e-1,
                5.868130571528221680075287e-1,
                1.601578430025566669377666e-2,
            ],
            [
                -5.867097483050034164837246e-1,
                -8.097223047554166535141582e-1,
                1.102090855853988840274679e-2,
            ],
            [
                1.943555081990713091322811e-2,
                -4.744855475363895130622923e-4,
                9.998109992532548580613388e-1,
            ],
        ],
    ];

    /// The worst miss measured is 9.703e-14, two centuries out: the ERA's sum there is 201 turns,
    /// and half an ulp of it is 9e-14 rad. The bound is 3.1 times that.
    const CHAIN_BOUND: f64 = 3.0e-13;

    /// The transform is the IERS chain: polar motion after the Earth Rotation Angle after the
    /// precession-nutation of the offset pole, at six epochs. (G3)
    ///
    /// Each factor has its own gate — `X, Y, s` and the ERA in `spacelab_fa`, the CIO product and
    /// `Wᵀ` in `derivations/cio_rotation.mac` — so what this decides is the assembly: the order of
    /// the three rotations, UT1 for the ERA and TT for the rest, `dX` to `X` and `dY` to `Y` with
    /// the model `s`, and the TIO locator's sign and rate.
    #[test]
    fn the_transform_is_the_iers_chain_at_six_epochs() {
        let mut worst = (0.0_f64, 0.0_f64);
        for (((tt, ut1), eop), exact) in CHAIN_TT_DAYS
            .into_iter()
            .zip(CHAIN_UT1_DAYS)
            .zip(CHAIN_EOP)
            .zip(CHAIN_EXACT)
        {
            let [xp, yp, dx, dy] = eop.map(Angle::from_arcseconds);
            let got = gcrf_to_itrf(
                Epoch::<Ut1>::from_days_since_j2000(ut1),
                Epoch::<Tt>::from_days_since_j2000(tt),
                xp,
                yp,
                dx,
                dy,
                0.0,
            )
            .rotation
            .matrix()
            .rows;
            for i in 0..3 {
                for j in 0..3 {
                    let miss = (got[i][j] - exact[i][j]).abs();
                    if miss > worst.0 || miss.is_nan() {
                        worst = (miss, tt);
                    }
                }
            }
        }
        assert!(
            worst.0 < CHAIN_BOUND,
            "{:.3e} at TT day {}",
            worst.0,
            worst.1
        );
    }

    /// `(X, Y, s)` as `(mantissa, exponent)` pairs, and the GCRS→CIRS matrix
    /// `derivations/cio_rotation.mac` prints for them from the IERS 2010 Eq. 5.10 closed form at
    /// 40 digits. Three are of present-day size; the fourth has a 13° pole offset, where
    /// `1 − cos d` is not small.
    #[allow(
        clippy::excessive_precision,
        clippy::unreadable_literal,
        clippy::type_complexity
    )]
    const CASES: [([(i64, i32); 3], [[f64; 3]; 3]); 4] = [
        (
            [(3267445786254547, -62), (83886073, -41), (-3, -28)],
            [
                [
                    9.999997490036662075662940902308294757828e-1,
                    -2.337968720730797555589071507533160256693e-9,
                    -7.085143644134829655102491968485704371208e-4,
                ],
                [
                    -2.468970769826814621173186740365476844864e-8,
                    9.999999992724043570145490821597065699424e-1,
                    -3.81469615547534819656589023648159948284e-5,
                ],
                [
                    7.085143639871574038763268887919366534334e-4,
                    3.814696947301854379475116729736328125e-5,
                    9.999997482760706894809177225498985003667e-1,
                ],
            ],
        ),
        (
            [(-5170398013726189, -63), (5926573826723, -57), (7, -30)],
            [
                [
                    9.999998428774432826359728349371787270468e-1,
                    5.007262662632350061497212617046828718233e-9,
                    5.605756761777641967510716118577180160685e-4,
                ],
                [
                    1.804577767741396627773361421878739570767e-8,
                    9.999999991544137687001629531308148198092e-1,
                    -4.112386334276688596877642026386882934078e-5,
                ],
                [
                    -5.605756759096671088329899568236669438193e-4,
                    4.112386699730435912769621609186287969351e-5,
                    9.999998420318570938368575926194542511132e-1,
                ],
            ],
        ),
        (
            [(4194303, -34), (-8191, -38), (1, -26)],
            [
                [
                    9.999999701976912679923998188120656811557e-1,
                    -1.489752365994281962426863629701125739883e-8,
                    -2.44140566792783094427286361972940378166e-4,
                ],
                [
                    1.490479828366348044770445477108443349681e-8,
                    9.999999999999994450510981504628280659289e-1,
                    2.97950464309484875039011620351273689558e-8,
                ],
                [
                    2.441405667923390865325927734375e-4,
                    -2.979868440888822078704833984375e-8,
                    9.999999701976909350880995855852266371393e-1,
                ],
            ],
        ),
        (
            [(3, -4), (-1, -3), (1, -10)],
            [
                [
                    9.821808010302709914592395778520933726139e-1,
                    1.090258574362622238584617323041067803428e-2,
                    -1.876219808861373896285575893241191737738e-1,
                ],
                [
                    1.283059186223401473229082212068335630442e-2,
                    9.920968368179100681974465949815133559238e-1,
                    1.248168349557137906297511047122136737082e-1,
                ],
                [
                    1.875e-1,
                    -1.25e-1,
                    9.742785792574934776091885670970532064053e-1,
                ],
            ],
        ),
    ];

    fn value((mantissa, exponent): (i64, i32)) -> f64 {
        #[allow(clippy::cast_precision_loss)] // every mantissa has at most 53 bits
        let m = mantissa as f64;
        libm::ldexp(m, exponent)
    }

    /// (G1) The coded product `R₃(−(E+s))·R₂(d)·R₃(E)` is the IERS 2010 Eq. 5.10 closed form
    /// `R₃(−s)·[[1−aX², −aXY, X], [−aXY, 1−aY², Y], [−X, −Y, 1−a(X²+Y²)]]ᵀ`, `a = 1/(1+cos d)`:
    /// `derivations/cio_rotation.mac` proves the identity and prints these matrices, and
    /// `derivations/cio_rotation.gp` rebuilds them from the rotation product. The entries agree to
    /// 4.5e-16, four ulps of the unit diagonal; the largest gap measured is 1.1e-16, one ulp below 1.
    #[test]
    fn the_cio_rotation_is_the_iers_closed_form() {
        for (xys, want) in CASES {
            let [x, y, s] = xys.map(value);
            let got = gcrf_to_cirs(x, y, s).matrix().rows;
            for i in 0..3 {
                for j in 0..3 {
                    let gap = (got[i][j] - want[i][j]).abs();
                    assert!(
                        gap < 4.5e-16,
                        "(X, Y, s) = ({x}, {y}, {s}), entry ({i}, {j}): {} against {}",
                        got[i][j],
                        want[i][j]
                    );
                }
            }
        }
    }

    /// (G1) The CIP is the CIRS `z`-axis: `derivations/cio_rotation.mac` shows the GCRS→CIRS
    /// matrix's last row is `(X, Y, √(1 − X² − Y²))`, so the CIRS spin `(0, 0, ω)` that
    /// [`gcrf_to_itrf`] rotates back is `ω` times the pole the angles were built from.
    #[test]
    fn the_cirs_pole_is_the_cip_the_angles_name() {
        for (xys, _) in CASES {
            let [x, y, s] = xys.map(value);
            let pole = gcrf_to_cirs(x, y, s).inverse().matrix() * Vec3::new(0.0, 0.0, 1.0);
            let z = libm::sqrt(1.0 - x * x - y * y);
            for (got, want) in [(pole.x, x), (pole.y, y), (pole.z, z)] {
                assert!(
                    (got - want).abs() < 4.5e-16,
                    "(X, Y, s) = ({x}, {y}, {s}): {got} against {want}"
                );
            }
        }
    }
}
