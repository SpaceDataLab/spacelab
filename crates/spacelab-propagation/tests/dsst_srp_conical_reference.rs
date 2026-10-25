//! `dsst::srp`'s mean rates against the average that defines them, `(1/2π)∫νF dλ`: the cannonball
//! in a spherical Earth's conical shadow, taken at 80 digits by `derivations/dsst_srp_conical_average.gp`
//! along routes that share nothing with the crate — the contacts by Brent's method on the angles of
//! the two discs, `ν` from the textbook lens, `F` from exact derivatives of a Cartesian → equinoctial
//! conversion, double-exponential quadrature once in the true and once in the mean longitude.
//!
//! Six orbits: the Orekit LEO with its Sun of 2024-01-01; an `e = 0.73` HEO with the Sun in the
//! orbit plane; a GEO with the Sun 5° out of it; a low LEO in the penumbra alone, half the Sun hidden
//! at its deepest, and the same orbit `1e-6` short of the umbra cone; the Orekit LEO with the Sun on
//! the orbit normal, never in shadow. The gate is the largest difference of the six rates, `da` over
//! `a` so that all six are rates of angles, against the largest magnitude `(1/2π)∫|νFᵢ| dλ`: the
//! scale of the rounding of the shipped sum.

use spacelab_propagation::dsst::{DsstAuxiliary, DsstSolarRadiationPressure};

const MU: f64 = 3.986_004_415e14;
const AE: f64 = 6.378_136_46e6;
const CR: f64 = 1.5;
const AREA_OVER_MASS: f64 = 0.01;

/// `(name, [a, k, h, q, p], Sun (m), [da, dk, dh, dq, dp, dλ], magnitudes)`, printed by
/// `derivations/dsst_srp_conical_average.gp`, which `derivations/verify.sh` re-runs to check that
/// every row here is what it prints.
type Row = (&'static str, [f64; 5], [f64; 3], [f64; 6], [f64; 6]);

#[expect(clippy::unreadable_literal, reason = "printed by the script and checked verbatim")]
#[rustfmt::skip]
const REFERENCE: [Row; 6] = [
    ("leo", [7.2e6, 3.0901699437494747e-3, 9.510565162951536e-3, 8.54890329218965e-1, 7.69746709859868e-1], [2.4813057806275562e10, -1.3303312669583466e11, -5.7667965236002365e10], [-1.6615708385821207e-9, 6.4262324170324424e-12, -2.06625760143008e-12, 5.52583468572943e-13, 1.7167791003208047e-12, 2.9341760699808422e-12], [4.386e-5, 6.426e-12, 2.094e-12, 5.198e-12, 4.265e-12, 8.106e-12]),
    ("heo", [2.4396e7, 7e-1, 2.1e-1, 6.1e-2, 0.0], [1.4291630457520438e11, 4.38814085345744e10, 5.37352673419602e9], [1.6689064838557296e-6, -4.365971473730571e-12, 1.4114119261842459e-11, -1.00489574823795e-29, -2.967912848652489e-30, -1.8308965267898458e-11], [5.054e-4, 4.547e-12, 1.411e-11, 1.047e-29, 5.810e-30, 2.117e-11]),
    ("geo", [4.2164e7, 1e-3, -5e-4, 0.0, 0.0], [8.052018621905852e10, 1.2540276000821294e11, 1.3044934325039999e10], [8.329422677671144e-8, -2.723047643724931e-11, 1.7484503762850646e-11, -1.9453193736043643e-14, -3.329050828573109e-14, 1.7502960623664183e-12], [1.185e-3, 2.723e-11, 1.748e-11, 5.968e-13, 5.852e-13, 2.647e-11]),
    ("penumbra", [7.078137e6, 1e-3, -5e-4, 2e-1, -1e-1], [-5.440384899929082e10, 4.164670880424509e9, 1.3929249674154752e11], [5.092803905613422e-9, -5.30222892532885e-12, -2.4269444944719567e-12, 5.056815282940763e-14, -9.883069505232516e-14, 1.7191696197712684e-13], [3.553e-5, 5.302e-12, 2.427e-12, 2.701e-12, 2.649e-12, 4.229e-12]),
    ("graze", [7.078137e6, 1e-3, -5e-4, 2e-1, -1e-1], [-5.4535206342468445e10, 4.553954408493722e9, 1.3922893264836755e11], [8.781078271280105e-9, -5.267834278982385e-12, -2.4114463965772083e-12, 8.235086884423418e-14, -1.6775080381767337e-13, 2.9331328531667083e-13], [3.564e-5, 5.268e-12, 2.411e-12, 2.666e-12, 2.577e-12, 4.136e-12]),
    ("sunlit", [7.2e6, 3.0901699437494747e-3, 9.510565162951536e-3, 8.54890329218965e-1, 7.69746709859868e-1], [9.912634261097038e10, -1.1009095665298997e11, -2.0819999562341904e10], [-6.5484090091433e-103, -7.60673387466132e-16, 2.4715776598824463e-16, 4.950316485266417e-14, 1.5235507550511716e-13, 7.843330486971498e-14], [4.056e-11, 6.425e-14, 2.062e-14, 6.800e-12, 6.797e-12, 6.733e-12]),
];

/// Measured 4.0e-15 at worst, on the orbit in the penumbra alone, where the rule's nodes carry the
/// rounding of `ν` (`dsst::srp`'s `the_penumbra_rule_has_converged`); 1.0e-16 to 3.7e-16 on the
/// others.
const BOUND: f64 = 2e-14;

#[test]
fn the_rates_are_the_average_over_the_conical_shadow() {
    let srp = DsstSolarRadiationPressure::new(CR, AREA_OVER_MASS, MU, AE);
    let mut worst = 0.0_f64;
    for (name, [a, k, h, q, p], sun, want, magnitude) in REFERENCE {
        let got = srp.mean_element_rates(&DsstAuxiliary::new(a, k, h, q, p, 0.0, MU), sun);
        let weight = |i: usize| if i == 0 { 1.0 / a } else { 1.0 };
        let scale = (0..6).fold(0.0_f64, |m, i| m.max(weight(i) * magnitude[i]));
        let miss = (0..6).fold(0.0_f64, |m, i| m.max(weight(i) * (got[i] - want[i]).abs())) / scale;
        assert!(miss < BOUND, "{name}: {miss:.3e}, {got:?} against {want:?}");
        worst = worst.max(miss);
    }
    std::println!("conical SRP rates against the 80-digit average: worst {worst:.3e}");
}
