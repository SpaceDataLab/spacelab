//! `eclipse` against values that decide. The fractions against `derivations/eclipse_beta.gp`'s
//! 60-digit values on exact binary64 inputs, there taken as roots of the shadow conditions
//! themselves rather than from the closed forms; the shadow functions integrated over an orbit;
//! β against the plane of a propagated orbit; the seasons against a dense sampling of β.
//!
//! The cylinder's rows are `(r, β, f(|β| + δ), f(β), f(|β| − δ))`, `δ = 8u·tan β*`: at the onset
//! `f` is a square root of `β* − |β|`, so the gate is that the code's `f` is the exact one at a β
//! within `δ` of the input (G2, G3), and a few ulps of `f` away from the onset. The cones' rows are
//! `(r, D, β, umbra, penumbra, mean of 1 − ν)`. `derivations/eclipse_beta.gp`, which
//! `derivations/verify.sh` runs, checks that every row here is what it prints.

#![expect(
    clippy::expect_used,
    reason = "a refused input in a test is a failed test"
)]
#![expect(
    clippy::float_cmp,
    reason = "the rows are keyed by exact doubles, and a zero fraction or a window edge at the \
              span's end is exact by construction"
)]

use core::f64::consts::{FRAC_PI_2, PI, TAU};

use spacelab_core::{Angle, Vec3};
use spacelab_coverage::windows;
use spacelab_elements::{
    KeplerianElements, SUN_SYNCHRONOUS_NODE_RATE, sun_synchronous_inclination,
};
use spacelab_propagation::eclipse::{
    CircularOrbit, EclipseError, SUN_DIRECTION_RATE, beta_angle, beta_at, conical_eclipse,
    critical_beta, eclipse_fraction, eclipse_seasons,
};
use spacelab_propagation::ephemeris::{AnalyticSun, Ephemeris};
use spacelab_propagation::shadow::lighting_ratio;

const RE: f64 = 6_378_137.0;
const MU: f64 = 3.986_004_418e14;
/// EGM2008's `J₂`, the value the bridges' tests use.
const J2: f64 = 1.082_626_68e-3;
const AU: f64 = 1.495_978_707e11;
/// 2026-01-01 00:00 TT, seconds past J2000 (2000-01-01 12:00 TT): 9496.5 days.
const EPOCH_2026: f64 = 9_496.5 * 86_400.0;
const DAY: f64 = 86_400.0;
const U: f64 = f64::EPSILON / 2.0;

/// The spacing of the doubles at `x`.
fn ulp(x: f64) -> f64 {
    let a = x.abs();
    f64::from_bits(a.to_bits() + 1) - a
}

type CylinderRow = (f64, f64, f64, f64, f64);

#[expect(clippy::unreadable_literal, reason = "printed by the script and checked verbatim")]
#[rustfmt::skip]
const CYLINDER: [CylinderRow; 70] = [
    (6.778137e6, 0.0, 3.900996182292888e-1, 3.900996182292888e-1, 3.900996182292888e-1),
    (6.778137e6, 1e-1, 3.89524605786078e-1, 3.8952460578607806e-1, 3.8952460578607806e-1),
    (6.778137e6, 5e-1, 3.739757612390439e-1, 3.7397576123904414e-1, 3.739757612390443e-1),
    (6.778137e6, 1e0, 2.84529129750007e-1, 2.8452912975000794e-1, 2.8452912975000894e-1),
    (6.778137e6, -7e-1, 3.5409130518615023e-1, 3.5409130518615056e-1, 3.540913051861509e-1),
    (6.778137e6, 1.102980685317585e0, 2.2979703092910927e-1, 2.2979703092911102e-1, 2.297970309291128e-1),
    (6.778137e6, 1.2245340947973167e0, 2.370676599128243e-2, 2.3706765991311628e-2, 2.3706765991340827e-2),
    (6.778137e6, 1.2255330947973166e0, 7.506081556600473e-4, 7.506081565868339e-4, 7.506081575136207e-4),
    (6.778137e6, 1.2255340937973165e0, 2.3736315364713752e-5, 2.373634467241012e-5, 2.3736373980070305e-5),
    (6.778137e6, 1.2255340947963165e0, 7.497162693405798e-7, 7.506435897462332e-7, 7.515697659833488e-7),
    (6.778137e6, 1.2255340947973166e0, 0.0, 1.3036476533763302e-9, 3.732310022446552e-8),
    (6.778137e6, 1.2255340947983167e0, 0.0, 0.0, 0.0),
    (6.778137e6, 1.2255350947973165e0, 0.0, 0.0, 0.0),
    (6.778137e6, 1.5e0, 0.0, 0.0, 0.0),
    (7.078137e6, 0.0, 3.5724196769119426e-1, 3.5724196769119426e-1, 3.5724196769119426e-1),
    (7.078137e6, 1e-1, 3.564724759372249e-1, 3.56472475937225e-1, 3.56472475937225e-1),
    (7.078137e6, 5e-1, 3.355019825565965e-1, 3.355019825565967e-1, 3.3550198255659686e-1),
    (7.078137e6, 1e0, 2.034923105773906e-1, 2.034923105773918e-1, 2.0349231057739306e-1),
    (7.078137e6, -7e-1, 3.0813543372211716e-1, 3.081354337221175e-1, 3.081354337221179e-1),
    (7.078137e6, 1.0100778671273565e0, 1.9654007382073194e-1, 1.9654007382073327e-1, 1.9654007382073457e-1),
    (7.078137e6, 1.1213087412526184e0, 2.0501159144379225e-2, 2.050115914439811e-2, 2.0501159144416994e-2),
    (7.078137e6, 1.1223077412526183e0, 6.489420295595384e-4, 6.48942030158443e-4, 6.489420307573476e-4),
    (7.078137e6, 1.1223087402526182e0, 2.0521350392909053e-5, 2.0521369332001267e-5, 2.0521388271076005e-5),
    (7.078137e6, 1.1223087412516182e0, 6.48353889052468e-7, 6.489530624831192e-7, 6.495516832115658e-7),
    (7.078137e6, 1.1223087412526183e0, 0.0, 0.0, 2.744743280596738e-8),
    (7.078137e6, 1.1223087412536183e0, 0.0, 0.0, 0.0),
    (7.078137e6, 1.1223097412526182e0, 0.0, 0.0, 0.0),
    (7.078137e6, 1.5e0, 0.0, 0.0, 0.0),
    (8.378137e6, 0.0, 2.7543041442127886e-1, 2.7543041442127886e-1, 2.7543041442127886e-1),
    (8.378137e6, 1e-1, 2.7406665656379314e-1, 2.7406665656379314e-1, 2.740666565637932e-1),
    (8.378137e6, 5e-1, 2.353598450464426e-1, 2.353598450464428e-1, 2.35359845046443e-1),
    (8.378137e6, 1e0, 0.0, 0.0, 0.0),
    (8.378137e6, -7e-1, 1.7793770442038945e-1, 1.779377044203899e-1, 1.7793770442039034e-1),
    (8.378137e6, 7.787611498689736e-1, 1.353291743158268e-1, 1.3532917431582753e-1, 1.3532917431582825e-1),
    (8.378137e6, 8.642901665210818e-1, 1.5413653198032293e-2, 1.541365319804032e-2, 1.5413653198048344e-2),
    (8.378137e6, 8.652891665210818e-1, 4.8776436118143406e-4, 4.87764361435748e-4, 4.877643616900619e-4),
    (8.378137e6, 8.652901655210818e-1, 1.5424466204936042e-5, 1.5424474247067988e-5, 1.542448228919574e-5),
    (8.378137e6, 8.652901665200818e-1, 4.875115869396148e-7, 4.877659670893117e-7, 4.880202146435754e-7),
    (8.378137e6, 8.652901665210818e-1, 0.0, 2.548599800237589e-9, 1.595577046501781e-8),
    (8.378137e6, 8.652901665220818e-1, 0.0, 0.0, 0.0),
    (8.378137e6, 8.652911665210818e-1, 0.0, 0.0, 0.0),
    (8.378137e6, 1.5e0, 0.0, 0.0, 0.0),
    (4.2164137e7, 0.0, 4.833604493327078e-2, 4.833604493327078e-2, 4.833604493327078e-2),
    (4.2164137e7, 1e-1, 3.6436156921189386e-2, 3.643615692118943e-2, 3.643615692118946e-2),
    (4.2164137e7, 5e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1e0, 0.0, 0.0, 0.0),
    (4.2164137e7, -7e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.366669472993547e-1, 2.113519124875693e-2, 2.113519124875702e-2, 2.113519124875711e-2),
    (4.2164137e7, 1.5085216366594964e-1, 5.559240204593432e-3, 5.559240204593808e-3, 5.5592402045941845e-3),
    (4.2164137e7, 1.5185116366594964e-1, 1.76097419093168e-4, 1.7609741910513537e-4, 1.7609741911710274e-4),
    (4.2164137e7, 1.5185216266594964e-1, 5.568698396839236e-6, 5.56869877528284e-6, 5.5686991537264175e-6),
    (4.2164137e7, 1.5185216366494964e-1, 1.7608590576397094e-7, 1.7609787359722134e-7, 1.7610984061717802e-7),
    (4.2164137e7, 1.5185216366594964e-1, 0.0, 0.0, 2.0236014470397915e-9),
    (4.2164137e7, 1.5185216366694965e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.5185316366594964e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.5e0, 0.0, 0.0, 0.0),
    (6.378138e6, 0.0, 4.9982175466392714e-1, 4.9982175466392714e-1, 4.9982175466392714e-1),
    (6.378138e6, 1e-1, 4.998208597085452e-1, 4.9982085970854523e-1, 4.9982085970854523e-1),
    (6.378138e6, 5e-1, 4.9979689051880294e-1, 4.997968905188031e-1, 4.9979689051880327e-1),
    (6.378138e6, 1e0, 4.996701006885735e-1, 4.996701006885743e-1, 4.996701006885751e-1),
    (6.378138e6, -7e-1, 4.9976695147620614e-1, 4.997669514762064e-1, 4.9976695147620676e-1),
    (6.378138e6, 1.4132127173008981e0, 4.988641866646727e-1, 4.98864186664684e-1, 4.988641866646954e-1),
    (6.378138e6, 1.5692363525565536e0, 3.83130115451566e-1, 3.8313011557603754e-1, 3.831301157005091e-1),
    (6.378138e6, 1.5702353525565536e0, 1.900893719191306e-2, 1.900895224462344e-2, 1.9008967297321832e-2),
    (6.378138e6, 1.5702363515565534e0, 6.010852762146542e-4, 6.015625356508218e-4, 6.02039416743236e-4),
    (6.378138e6, 1.5702363525555534e0, 0.0, 1.9024280650881715e-5, 3.0592482324968784e-5),
    (6.378138e6, 1.5702363525565535e0, 0.0, 1.1447215015697922e-7, 2.395808477532525e-5),
    (6.378138e6, 1.5702363525575536e0, 0.0, 0.0, 1.4562955617681336e-5),
    (6.378138e6, 1.5702373525565534e0, 0.0, 0.0, 0.0),
    (6.378138e6, 1.5e0, 4.9748014928070605e-1, 4.9748014928076245e-1, 4.974801492808188e-1),
];

type ConeRow = (f64, f64, f64, f64, f64, f64);

#[expect(clippy::unreadable_literal, reason = "printed by the script and checked verbatim")]
#[rustfmt::skip]
const CONE: [ConeRow; 42] = [
    (7.078137e6, 1.495978707e11, 0.0, 3.557752440059739e-1, 3.587358340891776e-1, 3.5725477221867735e-1),
    (7.078137e6, 1.495978707e11, 5e-1, 3.33768973083297e-1, 3.3726550382414777e-1, 3.355169119885147e-1),
    (7.078137e6, 1.495978707e11, 1.1176008928982837e0, 6.450782021419275e-3, 6.271652213344639e-2, 4.2856795071256525e-2),
    (7.078137e6, 1.495978707e11, 1.1200048170754509e0, 0.0, 5.423462379155847e-2, 2.756922836984123e-2),
    (7.078137e6, 1.495978707e11, 1.1223087412526183e0, 0.0, 4.451797606793564e-2, 1.3723770801338171e-2),
    (7.078137e6, 1.495978707e11, 1.1246553011032927e0, 0.0, 3.1552092522877315e-2, 3.728169981863123e-3),
    (7.078137e6, 1.495978707e11, 1.1271018609539674e0, 0.0, 0.0, 0.0),
    (7.078137e6, 1.471e11, 0.0, 3.5575033773328735e-1, 3.5876120127614586e-1, 3.572549764101105e-1),
    (7.078137e6, 1.471e11, 5e-1, 3.3373953177952104e-1, 3.372954365824592e-1, 3.355171465313225e-1),
    (7.078137e6, 1.471e11, 1.1175226475349833e0, 6.450140965499245e-3, 6.323779073270244e-2, 4.320652473750331e-2),
    (7.078137e6, 1.471e11, 1.1199656943938008e0, 0.0, 5.469233745383672e-2, 2.780115251368061e-2),
    (7.078137e6, 1.471e11, 1.1223087412526183e0, 0.0, 4.489541282946111e-2, 1.3839718307952059e-2),
    (7.078137e6, 1.471e11, 1.1246951477874036e0, 0.0, 3.182085790391864e-2, 3.759806127463068e-3),
    (7.078137e6, 1.471e11, 1.127181554322189e0, 0.0, 0.0, 0.0),
    (7.078137e6, 1.521e11, 0.0, 3.5579937259009126e-1, 3.587112589829715e-1, 3.572545739813617e-1),
    (7.078137e6, 1.521e11, 5e-1, 3.3379749466898606e-1, 3.3723650528046184e-1, 3.355166841650217e-1),
    (7.078137e6, 1.521e11, 1.1176766950808883e0, 6.451403196334647e-3, 6.2207257674649005e-2, 4.251513433747083e-2),
    (7.078137e6, 1.521e11, 1.1200427181667532e0, 0.0, 5.378746117572008e-2, 2.734263240500913e-2),
    (7.078137e6, 1.521e11, 1.1223087412526183e0, 0.0, 4.4149281610114735e-2, 1.3610499087404226e-2),
    (7.078137e6, 1.521e11, 1.1246166986167336e0, 0.0, 3.128958290093558e-2, 3.697267284317477e-3),
    (7.078137e6, 1.521e11, 1.127024655980849e0, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.495978707e11, 0.0, 4.686932124805037e-2, 4.982991133125411e-2, 4.834395032549077e-2),
    (4.2164137e7, 1.495978707e11, 5e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.495978707e11, 1.4714431531161504e-1, 1.7333481338631179e-3, 1.7069736857812393e-2, 1.1579155115752235e-2),
    (4.2164137e7, 1.495978707e11, 1.4954823948878235e-1, 0.0, 1.4786386019651985e-2, 7.453171394638736e-3),
    (4.2164137e7, 1.495978707e11, 1.5185216366594964e-1, 0.0, 1.2156669331373373e-2, 3.7108928562903432e-3),
    (4.2164137e7, 1.495978707e11, 1.5419872351662417e-1, 0.0, 8.629747280716147e-3, 1.008133669240964e-3),
    (4.2164137e7, 1.495978707e11, 1.5664528336729872e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.471e11, 0.0, 4.684441497536385e-2, 4.985527851822236e-2, 4.834398672486149e-2),
    (4.2164137e7, 1.471e11, 5e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.471e11, 1.4706606994831475e-1, 1.7328806506928026e-3, 1.7212502164701732e-2, 1.1672772104021739e-2),
    (4.2164137e7, 1.471e11, 1.495091168071322e-1, 0.0, 1.4912359473982109e-2, 7.515385625543358e-3),
    (4.2164137e7, 1.471e11, 1.5185216366594964e-1, 0.0, 1.226103436313929e-2, 3.742018135198814e-3),
    (4.2164137e7, 1.471e11, 1.54238570200735e-1, 0.0, 8.70440107751899e-3, 1.016627586278417e-3),
    (4.2164137e7, 1.471e11, 1.5672497673552038e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.521e11, 0.0, 4.6893449832167755e-2, 4.980533622504804e-2, 4.83439119465068e-2),
    (4.2164137e7, 1.521e11, 5e-1, 0.0, 0.0, 0.0),
    (4.2164137e7, 1.521e11, 1.4722011749421976e-1, 1.733800910187857e-3, 1.693028052257628e-2, 1.1487676813434822e-2),
    (4.2164137e7, 1.521e11, 1.4958614058008468e-1, 0.0, 1.4663345046242147e-2, 7.392374556076282e-3),
    (4.2164137e7, 1.521e11, 1.5185216366594964e-1, 0.0, 1.2054752719627776e-2, 3.6804803656195944e-3),
    (4.2164137e7, 1.521e11, 1.5416012103006507e-1, 0.0, 8.556858234995798e-3, 9.998352027786527e-4),
    (4.2164137e7, 1.521e11, 1.5656807839418052e-1, 0.0, 0.0, 0.0),
];

/// The code's `f` lies in the band `[f(|β| + δ), f(|β| − δ)]` of every row, widened by four ulps
/// of `f`: it is the exact fraction at a β within `δ = 8u·tan β*` of the input. Away from the onset
/// the band is under an ulp and the four ulps decide; within `10⁻⁹` of it the band does. (G2, G3)
#[test]
fn the_fraction_is_exact_at_a_beta_within_its_band() {
    let mut worst_ulps = 0.0_f64;
    for (r, beta, lo, f, hi) in CYLINDER {
        let got = eclipse_fraction(r, RE, beta).expect("in the domain");
        let slack = 4.0 * ulp(hi);
        assert!(
            got >= lo - slack && got <= hi + slack,
            "r = {r}, beta = {beta}: f = {got:e}, outside [{lo:e}, {hi:e}] (exact {f:e})"
        );
        if hi - lo <= ulp(f) {
            worst_ulps = worst_ulps.max((got - f).abs() / ulp(f));
        }
    }
    println!("away from the onset, the worst f is {worst_ulps} ulp from the 60-digit value");
}

/// `β = 0` gives `asin(R/r)/π`, which `eclipse_beta.gp` shows the row's `f` is, to 2 ulps; at and
/// above `β*` the fraction is exactly 0, and `β*` is the row's (64.30° at 700 km). (G3)
#[test]
fn beta_zero_gives_asin_r_over_r_over_pi_and_beta_star_gives_zero() {
    for (r, beta, lo, f, hi) in CYLINDER {
        let got = eclipse_fraction(r, RE, beta).expect("in the domain");
        if beta == 0.0 {
            assert!(
                (got - f).abs() <= 2.0 * ulp(f),
                "r = {r}: f(0) = {got:e}, asin(R/r)/pi = {f:e}"
            );
        }
        if hi == 0.0 {
            assert!(
                got == 0.0 && lo == 0.0,
                "r = {r}, beta = {beta}: f = {got:e} above beta*"
            );
        }
    }
    for r in [
        6_778_137.0,
        7_078_137.0,
        8_378_137.0,
        42_164_137.0,
        6_378_138.0,
    ] {
        let star = critical_beta(r, RE).expect("above the surface");
        // The rows' eleventh β of each radius is β* rounded to the nearest double.
        let row = CYLINDER
            .iter()
            .find(|row| row.0 == r)
            .map(|row| row.1)
            .expect("a row");
        let tabled = CYLINDER
            .iter()
            .filter(|row| row.0 == r)
            .nth(10)
            .map(|row| row.1)
            .expect("eleven rows");
        assert!(
            row == 0.0 && (star - tabled).abs() <= ulp(tabled),
            "r = {r}: beta* = {star}"
        );
        for b in [
            star * (1.0 + 4.0 * U),
            FRAC_PI_2,
            -FRAC_PI_2,
            -star * (1.0 + 4.0 * U),
        ] {
            assert_eq!(eclipse_fraction(r, RE, b).expect("in the domain"), 0.0);
        }
    }
    let star = critical_beta(RE + 700_000.0, RE)
        .expect("above the surface")
        .to_degrees();
    assert!(
        (star - 64.30).abs() < 0.005,
        "beta* at 700 km is {star} degrees"
    );
}

/// The cones' arcs against the roots of `lighting_ratio`'s two conditions and the mean of `1 − ν`
/// against `intnum` of the textbook lens, at 60 digits. An arc's first-order error from `8u` in
/// its onset's sine is `8u/(π tan(πf) cos²β)`, which grows at the onset as the cylinder's does.
/// (G2)
#[test]
fn the_cones_hold_their_60_digit_arcs_and_mean() {
    let mut worst_mean = 0.0_f64;
    for (r, d, beta, umbra, penumbra, mean) in CONE {
        let got = conical_eclipse(r, RE, d, beta).expect("in the domain");
        for (name, value, exact) in [
            ("umbra", got.umbra, umbra),
            ("penumbra", got.penumbra, penumbra),
        ] {
            let allowance = if exact == 0.0 {
                0.0
            } else {
                4.0 * ulp(exact) + 8.0 * U / (PI * (PI * exact).tan() * beta.cos().powi(2))
            };
            assert!(
                (value - exact).abs() <= allowance,
                "r = {r}, D = {d}, beta = {beta}: {name} {value:e} against {exact:e}"
            );
        }
        worst_mean = worst_mean.max((got.mean_shadow - mean).abs());
        assert!(got.umbra <= got.mean_shadow && got.mean_shadow <= got.penumbra);
    }
    println!("the worst mean of 1 - nu is {worst_mean:e} from the 60-digit value");
    assert!(worst_mean < 2e-15, "mean of 1 - nu off by {worst_mean:e}");
}

/// The satellite at `v` from the anti-Sun point and the Sun at `d` on `+x`, `β` above the plane.
fn satellite(r: f64, v: f64) -> [f64; 3] {
    [-r * v.cos(), r * v.sin(), 0.0]
}
fn sun_at(d: f64, beta: f64) -> [f64; 3] {
    [d * beta.cos(), 0.0, d * beta.sin()]
}

/// The cylinder's shadow test of `PanelSrp`: behind the body and within its radius of the axis.
fn in_cylinder(r: f64, beta: f64, v: f64) -> bool {
    let p = satellite(r, v);
    let along = p[0] * beta.cos() + p[2] * beta.sin();
    along < 0.0 && r * r - along * along < RE * RE
}

/// The edge of a set on `[0, π]` that holds at 0, by bisection on its indicator.
fn edge(inside: impl Fn(f64) -> bool) -> f64 {
    if !inside(0.0) {
        return 0.0;
    }
    if inside(PI) {
        return PI;
    }
    let (mut a, mut b) = (0.0_f64, PI);
    for _ in 0..80 {
        let m = f64::midpoint(a, b);
        if inside(m) { a = m } else { b = m }
    }
    a
}

/// The midpoint sum of `g` over `[0, 2π)` at `n` points, over `2π`.
fn orbit_mean(n: u32, g: impl Fn(f64) -> f64) -> f64 {
    let h = TAU / f64::from(n);
    (0..n).map(|k| g((f64::from(k) + 0.5) * h)).sum::<f64>() / f64::from(n)
}

/// The cylinder's fraction is its shadow test integrated over an orbit: a midpoint sum of the
/// indicator at 2¹⁸ points, within its two cells, and the edge by bisection on it. (S1)
#[test]
fn the_cylinder_fraction_is_its_shadow_test_integrated_over_an_orbit() {
    let n = 1_u32 << 18;
    let (mut worst_sum, mut worst_edge) = (0.0_f64, 0.0_f64);
    for r in [6_778_137.0, 7_078_137.0, 42_164_137.0] {
        let star = critical_beta(r, RE).expect("above the surface");
        for beta in [0.0, 0.3 * star, -0.6 * star, 0.95 * star] {
            let f = eclipse_fraction(r, RE, beta).expect("in the domain");
            let summed = orbit_mean(n, |v| f64::from(u8::from(in_cylinder(r, beta, v))));
            assert!(
                (summed - f).abs() <= 2.0 / f64::from(n),
                "r = {r}, beta = {beta}: {summed} vs {f}"
            );
            let bisected = edge(|v| in_cylinder(r, beta, v)) / PI;
            assert!(
                (bisected - f).abs() < 1e-13,
                "r = {r}, beta = {beta}: {bisected} vs {f}"
            );
            worst_sum = worst_sum.max((summed - f).abs());
            worst_edge = worst_edge.max((bisected - f).abs());
        }
    }
    println!("the summed indicator is {worst_sum:e} and the bisected edge {worst_edge:e} from f");
}

/// The cones are `lighting_ratio` integrated over an orbit: the arcs where it is 0 and below 1 by
/// bisection on it, the mean of `1 − ν` by a midpoint sum at 2¹⁸ points. (S1)
#[test]
fn the_cones_are_lighting_ratio_integrated_over_an_orbit() {
    let n = 1_u32 << 18;
    let r = RE + 700_000.0;
    let star = critical_beta(r, RE).expect("above the surface");
    let cone = conical_eclipse(r, RE, AU, 0.0).expect("in the domain");
    let (mut worst_edge, mut worst_mean) = (0.0_f64, 0.0_f64);
    for beta in [
        0.0,
        0.5,
        cone.umbra_onset - 1e-3,
        f64::midpoint(star, cone.penumbra_onset),
    ] {
        let got = conical_eclipse(r, RE, AU, beta).expect("in the domain");
        let nu = |v: f64| lighting_ratio(satellite(r, v), sun_at(AU, beta), RE);
        let umbra = edge(|v| nu(v) == 0.0) / PI;
        let penumbra = edge(|v| nu(v) < 1.0) / PI;
        let mean = orbit_mean(n, |v| 1.0 - nu(v));
        // `ν` leaves 0 and 1 as the 3/2 power of the depth, so it rounds to 0 or 1 within about
        // `u^(2/3)` of the Sun's radius past each contact: 1e-13 rad of separation, and 1e-12 of
        // a revolution where the arc's end moves slowest, 1e-3 rad inside the umbra's onset.
        assert!(
            (umbra - got.umbra).abs() < 1e-10,
            "beta = {beta}: umbra {umbra} vs {}",
            got.umbra
        );
        assert!(
            (penumbra - got.penumbra).abs() < 1e-10,
            "beta = {beta}: penumbra {penumbra} vs {}",
            got.penumbra
        );
        assert!(
            (mean - got.mean_shadow).abs() < 1e-9,
            "beta = {beta}: mean {mean} vs {}",
            got.mean_shadow
        );
        worst_edge = worst_edge
            .max((umbra - got.umbra).abs())
            .max((penumbra - got.penumbra).abs());
        worst_mean = worst_mean.max((mean - got.mean_shadow).abs());
    }
    println!(
        "the bisected arcs are {worst_edge:e} and the summed mean {worst_mean:e} from the cones'"
    );
    // Ten metres above the surface with the Sun on the orbit normal the orbit is the terminator:
    // the Sun is half below the horizon everywhere, so the whole revolution is in the penumbra.
    let low = RE + 10.0;
    let got = conical_eclipse(low, RE, AU, FRAC_PI_2).expect("in the domain");
    let mean = orbit_mean(n, |v| {
        1.0 - lighting_ratio(satellite(low, v), sun_at(AU, FRAC_PI_2), RE)
    });
    assert!(got.penumbra == 1.0 && got.umbra == 0.0, "{got:?}");
    assert!(
        (mean - got.mean_shadow).abs() < 1e-9,
        "mean {mean} vs {}",
        got.mean_shadow
    );
    // Beyond the umbra's vertex, 1.38e9 m behind the Earth, there is no umbra.
    let far = conical_eclipse(2e9, RE, AU, 0.0).expect("in the domain");
    assert!(
        far.umbra == 0.0 && far.umbra_onset < 0.0 && far.penumbra > 0.0,
        "{far:?}"
    );
}

/// β is the Sun's elevation above the plane of an orbit propagated by the two-body conversion of
/// `spacelab-elements`: `ĥ = r × v/|r × v|` at four points of each orbit. (S1)
#[test]
fn beta_is_the_sun_elevation_above_a_propagated_orbit_plane() {
    let sun = AnalyticSun {
        epoch_j2000_seconds: EPOCH_2026,
    };
    let s = sun.position(0.0);
    let s_hat = Vec3::new(s[0], s[1], s[2]).scale(1.0 / Vec3::new(s[0], s[1], s[2]).norm());
    let mut worst = 0.0_f64;
    for (inc, raan) in [
        (0.3, 1.0),
        (1.7, 4.0),
        (98.2_f64.to_radians(), 3.3),
        (0.0, 0.0),
        (PI, 2.0),
    ] {
        let want = beta_angle(raan, inc, s).expect("a plane");
        for u in [0.0, 1.0, 2.5, 4.0] {
            let elements = KeplerianElements {
                semi_major_axis: RE + 700_000.0,
                eccentricity: 0.0,
                inclination: Angle::from_radians(inc),
                raan: Angle::from_radians(raan),
                arg_periapsis: Angle::from_radians(0.0),
                true_anomaly: Angle::from_radians(u),
                mu: MU,
            };
            let (r, v) = elements.to_state();
            let h = r.cross(v);
            let h_hat = h.scale(1.0 / h.norm());
            let got = h_hat.dot(s_hat).asin();
            assert!(
                (got - want).abs() < 1e-14,
                "i = {inc}, raan = {raan}, u = {u}: {got} vs {want}"
            );
            worst = worst.max((got - want).abs());
        }
    }
    println!("beta is {worst:e} rad from the propagated plane's");
}

/// At the epoch β is [`beta_angle`] to the bit, and at J2000, where the pole of date is the J2000
/// pole, the node turns about it: β at `t` is the β of the plane with node `Ω + Ω̇t`. (G2)
#[test]
fn beta_at_turns_the_node_about_the_pole() {
    let orbit = CircularOrbit {
        radius: RE + 700_000.0,
        inclination: 1.2,
        raan: 0.4,
        node_rate: -1.1e-6,
        period: 5_926.0,
    };
    for epoch in [0.0, EPOCH_2026] {
        let sun = AnalyticSun {
            epoch_j2000_seconds: epoch,
        };
        let at_epoch = beta_at(&orbit, sun, 0.0).expect("in the span");
        assert_eq!(
            at_epoch,
            beta_angle(orbit.raan, orbit.inclination, sun.position(0.0)).expect("a plane")
        );
    }
    let sun = AnalyticSun {
        epoch_j2000_seconds: 0.0,
    };
    for t in [1.0, 3.7 * DAY, 40.0 * DAY, 300.0 * DAY] {
        let got = beta_at(&orbit, sun, t).expect("in the span");
        let want = beta_angle(
            orbit.raan + orbit.node_rate * t,
            orbit.inclination,
            sun.position(t),
        )
        .expect("a plane");
        assert!((got - want).abs() < 1e-13, "t = {t}: {got} vs {want}");
    }
}

/// The analytic Sun turns no faster than [`SUN_DIRECTION_RATE`]: central differences one day apart
/// over the two centuries about J2000, and within 0.3 % of it, at perihelion.
#[test]
fn the_sun_turns_no_faster_than_its_bound() {
    let sun = AnalyticSun {
        epoch_j2000_seconds: 0.0,
    };
    let unit = |t: f64| {
        let p = sun.position(t);
        let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        [p[0] / n, p[1] / n, p[2] / n]
    };
    let h = 600.0;
    let mut fastest = 0.0_f64;
    let days = 36_525 - 1;
    for k in -days..=days {
        let t = f64::from(k) * DAY;
        let (a, b) = (unit(t - h), unit(t + h));
        let rate = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
            / (2.0 * h);
        fastest = fastest.max(rate);
    }
    println!("fastest {fastest:e} rad/s, bound {SUN_DIRECTION_RATE:e}");
    assert!((0.997 * SUN_DIRECTION_RATE..=SUN_DIRECTION_RATE).contains(&fastest));
}

/// A circular orbit at `altitude`, inclination `inc` and node `raan`, its node at the `J₂` secular
/// rate and its period the draconitic one, both from `spacelab-elements`.
fn j2_orbit(altitude: f64, inc: f64, raan: f64) -> CircularOrbit {
    let elements = KeplerianElements {
        semi_major_axis: RE + altitude,
        eccentricity: 0.0,
        inclination: Angle::from_radians(inc),
        raan: Angle::from_radians(raan),
        arg_periapsis: Angle::from_radians(0.0),
        true_anomaly: Angle::from_radians(0.0),
        mu: MU,
    };
    CircularOrbit {
        radius: RE + altitude,
        inclination: inc,
        raan,
        node_rate: elements.nodal_regression_rate(RE, J2),
        period: TAU / elements.draconitic_mean_motion(RE, J2),
    }
}

/// An equatorial orbit's β is the Sun's declination, never above 23.44°, so a year at 700 km is
/// one season; β changes sign at the equinoxes, so the longest eclipse is `asin(R/r)/π·T`. A
/// geostationary orbit's `β* = 8.70°` gives two seasons, each about its equinox, and the same
/// longest eclipse, 69.4 min. (G3)
#[test]
fn equatorial_orbits_have_their_seasons_in_closed_form() {
    let sun = AnalyticSun {
        epoch_j2000_seconds: EPOCH_2026,
    };
    let year = 365.0 * DAY;
    let low = CircularOrbit {
        radius: RE + 700_000.0,
        inclination: 0.0,
        raan: 0.0,
        node_rate: 0.0,
        period: 5_926.0,
    };
    let got = eclipse_seasons(&low, RE, sun, 0.0, year, 1.0).expect("in the domain");
    assert!(got.seasons.len() == 1 && got.seasons[0].rise == 0.0 && got.seasons[0].set == year);
    assert_eq!(got.full_sun, []);
    let longest = (RE / low.radius).asin() / PI * low.period;
    assert!(got.least_beta.lower == 0.0 && got.least_beta.upper == 0.0);
    assert!((got.longest_eclipse.upper - longest).abs() <= 4.0 * ulp(longest));
    assert_eq!(got.longest_eclipse.lower, got.longest_eclipse.upper);

    let geo = CircularOrbit {
        radius: 42_164_137.0,
        inclination: 0.0,
        raan: 0.0,
        node_rate: 0.0,
        period: 86_164.090_5,
    };
    let got = eclipse_seasons(&geo, RE, sun, 0.0, year, 1.0).expect("in the domain");
    // The equinoxes of 2026: 03-20 14:46 and 09-23 00:05 UTC, days 78.6 and 265.0 of the year.
    assert!(got.seasons.len() == 2 && got.full_sun.len() == 3, "{got:?}");
    for (season, equinox) in got.seasons.iter().zip([78.6 * DAY, 265.0 * DAY]) {
        let days = season.duration() / DAY;
        assert!(
            season.contains(equinox) && (40.0..50.0).contains(&days),
            "{season:?}: {days} days"
        );
    }
    let longest = (RE / geo.radius).asin() / PI * geo.period;
    assert!((got.longest_eclipse.upper - longest).abs() <= 4.0 * ulp(longest));
    assert!(
        (longest / 60.0 - 69.4).abs() < 0.05,
        "{} min",
        longest / 60.0
    );
}

/// `ρ − |sin β|` sampled every ten minutes: every sign change between two samples holds exactly
/// one season edge, every edge is such a change, and the seasons and the full-sun intervals tile
/// the window. The least `|β|` encloses the least sample's from below and the longest eclipse
/// holds the sampled one. Four orbits: a dawn–dusk and a 10:30 sun-synchronous at 700 km, one at
/// 51.6° and 420 km whose node turns 5° a day and whose β changes sign, and one at 55° and 1200 km
/// with two full-sun intervals of about ten days. (S1)
#[test]
fn the_seasons_are_the_sign_changes_of_a_dense_sampling() {
    let sun = AnalyticSun {
        epoch_j2000_seconds: EPOCH_2026,
    };
    let sun_ra = {
        let s = sun.position(0.0);
        s[1].atan2(s[0])
    };
    let sso =
        sun_synchronous_inclination(RE + 700_000.0, 0.0, MU, RE, J2, SUN_SYNCHRONOUS_NODE_RATE)
            .expect("700 km has a sun-synchronous inclination")
            .radians();
    let cases = [
        (
            "dawn-dusk",
            j2_orbit(700_000.0, sso, sun_ra - FRAC_PI_2),
            365,
        ),
        (
            "10:30",
            j2_orbit(700_000.0, sso, sun_ra - 1.5 * 15_f64.to_radians()),
            365,
        ),
        (
            "51.6 degrees",
            j2_orbit(420_000.0, 51.6_f64.to_radians(), 1.0),
            120,
        ),
        (
            "55 degrees",
            j2_orbit(1_200_000.0, 55_f64.to_radians(), 1.0),
            120,
        ),
    ];
    for (name, orbit, days) in cases {
        matches_a_dense_sampling(name, &orbit, sun, days);
    }
}

/// `eclipse_seasons` over `days` against `β` sampled every ten minutes: one edge per sign change
/// of `ρ − |sin β|`, inside its sample interval; the seasons and the full-sun intervals tiling the
/// span; the least `|β|` and the longest eclipse enclosing the sampled ones.
fn matches_a_dense_sampling(name: &str, orbit: &CircularOrbit, sun: AnalyticSun, days: u32) {
    let tol = 1.0;
    let span = f64::from(days) * DAY;
    let got = eclipse_seasons(orbit, RE, sun, 0.0, span, tol).expect("in the domain");
    let rho = RE / orbit.radius;
    let g = |t: f64| rho - beta_at(orbit, sun, t).expect("in the span").sin().abs();
    let step = 600.0;
    let samples = days * 144;
    let mut changes = Vec::new();
    let mut least = f64::INFINITY;
    for k in 0..samples {
        let (a, b) = (f64::from(k) * step, (f64::from(k + 1) * step).min(span));
        if (g(a) < 0.0) != (g(b) < 0.0) {
            changes.push((a, b));
        }
        least = least.min(beta_at(orbit, sun, a).expect("in the span").abs());
    }
    let mut edges: Vec<f64> = got.seasons.iter().flat_map(|w| [w.rise, w.set]).collect();
    edges.retain(|&t| t != 0.0 && t != span);
    assert_eq!(
        edges.len(),
        changes.len(),
        "{name}: edges {edges:?}, sampled changes {changes:?}"
    );
    for (edge, (a, b)) in edges.iter().zip(&changes) {
        assert!(
            *edge >= a - tol && *edge <= b + tol,
            "{name}: edge {edge} outside [{a}, {b}]"
        );
    }
    let mut tiles: Vec<_> = got.seasons.iter().chain(&got.full_sun).copied().collect();
    tiles.sort_by(|x, y| x.rise.total_cmp(&y.rise));
    assert!(
        tiles.first().is_some_and(|w| w.rise == 0.0) && tiles.last().is_some_and(|w| w.set == span)
    );
    assert!(
        tiles.windows(2).all(|p| p[0].set == p[1].rise),
        "{name}: {tiles:?}"
    );
    assert!(got.least_beta.lower <= least && got.least_beta.lower <= got.least_beta.upper);
    let at_least = eclipse_fraction(orbit.radius, RE, least).expect("in the domain") * orbit.period;
    assert!(
        got.longest_eclipse.upper >= at_least,
        "{name}: {:?} vs sampled {at_least}",
        got.longest_eclipse
    );
    assert!(
        got.longest_eclipse.upper - got.longest_eclipse.lower <= tol,
        "{name}: {:?}",
        got.longest_eclipse
    );
    println!(
        "{name}: {} seasons, {} full-sun, least |beta| in [{:.6}, {:.6}] deg, longest eclipse [{:.3}, {:.3}] s",
        got.seasons.len(),
        got.full_sun.len(),
        got.least_beta.lower.to_degrees(),
        got.least_beta.upper.to_degrees(),
        got.longest_eclipse.lower,
        got.longest_eclipse.upper
    );
}

/// The negative case the dense sampling exists for (rule 16): the same scan with a rate bound 50
/// times too small steps over the two full-sun intervals of the 55° orbit at 1200 km and returns
/// one season where the shipped bound, and the sampling, find three.
#[test]
fn a_rate_bound_too_small_steps_over_full_sun() {
    let sun = AnalyticSun {
        epoch_j2000_seconds: EPOCH_2026,
    };
    let orbit = j2_orbit(1_200_000.0, 55_f64.to_radians(), 1.0);
    let span = 120.0 * DAY;
    let rho = RE / orbit.radius;
    let g = |t: f64| rho - beta_at(&orbit, sun, t).expect("in the span").sin().abs();
    let count = |rate: f64| {
        let mut n = 0;
        windows(g, rate, 0.0, span, 1.0, |_| n += 1);
        n
    };
    let shipped = eclipse_seasons(&orbit, RE, sun, 0.0, span, 1.0).expect("in the domain");
    let planted = count(SUN_DIRECTION_RATE / 50.0);
    assert!(
        shipped.seasons.len() == 3 && shipped.full_sun.len() == 2,
        "{shipped:?}"
    );
    assert!(planted == 1, "the planted bound found {planted} seasons");
}

/// A geometry outside the model's domain is refused with its reason. (S2)
#[test]
fn bad_geometry_is_refused() {
    let r = RE + 700_000.0;
    assert_eq!(eclipse_fraction(RE, RE, 0.0), Err(EclipseError::BadRadius));
    assert_eq!(
        eclipse_fraction(r, f64::NAN, 0.0),
        Err(EclipseError::BadRadius)
    );
    assert_eq!(
        eclipse_fraction(f64::INFINITY, RE, 0.0),
        Err(EclipseError::BadRadius)
    );
    assert_eq!(eclipse_fraction(r, -1.0, 0.0), Err(EclipseError::BadRadius));
    assert_eq!(
        eclipse_fraction(r, RE, f64::NAN),
        Err(EclipseError::BadBeta)
    );
    assert_eq!(eclipse_fraction(r, RE, 1.6), Err(EclipseError::BadBeta));
    assert_eq!(critical_beta(r, 0.0), Err(EclipseError::BadRadius));
    assert_eq!(
        beta_angle(0.0, -0.1, [1.0, 0.0, 0.0]),
        Err(EclipseError::BadPlane)
    );
    assert_eq!(
        beta_angle(f64::NAN, 0.1, [1.0, 0.0, 0.0]),
        Err(EclipseError::BadPlane)
    );
    assert_eq!(beta_angle(0.0, 0.1, [0.0; 3]), Err(EclipseError::BadSun));
    assert_eq!(
        beta_angle(0.0, 0.1, [f64::NAN, 0.0, 1.0]),
        Err(EclipseError::BadSun)
    );
    assert_eq!(
        conical_eclipse(r, RE, 1e9, 0.0),
        Err(EclipseError::BadSunDistance)
    );
    assert_eq!(
        conical_eclipse(r, RE, f64::INFINITY, 0.0),
        Err(EclipseError::BadSunDistance)
    );
    assert_eq!(
        conical_eclipse(2e11, RE, AU, 0.0),
        Err(EclipseError::BadSunDistance)
    );
}

/// An orbit or a span outside the model's domain is refused with its reason. (S2)
#[test]
fn bad_orbits_and_spans_are_refused() {
    let sun = AnalyticSun {
        epoch_j2000_seconds: EPOCH_2026,
    };
    let good = CircularOrbit {
        radius: RE + 700_000.0,
        inclination: 1.0,
        raan: 0.0,
        node_rate: 0.0,
        period: 5_926.0,
    };
    let year = 365.0 * DAY;
    let seasons = |o: &CircularOrbit, t0: f64, t1: f64, tol: f64| {
        eclipse_seasons(o, RE, sun, t0, t1, tol).map(|_| ())
    };
    assert_eq!(
        seasons(
            &CircularOrbit {
                period: 0.0,
                ..good
            },
            0.0,
            year,
            1.0
        ),
        Err(EclipseError::BadRates)
    );
    assert_eq!(
        seasons(
            &CircularOrbit {
                node_rate: f64::NAN,
                ..good
            },
            0.0,
            year,
            1.0
        ),
        Err(EclipseError::BadRates)
    );
    assert_eq!(
        seasons(
            &CircularOrbit {
                inclination: 4.0,
                ..good
            },
            0.0,
            year,
            1.0
        ),
        Err(EclipseError::BadPlane)
    );
    assert_eq!(
        seasons(
            &CircularOrbit {
                radius: 1.0,
                ..good
            },
            0.0,
            year,
            1.0
        ),
        Err(EclipseError::BadRadius)
    );
    assert_eq!(
        seasons(&good, year, 0.0, 1.0),
        Err(EclipseError::BadInterval)
    );
    assert_eq!(
        seasons(&good, f64::NAN, year, 1.0),
        Err(EclipseError::BadInterval)
    );
    assert_eq!(
        seasons(&good, 0.0, 80.0 * 365.25 * DAY, 1.0),
        Err(EclipseError::OutsideSunSpan)
    );
    assert_eq!(
        seasons(&good, 0.0, year, 0.0),
        Err(EclipseError::BadTolerance)
    );
    assert_eq!(
        beta_at(&good, sun, f64::INFINITY),
        Err(EclipseError::BadInterval)
    );
    assert_eq!(
        beta_at(&good, sun, -127.0 * 365.25 * DAY),
        Err(EclipseError::OutsideSunSpan)
    );
}
