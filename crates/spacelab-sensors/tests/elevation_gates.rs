//! `ElevationMask` against its defining vectors: the footprint edge in closed form.
//!
//! `derivations/elevation_footprint.mac` proves that on a sphere the cell at Earth-central angle
//! `lam = pi/2 - eps - asin(cos(eps)/L)` from the sub-satellite point sees the satellite at
//! exactly `eps`, at slant range `sqrt(r_s^2 - (R cos eps)^2) - R sin eps`;
//! `derivations/elevation_footprint.gp` evaluates both at 120 digits from the vectors and prints
//! the rows below.

#![allow(
    clippy::expect_used,
    clippy::float_cmp,
    clippy::excessive_precision,
    clippy::unreadable_literal
)]

use core::f64::consts::FRAC_PI_2;

use spacelab_core::Vec3;
use spacelab_sensors::{AccessGeometry, Constraint, ElevationMask, elevation::elevation};

/// The sphere of `derivations/elevation_footprint.gp`.
const R: f64 = 6_378_137.0;

/// `(h_m, eps_deg, lam_rad, slant_m)`, printed verbatim by `derivations/elevation_footprint.gp`
/// at 120 digits and cut to 25 here.
const EDGE: [(f64, f64, f64, f64); 30] = [
    (
        500000.0,
        0.0,
        3.836465158927483535429614e-1,
        2.574516847876509818821110e6,
    ),
    (
        500000.0,
        5.0,
        3.056998934906724181903116e-1,
        2.077956127651206684770193e6,
    ),
    (
        500000.0,
        10.0,
        2.451503638624737108305244e-1,
        1.695091195937326299460795e6,
    ),
    (
        500000.0,
        25.0,
        1.363978509447270706383602e-1,
        1.031941724402940872078776e6,
    ),
    (
        500000.0,
        60.0,
        4.148511543585046947449917e-2,
        5.705169373114309894241086e5,
    ),
    (
        500000.0,
        89.0,
        1.268866577929246180922568e-3,
        5.000706262019652767842537e5,
    ),
    (
        800000.0,
        0.0,
        4.766204303601609703857355e-1,
        3.293177675133851626299222e6,
    ),
    (
        800000.0,
        5.0,
        3.966721755450382565458527e-1,
        2.783874338172048624223427e6,
    ),
    (
        800000.0,
        10.0,
        3.307214218945261453213083e-1,
        2.366881948499214753418173e6,
    ),
    (
        800000.0,
        25.0,
        1.982823507535167623937772e-1,
        1.560165149823921885858012e6,
    ),
    (
        800000.0,
        60.0,
        6.323366609921286120570795e-2,
        9.071949850664925595563491e5,
    ),
    (
        800000.0,
        89.0,
        1.945326994092209336184957e-3,
        8.001082782689087494416961e5,
    ),
    (
        2000000.0,
        0.0,
        7.055061602738147758655616e-1,
        5.432545259820667154457158e6,
    ),
    (
        2000000.0,
        5.0,
        6.226957111323401686841098e-1,
        4.905021028122866106571436e6,
    ),
    (
        2000000.0,
        10.0,
        5.486278066801545006197275e-1,
        4.436744223438346062163337e6,
    ),
    (
        2000000.0,
        25.0,
        3.730342090130085731597148e-1,
        3.368999364706006845591003e6,
    ),
    (
        2000000.0,
        60.0,
        1.331086158594623247739329e-1,
        2.223823922975974990951504e6,
    ),
    (
        2000000.0,
        89.0,
        4.166673499323490627731265e-3,
        2.000231917992468702931602e6,
    ),
    (
        20200000.0,
        0.0,
        1.328454345768369118158457e0,
        2.580148706567123429140516e7,
    ),
    (
        20200000.0,
        5.0,
        1.242128445981080269493609e0,
        2.525158342292982178724592e7,
    ),
    (
        20200000.0,
        10.0,
        1.157675216727042519546714e0,
        2.471769558467162008371336e7,
    ),
    (
        20200000.0,
        25.0,
        9.152189019619758024667197e-1,
        2.324638999807212730376869e7,
    ),
    (
        20200000.0,
        60.0,
        4.033205637425577893488463e-1,
        2.086248905220265691926248e7,
    ),
    (
        20200000.0,
        89.0,
        1.326510714557205117900997e-2,
        2.020073831932083746704238e7,
    ),
    (
        35786000.0,
        0.0,
        1.418944163128946982106644e0,
        4.167893733486975552978051e7,
    ),
    (
        35786000.0,
        5.0,
        1.332260000810660042688024e0,
        4.112675299193481191772746e7,
    ),
    (
        35786000.0,
        10.0,
        1.246735699519634203079352e0,
        4.058609859089754635544579e7,
    ),
    (
        35786000.0,
        25.0,
        9.969343842183216890671829e-1,
        3.907049329323332888724998e7,
    ),
    (
        35786000.0,
        60.0,
        4.478918563279071367280304e-1,
        3.651973336241052094723698e7,
    ),
    (
        35786000.0,
        89.0,
        1.481327717845913950288381e-2,
        3.578682448595130711803641e7,
    ),
];

/// The polar cell, its zenith, and the satellite at central angle `lam` and altitude `h`.
fn geometry(h: f64, lam: f64) -> AccessGeometry {
    AccessGeometry {
        satellite: Vec3::new((R + h) * lam.sin(), 0.0, (R + h) * lam.cos()),
        cell: Vec3::new(0.0, 0.0, R),
        zenith: Vec3::new(0.0, 0.0, 1.0),
    }
}

/// (G2) On every edge row the defining vectors give back the mask: `elevation` returns `eps`
/// within `32 eps_mach (1 + 1/cos eps)` rad (the satellite's coordinates carry an ulp of
/// `R + h`, and `asin` amplifies by `1/cos eps`), the mask's `g = t - sin(eps)|d|` vanishes within
/// `64 eps_mach |d|` m, and `|d|` is the 120-digit slant range within `32 eps_mach (R + h)/|d|`
/// relative.
#[test]
fn the_mask_footprint_edge_is_where_the_closed_form_puts_it() {
    for (h, eps_deg, lam, slant) in EDGE {
        let eps = eps_deg.to_radians();
        let geom = geometry(h, lam);
        let d = geom.line_of_sight().norm();

        let got = elevation(geom.satellite, geom.cell, geom.zenith);
        let tol = 32.0 * f64::EPSILON * (1.0 + 1.0 / eps.cos());
        assert!(
            (got - eps).abs() <= tol,
            "h {h} m, mask {eps_deg} deg: elevation {got:.17e} rad, want {eps:.17e} (tol {tol:.1e})"
        );

        let g = ElevationMask::new(eps).expect("mask in range").g(&geom);
        assert!(
            g.abs() <= 64.0 * f64::EPSILON * d,
            "h {h} m, mask {eps_deg} deg: g = {g:.3e} m on the edge"
        );

        let rel = (d - slant).abs() / slant;
        assert!(
            rel <= 32.0 * f64::EPSILON * (R + h) / d,
            "h {h} m, mask {eps_deg} deg: |d| = {d:.17e} m, slant range {slant:.17e} m"
        );
    }
}

/// (G3) The exact special cases: straight overhead the elevation is `pi/2` to the ulp and the
/// horizon mask covers; at the antipode it is `-pi/2`; at the horizon tangency `cos(lam) = 1/L`
/// (the `eps = 0` rows) it is 0 within `32 eps_mach` rad and `g` of the horizon mask within
/// `64 eps_mach |d|`.
#[test]
fn overhead_antipode_and_horizon_are_exact() {
    for h in [500e3, 35_786e3] {
        let overhead = geometry(h, 0.0);
        assert_eq!(
            elevation(overhead.satellite, overhead.cell, overhead.zenith),
            FRAC_PI_2
        );
        assert!(ElevationMask::HORIZON.covered(&overhead));

        let antipode = geometry(h, core::f64::consts::PI);
        let below = elevation(antipode.satellite, antipode.cell, antipode.zenith);
        assert!(
            (below + FRAC_PI_2).abs() <= 2.0 * f64::EPSILON,
            "antipode at {below}"
        );
        assert!(!ElevationMask::HORIZON.covered(&antipode));

        let lam = (R / (R + h)).acos();
        let tangent = geometry(h, lam);
        let d = tangent.line_of_sight().norm();
        let at = elevation(tangent.satellite, tangent.cell, tangent.zenith);
        assert!(at.abs() <= 32.0 * f64::EPSILON, "tangency at {at:.3e} rad");
        assert!(ElevationMask::HORIZON.g(&tangent).abs() <= 64.0 * f64::EPSILON * d);
    }
}

/// (S1) The elevation keeps its digits next to the zenith and the nadir (Ф28.N, C3 row 15). `ZENITH`
/// is the double nearest `u = (2, −3, 6)/7` and each satellite the doubles nearest
/// `cos t·u + sin t·w`, `w = (3, 6, 2)/7`, at `t = 10⁻ᵏ` and `π − 10⁻ᵏ`, k = 1..12, seen from the
/// origin so that the line of sight is that vector exactly; `derivations/angle_from_vectors.gp`
/// gives its elevation `atan2(d·ẑ, |d × ẑ|)` at 100 digits. The bound is 2u of absolute error plus
/// one ulp of the elevation (`derivations/angle_from_vectors.mac` 4). Measured: exact on 23 rows
/// and one ulp of π/2 on one. `asin(d·ẑ/|d|)` was 6 u off at `π/2 − 10⁻¹`, 7638 u at
/// `π/2 − 10⁻⁴` and 1.9e8 u (2.1e-8 rad) at `π/2 − 10⁻¹¹`; next to the nadir, 1.3e8 u at
/// `−π/2 + 10⁻¹²`.
#[test]
fn the_elevation_keeps_its_digits_next_to_the_zenith_and_the_nadir() {
    const ZENITH: [f64; 3] = [
        0.2857142857142856984253854,
        -0.4285714285714285476380780,
        0.8571428571428570952761561,
    ];
    #[rustfmt::skip]
    const ROWS: [([f64; 3], f64); 24] = [
    ([0.3270726543566480071234537, -0.3408588565647297508043323, 0.8813845464231158111090281], 1.470796326794896605638574),
    ([0.2899856429765472198845089, -0.4199787144635708235895777, 0.8599570955954749518213021], 1.560796326794896623200989),
    ([0.2861427142142975998950760, -0.4277140715714464191954391, 0.8574281428095594792537781], 1.569796326794896640418848),
    ([0.2857571414285000055421904, -0.4284857121430000193029741, 0.8571714242856666166048285], 1.570696326794896647701247),
    ([0.2857185714142856380881597, -0.4285628571214287196866621, 0.8571457142428571263081949], 1.570786326794896617027660),
    ([0.2857147142855714361253661, -0.4285705714283571388101279, 0.8571431428567143351315849], 1.570795326794896598239198),
    ([0.2857143285714271452313540, -0.4285713428571407135336813, 0.8571428857142814328184954], 1.570796226794896616355589),
    ([0.2857142899999999818838603, -0.4285714199999999807211282, 0.8571428599999999509151394], 1.570796316794896624494880),
    ([0.2857142861428571545268085, -0.4285714277142856909463831, 0.8571428574285714363512056], 1.570796325794896592002102),
    ([0.2857142857571428606888730, -0.4285714284857142786222539, 0.8571428571714285737925820], 1.570796326694896610957285),
    ([0.2857142857185714368561946, -0.4285714285628571262876108, 0.8571428571457142542300289], 1.570796326784896610473741),
    ([0.2857142857147143000240419, -0.4285714285705714554630674, 0.8571428571431428666826946], 1.570796326793896633422752),
    ([-0.2415011543736524379788477, 0.5120018565307208335823930, -0.8243368797677854686867249], -1.470796326794896661664096),
    ([-0.2814143572615472321807317, 0.4371212858935708545082832, -0.8542429051188082933521173], -1.560796326794896591719818),
    ([-0.2852855715000118808077900, 0.4294283570000178573700111, -0.8568567143333690738771224], -1.569796326794896624621968),
    ([-0.2856714271429285911807483, 0.4286571407141428480258583, -0.8571142814286190070305338], -1.570696326794896623910754),
    ([-0.2857099999857143357573364, 0.4285799999785712688371575, -0.8571399999571428507394444], -1.570786326794896609096623),
    ([-0.2857138571427143003411686, 0.4285722857140714103785228, -0.8571425714281428742680191], -1.570795326794896645820153),
    ([-0.2857142428571414205507040, 0.4285715142857121628949812, -0.8571428285714243200388296], -1.570796226794896608425428),
    ([-0.2857142814285714149669104, 0.4285714371428571145550279, -0.8571428542857142396371728], -1.570796316794896624494880),
    ([-0.2857142852857142978351135, 0.4285714294285714043297730, -0.8571428568571428652234090], -1.570796325794896647513253),
    ([-0.2857142856714285916730489, 0.4285714286571428721650534, -0.8571428571142857277820326], -1.570796326694896618887448),
    ([-0.2857142857100000155057273, 0.4285714285800000244996966, -0.8571428571400000473445857], -1.570796326784896618403918),
    ([-0.2857142857138571523378801, 0.4285714285722856953242399, -0.8571428571425714348919200], -1.570796326793896641353042),
    ];
    let zenith = Vec3::new(ZENITH[0], ZENITH[1], ZENITH[2]);
    let cell = Vec3::new(0.0, 0.0, 0.0);
    let u = f64::EPSILON / 2.0;
    for (d, want) in ROWS {
        let got = elevation(Vec3::new(d[0], d[1], d[2]), cell, zenith);
        assert!(
            (got - want).abs() <= 2.0 * u + (want.abs().next_up() - want.abs()),
            "elevation at {want:e}: {got:.17e}, {:.3} u off",
            (got - want).abs() / u
        );
    }
}
