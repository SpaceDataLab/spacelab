//! GPT3 held to the model's definition, evaluated in Maxima from the grid itself.
//!
//! `derivations/gpt3_evaluation.mac` reads the vendored `gpt3_5.grd` (which `verify.sh` holds against
//! TU Wien's file), evaluates the seasonal harmonics, the height reduction and the bilinear weights
//! in exact rationals and 40-digit bigfloats, and checks that every entry below is its value rounded
//! to the digits shown. Sites 1 and 2 are the two MATLAB reference cases Orekit carries; the others
//! reach a southern summer in the Andes, the 180° seam, a longitude past 180°, a site below the
//! ellipsoid, 5 km of height, both polar continuations and the origin on day 0.

// The literals are Maxima's 17 significant digits, read back by the derivation as written.
#![allow(clippy::excessive_precision, clippy::unreadable_literal)]

use spacelab_gnss::{Gpt3Result, gpt3};

/// Latitude and longitude in degrees, day of year, height in metres, then `ah`, `aw`, pressure
/// \[Pa\], temperature \[K\], water-vapour pressure \[Pa\], `Tm` \[K\], `lambda`, `Gn_h`, `Ge_h`,
/// `Gn_w`, `Ge_w`.
#[rustfmt::skip]
const GPT3_MAXIMA: [[f64; 15]; 10] = [
    [48.2, 16.37, 215.0, 156.0,
     1.2581539983279298e-3, 5.6821231105991382e-4, 1.0025772970994986e5, 2.9521023945230681e2,
     1.6676583414676177e3, 2.8106476696202875e2, 2.6977323268442856e0, -1.773443189177917e-4,
     -4.8307659502684979e-5, -8.5318215243896717e-5, -3.5865916578218664e-5],
    [77.5, 137.5, 92.0, 0.0,
     1.1715773647121051e-3, 5.7405657722569631e-4, 1.0199008681852084e5, 2.5456039880485554e2,
     1.2513690203407468e2, 2.5021189653242537e2, 1.8133320811615252e0, 4.818019541989097e-6,
     1.5416832499007502e-4, 1.0179381439311358e-6, 3.3920052345590095e-6],
    [-33.45, -70.66, 15.5, 2500.0,
     1.2633330029440148e-3, 5.343849617483344e-4, 7.5876522658466569e4, 2.8478703173059406e2,
     4.4613974879933864e2, 2.851170954469291e2, 4.2515943008054743e0, 1.7880214719607535e-4,
     4.5013361167158448e-5, 3.412042580938365e-5, 1.8855315996827397e-4],
    [-5.0, 180.0, 300.0625, 50.0,
     1.2791071897350469e-3, 6.031007725045141e-4, 1.0061004690174837e5, 3.0083003587244995e2,
     2.6610342692158054e3, 2.8877800312300608e2, 2.9209787865398978e0, -3.4348607867905672e-5,
     -2.3459240715450128e-5, -1.5611820154478384e-4, -8.9483808424965579e-5],
    [35.0, 200.0, 45.75, 1200.0,
     1.2384891735277949e-3, 5.6014061278183403e-4, 8.8014979869850841e4, 2.7601131984128982e2,
     6.3875667259632865e2, 2.7525537273273142e2, 3.2754098771110535e0, -3.7745344990389545e-4,
     3.4414270975475041e-5, -5.4177824498639627e-5, -1.9297344217577776e-6],
    [31.5, 35.5, 160.0, -430.0,
     1.2739897953270291e-3, 5.1461343567156164e-4, 1.0602459359310582e5, 3.0377211707585173e2,
     2.1084728875030676e3, 2.8786246965092239e2, 4.3783084091013944e0, -1.5901798491529841e-4,
     7.5810865129848878e-6, 2.0418237062400632e-5, 5.6998125162650667e-5],
    [30.0, 90.0, 200.0, 5000.0,
     1.2767519941656793e-3, 4.6452191434116729e-4, 5.5427527422784657e4, 2.779416510281281e2,
     6.4945249073064652e2, 2.7189337140519829e2, 2.8509315942046232e0, 2.5131833801059488e-5,
     -1.3175632175615488e-5, 9.8410805812666777e-5, 2.0245822243181602e-8],
    [-89.0, -100.0, 365.0, 2800.0,
     1.1817499519704718e-3, 4.7039336807826325e-4, 6.8708458952006389e4, 2.4870086308762238e2,
     4.8453726998781912e1, 2.4210362878044387e2, 2.720264934940317e0, -6.748174293649564e-5,
     -2.0896460863145283e-5, 2.1044867283642372e-6, -4.0388542932465579e-6],
    [89.9, 16.37, 1.0, 0.0,
     1.1505152854648294e-3, 5.9385713593575024e-4, 1.0163549037402535e5, 2.4577965090916644e2,
     5.4683979275106712e1, 2.4630708130846001e2, 5.5890580804364481e-1, -2.3117904359702528e-5,
     4.9618086505780121e-5, -7.7838931572955746e-6, 6.4405624826155336e-7],
    [0.0, 0.0, 0.0, 0.0,
     1.27526e-3, 6.14485e-4, 1.0127395468529191e5, 2.9963663675e2,
     2.7297056205738316e3, 2.8755e2, 2.602375e0, 4.775e-6,
     -9.6e-6, -5.62e-5, 6.37e-5],
];

fn fields(r: &Gpt3Result) -> [f64; 11] {
    [
        r.ah,
        r.aw,
        r.pressure,
        r.temperature,
        r.water_vapor_pressure,
        r.tm,
        r.lambda,
        r.gnh,
        r.geh,
        r.gnw,
        r.gew,
    ]
}

/// Each field's typical size over the globe. A field that passes through zero (the gradients: site
/// 7's `Ge_w` is 2e-8, a sum of terms near 1e-5) carries rounding relative to its terms, not to its
/// value, so the difference is taken relative to the larger of the two.
const SCALE: [f64; 11] = [1e-3, 1e-4, 1e5, 1e2, 1e1, 1e2, 1.0, 1e-5, 1e-5, 1e-5, 1e-5];

/// Double rounding over the parse, the harmonics, one exponential and four weights; 1e-13 is about
/// 450 units in the last place and far below any change of formula, coefficient or cell.
const TOL: f64 = 1e-13;

#[test]
fn gpt3_is_the_grid_evaluated_by_its_definition() {
    let mut worst: f64 = 0.0;
    for (i, row) in GPT3_MAXIMA.iter().enumerate() {
        let r = gpt3(row[0].to_radians(), row[1].to_radians(), row[2], row[3]);
        for (k, (got, want)) in fields(&r).into_iter().zip(&row[4..]).enumerate() {
            let rel = (got - want).abs() / want.abs().max(SCALE[k]);
            worst = worst.max(rel);
            assert!(
                rel <= TOL,
                "site {}, field {k}: {got} against Maxima's {want}",
                i + 1
            );
        }
    }
    println!("worst scaled difference {worst:e}");
}
