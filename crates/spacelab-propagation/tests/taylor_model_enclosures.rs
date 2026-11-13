//! Every integrator against validated enclosures of the exact flow (plans/17 Ф28.H).
//!
//! `derivations/taylor_model_enclosures.jl` encloses, with TaylorModels.jl, the exact state at the
//! end of three dimensionless arcs: a Kepler orbit, a J2 orbit and the Earth–Moon L1 halo. Each
//! box is under 1e-13 wide, so it is a reference with a certified radius:
//! `max(|x − lo|, |x − hi|)` bounds an integrator's true error from above and the distance from
//! `x` to the box bounds it from below. `derivations/taylor_model_enclosures.mac` checks the Kepler
//! box against the closed form, and derives the J2 and CR3BP fields written here from their
//! potentials. IAS15 and RKN1210 take a position-only acceleration, so they skip the halo.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use spacelab_autodiff::Scalar;
use spacelab_propagation::driver::{Controls, DEFAULT_MAX_STEPS, integrate};
use spacelab_propagation::forces::{Acceleration, Newtonian, Plain};
use spacelab_propagation::ias15::integrate_ias15;
use spacelab_propagation::taylor::integrate_taylor;
use spacelab_propagation::{Dop853, Tolerance, rkn1210};

const FIXTURE: &str = include_str!("fixtures/taylor_model_enclosures.txt");
/// The constants the generator writes; `taylor_model_enclosures.mac` derives the fields they enter.
const J2: f64 = 1.082_626_683_553_151_3e-3;
const MU: f64 = 0.012_150_584_460_351;

struct Case {
    name: String,
    ic: [f64; 6],
    t1: f64,
    lo: [f64; 6],
    hi: [f64; 6],
}

fn doubles(line: &str, key: &str) -> Vec<f64> {
    let rest = line.strip_prefix(key).expect("the fixture's line order");
    rest.split_whitespace()
        .map(|w| f64::from_bits(w.parse().expect("a u64 bit pattern")))
        .collect()
}

fn cases() -> Vec<Case> {
    let lines: Vec<&str> = FIXTURE.lines().filter(|l| !l.starts_with('#')).collect();
    lines
        .chunks(5)
        .map(|c| Case {
            name: c[0].strip_prefix("case ").expect("a case line").to_owned(),
            ic: doubles(c[1], "ic ").try_into().unwrap(),
            t1: doubles(c[2], "t1 ")[0],
            lo: doubles(c[3], "lo ").try_into().unwrap(),
            hi: doubles(c[4], "hi ").try_into().unwrap(),
        })
        .collect()
}

fn case(name: &str) -> Case {
    cases()
        .into_iter()
        .find(|c| c.name == name)
        .expect("the case")
}

/// The certified error bounds of `x`: `(below, above)` with `below ≤ |x − exact| ≤ above`, the
/// largest over the six components.
fn certified_error(c: &Case, x: &[f64; 6]) -> (f64, f64) {
    x.iter().zip(c.lo.iter().zip(&c.hi)).fold(
        (0.0_f64, 0.0_f64),
        |(below, above), (&xi, (&lo, &hi))| {
            (
                below.max((lo - xi).max(xi - hi).max(0.0)),
                above.max((xi - lo).abs().max((xi - hi).abs())),
            )
        },
    )
}

struct Kepler;
impl Acceleration for Kepler {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let r3inv = (r2 * r2.sqrt()).recip();
        [-r[0] * r3inv, -r[1] * r3inv, -r[2] * r3inv]
    }
}

struct ZonalJ2;
impl Acceleration for ZonalJ2 {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], _v: &[S; 3]) -> [S; 3] {
        let one = S::from_f64(1.0);
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        let r3inv = (r2 * r2.sqrt()).recip();
        let s = S::from_f64(1.5 * J2) / r2;
        let q = S::from_f64(5.0) * r[2] * r[2] / r2;
        let equatorial = r3inv * (one + s * (one - q));
        let polar = r3inv * (one + s * (S::from_f64(3.0) - q));
        [-r[0] * equatorial, -r[1] * equatorial, -r[2] * polar]
    }
}

struct Cr3bp;
impl Acceleration for Cr3bp {
    fn acceleration<S: Scalar>(&self, _t: S, r: &[S; 3], v: &[S; 3]) -> [S; 3] {
        let mu = S::from_f64(MU);
        let d1 = r[0] + mu;
        let d2 = r[0] - S::from_f64(1.0) + mu;
        let r1s = d1 * d1 + r[1] * r[1] + r[2] * r[2];
        let r2s = d2 * d2 + r[1] * r[1] + r[2] * r[2];
        let a1 = S::from_f64(1.0 - MU) / (r1s * r1s.sqrt());
        let a2 = mu / (r2s * r2s.sqrt());
        let two = S::from_f64(2.0);
        [
            r[0] + two * v[1] - a1 * d1 - a2 * d2,
            r[1] - two * v[0] - a1 * r[1] - a2 * r[1],
            -a1 * r[2] - a2 * r[2],
        ]
    }
}

fn position_only<F: Acceleration>(force: &F) -> impl Fn(&[f64; 3]) -> [f64; 3] + '_ {
    |r| force.acceleration(0.0, r, &[0.0; 3])
}

fn split(y: &[f64; 6]) -> ([f64; 3], [f64; 3]) {
    ([y[0], y[1], y[2]], [y[3], y[4], y[5]])
}

fn join(r: [f64; 3], v: [f64; 3]) -> [f64; 6] {
    [r[0], r[1], r[2], v[0], v[1], v[2]]
}

fn dop853<F: Acceleration>(force: F, c: &Case) -> [f64; 6] {
    integrate(
        &Dop853,
        &Plain(Newtonian(force)),
        0.0,
        &c.ic,
        c.t1,
        1e-3,
        Controls::new(Tolerance::new(1e-15, 1e-15), 1e-12),
    )
    .expect("DOP853")
    .final_state()
}

fn taylor<F: Acceleration>(force: F, c: &Case) -> [f64; 6] {
    integrate_taylor::<6, 26>(&Newtonian(force), 0.0, &c.ic, c.t1, 1e-16).expect("Taylor")
}

fn rkn<F: Acceleration>(force: &F, c: &Case) -> [f64; 6] {
    let (r0, v0) = split(&c.ic);
    let accel = position_only(force);
    let (r, v) = rkn1210::integrate(
        &|_t, r: &[f64; 3]| accel(r),
        0.0,
        &r0,
        &v0,
        c.t1,
        1e-2,
        Tolerance::new(1e-15, 1e-15),
        100_000,
    )
    .expect("RKN1210");
    join(r, v)
}

fn ias15<F: Acceleration>(force: &F, c: &Case) -> [f64; 6] {
    let (r0, v0) = split(&c.ic);
    let (r, v) = integrate_ias15(
        &position_only(force),
        &r0,
        &v0,
        0.0,
        c.t1,
        1e-2,
        DEFAULT_MAX_STEPS,
    )
    .expect("IAS15");
    join(r, v)
}

fn check(c: &Case, method: &str, x: &[f64; 6], bound: f64) {
    let (below, above) = certified_error(c, x);
    println!(
        "{:<6} {method:<8} error in [{below:.2e}, {above:.2e}]",
        c.name
    );
    assert!(
        above <= bound,
        "{} by {method}: the error may reach {above:.3e}, past {bound:.1e}",
        c.name
    );
}

#[test]
fn every_box_is_narrow_enough_to_be_a_reference() {
    let all = cases();
    assert_eq!(all.len(), 3);
    for c in &all {
        let width = (0..6).map(|i| c.hi[i] - c.lo[i]).fold(0.0, f64::max);
        assert!(
            (0..6).all(|i| c.lo[i] <= c.hi[i]) && width < 1e-13,
            "{}: width {width:.3e}",
            c.name
        );
    }
}

#[test]
fn the_kepler_arc_lands_in_its_enclosure() {
    let c = case("kepler");
    // Measured: at most 5.0e-14 (DOP853), the box itself 6.8e-14 wide.
    check(&c, "DOP853", &dop853(Kepler, &c), 2e-13);
    check(&c, "Taylor", &taylor(Kepler, &c), 2e-13);
    check(&c, "RKN1210", &rkn(&Kepler, &c), 2e-13);
    check(&c, "IAS15", &ias15(&Kepler, &c), 2e-13);
}

#[test]
fn the_j2_arc_lands_in_its_enclosure() {
    let c = case("j2");
    // Measured: at most 5.1e-14 (DOP853), the box itself 8.7e-14 wide.
    check(&c, "DOP853", &dop853(ZonalJ2, &c), 2e-13);
    check(&c, "Taylor", &taylor(ZonalJ2, &c), 2e-13);
    check(&c, "RKN1210", &rkn(&ZonalJ2, &c), 2e-13);
    check(&c, "IAS15", &ias15(&ZonalJ2, &c), 2e-13);
}

#[test]
fn the_halo_period_lands_in_its_enclosure() {
    let c = case("halo");
    // Measured: at most 3.5e-14 (DOP853), the box itself 2.8e-14 wide.
    check(&c, "DOP853", &dop853(Cr3bp, &c), 1.5e-13);
    check(&c, "Taylor", &taylor(Cr3bp, &c), 1.5e-13);
}

#[test]
fn the_halo_closes_after_its_period() {
    // The start is the orbit `periodic::richardson_halo(EARTH_MOON, 0, 0.2, true)` returns (pinned
    // in spacelab-astro), so the box at its period bounds how far the corrector's orbit is from
    // closing. Measured: 9.8e-13, under the corrector's 1e-11 acceptance.
    let c = case("halo");
    let (_, gap) = certified_error(&c, &c.ic);
    println!("the halo returns within {gap:.2e} of its start");
    assert!(gap <= 4e-12, "the halo closes only to {gap:.3e}");
}
