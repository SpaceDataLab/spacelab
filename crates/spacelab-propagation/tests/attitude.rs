//! RKMK4 attitude: the free rigid body's angular-momentum magnitude and kinetic energy are
//! conserved, the attitude stays exactly orthogonal (`RᵀR = I`), and the method converges at order 4.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use spacelab_propagation::attitude::{
    Inertia, Mat3, Vec3, exp_so3, integrate_attitude, integrate_attitude_torqued, rotate_inertia,
    solid_box_inertia, sum_inertia, thin_plate_inertia, translate_inertia,
};

const INERTIA: Vec3 = [1.0, 2.0, 3.0]; // an asymmetric top
const W0: Vec3 = [1.0, 0.3, 0.6]; // tumbling
const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

fn angular_momentum_mag(w: Vec3) -> f64 {
    let p = [INERTIA[0] * w[0], INERTIA[1] * w[1], INERTIA[2] * w[2]];
    (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
}

fn energy(w: Vec3) -> f64 {
    0.5 * (INERTIA[0] * w[0] * w[0] + INERTIA[1] * w[1] * w[1] + INERTIA[2] * w[2] * w[2])
}

/// Largest entry of `|RᵀR − I|` — how far the attitude has drifted off `SO(3)`.
fn orthogonality_defect(r: &Mat3) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            let dot = r[0][i] * r[0][j] + r[1][i] * r[1][j] + r[2][i] * r[2][j];
            let target = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot - target).abs());
        }
    }
    worst
}

#[test]
fn rkmk4_conserves_momentum_energy_and_orthogonality() {
    // 2000 steps of a tumbling asymmetric top.
    let steps = 2000;
    let dt = 0.01;
    let (r, w) = integrate_attitude(&IDENTITY, W0, INERTIA, dt, steps);

    let l_err =
        (angular_momentum_mag(w) - angular_momentum_mag(W0)).abs() / angular_momentum_mag(W0);
    let e_err = (energy(w) - energy(W0)).abs() / energy(W0);
    let ortho = orthogonality_defect(&r);
    println!(
        "RKMK4 over {steps} steps: |L| rel err {l_err:.3e}, energy rel err {e_err:.3e}, RᵀR−I {ortho:.3e}"
    );

    // The two Casimir/energy invariants are conserved to the method's accuracy over the whole run.
    assert!(l_err < 1e-6, "angular momentum not conserved: {l_err:.3e}");
    assert!(e_err < 1e-6, "energy not conserved: {e_err:.3e}");
    // The attitude stays on SO(3) to machine precision — the whole point of a Lie-group integrator.
    assert!(ortho < 1e-13, "attitude drifted off SO(3): {ortho:.3e}");
}

#[test]
fn rkmk4_converges_at_order_four() {
    // Compare the final angular velocity at N and 2N steps to a fine reference; error should quarter
    // twice (halve step → error /16).
    let span = 5.0;
    let reference = integrate_attitude(&IDENTITY, W0, INERTIA, span / 8000.0, 8000).1;
    let diff = |n: usize| {
        let w = integrate_attitude(&IDENTITY, W0, INERTIA, span / n as f64, n).1;
        ((w[0] - reference[0]).powi(2)
            + (w[1] - reference[1]).powi(2)
            + (w[2] - reference[2]).powi(2))
        .sqrt()
    };
    let coarse = diff(250);
    let fine = diff(500);
    let order = (coarse / fine).log2();
    println!("RKMK4: err {coarse:.3e} → {fine:.3e} (250→500 steps), order {order:.2}");
    assert!(
        (order - 4.0).abs() < 0.4,
        "measured order {order:.2} is not 4"
    );
}

#[test]
fn exp_of_a_full_turn_returns_to_identity() {
    // exp([ω]×) with |ω| = 2π about any axis is the identity — a direct check of the Rodrigues map.
    use spacelab_propagation::attitude::exp_so3;
    let axis = [0.0, 0.0, 1.0];
    let two_pi = std::f64::consts::TAU;
    let r = exp_so3([axis[0] * two_pi, axis[1] * two_pi, axis[2] * two_pi]);
    let defect = orthogonality_defect(&r);
    let to_identity =
        (r[0][0] - 1.0).abs() + (r[1][1] - 1.0).abs() + (r[2][2] - 1.0).abs() + r[0][1].abs();
    println!("exp(2π ẑ): RᵀR−I {defect:.3e}, distance to I {to_identity:.3e}");
    assert!(defect < 1e-14, "not a rotation: {defect:.3e}");
    assert!(
        to_identity < 1e-12,
        "2π rotation is not the identity: {to_identity:.3e}"
    );
}

/// An inertia with no unit component. `INERTIA`'s first entry is exactly `1.0`, which makes
/// `euler`'s division by it indistinguishable from a multiplication — measured, not assumed: both
/// mutants of `m[0] / inertia[0]` survived every test in this file.
const INERTIA_B: Vec3 = [1.7, 2.3, 3.1];

#[test]
fn the_conservation_laws_hold_for_an_inertia_with_no_unit_component() {
    let steps = 2000;
    let dt = 0.01;
    let (r, w) = integrate_attitude(&IDENTITY, W0, INERTIA_B, dt, steps);
    let body_l = |w: Vec3| {
        [
            INERTIA_B[0] * w[0],
            INERTIA_B[1] * w[1],
            INERTIA_B[2] * w[2],
        ]
    };
    let mag = |v: Vec3| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let energy_b = |w: Vec3| {
        0.5 * (INERTIA_B[0] * w[0] * w[0] + INERTIA_B[1] * w[1] * w[1] + INERTIA_B[2] * w[2] * w[2])
    };

    let l_err = (mag(body_l(w)) - mag(body_l(W0))).abs() / mag(body_l(W0));
    let e_err = (energy_b(w) - energy_b(W0)).abs() / energy_b(W0);
    let l0 = body_l(W0);
    let l = apply(&r, body_l(w));
    let moved = (0..3).fold(0.0_f64, |m, i| m.max((l[i] - l0[i]).abs()));
    println!(
        "asymmetric inertia {INERTIA_B:?}: |L| rel err {l_err:.3e}, energy {e_err:.3e}, |ΔL| {moved:.3e}"
    );
    // `I·ω̇ = (Iω)×ω` is perpendicular to `Iω`, which is exactly why `|Iω|` is conserved. Divide by
    // the wrong inertia and it stops being perpendicular, so this is the assertion that sees it.
    assert!(l_err < 1e-9, "angular momentum not conserved: {l_err:.3e}");
    assert!(e_err < 1e-9, "energy not conserved: {e_err:.3e}");
    assert!(
        moved < 1e-9,
        "the inertial angular momentum moved: {moved:.3e}"
    );
}

#[test]
fn below_the_small_angle_threshold_the_series_is_its_own_leading_term() {
    // `exp_so3` switches to `(1 − θ²/6, 1/2 − θ²/24)` below θ = 1e-8, and nothing above exercises
    // that branch: the smallest angle anywhere else is a full RKMK step. Inside it the corrections
    // are θ²/6 ≤ 1.7e-17 and θ²/24 ≤ 4.2e-18, both under the half-ulp of 1.0 (1.1e-16) and of 0.5
    // (5.6e-17), so in f64 the coefficients are *exactly* 1.0 and 0.5 — anything else is a wrong
    // coefficient, not a rounding difference.
    //
    // The two are separated without re-deriving Rodrigues: `K` is skew and `K²` symmetric, so the
    // antisymmetric part of `R = I + a·K + b·K²` is `a·K` alone and the symmetric off-diagonal part
    // is `b·K²` alone.
    let theta = 9.9e-9;
    let axis = [0.36, 0.48, 0.8]; // 0.36² + 0.48² + 0.8² = 1 exactly
    let w = [axis[0] * theta, axis[1] * theta, axis[2] * theta];
    let r = exp_so3(w);

    // a·K: (R₂₁ − R₁₂)/2 = a·w₀, and a is exactly 1 here.
    let anti = [
        (r[2][1] - r[1][2]) / 2.0,
        (r[0][2] - r[2][0]) / 2.0,
        (r[1][0] - r[0][1]) / 2.0,
    ];
    // b·K²: (R₀₁ + R₁₀)/2 = b·w₀·w₁, and b is exactly 1/2 here.
    let sym = f64::midpoint(r[0][1], r[1][0]);

    println!("exp_so3(θ={theta:e}): antisymmetric {anti:?}, symmetric off-diagonal {sym:e}");
    for i in 0..3 {
        assert_eq!(
            anti[i].to_bits(),
            w[i].to_bits(),
            "sinθ/θ is not exactly 1 below the threshold: {:e} vs {:e}",
            anti[i],
            w[i]
        );
    }
    // `b` gets a tolerance, not a bit comparison, and it cannot be tightened. `R₀₁` and `R₁₀` are
    // ±a·w₂ ≈ ±7.9e−9 plus the same b·w₀w₁ ≈ 8.5e−18, so recovering the symmetric part cancels two
    // numbers nine orders larger than the answer: each carries half an ulp, ≈8.6e−25, which is 1e−7
    // of the 1.7e−17 that survives. On the diagonal `b` fares worse — `1 − b(w₁²+w₂²)` is
    // `1 − 4.3e−17`, inside the half-ulp of 1.0, so it rounds back to exactly 1.0. Below the
    // threshold `b` has no observable effect on `R` by any route, which is why its mutants are
    // recorded as equivalent in `.cargo/mutants.toml` rather than killed here.
    let want = 0.5 * w[0] * w[1];
    assert!(
        ((sym - want) / want).abs() < 1.0e-6,
        "(1−cosθ)/θ² is not 1/2 below the threshold: {sym:e} vs {want:e}"
    );
}

#[test]
fn above_the_small_angle_threshold_the_second_coefficient_keeps_full_precision() {
    // Nothing else here can see `b = (1−cosθ)/θ²` above the threshold. It only ever reaches `R`
    // multiplied back by `K² = O(θ²)`, so however wrong it is the diagonal moves by at most an ulp
    // of 1.0 — every assertion in this file reads `R` at that scale and none of them can fail.
    //
    // A rotation about an axis in the xy-plane makes it readable: `K₀₁ = −w₂ = 0` and
    // `K²₀₁ = w₀w₁`, so `R₀₁ = b·w₀·w₁` with the `a·K` term identically absent — no cancellation,
    // no extraction, `b` scaled by a constant. With `w = 2⁻ᵏ·(3, 4, 0)` the ingredients are exact
    // in binary: `t2 = 25·4⁻ᵏ`, `θ = 5·2⁻ᵏ`, `w₀w₁ = 12·4⁻ᵏ`, so `R₀₁ = (12/25)(1 − cos θ)` with
    // nothing rounded on the way in.
    //
    // The references are that closed form at 60 decimal digits in mpmath, rounded once to f64 —
    // an oracle the module shares no code with. Eight ulp is the bound; the half-angle form sits
    // at one to two, and `(1.0 − cos θ)/t2` misses by 2.8e-1 at k = 28, 4.9e-10 at k = 16 and
    // 1.5e-12 at k = 10. It is not monotone in θ, because the error is where `cos θ` happens to
    // land on the f64 grid rather than a smooth function of the angle — which is why this samples
    // the whole band instead of trusting one point.
    const REFERENCE: [(i32, f64); 7] = [
        (29, 2.081_668_171_172_168_5e-17),
        (28, 8.326_672_684_688_674e-17),
        (26, 1.332_267_629_550_187_3e-15),
        (22, 3.410_605_131_648_077e-13),
        (16, 1.396_983_861_245_591_4e-9),
        (10, 5.722_034_529_762_763e-6),
        (4, 2.324_738_493_687_734_2e-2),
    ];
    let mut worst = 0.0_f64;
    for (k, want) in REFERENCE {
        let s = (2.0_f64).powi(-k);
        let w = [3.0 * s, 4.0 * s, 0.0];
        let got = exp_so3(w)[0][1];
        let ulps = ((got - want) / want).abs() / f64::EPSILON;
        println!(
            "exp_so3 θ = {:e}: R₀₁ = {got:e} vs {want:e}, {ulps:.2} ulp",
            5.0 * s
        );
        assert!(
            ulps < 8.0,
            "(1−cosθ)/θ² has lost precision at θ = {:e}: {ulps:.3e} ulp",
            5.0 * s
        );
        worst = worst.max(ulps);
    }
    println!("worst (1−cosθ)/θ² error across the band: {worst:.2} ulp");
}

#[test]
fn only_a_symmetric_positive_definite_matrix_is_an_inertia_tensor() {
    // The constructor is the invariant, so this is the list of things it has to refuse. Each one is
    // a matrix that would let the dynamics run and return numbers: an asymmetric tensor gives a
    // non-conservative `I⁻¹`, an indefinite one accelerates the body without bound, a singular one
    // divides by zero, and NaN propagates silently through every step.
    let good = [[140.0, 12.0, -7.0], [12.0, 620.0, 4.0], [-7.0, 4.0, 700.0]];
    let inertia = Inertia::from_tensor(good).unwrap();

    // `I⁻¹ I = E` to rounding — the inverse is formed once and everything downstream trusts it.
    // `good` is symmetric, so its column `j` is its row `j`.
    let mut worst = 0.0_f64;
    for (i, row) in inertia.inverse().iter().enumerate() {
        for (j, column) in good.iter().enumerate() {
            let entry: f64 = row.iter().zip(column).map(|(a, b)| a * b).sum();
            worst = worst.max((entry - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    println!("|I⁻¹I − E| = {worst:.3e}");
    assert!(worst < 1e-15, "the stored inverse is not one: {worst:.3e}");

    let bad: [(&str, Mat3); 5] = [
        (
            "asymmetric",
            [[140.0, 12.0, 0.0], [11.0, 620.0, 0.0], [0.0, 0.0, 700.0]],
        ),
        (
            "negative moment",
            [[-140.0, 0.0, 0.0], [0.0, 620.0, 0.0], [0.0, 0.0, 700.0]],
        ),
        (
            "singular",
            [[140.0, 0.0, 0.0], [0.0, 620.0, 0.0], [0.0, 0.0, 0.0]],
        ),
        (
            "indefinite",
            [[140.0, 900.0, 0.0], [900.0, 620.0, 0.0], [0.0, 0.0, 700.0]],
        ),
        (
            "not a number",
            [
                [140.0, f64::NAN, 0.0],
                [f64::NAN, 620.0, 0.0],
                [0.0, 0.0, 700.0],
            ],
        ),
    ];
    for (why, tensor) in bad {
        assert!(
            Inertia::from_tensor(tensor).is_none(),
            "a {why} matrix was accepted as an inertia tensor"
        );
    }
    // Symmetric to a relative 1e-12 of the trace is the rounding an assembled tensor carries, and it
    // is accepted; ten times that is a mistake, and it is not.
    let trace = good[0][0] + good[1][1] + good[2][2];
    let mut nearly = good;
    nearly[0][1] += 0.9e-12 * trace;
    assert!(
        Inertia::from_tensor(nearly).is_some(),
        "rounding was refused"
    );
    nearly[0][1] += 9.0e-12 * trace;
    assert!(
        Inertia::from_tensor(nearly).is_none(),
        "a visibly asymmetric tensor was accepted"
    );
}

#[test]
fn a_box_split_in_half_reassembles_by_the_parallel_axis_theorem() {
    // Two half-boxes, each with its own tensor about its own centroid and shifted a quarter-length
    // out, must add up to the whole box's tensor. The identity is exact in real arithmetic and it
    // exercises the assembly the way a real spacecraft is built — parts, offsets, sum — while the
    // answer comes from the closed form for the *whole*, which knows nothing of the split.
    let (x, y, z) = (1.2, 1.5, 2.0);
    let mass = 850.0;
    let whole = solid_box_inertia(mass, [x, y, z]);
    let half = solid_box_inertia(0.5 * mass, [0.5 * x, y, z]);
    let assembled = sum_inertia(&[
        translate_inertia(&half, 0.5 * mass, [0.25 * x, 0.0, 0.0]),
        translate_inertia(&half, 0.5 * mass, [-0.25 * x, 0.0, 0.0]),
    ]);

    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((assembled[i][j] - whole[i][j]).abs());
        }
    }
    println!("box {whole:?}\nfrom halves {assembled:?}\nworst |Δ| {worst:.3e}");
    assert!(
        worst < 1e-12 * whole[2][2],
        "the halves do not reassemble: {worst:.3e}"
    );

    // A flat panel is a lamina, so the perpendicular-axis theorem applies: the moment about its
    // normal is the sum of the two in-plane moments. Nothing in `thin_plate_inertia` enforces it.
    let plate = thin_plate_inertia(40.0, [1.4, 2.6]);
    let defect = (plate[2][2] - plate[0][0] - plate[1][1]).abs() / plate[2][2];
    println!(
        "plate diag ({}, {}, {})",
        plate[0][0], plate[1][1], plate[2][2]
    );
    assert!(
        defect < 1e-15,
        "the panel does not obey the perpendicular-axis theorem: {defect:.3e}"
    );
}

#[test]
fn rotating_a_tensor_carries_its_principal_axes_with_it() {
    // `R I Rᵀ` is the tensor of the *rotated* body: its principal axes are the rotated ones, with
    // the same moments. Testing it on the axes rather than entry by entry is what separates it from
    // `Rᵀ I R`, which a quarter turn about a principal axis could not tell apart.
    let moments = [140.0, 620.0, 700.0];
    let rotated = rotate_inertia(&exp_so3([0.31, -0.47, 0.82]), &{
        let mut m = [[0.0; 3]; 3];
        for k in 0..3 {
            m[k][k] = moments[k];
        }
        m
    });
    let r = exp_so3([0.31, -0.47, 0.82]);

    let mut worst = 0.0_f64;
    for k in 0..3 {
        let axis = [r[0][k], r[1][k], r[2][k]]; // R ê_k
        let image = apply(&rotated, axis);
        for i in 0..3 {
            worst = worst.max((image[i] - moments[k] * axis[i]).abs());
        }
    }
    let trace_defect =
        (rotated[0][0] + rotated[1][1] + rotated[2][2] - moments.iter().sum::<f64>()).abs();
    println!("rotated tensor: worst |I'v − λv| {worst:.3e}, trace defect {trace_defect:.3e}");
    assert!(
        worst < 1e-12,
        "the principal axes did not rotate with the body: {worst:.3e}"
    );
    assert!(trace_defect < 1e-12, "the trace moved: {trace_defect:.3e}");
    assert!(
        Inertia::from_tensor(rotated).is_some(),
        "a rotated inertia tensor is still an inertia tensor"
    );
}

#[test]
fn a_constant_moment_about_a_principal_axis_spins_the_body_up_exactly() {
    // The one case with a closed form for both halves of the state. A constant body moment along a
    // principal axis, from rest, keeps ω on that axis — `(Iω)×ω = 0` there — so `ω(t) = Mt/I₃` and
    // the body turns about a *fixed* axis through `½(M/I₃)t²`. Linear ω and a quadratic angle are
    // both integrated exactly by a fourth-order method, so the tolerance is rounding, not truncation,
    // and any wrong quadrature weight or sign shows up at once.
    let inertia = Inertia::principal([120.0, 150.0, 180.0]).unwrap();
    let moment = 0.4;
    let span = 100.0;
    let steps = 1000;
    let (r, w) = integrate_attitude_torqued(
        &IDENTITY,
        [0.0; 3],
        0.0,
        &inertia,
        span / steps as f64,
        steps,
        |_, _, _| [0.0, 0.0, moment],
    );

    let want_rate = moment * span / 180.0;
    let want_angle = 0.5 * (moment / 180.0) * span * span;
    let exact = exp_so3([0.0, 0.0, want_angle]);
    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((r[i][j] - exact[i][j]).abs());
        }
    }
    let rate_error = (w[2] - want_rate).abs() / want_rate + w[0].abs() + w[1].abs();
    println!(
        "spin-up under {moment} N·m: ω = {w:?} (want ẑ·{want_rate}), angle {want_angle} rad, |R − exp| {worst:.3e}"
    );
    // The truncation error is zero here, so what is left is the rounding of a thousand accumulated
    // increments — measured at 2.2e-14 relative, inside the `n·ε = 2.2e-13` such a sum can carry.
    assert!(rate_error < 1e-13, "ω is not Mt/I₃: {rate_error:.3e}");
    assert!(
        worst < 1e-12,
        "the attitude is not the exact rotation: {worst:.3e}"
    );
}

#[test]
fn with_no_moment_the_torqued_integrator_is_the_free_body() {
    // The torqued stepper carries a full inverse where the free one divides by three principal
    // moments, and it forms the stage attitudes the free one never needs. At zero moment the two
    // must still be the same integrator — this is what says the generalisation did not change the
    // method, only what it can be handed.
    let steps = 2000;
    let dt = 0.01;
    let inertia = Inertia::principal(INERTIA).unwrap();
    let (r_free, w_free) = integrate_attitude(&IDENTITY, W0, INERTIA, dt, steps);
    let (r, w) =
        integrate_attitude_torqued(&IDENTITY, W0, 0.0, &inertia, dt, steps, |_, _, _| [0.0; 3]);

    let mut worst_r = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst_r = worst_r.max((r[i][j] - r_free[i][j]).abs());
        }
    }
    let worst_w = (0..3).fold(0.0_f64, |m, i| m.max((w[i] - w_free[i]).abs()));
    println!(
        "free body vs torqued at M = 0 over {steps} steps: |ΔR| {worst_r:.3e}, |Δω| {worst_w:.3e}"
    );
    assert!(worst_r < 1e-12, "the attitudes diverged: {worst_r:.3e}");
    assert!(
        worst_w < 1e-12,
        "the angular velocities diverged: {worst_w:.3e}"
    );
}

/// `R · v`, mapping a body vector into the inertial frame.
fn apply(r: &Mat3, v: Vec3) -> Vec3 {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

#[test]
fn the_inertial_angular_momentum_vector_does_not_move() {
    // Everything above this line is blind to `R`. `|Iω|` and the energy are functions of ω alone,
    // the order test compares ω, and `RᵀR = I` cannot fail for any mutation whatever: `R` is built
    // as `R₀·exp(σ₁)·exp(σ₂)…`, a product of exact exponentials of skew matrices, so orthogonality
    // is guaranteed by the construction rather than earned by the arithmetic. That left the whole
    // RKMK branch -- `ka`, `kb`, `kc`, `kd`, `σ`, and `dexp⁻¹` -- unobserved.
    //
    // The invariant that does see it is the defining conservation law of the free rigid body: the
    // angular momentum is fixed in the *inertial* frame, all three components of `R·(Iω)`, not just
    // its magnitude. It couples `R` to ω, so a wrong rotation breaks it at once.
    let steps = 2000;
    let dt = 0.01;
    let body_l = |w: Vec3| [INERTIA[0] * w[0], INERTIA[1] * w[1], INERTIA[2] * w[2]];
    let l0 = apply(&IDENTITY, body_l(W0));

    let mut worst = 0.0_f64;
    for n in [1, steps / 4, steps / 2, steps] {
        let (r, w) = integrate_attitude(&IDENTITY, W0, INERTIA, dt, n);
        let l = apply(&r, body_l(w));
        for i in 0..3 {
            worst = worst.max((l[i] - l0[i]).abs());
        }
    }
    println!("RKMK4 inertial |ΔL| over {steps} steps: {worst:.3e}");
    assert!(
        worst < 1e-9,
        "the inertial angular momentum moved: {worst:.3e}"
    );
}

#[test]
fn a_spin_about_a_principal_axis_is_an_exact_rotation() {
    // Started along a principal axis, Euler's equation gives `(Iω)×ω = 0`: ω never changes, and the
    // attitude is the exact one-parameter subgroup `R(t) = exp([0,0,ω₃t]×)`. That is a closed form
    // for `R` itself, not for an invariant of it, and it pins the σ quadrature -- the `h/6` and the
    // two doubled stages -- because a wrong weight accumulates a wrong angle every step.
    let w3 = 0.7;
    let steps = 500;
    let dt = 0.01;
    let (r, w) = integrate_attitude(&IDENTITY, [0.0, 0.0, w3], INERTIA, dt, steps);
    let exact = exp_so3([0.0, 0.0, w3 * dt * steps as f64]);

    let mut worst = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            worst = worst.max((r[i][j] - exact[i][j]).abs());
        }
    }
    let spun_up = (w[2] - w3).abs() + w[0].abs() + w[1].abs();
    println!("principal-axis spin: |R − exp(ω₃T)| {worst:.3e}, |Δω| {spun_up:.3e}");
    assert!(spun_up < 1e-15, "ω left the principal axis: {spun_up:.3e}");
    assert!(
        worst < 1e-12,
        "attitude is not the exact rotation: {worst:.3e}"
    );
}

#[test]
fn the_exponential_obeys_the_identities_a_rotation_must() {
    // The one `exp_so3` test above spins about `ẑ` by exactly 2π: two components of the axis are
    // zero and `sin θ = 0`, `cos θ = 1` kill most of Rodrigues' arithmetic. These are the general
    // identities instead, on an axis with no zero component and an angle with no special value:
    // the axis is fixed, the trace gives the angle back, and a perpendicular vector turns by
    // exactly that angle. None of them re-implements the formula under test.
    let axis = {
        let a: Vec3 = [0.3, -0.5, 0.81];
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let theta = 1.234_567;
    let r = exp_so3([axis[0] * theta, axis[1] * theta, axis[2] * theta]);

    let fixed = apply(&r, axis);
    let axis_moved = (0..3).fold(0.0_f64, |m, i| m.max((fixed[i] - axis[i]).abs()));
    let trace = r[0][0] + r[1][1] + r[2][2];

    // Any unit vector perpendicular to the axis; `u × axis` is perpendicular to both.
    let u = {
        let c = [
            axis[1] * 1.0 - axis[2] * 0.0,
            axis[2] * 0.0 - axis[0] * 1.0,
            axis[0] * 0.0 - axis[1] * 0.0,
        ];
        let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        [c[0] / n, c[1] / n, c[2] / n]
    };
    let ru = apply(&r, u);
    let cos_turn = u[0] * ru[0] + u[1] * ru[1] + u[2] * ru[2];
    let cross = [
        u[1] * ru[2] - u[2] * ru[1],
        u[2] * ru[0] - u[0] * ru[2],
        u[0] * ru[1] - u[1] * ru[0],
    ];
    let sin_turn = axis[0] * cross[0] + axis[1] * cross[1] + axis[2] * cross[2];

    println!(
        "exp_so3(θ={theta}): axis drift {axis_moved:.3e}, trace {trace:.15}, \
         turn (cos, sin) = ({cos_turn:.15}, {sin_turn:.15})"
    );
    assert!(
        axis_moved < 1e-15,
        "the axis is not fixed: {axis_moved:.3e}"
    );
    assert!(
        (trace - (1.0 + 2.0 * theta.cos())).abs() < 1e-14,
        "trace {trace} does not encode θ = {theta}"
    );
    assert!(
        (cos_turn - theta.cos()).abs() < 1e-14 && (sin_turn - theta.sin()).abs() < 1e-14,
        "a perpendicular vector did not turn by θ"
    );
}

#[test]
fn the_closed_form_box_and_plate_tensors_are_the_published_ones() {
    // Three distinct edge lengths, so a transposed pair of terms cannot hide behind a symmetry, and
    // the assertion is on `12 I_aa / m`, which is the bracket itself and nothing else. The
    // reassembly test above compares the whole against its own halves and would pass with any
    // consistent formula; this one compares against `y² + z²`, `x² + z²`, `x² + y²` written out.
    //
    // No edge is 0 or 2, and that is not cosmetic: `t·t` and `t + t` agree at exactly those two
    // values, so a squaring written as a doubling is invisible on a 2 m edge — which is what a 2 m
    // edge here hid until a mutation run said so. The `thin_plate_inertia` case below passes
    // `z = 0` by construction and cannot see the `z` term at all, which is why the cuboid has to.
    let (x, y, z) = (1.2, 1.5, 3.0);
    let mass = 850.0;
    let tensor = solid_box_inertia(mass, [x, y, z]);
    for (axis, want) in [(0, 2.25 + 9.0), (1, 1.44 + 9.0), (2, 1.44 + 2.25)] {
        let got = tensor[axis][axis] * 12.0 / mass;
        assert!(
            (got - want).abs() < 1e-12 * want,
            "12 I_{axis}{axis}/m is {got}, not {want}"
        );
    }
    // All three brackets are distinct, so no permutation of them reproduces this tensor.
    assert!(tensor[0][0] > tensor[1][1] && tensor[1][1] > tensor[2][2]);
    for (i, row) in tensor.iter().enumerate() {
        for (j, entry) in row.iter().enumerate() {
            assert!(
                i == j || entry.to_bits() == 0.0_f64.to_bits(),
                "a cuboid on its own axes has no product of inertia: ({i},{j}) = {entry}"
            );
        }
    }

    // The lamina is the cuboid of zero thickness, so its `zz` bracket is the whole of `a² + b²` and
    // its two in-plane brackets are one side each.
    let plate = thin_plate_inertia(40.0, [1.4, 2.6]);
    for (axis, want) in [(0, 2.6 * 2.6), (1, 1.4 * 1.4), (2, 1.4 * 1.4 + 2.6 * 2.6)] {
        let got = plate[axis][axis] * 12.0 / 40.0;
        assert!(
            (got - want).abs() < 1e-12 * want,
            "plate 12 I_{axis}{axis}/m is {got}, not {want}"
        );
    }
}

#[test]
fn the_parallel_axis_shift_is_the_full_dyadic_not_just_its_trace() {
    // `m(|d|²E − d dᵀ)` with an offset that has all three components, so every product in the
    // dyadic and every term of `|d|²` is exercised. The reassembly test shifts only along x, where
    // two thirds of `|d|²` and five of the six dyadic entries are zero.
    let offset: Vec3 = [0.7, -1.3, 2.1];
    let mass = 40.0;
    let shifted = translate_inertia(&[[0.0; 3]; 3], mass, offset);

    // |d|² = 0.49 + 1.69 + 4.41 = 6.59, and every entry below follows from it by hand.
    let want = [
        [244.0, 36.4, -58.8],
        [36.4, 196.0, 109.2],
        [-58.8, 109.2, 87.2],
    ];
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (shifted[i][j] - want[i][j]).abs() < 1e-12,
                "entry ({i},{j}) is {}, not {}",
                shifted[i][j],
                want[i][j]
            );
        }
    }
    // The trace of the shift is `2 m |d|²` — an independent handle on `|d|²` alone, which the six
    // entries above share.
    let trace = shifted[0][0] + shifted[1][1] + shifted[2][2];
    assert!(
        (trace - 2.0 * mass * 6.59).abs() < 1e-12,
        "trace {trace} is not 2m|d|²"
    );
    // And it adds to whatever was there rather than replacing it.
    let base = solid_box_inertia(10.0, [1.0, 2.0, 3.0]);
    let moved = translate_inertia(&base, mass, offset);
    assert!((moved[0][0] - base[0][0] - want[0][0]).abs() < 1e-12);
}

#[test]
fn the_tensor_and_its_inverse_multiply_a_vector_row_by_row() {
    // `I ω` for a full tensor and a velocity with three distinct non-zero components, against the
    // three sums done by hand. Every other test in this file uses a diagonal tensor or a velocity
    // along an axis, either of which leaves two of the three terms in each row multiplied by zero.
    let good = [[140.0, 12.0, -7.0], [12.0, 620.0, 4.0], [-7.0, 4.0, 700.0]];
    let inertia = Inertia::from_tensor(good).unwrap();
    let w: Vec3 = [0.3, -0.7, 1.1];

    // 140(0.3) + 12(−0.7) − 7(1.1) = 42 − 8.4 − 7.7
    // 12(0.3) + 620(−0.7) + 4(1.1) = 3.6 − 434 + 4.4
    // −7(0.3) + 4(−0.7) + 700(1.1) = −2.1 − 2.8 + 770
    let momentum = inertia.apply(w);
    for (axis, want) in [(0, 25.9), (1, -426.0), (2, 765.1)] {
        assert!(
            (momentum[axis] - want).abs() < 1e-12,
            "(Iω)_{axis} is {}, not {want}",
            momentum[axis]
        );
    }
    assert_eq!(
        inertia.tensor(),
        &good,
        "the accepted tensor is the one given"
    );

    // `I⁻¹` runs through the same routine, so solving the momentum back must return the velocity.
    let back = inertia.solve(momentum);
    for axis in 0..3 {
        assert!(
            (back[axis] - w[axis]).abs() < 1e-13,
            "I⁻¹(Iω) is {back:?}, not {w:?}"
        );
    }
}

#[test]
fn the_positive_definite_test_agrees_with_an_independent_expansion() {
    // Sylvester's criterion computed twice: once inside the constructor from a first-row cofactor
    // expansion, once here from the *third-row* one. They are the same number in exact arithmetic
    // and a different sequence of operations in `f64`, so a single wrong operator inside the
    // constructor's determinant cannot survive agreement over a sweep that crosses the boundary in
    // both directions.
    let mut state = 0x2545_F491_4F6C_DD1D_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // In [−3, 3].
        (state >> 11) as f64 * (6.0 / 9_007_199_254_740_992.0) - 3.0
    };

    let (mut accepted, mut rejected) = (0usize, 0usize);
    for _ in 0..600 {
        // A diagonal biased positive, so a useful fraction of the draws are definite.
        let (xx, yy, zz) = (next() + 3.2, next() + 3.2, next() + 3.2);
        let (xy, xz, yz) = (next(), next(), next());
        let tensor = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]];
        let minor2 = xx * yy - xy * xy;
        let determinant =
            xz * (xy * yz - yy * xz) - yz * (xx * yz - xy * xz) + zz * (xx * yy - xy * xy);
        let want = xx > 0.0 && minor2 > 0.0 && determinant > 0.0;
        assert_eq!(
            Inertia::from_tensor(tensor).is_some(),
            want,
            "{tensor:?} has minors ({xx}, {minor2}, {determinant})"
        );
        if want { accepted += 1 } else { rejected += 1 }
    }
    println!("sweep: {accepted} accepted, {rejected} rejected");
    assert!(
        accepted >= 50 && rejected >= 50,
        "the sweep has to straddle the boundary: {accepted} accepted, {rejected} rejected"
    );

    // Three matrices that are not random, each one a specific way the criterion can be got wrong.
    //
    // Indefinite with a *positive* determinant: the leading 2×2 minor is −7 while `det` is +7, so a
    // criterion that tests the determinant alone — or that takes any one minor as sufficient —
    // accepts a body whose kinetic energy is unbounded below.
    assert!(Inertia::from_tensor([[3.0, 4.0, 0.0], [4.0, 3.0, 0.0], [0.0, 0.0, -1.0]]).is_none());
    // An infinite moment has an infinite trace, and every minor formed from it is also infinite and
    // therefore "positive". The finiteness gate is what refuses it, not the minors.
    assert!(
        Inertia::from_tensor([[f64::INFINITY, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .is_none()
    );
    // Asymmetry of exactly the tolerance is accepted — the bound is inclusive, and an assembled
    // tensor that lands exactly on it is rounding, not a mistake.
    let tol = 1e-12 * 3.0;
    assert!(
        Inertia::from_tensor([[1.0, tol, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).is_some(),
        "an asymmetry of exactly 1e-12 of the trace was refused"
    );

    // Scale, not shape: below about 1e-100 the second minor and the determinant *underflow* to zero
    // even for a diagonal body that is manifestly definite, and the constructor then refuses it. It
    // is the right refusal — a validating constructor should not accept what its own arithmetic
    // cannot verify — but it is a scale limit rather than a physical one, and it is here so that it
    // is a recorded property rather than a surprise. No spacecraft has 1e-120 kg·m².
    assert!(
        Inertia::principal([1e-120, 1e-120, 1e-120]).is_none(),
        "the determinant underflowed and the body was accepted anyway"
    );
    assert!(
        Inertia::principal([1e-200, 1e-200, 1e300]).is_none(),
        "the second minor underflowed and the body was accepted anyway"
    );
    // The same shapes at spacecraft scale are accepted, so the refusals above are the scale and not
    // the diagonal.
    assert!(Inertia::principal([1.0, 1.0, 1.0]).is_some());
    assert!(Inertia::principal([1e-4, 1e-4, 1e4]).is_some());
}

#[test]
fn the_torqued_derivative_is_eulers_equation_with_the_moment() {
    // `ω̇ = I⁻¹(M + (Iω) × ω)`, component by component, by Richardson-extrapolating the integrator's
    // own first step. The spin-up test above puts both the moment and the angular velocity on one
    // principal axis, where the gyroscopic term vanishes and two of the three components are zero;
    // here every component of `M` and of `(Iω) × ω` is non-zero and they are of comparable size.
    let inertia = Inertia::principal([1.0, 2.0, 3.0]).unwrap();
    let w0: Vec3 = [1.0, 0.3, 0.6];
    let moment: Vec3 = [0.4, -0.9, 0.25];

    let momentum = [w0[0], 2.0 * w0[1], 3.0 * w0[2]];
    let gyroscopic = [
        momentum[1] * w0[2] - momentum[2] * w0[1],
        momentum[2] * w0[0] - momentum[0] * w0[2],
        momentum[0] * w0[1] - momentum[1] * w0[0],
    ];
    let moments: Vec3 = [1.0, 2.0, 3.0];
    let want: Vec3 = [
        (moment[0] + gyroscopic[0]) / moments[0],
        (moment[1] + gyroscopic[1]) / moments[1],
        (moment[2] + gyroscopic[2]) / moments[2],
    ];
    for (axis, term) in gyroscopic.iter().enumerate() {
        assert!(
            term.abs() > 0.1 * moment[axis].abs(),
            "component {axis} would not show a lost gyroscopic term"
        );
    }

    let derivative = |h: f64| -> Vec3 {
        let (_, w) =
            integrate_attitude_torqued(&IDENTITY, w0, 0.0, &inertia, h, 1, |_, _, _| moment);
        [(w[0] - w0[0]) / h, (w[1] - w0[1]) / h, (w[2] - w0[2]) / h]
    };
    let h = 1e-3;
    let coarse = derivative(h);
    let fine = derivative(0.5 * h);
    // D(h) = ω̇ + (h/2)ω̈ + O(h²), so 2D(h/2) − D(h) removes the linear term and leaves the
    // quadratic one — measured 3.5e-9 at h = 1e-3, which is the `h²ω⃛/4` this cannot subtract and
    // not a defect. Shrinking `h` further trades it for the cancellation in `ω(h) − ω₀`.
    for axis in 0..3 {
        let extrapolated = 2.0 * fine[axis] - coarse[axis];
        println!(
            "ω̇_{axis} residual {:.3e}",
            (extrapolated - want[axis]).abs()
        );
        assert!(
            (extrapolated - want[axis]).abs() < 1e-6,
            "ω̇_{axis} is {extrapolated}, not {}",
            want[axis]
        );
    }
}

#[test]
fn a_time_dependent_moment_is_sampled_at_the_stage_times() {
    // `M(t) = k t x̂` on a spherical body from rest: `ω̇ = kt/I` is linear in `t`, so the four-stage
    // quadrature is Simpson's rule and is *exact* — `ω(T) = kT²/(2I)` and the swept angle is
    // `kT³/(6I)`, both to rounding. Nothing else in this file hands the integrator a moment that
    // depends on `t` at all, so the three stage times could be anything.
    let inertia = Inertia::principal([2.0, 2.0, 2.0]).unwrap();
    let k = 0.75;
    let total = 4.0;
    let steps = 8;
    let (attitude, w) = integrate_attitude_torqued(
        &IDENTITY,
        [0.0; 3],
        0.0,
        &inertia,
        total / steps as f64,
        steps,
        |_, _, t| [k * t, 0.0, 0.0],
    );

    let want_rate = k * total * total / (2.0 * 2.0);
    assert!(
        (w[0] - want_rate).abs() < 1e-13 * want_rate,
        "ω_x is {}, not kT²/2I = {want_rate}",
        w[0]
    );
    assert!(
        w[1].abs() < 1e-15 && w[2].abs() < 1e-15,
        "a moment along x spun the body about {w:?}"
    );

    // The attitude is a rotation about x through ∫ω dt = kT³/(6I), a cubic the same quadrature
    // integrates exactly.
    let want_angle = k * total * total * total / (6.0 * 2.0);
    let got_angle = attitude[2][1].atan2(attitude[1][1]);
    let turns = ((want_angle - got_angle) / core::f64::consts::TAU).round();
    assert!(
        (got_angle + turns * core::f64::consts::TAU - want_angle).abs() < 1e-12,
        "the swept angle is {got_angle} + {turns} turns, not kT³/6I = {want_angle}"
    );
}
