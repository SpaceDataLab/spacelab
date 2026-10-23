//! Environmental torques on a box-wing: each one against a closed form it cannot have inherited from
//! the implementation, plus the GEO torque budget the design of an attitude-control system starts
//! from.

#![allow(
    clippy::unwrap_used,
    // The two array comparisons below are between a method and the free function it forwards to,
    // on the same inputs: bit-identical or the delegation is broken, which is the point.
    clippy::float_cmp
)]

use spacelab_propagation::attitude::{Inertia, Mat3, Vec3, exp_so3, integrate_attitude_torqued};
use spacelab_propagation::spacecraft::{BoxWing, Facet};
use spacelab_propagation::torque::{RigidBoxWing, box6_arms, gravity_gradient, magnetic};

/// IAU 2015 Resolution B3 nominal total solar irradiance at 1 au, W·m⁻².
const SOLAR_FLUX: f64 = 1361.0;
/// Exact by the SI definition of the metre, m·s⁻¹.
const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// Solar radiation pressure at 1 au on a perfectly absorbing surface, N·m⁻².
const SOLAR_PRESSURE: f64 = SOLAR_FLUX / SPEED_OF_LIGHT;

fn norm(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn unit_inertia() -> Inertia {
    Inertia::principal([120.0, 150.0, 180.0]).unwrap()
}

/// One flat panel of `area` facing **+Z**, its centre of pressure `arm` metres along **+X** of the
/// centre of mass. Everything about the geometry is chosen so the answer is a single component.
fn one_panel(area: f64, arm: f64, absorption: f64, reflection: f64, drag: f64) -> RigidBoxWing<1> {
    RigidBoxWing {
        shape: BoxWing {
            facets: [Facet::fixed(
                [0.0, 0.0, 1.0],
                area,
                false,
                drag,
                0.0,
                absorption,
                reflection,
            )],
            mass: 1000.0,
            drag_factor: 1.0,
            radiation_factor: 1.0,
        },
        arms: [[arm, 0.0, 0.0]],
        inertia: unit_inertia(),
        dipole: [0.0, 0.0, 0.0],
    }
}

#[test]
fn an_absorbing_panel_feels_the_flat_plate_pressure_moment() {
    // A black plate normal to the Sun absorbs the whole momentum flux and re-emits none of it, so
    // the force is `P·A` along the flux and nothing else — the one case where the Orekit reflection
    // model collapses to a formula with no free parameters. With the arm perpendicular to it the
    // moment is `P·A·d` about ŷ exactly, and the two components that should vanish are what catch a
    // transposed cross product.
    let (area, arm) = (12.0, 0.8);
    let body = one_panel(area, arm, 1.0, 0.0, 0.0);
    let flux = [0.0, 0.0, -SOLAR_PRESSURE]; // Sun on the +Z side, pushing the panel −Z
    let torque = body.srp_torque(&flux, &[0.0, 0.0, 1.0]);

    let want = SOLAR_PRESSURE * area * arm;
    let error = (torque[1] - want).abs() / want;
    println!("absorbing panel: T = {torque:?} N·m, closed form T_y = {want:.6e} N·m ({error:.2e})");
    assert!(error < 1e-15, "moment is not P·A·d: {error:.3e}");
    assert_eq!(
        (torque[0], torque[2]),
        (0.0, 0.0),
        "a moment appeared off the arm×force axis"
    );
}

#[test]
fn a_windward_panel_feels_the_dynamic_pressure_moment() {
    // Same geometry for drag: a plate normal to the flow, no lift, so the force is the textbook
    // `½ρv²·C_D·A` along the flow. Nothing here shares an arithmetic path with the SRP case above.
    let (area, arm, drag) = (12.0, 0.8, 2.2);
    let body = one_panel(area, arm, 1.0, 0.0, drag);
    let (rho, speed) = (2.5e-12, 7_600.0);
    let torque = body.drag_torque(rho, &[0.0, 0.0, -speed], &[0.0, 0.0, 1.0]);

    let want = 0.5 * rho * speed * speed * drag * area * arm;
    let error = (torque[1] - want).abs() / want;
    println!("windward panel: T = {torque:?} N·m, closed form T_y = {want:.6e} N·m ({error:.2e})");
    assert!(error < 1e-15, "moment is not ½ρv²·C_D·A·d: {error:.3e}");
    assert_eq!(
        (torque[0], torque[2]),
        (0.0, 0.0),
        "a moment appeared off the arm×force axis"
    );
}

#[test]
fn moving_the_centre_of_mass_moves_the_moment_by_exactly_c_cross_f() {
    // The one identity that ties this module to the force model it borrows from: shifting every arm
    // by the same `c` must change the total moment by `−c × F`, with `F` the total force — which the
    // force model reports as `m·a` and this module never computes. It holds for any illumination, so
    // the flux is deliberately off-axis, lighting three facets whose individual moments do not
    // vanish. A facet whose arm is paired with the wrong facet's force breaks it immediately.
    let (x, y, z) = (1.2, 1.5, 2.0);
    let mass = 850.0;
    let shape = BoxWing::<6>::box6(x, y, z, mass, 2.2, 0.0, 0.7, 0.2);
    let centroids = box6_arms(x, y, z);
    let offset = [0.11, -0.07, 0.23];

    let direction = {
        let d = [0.48, -0.6, -0.64]; // unit: 0.48² + 0.6² + 0.64² = 1
        [
            SOLAR_PRESSURE * d[0],
            SOLAR_PRESSURE * d[1],
            SOLAR_PRESSURE * d[2],
        ]
    };
    let sun = [-0.48, 0.6, 0.64];

    let at_centre = RigidBoxWing {
        shape,
        arms: centroids,
        inertia: unit_inertia(),
        dipole: [0.0; 3],
    };
    let mut shifted = at_centre;
    for (arm, centroid) in shifted.arms.iter_mut().zip(centroids) {
        *arm = [
            centroid[0] - offset[0],
            centroid[1] - offset[1],
            centroid[2] - offset[2],
        ];
    }

    let acceleration = shape.srp_acceleration(&direction, &sun);
    let force = [
        mass * acceleration[0],
        mass * acceleration[1],
        mass * acceleration[2],
    ];
    let t0 = at_centre.srp_torque(&direction, &sun);
    let moved = shifted.srp_torque(&direction, &sun);
    // −c × F = F × c.
    let want = [
        force[1] * offset[2] - force[2] * offset[1],
        force[2] * offset[0] - force[0] * offset[2],
        force[0] * offset[1] - force[1] * offset[0],
    ];

    let residual = norm([
        moved[0] - t0[0] - want[0],
        moved[1] - t0[1] - want[1],
        moved[2] - t0[2] - want[2],
    ]);
    let relative = residual / norm(want);
    println!(
        "box under off-axis flux: T(centre) = {t0:?}, T(shifted) − T(centre) = {:?}, F × c = {want:?}, residual {residual:.3e} ({relative:.2e})",
        [moved[0] - t0[0], moved[1] - t0[1], moved[2] - t0[2]]
    );
    assert!(
        relative < 1e-15,
        "the moment did not shift by F × c: {relative:.3e}"
    );
    assert!(
        norm(moved) > 1e-9,
        "the test flux lit nothing: T = {moved:?}"
    );

    // And `T(centre)` is *identically* zero, which is worth stating rather than passing over. Each
    // facet's normal force acts along its own arm and makes no moment; the tangential parts sum to
    // `Σᵢ (V/2)(n̂ᵢ·f)(n̂ᵢ × f)(C_s−1)/psr`, in which the box's three lit faces all carry the same
    // `A·L/2 = V/2` and the sum telescopes to `(Σ f_k ê_k) × f = f × f = 0`. It is why a symmetric
    // body with its centre of mass at its centre is torque-free under *any* illumination, and why
    // the offset above is what the whole disturbance is made of.
    assert!(
        norm(t0) < 1e-15 * norm(force),
        "a centred box felt a net solar moment: {t0:?}"
    );
}

#[test]
fn the_gravity_gradient_on_a_slender_body_is_the_sin_two_theta_law() {
    // For an axisymmetric body — moments `(I_x, I_t, I_t)` — with the primary's direction at angle θ
    // from the symmetry axis in the x–z plane, the whole torque is about ŷ and equals
    // `(3μ/R³)·½(I_x − I_t)·sin 2θ`. It is the standard result (Wertz eq. 17-51 specialised, Markley
    // & Crassidis §3.4.2) and it fixes the factor 3, the R⁻³ scaling and the sign together.
    let (ix, it) = (140.0, 620.0);
    let inertia = Inertia::principal([ix, it, it]).unwrap();
    let mu = 3.986_004_418e14;
    let radius = 7.0e6;

    let mut worst: f64 = 0.0;
    for degrees in [5.0_f64, 23.0, 45.0, 67.0, 100.0, 174.0] {
        let theta: f64 = degrees.to_radians();
        let r = [radius * theta.cos(), 0.0, radius * theta.sin()];
        let got = gravity_gradient(&inertia, r, mu);
        let want = 3.0 * mu / radius.powi(3) * 0.5 * (ix - it) * (2.0 * theta).sin();
        let error = (got[1] - want).abs() / want.abs();
        println!("θ = {degrees:5.1}°: T_y = {:.6e} vs {want:.6e} N·m", got[1]);
        assert!(
            got[0] == 0.0 && got[2] == 0.0,
            "an out-of-plane moment appeared at θ = {degrees}°: {got:?}"
        );
        worst = worst.max(error);
    }
    println!("worst relative error against the sin 2θ law: {worst:.3e}");
    assert!(worst < 1e-14, "not the sin 2θ law: {worst:.3e}");
}

#[test]
fn the_gravity_gradient_is_blind_to_the_sign_of_the_direction() {
    // Both factors of `R × I R` flip together, so nadir and zenith give the same torque — bit for
    // bit, since negation is exact in binary floating point. This is why the module takes a position
    // vector and not a "nadir direction" whose convention a caller could get backwards.
    let inertia =
        Inertia::from_tensor([[140.0, 12.0, -7.0], [12.0, 620.0, 4.0], [-7.0, 4.0, 700.0]])
            .unwrap();
    let mu = 3.986_004_418e14;
    let r = [3.1e6, -5.4e6, 2.2e6];
    let up = gravity_gradient(&inertia, r, mu);
    let down = gravity_gradient(&inertia, [-r[0], -r[1], -r[2]], mu);
    println!("R → −R: {up:?} vs {down:?}");
    for i in 0..3 {
        assert_eq!(
            up[i].to_bits(),
            down[i].to_bits(),
            "the gravity gradient changed sense with the vector"
        );
    }
}

#[test]
fn a_spherically_symmetric_body_feels_no_gravity_gradient() {
    // `I = λE` makes `I R` parallel to `R`, and the products cancel exactly rather than to a
    // tolerance. A body has to be *unequal* about its axes to feel the gradient at all.
    let inertia = Inertia::principal([450.0, 450.0, 450.0]).unwrap();
    let torque = gravity_gradient(&inertia, [3.1e6, -5.4e6, 2.2e6], 3.986_004_418e14);
    println!("sphere: T = {torque:?}");
    assert!(
        torque[0] == 0.0 && torque[1] == 0.0 && torque[2] == 0.0,
        "a sphere felt a gravity gradient: {torque:?}"
    );
}

#[test]
fn the_magnetic_moment_is_perpendicular_to_the_dipole_and_the_field() {
    // `m × B` with a 1 A·m² residual dipole in a 30 000 nT field — a small satellite's usual order —
    // is tens of μN·m, comparable to everything else in low orbit. The check is the cross product's
    // two defining properties, magnitude `mB sin α` and perpendicularity to both factors, neither of
    // which a transposed component survives.
    let dipole = [0.6, -0.3, 0.2];
    let field = [22_000e-9, -14_000e-9, 31_000e-9]; // tesla, from IGRF's nanotesla
    let torque = magnetic(dipole, field);

    let cos_alpha = (dipole[0] * field[0] + dipole[1] * field[1] + dipole[2] * field[2])
        / (norm(dipole) * norm(field));
    let want = norm(dipole) * norm(field) * (1.0 - cos_alpha * cos_alpha).sqrt();
    let error = (norm(torque) - want).abs() / want;
    let along_dipole = torque[0] * dipole[0] + torque[1] * dipole[1] + torque[2] * dipole[2];
    let along_field = torque[0] * field[0] + torque[1] * field[1] + torque[2] * field[2];
    println!(
        "m × B = {torque:?} N·m, |T| = {:.6e} vs mB sin α = {want:.6e}, projections {along_dipole:.3e} / {along_field:.3e}",
        norm(torque)
    );
    assert!(error < 1e-14, "|m × B| is not mB sin α: {error:.3e}");
    assert!(
        along_dipole.abs() < 1e-24 && along_field.abs() < 1e-24,
        "the moment is not perpendicular to its factors"
    );
}

#[test]
fn at_geo_solar_radiation_dominates_the_aerodynamic_moment() {
    // The budget an attitude-control system is sized from. Same 100 m² array, same 1 m arm, same
    // attitude, so every geometric factor divides out and what is left is the ratio of the two
    // pressures: 4.54 μPa of sunlight against ½ρv² of atmosphere.
    //
    // The density is the pessimistic end of the GEO range — 1e-16 kg·m⁻³ is a disturbed upper bound,
    // against the ~1e-19 the quiet models give — precisely so that the assertion cannot be met by
    // choosing a favourable atmosphere. At 1e-19 the margin is three further orders.
    let (area, arm) = (100.0, 1.0);
    let body = one_panel(area, arm, 0.7, 0.2, 2.2);
    let (rho, speed) = (1.0e-16, 3_074.7); // GEO circular speed
    let sun = [0.0, 0.0, 1.0];

    let srp = body.srp_torque(&[0.0, 0.0, -SOLAR_PRESSURE], &sun);
    let aero = body.drag_torque(rho, &[0.0, 0.0, -speed], &sun);
    let ratio = norm(srp) / norm(aero);

    // At normal incidence the Orekit reflection model gives `P·A·(1 + C_s + ⅔C_d)`.
    let want_srp = SOLAR_PRESSURE * area * arm * (1.0 + 0.2 + 2.0 / 3.0 * 0.1);
    let error = (norm(srp) - want_srp).abs() / want_srp;
    println!(
        "GEO budget, 100 m² at 1 m: SRP {:.4} μN·m (closed form {:.4}, {error:.2e}), aero {:.4e} μN·m, ratio {ratio:.3e}",
        norm(srp) * 1e6,
        want_srp * 1e6,
        norm(aero) * 1e6
    );
    assert!(
        error < 1e-14,
        "SRP is not P·A(1 + C_s + ⅔C_d)·d: {error:.3e}"
    );
    assert!(
        (500.0..1100.0).contains(&(norm(srp) * 1e6)),
        "the solar moment left its expected band: {:.3} μN·m",
        norm(srp) * 1e6
    );
    assert!(
        ratio > 1e3,
        "solar radiation does not dominate at GEO: ratio {ratio:.3e}"
    );
}

#[test]
fn a_gravity_gradient_librates_a_slender_body_about_the_vertical() {
    // The torque and the integrator together, against a period nothing in either computes.
    //
    // Hold the direction to the primary fixed in inertial space and let a body with moments
    // `(I_x, I_t, I_t)`, `I_x < I_t`, hang from it: the gravity gradient is `3n²·½(I_x − I_t)·sin 2θ`
    // about ŷ, so for small θ the body is a pendulum with `θ̈ = −3n²(I_t − I_x)/I_t · θ` and period
    // `2π / (n√(3(I_t − I_x)/I_t))`. That the minor axis is the stable one is the reason
    // gravity-gradient booms point down at all.
    //
    // Fixing the vertical is what makes the closed form this simple: in a real orbit the vertical
    // itself turns at `n` and the pitch libration picks up the familiar orbital-rate terms. This is
    // the pendulum limit, and it is deliberately the one with no free parameters.
    let (ix, it) = (140.0, 620.0);
    let inertia = Inertia::principal([ix, it, it]).unwrap();
    let mu = 3.986_004_418e14;
    let radius = 7.0e6_f64;
    let mean_motion = (mu / radius.powi(3)).sqrt();
    let want_period = std::f64::consts::TAU / (mean_motion * (3.0 * (it - ix) / it).sqrt());

    let theta0 = 0.02;
    let mut r = exp_so3([0.0, theta0, 0.0]);
    let mut w = [0.0; 3];
    let dt = 1.0;
    let moment = |r: &Mat3, _w: Vec3, _t: f64| {
        // Body-frame vector to the primary is Rᵀ·(inertial vector), the attitude mapping body→inertial.
        let to_primary = [r[0][0] * radius, r[0][1] * radius, r[0][2] * radius];
        gravity_gradient(&inertia, to_primary, mu)
    };

    // θ from the attitude: R = exp([0, θ, 0]) has R₀₀ = cos θ and R₂₀ = −sin θ.
    let angle = |r: &Mat3| (-r[2][0]).atan2(r[0][0]);
    let mut crossings: Vec<f64> = Vec::new();
    let mut previous = theta0;
    let mut worst_amplitude: f64 = 0.0;
    for step in 1..12_000 {
        let (rn, wn) = integrate_attitude_torqued(&r, w, 0.0, &inertia, dt, 1, moment);
        r = rn;
        w = wn;
        let theta = angle(&r);
        worst_amplitude = worst_amplitude.max(theta.abs());
        if previous.signum() != theta.signum() {
            // Linear interpolation onto the zero crossing; the sampling is 1 s in a 3800 s period.
            let t = f64::from(step) * dt - dt * theta / (theta - previous);
            crossings.push(t);
        }
        previous = theta;
    }

    assert!(
        crossings.len() >= 3,
        "the body did not librate: {} crossings in 12 000 s",
        crossings.len()
    );
    let measured = crossings[2] - crossings[0];
    let error = (measured - want_period).abs() / want_period;
    println!(
        "libration: measured period {measured:.2} s vs closed form {want_period:.2} s ({error:.2e}), \
         amplitude stayed at {worst_amplitude:.6} rad from {theta0}"
    );
    assert!(
        error < 1e-3,
        "the libration period is not 2π/(n√(3(I_t−I_x)/I_t)): {error:.3e}"
    );
    // A conservative torque returns the body to its starting amplitude; a sign error in the
    // integrator's gyroscopic term would pump or damp it.
    assert!(
        (worst_amplitude - theta0).abs() < 1e-6,
        "the libration amplitude drifted: {worst_amplitude} from {theta0}"
    );
}

#[test]
fn moving_the_centre_of_mass_moves_the_aerodynamic_moment_by_exactly_c_cross_f() {
    // The `F × c` identity again, for drag rather than solar pressure. It is not the same code path
    // — a different facet loop, a different coefficient, a different cut-off for the facets facing
    // away — and the summation that turns per-facet moments into a total is its own.
    let (x, y, z) = (1.2, 1.5, 2.0);
    let mass = 850.0;
    let shape = BoxWing::<6>::box6(x, y, z, mass, 2.2, 0.0, 0.7, 0.2);
    let centroids = box6_arms(x, y, z);
    let offset = [0.11, -0.07, 0.23];

    // Off-axis, so three faces are windward and the total moment has all three components.
    let speed = 7500.0;
    let v_rel = [0.48 * speed, -0.6 * speed, -0.64 * speed];
    let sun = [-0.48, 0.6, 0.64];
    let rho = 3.0e-12;

    let at_centre = RigidBoxWing {
        shape,
        arms: centroids,
        inertia: unit_inertia(),
        dipole: [0.0; 3],
    };
    let mut shifted = at_centre;
    for (arm, centroid) in shifted.arms.iter_mut().zip(centroids) {
        *arm = [
            centroid[0] - offset[0],
            centroid[1] - offset[1],
            centroid[2] - offset[2],
        ];
    }

    let acceleration = shape.drag_acceleration(rho, &v_rel, &sun);
    let force = [
        mass * acceleration[0],
        mass * acceleration[1],
        mass * acceleration[2],
    ];
    let t0 = at_centre.drag_torque(rho, &v_rel, &sun);
    let moved = shifted.drag_torque(rho, &v_rel, &sun);
    let want = [
        force[1] * offset[2] - force[2] * offset[1],
        force[2] * offset[0] - force[0] * offset[2],
        force[0] * offset[1] - force[1] * offset[0],
    ];
    // Every component has to be worth measuring, or a lost accumulator hides in a zero.
    for axis in 0..3 {
        assert!(
            want[axis].abs() > 0.05 * norm(want),
            "component {axis} of F × c is negligible: {want:?}"
        );
    }

    let residual = norm([
        moved[0] - t0[0] - want[0],
        moved[1] - t0[1] - want[1],
        moved[2] - t0[2] - want[2],
    ]);
    println!(
        "box in a 7.5 km/s wind: T(centre) = {t0:?}, T(shifted) − T(centre) = {:?}, F × c = {want:?}",
        [moved[0] - t0[0], moved[1] - t0[1], moved[2] - t0[2]]
    );
    assert!(
        residual / norm(want) < 1e-14,
        "the aerodynamic moment did not shift by F × c: {:.3e}",
        residual / norm(want)
    );
    // And a centred box feels no net aerodynamic moment either, for the same telescoping reason.
    assert!(
        norm(t0) < 1e-14 * norm(force),
        "a centred box felt a net aerodynamic moment: {t0:?}"
    );
}

#[test]
fn the_six_box_arms_are_the_face_centroids() {
    // `box6`'s facet order is `−X, +X, −Y, +Y, −Z, +Z`, and each arm is that face's centroid: half
    // the box's own dimension out along its normal. Getting a sign or a factor wrong here pairs
    // every facet's force with the wrong lever and is invisible to any test that only checks a
    // total, because the box is symmetric.
    let arms = box6_arms(2.0, 4.0, 6.0);
    let want = [
        [-1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, -3.0],
        [0.0, 0.0, 3.0],
    ];
    assert_eq!(
        arms, want,
        "the face centroids are not half a dimension out"
    );
}

#[test]
fn the_body_methods_are_the_free_functions_on_this_bodys_own_properties() {
    // `RigidBoxWing`'s gravity-gradient and magnetic methods carry the body's stored inertia and
    // dipole into the free functions, and there is nothing else in them. What has to be checked is
    // that they carry the *body's* values — a method that ignored its `self` and returned zero, or
    // that read the wrong field, would pass any test written only against the free functions.
    let inertia = Inertia::principal([120.0, 480.0, 500.0]).unwrap();
    let dipole = [0.4, -0.15, 0.9];
    let shape = BoxWing::<6>::box6(1.0, 1.0, 1.0, 100.0, 2.2, 0.0, 0.7, 0.2);
    let body = RigidBoxWing {
        shape,
        arms: box6_arms(1.0, 1.0, 1.0),
        inertia,
        dipole,
    };

    // A direction that is not a principal axis, so `I R` is not parallel to `R` and the torque is
    // not zero.
    let nadir = [3.0e6, -4.0e6, 5.0e6];
    let mu = 3.986_004_418e14;
    let gradient = body.gravity_gradient_torque(nadir, mu);
    assert_eq!(gradient, gravity_gradient(&inertia, nadir, mu));
    assert!(
        norm(gradient) > 1.0e-9,
        "the test geometry produced no gravity gradient: {gradient:?}"
    );

    let field = [2.1e-5, -1.4e-5, 3.3e-5];
    let magnetic_moment = body.magnetic_torque(field);
    assert_eq!(magnetic_moment, magnetic(dipole, field));
    assert!(
        norm(magnetic_moment) > 1.0e-9,
        "the test field produced no magnetic moment: {magnetic_moment:?}"
    );
}
