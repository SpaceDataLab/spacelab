//! The photometric forward model: the magnitude zero point against the photometric route, the
//! reflectance model against its own energy budget, and a spinning box against the symmetry its
//! light curve has to have.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use core::f64::consts::{FRAC_PI_2, PI, TAU};

use spacelab_propagation::attitude::exp_so3;
use spacelab_propagation::lightcurve::{
    PhotometricBoxWing, SOLAR_IRRADIANCE, SUN_V_MAGNITUDE, SiderealPeriod, SynodicPeriod,
    ViewGeometry,
};
use spacelab_propagation::spacecraft::{BoxWing, Facet};

/// One flat facet facing **+Z**, with the given coefficients and specular exponent.
fn one_facet(area: f64, absorption: f64, reflection: f64, exponent: f64) -> PhotometricBoxWing<1> {
    PhotometricBoxWing {
        shape: BoxWing {
            facets: [Facet::fixed(
                [0.0, 0.0, 1.0],
                area,
                false,
                2.2,
                0.0,
                absorption,
                reflection,
            )],
            mass: 500.0,
            drag_factor: 1.0,
            radiation_factor: 1.0,
        },
        specular_exponent: [exponent],
    }
}

#[test]
fn the_magnitude_zero_point_agrees_with_the_photometric_route() {
    // This module's chain is radiometric end to end: an irradiance ratio and the Sun's magnitude.
    // The photometric literature states the same zero point the other way round — a `V = 0` source
    // delivers 2.54e-6 lux — and the two have to agree, or one of the two constants is wrong.
    //
    // Run the photometric definition up to the Sun: 2.54e-6 · 10^(0.4 × 26.74) = 1.26e5 lux, against
    // the measured extraterrestrial solar illuminance of about 1.3e5 lux. They agree to 3%, which is
    // the precision `m_☉ = −26.74` itself carries (0.03 mag is 3% in flux).
    //
    // Nothing downstream depends on the conversion, and that is the point: because the model divides
    // by `E_☉` in the same units it multiplies by, the luminous efficacy of sunlight — 1361 W·m⁻²
    // maps to 1.3e5 lux, so about 95 lm/W, not the 683 lm/W of monochromatic 555 nm light — cancels
    // out entirely. Getting it wrong would be a 7× error if the chain were photometric.
    const ZERO_MAGNITUDE_LUX: f64 = 2.54e-6;
    const SOLAR_ILLUMINANCE_LUX: f64 = 1.3e5;

    let implied = ZERO_MAGNITUDE_LUX * libm::pow(10.0, -0.4 * SUN_V_MAGNITUDE);
    let disagreement = (implied - SOLAR_ILLUMINANCE_LUX).abs() / SOLAR_ILLUMINANCE_LUX;
    let percent = 100.0 * disagreement;
    let efficacy = SOLAR_ILLUMINANCE_LUX / SOLAR_IRRADIANCE;
    println!(
        "V=0 at {ZERO_MAGNITUDE_LUX:e} lux and m_☉ = {SUN_V_MAGNITUDE} imply {implied:.3e} lux for the Sun, \
         against {SOLAR_ILLUMINANCE_LUX:.3e} measured ({percent:.1} %); solar efficacy {efficacy:.1} lm/W"
    );
    assert!(
        disagreement < 0.05,
        "the magnitude zero point and the solar magnitude disagree by {percent:.1} %"
    );
}

/// One Lambertian facet against `I = E·A·C_d/π` and the magnitude worked by hand. (G3)
#[test]
fn a_lambertian_plate_at_a_thousand_kilometres_is_fourth_magnitude() {
    // The whole chain on one facet, against arithmetic done by hand:
    //   I = E·A·cos θᵢ·C_d/π · cos θₒ = 1361 × 1 × 1 × (1/π) × 1 = 433.2 W/sr
    //   E_obs = I/d² = 433.2 / (1e6)² = 4.332e-10 W/m²
    //   m = −26.74 − 2.5 log₁₀(4.332e-10 / 1361) = −26.74 + 31.24 = +4.50
    //
    // The sanity check on that number is the ISS: about 1000 m² of solar array at 400 km is 7.5
    // magnitudes brighter for the area and 2 more for the range, giving −5, and the ISS is reported
    // at −4 to −5.
    let plate = one_facet(1.0, 0.0, 0.0, 0.0);
    let view = ViewGeometry::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0e6, 1.0).unwrap();

    let intensity = plate.radiant_intensity(&view);
    let want_intensity = SOLAR_IRRADIANCE / PI;
    let want_magnitude =
        SUN_V_MAGNITUDE - 2.5 * libm::log10(want_intensity / (1.0e6 * 1.0e6) / SOLAR_IRRADIANCE);
    let magnitude = plate.apparent_magnitude(&view);

    println!(
        "1 m² Lambertian plate at 1000 km, zero phase: I = {intensity:.4} W/sr (want {want_intensity:.4}), \
         m = {magnitude:.4} (want {want_magnitude:.4})"
    );
    assert!(
        (intensity - want_intensity).abs() / want_intensity < 1e-15,
        "the intensity is not E·A·C_d/π"
    );
    assert!(
        (magnitude - want_magnitude).abs() < 1e-13,
        "the magnitude is not the irradiance ratio against the Sun"
    );
    assert!(
        (4.0..5.0).contains(&magnitude),
        "a square metre at 1000 km should be near fourth magnitude, not {magnitude:.2}"
    );
}

/// `∮ I dω` over the visible hemisphere by the midpoint rule, for a facet whose normal is **+Z**
/// and a Sun direction `sun`. Axisymmetric only when the Sun is on the pole, so the φ integral is
/// done too.
fn reflected_power(body: &PhotometricBoxWing<1>, sun: [f64; 3], theta_steps: usize) -> f64 {
    let phi_steps = 720_i32;
    let d_theta = FRAC_PI_2 / theta_steps as f64;
    let d_phi = TAU / f64::from(phi_steps);
    let mut total = 0.0;
    for i in 0..theta_steps {
        let theta = (i as f64 + 0.5) * d_theta;
        let (sin_t, cos_t) = (libm::sin(theta), libm::cos(theta));
        let mut ring = 0.0;
        for j in 0..phi_steps {
            let phi = (f64::from(j) + 0.5) * d_phi;
            let observer = [sin_t * libm::cos(phi), sin_t * libm::sin(phi), cos_t];
            let view = ViewGeometry::new(sun, observer, 1.0, 1.0).unwrap();
            ring += body.radiant_intensity(&view);
        }
        total += ring * d_phi * sin_t * d_theta;
    }
    total
}

/// The reflected power, integrated over the hemisphere, equals `E·A·(C_d + C_s)` at normal
/// incidence for every specular exponent. (G4)
#[test]
fn the_reflectance_model_reflects_exactly_what_its_coefficients_claim() {
    // The property that makes the specular exponent a *distribution* parameter and not a brightness
    // knob: however tightly the lobe is bunched, the power leaving the facet is the same. Integrate
    // the intensity over the hemisphere and it must come back as `E·A·cos θᵢ·(C_d + C_s)` — the
    // fraction of the intercepted power the coefficients say is not absorbed.
    //
    // At normal incidence the whole lobe is above the horizon and the identity is exact. The
    // quadrature is a midpoint rule, so the residual below is its own error, not the model's.
    let area = 2.5;
    for exponent in [0.0, 1.0, 8.0, 40.0] {
        let body = one_facet(area, 0.25, 0.35, exponent); // C_a = 0.25, C_s = 0.35, C_d = 0.40
        let power = reflected_power(&body, [0.0, 0.0, 1.0], 2000);
        let want = SOLAR_IRRADIANCE * area * (0.40 + 0.35);
        let error = (power - want).abs() / want;
        println!("k = {exponent:5.1}: ∮I dω = {power:.4} W, want {want:.4} W ({error:.2e})");
        assert!(
            error < 1e-5,
            "the model does not conserve energy at k = {exponent}: {error:.3e}"
        );
    }

    // At oblique incidence part of the lobe points below the horizon, and Phong has no mechanism to
    // put that power anywhere else: the facet then reflects *less* than its coefficients allow. That
    // is a known artefact of the lobe shape, stated rather than hidden — what must never happen is
    // reflecting more, which would be a light curve brighter than the sunlight falling on it.
    //
    // How much is lost is not monotone in `k`, which is worth measuring rather than guessing. At 75°
    // incidence the mirror direction sits 15° above the horizon, and the measured losses are 0.0% at
    // `k = 0`, **10.3%** at `k = 8` and **2.2%** at `k = 40`: a uniform lobe has nothing to clip, a
    // lobe of intermediate width spills most of its skirt below the horizon, and a lobe narrow
    // enough to fit inside those 15° mostly clears it again.
    let cos_in = libm::cos(75.0_f64.to_radians());
    let sun = [libm::sin(75.0_f64.to_radians()), 0.0, cos_in];
    for exponent in [0.0, 8.0, 40.0] {
        let body = one_facet(area, 0.25, 0.35, exponent);
        let power = reflected_power(&body, sun, 2000);
        let ceiling = SOLAR_IRRADIANCE * area * cos_in * (0.40 + 0.35);
        println!(
            "75° incidence, k = {exponent:5.1}: ∮I dω = {power:.4} W against a ceiling of {ceiling:.4} W \
             ({:.1} % of it)",
            100.0 * power / ceiling
        );
        // The tolerance is the quadrature's own, measured at 5e-7 above; the physics is `<=`.
        assert!(
            power <= ceiling * (1.0 + 1e-5),
            "the facet reflected more than it received at k = {exponent}"
        );
        // Asserting the loss is real where it should be, and absent where it should not, is what
        // keeps the paragraph above from being a comfortable story about a bound never reached.
        if exponent == 0.0 {
            assert!(
                power > ceiling * (1.0 - 1e-5),
                "a uniform lobe cannot clip, yet it lost {:.2} %",
                100.0 * (1.0 - power / ceiling)
            );
        } else {
            assert!(
                power < 0.99 * ceiling,
                "the clipped lobe lost nothing at k = {exponent}: {power:.4} of {ceiling:.4}"
            );
        }
    }
}

#[test]
fn a_mirror_glints_only_near_the_mirror_direction() {
    // What a light curve of a specular object is *made* of. With `C_s = 1` and no diffuse component
    // the facet is nearly dark until the observer crosses the mirror direction, and then flashes.
    // Two things are pinned: where the flash is, found by scanning rather than assumed, and its
    // shape, which must be `cosᵏψ` off the mirror direction with no other angular dependence — the
    // `cos θₒ` a Lambertian carries is absent here, and that difference is the whole reason a glint
    // is narrow.
    let exponent = 60.0;
    let body = one_facet(1.0, 0.0, 1.0, exponent);
    let incidence = 40.0_f64.to_radians();
    let sun = [libm::sin(incidence), 0.0, libm::cos(incidence)];
    // Mirror of ŝ about ẑ: the same zenith angle on the other side.
    let mirror_theta = incidence;

    let at = |theta: f64| {
        let observer = [libm::sin(theta), 0.0, libm::cos(theta)];
        body.radiant_intensity(&ViewGeometry::new(sun, observer, 1.0, 1.0).unwrap())
    };
    let peak = at(-mirror_theta);

    // Scan for the maximum without assuming where it is.
    let mut best = (f64::NEG_INFINITY, 0.0);
    for i in 0..=1800 {
        let theta = -FRAC_PI_2 + PI * f64::from(i) / 1800.0;
        let value = at(theta);
        if value > best.0 {
            best = (value, theta);
        }
    }
    println!(
        "k = {exponent} at 40° incidence: peak {peak:.4} W/sr, scanned maximum at {:.3}° \
         (mirror is {:.3}°)",
        best.1.to_degrees(),
        -mirror_theta.to_degrees()
    );
    assert!(
        (best.1 + mirror_theta).abs() < 2.0 * PI / 1800.0,
        "the glint is not at the mirror direction: {:.3}° vs {:.3}°",
        best.1.to_degrees(),
        -mirror_theta.to_degrees()
    );

    let mut worst: f64 = 0.0;
    for off_degrees in [2.0_f64, 5.0, 10.0, 20.0, 35.0] {
        let psi = off_degrees.to_radians();
        let got = at(-mirror_theta + psi) / peak;
        let want = libm::pow(libm::cos(psi), exponent);
        println!("  {off_degrees:4.1}° off the mirror: {got:.6e} of the peak, want {want:.6e}");
        worst = worst.max((got - want).abs() / want);
    }
    assert!(worst < 1e-12, "the lobe is not cosᵏψ: {worst:.3e}");

    // And it is on the far side of the normal from the Sun: at the Sun's own zenith angle the
    // observer is 80° off the mirror direction, which `cos⁶⁰` puts 45 orders down.
    let wrong_side = at(mirror_theta);
    println!("  the Sun's own side: {wrong_side:.4e} W/sr");
    assert!(
        wrong_side < 1e-30 * peak,
        "the reflection is not on the far side of the normal: {wrong_side:.3e}"
    );
}

#[test]
fn nothing_reflects_from_umbra_or_from_a_face_that_is_turned_away() {
    let plate = one_facet(1.0, 0.0, 0.0, 0.0);
    let lit = ViewGeometry::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0e6, 1.0).unwrap();
    let umbra = ViewGeometry::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0e6, 0.0).unwrap();
    let behind = ViewGeometry::new([0.0, 0.0, 1.0], [0.0, 0.0, -1.0], 1.0e6, 1.0).unwrap();
    let unlit = ViewGeometry::new([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 1.0e6, 1.0).unwrap();

    println!(
        "lit {:.4e}, umbra {:.4e}, observer behind {:.4e}, Sun behind {:.4e} W/sr",
        plate.radiant_intensity(&lit),
        plate.radiant_intensity(&umbra),
        plate.radiant_intensity(&behind),
        plate.radiant_intensity(&unlit)
    );
    assert!(plate.radiant_intensity(&lit) > 0.0);
    for dark in [umbra, behind, unlit] {
        assert!(plate.radiant_intensity(&dark) == 0.0, "something reflected");
        assert!(
            plate.apparent_magnitude(&dark).is_infinite(),
            "an object that reflects nothing has to be infinitely faint"
        );
    }

    // A penumbra is a fraction, not a boolean: half the disc is half the light, which is 0.753 mag.
    let half = ViewGeometry::new([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1.0e6, 0.5).unwrap();
    let dimming = plate.apparent_magnitude(&half) - plate.apparent_magnitude(&lit);
    println!("half the solar disc costs {dimming:.6} mag");
    assert!(
        (dimming - 2.5 * libm::log10(2.0)).abs() < 1e-12,
        "penumbra is not a linear scaling of the illumination: {dimming}"
    );
}

/// A cube spun about a body axis, Sun and observer equatorial: the curve repeats every quarter
/// turn to rounding. (G4)
#[test]
fn a_box_spinning_about_a_body_axis_has_a_four_fold_light_curve() {
    // The end-to-end geometry test, and the one structural claim a box's light curve makes: spin a
    // cube about ẑ with the Sun and the observer both in the equatorial plane, and the four side
    // faces take the same turn one after another. The curve must repeat every quarter revolution —
    // not approximately, but to rounding, since it is the same four facets in the same geometry.
    let cube = PhotometricBoxWing {
        shape: BoxWing::<6>::box6(1.5, 1.5, 1.5, 800.0, 2.2, 0.0, 0.6, 0.25),
        specular_exponent: [12.0; 6],
    };
    let sun_inertial = [1.0, 0.0, 0.0];
    let observer_inertial = {
        // 30° of phase angle, in the equatorial plane so the ±Z faces never contribute.
        let a = 30.0_f64.to_radians();
        [libm::cos(a), libm::sin(a), 0.0]
    };

    let magnitude = |spin: f64| {
        let attitude = exp_so3([0.0, 0.0, spin]);
        // `exp_so3` maps body→inertial, so its transpose is the inertial→body matrix the geometry
        // wants; passing the matrix itself would spin the body the wrong way.
        let inertial_to_body = [
            [attitude[0][0], attitude[1][0], attitude[2][0]],
            [attitude[0][1], attitude[1][1], attitude[2][1]],
            [attitude[0][2], attitude[1][2], attitude[2][2]],
        ];
        let view = ViewGeometry::from_inertial(
            sun_inertial,
            observer_inertial,
            1.0e6,
            1.0,
            &inertial_to_body,
        )
        .unwrap();
        cube.apparent_magnitude(&view)
    };

    let samples = 240;
    let mut worst_quarter: f64 = 0.0;
    let mut brightest = (f64::INFINITY, 0.0);
    let mut faintest = (f64::NEG_INFINITY, 0.0);
    for i in 0..samples {
        let spin = TAU * f64::from(i) / f64::from(samples);
        let here = magnitude(spin);
        worst_quarter = worst_quarter.max((here - magnitude(spin + FRAC_PI_2)).abs());
        if here < brightest.0 {
            brightest = (here, spin);
        }
        if here > faintest.0 {
            faintest = (here, spin);
        }
    }
    println!(
        "spinning cube: quarter-turn repeat within {worst_quarter:.3e} mag, \
         brightest {:.3} at {:.1}°, faintest {:.3} at {:.1}°",
        brightest.0,
        brightest.1.to_degrees(),
        faintest.0,
        faintest.1.to_degrees()
    );
    assert!(
        worst_quarter < 1e-12,
        "the light curve is not four-fold symmetric: {worst_quarter:.3e} mag"
    );
    // And it is a light *curve* — the amplitude has to be real, or the test above would pass on a
    // constant.
    assert!(
        faintest.0 - brightest.0 > 0.5,
        "the curve is flat: {:.3} mag of amplitude",
        faintest.0 - brightest.0
    );
}

#[test]
fn the_synodic_period_is_the_sidereal_one_seen_through_a_moving_geometry() {
    let sidereal = SiderealPeriod::new(3600.0).unwrap();

    // A geometry that does not move gives the two periods the same value.
    let still = sidereal.to_synodic(0.0).unwrap();
    assert!(
        (still.seconds() - sidereal.seconds()).abs() < 1e-9,
        "a static geometry changed the period: {}",
        still.seconds()
    );

    // A bisector creeping forward at a thousandth of the spin rate stretches the observed period by
    // the same thousandth, exactly as a solar day exceeds a sidereal one.
    let spin_rate = TAU / 3600.0;
    let creeping = sidereal.to_synodic(1e-3 * spin_rate).unwrap();
    let want = 3600.0 / (1.0 - 1e-3);
    println!(
        "sidereal 3600 s with a bisector at 0.1% of the spin: synodic {:.6} s (want {want:.6})",
        creeping.seconds()
    );
    assert!(
        (creeping.seconds() - want).abs() / want < 1e-12,
        "the synodic period is not 2π/(ω_sid − ω_pab)"
    );
    // And the conversion inverts.
    let back = creeping.to_sidereal(1e-3 * spin_rate).unwrap();
    assert!(
        (back.seconds() - 3600.0).abs() < 1e-9,
        "the round trip lost the period: {}",
        back.seconds()
    );

    // The one case that has no answer: a body spinning at exactly the bisector rate shows the same
    // aspect forever, and its light curve is flat rather than long-period.
    assert!(
        sidereal.to_synodic(spin_rate).is_none(),
        "a body locked to the geometry was given a finite light-curve period"
    );

    assert!(SiderealPeriod::new(0.0).is_none());
    assert!(SiderealPeriod::new(f64::NAN).is_none());
    assert!(SynodicPeriod::new(-1.0).is_none());
}

#[test]
fn a_geometry_that_cannot_be_observed_is_refused() {
    let ok = [0.0, 0.0, 1.0];
    assert!(
        ViewGeometry::new([0.0; 3], ok, 1.0e6, 1.0).is_none(),
        "zero Sun direction"
    );
    assert!(
        ViewGeometry::new(ok, [0.0; 3], 1.0e6, 1.0).is_none(),
        "zero observer direction"
    );
    assert!(ViewGeometry::new(ok, ok, 0.0, 1.0).is_none(), "zero range");
    assert!(
        ViewGeometry::new(ok, ok, -1.0, 1.0).is_none(),
        "negative range"
    );
    assert!(
        ViewGeometry::new(ok, ok, 1.0e6, 1.5).is_none(),
        "over-illuminated"
    );
    assert!(
        ViewGeometry::new(ok, ok, 1.0e6, -0.1).is_none(),
        "negative illumination"
    );
    assert!(
        ViewGeometry::new(ok, ok, f64::NAN, 1.0).is_none(),
        "range is not a number"
    );
    assert!(
        ViewGeometry::new(ok, ok, f64::INFINITY, 1.0).is_none(),
        "range is not finite"
    );
    assert!(
        ViewGeometry::new([f64::NAN, 0.0, 0.0], ok, 1.0e6, 1.0).is_none(),
        "the Sun direction is not a number"
    );

    // The directions are normalised, not merely accepted: a Sun vector ten units long is the same
    // geometry, so it has to give the same brightness.
    let plate = one_facet(1.0, 0.0, 0.0, 0.0);
    let unit = ViewGeometry::new([0.3, 0.4, 0.866_025_403_784_438_6], ok, 1.0e6, 1.0).unwrap();
    let long = ViewGeometry::new([3.0, 4.0, 8.660_254_037_844_387], ok, 1.0e6, 1.0).unwrap();
    let difference = (plate.radiant_intensity(&unit) - plate.radiant_intensity(&long)).abs();
    println!("a ten-fold Sun vector changes the intensity by {difference:.3e} W/sr");
    assert!(difference < 1e-13, "the direction was not normalised");
    assert!(
        (unit.phase_angle() - libm::acos(0.866_025_403_784_438_6)).abs() < 1e-12,
        "the phase angle is not the Sun–object–observer angle"
    );
}

#[test]
fn the_geometry_reports_back_the_directions_it_normalised_and_the_numbers_it_kept() {
    // Every one of these four had been written and never read: the tests above reach the fields
    // through `radiant_intensity`, which uses them internally, so an accessor could have returned a
    // constant with the whole suite still green. The vectors are Pythagorean quadruples so that the
    // norm is an exact integer and the expected components are one rounding away from the input.
    let view = ViewGeometry::new([2.0, 3.0, 6.0], [1.0, 4.0, 8.0], 1.234_5e6, 0.375).unwrap();

    for (got, want, axis) in [
        (view.sun_body(), [2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0], "Sun"),
        (
            view.observer_body(),
            [1.0 / 9.0, 4.0 / 9.0, 8.0 / 9.0],
            "observer",
        ),
    ] {
        for k in 0..3 {
            assert!(
                (got[k] - want[k]).abs() <= f64::EPSILON,
                "{axis} direction component {k}: {} against {}",
                got[k],
                want[k]
            );
        }
    }
    // Neither 1.234 5e6 nor 0.375 is 0, 1 or −1 — the three constants a mutated accessor reaches for.
    assert!((view.range() - 1.234_5e6).abs() < 1e-9, "range");
    assert!(
        (view.sunlit_fraction() - 0.375).abs() < 1e-15,
        "sunlit fraction"
    );
}

/// One tilted double-sided facet, pure Lambertian: `absorption = reflection = 0` leaves
/// `diffuse = 1`, so the intensity is `E·A·cos θᵢ·cos θₒ/π` and nothing else.
fn tilted_double_sided(area: f64) -> PhotometricBoxWing<1> {
    PhotometricBoxWing {
        shape: BoxWing {
            facets: [Facet::fixed(
                [2.0, 3.0, 6.0],
                area,
                true,
                2.2,
                0.0,
                0.0,
                0.0,
            )],
            mass: 500.0,
            drag_factor: 1.0,
            radiation_factor: 1.0,
        },
        specular_exponent: [0.0],
    }
}

#[test]
fn a_double_sided_facet_turns_its_normal_round_only_when_the_sun_is_behind_it() {
    // Not one fixture above declares a double-sided facet, so the whole flip was dead code as far as
    // the suite was concerned. Both halves of the condition need their own geometry: sunlit-in-front
    // says the flip must *not* happen, back-lit says it must, and no single view can say both.
    //
    // The normal is (2,3,6)/7 rather than a body axis on purpose. Negating a vector two of whose
    // components are ±0.0 is indistinguishable from not negating it, so an axis-aligned facet cannot
    // see two thirds of the negation at all.
    let area = 3.0;
    let plate = tilted_double_sided(area);
    let normal = [2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0];
    let behind = [-normal[0], -normal[1], -normal[2]];
    let lambertian = SOLAR_IRRADIANCE * area / PI;

    // Sun and observer in front: cos θᵢ = cos θₒ = 1, and the facet must be taken as it stands.
    let front = ViewGeometry::new(normal, normal, 1.0e6, 1.0).unwrap();
    let front_intensity = plate.radiant_intensity(&front);
    println!("front-lit: {front_intensity:.6e} W/sr against {lambertian:.6e}");
    assert!(
        (front_intensity - lambertian).abs() < 1e-9 * lambertian,
        "a facet lit on its own side must not be flipped"
    );

    // Sun and observer both behind: the flipped normal is the Sun direction, so cos θᵢ = cos θₒ = 1
    // again and the intensity is the same number — which is exactly what "double-sided" means.
    let back = ViewGeometry::new(behind, behind, 1.0e6, 1.0).unwrap();
    let back_intensity = plate.radiant_intensity(&back);
    println!("back-lit: {back_intensity:.6e} W/sr against {lambertian:.6e}");
    assert!(
        (back_intensity - lambertian).abs() < 1e-9 * lambertian,
        "a double-sided facet lit from behind reflects as much as one lit in front"
    );

    // Each component of the negation carries a different amount of the answer, which is what makes
    // the three of them separable: viewed along −n̂, a normal with one sign left unflipped gives
    // cos θₒ = 41/49, 31/49 and −23/49 for components x, y and z — the last of which is negative, so
    // that facet drops out of the sum entirely.
    for (component, cosine) in [
        (0, 41.0_f64 / 49.0),
        (1, 31.0_f64 / 49.0),
        (2, -23.0_f64 / 49.0),
    ] {
        let wrong = lambertian * cosine.max(0.0);
        assert!(
            (back_intensity - wrong).abs() > 1e-3 * lambertian,
            "leaving component {component} of the normal unflipped is indistinguishable here"
        );
    }

    // A single-sided facet in the same back-lit geometry sees nothing at all — the flip is the only
    // thing that made the difference above.
    let opaque = one_facet(area, 0.0, 0.0, 0.0);
    let edge_on = ViewGeometry::new([0.0, 0.0, -1.0], [0.0, 0.0, -1.0], 1.0e6, 1.0).unwrap();
    assert!(
        opaque.radiant_intensity(&edge_on) == 0.0,
        "a single-sided facet lit from behind reflects nothing"
    );
}

#[test]
fn a_spin_the_geometry_exactly_cancels_implies_no_sidereal_period() {
    // `to_sidereal` adds the bisector rate to the observed one and the sum is the inertial spin
    // rate. Two ways for that sum to leave the domain, and the guard has to reject both, because
    // what it hands back is a `SiderealPeriod` — a type whose own constructor refuses anything but
    // a positive finite number. A guard that admits either produces a value `SiderealPeriod::new`
    // would not have made.
    let synodic = SynodicPeriod::new(4.0 * 3600.0).unwrap();

    // Exact cancellation: the body shows the same face to the geometry forever, so the light curve
    // has no period and there is no inertial rotation behind it. `TAU/P` is computed the same way on
    // both sides, so the sum is exactly 0.0 rather than nearly so — the one input that separates
    // `> 0.0` from `>= 0.0`.
    let cancelling = -(TAU / synodic.seconds());
    assert!(
        synodic.to_sidereal(cancelling).is_none(),
        "a spin the geometry exactly cancels has no sidereal period"
    );

    // An unbounded bisector rate is not finite, and `TAU/∞` is a zero-second period — positive, so
    // the sign test alone lets it through.
    assert!(
        synodic.to_sidereal(f64::INFINITY).is_none(),
        "an infinite bisector rate has no sidereal period"
    );
    assert!(
        synodic.to_sidereal(f64::NAN).is_none(),
        "a bisector rate that is not a number has no sidereal period"
    );

    // Everything the guard does admit is something `SiderealPeriod::new` would have accepted.
    for step in 1..40 {
        let rate = cancelling * (1.0 - f64::from(step) / 20.0);
        if let Some(sidereal) = synodic.to_sidereal(rate) {
            assert!(
                SiderealPeriod::new(sidereal.seconds()).is_some(),
                "to_sidereal produced {} s, which its own constructor refuses",
                sidereal.seconds()
            );
        }
    }
}
