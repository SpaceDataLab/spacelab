//! Measurement-model partials from AD, against the closed forms.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::needless_range_loop)]
#![allow(clippy::many_single_char_names)]

use spacelab_autodiff::jacobian;
use spacelab_estimation::measurement::{
    Station, Topocentric, azimuth, declination, elevation, range, range_rate, right_ascension,
};

#[test]
fn range_partials_match_the_closed_form() {
    // ∂ρ/∂r = Δr/ρ, ∂ρ/∂v = 0.
    let station = Station::fixed([100.0, -200.0, 50.0]);
    let sat = [7000.0, 1000.0, 500.0, 1.0, 7.0, 0.5];
    let (val, jac) = jacobian::<6, 1, _>(|s| [range(s, &station)], &sat);

    let dr = [sat[0] - 100.0, sat[1] + 200.0, sat[2] - 50.0];
    let rho = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
    assert!((val[0] - rho).abs() < 1e-9);
    for i in 0..3 {
        assert!((jac[i][0] - dr[i] / rho).abs() < 1e-12, "∂ρ/∂r[{i}]");
        assert!(jac[i + 3][0].abs() < 1e-15, "∂ρ/∂v[{i}] must be zero");
    }
}

#[test]
fn range_rate_partials_match_the_closed_form() {
    // ρ̇ = (Δr·v)/ρ. ∂ρ̇/∂v = Δr/ρ. ∂ρ̇/∂r = v/ρ − ρ̇·Δr/ρ².
    let station = Station::fixed([100.0, -200.0, 50.0]);
    let sat = [7000.0, 1000.0, 500.0, 1.0, 7.0, 0.5];
    let (val, jac) = jacobian::<6, 1, _>(|s| [range_rate(s, &station)], &sat);

    let dr = [sat[0] - 100.0, sat[1] + 200.0, sat[2] - 50.0];
    let v = [sat[3], sat[4], sat[5]];
    let rho = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
    let rdot = (dr[0] * v[0] + dr[1] * v[1] + dr[2] * v[2]) / rho;
    assert!((val[0] - rdot).abs() < 1e-9);
    for i in 0..3 {
        // velocity partials
        assert!((jac[i + 3][0] - dr[i] / rho).abs() < 1e-12, "∂ρ̇/∂v[{i}]");
        // position partials
        let want = v[i] / rho - rdot * dr[i] / (rho * rho);
        assert!(
            (jac[i][0] - want).abs() < 1e-9,
            "∂ρ̇/∂r[{i}]: {} vs {want}",
            jac[i][0]
        );
    }
}

#[test]
fn right_ascension_partials_match_the_closed_form() {
    // α = atan2(Δy, Δx). ∂α/∂x = −Δy/ρ_xy², ∂α/∂y = Δx/ρ_xy², and nothing else.
    let station = Station::fixed([100.0, -200.0, 50.0]);
    let sat = [7000.0, 1000.0, 500.0, 1.0, 7.0, 0.5];
    let (val, jac) = jacobian::<6, 1, _>(|s| [right_ascension(s, &station)], &sat);

    let (dx, dy) = (sat[0] - 100.0, sat[1] + 200.0);
    let rho_xy2 = dx * dx + dy * dy;
    assert!((val[0] - dy.atan2(dx)).abs() < 1e-12);
    assert!((jac[0][0] - (-dy / rho_xy2)).abs() < 1e-15, "∂α/∂x");
    assert!((jac[1][0] - (dx / rho_xy2)).abs() < 1e-15, "∂α/∂y");
    assert!(jac[2][0].abs() < 1e-18, "∂α/∂z must be zero");
    for i in 3..6 {
        assert!(jac[i][0].abs() < 1e-18, "∂α/∂v must be zero");
    }
}

#[test]
fn declination_partials_match_the_closed_form() {
    // δ = atan2(Δz, ρ_xy). ∂δ/∂x = −ΔxΔz/(ρ²ρ_xy), ∂δ/∂y = −ΔyΔz/(ρ²ρ_xy), ∂δ/∂z = ρ_xy/ρ².
    let station = Station::fixed([100.0, -200.0, 50.0]);
    let sat = [7000.0, 1000.0, 500.0, 1.0, 7.0, 0.5];
    let (val, jac) = jacobian::<6, 1, _>(|s| [declination(s, &station)], &sat);

    let (dx, dy, dz) = (sat[0] - 100.0, sat[1] + 200.0, sat[2] - 50.0);
    let rho_xy = (dx * dx + dy * dy).sqrt();
    let rho2 = dx * dx + dy * dy + dz * dz;
    assert!((val[0] - dz.atan2(rho_xy)).abs() < 1e-12);
    assert!(
        (jac[0][0] - (-dx * dz / (rho2 * rho_xy))).abs() < 1e-15,
        "∂δ/∂x"
    );
    assert!(
        (jac[1][0] - (-dy * dz / (rho2 * rho_xy))).abs() < 1e-15,
        "∂δ/∂y"
    );
    assert!((jac[2][0] - (rho_xy / rho2)).abs() < 1e-15, "∂δ/∂z");
    for i in 3..6 {
        assert!(jac[i][0].abs() < 1e-18, "∂δ/∂v must be zero");
    }
}

#[test]
fn azimuth_and_elevation_match_the_horizon_geometry() {
    // Station at the equator on the prime meridian: up = x̂, east = ŷ, north = ẑ.
    let re = 6_378_137.0;
    let topo = Topocentric::from_geodetic([re, 0.0, 0.0], 0.0, 0.0);

    let el = |sat: [f64; 6]| elevation::<f64>(&sat, &topo).to_degrees();
    let az = |sat: [f64; 6]| azimuth::<f64>(&sat, &topo).to_degrees();

    // straight up → 90° elevation
    assert!(
        (el([re + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0]) - 90.0).abs() < 1e-9,
        "overhead elevation"
    );
    // due east on the horizon → elevation 0°, azimuth 90°
    assert!(
        el([re, 500e3, 0.0, 0.0, 0.0, 0.0]).abs() < 1e-9,
        "east elevation"
    );
    assert!(
        (az([re, 500e3, 0.0, 0.0, 0.0, 0.0]) - 90.0).abs() < 1e-9,
        "east azimuth"
    );
    // due north on the horizon → azimuth 0°
    assert!(
        az([re, 0.0, 500e3, 0.0, 0.0, 0.0]).abs() < 1e-9,
        "north azimuth"
    );
    // equal up/east/north → azimuth 45° (north-east), elevation atan2(1, √2) = 35.264°
    let ne = [re + 100e3, 100e3, 100e3, 0.0, 0.0, 0.0];
    assert!(
        (az(ne) - 45.0).abs() < 1e-9,
        "north-east azimuth: {}",
        az(ne)
    );
    assert!(
        (el(ne) - 35.264_389_682_754_654).abs() < 1e-9,
        "north-east elevation: {}",
        el(ne)
    );
}

#[test]
fn azimuth_and_elevation_partials_match_the_closed_form() {
    // ∂el/∂r = [ρ_h û − (u/ρ_h)(e ê + n n̂)] / ρ², ∂az/∂r = (n ê − e n̂) / ρ_h².
    let topo = Topocentric::from_geodetic([5_000e3, 2_000e3, 3_000e3], 0.6, 1.2);
    let sat = [6_000e3, 2_500e3, 3_800e3, 1.0, 7.0, 0.5];

    let dr = [
        sat[0] - topo.position[0],
        sat[1] - topo.position[1],
        sat[2] - topo.position[2],
    ];
    let proj = |axis: &[f64; 3]| dr[0] * axis[0] + dr[1] * axis[1] + dr[2] * axis[2];
    let (e, n, u) = (proj(&topo.east), proj(&topo.north), proj(&topo.up));
    let rho_h = (e * e + n * n).sqrt();
    let rho2 = e * e + n * n + u * u;

    let (_, jac_el) = jacobian::<6, 1, _>(|s| [elevation(s, &topo)], &sat);
    let (_, jac_az) = jacobian::<6, 1, _>(|s| [azimuth(s, &topo)], &sat);
    for k in 0..3 {
        let grad_el =
            (rho_h * topo.up[k] - (u / rho_h) * (e * topo.east[k] + n * topo.north[k])) / rho2;
        let grad_az = (n * topo.east[k] - e * topo.north[k]) / (rho_h * rho_h);
        assert!(
            (jac_el[k][0] - grad_el).abs() < 1e-15,
            "∂el/∂r[{k}]: {} vs {grad_el}",
            jac_el[k][0]
        );
        assert!(
            (jac_az[k][0] - grad_az).abs() < 1e-15,
            "∂az/∂r[{k}]: {} vs {grad_az}",
            jac_az[k][0]
        );
    }
    for k in 3..6 {
        assert!(
            jac_el[k][0].abs() < 1e-18 && jac_az[k][0].abs() < 1e-18,
            "angle ∂/∂v must be zero"
        );
    }
}

/// Range rate is a *relative* rate: it depends on the two velocities only through their difference.
///
/// Every other fixture here uses [`Station::fixed`], whose velocity is zero, where subtracting the
/// station velocity and adding it are the same operation — so the module's own doc, "in an inertial
/// frame the station moves and the caller supplies its velocity", had nothing holding it up. Two
/// statements that need a moving station: boosting both bodies by a common velocity leaves the rate
/// alone, and a station that keeps pace with the satellite reads exactly zero.
#[test]
fn range_rate_sees_only_the_relative_velocity() {
    let sat = [7_000e3, 1_200e3, -800e3, -1.1e3, 7.2e3, 0.4e3];
    let position = [4_200e3, 900e3, -600e3];
    let station_velocity = [-65.0, 306.0, 0.0]; // an ITRF site seen from an inertial frame

    let moving = Station {
        position,
        velocity: station_velocity,
    };
    let rate = range_rate(&sat, &moving);

    // Galilean invariance: add the same velocity to both bodies.
    let boost = [123.0, -456.0, 789.0];
    let mut boosted_sat = sat;
    for k in 0..3 {
        boosted_sat[3 + k] += boost[k];
    }
    let boosted = Station {
        position,
        velocity: [
            station_velocity[0] + boost[0],
            station_velocity[1] + boost[1],
            station_velocity[2] + boost[2],
        ],
    };
    assert!(
        (range_rate(&boosted_sat, &boosted) - rate).abs() < 1e-9,
        "a common boost moved the range rate: {} vs {rate}",
        range_rate(&boosted_sat, &boosted)
    );

    // Keeping pace: the separation is momentarily frozen, so the rate is zero exactly.
    let keeping_pace = Station {
        position,
        velocity: [sat[3], sat[4], sat[5]],
    };
    let paced = range_rate(&sat, &keeping_pace);
    assert!(
        paced.abs() < 1e-12,
        "a station moving with the satellite must read zero, not {paced}"
    );
    assert!(
        rate.abs() > 100.0,
        "the fixture has to have a rate to lose: {rate}"
    );
}

/// The horizon triad is the orthonormal frame carried by the geodetic normal.
///
/// `up` is the normal to the ellipsoid at (φ, λ), and `north` and `east` are the directions in which
/// that normal turns as φ and λ increase — `∂up/∂φ` and `∂up/∂λ / cos φ`. That derivative relation is
/// what makes the triad the *horizon* frame rather than three unit vectors that happen to be
/// perpendicular, and it derives east and north from up alone, so it is not a second copy of the
/// construction. Checked here against a central difference of `up` itself, alongside orthonormality
/// and the right-handed order east × north = up.
#[test]
fn the_horizon_triad_is_the_frame_the_geodetic_normal_carries() {
    let h = 1e-6;
    for (lat, lon) in [
        (0.0, 0.0),
        (0.6, 1.2),
        (-0.9, -2.7),
        (1.4, 3.0),
        (-1.2, 0.35),
    ] {
        let at = |p: f64, l: f64| Topocentric::from_geodetic([0.0; 3], p, l).up;
        let t = Topocentric::from_geodetic([0.0; 3], lat, lon);
        let (e, n, u) = (t.east, t.north, t.up);

        // up is the normal itself.
        let (sp, cp) = (lat.sin(), lat.cos());
        let (sl, cl) = (lon.sin(), lon.cos());
        for (k, want) in [cp * cl, cp * sl, sp].into_iter().enumerate() {
            assert!(
                (u[k] - want).abs() < 1e-15,
                "φ={lat} λ={lon}: up[{k}] = {}, want {want}",
                u[k]
            );
        }

        // Orthonormal, and right-handed in the order east, north, up.
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for (name, v) in [("east", e), ("north", n), ("up", u)] {
            assert!(
                (dot(v, v) - 1.0).abs() < 1e-15,
                "φ={lat} λ={lon}: |{name}|² = {}",
                dot(v, v)
            );
        }
        for (name, d) in [("e·n", dot(e, n)), ("e·u", dot(e, u)), ("n·u", dot(n, u))] {
            assert!(d.abs() < 1e-15, "φ={lat} λ={lon}: {name} = {d}");
        }
        let cross = [
            e[1] * n[2] - e[2] * n[1],
            e[2] * n[0] - e[0] * n[2],
            e[0] * n[1] - e[1] * n[0],
        ];
        for k in 0..3 {
            assert!(
                (cross[k] - u[k]).abs() < 1e-15,
                "φ={lat} λ={lon}: (e × n)[{k}] = {}, up says {}",
                cross[k],
                u[k]
            );
        }

        // north = ∂up/∂φ and east = (∂up/∂λ) / cos φ, by central difference.
        let (up_lat_p, up_lat_m) = (at(lat + h, lon), at(lat - h, lon));
        let (up_lon_p, up_lon_m) = (at(lat, lon + h), at(lat, lon - h));
        for k in 0..3 {
            let d_lat = (up_lat_p[k] - up_lat_m[k]) / (2.0 * h);
            let d_lon = (up_lon_p[k] - up_lon_m[k]) / (2.0 * h) / cp;
            assert!(
                (d_lat - n[k]).abs() < 1e-9,
                "φ={lat} λ={lon}: ∂up/∂φ[{k}] = {d_lat}, north says {}",
                n[k]
            );
            assert!(
                (d_lon - e[k]).abs() < 1e-9,
                "φ={lat} λ={lon}: ∂up/∂λ[{k}]/cos φ = {d_lon}, east says {}",
                e[k]
            );
        }
    }
}
