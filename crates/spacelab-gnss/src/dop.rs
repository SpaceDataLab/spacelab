//! Dilution of precision — how a set of visible-satellite directions amplifies range error into
//! position/time error. This is the metric GNSS engineers use to score constellation geometry, and it
//! composes directly with the broadcast orbits in this crate (propagate → positions → [`dop`]).
//!
//! With a receiver at `p` and satellites at `sᵢ`, each contributes a unit line-of-sight `êᵢ` and a
//! design-matrix row `[êᵢ·E, êᵢ·N, êᵢ·U, 1]` in the local East/North/Up frame (the trailing `1` is the
//! receiver-clock unknown). The cofactor matrix `Q = (GᵀG)⁻¹` gives, from its diagonal, the standard
//! DOPs: `HDOP=√(Q_EE+Q_NN)`, `VDOP=√Q_UU`, `PDOP=√(Q_EE+Q_NN+Q_UU)`, `TDOP=√Q_TT`, `GDOP=√tr Q`.
//!
//! Gate: `derivations/gnss_dop.mac` inverts `GᵀG` exactly on a sky of rational unit vectors and on
//! the regular tetrahedron, and derives the zenith-plus-ring closed form for rings of 3, 4, 6, 8
//! and 12; `derivations/gnss_dop.g` re-inverts both over ℚ in GAP. The module tests
//! `dop_is_the_root_of_the_cofactor_traces_on_rational_geometry` (G1) and
//! `the_regular_tetrahedron_has_its_closed_form_dops` (G3) hold the shipped `dop` to them.

// The single letters (e, n, u, p, s) are the East/North/Up and point notation.
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)]

fn norm(v: [f64; 3]) -> f64 {
    libm::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n < 1e-300 {
        v
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Invert a 4×4 matrix by Gauss–Jordan elimination with partial pivoting; `None` if (near-)singular.
fn invert4(m: [[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let mut a = m;
    let mut inv = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for col in 0..4 {
        let mut pivot = col;
        for r in (col + 1)..4 {
            if libm::fabs(a[r][col]) > libm::fabs(a[pivot][col]) {
                pivot = r;
            }
        }
        if libm::fabs(a[pivot][col]) < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let p = a[col][col];
        for j in 0..4 {
            a[col][j] /= p;
            inv[col][j] /= p;
        }
        for r in 0..4 {
            if r != col {
                let f = a[r][col];
                for j in 0..4 {
                    a[r][j] -= f * a[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    Some(inv)
}

/// The dilution-of-precision factors for a receiver–satellite geometry. Dimensionless multipliers of
/// the range-error standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dop {
    /// Geometric DOP (position + clock).
    pub gdop: f64,
    /// Position DOP.
    pub pdop: f64,
    /// Horizontal DOP.
    pub hdop: f64,
    /// Vertical DOP.
    pub vdop: f64,
    /// Time DOP.
    pub tdop: f64,
}

/// Dilution of precision for a `receiver` (ECEF, m) seeing `satellites` (ECEF, m). Local vertical is
/// the geocentric radial. Returns `None` with fewer than four satellites or a rank-deficient geometry.
#[must_use]
pub fn dop(receiver: [f64; 3], satellites: &[[f64; 3]]) -> Option<Dop> {
    if satellites.len() < 4 {
        return None;
    }
    // Local East/North/Up basis (geocentric up), with a pole fallback for a degenerate East.
    let up = normalize(receiver);
    let mut east = cross([0.0, 0.0, 1.0], up);
    if norm(east) < 1e-9 {
        east = [1.0, 0.0, 0.0];
    }
    let east = normalize(east);
    let north = cross(up, east);

    // Accumulate the normal matrix N = GᵀG over the design rows [e, n, u, 1].
    let mut n = [[0.0f64; 4]; 4];
    for s in satellites {
        let los = normalize([s[0] - receiver[0], s[1] - receiver[1], s[2] - receiver[2]]);
        let row = [dot(los, east), dot(los, north), dot(los, up), 1.0];
        for i in 0..4 {
            for j in 0..4 {
                n[i][j] += row[i] * row[j];
            }
        }
    }

    // `N = GᵀG` is positive semi-definite by construction, so an `invert4` that clears its pivot
    // test leaves `Q = N⁻¹` positive definite and every `Q_ii = ‖N^{-1/2}eᵢ‖²` strictly positive —
    // bounded below by `1/tr N = 1/(2m)`, since each design row has `row·row = 2`. A diagonal-sign
    // re-check would therefore never fire (and could not catch NaN, which fails every `<`).
    let q = invert4(n)?;
    Some(Dop {
        gdop: libm::sqrt(q[0][0] + q[1][1] + q[2][2] + q[3][3]),
        pdop: libm::sqrt(q[0][0] + q[1][1] + q[2][2]),
        hdop: libm::sqrt(q[0][0] + q[1][1]),
        vdop: libm::sqrt(q[2][2]),
        tdop: libm::sqrt(q[3][3]),
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    const RE: f64 = 6_378_137.0;
    const R_GPS: f64 = 26_560_000.0;

    /// A satellite at elevation `el` and azimuth `az` (radians) seen from a receiver on the +x axis,
    /// in ECEF — East = +y, North = +z, Up = +x there.
    fn sat(el: f64, az: f64) -> [f64; 3] {
        let (ce, se) = (libm::cos(el), libm::sin(el));
        // Local ENU direction, then map E→+y, N→+z, U→+x.
        let (e, n, u) = (ce * libm::sin(az), ce * libm::cos(az), se);
        [RE + R_GPS * u, R_GPS * e, R_GPS * n]
    }

    #[test]
    fn invert4_reproduces_the_identity() {
        let m = [
            [4.0, 1.0, 0.0, 0.5],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.0, 0.3],
            [0.5, 0.0, 0.3, 1.5],
        ];
        let inv = invert4(m).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += m[i][k] * inv[k][j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((s - expect).abs() < 1e-10, "({i},{j}) = {s}");
            }
        }
    }

    #[test]
    fn fewer_than_four_satellites_has_no_dop() {
        let rx = [RE, 0.0, 0.0];
        assert!(dop(rx, &[sat(1.5, 0.0), sat(0.3, 0.0), sat(0.3, 2.0)]).is_none());
    }

    #[test]
    fn a_well_spread_sky_gives_a_modest_dop() {
        let rx = [RE, 0.0, 0.0];
        let tau = core::f64::consts::TAU;
        // One near zenith plus a ring of four at 25° elevation, evenly in azimuth.
        let mut sats = vec![sat(1.4, 0.0)];
        for k in 0..4 {
            sats.push(sat(25.0_f64.to_radians(), tau * f64::from(k) / 4.0));
        }
        let d = dop(rx, &sats).expect("well-conditioned geometry");
        assert!(d.pdop > 1.0 && d.pdop < 6.0, "pdop {}", d.pdop);
        assert!(d.gdop >= d.pdop, "gdop {} < pdop {}", d.gdop, d.pdop);
        // GDOP² = PDOP² + TDOP², so the three are consistent. (These two are identities in the
        // same `q` diagonal and hold for *any* `q` — they pin the packing, nothing more; the
        // numbers themselves are pinned by the closed-form test below.)
        assert!((d.gdop.powi(2) - d.pdop.powi(2) - d.tdop.powi(2)).abs() < 1e-9);
        assert!((d.pdop.powi(2) - d.hdop.powi(2) - d.vdop.powi(2)).abs() < 1e-9);
    }

    /// Every DOP against the closed form for a zenith-plus-ring sky.
    ///
    /// The band above admits a great many wrong numbers, and the two consistency assertions are
    /// algebraic identities in `q`'s own diagonal — true of any `q` whatever, mutated or not. Every
    /// other fixture here also puts the receiver on the `+x` axis, where two coordinates are zero
    /// and the ENU construction, the line-of-sight difference and `norm`/`cross`/`dot` all lose
    /// terms to the multiplication by zero.
    ///
    /// So: a receiver with no zero coordinate, an ENU basis written out in explicit trigonometry
    /// rather than borrowed from `cross`/`normalize`, and the analytic `Q` for `n` satellites
    /// uniformly spaced in azimuth at elevation `el` plus one at zenith. With `s = sin el`,
    /// `c = cos el` and `D = n(1 − s)²`, the normal matrix is block diagonal and inverts by hand to
    /// `Q_EE = Q_NN = 2/(n c²)`, `Q_UU = (1 + n)/D`, `Q_TT = (1 + n s²)/D`. The result depends on
    /// neither the receiver's latitude nor the ring's azimuth origin, which is itself the check
    /// that the ENU frame is built correctly.
    #[test]
    fn dop_matches_the_closed_form_for_a_zenith_plus_ring_sky() {
        const N_RING: u32 = 4;
        let el = 25.0_f64.to_radians();
        let (lat, lon) = (45.0_f64.to_radians(), 30.0_f64.to_radians());
        let (sp, cp) = (libm::sin(lat), libm::cos(lat));
        let (sl, cl) = (libm::sin(lon), libm::cos(lon));
        let up = [cp * cl, cp * sl, sp];
        let east = [-sl, cl, 0.0];
        let north = [-sp * cl, -sp * sl, cp];
        let rx = [RE * up[0], RE * up[1], RE * up[2]];
        let at = |el: f64, az: f64| {
            let (ce, se) = (libm::cos(el), libm::sin(el));
            let (e, n) = (ce * libm::sin(az), ce * libm::cos(az));
            core::array::from_fn(|k| rx[k] + R_GPS * (e * east[k] + n * north[k] + se * up[k]))
        };

        let mut sats = vec![at(core::f64::consts::FRAC_PI_2, 0.0)];
        for k in 0..N_RING {
            sats.push(at(
                el,
                core::f64::consts::TAU * f64::from(k) / f64::from(N_RING),
            ));
        }
        let d = dop(rx, &sats).expect("well-conditioned geometry");

        let (s, c) = (libm::sin(el), libm::cos(el));
        let n = f64::from(N_RING);
        let den = n * (1.0 - s) * (1.0 - s);
        let (q_ee, q_uu, q_tt) = (2.0 / (n * c * c), (1.0 + n) / den, (1.0 + n * s * s) / den);
        for (got, want, name) in [
            (d.hdop, libm::sqrt(2.0 * q_ee), "hdop"),
            (d.vdop, libm::sqrt(q_uu), "vdop"),
            (d.tdop, libm::sqrt(q_tt), "tdop"),
            (d.pdop, libm::sqrt(2.0 * q_ee + q_uu), "pdop"),
            (d.gdop, libm::sqrt(2.0 * q_ee + q_uu + q_tt), "gdop"),
        ] {
            assert!((got - want).abs() < 1e-12 * want, "{name}: {got} vs {want}");
        }
    }

    /// A receiver at latitude 45°, longitude 30° — no zero coordinate — and a function placing a
    /// satellite along a given East/North/Up direction from it, with the ENU basis written out in
    /// trigonometry rather than borrowed from `cross`/`normalize`.
    fn sky_at_45_30(directions: &[[f64; 3]]) -> ([f64; 3], Vec<[f64; 3]>) {
        let (lat, lon) = (45.0_f64.to_radians(), 30.0_f64.to_radians());
        let (sp, cp) = (libm::sin(lat), libm::cos(lat));
        let (sl, cl) = (libm::sin(lon), libm::cos(lon));
        let up = [cp * cl, cp * sl, sp];
        let east = [-sl, cl, 0.0];
        let north = [-sp * cl, -sp * sl, cp];
        let rx = [RE * up[0], RE * up[1], RE * up[2]];
        let sats = directions
            .iter()
            .map(|d| {
                core::array::from_fn(|k| {
                    rx[k] + R_GPS * (d[0] * east[k] + d[1] * north[k] + d[2] * up[k])
                })
            })
            .collect();
        (rx, sats)
    }

    /// (G1) Every DOP is the root of its trace of `(GᵀG)⁻¹`, against the exact rationals of
    /// `derivations/gnss_dop.mac` (re-inverted over ℚ by GAP in `derivations/gnss_dop.g`).
    ///
    /// Six lines of sight that are rational unit vectors — Pythagorean quadruples `(2,2,1)/3`,
    /// `(−6,2,3)/7`, `(3,−4,0)/5`, `(−2,−6,9)/11`, `(12,−4,3)/13` and the zenith — so `Q` is a
    /// matrix of rationals and each DOP the square root of a rational: `Q_EE = 170814343/278268670`,
    /// `Q_NN = 196061117/278268670`, `Q_UU = 106318505/55653734`, `Q_TT = 199779533/278268670`.
    /// No two rows are related by a symmetry, so `Q` has no zero off-diagonal entry to hide an
    /// index error, and the receiver sits on no axis. The residual is the rounding of the ENU
    /// basis and of the line-of-sight normalisation, a few parts in 1e16.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn dop_is_the_root_of_the_cofactor_traces_on_rational_geometry() {
        let directions: [[f64; 3]; 6] = [
            [0.0, 0.0, 1.0],
            [2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0],
            [-6.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0],
            [3.0 / 5.0, -4.0 / 5.0, 0.0],
            [-2.0 / 11.0, -6.0 / 11.0, 9.0 / 11.0],
            [12.0 / 13.0, -4.0 / 13.0, 3.0 / 13.0],
        ];
        let (rx, sats) = sky_at_45_30(&directions);
        let d = dop(rx, &sats).expect("full-rank geometry");
        for (got, want, name) in [
            (
                d.gdop,
                1.986_634_435_484_349_926_596_203_094_111_732_894_561,
                "gdop",
            ),
            (
                d.pdop,
                1.796_880_299_113_656_518_942_895_229_090_418_949_024,
                "pdop",
            ),
            (
                d.hdop,
                1.148_225_467_974_813_111_790_756_685_810_648_289_647,
                "hdop",
            ),
            (
                d.vdop,
                1.382_156_678_541_476_045_744_268_645_032_337_023_783,
                "vdop",
            ),
            (
                d.tdop,
                8.473_119_678_780_879_761_251_500_658_696_976_273_513e-1,
                "tdop",
            ),
        ] {
            assert!(
                (got - want).abs() < 1e-14 * want,
                "{name}: {got:.17} vs {want:.17}, rel {:.2e}",
                (got - want).abs() / want
            );
        }
    }

    /// (G3) The regular tetrahedron — the zenith and three directions at `sin el = −1/3`, 120°
    /// apart in azimuth — has `Σeᵢ = 0` and `Σeᵢeᵢᵀ = (4/3)I`, so `Q = diag(3/4, 3/4, 3/4, 1/4)` and
    /// the DOPs are `GDOP = √10/2`, `PDOP = 3/2`, `HDOP = √(3/2)`, `VDOP = √3/2`, `TDOP = 1/2`
    /// (`derivations/gnss_dop.mac`, `derivations/gnss_dop.g`). Three of the four are below the
    /// horizon, which `dop` does not ask about and the closed form does not need.
    #[test]
    fn the_regular_tetrahedron_has_its_closed_form_dops() {
        let (s, c) = (-1.0 / 3.0, 2.0 * core::f64::consts::SQRT_2 / 3.0);
        let mut directions = vec![[0.0, 0.0, 1.0]];
        for k in 0..3 {
            let az = core::f64::consts::TAU * f64::from(k) / 3.0;
            directions.push([c * libm::sin(az), c * libm::cos(az), s]);
        }
        let (rx, sats) = sky_at_45_30(&directions);
        let d = dop(rx, &sats).expect("the tetrahedron is full rank");
        for (got, want, name) in [
            (d.gdop, libm::sqrt(10.0) / 2.0, "gdop"),
            (d.pdop, 1.5, "pdop"),
            (d.hdop, libm::sqrt(1.5), "hdop"),
            (d.vdop, libm::sqrt(3.0) / 2.0, "vdop"),
            (d.tdop, 0.5, "tdop"),
        ] {
            assert!(
                (got - want).abs() < 1e-14 * want,
                "{name}: {got:.17} vs {want:.17}"
            );
        }
    }

    /// The vector helpers, directly. Both call sites inside `dop` annihilate most of their terms —
    /// `cross([0,0,1], up)` has two zero components in its first argument, and `east[2]` is
    /// structurally zero in the second — so nine of `cross`'s operators and five of `dot`'s were
    /// never exercised on a general argument.
    #[test]
    // Every product, sum and difference here is a small integer, and `sqrt(49)` is correctly
    // rounded to exactly 7 — so the equalities are exact, and a tolerance would only weaken them.
    #[allow(clippy::float_cmp)]
    fn the_vector_helpers_are_the_textbook_cross_and_dot() {
        assert_eq!(cross([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), [-3.0, 6.0, -3.0]);
        assert_eq!(dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 32.0);
        assert_eq!(norm([2.0, 3.0, 6.0]), 7.0);
    }

    /// The three ways `invert4` can decline, and the one way it must not.
    #[test]
    fn invert4_pivots_and_refuses_a_singular_matrix() {
        // A permutation matrix is its own inverse, and its leading entry is zero — so it is
        // solvable only if the pivot search actually swaps rows.
        let p = [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ];
        assert_eq!(invert4(p).unwrap(), p);

        let mut singular = [[0.0; 4]; 4];
        for i in 0..3 {
            singular[i][i] = 1.0;
        }
        assert!(invert4(singular).is_none(), "a zero row is not invertible");

        // The pivot floor is exclusive: 1e-12 itself is still accepted.
        let mut tiny = singular;
        tiny[3][3] = 1e-12;
        assert!(invert4(tiny).is_some(), "1e-12 is not below the floor");
    }

    /// The two degenerate receivers: one with no local vertical at all, one where the East
    /// construction collapses and the fallback has to stand in.
    #[test]
    fn a_receiver_without_a_usable_vertical_falls_back_or_declines() {
        let tau = core::f64::consts::TAU;
        let sky: Vec<[f64; 3]> = (0..4)
            .map(|k| sat(25.0_f64.to_radians(), tau * f64::from(k) / 4.0))
            .chain(core::iter::once(sat(1.4, 0.0)))
            .collect();
        assert!(dop([0.0; 3], &sky).is_none(), "the geocentre has no zenith");

        // At the pole `cross([0,0,1], up)` is the zero vector and the fallback East is what makes
        // the frame; the DOPs are the same closed-form numbers as at any other latitude.
        let pole = [0.0, 0.0, RE];
        let el = 25.0_f64.to_radians();
        let at = |el: f64, az: f64| {
            let (ce, se) = (libm::cos(el), libm::sin(el));
            let (e, n) = (ce * libm::sin(az), ce * libm::cos(az));
            [R_GPS * e, R_GPS * n, RE + R_GPS * se]
        };
        let mut sats = vec![at(core::f64::consts::FRAC_PI_2, 0.0)];
        for k in 0..4 {
            sats.push(at(el, tau * f64::from(k) / 4.0));
        }
        let d = dop(pole, &sats).expect("the fallback East gives a usable frame");
        let (s, c) = (libm::sin(el), libm::cos(el));
        let den = 4.0 * (1.0 - s) * (1.0 - s);
        let want = libm::sqrt(2.0 * (2.0 / (4.0 * c * c)) + 5.0 / den + (1.0 + 4.0 * s * s) / den);
        assert!(
            (d.gdop - want).abs() < 1e-12 * want,
            "gdop {} vs {want}",
            d.gdop
        );
    }

    #[test]
    fn adding_satellites_never_worsens_geometry() {
        let rx = [RE, 0.0, 0.0];
        let tau = core::f64::consts::TAU;
        // A spanning base: one high, three low (so Up and the clock column aren't proportional).
        let mut four = vec![sat(80.0_f64.to_radians(), 0.0)];
        for k in 0..3 {
            four.push(sat(15.0_f64.to_radians(), tau * f64::from(k) / 3.0));
        }
        let mut more = four.clone();
        more.push(sat(45.0_f64.to_radians(), 60.0_f64.to_radians())); // an extra measurement
        let d4 = dop(rx, &four).expect("4-sat geometry");
        let d5 = dop(rx, &more).expect("5-sat geometry");
        // More measurements can only shrink the covariance ⇒ GDOP cannot increase.
        assert!(
            d5.gdop <= d4.gdop + 1e-9,
            "gdop rose: {} → {}",
            d4.gdop,
            d5.gdop
        );
    }

    #[test]
    fn clustered_satellites_are_singular_or_huge() {
        let rx = [RE, 0.0, 0.0];
        // Four satellites within a degree of each other near zenith — rank-deficient horizontally.
        let sats: Vec<[f64; 3]> = (0..4)
            .map(|k| sat(1.5 - f64::from(k) * 0.003, 0.1 * f64::from(k)))
            .collect();
        match dop(rx, &sats) {
            None => {}
            Some(d) => assert!(d.hdop > 50.0, "expected huge HDOP, got {}", d.hdop),
        }
    }
}
