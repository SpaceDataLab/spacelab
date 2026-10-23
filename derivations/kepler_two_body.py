"""Exact two-body propagation at 60 decimal digits, as an oracle for `spacelab_astro::propagate`.

The formulation is deliberately not the one under test. `propagate` solves the *universal* Kepler
equation in Sundman's variable chi and rebuilds the state from Stumpff functions; this script goes
through classical elements, solves Kepler's equation in the eccentric (or hyperbolic) anomaly, and
uses the Lagrange coefficients written in that anomaly. The two share only the physics.

Run: `uv run --with mpmath python derivations/kepler_two_body.py`. It first reproduces the rows
already committed in `crates/spacelab-astro/src/kepler.rs` -- a formulation that could not do that
has no business emitting new ones -- and then prints the near-parabolic table.
"""

from mpmath import (
    atan,
    atan2,
    cos,
    cosh,
    matrix,
    mp,
    mpf,
    norm,
    sin,
    sinh,
    sqrt,
    tan,
)

mp.dps = 60

MU = mpf(398600)


def cross(a, b):
    return matrix(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def propagate(r0v, v0v, mu, dt):
    """The state at `t0 + dt` on the two-body conic through `(r0v, v0v)`."""
    r0 = norm(r0v)
    v2 = dot(v0v, v0v)
    h = cross(r0v, v0v)
    c = v2 - mu / r0
    rv = dot(r0v, v0v)
    e_vec = matrix([(c * r0v[i] - rv * v0v[i]) / mu for i in range(3)])
    e = norm(e_vec)
    a = 1 / (2 / r0 - v2 / mu)

    # True anomaly of the initial state, signed by the radial velocity.
    nu0 = atan2(dot(cross(e_vec, r0v), h / norm(h)), dot(e_vec, r0v))

    if a > 0:
        big_e0 = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(nu0 / 2))
        m0 = big_e0 - e * sin(big_e0)
        m = m0 + sqrt(mu / a**3) * dt
        big_e = m if e < mpf("0.8") else m + e * sin(m)
        for _ in range(200):
            step = (big_e - e * sin(big_e) - m) / (1 - e * cos(big_e))
            if step == 0:
                break
            big_e -= step
        d = big_e - big_e0
        f = 1 - a / r0 * (1 - cos(d))
        g = dt - sqrt(a**3 / mu) * (d - sin(d))
        rv_new = matrix([f * r0v[i] + g * v0v[i] for i in range(3)])
        rn = norm(rv_new)
        fd = -sqrt(mu * a) / (rn * r0) * sin(d)
        gd = 1 - a / rn * (1 - cos(d))
    else:
        ah = -a
        big_h0 = 2 * mp.atanh(sqrt((e - 1) / (e + 1)) * tan(nu0 / 2))
        n0 = e * sinh(big_h0) - big_h0
        n = n0 + sqrt(mu / ah**3) * dt
        big_h = mp.asinh(n / e) if abs(n) > 1 else n / (e - 1)
        for _ in range(400):
            step = (e * sinh(big_h) - big_h - n) / (e * cosh(big_h) - 1)
            if step == 0:
                break
            big_h -= step
        d = big_h - big_h0
        f = 1 - a / r0 * (1 - cosh(d))
        g = dt - sqrt(ah**3 / mu) * (sinh(d) - d)
        rv_new = matrix([f * r0v[i] + g * v0v[i] for i in range(3)])
        rn = norm(rv_new)
        fd = -sqrt(mu * ah) / (rn * r0) * sinh(d)
        gd = 1 - a / rn * (1 - cosh(d))

    vv_new = matrix([fd * r0v[i] + gd * v0v[i] for i in range(3)])
    # `f g' - f' g = 1` is the Wronskian of the two-body flow; it is one digit of self-check that
    # costs nothing, and it fails loudly if a branch above is written with the wrong sign.
    assert abs(f * gd - fd * g - 1) < mpf(10) ** (-50), (
        f"Wronskian off at dt={dt}: {f * gd - fd * g}"
    )
    return rv_new, vv_new, e, a


def rust_float(x):
    """`x` at 25 significant digits with the crate's digit grouping."""
    s = mp.nstr(x, 25, strip_zeros=False)
    if "e" in s:
        return s
    neg, s = (s.startswith("-"), s.lstrip("-"))
    whole, _, frac = s.partition(".")
    whole = "_".join(
        [whole[: len(whole) % 3 or 3]]
        + [whole[i : i + 3] for i in range(len(whole) % 3 or 3, len(whole), 3)]
    )
    frac = "_".join(frac[i : i + 3] for i in range(0, len(frac), 3))
    return ("-" if neg else "") + whole + "." + frac


COMMITTED = {
    # (name, r0, v0, dt): the first row of each committed table, to validate the formulation.
    "ellipse": (
        (mpf(7000), mpf(1200), mpf(-800)),
        (mpf("-1.3"), mpf("6.9"), mpf("2.4")),
        mpf("1e-3"),
        (
            "6999.998699996178542779229",
            "1200.006899999344891752203",
            "-799.9975999995632624416608",
            "-1.300007642914409864244779",
            "6.899998689782213000870119",
            "2.400000873474703356454070",
        ),
    ),
    "hyperbola": (
        (mpf(8000), mpf(1500), mpf(-900)),
        (mpf("1.1"), mpf("9.5"), mpf("3.2")),
        mpf("1e-3"),
        None,
    ),
}


def check_committed():
    """Reproduce the first committed ellipse row.

    The position matches to 1e-22, the limit of the 25-digit literals. The velocity matches only to
    5e-17: that row's velocity carries about a quarter of an ulp of error -- it violates angular
    momentum by 5e-17 and energy by 8e-17 relative, where the state computed here conserves both to
    1e-62 -- so the committed velocities are good to double precision and no further. That is well
    inside the 8-epsilon tolerance the test allows, and it is the reason that tolerance cannot be
    tightened against those rows.
    """
    r0v_t, v0v_t, dt, want = COMMITTED["ellipse"]
    rv, vv, e, a = propagate(matrix(r0v_t), matrix(v0v_t), MU, dt)
    got = [rv[0], rv[1], rv[2], vv[0], vv[1], vv[2]]
    rel = [abs(g - mpf(w)) / abs(mpf(w)) for g, w in zip(got, want, strict=True)]
    print(f"committed ellipse row (e={mp.nstr(e, 6)}, a={mp.nstr(a, 10)}):")
    print(
        f"  position worst relative {mp.nstr(max(rel[:3]), 3)}, velocity worst {mp.nstr(max(rel[3:]), 3)}"
    )
    assert max(rel[:3]) < mpf(10) ** (-21), (
        "the formulation does not reproduce the committed positions"
    )
    assert max(rel[3:]) < mpf(10) ** (-16), "the velocity is further out than one ulp"


def periapsis_state(r_p, e, mu):
    """A periapsis state at `r_p` with eccentricity `e`, rotated 3-1-3 by fixed angles so that no
    component is zero and no axis of the solver can be corrupted unnoticed."""
    v_p = sqrt(mu * (1 + e) / r_p)
    om, inc, raan = mpf("0.7"), mpf("0.5"), mpf("1.1")
    rot = []
    for base in ([r_p, mpf(0), mpf(0)], [mpf(0), v_p, mpf(0)]):
        x, y = (
            base[0] * cos(om) - base[1] * sin(om),
            base[0] * sin(om) + base[1] * cos(om),
        )
        y, z = y * cos(inc), y * sin(inc)
        x, y = x * cos(raan) - y * sin(raan), x * sin(raan) + y * cos(raan)
        # Round the epoch state to double before propagating it. The Rust caller can only pass a
        # double, so the oracle has to be the exact flow of the state it will actually be handed --
        # otherwise the comparison charges the solver for the rounding of its own input.
        rot.append(matrix([mpf(float(c)) for c in (x, y, z)]))
    return rot[0], rot[1]


def emit(name, e, spans):
    r0v, v0v = periapsis_state(mpf(7000), e, MU)
    print(f"\n// {name}: e = {e}, periapsis 7000 km")
    print("    const " + name.upper() + f": [(f64, [f64; 6]); {len(spans)}] = [")
    for dt in spans:
        rv, vv, ecc, a = propagate(r0v, v0v, MU, mpf(dt))
        print(f"        ({dt},")
        print("            [")
        for x in (rv[0], rv[1], rv[2], vv[0], vv[1], vv[2]):
            print(f"                {rust_float(x)},")
        print("            ],")
        print("        ),")
    print("    ];")
    print("    // r0 =", [rust_float(x) for x in r0v])
    print("    // v0 =", [rust_float(x) for x in v0v])
    print("    // e  =", mp.nstr(ecc, 20), " a =", mp.nstr(a, 20))


if __name__ == "__main__":
    check_committed()
    emit(
        "near_parabolic_ellipse",
        mpf("0.999"),
        ["1e-3", "1.0", "100.0", "5828.5", "291426.0", "1e7"],
    )
    emit(
        "near_parabolic_hyperbola",
        mpf("1.0001"),
        ["1e-3", "1.0", "100.0", "5828.5", "291426.0", "1e7"],
    )
