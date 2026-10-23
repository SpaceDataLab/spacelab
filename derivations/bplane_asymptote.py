"""60-dps oracles for `spacelab_astro::patched`.

`hyperbolic_arc` is evaluated from its defining relations (vis-viva, the eccentricity of a conic
through a given periapsis, the asymptote turn angle). `b_plane` is checked against the *definition*
of the aiming plane: propagate the hyperbola back onto the incoming asymptote and read S = v-hat
and B = r - (r.S) S off it, then re-express in closed form.
"""

from mpmath import asin, cosh, findroot, matrix, mp, mpf, norm, nstr, sinh, sqrt

mp.dps = 60


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


def unit(a):
    return a / norm(a)


def group(x, digits=25):
    s = nstr(x, digits, strip_zeros=False)
    if "e" in s:
        m, e = s.split("e")
        return f"{grp(m)}e{int(e)}"
    return grp(s)


def grp(s):
    neg = s.startswith("-")
    s = s.lstrip("-")
    ip, _, fp = s.partition(".")
    out = []
    for k in range(0, len(ip), 3):
        lo = max(len(ip) - k - 3, 0)
        out.append(ip[lo : len(ip) - k])
    ip_g = "_".join(reversed(out)) if len(ip) > 4 else ip
    fp_g = "_".join(fp[k : k + 3] for k in range(0, len(fp), 3)) if fp else ""
    return ("-" if neg else "") + (f"{ip_g}.{fp_g}" if fp_g else ip_g)


print("=== hyperbolic_arc ===")
for v_inf, mu, r_p in [
    (mpf(3), mpf("398600"), mpf("6578")),
    (mpf("1.35"), mpf("42828.37"), mpf("3689.5")),
    (mpf("5.72"), mpf("126686534"), mpf("78240")),
]:
    v2 = v_inf * v_inf
    a = -mu / v2
    e = 1 + r_p * v2 / mu
    vp = sqrt(v2 + 2 * mu / r_p)
    delta = 2 * asin(mu / (mu + r_p * v2))
    # Cross-check: the turn angle must also be 2 asin(1/e).
    assert abs(delta - 2 * asin(1 / e)) < mpf(10) ** -50
    print(
        f"    ({group(v_inf)}, {group(mu)}, {group(r_p)}, {group(a)}, {group(e)},"
        f" {group(vp)}, {group(delta)}),"
    )


def b_plane_from_definition(r_vec, v_vec, mu, north):
    rn = norm(r_vec)
    v2 = dot(v_vec, v_vec)
    h_vec = cross(r_vec, v_vec)
    h_hat = unit(h_vec)
    a = 1 / (2 / rn - v2 / mu)
    rv = dot(r_vec, v_vec)
    c = v2 - mu / rn
    e_vec = matrix([(c * r_vec[i] - rv * v_vec[i]) / mu for i in range(3)])
    e = norm(e_vec)
    e_hat = unit(e_vec)
    q_hat = cross(h_hat, e_hat)

    # Far back on the incoming branch, from the hyperbolic Kepler equation.
    n = sqrt(mu / (-a) ** 3)
    h_far = -mp.acosh(mpf(10) ** 16 / e)
    m_far = e * sinh(h_far) - h_far
    big_h = findroot(lambda x: e * sinh(x) - x - m_far, h_far)
    x = a * (cosh(big_h) - e)
    y = -a * sqrt(e * e - 1) * sinh(big_h)
    hdot = n / (e * cosh(big_h) - 1)
    r_far = matrix([x * e_hat[i] + y * q_hat[i] for i in range(3)])
    v_far = matrix(
        [
            a * sinh(big_h) * hdot * e_hat[i]
            - a * sqrt(e * e - 1) * cosh(big_h) * hdot * q_hat[i]
            for i in range(3)
        ]
    )
    s_def = unit(v_far)
    b_def = matrix([r_far[i] - dot(r_far, s_def) * s_def[i] for i in range(3)])

    # The closed form the code should use.
    sqe = sqrt(e * e - 1)
    s_cf = matrix([(e_hat[i] + sqe * q_hat[i]) / e for i in range(3)])
    b_mag = abs(a) * sqe
    b_cf = matrix([b_mag * v for v in cross(s_cf, h_hat)])
    assert norm(s_cf - s_def) < mpf(10) ** -12, norm(s_cf - s_def)
    assert norm(b_cf - b_def) / b_mag < mpf(10) ** -12, norm(b_cf - b_def) / b_mag

    t_hat = unit(cross(s_cf, north))
    r_hat = cross(s_cf, t_hat)
    return dot(b_cf, t_hat), dot(b_cf, r_hat), b_mag, s_cf


print("=== b_plane ===")
CASES = [
    (
        matrix([mpf("10000"), mpf("2000"), mpf("500")]),
        matrix([mpf(1), mpf(11), mpf("0.5")]),
        mpf("398600"),
        matrix([mpf(0), mpf(0), mpf(1)]),
    ),
    (
        matrix([mpf("-8200"), mpf("14300"), mpf("-3100")]),
        matrix([mpf("5.7"), mpf("-1.9"), mpf("4.2")]),
        mpf("42828.37"),
        matrix([mpf("0.4"), mpf("-0.6"), mpf("1.86")]),
    ),
]
for r_vec, v_vec, mu, north in CASES:
    bt, br, bm, s = b_plane_from_definition(r_vec, v_vec, mu, north)
    print(f"    ({group(bt)}, {group(br)}, {group(bm)}),")
    print("      # S =", [nstr(x, 20) for x in s])
