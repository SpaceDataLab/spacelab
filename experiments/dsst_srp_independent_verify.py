"""Independent verification of the DSST SRP mean-element rates (vectorized).

Same convention-robust machinery as the third-body check (finite-difference d(el)/dv from a
textbook rv<->classical conversion, orbit-average d(el)/dv . a_pert), with the SRP acceleration:
a = Cr (A/m) P_ref (d_ref/|r_sun-r|)^2 * (unit away from Sun) * lighting, and an INDEPENDENT
geometric cylindrical shadow (in shadow iff behind Earth on the anti-Sun axis and within the
Earth-radius cylinder) -- Orekit's getLLimits umbra model, computed here from first principles.

# Why the quadrature is split at the shadow boundary

A uniform grid in mean anomaly is spectrally accurate on a smooth periodic integrand and only
O(1/N) on a discontinuous one, and the lighting factor is a step. That costs `da` far more than
the other five components, because a constant force over a *closed* orbit averages to zero
semi-major-axis rate exactly: the whole of `<da/dt>` is the shadow, so the quadrature error is
measured against a difference of cancelling halves rather than against the integral. Measured on
the fixture below, a 8e5-point uniform grid gives `da` to 1.4e-2 relative and the rest to
3e-3 -- an order of magnitude worse than the 2e-4 Orekit deviation it is supposed to bound.

Locating the two shadow crossings by bisection and running Gauss-Legendre over the lit arc alone
integrates only analytic pieces: converged by order 16, and the crossings land within `4e-8 rad`
of `l_limits` instead of the `7.6e-6 rad` the scan managed. The d(el)/dv step is Richardson-
extrapolated on top, though at that point it no longer moves the answer.

# What the residual then turned out to be

With the quadrature converged, the remaining `3e-3` against Orekit stopped being numerical error
and started being a model difference: the binary cylinder is not the lighting DSST uses. Adding the
conical penumbra -- the physical two-disc occultation, from sky geometry rather than Orekit's
algebra -- takes `da` from `3.2e-3` to `4.5e-4` and `dq` from `3.3e-3` to `1.2e-4`, and puts this
script between spacelab and Orekit, `2.5x` closer to spacelab. That is the comparison worth making,
and `dsst::srp::fixed_state_rates_match_the_independent_vop_average` asserts it.

Three shadow treatments are reported, because the differences between them are the finding:

* `binary cylinder, cylinder arc` -- what this script did before;
* `conical ratio, cylinder arc` -- the model DSST specifies, independently implemented. The
  accuracy statement;
* `conical ratio, conical arcs` -- the physics, integrated over the whole orbit. It lands `3.1e-3`
  from Orekit, i.e. the DSST scheme's own approximation (truncating the penumbra at the cylindrical
  limits) is worth about `0.3%` of `da`, an order of magnitude more than the implementation
  differences being argued over.

`--converge` prints every sweep the numbers above came from.
"""

import sys

import numpy as np

MU = 3.986004415e14


def rv_from_classical_vec(a, e, i, Om, w, Marr):
    E = Marr.copy()
    for _ in range(80):
        E = E - (E - e * np.sin(E) - Marr) / (1 - e * np.cos(E))
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r = a * (1 - e * np.cos(E))
    pp = a * (1 - e * e)
    rp = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)], axis=1)
    vf = np.sqrt(MU / pp)
    vp = np.stack([-vf * np.sin(nu), vf * (e + np.cos(nu)), np.zeros_like(nu)], axis=1)

    def R3(t):
        return np.array(
            [[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]]
        )

    def R1(t):
        return np.array(
            [[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]]
        )

    Q = R3(-Om) @ R1(-i) @ R3(-w)
    return rp @ Q.T, vp @ Q.T, nu


def classical_from_rv_vec(rv, vv):
    r = np.linalg.norm(rv, axis=1)
    v2 = np.sum(vv * vv, axis=1)
    a = 1.0 / (2.0 / r - v2 / MU)
    hvec = np.cross(rv, vv)
    h = np.linalg.norm(hvec, axis=1)
    rdv = np.sum(rv * vv, axis=1)
    evec = ((v2 - MU / r)[:, None] * rv - rdv[:, None] * vv) / MU
    e = np.linalg.norm(evec, axis=1)
    i = np.arccos(np.clip(hvec[:, 2] / h, -1, 1))
    nvec = np.stack([-hvec[:, 1], hvec[:, 0], np.zeros_like(h)], axis=1)
    Om = np.arctan2(nvec[:, 1], nvec[:, 0])
    w = np.arctan2(
        np.sum(np.cross(nvec, evec) * hvec, axis=1) / h, np.sum(nvec * evec, axis=1)
    )
    nu = np.arctan2(
        np.sum(np.cross(evec, rv) * hvec, axis=1) / h, np.sum(evec * rv, axis=1)
    )
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    return np.stack([a, e, i, Om, w, M], axis=1)


def equinoctial_from_classical(el):
    a, e, i, Om, w, M = el
    pom = Om + w
    th = np.tan(i / 2)
    return np.array(
        [a, e * np.cos(pom), e * np.sin(pom), th * np.cos(Om), th * np.sin(Om), M + pom]
    )


def srp_accel(rv, r_sun, cr, aom, p_ref, d_ref, ae, shadow="cylinder"):
    """`shadow`: `cylinder` for the binary umbra DSST integrates between, `cone` for the physics."""
    d = rv - r_sun[None, :]
    dist = np.linalg.norm(d, axis=1)
    ahat = d / dist[:, None]
    k = cr * aom * p_ref * (d_ref / dist) ** 2
    lit = (
        conical_ratio(rv, r_sun, ae)
        if shadow == "cone"
        else np.where(shadow_depth(rv, r_sun, ae) > 0.0, 0.0, 1.0)
    )
    return k[:, None] * ahat * lit[:, None], lit


def conical_ratio(rv, r_sun, ae, r_sun_body=6.957e8):
    """Fraction of the solar disc still visible: 1 in full Sun, 0 in umbra, a lens overlap between.

    Elementary sky geometry, not the Orekit algebra: two circles of apparent radii `a_s` (Sun) and
    `a_e` (Earth limb) separated by `sep`, and the visible fraction is `1 - overlap/(pi a_s^2)` with
    the overlap the standard circular-segment sum. This is the *physical* model; the cylinder in
    [`shadow_depth`] is the approximation DSST integrates between.
    """
    sun_sat = rv - r_sun[None, :]
    d = np.linalg.norm(sun_sat, axis=1)
    r = np.linalg.norm(rv, axis=1)
    cos_sep = np.sum(sun_sat * rv, axis=1) / (d * r)
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    a_s = np.arcsin(r_sun_body / d)
    a_e = np.arcsin(ae / r)

    lens = np.zeros_like(sep)
    partial = (sep < a_s + a_e) & (sep > np.abs(a_e - a_s))
    s, p, q = sep[partial], a_s[partial], a_e[partial]
    c1 = np.clip((s * s + p * p - q * q) / (2 * s * p), -1.0, 1.0)
    c2 = np.clip((s * s + q * q - p * p) / (2 * s * q), -1.0, 1.0)
    area = p * p * (np.arccos(c1) - c1 * np.sqrt(1 - c1 * c1)) + q * q * (
        np.arccos(c2) - c2 * np.sqrt(1 - c2 * c2)
    )
    lens[partial] = area / (np.pi * a_s[partial] ** 2)
    lens[sep <= a_e - a_s] = 1.0
    return 1.0 - lens


def shadow_depth(rv, r_sun, ae):
    """Positive inside the cylindrical umbra, negative outside; smooth across each crossing.

    `ae - perp` on the anti-Sun side and `-(ae + perp)` on the Sun side: one continuous function
    of the true anomaly whose two sign changes are the shadow entry and exit, so a bisection can
    find them without special-casing which crossing it is on.
    """
    shat = r_sun / np.linalg.norm(r_sun)
    rpar = rv @ shat
    perp = np.linalg.norm(rv - rpar[:, None] * shat[None, :], axis=1)
    return np.where(rpar < 0.0, ae - perp, -(ae + perp))


def mean_anomaly_of_true(e, nu):
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    return E - e * np.sin(E)


def cone_depth(rv, r_sun, ae, r_sun_body=6.957e8, umbra=True):
    """Positive inside the umbra (`umbra`) or inside umbra+penumbra, negative outside."""
    sun_sat = rv - r_sun[None, :]
    d = np.linalg.norm(sun_sat, axis=1)
    r = np.linalg.norm(rv, axis=1)
    sep = np.arccos(np.clip(np.sum(sun_sat * rv, axis=1) / (d * r), -1.0, 1.0))
    a_s = np.arcsin(r_sun_body / d)
    a_e = np.arcsin(ae / r)
    return (a_e - a_s) - sep if umbra else (a_e + a_s) - sep


def shadow_arc(a, e, i, Om, w, r_sun, ae, coarse=4096, depth_fn=None):
    """The lit arc as `(M_start, M_end)` with `M_end > M_start`, and the crossing true longitudes.

    Coarse scan for the sign changes of `depth_fn` (default [`shadow_depth`], the cylinder), then
    200 bisections each, which reaches the f64 floor on an interval of 2*pi/4096. Returns `None`
    when the orbit never crosses that boundary.
    """
    depth_fn = depth_fn or (lambda rv: shadow_depth(rv, r_sun, ae))

    def depth(nu):
        rv, _, _ = rv_from_classical_vec(
            a, e, i, Om, w, mean_anomaly_of_true(e, np.atleast_1d(nu))
        )
        return depth_fn(rv)

    nus = np.linspace(0.0, 2 * np.pi, coarse, endpoint=False)
    d = depth(nus)
    idx = np.where(np.sign(d) != np.sign(np.roll(d, -1)))[0]
    if len(idx) != 2:
        return None
    lo = nus[idx]
    hi = np.where(idx + 1 < coarse, nus[(idx + 1) % coarse], 2 * np.pi)
    dlo = d[idx]
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        dmid = depth(mid)
        same = np.sign(dmid) == np.sign(dlo)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
        dlo = np.where(same, dmid, dlo)
    nu_cross = 0.5 * (lo + hi)
    # The crossing that leaves the umbra is the one where the depth goes positive -> negative.
    entering = depth(nu_cross - 1e-6) < 0.0
    nu_in = nu_cross[entering][0]
    nu_out = nu_cross[~entering][0]
    m_in, m_out = mean_anomaly_of_true(e, nu_in), mean_anomaly_of_true(e, nu_out)
    lit = (m_out, m_in + 2 * np.pi) if m_out > m_in else (m_out, m_in)
    lon = [float((Om + w + nu + np.pi) % (2 * np.pi) - np.pi) for nu in (nu_in, nu_out)]
    return lit, sorted(lon)


def conical_intervals(a, e, i, Om, w, r_sun, ae):
    """The three M-intervals the conical ratio is non-zero on: ramp up, full Sun, ramp down.

    The ratio is C^0 but not C^1 at the outer penumbra boundary, so Gauss-Legendre is run on each
    piece separately rather than across the kink. The umbra contributes nothing and is dropped.
    """
    umbra, _ = shadow_arc(
        a,
        e,
        i,
        Om,
        w,
        r_sun,
        ae,
        depth_fn=lambda rv: cone_depth(rv, r_sun, ae, umbra=True),
    )
    outer, _ = shadow_arc(
        a,
        e,
        i,
        Om,
        w,
        r_sun,
        ae,
        depth_fn=lambda rv: cone_depth(rv, r_sun, ae, umbra=False),
    )
    # `shadow_arc` returns the *lit* arc, so `umbra[0]` is the umbra exit and `outer[0]` the outer
    # penumbra exit; the outer boundary is always crossed first on the way out.
    u0, u1 = umbra
    o0, o1 = outer
    while o0 < u0:
        o0 += 2 * np.pi
    while o1 > u1:
        o1 -= 2 * np.pi
    return [(u0, o0), (o0, o1), (o1, u1)]


def rates_at(a, e, i, Om, w, accel_fn, marr, dv):
    """`d(classical)/dt` from the perturbing acceleration at each mean anomaly in `marr`."""
    rv, vv, _ = rv_from_classical_vec(a, e, i, Om, w, marr)
    ap, lit = accel_fn(rv)
    rate = np.zeros((len(marr), 6))
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = dv
        hp = classical_from_rv_vec(rv, vv + ej)
        hm = classical_from_rv_vec(rv, vv - ej)
        dcol = (hp - hm) / (2 * dv)
        for idx in (3, 4, 5):
            dcol[:, idx] = ((hp[:, idx] - hm[:, idx] + np.pi) % (2 * np.pi) - np.pi) / (
                2 * dv
            )
        rate += dcol * ap[:, j][:, None]
    return rate, lit


def mean_rates_uniform(a, e, i, Om, w, accel_fn, n=800_000, dv=1e-3):
    """The original rectangle rule on a uniform mean-anomaly grid, kept for the convergence table."""
    marr = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rate, lit = rates_at(a, e, i, Om, w, accel_fn, marr, dv)
    return rate.mean(axis=0), lit.mean()


def mean_rates_split(
    a, e, i, Om, w, accel_fn, arc, order=256, dv=1e-3, richardson=True
):
    """Gauss-Legendre over the lit arc alone, with the d(el)/dv step Richardson-extrapolated."""
    x, wts = np.polynomial.legendre.leggauss(order)
    m0, m1 = arc
    marr = 0.5 * (m1 - m0) * x + 0.5 * (m1 + m0)
    scale = 0.5 * (m1 - m0) / (2 * np.pi)

    def average(step):
        rate, lit = rates_at(a, e, i, Om, w, accel_fn, marr, step)
        assert lit.min() > 0.0, "a Gauss node landed in full umbra: the arc is wrong"
        return scale * (wts[:, None] * rate).sum(axis=0)

    coarse = average(dv)
    if not richardson:
        return coarse
    # Central differences are O(h^2), so one halving removes the leading term.
    return (4.0 * average(0.5 * dv) - coarse) / 3.0


def mean_rates_intervals(a, e, i, Om, w, accel_fn, intervals, order=256, dv=1e-3):
    """Gauss-Legendre on each interval, summed — for an integrand with kinks between them."""
    return sum(
        mean_rates_split(a, e, i, Om, w, accel_fn, arc, order=order, dv=dv)
        for arc in intervals
    )


def classical_rates_to_equinoctial(el_mean, cr):
    J = np.zeros((6, 6))
    step = np.array([1.0, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8])
    for j in range(6):
        ej = np.zeros(6)
        ej[j] = step[j]
        ep = equinoctial_from_classical(el_mean + ej)
        em = equinoctial_from_classical(el_mean - ej)
        d = (ep - em) / (2 * step[j])
        d[5] = ((ep[5] - em[5] + np.pi) % (2 * np.pi) - np.pi) / (2 * step[j])
        J[:, j] = d
    return J @ cr


EQ = [
    7.2e6,
    3.09016994374947470e-03,
    9.51056516295153600e-03,
    8.54890329218965000e-01,
    7.69746709859868000e-01,
    1.43116998663535020e00,
]
SUN_POS = np.array(
    [2.48130578062755620e10, -1.33033126695834660e11, -5.76679652360023650e10]
)
CR, AREA, MASS, P_REF, D_REF, AE = (
    1.5,
    10.0,
    1000.0,
    4.56e-6,
    149_597_870_000.0,
    6.37813646e6,
)
RATES = [
    -1.66681113612731410e-09,
    6.42088118316530040e-12,
    -2.06453695796115100e-12,
    5.54383062682849500e-13,
    1.72237499933817420e-12,
    2.94388190334062430e-12,
]
LL = (-4.39803656243010300e00, 6.34076980768220600e-01)
LABELS = ["da", "dk", "dh", "dq", "dp", "dl"]

a, k, h, q, p, lM = EQ
e = np.hypot(k, h)
pom = np.arctan2(h, k)
i = 2 * np.arctan(np.hypot(q, p))
Om = np.arctan2(p, q)
w = pom - Om
M0 = lM - pom
el_mean = np.array([a, e, i, Om, w, M0])


def accel_with(shadow):
    return lambda rv: srp_accel(
        rv, SUN_POS, CR, AREA / MASS, P_REF, D_REF, AE, shadow=shadow
    )


arc, lon = shadow_arc(a, e, i, Om, w, SUN_POS, AE)
cone_arcs = conical_intervals(a, e, i, Om, w, SUN_POS, AE)
# The conical ratio has a kink where the outer penumbra ends, so the DSST arc is cut there too:
# Gauss-Legendre is spectral only across an analytic piece.
cylinder_arc_pieces = [
    (arc[0], cone_arcs[1][0]),
    (cone_arcs[1][0], cone_arcs[1][1]),
    (cone_arcs[1][1], arc[1]),
]
accel = accel_with("cylinder")
print(
    f"cylinder entry/exit true-lon {lon}  (Orekit getLLimits {LL[0]:.6f}, {LL[1]:.6f})"
)
print(
    f"lit arc in M: [{arc[0]:.9f}, {arc[1]:.9f}], fraction {(arc[1] - arc[0]) / (2 * np.pi):.6f}"
)
print("conical pieces in M: " + ", ".join(f"[{x:.6f}, {y:.6f}]" for x, y in cone_arcs))


def report(label, eqr, ref=RATES):
    rel = [abs(g - r) / abs(r) for g, r in zip(eqr, ref, strict=True)]
    print(
        f"  {label:<34}"
        + "  ".join(f"{name} {v:.3e}" for name, v in zip(LABELS, rel, strict=True))
    )
    return rel


if "--converge" in sys.argv:
    print(
        "\nquadrature: uniform rectangle rule on the discontinuous integrand (cylinder shadow)"
    )
    for n in (50_000, 200_000, 800_000):
        cr_rates, frac = mean_rates_uniform(a, e, i, Om, w, accel, n=n)
        report(
            f"N={n} lit {frac:.5f}", classical_rates_to_equinoctial(el_mean, cr_rates)
        )

    print(
        "\nquadrature: Gauss-Legendre over the lit arc, no Richardson (cylinder shadow)"
    )
    for order in (16, 32, 64, 128, 256):
        cr_rates = mean_rates_split(
            a, e, i, Om, w, accel, arc, order=order, richardson=False
        )
        report(f"n={order}", classical_rates_to_equinoctial(el_mean, cr_rates))

    print("\nfinite difference: d(el)/dv step at order 256, Richardson off / on")
    for dv in (1e-1, 1e-2, 1e-3, 1e-4):
        for rich in (False, True):
            cr_rates = mean_rates_split(
                a, e, i, Om, w, accel, arc, order=256, dv=dv, richardson=rich
            )
            report(
                f"dv={dv:.0e} {'R' if rich else ' '}",
                classical_rates_to_equinoctial(el_mean, cr_rates),
            )

    print("\nconical ratio on the DSST arc, Gauss order sweep over the three pieces")
    for order in (16, 64, 256, 512):
        cr_rates = mean_rates_intervals(
            a, e, i, Om, w, accel_with("cone"), cylinder_arc_pieces, order=order
        )
        report(f"n={order}", classical_rates_to_equinoctial(el_mean, cr_rates))

    print("\nconical ratio on the DSST arc, d(el)/dv step at order 256")
    for dv in (1e-1, 1e-2, 1e-3, 1e-4):
        cr_rates = mean_rates_intervals(
            a, e, i, Om, w, accel_with("cone"), cylinder_arc_pieces, dv=dv
        )
        report(f"dv={dv:.0e}", classical_rates_to_equinoctial(el_mean, cr_rates))

print("\nrelative to Orekit getMeanElementRate:")
variants = {
    "binary cylinder, cylinder arc": mean_rates_split(a, e, i, Om, w, accel, arc),
    "conical ratio, cylinder arc": mean_rates_intervals(
        a, e, i, Om, w, accel_with("cone"), cylinder_arc_pieces
    ),
    "conical ratio, conical arcs": mean_rates_intervals(
        a, e, i, Om, w, accel_with("cone"), cone_arcs
    ),
}
eqrs = {k: classical_rates_to_equinoctial(el_mean, v) for k, v in variants.items()}
for lab, eqr in eqrs.items():
    report(lab, eqr)
print(
    "  " + f"{'spacelab dsst::srp (for reference)':<34}"
    "da 3.455e-04  dk 5.311e-05  dh 5.311e-05  dq 2.049e-04  dp 2.050e-04  dl 2.080e-04"
)

print("\nthe VOP orbit-average, conical ratio over the DSST cylinder arc:")
for lab, v in zip(LABELS, eqrs["conical ratio, cylinder arc"], strict=True):
    print(f"    {lab}: {v:+.17e}")
