"""Independent verification of the DSST third-body mean-element rates (vectorized).

Convention-robust: no hand-derived equinoctial Gauss VOP. Under Keplerian motion the
osculating slow elements are stationary, so the perturbed rate of an element is
d(el)/dt = (d el / d v) . a_pert (the fast angle M additionally carries the mean motion n,
which the port adds separately). We finite-difference (d el / d v) from a textbook rv<->
classical conversion (verified by round-trip), orbit-average over one revolution, then map
classical -> Orekit-equinoctial. Compared against the port/Orekit golden.
"""
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
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def R1(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])

    Q = R3(-Om) @ R1(-i) @ R3(-w)
    return rp @ Q.T, vp @ Q.T


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
    w = np.arctan2(np.sum(np.cross(nvec, evec) * hvec, axis=1) / h, np.sum(nvec * evec, axis=1))
    nu = np.arctan2(np.sum(np.cross(evec, rv) * hvec, axis=1) / h, np.sum(evec * rv, axis=1))
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    M = E - e * np.sin(E)
    return np.stack([a, e, i, Om, w, M], axis=1)


def equinoctial_from_classical(el):
    a, e, i, Om, w, M = el
    pom = Om + w
    th = np.tan(i / 2)
    return np.array([a, e * np.cos(pom), e * np.sin(pom),
                     th * np.cos(Om), th * np.sin(Om), M + pom])


def mean_rates(a, e, i, Om, w, r3, mu3, N=500000):
    dv = 1e-3
    Marr = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rv, vv = rv_from_classical_vec(a, e, i, Om, w, Marr)
    d = r3[None, :] - rv
    ap = mu3 * (d / np.linalg.norm(d, axis=1)[:, None] ** 3 - r3 / np.linalg.norm(r3) ** 3)
    rate = np.zeros((N, 6))
    for j in range(3):
        ej = np.zeros(3)
        ej[j] = dv
        hp = classical_from_rv_vec(rv, vv + ej)
        hm = classical_from_rv_vec(rv, vv - ej)
        dcol = (hp - hm) / (2 * dv)
        # unwrap angle columns (Om, w, M)
        for idx in (3, 4, 5):
            dcol[:, idx] = ((hp[:, idx] - hm[:, idx] + np.pi) % (2 * np.pi) - np.pi) / (2 * dv)
        rate += dcol * ap[:, j][:, None]
    return rate.mean(axis=0)  # [da, de, di, dOm, dw, dM]_pert


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


def verify(name, eq_state, r3, mu3, golden):
    a, k, h, q, p, lM = eq_state
    e = np.hypot(k, h)
    pom = np.arctan2(h, k)
    i = 2 * np.arctan(np.hypot(q, p))
    Om = np.arctan2(p, q)
    w = pom - Om
    M0 = lM - pom
    el_mean = np.array([a, e, i, Om, w, M0])
    rv0, vv0 = rv_from_classical_vec(a, e, i, Om, w, np.array([M0]))
    eq_rt = equinoctial_from_classical(classical_from_rv_vec(rv0, vv0)[0])
    rt_err = np.max(np.abs(np.array([a, np.cos(pom) * e, np.sin(pom) * e,
                                     np.tan(i / 2) * np.cos(Om), np.tan(i / 2) * np.sin(Om), lM]) - eq_rt))
    cr = mean_rates(a, e, i, Om, w, r3, mu3)
    eqr = classical_rates_to_equinoctial(el_mean, cr)
    print(f"[{name}] e={e:.5f} i={np.degrees(i):.3f} Om={np.degrees(Om):.3f} w={np.degrees(w):.3f} "
          f"M0={np.degrees(M0):.3f}  round-trip err {rt_err:.2e}")
    for lab, got, want in zip(["da", "dk", "dh", "dq", "dp", "dl"], eqr, golden, strict=True):
        rel = "n/a" if abs(want) < 1e-20 else f"{abs(got - want) / abs(want):.2e}"
        print(f"    {lab}: num {got:+.6e}   Orekit {want:+.6e}   rel {rel}")


EQ = [7.2e6, 3.09016994374947470e-03, 9.51056516295153600e-03,
      8.54890329218965000e-01, 7.69746709859868000e-01, 1.43116998663535020e+00]
SUN_POS = np.array([2.48130578062755620e10, -1.33033126695834660e11, -5.76679652360023650e10])
MOON_POS = np.array([-3.67980874244052000e08, 1.42721023286279650e08, 8.93144218169177900e07])
verify("SUN", EQ, SUN_POS, 1.32712440041279460e20,
       [0.0, 2.16956373398949160e-14, -1.00495702004443560e-14,
        -9.93909076449685000e-12, -3.09118088926159100e-11, 2.72444232947602650e-11])
verify("MOON", EQ, MOON_POS, 4.90280011845755100e12,
       [0.0, 1.72603527350549330e-12, -1.13161201468628720e-12,
        4.36466413189831100e-11, 2.46913560075631850e-11, 8.89245323174147000e-11])
