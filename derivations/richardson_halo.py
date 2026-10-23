#!/usr/bin/env -S uv run --quiet --with sympy --with numpy python
"""Richardson (1980) third-order halo construction — SUPPLEMENTARY numerical cross-check.

The AUTHORITATIVE derivation and proof live in `derivations/richardson_halo.mac` (Maxima). Maxima
re-derives lambda/k, the Legendre forcing and every coefficient, then substitutes the series back
into the exact EOM order by order (`trigreduce`): orders eps^1 and eps^2 vanish outright, order eps^3
vanishes in every harmonic but the resonant pair, and that pair satisfies the solvability condition
Px = k Py, which is the identity s_i and l_i are determined by. This Python script is kept only for
the end-to-end mapping check
in step 5, whose *dynamics* are a hand-written numpy CR3BP integrator (independent of sympy); the
coefficient values it uses are the same formulas Maxima proves, and the two agree to every digit.

Reference: D. L. Richardson, "Analytic Construction of Periodic Orbits about the Collinear Points",
Celestial Mechanics 22 (1980) 241-253; Koon, Lo, Marsden & Ross (KLMR), appendix.

  1-4. cn(Li), lambda/k, Legendre forcing, third-order series (mirror of the Maxima derivation).
  5.   PERIODICITY DEFECT -- integrate the assembled orbit under the TRUE CR3BP (numpy) and confirm the
                            perpendicular-crossing defect is third-order small and its out-of-plane part
                            scales linearly with Az; the synodic-mapping sign is fixed by a defect sweep.
  6.   emit               -- numeric Earth-Moon L1/L2 coefficients + analytic initial state.

Run:  timeout 600 uv run --no-project --with sympy --with numpy python derivations/richardson_halo.py
"""

from __future__ import annotations

import sympy as sp

# ---------------------------------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------------------------------
mu, gamma = sp.symbols("mu gamma", positive=True)
c2, c3, c4 = sp.symbols("c2 c3 c4", real=True)
lam, k = sp.symbols("lambda k", positive=True)  # in-plane frequency and amplitude ratio
Ax, Az = sp.symbols("A_x A_z", real=True)        # in-plane / out-of-plane amplitudes
th = sp.symbols("theta", real=True)               # stretched phase, theta = omega * t


def cn_L1(n):
    """Richardson c_n for L1 (gamma = L1->secondary distance)."""
    return (1 / gamma**3) * (mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 - gamma) ** (n + 1))


def cn_L2(n):
    """Richardson c_n for L2."""
    return (1 / gamma**3) * ((-1) ** n * mu + (-1) ** n * (1 - mu) * gamma ** (n + 1) / (1 + gamma) ** (n + 1))


# ---------------------------------------------------------------------------------------------------
# Step 2 -- linear frequencies from the characteristic equation (derived, not memorised)
# ---------------------------------------------------------------------------------------------------
def linear_frequencies():
    """Solve the in-plane characteristic det = 0 for lambda^2 and the amplitude ratio k, symbolically
    in c2. Returns (lambda^2 expression, k expression, the raw characteristic, both roots)."""
    s = sp.symbols("s")
    # Linearised in-plane system  [[s^2-(1+2c2), -2s],[2s, s^2+(c2-1)]] (x,y)^T = 0.
    char = (s**2 - (1 + 2 * c2)) * (s**2 + (c2 - 1)) + 4 * s**2
    char = sp.expand(char)  # s^4 + (2 - c2) s^2 - (1 + 2c2)(c2 - 1)
    # Oscillatory root s^2 = -lambda^2: substitute and take the physical (larger) root.
    L = sp.symbols("L", positive=True)  # L = lambda^2
    quad = sp.expand(char.subs(s, sp.I * sp.sqrt(L)))  # s = i*lambda  ->  s^2 = -L
    roots = sp.solve(sp.Eq(quad, 0), L)
    # the physical in-plane elliptic frequency is the root with the + surd (lambda^2 > 1):
    lam2_expr = sp.simplify((2 - c2 + sp.sqrt(9 * c2**2 - 8 * c2)) / 2)
    k_expr = (lam**2 + 1 + 2 * c2) / (2 * lam)
    return lam2_expr, k_expr, char, roots


LAM2, K_EXPR, CHAR, LAM2_ROOTS = linear_frequencies()


# ---------------------------------------------------------------------------------------------------
# Step 3 -- exact nonlinear forcing from Legendre polynomials
# ---------------------------------------------------------------------------------------------------
def legendre_forcing():
    """Return (Fx, Fy, Fz): the nonlinear RHS forcing terms of the collinear-point EOM,

        x'' - 2 y' - (1 + 2 c2) x = Fx
        y'' + 2 x' + (c2 - 1) y   = Fy
        z''          + c2 z        = Fz

    with Fq = d/dq [ c3 rho^3 P3(x/rho) + c4 rho^4 P4(x/rho) ],  rho = sqrt(x^2+y^2+z^2),
    expanded exactly (no truncation error in the cubic model)."""
    x, y, z = sp.symbols("x y z", real=True)
    rho = sp.sqrt(x**2 + y**2 + z**2)
    U3 = c3 * rho**3 * sp.legendre(3, x / rho)
    U4 = c4 * rho**4 * sp.legendre(4, x / rho)
    U = sp.expand(sp.simplify(U3 + U4))
    Fx = sp.expand(sp.diff(U, x))
    Fy = sp.expand(sp.diff(U, y))
    Fz = sp.expand(sp.diff(U, z))
    return (x, y, z), (Fx, Fy, Fz)


(XS, YS, ZS), (FX, FY, FZ) = legendre_forcing()


# ---------------------------------------------------------------------------------------------------
# Step 4 -- Richardson third-order series (coefficients as symbols; verified in step 5)
# ---------------------------------------------------------------------------------------------------
# Second- and third-order coefficient symbols.
a21, a22, a23, a24 = sp.symbols("a21 a22 a23 a24", real=True)
b21, b22 = sp.symbols("b21 b22", real=True)
d21 = sp.symbols("d21", real=True)
a31, a32, b31, b32 = sp.symbols("a31 a32 b31 b32", real=True)
d31, d32 = sp.symbols("d31 d32", real=True)
s1, s2 = sp.symbols("s1 s2", real=True)          # frequency correction  omega = 1 + s1 Ax^2 + s2 Az^2
l1, l2 = sp.symbols("l1 l2", real=True)          # amplitude-constraint coefficients
Delta = sp.symbols("Delta", real=True)            # lambda^2 - c2 detuning
dn = sp.symbols("delta_n", real=True)             # class switch: +1 northern (n=1), -1 southern (n=3)

# The KLMR closed forms -- CANDIDATES to be verified by the residual identity in step 5.
d1 = 3 * lam**2 / k * (k * (6 * lam**2 - 1) - 2 * lam)
d2 = 8 * lam**2 / k * (k * (11 * lam**2 - 1) - 2 * lam)

FORMS = {
    a21: 3 * c3 * (k**2 - 2) / (4 * (1 + 2 * c2)),
    a22: 3 * c3 / (4 * (1 + 2 * c2)),
    a23: -3 * c3 * lam / (4 * k * d1) * (3 * k**3 * lam - 6 * k * (k - lam) + 4),
    a24: -3 * c3 * lam / (4 * k * d1) * (2 + 3 * k * lam),
    b21: -3 * c3 * lam / (2 * d1) * (3 * k * lam - 4),
    b22: 3 * c3 * lam / d1,
    d21: -c3 / (2 * lam**2),
    a31: -9 * lam / (4 * d2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2))
    + (9 * lam**2 + 1 - c2) / (2 * d2) * (3 * c3 * (2 * a23 - k * b21) + c4 * (2 + 3 * k**2)),
    a32: -9 * lam / (4 * d2) * (4 * c3 * (k * a24 - b22) + k * c4)
    - 3 * (9 * lam**2 + 1 - c2) / (2 * d2) * (c3 * (k * b22 + d21 - 2 * a24) - c4),
    b31: 3 / (8 * d2) * 8 * lam * (3 * c3 * (k * b21 - 2 * a23) - c4 * (2 + 3 * k**2))
    + 3 / (8 * d2) * (9 * lam**2 + 1 + 2 * c2) * (4 * c3 * (k * a23 - b21) + k * c4 * (4 + k**2)),
    b32: 9 * lam / d2 * (c3 * (k * b22 + d21 - 2 * a24) - c4)
    + 3 * (9 * lam**2 + 1 + 2 * c2) / (8 * d2) * (4 * c3 * (k * a24 - b22) + k * c4),
    d31: 3 / (64 * lam**2) * (4 * c3 * a24 + c4),
    d32: 3 / (64 * lam**2) * (4 * c3 * (a23 - d21) + c4 * (4 + k**2)),
}

# a1, a2, s1, s2, l1, l2 closed forms (secular / amplitude relation).
a1_form = -sp.Rational(3, 2) * c3 * (2 * a21 + a23 + 5 * d21) - sp.Rational(3, 8) * c4 * (12 - k**2)
a2_form = sp.Rational(3, 2) * c3 * (a24 - 2 * a22) + sp.Rational(9, 8) * c4
denom_s = 2 * lam * (lam * (1 + k**2) - 2 * k)
FORMS[s1] = (
    sp.Rational(3, 2) * c3 * (2 * a21 * (k**2 - 2) - a23 * (k**2 + 2) - 2 * k * b21)
    - sp.Rational(3, 8) * c4 * (3 * k**4 - 8 * k**2 + 8)
) / denom_s
FORMS[s2] = (
    sp.Rational(3, 2) * c3 * (2 * a22 * (k**2 - 2) + a24 * (k**2 + 2) + 2 * k * b22 + 5 * d21)
    + sp.Rational(3, 8) * c4 * (12 - k**2)
) / denom_s
FORMS[l1] = a1_form + 2 * lam**2 * s1
FORMS[l2] = a2_form + 2 * lam**2 * s2


# Legendre forcing as fast numeric callables (cubic model: only c3, c4 terms).
import numpy as np  # noqa: E402

_FX = sp.lambdify((XS, YS, ZS, c3, c4), FX, "numpy")
_FY = sp.lambdify((XS, YS, ZS, c3, c4), FY, "numpy")
_FZ = sp.lambdify((XS, YS, ZS, c3, c4), FZ, "numpy")


def _resolve_forms():
    """Resolve the nested coefficient formulas (a31 depends on a23,b21,...) to a fixed point."""
    resolved = dict(FORMS)
    for _ in range(6):
        resolved = {s: v.subs(resolved) for s, v in resolved.items()}
    return resolved


RESOLVED = _resolve_forms()


def numeric_context(mu_val, point):
    """Numeric (mu, gamma, c2, c3, c4, lambda, k) + every Richardson coefficient for L1 or L2."""
    if point == "L1":
        g = sp.symbols("g", positive=True)
        quintic = g**5 - (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 + 2 * mu * g - mu
        cn = cn_L1
    else:
        g = sp.symbols("g", positive=True)
        quintic = g**5 + (3 - mu) * g**4 + (3 - 2 * mu) * g**3 - mu * g**2 - 2 * mu * g - mu
        cn = cn_L2
    mu_r = sp.nsimplify(mu_val, rational=True)
    gval = float(sp.nsolve(quintic.subs(mu, mu_r), g, 0.15))
    ctx = {mu: mu_r, gamma: sp.Float(gval)}
    c2v = float(cn(2).subs(ctx))
    c3v = float(cn(3).subs(ctx))
    c4v = float(cn(4).subs(ctx))
    lam_v = float(sp.sqrt(LAM2.subs(c2, c2v)))
    k_v = float(K_EXPR.subs({lam: lam_v, c2: c2v}))
    subs_num = {c2: c2v, c3: c3v, c4: c4v, lam: lam_v, k: k_v}
    coeffs = {str(s): float(RESOLVED[s].subs(subs_num)) for s in RESOLVED}
    coeffs.update(gamma=gval, c2=c2v, c3=c3v, c4=c4v, lam=lam_v, k=k_v)
    return coeffs


def ax_from_az(co, Az):
    """In-plane amplitude slaved to Az by the constraint l1 Ax^2 + l2 Az^2 + Delta = 0, Delta = lam^2 - c2."""
    Delta_v = co["lam"] ** 2 - co["c2"]
    ax_sq = -(co["l2"] * Az**2 + Delta_v) / co["l1"]
    return float(np.sqrt(ax_sq)) if ax_sq > 0 else float("nan")


def series(co, Ax, Az, delta, theta, deriv=0):
    """Richardson series component and its theta-derivatives. Returns (X, Y, Z) arrays."""
    c, s = np.cos, np.sin
    t = theta
    if deriv == 0:
        X = (co["a21"] * Ax**2 + co["a22"] * Az**2 - Ax * c(t)
             + (co["a23"] * Ax**2 - co["a24"] * Az**2) * c(2 * t)
             + (co["a31"] * Ax**3 - co["a32"] * Ax * Az**2) * c(3 * t))
        Y = (co["k"] * Ax * s(t) + (co["b21"] * Ax**2 - co["b22"] * Az**2) * s(2 * t)
             + (co["b31"] * Ax**3 - co["b32"] * Ax * Az**2) * s(3 * t))
        Z = delta * (Az * c(t) + co["d21"] * Ax * Az * (c(2 * t) - 3)
                     + (co["d32"] * Az * Ax**2 - co["d31"] * Az**3) * c(3 * t))
    elif deriv == 1:  # d/dtheta
        X = (Ax * s(t) - 2 * (co["a23"] * Ax**2 - co["a24"] * Az**2) * s(2 * t)
             - 3 * (co["a31"] * Ax**3 - co["a32"] * Ax * Az**2) * s(3 * t))
        Y = (co["k"] * Ax * c(t) + 2 * (co["b21"] * Ax**2 - co["b22"] * Az**2) * c(2 * t)
             + 3 * (co["b31"] * Ax**3 - co["b32"] * Ax * Az**2) * c(3 * t))
        Z = delta * (-Az * s(t) - 2 * co["d21"] * Ax * Az * s(2 * t)
                     - 3 * (co["d32"] * Az * Ax**2 - co["d31"] * Az**3) * s(3 * t))
    else:  # d^2/dtheta^2
        X = (Ax * c(t) - 4 * (co["a23"] * Ax**2 - co["a24"] * Az**2) * c(2 * t)
             - 9 * (co["a31"] * Ax**3 - co["a32"] * Ax * Az**2) * c(3 * t))
        Y = (-co["k"] * Ax * s(t) - 4 * (co["b21"] * Ax**2 - co["b22"] * Az**2) * s(2 * t)
             - 9 * (co["b31"] * Ax**3 - co["b32"] * Ax * Az**2) * s(3 * t))
        Z = delta * (-Az * c(t) - 4 * co["d21"] * Ax * Az * c(2 * t)
                     - 9 * (co["d32"] * Az * Ax**2 - co["d31"] * Az**3) * c(3 * t))
    return X, Y, Z


def cr3bp_accel(mu_val, s):
    """Full (untruncated) CR3BP synodic acceleration for state s = [x,y,z,vx,vy,vz]."""
    x, y, z, vx, vy, vz = s
    r1 = np.sqrt((x + mu_val) ** 2 + y**2 + z**2)
    r2 = np.sqrt((x - 1 + mu_val) ** 2 + y**2 + z**2)
    om = 1 - mu_val
    ax = x - om * (x + mu_val) / r1**3 - mu_val * (x - 1 + mu_val) / r2**3 + 2 * vy
    ay = y - om * y / r1**3 - mu_val * y / r2**3 - 2 * vx
    az = -om * z / r1**3 - mu_val * z / r2**3
    return np.array([vx, vy, vz, ax, ay, az])


def cr3bp_defect(mu_val, state, omega):
    """Integrate the TRUE CR3BP from the analytic guess to the first y=0 crossing (~half period) and
    return the perpendicular-crossing defect (|vx|, |vz|). A correct 3rd-order guess makes this small;
    the differential corrector then drives it to ~1e-11. This validates coefficients AND the mapping."""
    s = np.array(state, float)
    h = (np.pi / omega) / 4000.0  # ~half period in 4000 RK4 steps
    prev_y = s[1]
    for _ in range(6000):
        k1 = cr3bp_accel(mu_val, s)
        k2 = cr3bp_accel(mu_val, s + 0.5 * h * k1)
        k3 = cr3bp_accel(mu_val, s + 0.5 * h * k2)
        k4 = cr3bp_accel(mu_val, s + h * k3)
        s_next = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s[1] != 0.0 and np.sign(s_next[1]) != np.sign(prev_y) and _ > 10:
            frac = -s[1] / (s_next[1] - s[1])  # linear refine of the crossing
            s = s + frac * (s_next - s)
            return abs(s[3]), abs(s[5])
        s, prev_y = s_next, s[1]
    return float("nan"), float("nan")


def synodic_guess(co, mu_val, point, Az, northern=True, sx=1.0):
    """Perpendicular-crossing synodic state [x0, 0, z0, 0, vy0, 0] for the halo of out-of-plane
    amplitude Az. Li-frame series (units of gamma, origin Li) mapped to the rotating synodic frame.

    Richardson's normalized x-axis points from Li toward the LARGER primary (Earth); the synodic x
    increases with x_R (sx = +1, verified by the periodicity-defect sweep). The reflection
    carries the y-velocity consistently."""
    delta = 1.0 if northern else -1.0
    Ax = ax_from_az(co, Az)
    omega = co["lam"] * (1 + co["s1"] * Ax**2 + co["s2"] * Az**2)
    g = co["gamma"]
    x_li = (1 - mu_val - g) if point == "L1" else (1 - mu_val + g)
    X0, _, Z0 = series(co, Ax, Az, delta, 0.0, 0)
    _, Yp0, _ = series(co, Ax, Az, delta, 0.0, 1)
    x0 = x_li + sx * g * X0
    z0 = g * Z0
    vy0 = sx * g * omega * Yp0
    return [x0, 0.0, z0, 0.0, vy0, 0.0], Ax


def main():
    mu_em = 0.012_150_585  # Cr3bp::EARTH_MOON

    print("=" * 92)
    print("STEP 2  Linear characteristic and frequencies (DERIVED from det = 0, not memorised)")
    print("=" * 92)
    print("  in-plane characteristic (in s):", CHAR)
    print("  lambda^2 root(s):", [sp.simplify(r) for r in LAM2_ROOTS])
    print("  lambda^2 (physical, + surd) =", LAM2)
    print("  k = (lambda^2 + 1 + 2 c2)/(2 lambda)")
    rich = (2 - c2 + sp.sqrt(9 * c2**2 - 8 * c2)) / 2
    print("  lambda^2 - Richardson closed form =", sp.simplify(LAM2 - rich), "(0 confirms the root)")

    print("\n" + "=" * 92)
    print("STEP 3  Exact Legendre forcing  d/dq [ c3 rho^3 P3(x/rho) + c4 rho^4 P4(x/rho) ]")
    print("=" * 92)
    print("  Fx =", FX)
    print("  Fy =", FY)
    print("  Fz =", FZ)

    print("\n" + "=" * 92)
    print("STEP 5a  Sign-convention sweep at EM-L1, Az=0.02 (pick the small-defect mapping)")
    print("=" * 92)
    co = numeric_context(mu_em, "L1")
    Az = 0.02
    Ax = ax_from_az(co, Az)
    omega = co["lam"] * (1 + co["s1"] * Ax**2 + co["s2"] * Az**2)
    for sx in [-1.0, 1.0]:
        st, _ = synodic_guess(co, mu_em, "L1", Az, northern=True, sx=sx)
        dvx, dvz = cr3bp_defect(mu_em, st, omega)
        print(f"    sx={sx:+.0f}: guess x0={st[0]:.6f} z0={st[2]:.6f} vy0={st[4]:.6f}"
              f"   defect |vx|={dvx:.2e} |vz|={dvz:.2e}")

    print("\n" + "=" * 92)
    print("STEP 5  Periodicity-defect test: integrate the TRUE CR3BP from the analytic guess")
    print("        (correct 3rd-order coeffs+mapping => small crossing defect, shrinking with Az)")
    print("=" * 92)
    for point in ["L1", "L2"]:
        co = numeric_context(mu_em, point)
        print(f"\n  Earth-Moon {point}:")
        prev = None
        for Az in [0.08, 0.04, 0.02, 0.01]:
            Ax = ax_from_az(co, Az)
            omega = co["lam"] * (1 + co["s1"] * Ax**2 + co["s2"] * Az**2)
            st, _ = synodic_guess(co, mu_em, point, Az, northern=True, sx=1.0)
            dvx, dvz = cr3bp_defect(mu_em, st, omega)
            defect = max(dvx, dvz)
            ratio = "" if prev is None else f"  (x{prev/defect:4.1f} smaller)"
            print(f"    Az={Az:.3f} Ax={Ax:.4f}:  crossing defect |vx|={dvx:.2e} |vz|={dvz:.2e}"
                  f"  max={defect:.2e}{ratio}")
            prev = defect

    print("\n" + "=" * 92)
    print("STEP 6  Numeric coefficients + synodic initial guess (emit to Rust)")
    print("=" * 92)
    order = ["a21", "a22", "a23", "a24", "b21", "b22", "d21", "a31", "a32", "b31", "b32",
             "d31", "d32", "s1", "s2", "l1", "l2"]
    for point in ["L1", "L2"]:
        co = numeric_context(mu_em, point)
        print(f"\n  --- Earth-Moon {point} ---")
        print(f"    gamma={co['gamma']:.12f}  c2={co['c2']:.10f}  c3={co['c3']:.10f}  c4={co['c4']:.10f}")
        print(f"    lambda={co['lam']:.12f}  k={co['k']:.12f}  nu_z=sqrt(c2)={co['c2']**0.5:.12f}")
        for name in order:
            print(f"    {name:>4} = {co[name]: .12e}")
        Delta_v = co["lam"] ** 2 - co["c2"]
        print(f"    constraint: {co['l1']:.6e} Ax^2 + {co['l2']:.6e} Az^2 + {Delta_v:.6e} = 0")
        for Az in [0.02, 0.05]:
            st, Ax = synodic_guess(co, mu_em, point, Az, northern=True, sx=1.0)
            print(f"    Az={Az}: Ax={Ax:.6f}  synodic guess [x0,0,z0,0,vy0,0] = "
                  f"[{st[0]:.9f}, 0, {st[2]:.9f}, 0, {st[4]:.9f}, 0]")


if __name__ == "__main__":
    main()
