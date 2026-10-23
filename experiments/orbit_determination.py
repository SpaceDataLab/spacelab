#!/usr/bin/env python
"""End-to-end orbit determination through the spacelab bridge — the estimation vertical slice.

Closes the loop that the unit tests only touch piecewise: a truth orbit is propagated with the dense
DOP853 integrator, synthetic range + range-rate tracking is generated from a ground network (with
noise and a horizon gate), and the epoch state is recovered from a deliberately wrong initial guess by
Gauss-Newton whose residual model re-propagates the trial state at every iterate. This is the case the
`gauss_newton` smoke test cannot exercise: the measurement Jacobian is dense and computed by finite
differences through the propagator, not an identity, so a wrong residual sign would diverge instead of
silently "converging" at the initial guess.

Simplification: stations are treated as fixed in the inertial (propagation) frame — this is an
estimation-convergence demo, not a frames demo, so Earth rotation of the station is intentionally
omitted. Visibility is still gated by a geometric inertial-horizon test.

Build the wheel first, then run with it on the path:
  cd crates/spacelab-python-harness && maturin build --release -o /tmp/w
  uv run --no-project --with /tmp/w/*.whl --with numpy python experiments/orbit_determination.py
"""

from __future__ import annotations

import math

import numpy as np
import spacedatalab as sl

MU = 3.986004418e14  # m^3/s^2, WGS-84 GM
J2 = 1.08262668e-3  # Earth second zonal harmonic
RE = 6_378_137.0  # m, equatorial radius
TOL = 1e-9  # dense-integration tolerance for both truth and residual re-propagations

SIGMA_RANGE = 10.0  # m
SIGMA_RATE = 0.05  # m/s
FD_STEP = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3])  # finite-difference deltas: m for r, m/s for v


def truth_state() -> np.ndarray:
    """A ~550 km, 53deg-inclined near-circular LEO epoch state (ECI, m and m/s)."""
    a = 6_928_137.0
    r0 = np.array([a, 0.0, 0.0])
    vc = math.sqrt(MU / a)
    inc = math.radians(53.0)
    v0 = np.array([0.0, vc * math.cos(inc), vc * math.sin(inc)])
    return np.concatenate([r0, v0])


def period(a: float) -> float:
    return 2.0 * math.pi * math.sqrt(a**3 / MU)


# Inertial-fixed tracking network spread around the globe (m). More, well-separated stations over a
# multi-orbit arc keep the epoch state geometrically observable — sparse tracking fits the noise but
# leaves the epoch under-constrained.
def _station(lon_deg: float, lat_deg: float) -> np.ndarray:
    lon, lat = math.radians(lon_deg), math.radians(lat_deg)
    return RE * np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


STATIONS = [
    _station(0.0, 5.0),
    _station(72.0, 35.0),
    _station(144.0, -20.0),
    _station(216.0, 55.0),
    _station(288.0, -40.0),
]


def visible(sat_pos: np.ndarray, station: np.ndarray) -> bool:
    """Satellite is above the station's local inertial horizon: (sat - station)·station > 0."""
    return float(np.dot(sat_pos - station, station)) > 0.0


def measure(state6: tuple[list[float], list[float]], station: np.ndarray, kind: str) -> float:
    r, v = state6
    sat = [r[0], r[1], r[2], v[0], v[1], v[2]]
    st = [station[0], station[1], station[2]]
    return sl.range(sat, st) if kind == "range" else sl.range_rate(sat, st)


def make_observations(truth: np.ndarray, arc: float, n_epochs: int, rng: np.random.Generator):
    """Propagate the truth orbit once and sample noisy range + range-rate where geometrically visible.

    Returns a list of (t, station, kind, value, sigma).
    """
    eph = sl.propagate_dense(list(truth[:3]), list(truth[3:]), arc, MU, J2, RE, TOL)
    epochs = np.linspace(0.0, arc, n_epochs)
    obs = []
    for t in epochs:
        r, v = eph.eval(float(t))
        sat_pos = np.array(r)
        for st in STATIONS:
            if not visible(sat_pos, st):
                continue
            for kind, sigma in (("range", SIGMA_RANGE), ("range_rate", SIGMA_RATE)):
                noisy = measure((r, v), st, kind) + rng.normal(0.0, sigma)
                obs.append((float(t), st, kind, noisy, sigma))
    return obs


def make_residual_model(obs, arc: float):
    """A Gauss-Newton residual model: residual = computed(x) - observed, h = d(residual)/dx by FD.

    One nominal propagation plus six perturbed propagations per iterate; each dense interpolant is
    evaluated at every observation epoch, so the whole Jacobian costs 7 integrations, not 7*N.
    """

    def predictions(x: np.ndarray) -> np.ndarray:
        eph = sl.propagate_dense(list(x[:3]), list(x[3:]), arc, MU, J2, RE, TOL)
        out = np.empty(len(obs))
        for k, (t, st, kind, _value, _sigma) in enumerate(obs):
            out[k] = measure(eph.eval(t), st, kind)
        return out

    def model(x_list):
        x = np.asarray(x_list, dtype=float)
        base = predictions(x)
        jac = np.empty((len(obs), 6))
        for j in range(6):
            xp = x.copy()
            xp[j] += FD_STEP[j]
            jac[:, j] = (predictions(xp) - base) / FD_STEP[j]
        rows = []
        for k, (_t, _st, _kind, value, sigma) in enumerate(obs):
            weight = 1.0 / (sigma * sigma)
            rows.append((jac[k].tolist(), float(base[k] - value), weight))
        return rows

    return model


def main() -> None:
    rng = np.random.default_rng(42)
    truth = truth_state()
    arc = 2.0 * period(float(np.linalg.norm(truth[:3])))
    obs = make_observations(truth, arc, n_epochs=120, rng=rng)
    print(f"observations: {len(obs)} range/range-rate rows over {arc:.0f} s from {len(STATIONS)} stations")

    model = make_residual_model(obs, arc)

    # At the truth state, weighted cost should sit near the number of observations (each term ~ 1).
    truth_rows = model(truth.tolist())
    truth_cost = sum((r[1] * math.sqrt(r[2])) ** 2 for r in truth_rows)
    print(f"weighted cost at truth: {truth_cost:.1f}  (~{len(obs)} expected)")

    guess = truth + np.array([1_000.0, -1_000.0, 1_000.0, 1.0, -1.0, 1.0])
    x_est, cov, converged, iters, cost = sl.gauss_newton(model, guess.tolist())
    x_est = np.asarray(x_est)
    sigma = np.sqrt(np.diag(np.asarray(cov)))  # formal 1σ uncertainties, [m×3, m/s×3]

    pos_err0 = np.linalg.norm(guess[:3] - truth[:3])
    vel_err0 = np.linalg.norm(guess[3:] - truth[3:])
    pos_err = np.linalg.norm(x_est[:3] - truth[:3])
    vel_err = np.linalg.norm(x_est[3:] - truth[3:])
    sigma_pos = np.linalg.norm(sigma[:3])
    sigma_vel = np.linalg.norm(sigma[3:])

    print(f"converged={converged} iters={iters} cost={cost:.1f}")
    print(f"epoch position error: {pos_err0:8.1f} m  ->  {pos_err:7.2f} m   (formal 1σ {sigma_pos:6.2f} m)")
    print(f"epoch velocity error: {vel_err0:8.4f} m/s -> {vel_err:8.5f} m/s (formal 1σ {sigma_vel:.5f} m/s)")

    assert converged, "Gauss-Newton did not converge"
    assert pos_err < pos_err0 / 10.0, "position error not reduced by at least 10x"
    assert vel_err < vel_err0 / 10.0, "velocity error not reduced by at least 10x"
    # Consistency: the actual error must sit within a few formal sigma — an estimate whose true error
    # exceeds its own covariance is overconfident (a worse failure than a large-but-honest error bar).
    assert pos_err < 3.0 * sigma_pos, f"position error {pos_err:.2f} m exceeds 3σ ({3 * sigma_pos:.2f} m)"
    assert vel_err < 3.0 * sigma_vel, f"velocity error {vel_err:.5f} m/s exceeds 3σ"
    # A good fit sits near the noise floor: weighted cost ~ number of observations, not orders above it.
    assert cost < 5.0 * len(obs), f"post-fit cost {cost:.1f} far above the noise floor {len(obs)}"
    print("orbit determination recovered the epoch state, consistent with its formal covariance")


if __name__ == "__main__":
    main()
