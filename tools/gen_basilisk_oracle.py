#!/usr/bin/env python3
"""Basilisk as the independent implementation behind `spacelab-control`'s attitude loop.

Every other pillar in this workspace is checked against somebody else's code; the control pillar had
nobody, because Orekit has no closed-loop control. Basilisk (AVS Lab, University of Colorado at
Boulder) does, and its `mrpFeedback` module is the same Schaub-Junkins law `spacelab_control`'s
`MrpLyapunov` implements -- which is what makes this a like-for-like comparison rather than two
programs solving adjacent problems.

**Licence, confirmed from the repository and not from memory.** `AVSLab/basilisk`'s `LICENSE` is the
**ISC License**, "Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at
Boulder", granting permission "to use, copy, modify, and/or distribute this software for any purpose
with or without fee ... provided that the above copyright notice and this permission notice appear in
all copies". That is permissive: the numbers below could be redistributed and so could code, given
the notice. It changes nothing about how the mathematics is written -- the Rust is implemented from
Schaub & Junkins with a citation, as it was before this file existed, and Basilisk is read only to
settle conventions that a paper leaves ambiguous.

Basilisk is an **oracle, never a dependency**: `uv run --with bsk` builds an ephemeral environment,
so nothing reaches `uv.lock`, and the workspace stays dependency-free at runtime.

**Three legs, because there are three separable claims.**

  * `basilisk_mrp_feedback.txt` -- the control law alone, at sampled states, with no integrator
    anywhere. A sign or a factor shows here with nothing to hide behind. Both of Basilisk's
    `controlLawType` variants are exported, because they are *different laws* and the default is not
    ours (see below).
  * `basilisk_slew.txt` -- the closed loop, torque-free, at four step sizes. Basilisk integrates
    `(sigma_BN, omega_BN)` with RK4 and a zero-order-held command; spacelab integrates the same ODE
    with the RKMK4 Lie-group step directly on `SO(3)`. Two different integrators on one system, so
    the residual is expected to *fall with the step* rather than to vanish, and the order it falls at
    is the check.
  * `basilisk_gravity_gradient.txt` -- one environmental torque, at sampled attitudes and positions.
    `3 mu / r^5 * r_B x ([I] r_B)` is short enough to get wrong in exactly one place.

**The finding that made the first leg worth writing.** Basilisk's `mrpFeedback` has a
`controlLawType` switch, and the default (`0`) is *not* the law with full gyroscopic cancellation. It
crosses `omega_RN + K_i z` into `[I] omega_BN`, where `1` crosses `omega_BN`. For regulation to an
inertially fixed reference `omega_RN` is zero, so the default contributes **no gyroscopic term at
all** while `spacelab`'s law contributes `omega x [I] omega`. Comparing against the default would
have shown a torque discrepancy growing with the square of the rate and read as a spacelab defect.
Both are exported so the Rust test measures that gap rather than asserting it away.

    uv run --with bsk python tools/gen_basilisk_oracle.py
"""

from __future__ import annotations

import pathlib

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.simulation import GravityGradientEffector, extForceTorque, spacecraft
from Basilisk.utilities import SimulationBaseClass, macros, simIncludeGravBody

FIXTURES = (
    pathlib.Path(__file__).resolve().parents[1]
    / "crates/spacelab-control/tests/fixtures"
)

# A small asymmetric spacecraft: no two principal moments equal, so a wrong axis cannot hide behind a
# symmetry. Gains are Basilisk's own scenario values, which keeps the closed loop well damped over
# the span sampled here rather than tuned to make the residual look good.
INERTIA = np.diag([10.0, 12.0, 8.0])
K_GAIN, P_GAIN = 0.35, 4.0

# Sampled control states. `omega_RN` is zero in the first block and non-zero in the second, because
# the two exercise different terms: with a fixed reference the laws must agree exactly, and with a
# moving one they must differ by exactly the tracking feedforward `[I](domega_RN - omega x omega_RN)`
# that `MrpLyapunov`'s doc comment says it omits. A claim in a comment is worth measuring.
LAW_CASES = [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.1, -0.2, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.0, 0.0, 0.0], [0.01, -0.02, 0.03], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.3, 0.1, -0.25], [0.02, 0.01, -0.03], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([-0.41, 0.33, 0.19], [0.11, -0.07, 0.13], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.9, -0.15, 0.4], [-0.3, 0.25, 0.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.05, 0.05, 0.05], [0.001, 0.002, -0.001], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([0.3, 0.1, -0.25], [0.02, 0.01, -0.03], [0.001, 0.0, -0.002], [0.0, 0.0, 0.0]),
    (
        [-0.2, 0.4, 0.1],
        [0.05, -0.05, 0.02],
        [0.0011, -0.0004, 0.0009],
        [1e-5, 2e-5, -3e-5],
    ),
    (
        [0.12, -0.34, 0.56],
        [-0.02, 0.04, -0.06],
        [-0.002, 0.001, 0.0015],
        [-4e-5, 0.0, 1e-5],
    ),
]

# The slew: a 2.09 rad (120 deg) initial error about a tilted axis, with an initial rate that is not
# aligned with it, so all three body axes carry load and the gyroscopic term is not decorative.
SLEW_SIGMA0 = [0.35, -0.20, 0.15]
SLEW_OMEGA0 = [0.03, 0.02, -0.04]
SLEW_SPAN = 120.0
SLEW_STEPS = (1.0, 0.5, 0.25, 0.125)
SLEW_SAMPLE = 10.0

MU_EARTH = 3.986004415e14
GG_CASES = [
    ([7000.0e3, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ([7000.0e3, 0.0, 0.0], [0.2, 0.0, 0.0]),
    ([0.0, 6800.0e3, 1200.0e3], [0.1, -0.2, 0.3]),
    ([-4200.0e3, 3100.0e3, 5000.0e3], [-0.35, 0.15, 0.45]),
    ([2000.0e3, -2000.0e3, 41000.0e3], [0.05, 0.05, -0.05]),
]


def mrp_to_dcm(sigma: list[float]) -> np.ndarray:
    """`[BN]` from the MRP `sigma_BN`, Schaub & Junkins eq. (3.147)."""
    s = np.asarray(sigma, dtype=float)
    s2 = float(s @ s)
    tilde = np.array([[0.0, -s[2], s[1]], [s[2], 0.0, -s[0]], [-s[1], s[0], 0.0]])
    return (
        np.eye(3) + (8.0 * tilde @ tilde - 4.0 * (1.0 - s2) * tilde) / (1.0 + s2) ** 2
    )


def law_torques() -> list[tuple[int, int, np.ndarray]]:
    """One `mrpFeedback` instance per (law, case), all stepped once in a single task."""
    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("law")
    dt = macros.sec2nano(1.0)
    proc.addTask(sim.CreateNewTask("t", dt))

    keep, tagged = [], []
    for law in (0, 1):
        for index, (sigma, w_br, w_rn, dw_rn) in enumerate(LAW_CASES):
            module = mrpFeedback.mrpFeedback()
            module.ModelTag = f"law{law}case{index}"
            module.K, module.P = K_GAIN, P_GAIN
            module.Ki = (
                -1.0
            )  # a negative gain is Basilisk's own switch for "no integral feedback"
            module.integralLimit = 0.0
            module.knownTorquePntB_B = [0.0, 0.0, 0.0]
            module.controlLawType = law

            vehicle = messaging.VehicleConfigMsgPayload()
            vehicle.ISCPntB_B = list(INERTIA.flatten())
            veh_msg = messaging.VehicleConfigMsg().write(vehicle)
            module.vehConfigInMsg.subscribeTo(veh_msg)

            guidance = messaging.AttGuidMsgPayload()
            guidance.sigma_BR = list(sigma)
            guidance.omega_BR_B = list(w_br)
            guidance.omega_RN_B = list(w_rn)
            guidance.domega_RN_B = list(dw_rn)
            guid_msg = messaging.AttGuidMsg().write(guidance)
            module.guidInMsg.subscribeTo(guid_msg)

            recorder = module.cmdTorqueOutMsg.recorder()
            sim.AddModelToTask("t", module)
            sim.AddModelToTask("t", recorder)
            # The Python wrappers own the message buffers; without a live reference the module reads
            # freed memory and returns 1e197 rather than failing. Measured, not guessed.
            keep.extend((veh_msg, guid_msg))
            tagged.append((law, index, recorder))

    sim.InitializeSimulation()
    sim.ConfigureStopTime(dt)
    sim.ExecuteSimulation()
    return [
        (law, index, np.array(rec.torqueRequestBody)[-1]) for law, index, rec in tagged
    ]


def slew(
    step: float, *, actuator_first: bool
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Torque-free closed loop: `mrpFeedback` (law 1) commanding an ideal body torque.

    No gravity body is attached, so the only moment is the commanded one and the residual against
    spacelab is an integrator difference and nothing else. The reference is inertially fixed at the
    identity and reaches the controller through Basilisk's own `inertial3D` and `attTrackingError`,
    so no hand-written error algebra enters on this side of the comparison.

    `actuator_first` decides where `ExtForceTorque` sits in the task's execution order, and that one
    choice moves the answer by five orders of magnitude. The effector latches the command message
    into the field the hub's equations of motion read; the hub integrates `[t_{k-1}, t_k]`. Ordered
    **after** the hub the effector latches at step `k` a command the controller wrote at step `k-1`,
    so the hub applies it over the *next* interval and the whole loop runs one step late, with the
    first interval torque-free because the message is still zero-initialised. Ordered **before** the
    hub the same command drives its own interval and the loop is exact. Measured, not read off the
    source: modelling the lag takes the residual against spacelab from 6.7e-3 rad to 8.1e-8 rad and
    the convergence from first order to fourth. Both orderings are emitted so the test can show it.
    """
    from Basilisk.fswAlgorithms import attTrackingError, inertial3D
    from Basilisk.simulation import simpleNav

    sim = SimulationBaseClass.SimBaseClass()
    proc = sim.CreateNewProcess("slew")
    dt = macros.sec2nano(step)
    proc.addTask(sim.CreateNewTask("t", dt))

    hub = spacecraft.Spacecraft()
    hub.ModelTag = "hub"
    hub.hub.mHub = 750.0
    hub.hub.IHubPntBc_B = [list(row) for row in INERTIA]
    hub.hub.sigma_BNInit = [[v] for v in SLEW_SIGMA0]
    hub.hub.omega_BN_BInit = [[v] for v in SLEW_OMEGA0]
    sim.AddModelToTask("t", hub, 90)

    actuator = extForceTorque.ExtForceTorque()
    actuator.ModelTag = "torque"
    hub.addDynamicEffector(actuator)
    sim.AddModelToTask("t", actuator, 100 if actuator_first else 80)

    nav = simpleNav.SimpleNav()
    nav.ModelTag = "nav"
    nav.scStateInMsg.subscribeTo(hub.scStateOutMsg)
    sim.AddModelToTask("t", nav, 70)

    reference = inertial3D.inertial3D()
    reference.ModelTag = "reference"
    reference.sigma_R0N = [0.0, 0.0, 0.0]
    sim.AddModelToTask("t", reference, 60)

    error = attTrackingError.attTrackingError()
    error.ModelTag = "error"
    error.sigma_R0R = [0.0, 0.0, 0.0]
    error.attRefInMsg.subscribeTo(reference.attRefOutMsg)
    error.attNavInMsg.subscribeTo(nav.attOutMsg)
    sim.AddModelToTask("t", error, 50)

    control = mrpFeedback.mrpFeedback()
    control.ModelTag = "control"
    control.K, control.P = K_GAIN, P_GAIN
    control.Ki = -1.0
    control.integralLimit = 0.0
    control.knownTorquePntB_B = [0.0, 0.0, 0.0]
    control.controlLawType = 1

    vehicle = messaging.VehicleConfigMsgPayload()
    vehicle.ISCPntB_B = list(INERTIA.flatten())
    veh_msg = messaging.VehicleConfigMsg().write(vehicle)
    control.vehConfigInMsg.subscribeTo(veh_msg)
    control.guidInMsg.subscribeTo(error.attGuidOutMsg)
    actuator.cmdTorqueInMsg.subscribeTo(control.cmdTorqueOutMsg)
    sim.AddModelToTask("t", control, 40)

    recorder = hub.scStateOutMsg.recorder(macros.sec2nano(SLEW_SAMPLE))
    sim.AddModelToTask("t", recorder, 10)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(SLEW_SPAN))
    sim.ExecuteSimulation()

    # `veh_msg` is referenced here so the Python wrapper that owns its buffer outlives the run: a
    # collected message leaves the module reading freed memory and returning 1e197 rather than
    # failing. Measured while writing this file, not guessed.
    assert veh_msg is not None
    times = np.array(recorder.times()) * macros.NANO2SEC
    return [
        (float(t), np.array(s), np.array(w))
        for t, s, w in zip(times, recorder.sigma_BN, recorder.omega_BN_B, strict=True)
    ]


def gravity_gradients() -> list[tuple[int, np.ndarray]]:
    """`GravityGradientEffector`'s torque at sampled positions and attitudes.

    The effector reads the hub's own state, so each case gets its own one-step simulation rather than
    a shared one: the point is the algebraic torque, not a trajectory.
    """
    out = []
    for index, (r_n, sigma) in enumerate(GG_CASES):
        sim = SimulationBaseClass.SimBaseClass()
        proc = sim.CreateNewProcess(f"gg{index}")
        dt = macros.sec2nano(1.0)
        proc.addTask(sim.CreateNewTask("t", dt))

        hub = spacecraft.Spacecraft()
        hub.ModelTag = "hub"
        hub.hub.mHub = 750.0
        hub.hub.IHubPntBc_B = [list(row) for row in INERTIA]
        hub.hub.r_CN_NInit = [[v] for v in r_n]
        hub.hub.v_CN_NInit = [[0.0], [0.0], [0.0]]
        hub.hub.sigma_BNInit = [[v] for v in sigma]
        hub.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

        gravity = simIncludeGravBody.gravBodyFactory()
        earth = gravity.createEarth()
        earth.isCentralBody = True
        earth.mu = MU_EARTH
        gravity.addBodiesTo(hub)

        effector = GravityGradientEffector.GravityGradientEffector()
        effector.ModelTag = "gg"
        effector.addPlanetName(earth.planetName)
        hub.addDynamicEffector(effector)

        sim.AddModelToTask("t", hub)
        sim.AddModelToTask("t", effector)
        recorder = effector.gravityGradientOutMsg.recorder()
        sim.AddModelToTask("t", recorder)

        sim.InitializeSimulation()
        sim.ConfigureStopTime(dt)
        sim.ExecuteSimulation()
        out.append((index, np.array(recorder.gravityGradientTorque_B)[0]))
    return out


HEADER = (
    "# Basilisk {version} (AVSLab/basilisk, ISC License, Copyright (c) 2016 Autonomous Vehicle\n"
    "# Systems Lab, University of Colorado at Boulder). Oracle only -- never a dependency.\n"
    "# Generated by tools/gen_basilisk_oracle.py -- do not edit.\n"
)


def basilisk_version() -> str:
    import Basilisk

    return getattr(Basilisk, "__version__", "unknown")


def vec(v) -> str:
    return " ".join(f"{x:+.15e}" for x in v)


def write_law(rows) -> None:
    lines = [
        HEADER.format(version=basilisk_version()),
        f"# inertia diag {vec(np.diag(INERTIA))}  K {K_GAIN:.15e}  P {P_GAIN:.15e}\n",
        "#\n",
        "# controlLawType 0 crosses (omega_RN + Ki z) into [I]omega_BN; 1 crosses omega_BN, and 1 is\n",
        "# the law spacelab implements. Ki < 0 disables the integral term, so z is zero throughout.\n",
        "#\n",
        "# law case sigma_BR(3) omega_BR_B(3) omega_RN_B(3) domega_RN_B(3) torqueRequestBody(3)\n",
    ]
    for law, index, torque in rows:
        sigma, w_br, w_rn, dw_rn = LAW_CASES[index]
        lines.append(
            f"{law} {index:2d} {vec(sigma)} {vec(w_br)} {vec(w_rn)} {vec(dw_rn)} {vec(torque)}\n"
        )
    (FIXTURES / "basilisk_mrp_feedback.txt").write_text("".join(lines))


def write_slew(runs) -> None:
    lines = [
        HEADER.format(version=basilisk_version()),
        f"# inertia diag {vec(np.diag(INERTIA))}  K {K_GAIN:.15e}  P {P_GAIN:.15e}\n",
        f"# sigma_BN(0) {vec(SLEW_SIGMA0)}  omega_BN(0) {vec(SLEW_OMEGA0)}\n",
        "# Torque-free apart from the commanded moment; reference inertially fixed at the identity;\n",
        "# controlLawType 1; command held over each step (Basilisk runs the FSW task once a step).\n",
        "#\n",
        "# lag 0: ExtForceTorque ordered before the hub, so a command drives its own interval.\n",
        "# lag 1: ordered after it, so every command drives the *next* interval and the first is\n",
        "#        torque-free -- Basilisk's default insertion order, and worth five orders of\n",
        "#        magnitude in the residual. See slew() for the mechanism.\n",
        "#\n",
        "# lag step_s t_s sigma_BN(3) omega_BN_B(3)\n",
    ]
    for lag, step, samples in runs:
        for t, sigma, omega in samples:
            lines.append(f"{lag:d} {step:.6f} {t:12.4f} {vec(sigma)} {vec(omega)}\n")
    (FIXTURES / "basilisk_slew.txt").write_text("".join(lines))


def write_gravity_gradient(rows) -> None:
    lines = [
        HEADER.format(version=basilisk_version()),
        f"# inertia diag {vec(np.diag(INERTIA))}  mu {MU_EARTH:.15e} m^3/s^2\n",
        "#\n",
        "# case r_BN_N(3) sigma_BN(3) torque_B(3)\n",
    ]
    for index, torque in rows:
        r_n, sigma = GG_CASES[index]
        lines.append(f"{index:2d} {vec(r_n)} {vec(sigma)} {vec(torque)}\n")
    (FIXTURES / "basilisk_gravity_gradient.txt").write_text("".join(lines))


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    write_law(law_torques())
    write_slew(
        [
            (lag, step, slew(step, actuator_first=lag == 0))
            for lag in (0, 1)
            for step in SLEW_STEPS
        ]
    )
    write_gravity_gradient(gravity_gradients())
    print(f"wrote three fixtures to {FIXTURES}")


if __name__ == "__main__":
    main()
