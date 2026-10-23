"""Combined DSST mean propagation: J2..Jn zonal + Sun/Moon third-body + SRP.

Propagates a sun-synchronous LEO under the full conservative mean-element rates
(`dsst_propagate_mean_full`) and contrasts it with the zonal-only mean propagation
(`dsst_zonal_propagate_mean`), isolating the lunisolar + SRP contribution to the mean-element drift
over a week. Each summed contribution is individually bit-exact against Orekit; this shows them
composed into one semi-analytical propagator.

Run after `maturin develop` (or `uvx maturin develop`) has installed `spacelab`.
"""
import numpy as np
import spacedatalab as sp

# Sun-synchronous LEO, MEAN equinoctial state [a, k=ex, h=ey, q=hx, p=hy, lambda].
a0 = 7.2e6
init = [a0, 3.09016994374947470e-03, 9.51056516295153600e-03,
        8.54890329218965000e-01, 7.69746709859868000e-01, 1.43116998663535020e+00]
epoch = 0.0  # seconds past J2000 (2000-01-01 12:00 TT); the analytic Sun/Moon advance from here.
days = 7
times = list(np.linspace(0.0, days * 86400.0, days + 1))

full = sp.dsst_propagate_mean_full(init, times, epoch)
zonal = sp.dsst_zonal_propagate_mean(init, times)

print(f"{'day':>4} {'a_km':>12} {'e':>10} {'RAAN_deg':>10}  "
      f"{'Δe (lunisolar+SRP)':>20} {'Δi_vec':>12}")
for k, t in enumerate(times):
    af, kf, hf, qf, pf, lf = full[k]
    _, kz, hz, qz, pz, _ = zonal[k]
    ef = np.hypot(kf, hf)
    raan = np.degrees(np.arctan2(pf, qf))
    d_ecc = np.hypot(kf - kz, hf - hz)   # eccentricity-vector difference = third-body + SRP
    d_inc = np.hypot(qf - qz, pf - pz)   # inclination-vector difference
    print(f"{t / 86400:4.0f} {af / 1e3:12.3f} {ef:10.6f} {raan:10.4f}  {d_ecc:20.3e} {d_inc:12.3e}")

print("\nThe (e, i)-vector difference from the zonal-only run is the lunisolar + SRP perturbation "
      "that the combined propagator adds. a stays ~constant (all three conservative forces are "
      "secular-a-free on average).")
