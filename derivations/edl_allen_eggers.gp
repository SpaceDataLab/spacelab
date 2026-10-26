/* Second engine for derivations/edl_allen_eggers.mac: the Allen-Eggers peak of the vertical
   ballistic entry the gate in crates/spacelab-astro/src/edl.rs flies, at 100 digits.

   Mars atmosphere (rho0 = 0.020 kg/m^3, H = 11 100 m, R = 3 389 500 m) with mu = 0, a ballistic
   vehicle of beta = 100 kg/m^2, entering vertically (s = -sin(gamma) = 1 for the double nearest
   -pi/2, whose sine rounds to -1) at 6000 m/s from 120 km. Every input is taken as the double the
   test passes. Printed: h*, V*, D* in m/s^2 and in Earth-g (9.80665 m/s^2), and the speed at the
   surface, where the run stops. The closed forms are checked against a direct numerical
   integration of dV/dh = rho V/(2 beta s) (FAIL above 1e-60 relative), so the two routes agree
   independently of the algebra.

       timeout 300 gp -q -f derivations/edl_allen_eggers.gp </dev/null                           */

default(realprecision, 100);
dbl(x) = if (x == 0, 0, my(e = exponent(x)); round(x * 2^(52 - e)) / 2^(52 - e));
rho0 = dbl(0.020); H = 11100; beta = 100; Ve = 6000; he = 120000; s = 1; g0 = dbl(9.80665);
rho(h) = rho0 * exp(-h / H);
V(h) = Ve * exp(-(rho(h) - rho(he)) * H / (2 * beta * s));
D(h) = rho(h) * V(h)^2 / (2 * beta);
hs = H * log(rho0 * H / (beta * s));
Vs = Ve * exp(-1/2 + rho(he) * H / (2 * beta * s));
Ds = s * Ve^2 / (2 * H) * exp(-1 + rho(he) * H / (beta * s));
lit(x) = Strprintf("%.25g", x);
print("peak_altitude_m: ", lit(hs));
print("peak_speed_m_s: ", lit(Vs));
print("peak_deceleration_m_s2: ", lit(Ds));
print("peak_deceleration_g: ", lit(Ds / g0));
print("surface_speed_m_s: ", lit(V(0)));
{
  /* log V as the integral of dlogV/dh = rho/(2 beta s) from he down to h. */
  my(direct(h) = Ve * exp(-intnum(x = h, he, rho(x) / (2 * beta * s))), w = 0);
  foreach ([hs, 0, 50000], h, w = max(w, abs(direct(h) / V(h) - 1)));
  w = max(w, abs(D(hs) / Ds - 1)) + abs(V(hs) / Vs - 1);
  printf("closed forms vs direct integration, worst relative: %.3g\n", w);
  if (w > 10^-60, print("FAIL: the closed form does not solve the entry equation"));
  /* The peak is a maximum: D falls 1 m either side. */
  if (!(D(hs + 1) < Ds && D(hs - 1) < Ds), print("FAIL: h* is not a maximum"));
}
