/* Second engine for derivations/targeting_hohmann.mac: Hohmann's first burn for the targeting gate
   in crates/spacelab-astro/src/targeting.rs, at 60 digits, found as the root of the apoapsis
   radius of the burned state by solve() rather than from the closed form, then checked against it.

   mu = 3.986004418e14 m^3/s^2, r1 = 6678 km, r2 = 42 164 km. The apoapsis radius is formed from
   the state the test builds, (r1, 0, 0), (0, vc + dv, 0), via the eccentricity vector and the
   energy -- the same route as the test's `apoapsis_radius`, at 60 digits. FAIL if the root and the
   closed form differ by more than 1e-50 m/s.

       timeout 300 gp -q -f derivations/targeting_hohmann.gp </dev/null                          */

default(realprecision, 60);
mu = 3986004418 * 10^5; r1 = 6678000; r2 = 42164000;
vc = sqrt(mu / r1);
apo(dv) = {
  my(r = [r1, 0, 0], v = [0, vc + dv, 0], rn = r1, v2 = v * v~, rv = r * v~, e, a);
  e = ((v2 - mu / rn) * r - rv * v) / mu;
  a = -mu / (2 * (v2 / 2 - mu / rn));
  a * (1 + sqrt(e * e~));
}
root = solve(dv = 1000, 3000, apo(dv) - r2);
closed = vc * (sqrt(2 * r2 / (r1 + r2)) - 1);
slope = derivnum(dv = root, apo(dv));
xs = sqrt(2 * r2 / (r1 + r2));
printf("hohmann_dv_m_s: %.40g\n", root);
printf("apoapsis_slope_m_per_m_s: %.40g\n", slope);
printf("slope closed form x (r1 + r2)^2/(r1 vc): %.40g\n", xs * (r1 + r2)^2 / (r1 * vc));
if (abs(root - closed) > 10^-50, print("FAIL: the root of the apoapsis constraint is not Hohmann's burn"));
/* derivnum keeps about half the working digits. */
if (abs(slope / (xs * (r1 + r2)^2 / (r1 * vc)) - 1) > 10^-25, print("FAIL: the slope is not its closed form"));
