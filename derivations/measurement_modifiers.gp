/* The Shapiro delay of one leg from its defining integral, at 50 digits: the second engine for
   derivations/measurement_modifiers.mac and the reference the `ShapiroDelay` gate in
   crates/spacelab-estimation/src/modifier.rs asserts against.

       D = (2 GM/c^2) * int_0^1 |P2 - P1| / |P1 + t (P2 - P1)| dt

   along the straight leg from the station P1 to the satellite P2, by intnum -- no logarithm, no
   closed form -- against the IERS closed form (2 GM/c^2) log((r1 + r2 + p)/(r1 + r2 - p)).

       timeout 300 gp -q -f derivations/measurement_modifiers.gp </dev/null                          */

default(realprecision, 60);
c = 299792458; gm = 3986004418 * 10^5;
nrm(v) = sqrt(v * v~);

quad(P1, P2) = {
  my(d = P2 - P1);
  (2 * gm / c^2) * intnum(t = 0, 1, nrm(d) / nrm(P1 + t * d));
}
closed(P1, P2) = {
  my(r1 = nrm(P1), r2 = nrm(P2), p = nrm(P2 - P1));
  (2 * gm / c^2) * log((r1 + r2 + p) / (r1 + r2 - p));
}

{
my(legs = [["module-test leg", [4075530, 931781, 4801618], [3961000, 2614000, 5215000]],
           /* r1 = 7000 km, and a point placed so that r2 = 7500 km and p = 900 km. */
           ["(7000000, 7500000, 900000) m", [7000000, 0, 0],
            [(7000000^2 + 7500000^2 - 900000^2) / (2 * 7000000),
             sqrt(7500000^2 - ((7000000^2 + 7500000^2 - 900000^2) / (2 * 7000000))^2), 0]]],
   bad = 0);
for (i = 1, #legs,
  my(q = quad(legs[i][2], legs[i][3]), cf = closed(legs[i][2], legs[i][3]));
  if (abs(q - cf) > 10^-50 * abs(cf), bad++);
  printf("%s: integral %s m, closed form - integral %.1e m\n", legs[i][1], Strprintf("%.40e", q), cf - q));
printf("%d of %d legs with the closed form off the integral by more than 1e-50 relative\n", bad, #legs);
}
