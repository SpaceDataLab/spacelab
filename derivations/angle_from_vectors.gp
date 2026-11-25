/* Second engine for derivations/angle_from_vectors.mac (plans/18 Ф28.N, C3 row 15): the decade
   tables of the angle sites, at 100 digits on exact doubles. Each section prints the rows a test
   carries, and checks that every row is the angle it was built from; a row that is not prints FAIL.

   1. The angle between two vectors, and the elevation of one above the plane normal to the other.
      a = dbl((2, -3, 6)/7); b = dbl(cos t u + sin t w) with u = (2, -3, 6)/7 and w = (3, 6, 2)/7
      orthonormal, t = 10^-k and pi - 10^-k, k = 1..12. Rows: b, then atan2(|a x b|, a.b) or
      atan2(a.b, |a x b|), exact for the doubles a and b, which the tests pass to the code as they
      are. The angle: the solar and view zenith and the scattering angle of
      `the_angles_keep_their_digits_next_to_zero_and_pi` (spacelab-earthvision/src/geometry.rs) and
      the phase angle of `the_phase_angle_keeps_its_digits_next_to_opposition_and_conjunction`
      (spacelab-propagation/tests/lightcurve.rs). The elevation:
      `the_elevation_keeps_its_digits_next_to_the_zenith_and_the_nadir`
      (spacelab-sensors/tests/elevation_gates.rs) and
      `the_altitude_keeps_its_digits_next_to_the_zenith_and_the_nadir`
      (spacelab-astrometry/src/reduce.rs).
   2. The declination next to the celestial poles: d = dbl(cos t z + sin t (3, 4, 0)/5) and its
      mirror in the equator, t = 10^-k. Rows: d, atan2(d_z, sqrt(d_x^2 + d_y^2)), for
      `the_declination_keeps_its_digits_next_to_the_poles` (spacelab-astrometry/tests/reduction.rs).

       timeout 120 gp -q -f derivations/angle_from_vectors.gp </dev/null                         */

default(realprecision, 100);
/* The nearest double. The argument is made a t_REAL first: PARI's exponent() of a fraction can be one
   above floor(log2|x|) (exponent(2/7) is -1), and the rounding would then keep 52 bits. */
dbl(x) = if (x == 0, 0, my(y = x + 0., e = exponent(y)); round(y * 2^(52 - e)) / 2^(52 - e));
fmt(v) = concat([ch | ch <- Vec(Strprintf("%.25g", v)), ch != " "]);
cr(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
dt(a, b) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
nrm(v) = sqrt(dt(v, v));
/* atan2(y, x) */
at2(y, x) = arg(x + I*y);
fails = 0;
/* A row is the angle it was built from to within 2^-50: the doubles moved it by less than that. */
check(name, got, want) = if (abs(got - want) > 2^-50, fails++; print("FAIL ", name, ": ", got, " against ", want));
{
  my(u = [2, -3, 6]/7, w = [3, 6, 2]/7, a = apply(dbl, u));
  printf("    // a = (%s, %s, %s)\n", fmt(a[1]), fmt(a[2]), fmt(a[3]));
  my(rows = List());
  for (s = 0, 1,
    for (k = 1, 12,
      my(t = 10^-k, b = apply(dbl, if (s, -cos(t), cos(t)) * u + sin(t) * w));
      my(ang = at2(nrm(cr(a, b)), dt(a, b)), el = at2(dt(a, b), nrm(cr(a, b))));
      check("angle", ang, if (s, Pi - t, t));
      check("elevation", el, if (s, -Pi/2 + t, Pi/2 - t));
      listput(rows, [b, ang, el])));
  print("    // angle: b, atan2(|a x b|, a.b)");
  foreach (rows, r,
    printf("        ([%s, %s, %s], %s),\n", fmt(r[1][1]), fmt(r[1][2]), fmt(r[1][3]), fmt(r[2])));
  print("    // elevation: b, atan2(a.b, |a x b|)");
  foreach (rows, r,
    printf("        ([%s, %s, %s], %s),\n", fmt(r[1][1]), fmt(r[1][2]), fmt(r[1][3]), fmt(r[3])));
}
{
  print("    // declination: d, atan2(d_z, sqrt(d_x^2 + d_y^2))");
  for (s = 0, 1,
    for (k = 1, 12,
      my(t = 10^-k, d = apply(dbl, [3/5 * sin(t), 4/5 * sin(t), if (s, -cos(t), cos(t))]));
      my(dec = at2(d[3], sqrt(d[1]^2 + d[2]^2)));
      check("declination", dec, if (s, -Pi/2 + t, Pi/2 - t));
      printf("        ([%s, %s, %s], %s),\n", fmt(d[1]), fmt(d[2]), fmt(d[3]), fmt(dec))));
}
if (fails, print("FAIL: ", fails, " rows are not the angle they were built from"));
