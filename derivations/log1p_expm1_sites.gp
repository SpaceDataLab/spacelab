/* Second engine for derivations/log1p_expm1_sites.mac (plans/18 Ф28.N, C3 row 14): the decade
   tables of the two small-argument sites, at 100 digits on exact doubles.

   MrpLyapunov::lyapunov, spacelab-control/src/attitude.rs: the potential 2K ln(1 + s.s). The test
   sets the error quaternion to (1, q1, 0, 0) with q1 = 2 s, s = dbl(sqrt(10^-k)), so the code's
   s = q1/2 is s exactly and its s.s is dbl(s^2) = x; K = 1/2 makes V = ln(1 + x) with no other
   rounding. Rows: q1, x, ln(1 + x).

   UsageViolations::assess, spacelab-estimation/src/conjunction_domain.rs: the inaccuracy
   indicator 1 - exp(-|L|). The test gives both encounters unit speed and width and minima 0 and
   2y, y = dbl(10^-k), so L = -y exactly. Rows: y, 1 - exp(-y).

       timeout 120 gp -q -f derivations/log1p_expm1_sites.gp </dev/null                          */

default(realprecision, 100);
default(path, "derivations:."); read("binary64.gp"); dbl = f64;
fmt(v) = concat([ch | ch <- Vec(Strprintf("%.25g", v)), ch != " "]);
{
  print("    // MRP potential: q1, x = s.s, ln(1 + x)");
  for (k = 1, 18,
    my(s = dbl(sqrt(10^-k)), x = dbl(s^2));
    printf("        (%s, %s, %s),\n", fmt(2 * s), fmt(x), fmt(log(1 + x))));
  print("    // inaccuracy indicator: y, 1 - exp(-y)");
  for (k = 1, 18,
    my(y = dbl(10^-k));
    printf("        (%s, %s),\n", fmt(y), fmt(1 - exp(-y))));
}
