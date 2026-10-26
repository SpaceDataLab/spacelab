/* Second engine for derivations/iod_classical.mac: sightings that satisfy Gauss's and Laplace's
   truncated models exactly, and the 100-digit solutions of those models for the doubles the gates
   in crates/spacelab-astro/src/iod.rs pass -- found by a route that shares nothing with the coded
   determinants and octics.

   mu = 398600 km^3/s^2; epochs t = 0, 300, 700 s.

   Gauss. A middle position r2, a first position r1, and r3 = (r2 - c1 r1)/c3 with Gauss's
   c1 = tau3/tau (1 + mu (tau^2 - tau3^2)/(6 |r2|^3)), c3 = -tau1/tau (1 + mu (tau^2 - tau1^2)/(6 |r2|^3)),
   so r2 = c1 r1 + c3 r3 holds exactly. Three separated stations R_i; the lines of sight
   L_i = (r_i - R_i)/|r_i - R_i| rounded to doubles. The model is then solved for those doubles
   directly: for a trial r the vector equation c1(r)(R1 + rho1 L1) - (R2 + rho2 L2) + c3(r)(R3 + rho3 L3) = 0
   is a 3x3 linear system in the ranges (matsolve), and r is the root of |R2 + rho2 L2| - r.
   Every sign change of that function on the coded scan window [0.01, 1000] |R2| is listed; the
   coded scan takes the first with all ranges positive. Output: r2 and the coded truncated-f,g
   velocity (f1 r3 - f3 r1)/(f1 g3 - f3 g1) at |r2|.

   Laplace. Three unit directions L_i (rounded), their quadratic's L', L'' at t2; a range rho,
   range rate and acceleration, and an observer velocity R'; then the observer acceleration
   R'' = -mu r/|r|^3 - rho'' L - 2 rho' L' - rho L'' that makes the projected equation of motion hold
   exactly at r = R2 + rho L, and R1, R3 from the quadratic R2 + R' tau + R'' tau^2/2 (rounded).
   Solved for the doubles directly: for a trial r, (L'' + mu L/r^3) rho + 2 L' rho' + L rho'' =
   -R'' - mu R/r^3 is linear in (rho, rho', rho''), and r is the root of |R + rho L| - r. Output:
   r2 = R + rho L and v2 = R' + rho L' + rho' L.

       timeout 300 gp -q -f derivations/iod_classical.gp </dev/null                              */

default(realprecision, 100);
dbl(x) = if (x == 0, 0, my(e = exponent(x)); round(x * 2^(52 - e)) / 2^(52 - e));
nrm(a) = sqrt(a * a~);
unit(a) = a / nrm(a);
fmt(v) = if (abs(v) < 10^-90, "0.0", concat([ch | ch <- Vec(Strprintf("%.25g", v)), ch != " "]));
vfmt(v) = Str("[", fmt(v[1]), ", ", fmt(v[2]), ", ", fmt(v[3]), "]");
mu = 398600; t1 = 0; t2 = 300; t3 = 700;
tau1 = t1 - t2; tau3 = t3 - t2; tau = t3 - t1;

/* Sign changes of f on the coded log-spaced scan, refined by solve(). */
roots(f, scale) = {
  my(lo = scale / 100, hi = 1000 * scale, n = 4096, out = List(), a, b, fa, fb);
  a = lo; fa = f(a);
  for (i = 1, n,
    b = lo * (hi / lo)^(i / n); fb = f(b);
    if (fa * fb <= 0 && fa != 0, listput(out, solve(x = a, b, f(x))));
    a = b; fa = fb);
  Vec(out);
}

/* ---- Gauss ---- */
c1f(r) = tau3 / tau * (1 + mu * (tau^2 - tau3^2) / (6 * r^3));
c3f(r) = -tau1 / tau * (1 + mu * (tau^2 - tau1^2) / (6 * r^3));
{
  gr2 = [6500, 2800, 1900]; gr1 = [7100, 400, 1200];
  my(r = nrm(gr2)); gr3 = (gr2 - c1f(r) * gr1) / c3f(r);
  gR = [[4000, -2000, 4300], [-1500, 5200, 3600], [3000, 4500, -3500]];
  gL = vector(3, i, apply(dbl, unit([gr1, gr2, gr3][i] - gR[i])));
}
gauss_ranges(r) = {
  my(M = matconcat([c1f(r) * gL[1]~, -gL[2]~, c3f(r) * gL[3]~]),
     b = -c1f(r) * gR[1]~ + gR[2]~ - c3f(r) * gR[3]~);
  matsolve(M, b);
}
gauss_miss(r) = my(rho = gauss_ranges(r)); nrm(gR[2] + rho[2] * gL[2]) - r;
{
  my(rs = roots(gauss_miss, nrm(gR[2])), r, rho, p1, p2, p3, f1, f3, g1, g3, k, v2);
  printf("gauss: roots of the model on the scan window: %s\n", rs);
  for (i = 1, #rs, my(q = gauss_ranges(rs[i]));
    printf("  r = %.20g, ranges %s\n", rs[i], [q[1], q[2], q[3]] * 1.));
  r = rs[1]; rho = gauss_ranges(r);
  if (!(rho[1] > 0 && rho[2] > 0 && rho[3] > 0), print("FAIL: the first Gauss root is not physical"));
  p1 = gR[1] + rho[1] * gL[1]; p2 = gR[2] + rho[2] * gL[2]; p3 = gR[3] + rho[3] * gL[3];
  k = mu / r^3;
  f1 = 1 - k * tau1^2 / 2; f3 = 1 - k * tau3^2 / 2; g1 = tau1 - k * tau1^3 / 6; g3 = tau3 - k * tau3^3 / 6;
  v2 = (f1 * p3 - f3 * p1) / (f1 * g3 - f3 * g1);
  printf("gauss: shift of r2 from rounding the lines of sight: %.3g of |r2|\n", nrm(p2 - gr2) / nrm(gr2));
  print("    // Gauss sightings: line of sight, observer");
  for (i = 1, 3, printf("    (%s, %s),\n", vfmt(gL[i]), vfmt(gR[i] * 1.)));
  printf("    // Gauss model solution: r2 %s\n    //                       v2 %s\n", vfmt(p2), vfmt(v2));
}

/* ---- Laplace ---- */
{
w1 = [(t2 - t3) / ((t1 - t2) * (t1 - t3)), ((t2 - t1) + (t2 - t3)) / ((t2 - t1) * (t2 - t3)),
      (t2 - t1) / ((t3 - t1) * (t3 - t2))];
w2 = [2 / ((t1 - t2) * (t1 - t3)), 2 / ((t2 - t1) * (t2 - t3)), 2 / ((t3 - t1) * (t3 - t2))];
}
comb(w, a) = w[1] * a[1] + w[2] * a[2] + w[3] * a[3];
{
  lL = vector(3, i, apply(dbl, unit([[1, 2, 2], [2, 3, 6], [4, 4, 7]][i])));
  my(Ld = comb(w1, lL), Ldd = comb(w2, lL), R2 = [6000, 1000, 2000], rho = 2500, rhod = 13/10,
     rhodd = 1/500, Rd = [-1/10, 2/5, 1/5], r, Rdd);
  r = R2 + rho * lL[2];
  Rdd = -mu * r / nrm(r)^3 - rhodd * lL[2] - 2 * rhod * Ld - rho * Ldd;
  lR = [apply(dbl, R2 + Rd * tau1 + Rdd * tau1^2 / 2), R2, apply(dbl, R2 + Rd * tau3 + Rdd * tau3^2 / 2)];
  lruth = [r, Rd + rho * Ld + rhod * lL[2]];
}
laplace_solve(r) = {
  my(L = lL[2], Ld = comb(w1, lL), Ldd = comb(w2, lL), R = lR[2], Rdd = comb(w2, lR),
     M = matconcat([(Ldd + mu * L / r^3)~, 2 * Ld~, L~]));
  matsolve(M, (-Rdd - mu * R / r^3)~);
}
laplace_miss(r) = my(s = laplace_solve(r)); nrm(lR[2] + s[1] * lL[2]) - r;
{
  my(rs = roots(laplace_miss, nrm(lR[2])), r, s, p2, v2);
  printf("laplace: roots of the model on the scan window: %s\n", rs);
  for (i = 1, #rs, my(q = laplace_solve(rs[i])); printf("  r = %.20g, range %.20g\n", rs[i], q[1]));
  r = rs[1]; s = laplace_solve(r);
  if (!(s[1] > 0), print("FAIL: the first Laplace root is not physical"));
  p2 = lR[2] + s[1] * lL[2];
  v2 = comb(w1, lR) + s[1] * comb(w1, lL) + s[2] * lL[2];
  printf("laplace: shift from rounding: r2 %.3g of |r2|, v2 %.3g of |v2|\n",
         nrm(p2 - lruth[1]) / nrm(lruth[1]), nrm(v2 - lruth[2]) / nrm(lruth[2]));
  print("    // Laplace sightings: line of sight, observer");
  for (i = 1, 3, printf("    (%s, %s),\n", vfmt(lL[i]), vfmt(lR[i] * 1.)));
  printf("    // Laplace model solution: r2 %s\n    //                         v2 %s\n", vfmt(p2), vfmt(v2));
}
