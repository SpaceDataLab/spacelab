/* Izzo's Lambert solver at 100 digits: the second engine for derivations/lambert_izzo.mac, and the
   references the `lambert_multirev` and `izzo_tof` gates in crates/spacelab-astro/src/lambert.rs
   assert the shipped f64 code against.

   Nothing here uses the forms the code evaluates. The time of flight is Lagrange's equation itself,
   in Izzo's x (x = cos(alpha/2), a = (s/2)/(1 - x^2)) for the checks on T, and in Lagrange's own
   alpha0 (sin^2(alpha0/2) = s/2a, alpha = alpha0 or 2 pi - alpha0) for the arcs:
       sqrt(mu) t = a^(3/2) [2 pi N + (alpha - sin alpha) - (beta - sin beta)],
       sin(beta/2) = sqrt((s - c)/s) sin(alpha0/2), beta < 0 past a half turn,
   and on the hyperbola its sinh form. The minimum time of an N-revolution family is the minimum of
   that equation over alpha0 on the alpha < pi branch, located as the zero of its numerical
   derivative; it is checked against the root of Izzo's dT/dx. The velocities come from the
   semi-latus rectum p = 4a (s - r1)(s - r2)/c^2 sin^2((alpha + beta)/2) and the Lagrange
   coefficients in the transfer angle, and each arc is then flown by the universal Kepler equation
   (the functions of derivations/lambert_universal.gp) and must land on r2 with v2 to 1e-80.

   Printed, in order: the identity checks at 100 digits on random points; the `IZZO_TOF` table of
   the module tests, recomputed and compared with the literals; the `NEAR_PARABOLA` rows, T at the
   doubles next to x = 1; the `MINIMUM_TIMES` rows; the `MULTIREV_ARCS` rows; the two exact
   parabolas at their double-rounded times of flight. A failed check prints FAIL.

       timeout 300 gp -q -f derivations/lambert_izzo.gp </dev/null                              */

default(realprecision, 100);

stumpff(z) = {
  my(c = 0, s = 0, t = 1, k = 0, x);
  if (abs(z) <= 1,
    while (k == 0 || abs(t) / (2 * k + 2)! > 10^-110,
      c += t / (2 * k + 2)!; s += t / (2 * k + 3)!; t *= -z; k++);
    return([c, s]));
  if (z > 0, x = sqrt(z); return([(1 - cos(x)) / z, (x - sin(x)) / x^3]));
  x = sqrt(-z);
  [(cosh(x) - 1) / (-z), (sinh(x) - x) / x^3];
}
kepler(mu, r0, rv, alpha, chi, dt) = {
  my(cs = stumpff(alpha * chi^2), sm = sqrt(mu));
  (rv / sm) * chi^2 * cs[1] + (1 - alpha * r0) * chi^3 * cs[2] + r0 * chi - sm * dt;
}
propagate(mu, r0v, v0v, dt) = {
  my(r0 = sqrt(r0v * r0v~), rv = r0v * v0v~, alpha = 2 / r0 - (v0v * v0v~) / mu, lo = 0, hi = 1,
     chi, cs, sm = sqrt(mu), f, g, rn, fd, gd, rvec);
  while (kepler(mu, r0, rv, alpha, hi, dt) < 0, lo = hi; hi *= 2);
  chi = solve(x = lo, hi, kepler(mu, r0, rv, alpha, x, dt));
  cs = stumpff(alpha * chi^2);
  f = 1 - chi^2 * cs[1] / r0; g = dt - chi^3 * cs[2] / sm;
  rvec = f * r0v + g * v0v; rn = sqrt(rvec * rvec~);
  fd = sm / (rn * r0) * (alpha * chi^3 * cs[2] - chi); gd = 1 - chi^2 * cs[1] / rn;
  [rvec, fd * r0v + gd * v0v];
}
cross(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]];
nrm(v) = sqrt(v * v~);
dbl(x) = my(e = exponent(x)); round(x * 2^(52 - e)) / 2^(52 - e);
fmt(v) = if (abs(v) < 10^-90, "0.0", concat([ch | ch <- Vec(Strprintf("%.25g", v)), ch != " "]));
vfmt(v) = Str("[", fmt(v[1]), ", ", fmt(v[2]), ", ", fmt(v[3]), "]");
fails = 0;
expect(name, ok) = if (!ok, fails++; print("FAIL: ", name));

/* The geometry as lambert.rs reads it: radii, chord, semi-perimeter, the long-way flag and the
   transfer angle in (0, 2 pi), by the sense of the z component of r1 x r2. */
geometry(r1v, r2v, prograde) = {
  my(r1 = nrm(r1v), r2 = nrm(r2v), c = nrm(r2v - r1v), s = (r1 + r2 + c) / 2, cz = cross(r1v, r2v)[3],
     longa = if (prograde, cz < 0, cz >= 0), ac = acos(r1v * r2v~ / (r1 * r2)));
  [r1, r2, c, s, longa, if (longa, 2 * Pi - ac, ac)];
}
lambda_of(r1v, r2v, prograde) = {
  my(g = geometry(r1v, r2v, prograde)); if (g[5], -1, 1) * sqrt(1 - g[3] / g[4]);
}

/* T(x) in the unit t sqrt(2 mu/s^3): Lagrange's equation in x. */
tizzo(x, lam, N) = {
  if (x == 1, return(2 / 3 * (1 - lam^3)));
  my(a = 1 / (1 - x^2), al, be);
  if (a > 0,
    al = 2 * acos(x); be = 2 * asin(lam * sqrt(1 - x^2));
    a^(3/2) * ((al - sin(al)) - (be - sin(be)) + 2 * Pi * N) / 2,
    al = 2 * acosh(x); be = 2 * asinh(lam * sqrt(x^2 - 1));
    (-a)^(3/2) * ((sinh(al) - al) - (sinh(be) - be)) / 2);
}
/* The form the code evaluates, for the identity check only. */
tstumpff(x, lam, N) = {
  my(w = sqrt(abs(1 - x^2)), al, be, sg = if (x < 1, 1, -1));
  al = if (x < 1, 2 * acos(x), 2 * acosh(x));
  be = if (x < 1, 2 * asin(lam * w), 2 * asinh(lam * w));
  ((al / w)^3 * stumpff(sg * al^2)[2] - (be / w)^3 * stumpff(sg * be^2)[2]) / 2
    + if (N > 0, Pi * N / w^3, 0);
}
izzo_dtdx(x, lam, T) = (3 * T * x - 2 + 2 * lam^3 * x / sqrt(1 - lam^2 * (1 - x^2))) / (1 - x^2);
/* The condition number of T's one subtraction, kappa = (|F(alpha)| + |F(beta)| + 2 pi N)/
   |F(alpha) - F(beta) + 2 pi N|, F(u) = u - sin u or sinh u - u: the factor by which the rounding
   of the two terms reaches T, inherent in the equation (it grows as lambda -> 1, c -> 0). At x = 1
   it is its limit (1 + |lambda|^3)/(1 - lambda^3). The tests' ulp budgets are 8 kappa. */
kappa(x, lam, N) = {
  if (x == 1, return((1 + abs(lam)^3) / (1 - lam^3)));
  my(w = sqrt(abs(1 - x^2)), fa, fb);
  if (x < 1,
    my(al = 2 * acos(x), be = 2 * asin(lam * w)); fa = al - sin(al); fb = be - sin(be),
    my(al = 2 * acosh(x), be = 2 * asinh(lam * w)); fa = sinh(al) - al; fb = sinh(be) - be);
  (abs(fa) + abs(fb) + 2 * Pi * N) / abs(fa - fb + 2 * Pi * N);
}

/* ---- 1. lambert_izzo.mac's identities at 100 digits on random points. ---- */
setrand(20261124);
rnd(lo, hi) = lo + (hi - lo) * random(1.) ;
{
  my(worst = [0, 0, 0, 0, 0]);
  for (k = 1, 40,
    my(r1v = [rnd(-9e3, 9e3), rnd(-9e3, 9e3), rnd(-9e3, 9e3)],
       r2v = [rnd(-4e4, 4e4), rnd(-4e4, 4e4), rnd(-4e4, 4e4)], g, n1n2, d, kk, absa, lam);
    for (pro = 0, 1,
      g = geometry(r1v, r2v, pro); n1n2 = g[1] * g[2]; d = r1v * r2v~; kk = nrm(cross(r1v, r2v));
      absa = if (d < 0, kk / sqrt(n1n2 - d), sqrt(n1n2 + d));
      lam = lambda_of(r1v, r2v, pro);
      /* 1-2: lambda = sqrt(r1 r2) cos(dnu/2)/s = |A|/(sqrt 2 s), signed by the sense. */
      worst[1] = max(worst[1], abs(lam - sqrt(n1n2) * cos(g[6] / 2) / g[4]));
      worst[1] = max(worst[1], abs(lam - if (g[5], -1, 1) * absa / (sqrt(2) * g[4])))));
  for (k = 1, 60,
    my(lam = rnd(-0.999, 0.999), N = k % 4, x = if (N == 0 && k % 3 == 0, rnd(1.001, 20), rnd(-0.999, 0.999)),
       T = tizzo(x, lam, N));
    /* 4-5: the Stumpff form is Lagrange's. */
    worst[2] = max(worst[2], abs(tstumpff(x, lam, N) / T - 1));
    /* 16-17: Izzo's dT/dx is the derivative, against a central difference taken at 300 digits on
       rational x and lambda, so that h = 1e-80 is resolved: truncation h^2, rounding 1e-220. */
    my(xq = round(x * 10^30) / 10^30, lq = round(lam * 10^30) / 10^30, h = 10^-80, fd, an);
    localprec(300);
    fd = (tizzo(xq + h, lq, N) - tizzo(xq - h, lq, N)) / (2 * h);
    an = izzo_dtdx(xq, lq, tizzo(xq, lq, N));
    worst[3] = max(worst[3], abs(fd / an - 1)));
  for (k = 1, 20,
    my(lam = rnd(-0.999, 0.999));
    /* 6-8: the limit at x = 1 from both sides. */
    worst[4] = max(worst[4], max(abs(tizzo(1 - 10^-40, lam, 0) - 2 / 3 * (1 - lam^3)),
                                 abs(tizzo(1 + 10^-40, lam, 0) - 2 / 3 * (1 - lam^3)))));
  /* 18 */
  worst[5] = abs(izzo_dtdx(0, 1 / 3, 17) + 2);
  printf("identities at 100 digits: lambda %.3g, T form %.3g, dT/dx %.3g, x -> 1 %.3g, dT/dx(0) + 2 %.3g\n",
         worst[1], worst[2], worst[3], worst[4], worst[5]);
  expect("lambda forms", worst[1] < 10^-90);
  expect("T in the Stumpff form", worst[2] < 10^-90);
  expect("Izzo's dT/dx", worst[3] < 10^-80);
  expect("T at x = 1 +- 1e-40", worst[4] < 10^-38);
  expect("dT/dx at x = 0", worst[5] < 10^-95);
}

/* ---- 2. IZZO_TOF: the module table's literals, recomputed. ---- */
{
  my(rows = [["0.0", "0.5", 0, "1.480210253088817069536076"], ["-0.9", "0.3", 0, "37.20601928472000445313341"],
     ["0.5", "0.9", 0, "0.3213882554999886628004576"], ["0.75", "0.45", 0, "0.7208031367246040107950984"],
     ["-0.25", "0.8", 0, "1.839046851196585578166719"], ["0.3", "0.2", 1, "4.742432784142829841881213"],
     ["-0.5", "0.7", 2, "13.30433815213550543478457"], ["0.0", "-0.5", 0, "1.661382400500976168926567"],
     ["-0.7", "-0.8", 0, "8.196593203105028973287010"], ["0.4", "-0.3", 1, "5.128706840676636927341485"],
     ["0.6", "-0.95", 3, "20.00282179450132532506031"], ["1.05", "0.2", 0, "0.6418958560667003134648856"],
     ["1.5", "0.4", 0, "0.4709958762024938184400980"], ["3.0", "0.9", 0, "0.06295435879031136660460637"],
     ["12.0", "0.6", 0, "0.05322346229764423099810463"], ["1.5", "-0.4", 0, "0.5516955127126932571137157"],
     ["2.5", "-0.85", 0, "0.5823615525963159046384228"], ["1.02", "-0.75", 0, "0.9381126730306875345381481"]],
     worst = 0);
  print("    // IZZO_TOF, (x, lambda, N, T, kappa) at the doubles nearest the decimals");
  for (i = 1, #rows,
    my(x = dbl(eval(rows[i][1])), lam = dbl(eval(rows[i][2])), T = tizzo(x, lam, rows[i][3]));
    worst = max(worst, abs(T / eval(rows[i][4]) - 1));
    printf("    (%s, %s, %d, %s, %.3f),\n", rows[i][1], rows[i][2], rows[i][3], fmt(T),
           kappa(x, lam, rows[i][3])));
  printf("IZZO_TOF literals against Lagrange's equation at 100 digits: worst relative %.3g\n", worst);
  expect("IZZO_TOF literals", worst < 10^-24);
}

/* ---- 3. NEAR_PARABOLA: T at the doubles next to x = 1, both conics, both signs of lambda. ---- */
{
  my(xs = [1 - 10^-3, 1 - 10^-6, 1 - 10^-8, 1 - 10^-10, 1 - 10^-13, 1 - 2^-53, 1, 1 + 2^-52, 1 + 10^-13,
           1 + 10^-10, 1 + 10^-8, 1 + 10^-6, 1 + 10^-3]);
  print("    // NEAR_PARABOLA, (x, lambda, T, kappa) with x the double nearest the decimal");
  foreach (["0.6", "-0.6", "0.95", "-0.95"], lam,
    foreach (xs, x0,
      my(x = dbl(x0), l = dbl(eval(lam)));
      printf("    (%.17g, %s, %s, %.3f),\n", x, lam, fmt(tizzo(x, l, 0)), kappa(x, l, 0))));
}

/* ---- 4. The minimum time of the N-revolution family, two ways. ---- */
mu = 398600;
geometries = [[[8000, 0, 0], [-2000, 8000, 3000]], [[6000, 2000, 1500], [1000, 5000, 4000]]];
tlag(s, c, N, al0, upper, longa) = {
  my(a = s / (2 * sin(al0 / 2)^2), al = if (upper, 2 * Pi - al0, al0),
     be = 2 * asin(sqrt((s - c) / s) * sin(al0 / 2)));
  if (longa, be = -be);
  [sqrt(a^3 / mu) * (2 * Pi * N + (al - sin(al)) - (be - sin(be))), a, al, be];
}
/* [t_min, alpha0 at the minimum, t at alpha0 = pi (the minimum-energy arc)] */
tmin(r1v, r2v, prograde, N) = {
  my(g = geometry(r1v, r2v, prograde), am);
  am = solve(z = 10^-8, Pi, derivnum(u = z, tlag(g[4], g[3], N, u, 0, g[5])[1]));
  [tlag(g[4], g[3], N, am, 0, g[5])[1], am, tlag(g[4], g[3], N, Pi, 0, g[5])[1]];
}
{
  my(worst_t = 0, worst_x = 0);
  print("    // MINIMUM_TIMES, (geometry, prograde, N, t_min in s)");
  for (i = 1, #geometries, for (k = 0, 1, for (N = 1, 3,
    my(r1v = geometries[i][1], r2v = geometries[i][2], pro = (k == 0), g = geometry(r1v, r2v, pro),
       m = tmin(r1v, r2v, pro, N), lam = lambda_of(r1v, r2v, pro), xm, unit);
    /* Izzo's dT/dx = 0 in x, on the time from Lagrange's equation in x. */
    xm = solve(x = 10^-30, 1 - 10^-30, izzo_dtdx(x, lam, tizzo(x, lam, N)));
    unit = sqrt(2 * mu / g[4]^3);
    worst_t = max(worst_t, abs(tizzo(xm, lam, N) / unit / m[1] - 1));
    worst_x = max(worst_x, abs(xm - cos(m[2] / 2)));
    printf("    (%d, %s, %d, %s),\n", i - 1, if (pro, "PROGRADE", "RETROGRADE"), N, fmt(m[1])))));
  printf("minimum times by Lagrange's equation and by Izzo's dT/dx: t %.3g relative, x %.3g\n",
         worst_t, worst_x);
  expect("minimum time, two ways", worst_t < 10^-80 && worst_x < 10^-40);
}

/* ---- 5. The arcs. ---- */
transfer_velocities(r1v, r2v, g, a, al, be, hyperbolic) = {
  my(p, f, gg, gd, dnu = g[6]);
  p = 4 * a * (g[4] - g[1]) * (g[4] - g[2]) / g[3]^2
      * if (hyperbolic, sinh((al + be) / 2)^2, sin((al + be) / 2)^2);
  f = 1 - g[2] / p * (1 - cos(dnu));
  gg = g[1] * g[2] * sin(dnu) / sqrt(mu * p);
  gd = 1 - g[1] / p * (1 - cos(dnu));
  [(r2v - f * r1v) / gg, (gd * r2v - r1v) / gg];
}
tof_hyperbola(s, c, a, longa) = {
  my(al = 2 * asinh(sqrt(s / (2 * a))), be = 2 * asinh(sqrt((s - c) / (2 * a))));
  if (longa, be = -be);
  [sqrt(a^3 / mu) * ((sinh(al) - al) - (sinh(be) - be)), al, be];
}
/* The N-revolution arc on Izzo's branch: right = the root with the larger x. N = 0 has one root,
   on the hyperbola below the parabolic time. */
arc(r1v, r2v, tof, prograde, N, right) = {
  my(g = geometry(r1v, r2v, prograde), s = g[4], c = g[3], longa = g[5], m, lo, sol, a, tp, upper);
  if (N == 0,
    tp = sqrt(2) / 3 * sqrt(s^3 / mu) * (1 - if (longa, -1, 1) * ((s - c) / s)^(3/2));
    if (tof < tp,
      lo = s / 2^20; hi = s;
      while (tof_hyperbola(s, c, hi, longa)[1] < tof, hi *= 2);
      a = solve(z = lo, hi, tof_hyperbola(s, c, z, longa)[1] - tof);
      sol = tof_hyperbola(s, c, a, longa);
      return(transfer_velocities(r1v, r2v, g, a, sol[2], sol[3], 1)));
    upper = tof > tlag(s, c, 0, Pi, 0, longa)[1];
    lo = Pi / 2;
    while ((tlag(s, c, 0, lo, upper, longa)[1] - tof) * if (upper, 1, -1) < 0, lo /= 2);
    a = solve(z = lo, Pi, tlag(s, c, 0, z, upper, longa)[1] - tof),
    m = tmin(r1v, r2v, prograde, N);
    if (tof < m[1], error("below the minimum time"));
    if (right,
      upper = 0; lo = m[2] / 2;
      while (tlag(s, c, N, lo, 0, longa)[1] < tof, lo /= 2);
      a = solve(z = lo, m[2], tlag(s, c, N, z, 0, longa)[1] - tof),
      if (tof <= m[3],
        upper = 0; a = solve(z = m[2], Pi, tlag(s, c, N, z, 0, longa)[1] - tof),
        upper = 1; lo = Pi / 2;
        while (tlag(s, c, N, lo, 1, longa)[1] < tof, lo /= 2);
        a = solve(z = lo, Pi, tlag(s, c, N, z, 1, longa)[1] - tof))));
  sol = tlag(s, c, N, a, upper, longa);
  transfer_velocities(r1v, r2v, g, sol[2], sol[3], sol[4], 0);
}
worst = 0;
close(r1v, r2v, tof, v) = {
  my(end = propagate(mu, r1v, v[1], tof), dr = end[1] - r2v, dv = end[2] - v[2]);
  worst = max(worst, max(nrm(dr) / nrm(r2v), nrm(dv) / nrm(v[1])));
}
/* The near-half-turn geometry: an ellipse a = 8000, e = 1/10 in a tilted plane, from nu = 0.2 to
   nu = 0.2 + pi - 1.5e-9, the positions rounded to doubles and the time to the double nearest
   Kepler's. 1 - c/s rounds below zero on it, which is how lambda was NaN. */
plane_e1 = [6000, 2000, 1500] / nrm([6000, 2000, 1500]);
plane_e2 = [1000, 5000, 4000] - ([1000, 5000, 4000] * plane_e1~) * plane_e1;
plane_e2 = plane_e2 / nrm(plane_e2);
ellipse_point(nu) = apply(dbl, 8000 * (1 - 1 / 100) / (1 + cos(nu) / 10) * (cos(nu) * plane_e1 + sin(nu) * plane_e2));
mean_anomaly(nu) = my(E = 2 * arg(sqrt(11 / 10) * cos(nu / 2) + I * sqrt(9 / 10) * sin(nu / 2))); E - sin(E) / 10;
{
  half_turn = [ellipse_point(1 / 5), ellipse_point(1 / 5 + Pi - 3 / 2 * 10^-9),
               dbl(sqrt(8000^3 / mu) * (mean_anomaly(1 / 5 + Pi - 3 / 2 * 10^-9) - mean_anomaly(1 / 5)))];
}
{
  my(cases = List(), tofs = [[[9500, 14000], [20000]], [[5100, 9000], [10000]]]);
  for (i = 1, #geometries, for (k = 0, 1,
    listput(cases, [geometries[i][1], geometries[i][2], 3600, k == 0, 0, 0]);
    for (N = 1, 2, foreach (tofs[i][N], t, for (b = 0, 1,
      listput(cases, [geometries[i][1], geometries[i][2], t, k == 0, N, b]))))));
  listput(cases, [half_turn[1], half_turn[2], half_turn[3], 1, 0, 0]);
  print("    // MULTIREV_ARCS, (r1, r2, tof, prograde, n_rev, long_branch, v1, v2)");
  for (i = 1, #cases,
    my(cs = cases[i], v = arc(cs[1], cs[2], cs[3], cs[4], cs[5], cs[6]));
    close(cs[1], cs[2], cs[3], v);
    printf("    (%s, %s, %s, %s, %d, %s, %s, %s),\n", vfmt(cs[1] * 1.), vfmt(cs[2] * 1.), fmt(cs[3] * 1.),
           if (cs[4], "PROGRADE", "RETROGRADE"), cs[5], if (cs[6], "HIGH_X", "LOW_X"), vfmt(v[1]), vfmt(v[2])));
  printf("multi-revolution arcs flown by the universal Kepler equation, worst relative miss: %.3g\n", worst);
  expect("every arc solves the boundary-value problem", worst < 10^-80);
  printf("half-turn geometry: sin of the transfer angle %.6g, 1 - c/s %.6g\n",
         nrm(cross(half_turn[1], half_turn[2])) / (nrm(half_turn[1]) * nrm(half_turn[2])),
         1 - geometry(half_turn[1], half_turn[2], 1)[3] / geometry(half_turn[1], half_turn[2], 1)[4]);
}

/* ---- 6. The two exact parabolas of lambert_izzo.mac 10-15, mu = 1, at the double times. ---- */
{
  my(save = mu, cases, w = 0);
  mu = 1;
  cases = [[[1, 0, 0], [0, 2, 0], 4 * sqrt(2) / 3, [0, sqrt(2), 0], [-1, 1, 0] / sqrt(2)],
           [[0, -2, 0], [-8, 6, 0], 40 * sqrt(2) / 3, [1, 1, 0] / sqrt(2), [-0.6, 0.2, 0] / sqrt(2)]];
  for (i = 1, #cases,
    my(cs = cases[i], t = dbl(cs[3]), v = arc(cs[1], cs[2], t, 1, 0, 0), g = geometry(cs[1], cs[2], 1),
       lam = lambda_of(cs[1], cs[2], 1), shift);
    close(cs[1], cs[2], t, v);
    shift = max(nrm(v[1] - cs[4]), nrm(v[2] - cs[5])) / nrm(cs[4]);
    w = max(w, abs(2 / 3 * (1 - lam^3) / sqrt(2 / g[4]^3) - cs[3]));
    printf("parabola %d: tof %s (double %s), lambda %s, v1 %s, v2 %s, rounding shift %.3g of |v1|\n", i,
           fmt(cs[3]), fmt(t), fmt(lam), vfmt(v[1]), vfmt(v[2]), shift);
    expect("the double time moves the parabola by under an ulp", shift < 2^-53));
  printf("2/3 (1 - lambda^3) in seconds against Barker's time: %.3g\n", w);
  expect("Euler's time is Barker's", w < 10^-95);
  mu = save;
}
printf("worst relative miss including the parabolas: %.3g\n", worst);
expect("every arc, parabolas included", worst < 10^-80);
if (fails == 0, print("all checks hold"), print("FAIL: ", fails, " checks"));
