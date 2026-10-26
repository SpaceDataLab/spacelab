/* Lambert arcs at 100 digits by Lagrange's theorem: the second engine for
   derivations/lambert_universal.mac and the reference the `lambert` gates in
   crates/spacelab-astro/src/lambert.rs assert the shipped f64 solver against.

   This does not use the universal variable z that lambert.rs iterates on. With chord c,
   semi-perimeter s = (r1 + r2 + c)/2 and transfer angle dnu, Lagrange's equation gives the time
   of flight on the conic of semi-major axis a,
       ellipse    t = sqrt(a^3/mu) [(al - sin al) - (be - sin be)],
                  sin(al0/2) = sqrt(s/(2a)), sin(be0/2) = sqrt((s - c)/(2a)),
       hyperbola  t = sqrt(|a|^3/mu) [(sinh al - al) - (sinh be - be)],
                  sinh(al/2) = sqrt(s/(2|a|)), sinh(be/2) = sqrt((s - c)/(2|a|)),
   with be = -be0 past dnu = pi and al = 2 pi - al0 past the minimum-energy time. t(a) is monotone
   on each branch, so a is found by bracketing and solve(). The semi-latus rectum
   p = 4 |a| (s - r1)(s - r2)/c^2 sin^2((al + be)/2) (sinh^2 on the hyperbola) and the Lagrange
   coefficients in dnu then give v1 = (r2 - f r1)/g, v2 = (gdot r2 - r1)/g.

   Each arc is then checked independently of all that: (r1, v1) is propagated by t with the
   universal Kepler equation (the functions of derivations/kepler_universal.gp) and must land on r2
   with velocity v2. The worst miss over every arc is printed; it must be below 1e-80.

   Rows printed for the `LAGRANGE_ARCS` table: (r1, r2, tof, sense, v1, v2) at 25 digits, the
   sense as the test module's `PROGRADE` / `RETROGRADE`.
   mu = 398600 km^3/s^2; the geometries are the test file's `GEOMETRIES` and Curtis Example 5.2.
   Two further rows give the exact conics of the .mac file (mu = 1) at the double-rounded times of
   flight, with the shift that rounding causes against the closed-form velocities.

       timeout 300 gp -q -f derivations/lambert_universal.gp </dev/null                          */

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
/* (r, v) after dt from (r0v, v0v), by the universal variable. */
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

/* The transfer angle as lambert.rs resolves it: the short way when the sense of r1 x r2 matches. */
transfer_angle(r1v, r2v, prograde) = {
  my(r1 = sqrt(r1v * r1v~), r2 = sqrt(r2v * r2v~), ac = acos(r1v * r2v~ / (r1 * r2)),
     cz = cross(r1v, r2v)[3]);
  if (prograde, if (cz >= 0, ac, 2 * Pi - ac), if (cz < 0, ac, 2 * Pi - ac));
}

tof_ellipse(mu, s, c, a, long_angle, long_time) = {
  my(al = 2 * asin(sqrt(s / (2 * a))), be = 2 * asin(sqrt((s - c) / (2 * a))));
  if (long_angle, be = -be); if (long_time, al = 2 * Pi - al);
  [sqrt(a^3 / mu) * ((al - sin(al)) - (be - sin(be))), al, be];
}
tof_hyperbola(mu, s, c, a, long_angle) = {
  my(al = 2 * asinh(sqrt(s / (2 * a))), be = 2 * asinh(sqrt((s - c) / (2 * a))));
  if (long_angle, be = -be);
  [sqrt(a^3 / mu) * ((sinh(al) - al) - (sinh(be) - be)), al, be];
}

arc(mu, r1v, r2v, tof, prograde) = {
  my(r1 = sqrt(r1v * r1v~), r2 = sqrt(r2v * r2v~), d = r2v - r1v, c = sqrt(d * d~),
     s = (r1 + r2 + c) / 2, dnu = transfer_angle(r1v, r2v, prograde), long_angle = dnu > Pi,
     tp, tm, a, lo, hi, sol, p, f, g, gd);
  /* Parabolic and minimum-energy times bound the branches. */
  tp = sqrt(2) / 3 * sqrt(s^3 / mu) * (1 - (if (long_angle, -1, 1)) * ((s - c) / s)^(3/2));
  tm = tof_ellipse(mu, s, c, s / 2, long_angle, 0)[1];
  if (tof < tp,
    /* Hyperbola: t rises from 0 to tp as |a| grows. */
    lo = s / 2^20; hi = s;
    while (tof_hyperbola(mu, s, c, hi, long_angle)[1] < tof, hi *= 2);
    while (tof_hyperbola(mu, s, c, lo, long_angle)[1] > tof, lo /= 2);
    a = solve(x = lo, hi, tof_hyperbola(mu, s, c, x, long_angle)[1] - tof);
    sol = tof_hyperbola(mu, s, c, a, long_angle);
    p = 4 * a * (s - r1) * (s - r2) / c^2 * sinh((sol[2] + sol[3]) / 2)^2,
    /* Ellipse, on the branch the time selects. */
    my(long_time = tof > tm);
    lo = s / 2; hi = s;
    while ((tof_ellipse(mu, s, c, hi, long_angle, long_time)[1] - tof) * (if (long_time, 1, -1)) < 0,
           hi *= 2);
    a = solve(x = lo, hi, tof_ellipse(mu, s, c, x, long_angle, long_time)[1] - tof);
    sol = tof_ellipse(mu, s, c, a, long_angle, long_time);
    p = 4 * a * (s - r1) * (s - r2) / c^2 * sin((sol[2] + sol[3]) / 2)^2);
  f = 1 - r2 / p * (1 - cos(dnu));
  g = r1 * r2 * sin(dnu) / sqrt(mu * p);
  gd = 1 - r1 / p * (1 - cos(dnu));
  [(r2v - f * r1v) / g, (gd * r2v - r1v) / g];
}

fmt(v) = if (abs(v) < 10^-90, "0.0", concat([ch | ch <- Vec(Strprintf("%.25g", v)), ch != " "]));
vfmt(v) = Str("[", fmt(v[1]), ", ", fmt(v[2]), ", ", fmt(v[3]), "]");
worst = 0;
check(mu, r1v, r2v, tof, v) = {
  my(end = propagate(mu, r1v, v[1], tof), dr = end[1] - r2v, dv = end[2] - v[2]);
  worst = max(worst, max(sqrt(dr * dr~) / sqrt(r2v * r2v~), sqrt(dv * dv~) / sqrt(v[1] * v[1]~)));
}

mu = 398600;
{
  geometries = [[[6000, 2000, 1500], [1000, 5000, 4000]], [[2000, 6000, 1500], [5000, 1000, 4000]]];
  tofs = [400, 1000, 3000, 10000, 40000, 120000];
  print("    // (r1, r2, tof, prograde, v1, v2)");
  for (i = 1, #geometries, for (k = 0, 1, for (j = 1, #tofs,
    my(r1v = geometries[i][1], r2v = geometries[i][2], pro = (k == 0), v = arc(mu, r1v, r2v, tofs[j], pro));
    check(mu, r1v, r2v, tofs[j], v);
    printf("    (%s, %s, %s, %s, %s, %s),\n", vfmt(r1v * 1.), vfmt(r2v * 1.), fmt(tofs[j] * 1.),
           if (pro, "PROGRADE", "RETROGRADE"), vfmt(v[1]), vfmt(v[2])))));
  my(r1v = [5000, 10000, 2100], r2v = [-14600, 2500, 7000], v = arc(mu, r1v, r2v, 3600, 1));
  check(mu, r1v, r2v, 3600, v);
  printf("    (%s, %s, %s, PROGRADE, %s, %s),\n", vfmt(r1v * 1.), vfmt(r2v * 1.), fmt(3600.),
         vfmt(v[1]), vfmt(v[2]));
}
printf("Lagrange-theorem arcs propagated by the universal Kepler equation, worst relative miss: %.3g\n", worst);
if (worst > 10^-80, print("FAIL: an arc does not solve the boundary-value problem"));

/* The exact conics, at the double-rounded times of flight. */
dbl(x) = my(e = exponent(x)); round(x * 2^(52 - e)) / 2^(52 - e);
{
  my(te = 2 * sqrt(2) * (Pi / 3 - sqrt(3) / 4), th = 2 * sqrt(3) - log(2 + sqrt(3)),
     ve = arc(1, [1, 0, 0], [0, 3/2, 0], dbl(te), 1), vh = arc(1, [1, 0, 0], [0, 3, 0], dbl(th), 1),
     d1, d2);
  check(1, [1, 0, 0], [0, 3/2, 0], dbl(te), ve);
  check(1, [1, 0, 0], [0, 3, 0], dbl(th), vh);
  d1 = ve[1] - [0, sqrt(3/2), 0]; d2 = ve[2] - sqrt(2/3) * [-1, 1/2, 0];
  printf("ellipse:   tof %s (double %s), v1 %s, v2 %s, rounding shift %.3g\n", fmt(te), fmt(dbl(te)),
         vfmt(ve[1]), vfmt(ve[2]), max(sqrt(d1 * d1~), sqrt(d2 * d2~)));
  d1 = vh[1] - [0, sqrt(3), 0]; d2 = vh[2] - [-1, 2, 0] / sqrt(3);
  printf("hyperbola: tof %s (double %s), v1 %s, v2 %s, rounding shift %.3g\n", fmt(th), fmt(dbl(th)),
         vfmt(vh[1]), vfmt(vh[2]), max(sqrt(d1 * d1~), sqrt(d2 * d2~)));
}
printf("worst relative miss including the exact conics: %.3g\n", worst);
if (worst > 10^-80, print("FAIL: an exact-conic arc does not solve the boundary-value problem"));
