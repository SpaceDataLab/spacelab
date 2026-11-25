/* The second engine for derivations/gpu_porkchop.mac, the counts of F115 on the crate's test grid,
   and the 100-digit reference for the one cell the f64 comparator refuses (plan 18 Ф28.M).

   1. Each identity of the .mac file at 100 digits at sample points; the series of C and S to z^4
      as exact power series.
   2. The test grid of `gpu_matches_cpu_porkchop` (crates/spacelab-gpu/src/lib.rs): 64 departures
      at 1 au, theta = 0.09 k, t = 10 k days; 64 arrivals at 1.524 au, theta = 0.5 + 0.04 k,
      t = (160 + 5 k) days; each state as the test builds it, the double product rounded to f32.
      In each sense and on each branch it counts the cells with a positive time of flight, those
      outside the |sin dnu| < 1e-2 mask, those whose root lies in the kernel's bracket (the short
      way's runs from the y = 0 edge to 39, the long way's from -50 to 39) and, on the prograde
      short way, those the shipped bracket could hold: it needed y(-50) > 0, and none has it.
   3. Departure 46, arrival 62, retrograde, short way: a 10-day arc at 252 km/s whose root lies
      next to the y = 0 edge. The f64 solver of spacelab-astro refused it until Ф28.N, its closure
      check's universal propagator having stopped on its step cap; here it is solved at 100 digits
      from the f32 inputs the test prints, and its arc is checked by Kepler's equation in the
      universal variable: alpha chi^2 = z, the time equation, and the arrival at r2. The departure
      velocity is printed for `lambert`'s test of the same cell.
   A failed check stops with an error.

       timeout 120 gp -q -f derivations/gpu_porkchop.gp </dev/null                               */

default(realprecision, 100);
check(name, ok) = if (!ok, error("check broken: ", name));
small(x) = abs(x) < 10^-90;

/* ---- 1. the identities ------------------------------------------------------------------------ */
pc(z) = 1/2 + z*(-1/24 + z*(1/720 + z*(-1/40320 + z*(1/3628800))));
ps(z) = 1/6 + z*(-1/120 + z*(1/5040 + z*(-1/362880 + z*(1/39916800))));
{
  my(t = (1 - cos(x)) / x^2 + O(x^10), h = (cosh(x) - 1) / x^2 + O(x^10),
     s = (x - sin(x)) / x^3 + O(x^10), sh = (sinh(x) - x) / x^3 + O(x^10));
  check("series", truncate(t) == pc(x^2) && truncate(h) == pc(-x^2)
                  && truncate(s) == ps(x^2) && truncate(sh) == ps(-x^2));
}
{
  my(n = 0, r1 = 1, r2 = 1524/1000, A = 6/5, mu = 7/3, t = 11/5);
  foreach([1/3, 2, 5, 6], w,
    my(Ce = (1 - cos(w)) / w^2, Se = (w - sin(w)) / w^3, Ch = (cosh(w) - 1) / w^2,
       Sh = (sinh(w) - w) / w^3, sCe = sqrt(Ce), sCh = sqrt(Ch));
    check("half angle", small(Ce - 2 * sin(w/2)^2 / w^2) && small(Ch - 2 * sinh(w/2)^2 / w^2));
    check("sqrt C", small(sCe - sqrt(2) * sin(w/2) / w) && small(sCh - sqrt(2) * sinh(w/2) / w));
    check("y", small(A * (w^2 * Se - 1) / sCe + sqrt(2) * A * cos(w/2))
               && small(A * (-w^2 * Sh - 1) / sCh + sqrt(2) * A * cosh(w/2)));
    /* dy/dz by a central difference in z at 100 digits, against the closed forms */
    my(e = 10^-30, yh(z) = r1 + r2 - sqrt(2) * A * cosh(sqrt(-z) / 2),
       ye(z) = r1 + r2 - sqrt(2) * A * cos(sqrt(z) / 2));
    check("dy/dz", abs((yh(-w^2 + e) - yh(-w^2 - e)) / (2 * e) - sqrt(2) * A * sinh(w/2) / (4 * w)) < 10^-50
                   && abs((ye(w^2 + e) - ye(w^2 - e)) / (2 * e) - sqrt(2) * A * sin(w/2) / (4 * w)) < 10^-50);
    n++);
  my(q = (r1 + r2) / (sqrt(2) * A), zmin = -(2 * acosh(q))^2);
  check("z_min", small(r1 + r2 - sqrt(2) * A * cosh(sqrt(-zmin) / 2)));
  foreach([1/10, 1, 2, 3], d,
    check("acosh argument", small((r1 + r2)^2 - 2 * r1 * r2 * (1 + cos(d)) - ((r1 - r2)^2 + 4 * r1 * r2 * sin(d/2)^2)));
    check("A forms", small((sin(d) * sqrt(r1 * r2 / (1 - cos(d))))^2 - r1 * r2 * (1 + cos(d)))
                     && small(sin(acos(cos(d))) - sqrt(1 - cos(d)^2)));
    check("A half angle", small(r1 * r2 * (1 + cos(d)) - 2 * r1 * r2 * cos(d/2)^2)));
  my(u = [3/7, -2/5, 1/11], v = [-5/13, 1/3, 7/17], c = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]]);
  check("Lagrange", c * c~ + (u * v~)^2 == (u * u~) * (v * v~));
  printf("identities at 100 digits: %d sample points, series exact\n", n);
}

/* ---- 2. the test grid ------------------------------------------------------------------------- */
/* round to p significant bits, ties to even */
rn(x, p) = {
  my(e, m, f, d);
  if (x == 0, return(0));
  e = exponent(x); m = x * 2^(p - 1 - e); f = floor(m); d = m - f;
  if (d > 1/2, f++, d == 1/2 && f % 2, f++);
  f * 2^(e - p + 1);
}
d64(x) = rn(x, 53);
f32(x) = rn(x, 24);
AU = 149597870700; DAY = 86400; MU = f32(d64(132712440018 * 10^9));
body(radius, theta) = [f32(d64(radius * d64(cos(theta)))), f32(d64(radius * d64(sin(theta)))), 0];
deps = vector(64, k, body(AU, d64((k - 1) * d64(9/100))));
arrs = vector(64, k, body(d64(d64(1524/1000) * AU), d64(1/2 + d64((k - 1) * d64(4/100)))));
dept = vector(64, k, f32((k - 1) * 10 * DAY));
arrt = vector(64, k, f32((160 + 5 * (k - 1)) * DAY));
cross(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]];
stumpff(z) = {
  my(x = sqrt(abs(z)));
  if (z > 0, [2 * sin(x/2)^2 / z, (x - sin(x)) / x^3], [2 * sinh(x/2)^2 / -z, (sinh(x) - x) / x^3]);
}
yz(z, r1, r2, A) = r1 + r2 - sqrt(2) * A * if (z > 0, cos(sqrt(z) / 2), cosh(sqrt(-z) / 2));
F(z, r1, r2, A, tof) = {
  my(y = yz(z, r1, r2, A), cs);
  if (y <= 0, return(-sqrt(MU) * tof));
  cs = stumpff(z);
  (y / cs[1])^(3/2) * cs[2] + A * sqrt(y) - sqrt(MU) * tof;
}
/* [r1, r2, |r1 x r2|, A, short way] */
geometry(r1v, r2v, prograde) = {
  my(r1 = sqrt(r1v * r1v~), r2 = sqrt(r2v * r2v~), c = cross(r1v, r2v), cn = sqrt(c * c~),
     d = r1v * r2v~, short = (c[3] >= 0) == prograde, A);
  A = if (d < 0, cn / sqrt(r1 * r2 - d), sqrt(r1 * r2 + d));
  [r1, r2, cn, if (short, A, -A), short];
}
{
  foreach([1, 0], prograde,
    my(n = [0, 0], open = [0, 0], held = [0, 0], shipped = 0);
    for (i = 1, 64, for (j = 1, 64,
      my(tof = f32(arrt[j] - dept[i]), g, b, inside);
      if (tof <= 0, next);
      g = geometry(deps[i][1..3], arrs[j][1..3], prograde); b = if (g[5], 1, 2);
      n[b]++;
      if (g[3] / (g[1] * g[2]) < 1/100, next);
      open[b]++;
      inside = (g[5] || F(-50, g[1], g[2], g[4], tof) < 0) && F(39, g[1], g[2], g[4], tof) > 0;
      held[b] += inside;
      if (prograde && g[5] && yz(-50, g[1], g[2], g[4]) > 0 && F(-50, g[1], g[2], g[4], tof) < 0
          && inside, shipped++)));
    printf("%s: short way %d cells, %d outside the mask, %d with the root in the bracket; long way %d, %d, %d\n",
           if (prograde, "prograde", "retrograde"), n[1], open[1], held[1], n[2], open[2], held[2]);
    if (prograde,
      check("prograde counts", n == [1283, 765] && open == [1273, 754] && held == [1273, 562]);
      check("shipped bracket", shipped == 0);
      printf("prograde, short way: the shipped bracket at z = -50 holds %d of the %d\n", shipped, held[1]),
      check("retrograde counts", n == [765, 1283] && open == [754, 1273] && held == [754, 1273])));
}

/* ---- 3. the cell the f64 comparator refuses ---------------------------------------------------- */
bits(b) = my(s = b >> 31, e = (b >> 23) % 256, m = b % 2^23); if (e == 0 && m == 0, 0, (-1)^s * (2^23 + m) * 2^(e - 150));
{
  my(r1v = apply(bits, [3516329313, 3521788898, 0]), vdep = apply(bits, [1187224214, 3330019977, 0]),
     r2v = apply(bits, [3528560700, 1359521192, 0]), tof = bits(1276831808) - bits(1276615808),
     mu = bits(1625700394), g, zmin, z, y, cs, fl, gl, v1, dv, c3, chi, alpha, rv, kep, rarr);
  my(th = d64(46 * d64(9/100)), speed = d64(sqrt(d64(132712440018 * 10^9) / AU)));
  check("the grid's states are the test's", r1v == deps[47][1..3] && r2v == arrs[63][1..3]
        && vdep == [f32(-d64(speed * d64(sin(th)))), f32(d64(speed * d64(cos(th)))), 0]
        && tof == f32(arrt[63] - dept[47]) && mu == MU);
  g = geometry(r1v, r2v, 0);
  check("short way", g[5]);
  zmin = -(2 * acosh((g[1] + g[2]) / (sqrt(2) * g[4])))^2;
  z = solve(x = zmin, 39, F(x, g[1], g[2], g[4], tof));
  y = yz(z, g[1], g[2], g[4]); cs = stumpff(z);
  fl = 1 - y / g[1]; gl = g[4] * sqrt(y / mu);
  v1 = (r2v - fl * r1v) / gl; dv = v1 - vdep; c3 = dv * dv~;
  /* Kepler's equation from (r1, v1) at chi = sqrt(y/C): it must give z, the time and r2. */
  chi = sqrt(y / cs[1]); alpha = 2 / g[1] - v1 * v1~ / mu; rv = r1v * v1~;
  kep = rv / sqrt(mu) * chi^2 * cs[1] + (1 - alpha * g[1]) * chi^3 * cs[2] + g[1] * chi - sqrt(mu) * tof;
  rarr = (1 - chi^2 * cs[1] / g[1]) * r1v + (tof - chi^3 * cs[2] / sqrt(mu)) * v1;
  check("alpha chi^2 = z", abs(alpha * chi^2 - z) < 10^-80 * abs(z));
  check("time equation", abs(kep) < 10^-80 * sqrt(mu) * tof);
  check("arrival", sqrt((rarr - r2v) * (rarr - r2v)~) < 10^-80 * g[2]);
  printf("departure 46, arrival 62, retrograde: z_min %.6f, z %.17g, |v1| %.6f m/s, C3 %.25g m^2/s^2\n",
         zmin, z, sqrt(v1 * v1~), c3);
  printf("departure 46, arrival 62, retrograde: v1 = [%.25g, %.25g, 0] m/s\n", v1[1], v1[2]);
}
