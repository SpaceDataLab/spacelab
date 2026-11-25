\\ Second engine for outward_rounding.mac: the hand lemmas behind the outward-rounded enclosures of
\\ `crates/spacelab-constellation` (plan 18 Ф28.V), re-checked at 60 digits. Where a lemma bounds a
\\ chain of double operations, the chain is replayed exactly: every operation rounded to the nearest
\\ binary64 by binary64.gp's f64, the exact value kept beside it as a rational, on seeded random and
\\ edge inputs. Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/outward_rounding.gp </dev/null
default(realprecision, 60);
default(path, "derivations:.");
read("binary64.gp");
setrand(20261125);
u = 2^-53; eta64 = 2^-1074;
chk(name, bad) = printf("%s: %s\n", name, if(bad == 0, 0, Str("FAIL ", bad)));
\\ a double in [-r, r], log-uniform in magnitude over 2^-k .. r, either sign
rnd(r, k) = f64((2*random(2) - 1) * r * 2^(-random(k*1000)/1000.));
dot3(a, b) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
crs(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
len(v) = sqrt(dot3(v, v));
unitv(v) = v / len(v);
ang(a, b) = arg(dot3(a, b) + I*len(crs(a, b)));

\\ 1. libm_iv: c/(1 - c 2^-52) <= c + 2^-40 and c 2^-1074/(1 - c 2^-52) <= 2^-1072, on the rows used
{
bad = 0;
foreach([27/32, 25/16, 15/16, 2], c,
  if(c/(1 - c*2^-52) > c + 2^-40 || c*eta64/(1 - c*2^-52) > 2^-1072, bad++));
chk("libm_iv_slack", bad);
}

\\ 2. PI is the double nearest pi, below it; the next double is above it
pid = 884279719003555/2^48;
chk("PI_nearest", (f64(Pi) != pid) + !(pid < Pi) + !(pid + 2^-51 > Pi));

\\ 3. The error-free residuals: two-sum, fma of a product, a quotient's and a square root's
\\ remainders are exact away from underflow, so their signs say on which side the exact value lies
{
bad = 0;
for(i = 1, 20000,
  my(a = rnd(4, 60), b = rnd(4, 60), s, bb, e, p, q, r, sq);
  s = f64(a + b); bb = f64(s - a); e = f64(f64(a - f64(s - bb)) + f64(b - bb));
  if(e != a + b - s, bad++);
  p = f64(a*b); if(abs(p) >= 2^-900 && f64(a*b - p) != a*b - p, bad++);
  q = f64(a/b); if(abs(a) >= 2^-900 && abs(q) >= 2^-900 && f64(a - q*b) != a - q*b, bad++);
  r = abs(a); sq = f64(sqrt(r)); if(r >= 2^-900 && f64(r - sq^2) != r - sq^2, bad++));
chk("error_free_residuals", bad);
}

\\ 4. atan t within |t|^3/3 of t, and |t|^3/3 <= |t| 2^-100 below 2^-50
{
bad = 0;
foreach([2^-50, 2^-60, 1e-3, 0.5, 1, 3], t, if(abs(atan(t) - t) > t^3/3, bad++));
if((2^-50)^2/3 > 2^-100, bad++);
chk("atan_series", bad);
}

\\ 5. The k-fold integrand -cos rho + (cos rho + A)/(p + q cos t), p = 1 + A cos rho, q = B sin rho,
\\ is at most 2 sin rho / kappa, kappa = |pi - acos A - rho| the pole's antipode's distance
{
bad = 0;
for(i = 1, 20000,
  my(A = 2*random(1.) - 1, rho = Pi*random(1.), t = 2*Pi*random(1.), B, p, q, kap, f);
  B = sqrt(1 - A^2); p = 1 + A*cos(rho); q = B*sin(rho); kap = abs(Pi - acos(A) - rho);
  if(kap > 1e-6, f = -cos(rho) + (cos(rho) + A)/(p + q*cos(t)); if(abs(f) > 2*sin(rho)/kap, bad++)));
chk("integrand_lipschitz", bad);
}

\\ 6. Two arcs of a circle: the symmetric difference is at most 2(|dc| + |dh|), dc on the circle
arc_meet(c1, h1, c2, h2) = sum(k = -1, 1, max(0, min(c1 + h1, c2 + h2 + 2*Pi*k) - max(c1 - h1, c2 - h2 + 2*Pi*k)));
{
bad = 0;
for(i = 1, 20000,
  my(c1 = 2*Pi*random(1.) - Pi, h1 = Pi*random(1.), c2, h2, dc, sd);
  c2 = c1 + (random(1.) - 1/2) * 2^(-random(20)); h2 = min(Pi, max(0, h1 + (random(1.) - 1/2) * 2^(-random(20))));
  c2 = c2 - 2*Pi*round(c2/(2*Pi));
  dc = vecmin(vector(5, k, abs(c1 - c2 - 2*Pi*(k - 3))));
  sd = 2*h1 + 2*h2 - 2*arc_meet(c1, h1, c2, h2);
  if(sd > 2*(dc + abs(h1 - h2)) + 1e-50, bad++));
chk("arc_symmetric_difference", bad);
}

\\ 7. The band within delta of a circle of radius rho is at most 4 pi delta min(1, sin rho + delta)
{
bad = 0;
for(i = 1, 20000,
  my(rho = Pi*random(1.), d = 2^(-random(40)) * random(1.), area);
  area = 2*Pi*(cos(max(0, rho - d)) - cos(min(Pi, rho + d)));
  if(area > 4*Pi*d*min(1, sin(rho) + d), bad++));
chk("band_bound", bad);
}

\\ 8. asin z <= (pi/2) z on [0, 1]; a chord's angle 2 asin(c/2) <= (pi/2) c; asin(u + 2^-1070) <= 2u
{
bad = 0;
for(i = 0, 1000, my(z = i/1000); if(asin(z) > Pi/2*z + 1e-55, bad++));
for(i = 0, 1000, my(c = 2*i/1000); if(2*asin(c/2) > Pi/2*c + 1e-55, bad++));
if(asin(u + 2^-1070) > 2*u, bad++);
chk("asin_bounds", bad);
}

\\ 9. The length of a double vector: n = RN(sqrt(RN(RN(RN(x^2) + RN(y^2)) + RN(z^2)))) is within
\\ 2.6u of |v| once n >= 2^-449, components of every magnitude, tiny ones included
{
bad = 0; worst = 0;
for(i = 1, 20000,
  my(v = vector(3, j, rnd(4, if(i % 3, 30, 600))), s, n, N);
  if(i % 7 == 0, v[1] = 0);
  s = f64(f64(f64(v[1]^2) + f64(v[2]^2)) + f64(v[3]^2)); n = f64(sqrt(s)); N = len(v);
  if(n >= 2^-449, worst = max(worst, abs(n - N)/N/u); if(abs(n - N) > 2.6*u*N, bad++);
     if(N > n*(1 + 4*u) || N < n*(1 - 4*u), bad++)));
chk("length_within_2_6u", bad);
printf("length_worst_relative_error_in_u = %.4f\n", worst);
}

\\ 10. The cross product's running bound (analysis.rs cross_rounded) covers its rounding
{
bad = 0;
for(i = 1, 20000,
  my(a = vector(3, j, rnd(2, 40)), b = vector(3, j, rnd(2, 40)), c = vector(3), bound = 0, ex);
  for(j = 1, 3,
    my(i1 = j % 3 + 1, i2 = (j + 1) % 3 + 1, t1 = f64(a[i1]*b[i2]), t2 = f64(a[i2]*b[i1]));
    c[j] = f64(t1 - t2); bound += u*(abs(c[j]) + abs(t1) + abs(t2)));
  bound += 6*eta64; ex = crs(a, b);
  if(len(c - ex) > bound, bad++));
chk("cross_rounding_bound", bad);
}

\\ 11. The plain dot product of two vectors with components at most 1 + 2^-40 is within 7u of the
\\ exact one (Gappa's alpha_dot_roundoff.gappa), unit vectors and their edges
{
bad = 0; worst = 0;
for(i = 1, 50000,
  my(a = vector(3, j, rnd(1 + 2^-40, 50)), b = vector(3, j, rnd(1 + 2^-40, 50)), d);
  if(i % 2, a = apply(f64, unitv(a)); b = apply(f64, unitv(b)));
  d = f64(f64(f64(a[1]*b[1]) + f64(a[2]*b[2])) + f64(a[3]*b[3]));
  worst = max(worst, abs(d - dot3(a, b))/u);
  if(abs(d - dot3(a, b)) > 7*u, bad++));
chk("dot_within_7u", bad);
printf("dot_worst_error_in_u = %.4f\n", worst);
}

\\ 12. located: x = RN(v RN(1/RN|v|)) is within (pi/2) e/|v| + 2u of the exact direction of any V
\\ within e of v, e < |v|/2
{
bad = 0;
for(i = 1, 20000,
  my(v = vector(3, j, rnd(2, 20)), e, w, V, r, x, lam);
  e = len(v) * 2^(-1 - random(50)) * random(1.);
  w = vector(3, j, 2*random(1.) - 1); V = v + w * (e / len(w));
  r = f64(1/f64(sqrt(f64(f64(f64(v[1]^2) + f64(v[2]^2)) + f64(v[3]^2)))));
  x = vector(3, j, f64(v[j]*r));
  lam = Pi/2 * e/len(v) + 2*u;
  if(ang(x, V) > lam, bad++));
chk("located_bound", bad);
}

\\ 13. A Kepler ellipse: dnu/dE <= sqrt((1 + e)/(1 - e)); E from nu inverts nu from E
{
bad = 0;
for(i = 1, 5000,
  my(e = 0.999*random(1.), E = 2*Pi*random(1.), nu, rate, back);
  nu = 2*atan(sqrt((1 + e)/(1 - e))*tan(E/2));
  rate = sqrt(1 - e^2)/(1 - e*cos(E));
  if(rate > sqrt((1 + e)/(1 - e)) * (1 + 1e-50), bad++);
  back = arg((e + cos(nu)) + I*sqrt(1 - e^2)*sin(nu));
  if(abs(sin(back - E)) > 1e-40, bad++));
chk("kepler_inverse_and_rate", bad);
}
