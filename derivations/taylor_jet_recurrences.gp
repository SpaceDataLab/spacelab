/* The Taylor-mode jets of spacelab-propagation/src/taylor.rs: the second engine for
   derivations/taylor_jet_recurrences.mac, against PARI's own power-series arithmetic.

   Each recurrence the `Jet` implements is run here on two concrete jets of 12 coefficients and
   compared with PARI's series of the same function: exactly for the rational operations (product,
   quotient, reciprocal), to 1e-50 at \p 60 for sqrt, u^p, exp, log, sin, cos and atan2. Then the
   two solution series the Maxima file derives are re-derived by Picard iteration on power series,
   y <- y0 + int f(t, y) -- not by the bootstrap recurrence -- and must equal the rationals printed
   there: the planar two-body problem (mu = 1) from x = (3/5, 4/5), v = (-1/2, 1), and
   y' = cos(t) y from y(0) = 1. Any departure ends in error(), which prints ***.

   Run: timeout 300 gp -q -f derivations/taylor_jet_recurrences.gp </dev/null
*/

default(realprecision, 60);
M = 12;
co(s) = vector(M, k, polcoef(s, k - 1, 't));
ser(w) = sum(k = 1, M, w[k] * 't^(k - 1)) + O('t^M);
u = [7/5, 1/3, -2/7, 5/11, 1/13, -3/4, 2/9, 0, 1/17, -1/19, 4/23, 1/29];
v = [3/2, -1/5, 1/6, 2/7, -3/8, 1/9, 0, 5/12, -1/13, 1/14, 2/15, -1/16];

cmpx(name, w, s, tol) =
{
  my(r = co(s), d = vecmax(apply(abs, w - r)));
  if (d > tol, error(name, ": recurrence off the series by ", d));
}
/* Product, quotient, reciprocal: exact. */
prodj(a, b) = vector(M, k, sum(j = 0, k - 1, a[j + 1] * b[k - j]));
quotj(a, b) = { my(w = vector(M)); for (k = 0, M - 1, w[k + 1] = (a[k + 1] - sum(j = 1, k, b[j + 1] * w[k - j + 1])) / b[1]); w; }
recipj(a) = { my(w = vector(M)); w[1] = 1 / a[1]; for (k = 1, M - 1, w[k + 1] = -sum(j = 1, k, a[j + 1] * w[k - j + 1]) / a[1]); w; }
sqrtj(a) = { my(w = vector(M)); w[1] = sqrt(a[1]); for (k = 1, M - 1, w[k + 1] = (a[k + 1] - sum(j = 1, k - 1, w[j + 1] * w[k - j + 1])) / (2 * w[1])); w; }
powj(a, p) = { my(w = vector(M)); w[1] = a[1]^p; for (k = 1, M - 1, w[k + 1] = sum(j = 0, k - 1, (p * (k - j) - j) * a[k - j + 1] * w[j + 1]) / (k * a[1])); w; }
expj(a) = { my(w = vector(M)); w[1] = exp(a[1]); for (k = 1, M - 1, w[k + 1] = sum(j = 1, k, j * a[j + 1] * w[k - j + 1]) / k); w; }
lnj(a) = { my(w = vector(M)); w[1] = log(a[1]); for (k = 1, M - 1, w[k + 1] = (k * a[k + 1] - sum(j = 1, k - 1, j * w[j + 1] * a[k - j + 1])) / (k * a[1])); w; }
sincosj(a) =
{
  my(s = vector(M), c = vector(M));
  s[1] = sin(a[1]); c[1] = cos(a[1]);
  for (k = 1, M - 1,
    s[k + 1] = sum(j = 1, k, j * a[j + 1] * c[k - j + 1]) / k;
    c[k + 1] = -sum(j = 1, k, j * a[j + 1] * s[k - j + 1]) / k);
  [s, c];
}
derivj(a) = concat(vector(M - 1, k, k * a[k + 1]), [0]);
atan2j(y, x) =
{
  my(d = quotj(prodj(x, derivj(y)) - prodj(y, derivj(x)), prodj(x, x) + prodj(y, y)), w = vector(M));
  w[1] = atan(y[1] / x[1]);
  for (k = 1, M - 1, w[k + 1] = d[k] / k);
  w;
}
U = ser(u); V = ser(v);
if (prodj(u, v) != co(U * V), error("product"));
if (quotj(u, v) != co(U / V), error("quotient"));
if (recipj(u) != co(1 / U), error("reciprocal"));
TOL = 1e-50;
cmpx("sqrt", sqrtj(u), sqrt(U), TOL);
cmpx("powf(-3/2)", powj(u, -3/2), U^(-3/2), TOL);
cmpx("powf(1.7)", powj(u, 1.7), exp(1.7 * log(U)), TOL);
cmpx("exp", expj(u), exp(U), TOL);
cmpx("ln", lnj(u), log(U), TOL);
sc = sincosj(u);
cmpx("sin", sc[1], sin(U), TOL);
cmpx("cos", sc[2], cos(U), TOL);
cmpx("atan2", atan2j(u, v), atan(U / V), TOL);
print("jet recurrences: product, quotient, reciprocal exact; sqrt, powf, exp, ln, sin, cos, atan2 within 1e-50 of PARI's series");

/* Picard iteration on series through t^7. */
K = 8;
picard(f, y0) =
{
  my(y = vector(#y0, i, y0[i] + O('t^K)));
  for (it = 1, K, my(fy = f(y)); y = vector(#y0, i, y0[i] + intformal(fy[i], 't)));
  vector(#y0, i, vector(K, k, polcoef(y[i], k - 1, 't)));
}
kep(s) = my(r3 = (s[1]^2 + s[2]^2)^(-3/2)); [s[3], s[4], -s[1] * r3, -s[2] * r3];
tb = picard(kep, [3/5, 4/5, -1/2, 1]);
es = picard(s -> [cos('t) * s[1]], [1])[1];
/* The bootstrap coefficients derivations/taylor_jet_recurrences.mac prints. */
{
  maxima_tb = [[3/5, -1/2, -3/10, 7/30, -9/80, 1/30, 7/800, -149/6300],
               [4/5, 1, -2/5, 1/30, 7/120, -1/15, 167/3600, -1081/50400],
               [-1/2, -3/5, 7/10, -9/20, 1/6, 21/400, -149/900, 5989/33600],
               [1, -4/5, 1/10, 7/30, -1/3, 167/600, -1081/7200, 953/50400]];
}
maxima_es = [1, 1, 1/2, 0, -1/8, -1/15, -1/240, 1/90];
if (tb != maxima_tb, error("two-body: Picard ", tb, " vs bootstrap ", maxima_tb));
if (es != maxima_es, error("cos(t) y: Picard ", es, " vs bootstrap ", maxima_es));
if (es != vector(K, k, polcoef(exp(sin('t + O('t^K))), k - 1, 't)), error("exp(sin t) series"));
print("two-body and cos(t) y: Picard series through t^7 equal the bootstrap rationals");
quit;
