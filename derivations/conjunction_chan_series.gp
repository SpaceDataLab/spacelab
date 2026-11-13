/* Chan's equal-area series at 80 digits: the reference the published-case test of
   spacelab-estimation/src/conjunction.rs holds `probability_chan` to.

   Chan replaces the hard body by the circle of equal area in the whitened plane, which reduces Pc
   to a non-central chi-square law with two degrees of freedom:
       Pc_Chan = P(chi'^2_2(v) <= u),  u = R^2 / (sx sy),  v = ux^2/sx^2 + uy^2/sy^2,
   for the diagonal covariance diag(sx^2, sy^2), miss (ux, uy) and radius R. Two routes:
   a. the Poisson mixture the module sums, P(I > J) = sum_{i >= 1} Pois(i; u/2) P(J <= i - 1),
      J ~ Pois(v/2), in 80-digit arithmetic until a term leaves 78 digits;
   b. the Rice density integrated, int_0^u exp(-(x + v)/2) I_0(sqrt(v x)) / 2 dx.
   1. the two routes agree to 1e-50 on every case;
   2. every published value is the series rounded to the figures it is quoted at (Chan 1997/2008,
      Alfano 2005, two operational CSM/CDM cases): the published numbers are Chan's, not Pc, which
      `derivations/conjunction_pc.gp` integrates and which is 3.2 times the series on Alfano 3;
   3. the rows the test carries: label, the series.

       timeout 300 gp -q -f derivations/conjunction_chan_series.gp </dev/null                     */

default(realprecision, 80);

chan(ux, uy, sx, sy, R) = {
  my(b = R^2 / (2 * sx * sy), a = (ux^2 / sx^2 + uy^2 / sy^2) / 2);
  my(q = exp(-b), pj = exp(-a), cdf = 0, s = 0, i = 0, term);
  while (1,
    i++; q *= b / i; cdf += pj; pj *= a / i; term = q * cdf; s += term;
    if (i > b && term < 10^-78 * s, break));
  s;
}

rice(ux, uy, sx, sy, R) = {
  my(u = R^2 / (sx * sy), v = ux^2 / sx^2 + uy^2 / sy^2);
  intnum(x = 0, u, exp(-(x + v) / 2) * besseli(0, sqrt(v * x)) / 2);
}

figures(str) = {
  my(d = Vec(str), n = 0, seen = 0);
  for (i = 1, #d,
    if (d[i] == "e", break);
    if (d[i] >= "0" && d[i] <= "9" && (d[i] != "0" || seen), seen = 1; n++));
  n;
}
rounds_to(x, str) = {
  my(n = figures(str), e = floor(log(abs(x)) / log(10)) - n + 1);
  round(x / 10^e) == round(eval(str) / 10^e);
}

/* label, ux, uy, sx, sy, R, the published value of Chan's series (or "" where none). */
{
cases = [
  ["Chan 01", 0.0, 10.0, 25.0, 50.0, 5.0, "9.754e-3"],
  ["Chan 02", 10.0, 0.0, 25.0, 50.0, 5.0, "9.189e-3"],
  ["Chan 03", 0.0, 10.0, 25.0, 75.0, 5.0, "6.586e-3"],
  ["Chan 04", 10.0, 0.0, 25.0, 75.0, 5.0, "6.135e-3"],
  ["Chan 05", 0.0, 1000.0, 1000.0, 3000.0, 10.0, "1.577e-5"],
  ["Chan 06", 1000.0, 0.0, 1000.0, 3000.0, 10.0, "1.011e-5"],
  ["Chan 07", 0.0, 10000.0, 1000.0, 3000.0, 10.0, "6.443e-8"],
  ["Chan 08", 10000.0, 0.0, 1000.0, 3000.0, 10.0, "3.216e-27"],
  ["Chan 09", 0.0, 10000.0, 1000.0, 10000.0, 10.0, "3.033e-6"],
  ["Chan 10", 10000.0, 0.0, 1000.0, 10000.0, 10.0, "9.645e-28"],
  ["Chan 11", 0.0, 5000.0, 1000.0, 3000.0, 50.0, "1.039e-4"],
  ["Chan 12", 5000.0, 0.0, 1000.0, 3000.0, 50.0, "1.556e-9"],
  ["CSM 1", 84.875546, 60.583685, 57.918666, 152.8814468, 10.3, "1.8934e-3"],
  ["CDM 2", -692.362272, 4475.456261, 193.454603, 562.027293, 13.2, "2.2880e-20"],
  ["Alfano 3", -3.8872073, 0.1591646, 1.4101830, 114.2585190, 15.0, "3.1264e-2"],
  ["Alfano 5", -1.2217895, 2.1230067, 0.0373279, 177.8109003, 10.0, ""]];
}

/* A Rust literal with 17 significant digits. */
lit(x) = {
  my(e = floor(log(abs(x)) / log(10)), m = x / 10^e);
  if (abs(round(m * 10^16)) >= 10^17, e++; m = x / 10^e);
  strprintf("%.16fe%d", m, e);
}

/* Exact decimal inputs: the f64 the test parses is within half an ulp of these. */
num(x) = eval(Str(x));
{
my(rows = vector(#cases), routes = 0, published = 0);
for (k = 1, #cases,
  my(c = cases[k], ux = num(c[2]), uy = num(c[3]), sx = num(c[4]), sy = num(c[5]), R = num(c[6]));
  my(ch = chan(ux, uy, sx, sy, R), ri = rice(ux, uy, sx, sy, R));
  if (abs(ri / ch - 1) > 1e-50, routes++; printf("  %s: the two routes differ by %.3e\n", c[1], abs(ri / ch - 1)));
  if (c[7] != "" && !rounds_to(ch, c[7]), published++; printf("  %s: %s is not the series %.8g rounded\n", c[1], c[7], ch));
  rows[k] = strprintf("        (\"%s\", %s),", c[1], lit(ch)));
printf("(1) holds: Chan's series by the Poisson mixture and the Rice integral, %d off\n", routes);
printf("(2) holds: every published value is the series rounded, %d off\n", published);
print("(3) rows: label, Chan's series");
for (k = 1, #rows, print(rows[k]));
if (routes + published == 0, print("DONE"));
}
