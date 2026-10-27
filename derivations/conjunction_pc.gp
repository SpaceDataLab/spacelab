/* The short-encounter collision probability at 45 digits from its definition: the second engine for
   derivations/conjunction_pc.mac and the reference values the `probability_foster` /
   `probability_patera` gate in crates/spacelab-estimation/src/conjunction.rs asserts against.

       Pc = (1/(2 pi sqrt det C)) * double integral over x^2 + y^2 <= R^2 of exp(-d' C^-1 d / 2),
       d = (x, y) - m.

   Neither Foster's polar form nor Patera's contour is used. The inner integral over x at fixed y
   is the conditional normal of x given y -- mean mx + (c01/c11)(y - my), variance
   c00 - c01^2/c11 -- times the marginal normal of y, so it is an erfc pair; the outer integral over
   y = R sin(theta), theta in [-pi/2, pi/2], is smooth and is split into `PIECES` panels for
   intnum. A second, independent route checks the first: the same integral with the roles of x
   and y swapped (outer x, inner y). The two orders share nothing but the integrand.

   For a circular covariance C = s^2 I the integral has the exact Poisson-mixture form of
   derivations/conjunction_pc.mac,

       Pc = sum_k Pois(k; d^2/2s^2) * P(Pois(R^2/2s^2) >= k + 1),

   and for a centred one 1 - exp(-R^2/2s^2); both are checked against the quadrature.
   The inputs are the decimal literals the Rust test passes; their f64 rounding moves Pc by less
   than 1e-15 relative, far inside the test's tolerance.

       timeout 300 gp -q -f derivations/conjunction_pc.gp </dev/null                                */

default(realprecision, 60);
PIECES = 48;

/* Phi(z) = P(N(0,1) <= z), via erfc so the far tails keep their digits. */
Phi(z) = erfc(-z / sqrt(2)) / 2;

/* Outer variable u (axis iu), inner v (axis iv). */
pc_order(m, C, R, iu, iv) = {
  my(su2 = C[iu, iu], cov = C[iu, iv], sv2c = C[iv, iv] - cov^2 / C[iu, iu], tot = 0, a, b);
  my(f = (th) -> my(u = R * sin(th), w = R * cos(th), mc = m[iv] + cov / su2 * (u - m[iu]),
                    sd = sqrt(sv2c));
     exp(-(u - m[iu])^2 / (2 * su2)) / sqrt(2 * Pi * su2)
       * (Phi((w - mc) / sd) - Phi((-w - mc) / sd)) * R * cos(th));
  for (k = 0, PIECES - 1,
    a = -Pi / 2 + Pi * k / PIECES; b = -Pi / 2 + Pi * (k + 1) / PIECES;
    tot += intnum(t = a, b, f(t)));
  tot;
}

pc(m, C, R) = pc_order(m, C, R, 2, 1);
pc_swapped(m, C, R) = pc_order(m, C, R, 1, 2);

/* The circular-covariance Poisson mixture, summed until the terms leave 60 digits. */
pc_poisson(d, s, R) = {
  my(a = d^2 / (2 * s^2), b = R^2 / (2 * s^2), tot = 0, pj = exp(-a), cdf = 0, qi = exp(-b), i = 0);
  /* P(I > J) = sum_{i>=1} P(I = i) P(J <= i - 1). */
  while (1,
    cdf += pj; pj *= a / (i + 1);
    i++; qi *= b / i;
    tot += qi * cdf;
    if (i > b + 10 && qi * cdf < 10^-70 * tot, break));
  tot;
}

diag(sx, sy) = [sx^2, 0; 0, sy^2];
fmt(v) = Strprintf("%.40e", v);

/* (label, miss, covariance, radius) */
{
cases = [
  ["circular-centred", [0, 0], diag(50, 50), 20],
  ["circular-offset", [30, 40], diag(50, 50), 20],
  ["elongated-3-to-1", [10, 25], diag(30, 90), 15],
  ["grazing", [149/10, 0], diag(20, 60), 15],
  ["far-offset", [100, 0], diag(40, 120), 10],
  ["alfano-3", [-38872073/10^7, 1591646/10^7], diag(14101830/10^7, 1142585190/10^7), 15],
  ["alfano-5", [-12217895/10^7, 21230067/10^7], diag(373279/10^7, 1778109003/10^7), 10],
  ["chan-08-deep-tail", [10000, 0], diag(1000, 3000), 10],
  ["cdm-2-deep-tail", [-692362272/10^6, 4475456261/10^6], diag(193454603/10^6, 562027293/10^6), 132/10],
  ["mid-tail", [150, 200], diag(40, 90), 5],
  ["correlated", [30, -20], [900, 350; 350, 400], 10]
];
}

{
my(bad = 0);
print("Pc from the defining double integral, 40 digits shown:");
for (i = 1, #cases,
  my(c = cases[i], p = pc(c[2], c[3], c[4]), q = pc_swapped(c[2], c[3], c[4]), rel = abs(p - q) / p);
  if (rel > 10^-35, bad++);
  printf("  %-20s Pc = %s   |order swap|/Pc = %.1e\n", c[1], fmt(p), rel));
printf("%d of %d cases with the two integration orders apart by more than 1e-35\n", bad, #cases);
}

/* (miss distance, sigma, radius) of circular covariances, against the Poisson mixture. */
{
my(bad = 0, circ = [[0, 50, 20], [50, 50, 20], [1, 1, 1], [sqrt(1200), 1, sqrt(1200)]]);
for (i = 1, #circ,
  my(d = circ[i][1], s = circ[i][2], R = circ[i][3],
     p = pc([d, 0], diag(s, s), R), q = pc_poisson(d, s, R));
  if (abs(p - q) > 10^-35 * p, bad++);
  printf("  circular d=%.6g s=%.6g R=%.6g: quadrature %s, Poisson mixture %s\n", d, s, R, fmt(p), fmt(q)));
printf("%d circular cases with the Poisson mixture off the quadrature by more than 1e-35\n", bad);
my(p0 = pc([0, 0], diag(50, 50), 20));
printf("centred closed form 1 - exp(-R^2/2s^2) minus the quadrature: %.1e\n", 1 - exp(-400 / 5000) - p0);
}
