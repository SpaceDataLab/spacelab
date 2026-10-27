/* Chi-square quantiles at 60 digits -- the exact values the consistency bounds of
   spacelab-estimation/src/consistency.rs and the gate of src/robust.rs are measured against, and
   the second engine for derivations/chi_square_quantiles.mac.

   `chi_square_mean_interval(dof, runs, z)` returns the Wilson-Hilferty cube
       WH(k, z) = k (1 - 2/(9k) + z sqrt(2/(9k)))^3,   k = dof runs,
   at -z and +z, divided by runs, as the Phi(-z) and Phi(z) quantiles of the mean of runs chi^2(dof)
   samples. Here:
   1. the exact quantile q(k, p), the root of P(k/2, x/2) = p, against Maxima's at the fixture
      points (same digits printed by both);
   2. WH(k, +-z) and q(k, Phi(+-z)) at the fixture points, at 25 digits, for the tests;
   3. the error envelope the module docs state: over k = 1..400 and z in {1.96, 2.576, 3} the
      largest |WH - q| on either side, and where it is reached;
   4. the lower bound WH(k, -z) is negative exactly when z > (1 - 2/(9k)) / sqrt(2/(9k)), which
      for k = 1 is z > 7/(3 sqrt 2) = 1.6499..., where the exact quantile is positive.
   Every check prints a count of departures; a clean run prints `0 off`.

       timeout 300 gp -q -f derivations/chi_square_quantiles.gp </dev/null                         */

default(realprecision, 60);

Phi(z) = 1 - erfc(z / sqrt(2)) / 2;
q(k, p) = solve(x = 10^-50, 20 * k + 200, 1 - incgam(k / 2, x / 2) / gamma(k / 2) - p);
wh(k, z) = my(a = 2 / (9 * k), t = 1 - a + z * sqrt(a)); k * t^3;

/* 2. The fixture points of the tests: (dof, runs, z). */
pts = [[1, 1, 1.96], [1, 1, 2.576], [1, 1, 3], [3, 1, 1.96], [3, 1, 2.576], [3, 1, 3], [2, 50, 1.96]];
{
for (i = 1, #pts,
  my(d = pts[i][1], n = pts[i][2], z = pts[i][3], k = d * n);
  printf("dof %d runs %d z %.3f: WH mean bounds (%.25g, %.25g), exact (%.25g, %.25g)\n", d, n, z,
    wh(k, -z) / n, wh(k, z) / n, q(k, 1 - Phi(z)) / n, q(k, Phi(z)) / n));
}

/* 1. The quantile is the root it claims to be: P(k/2, q/2) = p to the working precision. */
{
bad = 0;
for (i = 1, #pts, my(k = pts[i][1] * pts[i][2], z = pts[i][3]);
  foreach ([Phi(z), 1 - Phi(z)], p,
    if (abs(1 - incgam(k / 2, q(k, p) / 2) / gamma(k / 2) - p) > 10^-50, bad++)));
printf("quantiles that miss their defining equation by more than 1e-50: %d off\n", bad);
}

/* 3. The envelope. */
{
worst = 0; at = 0;
for (k = 1, 400,
  foreach ([1.96, 2.576, 3], z,
    my(e = max(abs(wh(k, -z) - q(k, 1 - Phi(z))), abs(wh(k, z) - q(k, Phi(z)))));
    if (e > worst, worst = e; at = [k, z])));
printf("largest |WH - exact| over k = 1..400, |z| <= 3: %.6g at k = %d, z = %.3f\n", worst, at[1], at[2]);
printf("envelopes above 0.27: %d off\n", worst > 0.27);
worst = 0;
for (k = 1, 400, my(z = 1.96);
  worst = max(worst, max(abs(wh(k, -z) - q(k, 1 - Phi(z))), abs(wh(k, z) - q(k, Phi(z))))));
printf("largest |WH - exact| over k = 1..400 at z = 1.96: %.6g\n", worst);
printf("envelopes above 0.10 at z = 1.96: %d off\n", worst > 0.10);
}

/* 4. The negative lower bound at one degree of freedom. */
zc = (1 - 2/9) / sqrt(2/9);
printf("k = 1 lower bound turns negative at z = %.25g (7/(3 sqrt 2) = %.25g)\n", zc, 7 / (3 * sqrt(2)));
{printf("sign of WH(1, -1.96): %d, of the exact quantile: %d -- departures from (-1, 1): %d off\n",
  sign(wh(1, -1.96)), sign(q(1, 1 - Phi(1.96))), (sign(wh(1, -1.96)) != -1) + (sign(q(1, 1 - Phi(1.96))) != 1));}
