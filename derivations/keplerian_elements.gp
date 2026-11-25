/* Kepler's equation and the element-to-state map at 60 digits: the second engine for
   derivations/keplerian_elements.mac and the reference values that
   crates/spacelab-elements/tests/keplerian.rs asserts the shipped f64 code against.

   1. Kepler's equation M = E - e sin E is solved by bracketed root finding on [M - e, M + e], where
      E - e sin E - M changes sign because |E - M| = |e sin E| <= e. Every root is substituted back
      and the worst residual printed. Each `kepler:` row is (e, M, nu): nu = atan2(sqrt(1 - e^2)
      sin E, cos E - e) in [0, 2 pi), the true anomaly `at_mean_anomaly` returns. Eccentricities are
      dyadic (exact doubles) from 0 to 1 - 2^-16, plus the double nearest 0.8, where the solver's
      starting guess switches; mean anomalies are halves from 0 to 6, 2^-20, 2^-10 and 201/32.

   2. Each `state:` row is (a, e, i, Omega, omega, nu, r, v): dyadic elements and the state of
      r = p/(1 + e cos nu) (cos nu, sin nu, 0), v = sqrt(mu/p) (-sin nu, e + cos nu, 0) rotated by
      R3(-Omega) R1(-i) R3(-omega), mu = 3.986004418e14 exactly.

   3. Each `decade:` row is the decade table that crates/spacelab-elements/src/keplerian.rs asserts
      its three anomaly maps against, at 100 digits (described at the section).

   4. keplerian_elements.mac section 6, recomputed at 100 digits on the decade table's grid.

       timeout 120 gp -q -f derivations/keplerian_elements.gp </dev/null                          */

default(realprecision, 60);

default(path, "derivations:."); read("binary64.gp"); dbl = f64;

fmt(v) = if (abs(v) < 10^-50, "0.0", concat([c | c <- Vec(Strprintf("%.25g", v)), c != " "]));

kepler_E(e, M) = if (e == 0, M * 1., solve(E = M - e, M + e, E - e * sin(E) - M));

wrap(x) = x - 2 * Pi * floor(x / (2 * Pi));

eccs = [0, 1/16, 5/16, dbl(4/5), 13/16, 243/256, 1013/1024, 1023/1024, 65535/65536];
means = concat([vector(13, k, (k - 1) / 2), [2^-20, 2^-10, 201/32]]);
worst = 0;
{
  for (i = 1, #eccs, for (j = 1, #means,
    my(e = eccs[i], M = means[j], E = kepler_E(e, M), nu);
    worst = max(worst, abs(E - e * sin(E) - M));
    nu = wrap(arg((cos(E) - e) + I * sqrt(1 - e^2) * sin(E)));
    printf("kepler:     (%s, %s, %s),\n", fmt(e * 1.), fmt(M * 1.), fmt(nu))));
}
printf("kepler residual |E - e sin E - M|, worst over %d roots: %.3g\n", #eccs * #means, worst);

mu = 398600441800000;
rot1(t) = [1, 0, 0; 0, cos(t), sin(t); 0, -sin(t), cos(t)];
rot3(t) = [cos(t), sin(t), 0; -sin(t), cos(t), 0; 0, 0, 1];
state(a, e, inc, W, w, nu) = {
  my(p = a * (1 - e^2), r = p / (1 + e * cos(nu)), s = sqrt(mu / p),
     R = rot3(-W) * rot1(-inc) * rot3(-w));
  [R * [r * cos(nu), r * sin(nu), 0]~, R * [-s * sin(nu), s * (e + cos(nu)), 0]~];
}

/* LEO near-SSO, GTO, Molniya, near-GEO, retrograde, e = 15/16 near perigee, near-polar. */
{els = [
  [7000000, 1/1024, 55/32, 1/2, 5/4, 2],
  [24500000, 187/256, 1/8, 3, 4, 1/4],
  [26600000, 23/32, 1133/1024, 5, 151/32, 3],
  [42164000, 1/4096, 1/1024, 1, 2, 11/2],
  [7500000, 1/8, 11/4, 3/4, 1/2, 3/2],
  [10000000, 15/16, 1/2, 5/2, 6, 1/32],
  [8000000, 1/4, 201/128, 9/2, 13/4, 17/4]
];}
{
  for (i = 1, #els,
    my(k = els[i], s = state(k[1], k[2], k[3], k[4], k[5], k[6]));
    printf("state:      (%s, %s, %s, %s, %s, %s, [%s, %s, %s], [%s, %s, %s]),\n",
           fmt(k[1] * 1.), fmt(k[2] * 1.), fmt(k[3] * 1.), fmt(k[4] * 1.), fmt(k[5] * 1.),
           fmt(k[6] * 1.), fmt(s[1][1]), fmt(s[1][2]), fmt(s[1][3]), fmt(s[2][1]), fmt(s[2][2]),
           fmt(s[2][3])));
}

/* 3. The near-parabolic decade table, at 100 digits. Each `decade:` row is (e, x, E(M = x),
      M(E = x), E(nu = x)): the eccentric anomaly solving Kepler's equation at the mean anomaly x,
      the mean anomaly of the eccentric anomaly x, and the eccentric anomaly of the true anomaly x,
      all in [0, 2 pi). e runs over 0, 1/2, the doubles nearest 0.9 and 0.99, and the doubles
      nearest 1 - 10^-k for k = 3, 4, 5, 6, 8, 10, 12; x over the doubles nearest 1e-12 to 0.1 at
      perigee, 1/2, 1, 2, 3, pi - 1e-6 at apoapsis, 7/2, 5, and 2 pi - 0.1, 2 pi - 1e-3,
      2 pi - 1e-6 on the far side of perigee. */
default(realprecision, 100);
{
  decs = concat([0, 1/2, dbl(9/10), dbl(99/100)], [dbl(1 - 10^-k) | k <- [3, 4, 5, 6, 8, 10, 12]]);
  xs = [dbl(10^-12), dbl(10^-9), dbl(10^-6), dbl(10^-4), dbl(10^-3), dbl(10^-2), dbl(1/10), 1/2, 1,
        2, 3, dbl(Pi - 10^-6), 7/2, 5, dbl(2 * Pi - 1/10), dbl(2 * Pi - 10^-3), dbl(2 * Pi - 10^-6)];
  dworst = 0;
  for (i = 1, #decs, for (j = 1, #xs,
    my(e = decs[i], x = xs[j], Em, Me, En);
    Em = if (e == 0, x, solve(E = x - e, x + e, E - e * sin(E) - x));
    dworst = max(dworst, abs(Em - e * sin(Em) - x));
    Me = wrap(x - e * sin(x));
    En = wrap(2 * arg(sqrt(1 + e) * cos(x / 2) + I * sqrt(1 - e) * sin(x / 2)));
    printf("decade:     (%s, %s, %s, %s, %s),\n", fmt(e * 1.), fmt(x * 1.), fmt(Em), fmt(Me),
           fmt(En))));
  printf("decade residual |E - e sin E - M|, worst over %d roots: %.3g\n", #decs * #xs, dworst);
}

/* 4. The identities of keplerian_elements.mac section 6, at 100 digits on a grid of e and angle:
      the fold, the rearranged residual and slope, the nested x - sin x series for |x| <= 1 (to its
      truncation, 2.5e-22 relative), the maximum of sin E/(1 - e cos E), and the half-angle form. */
{
  my(worst = 0, wser = 0, wmax = 0);
  for (i = 1, #decs, for (j = 1, #xs,
    my(e = decs[i], x = xs[j], c, s, q = 1, ser, Ec, Eh);
    worst = max(worst, abs(((2 * Pi - x) - e * sin(2 * Pi - x)) - (2 * Pi - (x - e * sin(x)))));
    worst = max(worst, abs((1 - e) * sin(x) + (x - sin(x)) - (x - e * sin(x))));
    worst = max(worst, abs((1 - e) + 2 * e * sin(x / 2)^2 - (1 - e * cos(x))));
    if (x <= 1,
      foreach([20, 18, 16, 14, 12, 10, 8, 6, 4], n, q = 1 - x^2 * q / (n * (n + 1)));
      ser = x^3 / 6 * q;
      wser = max(wser, abs(ser - (x - sin(x))) / (x - sin(x))));
    c = sqrt(1 + e) * cos(x / 2); s = sqrt(1 - e) * sin(x / 2);
    Eh = wrap(2 * arg(c + I * s));
    Ec = wrap(arg((e + cos(x)) + I * sqrt(1 - e^2) * sin(x)));
    worst = max(worst, min(abs(Eh - Ec), 2 * Pi - abs(Eh - Ec)));
    if (e > 0 && e < 1,
      wmax = max(wmax, abs(sin(acos(e)) / (1 - e * cos(acos(e))) - 1 / sqrt(1 - e^2))))));
  if (worst > 10^-90 || wmax > 10^-90 || wser > 25 / 10^23,
    error("a section-6 identity fails: ", [worst, wmax, wser]));
  printf("section 6 identities, worst |difference| %.3g; x - sin x series, worst relative %.3g; max of sin E/(1 - e cos E), worst %.3g\n",
         worst, wser, wmax);
}
