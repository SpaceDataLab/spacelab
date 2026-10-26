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

       timeout 120 gp -q -f derivations/keplerian_elements.gp </dev/null                          */

default(realprecision, 60);

dbl(x) = {
  my(r = x * 1., e);
  if (r == 0, return(0));
  e = exponent(r);
  round(x * 2^(52 - e)) * 2^(e - 52);
}

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
