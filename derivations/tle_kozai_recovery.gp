\\ Second engine for tle_kozai_recovery.mac: the recipe and the relation at 60 digits, and one element
\\ set's semi-major axis for tools/test_lifetime_campaign.py.
\\ Run: timeout 120 gp -q -f derivations/tle_kozai_recovery.gp </dev/null
default(realprecision, 60);

recipe(a1, S) = {
  my(d1 = S / a1^2, a0 = a1 * (1 - d1 / 3 - d1^2 - 134 / 81 * d1^3));
  a0 / (1 - S / a0^2);
}

{
  my(worst = 0);
  foreach([1e-4, 3e-4, 1e-3, 1.5e-3], d,
    my(truth = solve(x = 0.5, 2, x * (1 + d / x^2)^(-2/3) - 1), r = recipe(1, d));
    my(scaled = (r - truth) / (4 / 3 * d^2));
    worst = max(worst, abs(scaled - 1));
    printf("d %.4g recipe - relation %.6g, against (4/3) d^2: %.6f\n", d, r - truth, scaled));
  if (worst > 0.01, error("not (4/3) d^2 to leading order"));

  \\ WGS-72, n = 15.5 rev/day, e = 0.001, i = 51.6 deg: pinned in the tool's test.
  my(xke = 60 / sqrt(6378.135^3 / 398600.8), k2 = 0.001082616 / 2);
  my(n0 = 15.5 * 2 * Pi / 1440, e = 1 / 1000, i = 516 / 10 * Pi / 180);
  my(S = 3 / 2 * k2 * (3 * cos(i)^2 - 1) / (1 - e^2)^(3 / 2), a1 = (xke / n0)^(2 / 3));
  printf("a'' = %.40g m\n", recipe(a1, S) * 6378135);
}
quit;
