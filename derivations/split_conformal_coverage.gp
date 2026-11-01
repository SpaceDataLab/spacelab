\\ Second engine for split_conformal_coverage.mac: the Beta-binomial law in exact rationals, and the
\\ 5-95 % bands tools/lifetime-campaign.py prints, which tools/test_lifetime_campaign.py pins.
\\ Run: timeout 120 gp -q -f derivations/split_conformal_coverage.gp </dev/null

bb(x, m, a, b) = binomial(m, x) * (x + a - 1)! * (m - x + b - 1)! / (m + a + b - 1)! * (a + b - 1)! / ((a - 1)! * (b - 1)!);

band(m, n, alpha) = {
  my(k = ceil((n + 1) * (1 - alpha)), a = k, b = n + 1 - k, total = 0, lo = -1);
  for (x = 0, m,
    total += bb(x, m, a, b);
    if (lo < 0 && total >= 1/20, lo = x);
    if (total >= 19/20, return([lo, x])));
  [lo, m];
}

{
  foreach([[20, 9], [30, 19], [50, 29], [100, 49], [100, 99]], g,
    my(n = g[1], m = g[2], k = ceil((n + 1) * 9/10), a = k, b = n + 1 - k);
    if (sum(x = 0, m, bb(x, m, a, b)) != 1, error("the law does not sum to one"));
    if (sum(x = 0, m, x * bb(x, m, a, b)) != m * k / (n + 1), error("the mean is not m k/(n + 1)"));
    printf("n %d m %d k %d band %d %d\n", n, m, k, band(m, n, 1/10)[1], band(m, n, 1/10)[2]));
}
quit;
