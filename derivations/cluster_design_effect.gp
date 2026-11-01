\\ Second engine for cluster_design_effect.mac: the covariance sum, the enumerated cluster bootstrap and
\\ the AR(1) factor, in exact rationals.
\\ Run: timeout 120 gp -q -f derivations/cluster_design_effect.gp </dev/null

{
  for (m = 2, 12, foreach([1/7, 1/3, 9/10, -1/(m - 1)], rho,
    my(v = matrix(m, m, i, j, if (i == j, 1, rho)));
    if (sum(i = 1, m, sum(j = 1, m, v[i, j])) != m * (1 + (m - 1) * rho), error("cluster sum, m = ", m));
    if (matdet(v) != (1 - rho)^(m - 1) * (1 + (m - 1) * rho), error("determinant, m = ", m))));
  foreach([[3, 1, 4, 1], [2, 7, 1, 8], [1/2, 1/3, 1/5, 1/7]], c,
    my(k = #c, cbar = vecsum(c) / k, first = 0, second = 0, n = k^k);
    forvec(t = vector(k, i, [1, k]),
      my(mean = sum(i = 1, k, c[t[i]]) / k);
      first += mean; second += mean^2);
    first /= n; second = second / n - first^2;
    if (first != cbar, error("bootstrap mean"));
    if (second != sum(i = 1, k, (c[i] - cbar)^2) / k^2, error("bootstrap variance")));
  foreach([1/10, 1/2, 9/10, 99/100], phi,
    for (m = 1, 40,
      my(direct = 1 + 2 * sum(h = 1, m - 1, (1 - h / m) * phi^h));
      my(closed = (1 + phi) / (1 - phi) - 2 * phi * (1 - phi^m) / (m * (1 - phi)^2));
      if (direct != closed, error("AR(1), m = ", m))));
  print("cluster_design_effect: pari agrees");
}
