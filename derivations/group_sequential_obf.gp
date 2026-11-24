\\ Second engine for group_sequential_obf.mac: the group-sequential rule of SDTM-1's blind test
\\ (docs/sdtm1-blind-protocol.md §9 A2; plans/18 Ф30.N), at 30 digits.
\\
\\ Z_k = B(t_k) / sqrt(t_k) for a standard Brownian motion B and information fractions t1 < t2 < t3 = 1:
\\ the canonical joint law of a cumulative mean over clusters with independent increments. The crossing
\\ probability is 1 minus a nested integral over the first two looks, the third in closed form.
\\ Run: timeout 300 gp -q -f derivations/group_sequential_obf.gp </dev/null

default(realprecision, 30);
Phi(x) = 1 - erfc(x / sqrt(2)) / 2;
phi(x) = exp(-x^2 / 2) / sqrt(2 * Pi);
cross(z1, z2, z3, t1, t2) = {
  my(b1 = z1 * sqrt(t1), b2 = z2 * sqrt(t2), s12 = sqrt(t2 - t1), s23 = sqrt(1 - t2));
  1 - intnum(x = -b1, b1, phi(x / sqrt(t1)) / sqrt(t1)
        * intnum(y = -b2, b2, phi((y - x) / s12) / s12 * (Phi((z3 - y) / s23) - Phi((-z3 - y) / s23))));
}
\\ Two looks at t1 < 1, for the spend of the second look.
cross2(z1, z2, t1) = {
  my(b1 = z1 * sqrt(t1), s12 = sqrt(1 - t1));
  1 - intnum(x = -b1, b1, phi(x / sqrt(t1)) / sqrt(t1) * (Phi((z2 - x) / s12) - Phi((-z2 - x) / s12)));
}
check(name, ok) = if (!ok, error(name), print(name, ": ok"));
two_sided(z) = 2 * (1 - Phi(z));

{
  \\ (c) The O'Brien-Fleming constant: bounds C sqrt(3/k), two-sided level 0.05 at t = (1/3, 2/3, 1).
  my(C = solve(c = 1.9, 2.1, cross(c * sqrt(3), c * sqrt(3/2), c, 1/3, 2/3) - 1/20));
  print("C = ", C);
  check("C rounds to 2.004036", round(C * 10^6) == 2004036);
  my(up = [ceil(C * sqrt(3) * 10^4), ceil(C * sqrt(3/2) * 10^4), ceil(C * 10^4)] / 10^4);
  my(down = [floor(C * sqrt(3) * 10^3), floor(C * sqrt(3/2) * 10^3), floor(C * 10^3)] / 10^3);
  print("bounds ", [C * sqrt(3), C * sqrt(3/2), C], ", rounded up ", up * 1.0, ", down ", down * 1.0);
  check("rounded up: 3.4711, 2.4545, 2.0041", up == [34711, 24545, 20041] / 10^4);
  check("rounded down: 3.471, 2.454, 2.004", down == [3471, 2454, 2004] / 10^3);
  my(level_down = cross(3.471, 2.454, 2.004, 1/3, 2/3));
  my(level_up = cross(3.4711, 2.4545, 2.0041, 1/3, 2/3));
  print("level, bounds rounded down ", level_down, ", rounded up ", level_up);
  check("level rounded down is 0.050012", round(level_down * 10^6) == 50012);
  check("level rounded up is 0.0499923", round(level_up * 10^7) == 499923);

  \\ The same fixed bounds at other information fractions. Over every (t1, t2) the level lies between
  \\ the last bound's own level (the looks coincide: the union bound is attained) and three
  \\ independent looks (Sidak's inequality is attained at zero correlation).
  my(p = [two_sided(3.4711), two_sided(2.4545), two_sided(2.0041)]);
  my(low = p[3], high = 1 - prod(k = 1, 3, 1 - p[k]));
  print("nominal two-sided levels ", p, "; range [", low, ", ", high, "]");
  check("first look's level is 0.00052", round(p[1] * 10^5) == 52);
  check("range low is 0.0451", round(low * 10^4) == 451);
  check("range high is 0.0590", round(high * 10^4) == 590);
  my(at = cross(3.4711, 2.4545, 2.0041, 1/10, 3/10));
  print("level at (1/10, 3/10) ", at);
  check("level at (1/10, 3/10) is 0.0554", round(at * 10^4) == 554);
  my(worst = [1, 0], v);
  for (i = 1, 4, for (j = i + 1, 5,
    v = cross(3.4711, 2.4545, 2.0041, i / 6, j / 6);
    worst = [min(worst[1], v), max(worst[2], v)]));
  print("fixed bounds on the (i/6, j/6) grid: ", worst);
  check("every grid level lies inside the range", low < worst[1] && worst[2] < high);

  \\ (a) The spend of the registered bounds at equal information, by look: what A2's rule may spend.
  my(a1 = p[1], a2 = cross2(3.4711, 2.4545, 1/2), a3 = level_up);
  print("SPEND ", [a1, a2, a3]);
  check("the spend rises", 0 < a1 && a1 < a2 && a2 < a3 && a3 < 1/20);

  \\ (b) Equal weights: the CR2 variance is s^2 / K and Satterthwaite's nu is K - 1.
  for (K = 2, 12,
    my(w = vector(K, j, 1 / K), s2 = norml2(w), c = vector(K, j, w[j]^2 / (1 - 2 * w[j] + s2)));
    my(B = matrix(K, K, j, k, (j == k) - w[j] - w[k] + s2));
    my(tr1 = sum(j = 1, K, c[j] * B[j, j]), tr2 = sum(j = 1, K, sum(k = 1, K, c[j] * c[k] * B[j, k]^2)));
    check(Str("tr A = sum w^2, K = ", K), tr1 == s2);
    check(Str("nu = K - 1, K = ", K), tr1^2 / tr2 == K - 1));
  \\ Unequal weights in exact rationals: tr A = sum w^2 (the CR2 variance is unbiased under its model).
  foreach([[3, 1, 4, 1, 5], [2, 7, 1, 8, 2, 8], [1, 1, 20, 20, 3, 3, 27]], n,
    my(K = #n, w = n / vecsum(n), s2 = norml2(w), c = vector(K, j, w[j]^2 / (1 - 2 * w[j] + s2)));
    my(B = matrix(K, K, j, k, (j == k) - w[j] - w[k] + s2));
    my(tr1 = sum(j = 1, K, c[j] * B[j, j]), tr2 = sum(j = 1, K, sum(k = 1, K, c[j] * c[k] * B[j, k]^2)));
    check(Str("tr A = sum w^2, sizes ", n), tr1 == s2);
    print("nu at sizes ", n, ": ", tr1^2 / tr2 * 1.0, " (K - 1 = ", K - 1, ")"));
}
