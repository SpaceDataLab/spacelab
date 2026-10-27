\\ Second engine for derivations/particle_resampling.mac: the expected systematic-resampling counts,
\\ in exact rationals, from the definition. For each weight vector the count n_i(U) is piecewise
\\ constant in U with breakpoints frac(M C_i); on each piece it is read by counting the k whose
\\ comb point (k + U)/M lands in [C_(i-1), C_i), at the piece's midpoint. E[n_i] sums piece length
\\ times count and must equal M w_i. Also the residual scheme's expectation.
\\
\\ Run: timeout 300 gp -q -f derivations/particle_resampling.gp </dev/null
expected_systematic(w) =
{
  my(M = #w, C = vector(M + 1), cuts, e = vector(M));
  for (i = 1, M, C[i + 1] = C[i] + w[i]);
  cuts = vecsort(concat([0, 1], vector(M + 1, i, frac(M * C[i]))), , 8);
  for (s = 1, #cuts - 1,
    my(u = (cuts[s] + cuts[s + 1]) / 2, len = cuts[s + 1] - cuts[s]);
    for (k = 0, M - 1,
      my(pt = (k + u) / M);
      for (i = 1, M, if (pt >= C[i] && pt < C[i + 1], e[i] += len))));
  e;
}
expected_residual(w) =
{
  my(M = #w, fl = vector(M, i, floor(M * w[i])), r, R);
  r = vector(M, i, M * w[i] - fl[i]);
  R = M - vecsum(fl);
  if (vecsum(r) == 0, return(fl));
  vector(M, i, fl[i] + R * r[i] / vecsum(r));
}
cases = [[1/8, 3/16, 5/16, 1/4, 1/8], [1/10, 2/10, 4/10, 25/100, 5/100], [1/3, 1/7, 11/21, 0]];
off = 0; total = 0;
{
  for (c = 1, #cases,
    my(w = cases[c], M = #w, es = expected_systematic(w), er = expected_residual(w));
    for (i = 1, M,
      total += 2;
      if (es[i] != M * w[i], off++);
      if (er[i] != M * w[i], off++)));
}
printf("systematic and residual expected counts: %d of %d differ from M w_i\n", off, total);
printf("expected systematic counts of the Rust fixture: %s\n", expected_systematic(cases[1]));
quit;
