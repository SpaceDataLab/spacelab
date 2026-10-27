\\ Second engine for derivations/gas_k_distribution.mac:
\\  * the module test's transmittances, recomputed from the exponential sums at 50 digits, against
\\    Maxima's 40;
\\  * Jensen's sign for a merge, sampled: the transmittance lost by replacing two terms with their
\\    total weight at their weighted mean depth, w1 e^-t1 + w2 e^-t2 - W e^-tbar, over a grid of
\\    weights and depths, must never be negative.
default(realprecision, 50);
T(w, k, c) = sum(j = 1, #w, w[j]*exp(-k[j]*c));
bw = [55/100, 30/100, 12/100, 3/100]; bk = [4/1000, 9/100, 14/10, 26];
ow = [7/10, 3/10]; ok = [2/100, 5];
literals = [[T(bw, bk, 13/10*8/5), 0.8007503692183774790270023815564327570129], \
            [T(bw, bk, 1), 0.8515753853937098157215998250902171928408], \
            [T(bw, bk, 2)*T(ow, ok, 9/10), 0.555091411363238279191508539011546061539]];
{
  my(bad = 0);
  for (i = 1, #literals, if (abs(literals[i][1] - literals[i][2]) > 1e-39, bad++));
  printf("k-distribution transmittances: %d of %d off Maxima by more than 1e-39\n", bad, #literals);
}
{
  my(negative = 0, n = 0);
  for (a = 1, 19, my(w1 = a/20, w2 = 1 - a/20);
    for (i = 0, 20, for (j = 0, 20,
      my(t1 = i/2, t2 = j/2, tbar = (w1*t1 + w2*t2)/(w1 + w2));
      n++;
      if (w1*exp(-t1) + w2*exp(-t2) - (w1 + w2)*exp(-tbar) < -1e-45, negative++))));
  printf("merge loss: %d of %d sampled merges transmit more than the terms they replace\n", negative, n);
}
