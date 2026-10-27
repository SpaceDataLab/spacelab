/* Second engine for derivations/sar_range_doppler.mac, at 100 digits.

   1. The sinc constants by PARI's own root finder: the half-power root of sin(pi x)/(pi x) = 1/sqrt 2
      (the -3 dB width is twice it, in units of 1/B) and the first sidelobe, the root of
      tan(pi x) = pi x in (1, 3/2), with its level in dB.
   2. K_src without the closed form: the range-frequency curvature of the exact 2-D spectrum phase
      -4 pi R0 sqrt((f0 + ft)^2 - c^2 f_eta^2/(4 V^2))/c, by numerical differentiation at ft = 0,
      is +2 pi/K_src. Evaluated for the three wide-migration cases the sar.rs SRC gate uses (migration
      factor 0.75, 0.50, -0.90), against the closed form 2 V^2 (f0 D)^3/(R0 c f_eta^2).
   3. The discrete matched filter: for a 200-sample chirp, the lag sums computed term by term against
      the Dirichlet kernel |sin(pi K m dt^2 (N - m))/sin(pi K m dt^2)| at every lag.

       timeout 300 gp -q -f derivations/sar_range_doppler.gp </dev/null                            */

default(realprecision, 100);

x0 = solve(x = 0.3, 0.6, sin(Pi * x) / (Pi * x) - 1 / sqrt(2));
x1 = solve(x = 1.2, 1.49, tan(Pi * x) - Pi * x);
printf("half-power width x B = 2 x0 = %.40g\n", 2 * x0);
printf("first sidelobe at x = %.40g, level %.40g dB\n", x1, 20 * log(abs(sin(Pi * x1) / (Pi * x1))) / log(10));

c = 299792458;
{
my(cases = [[3/10, 10000, 2000, 500000], [3/50, 5000, 300, 800000], [1/4, -7200, 1000, 250000]],
   worst = 0);
for (i = 1, #cases,
  my(lam = cases[i][1], f = cases[i][2], V = cases[i][3], R0 = cases[i][4], f0 = c / lam,
     curv, ksrc, D = sqrt(1 - (lam * f / (2 * V))^2));
  curv = derivnum(t = 0, -4 * Pi * R0 * sqrt((f0 + t)^2 - c^2 * f^2 / (4 * V^2)) / c, 2);
  ksrc = 2 * Pi / curv;
  worst = max(worst, abs(ksrc / (2 * V^2 * (f0 * D)^3 / (R0 * c * f^2)) - 1));
  printf("K_src(lambda = %s, f_eta = %d, V = %d, R0 = %d) = %.25g\n", lam, f, V, R0, ksrc));
printf("spectrum curvature vs the closed-form K_src: worst relative gap %.3e\n", worst);
}

{
my(N = 200, K = 37 * 10^12, dt = 10^-8, h = (N - 1) / 2, worst = 0, cc);
cc = vector(N, i, exp(I * Pi * K * dt^2 * (i - 1 - h)^2));
for (m = 1, N - 1,
  my(s = sum(i = 1, N - m, cc[i + m] * conj(cc[i])), x = Pi * K * m * dt^2);
  worst = max(worst, abs(abs(s) - abs(sin(x * (N - m)) / sin(x)))));
printf("matched-filter lag sums vs the Dirichlet kernel, N = 200: worst gap %.3e\n", worst);
}
