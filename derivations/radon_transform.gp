/* Second engine for derivations/radon_transform.mac, at 60 digits, and the discretisation bounds
   the Gaussian gates in crates/spacelab-signal/src/radon.rs rest on.

   1. h[n] = integral_{-1/2}^{1/2} |nu| cos(2 pi nu n) dnu by quadrature, n = 0 .. 12, against the
      Ram-Lak closed form;
   2. integral_0^pi a b/(pi sig^2) dtheta = 1, sig^2 = a^2 cos^2 + b^2 sin^2, a = 3, b = 4;
   3. the Rust projection gate integrates a sampled Gaussian by summing it over unit-spaced rows:
      sum_j exp(-j^2/(2 b^2)) over |j| <= 32 against sqrt(2 pi) b, the Poisson-summation aliasing
      plus the truncation at the image edge, for b = 3, 4;
   4. the Rust FBP gate feeds the analytic sinogram of that Gaussian on 65 unit-spaced detectors
      and 60 angles k pi/60: the discrete ramp-filtered value at s = 0, sum_j p(j) h[-j], against
      a b/(pi sig^2) at every angle, and the equispaced angle mean (1/60) sum a b/sig^2 against 1.
   3 and 4 are reported as the worst relative gap, which must sit far below the 1e-12 the gates
   allow the f64 code.

       timeout 300 gp -q -f derivations/radon_transform.gp </dev/null                              */

default(realprecision, 60);

ramlak(n) = if (n == 0, 1/4, n % 2 == 0, 0, -1 / (Pi^2 * n^2));
{
my(off = 0);
for (n = 0, 12,
  if (abs(2 * intnum(f = 0, 1/2, f * cos(2 * Pi * f * n)) - ramlak(n)) > 10^-50, off++));
printf("ram_lak lags 0 .. 12: %d of 13 off by more than 1e-50\n", off);
}

a = 3; b = 4;
sig2(t) = a^2 * cos(t)^2 + b^2 * sin(t)^2;
printf("fbp angle integral - 1: %.3e\n", intnum(t = 0, Pi, a * b / (Pi * sig2(t))) - 1);

{
my(w = 0);
foreach ([3, 4], sg,
  w = max(w, abs(sum(j = -32, 32, exp(-j^2 / (2 * sg^2))) / (sqrt(2 * Pi) * sg) - 1)));
printf("row sum of a unit-spaced Gaussian, sigma 3 and 4: worst relative gap %.3e\n", w);
}

{
my(n = 60, wr = 0, mean = 0);
for (k = 0, n - 1,
  my(t = k * Pi / n, s2 = sig2(t), p(j) = sqrt(2 * Pi) * a * b / sqrt(s2) * exp(-j^2 / (2 * s2)),
     filt = sum(j = -32, 32, p(j) * ramlak(-j)));
  wr = max(wr, abs(filt / (a * b / (Pi * s2)) - 1));
  mean += a * b / s2 / n);
printf("ramp-filtered analytic sinogram at s = 0: worst relative gap %.3e\n", wr);
printf("equispaced angle mean (1/60) sum a b/sig^2 - 1: %.3e\n", mean - 1);
}
