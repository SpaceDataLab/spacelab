/* Second engine for derivations/tomosar_beamforming.mac, at 60 digits.

   1. The steering phase from the exact geometry: the double difference
      R(b, s) - R(b, 0) - R(0, s) + R(0, 0), R = sqrt(r^2 + (s - b)^2), against -b s/r for
      r = 600 km, baselines to 300 m and elevations to 60 m; the relative gap is printed divided by
      (b^2 + s^2)/r^2, so a bounded number shows the first-order term is exact to that order.
   2. Bartlett from its definition a^H R a/a^H a, R = a0 a0^H, N = 8 baselines b_n = n d, against the
      Dirichlet closed form at 200 elevations, and its value at the N - 1 Rayleigh nulls.
   3. Capon from its definition 1/(a^H (R + delta I)^-1 a), by PARI's own matsolve, against the
      Sherman-Morrison closed form delta/(N - |a^H a0|^2/(delta + N)) at the same elevations.

       timeout 300 gp -q -f derivations/tomosar_beamforming.gp </dev/null                          */

default(realprecision, 60);

{
my(r = 600000, worst = 0);
foreach ([20, 75, 300], b, foreach ([-60, 3, 17, 60], s,
  my(R(b, s) = sqrt(r^2 + (s - b)^2), dd = R(b, s) - R(b, 0) - R(0, s) + R(0, 0));
  worst = max(worst, abs(dd / (-b * s / r) - 1) / ((b^2 + s^2) / r^2))));
printf("steering double difference vs -b s/r: worst relative gap %.3f x (b^2 + s^2)/r^2\n", worst);
}

lam = 31/1000; r = 600000; d = 20; N = 8; s0 = 12; dl = 1/10;
steer(s) = vector(N, n, exp(-I * 4 * Pi * (n - 1) * d * s / (lam * r)))~;
a0 = steer(s0);
Rm = a0 * conj(a0~);
dirichlet(s) = my(u = 4 * Pi * d * (s - s0) / (lam * r)); if (abs(sin(u / 2)) < 10^-50, N, sin(N * u / 2)^2 / (N * sin(u / 2)^2));
{
my(wb = 0, wc = 0, wn = 0);
for (k = 0, 199,
  my(s = -40 + k * 0.53, a = steer(s), pb, pc, g);
  pb = real(conj(a~) * Rm * a) / real(conj(a~) * a);
  wb = max(wb, abs(pb - dirichlet(s)));
  pc = 1 / real(conj(a~) * matsolve(Rm + dl * matid(N), a));
  g = norm(conj(a~) * a0);
  wc = max(wc, abs(pc - dl / (N - g / (dl + N)))));
for (k = 1, N - 1,
  my(a = steer(s0 + k * lam * r / (2 * N * d)));
  wn = max(wn, abs(real(conj(a~) * Rm * a) / N)));
printf("bartlett vs the Dirichlet kernel: worst gap %.3e\n", wb);
printf("bartlett at the %d Rayleigh nulls: worst value %.3e\n", N - 1, wn);
printf("capon vs Sherman-Morrison: worst gap %.3e\n", wc);
}
