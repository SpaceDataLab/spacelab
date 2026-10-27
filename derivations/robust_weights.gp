/* Robust weights at 60 digits: the second engine for derivations/robust_weights.mac and the values
   `robust.rs::the_weights_are_the_roots_of_their_defining_equations` asserts the f64 code against.

   1. Huber: the root in (0, 1) of w^3 nu^2 - c^2 p w - c^2 R = 0, by polroots at the test's (p, R,
      nu, c) doubles, and the definition it stands for, w = c/|d| with d = nu/sqrt(p + R/w).
   2. Student-t: R_eff = f - p, f the positive root of (n + 1) f^2 - [(n + 1) p + n R] f - R nu^2,
      and the E-step it stands for, R/R_eff = (n + 1)/(n + nu^2/f).
   3. The Huber Gaussian efficiency at c = 1.345.
   4. The influence asymptotes at nu = 1e40: Huber K nu / nu^(1/3) -> p c^(2/3) R^(-2/3), Student-t
      K nu -> p sqrt((n + 1)/R).
   Every check prints a count of departures; a clean run prints `0 off`.

       timeout 300 gp -q -f derivations/robust_weights.gp </dev/null                               */

default(realprecision, 60);

f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
huber(p, r, nu, c) = {
  my(z = polroots(nu^2 * 'w^3 - c^2 * p * 'w - c^2 * r), w = 0);
  for (i = 1, #z, if (abs(imag(z[i])) < 10^-50 && real(z[i]) > 0, w = real(z[i])));
  w;
}
student(p, r, nu, n) = {
  my(a = n + 1, b = a * p + r * n);
  (b + sqrt(b^2 + 4 * a * r * nu^2)) / (2 * a) - p;
}

/* 1. */
{
H = [[4, 1, 10, f64(1.345)], [1/4, 4, 6, 1], [10, f64(0.1), 50, 5/2], [1/2, 1/4, 3, 1/2],
     [0, 1/4, 5, 1]];
bad = 0;
for (i = 1, #H,
  my([p, r, nu, c] = H[i], w = huber(p, r, nu, c), d = nu / sqrt(p + r / w));
  printf("huber p %.4g R %.4g nu %.4g c %.6g: w = %.25g\n", p, r, nu, c, w);
  if (!(w > 0 && w < 1) || abs(w - c / abs(d)) > 10^-50, bad++));
printf("huber roots off their definition by more than 1e-50: %d off\n", bad);
}

/* 2. */
{
T = [[4, 1, 5, 5], [1/4, 4, 100, 4], [1, 1/2, 0, 10], [f64(0.1), 2, 1000, 3]];
bad = 0;
for (i = 1, #T,
  my([p, r, nu, n] = T[i], e = student(p, r, nu, n), f = p + e);
  printf("student-t p %.4g R %.4g nu %.4g n %.4g: R_eff = %.25g\n", p, r, nu, n, e);
  if (abs(r / e - (n + 1) / (n + nu^2 / f)) > 10^-50, bad++));
printf("student-t variances off their E-step by more than 1e-50: %d off\n", bad);
}

/* 3. */
Phi(x) = 1 - erfc(x / sqrt(2)) / 2;
dens(x) = exp(-x^2 / 2) / sqrt(2 * Pi);
{
c = 1345 / 1000;
psi2 = intnum(x = -c, c, x^2 * dens(x)) + 2 * c^2 * (1 - Phi(c));
eff = (2 * Phi(c) - 1)^2 / psi2;
printf("huber Gaussian efficiency at c = 1.345: %.25g\n", eff);
printf("efficiency further than 5e-4 from 0.95: %d off\n", abs(eff - 0.95) > 5e-4);
}

/* 4. */
{
nu = 10^40; p = 4; r = 1; c = 2;
w = huber(p, r, nu, c);
k = p * nu / (p + r / w);
lim = p * c^(2/3) * r^(-2/3);
printf("huber K nu / nu^(1/3) at nu = 1e40: %.20g, limit %.20g\n", k / nu^(1/3), lim);
n = 5;
kt = p * nu / (p + student(p, r, nu, n));
printf("student-t K nu at nu = 1e40: %.20g, limit %.20g\n", kt, p * sqrt((n + 1) / r));
printf("asymptotes further than 1e-10 relative from their limits: %d off\n",
  (abs(k / nu^(1/3) / lim - 1) > 1e-10) + (abs(kt / (p * sqrt((n + 1) / r)) - 1) > 1e-10));
}
