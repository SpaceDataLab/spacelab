/* Second engine for derivations/gaussian_scores.mac: the normal CRPS's closed form against its defining
   integral, int (Phi(x) - H(x - z))^2 dx, and against E|X - z| - E|X - X'|/2, at 50 digits. Prints `PASS`
   lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/gaussian_scores.gp </dev/null                                     */

default(realprecision, 50);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
/* Phi and 1 - Phi each through erfc, so neither tail is a difference of nearly equal numbers. */
Phi(t) = erfc(-t/sqrt(2))/2;
Phic(t) = erfc(t/sqrt(2))/2;
phi(t) = exp(-t^2/2)/sqrt(2*Pi);
crps(z) = z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(Pi);
/* intnum on pieces of length at most 2: one Gauss rule over 40 units of a Gaussian tail loses digits. */
piece(f, a, b) = my(s = 0, n = max(1, ceil((b - a)/2))); for (k = 0, n - 1, s += intnum(x = a + k*(b - a)/n, a + (k + 1)*(b - a)/n, f(x))); s;
/* Beyond |x| = 40 every integrand below is under exp(-800). */
defining(z) = piece(x -> Phi(x)^2, -40, z) + piece(x -> Phic(x)^2, z, 40);
absdev(z) = piece(x -> (z - x)*phi(x), -40, z) + piece(x -> (x - z)*phi(x), z, 40);
{
  my(worst = 0, worst_abs = 0);
  foreach([-3.7, -2.3, -0.4, 0, 0.7, 1.9, 3.1, 5.5], z,
    worst = max(worst, abs(defining(z) - crps(z)));
    worst_abs = max(worst_abs, abs(absdev(z) - (z*(2*Phi(z) - 1) + 2*phi(z)))));
  check("1 E|X - z| = z (2 Phi(z) - 1) + 2 phi(z) at eight z, within 1e-45", worst_abs < 1e-45);
  check("3-4 the closed form is the defining integral at eight z, within 1e-45", worst < 1e-45);
}
{
  check("2 E|X - X'| = 2/sqrt(pi), X - X' being N(0, 2)",
    abs(2*piece(u -> u*exp(-u^2/4)/sqrt(4*Pi), 0, 60) - 2/sqrt(Pi)) < 1e-45);
  check("4 the closed form at 0 is (sqrt 2 - 1)/sqrt(pi)", abs(crps(0) - (sqrt(2) - 1)/sqrt(Pi)) < 1e-45);
  check("5 the closed form is even", abs(crps(-1.3) - crps(1.3)) < 1e-45);
}
/* The scaled score, sigma crps(r/sigma), against the defining integral of N(0, sigma^2) directly. */
{
  my(s = 0.27, r = -0.41, direct);
  direct = piece(x -> Phi(x/s)^2, -40*s, r) + piece(x -> Phic(x/s)^2, r, 40*s);
  check("the score scales: sigma crps(r/sigma) is N(0, sigma^2)'s integral", abs(direct - s*crps(r/s)) < 1e-45);
}
