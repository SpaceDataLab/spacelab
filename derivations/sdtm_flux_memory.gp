/* Second engine for derivations/sdtm_flux_memory.mac: the daily flux's first-order memory at 60 digits.
   Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_flux_memory.gp </dev/null                                      */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
near(a, b, tol) = abs(a - b) <= tol*max(1, abs(b));
rk4(f, y0, t, n) = {
  my(y = y0, s = t/n, k1, k2, k3, k4);
  for (i = 1, n, k1 = f(y); k2 = f(y + s/2*k1); k3 = f(y + s/2*k2); k4 = f(y + s*k3);
    y = y + s/6*(k1 + 2*k2 + 2*k3 + k4));
  y;
}
advance(G, F, h, tau) = F + (G - F)*exp(-h/tau);
/* (1): ten days of a varying daily flux, the recursion against RK4 on the same held-per-day flux. */
{
  my(F = [70.0, 72.0, 180.0, 95.0, 160.0, 210.0, 150.0, 90.0, 85.0, 80.0], tau = 18.0, G = 70.0, M = 70.0);
  for (k = 1, #F,
    G = advance(G, F[k], 24.0, tau);
    M = rk4(y -> (F[k] - y)/tau, M, 24.0, 4000));
  check("1 the daily recursion is exact for a flux held over each day", near(G, M, 1e-12));
}
{
  my(F = 123.4, G = 87.6);
  check("2 tau -> 0 returns the day's flux: equal at 60 digits, the remainder below 1e-10000",
    advance(G, F, 24.0, 1e-3) == F && abs((G - F)*exp(-24.0/1e-3)) < 1e-10000);
  check("3 a held flux is a fixed point", advance(F, F, 24.0, 18.0) == F);
}
{
  my(a = exp(-24.0/18.0), A = 500.0, G = advance(0.0, A, 24.0, 18.0), total = G);
  for (n = 1, 2000, G = advance(G, 0.0, 24.0, 18.0); total += G);
  check("5 a one-day spike's excess sums to its height", near(total, A, 1e-50));
  my(S = 0.0); for (n = 0, 6, S = advance(S, A, 24.0, 18.0));
  check("6 a held step after day 6 is A (1 - a^7)", near(S, A*(1 - a^7), 1e-50));
  check("7 the spike's largest excess is A (1 - a); 1 - 1/e at tau 24 h",
    near(advance(0.0, A, 24.0, 18.0), A*(1 - a), 1e-50)
    && near(advance(0.0, 1.0, 24.0, 24.0), 0.6321205588285577, 1e-15));
}
