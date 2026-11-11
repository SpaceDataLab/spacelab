/* Second engine for derivations/sdtm_overcooling.mac: the two Ф23.J rivals against RK4 at 60 digits.
   Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_overcooling.gp </dev/null                                     */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
near(a, b, tol) = abs(a - b) <= tol*max(1, abs(b));
/* RK4 on y' = f(y) from y0 over t in n steps; f takes and returns a column vector. */
rk4(f, y0, t, n) = {
  my(y = y0, s = t/n, k1, k2, k3, k4);
  for (i = 1, n, k1 = f(y); k2 = f(y + s/2*k1); k3 = f(y + s/2*k2); k4 = f(y + s*k3);
    y = y + s/6*(k1 + 2*k2 + 2*k3 + k4));
  y;
}
/* The delta form at H, tc, tn, Hs, delta; y = [D, N]~. */
deltaf(H, tc, tn, Hs, dl) = (y -> [(H - y[1])/tc - (y[1] + dl)*y[2]/(tc*Hs), (H - y[2])/tn]~);
/* Its step with N frozen at its exact mean over h, as fit-sdtm-corpus.py's cooling_grid takes it. */
deltastep(D0, N0, H, h, tc, tn, Hs, dl) = {
  my(nb = H + (N0 - H)*tn/h*(1 - exp(-h/tn)), lam = (1 + nb/Hs)/tc, tg = (H - dl*nb/Hs)/(1 + nb/Hs));
  tg + (D0 - tg)*exp(-lam*h);
}
{
  my(tc = 9.0, tn = 30.0, Hs = 250.0, dl = 40.0, H = 120.0, D0 = 60.0, N0 = 200.0, errs);
  /* With N held at a constant, the frozen step is exact: tn -> infinity and N0 = H. */
  check("A the frozen delta step is exact for a frozen N", near(deltastep(D0, N0, N0, 3.0, tc, 10^40, Hs, dl),
    rk4(y -> [(N0 - y[1])/tc - (y[1] + dl)*N0/(tc*Hs), 0]~, [D0, N0]~, 3.0, 2000)[1], 1e-12));
  /* With N moving, its local error starts at h^3: halving h divides it by a ratio that rises to 8
     (6.8 at 3 h, 7.8 at 0.375 h, 7.9 at 0.1875 h). */
  errs = vector(3, k, my(h = 0.375/2^(k - 1));
    abs(deltastep(D0, N0, H, h, tc, tn, Hs, dl) - rk4(deltaf(H, tc, tn, Hs, dl), [D0, N0]~, h, 4000)[1]));
  check("A its local error falls as h^3 with N moving", abs(errs[1]/errs[2]/8 - 1) < 0.03 && abs(errs[2]/errs[3]/8 - 1) < 0.015);
  /* The steady state h (Hs - delta)/(Hs + h). */
  check("A steady state h (Hs - delta)/(Hs + h)",
    near(rk4(deltaf(H, tc, tn, Hs, dl), [0, 0]~, 2000.0, 20000)[1], H*(Hs - dl)/(Hs + H), 1e-12));
  /* A storm of 24 h at ap 300, then a quiet ap 5: the response goes below zero exactly while
     delta N > H Hs, i.e. while N > 5 Hs/delta = 31.25. */
  my(y = rk4(deltaf(300.0, tc, tn, Hs, dl), [0, 0]~, 24.0, 2400), below = 0, agree = 1, step = 0.25, z, s);
  for (i = 1, 960, z = rk4(deltaf(5.0, tc, tn, Hs, dl), y, step, 25);
    s = (5.0*Hs - dl*z[2]) > 0;
    if (z[1] < 0, below = 1);
    /* The target's sign leads D's: once the target is above zero D cannot fall below it again. */
    if (s && z[1] < 0 && (5.0*Hs - dl*y[2]) > 0 && z[1] < y[1], agree = 0);
    y = z);
  check("A a quiet drive under a high N overcools, and recovers once delta N < H Hs", below && agree && y[1] > 0);
}
oddlog(x) = sign(x)*log(1 + abs(x)/15);
{
  my(xs = [0.001, 1.0, 15.0, 300.0], ok = 1);
  for (k = 1, #xs, ok = ok && oddlog(-xs[k]) == -oddlog(xs[k]) && near(oddlog(xs[k]), log1p(xs[k]/15), 1e-50));
  ok = ok && near(oddlog(10^-30)/10^-30, 1/15, 1e-25) && near(oddlog(-10^-30)/(-10^-30), 1/15, 1e-25);
  check("A the odd extension: odd, ln(1 + D/15) above zero, slope 1/15 at 0 from both sides", ok);
}
/* The linear form; y = [D, N]~. */
linf(H, tc, tn, bt) = (y -> [(H - y[1] - bt*y[2])/tc, (H - y[2])/tn]~);
linexact(D0, N0, H, t, tc, tn, bt) = {
  my(K = -bt*(N0 - H)*tn/(tn - tc));
  (1 - bt)*H + (D0 - (1 - bt)*H - K)*exp(-t/tc) + K*exp(-t/tn);
}
{
  my(tc = 9.0, tn = 30.0, bt = 0.35, ok = 1, Q = 200.0, D0 = (1 - bt)*Q, t);
  ok = near(linexact(50.0, 180.0, 120.0, 7.0, tc, tn, bt), rk4(linf(120.0, tc, tn, bt), [50.0, 180.0]~, 7.0, 4000)[1], 1e-12);
  check("B the linear step against RK4 on the full system", ok);
  check("B the steady state (1 - beta) H, linear: twice the drive, twice the state",
    near(rk4(linf(240.0, tc, tn, bt), [0, 0]~, 2000.0, 20000)[1], 2*rk4(linf(120.0, tc, tn, bt), [0, 0]~, 2000.0, 20000)[1], 1e-12)
    && near(rk4(linf(120.0, tc, tn, bt), [0, 0]~, 2000.0, 20000)[1], (1 - bt)*120.0, 1e-12));
  t = 600.0;
  check("B after the drive stops it undershoots and ends at K e^(-t/tn), K < 0",
    linexact(D0, Q, 0, t, tc, tn, bt) < 0
    && near(linexact(D0, Q, 0, t, tc, tn, bt)*exp(t/tn), -bt*Q*tn/(tn - tc), 1e-12));
}
