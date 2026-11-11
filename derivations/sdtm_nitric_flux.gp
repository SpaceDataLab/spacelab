/* Second engine for derivations/sdtm_nitric_flux.mac: the flux-scaled nitric-oxide time at 60 digits.
   Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_nitric_flux.gp </dev/null                                      */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
near(a, b, tol) = abs(a - b) <= tol*max(1, abs(b));
rk4(f, y0, t, n) = {
  my(y = y0, s = t/n, k1, k2, k3, k4);
  for (i = 1, n, k1 = f(y); k2 = f(y + s/2*k1); k3 = f(y + s/2*k2); k4 = f(y + s*k3);
    y = y + s/6*(k1 + 2*k2 + 2*k3 + k4));
  y;
}
tn(tn0, F, kn) = tn0*(F/150)^(-kn);
{
  check("1 kn = 0 is the shipped time", tn(20.0, 87.0, 0) == 20.0);
  check("2 the elasticity is -kn",
    near((log(tn(20.0, 87.0*(1 + 10^-30), 1.3)) - log(tn(20.0, 87.0, 1.3)))/log(1 + 10^-30), -1.3, 1e-25));
}
/* (3): N over two 3 h steps with the time held per step, against RK4 on the same piecewise-constant time. */
{
  my(H = [200.0, 40.0], T = [tn(20.0, 210.0, 1.2), tn(20.0, 205.0, 1.2)], N = 15.0, M = 15.0);
  for (k = 1, 2,
    N = H[k] + (N - H[k])*exp(-3.0/T[k]);
    M = rk4(y -> [(H[k] - y[1])/T[k]]~, [M]~, 3.0, 3000)[1]);
  check("3 the step with the time held per interval is exact for it", near(N, M, 1e-12));
}
/* (4): the steady state of the full system under a held drive is H/(1 + H/Hs) at two different times. */
{
  my(H = 150.0, tc = 6.0, Hs = 300.0, f, out = vector(2));
  for (k = 1, 2, my(T = [10.0, 60.0][k]);
    f = (y -> [(H - y[1])/tc - y[1]*y[2]/(tc*Hs), (H - y[2])/T]~);
    out[k] = rk4(f, [0, 0]~, 3000.0, 30000)[1]);
  check("4 the steady state H/(1 + H/Hs) does not depend on the time",
    near(out[1], H/(1 + H/Hs), 1e-12) && near(out[2], H/(1 + H/Hs), 1e-12));
}
{
  check("5 kn for a ratio 3, 7/2 and 4 between F 70 and 200 against Maxima",
    near(log(3)/log(200/70), 1.0464746960755194, 1e-15) && near(log(4)/log(200/70), 1.3205040442273863, 1e-15)
    && near(log(7/2)/log(200/70), 1.19330974196566946430933276836, 1e-29)
    && near(tn(20.0, 70.0, log(3)/log(200/70))/tn(20.0, 200.0, log(3)/log(200/70)), 3, 1e-50));
}
