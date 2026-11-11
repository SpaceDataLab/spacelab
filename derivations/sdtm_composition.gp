/* Second engine for derivations/sdtm_composition.mac: the shipped boundary's oxygen mass fraction w and
   d ln rho/d T∞ at 200, 250, 300 and 490 km, the response at 200 and 300 km with 250 and 490 km held, and the
   exact response to GOLD's c, solved by Newton's method, all at 60 digits against Maxima's printed values.
   Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_composition.gp </dev/null                                     */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
R = 6356766; z0 = 120000; g0 = 980665/100000*(R/(R + z0))^2; kB = 1380649/10^29; amu = 166053906892/10^38;
T0 = 380; sg = 3341439818644104/10^20; sc = 49658530381854077/10^17;
sp = [[280134/10^4, 4*10^17, 0], [319988/10^4, 75*10^15, 0], [159994/10^4, 76*10^15, 0], \
      [39948/10^3, 18*10^14, 0], [40026/10^4, 34*10^12, -38/100]];
geo(z) = (z - z0)*(R + z0)/(R + z);
temp(Ti, x) = Ti - (Ti - T0)*exp(-sg*x);
i1(Ti, x) = (sg*x + log(temp(Ti, x)/T0))/(sg*Ti);
partial(k, Ti, x, cc) = sp[k][1]*amu*sc*sp[k][2]*if (k == 3, exp(cc), 1) \
  *(T0/temp(Ti, x))^(1 + sp[k][3])*exp(-sp[k][1]*amu*g0/kB*i1(Ti, x));
lnrho(Ti, x, cc) = log(sum(k = 1, 5, partial(k, Ti, x, cc)));
wO(Ti, x) = partial(3, Ti, x, 0)/sum(k = 1, 5, partial(k, Ti, x, 0));
dlnT(Ti, x) = my(h = 10^-20); (lnrho(Ti + h, x, 0) - lnrho(Ti - h, x, 0))/(2*h);
near(a, b, tol) = abs(a - b) <= tol*abs(b);
{
  my(z = [200000, 250000, 300000, 490000],
     maxw = [[0.38866192176198305, 0.6300169368422207, 0.8120864296965623, 0.8397192487765388],
             [0.2908741220196561, 0.45218570417572646, 0.6172836536311168, 0.9237376938670894]],
     maxd = [[0.002326011527589191, 0.004095467364194523, 0.005468603306537769, 0.009812499969098314],
             [0.0012183443381514763, 0.002276205115003254, 0.003093215100963225, 0.0054720059144072125]],
     temps = [700, 1000], ok = 1);
  for (j = 1, 2, for (k = 1, 4,
    ok = ok && near(wO(temps[j], geo(z[k])), maxw[j][k], 1e-14) && near(dlnT(temps[j], geo(z[k])), maxd[j][k], 1e-14)));
  check("w and d ln rho/d T∞ at four heights and two T∞ against Maxima to 1e-14", ok);
}
lin(Ti, z3) = {
  my(x = [geo(250000), geo(490000), geo(z3)], w = vector(3, k, wO(Ti, x[k])), d = vector(3, k, dlnT(Ti, x[k])));
  (w[3] - w[1]) - (d[3] - d[1])*(w[1] - w[2])/(d[1] - d[2]);
}
{
  my(maxl = [[-0.17645089340642828, 0.13170249276718132], [-0.005220390792095549, 0.04454516514261621]],
     temps = [700, 1000], ok = 1);
  for (j = 1, 2, ok = ok && near(lin(temps[j], 200000), maxl[j][1], 1e-12) && near(lin(temps[j], 300000), maxl[j][2], 1e-12));
  check("the linear response at 200 and 300 km against Maxima to 1e-12", ok);
}
/* Newton on (s, T∞) so that ln rho at 250 and 490 km does not move when oxygen carries e^cg. */
exact(Ti, cg) = {
  my(x1 = geo(250000), x2 = geo(490000), v = [-1/2, 9*Ti/10]~, h = 10^-25, F, J);
  /* In parentheses: an unbracketed GP definition takes the rest of the sequence as its body. */
  F = (u -> [lnrho(u[2], x1, cg) + u[1] - lnrho(Ti, x1, 0), lnrho(u[2], x2, cg) + u[1] - lnrho(Ti, x2, 0)]~);
  for (n = 1, 60,
    J = matrix(2, 2, i, k, (F(v + h*matid(2)[, k])[i] - F(v - h*matid(2)[, k])[i])/(2*h));
    v = v - matsolve(J, F(v)));
  [v[1], v[2], exp(lnrho(v[2], geo(200000), cg) + v[1] - lnrho(Ti, geo(200000), 0)) - 1,
   exp(lnrho(v[2], geo(300000), cg) + v[1] - lnrho(Ti, geo(300000), 0)) - 1];
}
{
  my(cg = log(1040/1000/(487542453257362/10^15)),
     maxe = [[-0.47077086515274025, 681.4738609203146, -0.13326845085289873, 0.08920076079413808],
             [-0.2453789507826217, 919.6721963999505, -0.04856474306721814, 0.0519296942013554]],
     temps = [700, 1000], ok = 1, e);
  for (j = 1, 2, e = exact(temps[j], cg); print("T∞ ", temps[j], " K exact: ", apply(x -> precision(x, 15), e));
    for (k = 1, 4, ok = ok && near(e[k], maxe[j][k], 1e-9)));
  check("the exact response to GOLD's c against Maxima's Newton to 1e-9", ok);
}
