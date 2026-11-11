/* Second engine for derivations/sdtm_column.mac, by intnum at 60 digits. The column above x is n T/b for a
   linear temperature profile and for the Bates profile at the shipped sigma; the column ratio O/N2 in
   geopotential height against Maxima's 40 digits; the geometric depth and ratio at Tinf = 700, 1000, 1500 K
   against Maxima's QUADPACK values, and the ratio's rise with Tinf. Prints `PASS` lines; a failed check prints
   `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_column.gp </dev/null                                          */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
R = 6356766; z0 = 120000; a = R + z0;
g0 = 980665/100000*(R/a)^2;
kB = 1380649/10^29; mu = 166053906892/10^38;
T0 = 380;
sg = 1.*3341439818644104/10^20;
sc = 49658530381854077/10^17;
bN = 1.*280134/10^4*mu*g0/kB; bO = 1.*159994/10^4*mu*g0/kB;
nN0 = sc*4*10^17; nO0 = sc*76*10^15;
Nst = 10^21; ztop = 5000000;

/* Any T: a linear profile T = T0 + c x, whose I1 is log(T/T0)/c. */
Tl(y) = T0 + y/100;
nl(y) = nO0*(T0/Tl(y))*exp(-100*bO*log(Tl(y)/T0));
check("linear T: int_x^oo n = n T/b to 1e-50", \
  abs(intnum(y = 15000, [+oo, -100*bO], nl(y))/(nl(15000)*Tl(15000)/bO) - 1) < 10^-50);

\\ Beyond sg x = 1e4 the exponential is below 1e-4000, far under the 60 digits kept.
TS(t, x) = if (sg*x > 10^4, t, t - (t - T0)*exp(-sg*x));
IS(t, x) = (sg*x + log(TS(t, x)/T0))/(sg*t);
densx(n, b, t, x) = n*(T0/TS(t, x))*exp(-b*IS(t, x));
geo(z) = (z - z0)*a/(R + z);
dens(n, b, t, z) = densx(n, b, t, geo(z));
\\ Between two heights, in 50 km pieces: the column's difference; the top's limit is Maxima's (6).
C(t, x) = densx(nO0, bO, t, x)*TS(t, x)/bO;
check("Bates at Tinf 1000 K: int_x1^x2 n = (n T/b)(x1) - (n T/b)(x2) to 1e-50", \
  abs(sum(i = 0, 79, intnum(y = 15000 + 50000*i, 65000 + 50000*i, densx(nO0, bO, 1000, y))) \
      /(C(1000, 15000) - C(1000, 4015000)) - 1) < 10^-50);

pp = (bN/bO)*(nO0/nN0)*(nN0*T0/(bN*Nst))^(1 - bO/bN);
check("SigmaO/N2 in geopotential height against Maxima to 1e-38", \
  abs(pp/4.817847255698109524858117591090534603324e-1 - 1) < 10^-38);
print("SigmaO/N2 in geopotential height: ", pp);

col(n, b, t, z) = {
  my(p = [z, z + 20000, z + 60000, z + 150000, z + 400000, z + 1000000, ztop]);
  sum(i = 1, #p - 1, intnum(y = p[i], p[i + 1], dens(n, b, t, y)));
}
maxima = [[700, 131917.0564451929, 0.48597135751592835], [1000, 133576.24173515197, 0.48754245325736195], \
          [1500, 136970.76891199814, 0.4905532759560734]];
{
  my(prev = 0);
  for (j = 1, 3,
    my(t = maxima[j][1], z = maxima[j][2], r);
    for (k = 1, 6, z = z + (col(nN0, bN, t, z) - Nst)/dens(nN0, bN, t, z));
    r = col(nO0, bO, t, z)/col(nN0, bN, t, z);
    print("Tinf ", t, " K: geometric depth z ", z, " m; SigmaO/N2 ", r);
    check(Str("Tinf ", t, " K: N2 column at the depth is 1e21 to 1e-50"), abs(col(nN0, bN, t, z)/Nst - 1) < 10^-50);
    check(Str("Tinf ", t, " K: depth and ratio against Maxima's QUADPACK to 1e-11"),
      abs(z/maxima[j][2] - 1) < 10^-11 && abs(r/maxima[j][3] - 1) < 10^-11);
    check(Str("Tinf ", t, " K: the geometric ratio exceeds the geopotential one and rises with Tinf"), r > pp && r > prev);
    prev = r);
}
