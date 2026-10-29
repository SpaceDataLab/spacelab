/* Second engine for derivations/sdtm_storm_gain.mac: g(F) = (F/150)^kappa at 60 digits against
   the reference values Maxima prints at kappa = -1.3, the reciprocal g(F) g(150^2/F) = 1, and the
   homogeneity of the cooling state's D step at a sample point. Prints `PASS` lines; a failed check
   prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_storm_gain.gp </dev/null                          */

default(realprecision, 60);
g(F, k) = (F/150)^k;
k = -13/10;
x = [75, 150, 225, 300];
ref = [2.46228882668983256899878613834, 1.0, 0.59031166219703741059838788771, \
  0.406126198178117761304854691387];
for (j = 1, 4, print("g at F = ", x[j], if (abs(g(x[j], k) - ref[j]) < 10^-28, ": PASS", ": FAIL"), \
  " ", Strprintf("%.30g", g(x[j], k))));
print(if (vecmax(vector(40, j, abs(g(10*j, k)*g(150^2/(10*j), k) - 1))) < 10^-55, \
  "reciprocal about 150: PASS", "reciprocal about 150: FAIL"));
dstep(d, n, u, hs, th, tn, h) = {
  my(m = u + (n - u)*tn/h*(1 - exp(-h/tn)), rate = (1 + m/hs)/th, target = u/(th*rate));
  target + (d - target)*exp(-rate*h);
}
G = g(225, k);
print(if (abs(dstep(G*31, G*120, G*300, G*25, 17.4, 32.6, 3) - G*dstep(31, 120, 300, 25, 17.4, 32.6, 3)) \
  < 10^-55, "D step homogeneous: PASS", "D step homogeneous: FAIL"));
