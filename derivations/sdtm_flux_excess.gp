/* Second engine for derivations/sdtm_flux_excess.mac: S(d) = D tanh(d/D) at 60 digits against the
   reference values Maxima prints at D = 0.27, the flux rewrite F' = Fa + 100 S((F - Fa)/100)
   returning S, and the bound |S| < D. Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_flux_excess.gp </dev/null                         */

default(realprecision, 60);
S(d, D) = D*tanh(d/D);
D = 27/100;
x = [8236/1000, 712/1000, -13/100, 1/1000];
ref = [0.269999999999999999999999998273, 0.267247841689291188733474356195, \
  -0.120805909547456328426797669961, 0.000999995427551380603497317931118];
for (k = 1, 4, print("S at d = ", x[k], if (abs(S(x[k], D) - ref[k]) < 10^-28, ": PASS", ": FAIL"), \
  " ", Strprintf("%.30g", S(x[k], D))));
Fa = 162; F = 233.2;
print(if (abs((Fa + 100*S((F - Fa)/100, D) - Fa)/100 - S((F - Fa)/100, D)) < 10^-55, \
  "flux rewrite: PASS", "flux rewrite: FAIL"));
print(if (vecmax(vector(400, k, abs(S(k/10 - 20, D)))) < D, "bounded by D: PASS", "bounded by D: FAIL"));
