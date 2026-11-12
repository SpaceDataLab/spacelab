/* Second engine for derivations/brouwer_m2_phipps.mac, numerically at 100 digits.

   (1) l'' = l0 + m t + M2 t^2 has second derivative 2 M2, by numerical differentiation.
   (2) a(m) = (mu/m^2)^(1/3) has da/dm = -(2/3) a/m, so a-dot = -(4/3)(a/m) M2 when m-dot = 2 M2:
       at random mu, m, M2, by numerical differentiation.

       gp -q -f derivations/brouwer_m2_phipps.gp </dev/null                                        */

default(realprecision, 100);
setrand(20261112);
u01() = random(10^30)/10^30;

worst = 0;
{
for (k = 1, 30,
  my(mu = 1 + u01(), m = 1/2 + u01(), M2 = u01()/1000, l0 = u01(), a, dadm, l2);
  l2 = derivnum(t = u01(), derivnum(s = t, l0 + m*s + M2*s^2));
  worst = max(worst, abs(l2 - 2*M2));
  a = (mu/m^2)^(1/3);
  dadm = derivnum(x = m, (mu/x^2)^(1/3));
  worst = max(worst, abs(dadm + 2/3*a/m));
  worst = max(worst, abs(dadm*2*M2 + 4*a*M2/(3*m))));
}
print("worst: ", worst);
if (worst < 10^-60, print("Phipps' drift rates hold at 100 digits"), print("FAIL: Phipps' drift rates"));
