/* Decay epochs of a circular orbit through a static exponential atmosphere at 60 digits: the second
   engine for derivations/lifetime_exponential_decay.mac and the reference the lifetime gate in
   crates/spacelab-propagation/src/lifetime.rs asserts against.

       t(a) = sqrt(pi H) exp(-r0/H)/(rho0 B sqrt(mu)) [erfi(sqrt(a0/H)) - erfi(sqrt(a/H))]

   is evaluated from erfi(x) = -i erf(i x) = -i (1 - erfc(i x)), and independently as the quadrature
   t(a) = int_a^a0 dx/(rho0 exp(-(x - r0)/H) B sqrt(mu x)); the two are printed with their
   relative difference. Inputs are the doubles the Rust test builds: mu = 3.986004415e14 (exact),
   Re = 6378136.46, rho0 = 2e-11 and B = 2.2 * 0.02, each the nearest double, r0 = Re + 400 km and
   a0 = Re + 400 km rounded as f64 sums are; H = 60 km. Each `epoch:` row is
   (threshold altitude m, t s) for the threshold a = Re + altitude.

       timeout 120 gp -q -f derivations/lifetime_exponential_decay.gp </dev/null                   */

default(realprecision, 60);

dbl(x) = {
  my(r = x * 1., e);
  if (r == 0, return(0));
  e = exponent(r);
  round(x * 2^(52 - e)) * 2^(e - 52);
}

mu = 398600441500000;
Re = dbl(637813646 / 100);
rho0 = dbl(2 / 10^11);
B = dbl(dbl(22 / 10) * dbl(2 / 100));
H = 60000;
r0 = dbl(Re + 400000);
a0 = r0;

erfi(x) = real(-I * (1 - erfc(I * x)));
K = sqrt(Pi * H) * exp(-r0 / H) / (rho0 * B * sqrt(mu));
t_closed(a) = K * (erfi(sqrt(a0 / H)) - erfi(sqrt(a / H)));
t_quad(a) = intnum(x = a, a0, 1 / (rho0 * exp(-(x - r0) / H) * B * sqrt(mu * x)));

worst = 0;
{
  for (k = 1, 6,
    my(alt = [350000, 300000, 250000, 200000, 150000, 120000][k], a = dbl(Re + alt),
       tc = t_closed(a), tq = t_quad(a));
    worst = max(worst, abs(tc / tq - 1));
    printf("epoch:  (%d.0, %.25g),\n", alt, tc));
}
printf("closed form vs quadrature, worst relative difference: %.3g\n", worst);
