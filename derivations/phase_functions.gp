\\ Second engine for derivations/phase_functions.mac, by quadrature and root-finding rather than
\\ by algebra:
\\  * Henyey-Greenstein: int p = 1, int mu p = g and int P_l p = g^l (l = 0..8) by intnum, at seven g;
\\  * the HG sampler: F(mu(xi)) = xi with F the quadrature of p, not the closed-form CDF;
\\  * Rayleigh: the sampled cosine is the real root of mu^3 + 3 mu + 4 - 8 xi (polroots), at 33 xi;
\\  * the module test's literals, re-derived and compared with Maxima's 40 digits.
default(realprecision, 60);
hg(g, m) = (1 - g^2)/(2*(1 + g^2 - 2*g*m)^(3/2));
hgsample(g, x) = my(t = (1 - g^2)/(1 - g + 2*g*x)); (1 + g^2 - t^2)/(2*g);
gs = [-9/10, -1/2, -1/10, 1/10, 1/2, 3/4, 9/10];
{
  my(bad = 0, n = 0);
  for (i = 1, #gs, my(g = gs[i]);
    n++; if (abs(intnum(m = -1, 1, hg(g, m)) - 1) > 1e-40, bad++);
    n++; if (abs(intnum(m = -1, 1, m*hg(g, m)) - g) > 1e-40, bad++);
    for (l = 0, 8, n++; if (abs(intnum(m = -1, 1, pollegendre(l, m)*hg(g, m)) - g^l) > 1e-40, bad++));
    forstep (k = 1, 15, 2, my(x = k/16, mu = hgsample(g, x));
      n++; if (abs(intnum(m = -1, mu, hg(g, m)) - x) > 1e-40, bad++)));
  printf("henyey-greenstein: %d of %d quadrature checks off by more than 1e-40\n", bad, n);
}
rayleighroot(x) = my(r = polroots(Pol([1, 0, 3, 4 - 8*x]))); real(r[1]) * (abs(imag(r[1])) < 1e-50) \
  + real(r[2]) * (abs(imag(r[2])) < 1e-50) + real(r[3]) * (abs(imag(r[3])) < 1e-50);
rayleighcardano(x) = my(u = 4*x - 2, s = sqrt(u^2 + 1), a = if (u >= 0, (u + s)^(1/3), (1/(s - u))^(1/3))); a - 1/a;
{
  my(bad = 0);
  for (k = 0, 32, my(x = k/32); if (abs(rayleighroot(x) - rayleighcardano(x)) > 1e-45, bad++));
  printf("rayleigh: %d of 33 Cardano roots off the cubic's real root by more than 1e-45\n", bad);
}
maxima = [[hg(3/4, 1/2), 0.2986847210443541426252609333999227411096], \
          [hgsample(3/4, 3/8), 1147/1352], \
          [hgsample(-5/8, 13/16), -5/16], \
          [rayleighroot(1/4), -0.5960716379833215231128054143996818281133], \
          [rayleighroot(7/8), 0.8177316738868235060940870704093553698711], \
          [rayleighroot(5/16), -0.4662205239107734273913577876420742203745]];
{
  my(bad = 0);
  for (i = 1, #maxima, if (abs(maxima[i][1] - maxima[i][2]) > 1e-38, bad++));
  printf("module-test literals: %d of %d off Maxima by more than 1e-38\n", bad, #maxima);
}
