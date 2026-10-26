\\ Second engine for derivations/photometry_closed_forms.mac.
\\  * the diffuse-sphere phase law as the double integral of cos(i) cos(e) over the lit-visible lune,
\\    by numerical quadrature (intnum), against [sin a + (pi - a) cos a]/pi;
\\  * the V-band zero point as an exact rational;
\\  * the limiting magnitude by solving the CCD equation SNR(v) = target directly in v (no quadratic),
\\    against Maxima's closed-form root.
default(realprecision, 50);
phi(a) = (sin(a) + (Pi - a)*cos(a))/Pi;
lune(a) = intnum(b = -Pi/2, Pi/2, intnum(l = a - Pi/2, Pi/2, cos(b)^3*cos(l)*cos(l - a)));
{
  my(worst = 0);
  forstep (k = 0, 6, 1, my(a = k*Pi/6); worst = max(worst, abs(lune(a)/lune(0) - phi(a))));
  printf("diffuse sphere: max |quadrature - closed form| over a = k pi/6 = %.2e  %s\n", worst,
         if (worst < 1e-40, "ok", "MISMATCH"));
}
Z = (363/10^4) * (84/10^9) * (545/10^9) / ((662607015/10^42) * 299792458);
printf("zero point = %s  %s\n", Z, if (Z == 7913400000000000000000000/945926598642347, "ok", "MISMATCH"));
rate = Z * Pi*(1 - 9/100)/4 * 6/10 * 85/100 * 30;
snr(v, sky) = my(S = rate*10^(-2*v/5), n = Pi*4); S/sqrt(S + n*(rate*10^(-2*sky/5) + 30/100 + 25));
maxima = [[218/10, 5, 21.35247580048740310961465657900928092323], \
          [19, 5, 20.05839322230813946063991352052835760709], \
          [218/10, 10, 20.54586178000252402156335878827407983778], \
          [218/10, 3, 21.92875837122376227487199582762950212177]];
{
  for (i = 1, #maxima,
    my(p = maxima[i], v = solve(x = -5, 40, snr(x, p[1]) - p[2]), d = abs(v - p[3]));
    printf("limiting magnitude sky %.1f SNR %d: %.34g  |solve - Maxima| = %.2e  %s\n",
           p[1], p[2], v, d, if (d < 1e-36, "ok", "MISMATCH")));
}
