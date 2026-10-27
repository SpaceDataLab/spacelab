\\ Second engine for derivations/ice_geometric_optics.mac, by PARI's own Bessel functions rather than
\\ the series Maxima sums:
\\  * J_1 at the module test's thirteen arguments and its first zero, against Maxima's 40 digits;
\\  * the two identities the downward recurrence rests on -- J_(n-1) = (2n/x) J_n - J_(n+1) and
\\    1 = J_0 + 2 (J_2 + J_4 + ...) -- at sampled arguments;
\\  * the eight-term Hankel expansion the code switches to at x = 100, against besselj, as a
\\    fraction of the envelope sqrt(2/(pi x));
\\  * the minimum deviations and the aperture size parameter, against Maxima.
default(realprecision, 60);
{
  my(bad = 0,
     ref = [[1/1000000, 4.999999999999375000000000026041666666666e-7],
            [25/1000000, 1.249999999902343750002543131510383552975e-5],
            [1/2, 2.422684576748738863839545761415316408006e-1],
            [1, 4.400505857449335159596822037189149131274e-1],
            [2, 5.767248077568733872024482422691370869203e-1],
            [5, -3.275791375914652220377343219101691327609e-1],
            [8, 2.346363468539146243812766515904546115488e-1],
            [10, 4.347274616886143666974876802585928830627e-2],
            [20, 6.6833124175850045578992974193646719983e-2],
            [50, -9.751182812517513766145895387370161404049e-2],
            [100, -7.714535201411215803268549492723447021161e-2],
            [400, -9.222058428586351254235216748846469409394e-3],
            [1000, 4.728311907089523917576071901216916285418e-3]]);
  for (i = 1, #ref, my(x = ref[i][1], w = ref[i][2]);
    if (abs(besselj(1, x) - w) > 1e-39 * max(abs(w), 1e-3), bad++));
  printf("bessel J1: %d of %d values off Maxima by more than 1e-39 relative\n", bad, #ref);
  my(z = solve(v = 38/10, 39/10, besselj(1, v)));
  printf("first zero of J1: |besselj zero - Maxima| = %.1e, %d of 1 off by more than 1e-39\n",
         abs(z - 3.831705970207512315614435886308160766565),
         abs(z - 3.831705970207512315614435886308160766565) > 1e-39);
}
{
  my(bad = 0, n = 0);
  foreach ([1/10, 1, 7, 30, 99], x,
    for (k = 1, 40, n++;
      if (abs(besselj(k - 1, x) - (2*k/x*besselj(k, x) - besselj(k + 1, x))) > 1e-50, bad++));
    n++;
    if (abs(besselj(0, x) + 2*sum(k = 1, 120, besselj(2*k, x)) - 1) > 1e-50, bad++));
  printf("recurrence and normalisation: %d of %d identities off by more than 1e-50\n", bad, n);
}
hankel(x) = {
  my(term = 1, p = 1, q = 0, s, chi = x - 3*Pi/4);
  for (k = 1, 8,
    term *= (4 - (2*k - 1)^2)/(8*k*x);
    s = if ((k \ 2) % 2 == 0, term, -term);
    if (k % 2 == 0, p += s, q += s));
  sqrt(2/(Pi*x))*(p*cos(chi) - q*sin(chi));
}
{
  my(worst = 0, bad = 0);
  for (i = 0, 400, my(x = 100*10^(i/200), r = abs(hankel(x) - besselj(1, x))/sqrt(2/(Pi*x)));
    worst = max(worst, r); if (r > 1e-15, bad++));
  printf("hankel expansion: %d of 401 arguments in [100, 1e4] past 1e-15 of the envelope (worst %.2e)\n",
         bad, worst);
}
{
  my(md(a) = (2*asin(13105/10000*sin(a/2)) - a)*180/Pi, bad = 0,
     pairs = [[md(Pi/3), 21.87721799402075648137179703994621629527],
              [md(Pi/2), 45.84107772585809094403802709225234137759],
              [200*sqrt((24 + 3*sqrt(3))/(4*Pi)), 304.8511760626216161463692935850979915572]]);
  for (i = 1, 3, if (abs(pairs[i][1] - pairs[i][2]) > 1e-38*abs(pairs[i][2]), bad++));
  printf("deviations and aperture: %d of 3 off Maxima by more than 1e-38 relative\n", bad);
}
