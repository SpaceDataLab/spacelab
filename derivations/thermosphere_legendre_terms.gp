\\ Second engine for derivations/thermosphere_legendre_terms.mac: the coded associated Legendre set
\\ against P_n^m(x) = (1 - x^2)^(m/2) d^m P_n/dx^m from Rodrigues, exactly, on rational points of the
\\ unit circle (c, s) = (sin phi, cos phi); and the local-time recurrences against cos kh, sin kh at
\\ 40 digits.
\\
\\ Run: timeout 300 gp -q -f derivations/thermosphere_legendre_terms.gp </dev/null
\\ Prints PASS lines only; any FAIL or *** is a failure.

default(realprecision, 40);
dern(P, m) = for (k = 1, m, P = deriv(P, x)); P;
leg(n) = dern((x^2 - 1)^n, n)/(2^n*n!);
assoc(n, m, c, s) = s^m*subst(dern(leg(n), m), x, c);
coded(c, s) =
{
  my(p40 = 35/8*c^4 - 15/4*c^2 + 3/8, p50 = c*(63/8*c^4 - 35/4*c^2 + 15/8),
     p32 = 15*c*s^2, p42 = s^2*(105/2*c^2 - 15/2), p52 = 3*c*p42 - 2*p32);
  [[c, 1, 0], [3/2*c^2 - 1/2, 2, 0], [c*(5/2*c^2 - 3/2), 3, 0], [p40, 4, 0], [p50, 5, 0],
   [(11/2*c*p50 - 5/2*p40)/3, 6, 0],
   [s, 1, 1], [3*c*s, 2, 1], [s*(15/2*c^2 - 3/2), 3, 1], [c*s*(35/2*c^2 - 15/2), 4, 1],
   [s*(315/8*c^4 - 105/4*c^2 + 15/8), 5, 1],
   [3*s^2, 2, 2], [p32, 3, 2], [p42, 4, 2], [p52, 5, 2], [11/4*c*p52 - 7/4*p42, 6, 2],
   [15*s^3, 3, 3]];
}
{
  my(points = [[0, 1], [3/5, 4/5], [-3/5, 4/5], [5/13, 12/13], [-12/13, 5/13], [1, 0], [-1, 0],
               [8/17, 15/17], [-20/29, 21/29]], bad = 0, count = 0);
  for (k = 1, #points,
    my(c = points[k][1], s = points[k][2], set = coded(c, s));
    for (j = 1, #set,
      count++;
      if (set[j][1] != assoc(set[j][2], set[j][3], c, s), bad++)));
  print(if (bad == 0, "PASS", "FAIL"), " the coded Legendre set is P_n^m exactly at ", count,
        " rational points");
}
{
  my(worst = 0.);
  for (k = 0, 47,
    my(h = 2*Pi*k/48 + 1/7, ch = cos(h), sh = sin(h), c2h = ch^2 - sh^2, s2h = 2*ch*sh);
    worst = max(worst, vecmax(abs([c2h - cos(2*h), s2h - sin(2*h),
                                   c2h*ch - s2h*sh - cos(3*h), s2h*ch + c2h*sh - sin(3*h)]))));
  print(if (worst < 10^-38, "PASS", "FAIL"), " the local-time recurrences are cos kh, sin kh to ",
        Strprintf("%.3g", worst));
}
