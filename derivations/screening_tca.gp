/* Times of closest approach at 50 digits as roots of their defining equation: the second engine for
   derivations/screening_tca.mac and the reference values the `closest_approaches` gate in
   crates/spacelab-estimation/src/screening.rs asserts the shipped bisection against.

   A closest approach is a zero of f'(t) = dr . dv (the derivative of |dr|^2/2) where the sign goes
   from negative to positive. For the two geometries of the module tests -- circular orbits of radii
   r1 = 7000 km and r2 = 7500 km, the second started 1 rad ahead, coplanar and with the second plane
   tilted by 63.4 deg -- every such zero in the test's span is bracketed on a 15 s grid and polished
   with solve(). For the coplanar pair the zeros also have the closed form of screening_tca.mac,
   t_j = (phase + 2 pi j)/(n1 - n2), and the two routes are compared.

       timeout 300 gp -q -f derivations/screening_tca.gp </dev/null                                */

default(realprecision, 60);

MU = 3986004418 * 10^5;
r1 = 7000000; r2 = 7500000; phase = 1;
n1 = sqrt(MU / r1^3); n2 = sqrt(MU / r2^3);

/* [dr, dv] for the pair, the second plane tilted by inc about the x axis (inc = 0: coplanar). */
rel(t, inc) = {
  my(s1 = sin(n1 * t), c1 = cos(n1 * t), s2 = sin(n2 * t + phase), c2 = cos(n2 * t + phase),
     si = sin(inc), ci = cos(inc));
  [[r2 * c2 - r1 * c1, r2 * s2 * ci - r1 * s1, r2 * s2 * si],
   [-r2 * n2 * s2 + r1 * n1 * s1, r2 * n2 * c2 * ci - r1 * n1 * c1, r2 * n2 * c2 * si]];
}
rate(t, inc) = my(x = rel(t, inc)); x[1] * x[2]~;
sep(t, inc) = my(x = rel(t, inc)); sqrt(x[1] * x[1]~);

minima(inc, span, step) = {
  my(out = List(), t0 = 0, f0 = rate(0, inc), t1, f1);
  while (t0 < span,
    t1 = min(t0 + step, span); f1 = rate(t1, inc);
    if (f0 < 0 && f1 >= 0, listput(out, solve(t = t0, t1, rate(t, inc))));
    t0 = t1; f0 = f1);
  Vec(out);
}

fmt(v) = Strprintf("%.40e", v);

{
my(span = 2 * (2 * Pi / (n1 - n2)), tc = minima(0, span, 15), bad = 0);
print("coplanar pair, two beats (", fmt(span), " s):");
for (j = 1, #tc,
  my(closed = (phase + 2 * Pi * (j - 1)) / (n1 - n2));
  if (abs(tc[j] - closed) > 10^-40, bad++);
  printf("  tca %s s   distance %s m   closed form - root %.1e\n",
         fmt(tc[j]), fmt(sep(tc[j], 0)), closed - tc[j]));
printf("%d coplanar roots off the closed form by more than 1e-40 s\n", bad);
}

{
my(inc = 634 / 10 * Pi / 180, span = 4 * 2 * Pi / n1, tc = minima(inc, span, 15), bad = 0);
print("inclined pair, 63.4 deg, four revolutions of the lower orbit (", fmt(span), " s):");
for (j = 1, #tc,
  my(x = rel(tc[j], inc));
  if (abs(rate(tc[j], inc)) > 10^-40, bad++);
  printf("  tca %s s   distance %s m   speed %s m/s\n",
         fmt(tc[j]), fmt(sep(tc[j], inc)), fmt(sqrt(x[2] * x[2]~))));
printf("%d inclined roots with a residual range rate above 1e-40 m^2/s\n", bad);
}
