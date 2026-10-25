/* The reference values `spacelab-propagation`'s lens is gated on, at 200 digits.

   `shadow::lens_area(d, r_e, r_s)` is the area two discs share in the plane, radii r_e (the
   occulter) and r_s (the Sun), centres d apart. Here it is evaluated from the textbook
   circle-intersection formula, a different arrangement from the shipped one,

       A = r_s^2 acos((d^2 + r_s^2 - r_e^2)/(2 d r_s)) + r_e^2 acos((d^2 + r_e^2 - r_s^2)/(2 d r_e))
           - sqrt((r_e + r_s - d)(d - r_e + r_s)(d + r_e - r_s)(d + r_e + r_s))/2,

   on exact binary64 inputs: every radius and separation below is rounded to the nearest double
   first, and the formula is then evaluated on that double as an exact rational, so the reference
   is the area *of the inputs the test passes*, not of the decimals they were made from. The
   acos arguments and Heron's product are exact rationals; only acos, sqrt and pi are rounded, at
   200 digits. Near a tangency the acos form loses digits -- that is the point of the shipped one
   -- but at 200 digits it keeps more than 160 at the closest row, 2^-52 from the tangency.

   Seven radius pairs: the Earth from 400 km, GEO and the Moon's distance against the Sun at 1 au;
   the shadow.rs test pair (0.1, 0.01); an occulter 6 % larger than the Sun, one equal to it, and
   one 0.6 of it, which sits inside the Sun's disc. For each, separations 2^-2 ... 2^-52 (relative)
   inside both tangencies, the tangencies themselves and their neighbouring doubles, five points
   across the penumbra and three towards d = 0.

   Each lens row is (r_e, r_s, d, A_hi, A_lo, nu_hi, nu_lo, kappa): A = A_hi + A_lo and
   nu = 1 - A/(pi r_s^2) as double-double pairs, kappa = 4K/A the area's condition number in d
   (0 on the plateaus). The x - sin x rows are (x, hi, lo), for the series the shipped segment uses
   below x = 2. Every row is printed in Rust syntax and must appear verbatim in
   crates/spacelab-propagation/tests/shadow_lens.rs or src/shadow.rs; `derivations/verify.sh`
   checks that, and `shadow_lens_arb.c` checks every value against an Arb ball.

       gp -q -f derivations/shadow_lens.gp </dev/null                                            */

default(realprecision, 200);

/* The nearest binary64 to x, normal range, as an exact rational. */
dbl(x) = {
  my(r = x * 1., e);
  if (r == 0, return(0));
  e = exponent(r);
  round(x * 2^(52 - e)) * 2^(e - 52);
}
/* The neighbouring doubles of a double y > 0. */
up(y) = 2^(exponent(y * 1.) - 52) + y;
down(y) = { my(e = exponent(y * 1.)); if (y == 2^e, y - 2^(e - 53), y - 2^(e - 52)); }

/* The shortest decimal in Rust syntax that parses back to the same double: at most 17 digits. */
lit(y) = {
  my(s, e10, mant);
  if (y == 0, return("0.0"));
  e10 = floor(log(abs(y * 1.)) / log(10));
  mant = y * 1. / 10^e10;
  if (abs(mant) >= 10, mant /= 10; e10++);
  if (abs(mant) < 1, mant *= 10; e10--);
  for (k = 0, 16,
    s = Strprintf(Str("%.", k, "fe%d"), mant, e10);
    if (dbl(eval(s)) == y, return(s)));
  error("no round trip for ", y);
}
sci(x) = {
  my(e10, mant);
  if (x == 0, return("0.0"));
  e10 = floor(log(abs(x * 1.)) / log(10));
  mant = x * 1. / 10^e10;
  if (abs(mant) >= 10, mant /= 10; e10++);
  Strprintf("%.3fe%d", mant, e10);
}
/* A double-double pair: hi the nearest double, lo the nearest double to the rest. */
pair(x) = { my(hi = dbl(x)); [hi, if (x == hi, 0, dbl(x - hi))]; }

heron(re, rs, d) = (re + rs - d) * (d - re + rs) * (d + re - rs) * (d + re + rs);
area(re, rs, d) = {
  if (d >= re + rs, return(0));
  if (d <= abs(re - rs), return(Pi * min(re, rs)^2));
  rs^2 * acos((d^2 + rs^2 - re^2) / (2 * d * rs))
    + re^2 * acos((d^2 + re^2 - rs^2) / (2 * d * re)) - sqrt(heron(re, rs, d)) / 2;
}
kappa(re, rs, d) = if (d >= re + rs || d <= abs(re - rs), 0, sqrt(heron(re, rs, d)) / area(re, rs, d));

lens_row(re, rs, d) = {
  my(a = area(re, rs, d), ap, np);
  ap = pair(a);
  np = pair(1 - a / (Pi * rs^2));
  printf("    (%s, %s, %s, %s, %s, %s, %s, %s),\n", lit(re), lit(rs), lit(d), lit(ap[1]), lit(ap[2]),
         lit(np[1]), lit(np[2]), sci(kappa(re, rs, d)));
}

rsun = dbl(asin(695700000 / 149597870700));
{
pairs = [[dbl(asin(6378137 / 6778137)), rsun], [dbl(asin(6378137 / 42164172)), rsun],
         [dbl(asin(6378137 / 384400000)), rsun], [dbl(1/10), dbl(1/100)],
         [dbl(106/100 * rsun), rsun], [rsun, rsun], [dbl(6/10 * rsun), rsun]];
}
print("// lens");
{
for (i = 1, #pairs,
  my(re = pairs[i][1], rs = pairs[i][2], s = re + rs, m = abs(re - rs), ds = List());
  foreach ([2, 10, 20, 30, 40, 48, 52], j, listput(ds, dbl(s - s * 2^-j)));
  listput(ds, dbl(s)); listput(ds, up(dbl(s)));
  if (m > 0,
    foreach ([2, 10, 20, 30, 40, 48, 52], j, listput(ds, dbl(m + m * 2^-j)));
    listput(ds, dbl(m)); listput(ds, down(dbl(m))));
  foreach ([1, 2, 4, 6, 7], k, listput(ds, dbl(m + (s - m) * k / 8)));
  foreach ([1, 20, 52], j, listput(ds, dbl(rs * 2^-j)));
  foreach (vecsort(Vec(ds), , 8), d, if (d > 0, lens_row(re, rs, d)));
);
}
print("// x - sin x");
{
foreach ([2^-30, dbl(1/10^8), dbl(1/1000), 1/2, 1, 3/2, down(2), 2, up(2), 5/2, dbl(Pi), 6, dbl(2 * Pi)], x,
  my(p = pair(x - sin(x)));
  printf("    (%s, %s, %s),\n", lit(x), lit(p[1]), lit(p[2])));
}
quit
