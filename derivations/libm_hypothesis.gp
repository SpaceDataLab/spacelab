/* The second engine for H-libm (plans/18 Ф28.J), in PARI/GP at 600 digits,
   a different engine from libm_hypothesis_arb.c's Arb balls.

   verify.sh appends one call per row of the C file's report and one for its
   hardest argument, and pipes the whole into `gp -q -f`:

   worst(n, row, fn, a1, a2, r, arb)   the error of libm's answer r at the input
       the C file reports as the row's largest, recomputed here as
       |r - f| / ulp(f) with f at 600 digits and ulp(f) = 2^(max(e, -1022) - 52),
       e = floor(log2 |f|), on the exact doubles the hex digits encode. It must
       agree with Arb's value to 1e-5 ulp (the C file prints six decimals).
   hardest(n, m, k, dist)   the hardest argument of the reduction by pi/2,
       found again from PARI's own continued fraction of frac(2^(k+1)/pi) for
       every binade k = -52 ... 971: the denominators in [2^52, 2^53), the last
       semiconvergent below 2^53 and the smallest multiple of a smaller
       denominator at or above 2^52, and the least |m 2^k - j pi/2| over all of
       them. It must be the C file's m 2^k, at the same distance to 1e-5.

   Each call prints `(n) holds: ...` or stops with an error, so a mismatch is a
   `***` line and no `holds`. */

default(realprecision, 600);

dbl(h) =
{
  my(b = eval(Str("0x", h)), s = b >> 63, e = bitand(b >> 52, 2047),
     m = bitand(b, 2^52 - 1));
  (-1)^s * if (e == 0, m * 2^-1074, (2^52 + m) * 2^(e - 1075));
}

exact(fn, x, y) =
{
  if (fn == "sin" || fn == "sincos_sin", return (sin(x)));
  if (fn == "cos" || fn == "sincos_cos", return (cos(x)));
  if (fn == "tan", return (tan(x)));
  if (fn == "exp", return (exp(x)));
  if (fn == "log", return (log(x)));
  if (fn == "acos", return (acos(x)));
  if (fn == "asin", return (asin(x)));
  if (fn == "atan", return (atan(x)));
  if (fn == "sqrt", return (sqrt(x)));
  if (fn == "cbrt", return (sign(x) * sqrtn(abs(x), 3)));
  if (fn == "atan2", return (arg(y + I * x)));
  if (fn == "hypot", return (sqrt(x^2 + y^2)));
  error("unknown function ", fn);
}

worst(n, row, fn, a1, a2, r, arb) =
{
  my(x = dbl(a1), y = if (a2 == "", 0, dbl(a2)), f = exact(fn, x, y), e);
  e = abs(dbl(r) - f) / 2^(max(exponent(f), -1022) - 52);
  if (abs(e - arb) > 1e-5,
    error(row, ": PARI finds ", e, " ulp where Arb found ", arb));
  printf("(%d) holds: %s, %s at its worst input, %.6f ulp (Arb %.6f)\n",
         n, row, fn, e, arb);
}

hardest(n, mc, kc, dc) =
{
  my(best = [1, 0, 0], two52 = 2^52, two53 = 2^53);
  for (k = -52, 971,
    my(cf = contfrac(frac(2^(k + 1) / Pi), , 200), q = 1, qp = 0, c = List());
    for (i = 2, #cf,
      my(a = cf[i], t, qn);
      if (q >= two52, listput(c, q),
        t = ceil(two52 / q) * q; if (t < two53, listput(c, t)));
      t = min(a, (two53 - 1 - qp) \ q);
      if (t > 0 && qp + t * q >= two52 && qp + t * q < two53,
        listput(c, qp + t * q));
      qn = a * q + qp; qp = q; q = qn;
      if (q >= two53, break));
    for (j = 1, #c,
      my(x = c[j] * 2^k, d = abs(x - round(2 * x / Pi) * Pi / 2));
      if (d < best[1], best = [d, c[j], k])));
  if (best[2] != mc || best[3] != kc || abs(best[1] / dc - 1) > 1e-5,
    error("the hardest argument is ", best[2], " 2^", best[3], " at ",
          best[1], ", not ", mc, " 2^", kc, " at ", dc));
  printf("(%d) holds: the double nearest a multiple of pi/2 is %d 2^%d, %.6g from it (2^%.2f)\n",
         n, mc, kc, best[1], log(best[1]) / log(2));
}
