/* The binary64 (and binary32) a real rounds to, as an exact rational: IEEE 754 round to nearest,
   ties to even, with gradual underflow; a value that rounds beyond the largest finite is an error.
   The PARI derivations that model the doubles a test passes read this file, as

       default(path, "derivations:."); read("binary64.gp");

   so that it runs from the repository root, as the derivations' headers say, and from
   derivations/, as verify.sh runs it.

   Why the binade is not exponent(x): for a t_FRAC a/b PARI *defines* exponent() as
   exponent(a) - exponent(b) (users' manual, exponent), one above the binade whenever a's leading
   bits are below b's: exponent(1/3) = -1 while 1/3 < 1/2. Scaled by 2^(52 - exponent(x)), such an x
   is rounded to 52 bits, and lands one ulp off whenever its double has an odd significand. PARI's
   round() also breaks a tie towards +oo, not to even. Here the binade is exponent() corrected by
   one comparison with a power of two, exact for every type, and the tie is broken by hand.

   On read the file checks itself against doubles fixed elsewhere -- the bit patterns of Julia's
   Float64 and Float32 of each value, each recomputed in Maxima as float() of the exact value, and
   by rust_literals.mac's nearest_double_q for the normal ones -- prints a FAIL line for any miss
   and stops with an error, so a wrong rounding fails the gate of every derivation that reads it.
   verify.sh lists every case, and keeps the idiom this file replaced as its planted defect. */

binary_round(x, p, emin) = {
  my(a = abs(x), e, m, f, d, y);
  if (a == 0, return(0));
  e = exponent(a); if (a < 2^e, e--);
  e = max(e, emin);
  m = a * 2^(p - 1 - e); f = floor(m); d = m - f;
  if (d > 1/2 || (d == 1/2 && f % 2), f++);
  y = f * 2^(e - p + 1);
  if (y >= 2^(2 - emin), error("binary_round: ", x, " is beyond the largest finite of ", p, " bits"));
  sign(x) * y;
}
f64(x) = binary_round(x, 53, -1022);
f32(x) = binary_round(x, 24, -126);

/* The exact value of an IEEE bit pattern with p - 1 fraction bits and w exponent bits. */
binary_bits(h, p, w) = {
  my(s = h >> (p - 1 + w), e = bitand(h >> (p - 1), 2^w - 1), m = bitand(h, 2^(p - 1) - 1),
     bias = 2^(w - 1) - 1);
  (-1)^s * if (e == 0, m * 2^(2 - bias - p), (2^(p - 1) + m) * 2^(e - bias - p + 1));
}

/* The number of misses of fn64 and fn32 against the fixed doubles; each miss is printed as FAIL
   with its distance in units in the last place, and each hit as ok when verbose. */
binary64_check(fn64, fn32, verbose) = {
  my(cases = [["1/3", 0x3fd5555555555555], ["2/7", 0x3fd2492492492492], ["-19/10", 0xbffe666666666666],
               ["4/100", 0x3fa47ae147ae147b], ["1 - 10^-3", 0x3feff7ced916872b],
               ["(3*2^52 - 1)/(3*2^52)", 0x3fefffffffffffff], ["2^53 + 1", 0x4340000000000000],
               ["-(2^53 + 3)", 0xc340000000000002], ["5/(3*2^1074)", 0x0000000000000002],
               ["(3*2^52 - 1)/(3*2^1074)", 0x0010000000000000], ["Pi", 0x400921fb54442d18]],
     cases32 = [["1/3", 0x3eaaaaab], ["4/100", 0x3d23d70a], ["2^24 + 1", 0x4b800000],
                ["5/(3*2^149)", 0x00000002]],
     bad = 0,
     one = (fn, p, w, c) -> my(got = fn(eval(c[1])), want = binary_bits(c[2], p, w),
                               ulp = 2^(max(exponent(abs(want)), 2 - 2^(w - 1)) - p + 1));
       if (verbose || got != want,
         printf("%s binary%d(%s) = %.17g, %.4g ulp from the double %.17g\n", if (got == want, "ok", "FAIL"),
                p + w, c[1], got, (got - want) / ulp, want));
       got != want);
  foreach (cases, c, bad += one(fn64, 53, 11, c));
  foreach (cases32, c, bad += one(fn32, 24, 8, c));
  bad;
}
if (binary64_check(f64, f32, 0), error("binary64.gp: the rounding misses a double it is checked against"));
