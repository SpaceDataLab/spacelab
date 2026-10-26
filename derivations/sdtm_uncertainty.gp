/* Second engine for derivations/sdtm_uncertainty.mac: the two-sided standard-normal quantiles of
   crates/spacelab-propagation/src/sdtm/uncertainty.rs at 100 digits, by root-finding
   erfc(z/sqrt 2) = 1 - p rather than through an inverse erf, and the shipped f64 literals held to
   half an ulp of them. Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_uncertainty.gp </dev/null                          */

default(realprecision, 100);
check(tag, p, literal) =
{
  my(z = solve(x = 1, 3, erfc(x/sqrt(2)) - (1 - p)), ulp = if (literal < 2, 2^-52, 2^-51), d);
  d = abs(literal - z);
  print(tag, ": z = ", Strprintf("%.60g", z));
  print(tag, if (d <= ulp/2, ": PASS", ": FAIL"), " literal is ", Strprintf("%.4f", d/ulp), " ulp from z");
  print(tag, if (abs(1 - erfc(z/sqrt(2)) - p) < 10^-95, ": PASS", ": FAIL"), " coverage residual");
}
/* The decimal literal as written in uncertainty.rs, an exact rational, rounded to the double the
   compiler makes of it: 53 significant bits, ulp 2^-52 in [1, 2) and 2^-51 in [2, 4). */
dbl(q) = my(e = if (q < 2, 52, 51)); round(q*2^e)/2^e;
check("normal_90", 9/10, dbl(16448536269514727/10^16));
check("normal_95", 95/100, dbl(19599639845400543/10^16));
check("normal_99", 99/100, dbl(2575829303548901/10^15));
