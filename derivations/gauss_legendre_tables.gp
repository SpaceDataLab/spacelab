/* The Gauss–Legendre tables of spacelab-propagation/src/dsst/gaussian.rs (`POINTS_n`, `WEIGHTS_n`,
   n in GAUSS_ORDER), generated from their definition rather than transcribed:

       nodes   x_i = the n real roots of the Legendre polynomial P_n,
       weights w_i = 2 / ((1 - x_i^2) P_n'(x_i)^2),

   at 80 digits, each rounded to the nearest double and printed in the shortest decimal that parses
   back to that double. The output is the Rust text of the tables; verify.sh compares it with the
   block in gaussian.rs byte for byte. derivations/gauss_legendre_tables.mac re-derives the same
   doubles by Newton's method on P_n in Maxima and checks the rule's exactness.

   Run from derivations/, as verify.sh does:
       timeout 120 gp -q -f gauss_legendre_tables.gp </dev/null                                 */

default(realprecision, 80);

/* The nearest double to x (53-bit significand; a tie cannot occur for these irrationals). */
nd(x) = if (x == 0, 0, my(e = exponent(x)); round(x * 2^(52 - e)) * 2^(e - 52));

/* The shortest %.Ng decimal whose nearest double is d. */
{shortest(d) = for (k = 1, 17, my(s = Strprintf(Str("%.", k, "g"), d));
   if (nd(eval(s)) == d, return(s)));
 error("no 17-digit decimal round-trips ", d);}

{table(name, n, v) = my(s = Str("const ", name, "_", n, ": [f64; ", n, "] = ["));
  for (i = 1, n, s = Str(s, if (i > 1, ", ", ""), shortest(v[i])));
  print("#[rustfmt::skip]");
  print(s, "];");}

orders = [12, 16, 20, 24, 32, 40, 48];
{nodes = vector(#orders, j, my(n = orders[j], p = pollegendre(n), r = polrootsreal(p));
  [n, r, vector(n, i, 2 / ((1 - r[i]^2) * subst(p', 'x, r[i])^2))]);}
for (j = 1, #orders, table("POINTS", orders[j], apply(nd, nodes[j][2])));
for (j = 1, #orders, table("WEIGHTS", orders[j], apply(nd, nodes[j][3])));
