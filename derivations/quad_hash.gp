/* Second engine for derivations/quad_hash.mac: the quad code as PARI's exact complex rationals,
   w = (1 + I)(z - a)/(b - a), with the canonical ordering applied to w directly rather than to the
   real-coordinate formulas the code and Maxima use.

   The integer quad of the Rust gate, its image under z -> (3 + 4I) z + 17 - 5I, and the same quad
   listed with B first must all give the code Maxima printed, exactly; the summary counts the
   checks that differ.

       timeout 300 gp -q -f derivations/quad_hash.gp </dev/null                                   */

off = 0; checked = 0;
{
same(x, y) =
  checked++;
  if (x != y, off++; print("  off: ", x, " vs ", y));
}
{
canon(P) =
  my(best = -1, ia, ib, rest = List(), c, d, wc, wd, t);
  for (i = 1, 4, for (j = i + 1, 4, my(s = norm(P[i] - P[j])); if (s > best, best = s; ia = i; ib = j)));
  for (k = 1, 4, if (k != ia && k != ib, listput(rest, k)));
  c = rest[1]; d = rest[2];
  wc = (1 + I)*(P[c] - P[ia])/(P[ib] - P[ia]); wd = (1 + I)*(P[d] - P[ia])/(P[ib] - P[ia]);
  if (real(wc) > real(wd), t = c; c = d; d = t; t = wc; wc = wd; wd = t);
  if (real(wc) + real(wd) > 1,
    t = ia; ia = ib; ib = t;
    wc = (1 + I)*(P[c] - P[ia])/(P[ib] - P[ia]); wd = (1 + I)*(P[d] - P[ia])/(P[ib] - P[ia]);
    if (real(wc) > real(wd), t = c; c = d; d = t; t = wc; wc = wd; wd = t));
  [[real(wc), imag(wc), real(wd), imag(wd)], [ia - 1, ib - 1, c - 1, d - 1]];
}
{
my(quad = [0, 37 + 11*I, 14 + 19*I, 21 - 4*I], code = [89/745, 638/745, 556/745, 177/745], r);
r = canon(quad); same(r[1], code); same(r[2], [0, 1, 2, 3]);
r = canon(apply(z -> (3 + 4*I)*z + 17 - 5*I, quad)); same(r[1], code); same(r[2], [0, 1, 2, 3]);
r = canon([quad[2], quad[1], quad[3], quad[4]]); same(r[1], code); same(r[2], [1, 0, 2, 3]);
/* both inner stars inside the circle on AB: |w - (1 + I)/2|^2 <= 1/2 */
same(norm(code[1] + I*code[2] - (1 + I)/2) <= 1/2, 1);
same(norm(code[3] + I*code[4] - (1 + I)/2) <= 1/2, 1);
}
printf("quad_hash: %d values checked, %d differ\n", checked, off);
