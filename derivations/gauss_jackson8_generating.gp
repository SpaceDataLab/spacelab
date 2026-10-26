/* Gauss-Jackson 8 (spacelab-propagation/src/gj.rs): the second engine for
   derivations/gauss_jackson8_generating.mac, by the other route. Maxima expands generating
   functions; here each weight set is the exact solution of its moment (exactness) system, h = 1,
   t_n = 0:
     Stormer, nodes 0, -1, ..., -7:  sum_j w_j t_j^(m-2) = (1 + (-1)^m) / (m (m - 1)),  m = 2..9;
     Cowell,  nodes 1,  0, ..., -6:  the same system;
     Adams-Moulton (velocity), nodes 1, 0, ..., -6:  sum_j w_j t_j^(m-1) = 1/m,      m = 1..8.
   Each solution must equal the generating-function rationals Maxima prints, and every literal in
   gj.rs (STORMER, COWELL) and abm.rs (AM8) must be that weight correctly rounded at its last decimal
   place. The next moment (m = 10, resp. 9) must fail, so the order is exactly 8. Any departure ends
   in error(), which prints ***.

   Run: timeout 300 gp -q -f derivations/gauss_jackson8_generating.gp </dev/null
*/

contains(s, pat) =
{
  my(a = Vecsmall(s), b = Vecsmall(pat), n = #a, m = #b);
  for (i = 1, n - m + 1, if (a[i .. i + m - 1] == b, return(1)));
  0;
}
strip(s) =
{
  my(keep = select(ch -> ch != " " && ch != "[" && ch != "]" && ch != ",", Vec(s)));
  if (#keep, concat(keep), "");
}
/* A decimal literal as [exact rational, fractional digits]. */
dec(s) =
{
  my(v = Vec(s), sg = 1, digits = List(), frac = 0, seen = 0, i = 1);
  if (v[1] == "-", sg = -1; i = 2);
  for (k = i, #v,
    if (v[k] == "e", error("exponent form not expected here: ", s));
    if (v[k] == ".", seen = 1, listput(~digits, v[k]); frac += seen));
  [sg * eval(concat(Vec(digits))) / 10^frac, frac];
}
rustconst(file, name) =
{
  my(lines = iferr(readstr(file), E, readstr(Str("../", file))), out = List(), inside = 0);
  my(pat = Str("const ", name, ":"));
  for (k = 1, #lines,
    my(l = lines[k]);
    if (inside,
      if (strip(l) == ";", return(Vec(out)));
      my(w = strip(l)); if (w != "", listput(~out, dec(w))),
      if (contains(l, pat), inside = 1)));
  error("constant not found: ", name);
}

pw(x, m) = if (m == 0, 1, x^m);
second(nodes) = matsolve(matrix(8, 8, r, j, pw(nodes[j], r - 1)), vector(8, r, my(m = r + 1); (1 + (-1)^m) / (m * (m - 1)))~)~;
first(nodes) = matsolve(matrix(8, 8, r, j, pw(nodes[j], r - 1)), vector(8, r, 1/r)~)~;
stormer = second(vector(8, j, 1 - j));
cowell = second(vector(8, j, 2 - j));
am8 = first(vector(8, j, 2 - j));

if (stormer != [22081/15120, -4511/2240, 40933/10080, -300227/60480, 9857/2520, -39017/20160, 3319/6048, -275/4032], error("Stormer ", stormer));
if (cowell != [275/4032, 13831/15120, -2099/20160, 811/3360, -11477/60480, 29/315, -517/20160, 19/6048], error("Cowell ", cowell));
if (am8 != [5257/17280, 139849/120960, -4511/4480, 123133/120960, -88547/120960, 1537/4480, -11351/120960, 275/24192], error("Adams-Moulton ", am8));

next2(w, nodes) = (1 + 1) / (10 * 9) - sum(j = 1, 8, w[j] * nodes[j]^8);
next1(w, nodes) = 1/9 - sum(j = 1, 8, w[j] * nodes[j]^8);
e = [next2(stormer, vector(8, j, 1 - j)), next2(cowell, vector(8, j, 2 - j)), next1(am8, vector(8, j, 2 - j))];
if (#select(x -> x == 0, e), error("a formula is exact one degree further: ", e));

check(tag, exact, lits) =
{
  if (#lits != 8, error(tag, ": expected 8 literals"));
  for (j = 1, 8,
    my(v = lits[j][1], k = lits[j][2]);
    if (round(exact[j] * 10^k) != v * 10^k, error(tag, " w", j - 1, " literal is not the rounded weight")));
}
check("Stormer", stormer, rustconst("crates/spacelab-propagation/src/gj.rs", "STORMER"));
check("Cowell", cowell, rustconst("crates/spacelab-propagation/src/gj.rs", "COWELL"));
check("Adams-Moulton", am8, rustconst("crates/spacelab-propagation/src/abm.rs", "AM8"));

print("gauss-jackson 8: moment solutions equal the generating-function weights; 24 literals correctly rounded");
print("residual of the first moment past order 8 (Stormer, Cowell, Adams-Moulton), nonzero: ", e);
quit;
