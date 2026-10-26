/* RKN12(10)17 (spacelab-propagation/src/rkn1210.rs) against the special Runge-Kutta-Nystrom order
   conditions: the second engine for derivations/rkn1210_order.mac, which states the conditions.

   The literals are parsed from crates/spacelab-propagation/src/rkn1210_consts.rs into the exact
   rationals they denote, and every condition is evaluated in exact rational arithmetic:
       position  sum bhat_i Phi_i(T)  = q(T)/((|T|+1)(|T|+2)),   |T| + 2 <= p,
       velocity  sum bphat_i Phi_i(T) = q(T)/(|T|+1),            |T| + 1 <= p,
   with p = 12 for (bhat, bphat) and p = 10 for (b, bp). Trees are enumerated here as multisets in
   pool order (no sorting), the velocity leaf being the integer 0. A condition holds when its
   residual is below 1e-26 of sum |b_i Phi_i| + |target|; at order p + 1 at least one condition per
   weight vector must fail by more than 1e-6. Any departure ends in error(), which prints ***.

   Run: timeout 300 gp -q -f derivations/rkn1210_order.gp </dev/null
*/

contains(s, pat) =
{
  my(a = Vecsmall(s), b = Vecsmall(pat), n = #a, m = #b);
  for (i = 1, n - m + 1, if (a[i .. i + m - 1] == b, return(1)));
  0;
}
strip(s) =
{
  my(v = Vec(s), keep = select(ch -> ch != " " && ch != "[" && ch != "]" && ch != ",", v));
  if (#keep, concat(keep), "");
}
dec(s) =
{
  my(v = Vec(s), sg = 1, e = 0, digits = List(), frac = 0, seen = 0, i = 1);
  if (v[1] == "-", sg = -1; i = 2);
  while (i <= #v,
    my(ch = v[i]);
    if (ch == "e", e = eval(concat(v[i + 1 .. #v])); break);
    if (ch == ".", seen = 1, listput(~digits, ch); frac += seen);
    i++);
  sg * eval(concat(Vec(digits))) * 10^(e - frac);
}
splitc(s) =
{
  my(v = Vec(s), out = List(), cur = List());
  for (i = 1, #v,
    if (v[i] == ",", listput(~out, if (#cur, concat(Vec(cur)), "")); cur = List(),
      listput(~cur, v[i])));
  if (#cur, listput(~out, concat(Vec(cur))));
  Vec(out);
}
rustconst(lines, name) =
{
  my(out = List(), inside = 0, pat = Str("pub const ", name, ":"));
  for (k = 1, #lines,
    my(l = lines[k]);
    if (inside,
      if (strip(l) == ";", return(Vec(out)));
      foreach (splitc(l), tok, my(w = strip(tok)); if (w != "", listput(~out, dec(w)))),
      if (contains(l, pat), inside = 1)));
  error("constant not found: ", name);
}

file = "crates/spacelab-propagation/src/rkn1210_consts.rs";
lines = iferr(readstr(file), E, readstr(Str("../", file)));
S = 17;
C = rustconst(lines, "C");
BHAT = rustconst(lines, "BHAT");
BPHAT = rustconst(lines, "BPHAT");
B = rustconst(lines, "B");
BP = rustconst(lines, "BP");
AF = rustconst(lines, "A");
if ([#C, #BHAT, #BPHAT, #B, #BP, #AF] != [S, S, S, S, S, S^2], error("table shape"));
A = matrix(S, S, i, j, AF[(i - 1) * S + j]);
for (i = 1, S, for (j = i, S, if (A[i, j] != 0, error("A is not strictly lower triangular"))));

wt(x) = if (type(x) == "t_INT", 1, sz(x) + 2);
sz(t) = sum(k = 1, #t, wt(t[k]));
TREES = Map();
pool(n) =
{
  my(P = List([0]));
  for (j = 0, n - 2, foreach (trees(j), t, listput(~P, t)));
  Vec(P);
}
msets(m, k, P) =
{
  if (m == 0, return([[]]));
  my(out = List());
  for (i = k, #P,
    my(w = wt(P[i]));
    if (w <= m, foreach (msets(m - w, i, P), f, listput(~out, concat([P[i]], f)))));
  Vec(out);
}
trees(n) =
{
  my(r);
  if (mapisdefined(TREES, n, &r), return(r));
  r = if (n == 0, [[]], msets(n, 1, pool(n)));
  mapput(TREES, n, r);
  r;
}
q(t) = prod(k = 1, #t, if (type(t[k]) == "t_INT", 1, q(t[k]) / ((sz(t[k]) + 1) * (sz(t[k]) + 2))));
PHI = Map();
phi(t) =
{
  my(r);
  if (mapisdefined(PHI, t, &r), return(r));
  r = vector(S, i, 1);
  for (k = 1, #t,
    if (type(t[k]) == "t_INT", r = vector(S, i, r[i] * C[i]),
      my(w = phi(t[k])~); r = vector(S, i, r[i] * (A[i, ] * w))));
  mapput(PHI, t, r);
  r;
}
relres(w, t, target) =
{
  my(p = phi(t), x = sum(i = 1, S, w[i] * p[i]));
  abs(x - target) / (sum(i = 1, S, abs(w[i] * p[i])) + abs(target));
}
res(w, t, kind) =
{
  if (kind == "position", relres(w, t, q(t) / ((sz(t) + 1) * (sz(t) + 2))),
    relres(w, t, q(t) / (sz(t) + 1)));
}
treesize(kind, ord) = if (kind == "position", ord - 2, ord - 1);

TOL = 1/10^26;
{
  foreach ([["bhat", BHAT, "position", 2, 12], ["bphat", BPHAT, "velocity", 1, 12],
            ["b", B, "position", 2, 10], ["bp", BP, "velocity", 1, 10]], row,
    my(tag = row[1], w = row[2], kind = row[3], worst = 0, count = 0);
    for (ord = row[4], row[5],
      foreach (trees(treesize(kind, ord)), t,
        my(r = res(w, t, kind));
        count++;
        worst = max(worst, r);
        if (r > TOL, error(tag, " ", kind, " order ", ord, " tree ", t, " residual ", r * 1.))));
    my(fails = #select(t -> res(w, t, kind) > 1/10^6, trees(treesize(kind, row[5] + 1))));
    if (fails == 0, error(tag, " ", kind, " is of order ", row[5] + 1));
    printf("rkn1210 %s %s: %d conditions through order %d hold (worst %.3e); %d of %d miss by more than 1e-6 at order %d\n",
      tag, kind, count, row[5], worst * 1., fails, #trees(treesize(kind, row[5] + 1)), row[5] + 1));
}
counts = vector(13, n, #trees(n - 1));
if (counts != [1, 1, 2, 3, 6, 10, 20, 36, 72, 137, 275, 541, 1098], error("tree counts ", counts));
print("special-Nystrom trees per size 0..12: ", counts);
quit;
