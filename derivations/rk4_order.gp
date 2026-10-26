/* Classical RK4 (spacelab-propagation/src/rk4.rs) against the Butcher order conditions, in exact
   rationals: the second engine for derivations/rk4_order.mac.

   The trees are built differently here: every tree of order n comes from one of order n - 1 by
   adding a leaf to one of its vertices, canonicalised by sorting subtrees recursively and
   deduplicated. Their counts (1, 1, 2, 4, 9) and the residuals of sum_i b_i Phi_i(t) - 1/gamma(t)
   must match the Maxima run: zero through order 4, the nine order-5 values nonzero.
   The tableau is the one rk4_step forms: c = (0, 1/2, 1/2, 1), a21 = a32 = 1/2, a43 = 1,
   b = (1, 2, 2, 1)/6.

   Run: timeout 300 gp -q -f derivations/rk4_order.gp </dev/null
*/

S = 4;
C = [0, 1/2, 1/2, 1];
A = [0, 0, 0, 0; 1/2, 0, 0, 0; 0, 1/2, 0, 0; 0, 0, 1, 0];
B = [1/6, 1/3, 1/3, 1/6];

canon(t) = vecsort(apply(canon, t), lex);
grow(t) =
{
  my(out = List([canon(concat(t, [[]]))]));
  for (i = 1, #t,
    foreach (grow(t[i]), u,
      my(w = t); w[i] = u; listput(~out, canon(w))));
  Vec(out);
}
trees = vector(5);
trees[1] = [[]];
for (n = 2, 5, trees[n] = vecsort(concat(apply(grow, trees[n - 1])), lex, 8));

ord(t) = 1 + sum(k = 1, #t, ord(t[k]));
gam(t) = ord(t) * prod(k = 1, #t, gam(t[k]));
phi(t) =
{
  my(p = vector(S, i, 1));
  for (k = 1, #t, my(w = phi(t[k])~); p = vector(S, i, p[i] * (A[i, ] * w)));
  p;
}
res(t) = sum(i = 1, S, B[i] * phi(t)[i]) - 1/gam(t);

if (apply(length, trees) != [1, 1, 2, 4, 9], error("tree counts ", apply(length, trees)));
if (vector(S, i, vecsum(A[i, ])) != C, error("row sums of A are not the nodes c"));
{
  for (n = 1, 4,
    foreach (trees[n], t,
      if (res(t) != 0, error("order ", n, " tree ", t, " residual ", res(t)))));
}
r5 = apply(res, trees[5]);
if (#select(x -> x == 0, r5) > 0, error("an order-5 condition holds: ", r5));
print("rk4 row sums equal c; 8 order conditions through 4 exact: ok");
print("rk4 order-5 residuals (none zero): ", r5);
quit;
