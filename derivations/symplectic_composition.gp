/* The symplectic compositions of spacelab-propagation/src/symplectic.rs -- leapfrog, yoshida4,
   yoshida6, yoshida8 -- checked directly on the sub-step fractions the code builds: the second
   engine for derivations/symplectic_composition.mac, which proves the triple-jump construction.

   One velocity-Verlet sub-step of fraction g is exp(g B/2) exp(g A) exp(g B/2), A the drift and B
   the kick, in the free associative algebra on {A, B}; with h = 1 a word of length d carries h^d.
   The exact flow is exp(A + B), whose coefficient on every word of length d is 1/d!. A composition
   is of order p iff its product agrees with that on every word of length <= p. Computed here at
   \p 100 through words of length 9 (1023 words), for the fraction lists
       leapfrog  [1],
       yoshida4  [w1, w0, w1],  w1 = 1/(2 - 2^(1/3)), w0 = -2^(1/3) w1,
       yoshida6  triple jump of yoshida4 with z = 1/(2 - 2^(1/5)),
       yoshida8  triple jump of yoshida6 with z = 1/(2 - 2^(1/7)),
   exactly as `triple_jump` lays them out: [z s, (1 - 2z) s, z s]. Required: every word of length
   <= p within 1e-90, some word of length p + 1 off by more than 1e-6, and Psi(h) Psi(-h) = 1 within
   1e-90 through length 9 (time symmetry). Any departure ends in error(), which prints ***.

   It also prints the one-step map of each method on the unit oscillator x'' = -x at h = 1/2 --
   the product of the sub-steps' kick [1, 0; -c, 1] and drift [1, c; 0, 1] matrices -- which the
   test `symplectic::gate_tests::one_step_on_the_oscillator_is_the_composition_map` holds the
   shipped functions to.

   Run: timeout 300 gp -q -f derivations/symplectic_composition.gp </dev/null
*/

default(realprecision, 100);
W = 9;
NW = 2^(W + 1) - 1;
len(w) = #binary(w) - 1;

/* X * exp(c L), L = 0 for A, 1 for B: strip up to all trailing L letters of each word. */
mulexp(X, c, L) =
{
  my(Y = vector(NW));
  for (w = 1, NW,
    my(s = X[w], p = w, k = 0);
    while (p > 1 && p % 2 == L, p = p >> 1; k++; s += X[p] * c^k / k!);
    Y[w] = s);
  Y;
}
substep(X, g) = mulexp(mulexp(mulexp(X, g/2, 1), g, 0), g/2, 1);
compose(fr) =
{
  my(X = vector(NW)); X[1] = 1;
  foreach (fr, g, X = substep(X, g));
  X;
}
triple(base, q) = my(z = 1 / (2 - 2^(1/q))); concat([z * base, (1 - 2*z) * base, z * base]);
s2 = [1];
s4 = [1 / (2 - 2^(1/3)), -2^(1/3) / (2 - 2^(1/3)), 1 / (2 - 2^(1/3))];
s6 = triple(s4, 5);
s8 = triple(s6, 7);

/* (X Y)[w] = sum over splits w = u v of X[u] Y[v]. */
mul(X, Y) =
{
  my(Z = vector(NW));
  for (w = 1, NW,
    my(d = len(w), s = 0);
    for (k = 0, d, my(u = w >> k, v = 2^k + (w % 2^k)); s += X[u] * Y[v]);
    Z[w] = s);
  Z;
}
reversed(X) = vector(NW, w, (-1)^len(w) * X[w]);

order_check(name, fr, p) =
{
  my(X = compose(fr), dev = vector(W, d, 0), sym);
  for (w = 2, NW, my(d = len(w)); dev[d] = max(dev[d], abs(X[w] - 1 / d!)));
  if (abs(X[1] - 1) > 0, error(name, ": empty word"));
  for (d = 1, p, if (dev[d] > 1e-90, error(name, ": length ", d, " off by ", dev[d])));
  if (dev[p + 1] < 1e-6, error(name, ": order above ", p));
  sym = mul(X, reversed(X)); sym[1] -= 1;
  if (vecmax(apply(abs, sym)) > 1e-90, error(name, ": not time-symmetric"));
  printf("%s: %d sub-steps, exact through words of length %d, worst length-%d deviation %.3e; Psi(h) Psi(-h) = 1\n",
    name, #fr, p, p + 1, dev[p + 1]);
}
order_check("leapfrog", s2, 2);
order_check("yoshida4", s4, 4);
order_check("yoshida6", s6, 6);
order_check("yoshida8", s8, 8);

/* The one-step map on x'' = -x, column (x, v), h = 1/2. */
oscmap(fr, h) =
{
  my(M = matid(2));
  foreach (fr, g,
    my(c = g * h);
    M = [1, 0; -c/2, 1] * [1, c; 0, 1] * [1, 0; -c/2, 1] * M);
  M;
}
H = 1/2;
R = [cos(H), sin(H); -sin(H), cos(H)];
{
  foreach ([["leapfrog", s2], ["yoshida4", s4], ["yoshida6", s6], ["yoshida8", s8]], row,
    my(M = oscmap(row[2], H), E = M - R);
    if (abs(matdet(M) - 1) > 1e-90, error(row[1], ": det of the oscillator map is not 1"));
    printf("%s oscillator map at h = 1/2: [%.36e, %.36e, %.36e, %.36e]; |M - R| = %.3e\n",
      row[1], M[1, 1], M[2, 1], M[1, 2], M[2, 2], vecmax(apply(abs, concat(Vec(E[, 1]), Vec(E[, 2]))))));
}
quit;
