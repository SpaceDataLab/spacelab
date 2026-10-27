/* Second engine for derivations/polsar_decomposition.mac, at 60 digits.

   The same canonical scatterers, from the scattering matrix through PARI's own eigen-solver
   (mateigen on the Hermitian coherency matrix): sphere, dihedral, horizontal dipole and the
   six-look dipole cloud psi_m = m pi/6, each against its closed (H, A, alpha_bar). Then, for 200
   pseudo-random complex scattering triples, the characteristic polynomial of k k^H is
   l^3 - span l^2: rank one, its eigenvalues summing to the span |S_hh|^2 + 2|S_hv|^2 + |S_vv|^2.

       timeout 300 gp -q -f derivations/polsar_decomposition.gp </dev/null                         */

default(realprecision, 60);
tol = 10^-40;
rnd() = seed = (1103515245 * seed + 12345) % 2^31; seed / 2^30 - 1;

pauli(s) = [s[1] + s[3], s[1] - s[3], 2 * s[2]]~ / sqrt(2);
outer(k) = k * conj(k~);

/* (H, A, alpha_bar), eigenvalues sorted decreasing; clean negative round-off to zero. */
haalpha(t) = {
  my(ev = mateigen(t, 1), l = real(ev[1]), v = ev[2], n = #l, idx, p, h = 0, al = 0, an);
  idx = vecsort(l, , 1 + 4);
  l = vector(n, i, max(l[idx[i]], 0));
  p = l / vecsum(l);
  for (i = 1, n,
    if (p[i] > tol, h -= p[i] * log(p[i]) / log(3));
    my(e = v[, idx[i]]);
    al += p[i] * acos(min(1, abs(e[1]) / sqrt(norml2(e)))));
  an = if (p[2] + p[3] > tol, (p[2] - p[3]) / (p[2] + p[3]), 0);
  [h, an, al];
}

{
my(cases = [
     ["sphere", outer(pauli([1, 0, 1])), [0, 0, 0]],
     ["dihedral", outer(pauli([1, 0, -1])), [0, 0, Pi / 2]],
     ["horizontal dipole", outer(pauli([1, 0, 0])), [0, 0, Pi / 4]],
     ["dipole cloud", sum(m = 0, 5, my(c = cos(m * Pi / 6), s = sin(m * Pi / 6));
                          outer(pauli([c^2, c * s, s^2]))) / 6, [3 * log(2) / (2 * log(3)), 0, Pi / 4]]],
   off = 0, total = 0);
for (i = 1, #cases,
  my(r = haalpha(cases[i][2]), w = cases[i][3]);
  for (j = 1, 3, total++; if (abs(r[j] - w[j]) > tol, off++; print("  ", cases[i][1], " ", j, ": ", r[j]))));
printf("canonical scatterers: %d of %d quantities off by more than 1e-40\n", off, total);
}

{
my(off = 0);
seed = 12345;
for (t = 1, 200,
  my(s = vector(3, j, rnd() + I * rnd()), k = pauli(s), cp = Vec(charpoly(outer(k))),
     span = norm(s[1]) + 2 * norm(s[2]) + norm(s[3]));
  /* rank one with trace = span: the characteristic polynomial is l^3 - span l^2 */
  if (normlp(cp - [1, -span, 0, 0]) > tol, off++));
printf("random triples: %d of 200 off by more than 1e-40 in span or rank\n", off);
}
