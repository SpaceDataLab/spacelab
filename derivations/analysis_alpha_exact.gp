\\ Second engine for analysis_alpha_exact.mac: the candidate-set alpha-characteristic of
\\ `crates/spacelab-constellation/src/analysis.rs` re-implemented at 60 digits. Checked against the
\\ closed forms (regular polyhedra, one site, two sites at pi - 10^-m, Walker 2/1/0), against a grid
\\ on sets with antipodal and coincident sites, and on the Walker shift that is a rotation. It also
\\ evaluates alpha at two instants of the six Walker cases of the pass-12 table, the values the Rust
\\ test `the_f64_code_holds_the_60_digit_values_of_six_walker_instants` holds its enclosures to.
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/analysis_alpha_exact.gp </dev/null
default(realprecision, 60);
nrm(v) = v / sqrt(v*v~);
crs(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
\\ two points of the great circle orthogonal to r
equator(r) = my(e = if(abs(r[1]) < 1/2, [1, 0, 0], [0, 1, 0]), u1 = nrm(crs(r, e))); [u1, nrm(crs(r, u1))];
\\ antipodes; anti-midpoints of non-antipodal pairs, two equator points of an antipodal one; both
\\ poles of every triple of distinct sites
cands(S) = {
  my(n = #S, c = List());
  for(i = 1, n,
    listput(c, -S[i]);
    for(j = i + 1, n,
      my(m = S[i] + S[j]);
      if(sqrt(m*m~) < 1e-40,
        my(q = equator(S[i])); listput(c, q[1]); listput(c, q[2]),
        listput(c, nrm(-m)));
      for(l = j + 1, n,
        my(p = crs(S[j] - S[i], S[l] - S[i]));
        if(sqrt(p*p~) > 1e-40, listput(c, nrm(p)); listput(c, nrm(-p))))));
  Vec(c);
}
kth(x, S, k) = vecsort(vector(#S, i, x*S[i]~), , 4)[k];
alpha(S, k) = my(c = cands(S)); acos(max(-1, min(1, vecmin(vector(#c, j, kth(c[j], S, k))))));
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, 0, Str("FAIL ", e)));

tet = apply(v -> v/sqrt(3), [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]);
oct = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
a3 = acos(1/3); ar3 = acos(1/sqrt(3));
twant = [a3, Pi - a3, Pi - ar3, Pi];
owant = [ar3, Pi/2, Pi/2, Pi - ar3, 3*Pi/4, Pi];
for(k = 1, 4, chk(Str("tetrahedron_k", k), alpha(tet, k) - twant[k], 1e-50));
for(k = 1, 6, chk(Str("octahedron_k", k), alpha(oct, k) - owant[k], 1e-50));

chk("one_site_is_pi", alpha([[1, 0, 0]], 1) - Pi, 1e-50);
chk("two_antipodal_sites_are_half_pi", alpha([[1, 0, 0], [-1, 0, 0]], 1) - Pi/2, 1e-50);
chk("two_sites_k2_is_pi", alpha([[1, 0, 0], [0, 1, 0]], 2) - Pi, 1e-50);
\\ two sites d = pi - 10^-m apart: pi - d/2
{
  for(m = 3, 13,
    my(d = Pi - 10^-m, S = [[1, 0, 0], [cos(d), sin(d), 0]]);
    chk(Str("two_sites_at_pi_minus_1e-", m), alpha(S, 1) - (Pi - d/2), 1e-45));
}

\\ Walker Delta T/P/F at inclination inc: plane p at node 2 pi p/P, slot s at u = 2 pi s/S + 2 pi F p/T + u0
pos(inc, node, u) = [cos(node)*cos(u) - sin(node)*sin(u)*cos(inc), sin(node)*cos(u) + cos(node)*sin(u)*cos(inc), sin(u)*sin(inc)];
walker(T, P, F, inc, u0) = my(S = T/P); concat(vector(P, p, vector(S, s, pos(inc, 2*Pi*(p - 1)/P, 2*Pi*(s - 1)/S + 2*Pi*F*(p - 1)/T + u0))));
for(j = 0, 3, chk(Str("walker_2_1_0_instant_", j), alpha(walker(2, 1, 0, 55*Pi/180, 0.9*j), 1) - Pi/2, 1e-50));

\\ The shift by 2 pi F/T turned by 2 pi/P is the pattern: each turned satellite has a partner.
rz(a, v) = [cos(a)*v[1] - sin(a)*v[2], sin(a)*v[1] + cos(a)*v[2], v[3]];
shiftgap(T, P, F, inc, u0) = {
  my(A = walker(T, P, F, inc, u0), B = apply(v -> rz(2*Pi/P, v), walker(T, P, F, inc, u0 + 2*Pi*F/T)));
  vecmax(vector(#B, i, vecmin(vector(#A, j, my(d = B[i] - A[j]); sqrt(d*d~)))));
}
{
  chk("walker_shift_is_a_rotation", shiftgap(24, 3, 1, 56*Pi/180, 0.3) + shiftgap(18, 6, 2, 55*Pi/180, 1.1)
    + shiftgap(7, 7, 5, 5569/100*Pi/180, 0.2) + shiftgap(12, 4, 3, 70*Pi/180, 2.0), 1e-50);
}

\\ No point of a grid is farther than the best candidate: sets with antipodal and coincident sites.
fib(i, N) = my(z = 1 - (2*i + 1)/N, r = sqrt(1 - z^2), f = Pi*(3 - sqrt(5))*i); [r*cos(f), r*sin(f), z];
gridmax(S, k, N) = vecmax(vector(N, i, acos(max(-1, min(1, kth(fib(i - 1, N), S, k))))));
setrand(20261124);
rnd() = nrm(vector(3, i, random(2.) - 1));
excess = 0;
{
  for(trial = 1, 6,
    my(a = rnd(), b = rnd(), c = rnd(), S = [a, -a, b, b, c, rnd()][1 .. 3 + trial % 4]);
    for(k = 1, #S - 1, excess = max(excess, gridmax(S, k, 400) - alpha(S, k))));
}
chk("grid_is_never_above_the_candidates", max(excess, 0), 1e-50);

\\ alpha at u0 = 0 and u0 = 3/10 of the six Walker cases, to 22 digits, as the Rust test holds them.
{
  six = [[5, 5, 1, 4366/100, 1], [7, 7, 5, 5569/100, 2], [6, 6, 4, 5313/100, 1],
         [24, 3, 1, 55, 1], [24, 6, 1, 55, 4], [18, 6, 2, 55, 1]];
  held = [[1.180161571420008205745, 1.206842792871513127863], [1.360449745023500894703, 1.400138747559720321611],
          [0.9740538582535637711503, 1.055324877647860854325], [0.6605835056076490213617, 0.6596226073121076325114],
          [1.004577893062186243963, 0.9935669631037288383943], [0.6498683041587793444892, 0.6553892789088968441234]];
}
{
  for(c = 1, 6,
    my(w = six[c]);
    for(j = 1, 2,
      my(v = alpha(walker(w[1], w[2], w[3], w[4]*Pi/180, [0, 3/10][j]), w[5]));
      chk(Str("walker_", w[1], "_", w[2], "_", w[3], "_at_", [0, 3/10][j]), v - held[c][j], 1e-21)));
}
