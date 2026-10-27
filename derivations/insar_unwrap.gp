/* Second engine for derivations/insar_unwrap.mac, at 100 digits.

   For N = 1 .. 16 and every k < N:
     1. the DCT-II vector v_n = cos(pi k (2n+1)/(2N)) under the mirrored (Neumann) second
        difference is (2 cos(pi k/N) - 2) v, at every n including both ends;
     2. the coded inverse (1/N)[F_0 + 2 sum_(k>=1) F_k cos(...)] times the coded forward is the
        identity.
   And the vortex residue: the four wrapped steps of atan2(r - 1/2, c - 1/2), which is
   arg((c - 1/2) + i (r - 1/2)), around the loop `residues` walks sum to +2 pi.

       timeout 300 gp -q -f derivations/insar_unwrap.gp </dev/null                                 */

default(realprecision, 100);
tol = 10^-90;

{
my(eig = 0, inv = 0, cnt = 0);
for (N = 1, 16,
  for (k = 0, N - 1,
    my(v = vector(N, n, cos(Pi * k * (2 * n - 1) / (2 * N))), lam = 2 * cos(Pi * k / N) - 2);
    for (n = 1, N,
      my(lo = if (n == 1, v[1], v[n - 1]), hi = if (n == N, v[N], v[n + 1]));
      cnt++;
      if (abs(lo - 2 * v[n] + hi - lam * v[n]) > tol, eig++)));
  my(F = matrix(N, N, kk, j, cos(Pi * (kk - 1) * (2 * j - 1) / (2 * N))),
     G = matrix(N, N, j, kk, if (kk == 1, 1, 2) * cos(Pi * (kk - 1) * (2 * j - 1) / (2 * N)) / N),
     D = G * F - matid(N));
  if (normlp(concat(Vec(D), [])) > tol, inv++));
printf("neumann eigenvectors, N = 1 .. 16: %d of %d samples off by more than 1e-90\n", eig, cnt);
printf("dct inverse, N = 1 .. 16: %d of 16 lengths off by more than 1e-90\n", inv);
}

{
my(wrap(x) = x - 2 * Pi * floor((x + Pi) / (2 * Pi)), vx(p) = arg((p[2] - 1/2) + I * (p[1] - 1/2)),
   loop = [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]], s = 0);
for (i = 1, 4, s += wrap(vx(loop[i + 1]) - vx(loop[i])));
printf("vortex loop sum / (2 pi): %.3f\n", s / (2 * Pi));
}
