\\ Second engine for kfold_coverage_area.mac: the boundary-arc algorithm of
\\ `crates/spacelab-constellation/src/kfold.rs` re-implemented at 60 digits, and checked against
\\ (a) the arc integral's closed form against numerical quadrature of the 1-form along arcs;
\\ (b) the lens of two caps by Gauss-Bonnet against a one-dimensional integral over colatitude;
\\ (c) a different method, the integral over colatitude of the longitude covered at least k times,
\\     on a twelve-cap set and on three circles through one point;
\\ (d) the regular polyhedra, three hemispheres by Girard's theorem, the invariant sum_k A_k;
\\ (e) as (c), on circles 1e-8 and 1e-12 apart that pass through one point, given as the doubles the
\\     Rust test reads.
\\ It prints the 60-digit areas of the twelve-cap set and of the triple point, which the Rust test
\\ `the_f64_areas_hold_the_60_digit_values` holds its f64 areas to, and those of (e), which
\\ `near_coincident_circles_through_one_point` does.
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/kfold_coverage_area.gp </dev/null
default(realprecision, 60);
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, Strprintf("0   (|residual| %.1e)", abs(e)), Str("FAIL ", e)));
dt(a, b) = a * b~;
crs(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
nrm(v) = v / sqrt(dt(v, v));
ang(a, b) = arg(dt(a, b) + I * sqrt(dt(crs(a, b), crs(a, b))));
capa(r) = 2 * Pi * (1 - cos(r));

\\ A circle: [c, u, v, rho] with (u, v, c) right-handed.
circ(c, r) = {
  my(e = if(abs(c[1]) <= abs(c[2]) && abs(c[1]) <= abs(c[3]), [1, 0, 0],
             if(abs(c[2]) <= abs(c[3]), [0, 1, 0], [0, 0, 1])), u = nrm(crs(e, c)));
  [c, u, crs(c, u), r];
}
cpsi(C, x) = arg(dt(x, C[2]) + I * dt(x, C[3]));

\\ Relation of circles i and j: [0, i inside j, j inside i] or [1, enter, exit, h_i, h_j]: enter and
\\ exit on circle i, the arcs inside the other cap 2 h_i long on circle i and 2 h_j on circle j.
rel(Ci, Cj) = {
  my(ci = Ci[1], cj = Cj[1], ri = Ci[4], rj = Cj[4], cr = crs(ci, cj), d, nn, cm, co, lst);
  d = ang(ci, cj); nn = d - abs(ri - rj); cm = ri + rj - d; co = 2 * Pi - ri - rj - d;
  lst = min(nn, min(cm, co));
  if(lst <= 0,
    if(lst == nn, return([0, ri <= rj, ri > rj]));
    if(lst == cm, return([0, 0, 0]), return([0, 1, 1])));
  my(sd = sin(cm / 2), sp = sin(co / 2), si = sin((d + rj - ri) / 2), sj = sin((d + ri - rj) / 2));
  my(h = 2 * arg(sqrt(sp * sj) + I * sqrt(sd * si)), w = nrm(crs(cr, ci)), t = crs(ci, w));
  my(foot = cos(ri) * ci + sin(ri) * cos(h) * w, side = sin(ri) * sin(h) * t);
  [1, foot - side, foot + side, h, 2 * arg(sqrt(sp * si) + I * sqrt(sd * sj))];
}

\\ Circle C runs inside the other cap on the arc (x1, x2), len radians long: [1 if it is inside at
\\ psi = -pi, the two events in the order a stable sort must keep]. Ends whose angles differ from len
\\ by more than pi rounded past each other: they are taken as one point, len picks the side.
ival(C, x1, x2, len) = {
  my(a = cpsi(C, x1), b = cpsi(C, x2), ms, w);
  ms = if(b < a, b - a + 2 * Pi, b - a);
  if(b == a || abs(ms - len) > Pi, b = a);
  w = if(b == a, len > Pi, a > b);
  [w, if(w, [[b, x2, -1], [a, x1, 1]], [[a, x1, 1], [b, x2, -1]])];
}

\\ The integral of omega along circle C from the point x1 (angle p1) to x2 (angle p2 > p1), pole n.
arcform(C, n) = {
  my(c = C[1], a = cos(C[4]), b = sin(C[4]), A = dt(n, c), B = sqrt(dt(n, C[2])^2 + dt(n, C[3])^2));
  my(p = 1 + A * a, q = b * B, P = p + q, e = abs(a + A), sg = if(a + A < 0, -1, 1));
  [crs(c, n), a, sg, A * a, 2 * P, (P + e)^2];
}
gee(F, n, x) = arg(F[6] + F[5] * (dt(n, x) - F[4]) + I * F[5] * dt(F[1], x));
arcint(F, dp, dg) = (F[3] - F[2]) * dp - 2 * F[3] * dg;

\\ The boundary-arc areas A_1..A_m of caps given as [[c, rho], ...] (c any non-zero vector).
kfold(caps) = {
  my(m = #caps, C = vector(m, i, circ(nrm(caps[i][1]), caps[i][2])), s, best = -1, n);
  my(cands = List([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]));
  forvec(sg = [[0,1],[0,1],[0,1]], listput(cands, nrm([(-1)^sg[1], (-1)^sg[2], (-1)^sg[3]])));
  for(i = 1, m, listput(cands, C[i][1]); listput(cands, -C[i][1]));
  foreach(cands, x, my(cl = vecmin(vector(m, i, abs(ang(x, C[i][1]) - C[i][4]))));
    if(cl > best, best = cl; s = x));
  n = -s;
  my(base = vector(m), ev = vector(m, i, List()), Ar = vector(m));
  for(i = 1, m, for(j = i + 1, m,
    my(r = rel(C[i], C[j]));
    if(r[1] == 0,
      base[i] += r[2]; base[j] += r[3],
      \\ circle i inside j from enter to exit; circle j inside i from exit to enter
      my(vi = ival(C[i], r[2], r[3], 2 * r[4]), vj = ival(C[j], r[3], r[2], 2 * r[5]));
      base[i] += vi[1]; foreach(vi[2], e, listput(ev[i], e));
      base[j] += vj[1]; foreach(vj[2], e, listput(ev[j], e)))));
  for(i = 1, m,
    my(F = arcform(C[i], n), E = vecsort(Vec(ev[i]), 1), dep = base[i], L = #E);
    if(L == 0, Ar[dep + 1] += arcint(F, 2 * Pi, 0); next);
    for(k = 1, L,
      dep += E[k][3];
      my(pn = if(k < L, E[k + 1][1], E[1][1] + 2 * Pi), xn = if(k < L, E[k + 1][2], E[1][2]));
      Ar[dep + 1] += arcint(F, pn - E[k][1], gee(F, n, xn) - gee(F, n, E[k][2]))));
  my(ds = sum(i = 1, m, dt(s, C[i][1]) > cos(C[i][4])));
  for(k = 1, ds, Ar[k] += 4 * Pi);
  Ar;
}

\\ ---- (c) the colatitude integral: L_k(theta) by a sweep in longitude, about the z axis ----
covlong(pol, th) = {
  my(m = #pol, full = 0, ev = List(), byc = vector(m + 1), at = 0, cnt, out = vector(m), acc = 0);
  for(i = 1, m,
    my(ti = pol[i][1], ph = pol[i][2], r = pol[i][3], w = (cos(r) - cos(th) * cos(ti)) / (sin(th) * sin(ti)));
    if(w <= -1, full++,
      if(w < 1,
        my(h = acos(w), st = (ph - h) % (2 * Pi), en = st + 2 * h);
        if(en <= 2 * Pi, listput(ev, [st, 1]); listput(ev, [en, -1]),
          listput(ev, [st, 1]); listput(ev, [2 * Pi, -1]); listput(ev, [0, 1]); listput(ev, [en - 2 * Pi, -1])))));
  my(E = vecsort(Vec(ev), 1));
  cnt = full;
  for(k = 1, #E, byc[cnt + 1] += E[k][1] - at; at = E[k][1]; cnt += E[k][2]);
  byc[cnt + 1] += 2 * Pi - at;
  forstep(k = m, 1, -1, acc += byc[k + 1]; out[k] = acc);
  out;
}
\\ crossings of two circles by the vector formula (not the half-angle route)
vxings(ci, ri, cj, rj) = {
  my(g = dt(ci, cj), ai = cos(ri), aj = cos(rj), k = 1 - g^2, D);
  D = k - ai^2 - aj^2 + 2 * ai * aj * g;
  if(k <= 0 || D <= 0, return([]));
  my(mid = ((ai - aj * g) / k) * ci + ((aj - ai * g) / k) * cj, off = (sqrt(D) / k) * crs(ci, cj));
  [mid + off, mid - off];
}
colat(caps) = {
  my(m = #caps, U = vector(m, i, nrm(caps[i][1])), pol, cuts = List([0, Pi]), tot = vector(m));
  pol = vector(m, i, [acos(U[i][3]), arg(U[i][1] + I * U[i][2]), caps[i][2]]);
  for(i = 1, m, my(ti = pol[i][1], r = pol[i][3]);
    foreach([abs(ti - r), min(ti + r, 2 * Pi - ti - r)], t, if(t > 0 && t < Pi, listput(cuts, t))));
  for(i = 1, m, for(j = i + 1, m,
    foreach(vxings(U[i], caps[i][2], U[j], caps[j][2]), x, listput(cuts, acos(x[3] / sqrt(dt(x, x)))))));
  my(K = vecsort(Vec(cuts), , 8));
  for(k = 1, #K - 1, my(a = K[k], b = K[k + 1]);
    tot += intnum(tau = 0, Pi,
      my(th = a + (b - a) * (1 - cos(tau)) / 2); covlong(pol, th) * sin(th) * (b - a) * sin(tau) / 2));
  tot;
}

\\ ---- (a) the arc integral against quadrature of omega(X') along random arcs ----
setrand(20261125);
rnd() = random(2.) - 1.;
rdir() = nrm([rnd(), rnd(), rnd()]);
{
  my(worst = 0);
  for(trial = 1, 12,
    my(C = circ(rdir(), 0.2 + 2.6 * random(1.)), n = rdir(), F, p1 = -3 + 2 * random(1.), p2);
    p2 = p1 + 4 * random(1.);
    F = arcform(C, n);
    my(X(t) = cos(C[4]) * C[1] + sin(C[4]) * (cos(t) * C[2] + sin(t) * C[3]));
    my(dX(t) = sin(C[4]) * (-sin(t) * C[2] + cos(t) * C[3]));
    my(quad = intnum(t = p1, p2, dt(n, crs(X(t), dX(t))) / (1 + dt(n, X(t)))));
    my(form = arcint(F, p2 - p1, gee(F, n, X(p2)) - gee(F, n, X(p1))));
    worst = max(worst, abs(quad - form)));
  chk("arc_integral_against_quadrature_of_omega", worst, 1e-50);
}

\\ ---- (b) the lens by Gauss-Bonnet against a one-dimensional integral ----
lens(d, r1, r2) = {
  2 * (Pi - acos((cos(d) - cos(r1) * cos(r2)) / (sin(r1) * sin(r2)))
    - cos(r1) * acos((cos(r2) - cos(d) * cos(r1)) / (sin(d) * sin(r1)))
    - cos(r2) * acos((cos(r1) - cos(d) * cos(r2)) / (sin(d) * sin(r2))));
}
\\ about c1: the circle of colatitude th meets cap 2 (centre d away) in 2 acos(w) of longitude
lens1d(d, r1, r2) = {
  my(lo = abs(d - r2), hi = min(r1, d + r2), f(th) = my(w = (cos(r2) - cos(th) * cos(d)) / (sin(th) * sin(d)));
    2 * acos(max(-1, min(1, w))) * sin(th));
  my(inner = if(r2 > d, intnum(th = 0, min(r2 - d, r1), 2 * Pi * sin(th)), 0));
  if(hi <= lo, return(inner));
  inner + intnum(tau = 0, Pi, my(th = lo + (hi - lo) * (1 - cos(tau)) / 2); f(th) * (hi - lo) * sin(tau) / 2);
}
{
  my(worst = 0);
  foreach([[0.9, 0.7, 0.5], [1.5, 1.2, 0.4], [2.5, 1.9, 1.6], [1.2, 2.0, 1.0], [1.2, 0.8, 0.8], [2.9, 1.7, 1.5]], t,
    my(d = t[1], r1 = t[2], r2 = t[3]);
    worst = max(worst, abs(lens(d, r1, r2) - lens1d(d, r1, r2)));
    \\ the algorithm's A_2 on the two caps
    worst = max(worst, abs(kfold([[[0, 0, 1], r1], [[sin(d), 0, cos(d)], r2]])[2] - lens(d, r1, r2))));
  chk("lens_gauss_bonnet_against_the_1d_integral_and_the_arcs", worst, 1e-40);
}

\\ ---- (d) the regular polyhedra, Girard, the invariant ----
inv(caps, A) = vecsum(A) - sum(i = 1, #caps, capa(caps[i][2]));
pcheck(name, V, r, want) = {
  my(caps = vector(#V, i, [V[i], r]), A = kfold(caps));
  chk(Str(name, "_areas"), vecmax(abs(A - want)), 1e-45);
  chk(Str(name, "_invariant"), inv(caps, A), 1e-45);
}
tet = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
oct = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
cub = [[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]];
g = (1 + sqrt(5)) / 2;
ico = [[0,1,g],[0,-1,g],[0,1,-g],[0,-1,-g],[1,g,0],[-1,g,0],[1,-g,0],[-1,-g,0],[g,0,1],[-g,0,1],[g,0,-1],[-g,0,-1]];
ri = acos(dt(ico[1] + ico[2] + ico[9], ico[1]) / sqrt(dt(ico[1] + ico[2] + ico[9], ico[1] + ico[2] + ico[9]) * dt(ico[1], ico[1])));
pcheck("tetrahedron_acos_1_3", tet, acos(1/3), [4*Pi, 4*Pi/3, 0, 0]);
pcheck("tetrahedron_pi_minus_acos_1_3", tet, Pi - acos(1/3), [4*Pi, 4*Pi, 8*Pi/3, 0]);
pcheck("octahedron_acos_1_sqrt3", oct, acos(1/sqrt(3)), [4*Pi, 8*Pi - 4*sqrt(3)*Pi, 0, 0, 0, 0]);
pcheck("octahedron_half_pi", oct, Pi/2, [4*Pi, 4*Pi, 4*Pi, 0, 0, 0]);
pcheck("cube_acos_1_sqrt3", cub, acos(1/sqrt(3)), concat([4*Pi, 12*Pi - 16*Pi/sqrt(3)], vector(6)));
pcheck("icosahedron_face_circumradius", ico, ri, concat([4*Pi, 12*capa(ri) - 4*Pi], vector(10)));
{
  my(worst = 0);
  for(trial = 1, 8,
    my(c = vector(3, k, rdir()), th = 0, A);
    for(i = 1, 3, for(j = i + 1, 3, th += ang(c[i], c[j])));
    A = kfold(vector(3, i, [c[i], Pi/2]));
    worst = max(worst, vecmax(abs(A - [2*Pi + th, 2*Pi, 2*Pi - th]))));
  chk("three_hemispheres_by_girard", worst, 1e-45);
}

\\ ---- (c) twelve caps, and three circles through one point: two methods, the values ----
{
  twelve = [[[3,1,2], 12/16], [[-1,4,1], 10/16], [[2,-3,1], 14/16], [[-2,-1,3], 9/16],
            [[1,2,-4], 13/16], [[-3,2,-1], 11/16], [[4,-1,-2], 15/16], [[0,-2,-3], 8/16],
            [[-4,-3,1], 12/16], [[2,4,3], 16/16], [[1,-4,4], 10/16], [[-1,1,-1], 32/16]];
}
\\ circles of half-angle rho through the pole (0, 0, 1): centres rho from it at azimuths 0, 2, 4
{
  triple = vector(3, i, my(r = [8/16, 11/16, 10/16][i], az = [0, 2, 4][i]);
    [[sin(r) * cos(az), sin(r) * sin(az), cos(r)], r]);
}
\\ the colatitude axis must not be a crossing: tilt the triple point off the z axis for the integral
tilt(caps) = {
  my(R = [1, 0, 0; 0, cos(3/10), -sin(3/10); 0, sin(3/10), cos(3/10)]);
  vector(#caps, i, [(R * caps[i][1]~)~, caps[i][2]]);
}
{
  foreach([["twelve_caps", twelve], ["triple_point", triple]], cs,
    my(caps = cs[2], A = kfold(caps), B = colat(tilt(caps)));
    chk(Str(cs[1], "_arcs_against_colatitude_integral"), vecmax(abs(A - B)), 1e-30);
    chk(Str(cs[1], "_invariant"), inv(caps, A), 1e-45);
    printf("%s A_k: %s\n", cs[1], strjoin(apply(x -> Strprintf("%.22e", x), A), ", ")));
}

\\ ---- (e) circles 1e-8 and 1e-12 apart through one point: two methods, the values ----
\\ The inputs of `near_coincident_circles_through_one_point`: the decimals the test writes, each
\\ beside the binary64 value Rust reads it as, M 2^k with 2^52 <= |M| < 2^53 (from Julia's frexp).
\\ The areas are computed on the binary64 values; the check shows each is the double nearest its
\\ decimal, |decimal - M 2^k| < 2^(k-1), half the spacing of the doubles there.
{
  b64rows = [
  [-8.760328444897426e-1, -7890602384016641, -53], [-2.458666227100355e-1, -8858278643358066, -55], [4.1486872527628704e-1, 7473610546247836, -54], [8.736044060868489e-1, 7868728955443912, -53],
  [-8.760328451510717e-1, -7890602389973364, -53], [-2.4586661932938225e-1, -8858278521557196, -55], [4.148687258833314e-1, 7473610557183375, -54], [8.736044091587652e-1, 7868728983113275, -53],
  [-8.760328449406597e-1, -7890602388078141, -53], [-2.4586662847112248e-1, -8858278850923099, -55], [4.1486872090990096e-1, 7473610467590017, -54], [8.73604398848117e-1, 7868728890243212, -53],
  [-8.760328443534356e-1, -7890602382788897, -53], [-2.4586663133212508e-1, -8858278954001581, -55], [4.148687204543413e-1, 7473610459383384, -54], [8.736043962833798e-1, 7868728867142113, -53],
  [-7.977735752844216e-1, -7185705552753899, -53], [-5.192310457563937e-1, -4676817488375375, -53], [3.0652315361263205e-1, 5521830241561116, -54], [9.655900539024349e-1, 8697262013895326, -53],
  [-7.977735752842509e-1, -7185705552752361, -53], [-5.192310457563492e-1, -4676817488374974, -53], [3.0652315361315197e-1, 5521830241570482, -54], [9.655900539025168e-1, 8697262013896064, -53],
  [-7.977735752838844e-1, -7185705552749060, -53], [-5.192310457572321e-1, -4676817488382927, -53], [3.065231536126101e-1, 5521830241560721, -54], [9.655900539014648e-1, 8697262013886588, -53]];
  my(worst = 0);
  for(i = 1, #b64rows, my(r = b64rows[i]);
    if(abs(r[2]) < 2^52 || abs(r[2]) >= 2^53, error("mantissa out of range"));
    worst = max(worst, abs(r[1] - r[2] * 2^r[3]) / 2^(r[3] - 1)));
  chk("inputs_are_the_doubles_nearest_the_decimals", worst, 1);
  b64 = vector(#b64rows, i, b64rows[i][2] * 2^b64rows[i][3]);
  apart = vector(4, c, [b64[4*c-3..4*c-1], b64[4*c]]);
  moved = vector(3, c, [b64[16+4*c-3..16+4*c-1], b64[16+4*c]]);
}
{
  foreach([["four_circles_1e-8_apart", apart], ["three_circles_1e-12_apart", moved]], cs,
    my(caps = cs[2], A = kfold(caps), B = colat(caps));
    chk(Str(cs[1], "_arcs_against_colatitude_integral"), vecmax(abs(A - B)), 1e-30);
    chk(Str(cs[1], "_invariant"), inv(caps, A), 1e-45);
    printf("%s A_k: %s\n", cs[1], strjoin(apply(x -> Strprintf("%.22e", x), A), ", ")));
}

\\ ---- (f) the near-coincident corpus ----
\\ 336 random sets, 24 for each delta = 1e-6 ... 1e-12 and three or four circles: centres within
\\ delta/2 per coordinate of one centre, each half-angle its centre's distance to one point, so the
\\ circles pass through it; every input then rounded to binary64. The areas are the same algorithm
\\ at 60 digits on those doubles. The file below holds them for the Rust test
\\ `near_coincident_corpus_within_its_bounds`; this section recomputes it and checks the file holds
\\ the same doubles and the same areas to 1e-28. KFOLD_WRITE_FIXTURE=1 writes the file instead.
\\ Every double is the nearest to its 60-digit value, which the rounding checks; any double would
\\ do, since both sides read the same M 2^k.
tob64(x) = {
  if(x == 0, return([0, 0]));
  my(e = exponent(x), M, k);
  if(!(2^e <= abs(x) && abs(x) < 2^(e + 1)), error("tob64: exponent"));
  k = e - 52; M = round(x / 2^k);
  if(abs(M) == 2^53, M /= 2; k++);
  if(abs(x - M * 2^k) >= 2^(k - 1), error("tob64: not the nearest double"));
  [M, k];
}
{
  my(lines = List(), sets = List(), file = "crates/spacelab-constellation/tests/fixtures/kfold_near_coincident.txt");
  setrand(31004);
  listput(lines, "# The near-coincident corpus of `kfold_areas` (plan 18 Ф31.D), written by");
  listput(lines, "# derivations/kfold_coverage_area.gp, section (f): our own numbers, PARI/GP at 60 digits.");
  listput(lines, "# set e m: m circles through one point, about 10^-e apart");
  listput(lines, "# cap Mx kx My ky Mz kz Mr kr: the centre (Mx 2^kx, My 2^ky, Mz 2^kz), the half-angle Mr 2^kr");
  listput(lines, "# areas A_1 ... A_m: the boundary-arc algorithm at 60 digits on those binary64 inputs");
  for(ex = 6, 12, my(delta = 10.^(-ex));
    foreach([3, 4], m,
      for(trial = 1, 24,
        my(p = rdir(), c0 = rdir(), q, caps, A);
        while(abs(dt(p, c0)) > 0.95, c0 = rdir());
        q = vector(m, i, my(c = nrm(c0 + (delta / 2) * [rnd(), rnd(), rnd()]));
          concat(apply(tob64, c), [tob64(ang(c, p))]));
        caps = vector(m, i, [vector(3, k, q[i][k][1] * 2^q[i][k][2]), q[i][4][1] * 2^q[i][4][2]]);
        A = kfold(caps);
        listput(sets, [q, A]);
        listput(lines, Strprintf("set %d %d", ex, m));
        for(i = 1, m, listput(lines, Str("cap ", strjoin(vector(8, k, Str(q[i][(k + 1) \ 2][2 - k % 2])), " "))));
        listput(lines, Str("areas ", strjoin(apply(x -> Strprintf("%.30f", x), A), " "))))));
  if(getenv("KFOLD_WRITE_FIXTURE"),
    my(f = Str("../", file), h = fileopen(f, "w"));
    foreach(lines, l, filewrite(h, l));
    fileclose(h);
    printf("wrote %s: %d sets\n", f, #sets),
    my(got = iferr(readstr(Str("../", file)), E, readstr(file)), worst = 0, same = 1, s = 0, r = 0);
    \\ walk the file: each `cap` line must name the same integers, each `areas` line the same values
    foreach(got, l, my(t = strsplit(l, " "));
      if(t[1] == "set", s++; r = 0);
      if(t[1] == "cap", r++;
        same = same && vector(8, k, eval(t[k + 1])) == vector(8, k, sets[s][1][r][(k + 1) \ 2][2 - k % 2]));
      if(t[1] == "areas",
        worst = max(worst, vecmax(abs(vector(#t - 1, k, eval(t[k + 1])) - sets[s][2])))));
    chk("near_coincident_corpus_inputs_are_the_fixtures", if(same && s == #sets, 0, 1), 1/2);
    chk("near_coincident_corpus_areas_are_the_fixtures", worst, 1e-28));
}
