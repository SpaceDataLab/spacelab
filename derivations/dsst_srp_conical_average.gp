/* The reference values `spacelab-propagation`'s DSST solar radiation pressure is gated on, at 80
   digits: the mean rates (1/2 pi) Int nu F d(lambda) and the short-period Fourier coefficients
   (1/pi) Int nu F (cos jL, sin jL) dL over the whole orbit, F Gauss's rates of the cannonball law
   C_R (S/m) P_ref d_ref^2 (r - s)/|r - s|^3 and nu the fraction of the Sun's disc in view.

   Nothing here is the shipped algorithm.
   * The contacts are the zeros of d - (r_E + r_S) and d - (r_E - r_S), the separation of the two
     discs seen from the satellite against the sum and the difference of their angular radii, by a
     scan and Brent's method (`solve`) on the angles themselves: no cone and no quartic.
   * nu is the textbook lens, the acos form of the circle-intersection area.
   * F is d(element)/dv . acceleration, the partials the exact derivatives of a Cartesian ->
     equinoctial conversion, taken as the x-coefficient of power series.
   * The integrals are double-exponential quadrature (`intnum`) on each piece between contacts,
     once in the true longitude L, with d(lambda) = (r/a)^2 dL / B, and once in the mean longitude
     lambda, through Kepler's equation. A piece between two penumbra contacts is cut where the umbra
     edge comes nearest, so that its near-singularity sits at an end, where the quadrature is fast.
   The checks: (1) every piece is what nu says it is; (2) at every contact the shipped cone
   function S = 1 - cos^2 Phi - (m w - tau cos Phi)^2 vanishes, to 1e-70; (3) the two routes agree
   to 1e-40 of the rates' scale, which bounds the error of both; (4) the LEO fixture at 50 digits
   agrees with 80 to 1e-40.

   Six fixtures, the constants of the Rust tests: the Orekit LEO with its Sun of 2024-01-01; an
   eccentric HEO with the Sun in the orbit plane; a GEO with the Sun 5 degrees out of it; a low LEO
   in the penumbra alone, half the Sun hidden at its deepest, and the same orbit 1e-6 short of the
   umbra cone; the Orekit LEO with the Sun on the orbit normal, never in shadow. Each Sun is rounded
   to doubles here, and every input is the exact rational of its double.

   Each rate row is (name, [a, k, h, q, p], sun, rates, magnitudes): the magnitudes are
   (1/2 pi) Int |nu F_i| d(lambda), the scale of the rounding of the shipped sum; they go to
   crates/spacelab-propagation/tests/dsst_srp_conical_reference.rs. The Fourier coefficients are
   internal to the crate, so their rows go to the tests of src/dsst/srp.rs, eight spaces in: each
   Fourier orbit (name, [a, k, h, q, p], sun), each Fourier row (name, j, C_j, S_j), each Fourier
   scale row (name, (1/pi) Int |nu F_i| dL). Every row is printed in Rust syntax and must appear
   verbatim in one of the two files; `derivations/verify.sh` checks that.

       gp -q -f derivations/dsst_srp_conical_average.gp </dev/null                              */

default(realprecision, 80);

/* The nearest binary64 to x, normal range, as an exact rational. */
dbl(x) = {
  my(r = x * 1., e);
  if (r == 0, return(0));
  e = exponent(r);
  round(x * 2^(52 - e)) * 2^(e - 52);
}
/* The shortest decimal in Rust syntax that parses back to the same double: at most 17 digits. */
lit(y) = {
  my(s, e10, mant);
  if (y == 0, return("0.0"));
  e10 = floor(log(abs(y * 1.)) / log(10));
  mant = y * 1. / 10^e10;
  if (abs(mant) >= 10, mant /= 10; e10++);
  if (abs(mant) < 1, mant *= 10; e10--);
  for (k = 0, 16,
    s = Strprintf(Str("%.", k, "fe%d"), mant, e10);
    if (dbl(eval(s)) == y, return(s)));
  error("no round trip for ", y);
}
sci(x) = {
  my(e10, mant);
  if (x == 0, return("0.0"));
  e10 = floor(log(abs(x * 1.)) / log(10));
  mant = x * 1. / 10^e10;
  if (abs(mant) >= 10, mant /= 10; e10++);
  Strprintf("%.3fe%d", mant, e10);
}
vlit(v) = { my(s = "["); for (i = 1, #v, s = Str(s, if (i > 1, ", ", ""), lit(dbl(v[i])))); Str(s, "]"); }
vsci(v) = { my(s = "["); for (i = 1, #v, s = Str(s, if (i > 1, ", ", ""), sci(v[i]))); Str(s, "]"); }
must(ok, what) = if (ok, print(what), error("FAILED: ", what));

cross3(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
dot3(a, b) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
norm3(a) = sqrt(dot3(a, a));
frame(q, p) = {
  my(c = 1 + p^2 + q^2);
  [[(1 - p^2 + q^2)/c, 2*p*q/c, -2*p/c], [2*p*q/c, (1 + p^2 - q^2)/c, 2*q/c]];
}
cst(z) = if (type(z) == "t_SER", polcoef(z, 0, x), z);
\\ atan2 that also takes power series: the branch from the constant terms.
at2(y, x0) = {
  my(yc = cst(y), xc = cst(x0));
  if (abs(xc) >= abs(yc),
    atan(y/x0) + if (xc < 0, if (yc >= 0, Pi, -Pi), 0),
    -atan(x0/y) + if (yc >= 0, Pi/2, -Pi/2));
}

\\ The constants of the Rust tests, each the double nearest its literal.
MU = 398600441500000;
CR = 3/2;
AOM = dbl(1/100);
AE = dbl(637813646/100);
PREF = dbl(456/10^8);
DREF = 149597870000;
RSUN = 695700000;
AU = 149597870700;
KREF = CR * AOM * PREF * DREF^2;

\\ The orbit and the Sun, set by setorbit([a, k, h, q, p], sun).
setorbit(el, sun) = {
  A = el[1]; K = el[2]; H = el[3]; Q = el[4]; P = el[5];
  my(fg = frame(Q, P));
  F = fg[1]; G = fg[2];
  BB = sqrt(1 - K^2 - H^2);
  PSL = A * BB^2;
  SUN = sun;
}
\\ A Sun at 1 au, gamma the sine of its angle out of the plane of el and phi its angle from f in
\\ the plane, rounded to doubles.
triad(el, gamma, phi) = {
  my(fg = frame(el[4], el[5]), w = cross3(fg[1], fg[2]), pl = sqrt(1 - gamma^2));
  vector(3, j, dbl(AU * (pl*cos(phi)*fg[1][j] + pl*sin(phi)*fg[2][j] + gamma*w[j])));
}

\\ Position and velocity at true longitude L.
state(L) = {
  my(c = cos(L), s = sin(L), w = 1 + K*c + H*s, rr = PSL / w);
  [rr*(c*F + s*G), sqrt(MU/PSL) * (-(H + s)*F + (K + c)*G)];
}
\\ Position and velocity at mean longitude lam, through the eccentric longitude.
state_mean(lam) = {
  my(ff = lam, b = 1/(1 + BB), sf, cf, xx, yy, rr, vs, xd, yd);
  for (i = 1, 200, ff -= (ff - K*sin(ff) + H*cos(ff) - lam)/(1 - K*cos(ff) - H*sin(ff)));
  sf = sin(ff); cf = cos(ff);
  xx = A*((1 - H^2*b)*cf + H*K*b*sf - K);
  yy = A*((1 - K^2*b)*sf + H*K*b*cf - H);
  rr = A*(1 - K*cf - H*sf);
  vs = A^2*sqrt(MU/A^3)/rr;
  xd = vs*(H*K*b*cf - (1 - H^2*b)*sf);
  yd = vs*((1 - K^2*b)*cf - H*K*b*sf);
  [xx*F + yy*G, xd*F + yd*G];
}
\\ True longitude to mean longitude: continuous in L.
mean_lon(lv) = {
  my(sl = sin(lv), cl = cos(lv), le);
  le = lv + 2*atan((H*cl - K*sl)/(1 + BB + K*cl + H*sl));
  le - K*sin(le) + H*cos(le);
}

\\ [a, k, h, q, p, lambda] of a position and velocity; takes power series.
elements(r, v) = {
  my(hv = cross3(r, v), hn = sqrt(dot3(hv, hv)), w = hv/hn, q = -w[2]/(1 + w[3]),
     p = w[1]/(1 + w[3]), rn = sqrt(dot3(r, r)), a = 1/(2/rn - dot3(v, v)/MU),
     ev = cross3(v, hv)/MU - r/rn, fg = frame(q, p), k = dot3(ev, fg[1]), h = dot3(ev, fg[2]),
     lv = at2(dot3(r, fg[2]), dot3(r, fg[1])), sl = sin(lv), cl = cos(lv), le);
  le = lv + 2*atan((h*cl - k*sl)/(1 + sqrt(1 - k^2 - h^2) + k*cl + h*sl));
  [a, k, h, q, p, le - k*sin(le) + h*cos(le)];
}
\\ Gauss's rates dE/dv . acc, the velocity partials as the x-coefficient of a series.
gauss(r, v, acc) = {
  my(out = vector(6));
  for (j = 1, 3,
    my(vs = v, e);
    vs[j] = v[j] + x + O(x^2);
    e = elements(r, vs);
    for (i = 1, 6, out[i] += polcoef(e[i], 1, x) * acc[j]));
  out;
}

\\ The visible fraction of the Sun's disc: the textbook lens, clamped acos arguments.
clamp(t) = max(-1, min(1, t));
geometry(r) = {
  my(ts = SUN - r, ds = norm3(ts), rn = norm3(r));
  [acos(clamp(-dot3(ts, r)/(ds*rn))), asin(AE/rn), asin(RSUN/ds)];
}
nu(r) = {
  my(g = geometry(r), d = g[1], re = g[2], rs = g[3], a);
  if (re <= rs, error("the occulter must look larger than the Sun"));
  if (d >= re + rs, return(1));
  if (d <= re - rs, return(0));
  a = rs^2*acos(clamp((d^2 + rs^2 - re^2)/(2*d*rs))) + re^2*acos(clamp((d^2 + re^2 - rs^2)/(2*d*re)))
      - sqrt((re + rs - d)*(d - re + rs)*(d + re - rs)*(d + re + rs))/2;
  1 - a/(Pi*rs^2);
}
cannonball(r, n) = my(d = r - SUN); KREF * n * d / norm3(d)^3;

\\ The contacts in true longitude: zeros of d - (re + rs) and of d - (re - rs), by a scan and
\\ Brent's method. Each is [L, side], side 1 for the penumbra cone and -1 for the umbra cone.
gside(L, side) = my(g = geometry(state(L)[1])); g[1] - g[2] - side*g[3];
contacts() = {
  my(out = List(), n = 4096);
  foreach ([1, -1], side,
    my(prev = gside(0, side));
    for (i = 1, n,
      my(l1 = 2*Pi*i/n, cur = gside(l1, side));
      if (sign(prev) != sign(cur) && prev != 0,
        listput(out, [solve(t = l1 - 2*Pi/n, l1, gside(t, side)), side]));
      prev = cur));
  vecsort(Vec(out), 1);
}
\\ The least of the umbra edge on [a, b], by golden-section search: to the square root of the
\\ precision, which is all a cut needs.
deepest(a, b) = {
  my(g = (sqrt(5) - 1)/2, lo = a, hi = b, x1 = hi - g*(hi - lo), x2 = lo + g*(hi - lo),
     f1 = gside(x1, -1), f2 = gside(x2, -1));
  for (i = 1, 200,
    if (f1 < f2, hi = x2; x2 = x1; f2 = f1; x1 = hi - g*(hi - lo); f1 = gside(x1, -1),
                 lo = x1; x1 = x2; f1 = f2; x2 = lo + g*(hi - lo); f2 = gside(x2, -1)));
  (lo + hi)/2;
}
\\ Pieces between consecutive contacts, kind 1 lit, 2 penumbra; umbra pieces are dropped. A
\\ penumbra piece between two penumbra contacts is cut at its deepest point.
pieces(cs) = {
  my(out = List(), m = #cs);
  if (m == 0, return([[-Pi, Pi, 1]]));
  for (i = 1, m,
    my(a = cs[i], b = if (i < m, cs[i + 1], [cs[1][1] + 2*Pi, cs[1][2]]),
       n = nu(state((a[1] + b[1])/2)[1]));
    if (n == 1, listput(out, [a[1], b[1], 1]),
      if (n != 0,
        if (a[2] == 1 && b[2] == 1,
          my(c = deepest(a[1], b[1]));
          listput(out, [a[1], c, 2]); listput(out, [c, b[1], 2]),
          listput(out, [a[1], b[1], 2])))));
  Vec(out);
}

JMAX = 12;
\\ Over L: the six mean-rate integrands (r/a)^2 F / B, their absolute values, then F cos jL and
\\ F sin jL, j = 1..JMAX, and |F|.
integrand_l(L, fourier) = {
  my(st = state(L), r = st[1], n = nu(r), out = vector(12 + if (fourier, 12*JMAX + 6, 0)), fr, roa);
  if (n == 0, return(out));
  fr = gauss(r, st[2], cannonball(r, n));
  roa = PSL/(1 + K*cos(L) + H*sin(L))/A;
  for (i = 1, 6, out[i] = roa^2*fr[i]/BB; out[6 + i] = abs(out[i]));
  if (fourier,
    for (j = 1, JMAX,
      my(c = cos(j*L), s = sin(j*L));
      for (i = 1, 6,
        out[12 + 12*(j - 1) + i] = fr[i]*c;
        out[12 + 12*(j - 1) + 6 + i] = fr[i]*s));
    for (i = 1, 6, out[12 + 12*JMAX + i] = abs(fr[i])));
  out;
}
integrand_m(lam) = {
  my(st = state_mean(lam), r = st[1], n = nu(r));
  if (n == 0, return(vector(6)));
  gauss(r, st[2], cannonball(r, n));
}

\\ The reference of the orbit set by setorbit: [contacts, pieces, rates by L, rates by lambda,
\\ magnitudes, Fourier C and S as a 12-vector per j, Fourier magnitudes].
reference(fourier) = {
  my(cs = contacts(), ps = pieces(cs), tl = vector(12 + if (fourier, 12*JMAX + 6, 0)), tm = vector(6));
  for (i = 1, #ps,
    my(a = ps[i][1], b = ps[i][2]);
    tl += intnum(t = a, b, integrand_l(t, fourier));
    \\ Kepler's equation puts branch points 0.15 off the real lambda axis near an e = 0.73 perigee,
    \\ against 0.86 in L: eight times the nodes there.
    my(la = mean_lon(a), lb = mean_lon(b));
    tm += intnum(t = la, lb, integrand_m(t), intnuminit(la, lb, 3)));
  [cs, ps, tl[1..6]/(2*Pi), tm/(2*Pi), tl[7..12]/(2*Pi),
   if (fourier, vector(JMAX, j, tl[12 + 12*(j - 1) + 1 .. 12 + 12*j]/Pi), []),
   if (fourier, tl[12 + 12*JMAX + 1 .. 12 + 12*JMAX + 6]/Pi, [])];
}

\\ The shipped cone function of side `side` at L, from its own algebra: m = ae/(p cos theta),
\\ tau = tan theta, sin theta = (ae + side R_sun)/|s|.
cone(L, side) = {
  my(sh = SUN/norm3(SUN), al = dot3(sh, F), be = dot3(sh, G), sn = (AE + side*RSUN)/norm3(SUN),
     cs = sqrt(1 - sn^2), m = AE/(PSL*cs), tau = sn/cs, c = cos(L), s = sin(L),
     cphi = al*c + be*s, rad = m*(1 + K*c + H*s) - tau*cphi);
  1 - cphi^2 - rad^2;
}

\\ ---- the fixtures ------------------------------------------------------------------------------
{
LEO = [7200000, dbl(30901699437494747/10^19), dbl(9510565162951536/10^18),
       dbl(854890329218965/10^15), dbl(769746709859868/10^15)];
LOW = [7078137, dbl(1/1000), dbl(-5/10000), dbl(2/10), dbl(-1/10)];
}
{
FIXTURES = [
  ["leo", LEO, [dbl(24813057806275562/10^6), dbl(-133033126695834660/10^6), dbl(-57667965236002365/10^6)], 1],
  ["heo", [24396000, dbl(7/10), dbl(21/100), dbl(61/1000), 0], 0, 0],
  ["geo", [42164000, dbl(1/1000), dbl(-5/10000), 0, 0], 0, 0],
  ["penumbra", LOW, 0, 0],
  ["graze", LOW, 0, 1],
  ["sunlit", LEO, 0, 0]];
FIXTURES[2][3] = triad(FIXTURES[2][2], 0, 3/10);
FIXTURES[3][3] = triad(FIXTURES[3][2], dbl(872/10000), 1);
FIXTURES[4][3] = triad(LOW, dbl(9011/10000), 2);
\\ 1e-6 short of the umbra cone: its tangency, gamma = 0.8998906042281511, from the shipped
\\ contacts; the check below confirms the penumbra alone.
FIXTURES[5][3] = triad(LOW, dbl(8998906042281511/10^16 * (1 + 1/10^6)), 2);
FIXTURES[6][3] = triad(LEO, 1, 0);
}

ROWS = List(); ORBITS = List(); FOURIER = List(); SCALES = List();
bad_kind = 0; checked = 0; worst_cone = 0; worst_route = 0; contacts_seen = 0; graze_dip = 0;
{
for (f = 1, #FIXTURES,
  my(name = FIXTURES[f][1], el = FIXTURES[f][2], sun = FIXTURES[f][3], fourier = FIXTURES[f][4], rf,
     scale);
  setorbit(el, sun);
  rf = reference(fourier);
  \\ (1) every piece is what nu says it is, and the gaps between them are umbra.
  for (i = 1, #rf[2],
    my(a = rf[2][i][1], b = rf[2][i][2], kind = rf[2][i][3]);
    for (j = 1, 3,
      my(n = nu(state(a + (b - a)*j/4)[1]));
      checked++;
      if (if (kind == 1, n != 1, n <= 0 || n >= 1), bad_kind++)));
  \\ (2) the shipped cone function vanishes at every contact.
  foreach (rf[1], c, contacts_seen++; worst_cone = max(worst_cone, abs(cone(c[1], c[2]))));
  \\ (3) the two routes.
  scale = vecmax(abs(rf[5]));
  worst_route = max(worst_route, vecmax(abs(rf[3] - rf[4])) / scale);
  print("  ", name, " routes: ", apply(z -> z * 1., (rf[3] - rf[4]) / scale));
  print1(name, ": ", #rf[1], " contacts ", apply(c -> c[2], rf[1]), ", pieces ",
         apply(p -> p[3], rf[2]), "\n");
  if (name == "graze",
    for (i = 1, #rf[2] - 1, if (rf[2][i][3] == 2 && rf[2][i + 1][3] == 2, graze_dip = rf[2][i][2]));
    if (graze_dip != 0,
      print("  graze: least nu ", nu(state(graze_dip)[1]) * 1., ", umbra edge ",
            gside(graze_dip, -1) * 1.)));
  listput(ROWS, Strprintf("    (\"%s\", %s, %s, %s, %s),", name, vlit(el), vlit(sun), vlit(rf[3]), vsci(rf[5])));
  if (fourier,
    for (j = 1, JMAX,
      listput(FOURIER, Strprintf("        (\"%s\", %d, %s, %s),", name, j, vlit(rf[6][j][1..6]), vlit(rf[6][j][7..12]))));
    listput(ORBITS, Strprintf("        (\"%s\", %s, %s),", name, vlit(el), vlit(sun)));
    listput(SCALES, Strprintf("        (\"%s\", %s),", name, vsci(rf[7]))));
  if (name == "leo", LEO_REF = rf[3]));
}
{
must(#ROWS == #FIXTURES && checked > 0 && contacts_seen > 0 && graze_dip != 0,
     "(0) holds: every fixture computed, the graze cut at its deepest point");
}
must(bad_kind == 0, "(1) holds: every piece is lit, or in the penumbra, at three inner points, as nu says");
print("  worst shipped cone function at a contact: ", worst_cone * 1.);
must(worst_cone < 1e-70, "(2) holds: the shipped cone function vanishes at every contact of both cones");
print("  worst route disagreement: ", worst_route * 1.);
must(worst_route < 1e-40, "(3) holds: the rates in the true and in the mean longitude agree");
{
  default(realprecision, 50);
  setorbit(FIXTURES[1][2], FIXTURES[1][3]);
  my(low = reference(0)[3]);
  default(realprecision, 80);
  my(d = vecmax(abs(low - LEO_REF)) / vecmax(abs(LEO_REF)));
  print("  LEO at 50 digits against 80: ", d * 1.);
  must(d < 1e-40, "(4) holds: the LEO rates at 50 digits agree with 80");
}
print("// rates");
foreach (ROWS, r, print(r));
print("// fourier orbits");
foreach (ORBITS, r, print(r));
print("// fourier");
foreach (FOURIER, r, print(r));
print("// fourier scale");
foreach (SCALES, r, print(r));
print("DONE");
quit
