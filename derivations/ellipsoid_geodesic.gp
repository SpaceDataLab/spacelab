/* Geodesics on WGS 84 at 50 digits, by quadrature and root finding -- the second engine for
   derivations/ellipsoid_geodesic.mac and the reference values crates/spacelab-earth/tests/geodesic.rs
   asserts the shipped f64 code against.

   1. The series coefficients, derived again by a different route than the Maxima file: the integrand
      of I1 times (1 - eps) is |1 - eps z| = sqrt((1 - eps z)(1 - eps/z)), z = exp(2 i sigma), whose
      binomial series give the Fourier coefficients directly; I2 and I3 the same way; the reversion of
      I1 by Lagrange's formula sigma = tau + sum_k (-1)^k/k! (d/dtau)^(k-1) [g(tau)^k] where
      tau = sigma + g(sigma). Printed as `COEF` lines; derivations/verify.sh requires them to equal
      the Maxima file's.

   2. Reference geodesics. NOT the series: the distance is b times the quadrature of
      sqrt(1 + k^2 sin^2 sigma), and the longitude the quadrature of
          d lambda/d sigma = (1 - f) sin alpha0 sqrt(1 + k^2 sin^2 sigma)/(1 - cos^2 alpha0 sin^2 sigma)
      (Clairaut and the metric; the Maxima file derives it, section 1), split at its peaks. The
      direct problem solves the distance equation for sigma2 by bracketed root finding; the inverse
      problem puts the points in Karney's canonical order (phi1 <= 0, |phi2| <= |phi1|,
      0 <= lambda12 <= pi), finds the azimuth whose first northward crossing of latitude phi2 has the
      given lambda12 by bracketed root finding over [0, pi], and undoes the order. Meridians
      (lambda12 = 0) and the equator (phi1 = phi2 = 0, lambda12 <= (1 - f) pi) are closed forms;
      coincident points are 0 apart with both azimuths 0. The reduced length m12 is Karney's eq. (38)
      with J = I1 - I2 by quadrature. Every inverse solution is then run through this file's direct
      solution and must land on its second point. Karney's Tables 2, 3 and 4-6 are reproduced to
      the digits printed. Each check prints `ok <name>` or `FAIL <name>`.

   Inputs are doubles: every latitude, longitude, azimuth and distance below is rounded to the
   nearest double before use, so the rows hold the exact inputs the f64 code sees. Rows are Rust
   array literals: inverse (lat1, lon1, lat2, lon2, s12, azi1, azi2, m12), direct (lat1, lon1, azi1,
   s12, lat2, lon2, azi2); radians and metres; results to 25 significant digits.

       timeout 300 gp -q -f derivations/ellipsoid_geodesic.gp </dev/null                          */

default(realprecision, 50);

/* ------------------------------------------------------------------------------------------------
   1. Coefficients. x is the series variable (PARI's highest priority).
   ------------------------------------------------------------------------------------------------ */
NORD = 6; N3 = 5;
lcoef(r, l, w) = { my(M = 2*NORD + 2); polcoef(simplify(r*w^M), l + M, w); }
fourier(S, l, nmax) = Ser(vector(nmax + 1, j, lcoef(polcoef(S, j - 1, x), l, z)), x) + O(x^(nmax + 1));
coefline(name, l, i, j, r) = if (r != 0, printf("COEF %s %d %d %d %s\n", name, l, i, j, Str(r)));
evenline(name, l, S) = for (j = 0, NORD, coefline(name, l, 0, j, polcoef(S, j, x)));

R1 = sqrt((1 - x*z)*(1 - x/z) + O(x^(NORD + 1)));
c0 = fourier(R1, 0, NORD);
C1s = vector(NORD, l, fourier(R1, l, NORD)/(l*c0));
d0 = fourier(1/R1, 0, NORD);
C2s = vector(NORD, l, fourier(1/R1, l, NORD)/(l*d0));
evenline("A1", 0, c0);
evenline("A2", 0, d0);
for (l = 1, NORD, evenline("C1", l, C1s[l]));
/* Lagrange reversion: u = exp(2 i tau), d/dtau = 2 i u d/du, sin 2 l tau = (u^l - u^-l)/(2 i). */
{
  my(g = sum(l = 1, NORD, C1s[l]*(u^l - u^-l)/(2*I)), rev = 0, gk = 1, h);
  for (k = 1, NORD,
    gk *= g; h = gk;
    for (i = 1, k - 1, h = 2*I*u*deriv(h, u));
    rev += (-1)^k/k!*h);
  for (m = 1, NORD,
    evenline("C1P", m, Ser(vector(NORD + 1, j, 2*I*lcoef(polcoef(rev, j - 1, x), m, u)), x) + O(x^(NORD + 1))));
}
for (l = 1, NORD, evenline("C2", l, C2s[l]));
/* I3 jointly in n and eps: x scales both, eps = x E, n = x N; (2 - f)/(1 + (1 - f) R/(1 - eps))
   with f = 2n/(1 + n). */
{
  my(g3 = 2*(1 - x*E)/((1 + x*N)*(1 - x*E) + (1 - x*N)*sqrt((1 - x*E*z)*(1 - x*E/z) + O(x^(N3 + 1)))),
     a3 = fourier(g3, 0, N3), p);
  p = subst(truncate(a3), x, 1);
  for (j = 0, N3, for (i = 0, N3 - j, coefline("A3", 0, i, j, polcoef(polcoef(p, j, E), i, N))));
  for (l = 1, N3,
    p = subst(truncate(fourier(g3, l, N3)/(l*a3)), x, 1);
    for (j = l, N3, for (i = 0, N3 - j, coefline("C3", l, i, j, polcoef(polcoef(p, j, E), i, N)))));
}

/* ------------------------------------------------------------------------------------------------
   2. Reference geodesics. a = 6378137 m and 1/f = 298.257223563 exactly (NIMA TR8350.2 sect. 3).
   ------------------------------------------------------------------------------------------------ */
a = 6378137;
f = 1/(298257223563/10^9);
b = a*(1 - f);
e2 = f*(2 - f);
ep2 = e2/(1 - e2);

/* the phase of x + i y; a real zero (0.E-57) on either axis is read as zero, which arg() refuses */
atan2(y, x) = if (y == 0, if (x < 0, Pi, 0), if (x == 0, sign(y)*Pi/2, arg(x + I*y)));
/* the nearest double: 53 significant bits */
tof64(v) = if (v == 0, 0, my(e = exponent(v), q = 2^(52 - e)); round(v*q)/q);
/* a Rust f64 literal: 17 significant digits round-trip a double; 25 for a result */
lit(v, d) = {
  my(s);
  if (v == 0, return("0.0"));
  s = concat([c | c <- Vec(Strprintf(Str("%.", d, "g"), v)), c != " "]);
  if (#select(c -> c == "." || c == "e" || c == "E", Vec(s)) == 0, s = concat(s, ".0"));
  s;
}
row(v) = {
  my(s = "    [");
  for (i = 1, #v, s = concat(s, concat(if (i > 1, ", ", ""), lit(v[i][1], v[i][2]))));
  print(concat(s, "],"));
}
/* angle reduced to (-pi, pi] */
reduce(t) = { my(r = t - 2*Pi*round(t/(2*Pi))); if (r <= -Pi, r + 2*Pi, r); }
redlat(phi) = if (phi == 0, [0, 1], my(s = (1 - f)*sin(phi), c = cos(phi), h = sqrt(s^2 + c^2)); [s/h, c/h]);
latitude_of(sb, cb) = atan2(sb, (1 - f)*cb);

/* lambda(s2) - lambda(s1) along a geodesic with sin/cos alpha0 = sa0, ca0. The integrand
   (1 - f) sa0 D/(1 - ca0^2 sin^2), D = sqrt(1 + k^2 sin^2), peaks at sigma = pi/2 + m pi with width
   ~ sa0, so it is split as (1 - f) Dp sa0/(1 - ca0^2 sin^2) - (1 - f) sa0 k^2 cos^2/((D + Dp)(1 - ca0^2 sin^2)),
   Dp = sqrt(1 + k^2): the first term integrates to (1 - f) Dp omega12, omega continuous through the
   peaks (tan omega = sa0 tan sigma), and the second is bounded by sa0 e'^2 and goes to quadrature.
   On a meridian (sa0 = 0) each peak is a pole crossing and adds pi. */
omega_c(sa0, t) = {
  my(s = abs(sa0));
  sign(sa0)*(t - atan2((1 - s)*sin(t)*cos(t), cos(t)^2 + s*sin(t)^2));
}
lamint(sa0, ca0, s1, s2) = {
  my(sgn = 1, lo = s1, hi = s2, k2 = ep2*ca0^2, dp = sqrt(1 + ep2*ca0^2), pts, acc = 0, peaks = 0);
  if (s2 < s1, sgn = -1; lo = s2; hi = s1);
  pts = [lo];
  for (m = ceil((lo - Pi/2)/Pi), floor((hi - Pi/2)/Pi),
    my(p = Pi/2 + m*Pi); if (p > lo && p < hi, pts = concat(pts, p); peaks++));
  pts = concat(pts, hi);
  if (sa0 == 0, return(sgn*peaks*Pi));
  for (i = 1, #pts - 1,
    acc += intnum(t = pts[i], pts[i + 1],
                  cos(t)^2/((sqrt(1 + k2*sin(t)^2) + dp)*(1 - ca0^2*sin(t)^2))));
  sgn*((1 - f)*dp*(omega_c(sa0, hi) - omega_c(sa0, lo)) - (1 - f)*sa0*k2*acc);
}
arclen(ca0, s1, s2) = my(k2 = ep2*ca0^2); b*intnum(t = s1, s2, sqrt(1 + k2*sin(t)^2));
/* the reduced length m12 (Karney eq. 38), J = I1 - I2 by quadrature (eq. 40) */
redlen(ca0, s1, s2) = {
  my(k2 = ep2*ca0^2, w1 = sqrt(1 + k2*sin(s1)^2), w2 = sqrt(1 + k2*sin(s2)^2));
  b*(w2*cos(s1)*sin(s2) - w1*sin(s1)*cos(s2)
     - cos(s1)*cos(s2)*intnum(t = s1, s2, k2*sin(t)^2/sqrt(1 + k2*sin(t)^2)));
}

/* The direct problem: [lat2, lon2, azi2]. */
direct(lat1, lon1, azi1, s12) = {
  my(B1 = redlat(lat1), sa1 = sin(azi1), ca1 = cos(azi1), sa0, ca0, s1, s2, lo, hi, sb2, cb2, lam12);
  sa0 = sa1*B1[2]; ca0 = sqrt(ca1^2 + (sa1*B1[1])^2);
  s1 = atan2(B1[1], ca1*B1[2]);
  if (s12 == 0, return([lat1, reduce(lon1), atan2(sa1, ca1)]));
  /* the integrand lies in [1, sqrt(1 + k^2)], which brackets sigma12 */
  lo = s12/(b*sqrt(1 + ep2*ca0^2)); hi = s12/b;
  s2 = solve(t = s1 + min(lo, hi) - 1e-30, s1 + max(lo, hi) + 1e-30, arclen(ca0, s1, t) - s12);
  sb2 = ca0*sin(s2); cb2 = sqrt(sa0^2 + (ca0*cos(s2))^2);
  lam12 = lamint(sa0, ca0, s1, s2);
  [latitude_of(sb2, cb2), reduce(lon1 + lam12), atan2(sa0, ca0*cos(s2))];
}

/* lambda12 of the geodesic leaving canonical point 1 at azimuth alp1, to its first northward
   crossing of beta2 (Karney eq. 45); also returns what the distance needs. */
hybrid(B1, B2, alp1) = {
  my(sb1 = B1[1], cb1 = B1[2], sb2 = B2[1], cb2 = B2[2], sa1 = sin(alp1), ca1 = cos(alp1), sa0, ca0, ca2, s1, s2);
  sa0 = sa1*cb1; ca0 = sqrt(ca1^2 + (sa1*sb1)^2);
  ca2 = if (cb2 == cb1, abs(ca1), sqrt(ca1^2*cb1^2 + cb2^2 - cb1^2)/cb2);
  s1 = if (sb1 == 0 && ca1 < 0, -Pi, atan2(sb1, ca1*cb1));
  s2 = atan2(sb2, ca2*cb2);
  [lamint(sa0, ca0, s1, s2), s1, s2, sa0, ca0, sa0/cb2, ca2];
}

/* The inverse problem: [s12, azi1, azi2, m12]. */
inverse(lat1, lon1, lat2, lon2) = {
  my(lam, lonsign = 1, swapped = 0, latsign = 1, p1 = lat1, p2 = lat2, l1 = lon1, l2 = lon2, B1, B2,
     sa1, ca1, sa2, ca2, s12, m12, h, alp);
  /* the swap first: it reverses the sign of lambda12, which the longitude flip then fixes */
  if (abs(p1) < abs(p2), swapped = 1; [p1, p2] = [p2, p1]; [l1, l2] = [l2, l1]);
  lam = reduce(l2 - l1);
  if (lam < 0, lonsign = -1; lam = -lam);
  if (p1 > 0, latsign = -1; p1 = -p1; p2 = -p2);
  B1 = redlat(p1); B2 = redlat(p2);
  if (lam == 0 && p1 == p2,
    /* coincident points: the distance is zero and the azimuths are taken as 0 */
    return([0, 0, 0, 0]));
  if (lam == 0,
    /* the meridian, northward from the more southern point */
    s12 = arclen(1, atan2(B1[1], B1[2]), atan2(B2[1], B2[2]));
    m12 = redlen(1, atan2(B1[1], B1[2]), atan2(B2[1], B2[2]));
    sa1 = 0; ca1 = 1; sa2 = 0; ca2 = 1,
  if (B1[1] == 0 && B2[1] == 0 && lam <= (1 - f)*Pi,
    s12 = a*lam; m12 = b*sin(lam/(1 - f)); sa1 = 1; ca1 = 0; sa2 = 1; ca2 = 0,
    /* the general case; on the same parallel lambda12 is 0 for alpha1 <= pi/2 */
    alp = solve(t = if (B1 == B2, Pi/2, 0), Pi, hybrid(B1, B2, t)[1] - lam);
    h = hybrid(B1, B2, alp);
    s12 = arclen(h[5], h[2], h[3]);
    m12 = redlen(h[5], h[2], h[3]);
    sa1 = sin(alp); ca1 = cos(alp); sa2 = h[6]; ca2 = h[7]));
  /* undo the order: latitude flip -> cos negated; swap -> ends exchanged and reversed;
     longitude flip -> sin negated */
  if (latsign < 0, ca1 = -ca1; ca2 = -ca2);
  if (swapped, [sa1, ca1, sa2, ca2] = [-sa2, -ca2, -sa1, -ca1]);
  if (lonsign < 0, sa1 = -sa1; sa2 = -sa2);
  [s12, atan2(sa1, ca1), atan2(sa2, ca2), m12];
}

/* lands-on check: the inverse's azimuth and distance, through the direct solution, reach point 2 */
miss(lat1, lon1, lat2, lon2, r) = {
  my(d = direct(lat1, lon1, r[2], r[1]));
  sqrt((a*(d[1] - lat2))^2 + (a*cos(lat2)*reduce(d[2] - lon2))^2);
}

/* ---- the inverse rows ---- */
setrand(20261124);
u01() = random(2^62)/2^62;
rlat() = tof64(asin(2*u01() - 1));
rlon() = tof64(Pi*(2*u01() - 1));
HALFPI = tof64(Pi/2);
PI64 = tof64(Pi);
pts = List();
/* 40 random pairs, uniform on the sphere */
for (i = 1, 40, listput(pts, [rlat(), rlon(), rlat(), rlon()]));
/* 18 nearly antipodal, placed in Karney's astroid units about the antipode of point 1:
   lambda12 = pi + x f pi cos beta1, beta2 = -beta1 + y f pi cos^2 beta1 (eq. 53) */
{
  foreach([[-0.1, -0.05], [-0.382344, -0.220189], [-0.9, -0.9], [-1.0, 0], [-0.5, 0], [-1.5, 0],
           [-3.0, -0.2], [-0.05, -2.0], [-0.0001, -0.0001], [-2.0, -1.0e-6], [-0.7, -0.3], [-0.2, -1.2]],
    xy,
    my(lat1 = -tof64(0.15 + 1.3*u01()), B1 = redlat(lat1), beta1 = atan2(B1[1], B1[2]), lon1 = rlon(),
       lat2, beta2, lam12, sc = f*Pi*B1[2]);
    beta2 = -beta1 + xy[2]*sc*B1[2];
    lat2 = if (xy[2] == 0, -lat1, tof64(atan(tan(beta2)/(1 - f))));
    lam12 = Pi + xy[1]*sc;
    listput(pts, [lat1, lon1, lat2, tof64(reduce(lon1 + lam12))]));
  /* the same, mirrored and swapped, through the canonical transformation */
  listput(pts, [tof64(0.6), tof64(2.0), tof64(-0.598), tof64(2.0 - 3.139)]);
  listput(pts, [tof64(-0.2), tof64(-3.0), tof64(0.4), tof64(0.1415)]);
  /* both ends on the equator, beyond the conjugate point of the equator */
  listput(pts, [0, 0, 0, tof64(3.1350)]);
  listput(pts, [0, tof64(1.0), 0, tof64(1.0 - 3.1414)]);
  listput(pts, [0, 0, 0, PI64]);
  listput(pts, [tof64(0.3), 0, tof64(-0.3), PI64]);
}
/* 8 short lines */
{
  foreach([1e-12, 1e-9, 1e-6, 1e-3], d,
    my(lat1 = rlat(), lon1 = rlon());
    listput(pts, [lat1, lon1, tof64(lat1 + d*0.6), tof64(lon1 + d*0.8)]);
    listput(pts, [lat1, lon1, tof64(lat1 - d*0.3), lon1]));
}
/* poles, meridians, the equator, coincident points */
{
  listput(pts, [HALFPI, tof64(0.3), tof64(0.7), tof64(-2.1)]);
  listput(pts, [-HALFPI, 0, tof64(-0.2), tof64(1.5)]);
  listput(pts, [HALFPI, 0, -HALFPI, tof64(0.25)]);
  listput(pts, [HALFPI, 0, 0, 0]);
  listput(pts, [tof64(1.5707), tof64(0.1), tof64(-1.5706), tof64(3.0)]);
  listput(pts, [tof64(-1.2), tof64(0.5), tof64(0.4), tof64(0.5)]);
  listput(pts, [tof64(0.9), tof64(-1.0), tof64(0.9), tof64(-1.0)]);
  listput(pts, [tof64(-0.5), 0, tof64(0.3), PI64]);
  listput(pts, [0, 0, 0, tof64(0.5)]);
  listput(pts, [0, tof64(-3.0), 0, tof64(2.5)]);
  listput(pts, [0, 0, 0, tof64(3.13)]);
  listput(pts, [0, 0, 0, 0]);
  listput(pts, [tof64(0.7), tof64(0.2), tof64(0.7), tof64(1.9)]);
  listput(pts, [tof64(-0.7), tof64(0.2), tof64(-0.7), tof64(-2.9)]);
}
print("BEGIN inverse");
worst_miss = 0;
{
  for (i = 1, #pts,
    my(p = pts[i], r = inverse(p[1], p[2], p[3], p[4]));
    if (r[1] > 0, worst_miss = max(worst_miss, miss(p[1], p[2], p[3], p[4], r)));
    row([[p[1], 17], [p[2], 17], [p[3], 17], [p[4], 17], [r[1], 25], [r[2], 25], [r[3], 25], [r[4], 25]]));
}
print("END inverse");
printf("inverse: %d rows; direct(inverse) lands on point 2, worst miss %.3g m\n", #pts, worst_miss);

/* ---- the direct rows ---- */
dts = List();
for (i = 1, 30, listput(dts, [rlat(), rlon(), rlon(), tof64(2.0e7*u01())]));
{
  listput(dts, [rlat(), rlon(), rlon(), tof64(-1.3e7)]);
  listput(dts, [rlat(), rlon(), rlon(), tof64(3.5e7)]);
  listput(dts, [rlat(), rlon(), rlon(), tof64(9.0e7)]);
  listput(dts, [0, tof64(0.4), HALFPI, tof64(1.0e7)]);
  listput(dts, [0, 0, 0, tof64(1.0001965729e7)]);
  listput(dts, [HALFPI, tof64(0.2), tof64(2.5), tof64(5.0e6)]);
  listput(dts, [-HALFPI, 0, 0, tof64(2.0e7)]);
  listput(dts, [tof64(0.4), tof64(1.0), PI64, tof64(1.5e7)]);
  listput(dts, [rlat(), rlon(), rlon(), tof64(1e-3)]);
  listput(dts, [rlat(), rlon(), rlon(), 1]);
  listput(dts, [tof64(0.5), tof64(-0.3), tof64(1.2), 0]);
}
print("BEGIN direct");
{
  for (i = 1, #dts,
    my(p = dts[i], r = direct(p[1], p[2], p[3], p[4]));
    row([[p[1], 17], [p[2], 17], [p[3], 17], [p[4], 17], [r[1], 25], [r[2], 25], [r[3], 25]]));
}
print("END direct");
printf("direct: %d rows\n", #dts);

/* ---- closed forms (G3) and Karney's printed examples ---- */
/* `ok <name>`, or `FAIL <name>` when the value is not within half a unit of the last digit printed */
check(name, got, want, tol) = {
  if (abs(got - want) <= tol, printf("ok %s\n", name),
      printf("FAIL %s: %.20g, expected %.20g\n", name, got, want));
}
check("inverse rows land on point 2 through the direct solution", worst_miss, 0, 1e-30);
Q = a*ellE(sqrt(e2));
printf("closed form: quarter meridian a E(e) = %.25g m\n", Q);
check("quarter meridian by quadrature is a E(e)", arclen(1, 0, Pi/2), Q, 1e-40);
check("equator up to its conjugate point is a lambda12", inverse(0, 0, 0, 3)[1], 3*a, 1e-40);
{
  my(d = direct(40*Pi/180, 0, 30*Pi/180, 10^7), r = inverse(-30*Pi/180, 0, 29.9*Pi/180, 179.8*Pi/180),
     r3 = inverse(-30.12345*Pi/180, 0, -30.12344*Pi/180, 0.00005*Pi/180),
     h5 = hybrid(redlat(-30*Pi/180), redlat(29.9*Pi/180), 161.914*Pi/180), deg = 180/Pi);
  check("Karney Table 2 phi2", d[1]*deg, 41.79331020506, 5e-12);
  check("Karney Table 2 lambda12", d[2]*deg, 137.84490004377, 5e-12);
  check("Karney Table 2 alpha2", d[3]*deg, 149.09016931807, 5e-12);
  check("Karney Tables 4-6 s12", r[1], 19989832.827610, 5e-7);
  check("Karney Tables 4-6 alpha1", r[2]*deg, 161.89052473633, 5e-12);
  check("Karney Tables 4-6 alpha2", r[3]*deg, 18.09073724574, 5e-12);
  check("Karney Table 5 m12 at alpha1 = 161.914 deg", redlen(h5[5], h5[2], h5[3]), 57288.000110, 5e-7);
  check("Karney Table 3 s12", r3[1], 4.944208, 5e-7);
  check("Karney Table 3 alpha1", r3[2]*deg, 77.04353354237, 5e-12);
  check("Karney Table 3 alpha2", r3[3]*deg, 77.04350844913, 5e-12);
}
