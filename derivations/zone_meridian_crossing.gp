/* Second engine for derivations/zone_meridian_crossing.mac: where an edge of a zone crosses a
   meridian, at 50 digits, by a route that shares nothing with the f64 code of
   crates/spacelab/src/zone.rs. The rows between BEGIN and END are the reference the unit tests of
   zone.rs hold the crossings to; derivations/verify.sh requires them byte for byte.

   The edge is the shortest geodesic from (lat1, lon1) to (lat2, lon2). Its azimuth alpha1 at the
   first point solves Karney's hybrid problem (eq. 45) by bracketed root finding over [0, pi], with
   the longitude as a quadrature on the auxiliary sphere,
       d lambda / d sigma = (1 - f) sin alpha0 sqrt(1 + k^2 sin^2 sigma) / (1 - cos^2 alpha0 sin^2 sigma),
   split at its peaks as in derivations/ellipsoid_geodesic.gp, whose quadrature and inverse this
   file restates (the distance and the reduced length dropped). The crossing of longitude lon is the
   sigma at which that quadrature from sigma1 reaches the reduced offset of lon along the edge's
   sweep, by root finding, and its latitude is atan(tan beta / (1 - f)), sin beta = cos alpha0 sin sigma.
   Each crossing is checked to lie on the edge: its sigma between the ends', its longitude the one
   asked for. Each check prints `ok <name>` or `FAIL <name>`.

   Inputs are doubles: every angle is rounded to the nearest double (53 significant bits, ties to
   even) before use, so a row holds the exact inputs the f64 code sees, printed with 17
   significant digits, which read back as the same double. Rows: [lat1, lon1, lat2, lon2, lon,
   lat_crossing], radians, the crossing to 25 significant digits.

       timeout 300 gp -q -f derivations/zone_meridian_crossing.gp </dev/null                       */

default(realprecision, 50);

a = 6378137;
f = 1/(298257223563/10^9);
e2 = f*(2 - f);
ep2 = e2/(1 - e2);

/* The double nearest v, ties to even: v scaled by a power of two into [2^52, 2^53), rounded to an
   integer, scaled back. The exponent is found by comparison, not read off the float. */
f64near(v) = {
  my(s, m, e, q, t, r);
  if (v == 0, return(0));
  s = sign(v); m = abs(v);
  e = floor(log(m)/log(2));
  while (2^e > m, e--);
  while (2^(e + 1) <= m, e++);
  q = 2^(52 - e); t = m*q; r = floor(t);
  if (t - r > 1/2 || (t - r == 1/2 && r % 2 == 1), r++);
  s*r/q;
}
rad(d) = f64near(d*Pi/180);
lit(v, d) = {
  my(s);
  if (v == 0, return("0.0"));
  s = concat([c | c <- Vec(Strprintf(Str("%.", d, "g"), v)), c != " "]);
  if (#select(c -> c == "." || c == "e" || c == "E", Vec(s)) == 0, s = concat(s, ".0"));
  s;
}

atan2(y, x) = if (y == 0, if (x < 0, Pi, 0), if (x == 0, sign(y)*Pi/2, arg(x + I*y)));
reduce(t) = { my(r = t - 2*Pi*round(t/(2*Pi))); if (r <= -Pi, r + 2*Pi, r); }
redlat(phi) = if (phi == 0, [0, 1], my(s = (1 - f)*sin(phi), c = cos(phi), h = sqrt(s^2 + c^2)); [s/h, c/h]);
latitude_of(sb, cb) = atan2(sb, (1 - f)*cb);

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
hybrid(B1, B2, alp1) = {
  my(sb1 = B1[1], cb1 = B1[2], sb2 = B2[1], cb2 = B2[2], sa1 = sin(alp1), ca1 = cos(alp1), sa0, ca0, ca2, s1, s2);
  sa0 = sa1*cb1; ca0 = sqrt(ca1^2 + (sa1*sb1)^2);
  ca2 = if (cb2 == cb1, abs(ca1), sqrt(ca1^2*cb1^2 + cb2^2 - cb1^2)/cb2);
  s1 = if (sb1 == 0 && ca1 < 0, -Pi, atan2(sb1, ca1*cb1));
  s2 = atan2(sb2, ca2*cb2);
  [lamint(sa0, ca0, s1, s2), s1, s2, sa0, ca0, sa0/cb2, ca2];
}
/* The azimuth of the shortest geodesic at the first point, in (-pi, pi]: Karney's canonical order
   (phi1 <= 0, |phi2| <= |phi1|, 0 <= lambda12 <= pi), the hybrid root, the order undone. */
azimuth1(lat1, lon1, lat2, lon2) = {
  my(lam, lonsign = 1, swapped = 0, latsign = 1, p1 = lat1, p2 = lat2, l1 = lon1, l2 = lon2, B1, B2,
     sa1, ca1, sa2, ca2, h, alp);
  if (abs(p1) < abs(p2), swapped = 1; [p1, p2] = [p2, p1]; [l1, l2] = [l2, l1]);
  lam = reduce(l2 - l1);
  if (lam < 0, lonsign = -1; lam = -lam);
  if (p1 > 0, latsign = -1; p1 = -p1; p2 = -p2);
  B1 = redlat(p1); B2 = redlat(p2);
  if (lam == 0,
    sa1 = 0; ca1 = 1; sa2 = 0; ca2 = 1,
  if (B1[1] == 0 && B2[1] == 0 && lam <= (1 - f)*Pi,
    sa1 = 1; ca1 = 0; sa2 = 1; ca2 = 0,
    alp = solve(t = if (B1 == B2, Pi/2, 0), Pi, hybrid(B1, B2, t)[1] - lam);
    h = hybrid(B1, B2, alp);
    sa1 = sin(alp); ca1 = cos(alp); sa2 = h[6]; ca2 = h[7]));
  if (latsign < 0, ca1 = -ca1; ca2 = -ca2);
  if (swapped, [sa1, ca1, sa2, ca2] = [-sa2, -ca2, -sa1, -ca1]);
  if (lonsign < 0, sa1 = -sa1; sa2 = -sa2);
  atan2(sa1, ca1);
}

/* The latitude at which the edge from (lat1, lon1) to (lat2, lon2) crosses longitude lon, and
   checks that the crossing lies on the edge. */
crossing(lat1, lon1, lat2, lon2, lon) = {
  my(alp = azimuth1(lat1, lon1, lat2, lon2), B1 = redlat(lat1), sa0, ca0, s1, sweep, target, sg,
     s2, sx, sb, cb);
  sa0 = sin(alp)*B1[2]; ca0 = sqrt(cos(alp)^2 + (sin(alp)*B1[1])^2);
  s1 = atan2(B1[1], cos(alp)*B1[2]);
  sweep = reduce(lon2 - lon1);
  target = reduce(lon - lon1);
  if (sweep > 0 && target < 0, target += 2*Pi);
  if (sweep < 0 && target > 0, target -= 2*Pi);
  if (abs(target) >= abs(sweep), error("the longitude is outside the edge's sweep"));
  /* the far end: the sigma at which the quadrature reaches the whole sweep */
  s2 = solve(t = s1, s1 + Pi - 1e-30, lamint(sa0, ca0, s1, t) - sweep);
  sx = solve(t = s1, s2, lamint(sa0, ca0, s1, t) - target);
  sb = ca0*sin(sx); cb = sqrt(sa0^2 + (ca0*cos(sx))^2);
  /* the far end lands on the second vertex */
  ok_end = max(ok_end, abs(latitude_of(ca0*sin(s2), sqrt(sa0^2 + (ca0*cos(s2))^2)) - lat2));
  ok_lon = max(ok_lon, abs(reduce(lon1 + lamint(sa0, ca0, s1, sx) - lon)));
  latitude_of(sb, cb);
}

ok_end = 0; ok_lon = 0;
/* [lat1, lon1, lat2, lon2, lon] in degrees: a mid-latitude edge; one across the antimeridian,
   westward; a long northward edge; one heading south-west; a short edge near the equator; a
   steep edge, 4 degrees from a meridian; an edge passing 1.5 degrees from the north pole; one in
   the far south; one eastward along a parallel's chord; a 9000 km edge. */
{
  cases = [[10, 20, 30, 50, 35],
           [40, 170, 50, -160, -175],
           [-45, 10, 60, 40, 27],
           [55, -30, 20, -70, -41],
           [0.5, 0, -0.5, 3, 1.25],
           [-30, 100, 30, 104, 101],
           [80, 0, 80, 170, 90],
           [-70, -120, -80, 150, -165],
           [45, -10, 45, 10, 3],
           [-20, -60, 50, 30, -10]];
}
print("BEGIN zone_meridian_crossing.gp");
{
  for (i = 1, #cases,
    my(c = cases[i], v = vector(5, j, rad(c[j])), x);
    x = crossing(v[1], v[2], v[3], v[4], v[5]);
    print("    [", lit(v[1], 17), ", ", lit(v[2], 17), ", ", lit(v[3], 17), ", ", lit(v[4], 17), ", ",
          lit(v[5], 17), ", ", lit(x, 25), "],"));
}
print("END zone_meridian_crossing.gp");
printf("%s far ends land on the second vertex (worst %.3g rad)\n", if (ok_end < 1e-40, "ok", "FAIL"), ok_end);
printf("%s crossings lie on the asked longitude (worst %.3g rad)\n", if (ok_lon < 1e-40, "ok", "FAIL"), ok_lon);
/* the double rounding: exact on a double, ties to even, and the nearest otherwise */
{
  printf("%s f64near rounds to the nearest double, ties to even\n",
         if (f64near(1 + 2^-53) == 1 && f64near(1 + 3*2^-53) == 1 + 2^-51 && f64near(0.1) == 3602879701896397/2^55
             && f64near(-Pi) == -884279719003555/2^48 && f64near(2^-60) == 2^-60, "ok", "FAIL"));
}
