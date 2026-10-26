/* WGS-84 geodetic coordinates from their defining equation, at 80 digits -- the second engine for
   derivations/wgs84_geodesy.mac and the reference values crates/spacelab-earth/tests/wgs84.rs
   asserts the shipped f64 code against.

   The ellipsoid is the two defining parameters a = 6378137 m and 1/f = 298.257223563, taken as
   exact rationals; b = a(1 - f) and e^2 = f(2 - f) are then exact rationals too.

   The inverse map is NOT Vermeille's closed form (the shipped algorithm) but the equation that
   defines the geodetic latitude: the point lies on the ellipsoid normal at latitude phi,
       F(phi) = rho sin phi - z cos phi - e^2 N(phi) sin phi cos phi = 0,   rho = sqrt(x^2 + y^2),
   solved by bracketed root finding between the geocentric latitude atan(z/rho), where
   F = -e^2 N sin cos has the sign of -z, and the pole of the same hemisphere, where F = rho sign(z);
   the height is then h = rho cos phi + z sin phi - a W(phi).
   derivations/wgs84_geodesy.mac shows the forward map satisfies both lines identically. Every
   solution is re-substituted into the forward map here and the residual printed.

   Rows are printed in Rust syntax. Inverse rows: (x, y, z, latitude rad, longitude rad, height m),
   integer-metre inputs (exact doubles). Forward rows: (latitude rad, longitude rad, height m, x, y,
   z), dyadic angles and integer heights (exact doubles).

       timeout 120 gp -q -f derivations/wgs84_geodesy.gp </dev/null                               */

default(realprecision, 80);

a = 6378137;
f = 1 / (298257223563 / 10^9);
b = a * (1 - f);
e2 = f * (2 - f);

W(s) = sqrt(1 - e2 * s^2);

forward(phi, lam, h) = {
  my(s = sin(phi), c = cos(phi), n = a / W(s));
  [(n + h) * c * cos(lam), (n + h) * c * sin(lam), (n * (1 - e2) + h) * s];
}

normal_residual(t, rho, z) = rho * sin(t) - z * cos(t) - e2 * a / W(sin(t)) * sin(t) * cos(t);

latitude(rho, z) = {
  my(lo, hi);
  if (rho == 0, return(sign(z) * Pi / 2));
  if (z == 0, return(0));
  lo = atan(z / rho); hi = sign(z) * Pi / 2;
  if (normal_residual(lo, rho, z) * normal_residual(hi, rho, z) >= 0,
      error("no sign change at ", [rho, z]));
  solve(t = min(lo, hi), max(lo, hi), normal_residual(t, rho, z));
}

geodetic(x, y, z) = {
  my(rho = sqrt(x^2 + y^2), phi = latitude(rho, z));
  [phi, atan2(y, x), rho * cos(phi) + z * sin(phi) - a * W(sin(phi))];
}

atan2(y, x) = if (x == 0 && y == 0, 0, arg(x + I * y));

/* 25 significant digits as a Rust literal: PARI writes "1.5 e-7", Rust needs "1.5e-7". */
fmt(v) = if (abs(v) < 10^-60, "0.0", concat([c | c <- Vec(Strprintf("%.25g", v)), c != " "]));

printf("constants: b = %.40g\n", b);
printf("constants: e2 = %.40g\n", e2);
printf("constants: f = %.40g\n", f);

/* Equator, poles, near-axis, mid latitudes, LEO to GEO and beyond, below the surface, and well
   inside the Earth but outside the 43 km lens Vermeille refuses. */
{pts = [
  [6378137, 0, 0], [7378137, 0, 0], [0, 0, 6356752], [0, 0, -7000000], [1, 0, 6400000],
  [-3, 2, -6356000], [4510731, 4510731, 0], [3194419, -3194419, 4487348],
  [4000000, 3000000, 5000000], [-2500000, 1200000, -5800000], [42164000, 0, 0],
  [29814000, 29814000, 1000000], [-1000000, -6000000, 2900000], [4400000, 0, 4500000],
  [100000, 0, 50000], [600000, -800000, -300000], [1, 1, 100000000], [6378137, 1, 1]
];}
worst = 0;
{
  for (i = 1, #pts,
    my(p = pts[i], g = geodetic(p[1], p[2], p[3]), back = forward(g[1], g[2], g[3]));
    worst = max(worst, normlp(back - p));
    printf("    (%d.0, %d.0, %d.0, %s, %s, %s),\n", p[1], p[2], p[3], fmt(g[1]), fmt(g[2]), fmt(g[3])));
}
printf("inverse: forward(geodetic(p)) - p, worst over %d points: %.3g m\n", #pts, worst);

/* Forward map at dyadic latitudes and longitudes (exact doubles), integer heights. */
{fwd = [
  [0, 0, 0], [1/2, 1/4, 0], [-5/8, 3, 550000], [3/4, -2, 700000], [25/16, 1, 35786000],
  [-1/1024, 1/1024, -1000], [201/128, -3/2, 400000], [-201/128, 5/2, 0]
];}
{
  for (i = 1, #fwd,
    my(q = fwd[i], x = forward(q[1], q[2], q[3]));
    printf("    (%s, %s, %d.0, %s, %s, %s),\n", fmt(q[1] * 1.), fmt(q[2] * 1.), q[3],
           fmt(x[1]), fmt(x[2]), fmt(x[3])));
}
