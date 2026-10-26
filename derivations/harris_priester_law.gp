/* The modified Harris-Priester density at 50 digits -- the second engine for
   derivations/harris_priester_law.mac and the reference values the module tests in
   crates/spacelab-propagation/src/harris_priester/mod.rs assert the shipped f64 code against.

   The law is written from its definition, not from the shipped code: the altitude is the radial
   distance r - R(phi') above the WGS-84 ellipsoid (a = 6378137 m and 1/f = 298.257223563 as exact
   rationals), the min and max densities are exponential in h between the bracketing nodes, and
       rho = rho_min + (rho_max - rho_min) cos(psi/2)^n,
   psi the angle between the satellite and the bulge apex, the Sun direction turned pi/6 east about
   +z. Above the 1000 km ceiling the density is 0; below 100 km the lowest bracket's exponential
   continues. The table is `harris_priester/tables.rs` verbatim, the decimal literals of the
   mean-activity table (Montenbruck & Gill, "Satellite Orbits", Springer 2005, §3.5).

   Each row is (satellite position m, n, rho kg/m^3) for the shared Sun position below; positions
   are integer metres (exact doubles). Equatorial points sit at exactly tabulated altitudes.

       timeout 120 gp -q -f derivations/harris_priester_law.gp </dev/null                         */

default(realprecision, 50);

{table = [
  [100000, 4.974e-07, 4.974e-07], [120000, 2.490e-08, 2.490e-08],
  [130000, 8.377e-09, 8.710e-09], [140000, 3.899e-09, 4.059e-09],
  [150000, 2.122e-09, 2.215e-09], [160000, 1.263e-09, 1.344e-09],
  [170000, 8.008e-10, 8.758e-10], [180000, 5.283e-10, 6.010e-10],
  [190000, 3.617e-10, 4.297e-10], [200000, 2.557e-10, 3.162e-10],
  [210000, 1.839e-10, 2.396e-10], [220000, 1.341e-10, 1.853e-10],
  [230000, 9.949e-11, 1.455e-10], [240000, 7.488e-11, 1.157e-10],
  [250000, 5.709e-11, 9.308e-11], [260000, 4.403e-11, 7.555e-11],
  [270000, 3.430e-11, 6.182e-11], [280000, 2.697e-11, 5.095e-11],
  [290000, 2.139e-11, 4.226e-11], [300000, 1.708e-11, 3.526e-11],
  [320000, 1.099e-11, 2.511e-11], [340000, 7.214e-12, 1.819e-11],
  [360000, 4.824e-12, 1.337e-11], [380000, 3.274e-12, 9.955e-12],
  [400000, 2.249e-12, 7.492e-12], [420000, 1.558e-12, 5.684e-12],
  [440000, 1.091e-12, 4.355e-12], [460000, 7.701e-13, 3.362e-12],
  [480000, 5.474e-13, 2.612e-12], [500000, 3.916e-13, 2.042e-12],
  [520000, 2.819e-13, 1.605e-12], [540000, 2.042e-13, 1.267e-12],
  [560000, 1.488e-13, 1.005e-12], [580000, 1.092e-13, 7.997e-13],
  [600000, 8.070e-14, 6.390e-13], [620000, 6.012e-14, 5.123e-13],
  [640000, 4.519e-14, 4.121e-13], [660000, 3.430e-14, 3.325e-13],
  [680000, 2.632e-14, 2.691e-13], [700000, 2.043e-14, 2.185e-13],
  [720000, 1.607e-14, 1.779e-13], [740000, 1.281e-14, 1.452e-13],
  [760000, 1.036e-14, 1.190e-13], [780000, 8.496e-15, 9.776e-14],
  [800000, 7.069e-15, 8.059e-14], [840000, 4.680e-15, 5.741e-14],
  [880000, 3.200e-15, 4.210e-14], [920000, 2.210e-15, 3.130e-14],
  [960000, 1.560e-15, 2.360e-14], [1000000, 1.150e-15, 1.810e-14]
];}

a = 6378137;
f = 1 / (298257223563 / 10^9);
e2 = f * (2 - f);

height(p) = {
  my(r = sqrt(p * p~), s = p[3] / r);
  r - a * sqrt((1 - e2) / (1 - e2 * (1 - s^2)));
}

/* The exponential through the bracket containing h, the lowest bracket below the floor. */
interp(h, col) = {
  my(i = 1);
  while (i < #table - 1 && h > table[i + 1][1], i++);
  my(h0 = table[i][1], h1 = table[i + 1][1], r0 = table[i][col], r1 = table[i + 1][col]);
  r0 * exp(-(h - h0) / ((h1 - h0) / log(r0 / r1)));
}

density(sun, p, n) = {
  my(h = height(p), u, apex, cpsi, k);
  if (h > table[#table][1], return(0));
  u = sun / sqrt(sun * sun~);
  apex = [u[1] * cos(Pi / 6) - u[2] * sin(Pi / 6), u[1] * sin(Pi / 6) + u[2] * cos(Pi / 6), u[3]];
  cpsi = apex * p~ / sqrt(p * p~);
  k = sqrt((1 + cpsi) / 2);
  interp(h, 2) + (interp(h, 3) - interp(h, 2)) * k^n;
}

fmt(v) = if (abs(v) < 10^-300, "0.0", concat([c | c <- Vec(Strprintf("%.25g", v)), c != " "]));

/* Right ascension 20 deg, declination -15 deg, 1 au, rounded to integer metres. */
sun = [135786003622, 49422063548, -38718778044];
apex_dir(r) = {
  my(u = sun / sqrt(sun * sun~));
  [u[1] * cos(Pi / 6) - u[2] * sin(Pi / 6), u[1] * sin(Pi / 6) + u[2] * cos(Pi / 6), u[3]] * r;
}
{
  pts = [
    /* equator at tabulated altitudes: the nodes themselves, and one between two nodes */
    [[a + 100000, 0, 0], 4], [[a + 400000, 0, 0], 4], [[0, a + 420000, 0], 2],
    [[-(a + 800000), 0, 0], 6], [[a + 1000000, 0, 0], 4], [[0, -(a + 310000), 0], 4],
    /* above the ceiling, and below the floor */
    [[a + 1050000, 0, 0], 4], [[a + 90000, 0, 0], 4],
    /* off the equator, both hemispheres */
    [[4100000, 4900000, 1100000], 4], [[-4400000, 3700000, 3300000], 6],
    [[-3100000, -3700000, 4800000], 2], [[3600000, -6200000, 1300000], 4],
    [[0, 0, 6856752], 4], [[1, 1, -7056752], 2]
  ];
  /* near the apex and the antipode, rounded to integer metres */
  foreach ([250000, 550000, 950000], alt,
    my(d = apex_dir(a + alt));
    pts = concat(pts, [[round(d), 4], [round(-d), 4], [round(d), 6]]));
}
{
  for (i = 1, #pts,
    my(p = pts[i][1], n = pts[i][2]);
    printf("    ([%d.0, %d.0, %d.0], %d.0, %s),\n", p[1], p[2], p[3], n, fmt(density(sun, p, n))));
}
