/* The IS-GPS-200 user algorithm at 60 digits in PARI/GP: the second engine for
   derivations/gps_broadcast_orbit.mac.

   Written independently of the Maxima file: Kepler's equation by solve() on a bracket rather than
   Newton, the true anomaly in the ICD's half-angle form rather than atan2, the BeiDou GEO rotation
   as explicit Rz Rx matrices, and the velocity as derivnum() of the position rather than a
   symbolic derivative. Inputs are the module sample's doubles, converted to their exact binary
   values. Every value Maxima prints at 40 digits must agree to 1e-30 relative; a mismatch prints
   `FAIL`, and the last line counts them.

       timeout 300 gp -q -f derivations/gps_broadcast_orbit.gp </dev/null                            */

default(realprecision, 60);

f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }

bad = 0;
check(name, got, want) = {
  my(d = abs(got - want) / abs(want));
  if (d > 1e-30, bad++; printf("FAIL %s: %.45g vs %.45g\n", name, got, want),
                 printf("%s: ok (rel %.2e)\n", name, d));
}

/* eph = [sqrt_a, e, i0, omega0, omega, m0, delta_n, idot, omega_dot, cuc, cus, crc, crs, cic, cis,
   toe]; geo = 1 for the BDS-SIS-ICD GEO path. */
position(eph, tk, mu, oe, geo) = {
  my(A = eph[1]^2, e = eph[2], n = sqrt(mu / A^3) + eph[7], M = eph[6] + n * tk, E, nu, phi,
     u, r, i, xp, yp, Om, g, Rz, Rx, s = sin(2 * phi), c);
  E = solve(x = M - 1, M + 1, x - e * sin(x) - M);
  nu = 2 * atan(sqrt((1 + e) / (1 - e)) * tan(E / 2));
  phi = nu + eph[5];
  s = sin(2 * phi); c = cos(2 * phi);
  u = phi + eph[11] * s + eph[10] * c;
  r = A * (1 - e * cos(E)) + eph[13] * s + eph[12] * c;
  i = eph[3] + eph[8] * tk + eph[15] * s + eph[14] * c;
  xp = r * cos(u); yp = r * sin(u);
  Om = eph[4] + (eph[9] - if (geo, 0, oe)) * tk - oe * eph[16];
  g = [xp * cos(Om) - yp * cos(i) * sin(Om), xp * sin(Om) + yp * cos(i) * cos(Om), yp * sin(i)]~;
  if (!geo, return(g));
  Rz = [cos(oe * tk), sin(oe * tk), 0; -sin(oe * tk), cos(oe * tk), 0; 0, 0, 1];
  Rx = [1, 0, 0; 0, cos(-5 * Pi / 180), sin(-5 * Pi / 180); 0, -sin(-5 * Pi / 180), cos(-5 * Pi / 180)];
  Rz * Rx * g;
}

{sample = apply(f64, [5153.65, 0.0062, 0.9599, -1.234, 0.567, 0.321, 4.5e-9, -2.1e-10, -8.1e-9,
                     1.2e-6, 8.3e-6, 210.0, 35.0, -1.1e-7, 9.0e-8, 216000.0]);}
mu_gps = 398600500000000; oe_gps = f64(7.2921151467e-5);
{want = [
  [0, [-1.608349047629949522608307177496015129125e7, 1.245509414443383025514302169956380305266e7,
       1.683329974311097511202157632207521635224e7],
      [4.386156686763534240872302271518575758199e2, -2.132868891403938462852896741488921059079e3,
       2.009134984342218859848980124961002776766e3]],
  [1800, [-1.541415114366485200684389006242593955029e7, 8.228261553544978826839208835223353847592e6,
          1.982090795702325132508994429627142134235e7],
         [2.838977683361724098871682382161423540394e2, -2.535947046765945106357328123330572673875e3,
          1.291283960177011331596598677536856884211e3]],
  [-7200, [-1.727221023687484819761309720069030491116e7, 1.967569356933729596983169688362624037098e7,
           -3.674667302932292817188733379826896841437e6],
          [-5.39625783743251115366765879699559491198e2, 9.182671115107704676621603652503407311922e1,
           3.144095093111117689203125539001010362227e3]]];}
axes = ["x", "y", "z"];
{for (j = 1, #want, my(tk = want[j][1], p = position(sample, tk, mu_gps, oe_gps, 0),
                      v = derivnum(t = tk, position(sample, t, mu_gps, oe_gps, 0)));
  for (k = 1, 3,
    check(Str("GPS tk = ", tk, " position ", axes[k]), p[k], want[j][2][k]);
    check(Str("GPS tk = ", tk, " velocity ", axes[k]), v[k], want[j][3][k])));}

{geo = apply(f64, [6493.4, 0.0004, 0.02, -1.234, 0.567, 0.321, 4.5e-9, -2.1e-10, -8.1e-9,
                  1.2e-6, 8.3e-6, 210.0, 35.0, -1.1e-7, 9.0e-8, 216000.0]);}
p = position(geo, 300, 398600441800000, f64(7.292115e-5), 1);
{wgeo = [-3.899982835142000127548946848277510675753e7, 1.586104032190708188350330687563835373037e7,
         1.980797178700234885081610668226004636799e6];}
for (k = 1, 3, check(Str("BeiDou GEO tk = 300 position ", axes[k]), p[k], wgeo[k]));
printf("gps broadcast orbit: %d values off by more than 1e-30\n", bad);
quit;
