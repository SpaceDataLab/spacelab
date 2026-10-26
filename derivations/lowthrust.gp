/* Low-thrust acceleration references at 100 digits: the second engine for
   derivations/lowthrust.mac and the values the acceleration gate in
   crates/spacelab-astro/src/lowthrust.rs asserts the shipped f64 code against.

   -mu r/|r|^3 + aT u at the test's (r, v) doubles, mu = 398600 km^3/s^2, aT = 3.7e-5 (as a double),
   for tangential, retrograde and inertial steering; and the energy rate v . a + mu (r . v)/|r|^3,
   which derivations/lowthrust.mac shows is aT v.u, against that closed form.
   Every mismatch prints `FAIL`.

       timeout 300 gp -q -f derivations/lowthrust.gp </dev/null                                    */

default(realprecision, 100);

f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
check(name, got, want, tol) = {
  my(d = abs(got - want));
  if (d > tol, printf("FAIL %s: %.40g vs %.40g, |d| = %.3e\n", name, got, want, d),
               printf("%s: ok (|d| = %.3e)\n", name, d));
}
unit(a) = a / sqrt(a * a~);

mu = 398600;
r = apply(f64, [6811.0, -2743.0, 1902.0]); v = apply(f64, [1.734, 6.219, -2.845]);
d = apply(f64, [0.41, -0.72, 0.56]); aT = f64(3.7e-5);
g = -mu * r / sqrt(r * r~)^3;
laws = [unit(v), -unit(v), unit(d)];
names = ["Tangential", "Retrograde", "Inertial"];
{want = [[-0.006212348343026272914044675, 0.002538181155593251958807047, -0.001752283544005648307675690],
        [-0.006230535615742461862434361, 0.002472952407599031038092380, -0.001722443410858267937348995],
        [-0.006206272737827484630136448, 0.002478928113496249825082230, -0.001716644513354264596043664]];}
{for (k = 1, 3,
  my(a = g + aT * laws[k]);
  for (i = 1, 3, check(Str(names[k], " a[", i, "]"), a[i], want[k][i], 1e-27));
  check(Str(names[k], " energy rate"), v * a~ + mu * (r * v~) / sqrt(r * r~)^3,
        aT * (v * laws[k]~), 1e-90))};
quit;
