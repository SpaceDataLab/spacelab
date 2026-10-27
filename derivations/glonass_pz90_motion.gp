/* The GLONASS PZ-90 right-hand side in PARI/GP: the second engine for
   derivations/glonass_pz90_motion.mac.

   The acceleration is formed as grad U - 2 Omega x v - Omega x (Omega x r) + a_ls, with grad U taken
   by PARI's own differentiation of U = mu/r - mu J2 ae^2 (3 z^2 - r^2)/(2 r^5), written as a
   rational function of x, y, z and an independent q = r and differentiated by the chain rule
   d/dx = partial_x + (x/r) partial_q, at the module test's state in exact rationals and 60 digits. It must equal the ICD's grouping to 1e-50 and the
   Rust test's references (the f64 values printed by tools/gen_glonass_oracle.mac) to 1e-15.
   A mismatch prints `FAIL`; the last line counts them.

       timeout 300 gp -q -f derivations/glonass_pz90_motion.gp </dev/null                            */

default(realprecision, 60);
bad = 0;
check(name, got, want, tol) = {
  my(d = abs(got - want) / abs(want));
  if (d > tol, bad++; printf("FAIL %s: %.45g vs %.45g\n", name, got, want),
               printf("%s: ok (rel %.2e)\n", name, d));
}

mu = 398600441800000; ae = 6378136; J2 = 108262575 / 10^11; om = 7292115 / 10^11;
st = [15000000, -8000000, 19000000]; vel = [1500, 3000, -900]; acc = [1, -2, 3] / 10^6;

/* U(x, y, z) with r = sqrt(x^2 + y^2 + z^2); derivatives by the chain rule through q = r. */
U = mu / q - mu * J2 * ae^2 * (3 * z^2 - q^2) / (2 * q^5);
grad(f) = {
  my(fq = deriv(f, q), sub = [x, y, z], s = [st[1], st[2], st[3]], r = sqrt(s * s~), g);
  g = vector(3, k, deriv(f, sub[k]) + fq * sub[k] / q);
  vector(3, k, substvec(g[k], [x, y, z, q], [s[1], s[2], s[3], r]));
}
cross(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]];
W = [0, 0, om];
derived = grad(U) - 2 * cross(W, vel) - cross(W, cross(W, st)) + acc;

r = sqrt(st * st~); k = (3/2) * J2 * mu * ae^2 / r^5; qz = 5 * st[3]^2 / r^2;
{icd = [-mu * st[1] / r^3 + k * st[1] * (qz - 1) + om^2 * st[1] + 2 * om * vel[2] + acc[1],
       -mu * st[2] / r^3 + k * st[2] * (qz - 1) + om^2 * st[2] - 2 * om * vel[1] + acc[2],
       -mu * st[3] / r^3 + k * st[3] * (qz - 3) + acc[3]];}
want = [1.5656172518917186e-1, -6.891682343422499e-2, -4.570127719226161e-1];
axes = ["x", "y", "z"];
{for (j = 1, 3,
  check(Str("row ", axes[j], ": gradient route = ICD grouping"), derived[j], icd[j], 1e-50);
  check(Str("row ", axes[j], ": gradient route = Rust reference"), derived[j], want[j], 1e-15));}
printf("glonass pz90 motion: %d rows off\n", bad);
quit;
