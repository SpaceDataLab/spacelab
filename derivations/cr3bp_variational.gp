/* The CR3BP variational Jacobian at 100 digits: the second engine for
   derivations/cr3bp_variational.mac and the reference the `jacobian` gate in
   crates/spacelab-astro/src/periodic.rs asserts the shipped f64 code against.

   A = df/ds is taken by numerical differentiation (`derivnum`) of the vector field itself -- no
   Hessian formula is written here -- at the test's state, with the Earth-Moon mass ratio, both as
   the doubles the test passes. Checked: the coded Hessian entries against it, tr A = 0, and
   A' Omega + Omega A = 0 for the synodic symplectic form. Every mismatch prints `FAIL`.

       timeout 300 gp -q -f derivations/cr3bp_variational.gp </dev/null                           */

default(realprecision, 100);

f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
check(name, got, want, tol) = {
  my(d = abs(got - want));
  if (d > tol, printf("FAIL %s: %.40g vs %.40g, |d| = %.3e\n", name, got, want, d),
               printf("%s: ok (|d| = %.3e)\n", name, d));
}

mu = f64(0.012150585);
field(s) = {
  my(x = s[1], y = s[2], z = s[3], r1 = sqrt((x + mu)^2 + y^2 + z^2),
     r2 = sqrt((x - 1 + mu)^2 + y^2 + z^2));
  [s[4], s[5], s[6],
   x + 2 * s[5] - (1 - mu) * (x + mu) / r1^3 - mu * (x - 1 + mu) / r2^3,
   y - 2 * s[4] - (1 - mu) * y / r1^3 - mu * y / r2^3,
   -(1 - mu) * z / r1^3 - mu * z / r2^3];
}
s0 = apply(f64, [0.85, 0.05, 0.03, 0.01, 0.12, -0.02]);
{A = matrix(6, 6, i, j,
  derivnum(t = s0[j], field(vector(6, k, if (k == j, t, s0[k])))[i]));}

/* The coded entries, transcribed from `jacobian`. */
{
  my(x = s0[1], y = s0[2], z = s0[3], m1 = 1 - mu, m2 = mu, d1 = x + mu, d2 = x - 1 + mu,
     r1 = sqrt(d1^2 + y^2 + z^2), r2 = sqrt(d2^2 + y^2 + z^2), U);
  U = [1 - m1 / r1^3 - m2 / r2^3 + 3 * m1 * d1^2 / r1^5 + 3 * m2 * d2^2 / r2^5,
       1 - m1 / r1^3 - m2 / r2^3 + 3 * m1 * y^2 / r1^5 + 3 * m2 * y^2 / r2^5,
       -m1 / r1^3 - m2 / r2^3 + 3 * m1 * z^2 / r1^5 + 3 * m2 * z^2 / r2^5,
       3 * m1 * d1 * y / r1^5 + 3 * m2 * d2 * y / r2^5,
       3 * m1 * d1 * z / r1^5 + 3 * m2 * d2 * z / r2^5,
       3 * m1 * y * z / r1^5 + 3 * m2 * y * z / r2^5];
  my(names = ["uxx", "uyy", "uzz", "uxy", "uxz", "uyz"], at = [[4,1], [5,2], [6,3], [4,2], [4,3], [5,3]]);
  for (k = 1, 6,
    check(Str("coded ", names[k], " vs derivnum"), U[k], A[at[k][1], at[k][2]], 1e-80);
    printf("  %s = %.40g\n", names[k], A[at[k][1], at[k][2]]));
}
/* The rest of A is exact: identity coupling, Coriolis, zero position block. */
{K = matrix(6, 6, i, j, if (i <= 3 && j == i + 3, 1, if (i == 4 && j == 5, 2, if (i == 5 && j == 4, -2, 0))));}
{for (i = 1, 6, for (j = 1, 6,
  if (i <= 3 || j >= 4, check(Str("A[", i - 1, "][", j - 1, "]"), A[i, j], K[i, j], 1e-80))));}
check("trace A", trace(A), 0, 1e-80);
{Om = [0,-2,0,1,0,0; 2,0,0,0,1,0; 0,0,0,0,0,1; -1,0,0,0,0,0; 0,-1,0,0,0,0; 0,0,-1,0,0,0];}
E = A~ * Om + Om * A;
check("max |A' Omega + Omega A|", vecmax(apply(abs, concat(Vec(E)))), 0, 1e-80);
quit;
