/* Second engine for derivations/density_calibration_ridge.mac: the same degree-2 ridge fit of
   crates/spacelab-propagation/src/density_calibration.rs, two more ways.

   1. exact rationals: the column-scaled system (Bt'W Bt + ridge D^-2) ct = Bt'W y by matsolve,
      c = ct / D -- every entry of Bt is rational at the Pythagorean sample points;
   2. a Cholesky factorisation, written out here, of the unscaled (B'W B + ridge I) at 100 digits,
      the basis taken from pollegendre and Rodrigues rather than from the module's recursion.
   Both are held to each other and to the 40 digits Maxima prints. Prints `PASS` lines and the
   coefficients; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/density_calibration_ridge.gp </dev/null                */

default(realprecision, 100);
nfac2(n, m) = (if (m == 0, 1, 2))*(2*n + 1)*(n - m)!/(n + m)!;
/* Columns in the module's packing at degree 2: [n, m, trig] with trig 0 = cos, 1 = sin. */
cols = [[0,0,0], [1,0,0], [1,1,0], [1,1,1], [2,0,0], [2,1,0], [2,1,1], [2,2,0], [2,2,1]];
dsq = vector(9, k, my(c = cols[k]); (if (c[2] > 0, 2, 1))*nfac2(c[1], c[2]));
/* The rational part of a column at (x, s, cos l, sin l): s^m d^m P_n/dx^m times cos/sin(m l). */
trig(m, cl, sl, which) = my(z = (cl + I*sl)^m); if (which, imag(z), real(z));
dern(P, m) = for (i = 1, m, P = deriv(P, 'x)); P;
{
scaled(k, x0, s, cl, sl) =
  my(c = cols[k]);
  s^c[2]*subst(dern(pollegendre(c[1]), c[2]), 'x, x0)*trig(c[2], cl, sl, c[3]);
}
lats = [[0, 1], [3/5, 4/5], [-3/5, 4/5], [5/13, 12/13], [-12/13, 5/13], [4/5, 3/5], [-7/25, 24/25]];
lsts = [[1, 0], [3/5, 4/5], [0, 1], [-5/13, 12/13], [-1, 0], [-4/5, -3/5], [8/17, -15/17]];
ridge = 1/8;
samples = List(); idx = 0;
{
for (i = 1, #lats, for (j = 1, #lsts,
  idx++;
  listput(samples, [lats[i][1], lats[i][2], lsts[j][1], lsts[j][2], 1 + (idx % 3)/2, ((7*idx) % 11 - 5)/20])));
}
listput(samples, [3/5, 4/5, 1, 0, -1, 100]);
kept = select(q -> q[5] > 0, Vec(samples));
print(if (#kept == #samples - 1, "PASS", "FAIL"), " the non-positive weight is dropped");

/* 1. Exact. */
At = matrix(9, 9); rt = vector(9)~;
{
for (i = 1, #kept,
  my(q = kept[i], b = vector(9, k, scaled(k, q[1], q[2], q[3], q[4]))~);
  At += q[5]*b*b~; rt += q[5]*q[6]*b);
}
At += ridge*matdiagonal(vector(9, k, 1/dsq[k]));
ct = matsolve(At, rt);
{
print(if (#select(v -> type(v) != "t_INT" && type(v) != "t_FRAC", Vec(ct)) == 0, "PASS", "FAIL"),
      " the scaled solution is rational");
}
exact = vector(9, k, ct[k]/sqrt(dsq[k]));

/* 2. Cholesky on the unscaled system, basis values in reals. */
A = matrix(9, 9); r = vector(9)~;
{
for (i = 1, #kept,
  my(q = kept[i], b = vector(9, k, sqrt(dsq[k])*scaled(k, q[1], q[2], q[3], q[4]))~);
  A += q[5]*b*b~; r += q[5]*q[6]*b);
}
A += ridge*matid(9);
{
cholesky(M) =
  my(n = #M, L = matrix(n, n));
  for (j = 1, n,
    my(d = M[j, j] - sum(k = 1, j - 1, L[j, k]^2));
    if (d <= 0, error("not positive definite"));
    L[j, j] = sqrt(d);
    for (i = j + 1, n, L[i, j] = (M[i, j] - sum(k = 1, j - 1, L[i, k]*L[j, k]))/L[j, j]));
  L;
}
L = cholesky(A);
yv = vector(9)~;
for (i = 1, 9, yv[i] = (r[i] - sum(k = 1, i - 1, L[i, k]*yv[k]))/L[i, i]);
chol = vector(9);
forstep (i = 9, 1, -1, chol[i] = (yv[i] - sum(k = i + 1, 9, L[k, i]*chol[k]))/L[i, i]);
d = vecmax(vector(9, k, abs(chol[k] - exact[k])));
print(if (d < 10^-90, "PASS", "FAIL"), " Cholesky route = exact route, worst ", Strprintf("%.3g", d));

/* 3. Maxima's 40 digits. */
{
maxima = [3.991886066339695479785650917647154737132e-4, 6.74128686395576831213617006473300160594e-3,
  -1.513262697598961599559731085903958035477e-2, -1.211757795574940054943234913115365133629e-2,
  -1.183572393512159245116984193662544329991e-2, 2.296616759764013093122590156029985436361e-2,
  -9.986650000321766828316610726228466742806e-3, -1.755920858958894574924007585739119089178e-2,
  -9.748357157258055805197073221485300295581e-3];
}
d = vecmax(vector(9, k, abs(maxima[k] - exact[k])));
print(if (d < 10^-40, "PASS", "FAIL"), " exact route = Maxima's digits, worst ", Strprintf("%.3g", d));
for (k = 1, 9, print("ridge_coefficient_", k - 1, " = ", Strprintf("%.25e", exact[k])));
