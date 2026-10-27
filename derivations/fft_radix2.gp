/* Second engine for derivations/fft_radix2.mac: the DFT in exact cyclotomic arithmetic.

   Every value lives in Q(zeta_N), zeta = exp(2 pi i/N), held as a polynomial in zeta modulo the
   N-th cyclotomic polynomial, so nothing here is rounded. The forward kernel is w = zeta^-1 and
   i = zeta^(N/4). For N = 4, 8, ..., 64 and the Gaussian-integer input
   x_n = ((n mod 5) - 2) + i((3n mod 7) - 3) it counts the entries where
     1. the recursive radix-2 (Danielson-Lanczos) transform differs from the direct DFT sum;
     2. (1/N) sum_k X_k w^(-n k) differs from x_n -- the inverse with its 1/N;
     3. sum |X_k|^2 differs from N sum |x_n|^2 (Parseval), conj taking zeta to zeta^-1.
   Then it prints the N = 16 transform at 40 digits, in the format of the .mac file, so the two
   engines' references can be compared line by line.

       timeout 300 gp -q -f derivations/fft_radix2.gp </dev/null                                   */

default(realprecision, 60);

cyc(N, e) = Mod(x, polcyclo(N))^(e % N);
cconj(N, z) = Mod(subst(lift(z), x, x^(N - 1)), polcyclo(N));
gi(N, n) = ((n % 5) - 2) + ((3 * n % 7) - 3) * cyc(N, N / 4);

direct(N, v) = vector(N, k, sum(n = 0, N - 1, v[n + 1] * cyc(N, -n * (k - 1))));

/* Radix-2 recursion on an n-point sub-transform, all of it in Q(zeta_N): the n-point kernel is
   zeta_N^(N/n), so the twiddle of output k is zeta_N^(-k N/n). */
radix2(N, n, v) = {
  if (n == 1, return(v));
  my(h = n / 2, e, o, out = vector(n));
  e = radix2(N, h, vector(h, j, v[2 * j - 1]));
  o = radix2(N, h, vector(h, j, v[2 * j]));
  for (k = 1, h,
    my(t = cyc(N, -(k - 1) * (N / n)) * o[k]);
    out[k] = e[k] + t; out[k + h] = e[k] - t);
  out;
}

{
my(split = 0, inv = 0, pars = 0, entries = 0);
for (e = 2, 6,
  my(N = 2^e, v = vector(N, n, gi(N, n - 1)), X = direct(N, v), R = radix2(N, N, v * Mod(1, polcyclo(N))));
  for (k = 1, N, entries++; if (R[k] != X[k], split++));
  for (n = 1, N,
    if (sum(k = 1, N, X[k] * cyc(N, (n - 1) * (k - 1))) / N != v[n], inv++));
  if (sum(k = 1, N, X[k] * cconj(N, X[k])) != N * sum(n = 1, N, v[n] * cconj(N, v[n])), pars++));
printf("danielson_lanczos (N = 4 .. 64): %d of %d entries differ\n", split, entries);
printf("inverse_with_1_over_n (N = 4 .. 64): %d of %d samples differ\n", inv, entries);
printf("parseval (N = 4 .. 64): %d of 5 lengths differ\n", pars);
}

{
my(N = 16, v = vector(N, n, gi(N, n - 1)), X = direct(N, v), z);
for (k = 1, N,
  z = subst(lift(X[k]), x, exp(2 * Pi * I / N));
  printf("X[%d] = %.40g  %.40g\n", k - 1, real(z), imag(z)));
}
