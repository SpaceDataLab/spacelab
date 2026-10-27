\\ Second engine for derivations/se3_exponential.mac: at phi = (3/10, -7/10, 11/10), 100 digits, the
\\ series definitions of Exp, V and V^-1 (sum_k K^k/k!, sum_k K^k/(k+1)!, sum_k B_k/k! K^k with
\\ B_1 = -1/2) against the closed forms se3.rs evaluates. Prints the count of entries that disagree
\\ by more than 1e-90 and the 20-digit fixture of the Rust test.
\\
\\ Run: timeout 300 gp -q -f derivations/se3_exponential.gp </dev/null
default(realprecision, 100);
hat(u) = [0, -u[3], u[2]; u[3], 0, -u[1]; -u[2], u[1], 0];
phi = [3/10, -7/10, 11/10];
K = hat(phi);
th = sqrt(phi[1]^2 + phi[2]^2 + phi[3]^2);
I3 = matid(3);
nmax = 200;
expS = sum(k = 0, nmax, K^k / k!);
jacS = sum(k = 0, nmax, K^k / (k + 1)!);
jinvS = I3 - K / 2 + sum(k = 2, nmax, bernfrac(k) / k! * K^k);
expC = I3 + sin(th) / th * K + (1 - cos(th)) / th^2 * K^2;
jacC = I3 + (1 - cos(th)) / th^2 * K + (th - sin(th)) / th^3 * K^2;
jinvC = I3 - K / 2 + (1 / th^2 - (1 + cos(th)) / (2 * th * sin(th))) * K^2;
off = 0;
{
  for (i = 1, 3, for (j = 1, 3,
    if (abs(expS[i, j] - expC[i, j]) > 1e-90, off++);
    if (abs(jacS[i, j] - jacC[i, j]) > 1e-90, off++);
    if (abs(jinvS[i, j] - jinvC[i, j]) > 1e-90, off++);
    if (abs((jacC * jinvC)[i, j] - I3[i, j]) > 1e-90, off++)));
}
printf("se3 series against closed forms: %d of 36 entries off by more than 1e-90\n", off);
t = jacS * [1, -2, 1/2]~;
printf("Exp[0][1] = %.20g, V[2][1] = %.20g, Vinv[1][2] = %.20g, t[0] = %.20g\n", \
       expS[1, 2], jacS[3, 2], jinvS[2, 3], t[1]);
quit;
