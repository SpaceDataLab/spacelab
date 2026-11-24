\\ Second engine for derivations/se3_exponential.mac: at phi = (3/10, -7/10, 11/10), 100 digits, the
\\ series definitions of Exp, V and V^-1 (sum_k K^k/k!, sum_k K^k/(k+1)!, sum_k B_k/k! K^k with
\\ B_1 = -1/2) against the closed forms se3.rs evaluates. Prints the count of entries that disagree
\\ by more than 1e-90 and the 20-digit fixture of the Rust test. Last, item 7's half-turn fixture
\\ from Rodrigues' formula and Shepperd's quaternion, against Maxima's; it prints FAIL if they differ.
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

\\ Item 7's fixture (plans/18 Ф28.K, F117), from Rodrigues' formula here: Exp(th n) for
\\ th = pi - 10^-k rounded to doubles, and the exact logarithm of those doubles through Shepperd's
\\ quaternion, against the 40 digits se3_exponential.mac prints and se3.rs's test reads at 25.
f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
\\ Shepperd's quaternion of an exact 3x3 matrix on the branch the code takes, sign-fixed to w >= 0,
\\ and from it [angle, rotation vector]; atan2(s, w) is arg(w + i s), w >= 0.
shep(m) = {
  my(tr = m[1, 1] + m[2, 2] + m[3, 3], b = 1, c, rt, h, q, s, t);
  c = [1 + tr, 1 + 2 * m[1, 1] - tr, 1 + 2 * m[2, 2] - tr, 1 + 2 * m[3, 3] - tr];
  for (i = 2, 4, if (c[i] > c[b], b = i));
  rt = sqrt(c[b]); h = 1 / (2 * rt);
  q = if (b == 1, [rt / 2, (m[3, 2] - m[2, 3]) * h, (m[1, 3] - m[3, 1]) * h, (m[2, 1] - m[1, 2]) * h],
      if (b == 2, [(m[3, 2] - m[2, 3]) * h, rt / 2, (m[1, 2] + m[2, 1]) * h, (m[1, 3] + m[3, 1]) * h],
      if (b == 3, [(m[1, 3] - m[3, 1]) * h, (m[1, 2] + m[2, 1]) * h, rt / 2, (m[2, 3] + m[3, 2]) * h],
                  [(m[2, 1] - m[1, 2]) * h, (m[1, 3] + m[3, 1]) * h, (m[2, 3] + m[3, 2]) * h, rt / 2])));
  if (q[1] < 0, q = -q);
  s = sqrt(q[2]^2 + q[3]^2 + q[4]^2);
  t = 2 * arg(q[1] + s * sqrt(-1));
  [t, t / s * [q[2], q[3], q[4]]];
}
rod(n, t) = { my(K = [0, -n[3], n[2]; n[3], 0, -n[1]; -n[2], n[1], 0]); matid(3) + sin(t) * K + (1 - cos(t)) * K^2; }
axes = [[2, -3, 6] / 7, [6, 2, -3] / 7, [-3, 6, 2] / 7];
{
HT = [[8.690264724542266626139756637243498379342e-1, -1.303539708681339969236466820270258617018e0, 2.607079417362679889103940289907984954268e0],
  [2.684222274505537066513231134946910872561e0, 8.947407581685123217016710469625359451168e-1, -1.342111137252768533256615567473455436281e0],
  [-1.345968280109911377999049684288772568244e0, 2.691936560219822755998099368577545136489e0, 8.973121867399409525650803821870219455079e-1],
  [8.975693295970837763004945261062816390849e-1, -1.346353994395625715313871770365550865105e0, 2.692707988791251328901483578318844917255e0],
  [2.692785131648394237989631835423286058634e0, 8.975950438827980454201525964479361787891e-1, -1.346392565824197042697935383152034887617e0],
  [-1.346396422967054287256023094804662942773e0, 2.692792845934108472782580712503681705919e0, 8.975976153113694570177050784660125087640e-1],
  [8.975978724542266749775086203925063538798e-1, -1.346396808681339936169141965262612408217e0, 2.692793617362679923203031240742656231502e0],
  [2.692793694505537052556300656185420041567e0, 8.975978981685123678070164745234820395773e-1, -1.346396847252768551710524711785223059366e0],
  [-1.346396851109911325854776521509349583889e0, 2.692793702219822804303799782324083938759e0, 8.975979007399409347679332607746946462529e-1],
  [8.975979009970837927006276076149473945265e-1, -1.346396851495625612753818019912212841854e0, 2.692793702991251378101882822844842183580e0]];
}
{
my(off = 0, worst = 0.);
for (k = 1, 10,
  my(n = axes[(k - 1) % 3 + 1], m = matrix(3, 3, i, j, f64(rod(n, Pi - 10^-k)[i, j])), g = shep(m)[2]);
  for (i = 1, 3, my(d = abs(g[i] - HT[k][i])); worst = max(worst, d); if (d > 1e-38, off++)));
printf("half-turn logarithms off Maxima's by more than 1e-38: %d off (largest %.3g)\n", off, worst);
if (off, print("FAIL: the fixture disagrees with Maxima's"));
}
quit;
