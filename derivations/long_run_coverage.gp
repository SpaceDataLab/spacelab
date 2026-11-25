\\ Second engine for long_run_coverage.mac: the long-run coverage statistics of
\\ `crates/spacelab-constellation/src/long_run.rs` at 60 digits, by routes that share nothing with
\\ the code but the definitions.
\\
\\ (a) The incommensurate integrals over the whole revolution u in [-pi/2, 3pi/2], with alpha
\\     continued by its branch and Delta by its arccos form: P = (1/2 pi^2) int Delta du by intnum,
\\     twice (plain, and after u = m + r sin(theta)); nu = (TV(h+) + TV(h-))/(4 pi) from the roots of
\\     h' by its closed form, sampled at 3000 points a piece and solved; and the check that the
\\     integral of h' between consecutive roots is the difference of h, so the roots are those of
\\     the derivative of the h whose variation is taken.
\\ (b) G3, a polar orbit over an equatorial point: P = I(lambda)/pi^2 with
\\     I(lambda) = 2 int_0^lambda sin(mu) K(sin mu) dmu, K by the AGM, and
\\     nu = (2/pi)(kappa u* + acos(cos lambda/cos u*)) with u* from the quadratic in cos^2 u; both
\\     against (a). And P = 1/2 at lambda = pi/2 for any inclination.
\\ (c) The commensurate finite sums along the trajectory itself: f(t) = G . S(t) - cos(lambda) over
\\     one cycle t in [0, 2 pi Np], sampled 1000 times a revolution, every sign change solved, every
\\     sampled local maximum below zero refined on f' so that a grazing pass is not missed; the
\\     passes are the rising roots and the time covered the sum of the intervals with f > 0. Then
\\     the bounds of the module docs, |P_rep - P| <= 1/Np, against (a).
\\ (d) The rows `tests/long_run.rs` holds the code to, at the doubles the test passes, and the check
\\     that each row is in that file verbatim.
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/long_run_coverage.gp </dev/null
default(realprecision, 60);
default(path, "derivations:."); read("binary64.gp");
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, Strprintf("0   (|residual| %.1e)", abs(e)), Str("FAIL ", e)));

\\ The shortest decimal in Rust syntax that parses back to the same double: at most 17 digits.
lit(y) = {
  my(s, e10, mant);
  if(y == 0, return("0.0"));
  e10 = floor(log(abs(y * 1.)) / log(10));
  mant = y * 1. / 10^e10;
  if(abs(mant) >= 10, mant /= 10; e10++);
  if(abs(mant) < 1, mant *= 10; e10--);
  for(k = 0, 16,
    s = Strprintf(Str("%.", k, "fe%d"), mant, e10);
    if(f64(eval(s)) == y, return(s)));
  error("no round trip for ", y);
}
deg(x) = f64(x * Pi / 180);

\\ ---- The case: inclination, latitude, footprint, kappa (globals, set by setcase) ----
setcase(i, phi, lam, kap) = CI = cos(i); SI = sin(i); PHI = phi; LAM = lam; KAP = kap; SIG = if(CI < 0, -1, 1);
phis(u) = asin(SI * sin(u));
cfun(ps) = (cos(LAM) - sin(PHI) * sin(ps)) / (cos(PHI) * cos(ps));
dlt(u) = my(c = cfun(phis(u))); if(c >= 1, 0, if(c <= -1, Pi, acos(c)));
\\ alpha continued through the revolution: the branch of atan(cos i tan u) plus sigma pi a half-turn
alf(u) = if(abs(cos(u)) < 10^-55, SIG * u, atan(CI * tan(u)) + round(u / Pi) * Pi * SIG);
hh(s, u) = KAP * u - alf(u) + s * dlt(u);
dprime(u) = {
  my(ps = phis(u), dc = (cos(LAM) * sin(ps) - sin(PHI)) / (cos(PHI) * cos(ps)^2));
  -dc * (SI * cos(u) / cos(ps)) / sin(dlt(u));
}
hprime(s, u) = KAP - CI / (1 - SI^2 * sin(u)^2) + s * dprime(u);

\\ The cuts of [-pi/2, 3pi/2]: the turning points and the band edges of both halves.
cuts() = {
  my(c = List([-Pi/2, Pi/2, 3*Pi/2]), e, s);
  foreach([PHI - LAM, PHI + LAM, Pi - LAM - PHI, LAM - Pi - PHI], pe,
    if(abs(pe) < Pi/2 && SI > 0, s = sin(pe) / SI;
      if(abs(s) < 1, e = asin(s); listput(c, e); listput(c, Pi - e))));
  vecsort(Vec(c), , 8);
}
band(u) = my(d = dlt(u)); if(d == 0, 0, if(d == Pi, 2, 1));

\\ The ends and the roots of h' on (a, b), sampled at n points clustered at the ends.
knots(s, a, b, n) = {
  my(m = (a + b) / 2, r = (b - a) / 2, k = List([a]), us, fs);
  us = vector(n, j, m + r * sin(-Pi/2 + Pi * j / (n + 1)));
  fs = vector(n, j, hprime(s, us[j]));
  for(j = 1, n - 1, if(sign(fs[j]) != sign(fs[j + 1]) && fs[j] != 0,
    listput(k, solve(u = us[j], us[j + 1], hprime(s, u)))));
  listput(k, b); Vec(k);
}

\\ (a): [P by intnum, P after the sine substitution, nu, the largest |int h' - Delta h|]
integrals() = {
  my(c = cuts(), di = 0, di2 = 0, tv = 0, worst = 0, a, b, bd, k, m, r, step);
  for(j = 1, #c - 1, a = c[j]; b = c[j + 1]; bd = band((a + b) / 2);
    if(bd == 2, di += Pi * (b - a); di2 += Pi * (b - a));
    if(bd == 1,
      di += intnum(u = a, b, dlt(u));
      m = (a + b) / 2; r = (b - a) / 2;
      di2 += intnum(t = -Pi/2, Pi/2, dlt(m + r * sin(t)) * r * cos(t));
      foreach([1, -1], s,
        k = knots(s, a, b, 3000);
        for(q = 1, #k - 1,
          step = hh(s, k[q + 1]) - hh(s, k[q]);
          tv += abs(step);
          if(q > 1 && q < #k - 1, worst = max(worst, abs(intnum(u = k[q], k[q + 1], hprime(s, u)) - step)))))));
  [di / (2 * Pi^2), di2 / (2 * Pi^2), tv / (4 * Pi), worst];
}

\\ ---- (a) the M0 cases and more: [i deg, phi deg, lambda deg, kappa as a decimal] ----
\\ M0 (plans/18): Landsat-like, ISS at 35 and 60 degrees, near-polar at 85, low inclination; then a
\\ low orbit with the ground point always in its band, one with a band of Delta = pi, a retrograde
\\ one past 90 degrees, and footprints of a hemisphere.
{
cases = [[97.8, 35, 8, "0.0660234717"], [51.6, 35, 20, "0.0647731"], [51.6, 60, 20, "0.0647731"],
         [98.0, 85, 12, "0.0671"], [30.0, 10, 15, "0.0634199"], [5.0, 3, 10, "0.0702"],
         [85.0, 80, 20, "0.0655"], [140.0, -30, 25, "0.0701"], [51.6, 35, 90, "0.0647731"],
         [97.8, -70, 90, "0.0660234717"]];
}
introws = List(); worst_sub = 0; worst_deriv = 0; half_worst = 0;
{
  foreach(cases, c,
    my(i = deg(c[1]), phi = deg(c[2]), lam = if(c[3] == 90, f64(Pi / 2), deg(c[3])), kap = f64(eval(c[4])), res);
    setcase(i, phi, lam, kap);
    res = integrals();
    worst_sub = max(worst_sub, abs(res[1] - res[2]));
    worst_deriv = max(worst_deriv, res[4]);
    if(c[3] == 90, half_worst = max(half_worst, abs(res[1] - 1/2)));
    listput(introws, Strprintf("    (%s, %s, %s, %s, %s, %s),", lit(i), lit(phi), lit(lam), lit(kap), lit(f64(res[1])), lit(f64(res[3])))));
}
chk("p_by_intnum_and_after_the_sine_substitution_agree", worst_sub, 1e-35);
chk("the_integral_of_h_prime_between_its_roots_is_the_step_of_h", worst_deriv, 1e-35);
\\ lambda is the double below pi/2, so P sits below 1/2 by about (pi/2 - lambda) / pi
chk("p_is_one_half_for_a_hemisphere_footprint", half_worst, 1e-15);

\\ ---- (b) G3: a polar orbit over an equatorial point ----
ellk(k) = Pi / (2 * agm(1, sqrt(1 - k^2)));
polar_p(lam) = 2 * intnum(mu = 0, lam, sin(mu) * ellk(sin(mu))) / Pi^2;
polar_nu(lam, kap) = {
  my(c2 = cos(lam)^2, y, us);
  y = (-c2 * (1 - kap^2) + sqrt(c2^2 * (1 - kap^2)^2 + 4 * kap^2 * c2)) / (2 * kap^2);
  us = acos(sqrt(y));
  (2 / Pi) * (kap * us + acos(cos(lam) / cos(us)));
}
polarrows = List(); polar_worst = [0, 0];
{
  foreach([[5, "0.0660234717"], [20, "0.0647731"], [45, "0.0671"], [60, "0.0702"]], c,
    my(i = f64(Pi / 2), lam = deg(c[1]), kap = f64(eval(c[2])), res, p, nu);
    setcase(i, 0, lam, kap);
    res = integrals();
    \\ the double pi/2 leaves cos i = 6e-17, which moves the general answer by that order
    p = polar_p(lam); nu = polar_nu(lam, kap);
    polar_worst = [max(polar_worst[1], abs(res[1] - p)), max(polar_worst[2], abs(res[3] - nu))];
    listput(polarrows, Strprintf("    (%s, %s, %s, %s),", lit(lam), lit(kap), lit(f64(p)), lit(f64(nu)))));
}
chk("polar_equatorial_p_is_i_of_lambda_over_pi_squared", polar_worst[1], 1e-14);
chk("polar_equatorial_nu_is_the_closed_form", polar_worst[2], 1e-14);
\\ and with cos i = 0 exactly, to the working precision
{
  my(w = 0);
  foreach([5, 20, 45, 60], d, setcase(Pi / 2, 0, d * Pi / 180, 0.0647731);
    my(res = integrals()); w = max(w, abs(res[1] - polar_p(d * Pi / 180)) + abs(res[3] - polar_nu(d * Pi / 180, 0.0647731))));
  chk("polar_equatorial_closed_forms_at_cos_i_zero", w, 1e-35);
}

\\ ---- (c) the commensurate finite sums along the trajectory ----
X0 = 0;
fvis(t) = my(x = X0 - KAP * t); cos(PHI) * (cos(x) * cos(t) - sin(x) * CI * sin(t)) + sin(PHI) * SI * sin(t) - cos(LAM);
fvisp(t) = {
  my(x = X0 - KAP * t);
  cos(PHI) * (KAP * sin(x) * cos(t) - cos(x) * sin(t) + KAP * CI * cos(x) * sin(t) - CI * sin(x) * cos(t)) + sin(PHI) * SI * cos(t);
}
\\ [passes, time covered] over [0, 2 pi Np]
trajectory(Np, per) = {
  my(n = Np * per, h = 2 * Pi / per, roots = List(), t0, t1, f0, f1, f2, fm, tm, passes = 0, cov = 0, prev, first);
  f0 = fvis(0); f1 = fvis(h); first = f0;
  for(j = 1, n,
    t0 = (j - 1) * h; t1 = j * h; f2 = fvis(t1 + h);
    if(f0 <= 0 && f1 > 0, listput(roots, [solve(t = t0, t1, fvis(t)), 1]));
    if(f0 > 0 && f1 <= 0, listput(roots, [solve(t = t0, t1, fvis(t)), -1]));
    \\ a sampled local maximum below zero: is there a pass inside?
    if(j < n && f1 <= 0 && f1 >= f0 && f1 >= f2 && fvisp(t0) > 0 && fvisp(t1 + h) < 0,
      tm = solve(t = t0, t1 + h, fvisp(t)); fm = fvis(tm);
      if(fm > 0, listput(roots, [solve(t = t0, tm, fvis(t)), 1]); listput(roots, [solve(t = tm, t1 + h, fvis(t)), -1])));
    f0 = f1; f1 = f2);
  roots = vecsort(Vec(roots), 1);
  \\ the trajectory closes after the cycle, so a pass across t = 0 is one pass
  prev = if(first > 0, 0, -1);
  for(q = 1, #roots, if(roots[q][2] == 1, passes++; prev = roots[q][1], cov += roots[q][1] - prev));
  if(f0 > 0, cov += n * h - prev);
  [passes, cov];
}
\\ [Np, Nd, i deg, phi deg, lambda deg, offset as a fraction of a slot]
{
comcases = [[233, 16, 98.2, 35, 8, "0.1"], [233, 16, 98.2, 35, 8, "0.62"], [43, 3, 65, 20, 15, "0.37"],
            [15, 1, 51.6, 60, 20, "0.5"], [143, 10, 98.62, 85, 12, "0.05"], [29, 2, 30, 10, 15, "0.9"],
            [14, 1, 97.8, 0, 3, "0.25"]];
}
comrows = List(); com_bound = -1;
{
  foreach(comcases, c,
    my(Np = c[1], Nd = c[2], i = deg(c[3]), phi = deg(c[4]), lam = deg(c[5]), slot = 2 * Pi / Np, off, tr, p, nu, res);
    off = f64(eval(c[6]) * slot);
    setcase(i, phi, lam, Nd / Np);
    \\ x0, the node east of the ground point, is minus the offset
    X0 = -off;
    tr = trajectory(Np, 1000);
    p = tr[2] / (2 * Pi * Np);
    res = integrals();
    com_bound = max(com_bound, abs(p - res[1]) * Np - 1);
    listput(comrows, Strprintf("    (%d, %d, %s, %s, %s, %s, %d, %s),", Np, Nd, lit(i), lit(phi), lit(lam), lit(off), tr[1], lit(f64(p)))));
}
chk("commensurate_p_is_within_one_over_np_of_the_integral", max(com_bound, 0), 1e-30);

\\ ---- (d) the rows ----
print("incommensurate rows (i, phi, lambda, kappa, coverage, passes per revolution):"); foreach(introws, s, print(s));
print("polar rows over an equatorial point (lambda, kappa, coverage, passes per revolution):"); foreach(polarrows, s, print(s));
print("commensurate rows (Np, Nd, i, phi, lambda, ground offset, passes per cycle, coverage):"); foreach(comrows, s, print(s));
{
  my(file = "crates/spacelab-constellation/tests/long_run.rs", text, missing = 0, have = Map());
  text = iferr(readstr(Str("../", file)), E, iferr(readstr(file), E2, []));
  foreach(text, line, mapput(have, line, 1));
  foreach([introws, polarrows, comrows], rs, foreach(rs, s, if(!mapisdefined(have, s), missing++)));
  chk("every_row_is_in_tests_long_run_rs_verbatim", missing, 1/2);
}
