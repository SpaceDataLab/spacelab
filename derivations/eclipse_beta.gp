\\ Second engine for eclipse_beta.mac: the eclipse fractions of
\\ `crates/spacelab-propagation/src/eclipse.rs` at 60 digits, by methods that share nothing with it.
\\
\\ (a) The cylinder: the half-arc in shadow as the root, in the angle v from the anti-Sun point, of
\\     the defining condition |p|^2 - (p.s)^2 - R^2 = 0, found by solve() on the circle; against
\\     acos(cos(beta*)/cos(beta)) and the shipped atan2 form, and at beta = 0 against asin(R/r).
\\ (b) The cones: the half-arcs as the roots of the apparent-disc conditions of
\\     `shadow::lighting_ratio`, separation = alpha_E -/+ alpha_S with the Sun at its distance,
\\     against the closed onsets phi_u = beta* - asin((R_S - R)/D) and phi_p = beta* + asin((R_S + R)/D).
\\ (c) The orbit mean of the light the cones take, 1 - nu, by intnum of the textbook circle-
\\     intersection lens (not the shipped segment form) across the penumbra arc.
\\ (d) The onset: f against its series (1/pi) sqrt(2 tan(beta*) d) (1 - c d), c from the .mac.
\\
\\ It prints the rows `tests/eclipse.rs` holds the f64 code to, each on exact binary64 inputs (the
\\ radius, the angle and the distance as the doubles the test passes), and checks that each row is
\\ in that file verbatim. A cylinder row is (r, beta, lo, f, hi): f the fraction, and lo and hi the
\\ fractions at |beta| + delta and |beta| - delta, delta = 8 u tan(beta*), the band an error in beta
\\ of delta moves f across. A cone row is (r, D, beta, umbra, penumbra, mean).
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/eclipse_beta.gp </dev/null
default(realprecision, 60);
default(path, "derivations:."); read("binary64.gp");
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, Strprintf("0   (|residual| %.1e)", abs(e)), Str("FAIL ", e)));
u = 2^-53;
Re = 6378137; Rs = 6957 * 10^5; au = 149597870700;

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

\\ ---- (a) The cylinder ----
fcyl_acos(r, b) = my(c = sqrt(1 - (Re / r)^2) / cos(b)); if(c >= 1, 0, acos(c) / Pi);
fcyl_atan2(r, b) = my(rho = Re / r, s = sin(abs(b))); if(s >= rho, 0, atan(sqrt((rho - s) * (rho + s)) / sqrt((1 - rho) * (1 + rho))) / Pi);
\\ the root of the condition on the circle: p = r(-cos v cos b, sin v, -cos v sin b) with the Sun on +x
perp2(r, b, v) = r^2 - (r * cos(v) * cos(b))^2 - Re^2;
fcyl_root(r, b) = if(perp2(r, b, 0) >= 0, 0, solve(v = 0, Pi / 2, perp2(r, b, v)) / Pi);
{
  my(worst = 0, worst2 = 0);
  foreach([6778137, 7078137, 8378137, 42164137, 6378138], r,
    my(bs = asin(Re / r));
    foreach([0, 1/10, 1/2, 1, bs * 9/10, bs * 99/100, bs - 10^-6], b,
      worst = max(worst, abs(fcyl_root(r, b) - fcyl_acos(r, b)));
      worst2 = max(worst2, abs(fcyl_atan2(r, b) - fcyl_acos(r, b)))));
  chk("cylinder_half_arc_is_the_root_of_the_shadow_condition", worst, 1e-50);
  chk("cylinder_atan2_form_is_the_acos_form", worst2, 1e-50);
  chk("cylinder_at_beta_zero_is_asin_rho_over_pi",
      vecmax(apply(r -> abs(fcyl_atan2(r, 0) - asin(Re / r) / Pi), [6778137, 7078137, 42164137])), 1e-55);
  chk("cylinder_at_beta_star_is_zero", fcyl_atan2(7078137, asin(Re / 7078137)), 1e-25);
}

\\ ---- (d) The onset series: pi f = sqrt(2 tan(bs) d) (1 + (2 cos^2 bs - 5)/(12 sin bs cos bs) d + O(d^2)) ----
{
  my(r = 7078137, bs = asin(Re / r), c1 = (2 * cos(bs)^2 - 5) / (12 * sin(bs) * cos(bs)), q);
  q = vector(9, j, my(d = 10^-(j + 3), f = fcyl_atan2(r, bs - d), lead = sqrt(2 * tan(bs) * d) / Pi);
                   (f / (lead * (1 + c1 * d)) - 1) / d^2);
  \\ what the series leaves is second order: over d^2 it settles to one constant as d falls from
  \\ 1e-4 to 1e-12; a wrong c1 would leave a first-order residual, which over d^2 grows as 1/d
  chk("onset_series_residual_is_second_order", q[9] / q[5] - 1, 1e-6);
}

\\ ---- (b) The cones, from the apparent-disc conditions ----
\\ the satellite at v from the anti-Sun point, the Sun at D on +x, beta above the orbit plane
sat(r, v) = [-r * cos(v), r * sin(v), 0];
sunp(D, b) = [D * cos(b), 0, D * sin(b)];
dt(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
crs(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
ang(a, b) = my(c = crs(a, b)); atan(sqrt(dt(c, c)) / dt(a, b)) + if(dt(a, b) < 0, Pi, 0);
\\ [separation, alpha_E, alpha_S] seen from the satellite
discs(r, D, b, v) = {
  my(p = sat(r, v), S = sunp(D, b), q = S - p);
  [ang(-p, q), asin(Re / r), asin(Rs / sqrt(dt(q, q)))];
}
gumbra(r, D, b, v) = my(x = discs(r, D, b, v)); x[1] - (x[2] - x[3]);
gpen(r, D, b, v) = my(x = discs(r, D, b, v)); x[1] - (x[2] + x[3]);
froot(g, r, D, b) = if(g(r, D, b, 0) >= 0, 0, solve(v = 0, Pi, g(r, D, b, v)) / Pi);
\\ the closed onsets of the .mac and the cylinder formula with them
onsets(r, D) = my(bs = asin(Re / r)); [bs - asin((Rs - Re) / D), bs + asin((Rs + Re) / D)];
fcone(ph, b) = if(ph <= 0, 0, my(c = cos(ph) / cos(b)); if(c >= 1, 0, acos(c) / Pi));
{
  my(wu = 0, wp = 0);
  foreach([6778137, 7078137, 42164137], r,
    foreach([au, 147100000000, 152100000000], D,
      my(ph = onsets(r, D));
      foreach([0, 1/10, ph[1] * 9/10, ph[1] - 10^-6, (ph[1] + ph[2]) / 2, ph[2] - 10^-6], b,
        wu = max(wu, abs(froot(gumbra, r, D, b) - fcone(ph[1], b)));
        wp = max(wp, abs(froot(gpen, r, D, b) - fcone(ph[2], b))))));
  chk("umbra_half_arc_is_the_root_of_separation_equal_alpha_E_minus_alpha_S", wu, 1e-45);
  chk("penumbra_half_arc_is_the_root_of_separation_equal_alpha_E_plus_alpha_S", wp, 1e-45);
}

\\ ---- (c) The orbit mean of 1 - nu, nu from the textbook lens ----
lens(d, a, b) = {
  if(d >= a + b, return(0));
  if(d <= abs(a - b), return(Pi * min(a, b)^2));
  b^2 * acos((d^2 + b^2 - a^2) / (2 * d * b)) + a^2 * acos((d^2 + a^2 - b^2) / (2 * d * a))
    - sqrt((a + b - d) * (d - a + b) * (d + a - b) * (d + a + b)) / 2;
}
lost(r, D, b, v) = my(x = discs(r, D, b, v)); lens(x[1], x[2], x[3]) / (Pi * x[3]^2);
mean_lost(r, D, b) = {
  my(ph = onsets(r, D), vu = Pi * fcone(ph[1], b), vp = Pi * fcone(ph[2], b));
  if(vp == 0, return(0));
  (vu + intnum(v = vu, vp, lost(r, D, b, v))) / Pi;
}
{
  \\ the quadrature has converged: the same integral at 80 digits
  my(a = mean_lost(7078137, au, 1/2), c);
  default(realprecision, 80); c = mean_lost(7078137, au, 1/2); default(realprecision, 60);
  chk("mean_lost_light_quadrature_converged", a - c, 1e-40);
  \\ the mean lies between the umbra and the penumbra fractions
  my(ph = onsets(7078137, au), m = mean_lost(7078137, au, 1/2));
  chk("mean_lost_light_between_umbra_and_penumbra",
      if(fcone(ph[1], 1/2) < m && m < fcone(ph[2], 1/2), 0, 1), 1/2);
}

\\ ---- The rows `tests/eclipse.rs` holds the code to ----
rows = List();
{
  my(r0 = 7078137);
  foreach([6778137, 7078137, 8378137, 42164137, 6378138], r,
    my(bs = f64(asin(Re / r)), delta = 8 * u * tan(bs), bl);
    bl = [0, f64(1/10), f64(1/2), 1, f64(-7/10), f64(bs * 9/10), f64(bs - 10^-3), f64(bs - 10^-6),
          f64(bs - 10^-9), f64(bs - 10^-12), bs, f64(bs + 10^-12), f64(bs + 10^-6), f64(3/2)];
    foreach(bl, b,
      my(a = abs(b), f = fcyl_acos(r, a), lo = fcyl_acos(r, a + delta), hi = fcyl_acos(r, max(a - delta, 0)));
      listput(rows, Strprintf("    (%s, %s, %s, %s, %s),", lit(r), lit(b), lit(f64(lo)), lit(f64(f)), lit(f64(hi))))));
}
crows = List();
{
  foreach([7078137, 42164137], r,
    foreach([au, 147100000000, 152100000000], D,
      my(ph = onsets(r, D), bs = asin(Re / r), bl);
      bl = [0, f64(1/2), f64(ph[1] - 10^-4), f64((ph[1] + bs) / 2), f64(bs), f64((bs + ph[2]) / 2), f64(ph[2] + 10^-4)];
      foreach(bl, b,
        my(um = froot(gumbra, r, D, b), pe = froot(gpen, r, D, b), me = mean_lost(r, D, b));
        listput(crows, Strprintf("    (%s, %s, %s, %s, %s, %s),", lit(r), lit(D), lit(b), lit(f64(um)), lit(f64(pe)), lit(f64(me)))))));
}
print("cylinder rows (r, beta, f(|beta| + delta), f(beta), f(|beta| - delta)):");
foreach(rows, s, print(s));
print("cone rows (r, D, beta, umbra, penumbra, mean of 1 - nu):");
foreach(crows, s, print(s));
{
  my(file = "crates/spacelab-propagation/tests/eclipse.rs", text, missing = 0);
  text = iferr(readstr(Str("../", file)), E, iferr(readstr(file), E2, []));
  my(have = Map());
  foreach(text, line, mapput(have, line, 1));
  foreach(rows, s, if(!mapisdefined(have, s), missing++));
  foreach(crows, s, if(!mapisdefined(have, s), missing++));
  chk("every_row_is_in_tests_eclipse_rs_verbatim", missing, 1/2);
}
