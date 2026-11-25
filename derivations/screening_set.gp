\\ The synthetic all-against-all screening set of plan 18 Ф21.O (D-47): its generator and the second
\\ engine for screening_set.mac.
\\
\\ Every object is a circular two-body orbit, so every pair the radial bound cannot separate has its
\\ closest approaches in closed form:
\\   coplanar, radii r_a != r_b: |dr|^2 = r_a^2 + r_b^2 - 2 r_a r_b cos(du), du = u_b - u_a, so the
\\     minima are du = 2 pi k at distance |r_b - r_a|, t_k = (2 pi k - du_0)/(n_b - n_a);
\\   one radius r, planes crossing at gamma: with theta measured from the mutual node l = h_a x h_b,
\\     |dr|^2 = 2 r^2 (1 - A cos(delta) - B cos(2 theta_a + delta)), A = (1 + cos gamma)/2,
\\     B = (1 - cos gamma)/2, delta = theta_b - theta_a, so the minima are 2 theta_a + delta = 2 pi k,
\\     two per revolution, at distance 2 r cos(gamma/2) |sin(delta/2)|;
\\   one plane and one radius: the separation is constant, 2 r |sin(du_0/2)|, and has no minimum.
\\ Every other pair has radii further apart than the threshold, and |x_a - x_b| >= |r_a - r_b|.
\\
\\ This file (1) builds the catalogue from the design below, each angle rounded to the 2^-32 rad grid
\\ the fixture stores, (2) computes every key from those stored values in the closed forms above at
\\ 80 digits, (3) checks the keys by a second route on the Cartesian motion: every listed minimum is
\\ a negative-to-positive root of dr . dv with |dr| equal to its distance there, and on every planted
\\ pair and every fifth shell pair a 60 s scan of dr . dv polished with solve() finds exactly the
\\ listed roots, and (4) compares the result with the committed fixture line by line, or writes it.
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\
\\ Run from derivations/:  timeout 300 gp -q -f screening_set.gp </dev/null
\\ To rewrite the fixture: SCREENING_SET_WRITE=1 timeout 300 gp -q -f screening_set.gp </dev/null

default(realprecision, 80);
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, Strprintf("0   (|residual| %.1e)", abs(e)), Str("FAIL ", e)));

MU = 398600441800000;           \\ m^3/s^2, the value spacelab-estimation's screening tests use
W = 86400;                      \\ the window [0, W], s
D = 10000;                      \\ the screening threshold, m
UNIT = 2^-32;                   \\ the angle grid, rad
FILE = "crates/spacelab-testkit/tests/fixtures/screening_set.txt";

dt(a, b) = a * b~;
crs(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
atan2(y, x) = arg(x + I * y);
deg(x) = x * Pi / 180;
\\ An angle on the grid, reduced to [0, 2 pi): an integer number of units.
grid(x) = round((x - 2 * Pi * floor(x / (2 * Pi))) / UNIT);

\\ An object: [group, r, i, raan, u0, planted], r in metres and each angle an integer number of UNITs;
\\ `planted` numbers the crossing and coplanar pairs, 0 elsewhere, and is not written.
P(o) = my(O = o[4] * UNIT); [cos(O), sin(O), 0];
Q(o) = my(O = o[4] * UNIT, i = o[3] * UNIT); [-sin(O) * cos(i), cos(O) * cos(i), sin(i)];
H(o) = my(O = o[4] * UNIT, i = o[3] * UNIT); [sin(O) * sin(i), -cos(O) * sin(i), cos(i)];
mm(o) = sqrt(MU / o[2]^3);
\\ The motion, from an object's stored values: [r, n, p, q, u0], built once per object as M[k].
prep(o) = [o[2], mm(o), P(o), Q(o), o[5] * UNIT];
pos(m, t) = my(u = m[5] + m[2] * t); m[1] * (m[3] * cos(u) + m[4] * sin(u));
vel(m, t) = my(u = m[5] + m[2] * t); m[1] * m[2] * (-m[3] * sin(u) + m[4] * cos(u));
rate(a, b, t) = dt(pos(M[b], t) - pos(M[a], t), vel(M[b], t) - vel(M[a], t));
sep(a, b, t) = my(d = pos(M[b], t) - pos(M[a], t)); sqrt(dt(d, d));

\\ ---- (1) the design ----
objs = List();
\\ The shell: Walker 40/5/0 at 550 km, 55 deg, the first plane at RAAN 15 deg, a 10 deg common phase.
{
  my(T = 40, Pl = 5, F = 0, S = T / Pl);
  for (p = 0, Pl - 1, for (s = 0, S - 1,
    listput(objs, ["shell", 6928000, grid(deg(55)), grid(deg(15) + 2 * Pi * p / Pl),
                   grid(deg(10) + 2 * Pi * s / S + 2 * Pi * F * p / T), 0])));
}
\\ Crossing pairs, one radius each: [target distance m, i_a, raan_a, i_b, raan_b (deg), first TCA s].
\\ Six below the threshold, five at or beyond it (the hard negatives).
{CROSSING = [[10, 51.6, 30, 97.8, 120, 1000], [100, 53, 200, 53, 260, 1700],
            [1000, 98, 10, 82, 100, 2300], [5000, 20, 50, 160, 60, 600],
            [9999, 45, 300, 46, 301, 1400], [9999.99, 72, 220, 30, 35, 2000],
            [10000.01, 70, 100, 120, 340, 900], [10001, 30, 0, 60, 90, 1200],
            [10100, 87, 45, 40, 170, 2600], [12000, 63.4, 150, 63.4, 330, 500],
            [50000, 85, 80, 89, 260, 1900]];}
{
  for (q = 1, #CROSSING,
    my(c = CROSSING[q], r = 7000000 + 25000 * (q - 1), a, b, h, l, sa, sb, g, delta, n, t0 = c[6]);
    a = ["crossing", r, grid(deg(c[2])), grid(deg(c[3])), 0, q];
    b = ["crossing", r, grid(deg(c[4])), grid(deg(c[5])), 0, q];
    l = crs(H(a), H(b)); l = l / sqrt(dt(l, l));
    sa = atan2(dt(l, Q(a)), dt(l, P(a))); sb = atan2(dt(l, Q(b)), dt(l, P(b)));
    g = atan2(sqrt(dt(crs(H(a), H(b)), crs(H(a), H(b)))), dt(H(a), H(b)));
    delta = 2 * asin(c[1] / (2 * r * cos(g / 2)));
    n = mm(a);
    \\ at t0 theta_a = -delta/2 and theta_b = +delta/2: a minimum of the separation
    a[5] = grid(sa - delta / 2 - n * t0); b[5] = grid(sb + delta / 2 - n * t0);
    listput(objs, a); listput(objs, b));
}
\\ Coplanar pairs: [radius gap m, i, raan, u_a (deg), alignment time s]. Four below the threshold,
\\ two beyond it.
{COPLANAR = [[300, 28.5, 45, 10, 43000], [1000, 51.6, 135, 200, 30000], [5000, 97.5, 225, 80, 60000],
            [9999, 65, 315, 300, 20000], [10001, 40, 10, 150, 50000], [15000, 75, 100, 250, 70000]];}
{
  for (p = 1, #COPLANAR,
    my(c = COPLANAR[p], r = 7300000 + 40000 * (p - 1), a, b);
    a = ["coplanar", r, grid(deg(c[2])), grid(deg(c[3])), grid(deg(c[4])), #CROSSING + p];
    b = ["coplanar", r + c[1], a[3], a[4], 0, #CROSSING + p];
    b[5] = grid(a[5] * UNIT + (mm(a) - mm(b)) * c[5]);
    listput(objs, a); listput(objs, b));
}
\\ The background: 26 orbits, each at a radius of its own, 6 below the shell and 20 above the rest.
{
  for (j = 0, 25,
    my(r = if (j < 6, 6650000 + 40000 * j, 7560000 + 35000 * (j - 6)));
    listput(objs, ["background", r, grid(deg(5 + (37 * j) % 170)), grid(deg((97 * j) % 360)),
                   grid(deg((131 * j + 17) % 360)), 0]));
}
objs = Vec(objs);
N = #objs;
M = vector(N, k, prep(objs[k]));

\\ ---- (2) the keys ----
\\ The closed-form minima of a pair in (0, W) and its distance: [kind, dmin, [t_1, ...]].
keys(a, b) = {
  my(same = a[3] == b[3] && a[4] == b[4]);
  if (same && a[2] != b[2],
    my(na = mm(a), nb = mm(b), du0 = (b[5] - a[5]) * UNIT, k0, k1);
    \\ t_k = (2 pi k - du0)/(nb - na) in (0, W), whatever the sign of nb - na
    my(lo = du0 / (2 * Pi), hi = (du0 + (nb - na) * W) / (2 * Pi));
    k0 = floor(min(lo, hi)) + 1; k1 = ceil(max(lo, hi)) - 1;
    return(["coplanar", abs(b[2] - a[2]), vecsort(vector(max(0, k1 - k0 + 1), j, (2 * Pi * (k0 + j - 1) - du0) / (nb - na)))]));
  if (same,
    return(["comoving", 2 * a[2] * abs(sin((b[5] - a[5]) * UNIT / 2)), []]));
  if (a[2] != b[2], error("pair with radii ", a[2], " and ", b[2], " in distinct planes has no closed form"));
  my(r = a[2], n = mm(a), l = crs(H(a), H(b)), sa, sb, g, delta, th0, k0, k1);
  l = l / sqrt(dt(l, l));
  sa = atan2(dt(l, Q(a)), dt(l, P(a))); sb = atan2(dt(l, Q(b)), dt(l, P(b)));
  g = atan2(sqrt(dt(crs(H(a), H(b)), crs(H(a), H(b)))), dt(H(a), H(b)));
  delta = (b[5] * UNIT - sb) - (a[5] * UNIT - sa);
  th0 = a[5] * UNIT - sa;
  \\ minima at theta_a = pi k - delta/2, theta_a(t) = th0 + n t
  k0 = floor((th0 + delta / 2) / Pi) + 1; k1 = ceil((th0 + delta / 2 + n * W) / Pi) - 1;
  ["crossing", 2 * r * cos(g / 2) * abs(sin(delta / 2)), vector(max(0, k1 - k0 + 1), j, (Pi * (k0 + j - 1) - delta / 2 - th0) / n)];
}

lines = List();
pairs = List();     \\ [a, b, key, planted]
{
  for (a = 1, N, for (b = a + 1, N,
    my(planted = objs[a][6] && objs[a][6] == objs[b][6]);
    if (abs(objs[a][2] - objs[b][2]) > D && !planted, next);
    listput(pairs, [a, b, keys(objs[a], objs[b]), planted])));
}
fmt(x) = Strprintf("%.24f", x);
EV = 0; NEG = 0; SHELLMIN = 10^30; NEGMIN = 10^30; EVMAX = 0;
{
  my(header = List(), body = List(), mins = List());
  foreach (pairs, p, my(a = p[1], b = p[2], k = p[3]);
    listput(body, Str("pair ", a - 1, " ", b - 1, " ", k[1], " ", fmt(k[2])));
    if (!p[4], SHELLMIN = min(SHELLMIN, k[2]));
    foreach (k[3], t,
      if (k[2] < D, EV++; EVMAX = max(EVMAX, k[2]); listput(mins, Str("event ", a - 1, " ", b - 1, " ", fmt(t), " ", fmt(k[2]))),
        if (p[4], NEG++; NEGMIN = min(NEGMIN, k[2]); listput(mins, Str("negative ", a - 1, " ", b - 1, " ", fmt(t), " ", fmt(k[2])))))));
  foreach ([
    "# The synthetic all-against-all screening set of plan 18 Ф21.O (D-47), written by",
    "# derivations/screening_set.gp and re-derived by derivations/screening_set.mac: our own numbers.",
    "# Every object is a circular two-body orbit about a point mass `mu` (m^3/s^2): at time t (s),",
    "#   u = u0 + n t, n = sqrt(mu / r^3), p = (cos raan, sin raan, 0),",
    "#   q = (-sin raan cos i, cos raan cos i, sin i), position r (p cos u + q sin u) (m),",
    "#   velocity r n (-p sin u + q cos u) (m/s).",
    "# object k group r i raan u0: object k (0, 1, ...); r an integer number of metres; i, raan and u0",
    "#   integers in units of 2^angle_exponent rad, so every input is an exact binary64.",
    "# pair a b kind dmin: every pair whose radii differ by no more than the threshold, and each planted",
    "#   pair, with the least distance it ever reaches, m. kind: coplanar (one plane, two radii),",
    "#   crossing (one radius, two planes), comoving (one plane, one radius: the separation is constant).",
    "#   Every other pair has radii further apart than the threshold, and never comes closer than that.",
    "# event a b tca distance: every local minimum of the separation in the open window, over all",
    "#   pairs, at a distance below the threshold: the answer key.",
    "# negative a b tca distance: every other local minimum in the window of the planted pairs (groups",
    "#   crossing and coplanar): the hard negatives, at or beyond the threshold.",
    "# Keys are closed forms at 80 digits, printed to 24 decimals; at both ends of the window every",
    "#   pair with an event is further apart than the threshold."], h, listput(header, h));
  listput(header, Strprintf("# %d objects, %d pairs keyed, %d events, %d negatives.", N, #pairs, EV, NEG));
  listput(header, Str("mu ", MU));
  listput(header, Str("window 0 ", W));
  listput(header, Str("threshold ", D));
  listput(header, "angle_exponent -32");
  for (k = 1, N, my(o = objs[k]);
    listput(header, Str("object ", k - 1, " ", o[1], " ", o[2], " ", o[3], " ", o[4], " ", o[5])));
  lines = concat(concat(Vec(header), Vec(body)), Vec(mins));
}
{printf("%d objects, %d keyed pairs, %d events (largest %.3f m), %d negatives (smallest %.3f m), shell minimum %.3f m\n",
       N, #pairs, EV, EVMAX, NEG, NEGMIN, SHELLMIN);}

\\ ---- (3) the second route, on the Cartesian motion ----
\\ Every pair not keyed is radially separated: by construction above, restated over all pairs.
{
  my(bad = 0, keyed = Set(apply(p -> [p[1], p[2]], Vec(pairs))));
  for (a = 1, N, for (b = a + 1, N,
    if (!setsearch(keyed, [a, b]) && abs(objs[a][2] - objs[b][2]) <= D, bad++)));
  chk("every_unkeyed_pair_has_radii_further_apart_than_the_threshold", bad, 1/2);
}
\\ Every listed minimum: dr . dv changes sign from negative to positive within 1e-30 s of it, and the
\\ separation there is the pair's distance. The same for one closed-form minimum of every shell pair.
{
  my(eps = 10^-30, sgn = 0, worst = 0);
  foreach (pairs, p, my(a = p[1], b = p[2], k = p[3], ts = k[3]);
    if (k[1] == "comoving", next);
    if (#ts == 0, error("a non-constant pair with no minimum in the window"));
    if (!p[4] && k[2] >= D, ts = [ts[1]]);
    foreach (ts, t,
      if (!(rate(a, b, t - eps) < 0 && rate(a, b, t + eps) > 0), sgn++);
      worst = max(worst, abs(sep(a, b, t) - k[2]))));
  chk("every_closed_form_minimum_is_a_negative_to_positive_root_of_the_range_rate", sgn, 1/2);
  chk("the_separation_at_every_closed_form_minimum_is_its_distance", worst, 1e-40);
}
\\ A comoving pair: the range rate vanishes and the separation is its constant, at three epochs.
{
  my(worst = 0);
  foreach (pairs, p, my(a = p[1], b = p[2], k = p[3]);
    if (k[1] != "comoving", next);
    foreach ([0, W / 3, W], t, worst = max(worst, max(abs(sep(a, b, t) - k[2]), abs(rate(a, b, t)) / objs[a][2]))));
  chk("every_comoving_pair_keeps_its_distance", worst, 1e-60);
}
\\ No event pair is within the threshold at either end of the window.
{
  my(bad = 0);
  foreach (pairs, p, if (p[3][2] < D,
    if (sep(p[1], p[2], 0) <= D || sep(p[1], p[2], W) <= D, bad++)));
  chk("no_event_pair_is_within_the_threshold_at_either_end_of_the_window", bad, 1/2);
}
\\ The scan: on every planted pair and every fifth shell pair that is not comoving, the roots of the
\\ range rate a 60 s scan brackets, polished by solve(), are the closed-form minima.
{
  my(worst = 0, count = 0, j = 0);
  foreach (pairs, p, my(a = p[1], b = p[2], k = p[3], found = List(), t0 = 0, f0, t1, f1);
    if (k[1] == "comoving", next);
    j++;
    if (!p[4] && j % 5, next);
    f0 = rate(a, b, 0);
    while (t0 < W,
      t1 = min(t0 + 60, W); f1 = rate(a, b, t1);
      if (f0 < 0 && f1 >= 0, listput(found, solve(t = t0, t1, rate(a, b, t))));
      t0 = t1; f0 = f1);
    count++;
    if (#found != #k[3], error("pair ", p[1] - 1, " ", p[2] - 1, ": the scan finds ", #found, " minima, the closed form ", #k[3]));
    for (i = 1, #found, worst = max(worst, abs(found[i] - k[3][i]))));
  printf("scanned %d pairs\n", count);
  chk("every_scanned_root_of_the_range_rate_is_a_closed_form_minimum", worst, 1e-50);
}

\\ ---- (4) the fixture ----
{
  if (getenv("SCREENING_SET_WRITE"),
    my(h = fileopen(Str("../", FILE), "w"));
    foreach (lines, l, filewrite(h, l));
    fileclose(h);
    printf("wrote ../%s: %d lines\n", FILE, #lines),
    my(got = iferr(readstr(Str("../", FILE)), E, readstr(FILE)), diff = 0);
    if (#got != #lines, diff = abs(#got - #lines),
      for (i = 1, #lines, if (got[i] != lines[i], diff++)));
    chk("the_fixture_is_the_regenerated_set_line_for_line", diff, 1/2));
}
