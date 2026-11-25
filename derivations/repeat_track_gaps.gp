\\ Second engine for repeat_track_gaps.mac: the gap sets, the sub-cycles and the swath rule of
\\ `crates/spacelab-constellation/src/repeat_track.rs`, by methods that share nothing with it.
\\
\\ (a) The crossings from the rates: the j-th equator crossing at time j T/2 lies j pi (1 - kappa)
\\     east of the first, kappa = Nd/Np; in half-slots pi/Np that is j (Np - Nd).
\\ (b) G3, the three-distance property and its multiplicities: for every Np <= 120, every Nd < 2 Np
\\     prime to Np, the crossings are inserted one at a time into a circle of cells and the gap
\\     multiset is kept by counting; after every insertion it equals the code's formula (N - u gaps
\\     of A, N - v of B, u + v - N of A + B), it has at most three lengths, and with three the
\\     largest is the sum of the others. Ascending nodes only, and both kinds in time order.
\\ (c) The sub-cycles: the record returns of the ascending node, by their definition, are the
\\     distinct denominators of the convergents of Nd/Np, for every Np <= 300.
\\ (d) The swath rule at 60 digits: the distance from an equatorial point to the track's tangent
\\     plane by vectors, against asin(sin x sin i'); sin i' from the Earth-fixed velocity at both
\\     nodes by a 60-digit difference quotient; the planar limit as R grows.
\\ (e) The rows `tests/repeat_track.rs` holds the code to, decided here in exact arithmetic on the
\\     doubles the test passes, each with its margin from the boundary; and the check that each row
\\     is in that file verbatim.
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/repeat_track_gaps.gp </dev/null
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

\\ ---- (a) The crossings from the rates ----
{
  my(bad = 0);
  for(Np = 1, 40, for(Nd = 1, 2 * Np - 1, if(gcd(Np, Nd) == 1,
    my(kappa = Nd / Np);
    for(j = 0, 4 * Np,
      \\ east longitude of the j-th crossing: the orbit carries it j pi, the Earth turns j pi kappa under it
      my(east = j * Pi * (1 - kappa), half = (east / (Pi / Np)));
      if(abs(half - j * (Np - Nd)) > 1e-50, bad++);
      if(j % 2 == 0 && (round(half) - (-(j / 2) * 2 * Nd)) % (2 * Np) != 0, bad++)))));
  chk("crossing_j_is_j_times_np_minus_nd_half_slots_east", bad, 1/2);
}

\\ ---- (b) G3: the gap multiset after every insertion, against the formula ----
\\ Insert r_j = j a mod q for j = 1 .. M - 1 (stopping at a repeat); after each, compare.
gapcheck(a, q, M) = {
  my(occ = vector(q), cnt = vector(q), distinct = 1, bad = 0, three = 0, n = 1);
  my(u = 0, ru = q, v = 0, rv = -1, p, l, r, gl, g1, g2, la);
  occ[1] = 1; cnt[q] = 1;
  for(j = 1, M - 1,
    p = (j * a) % q;
    if(occ[p + 1], break);
    l = (p - 1 + q) % q; while(!occ[l + 1], l = (l - 1 + q) % q);
    r = (p + 1) % q; while(!occ[r + 1], r = (r + 1) % q);
    gl = (r - l + q) % q; if(gl == 0, gl = q);
    g1 = (p - l + q) % q; g2 = (r - p + q) % q;
    cnt[gl]--; if(cnt[gl] == 0, distinct--);
    if(cnt[g1] == 0, distinct++); cnt[g1]++;
    if(cnt[g2] == 0, distinct++); cnt[g2]++;
    occ[p + 1] = 1; n++;
    if(p < ru, ru = p; u = j); if(p > rv, rv = p; v = j);
    la = select(x -> x[2] > 0, [[ru, n - u], [q - rv, n - v], [ru + q - rv, u + v - n]]);
    if(la[1][1] == la[2][1], la = concat([[la[1][1], la[1][2] + la[2][2]]], la[3..#la]));
    if(#la != distinct || #select(x -> cnt[x[1]] != x[2], la) > 0, bad++);
    if(distinct > 3, three++);
    if(distinct == 3, my(ls = vecsort(apply(x -> x[1], la))); if(ls[3] != ls[1] + ls[2], three++)));
  [bad, three, n - 1];
}
{
  my(sets = 0, inserted = 0, bad = 0, three = 0, res);
  for(Np = 1, 120,
    for(Nd = 1, 2 * Np - 1, if(gcd(Np, Nd) == 1,
      my(s = (Np - Nd) % (2 * Np), c = gcd(s, 2 * Np));
      sets++;
      res = gapcheck((-Nd) % Np, Np, Np); bad += res[1]; three += res[2]; inserted += res[3];
      res = gapcheck(s / c, 2 * Np / c, 2 * Np); bad += res[1]; three += res[2]; inserted += res[3])));
  printf("(b) %d pairs Np/Nd, %d gap sets after an insertion\n", sets, inserted);
  chk("every_gap_multiset_is_the_formula", bad, 1/2);
  chk("at_most_three_gaps_the_largest_the_sum_of_the_others", three, 1/2);
}

\\ ---- (c) The sub-cycles ----
records(Np, Nd) = {
  my(best = Np, out = List(), o);
  for(k = 1, Np, o = centerlift(Mod(-k * Nd, Np)); if(abs(o) < best, best = abs(o); listput(out, [k, o])));
  Vec(out);
}
convergent_denominators(Np, Nd) = {
  my(cf = contfrac(Nd / Np), qs = List());
  for(m = 1, #cf, listput(qs, contfracpnqn(cf[1..m])[2, 1]));
  Set(Vec(qs));
}
{
  my(bad = 0, pairs = 0);
  for(Np = 1, 300, for(Nd = 1, 2 * Np - 1, if(gcd(Np, Nd) == 1,
    pairs++;
    if(Set(apply(x -> x[1], records(Np, Nd))) != convergent_denominators(Np, Nd), bad++))));
  printf("(c) %d pairs\n", pairs);
  chk("sub_cycles_are_the_convergent_denominators", bad, 1/2);
}

\\ ---- (d) The swath rule at 60 digits ----
\\ The track's tangent plane at the node through the equatorial point (1, 0, 0), inclination ip.
\\ The sub-point in the frame turning with the Earth relative to the node, u the argument of latitude:
\\ R3(-kappa u) applied to the inertial sub-point.
subpoint(i, kappa, u) = my(x = cos(u), y = cos(i) * sin(u), z = sin(i) * sin(u), t = -kappa * u); [x * cos(t) - y * sin(t), x * sin(t) + y * cos(t), z];
{
  my(worst = 0, worst2 = 0, nodes = 0);
  foreach([1/10, 7/10, 3/2, 2, 3], ip,
    foreach([1/1000, 1/10, 1, 3/2], x,
      my(P = [cos(x), sin(x), 0], N = [0, -sin(ip), cos(ip)], d);
      d = asin(abs(P * N~));
      worst = max(worst, abs(d - asin(sin(x) * sin(ip))))));
  chk("distance_to_the_tangent_plane_is_asin_sin_x_sin_ip", worst, 1e-55);
  \\ the Earth-fixed velocity at both nodes: position R3(-(kappa) u) applied to the inertial sub-point
  foreach([[98 * Pi / 180, 16 / 233], [51 * Pi / 180, 1 / 15], [120 * Pi / 180, 2 / 29], [5 * Pi / 180, 1 / 2]], ik,
    my(i = ik[1], kappa = ik[2], h = 10^-20, vel, sinip);
    foreach([0, Pi], u0,
      vel = (subpoint(i, kappa, u0 + h) - subpoint(i, kappa, u0 - h)) / (2 * h);
      \\ at the node the east unit vector is z x r and north is z
      my(r = subpoint(i, kappa, u0), east = [-r[2], r[1], 0], north = vel[3], ve = vel * east~);
      sinip = abs(north) / sqrt(north^2 + ve^2);
      nodes++;
      worst2 = max(worst2, abs(sinip - sin(i) / sqrt(sin(i)^2 + (cos(i) - kappa)^2)))));
  chk("sin_ip_from_the_earth_fixed_velocity_at_both_nodes", worst2, 1e-30);
  chk("eight_nodes_were_differenced", nodes - 8, 1/2);
  \\ the planar limit: 2 R asin(sin(W/2R)/sin ip) -> W/sin ip
  my(W = 185000, ip = 1.3, lim = W / sin(ip), dev = []);
  foreach([10^7, 10^9, 10^11], R, dev = concat(dev, abs(2 * R * asin(sin(W / (2 * R)) / sin(ip)) - lim) * R^2));
  \\ the deviation times R^2 tends to a constant: the next term is O(1/R^2)
  chk("planar_limit_is_w_over_sin_ip_with_an_r_minus_two_term", abs(dev[3] - dev[2]) / dev[3], 1e-6);
}

\\ ---- (d') What the tangent circle leaves out ----
\\ The true track curves away from its tangent circle as the Earth turns under it. The distance from
\\ the equatorial point at the swath's edge x_e to the true track (a 60-digit minimum over u) against
\\ the rule's asin(sin x_e sin i'): limitations.md quotes these four.
\\ minus the cosine of the distance from (cos xe, sin xe, 0) to the track point at u
negdot(i, kap, xe, u) = my(t = -kap * u, a = cos(u), b = cos(i) * sin(u)); -(cos(xe) * (a * cos(t) - b * sin(t)) + sin(xe) * (a * sin(t) + b * cos(t)));
{
  my(worst = 0);
  foreach([[98.2, 16/233, 185000], [98.62, 10/143, 290000], [51.6, 1/15, 2000000], [98.65, 27/385, 1270000]], c,
    my(i = c[1] * Pi / 180, kap = c[2], R = 6378137, W = c[3], sip = sin(i) / sqrt(sin(i)^2 + (cos(i) - kap)^2));
    my(xe = asin(sin(W / (2 * R)) / sip), lo = -1/2, hi = 1/2, g = (sqrt(5) - 1) / 2, m1, m2, dt, rel);
    for(k = 1, 220, m1 = hi - g * (hi - lo); m2 = lo + g * (hi - lo); if(negdot(i, kap, xe, m1) < negdot(i, kap, xe, m2), hi = m2, lo = m1));
    dt = acos(-negdot(i, kap, xe, (lo + hi) / 2)); rel = dt / (W / (2 * R)) - 1;
    worst = max(worst, abs(rel));
    printf("(d') i %.2f deg, %d/%d, W %d km: the true track is %.3e (relative) from the rule's edge\n", c[1], denominator(kap), numerator(kap), W / 1000, rel));
  chk("the_tangent_circle_is_within_2e-4_of_the_track_at_these_edges", worst > 2e-4, 1/2);
}

\\ ---- (e) The rows ----
\\ Exact gap sets by sorting (not the formula), in half-slots.
sorted_gaps(Np, Nd, n, both) = {
  my(pts);
  if(both, pts = Set(vector(2 * n, j, ((j - 1) * (Np - Nd)) % (2 * Np))),
           pts = Set(vector(n, k, ((k - 1) * (-2 * Nd)) % (2 * Np))));
  my(m = #pts, g, out = List(), c = 1);
  if(m == 1, return([[2 * Np, 1]]));
  g = vecsort(vector(m, k, if(k < m, pts[k + 1] - pts[k], 2 * Np - pts[m] + pts[1])));
  for(k = 2, m + 1, if(k <= m && g[k] == g[k - 1], c++, listput(out, [g[k - 1], c]); c = 1));
  Vec(out);
}
gaprow(Np, Nd, n, both) = {
  my(gs = sorted_gaps(Np, Nd, n, both), s = "");
  for(k = 1, #gs, s = Str(s, if(k > 1, ", ", ""), "(", gs[k][1], ", ", gs[k][2], ")"));
  Strprintf("    (%d, %d, %d, %s, &[%s]),", Np, Nd, n, if(both, "Nodes::Both", "Nodes::Ascending"), s);
}
subrow(Np, Nd) = {
  my(rs = records(Np, Nd), s = "");
  for(k = 1, #rs, s = Str(s, if(k > 1, ", ", ""), "(", rs[k][1], ", ", rs[k][2], ")"));
  Strprintf("    (%d, %d, &[%s]),", Np, Nd, s);
}
\\ The exact decision: half the largest gap x_g = G pi / (2 Np), covered iff sin(x_g) sin i' <= sin(W/2R),
\\ kappa = Nd/Np exactly, i the double the test passes. Returns the relative margin rhs/lhs - 1.
margin(Np, Nd, i, W, R, G) = {
  my(kappa = Nd / Np, sip = sin(i) / sqrt(sin(i)^2 + (cos(i) - kappa)^2), xg = G * Pi / (2 * Np));
  if(G > Np, return(-1));
  sin(W / (2 * R)) / (sin(xg) * sip) - 1;
}
largest(Np, Nd, n, both) = vecmax(apply(x -> x[1], sorted_gaps(Np, Nd, n, both)));
first_cover(Np, Nd, i, W, R, both) = {
  for(n = 1, Np, my(m = margin(Np, Nd, i, W, R, largest(Np, Nd, n, both)));
    if(m >= 0, return([n, m, if(n > 1, margin(Np, Nd, i, W, R, largest(Np, Nd, n - 1, both)), -1)])));
  [0, 0, 0];
}
subrows = List(); gaprows = List(); swathrows = List(); edgerows = List(); widthrows = List();
foreach([[233, 16], [143, 10], [175, 12], [385, 27], [15, 1], [29, 2]], p, listput(subrows, subrow(p[1], p[2])));
\\ Sentinel-2 after 14 revolutions (M0-10: {7, 13}); Landsat at its sub-cycles; a pair with both odd.
{
  foreach([[143, 10, 14], [143, 10, 43], [233, 16, 14], [233, 16, 15], [233, 16, 29], [233, 16, 102], [233, 16, 233],
           [175, 12, 100], [385, 27, 200], [15, 1, 7], [15, 1, 15], [43, 3, 20], [1, 1, 1], [1, 2, 1], [2, 1, 2]], p,
    listput(gaprows, gaprow(p[1], p[2], p[3], 0));
    listput(gaprows, gaprow(p[1], p[2], p[3], 1)));
}
\\ Swaths: (Np, Nd, inclination in degrees, W metres), each node kind; R the WGS 84 equatorial radius.
{
  my(R = 6378137, worst_margin = 1);
  foreach([[233, 16, 98.2, 185000], [143, 10, 98.62, 290000], [175, 12, 98.18, 250000], [385, 27, 98.65, 1270000],
           [15, 1, 51.6, 2000000], [43, 3, 65, 900000], [233, 16, 98.2, 20000]], c,
    my(i = f64(c[3] * Pi / 180), W = c[4]);
    foreach([0, 1], both,
      my(fc = first_cover(c[1], c[2], i, W, R, both));
      if(fc[1] > 0, worst_margin = min(worst_margin, min(fc[2], -fc[3])));
      listput(swathrows, Strprintf("    (%d, %d, %s, %s, %s, %s, %d),", c[1], c[2], if(both, "Nodes::Both", "Nodes::Ascending"),
        lit(i), lit(R), lit(W), fc[1]))));
  printf("(e) the smallest relative margin of a first covering revolution from the boundary: %.2e\n", worst_margin);
  chk("every_swath_row_is_decided_away_from_its_boundary", worst_margin < 1e-9, 1/2);
  \\ Boundary rows: W* = 2 R asin(sin(x_g) sin i') at revolution n; the test needs the decision open
  \\ below W*, open at the double nearest W* (inside the allowance, the safe side), closed above.
  foreach([[233, 16, 98.2, 102, 0], [233, 16, 98.2, 233, 1], [143, 10, 98.62, 43, 1], [15, 1, 51.6, 15, 0]], c,
    my(Np = c[1], Nd = c[2], i = f64(c[3] * Pi / 180), n = c[4], both = c[5], G = largest(Np, Nd, n, both));
    my(sip = sin(i) / sqrt(sin(i)^2 + (cos(i) - Nd / Np)^2), Wstar = 2 * R * asin(sin(G * Pi / (2 * Np)) * sip));
    my(lo = f64(Wstar * (1 - 10^-12)), mid = f64(Wstar), hi = f64(Wstar * (1 + 10^-12)));
    if(margin(Np, Nd, i, lo, R, G) >= 0 || margin(Np, Nd, i, hi, R, G) <= 0, error("boundary row ", c));
    listput(edgerows, Strprintf("    (%d, %d, %s, %s, %s, %d, %s, %s, %s),", Np, Nd, if(both, "Nodes::Both", "Nodes::Ascending"),
      lit(i), lit(R), n, lit(lo), lit(mid), lit(hi))));
}
\\ The equator one crossing's swath covers, 2 R asin(sin(W/2R)/sin i'), at 60 digits; R = 10^12 is
\\ the planar limit W/sin i' to O((W/R)^2).
{
  foreach([[233, 16, 98.2, 6378137, 185000], [143, 10, 98.62, 6378137, 290000], [15, 1, 51.6, 6378137, 2000000],
           [43, 3, 65, 6378137, 900000], [233, 16, 98.2, 10^12, 185000], [29, 2, 120, 6378137, 3000000]], c,
    my(Np = c[1], Nd = c[2], i = f64(c[3] * Pi / 180), R = c[4], W = c[5]);
    my(sip = sin(i) / sqrt(sin(i)^2 + (cos(i) - Nd / Np)^2), w = 2 * R * asin(sin(W / (2 * R)) / sip));
    listput(widthrows, Strprintf("    (%d, %d, %s, %s, %s, %s),", Np, Nd, lit(i), lit(R), lit(W), lit(f64(w)))));
}
print("sub-cycle rows (Np, Nd, [(revolution, offset slots east)]):"); foreach(subrows, s, print(s));
print("gap rows (Np, Nd, revolutions, nodes, [(length half-slots, count)]):"); foreach(gaprows, s, print(s));
print("swath rows (Np, Nd, nodes, i, R, W, first covering revolution or 0):"); foreach(swathrows, s, print(s));
print("boundary rows (Np, Nd, nodes, i, R, n, W below, W nearest the boundary, W above):"); foreach(edgerows, s, print(s));
print("width rows (Np, Nd, i, R, W, the equator one crossing's swath covers):"); foreach(widthrows, s, print(s));
{
  my(file = "crates/spacelab-constellation/tests/repeat_track.rs", text, missing = 0, have = Map());
  text = iferr(readstr(Str("../", file)), E, iferr(readstr(file), E2, []));
  foreach(text, line, mapput(have, line, 1));
  foreach([subrows, gaprows, swathrows, edgerows, widthrows], rs, foreach(rs, s, if(!mapisdefined(have, s), missing++)));
  chk("every_row_is_in_tests_repeat_track_rs_verbatim", missing, 1/2);
}
