\\ S2 of plan 18 Ф28.V: each enclosure the shipped f64 code returns must contain the value at 60
\\ digits. Read before the lines `crates/spacelab-constellation/examples/enclosure_s2.rs` prints:
\\
\\   K([[x, y, z, rho], ...], [lo_1, ...], [hi_1, ...]);   kfold_areas, every level
\\   A(k, [[x, y, z], ...], lo, hi);                         covering_radius
\\   S(...);                                                 a site ball (analysis.rs `site_lines`)
\\
\\ every double as the hex of its bits. The 60-digit values are kfold() of kfold_coverage_area.gp
\\ and alpha() of analysis_alpha_exact.gp on the same doubles; a site's exact direction is the
\\ elements' closed form with Kepler's equation solved at 60 digits. A miss is a value outside its
\\ interval by more than 1e-40, far above those algorithms' own error at these inputs. The last
\\ call, report(), prints one line per kind: the count, the misses, the tightest margin (the
\\ smallest distance from a value to the nearer end) and the widest interval. Containment on these
\\ inputs is evidence about how the proof reads the code; it is not the proof.
\\
\\ Run: { cat derivations/enclosure_s2.gp; <the example's lines>; echo 'report();'; } | gp -q
default(realprecision, 60);
default(path, "derivations:.");
read("binary64.gp");
read("kfold_coverage_area.gp");
read("analysis_alpha_exact.gp");
b64(h) = binary_bits(h, 53, 11);
s2 = Map();
\\ Per kind: checked, missed, tightest margin, widest, tightest margin above 0, ends hit exactly.
s2_add(kind, miss, margin, width) = {
  my(r = if(mapisdefined(s2, kind), mapget(s2, kind), [0, 0, 10^9, 0, 10^9, 0]));
  r[1]++; r[2] += miss; r[3] = min(r[3], margin); r[4] = max(r[4], width);
  if(margin > 0, r[5] = min(r[5], margin), if(margin == 0, r[6]++));
  mapput(s2, kind, r);
}
\\ One interval against one value: [miss, margin, width].
s2_one(v, lo, hi) = [v < lo - 1e-40 || v > hi + 1e-40, min(v - lo, hi - v), hi - lo];
K(caps, lo, hi) = {
  my(c = apply(q -> [[b64(q[1]), b64(q[2]), b64(q[3])], b64(q[4])], caps), A = kfold(c), miss = 0);
  for(k = 1, #A, my(r = s2_one(A[k], b64(lo[k]), b64(hi[k])));
    miss += r[1]; s2_add("kfold level", r[1], r[2], r[3]));
  if(miss, printf("S2 MISS kfold %s\n", caps));
}
A(k, sites, lo, hi) = {
  my(S = apply(q -> nrm([b64(q[1]), b64(q[2]), b64(q[3])]), sites), v, r);
  v = if(k == #S, Pi, alpha(S, k));
  r = s2_one(v, b64(lo), b64(hi));
  s2_add("alpha", r[1], r[2], r[3]);
  if(r[1], printf("S2 MISS alpha k=%d %s\n", k, sites));
}
\\ A site: elements a, e, i, raan, argp, nu0, mu and t as doubles, or the exact Walker slot
\\ [P, S, F, p, s, pattern] in place of raan and nu0 (argp 0), then the site and its radius.
kepler_E(M, e) = my(E = if(e < 0.8, M, Pi)); for(j = 1, 200, E = E - (E - e*sin(E) - M)/(1 - e*cos(E))); E;
dir(raan, inc, u) = [cos(raan)*cos(u) - sin(raan)*cos(inc)*sin(u), sin(raan)*cos(u) + cos(raan)*cos(inc)*sin(u), sin(inc)*sin(u)];
S(el, slot, t, site, radius) = {
  my(a = b64(el[1]), e = b64(el[2]), inc = b64(el[3]), raan, u, n, D, gap);
  t = b64(t); n = sqrt(b64(el[7]) / a^3);
  if(#slot,
    my(P = slot[1], Sp = slot[2], F = slot[3], p = slot[4], s = slot[5], span = if(slot[6], Pi, 2*Pi));
    raan = span * p / P; u = 2*Pi*s/Sp + 2*Pi*F*p/(P*Sp) + n*t,
    my(nu0 = b64(el[6]), E0, M, E, nu);
    raan = b64(el[4]);
    E0 = arg((e + cos(nu0)) + I*sqrt(1 - e^2)*sin(nu0));
    M = E0 - e*sin(E0) + n*t; M = M - 2*Pi*floor(M/(2*Pi));
    E = kepler_E(M, e);
    nu = arg((cos(E) - e) + I*sqrt(1 - e^2)*sin(E));
    u = b64(el[5]) + nu);
  D = dir(raan, inc, u);
  gap = sqrt(norml2(apply(b64, site) - D));
  s2_add("site", gap > b64(radius) + 1e-40, b64(radius) - gap, b64(radius));
  if(gap > b64(radius) + 1e-40, printf("S2 MISS site %s %s %s\n", el, slot, t));
}
report() = {
  foreach(["kfold level", "alpha", "site"], kind, if(mapisdefined(s2, kind),
    my(r = mapget(s2, kind));
    printf("S2 %s: %d checked, %d missed, tightest margin %.3e (above 0: %.3e; %d ends hit exactly), widest %.3e\n",
           kind, r[1], r[2], r[3], r[5], r[6], r[4])));
}
