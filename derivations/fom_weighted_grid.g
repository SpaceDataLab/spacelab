# Second engine for derivations/fom_weighted_grid.mac: the four-point grid of the fom.rs gate test,
# each point's windows enumerated in exact rationals, its metrics computed from their definitions,
# then aggregated as constellation_fom does, against Maxima's values.
T := 86400;;
Point := function(P, d, a, N)
  local windows, gaps;
  windows := List([0 .. N - 1], i -> [a + i * P, a + i * P + d]);
  gaps := Concatenation([windows[1][1]],
                        List([2 .. N], i -> windows[i][1] - windows[i - 1][2]),
                        [T - windows[N][2]]);
  return [Sum(windows, w -> w[2] - w[1]) / T, Maximum(gaps), Sum(gaps, g -> g^2) / (2 * T)];
end;;
pts := [Point(5400, 600, 1000, 16), Point(5400, 600, 0, 16), Point(86400, 86400, 0, 1),
        Point(10800, 900, 3000, 8)];;
w := [1, 2, 3, 4];;
WMean := k -> Sum([1 .. 4], i -> w[i] * pts[i][k]) / Sum(w);;
checks := [
  ["mean coverage", WMean(1), 11/30],
  ["mean max gap", WMean(2), 5400],
  ["mean response", WMean(3), 254315/108],
  ["min coverage", Minimum(List(pts, p -> p[1])), 1/12],
  ["worst max gap", Maximum(List(pts, p -> p[2])), 9900],
  ["worst response", Maximum(List(pts, p -> p[3])), 51575/12],
  ["fully covered points", Number(pts, p -> p[1] = 1), 1]
];;
for c in checks do
  Print(c[1], ": ", c[2], "  ", c[2] = c[3], "\n");
od;
QUIT;
