# Second engine for derivations/metrics_periodic_gaps.mac: the periodic access pattern of
# crates/spacelab-coverage/tests/metrics.rs, enumerated window by window in exact rationals, its
# metrics computed from their definitions (not from the closed forms), against Maxima's values.
P := 5400;; d := 600;; a := 1000;; N := 16;; T := 86400;;
rises := List([0 .. N - 1], i -> a + i * P);;
windows := List(rises, r -> [r, r + d]);;
gaps := Concatenation([windows[1][1] - 0],
                      List([2 .. N], i -> windows[i][1] - windows[i - 1][2]),
                      [T - windows[N][2]]);;
covered := Sum(windows, w -> w[2] - w[1]);;
# Mean response from its definition: the integral of the wait b - t over each gap [a, b], done as
# the exact antiderivative b t - t^2/2 between the gap's ends.
starts := Concatenation([0], List(windows, w -> w[2]));;
ends := Concatenation(List(windows, w -> w[1]), [T]);;
wait := Sum([1 .. N + 1], i -> (ends[i] * ends[i] - ends[i]^2 / 2)
                              - (ends[i] * starts[i] - starts[i]^2 / 2));;
checks := [
  ["trailing gap", gaps[N + 1], 3800],
  ["coverage fraction", covered / T, 1/9],
  ["max gap", Maximum(gaps), 4800],
  ["mean gap", Sum(gaps) / Length(gaps), 76800/17],
  ["mean response", wait / T, 112825/54],
  ["mean response = sum gap^2/(2T)", Sum(gaps, g -> g^2) / (2 * T), 112825/54]
];;
for c in checks do
  Print(c[1], ": ", c[2], "  ", c[2] = c[3], "\n");
od;
QUIT;
