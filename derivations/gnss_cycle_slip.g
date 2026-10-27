# Least-squares polynomial predictions in exact rationals: the second engine for
# derivations/gnss_cycle_slip.mac and the values the gate in crates/spacelab-gnss/src/cycle_slip.rs
# asserts the shipped f64 `predict` against.
#
# GAP solves the normal equations (V^T V) a = V^T y over the rationals in raw time -- no shift, no
# scale -- so agreement with Maxima's shifted-and-scaled rationals is the invariance claim as well.
# Every mismatch prints `MISMATCH`; the summary counts them.
#
#     timeout 300 gap -q -A --quitonbreak derivations/gnss_cycle_slip.g

predict := function(ts, ys, p, tstar)
  local V, a;
  V := List(ts, t -> List([0 .. p], k -> t^k));
  a := SolutionMat(TransposedMat(TransposedMat(V) * V), TransposedMat(V) * ys);
  return Sum([0 .. p], k -> a[k + 1] * tstar^k);
end;

ys := [1/4, -9/8, 11/4, 3/8, 21/4, -7/8];
uniform := [0, 30, 60, 90, 120, 150];
irregular := [0, 25, 55, 90, 130, 175];
cases := [
  ["uniform order 2, t* = 180", uniform, 2, 180, -43/40],
  ["irregular order 2, t* = 230", irregular, 2, 230, -634102487/154086672],
  ["irregular order 1, t* = 230", irregular, 1, 230, 92791/41608],
  ["irregular order 2, t* = 70", irregular, 2, 70, 119020787/51362224],
];
bad := 0;
for c in cases do
  got := predict(c[2], ys, c[3], c[4]);
  if got <> c[5] then
    bad := bad + 1;
    Print("MISMATCH ", c[1], ": ", got, " vs Maxima ", c[5], "\n");
  else
    Print(c[1], ": ", got, "\n");
  fi;
od;

# The blind pair of the geometry-free test: 77 lambda1 = 60 lambda2 with lambda = c/f.
if 77 / (154 * 10230000) <> 60 / (120 * 10230000) then
  bad := bad + 1;
  Print("MISMATCH the (77, 60) slip moves L1 - L2\n");
fi;
Print("gnss cycle slip: ", bad, " mismatches\n");
QUIT;
