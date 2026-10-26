# The batch kernels of crates/spacelab-sensors/src/kernel.rs on an exact integer lattice: how many
# cells each predicate covers, counted in GAP over the integers, for the gate
# `the_kernels_decide_the_defining_inequalities_exactly_on_an_integer_lattice` in
# crates/spacelab-sensors/tests/kernel_gates.rs.
#
# Cells: every integer point of [-6, 6]^3, x outermost, z innermost; the zenith of cell (x, y, z)
# is the axis number ((x + y + z) mod 6) of [+x, -x, +y, -y, +z, -z]. Satellites (3, -2, 9),
# (-7, 5, 4), (8, 8, -8), all outside the cube. With d = S - C, t = d . zhat, q = d . d,
# r = S, n2 = r . r, u~ = n2 - C . r, and the boresight b = (0, 0, -1) with u = (C - S) . b:
#
#   horizon       eps >= 0         <=>  t >= 0
#   mask +30 deg  t >= q^(1/2)/2   <=>  t >= 0 and 4 t^2 >= q
#   mask -30 deg  t >= -q^(1/2)/2  <=>  t >= 0 or  4 t^2 <= q
#   nadir cone, cos(alpha) = 3/4  <=>  u~ >= 0 and 16 u~^2 >= 9 q n2
#   fixed cone, the same, about b  <=>  u >= 0 and 16 u^2 >= 9 q
#
# The right-hand sides are the squarings proved in derivations/elevation_footprint.mac (6) and
# derivations/cone_footprint.mac (3)-(5); on integers they are decided exactly. Prints one line
# per satellite and predicate, then the totals over the three satellites (what
# `elevation_accumulate_into` must count).
#
#   gap -q -A --quitonbreak derivations/kernel_lattice.g

axes := [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];;
cells := [];;
for x in [-6 .. 6] do for y in [-6 .. 6] do for z in [-6 .. 6] do
  Add(cells, [[x, y, z], axes[((x + y + z) mod 6) + 1]]);
od; od; od;
sats := [[3, -2, 9], [-7, 5, 4], [8, 8, -8]];;
names := ["horizon", "mask_plus_30", "mask_minus_30", "nadir_cone", "fixed_cone"];;

verdicts := function(S, cell)
  local C, z, d, t, q, n2, ut, u;
  C := cell[1]; z := cell[2];
  d := S - C; t := d * z; q := d * d;
  n2 := S * S; ut := n2 - C * S; u := (C - S) * [0, 0, -1];
  return [t >= 0,
          t >= 0 and 4 * t^2 >= q,
          t >= 0 or 4 * t^2 <= q,
          ut >= 0 and 16 * ut^2 >= 9 * q * n2,
          u >= 0 and 16 * u^2 >= 9 * q];
end;;

totals := [0, 0, 0, 0, 0];;
for i in [1 .. Length(sats)] do
  counts := [0, 0, 0, 0, 0];
  for cell in cells do
    v := verdicts(sats[i], cell);
    for k in [1 .. 5] do if v[k] then counts[k] := counts[k] + 1; fi; od;
  od;
  for k in [1 .. 5] do
    Print("satellite ", i, " ", names[k], ": ", counts[k], " of ", Length(cells), "\n");
    totals[k] := totals[k] + counts[k];
  od;
od;
for k in [1 .. 5] do Print("total ", names[k], ": ", totals[k], "\n"); od;
QUIT;
