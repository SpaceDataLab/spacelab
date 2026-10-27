# The DOP cofactor matrix on rational geometry, in exact rationals: the second engine for
# derivations/gnss_dop.mac.
#
# The same six Pythagorean-quadruple lines of sight and the regular tetrahedron's normal matrix,
# inverted by GAP over the rationals; the diagonal must equal the rationals Maxima prints, and the
# tetrahedron's cofactor must be diag(3/4, 3/4, 3/4, 1/4). Every mismatch prints `MISMATCH`.
#
#     timeout 300 gap -q -A --quitonbreak derivations/gnss_dop.g

cofactor := function(los)
  local G;
  G := List(los, l -> Concatenation(l, [1]));
  return Inverse(TransposedMat(G) * G);
end;

bad := 0;
expect := function(name, got, want)
  if got <> want then
    bad := bad + 1;
    Print("MISMATCH ", name, ": ", got, " vs Maxima ", want, "\n");
  else
    Print(name, ": ", got, "\n");
  fi;
end;

sky := [[0, 0, 1], [2, 2, 1] / 3, [-6, 2, 3] / 7, [3, -4, 0] / 5, [-2, -6, 9] / 11, [12, -4, 3] / 13];
for l in sky do
  if l * l <> 1 then bad := bad + 1; Print("MISMATCH not a unit vector: ", l, "\n"); fi;
od;
Q := cofactor(sky);
expect("Q_EE", Q[1][1], 170814343 / 278268670);
expect("Q_NN", Q[2][2], 196061117 / 278268670);
expect("Q_UU", Q[3][3], 106318505 / 55653734);
expect("Q_TT", Q[4][4], 199779533 / 278268670);

# The tetrahedron's normal matrix needs no square root: with c = cos el, the ring's sums are
# sum e = 0, sum e^2 = sum n^2 = 3 c^2 / 2 = 4/3, sum u^2 = 1 + 3 s^2 = 4/3, sum u = 1 + 3 s = 0.
N := [[4/3, 0, 0, 0], [0, 4/3, 0, 0], [0, 0, 4/3, 0], [0, 0, 0, 4]];
expect("tetrahedron cofactor", Inverse(N), DiagonalMat([3/4, 3/4, 3/4, 1/4]));
Print("gnss dop: ", bad, " mismatches\n");
QUIT;
