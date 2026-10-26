# Second engine for mozhaev_point_groups.mac: the finite rotation groups of mozhaev.rs in exact
# arithmetic (integers, cyclotomics), their orders and isomorphism types, the polyhedra they are
# the full rotation groups of, and the trace histograms the Rust gate compares against.
# Each check prints `<name>: 0` (0 = holds).
# Run: timeout 120 gap -q -A --quitonbreak derivations/mozhaev_point_groups.g
ok := function(b) if b then return 0; else return 1; fi; end;
run := function()
  local perms, signed, T, O, phi, c, k, sk, K, r3, r5, Y, tet, cube, ico, proper48, hist,
        n, rz, Cn, Dn, flip, ngon;
  # Signed permutation matrices P*S; the code keeps those with sign(P) = det(S) (proper), and for
  # T only the even P.
  perms := List(SymmetricGroup(3), p -> PermutationMat(p, 3));
  signed := Concatenation(List(perms, P -> List(Tuples([1, -1], 3), s -> P * DiagonalMat(s))));
  proper48 := Filtered(signed, m -> DeterminantMat(m) = 1);
  O := proper48;
  T := Filtered(proper48, m -> PermutationMat(PermList(List(m, r -> PositionProperty(r, x -> x <> 0))), 3) in
                                List(AlternatingGroup(3), p -> PermutationMat(p, 3)));
  Print("octahedral_has_24_elements: ", ok(Size(O) = 24), "\n");
  Print("octahedral_is_S4: ", ok(IdGroup(Group(O)) = IdGroup(SymmetricGroup(4))), "\n");
  Print("tetrahedral_has_12_elements: ", ok(Size(T) = 12), "\n");
  Print("tetrahedral_is_A4: ", ok(IdGroup(Group(T)) = IdGroup(AlternatingGroup(4))), "\n");
  # The proper rotations preserving the tetrahedron are exactly T; those preserving the cube, O.
  tet := Set([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]);
  cube := Set(Tuples([1, -1], 3));
  Print("tetrahedral_is_the_rotation_group_of_the_tetrahedron: ",
        ok(Set(Filtered(proper48, m -> Set(tet, v -> v * TransposedMat(m)) = tet)) = Set(T)), "\n");
  Print("octahedral_is_the_rotation_group_of_the_cube: ",
        ok(Set(Filtered(proper48, m -> Set(cube, v -> v * TransposedMat(m)) = cube)) = Set(O)), "\n");
  # Icosahedral: over Q(sqrt 5), sqrt 5 = ER(5).
  phi := (1 + ER(5)) / 2;
  c := (ER(5) - 1) / 4;
  k := [0, 1, phi]; sk := k / 2;
  K := [[0, -sk[3], sk[2]], [sk[3], 0, -sk[1]], [-sk[2], sk[1], 0]];
  r5 := c * IdentityMat(3) + K + (1 - c) * List(k, x -> List(k, y -> x * y)) / (1 + phi^2);
  r3 := [[0, 0, 1], [1, 0, 0], [0, 1, 0]];
  Y := Group(r3, r5);
  Print("icosahedral_has_60_elements: ", ok(Size(Y) = 60), "\n");
  Print("icosahedral_is_A5: ", ok(IdGroup(Y) = IdGroup(AlternatingGroup(5))), "\n");
  ico := Set(Concatenation(List(Tuples([1, -1], 2), s -> [[0, s[1], s[2]*phi], [s[2]*phi, 0, s[1]], [s[1], s[2]*phi, 0]])));
  Print("icosahedral_preserves_the_icosahedron: ",
        ok(ForAll(Elements(Y), m -> Set(ico, v -> v * TransposedMat(m)) = ico)), "\n");
  hist := Collected(List(Elements(Y), TraceMat));
  Print("icosahedral_trace_histogram: ",
        ok(hist = Collected(Concatenation([3], ListWithIdenticalEntries(15, -1), ListWithIdenticalEntries(20, 0),
                                         ListWithIdenticalEntries(12, phi), ListWithIdenticalEntries(12, 1 - phi)))), "\n");
  Print("tetrahedral_trace_histogram: ",
        ok(Collected(List(T, TraceMat)) = [[-1, 3], [0, 8], [3, 1]]), "\n");
  Print("octahedral_trace_histogram: ",
        ok(Collected(List(O, TraceMat)) = [[-1, 9], [0, 8], [1, 6], [3, 1]]), "\n");
  # Cyclic and dihedral: rotation about z by 2 pi/n in cyclotomics, the flip diag(1, -1, -1).
  flip := DiagonalMat([1, -1, -1]);
  for n in [2..8] do
    rz := [[(E(n) + E(n)^-1)/2, -(E(n) - E(n)^-1)/(2*E(4)), 0],
           [(E(n) - E(n)^-1)/(2*E(4)), (E(n) + E(n)^-1)/2, 0], [0, 0, 1]];
    Cn := Group(rz); Dn := Group(rz, flip);
    ngon := Set(List([0..n-1], j -> [1, 0, 0] * TransposedMat(rz^j)));
    Print("cyclic_", n, ": ", ok(Size(Cn) = n and IsCyclic(Cn)), "\n");
    Print("dihedral_", n, ": ", ok(Size(Dn) = 2*n and IdGroup(Dn) = IdGroup(DihedralGroup(2*n))
          and ForAll(Elements(Dn), m -> Set(ngon, v -> v * TransposedMat(m)) = ngon)), "\n");
  od;
end;
run();
QUIT;
