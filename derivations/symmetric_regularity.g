# Orbit-stabilizer counts and the regularity condition for symmetric.rs, in exact arithmetic.
# Each check prints `<name>: 0` (0 = holds, or the number of counterexamples).
# Run: timeout 120 gap -q -A --quitonbreak derivations/symmetric_regularity.g
ok := function(b) if b then return 0; else return 1; fi; end;
run := function()
  local perms, signed, O, T, phi, c, k, K, r5, Y, orb, n, m, q, r, f, D, X, bad, x, y;
  perms := List(SymmetricGroup(3), p -> PermutationMat(p, 3));
  signed := Concatenation(List(perms, P -> List(Tuples([1, -1], 3), s -> P * DiagonalMat(s))));
  O := Group(Filtered(signed, mm -> DeterminantMat(mm) = 1));
  T := Group(Filtered(signed, mm -> DeterminantMat(mm) = 1 and
              PermutationMat(PermList(List(mm, row -> PositionProperty(row, e -> e <> 0))), 3) in
              List(AlternatingGroup(3), p -> PermutationMat(p, 3))));
  phi := (1 + ER(5)) / 2; c := (ER(5) - 1) / 4; k := [0, 1, phi];
  K := [[0, -k[3], k[2]], [k[3], 0, -k[1]], [-k[2], k[1], 0]] / 2;
  r5 := c * IdentityMat(3) + K + (1 - c) * List(k, a -> List(k, b -> a * b)) / (1 + phi^2);
  Y := Group([[0, 0, 1], [1, 0, 0], [0, 1, 0]], r5);
  # The seed's orbit plane is carried to |G| / |Stab(n)| distinct oriented normals, each holding
  # |Stab(n)| satellites: [group, normal, planes].
  orb := function(G, v) return Size(Orbit(G, v, OnRight)); end;
  Print("tetrahedral_pole_planes_6: ", ok(orb(T, [0, 0, 1]) = 6), "\n");
  Print("tetrahedral_threefold_planes_4: ", ok(orb(T, [1, 1, 1]) = 4), "\n");
  Print("tetrahedral_generic_planes_12: ", ok(orb(T, [1, 2, 5]) = 12), "\n");
  Print("octahedral_pole_planes_6: ", ok(orb(O, [0, 0, 1]) = 6), "\n");
  Print("octahedral_threefold_planes_8: ", ok(orb(O, [1, 1, 1]) = 8), "\n");
  Print("octahedral_twofold_planes_12: ", ok(orb(O, [1, 1, 0]) = 12), "\n");
  Print("icosahedral_fivefold_planes_12: ", ok(orb(Y, [0, 1, phi]) = 12), "\n");
  Print("icosahedral_threefold_planes_20: ", ok(orb(Y, [1, 1, 1]) = 20), "\n");
  Print("icosahedral_twofold_planes_30: ", ok(orb(Y, [0, 0, 1]) = 30), "\n");
  # Regularity. Rotations about z by 2 pi/n and the flip about x act freely on a seed whose normal
  # is on no group axis, so its images are labelled by group elements, and with time shifts by
  # 2 pi/m of mean anomaly (stagger q, sigma = 2 pi q/m) the emitted set is
  #   cyclic   {(r^j, j q)},   dihedral  {(r^j, j q), (r^j f, j q)}   in  D_n x Z_m.
  # It is the orbit of a group iff it is a subgroup. Claimed: cyclic iff n q = 0 (mod m);
  # dihedral iff n q = 0 and 2 q = 0 (mod m).
  bad := rec(cyclic := 0, dihedral := 0);
  for n in [2..8] do
    D := DihedralGroup(IsPermGroup, 2*n);
    r := First(GeneratorsOfGroup(D), g -> Order(g) = n);
    f := First(Elements(D), g -> Order(g) = 2 and not g in Group(r));
    for m in [1..12] do for q in [0..m-1] do
      X := Set([0..n-1], j -> [r^j, (j*q) mod m]);
      if ForAll(X, a -> ForAll(X, b -> [a[1]*b[1], (a[2] + b[2]) mod m] in X)) <> ((n*q) mod m = 0) then
        bad.cyclic := bad.cyclic + 1;
      fi;
      X := Union(X, Set([0..n-1], j -> [r^j*f, (j*q) mod m]));
      if ForAll(X, a -> ForAll(X, b -> [a[1]*b[1], (a[2] + b[2]) mod m] in X))
         <> ((n*q) mod m = 0 and (2*q) mod m = 0) then
        bad.dihedral := bad.dihedral + 1;
      fi;
    od; od;
  od;
  Print("cyclic_arm_is_regular_iff_n_stagger_is_whole_turns: ", bad.cyclic, "\n");
  Print("dihedral_arm_is_regular_iff_also_2_stagger_is_whole_turns: ", bad.dihedral, "\n");
end;
run();
QUIT;
