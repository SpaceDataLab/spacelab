# Second engine for lattice_flower.mac. Scaled by T = No*Nso, the module's satellite (i, j) is the
# integer point [i*Nso, Nc*i + j*No] mod T. For every 1 <= No, Nso <= 8 and 0 <= Nc < No this
# checks that the No*Nso points are distinct; that they are exactly the solutions in (Z/T)^2 of the
# Hermite-normal-form system of Avendano, Davis & Mortari (2013) with the paper's configuration
# number Nc' = (No - Nc) mod No:  No*x = 0 and Nc'*x + Nso*y = 0 (mod T); and that they equal the
# Walker Delta pattern T/No/Nc of walker_phasing.g. Each line prints the number of counterexamples.
# Run: timeout 120 gap -q -A --quitonbreak derivations/lattice_flower.g
run := function()
  local bad, No, Nso, Nc, T, pat, hnf, walker, ncp;
  bad := rec(distinct := 0, hnf := 0, walker := 0);
  for No in [1..8] do for Nso in [1..8] do for Nc in [0..No-1] do
    T := No*Nso;
    pat := Set(Cartesian([0..No-1], [0..Nso-1]), ij -> [ij[1]*Nso mod T, (Nc*ij[1] + ij[2]*No) mod T]);
    if Size(pat) <> T then bad.distinct := bad.distinct + 1; fi;
    ncp := (No - Nc) mod No;
    hnf := Filtered(Cartesian([0..T-1], [0..T-1]),
                    xy -> (No*xy[1]) mod T = 0 and (ncp*xy[1] + Nso*xy[2]) mod T = 0);
    if Set(hnf) <> pat then bad.hnf := bad.hnf + 1; fi;
    walker := Set(Cartesian([0..No-1], [0..Nso-1]), ps -> [ps[1]*Nso mod T, (ps[2]*No + ps[1]*Nc) mod T]);
    if walker <> pat then bad.walker := bad.walker + 1; fi;
  od; od; od;
  Print("lattice_has_No_Nso_distinct_satellites: ", bad.distinct, "\n");
  Print("lattice_is_the_hnf_solution_set_with_the_paper_Nc: ", bad.hnf, "\n");
  Print("circular_lattice_is_walker_delta: ", bad.walker, "\n");
end;
run();
QUIT;
