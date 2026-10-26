\\ Second engine for derivations/root_reference_roots.mac: the four reference roots of
\\ crates/spacelab-coverage/tests/root.rs from PARI's own solvers at 60 digits, against the
\\ 34-digit literals the test holds.
default(realprecision, 60);
lit = [2.094551481542326591482386540579303, 0.7390851332151606416553120876738734, \
       0.5671432904097838729999686622103555, 1.414213562373095048801688724209698];
ref = [polrootsreal(x^3 - 2*x - 5)[1], solve(x = 0, 1.5, cos(x) - x), lambertw(1), sqrt(2)];
names = ["ROOT_CUBIC", "ROOT_DOTTIE", "ROOT_OMEGA", "ROOT_SQRT2"];
{
  for (i = 1, 4,
    d = abs(lit[i] - ref[i]);
    printf("%s: |literal - PARI| = %.2e  %s\n", names[i], d, if (d < 1e-33, "ok", "MISMATCH")));
}
