\\ Second engine for derivations/rpc_bias_affine.mac: the six-point affine fit of the Rust gate
\\ `the_affine_fit_solves_the_normal_equations` in crates/spacelab-earthvision/src/rpc_adjust.rs,
\\ solved exactly in rationals. The design rows [1, col, row] sit at the image coordinates of the
\\ Pleiades fixture (around column 20000, row 19400) and the targets are dyadic, so every input is
\\ the double the gate passes. Prints `<name>: 0` for the exact optimality conditions, then the exact
\\ solution and the 2-norm condition numbers of A^T A, raw and equilibrated, that bound an f64
\\ normal-equation solve.
\\
\\ Run: timeout 300 gp -q -f derivations/rpc_bias_affine.gp </dev/null
default(realprecision, 60);
chk(name, e) = printf("%s: %s\n", name, if(e == 0, 0, e));
fmt(v) = concat([c | c <- Vec(Strprintf("%.30e", v)), c != " "]);
cols = [19620, 20380.5, 19750.25, 20240, 19999.5, 20111.75];
rows = [19150, 19660.5, 19900.75, 19040.25, 19403.5, 19555];
targets = [12.5, 13.25, 11.875, 12.0625, 12.75, 13.5];
A = matrix(6, 3, i, j, [1, cols[i], rows[i]][j]);
A = apply(x -> bestappr(x), A);
t = apply(x -> bestappr(x), targets)~;
N = A~*A; b = A~*t;
x = matsolve(N, b);
chk("normal_equations_hold_exactly", vecmax(apply(abs, N*x - b)));
chk("residual_is_orthogonal_to_the_design", vecmax(apply(abs, A~*(t - A*x))));
\\ The minimum is unique: N is positive definite (all leading minors positive).
chk("gram_matrix_is_positive_definite", (N[1,1] > 0 && matdet(N[1..2, 1..2]) > 0 && matdet(N) > 0) - 1);
ev = real(polroots(charpoly(N)));
printf("solution = [%s, %s, %s]\n", fmt(x[1]), fmt(x[2]), fmt(x[3]));
printf("solution (exact) = %s\n", x);
printf("condition number of A^T A = %s\n", fmt(vecmax(ev)/vecmin(ev)));
\\ Gaussian elimination is insensitive to the column scaling, so the bound that matters is the
\\ condition number of the equilibrated Gram matrix D^-1 N D^-1, D = sqrt(diag N).
Dinv = matdiagonal(vector(3, i, 1/sqrt(N[i, i])));
evs = real(polroots(charpoly(Dinv*N*Dinv)));
printf("condition number of the equilibrated A^T A = %s\n", fmt(vecmax(evs)/vecmin(evs)));
