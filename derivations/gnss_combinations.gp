/* Dual-frequency GNSS combinations in PARI/GP: the second engine for derivations/gnss_combinations.mac.

   The same observation equations, P_i = rho + ion/f_i^2 and L_i = rho - ion/f_i^2 + (c/f_i) N_i, as
   PARI rational functions in (f1, f2, rho, ion, c, N1, N2) -- `ion`, since I is PARI's imaginary
   unit -- exact, no floating point, and the cancellations the module claims, each a difference
   that must be the zero rational function.
   Then the ICD wavelengths as exact rationals at 60 digits against the 40-digit values the Maxima
   file prints and the Rust gate stores. Every mismatch prints `FAIL`.

       timeout 300 gp -q -f derivations/gnss_combinations.gp </dev/null                             */

default(realprecision, 60);

bad = 0;
iszero(name, e) = if (e != 0, bad++; printf("FAIL %s: %s\n", name, e), printf("%s: 0\n", name));

IF(x1, x2) = (f1^2 * x1 - f2^2 * x2) / (f1^2 - f2^2);
WL(x1, x2) = (f1 * x1 - f2 * x2) / (f1 - f2);
NL(x1, x2) = (f1 * x1 + f2 * x2) / (f1 + f2);

P1 = rho + ion / f1^2; P2 = rho + ion / f2^2;
L1 = rho - ion / f1^2 + (c / f1) * N1; L2 = rho - ion / f2^2 + (c / f2) * N2;

iszero("if_code_is_rho", IF(P1, P2) - rho);
iszero("if_phase_is_rho_plus_a_float_ambiguity", IF(L1, L2) - rho - IF((c / f1) * N1, (c / f2) * N2));
iszero("wl_phase", WL(L1, L2) - (rho + ion / (f1 * f2) + c / (f1 - f2) * (N1 - N2)));
iszero("nl_code", NL(P1, P2) - (rho + ion / (f1 * f2)));
iszero("mw", WL(L1, L2) - NL(P1, P2) - c / (f1 - f2) * (N1 - N2));
iszero("graphic", (P1 + L1) / 2 - (rho + (c / f1) * N1 / 2));
iszero("nl_gain_squared", (f1^2 + f2^2) / (f1 + f2)^2 - (1/2 + (f1 - f2)^2 / (2 * (f1 + f2)^2)));

f0 = 10230000; cc = 299792458;
check(name, x, want) = {
  my(d = abs(x - want) / want);
  if (d > 1e-38, bad++; printf("FAIL %s: %.45g vs %.45g\n", name, x, want),
                 printf("%s: ok (rel %.2e)\n", name, d));
}
check("lambda1", cc / (154 * f0), 1.902936727983648804763174264640540300364e-1);
check("lambda2", cc / (120 * f0), 2.442102134245682632779406972955360052134e-1);
check("lambda5", cc / (115 * f0), 2.548280487908538399421989884822984402227e-1);
check("lambda_wl_l1_l2", cc / (34 * f0), 8.619184003220056350986142257489506066356e-1);
check("lambda_nl_l1_l2", cc / (274 * f0), 1.069533781421466846472732980856362066628e-1);
check("lambda_wl_l1_l5", cc / (39 * f0), 7.514160413063638870090482993708800160413e-1);
check("lambda_nl_l1_l5", cc / (269 * f0), 1.089413591485062884511259616188264707272e-1);
printf("gnss combinations: %d checks off\n", bad);
quit;
