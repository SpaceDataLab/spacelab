/* Second engine for derivations/kronrod_g7k15.mac: the Gauss G7 / Kronrod K15 nodes and weights
   of the NeQuick-G slant-TEC quadrature (ICD issue 1.2, Annex F.2.6), from their definitions at
   80 digits, against the 33-digit values the ICD prints and galileo.rs stores.

   G7 nodes are the roots of P7, G7 weights 2/((1 - x^2) P7'(x)^2); the eight extra K15 nodes are
   the roots of the Stieltjes polynomial E8 (monic, degree 8, orthogonal to P7 x^k for k = 0..7,
   solved here as an exact rational linear system), and the K15 weights are those of the
   interpolatory rule on all fifteen nodes.

       timeout 300 gp -q -f derivations/kronrod_g7k15.gp </dev/null                              */

default(realprecision, 80);

P7 = pollegendre(7);
integ(p) = {my(q = intformal(p)); subst(q, x, 1) - subst(q, x, -1)};
{c = matsolve(matrix(4, 4, k, j, integ(x^(2 * j - 2) * P7 * x^(2 * k - 1))),
             vector(4, k, -integ(x^8 * P7 * x^(2 * k - 1)))~);}
E8 = x^8 + sum(j = 1, 4, c[j] * x^(2 * j - 2));
orth = sum(k = 0, 7, abs(integ(E8 * P7 * x^k)));
printf("Stieltjes E8 orthogonality residual over k = 0..7 (exact rationals): %d\n", orth);

g7 = vecsort(real(polroots(P7)));
k15 = vecsort(concat(g7, real(polroots(E8))));
dP7 = deriv(P7);
wg = vector(7, i, 2 / ((1 - g7[i]^2) * subst(dP7, x, g7[i])^2));
lag(i) = prod(j = 1, 15, if (j == i, 1, (x - k15[j]) / (k15[i] - k15[j])));
wk = vector(15, i, integ(lag(i)));

exact(w, n, k) = sum(i = 1, #n, w[i] * if (k == 0, 1, n[i]^k)) - if (k % 2, 0, 2 / (k + 1));
{printf("K15 moments x^0..x^22 off by more than 1e-60: %d\n",
  sum(k = 0, 22, abs(exact(wk, k15, k)) > 1e-60));}
{printf("G7 moments x^0..x^13 off by more than 1e-60: %d\n",
  sum(k = 0, 13, abs(exact(wg, g7, k)) > 1e-60));}
{printf("K15 moment x^24 departs from 2/25 by %.3e (the rule's degree is 22)\n",
  abs(exact(wk, k15, 24)));}

{icd_nodes = [-0.991455371120812639206854697526329, -0.949107912342758524526189684047851,
  -0.864864423359769072789712788640926, -0.741531185599394439863864773280788,
  -0.586087235467691130294144838258730, -0.405845151377397166906606412076961,
  -0.207784955007898467600689403773245, 0.,
  0.207784955007898467600689403773245, 0.405845151377397166906606412076961,
  0.586087235467691130294144838258730, 0.741531185599394439863864773280788,
  0.864864423359769072789712788640926, 0.949107912342758524526189684047851,
  0.991455371120812639206854697526329];}
{icd_wk = [0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
  0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
  0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
  0.204432940075298892414161999234649, 0.209482141084727828012999174891714,
  0.204432940075298892414161999234649, 0.190350578064785409913256402421014,
  0.169004726639267902826583426598550, 0.140653259715525918745189590510238,
  0.104790010322250183839876322541518, 0.063092092629978553290700663189204,
  0.022935322010529224963732008058970];}
{icd_wg = [0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
  0.381830050505118944950369775488975, 0.417959183673469387755102040816327,
  0.381830050505118944950369775488975, 0.279705391489276667901467771423780,
  0.129484966168869693270611432679082];}
{printf("ICD K15 nodes off by more than 5e-34: %d of 15\n",
  sum(i = 1, 15, abs(icd_nodes[i] - k15[i]) > 5e-34));}
{printf("ICD K15 weights off by more than 5e-34: %d of 15\n",
  sum(i = 1, 15, abs(icd_wk[i] - wk[i]) > 5e-34));}
{printf("ICD G7 weights off by more than 5e-34: %d of 7\n",
  sum(i = 1, 7, abs(icd_wg[i] - wg[i]) > 5e-34));}
{printf("odd K15 nodes that are not the G7 nodes (1e-70): %d of 7\n",
  sum(i = 1, 7, abs(k15[2 * i] - g7[i]) > 1e-70));}
printf("centre G7 weight - 512/1225: %.3e\n", wg[4] - 512 / 1225);

/* The Maxima derivation's 40-digit literals, as the Rust gate stores them. */
{mac_nodes = [0.2077849550078984676006894037732449134798, 0.4058451513773971669066064120769614633474,
  0.5860872354676911302941448382587295984368, 0.7415311855993944398638647732807884070741,
  0.864864423359769072789712788640926201211, 0.9491079123427585245261896840478512624008,
  0.991455371120812639206854697526328516642];}
{mac_wk = [0.2094821410847278280129991748917142636978, 0.2044329400752988924141619992346490847165,
  0.1903505780647854099132564024210136828261, 0.1690047266392679028265834265985502841062,
  0.1406532597155259187451895905102379203999, 0.1047900103222501838398763225415180174438,
  0.06309209262997855329070066318920428666507, 0.02293532201052922496373200805896959199356];}
{mac_wg = [0.4179591836734693877551020408163265306122, 0.3818300505051189449503697754889751338784,
  0.2797053914892766679014677714237795824869, 0.1294849661688696932706114326790820183286];}
{printf("Maxima 40-digit nodes and weights off by more than 1e-38: %d of 19\n",
  sum(i = 1, 7, abs(mac_nodes[i] - k15[8 + i]) > 1e-38)
  + sum(i = 1, 8, abs(mac_wk[i] - wk[7 + i]) > 1e-38)
  + sum(i = 1, 4, abs(mac_wg[i] - wg[3 + i]) > 1e-38));}
