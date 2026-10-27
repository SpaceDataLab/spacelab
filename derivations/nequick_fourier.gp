/* Second engine for derivations/nequick_fourier.mac: the CCIR Fourier collapse (ICD issue 1.2,
   Eq. 19, 44, 46, 49 to 51) recomputed at 60 digits from the April and November coefficient files,
   against the 30-digit values Maxima prints and fourier.rs's gate test stores.

   Run from derivations/, as verify.sh does:
       timeout 300 gp -q -f nequick_fourier.gp </dev/null                           */

default(realprecision, 60);

{read_ccir(month) = my(lines = readstr(Str("../crates/spacelab-ionosphere/assets/ccir", month + 10,
    ".asc")), out = List());
  for (i = 1, #lines,
    foreach (strsplit(lines[i], " "), t, if (#t > 0, listput(out, eval(t)))));
  if (#out != 2858, error("not a CCIR file"));
  Vec(out)}

{collapse(tok, az, ut) = my(azr = sqrt(167273 + (az - 637/10) * 11236/10) - 40899/100, x = azr / 100,
    T = (15 * ut - 180) * Pi / 180, f2, fm3, s);
  f2 = vector(76, j, my(a(k) = tok[k + 13*(j-1)] * (1 - x) + tok[k + 13*(j-1) + 13*76] * x);
    a(1) + sum(k = 1, 6, a(2*k) * sin(k*T) + a(2*k+1) * cos(k*T)));
  fm3 = vector(49, j, my(a(k) = tok[1976 + k + 9*(j-1)] * (1 - x)
                                + tok[1976 + k + 9*(j-1) + 9*49] * x);
    a(1) + sum(k = 1, 4, a(2*k) * sin(k*T) + a(2*k+1) * cos(k*T)));
  [azr, f2, fm3]}

/* (azr, cf2 at columns 1 2 12 13 14 40 75 76, cm3 at 1 2 7 8 9 30 49), from the Maxima run. */
{mac = [
  [4, 14, 120, 71.1471470736252042112660461417,
   [9.12223593604019594988146689593, 0.204212382122895880385204145946,
    -99.0823125892341032503875837008, 2.05544594453157225857730150081,
    0.851630079410417149926445335472, 2.34377317496028190014896413334,
    -0.014781480904526359623634447616, -0.0272574865798739655502607101637],
   [2.79882805689102154993477808498, -0.00450943701251093231918203593633,
    0.688014772571126752667163770067, -0.372045977159391120901205773178,
    -0.0991449548708951718712978771456, -1.55801874754669734120870904393,
    -0.0151825996287427945130093226789]],
  [11, 7/2, 200, 157.066251621691746509393615302,
   [10.9728045605238602941331768435, 1.66278053212675456499885848235,
    60.7310060787375204704745903564, -2.12385146309532944628296622518,
    1.16449980694137888042277492873, 1.0988111689206664580960341577,
    -0.133278786554370562432571720446, 0.0700815342838702520918425993297],
   [2.47238864474734654005027887128, 0.197104744476384855600274894708,
    0.336970444466532078785576835126, 0.469697848917377625591586484224,
    0.0382162233672680513144405590261, 0.248554406595724591554708649932,
    0.0244030564611182989549593536015]]];}

f2cols = [1, 2, 12, 13, 14, 40, 75, 76];
m3cols = [1, 2, 7, 8, 9, 30, 49];
{bad = 0; n = 0;
  for (c = 1, #mac, my(m = mac[c], v = collapse(read_ccir(m[1]), m[3], m[2]));
    n++; bad += abs(v[1] - m[4]) > 1e-27 * abs(m[4]);
    for (i = 1, #f2cols, n++; bad += abs(v[2][f2cols[i]] - m[5][i]) > 1e-27 * max(1, abs(m[5][i])));
    for (i = 1, #m3cols, n++; bad += abs(v[3][m3cols[i]] - m[6][i]) > 1e-27 * max(1, abs(m[6][i]))));
  printf("Maxima's 30-digit collapse values off by more than 1e-27: %d of %d\n", bad, n);}
