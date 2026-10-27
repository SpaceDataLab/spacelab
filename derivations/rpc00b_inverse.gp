\\ Second engine for derivations/rpc00b.mac: the image-to-ground solve of
\\ crates/spacelab-earthvision/src/rpc.rs on the real Pleiades RPC00B of
\\ crates/spacelab-earthvision/tests/fixtures/pleiades_reunion.rpb, at 60 digits. Every offset, scale
\\ and coefficient is first rounded to the double the fixture's decimal parses to (dbl), so the roots
\\ are the roots for the camera the Rust gate `the_inverse_is_the_root_of_the_rpc_equations` builds.
\\ The 2x2 system  NumL/DenL(P, L, H) = rn,  NumS/DenS(P, L, H) = cn  is solved by Newton with the
\\ Jacobian from PARI's own polynomial derivatives (not the hand-written gradient), to a step below
\\ 1e-55. Each root prints `<name>: 0` when it re-images to its pixel to 1e-50 normalised units,
\\ then its row (row, col, height, latitude, longitude).
\\
\\ Run: timeout 300 gp -q -f derivations/rpc00b_inverse.gp </dev/null
default(realprecision, 60);
chk(name, e) = printf("%s: %s\n", name, if(abs(e) < 1e-50, 0, e));
fmt(v) = concat([c | c <- Vec(Strprintf("%.30e", v)), c != " "]);
\\ The double nearest a real: 53 significant bits.
dbl(x) = if(x == 0, 0, my(e = exponent(x)); round(x*2^(52 - e))/2^(52 - e));

\\ Transcribed from the fixture.
lineOffset = dbl(19403.5);
sampOffset = dbl(19999.5);
latOffset = dbl(-21.2316081288);
longOffset = dbl(55.7119698801);
heightOffset = dbl(1295.0);
lineScale = dbl(512.0);
sampScale = dbl(512.0);
latScale = dbl(0.0911805852907);
longScale = dbl(0.0985353286675);
heightScale = dbl(1315.0);
lineNumCoef = apply(dbl, [-37.284870906, -0.389307964671, -39.0126569672, 0.756244483967, 0.0365724832883, -0.000699590543671, 5.69148667027e-05, 0.00488795358124, -0.0493487209079, -0.00486335415172, 6.61460426948e-05, -4.3251938614e-05, -0.00106436732503, -5.45489448461e-05, -0.00330149225713, -0.0169088089294, -0.00493745513823, 6.47041405124e-05, 0.000507944645931, 9.58883770134e-05]);
lineDenCoef = apply(dbl, [1.0, 0.000997771806716, 0.000893795146776, -2.56359129684e-05, -1.70851501528e-05, -3.21867506534e-07, 2.1532776166e-05, 8.13369760723e-05, -0.000270733342464, 0.000126355623049, -3.65549547543e-07, 1.73348528132e-07, 1.0566912918e-05, 2.48524027091e-07, 5.68010228875e-07, -3.14981737526e-06, -1.46513589459e-07, -1.44200775386e-08, 1.59078686184e-06, -3.43796798432e-09]);
sampNumCoef = apply(dbl, [-13.5564562154, 39.3860841344, -0.0427740622694, 0.275292011929, 0.0209037688985, 0.0398356057957, -0.0178925782936, -0.0421615828479, 0.0241060371425, 0.000605633181358, 8.44134747117e-05, -0.000310265910803, -0.000334358203674, -9.98841387442e-05, 0.00150262989976, 0.00587427322315, -6.15040098618e-06, -0.000108287671902, -0.000380016064247, -5.97860985933e-07]);
sampDenCoef = apply(dbl, [1.0, -0.000284860254189, -0.00052978538308, -0.00103796003315, -1.31064144938e-05, 1.15489105827e-06, -4.43060264739e-07, 2.9975738656e-06, 2.01322436172e-06, -4.17753982621e-06, -1.26431542483e-07, -7.81727809737e-09, -1.16012565638e-07, 3.82954839127e-09, 1.99730782682e-07, 2.12670502991e-06, 2.63893880667e-09, 2.15736481798e-08, -7.45465130415e-08, 5.17836239128e-09]);

\\ RPC00B term order: 1, L, P, H, LP, LH, PH, L^2, P^2, H^2, PLH, L^3, LP^2, LH^2, L^2P, P^3, PH^2,
\\ L^2H, P^2H, H^3.
terms(P, L, H) = [1, L, P, H, L*P, L*H, P*H, L^2, P^2, H^2, P*L*H, L^3, L*P^2, L*H^2, L^2*P, P^3, P*H^2, L^2*H, P^2*H, H^3];
rpoly(c, P, L, H) = c*terms(P, L, H)~;
solve_pixel(row, col, h, tag) = {
  my(rn = (row - lineOffset)/lineScale, cn = (col - sampOffset)/sampScale,
     H = (h - heightOffset)/heightScale, F, J, x = [0., 0.]~, step = 1, k = 0, lat, lon);
  F = [rpoly(lineNumCoef, p, l, H)/rpoly(lineDenCoef, p, l, H) - rn,
       rpoly(sampNumCoef, p, l, H)/rpoly(sampDenCoef, p, l, H) - cn]~;
  J = matrix(2, 2, i, j, deriv(F[i], [p, l][j]));
  while(step > 1e-55 && k < 100,
    my(Fx = substvec(F, [p, l], Vec(x)), Jx = substvec(J, [p, l], Vec(x)), dx);
    dx = matsolve(Jx, Fx); x -= dx; step = vecmax(apply(abs, dx)); k++);
  chk(Str(tag, "_re_images_to_its_pixel"), vecmax(apply(abs, substvec(F, [p, l], Vec(x)))));
  lat = x[1]*latScale + latOffset;
  lon = x[2]*longScale + longOffset;
  printf("            (%.2f, %.2f, %.1f, %s, %s),\n", row, col, h, fmt(lat), fmt(lon));
}
print("rows (row, col, height, latitude, longitude):");
solve_pixel(19403.5, 19999.5, 1295, "centre");
solve_pixel(5000, 8000, 1595, "upper_left");
solve_pixel(30000, 32000, 795, "lower_right");
solve_pixel(19403.5, 19999.5, 2295, "centre_high");
solve_pixel(19150.25, 20311.75, 1043.5, "off_axis");
