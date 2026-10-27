\\ Second engine for derivations/ortho_bilinear_dem.mac: the ray-DEM intersection of
\\ crates/spacelab-earthvision/src/ortho.rs on the real Pleiades RPC00B of
\\ crates/spacelab-earthvision/tests/fixtures/pleiades_reunion.rpb over the planar DEM
\\     h = 1300 + 2048 (lat + 21.3125) + 1024 (lon - 55.75)   (metres, degrees),
\\ which a bilinear grid reproduces exactly (the .mac's item 2). The pixel's line of sight meets the
\\ surface where forward(lat, lon, h) = (row, col) and h = DEM(lat, lon); with h eliminated through
\\ the plane this is a 2x2 system in (P, L), solved by Newton at 60 digits with PARI's polynomial
\\ derivatives. The camera is rounded to the doubles the fixture parses to, as in
\\ derivations/rpc00b_inverse.gp, whose coefficient block is repeated here. Each root prints
\\ `<name>: 0` when it re-images to its pixel to 1e-50, then its row (row, col, lat, lon, h,
\\ dlat/dh, dlon/dh) -- the last two the ground drift per metre of height along the line of sight,
\\ which turns the bisection's height tolerance into a ground one.
\\
\\ Run: timeout 300 gp -q -f derivations/ortho_bilinear_dem.gp </dev/null
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
dem(lat, lon) = 1300 + 2048*(lat + 21.3125) + 1024*(lon - 55.75);
image(rn, cn, P, L, H) = [rpoly(lineNumCoef, P, L, H)/rpoly(lineDenCoef, P, L, H) - rn, rpoly(sampNumCoef, P, L, H)/rpoly(sampDenCoef, P, L, H) - cn]~;
ray(row, col, tag) = {
  my(rn = (row - lineOffset)/lineScale, cn = (col - sampOffset)/sampScale, F, J, G, Jh, rate,
     x = [0., 0.]~, step = 1, k = 0, lat, lon, h);
  F = image(rn, cn, p, l, (dem(p*latScale + latOffset, l*longScale + longOffset) - heightOffset)/heightScale);
  J = matrix(2, 2, i, j, deriv(F[i], [p, l][j]));
  while(step > 1e-55 && k < 100,
    my(Fx = substvec(F, [p, l], Vec(x)), Jx = substvec(J, [p, l], Vec(x)), dx);
    dx = matsolve(Jx, Fx); x -= dx; step = vecmax(apply(abs, dx)); k++);
  chk(Str(tag, "_re_images_to_its_pixel_on_the_dem"), vecmax(apply(abs, substvec(F, [p, l], Vec(x)))));
  lat = x[1]*latScale + latOffset; lon = x[2]*longScale + longOffset; h = dem(lat, lon);
  \\ Along the ray at this pixel, (P, L) move with the height hh: d(P, L)/dh = -J^-1 dG/dh.
  G = image(rn, cn, p, l, (hh - heightOffset)/heightScale);
  chk(Str(tag, "_lies_on_its_ray"), vecmax(apply(abs, substvec(G, [p, l, hh], [x[1], x[2], h]))));
  Jh = matrix(2, 2, i, j, substvec(deriv(G[i], [p, l][j]), [p, l, hh], [x[1], x[2], h]));
  rate = -matsolve(Jh, substvec(deriv(G, hh), [p, l, hh], [x[1], x[2], h]));
  printf("        (%.2f, %.2f, %s, %s, %s, %s, %s),\n", row, col, fmt(lat), fmt(lon), fmt(h),
    fmt(rate[1]*latScale), fmt(rate[2]*longScale));
}
print("rows (row, col, lat, lon, h, dlat/dh, dlon/dh):");
ray(19403.5, 19999.5, "centre");
ray(19150.25, 20311.75, "off_axis");
ray(19650.5, 19700.25, "lower_left");
