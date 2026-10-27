\\ Second engine for derivations/six_s_coupling_brdf.mac, by other routes:
\\  * the coupling's inverse, y = (r - ra)/(Td Tu), rho = y/(1 + s y), against the forward model at
\\    sampled atmospheres and surfaces;
\\  * the four-node Gauss-Legendre rule on [0, 1]: nodes as the polroots of the shifted P_4, weights
\\    as the integrals of the Lagrange basis polynomials over [0, 1] -- not the closed forms Maxima
\\    checks -- against Maxima's 40-digit literals;
\\  * RossThick and LiSparse-Reciprocal (h/b = 2, b/r = 1) at the module test's five geometries
\\    against Maxima's literals, and their reciprocity in (ts, tv) at sampled geometries.
default(realprecision, 50);
{
  my(bad = 0, n = 0);
  forstep (ra = 1/50, 1/5, 3/50, forstep (td = 1/2, 1, 1/4, forstep (s = 0, 3/10, 1/10,
    forstep (rho = 0, 9/10, 3/10,
      my(tu = 3/4, r = ra + td*tu*rho/(1 - s*rho), y = (r - ra)/(td*tu));
      n++; if (y/(1 + s*y) != rho, bad++)))));
  printf("coupling inverse: %d of %d surfaces not recovered exactly in Q\n", bad, n);
}
{
  my(p = subst(pollegendre(4), x, 2*x - 1), r = vecsort(real(polroots(p))), w = vector(4), bad = 0);
  for (i = 1, 4,
    my(l = prod(j = 1, 4, if (j == i, 1, (x - r[j])/(r[i] - r[j]))), L = intformal(l));
    w[i] = subst(L, x, 1) - subst(L, x, 0));
  my(nodes = [0.06943184420297371238802675555359524745214, 0.3300094782075718675986671204483776563997,
              0.6699905217924281324013328795516223436003, 0.9305681557970262876119732444464047525479],
     weights = [0.1739274225687269286865319746109997036177, 0.3260725774312730713134680253890002963823,
                0.3260725774312730713134680253890002963823, 0.1739274225687269286865319746109997036177]);
  for (i = 1, 4, if (abs(r[i] - nodes[i]) > 1e-39 || abs(w[i] - weights[i]) > 1e-39, bad++));
  printf("gauss-legendre on [0, 1]: %d of 4 (node, weight) pairs off Maxima by more than 1e-39\n", bad);
  my(exact = 0);
  for (k = 0, 7, if (abs(sum(i = 1, 4, w[i]*r[i]^k) - 1/(k + 1)) > 1e-45, exact++));
  printf("gauss-legendre on [0, 1]: %d of 8 monomials to degree 7 not integrated to 1e-45\n", exact);
}
ross(ts, tv, ph) = {
  my(cx = cos(ts)*cos(tv) + sin(ts)*sin(tv)*cos(ph), xi = acos(cx));
  ((Pi/2 - xi)*cx + sin(xi))/(cos(ts) + cos(tv)) - Pi/4;
}
li(ts, tv, ph) = {
  my(hb = 2, br = 1, a = atan(br*tan(ts)), b = atan(br*tan(tv)),
     cx = cos(a)*cos(b) + sin(a)*sin(b)*cos(ph),
     d2 = tan(a)^2 + tan(b)^2 - 2*tan(a)*tan(b)*cos(ph),
     sec = 1/cos(a) + 1/cos(b), ct = hb*sqrt(d2 + (tan(a)*tan(b)*sin(ph))^2)/sec, t);
  \\ No shadow overlap past cos t = 1: t = 0, as the code clamps. An exact 0, because GP's acos(1)
  \\ is a zero of precision 1e-28, which then caps the precision of the whole sum.
  if (ct > 1, ct = 1; t = 0, t = acos(ct));
  (t - sin(t)*ct)*sec/Pi - sec + (1 + cx)/(2*cos(a)*cos(b));
}
{
  my(d = Pi/180, bad = 0,
     cases = [[45, 30, 60, 0.06123861247758755675933171403581419327336,
                            -0.9552160455924883580595985280167486233678],
              [20, 40, 120, -0.08625877784687792235715933519234383905726,
                            -1.224820988564084733545341968275913626576],
              [60, 50, 180, 0.1413533729268955512947118444333786980095,
                            -2.532088886237956070404785301110833347872],
              [30, 0, 0, -0.03144289608768314041759405532093950762116,
                         -0.6982224735605751537087532896575439271428],
              [10, 55, 270, -0.03830821916517112579576787872560044201614,
                            -1.373702271166547775758579478927805015513]]);
  for (i = 1, #cases, my(c = cases[i]);
    if (abs(ross(c[1]*d, c[2]*d, c[3]*d) - c[4]) > 1e-39, bad++);
    if (abs(li(c[1]*d, c[2]*d, c[3]*d) - c[5]) > 1e-39, bad++));
  printf("brdf kernels: %d of %d values off Maxima by more than 1e-39\n", bad, 2*#cases);
}
{
  my(bad = 0);
  setrand(20261027);
  for (i = 1, 200,
    my(ts = random(1.)*1.4, tv = random(1.)*1.4, ph = 2*Pi*random(1.));
    if (abs(ross(ts, tv, ph) - ross(tv, ts, ph)) > 1e-45 || abs(li(ts, tv, ph) - li(tv, ts, ph)) > 1e-45,
      bad++));
  printf("brdf reciprocity: %d of 200 sampled geometries break it by more than 1e-45\n", bad);
}
