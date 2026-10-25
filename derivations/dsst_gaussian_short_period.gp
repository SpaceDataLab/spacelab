\\ `spacelab-propagation::dsst::gaussian_short_period`: PARI/GP, the second engine beside
\\ dsst_gaussian_short_period.mac, by other routes, all in exact arithmetic.
\\   (1) d(lambda)/dL = 1 + 2 sum (rho_k cos kL + sigma_k sin kL): the lambda-mean of exp(ijL) is
\\       rho_j + i sigma_j, by residues at three ellipses with rational B, j = 1..JMAX (the .mac file
\\       uses the generating function, for all e);
\\   (2) the product to sum at the shipped JMAX = 12, in exact rationals: the U1, V1 sums are the
\\       product's coefficients, the U2, V2 sums are, to order t^(JMAX+1) with rho_k, sigma_k of
\\       order t^k; as Laurent polynomials in z = exp(iL), not by product-to-sum identities;
\\   (3) the closed forms of F = cos L solve their equations and have zero lambda-mean, as rational
\\       functions of beta, th and x = exp(iF), so for every ellipse.
\\ Rational ellipses: e = 2 beta/(1 + beta^2), B = (1 - beta^2)/(1 + beta^2) and
\\ k + ih = e (1 + i th)^2/(1 + th^2).

must(ok, what) = if(ok, print(what), error("FAILED: ", what));
JMAX = 12;
ellipse(bet, th) = my(e = 2*bet/(1 + bet^2)); [e*(1 - th^2)/(1 + th^2), e*2*th/(1 + th^2), (1 - bet^2)/(1 + bet^2)];
\\ The orbit through x = exp(iF): cos F, sin F, r/a, X/a and Y/a, with X + iY = r exp(iL).
orbit(k, h, B) = {
  my(b = 1/(1 + B), cF = (x + 1/x)/2, sF = (x - 1/x)/(2*I));
  [cF, sF, 1 - k*cF - h*sF, (1 - h^2*b)*cF + h*k*b*sF - k, (1 - k^2*b)*sF + h*k*b*cF - h];
}
\\ The residue of g at x0, and the mean over F of a Laurent polynomial in x.
res(g, x0) = polcoef(subst(g, x, x0 + x) + O(x), -1, x);
meanf(f) = polcoef(f + O(x), 0, x);

\\ ---- (1) ------------------------------------------------------------------------------------------
\\ d(lambda) = (r/a) dF, and (1/2 pi) times the integral over F is the sum of the residues of g/x
\\ inside |x| = 1. With g = exp(ijL) r/a = (X + iY)^j/r^(j-1) (a = 1), they sit at x = 0 and at the
\\ root (1 - B)/(k - ih) of r, of modulus beta.
{
  my(bad = 0);
  foreach([[1/7, 1/3], [1/3, -2], [1/2, 3]], p,
    my(kh = ellipse(p[1], p[2]), k = kh[1], h = kh[2], B = kh[3], o = orbit(k, h, B),
       x0 = (1 - B)/(k - I*h));
    for(j = 1, JMAX,
      my(g = (o[4] + I*o[5])^j/(o[3]^(j - 1)*x));
      if(res(g, 0) + res(g, x0) != (1 + j*B)*(-1/(1 + B))^j*(k + I*h)^j, bad++)));
  print("lambda-means of exp(ijL) that differ from rho_j + i sigma_j, j = 1..JMAX: ", bad);
  must(bad == 0, "(1) holds: the lambda-mean of exp(ijL) is rho_j + i sigma_j exactly at e = 7/25, 3/5, 4/5, j = 1..12");
}

\\ ---- (2) ------------------------------------------------------------------------------------------
\\ A cosine-sine pair times z^M, as a polynomial in z; the product of two carries z^(2M).
M = 3*JMAX;
trig(c, s, j) = c*(z^(M + j) + z^(M - j))/2 + s*(z^(M + j) - z^(M - j))/(2*I);
cosco(p, m) = polcoef(p, 2*M + m, z) + polcoef(p, 2*M - m, z);
sinco(p, m) = I*(polcoef(p, 2*M + m, z) - polcoef(p, 2*M - m, z));
setrand(20261025);
rr() = (random(2001) - 1000)/997;
{
  my(CC = vector(JMAX, j, rr()), SS = vector(JMAX, j, rr()),
     RH = vector(3*JMAX, k, rr()*t^k), SG = vector(3*JMAX, k, rr()*t^k),
     W, P1, P2, U, V, bad = 0, low = oo);
  my(cc = j -> if(j <= JMAX, CC[j], 0), ss = j -> if(j <= JMAX, SS[j], 0));
  W = z^M + sum(k = 1, 3*JMAX, 2*trig(RH[k], SG[k], k));
  P1 = sum(j = 1, JMAX, trig(CC[j], SS[j], j))*W;
  for(m = 1, 2*JMAX,
    my(u, v);
    if(m <= JMAX,
      u = SS[m] + sum(k = 1, m - 1, cc(m - k)*SG[k] + ss(m - k)*RH[k])
        + sum(k = 1, JMAX - m, -cc(m + k)*SG[k] + ss(m + k)*RH[k]);
      v = CC[m] + sum(k = 1, m - 1, cc(m - k)*RH[k] - ss(m - k)*SG[k])
        + sum(k = 1, JMAX - m, cc(m + k)*RH[k] + ss(m + k)*SG[k]),
      u = sum(k = m - JMAX, m - 1, cc(m - k)*SG[k] + ss(m - k)*RH[k]);
      v = sum(k = m - JMAX, m - 1, cc(m - k)*RH[k] - ss(m - k)*SG[k]));
    u += sum(k = 1, JMAX, cc(k)*SG[m + k] - ss(k)*RH[m + k]);
    v += sum(k = 1, JMAX, cc(k)*RH[m + k] + ss(k)*SG[m + k]);
    if(u != sinco(P1, m) || v != cosco(P1, m), bad++));
  print("U1/V1 rows that differ from the Laurent product, m = 1..2 JMAX: ", bad);
  must(bad == 0, "(2) holds: U1, V1 are the coefficients of (F - <F>) dlambda/dL exactly at JMAX = 12");

  U = vector(2*JMAX, j, rr()*t^max(0, j - JMAX)); V = vector(2*JMAX, j, rr()*t^max(0, j - JMAX));
  P2 = sum(j = 1, 2*JMAX, trig(U[j], V[j], j))*W;
  for(m = 1, JMAX,
    my(u = V[m] + sum(l = 1, m - 1, U[m - l]*SG[l] + V[m - l]*RH[l])
             + sum(l = 1, JMAX, -U[m + l]*SG[l] + U[l]*SG[m + l] + V[m + l]*RH[l] - V[l]*RH[m + l]),
       v = U[m] + sum(l = 1, m - 1, U[m - l]*RH[l] - V[m - l]*SG[l])
             + sum(l = 1, JMAX, U[m + l]*RH[l] + U[l]*RH[m + l] + V[m + l]*SG[l] + V[l]*SG[m + l]));
    foreach([u - sinco(P2, m), v - cosco(P2, m)], d, if(d != 0, low = min(low, valuation(d, t)))));
  print("U2/V2: lowest order of t left: ", low);
  must(low >= JMAX + 1, "(2) holds: U2, V2 are the coefficients of eta_a's series times dlambda/dL to t^(JMAX+1)");
}

\\ ---- (3) ------------------------------------------------------------------------------------------
\\ a = n = 1 and the rate cos L in a alone. eta_a solves d(eta)/d(lambda) = cos L - <cos L> =
\\ cos L + k and eta_lambda solves d(eta_lambda)/d(lambda) = -(3/2) eta_a, both of zero lambda-mean;
\\ d/d(lambda) = (a/r) d/dF and d/dF = i x d/dx. The closed forms are those of the .mac file, in the
\\ eccentric longitude F.
{
  my(kh = ellipse(bet, th), k = kh[1], h = kh[2], B = kh[3], b = 1/(1 + kh[3]), o = orbit(k, h, B),
     cF = o[1], sF = o[2], r = o[3], X = o[4],
     P = 1 - k^2 - b*h^2, Q = b*B*h*k, R = h*B/2, ddf = f -> I*x*deriv(f, x), etaa, etal);
  etaa = P*sF + Q*cF + R;
  etal = -3/2*(-P*cF + Q*sF - (k*P + h*Q)*sF^2/2 + (h*P - k*Q)*sF*cF/2 - k*R*sF + h*R*cF
    - (k*P - 3*h*Q)/4);
  must(ddf(etaa) - (X + k*r) == 0, "(3) holds: eta_a solves n d(eta_a)/d(lambda) = cos L - <cos L> for every ellipse");
  must(ddf(etal) + 3/2*etaa*r == 0, "(3) holds: eta_lambda solves n d(eta_lambda)/d(lambda) = -(3/2)(n/a) eta_a for every ellipse");
  must(meanf(etaa*r) == 0 && meanf(etal*r) == 0, "(3) holds: eta_a and eta_lambda have zero lambda-mean for every ellipse");
}
print("DONE");
quit
