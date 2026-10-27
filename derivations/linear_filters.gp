/* The linear filters and smoothers of spacelab-estimation in exact rationals, by routes other than
   the Maxima derivations use: the second engine for derivations/kalman_square_root.mac,
   rts_smoother.mac, consider_covariance.mac, hinf_riccati.mac and unscented_transform.mac, and a
   check on the 25-digit literals the tests carry.

   1. kalman: the fixture's time update and two measurement updates in Joseph form.
   2. rts / fixed_lag: the smoothed states as the conditional of the jointly Gaussian
      (x0..x3, z0..z3) built in covariance form -- Sigma_xz Sigma_zz^-1 -- not by inverting the
      information matrix.
   3. consider: the error covariance E[e e^T] of the weighted least-squares estimate.
   4. hinf: two steps in Simon's L = [I - g Qbar P + H^T V^-1 H P]^-1 form.
   5. unscented: the sigma points of the scalar quadratic at 60 digits.
   Every literal is compared at 1e-24 relative; a clean run prints `0 off` for each group.

       timeout 300 gp -q -f derivations/linear_filters.gp </dev/null                               */

default(realprecision, 60);

off = 0;
near(got, want) = if (abs(got - want) > 10^-24 * max(1, abs(want)), off++; printf("  departs: %.30g vs %.30g\n", got, want));
nearm(G, W) = for (i = 1, #G~, for (j = 1, #G, near(G[i, j], W[i, j])));
report(name) = printf("%s literals off by more than 1e-24: %d off\n", name, off); off = 0;

/* The Kalman update in Joseph form, on the globals x and P. */
upd(h, r, nu) = {
  my(f = h * P * h~ + r, K = P * h~ / f, A = matid(#P) - K * h);
  x = x + K * nu; P = A * P * A~ + K * r * K~; f;
}
/* Cov(x_i, x_j) = F^i P0 F^j~ + sum_{l < min(i,j)} F^(i-1-l) Q F^(j-1-l)~ on the 2-state arc. */
cxx(i, j) = F^i * P0 * (F^j)~ + sum(l = 0, min(i, j) - 1, F^(i - 1 - l) * Q * (F^(j - 1 - l))~);
cond(n, k) = {
  my(Sxz, Szz, zbar, X);
  Sxz = matrix(2, n, a, j, (cxx(k, j - 1) * h~)[a]);
  Szz = matrix(n, n, i, j, h * cxx(i - 1, j - 1) * h~ + R * (i == j));
  zbar = vector(n, j, h * F^(j - 1) * m0)~;
  X = F^k * m0 + Sxz * Szz^-1 * (vector(n, j, zs[j])~ - zbar);
  [X, cxx(k, k) - Sxz * Szz^-1 * Sxz~];
}
/* One H-infinity step in Simon's form, on the globals x and P. */
adv(z) = {
  my(L = (matid(2) - g * P + H~ * V^-1 * H * P)^-1, K = F * P * L * H~ * V^-1, nu = z - H * x);
  x = F * x + K * nu; P = F * P * L * F~ + SW * SW~; nu;
}
uq(x) = 3/2 * x^2 - 1/4 * x;
ut(al, be, ka) = {
  my(c = al^2 * (1 + ka), lam = c - 1, gm = sqrt(c), mu = 3/4, sg = 1/2, pts, wm, wc, ybar);
  pts = [mu, mu + gm * sg, mu - gm * sg];
  wm = [lam / c, 1 / (2 * c), 1 / (2 * c)];
  wc = wm + [1 - al^2 + be, 0, 0];
  ybar = sum(i = 1, 3, wm[i] * uq(pts[i]));
  [ybar, sum(i = 1, 3, wc[i] * (uq(pts[i]) - ybar)^2)];
}

/* 1. */
{
F = [1, 1/2, 0; 0, 1, 1/4; 0, 0, 1];
SQ = [1/2, 0, 0; 1/4, 1/2, 0; 0, 1/8, 1/4];
L0 = [2, 0, 0; 1, 3, 0; -1, 4, 5];
P = F * L0 * L0~ * F~ + SQ * SQ~;
x = F * [1, -2, 1/2]~;
nearm(P, [8.75, 8, 3.5; 8, 18.4375, 21.5625; 3.5, 21.5625, 42.078125]);
near(upd([1, -1/2, 1/4], 3/4, 5/4), 5.0986328125);
near(x[1], 1.379046159739513503160314); near(x[2], -8.522074315265274851561004e-1);
near(x[3], 1.293909212794483815361042);
nearm(P, [2.544292281172189235778587, 3.397433441869373683202452, -7.259145757517716912468876e-2;
          3.397433441869373683202452, 1.502392980271978548170849e1, 1.891282800229841026623252e1;
          -7.259145757517716912468876e-2, 1.891282800229841026623252e1, 4.00214039456042903658303e1]);
near(upd([0, 1, 1/2], 1/2, -3/8), 4.444210879141926833939858e1);
near(x[1], 1.350685069847914382779419); near(x[2], -1.058771187750069686725063);
near(x[3], 9.654746686740005635716894e-1);
nearm(P, [2.290090870988838835980774, 1.545995492005960378352578, -3.016361410967656435689435;
          1.545995492005960378352578, 1.539258416664062861951302, -2.527680150065346519718703;
          -3.016361410967656435689435, -2.527680150065346519718703, 5.931185751118648377542346]);
report("kalman");
}

/* 2. x_{k+1} = F x_k + w_k, z_k = h x_k + v_k: x_k = F^k x0 + sum F^(k-1-j) w_j. */
{
F = [1, 1/2; 0, 1]; SQ = [1/4, 0; 1/8, 1/2]; Q = SQ * SQ~; L0 = [2, 0; 1/2, 1]; P0 = L0 * L0~;
m0 = [1/2, -1]~; h = [1, 1/4]; R = 1/4; zs = [3/4, 1/2, 7/4, 5/2];
r = cond(4, 0);
near(r[1][1], 6.684961323778554565379192e-1); near(r[1][2], 3.149556632598205730376299e-1);
nearm(r[2], [2.097033427242537922125578e-1, -1.647966084820253359673447e-1;
             -1.647966084820253359673447e-1, 3.305303941963770943361815e-1]);
r = cond(4, 1);
near(r[1][1], 8.522036876279131786144677e-1); near(r[1][2], 6.81480154868890232229366e-1);
nearm(r[2], [1.166256804073796065676383e-1, -8.211563153291958509734778e-2;
             -8.211563153291958509734778e-2, 2.742566761824024071608762e-1]);
r = cond(3, 0);
near(r[1][1], 8.095648548712416228023548e-1); near(r[1][2], 4.188414104940134996456812e-2);
nearm(r[2], [2.271170282646867382189181e-1, -1.985048716162756098722616e-1;
             -1.985048716162756098722616e-1, 3.957806251649791765079919e-1]);
report("rts / fixed_lag");
}

/* 3. e = G (Hc (c - cbar) + n), G = P Hx~ R^-1. */
{
Hx = [1, 0; 1, 1; 1, 2]; Hc = [1, 0; 1/2, 1; 1/4, -1]; R = matdiagonal([1/4, 1/2, 1]);
Pcc = [2, 1/2; 1/2, 1];
P = (Hx~ * R^-1 * Hx)^-1; G = P * Hx~ * R^-1;
nearm(P, [3/13, -2/13; -2/13, 7/26]);
nearm(-G * Hc, [-9.807692307692307692307692e-1, -2.307692307692307692307692e-1;
                4.038461538461538461538462e-1, 1.538461538461538461538462e-1]);
nearm(G * (Hc * Pcc * Hc~ + R) * G~, [2.434171597633136094674556, -1.103550295857988165680473;
                                       -1.103550295857988165680473, 6.812130177514792899408284e-1]);
report("consider");
}

/* 4. Simon's form. */
{
F = [1, 1/2; 0, 1]; H = [1, 1/4; 0, 1]; V = [1/2, 1/8; 1/8, 1/4]; SW = [1/4, 0; 1/8, 1/2];
L0 = [2, 0; 1/2, 1]; P = L0 * L0~; x = [1/2, -1]~; g = 1/8;
nu = adv([3/4, 1/2]~); near(nu[1], 1/2); near(nu[2], 3/2);
near(x[1], 8.129339039467427484545887e-1); near(x[2], 2.545886828340466000951022e-1);
nearm(P, [6.127615311459819305753685e-1, 2.066185211602472658107466e-1;
          2.066185211602472658107466e-1, 4.763715525439847836424156e-1]);
nu = adv([7/4, -1/4]~); near(nu[1], 8.734189253447456015216358e-1); near(nu[2], -5.045886828340466000951022e-1);
near(x[1], 1.321976336106491573864834); near(x[2], -4.177421395864781145500628e-2);
nearm(P, [4.268644552841899385016474e-1, 1.720205549559947494042773e-1;
          1.720205549559947494042773e-1, 4.324307212715680222427528e-1]);
near(1 / vecmax(real(polroots(charpoly(L0 * L0~)))), 2.312040466020173954678145e-1);
report("hinf");
}

/* 5. */
{
r = ut(1, 0, 2); near(r[1], 1.03125); near(r[2], 1.28125);
r = ut(1/2, 2, 2); near(r[1], 1.03125); near(r[2], 1.3515625);
report("unscented");
}
