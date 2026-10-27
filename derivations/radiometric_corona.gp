/* The solar-corona range correction from its defining path integral, at 50 digits: the second
   engine for derivations/radiometric_observables.mac and the reference values the
   `solar_corona_delay` gate in crates/spacelab-estimation/src/radiometric.rs asserts against.

       delay = (K / f^2) * int over the straight leg of n_e(r) ds,
       n_e(r) = sum_j c_j (r / R_sun)^(-k_j),   K = e^2 / (8 pi^2 eps0 m_e),

   in the arc-length variable by intnum -- no substitution r = p / cos(theta), no recursion, no
   tanh-sinh -- with R_sun = 696e6 m the model's own radius and e, eps0, m_e CODATA 2018 (e exact
   by the 2019 SI). The legs are integer-metre heliocentric positions in no coordinate plane, an
   Earth station at 1 AU and a spacecraft 1.5 AU beyond the Sun, passing 10 and 3 solar radii from
   it. The column is also taken in the closed form r = p / cos(theta) for integer exponents, as a
   check of the substitution itself.

       timeout 300 gp -q -f derivations/radiometric_corona.gp </dev/null                             */

default(realprecision, 60);
Rs = 696 * 10^6;
qe = 1602176634 / 10^28; me = 91093837015 / 10^41; eps0 = 88541878128 / 10^22;
K = qe^2 / (8 * Pi^2 * eps0 * me);
nrm(v) = sqrt(v * v~);

column(series, A, B) = {
  my(d = B - A, L = nrm(d), u = d / L);
  /* Split at the perpendicular foot, where the integrand peaks, and at decades of the impact
     parameter p either side of it, where it falls from its peak to its tails. */
  my(sf = -(A * u~), p = nrm(A + sf * u),
     f = (s) -> my(r = nrm(A + s * u)); sum(j = 1, #series, series[j][1] * (r / Rs)^(-series[j][2])));
  my(cuts = List([0, L, sf]));
  for (j = -1, 3, my(w = p * 10^j); if (sf - w > 0, listput(cuts, sf - w)); if (sf + w < L, listput(cuts, sf + w)));
  cuts = vecsort(Vec(cuts));
  sum(i = 1, #cuts - 1, intnum(s = cuts[i], cuts[i + 1], f(s)));
}

/* Closed form: c R^k p^(1-k) int cos^(k-2), theta from the receiver's angle to the transmitter's. */
column_closed(series, A, B) = {
  my(d = B - A, L = nrm(d), u = d / L, sf = -(A * u~), p = nrm(A + sf * u));
  my(t0 = atan(-sf / p), t1 = atan((L - sf) / p));
  sum(j = 1, #series, my(k = series[j][2]);
    series[j][1] * Rs^k * p^(1 - k) * intnum(t = t0, t1, cos(t)^(k - 2)));
}

receiver = [42742248771, 64113373157, 128226746314];
{
legs = [["10 R_sun", [-57422602848, -82788519117, -201261146550]],
        ["3 R_sun", [-62108120287, -92159553995, -195013789965]]];
series = [["k = 2", [[10^14, 2]]], ["k = 3", [[10^14, 3]]], ["k = 6", [[10^14, 6]]],
          ["k = 2.7", [[10^14, 27/10]]],
          ["three-term", [[10^14, 2], [3 * 10^15, 6], [10^14, 27/10]]]];
}
f = 84 * 10^8;

printf("K = e^2/(8 pi^2 eps0 m_e) = %s m^3/s^2\n", Strprintf("%.40e", K));
{
my(bad = 0);
for (i = 1, #legs, for (j = 1, #series,
  my(A = receiver, B = legs[i][2], col = column(series[j][2], A, B), cl = column_closed(series[j][2], A, B));
  if (abs(cl - col) > 10^-45 * col, bad++);
  printf("  %-8s %-10s X-band group delay %s m   closed form - integral %.1e relative\n",
         legs[i][1], series[j][1], Strprintf("%.40e", K * col / f^2), (cl - col) / col)));
printf("%d of %d columns with the cos-power closed form off the arc-length integral by more than 1e-45\n",
       bad, #legs * #series);
}
