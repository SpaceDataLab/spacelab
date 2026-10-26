/* Second engine for derivations/cone_footprint.mac, and the reference rows the gates in
   crates/spacelab-sensors/tests/cone_gates.rs and tests/agile.rs are held to.

   On the sphere R = 6378137 m, for each altitude h and half-angle alpha, at 120 digits:
   the cone edge lam_c = asin(L sin alpha) - alpha (present only while L sin alpha <= 1), the
   horizon edge lam_h = acos(1/L), and the agile edge min(lam_c, lam_h). The line of sight
   w = G - S from the satellite S = (R + h)(sin lam, 0, cos lam) to the polar cell G is built
   from the vectors, and the off-nadir angle measured from them: cos(theta) - cos(alpha) must
   vanish at lam_c, and the cell's rise d . zhat at lam_h. The tangency L sin alpha = 1, where
   the two edges meet, is one row per altitude. Then 1/(1 - e^2) = 1/(1 - f)^2 for the WGS-84
   f = 1/298.257223563, exactly, as `cone.rs`'s INVERSE_ONE_MINUS_E2 should round.

   Rows (h_m, alpha_rad, lam_cone_rad or -1 when the cone overhangs the limb, lam_horizon_rad).

       gp -q -f derivations/cone_footprint.gp </dev/null                                         */

default(realprecision, 120);
lit(x) = if (type(x) == "t_INT", Str(x, ".0"), x == 0, "0.0", my(e = floor(log(abs(x))/log(10))); Strprintf("%.24fe%d", x/10^e, e));

R = 6378137;
hs = [500000, 800000, 2000000, 35786000];
worst_cone = 0; worst_horizon = 0; rows = List();
{
off_nadir(lam, L) =
  my(S = L*R*[sin(lam), 0, cos(lam)], w = [0, 0, R] - S, b = -S/sqrt(S*S~));
  (w*b~)/sqrt(w*w~);
}
rise(lam, L) = L*R*cos(lam) - R;
{
for (i = 1, #hs,
  my(h = hs[i], L = (R + h)/R, lh = acos(1/L), alphas);
  alphas = [10*Pi/180, 30*Pi/180, 50*Pi/180, 60*Pi/180, asin(1/L)];
  worst_horizon = max(worst_horizon, abs(rise(lh, L))/R);
  for (j = 1, #alphas,
    my(a = alphas[j], lc = -1);
    \\ the tangency row is set exactly: L sin(asin(1/L)) rounds either side of 1
    if (j == #alphas, lc = Pi/2 - a, L*sin(a) < 1, lc = asin(L*sin(a)) - a);
    if (lc != -1, worst_cone = max(worst_cone, abs(off_nadir(lc, L) - cos(a))));
    listput(rows, Str("    (", lit(h), ", ", lit(a), ", ", if (lc == -1, "-1.0", lit(lc)), ", ",
                      lit(lh), "),"))));
}
print("cone edge |cos(theta) - cos(alpha)| worst: ", lit(worst_cone));
print("horizon edge |rise|/R worst: ", lit(worst_horizon));
for (k = 1, #rows, print(rows[k]));
f = 1/(298257223563/10^9);
print("inverse_one_minus_e2: ", lit(1/(1 - f*(2 - f))));
print("inverse_one_minus_e2 is 1/(1 - f)^2 exactly: ", 1/(1 - f*(2 - f)) == 1/(1 - f)^2);
