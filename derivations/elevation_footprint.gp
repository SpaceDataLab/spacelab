/* Second engine for derivations/elevation_footprint.mac, and the reference rows the gate
   `the_mask_footprint_edge_is_where_the_closed_form_puts_it` in
   crates/spacelab-sensors/tests/elevation_gates.rs is held to.

   On the sphere R = 6378137 m, for each altitude h and mask eps, the edge's Earth-central angle
   lam = pi/2 - eps - asin(cos(eps)/L), L = (R + h)/R, is evaluated at 120 digits; the line of
   sight from the polar cell to S = (R + h)(sin lam, 0, cos lam) is then built *from the
   vectors*, not from the closed form, and its rise t = d . zhat and length |d| checked:
   t - sin(eps)|d| must vanish (the defining equation) and |d| must equal the slant range
   sqrt((R + h)^2 - (R cos eps)^2) - R sin eps. The worst of each over the grid is printed, then
   one Rust row (h_m, eps_deg, lam_rad, slant_m) per point.

       gp -q -f derivations/elevation_footprint.gp </dev/null                                     */

default(realprecision, 120);
lit(x) = if (type(x) == "t_INT", Str(x, ".0"), x == 0, "0.0", my(e = floor(log(abs(x))/log(10))); Strprintf("%.24fe%d", x/10^e, e));

R = 6378137;
hs = [500000, 800000, 2000000, 20200000, 35786000];
masks = [0, 5, 10, 25, 60, 89];
worst_g = 0; worst_d = 0; rows = List();
{
for (i = 1, #hs, for (j = 1, #masks,
  my(h = hs[i], eps = masks[j]*Pi/180, L = (R + h)/R, lam, S, d, t, nd, slant);
  lam = Pi/2 - eps - asin(cos(eps)/L);
  S = (R + h)*[sin(lam), 0, cos(lam)];
  d = S - [0, 0, R];
  t = d[3];
  nd = sqrt(d*d~);
  slant = sqrt((R + h)^2 - (R*cos(eps))^2) - R*sin(eps);
  worst_g = max(worst_g, abs(t - sin(eps)*nd)/nd);
  worst_d = max(worst_d, abs(nd - slant)/nd);
  listput(rows, Str("    (", lit(h), ", ", lit(masks[j]), ", ", lit(lam), ", ", lit(slant), "),"))));
}
print("elevation edge residual |t - sin(eps)|d||/|d| worst: ", lit(worst_g));
print("elevation edge slant range |d| - d(eps) worst relative: ", lit(worst_d));
for (k = 1, #rows, print(rows[k]));
