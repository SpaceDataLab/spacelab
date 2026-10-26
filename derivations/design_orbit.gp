/* The design rows of derivations/design_orbit.mac, recomputed in PARI/GP at 60 digits: the second
   engine for the repeat-ground-track semi-major axis (PARI's own root finder, solve(), on the
   defining resonance Mdot + wdot = q (w_E - Wdot) written from the J2 rates) and the
   Sun-synchronous inclination acos(-rate / (3/2 n J2 (Re/p)^2)). Inputs are the doubles the Rust
   test passes, as exact rationals. Each row is printed beside its relative difference from the
   Maxima value, which this file carries as a literal; `agree` lines must all read below 1e-35.

       timeout 120 gp -q -f derivations/design_orbit.gp </dev/null                                */

default(realprecision, 60);

dbl(x) = {
  my(r = x * 1., e);
  if (r == 0, return(0));
  e = exponent(r);
  round(x * 2^(52 - e)) * 2^(e - 52);
}

mu = 398600441800000; Re = 6378137;
J2 = dbl(108262668 / 10^11); wE = dbl(72921151467 / 10^15);

rates(a, e, i) = {
  my(p = a * (1 - e^2), n = sqrt(mu / a^3), s = J2 * (Re / p)^2, c = cos(i));
  [n * (1 + 3/4 * s * sqrt(1 - e^2) * (3 * c^2 - 1)), 3/4 * n * s * (5 * c^2 - 1), -3/2 * n * s * c];
}
resonance(q, a, e, i) = my(r = rates(a, e, i)); r[1] + r[2] - q * (wE - r[3]);
rgt(revs, days, e, i) = {
  my(q = revs / days, a0 = (mu / (q * wE)^2)^(1/3));
  solve(a = 0.98 * a0, 1.02 * a0, resonance(q, a, e, i));
}

sun = 2 * Pi / (3652422 / 10000 * 86400);
sso(a, e) = my(p = a * (1 - e^2), n = sqrt(mu / a^3)); acos(-sun / (3/2 * n * J2 * (Re / p)^2));

agree(name, x, maxima) = printf("%s %.40g  agree %.3g\n", name, x, abs(x / maxima - 1));

{
  agree("RGT 233 16 0       ", rgt(233, 16, 0, 1755/1024), 7.077740718304228457322960926755511792865e6);
  agree("RGT 233 16 1/1024  ", rgt(233, 16, 1/1024, 1755/1024), 7.077740733383850964951490328493072397903e6);
  agree("RGT 233 16 1/16    ", rgt(233, 16, 1/16, 1755/1024), 7.077802863724393745586247823771917425021e6);
  agree("RGT 233 16 1/8     ", rgt(233, 16, 1/8, 1755/1024), 7.077993951498357218928617346359076227893e6);
  agree("RGT 15 1 0         ", rgt(15, 1, 0, 3/4), 6.868598118923256556684730974066775492612e6);
  agree("RGT 43 3 0         ", rgt(43, 3, 0, 1741/1024), 7.151347576037380331300768835620952423845e6);
  agree("RGT 2 1 23/32      ", rgt(2, 1, 23/32, 1133/1024), 2.655443149673837252331192019419302838905e7);
  agree("SSO 300 km 0       ", sso(Re + 300000, 0), 1.687244913105786831811427295084030264679);
  agree("SSO 600 km 0       ", sso(Re + 600000, 0), 1.706716801071948569342336811167370340927);
  agree("SSO 600 km 1/128   ", sso(Re + 600000, 1/128), 1.706700106810268414267368064116952871387);
  agree("SSO 900 km 13/64   ", sso(Re + 900000, 13/64), 1.715623680818809792688484998300690713182);
  agree("SSO 5000 km 0      ", sso(Re + 5000000, 0), 2.418979587254147953576282926677113868961);
}

/* The critical inclination acos(1/sqrt 5), where derivations/design_orbit.mac shows wdot = 0. */
printf("critical inclination acos(1/sqrt(5)) = %.40g rad\n", acos(1 / sqrt(5)));
