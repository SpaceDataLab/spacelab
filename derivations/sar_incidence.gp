/* Second engine for derivations/sar_incidence.mac, and the reference rows the gates in
   crates/spacelab-sensors/tests/sar_gates.rs are held to, at 120 digits.

   On the sphere R = 6378137 m, for each altitude h and off-nadir look angle eta, the look reaches
   the ground at lam = asin(L sin eta) - eta; the satellite is placed at that central angle from
   the polar cell, and the incidence measured from the vectors, acos(t/|d|). It must equal
   eta + lam. Rows (h_m, eta_rad, theta_inc_rad). Then look_side for one non-circular state, as
   (c - r) . (dhat x vhat) from the vectors, against -c . (r x v)/(|r| |v|).

       gp -q -f derivations/sar_incidence.gp </dev/null                                         */

default(realprecision, 120);
lit(x) = if (type(x) == "t_INT", Str(x, ".0"), x == 0, "0.0", my(e = floor(log(abs(x))/log(10))); Strprintf("%.24fe%d", x/10^e, e));
cross(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
nrm(a) = sqrt(a*a~);

R = 6378137;
hs = [500000, 693000, 2000000];
etas = [15, 20, 29, 37, 46];
worst = 0; rows = List();
{
for (i = 1, #hs, for (j = 1, #etas,
  my(h = hs[i], L = (R + h)/R, eta = etas[j]*Pi/180, lam, S, d, inc);
  lam = asin(L*sin(eta)) - eta;
  S = (R + h)*[sin(lam), 0, cos(lam)];
  d = S - [0, 0, R];
  inc = acos(d[3]/nrm(d));
  worst = max(worst, abs(inc - (eta + lam)));
  listput(rows, Str("    (", lit(h), ", ", lit(eta), ", ", lit(inc), "),"))));
}
print("incidence acos(t/|d|) - (eta + lam) worst: ", lit(worst));
for (k = 1, #rows, print(rows[k]));

r = [6800000, 1200000, 300000]; v = [-1000, 6500, 3000]; c = [6000000, 2000000, -1500000];
side = (c - r)*cross(-r/nrm(r), v/nrm(v))~;
{
print("look_side_m: ", lit(side), "   = -c.(r x v)/(|r||v|): ",
      abs(side + c*cross(r, v)~/(nrm(r)*nrm(v))) < 10^-100);
}
