/* Second engine for derivations/comms_link_budget.mac, and the reference values the gates in
   crates/spacelab-sensors/tests/comms_gates.rs are held to, at 120 digits.

   -10 log10(k) for the exact SI k = 1.380649e-23 J/K (Maxima prints the same term); the slant
   range from the triangle itself -- the satellite placed at Earth-central angle
   lam = pi/2 - e - asin(R cos(e)/r_s) and the distance measured from the vectors, not from the
   closed form comms.rs evaluates; and C/N0 = EIRP + G/T - 20 log10(4 pi d f/c) - L_atm/sin(e)
   - 10 log10(k) for one X-band budget (EIRP 20 dBW, G/T 10 dB/K, f = 8.2 GHz, L_atm 0.5 dB,
   h = 700 km, R = 6378137 m, c = 299792458 m/s). The margin's slope dM/de at 30 deg bounds how far
   a bisected break-even elevation may sit from the exact root.

       gp -q -f derivations/comms_link_budget.gp </dev/null                                     */

default(realprecision, 120);
lit(x) = if (type(x) == "t_INT", Str(x, ".0"), x == 0, "0.0", my(e = floor(log(abs(x))/log(10))); Strprintf("%.24fe%d", x/10^e, e));

R = 6378137; h = 700000; rs = R + h; cl = 299792458;
eirp = 20; gt = 10; f = 82*10^8; latm = 1/2;
kdb = -10*log(1380649/10^29)/log(10);
{
slant(e) =
  my(lam = Pi/2 - e - asin(R*cos(e)/rs), S = rs*[sin(lam), 0, cos(lam)], d = S - [0, 0, R]);
  sqrt(d*d~);
}
closed(e) = sqrt(rs^2 - (R*cos(e))^2) - R*sin(e);
cn0(e) = eirp + gt - 20*log(4*Pi*slant(e)*f/cl)/log(10) - latm/sin(e) + kdb;

print("boltzmann_db: ", lit(kdb));
worst = 0;
{
forstep (deg = 0, 90, 5,
  my(e = deg*Pi/180); worst = max(worst, abs(slant(e) - closed(e))/slant(e)));
}
print("slant range, triangle vs closed form, worst relative over 0..90 deg: ", lit(worst));
print("slant_zenith_m: ", lit(slant(Pi/2)), "  (= h exactly: ", abs(slant(Pi/2) - h) < 10^-100, ")");
{
print("slant_horizon_m: ", lit(slant(0)), "  (= sqrt(rs^2 - R^2): ",
      abs(slant(0) - sqrt(rs^2 - R^2)) < 10^-100, ")");
}
print("slant_30_m: ", lit(slant(Pi/6)));
print("cn0_5_dbhz: ", lit(cn0(5*Pi/180)));
print("cn0_30_dbhz: ", lit(cn0(Pi/6)));
print("cn0_90_dbhz: ", lit(cn0(Pi/2)));
print("margin_slope_30_db_per_rad: ", lit(derivnum(e = Pi/6, cn0(e))));
