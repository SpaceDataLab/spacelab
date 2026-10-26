/* Second engine for derivations/scan_catalogue.mac, and the reference values the gate in
   crates/spacelab-sensors/tests/scan_gates.rs is held to, at 120 digits.

   The swath-edge angles each instrument's closed form must reach at its first and last sample
   (rows: (pixels, first_roll_rad, last_roll_rad)), in the order the test lists the instruments;
   the VIIRS detector-column half-extent atan2(11.87/2, 824); and the ground displacement of the
   AVHRR swath edge that the GAC pixel-centre factor (1023.5 - 3.5)/1023.5 makes, on a 6371 km
   sphere at 833 and 870 km, from the near-side look geometry lam = asin(L sin eta) - eta.

       gp -q -f derivations/scan_catalogue.gp </dev/null                                        */

default(realprecision, 120);
lit(x) = if (type(x) == "t_INT", Str(x, ".0"), x == 0, "0.0", my(e = floor(log(abs(x))/log(10))); Strprintf("%.24fe%d", x/10^e, e));
deg = Pi/180;

/* Avhrr, AvhrrGac, Viirs, AmsuA, Mhs, Hirs4, Atms, Mwhs2: +H at sample 0, -H at sample n - 1. */
{
sym = [[2048, 5537/100], [409, 5537/100*1020/10235*10], [6400, 5628/100], [30, 483/10],
       [90, 49444/1000], [56, 495/10], [96, 527/10], [98, 5335/100]];
}
for (k = 1, #sym, print("    (", sym[k][1], ", ", lit(sym[k][2]*deg), ", ", lit(-sym[k][2]*deg), "),"));
/* Olci, SlstrNadir: 46.5 deg to -22.1 deg. Ascat: 53 to 25 deg, then -25 to -53 deg. */
print("    (4000, ", lit(465/10*deg), ", ", lit(-221/10*deg), "),");
print("    (3000, ", lit(465/10*deg), ", ", lit(-221/10*deg), "),");
print("    (42, ", lit(53*deg), ", ", lit(-53*deg), "),");
print("ascat_inner_rad: ", lit(25*deg));
print("viirs_pitch_extent_rad: ", lit(atan(1187/200/824)));

lam(L, eta) = asin(L*sin(eta)) - eta;
{
for (i = 1, 2,
  my(h = [833000, 870000][i], L = (6371000 + h)/6371000);
  print("gac_edge_displacement_m at ", h, " m: ",
        lit(6371000*(lam(L, 5537/100*deg) - lam(L, 5537/100*1020/10235*10*deg)))));
}
