/* The reference carrier of a broadcast Klobuchar set, in PARI/GP: the second engine for
   derivations/klobuchar_reference.mac.

   The carriers as exact rationals of 10.23 MHz, gamma_51 = (115/154)^2 = 13225/23716 and the S-band
   factor; the first-order dispersion law D(f) = K N/f^2 as a PARI rational function, against the
   scale (f_ref/f)^2 the function applies; then ISRO-IRNSS-ICD-SPS-1.1 Appendix H evaluated at 60
   digits on the fixture's NavIC set at the three geometries of the Maxima file, against the goldens
   that file prints and the module test holds. Every mismatch prints `FAIL`.

       timeout 300 gp -q -f derivations/klobuchar_reference.gp </dev/null                           */

default(realprecision, 60);

bad = 0;
iszero(name, e) = if (e != 0, bad++; printf("FAIL %s: %s\n", name, e), printf("%s: 0\n", name));

\\ 1. The carriers, MHz, and the two factors.
f0 = 1023/100; fl1 = 154*f0; fl5 = 115*f0; fs = 1218/5*f0;
iszero("l1_is_1575_42_mhz", fl1 - 157542/100);
iszero("l5_is_1176_45_mhz", fl5 - 117645/100);
iszero("s_is_2492_028_mhz", fs - 2492028/1000);
gamma51 = (fl5/fl1)^2;
iszero("gamma_51_is_13225_over_23716", gamma51 - 13225/23716);
iszero("s_factor_is_330625_over_1483524", (fl5/fs)^2 - 330625/1483524);

\\ 2. The scale is the dispersion law, in the variables k, n, f, fr.
disp(f) = k*n/f^2;
scale(fr, f) = my(ratio = fr/f); ratio*ratio;
iszero("scale_is_the_dispersion_law", disp(f) - disp(fr)*scale(fr, f));
iszero("navic_l1_is_gamma_51_of_navic_l5", t*scale(fl5, fl1) - gamma51*t*scale(fl5, fl5));
iszero("navic_s_is_the_icd_s_factor", t*scale(fl5, fs) - (fl5^2/fs^2)*t*scale(fl5, fl5));

\\ 3. Appendix H at 60 digits. Semicircles in, trigonometric arguments in radians, branches
\\ decided on the value; the NavIC set is the fixture's `I02 LNAV` record on Table 17's grid.
na = [45*2^-30, 35*2^-27, -52*2^-24, 16*2^-24];
nb = [82*2^11, 22*2^14, -127*2^16, 127*2^16];
cc = 299792458;
appendix_h(lat, lon, el, az, sod) = {
  my(e = el/Pi, phu = lat/Pi, lau = lon/Pi, psi, phi, lam, phm, tl, ff, amp, per, x, tt);
  psi = 137/10000/(e + 11/100) - 22/1000;
  phi = phu + psi*cos(az);
  if (phi > 416/1000, phi = 416/1000);
  if (phi < -416/1000, phi = -416/1000);
  lam = lau + psi*sin(az)/cos(Pi*phi);
  phm = phi + 64/1000*cos(Pi*(lam - 1617/1000));
  tl = 43200*lam + sod;
  tl -= 86400*floor(tl/86400);
  ff = 1 + 16*(53/100 - e)^3;
  amp = sum(j = 0, 3, na[j + 1]*phm^j); if (amp < 0, amp = 0);
  per = sum(j = 0, 3, nb[j + 1]*phm^j); if (per < 72000, per = 72000);
  x = 2*Pi*(tl - 50400)/per;
  tt = if (abs(x) < 157/100, ff*(5/10^9 + amp*(1 - x^2/2 + x^4/24)), ff*5/10^9);
  tt*cc;
}
check(name, x, want) = {
  my(d = abs(x - want)/abs(want));
  if (d > 1e-40, bad++; printf("FAIL %s: %.45g vs %.45g\n", name, x, want),
                 printf("%s: ok (rel %.2e)\n", name, d));
}
\\ The goldens klobuchar_reference.mac prints, to 45 digits.
{
cases = [["day", 29/128, 173/128, 3/4, 5/2, 31000,
          20.0206985018162277792662414082700253010660329926, 11.1643505518012992233427240101353973944424981585],
         ["night", 29/128, 173/128, 3/4, 5/2, 70000,
          2.09159709863839461879956046494817889801356731768, 1.16635906685329603784888628558524481051734810998],
         ["clamped", 5/4, 1/2, 1/8, 1/4, 50000,
          4.32423613542536476555661847130861025416320811314, 2.41136881813967148863578509373656479217863161141]];
}
{
  for (i = 1, #cases,
    my(c = cases[i], d5 = appendix_h(c[2], c[3], c[4], c[5], c[6]));
    check(Str(c[1], "_l5"), d5, c[7]);
    check(Str(c[1], "_l1"), d5*13225/23716, c[8]));
}
printf("klobuchar reference: %d checks off\n", bad);
quit;
