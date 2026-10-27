/* Second engine for derivations/gmst82.mac: GMST (IAU 1982) at the five reference epochs from the
   printed polynomial itself, with the 876600 h whole-day term kept and T formed directly from the
   nanosecond count, in exact rationals reduced modulo one day -- the route the code avoids -- and
   compared with Maxima's values of the coded split (pasted below). Also re-checks that the four
   coefficient bit patterns decode to the nearest doubles of the published decimals.

       timeout 300 gp -q -f derivations/gmst82.gp </dev/null                                     */

default(realprecision, 60);

C0 = 6731054841/100000; C1 = 8640184812866/1000000; C2 = 93104/1000000; C3 = -62/10000000;
nsday = 86400 * 10^9;
printed(T) = C0 + (876600 * 3600 + C1) * T + C2 * T^2 + C3 * T^3;
gmst(ns) = my(th = printed(ns / nsday / 36525)); 2 * Pi * (th - 86400 * floor(th / 86400)) / 86400;

epochs = [0, -232415160 * 10^9, 900075 * nsday / 100, 36525 * nsday + 12 * 10^9, -(1826250 * 86400 + 25920) * 10^9];
{maxima = [4.894961212823058751375704431175952857581, 2.663002216713348436839310469824036728281,
           1.082560959309939194850505924209108106535, 4.909283007871094720178042192396826163741,
           2.349829389055239016345653714577592776064];}

{off = 0; worst = 0.;
for (i = 1, #epochs, my(d = abs(gmst(epochs[i]) - maxima[i])); worst = max(worst, d); off += d > 1e-35);}
printf("%d of %d epochs off by more than 1e-35 rad from Maxima; largest gap %.3e rad\n", off, #epochs, worst);

{decode(b) = my(s = b >> 63, e = (b >> 52) % 2^11, m = b % 2^52); [(-1)^s * (2^52 + m) * 2^(e - 1075), 2^(e - 1075)];}
{bits = [0x40F06EE8C649906D, 0x41607AD71A02FF8F, 0x3FB7D5A9EB2074EB, 0xBEDA013305E6C9CE];
coefs = [C0, C1, C2, C3]; far = 0;
for (i = 1, 4, my(d = decode(bits[i])); far += abs(d[1] - coefs[i]) >= d[2] / 2);}
printf("coefficient bit patterns farther than half an ulp from their decimal: %d\n", far);
