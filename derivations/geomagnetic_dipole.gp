/* Second engine for derivations/geomagnetic_dipole.mac: the centred-dipole reference values from
   spherical-trigonometric closed forms rather than the vector constructions Maxima evaluates,
       pole latitude asin(-g10/M), pole longitude atan2(-h11, -g11), M = |(g10, g11, h11)|,
       magnetic latitude   asin(sin f sin f_p + cos f cos f_p cos(l - l_p)),
       magnetic longitude  atan2(cos f sin(l - l_p), cos f sin f_p cos(l - l_p) - sin f cos f_p),
       meridian rotation   (cos f sin f_p - sin f cos f_p cos(l_p - l), cos f_p sin(l_p - l)), normalized,
   with g10, g11, h11 and their secular variation read as exact decimals from the IAGA file, and
   compared with the values Maxima printed (pasted below).

   Run from derivations/, as verify.sh does:
       timeout 300 gp -q -f geomagnetic_dipole.gp </dev/null                       */

default(realprecision, 60);

words(s) = select(w -> w != "", strsplit(s, " "));
{decimal(t) = my(ch = Vec(t), neg = (ch[1] == "-"), u, parts, v);
  u = if (neg, strjoin(ch[2..#ch]), t);
  parts = strsplit(u, ".");
  v = if (#parts == 1, eval(parts[1]), eval(parts[1]) + eval(parts[2]) / 10^#parts[2]);
  if (neg, -v, v);}

{coef = Map();
foreach (readstr("../crates/spacelab-gnss/tests/fixtures/igrf14coeffs.txt"), l,
  my(t = words(l));
  if (#t > 3 && t[1] == "g/h", for (k = 1, #t, if (t[k] == "2025.0", c25 = k); if (t[k] == "2025-30", csv = k)));
  if (#t > 3 && t[2] == "1", mapput(coef, concat([t[1], t[2], t[3]]), [decimal(t[c25]), decimal(t[csv])])));}

{pole(year) = my(dt = year - 2025, g10, g11, h11, m);
  g10 = mapget(coef, "g10")[1] + dt * mapget(coef, "g10")[2];
  g11 = mapget(coef, "g11")[1] + dt * mapget(coef, "g11")[2];
  h11 = mapget(coef, "h11")[1] + dt * mapget(coef, "h11")[2];
  m = sqrt(g10^2 + g11^2 + h11^2);
  [asin(-g10 / m), atan2(-h11, -g11), m];}
atan2(y, x) = if (x > 0, atan(y / x), if (y >= 0, atan(y / x) + Pi, atan(y / x) - Pi));

{coords(f, l) = my(p = pole(2025), fp = p[1], lp = p[2], c, s, n);
  c = cos(f) * sin(fp) - sin(f) * cos(fp) * cos(lp - l);
  s = cos(fp) * sin(lp - l);
  n = sqrt(c^2 + s^2);
  [asin(sin(f) * sin(fp) + cos(f) * cos(fp) * cos(l - lp)),
   atan2(cos(f) * sin(l - lp), cos(f) * sin(fp) * cos(l - lp) - sin(f) * cos(fp)),
   c / n, s / n];}

{maxima_poles = [[1.410040345385084652804351241132443450879, -1.269950826609024893067310314044349777553, 2.973336537191846368005532461373558571173e4],
                 [1.412537568442043365173371321476978231379, -1.272330406576405612981672275427979592967, 2.967676464782675732974950516204943686405e4]];}
{maxima_points = [[9.107179692664895196843757699506943027127e-1, 2.380742697044007398825410601065239014113e-2, 9.999864402122356710807482700401141474398e-1, -5.207628218374812599097992029044542426162e-3],
                  [-6.592931672066791639928741206656271586312e-1, -2.99898909685567668133232270057321063092, 9.99663974844294220462510269811763792182e-1, 2.592175531298549970164206857204437974011e-2]];}

{off = 0; worst = 0.;
my(years = [2025, 4057/2], pts = [[3/4, -5/4], [-1/2, 2]]);
for (i = 1, 2, my(p = pole(years[i]));
  for (k = 1, 3, my(d = abs(p[k] - maxima_poles[i][k]) / max(1, abs(p[k]))); worst = max(worst, d); off += d > 1e-35));
for (i = 1, 2, my(q = coords(pts[i][1], pts[i][2]));
  for (k = 1, 4, my(d = abs(q[k] - maxima_points[i][k])); worst = max(worst, d); off += d > 1e-35));}
printf("%d of 14 dipole values off by more than 1e-35 from Maxima; largest gap %.3e\n", off, worst);
