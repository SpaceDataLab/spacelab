/* Second engine for derivations/sdtm_solar_wind.mac: Newell's coupling
   N = V^(4/3) B_T^(2/3) |sin(theta/2)|^(8/3), theta = atan2(By, Bz) = arg(Bz + i By), at 60 digits
   against the reference values Maxima prints, and the half-angle form
   sin^2(theta/2) = (1 - Bz/B_T)/2 at points in every quadrant. Prints `PASS` lines; a failed check
   prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_solar_wind.gp </dev/null                          */

default(realprecision, 60);
newell(v, y, z) = v^(4/3) * sqrt(y^2 + z^2)^(2/3) * abs(sin(arg(z + I*y)/2))^(8/3);
w = [[400, 3, -4], [650, -2, -12], [350, 4, 3], [800, 0, -20]];
ref = [7488.30176298685481230820456769, 29513.3759852484023348412770815, \
  843.549792461330495144929383662, 54719.2302936543036592994839214];
for (j = 1, 4, my(n = newell(w[j][1], w[j][2], w[j][3])); \
  print("N at ", w[j], if (abs(n/ref[j] - 1) < 10^-28, ": PASS", ": FAIL"), " ", Strprintf("%.30g", n)));
probe = [[3, -4], [-3, -4], [3, 4], [-3, 4], [0, 5], [0, -5], [5, 0], [-5, 0], [1/10, -7], [-7, 1/10]];
worst = vecmax(vector(#probe, j, my(y = probe[j][1], z = probe[j][2]); \
  abs(sin(arg(z + I*y)/2)^2 - (1 - z/sqrt(y^2 + z^2))/2)));
print(if (worst < 10^-55, "half-angle form in every quadrant: PASS", "half-angle form in every quadrant: FAIL"));
