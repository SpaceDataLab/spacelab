/* Second engine for derivations/attitude_law_frames.mac: the frames of
   crates/spacelab-propagation/src/attitude_law.rs built at 100 digits.

   Each law is the module's construction, transcribed -- Qsw, Tnw, Vnc, Lvlh, YawSteering,
   SunPointing -- and evaluated on 200 pseudo-random states (a fixed linear congruential sequence,
   so the run is reproducible). Checked on every state: |M M' - I| and |det M - 1| under 1e-95,
   Lvlh's y = -h^ and z = -r^, YawSteering's body y perpendicular to the Sun with the Sun at
   positive body x, SunPointing's Sun on body +z; and the spin law's composition
   M(t0 + t1 + t2) = M(t0 + t1) base' M(t0 + t2) with Rodrigues' exp. Prints `PASS` lines; a failed
   check prints `FAIL`.

       timeout 300 gp -q -f derivations/attitude_law_frames.gp </dev/null                      */

default(realprecision, 100);
cross(a, b) = [a[2]*b[3] - a[3]*b[2], a[3]*b[1] - a[1]*b[3], a[1]*b[2] - a[2]*b[1]];
dot(a, b) = sum(i = 1, 3, a[i]*b[i]);
unit(v) = v/sqrt(dot(v, v));
rows(x, y, z) = matrix(3, 3, i, j, [x, y, z][i][j]);
{
laws(r, v, sun) =
  my(rh = unit(r), hh = unit(cross(r, v)), vh = unit(v), z = -rh, s = unit(sun), sperp, y);
  my(yy, xs);
  sperp = s - dot(s, z)*z;
  yy = unit(cross(z, unit(sperp)));
  xs = unit(cross(unit(cross(s, vh)), s));
  [rows(rh, cross(hh, rh), hh), rows(vh, cross(hh, vh), hh), rows(vh, hh, cross(vh, hh)),
   rows(cross(-hh, -rh), -hh, -rh), rows(cross(yy, z), yy, z),
   rows(xs, cross(s, xs), s)];
}
names = ["qsw", "tnw", "vnc", "lvlh", "yaw", "sun"];
seed = 20261026;
rnd() = seed = (6364136223846793005*seed + 1442695040888963407) % 2^64; seed/2^64 - 1/2;
vec3(scale) = scale*[rnd(), rnd(), rnd()];
nrm(M) = vecmax(apply(abs, concat(Vec(M))));
worst = vector(6); misc = 0;
{
for (k = 1, 200,
  my(r = vec3(14000000), v = vec3(16000), sun = vec3(3*10^11), L = laws(r, v, sun));
  for (j = 1, 6,
    my(M = L[j]);
    worst[j] = max(worst[j], max(nrm(M*M~ - matid(3)), abs(matdet(M) - 1))));
  misc = max(misc, nrm(Mat(L[4][2,] + unit(cross(r, v)))));
  misc = max(misc, nrm(Mat(L[4][3,] + unit(r))));
  misc = max(misc, abs(dot(L[5][2,], sun))/sqrt(dot(sun, sun)));
  if (dot(L[5][1,], sun) <= 0, misc = 1);
  misc = max(misc, nrm(Mat(L[6][3,] - unit(sun)))));
}
for (j = 1, 6, print(if (worst[j] < 10^-95, "PASS ", "FAIL "), names[j], " orthonormal and right-handed, worst ", Strprintf("%.3g", worst[j])));
print(if (misc < 10^-95, "PASS", "FAIL"), " lvlh y = -h, z = -r; yaw y perp Sun, Sun at +x; sun z on the Sun, worst ", Strprintf("%.3g", misc));

/* The spin law with Rodrigues' exp. */
skew(q) = [0, -q[3], q[2]; q[3], 0, -q[1]; -q[2], q[1], 0];
expso3(w) = my(t = sqrt(dot(w, w)), K = skew(w)); matid(3) + sin(t)/t*K + 2*sin(t/2)^2/t^2*K^2;
base = expso3([0.3, -1.1, 0.7]); ax = unit([1, 2, -2]); rate = 0.5; t0 = 3;
M(t) = base*expso3(-rate*(t - t0)*ax);
{
d = nrm(M(t0 + 1.25 + 4.5) - M(t0 + 1.25)*base~*M(t0 + 4.5));
print(if (d < 10^-95, "PASS", "FAIL"), " spin composition, worst ", Strprintf("%.3g", d));
d = nrm(Mat(M(t0 + 2.75)*ax~ - base*ax~));
print(if (d < 10^-95, "PASS", "FAIL"), " spin fixes the inertial axis in the body, worst ", Strprintf("%.3g", d));
}
