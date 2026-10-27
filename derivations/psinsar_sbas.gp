/* Second engine for derivations/psinsar_sbas.mac, at 60 digits.

   1. The closed-form 2x2 least-squares pair `sbas_invert` returns, against PARI's own solve of the
      normal equations (matsolve), on 100 pseudo-random over-determined stacks of 6 .. 15 rows.
   2. The DEM-error coefficient from the exact flat-earth geometry: the height derivative of
      R1 - R2 inside a range cell, by a central difference at 60 digits, against the first-order
      B_perp/(r sin theta), for r = 800 km, theta = 38 deg and baselines of 10 m to 1 km. The
      relative gap is printed divided by B/r: a bounded value shows it is the O(B/r) remainder of a
      first-order expansion, which is what the Rust exact-geometry gate's tolerance rests on.

       timeout 300 gp -q -f derivations/psinsar_sbas.gp </dev/null                                 */

default(realprecision, 60);
rnd() = seed = (1103515245 * seed + 12345) % 2^31; seed / 2^30 - 1;

{
my(off = 0);
seed = 271828;
for (t = 1, 100,
  my(n = 6 + t % 10, c0 = vector(n, i, rnd()), c1 = vector(n, i, rnd()), ph = vector(n, i, 3 * rnd()),
     s00 = c0 * c0~, s01 = c0 * c1~, s11 = c1 * c1~, b0 = c0 * ph~, b1 = c1 * ph~, det, x);
  det = s00 * s11 - s01^2;
  x = matsolve([s00, s01; s01, s11], [b0, b1]~);
  if (abs((s11 * b0 - s01 * b1) / det - x[1]) + abs((s00 * b1 - s01 * b0) / det - x[2]) > 10^-40, off++));
printf("normal equations: %d of 100 stacks off by more than 1e-40\n", off);
}

/* R1 - R2 for a target at height z inside range cell r (cos theta_z = (Hp - z)/r), the second
   antenna displaced by (bh, bv). */
pathdiff(r, Hp, bh, bv, z) = {
  my(c = (Hp - z) / r, s = sqrt(1 - c^2), T = [r * s, Hp - r * c]);
  sqrt(T[1]^2 + (T[2] - Hp)^2) - sqrt((T[1] - bh)^2 + (T[2] - Hp - bv)^2);
}

{
my(r = 800000, th = 38 * Pi / 180, Hp = r * cos(th), h = 10^-20, worst = 0);
foreach ([10, 100, 300, 1000], B,
  foreach ([[1, 0], [0, 1], [3/5, 4/5], [-4/5, 3/5]], dir,
    my(bh = B * dir[1], bv = B * dir[2], d, want);
    d = (pathdiff(r, Hp, bh, bv, h) - pathdiff(r, Hp, bh, bv, -h)) / (2 * h);
    want = (bh * cos(th) + bv * sin(th)) / (r * sin(th));
    worst = max(worst, abs(d / want - 1) / (B / r))));
printf("height sensitivity vs B_perp/(r sin theta): worst relative gap %.3f x B/r\n", worst);
}
