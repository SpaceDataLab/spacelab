\\ Second engine for derivations/access_hermite_speed.mac: the twelve residuals of the cubic Hermite
\\ interpolant in crates/spacelab/src/access.rs at 60 digits on 200 random intervals, and the speed
\\ bound itself: on 200 random three-dimensional intervals the velocity of the interpolant, sampled
\\ at 1001 points of [0, 1], never exceeds max(|v0|, |c|, |v1|), and reaches it wherever |v0| or
\\ |v1| is the largest of the three.
default(realprecision, 60);
setrand(31);
rnd(lo, hi) = lo + (hi - lo) * random(10^30) / 10^30;
H(x, h, r0, r1, v0, v1) = (2*x^3 - 3*x^2 + 1)*r0 + (x^3 - 2*x^2 + x)*h*v0 \
  + (3*x^2 - 2*x^3)*r1 + (x^3 - x^2)*h*v1;
dHdt(x, h, r0, r1, v0, v1) = ((6*x^2 - 6*x)*r0 + (3*x^2 - 4*x + 1)*h*v0 \
  + (6*x - 6*x^2)*r1 + (3*x^2 - 2*x)*h*v1) / h;
lerp(x, a, b) = (1 - x)*a + x*b;
casteljau(x, b0, b1, b2, b3) = my(c01 = lerp(x, b0, b1), c12 = lerp(x, b1, b2), \
  c23 = lerp(x, b2, b3)); lerp(x, lerp(x, c01, c12), lerp(x, c12, c23));
norm3(u) = sqrt(u[1]^2 + u[2]^2 + u[3]^2);
worst = vector(12);
{
  for (k = 1, 200,
    h = rnd(1e-3, 900); r0 = rnd(-4e7, 4e7); r1 = r0 + h * rnd(-8e3, 8e3);
    v0 = rnd(-8e3, 8e3); v1 = rnd(-8e3, 8e3); x = rnd(0, 1); sc = abs(r0) + abs(r1) + h*8e3;
    worst[1] = max(worst[1], abs(H(0, h, r0, r1, v0, v1) - r0) / sc);
    worst[2] = max(worst[2], abs(H(1, h, r0, r1, v0, v1) - r1) / sc);
    worst[3] = max(worst[3], abs(dHdt(0, h, r0, r1, v0, v1) - v0) / 8e3);
    worst[4] = max(worst[4], abs(dHdt(1, h, r0, r1, v0, v1) - v1) / 8e3);
    b0 = r0; b1 = r0 + h*v0/3; b2 = r1 - h*v1/3; b3 = r1;
    bez = (1 - x)^3*b0 + 3*x*(1 - x)^2*b1 + 3*x^2*(1 - x)*b2 + x^3*b3;
    worst[5] = max(worst[5], abs(H(x, h, r0, r1, v0, v1) - bez) / sc);
    worst[6] = max(worst[6], abs(casteljau(x, b0, b1, b2, b3) - bez) / sc);
    c = 3*(r1 - r0)/h - v0 - v1;
    worst[7] = max(worst[7],
      abs(dHdt(x, h, r0, r1, v0, v1) - (v0*(1 - x)^2 + c*2*x*(1 - x) + v1*x^2)) / 8e3);
    worst[8] = max(worst[8], abs((1 - x)^2 + 2*x*(1 - x) + x^2 - 1));
    a = vector(4, j, rnd(-1e3, 1e3) / 10^(2*j)); t0 = rnd(-1e3, 1e3);
    cu(t) = a[1] + a[2]*t + a[3]*t^2 + a[4]*t^3;
    dcu(t) = a[2] + 2*a[3]*t + 3*a[4]*t^2;
    sc9 = abs(cu(t0 + h*x)) + 1;
    worst[9] = max(worst[9],
      abs(H(x, h, cu(t0), cu(t0 + h), dcu(t0), dcu(t0 + h)) - cu(t0 + h*x)) / sc9);
    worst[10] = max(worst[10],
      abs((h*x)^4 - H(x, h, 0, h^4, 0, 4*h^3) - h^4*x^2*(1 - x)^2) / h^4);
    worst[11] = max(worst[11], abs((1/2)^2*(1/2)^2 - 1/16));
    worst[12] = max(worst[12], abs(2*(1/2)*(1 - 1/2)^2 - 2*(1/2)^2*(1 - 1/2))));
  for (i = 1, 12,
    printf("%s residual %d: %.3g\n", if (worst[i] < 1e-45, "ok", "FAIL"), i, worst[i]));
  over = 0; reach = 0;
  for (k = 1, 200,
    h = rnd(1e-3, 900);
    R0 = vector(3, j, rnd(-4e7, 4e7)); V0 = vector(3, j, rnd(-8e3, 8e3));
    V1 = vector(3, j, rnd(-8e3, 8e3));
    R1 = vector(3, j, R0[j] + h * rnd(-8e3, 8e3));
    C = vector(3, j, 3*(R1[j] - R0[j])/h - V0[j] - V1[j]);
    bound = max(norm3(V0), max(norm3(C), norm3(V1)));
    top = 0;
    for (m = 0, 1000,
      x = m / 1000;
      top = max(top, norm3(vector(3, j, dHdt(x, h, R0[j], R1[j], V0[j], V1[j])))));
    if (top > bound * (1 + 1e-50), over++);
    ends = max(norm3(V0), norm3(V1));
    if (bound > ends || abs(top - bound) < 1e-50 * bound, reach++));
  printf("%s the sampled speed never exceeds the hull bound: %d of 200 intervals over it\n",
    if (over == 0, "ok", "FAIL"), over);
  printf("%s the bound is reached wherever an end velocity is the largest: %d of 200\n",
    if (reach == 200, "ok", "FAIL"), reach);
}
