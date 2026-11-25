\\ Second engine for derivations/access_station_geometry.mac: the twelve residuals of the station
\\ geometry and the SGP4 speed bound of crates/spacelab/src/access.rs, evaluated at 60 digits on
\\ 200 points drawn from each claim's domain. Every residual must fall below 1e-50 of its scale.
default(realprecision, 60);
setrand(31);
rnd(lo, hi) = lo + (hi - lo) * random(10^30) / 10^30;
cross(u, v) = [u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]];
dot(u, v) = u[1]*v[1] + u[2]*v[2] + u[3]*v[3];
\\ geodetic_to_itrf at height hh. Defined at the top level: a GP function's body runs to the end of
\\ the sequence it opens, so one defined inside the loop would swallow the loop's remaining lines.
P(hh, a, e2, phi, lam) = my(N = a / sqrt(1 - e2 * sin(phi)^2)); \
  [(N + hh)*cos(phi)*cos(lam), (N + hh)*cos(phi)*sin(lam), (N*(1 - e2) + hh)*sin(phi)];
worst = vector(12);
{
  for (k = 1, 200,
    a = rnd(6.3e6, 6.4e6); e2 = rnd(0, 0.02); h = rnd(-500, 4e5);
    phi = rnd(-Pi/2, Pi/2); lam = rnd(-Pi, Pi);
    n = [cos(phi)*cos(lam), cos(phi)*sin(lam), sin(phi)];
    z = P(h, a, e2, phi, lam) - P(0, a, e2, phi, lam) - h * n;
    for (i = 1, 3, worst[i] = max(worst[i], abs(z[i]) / a));
    P0 = P(0, a, e2, phi, lam); b2 = a^2 * (1 - e2);
    worst[4] = max(worst[4], abs(P0[1]^2/a^2 + P0[2]^2/a^2 + P0[3]^2/b2 - 1));
    grad = [2*P0[1]/a^2, 2*P0[2]/a^2, 2*P0[3]/b2];
    c = cross(n, grad);
    for (i = 1, 3, worst[4 + i] = max(worst[4 + i], abs(c[i]) * a));
    worst[8] = max(worst[8], abs(dot(n, n) - 1));
    d = [rnd(-4e7, 4e7), rnd(-4e7, 4e7), rnd(-4e7, 4e7)]; s = rnd(-1, 1); nd = sqrt(dot(d, d));
    worst[9] = max(worst[9], abs((dot(d, n) - s*nd) - nd*(dot(d, n)/nd - s)) / nd);
    mu = rnd(3.9e14, 4.0e14); sa = rnd(6.5e6, 4.3e7); ecc = rnd(0, 0.95); rp = sa * (1 - ecc);
    vp2 = mu * (1 + ecc) / (sa * (1 - ecc));
    worst[10] = max(worst[10], abs(mu * (2/rp - 1/sa) - vp2) / vp2);
    nn = rnd(7e-5, 1.2e-3); ak = (mu / nn^2)^(1/3);
    worst[11] = max(worst[11], abs(nn^2 * ak^3 - mu) / mu);
    w = [rnd(-1e-4, 1e-4), rnd(-1e-4, 1e-4), rnd(-1e-4, 1e-4)];
    r = [rnd(-4e7, 4e7), rnd(-4e7, 4e7), rnd(-4e7, 4e7)];
    wr = cross(w, r);
    worst[12] = max(worst[12],
      abs(dot(wr, wr) - (dot(w, w) * dot(r, r) - dot(w, r)^2)) / (dot(w, w) * dot(r, r))));
  for (i = 1, 12,
    printf("%s residual %d: %.3g\n", if (worst[i] < 1e-50, "ok", "FAIL"), i, worst[i]));
}
