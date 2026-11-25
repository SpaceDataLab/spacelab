/* Second engine for derivations/access_zone_elevation.mac, at 60 digits, by routes that share no
   step with the Maxima proof: what crates/spacelab/src/zone.rs and access.rs rest on when they
   search a zone's boundary for the extreme elevation of a satellite.

     1  the gradient of sin eps on the ellipsoid, claims 3-4 of the Maxima file, against central
        differences of sin eps itself, at random points and satellites;
     2  exactly two normals of WGS 84 pass through every point above it: the quartic whose real
        roots t give the feet p_i = a_i^2 S_i / (a_i^2 + t) has exactly two real roots, counted by
        Sturm sequences, one t > 0 (the nadir) and one t < -a^2 (the far foot), for satellites
        from 1 m to 40 000 km up, over the poles and the equator;
     3  Case B is empty: a point on the sphere of diameter rho_i tangent at p from inside, along a
        principal direction, is strictly inside the ellipsoid;
     4  the far foot: the code's equation in x = -(a^2 + t), a^2 r^2 / x^2 + b^2 z^2 / (x + c)^2 = 1
        with c = a^2 - b^2, has its root in [sqrt(a^2 r^2 + b^2 z^2) - c, sqrt(a^2 r^2 + b^2 z^2)],
        and its foot has S - p along -n, sin eps = -1, at the latitude
        atan2(-z x, r (x + c)) and the longitude opposite the satellite's;
     5  |f''|, f = sin eps along a unit-speed geodesic, under N2 + 4 kappa_max / D + 3 / D^2 with
        N2 = 2.5e-14 m^-2 and kappa_max = a / b^2: the geodesic integrated from its own equations
        (dphi/ds = cos alpha / M, dlam/ds = sin alpha / (N cos phi), dalpha/ds = sin alpha tan phi
        / N) by a classical Runge-Kutta step at 60 digits, and f'' its second difference, on
        random geodesics and satellites down to 10 m above the surface.
   Each check prints `ok <name>` or `FAIL <name>`.

       timeout 300 gp -q -f derivations/access_zone_elevation.gp </dev/null                       */

default(realprecision, 60);
setrand(20261125);

a = 6378137;
f = 1/(298257223563/10^9);
b = a*(1 - f);
e2 = f*(2 - f);
c2 = a^2 - b^2;

/* A uniform deviate in [lo, hi) at full precision. */
unif(lo, hi) = lo + (hi - lo)*random(2^200)/2^200;

Nr(phi) = a/sqrt(1 - e2*sin(phi)^2);
Mr(phi) = a*(1 - e2)/(1 - e2*sin(phi)^2)^(3/2);
surf(phi, lam) = my(n = Nr(phi)); [n*cos(phi)*cos(lam), n*cos(phi)*sin(lam), n*(1 - e2)*sin(phi)];
nrm(phi, lam) = [cos(phi)*cos(lam), cos(phi)*sin(lam), sin(phi)];
eN(phi, lam) = [-sin(phi)*cos(lam), -sin(phi)*sin(lam), cos(phi)];
eE(phi, lam) = [-sin(lam), cos(lam), 0];
dot(u, v) = sum(i = 1, 3, u[i]*v[i]);
norm2(u) = sqrt(dot(u, u));
/* GP has no atan2; arg is the same angle. */
atan2(y, x) = arg(x + I*y);
/* A satellite at geodetic (phi, lam, h). */
sat(phi, lam, h) = surf(phi, lam) + h*nrm(phi, lam);
sineps(S, phi, lam) = my(d = S - surf(phi, lam)); dot(d, nrm(phi, lam))/norm2(d);

/* 1: the gradient. */
{
  worst = 0;
  for (k = 1, 200,
    my(phi = unif(-1.5, 1.5), lam = unif(-3, 3), S = sat(unif(-1.5, 1.5), unif(-3, 3), unif(1e3, 4e7)),
       d, D, u, step = 1e-20, gphi, glam, wphi, wlam);
    d = S - surf(phi, lam); D = norm2(d); u = dot(d, nrm(phi, lam));
    gphi = (sineps(S, phi + step, lam) - sineps(S, phi - step, lam))/(2*step);
    glam = (sineps(S, phi, lam + step) - sineps(S, phi, lam - step))/(2*step);
    wphi = (Mr(phi)/D)*(1/Mr(phi) + u/D^2)*dot(d, eN(phi, lam));
    wlam = (Nr(phi)*cos(phi)/D)*(1/Nr(phi) + u/D^2)*dot(d, eE(phi, lam));
    worst = max(worst, max(abs(gphi - wphi), abs(glam - wlam))));
  printf("%s the gradient of sin eps is claims 3-4 (worst %.3g against differences)\n",
         if (worst < 1e-30, "ok", "FAIL"), worst);
}

/* 2: two normals. With F(t) = a^2 r^2/(a^2 + t)^2 + b^2 z^2/(b^2 + t)^2, the feet solve F = 1,
   that is a^2 r^2 (b^2 + t)^2 + b^2 z^2 (a^2 + t)^2 - (a^2 + t)^2 (b^2 + t)^2 = 0. The satellite is
   rounded to the micrometre so that the quartic is exact and Sturm's count is a proof for it. */
{
  bad = 0; tested = 0;
  heights = [1, 10, 1e3, 4e5, 7e5, 2e7, 3.6e7, 4e7];
  for (k = 1, 400,
    my(phi = if (k % 4 == 0, unif(-1.5707, 1.5707), if (k % 4 == 1, unif(-1e-3, 1e-3),
                                                     unif(1.5, 1.5707963))),
       h = heights[1 + k % #heights]*unif(1, 1.1), S, r2, z, q);
    S = apply(v -> round(v*10^6)/10^6, sat(phi, unif(-3, 3), h)); r2 = S[1]^2 + S[2]^2; z = S[3];
    q = a^2*r2*(b^2 + 'x)^2 + b^2*z^2*(a^2 + 'x)^2 - (a^2 + 'x)^2*(b^2 + 'x)^2;
    tested++;
    if (polsturm(q) != 2 || polsturm(q, [0, +oo]) != 1 || polsturm(q, [-oo, -a^2]) != 1, bad++));
  printf("%s exactly two normals through each of %d points 1 m to 40 000 km up (%d not)\n",
         if (bad == 0, "ok", "FAIL"), tested, bad);
}

/* 3: Case B. S = p + rho_i sin psi (cos psi e_i - sin psi n), e_i a principal direction. */
{
  worst = -1;
  for (k = 1, 400,
    my(phi = unif(-1.5707, 1.5707), lam = unif(-3, 3), psi = unif(1e-6, Pi - 1e-6), p, n, e, rho, S, q);
    p = surf(phi, lam); n = nrm(phi, lam);
    if (k % 2, e = eN(phi, lam); rho = Mr(phi), e = eE(phi, lam); rho = Nr(phi));
    S = p + rho*sin(psi)*(cos(psi)*e - sin(psi)*n);
    q = (S[1]^2 + S[2]^2)/a^2 + S[3]^2/b^2 - 1;
    worst = max(worst, q));
  printf("%s every Case B point is strictly inside the ellipsoid (largest S^T A^-1 S - 1: %.3g)\n",
         if (worst < 0, "ok", "FAIL"), worst);
}

/* 4: the far foot, by the code's route. */
farres(x, r, z) = a^2*r^2/x^2 + b^2*z^2/(x + c2)^2 - 1;
{
  worst = -1; outside = 0; tested = 0;
  for (k = 1, 300,
    my(phi = unif(-1.5707963, 1.5707963), lam = unif(-3, 3), h = 10^unif(0, 7.6), S, r, z, R, x,
       lat, lon, pf, d);
    S = sat(phi, lam, h); r = sqrt(S[1]^2 + S[2]^2); z = S[3];
    R = sqrt(a^2*r^2 + b^2*z^2);
    if (farres(R - c2, r, z) < 0 || farres(R, r, z) > 0, outside++; next);
    x = solve(t = R - c2, R, farres(t, r, z));
    tested++;
    lat = atan2(-z*x, r*(x + c2));
    lon = atan2(S[2], S[1]) + Pi;
    pf = surf(lat, lon);
    d = S - pf;
    worst = max(worst, abs(dot(d, nrm(lat, lon))/norm2(d) + 1)));
  printf("%s the far foot of %d satellites is in its bracket with sin eps = -1 (worst %.3g; %d outside)\n",
         if (tested == 300 && worst >= 0 && worst < 1e-40 && outside == 0, "ok", "FAIL"), tested,
         worst, outside);
}

/* 5: |f''| along geodesics against the bound. One classical Runge-Kutta step of the geodesic
   equations in (phi, lam, alpha). */
geo(y) = {
  my(phi = y[1], al = y[3]);
  [cos(al)/Mr(phi), sin(al)/(Nr(phi)*cos(phi)), sin(al)*tan(phi)/Nr(phi)];
}
rk4(y, hs) = {
  my(k1 = geo(y), k2 = geo(y + hs/2*k1), k3 = geo(y + hs/2*k2), k4 = geo(y + hs*k3));
  y + hs/6*(k1 + 2*k2 + 2*k3 + k4);
}
/* The geodesic from y0 advanced by s, in n steps. */
along(y0, s, n) = { my(y = y0); for (i = 1, n, y = rk4(y, s/n)); y; }
{
  N2 = 25/10^15; kmax = a/b^2;
  worst = 0; tested = 0;
  for (k = 1, 300,
    my(phi = unif(-1.45, 1.45), lam = unif(-3, 3), al = unif(-Pi, Pi), y0, H, ym, yp, S,
       fm, f0, fp, fpp, D, bound);
    y0 = [phi, lam, al];
    /* the satellite near the point or far from it, down to 10 m above the surface */
    S = if (k % 3 == 0, sat(phi + unif(-0.01, 0.01), lam + unif(-0.01, 0.01), 10^unif(1, 5)),
                        sat(unif(-1.5, 1.5), unif(-3, 3), 10^unif(3, 7.6)));
    D = norm2(S - surf(phi, lam));
    H = D/1000;
    ym = along(y0, -H, 40); yp = along(y0, H, 40);
    fm = sineps(S, ym[1], ym[2]); f0 = sineps(S, phi, lam); fp = sineps(S, yp[1], yp[2]);
    fpp = (fp - 2*f0 + fm)/H^2;
    bound = N2 + 4*kmax/D + 3/D^2;
    tested++;
    worst = max(worst, abs(fpp)/bound));
  printf("%s |f''| along %d geodesics stays under N2 + 4 kappa/D + 3/D^2 (largest share %.4f)\n",
         if (worst > 0 && worst < 1, "ok", "FAIL"), tested, worst);
}
