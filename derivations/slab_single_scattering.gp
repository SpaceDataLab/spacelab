\\ Second engine for derivations/slab_single_scattering.mac, by quadrature and sampling:
\\  * the next-event estimator's first-collision expectation, int_0^tau (1/mu_s) e^(-t/mu_s)
\\    omega p e^(-t/mu_v)/(2 mu_v) dt, by intnum, against the closed form at 36 geometries and
\\    against Maxima's 40-digit literal for the module test;
\\  * the hemispheric integral of the single-scattering reflectance of an isotropic layer against
\\    the first-order plane albedo counted over collision depth, the escape over the upward
\\    hemisphere being the exponential integral E_2(t) = e^-t - t E_1(t) -- the classical route,
\\    not the one the closed form takes -- at 12 (mu_s, tau);
\\  * `scatter_direction` at 300 pseudo-random (d, cos_t, phi): a unit vector at cos_t from d.
default(realprecision, 50);
\\ exp, but 0 for the grazing-direction arguments intnum samples next to mu = 0.
E(x) = if (x < -10^4, 0, exp(x));
hg(g, m) = (1 - g^2)/(2*(1 + g^2 - 2*g*m)^(3/2));
rho1(w, p, tau, ms, mv) = w*p*(1 - E(-tau*(1/ms + 1/mv)))/(2*(ms + mv));
nee(w, p, tau, ms, mv) = intnum(t = 0, tau, (1/ms)*exp(-t/ms)*w*p*exp(-t/mv)/(2*mv));
{
  my(bad = 0, n = 0);
  foreach ([1/20, 1/2, 2, 8], tau, foreach ([1/5, 4/5, 1], ms, foreach ([3/10, 3/5, 1], mv,
    n++; if (abs(nee(9/10, 1/3, tau, ms, mv) - rho1(9/10, 1/3, tau, ms, mv)) > 1e-45, bad++))));
  printf("first-collision expectation: %d of %d geometries off the closed form by more than 1e-45\n", bad, n);
  my(lit = nee(1, hg(3/10, -24/25), 1, 4/5, 3/5));
  printf("module-test literal: |quadrature - Maxima| = %.1e, %d of 1 off by more than 1e-39\n",
         abs(lit - 0.0714792150410902061369450414069634477139),
         abs(lit - 0.0714792150410902061369450414069634477139) > 1e-39);
}
{
  my(bad = 0, n = 0);
  foreach ([1/5, 1/2, 9/10], ms, foreach ([1/10, 1/2, 2, 6], tau,
    my(h = intnum(m = 0, 1, 2*m*rho1(1, 1/2, tau, ms, m)),
       c = (1/2)*intnum(t = 0, tau, (1/ms)*exp(-t/ms)*(exp(-t) - t*eint1(t))));
    \\ intnum's own accuracy is the limit here: the t log t endpoint of t E_1(t) at tau = 1/10
    \\ costs it 2e-27, against 1e-40 where the layer is thick.
    n++; if (abs(h - c) > 1e-25, bad++)));
  printf("hemispheric integral: %d of %d (mu_s, tau) off the counted albedo by more than 1e-25\n", bad, n);
}
scatter(d, ct, ph) = {
  my(st = sqrt(1 - ct^2), h = sqrt(1 - d[3]^2), k = st/h);
  [k*(d[1]*d[3]*cos(ph) - d[2]*sin(ph)) + d[1]*ct, k*(d[2]*d[3]*cos(ph) + d[1]*sin(ph)) + d[2]*ct,
   -h*st*cos(ph) + d[3]*ct];
}
{
  my(bad = 0);
  setrand(20261027);
  for (i = 1, 300,
    my(z = 2*random(1.)-1, q = 2*Pi*random(1.), ct = 2*random(1.)-1, ph = 2*Pi*random(1.),
       d = [sqrt(1 - z^2)*cos(q), sqrt(1 - z^2)*sin(q), z], o = scatter(d, ct, ph));
    if (abs(o*o~ - 1) > 1e-40 || abs(o*d~ - ct) > 1e-40, bad++));
  printf("scatter_direction: %d of 300 draws not a unit vector at cos_t from d to 1e-40\n", bad);
}
