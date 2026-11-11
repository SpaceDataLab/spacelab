/* Second engine for derivations/drag_error_growth.mac: a nonlinear planar two-body orbit at 400 km with drag in
   an exponential atmosphere (scale height 60 km, 3e-12 kg/m^3 at 400 km, C_D A/m = 0.022 m^2/kg), run by
   classical RK4 at 10 s with the density and with 1.1 times it. The along-track separation after 6, 24 and
   72 h against the Clohessy-Wiltshire solution 4 f (1 - cos n t)/n^2 - (3/2) f t^2, f = -eps a_D at the
   start, and against its secular term (3/2) eps a_D t^2. The orbit's own decay raises a_D by about 1.5 %
   over the 72 h, which the constant-force solution does not carry: the run is 1 % ahead at 72 h, the bound 2 %. Prints `PASS` lines; a
   failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/drag_error_growth.gp </dev/null                                      */

default(realprecision, 38);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
mu = 3986004418 * 10^5; r0 = 6378137 + 400000;
rho0 = 3 * 10^-12; H = 60000; B = 22/1000; eps = 1/10;
force(s, k) = {
  my(r = sqrt(s[1]^2 + s[2]^2), v = sqrt(s[3]^2 + s[4]^2), g = -mu/r^3,
     d = -k*rho0*exp(-(r - r0)/H)*v*B/2);
  [s[3], s[4], g*s[1] + d*s[3], g*s[2] + d*s[4]];
}
rk4(s, k, h) = {
  my(k1 = force(s, k), k2 = force(s + h/2*k1, k), k3 = force(s + h/2*k2, k), k4 = force(s + h*k3, k));
  s + h/6*(k1 + 2*k2 + 2*k3 + k4);
}
{
  my(v0 = sqrt(mu/r0), n = v0/r0, aD = rho0*v0^2*B/2, f = -eps*aD, h = 10.,
     s = [r0, 0., 0., v0], p = [r0, 0., 0., v0], marks = [6, 24, 72], m = 1);
  for (i = 1, 72*360,
    s = rk4(s, 1, h); p = rk4(p, 1 + eps, h);
    if (m <= #marks && i == marks[m]*360,
      my(t = i*h, r = sqrt(s[1]^2 + s[2]^2),
         along = r*arg(conj(s[1] + I*s[2])*(p[1] + I*p[2])),
         cw = 4*f*(1 - cos(n*t))/n^2 - 3/2*f*t^2, secular = 3/2*eps*aD*t^2);
      print(marks[m], " h: along-track ", along, " m; Clohessy-Wiltshire ", cw, " m; (3/2) eps a_D t^2 ", secular,
            " m");
      check(Str(marks[m], " h: the run within 2 % of Clohessy-Wiltshire and of its secular term"),
        abs(along/cw - 1) < 2/100 && abs(along/secular - 1) < 2/100);
      m++));
}
