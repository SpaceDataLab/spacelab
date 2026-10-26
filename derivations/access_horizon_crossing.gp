\\ Second engine for derivations/access_horizon_crossing.mac: the rise/set of the circular pass in
\\ crates/spacelab-coverage/tests/access.rs, found by solving the switching function g(th) = 0
\\ directly (no closed form), then compared with Maxima's lam(e) = acos(R cos e / r) - e.
default(realprecision, 60);
RE = 6378137; MU = 3986004418 * 10^5; rr = 7000 * 10^3; w = sqrt(MU / rr^3);
g(th, el) = rr*cos(th) - RE - sin(el)*sqrt(rr^2 + RE^2 - 2*rr*RE*cos(th));
lam(el) = acos(RE*cos(el)/rr) - el;
{
  for (k = 0, 1,
    el = k * 10 * Pi / 180;
    th = solve(x = 1e-3, Pi/2, g(x, el));
    d = abs(th - lam(el));
    printf("mask %d deg: |solve - closed form| = %.3e  %s\n", 10*k, d, if (d < 1e-50, "ok", "MISMATCH"));
    printf("  rise %.34g  set %.34g\n", 3000 - th/w, 3000 + th/w));
}
