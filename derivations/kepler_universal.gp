/* The root of the universal Kepler equation at 100 digits: the second engine for
   derivations/kepler_universal.mac and the reference values the `universal_anomaly` gate in
   crates/spacelab-astro/src/kepler.rs asserts the shipped f64 solver against.

       F(chi) = (r0 vr0/sqrt mu) chi^2 C(z) + (1 - alpha r0) chi^3 S(z) + r0 chi - sqrt(mu) dt,
       z = alpha chi^2,  alpha = 2/r0 - v0^2/mu,

   with C and S summed as their power series C = sum (-z)^k/(2k+2)!, S = sum (-z)^k/(2k+3)! for
   |z| <= 1 (no cancellation) and from their closed forms beyond. F is strictly increasing
   (dF/dchi = r > 0, derivations/kepler_universal.mac), so the root is bracketed by doubling out
   from chi = 0 in the direction of dt and found by solve(). Every input is a double: the states
   are dyadic (mu = 398600 km^3/s^2), so the reference is the root for the doubles the test passes.

   Each row is (r0, v0, dt, chi, r, scale): the root, the radius r(chi) = dF/dchi there, and the
   floor any f64 evaluation of F has, in units of epsilon: the sum of the magnitudes of F's four
   terms, plus |dF/dalpha| (2/r0 + v0^2/mu) -- alpha = 2/r0 - v0^2/mu is formed from rounded
   operands, and on a long hyperbolic arc F depends on it exponentially.
   Conics: an ellipse, a hyperbola, e = 0.999 and e = 1.0001 from a 7000 km periapsis, the double
   rounding of e = 1 (|alpha| below 1e-16), and e = 3; spans from 2^-10 s to 1e7 s, both signs.

   A second table, in SI heliocentric units (mu the double nearest 1.32712440018e20 m^3/s^2): two
   fast hyperbolas at 1 au, one the 252 km/s arc of the porkchop cell that `universal_anomaly` once
   ran out its step cap on (departure 46, arrival 62, retrograde, its velocity on a 2^-10 grid),
   from a day to a year and backwards. There alpha r0 is -72 and -4.9, but alpha itself is above
   -1e-9 per metre, the threshold the solver once compared it with.

       timeout 120 gp -q -f derivations/kepler_universal.gp </dev/null                             */

default(realprecision, 100);

mu = 398600;

stumpff(z) = {
  my(c = 0, s = 0, t = 1, k = 0, x);
  if (abs(z) <= 1,
    /* t = (-z)^k; terms fall at least as fast as 1/(2k+2)! */
    while (k == 0 || abs(t) / (2 * k + 2)! > 10^-110,
      c += t / (2 * k + 2)!; s += t / (2 * k + 3)!; t *= -z; k++);
    return([c, s]));
  if (z > 0, x = sqrt(z); return([(1 - cos(x)) / z, (x - sin(x)) / x^3]));
  x = sqrt(-z);
  [(cosh(x) - 1) / (-z), (sinh(x) - x) / x^3];
}

terms(r0, rv, alpha, chi, dt) = {
  my(cs = stumpff(alpha * chi^2), sm = sqrt(mu));
  [(rv / sm) * chi^2 * cs[1], (1 - alpha * r0) * chi^3 * cs[2], r0 * chi, -sm * dt];
}
kepler(r0, rv, alpha, chi, dt) = vecsum(terms(r0, rv, alpha, chi, dt));
radius(r0, rv, alpha, chi) = {
  my(z = alpha * chi^2, cs = stumpff(z), sm = sqrt(mu));
  (rv / sm) * chi * (1 - z * cs[2]) + (1 - alpha * r0) * chi^2 * cs[1] + r0;
}

root(r, v, dt) = {
  my(r0 = sqrt(r * r~), rv = r * v~, alpha = 2 / r0 - (v * v~) / mu, dir = sign(dt), lo = 0, hi);
  hi = dir;
  while (dir * kepler(r0, rv, alpha, hi, dt) < 0, lo = hi; hi *= 2);
  my(chi = solve(x = min(lo, hi), max(lo, hi), kepler(r0, rv, alpha, x, dt)));
  my(h = 10^-45, dFda = (kepler(r0, rv, alpha + h, chi, dt) - kepler(r0, rv, alpha - h, chi, dt)) / (2 * h));
  [chi, radius(r0, rv, alpha, chi), vecsum(apply(abs, terms(r0, rv, alpha, chi, dt)))
     + abs(dFda) * (2 / r0 + (v * v~) / mu)];
}

grid(x) = round(x * 2^30) / 2^30;
fmt(v) = if (abs(v) < 10^-90, "0.0", concat([c | c <- Vec(Strprintf("%.25g", v)), c != " "]));
vfmt(v) = Str("[", fmt(v[1]), ", ", fmt(v[2]), ", ", fmt(v[3]), "]");

/* Periapsis state at 7000 km tilted out of plane, the speed rounded to a 2^-30 grid. */
periapsis(e) = my(vp = sqrt(mu * (1 + e) / 7000)); [[7000, 0, 0], [0, grid(vp * 3/5), grid(vp * 4/5)]];

{
  cases = [
    [[7000, 1200, -800], [-5/4, 55/8, 19/8]],
    [[8000, 1500, -900], [9/8, 19/2, 13/4]],
    periapsis(999/1000), periapsis(10001/10000), periapsis(1), periapsis(3)
  ];
  spans = [2^-10, 1, 100, 5000, 300000, 10^7, -3000, -100];
}
worst = 0;
{
  for (i = 1, #cases, for (j = 1, #spans,
    my(r = cases[i][1], v = cases[i][2], dt = spans[j], sol = root(r, v, dt), r0 = sqrt(r * r~),
       alpha = 2 / r0 - (v * v~) / mu);
    worst = max(worst, abs(kepler(r0, r * v~, alpha, sol[1], dt)) / sol[3]);
    printf("    (%s, %s, %s, %s, %s, %s),\n", vfmt(r * 1.), vfmt(v * 1.), fmt(dt * 1.),
           fmt(sol[1]), fmt(sol[2]), fmt(sol[3]))));
}
printf("universal Kepler residual |F(chi)|/scale, worst over %d roots: %.3g\n", #cases * #spans, worst);

/* ---- the heliocentric table -------------------------------------------------------------------- */
default(path, "derivations:."); read("binary64.gp"); d64 = f64;
mu = d64(132712440018 * 10^9);
grid10(x) = round(x * 2^10) / 2^10;
{
  helio = [
    [[-81028456448, -125753376768, 0], [grid10(-16860502075749397/10^11), grid10(18673036111446476/10^11), 0]],
    [[100000000000, -110000000000, 7000000000], [60000, 50000, -8000]]
  ];
  hspans = [[86400, 864000, 2592000, 31557600, -864000], [10^6, 10^7, 10^8, -3 * 10^7, 31557600]];
}
worst = 0;
{
  for (i = 1, #helio, for (j = 1, #hspans[i],
    my(r = helio[i][1], v = helio[i][2], dt = hspans[i][j], sol = root(r, v, dt), r0 = sqrt(r * r~),
       alpha = 2 / r0 - (v * v~) / mu);
    if (!(alpha * r0 < -1 && alpha > -10^-9), error("not the regime the table is for: ", alpha));
    worst = max(worst, abs(kepler(r0, r * v~, alpha, sol[1], dt)) / sol[3]);
    printf("    (%s, %s, %s, %s, %s, %s),\n", vfmt(r * 1.), vfmt(v * 1.), fmt(dt * 1.),
           fmt(sol[1]), fmt(sol[2]), fmt(sol[3]))));
}
{
  printf("heliocentric universal Kepler residual |F(chi)|/scale, worst over %d roots: %.3g\n",
         #hspans[1] + #hspans[2], worst);
}
printf("mu = %s\n", fmt(mu * 1.));
