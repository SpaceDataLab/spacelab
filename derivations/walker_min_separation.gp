\\ Second engine for walker_min_separation.mac: the minimum distance between two satellites of a
\\ Walker or circular lattice shell, `crates/spacelab-constellation/src/separation.rs` (plans/18
\\ Ф27.J), at 60 digits.
\\ (a) For 200 random pairs the minimum over time is found without the closed form: the dot product
\\     of the two rotated position vectors is sampled at 256 instants of one orbit and the best
\\     sample refined by a root of its time derivative; it must equal 1 - 2 q^2.
\\ (b) On every Walker Delta and Star with P, S <= 7, every phasing F < P + 2 and seven inclinations,
\\     the minimum over all pairs of the per-pair form equals the class form the code evaluates:
\\     one plane offset dp per class, the nearest point of a coset of step pi/S.
\\ (c) The values the Rust test `the_f64_code_holds_the_60_digit_values` holds its enclosures to,
\\     each recomputed by (b)'s brute force over all pairs and checked against the literal.
\\ (d) A planted sign error in the phasing term is seen by (b) and by (c).
\\ Prints `<name>: 0` when a check holds and `<name>: FAIL ...` when it does not.
\\ Run: timeout 300 gp -q -f derivations/walker_min_separation.gp </dev/null
default(realprecision, 60);
chk(name, e, tol) = printf("%s: %s\n", name, if(abs(e) < tol, 0, Str("FAIL ", e)));

\\ ---- (a) the minimum over time without the closed form ----
rz(w) = [cos(w), -sin(w), 0; sin(w), cos(w), 0; 0, 0, 1];
rx(i) = [1, 0, 0; 0, cos(i), -sin(i); 0, sin(i), cos(i)];
pos(w, i, u) = rz(w) * rx(i) * [cos(u); sin(u); 0];
vel(w, i, u) = rz(w) * rx(i) * [-sin(u); cos(u); 0];
\\ mean motion 1: u_a = u0 + t, u_b = u0 + du + t
dotab(i, wa, dw, u0, du, t) = (pos(wa, i, u0 + t)~ * pos(wa + dw, i, u0 + du + t))[1, 1];
{ddot(i, wa, dw, u0, du, t) = (vel(wa, i, u0 + t)~ * pos(wa + dw, i, u0 + du + t)
  + pos(wa, i, u0 + t)~ * vel(wa + dw, i, u0 + du + t))[1, 1];}
qpair(i, dw, du) = sin(du/2)*cos(dw/2) + cos(i)*cos(du/2)*sin(dw/2);
largest_cos(i, wa, dw, u0, du) = {
  my(m = 256, h = 2*Pi/m, best = -2, tb = 0, c, d0, d1, t);
  for(k = 0, m - 1, c = dotab(i, wa, dw, u0, du, k*h); if(c > best, best = c; tb = k*h));
  d0 = ddot(i, wa, dw, u0, du, tb - h); d1 = ddot(i, wa, dw, u0, du, tb + h);
  \\ a constant dot product (coplanar or co-rotating) has no root to refine
  if(abs(d0) < 1e-50 && abs(d1) < 1e-50, return(best));
  t = solve(s = tb - h, tb + h, ddot(i, wa, dw, u0, du, s));
  max(best, dotab(i, wa, dw, u0, du, t));
}
setrand(20261125);
rnd() = random(2^64) / 2^64;
{
  my(worst = 0);
  for(k = 1, 200,
    my(i = Pi*rnd(), wa = 2*Pi*rnd(), dw = 2*Pi*rnd(), u0 = 2*Pi*rnd(), du = 2*Pi*rnd(),
       q = qpair(i, dw, du));
    worst = max(worst, abs(largest_cos(i, wa, dw, u0, du) - (1 - 2*q^2))));
  chk("time_minimum_of_200_random_pairs_is_the_closed_form", worst, 1e-45);
}

\\ ---- (b) the class form against every pair ----
\\ satellite (p, s) of i: T/P/F, F taken mod P as the generator does; star = 1 halves the node span
sat(P, S, F, star, p, s) = [if(star, Pi, 2*Pi)*p/P, 2*Pi*s/S + 2*Pi*p*(F % P)/(P*S)];
brute(i, P, S, F, star) = {
  my(T = P*S, best = 2, a, b);
  for(x = 0, T - 1, a = sat(P, S, F, star, x \ S, x % S);
    for(y = x + 1, T - 1, b = sat(P, S, F, star, y \ S, y % S);
      best = min(best, abs(qpair(i, b[1] - a[1], b[2] - a[2])))));
  best;
}
\\ the code's form: classes dp = 0..P/2 (Delta) or 0..P-1 (Star); sign = -1 plants the defect
classq(i, P, S, F, star, sign) = {
  my(best = 2, top = if(star, P - 1, P \ 2));
  for(dp = 0, top,
    my(q);
    if(dp == 0,
      if(S < 2, next); q = sin(Pi/S),
      my(b = if(star, Pi*dp/(2*P), Pi*dp/P), cb = cos(b), yb = cos(i)*sin(b),
         rho = sqrt(cb^2 + yb^2), phi = arg(cb + I*yb),
         y = ((sign*dp*(F % P)) % P)/P + S*phi/Pi);
      q = rho*sin(Pi/S*abs(y - round(y))));
    best = min(best, q));
  best;
}
incs = [0, 1, 56*Pi/180, 864*Pi/1800, Pi/2, 2*Pi/3, Pi];
{
  my(worst = 0, shells = 0, seen = 0);
  for(P = 1, 7, for(S = 1, 7, if(P*S < 2, next);
    for(F = 0, P + 1, for(star = 0, 1, foreach(incs, i,
      my(q = brute(i, P, S, F, star));
      worst = max(worst, abs(classq(i, P, S, F, star, 1) - q));
      if(abs(classq(i, P, S, F, star, -1) - q) > 1e-30, seen++);
      shells++)))));
  printf("class form against every pair on %d shells, worst %.3g\n", shells, worst);
  chk("class_form_is_the_minimum_over_every_pair", worst, 1e-50);
  printf("the planted sign error changes the answer on %d of them\n", seen);
  chk("planted_phasing_sign_is_seen_by_the_brute_force", if(seen > 0, 0, 1), 1/2);
}

\\ ---- (c) the values the Rust test holds ----
\\ [label, inclination, P, S, F, star, radius in metres, psi_min held, chord held]
deg(x) = x*Pi/180;
{held = [
  ["galileo 24/3/1 at 56", deg(56), 3, 8, 1, 0, 29600318,
   0.1598113707890583549856, 4725435.060682667288297],
  ["globalstar 48/8/1 at 52", deg(52), 8, 6, 1, 0, 7792137,
   0.1748572907451338051791, 1360776.840645342741334],
  ["iridium star 66/6/2 at 86.4", deg(864/10), 6, 11, 2, 1, 7158137,
   0.07156334527594328243304, 512150.9264667877266972],
  ["star 66/6/8 is 66/6/2", deg(864/10), 6, 11, 8, 1, 7158137,
   0.07156334527594328243304, 512150.9264667877266972],
  ["phasing 0: 12/3/0 at 50", deg(50), 3, 4, 0, 0, 7000000,
   0.08016551146340497945592, 561008.3301977001547150],
  ["one plane: 6/1/0 at 30", deg(30), 1, 6, 0, 0, 7000000,
   1.047197551196597746154, 7000000.000000000000000],
  ["one plane, two satellites: 2/1/0 at 40", deg(40), 1, 2, 0, 0, 7000000,
   3.141592653589793238463, 14000000.00000000000000],
  ["equatorial 4/2/1", 0, 2, 2, 1, 0, 7000000,
   1.570796326794896619231, 9899494.936611665341612],
  ["equatorial 12/4/1: four on each of three points", 0, 4, 3, 1, 0, 7000000,
   0, 0],
  ["equatorial 4/2/0: co-located", 0, 2, 2, 0, 0, 7000000,
   0, 0],
  ["polar 15/5/1", Pi/2, 5, 3, 1, 0, 7000000,
   0.2520436791177251363824, 1759639.489236153713356],
  ["polar 12/4/1: planes 0 and 2 counter-rotate", Pi/2, 4, 3, 1, 0, 7000000,
   0, 0],
  ["retrograde 12/3/1 at 120", deg(120), 3, 4, 1, 0, 7000000,
   0.4364690287349192555862, 3031088.913245535263673],
  ["2/2/0 at 30: pi - 2 i", deg(30), 2, 1, 0, 0, 7000000,
   2.094395102393195492308, 12124355.65298214105469],
  ["starlink-like 1584/72/17 at 53", deg(53), 72, 22, 17, 0, 6928137,
   0.002911555714658274537849, 20171.64974935719710328]
];}
{
  my(worst = 0, worst_chord = 0, seen = 0);
  for(k = 1, #held,
    my(h = held[k], q = brute(h[2], h[3], h[4], h[5], h[6]), psi = 2*asin(q));
    printf("%s: psi_min = %.22g rad, chord = %.22g m\n", h[1], psi, 2*h[7]*q);
    worst = max(worst, abs(psi - h[8]));
    worst_chord = max(worst_chord, abs(2*h[7]*q - h[9]) / h[7]);
    if(abs(2*asin(classq(h[2], h[3], h[4], h[5], h[6], -1)) - h[8]) > 1e-15, seen++));
  chk("held_angles_are_the_brute_force_minimum", worst, 1e-21);
  chk("held_chords_are_the_brute_force_minimum", worst_chord, 1e-21);
  printf("the planted sign error moves %d of the %d held values\n", seen, #held);
  chk("planted_phasing_sign_moves_a_held_value", if(seen > 0, 0, 1), 1/2);
}
