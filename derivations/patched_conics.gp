/* Second engine for derivations/patched_conics.mac, and the 100-digit check of the reference values
   the gates in crates/spacelab-astro/src/patched.rs assert against (`ARCS`, `AIMING`, `ASYMPTOTE`).

   Hyperbolic arcs by a route other than the closed forms patched.rs codes: from the energy
   E = v_inf^2/2 and the angular momentum h = r_p v_p (v_p by vis-viva), a = -mu/(2E) and
   e = sqrt(1 + 2 E h^2/mu^2); the turn angle is 2 nu_inf - pi with nu_inf = acos(-1/e).

   The B-plane from its definition: the state is placed on its own conic, the conic is followed back
   to the true anomaly nu = -nu_inf + 10^-40 -- 1e40 times the periapsis radius out on the incoming
   asymptote -- and there S is the unit velocity and B = r - (r.S) S, resolved on
   T = (S x N)/|S x N|, R = S x T. Truncating the limit at 1e-40 leaves an O(1e-40) error; the
   working precision of 160 digits absorbs the 1e40-fold cancellation in r - (r.S) S.

   Every value is compared with the 25-digit constant in the test module; a mismatch above 1e-22
   relative prints FAIL.

       timeout 300 gp -q -f derivations/patched_conics.gp </dev/null                              */

default(realprecision, 160);
cross(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]];
nrm(a) = sqrt(a * a~);
fails = 0; checks = 0;
same(name, got, want) = {
  my(d = abs(got - want) / max(abs(want), 10^-30));
  checks++;
  printf("%s: %.30g (table %.25g, relative %.2g)\n", name, got, want, d);
  if (d > 10^-22, fails++; print("FAIL: ", name));
}

/* ARCS: (v_inf, mu, r_p, a, e, v_p, turn) -- the table's last four columns at 25 digits. */
{
arcs = [
  [3, 398600, 6578, -44288.88888888888888888888889, 1.148524836929252383341696, 11.41016439963220842504917, 2.113174817088339117015914],
  [135/100, 4282837/100, 36895/10, -23499.79149519890260631001, 1.157001393001881696641735, 5.003883943456582533791297, 2.087526679168519459675233],
  [572/100, 126686534, 78240, -3872027.177368086458995550, 1.020206469742080085638778, 57.19376481167622290530697, 2.742874867959808662240267]
];
}
{
  for (i = 1, #arcs,
    my(vi = arcs[i][1], mu = arcs[i][2], rp = arcs[i][3], E = vi^2 / 2, vp, h, a, e);
    vp = sqrt(2 * (E + mu / rp)); h = rp * vp;
    a = -mu / (2 * E); e = sqrt(1 + 2 * E * h^2 / mu^2);
    same(Str("arc", i, "_semi_major_axis"), a, arcs[i][4]);
    same(Str("arc", i, "_eccentricity"), e, arcs[i][5]);
    same(Str("arc", i, "_periapsis_speed"), vp, arcs[i][6]);
    same(Str("arc", i, "_turn_angle"), 2 * acos(-1 / e) - Pi, arcs[i][7]));
}

/* AIMING: (r, v, mu, N, B.T, B.R, |B|). The table was evaluated on the decimal inputs as written;
   the test passes their doubles, and the shift that rounding causes is printed and must stay below
   a tenth of the test's 1e-12 |B| bound. */
dbl(x) = if (x == 0, 0, my(e = exponent(x)); round(x * 2^(52 - e)) / 2^(52 - e));
{
aiming = [
  [[10000, 2000, 500], [1, 11, 1/2], 398600, [0, 0, 1],
   16278.02163527225319243382, -46.74561837551210412395808, 16278.08875486397109825241],
  [[-8200, 14300, -3100], [57/10, -19/10, 42/10], 4282837/100, [4/10, -6/10, 186/100],
   -12469.13907847655236498609, -137.4602508349918769888052, 12469.89673890488575136211]
];
}
asymptote = [0.6705831875261181787291, 0.7395040223885868070710, 0.05875363374644604107500];
/* [B.T, B.R, |B|, S] on the conic of (r, v) under mu, against reference north. */
aim(r, v, mu, north) = {
  my(h, hh, ev, e, P, Q, p, nu, rr, vv, S, B, T, R);
  h = cross(r, v); hh = h / nrm(h);
  ev = ((v * v~ - mu / nrm(r)) * r - (r * v~) * v) / mu; e = nrm(ev);
  P = ev / e; Q = cross(hh, P); p = (h * h~) / mu;
  nu = -acos(-1 / e) + 10^-40;
  rr = p / (1 + e * cos(nu)) * (cos(nu) * P + sin(nu) * Q);
  vv = sqrt(mu / p) * (-sin(nu) * P + (e + cos(nu)) * Q);
  S = vv / nrm(vv);
  B = rr - (rr * S~) * S;
  T = cross(S, north); T = T / nrm(T); R = cross(S, T);
  [B * T~, B * R~, nrm(B), S];
}
{
  for (i = 1, #aiming,
    my(c = aiming[i], x = aim(c[1], c[2], c[3], c[4]),
       y = aim(apply(dbl, c[1]), apply(dbl, c[2]), dbl(c[3]), apply(dbl, c[4])), shift);
    same(Str("aiming", i, "_b_dot_t"), x[1], c[5]);
    same(Str("aiming", i, "_b_dot_r"), x[2], c[6]);
    same(Str("aiming", i, "_b_magnitude"), x[3], c[7]);
    if (i == 1, for (k = 1, 3, same(Str("asymptote_", k), x[4][k], asymptote[k])));
    shift = vecmax(apply(abs, [y[1] - x[1], y[2] - x[2], y[3] - x[3]])) / x[3];
    printf("aiming%d: shift from rounding the inputs to doubles %.2g of |B|\n", i, shift);
    if (shift > 10^-13, print("FAIL: aiming", i, " rounding shift above a tenth of the test bound")));
}
printf("patched-conic reference values: %d checked, %d off by more than 1e-22\n", checks, fails);
if (checks != 21, print("FAIL: expected 21 checks, ran ", checks));
