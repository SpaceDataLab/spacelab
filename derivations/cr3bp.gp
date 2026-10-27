/* The CR3BP reference values at 100 digits: the second engine for derivations/cr3bp.mac and the
   values the gates in crates/spacelab-astro/src/cr3bp.rs assert the shipped f64 code against.

   1. The mass ratio of `Cr3bp::SUN_EARTH`, Sun to Earth (the Moon not added), from the IAU 2009
      System of Astronomical Constants (Luzum et al. 2011, Celest. Mech. Dyn. Astr. 110:293-304):
          GM_S/GM_E = 332946.0487,   mu = 1 / (1 + GM_S/GM_E),
      exact in rationals, and its correctly rounded double. The Sun to Earth+Moon ratio, with
      M_M/M_E = 0.0123000371, is printed beside it: the constant's doc once claimed that one.
   2. The collinear points as the real roots in (0, 1) of the three quintics derivations/cr3bp.mac
      derives from dOmega/dx = 0, for the doubles the tests pass as mu (Earth-Moon, Sun-Earth+Moon,
      and mu = 0.4639); each is compared with the literal the Rust test carries.
   3. Omega, C and the synodic acceleration at the test's field point (P, V), from the definitions,
      compared with the literals in the Rust tests.
   4. grad Omega at L4/L5 for the same mass ratios: zero to the working precision.
   Every mismatch prints `FAIL`; a clean run prints none.

       timeout 300 gp -q -f derivations/cr3bp.gp </dev/null                                        */

default(realprecision, 100);

/* The exact value of the double nearest x (ties are measure-zero for these inputs). */
f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
check(name, got, want, tol) = {
  my(d = abs(got - want));
  if (d > tol, printf("FAIL %s: %.40g vs %.40g, |d| = %.3e\n", name, got, want, d),
               printf("%s: ok (|d| = %.3e)\n", name, d));
}

/* 1. Sun to Earth. */
mse = 3329460487 / 10^4; mme = 123000371 / 10^10;
mu_sun_earth = 1 / (1 + mse);
mu_sun_emb = (1 + mme) / (1 + mme + mse);
printf("SUN_EARTH mu (Sun : Earth), exact: %.40g\n", mu_sun_earth);
printf("  as a double: %.25g\n", f64(mu_sun_earth));
check("SUN_EARTH literal", f64(3.0034805952013234e-6), f64(mu_sun_earth), 0);
mu_earth_moon = mme / (1 + mme);
printf("EARTH_MOON mu (Moon : Earth+Moon), exact: %.40g\n", mu_earth_moon);
printf("  as a double: %.25g\n", f64(mu_earth_moon));
check("EARTH_MOON literal", f64(0.012150584460351), f64(mu_earth_moon), 0);
printf("  Sun : Earth+Moon would be %.25g\n", mu_sun_emb);
{printf("  the former constant 3.0034806e-6 is %.3e from Sun:Earth and %.3e from Sun:Earth+Moon\n",
       3.0034806e-6 / mu_sun_earth - 1, 3.0034806e-6 / mu_sun_emb - 1)};

/* 2. Collinear points. */
q1(m) = 'g^5 - (3 - m)*'g^4 + (3 - 2*m)*'g^3 - m*'g^2 + 2*m*'g - m;
q2(m) = 'g^5 + (3 - m)*'g^4 + (3 - 2*m)*'g^3 - m*'g^2 - 2*m*'g - m;
q3(m) = 'g^5 + (2 + m)*'g^4 + (1 + 2*m)*'g^3 - (1 - m)*'g^2 - 2*(1 - m)*'g - (1 - m);
unit_root(p) = {
  my(r = select(t -> t > 0 && t < 1, polrootsreal(p)));
  if (#r != 1, error("expected one root in (0, 1), got ", r));
  r[1];
}
collinear(m) = [1 - m - unit_root(q1(m)), 1 - m + unit_root(q2(m)), -m - unit_root(q3(m))];

{systems = [
  ["Earth-Moon", f64(mu_earth_moon),
   [0.8369151314273821600466932650851427, 1.155682161024676951248163210236372,
    -1.005062645331442005708330094870261]],
  ["Sun-Earth", f64(mu_sun_earth),
   [0.9900265938700224097363023403997850, 1.010034116422946031576144465434425,
    -1.000001251450247999078835863664773]],
  ["near equal mass", f64(0.4639),
   [0.05098507556760716648324714833465792, 1.210662680043992900440520787331401,
    -1.185643513328397515409973511296275]]
]};
{for (i = 1, #systems,
  my(s = systems[i], l = collinear(s[2]));
  printf("%s mu = %.25g\n", s[1], s[2]);
  for (k = 1, 3,
    printf("  L%d x = %.40g   (double %.17g)\n", k, l[k], f64(l[k]));
    check(Str(s[1], " L", k), l[k], s[3][k], 1e-33)))};

/* 3. The field point. */
P = [f64(0.31), f64(-0.47), f64(0.23)]; V = [f64(0.17), f64(-0.29), f64(0.41)];
m = f64(mu_earth_moon);
r1 = sqrt((P[1] + m)^2 + P[2]^2 + P[3]^2); r2 = sqrt((P[1] - 1 + m)^2 + P[2]^2 + P[3]^2);
Om = (P[1]^2 + P[2]^2) / 2 + (1 - m) / r1 + m / r2;
check("Omega(P)", Om, 1.780317787318750078456715328779219, 1e-24);
check("C(P, V)", 2 * Om - V * V~, 3.279535574637500184380348286784810, 1e-24);
{acc = [P[1] + 2 * V[2] - (1 - m) * (P[1] + m) / r1^3 - m * (P[1] - 1 + m) / r2^3,
       P[2] - 2 * V[1] - (1 - m) * P[2] / r1^3 - m * P[2] / r2^3,
       -(1 - m) * P[3] / r1^3 - m * P[3] / r2^3]};
check("a.x", acc[1], -1.628505363613640063773917538670481, 1e-24);
check("a.y", acc[2], 1.200216013413323035419682785396339, 1e-24);
check("a.z", acc[3], -0.9837227299682220083692688599076874, 1e-24);

/* 4. L4/L5. */
gradOmega(m, x, y) = {
  my(a = sqrt((x + m)^2 + y^2), b = sqrt((x - 1 + m)^2 + y^2));
  [x - (1 - m) * (x + m) / a^3 - m * (x - 1 + m) / b^3, y - (1 - m) * y / a^3 - m * y / b^3];
}
{for (i = 1, #systems,
  my(mm = systems[i][2]);
  for (sg = -1, 1, if (sg == 0, next);
    my(gr = gradOmega(mm, 1/2 - mm, sg * sqrt(3) / 2));
    check(Str(systems[i][1], if (sg > 0, " L4", " L5"), " |grad Omega|"), normlp(gr), 0, 1e-90)))};
quit;
