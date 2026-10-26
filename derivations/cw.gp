/* Clohessy-Wiltshire references at 100 digits: the second engine for derivations/cw.mac and the
   values the gates in crates/spacelab-astro/src/cw.rs assert the shipped f64 code against.

   1. Phi(dt) s0 as exp(A dt) s0, summed as the power series of the CW system matrix A -- no
      trigonometric closed form is written here -- at the test's (n, dt, s0) doubles, and
      det exp(A dt) = 1.
   2. `relative_lvlh_state` at the test's chief and deputy doubles: the LVLH triad from r and
      r x v, the projection onto it, less n z-hat x rho.
   Every mismatch prints `FAIL`.

       timeout 300 gp -q -f derivations/cw.gp </dev/null                                           */

default(realprecision, 100);

f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
check(name, got, want, tol) = {
  my(d = abs(got - want));
  if (d > tol, printf("FAIL %s: %.40g vs %.40g, |d| = %.3e\n", name, got, want, d),
               printf("%s: ok (|d| = %.3e)\n", name, d));
}
expm(M) = {
  my(S = matid(#M), T = matid(#M), k = 0);
  while (k < 400, k++; T = T * M / k; S += T; if (normlp(T) < 10^-110, break));
  S;
}

n = f64(0.0011); dt = f64(1234.5);
s0 = apply(f64, [1000.0, -2000.0, 500.0, 1.0, -0.5, 0.3])~;
{A = [0, 0, 0, 1, 0, 0; 0, 0, 0, 0, 1, 0; 0, 0, 0, 0, 0, 1;
     3 * n^2, 0, 0, 0, 2 * n, 0; 0, 0, 0, -2 * n, 0, 0; 0, 0, -n^2, 0, 0, 0];}
Phi = expm(A * dt);
got = Phi * s0;
{want = [3537.795509123616011503969, -5642.604277551699946432585, 372.1942260117101102469815,
        2.459340147346565387335485, -6.083150120071955561650064, -0.4742156306254448145483056];}
names = ["x", "y", "z", "vx", "vy", "vz"];
for (i = 1, 6, printf("  %s = %.40g\n", names[i], got[i]); check(Str("Phi s0 ", names[i]), got[i], want[i], 1e-21));
check("det exp(A dt)", matdet(Phi), 1, 1e-90);

/* 2. relative_lvlh_state. */
cross(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]];
unit(a) = a / sqrt(a * a~);
cr = apply(f64, [3000e3, 4000e3, 5000e3]); cv = apply(f64, [-4200.0, 1300.0, 1500.0]);
rho = apply(f64, [1000.0, -2000.0, 500.0]); vrel = apply(f64, [1.7, -0.9, 2.3]);
ex = unit(cr); ez = unit(cross(cr, cv)); ey = cross(ez, ex);
rl = [rho * ex~, rho * ey~, rho * ez~];
vl = [vrel * ex~, vrel * ey~, vrel * ez~] - [-n * rl[2], n * rl[1], 0];
lvlh = concat(rl, vl);
{want = [-353.5533905932737622004222, -1301.285520896677504480514, 1852.472939911582809357446,
        0.4070635580986780652728290, -0.6628830081639689664526549, 2.122200306347353030709691];}
for (i = 1, 6, check(Str("relative_lvlh_state ", names[i]), lvlh[i], want[i], 1e-21));
quit;
