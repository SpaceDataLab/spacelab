/* Minimum-energy optimal-control references at 100 digits: the second engine for
   derivations/optimal_pmp.mac and the values the gates in crates/spacelab-astro/src/optimal.rs
   assert the shipped f64 code against.

   1. Hamilton's equations at the test's augmented state AUGMENTED (mu = 3.986004418e14), by
      numerical differentiation (`derivnum`) of the scalar Hamiltonian
          H = lr.v - mu (lv.r)/|r|^3 - |lv|^2/2
      alone -- none of the twelve rates is written here -- giving (dH/dlr, dH/dlv, -dH/dr, -dH/dv),
      and the cost rate |lv|^2/2.
   2. The double-integrator transfer (mu = 0) of the closed-form test: the optimal costate,
      the energy cost and the control at five times, from the closed form
      derivations/optimal_pmp.mac derives, at the test's inputs read as exact decimals (the
      literals were generated that way; at the doubles the values move by under 4e-18, seven
      decades inside the test's 1e-11 relative bound, and the shift is printed).
   Every mismatch prints `FAIL`.

       timeout 300 gp -q -f derivations/optimal_pmp.gp </dev/null                                  */

default(realprecision, 100);

f64(x) = { my(e, k); if (x == 0, return(0)); e = exponent(x * 1.); k = 52 - e; round(x * 2^k) / 2^k; }
check(name, got, want, tol) = {
  my(d = abs(got - want));
  if (d > tol, printf("FAIL %s: %.40g vs %.40g, |d| = %.3e\n", name, got, want, d),
               printf("%s: ok (|d| = %.3e)\n", name, d));
}

/* 1. Hamilton's equations. */
mu = f64(3.986004418e14);
{y0 = apply(f64, [7.1e6, -2.4e6, 1.9e6, 1.1e3, 6.9e3, -2.2e3, 3.1e-7, -1.4e-7, 8.2e-8,
                  -2.7e-4, 5.3e-4, 1.6e-4]);}
ham(y) = {
  my(r = y[1..3], v = y[4..6], lr = y[7..9], lv = y[10..12], rn = sqrt(r * r~));
  lr * v~ - mu * (lv * r~) / rn^3 - (lv * lv~) / 2;
}
dH(j) = derivnum(t = y0[j], ham(vector(12, k, if (k == j, t, y0[k]))));
/* (r', v', lr', lv') = (dH/dlr, dH/dlv, -dH/dr, -dH/dv) */
rates = vector(12, i, if (i <= 6, dH(i + 6), -dH(i - 6)));
{for (i = 1, 12, printf("  rate[%d] = %.40g\n", i - 1, rates[i]));}
printf("  rate[12] = %.40g\n", y0[10..12] * y0[10..12]~ / 2);
check("rate 12", y0[10..12] * y0[10..12]~ / 2, 1.896999999999999927840708e-7, 1e-31);
/* The rates the Rust test carries. */
{want = [1100, 6900, -2200,
        -6.122701787884325022996414, 2.069206942383433810610829, -1.638701746053551766731362,
        6.536437345663205721460750e-10, 1.574086070670416386914619e-10, 3.752119511509803033194804e-10,
        -3.1e-7, 1.4e-7, -8.2e-8];}
{for (i = 1, 12, check(Str("rate ", i - 1), rates[i], if (i <= 3 || i >= 10, y0[if (i <= 3, i + 3, i - 3)] * if (i >= 10, -1, 1), want[i]),
                      if (i <= 3 || i >= 10, 1e-80, 1e-24 * max(1, abs(want[i])))));}

/* 2. The double integrator. */
T = 137;
closed(R0, V0, RF, VF) = {
  my(dr = RF - R0 - V0 * T, dv = VF - V0);
  [concat(-12 * dr / T^3 + 6 * dv / T^2, -6 * dr / T^2 + 2 * dv / T), dr, dv];
}
{dec = closed([1200, -430, 260], [17/10, 23/10, -9/10], [1580, 65, -140], [-6/10, 11/10, 45/100]);}
{dbl = closed(apply(f64, [1200.0, -430.0, 260.0]), apply(f64, [1.7, 2.3, -0.9]),
             apply(f64, [1580.0, 65.0, -140.0]), apply(f64, [-0.6, 1.1, 0.45]));}
printf("  costate shift from decimal to double inputs: %.3e\n", normlp(dbl[1] - dec[1], oo));
costate = dec[1]; dr = dec[2]; dv = dec[3];
{want = [-0.001421741783411301365467907, -0.001223169280919422576363494, 0.001722867299822311444597455,
        -0.08060099099579093185571954, -0.07502797165538920560498695, 0.1081623954392881879695242];}
for (i = 1, 6, check(Str("costate ", i - 1), costate[i], want[i], 1e-24 * abs(want[i])));
J = sum(i = 1, 3, (12 * dr[i]^2 / T^3 - 12 * dr[i] * dv[i] / T^2 + 4 * dv[i]^2 / T) / 2);
check("energy cost", J, 0.7260964227782027594033180, 1e-24);
u(t) = (T - t) * (12 * dr / T^3 - 6 * dv / T^2) + (-6 * dr / T^2 + 4 * dv / T);
{wantu = [[0.0, [0.08060099099579093185571954, 0.07502797165538920560498695, -0.1081623954392881879695242]],
         [34.25, [0.03190633491395386008844371, 0.03313442378389898236453727, -0.04915419042037402099206138]],
         [68.5, [-0.01678832116788321167883212, -0.008759124087591240875912409, 0.009854014598540145985401460]],
         [102.75, [-0.06548297724972028344610794, -0.05065267195908146411636209, 0.06886221961745431296286430]],
         [137.0, [-0.1141776333315573552133838, -0.09254621983057168735681176, 0.1278704246363684799403271]]];}
{for (k = 1, 5, my(t = f64(wantu[k][1]), uu = u(t));
  for (i = 1, 3, check(Strprintf("u(%.2f)[%d]", wantu[k][1], i - 1), uu[i], wantu[k][2][i], 1e-24 * abs(wantu[k][2][i]))))};
quit;
