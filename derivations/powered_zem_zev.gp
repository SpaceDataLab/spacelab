/* Second engine for derivations/powered_zem_zev.mac: the reference values of the gates in
   crates/spacelab-astro/src/powered.rs (`OPTIMAL_TRAJECTORY`, `DESCENT_OUTCOME`,
   `OPTIMAL_DELTA_V`), recomputed in exact rationals at 100 digits.

   The fixture: start (1200, -450, 2600) m at (85, 37, -110) m/s, pad (640, 215, 12) m, arrival
   (1.5, -0.8, -0.6) m/s, gravity (0.04, -0.03, -3.71) m/s^2, T = 42 s -- every input a decimal and
   taken here as the exact rational it denotes.

   1. The open-loop optimum a(t) = A + B t from the two terminal conditions alone (solved as a
      linear system, not through the feedback form), sampled where `OPTIMAL_TRAJECTORY` samples it,
      together with the state there.
   2. The closed loop of `powered_descent` (target velocity 0) with 1, 2, 3 and 5 held-command
      steps, in exact arithmetic: the miss, the residual speed, and the sum of |a_T| h.
   3. The propellant of the continuous optimum to rest at the pad, int_0^T |A + B t| dt, by
      intnum.
   Every value is compared with the test's constant; a mismatch above 1e-22 relative (or 1e-22
   absolute where the constant is zero) prints FAIL.

       timeout 300 gp -q -f derivations/powered_zem_zev.gp </dev/null                             */

default(realprecision, 100);
r0 = [1200, -450, 2600]; v0 = [85, 37, -110]; pad = [640, 215, 12];
varr = [3/2, -4/5, -3/5]; g = [1/25, -3/100, -371/100]; T = 42;
nrm(a) = sqrt(a * a~);
fails = 0; checks = 0;
same(name, got, want) = {
  my(d = if (want == 0, abs(got), abs(got - want) / abs(want)));
  checks++;
  if (d > 10^-22, fails++; printf("FAIL: %s = %.30g, table %.25g\n", name, got, want));
}

/* 1: x(T) = x0 + v0 T + (g + A) T^2/2 + B T^3/6, v(T) = v0 + (g + A) T + B T^2/2, per axis. */
affine(xf, vf) = {
  my(A = vector(3), B = vector(3), M = [T^2/2, T^3/6; T, T^2/2], s);
  for (i = 1, 3,
    s = matsolve(M, [xf[i] - r0[i] - v0[i] * T - g[i] * T^2/2; vf[i] - v0[i] - g[i] * T]);
    A[i] = s[1, 1]; B[i] = s[2, 1]);
  [A, B];
}
{
  my(AB = affine(pad, varr), A = AB[1], B = AB[2],
     table = [[42, [-10.11142857142857142857143, -1.193809523809523809523810, 5.412040816326530612244898]],
              [63/2, [-6.069761904761904761904762, -1.031904761904761904761905, 5.863401360544217687074830]],
              [21, [-2.028095238095238095238095, -0.87, 6.314761904761904761904762]],
              [21/2, [2.013571428571428571428571, -0.7080952380952380952380952, 6.766122448979591836734694]],
              [21/10, [5.246904761904761904761905, -0.5785714285714285714285714, 7.127210884353741496598639]]]);
  for (k = 1, #table,
    my(t = T - table[k][1], a = A + B * t);
    for (i = 1, 3, same(Str("optimal_command_tgo_", table[k][1], "_axis_", i), a[i], table[k][2][i])));
  /* The sampled states: r(t) = r0 + v0 t + (g + A) t^2/2 + B t^3/6. */
  my(t = T - 63/2, r = r0 + v0 * t + (g + A) * t^2/2 + B * t^3/6, v = v0 + (g + A) * t + B * t^2/2);
  same("state_tgo_31.5_x", r[1], 1611.578125); same("state_tgo_31.5_vz", v[3], -89.75892857142857142857143);
}

/* 2: the coded loop, exact. */
cmd(x, v, tg) = 6 * (pad - (x + v * tg + g * tg^2/2)) / tg^2 - 2 * ([0, 0, 0] - (v + g * tg)) / tg;
loop(n) = {
  my(h = T / n, x = r0, v = v0, dv = 0, a);
  for (k = 0, n - 1,
    a = cmd(x, v, (n - k) * h); dv += nrm(a) * h;
    x = x + (v + (a + g) * h / 2) * h; v = v + (a + g) * h);
  [nrm(x - pad), nrm(v), dv];
}
{
  my(table = [[1, 4728.151012816743779516674, 337.7250723440531271083338, 481.2637987971389701215998],
              [2, 2364.075506408371889758337, 337.7250723440531271083338, 738.5421150448222776109897],
              [3, 0, 0, 359.2096070271262006486131],
              [5, 0, 0, 342.5799046241530231158781]]);
  for (k = 1, #table,
    my(n = table[k][1], o = loop(n));
    same(Str("loop_", n, "_position_error"), o[1], table[k][2]);
    same(Str("loop_", n, "_velocity_error"), o[2], table[k][3]);
    same(Str("loop_", n, "_delta_v"), o[3], table[k][4]));
}

/* 3: the continuous optimum to rest at the pad. */
{
  my(AB = affine(pad, [0, 0, 0]), A = AB[1], B = AB[2]);
  same("optimal_delta_v", intnum(t = 0, T, nrm(A + B * t)), 338.1382847886351560776051);
}
printf("powered-descent reference values: %d checked, %d off by more than 1e-22\n", checks, fails);
if (checks != 30, print("FAIL: expected 30 checks, ran ", checks));
