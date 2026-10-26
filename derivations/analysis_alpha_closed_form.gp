\\ Second engine for analysis_alpha_closed_form.mac: the sampled alpha-characteristic evaluated from
\\ its definition on vectors (positions, a Fibonacci grid, sort, worst case) at 60 digits, against
\\ the closed forms. Prints `<name>: 0` when they agree to 1e-50.
\\ Run: timeout 120 gp -q -f derivations/analysis_alpha_closed_form.gp </dev/null
default(realprecision, 60);
fib(k, n) = my(z = 1 - (2*k + 1)/n, r = sqrt(1 - z^2), f = Pi*(3 - sqrt(5))*k); [r*cos(f), r*sin(f), z];
pos(inc, u) = [cos(u), sin(u)*cos(inc), sin(u)*sin(inc)];
alpha(inc, S, ngrid, nt, L) = {
  my(worst = 0);
  for(i = 0, nt - 1,
    for(k = 0, ngrid - 1,
      my(p = fib(k, ngrid), d = vecsort(vector(S, s, acos(pos(inc, 2*Pi*(i/nt + (s - 1)/S)) * p~))));
      worst = max(worst, d[L])));
  worst;
}
chk(name, e) = printf("%s: %s\n", name, if(abs(e) < 1e-50, 0, e));
chk("one_satellite_five_instants", alpha(0, 1, 1, 5, 1) - 4*Pi/5);
chk("three_satellites_five_instants_fold_1", alpha(0, 3, 1, 5, 1) - 4*Pi/15);
chk("three_satellites_five_instants_fold_2", alpha(0, 3, 1, 5, 2) - 2*Pi/3);
chk("three_satellites_five_instants_fold_3", alpha(0, 3, 1, 5, 3) - 14*Pi/15);
chk("polar_alpha", alpha(Pi/2, 1, 2, 4, 1) - 5*Pi/6);
