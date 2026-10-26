\\ Second engine for derivations/observability_diurnal_arc.mac.
\\  * the diurnal arcs by solving the altitude equation sin h0 = sin p sin d + cos p cos d cos H for H
\\    directly (solve, not acos), against Maxima's closed forms;
\\  * the sidereal rate in deg/h and the Young (1994) rationals, exact;
\\  * the Young denominator's real root, Kasten & Young at the apparent horizon, and the Saemundsson
\\    slope bound.
default(realprecision, 50);
ok(d, eps) = if (d < eps, "ok", "MISMATCH");
deg = Pi/180;
arc(p, d, h0) = solve(H = 0, Pi, sin(p)*sin(d) + cos(p)*cos(d)*cos(H) - sin(h0));
{
  my(cases = [[0, 0, 0, Pi/2], [30, 30, 30, acos(1/3)], [45, 30, 0, acos(-1/sqrt(3))], \
              [0, 0, -12, Pi/2 + 12*deg]]);
  for (i = 1, #cases, my(c = cases[i], H = arc(c[1]*deg, c[2]*deg, c[3]*deg), e = abs(H - c[4]));
    printf("arc lat %d dec %d alt %d: H = %.34g  %s\n", c[1], c[2], c[3], H, ok(e, 1e-45)));
}
k = 100273781191135448/10^17;
printf("sidereal rate %s deg/h  %s\n", k*15, if (k*15 == 150410671786703172/10^16, "ok", "MISMATCH"));
printf("solar-rate error per hour of hour angle %.20g s\n", 3600*(1 - 1/k));
yn(c) = 1002432/10^6*c^2 + 148386/10^6*c + 96467/10^7;
yd(c) = c^3 + 149864/10^6*c^2 + 102963/10^7*c + 303978/10^9;
printf("Young horizon %s  %s;  zenith %s  %s\n", yn(0)/yd(0), if (yn(0)/yd(0) == 4823350/151989, "ok", "MISMATCH"), \
       yn(1)/yd(1), if (yn(1)/yd(1) == 580232350/580232139, "ok", "MISMATCH"));
printf("Young denominator real roots %s\n", polrootsreal(yd(x)));
ky(h) = 1/(sin(h*deg) + 50572/10^5*(h + 607995/10^5)^(-16364/10^4));
printf("Kasten-Young apparent horizon %.34g\n", ky(0));
u(h) = h + 103/10/(h + 511/100);
printf("Saemundsson |R'| bound %.20g (< 1: %s)\n", (102/100/60)*deg/sin(u(-1)*deg)^2, \
       if ((102/100/60)*deg/sin(u(-1)*deg)^2 < 1, "ok", "MISMATCH"));
