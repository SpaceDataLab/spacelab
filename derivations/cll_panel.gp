\\ The CLL flat panel of `derivations/cll_panel.mac` at 40 digits: the reference rows that
\\ `drag_coefficient::tests::cll_panel_matches_the_40_digit_quadrature` holds the shipped
\\ quadrature to. Each row integrates the reflected normal momentum over the incident Maxwellian
\\ with intnum, the mean reflected normal speed in the Bessel form that cll_panel.mac (4) equates
\\ to 1F1, so neither the code's two branches of h nor its Gauss-Kronrod rule enters.
\p 40
m(x, a) = {
  if (a == 1, return(sqrt(Pi)/2));
  my(q = (1 - a)*x^2/a);
  sqrt(Pi*a)/2*exp(-q/2)*((1 + q)*besseli(0, q/2) + q*besseli(1, q/2));
}
cre(s, g, a, tw, ti) = {
  my(w = sqrt(tw/ti)/s, lo = max(0, g - 12/s), hi = max(g, 0) + 12/s);
  my(f = t -> t*s/sqrt(Pi)*exp(-s^2*(t - g)^2)*m(t/w, a), mid = max(g, 0));
  \\ Split at the peak: one double-exponential rule across a peak 1/s wide stalls near 1e-23.
  2*w*(if (mid > lo, intnum(t = lo, mid, f(t)), 0) + intnum(t = mid, hi, f(t)));
}
panel(s, g, a, sg, tw, ti) = {
  my(l = sqrt(1 - g^2), z = 2 - erfc(s*g), e = exp(-s^2*g^2), cp, ct);
  cp = (g^2 + 1/(2*s^2))*z + g*e/(sqrt(Pi)*s) + cre(s, g, a, tw, ti);
  ct = sg*(l*g*z + l*e/(s*sqrt(Pi)));
  [cp, ct, cp*g + ct*l, cp*l - ct*g];
}
\\ The a = 1 row against Sentman's closed form at alpha = 1: the CLL limit of (6).
sentman1(s, g, tw, ti) = {
  my(l = sqrt(1 - g^2), z = 2 - erfc(s*g), e = exp(-s^2*g^2), vr = sqrt(tw/ti)/s);
  (g^2 + 1/(2*s^2))*z + g*e/(sqrt(Pi)*s) + vr/2*(sqrt(Pi)*g*z + e/s);
}
d = abs(panel(7.4549, cos(Pi/4), 1, 1, 300, 1000)[1] - sentman1(7.4549, cos(Pi/4), 300, 1000));
if (d < 1e-35, print("PASS: a = 1 is Sentman at alpha = 1 (", d, ")"), print("FAIL: a = 1 differs by ", d));
\\ The rows: s, cos(delta), alpha_n, sigma_t, T_w, T_inf.
{rows = [[7.4549, 1, 0.5, 1, 300, 1000], [7.4549, cos(15*Pi/180), 0.9, 0.7, 300, 1000],
        [7.4549, cos(45*Pi/180), 0.5, 1, 300, 1000], [7.4549, cos(75*Pi/180), 0.05, 0.3, 300, 1000],
        [7.4549, cos(89*Pi/180), 0.75, 1, 300, 1000], [7.4549, 0, 0.5, 1, 300, 1000],
        [7.4549, -0.2, 0.5, 1, 300, 1000], [3, 0.6, 0.2, 0.5, 400, 700],
        [12, 1, 0.999, 1, 300, 1500], [9, 1, 1e-4, 0, 300, 900]];}
{
  for (i = 1, #rows,
    my(r = rows[i], c = panel(r[1], r[2], r[3], r[4], r[5], r[6]));
    printf("ROW %s %.20e %.20e %.20e %.20e\n", i, c[1], c[2], c[3], c[4]));
}
\\ h(q) = 1F1(-1/2; 1; -q) in the Bessel form, across both of the code's branches (seam at 60).
hb(q) = exp(-q/2)*((1 + q)*besseli(0, q/2) + q*besseli(1, q/2));
foreach([1/1000, 1/10, 1, 17, 60, 61, 300, 10^4, 10^6], q, printf("H %s %.20e\n", q, hb(q)));
\\ A Langmuir coverage: K = 1.44e6 /Pa, n_O = 1e14 /m^3, T = 1000 K, k_B exact.
kp = 144*10^4 * 10^14 * 1380649/10^29 * 1000; printf("THETA %.20e\n", kp/(1 + kp));
\\ Hyperthermal limit of (7b): at normal incidence c_p -> 2 (1 + sqrt(1 - a)).
h = panel(2000, 1, 0.5, 1, 300, 1000)[1];
if (abs(h - 2*(1 + sqrt(1/2))) < 1e-5, print("PASS: s = 2000 normal incidence ", h), print("FAIL: hyperthermal ", h));
quit
