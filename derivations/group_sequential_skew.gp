\\ Second engine for group_sequential_skew.mac: the skewed null of SDTM-1's blind test
\\ (docs/sdtm1-blind-protocol.md §9 A2, Scope; plans/18 Ф30.N), at 30 digits by quadrature.
\\
\\ (a) The standardised gamma (G - k) / sqrt(k), G ~ Gamma(k, 1): its first four moments by numerical
\\     integration of the density, at the shapes the protocol uses and at k = 4 / gamma^2.
\\ (b) The pieces of the Studentized mean's one-term Edgeworth expansion: the normal fourth moment behind
\\     E[A^3 B] = 3 gamma, the Hermite antiderivatives, and the correction gamma (2 x^2 + 1) phi / 6 at
\\     sample points, from kappa1 = -gamma / 2 and kappa3 = -2 gamma (per sqrt(n)).
\\ Run: timeout 300 gp -q -f derivations/group_sequential_skew.gp </dev/null

default(realprecision, 30);
check(name, ok) = if (!ok, error(name), print(name, ": ok"));
close(a, b) = abs(a - b) < 1e-20;
phi(x) = exp(-x^2 / 2) / sqrt(2 * Pi);

{
  \\ (a) Moments of (G - k) / sqrt(k). The density x^(k - 1) e^(-x) / Gamma(k) has an end-point
  \\ singularity x^(k - 1) at 0 and decays as e^(-x).
  my(moment(k, p) = intnum(x = [0, k - 1], [oo, 1], ((x - k) / sqrt(k))^p * x^(k - 1) * exp(-x)) / gamma(k));
  foreach([1, 1/2, 2, 4 / (7/5)^2, 4 / (1/2)^2], k,
    my(m = vector(4, p, moment(k, p)), mass = moment(k, 0));
    print("k = ", k * 1.0, ": mass ", mass, ", moments ", m);
    check(Str("mass 1, k = ", k), close(mass, 1));
    check(Str("mean 0, k = ", k), abs(m[1]) < 1e-20);
    check(Str("variance 1, k = ", k), close(m[2], 1));
    check(Str("skewness 2 / sqrt(k), k = ", k), close(m[3], 2 / sqrt(k)));
    check(Str("kurtosis 3 + 6 / k, k = ", k), close(m[4], 3 + 6 / k)));
  foreach([2, 7/5, 1/2, 3/10], g,
    check(Str("k = 4 / gamma^2 gives skewness ", g), close(2 / sqrt(4 / g^2), g)));
  \\ k = 1: the exponential less 1, of skewness 2, the shape `skewed`.
  check("k = 1 is the exponential less 1",
    close(intnum(x = 0, [oo, 1], (x - 1)^3 * exp(-x)), 2));

  \\ (b) A standard normal: E[A^4] = 3, so B = gamma A + c W with W independent gives E[A^3 B] = 3 gamma.
  check("normal fourth moment 3", close(intnum(x = [-oo, 1], [oo, 1], x^4 * phi(x)), 3));
  check("normal third moment 0", abs(intnum(x = [-oo, 1], [oo, 1], x^3 * phi(x))) < 1e-20);
  \\ The Hermite antiderivatives, d/dx(-phi He2) = phi He3, by quadrature from -oo.
  foreach([-2.4545, -1, 0.3, 2.0041], x0,
    check(Str("int phi He3 = -phi He2 at ", x0),
      close(intnum(x = [-oo, 1], x0, phi(x) * (x^3 - 3 * x)), -phi(x0) * (x0^2 - 1)));
    check(Str("int phi He1 = -phi at ", x0),
      close(intnum(x = [-oo, 1], x0, phi(x) * x), -phi(x0))));
  \\ The correction -phi (kappa1 + kappa3 He2 / 6) with kappa1 = -gamma / 2, kappa3 = -2 gamma.
  foreach([[2, -3.4711], [-7/5, 2.4545], [1/2, -2.0041], [-2, 0]], c,
    my(g = c[1], x = c[2]);
    check(Str("Edgeworth term at gamma = ", g, ", x = ", x),
      close(-phi(x) * (-g / 2 + (-2 * g) * (x^2 - 1) / 6), g * (2 * x^2 + 1) * phi(x) / 6)));
}
