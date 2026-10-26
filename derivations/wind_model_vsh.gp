/* Second engine for derivations/wind_model_vsh.mac: the vector-spherical-harmonic basis of
   crates/spacelab-propagation/src/wind_model.rs (`angular_basis`) at degree 6, from Rodrigues'
   formula and the closed-form phi-derivative rather than from the module's three recursions, at
   100 digits, printed to 25 for the gate test. With x = sin phi, s = cos phi and
   Q = d^m P_n/dx^m, Pbar_nm = N s^m Q, so
       dPbar/dphi = N (s^(m+1) Q'(x) - m x s^(m-1) Q),    Pbar/cos phi = N s^(m-1) Q   (m >= 1),
   and each term is spheroidal (m q dtrig, dp trig) then toroidal (-dp trig, m q dtrig), packed by
   n = 1..6, m = 0..n, cos before sin, no sin at m = 0.

   Also checks, at 100 digits, the identity the module's pole argument rests on: the eastward
   component of every term stays bounded as phi -> pi/2 (m q dtrig is a polynomial), and the
   spheroidal north component equals a central difference of Y at step 1e-30. Prints `PASS` lines;
   a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/wind_model_vsh.gp </dev/null                           */

default(realprecision, 100);
nf(n, m) = sqrt((if (m == 0, 1, 2))*(2*n + 1)*(n - m)!/(n + m)!);
dern(P, m) = for (i = 1, m, P = deriv(P, 'x)); P;
pbar(n, m, ph) = my(Q = dern(pollegendre(n), m)); nf(n, m)*cos(ph)^m*subst(Q, 'x, sin(ph));
{
dpbar(n, m, ph) =
  my(Q = dern(pollegendre(n), m), x0 = sin(ph), s = cos(ph));
  nf(n, m)*(s^(m + 1)*subst(deriv(Q, 'x), 'x, x0) - if (m, m*x0*s^(m - 1)*subst(Q, 'x, x0), 0));
}
qbar(n, m, ph) = my(Q = dern(pollegendre(n), m)); if (m, nf(n, m)*cos(ph)^(m - 1)*subst(Q, 'x, sin(ph)), 0);
{
basis(deg, ph, lst) =
  my(lam = lst*2*Pi/24, out = List());
  for (n = 1, deg, for (m = 0, n,
    my(c = cos(m*lam), sn = sin(m*lam));
    foreach ([[c, -sn], [sn, c]], f,
      if (m == 0 && f[1] == sn, next);
      my(dl = m*qbar(n, m, ph)*f[2], dph = dpbar(n, m, ph)*f[1]);
      listput(out, [dl, dph]);
      listput(out, [-dph, dl]))));
  Vec(out);
}
/* A Rust f64 literal with 25 significant digits; an exact zero as 0.0. */
lit(v) = if (v == 0, "0.0", my(e = floor(log(abs(v))/log(10))); Strprintf("%.24fe%d", v/10^e, e));
emit(tag, deg, ph, lst) =
{
  my(b = basis(deg, ph, lst));
  print("// ", tag, ": degree ", deg, ", latitude ", Strprintf("%.4f", ph), " rad, local solar time ",
        Strprintf("%.2f", lst), " h, ", #b, " terms");
  for (k = 1, #b, print("[", lit(b[k][1]), ", ", lit(b[k][2]), "],"));
}

/* The central difference of Y_nm cos(m lam) in phi against the closed form, every term to degree 6. */
{
worst = 0;
for (n = 1, 6, for (m = 0, n,
  my(ph = 0.625, h = 10^-30, fd = (pbar(n, m, ph + h) - pbar(n, m, ph - h))/(2*h));
  worst = max(worst, abs(fd - dpbar(n, m, ph)))));
print(if (worst < 10^-50, "PASS", "FAIL"), " dPbar/dphi closed form = central difference, worst ", Strprintf("%.3g", worst));
}
/* Bounded at the pole: q at phi = pi/2 - 1e-40 is finite and equals its polynomial limit. */
{
worst = 0;
for (n = 1, 6, for (m = 1, n,
  my(Q = dern(pollegendre(n), m), lim = if (m == 1, nf(n, m)*subst(Q, 'x, 1), 0));
  worst = max(worst, abs(qbar(n, m, Pi/2 - 10^-40) - lim))));
print(if (worst < 10^-35, "PASS", "FAIL"), " Pbar/cos phi tends to its polynomial value at the pole, worst ", Strprintf("%.3g", worst));
}
emit("WIND_BASIS_MID", 6, 0.625, 5.5);
emit("WIND_BASIS_POLAR", 6, -1.5, 19.25);
