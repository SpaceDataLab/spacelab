/* Second engine for derivations/edge_mtf.mac, by quadrature and numerical differentiation.

   - The Gaussian LSF's transform by intnum against exp(-2 pi^2 s^2 f^2), at s = 3/2 (the Rust
     gate's width) and five frequencies;
   - the box of width 1/4 against sinc(f/4), and the two boxes' triangle against sinc(f/4)^2;
   - the metres per degree as the speed of the meridian ellipse (N cos phi, N (1 - e^2) sin phi) and
     of the parallel circle N cos phi, by derivnum, against Maxima's 40-digit values.

   The summary counts the values off by more than 1e-30.

       timeout 300 gp -q -f derivations/edge_mtf.gp </dev/null                                    */

default(realprecision, 60);
off = 0; checked = 0;
{
chk(x, y) =
  checked++;
  if (abs(x - y) > 1e-30, off++; print("  off: ", x, " vs ", y));
}
s = 3/2; h = 1/4;
g(x) = exp(-x^2/(2*s^2))/(s*sqrt(2*Pi));
snc(x) = if (x == 0, 1, sin(Pi*x)/(Pi*x));
Phi(u) = 1 - erfc(u/sqrt(2))/2;
/* Two boxes of width h convolve to the triangle of half-width h; its transform by quadrature is
   the factor binning and differencing put on the Gaussian's, which is therefore G sinc(f h)^2 by
   the convolution theorem. Each integrand is even, so the quadratures run over [0, oo) in pieces
   short enough for the peak. */
tri(t) = (1 - t/h)/h;
ft_even(F, f, cuts) = 2*sum(i = 1, #cuts - 1, intnum(x = cuts[i], cuts[i + 1], F(x)*cos(2*Pi*f*x)));
{
foreach ([1/20, 1/10, 1/4, 2/5, 3/5], f,
  my(G = exp(-2*Pi^2*s^2*f^2));
  chk(ft_even(g, f, [0, 1, 2, 4, 8, 16, 32]), G);
  chk(ft_even(x -> 1/h, f, [0, h/2]), snc(f*h));
  chk(ft_even(tri, f, [0, h]), snc(f*h)^2));
}
A = 6378137; FF = 1/(298257223563/10^9); E2 = FF*(2 - FF);
N(p) = A/sqrt(1 - E2*sin(p)^2);
{
my(want = [[0, 110574.2758215943614803342870362612553044, 111319.4907932735726477133826705565482628],
           [30, 110852.4424956170990679559492678767970814, 96486.28025089651228022381061278254040082],
           [45, 111131.7774141756347865022310313295226385, 78846.83509397810759420340430640034798517],
           [60, 111412.2874577794398659627403507355177838, 55800.00157243612871400347129539486135276],
           [90, 111693.9795591275024527158027545731241728, 0]]);
for (i = 1, #want,
  my(ph = want[i][1]*Pi/180, dp, dz);
  dp = derivnum(p = ph, N(p)*cos(p)); dz = derivnum(p = ph, N(p)*(1 - E2)*sin(p));
  chk(sqrt(dp^2 + dz^2)*Pi/180, want[i][2]);
  chk(N(ph)*cos(ph)*Pi/180, want[i][3]));
}
printf("edge_mtf: %d values checked, %d off by more than 1e-30\n", checked, off);
