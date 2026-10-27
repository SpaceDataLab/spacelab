/* The two-way light time at 60 digits in PARI/GP: the second engine for
   derivations/slr_two_way_range.mac.

   The Maxima file solves both legs as the positive roots of their quadratics. Here the defining
   equations are iterated instead -- tau <- |s(t + tau) - x(t)|/c and its down-leg analogue, the
   scheme `two_way_light_time` uses -- to a fixed point at 60 digits, on the same geometry. And the
   Shapiro closed form is re-derived numerically: (2GM/c^2) int dl/r by intnum on a 3D leg with the
   test's (r1, r2, p). A mismatch prints `FAIL`; the last line counts them.

       timeout 300 gp -q -f derivations/slr_two_way_range.gp </dev/null                              */

default(realprecision, 60);
bad = 0;
check(name, got, want, tol) = {
  my(d = abs(got - want) / abs(want));
  if (d > tol, bad++; printf("FAIL %s: %.45g vs %.45g\n", name, got, want),
               printf("%s: ok (rel %.2e)\n", name, d));
}
norm2(v) = sqrt(v * v~);

c = 299792458;
X0 = [4075580, 931854, 4801568]; VS = [-68, 297, 1];
S0 = [-1541300, 7124900, 9918400]; VT = [-4127, 2213, -1712];
sta(t) = X0 + VS * t;
sat(t) = S0 + VT * t;

up = 0.; for (k = 1, 40, up = norm2(sat(up) - sta(0)) / c);
down = up; for (k = 1, 40, down = norm2(sat(up) - sta(up + down)) / c);
check("tau_up", up, 3.269727376084914192625678246738186294558e-2, 1e-38);
check("tau_down", down, 3.269722421616863371440928445809383783869e-2, 1e-38);
check("range", c * (up + down) / 2, 9.802388644103093234503902480390431578695e6, 1e-38);

gm = 398600441800000;
shapiro_integral(P1, P2) = {
  my(d = P2 - P1, len = norm2(d));
  2 * gm / c^2 * intnum(t = 0, 1, len / norm2(P1 + t * d));
}
r1 = 7000000; r2 = 7500000; p = 900000;
ct = (r1^2 + r2^2 - p^2) / (2 * r1 * r2);
P1 = [r1, 0, 0]; P2 = [r2 * ct, r2 * sqrt(1 - ct^2), 0];
{check("Shapiro, (7000000, 7500000, 900000)", shapiro_integral(P1, P2),
      1.102527717326717568931198240043126469883e-3, 1e-38);}
re = 6378137;
{check("Shapiro, LAGEOS zenith", shapiro_integral([re, 0, 0], [re + 12270000, 0, 0]),
      9.516418837725942153383911638749454181935e-3, 1e-38);}
printf("slr two-way range: %d values off\n", bad);
quit;
