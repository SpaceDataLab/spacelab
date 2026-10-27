/* The multi-terminal observables on the fixture of derivations/multistatic_observables.mac, at 60
   digits by a second route: every range rate is the derivative of the separation along the
   straight-line motion, taken by PARI's own numerical derivative (derivnum) of |A(t) - B(t)| rather
   than as a dot product over a norm, and the bistatic gradient likewise from derivnum of the range
   sum in each coordinate. The results are compared with the 30-digit Maxima values the
   `every_observable_matches_the_maxima_closed_form` and
   `the_bistatic_gradient_is_the_sum_of_the_two_unit_vectors` gates in
   crates/spacelab-estimation/src/multistatic.rs assert.

       timeout 300 gp -q -f derivations/multistatic_observables.gp </dev/null                        */

default(realprecision, 60);
c = 299792458; fc = 2200000000;
nrm(v) = sqrt(v * v~);

Sat = [3961000, 2614000, 5215000]; Vel = [-3140, 6190, -700];
T1 = [4075530, 931781, 4801618]; V1 = [-679/10, 2972/10, 0];
T2 = [3183899, 1421478, 5322820]; V2 = [-1036/10, 2322/10, 0];
SatB = [4512000, 1878000, 4903000]; VelB = [-2050, 5780, 2400];

sep_rate(A, va, B, vb) = derivnum(t = 0, nrm((A + t * va) - (B + t * vb)));
gradb(k) = {
  my(e = vector(3, i, i == k));
  derivnum(h = 0, nrm(Sat + h * e - T1) + nrm(Sat + h * e - T2));
}

{
my(got = [nrm(Sat - T1) + nrm(Sat - T2),
          sep_rate(Sat, Vel, T1, V1) + sep_rate(Sat, Vel, T2, V2),
          (nrm(Sat - T2) - nrm(Sat - T1)) / c,
          fc * (sep_rate(Sat, Vel, T1, V1) - sep_rate(Sat, Vel, T2, V2)) / c,
          nrm(Sat - SatB),
          sep_rate(Sat, Vel, SatB, VelB),
          gradb(1), gradb(2), gradb(3)]);
my(maxima = [3.16350019133434775975857394752e6, 9.12320439456152896377995923281e3,
             -1.02936463346488750872129618964e-3, 1.73842815162547150815909549816e4,
             9.70897007926175452790431266953e5, -6.67939024124905219627190708466e1,
             4.78425527005991124489541937078e-1, 1.80441329126225152612351648908e0,
             1.62583515565325185868685719401e-1]);
my(names = ["bistatic range, m", "bistatic range rate, m/s", "TDOA, s", "FDOA, Hz",
            "inter-satellite range, m", "inter-satellite range rate, m/s",
            "d(bistatic)/dx", "d(bistatic)/dy", "d(bistatic)/dz"]);
my(bad = 0);
for (i = 1, #got,
  my(rel = abs(got[i] - maxima[i]) / abs(maxima[i]));
  if (rel > 10^-29, bad++);
  printf("  %-32s %s   |PARI - Maxima|/value %.1e\n", names[i], Strprintf("%.40e", got[i]), rel));
printf("%d of %d observables off the 30-digit Maxima values by more than 1e-29 relative\n", bad, #got);
}
