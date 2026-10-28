/* Second engine for derivations/sdtm_no_cooling.mac: the frozen-N step of SDTM-1's heating and
   nitric-oxide cooling state at 60 digits, against the reference values Maxima prints, and the
   steady state H/(1 + H/Hs) reached by iterating the step. Prints `PASS` lines; a failed check
   prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_no_cooling.gp </dev/null                           */

default(realprecision, 60);
/* tc = 6 h, tn = 20 h, Hs = 300 ap, h = 3 h. */
holdstep(D, N, H) =
{
  my(nb = H + (N - H)*20/3*(1 - exp(-3/20)), l = (1 + nb/300)/6, ds = H/(6*l));
  [ds + (D - ds)*exp(-l*3), H + (N - H)*exp(-3/20)];
}
s = [0, 0];
ref = [0, 31.34059057975231056847707814, 95.4768583724596384638250588983, 73.4330192797666520349895091237];
drive = [0, 80, 200, 50];
for (k = 1, 4, s = holdstep(s[1], s[2], drive[k]); \
  print("step H = ", drive[k], if (abs(s[1] - ref[k]) < 10^-26, ": PASS", ": FAIL"), " D = ", Strprintf("%.30g", s[1])));
s = [0, 0];
for (k = 1, 2000, s = holdstep(s[1], s[2], 50));
print(if (abs(s[1] - 50/(1 + 50/300)) < 10^-50, "steady state: PASS", "steady state: FAIL"), " D = ", Strprintf("%.30g", s[1]));
