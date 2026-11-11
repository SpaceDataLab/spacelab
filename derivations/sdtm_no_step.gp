/* Second engine for derivations/sdtm_no_step.mac: the exact D after one 3 h interval of SDTM-1's heating
   with nitric-oxide cooling, by intnum at 60 digits, and the frozen-N step taken 1, 2 and 4 times, at
   the shipped constants and the gate's four states, against the 40-digit values Maxima prints; and the
   step's error falling as 1/k^2 once k is large enough for the h^3 term to lead (4, 8, 16). Prints
   `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_no_step.gp </dev/null                          */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
tc = 6295024178295275/(3600*10^11);
tn = 11864859040774021/(3600*10^11);
g = 1/(tc*2477339239626375/10^14);
h = 3;
A(x, n0, H) = x/tc + g*(H*x + (n0 - H)*tn*(1 - exp(-x/tn)));
exact(d0, n0, H) = exp(-A(h, n0, H))*(d0 + H/tc*intnum(x = 0, h, exp(A(x, n0, H))));
frozen(d0, n0, H, k) = {
  my(d = d0, n = n0, s = h/k, nb, l, ds);
  for (i = 1, k,
    nb = H + (n - H)*tn/s*(1 - exp(-s/tn));
    l = 1/tc + g*nb;
    ds = (H/tc)/l;
    d = ds + (d - ds)*exp(-l*s);
    n = H + (n - H)*exp(-s/tn));
  d;
}
states = [[7/(1 + 7*g*tc), 7, 300], [100, 50, 300], [200, 250, 7], [25, 25, 80]];
maxima = [ \
  [4.762337486992401450533057980401915786359e1, 4.826744663563632020381657310165554284854e1, \
   4.778366668363259473193813576693850872342e1, 4.766340293714888448323564590456697131921e1], \
  [9.351877850432057488886992206688563205822e1, 9.399891782236027379963198655501548586989e1, \
   9.363857968322325884265743242320556331544e1, 9.354871476828520861079620864448827165565e1], \
  [3.268600295901665138825566032361688976479e1, 3.267959654264993852352258976931277643558e1, \
   3.268434417928730758137188963621558291568e1, 3.268558450894311245253326329870819789305e1], \
  [2.890174578042530797076779496998899109804e1, 2.893341104342852857882427575800337455589e1, \
   2.890965627028522526860431190234301984433e1, 2.890372304109069530243170720948717997562e1]];
{
  for (j = 1, 4,
    my(s = states[j], m = maxima[j], ours);
    ours = [exact(s[1], s[2], s[3]), frozen(s[1], s[2], s[3], 1), frozen(s[1], s[2], s[3], 2),
            frozen(s[1], s[2], s[3], 4)];
    check(Str("state ", j, ": exact and frozen x1, x2, x4 against Maxima to 1e-36"),
      vecmax(vector(4, i, abs(ours[i] - m[i])/abs(m[i]))) < 10^-36);
    my(e = vector(3, i, frozen(s[1], s[2], s[3], 2^(i + 1)) - ours[1]));
    check(Str("state ", j, ": the frozen error falls as 1/k^2, k = 4, 8, 16"),
      abs(e[1]/e[2] - 4) < 0.02 && abs(e[2]/e[3] - 4) < 0.02));
}
