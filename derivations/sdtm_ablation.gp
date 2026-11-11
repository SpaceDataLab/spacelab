/* Second engine for derivations/sdtm_ablation.mac: each rung's limit approached numerically at 60
   digits, at a sample state. A limit at infinity or zero is evaluated near it. The two-constant
   kernel's fast weight vanishes only linearly in `to`, so at `to` = 1e-5 its distance from the slow
   filter is checked against its exact first-order term, (slow filter - index) to/(tr - to), with the
   fast filter's own e^(-3e5) left over. Prints `PASS` lines; a failed check prints `FAIL`.

       timeout 300 gp -q -f derivations/sdtm_ablation.gp </dev/null                          */

default(realprecision, 60);
check(name, ok) = print(name, if (ok, ": PASS", ": FAIL"));
first(y, u, k, h) = u + (y - u)*exp(-h/k);
nmean(n, u, tn, h) = u + (n - u)*tn/h*(1 - exp(-h/tn));
dstep(y, n, u, hs, th, tn, h) = {
  my(m = nmean(n, u, tn, h), rate = (1 + m/hs)/th, target = u/(th*rate));
  target + (y - target)*exp(-rate*h);
}
h = 3; th = 17.4; tn = 32.6; tr = 15.0; y = 31; n = 120; u = 300; yo = 260;

check("solar wind, share 1e-40", abs((1 - 10^-40)*37 + 10^-40*120 - 37) < 10^-37);
check("gain, kappa 1e-40", vecmax(vector(4, j, abs((75*j/150)^(10^-40) - 1))) < 10^-38);
check("flux saturation, D 1e30", abs(10^30*tanh(0.8/10^30) - 0.8) < 10^-58);
check("cooling, H_s 1e50", abs(dstep(y, n, u, 10^50, th, tn, h) - first(y, u, th, h)) < 10^-45);
check("first-order, tau 1e-2", abs(first(y, u, 10^-2, h) - u) < 10^-120);
to = 10^-5;
slow = first(y, u, tr, h);
kernel = (tr*slow - to*first(yo, u, to, h))/(tr - to);
check("kernel, step at to 1e-5", abs(kernel - slow - (slow - u)*to/(tr - to)) < 10^-50);
t = 5;
impulse = (exp(-t/tr) - exp(-t/to))/(tr - to);
check("kernel, impulse response at to 1e-5", \
  abs(impulse - exp(-t/tr)/tr - exp(-t/tr)*to/(tr*(tr - to))) < 10^-50);
meet = (th*first(y, u, th, h) - to*first(yo, u, to, h))/(th - to);
check("cooling and kernel meet at th = tr", \
  abs(dstep(y, n, u, 10^50, th, tn, h) - first(y, u, th, h)) < 10^-45 \
  && abs(meet - first(y, u, th, h) - (first(y, u, th, h) - u)*to/(th - to)) < 10^-50);
check("the log storm regressor at a state 1e-40", abs(log(1 + 10^-40/15)) < 10^-41);
