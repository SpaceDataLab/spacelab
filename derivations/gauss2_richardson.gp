/* Second engine for derivations/gauss2_richardson.mac: the doubling two-point Gauss rule with
   Richardson extrapolation of the NeQuick 2 slant-TEC integration, at 60 digits on [0, 1], for
   the interval counts the code actually reaches (8 to 1024), with g = h/sqrt(3) and with the
   printed 0.5773502691896 h.

       timeout 300 gp -q -f derivations/gauss2_richardson.gp </dev/null                         */

default(realprecision, 60);

{gn(k, n, c) = my(h = 1 / n, y = (h - c * h) / 2);
  h / 2 * sum(i = 0, n - 1, (y + i * h)^k + (y + i * h + c * h)^k)}
rich(k, n, c) = gn(k, 2 * n, c) + (gn(k, 2 * n, c) - gn(k, n, c)) / 15;

c0 = 1 / sqrt(3);
cp = 5773502691896 / 10^13;
printf("printed constant - 1/sqrt(3): %.3e (13-decimal rounding bound 5e-14)\n", cp - c0);
ns = [8, 16, 32, 64, 128, 256, 512];
{printf("Gauss G_n off on x^0..x^3 by more than 1e-55, n = 8..512: %d of %d\n",
  sum(j = 1, #ns, sum(k = 0, 3, abs(gn(k, ns[j], c0) - 1 / (k + 1)) > 1e-55)), 4 * #ns);}
{printf("Richardson off on x^0..x^5 by more than 1e-55, n = 8..512: %d of %d\n",
  sum(j = 1, #ns, sum(k = 0, 5, abs(rich(k, ns[j], c0) - 1 / (k + 1)) > 1e-55)), 6 * #ns);}
printf("Richardson departure on x^6 at n = 8: %.3e (the rule's degree is 5)\n", rich(6, 8, c0) - 1 / 7);
{printf("largest Richardson departure on x^0..x^5 with the printed constant, n = 8..512: %.3e\n",
  vecmax(concat(vector(#ns, j, vector(6, k, abs(rich(k - 1, ns[j], cp) - 1 / k))))));}
